"""Rotamer enumeration, and the fixed "template" the MILP scores against.

Rotamers come from PyMOL's mutagenesis wizard, which already ships a rotamer
library and builds ideal side-chain geometry. ``scripts/harvest_rotamers.py``
runs inside a PyMOL-capable interpreter and returns coordinates as JSON; this
module finds that interpreter, drives it, and turns the JSON into
:class:`ipro_agent.milp.Rotamer` objects.

The interpreter dance exists because PyMOL commonly lives in its own conda
environment (a different Python from the one running ipro-agent), so importing
it is not an option.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .config import DesignPosition
from .energy.base import AtomSet
from .milp import Rotamer
from .structure import Structure

log = logging.getLogger(__name__)

HARVEST_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "harvest_rotamers.py"

BACKBONE = {"N", "CA", "C", "O", "OXT"}


class RotamerError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Finding a PyMOL interpreter
# ---------------------------------------------------------------------------


def find_pymol_python() -> str | None:
    """Return an interpreter that can ``import pymol``, or None.

    Order: this interpreter, ``$IPRO_PYMOL_PYTHON``, then conda environments.
    The environment variable exists so a user can pin the choice rather than
    rely on discovery.
    """
    import sys

    def works(py: str) -> bool:
        try:
            p = subprocess.run(
                [py, "-c", "import pymol"], capture_output=True, timeout=60,
                env={**os.environ, "PYTHONNOUSERSITE": "1"},
            )
            return p.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    if works(sys.executable):
        return sys.executable
    pinned = os.environ.get("IPRO_PYMOL_PYTHON")
    if pinned:
        if works(pinned):
            return pinned
        log.warning("IPRO_PYMOL_PYTHON=%s cannot import pymol", pinned)

    roots = [
        Path(os.environ.get("CONDA_PREFIX", "/nonexistent")).parent / "envs",
        Path.home() / "miniconda3" / "envs",
        Path.home() / "anaconda3" / "envs",
        Path("/opt/miniconda3/envs"),
        Path("/opt/anaconda3/envs"),
    ]
    for root in roots:
        if not root.is_dir():
            continue
        # Prefer an env whose name mentions pymol: much faster than probing all.
        envs = sorted(root.iterdir(), key=lambda p: (0 if "pymol" in p.name.lower() else 1, p.name))
        for env in envs:
            py = env / "bin" / "python"
            if py.exists() and works(str(py)):
                return str(py)
    exe = shutil.which("pymol")
    if exe:
        return exe  # a pymol launcher can run a script too
    return None


# ---------------------------------------------------------------------------
# Harvesting
# ---------------------------------------------------------------------------


def harvest(
    pdb: str | Path,
    positions: Sequence[DesignPosition],
    python: str | None = None,
    workdir: Path | None = None,
) -> tuple[dict[str, list[Rotamer]], dict[str, str], dict]:
    """Enumerate rotamers for every (position, allowed residue) pair.

    Returns ``(rotamers, native, errors)`` where ``rotamers`` maps a position
    label to a flat list of :class:`Rotamer` across all allowed identities,
    ``native`` maps a position label to its residue name in the input
    structure, and ``errors`` records residue types PyMOL refused.
    """
    py = python or find_pymol_python()
    if py is None:
        raise RotamerError(
            "no PyMOL available. Install PyMOL (conda install -c conda-forge pymol-open-source) "
            "or set IPRO_PYMOL_PYTHON to an interpreter that can import pymol. "
            "Run scripts/doctor.py to check."
        )
    if not HARVEST_SCRIPT.exists():
        raise RotamerError(f"harvest script missing: {HARVEST_SCRIPT}")

    job = {
        "pdb": str(Path(pdb).resolve()),
        "jobs": [
            {
                "chain": p.chain,
                "number": p.number,
                "icode": p.insertion_code,
                "allowed": list(p.allowed),
            }
            for p in positions
        ],
    }

    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="ipro_rot_"))
    tmp.mkdir(parents=True, exist_ok=True)
    job_path, out_path = tmp / "rotamer_jobs.json", tmp / "rotamers.json"
    job_path.write_text(json.dumps(job))

    # -I isolates the interpreter: the harvest script must not pick up modules
    # from the current directory (the input PDB's directory is untrusted).
    cmd = [py, "-I", str(HARVEST_SCRIPT), str(job_path), str(out_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if not out_path.exists():
        raise RotamerError(
            f"rotamer harvest produced no output (rc={proc.returncode}).\n"
            f"command: {' '.join(cmd)}\n"
            f"stderr tail: {(proc.stderr or '')[-800:]}"
        )
    data = json.loads(out_path.read_text())

    rotamers: dict[str, list[Rotamer]] = {}
    for pos_label, by_aa in data.get("rotamers", {}).items():
        flat: list[Rotamer] = []
        for aa, rots in by_aa.items():
            for rec in rots:
                names = [a[0] for a in rec["atoms"]]
                coords = np.array([a[1] for a in rec["atoms"]], dtype=float) if rec["atoms"] else np.zeros((0, 3))
                flat.append(
                    Rotamer(
                        position=pos_label,
                        aa=aa,
                        state=int(rec["state"]),
                        atoms=AtomSet(
                            names=names,
                            coords=coords,
                            resnames=[aa] * len(names),
                            reskeys=[("ROT", pos_label, "")] * len(names),
                        ),
                        strain=rec.get("strain"),
                    )
                )
        # Deterministic order: identity then state, so the LP is reproducible.
        flat.sort(key=lambda r: (r.aa, r.state))
        rotamers[pos_label] = flat

    errors = data.get("errors", {})
    if errors:
        log.warning("PyMOL could not build some rotamers: %s", errors)
    return rotamers, data.get("native", {}), errors


# ---------------------------------------------------------------------------
# The template
# ---------------------------------------------------------------------------


def build_template(
    structures: Iterable[Structure],
    design_positions: Sequence[str],
    include_hetero: bool = True,
) -> AtomSet:
    """Everything held fixed while the MILP chooses rotamers.

    That is: all backbone atoms, the complete side chains of residues that are
    *not* design positions, and (optionally) ligands/cofactors from any of the
    supplied structures -- which is how target and anti-target molecules enter
    the self-energy term.

    Design positions contribute only their backbone: their side chains are
    exactly what is being chosen, so including them would double count.
    """
    design = set(design_positions)
    names: list[str] = []
    coords: list[np.ndarray] = []
    resnames: list[str] = []
    reskeys: list[tuple] = []

    for st in structures:
        for res in st.residues():
            ref = st.ref(res)
            is_design = ref.label in design
            for at in res:
                if at.element == "H":
                    continue
                if is_design and at.get_name().strip() not in BACKBONE:
                    continue
                names.append(at.get_name().strip())
                coords.append(at.coord)
                resnames.append(res.get_resname().strip())
                reskeys.append(ref.key)
        if include_hetero:
            for res in st.hetero_residues():
                ref = st.ref(res)
                for at in res:
                    if at.element == "H":
                        continue
                    names.append(at.get_name().strip())
                    coords.append(at.coord)
                    resnames.append(res.get_resname().strip())
                    reskeys.append(ref.key)

    return AtomSet(
        names=names,
        coords=np.array(coords) if coords else np.zeros((0, 3)),
        resnames=resnames,
        reskeys=reskeys,
    )
