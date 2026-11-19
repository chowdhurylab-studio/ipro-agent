"""Rotamer enumeration, and the fixed "template" the MILP scores against.

Rotamers come from PyMOL's mutagenesis wizard, which already ships a rotamer
library and builds ideal side-chain geometry. ``ipro_agent/pymol_helpers/harvest_rotamers.py``
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
from .structure import Structure, is_hydrogen

log = logging.getLogger(__name__)

#: Run by PyMOL's interpreter, not imported. Inside the package so that it
#: ships in the wheel -- from the repo's top-level ``scripts/`` it resolved to a
#: ``site-packages/scripts/`` that does not exist, so a pip-installed copy could
#: validate a spec and then died at the first harvest.
HARVEST_SCRIPT = Path(__file__).resolve().parent / "pymol_helpers" / "harvest_rotamers.py"

#: Backbone atom names. ``OT1``/``OT2``/``OT`` are the CHARMM/AMBER spellings of
#: the C-terminal carboxylate that ``OXT`` covers in PDB files: without them, a
#: C-terminal design position in a file using those names had its carbonyl O and
#: both terminal oxygens treated as side chain and dropped from the template.
BACKBONE = {"N", "CA", "C", "O", "OXT", "OT1", "OT2", "OT"}


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
        """Can this really run our harvest script as a Python interpreter?

        Exit status alone is not the question. A **pymol launcher** given
        ``-c "import pymol"`` treats the string as a *file to open*, prints
        "Error: unsupported file type: import pymol", and still exits 0 -- so
        it was accepted as the harvest interpreter, and the run then started a
        windowless PyMOL session that waited for input until the whole job
        timed out, with no rotamers and no error. Require a sentinel on
        stdout: only a Python interpreter with pymol importable can print it.
        """
        try:
            p = subprocess.run(
                [py, "-I", "-c", "import pymol; print('IPRO_PYMOL_OK')"],
                capture_output=True, timeout=120, text=True,
                env={**os.environ, "PYTHONNOUSERSITE": "1"},
            )
            # Exact match on the LAST line, not a substring anywhere: the
            # launcher echoes its argument back ("Error: unsupported file
            # type: import pymol; print('IPRO_PYMOL_OK')"), so a substring
            # test for the sentinel passes on the very program it is meant to
            # reject. Only a real interpreter prints the sentinel alone.
            lines = [ln.strip() for ln in (p.stdout or "").splitlines() if ln.strip()]
            return p.returncode == 0 and bool(lines) and lines[-1] == "IPRO_PYMOL_OK"
        except (OSError, subprocess.SubprocessError):
            return False

    if works(sys.executable):
        return sys.executable
    pinned = os.environ.get("IPRO_PYMOL_PYTHON")
    if pinned:
        if works(pinned):
            return pinned
        log.warning("IPRO_PYMOL_PYTHON=%s cannot import pymol", pinned)

    # One shared list with the doctor's, in ipro_agent.doctor: two separate
    # lists had drifted apart, so the doctor searched roots the harvester did
    # not and reported PyMOL as resolved for a run that could not find it.
    from .doctor import conda_env_roots, env_interpreter

    for root in conda_env_roots():
        # Prefer an env whose name mentions pymol: much faster than probing all.
        envs = sorted(root.iterdir(),
                      key=lambda p: (0 if "pymol" in p.name.lower() else 1, p.name))
        for env in envs:
            py = env_interpreter(env)
            if py is not None and works(str(py)):
                return str(py)
    # A `pymol` on PATH is a launcher, not an interpreter: it cannot run the
    # harvest script the way this package invokes it. Probed, not assumed.
    exe = shutil.which("pymol")
    if exe and works(exe):
        return exe
    if exe:
        log.warning(
            "found a pymol launcher at %s, but it cannot run a Python script "
            "(`-c` makes it try to OPEN the code as a file). Point "
            "IPRO_PYMOL_PYTHON at the interpreter inside that installation "
            "instead -- usually <prefix>/bin/python.", exe,
        )
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
    job_path.write_text(json.dumps(job), encoding="utf-8")
    # The loop passes the same workdir every iteration, so a previous
    # iteration's rotamers.json is sitting right there. Reading it when this
    # harvest fails returned the rotamers of the PREVIOUS backbone: measured
    # with the backbone translated 10 A and PyMOL exiting 1, harvest returned
    # side chains 10.0 A from where they belonged, reported no errors, and a
    # hard-clash conformer (+453.111 against the real template) came back as
    # the best-scoring one at -1.036.
    out_path.unlink(missing_ok=True)

    # -I isolates the interpreter: the harvest script must not pick up modules
    # from the current directory (the input PDB's directory is untrusted).
    cmd = [py, "-I", str(HARVEST_SCRIPT), str(job_path), str(out_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0 or not out_path.exists():
        raise RotamerError(
            f"rotamer harvest failed (rc={proc.returncode}).\n"
            f"command: {' '.join(cmd)}\n"
            f"stderr tail: {(proc.stderr or '')[-800:]}"
        )
    data = json.loads(out_path.read_text(encoding="utf-8"))
    described = str(data.get("pdb", ""))
    if described and described != job["pdb"]:
        raise RotamerError(
            f"rotamer harvest describes {described!r} but was asked for "
            f"{job['pdb']!r}; refusing to use rotamers built on another structure"
        )

    # A rotamer's atoms must carry the residue key of the position they sit at,
    # NOT a synthetic key. The design position's own backbone (N, CA, C, O) is
    # part of the fixed template, so if the keys do not match, the energy
    # backend's same-residue exclusion never fires and every rotamer is scored
    # against its own backbone as though it were a different residue. That adds
    # a spurious +30 to +50 kcal/mol per rotamer, scaling with side-chain size,
    # which biases the whole objective toward the smallest residue and prunes
    # large side chains at the default clash threshold.
    key_of = {p.label: (p.chain, p.number, p.insertion_code) for p in positions}

    rotamers: dict[str, list[Rotamer]] = {}
    for pos_label, by_aa in data.get("rotamers", {}).items():
        reskey = key_of.get(pos_label)
        if reskey is None:
            raise RotamerError(
                f"harvest returned position {pos_label!r} which was not requested; "
                "cannot determine its residue key"
            )
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
                            reskeys=[reskey] * len(names),
                        ),
                        strain=rec.get("strain"),
                    )
                )
        # Deterministic order: identity then state, so the LP is reproducible.
        flat.sort(key=lambda r: (r.aa, r.state))
        rotamers[pos_label] = flat

    # A requested position that came back missing entirely is a silently
    # smaller design problem. The loop's own check scans only the keys that
    # are present (`[p for p, v in rots.items() if not v]`), so a position
    # PyMOL could not locate -- an incomplete crystal structure missing that
    # residue's CA is enough -- vanished without a word: measured, 2 positions
    # requested, 1 designed, nothing raised.
    missing = [p.label for p in positions if p.label not in rotamers]
    if missing:
        raise RotamerError(
            f"rotamer harvest returned nothing for requested position(s) {missing}. "
            f"It returned {sorted(rotamers)}. Check that those residues exist and "
            f"are complete in the structure; PyMOL reported: "
            f"{json.dumps(data.get('errors', {}))[:400]}"
        )

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

    The **first** structure is the design molecule; every later one has its
    residue keys prefixed with its index. A residue key is what the energy
    backend's same-residue exclusion compares, and a partner reusing chain id
    ``A`` -- the default for most single-chain PDB files -- collided with the
    design molecule's own residues: measured, 22 distinct residues collapsed to
    11, the partner's residue 115 lost its side chain to the design filter, and
    a ligand numbered like a design position became invisible to the rotamer it
    sat on top of (+84.699 kcal/mol scored as -6.282). Rotamer atoms carry the
    bare ``(chain, number, icode)`` key of their design position, so the design
    molecule must keep the unprefixed form.
    """
    design = set(design_positions)
    names: list[str] = []
    coords: list[np.ndarray] = []
    resnames: list[str] = []
    reskeys: list[tuple] = []
    elements: list[str] = []

    def take(res, ref, mol: int, drop_sidechain: bool) -> None:
        for at in res:
            if is_hydrogen(at):
                continue
            nm = at.get_name().strip()
            if drop_sidechain and nm not in BACKBONE:
                continue
            names.append(nm)
            coords.append(at.coord)
            resnames.append(res.get_resname().strip())
            reskeys.append(ref.key if mol == 0 else (mol, *ref.key))
            elements.append((at.element or "").strip().upper())

    for mol, st in enumerate(structures):
        for res in st.residues():
            ref = st.ref(res)
            take(res, ref, mol, ref.label in design)
        if include_hetero:
            for res in st.hetero_residues():
                ref = st.ref(res)
                # A design position can be on a HETATM-flagged residue -- MSE
                # and every other modified residue is one. Without this test
                # such a position kept its full native side chain in the fixed
                # template, so its own rotamers were fine (matching key, so the
                # exclusion fired) but its neighbours were scored against a
                # side chain that was supposed to have been removed: measured on
                # 2omf_A, TRP at A262 moved +78 to +284 kcal/mol depending on
                # the conformer and the chosen rotamer flipped from state 6 to
                # state 1.
                take(res, ref, mol, ref.label in design)

    return AtomSet(
        names=names,
        coords=np.array(coords) if coords else np.zeros((0, 3)),
        resnames=resnames,
        reskeys=reskeys,
        elements=elements if all(elements) else [],
    )
