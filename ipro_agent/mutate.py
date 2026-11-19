"""Committing a chosen design to coordinates.

This is the other half of the rotamer harvest. :mod:`ipro_agent.rotamers`
enumerates the side-chain conformations PyMOL's mutagenesis wizard offers at
each design position, :mod:`ipro_agent.milp` picks one per position, and this
module writes a PDB in which those choices are the actual atoms.

The choice is only meaningful because the *same* wizard applies it. A rotamer
arrives here as ``("TRP", 2)``: an amino acid and an integer index into PyMOL's
rotamer set for that residue, in the order PyMOL enumerates it. That integer
has no meaning outside that enumeration, so rebuilding the side chain with any
other tool -- even one using a published rotamer library -- would put a
different conformation into the structure than the MILP scored, and the
energies reported for the design would not describe the coordinates written.
``ipro_agent/pymol_helpers/apply_design.py`` is the PyMOL-side half of the contract.

The interpreter dance is the same as the harvest's: PyMOL usually lives in its
own conda environment, so it is driven as a subprocess rather than imported
(see :func:`ipro_agent.rotamers.find_pymol_python`).

What this module does *not* do: relax or minimize anything. A freshly applied
rotamer is ideal geometry dropped into a fixed backbone, so it can clash with
its surroundings -- that is what the MILP's pairwise terms are for, and what
the relaxation step (:meth:`ipro_agent.energy.base.EnergyBackend.relax`) exists
to repair afterwards.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .config import AA3, SpecError, DesignPosition
from .rotamers import find_pymol_python

log = logging.getLogger(__name__)

#: Run by PyMOL's interpreter, not imported; inside the package so it ships in
#: the wheel (see ipro_agent/pymol_helpers/__init__.py).
APPLY_SCRIPT = Path(__file__).resolve().parent / "pymol_helpers" / "apply_design.py"


class MutateError(RuntimeError):
    pass


@dataclass
class MutationResult:
    """What was written, and what was not.

    ``applied`` and ``errors`` are disjoint and together cover every requested
    position: a position is either in the structure as asked or accounted for.
    A result with non-empty ``errors`` is **not** the design that was requested
    -- ``pdb_out`` was still written (so it can be inspected) but it carries
    the native residue at every failed position, which is a different sequence
    with a different energy. Callers in the iteration loop should treat that as
    a failed iteration rather than scoring it.
    """

    pdb_out: Path
    #: Position label -> (three-letter residue, 1-based PyMOL rotamer state).
    applied: dict[str, tuple[str, int]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    #: One-letter sequence of the design chain after mutation. When the design
    #: spans several chains, their sequences are concatenated in chain-ID order.
    sequence: str = ""


def apply_design(
    pdb_in: str | Path,
    pdb_out: str | Path,
    choices: dict[str, tuple[str, int]],
    python: str | None = None,
    workdir: str | Path | None = None,
) -> MutationResult:
    """Apply one chosen rotamer per position and write the structure.

    ``choices`` maps a position label (``"A115"``, or ``"A115B"`` with an
    insertion code) to ``(three-letter amino acid, state)``, where ``state`` is
    the 1-based index returned by the harvest for that identity at that
    position. The states must come from a harvest of *this* structure: PyMOL
    builds rotamers against the local backbone, so a state index harvested from
    a different backbone conformation names a different conformation here.

    Positions are applied in sorted label order, one at a time, each against
    the structure the previous one produced. The order is fixed rather than
    dictionary order so that re-running a design writes byte-comparable
    coordinates.
    """
    pdb_in = Path(pdb_in).expanduser()
    pdb_out = Path(pdb_out).expanduser()
    if not pdb_in.exists():
        raise MutateError(f"no such structure file: {pdb_in}")
    if not choices:
        raise MutateError(
            "no choices given: apply_design writes a design, not a copy. "
            "Pass {label: (residue, state)} from the MILP solution, or use "
            "mutate_sequence() for an identity-only mutation list."
        )
    if not APPLY_SCRIPT.exists():
        raise MutateError(f"mutagenesis script missing: {APPLY_SCRIPT}")

    jobs = []
    for label in sorted(choices):
        aa, state = choices[label]
        aa = str(aa).upper()
        if aa not in AA3:
            raise MutateError(
                f"position {label}: {aa!r} is not one of the 20 standard amino "
                f"acids. PyMOL's mutagenesis wizard builds only these; use a "
                f"three-letter code such as 'TRP'."
            )
        state = int(state)
        if state < 1:
            raise MutateError(
                f"position {label}: rotamer state {state} is invalid. PyMOL "
                f"states are 1-based; state 1 is its first {aa} rotamer."
            )
        try:
            pos = DesignPosition.parse(label, allowed=[aa])
        except SpecError as exc:
            raise MutateError(f"cannot use {label!r} as a position label: {exc}") from exc
        if pos.label != label:
            # The label is the only carrier of (chain, number, icode) across
            # the PyMOL boundary, so it has to survive the round trip. If it
            # does not, the mutation would be applied to a different residue
            # than the one the MILP chose -- and that residue may well exist,
            # so nothing downstream would notice.
            raise MutateError(
                f"position label {label!r} does not survive re-parsing "
                f"(it reads back as {pos.label!r}, chain {pos.chain!r} "
                f"residue {pos.number}{pos.insertion_code}). Refusing to "
                f"mutate a residue that may not be the one requested."
            )
        jobs.append(
            {
                "chain": pos.chain,
                "number": pos.number,
                "icode": pos.insertion_code,
                "aa": aa,
                "state": state,
            }
        )

    py = python or find_pymol_python()
    if py is None:
        raise MutateError(
            "no PyMOL available. Install PyMOL (conda install -c conda-forge "
            "pymol-open-source) or set IPRO_PYMOL_PYTHON to an interpreter that "
            "can import pymol. Run scripts/doctor.py to check."
        )

    pdb_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="ipro_mut_"))
    tmp.mkdir(parents=True, exist_ok=True)
    job_path, res_path = tmp / "mutate_jobs.json", tmp / "mutate_result.json"
    job_path.write_text(
        json.dumps(
            {"pdb": str(pdb_in.resolve()), "out": str(pdb_out.resolve()), "jobs": jobs}
        )
    , encoding="utf-8")

    # -I isolates the interpreter: the script must not pick up modules from the
    # current directory (the input PDB's directory is untrusted).
    cmd = [py, "-I", str(APPLY_SCRIPT), str(job_path), str(res_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if not res_path.exists():
        raise MutateError(
            f"mutagenesis produced no output (rc={proc.returncode}).\n"
            f"command: {' '.join(cmd)}\n"
            f"stderr tail: {(proc.stderr or '')[-800:]}"
        )
    data = json.loads(res_path.read_text(encoding="utf-8"))

    applied = {k: (str(v[0]), int(v[1])) for k, v in data.get("applied", {}).items()}
    errors = dict(data.get("errors", {}))
    sequences = data.get("sequences", {})
    native = data.get("native", {})

    if not applied:
        raise MutateError(
            f"no mutation could be applied to {pdb_in.name}; "
            f"{pdb_out} would be a copy of the input, not a design.\n"
            + "\n".join(f"  {k}: {v}" for k, v in sorted(errors.items()))
        )
    if errors:
        log.warning(
            "%s: %d of %d positions were not mutated: %s",
            pdb_out.name, len(errors), len(jobs),
            "; ".join(f"{k}: {v}" for k, v in sorted(errors.items())),
        )
    if not pdb_out.exists():
        raise MutateError(f"PyMOL reported success but {pdb_out} was not written")

    log.info(
        "%s: %s",
        pdb_out.name,
        ", ".join(
            f"{native.get(k, '')}{k}{v[0]}" for k, v in sorted(applied.items())
        ),
    )
    return MutationResult(
        pdb_out=pdb_out,
        applied=applied,
        errors=errors,
        sequence="".join(sequences[c] for c in sorted(sequences)),
    )


def mutate_sequence(
    pdb_in: str | Path,
    pdb_out: str | Path,
    mutations: dict[str, str],
    **kw,
) -> MutationResult:
    """Apply a mutation list given as identities only, e.g. ``{"A115": "TRP"}``.

    For applying a published mutant without running the MILP. Every position
    takes rotamer state 1, which is **PyMOL's first rotamer for that residue --
    the most probable one in its library -- and not necessarily the least
    strained one in this structure**. The two differ often enough to matter: at
    A16 of ``tests/data/1crn.pdb`` the wizard offers seven TRP rotamers with
    bump scores 65.2, 37.9, 63.7, 77.1, 89.1, 92.6, 60.7, so state 1 is the
    fourth-least strained of the seven.

    So this is a convenience, not a packing step. To choose the state rather
    than take the first, harvest the position with
    :func:`ipro_agent.rotamers.harvest` (which returns each state's strain) and
    call :func:`apply_design`.
    """
    return apply_design(
        pdb_in, pdb_out, {k: (v, 1) for k, v in mutations.items()}, **kw
    )
