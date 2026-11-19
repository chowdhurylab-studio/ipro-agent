"""Rigid-body redocking of the target molecule, via AutoDock Vina.

Why there is a docking step at all
----------------------------------
IPRO does not hold the binding partner still. Every third iteration it
re-docks the target molecule as a rigid body, because a redesigned interface
changes where the target wants to sit, and scoring the new side chains against
the old pose would reward designs for a complex that no longer exists. Stock
IPRO does this with an in-house Monte-Carlo routine: 500 sub-moves per docking
run, each a Gaussian perturbation with standard deviation 0.2 A in translation
and 2.0 deg in rotation, accepted by a Metropolis test whose temperature is
annealed linearly from 3640 K to 2190 K over the run (``defaultDockingFrequency
= 3``, ``defaultDockingIterations = 500``, ``defaultSDMove = 0.2``,
``defaultSDRotate = 2.0``, ``defaultDockTempStart/End`` in
``refs/source/IPRO_Suite_mfa5147/repo/modules/STANDARDS.py``, assembled in
``DOCKING_FUNCTIONS.make_docking_information``; mirrored as the
``redock_*`` fields of :class:`ipro_agent.config.LoopSpec`).

Why Vina instead
----------------
That in-house routine is not available: it runs inside the IPRO Suite's CHARMM
harness, which is not part of this reimplementation. AutoDock Vina is used in
its place -- a deliberate substitution of an open, maintained, installable
program for an unavailable one, and it is not a reimplementation of IPRO's
docking.

The two should not be expected to agree, and the difference is not only
numerical:

* **Different search.** IPRO perturbs the *current* pose by small amounts and
  anneals -- a local refinement that stays near the pose it was given. Vina
  runs its own global search (Monte-Carlo plus BFGS local optimization) over
  the whole box, so it can return a pose unrelated to the input one. Keeping
  the box tight around the known site is what makes it behave like a local
  redock.
* **Different energy.** Vina's affinity is its own empirical scoring function
  in kcal/mol, fitted to binding data. It is not the force-field interaction
  energy IPRO's objective is built from, and the two are not interconvertible.
  So a Vina number must not be added into the IPRO objective: use Vina to
  *produce the pose*, then re-score that pose with the configured energy
  backend (:mod:`ipro_agent.energy.base`), which is the only number the
  accept/reject test may compare across iterations.
* **Not deterministic unless told to be.** Vina seeds from the clock by
  default. The IPRO loop reruns and compares iterations, so ``seed`` is
  plumbed through and should always be set for a run whose result has to be
  reproducible.

Vina is a separate executable, so this module is a subprocess wrapper. It is
written to be useful on a machine without Vina installed: ``dry_run`` builds
and returns the exact command line without executing it, which is also how the
command construction is tested.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

#: Executable names to look for, matching ``scripts/doctor.py``.
VINA_NAMES = ("vina", "vina_1.2.5", "vina.exe")

#: Vina runs can be minutes long at high exhaustiveness; this bounds a hung
#: subprocess rather than a slow one.
DOCK_TIMEOUT_S = 3600

_INSTALL_HELP = (
    "AutoDock Vina was not found. The docking step needs the `vina` executable "
    "(1.2 or newer).\n"
    "  install:  conda install -c conda-forge vina\n"
    "            or take a release binary from "
    "https://github.com/ccsb-scripps/AutoDock-Vina/releases\n"
    "  pin it:   export IPRO_VINA=/full/path/to/vina\n"
    "  check:    python3 scripts/doctor.py\n"
    "To run without it, set the docking backend to 'none' "
    "(Backends.dock) or redock: false on the target molecule. That is a "
    "reduced protocol: the target keeps its input pose for the whole run, so "
    "the design is optimized against a complex that never relaxes to it."
)


class DockError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# The search box
# ---------------------------------------------------------------------------


@dataclass
class DockBox:
    """The search space Vina is allowed to place the ligand in.

    Vina requires an explicit box; there is no default. 22.5 A per side is used
    here as a neutral starting cube -- it is 60 grid points at Vina's default
    0.375 A spacing -- but a box is a scientific choice, not a constant: it has
    to hold the ligand plus room to rotate, while a box much larger than the
    site turns a redock into a blind dock and makes the result depend on
    exhaustiveness. Prefer :meth:`around` on the atoms of the known site.
    """

    center: tuple[float, float, float]
    size: tuple[float, float, float] = (22.5, 22.5, 22.5)

    def __post_init__(self) -> None:
        # YAML and JSON give lists where the dataclass wants tuples.
        self.center = tuple(float(v) for v in self.center)
        self.size = tuple(float(v) for v in self.size)
        if len(self.center) != 3 or len(self.size) != 3:
            raise DockError(
                f"a docking box needs 3 centre and 3 size values; got "
                f"{len(self.center)} and {len(self.size)}"
            )
        if min(self.size) <= 0:
            raise DockError(f"docking box sides must be positive; got {self.size}")

    @classmethod
    def around(cls, coords, padding: float = 4.0) -> "DockBox":
        """A box enclosing ``coords`` (N,3) with ``padding`` A on every face.

        The centre is the midpoint of the bounding box rather than the centroid
        of the atoms, so that every supplied atom is inside the box even when
        the site is lopsided.

        ``padding`` is the only room the ligand gets to move, so a site defined
        by a handful of atoms gives a small box: widen the padding rather than
        trusting the default when the ligand is larger than the site atoms you
        passed.
        """
        pts = np.asarray(coords, dtype=float).reshape(-1, 3)
        if len(pts) == 0:
            raise DockError("cannot build a docking box around zero atoms")
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        return cls(
            center=tuple((lo + hi) / 2.0),
            size=tuple((hi - lo) + 2.0 * float(padding)),
        )

    def args(self) -> list[str]:
        """Vina's six box flags, in the order Vina documents them."""
        cx, cy, cz = self.center
        sx, sy, sz = self.size
        return [
            "--center_x", f"{cx:.3f}", "--center_y", f"{cy:.3f}", "--center_z", f"{cz:.3f}",
            "--size_x", f"{sx:.3f}", "--size_y", f"{sy:.3f}", "--size_z", f"{sz:.3f}",
        ]


@dataclass
class DockResult:
    """One ligand's docking outcome.

    ``poses`` and ``scores`` are parallel and sorted best (most negative)
    first, so ``poses[0]`` is the pose that ``best_score`` belongs to.

    ``best_score`` is NaN, with ``poses`` and ``scores`` empty, in the two
    cases where there is no pose to report: a ``dry_run``, and a ligand that
    failed inside :func:`dock_batch`. NaN rather than a large positive number
    on purpose -- it cannot be mistaken for a weak-but-real affinity, and it
    propagates into any arithmetic that forgets to check.
    """

    poses: list[Path] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    best_score: float = float("nan")
    command: list[str] = field(default_factory=list)
    stdout: str = ""


# ---------------------------------------------------------------------------
# Finding Vina
# ---------------------------------------------------------------------------


def find_vina() -> str | None:
    """Return a working ``vina`` executable, or None.

    Order: ``$IPRO_VINA``, then the names in :data:`VINA_NAMES` on PATH. The
    environment variable exists so a user can pin the choice rather than rely
    on discovery. Each candidate is run with ``--version`` so that a stale
    symlink or a wrapper script that cannot start is reported as missing here
    rather than as a docking failure later.
    """

    def works(exe: str) -> bool:
        try:
            p = subprocess.run(
                [exe, "--version"], capture_output=True, timeout=60,
                env={**os.environ, "PYTHONNOUSERSITE": "1"},
            )
            return p.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    pinned = os.environ.get("IPRO_VINA")
    if pinned:
        if works(pinned):
            return pinned
        log.warning("IPRO_VINA=%s cannot be run with --version", pinned)
    for name in VINA_NAMES:
        exe = shutil.which(name)
        if exe and works(exe):
            return exe
    return None


# ---------------------------------------------------------------------------
# Reading Vina's output
# ---------------------------------------------------------------------------


def scores_from_pdbqt(path: str | Path) -> list[float]:
    """Affinities (kcal/mol) from the ``REMARK VINA RESULT:`` lines, in file order.

    This is the preferred route. Vina writes one such line per pose from
    ``Vina::vina_remarks()``, as ``REMARK VINA RESULT:`` followed by the
    affinity and the lower- and upper-bound RMSDs from the best mode. The same
    numbers appear in the stdout table, but that table is printed only at
    ``--verbosity`` 1 or more and at a different precision, so the file is the
    more stable source.
    """
    out: list[float] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.startswith("REMARK VINA RESULT:"):
            continue
        fields = line[len("REMARK VINA RESULT:"):].split()
        if not fields:
            raise DockError(f"{path}:{n}: REMARK VINA RESULT line has no affinity")
        try:
            out.append(float(fields[0]))
        except ValueError as exc:
            raise DockError(
                f"{path}:{n}: cannot read an affinity from {fields[0]!r}"
            ) from exc
    return out


def scores_from_stdout(text: str) -> list[float]:
    """Affinities (kcal/mol) from Vina's result table on stdout.

    The table is::

        mode |   affinity | dist from best mode
             | (kcal/mol) | rmsd l.b.| rmsd u.b.
        -----+------------+----------+----------
           1       -7.1234     0.000      0.000

    A fallback for the rare case of a run whose output file cannot be read
    (``--score_only``, for instance, prints but writes nothing). Rows are
    recognised by shape -- an integer mode number followed by three floats --
    rather than by position, because the surrounding log text varies with
    verbosity and version.
    """
    out: list[float] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 4:
            continue
        try:
            mode = int(fields[0])
            values = [float(v) for v in fields[1:]]
        except ValueError:
            continue
        if mode != len(out) + 1:
            continue  # not a table row: modes are numbered 1, 2, 3, ...
        out.append(values[0])
    return out


def split_poses(pdbqt: str | Path, out_dir: str | Path, stem: str) -> list[Path]:
    """Write each ``MODEL`` block of a multi-pose PDBQT to its own file.

    Vina puts every binding mode in one output file as MODEL/ENDMDL records.
    The downstream steps want one structure per pose (to rebuild a complex and
    re-score it), so the file is split here rather than by shelling out to
    ``vina_split``, which is a separate executable that may not be installed
    alongside ``vina``. The MODEL/ENDMDL wrappers are dropped and every other
    record, REMARKs included, is kept verbatim.
    """
    pdbqt = Path(pdbqt)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = pdbqt.read_text(encoding="utf-8").splitlines(True)

    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if line.startswith("MODEL"):
            current = []
        elif line.startswith("ENDMDL"):
            if current is not None:
                blocks.append(current)
            current = None
        elif current is not None:
            current.append(line)
    if not blocks:
        # Single-pose output (--local_only, --score_only) has no MODEL records.
        blocks = [[ln for ln in lines if ln.strip()]]

    paths: list[Path] = []
    for i, block in enumerate(blocks, start=1):
        p = out_dir / f"{stem}_pose{i}.pdbqt"
        p.write_text("".join(block), encoding="utf-8")
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Docking
# ---------------------------------------------------------------------------


def dock(
    receptor: str | Path,
    ligand: str | Path,
    box: DockBox,
    out_dir: str | Path,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    seed: int | None = None,
    dry_run: bool = False,
    cpu: int | None = None,
) -> DockResult:
    """Dock one ligand into one receptor and return the poses, best first.

    Both inputs must already be PDBQT: Vina reads nothing else. See
    :func:`pdb_to_pdbqt` for a stopgap conversion and for why it is one.

    ``seed`` makes the run reproducible and should be set whenever the result
    is compared across iterations. Omitting it lets Vina seed itself, so two
    runs of the same design can disagree.

    ``dry_run`` returns the command that would be run, with no poses and a NaN
    ``best_score``, and touches nothing on disk -- not even to check that the
    inputs exist, so a command can be built before the inputs are prepared.
    It is how this function is exercised where Vina is not installed; the one
    difference from a real run is that the literal ``vina`` stands in for the
    executable path when none can be found.
    """
    if exhaustiveness < 1:
        raise DockError(f"exhaustiveness must be at least 1; got {exhaustiveness}")
    if num_modes < 1:
        raise DockError(f"num_modes must be at least 1; got {num_modes}")
    if cpu is not None and cpu < 1:
        raise DockError(f"cpu must be at least 1; got {cpu}")

    receptor, ligand = Path(receptor), Path(ligand)
    out_dir = Path(out_dir)
    stem = ligand.stem
    out_pdbqt = out_dir / f"{stem}_vina.pdbqt"

    exe = find_vina()
    if exe is None and not dry_run:
        raise DockError(_INSTALL_HELP)

    cmd = [
        exe or "vina",
        "--receptor", str(receptor),
        "--ligand", str(ligand),
        "--out", str(out_pdbqt),
        *box.args(),
        "--exhaustiveness", str(int(exhaustiveness)),
        "--num_modes", str(int(num_modes)),
    ]
    if seed is not None:
        cmd += ["--seed", str(int(seed))]
    if cpu is not None:
        cmd += ["--cpu", str(int(cpu))]

    if dry_run:
        return DockResult(command=cmd)

    for label, p in (("receptor", receptor), ("ligand", ligand)):
        if not p.exists():
            raise DockError(f"{label} file does not exist: {p}")
        if p.suffix.lower() != ".pdbqt":
            raise DockError(
                f"{label} {p.name} is not a .pdbqt file; Vina reads only PDBQT. "
                f"Convert it first (see pdb_to_pdbqt, or Meeko's "
                f"mk_prepare_ligand.py / mk_prepare_receptor.py)."
            )
    out_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=DOCK_TIMEOUT_S)
    if proc.returncode != 0:
        raise DockError(
            f"vina failed (rc={proc.returncode}).\n"
            f"command: {' '.join(cmd)}\n"
            f"stderr tail: {(proc.stderr or '')[-800:]}\n"
            f"stdout tail: {(proc.stdout or '')[-800:]}"
        )
    if not out_pdbqt.exists():
        raise DockError(
            f"vina returned 0 but wrote no poses to {out_pdbqt}.\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout tail: {(proc.stdout or '')[-800:]}"
        )

    scores = scores_from_pdbqt(out_pdbqt)
    if not scores:
        # Should not happen, but the table is the same numbers, so say what
        # happened and carry on rather than losing a completed run.
        scores = scores_from_stdout(proc.stdout or "")
        log.warning(
            "%s has no REMARK VINA RESULT lines; read %d affinity value(s) from "
            "vina's stdout table instead", out_pdbqt.name, len(scores),
        )
    if not scores:
        raise DockError(
            f"no affinities found in {out_pdbqt} or in vina's output.\n"
            f"command: {' '.join(cmd)}"
        )

    poses = split_poses(out_pdbqt, out_dir, stem)
    if len(poses) != len(scores):
        raise DockError(
            f"{out_pdbqt}: {len(poses)} pose(s) but {len(scores)} affinity "
            f"value(s); refusing to pair them up by guesswork"
        )

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    return DockResult(
        poses=[poses[i] for i in order],
        scores=[scores[i] for i in order],
        best_score=scores[order[0]],
        command=cmd,
        stdout=proc.stdout or "",
    )


def dock_batch(
    receptor: str | Path,
    ligands: Sequence[str | Path],
    box: DockBox,
    out_dir: str | Path,
    max_workers: int = 4,
    **kw,
) -> list[DockResult]:
    """Dock several ligands into one receptor, in parallel.

    Results come back in the order of ``ligands``, so they can be zipped with
    it. One ligand that fails does not stop the batch -- screening a set of
    targets or anti-targets should not be lost because one input is malformed.
    A failed ligand yields a :class:`DockResult` with no poses, a NaN
    ``best_score`` and the error text in ``stdout``, and the failure is logged;
    nothing is dropped from the returned list.

    Threads, not processes: each worker only waits on a Vina subprocess, so the
    GIL is irrelevant. Note that Vina itself defaults to every core it can
    detect, so ``max_workers`` workers will oversubscribe the machine -- pass
    ``cpu=`` to divide the cores if run time has to be predictable.
    """
    paths = [Path(p) for p in ligands]
    if not paths:
        return []
    stems = [p.stem for p in paths]
    clashes = sorted({s for s in stems if stems.count(s) > 1})
    if clashes:
        raise DockError(
            f"ligands share a file stem {clashes} and would overwrite each "
            f"other's poses in {out_dir}; rename them or dock them into "
            f"separate directories"
        )

    def one(lig: Path) -> DockResult:
        return dock(receptor, lig, box, out_dir, **kw)

    out: list[DockResult] = []
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        for lig, fut in zip(paths, [pool.submit(one, p) for p in paths]):
            try:
                out.append(fut.result())
            except Exception as exc:
                log.warning("docking %s failed: %s", lig.name, exc)
                out.append(DockResult(stdout=f"ERROR: {type(exc).__name__}: {exc}"))
    return out


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------


def pdb_to_pdbqt(
    pdb: str | Path,
    pdbqt: str | Path,
    receptor: bool = False,
    ph: float | None = None,
    obabel: str | None = None,
) -> Path:
    """Convert PDB to PDBQT with Open Babel. A stopgap, not the right tool.

    Vina reads only PDBQT, which carries partial charges and AutoDock atom
    types, and for a ligand a torsion tree saying which bonds may rotate. Open
    Babel will write that file, and it is usually already installed, which is
    why it is here -- but what it writes is not what the AutoDock developers
    maintain or validate:

    * **Ligands** should go through Meeko (``mk_prepare_ligand.py``), the
      supported path, which builds the torsion tree and atom typing Vina's
      scoring function was fitted against.
    * **Receptors** need the treatment Vina expects before conversion: polar
      hydrogens added (and non-polar ones merged) and charges assigned, by
      ``mk_prepare_receptor.py`` or AutoDockTools' ``prepare_receptor4.py``.
      ``receptor=True`` here only passes Open Babel's ``-xr``, which suppresses
      the torsion tree. It does not do the hydrogen or charge treatment: on
      ``tests/data/1crn.pdb`` (a hydrogen-free crystal structure) Open Babel
      3.1.0 writes all 327 atoms with a partial charge of ``+0.000``, so Vina's
      electrostatic and desolvation terms see an uncharged receptor.
      Affinities from a receptor prepared this way can rank poses within one
      run but are not comparable with published Vina numbers.

    ``ph`` adds hydrogens for that pH (Open Babel's ``-p``). Leave it unset to
    keep the hydrogens the input already has; setting it changes protonation
    states, which changes the score.
    """
    pdb, out = Path(pdb), Path(pdbqt)
    if not pdb.exists():
        raise DockError(f"no such structure file: {pdb}")
    exe = obabel or shutil.which("obabel")
    if exe is None:
        raise DockError(
            "obabel not found, so PDB cannot be converted to the PDBQT Vina "
            "needs.\n"
            "  install:  conda install -c conda-forge openbabel  (or brew "
            "install open-babel)\n"
            "  better:   pip install meeko, then mk_prepare_ligand.py for "
            "ligands and mk_prepare_receptor.py for receptors"
        )
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [exe, "-ipdb", str(pdb), "-opdbqt", "-O", str(out)]
    if receptor:
        cmd += ["-xr"]
    if ph is not None:
        cmd += ["-p", f"{ph:g}"]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if not out.exists() or out.stat().st_size == 0:
        raise DockError(
            f"obabel wrote no PDBQT (rc={proc.returncode}).\n"
            f"command: {' '.join(cmd)}\n"
            f"stderr tail: {(proc.stderr or '')[-800:]}"
        )
    return out
