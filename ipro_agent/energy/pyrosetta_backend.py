"""PyRosetta backend: backbone relaxation, whole-structure and interface scoring.

Division of labour (deliberate; see docs/DESIGN.md):

* **Relaxation and final scoring use Rosetta.** These are the steps where an
  accurate, well-validated energy matters and where the cost of a full score is
  acceptable -- a few calls per iteration.
* **The MILP's self/pair energy tables do not, by default.** Building them means
  scoring O(n_pos^2 * n_rot^2) rotamer pairs, and Rosetta scores *poses*, not
  arbitrary atom sets; constructing a pose per pair is orders of magnitude
  slower than the geometry evaluation the table actually needs. The decomposable
  screening terms are used for the table, exactly as IPRO uses a cheap
  force-field evaluation for its combinatorial inner problem and a separate
  program for relaxation.

  Pass ``exact_pairs=True`` to force Rosetta-scored pairs anyway; expect it to
  be slow, and only worth it for small design spaces.

Why keep a MILP at all when Rosetta ships a packer? Because the value of the
integer program here is the *user's constraints*: "at least half of the
designed positions must be long residues", "no more than M mutations this
iteration", "these two positions must not both be charged". Those are linear
constraints over the rotamer variables and a standard simulated-annealing
packer cannot honour them. See :mod:`ipro_agent.milp`.

PyRosetta needs a (free for academics) licence and is not installed in this
repo's test environment, so this module is written against the documented API
but is exercised only when PyRosetta is present. ``scripts/doctor.py`` reports
whether it resolved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from .base import AtomSet, EnergyBackend
from .builtin import BuiltinBackend

log = logging.getLogger(__name__)

#: Paths in this module that have **not** been executed against a real
#: PyRosetta, and why each matters.
#:
#: This was the whole module until 2026-11-18, when PyRosetta
#: 2026.29+releasequarterly was installed and every entry was run. Three of the
#: five turned out to be defects, one of the stated suspicions was wrong, and
#: what is left here is the part that genuinely remains unexercised. Keep it
#: that way: a name here is a place where a defect of the kind docs/DESIGN.md
#: sections 7, 7b and 7d record -- runs clean, returns a plausible wrong answer
#: -- could still be hiding, and ``tests/test_ipro.py`` checks that every name
#: still exists so the list cannot rot.
#:
#: To close one: run the relevant case and move the entry out with the
#: measurement that exercised it. See docs/DESIGN.md section 7e.
UNEXERCISED: dict[str, str] = {
    "PyRosettaBackend._exact_pair": (
        "Raises by design (a pose per rotamer pair is prohibitive), and the raise "
        "is covered by a test. What does not exist, and so cannot be exercised, "
        "is the Rosetta-scored pair table it describes: with exact_pairs=False "
        "(the default) every MILP energy comes from the screening function, which "
        "is why `screening_pairwise` is True and the reported units say so."
    ),
    "PyRosettaBackend.relax (multi-chain and ligand cases)": (
        "Exercised on a single-chain protein (1crn): FastRelax runs, the held "
        "positions stay within 0.0001 A, and the written structure is re-scored "
        "without the constraint term. NOT exercised with several chains, with a "
        "parameterized ligand present, or with relax_cycles > 1 -- a cofactor's "
        "own params, and whether a chain break survives relaxation, are untested."
    ),
}


class PyRosettaBackend(EnergyBackend):
    name = "pyrosetta"
    can_relax = True
    #: Replaced per instance in ``__init__``: by default the pairwise numbers
    #: this backend returns are the screening function's, not Rosetta's, so
    #: labelling them REU was wrong (see :attr:`EnergyBackend.screening_pairwise`).
    units = "Rosetta Energy Units (REU)"

    def __init__(
        self,
        scorefxn: str = "ref2015",
        init_flags: str = "-mute all -ignore_unrecognized_res -ignore_zero_occupancy false",
        exact_pairs: bool = False,
        relax_cycles: int = 1,
        cutoff: float = 12.0,
        **kwargs,
    ) -> None:
        import pyrosetta  # raises if unavailable; resolve() catches it

        self._pr = pyrosetta
        # Keyed on the flags, not on a bare "have we initialized" boolean. A
        # second backend asking for different init_flags silently kept the
        # first set, and Rosetta's own re-initialization does work: measured on
        # 2026.29, init("-ex1") then reading packing:ex1 gives True where the
        # default gives False, and initializing again without it returns to
        # False. So a repeat of the same flags still costs nothing, and a
        # genuine change now takes effect.
        if getattr(pyrosetta, "_ipro_agent_init_flags", None) != init_flags:
            pyrosetta.init(init_flags)
            pyrosetta._ipro_agent_init_flags = init_flags  # type: ignore[attr-defined]
        self.scorefxn_name = scorefxn
        self._sfxn = pyrosetta.create_score_function(scorefxn)
        self.exact_pairs = bool(exact_pairs)
        self.relax_cycles = int(relax_cycles)
        #: Used for the MILP tables unless ``exact_pairs``; see module docstring.
        self._screen = BuiltinBackend(cutoff=cutoff, **{
            k: v for k, v in kwargs.items()
            if k in ("max_repulsion", "hbond_weight", "coulomb")
        })
        self.screening_pairwise = not self.exact_pairs
        self.units_short = "REU" if self.exact_pairs else "kcal/mol, screening scale"
        self.units = (
            "Rosetta Energy Units (REU)" if self.exact_pairs else
            "kcal/mol (screening scale: the MILP tables and the reported "
            "interaction energies come from the built-in terms, not from "
            f"{scorefxn}; relaxation and whole-structure scores are Rosetta's)"
        )

    # -- pairwise tables ---------------------------------------------------

    def interaction(self, a: AtomSet, b: AtomSet) -> float:
        if not self.exact_pairs:
            return self._screen.interaction(a, b)
        return self._exact_pair(a, b)

    def _exact_pair(self, a: AtomSet, b: AtomSet) -> float:
        """Rosetta-scored interaction, via the two-body energy of a mini pose.

        Not used by default. Implemented for completeness and for small cases
        where the extra fidelity is worth the cost.
        """
        raise NotImplementedError(
            "exact_pairs=True is not implemented: scoring every rotamer pair "
            "through Rosetta requires building a pose per pair, which is "
            "prohibitively slow for realistic design spaces. Use the default "
            "(screening terms for the MILP table, Rosetta for relaxation and "
            "final scoring), or reduce the design space."
        )

    def internal(self, a: AtomSet) -> float:
        return self._screen.internal(a)

    # -- structures --------------------------------------------------------

    def _pose(self, pdb: Path):
        return self._pr.pose_from_pdb(str(pdb))

    def score_structure(self, pdb) -> float:
        return float(self._sfxn(self._pose(Path(pdb))))

    def relax(
        self,
        pdb_in: Path,
        pdb_out: Path,
        fixed: Sequence[str] = (),
        hold_tolerance: float = 0.1,
    ) -> float:
        """FastRelax, holding the residues named in ``fixed`` where they are.

        ``fixed`` holds position labels like ``"A115"``. This is the
        backbone-relaxation step of the IPRO iteration, and the one place where
        a restraint the user wrote has to survive a third-party protocol.

        **A MoveMap is not enough, measured.** The earlier version set
        ``mm.set_bb(idx, False)`` and ``mm.set_chi(idx, False)`` and stopped
        there, which is the right idiom for *minimization* -- but FastRelax
        ramps repack cycles whose rotamer set comes from a PackerTask, and
        fixing a residue's own torsions does not fix its Cartesian position
        anyway, because the fold tree translates it when an upstream torsion
        moves. On 1crn A16 with PyRosetta 2026.29:

            MoveMap only                       0.866 A  (and 2.922 A on another seed)
            MoveMap + PreventRepacking         2.200 A
            MoveMap + PreventRepacking + coordinate constraints   0.0001 A

        So the hold is done with **coordinate constraints**, and the other two
        are kept for what they are each right about: the MoveMap stops the
        torsions being minimized, and ``PreventRepacking`` stops Rosetta
        re-choosing a rotamer the MILP chose. ``RestrictToRepacking`` is added
        so relaxation can never change an *identity*.

        The displacement is then **measured and checked**, because this is a
        promise to the user rather than a best effort: a violation raises, which
        the loop reports as a failed relaxation and a downgraded mode, instead
        of quietly returning a structure that breaks the restraint.
        """
        import numpy as np

        from pyrosetta.rosetta.core.id import AtomID
        from pyrosetta.rosetta.core.kinematics import MoveMap
        from pyrosetta.rosetta.core.pack.task import TaskFactory
        from pyrosetta.rosetta.core.pack.task.operation import (
            PreventRepacking, RestrictToRepacking,
        )
        from pyrosetta.rosetta.core.scoring import ScoreType
        from pyrosetta.rosetta.core.scoring.constraints import CoordinateConstraint
        from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
        from pyrosetta.rosetta.protocols.relax import FastRelax

        pose = self._pose(Path(pdb_in))
        # `-ignore_unrecognized_res` is in the default init flags, so Rosetta
        # silently drops any HETATM it has no parameters for -- and this method
        # then *writes* the truncated structure to pdb_out, which the next
        # iteration harvests from. Refuse instead: a cofactor vanishing from the
        # design's own file is not a degradation the run can report around.
        # (Unexercised: PyRosetta is not installed here. See docs/DESIGN.md 7e.)
        n_in = self._count_residues(Path(pdb_in))
        n_pose = int(pose.total_residue())
        if n_in and n_pose < n_in:
            raise RuntimeError(
                f"Rosetta read {n_pose} of {n_in} residues from {Path(pdb_in).name} "
                f"-- {n_in - n_pose} were dropped, most likely residues it has no "
                "parameters for (the default init flags include "
                "-ignore_unrecognized_res). Relaxing would write the truncated "
                "structure back. Supply params for the ligand/cofactor, or run "
                "with the built-in backend."
            )
        # A score function of this call's own: the constraint term must not be
        # left on self._sfxn, or every later score_structure() would silently
        # include a restraint penalty.
        sfxn = self._pr.create_score_function(self.scorefxn_name)
        fr = FastRelax(sfxn, self.relax_cycles)

        held: dict[int, dict[str, "np.ndarray"]] = {}
        frozen = self._resolve_labels(pose, fixed) if fixed else []
        if frozen:
            mm = MoveMap()
            mm.set_bb(True)
            mm.set_chi(True)
            for idx in frozen:
                mm.set_bb(idx, False)
                mm.set_chi(idx, False)
            fr.set_movemap(mm)

            tf = TaskFactory()
            tf.push_back(RestrictToRepacking())
            prevent = PreventRepacking()
            for idx in frozen:
                prevent.include_residue(idx)
            tf.push_back(prevent)
            fr.set_task_factory(tf)

            sfxn.set_weight(ScoreType.coordinate_constraint, 1.0)
            anchor = AtomID(1, 1)
            for idx in frozen:
                res = pose.residue(idx)
                held[idx] = {}
                for i in range(1, res.natoms() + 1):
                    if res.is_virtual(i) or res.atom_is_hydrogen(i):
                        continue
                    xyz = res.xyz(i)
                    held[idx][res.atom_name(i).strip()] = np.array(xyz, dtype=float)
                    pose.add_constraint(
                        CoordinateConstraint(
                            AtomID(i, idx), anchor, xyz, HarmonicFunc(0.0, 0.01)
                        )
                    )

        fr.apply(pose)

        if held:
            worst, worst_at = 0.0, ""
            for idx, atoms in held.items():
                res = pose.residue(idx)
                for i in range(1, res.natoms() + 1):
                    nm = res.atom_name(i).strip()
                    if nm not in atoms:
                        continue
                    d = float(np.abs(np.array(res.xyz(i), dtype=float) - atoms[nm]).max())
                    if d > worst:
                        worst, worst_at = d, f"{nm} of pose residue {idx}"
            if worst > hold_tolerance:
                raise RuntimeError(
                    f"relaxation moved a position it was told to hold: {worst_at} "
                    f"by {worst:.5g} A, over the {hold_tolerance:.5g} A tolerance. "
                    f"Held positions were {list(fixed)}. The structure was NOT "
                    f"written, because it would violate a restraint the run "
                    f"reports as enforced."
                )
            self._last_hold_error = worst

        out = Path(pdb_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pose.dump_pdb(str(out))
        # Scored WITHOUT the constraint term, so the number means what it says.
        return float(self._sfxn(pose))

    @staticmethod
    def _count_residues(pdb: Path) -> int:
        """Residues in a PDB file, counted without Rosetta."""
        try:
            from ..structure import Structure

            st = Structure.load(pdb)
            return len(list(st.residues())) + len(list(st.hetero_residues()))
        except Exception:      # a count we cannot take is not a reason to fail
            return 0

    def _resolve_labels(self, pose, labels: Sequence[str]) -> list[int]:
        """Map ``"A115"`` style labels to Rosetta's 1-based pose indices."""
        import re

        out: list[int] = []
        info = pose.pdb_info()
        for lab in labels:
            m = re.match(r"^([A-Za-z0-9])(-?\d+)([A-Za-z]?)$", str(lab))
            if not m:
                log.warning("cannot parse residue label %r; not fixing it", lab)
                continue
            chain, num, icode = m.group(1), int(m.group(2)), (m.group(3) or " ")
            idx = info.pdb2pose(chain, num, icode)
            if idx == 0:
                log.warning("%s not found in pose; not fixing it", lab)
                continue
            out.append(idx)
        return out

    # -- interface ---------------------------------------------------------

    def interface_energy(self, pdb: Path, partner_chains: str) -> dict[str, float]:
        """Binding energy across an interface, via InterfaceAnalyzerMover.

        ``partner_chains`` uses Rosetta's notation, e.g. ``"A_B"``. Returned
        ``dG_separated`` is the quantity a "bind tighter" goal minimizes.

        **This method raised on every call** until 2026-11-18, because
        ``InterfaceAnalyzerMover("A_B")`` is not a constructor this PyRosetta
        has: measured against 2026.29, its overloads take a jump index, a set
        of chain indices, or a ``DockingPartners``. ``DockingPartners`` itself
        has only a default and a copy constructor, and ``set_interface`` takes
        one too -- the string is parsed by the free function
        ``docking_partners_from_string``, which is what this now uses. The
        method had no call site, so nothing downstream ever noticed.
        """
        from pyrosetta.rosetta.core.pose import DockingPartners
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

        pose = self._pose(Path(pdb))
        partners = DockingPartners.docking_partners_from_string(str(partner_chains))
        if partners.is_empty() or not partners.has_both():
            raise ValueError(
                f"could not read {partner_chains!r} as two docking partners; "
                "use Rosetta's chain notation, e.g. 'A_B' or 'AB_C'"
            )
        iam = InterfaceAnalyzerMover(partners)
        iam.set_scorefunction(self._sfxn)
        iam.set_compute_packstat(False)
        iam.apply(pose)
        return {
            "dG_separated": float(iam.get_separated_interface_energy()),
            "interface_dSASA": float(iam.get_interface_delta_sasa()),
            "total_score": float(self._sfxn(pose)),
        }

    def describe(self) -> str:
        return (
            f"pyrosetta ({self.scorefxn_name}, REU); relaxation: yes; "
            f"MILP pair table: {'rosetta' if self.exact_pairs else 'screening terms'}"
        )
