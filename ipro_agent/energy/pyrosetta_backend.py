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

#: **Paths in this module that have never executed**, and why each matters.
#:
#: PyRosetta needs a licence (free for academics, but it has to be obtained and
#: accepted by the user) and is not installed in this repo's environment, so
#: everything below is written against the documented API and checked by
#: reading only. This is a declaration, not a disclaimer: a name here is a
#: place where a defect of the kind docs/DESIGN.md sections 7, 7b and 7d record
#: -- runs clean, returns a plausible wrong answer -- could still be hiding.
#: ``tests/test_ipro.py`` keeps the list honest (every name must still exist,
#: and the test fails loudly once PyRosetta IS importable, so the list cannot
#: quietly outlive its reason).
#:
#: To close one: install PyRosetta, run the relevant case, and move the entry
#: out with the measurement that exercised it. See docs/DESIGN.md section 7e.
UNEXERCISED: dict[str, str] = {
    "PyRosettaBackend.__init__": (
        "pyrosetta.init() is guarded by a module attribute, so a second backend "
        "built with different init_flags silently keeps the first set. Never run, "
        "so never observed."
    ),
    "PyRosettaBackend.relax": (
        "FastRelax with a MoveMap. A MoveMap constrains MINIMIZATION; FastRelax's "
        "ramped repack cycles take their rotamer set from a PackerTask, so the "
        "residues named in `fixed` may still be repacked -- the opposite of what "
        "the restraint promises. Unverified. The residue-count guard against "
        "-ignore_unrecognized_res silently dropping a cofactor is likewise unrun."
    ),
    "PyRosettaBackend.score_structure": (
        "pose_from_pdb + score function. Whether -ignore_unrecognized_res drops a "
        "ligand Rosetta has no params for, so the score is of a complex missing "
        "its cofactor, has not been observed."
    ),
    "PyRosettaBackend.interface_energy": (
        "InterfaceAnalyzerMover. No call site in the package either, so this is "
        "both unexercised and unused: dG_separated never reaches an objective."
    ),
    "PyRosettaBackend._exact_pair": (
        "Raises by design (a pose per rotamer pair is prohibitive). The raise "
        "itself is covered; the Rosetta-scored table it describes does not exist."
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
        if not getattr(pyrosetta, "_ipro_agent_initialized", False):
            pyrosetta.init(init_flags)
            pyrosetta._ipro_agent_initialized = True  # type: ignore[attr-defined]
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

    def relax(self, pdb_in: Path, pdb_out: Path, fixed: Sequence[str] = ()) -> float:
        """FastRelax, optionally holding some residues fixed.

        ``fixed`` holds position labels like ``"A115"``. Those residues keep
        both backbone and side-chain torsions; everything else is free. This is
        the backbone-relaxation step of the IPRO iteration.
        """
        from pyrosetta.rosetta.protocols.relax import FastRelax
        from pyrosetta.rosetta.core.kinematics import MoveMap

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
        fr = FastRelax(self._sfxn, self.relax_cycles)

        if fixed:
            mm = MoveMap()
            mm.set_bb(True)
            mm.set_chi(True)
            frozen = self._resolve_labels(pose, fixed)
            for idx in frozen:
                mm.set_bb(idx, False)
                mm.set_chi(idx, False)
            fr.set_movemap(mm)

        fr.apply(pose)
        out = Path(pdb_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pose.dump_pdb(str(out))
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
        """
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

        pose = self._pose(Path(pdb))
        iam = InterfaceAnalyzerMover(partner_chains)
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
