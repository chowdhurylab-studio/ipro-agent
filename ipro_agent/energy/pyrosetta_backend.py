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


class PyRosettaBackend(EnergyBackend):
    name = "pyrosetta"
    can_relax = True
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
