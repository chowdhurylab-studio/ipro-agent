"""The IPRO iteration loop.

Order of operations per iteration, following IPRO's own implementation:

1. choose which design positions to re-decide this iteration (1-3 contiguous);
2. perturb the backbone of a short segment (1-5 residues, phi/psi offsets from
   N(0, 1.5 deg) truncated to +/-5 deg), then relieve the strain by restrained
   minimization;
3. solve the rotamer-selection MILP for the chosen positions and their
   neighbourhood (:mod:`ipro_agent.milp`);
4. every third iteration, rigid-body redock the target molecules;
5. relax the whole complex and compute its energy;
6. accept or reject by the Metropolis test, and add an integer cut so the
   sequence just evaluated is never revisited.

Two modes, and the loop reports which one ran
---------------------------------------------
**full** -- all six steps. Needs a backend that can relax (PyRosetta), because
steps 2 and 5 are meaningless without one: perturbing a backbone and then not
minimizing it just manufactures clashes.

**fixed-backbone** -- steps 1, 3, 5 (scoring only), 6. The backbone never
moves, so the search is over sequence and side-chain packing, driven by the
MILP and the integer cuts. This is what runs when PyRosetta is absent. It is a
real and widely used protocol, but it is *not* the published IPRO protocol, and
``state.json`` records ``mode: fixed-backbone`` so that no one mistakes one for
the other.

Because the backbone is constant in fixed-backbone mode, rotamers are harvested
from PyMOL once and reused, which is what makes a short run take seconds rather
than minutes.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import DesignSpec, THREE_TO_ONE
from .energy.base import AtomSet, EnergyBackend
from .energy import resolve as resolve_energy
from .goals import Goal, ScoreContext, build_goal, metropolis_accept
from .milp import Constraint, EnergyTables, RotamerMILP, Rotamer, build_tables
from .assemblies import CombinedTables, combine, split_vpos, vpos
from .docking import Pose, _pose_between, moved_atomset
from .docking import redock as mc_redock
from .restraints import RestraintError, RestraintPlan, plan_restraints
from .rotamers import build_template, harvest
from .state import IterationRecord, RunDir, Status
from .structure import Structure

log = logging.getLogger(__name__)

BACKBONE = {"N", "CA", "C", "O", "OXT"}


@dataclass
class RunResult:
    """Outcome of a whole run."""

    run_dir: Path
    mode: str
    best_score: float
    start_score: float
    best_mutations: dict[str, str]
    best_sequence: str
    iterations: int
    accepted: int
    #: Mutations as ``D115W`` strings, built by the loop (which knows the
    #: native identities and residue numbers).
    mutation_string: str = ""
    #: Designed coordinate files written for the best design.
    structures: list[str] = field(default_factory=list)
    ensemble: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        muts = self.mutation_string or ", ".join(
            f"{p}->{THREE_TO_ONE.get(a, a)}" for p, a in sorted(self.best_mutations.items())
        )
        return "\n".join([
            f"mode: {self.mode}",
            f"iterations: {self.iterations} (accepted {self.accepted})",
            f"score: {self.start_score:.4f} -> {self.best_score:.4f} "
            f"(change {self.best_score - self.start_score:+.4f})",
            f"mutations ({len(self.best_mutations)}): {muts or '(none -- native retained)'}",
        ])


# ---------------------------------------------------------------------------
# Atom bookkeeping
# ---------------------------------------------------------------------------


def _atoms_of(st: Structure, skip_sidechains_at: set[str] = frozenset()) -> AtomSet:
    names, coords, resn, keys = [], [], [], []
    for res in list(st.residues()) + list(st.hetero_residues()):
        ref = st.ref(res)
        drop_sc = ref.label in skip_sidechains_at
        for at in res:
            if at.element == "H":
                continue
            nm = at.get_name().strip()
            if drop_sc and nm not in BACKBONE:
                continue
            names.append(nm)
            coords.append(at.coord)
            resn.append(res.get_resname().strip())
            keys.append(ref.key)
    return AtomSet(
        names=names,
        coords=np.array(coords) if coords else np.zeros((0, 3)),
        resnames=resn,
        reskeys=keys,
    )



def _read_pose_coords(path) -> "np.ndarray":
    """Heavy-atom coordinates from a PDB or PDBQT pose file, in file order."""
    import pathlib as _pl

    coords = []
    for line in _pl.Path(path).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        el = line[76:78].strip().upper() if len(line) >= 78 else ""
        name = line[12:16].strip().upper()
        # PDBQT uses its own type column; hydrogens are H/HD/HS.
        if el in ("H", "HD", "HS") or (not el and name.startswith("H")):
            continue
        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(coords, dtype=float)

class IPROLoop:
    """Runs the IPRO procedure for one :class:`DesignSpec`."""

    def __init__(
        self,
        spec: DesignSpec,
        run_dir: str | Path,
        backend: EnergyBackend | None = None,
        goal: Goal | None = None,
        extra_constraints: Sequence[Constraint] = (),
    ) -> None:
        self.spec = spec
        self.run = RunDir(run_dir)
        self.backend = backend or resolve_energy(spec.backends.energy, cutoff=spec.loop.energy_cutoff)
        self.goal = goal or build_goal(spec)
        self.rng = random.Random(spec.loop.random_seed)
        self.constraints = list(self.goal.default_constraints()) + list(extra_constraints)
        # Three honest modes rather than two:
        #   full            perturb + relax + repack  (needs a relaxing backend)
        #   backbone-mc     perturb + repack + Metropolis, no gradient relaxation
        #   fixed-backbone  repack only
        if not spec.loop.perturb_backbone:
            self.mode = "fixed-backbone"
        elif self.backend.can_relax:
            self.mode = "full"
        else:
            self.mode = "backbone-mc"
        # Qualifications the goal raised at construction travel with the run.
        self.notes: list[str] = list(getattr(self.goal, "notes", []))
        if self.mode == "fixed-backbone":
            self.notes.append(
                "the backbone never moves (loop.perturb_backbone is false), so the "
                "search is over sequence and side-chain packing only. That is a "
                "legitimate and much cheaper protocol, but it is not IPRO's: set "
                "loop.perturb_backbone: true to perturb the backbone each iteration."
            )
        elif self.mode == "backbone-mc":
            self.notes.append(
                f"backbone Monte Carlo: the backbone is perturbed each iteration and "
                f"strained moves are rejected, but there is no gradient relaxation "
                f"afterwards because energy backend '{self.backend.name}' cannot "
                f"relax. The published protocol minimizes after perturbing; install "
                f"PyRosetta for that."
            )

        # Loaded structures, by role.
        self._design_st: Structure | None = None
        #: The design molecule's file as currently used -- the input, or the
        #: indel-edited copy. PyMOL harvests from a path, so this is the single
        #: source of truth for both harvesting and writing structures.
        self._design_pdb = None
        self._partners: dict[str, Structure] = {}
        self._rotamer_cache: dict[str, list[Rotamer]] | None = None
        self._native: dict[str, str] = {}
        self._visited: list[dict[str, str]] = []
        #: Filled in by run_loop once the rotamer tables exist.
        self._restraints: RestraintPlan = RestraintPlan()
        #: Native identity keyed by *virtual* position, built once the
        #: assemblies are combined. With one assembly it equals _native.
        self._native_v: dict[str, str] = {}
        #: Binding assemblies, resolved (one implicit one if none were given).
        self._assemblies = spec.resolved_assemblies()
        #: Assembly name -> its partner structures.
        self._asm_partners: dict[str, list[Structure]] = {}
        self._combined: CombinedTables | None = None
        #: Current rigid-body placement of each partner, per assembly. Keyed
        #: (assembly, molecule); absent means "as supplied".
        self._poses: dict[tuple[str, str], Pose] = {}
        #: Partner atoms as supplied, cached so a pose is always applied to the
        #: original coordinates rather than compounding on itself.
        self._base_atoms: dict[str, AtomSet] = {}
        #: Which placement backend resolved for this run.
        self._dock_backend: str = "none"
        #: Keys for notes that should be emitted at most once per run.
        self._once: set[str] = set()
        #: Most recent pore profile, kept for reporting.
        self._last_pore = None

    # -- setup -------------------------------------------------------------

    def prepare(self) -> None:
        """Load structures, check the design positions exist, harvest rotamers."""
        spec = self.spec
        dm = spec.design_molecule
        self._design_st = Structure.load(dm.path, dm.chains or None)
        self._design_pdb = dm.path

        if spec.indels:
            from .indel import apply_terminal, classify

            plan = classify(self._design_st, spec.indels)
            if plan.has_internal:
                # Refusing is the right answer: an internal edit without loop
                # closure leaves a broken peptide bond, which still looks like a
                # protein and scores like nonsense.
                detail = "\n  ".join(why for _s, why in plan.internal)
                raise NotImplementedError(
                    f"{len(plan.internal)} of {len(spec.indels)} requested "
                    f"insertion/deletion(s) are internal and cannot be done here:\n  "
                    f"{detail}\n"
                    "Changing a chain's length in the middle needs loop closure "
                    "(Protein-InDelMaker uses PyRosetta's GeneralizedKIC over a "
                    "6-residue window, then Cartesian FastRelax). Terminal indels "
                    "need no closure and are supported. Either move the edit to a "
                    "terminus, or build the variant with Protein-InDelMaker "
                    "(github.com/maranasgroup/Protein-InDelMaker) and design on "
                    "that structure. See docs/DESIGN.md."
                )
            result = apply_terminal(
                self._design_st, plan.terminal,
                phi=float(spec.goal_options.get("insert_phi", -139.0)),
                psi=float(spec.goal_options.get("insert_psi", 135.0)),
            )
            # Everything downstream reads the design molecule from a file
            # (PyMOL harvests rotamers from a path), so the edited structure is
            # written once and used from here on. The input file is untouched.
            edited = self.run.path / "logs" / "design_after_indels.pdb"
            edited.parent.mkdir(parents=True, exist_ok=True)
            self._design_st.save(edited)
            self._design_st = Structure.load(edited, dm.chains or None)
            self._design_pdb = edited
            self.notes.append(f"indels applied: {result.describe()}")
            self.notes.extend(result.notes)
            if result.added:
                self.notes.append(
                    "inserted residues are glycine in the structure; list them as "
                    "design positions to have their identities chosen: "
                    + ", ".join(result.added)
                )
            self.run.log(f"indels: {result.describe()}")

        for m in spec.targets + spec.anti_targets:
            self._partners[m.name] = Structure.load(m.path, m.chains or None)

        # Fail early and specifically if a stated position is not in the file:
        # a silently dropped design position would change the problem.
        missing = []
        for p in spec.positions:
            try:
                res = self._design_st.find(p.chain, p.number, p.insertion_code)
                p.native = res.get_resname().strip()
            except Exception as exc:
                missing.append(f"{p.label}: {exc}")
        if missing:
            raise ValueError(
                "design position(s) not found in "
                f"{self._design_pdb.name}:\n  " + "\n  ".join(missing)
            )
        self._native = {p.label: (p.native or "") for p in spec.positions}

        # Each assembly sees only its own partner molecules, so its template --
        # and therefore its rotamer energies -- differ. That is what makes
        # per-assembly repacking meaningful.
        for a in self._assemblies:
            self._asm_partners[a.name] = [
                self._partners[m.name] for m in spec.assembly_molecules(a)
                if m.name in self._partners
            ]
        if len(self._assemblies) > 1:
            self.notes.append(
                f"designing one sequence across {len(self._assemblies)} binding "
                "assemblies: rotamers are chosen separately in each, identity is "
                "shared, and every gating assembly must pass acceptance"
            )

        self._dock_backend = self._resolve_dock()
        self.run.log(f"placement backend: {self._dock_backend}")

        self.run.write_spec(spec.to_dict())
        self.run.log(f"prepared: mode={self.mode} backend={self.backend.describe()}")

        labels = [p.label for p in spec.positions]
        self.run.log(f"harvesting rotamers at {len(labels)} positions")
        rots, native, errors = harvest(
            self._design_pdb, spec.positions, workdir=self.run.path / "logs"
        )
        if errors:
            self.notes.append(f"PyMOL could not build some rotamers: {json.dumps(errors)[:400]}")
        # Trust the structure for native identity, but keep PyMOL's as a check.
        for lab, aa in native.items():
            if self._native.get(lab) and aa and self._native[lab] != aa:
                self.notes.append(
                    f"native identity disagreement at {lab}: structure says "
                    f"{self._native[lab]}, PyMOL says {aa}"
                )
        empty = [p for p, v in rots.items() if not v]
        if empty:
            raise ValueError(
                f"no rotamers could be built at {empty}. Check that the allowed "
                "residues are valid and that PyMOL's mutagenesis wizard works "
                "(scripts/doctor.py)."
            )
        self._rotamer_cache = rots
        n = sum(len(v) for v in rots.values())
        self.run.log(f"harvested {n} rotamers")
        if n > self.spec.loop.max_rotamers:
            self.notes.append(
                f"{n} rotamers exceeds max_rotamers={self.spec.loop.max_rotamers}; "
                "pruning will be tighter than usual"
            )

    # -- scoring -----------------------------------------------------------

    def _contexts_for(
        self, chosen: dict[str, Rotamer]
    ) -> tuple[ScoreContext, dict[str, ScoreContext]]:
        """Measure a candidate in every assembly.

        ``chosen`` is keyed by virtual position, so each assembly's own packing
        is used for its own partners. Returns the merged context the goal scores
        -- each partner's interaction energy measured in the assembly it belongs
        to -- and the per-assembly contexts the conjunctive acceptance test
        needs.
        """
        assert self._design_st is not None
        design_fixed = _atoms_of(self._design_st, {p.label for p in self.spec.positions})

        per: dict[str, ScoreContext] = {}
        merged = ScoreContext()
        weights = {a.name: a.weight for a in self._assemblies}

        for a in self._assemblies:
            rots = [
                r for v, r in chosen.items()
                if (split_vpos(v)[0] or a.name) == a.name
            ]
            design_all = design_fixed.extend(AtomSet.concat([r.atoms for r in rots]))
            ctx = ScoreContext()
            partners = self.spec.assembly_molecules(a)
            for m in partners:
                if m.name not in self._partners:
                    continue
                ctx.interaction[m.name] = self.backend.interaction(
                    design_all, self._partner_atoms(a.name, m.name)
                )
            ctx.design_only = self.backend.internal(design_all)
            ctx.total = ctx.design_only + sum(ctx.interaction.values()) + sum(
                self.backend.internal(self._partner_atoms(a.name, m.name))
                for m in partners if m.name in self._partners
            )
            if getattr(self.goal, "wants_geometry", False):
                ctx.geometry.update(self._measure_geometry(design_all))
            per[a.name] = ctx
            # Each partner is scored in its own assembly, so the merged view has
            # exactly one interaction energy per molecule -- which is what the
            # goal expects.
            merged.interaction.update(ctx.interaction)
            merged.design_only += weights[a.name] * ctx.design_only
            merged.total += weights[a.name] * ctx.total
            # The channel is a property of the design molecule, which is shared,
            # so the first assembly's measurement stands for all of them.
            for k, v in ctx.geometry.items():
                merged.geometry.setdefault(k, v)
        return merged, per

    def _context_for(self, chosen: dict[str, Rotamer]) -> ScoreContext:
        """Measure a candidate without writing it to disk.

        The candidate's atoms are the design molecule's fixed part (backbone
        everywhere, side chains of non-design residues) plus the chosen
        rotamers. Interaction energy with each partner is then a direct
        cross-term evaluation, since for disjoint sets
        ``E(A u B) = E(A) + E(B) + interaction(A, B)`` -- so the interaction
        energy IPRO defines as ``E(complex) - E(design) - E(partner)`` *is*
        ``interaction(design, partner)``, with no need to score three
        structures.
        """
        return self._contexts_for(chosen)[0]

    # -- measuring the channel ----------------------------------------------

    def _pore_axis(self):
        """Origin and direction of the channel being designed.

        Taken from ``goal_options.pore_axis`` when given -- two atom references
        spanning the channel, e.g. ``["A18:CA", "A119:CA"]`` -- and otherwise
        from the plane of the design positions themselves, which line the
        constriction. The latter reproduces OmpF's published eyelet; the
        protein's own principal axis does not, so it is never used.
        """
        import numpy as _np

        from .config import AtomRef
        from .pore import axis_from_lining, ring_flatness

        assert self._design_st is not None
        spec = self.spec
        given = spec.goal_options.get("pore_axis")
        if given:
            if len(given) != 2:
                raise ValueError(
                    "goal_options.pore_axis takes exactly two atom references "
                    f"spanning the channel, e.g. ['A18:CA', 'A119:CA']; got {given!r}"
                )
            pts = []
            for ref_text in given:
                ref = AtomRef.parse(ref_text)
                res = self._design_st.find(ref.chain, ref.number, ref.insertion_code)
                match = [a for a in res if a.get_name().strip().upper() == ref.atom]
                if not match:
                    raise ValueError(f"pore_axis: {ref} is not in the design molecule")
                pts.append(_np.asarray(match[0].coord, dtype=float))
            direction = pts[1] - pts[0]
            return (pts[0] + pts[1]) / 2.0, direction, 0.0

        cas = []
        for p in spec.positions:
            try:
                res = self._design_st.find(p.chain, p.number, p.insertion_code)
            except Exception:
                continue
            match = [a for a in res if a.get_name().strip().upper() == "CA"]
            if match:
                cas.append(_np.asarray(match[0].coord, dtype=float))
        if len(cas) < 3:
            raise ValueError(
                "a channel axis needs at least three design positions to fit a "
                "plane through, or goal_options.pore_axis given explicitly"
            )
        ring = _np.array(cas)
        origin, direction = axis_from_lining(ring)
        return origin, direction, ring_flatness(ring)

    def _measure_geometry(self, design_atoms: AtomSet) -> dict[str, float]:
        """Bottleneck of the channel through the designed side chains."""
        from .pore import pore_profile

        opts = self.spec.goal_options
        try:
            origin, direction, flatness = self._pore_axis()
        except Exception as exc:
            if "pore_axis" not in self._once:
                self._once.add("pore_axis")
                self.notes.append(
                    f"channel geometry not measured: {exc}. Any goal option that "
                    "depends on a pore radius is therefore inactive."
                )
            return {}

        if flatness > float(opts.get("pore_axis_flatness_warn", 0.7)):
            if "pore_flat" not in self._once:
                self._once.add("pore_flat")
                self.notes.append(
                    f"the design positions are not close to coplanar (flatness "
                    f"{flatness:.2f}, 0 = a flat ring), so the channel axis fitted "
                    "through them is unreliable. Give goal_options.pore_axis as two "
                    "atoms spanning the channel."
                )

        window = opts.get("pore_window")
        if window is None:
            # Centre an 8 A window on the lining residues, as PoreDesigner
            # measured only the constriction rather than the whole barrel.
            import numpy as _np

            proj = (design_atoms.coords - origin) @ direction
            lo = float(proj.min())
            half = float(opts.get("pore_window_half", 4.0))
            window = (-lo - half, -lo + half)
        else:
            window = (float(window[0]), float(window[1]))

        prof = pore_profile(
            design_atoms.coords, list(design_atoms.names),
            axis_origin=origin, axis_direction=direction,
            spacing=float(opts.get("pore_spacing", 0.5)),
            window=window,
            max_centre_offset=float(opts.get("pore_max_centre_offset", 6.0)),
        )
        if not prof.radii:
            return {}
        self._last_pore = prof
        return {
            "min_pore_radius": prof.min_radius,
            "min_pore_diameter": prof.min_diameter,
            "max_pore_radius": float(max(prof.radii)),
            "pore_axis_flatness": flatness,
        }

    # -- moving the backbone -------------------------------------------------

    def _rebuild_for_backbone(self, pdb_path) -> CombinedTables:
        """Re-harvest rotamers from a moved backbone and rebuild the tables.

        Everything has to be redone, not just the self energies: a rotamer's
        coordinates are built on the backbone, so moving the backbone changes
        the rotamer-rotamer pair energies too. That is the cost of perturbation
        and the reason it is opt-in.
        """
        rots, native, errors = harvest(
            pdb_path, self.spec.positions, workdir=self.run.path / "logs"
        )
        if errors and "perturb_harvest_errors" not in self._once:
            self._once.add("perturb_harvest_errors")
            self.notes.append(
                "after a backbone perturbation PyMOL could not build some "
                f"rotamers: {json.dumps(errors)[:300]}"
            )
        empty = [k for k, v in rots.items() if not v]
        if empty:
            raise RuntimeError(
                f"no rotamers could be built at {empty} on the perturbed backbone"
            )
        self._rotamer_cache = rots
        self._design_st = Structure.load(pdb_path)
        self._design_pdb = pdb_path
        return self.build_combined()

    def _perturb_once(self, iteration: int):
        """Perturb a copy of the current backbone and return (file, record)."""
        from .perturb import perturb_backbone

        assert self._design_st is not None
        dm = self.spec.design_molecule
        chain = (dm.chains or self._design_st.chain_ids())[0]
        work = self.run.path / "logs" / "backbone"
        work.mkdir(parents=True, exist_ok=True)
        candidate = work / f"iter{iteration:04d}.pdb"

        moved = self._design_st.copy()
        record = perturb_backbone(
            moved, chain,
            sigma_deg=self.spec.loop.perturbation_sigma_deg,
            cap_deg=self.spec.loop.perturbation_cap_deg,
            length_range=self.spec.loop.perturbation_segment,
            rng=self.rng,
        )
        moved.save(candidate)
        return candidate, record

    # -- writing structures -------------------------------------------------

    def _write_design(
        self,
        path,
        chosen: dict[str, "Rotamer"],
        assembly: str,
        header: list[str] | None = None,
    ):
        """Write the complex that was scored: mutated design plus its partners.

        The design molecule is rebuilt through PyMOL so the file carries the
        *chosen rotamer* at each design position, not just the right residue
        type -- otherwise the coordinates would not be the ones that produced
        the reported energy.
        """
        from pathlib import Path as _Path

        from .output import write_complex

        if self._design_st is None:
            return None
        picks = {
            (split_vpos(v)[1] or v): (r.aa, r.state)
            for v, r in chosen.items()
            if (split_vpos(v)[0] or assembly) == assembly
        }
        work = self.run.path / "logs" / "structure_build"
        work.mkdir(parents=True, exist_ok=True)
        design_only = work / "design_only.pdb"
        try:
            from .mutate import apply_design

            res = apply_design(self._design_pdb, design_only, picks)
            if res.errors:
                self.notes.append(
                    f"some positions could not be written into the structure: "
                    f"{json.dumps(res.errors)[:200]}"
                )
        except Exception as exc:
            if "no_structure_note" not in self._once:
                self._once.add("no_structure_note")
                self.notes.append(
                    "could not write designed coordinates "
                    f"({type(exc).__name__}: {exc}). The sequence and energies are "
                    "still reported; structure output needs a working PyMOL "
                    "(scripts/doctor.py)."
                )
            return None

        built = Structure.load(design_only)
        a = next(x for x in self._assemblies if x.name == assembly)
        partners = {
            m.name: self._partners[m.name]
            for m in self.spec.assembly_molecules(a)
            if m.name in self._partners
        }
        poses = {
            name: self._poses.get((assembly, name))
            for name in partners
            if self._poses.get((assembly, name)) is not None
        }
        written = write_complex(
            _Path(path), built, partners, poses, header=header or []
        )
        for note in written.notes:
            if note not in self.notes:
                self.notes.append(note)
        return written

    def _design_header(self, score: float, mutations: dict[str, str], assembly: str) -> list[str]:
        """REMARK lines so a stray PDB can still be traced back to its run."""
        muts = self._mutation_string(mutations) or "none"
        out = [
            "REMARK   1 WRITTEN BY ipro-agent",
            f"REMARK   1 RUN        {self.run.path.name}",
            f"REMARK   1 GOAL       {self.spec.goal}",
            f"REMARK   1 MODE       {self.mode}",
            f"REMARK   1 BACKEND    {self.backend.name}",
            f"REMARK   1 SCORE      {score:.4f} ({self.backend.units})",
            f"REMARK   1 MUTATIONS  {muts}",
        ]
        if len(self._assemblies) > 1:
            out.append(f"REMARK   1 ASSEMBLY   {assembly}")
        out.append(
            "REMARK   1 NOTE       energies from this backend are a screening"
            if not self.backend.can_relax else
            "REMARK   1 NOTE       see best/design.json for the score breakdown"
        )
        return out

    def _write_best_structures(
        self, chosen: dict[str, "Rotamer"], score: float, mutations: dict[str, str]
    ) -> list[str]:
        """One file per assembly, since the packing differs between them."""
        written: list[str] = []
        multi = len(self._assemblies) > 1
        for a in self._assemblies:
            name = f"design_{a.name}.pdb" if multi else "design.pdb"
            got = self._write_design(
                self.run.path / "best" / name, chosen, a.name,
                header=self._design_header(score, mutations, a.name),
            )
            if got is not None:
                written.append(got.describe())
                self.run.log(f"wrote best structure {got.describe()}")
        return written

    # -- placing the partners ----------------------------------------------

    def _resolve_dock(self) -> str:
        """Which placement program to use, honouring the user's choice.

        ``auto`` prefers AutoDock Vina when it is installed -- the same
        precedence as ``energy: auto`` preferring PyRosetta -- and falls back to
        IPRO's own local Monte Carlo otherwise.

        The two are different operations, not two qualities of one. Vina
        searches the box **globally** with its own scoring function and is much
        the heavier: it writes a receptor file, runs an external binary and
        parses poses. The Monte Carlo **refines** an existing placement in the
        six rigid-body degrees of freedom using the same energy the MILP uses,
        which is cheap and reproducible from a seed but will not find a pose
        that is not already roughly right. Choose by which job you have.
        """
        want = (self.spec.backends.dock or "auto").lower()
        mobile = any(m.redock for m in self.spec.molecules)
        if not mobile:
            if want in ("vina", "builtin"):
                self.notes.append(
                    f"backends.dock is {want!r} but no molecule has 'redock: true', "
                    "so nothing is re-placed. Set redock: true on the partners that "
                    "should move."
                )
            return "none"
        if want == "none":
            return "none"

        from .dock import find_vina

        have_vina = find_vina() is not None
        if want == "vina":
            if not have_vina:
                raise RuntimeError(
                    "backends.dock: vina was requested but no Vina binary was "
                    "found. Install it (see docs/INSTALL.md), or choose "
                    "backends.dock: builtin for IPRO's local rigid-body Monte "
                    "Carlo, or 'none' to keep the input poses."
                )
            return "vina"
        if want == "builtin":
            return "builtin"
        if have_vina:
            return "vina"
        self.notes.append(
            "backends.dock: auto resolved to 'builtin' (IPRO's local rigid-body "
            "Monte Carlo) because no AutoDock Vina binary was found. That refines "
            "a placement rather than searching for one; install Vina, or set "
            "backends.dock explicitly, if a global search is what you need."
        )
        return "builtin"

    def _partner_atoms(self, assembly: str, name: str) -> AtomSet:
        """A partner's atoms in this assembly, with its current pose applied."""
        if name not in self._base_atoms:
            self._base_atoms[name] = _atoms_of(self._partners[name])
        base = self._base_atoms[name]
        pose = self._poses.get((assembly, name))
        return base if pose is None else moved_atomset(base, pose)

    def _assembly_partner_map(self, assembly: str) -> dict[str, AtomSet]:
        a = next(x for x in self._assemblies if x.name == assembly)
        return {
            m.name: self._partner_atoms(assembly, m.name)
            for m in self.spec.assembly_molecules(a)
            if m.name in self._partners
        }

    def _design_atoms(self, chosen: dict[str, Rotamer], assembly: str) -> AtomSet:
        assert self._design_st is not None
        fixed = _atoms_of(self._design_st, {p.label for p in self.spec.positions})
        rots = [
            r for v, r in chosen.items()
            if (split_vpos(v)[0] or assembly) == assembly
        ]
        return fixed.extend(AtomSet.concat([r.atoms for r in rots]))

    def _redock_step(self, chosen: dict[str, Rotamer], iteration: int) -> dict[str, str]:
        """Re-place every mobile partner, in each assembly, and report what moved."""
        report: dict[str, str] = {}
        if self._dock_backend == "none":
            return report
        spec = self.spec
        for a in self._assemblies:
            mobile = {
                m.name for m in spec.assembly_molecules(a)
                if m.redock and m.name in self._partners
            }
            if not mobile:
                continue
            design = self._design_atoms(chosen, a.name)
            partners = self._assembly_partner_map(a.name)
            if self._dock_backend == "builtin":
                results = mc_redock(
                    design, partners, self.backend,
                    moves=spec.loop.redock_moves,
                    sigma_angstrom=spec.loop.redock_sigma_angstrom,
                    sigma_deg=spec.loop.redock_sigma_deg,
                    temperature_start_k=spec.loop.temperature_k,
                    temperature_end_k=spec.loop.redock_temperature_end_k,
                    rng=random.Random(self.rng.randrange(1 << 30)),
                    mobile=mobile,
                )
                for name, res in results.items():
                    # The returned pose is relative to the atoms handed in,
                    # which already carry the current pose. Re-fit from the
                    # ORIGINAL coordinates so poses never compound.
                    base = self._base_atoms[name]
                    current = self._partner_atoms(a.name, name)
                    self._poses[(a.name, name)] = _pose_between(
                        base.coords,
                        res.pose.apply(current.coords),
                        base.coords.mean(axis=0),
                    )
                    report[f"{a.name}/{name}"] = res.describe()
            else:
                report.update(self._redock_with_vina(a.name, mobile, chosen, iteration))
        if report:
            self.run.log(
                f"iteration {iteration}: redocked via {self._dock_backend}: "
                + "; ".join(f"{k} {v}" for k, v in sorted(report.items()))
            )
        return report

    def _redock_with_vina(
        self, assembly: str, mobile: set[str], chosen: dict[str, Rotamer], iteration: int
    ) -> dict[str, str]:
        """Place partners with AutoDock Vina, then re-score with the backend.

        Vina's affinity is on its own scale and is deliberately **never**
        returned into the objective: the *pose* is taken, and the energy that
        drives the loop stays the backend's. Otherwise the objective would
        depend on which docking program happened to be installed, and scores
        would not be comparable between runs.
        """
        from .dock import DockBox, dock
        from .mutate import apply_design

        spec = self.spec
        out: dict[str, str] = {}
        work = self.run.path / "logs" / f"dock_iter{iteration:04d}_{assembly}"
        work.mkdir(parents=True, exist_ok=True)

        receptor = work / "receptor.pdb"
        try:
            apply_design(
                self._design_pdb,
                receptor,
                {
                    (split_vpos(v)[1] or v): (r.aa, r.state)
                    for v, r in chosen.items()
                    if (split_vpos(v)[0] or assembly) == assembly
                },
            )
        except Exception as exc:
            self.notes.append(
                f"Vina redock skipped at iteration {iteration}: could not write the "
                f"receptor ({type(exc).__name__}: {exc})"
            )
            return out

        pts = []
        for p in spec.positions:
            try:
                pts.append(
                    self._design_st.coords(
                        self._design_st.find(p.chain, p.number, p.insertion_code)
                    )
                )
            except Exception:
                continue
        if pts:
            centre = tuple(float(x) for x in np.vstack(pts).mean(axis=0))
        else:
            centre = tuple(float(x) for x in self._design_atoms(chosen, assembly).centroid())
        side = float(spec.loop.vina_box_size)
        box = DockBox(center=centre, size=(side, side, side))

        for name in sorted(mobile):
            mol = spec.molecule(name)
            try:
                res = dock(
                    receptor=receptor, ligand=mol.path, box=box, out_dir=work,
                    exhaustiveness=spec.loop.vina_exhaustiveness,
                    num_modes=spec.loop.vina_num_modes,
                    seed=self.rng.randrange(1 << 30),
                )
            except Exception as exc:
                self.notes.append(
                    f"Vina redock of {name} failed at iteration {iteration} "
                    f"({type(exc).__name__}: {exc}); its placement was left unchanged"
                )
                continue
            placed = self._pose_from_vina(name, res)
            if placed is None:
                continue
            self._poses[(assembly, name)] = placed
            out[f"{assembly}/{name}"] = (
                f"vina best {res.best_score:+.2f} kcal/mol on its own scale "
                f"(objective still uses {self.backend.name}); {placed.describe()}"
            )
        return out

    def _pose_from_vina(self, name: str, res) -> Pose | None:
        """Rigid transform taking the input ligand onto Vina's best pose."""
        if not res.poses:
            return None
        try:
            coords = _read_pose_coords(res.poses[0])
        except Exception as exc:
            self.notes.append(f"could not read Vina's pose for {name}: {exc}")
            return None
        base = self._base_atoms.setdefault(name, _atoms_of(self._partners[name]))
        if len(coords) != len(base.coords):
            # Without a one-to-one atom mapping the transform cannot be fitted,
            # and guessing would move the ligand somewhere nobody chose.
            self.notes.append(
                f"Vina returned {len(coords)} atoms for {name} but the input has "
                f"{len(base.coords)}; the pose was NOT applied. Prepare the ligand "
                "with Meeko so the atom order round-trips."
            )
            return None
        return _pose_between(base.coords, coords, base.coords.mean(axis=0))

    def _refresh_self_energies(self, cb: CombinedTables) -> None:
        """Recompute self energies after the partners have moved.

        Only the **self** energies depend on where the partners are: a pair
        energy is rotamer-against-rotamer and the partners are not in it. So a
        redock invalidates O(positions x rotamers) terms, not the O(n^2) pair
        block, which is what makes re-placing partners every few iterations
        affordable.

        The rotamer *set* is left as it was pruned against the input pose, so
        the design space does not drift between iterations.
        """
        assert self._design_st is not None
        weights = {a.name: a.weight for a in self._assemblies}
        design_labels = {p.label for p in self.spec.positions}
        fixed = _atoms_of(self._design_st, design_labels)
        for a in self._assemblies:
            partners = self._assembly_partner_map(a.name)
            template = fixed.extend(AtomSet.concat(list(partners.values())))
            w = weights[a.name]
            for v in cb.by_assembly.get(a.name, []):
                for i, rot in enumerate(cb.tables.rotamers[v]):
                    cb.tables.self_energy[(v, i)] = w * self.backend.interaction(
                        rot.atoms, template
                    )

    # -- measurement without designing -------------------------------------

    def _tables_for(self, assembly_name: str) -> EnergyTables:
        """Energy tables against one assembly's template."""
        assert self._rotamer_cache is not None
        template = build_template(
            [self._design_st, *self._asm_partners.get(assembly_name, [])],
            [p.label for p in self.spec.positions],
        )
        return build_tables(
            self._rotamer_cache,
            template,
            self.backend,
            max_self_energy=float(self.spec.goal_options.get("max_self_energy", 50.0)),
            pair_eps=float(self.spec.goal_options.get("pair_eps", 1e-3)),
            energy_cutoff=self.spec.loop.energy_cutoff,
            native=self._native,
            reference_energies=self.spec.goal_options.get("reference_energies"),
            strain_weight=float(self.spec.goal_options.get("strain_weight", 0.0)),
        )

    def build_combined(self) -> CombinedTables:
        """Per-assembly tables, folded into one program over virtual positions.

        With a single assembly this returns that assembly's tables untouched, so
        the single-complex case is exactly the problem it was before assemblies
        existed.
        """
        if self._rotamer_cache is None:
            self.prepare()
        per = {a.name: self._tables_for(a.name) for a in self._assemblies}
        self._combined = combine(per, self._assemblies)
        # Positions are virtual once there are several assemblies, so every
        # lookup keyed by a design-position label needs the mapped form.
        self._native_v = {
            v: self._native.get(split_vpos(v)[1], "")
            for v in self._combined.tables.positions
        }
        return self._combined

    def build_tables(self) -> EnergyTables:
        """The combined tables, as one single-assembly problem."""
        return self.build_combined().tables

    def measure_native(self) -> tuple[ScoreContext, EnergyTables]:
        """Score the input as supplied, with the native identity where allowed.

        This is what ``ipro-agent score`` reports: the reference an objective
        may need (``goal_options.reference_interaction``), and a check that the
        molecules are actually in contact.
        """
        tables = self.build_tables()
        choice = self._native_choice(tables)
        return self._context_for(self._rotamers_for(tables, choice)), tables

    # -- the loop ----------------------------------------------------------

    def run_loop(self) -> RunResult:
        spec = self.spec
        cb = self.build_combined()
        tables = cb.tables
        self.run.log(f"MILP tables: {json.dumps({k: int(v) for k, v in tables.stats.items()})}")
        if cb.n_assemblies > 1:
            self.run.log(
                f"assemblies: {cb.n_assemblies}, reference {cb.reference!r}, "
                f"{len(cb.identity_links)} identity link(s)"
            )

        # Geometric restraints become vetoes on rotamer choices, which is exact:
        # the forbidden assignments are precisely those that violate them, so the
        # MILP optimum is the best design that complies. A restraint that cannot
        # be expressed is reported, never dropped in silence.
        if spec.restraints and cb.n_assemblies > 1:
            # A restraint is geometry, so it is measured inside each assembly
            # against that assembly's own partners, and vetoes that assembly's
            # virtual positions.
            merged = RestraintPlan()
            for a in self._assemblies:
                sub = EnergyTables(
                    positions=[split_vpos(v)[1] for v in cb.by_assembly[a.name]],
                    rotamers={
                        split_vpos(v)[1]: tables.rotamers[v]
                        for v in cb.by_assembly[a.name]
                    },
                    self_energy={},
                    pruned_per_position={
                        split_vpos(v)[1]: tables.pruned_per_position.get(v, 0)
                        for v in cb.by_assembly[a.name]
                    },
                )
                one = plan_restraints(
                    spec.restraints, sub, self._design_st, self._asm_partners[a.name]
                )
                merged.rotamer_vetoes |= {
                    (vpos(a.name, p), i) for p, i in one.rotamer_vetoes
                }
                merged.pair_vetoes |= {
                    (vpos(a.name, pi), ri, vpos(a.name, pj), rj)
                    for pi, ri, pj, rj in one.pair_vetoes
                }
                merged.native_violations.extend(
                    f"[{a.name}] {m}" for m in one.native_violations
                )
                merged.unsupported.extend(f"[{a.name}] {m}" for m in one.unsupported)
                merged.notes.extend(f"[{a.name}] {m}" for m in one.notes)
                merged.summary.extend(f"[{a.name}] {m}" for m in one.summary)
                merged.relax_fixed_positions.extend(one.relax_fixed_positions)
            self._restraints = merged
        else:
            self._restraints = plan_restraints(
                spec.restraints, tables, self._design_st,
                self._asm_partners.get(self._assemblies[0].name, []),
            )
        rp = self._restraints
        if spec.restraints:
            for line in rp.summary:
                self.run.log(f"restraint {line}")
            self.notes.extend(rp.summary)
            self.notes.extend(rp.notes)
            for v in rp.native_violations:
                self.notes.append(f"NATIVE VIOLATION: {v}")
            for u in rp.unsupported:
                self.notes.append(f"RESTRAINT NOT ENFORCED: {u}")
            self.run.log(
                f"restraints: {len(rp.rotamer_vetoes)} rotamer veto(es), "
                f"{len(rp.pair_vetoes)} pair veto(es), "
                f"{len(rp.unsupported)} not enforced"
            )
        if not spec.goal_options.get("reference_energies") and not self.backend.can_relax:
            self.notes.append(
                "no amino-acid reference energies were supplied and the backend has "
                "no built-in reference term. Nothing then corrects for "
                "unfolded-state propensity, so identities are decided by packing "
                "alone and the search is biased against large side chains at buried "
                "positions. Set goal_options.reference_energies, or use the "
                "PyRosetta backend (whose ref2015 includes the term), before "
                "treating identities as a design result."
            )

        # -- starting point: the native sequence, repacked --------------------
        start_choice = self._native_choice(tables, vetoed=rp.rotamer_vetoes)
        start_ctx, start_per_asm = self._contexts_for(
            self._rotamers_for(tables, start_choice)
        )
        start_score = self.goal.score(start_ctx)

        best_score = start_score
        # Tracked by identity, not by rotamer index: with backbone perturbation
        # the rotamer sets are rebuilt each iteration, so an index from an
        # earlier iteration would point at a different rotamer (or none).
        best_identities = {
            (split_vpos(p)[1] or p): tables.rotamers[p][i].aa
            for p, i in start_choice.items() if p in cb.sequence_positions
        }
        best_ctx = start_ctx
        # Written up front so a run that never improves on its starting point
        # still leaves coordinates behind; replaced whenever a better design is
        # found. Without this, a run where nothing beats the start produced a
        # score and a sequence but no structure at all.
        best_structures: list[str] = self._write_best_structures(
            self._rotamers_for(tables, start_choice),
            start_score,
            {
                p: aa for p, aa in best_identities.items()
                if self._native.get(p) and aa != self._native[p]
            },
        )
        ensemble: list[dict] = []

        self.run.update_state(
            status=Status.RUNNING,
            mode=self.mode,
            total_iterations=spec.loop.iterations,
            start_score=start_score,
            best_score=start_score,
            best_mutations={},
            backend=self.backend.name,
            goal=self.goal.describe(),
            notes=self.notes,
            started=time.time(),
        )

        current_choice = dict(start_choice)
        current_score = start_score
        current_asm_energy = {k: v.total for k, v in start_per_asm.items()}
        accepted = 0
        t_start = time.time()
        n_done = 0

        ordered = self._ordered_positions(tables, cb.sequence_positions)

        for it in range(1, spec.loop.iterations + 1):
            t0 = time.time()
            n_done = it
            greedy = (it / max(spec.loop.iterations, 1)) <= spec.loop.greedy_fraction

            # (2) perturb the backbone, then rebuild the rotamers and the
            # energy tables on it. A rejected iteration must put the backbone
            # back, or it random-walks regardless of the energy.
            snapshot = None
            pert = None
            if self.mode != "fixed-backbone":
                snapshot = (
                    self._design_st, self._rotamer_cache, cb, tables,
                    dict(self._native_v), rp, ordered,
                )
                try:
                    candidate, pert = self._perturb_once(it)
                    cb = self._rebuild_for_backbone(candidate)
                    tables = cb.tables
                    rp = self._plan_restraints_for(cb, tables)
                    ordered = self._ordered_positions(tables, cb.sequence_positions)
                except Exception as exc:
                    (
                        self._design_st, self._rotamer_cache, cb, tables,
                        self._native_v, rp, ordered,
                    ) = snapshot
                    snapshot = None
                    pert = None
                    if "perturb_failed" not in self._once:
                        self._once.add("perturb_failed")
                        self.notes.append(
                            f"a backbone perturbation could not be rebuilt "
                            f"({type(exc).__name__}: {exc}); that iteration ran on "
                            "the previous backbone"
                        )

            # (1) choose a contiguous window of positions to re-decide; every
            # other position is pinned to its current identity, so an iteration
            # is a local move rather than a fresh global solve.
            if snapshot is not None:
                current_choice = self._rebase_choice(current_choice, tables)
            window = self._choose_window(ordered)
            pins = [
                Constraint(
                    kind="require",
                    positions=[p],
                    residues=[tables.rotamers[p][current_choice[p]].aa],
                    label=f"pin_{p}",
                )
                for p in ordered
                if p not in window
            ]

            # (3) solve the MILP. Pins plus integer cuts can leave a window with
            # no unvisited assignment; that means "try somewhere else", not
            # "stop", so a few other windows are tried before giving up.
            sol = None
            last_exc: Exception | None = None
            for attempt in range(6):
                if attempt:
                    window = self._choose_window(ordered)
                    pins = [
                        Constraint(
                            kind="require",
                            positions=[p],
                            residues=[tables.rotamers[p][current_choice[p]].aa],
                            label=f"pin_{p}",
                        )
                        for p in ordered
                        if p not in window
                    ]
                milp = RotamerMILP(
                    tables,
                    constraints=self.constraints + pins,
                    native=self._native_v,
                    visited=self._visited if spec.loop.integer_cuts else [],
                    rotamer_vetoes=rp.rotamer_vetoes,
                    pair_vetoes=rp.pair_vetoes,
                    identity_links=cb.identity_links,
                    sequence_positions=cb.sequence_positions,
                )
                try:
                    sol = milp.solve(
                        solver=spec.backends.milp_solver,
                        lp_path=(self.run.path / "logs" / f"iter{it:04d}.lp") if it <= 3 else None,
                    )
                    break
                except RuntimeError as exc:
                    last_exc = exc
            if sol is None:
                # Every window tried is exhausted: the reachable design space
                # under these constraints has been enumerated. A clean stop.
                self.run.log(f"iteration {it}: no unvisited assignment found: {last_exc}")
                self.notes.append(
                    f"stopped at iteration {it}: no unvisited assignment remained "
                    f"under the current constraints ({last_exc})"
                )
                n_done = it - 1
                break

            chosen = self._rotamers_for(tables, sol.choices)
            # Virtual view, restricted to the sequence scope: this is what the
            # integer cuts and the sequence-level constraints act on.
            seq_scope = cb.sequence_positions
            identities_v = {
                p: tables.rotamers[p][i].aa
                for p, i in sol.choices.items() if p in seq_scope
            }
            # Real-label view, for mutations, the sequence string and reports.
            identities = {split_vpos(p)[1] or p: aa for p, aa in identities_v.items()}

            if spec.loop.integer_cuts:
                self._visited.append(dict(identities_v))

            # (4) every Nth iteration, re-place the mobile partners. The paper
            # puts this between rotamer selection and the energy evaluation, and
            # only every third iteration because it is the expensive step.
            dock_note = ""
            every = max(0, int(spec.loop.redock_every))
            if self._dock_backend != "none" and every and it % every == 0:
                moved = self._redock_step(chosen, it)
                if moved:
                    # Only self energies depend on where the partners are, so
                    # the O(n^2) pair block is left alone.
                    self._refresh_self_energies(cb)
                    dock_note = f"; redocked {len(moved)} partner(s)"

            # (5) score, in every assembly
            ctx, per_asm = self._contexts_for(chosen)
            score = self.goal.score(ctx)
            delta = score - current_score

            # (6) accept / reject
            hard_cap = max(
                spec.loop.hard_reject_kcal,
                spec.loop.hard_reject_fraction * abs(current_score),
            )
            if delta > hard_cap:
                ok, reason = False, f"worse by >{hard_cap:.1f} (hard reject)"
            else:
                ok, reason = metropolis_accept(
                    delta, spec.loop.kt, self.rng.random(), greedy=greedy
                )
            # A geometric filter discards a candidate before its energy counts
            # (PoreDesigner rejects a pore of the wrong size outright).
            veto = self.goal.filter_reject(ctx)
            if veto:
                ok, reason = False, f"filtered: {veto}"

            # IPRO: "all BAs must pass the selection criteria for the results of
            # the iteration to be retained." Applied as an extra gate on top of
            # the objective test, and only when there is more than one assembly,
            # so the single-complex case behaves exactly as before.
            if ok and len(self._assemblies) > 1:
                for a in self._assemblies:
                    if not a.must_pass:
                        continue
                    prev = current_asm_energy.get(a.name)
                    if prev is None:
                        continue
                    d_h = per_asm[a.name].total - prev
                    passed, _why = metropolis_accept(
                        d_h, spec.loop.kt, self.rng.random(), greedy=greedy
                    )
                    if not passed:
                        ok = False
                        reason = f"assembly {a.name} rejected it (dE={d_h:+.3f})"
                        break

            mutations = {
                p: aa for p, aa in identities.items()
                if self._native.get(p) and aa != self._native[p]
            }
            # Structures are written before the record so the record can name
            # them; appending first would leave the path out of iterations.jsonl.
            structure_rel = ""
            if ok and spec.loop.keep_structures:
                multi = len(self._assemblies) > 1
                for a in self._assemblies:
                    stem = f"iter{it:04d}" + (f"_{a.name}" if multi else "")
                    got = self._write_design(
                        self.run.path / "structures" / f"{stem}.pdb",
                        chosen, a.name,
                        header=self._design_header(score, mutations, a.name),
                    )
                    if got is not None and not structure_rel:
                        structure_rel = str(got.path.relative_to(self.run.path))

            rec = IterationRecord(
                iteration=it,
                score=score,
                accepted=ok,
                mutations=mutations,
                components={
                    **{k: float(v) for k, v in self.goal.components(ctx).items()},
                    **{k: float(v) for k, v in ctx.geometry.items()},
                },
                sequence=self._sequence_string(identities),
                reason=(
                    f"{reason}; redesigned {','.join(window)}{dock_note}"
                    + (f"; perturbed {pert.describe()}" if pert else "")
                ),
                seconds=time.time() - t0,
                structure=structure_rel,
            )
            self.run.append_iteration(rec)

            if not ok and snapshot is not None:
                # Put the backbone, rotamers and tables back as they were.
                (
                    self._design_st, self._rotamer_cache, cb, tables,
                    self._native_v, rp, ordered,
                ) = snapshot
                current_choice = self._rebase_choice(current_choice, tables)

            if ok:
                accepted += 1
                current_choice, current_score = dict(sol.choices), score
                current_asm_energy = {k: v.total for k, v in per_asm.items()}

            if score < best_score:
                best_score, best_ctx = score, ctx
                best_identities = dict(identities)
                self.run.update_state(
                    best_score=best_score,
                    best_mutations=mutations,
                    best_sequence=rec.sequence,
                )
                # Written now, while the coordinates that produced this score
                # still exist: with backbone perturbation they are gone by the
                # next iteration.
                best_structures = self._write_best_structures(
                    chosen, score, mutations
                )
            ensemble.append({
                "iteration": it, "score": score, "mutations": mutations,
                "accepted": ok, "milp_objective": sol.objective,
                "milp_solver": sol.solver, "proven_optimal": sol.proven_optimal,
            })

            if spec.loop.stop_after_accepted and accepted >= spec.loop.stop_after_accepted:
                self.run.log(f"stopping: reached {accepted} accepted designs")
                break
            if spec.loop.max_minutes and (time.time() - t_start) / 60.0 >= spec.loop.max_minutes:
                self.run.log("stopping: wall-clock budget reached")
                self.notes.append("stopped on max_minutes before the iteration cap")
                break

        # -- finish ----------------------------------------------------------
        ensemble.sort(key=lambda d: d["score"])
        ensemble = ensemble[: spec.loop.ensemble_size]
        (self.run.path / "ensemble" / "ensemble.json").write_text(json.dumps(ensemble, indent=2))


        best_mutations = {
            p: aa for p, aa in best_identities.items()
            if self._native.get(p) and aa != self._native[p]
        }
        best_seq = self._sequence_string(best_identities)

        result = RunResult(
            run_dir=self.run.path,
            mode=self.mode,
            best_score=best_score,
            start_score=start_score,
            best_mutations=best_mutations,
            best_sequence=best_seq,
            iterations=n_done,
            accepted=accepted,
            mutation_string=self._mutation_string(best_mutations),
            structures=list(best_structures),
            ensemble=ensemble,
            notes=self.notes,
        )
        structure_files = best_structures
        if structure_files:
            result.notes.append("designed coordinates: " + "; ".join(structure_files))

        (self.run.path / "best" / "design.json").write_text(json.dumps({
            "goal": self.goal.describe(),
            "goal_statement": self.spec.goal_statement,
            "mode": self.mode,
            "backend": self.backend.describe(),
            "start_score": start_score,
            "best_score": best_score,
            "identities": best_identities,
            "mutations": best_mutations,
            "mutation_string": self._mutation_string(best_mutations),
            "sequence": best_seq,
            "components": {k: float(v) for k, v in self.goal.components(best_ctx).items()},
            "structures": structure_files,
            "notes": self.notes,
        }, indent=2))

        self.run.update_state(
            status=Status.DONE,
            best_score=best_score,
            best_mutations=best_mutations,
            best_sequence=best_seq,
            accepted=accepted,
            finished=time.time(),
            notes=self.notes,
        )
        return result

    # -- helpers -----------------------------------------------------------

    def _native_choice(
        self, tables: EnergyTables, vetoed: set[tuple[str, int]] | None = None
    ) -> dict[str, int]:
        """Rotamer indices reproducing the native identity where it is offered.

        Where the native residue is not in the user's allowed set, the
        lowest-self-energy rotamer stands in -- and that is noted, because the
        "starting score" is then not the wild type's.
        """
        vetoed = vetoed or set()
        out: dict[str, int] = {}
        substituted = []
        blocked = []
        for p in tables.positions:
            nat = (self._native_v or self._native).get(p)
            allowed = [
                i for i in range(len(tables.rotamers[p])) if (p, i) not in vetoed
            ] or list(range(len(tables.rotamers[p])))
            idx = next(
                (i for i in allowed if tables.rotamers[p][i].aa == nat), None
            )
            if idx is None:
                idx = min(allowed, key=lambda i: tables.self_energy.get((p, i), 0.0))
                # Distinguish "the user did not allow the native residue" from
                # "a restraint forbade every native rotamer": they mean
                # different things to someone reading the starting score.
                if any(r.aa == nat for r in tables.rotamers[p]):
                    blocked.append(p)
                else:
                    substituted.append(p)
            out[p] = idx
        if blocked:
            real = sorted({split_vpos(p)[1] or p for p in blocked})
            self.notes.append(
                "the restraints forbid every native rotamer at "
                f"{', '.join(real)}, so the starting score is not the wild "
                "type's; the input structure does not satisfy the restraints there"
            )
        if substituted:
            real = sorted({split_vpos(p)[1] or p for p in substituted})
            self.notes.append(
                "the native residue is not an allowed choice at "
                f"{', '.join(real)}, so the reported starting score is for "
                "the best allowed residue there, not the wild type"
            )
        return out

    @staticmethod
    def _rotamers_for(tables: EnergyTables, choices: dict[str, int]) -> dict[str, Rotamer]:
        return {p: tables.rotamers[p][i] for p, i in choices.items()}

    def _ordered_positions(
        self, tables: EnergyTables, scope: set[str] | None = None
    ) -> list[str]:
        """Design positions in sequence order, so "contiguous" means something.

        Restricted to ``scope`` (the reference assembly's virtual positions) so
        that a window names each sequence position once rather than once per
        assembly; identity coupling then carries the choice to the others.
        """
        by_label = {p.label: p for p in self.spec.positions}
        candidates = [
            v for v in tables.positions if scope is None or v in scope
        ]

        def key(v: str):
            lab = split_vpos(v)[1] or v
            dp = by_label.get(lab)
            return (dp.chain if dp else lab, dp.number if dp else 0,
                    dp.insertion_code if dp else "")

        return sorted(candidates, key=key)

    def _choose_window(self, ordered: list[str]) -> list[str]:
        """A run of 1-3 design positions to re-decide, as IPRO does.

        Contiguity is in the *ordered list of design positions*, not in residue
        numbering: design positions are usually scattered through the sequence,
        so insisting on residue-adjacency would almost always give a window of
        one.
        """
        lo, hi = self.spec.loop.design_per_iteration
        k = min(len(ordered), self.rng.randint(lo, hi))
        if k >= len(ordered):
            return list(ordered)
        start = self.rng.randrange(0, len(ordered) - k + 1)
        return ordered[start : start + k]

    def _rebase_choice(
        self, choices: dict[str, int], tables: EnergyTables
    ) -> dict[str, int]:
        """Re-express an assignment against a rebuilt rotamer set.

        Rotamer indices are not comparable across backbones: a rebuild can prune
        differently, so index 3 may be a different residue or may not exist. The
        identity is the part that carries over, so each position is mapped onto
        its cheapest rotamer of the same identity.
        """
        out: dict[str, int] = {}
        for p in tables.positions:
            want = None
            if p in choices and p in tables.rotamers:
                old_list = tables.rotamers[p]
                idx = choices[p]
                if 0 <= idx < len(old_list):
                    want = old_list[idx].aa
            if want is None:
                want = self._native_v.get(p) or None
            candidates = [
                i for i, r in enumerate(tables.rotamers[p]) if r.aa == want
            ] or list(range(len(tables.rotamers[p])))
            out[p] = min(
                candidates, key=lambda i: tables.self_energy.get((p, i), 0.0)
            )
        return out

    def _plan_restraints_for(self, cb: CombinedTables, tables: EnergyTables):
        """Re-plan the restraint vetoes against rebuilt tables."""
        spec = self.spec
        if not spec.restraints:
            return RestraintPlan()
        if cb.n_assemblies > 1:
            merged = RestraintPlan()
            for a in self._assemblies:
                sub = EnergyTables(
                    positions=[split_vpos(v)[1] for v in cb.by_assembly[a.name]],
                    rotamers={
                        split_vpos(v)[1]: tables.rotamers[v]
                        for v in cb.by_assembly[a.name]
                    },
                    self_energy={},
                    pruned_per_position={
                        split_vpos(v)[1]: tables.pruned_per_position.get(v, 0)
                        for v in cb.by_assembly[a.name]
                    },
                )
                one = plan_restraints(
                    spec.restraints, sub, self._design_st, self._asm_partners[a.name]
                )
                merged.rotamer_vetoes |= {
                    (vpos(a.name, p), i) for p, i in one.rotamer_vetoes
                }
                merged.pair_vetoes |= {
                    (vpos(a.name, pi), ri, vpos(a.name, pj), rj)
                    for pi, ri, pj, rj in one.pair_vetoes
                }
            return merged
        return plan_restraints(
            spec.restraints, tables, self._design_st,
            self._asm_partners.get(self._assemblies[0].name, []),
        )

    def _mutation_string(self, mutations: dict[str, str]) -> str:
        """Mutations as ``D115W`` strings, in position order."""
        by_label = {p.label: p for p in self.spec.positions}
        parts = []
        for lab in sorted(mutations, key=lambda l: (
            by_label[l].chain if l in by_label else l,
            by_label[l].number if l in by_label else 0,
        )):
            nat = THREE_TO_ONE.get(self._native.get(lab, ""), "?")
            new = THREE_TO_ONE.get(mutations[lab], mutations[lab])
            num = by_label[lab].number if lab in by_label else lab
            parts.append(f"{nat}{num}{new}")
        return " ".join(parts)

    def _sequence_string(self, identities: dict[str, str]) -> str:
        """The design chain's full sequence with the chosen identities applied."""
        assert self._design_st is not None
        dm = self.spec.design_molecule
        chains = dm.chains or self._design_st.chain_ids()
        out: list[str] = []
        for ch in chains:
            for ref, letter in self._design_st.numbered_sequence(ch):
                aa = identities.get(ref.label)
                out.append(THREE_TO_ONE.get(aa, letter) if aa else letter)
        return "".join(out)


def run_design(
    spec: DesignSpec,
    run_dir: str | Path,
    extra_constraints: Sequence[Constraint] = (),
) -> RunResult:
    """Convenience entry point: prepare and run."""
    loop = IPROLoop(spec, run_dir, extra_constraints=extra_constraints)
    try:
        loop.prepare()
        return loop.run_loop()
    except Exception as exc:
        loop.run.update_state(status=Status.FAILED, error=f"{type(exc).__name__}: {exc}")
        loop.run.log(f"FAILED: {type(exc).__name__}: {exc}")
        raise
