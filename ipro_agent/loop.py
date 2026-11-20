"""The IPRO iteration loop.

Order of operations per iteration, following IPRO's own implementation:

1. perturb the backbone of a short segment (1-5 residues, phi/psi offsets from
   N(0, 1.5 deg) truncated to +/-5 deg) built AROUND a randomly chosen design
   position, as the Suite's step 1 does;
2. the design positions inside that perturbed region are the ones re-decided
   this iteration -- "only DPs within the perturbed region are permitted to
   mutate"; every other position is pinned to its current identity;
3. solve the rotamer-selection MILP over those positions
   (:mod:`ipro_agent.milp`);
4. every third iteration, rigid-body redock the target molecules;
5. relax the chosen design and compute its energy from the relaxed
   coordinates (2006 paper, step iv: the residues just chosen "may no longer
   be optimally interacting with the previous backbone");
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
import os
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import DesignSpec, SpecError, THREE_TO_ONE
from .energy.base import AtomSet, EnergyBackend
from .energy import resolve as resolve_energy
from .goals import Goal, ScoreContext, build_goal, metropolis_accept
from .milp import Constraint, EnergyTables, RotamerMILP, Rotamer, build_tables
from .assemblies import CombinedTables, combine, split_vpos, vpos
from .docking import Pose, _pose_between, moved_atomset
from .docking import redock as mc_redock
from .restraints import RestraintError, RestraintPlan, plan_restraints
from .output import write_atomset_pdb
from .rotamers import build_template, harvest
from .state import IterationRecord, RunDir, Status
from .structure import Structure, is_hydrogen

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


#: Atoms the rigid-body fit is computed from. Backbone only, deliberately: a
#: large rotamer change at a design position must not drag the placement with
#: it, and the backbone is the part that is comparable between the candidate
#: and its relaxed self.
_FIT_ATOMS = frozenset({"N", "CA", "C", "O"})


def _superpose_onto(mobile: AtomSet, reference: AtomSet) -> tuple[AtomSet, float, float]:
    """Kabsch-fit ``mobile`` onto ``reference`` and return it in that frame.

    **Why this exists.** ``backend.relax`` runs FastRelax in torsion space with
    nothing pinning the molecule in the lab frame -- the coordinate
    constraints and the MoveMap are installed only when ``fixed`` is non-empty,
    which it is unless the spec happens to carry a fix/tether restraint. So the
    whole design molecule swings several angstroms while every partner keeps
    its docked coordinates, and ``ctx.interaction[...]`` becomes a cross-term
    between a displaced design and an undisplaced partner.

    Measured with PyRosetta 2026.29 on ``examples/ompf_pore/2omf_A.pdb`` (2627
    heavy atoms), two draws of the SAME structure: mean lab-frame displacement
    11.599 and 6.411 A, centroid shift 10.221 and 5.701 A, while the superposed
    RMSD was only 1.228 and 1.192 A. The conformational change is small; the
    placement is what moved. On a ligand in a buried pocket the interaction
    energy went from -8.588 (unrelaxed) to +5.849 and +4.606 -- two draws of
    one structure disagreeing by 1.24 kcal/mol and carrying the wrong sign.
    Superposed, the same two draws give -0.969 and -1.188, agreeing to 0.22.

    The fit is energy-neutral for :meth:`EnergyBackend.internal` -- a rigid
    motion preserves every internal distance -- and repairs
    :meth:`EnergyBackend.interaction`, which is the term the binding and
    specificity goals optimize. It leaves the relaxation's own work untouched:
    superposed RMSD is unchanged by definition.

    Returns ``(fitted, mean_shift, centroid_shift)``, the last two being the
    rigid-body component that was removed, so the caller can report it.
    """
    if not len(mobile) or not len(reference):
        return mobile, 0.0, 0.0
    ref_at = {(str(k), n): i for i, (k, n) in enumerate(
        zip(reference.reskeys, reference.names))}
    pairs = [
        (i, ref_at[(str(k), n)])
        for i, (k, n) in enumerate(zip(mobile.reskeys, mobile.names))
        if n.strip().upper() in _FIT_ATOMS and (str(k), n) in ref_at
    ]
    if len(pairs) < 3:
        return mobile, 0.0, 0.0
    P = np.asarray(mobile.coords[[i for i, _ in pairs]], dtype=float)
    Q = np.asarray(reference.coords[[j for _, j in pairs]], dtype=float)
    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    H = (P - pc).T @ (Q - qc)
    U, _s, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    moved = (np.asarray(mobile.coords, dtype=float) - pc) @ R.T + qc
    mean_shift = float(np.linalg.norm(moved - mobile.coords, axis=1).mean())
    centroid_shift = float(np.linalg.norm(pc - qc))
    fitted = AtomSet(
        names=list(mobile.names), coords=moved,
        resnames=list(mobile.resnames), reskeys=list(mobile.reskeys),
        elements=list(mobile.elements),
    )
    return fitted, mean_shift, centroid_shift


def _hostname() -> str:
    """Recorded beside the pid: a pid is only meaningful on its own machine."""
    # platform.node(), not socket: the repo's own isolation test forbids
    # importing socket outside the one networked command, and this is a local
    # name lookup rather than anything on a wire.
    try:
        import platform

        return platform.node()
    except Exception:                                     # pragma: no cover
        return ""


def _atoms_of(
    st: Structure,
    skip_sidechains_at: set[str] = frozenset(),
    molecule: str = "",
) -> AtomSet:
    """One structure's heavy atoms as an :class:`AtomSet`.

    ``molecule`` prefixes the residue keys. It must be empty for the design
    molecule (whose keys rotamer atoms have to match) and set for every
    partner: the energy backend skips pairs sharing a residue key, so a partner
    reusing chain id ``A`` with overlapping residue numbers had its real
    contacts with the design silently dropped as "same residue".
    """
    names, coords, resn, keys, els = [], [], [], [], []
    for res in list(st.residues()) + list(st.hetero_residues()):
        ref = st.ref(res)
        drop_sc = ref.label in skip_sidechains_at
        for at in res:
            if is_hydrogen(at):
                continue
            nm = at.get_name().strip()
            if drop_sc and nm not in BACKBONE:
                continue
            names.append(nm)
            coords.append(at.coord)
            resn.append(res.get_resname().strip())
            keys.append(ref.key if not molecule else (molecule, *ref.key))
            els.append((at.element or "").strip().upper())
    return AtomSet(
        names=names,
        coords=np.array(coords) if coords else np.zeros((0, 3)),
        resnames=resn,
        reskeys=keys,
        elements=els if all(els) else [],
    )



def _read_pose_coords(path) -> "np.ndarray":
    """Heavy-atom coordinates from a PDB or PDBQT pose file, in file order."""
    import pathlib as _pl

    coords = []
    for line in _pl.Path(path).read_text(encoding="utf-8").splitlines():
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
        # Qualifications the goal raised travel with the run. The loop SHARES
        # the goal's list rather than copying it: a goal also raises notes while
        # scoring (a term it had to drop, an option it could not apply), and a
        # copy taken here left those out of state.json entirely. Measured, a
        # reshape_channel run whose radius term was never active -- objective
        # identically 0.0 every iteration -- reported a mutation string and a
        # best score with no caveat at all, against the house rule that a
        # design is never reported without its notes.
        goal_notes = getattr(self.goal, "notes", None)
        self.notes: list[str] = goal_notes if isinstance(goal_notes, list) else []
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
        self._input_pdb = None
        self._axis_clear_cache: dict = {}
        self._post_relaxed = 0
        self._post_relax_failed = 0
        #: The starting point's relaxation, counted apart from the iterations
        #: so the run cannot report "3 of 2 iteration(s)".
        self._baseline_relaxed = 0
        self._moved_this_iteration = False
        #: Per assembly, the atoms the last scoring actually measured.
        self._scored_atoms: dict[str, AtomSet] = {}
        #: Latched once, before the baseline is scored: either the whole run
        #: is on the relaxed basis or none of it is. See _relaxed_design_atoms.
        self._relax_disabled = False
        self._relax_failed_now = False
        self._last_cb = None
        self._last_tables = None
        #: Set while a run is in progress; see measure_native.
        self._relax_native = False
        self._partners: dict[str, Structure] = {}
        self._rotamer_cache: dict[str, list[Rotamer]] | None = None
        self._native: dict[str, str] = {}
        self._visited: list[dict[str, str]] = []
        #: Filled in by run_loop once the rotamer tables exist.
        self._restraints: RestraintPlan = RestraintPlan()
        #: How many iterations were relaxed, and how many tried and failed, so
        #: the reported mode can be checked against what actually happened.
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
            # `classify` raises its own notes -- chiefly the SEQRES check, which
            # says when a chain is modelled short of its deposited ends, so that
            # an "append after the last residue" is really an internal edit with
            # an unclosed break. `apply_terminal` takes plan.terminal rather
            # than the plan, so those notes do not travel in `result` and would
            # otherwise be computed and dropped.
            self.notes.extend(plan.notes)
            if result.added:
                self.notes.append(
                    "inserted residues are glycine in the structure; list them as "
                    "design positions to have their identities chosen: "
                    + ", ".join(result.added)
                )
            self.run.log(f"indels: {result.describe()}")

        # The starting structure, kept apart from _design_st because the loop
        # replaces that one with each perturbed candidate. _axis_clearance
        # needs the native side chains (see its docstring).
        self._input_pdb = self._design_pdb

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

        # Two settings that look live and are not. A user who sets them --
        # both shipped examples annotate them "[paper]" -- believes the
        # flanking residues and the 4.5 A shell are being repacked, and they
        # are not: the MILP's variables are exactly the design positions.
        if (spec.loop.perturb_backbone
                and tuple(spec.loop.design_per_iteration) != (1, 3)):
            self.notes.append(
                f"loop.design_per_iteration is set to "
                f"{tuple(spec.loop.design_per_iteration)}, and it is not read "
                f"while the backbone is perturbed: the window is the design "
                f"positions inside the perturbed region, which is how IPRO "
                f"defines it. The setting applies only with "
                f"perturb_backbone: false."
            )
        for name, default in (("perturbation_flank", 5), ("neighbour_cutoff", 4.5)):
            if getattr(spec.loop, name, default) != default:
                self.notes.append(
                    f"loop.{name} is set, but neither flanking-residue "
                    f"repacking nor the 4.5 A neighbour shell is implemented "
                    f"here -- the MILP's variables are exactly your design "
                    f"positions, so the surrounding side chains keep their "
                    f"input rotamers. The setting has no effect "
                    f"(docs/DESIGN.md section 4)."
                )
                break

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
                f"{n} rotamers exceeds max_rotamers="
                f"{self.spec.loop.max_rotamers}. This cap is ADVISORY here: "
                f"nothing tightens the pruning in response, the program is "
                f"written at this size and may be slow. (IPRO's own source "
                f"shrinks its window until the count fits; that loop is not "
                f"implemented.) Lower goal_options.rotamer_window, or design "
                f"fewer positions, if the MILP is too slow."
            )

    # -- scoring -----------------------------------------------------------

    def _contexts_for(
        self, chosen: dict[str, Rotamer], relax_tag: str | None = None
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
            # Step iv: relax what was just chosen, then measure THAT. Falls
            # back to the unrelaxed candidate, with a note, when it cannot.
            if relax_tag is not None:
                relaxed = self._relaxed_design_atoms(
                    design_all, f"{relax_tag}_{a.name}"
                )
                if relaxed is not None:
                    design_all = relaxed
            # The atoms that are about to be measured. _write_design emits
            # THESE, because the delivered structure must be the one the
            # reported numbers belong to: rebuilt through PyMOL from the
            # unrelaxed input, the PDB beside best/design.json measured a
            # 1.521 A bottleneck where the JSON said 1.999994, and the
            # relaxed structure that produced every reported energy existed
            # only under logs/relax.
            self._scored_atoms[a.name] = design_all
            ctx = ScoreContext()
            partners = self.spec.assembly_molecules(a)
            for m in partners:
                if m.name not in self._partners:
                    continue
                ctx.interaction[m.name] = self.backend.interaction(
                    design_all, self._partner_atoms(a.name, m.name)
                )
                # A number this large is not an interaction energy. Each pair
                # term is capped at +/-10, so +1000 means hundreds of atoms
                # overlapping -- a partner that was never docked, or two
                # molecules that are the same coordinates (measured: +149,211
                # for a "partner" pointing at the design molecule's own file
                # and chain). Reported rather than clamped, because the fix is
                # in the spec and no energy from this geometry means anything.
                if (ctx.interaction[m.name] > 1000.0
                        and f"clash_{m.name}" not in self._once):
                    self._once.add(f"clash_{m.name}")
                    self.notes.append(
                        f"the interaction energy with {m.name!r} is "
                        f"{ctx.interaction[m.name]:+.0f}, which is not a "
                        f"physical interaction: it means its atoms overlap the "
                        f"design molecule's. Either the partner was never "
                        f"docked into place, or it is the same coordinates as "
                        f"another molecule in the spec. No energy measured on "
                        f"this geometry can be interpreted -- fix the spec "
                        f"(ipro-agent validate reports the same-coordinates "
                        f"case) and measure again."
                    )
            ctx.design_only = self.backend.internal(design_all)
            ctx.total = ctx.design_only + sum(ctx.interaction.values()) + sum(
                self.backend.internal(self._partner_atoms(a.name, m.name))
                for m in partners if m.name in self._partners
            )
            if getattr(self.goal, "wants_geometry", False):
                ctx.geometry.update(self._measure_geometry(design_all))
            per[a.name] = ctx
            # Usually each partner appears in one assembly, so this is just a
            # copy. When a molecule appears in SEVERAL assemblies it has one
            # interaction energy per assembly, and `update` kept whichever came
            # last -- making the score depend on assembly ordering while
            # `merged.total` summed them all. Accumulate instead, weighted, so
            # the per-molecule view agrees with the total.
            for mol, e in ctx.interaction.items():
                merged.interaction[mol] = (
                    merged.interaction.get(mol, 0.0) + weights[a.name] * e
                )
            merged.design_only += weights[a.name] * ctx.design_only
            merged.total += weights[a.name] * ctx.total
            # The channel is a property of the design molecule, which is shared,
            # so the first assembly's measurement stands for all of them.
            for k, v in ctx.geometry.items():
                merged.geometry.setdefault(k, v)
        return merged, per

    def _context_for(self, chosen: dict[str, Rotamer],
                     relax_tag: str | None = None) -> ScoreContext:
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
        return self._contexts_for(chosen, relax_tag=relax_tag)[0]

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
            length = float(_np.linalg.norm(direction))
            if length < 1e-9:
                raise ValueError(
                    "goal_options.pore_axis names two atoms at the same place, "
                    "so there is no direction to measure along"
                )
            # NORMALISED. Returned raw, this silently deleted the entire pore
            # measurement on exactly the input the docs recommend as the
            # remedy for a bad fitted axis. `_measure_geometry` builds the
            # default window by projecting onto this vector, so with
            # |direction| = 17.43 A for the repo's own example pair
            # (["A18:CA", "A119:CA"]) the window came out 17x too large and
            # landed hundreds of angstroms off the molecule: pore_profile
            # returned zero slices, _measure_geometry returned {}, and
            # PoreDesigner's published sub-4 A filter read None and passed
            # everything. Silently -- the only note path is the exception
            # handler around this function, and nothing raised. The fitted
            # branch below returns a unit vector, which is why it never showed.
            return (pts[0] + pts[1]) / 2.0, direction / length, 0.0

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
        from .pore import pore_profile, vdw_radii

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
            elements=list(design_atoms.elements),
        )
        if not prof.radii:
            return {}
        self._last_pore = prof

        # How far the fitted axis origin sits from the nearest van der Waals
        # surface. The axis is the normal to the plane of the design positions
        # through their centroid, so it follows the channel only when those
        # positions *ring* the lumen; positions along one wall put the origin
        # inside the wall, and the profile then measures the sliver beside the
        # wall instead.
        #
        # Measure this on the INPUT structure, never on a candidate. The first
        # version measured the candidate's own atoms, which conflates two
        # different things: an axis buried in a wall, and a design that has
        # successfully filled the lumen. A narrowing design is *supposed* to
        # reduce the clearance, so the per-candidate number cannot discriminate
        # -- measured that way the sound 25-position spec read 1.30 A against
        # the unsound 4-position spec's 0.30 A, close enough that a threshold
        # rejected both and was reverted as useless (docs/DESIGN.md 7d).
        # Measured on the input structure with its native side chains the two
        # separate by an order of magnitude: 2.80 A against 0.30 A, so a
        # water's 1.40 A is a usable test after all.
        clearance = float(np.min(
            np.sqrt(((design_atoms.coords - origin) ** 2).sum(-1))
            - vdw_radii(list(design_atoms.names), list(design_atoms.elements))
        ))
        axis_ok = self._axis_clearance(origin)
        if axis_ok is not None and axis_ok < 1.4 and "pore_clearance" not in self._once:
            self._once.add("pore_clearance")
            self.notes.append(
                f"THE PORE RADIUS HERE IS NOT A PORE RADIUS. It is measured along "
                f"an axis fitted through the centroid of the "
                f"{len(self.spec.positions)} design position(s), and in the input "
                f"structure that axis origin clears the nearest van der Waals "
                f"surface by only {axis_ok:.2f} A -- less than the 1.40 A a water "
                f"molecule needs, so the origin is inside the protein, not in the "
                f"lumen. The reported {prof.min_radius:.2f} A is the gap beside a "
                f"wall, and any pore_diameter filter is applied to that gap. This "
                f"happens when the design positions do not surround the channel: "
                f"4 positions along one wall of OmpF read 1.53 A, which passes a "
                f"4 A filter, where the 25-position set reads 8.07 A at a "
                f"clearance of 2.80 A. Fix it by giving goal_options.pore_axis as "
                f"two atoms spanning the channel, or by adding positions on the "
                f"opposite wall."
            )
        # How far the narrowest slice's circle sits from the fitted axis. The
        # cap on this (`pore_max_centre_offset`) is a real parameter, and when
        # it binds the reported radius is the cap's answer rather than the
        # pore's -- on OmpF a cap of 4.0 pins the centre at exactly 4.000 and
        # reads 7.68 A, where a non-binding 6.0 reads 8.07 A. Nothing in the
        # record said which had happened, so a reader could not tell a lumen
        # width from a clamped one.
        narrowest = int(np.argmin(prof.radii))
        centre_offset = float(np.linalg.norm(np.cross(
            np.asarray(prof.centres[narrowest]) - np.asarray(prof.axis_origin),
            np.asarray(prof.axis_direction))))
        out = {
            "min_pore_radius": prof.min_radius,
            "min_pore_diameter": prof.min_diameter,
            "max_pore_radius": float(max(prof.radii)),
            "pore_axis_flatness": flatness,
            "pore_axis_clearance": clearance,
            "pore_centre_offset": centre_offset,
            "pore_max_centre_offset": float(opts.get("pore_max_centre_offset", 6.0)),
        }
        if axis_ok is not None:
            out["pore_axis_clearance_input"] = axis_ok
        return out

    def _axis_clearance(self, origin) -> "float | None":
        """Is the fitted axis origin in the lumen, or inside a wall?

        Measured once, on the input structure with its native side chains, for
        the reason given above: on a candidate this number also moves when the
        design works, so it cannot tell a bad axis from a good design.
        """
        from .pore import vdw_radii

        # Keyed by the origin rather than global. The axis is refitted whenever
        # the backbone moves -- `_design_st` is replaced below and `_pore_axis`
        # refits from it -- and a cache that returned before reading `origin`
        # answered every later iteration with iteration 0's axis, so a run with
        # perturb_backbone: true reported `pore_axis_clearance_input` for an
        # axis it had stopped using.
        key = tuple(np.round(np.asarray(origin, dtype=float), 3))
        cache = getattr(self, "_axis_clear_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._axis_clear_cache = cache
        if key in cache:
            return cache[key]

        path = getattr(self, "_input_pdb", None) or self._design_pdb
        if path is None:
            return None
        try:
            dm = self.spec.design_molecule
            st = Structure.load(path, dm.chains or None)
            atoms = [a for a in st.all_atoms() if not is_hydrogen(a)]
            coords = np.array([a.coord for a in atoms])
            names = [a.get_name() for a in atoms]
            elements = [(a.element or "").strip() or None for a in atoms]
            value = float(np.min(
                np.sqrt(((coords - origin) ** 2).sum(-1))
                - vdw_radii(names, elements)
            ))
        except Exception as exc:
            # This file's convention is that a silent degradation gets a note.
            # This returned None on any failure, which skipped the "THE PORE
            # RADIUS HERE IS NOT A PORE RADIUS" warning above with no note, no
            # log line and no trace of why -- and §7m names the fitted axis as
            # one of three suspects for the shipped case accepting nothing, so
            # it is the diagnostic least affordable to lose quietly.
            if "pore_clearance_failed" not in self._once:
                self._once.add("pore_clearance_failed")
                self.notes.append(
                    f"could not check whether the fitted pore axis origin lies "
                    f"in the lumen ({type(exc).__name__}: {exc}). Every pore "
                    f"radius below is reported without that check, so it may be "
                    f"the gap beside a wall rather than a channel width, and any "
                    f"pore_diameter filter is being applied to that gap."
                )
            return None
        cache[key] = value
        return value

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
        work = self.run.path / "logs" / "backbone"
        work.mkdir(parents=True, exist_ok=True)
        candidate = work / f"iter{iteration:04d}.pdb"

        moved = self._design_st.copy()
        # IPRO builds the perturbed region around a design position (Suite
        # step 1, "begins by randomly selecting a DP"); without an anchor the
        # region and the redesign are statistically independent. See
        # perturb.choose_segment.
        #
        # So the design position is drawn FIRST and the chain follows from it.
        # Fixing the chain to `(dm.chains or chain_ids())[0]` and then looking
        # for design positions on it took both failures at once whenever the
        # positions were on any other chain: the list was empty every
        # iteration, so every perturbation was unanchored, AND the only chain
        # ever moved was one with nothing being designed on it. Nothing in
        # config.py ties positions to the first chain, so such a spec
        # validates, and the skill advertises interface affinity and OptCDR,
        # which is exactly where a second chain shows up.
        allowed = list(dm.chains or self._design_st.chain_ids())
        here = [p for p in self.spec.positions if p.chain in allowed]
        if here:
            pick = self.rng.choice(here)
            chain, anchor, anchor_icode = pick.chain, pick.number, pick.insertion_code
        else:
            chain, anchor, anchor_icode = allowed[0], None, ""
            if "perturb_no_positions" not in self._once:
                self._once.add("perturb_no_positions")
                self.notes.append(
                    f"no design position lies on any chain of the design "
                    f"molecule ({', '.join(allowed)}), so every backbone "
                    f"perturbation is unanchored and the move is statistically "
                    f"independent of the redesign."
                )
        record = perturb_backbone(
            moved, chain,
            sigma_deg=self.spec.loop.perturbation_sigma_deg,
            cap_deg=self.spec.loop.perturbation_cap_deg,
            length_range=self.spec.loop.perturbation_segment,
            rng=self.rng,
            anchor=anchor,
            anchor_icode=anchor_icode,
        )
        # The region can still come back without its anchor: a proline anchor,
        # or one alone between chain breaks, falls through to an unanchored
        # draw. That is the 12.4%-overlap case §7h fixed, and until the record
        # carried `anchored` the two were indistinguishable in state.json.
        if record.anchored is False and "perturb_unanchored" not in self._once:
            self._once.add("perturb_unanchored")
            self.notes.append(
                f"at least one iteration perturbed a region that does NOT "
                f"contain its design position (anchor {chain}{anchor}"
                f"{anchor_icode}: proline, or alone between chain breaks), so "
                f"that iteration's backbone move was independent of the "
                f"redesign. `anchored: false` marks them in the record."
            )
        moved.save(candidate)
        # No relaxation here. The published order relaxes once per iteration,
        # AFTER the rotamer choice and the docking (see
        # _relaxed_design_atoms); relaxing the perturbed backbone while it
        # still carries the old side chains was this implementation's own
        # invention, and it meant the backbone never adapted to the mutation.
        return candidate, record

    def _relaxed_design_atoms(self, design_all: AtomSet, tag: str):
        """Published step iv: relax the structure AFTER the sequence is chosen.

        The 2006 paper puts backbone relaxation at step iv, *after* the
        side-chain/sequence optimization, and says why: the residues the MILP
        just chose "may no longer be optimally interacting with the previous
        backbone". The released source runs the same order
        (``Backbone_Perturbation -> Optimal_Rotamers -> Docking -> Relaxation
        -> Calculate_Energy``), and so does docs/DESIGN.md's own table. Only
        the code did not: ``backend.relax`` had exactly one call site, in
        ``_perturb_once``, so the backbone was relaxed while it still carried
        the *old* side chains and then never adapted to the mutation -- the
        whole point of the published step.

        What is relaxed is the design molecule carrying the chosen rotamers,
        written straight from the atoms that were about to be scored, so the
        energies and the coordinates cannot disagree. Partners are **not**
        present during the relaxation: the backend applies coordinate
        constraints, which keeps this a bounded local minimization rather than
        a refold, but it does mean the interface is not re-optimized against
        the partner. That deviation is recorded in docs/DESIGN.md section 4.

        Returns the relaxed atoms, or None when relaxation is unavailable or
        fails -- in which case the caller scores the unrelaxed candidate,
        exactly as before, and the run says so once.
        """
        if self.mode != "full" or not self.backend.can_relax or not len(design_all):
            # `fixed-backbone` says, in the run's own note, that the backbone
            # never moves. It was relaxing anyway when the backend could:
            # measured on the unmodified smoke spec with the PyRosetta
            # backend, the start's relaxed output differed from its input by a
            # mean of 15.7 A CA while the loop's own state never moved, so the
            # only structure that moved was the one being scored -- beside a
            # note promising the opposite.
            return None
        if self._relax_disabled:
            # The run is on the unrelaxed basis and must stay there.
            return None
        work = self.run.path / "logs" / "relax"
        work.mkdir(parents=True, exist_ok=True)
        src = work / f"{tag}_chosen.pdb"
        out = work / f"{tag}_relaxed.pdb"
        try:
            write_atomset_pdb(design_all, src)
            fixed = list(getattr(self._restraints, "relax_fixed_positions", ()) or ())
            self.backend.relax(src, out, fixed=fixed)
            if not out.exists():
                raise RuntimeError("relaxation wrote no file")
            atoms = _atoms_of(Structure.load(out), set())
            if not len(atoms):
                raise RuntimeError("the relaxed file holds no atoms")
            # Put it back in the frame it was scored in. Without this the
            # relaxation's rigid-body component -- measured at a 10 A centroid
            # shift on OmpF -- is read as a conformational change by every
            # term that involves a partner, a probe or a fitted axis, none of
            # which moved. See _superpose_onto.
            atoms, drift, centroid = _superpose_onto(atoms, design_all)
            if drift > 2.0 and "relax_drift" not in self._once:
                self._once.add("relax_drift")
                self.notes.append(
                    f"the relaxation moved the whole design molecule "
                    f"{drift:.1f} A through space (centroid shift "
                    f"{centroid:.1f} A) on top of changing its conformation. "
                    f"That rigid-body component is removed by superposing the "
                    f"relaxed structure back onto the one that was scored, "
                    f"because the partners and the fitted channel axis did not "
                    f"move with it. It is reported because a large value means "
                    f"the backbone is being relaxed with nothing holding it: "
                    f"add a `fix` restraint to pin it, and the relaxation stays "
                    f"local in the first place."
                )
            if len(atoms) != len(design_all) and "relax_atoms" not in self._once:
                self._once.add("relax_atoms")
                self.notes.append(
                    f"the relaxed structure has {len(atoms)} heavy atoms where the "
                    f"candidate had {len(design_all)}; the relaxation backend added "
                    f"or dropped atoms, and the reported energy is for what came "
                    f"back from it"
                )
        except Exception as exc:
            self._post_relax_failed += 1
            self._relax_failed_now = True
            if "post_relax" not in self._once:
                self._once.add("post_relax")
                self.notes.append(
                    f"the relaxation that the protocol runs AFTER choosing the "
                    f"sequence (2006 paper, step iv) failed "
                    f"({type(exc).__name__}: {exc}). A candidate that cannot be "
                    f"relaxed is not measurable on the same basis as the "
                    f"starting point, so that iteration is REJECTED rather than "
                    f"scored unrelaxed and compared against a relaxed baseline "
                    f"-- which previously manufactured a 78.86 kcal/mol "
                    f"'improvement' out of a 0.02 kcal/mol design."
                )
            return None
        if tag.startswith("start") or tag.startswith("native"):
            self._baseline_relaxed += 1
        else:
            self._post_relaxed += 1
        return atoms

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

        # Prefer the atoms that were scored. They already carry the chosen
        # rotamer at each design position -- that is what made them the
        # candidate -- and, in `full` mode, the relaxation that produced every
        # reported energy. Rebuilding through PyMOL from the unrelaxed input
        # instead delivered a structure that was not the one measured:
        # best/design.json said min_pore_radius 1.999994 beside a design.pdb
        # that measures 0.760.
        scored = self._scored_atoms.get(assembly)
        if scored is not None and len(scored):
            try:
                write_atomset_pdb(scored, design_only)
                built = Structure.load(design_only)
                return self._complex_from(path, built, assembly, header)
            except Exception as exc:
                if "scored_write" not in self._once:
                    self._once.add("scored_write")
                    self.notes.append(
                        f"could not write the scored coordinates directly "
                        f"({type(exc).__name__}: {exc}); falling back to "
                        f"rebuilding the design through PyMOL, which may not be "
                        f"the structure the reported energies belong to"
                    )
        try:
            from .mutate import apply_design

            res = apply_design(self._design_pdb, design_only, picks)
            if res.errors:
                # Do NOT fall through and write the file. The PDB carries a
                # REMARK naming the design and its energy, so emitting it with
                # an unapplied position would publish a structure that is the
                # native residue under a header claiming the mutation. No
                # structure is better than a wrong one that looks right.
                if "design_write_failed" not in self._once:
                    self._once.add("design_write_failed")
                    self.notes.append(
                        f"designed coordinates were NOT written: these positions "
                        f"could not be applied or verified -- "
                        f"{json.dumps(res.errors)[:300]}. The sequence and "
                        f"energies are still reported; the structure is omitted "
                        f"rather than written with the native residue in place."
                    )
                return None
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
        return self._complex_from(path, built, assembly, header)

    def _complex_from(self, path, built, assembly: str, header):
        """Write ``built`` plus this assembly's partners, at their poses."""
        from pathlib import Path as _Path

        from .output import write_complex

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
            # units_short, not units: a PDB record is 80 columns and the full
            # sentence made this line 230 characters. The long form is in
            # best/design.json, which has no column limit.
            f"REMARK   1 SCORE      {score:.4f} ({self.backend.units_short})",
            f"REMARK   1 MUTATIONS  {muts}",
        ]
        if not mutations:
            # This file is written up front from the native sequence, repacked,
            # and is only replaced once an ACCEPTED candidate beats it. So a
            # copy with no mutations is the starting point, not a design, and a
            # stray PDB has to say so on its own -- the whole reason this header
            # exists.
            out.append(
                "REMARK   1 WARNING    NO MUTATIONS: this is the input sequence,"
            )
            out.append(
                "REMARK   1 WARNING    repacked. It is NOT a design. See the run's notes."
            )
        if len(self._assemblies) > 1:
            out.append(f"REMARK   1 ASSEMBLY   {assembly}")
        out.append(
            # Keyed on where the pairwise numbers came from, not on whether the
            # backend can relax: the PyRosetta backend can relax and still
            # returns screening energies for the table and the report, so this
            # caveat was being dropped from exactly the file that needed it.
            "REMARK   1 NOTE       energies from this backend are a screening"
            if self.backend.screening_pairwise else
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
            self._base_atoms[name] = _atoms_of(self._partners[name], molecule=name)
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
                    # which already carry the current pose, so it is composed
                    # with that one to stay relative to the ORIGINAL
                    # coordinates. This used to re-fit the composed pose from
                    # coordinates, which threw away exactly what the fit
                    # cannot recover: for a collinear ligand, the rotation
                    # about its own axis. Composition is exact.
                    prior = self._poses.get((a.name, name)) or Pose()
                    self._poses[(a.name, name)] = prior.then(res.pose)
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
        from .dock import DockBox, dock, pdb_to_pdbqt
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

        # Vina reads PDBQT only, and a spec names PDB files -- which is the
        # format every example, every doc and the structure loader use. Without
        # this conversion `backends.dock: vina` could not dock at all: measured
        # with Vina 1.2.7 installed, every redock failed with "receptor
        # receptor.pdb is not a .pdbqt file" and the loop noted the failure and
        # left the placement unchanged, so a run that reported dock -> vina had
        # never moved a partner. `auto` resolves to vina whenever Vina is on
        # PATH, so installing Vina made docking stop rather than start.
        #
        # obabel's conversion is a stopgap and `pdb_to_pdbqt` says why in its
        # own docstring (receptor charges and atom types are not Meeko's); the
        # pose is taken and re-scored with this tool's backend, never with
        # Vina's affinity, which is what makes that acceptable here.
        def _as_pdbqt(path, label: str, receptor_flag: bool):
            p = Path(path)
            if p.suffix.lower() == ".pdbqt":
                return p
            out_path = work / f"{p.stem}_{label}.pdbqt"
            return pdb_to_pdbqt(p, out_path, receptor=receptor_flag)

        try:
            receptor_q = _as_pdbqt(receptor, "receptor", True)
        except Exception as exc:
            if "pdbqt" not in self._once:
                self._once.add("pdbqt")
                self.notes.append(
                    f"Vina redock skipped: the receptor could not be converted to "
                    f"PDBQT ({type(exc).__name__}: {exc})"
                )
            return out

        for name in sorted(mobile):
            mol = spec.molecule(name)
            try:
                ligand_q = _as_pdbqt(mol.path, f"lig_{name}", False)
                res = dock(
                    receptor=receptor_q, ligand=ligand_q, box=box, out_dir=work,
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
        base = self._base_atoms.setdefault(
            name, _atoms_of(self._partners[name], molecule=name)
        )
        if len(coords) != len(base.coords):
            # Without a one-to-one atom mapping the transform cannot be fitted,
            # and guessing would move the ligand somewhere nobody chose.
            self.notes.append(
                f"Vina returned {len(coords)} atoms for {name} but the input has "
                f"{len(base.coords)}; the pose was NOT applied. Prepare the ligand "
                "with Meeko so the atom order round-trips."
            )
            return None
        return _pose_between(
            base.coords, coords, base.coords.mean(axis=0), notes=self.notes
        )

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
        for a in self._assemblies:
            template = self._template_for(a.name)
            w = weights[a.name]
            for v in cb.by_assembly.get(a.name, []):
                for i, rot in enumerate(cb.tables.rotamers[v]):
                    cb.tables.self_energy[(v, i)] = w * self._self_energy(rot, template)

    def _self_energy(self, rot: Rotamer, template: AtomSet) -> float:
        """One rotamer's self energy, by the same formula ``build_tables`` uses.

        It has to be the same formula. When the refresh recomputed only the
        interaction term, every redock silently deleted the strain and
        reference-energy contributions that the initial build had added, so
        identities were ranked on one scale before the first redock and
        another after it -- a sequence could change with no physical cause.
        """
        strain_weight, refs = self._self_energy_terms()
        e = self.backend.interaction(rot.atoms, template)
        if strain_weight and rot.strain is not None:
            e += strain_weight * rot.strain
        return e + refs.get(rot.aa, 0.0)

    def _self_energy_terms(self) -> tuple[float, dict[str, float]]:
        return (
            float(self.spec.goal_options.get("strain_weight", 0.0)),
            self.spec.goal_options.get("reference_energies") or {},
        )

    def _template_for(self, assembly_name: str) -> AtomSet:
        """Everything held fixed for one assembly, partners at their CURRENT pose.

        One helper for both the initial table build and the post-redock
        refresh. When those two computed the template separately, the build
        read the input ``Structure`` objects and the refresh read ``_poses``,
        so after the first redock the MILP was still optimizing against a
        partner position the docking step had abandoned.

        The design structure goes through ``build_template`` so that design
        side chains are excluded. Partner atoms are added whole and never
        label-filtered: a partner that reuses the design's chain id must not
        lose its own side chain to a design-position label match.
        """
        assert self._design_st is not None
        fixed = build_template(
            [self._design_st], [p.label for p in self.spec.positions]
        )
        partners = self._assembly_partner_map(assembly_name)
        return fixed.extend(AtomSet.concat(list(partners.values())))

    # -- measurement without designing -------------------------------------

    def _tables_for(
        self,
        assembly_name: str,
        restrict_to: dict[str, set[tuple[str, int]]] | None = None,
    ) -> EnergyTables:
        """Energy tables against one assembly's template.

        ``restrict_to`` pins the candidate set to a given list of ``(aa,
        state)`` per position and switches pruning off, so that every assembly
        ends up with the *same* rotamers in the same order. See
        :meth:`build_combined`.
        """
        assert self._rotamer_cache is not None
        template = self._template_for(assembly_name)
        rots = self._rotamer_cache
        if restrict_to is not None:
            rots = {
                p: [r for r in rots.get(p, []) if (r.aa, r.state) in keep]
                for p, keep in restrict_to.items()
            }
        return build_tables(
            rots,
            template,
            self.backend,
            # IPRO prunes relative to the best rotamer at each position
            # (window 40 kcal/mol); an absolute ceiling is optional and off by
            # default, because as a default of 50 it removed TRP/PHE/TYR from
            # OmpF A18 and with them the published answer.
            keep_per_identity=int(
                self.spec.goal_options.get("keep_per_identity", 3)
            ),
            rotamer_window=(
                float("inf") if restrict_to is not None
                else float(self.spec.goal_options.get("rotamer_window", 40.0))
            ),
            max_self_energy=(
                None if restrict_to is not None
                else (
                    float(self.spec.goal_options["max_self_energy"])
                    if self.spec.goal_options.get("max_self_energy") is not None
                    else None
                )
            ),
            pair_eps=float(self.spec.goal_options.get("pair_eps", 1e-3)),
            energy_cutoff=self.spec.loop.energy_cutoff,
            native=self._native,
            reference_energies=self._self_energy_terms()[1],
            strain_weight=self._self_energy_terms()[0],
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
        if len(self._assemblies) > 1:
            # Each assembly pruned against its OWN partners, so a rotamer that
            # clashes in one assembly was dropped only there -- and `combine`
            # then refused the whole problem as internally inconsistent. That
            # hit exactly the case assemblies exist for: a rotamer that clashes
            # with the ANTI-target is the point, not a reason to discard it.
            #
            # So the surviving sets are unioned and every assembly is rebuilt
            # over that common set with pruning off. One rotamer list, one
            # index meaning the same conformer everywhere, per-assembly self
            # energies carrying the difference between the partners.
            union: dict[str, set[tuple[str, int]]] = {}
            for t in per.values():
                for pos, rl in t.rotamers.items():
                    union.setdefault(pos, set()).update((r.aa, r.state) for r in rl)
            dropped = sum(
                len(self._rotamer_cache.get(p, [])) - len(k) for p, k in union.items()
            )
            per = {
                a.name: self._tables_for(a.name, restrict_to=union)
                for a in self._assemblies
            }
            self.run.log(
                f"multi-assembly: {sum(len(v) for v in union.values())} rotamers "
                f"shared across {len(self._assemblies)} assemblies "
                f"({dropped} pruned in every assembly)"
            )
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
        # `score` is where a user looks before spending CPU, so the
        # unreachable-identity note belongs here too, not only in a run.
        self._note_dropped_identities(tables)
        choice = self._native_choice(tables)
        # `relax_native` exists because `ipro-agent score` is also used on its
        # own, to report a structure as supplied; there the relaxation would
        # change the number the user asked for. Inside a run it is always on,
        # so the reference and the candidates are measured the same way.
        tag = "native" if self._relax_native else None
        return self._context_for(self._rotamers_for(tables, choice), relax_tag=tag), tables

    # -- the loop ----------------------------------------------------------

    def run_loop(self) -> RunResult:
        spec = self.spec
        if not spec.positions and not spec.indels:
            # An empty `positions:` list ran to completion and reported
            # "ok: true, accepted: 3, mutations: {}" with no note, which reads
            # as "the optimizer found no improvement" rather than "nothing was
            # designed". Positions may legitimately arrive from an insertion
            # (section 1i), so indels count.
            raise SpecError(
                "the spec names no design positions and no indels, so there is "
                "nothing to design. Add positions:, or indels: that create some. "
                "('ipro-agent score' measures a structure without designing.)"
            )
        cb = self.build_combined()
        tables = cb.tables
        self.run.log(f"MILP tables: {json.dumps({k: int(v) for k, v in tables.stats.items()})}")
        self._note_dropped_identities(tables)
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
        # `screening_pairwise`, not `can_relax`: ref2015's `ref` term only
        # reaches the design if Rosetta is actually scoring the pair table,
        # which by default it is not -- so the advice at the end of this note
        # was being suppressed in the one case where it was still needed.
        if not spec.goal_options.get("reference_energies") and self.backend.screening_pairwise:
            self.notes.append(
                "no amino-acid reference energies were supplied and the backend "
                "has no built-in reference term, so nothing corrects for "
                "unfolded-state propensity and identities are decided by packing "
                "alone -- which on a fixed backbone disfavours large side chains "
                "at buried positions. This matters most when the objective IS the "
                "packing; it matters less when optimizing an interaction energy "
                "with a target. Set goal_options.reference_energies, or use the "
                "PyRosetta backend (ref2015 includes the term)."
            )

        # -- starting point: the native sequence, repacked --------------------
        # Relaxed through the same path as every candidate. Relaxing candidates
        # against an unrelaxed reference would credit each one with the
        # relaxation's own energy drop and accept nearly everything.
        self._relax_native = True
        start_choice = self._native_choice(tables, vetoed=rp.rotamer_vetoes)
        self._relax_failed_now = False
        start_ctx, start_per_asm = self._contexts_for(
            self._rotamers_for(tables, start_choice), relax_tag="start"
        )
        # The basis is latched HERE and never changes again. If the baseline
        # could not be relaxed, the whole run goes on the unrelaxed basis and
        # the baseline is re-scored on it -- this is the only moment at which
        # the basis can change coherently, because nothing has been accepted
        # yet. Afterwards a candidate that cannot be relaxed is rejected, not
        # re-based: scoring an unrelaxed candidate against a relaxed baseline
        # manufactured a 78.86 kcal/mol "improvement" out of a design worth
        # 0.02, and the mirror case rejected everything.
        if self._relax_failed_now and self.backend.can_relax:
            self._relax_disabled = True
            self._relax_failed_now = False
            self.notes.append(
                "the starting point could not be relaxed, so this whole run is "
                "on the UNRELAXED basis -- candidates are scored the same way "
                "the baseline was, which is what makes the comparison mean "
                "anything. The protocol is 'backbone-mc' in effect, not 'full'."
            )
            self.mode = "backbone-mc"
            start_ctx, start_per_asm = self._contexts_for(
                self._rotamers_for(tables, start_choice), relax_tag=None
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
            # So a status call can tell "running" from "died hours ago".
            pid=os.getpid(),
            host=_hostname(),
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
        current_aa = self._identities(current_choice, tables)
        current_score = start_score
        #: The complex energy of the accepted structure, which the hard-reject
        #: guard is measured against (see step 6).
        current_complex = start_ctx.total
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
                # _design_pdb belongs in here with _design_st: a rejected
                # perturbation that restored only the Structure left the file
                # on disk pointing at the rejected backbone, so scoring read
                # the restored coordinates while the written PDB and Vina read
                # the rejected ones.
                snapshot = (
                    self._design_st, self._design_pdb, self._rotamer_cache,
                    cb, tables, dict(self._native_v), rp, ordered,
                )
                try:
                    candidate, pert = self._perturb_once(it)
                    cb = self._rebuild_for_backbone(candidate)
                    tables = cb.tables
                    # The note was only ever raised before the loop started,
                    # so an identity that becomes unreachable once the
                    # backbone has moved -- the case a perturbing run exists
                    # to explore -- was never reported at all. `_once` keeps
                    # it to one note however many iterations rebuild.
                    self._note_dropped_identities(tables)
                    rp = self._plan_restraints_for(cb, tables)
                    ordered = self._ordered_positions(tables, cb.sequence_positions)
                except Exception as exc:
                    (
                        self._design_st, self._design_pdb, self._rotamer_cache,
                        cb, tables, self._native_v, rp, ordered,
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
                current_choice = self._rebase_choice(current_aa, tables, rp)
            window = self._choose_window(ordered, pert)
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
                if snapshot is not None:
                    # Leaving here with the perturbation still applied would
                    # make the final report describe a backbone no iteration
                    # ever accepted.
                    (
                        self._design_st, self._design_pdb, self._rotamer_cache,
                        cb, tables, self._native_v, rp, ordered,
                    ) = snapshot
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
            self._moved_this_iteration = False
            self._last_cb, self._last_tables = cb, tables
            every = max(0, int(spec.loop.redock_every))
            if self._dock_backend != "none" and every and it % every == 0:
                moved = self._redock_step(chosen, it)
                if moved:
                    self._moved_this_iteration = True
                    # Only self energies depend on where the partners are, so
                    # the O(n^2) pair block is left alone.
                    self._refresh_self_energies(cb)
                    dock_note = f"; redocked {len(moved)} partner(s)"

            # (5) relax what was chosen, then score it, in every assembly
            self._relax_failed_now = False
            ctx, per_asm = self._contexts_for(chosen, relax_tag=f"iter{it:04d}")
            relax_failed = self._relax_failed_now and not self._relax_disabled
            score = self.goal.score(ctx)
            delta = score - current_score

            # (6) accept / reject
            #
            # Two tests on two different quantities, as the released source
            # has them (`standard_decision`): the Metropolis window applies to
            # the OBJECTIVE, while the max(40 kcal/mol, 5%) cap applies to the
            # COMPLEX ENERGY -- `Energies[group]['Complex'] > complex + value`
            # -- which is what makes it a stability safeguard rather than a
            # second opinion about the objective. Applied to the objective, as
            # it was here, a goal that is not an energy (a pore radius, a
            # specificity difference) got a cap in the wrong units and the
            # stability guard had no default counterpart at all: a design could
            # destabilize the protein by any amount as long as the objective
            # improved.
            hard_cap = max(
                spec.loop.hard_reject_kcal,
                spec.loop.hard_reject_fraction * abs(current_complex),
            )
            d_complex = ctx.total - current_complex
            if d_complex > hard_cap:
                ok, reason = (
                    False,
                    f"complex energy worse by {d_complex:+.1f} > "
                    f"{hard_cap:.1f} (hard reject)",
                )
            else:
                ok, reason = metropolis_accept(
                    delta, spec.loop.kt, self.rng.random(), greedy=greedy
                )
            # A geometric filter discards a candidate before its energy counts
            # (PoreDesigner rejects a pore of the wrong size outright).
            veto = self.goal.filter_reject(ctx)
            if veto:
                ok, reason = False, f"filtered: {veto}"

            # A restraint whose atoms are all OUTSIDE the design positions --
            # the catalytic-distance restraints the ACS Catal 2017 thioesterase
            # work used, held to +/-0.2 A -- cannot be expressed as a rotamer
            # veto, so the planner reported "nothing to decide" and nothing
            # enforced it. But the perturbation moves the enzyme's own atoms
            # and the redock moves the partner's: measured, translating the
            # partner 1.0 A (5x the paper's tolerance, far less than the drift
            # of a few hundred redock moves at sigma 0.2) turned a satisfied
            # restraint into a violated one with rotamer_vetoes still empty and
            # is_empty() still True. Re-measured here, where it can still
            # reject the iteration.
            if relax_failed:
                # Not measurable on the run's basis, so it is not a candidate.
                ok, reason = False, "relaxation failed; not comparable"

            broke = self._broken_geometry_restraints(rp)
            if broke:
                ok, reason = False, f"restraint violated: {broke[0]}"
                if "geom_restraint" not in self._once:
                    self._once.add("geom_restraint")
                    self.notes.append(
                        f"a restraint with no design-position atom became "
                        f"violated during the run and the iteration was "
                        f"rejected -- {broke[0]}. These cannot be expressed as "
                        f"rotamer vetoes, so they are enforced by rejection "
                        f"instead; the perturbation and the redock both move "
                        f"them."
                    )

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
                    self._design_st, self._design_pdb, self._rotamer_cache,
                    cb, tables, self._native_v, rp, ordered,
                ) = snapshot
                current_choice = self._rebase_choice(current_aa, tables, rp)

            if ok:
                accepted += 1
                current_choice, current_score = dict(sol.choices), score
                current_complex = ctx.total
                current_aa = self._identities(current_choice, tables)
                current_asm_energy = {k: v.total for k, v in per_asm.items()}

            # `ok and`, not just the score. A candidate the goal's hard filter
            # vetoed is not a design at all -- PoreDesigner discards a pore of
            # the wrong size *before* its energy counts -- yet this tracker was
            # a sibling of `if ok:` rather than nested in it, so a rejected
            # candidate still overwrote best_score, best_mutations and
            # best_sequence in state.json and still had best/design.pdb written
            # for it. Measured on this repo's own full example: 14 iterations,
            # **0 accepted**, every one rejected with "bottleneck 7.12 A is
            # above pore_diameter_max 4.00 A" -- and `status` still printed a
            # 14-mutation best design at 0.7297 against a start of 1.2574. An
            # agent following SKILL.md's checklist ("the score improved; if
            # not, say the search found nothing") would have reported those 14
            # mutations as a pore-narrowing design when nothing met the
            # constraint.
            if ok and score < best_score:
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
                # Say so in the notes, as the max_minutes branch below does.
                # Without this the report showed "3/8 iterations" and gave no
                # reason, which reads like a crash rather than a setting.
                self.notes.append(
                    f"stopped after {accepted} accepted design(s) because "
                    f"loop.stop_after_accepted is {spec.loop.stop_after_accepted}; "
                    f"the iteration cap of {spec.loop.iterations} was not reached"
                )
                break
            if spec.loop.max_minutes and (time.time() - t_start) / 60.0 >= spec.loop.max_minutes:
                self.run.log("stopping: wall-clock budget reached")
                self.notes.append("stopped on max_minutes before the iteration cap")
                break

        # -- finish ----------------------------------------------------------
        if accepted == 0:
            # Said plainly, because the numbers alone read as a result: with no
            # accepted iteration the "best" design IS the starting point, and
            # best_score == start_score. SKILL.md tells an agent to report the
            # mutation string and the score change, so a run that designed
            # nothing has to announce that itself rather than leave an agent to
            # infer it from accepted: 0.
            self.notes.append(
                f"NO DESIGN WAS ACCEPTED: {n_done} iteration(s) ran and every one "
                f"was rejected, so the reported best design is the starting "
                f"point, not a redesign, and the score did not improve. Report "
                f"this as 'the search found nothing', and read the per-iteration "
                f"'reason' fields (ipro-agent status) to see what rejected them "
                f"-- a goal filter rejecting everything usually means the filter "
                f"window or the design positions need changing, not more iterations."
            )
        # The mode is a claim about what the protocol did, and it is reported
        # with every design, so it has to match what happened. "full" means
        # relaxation ran; if every attempt failed, say "backbone-mc".
        if self.mode == "full":
            if self._post_relaxed:
                per_iter = max(1, len(self._assemblies))
                ran_on = self._post_relaxed // per_iter
                self.notes.append(
                    f"backbone relaxation ran after the sequence choice on "
                    f"{ran_on} of {n_done} iteration(s)"
                    + (f" ({self._post_relaxed} relaxations, one per assembly)"
                       if per_iter > 1 else "")
                    + (f", failing on {self._post_relax_failed}"
                       if self._post_relax_failed else "")
                    + (" (the starting point was relaxed the same way, so the "
                       "comparison is like-for-like)"
                       if self._baseline_relaxed else "")
                )
            else:
                self.mode = "backbone-mc"
                self.notes.append(
                    "mode reported as 'backbone-mc', not 'full': the backend can "
                    "relax but no iteration was relaxed"
                    + (f" ({self._post_relax_failed} attempt(s) failed)"
                       if self._post_relax_failed else "")
                )
        ensemble.sort(key=lambda d: d["score"])
        ensemble = ensemble[: spec.loop.ensemble_size]
        (self.run.path / "ensemble" / "ensemble.json").write_text(json.dumps(ensemble, indent=2), encoding="utf-8")


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
        }, indent=2), encoding="utf-8")

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

    def _note_dropped_identities(self, tables) -> None:
        """Say when the design space is smaller than the spec advertises.

        An identity whose every rotamer is pruned is not reachable, but
        `validate` still reports the full product of the allowed sets --
        measured on the shipped smoke spec, 6561 sequences while TRP was
        absent from A18 entirely. That is the shape of defect this repo exists
        to avoid: it runs clean and the answer is drawn from a smaller space
        than the user asked for.
        """
        dropped = getattr(tables, "dropped_identities", None) or {}
        if not dropped or "dropped_ids" in self._once:
            return
        self._once.add("dropped_ids")
        detail = "; ".join(
            f"{split_vpos(p)[1] or p}: {', '.join(aas)}"
            for p, aas in sorted(dropped.items())
        )
        self.notes.append(
            f"these allowed residues are NOT reachable, because every rotamer "
            f"of them clashes with the current template beyond the "
            f"{self.spec.goal_options.get('rotamer_window', 40.0)} kcal/mol "
            f"pruning window -- {detail}. The design space is therefore smaller "
            f"than the spec advertises. This is measured against the side "
            f"chains that are fixed right now, so a large residue can be "
            f"excluded at a tight position even when the published design uses "
            f"it there (OmpF A18/TRP is exactly that case: its best rotamer is "
            f"51.9 kcal/mol above the best at that position). Raise "
            f"goal_options.rotamer_window to keep them, and expect the clash to "
            f"show up in the energy instead."
        )

    def _choose_window(self, ordered: list[str], pert=None) -> list[str]:
        """Which design positions get re-decided this iteration.

        With a perturbation, IPRO's answer is not a draw at all: the region is
        built around a design position and then "only DPs within the perturbed
        region are permitted to mutate" (Suite, step 1; 2006 paper, Fig. 5).
        So the window *is* the design positions inside the perturbed region,
        which is what couples the backbone move to the sequence move.

        Without one -- ``loop.perturb_backbone: false``, or a perturbation that
        failed -- there is no region to read, and the window falls back to a
        contiguous run of ``design_per_iteration`` positions. That fallback is
        this repo's choice, not the papers': neither states a 1-3 rule, and the
        released source has no mutation-count draw. Contiguity there is in the
        *ordered list of design positions*, not in residue numbering, since
        design positions are usually scattered and residue-adjacency would
        almost always give a window of one.
        """
        if pert is not None and getattr(pert, "residues", None):
            # Match on (number, insertion code), not the bare number. With
            # insertion codes present -- antibody numbering, which is in the
            # advertised scope -- a bare number counts A52 as inside a region
            # that in fact moved A52B. `keys` is absent only on a record from
            # an older run being resumed, so fall back to numbers then.
            chain = getattr(pert, "chain", None)
            keys = getattr(pert, "keys", None)
            if keys:
                moved = {(int(n), (c or "").strip()) for n, c in keys}
                def _inside(p) -> bool:
                    return (p.number, (p.insertion_code or "").strip()) in moved
            else:
                numbers = set(pert.residues)
                def _inside(p) -> bool:
                    return p.number in numbers
            inside_labels = {
                p.label for p in self.spec.positions
                if _inside(p) and (chain is None or p.chain == chain)
            }
            # `ordered` holds VIRTUAL positions once there is more than one
            # assembly ("a::A16"), while spec.positions hold plain labels
            # ("A16"), so a plain-label intersection was always empty and
            # every iteration fell back to the independent draw -- silently,
            # and only in the multi-assembly case. Measured on a two-assembly
            # spec: ordered ['a::A16', 'a::A17'], a region holding both design
            # positions, derived window [].
            inside = [p for p in ordered
                      if (split_vpos(p)[1] or p) in inside_labels]
            if inside:
                return inside
            # An anchored region always holds its design position, so this is
            # reachable only when the anchor was a proline or the position was
            # dropped from the tables. Fall back rather than skip the iteration.
        lo, hi = self.spec.loop.design_per_iteration
        k = min(len(ordered), self.rng.randint(lo, hi))
        if k >= len(ordered):
            return list(ordered)
        start = self.rng.randrange(0, len(ordered) - k + 1)
        return ordered[start : start + k]

    @staticmethod
    def _identities(choices: dict[str, int], tables: EnergyTables) -> dict[str, str]:
        """The amino acids an index assignment stands for, read against its own tables."""
        out: dict[str, str] = {}
        for p, i in choices.items():
            rots = tables.rotamers.get(p) or []
            if 0 <= i < len(rots):
                out[p] = rots[i].aa
        return out

    def _rebase_choice(
        self,
        identities: dict[str, str],
        tables: EnergyTables,
        plan: "RestraintPlan | None" = None,
    ) -> dict[str, int]:
        """Express a set of identities as indices into a rebuilt rotamer set.

        Rotamer indices are not comparable across backbones: a rebuild can
        prune differently, so index 3 may be a different residue or may not
        exist. So the caller carries *identities* and this maps each onto its
        cheapest rotamer of the same identity in the new tables.

        It used to take the old index map and read it out of the NEW tables --
        exactly the mistake the paragraph above warns about. A position whose
        current choice was TRP could come back as SER with no energy test
        applied, and since these indices then become ``require`` pins, the
        rest of the run was pinned to a residue nothing had chosen.

        ``plan`` excludes restraint-vetoed rotamers. Pinning a vetoed rotamer
        makes the next program infeasible, which the loop reports as "design
        space enumerated" -- a silent early stop with a plausible message.
        """
        vetoed = plan.rotamer_vetoes if plan is not None else set()
        out: dict[str, int] = {}
        for p in tables.positions:
            rots = tables.rotamers.get(p) or []
            if not rots:
                continue
            allowed = [i for i in range(len(rots)) if (p, i) not in vetoed]
            if not allowed:
                # Every rotamer here is vetoed; the program is infeasible with
                # or without a pin, so leave the cheapest and let the solver
                # report it rather than inventing a feasible-looking pin.
                allowed = list(range(len(rots)))
            want = identities.get(p) or self._native_v.get(p) or None
            candidates = [i for i in allowed if rots[i].aa == want] or allowed
            out[p] = min(
                candidates, key=lambda i: tables.self_energy.get((p, i), 0.0)
            )
        return out

    def _broken_geometry_restraints(self, rp) -> list[str]:
        """Restraints no rotamer can affect, re-measured against *now*.

        The plan is rebuilt after every perturbation, so its
        ``native_violations`` already reflect the moved backbone; the redock
        happens afterwards, so this re-plans when a partner has moved this
        iteration. Returns readable violation lines, newest measurement.
        """
        if not self.spec.restraints:
            return []
        plan = rp
        if self._moved_this_iteration:
            try:
                plan = self._plan_restraints_for(self._last_cb, self._last_tables)
            except Exception:
                return []          # a re-plan failure must not fail the run
        if not getattr(plan, "geometry_only", None):
            return []
        return list(plan.native_violations)

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
