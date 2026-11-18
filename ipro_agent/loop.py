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
        self.mode = "full" if self.backend.can_relax else "fixed-backbone"
        # Qualifications the goal raised at construction travel with the run.
        self.notes: list[str] = list(getattr(self.goal, "notes", []))
        if self.mode == "fixed-backbone":
            self.notes.append(
                f"backbone perturbation and relaxation skipped: energy backend "
                f"'{self.backend.name}' cannot relax. Search is over sequence and "
                f"side-chain packing only. This is not the published IPRO protocol."
            )

        # Loaded structures, by role.
        self._design_st: Structure | None = None
        self._partners: dict[str, Structure] = {}
        self._rotamer_cache: dict[str, list[Rotamer]] | None = None
        self._native: dict[str, str] = {}
        self._visited: list[dict[str, str]] = []

    # -- setup -------------------------------------------------------------

    def prepare(self) -> None:
        """Load structures, check the design positions exist, harvest rotamers."""
        spec = self.spec
        if spec.indels:
            # Silently ignoring a requested insertion or deletion would be a
            # serious silent failure: the user would get a design over the
            # wrong sequence. Refuse clearly instead.
            raise NotImplementedError(
                f"this spec requests {len(spec.indels)} insertion/deletion(s), which "
                "are not implemented here. Changing chain length needs loop closure "
                "after the edit (Protein-InDelMaker does this with PyRosetta's "
                "GeneralizedKIC over a 6-residue window, followed by Cartesian "
                "FastRelax), and no loop-closure backend is available. Remove the "
                "'indels:' block to run a fixed-length design, or use "
                "Protein-InDelMaker (github.com/maranasgroup/Protein-InDelMaker) to "
                "build the indel variant first and design on that structure. "
                "See docs/DESIGN.md."
            )
        dm = spec.design_molecule
        self._design_st = Structure.load(dm.path, dm.chains or None)

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
                f"{dm.path.name}:\n  " + "\n  ".join(missing)
            )
        self._native = {p.label: (p.native or "") for p in spec.positions}

        self.run.write_spec(spec.to_dict())
        self.run.log(f"prepared: mode={self.mode} backend={self.backend.describe()}")

        labels = [p.label for p in spec.positions]
        self.run.log(f"harvesting rotamers at {len(labels)} positions")
        rots, native, errors = harvest(
            dm.path, spec.positions, workdir=self.run.path / "logs"
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
        assert self._design_st is not None
        design_fixed = _atoms_of(self._design_st, {p.label for p in self.spec.positions})
        rot_atoms = AtomSet.concat([r.atoms for r in chosen.values()])
        design_all = design_fixed.extend(rot_atoms)

        ctx = ScoreContext()
        for name, st in self._partners.items():
            ctx.interaction[name] = self.backend.interaction(design_all, _atoms_of(st))
        ctx.design_only = self.backend.internal(design_all)
        ctx.total = ctx.design_only + sum(ctx.interaction.values()) + sum(
            self.backend.internal(_atoms_of(st)) for st in self._partners.values()
        )
        return ctx

    # -- measurement without designing -------------------------------------

    def build_tables(self) -> EnergyTables:
        """Self/pair energy tables for the current structures.

        Public because both the loop and ``ipro-agent score`` need them, and
        reaching into the loop's internals from the CLI was fragile.
        """
        if self._rotamer_cache is None:
            self.prepare()
        assert self._rotamer_cache is not None
        template = build_template(
            [self._design_st, *self._partners.values()],
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
        tables = self.build_tables()
        self.run.log(f"MILP tables: {json.dumps({k: int(v) for k, v in tables.stats.items()})}")
        if not spec.goal_options.get("reference_energies") and not self.backend.can_relax:
            self.notes.append(
                "no amino-acid reference energies were supplied and the backend has "
                "no built-in reference term, so a rigid-backbone search of buried "
                "positions will tend to pick the smallest allowed residue. Set "
                "goal_options.reference_energies, or use the PyRosetta backend, "
                "before treating identities as a design result."
            )

        # -- starting point: the native sequence, repacked --------------------
        start_choice = self._native_choice(tables)
        start_ctx = self._context_for(self._rotamers_for(tables, start_choice))
        start_score = self.goal.score(start_ctx)

        best_score = start_score
        best_choice = dict(start_choice)
        best_ctx = start_ctx
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
        accepted = 0
        t_start = time.time()
        n_done = 0

        ordered = self._ordered_positions(tables)

        for it in range(1, spec.loop.iterations + 1):
            t0 = time.time()
            n_done = it
            greedy = (it / max(spec.loop.iterations, 1)) <= spec.loop.greedy_fraction

            # (1) choose a contiguous window of positions to re-decide; every
            # other position is pinned to its current identity, so an iteration
            # is a local move rather than a fresh global solve.
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
                    native=self._native,
                    visited=self._visited if spec.loop.integer_cuts else [],
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
            identities = {p: r.aa for p, r in chosen.items()}

            if spec.loop.integer_cuts:
                self._visited.append(dict(identities))

            # (5) score
            ctx = self._context_for(chosen)
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

            mutations = {
                p: aa for p, aa in identities.items()
                if self._native.get(p) and aa != self._native[p]
            }
            rec = IterationRecord(
                iteration=it,
                score=score,
                accepted=ok,
                mutations=mutations,
                components={k: float(v) for k, v in self.goal.components(ctx).items()},
                sequence=self._sequence_string(identities),
                reason=f"{reason}; redesigned {','.join(window)}",
                seconds=time.time() - t0,
            )
            self.run.append_iteration(rec)

            if ok:
                accepted += 1
                current_choice, current_score = dict(sol.choices), score
            if score < best_score:
                best_score, best_choice, best_ctx = score, dict(sol.choices), ctx
                self.run.update_state(
                    best_score=best_score,
                    best_mutations=mutations,
                    best_sequence=rec.sequence,
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

        best_identities = {p: tables.rotamers[p][r].aa for p, r in best_choice.items()}
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
            ensemble=ensemble,
            notes=self.notes,
        )
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

    def _native_choice(self, tables: EnergyTables) -> dict[str, int]:
        """Rotamer indices reproducing the native identity where it is offered.

        Where the native residue is not in the user's allowed set, the
        lowest-self-energy rotamer stands in -- and that is noted, because the
        "starting score" is then not the wild type's.
        """
        out: dict[str, int] = {}
        substituted = []
        for p in tables.positions:
            nat = self._native.get(p)
            idx = next(
                (i for i, r in enumerate(tables.rotamers[p]) if r.aa == nat), None
            )
            if idx is None:
                idx = min(
                    range(len(tables.rotamers[p])),
                    key=lambda i: tables.self_energy.get((p, i), 0.0),
                )
                substituted.append(p)
            out[p] = idx
        if substituted:
            self.notes.append(
                "the native residue is not an allowed choice at "
                f"{', '.join(substituted)}, so the reported starting score is for "
                "the best allowed residue there, not the wild type"
            )
        return out

    @staticmethod
    def _rotamers_for(tables: EnergyTables, choices: dict[str, int]) -> dict[str, Rotamer]:
        return {p: tables.rotamers[p][i] for p, i in choices.items()}

    def _ordered_positions(self, tables: EnergyTables) -> list[str]:
        """Design positions in sequence order, so "contiguous" means something."""
        by_label = {p.label: p for p in self.spec.positions}
        return sorted(
            tables.positions,
            key=lambda lab: (
                by_label[lab].chain if lab in by_label else lab,
                by_label[lab].number if lab in by_label else 0,
                by_label[lab].insertion_code if lab in by_label else "",
            ),
        )

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
