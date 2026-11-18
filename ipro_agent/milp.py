"""The rotamer-selection MILP: choose one rotamer per design position.

This is the mathematical core of IPRO. Given a set of design positions, the
amino acids permitted at each, and a rotamer library, choose exactly one
rotamer at each position so as to minimize the total energy -- subject to
whatever additional constraints the user has stated.

The formulation
---------------
Index sets: positions ``i in P``; rotamers ``r in R_i`` at position ``i``
(each rotamer carries an amino-acid identity ``aa(i,r)``).

Decision variables::

    y[i,r] in {0,1}          rotamer r is chosen at position i
    w[i,r,j,s] in [0,1]      rotamers r at i and s at j are both chosen

Objective::

    min  sum_{i,r} E_self(i,r) * y[i,r]
       + sum_{i<j} sum_{r,s} E_pair(i,r,j,s) * w[i,r,j,s]                  (1)

``E_self(i,r)`` is rotamer r's interaction with everything held fixed -- the
backbone, the side chains of non-design residues, and the target molecules --
plus its internal strain. ``E_pair`` is the interaction between two chosen
rotamers.

Constraints::

    sum_{r in R_i} y[i,r] = 1                     for all i              (2)
    sum_{s in R_j} w[i,r,j,s] = y[i,r]            for all i,r and j != i (3)
    w[i,r,j,s] >= 0                                                      (4)

(2) picks exactly one rotamer per position. (3) is the standard linearization
of the quadratic term: summing the pair variable over all rotamers of the other
position must recover the single-position variable. Together with (2) and
binary ``y``, this forces ``w`` to take the value ``y[i,r]*y[j,s]`` at any
optimum, so ``w`` needs no integrality constraint of its own -- which is what
keeps the problem tractable. This is the classical rotamer-assignment
linearization used by IPRO; see docs/MILP.md for the citation and the proof
sketch.

Why an integer program instead of a packer
------------------------------------------
A simulated-annealing side-chain packer (as in Rosetta) solves (1)-(2) well. It
cannot express the *user's* constraints, which is the whole point of this tool:

    at most M mutations this iteration                                   (5)
    at least K positions drawn from a named residue class                (6)
    a given position must / must not take a given identity               (7)
    total energy of the chosen assignment must not exceed a cutoff       (8)

All four are linear in ``y`` (and ``w``, for (8)), so they drop straight into
the same program. None is hypothetical:

* (6) is the PoreDesigner requirement that at least half of the designed
  positions be "long" residues (W/F/Y/M);
* (8) is IPRO's own stability constraint, which caps the assignment's energy
  rather than only minimizing it -- in the 2006 formulation,
  ``sum X(E_rb + E_rs) + sum Z E_rr + E_sb <= E_cutoff``.

See docs/MILP.md for the correspondence to the published equation numbers.

Practical size control
----------------------
The pair-variable count grows as (positions x rotamers)^2, so three prunings
are applied before the program is written, each reported in the result:

* **rotamer pruning (generalized dead-end elimination)** -- a rotamer whose
  self-energy exceeds ``max_self_energy`` clashes irrecoverably with fixed
  atoms and can never appear in an optimal assignment, so it is dropped. This
  is IPRO's ``X_ir = 0 if E_rs(i,r) > delta_i`` condition. Implemented with two
  safeguards the bare rule lacks: a position never loses its last rotamer
  (which would make the program infeasible), and the native identity is never
  pruned (so "do not mutate" stays reachable and a mutation can be rejected);
* **position-pair pruning** -- two positions whose rotamer clouds cannot reach
  each other (centroid separation beyond the sum of their radii plus the energy
  cutoff) contribute no pair variables at all;
* **pair-term pruning** -- individual pair energies with ``|E| < eps`` are
  dropped, since they cannot change the optimum materially.
"""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .config import AA_GROUPS, THREE_TO_ONE
from .energy.base import AtomSet, EnergyBackend

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rotamers
# ---------------------------------------------------------------------------


@dataclass
class Rotamer:
    """One side-chain conformation of one amino acid at one position."""

    position: str          # "A115"
    aa: str                # "TRP"
    state: int             # 1-based index within PyMOL's rotamer set
    atoms: AtomSet
    strain: float | None = None   # PyMOL's internal strain, if reported

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.position, self.aa, self.state)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Rotamer({self.position} {self.aa} #{self.state}, {len(self.atoms)} atoms)"


@dataclass
class Constraint:
    """A user-stated linear constraint over rotamer choices.

    ``kind``:
      ``max_mutations``   at most ``count`` positions differ from native
      ``min_class``       at least ``count`` positions take an identity in ``residues``
      ``max_class``       at most ``count`` positions take an identity in ``residues``
      ``require``         ``positions[0]`` must take an identity in ``residues``
      ``forbid``          ``positions[0]`` must not take an identity in ``residues``
      ``max_energy``      the assignment's objective value must be <= ``count``
                          (IPRO's stability constraint; ``count`` is in the
                          backend's energy units, so it is not portable between
                          backends -- set it from a measured baseline)

    ``count`` may be given as a fraction of the number of design positions by
    passing ``fraction`` instead; it is rounded up.
    """

    kind: str
    count: int | None = None
    fraction: float | None = None
    residues: list[str] = field(default_factory=list)
    positions: list[str] = field(default_factory=list)
    label: str = ""

    def resolve_count(self, n_positions: int) -> int:
        if self.count is not None:
            return int(self.count)
        if self.fraction is not None:
            return int(math.ceil(self.fraction * n_positions))
        raise ValueError(f"constraint {self.kind} needs 'count' or 'fraction'")

    def residue_set(self) -> set[str]:
        out: set[str] = set()
        for tok in self.residues:
            t = str(tok).strip()
            if t.lower() in AA_GROUPS:
                out.update(AA_GROUPS[t.lower()])
            elif len(t) == 1:
                from .config import ONE_TO_THREE
                if t.upper() in ONE_TO_THREE:
                    out.add(ONE_TO_THREE[t.upper()])
            else:
                out.add(t.upper())
        return out


# ---------------------------------------------------------------------------
# Energy tables
# ---------------------------------------------------------------------------


@dataclass
class EnergyTables:
    """Self and pair energies over a pruned rotamer set."""

    positions: list[str]
    rotamers: dict[str, list[Rotamer]]
    self_energy: dict[tuple[str, int], float] = field(default_factory=dict)
    pair_energy: dict[tuple[str, int, str, int], float] = field(default_factory=dict)
    #: Diagnostics, reported so a surprising answer can be traced.
    stats: dict[str, float] = field(default_factory=dict)

    def rot_index(self, pos: str) -> list[int]:
        return list(range(len(self.rotamers[pos])))

    def n_pair_terms(self) -> int:
        return len(self.pair_energy)


def build_tables(
    rotamers: dict[str, list[Rotamer]],
    template: AtomSet,
    backend: EnergyBackend,
    *,
    strain_weight: float = 0.0,
    max_self_energy: float = 50.0,
    pair_eps: float = 1e-3,
    energy_cutoff: float = 12.0,
    native: dict[str, str] | None = None,
) -> EnergyTables:
    """Compute self/pair energies, pruning as described in the module docstring.

    ``template`` is everything held fixed this iteration: backbone atoms,
    non-design side chains, and the target/anti-target molecules. Because the
    template is identical for every rotamer choice, its internal energy is a
    constant and is deliberately *not* included -- the MILP only needs terms
    that depend on the choice. Absolute energies come from the loop's
    whole-structure scoring instead.

    ``strain_weight`` scales the rotamer-library strain into the self energy. It
    defaults to 0: PyMOL's strain is on its own scale and mixing it into a
    kcal/mol total without calibration would be unjustified. Set it
    deliberately if you want rotamer-probability-like behaviour.
    """
    native = native or {}
    positions = sorted(rotamers)
    tab = EnergyTables(positions=positions, rotamers={}, stats={})

    # -- self energies, with rotamer pruning ------------------------------
    n_in = n_pruned = 0
    for pos in positions:
        kept: list[Rotamer] = []
        scored: list[tuple[float, Rotamer]] = []
        for rot in rotamers[pos]:
            n_in += 1
            e = backend.interaction(rot.atoms, template)
            if strain_weight and rot.strain is not None:
                e += strain_weight * rot.strain
            scored.append((e, rot))

        # Keep anything below the clash threshold; always keep at least one
        # rotamer per position (else the program is infeasible) and always keep
        # the native identity if it was offered, so "no mutation" stays
        # reachable -- IPRO must be able to reject a mutation.
        nat = native.get(pos)
        best = min(scored, key=lambda t: t[0])[1] if scored else None
        for e, rot in scored:
            keep = e <= max_self_energy or rot is best or (nat and rot.aa == nat)
            if keep:
                kept.append(rot)
            else:
                n_pruned += 1
        tab.rotamers[pos] = kept
        by_key = {rot.key: e for e, rot in scored}
        for idx, rot in enumerate(kept):
            tab.self_energy[(pos, idx)] = by_key[rot.key]

    # -- pair energies, with position-pair and term pruning ---------------
    n_pairs_possible = 0
    n_pairs_kept = 0
    n_pos_pairs_skipped = 0
    for a in range(len(positions)):
        for b in range(a + 1, len(positions)):
            pi, pj = positions[a], positions[b]
            Ri, Rj = tab.rotamers[pi], tab.rotamers[pj]
            if not Ri or not Rj:
                continue
            n_pairs_possible += len(Ri) * len(Rj)

            # Can any rotamer at pi reach any rotamer at pj?
            ci = np.mean([r.atoms.centroid() for r in Ri if len(r.atoms)] or [np.zeros(3)], axis=0)
            cj = np.mean([r.atoms.centroid() for r in Rj if len(r.atoms)] or [np.zeros(3)], axis=0)
            radi = max((r.atoms.bounding_radius() for r in Ri), default=0.0)
            radj = max((r.atoms.bounding_radius() for r in Rj), default=0.0)
            if np.linalg.norm(ci - cj) > radi + radj + energy_cutoff:
                n_pos_pairs_skipped += 1
                continue

            for ri, rot_i in enumerate(Ri):
                for rj, rot_j in enumerate(Rj):
                    e = backend.interaction(rot_i.atoms, rot_j.atoms)
                    if abs(e) >= pair_eps:
                        tab.pair_energy[(pi, ri, pj, rj)] = e
                        n_pairs_kept += 1

    tab.stats = {
        "rotamers_in": n_in,
        "rotamers_pruned": n_pruned,
        "rotamers_kept": sum(len(v) for v in tab.rotamers.values()),
        "position_pairs_skipped": n_pos_pairs_skipped,
        "pair_terms_possible": n_pairs_possible,
        "pair_terms_kept": n_pairs_kept,
    }
    log.info(
        "MILP tables: %d rotamers kept (%d pruned), %d pair terms of %d possible",
        tab.stats["rotamers_kept"], n_pruned, n_pairs_kept, n_pairs_possible,
    )
    return tab


# ---------------------------------------------------------------------------
# The program
# ---------------------------------------------------------------------------


@dataclass
class MILPSolution:
    """The chosen rotamer at each position, plus provenance."""

    choices: dict[str, int]              # position -> index into tables.rotamers[pos]
    objective: float
    status: str
    solver: str
    #: Path of the written LP file, if one was kept.
    lp_path: Path | None = None
    seconds: float = 0.0
    #: True only if the solver proved optimality.
    proven_optimal: bool = False

    def identities(self, tables: EnergyTables) -> dict[str, str]:
        return {p: tables.rotamers[p][i].aa for p, i in self.choices.items()}

    def mutations(self, tables: EnergyTables, native: dict[str, str]) -> dict[str, str]:
        out = {}
        for p, i in self.choices.items():
            aa = tables.rotamers[p][i].aa
            if native.get(p) and aa != native[p]:
                out[p] = aa
        return out


class RotamerMILP:
    """Builds, writes and solves the rotamer-selection program."""

    def __init__(
        self,
        tables: EnergyTables,
        constraints: Sequence[Constraint] = (),
        native: dict[str, str] | None = None,
        visited: Sequence[dict[str, str]] = (),
    ) -> None:
        self.t = tables
        self.constraints = list(constraints)
        self.native = dict(native or {})
        #: Sequences already evaluated, as position -> amino acid. Each becomes
        #: an integer ("no good") cut so the optimizer cannot return it again:
        #:
        #:     sum_{i} sum_{r : aa(i,r) = A[i]} y[i,r]  <=  |P| - 1
        #:
        #: which is violated only by the assignment A itself. This is IPRO's
        #: own device for never revisiting a design, and it is what makes the
        #: iteration loop explore rather than re-solve the same optimum.
        self.visited = [dict(v) for v in visited]
        # Stable integer ids keep LP variable names short and deterministic.
        self._pi = {p: i for i, p in enumerate(tables.positions)}

    # -- naming ------------------------------------------------------------

    def _y(self, pos: str, r: int) -> str:
        return f"y_{self._pi[pos]}_{r}"

    def _w(self, pi: str, r: int, pj: str, s: int) -> str:
        return f"w_{self._pi[pi]}_{r}_{self._pi[pj]}_{s}"

    # -- LP emission -------------------------------------------------------

    def to_lp(self) -> str:
        """Write the program in CPLEX LP format, commented.

        The comments are not decoration: this file is the artifact that records
        what formulation was actually solved, and it is what a user should read
        to check that their stated constraints were encoded as they meant them.
        """
        t = self.t
        L: list[str] = []
        L.append("\\ IPRO rotamer-selection MILP, written by ipro-agent")
        L.append("\\ Objective (1): min sum E_self*y + sum E_pair*w")
        L.append(f"\\ positions: {len(t.positions)}")
        for p in t.positions:
            ids = ",".join(
                f"{THREE_TO_ONE.get(r.aa, r.aa)}#{r.state}" for r in t.rotamers[p]
            )
            L.append(f"\\   {p} (native {self.native.get(p, '?')}): {len(t.rotamers[p])} rotamers [{ids}]")
        L.append(f"\\ pair terms: {len(t.pair_energy)}")
        L.append("")

        # -- objective
        L.append("Minimize")
        terms: list[str] = []
        for p in t.positions:
            for r in range(len(t.rotamers[p])):
                c = t.self_energy.get((p, r), 0.0)
                if c:
                    terms.append(f"{c:+.6g} {self._y(p, r)}")
        for (pi, r, pj, s), c in sorted(t.pair_energy.items()):
            if c:
                terms.append(f"{c:+.6g} {self._w(pi, r, pj, s)}")
        if not terms:
            terms.append("0 ZERO_OBJ")
        L.append(" obj: " + _wrap_terms(terms))
        L.append("")

        # -- constraints
        L.append("Subject To")

        L.append("\\ (2) exactly one rotamer per design position")
        for p in t.positions:
            vs = [self._y(p, r) for r in range(len(t.rotamers[p]))]
            if vs:
                L.append(f" one_{self._pi[p]}: " + _wrap_terms([f"+1 {v}" for v in vs]) + " = 1")

        L.append("\\ (3) pair-variable linearization: sum_s w[i,r,j,s] = y[i,r]")
        # Group pair terms by (i,r,j) so each group yields one constraint. Only
        # position pairs that survived pruning appear, and the constraint is
        # written for both orientations, which is what makes w consistent.
        groups: dict[tuple[str, int, str], list[str]] = {}
        for (pi, r, pj, s) in t.pair_energy:
            groups.setdefault((pi, r, pj), []).append(self._w(pi, r, pj, s))
            groups.setdefault((pj, s, pi), []).append(self._w(pi, r, pj, s))
        for n, ((p, r, q), ws) in enumerate(sorted(groups.items())):
            lhs = _wrap_terms([f"+1 {w}" for w in sorted(set(ws))])
            L.append(f" link{n}: {lhs} -1 {self._y(p, r)} = 0")

        # -- user constraints
        for n, con in enumerate(self.constraints):
            L.extend(self._emit_constraint(n, con))

        # -- integer cuts over already-visited sequences
        if self.visited:
            L.append("\\ (9) integer cuts: forbid sequences already evaluated")
        for n, assignment in enumerate(self.visited):
            vs = self._cut_vars(assignment)
            if not vs:
                continue
            L.append(
                f" cut{n}: " + _wrap_terms([f"+1 {v}" for v in vs])
                + f" <= {len(assignment) - 1}"
            )

        # -- variable declarations
        L.append("")
        L.append("Bounds")
        for (pi, r, pj, s) in sorted(t.pair_energy):
            L.append(f" 0 <= {self._w(pi, r, pj, s)} <= 1")
        if not t.pair_energy:
            pass
        L.append("")
        L.append("Binaries")
        bins = [self._y(p, r) for p in t.positions for r in range(len(t.rotamers[p]))]
        if not bins:
            bins = ["ZERO_OBJ"]
        for chunk in _chunks(bins, 10):
            L.append(" " + " ".join(chunk))
        L.append("End")
        return "\n".join(L) + "\n"

    def _emit_constraint(self, n: int, con: Constraint) -> list[str]:
        t = self.t
        npos = len(t.positions)
        out: list[str] = []
        tag = con.label or f"{con.kind}{n}"

        if con.kind == "max_mutations":
            k = con.resolve_count(npos)
            vs = [
                self._y(p, r)
                for p in t.positions
                for r, rot in enumerate(t.rotamers[p])
                if self.native.get(p) and rot.aa != self.native[p]
            ]
            out.append(f"\\ (5) at most {k} mutations")
            if vs:
                out.append(f" {tag}: " + _wrap_terms([f"+1 {v}" for v in vs]) + f" <= {k}")

        elif con.kind in ("min_class", "max_class"):
            k = con.resolve_count(npos)
            want = con.residue_set()
            vs = [
                self._y(p, r)
                for p in t.positions
                for r, rot in enumerate(t.rotamers[p])
                if rot.aa in want
            ]
            rel = ">=" if con.kind == "min_class" else "<="
            short = ",".join(sorted(THREE_TO_ONE.get(a, a) for a in want))
            out.append(f"\\ (6) {con.kind}: {rel} {k} positions from {{{short}}}")
            if vs:
                out.append(f" {tag}: " + _wrap_terms([f"+1 {v}" for v in vs]) + f" {rel} {k}")
            elif con.kind == "min_class" and k > 0:
                out.append(
                    f"\\ WARNING: no rotamer anywhere has an identity in {{{short}}}; "
                    "this constraint cannot be satisfied and is omitted so the "
                    "infeasibility is reported here rather than by the solver"
                )

        elif con.kind in ("require", "forbid"):
            want = con.residue_set()
            for p in (con.positions or t.positions):
                if p not in t.rotamers:
                    continue
                vs = [
                    self._y(p, r)
                    for r, rot in enumerate(t.rotamers[p])
                    if rot.aa in want
                ]
                if not vs:
                    continue
                rel, rhs = ("= 1", 1) if con.kind == "require" else ("= 0", 0)
                out.append(f"\\ (7) {con.kind} at {p}: {sorted(want)}")
                out.append(f" {tag}_{self._pi[p]}: " + _wrap_terms([f"+1 {v}" for v in vs]) + f" {rel}")

        elif con.kind == "max_energy":
            cap = con.resolve_count(npos)
            terms: list[str] = []
            for p in t.positions:
                for r in range(len(t.rotamers[p])):
                    c = t.self_energy.get((p, r), 0.0)
                    if c:
                        terms.append(f"{c:+.6g} {self._y(p, r)}")
            for (pi, r, pj, s), c in sorted(t.pair_energy.items()):
                if c:
                    terms.append(f"{c:+.6g} {self._w(pi, r, pj, s)}")
            out.append(f"\\ (8) IPRO stability constraint: assignment energy <= {cap}")
            if terms:
                out.append(f" {tag}: " + _wrap_terms(terms) + f" <= {cap}")

        else:
            out.append(f"\\ WARNING: unknown constraint kind {con.kind!r}, ignored")
        return out

    def _cut_vars(self, assignment: dict[str, str]) -> list[str]:
        """Variables whose simultaneous selection reproduces ``assignment``.

        An assignment names an amino acid per position, not a rotamer state, so
        every rotamer of that identity contributes -- the cut forbids the
        *sequence*, leaving the optimizer free to repack it differently if some
        other constraint later makes that necessary.
        """
        out: list[str] = []
        for p, aa in assignment.items():
            if p not in self.t.rotamers:
                continue
            out.extend(
                self._y(p, r)
                for r, rot in enumerate(self.t.rotamers[p])
                if rot.aa == aa
            )
        return out

    # -- solving -----------------------------------------------------------

    def solve(
        self,
        solver: str = "auto",
        lp_path: Path | None = None,
        time_limit: float | None = None,
    ) -> MILPSolution:
        """Solve, writing the LP file to ``lp_path`` if given.

        Tries, in order: PuLP (with whatever solver it has), a solver binary via
        LP file, then a documented greedy heuristic. The heuristic never claims
        optimality.
        """
        import time

        t0 = time.time()
        lp_text = self.to_lp()
        if lp_path:
            lp_path = Path(lp_path)
            lp_path.parent.mkdir(parents=True, exist_ok=True)
            lp_path.write_text(lp_text)

        order = (
            ["pulp", "binary", "greedy"]
            if solver == "auto"
            else ([solver] if solver != "greedy" else ["greedy"])
        )
        last_err: Exception | None = None
        for which in order:
            try:
                if which == "pulp":
                    sol = self._solve_pulp(time_limit)
                elif which == "binary":
                    sol = self._solve_binary(lp_text, time_limit)
                else:
                    sol = self._solve_greedy()
                sol.lp_path = lp_path
                sol.seconds = time.time() - t0
                return sol
            except Exception as exc:  # try the next option
                last_err = exc
                log.info("MILP solver %r unavailable or failed: %s", which, exc)
        raise RuntimeError(f"no MILP solver succeeded; last error: {last_err}")

    # -- solver implementations -------------------------------------------

    def _solve_pulp(self, time_limit: float | None) -> MILPSolution:
        import pulp  # noqa: F401  (raises if absent)

        t = self.t
        prob = pulp.LpProblem("ipro_rotamer_selection", pulp.LpMinimize)
        y = {
            (p, r): pulp.LpVariable(self._y(p, r), cat="Binary")
            for p in t.positions
            for r in range(len(t.rotamers[p]))
        }
        w = {
            k: pulp.LpVariable(self._w(*k), lowBound=0, upBound=1)
            for k in t.pair_energy
        }
        prob += (
            pulp.lpSum(t.self_energy.get((p, r), 0.0) * v for (p, r), v in y.items())
            + pulp.lpSum(t.pair_energy[k] * v for k, v in w.items())
        )
        for p in t.positions:
            prob += pulp.lpSum(y[(p, r)] for r in range(len(t.rotamers[p]))) == 1
        groups: dict[tuple[str, int, str], list] = {}
        for (pi, r, pj, s), v in w.items():
            groups.setdefault((pi, r, pj), []).append(v)
            groups.setdefault((pj, s, pi), []).append(v)
        for (p, r, _q), vs in groups.items():
            prob += pulp.lpSum(vs) == y[(p, r)]
        self._add_pulp_user_constraints(prob, pulp, y, w)
        for assignment in self.visited:
            vs = [
                y[(p, r)]
                for p, aa in assignment.items()
                if p in t.rotamers
                for r, rot in enumerate(t.rotamers[p])
                if rot.aa == aa
            ]
            if vs:
                prob += pulp.lpSum(vs) <= len(assignment) - 1

        solver_obj = None
        for name in ("CPLEX_CMD", "GUROBI_CMD", "PULP_CBC_CMD", "COIN_CMD", "GLPK_CMD"):
            if name in pulp.listSolvers(onlyAvailable=True):
                kw = {"msg": False}
                if time_limit:
                    kw["timeLimit"] = time_limit
                solver_obj = getattr(pulp, name)(**kw)
                break
        if solver_obj is None:
            raise RuntimeError("pulp has no available solver")
        prob.solve(solver_obj)
        status = pulp.LpStatus[prob.status]
        if status not in ("Optimal",):
            raise RuntimeError(f"pulp status {status}")
        choices = {}
        for p in t.positions:
            for r in range(len(t.rotamers[p])):
                if y[(p, r)].value() and y[(p, r)].value() > 0.5:
                    choices[p] = r
                    break
        return MILPSolution(
            choices=choices,
            objective=float(pulp.value(prob.objective) or 0.0),
            status=status,
            solver=f"pulp/{type(solver_obj).__name__}",
            proven_optimal=True,
        )

    def _add_pulp_user_constraints(self, prob, pulp, y, w=None) -> None:
        t = self.t
        npos = len(t.positions)
        w = w or {}
        for con in self.constraints:
            if con.kind == "max_energy":
                prob += (
                    pulp.lpSum(
                        t.self_energy.get((p, r), 0.0) * v for (p, r), v in y.items()
                    )
                    + pulp.lpSum(t.pair_energy[k] * v for k, v in w.items())
                ) <= con.resolve_count(npos)
                continue
            if con.kind == "max_mutations":
                k = con.resolve_count(npos)
                vs = [
                    y[(p, r)]
                    for p in t.positions
                    for r, rot in enumerate(t.rotamers[p])
                    if self.native.get(p) and rot.aa != self.native[p]
                ]
                if vs:
                    prob += pulp.lpSum(vs) <= k
            elif con.kind in ("min_class", "max_class"):
                k = con.resolve_count(npos)
                want = con.residue_set()
                vs = [
                    y[(p, r)]
                    for p in t.positions
                    for r, rot in enumerate(t.rotamers[p])
                    if rot.aa in want
                ]
                if vs:
                    if con.kind == "min_class":
                        prob += pulp.lpSum(vs) >= k
                    else:
                        prob += pulp.lpSum(vs) <= k
            elif con.kind in ("require", "forbid"):
                want = con.residue_set()
                for p in (con.positions or t.positions):
                    if p not in t.rotamers:
                        continue
                    vs = [
                        y[(p, r)]
                        for r, rot in enumerate(t.rotamers[p])
                        if rot.aa in want
                    ]
                    if vs:
                        prob += pulp.lpSum(vs) == (1 if con.kind == "require" else 0)

    def _solve_binary(self, lp_text: str, time_limit: float | None) -> MILPSolution:
        """Solve by handing an LP file to a solver binary (glpsol / cbc / cplex)."""
        glpsol = shutil.which("glpsol")
        cbc = shutil.which("cbc")
        if not (glpsol or cbc):
            raise RuntimeError("no solver binary (glpsol, cbc) on PATH")

        with tempfile.TemporaryDirectory(prefix="ipro_milp_") as td:
            lp = Path(td) / "model.lp"
            out = Path(td) / "model.sol"
            lp.write_text(lp_text)
            if glpsol:
                cmd = ["glpsol", "--lp", str(lp), "-o", str(out)]
                if time_limit:
                    cmd += ["--tmlim", str(int(time_limit))]
                name = "glpsol"
            else:
                cmd = [cbc, str(lp), "solve", "solu", str(out)]
                if time_limit:
                    cmd = [cbc, str(lp), "sec", str(int(time_limit)), "solve", "solu", str(out)]
                name = "cbc"
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=max(60.0, (time_limit or 0) + 60))
            if not out.exists():
                raise RuntimeError(
                    f"{name} produced no solution file (rc={proc.returncode}): "
                    f"{(proc.stdout + proc.stderr)[-400:]}"
                )
            text = out.read_text()
            vals, status = (
                _parse_glpk_solution(text) if name == "glpsol" else _parse_cbc_solution(text)
            )

        if not vals:
            raise RuntimeError(f"{name} solution file had no variable values")
        choices: dict[str, int] = {}
        for p in self.t.positions:
            best_r, best_v = None, -1.0
            for r in range(len(self.t.rotamers[p])):
                v = vals.get(self._y(p, r), 0.0)
                if v > best_v:
                    best_r, best_v = r, v
            if best_r is None:
                raise RuntimeError(f"no rotamer variable found for {p}")
            choices[p] = best_r
        obj = self.evaluate(choices)
        # glpsol reports "INTEGER OPTIMAL" for a MIP and "OPTIMAL" for an LP;
        # CBC reports "Optimal". Anything else (time limit, feasible-only) must
        # not be reported as proven optimal.
        s = status.lower()
        proven = "optimal" in s and "not" not in s
        return MILPSolution(
            choices=choices, objective=obj, status=status, solver=name,
            proven_optimal=proven,
        )

    def _solve_greedy(self) -> MILPSolution:
        """Iterated best-improvement over single-position reassignments.

        A documented fallback for when no solver is installed. It respects the
        one-rotamer-per-position constraint by construction and *checks* the
        user constraints, rejecting moves that would violate them -- but it
        does **not** prove optimality, and ``proven_optimal`` stays False. Treat
        its answers as a lower bound on quality, not as the MILP optimum.
        """
        t = self.t
        choices = {}
        for p in t.positions:
            rs = range(len(t.rotamers[p]))
            choices[p] = min(rs, key=lambda r: t.self_energy.get((p, r), 0.0))

        # Repair toward feasibility, then improve.
        for _ in range(200):
            if self._feasible(choices):
                break
            if not self._repair_step(choices):
                break

        improved = True
        rounds = 0
        while improved and rounds < 100:
            improved = False
            rounds += 1
            for p in t.positions:
                cur = choices[p]
                best, best_e = cur, self.evaluate(choices)
                for r in range(len(t.rotamers[p])):
                    if r == cur:
                        continue
                    choices[p] = r
                    if not self._feasible(choices):
                        continue
                    e = self.evaluate(choices)
                    if e < best_e - 1e-9:
                        best, best_e = r, e
                choices[p] = best
                if best != cur:
                    improved = True
        feas = self._feasible(choices)
        return MILPSolution(
            choices=choices,
            objective=self.evaluate(choices),
            status="heuristic" + ("" if feas else "-INFEASIBLE"),
            solver="greedy (built-in heuristic; optimality NOT proven)",
            proven_optimal=False,
        )

    # -- helpers -----------------------------------------------------------

    def evaluate(self, choices: dict[str, int]) -> float:
        """Objective (1) at a given assignment. Used by the heuristic and tests."""
        t = self.t
        e = sum(t.self_energy.get((p, r), 0.0) for p, r in choices.items())
        for (pi, ri, pj, rj), c in t.pair_energy.items():
            if choices.get(pi) == ri and choices.get(pj) == rj:
                e += c
        return float(e)

    def _feasible(self, choices: dict[str, int]) -> bool:
        t = self.t
        npos = len(t.positions)
        ids = {p: t.rotamers[p][r].aa for p, r in choices.items()}
        for con in self.constraints:
            if con.kind == "max_mutations":
                k = con.resolve_count(npos)
                n = sum(1 for p, aa in ids.items() if self.native.get(p) and aa != self.native[p])
                if n > k:
                    return False
            elif con.kind in ("min_class", "max_class"):
                k = con.resolve_count(npos)
                want = con.residue_set()
                n = sum(1 for aa in ids.values() if aa in want)
                if con.kind == "min_class" and n < k:
                    return False
                if con.kind == "max_class" and n > k:
                    return False
            elif con.kind == "require":
                want = con.residue_set()
                for p in (con.positions or t.positions):
                    if p in ids and ids[p] not in want:
                        return False
            elif con.kind == "forbid":
                want = con.residue_set()
                for p in (con.positions or t.positions):
                    if p in ids and ids[p] in want:
                        return False
            elif con.kind == "max_energy":
                if self.evaluate(choices) > con.resolve_count(npos):
                    return False
        # Integer cuts: an assignment matching a visited sequence at every
        # position is forbidden.
        for assignment in self.visited:
            if assignment and all(ids.get(p) == aa for p, aa in assignment.items()):
                return False
        return True

    def _repair_step(self, choices: dict[str, int]) -> bool:
        """Change one position to move toward feasibility. True if it changed."""
        t = self.t
        npos = len(t.positions)
        for con in self.constraints:
            if con.kind == "min_class":
                want = con.residue_set()
                k = con.resolve_count(npos)
                have = sum(1 for p, r in choices.items() if t.rotamers[p][r].aa in want)
                if have >= k:
                    continue
                # Switch the cheapest position that can take a wanted identity.
                cands = []
                for p in t.positions:
                    if t.rotamers[p][choices[p]].aa in want:
                        continue
                    for r, rot in enumerate(t.rotamers[p]):
                        if rot.aa in want:
                            cands.append((t.self_energy.get((p, r), 0.0), p, r))
                if not cands:
                    return False
                _, p, r = min(cands)
                choices[p] = r
                return True
            if con.kind == "require":
                want = con.residue_set()
                for p in (con.positions or t.positions):
                    if p in choices and t.rotamers[p][choices[p]].aa not in want:
                        for r, rot in enumerate(t.rotamers[p]):
                            if rot.aa in want:
                                choices[p] = r
                                return True
            if con.kind == "forbid":
                want = con.residue_set()
                for p in (con.positions or t.positions):
                    if p in choices and t.rotamers[p][choices[p]].aa in want:
                        alts = [(t.self_energy.get((p, r), 0.0), r)
                                for r, rot in enumerate(t.rotamers[p]) if rot.aa not in want]
                        if alts:
                            choices[p] = min(alts)[1]
                            return True
            if con.kind == "max_mutations":
                k = con.resolve_count(npos)
                muts = [p for p, r in choices.items()
                        if self.native.get(p) and t.rotamers[p][r].aa != self.native[p]]
                if len(muts) <= k:
                    continue
                # Revert the most expensive mutation to native, if offered.
                for p in sorted(muts, key=lambda p: -self.t.self_energy.get((p, choices[p]), 0.0)):
                    nat = self.native.get(p)
                    for r, rot in enumerate(t.rotamers[p]):
                        if rot.aa == nat:
                            choices[p] = r
                            return True
        return False


# ---------------------------------------------------------------------------
# Solver-output parsing and LP text helpers
# ---------------------------------------------------------------------------


def _parse_glpk_solution(text: str) -> tuple[dict[str, float], str]:
    """Parse ``glpsol -o`` plain-text output.

    Format is a header block then one line per row and per column; column lines
    carry the variable name and its value. Names can be pushed onto their own
    line when long, so the value is taken from the token stream rather than a
    fixed column.
    """
    vals: dict[str, float] = {}
    status = "unknown"
    in_cols = False
    pending_name: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        low = line.strip().lower()
        if low.startswith("status:"):
            status = line.split(":", 1)[1].strip()
        if low.startswith("no.") and "column name" in low:
            in_cols = True
            continue
        if not in_cols:
            continue
        if not line.strip() or set(line.strip()) <= set("- "):
            continue
        toks = line.split()
        if pending_name is not None:
            # Previous line held only the index and name.
            for tk in toks:
                try:
                    vals[pending_name] = float(tk)
                    break
                except ValueError:
                    continue
            pending_name = None
            continue
        if toks[0].isdigit():
            if len(toks) >= 2:
                name = toks[1]
                num = None
                for tk in toks[2:]:
                    tk2 = tk.replace("*", "")
                    try:
                        num = float(tk2)
                        break
                    except ValueError:
                        continue
                if num is None:
                    pending_name = name
                else:
                    vals[name] = num
    return vals, status


def _parse_cbc_solution(text: str) -> tuple[dict[str, float], str]:
    """Parse CBC ``solu`` output: an optional status line then ``idx name value``."""
    vals: dict[str, float] = {}
    status = "unknown"
    for i, line in enumerate(text.splitlines()):
        toks = line.split()
        if not toks:
            continue
        if i == 0 and not toks[0].isdigit():
            status = " ".join(toks[:2])
            continue
        if len(toks) >= 3 and toks[0].isdigit():
            try:
                vals[toks[1]] = float(toks[2])
            except ValueError:
                continue
    return vals, status


def _wrap_terms(terms: Iterable[str], width: int = 240) -> str:
    """Join LP terms, wrapping so no line exceeds LP-format line limits."""
    out: list[str] = []
    line = ""
    for t in terms:
        if len(line) + len(t) + 1 > width:
            out.append(line)
            line = ""
        line = f"{line} {t}".strip()
    if line:
        out.append(line)
    return "\n     ".join(out)


def _chunks(seq: Sequence[str], n: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]
