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

* **rotamer pruning** -- a rotamer whose self-energy is more than
  ``rotamer_window`` kcal/mol worse than the **best rotamer at that same
  position** is dropped. That relative form is IPRO's published rule
  (``ROTAMERS.py``: keep ``sorted[n][i] <= sorted[n][0] + window`` with a
  40 kcal/mol window), and the window is why: it is a statement about which
  conformers are competitive *there*, not an absolute energy.

  **It is not dead-end elimination and the claim has been withdrawn.** A
  rotamer with a high self-energy can still be optimal, because a favourable
  pair term can outweigh it: constructed with this package's own MILP and
  glpsol, with self energies ``{A1 r0: 0.0, A1 r1: 51.0}`` and a pair term
  ``(A1,r1,A2,r0): -60.0``, the full table's optimum is **-9.000** choosing
  r1, while an absolute >50 prune gives **+0.000** choosing r0 -- the prune
  changed the answer. A true DEE criterion (Goldstein) compares against the
  pair-energy extrema; this does not, so it is a heuristic filter, reported as
  one in ``EnergyTables.stats``.

  Two safeguards the bare rule lacks: a position never loses its last rotamer
  (which would make the program infeasible), and the native identity is never
  pruned (so "do not mutate" stays reachable and a mutation can be rejected);
  ``max_self_energy``, if set, is an additional absolute ceiling for genuine
  clashes, and defaults to off -- as an absolute default of 50 it pruned
  TRP, PHE and TYR out of OmpF position A18 entirely, which removed the
  published V18W answer from the design space (docs/DESIGN.md 7d, defect 25);
* **position-pair pruning** -- two positions whose rotamer clouds cannot reach
  each other (centroid separation beyond the sum of their radii plus the energy
  cutoff) contribute no pair variables at all;
* **negligible-block pruning** -- a position pair is dropped only when *every*
  interaction in its block is below ``pair_eps``. Individual terms are never
  dropped from a surviving block: the linearization requires a complete block
  (see :func:`build_tables`).
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

#: Solvers a user may name in ``backends.milp_solver``, mapped to the PuLP
#: class that drives them and the binary ``_solve_binary`` can read a solution
#: from (None where this code cannot parse that solver's output directly).
NAMED_SOLVERS: dict[str, tuple[str, str | None]] = {
    "cplex": ("CPLEX_CMD", None),
    "gurobi": ("GUROBI_CMD", None),
    "cbc": ("PULP_CBC_CMD", "cbc"),
    "coin": ("COIN_CMD", "cbc"),
    "highs": ("HiGHS", None),
    "scip": ("SCIP_CMD", None),
    "glpk": ("GLPK_CMD", "glpsol"),
}


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
        """A whole number of positions. Only for counting constraints."""
        if self.count is not None:
            return int(self.count)
        if self.fraction is not None:
            return int(math.ceil(self.fraction * n_positions))
        raise ValueError(f"constraint {self.kind} needs 'count' or 'fraction'")

    def resolve_value(self, n_positions: int) -> float:
        """A real-valued right-hand side, for ``max_energy``.

        Separate from :meth:`resolve_count` because an energy cutoff is an
        energy, not a count: ``int()`` would turn a -12.5 kcal/mol cap into
        -12, and a -0.5 cap into 0, silently changing a documented threshold.
        """
        if self.count is not None:
            return float(self.count)
        if self.fraction is not None:
            return float(self.fraction) * n_positions
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


def _enclosing_radius(rots: Sequence["Rotamer"], centre: np.ndarray) -> float:
    """Radius of a sphere about ``centre`` enclosing every atom of ``rots``.

    Used by the position-pair pre-screen, which must never skip a block that
    holds a reachable pair. ``AtomSet.bounding_radius()`` is measured about the
    set's own centroid and so does not bound the set seen from anywhere else.
    """
    far = 0.0
    for r in rots:
        if len(r.atoms) == 0:
            continue
        d = np.sqrt(((r.atoms.coords - centre) ** 2).sum(-1)).max()
        far = max(far, float(d))
    return far


@dataclass
class EnergyTables:
    """Self and pair energies over a pruned rotamer set."""

    positions: list[str]
    rotamers: dict[str, list[Rotamer]]
    self_energy: dict[tuple[str, int], float] = field(default_factory=dict)
    pair_energy: dict[tuple[str, int, str, int], float] = field(default_factory=dict)
    #: Diagnostics, reported so a surprising answer can be traced.
    stats: dict[str, float] = field(default_factory=dict)
    #: How many rotamers dead-end elimination removed at each position. Kept
    #: per position because a restraint that forbids every surviving rotamer
    #: needs to say whether pruning is why.
    pruned_per_position: dict[str, int] = field(default_factory=dict)
    #: Allowed identities that lost EVERY rotamer at a position, per position.
    #: Reported because this is the quiet way a design space shrinks: the spec
    #: offers 9 residues at A18 and `validate` advertises 6561 sequences, while
    #: TRP is not reachable there at all. Measured on the shipped smoke spec,
    #: TRP's best rotamer at A18 sits 51.9 kcal/mol above the best rotamer
    #: (MET, -7.06) on the screening scale -- a real clash with the UNRELAXED
    #: wild-type pocket, and exactly the residue the published design uses.
    dropped_identities: dict[str, list[str]] = field(default_factory=dict)

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
    max_self_energy: float | None = None,
    rotamer_window: float = 40.0,
    keep_per_identity: int = 3,
    pair_eps: float = 1e-3,
    energy_cutoff: float = 12.0,
    native: dict[str, str] | None = None,
    reference_energies: dict[str, float] | None = None,
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

    ``reference_energies`` adds a per-amino-acid constant to the self energy.
    This is a **Rosetta-style** correction, not an IPRO feature: Rosetta's
    ref2015 includes a ``ref`` term for unfolded-state propensity, which the
    PyRosetta backend therefore gets for free. The released IPRO source has no
    such per-residue-type term -- what it adds per rotamer is ``-ln(p)``, a
    rotamer *probability* penalty read from ``.prob`` files in its rotamer
    library, and that term is 0 for GLY and ALA, so it pushes toward small
    residues rather than away from them.

    Without any such constant, identities are decided by packing alone, which on
    a fixed backbone is biased against large side chains at buried positions: a
    cavity sized for the native residue has no room for a bigger one, and
    nothing in the energy rewards filling space. That bias matters most when the
    objective *is* the packing (``stabilize``); it matters much less when the
    objective is an interaction energy with a target, which is what IPRO
    optimizes.

    No default table is supplied here, because inventing one would put
    unsourced numbers at the centre of the objective. Supply your own (via
    ``goal_options.reference_energies``), or use the PyRosetta backend. If you
    leave it empty and see an all-small-residue answer, that is the expected
    behaviour and not a result.
    """
    native = native or {}
    reference_energies = reference_energies or {}
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
            if strain_weight:
                if rot.strain is None:
                    # Skipping it silently applied the strain penalty to some
                    # identities and not others: PyMOL emits no bump score for
                    # ALA (its library has no entry), so with strain_weight set,
                    # ALA alone competed without the penalty every other
                    # identity paid.
                    raise ValueError(
                        f"goal_options.strain_weight is set but rotamer "
                        f"{rot.aa}#{rot.state} at {pos} carries no strain value, "
                        f"so the penalty would apply to some identities and not "
                        f"others. PyMOL reports no bump score for every identity "
                        f"(ALA in particular); either leave strain_weight unset "
                        f"or drop that identity."
                    )
                e += strain_weight * rot.strain
            e += reference_energies.get(rot.aa, 0.0)
            scored.append((e, rot))

        # IPRO's rule, which is RELATIVE to the best rotamer at this position:
        # a conformer is competitive or it is not, and that question has no
        # absolute answer. An absolute threshold measured six rotamers at one
        # position with self energies 60, 62, 70, 75, 90, 99 -- a 39 kcal/mol
        # band, well inside IPRO's 40 kcal/mol window -- and kept ONE, the
        # native, via the safeguard below; IPRO keeps all six.
        #
        # Always keep at least one rotamer per position (else the program is
        # infeasible) and always keep the native identity if it was offered, so
        # "no mutation" stays reachable -- IPRO must be able to reject a
        # mutation.
        nat = native.get(pos)
        best_pair = min(scored, key=lambda t: t[0]) if scored else None
        best = best_pair[1] if best_pair else None
        floor = best_pair[0] if best_pair else 0.0

        # IPRO's rule is the window AND a per-kind retention floor: its
        # ROTAMERS.py keeps N=3 of each identity and only lowers N once the
        # total exceeds the cap. Implementing the window alone made the design
        # space SMALLER than the absolute threshold it replaced -- measured on
        # the shipped smoke spec, 115 rotamers kept with nothing dropped
        # became 95 with TRP and TYR gone from A18, PRO from A20 and PHE from
        # A38. A18/TRP is the published PoreDesigner answer (V18W), so the
        # "more faithful" rule deleted the one residue the example exists to
        # find. Its best rotamer sits 51.88 kcal/mol above the position's best
        # (MET, -7.06), which is a real clash with the unrelaxed wild-type
        # pocket -- and exactly why an identity must not be dropped outright
        # on the strength of a template it has not been relaxed against.
        by_identity: dict[str, list[float]] = {}
        for e, rot in scored:
            by_identity.setdefault(rot.aa, []).append(e)
        kept_floor = {
            aa: sorted(es)[: max(0, keep_per_identity)][-1:]
            for aa, es in by_identity.items()
        }
        floor_for = {aa: (v[0] if v else None) for aa, v in kept_floor.items()}

        for e, rot in scored:
            per_kind = floor_for.get(rot.aa)
            keep = (
                e <= floor + rotamer_window
                or rot is best
                or (nat and rot.aa == nat)
                or (per_kind is not None and e <= per_kind)
            )
            if keep and max_self_energy is not None:
                keep = (e <= max_self_energy or rot is best
                        or (nat and rot.aa == nat))
            if keep:
                kept.append(rot)
            else:
                n_pruned += 1
        tab.rotamers[pos] = kept
        tab.pruned_per_position[pos] = len(scored) - len(kept)
        offered = {r.aa for _e, r in scored}
        survived = {r.aa for r in kept}
        if offered - survived:
            tab.dropped_identities[pos] = sorted(offered - survived)
        # Keyed by object identity, not by (position, aa, state): a library
        # with a duplicate (aa, state) would otherwise collapse two distinct
        # rotamers onto one energy.
        by_id = {id(rot): e for e, rot in scored}
        for idx, rot in enumerate(kept):
            tab.self_energy[(pos, idx)] = by_id[id(rot)]

    # -- pair energies, with position-pair and term pruning ---------------
    n_pairs_possible = 0
    n_pairs_kept = 0
    n_pos_pairs_skipped = 0
    n_blocks_dropped = 0
    for a in range(len(positions)):
        for b in range(a + 1, len(positions)):
            pi, pj = positions[a], positions[b]
            Ri, Rj = tab.rotamers[pi], tab.rotamers[pj]
            if not Ri or not Rj:
                continue
            n_pairs_possible += len(Ri) * len(Rj)

            # Can any rotamer at pi reach any rotamer at pj? The radius has to
            # be measured about the SAME point the separation is measured from.
            # `bounding_radius()` is measured about each rotamer's own centroid,
            # which does not enclose it when seen from the mean centroid, so the
            # test was not conservative: measured with two one-atom rotamers 40 A
            # apart at one position, a real -0.014002 kcal/mol pair 6.0 A away
            # was dropped and the block recorded 0 pair terms. An enclosing
            # radius about the mean centroid cannot skip a block that has a
            # reachable pair in it.
            ci = np.mean([r.atoms.centroid() for r in Ri if len(r.atoms)] or [np.zeros(3)], axis=0)
            cj = np.mean([r.atoms.centroid() for r in Rj if len(r.atoms)] or [np.zeros(3)], axis=0)
            radi = _enclosing_radius(Ri, ci)
            radj = _enclosing_radius(Rj, cj)
            if np.linalg.norm(ci - cj) > radi + radj + energy_cutoff:
                n_pos_pairs_skipped += 1
                continue

            # A position-pair block is kept or dropped WHOLE. Dropping
            # individual terms within a block is unsound: the linearization
            # constraint sum_s w[i,r,j,s] = y[i,r] is only valid when every
            # rotamer s of position j has a w variable. With a partial block the
            # two orientations of that constraint force y[j,s'] = y[i,r] for the
            # surviving s', which makes the rotamers whose terms were dropped
            # unselectable -- the solver then returns a suboptimal assignment
            # and reports it as proven optimal, or declares a feasible program
            # infeasible.
            block: dict[tuple[str, int, str, int], float] = {}
            largest = 0.0
            for ri, rot_i in enumerate(Ri):
                for rj, rot_j in enumerate(Rj):
                    e = backend.interaction(rot_i.atoms, rot_j.atoms)
                    block[(pi, ri, pj, rj)] = e
                    largest = max(largest, abs(e))
            if largest < pair_eps:
                # Every interaction in this block is negligible, so the two
                # positions are effectively independent; dropping the whole
                # block emits no constraint for the pair at all, which is sound.
                n_blocks_dropped += 1
                continue
            tab.pair_energy.update(block)
            n_pairs_kept += len(block)

    tab.stats = {
        "rotamers_in": n_in,
        "rotamers_pruned": n_pruned,
        "rotamers_kept": sum(len(v) for v in tab.rotamers.values()),
        "position_pairs_skipped": n_pos_pairs_skipped,
        "negligible_blocks_dropped": n_blocks_dropped,
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


class Infeasible(RuntimeError):
    """No assignment satisfies the constraints.

    Distinct from a solver being missing or crashing, and deliberately *not*
    retried with a different solver: infeasibility is a property of the program,
    so substituting a heuristic would mean returning an assignment that violates
    the user's constraints or re-visits a sequence an integer cut forbade. The
    caller is expected to change the program -- a different window of positions,
    fewer cuts -- or to stop.
    """


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
        rotamer_vetoes: Iterable[tuple[str, int]] = (),
        pair_vetoes: Iterable[tuple[str, int, str, int]] = (),
        identity_links: Sequence[tuple[str, str]] = (),
        sequence_positions: Iterable[str] | None = None,
    ) -> None:
        self.t = tables
        self.constraints = list(constraints)
        self.native = dict(native or {})
        #: Rotamer choices forbidden outright, as (position, rotamer index).
        #: Produced by :mod:`ipro_agent.restraints` from geometric restraints,
        #: and emitted as y[i,r] = 0. Kept as plain sets rather than a richer
        #: object so that milp.py need not import the restraint machinery that
        #: imports it.
        self.rotamer_vetoes = {(str(p), int(i)) for p, i in rotamer_vetoes}
        #: Rotamer combinations forbidden together, emitted as
        #: y[i,r] + y[j,s] <= 1. That form is used rather than w[i,r,j,s] = 0
        #: because a negligible pair block has no w variables at all, while y
        #: always exists.
        self.pair_vetoes = {
            (str(pi), int(ri), str(pj), int(rj)) for pi, ri, pj, rj in pair_vetoes
        }
        #: Pairs of positions whose chosen *identity* must agree. Used for
        #: multiple binding assemblies, where the same sequence is packed
        #: separately in each: see :mod:`ipro_agent.assemblies`.
        self.identity_links = [(str(a), str(b)) for a, b in identity_links]
        #: Positions that sequence-level constraints (mutation counts, residue
        #: classes, require/forbid, integer cuts) are counted over. With
        #: several assemblies this is one representative set, because identity
        #: is coupled and counting every assembly would multiply each budget by
        #: the number of assemblies. Defaults to every position.
        self.sequence_positions = (
            set(tables.positions) if sequence_positions is None
            else {str(p) for p in sequence_positions}
        )
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
            if not vs:
                # Emitting no row for a rotamer-less position would silently
                # produce a program with no assignment constraint there, whose
                # "solution" leaves that position undefined.
                raise Infeasible(
                    f"design position {p} has no rotamers, so no residue can be "
                    "assigned there. Check that its allowed residues are valid "
                    "and that PyMOL could build them (scripts/doctor.py)."
                )
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

        # -- identity coupling across binding assemblies
        if self.identity_links:
            L.append(
                "\\ (12) identity is shared across assemblies: for each permitted "
                "amino acid, the\n\\     rotamers of that identity must be chosen in "
                "both positions or in neither"
            )
            n = 0
            for a, b in self.identity_links:
                if a not in t.rotamers or b not in t.rotamers:
                    continue
                for aa in sorted({r.aa for r in t.rotamers[a]} | {r.aa for r in t.rotamers[b]}):
                    left = [
                        f"+1 {self._y(a, i)}"
                        for i, r in enumerate(t.rotamers[a]) if r.aa == aa
                    ]
                    right = [
                        f"-1 {self._y(b, i)}"
                        for i, r in enumerate(t.rotamers[b]) if r.aa == aa
                    ]
                    if not left and not right:
                        continue
                    L.append(f" idlink{n}: " + _wrap_terms(left + right) + " = 0")
                    n += 1

        # -- restraint vetoes
        if self.rotamer_vetoes:
            L.append("\\ (10) restraint vetoes: forbidden rotamers")
            for p in t.positions:
                vs = [
                    self._y(p, r) for r in range(len(t.rotamers[p]))
                    if (p, r) in self.rotamer_vetoes
                ]
                if vs:
                    L.append(
                        f" veto_{self._pi[p]}: "
                        + _wrap_terms([f"+1 {v}" for v in vs]) + " = 0"
                    )
        if self.pair_vetoes:
            L.append("\\ (11) restraint vetoes: forbidden rotamer pairs")
            for n, (pi, ri, pj, rj) in enumerate(sorted(self.pair_vetoes)):
                if pi not in t.rotamers or pj not in t.rotamers:
                    continue
                L.append(
                    f" vpair{n}: +1 {self._y(pi, ri)} +1 {self._y(pj, rj)} <= 1"
                )

        # -- integer cuts over already-visited sequences
        if self.visited:
            L.append("\\ (9) integer cuts: forbid sequences already evaluated")
        for n, assignment in enumerate(self.visited):
            vs = self._cut_vars(assignment)
            if not vs:
                continue
            # The right-hand side counts the positions actually constrained, not
            # the length of the caller's dict: a visited assignment naming a
            # position absent from these tables would otherwise leave the cut
            # slack by one and let the sequence come back.
            covered = len({p for p in assignment if p in self.t.rotamers})
            L.append(
                f" cut{n}: " + _wrap_terms([f"+1 {v}" for v in vs])
                + f" <= {covered - 1}"
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
                for p in self._seq_positions()
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
                for p in self._seq_positions()
                for r, rot in enumerate(t.rotamers[p])
                if rot.aa in want
            ]
            rel = ">=" if con.kind == "min_class" else "<="
            short = ",".join(sorted(THREE_TO_ONE.get(a, a) for a in want))
            out.append(f"\\ (6) {con.kind}: {rel} {k} positions from {{{short}}}")
            if vs:
                out.append(f" {tag}: " + _wrap_terms([f"+1 {v}" for v in vs]) + f" {rel} {k}")
            elif con.kind == "min_class" and k > 0:
                raise Infeasible(
                    f"constraint {tag} requires at least {k} position(s) with an "
                    f"identity in {{{short}}}, but no rotamer at any design "
                    "position has one. Widen the allowed residues, or drop the "
                    "constraint."
                )

        elif con.kind in ("require", "forbid"):
            want = con.residue_set()
            for p in (con.positions or self._seq_positions()):
                if p not in t.rotamers:
                    continue
                vs = [
                    self._y(p, r)
                    for r, rot in enumerate(t.rotamers[p])
                    if rot.aa in want
                ]
                if not vs:
                    if con.kind == "require":
                        raise Infeasible(
                            f"constraint {tag} requires {p} to take an identity in "
                            f"{sorted(want)}, but no rotamer there has one. Add it "
                            f"to that position's allowed residues."
                        )
                    continue   # 'forbid' with nothing to forbid is satisfied
                rel, rhs = ("= 1", 1) if con.kind == "require" else ("= 0", 0)
                out.append(f"\\ (7) {con.kind} at {p}: {sorted(want)}")
                out.append(f" {tag}_{self._pi[p]}: " + _wrap_terms([f"+1 {v}" for v in vs]) + f" {rel}")

        elif con.kind == "max_energy":
            cap = con.resolve_value(npos)
            terms: list[str] = []
            for p in t.positions:
                for r in range(len(t.rotamers[p])):
                    c = t.self_energy.get((p, r), 0.0)
                    if c:
                        terms.append(f"{c:+.6g} {self._y(p, r)}")
            for (pi, r, pj, s), c in sorted(t.pair_energy.items()):
                if c:
                    terms.append(f"{c:+.6g} {self._w(pi, r, pj, s)}")
            out.append(f"\\ (8) IPRO stability constraint: assignment energy <= {cap:.6g}")
            if terms:
                out.append(f" {tag}: " + _wrap_terms(terms) + f" <= {cap:.6g}")

        else:
            out.append(f"\\ WARNING: unknown constraint kind {con.kind!r}, ignored")
        return out

    def _identity_groups(self) -> list[list[str]]:
        """Positions tied together by identity links, as connected components."""
        if not self.identity_links:
            return []
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in self.identity_links:
            if a not in self.t.rotamers or b not in self.t.rotamers:
                continue
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        groups: dict[str, list[str]] = {}
        for x in list(parent):
            groups.setdefault(find(x), []).append(x)
        return [sorted(v) for v in groups.values() if len(v) > 1]

    def _seq_positions(self) -> list[str]:
        """Positions that sequence-level constraints count over, in order."""
        return [p for p in self.t.positions if p in self.sequence_positions]

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
            lp_path.write_text(lp_text, encoding="utf-8")

        # A named solver must actually be tried. Previously any name that was
        # not "pulp" or "binary" fell through to the greedy heuristic, so every
        # documented choice ("cbc", "glpk", "cplex", "gurobi", "highs") and
        # every typo silently ran the heuristic and reported a design that no
        # solver had endorsed.
        want = (solver or "auto").strip().lower()
        if want in ("auto", ""):
            order, prefer = ["pulp", "binary", "greedy"], None
        elif want == "greedy":
            order, prefer = ["greedy"], None
        elif want in ("pulp", "binary"):
            order, prefer = [want], None
        elif want in NAMED_SOLVERS:
            # Either route can drive these; PuLP first, then an LP file.
            order, prefer = ["pulp", "binary"], want
        else:
            raise ValueError(
                f"unknown milp_solver {solver!r}. Choose 'auto', 'greedy', "
                f"'pulp', 'binary', or one of: {', '.join(sorted(NAMED_SOLVERS))}"
            )
        last_err: Exception | None = None
        for which in order:
            try:
                if which == "pulp":
                    sol = self._solve_pulp(time_limit, prefer=prefer)
                elif which == "binary":
                    sol = self._solve_binary(lp_text, time_limit, prefer=prefer)
                else:
                    sol = self._solve_greedy()
                sol.lp_path = lp_path
                sol.seconds = time.time() - t0
                return sol
            except Infeasible:
                # Propagate immediately: see Infeasible's docstring. Trying
                # another solver, or the heuristic, would return an assignment
                # that breaks the constraints we were asked to honour.
                raise
            except Exception as exc:  # try the next option
                last_err = exc
                log.info("MILP solver %r unavailable or failed: %s", which, exc)
        raise RuntimeError(f"no MILP solver succeeded; last error: {last_err}")

    # -- solver implementations -------------------------------------------

    def _solve_pulp(self, time_limit: float | None, prefer: str | None = None) -> MILPSolution:
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
        for a, b in self.identity_links:
            if a not in t.rotamers or b not in t.rotamers:
                continue
            for aa in sorted({r.aa for r in t.rotamers[a]} | {r.aa for r in t.rotamers[b]}):
                left = [y[(a, i)] for i, r in enumerate(t.rotamers[a]) if r.aa == aa]
                right = [y[(b, i)] for i, r in enumerate(t.rotamers[b]) if r.aa == aa]
                if left or right:
                    prob += pulp.lpSum(left) == pulp.lpSum(right)
        for (p, r) in self.rotamer_vetoes:
            if (p, r) in y:
                prob += y[(p, r)] == 0
        for (pi, ri, pj, rj) in self.pair_vetoes:
            if (pi, ri) in y and (pj, rj) in y:
                prob += y[(pi, ri)] + y[(pj, rj)] <= 1
        for assignment in self.visited:
            vs = [
                y[(p, r)]
                for p, aa in assignment.items()
                if p in t.rotamers
                for r, rot in enumerate(t.rotamers[p])
                if rot.aa == aa
            ]
            if vs:
                covered = len({p for p in assignment if p in t.rotamers})
                prob += pulp.lpSum(vs) <= covered - 1

        solver_obj = None
        # Order: commercial first (IPRO itself used CPLEX), then open solvers.
        # HiGHS is included because PuLP 4.x no longer bundles CBC, so a modern
        # `pip install pulp` may have HiGHS and nothing else.
        candidates = (
            (NAMED_SOLVERS[prefer][0],) if prefer else (
                "CPLEX_CMD", "GUROBI_CMD", "PULP_CBC_CMD", "COIN_CMD",
                "HiGHS", "HiGHS_CMD", "SCIP_CMD", "GLPK_CMD",
            )
        )
        for name in candidates:
            if name in pulp.listSolvers(onlyAvailable=True):
                kw = {"msg": False}
                if time_limit:
                    kw["timeLimit"] = time_limit
                solver_obj = getattr(pulp, name)(**kw)
                break
        if solver_obj is None:
            raise RuntimeError(
                f"pulp cannot provide solver {prefer!r}" if prefer
                else "pulp has no available solver"
            )
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
                ) <= con.resolve_value(npos)
                continue
            if con.kind == "max_mutations":
                k = con.resolve_count(npos)
                vs = [
                    y[(p, r)]
                    for p in self._seq_positions()
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
                    for p in self._seq_positions()
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
                for p in (con.positions or self._seq_positions()):
                    if p not in t.rotamers:
                        continue
                    vs = [
                        y[(p, r)]
                        for r, rot in enumerate(t.rotamers[p])
                        if rot.aa in want
                    ]
                    if vs:
                        prob += pulp.lpSum(vs) == (1 if con.kind == "require" else 0)

    def _solve_binary(
        self, lp_text: str, time_limit: float | None, prefer: str | None = None
    ) -> MILPSolution:
        """Solve by handing an LP file to a solver binary (glpsol or cbc).

        Only these two have solution formats this code parses; a user who names
        another solver gets a clear error rather than a heuristic answer.
        """
        if prefer:
            binary = NAMED_SOLVERS[prefer][1]
            if binary is None:
                raise RuntimeError(
                    f"solver {prefer!r} cannot be driven through an LP file here "
                    "(its solution format is not parsed); install pulp to use it"
                )
            if shutil.which(binary) is None:
                raise RuntimeError(f"solver binary {binary!r} for {prefer!r} not on PATH")
            glpsol = shutil.which(binary) if binary == "glpsol" else None
            cbc = shutil.which(binary) if binary == "cbc" else None
        else:
            glpsol = shutil.which("glpsol")
            cbc = shutil.which("cbc")
        if not (glpsol or cbc):
            raise RuntimeError("no solver binary (glpsol, cbc) on PATH")

        with tempfile.TemporaryDirectory(prefix="ipro_milp_") as td:
            lp = Path(td) / "model.lp"
            out = Path(td) / "model.sol"
            lp.write_text(lp_text, encoding="utf-8")
            if glpsol:
                cmd = [glpsol, "--lp", str(lp), "-o", str(out)]
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
            text = out.read_text(encoding="utf-8")
            vals, status = (
                _parse_glpk_solution(text) if name == "glpsol" else _parse_cbc_solution(text)
            )

        if not vals:
            raise RuntimeError(f"{name} solution file had no variable values")

        # A solver that proves infeasibility still writes a solution file, with
        # every variable at zero. Trusting it would invent a design the solver
        # never endorsed (the first rotamer at every position), so reject any
        # status that does not denote a usable solution...
        low = status.lower()
        for bad in ("infeasible", "no primal", "empty"):
            if bad in low:
                raise Infeasible(f"{name} proved the program infeasible (status {status!r})")
        for bad in ("undefined", "unbounded"):
            if bad in low:
                raise RuntimeError(f"{name} reported status {status!r}")

        # ...and verify structurally that what came back really is an
        # assignment: exactly one selected rotamer per position. This catches a
        # silently truncated or misparsed solution file, which a status string
        # cannot.
        choices: dict[str, int] = {}
        for p in self.t.positions:
            col = [vals.get(self._y(p, r), 0.0) for r in range(len(self.t.rotamers[p]))]
            if not col:
                raise RuntimeError(f"no rotamer variables for position {p}")
            total = sum(col)
            if abs(total - 1.0) > 1e-6:
                raise RuntimeError(
                    f"{name} returned a non-assignment at {p}: selected weight "
                    f"{total:.6g}, expected 1 (status was {status!r})"
                )
            choices[p] = max(range(len(col)), key=lambda i: col[i])
        obj = self.evaluate(choices)
        # glpsol reports "INTEGER OPTIMAL" for a MIP and "OPTIMAL" for an LP;
        # CBC reports "Optimal". Anything else (time limit, feasible-only) must
        # not be reported as proven optimal.
        s = status.lower().replace("-", " ")
        # "INTEGER NON-OPTIMAL" contains "optimal"; so does "not optimal".
        proven = "optimal" in s and not any(
            w in s for w in ("non optimal", "not optimal", "suboptimal", "sub optimal")
        )
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
        def allowed_at(p: str) -> list[int]:
            return [
                r for r in range(len(t.rotamers[p]))
                if (p, r) not in self.rotamer_vetoes
            ] or list(range(len(t.rotamers[p])))

        choices: dict[str, int] = {}
        groups = self._identity_groups()
        if groups:
            # With identity coupling, choosing each position independently
            # breaks the link immediately. Pick one identity per group -- the
            # one whose best packing is cheapest summed over the group -- then
            # take each member's cheapest rotamer of that identity. This starts
            # feasible instead of relying on repair.
            for members in groups:
                shared = set.intersection(*(
                    {r.aa for i, r in enumerate(t.rotamers[m]) if i in set(allowed_at(m))}
                    for m in members
                )) or set.intersection(*({r.aa for r in t.rotamers[m]} for m in members))
                best_aa, best_cost, best_pick = None, None, {}
                for aa in sorted(shared):
                    cost, pick = 0.0, {}
                    ok = True
                    for m in members:
                        cands = [
                            (t.self_energy.get((m, i), 0.0), i)
                            for i in allowed_at(m) if t.rotamers[m][i].aa == aa
                        ]
                        if not cands:
                            ok = False
                            break
                        e, i = min(cands)
                        cost += e
                        pick[m] = i
                    if ok and (best_cost is None or cost < best_cost):
                        best_aa, best_cost, best_pick = aa, cost, pick
                if best_aa is None:
                    for m in members:
                        choices[m] = min(allowed_at(m), key=lambda r: t.self_energy.get((m, r), 0.0))
                else:
                    choices.update(best_pick)
        for p in t.positions:
            if p not in choices:
                choices[p] = min(allowed_at(p), key=lambda r: t.self_energy.get((p, r), 0.0))

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
        if not self._feasible(choices):
            # Returning an infeasible assignment would mean handing back a
            # design that violates the user's constraints, or one an integer
            # cut already forbade. Say so instead.
            raise Infeasible(
                "the built-in heuristic could not find an assignment satisfying "
                "the constraints (it does not prove infeasibility; install a MILP "
                "solver for a definitive answer)"
            )
        return MILPSolution(
            choices=choices,
            objective=self.evaluate(choices),
            status="heuristic",
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
        for p, r in choices.items():
            if (p, r) in self.rotamer_vetoes:
                return False
        for (pi, ri, pj, rj) in self.pair_vetoes:
            if choices.get(pi) == ri and choices.get(pj) == rj:
                return False
        for a, b in self.identity_links:
            if a in choices and b in choices:
                if t.rotamers[a][choices[a]].aa != t.rotamers[b][choices[b]].aa:
                    return False
        ids = {
            p: t.rotamers[p][r].aa for p, r in choices.items()
            if p in self.sequence_positions
        }
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
                for p in (con.positions or self._seq_positions()):
                    if p in ids and ids[p] not in want:
                        return False
            elif con.kind == "forbid":
                want = con.residue_set()
                for p in (con.positions or self._seq_positions()):
                    if p in ids and ids[p] in want:
                        return False
            elif con.kind == "max_energy":
                if self.evaluate(choices) > con.resolve_value(npos):
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

        # Identity coupling first: it is structural, and leaving it broken makes
        # every other repair meaningless.
        for a, b in self.identity_links:
            if a not in choices or b not in choices:
                continue
            if t.rotamers[a][choices[a]].aa == t.rotamers[b][choices[b]].aa:
                continue
            best: tuple[float, str, int] | None = None
            for src, dst in ((a, b), (b, a)):
                want = t.rotamers[src][choices[src]].aa
                for k, rot in enumerate(t.rotamers[dst]):
                    if rot.aa != want or (dst, k) in self.rotamer_vetoes:
                        continue
                    cost = t.self_energy.get((dst, k), 0.0)
                    if best is None or cost < best[0]:
                        best = (cost, dst, k)
            if best is None:
                return False
            choices[best[1]] = best[2]
            return True

        # Vetoes first: they are hard and cheap to repair.
        for p, r in list(choices.items()):
            if (p, r) in self.rotamer_vetoes:
                alts = [
                    (t.self_energy.get((p, k), 0.0), k)
                    for k in range(len(t.rotamers[p]))
                    if (p, k) not in self.rotamer_vetoes
                ]
                if not alts:
                    return False
                choices[p] = min(alts)[1]
                return True
        for (pi, ri, pj, rj) in sorted(self.pair_vetoes):
            if choices.get(pi) != ri or choices.get(pj) != rj:
                continue
            # Move one of the two positions, but only to a rotamer that is not
            # itself vetoed *against the other position's current choice*.
            # Picking merely the cheapest alternative lands on another forbidden
            # pair whenever many pairs are excluded, and the repair then stalls
            # on a problem that is in fact feasible.
            best: tuple[float, str, int] | None = None
            for p, other, other_r in ((pi, pj, rj), (pj, pi, ri)):
                for k in range(len(t.rotamers[p])):
                    if k == choices[p] or (p, k) in self.rotamer_vetoes:
                        continue
                    a, b = (p, k), (other, other_r)
                    forbidden = (
                        (a[0], a[1], b[0], b[1]) in self.pair_vetoes
                        or (b[0], b[1], a[0], a[1]) in self.pair_vetoes
                    )
                    if forbidden:
                        continue
                    cost = t.self_energy.get((p, k), 0.0)
                    if best is None or cost < best[0]:
                        best = (cost, p, k)
            if best is None:
                return False
            choices[best[1]] = best[2]
            return True
        for con in self.constraints:
            if con.kind == "min_class":
                want = con.residue_set()
                k = con.resolve_count(npos)
                have = sum(
                    1 for p, r in choices.items()
                    if p in self.sequence_positions and t.rotamers[p][r].aa in want
                )
                if have >= k:
                    continue
                # Switch the cheapest position that can take a wanted identity.
                cands = []
                for p in self._seq_positions():
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
                for p in (con.positions or self._seq_positions()):
                    if p in choices and t.rotamers[p][choices[p]].aa not in want:
                        for r, rot in enumerate(t.rotamers[p]):
                            if rot.aa in want:
                                choices[p] = r
                                return True
            if con.kind == "forbid":
                want = con.residue_set()
                for p in (con.positions or self._seq_positions()):
                    if p in choices and t.rotamers[p][choices[p]].aa in want:
                        alts = [(t.self_energy.get((p, r), 0.0), r)
                                for r, rot in enumerate(t.rotamers[p]) if rot.aa not in want]
                        if alts:
                            choices[p] = min(alts)[1]
                            return True
            if con.kind == "max_class":
                # Too many positions in the class: move the costliest member
                # out to its cheapest non-member rotamer.
                want = con.residue_set()
                k = con.resolve_count(npos)
                members = [
                    p for p, r in choices.items()
                    if p in self.sequence_positions and t.rotamers[p][r].aa in want
                ]
                if len(members) <= k:
                    continue
                best: tuple[float, str, int] | None = None
                for p in members:
                    alts = [
                        (t.self_energy.get((p, r), 0.0), p, r)
                        for r, rot in enumerate(t.rotamers[p])
                        if rot.aa not in want
                    ]
                    if alts:
                        cand = min(alts)
                        if best is None or cand[0] < best[0]:
                            best = cand
                if best is None:
                    return False
                choices[best[1]] = best[2]
                return True

            if con.kind == "max_energy":
                # Nothing to swap *to*: the violation is the objective value
                # itself. Take the single-position change that lowers it most,
                # which is progress toward the cap even while still above it.
                if self.evaluate(choices) <= con.resolve_value(npos):
                    continue
                cur = self.evaluate(choices)
                best_move: tuple[float, str, int] | None = None
                for p in t.positions:
                    keep = choices[p]
                    for r in range(len(t.rotamers[p])):
                        if r == keep:
                            continue
                        choices[p] = r
                        e = self.evaluate(choices)
                        if e < cur - 1e-12 and (best_move is None or e < best_move[0]):
                            best_move = (e, p, r)
                    choices[p] = keep
                if best_move is None:
                    return False
                choices[best_move[1]] = best_move[2]
                return True

            if con.kind == "max_mutations":
                k = con.resolve_count(npos)
                muts = [p for p, r in choices.items()
                        if p in self.sequence_positions
                        and self.native.get(p) and t.rotamers[p][r].aa != self.native[p]]
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
    for line in text.splitlines():
        toks = line.split()
        if not toks:
            continue   # a leading blank line must not lose the status
        if not toks[0].isdigit():
            if status == "unknown":
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
