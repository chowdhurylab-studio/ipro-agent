"""Turning geometric restraints into constraints the MILP can honour.

IPRO enforces four kinds of atom-position restraint "rigorously ... throughout
all functions". The useful observation for this implementation is that, during
rotamer selection, a restraint does not need a new kind of mathematics: the
only coordinates that change are side-chain atoms at design positions, so a
restraint is a **veto** on rotamer choices.

How each restraint is classified, by how many *design positions* its atoms
depend on:

===================  ==================================================
positions involved   becomes
===================  ==================================================
0                    nothing to decide. The quantity is fixed by the
                     input structure, so it is measured once and
                     reported; a violation is the input's, not the
                     design's.
1                    a **rotamer veto**: every rotamer at that position
                     whose geometry falls outside the interval is
                     forbidden (``y[i,r] = 0``).
2                    a **pair veto**: every combination of rotamers at
                     the two positions that falls outside the interval
                     is forbidden (``y[i,r] + y[j,s] <= 1``).
3 or more            **not expressible** pairwise. Reported, never
                     silently dropped.
===================  ==================================================

This is exact, not approximate: the vetoed assignments are precisely those that
violate the restraint, so the MILP optimum is the best design that satisfies it.

Two consequences worth knowing. A restraint naming a side-chain atom that some
residues do not have (``A20:CG`` with ALA or GLY among the allowed residues)
vetoes those rotamers, so the restraint quietly narrows the design space to
residue types that *have* that atom -- which is reported rather than left to be
discovered. And a ``fix`` or ``tether`` on a backbone atom cannot be enforced by
rotamer choice at all; in ``full`` mode it is handed to the relaxation backend,
and in ``fixed-backbone`` mode it is already satisfied because the backbone
never moves.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .config import AtomRef, Restraint, RestraintKind, SpecError
from .milp import EnergyTables, Rotamer
from .structure import Structure

log = logging.getLogger(__name__)

BACKBONE = {"N", "CA", "C", "O", "OXT"}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def dihedral(p0, p1, p2, p3) -> float:
    """Dihedral angle p0-p1-p2-p3 in degrees, in (-180, 180].

    Standard construction: project the two outer bonds onto the plane normal to
    the central bond and take the signed angle between them.
    """
    p0, p1, p2, p3 = (np.asarray(p, float) for p in (p0, p1, p2, p3))
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    n = np.linalg.norm(b1)
    if n < 1e-9:
        raise ValueError("dihedral is undefined: the central atoms coincide")
    b1 = b1 / n
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    if np.linalg.norm(v) < 1e-9 or np.linalg.norm(w) < 1e-9:
        raise ValueError("dihedral is undefined: three atoms are collinear")
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1, v), w))
    return float(np.degrees(np.arctan2(y, x)))


def angular_deviation(value: float, target: float) -> float:
    """Shortest absolute separation between two angles, in degrees."""
    d = (float(value) - float(target) + 180.0) % 360.0 - 180.0
    return abs(d)


# ---------------------------------------------------------------------------
# Atom resolution
# ---------------------------------------------------------------------------


class RestraintError(RuntimeError):
    """A restraint names something that cannot be found or measured."""


@dataclass
class _Resolved:
    """Where an atom's coordinate comes from."""

    ref: AtomRef
    #: Set when the coordinate is fixed by the input structure.
    fixed: np.ndarray | None = None
    #: Set when the coordinate depends on the rotamer chosen at this position.
    position: str | None = None

    @property
    def is_rotamer(self) -> bool:
        return self.position is not None


def _fixed_atom_index(structures: Iterable[Structure]) -> dict[tuple, np.ndarray]:
    """Map ``(chain, number, icode, atom name)`` to a coordinate.

    A restraint names an atom by residue and atom name only -- there is no way
    to say *which molecule* -- so if a partner reuses the design molecule's
    chain id the reference is genuinely ambiguous. This used to keep whichever
    structure came first, so ``A16:CA`` could silently be measured on a
    molecule 50 A from the one the user meant, and every veto derived from it
    was wrong. It now refuses: ambiguity in the input is not something to
    resolve by ordering.
    """
    out: dict[tuple, np.ndarray] = {}
    clashes: set[tuple] = set()
    for st in structures:
        if st is None:
            continue
        for res in list(st.residues()) + list(st.hetero_residues()):
            ref = st.ref(res)
            for at in res:
                key = (ref.chain, ref.number, ref.insertion_code, at.get_name().strip().upper())
                coord = np.asarray(at.coord, dtype=float)
                if key in out:
                    if not np.allclose(out[key], coord, atol=1e-6):
                        clashes.add(key)
                    continue
                out[key] = coord
    if clashes:
        shown = ", ".join(
            f"{c}{n}{i}:{a}" for c, n, i, a in sorted(clashes, key=str)[:6]
        )
        raise RestraintError(
            f"{len(clashes)} atom reference(s) are ambiguous because more than one "
            f"molecule uses the same chain id and residue number: {shown}"
            + (" ..." if len(clashes) > 6 else "")
            + ". A restraint names an atom by chain, residue and atom name only, "
            "so it cannot say which molecule is meant. Give the partner molecules "
            "distinct chain ids (which the output PDB needs anyway) and re-run."
        )
    return out


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@dataclass
class RestraintPlan:
    """What the restraints forbid, and what they could not cover."""

    #: ``(position, rotamer index)`` pairs that no design may use.
    rotamer_vetoes: set[tuple[str, int]] = field(default_factory=set)
    #: ``(pos_i, rot_i, pos_j, rot_j)`` combinations that may not co-occur.
    pair_vetoes: set[tuple[str, int, str, int]] = field(default_factory=set)
    #: Restraints the input structure already violates, as readable lines.
    native_violations: list[str] = field(default_factory=list)
    #: Restraints that could not be turned into constraints, and why.
    unsupported: list[str] = field(default_factory=list)
    #: Design-position labels whose backbone must be held during relaxation.
    relax_fixed_positions: list[str] = field(default_factory=list)
    #: Restraints that no rotamer choice can affect, kept live rather than
    #: dropped. The ACS Catal 2017 thioesterase work preserved catalysis with
    #: exactly these -- a distance between two non-design atoms, held to
    #: +/-0.2 A -- and they were reported as "nothing to decide" and then
    #: enforced nowhere, although the perturbation and the redock both move
    #: them. Each entry is (label, atom refs, lo, hi, unit, strict).
    geometry_only: list[tuple] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: Per-restraint summary, for reporting.
    summary: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.rotamer_vetoes or self.pair_vetoes)


def plan_restraints(
    restraints: Sequence[Restraint],
    tables: EnergyTables,
    design: Structure | None,
    partners: Iterable[Structure] = (),
) -> RestraintPlan:
    """Work out which rotamer choices each restraint forbids.

    Raises :class:`RestraintError` if a restraint names an atom that exists
    nowhere, or if it forbids every rotamer at some position (which would make
    the program infeasible for a reason the user should hear stated plainly
    rather than as "no solution").
    """
    plan = RestraintPlan()
    if not restraints:
        return plan

    fixed = _fixed_atom_index([design, *partners])
    design_positions = set(tables.positions)

    for n, r in enumerate(restraints):
        label = r.label or f"{r.kind.value}#{n + 1}"
        try:
            _plan_one(r, label, plan, tables, fixed, design_positions)
        except RestraintError:
            raise
        except Exception as exc:   # a bad restraint must name itself
            raise RestraintError(f"restraint {label}: {exc}") from exc

    # Refuse early, with the restraint named, rather than leaving the solver to
    # report an unexplained infeasibility.
    for pos in tables.positions:
        n_rot = len(tables.rotamers[pos])
        vetoed = sum(1 for (p, i) in plan.rotamer_vetoes if p == pos)
        if n_rot and vetoed >= n_rot:
            pruned = int(tables.pruned_per_position.get(pos, 0))
            extra = ""
            if pruned:
                # Pruning runs before restraints are applied, so the rotamers
                # that would have satisfied the restraint may already be gone.
                # Saying "0 of 1 survive" without this is baffling when the user
                # knows the residue has many rotamers.
                extra = (
                    f" Note that the rotamer prune had already discarded "
                    f"{pruned} rotamer(s) at {pos} for scoring far worse than "
                    f"the best one there, leaving only {n_rot}; if the "
                    f"restraint can only be met by one of those, raise "
                    f"goal_options.rotamer_window to keep more of them, and "
                    f"treat the conflict between the restraint and the "
                    f"structure as the real finding."
                )
            raise RestraintError(
                f"the restraints forbid every rotamer at {pos} ({vetoed} of {n_rot} "
                f"surviving). Loosen a tolerance, widen that position's allowed "
                f"residues, or drop the restraint.{extra}"
            )
    return plan


def check_restraints(
    restraints: Sequence[Restraint],
    design: Structure | None,
    partners: Iterable[Structure] = (),
    design_positions: Iterable[str] = (),
) -> tuple[list[str], list[str]]:
    """Cheap pre-flight: do the named atoms exist, and what can be measured now?

    Returns ``(problems, observations)``. Unlike :func:`plan_restraints` this
    needs no rotamer library, so ``ipro-agent validate`` can run it in a second
    and catch a misspelled atom name before anything expensive starts.
    """
    problems: list[str] = []
    observations: list[str] = []
    if not restraints:
        return problems, observations

    fixed = _fixed_atom_index([design, *partners])
    dps = set(design_positions)

    for n, r in enumerate(restraints):
        label = r.label or f"{r.kind.value}#{n + 1}"
        moving: list[AtomRef] = []
        coords: list[np.ndarray] = []
        bad = False
        for ref in r.refs:
            if ref.position in dps and ref.atom not in BACKBONE:
                moving.append(ref)
                continue
            try:
                res = _resolve(ref, label, EnergyTables(positions=[], rotamers={}), fixed, dps)
            except RestraintError as exc:
                problems.append(str(exc))
                bad = True
                continue
            coords.append(res.fixed)
        if bad:
            continue
        n_pos = len({m.position for m in moving})
        if n_pos >= 3:
            problems.append(
                f"restraint {label} depends on {n_pos} design positions "
                f"({', '.join(sorted({m.position for m in moving}))}); restraints "
                "spanning three or more design positions cannot be enforced"
            )
            continue
        if not moving:
            if len(coords) == 1:
                # A fix or tether on an atom no design position can move. It
                # cannot be violated: the atom is part of the fixed template,
                # so its displacement is identically zero. Running it through
                # the multi-atom measurement reported a perfectly valid
                # backbone `fix` as a problem and made `validate` fail.
                observations.append(
                    f"{label}: {r.refs[0]} is held fixed and no design position "
                    f"can move it, so the restraint is satisfied by construction"
                )
                continue
            try:
                value = (
                    distance(coords[0], coords[1])
                    if r.kind is RestraintKind.DISTANCE else dihedral(*coords)
                )
            except Exception as exc:
                problems.append(f"restraint {label}: cannot be measured ({exc})")
                continue
            lo, hi = r.interval()
            ok = (
                angular_deviation(value, float(r.target)) <= float(r.tolerance)
                if r.kind is RestraintKind.DIHEDRAL
                else lo - 1e-9 <= value <= hi + 1e-9
            )
            line = (
                f"{label}: {value:.3f} {r.unit} in the input structure, "
                f"{'satisfied' if ok else 'VIOLATED'} (no design position involved, "
                "so no choice of rotamer can change it)"
            )
            (observations if ok else problems).append(line)
        else:
            observations.append(
                f"{label}: depends on {n_pos} design position(s) "
                f"({', '.join(sorted({m.position for m in moving}))}); will be "
                f"enforced as {'rotamer' if n_pos == 1 else 'pair'} vetoes"
            )
            # A side-chain atom that only some residue types have silently
            # narrows the design space; say so before the run, not after.
            for m in moving:
                if m.atom not in ("CB",):
                    observations.append(
                        f"  note: {label} names {m}, so only residues that have an "
                        f"atom called {m.atom} can be chosen at {m.position}"
                    )
    return problems, observations


def _plan_one(
    r: Restraint,
    label: str,
    plan: RestraintPlan,
    tables: EnergyTables,
    fixed: dict[tuple, np.ndarray],
    design_positions: set[str],
) -> None:
    lo, hi = r.interval()

    if r.kind is RestraintKind.DIHEDRAL and (r.minimum is not None or r.maximum is not None):
        # A one-sided bound on a circular quantity has no well-defined meaning
        # without saying which way round; refuse instead of guessing.
        raise RestraintError(
            f"restraint {label}: a dihedral takes 'target' with 'tolerance', not "
            "'minimum'/'maximum' -- a one-sided bound on an angle is ambiguous"
        )

    resolved = [_resolve(ref, label, tables, fixed, design_positions) for ref in r.refs]
    positions = sorted({x.position for x in resolved if x.is_rotamer})

    # -- displacement restraints (fix / tether) ---------------------------
    if r.is_displacement:
        only = resolved[0]
        if not only.is_rotamer:
            plan.summary.append(
                f"{label}: satisfied by construction (atom {only.ref} does not move)"
            )
            # A backbone atom of a design position still matters once a
            # relaxing backend is in play.
            if only.ref.position in design_positions:
                plan.relax_fixed_positions.append(only.ref.position)
                plan.notes.append(
                    f"{label} names backbone atom {only.ref}; it is held during "
                    "relaxation in 'full' mode and is already fixed in "
                    "'fixed-backbone' mode"
                )
            return
        start = fixed.get(
            (only.ref.chain, only.ref.number, only.ref.insertion_code, only.ref.atom)
        )
        if start is None:
            raise RestraintError(
                f"restraint {label}: atom {only.ref} is a side chain being designed and "
                "is not present in the input structure, so there is no starting "
                "position to measure displacement from"
            )
        n_veto = _veto_single(
            plan, tables, only.position, label, lo, hi,
            lambda rot: distance(_rotamer_coord(rot, only.ref, label), start),
        )
        plan.summary.append(
            f"{label}: displacement of {only.ref} in [{lo:g}, {hi:g}] A "
            f"-> {n_veto} rotamer(s) vetoed"
        )
        return

    # -- measurements over several atoms ----------------------------------
    def measure(coords: list[np.ndarray]) -> float:
        if r.kind is RestraintKind.DISTANCE:
            return distance(coords[0], coords[1])
        if len(coords) == 1:
            # A fix/tether whose atom nothing can move: displacement is zero.
            return 0.0
        return dihedral(*coords)

    def within(value: float) -> bool:
        if r.kind is RestraintKind.DIHEDRAL:
            # Circular comparison about the target.
            return angular_deviation(value, float(r.target)) <= float(r.tolerance)
        return lo - 1e-9 <= value <= hi + 1e-9

    if len(positions) == 0:
        coords = [x.fixed for x in resolved]
        value = measure(coords)
        ok = within(value)
        # Recorded, not discarded. No ROTAMER choice can change this, which is
        # all the old message said -- but the backbone perturbation and the
        # rigid-body redock both can, so the run has to keep measuring it.
        plan.geometry_only.append(
            (label, [x.ref for x in resolved], lo, hi, r.unit, bool(r.strict),
             r.kind.value, None if r.target is None else float(r.target),
             None if r.tolerance is None else float(r.tolerance))
        )
        plan.summary.append(
            f"{label}: {value:.3f} {r.unit} in the input structure "
            f"({'satisfied' if ok else 'VIOLATED'}); no rotamer choice can "
            f"change it, so it is re-measured after each perturbation and "
            f"redock instead"
        )
        if not ok:
            # "the input structure" only on the first plan: the loop re-plans
            # after each perturbation and each redock, and a violation found
            # then belongs to the candidate, not to the user's file.
            msg = (
                f"{label} is violated by the coordinates as they stand: "
                f"measured {value:.3f} {r.unit}, wanted [{lo:g}, {hi:g}]. No "
                f"choice of rotamer can change it -- only the backbone "
                f"perturbation or the partner's placement can, so the "
                f"iteration is rejected rather than repacked."
            )
            if r.strict:
                raise RestraintError(msg)
            plan.native_violations.append(msg)
        return

    if len(positions) == 1:
        pos = positions[0]
        n_veto = _veto_single(
            plan, tables, pos, label, lo, hi,
            lambda rot: measure([
                _rotamer_coord(rot, x.ref, label) if x.is_rotamer else x.fixed
                for x in resolved
            ]),
            predicate=within,
        )
        plan.summary.append(
            f"{label}: {r.kind.value} depends on {pos} alone -> {n_veto} rotamer(s) vetoed"
        )
        return

    if len(positions) == 2:
        pi, pj = positions
        n_veto = 0
        for ri, rot_i in enumerate(tables.rotamers[pi]):
            if (pi, ri) in plan.rotamer_vetoes:
                continue
            for rj, rot_j in enumerate(tables.rotamers[pj]):
                if (pj, rj) in plan.rotamer_vetoes:
                    continue
                by_pos = {pi: rot_i, pj: rot_j}
                try:
                    coords = [
                        _rotamer_coord(by_pos[x.position], x.ref, label)
                        if x.is_rotamer else x.fixed
                        for x in resolved
                    ]
                    value = measure(coords)
                except RestraintError:
                    # This combination cannot even be measured (a missing side
                    # chain atom): forbid it, same as a violation.
                    plan.pair_vetoes.add((pi, ri, pj, rj))
                    n_veto += 1
                    continue
                if not within(value):
                    plan.pair_vetoes.add((pi, ri, pj, rj))
                    n_veto += 1
        total = len(tables.rotamers[pi]) * len(tables.rotamers[pj])
        if n_veto >= total:
            raise RestraintError(
                f"restraint {label} forbids every combination of rotamers at {pi} and "
                f"{pj} ({n_veto} of {total}). Loosen it or widen those positions."
            )
        plan.summary.append(
            f"{label}: {r.kind.value} spans {pi} and {pj} -> {n_veto} of {total} "
            "pair(s) vetoed"
        )
        return

    plan.unsupported.append(
        f"{label}: its atoms depend on {len(positions)} design positions "
        f"({', '.join(positions)}), which cannot be expressed as a rotamer or "
        "pairwise veto. Restraints spanning three or more design positions are "
        "not supported; the restraint is NOT enforced."
    )


def _veto_single(
    plan: RestraintPlan,
    tables: EnergyTables,
    pos: str,
    label: str,
    lo: float,
    hi: float,
    measure_rot,
    predicate=None,
) -> int:
    """Veto every rotamer at ``pos`` whose measurement fails."""
    ok = predicate or (lambda v: lo - 1e-9 <= v <= hi + 1e-9)
    n = 0
    for idx, rot in enumerate(tables.rotamers[pos]):
        try:
            value = measure_rot(rot)
        except RestraintError:
            plan.rotamer_vetoes.add((pos, idx))   # cannot be measured => cannot comply
            n += 1
            continue
        if not ok(value):
            plan.rotamer_vetoes.add((pos, idx))
            n += 1
    return n


def _resolve(
    ref: AtomRef,
    label: str,
    tables: EnergyTables,
    fixed: dict[tuple, np.ndarray],
    design_positions: set[str],
) -> _Resolved:
    """Decide whether an atom moves with a rotamer or is fixed."""
    is_sidechain = ref.atom not in BACKBONE
    if ref.position in design_positions and is_sidechain:
        return _Resolved(ref=ref, position=ref.position)
    coord = fixed.get((ref.chain, ref.number, ref.insertion_code, ref.atom))
    if coord is None:
        near = sorted({
            k[3] for k in fixed
            if (k[0], k[1], k[2]) == (ref.chain, ref.number, ref.insertion_code)
        })
        if near:
            raise RestraintError(
                f"restraint {label}: residue {ref.position} has no atom {ref.atom!r}; "
                f"it has {', '.join(near)}"
            )
        raise RestraintError(
            f"restraint {label}: no residue {ref.position} in the supplied structures"
        )
    return _Resolved(ref=ref, fixed=coord)


def _rotamer_coord(rot: Rotamer, ref: AtomRef, label: str) -> np.ndarray:
    """Coordinate of ``ref.atom`` within this rotamer."""
    names = [n.strip().upper() for n in rot.atoms.names]
    try:
        i = names.index(ref.atom)
    except ValueError:
        raise RestraintError(
            f"restraint {label}: rotamer {rot.aa} at {rot.position} has no atom "
            f"{ref.atom!r} (it has {', '.join(names) or 'no side-chain atoms'})"
        ) from None
    return rot.atoms.coords[i]
