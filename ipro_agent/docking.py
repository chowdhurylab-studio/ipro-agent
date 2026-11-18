"""IPRO's redocking step: local rigid-body Monte Carlo with annealing.

Worth being precise about what this is, because it is easy to assume otherwise.
Step three of an IPRO iteration, as the IPRO Suite paper specifies it
(Pantazes et al., *J Comput Chem* **36**:251-263, 2015, which the paper calls a
"local, rigid-body" redocking), is summarized here rather than quoted, since
that paper is paywalled and this file is MIT licensed:

* each target molecule is perturbed along and about X, Y and Z, by Gaussian
  offsets centred at zero with sigma **0.2 A** and **2.0 degrees**;
* a move is kept if the **net interaction energy** improves;
* **500** such moves are performed per molecule;
* scoring uses **the same energy function as rotamer selection**, not a
  docking-specific one;
* simulated annealing runs on a constant cooling schedule from **3640 K** to
  **2190 K**, the two temperatures that retain 25% and 10% of positions within
  10 kcal/mol of the best solution so far.

Those are facts about a protocol, not the paper's expression of them. See
docs/DESIGN.md section 1e for the same numbers with their provenance.

So it is **not** a docking program. It is a short local search in the six
rigid-body degrees of freedom, scored with the same energy the MILP uses, which
means it needs no external dependency and is reproducible from a seed. That is
what this module implements.

AutoDock Vina solves a different problem -- a *global* search with its own
scoring function -- and is the right tool when there is no starting pose at all.
It stays available as the explicit ``ipro-agent dock`` command, and can be opted
into for this step (``backends.dock: vina``), with the caveat recorded in
:func:`redock` that doing so replaces local refinement with global re-placement.

The objective, following the paper, is the **net interaction energy of all the
partners with each other and with the design molecule** -- not each partner
scored in isolation, since partners can touch.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field

import numpy as np

from .energy.base import AtomSet, EnergyBackend

log = logging.getLogger(__name__)

#: Gas constant in kcal/(mol K), as elsewhere in this package.
R_KCAL = 1.987e-3


@dataclass
class Pose:
    """A rigid-body placement: rotate about the body's own centroid, then shift.

    Stored as an accumulated rotation and translation rather than as coordinates
    so it can be applied to a full-atom structure later (including hydrogens and
    any atoms the scoring set left out).
    """

    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    #: Centroid the rotation is taken about, in the input frame.
    centre: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def apply(self, coords: np.ndarray) -> np.ndarray:
        c = np.asarray(coords, dtype=float)
        return (c - self.centre) @ self.rotation.T + self.centre + self.translation

    @property
    def affine(self) -> tuple[np.ndarray, np.ndarray]:
        """This pose as the plain affine map ``p -> M p + v``."""
        r = np.asarray(self.rotation, dtype=float)
        c = np.asarray(self.centre, dtype=float)
        t = np.asarray(self.translation, dtype=float)
        return r, c - r @ c + t

    @classmethod
    def from_affine(cls, m: np.ndarray, v: np.ndarray) -> "Pose":
        return cls(
            rotation=np.asarray(m, dtype=float),
            translation=np.asarray(v, dtype=float),
            centre=np.zeros(3),
        )

    def then(self, other: "Pose") -> "Pose":
        """The single pose equivalent to applying ``self`` and then ``other``.

        Exact. Composing beats re-fitting two coordinate sets, which cannot
        recover a rotation about the axis of a collinear body at all.
        """
        m1, v1 = self.affine
        m2, v2 = other.affine
        return Pose.from_affine(m2 @ m1, m2 @ v1 + v2)

    @property
    def is_identity(self) -> bool:
        return (
            np.allclose(self.rotation, np.eye(3), atol=1e-12)
            and np.allclose(self.translation, 0.0, atol=1e-12)
        )

    def rmsd_of(self, coords: np.ndarray) -> float:
        """How far this pose moves the given atoms, in Angstroms."""
        c = np.asarray(coords, dtype=float)
        if c.size == 0:
            return 0.0
        return float(np.sqrt(((self.apply(c) - c) ** 2).sum(axis=1).mean()))

    def describe(self) -> str:
        ang = math.degrees(
            math.acos(max(-1.0, min(1.0, (np.trace(self.rotation) - 1.0) / 2.0)))
        )
        return (
            f"translation {np.linalg.norm(self.translation):.3f} A, "
            f"rotation {ang:.2f} deg"
        )


def _random_rotation(sigma_deg: float, rng: random.Random) -> np.ndarray:
    """Small rotation: a Gaussian angle about a uniformly random axis.

    The paper perturbs "around the X, Y and Z axes" with a Gaussian of 2.0
    degrees. Composing three such rotations is equivalent, to first order, to
    one Gaussian rotation about a random axis with the same per-axis spread, and
    the single-axis form cannot accumulate the ordering bias that applying three
    in sequence does.
    """
    axis = np.array([rng.gauss(0.0, 1.0) for _ in range(3)])
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    axis = axis / n
    angle = math.radians(rng.gauss(0.0, sigma_deg))
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    # Rodrigues' rotation formula.
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


@dataclass
class RedockResult:
    """Outcome of one partner's local search."""

    pose: Pose
    start_energy: float
    best_energy: float
    moves: int
    accepted: int
    #: True when the search left the partner where it started.
    unchanged: bool = False

    @property
    def improvement(self) -> float:
        return self.best_energy - self.start_energy

    def describe(self) -> str:
        return (
            f"{self.start_energy:+.3f} -> {self.best_energy:+.3f} "
            f"({self.improvement:+.3f}) in {self.moves} moves, "
            f"{self.accepted} accepted; {self.pose.describe()}"
        )


def net_interaction(
    design: AtomSet,
    partners: dict[str, AtomSet],
    backend: EnergyBackend,
) -> float:
    """Net interaction energy of the partners with the design and each other.

    The paper's objective for this step. Partner-partner terms matter whenever
    more than one partner is present, which is exactly when a redock can push
    two of them into each other.
    """
    total = 0.0
    names = sorted(partners)
    for n in names:
        total += backend.interaction(design, partners[n])
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            total += backend.interaction(partners[a], partners[b])
    return total


def redock(
    design: AtomSet,
    partners: dict[str, AtomSet],
    backend: EnergyBackend,
    *,
    moves: int = 500,
    sigma_angstrom: float = 0.2,
    sigma_deg: float = 2.0,
    temperature_start_k: float = 3640.0,
    temperature_end_k: float = 2190.0,
    rng: random.Random | None = None,
    mobile: set[str] | None = None,
) -> dict[str, RedockResult]:
    """Locally refine each partner's rigid-body placement.

    One Monte Carlo chain over all mobile partners jointly: each move perturbs
    one randomly chosen partner, the *net* interaction energy is re-evaluated,
    and the move is kept if it improved or by the Metropolis test at the current
    temperature. Temperature falls linearly from ``temperature_start_k`` to
    ``temperature_end_k`` across the run ("a constant cooling schedule").

    Returns the best pose found for each mobile partner. Poses are relative to
    the coordinates passed in, so applying one twice would move a partner twice.

    ``rng`` makes the search reproducible, which the published protocol is not:
    IPRO seeds nothing.
    """
    rng = rng or random.Random(0)
    movable = sorted(mobile if mobile is not None else partners)
    movable = [m for m in movable if m in partners and len(partners[m])]
    if not movable:
        return {}

    current = {n: np.array(a.coords, dtype=float) for n, a in partners.items()}
    # The transform is accumulated as ``p -> M p + v`` alongside the
    # coordinates. It used to be recovered at the end by fitting start to
    # finish, which cannot see a rotation about the axis of a one- or
    # two-atom or collinear body: those atoms do not move under it. The
    # atoms left out of the scoring set (hydrogens) were then placed by
    # whatever spin the fit happened to return.
    xform = {n: (np.eye(3), np.zeros(3)) for n in movable}

    def sets_from(coords: dict[str, np.ndarray]) -> dict[str, AtomSet]:
        return {
            n: AtomSet(
                names=list(partners[n].names), coords=coords[n],
                resnames=list(partners[n].resnames), reskeys=list(partners[n].reskeys),
            )
            for n in partners
        }

    start_energy = net_interaction(design, sets_from(current), backend)
    best_energy = cur_energy = start_energy
    best = {n: current[n].copy() for n in movable}
    best_xform = dict(xform)
    accepted = 0

    for step in range(max(0, int(moves))):
        frac = (step / (moves - 1)) if moves > 1 else 1.0
        temp = temperature_start_k + frac * (temperature_end_k - temperature_start_k)
        kt = max(R_KCAL * temp, 1e-12)

        name = movable[rng.randrange(len(movable))]
        rot = _random_rotation(sigma_deg, rng)
        shift = np.array([rng.gauss(0.0, sigma_angstrom) for _ in range(3)])
        centre = current[name].mean(axis=0)
        trial = (current[name] - centre) @ rot.T + centre + shift

        keep = current[name]
        current[name] = trial
        trial_energy = net_interaction(design, sets_from(current), backend)
        delta = trial_energy - cur_energy
        if delta <= 0 or rng.random() < math.exp(-min(delta / kt, 700.0)):
            cur_energy = trial_energy
            accepted += 1
            # This move is p -> rot (p - centre) + centre + shift; composed
            # with the accumulated p -> M p + v that is M' = rot M,
            # v' = rot (v - centre) + centre + shift.
            m, v = xform[name]
            xform[name] = (rot @ m, rot @ (v - centre) + centre + shift)
            if trial_energy < best_energy:
                best_energy = trial_energy
                best = {n: current[n].copy() for n in movable}
                best_xform = dict(xform)
        else:
            current[name] = keep

    out: dict[str, RedockResult] = {}
    for n in movable:
        m, v = best_xform[n]
        pose = Pose.from_affine(m, v)
        out[n] = RedockResult(
            pose=pose,
            start_energy=start_energy,
            best_energy=best_energy,
            moves=int(moves),
            accepted=accepted,
            unchanged=pose.rmsd_of(partners[n].coords) < 1e-9,
        )
    return out


def _pose_between(
    start: np.ndarray,
    end: np.ndarray,
    centre: np.ndarray,
    notes: list[str] | None = None,
) -> Pose:
    """Recover the rigid transform taking ``start`` to ``end`` (Kabsch).

    Only for the case where coordinates are all there is -- a docking program
    that returns a repositioned ligand, say. Prefer :meth:`Pose.then` whenever
    the transforms themselves are in hand, because this fit has a blind spot:

    If the body spans fewer than two dimensions (one atom, two atoms, or any
    collinear set) a rotation *about the body's own axis* moves none of its
    atoms, so no fit can see it. The fitted atoms come back exact either way
    while anything off that axis -- a hydrogen, a branch the scoring set
    dropped -- is placed by whichever spin the SVD happened to produce; up to
    5 A in practice. Rather than pick a spin at random, this returns the
    zero-twist rotation: the smallest rotation that carries the body's axis
    onto its new one, and nothing more. ``notes`` records when that happened,
    because the result is then a choice and not a measurement.
    """
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    if a.shape != b.shape or len(a) == 0:
        return Pose(centre=np.asarray(centre, dtype=float))
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    ac, bc = a - ca, b - cb

    sv = np.linalg.svd(ac, compute_uv=False)
    span = int((sv > max(sv[0] * 1e-8, 1e-9)).sum()) if sv.size else 0
    if span == 0:
        return Pose(translation=cb - ca, centre=ca)
    if span == 1:
        u = np.linalg.svd(ac)[2][0]
        v = np.linalg.svd(bc)[2][0]
        if float((ac @ u) @ (bc @ v)) < 0:
            v = -v
        if notes is not None:
            notes.append(
                "the moved body is collinear, so a rotation about its own axis "
                "cannot be recovered from coordinates; the zero-twist rotation "
                "was used and any off-axis atoms (e.g. hydrogens) may be "
                "rotated about that axis relative to the docking program's output"
            )
        return Pose(rotation=_min_rotation(u, v), translation=cb - ca, centre=ca)

    H = ac.T @ bc
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    # start -> end is: rotate about ca, then shift by (cb - ca)
    return Pose(rotation=R, translation=cb - ca, centre=ca)


def _min_rotation(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Smallest rotation carrying unit vector ``u`` onto unit vector ``v``."""
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    c = float(u @ v)
    if c > 1.0 - 1e-12:
        return np.eye(3)
    axis = np.cross(u, v)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:  # antiparallel: any perpendicular axis, turned half a turn
        helper = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(u, helper)
        axis /= np.linalg.norm(axis)
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * (k @ k)
    axis /= n
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + n * k + (1.0 - c) * (k @ k)


def moved_atomset(original: AtomSet, pose: Pose) -> AtomSet:
    """``original`` with ``pose`` applied."""
    return AtomSet(
        names=list(original.names),
        coords=pose.apply(original.coords),
        resnames=list(original.resnames),
        reskeys=list(original.reskeys),
    )
