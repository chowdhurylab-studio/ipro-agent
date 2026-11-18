"""Backbone perturbation: the move that makes this IPRO and not a packer.

The published step, from the Suite paper and confirmed against the released
source, is: pick a short run of consecutive residues and nudge their phi/psi by
offsets drawn from a Gaussian, rejection-truncated to a small cap. The source
uses ``random.gauss(0, 1.5)`` truncated to +/-5 degrees over 1-5 consecutive
residues; the 2006 paper reports sigma 1.65 degrees over the same window.

Why it matters: without it the search only ever repacks side chains on a fixed
backbone, which is a different (and much weaker) method. A 5-degree change in a
backbone dihedral moves atoms several Angstroms further down the chain, which is
enough to open or close a pocket that no rotamer choice could.

What this module does **not** do is the restrained minimization the published
protocol runs afterwards. That needs a force field with gradients; see
:mod:`ipro_agent.loop` for how the two available modes differ. A perturbation
followed by an optimal repack and a Metropolis test is a coherent backbone Monte
Carlo -- strained moves simply get rejected -- but it is not the published
protocol, and the loop says which one ran.

Geometry
--------
Rotating a backbone dihedral is a rigid rotation of everything downstream of its
axis:

* **phi(i)** is the dihedral ``C(i-1)-N(i)-CA(i)-C(i)``. Its axis is
  ``N(i) -> CA(i)``, so it moves every atom of residue ``i`` except ``N`` (and
  ``N``'s hydrogen), plus every atom of every later residue. The side chain of
  ``i`` moves with it.
* **psi(i)** is ``N(i)-CA(i)-C(i)-N(i+1)``. Its axis is ``CA(i) -> C(i)``, so it
  moves ``O(i)`` and every later residue. ``C(i)`` lies on the axis and the side
  chain hangs off ``CA``, so neither moves.

Only the design molecule's chain is touched; partners move as rigid bodies in
:mod:`ipro_agent.docking` instead.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: Atoms of a residue that stay put when phi rotates (they are on the axis or
#: before it).
_PHI_FIXED = {"N", "H", "H1", "H2", "H3", "HN"}
#: Atoms of a residue that move when psi rotates. C is on the axis; the side
#: chain hangs off CA, which is before it.
_PSI_MOVES = {"O", "OXT", "OT1", "OT2"}


@dataclass
class Perturbation:
    """What was changed, so a move can be reported and reproduced."""

    chain: str
    residues: list[int] = field(default_factory=list)
    phi_deg: list[float] = field(default_factory=list)
    psi_deg: list[float] = field(default_factory=list)
    max_atom_shift: float = 0.0

    def describe(self) -> str:
        if not self.residues:
            return "no perturbation applied"
        span = (
            f"{self.chain}{self.residues[0]}"
            if len(self.residues) == 1
            else f"{self.chain}{self.residues[0]}-{self.residues[-1]}"
        )
        return (
            f"{span}: phi {', '.join(f'{x:+.2f}' for x in self.phi_deg)} deg, "
            f"psi {', '.join(f'{x:+.2f}' for x in self.psi_deg)} deg; "
            f"largest atom shift {self.max_atom_shift:.2f} A"
        )


def truncated_gauss(sigma: float, cap: float, rng: random.Random) -> float:
    """A Gaussian offset, rejection-truncated to ``+/-cap``.

    Rejection, not clipping: clipping would pile probability mass up on the two
    bounds, which is a different distribution from the one the paper specifies.
    """
    if sigma <= 0:
        return 0.0
    for _ in range(100):
        x = rng.gauss(0.0, sigma)
        if abs(x) <= cap:
            return x
    return math.copysign(cap, rng.gauss(0.0, 1.0))


def _rotate_about(coords: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate points about the line through ``origin`` along ``axis``."""
    n = np.linalg.norm(axis)
    if n < 1e-12 or abs(angle_deg) < 1e-12:
        return coords
    k = axis / n
    a = math.radians(angle_deg)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    R = np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * (K @ K)
    return (coords - origin) @ R.T + origin


def choose_segment(
    structure,
    chain: str,
    length_range: tuple[int, int] = (1, 5),
    rng: random.Random | None = None,
    avoid_proline: bool = True,
) -> list:
    """A run of consecutive residues to perturb.

    Proline is skipped by default: its phi is locked by the pyrrolidine ring, so
    perturbing it is geometrically meaningless. The 2006 paper says explicitly
    that the chosen segment is proline-free.
    """
    rng = rng or random.Random(0)
    residues = list(structure.residues(chain))
    if not residues:
        return []
    lo, hi = length_range
    want = rng.randint(max(1, lo), max(1, hi))

    # Consecutive in residue numbering, not merely adjacent in the list: a gap
    # in numbering is a chain break and rotating across it is not a local move.
    runs: list[list] = []
    current: list = []
    for res in residues:
        num = int(res.id[1])
        if avoid_proline and res.get_resname().strip() == "PRO":
            current = []
            continue
        if current and num != int(current[-1].id[1]) + 1:
            current = []
        current.append(res)
        runs.append(list(current[-want:]))
    runs = [r for r in runs if len(r) == want]
    if not runs:
        # Fall back to the longest available run rather than refusing to move.
        singles = [
            [r] for r in residues
            if not (avoid_proline and r.get_resname().strip() == "PRO")
        ]
        return rng.choice(singles) if singles else []
    return rng.choice(runs)


def perturb_backbone(
    structure,
    chain: str,
    *,
    sigma_deg: float = 1.5,
    cap_deg: float = 5.0,
    length_range: tuple[int, int] = (1, 5),
    rng: random.Random | None = None,
    segment: list | None = None,
) -> Perturbation:
    """Perturb phi/psi of a short segment **in place**.

    Pass a :class:`Structure` copy if the original is still needed; this edits
    the coordinates it is given.
    """
    rng = rng or random.Random(0)
    seg = segment if segment is not None else choose_segment(
        structure, chain, length_range, rng
    )
    if not seg:
        return Perturbation(chain=chain)

    ordered = list(structure.residues(chain))
    index = {id(r): i for i, r in enumerate(ordered)}
    before = {
        id(a): np.array(a.coord, dtype=float)
        for r in ordered for a in r
    }

    out = Perturbation(chain=chain, residues=[int(r.id[1]) for r in seg])

    for res in seg:
        i = index[id(res)]
        atoms = {a.get_name().strip().upper(): a for a in res}
        later = [a for r in ordered[i + 1:] for a in r]

        # -- phi: axis N -> CA, moves the rest of this residue and all later ones
        if "N" in atoms and "CA" in atoms:
            angle = truncated_gauss(sigma_deg, cap_deg, rng)
            out.phi_deg.append(angle)
            origin = np.array(atoms["N"].coord, dtype=float)
            axis = np.array(atoms["CA"].coord, dtype=float) - origin
            movers = [
                a for a in res if a.get_name().strip().upper() not in _PHI_FIXED
            ] + later
            _apply(movers, origin, axis, angle)
        else:
            out.phi_deg.append(0.0)

        # -- psi: axis CA -> C, moves O and all later residues
        if "CA" in atoms and "C" in atoms:
            angle = truncated_gauss(sigma_deg, cap_deg, rng)
            out.psi_deg.append(angle)
            origin = np.array(atoms["CA"].coord, dtype=float)
            axis = np.array(atoms["C"].coord, dtype=float) - origin
            movers = [
                a for a in res if a.get_name().strip().upper() in _PSI_MOVES
            ] + later
            _apply(movers, origin, axis, angle)
        else:
            out.psi_deg.append(0.0)

    shift = 0.0
    for r in ordered:
        for a in r:
            old = before.get(id(a))
            if old is not None:
                shift = max(shift, float(np.linalg.norm(np.array(a.coord) - old)))
    out.max_atom_shift = shift
    return out


def _apply(atoms, origin: np.ndarray, axis: np.ndarray, angle_deg: float) -> None:
    if not atoms or abs(angle_deg) < 1e-12:
        return
    coords = np.array([a.coord for a in atoms], dtype=float)
    moved = _rotate_about(coords, origin, axis, angle_deg)
    for a, c in zip(atoms, moved):
        a.set_coord(c)


def write_perturbed(structure, path: str | Path) -> Path:
    """Save a perturbed structure so PyMOL can re-harvest rotamers from it."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return structure.save(out)


def backbone_bond_lengths(structure, chain: str) -> dict[str, float]:
    """Every backbone bond length, for checking a perturbation left them alone.

    The invariant worth testing is **invariance**, not ideality: a rigid
    rotation about a bond axis cannot change any bond length, whatever the input
    geometry was. Comparing against ideal values instead would flag the input
    structure's own deviations -- 1CRN, a 1.5 A crystal structure, has an N-CA
    bond of 1.518 A at residue 37 before anything is touched.

    This is the one cheap check that catches a wrong rotation axis or a wrong
    set of moving atoms, which otherwise produce a plausible-looking structure
    with silently broken geometry.
    """
    out: dict[str, float] = {}
    residues = list(structure.residues(chain))
    for res in residues:
        atoms = {a.get_name().strip().upper(): np.array(a.coord, float) for a in res}
        num = int(res.id[1])
        for a, b in (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB")):
            if a in atoms and b in atoms:
                out[f"{chain}{num}:{a}-{b}"] = float(
                    np.linalg.norm(atoms[a] - atoms[b])
                )
    for x, y in zip(residues, residues[1:]):
        ax = {a.get_name().strip().upper(): np.array(a.coord, float) for a in x}
        ay = {a.get_name().strip().upper(): np.array(a.coord, float) for a in y}
        if "C" in ax and "N" in ay and int(y.id[1]) == int(x.id[1]) + 1:
            out[f"{chain}{int(x.id[1])}-{int(y.id[1])}:peptide"] = float(
                np.linalg.norm(ax["C"] - ay["N"])
            )
    return out


def bonds_unchanged(
    before: dict[str, float], after: dict[str, float], tolerance: float = 1e-6
) -> list[str]:
    """Bonds whose length changed; a correct perturbation changes none."""
    bad = []
    for key, was in before.items():
        now = after.get(key)
        if now is None:
            bad.append(f"{key} disappeared")
        elif abs(now - was) > tolerance:
            bad.append(f"{key}: {was:.4f} -> {now:.4f} A")
    return bad
