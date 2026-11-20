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
#: chain hangs off CA, which is before it. HXT/HO belong here too: they are
#: bonded to OXT, so leaving them out stretched the O-H bond from 0.954 to
#: 0.983 A every time psi moved on a C-terminal residue.
_PSI_MOVES = {"O", "OXT", "OT1", "OT2", "HXT"}


@dataclass
class Perturbation:
    """What was changed, so a move can be reported and reproduced."""

    chain: str
    residues: list[int] = field(default_factory=list)
    phi_deg: list[float] = field(default_factory=list)
    psi_deg: list[float] = field(default_factory=list)
    max_atom_shift: float = 0.0
    #: ``(residue number, insertion code)`` for each entry of ``residues``.
    #: ``residues`` alone cannot address a structure that uses insertion codes
    #: -- 52, 52A and 52B all read back as 52 -- and antibody numbering is
    #: exactly where they live, so the window check matches on these.
    keys: list[tuple[int, str]] = field(default_factory=list)
    #: The design position the region was built around, as ``(number, icode)``,
    #: or None when the caller asked for no anchor.
    anchor: "tuple[int, str] | None" = None
    #: True when ``anchor`` is inside ``keys``. False means the region was
    #: chosen without it -- the anchor was a proline, or it sits alone between
    #: chain breaks -- and the iteration therefore has an UNCOUPLED
    #: perturbation, which is the 12.4%-overlap defect §7h fixed. The caller
    #: must be able to tell those apart, and before this field it could not:
    #: the record looked identical either way.
    anchored: "bool | None" = None

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
    anchor: int | None = None,
    anchor_icode: str = "",
) -> list:
    """A run of consecutive residues to perturb.

    Proline is skipped by default: its phi is locked by the pyrrolidine ring, so
    perturbing it is geometrically meaningless. The 2006 paper says explicitly
    that the chosen segment is proline-free.

    ``anchor`` is a residue number the region must contain, and passing one is
    what makes the move IPRO's. The Suite paper's step 1 "begins by randomly
    selecting a DP", then draws the region size and the DP's place inside it;
    the released source does the same (a uniform design-position choice, then
    ``randint(0, 4)`` split into residues before and after). Without an anchor
    the region lands anywhere in the chain, which decouples the perturbation
    from the redesign: measured on the shipped 25-position OmpF spec, an
    unanchored region touches a design position in 12.4% of draws, where IPRO's
    probability is 1 by construction. The energy then moves in a region the
    MILP has no variables for, and the Metropolis test rules on that.
    """
    rng = rng or random.Random(0)
    residues = list(structure.residues(chain))
    if not residues:
        return []
    lo, hi = length_range
    want = rng.randint(max(1, lo), max(1, hi))

    if anchor is not None:
        run = _run_containing(residues, anchor, avoid_proline, anchor_icode)
        if run:
            key = (int(anchor), (anchor_icode or "").strip())
            at = next((i for i, r in enumerate(run) if residue_key(r) == key),
                      None)
            if at is None:
                at = next(i for i, r in enumerate(run)
                          if int(r.id[1]) == int(anchor))
            # Size and placement as the source draws them: the extra residues
            # are split before and after the design position.
            extra = min(want - 1, len(run) - 1)
            before = rng.randint(0, extra)
            after = extra - before
            start = at - before
            if start < 0:
                after += -start
                start = 0
            end = min(len(run), start + before + after + 1)
            start = max(0, end - (extra + 1))
            return run[start:end]
        # The anchor is a proline or sits alone between chain breaks; fall
        # through to an unanchored region rather than refusing to move, and say
        # so through the Perturbation record.

    # Consecutive in residue numbering, not merely adjacent in the list: a gap
    # in numbering is a chain break and rotating across it is not a local move.
    runs: list[list] = []
    current: list = []
    for res in residues:
        if avoid_proline and res.get_resname().strip() == "PRO":
            current = []
            continue
        if current and not _follows(current[-1], res):
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


def residue_key(res) -> tuple[int, str]:
    """``(number, insertion code)``, which is what actually names a residue.

    Keying by the bare number collapses 52, 52A and 52B onto one entry, and
    the last one read wins. In the perturbation path that silently dropped the
    design position from the region while the record still claimed it moved.
    """
    return int(res.id[1]), (res.id[2] or "").strip()


def _follows(a, b) -> bool:
    """Does ``b`` come straight after ``a`` in the chain's own numbering?

    Same number means an insertion code steps forward (52 -> 52A -> 52B);
    otherwise the number must step by exactly one. A larger gap is a chain
    break, and rotating across one is not a local move.
    """
    na, nb = int(a.id[1]), int(b.id[1])
    return nb == na or nb == na + 1


def _run_containing(residues: list, number: int, avoid_proline: bool,
                    icode: str = "") -> list:
    """The maximal numbering-contiguous, proline-free run holding the residue.

    Walks the ordered list outward from the anchor rather than indexing a
    ``{number: residue}`` map, so insertion codes neither collapse nor break
    contiguity.
    """
    icode = (icode or "").strip()
    want = (int(number), icode)
    idx = next((i for i, r in enumerate(residues)
                if residue_key(r) == want), None)
    if idx is None and not icode:
        # A caller that knows only the number still gets the first residue
        # carrying it, rather than nothing.
        idx = next((i for i, r in enumerate(residues)
                    if int(r.id[1]) == int(number)), None)
    if idx is None:
        return []

    def blocked(res) -> bool:
        return avoid_proline and res.get_resname().strip() == "PRO"

    if blocked(residues[idx]):
        return []
    lo = idx
    while lo > 0 and _follows(residues[lo - 1], residues[lo]) \
            and not blocked(residues[lo - 1]):
        lo -= 1
    hi = idx
    while hi + 1 < len(residues) and _follows(residues[hi], residues[hi + 1]) \
            and not blocked(residues[hi + 1]):
        hi += 1
    return residues[lo:hi + 1]


def perturb_backbone(
    structure,
    chain: str,
    *,
    sigma_deg: float = 1.5,
    cap_deg: float = 5.0,
    length_range: tuple[int, int] = (1, 5),
    rng: random.Random | None = None,
    segment: list | None = None,
    anchor: int | None = None,
    anchor_icode: str = "",
) -> Perturbation:
    """Perturb phi/psi of a short segment **in place**.

    Pass a :class:`Structure` copy if the original is still needed; this edits
    the coordinates it is given.
    """
    rng = rng or random.Random(0)
    seg = segment if segment is not None else choose_segment(
        structure, chain, length_range, rng, anchor=anchor,
        anchor_icode=anchor_icode,
    )
    anchor_key = (
        (int(anchor), (anchor_icode or "").strip()) if anchor is not None
        else None
    )
    if not seg:
        return Perturbation(chain=chain, anchor=anchor_key,
                            anchored=None if anchor is None else False)

    ordered = list(structure.residues(chain))
    index = {id(r): i for i, r in enumerate(ordered)}
    before = {
        id(a): np.array(a.coord, dtype=float)
        for r in ordered for a in r
    }

    keys = [residue_key(r) for r in seg]
    out = Perturbation(
        chain=chain,
        residues=[n for n, _ in keys],
        keys=keys,
        anchor=anchor_key,
        # An anchored region always contains its design position, so a False
        # here means the anchor was refused (proline, or alone between chain
        # breaks) and choose_segment fell through to an unanchored draw. The
        # comment in choose_segment promised to "say so through the
        # Perturbation record"; this is that.
        anchored=None if anchor_key is None else (anchor_key in keys),
    )

    for res in seg:
        i = index[id(res)]
        atoms = {a.get_name().strip().upper(): a for a in res}
        later = [a for r in ordered[i + 1:] for a in r]

        # phi needs the preceding residue's C and psi the next residue's N, so
        # at the chain's own termini one of them is undefined. Rotating about
        # an undefined torsion is not a conformational change at all: measured
        # on 2OMF chain A, a forced +5 deg "phi" at residue 1 moves atoms up to
        # 3.058 A while changing every internal interatomic distance by at most
        # 2.1e-14 A -- a rigid-body rotation of the whole molecule, which the
        # energy cannot see but the docking and the pore axis can. At interior
        # residue 151 the same perturbation changes internal distances by up to
        # 3.483 A, which is the intended move. The source skips both terminals.
        first_res = i == 0
        last_res = i == len(ordered) - 1

        # -- phi: axis N -> CA, moves the rest of this residue and all later ones
        if "N" in atoms and "CA" in atoms and not first_res:
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
        if "CA" in atoms and "C" in atoms and not last_res:
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
