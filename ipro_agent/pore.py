"""Measuring a channel: how wide is the narrowest point?

PoreDesigner's acceptance filter is geometric, not energetic: a redesign is kept
only if the pore's bottleneck falls inside a requested window. Its SI describes
cutting the constriction into slices every 0.5 A and fitting, in each, the
largest ellipse that touches the van der Waals surfaces without enclosing an
atom; the bottleneck is the smallest major axis over those slices.

This module computes the same quantity with a circle rather than an ellipse. The
difference is worth stating: an inscribed **circle** is bounded by the narrowest
direction, so for an elliptical lumen it reports something close to the *minor*
axis, while the paper reports the *major* axis of a fitted ellipse. For a
near-circular pore the two agree; for a slit-like one the circle is smaller.
Numbers from here are therefore comparable across runs of this tool but are not
the paper's number, and `docs/DESIGN.md` says so.

What is measured, per slice:

    radius(slice) = max over candidate centres c of
                        min over atoms a of ( |c - centre(a)| - r_vdw(a) )

that is, the largest circle that fits without overlapping any atom's van der
Waals sphere. The bottleneck is the minimum of that over all slices.

The axis has to come from somewhere. A barrel's axis is not reliably its
largest-variance direction (a squat barrel's is the smallest), so guessing is
avoided: give two atoms that span the channel, and the axis is the line between
them. Failing that the principal axis of the CA atoms is used and the run says
it guessed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

#: Van der Waals radii in Angstroms, Bondi (1964), by element. Used only for
#: the geometric "does an atom block this circle" test, so element-level is
#: appropriate here in a way it is not for an energy.
VDW = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20,
    "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "SE": 1.90,
}
_VDW_DEFAULT = 1.70


@dataclass
class PoreProfile:
    """The channel's radius along its axis."""

    #: Distance along the axis, from the first slice.
    positions: list[float] = field(default_factory=list)
    #: Largest inscribed circle radius at each slice.
    radii: list[float] = field(default_factory=list)
    #: Centre of the widest circle at each slice, in Cartesian space.
    centres: list[np.ndarray] = field(default_factory=list)
    axis_origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axis_direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    #: True when the axis was guessed rather than given.
    axis_guessed: bool = False

    @property
    def min_radius(self) -> float:
        return float(min(self.radii)) if self.radii else float("nan")

    @property
    def min_diameter(self) -> float:
        return 2.0 * self.min_radius

    @property
    def bottleneck_position(self) -> float:
        if not self.radii:
            return float("nan")
        return self.positions[int(np.argmin(self.radii))]

    def describe(self) -> str:
        if not self.radii:
            return "no slices measured"
        guess = " (axis guessed)" if self.axis_guessed else ""
        return (
            f"bottleneck {self.min_diameter:.2f} A diameter at {self.bottleneck_position:.1f} A "
            f"along the axis; widest {2 * max(self.radii):.2f} A; "
            f"{len(self.radii)} slices{guess}"
        )


def vdw_radii(names: list[str], elements: list[str] | None = None) -> np.ndarray:
    """Per-atom vdW radius. Pass ``elements`` (the file's element column) when
    the caller has it: deriving the element from the atom name alone cannot
    reach a two-letter symbol, so an FE or CL lining the lumen was measured
    with fluorine's or carbon's radius. ``VDW``'s own CL/BR/SE entries were
    unreachable for the same reason -- the identical defect that
    ``energy.base.guess_element`` documents."""
    from .energy.base import guess_element

    out = np.empty(len(names))
    for i, nm in enumerate(names):
        el = (elements[i] if elements and i < len(elements) and elements[i]
              else guess_element(nm))
        out[i] = VDW.get(str(el).strip().upper(), _VDW_DEFAULT)
    return out


def principal_axis(coords: np.ndarray) -> np.ndarray:
    """Largest-variance direction of a point cloud."""
    c = np.asarray(coords, dtype=float)
    centred = c - c.mean(axis=0)
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    return vt[0] / np.linalg.norm(vt[0])


def _largest_inscribed_circle(
    points2d: np.ndarray,
    radii: np.ndarray,
    centre0: np.ndarray,
    max_centre_offset: float,
    coarse: int = 17,
    refine_rounds: int = 4,
) -> tuple[float, np.ndarray]:
    """Largest circle that fits, **centred near the axis**.

    The centre is constrained to within ``max_centre_offset`` of ``centre0``,
    and that constraint is the whole measurement. Without it the answer is
    meaningless: in unbounded space the largest circle that overlaps no atom is
    out in the solvent, so an unconstrained search walks off the protein and
    reports the distance to the bulk rather than the width of the lumen. (This
    is not hypothetical -- it is what the first version of this function did,
    reporting a constriction one slice away from the real one because a grid
    corner outside the barrel scored higher than the true bottleneck.)

    Coarse grid then successive refinement: the clearance function
    ``min_a(|c - p_a| - r_a)`` is piecewise smooth with one broad maximum inside
    a lumen, so a few rounds land on it without a fine grid over the whole disc.
    """
    limit = float(max_centre_offset)
    best_r, best_c = -np.inf, centre0.astype(float).copy()
    span = limit
    centre = centre0.astype(float).copy()
    for _ in range(max(1, refine_rounds)):
        xs = np.linspace(centre[0] - span, centre[0] + span, coarse)
        ys = np.linspace(centre[1] - span, centre[1] + span, coarse)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
        # Hard constraint: the centre stays within `limit` of the axis.
        keep = ((grid - centre0) ** 2).sum(axis=1) <= limit ** 2 + 1e-9
        grid = grid[keep]
        if len(grid) == 0:
            break
        if len(points2d) == 0:
            return limit, centre0.astype(float)
        d = np.sqrt(
            ((grid[:, None, :] - points2d[None, :, :]) ** 2).sum(-1)
        ) - radii[None, :]
        clearance = d.min(axis=1)
        k = int(np.argmax(clearance))
        if clearance[k] > best_r:
            best_r, best_c = float(clearance[k]), grid[k].copy()
        centre = grid[k]
        span /= (coarse - 1) / 2.0
    return max(best_r, 0.0), best_c


def pore_profile(
    coords: np.ndarray,
    names: list[str],
    axis_origin: np.ndarray | None = None,
    axis_direction: np.ndarray | None = None,
    *,
    spacing: float = 0.5,
    window: tuple[float, float] | None = None,
    max_centre_offset: float = 6.0,
    elements: list[str] | None = None,
) -> PoreProfile:
    """Radius of the largest inscribed circle, slice by slice along the axis.

    ``window`` restricts the measurement to a span along the axis, measured from
    the first slice; PoreDesigner measured only the ~8 A constriction region
    rather than the whole barrel, and doing the same avoids reporting the open
    mouth of a funnel as its bottleneck.

    ``max_centre_offset`` is how far the measured circle's centre may sit from
    the axis. It is a real parameter, not a tolerance: it is what makes the
    question "how wide is the lumen" rather than "how far is the nearest atom",
    and a value much larger than the pore lets the search escape into solvent.

    The default is 6.0 and used to be 4.0 here while ``loop._measure_geometry``
    passed 6.0, so the same structure measured differently depending on which
    one you called, and no spec in the repo sets the key. 6.0 is the one to
    keep, which is not obvious: on OmpF (2OMF chain A, the 25 published
    positions) the cap is still *binding* at 4.0 -- the winning circle's centre
    sits at exactly 4.000 A, i.e. clamped -- and the reading is 7.68 A. It
    stops binding between 5.0 and 6.0, where the centre settles at 3.14 A of
    its own accord and the reading is 8.07 A, which is the figure §4 records
    against the published 11 x 7 A eyelet. A cap low enough to bind is
    measuring the parameter rather than the pore, so the geometry record now
    reports the winning centre's offset alongside the radius: when that number
    equals the cap, the reading is an artefact of this setting.

    Each slice is a true cross-section: an atom obstructs it only if its vdW
    sphere actually reaches that plane, and then by the radius of the circle
    where sphere and plane intersect. There is deliberately no slab thickness.
    An earlier version admitted every atom within a slab and gave the ones
    outside their own radius an in-plane radius of zero, which left them in
    the slice as *point* obstacles -- and a point still blocks an inscribed
    circle, so a carbon 2.5 A off-plane reported a 4.3 A lumen as 2.2 A.
    """
    c = np.asarray(coords, dtype=float)
    if c.size == 0:
        return PoreProfile()
    r = vdw_radii(names, elements)

    guessed = axis_direction is None
    direction = (
        principal_axis(c) if axis_direction is None
        else np.asarray(axis_direction, dtype=float)
    )
    n = np.linalg.norm(direction)
    if n < 1e-12:
        raise ValueError("the pore axis direction is degenerate")
    direction = direction / n
    origin = (
        c.mean(axis=0) if axis_origin is None
        else np.asarray(axis_origin, dtype=float)
    )

    # An orthonormal frame with `direction` as z.
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(helper @ direction)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    ex = helper - (helper @ direction) * direction
    ex /= np.linalg.norm(ex)
    ey = np.cross(direction, ex)

    rel = c - origin
    z = rel @ direction
    x = rel @ ex
    y = rel @ ey

    lo, hi = float(z.min()), float(z.max())
    if window is not None:
        w0, w1 = window
        lo, hi = lo + float(w0), lo + float(w1)
    if hi <= lo:
        return PoreProfile(axis_origin=origin, axis_direction=direction, axis_guessed=guessed)

    prof = PoreProfile(axis_origin=origin, axis_direction=direction, axis_guessed=guessed)
    steps = max(1, int(np.floor((hi - lo) / max(spacing, 1e-6))) + 1)
    for k in range(steps):
        zc = lo + k * spacing
        # Atoms whose vdW sphere reaches this plane; their in-plane radius is
        # the circle where the sphere cuts the plane.
        dz = np.abs(z - zc)
        near = dz < r
        if not near.any():
            continue
        eff = np.sqrt(np.maximum(r[near] ** 2 - dz[near] ** 2, 0.0))
        pts = np.stack([x[near], y[near]], axis=1)
        rad, centre2d = _largest_inscribed_circle(
            pts, eff, np.zeros(2), max_centre_offset
        )
        prof.positions.append(zc - lo)
        prof.radii.append(float(rad))
        prof.centres.append(origin + zc * direction + centre2d[0] * ex + centre2d[1] * ey)
    return prof


def axis_from_lining(ca_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Channel axis from the residues that line it: the normal to their plane.

    Residues lining a constriction sit in a ring, so their CA atoms are roughly
    coplanar and the normal to that plane is the channel axis. This uses exactly
    the information the user already supplies -- the design positions -- and is
    far better than guessing from the whole protein: on OmpF (2OMF chain A) the
    25 published constriction positions give a plane whose singular values are
    (37.9, 31.6, 16.8), and taking the third as the axis measures the eyelet at
    **8.1 A** against the published 11 x 7 A. The protein's own largest-variance
    direction gives 1.4 A, i.e. it is not the barrel axis at all.

    Returns ``(origin, direction)``: the ring's centroid and its plane normal.
    """
    c = np.asarray(ca_coords, dtype=float)
    if len(c) < 3:
        raise ValueError(
            "at least 3 lining residues are needed to define a channel axis; "
            "give goal_options.pore_axis explicitly instead"
        )
    centre = c.mean(axis=0)
    _u, _s, vt = np.linalg.svd(c - centre, full_matrices=False)
    normal = vt[-1]
    return centre, normal / np.linalg.norm(normal)


def ring_flatness(ca_coords: np.ndarray) -> float:
    """How plane-like the lining residues are: third singular value over second.

    Near 0 means a flat ring, so the plane normal is a trustworthy axis. Near 1
    means the residues are not coplanar and the axis from them is meaningless --
    worth saying out loud rather than reporting a number from it.
    """
    c = np.asarray(ca_coords, dtype=float)
    if len(c) < 3:
        return 1.0
    _u, s, _vt = np.linalg.svd(c - c.mean(axis=0), full_matrices=False)
    return float(s[-1] / s[-2]) if s[-2] > 1e-9 else 1.0


def profile_from_atomset(atoms, **kwargs) -> PoreProfile:
    """Convenience wrapper for an :class:`ipro_agent.energy.base.AtomSet`."""
    return pore_profile(atoms.coords, list(atoms.names), **kwargs)
