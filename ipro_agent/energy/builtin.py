"""A built-in screening energy, used when PyRosetta is not installed.

**Read this before trusting a number from it.** This is a *screening* function,
not a publication-grade force field. Its purpose is to make the IPRO loop and
the example case runnable and testable on a machine with no licensed software,
and to let the MILP be exercised and verified. For energies you intend to
report, use the PyRosetta backend (``backends.energy: pyrosetta``), which is
the force field this kind of redesign work actually uses.

What it computes, per pair of heavy atoms from different residues:

1. **Lennard-Jones 12-6**, Amber functional form

       E_LJ = eps_ij * [ (Rmin_ij / r)^12  -  2 * (Rmin_ij / r)^6 ]

   with ``Rmin_ij = Rmin_i + Rmin_j`` and ``eps_ij = sqrt(eps_i * eps_j)``.
   Parameters are the Amber parm94/99 values (Cornell et al., J. Am. Chem. Soc.
   1995, 117, 5179) taken at the **element** level rather than per atom type --
   that is the main approximation here.

2. **Capped repulsion.** The repulsive branch is clamped at ``max_repulsion``
   per pair. Rotamer libraries routinely produce hard clashes whose raw 1/r^12
   values are astronomically large; left uncapped they dominate the MILP
   objective and make the integer program numerically ill-conditioned. Softened
   repulsion is standard practice in side-chain packing.

3. **Charged-pair Coulomb** with a distance-dependent dielectric
   ``eps(r) = 4r``, applied only to formally charged side-chain atoms. A full
   partial-charge set is deliberately *not* used: an incomplete one would be
   worse than an explicit, documented restriction to the groups that dominate
   electrostatics in design scoring.

4. **A geometric hydrogen-bond well** between N/O donor-acceptor heavy atoms,
   as a distance-only term (no angular dependence).

All terms are deterministic and pairwise, which is what the MILP requires.
"""

from __future__ import annotations

import numpy as np

from .base import AtomSet, EnergyBackend

# --- Lennard-Jones parameters, Amber parm94/99, element level ---------------
# Rmin (A) is the distance of minimum energy for a like-pair halved, i.e. the
# per-atom radius; eps (kcal/mol) is the well depth.
LJ: dict[str, tuple[float, float]] = {
    "C": (1.9080, 0.1094),   # parm99 CT (sp3 carbon)
    "N": (1.8240, 0.1700),   # parm99 N (amide nitrogen)
    "O": (1.6612, 0.2100),   # parm99 O (carbonyl oxygen)
    "S": (2.0000, 0.2500),   # parm99 S
    "H": (1.4870, 0.0157),   # parm99 HC
    "P": (2.1000, 0.2000),   # parm99 P
    "F": (1.7500, 0.0610),
    "CL": (1.9480, 0.2650),
    "BR": (2.2200, 0.3200),
    "I": (2.3500, 0.4000),
    "SE": (2.1030, 0.2910),
}
_LJ_DEFAULT = (1.9080, 0.1094)

# --- Formal charges on ionizable side-chain atoms ---------------------------
# Keyed (resname, atom name). Carboxylate charge is split over both oxygens;
# the guanidinium charge over its three nitrogens. HIS is left neutral, the
# usual default at pH 7 for design.
FORMAL_CHARGE: dict[tuple[str, str], float] = {
    ("ASP", "OD1"): -0.5, ("ASP", "OD2"): -0.5,
    ("GLU", "OE1"): -0.5, ("GLU", "OE2"): -0.5,
    ("LYS", "NZ"): 1.0,
    ("ARG", "NE"): 1 / 3, ("ARG", "NH1"): 1 / 3, ("ARG", "NH2"): 1 / 3,
    # Chain termini are not special-cased: they are backbone, hence template,
    # and contribute a constant that cancels in energy *differences*.
}

#: Heavy atoms treated as H-bond donors or acceptors (distance-only term).
HBOND_ATOMS = {"N", "O"}

#: 332.0637 kcal*A/(mol*e^2) -- Coulomb constant in these units.
COULOMB_K = 332.0637


class BuiltinBackend(EnergyBackend):
    name = "builtin-screen"
    can_relax = False
    units = "kcal/mol (screening scale, not comparable to Rosetta energy units)"

    def __init__(
        self,
        cutoff: float = 12.0,
        max_repulsion: float = 10.0,
        hbond_weight: float = 1.5,
        coulomb: bool = True,
        **_ignored,
    ) -> None:
        self.cutoff = float(cutoff)
        self.max_repulsion = float(max_repulsion)
        self.hbond_weight = float(hbond_weight)
        self.coulomb = bool(coulomb)

    # -- parameter lookup -------------------------------------------------

    @staticmethod
    def _lj_arrays(s: AtomSet) -> tuple[np.ndarray, np.ndarray]:
        rmin = np.empty(len(s))
        eps = np.empty(len(s))
        for i, el in enumerate(s.elements):
            r, e = LJ.get(el, _LJ_DEFAULT)
            rmin[i] = r
            eps[i] = e
        return rmin, eps

    @staticmethod
    def _charges(s: AtomSet) -> np.ndarray:
        q = np.zeros(len(s))
        for i, (nm, rn) in enumerate(zip(s.names, s.resnames)):
            q[i] = FORMAL_CHARGE.get((rn.strip().upper(), nm.strip().upper()), 0.0)
        return q

    @staticmethod
    def _is_hbond(s: AtomSet) -> np.ndarray:
        return np.array([el in HBOND_ATOMS for el in s.elements], dtype=bool)

    # -- the contract ------------------------------------------------------

    def interaction(self, a: AtomSet, b: AtomSet) -> float:
        """Sum of pair terms between atoms of ``a`` and ``b``.

        Pairs belonging to the same residue (same ``reskeys`` entry) are
        skipped, so this stays a genuine *inter*-residue interaction even when
        the two sets happen to share a residue.
        """
        if len(a) == 0 or len(b) == 0:
            return 0.0

        d = np.sqrt(((a.coords[:, None, :] - b.coords[None, :, :]) ** 2).sum(-1))
        within = d <= self.cutoff
        if not within.any():
            return 0.0

        # Exclude same-residue pairs.
        if any(a.reskeys) and any(b.reskeys):
            ak = np.array([hash(k) for k in a.reskeys])
            bk = np.array([hash(k) for k in b.reskeys])
            within &= ak[:, None] != bk[None, :]
            if not within.any():
                return 0.0

        # Guard against exactly coincident atoms (duplicate input coordinates):
        # they would give inf and silently poison the objective.
        d = np.where(d < 0.3, 0.3, d)

        ra, ea = self._lj_arrays(a)
        rb, eb = self._lj_arrays(b)
        rmin = ra[:, None] + rb[None, :]
        eps = np.sqrt(ea[:, None] * eb[None, :])

        ratio = rmin / d
        r6 = ratio ** 6
        lj = eps * (r6 * r6 - 2.0 * r6)
        # Cap only the repulsive side; attraction is bounded by -eps anyway.
        lj = np.where(lj > self.max_repulsion, self.max_repulsion, lj)

        total = float(lj[within].sum())

        if self.coulomb:
            qa = self._charges(a)
            qb = self._charges(b)
            if np.any(qa) and np.any(qb):
                qq = qa[:, None] * qb[None, :]
                nz = within & (qq != 0.0)
                if nz.any():
                    # eps(r) = 4r  =>  E = k q1 q2 / (4 r^2)
                    total += float((COULOMB_K * qq[nz] / (4.0 * d[nz] ** 2)).sum())

        if self.hbond_weight:
            ha = self._is_hbond(a)
            hb = self._is_hbond(b)
            if ha.any() and hb.any():
                mask = within & ha[:, None] & hb[None, :]
                # Flat-bottomed well over 2.6-3.4 A, tapering to zero by 4.0 A.
                dd = d[mask]
                w = np.clip((4.0 - dd) / 0.6, 0.0, 1.0)
                w = np.where(dd < 2.4, 0.0, w)  # too close: LJ already penalizes
                total -= float(self.hbond_weight * w.sum())

        return total

    def internal(self, a: AtomSet) -> float:
        """Self-interaction within one atom set, excluding same-residue pairs."""
        if len(a) < 2:
            return 0.0
        # interaction() skips same-residue pairs and double counts i<j and j<i,
        # so halve it.
        return 0.5 * self.interaction(a, a)

    def score_structure(self, pdb) -> float:
        """Whole-structure non-bonded energy; for reporting, not for the MILP."""
        from ..structure import Structure

        st = Structure.load(pdb)
        names, coords, resn, keys = [], [], [], []
        for res in list(st.residues()) + list(st.hetero_residues()):
            ref = st.ref(res)
            for at in res:
                if at.element == "H":
                    continue
                names.append(at.get_name())
                coords.append(at.coord)
                resn.append(res.get_resname().strip())
                keys.append(ref.key)
        s = AtomSet(names=names, coords=np.array(coords), resnames=resn, reskeys=keys)
        return self.internal(s)
