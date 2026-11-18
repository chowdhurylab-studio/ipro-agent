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

2. **A bounded pair term.** The repulsive LJ branch is clamped at
   ``max_repulsion``, and then the *whole* pair energy -- LJ plus Coulomb plus
   the H-bond well -- is bounded to ``[-max_repulsion, +max_repulsion]``.
   Rotamer libraries routinely produce hard clashes whose raw 1/r^12 values are
   astronomically large; left uncapped they dominate the MILP objective and make
   the integer program numerically ill-conditioned. Softened repulsion is
   standard practice in side-chain packing. The bound is two-sided because the
   Coulomb term diverges in the *attractive* direction too, which made a buried
   counter-charge the most favourable thing the function could report.

3. **Charged-pair Coulomb** with a distance-dependent dielectric
   ``eps(r) = 4r``, applied only to formally charged side-chain atoms. A full
   partial-charge set is deliberately *not* used: an incomplete one would be
   worse than an explicit, documented restriction to the groups that dominate
   electrostatics in design scoring.

4. **A geometric hydrogen-bond well** between N/O donor-acceptor heavy atoms,
   as a distance-only term (no angular dependence).

Pairs that are **not** non-bonded are excluded: atoms of the same residue, and
atoms that are 1-2 or 1-3 across a peptide bond (see :data:`PEPTIDE_12_13`).
Without that second exclusion a chain's own covalent geometry was scored as
packing, which on the example structure was larger than the entire genuine
non-bonded sum and decided one backbone perturbation in five.

There is **no desolvation term**: a buried polar or charged group pays no
penalty for being buried. That is the largest single omission relative to the
CHARMM/Lazaridis-Karplus energy IPRO used, and it is why charged positions need
reading with care.

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

#: Backbone atom names, for the peptide-bond exclusions below.
BACKBONE_NAMES = {"N", "CA", "C", "O", "OXT", "OT1", "OT2", "OT"}

#: Atom-name pairs that are covalently 1-2 or 1-3 across a peptide bond, given
#: as (atom of residue i, atom of residue i+1):
#:
#:     ... CA(i) - C(i)(=O(i)) - N(i+1) - CA(i+1) ...
#:
#: 1-2 is C(i)-N(i+1) itself; the 1-3 pairs are those two bonds' other ends,
#: plus CD(i+1) which is bonded to N in proline. Everything else between
#: sequential residues is 1-4 or further and is a genuine non-bonded pair.
PEPTIDE_12_13: frozenset[tuple[str, str]] = frozenset({
    ("C", "N"),                     # 1-2, the peptide bond
    ("CA", "N"), ("O", "N"),        # 1-3 through C(i)
    ("C", "CA"), ("C", "CD"),       # 1-3 through N(i+1), CD for proline
})

#: Only these names can appear in :data:`PEPTIDE_12_13`, so only these need a
#: code in the lookup table.
_PEPTIDE_ATOMS = ("N", "CA", "C", "O", "CD")


class BuiltinBackend(EnergyBackend):
    name = "builtin-screen"
    can_relax = False
    units = "kcal/mol (screening scale, not comparable to Rosetta energy units)"
    units_short = "kcal/mol, screening scale"

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

    @staticmethod
    def _residue_order(
        s: AtomSet, chain_ids: dict[object, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per atom: its residue number, and a code for the chain it is in.

        A residue key is ``(chain, number, icode)``, optionally prefixed by a
        molecule discriminator. The number is the first integer in the key and
        everything else is the chain identity, mapped through the shared
        ``chain_ids`` dict so the codes are comparable between the two sets.
        Two residues therefore count as sequential only within one chain of one
        molecule. A residue with an insertion code is not treated as sequential
        with its base residue, so a 115/115A pair keeps its 1-2 and 1-3 pairs;
        that is a rare case and it only reinstates the old overcount.
        """
        nums = np.full(len(s), -(10 ** 9), dtype=np.int64)
        codes = np.empty(len(s), dtype=np.int64)
        for i, k in enumerate(s.reskeys):
            parts = tuple(k) if isinstance(k, tuple) else (k,)
            num = None
            rest = []
            for p in parts:
                if num is None and isinstance(p, int) and not isinstance(p, bool):
                    num = p
                else:
                    rest.append(p)
            if num is not None:
                nums[i] = num
            key = tuple(rest)
            codes[i] = chain_ids.setdefault(key, len(chain_ids))
        return nums, codes

    @staticmethod
    def _peptide_codes(s: AtomSet) -> np.ndarray:
        """Index into :data:`_PEPTIDE_ATOMS`, or the last row (never excluded)."""
        other = len(_PEPTIDE_ATOMS)
        out = np.full(len(s), other, dtype=np.int64)
        for i, nm in enumerate(s.names):
            n = nm.strip().upper()
            if n in _PEPTIDE_ATOMS:
                out[i] = _PEPTIDE_ATOMS.index(n)
        return out

    @classmethod
    def _peptide_table(cls) -> np.ndarray:
        n = len(_PEPTIDE_ATOMS) + 1
        t = np.zeros((n, n), dtype=bool)
        for i, ai in enumerate(_PEPTIDE_ATOMS):
            for j, aj in enumerate(_PEPTIDE_ATOMS):
                t[i, j] = (ai, aj) in PEPTIDE_12_13
        return t

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

        # Exclude same-residue pairs. Keys are mapped to small integers through
        # a dict rather than hashed: a hash collision would silently drop a real
        # interaction, and silently dropping terms is exactly the failure mode
        # this function must not have. Residue keys are always populated (see
        # AtomSet.__post_init__), so this is unconditional -- in particular it
        # guarantees an atom is never scored against itself.
        ids: dict[object, int] = {}
        ak = np.fromiter(
            (ids.setdefault(k, len(ids)) for k in a.reskeys), dtype=np.int64, count=len(a)
        )
        bk = np.fromiter(
            (ids.setdefault(k, len(ids)) for k in b.reskeys), dtype=np.int64, count=len(b)
        )
        within &= ak[:, None] != bk[None, :]
        if not within.any():
            return 0.0

        # Exclude pairs that are covalently 1-2 or 1-3 across a peptide bond.
        # The same-residue test above cannot see them -- they span two residues
        # -- so a crystal structure's own chain connectivity was being scored as
        # if it were packing. Measured on examples/ompf_pore/2omf_A.pdb:
        # internal() reported +11132.191 kcal/mol, of which +3390.000 was the
        # 339 peptide bonds (every one pinned at the repulsion cap) and
        # +10390.954 the 1048 1-3 pairs, against a genuine non-bonded sum of
        # -1070.880. Worse than the offset, it steered the search: of 25
        # default backbone perturbations, 5 were hard-rejected on chain
        # connectivity that a perturbation cannot change.
        chain_ids: dict[object, int] = {}
        na, ca_chain = self._residue_order(a, chain_ids)
        nb, cb_chain = self._residue_order(b, chain_ids)
        same_chain = ca_chain[:, None] == cb_chain[None, :]
        fwd = same_chain & (nb[None, :] == na[:, None] + 1)
        rev = same_chain & (nb[None, :] == na[:, None] - 1)
        if fwd.any() or rev.any():
            table = self._peptide_table()
            ca_code = self._peptide_codes(a)
            cb_code = self._peptide_codes(b)
            bonded = (fwd & table[ca_code[:, None], cb_code[None, :]]) | (
                rev & table[cb_code[None, :], ca_code[:, None]]
            )
            within &= ~bonded
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

        pair = lj

        qa = self._charges(a)
        qb = self._charges(b)
        qq = qa[:, None] * qb[None, :]

        if self.coulomb and np.any(qa) and np.any(qb):
            nz = qq != 0.0
            if (within & nz).any():
                # eps(r) = 4r  =>  E = k q1 q2 / (4 r^2)
                coul = np.zeros_like(pair)
                np.divide(
                    COULOMB_K * qq, 4.0 * d ** 2, out=coul, where=nz
                )
                pair = pair + coul

        if self.hbond_weight:
            ha = self._is_hbond(a)
            hb = self._is_hbond(b)
            if ha.any() and hb.any():
                # The H-bond bonus applies only where neither atom's formal
                # charge already decides the interaction:
                #   qq > 0  like-charged (e.g. two ARG NH1). A bonus here flips
                #           the sign of a repulsion -- two ARG nitrogens at
                #           3.4 A came out attractive.
                #   qq < 0  oppositely charged. The Coulomb term already
                #           rewards it; adding an H-bond well double counts the
                #           same physical contact.
                # What is left is the neutral polar donor/acceptor pairs the
                # term is actually meant for. This remains a distance-only
                # approximation with no angular dependence.
                mask = ha[:, None] & hb[None, :] & (qq == 0.0)
                # ... and not between two backbone atoms of sequential
                # residues. Their separation is fixed by the peptide geometry,
                # so it cannot represent a hydrogen bond: on 2omf_A this term
                # was paying -1.5 kcal/mol to each of 339 N(i)...N(i+1) pairs
                # -- one per peptide bond, two amide nitrogens that cannot bond
                # to each other -- and 98 O(i)...O(i+1) pairs, -598.407 in all.
                # A side chain reaching across a peptide bond is a real H-bond
                # and is left alone.
                bb_a = np.array(
                    [nm.strip().upper() in BACKBONE_NAMES for nm in a.names], dtype=bool
                )
                bb_b = np.array(
                    [nm.strip().upper() in BACKBONE_NAMES for nm in b.names], dtype=bool
                )
                mask &= ~((fwd | rev) & bb_a[:, None] & bb_b[None, :])
                mask &= within
                if mask.any():
                    # Flat-bottomed well over 2.4-3.4 A, tapering out by 4.0 A.
                    w = np.clip((4.0 - d) / 0.6, 0.0, 1.0)
                    w = np.where(d < 2.4, 0.0, w)   # too close: LJ penalizes
                    pair = pair - self.hbond_weight * np.where(mask, w, 0.0)

        # Bound the whole pair term, not just its LJ part. The repulsion cap was
        # documented as "10 kcal/mol per atom pair" but applied to LJ alone,
        # while the Coulomb term runs as 1/r^2 down to the 0.3 A clamp with no
        # bound at all. So the most *attractive* thing this function could
        # produce was a hard clash between opposite charges: measured, ASP OD1
        # against LYS NZ gave -3.577 at a 3.50 A salt bridge but -31.508 at
        # 1.00 A and -451.200 at 0.30 A. A rotamer burying NZ inside a template
        # oxygen scored -71.957 against the native alanine's -1.391 and came
        # back from the MILP as the proven-optimal design. Like charges ran away
        # the same way (+932 for one pair at 0.30 A), past the max_self_energy
        # and hard_reject thresholds that are calibrated against this bound.
        # Only pairs closer than about 1.8 A are affected; every real contact
        # is well inside the cap.
        np.clip(pair, -self.max_repulsion, self.max_repulsion, out=pair)
        return float(pair[within].sum())

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
        names, coords, resn, keys, els = [], [], [], [], []
        for res in list(st.residues()) + list(st.hetero_residues()):
            ref = st.ref(res)
            for at in res:
                if (at.element or "").strip().upper() in ("H", "D"):
                    continue
                names.append(at.get_name())
                coords.append(at.coord)
                resn.append(res.get_resname().strip())
                keys.append(ref.key)
                els.append((at.element or "").strip().upper())
        s = AtomSet(
            names=names, coords=np.array(coords), resnames=resn, reskeys=keys,
            elements=els if all(els) else [],
        )
        return self.internal(s)
