"""The energy-backend contract.

Everything the optimizer needs from an energy model is expressed here, so that
PyRosetta, a built-in screening function, or anything else can be swapped in.

The contract is shaped by what the MILP needs. IPRO's rotamer-selection problem
is only a *mixed-integer linear* program because the energy is assumed
**pairwise decomposable** over rotamer choices:

    E(choice) = sum_i E_self(i, r_i) + sum_{i<j} E_pair(i, r_i; j, r_j)

where ``E_self`` is a rotamer's interaction with everything held fixed (the
template: backbone, non-design side chains, and the target molecules) plus its
own internal strain, and ``E_pair`` is the interaction between two chosen
rotamers. A backend therefore only has to answer one question well --
*what is the interaction energy between these two sets of atoms?* -- and the
self/pair tables are built from it by :mod:`ipro_agent.milp`.

Backends must be **deterministic**: the same atoms must give the same number, or
the MILP's optimum stops being reproducible and accept/reject becomes noise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import itertools

import numpy as np

#: Marker and counter for atoms supplied without residue keys; see
#: :meth:`AtomSet.__post_init__`.
_UNKEYED = "__unkeyed__"
_KEY_COUNTER = itertools.count()

#: Element symbols of two letters that occur in protein structure files. A
#: standard amino acid never has a side-chain atom named like one of these, so
#: within a non-amino-acid residue the whole name is the element.
_TWO_LETTER = {
    "CL", "BR", "SE", "FE", "ZN", "MG", "MN", "CU", "NI", "CO", "CA", "NA",
    "CD", "HG", "PT", "AU", "AG", "MO", "SR", "BA", "CS", "RB", "LI", "AL",
    "SI", "AS", "SB", "PB", "TL", "GD", "SM", "EU", "YB", "TB", "HO", "W",
}

#: The 20 standard amino acids, where an atom name like ``CA`` or ``SD`` is a
#: protein atom rather than an element symbol. Written out rather than imported
#: from config to keep the energy layer free of spec types.
_STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    # Protonation variants and the common alternatives, still protein.
    "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "ASH", "GLH",
    "LYN", "MSE",
}


#: ``(residue, atom)`` pairs inside residues that are otherwise protein, where
#: the atom name really is a two-letter element symbol. Selenomethionine is the
#: common case: MSE has to count as protein so that its ``CA`` stays the alpha
#: carbon rather than calcium, but its ``SE`` is genuinely selenium.
_RESIDUE_ATOM_ELEMENTS = {
    ("MSE", "SE"): "SE", ("SEC", "SE"): "SE", ("CSE", "SE"): "SE",
}


def guess_element(name: str, resname: str = "") -> str:
    """Element symbol for a PDB atom name, as well as a name alone allows.

    PDB atom names put the element first for protein atoms (CB, NZ, OD1, SD),
    so taking the first alphabetic character is right *there* -- but it was
    applied to everything, which made every two-letter element unreachable:
    ``FE`` typed as fluorine, ``CL`` and ``CA`` as carbon, ``ZN`` and ``MG`` as
    the default. Measured, a backbone oxygen 2.40 A from an iron came out at
    +5.83 kcal/mol using fluorine's parameters, and a sodium ion was typed as
    nitrogen and so handed out -1.5 kcal/mol hydrogen bonds. ``LJ``'s own
    ``CL``/``BR``/``SE`` entries were dead code.

    The ambiguity is real: ``CA`` is the alpha carbon of a residue and calcium
    in an ion, and a name alone cannot tell them apart. The residue name can:
    inside a standard amino acid the name is a protein atom, and a monatomic
    ion is named after itself (``HETATM ... CA    CA``). Prefer the file's own
    element column (``AtomSet.elements``) wherever the caller has it.
    """
    s = str(name).strip().upper().lstrip("0123456789")
    if not s:
        return "C"
    rn = str(resname).strip().upper()
    if (rn, s) in _RESIDUE_ATOM_ELEMENTS:
        return _RESIDUE_ATOM_ELEMENTS[(rn, s)]
    if rn not in _STANDARD_RESIDUES:
        if s == rn and s in _TWO_LETTER:          # monatomic ion: CA/CA, ZN/ZN
            return s
        if s in _TWO_LETTER:                      # FE, CL, SE of a cofactor
            return s
    return s[0]


@dataclass
class AtomSet:
    """A bag of atoms with just enough typing to score them.

    Deliberately not a Biopython object: rotamer coordinates arrive from PyMOL
    as plain lists, and the template comes from Biopython, so both are
    normalized into this.
    """

    names: list[str] = field(default_factory=list)
    coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    #: Parent residue name per atom (ASP, TRP, ...), for charge/H-bond typing.
    resnames: list[str] = field(default_factory=list)
    #: Residue key per atom, so a backend can skip intra-residue pairs.
    reskeys: list[tuple] = field(default_factory=list)
    #: Element symbol per atom. Supply it whenever the source knows it (the
    #: PDB element column, which Biopython parses): guessing from the atom
    #: name cannot be made reliable, because ``CA`` is the alpha carbon in a
    #: residue and calcium in an ion. Left empty, it is guessed by
    #: :func:`guess_element` and the guess is only as good as the name.
    elements: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=float).reshape(-1, 3)
        n = len(self.coords)
        if len(self.names) != n:
            raise ValueError(f"{len(self.names)} names for {n} coords")
        # Checked, not truncated. A short resnames list silently zeroed the
        # trailing atoms' formal charges, which turned a measured -7.21
        # kcal/mol carboxylate/ammonium interaction into +5.60 with nothing
        # raised.
        for label, seq in (("resnames", self.resnames), ("reskeys", self.reskeys),
                           ("elements", self.elements)):
            if seq and len(seq) != n:
                raise ValueError(f"{len(seq)} {label} for {n} coords")
        if not self.resnames:
            self.resnames = [""] * n
        if not self.elements:
            self.elements = [
                guess_element(nm, rn) for nm, rn in zip(self.names, self.resnames)
            ]
        if not self.reskeys:
            # Each atom gets a distinct placeholder key rather than a shared
            # empty tuple. With a shared key the backend's same-residue
            # exclusion either skipped everything or -- worse, when the guard
            # treated the empty tuple as "no information" -- let an atom be
            # scored against itself, which made internal() report a large
            # repulsion for atoms that were simply the same atom. Distinct keys
            # mean an atom never pairs with itself, and two separately
            # constructed sets never collide.
            self.reskeys = [(_UNKEYED, next(_KEY_COUNTER)) for _ in range(n)]

    def __len__(self) -> int:
        return len(self.coords)

    def extend(self, other: "AtomSet") -> "AtomSet":
        if len(other) == 0:
            return self
        if len(self) == 0:
            return other
        return AtomSet(
            names=self.names + other.names,
            coords=np.vstack([self.coords, other.coords]),
            resnames=self.resnames + other.resnames,
            reskeys=self.reskeys + other.reskeys,
            elements=self.elements + other.elements,
        )

    @classmethod
    def concat(cls, parts: Sequence["AtomSet"]) -> "AtomSet":
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls()
        return cls(
            names=[n for p in parts for n in p.names],
            coords=np.vstack([p.coords for p in parts]),
            resnames=[r for p in parts for r in p.resnames],
            reskeys=[k for p in parts for k in p.reskeys],
            elements=[e for p in parts for e in p.elements],
        )

    def bounding_radius(self) -> float:
        if len(self) == 0:
            return 0.0
        c = self.coords.mean(axis=0)
        return float(np.sqrt(((self.coords - c) ** 2).sum(-1)).max())

    def centroid(self) -> np.ndarray:
        return self.coords.mean(axis=0) if len(self) else np.zeros(3)


class EnergyBackend(ABC):
    """Scores atom sets and, optionally, relaxes structures."""

    #: Short identifier reported in results so a number can be traced to a model.
    name: str = "abstract"
    #: True if :meth:`relax` does real backbone relaxation.
    can_relax: bool = False
    #: Units of the returned energies, for honest reporting.
    units: str = "kcal/mol"
    #: The same thing in a few words, for a fixed-column file header. A PDB
    #: record is 80 columns; the full `units` sentence made one REMARK line
    #: 230 characters long, which is not a valid PDB record at all.
    units_short: str = "kcal/mol"
    #: True when :meth:`interaction` and :meth:`internal` -- the numbers the
    #: MILP optimizes and the loop reports -- come from the built-in screening
    #: function rather than this backend's own force field. It is a separate
    #: question from :attr:`can_relax`, and conflating the two mislabelled a
    #: run's output: the PyRosetta backend can relax, so the "these are
    #: screening energies" caveat was suppressed, while by default its pair
    #: tables *are* the screening terms and the written PDB carried them under
    #: a "Rosetta Energy Units" heading.
    screening_pairwise: bool = True

    @abstractmethod
    def interaction(self, a: AtomSet, b: AtomSet) -> float:
        """Non-bonded interaction energy between two disjoint atom sets.

        Must be symmetric and must ignore pairs within the same residue.
        """

    def internal(self, a: AtomSet) -> float:
        """Internal (intra-set) energy. Default 0: rotamer strain is supplied
        separately by the rotamer library, so most backends need not model it."""
        return 0.0

    def relax(self, pdb_in: Path, pdb_out: Path, fixed: Sequence[str] = ()) -> float:
        """Relax a structure, writing to ``pdb_out``; return its score.

        Backends that cannot relax must raise :class:`NotImplementedError` and
        leave ``can_relax`` False, so the loop reports a reduced protocol rather
        than silently skipping a step the papers include.
        """
        raise NotImplementedError(
            f"{self.name} cannot relax structures; install PyRosetta for the "
            "backbone-relaxation step (see docs/INSTALL.md)"
        )

    def score_structure(self, pdb: Path) -> float:
        """Total energy of a whole structure. Used for reporting, not the MILP."""
        raise NotImplementedError(f"{self.name} has no whole-structure score")

    def describe(self) -> str:
        return (
            f"{self.name} ({self.units}); "
            f"relaxation: {'yes' if self.can_relax else 'no'}"
        )


def resolve(preference: str = "auto", **kwargs) -> EnergyBackend:
    """Pick a backend by name, falling back with a clear message.

    ``auto`` prefers PyRosetta (the force field IPRO-era work and the user's
    stated dependency use) and falls back to the built-in screening function so
    that the loop and the example case remain runnable without it.
    """
    pref = (preference or "auto").lower()
    if pref in ("pyrosetta", "rosetta", "auto"):
        try:
            from .pyrosetta_backend import PyRosettaBackend

            return PyRosettaBackend(**kwargs)
        except Exception as exc:
            if pref != "auto":
                raise RuntimeError(
                    f"energy backend 'pyrosetta' requested but unavailable: {exc}"
                ) from exc
    if pref in ("builtin", "auto", "screen"):
        from .builtin import BuiltinBackend

        return BuiltinBackend(**kwargs)
    raise ValueError(f"unknown energy backend {preference!r}")
