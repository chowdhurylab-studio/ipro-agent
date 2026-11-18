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

import numpy as np


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

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords, dtype=float).reshape(-1, 3)
        n = len(self.coords)
        if len(self.names) != n:
            raise ValueError(f"{len(self.names)} names for {n} coords")
        if not self.resnames:
            self.resnames = [""] * n
        if not self.reskeys:
            self.reskeys = [()] * n

    def __len__(self) -> int:
        return len(self.coords)

    @property
    def elements(self) -> list[str]:
        """Element guessed from the PDB atom name.

        PDB atom names put the element first for protein atoms (CB, NZ, OD1,
        SD), so the first alphabetic character is the element. Two-letter
        elements do not occur among standard protein side-chain atoms.
        """
        out = []
        for nm in self.names:
            s = nm.strip().lstrip("0123456789")
            out.append(s[0].upper() if s else "C")
        return out

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
