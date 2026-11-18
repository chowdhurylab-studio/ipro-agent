"""Structure handling: loading, selection, neighbours, sequences.

Thin layer over Biopython's PDB parser. Its job is to give the rest of the
package a stable vocabulary -- :class:`ResidueRef`, neighbour lists, interface
detection -- so that the optimizer never touches file formats directly, and so
a different structure backend could be swapped in.

Conventions
-----------
* A residue is identified by ``(chain, number, insertion_code)``, matching
  :class:`ipro_agent.config.DesignPosition`.
* Coordinates are numpy arrays in Angstroms.
* Hydrogens are kept if present but never required; the built-in energy model
  is heavy-atom based (see :mod:`ipro_agent.energy.builtin`).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

try:
    from Bio.PDB import PDBParser, PDBIO, Select, MMCIFParser
    from Bio.PDB.Structure import Structure as BioStructure
    from Bio.PDB.Residue import Residue as BioResidue
    _HAVE_BIO = True
except ModuleNotFoundError:  # pragma: no cover
    _HAVE_BIO = False
    BioStructure = object  # type: ignore
    BioResidue = object  # type: ignore

from .config import THREE_TO_ONE

#: Residue names that are not amino acids but are commonly present and should
#: be treated as ligand/solvent rather than chain residues.
WATERS = {"HOH", "WAT", "DOD", "TIP3"}


@dataclass(frozen=True)
class ResidueRef:
    """A reference to one residue, independent of any parser object."""

    chain: str
    number: int
    insertion_code: str = ""
    name: str = ""

    @property
    def label(self) -> str:
        return f"{self.chain}{self.number}{self.insertion_code}".strip()

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.chain, self.number, self.insertion_code)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name}{self.label}" if self.name else self.label


class StructureError(RuntimeError):
    """A structure could not be read, or does not contain what was asked for."""


class Structure:
    """A loaded structure, restricted to the chains of interest."""

    def __init__(self, bio: BioStructure, path: Path | None = None, model: int = 0):
        if not _HAVE_BIO:  # pragma: no cover
            raise StructureError("Biopython is required: pip install biopython")
        self._bio = bio
        self.path = path
        try:
            self.model = bio[model] if not isinstance(bio, type(None)) else None
        except KeyError as exc:
            raise StructureError(f"model {model} not in {path}") from exc

    # -- loading ----------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str | Path,
        chains: Sequence[str] | None = None,
        model: int = 0,
    ) -> "Structure":
        """Read a PDB or mmCIF file.

        ``chains`` restricts to those chain IDs; a chain named in ``chains``
        that is absent from the file is an error rather than a silent omission,
        because a missing partner would change the energy without warning.
        """
        if not _HAVE_BIO:  # pragma: no cover
            raise StructureError("Biopython is required: pip install biopython")
        p = Path(path).expanduser()
        if not p.exists():
            raise StructureError(f"no such structure file: {p}")
        parser = (
            MMCIFParser(QUIET=True)
            if p.suffix.lower() in (".cif", ".mmcif")
            else PDBParser(QUIET=True)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bio = parser.get_structure(p.stem, str(p))
        st = cls(bio, path=p, model=model)
        if chains:
            have = {c.id for c in st.model}
            missing = [c for c in chains if c not in have]
            if missing:
                raise StructureError(
                    f"{p.name}: requested chain(s) {missing} not present; "
                    f"file has {sorted(have)}"
                )
            for ch in list(st.model):
                if ch.id not in chains:
                    st.model.detach_child(ch.id)
        return st

    # -- iteration --------------------------------------------------------

    def chain_ids(self) -> list[str]:
        return [c.id for c in self.model]

    def residues(self, chain: str | None = None, amino_acids_only: bool = True) -> Iterator[BioResidue]:
        for ch in self.model:
            if chain is not None and ch.id != chain:
                continue
            for res in ch:
                if amino_acids_only:
                    if res.get_resname().strip() in WATERS:
                        continue
                    # hetero flag non-blank means HETATM (ligand, ion, water)
                    if res.id[0].strip():
                        continue
                yield res

    def hetero_residues(self, chain: str | None = None) -> Iterator[BioResidue]:
        """Ligands, cofactors and ions (not water)."""
        for ch in self.model:
            if chain is not None and ch.id != chain:
                continue
            for res in ch:
                if res.id[0].strip() and res.get_resname().strip() not in WATERS:
                    yield res

    def ref(self, res: BioResidue) -> ResidueRef:
        ch = res.get_parent().id
        _, num, icode = res.id
        return ResidueRef(ch, int(num), icode.strip(), res.get_resname().strip())

    def find(self, chain: str, number: int, insertion_code: str = "") -> BioResidue:
        try:
            ch = self.model[chain]
        except KeyError as exc:
            raise StructureError(
                f"chain {chain!r} not in {self.path.name if self.path else '?'}; "
                f"have {self.chain_ids()}"
            ) from exc
        for res in ch:
            _, num, icode = res.id
            if int(num) == int(number) and icode.strip() == (insertion_code or "").strip():
                return res
        raise StructureError(
            f"residue {chain}{number}{insertion_code} not found; chain {chain} spans "
            f"{self._chain_range(chain)}"
        )

    def _chain_range(self, chain: str) -> str:
        nums = [int(r.id[1]) for r in self.model[chain]]
        return f"{min(nums)}-{max(nums)}" if nums else "(empty)"

    # -- sequence ---------------------------------------------------------

    def sequence(self, chain: str | None = None) -> str:
        """One-letter sequence; non-standard residues become 'X'."""
        return "".join(
            THREE_TO_ONE.get(r.get_resname().strip(), "X")
            for r in self.residues(chain)
        )

    def numbered_sequence(self, chain: str) -> list[tuple[ResidueRef, str]]:
        return [
            (self.ref(r), THREE_TO_ONE.get(r.get_resname().strip(), "X"))
            for r in self.residues(chain)
        ]

    # -- geometry ---------------------------------------------------------

    def coords(self, res: BioResidue, heavy_only: bool = True) -> np.ndarray:
        return np.array(
            [a.coord for a in res if not (heavy_only and a.element == "H")],
            dtype=float,
        )

    def all_atoms(self, heavy_only: bool = True) -> list:
        out = []
        for ch in self.model:
            for res in ch:
                if res.get_resname().strip() in WATERS:
                    continue
                for a in res:
                    if heavy_only and a.element == "H":
                        continue
                    out.append(a)
        return out

    def neighbours(
        self,
        target: BioResidue,
        cutoff: float = 8.0,
        same_chain: bool = True,
        amino_acids_only: bool = True,
    ) -> list[ResidueRef]:
        """Residues with any heavy atom within ``cutoff`` A of ``target``.

        Used to decide which non-design residues must be repacked alongside a
        mutation: IPRO re-optimizes the rotamers of a mutated position *and its
        neighbourhood*, because a new side chain invalidates the packing around
        it.
        """
        tc = self.coords(target)
        if tc.size == 0:
            return []
        out: list[ResidueRef] = []
        tref = self.ref(target)
        for res in self.residues(amino_acids_only=amino_acids_only):
            r = self.ref(res)
            if r.key == tref.key:
                continue
            if not same_chain and r.chain == tref.chain:
                continue
            rc = self.coords(res)
            if rc.size == 0:
                continue
            # cdist without scipy: broadcasting is fine at residue scale
            d = np.sqrt(((tc[:, None, :] - rc[None, :, :]) ** 2).sum(-1)).min()
            if d <= cutoff:
                out.append(r)
        return out

    def center_of_mass(self, residues: Iterable[BioResidue] | None = None) -> np.ndarray:
        pts = []
        for res in (residues if residues is not None else self.residues()):
            c = self.coords(res)
            if c.size:
                pts.append(c)
        if not pts:
            raise StructureError("no atoms to average")
        return np.vstack(pts).mean(axis=0)

    # -- writing ----------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        io = PDBIO()
        io.set_structure(self._bio)
        io.save(str(p))
        return p

    def copy(self) -> "Structure":
        import copy as _copy
        return Structure(_copy.deepcopy(self._bio), path=self.path)


# ---------------------------------------------------------------------------
# Multi-molecule assembly
# ---------------------------------------------------------------------------


def interface_residues(
    design: Structure,
    partner: Structure,
    cutoff: float = 5.0,
) -> list[ResidueRef]:
    """Residues of ``design`` with a heavy atom within ``cutoff`` A of ``partner``.

    This is how design positions get proposed automatically when the user says
    "the interface" instead of listing residue numbers: it is the operational
    definition of an interface used throughout the IPRO case studies (a
    distance cutoff from the target molecule).
    """
    pts = []
    for r in list(partner.residues()) + list(partner.hetero_residues()):
        c = partner.coords(r)
        if c.size:
            pts.append(c)
    if not pts:
        raise StructureError("partner molecule has no heavy atoms")
    pc = np.vstack(pts)

    out: list[ResidueRef] = []
    for res in design.residues():
        rc = design.coords(res)
        if rc.size == 0:
            continue
        d = np.sqrt(((rc[:, None, :] - pc[None, :, :]) ** 2).sum(-1)).min()
        if d <= cutoff:
            out.append(design.ref(res))
    return out


def residues_near_point(
    st: Structure, point: np.ndarray, cutoff: float = 8.0
) -> list[ResidueRef]:
    """Residues within ``cutoff`` of a point -- e.g. a pore axis or a site."""
    out = []
    for res in st.residues():
        c = st.coords(res)
        if c.size == 0:
            continue
        if np.sqrt(((c - point) ** 2).sum(-1)).min() <= cutoff:
            out.append(st.ref(res))
    return out
