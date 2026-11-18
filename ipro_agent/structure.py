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

from .config import THREE_TO_ONE, position_label

#: Residue names that are not amino acids but are commonly present and should
#: be treated as ligand/solvent rather than chain residues.
WATERS = {"HOH", "WAT", "DOD", "TIP3"}

#: Biopython warning text that means atoms or residues were **discarded**, not
#: merely reformatted. These are the ones that must not be swallowed. A
#: "discontinuous chain" warning is deliberately NOT here: ATOM records
#: followed by HETATM records of the same chain are ordinary and lose nothing.
_LOSSY_WARNINGS = ("redefined", "defined twice", "duplicate")

#: Hydrogen isotopes. The heavy-atom filters tested ``element == "H"`` only, so
#: deuterium from a neutron structure counted as a heavy atom and was scored
#: with carbon's Lennard-Jones parameters.
HYDROGEN_ELEMENTS = {"H", "D", "T"}


def is_hydrogen(atom) -> bool:
    """True for H/D/T, by the file's element column."""
    return (getattr(atom, "element", "") or "").strip().upper() in HYDROGEN_ELEMENTS


@dataclass(frozen=True)
class ResidueRef:
    """A reference to one residue, independent of any parser object."""

    chain: str
    number: int
    insertion_code: str = ""
    name: str = ""

    @property
    def label(self) -> str:
        """Same round-tripping text form a design position uses.

        These two labels are compared as strings all over the package (the
        template's design filter, the loop's identity map, the restraint
        planner), so when this one concatenated chain and number without a
        separator it disagreed with ``DesignPosition.label`` for every
        non-letter chain id: chain ``1`` residue 115 was ``1115`` here and
        ``1:115`` there, and ``1115`` re-parses as chain A residue 1115. The
        measured result was a mutation applied 40 A away, to a residue of the
        same type, with nothing raised. See :func:`ipro_agent.config.position_label`.
        """
        return position_label(self.chain, self.number, self.insertion_code)

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
            # A file the parser accepted but that holds no models at all is
            # almost always not a coordinate file: a FASTA, a sequence dump or
            # a text file with a .pdb name parses "successfully" and yields
            # nothing. "model 0 not in <path>" is true but tells the reader
            # nothing they can act on.
            n_models = len(list(bio)) if bio is not None else 0
            if n_models == 0:
                raise StructureError(
                    f"{path} contains no atoms the parser recognised. A PDB "
                    f"needs ATOM/HETATM records -- a FASTA, a sequence listing "
                    f"or any text file with a .pdb name parses to nothing. "
                    f"Check the first lines of the file."
                ) from exc
            raise StructureError(
                f"model {model} not in {path}: it has {n_models} "
                f"model(s), numbered {[m.id for m in bio]}"
            ) from exc

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
        # Every structure in this package arrives through here, so the
        # unreadable-input cases belong here too. They used to surface as the
        # parser's own exceptions -- `ValueError: Empty file.` and
        # `IsADirectoryError` -- which callers that catch StructureError (every
        # one of them) let through as a bare traceback.
        if p.is_dir():
            raise StructureError(
                f"{p} is a directory, not a structure file. Did you mean a file "
                f"inside it?"
            )
        if p.stat().st_size == 0:
            raise StructureError(f"{p} is empty (0 bytes): nothing to read")
        # QUIET=False, deliberately: QUIET suppresses the construction warnings
        # inside the parser, so there is nothing left to inspect. They are
        # captured here instead, which keeps the console as quiet as before
        # while letting the ones that mean data loss be acted on.
        parser = (
            MMCIFParser(QUIET=False)
            if p.suffix.lower() in (".cif", ".mmcif")
            else PDBParser(QUIET=False)
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                bio = parser.get_structure(p.stem, str(p))
            except StructureError:
                raise
            except Exception as exc:
                # Anything the parser throws becomes the one exception type
                # callers handle, with the parser's own words kept.
                raise StructureError(
                    f"cannot parse {p.name} as "
                    f"{'mmCIF' if p.suffix.lower() in ('.cif', '.mmcif') else 'PDB'}"
                    f" ({type(exc).__name__}: {exc}). Check it is a coordinate "
                    f"file and not, say, a FASTA, a gzipped PDB, or an mmCIF "
                    f"named .pdb."
                ) from exc
        # Biopython's construction warnings are the only signal that the file
        # was not read as written: a residue id used twice, an atom name used
        # twice in one residue, blank altLocs on a duplicated residue. With
        # QUIET=True *and* the warnings discarded, two copies of a molecule
        # pasted into one chain loaded as 82 atoms of 164, and a chain holding
        # two different residues numbered 115 silently dropped the whole TRP --
        # every energy afterwards computed on half the molecule, with the
        # reduced counts reported back as the file's contents.
        lossy = [
            str(w.message) for w in caught
            if any(t in str(w.message).lower() for t in _LOSSY_WARNINGS)
        ]
        if lossy:
            raise StructureError(
                f"{p.name}: the PDB parser discarded atoms or residues while "
                f"reading this file, so what was loaded is not what the file "
                f"contains:\n  " + "\n  ".join(sorted(set(lossy))[:6]) +
                "\nFix the file (duplicate residue numbers, repeated atom names, "
                "or two molecules merged into one chain) rather than running on "
                "part of it."
            )
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
            [a.coord for a in res if not (heavy_only and is_hydrogen(a))],
            dtype=float,
        )

    def all_atoms(self, heavy_only: bool = True) -> list:
        out = []
        for ch in self.model:
            for res in ch:
                if res.get_resname().strip() in WATERS:
                    continue
                for a in res:
                    if heavy_only and is_hydrogen(a):
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
        """Write the model this Structure holds -- and only that model.

        Writing ``self._bio`` wrote the whole file back: every model and every
        chain, including the chains ``load(chains=...)`` was asked to exclude
        (the filter only detaches from model 0). Measured, a Structure holding
        2 atoms of chain A saved 6 ATOM lines across 2 models and chains
        A and B. These are the files PyMOL and Vina then read, so the partner
        the user excluded was back in the scoring input while this object's own
        energies never saw it.
        """
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        io = PDBIO()
        io.set_structure(self.model if self.model is not None else self._bio)
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
