"""Writing designed structures out.

A design that exists only as a sequence and a score is not much use: the point
of structure-based design is the coordinates. This module assembles the complex
that was actually scored -- the mutated design molecule with the chosen rotamer
at each design position, plus every partner at its current placement -- into one
PDB file.

Two details that matter for the file being usable:

* **chain ids are kept distinct.** A partner that happens to use the same chain
  id as the design molecule would otherwise merge into it and the file would
  describe a different molecule than the one scored. Clashing partners are
  remapped to free letters and the mapping is reported.
* **the partner's current pose is applied**, so the file shows where the redock
  put it, not where it started.

With several binding assemblies the packing differs per assembly, so one file is
written per assembly.
"""

from __future__ import annotations

import logging
import string
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_PDB_CHAINS = string.ascii_uppercase + string.ascii_lowercase + string.digits


@dataclass
class WrittenComplex:
    """Where a structure went, and anything the reader should know about it."""

    path: Path
    atoms: int
    #: Per molecule, the chain ids it was written under: ``{"<design>": {"A": "A"},
    #: "L1": {"A": "B"}}``. Keyed by molecule because two molecules can both use
    #: chain A, and a flat map would lose one of them.
    chain_map: dict[str, dict[str, str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    #: Key used for the design molecule in :attr:`chain_map`.
    DESIGN = "<design>"

    def renamed(self) -> dict[str, tuple[str, str]]:
        """Molecules whose chain id had to change, as {molecule: (old, new)}."""
        out: dict[str, tuple[str, str]] = {}
        for mol, mapping in self.chain_map.items():
            for old_id, new_id in mapping.items():
                if old_id != new_id:
                    out[mol] = (old_id, new_id)
        return out

    def describe(self) -> str:
        bits = [f"{self.path.name}: {self.atoms} atoms"]
        ren = self.renamed()
        if ren:
            bits.append(
                "chains remapped "
                + ", ".join(f"{m} {o}->{n}" for m, (o, n) in sorted(ren.items()))
            )
        return "; ".join(bits)


def _resseq_field(resseq, notes: list[str] | None = None) -> str:
    """The 4-column resSeq field, which must hold a NEGATIVE number as itself.

    An indel that prepends residues numbers them downward (A0, A-1, ...) so
    that nothing existing is renumbered. ``-1 % 10000`` is 9999 in Python, so
    the old masking wrote A-1 as residue 9999 -- a residue that reads back at
    the wrong end of the chain, which silently reorders the model. ``-999``
    through ``9999`` all fit the column as written.
    """
    num = int(resseq)
    if -999 <= num <= 9999:
        return f"{num:4d}"
    if notes is not None:
        notes.append(
            f"residue number {num} does not fit the 4-column PDB resSeq field; "
            f"written as {num % 10000} -- the numbering in this file is not the "
            f"numbering used in the design"
        )
    return f"{num % 10000:4d}"


def _serial_field(serial: int, notes: list[str] | None = None) -> str:
    """The 5-column atom serial, noting the wrap rather than hiding it."""
    if serial <= 99999:
        return f"{serial:5d}"
    if notes is not None and not any("atom serial" in n for n in notes):
        notes.append(
            f"more than 99999 atoms: serials wrap, so they are no longer unique "
            f"in this file (first wrap at atom {serial})"
        )
    return f"{serial % 100000:5d}"


def _atom_records(
    structure,
    chain_for: dict[str, str],
    transform=None,
    start_serial: int = 1,
    notes: list[str] | None = None,
) -> tuple[list[str], int]:
    """PDB ATOM/HETATM lines for a structure, renaming chains and moving atoms.

    Written by hand rather than through Biopython's PDBIO because the chain
    remapping and the rigid-body transform both have to be applied on the way
    out, and because the serial numbering has to continue across several
    molecules in one file.
    """
    lines: list[str] = []
    serial = start_serial
    model = structure.model
    for chain in model:
        out_chain = chain_for.get(chain.id, chain.id)
        wrote_any = False
        for res in chain:
            hetero = bool(res.id[0].strip())
            resname = res.get_resname().strip()[:3]
            _, resseq, icode = res.id
            for atom in res:
                coord = np.asarray(atom.coord, dtype=float)
                if transform is not None:
                    coord = transform(coord.reshape(1, 3))[0]
                name = atom.get_name()
                # PDB column rules: a 1-3 character name is offset by one unless
                # the element symbol is two characters.
                el = (atom.element or "").strip()
                if len(name) >= 4 or len(el) == 2:
                    name_field = f"{name:<4s}"
                else:
                    name_field = f" {name:<3s}"
                alt = getattr(atom, "altloc", " ") or " "
                occ = getattr(atom, "occupancy", 1.0)
                occ = 1.0 if occ is None else occ
                b = getattr(atom, "bfactor", 0.0)
                b = 0.0 if b is None else b
                lines.append(
                    f"{'HETATM' if hetero else 'ATOM  '}"
                    f"{_serial_field(serial, notes)} {name_field}{alt}{resname:>3s} "
                    f"{out_chain[:1]}{_resseq_field(resseq, notes)}{(icode or ' ')[:1]}   "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                    f"{occ:6.2f}{b:6.2f}          {el:>2s}"
                )
                serial += 1
                wrote_any = True
        if wrote_any:
            lines.append(f"TER   {_serial_field(serial, notes)}")
            serial += 1
    return lines, serial


def write_complex(
    path: str | Path,
    design_structure,
    partners: dict[str, object],
    poses: dict[str, object] | None = None,
    header: list[str] | None = None,
) -> WrittenComplex:
    """Write the design molecule and its partners into one PDB.

    ``design_structure`` should already carry the chosen side chains (write it
    with :func:`ipro_agent.mutate.apply_design` first, then load it back).
    ``poses`` maps a partner name to a :class:`ipro_agent.docking.Pose`.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    poses = poses or {}
    notes: list[str] = []

    used = {c.id for c in design_structure.model}
    design_map = {c: c for c in used}
    chain_map: dict[str, dict[str, str]] = {WrittenComplex.DESIGN: design_map}
    free = [c for c in _PDB_CHAINS if c not in used]

    def take_chain() -> str | None:
        """Next genuinely unused chain id.

        ``free`` is built once, but ``used`` grows as partners keep their own
        ids -- so popping blindly could hand out an id a previous partner had
        already kept, writing two molecules into one chain. Both then read
        back as a single molecule and only one of them is reported as renamed.
        """
        while free:
            c = free.pop(0)
            if c not in used:
                return c
        return None

    lines: list[str] = list(header or [])
    body, serial = _atom_records(design_structure, design_map, None, 1, notes)
    lines += body

    for name in sorted(partners):
        st = partners[name]
        mapping: dict[str, str] = {}
        for chain in st.model:
            if chain.id in used:
                new = take_chain()
                if new is None:
                    notes.append(
                        f"ran out of chain ids writing {name}; its chain "
                        f"{chain.id!r} collides with another molecule and the "
                        "file will merge them"
                    )
                    mapping[chain.id] = chain.id
                    continue
                mapping[chain.id] = new
                notes.append(f"{name}: chain {chain.id} written as {new}")
            else:
                mapping[chain.id] = chain.id
            used.add(mapping[chain.id])
        pose = poses.get(name)
        body, serial = _atom_records(
            st, mapping, (pose.apply if pose is not None else None), serial, notes
        )
        lines += body
        chain_map[name] = mapping

    lines.append("END")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_atoms = sum(1 for line in lines if line.startswith(("ATOM", "HETATM")))
    return WrittenComplex(path=out, atoms=n_atoms, chain_map=chain_map, notes=notes)


def write_atomset_pdb(atoms, path) -> int:
    """Write an :class:`~ipro_agent.energy.base.AtomSet` as a PDB.

    Needed because the structure that gets relaxed after the MILP is a
    *candidate*: the design template plus the chosen rotamers, which exists
    only as an AtomSet. Rebuilding it through PyMOL instead would put the
    rotamer-harvesting dependency in the scoring path and, worse, would let
    the written coordinates differ from the ones that were scored.

    Residue keys are ``(chain, number, insertion_code)`` (rotamer atoms carry
    the key of the position they belong to), so the records come out grouped by
    residue with the right chain and numbering. Returns the number of atoms
    written.
    """
    from pathlib import Path as _Path

    path = _Path(path)
    lines: list[str] = []
    serial = 0

    # Grouped by residue, in chain/number order. Written in AtomSet order the
    # file was malformed: a candidate's atoms are "template (minus the design
    # positions' side chains)" followed by "the chosen rotamers", so residue
    # 16's backbone appeared near the top and its side chain at the very end,
    # which reads as a DUPLICATE residue. Measured on a real full-mode run,
    # Biopython refused the file outright ("Blank altlocs in duplicate residue
    # SER (' ', 16, ' ')") while PyRosetta accepted it and silently returned a
    # structure with one atom more than it was given.
    def sort_key(i: int):
        key = atoms.reskeys[i]
        if isinstance(key, tuple) and len(key) >= 2 and isinstance(key[1], int):
            return (str(key[0]), int(key[1]),
                    str(key[2]) if len(key) > 2 and key[2] else "", i)
        return ("", 0, "", i)

    order = sorted(range(len(atoms)), key=sort_key)
    # Backbone first within each residue, as a PDB conventionally has it.
    _BB_ORDER = {"N": 0, "CA": 1, "C": 2, "O": 3, "OXT": 4}
    order.sort(key=lambda i: (sort_key(i)[:3],
                              _BB_ORDER.get(atoms.names[i].strip().upper(), 9)))

    # ONE residue name per residue, and it is the DESIGNED one. A candidate's
    # template keeps the native name on the backbone it contributes while the
    # chosen rotamer carries the new name, so the same residue came out as two
    # different residues -- measured on a real full-mode run, residue 16 was
    # written "CYS" for N/CA/C/O and "SER" for CB/OG, which Biopython refuses
    # ("Blank altlocs in duplicate residue SER") and which PyRosetta accepted
    # while silently returning a structure one atom larger than it was given.
    # The side chain names the residue: it is what the mutation changed.
    _BB = set(_BB_ORDER)
    resname_for: dict[tuple, str] = {}
    for i in order:
        k = sort_key(i)[:3]
        nm = atoms.names[i].strip().upper()
        rn = (atoms.resnames[i] or "").strip()
        if not rn:
            continue
        if k not in resname_for or (nm not in _BB and resname_for[k] != rn):
            if k in resname_for and nm in _BB:
                continue
            resname_for[k] = rn

    for i in order:
        key = atoms.reskeys[i]
        chain, number, icode = "A", 1, " "
        if isinstance(key, tuple) and len(key) >= 2 and isinstance(key[1], int):
            chain = str(key[0])[:1] or "A"
            number = int(key[1])
            icode = (str(key[2])[:1] if len(key) > 2 and key[2] else " ")
        serial += 1
        # Columns, checked rather than hoped for. A residue number past 9999
        # or a coordinate needing more than four digits before the point
        # overflows its field and shifts every column after it, which Bio.PDB
        # rejects outright ("Invalid or missing coordinate(s)") -- measured
        # with resSeq 10000 and with x = 1234.567, where the y field ate the
        # sign and the z field started mid-number. This file is what the user
        # is handed, so a structure that cannot be written is reported, never
        # written wrong.
        x, y, z = atoms.coords[i]
        if not (-999 <= number <= 9999):
            raise ValueError(
                f"residue number {number} does not fit a PDB resSeq field "
                f"(-999 to 9999). This structure cannot be written as PDB; "
                f"renumber it, or split it into chains."
            )
        for axis, value in (("x", x), ("y", y), ("z", z)):
            if not (-999.999 <= float(value) <= 9999.999):
                raise ValueError(
                    f"{axis} = {value:.3f} does not fit a PDB coordinate field "
                    f"(-999.999 to 9999.999). Centre the coordinates before "
                    f"writing, or the file will be silently misaligned."
                )
        name = atoms.names[i][:4]
        # Atom names are columns 13-16, and the convention is that a
        # one-letter element starts in column 14. A left-justified "CA" reads
        # as calcium to most parsers, which is the element-typing bug this
        # repo already fixed once on the reading side.
        if len(name) < 4 and len(atoms.elements[i].strip()) < 2:
            name = f" {name:<3}"
        else:
            name = f"{name:<4}"
        resname = resname_for.get(sort_key(i)[:3], atoms.resnames[i])
        lines.append(
            f"ATOM  {serial:5d} {name}{resname[:3]:>4}{chain:>2}"
            f"{number:4d}{icode}   {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
            f"          {atoms.elements[i].strip()[:2]:>2}"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return serial
