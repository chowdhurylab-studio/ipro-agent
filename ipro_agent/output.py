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


def _atom_records(
    structure, chain_for: dict[str, str], transform=None, start_serial: int = 1
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
                    f"{serial % 100000:5d} {name_field}{alt}{resname:>3s} "
                    f"{out_chain[:1]}{int(resseq) % 10000:4d}{(icode or ' ')[:1]}   "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                    f"{occ:6.2f}{b:6.2f}          {el:>2s}"
                )
                serial += 1
                wrote_any = True
        if wrote_any:
            lines.append(f"TER   {serial % 100000:5d}")
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

    lines: list[str] = list(header or [])
    body, serial = _atom_records(design_structure, design_map, None, 1)
    lines += body

    for name in sorted(partners):
        st = partners[name]
        mapping: dict[str, str] = {}
        for chain in st.model:
            if chain.id in used:
                if not free:
                    notes.append(
                        f"ran out of chain ids writing {name}; its chain "
                        f"{chain.id!r} collides with the design molecule and the "
                        "file will merge them"
                    )
                    mapping[chain.id] = chain.id
                    continue
                new = free.pop(0)
                mapping[chain.id] = new
                notes.append(f"{name}: chain {chain.id} written as {new}")
            else:
                mapping[chain.id] = chain.id
            used.add(mapping[chain.id])
        pose = poses.get(name)
        body, serial = _atom_records(
            st, mapping, (pose.apply if pose is not None else None), serial
        )
        lines += body
        chain_map[name] = mapping

    lines.append("END")
    out.write_text("\n".join(lines) + "\n")
    n_atoms = sum(1 for line in lines if line.startswith(("ATOM", "HETATM")))
    return WrittenComplex(path=out, atoms=n_atoms, chain_map=chain_map, notes=notes)
