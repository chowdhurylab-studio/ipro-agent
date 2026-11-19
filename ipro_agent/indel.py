"""Insertions and deletions at a chain terminus.

Why only the termini. Changing the length of a chain in the *middle* leaves two
cut ends that have to be rejoined, and getting that right is loop closure --
Protein-InDelMaker does it with PyRosetta's GeneralizedKIC over a six-residue
window, and IPRO+/- reports an eleven-residue CCD loop run 3000 times. There is
no honest shortcut: an internal edit without closure produces a chain with a
broken peptide bond, which still looks like a protein and scores like nonsense.
So internal indels are refused, with a pointer.

At a **terminus** there is no second cut end, so the edit is exact:

* a **deletion** removes residues and leaves every remaining atom untouched;
* an **insertion** appends residues built from ideal internal coordinates at
  chosen backbone dihedrals, which is a well-defined construction.

One invariant is enforced above all others: **an indel never renumbers an
existing residue.** Design positions are named by residue number, so a
renumbering would silently point them at different residues and the run would
optimize something nobody asked for. C-terminal insertions therefore continue
upward from the last residue, N-terminal insertions count *downward* from the
first (negative numbers are legal in a PDB and the position parser accepts
them), and a deletion simply leaves gaps in the numbering.

Inserted residues are built as **glycine**. That is not a limitation: the
inserted position becomes an ordinary design position, so the MILP picks its
identity and PyMOL builds the side chain, exactly as for any other position.
It also mirrors what IPRO does internally, mutating side chains to glycine
before it perturbs a backbone.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import ONE_TO_THREE, IndelKind, IndelSpec, SpecError

log = logging.getLogger(__name__)

# Ideal protein backbone geometry (Engh & Huber style values, in A and degrees).
B_N_CA = 1.458
B_CA_C = 1.525
B_C_N = 1.329      # the peptide bond
B_C_O = 1.231
A_N_CA_C = 111.2
A_CA_C_N = 116.2
A_C_N_CA = 121.7
A_CA_C_O = 120.8
OMEGA = 180.0      # trans peptide

#: Backbone dihedrals used for an inserted residue when none is given: an
#: extended conformation, which is the neutral choice for a terminal tail and
#: keeps the new residues out of the existing structure.
DEFAULT_PHI = -139.0
DEFAULT_PSI = 135.0


class IndelError(RuntimeError):
    """An insertion or deletion cannot be performed as asked."""


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def place_atom(
    a: np.ndarray, b: np.ndarray, c: np.ndarray,
    bond: float, angle_deg: float, dihedral_deg: float,
) -> np.ndarray:
    """Position a fourth atom from three existing ones and internal coordinates.

    ``bond`` is |CD|, ``angle_deg`` is the angle B-C-D, and ``dihedral_deg`` is
    the dihedral A-B-C-D. This is the standard natural-extension-reference-frame
    construction; it is checked by round-tripping (place an atom, then measure
    its bond, angle and dihedral back) in the tests, which is the only way to be
    sure of the sign conventions.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    bc = c - b
    n_bc = np.linalg.norm(bc)
    if n_bc < 1e-9:
        raise IndelError("cannot place an atom: two reference atoms coincide")
    bc = bc / n_bc

    ab = b - a
    normal = np.cross(ab, bc)
    n_norm = np.linalg.norm(normal)
    if n_norm < 1e-9:
        raise IndelError(
            "cannot place an atom: the three reference atoms are collinear, so "
            "the dihedral is undefined"
        )
    normal = normal / n_norm

    theta = math.radians(angle_deg)
    chi = math.radians(dihedral_deg)
    # Local frame: -bc is the direction back toward C's neighbour.
    d = (
        -bond * math.cos(theta) * bc
        + bond * math.sin(theta) * math.cos(chi) * np.cross(normal, bc)
        + bond * math.sin(theta) * math.sin(chi) * normal
    )
    return c + d


def measure_dihedral(p0, p1, p2, p3) -> float:
    """Dihedral in degrees, for checking :func:`place_atom`."""
    from .restraints import dihedral

    return dihedral(p0, p1, p2, p3)


def measure_angle(a, b, c) -> float:
    a, b, c = (np.asarray(x, float) for x in (a, b, c))
    v1, v2 = a - b, c - b
    cosang = float(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cosang))))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@dataclass
class IndelPlan:
    """What will be done, and what cannot be."""

    terminal: list[tuple[IndelSpec, str]] = field(default_factory=list)
    internal: list[tuple[IndelSpec, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def has_internal(self) -> bool:
        return bool(self.internal)


def _seqres_span(structure, chain, first, last, numbers) -> str | None:
    """Warn when the modelled chain is shorter than the deposited sequence.

    An insertion "after the last residue" is only a terminal edit if that
    residue really is the end of the chain. On a structure with disordered
    termini it is not, and appending there puts a free chain end in the middle
    of the molecule -- the same unclosed break that :func:`classify` refuses
    internal indels for, except nothing catches it. Returns a note naming the
    discrepancy, or None when SEQRES is absent or agrees.

    Read from the raw file rather than Biopython's header, because
    ``PDBParser`` does not retain SEQRES.
    """
    path = getattr(structure, "path", None)
    if not path:
        return None
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    declared = 0
    for line in text.splitlines():
        if line.startswith("SEQRES") and len(line) > 11 and line[11] == str(chain):
            declared += len(line[19:70].split())
    if not declared:
        return None                      # no SEQRES for this chain: say nothing

    observed = len(numbers)
    if observed >= declared:
        return None
    return (
        f"chain {chain} is modelled as {observed} residues ({first}-{last}) but "
        f"SEQRES declares {declared}: {declared - observed} residue(s) are "
        f"missing from the coordinates. A terminal indel is only exact at a real "
        f"chain end -- if the missing residues are at the terminus being edited, "
        f"this is an INTERNAL edit with an unclosed break, however much it looks "
        f"terminal. Check which residues are absent before trusting the result."
    )


def classify(structure, indels: list[IndelSpec]) -> IndelPlan:
    """Split the requested indels into terminal (doable) and internal (not).

    ``"N"`` or ``"C"`` is recorded for each terminal edit, since the two are
    built in opposite directions.
    """
    plan = IndelPlan()
    for spec in indels:
        try:
            residues = list(structure.residues(spec.chain))
        except Exception as exc:
            raise IndelError(f"chain {spec.chain!r} not in the structure: {exc}") from exc
        if not residues:
            raise IndelError(f"chain {spec.chain!r} has no residues")
        numbers = [int(r.id[1]) for r in residues]
        first, last = numbers[0], numbers[-1]

        # "Terminal" has to mean the end of the *molecule*, not the end of what
        # was modelled. This module's whole justification is that a terminus has
        # no second cut end, so the edit is exact -- but on a structure whose
        # termini are disordered and therefore absent from the coordinates, the
        # last modelled residue sits in the middle of the real chain, and
        # "appending" after it creates exactly the unclosed break that internal
        # indels are refused for. SEQRES, when the file has it, says how long
        # the chain really is.
        modelled = _seqres_span(structure, spec.chain, first, last, numbers)
        if modelled is not None:
            plan.notes.append(modelled)

        if spec.kind is IndelKind.INSERT:
            # An insertion is terminal when it goes before the first residue or
            # after the last one.
            if spec.after == last:
                plan.terminal.append((spec, "C"))
            elif spec.after == first - 1:
                plan.terminal.append((spec, "N"))
            else:
                plan.internal.append((
                    spec,
                    f"inserting after {spec.chain}{spec.after} is internal "
                    f"(chain spans {first}-{last}); only after {last} "
                    f"(C-terminus) or before {first} (N-terminus) needs no loop closure",
                ))
        else:
            want = sorted(int(x) for x in spec.residues)
            contiguous = want == list(range(want[0], want[0] + len(want)))
            if not contiguous:
                plan.internal.append((
                    spec, f"deleting {want} is not a contiguous run"
                ))
            elif want[-1] == last:
                plan.terminal.append((spec, "C"))
            elif want[0] == first:
                plan.terminal.append((spec, "N"))
            else:
                plan.internal.append((
                    spec,
                    f"deleting {want} leaves a gap in the middle of "
                    f"{spec.chain} ({first}-{last}); closing it needs loop closure",
                ))
    return plan


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------


@dataclass
class IndelResult:
    removed: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        bits = []
        if self.removed:
            bits.append(f"removed {len(self.removed)} residue(s): {', '.join(self.removed)}")
        if self.added:
            bits.append(f"added {len(self.added)} residue(s): {', '.join(self.added)}")
        return "; ".join(bits) or "no change"


def _new_residue(chain, number: int, resname: str, atoms: dict[str, np.ndarray]):
    """Build a Biopython residue with the given backbone atoms."""
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Residue import Residue

    res = Residue((" ", int(number), " "), resname, " ")
    for i, (name, coord) in enumerate(atoms.items()):
        element = name[0]
        res.add(Atom(
            name=name, coord=np.asarray(coord, dtype=float), bfactor=0.0,
            occupancy=1.0, altloc=" ", fullname=f" {name:<3s}",
            serial_number=i + 1, element=element,
        ))
    return res


def apply_terminal(
    structure,
    indels: list[tuple[IndelSpec, str]],
    phi: float = DEFAULT_PHI,
    psi: float = DEFAULT_PSI,
) -> IndelResult:
    """Apply terminal insertions and deletions **in place**.

    Deletions run before insertions so that "delete the last three, then append
    five" means what it says.
    """
    out = IndelResult()

    deletions = [(s, e) for s, e in indels if s.kind is IndelKind.DELETE]
    insertions = [(s, e) for s, e in indels if s.kind is IndelKind.INSERT]

    for spec, _end in deletions:
        chain = structure.model[spec.chain]
        for num in sorted({int(x) for x in spec.residues}, reverse=True):
            target = None
            for res in list(chain):
                if int(res.id[1]) == num and not res.id[0].strip():
                    target = res
                    break
            if target is None:
                raise IndelError(
                    f"cannot delete {spec.chain}{num}: no such residue"
                )
            chain.detach_child(target.id)
            out.removed.append(f"{spec.chain}{num}")

    for spec, end in insertions:
        seq = spec.sequence.upper() or ("G" * int(spec.length or 0))
        if not seq:
            raise IndelError("insertion has neither a sequence nor a length")
        out.notes.append(
            f"inserted residues are built as glycine and become design "
            f"positions; the requested identities ({spec.sequence or 'unspecified'}) "
            "are produced by the design step, not by the insertion"
            if spec.sequence else
            f"inserted {len(seq)} glycine(s) as design positions"
        )
        added = (
            _extend_c_terminus(structure, spec.chain, len(seq), phi, psi)
            if end == "C"
            else _extend_n_terminus(structure, spec.chain, len(seq), phi, psi)
        )
        out.added.extend(added)
    return out


def _backbone(res) -> dict[str, np.ndarray]:
    got = {}
    for atom in res:
        name = atom.get_name().strip().upper()
        if name in ("N", "CA", "C", "O"):
            got[name] = np.asarray(atom.coord, dtype=float)
    return got


def _extend_c_terminus(structure, chain_id: str, count: int, phi: float, psi: float) -> list[str]:
    """Append ``count`` glycines after the last residue, numbering upward."""
    chain = structure.model[chain_id]
    added: list[str] = []
    for _ in range(count):
        residues = [r for r in chain if not r.id[0].strip()]
        last = residues[-1]
        bb = _backbone(last)
        for need in ("N", "CA", "C"):
            if need not in bb:
                raise IndelError(
                    f"cannot extend {chain_id} past {int(last.id[1])}: it has no {need}"
                )
        number = int(last.id[1]) + 1
        if any(int(r.id[1]) == number for r in chain):
            raise IndelError(
                f"cannot append {chain_id}{number}: that number is already used. "
                "An indel must never renumber existing residues."
            )
        # N of the new residue is set by the previous residue's psi. Note what
        # that means: the residue we are extending past had NO psi (it was the
        # terminus), so appending *chooses* one for it. Its carbonyl O was
        # placed from the crystal geometry and is now inconsistent with the
        # nitrogen we are about to create, so it is re-placed below.
        n = place_atom(bb["N"], bb["CA"], bb["C"], B_C_N, A_CA_C_N, psi)
        # CA follows from the trans peptide bond.
        ca = place_atom(bb["CA"], bb["C"], n, B_N_CA, A_C_N_CA, OMEGA)
        # C follows from the new residue's phi.
        c = place_atom(bb["C"], n, ca, B_CA_C, A_N_CA_C, phi)
        # O(i) is anti to N(i+1) across the CA-C bond, so the N-CA-C-O
        # dihedral is psi + 180. A constant 180 lands it on top of N(i+1).
        o = place_atom(n, ca, c, B_C_O, A_CA_C_O, psi + 180.0)
        _replace_carbonyl_o(last, bb, psi)
        _drop_terminal_oxygen(last)
        chain.add(_new_residue(chain, number, "GLY", {"N": n, "CA": ca, "C": c, "O": o}))
        added.append(f"{chain_id}{number}")
    return added


def _extend_n_terminus(structure, chain_id: str, count: int, phi: float, psi: float) -> list[str]:
    """Prepend ``count`` glycines before the first residue, numbering downward.

    Downward so that no existing residue is renumbered. Negative numbers are
    legal in a PDB and the design-position parser accepts them.
    """
    chain = structure.model[chain_id]
    added: list[str] = []
    for _ in range(count):
        residues = [r for r in chain if not r.id[0].strip()]
        first = residues[0]
        bb = _backbone(first)
        for need in ("N", "CA", "C"):
            if need not in bb:
                raise IndelError(
                    f"cannot extend {chain_id} before {int(first.id[1])}: no {need}"
                )
        number = int(first.id[1]) - 1
        if any(int(r.id[1]) == number for r in chain):
            raise IndelError(
                f"cannot prepend {chain_id}{number}: that number is already used. "
                "An indel must never renumber existing residues."
            )
        # Build backwards: C of the new residue bonds to N of the first.
        c = place_atom(bb["C"], bb["CA"], bb["N"], B_C_N, A_C_N_CA, phi)
        ca = place_atom(bb["CA"], bb["N"], c, B_CA_C, A_CA_C_N, OMEGA)
        n = place_atom(bb["N"], c, ca, B_N_CA, A_N_CA_C, psi)
        # Anti to the nitrogen it bonds to -- see _extend_c_terminus. The
        # existing first residue needs no repair: prepending sets ITS phi,
        # which positions the atom we just built and moves nothing of its own.
        o = place_atom(n, ca, c, B_C_O, A_CA_C_O, psi + 180.0)
        new = _new_residue(chain, number, "GLY", {"N": n, "CA": ca, "C": c, "O": o})
        chain.add(new)
        added.append(f"{chain_id}{number}")
        # Re-sort inside the loop, not after it. Biopython appends, so without
        # this the next pass still sees the *original* first residue and tries
        # to create the same number again -- which only shows up when more than
        # one residue is prepended.
        _sort_chain(chain)
    return added


def _replace_carbonyl_o(res, bb: dict[str, np.ndarray], psi: float) -> None:
    """Re-place a residue's carbonyl O once its psi has been chosen.

    Only needed when appending: the old terminus had no psi, so the O it
    carries points wherever the crystal left it. Left alone it sits ~1.9 A
    from the nitrogen of the appended residue.
    """
    for atom in res:
        if atom.get_name().strip().upper() == "O":
            atom.set_coord(
                place_atom(bb["N"], bb["CA"], bb["C"], B_C_O, A_CA_C_O, psi + 180.0)
            )


def _drop_terminal_oxygen(res) -> None:
    """Remove OXT from a residue that has stopped being the C-terminus.

    OXT is the second carboxylate oxygen of a free C-terminus. After an
    append the residue is mid-chain, so the atom is not merely redundant --
    it occupies the position the new nitrogen needs.
    """
    for name in ("OXT", "HXT"):
        for atom in list(res):
            if atom.get_name().strip().upper() == name:
                res.detach_child(atom.get_id())


def _sort_chain(chain) -> None:
    """Order a chain's residues N-to-C, which every consumer here assumes."""
    chain.child_list.sort(
        key=lambda r: (r.id[0].strip() != "", int(r.id[1]), r.id[2])
    )
    # Biopython also keeps a dict view; rebuild it so lookups stay consistent.
    chain.child_dict = {r.id: r for r in chain.child_list}


def inserted_positions(result: IndelResult) -> list[str]:
    """Labels of the residues an insertion created, for use as design positions."""
    return list(result.added)
