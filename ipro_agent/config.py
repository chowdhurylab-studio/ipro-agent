"""Design-problem specification: what the user wants, on which molecules, where.

This module is the *generalization layer* over IPRO. Stock IPRO is configured by
editing per-case Python input decks that hard-code the molecules, the design
positions and the objective of one published case study. Here the same
information becomes a declarative :class:`DesignSpec` that an agent can write
from a plain-language goal, validate, and hand to the optimizer unchanged.

Vocabulary (following IPRO's own terms, see docs/DESIGN.md):

design molecule
    The molecule being redesigned. Mutations are only ever applied here.
target molecule
    A binding partner held fixed. The objective rewards *better* interaction
    with it.
anti-target molecule
    A binding partner held fixed whose interaction the objective *penalizes*.
    Stock IPRO has no such role; it is the minimal addition that lets one
    formulation express specificity switching (bind A, not B) rather than
    plain affinity maturation. See :mod:`ipro_agent.goals`.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# Amino-acid vocabulary
# ---------------------------------------------------------------------------

AA3 = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
ONE_TO_THREE = dict(zip("ARNDCQEGHILKMFPSTWYV", [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]))
# Written out rather than inverted from ONE_TO_THREE so the mapping is readable
# and greppable; checked for consistency in tests/test_ipro.py.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

#: Named shortcuts an agent (or user) can use instead of listing residues.
#: These are *conveniences for stating a design space*, not claims about
#: chemistry; the groupings follow common usage in the IPRO case studies.
AA_GROUPS: dict[str, list[str]] = {
    "all": list(AA3),
    # IPRO case studies routinely exclude PRO (backbone-distorting) and CYS
    # (spurious disulfides) from a free design space; this is the usual default.
    "standard": [a for a in AA3 if a not in ("PRO", "CYS")],
    "hydrophobic": ["ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "GLY"],
    "polar": ["SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS"],
    "charged": ["ASP", "GLU", "LYS", "ARG", "HIS"],
    "negative": ["ASP", "GLU"],
    "positive": ["LYS", "ARG", "HIS"],
    "aromatic": ["PHE", "TRP", "TYR", "HIS"],
    "small": ["GLY", "ALA", "SER", "CYS"],
    "large": ["PHE", "TRP", "TYR", "ARG", "LYS", "MET"],
    # Used by pore/channel work to line a lumen without adding charge.
    "nonpolar_nogly": ["ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP"],
}


def as_sequence(value: Any) -> list:
    """Wrap a scalar in a list instead of iterating it.

    A YAML scalar string is itself an iterable of characters, so every place
    that accepted "a list of X" also silently accepted one string and took it
    apart letter by letter. ``allowed: standard`` became the two residues whose
    codes happen to spell out of "s","t","a","n"... ; ``positions: "115"``
    became three positions A1, A1, A5; and ``residues: "12"`` deleted residues
    1 and 2 -- defeating section 1i's invariant that an indel never touches a
    residue nobody named. All of it ran clean. A scalar is what the user wrote,
    so it is wrapped, not iterated.
    """
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return [value]
    return list(value)


def position_label(chain: str, number: int, insertion_code: str = "") -> str:
    """The text form of a residue position, which must round-trip.

    ``parse(position_label(c, n, i)) == (c, n, i)`` for every spelling. A
    letter chain can simply concatenate (``A115``), but a numeric chain id has
    to keep its separator: ``1:115`` written as ``1115`` reads back as chain A
    residue 1115 and the mutation lands on a different residue -- silently,
    because residue 1115 may well exist. The PDB permits any printable
    character as a chain id, so the test is "is it a letter", not "is it a
    digit".

    One function rather than three copies: :class:`DesignPosition` had the
    separator, while :class:`ResidueRef` (structure.py) and :class:`AtomRef`
    did not, so the three disagreed for exactly the inputs that matter.
    """
    sep = "" if str(chain).isalpha() else ":"
    return f"{chain}{sep}{number}{insertion_code}".strip()


def expand_allowed(tokens: Iterable[str]) -> list[str]:
    """Expand a mixed list of group names, 1-letter and 3-letter codes.

    >>> expand_allowed(["hydrophobic", "W"])[:3]
    ['ALA', 'GLY', 'ILE']
    >>> expand_allowed(["ALA", "A"])
    ['ALA']
    >>> expand_allowed("standard") == expand_allowed(["standard"])
    True

    Raises :class:`SpecError` on an unrecognized token, rather than silently
    dropping it -- a dropped residue would quietly shrink the design space and
    change the answer.
    """
    out: list[str] = []
    for tok in as_sequence(tokens):
        t = str(tok).strip()
        if not t:
            continue
        key = t.lower()
        if key in AA_GROUPS:
            out.extend(AA_GROUPS[key])
        elif t.upper() in AA3:
            out.append(t.upper())
        elif len(t) == 1 and t.upper() in ONE_TO_THREE:
            out.append(ONE_TO_THREE[t.upper()])
        else:
            raise SpecError(
                f"unrecognized amino-acid token {t!r}; use a 1- or 3-letter code "
                f"or one of the groups: {', '.join(sorted(AA_GROUPS))}"
            )
    # Deduplicate, keep deterministic order so MILP variable order is stable
    # across runs (reproducibility: same spec -> same LP -> same solution).
    return sorted(set(out))


class SpecError(ValueError):
    """The design specification is unusable as written."""


_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}


def _as_bool(what: str, value: Any) -> bool:
    """Strict bool coercion: a quoted ``"false"`` must not read as True."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise SpecError(f"{what} must be true or false; got {value!r}")


# ---------------------------------------------------------------------------
# Molecules
# ---------------------------------------------------------------------------


class Role(str, Enum):
    DESIGN = "design"
    TARGET = "target"
    ANTI_TARGET = "anti_target"


class MolKind(str, Enum):
    PROTEIN = "protein"
    PEPTIDE = "peptide"
    LIGAND = "ligand"          # small molecule / substrate
    TRANSITION_STATE = "transition_state"  # TS analogue (OptZyme-style, kcat)


@dataclass
class Molecule:
    """One structural participant in the design problem."""

    name: str
    path: Path
    kind: MolKind = MolKind.PROTEIN
    role: Role = Role.TARGET
    #: Chain IDs to take from ``path``. Empty means "all chains in the file".
    chains: list[str] = field(default_factory=list)
    #: True if this molecule should be (re)docked each iteration rather than
    #: kept in its input pose. Ligands that must find a new pose after the
    #: pocket is mutated need this; a crystallographic partner usually does not.
    redock: bool = False

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser()
        self.kind = MolKind(self.kind)
        self.role = Role(self.role)
        self.chains = [str(c) for c in as_sequence(self.chains)]
        if not self.name:
            raise SpecError("molecule needs a name")

    @property
    def is_flexible_backbone(self) -> bool:
        """Whether backbone perturbation/relaxation applies to this molecule."""
        return self.kind in (MolKind.PROTEIN, MolKind.PEPTIDE)


# ---------------------------------------------------------------------------
# Design positions
# ---------------------------------------------------------------------------

_POS_RE = re.compile(
    r"^\s*(?:(?P<chain>[A-Za-z])[:/_]?|(?P<nchain>[A-Za-z0-9])[:/_])?"
    r"(?P<num>-?\d+)(?P<icode>[A-Za-z])?\s*$"
)
# A chain id is either a LETTER (with an optional separator), or any
# alphanumeric followed by a REQUIRED separator. Without that asymmetry the
# optional chain group eats the first digit of a bare residue number, so "115"
# parsed as chain "1", residue 15 -- silently designing the wrong residue. A
# numeric chain id must therefore be written with a separator, as "1:115". The
# hyphen is deliberately not a separator, so "A-5" is residue -5 of chain A.


@dataclass
class DesignPosition:
    """A position in the design molecule where mutation is permitted.

    ``allowed`` is the set of amino acids the user permits here. This is the
    user's hard constraint on the design space and becomes, directly, the index
    set of the MILP's rotamer variables at this position
    (:mod:`ipro_agent.milp`).
    """

    chain: str
    number: int
    allowed: list[str] = field(default_factory=lambda: list(AA_GROUPS["standard"]))
    insertion_code: str = ""
    #: Native identity, filled in from the structure during validation. Kept so
    #: the output can be reported as "D115W" and so "keep native" can be scored.
    native: str | None = None

    def __post_init__(self) -> None:
        self.allowed = expand_allowed(self.allowed)
        if not self.allowed:
            raise SpecError(f"position {self.label} has an empty allowed-residue set")
        self.chain = str(self.chain)
        self.number = int(self.number)

    @property
    def label(self) -> str:
        """Text form, which must round-trip: ``parse(p.label).key == p.key``.

        See :func:`position_label`.
        """
        return position_label(self.chain, self.number, self.insertion_code)

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.chain, self.number, self.insertion_code)

    @classmethod
    def parse(cls, text: str, allowed: Sequence[str] | None = None) -> "DesignPosition":
        """Parse ``"A115"``, ``"A:115"``, ``"115"`` or ``"A115B"`` (insertion code).

        >>> DesignPosition.parse("A115", ["hydrophobic"]).label
        'A115'
        """
        m = _POS_RE.match(str(text))
        if not m:
            raise SpecError(
                f"cannot parse position {text!r}; expected forms like 'A115', "
                "'A:115', '115' or 'A115B'"
            )
        # `if allowed` conflated "not given" with "given as empty": a spec
        # written `allowed: []` -- or `allowed:` with nothing after it, which
        # YAML reads as None inside a list of one -- silently became the full
        # 18-residue `standard` set, so a user who meant "nothing may change
        # here" got the widest alphabet in the tool and a design space 18x
        # larger per position than they asked for. Absent is the default;
        # empty is an error, which __post_init__ already raises.
        if allowed is None:
            chosen = list(AA_GROUPS["standard"])
        else:
            chosen = list(allowed)
        return cls(
            chain=m.group("chain") or m.group("nchain") or "A",
            number=int(m.group("num")),
            insertion_code=(m.group("icode") or "").upper(),
            allowed=chosen,
        )


class IndelKind(str, Enum):
    INSERT = "insert"
    DELETE = "delete"


@dataclass
class IndelSpec:
    """An insertion or deletion to attempt (Protein-InDelMaker capability).

    ``after`` is the residue the inserted stretch follows. For a deletion,
    ``residues`` lists the positions to remove. Insertions and deletions change
    the residue numbering, so they are applied *before* the design positions are
    resolved against the structure.
    """

    kind: IndelKind
    chain: str
    #: For INSERT: insert immediately after this residue number.
    after: int | None = None
    #: For INSERT: the sequence to insert (1-letter codes), or its length if the
    #: identities are themselves to be designed.
    sequence: str = ""
    length: int | None = None
    #: For DELETE: residue numbers to remove.
    residues: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.kind = IndelKind(self.kind)
        # Normalized here, not at the point of use: ``residues: "12"`` is one
        # residue the user named, and iterating the string deleted residues 1
        # and 2 instead while reporting success.
        try:
            self.residues = [int(x) for x in as_sequence(self.residues)]
        except (TypeError, ValueError) as exc:
            raise SpecError(
                f"indel 'residues' must be residue numbers; got {self.residues!r}"
            ) from exc
        if self.kind is IndelKind.INSERT:
            if self.after is None:
                raise SpecError("insertion needs 'after' (residue to insert behind)")
            if not self.sequence and not self.length:
                raise SpecError("insertion needs either 'sequence' or 'length'")
            if self.sequence:
                bad = set(self.sequence.upper()) - set(ONE_TO_THREE)
                if bad:
                    raise SpecError(f"insertion sequence has non-residue letters: {sorted(bad)}")
                self.length = len(self.sequence)
        else:
            if not self.residues:
                raise SpecError("deletion needs 'residues'")


# ---------------------------------------------------------------------------
# Binding assemblies
# ---------------------------------------------------------------------------


@dataclass
class Assembly:
    """One binding topology the design must work in -- IPRO's **BA**.

    "The totality of the participating molecules interacting with a particular
    binding topology is referred to as a binding assembly (BA)." A design
    problem may have several, and the rule that makes them useful is:

        the optimal combination of rotamers is selected **separately for each
        BA**, whereas the **residue identity is the same in all of them**

    so one sequence is evaluated in several structural contexts at once, each
    free to repack. That is what lets a single run demand "binds C12 *and*
    rejects C14" rather than ranking the two afterwards.

    ``molecules`` names the participants. The design molecule is always a
    participant and need not be listed. ``weight`` scales this assembly's
    contribution to the combined objective; ``must_pass`` decides whether the
    acceptance test requires it (IPRO: "all BAs must pass the selection
    criteria for the results of the iteration to be retained").
    """

    name: str
    molecules: list[str] = field(default_factory=list)
    weight: float = 1.0
    must_pass: bool = True

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise SpecError("every assembly needs a name")
        self.name = str(self.name).strip()
        self.molecules = [str(m) for m in as_sequence(self.molecules)]
        dup = sorted({n for n in self.molecules if self.molecules.count(n) > 1})
        if dup:
            # A molecule listed twice was kept twice, so its own internal
            # energy entered the complex energy once per listing (+1588.34
            # kcal/mol measured on the smoke case) while the interaction map,
            # keyed by name, showed one copy. The two then disagreed.
            raise SpecError(
                f"assembly {self.name!r} lists molecule(s) {dup} more than once; "
                "each participant may appear once (its energy would otherwise be "
                "counted once per listing)"
            )
        if self.weight <= 0:
            raise SpecError(
                f"assembly {self.name!r} has weight {self.weight}; weights scale a "
                "contribution to the objective and must be positive (use a goal "
                "with an anti-target to penalize an interaction)"
            )


# ---------------------------------------------------------------------------
# Restraints
# ---------------------------------------------------------------------------

_ATOM_RE = re.compile(r"^\s*(?P<pos>[^:\s]+)\s*[:.]\s*(?P<atom>[A-Za-z0-9'*]+)\s*$")


@dataclass(frozen=True)
class AtomRef:
    """One atom, named by its residue and atom name: ``"A115:CB"``.

    The residue part accepts the same spellings as a design position, so
    ``A115:CB``, ``A:115:CB`` and ``X1:C1`` (a ligand atom) all work.
    """

    chain: str
    number: int
    atom: str
    insertion_code: str = ""

    @property
    def position(self) -> str:
        """The residue part, in the same round-tripping form a design position
        uses -- ``restraints._resolve`` compares the two as text, so when this
        dropped the separator a restraint on a numeric-chain design position
        was reclassified as "held fixed, satisfied by construction" and was
        never enforced."""
        return position_label(self.chain, self.number, self.insertion_code)

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.chain, self.number, self.insertion_code)

    def __str__(self) -> str:
        return f"{self.position}:{self.atom}"

    @classmethod
    def parse(cls, text: str) -> "AtomRef":
        m = _ATOM_RE.match(str(text))
        if not m:
            raise SpecError(
                f"cannot parse atom reference {text!r}; expected "
                "'<chain><residue>:<atom>', e.g. 'A115:CB' or 'X1:C1'"
            )
        pos = DesignPosition.parse(m.group("pos"), allowed=["ALA"])
        return cls(
            chain=pos.chain, number=pos.number,
            insertion_code=pos.insertion_code, atom=m.group("atom").upper(),
        )


class RestraintKind(str, Enum):
    #: Hold a distance between two atoms. The capability the published
    #: thioesterase work used to preserve catalytic geometry while redesigning
    #: specificity.
    DISTANCE = "distance"
    #: Keep one atom near where it started.
    TETHER = "tether"
    #: Hold a dihedral defined by four atoms.
    DIHEDRAL = "dihedral"
    #: Keep one atom exactly where it started; a tether with a tight tolerance.
    FIX = "fix"


#: Tolerance used for a ``fix`` when the user gives none (Angstroms).
FIX_TOLERANCE = 0.05


@dataclass
class Restraint:
    """A geometric condition every accepted design must satisfy.

    IPRO enforces four kinds of atom-position restraint "rigorously ... throughout
    all functions": atoms fixed in place, atoms held near their input positions, a
    distance between two atoms, and a dihedral over four atoms. All four are
    expressed here.

    Every restraint reduces to an **interval** that a measured geometric quantity
    must lie in. Give either ``target`` with ``tolerance``, or ``minimum`` and/or
    ``maximum``:

    * ``target: 3.2, tolerance: 0.2`` -> the quantity must lie in [3.0, 3.4];
    * ``minimum: 4.0`` -> at least 4.0 (keep two atoms apart);
    * ``maximum: 3.5`` -> at most 3.5 (keep a contact).

    For ``tether`` and ``fix``, ``target`` is omitted: the reference is the atom's
    position in the input structure, and the interval bounds its displacement.

    Distances are Angstroms, dihedrals degrees.
    """

    kind: RestraintKind | str
    atoms: list[str] = field(default_factory=list)
    target: float | None = None
    tolerance: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    label: str = ""
    #: Refuse to run when the input structure already violates this restraint,
    #: rather than reporting it and continuing.
    strict: bool = False

    #: Resolved atom references, filled in by __post_init__.
    refs: list[AtomRef] = field(default_factory=list, repr=False)

    _ARITY = {
        RestraintKind.DISTANCE: 2,
        RestraintKind.TETHER: 1,
        RestraintKind.FIX: 1,
        RestraintKind.DIHEDRAL: 4,
    }

    def __post_init__(self) -> None:
        self.kind = RestraintKind(str(self.kind).lower())
        self.atoms = [str(a) for a in as_sequence(self.atoms)]
        self.refs = [AtomRef.parse(a) for a in self.atoms]
        want = self._ARITY[self.kind]
        if len(self.refs) != want:
            raise SpecError(
                f"a {self.kind.value} restraint needs exactly {want} atom(s); "
                f"got {len(self.refs)} ({self.atoms})"
            )
        if self.kind is RestraintKind.FIX and self.tolerance is None and self.maximum is None:
            # Only when the user said nothing at all. `interval()` takes
            # min(tolerance, maximum), so installing the default alongside an
            # explicit `maximum: 2.0` enforced 0.05 A instead -- a restraint 40x
            # tighter than the one written, which narrows a design rather than
            # refusing it. The error message three lines below offers 'maximum'
            # as the way to state the allowance, so it has to be honoured.
            self.tolerance = FIX_TOLERANCE
        if self.kind in (RestraintKind.TETHER, RestraintKind.FIX):
            if self.target is not None:
                raise SpecError(
                    f"a {self.kind.value} restraint measures displacement from the "
                    "input structure, so it takes no 'target'; use 'tolerance' "
                    "(or 'maximum') to say how far the atom may move"
                )
            if self.tolerance is None and self.maximum is None:
                raise SpecError(
                    f"a {self.kind.value} restraint needs 'tolerance' or 'maximum' "
                    "(how far the atom may move, in Angstroms)"
                )
        if self.target is not None and self.tolerance is None:
            raise SpecError(
                f"restraint {self.describe()} gives 'target' but no 'tolerance'; "
                "an exact equality can never be met by floating-point geometry"
            )
        if self.target is None and self.tolerance is not None and self.kind not in (
            RestraintKind.TETHER, RestraintKind.FIX
        ):
            raise SpecError(
                f"restraint {self.describe()} gives 'tolerance' but no 'target'"
            )
        if self.target is None and self.tolerance is None and (
            self.minimum is None and self.maximum is None
        ):
            raise SpecError(
                f"restraint {self.describe()} states no condition; give "
                "'target'+'tolerance', or 'minimum'/'maximum'"
            )
        lo, hi = self.interval()
        if lo > hi:
            raise SpecError(f"restraint {self.describe()} has an empty interval [{lo}, {hi}]")

    def interval(self) -> tuple[float, float]:
        """The closed interval the measured quantity must fall in."""
        lo, hi = -math.inf, math.inf
        if self.is_displacement:
            # A displacement is a distance from the input position, so it can
            # never be negative, whether the allowance was given as 'tolerance'
            # or as 'maximum'.
            lo = 0.0
        if self.target is not None and self.tolerance is not None:
            lo, hi = self.target - self.tolerance, self.target + self.tolerance
        elif self.tolerance is not None:   # tether / fix: displacement from input
            lo, hi = 0.0, self.tolerance
        if self.minimum is not None:
            lo = max(lo, float(self.minimum))
        if self.maximum is not None:
            hi = min(hi, float(self.maximum))
        return lo, hi

    @property
    def is_displacement(self) -> bool:
        return self.kind in (RestraintKind.TETHER, RestraintKind.FIX)

    @property
    def unit(self) -> str:
        return "deg" if self.kind is RestraintKind.DIHEDRAL else "A"

    def describe(self) -> str:
        lo, hi = self.interval()
        atoms = ", ".join(str(r) for r in self.refs) or ",".join(map(str, self.atoms))
        what = "displacement" if self.is_displacement else self.kind.value
        bound = (
            f"[{lo:g}, {hi:g}]" if math.isfinite(lo) and math.isfinite(hi)
            else (f">= {lo:g}" if math.isfinite(lo) else f"<= {hi:g}")
        )
        return f"{self.label or self.kind.value}: {what}({atoms}) in {bound} {self.unit}"


# ---------------------------------------------------------------------------
# Optimizer settings
# ---------------------------------------------------------------------------


#: Gas constant in kcal/(mol K). IPRO's acceptance temperature is quoted in
#: kelvin, so converting it to a kT in kcal/mol needs this.
R_KCAL = 1.987e-3


@dataclass
class LoopSettings:
    """Controls for the IPRO iteration loop.

    Defaults are stock IPRO's published values wherever one exists; each is
    cited in docs/DESIGN.md. They are exposed because the right iteration count
    is case-dependent and because an agent running a smoke test must be able to
    ask for a very short run.
    """

    #: IPRO's own coded default is 3000 iterations (the Suite paper quotes a
    #: 10,000 cap). The default here is deliberately small so that a run is a
    #: considered choice rather than an accidental multi-day job; raise it for
    #: real work.
    iterations: int = 100
    #: Stop early once this many designs have been accepted (IPRO's other
    #: termination condition). None means "run to the iteration cap".
    stop_after_accepted: int | None = None
    #: Advisory cap on rotamers per MILP. IPRO's released source caps at 1300
    #: (the Suite paper says 1400) and **shrinks its pruning window until the
    #: count fits** -- ``ROTAMERS.py``: ``while total > MAX``, window *= 0.95,
    #: then N -= 1 once the window falls below 15% of its start, raising only
    #: after 300 s of failing to shrink. Nothing here tightens anything: the
    #: run reports the overrun and carries on. The comment used to claim the
    #: tightening, which is worse than no comment at all.
    max_rotamers: int = 1300

    #: **NOT IMPLEMENTED, and kept only so an old spec still loads.** IPRO
    #: repacks a 4.5 A shell around the perturbation window, and the 2006
    #: paper's step iii additionally lets the five flanking residues either
    #: side change rotamer (not identity). Neither exists here: the MILP's
    #: variables are exactly the user's design positions, so the surrounding
    #: native side chains keep their input rotamers and cannot relieve the
    #: strain a perturbation introduces. Setting this changes nothing, and the
    #: run says so. See docs/DESIGN.md section 4.
    neighbour_cutoff: float = 4.5
    #: Non-bonded energy cutoff (A). IPRO: 12 A vdW.
    energy_cutoff: float = 12.0

    #: How many contiguous positions are re-designed per iteration, **and only
    #: when the backbone does not move**. With ``perturb_backbone: true`` the
    #: window is the design positions inside the perturbed region, as IPRO
    #: defines it (Suite step 1; 2006 Fig. 5), and this setting is not read at
    #: all -- the run says so when it is set to something other than the
    #: default. Neither the Suite paper nor the released source states a 1-3
    #: rule; it is this repo's fallback for the fixed-backbone case, where
    #: there is no region to read a window from.
    design_per_iteration: tuple[int, int] = (1, 3)
    #: Perturb the backbone each iteration. Off by default because it costs a
    #: PyMOL rotamer re-harvest and a full energy-table rebuild per iteration,
    #: and because a fixed-backbone search is a legitimate (cheaper) protocol.
    #: Turning it on is what makes the run IPRO rather than a side-chain packer:
    #: a 5-degree dihedral change moves atoms 0.2-2.6 A further down the chain,
    #: which can open a pocket no rotamer choice could.
    perturb_backbone: bool = False
    #: Length of the backbone segment perturbed each iteration. IPRO's source
    #: picks 1-5 consecutive residues; this is a separate choice from which
    #: positions are re-designed.
    perturbation_segment: tuple[int, int] = (1, 5)
    #: Backbone perturbation: phi/psi offsets ~ N(0, sigma), rejection-truncated
    #: at +/-cap. IPRO's source uses random.gauss(0, 1.5) truncated to +/-5 deg;
    #: the 2006 paper reports sigma = 1.65 deg.
    perturbation_sigma_deg: float = 1.5
    perturbation_cap_deg: float = 5.0
    #: **NOT IMPLEMENTED** -- see :attr:`neighbour_cutoff`. Kept so an old
    #: spec still loads; setting it changes nothing.
    perturbation_flank: int = 5

    #: Rigid-body redocking of target molecules every Nth iteration.
    redock_every: int = 3
    #: Monte-Carlo controls, used by the 'builtin' dock backend.
    redock_moves: int = 500
    redock_sigma_angstrom: float = 0.2
    redock_sigma_deg: float = 2.0
    #: Docking anneals from loop.temperature_k down to this. IPRO: 3640 -> 2190 K,
    #: retaining 25% and 10% of positions within 10 kcal/mol of the best so far.
    redock_temperature_end_k: float = 2190.0
    #: Vina controls, used by the 'vina' dock backend.
    vina_exhaustiveness: int = 8
    vina_num_modes: int = 9
    #: Box side (A) centred on the design positions when Vina places a partner.
    vina_box_size: float = 22.5

    #: Metropolis acceptance temperature in kelvin. IPRO's default, 3640 K, is
    #: calibrated so that a 10 kcal/mol worsening is accepted 25% of the time:
    #: exp(-10 / (R * 3640)) = 0.251 with R = 1.987e-3. Verified in
    #: tests/test_ipro.py. IPRO's source draws an acceptance *window*
    #: -R*T*ln(U) and compares it to the energy change, which is algebraically
    #: the same test.
    temperature_k: float = 3640.0
    #: IPRO runs the first fraction of iterations as pure downhill search, with
    #: no thermal acceptance (its source disables simulated annealing for the
    #: first 10% of iterations).
    greedy_fraction: float = 0.10
    #: Regardless of the Metropolis test, reject outright if the complex energy
    #: worsens by more than max(hard_reject_kcal, hard_reject_fraction * |E|).
    #: IPRO uses 40 kcal/mol or 5%.
    hard_reject_kcal: float = 40.0
    hard_reject_fraction: float = 0.05

    #: Forbid revisiting an already-evaluated sequence, via an integer ("no
    #: good") cut added to the MILP. This is IPRO's own behaviour.
    integer_cuts: bool = True

    #: Keep this many best designs as an ensemble rather than only the single best.
    ensemble_size: int = 10
    random_seed: int = 0
    #: Optional wall-clock budget; the loop checkpoints and stops cleanly.
    max_minutes: float | None = None
    #: Write the accepted structure of every iteration (costs disk).
    keep_structures: bool = False

    @property
    def kt(self) -> float:
        """Acceptance kT in kcal/mol (the energy scale of the backend)."""
        return R_KCAL * self.temperature_k

    #: The only settings for which "no value" is a meaning rather than a
    #: mistake: both document None as "no limit".
    _NULLABLE = frozenset({"stop_after_accepted", "max_minutes"})

    def __post_init__(self) -> None:
        from dataclasses import fields as _dc_fields

        for f in _dc_fields(self):
            val = getattr(self, f.name)
            if val is None and f.name not in self._NULLABLE:
                # `integer_cuts:` written with no value parses as null, which is
                # falsy, so a documented True default silently became False:
                # measured, that turned off IPRO's integer cuts and a 6-iteration
                # run "accepted" six copies of one design. A blank key is a
                # mistake, not a value.
                raise SpecError(
                    f"loop.{f.name} has no value. A YAML key written with nothing "
                    f"after the colon parses as null, which would silently replace "
                    f"the default ({f.default!r}). Give it a value or remove the key."
                )
            if isinstance(f.default, bool) and not isinstance(val, bool):
                # A quoted "false" is a non-empty string, i.e. truthy.
                setattr(self, f.name, _as_bool(f"loop.{f.name}", val))

        # YAML gives lists where the dataclass wants tuples.
        for attr in ("design_per_iteration", "perturbation_segment"):
            val = getattr(self, attr)
            if isinstance(val, list):
                setattr(self, attr, tuple(val))
            lo, hi = getattr(self, attr)
            if lo < 1 or hi < lo:
                raise SpecError(f"{attr} must be (lo<=hi) with lo>=1; got {(lo, hi)}")
        # Zero is a mistake, not "unset". `--iterations 0` and
        # `--max-minutes 0` were both silently dropped (the CLI tested
        # truthiness), so a user asking for a zero-length run got the spec's
        # full 6 iterations instead, and `max_minutes: 0` in a spec meant NO
        # limit -- the opposite of what it says.
        try:
            self.iterations = int(self.iterations)
        except (TypeError, ValueError) as exc:
            raise SpecError(
                f"loop.iterations must be a whole number; got "
                f"{self.iterations!r}"
            ) from exc
        if self.iterations < 1:
            raise SpecError(
                f"loop.iterations is {self.iterations}; a run needs at least 1. "
                f"(Zero is not 'unlimited' and not 'default' -- say what you "
                f"mean, or leave the key out.)"
            )
        if self.max_minutes is not None:
            try:
                self.max_minutes = float(self.max_minutes)
            except (TypeError, ValueError) as exc:
                raise SpecError(
                    f"loop.max_minutes must be a number of minutes or null; "
                    f"got {self.max_minutes!r}"
                ) from exc
            if self.max_minutes <= 0:
                raise SpecError(
                    f"loop.max_minutes is {self.max_minutes}; a budget must be "
                    f"positive. Use null (or leave the key out) for no limit."
                )
        if self.random_seed is not None:
            try:
                self.random_seed = int(self.random_seed)
            except (TypeError, ValueError) as exc:
                raise SpecError(
                    f"loop.random_seed must be a whole number or null; got "
                    f"{self.random_seed!r}"
                ) from exc
        if self.temperature_k <= 0:
            raise SpecError("temperature_k must be positive")
        if not 0.0 <= self.greedy_fraction <= 1.0:
            raise SpecError("greedy_fraction must be between 0 and 1")


@dataclass
class Backends:
    """Which external programs to use for each job.

    ``auto`` resolves at run time to the best available option and falls back to
    the built-in implementations, so the loop is runnable (and the example case
    testable) on a machine without PyRosetta, PyMOL or Vina installed. Run
    ``scripts/doctor.py`` to see what resolved.
    """

    energy: str = "auto"       # auto | pyrosetta | builtin
    #: auto | pymol. There is no built-in mutagenesis: rotamers come from
    #: PyMOL's library, so PyMOL is effectively required.
    mutate: str = "auto"
    #: Which program places the partner molecules each redock.
    #:
    #: ``vina``    AutoDock Vina: a **global** search with its own scoring
    #:             function. Finds a pose when there is no good starting one,
    #:             and is the heavier of the two by a wide margin. Its affinity
    #:             is on its own scale and is never added to the objective --
    #:             the pose is taken and re-scored with the energy backend.
    #: ``builtin`` IPRO's own step: a **local** rigid-body Monte Carlo over the
    #:             six rigid-body degrees of freedom, scored with the same
    #:             energy as rotamer selection (500 moves, sigma 0.2 A / 2.0
    #:             deg, annealed 3640 K -> 2190 K). Cheap, reproducible from a
    #:             seed, and refines a pose rather than finding one.
    #: ``auto``    Vina when it is installed, otherwise ``builtin``.
    #: ``none``    Keep the input poses.
    #:
    #: These are different operations, not two qualities of the same one. See
    #: docs/DESIGN.md for the trade-off table.
    dock: str = "auto"
    milp_solver: str = "auto"  # auto | cbc | glpk | cplex | gurobi | highs | greedy

    def __post_init__(self) -> None:
        if self.dock not in ("auto", "vina", "builtin", "none"):
            raise SpecError(
                f"backends.dock must be 'auto', 'vina', 'builtin' or 'none' "
                f"(got {self.dock!r}). 'vina' is a global search; 'builtin' is "
                "IPRO's local rigid-body Monte Carlo. They do different jobs -- "
                "see docs/DESIGN.md."
            )
        if self.mutate not in ("auto", "pymol"):
            raise SpecError(
                f"backends.mutate must be 'auto' or 'pymol' (got {self.mutate!r}); "
                "there is no built-in mutagenesis -- rotamers come from PyMOL"
            )


# ---------------------------------------------------------------------------
# The specification
# ---------------------------------------------------------------------------


@dataclass
class DesignSpec:
    """A complete, self-contained statement of a redesign problem."""

    #: The user's goal in their own words. Carried through to every report so a
    #: design can always be traced back to what was actually asked for.
    goal_statement: str
    #: Goal type; see :mod:`ipro_agent.goals` for the objective each implies.
    goal: str
    molecules: list[Molecule] = field(default_factory=list)
    positions: list[DesignPosition] = field(default_factory=list)
    indels: list[IndelSpec] = field(default_factory=list)
    #: Geometric conditions every accepted design must satisfy.
    restraints: list[Restraint] = field(default_factory=list)
    #: Binding topologies to design in simultaneously. Empty means one implicit
    #: assembly containing every molecule, which is the single-complex case.
    assemblies: list[Assembly] = field(default_factory=list)
    loop: LoopSettings = field(default_factory=LoopSettings)
    backends: Backends = field(default_factory=Backends)
    #: Free-form, goal-specific knobs (e.g. a target pore radius). Validated by
    #: the goal implementation, not here.
    goal_options: dict[str, Any] = field(default_factory=dict)
    name: str = "design"
    notes: str = ""

    def __post_init__(self) -> None:
        dup_mol = sorted({
            m.name for m in self.molecules
            if sum(1 for o in self.molecules if o.name == m.name) > 1
        })
        if dup_mol:
            # Every lookup downstream is by name -- `spec.molecule(name)`, the
            # interaction map, the assembly membership list -- but the goals
            # summed over molecule *objects*. A duplicated name therefore
            # counted one measured interaction energy twice (29.564 reported as
            # 59.128), inflating the selectivity gap 2.1x with no error.
            raise SpecError(
                f"molecule name(s) {dup_mol} are used more than once. Molecules "
                "are referred to by name everywhere (assemblies, goals, "
                "reports), so a repeated name is ambiguous."
            )
        seen: dict[tuple, DesignPosition] = {}
        for p in self.positions:
            if p.key in seen:
                # The MILP built variables from the last entry, `position()`
                # returned the first, and `design_space_size()` multiplied
                # both: a position pinned to TRP twice reported 4 sequences for
                # one position with 2 choices, and a contradictory pair meant
                # the run could not produce the residue the user asked for.
                raise SpecError(
                    f"design position {p.label} is listed more than once "
                    f"(allowed sets {seen[p.key].allowed} and {p.allowed}). "
                    "List each position once, with the union of what it may be."
                )
            seen[p.key] = p

    # -- derived views ----------------------------------------------------

    @property
    def design_molecule(self) -> Molecule:
        ms = [m for m in self.molecules if m.role is Role.DESIGN]
        if len(ms) != 1:
            raise SpecError(
                f"exactly one molecule must have role 'design'; found {len(ms)}"
            )
        return ms[0]

    def molecule(self, name: str) -> Molecule:
        for m in self.molecules:
            if m.name == name:
                return m
        raise SpecError(
            f"no molecule named {name!r}; have "
            f"{', '.join(repr(m.name) for m in self.molecules)}"
        )

    def resolved_assemblies(self) -> list[Assembly]:
        """The assemblies to design in, with the single-complex default filled in.

        With no ``assemblies:`` block this returns one assembly containing every
        molecule, so the single-complex case needs no extra configuration and
        behaves exactly as it did before assemblies existed.
        """
        if not self.assemblies:
            return [Assembly(
                name="complex",
                molecules=[m.name for m in self.molecules if m.role is not Role.DESIGN],
            )]
        out: list[Assembly] = []
        design = self.design_molecule.name
        seen: set[str] = set()
        for a in self.assemblies:
            if a.name in seen:
                raise SpecError(f"two assemblies are both named {a.name!r}")
            seen.add(a.name)
            names = [n for n in a.molecules if n != design]
            for n in names:
                self.molecule(n)          # raises with a helpful message
            out.append(Assembly(
                name=a.name, molecules=names, weight=a.weight, must_pass=a.must_pass
            ))
        # A molecule in no assembly would be silently ignored, which would change
        # the problem without saying so.
        used = {n for a in out for n in a.molecules}
        orphan = [
            m.name for m in self.molecules
            if m.role is not Role.DESIGN and m.name not in used
        ]
        if orphan:
            raise SpecError(
                f"molecule(s) {orphan} belong to no assembly, so they would not be "
                "scored at all. Add them to an assembly, or remove them."
            )
        if not any(a.must_pass for a in out):
            raise SpecError(
                "no assembly has must_pass; at least one must gate acceptance, or "
                "nothing constrains whether an iteration is kept"
            )
        return out

    def assembly_molecules(self, a: Assembly) -> list[Molecule]:
        return [self.molecule(n) for n in a.molecules]

    @property
    def targets(self) -> list[Molecule]:
        return [m for m in self.molecules if m.role is Role.TARGET]

    @property
    def anti_targets(self) -> list[Molecule]:
        return [m for m in self.molecules if m.role is Role.ANTI_TARGET]

    def position(self, chain: str, number: int, icode: str = "") -> DesignPosition:
        for p in self.positions:
            if p.key == (chain, number, icode):
                return p
        raise KeyError(f"{chain}{number}{icode} is not a design position")

    # -- size / feasibility ----------------------------------------------

    def design_space_size(self) -> int:
        """Number of distinct sequences the spec permits (product of choices).

        Reported by ``validate`` because it is the single most useful number for
        telling a user whether their problem is a smoke test or a month of CPU.
        """
        n = 1
        for p in self.positions:
            n *= len(p.allowed)
        return n

    # -- (de)serialization ------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DesignSpec":
        if not isinstance(d, dict):
            raise SpecError(
                f"a spec must be a mapping of keys to values, not a "
                f"{type(d).__name__}"
            )
        d = dict(d)
        # Named explicitly, because `cls(**d)` reported them as
        # `TypeError: DesignSpec.__init__() missing 1 required positional
        # argument: 'goal_statement'` -- a Python-internals message for a
        # missing line in the user's own file.
        for required, why in (
            ("goal", "bind_tighter, switch_specificity, reshape_channel, "
                      "improve_catalysis or stabilize (`ipro-agent goals` "
                      "lists what each one optimizes)"),
            ("goal_statement", "the objective in your own words, which travels "
                               "with the design and into the report"),
        ):
            if not d.get(required):
                raise SpecError(
                    f"the spec has no `{required}:`. It needs one: {why}."
                )
        try:
            mols = [Molecule(**m) for m in as_sequence(d.pop("molecules", []))]
        except TypeError as exc:
            raise SpecError(
                f"a molecule entry has the wrong keys ({exc}). Each needs "
                f"name, path, role and kind -- see `ipro://example-spec`."
            ) from exc

        positions: list[DesignPosition] = []
        raw_positions = d.pop("positions", [])
        if isinstance(raw_positions, dict):
            # `as_sequence` over a mapping yields its KEYS, so
            #     positions:
            #       A18: [TRP]
            #       A20: [TRP]
            # parsed as two positions with the full 18-residue default and the
            # requested alphabets silently dropped -- measured, a design space
            # of 324 sequences where the user had asked for 1. The list form
            # is the only one that carries `allowed`.
            example = next(iter(raw_positions.items()), ("A18", ["TRP"]))
            raise SpecError(
                "`positions:` must be a LIST, not a mapping -- a mapping drops "
                "every allowed-residue set and silently widens the alphabet to "
                "all 18. Write:\n"
                f"  positions:\n"
                f"    - {{position: {example[0]}, allowed: {example[1]}}}"
            )
        for item in as_sequence(raw_positions):
            if isinstance(item, (str, int)):
                positions.append(DesignPosition.parse(item))
            elif isinstance(item, dict):
                # Two accepted spellings: {"position": "A115", "allowed": [...]}
                # or the explicit {"chain": "A", "number": 115, ...}.
                if "position" in item:
                    # Read every key, not just these two. Reading only
                    # `position` and `allowed` silently dropped a misspelled
                    # `alowed:` (18 residues instead of 1) and an
                    # `insertion_code:` (designing residue 16 instead of 16B),
                    # while the explicit spelling rejected the same input.
                    rest = {k: v for k, v in item.items() if k != "position"}
                    allowed = rest.pop("allowed", None)
                    p = DesignPosition.parse(item["position"], allowed)
                    unknown = set(rest) - {"insertion_code", "native"}
                    if unknown:
                        raise SpecError(
                            f"position {item['position']!r} has unknown key(s) "
                            f"{sorted(unknown)}; a typo must not silently change "
                            "the design space"
                        )
                    if rest.get("insertion_code"):
                        p.insertion_code = str(rest["insertion_code"]).upper()
                    if rest.get("native"):
                        p.native = rest["native"]
                    positions.append(p)
                else:
                    positions.append(DesignPosition(**item))
            else:
                raise SpecError(f"cannot read position entry {item!r}")

        indels = [IndelSpec(**i) for i in as_sequence(d.pop("indels", []))]
        restraints = [Restraint(**r) for r in as_sequence(d.pop("restraints", []))]
        assemblies = [Assembly(**a) for a in as_sequence(d.pop("assemblies", []))]
        loop = LoopSettings(**d.pop("loop", {}))
        backends = Backends(**d.pop("backends", {}))

        unknown = set(d) - {
            "goal_statement", "goal", "goal_options", "name", "notes",
        }
        if unknown:
            raise SpecError(
                f"unknown top-level key(s) in spec: {sorted(unknown)}. "
                "Misspelled keys are rejected rather than ignored so a typo "
                "cannot silently change the design problem."
            )
        return cls(
            molecules=mols, positions=positions, indels=indels,
            restraints=restraints, assemblies=assemblies,
            loop=loop, backends=backends, **d,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DesignSpec":
        """Load from YAML (preferred) or JSON, chosen by content, not suffix."""
        p = Path(path).expanduser()
        text = p.read_text(encoding="utf-8")
        data: Any
        if p.suffix.lower() in (".json",):
            data = json.loads(text)
        else:
            try:
                import yaml  # type: ignore
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise SpecError(
                    "reading a YAML spec needs PyYAML (pip install pyyaml); "
                    "or write the spec as .json"
                ) from exc
            try:
                data = yaml.safe_load(text)
            except yaml.YAMLError as exc:
                # A tab for indentation, an unclosed bracket, a stray colon --
                # all ordinary mistakes, all of which came out as a 20-frame
                # traceback ending in yaml.scanner.ScannerError. PyYAML's own
                # message carries the line and column, so it is the useful
                # part; the frames are not.
                where = ""
                mark = getattr(exc, "problem_mark", None)
                if mark is not None:
                    where = f" at line {mark.line + 1}, column {mark.column + 1}"
                detail = getattr(exc, "problem", None) or str(exc).splitlines()[0]
                raise SpecError(
                    f"{p} is not valid YAML{where}: {detail}. "
                    f"(A tab character cannot indent YAML -- use spaces.)"
                ) from exc
        if not isinstance(data, dict):
            raise SpecError(
                f"{p} does not contain a mapping at the top level"
                + (f" (it parsed as {type(data).__name__})" if data is not None
                   else " (it is empty)")
            )
        spec = cls.from_dict(data)
        # Resolve relative molecule paths against the spec file's directory, so
        # an example case is portable.
        for m in spec.molecules:
            if not m.path.is_absolute():
                m.path = (p.parent / m.path).resolve()
        return spec

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for m in d["molecules"]:
            m["path"] = str(m["path"])
            m["kind"] = str(m["kind"].value if hasattr(m["kind"], "value") else m["kind"])
            m["role"] = str(m["role"].value if hasattr(m["role"], "value") else m["role"])
        for i in d["indels"]:
            i["kind"] = str(i["kind"].value if hasattr(i["kind"], "value") else i["kind"])
        for a in d.get("assemblies", []):
            pass   # plain fields only
        for r in d.get("restraints", []):
            r["kind"] = str(r["kind"].value if hasattr(r["kind"], "value") else r["kind"])
            r.pop("refs", None)   # derived; re-parsed from `atoms` on load
        return d

    def summary(self) -> str:
        """Compact human/agent-readable digest (cheap to print in a transcript)."""
        lines = [
            f"spec: {self.name}",
            f"goal: {self.goal}  -- {self.goal_statement!r}",
            f"design molecule: {self.design_molecule.name} ({self.design_molecule.path.name})",
        ]
        if self.targets:
            lines.append(
                "targets: " + ", ".join(f"{m.name}[{m.kind.value}]" for m in self.targets)
            )
        if self.anti_targets:
            lines.append(
                "anti-targets: " + ", ".join(f"{m.name}[{m.kind.value}]" for m in self.anti_targets)
            )
        lines.append(f"design positions ({len(self.positions)}):")
        for p in self.positions:
            nat = f"{p.native}->" if p.native else ""
            lines.append(
                f"  {p.label}: {nat}{len(p.allowed)} allowed "
                f"({','.join(THREE_TO_ONE.get(a, a) for a in p.allowed)})"
            )
        for i in self.indels:
            if i.kind is IndelKind.INSERT:
                lines.append(f"  insert {i.sequence or f'{i.length} residues'} after {i.chain}{i.after}")
            else:
                lines.append(f"  delete {i.chain}{i.residues}")
        for r in self.restraints:
            lines.append(f"  restraint {r.describe()}")
        asm = self.resolved_assemblies()
        if len(asm) > 1 or self.assemblies:
            lines.append(f"assemblies ({len(asm)}):")
            for a in asm:
                gate = "" if a.must_pass else "  (not gating acceptance)"
                lines.append(
                    f"  {a.name}: {', '.join(a.molecules) or '(design molecule alone)'}"
                    f"  weight {a.weight:g}{gate}"
                )
        n = self.design_space_size()
        lines.append(f"design space: {n:.3g} sequences" if n > 1e6 else f"design space: {n} sequences")
        lines.append(f"iterations: {self.loop.iterations}")
        return "\n".join(lines)
