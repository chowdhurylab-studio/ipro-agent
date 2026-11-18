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
# and greppable; checked for consistency in tests/test_config.py.
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


def expand_allowed(tokens: Iterable[str]) -> list[str]:
    """Expand a mixed list of group names, 1-letter and 3-letter codes.

    >>> expand_allowed(["hydrophobic", "W"])[:3]
    ['ALA', 'GLY', 'ILE']
    >>> expand_allowed(["ALA", "A"])
    ['ALA']

    Raises :class:`SpecError` on an unrecognized token, rather than silently
    dropping it -- a dropped residue would quietly shrink the design space and
    change the answer.
    """
    out: list[str] = []
    for tok in tokens:
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
        return f"{self.chain}{self.number}{self.insertion_code}".strip()

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
        return cls(
            chain=m.group("chain") or m.group("nchain") or "A",
            number=int(m.group("num")),
            insertion_code=(m.group("icode") or "").upper(),
            allowed=list(allowed) if allowed else list(AA_GROUPS["standard"]),
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
    #: Hard cap on rotamers per MILP, as IPRO caps at 1300. Exceeding it
    #: triggers tighter pruning rather than an unsolvable program.
    max_rotamers: int = 1300

    #: Residues within this distance (A) of a perturbed/mutated position are
    #: repacked alongside it, even if not design positions. IPRO uses a 4.5 A
    #: shell around the perturbation window.
    neighbour_cutoff: float = 4.5
    #: Non-bonded energy cutoff (A). IPRO: 12 A vdW.
    energy_cutoff: float = 12.0

    #: How many contiguous positions are re-designed per iteration. The IPRO
    #: Suite paper describes 1-3 contiguous residues; the whole design set is
    #: not re-chosen every iteration.
    design_per_iteration: tuple[int, int] = (1, 3)
    #: Length of the backbone segment perturbed each iteration. IPRO's source
    #: picks 1-5 consecutive residues; this is a separate choice from which
    #: positions are re-designed.
    perturbation_segment: tuple[int, int] = (1, 5)
    #: Backbone perturbation: phi/psi offsets ~ N(0, sigma), rejection-truncated
    #: at +/-cap. IPRO's source uses random.gauss(0, 1.5) truncated to +/-5 deg;
    #: the 2006 paper reports sigma = 1.65 deg.
    perturbation_sigma_deg: float = 1.5
    perturbation_cap_deg: float = 5.0
    #: Half-width (residues) of the perturbation window either side of the
    #: chosen segment.
    perturbation_flank: int = 5

    #: Rigid-body redocking of target molecules every Nth iteration.
    redock_every: int = 3
    redock_moves: int = 500
    redock_sigma_angstrom: float = 0.2
    redock_sigma_deg: float = 2.0

    #: Metropolis acceptance temperature in kelvin. IPRO's default, 3640 K, is
    #: calibrated so that a 10 kcal/mol worsening is accepted 25% of the time:
    #: exp(-10 / (R * 3640)) = 0.251 with R = 1.987e-3. Verified in
    #: tests/test_loop.py. IPRO's source draws an acceptance *window*
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

    def __post_init__(self) -> None:
        # YAML gives lists where the dataclass wants tuples.
        for attr in ("design_per_iteration", "perturbation_segment"):
            val = getattr(self, attr)
            if isinstance(val, list):
                setattr(self, attr, tuple(val))
            lo, hi = getattr(self, attr)
            if lo < 1 or hi < lo:
                raise SpecError(f"{attr} must be (lo<=hi) with lo>=1; got {(lo, hi)}")
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
    dock: str = "auto"         # auto | vina | none
    milp_solver: str = "auto"  # auto | cbc | glpk | cplex | gurobi | highs | greedy

    def __post_init__(self) -> None:
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
    loop: LoopSettings = field(default_factory=LoopSettings)
    backends: Backends = field(default_factory=Backends)
    #: Free-form, goal-specific knobs (e.g. a target pore radius). Validated by
    #: the goal implementation, not here.
    goal_options: dict[str, Any] = field(default_factory=dict)
    name: str = "design"
    notes: str = ""

    # -- derived views ----------------------------------------------------

    @property
    def design_molecule(self) -> Molecule:
        ms = [m for m in self.molecules if m.role is Role.DESIGN]
        if len(ms) != 1:
            raise SpecError(
                f"exactly one molecule must have role 'design'; found {len(ms)}"
            )
        return ms[0]

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
        d = dict(d)
        mols = [Molecule(**m) for m in d.pop("molecules", [])]

        positions: list[DesignPosition] = []
        for item in d.pop("positions", []):
            if isinstance(item, str):
                positions.append(DesignPosition.parse(item))
            elif isinstance(item, dict):
                # Two accepted spellings: {"position": "A115", "allowed": [...]}
                # or the explicit {"chain": "A", "number": 115, ...}.
                if "position" in item:
                    positions.append(
                        DesignPosition.parse(item["position"], item.get("allowed"))
                    )
                else:
                    positions.append(DesignPosition(**item))
            else:
                raise SpecError(f"cannot read position entry {item!r}")

        indels = [IndelSpec(**i) for i in d.pop("indels", [])]
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
            loop=loop, backends=backends, **d,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DesignSpec":
        """Load from YAML (preferred) or JSON, chosen by content, not suffix."""
        p = Path(path).expanduser()
        text = p.read_text()
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
            data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise SpecError(f"{p} does not contain a mapping at the top level")
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
        n = self.design_space_size()
        lines.append(f"design space: {n:.3g} sequences" if n > 1e6 else f"design space: {n} sequences")
        lines.append(f"iterations: {self.loop.iterations}")
        return "\n".join(lines)
