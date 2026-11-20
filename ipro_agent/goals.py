"""From a stated design goal to an objective function.

This is the module that generalizes IPRO. Stock IPRO (and each of its
derivatives -- OptCDR, OptZyme, PoreDesigner, IPRO+/-) hard-codes one objective
for one kind of question. Here each kind of question is a :class:`Goal` that
knows how to turn measured energies into a single number to minimize, which
default constraints it implies, and -- importantly -- which published work
justifies that encoding.

The common currency is the **interaction energy**, defined as IPRO's source
defines it:

    IE(design, partner) = E(complex) - E(design alone) - E(partner alone)

A more negative IE means tighter association.

Sign convention
---------------
Every goal returns a ``score`` that the loop **minimizes**. This matters
because the published work is inconsistent: standard IPRO minimizes an
interaction energy, while PoreDesigner *maximizes* one (a tighter pore means a worse
water interaction, so
a worse water interaction is the aim). Rather than carry that inconsistency
into the code, each goal is responsible for emitting an already-normalized
score, and says in its docstring which direction the source paper used.

What these objectives can and cannot do
---------------------------------------
Interaction energy of this kind reproduces the *rank* and *sign* of a change
much better than its magnitude. The published IPRO-family work reports
experimental hit rates around 24-43%, and at least one case missed a 250-fold
K_m shift. Treat a score difference as a hypothesis to test, not a predicted
affinity. See docs/DESIGN.md for the specific caveats per paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import AA_GROUPS, DesignSpec, Role, MolKind, SpecError, as_sequence
from .milp import Constraint

#: ``goal_options`` keys read outside this module, so a goal must not reject
#: them. Kept here beside the per-goal sets because this is where the whole
#: vocabulary is enforced; each is grepped from its reader.
SHARED_OPTION_KEYS: frozenset[str] = frozenset({
    "reference_energies",        # milp.build_tables, cli, loop._self_energy_terms
    "strain_weight",             # loop._self_energy_terms
    "max_self_energy",           # loop._tables_for, restraints
    "rotamer_window",            # loop._tables_for (IPRO's relative prune)
    "pair_eps",                  # loop._tables_for
    "insert_phi", "insert_psi",  # loop, terminal insertions
    "pore_axis", "pore_axis_flatness_warn",   # loop._pore_axis
    "pore_window", "pore_window_half",        # loop._measure_geometry
    "pore_spacing", "pore_max_centre_offset", # loop._measure_geometry
})


class GoalKind(str, Enum):
    """The design questions this tool can express."""

    BIND_TIGHTER = "bind_tighter"
    SWITCH_SPECIFICITY = "switch_specificity"
    RESHAPE_CHANNEL = "reshape_channel"
    IMPROVE_CATALYSIS = "improve_catalysis"
    STABILIZE = "stabilize"
    CUSTOM = "custom"


@dataclass
class ScoreContext:
    """Everything measured on one candidate structure.

    Populated by the loop; consumed by a goal. Keeping this a plain record
    (rather than letting goals reach into structures themselves) is what makes
    a goal testable without running any external program.
    """

    #: Interaction energy per partner molecule name.
    interaction: dict[str, float] = field(default_factory=dict)
    #: Whole-complex energy, for goals that constrain or minimize stability.
    total: float = 0.0
    #: Energy of the design molecule alone (a stability proxy).
    design_only: float = 0.0
    #: Geometric measurements, e.g. ``{"min_pore_radius": 3.1}``.
    geometry: dict[str, float] = field(default_factory=dict)
    #: The MILP objective value of the chosen packing, for reporting.
    milp_objective: float = 0.0

    def ie(self, name: str) -> float:
        if name not in self.interaction:
            raise KeyError(
                f"no interaction energy recorded for molecule {name!r}; "
                f"have {sorted(self.interaction)}"
            )
        return self.interaction[name]


class Goal:
    """Base class: turns a :class:`ScoreContext` into a number to minimize."""

    kind: GoalKind = GoalKind.CUSTOM
    #: One-line statement of what is optimized, shown in reports.
    objective_text: str = ""
    #: Where this encoding comes from. Required: an unsourced objective is a
    #: guess, and the user needs to know which it is.
    source: str = ""
    #: ``goal_options`` keys this goal reads. Anything outside this set (plus
    #: :data:`SHARED_OPTION_KEYS`) is refused, because a misspelled option is
    #: inert: ``pore_diameter_maximum`` for ``pore_diameter_max`` meant
    #: PoreDesigner's published hard filter never ran, and the design shipped
    #: unfiltered with nothing said. The repo already refuses unknown keys
    #: everywhere else in the spec; this was the one hole, and it carried every
    #: objective weight.
    OPTION_KEYS: frozenset[str] = frozenset()
    #: True if the goal reads ``spec.anti_targets``. A declared anti-target
    #: that no goal term reads is reported rather than ignored.
    uses_anti_targets: bool = False

    def __init__(self, spec: DesignSpec) -> None:
        self.spec = spec
        self.options: dict[str, Any] = dict(spec.goal_options)
        #: Qualifications the goal wants carried into the run's report -- a
        #: degenerate objective, a missing molecule, an unused option. The loop
        #: *shares* this list (it does not copy it), so a note raised later,
        #: while scoring, still reaches state.json.
        self.notes: list[str] = []
        self._check_option_keys()
        self.validate()
        if self.spec.anti_targets and not self.uses_anti_targets:
            self.notes.append(
                f"goal {self.kind.value!r} does not read anti-targets, so "
                f"{[m.name for m in self.spec.anti_targets]} "
                "is held fixed and scored into the complex energy but does NOT "
                "penalize the objective. Use goal 'switch_specificity' to select "
                "against a decoy, or give it role 'target' if it should be bound."
            )

    def _check_option_keys(self) -> None:
        known = set(self.OPTION_KEYS) | set(SHARED_OPTION_KEYS)
        unknown = sorted(set(self.options) - known)
        if unknown:
            raise SpecError(
                f"unknown goal_options key(s) {unknown} for goal "
                f"{self.kind.value!r}. This goal reads "
                f"{sorted(self.OPTION_KEYS) or '(no goal-specific options)'}; "
                f"keys read elsewhere in the run: {sorted(SHARED_OPTION_KEYS)}. "
                "Misspelled keys are rejected rather than ignored so a typo "
                "cannot silently change the design problem."
            )

    def _number(self, name: str, default: "float | None" = None) -> "float | None":
        """A numeric option, refused at validation rather than at run time.

        ``target_radius: "two"`` passed ``ipro-agent validate`` ("spec is
        usable", no problems) and then killed the run with
        ``ValueError: could not convert string to float: 'two'`` -- measured,
        at iteration 0 of a run the user had already waited for. Every numeric
        option goes through here, and every goal's ``validate`` calls it, so
        the spec fails before any CPU is spent.
        """
        if self.options.get(name) is None:
            return default
        raw = self.options[name]
        if isinstance(raw, bool):
            raise SpecError(
                f"goal_options.{name} must be a number; got the boolean {raw!r}"
            )
        try:
            val = float(raw)
        except (TypeError, ValueError) as exc:
            raise SpecError(
                f"goal_options.{name} must be a number; got "
                f"{type(raw).__name__} {raw!r}"
            ) from exc
        if not math.isfinite(val):
            raise SpecError(f"goal_options.{name} must be finite; got {raw!r}")
        return val

    def _check_numeric_options(self, *names: str) -> None:
        """Coerce each named option now, for its error message alone."""
        for name in names:
            self._number(name)

    def _positive(self, name: str, default: float) -> float:
        """A weight, refused if it is zero or negative.

        A weight scales a contribution to the objective; zero deletes the term
        and a negative value *inverts* it. Measured with ``anti_weight: -1``,
        binding the anti-target 8 kcal/mol tighter read as a 20 kcal/mol
        improvement -- the loop optimized the opposite of the request and its
        objective curve still fell. ``Assembly.weight`` is refused on the same
        grounds (config.py), so the convention already exists.
        """
        if name not in self.options:
            return float(default)
        try:
            val = float(self.options[name])
        except (TypeError, ValueError) as exc:
            raise SpecError(f"goal_options.{name} must be a number") from exc
        if val <= 0 or not math.isfinite(val):
            raise SpecError(
                f"goal_options.{name} is {val}; a weight scales a contribution to "
                "the objective and must be positive and finite (a negative weight "
                "inverts the term, so the run would optimize the opposite of what "
                "was asked). To drop a term, remove the option it weights."
            )
        return val

    # -- interface ---------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`SpecError` if the spec cannot support this goal."""

    #: True when the goal reads ``ScoreContext.geometry``. Measuring a channel
    #: costs 0.1-0.6 s per iteration, so goals that do not need it do not pay.
    wants_geometry: bool = False

    def score(self, ctx: ScoreContext) -> float:
        raise NotImplementedError

    def filter_reject(self, ctx: ScoreContext) -> str | None:
        """A reason to discard this candidate outright, or None.

        Distinct from a bad score: PoreDesigner discards any redesign whose
        bottleneck falls outside the requested window *before* evaluating its
        interaction energy, rather than letting a good energy outweigh a pore of
        the wrong size. Returning a reason here rejects the iteration however
        good its score was.
        """
        return None

    def components(self, ctx: ScoreContext) -> dict[str, float]:
        """Named parts of the score, so a design can be explained."""
        return {"score": self.score(ctx)}

    def default_constraints(self) -> list[Constraint]:
        return []

    def describe(self) -> str:
        return f"{self.kind.value}: {self.objective_text}  [{self.source}]"

    # -- helpers -----------------------------------------------------------

    def _require_targets(self, n: int = 1) -> None:
        if len(self.spec.targets) < n:
            raise SpecError(
                f"goal {self.kind.value!r} needs at least {n} molecule(s) with "
                f"role 'target'; the spec has {len(self.spec.targets)}"
            )

    def _sum_ie(self, ctx: ScoreContext, molecules) -> float:
        """Sum the interaction energies of ``molecules``, once each.

        ``ctx.interaction`` is keyed by molecule *name*, so summing over
        molecule objects counted a repeated name once per object. Duplicate
        names are now refused in :class:`~ipro_agent.config.DesignSpec`; this
        sums over names as well so the two cannot disagree again.
        """
        seen: list[str] = []
        for m in molecules:
            if m.name not in seen:
                seen.append(m.name)
        return sum(ctx.ie(n) for n in seen)


# ---------------------------------------------------------------------------
# Affinity
# ---------------------------------------------------------------------------


class BindTighter(Goal):
    """Maximize affinity for the target(s).

    score = sum over targets of IE(design, target)

    Minimizing this makes the interaction more favourable. This is the stock
    IPRO objective and the natural encoding for protein-protein interface
    affinity maturation and for peptide binders.

    Optionally constrain stability rather than ignoring it: with
    ``goal_options.max_total_energy`` set, candidates whose whole-complex energy
    exceeds that value are rejected. IPRO+/- uses the mirror-image form,
    minimizing complex energy subject to keeping IE within 25% of wild type.
    """

    kind = GoalKind.BIND_TIGHTER
    objective_text = "minimize sum of interaction energies with the target molecules"
    source = "IPRO (Saraf et al., Biophys J 2006); IPRO+/- (Chowdhury et al., Structure 2020)"
    OPTION_KEYS = frozenset({"max_total_energy"})

    def validate(self) -> None:
        self._require_targets(1)
        self._check_numeric_options("reference_interaction",
                                    "min_interaction_fraction")

    def score(self, ctx: ScoreContext) -> float:
        return self._sum_ie(ctx, self.spec.targets)

    def filter_reject(self, ctx: ScoreContext) -> str | None:
        """Enforce ``max_total_energy``, which the docstring above promises.

        It was documented and read nowhere: a spec asking for a −50 kcal/mol
        stability cap ran to completion on candidates at +500 with no note.
        ``ctx.total`` was already measured and carried for this purpose.
        """
        cap = self.options.get("max_total_energy")
        if cap is None:
            return None
        if ctx.total > float(cap):
            return (
                f"whole-complex energy {ctx.total:.3f} exceeds "
                f"max_total_energy {float(cap):.3f}"
            )
        return None

    def components(self, ctx: ScoreContext) -> dict[str, float]:
        out = {f"IE[{m.name}]": ctx.ie(m.name) for m in self.spec.targets}
        out["score"] = self.score(ctx)
        out["total_energy"] = ctx.total
        return out


# ---------------------------------------------------------------------------
# Specificity
# ---------------------------------------------------------------------------


class SwitchSpecificity(Goal):
    """Bind the target, not the anti-target.

    score = sum IE(targets) - weight * sum IE(anti-targets)

    This is the ``Delta IE`` metric of the acyl-ACP thioesterase redesign
    (Grisewood et al., ACS Catal 2017), which ranked designs by
    ``IE_short - IE_long`` -- the difference between the substrate wanted and
    the substrate to be rejected -- and minimized it. The 2006 IPRO paper does
    the same thing with a decoy ligand (D- versus L-arabinose in its AraC case).

    ``weight`` (``goal_options.anti_weight``, default 1.0) trades off how much
    rejecting the anti-target matters relative to binding the target. With
    weight 1 the score is a pure selectivity difference and is indifferent to
    binding both equally well or equally badly; raise it to push harder against
    the anti-target, and add ``max_target_ie`` to insist the wanted interaction
    stays below a floor.
    """

    kind = GoalKind.SWITCH_SPECIFICITY
    objective_text = "minimize IE(target) - w * IE(anti-target), i.e. the selectivity gap"
    source = "Grisewood et al., ACS Catal 2017 (Delta IE); Saraf et al., Biophys J 2006 (decoy ligand)"
    OPTION_KEYS = frozenset({"anti_weight", "max_target_ie"})
    uses_anti_targets = True

    def validate(self) -> None:
        self._require_targets(1)
        self._check_numeric_options("anti_weight", "reference_interaction",
                                    "min_interaction_fraction")
        if not self.spec.anti_targets:
            raise SpecError(
                "goal 'switch_specificity' needs at least one molecule with "
                "role 'anti_target' -- the binding partner to select against. "
                "Without one, use goal 'bind_tighter'."
            )
        # At construction, not at the first score(): `ipro-agent validate` must
        # catch a sign typo in the weight before a run spends hours optimizing
        # the opposite of the request.
        self._positive("anti_weight", 1.0)

    @property
    def weight(self) -> float:
        return self._positive("anti_weight", 1.0)

    def score(self, ctx: ScoreContext) -> float:
        want = self._sum_ie(ctx, self.spec.targets)
        avoid = self._sum_ie(ctx, self.spec.anti_targets)
        s = want - self.weight * avoid
        cap = self.options.get("max_target_ie")
        if cap is not None and want > float(cap):
            # Penalize, rather than reject, so the loop can still move: a hard
            # rejection here would strand the search when every candidate in a
            # neighbourhood violates the floor.
            s += (want - float(cap)) ** 2
        return s

    def components(self, ctx: ScoreContext) -> dict[str, float]:
        want = self._sum_ie(ctx, self.spec.targets)
        avoid = self._sum_ie(ctx, self.spec.anti_targets)
        return {
            "IE_target": want,
            "IE_anti_target": avoid,
            "selectivity_gap": want - avoid,
            "score": self.score(ctx),
        }


# ---------------------------------------------------------------------------
# Channels and pores
# ---------------------------------------------------------------------------


class ReshapeChannel(Goal):
    """Narrow or widen a channel, pore or substrate access path.

    PoreDesigner (Chowdhury et al., Nat Commun 2018) narrowed the OmpF pore by
    lining its lumen with bulky residues, and encoded that by **maximizing**
    the interaction energy with a single-file water probe threaded through the
    channel -- a worse water interaction means a tighter lumen -- while a MILP
    constraint forced at least half the designed positions to take a long
    residue (W/F/Y/M).

    **Narrowing is not occluding, and the objective alone cannot tell the
    difference.** The published aim was a selective channel that still passes
    water single-file: their designs were filtered to a sub-4 A bottleneck and
    then *verified to conduct*, and the paper's own constraint keeps the lumen
    open. This objective is monotonic -- the tighter the worse the probe
    interaction, so a fully blocked pore is its maximum. Measured with
    `spec_full.yaml`'s own options (direction narrow, `pore_diameter_max` 4.0,
    `pore_diameter_min` null): a candidate with a 0.8 A bottleneck and the
    probe clashing at IE +50 passes `filter_reject` and scores -50.0, beating
    a paper-style compliant design at 3.5 A (+0.200). So **set
    ``pore_diameter_min``** -- about 2.8 A for water -- or the best-scoring
    design is a plug. The goal says so in a note when narrowing without one.

    Here the direction is a parameter, because the opposite request (widen a
    substrate channel) is just as common:

    ``goal_options.direction``
        ``"narrow"`` (default, as published): score = -IE(probe), optionally
        plus a term rewarding occlusion.
        ``"widen"``: score = +IE(probe); a more favourable probe interaction
        means a more open, better-solvated lumen.

    ``goal_options.target_radius``
        If set, and the loop supplies ``geometry["min_pore_radius"]``, the score
        adds ``(radius - target)^2``, turning the goal into "reach this radius"
        rather than "go as far as possible".

    Note the honest limitation: without the geometric measurement, this goal is
    optimizing a *proxy* (probe interaction), not the pore size itself. The
    published work measured pore size by separate analysis and by MD, not inside
    the optimization loop.
    """

    kind = GoalKind.RESHAPE_CHANNEL
    objective_text = "narrow or widen a lumen via a probe interaction energy and/or its measured radius"
    source = "PoreDesigner (Chowdhury et al., Nat Commun 2018)"
    wants_geometry = True
    OPTION_KEYS = frozenset({
        "direction", "target_radius", "radius_weight",
        "pore_diameter_min", "pore_diameter_max",
        "long_fraction", "long_residues",
    })

    def validate(self) -> None:
        d = str(self.options.get("direction", "narrow")).lower()
        if d not in ("narrow", "widen"):
            raise SpecError(
                f"goal_options.direction must be 'narrow' or 'widen'; got {d!r}"
            )
        self._positive("radius_weight", 1.0)
        # Coerced here so a typo'd number fails `validate` rather than the run:
        # `target_radius: "two"` reported "spec is usable" and then died at
        # iteration 0 with ValueError.
        self._check_numeric_options(
            "target_radius", "pore_diameter_min", "pore_diameter_max",
            "long_fraction", "pore_spacing", "pore_max_centre_offset",
            "pore_window_half", "pore_axis_flatness_warn",
        )
        # `pore_diameter_min/max` are a hard acceptance FILTER (filter_reject),
        # not a term in the objective -- nothing in score() reads them. Counting
        # them as a geometric objective suppressed the degenerate-objective
        # warning for exactly the spec that states the paper's published filter:
        # a 3.2 A and a 3.8 A candidate both scored -100.0 and the user was told
        # nothing, while a spec with no options at all got the warning.
        # Narrowing with no floor: the objective's maximum is a blocked pore.
        if (d == "narrow"
                and self.options.get("pore_diameter_min") is None
                and not any("pore_diameter_min" in n for n in self.notes)):
            self.notes.append(
                "narrowing with no `goal_options.pore_diameter_min`: this "
                "objective is monotonic, so a FULLY OCCLUDED pore is its "
                "best possible score -- measured, a 0.8 A bottleneck with the "
                "water probe clashing (+50) scores -50.0 and beats a "
                "paper-style 3.5 A design (+0.200). PoreDesigner's designs "
                "were required to still conduct water single-file. Set "
                "pore_diameter_min (about 2.8 A for water) unless you really "
                "do want a plug, and do not report a sub-4 A bottleneck as a "
                "selective channel without it."
            )
        has_objective = self.options.get("target_radius") is not None
        if not self.spec.targets and not has_objective:
            # Constraint-only mode. This is a real situation, not a misuse:
            # PoreDesigner's probe was a water wire taken from an MD trajectory
            # that was never deposited, and this tool implements no pore-geometry
            # measurement, so neither half of the published objective is
            # available. What remains is the composition constraint (>= half the
            # positions long) plus packing quality, which is a coherent thing to
            # optimize -- but it is a weaker statement than the paper's, so it is
            # recorded loudly rather than failing or pretending otherwise.
            self.notes.append(
                "reshape_channel has no probe molecule and no geometric target, "
                "so the objective is degenerate: it minimizes the design "
                "molecule's own packing energy and the residue-class constraint "
                "does the real work. Set goal_options.target_radius (to aim at a "
                "radius), or pore_diameter_min/max (the published hard filter), "
                "or give a probe molecule role 'target'."
            )

    @property
    def direction(self) -> str:
        return str(self.options.get("direction", "narrow")).lower()

    def filter_reject(self, ctx: ScoreContext) -> str | None:
        """PoreDesigner's acceptance filter: discard a pore of the wrong size.

        Applied before the energy matters, as published -- a design with a
        wonderful interaction energy and a bottleneck outside the requested
        window is not a candidate at all.
        """
        d = ctx.geometry.get("min_pore_diameter")
        if d is None:
            return None
        lo = self.options.get("pore_diameter_min")
        hi = self.options.get("pore_diameter_max")
        if lo is not None and d < float(lo):
            return f"bottleneck {d:.2f} A is below pore_diameter_min {float(lo):.2f} A"
        if hi is not None and d > float(hi):
            return f"bottleneck {d:.2f} A is above pore_diameter_max {float(hi):.2f} A"
        return None

    def score(self, ctx: ScoreContext) -> float:
        tgt = self.options.get("target_radius")
        if not self.spec.targets and tgt is None:
            # Constraint-only mode; see validate(). Packing energy stands in.
            return ctx.design_only if ctx.design_only else ctx.total
        s = 0.0
        if self.spec.targets:
            ie = self._sum_ie(ctx, self.spec.targets)
            # 'narrow' wants IE to go UP, so minimize its negation.
            s += -ie if self.direction == "narrow" else ie
        if tgt is not None:
            r = ctx.geometry.get("min_pore_radius")
            if r is None:
                # Asked to hit a radius, but nothing measures one. Silently
                # dropping the term would make the score mean something other
                # than what was requested.
                if not any("target_radius" in n for n in self.notes):
                    self.notes.append(
                        "goal_options.target_radius was set but no pore-geometry "
                        "measurement is available, so the radius term is not in "
                        "the score; only the probe-interaction term is active"
                    )
            else:
                s += self._positive("radius_weight", 1.0) * (r - float(tgt)) ** 2
        return s

    def default_constraints(self) -> list[Constraint]:
        """PoreDesigner's ">= 50% long residues" requirement, when narrowing.

        Emitted only for the published direction and only if the user has not
        set ``long_fraction`` to 0, so it can be switched off deliberately.
        """
        frac = self.options.get("long_fraction")
        if frac is None:
            frac = 0.5 if self.direction == "narrow" else 0.0
        frac = float(frac)
        if frac <= 0:
            return []
        residues = [
            str(r).upper()
            for r in as_sequence(self.options.get("long_residues") or ["TRP", "PHE", "TYR", "MET"])
        ]
        return [
            Constraint(
                kind="min_class",
                fraction=frac,
                residues=residues,
                label="long_residue_fraction",
            )
        ]

    def components(self, ctx: ScoreContext) -> dict[str, float]:
        out: dict[str, float] = {"score": self.score(ctx)}
        if self.spec.targets:
            out["IE_probe"] = self._sum_ie(ctx, self.spec.targets)
        if "min_pore_radius" in ctx.geometry:
            out["min_pore_radius"] = ctx.geometry["min_pore_radius"]
        return out


# ---------------------------------------------------------------------------
# Catalysis
# ---------------------------------------------------------------------------


class ImproveCatalysis(Goal):
    """Improve a kinetic parameter, via OptZyme's transition-state encoding.

    OptZyme (Grisewood et al., PLoS ONE 2013) related interaction energies to
    kinetics, giving three different objectives depending on which parameter is
    wanted. Set ``goal_options.parameter``:

    ``"km"``
        Minimize IE with the **substrate**. OptZyme's Eq 7: ``IE_S = RT ln K_M``,
        so a more favourable substrate interaction means a lower K_M.

    ``"kcat_over_km"`` (default)
        Minimize IE with a **transition-state analogue**. OptZyme's Eq 17:
        ``Delta IE_TSA = -RT [ln(kcat/K_M) - ln(kcat_wt/K_M_wt)]``, so lowering
        the TSA interaction energy raises the specificity constant.

    ``"kcat"``
        Minimize ``IE_TSA / RT_TSA - IE_S / RT_S``, from OptZyme's Eq 18, which
        separates the turnover number from substrate binding. This needs **both**
        a transition-state analogue and a substrate molecule.

        **The two divisors are different, and that is the whole point.** The
        paper states them outright for GUS/pNP-GLU: ``(RT)_TSA = 15.3 kJ/mol``
        and ``(RT)_S = 386.7 kJ/mol``, self-consistent with the weights it also
        gives (``1/(RT)_TSA = 0.06`` and ``1/(RT)_S = 0.002 mol/kJ``), so the
        substrate term carries 4% of the TSA term's weight. Dividing both by one
        RT -- which this code did -- sets that relative weight to 1.0 and
        changes the argmin, not just the scale: for candidates A (IE_TSA -15.0,
        IE_S -2.0) and B (IE_TSA -20.0, IE_S -12.0), Eq 18 gives A -4.080 and
        B -5.339 so B wins, while one RT gave A -21.944 and B -13.504 so A won
        -- rejecting the candidate that binds the transition state 5 kcal/mol
        better.

        These are **fitted correlating constants, not temperatures**: the paper
        says plainly that "experimental and correlating temperatures do not
        match", and they are system-specific to GUS/pNP-GLU. (Its parenthetical
        temperatures, 4.65e4 K and 1840 K, are transposed relative to its own
        RT values -- R*4.65e4 = 386.6 kJ/mol -- so the RT pair is the one to
        trust.) Override with ``goal_options.rt_tsa_kcal`` and
        ``goal_options.rt_substrate_kcal`` for another system.

    The molecule roles carry the distinction: a molecule of kind
    ``transition_state`` is the analogue, one of kind ``ligand`` is the
    substrate. ``goal_options.temperature_k`` sets RT for ``km`` and
    ``kcat_over_km``, where one RT is what the paper's Eqs 7 and 17 use; for
    ``kcat`` it is refused, because a single RT is meaningful for neither term.

    A warning worth heeding: the ACS Catal 2017 thioesterase work deliberately
    did *not* model the transition state, imposing catalytic-distance restraints
    instead and stating that it aimed to preserve, not boost, activity. Boosting
    catalysis is the hardest of these goals and the one where a computed score
    is least trustworthy.
    """

    kind = GoalKind.IMPROVE_CATALYSIS
    objective_text = "minimize the interaction energy that encodes the chosen kinetic parameter"
    source = "OptZyme (Grisewood et al., PLoS ONE 2013), Eqs 7, 17, 18"
    OPTION_KEYS = frozenset({
        "parameter", "temperature_k", "rt_tsa_kcal", "rt_substrate_kcal",
    })

    R_KCAL = 1.987e-3
    #: OptZyme's Eq-18 constants for GUS/pNP-GLU, converted from the paper's
    #: kJ/mol: 15.3 / 4.184 and 386.7 / 4.184. Fitted, system-specific, and
    #: 25.3x apart -- see the class docstring.
    RT_TSA_KCAL = 15.3 / 4.184
    RT_SUBSTRATE_KCAL = 386.7 / 4.184

    @property
    def parameter(self) -> str:
        return str(self.options.get("parameter", "kcat_over_km")).lower()

    def _tsa(self):
        return [m for m in self.spec.targets if m.kind is MolKind.TRANSITION_STATE]

    def _substrate(self):
        return [m for m in self.spec.targets if m.kind is MolKind.LIGAND]

    def validate(self) -> None:
        self._require_targets(1)
        self._check_numeric_options("temperature_k")
        # Positive, not merely numeric. These are divisors in Eq 18: zero
        # raises mid-run, and a negative silently INVERTS that term, so a
        # design that binds the transition state worse scores better.
        for name, default in (("rt_tsa_kcal", self.RT_TSA_KCAL),
                              ("rt_substrate_kcal", self.RT_SUBSTRATE_KCAL)):
            if self.options.get(name) is not None:
                value = self._number(name, default)
                if value is None or value <= 0:
                    raise SpecError(
                        f"goal_options.{name} is {self.options[name]!r}; it is "
                        f"a divisor in OptZyme's Eq 18 and must be positive. A "
                        f"negative value inverts that term, so a worse "
                        f"transition-state interaction would score better."
                    )
        p = self.parameter
        if p not in ("km", "kcat", "kcat_over_km"):
            raise SpecError(
                f"goal_options.parameter must be 'km', 'kcat' or 'kcat_over_km'; got {p!r}"
            )
        if p in ("kcat_over_km", "kcat") and not self._tsa():
            raise SpecError(
                f"parameter {p!r} is encoded through a transition-state analogue, so the "
                "spec needs a target molecule with kind 'transition_state'. "
                "Use parameter 'km' to optimize substrate binding instead."
            )
        if p == "kcat" and "temperature_k" in self.options:
            raise SpecError(
                "goal_options.temperature_k does not apply to parameter 'kcat': "
                "Eq 18 weights the two interaction energies by DIFFERENT fitted "
                "constants ((RT)_TSA = 15.3 kJ/mol, (RT)_S = 386.7 kJ/mol -- a "
                "factor of 25), and one temperature for both over-weights the "
                "substrate term 25-fold and changes which candidate wins. Set "
                "rt_tsa_kcal and rt_substrate_kcal (kcal/mol) if your system is "
                "not GUS/pNP-GLU."
            )
        if p == "kcat" and not self._substrate():
            raise SpecError(
                "parameter 'kcat' needs both a 'transition_state' molecule and a "
                "'ligand' (substrate) molecule, since it is the difference of the two terms"
            )
        if p == "km" and not self._substrate():
            # Eq 7 is IE with the *substrate*. With no molecule of kind
            # 'ligand', score() silently summed every target instead: measured,
            # an unrelated protein partner at -40 entered a "K_M" score of -49
            # that components() reported as -9, so the number could not even be
            # explained. One target is unambiguous (it is the substrate,
            # whatever its kind, and that is recorded); several are not.
            if len(self.spec.targets) == 1:
                only = self.spec.targets[0]
                self.notes.append(
                    f"parameter 'km' is OptZyme Eq 7, IE with the substrate, but no "
                    f"target molecule has kind 'ligand'. Using the only target, "
                    f"{only.name!r} (kind {only.kind.value!r}), as the substrate. Set "
                    f"its kind to 'ligand' to make this explicit."
                )
            else:
                raise SpecError(
                    "parameter 'km' is encoded as the interaction energy with the "
                    "substrate, but no target molecule has kind 'ligand' and there "
                    f"are {len(self.spec.targets)} targets "
                    f"({', '.join(m.name for m in self.spec.targets)}), so which one "
                    "is the substrate is ambiguous. Give the substrate kind 'ligand'."
                )

    @property
    def _rt(self) -> float:
        return self.R_KCAL * float(self.options.get("temperature_k", 298.15))

    @property
    def _rt_pair(self) -> tuple[float, float]:
        """``(RT_TSA, RT_substrate)`` in kcal/mol, for Eq 18."""
        return (
            float(self.options.get("rt_tsa_kcal", self.RT_TSA_KCAL)),
            float(self.options.get("rt_substrate_kcal", self.RT_SUBSTRATE_KCAL)),
        )

    def score(self, ctx: ScoreContext) -> float:
        if self.parameter == "km":
            return self._sum_ie(ctx, self._substrate() or self.spec.targets)
        if self.parameter == "kcat_over_km":
            return self._sum_ie(ctx, self._tsa())
        rt_tsa, rt_s = self._rt_pair
        return (
            self._sum_ie(ctx, self._tsa()) / rt_tsa
            - self._sum_ie(ctx, self._substrate()) / rt_s
        )

    def components(self, ctx: ScoreContext) -> dict[str, float]:
        out: dict[str, float] = {"parameter_score": self.score(ctx)}
        for m in self._tsa():
            out[f"IE_TSA[{m.name}]"] = ctx.ie(m.name)
        for m in self._substrate():
            out[f"IE_substrate[{m.name}]"] = ctx.ie(m.name)
        if self.parameter == "kcat":
            rt_tsa, rt_s = self._rt_pair
            out["RT_TSA_kcal"] = rt_tsa
            out["RT_substrate_kcal"] = rt_s
            out["weighted_IE_TSA"] = self._sum_ie(ctx, self._tsa()) / rt_tsa
            out["weighted_IE_substrate"] = (
                self._sum_ie(ctx, self._substrate()) / rt_s
            )
        out["score"] = self.score(ctx)
        return out


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


class Stabilize(Goal):
    """Minimize the energy of the complex.

    ``score = E(complex)`` when the spec has partners, and ``E(design alone)``
    when it does not -- with no partner the two are the same quantity.

    Useful on its own, and as the inner objective of IPRO+/-'s constrained
    form: minimize complex energy while holding the interaction energy within a
    fraction of wild type. Set ``goal_options.min_interaction_fraction`` (e.g.
    0.75) with at least one target to get that behaviour.
    """

    kind = GoalKind.STABILIZE
    objective_text = (
        "minimize the energy of the complex (the design molecule alone when "
        "there is no partner)"
    )
    source = "IPRO+/- (Chowdhury et al., Structure 2020), constrained formulation"
    OPTION_KEYS = frozenset({"min_interaction_fraction", "reference_interaction"})

    def validate(self) -> None:
        # Checked here, not only when the first candidate is scored. The error
        # raised from score() told the user to "measure it with 'ipro-agent
        # score'", and that command builds the same goal and dies with the same
        # error -- a recovery path that cannot be followed.
        self._check_numeric_options("min_interaction_fraction",
                                    "reference_interaction")
        if self.options.get("min_interaction_fraction") is None:
            return
        self._require_targets(1)
        if self.options.get("reference_interaction") is None:
            raise SpecError(
                "goal_options.min_interaction_fraction needs "
                "goal_options.reference_interaction (the wild-type interaction "
                "energy to stay within a fraction of). Measure it by running "
                "'ipro-agent score' on the input structure with goal "
                "'bind_tighter', then set the value here."
            )
        self._positive("min_interaction_fraction", 0.75)

    def _floor(self) -> float | None:
        frac = self.options.get("min_interaction_fraction")
        if frac is None or not self.spec.targets:
            return None
        return float(frac) * float(self.options["reference_interaction"])

    def filter_reject(self, ctx: ScoreContext) -> str | None:
        """IPRO+/-'s formulation states the interaction floor as a CONSTRAINT.

        docs/DESIGN.md section 1 records it as ``min E_complex s.t. IE >=
        0.75 * IE_WT``. As a unit-weight quadratic penalty it was not a
        constraint at all: measured, a candidate that destroyed the interaction
        entirely (IE 0.0 against a floor of -15.0) scored -115.0, *better* than
        a compliant candidate at -100.0, and was kept as the run's best design.
        """
        floor = self._floor()
        if floor is None:
            return None
        ie = self._sum_ie(ctx, self.spec.targets)
        if ie > floor:   # less negative than allowed
            return (
                f"interaction energy {ie:.3f} is above the floor {floor:.3f} "
                f"({self.options['min_interaction_fraction']} x reference "
                f"{self.options['reference_interaction']})"
            )
        return None

    def score(self, ctx: ScoreContext) -> float:
        # The COMPLEX energy when there are partners, which is what both
        # DESIGN.md and this class's own docstring cite (IPRO+/-: minimize the
        # complex energy subject to an interaction-energy floor). Scoring the
        # design molecule alone chose differently: measured with candidates
        # A (design_only -100.0, IE -20.0, complex -120.0) and B (-103.0,
        # -15.0, -118.0), the design-only objective picks B while min
        # E_complex picks A -- and both pass the floor, so nothing else caught
        # it. With no partner the complex IS the design molecule, which is the
        # original behaviour.
        s = ctx.total if self.spec.targets else (
            ctx.design_only if ctx.design_only else ctx.total
        )
        floor = self._floor()
        if floor is not None:
            # Kept as a gradient *inside* the feasible region's edge so the
            # search is pushed away from the floor, with filter_reject above as
            # the constraint proper.
            ie = self._sum_ie(ctx, self.spec.targets)
            if ie > floor:
                s += (ie - floor) ** 2
        return s

    def components(self, ctx: ScoreContext) -> dict[str, float]:
        out = {"design_energy": ctx.design_only, "total_energy": ctx.total}
        if self.spec.targets:
            out["IE_target"] = self._sum_ie(ctx, self.spec.targets)
        out["score"] = self.score(ctx)
        return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GOALS: dict[str, type[Goal]] = {
    GoalKind.BIND_TIGHTER.value: BindTighter,
    GoalKind.SWITCH_SPECIFICITY.value: SwitchSpecificity,
    GoalKind.RESHAPE_CHANNEL.value: ReshapeChannel,
    GoalKind.IMPROVE_CATALYSIS.value: ImproveCatalysis,
    GoalKind.STABILIZE.value: Stabilize,
}

#: Plain-language phrases an agent is likely to see, mapped to a goal. This is a
#: convenience for dispatch, not language understanding: the agent should pick
#: the goal deliberately and record the user's own words in
#: ``DesignSpec.goal_statement``.
GOAL_SYNONYMS: dict[str, str] = {
    "affinity": "bind_tighter",
    "bind": "bind_tighter",
    "binding": "bind_tighter",
    "tighter": "bind_tighter",
    "interface": "bind_tighter",
    "specificity": "switch_specificity",
    "selectivity": "switch_specificity",
    "discriminate": "switch_specificity",
    # The 2006 paper's own word for the partner to select against ("a decoy
    # ligand"), and the one a user is most likely to reach for. Without it,
    # "bind the target and not the decoy" matched only "bind" and suggested
    # plain affinity maturation.
    "decoy": "switch_specificity",
    "pore": "reshape_channel",
    "channel": "reshape_channel",
    "lumen": "reshape_channel",
    "widen": "reshape_channel",
    "narrow": "reshape_channel",
    "catalysis": "improve_catalysis",
    "catalytic": "improve_catalysis",
    "activity": "improve_catalysis",
    "kcat": "improve_catalysis",
    "km": "improve_catalysis",
    "enzyme": "improve_catalysis",
    "stability": "stabilize",
    "stabilize": "stabilize",
    "thermostability": "stabilize",
}


def build_goal(spec: DesignSpec) -> Goal:
    """Instantiate the goal named by ``spec.goal``, validating the spec."""
    name = str(spec.goal).strip().lower()
    if name not in GOALS:
        hint = GOAL_SYNONYMS.get(name)
        if hint:
            raise SpecError(
                f"unknown goal {spec.goal!r}; did you mean {hint!r}? "
                f"Valid goals: {', '.join(sorted(GOALS))}"
            )
        raise SpecError(
            f"unknown goal {spec.goal!r}. Valid goals: {', '.join(sorted(GOALS))}"
        )
    return GOALS[name](spec)


#: When a statement matches several goals, the more specific question wins.
#: Matching in word order answered the wrong question: "change its binding
#: specificity from C12 to C14" hit "binding" first and suggested
#: `bind_tighter`, which optimizes plain affinity and ignores the decoy
#: entirely. `bind_tighter` is the fallback, so it is last.
_GOAL_PRIORITY: tuple[str, ...] = (
    "switch_specificity",
    "improve_catalysis",
    "reshape_channel",
    "stabilize",
    "bind_tighter",
)


def suggest_goal(statement: str) -> str | None:
    """Best-guess goal for a plain-language statement, or None.

    Deliberately crude keyword matching. It exists so an agent can *offer* a
    goal and have the user confirm, not so the tool can silently decide what
    the user meant.
    """
    words = "".join(c.lower() if c.isalnum() else " " for c in statement).split()
    hits = {GOAL_SYNONYMS[w] for w in words if w in GOAL_SYNONYMS}
    for goal in _GOAL_PRIORITY:
        if goal in hits:
            return goal
    return None


def metropolis_accept(
    delta: float, kt: float, u: float, greedy: bool = False
) -> tuple[bool, str]:
    """IPRO's accept/reject test.

    ``delta`` is new minus old score (negative is an improvement), ``kt`` the
    acceptance temperature in the score's energy units, and ``u`` a uniform
    random draw in (0, 1]. Returns ``(accepted, reason)``.

    An improvement is always accepted. Otherwise the candidate is accepted with
    probability ``exp(-delta / kt)``, which IPRO's source implements in the
    algebraically identical form "accept if ``delta < -kt * ln(u)``".
    ``greedy=True`` disables thermal acceptance, as IPRO does for its first 10%
    of iterations.
    """
    if delta <= 0:
        return True, "improved"
    if greedy:
        return False, "worse (greedy phase)"
    if kt <= 0:
        return False, "worse (zero temperature)"
    window = -kt * math.log(max(u, 1e-300))
    return (delta < window), ("metropolis" if delta < window else "worse")
