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
interaction energy, while PoreDesigner *maximizes* one (it occludes a pore, so
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

from .config import AA_GROUPS, DesignSpec, Role, MolKind, SpecError
from .milp import Constraint


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

    def __init__(self, spec: DesignSpec) -> None:
        self.spec = spec
        self.options: dict[str, Any] = dict(spec.goal_options)
        #: Qualifications the goal wants carried into the run's report -- a
        #: degenerate objective, a missing molecule, an unused option. The loop
        #: collects these into state.json notes.
        self.notes: list[str] = []
        self.validate()

    # -- interface ---------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`SpecError` if the spec cannot support this goal."""

    def score(self, ctx: ScoreContext) -> float:
        raise NotImplementedError

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
        return sum(ctx.ie(m.name) for m in molecules)


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

    def validate(self) -> None:
        self._require_targets(1)

    def score(self, ctx: ScoreContext) -> float:
        return self._sum_ie(ctx, self.spec.targets)

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

    def validate(self) -> None:
        self._require_targets(1)
        if not self.spec.anti_targets:
            raise SpecError(
                "goal 'switch_specificity' needs at least one molecule with "
                "role 'anti_target' -- the binding partner to select against. "
                "Without one, use goal 'bind_tighter'."
            )

    @property
    def weight(self) -> float:
        return float(self.options.get("anti_weight", 1.0))

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
    channel -- a worse water interaction means a more occluded lumen -- while a
    MILP constraint forced at least half the designed positions to take a long
    residue (W/F/Y/M).

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
    objective_text = "narrow or widen a lumen via a probe-molecule interaction energy"
    source = "PoreDesigner (Chowdhury et al., Nat Commun 2018)"

    def validate(self) -> None:
        d = str(self.options.get("direction", "narrow")).lower()
        if d not in ("narrow", "widen"):
            raise SpecError(
                f"goal_options.direction must be 'narrow' or 'widen'; got {d!r}"
            )
        if not self.spec.targets and self.options.get("target_radius") is None:
            # Constraint-only mode. This is a real situation, not a misuse:
            # PoreDesigner's probe was a water wire taken from an MD trajectory
            # that was never deposited, and this tool implements no pore-geometry
            # measurement, so neither half of the published objective is
            # available. What remains is the composition constraint (>= half the
            # positions long) plus packing quality, which is a coherent thing to
            # optimize -- but it is a weaker statement than the paper's, so it is
            # recorded loudly rather than failing or pretending otherwise.
            self.notes.append(
                "reshape_channel has no probe molecule and no target_radius, so "
                "the objective is degenerate: it minimizes the design molecule's "
                "own packing energy and the residue-class constraint does the "
                "real work. This is NOT the published pore objective. Give the "
                "probe molecule role 'target' (the direction option, not the "
                "role, sets whether its interaction is strengthened or "
                "weakened), or set a target_radius, to optimize the channel "
                "itself."
            )

    @property
    def direction(self) -> str:
        return str(self.options.get("direction", "narrow")).lower()

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
                s += float(self.options.get("radius_weight", 1.0)) * (r - float(tgt)) ** 2
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
        return [
            Constraint(
                kind="min_class",
                fraction=frac,
                residues=list(self.options.get("long_residues", ["TRP", "PHE", "TYR", "MET"])),
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

    The molecule roles carry the distinction: a molecule of kind
    ``transition_state`` is the analogue, one of kind ``ligand`` is the
    substrate. Temperatures default to 298.15 K and can be set per term with
    ``goal_options.temperature_k``.

    A warning worth heeding: the ACS Catal 2017 thioesterase work deliberately
    did *not* model the transition state, imposing catalytic-distance restraints
    instead and stating that it aimed to preserve, not boost, activity. Boosting
    catalysis is the hardest of these goals and the one where a computed score
    is least trustworthy.
    """

    kind = GoalKind.IMPROVE_CATALYSIS
    objective_text = "minimize the interaction energy that encodes the chosen kinetic parameter"
    source = "OptZyme (Grisewood et al., PLoS ONE 2013), Eqs 7, 17, 18"

    R_KCAL = 1.987e-3

    @property
    def parameter(self) -> str:
        return str(self.options.get("parameter", "kcat_over_km")).lower()

    def _tsa(self):
        return [m for m in self.spec.targets if m.kind is MolKind.TRANSITION_STATE]

    def _substrate(self):
        return [m for m in self.spec.targets if m.kind is MolKind.LIGAND]

    def validate(self) -> None:
        self._require_targets(1)
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
        if p == "kcat" and not self._substrate():
            raise SpecError(
                "parameter 'kcat' needs both a 'transition_state' molecule and a "
                "'ligand' (substrate) molecule, since it is the difference of the two terms"
            )

    @property
    def _rt(self) -> float:
        return self.R_KCAL * float(self.options.get("temperature_k", 298.15))

    def score(self, ctx: ScoreContext) -> float:
        if self.parameter == "km":
            return self._sum_ie(ctx, self._substrate() or self.spec.targets)
        if self.parameter == "kcat_over_km":
            return self._sum_ie(ctx, self._tsa())
        rt = self._rt
        return self._sum_ie(ctx, self._tsa()) / rt - self._sum_ie(ctx, self._substrate()) / rt

    def components(self, ctx: ScoreContext) -> dict[str, float]:
        out: dict[str, float] = {"parameter_score": self.score(ctx)}
        for m in self._tsa():
            out[f"IE_TSA[{m.name}]"] = ctx.ie(m.name)
        for m in self._substrate():
            out[f"IE_substrate[{m.name}]"] = ctx.ie(m.name)
        out["score"] = self.score(ctx)
        return out


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


class Stabilize(Goal):
    """Minimize the energy of the design molecule itself.

    score = E(design alone)

    Useful on its own, and as the inner objective of IPRO+/-'s constrained
    form: minimize complex energy while holding the interaction energy within a
    fraction of wild type. Set ``goal_options.min_interaction_fraction`` (e.g.
    0.75) with at least one target to get that behaviour.
    """

    kind = GoalKind.STABILIZE
    objective_text = "minimize the energy of the design molecule"
    source = "IPRO+/- (Chowdhury et al., Structure 2020), constrained formulation"

    def score(self, ctx: ScoreContext) -> float:
        s = ctx.design_only if ctx.design_only else ctx.total
        frac = self.options.get("min_interaction_fraction")
        if frac is not None and self.spec.targets:
            ref = self.options.get("reference_interaction")
            if ref is None:
                raise SpecError(
                    "goal_options.min_interaction_fraction needs "
                    "goal_options.reference_interaction (the wild-type interaction "
                    "energy to stay within a fraction of); measure it with "
                    "'ipro-agent score' on the input structure first"
                )
            ie = self._sum_ie(ctx, self.spec.targets)
            floor = float(frac) * float(ref)
            if ie > floor:  # less negative than allowed
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


def suggest_goal(statement: str) -> str | None:
    """Best-guess goal for a plain-language statement, or None.

    Deliberately crude keyword matching. It exists so an agent can *offer* a
    goal and have the user confirm, not so the tool can silently decide what
    the user meant.
    """
    words = "".join(c.lower() if c.isalnum() else " " for c in statement).split()
    for w in words:
        if w in GOAL_SYNONYMS:
            return GOAL_SYNONYMS[w]
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
