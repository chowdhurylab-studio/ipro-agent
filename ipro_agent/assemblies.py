"""Designing one sequence across several binding assemblies.

IPRO's rule, stated in the Suite paper:

    the optimal combinations of rotamers are selected sequentially for each BA
    whereas **residue identity remains the same for all BAs**

so a design problem may put the same sequence into several structural contexts
at once -- bound to C12 and bound to C14, say -- each free to repack, with
acceptance requiring that *all* of them pass.

How this is implemented
-----------------------
Rather than add a third index to every variable in
:mod:`ipro_agent.milp` (and so a third encoding of the same program to keep in
step -- the mistake that produced defect #1 in docs/DESIGN.md §7), a
multi-assembly problem is rewritten as a *single* larger single-assembly
problem:

* each (assembly, design position) pair becomes one **virtual position**,
  written ``"assembly::A115"``. Constraint (2), "exactly one rotamer per
  position", then already says "one rotamer per position **per assembly**",
  which is precisely IPRO's per-BA packing.
* pair terms exist only *within* an assembly, since rotamers in different
  assemblies do not interact. Cross-assembly blocks are simply never built.
* identity is tied across assemblies by one extra linear family, for each
  design position ``i``, each non-reference assembly ``h``, and each permitted
  amino acid ``a``:

      sum_{r : aa(r) = a} y[ref::i, r]  -  sum_{r : aa(r) = a} y[h::i, r]  =  0

  If the rotamer chosen at ``i`` in the reference assembly has identity ``a``,
  the left sum is 1, so the right sum must be 1 too and ``h`` must also choose
  an ``a`` rotamer. This is the published coupling ``Y[i,a] = sum_r X[i,r,h]``
  with the identity variable ``Y`` eliminated, so it needs **no new variables**.

* sequence-level user constraints (mutation counts, residue-class cardinality,
  require/forbid, integer cuts) are scoped to the reference assembly's virtual
  positions. Counting them in every assembly would multiply every budget by the
  number of assemblies; because identity is coupled, counting once is exact.

What is *not* supported
-----------------------
Every assembly shares the design molecule's **backbone**. IPRO allows each BA
its own structure, which matters once backbone perturbation is in play. Here
the assemblies differ in which partner molecules are present -- enough for the
specificity and multi-partner cases, not enough to design across genuinely
different conformations. Stated in docs/DESIGN.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import Assembly
from .milp import EnergyTables

log = logging.getLogger(__name__)

#: Separator for virtual position labels. Chain ids are alphanumeric and
#: residue numbers are digits, so this cannot collide with a real label.
SEP = "::"


def vpos(assembly: str, label: str) -> str:
    if SEP in assembly or assembly.endswith(":"):
        # Otherwise split_vpos cuts at the wrong colon and the whole assembly
        # silently vanishes from by_assembly -- its rotamers are never scored.
        raise ValueError(
            f"assembly name {assembly!r} may not contain {SEP!r} or end in ':': "
            f"the name is joined to a position label with {SEP!r} to form the "
            f"virtual positions the combined program is written over"
        )
    return f"{assembly}{SEP}{label}"


def split_vpos(virtual: str) -> tuple[str, str]:
    """``"with_C12::A115"`` -> ``("with_C12", "A115")``."""
    assembly, sep, label = virtual.partition(SEP)
    if not sep:
        return "", virtual
    return assembly, label


@dataclass
class CombinedTables:
    """One single-assembly problem standing in for the multi-assembly one."""

    tables: EnergyTables
    #: Pairs of virtual positions whose identity must agree.
    identity_links: list[tuple[str, str]] = field(default_factory=list)
    #: Virtual positions that sequence-level constraints are counted over.
    sequence_positions: set[str] = field(default_factory=set)
    #: Assembly name -> its virtual positions, for reporting and scoring.
    by_assembly: dict[str, list[str]] = field(default_factory=dict)
    reference: str = ""

    @property
    def n_assemblies(self) -> int:
        return len(self.by_assembly)


def combine(
    per_assembly: dict[str, EnergyTables],
    assemblies: list[Assembly],
    reference: str | None = None,
) -> CombinedTables:
    """Fold per-assembly tables into one program over virtual positions.

    ``per_assembly`` maps an assembly name to tables built against *that*
    assembly's template (its own partner molecules). The rotamer lists must
    agree across assemblies for each design position, since the backbone is
    shared; a mismatch is a programming error and is rejected.
    """
    if not per_assembly:
        raise ValueError("no assemblies to combine")

    names = [a.name for a in assemblies if a.name in per_assembly]
    if not names:
        raise ValueError("none of the assemblies has tables")
    ref = reference or names[0]
    if ref not in per_assembly:
        raise ValueError(f"reference assembly {ref!r} has no tables")
    weight = {a.name: float(a.weight) for a in assemblies}

    # Single assembly: hand back its tables untouched, so the single-complex
    # case is byte-for-byte the problem it was before assemblies existed.
    if len(names) == 1:
        only = names[0]
        t = per_assembly[only]
        return CombinedTables(
            tables=t,
            identity_links=[],
            sequence_positions=set(t.positions),
            by_assembly={only: list(t.positions)},
            reference=only,
        )

    positions_ref = list(per_assembly[ref].positions)
    for h in names:
        if list(per_assembly[h].positions) != positions_ref:
            raise ValueError(
                f"assembly {h!r} has design positions {per_assembly[h].positions} "
                f"but the reference {ref!r} has {positions_ref}; all assemblies must "
                "share the same design positions"
            )
        for p in positions_ref:
            a_rots = per_assembly[h].rotamers[p]
            b_rots = per_assembly[ref].rotamers[p]
            if len(a_rots) != len(b_rots) or any(
                x.aa != y.aa or x.state != y.state for x, y in zip(a_rots, b_rots)
            ):
                raise ValueError(
                    f"assembly {h!r} and {ref!r} disagree on the rotamer set at {p}. "
                    "Assemblies share the design molecule's backbone, so their "
                    "rotamers must be identical; this is an internal inconsistency."
                )

    combined = EnergyTables(positions=[], rotamers={})
    by_assembly: dict[str, list[str]] = {}

    for h in names:
        th = per_assembly[h]
        w = weight.get(h, 1.0)
        vlist: list[str] = []
        for p in th.positions:
            v = vpos(h, p)
            vlist.append(v)
            combined.positions.append(v)
            combined.rotamers[v] = th.rotamers[p]
            combined.pruned_per_position[v] = th.pruned_per_position.get(p, 0)
            # Carry the identities the prune removed, too. Only the *count*
            # was copied, so a combined run reported a clean design space
            # while an identity the spec allows had been deleted from it --
            # which is precisely the §7k blocker, where the prune removed
            # PoreDesigner's published V18W and nothing said so.
            dropped = getattr(th, "dropped_identities", None) or {}
            if dropped.get(p):
                combined.dropped_identities[v] = list(dropped[p])
            for i in range(len(th.rotamers[p])):
                combined.self_energy[(v, i)] = w * th.self_energy.get((p, i), 0.0)
        for (pi, ri, pj, rj), e in th.pair_energy.items():
            combined.pair_energy[(vpos(h, pi), ri, vpos(h, pj), rj)] = w * e
        by_assembly[h] = vlist

    # Tie identity: reference against each other assembly, per position.
    links = [(vpos(ref, p), vpos(h, p)) for p in positions_ref for h in names if h != ref]

    combined.stats = {
        "assemblies": len(names),
        "virtual_positions": len(combined.positions),
        "rotamers_kept": sum(len(v) for v in combined.rotamers.values()),
        "pair_terms_kept": len(combined.pair_energy),
        "identity_links": len(links),
    }
    log.info(
        "combined %d assemblies into %d virtual positions, %d identity links",
        len(names), len(combined.positions), len(links),
    )
    return CombinedTables(
        tables=combined,
        identity_links=links,
        sequence_positions={vpos(ref, p) for p in positions_ref},
        by_assembly=by_assembly,
        reference=ref,
    )


def choices_for(
    combined: CombinedTables, assembly: str, choices: dict[str, int]
) -> dict[str, int]:
    """Project a combined solution onto one assembly's real position labels.

    With a single assembly the combined positions are not virtual, so
    ``split_vpos`` returns no assembly name; those belong to the only
    assembly there is. Returning ``{}`` for that case made this silently
    useless outside multi-assembly runs.
    """
    out: dict[str, int] = {}
    for v, idx in choices.items():
        h, label = split_vpos(v)
        if h == assembly or (not h and assembly == combined.reference):
            out[label] = idx
    return out


def identities_agree(combined: CombinedTables, choices: dict[str, int]) -> list[str]:
    """Positions where the assemblies disagree on identity -- must be empty.

    Used as a check rather than as logic: the coupling constraints make this
    impossible, so a non-empty result means the constraints were not applied.
    """
    t = combined.tables
    bad: list[str] = []
    seen: dict[str, dict[str, str]] = {}
    for v, idx in choices.items():
        h, label = split_vpos(v)
        seen.setdefault(label, {})[h] = t.rotamers[v][idx].aa
    for label, by_h in seen.items():
        if len(set(by_h.values())) > 1:
            bad.append(f"{label}: " + ", ".join(f"{h}={aa}" for h, aa in sorted(by_h.items())))
    return bad
