# Design document: what came from where

This repo is an **independent reimplementation** of the published IPRO procedure.
It was written from the papers and from reading the released source for
behaviour the papers leave ambiguous; no upstream code is reproduced.

This document exists so that any behaviour here can be traced to its source, and
so that every place this implementation *differs* from the published method is
written down rather than discovered later. Read it before reporting a number
from this tool as an IPRO result.

---

## 1. Capability → source

| Capability | Where it is | Source |
|---|---|---|
| Iteration loop: perturb → optimal rotamers → redock → relax → score → accept/reject → cut | `loop.py` | IPRO Suite (JCC 2015), §Methods; order confirmed against the released source's `Start → Backbone_Perturbation → Optimal_Rotamers → Docking → Relaxation → Calculate_Energy → End_Iteration` |
| Rotamer-selection MILP: objective, assignment constraint, bilinear linearization | `milp.py` | Saraf et al. (Biophys J 2006) Eqs 3–10; reproduced in full in the PoreDesigner SI (CC-BY); variable declarations confirmed in `ROTAMERS.py`/`GAMS.py` |
| Generalized dead-end elimination (self-energy pruning) | `milp.build_tables` | Biophys J 2006, `X_ir = 0 if E_rs(i,r) > δ_i` |
| Integer ("no good") cuts — never revisit a design | `milp.RotamerMILP.visited` | IPRO Suite constraint (10); `≤ len−1` cut form in the source |
| Mutation-count limit | `Constraint("max_mutations")` | IPRO Suite constraint (5) |
| Permitted-amino-acid enforcement per position | `Constraint("require"/"forbid")`, and `DesignPosition.allowed` | IPRO Suite constraint (7) |
| Stability energy cap | `Constraint("max_energy")` | Biophys J 2006 stability constraint |
| Residue-class cardinality (≥ 50% long side chains) | `Constraint("min_class")` | PoreDesigner (Nat Commun 2018) MILP constraint 4 |
| Interaction energy `IE = E(complex) − E(design) − E(partner)` | `loop.IPROLoop._context_for` | IPRO source's `Calculate_Energy`; same definition in ACS Catal 2017 |
| Metropolis accept/reject at T = 3640 K | `goals.metropolis_accept` | IPRO Suite: the interaction energy goes into a Boltzmann factor at **3640 K**, "which retains 25% of designs within 10 kcal mol⁻¹ of the best design"; the released source implements the algebraically identical `window = −R·T·ln(U)` form |
| Greedy opening phase (no thermal acceptance early) | `LoopSettings.greedy_fraction` | IPRO source disables annealing for the first 10% of iterations |
| Hard rejection of large worsenings | `LoopSettings.hard_reject_*` | IPRO source rejects a worsening beyond `max(40 kcal/mol, 5%)` |
| Design ensemble rather than a single answer | `RunResult.ensemble` | IPRO Suite; source default of 10 ensembles |
| **Goal: affinity** — minimize `IE(target)` | `goals.BindTighter` | IPRO (Biophys J 2006); IPRO+/− (Structure 2020) |
| **Goal: specificity** — minimize `IE(target) − w·IE(anti-target)` | `goals.SwitchSpecificity` | ACS Catal 2017's ranking metric `ΔIE = IE_short − IE_long`; the 2006 AraC case's decoy ligand (D- vs L-arabinose) |
| **Goal: channel** — probe interaction, direction as a parameter | `goals.ReshapeChannel` | PoreDesigner (Nat Commun 2018), which *maximized* `IE` with a single-file water wire to occlude the lumen |
| **Goal: catalysis** — `K_M`, `k_cat/K_M`, `k_cat` | `goals.ImproveCatalysis` | OptZyme (PLoS ONE 2013) Eq 7 (`IE_S = RT ln K_M`), Eq 17 (`ΔIE_TSA = −RT[ln(k_cat/K_M) − ln(k_cat^WT/K_M^WT)]`), Eq 18 |
| **Goal: stability**, optionally constrained by interaction | `goals.Stabilize` | IPRO+/− (Structure 2020): `min E_complex s.t. IE ≥ 0.75·IE_WT` |
| Design positions proposed by distance to a partner | `cli.cmd_positions` | the distance-cutoff rule used throughout the case studies |
| Terminal insertions and deletions | `indel.py`, applied in `IPROLoop.prepare` | IPRO+/− / Protein-InDelMaker provide the general case; only the terminal case is exact without loop closure |
| Channel bottleneck measurement, and the hard filter on it | `pore.py`, `goals.ReshapeChannel.filter_reject` | PoreDesigner's acceptance filter: slices every 0.5 A through the constriction, largest inscribed figure per slice, discard a design outside the requested window |
| Backbone perturbation (1-5 consecutive residues, phi/psi ~ N(0, 1.5 deg) truncated to +/-5 deg) | `perturb.py`, `loop.perturb_backbone` | IPRO Suite step 2 and the released source's `random.gauss(0, 1.5)` rejection-truncated to +/-5 deg; the 2006 paper reports sigma 1.65 deg |
| Designed coordinates written out (best, and per iteration) | `output.py` | Deliverable in its own right: a design that exists only as a sequence and a score cannot be inspected or taken to the bench |
| Re-placing partner molecules each Nth iteration | `docking.py` (local MC), `dock.py` (Vina), chosen by `backends.dock` | IPRO Suite step 3: "a local, rigid-body redocking of the TMs ... uses the same energy functions as the rotamer selection step", 500 moves, sigma 0.2 A / 2.0 deg, annealed 3640 K to 2190 K |
| Multiple binding assemblies (one sequence, per-assembly packing, conjunctive acceptance) | `assemblies.py`, `config.Assembly` | IPRO Suite: "rotamers are selected sequentially for each BA whereas residue identity remains the same for all BAs", and "all BAs must pass the selection criteria" |
| Atom-position restraints (fix, tether, distance, dihedral) | `restraints.py`, `Constraint` vetoes in `milp.py` | IPRO Suite: four kinds of restraint "rigorously enforced throughout all functions"; the ±0.2 Å catalytic-distance form is from ACS Catal 2017 |
| Design/target/anti-target vocabulary | `config.Role` | IPRO's own **DM** / **TM** / **DP** input vocabulary (design molecule, target molecule, design position); `anti_target` is this repo's addition |
| Rotamer enumeration and mutagenesis | `rotamers.py`, `mutate.py`, `scripts/*.py` | PyMOL's mutagenesis wizard — **a substitution**, see §3 |

### 1a. Position selection, as actually published

`ipro-agent positions` proposes design positions by distance to a partner
molecule. That is one step of the published procedure, not all of it. The
thioesterase work (Grisewood et al., *ACS Catal* 2017, SI Table S4) sorts
candidate residues by distance from the ligand and then takes **the eight
closest that are not excluded**, where a residue is excluded if any of the
following holds:

| Exclusion | Can this tool detect it? |
|---|---|
| part of the catalytic triad | no — needs the user |
| forms the oxyanion hole | no |
| shown important for function by mutagenesis | no |
| within 4.5 Å of the reactive atom (the thioester sulfur) | **yes**, given the atom |
| Cα more than 0.75 Å closer than Cβ to the ligand's discriminating atom (side chain points away) | **yes**, given the atom |
| ≥40% conservation in the family | no — needs an alignment |
| the family alignment has a gap at that position | no |
| in one of the protein's flexible loops | no |
| an earlier round showed mutating it abolished activity | no |

Two of the nine are geometric and computable; the rest require knowledge this
tool does not have. That is why `positions` is documented as a *proposal* and
why `.claude/skills/ipro-agent/SKILL.md` makes the agent walk the user through the list rather than
accepting a distance cutoff as the answer. Note also that the sorting metric
itself changed between rounds in that paper (rounds 1–2 used a different
distance definition from rounds 3–4), so even the computable part is a choice,
not a constant.

The same SI reports the correlation between computed and experimentally
estimated ΔΔG (Figure S5), measured against myristoyl-ACP for round 3 and
lauroyl-ACP for round 4 — the basis for the caveat in §5 that these objectives
rank better than they predict magnitudes.

### 1b. IPRO's input language, and what this tool's spec covers

The Suite paper defines the vocabulary verbatim:

| Term | Definition (paper's wording, condensed) | Here |
|---|---|---|
| **DM** — design molecule | a protein permitted to mutate | `role: design` |
| **TM** — target molecule | what it binds: a protein, peptide or small ligand | `role: target` |
| **DP** — design position | the residues within a DM allowed to mutate | `positions:` |
| **BA** — binding assembly | the totality of molecules interacting with a particular binding topology; multiple DMs and TMs may take part | **implemented** — see §1d (`assemblies:`). One shared backbone across assemblies is the remaining limit. |

It also settles a definition this implementation depends on. IPRO deliberately
does **not** use binding energy:

```
binding energy     =  G(BA,min) − G(DMs,min) − G(TMs,min)
interaction energy =  G(BA,min) − G(DMs)     − G(TMs)      <- IPRO uses this
```

The components are *not* separately re-minimized, "as it is less dependent on
the force field used and requires fewer calculations". That is exactly what
`IPROLoop._context_for` computes — the cross-term on the complex conformation —
so the quantity is right, and the reason for preferring it is now sourced
rather than inferred.

**The paper's full list of required user input**, against what `DesignSpec`
accepts:

| IPRO requires | Here |
|---|---|
| user's name, experiment name | `name` (no user field; provenance is the git history) |
| the DMs and TMs | `molecules` |
| the DPs | `positions` |
| the kinds of amino acid mutations permitted for each DP | `positions[].allowed` |
| the temperature for the Metropolis criterion | `loop.temperature_k` |
| number of design iterations | `loop.iterations` |
| how to run docking | `backends.dock`, and the separate `dock` command |
| how to repack rotamers | implicit (the MILP is always used) |
| how to use the force field | `backends.energy` |
| how to use implicit solvation | **not modelled at all** by the built-in energy — it has no desolvation term (see §4). PyRosetta's score function brings one. |
| the BAs | `assemblies:` — see §1d |
| whether to share Metropolis-retained results between processors | **not applicable** (single process) |
| how to do ensemble structure refinements | partial — `loop.ensemble_size` keeps an ensemble but does not refine it |

Two consequences worth stating plainly. First, IPRO supports **multiple binding
assemblies** and requires that *all* of them pass the acceptance criterion for
an iteration to be retained; this tool models one assembly, so a design that
must satisfy several binding topologies at once cannot be expressed. Second,
IPRO offers **four kinds of atom-position restraint, "rigorously enforced
throughout all functions"** — atoms fixed in place, atoms held near their input
positions, a distance restraint between two atoms, and a dihedral restraint on
four atoms. **All four are implemented here** (`restraints:`); §1c gives the
mechanism. That matters more than it sounds: the distance restraint is what the
thioesterase work used to preserve catalytic geometry while redesigning
specificity (catalytic distances held to ±0.2 Å), so "redesign this site but
keep the catalysis intact" is expressible. The remaining limits are stated in
§1c: a restraint spanning three or more design positions is reported rather
than enforced as a veto, and a restraint that cannot name which molecule it
means is refused rather than guessed.

### 1c. How restraints are enforced

IPRO offers four kinds of atom-position restraint "rigorously enforced
throughout all functions". All four are expressed here, and the mechanism is
worth stating because it is exact rather than approximate.

During rotamer selection the only coordinates that change are side-chain atoms
at design positions. A restraint is therefore a **veto** on rotamer choices,
classified by how many design positions its atoms depend on:

| Design positions involved | Becomes | MILP form |
|---|---|---|
| 0 | nothing to decide — measured once and reported | — |
| 1 | a rotamer veto | `y[i,r] = 0` |
| 2 | a pair veto | `y[i,r] + y[j,s] ≤ 1` |
| ≥3 | **not expressible**; reported, never dropped silently | — |

The vetoed assignments are *precisely* those that violate the restraint, so the
MILP optimum is the best design that complies. Verified against constrained
brute force for both the one-position and two-position cases, on both solver
paths (`tests/test_ipro.py::test_restraint_*`).

`y[i,r] + y[j,s] ≤ 1` is used rather than `w[i,r,j,s] = 0` because a negligible
pair block has no `w` variables at all (see §7 defect 1), while `y` always
exists.

Three things learned while implementing it, all of which the tool now reports
rather than leaving to be discovered:

1. **A restraint on Cβ is nearly vacuous.** Cβ's position is pinned by the
   backbone — across 11 rotamers at one position it moved by 0.01 Å. Restrain a
   side-chain *tip* atom (the same position's Cζ2 ranged over 3.8–9.0 Å).
2. **Naming a side-chain atom silently narrows the residue types.** A restraint
   on `A20:CG` cannot be satisfied by ALA or GLY, so those rotamers are vetoed.
   `ipro-agent validate` says so before the run.
3. **Dead-end elimination runs before restraints**, so the rotamers that would
   have satisfied a restraint may already have been pruned for clashing. When
   that leaves nothing, the error says how many were pruned and names
   `goal_options.max_self_energy` — and points out that a restraint satisfiable
   only by a clashing rotamer is itself the finding.

A one-sided bound (`minimum`/`maximum`) on a **dihedral** is refused: it has no
unambiguous meaning on a circular quantity. Dihedrals take `target` with
`tolerance`, compared with proper wraparound.

A `fix` or `tether` on a **backbone** atom cannot be enforced by rotamer choice.
In `full` mode it is handed to the relaxation backend; in `fixed-backbone` mode
it is already satisfied, and the run says so.

### 1d. How binding assemblies are implemented

IPRO's rule is that one sequence is packed **separately in each assembly**, and
that every assembly must pass acceptance. Rather than add a third index to
every variable -- and so a fourth encoding of the same program to keep in step,
which is how §7's defect 1 happened -- a multi-assembly problem is rewritten as
a single larger one:

* each (assembly, design position) pair becomes a **virtual position**
  `"with_C12::A115"`. Constraint (2) then already reads "one rotamer per
  position **per assembly**", which is the per-BA packing, for free.
* pair terms exist only within an assembly; cross-assembly blocks are never
  built.
* identity is tied across assemblies by one extra linear family -- for each
  position `i`, each non-reference assembly `h`, and each permitted amino acid
  `a`:

      sum_{r : aa(r)=a} y[ref::i, r]  -  sum_{r : aa(r)=a} y[h::i, r]  =  0

  This is the published coupling `Y[i,a] = sum_r X[i,r,h]` with the identity
  variable `Y` eliminated, so it needs **no new variables**.
* sequence-level constraints (mutation counts, residue-class cardinality,
  require/forbid, integer cuts) are counted over the reference assembly only.
  Counting them in every assembly would silently multiply every budget by the
  number of assemblies — verified not to
  (`test_assembly_does_not_multiply_sequence_budgets`).

**Acceptance.** The objective test on the combined score runs as before; with
more than one assembly, each gating assembly must *additionally* pass its own
Metropolis test on its own energy change. That is the paper's "all BAs must
pass". Single-assembly runs take exactly the path they always did, so nothing
about the existing behaviour changed — the whole test suite passes unaltered.

**Verified:** identity is shared while packing differs per assembly, and the
optimum equals a brute force that chooses an identity per position and then
packs each assembly independently (`test_assembly_*`).

**What is not supported.** Every assembly shares the design molecule's
**backbone**; they differ only in which partner molecules are present. IPRO
allows each BA its own structure, which matters once backbone perturbation is in
play. Also note a practical limit observed while testing: on a real buried site,
the partner's contribution to a rotamer's energy can be much smaller than the
intrinsic packing differences between rotamers, so the per-assembly optimum is
often the same rotamer in every assembly. The capability is exact; whether it
*changes the answer* depends on the partner mattering enough.

**Selecting against a partner** is still the `switch_specificity` goal's job,
not an assembly's. Assembly weights are positive and each assembly's packing is
minimized, because that is what the protein will actually do when it meets that
partner; the trade-off between partners lives in the objective and in
acceptance.

### 1e. Placing the partners: two different operations

`backends.dock` chooses, and the two options are **not** two qualities of the
same thing:

| | `builtin` | `vina` |
|---|---|---|
| what it is | IPRO's own step: local rigid-body Monte Carlo | AutoDock Vina: global search in a box |
| search scope | the six rigid-body degrees of freedom, near the current pose | the whole box, any orientation |
| scoring | **the same energy as rotamer selection** | Vina's own scoring function |
| finds a pose from nothing? | no — it refines one | yes, that is what it is for |
| measured cost | **0.15–0.40 s** per 500 moves (327-atom receptor, 3–60 atom partner; ~0.3–0.8 ms/move) | seconds to minutes per ligand at exhaustiveness 8, plus receptor/ligand prep and file I/O per call |
| reproducible | yes, from `loop.random_seed` | with `--seed`, subject to Vina's own determinism |
| external dependency | none | a Vina binary, and ideally Meeko for ligand prep |

`auto` prefers Vina when it is installed and falls back to `builtin`, with a
note saying so — the same precedence as `energy: auto` preferring PyRosetta.
An explicit `dock: vina` with no Vina **raises**; it never quietly degrades to
the cheaper option, because the two answer different questions.

**Vina's affinity never enters the objective.** The pose is taken and the energy
that drives the loop stays the backend's. Otherwise the objective would depend
on which docking program happened to be installed and scores would not be
comparable between runs. Vina's number is reported alongside, labelled as being
on its own scale.

**Why re-placing partners is affordable at all.** Only the **self** energies
depend on where the partners are — a pair energy is rotamer-against-rotamer and
the partners do not appear in it. So a redock invalidates
O(positions x rotamers) terms rather than the O(n^2) pair block, and
`_refresh_self_energies` updates only those. The rotamer *set* is left as it was
pruned against the input pose, so the design space does not drift between
iterations. There is a test that refreshing with unmoved partners changes
nothing, which also proves the two template constructions agree.

**A caveat on Vina as the redock.** The published step is a *local refinement*.
Using a global search in its place changes the protocol: each redock re-places
the ligand rather than nudging it, so the trajectory is no longer a local walk.
That may be what you want — particularly if mutations have moved the pocket
enough that the old pose is meaningless — but it is a different protocol, and a
Vina pose is only applied when its atoms correspond one-to-one with the input
ligand's (otherwise the transform cannot be fitted, and the run says so rather
than moving the ligand somewhere nobody chose).

### 1f. Three modes, and why there are three rather than two

`loop.perturb_backbone` decides whether the backbone moves. Combined with
whether the energy backend can relax, that gives three modes, and the run
reports which one it was in (`state.json`, the run summary, and a REMARK in
every structure it writes):

| mode | backbone | relaxation | needs |
|---|---|---|---|
| `fixed-backbone` | never moves | — | nothing |
| `backbone-mc` | perturbed every iteration, strained moves rejected | **none** | nothing |
| `full` | perturbed every iteration | restrained minimization | PyRosetta |

An earlier version of this document claimed perturbation was skipped because
"perturbing a backbone without minimizing afterwards only manufactures
clashes". That is half true and was the wrong conclusion: a perturbation
followed by an optimal repack and a Metropolis test *is* a coherent backbone
Monte Carlo -- strained moves simply get rejected -- and it is far closer to
IPRO than never moving the backbone at all. It is still not the published
protocol, which minimizes after perturbing, so it is named separately rather
than called `full`.

**What perturbation costs.** Moving the backbone invalidates the rotamers
themselves, since a rotamer's coordinates are built on it -- so both the self
*and* the pair energies have to be rebuilt, unlike a redock which only
invalidates the self energies. Each perturbed iteration therefore pays one
PyMOL rotamer re-harvest plus a full table rebuild: measured at about
**1.3 s/iteration** on a 46-residue chain with two design positions. That is
why it is opt-in.

**A rejected perturbation is undone.** Otherwise the backbone would random-walk
regardless of the energy, which would make the Metropolis test meaningless. The
loop snapshots the structure, rotamers, tables and restraint plan before
perturbing and restores them on rejection. Verified end to end: in an 8-iteration
run, iteration 5 was rejected at 0.72 A CA-RMSD from the input and iteration 6
started from 0.13 A, i.e. from the last *accepted* state rather than the
rejected one.

**Rotamer indices are not comparable across backbones.** A rebuild can prune
differently, so an index from an earlier iteration may name a different residue
or none at all. The best design is therefore tracked by *identity*, its
structure is written at the moment it is found, and a carried-over assignment is
re-expressed against the rebuilt rotamer set by identity
(`IPROLoop._rebase_choice`).

### 1g. What the run leaves on disk

`best/design.pdb` holds the complex that produced the reported score: the design
molecule with the **chosen rotamer** at each design position (not merely the
right residue type, which would not reproduce the energy), plus every partner at
its current placement. With several assemblies there is one file per assembly,
since the packing differs. `loop.keep_structures: true` additionally writes
every accepted iteration to `structures/iter####.pdb`, and each iteration record
in `iterations.jsonl` names its file.

Each file carries REMARK provenance -- run, goal, mode, backend, score,
mutations -- so a stray PDB can still be traced back. Chain ids are kept
distinct: a partner that reuses the design molecule's chain id is remapped and
the mapping is reported, because merging them would describe a different
molecule than the one scored.

### 1h. Measuring the channel

PoreDesigner's filter is geometric: a redesign is kept only if the bottleneck
falls inside a requested window, decided *before* the interaction energy
matters. That is now implemented.

**What is measured.** Slices every 0.5 A along the channel axis; in each, the
largest circle that overlaps no atom's van der Waals sphere (Bondi radii); the
bottleneck is the smallest of those. The paper fits an **ellipse** and reports
its major axis, so the numbers are not the same quantity: an inscribed circle is
bounded by the *narrow* direction, so for a slit-like lumen this reports
something closer to the minor axis. Comparable across runs of this tool, not
identical to the paper's.

**Where the axis comes from.** This matters more than the measurement. A
barrel's axis is *not* its largest-variance direction, so guessing from the
protein fails outright: on OmpF the whole-protein principal axis measures the
eyelet at **1.4 A**. The default instead fits a plane through the **design
positions**, which line the constriction, and takes its normal. On 2OMF chain A
the 25 published positions give an eyelet of **6.24 A** (re-measured
2026-11-18; an earlier version of this section recorded 8.1 A, which does not
reproduce), against the published 11 x 7 A — the right ballpark for a circle
bounded by a 7 A minor axis, from an independent reimplementation.
`goal_options.pore_axis: ["A18:CA", "A119:CA"]` overrides it with two atoms
spanning the channel, and the run warns when the lining residues are too far
from coplanar for the fitted axis to mean anything.

**The measurement had a bug worth recording**, because the symptom was subtle.
The first version let the circle's centre range freely within a wide radius of
the axis. In unbounded space the largest circle overlapping no atom is out in
the solvent, so the search walked off the protein and reported the distance to
the bulk — placing the bottleneck one slice away from the real one while looking
entirely plausible. The centre is now hard-constrained to
`max_centre_offset` of the axis, and that constraint *is* the measurement: it is
what makes the question "how wide is the lumen" rather than "how far is the
nearest atom". There is a regression test.

**A geometric objective only bites if the design positions ring the lumen** —
and when they do not, the number it reports is not a pore measurement at all.
Re-measured 2026-11-18 (the figures in earlier versions of this section were not
reproducible; these are from `ipro-agent score` on the shipped specs):

| spec | design positions | axis clearance | bottleneck | radius score |
|---|---|---|---|---|
| `spec_full.yaml` | 25, ringing the constriction | 1.30 A | **6.24 A** | 1.257 |
| `spec_smoke.yaml` | 4, all on one wall | **0.30 A** | 1.53 A | 1.528 |

"Axis clearance" is how far the fitted axis origin sits from the nearest van der
Waals surface. A water needs 1.4 A. The smoke spec's four positions put that
origin **inside the wall**, so its "1.53 A bottleneck" is the sliver beside the
wall — and it silently *passes* the published sub-4 A filter. §7d records this as
a confirmed defect left unfixed, with the attempted fix and why it was reverted.
Give `goal_options.pore_axis` explicitly whenever the design positions do not
surround the channel.

**Cost.** 127 ms per measurement over an 8 A constriction window on a
2627-atom structure (555 ms over the whole barrel), and only goals that declare
`wants_geometry` pay it.

### 1i. Insertions and deletions: why only at a terminus

Changing a chain's length in the **middle** leaves two cut ends that must be
rejoined, and doing that correctly is loop closure. There is no honest
shortcut: an internal edit without closure produces a chain with a broken
peptide bond, which still looks like a protein and scores like nonsense. So
internal indels are **refused**, naming the specific reason and pointing at
Protein-InDelMaker.

At a **terminus** there is no second cut end, so the edit is exact and is
implemented:

* a **deletion** removes residues and leaves every remaining atom untouched;
* an **insertion** builds residues from ideal internal coordinates
  (N–CA 1.458 A, CA–C 1.525 A, peptide C–N 1.329 A, N–CA–C 111.2 deg,
  CA–C–N 116.2 deg, C–N–CA 121.7 deg, omega 180 deg) at chosen phi/psi,
  extended by default.

**The invariant that matters most: an indel never renumbers an existing
residue.** Design positions are named by residue number, so a renumbering would
silently point them at different residues and the run would optimize something
nobody asked for. C-terminal insertions therefore count upward from the last
residue; N-terminal insertions count **downward** from the first, so a chain
starting at 1 gains residues 0, −1, … (negative numbers are legal in a PDB and
the position parser accepts them); deletions simply leave gaps.

**Inserted residues are glycine, and that is the point.** The inserted position
becomes an ordinary design position, so the MILP chooses its identity and PyMOL
builds the side chain — the same path as any other position. It also mirrors
IPRO, which mutates side chains to glycine before perturbing a backbone.

Verified: `place_atom` round-trips its bond, angle and dihedral to machine
precision over 200 random reference frames; appended and prepended residues have
ideal trans backbone geometry; existing numbering is preserved in all four
cases; and end to end, appending two residues and listing them as design
positions yields a 48-residue chain with designed identities at 47 and 48,
properly bonded.

One bug worth recording, since it only appeared with more than one residue: the
chain was re-sorted *after* the insertion loop rather than inside it, so the
second pass still saw the original first residue and tried to reuse its number.

### The one concept added here

`role: anti_target` is **not** in IPRO. It is the minimal addition that lets a
single objective express *specificity* (bind A, not B) rather than plain
affinity. The published work achieved the same thing case by case — a decoy
ligand in the 2006 AraC study, a `ΔIE` ranking metric computed after the fact in
ACS Catal 2017. Making it a molecule role means `switch_specificity` optimizes
the selectivity gap directly, inside the loop, instead of post-hoc.

---

## 2. Numeric parameters and their provenance

Every default in `LoopSettings` is either published or marked as this repo's
choice. Where two sources disagree, both are recorded and the chosen value is
stated.

| Parameter | Value here | Source | Note |
|---|---|---|---|
| `temperature_k` | 3640 K | IPRO Suite, confirmed in the paper itself | The paper states the Boltzmann factor uses 3640 K, "which retains 25% of designs within 10 kcal mol⁻¹ of the best design", and separately that docking anneals from 3640 K to 2190 K, retaining 25% and **10%** of positions within 10 kcal/mol. **Both calibrations were checked and both hold with `R = 1.987e-3 kcal/(mol·K)`:** `exp(−10/(R·3640)) = 0.2509` and `exp(−10/(R·2190)) = 0.1005`. A secondary source's `k = 0.333e-3` fits neither (it gives 0.0003 and 1e-6), so it was discarded. Checked in `tests/test_ipro.py`. |
| docking anneal | 3640 K → 2190 K, **implemented** | IPRO Suite | A constant cooling schedule inside the 500-move rigid-body docking loop, run by `backends.dock: builtin` (the default when Vina is absent) every `redock_every` iterations — see §1e and `loop.redock_temperature_end_k`. Still true, and the reason this row used to read "not implemented": the **outer** accept/reject temperature is constant, and only the docking sub-loop anneals. |
| `perturbation_sigma_deg` | 1.5 | IPRO Suite and source (`random.gauss(0, 1.5)`) | The 2006 paper reports 1.65°. The source value was chosen. |
| `perturbation_cap_deg` | 5.0 | both | Rejection-truncated, not clipped. |
| `perturbation_segment` | (1, 5) | source (`randint(0,4)`-based split) | 1–5 consecutive residues. |
| `design_per_iteration` | (1, 3) | IPRO Suite | 1–3 contiguous positions re-designed per iteration. A *different* quantity from the perturbation segment. |
| `perturbation_flank` | 5 | IPRO Suite | ±5 residues around the segment → an 11-residue window. |
| `neighbour_cutoff` | 4.5 Å | source | Packing/repacking shell. |
| `energy_cutoff` | 12 Å | Biophys J 2006 | vdW cutoff. The same paper uses 3 Å for H-bonds and 9 Å for solvation; this implementation does not separate them. |
| `redock_every` | 3 | IPRO Suite | Every third iteration. |
| `redock_moves` / sigmas | 500 / 0.2 Å / 2.0° | IPRO Suite | Rigid-body Monte-Carlo sub-moves. |
| `iterations` | 100 | **this repo's choice** | IPRO's coded default is 3000; the Suite paper quotes a 10,000 cap; the 2006 paper used 200–10,000. A small default makes a long run a deliberate act. |
| `max_rotamers` | 1300 | source | |
| `ensemble_size` | 10 | source | |
| `greedy_fraction` | 0.10 | source | Annealing disabled for the first 10% of iterations. |
| `hard_reject_kcal` / `_fraction` | 40 / 0.05 | source | |
| repulsion cap | 10 kcal/mol per atom pair, **both signs** | **this repo's choice** | The original clamps *objective coefficients* at 10000 kcal/mol. Same purpose: keep rotamer clashes from dominating the program. This was false until 2026-11-18: the bound applied to the Lennard-Jones term alone, while the Coulomb term ran unbounded as 1/r² down to the 0.3 Å clamp, so a buried counter-charge reached −451 kcal/mol for a single pair and was the most favourable thing the function could report (§7d #1). The **whole** pair term is now bounded to ±10. |
| LJ parameters | Amber parm94/99, element level | Cornell et al., *JACS* **117**:5179 (1995) | An approximation: per-element rather than per-atom-type. |
| reference energies | **none supplied** | — | A Rosetta-style correction, not an IPRO one: see the correction note in §4. No default is invented. |

---

## 3. Substitutions: this stack vs IPRO's

The original IPRO depends on **CHARMM c34b1** (licensed), **GAMS** or **CPLEX**,
prebuilt **Linux x86-64 C++ binaries**, an external rotamer library, and
**Python 2**. None of that runs on a current Mac, and the rotamer library is not
distributed with either release. Every substitution below is deliberate.

| Job | IPRO uses | This repo uses | Consequence |
|---|---|---|---|
| Energies for the MILP tables | CHARMM vdW + electrostatics + Lazaridis–Karplus solvation, softened LJ, `−ln(p)` rotamer-probability terms | PyRosetta, or a built-in screening function | **Energies are not comparable to the papers'.** Different force field, different units. |
| Backbone relaxation | CHARMM ABNR, ≤4000–5000 steps, dihedral restraints | PyRosetta FastRelax | Different minimizer; without PyRosetta there is no relaxation at all (see §4). |
| Rotamer library | backbone-dependent Dunbrack library, external, **not shipped with either release** | PyMOL's mutagenesis wizard library | Different rotamer sets, hence different optima. PyMOL's is not backbone-dependent in the same sense. |
| Mutagenesis | in-house, via CHARMM | PyMOL mutagenesis wizard | — |
| Docking | **in-house Monte-Carlo** (`docking.out`), 500 sub-moves | AutoDock Vina | Not expected to give identical poses. Vina was chosen because it is open and maintained; IPRO never used it. Vina 1.2.7 + Open Babel 3.2.1 were installed on 2026-11-18 and this path **executed for the first time**; it did not work until the PDB → PDBQT conversion was wired in (§7e). IPRO's own local rigid-body Monte Carlo is also implemented, as `backends.dock: builtin`, and is the closer match to the papers. |
| MILP solver | CPLEX via GAMS | CPLEX / Gurobi / CBC / HiGHS / GLPK, or an LP file | Same formulation, same optimum. |
| Language | Python 2 | Python ≥ 3.10 | — |

**Note:** this dependency stack (PyMOL, AutoDock Vina, PyRosetta) is this substitution set, not IPRO's own. It was adopted as
specified, and this table is the record of what that changes.

---

## 4. What is not implemented

| Capability | Status | Why, and what it would take |
|---|---|---|
| **Internal** insertions and deletions | Not implemented; refused with the specific reason. | Changing chain length *in the middle* needs loop closure. Protein-InDelMaker uses PyRosetta's GeneralizedKIC over a 6-residue window then Cartesian FastRelax (`repeats=5`); IPRO+/− reports an 11-residue CCD loop, ≤20 iterations, 3000 runs. **Terminal** indels need no closure and are implemented — see §1i. |
| **Rosetta-scored MILP pair tables** | `exact_pairs=True` raises. | Rosetta scores poses, not atom sets; a pose per rotamer pair is prohibitive. The screening terms build the table, as IPRO likewise uses a cheap evaluation for its combinatorial inner problem. |
| **Reference energies** | Mechanism provided, no values. | **Correction to an earlier claim in this document.** Two previous versions said IPRO carries a hardcoded 20-entry amino-acid reference-energy table and that its absence here was the tool's biggest practical limitation. Reading the released source directly, that is wrong on both counts. `SHARING.load_reference_Energies` loads *the best result so far* for the Metropolis comparison — run bookkeeping, not a residue table. The 20-entry tables in `STANDARDS.py` are naming and classification maps. What IPRO actually adds per rotamer is `-ln(p)` from `.prob` files in its rotamer library (`ROTAMERS.py:407`), a rotamer-probability penalty that is **0 for GLY and ALA** and therefore pushes *toward* small residues. So no per-residue-type unfolded-state term exists in IPRO at all. The `reference_energies` hook here is a **Rosetta-style** correction (ref2015 has a `ref` term), which the PyRosetta backend brings for free. The packing bias toward small residues is real, but it bites hardest when the objective is the packing itself (`stabilize`); IPRO optimizes an interaction energy with a target, where it bites far less. |
| Separate H-bond (3 Å) and solvation (9 Å) cutoffs | Single 12 Å cutoff. | Minor; recorded for completeness. |
| **Desolvation** | **Absent from the built-in energy.** | Correction to an earlier claim in this document, which described the built-in model as having a "solvation treatment" and mapped the paper's 9 Å solvation cutoff onto its 12 Å one. There is no such term: the built-in function is LJ + a bounded pair term + formal-charge Coulomb with a distance-dependent dielectric + a distance-only H-bond well, and nothing else. A buried polar or charged group therefore pays **no penalty for being buried**, which is the largest single omission relative to the CHARMM + Lazaridis–Karplus energy IPRO used, and it compounds the charge bookkeeping noted in §5. PyRosetta's ref2015 includes solvation — but only reaches the design through `exact_pairs=True`, which raises, so in practice the pair tables never have it (§7e). |

---

## 5. Honest limits of the objectives

Taken from the papers themselves, and worth repeating to anyone reading a score:

- Interaction energy of this kind reproduces the **rank and sign** of a change
  far better than its **magnitude**. IPRO+/− misses a 250-fold `K_m` shift;
  ACS Catal 2017 reports `ΔΔG` correlations of roughly r = 0.4–0.6.
- Experimental **hit rates are about 24–43%** across this family of methods.
- Accuracy **degrades with the number of insertions or deletions**.
- PoreDesigner publishes **no per-design energies**, names no MILP solver and no
  rotamer library, and drew its per-iteration mutation count at random.
- The two thioesterase papers use **different residue numbering for the same
  protein**, and present interaction-energy signs in opposite directions. This
  implementation normalizes signs so every goal is minimized; the papers do not.
- ACS Catal 2017 explicitly **does not model the transition state**, using
  catalytic-distance restraints to *preserve* rather than boost activity.
  Boosting catalysis is the least trustworthy of the goals here.

Three limits of the **built-in screening energy** specifically, measured during
the third review round (§7d) and left as they are because each is a modelling
choice rather than a defect — but each changes how a number should be read:

- **No desolvation term at all** (§4). A buried charge or polar group pays
  nothing for being buried. With no such penalty, charged residues at buried
  positions will look better than they are.
- **Formal charges are bookkeeping, not chemistry.** LYS keeps its whole +1 on
  NZ while ARG's +1 is split over three nitrogens, so two chemically equivalent
  salt bridges differ about sevenfold: one carboxylate oxygen against LYS NZ at
  2.80 Å scores **−4.087**, against ARG NH1 at the same distance **−0.557**, and
  whole side chains at better geometry still favour LYS (−6.174 vs −2.972). The
  split is defensible (guanidinium charge really is delocalized) and so is
  LYS's, but **the MILP will prefer LYS at any salt-bridge position** for
  reasons that are partly an artefact of this. HIS is left neutral throughout.
- **The 12 Å cutoff is a hard truncation, with no switching function.** A
  charged pair crossing it steps by 0.29 kcal/mol (ASP OD1 ··· LYS NZ:
  −0.288525 at 11.999 Å, 0.000000 at 12.001 Å) — about a fifth of the whole
  H-bond weight. The cutoff itself is the paper's (Biophys J 2006); the
  discontinuity matters most to the built-in docking backend, whose 500-move
  Monte Carlo is scored with this same function and so has steps in its
  landscape.

---

## 6. Sources, and what could not be obtained

### What this implementation was written from

Every capability in §1 traces to one of these. All are cited in full below.

| Source | What it supplied |
|---|---|
| Biophys J 2006 (PMC1459523) | the IPRO procedure itself: the iteration, the Metropolis test, the 12 Å cutoff |
| Biophys J 2007 (PMC1861794) | the specificity-switching case and its decoy-ligand treatment |
| J Comput Chem 2015 (10.1002/jcc.23796) | the IPRO Suite formulation — its equations (1), (2), (4) and (5) match docs/MILP.md; the DM/TM/DP/BA vocabulary in §1b; both temperature calibrations verified in §2 |
| PLoS ONE 2013 (OptZyme) | the three kinetic objectives and their equations 7, 17, 18 (§1, `goals.py`) |
| ACS Catal 2017 (PMC5784442) + its Supporting Information | the ΔIE selectivity metric, the catalytic-distance restraints, and the position-selection rule recorded in §1a |
| Nat Commun 2018 (PoreDesigner) + SI | the pore objective, the ≥50% long-residue constraint, and the sub-4 Å acceptance filter (§1h) |
| Structure 2020 (IPRO+/−) | the insertion/deletion formulation and its loop-closure requirement (§1i) |
| The released IPRO Suite source | **numeric defaults the papers leave ambiguous only** — the eight values marked "source" in §2's table, plus the variable declarations, coefficient clamp and cut forms noted in docs/MILP.md. No code was copied; see the note below. |
| Protein-InDelMaker | the published terminal-indel geometry this tool follows |

**On reading the released source.** It was read, and this document says so
wherever it mattered: §2 marks eight defaults as coming from the source rather
than a paper, and in one case (`perturbation_sigma_deg`) records that the
source's value was preferred over the paper's. That is why this repo is
described as an **independent reimplementation** and never as "clean-room" —
the latter means the implementers had no access to the original, which is not
what happened. What *is* true, and is what the MIT licence relies on, is that
**no upstream code is copied**: the implementation was written from the
formulations, and every value taken from the source is cited as such.

### Sources that could not be obtained, and what that leaves unresolved

| Missing | Why | Unresolved |
|---|---|---|
| **IPRO's rotamer library** | not in either release; referenced as an external path | why PyMOL's library is used instead, with different rotamer sets and therefore different optima (§3) |
| **A public IPRO+/− release** | the upstream repository is empty | indels are available publicly only as Protein-InDelMaker, so §1i follows that geometry |
| **PoreDesigner's water-wire probe** | never deposited; the SI publishes only fitted helix parameters | the example case cannot express the published pore objective, and substitutes a geometric one (§1h) |

The open-access 2018 Penn State dissertation by the Suite paper's second author
(`etda.libraries.psu.edu/catalog/15087mjg5185`) is the highest-value free source
still unread, if any of the above matters to you.

### Citations

1. Saraf, Moore, Goodey, Cao, Benkovic, Maranas. "IPRO: an iterative computational protein library redesign and optimization procedure." *Biophys J* **90**(11):4167–4180 (2006). PMC1459523.
2. Fazelinia, Cirino, Maranas. *Biophys J* **92**(6):2120–2130 (2007). PMC1861794.
3. Pantazes, Grisewood, Li, Gifford, Maranas. "The Iterative Protein Redesign and Optimization (IPRO) suite of programs." *J Comput Chem* **36**(4):251–263 (2015). doi:10.1002/jcc.23796.
4. Grisewood et al. "OptZyme: computational enzyme redesign using transition state analogues." *PLoS ONE* **8**(10):e75358 (2013).
5. Grisewood et al. "Computational redesign of acyl-ACP thioesterase with improved selectivity toward medium-chain-length fatty acids." *ACS Catal* **7**(6):3837–3849 (2017). doi:10.1021/acscatal.7b00408. PMC5784442.
6. Chowdhury, Ren, Shankla, Decker, Grisewood, Prabhakar, Baker, Golbeck, Aksimentiev, Kumar, Maranas. "PoreDesigner for tuning solute selectivity in a robust and highly permeable outer membrane pore." *Nat Commun* **9**:3661 (2018). doi:10.1038/s41467-018-06097-1. PMC6131167.
7. Chowdhury, Grisewood, Boorla, Yan, Pfleger, Maranas. "IPRO+/−: computational protein design tool allowing for insertions and deletions." *Structure* **28**(12):1344–1357.e4 (2020). doi:10.1016/j.str.2020.08.003.
8. Protein-InDelMaker. github.com/maranasgroup/Protein-InDelMaker.
9. Cornell et al. "A second generation force field…" *JACS* **117**:5179–5197 (1995) — the LJ parameters.
10. lucidfrontier45/autodock-vina-agent — the skills-packaging pattern this repo follows. Its skills live under `.agents/skills/<name>/SKILL.md`; this repo uses `.claude/skills/<name>/SKILL.md` so the user's own tooling auto-discovers it, and keeps a root `SKILL.md` stub so the original convention still resolves.

### Licensing

No upstream IPRO release carries a `LICENSE` file; the Maranas group's pages
describe the suite as freely available to academic users. **No upstream code is
reproduced in this repo** — the implementation was written from the published
formulations, which is the defensible route given that licence ambiguity.

**This repo is MIT licensed** (`LICENSE`, chosen 2026-11-18). The licence file
states its own scope explicitly, because three things in this tree are *not*
covered by it: the papers under `refs/` (third-party copyright, retained for
provenance checking only), the methods themselves (cite the papers above, not
this repo), and the optional dependencies (PyMOL, AutoDock Vina, Open Babel,
PyRosetta, MILP solvers), each separately licensed and separately installed.

---

## 7. Defects found by audit, and fixed

The implementation was reviewed independently against the three places the
program is encoded (the LP text, the PuLP builder, and the heuristic's
feasibility check). Twelve defects were confirmed by execution. They are
recorded here because every one produced a plausible-looking wrong answer
*without raising*, which is the failure mode this kind of code is most prone to,
and because anyone comparing results from an earlier checkout needs to know.

| # | Defect | Effect |
|---|---|---|
| 1 | Pair energies below `pair_eps` were dropped **individually**, leaving incomplete `w` blocks | The linearization `Σ_s w[i,r,j,s] = y[i,r]` is only valid over a complete block. Partial blocks forced `y[j,s'] = y[i,r]`, making the dropped rotamers unselectable: the solver returned a suboptimal design **marked proven optimal** (measured: +91.08 against a true optimum of +87.41), or called a feasible program infeasible. Blocks are now kept or dropped whole. |
| 2 | Rotamer atoms carried a synthetic residue key, not their position's | The design position's own backbone is in the fixed template, so the same-residue exclusion never fired and **every rotamer was scored against its own backbone**. This added roughly +30 to +50 kcal/mol per rotamer, scaling with side-chain size, biasing the whole objective toward small residues and pruning large side chains at the default clash threshold. Measured on 1CRN A16: ALA self-energy +27.8 → −0.07 after the fix. |
| 3 | `min_class` / `require` naming an identity no rotamer has were emitted only as an LP comment | The solver returned a design that violated the stated constraint, proven optimal. Reachable from `reshape_channel`'s automatic W/F/Y/M constraint. Now raises `Infeasible`. |
| 4 | Any `milp_solver` name other than `pulp`/`binary`/`greedy` fell through to the heuristic | Every documented choice (`cbc`, `glpk`, `cplex`, `gurobi`, `highs`) and every typo silently ran the heuristic. Now the name is honoured or an error is raised. |
| 5 | The heuristic had no repair branch for `max_class` or `max_energy` | It rejected every move and the loop reported a false "design space exhausted". |
| 6 | The H-bond well applied to every N/O–N/O pair | Two ARG NH1 at 3.4 Å scored **−0.83 (attractive)** instead of +0.67 — a sign flip on like charges — and a salt bridge was rewarded twice for one contact. The term is now restricted to pairs where **at least one atom is neutral** (`qq == 0`), which excludes both like-charged and oppositely-charged pairs and so fixes both symptoms. *This row said "neither atom carries a formal charge" until 2026-11-18, which is not what the code does and not what it should do*: a charged carboxylate acceptor against a neutral amide donor is a real hydrogen bond, and excluding it would cost 1.5 kcal/mol per charged–neutral polar contact. The code was right and the description was wrong (§7d). The well is additionally excluded between backbone atoms of sequentially bonded residues, whose separation the peptide geometry fixes (§7d #4). |
| 7 | `max_energy`'s cap went through `int()` | A −12.5 kcal/mol cap became −12, and −0.5 became 0. |
| 8 | `proven_optimal` accepted any status containing "optimal" | GLPK's `INTEGER NON-OPTIMAL` read as proven. |
| 9 | Self energies were keyed by `(position, aa, state)` | A library with a duplicated `(aa, state)` collapsed two rotamers onto one energy. |
| 10 | Atoms supplied without residue keys shared one empty-tuple key | `internal()` scored each atom **against itself** at a clamped 0.3 Å, inventing a large repulsion. |
| 11 | Integer cuts used `len(assignment) - 1` | A visited sequence naming a position absent from the tables left the cut slack by one, so the sequence could return. |
| 12 | A position with no rotamers emitted no assignment row; CBC's status was read only from line 0 | A program with an undefined position; an infeasible CBC result degraded to a generic error. |

Two earlier rounds of testing had already found and fixed: a solver proving
infeasibility while still writing a zeros solution file (so an assignment was
fabricated), the heuristic returning a sequence an integer cut forbade, a
position string like `"115"` parsing as chain 1 residue 15, and PyMOL's
`do_select` deleting the loaded object for insertion-coded positions.

Each has a regression test in `tests/test_ipro.py` named
`test_regression_*`.

**Consequence for anyone comparing runs:** results from a checkout before these
fixes are not trustworthy, particularly any that reported small residues at
buried positions.

### 7b. Second independent review

A second review was run over the parts the first did not reach -- the loop's
state handling, the geometry builders, and the file writers -- with
instructions to *execute* rather than read. Fourteen more defects were
confirmed, in the same family: all ran cleanly and returned a plausible wrong
answer.

| # | Defect | Effect |
|---|---|---|
| 1 | The carbonyl O of an inserted residue was placed at a fixed `N-CA-C-O` dihedral of 180° | It must be **psi ± 180**, since O(i) is anti to N(i+1). Any N-terminal insertion, or a C-terminal insertion of two or more residues at the default psi, put O(i) **0.871 Å** from N(i+1) (native 1CRN: 2.299 Å) with an O-C-N angle of 39.6° instead of ~123°. The structure looked like a protein and scored like nonsense. Appending also left the previous residue's O at its crystal direction (O–N 1.942 Å) and left a stale `OXT` in the position the new nitrogen needs. |
| 2 | `_tables_for` built its template from the input `Structure` objects, ignoring `_poses` | After the first redock the MILP kept optimizing against the partner position the docking step had abandoned. Divergence grew from 0.005 to **6.51 kcal/mol** over six iterations. The build and the post-redock refresh now share one `_template_for`, so they cannot drift apart again. |
| 3 | `_refresh_self_energies` recomputed only the interaction term | It silently dropped the `reference_energies` and `strain_weight` contributions `build_tables` had added, so identities were ranked on one scale before the first redock and another after it. Measured: a TRP self energy moving **+5.5 kcal/mol** with the partner unmoved. Refresh and a full rebuild now agree to 0. |
| 4 | `_rebase_choice` read the old rotamer index out of the **new** table | Exactly the mistake its own docstring warned of. A current choice of TRP came back as SER with no energy test, and those indices become `require` pins, so the remainder of the run was pinned to a residue nothing had chosen. It also ignored the restraint vetoes, so a pin could make the next program infeasible — which the loop reports as "design space enumerated", a silent early stop with a plausible message. The loop now carries identities, not indices. |
| 5 | A pore slice admitted every atom within a slab, then gave those outside their own radius an in-plane radius of **zero** | A radius-0 obstacle still blocks an inscribed circle, so a carbon 2.5 Å off-plane cut a true 4.300 Å lumen to **2.164 Å** and moved the reported bottleneck. Each slice is now a true sphere–plane cross-section and the undocumented `slab` parameter is gone. |
| 6 | `write_complex` masked residue numbers with `% 10000` | `-1 % 10000` is 9999 in Python, so an indel-created `A-1` was written as residue 9999 and read back at the wrong end of the chain. Negative numbers now write as themselves; a number that genuinely cannot fit the column is reported instead of silently wrapped. |
| 7 | The free-chain-id list was built once and never pruned as partners kept their own ids | A partner on chain B and another remapped from chain A both became B: **two molecules merged into one**, with only one remap noted. |
| 8 | `combine` required every assembly to offer an identical rotamer list | But each assembly prunes against its own partners, so a rotamer clashing with one assembly's partner aborted the whole run as an "internal inconsistency" — in exactly the case assemblies exist for, where clashing with the *anti-target* is the point. On a two-assembly specificity run the previous code raised `ValueError` and produced nothing. The surviving sets are now unioned and every assembly is rebuilt over that common set with pruning off, so one index means one conformer everywhere and the per-assembly self energies carry the difference. |
| 9 | The fixed-atom index kept whichever structure came first | A partner reusing the design molecule's chain id made `A16:CA` resolve to a molecule 50 Å away, and every veto derived from it was wrong. A restraint cannot name a molecule, so this is ambiguity in the input: it is now **refused** with an explanation rather than resolved by ordering. |
| 10 | The merged interaction map used `dict.update` | A molecule appearing in two assemblies kept only the last value (per-assembly 3167.8 / 5073.3 → scored 5073.3), so the result depended on assembly order while `merged.total` summed both. Now accumulated, weighted, so the two agree. |
| 11 | `check_restraints` sent a one-atom `fix`/`tether` into `dihedral(*coords)` | `ipro-agent validate` reported a perfectly valid backbone `fix` as a problem. (`plan_restraints` handled it correctly, so the two disagreed.) |
| 12 | The net docking pose was recovered by fitting start to finish (Kabsch) | A rotation about the axis of a one- or two-atom or collinear body moves none of its atoms, so no fit can see it: the fitted atoms came back exact while anything off that axis — hydrogens, which the scoring set drops — was placed by whichever spin the SVD produced (median 1.1 Å, max 5.0 Å). `redock` now **composes** its moves instead, which is exact for any body; `Pose.then` replaced a re-fit in the loop that was losing 0.997 Å; and where only coordinates exist (a docking program's output) the fit returns the zero-twist rotation and says that it did. |
| 13 | `_design_pdb` was not in the per-iteration snapshot | A rejected perturbation restored the `Structure` but not the file, so scoring read the restored coordinates while the written PDB and Vina read the rejected ones (0.070 Å apart at A16). The early `break` on an exhausted design space also left the backbone perturbed, so the final report described a backbone no iteration had accepted. |
| 14 | A design position's label was a lossy encoding of (chain, number, icode) | `1:115` became the text `"1115"` and re-parsed as **chain A residue 1115**, so the mutation was applied to a different residue — one that may well exist, so nothing downstream noticed. The label now round-trips for every spelling, and any future lossy label raises instead of mutating. Worse, `_write_design` only *noted* a failed mutation and still wrote the PDB, which carries a `REMARK` naming the design and its energy: the file claimed a mutation it did not contain. A failed or unverifiable mutation now suppresses the structure. |

Minor, fixed alongside: `_PSI_MOVES` omitted `HXT`, so psi stretched the
C-terminal O–H bond from 0.954 to 0.983 Å; `choices_for` returned `{}` in the
single-assembly case; an assembly name containing `::` was accepted and
silently dropped that assembly's rotamers; and `stop_after_accepted` ended a
run without adding a note, so the report showed "3/8 iterations" with no
reason given.

Reported **clean** by this review: identity-coupling completeness and
constraint scoping, `split_vpos` on `A-1` and `A115B`, the pore `window`
semantics and centre-offset re-enforcement, deleting a design position,
`_sort_chain`, `A0`/`A-1` parsing, the phi/psi moving sets, the annealing
endpoints, and `net_interaction`'s pair counting.

### 7c. The item the review could not check was a fifteenth defect

The one thing the reviewer left unverified — it had no PyMOL installed — was
whether PyMOL accepts `/prot//A/-1` for a negative residue number. It does
not, and the alternative is worse. A negative residue number is exactly what
an N-terminal insertion creates, since it numbers downward so that nothing
existing is renumbered:

| Selection | Selects |
|---|---|
| `/prot//A/-1` | **0 atoms** — the slash form does not take a leading minus at all |
| `prot and chain A and resi -1` | **residues −1, 0 and 1** — PyMOL reads `-` as its range operator, so this is the range "up to 1" (measured: 15 atoms where the residue has 4) |
| `prot and chain A and resi \-1` | 4 atoms, residue −1 only — the one correct form |

Both scripts used the slash form whenever there was no insertion code, so a
design position at a negative number harvested nothing. Measured end to end on
1CRN with two residues prepended and `A-1`/`A0` as the design positions: the
previous commit reported **one** mutation (`G0W`) where two positions were
requested, left `A-1` as glycine with only backbone atoms, and raised nothing.
It now reports `G-1S G0W`, with SER's OG and TRP's full indole built.

Each of the fifteen has a regression test. The suite is **94 tests**.

### 7d. Third review round: the five modules neither earlier review reached

The first two reviews covered the MILP and the LP text (§7), then the loop's
state handling, the geometry builders and the file writers (§7b). Five further
independent reviews were run in parallel over everything left: `energy/*.py`,
`goals.py`, `config.py`, `structure.py`, `rotamers.py`, `__main__.py` and
`scripts/*.py`. Each was told to *execute* rather than read, and to report a
measurement per finding. **Thirty-one more defects were confirmed**, all in the
same family: every one ran clean and returned a plausible wrong answer.

The suite is now **141 tests** (94 before), one per fix.

#### The energy function (`energy/base.py`, `energy/builtin.py`)

| # | Defect | Effect |
|---|---|---|
| 1 | The per-pair cap applied to the **LJ term only**; the Coulomb term runs as 1/r² down to the 0.3 Å clamp with no bound | So the most *attractive* value the function could return was a hard clash between opposite charges. ASP OD1 ··· LYS NZ: **−3.577** at a 3.50 Å salt bridge, **−31.508** at 1.00 Å, **−451.200** at 0.30 Å. A rotamer burying LYS NZ inside a template oxygen scored **−71.957** against the native alanine's −1.391 and came back from the MILP as the **proven-optimal** design. Like charges ran away identically (+932.399 for one pair at 0.30 Å), past the `max_self_energy` and `hard_reject_kcal` thresholds that are calibrated against this bound. The **whole pair term** is now bounded to ±`max_repulsion`, which is what §2 always claimed. Pairs closer than ~1.8 Å are the only ones affected; a real salt bridge is untouched (−3.577, unchanged). |
| 2 | Covalent **1-2 and 1-3 pairs across the peptide bond** were scored as non-bonded | The same-residue exclusion cannot see a bond that spans two residues. On `2omf_A`, `internal()` reported **+11132.191 kcal/mol**: +3390.000 from the 339 peptide bonds (every one pinned at the repulsion cap) and +10390.954 from the 1048 1-3 pairs, against a genuine non-bonded sum of −1070.880 — 108% of the LJ sum was covalent geometry. It also steered the search: **5 of 25** default backbone perturbations were hard-rejected on chain connectivity, which a perturbation cannot change. Now −1949.303, i.e. a favourable packing energy for a crystal structure. |
| 3 | `AtomSet.elements` took the first alphabetic character of the atom name, so **no two-letter element was reachable** | `LJ`'s own `CL`/`BR`/`SE` entries were dead code. A backbone O 2.40 Å from an **iron** scored +5.8272 using *fluorine's* parameters; Cl and Ca were typed as carbon; Na was typed as nitrogen and so handed out −1.5 kcal/mol hydrogen bonds. Metalloproteins and cofactor sites were scored on fabricated parameters. `AtomSet` now carries an `elements` list, every structure-derived set passes the PDB element column, and the name-only fallback resolves ions by residue name (with MSE's SE kept as selenium). |
| 4 | The H-bond well fired on **sequential backbone** N/O pairs | Their separation is fixed by the peptide geometry, so it cannot be a hydrogen bond. On `2omf_A` this paid −1.5 kcal/mol to each of **339 N(i)···N(i+1)** pairs — one per peptide bond, two amide nitrogens that cannot bond to each other — and 98 O(i)···O(i+1) pairs: **−598.407** in all. A side chain reaching across a peptide bond is a real H-bond and is left alone. |
| 5 | Only `names` was length-checked | A short `resnames` list silently zeroed the trailing atoms' formal charges: a measured −7.2102 kcal/mol carboxylate/ammonium interaction came out **+5.6009**. All four per-atom lists are now checked. |

#### The objective (`goals.py`)

| # | Defect | Effect |
|---|---|---|
| 6 | `goal_options` was never checked against anything | The one hole in the repo's own "a typo must not silently change the design problem" policy — and it carries every objective weight and the published acceptance filter. `pore_diameter_maximum` for `pore_diameter_max` left PoreDesigner's hard filter **inert**: a 20.00 Å bottleneck was not rejected, `notes` was empty, and the design shipped unfiltered. `antiweight: 5` left the weight at 1.0. Keys are now whitelisted per goal, plus the twelve read elsewhere in the run. **This also caught two dead keys in both shipped example specs** (`min_long_fraction`, `short_residues`), which read as though they were doing the work that `long_fraction` was actually doing. |
| 7 | A zero or negative weight was accepted and **inverted** the term it weighted | With `anti_weight: -1` and IE(target) = −10, binding the anti-target 8 kcal/mol *tighter* scored −28.000 against −12.000 for rejecting it: the loop optimized the opposite of the request, and its objective curve still fell. `radius_weight: -1` drove the lumen away from the requested size, unboundedly. `Assembly.weight` was already refused on exactly these grounds. |
| 8 | A note raised **while scoring** reached nobody | The loop copied `goal.notes` at construction. Measured: a `reshape_channel` run whose radius term was never active — objective identically 0.0 every iteration — reported a mutation string and a best score with **no caveat at all**, against the house rule that a design is never reported without its notes. The loop now shares the list. |
| 9 | IPRO+/−'s interaction floor was a unit-weight quadratic **penalty, not a constraint** | §1 records the form as `min E_complex s.t. IE ≥ 0.75·IE_WT`. With the floor at −15.0, a candidate that destroyed the interaction entirely (IE 0.0) scored **−115.000**, better than a compliant candidate at −100.000, and was kept as the run's best design. Now a `filter_reject`, with the penalty retained as a gradient inside the boundary. |
| 10 | `pore_diameter_min/max` counted as a geometric *objective* | They are a filter; nothing in `score()` reads them. Configuring the paper's published filter therefore **suppressed the degenerate-objective warning**: a 3.2 Å and a 3.8 Å candidate both scored −100.0 with nothing said, while a spec with no options at all got the warning. |
| 11 | `max_total_energy` was documented by `BindTighter` and read **nowhere** | A spec asking for a −50 kcal/mol stability cap ran to completion on candidates at +500 with no note. `ctx.total` was measured and carried for exactly this. Now enforced. |
| 12 | `parameter: "km"` silently fell back to **every** target | OptZyme Eq 7 is the IE with the *substrate*. With no `ligand`-kind molecule, an unrelated protein partner at −40 entered a "K_M" score of **−49.0** that `components()` reported as **−9.0** — the number could not even be explained. One target is now used explicitly and noted; several are refused as ambiguous. |
| 13 | A declared `anti_target` was silently ignored by every goal but `switch_specificity` | `bind_tighter`, `stabilize` and `reshape_channel` scored identically whether the decoy was bound at −10 or rejected at −2, with `notes` empty. Now reported. |
| 14 | `validate()` did not check `reference_interaction` | The error raised later from `score()` told the user to "measure it with `ipro-agent score`" — and that command builds the same goal and dies with the same error. The documented recovery path could not be followed. |
| 15 | `suggest_goal` returned the first keyword in **word order** | "change its binding specificity from C12 to C14" → `bind_tighter`, which optimizes plain affinity and ignores the decoy. Now ranked by specificity, and `decoy` — the 2006 paper's own word — was missing from the synonyms. |
| 16 | `_sum_ie` iterated molecule *objects* against a name-keyed dict | A copy-pasted molecule block counted one measured interaction energy twice: 29.5640 reported as **59.1280**, a selectivity gap 2.1× too large. Duplicate names are now refused in `DesignSpec` *and* the sum is over names, so the two cannot disagree again. |

#### The spec (`config.py`)

| # | Defect | Effect |
|---|---|---|
| 17 | A YAML **scalar string** where a sequence was expected was iterated character by character | `allowed: all` gave 2 residues of 20; `allowed: standard` 6 of 18; `positions: "115"` gave **three** positions A1, A1, A5. Worst, `indels: [{kind: delete, residues: "12"}]` reported `removed 2 residue(s): A2, A1` — it deleted residues 1 and 2 when residue 12 was named, defeating §1i's invariant that an indel never touches a residue nobody named, with `ok: true`. A scalar is now wrapped, not iterated. |
| 18 | The `{position: "A16", ...}` spelling read only `position` and `allowed` and **discarded every other key** | A misspelled `alowed:` silently widened a position from 1 residue to 18; an `insertion_code:` was dropped, so A16B was designed as A16 — §7b #14's failure mode re-entered by another door. The sibling `{chain:, number:}` spelling rejected the same input. |
| 19 | The same design position could be listed **twice** with contradictory `allowed` sets | The MILP built variables from the last entry, `position()` returned the first, and `design_space_size()` multiplied both (4 sequences reported for one position with 2 choices). The run could not produce the residue the user had pinned, and nothing said so. |
| 20 | No `None` coercion in `LoopSettings` | `integer_cuts:` written with no value parses as null, which is falsy, so a documented `True` default silently became `False`: integer cuts off, and a 6-iteration run "accepted" **six copies of one design**. A quoted `"false"` was a non-empty string and so read as `True`. Null is now refused except where it documents "no limit". |
| 21 | A `fix` restraint's explicit `maximum` was overridden by `FIX_TOLERANCE` | `maximum` of 0.05/0.1/0.2/0.5/1/2/4 was enforced at **0.05 Å** in every case — up to 80× tighter than written, which narrows a design rather than refusing it. The error message three lines below offers `maximum` as the way to state the allowance. |
| 22 | An empty `positions: []` ran to completion | `ok: true, accepted: 3, mutations: {}`, no note — which reads as "the optimizer found no improvement", not "nothing was designed". Now refused (positions may still arrive from an indel, §1i). |
| 23 | A molecule named twice **inside one assembly** was kept twice | Its own internal energy entered the complex energy once per listing (+1588.339 measured) while the interaction map, keyed by name, showed one copy — so `total_energy` and the per-molecule numbers disagreed. |
| 24 | `AtomRef.position` and `ResidueRef.label` dropped the separator `DesignPosition.label` uses | Chain `1` residue 115 was `1115` in two places and `1:115` in the third, and `1115` re-parses as **chain A residue 1115**. Measured: a mutation applied 40 Å away to a residue of the same type, nothing raised; and a restraint on such a position reclassified as "held fixed, satisfied by construction" and never enforced. §7b #14 fixed one of the three; all three now share one `position_label()`. |

#### Structures and rotamers (`structure.py`, `rotamers.py`, `scripts/`)

| # | Defect | Effect |
|---|---|---|
| 25 | **PyMOL's rotamers carried hydrogens** while every template is heavy-atom-only | The side chain brought its own backbone H and HA, which the template no longer held, so every rotamer gained a near-constant **+18 to +22 kcal/mol**. On a protonated `2omf_A` with the shipped defaults that pushed **94 of 160** rotamers past `max_self_energy` instead of 46 and deleted **TRP, PHE, TYR and PRO** from position A18 outright — so the published V18W answer was not in the design space, while `validate` reported 6561 sequences and no error anywhere. The side-chain hydrogens also re-ranked the conformers. `apply_design.py` had the mirror-image bug: it compared the applied side chain against the rotamer *including* H, so on protonated input it reported "not rotamer state 4: 1 atom(s) differ (H)", suppressed the structure write, and blamed the user's PyMOL build while `doctor.py` reported PyMOL working. |
| 26 | `Structure.load` discarded every Biopython construction warning | Those warnings are the only signal that the file was not read as written. Two copies of a molecule pasted into one chain loaded as **82 atoms of 164**; a chain holding two different residues numbered 115 silently dropped the whole TRP. Every energy afterwards was computed on half the molecule, and `inspect` reported the reduced counts as the file's contents. Now refused, with the parser's own message. |
| 27 | `Structure.save` wrote `self._bio`, not the model | The whole file went back out: every model and every chain, including the chains `load(chains=...)` was asked to exclude (the filter only detaches from model 0). A Structure holding 2 atoms of chain A saved 6 ATOM lines across 2 models and chains A and B. These are the files PyMOL and Vina read, so an excluded partner was back in the scoring input while the Structure's own energies never saw it. |
| 28 | `build_template`'s hetero loop had no `is_design` check | A design position on a HETATM-flagged residue — MSE and every other modified residue is one — kept its **full native side chain** in the fixed template. Its own rotamers were unaffected (matching key, so the exclusion fired), so the damage landed on its neighbours: TRP at a neighbouring position moved **+78 to +284 kcal/mol** depending on the conformer, and the chosen rotamer flipped from state 6 to state 1. |
| 29 | Residue keys carried **no molecule discriminator** | A residue key is what the backend's same-residue exclusion compares, and a partner reusing chain id `A` — the default for most single-chain PDB files — collided with the design molecule's own residues: 22 distinct residues collapsed to 11, the partner's residue 115 lost its side chain to the design filter, and a ligand numbered like a design position became **invisible** to the rotamer sitting on top of it (+84.699 scored as −6.282). The design molecule keeps the bare key that rotamer atoms must match; every partner is prefixed. |
| 30 | `harvest` read `rotamers.json` whenever it existed and never checked the return code | The loop passes the same workdir every iteration, so a failed re-harvest returned the **previous backbone's** rotamers: with the backbone translated 10 Å and PyMOL exiting 1, side chains came back 10.0 Å from where they belonged, `errors` was empty, and a hard-clash conformer (+453.111 against the real template) was the best-scoring one at −1.036. A requested position that came back missing was likewise dropped in silence (2 requested, 1 designed, nothing raised) — an incomplete crystal structure missing one CA is enough to trigger it. |
| 31 | `doctor.py` probed with no `-I`, so the working directory was on `sys.path` | From a directory holding a `pyrosetta.py` and a `vina/` folder, doctor reported **"OK PyRosetta ... OK AutoDock Vina"**, resolved energy to pyrosetta and dock to vina, and printed `degraded = []` — so the run looked not degraded at all and the mode line the house rules require was wrong in both directions. Neither is installed on this machine. The run itself probes with `-I`, so doctor saw modules the run could not. Also fixed alongside: the 20 s probe timeout was shorter than PyRosetta's real import (45 s measured), so a machine that *has* PyRosetta was reported as not having it; `IPRO_VINA` and `IPRO_PYMOL_PYTHON` — the documented pins the run honours — were ignored by doctor; and `milp.available` was true for a solver binary doctor had just called unusable, so `doctor.py --require milp` exited 0 and the greedy fallback (which proves no optimality) was invisible to any caller. |

Minor, fixed alongside: `scripts/selftest.py` caught `Exception`, not `BaseException`, so a single `SystemExit` in a test — or a `pytest.skip`, which its own docstring invites — ended the suite with **exit 0, no summary line and no record of what never ran**, which would have made every "94 tests passing" claim in this repo unfounded; `BACKBONE` omitted `OT1`/`OT2`/`OT`, so a C-terminal design position in a CHARMM/AMBER-named file lost its carbonyl and both terminal oxygens from the template; the heavy-atom filters tested `element == "H"` only, so deuterium from a neutron structure was scored with carbon's parameters; `pore.vdw_radii` had its own copy of the first-alphabetic-character element bug; the MILP's position-pair pre-screen compared mean-centroid separation against radii measured about each rotamer's *own* centroid, which is not an enclosing bound (a real −0.014002 kcal/mol pair 6.0 Å away was dropped and the block recorded 0 pair terms); and a `strain_weight` was applied to eight identities but not to ALA, for which PyMOL emits no bump score, so ALA competed without the cost the others paid.

Three documentation claims were found false and corrected rather than coded around: §2's "repulsion cap 10 kcal/mol **per atom pair**" was true of the LJ term only (now true of the pair), the built-in model was described as having a **solvation treatment** when it has none (§4, §5 and `doctor.py`'s own output said so), and §7 #6 stated the H-bond exclusion as "neither atom carries a formal charge" when the code requires only that *one* be neutral — the code is the better physics (a charged carboxylate acceptor against a neutral amide donor is a real hydrogen bond), so the **documentation** was wrong, not the code.

#### Reported clean by this round

The LJ functional form and combining rules (`E(Rmin) = −eps` exactly), Coulomb's sign and dimensions, the H-bond well's shape, pair symmetry (1.4e-14 on a 611-atom split), exact partitioning (`internal(A∪B) = internal(A)+internal(B)+interaction(A,B)` to 0.000e+00), iteration-order independence (≤4.5e-13 over five shuffles), empty and single-atom sets, every goal's **direction** (a better candidate scores lower, a worse one higher, for all five goals and all three OptZyme parameters), the multi-assembly specificity split, `metropolis_accept`'s edges, all 19 of §2's numeric defaults against the installed values, position parsing and `to_dict`/`from_dict` round-tripping, path resolution relative to the spec file, restraint interval validation, `2omf_A` load→save→load field-for-field identity, altLoc selection by occupancy, insertion codes and negative residue numbers end to end, the harvest↔apply state contract to 0.0005 Å, and `_selection()`'s escaping of negative residue numbers on PyMOL 3.1.0 (§7c stays fixed).

#### One confirmed defect left unfixed, deliberately

**The fitted pore axis can miss the lumen entirely, and the published acceptance
filter then passes on a number that is not a pore measurement.** The axis is the
normal to the plane of the design positions, through their centroid — which is
the pore centre only if those positions *ring* the lumen. The shipped smoke
spec's four positions all line one wall, so their centroid falls **inside** the
wall: measured, it clears the nearest van der Waals surface by **0.30 Å** where
a water needs 1.4 Å, and the profile then reports the sliver beside the wall as
the channel. That reads **1.53 Å**, which silently **passes** PoreDesigner's
published "bottleneck below 4 Å" filter — a design accepted as a sub-4 Å pore on
a measurement of the wrong thing. The 25-position spec, whose ring does surround
the lumen, clears 1.30 Å and measures 6.24 Å.

Three cheap discriminators were tried and all three fail, which is why this is
recorded rather than fixed:

| candidate test | bad axis (4 positions) | good axis (25 positions) | separates? |
|---|---|---|---|
| axis-origin clearance | 0.30 Å | 1.30 Å | **no** — both below a water's 1.4 Å |
| widest slice in the window | 3.765 Å | 4.752 Å | **no** — overlapping ranges |
| bottleneck ÷ clearance | 2.52 | 2.41 | **no** — the ratios are the same |

A clearance threshold was implemented and then **reverted** on the first of
those: it rejected the 25-position spec's sound reading too. Telling the two
apart needs the measurement reworked — a connected-passage test along the axis,
or fitting the axis through the lumen rather than through the design positions —
which is a change to the science, not a defect fix, so it is left for a decision
rather than guessed at.

What the run does instead is **report what it cannot vouch for**: the measured
clearance is returned as `pore_axis_clearance` alongside the radius, and a note
fires whenever it is below a water's radius, stating that the geometry alone
cannot confirm the axis is in the lumen and naming the known failure. That note
fires on *both* shipped specs, which is correct — the default axis fit is a
heuristic and its quality is genuinely unverified in both. **Treat any pore
radius from a spec whose design positions do not surround the channel as
meaningless, and give `goal_options.pore_axis` explicitly.** The numbers §1h and
both example specs quote are corrected below.

### 7e. The PyRosetta and Vina paths, both now executed

Both were listed as implemented and **neither had ever executed**, because
neither was installed. Both have now been installed and run, and between them
they held **seven more defects** -- which is the whole argument for not trusting
a path because it reads correctly.

**AutoDock Vina: installed, and the path was broken.** Vina 1.2.7 and Open Babel
3.2.1 were installed from conda-forge (`/opt/miniconda3/envs/ipro_vina`), and the
first execution found that **`backends.dock: vina` could not dock at all**. Vina
reads PDBQT only; a spec names PDB files, which is the format every example,
every document and the structure loader use; and nothing converted them. Every
redock failed with `receptor receptor.pdb is not a .pdbqt file`, the loop noted
the failure and left the placement unchanged — so a run reporting `dock -> vina`
had never moved a partner. Worse, `backends.dock: auto` resolves to Vina whenever
Vina is on `PATH`, so **installing Vina made docking stop rather than start**.
`pdb_to_pdbqt` existed for exactly this job and had no call site. With the
conversion wired in, the same run docks: score **62.6984 → −10.0636**, three
poses written, `redocked 1 partner(s)`. Open Babel's conversion is a stopgap and
`pdb_to_pdbqt`'s own docstring says why (its receptor charges and atom types are
not Meeko's); what makes that acceptable here is that Vina's affinity never
enters the objective — the *pose* is taken and re-scored with this tool's
backend.

**PyRosetta: installed, and three of the five declared paths were defective.**
PyRosetta **2026.29+releasequarterly** (native arm64, 1.5 GB wheel, 3.0 GB
installed) was installed into the conda env `ipro_rosetta` under its own
licence, which each user must obtain for themselves. Every path named in the old `UNEXERCISED` list was then
run. What it found:

| # | Defect | Measured |
|---|---|---|
| 1 | **`relax(fixed=[...])` did not hold anything.** The MoveMap idiom is right for *minimization*, but FastRelax ramps repack cycles whose rotamer set comes from a PackerTask -- and fixing a residue's own torsions does not fix its Cartesian position anyway, because the fold tree translates it when an upstream torsion moves. | On 1crn A16, side-chain displacement after one cycle: MoveMap only **0.866 Å** (2.922 Å on another seed); MoveMap + `PreventRepacking` **2.200 Å**; adding **coordinate constraints 0.0001 Å**. So a `fix` restraint on a design position -- which the run reports as enforced, and whose own note promises it is "held during relaxation in 'full' mode" -- was silently violated by the relaxation step. Now held with coordinate constraints, with the MoveMap and `PreventRepacking` kept for what each is right about, `RestrictToRepacking` so relaxation can never change an identity, and the displacement **measured and checked** afterwards: a violation raises (the loop then reports a failed relaxation and downgrades the mode) rather than writing a structure that breaks the restraint. Re-measured: held 0.0 Å, a free residue moves 0.376 Å. |
| 2 | **`interface_energy` raised `TypeError` on every call.** `InterfaceAnalyzerMover("A_B")` is not a constructor this PyRosetta has. | Its overloads take a jump index, a set of chain indices, or a `DockingPartners`; `DockingPartners` has only a default and a copy constructor; `set_interface` takes one too. The string is parsed by the free function `docking_partners_from_string`, which is what it now uses. It had no call site, so nothing downstream noticed. Re-measured: returns `dG_separated`, `interface_dSASA` and `total_score`. |
| 3 | **A second backend's `init_flags` were silently ignored.** The guard was a bare "have we initialized" boolean. | Rosetta's re-initialization does work and is reversible: `init("-ex1")` then reading `packing:ex1` gives True where the default gives False, and initializing again without it returns to False. The guard is now keyed on the flag string, so a repeat still costs nothing and a genuine change takes effect. |
| 4 | **The relax score was contaminable.** Setting `coordinate_constraint` on `self._sfxn` would have added a restraint penalty to every later `score_structure`. | The constraint weight now goes on a per-call copy of the score function, and the returned score comes from the unconstrained one. Verified: `coordinate_constraint` weight on `self._sfxn` is **0.0** after a relax, and the 0.04 REU gap between the returned score and a re-score of the written PDB is write precision, not a restraint term. |
| 5 | **The deliverable PDB was not a valid PDB.** The REMARK carrying the score embedded the backend's full `units` sentence. | With the PyRosetta backend that made one line **230 characters** where a PDB record is 80 columns. A short `units_short` label now goes in the header; the long form stays in `best/design.json`, which has no column limit. |
| 6 | **`doctor.py` reported a backend the run could not use.** Unlike PyMOL — driven as a subprocess, so any environment will do — PyRosetta must be importable *in process*, because `energy.resolve()` imports it directly. doctor searched the conda environments and counted a hit as available. | With PyRosetta installed only in `ipro_rosetta`, doctor run under the base interpreter printed `energy -> pyrosetta` and left PyRosetta out of `degraded`, while a run started with that same interpreter silently used the built-in screening function. The mode is reported with every design, so this was a scientific misstatement. Finding it elsewhere is now a hint (`how: "elsewhere"`, `available: False`) that names the interpreter to use instead. |
| 7 | **A module's `version` *function* was printed instead of called.** | doctor's probe did `getattr(m, "__version__", "") or getattr(m, "version", "")`; PyRosetta's `version` is a callable, so the mode report read `[import] <function version at 0x1644d8040>`. It now calls it, and prints the real build string. |

**One declared suspicion was wrong, and the truth is different.** The old entry
said `-ignore_unrecognized_res` "silently drops any HETATM Rosetta lacks
parameters for", which `relax` would write back truncated. Measured on four
cases, that is not what happens: an unrecognized residue of one *or* two atoms
makes `pose_from_pdb` **fail loudly** (`too many tries in fill_missing_atoms`),
a zinc is **kept** (47 residues for a 46-residue protein plus ion), and what is
silently dropped is **water** -- which `Structure.hetero_residues()` already
excludes, so the residue-count guard's two sides agree and it does not
false-fire. The guard stays as a cheap check; the claim is corrected.

**What is still unexercised**, and now the whole of `UNEXERCISED`: the
Rosetta-scored pair table behind `exact_pairs=True` (which raises by design, so
it does not exist to be run), and `relax` on anything other than a
single-chain protein -- several chains, a parameterized ligand, or
`relax_cycles > 1` are untested.

And before any of that could be reached: **`relax()` had no call site anywhere
in the package**, and neither did `RestraintPlan.relax_fixed_positions`, which
is collected for exactly that argument. So a run with a relaxing backend
reported mode **`full`** — documented as "perturb + relax + repack" — while
doing no relaxation at all; and because `can_relax` was true it also
**suppressed** the note saying the reported energies were screening energies,
writing screening kcal/mol into the deliverable PDB under a "Rosetta Energy
Units (REU)" heading. Both were fixed with a counting stand-in backend standing
in for Rosetta, and the real backend has now confirmed the wiring end to end: a
2-iteration `perturb_backbone` run with `backends.energy: pyrosetta` reports
`backbone relaxation ran on 2 of 2 iteration(s)`, mode `full`, and a REMARK
whose units name the screening scale — the pair tables really are the screening
terms unless `exact_pairs` is set, so the caveat now follows **where the
pairwise numbers came from** (`screening_pairwise`) rather than whether the
backend can relax.

Four of the regression tests for this section run only when PyRosetta is
importable, and are skipped otherwise, following the repo's existing convention
for PyMOL-gated tests. The suite is **141 tests** either way: 137 run
everywhere, and those four add real coverage in an environment that has the
licence. Both were run: 141 passed under `python3` (PyRosetta absent, so the
fallbacks are exercised) and 141 under the `ipro_rosetta` interpreter.

**Cost, for the record.** The wheel is 1.5 GB and the environment 3.0 GB; with
pip's cache the install took this machine from 31 GB free to 23 GB.

### 7f. Pre-publication readiness audit

Before the first public release, five further independent reviews were run over
the **49 tracked files** — the publication set, not the 1.1 GB working tree — on
the dimensions that matter when strangers and autonomous agents clone a repo:
what must not be published, whether a fresh clone works, whether an agent can
use it on a protein the repo has never seen, packaging and licence wiring, and
whether any claim overstates what was verified. Every finding was adversarially
verified; **23 survived**, plus 9 from a completeness critic.

One was a correctness blocker of the familiar kind:

| Defect | Effect |
|---|---|
| **A candidate the goal's hard filter rejected was still recorded as the run's best design.** The tracker was a sibling of `if ok:` rather than nested in it. | Measured on this repo's own `spec_full.yaml`: 14 iterations, **0 accepted**, every one rejected with "bottleneck 7.12 A is above pore_diameter_max 4.00 A" — and `status` still printed a **14-mutation best design** at 0.7297 against a start of 1.2574, with `best/design.pdb` written for it. An agent following SKILL.md's own checklist ("the score improved; if not, say the search found nothing") would have reported those mutations as a pore-narrowing design when nothing met the published constraint. Now gated on acceptance, and a run with `accepted: 0` states plainly that nothing was designed. |

Two were problems of **claim and licence** rather than code:

* **"Clean-room" was the wrong term, and it sat in the LICENSE as the
  justification for publishing.** Clean-room means the implementers had no
  access to the original. This repo had access: §2 marks eight defaults as
  taken from the released source, and in one case records preferring the
  source's value over the paper's. The verifier normalized every ≥40-character
  non-comment line of the tracked Python against all three upstream clones and
  found **two matches, both non-copyrightable boilerplate** — so the claim that
  actually matters, *no upstream code is copied*, is true and now stated
  everywhere the false one used to be.
* **A 125-word verbatim quotation from the paywalled JCC 2015 paper sat inside
  an MIT-licensed file**, which would have purported to license Wiley's prose
  for sublicensing and sale. The passage is now a summary of the protocol's
  facts (the sigmas, the move count, the annealing endpoints) with the citation
  kept, and the LICENSE carves out quotations explicitly.

Packaging was broken in a way that only an installed copy would show: `pip
install .` produced a working `ipro-agent` command that could validate a spec
and then **died at the first rotamer harvest**, because the two scripts PyMOL
runs lived in the repo's top-level `scripts/` and
`Path(__file__).parent.parent / "scripts"` resolves to a `site-packages/scripts/`
that does not exist. They now live in `ipro_agent/pymol_helpers/`, ship in the
wheel, and were verified from a clean venv outside the repo.

`doctor.py` — the gate AGENTS.md tells every agent to trust — reported
`can run: True` with no PyMOL, when PyMOL has no fallback and `run` cannot
produce a design without it. It now reports `can_run` (all three required
modules, not just numpy) and `can_design` (which additionally needs a
mutagenesis backend) separately, and names what is missing.

Also corrected: §1b claimed restraints and binding assemblies were "not
implemented" while §1c and §1d documented both; §2 called the docking anneal
"not implemented" while it runs by default; §6 was an internal acquisition log
and is now a sources table for a reader; `docs/INSTALL.md` said this repo has no
`pyproject.toml` and pasted an `environment.yml` that differed from the shipped
one in name and in five dependencies (a test now keeps those two identical); the
README offered no install command at all and claimed "there is no setup step for
the agent" when the one unavoidable step is conda-installing PyMOL; and the
modified `1crn_icode.pdb` fixture presented itself as the deposited entry.

From the critic, two additions the audit dimensions had not asked for: a
**Responsible use** section, since a tool published so that agents can propose
mutations to "any protein" said nothing about what it should not be pointed at;
and a rule, in both AGENTS.md and the skill's checklist, that **an agent must
report any constraint it loosened** to make a failing run succeed — the
troubleshooting tables offer exactly those knobs, and each one changes the
user's scientific question. The long-run recipe gained `--max-minutes`, `setsid`
and a "submit through the scheduler on a shared machine" line.

One further correctness fix came from the critic: a terminal indel trusted the
last residue *present in the coordinates*, so on a structure with disordered
termini — where that residue sits in the middle of the real chain — appending
created exactly the unclosed break internal indels are refused for. SEQRES is
now compared against the modelled span and the discrepancy reported.

The suite is **148 tests**.
