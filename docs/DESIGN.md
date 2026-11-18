# Design document: what came from where

This repo is a **clean-room reimplementation** of the published IPRO procedure.
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
why `SKILL.md` makes the agent walk the user through the list rather than
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
| **BA** — binding assembly | the totality of molecules interacting with a particular binding topology; multiple DMs and TMs may take part | **not modelled** — see below |

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
| how to use implicit solvation | **not exposed** — the built-in model's solvation treatment is fixed |
| the BAs | **not modelled** (single assembly only) |
| whether to share Metropolis-retained results between processors | **not applicable** (single process) |
| how to do ensemble structure refinements | partial — `loop.ensemble_size` keeps an ensemble but does not refine it |

Two consequences worth stating plainly. First, IPRO supports **multiple binding
assemblies** and requires that *all* of them pass the acceptance criterion for
an iteration to be retained; this tool models one assembly, so a design that
must satisfy several binding topologies at once cannot be expressed. Second,
IPRO offers **four kinds of atom-position restraint, "rigorously enforced
throughout all functions"** — atoms fixed in place, atoms held near their input
positions, a distance restraint between two atoms, and a dihedral restraint on
four atoms. **None is implemented here.** That matters more than it sounds: the
distance restraint is the mechanism the thioesterase work used to preserve
catalytic geometry while redesigning specificity (catalytic distances held to
±0.2 Å), so "redesign this site but keep the catalysis intact" is currently not
expressible. It is the single most useful thing to add next.

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
| docking anneal | not implemented | IPRO Suite | 3640 K → 2190 K on a constant cooling schedule, inside the 500-move rigid-body docking loop. The **outer** accept/reject temperature is constant; the annealing belongs to docking, which this tool does not run inside the iteration. |
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
| repulsion cap | 10 kcal/mol per atom pair | **this repo's choice** | The original clamps *objective coefficients* at 10000 kcal/mol. Same purpose: keep rotamer clashes from dominating the program. |
| LJ parameters | Amber parm94/99, element level | Cornell et al., *JACS* **117**:5179 (1995) | An approximation: per-element rather than per-atom-type. |
| reference energies | **none supplied** | — | IPRO carries a hardcoded 20-entry amino-acid reference-energy table whose values this repo does not have. No default is invented; see §4. |

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
| Docking | **in-house Monte-Carlo** (`docking.out`), 500 sub-moves | AutoDock Vina | Not expected to give identical poses. Vina was chosen because it is open and maintained; IPRO never used it. |
| MILP solver | CPLEX via GAMS | CPLEX / Gurobi / CBC / HiGHS / GLPK, or an LP file | Same formulation, same optimum. |
| Language | Python 2 | Python ≥ 3.10 | — |

**Note:** the user's requested dependency stack (PyMOL, AutoDock Vina,
PyRosetta) is this substitution set, not IPRO's own. It was adopted as
specified, and this table is the record of what that changes.

---

## 4. What is not implemented

| Capability | Status | Why, and what it would take |
|---|---|---|
| **Insertions and deletions** | **Not implemented.** A spec containing `indels:` raises `NotImplementedError` with a pointer to the alternative. | Changing chain length needs loop closure after the edit. Protein-InDelMaker does this with PyRosetta's GeneralizedKIC over a 6-residue window followed by Cartesian FastRelax (`repeats=5`); IPRO+/− reports an 11-residue CCD loop, ≤20 iterations, 3000 runs. No loop-closure backend is available here. `config.IndelSpec` models the input so the schema is ready. Workaround: build the variant with Protein-InDelMaker, then design on that structure. |
| **Backbone perturbation and relaxation** | Only in `mode: full`, which needs PyRosetta. Otherwise the loop runs `fixed-backbone` and says so. | Perturbing a backbone without minimizing afterwards only manufactures clashes, so the step is skipped rather than faked. |
| **Pore / channel geometry measurement** | Not implemented. `geometry["min_pore_radius"]` is never populated, so `reshape_channel`'s radius term is inert and PoreDesigner's hard bottleneck filter cannot be applied. The goal reports this. | PoreDesigner measured the bottleneck by fitting ellipses to ~16 slices through the constriction, outside the optimization loop. |
| **Vina re-docking inside the loop** | `dock.py` exists and is tested in dry-run; it is not wired into the iteration. | Vina is not installed in this environment, so a wired-in untested path would be a liability. When wiring it: re-score Vina poses with the energy backend, and **never** add Vina affinities to the objective — the scales are unrelated. |
| **Rosetta-scored MILP pair tables** | `exact_pairs=True` raises. | Rosetta scores poses, not atom sets; a pose per rotamer pair is prohibitive. The screening terms build the table, as IPRO likewise uses a cheap evaluation for its combinatorial inner problem. |
| **Reference energies** | Mechanism provided, no values. | IPRO's 20-entry table was not recovered. Inventing values would put unsourced numbers at the centre of the objective. Without one, nothing corrects for unfolded-state propensity, so identities are decided by packing alone and large side chains are disfavoured at buried positions. The loop emits a note when this applies. **Note on history:** an earlier version of this document attributed an observed all-alanine bias entirely to this gap. That was wrong — the dominant cause was a defect in which each rotamer was scored against its own backbone, inflating every side chain's energy in proportion to its size. That is fixed (see §7); the reference-energy gap is a real but smaller effect. |
| **Atom-position restraints** (4 kinds: fixed, tethered, distance, dihedral) | **Not implemented.** | IPRO enforces these throughout. The distance restraint is how the published work preserved catalytic geometry while redesigning specificity (±0.2 Å on catalytic distances). Without it, "redesign this pocket but keep the catalysis" cannot be stated. Highest-value addition. |
| **Multiple binding assemblies (BAs)** | Not modelled; one assembly only. | IPRO requires every BA to pass acceptance for an iteration to be retained, which is how it designs against several binding topologies at once. |
| Separate H-bond (3 Å) and solvation (9 Å) cutoffs | Single 12 Å cutoff. | Minor; recorded for completeness. |

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

---

## 6. Sources, and what could not be obtained

Obtained in full: Biophys J 2006 and 2007 (PMC1459523, PMC1861794 — neither has
supplementary material, confirmed); Nat Commun 2018 and its SI; Structure 2020
(including STAR Methods); ACS Catal 2017 main text (PMC5784442); the IPRO Suite
source, in two independent copies whose MILP files are byte-identical; and
Protein-InDelMaker.

**Not obtained, and the gaps that leaves:**

| Missing | Why | What is unresolved |
|---|---|---|
| ~~IPRO Suite paper, JCC 2015~~ (10.1002/jcc.23796) | **Obtained 2026-11-18** by the user through ISU's Wiley subscription. | Closed. Its equations (1), (2), (4), (5) confirm the formulation in docs/MILP.md verbatim; its input-requirement list and the DM/TM/DP/BA vocabulary are recorded in §1b; both temperature calibrations are verified in §2. |
| ~~ACS Catal 2017 supporting information~~ | **Obtained 2026-11-18** through the user's own Chrome; ACS serves Supporting Information free of charge even off campus. 30 pp., Tables S1–S6 + Figures S1–S5. | Closed. The position-selection rule it documents is now recorded in §1a. |
| **IPRO's rotamer library** | Not in either release; referenced as an external path | Why PyMOL's library is used instead. |
| **A public IPRO+/− release** | `maranasgroup/ipro--` is an empty repository | Indels exist publicly only as Protein-InDelMaker. |
| **PoreDesigner's water-wire probe** | Never deposited; the SI publishes only fitted helix parameters | The example case cannot express the published pore objective. |

Two routes remain open if the user wants these gaps closed: the ISU VPN plus
their own browser for the Wiley PDF and the ACS SI, and the open-access 2018
Penn State dissertation by the Suite paper's second author
(`etda.libraries.psu.edu/catalog/15087mjg5185`), which is the highest-value
free source still unread.

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
10. lucidfrontier45/autodock-vina-agent — the skills-packaging pattern this repo's `SKILL.md` and `AGENTS.md` follow.

### Licensing

No upstream IPRO release carries a `LICENSE` file; the Maranas group's pages
describe the suite as freely available to academic users. **No upstream code is
reproduced in this repo** — the implementation was written from the published
formulations, which is the defensible route given that licence ambiguity. This
repo has no licence chosen yet; add one before sharing it.

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
| 6 | The H-bond well applied to every N/O–N/O pair | Two ARG NH1 at 3.4 Å scored **−0.83 (attractive)** instead of +0.67 — a sign flip on like charges — and a salt bridge was rewarded twice for one contact. The term is now restricted to pairs where neither atom carries a formal charge. |
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
`test_regression_*`. The suite is 39 tests.

**Consequence for anyone comparing runs:** results from a checkout before these
fixes are not trustworthy, particularly any that reported small residues at
buried positions.
