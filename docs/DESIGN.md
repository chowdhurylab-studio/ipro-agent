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
| Metropolis accept/reject at T = 3640 K | `goals.metropolis_accept` | IPRO Suite, `P = exp(−ΔE/k_BT)`; the source's equivalent `window = −R·T·ln(U)` form |
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
| `temperature_k` | 3640 K | IPRO Suite | Calibrated so a 10 kcal/mol worsening is accepted ~25% of the time. **Verified:** with `R = 1.987e-3 kcal/(mol·K)`, `exp(−10/(R·3640)) = 0.2509`. One secondary source quotes `k = 0.333e-3 cal/K`, which gives `P = 0.0003` and is inconsistent with the stated calibration; it was not used. Checked in `tests/test_ipro.py`. |
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
| **Reference energies** | Mechanism provided, no values. | IPRO's 20-entry table was not recovered. Inventing values would put unsourced numbers at the centre of the objective. Without them, a rigid-backbone search of buried positions drifts to the smallest allowed residue — the loop emits a note when this applies, and the all-alanine answer it produces must not be reported as a design. |
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
| **IPRO Suite paper, JCC 2015** (10.1002/jcc.23796) | Closed access; Wiley returned 403 with a bot challenge; this machine is off campus with no VPN up | The full input-language specification and the rotamer MILP's published problem size. The method's technical content was recovered from CC-BY surrogates — chiefly the PoreDesigner SI — and from the source. |
| **ACS Catal 2017 supporting information** (Tables S1–S4) | `pubs.acs.org` returned Cloudflare 403; not open access on PMC | Per-design energies and the full position-selection tables for that case. |
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
