# Expected output — PoreDesigner / OmpF worked example

What you can compare a run of `spec_smoke.yaml` or `spec_full.yaml` against, what
you cannot, and why.

**Reference.** Chowdhury R, Ren T, Shankla M, Decker K, Grisewood M, Prabhakar J,
Baker C, Golbeck JH, Aksimentiev A, Kumar M, Maranas CD. *PoreDesigner for tuning
solute selectivity in a robust and highly permeable outer membrane pore.* Nature
Communications **9**, 3661 (2018). doi:10.1038/s41467-018-06097-1 · PMID 30202038 ·
PMCID PMC6131167.

Read this file before you compare anything. The short version: **the mutation sets,
pore diameters, hydrophobicity scores and design counts are checkable. Nothing about
the energies is.** The paper publishes no per-design energy, names no MILP solver, no
rotamer library and no design-stage force field, deposits no design structures, and
gives no random seed. This example is a faithful restatement of the *design problem*,
not a bit-for-bit rerun of the original software.

---

## 1. The input structure

`2omf_A.pdb` in this directory is the prepared design molecule. Provenance is in its
own `REMARK` records.

| Property | Value |
|---|---|
| Source | `https://files.rcsb.org/download/2OMF.pdb` (274,104 bytes, sha256 `40d7701d…6342d45`) |
| PDB entry | 2OMF, "OMPF PORIN", X-ray, 2.4 Å, one deposited chain (`A`) |
| Kept | chain A, 340 standard amino-acid residues, 2,627 protein atoms |
| Residue range | 1–340, contiguous, no gaps, no insertion codes, no altlocs, one model |
| Not added | no hydrogens, no energy minimisation, no pore-axis reorientation |
| Removed | 128 crystallographic waters (`HOH`), 12 `C8E` detergent fragments (72 atoms, residues 341–354) |

RCSB reports 2,827 deposited atoms for this entry; that total includes the 128 waters
and 72 detergent atoms, so the protein-only count is 2,627. Both numbers are right.

### Why waters and heteroatoms were stripped, and what it costs you

**The decision.** Protein atoms only.

* `C8E` is (hydroxyethyloxy)tri(ethyloxy)octane — octyl-POE, the detergent OmpF was
  crystallised in. It coats the lipid-facing outer belt of the barrel; its nearest
  atom is 19.4 Å from the centroid of the 25 design positions, and none is within
  10 Å. It is a crystallisation additive, not a biological partner, and most energy
  backends have no parameters for it.
* The 128 `HOH` are ordered 2.4 Å crystallographic waters. Three lie within 6 Å of
  the design-position centroid and 14 within 10 Å — i.e. some sit *in the lumen*.
  Kept, they would enter the rotamer problem as fixed, non-rotamer oxygens inside the
  constriction, penalising exactly the lumen-filling rotamers the design is supposed
  to choose, for waters that would in reality be displaced by a Trp side chain.

**The consequence, stated plainly.** PoreDesigner's target molecule was **not** these
waters. It was a single-file *water wire*: one of four representatives clustered from
~100,000 wires harvested from a 10 ns MD run of tetrameric AQP1, placed on the pore
axis and re-docked every iteration. That wire was never deposited. Supplementary
Table 1 publishes only the fitted elliptical-helix parameters of the four
representatives (wire 1: `x = 0.401 sinθ, y = 0.121 cosθ, z = 2.31 θ`; wires 2–4
similar), with **no water count and no O–O spacing**, so a rebuilt wire is a plausible
wire, not their wire. (Note also that Supplementary Table 1 writes *x* with sin and
*y* with cos, the reverse of the formula in the SI prose; transcribed here as printed.)

Therefore:

> **As shipped, this example has no target molecule, so it does not optimise the
> paper's objective.** The paper maximised the OmpF↔water-wire interaction energy —
> which in IPRO's sign convention means *weakening* binding to the wire, so the
> permeating water makes as few hydrogen bonds with the pore wall as possible. With no
> wire, only the geometric half of the problem (shrink the bottleneck below 4 Å with
> long hydrophobic side chains) is expressible. Both specs carry a commented-out
> `anti_target` block showing how to add a wire you build yourself — declared as an
> **anti-target**, not a target, because this tool's `target` role *rewards* better
> interaction and would invert the objective.

Stripping the waters costs you nothing you had: they were never the target.

### Native-identity verification (the check that matters)

The 25 design positions are **derived, not published** (§4). The whole derivation rests
on the residue numbering being right, so it was verified against the structure:

| Check | Result |
|---|---|
| Native identity at all 25 derived design positions | **25/25 match** |
| Native identity at the 26th position (Y310) | **match** (TYR) |
| Structure sequence vs the SI Supplementary Fig. 6 `>2omf` FASTA | **identical, 340/340 characters, no offset** |
| Independent re-derivation of the mutation sets from the SI FASTAs | reproduced §2 exactly; OCD-TFTrp diff = exactly the 25 positions; union over all three designs = 26 |

Verified identities: V18 L20 M38 Y40 R42 E62 N64 R82 L83 L109 G110 D113 M114 L115
E117 F118 G119 G120 D121 A123 Y124 S125 V130 G131 R132, plus Y310.

Independent sanity check on the numbering: the derived set contains R42, E62, R82,
D113, E117, D121 and R132 — exactly the canonical OmpF constriction charge cluster
from the porin literature. **If your own prep reports a mismatch at any of these
positions, stop: your numbering is off, and every mutation label downstream is wrong.**

---

## 2. The three published designs

Derived by gapless alignment of the four 340-aa sequences in SI Supplementary Fig. 6
(all four are exactly 340 residues, so the alignment is unambiguous). These sequences
are the paper's own published output and are the primary expected result.

| Design | Mutations | Mutation list (mature OmpF = 2OMF chain A numbering) |
|---|---|---|
| **CSD** — cork-screw design; expressed and tested; rejects glucose | **13** | `V18W L20W M38W Y40W N64F R82W G110W E117A G119W A123W S125W G131F Y310W` |
| **UCD** — uniform closure design; expressed and tested; rejects NaCl | **15** | `V18W L20W M38W Y40W E62P N64W R82W G110W D113W E117W F118G G119W A123W G131F Y310G` |
| **OCD-TFTrp** — off-centre closure, "total fill" Trp-only control; rejects sucrose | **25** | `V18W L20W M38W Y40W R42W E62W N64W R82W L83W L109W G110W D113W M114W L115W E117W F118W G119W G120W D121W A123W Y124W S125W V130W G131W R132W` |
| OCD-TFPhe — Phe-only twin control | (25, all → F by construction) | **sequence not published** |

Position by position (every position mutated in at least one design):

| Pos | WT | CSD | UCD | OCD-TFTrp | | Pos | WT | CSD | UCD | OCD-TFTrp |
|---:|:--:|:--:|:--:|:--:|---|---:|:--:|:--:|:--:|:--:|
| 18 | V | W | W | W | | 118 | F | F | **G** | W |
| 20 | L | W | W | W | | 119 | G | W | W | W |
| 38 | M | W | W | W | | 120 | G | G | G | W |
| 40 | Y | W | W | W | | 121 | D | D | D | W |
| 42 | R | R | R | W | | 123 | A | W | W | W |
| 62 | E | E | **P** | W | | 124 | Y | Y | Y | W |
| 64 | N | **F** | W | W | | 125 | S | W | S | W |
| 82 | R | W | W | W | | 130 | V | V | V | W |
| 83 | L | L | L | W | | 131 | G | **F** | **F** | W |
| 109 | L | L | L | W | | 132 | R | R | R | W |
| 110 | G | W | W | W | | **310** | **Y** | **W** | **G** | Y |
| 113 | D | D | W | W | | | | | | |
| 114 | M | M | M | W | | | | | | |
| 115 | L | L | L | W | | | | | | |
| 117 | E | **A** | W | W | | | | | | |

**Consensus across all three tested designs — 8 positions:**
`V18W L20W M38W Y40W R82W G110W G119W A123W`. All three also change position 64, 117
and 131. This consensus is the most useful thing to compare a run against.

**The smoke spec's four positions (A18, A20, A38, A40) are all in that consensus**, and
the published answer at every one of them, in every tested design, is **Trp**.

---

## 3. Published measured and computed properties

| Design | Constriction, major × minor (Å) | Solute exclusion limit | Inner-wall ΔG | Outer-wall ΔG | Overall ΔG | Single-channel permeability (cm³/s) |
|---|---|---|---|---|---|---|
| **Wild-type OmpF** | **11 × 7** (area 60.47 Å²) | ~**600 Da** (rejects PEG600) | −61.3 | −68.5 | −129.8 | 1.0 ± 0.23 × 10⁻¹² |
| **OCD** | **3.54 × 3.25** | ~**342 Da** (rejects sucrose) | −69.2 | −81.2 | −150.4 | > WT, ≈ UCD |
| **CSD** | **3.18 × 3.12** | ~**180 Da** (rejects glucose) | −97.7 | −76.5 | −174.2 | **4.4 ± 0.93 × 10⁻¹²** (highest) |
| **UCD** | **3.05 × 3.01** | ~**58.5 Da** (rejects NaCl) | −65.6 | −71.3 | −136.9 | > WT, ≈ OCD |

Hydrophobicity is a Kyte–Doolittle ΔG(water→ethanol) sum over pore-wall residues, in
kcal/mol; a larger negative value is more hydrophobic. A residue counts as *inner* wall
if the pore centre is further from its Cα than from its Cβ, else *outer*.

Three caveats on this table, all from the source:

1. **The diameters are post-MD.** They are measured on the final 100 frames of the MD
   trajectories, not at design time. The design-time criterion was simply "bottleneck
   below 4 Å"; Fig. 5e plots the pre-MD and post-MD diameters as separate series. A
   design-time bottleneck from your run is **not** directly comparable to 3.18 × 3.12 Å.
   Compare it to the `< 4 Å` constraint instead.
2. **Table 1's caption says "Ranges of values"** for the three hydrophobicity columns,
   so treat these as representative values for each design class, not exact point
   predictions.
3. Predicted hydrophobicity orders CSD > OCD > UCD > WT, and the measured
   single-channel permeability orders CSD > OCD ≈ UCD > WT. The ordering is the
   result; the individual numbers are a weaker claim than the ordering.

**Design-count bookkeeping** (arithmetic you can check without any compute):

| Quantity | Value |
|---|---|
| Designs with bottleneck < 4 Å | 40 = 2 OCD + 31 UCD + 7 CSD |
| Further designs spanning 4–10 Å in 1 Å bins | 17 = 10 OCD + 3 UCD + 4 CSD |
| Expressed and tested | 3 (OCD, CSD, UCD) + WT |

Selection rule for which design went to the bench: *the smallest predicted pore size
design from each type.*

---

## 4. What you can and cannot reproduce

### You can reasonably expect to reproduce

* **The design problem itself** — the molecule, the 25 positions, the 9-residue
  alphabet, the ≥50 % long-side-chain constraint, the sub-4 Å acceptance filter, the
  Metropolis criterion at 3640 K. These are all fully specified in the paper and the SI
  and are transcribed into the two specs here.
* **Qualitative convergence on the published chemistry** — a strong bias to Trp, long
  residues dominating the constriction, mutations concentrated in the published
  consensus positions.
* **The OCD-TFTrp control exactly.** Collapsing the alphabet to `[TRP]` removes the
  amino-acid choice entirely: the composition constraint becomes trivial and the MILP
  degenerates to rotamer packing of 25 tryptophans. The 25-mutation answer is published
  and is independent of the stochastic search. This is the only sequence-level
  assertion in this example that should hold run after run.
* **The wild-type pore geometry**, 11 × 7 Å and 60.47 Å², from a pore-profile pass over
  `2omf_A.pdb` alone. No design, no MD, seconds of compute.
* **Table 1's hydrophobicity numbers from the published sequences.** This is a pure
  sequence-plus-structure → number calculation, fully determined by published data, so
  it genuinely tests a hydrophobicity implementation without running the design loop.

### You cannot reproduce, and should not claim to

* **Any energy.** There are **no published per-design interaction energies** — not in
  the main text, not in Table 1, not in the SI. The only computed scalars published per
  design are pore diameters and hydrophobicity scores. Nothing in your run's energy
  column has a published counterpart.
* **The original software's numbers.** The paper names **no MILP solver**, **no rotamer
  library** and **no design-stage force field** (CHARMM36/CMAP appears only in the
  downstream NAMD validation, not in the design loop), and pins **no IPRO version or
  commit** — only a citation to the IPRO suite paper, Pantazes et al., *J Comput Chem*
  36:251–263 (2015). This tool uses a different energy backend, a different rotamer
  library and a different docking program. Energies are therefore not on the same scale
  as the original's, and *the argmin can differ even where the physics agrees*.
* **The exact mutation sets of CSD or UCD.** The search is stochastic in four separate
  ways: random choice of perturbed position, random φ/ψ offsets, Metropolis acceptance,
  and a per-iteration cap `M` on mutations that the paper **draws randomly from an
  unspecified distribution**. No seed is published. The paper itself reports 31 distinct
  UCD designs and 7 CSD designs — the published CSD and UCD are *one member each of a
  family*, chosen as the smallest pore of its class. Reproducing one specific member is
  not a meaningful target.
* **The objective, as shipped.** The water wire is absent (§1), so the DM↔TM
  interaction term the paper optimised is not being computed at all.
* **The design positions as published**, because they were never published. The paper
  says "the 25 pore constriction residues" and Supplementary Fig. 2b highlights them in
  a rendered image; no residue number appears in the text, Methods or SI. The list used
  here was recovered by diffing the published OCD-TFTrp sequence against wild type
  (that design mutates exactly those 25 positions to Trp, by construction) and
  re-verified against the structure. It is a sound derivation, independently
  corroborated by the charge-cluster check — **but it is a derivation, and a different
  reader could come to a different list.**
* **Anything from the deposited data**, because there is none. Data availability is
  "available from the corresponding authors upon reasonable request"; no design PDBs,
  no input decks, no energy tables were deposited.

### Known discrepancies carried forward from the source

These are unresolved in the paper. They are recorded rather than smoothed over, because
each one can make a run look wrong when it is not.

1. **Y310 — a 26th position.** CSD carries `Y310W` and UCD carries `Y310G`, both
   *outside* the 25-position OCD-TFTrp set. Y310 is a known pore-lining residue. Either
   the general runs used a 26th design position, or "25" counts only what Supplementary
   Fig. 2b highlights. The union of positions mutated across the three tested designs is
   **26, not 25**. The specs here run the 25 and flag 310 as an optional addition; say
   which you ran.
2. **Glycine appears in a published design but is not in the allowed alphabet.** UCD's
   `F118G` and `Y310G` introduce Gly, which set *A* = {W,F,Y,M,I,L,V,P,A} excludes.
   Either the tested construct carries changes that did not come from the design run, or
   Gly was permitted as a shrink/revert option not captured by *A* as written. A run
   constrained to *A* **cannot** produce `F118G`, and that is not a bug in your run.
3. **The composition constraint is stated twice, inconsistently.** The SI prose says "a
   minimum of 50 % of the newly introduced residues had to be either Trp, Phe or Tyr";
   formal MILP constraint 4 counts **Met** as long too. The specs here implement
   constraint 4. Dropping MET from `long_residues` is a defensible alternative and will
   give different answers.
4. **Figure 2 does not show the tested UCD.** Its panels label an `R82L` mutation
   relieving a clash with `Trp62`. The tested UCD sequence carries `R82W` and `E62P`.
   Fig. 2 illustrates a *different* member of the 31-design UCD family. **Do not use
   Fig. 2 as an expected output.**
5. **Supplementary Table 1's water-wire equations** swap sin and cos relative to the SI
   prose formula (§1).

---

## 5. How to check your run

Concrete comparisons, cheapest and most diagnostic first. Checks 1–4 are the ones that
mean something for the smoke spec.

**1. Is the design space what you think it is?** Print `spec.summary()`. For the smoke
spec: 4 positions, 9 allowed each, 6,561 sequences. For the full spec: 25 positions,
7.2 × 10²³. If you see 18 allowed residues at a position, an `allowed` list went
missing and defaulted to the `standard` group — the run is answering a different
question.

**2. Did the composition constraint hold?** At least 50 % of design positions should
carry a LONG residue (W/F/Y/M): ≥ 2 of 4 for the smoke spec, ≥ 13 of 25 for the full
one. A violation is a bug in the `min_class` constraint, not a scientific finding.
For reference, the published designs come in far above the floor: **CSD 12 of 13
mutations are long (92 %), UCD 12 of 15 (80 %), OCD-TFTrp 25 of 25 (100 %)**. Both
readings of the constraint (§4, discrepancy 3) give the same count here, because neither
non-TF design introduces a Met.

**3. Do the chosen identities concentrate in the LONG set?** This is the paper's central
qualitative claim, and it is checkable without any published energy. Count the LONG
fraction among *mutated* positions and compare with the published 80–92 %. A run that satisfies the
constraint at exactly the 50 % floor and fills the rest with Ala/Val has not reproduced
the paper's chemistry even if it passes check 2. Expect Trp specifically to dominate:
10 of 13 CSD mutations (77 %) and 11 of 15 UCD mutations (73 %) are Trp, and the rest
are almost all Phe.

**4. Are the mutations drawn from the published set?** Two levels:
   * *Identity at a consensus position.* `V18W L20W M38W Y40W R82W G110W G119W A123W`
     appear in all three tested designs. For the smoke spec, all four positions are
     consensus positions and the published answer at each is **Trp** — so "how many of
     A18, A20, A38, A40 came out Trp" is a clean 0–4 score against the paper.
   * *Set overlap.* For the full spec, intersect your mutated-position set with the 25
     and with each published design's set. The overlap should be substantial;
     position-for-position identity should **not** be expected (§4).

**4b. What the shipped smoke run actually produces.** Recorded so you can tell
a working install from a result: `spec_smoke.yaml` with its shipped seed accepts
3 iterations and reports **zero mutations** — 0 of 4 positions come out Trp. The
native residues win because nothing offsets the packing bias against large side
chains (no reference-energy term; the run says so in its `notes`). That is a
successful smoke test and **not** a reproduction of anything in the paper. For a
run that mutates, use `spec_full.yaml` or set
`goal_options.reference_energies`.

**5. Did the geometric filter actually bite?** Every accepted design should have a
bottleneck below the `pore_diameter_max` of 4.0 Å, and rejected proposals should
dominate the iteration log. If almost nothing is rejected, the pore measurement is not
running. Compare to the design-time constraint, not to the post-MD 3.0–3.6 Å diameters
(§3, caveat 1).

> **Check the axis before you believe any bottleneck.** The axis is fitted through
> the design positions' centroid, so it only follows the channel when those
> positions *ring* the lumen. The 25-position `spec_full.yaml` is sound — the axis
> origin clears the nearest van der Waals surface by **2.80 Å** in the input
> structure, where a water needs 1.4 Å — but the 4-position `spec_smoke.yaml`,
> whose positions all line one wall, puts that centroid **inside the wall**
> (clearance **0.30 Å**) and reports a **1.53 Å** "bottleneck" that is not a pore
> measurement at all. That number silently *passes* the 4 Å filter, so "almost
> nothing is rejected" can mean the measurement is wrong rather than absent, and a
> smoke run must never be read as having achieved a sub-4 Å pore. The run now says
> so outright ("THE PORE RADIUS HERE IS NOT A PORE RADIUS"); the remedy for such a
> spec is `goal_options.pore_axis`, or positions on the opposite wall.
> `docs/DESIGN.md` §7h records how the clearance test was fixed — it had been
> measured on the candidate's own atoms, where a successful narrowing design
> lowers it too, so it could not tell a buried axis from a design that worked.

> **Measured 2026-11-20, and it is a negative result.** This spec was run twice
> (81 and 21 iterations, fixed-backbone, built-in energy, seed 1) and accepted
> **nothing** either time. No candidate brought the bottleneck below 5.65 Å
> against a filter that requires under 4.00 Å, and widening
> `design_per_iteration` from `[1, 3]` to `[12, 25]` made the pore wider rather
> than narrower. So the design-stage comparisons below have not been
> demonstrated: this example currently reproduces the wild-type measurement and
> the bookkeeping, not the designs. `docs/DESIGN.md` §7m has the numbers and
> what has not been ruled out.

**6. Negative controls — things that must never appear.** No charged or polar residue,
no Gly and no Cys at any design position: the alphabet forbids them (MILP constraint 7).
If one appears, the allowed-residue set is not reaching the solver. Likewise, no residue
should change outside the declared design positions.

**7. Structure-free checks that need no run at all.** The wild-type pore from
`2omf_A.pdb` is published as 11 × 7 Å, area 60.47 Å²; this tool's inscribed-circle
measurement of **the deposited coordinates**, with the 25 design positions
defining the axis, is **8.07 Å** (bounded by the narrow direction, so ~7 Å is the
right comparison). Do not use `score`'s 6.24 Å for this check: that is the
starting *design*, because this spec's alphabet excludes the native residue at 13
of the 25 positions and scoring substitutes the best allowed rotamer there — the
run says so in a note, and §7h records the section that misread it. Table 1's four hydrophobicity rows
should be recoverable from the four published sequences. The design counts should add up:
40 = 2 + 31 + 7 and 17 = 10 + 3 + 4. These test the measurement and bookkeeping layers
on published data, independently of the stochastic search, which makes them the most
trustworthy assertions in this example.

**8. What to report.** Whichever backends resolved (`python scripts/doctor.py`), the
seed, whether the run ended on `stop_after_accepted` or hit the `iterations` cap, and
whether a water wire was present. A run that ended on the iteration cap did not meet the
paper's published termination condition of 30 accepted redesigns, and should not be
described as having completed the campaign.

**What would falsify this example rather than your run:** a native-identity mismatch at
any of the 25 positions (§1), or a failure to re-derive the §2 mutation sets from the SI
FASTAs. Both are checked by the prep step and both currently pass.

---

## 6. Provenance

* Design positions, mutation sets and the §3 tables: derived from SI Supplementary
  Fig. 6 sequences and Table 1 of the paper, re-verified in this repository against
  `2omf_A.pdb`.
* Local copies of the article PDF, the SI PDF and the Europe PMC full-text XML are kept
  under `refs/papers/` (git-ignored, not redistributed). The raw RCSB download is under
  `refs/pdb/` (git-ignored). Only the prepared structure, the two specs and this file
  are committed.
* PoreDesigner itself is distributed from `github.com/maranasgroup` and
  `maranasgroup.com/software.htm`, with a separate PoreAnalyzer module. No version,
  commit or release tag is given in the paper.
