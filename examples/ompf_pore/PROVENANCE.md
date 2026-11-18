# Provenance and verification status of this example

**Read this before trusting the design-position list or the mutation tables in
`EXPECTED.md`.**

## How these files were produced

`2omf_A.pdb`, `spec_smoke.yaml`, `spec_full.yaml` and `EXPECTED.md` were
assembled from the PoreDesigner paper (Chowdhury et al., *Nat Commun* 9:3661,
2018) and its supplementary information.

## What is verified

* Both specs parse against `DesignSpec` and pass
  `python3 -m ipro_agent validate`.
* `validate` reads the native residue at each design position directly from
  `2omf_A.pdb`. For the four smoke-spec positions it independently confirms
  **A18 VAL, A20 LEU, A38 MET, A40 TYR**, which are consistent with the
  published mutations V18W, L20W, M38W, Y40W. That is a real check of the
  residue numbering, performed by the tool rather than asserted.
* `spec_smoke.yaml` runs end to end and the implied PoreDesigner constraint
  (`min_class`, fraction 0.5, over W/F/Y/M) is present in the generated LP.

## What is NOT verified

* **The 25-position design list was derived, not published.** The paper does
  not print its design positions; they were reconstructed by aligning the
  sequences in the supplementary figures. Two of the published designs also
  carry a mutation at Y310, outside the 25.
* **The 25-position native-identity check is complete.**
  `python3 -m ipro_agent validate examples/ompf_pore/spec_full.yaml` passes, and
  reports that the native residue is not an allowed choice at 13 of the 25
  positions — which is intended, since the published designs mutate them, and
  the run's notes say so. Re-run it after any edit to the spec.
* **The water-wire probe molecule is not here and was never deposited.**
  Without it the `reshape_channel` objective is degenerate: the run optimizes
  packing subject to the composition constraint. The loop says so in its
  notes. This is weaker than the published objective.
* **Pore geometry IS measured now, and the paper's hard filter is applied** —
  `pore_diameter_min`/`pore_diameter_max` reject a design before its energy is
  considered, as published (see docs/DESIGN.md §1h). Two caveats carry:
  the inscribed-circle measurement is this tool's, not the paper's ellipse fit;
  and **the fitted axis can miss the lumen entirely** when the design positions
  do not ring the channel — in the 4-position smoke spec it runs inside the wall
  (0.30 Å clearance where a water needs 1.4 Å) and reports a 1.53 Å
  "bottleneck" that *passes* the sub-4 Å filter. The run reports the clearance
  and says it cannot vouch for the radius. docs/DESIGN.md §7d has the
  measurements; give `goal_options.pore_axis` to measure it directly.
* **Where `2omf_A.pdb` came from.** RCSB PDB entry **2OMF** (OmpF porin,
  Cowan et al. 1992), downloaded from `https://files.rcsb.org/download/2OMF.pdb`
  and reduced to chain A. PDB coordinate files are distributed by the RCSB
  without copyright restriction (CC0), which is why it can ship here; the MIT
  licence of this repo does not purport to cover it.

* **No published per-design energies exist** to compare against. The paper
  reports mutation sets, pore diameters and hydrophobicity scores only.

## Consequence

Treat this as a **worked example of the machinery**, and as a faithful
transcription of the paper's *design space and constraints* — not as a
reproduction of the paper's results. `EXPECTED.md` sets out which comparisons
are meaningful.

## What the smoke spec actually returns

With the shipped settings it **retains all four native residues** (V18, L20,
M38, Y40) and improves the score only by repacking their rotamers. That is a
real outcome, not a failure: all four natives are inside the paper's nine-residue
alphabet, so "do not mutate" is a legal answer, and on a fixed backbone with no
probe molecule and no reference-energy table there is nothing pushing toward a
larger side chain.

It is *not* the published answer (all four go to Trp in CSD, UCD and
OCD-TFTrp). Reproducing that direction needs the pieces this tool does not
have: the water-wire probe, a pore-geometry filter, and a reference-energy term
or PyRosetta. Treat the smoke run as evidence the machinery turns over, and
`EXPECTED.md` for what a real comparison would require.
