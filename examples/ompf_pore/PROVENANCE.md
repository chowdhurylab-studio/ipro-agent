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
* **The full 25-position native-identity check was never completed.** The
  process assembling these files was interrupted before it reported that
  check, so only the four smoke-spec positions have been confirmed above.
  Before relying on `spec_full.yaml`, run
  `python3 -m ipro_agent validate examples/ompf_pore/spec_full.yaml` and
  confirm every native identity matches the published mutation list.
* **The water-wire probe molecule is not here and was never deposited.**
  Without it the `reshape_channel` objective is degenerate: the run optimizes
  packing subject to the composition constraint. The loop says so in its
  notes. This is weaker than the published objective.
* **No pore-geometry measurement is implemented**, so the paper's hard
  bottleneck filter (discard any design whose constriction is outside the
  requested window) cannot be applied. The `pore_diameter_*` options in the
  specs are inert.
* **No published per-design energies exist** to compare against. The paper
  reports mutation sets, pore diameters and hydrophobicity scores only.

## Consequence

Treat this as a **worked example of the machinery**, and as a faithful
transcription of the paper's *design space and constraints* — not as a
reproduction of the paper's results. `EXPECTED.md` sets out which comparisons
are meaningful.
