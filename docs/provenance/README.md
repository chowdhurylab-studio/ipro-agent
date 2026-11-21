# Provenance

Evidence behind claims that ship in `README.md`, `LICENSE` and `docs/DESIGN.md`.
These files were produced during the audits recorded in DESIGN.md §7–§7h and
kept here because the claims they back are public while the measurements were
living in a temporary directory.

| file | backs |
|---|---|
| `audit_paper_fidelity.md` | the 41-agent paper audit: all 19 confirmed findings with their measurements, plus the 18 dropped and why. DESIGN.md §7h summarises it. |
| `audit_edge_cases.md` | the 51-agent edge-case audit: 32 confirmed findings, 15 dropped. |
| `ngram.py`, `cleanroom-grep.txt`, `leaks-refutation.txt` | the independence claim in `LICENSE` and `README.md` — no upstream code is copied. |
| `tour_capture.txt` | the real command output `docs/tour.gif` was rendered from. Re-render with `scripts/make_tour_gif.py`. |
| `relrun/score/` (at the repository root) | that PyMOL's mutagenesis wizard really ran here, which `docs/INSTALL.md` asserts as "verified on this machine": `logs/rotamers.json` records pymol-open-source **3.1.0** harvesting 9 rotamers at each of A18, A20, A38 and A40 — 36 in all — off `examples/ompf_pore/2omf_A.pdb`, with the native identities VAL/LEU/MET/TYR and an empty `errors` map. It is a **4-position smoke run**, so it is *not* the §7m evidence a reader might assume: §7m is two 25-position runs of 81 and 21 iterations, and this run's `best/`, `ensemble/` and `structures/` are empty. Kept because the harvest is the one step with no fallback. |

`scripts/measurements/` holds two scripts that produce numbers quoted in
DESIGN.md §7h which have no regression test of their own:

- `perturbation_overlap.py` — P(perturbed region touches a design position),
  0.1244 unanchored against 1.0000 anchored.
- `ring_flatness.py` — 0.531 for the sound 25-position axis against 0.053 for
  the unsound 4-position one, which is why flatness is *not* used as the
  discriminator (it separates them the wrong way).
