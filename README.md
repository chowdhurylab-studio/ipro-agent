# ipro-agent

An independent, agent-drivable reimplementation of the published **IPRO**
procedure — Iterative Protein Redesign and Optimization.

> **Unsupported research code.** Maintained by the
> [Chowdhury Lab](https://github.com/chowdhurylab-studio); please use GitHub
> issues. It implements a method published by the Maranas group (Penn State) —
> see [Citing](#citing) — and is not their software, not endorsed by them, and
> contains none of their code.

IPRO as published is a suite of programs wired to specific case studies: the
molecules, the design positions and the objective live in a per-case Python
input deck. This repo keeps the machinery and replaces the wiring with a
declarative problem statement, so the same optimizer can be pointed at an
arbitrary protein and an arbitrary stated goal — by a person or by an LLM agent.

You give it:

- a **goal** in plain language, mapped to one of five objectives;
- **structures** of the interacting partners (protein, peptide, small molecule,
  transition-state analogue);
- the **positions** where mutation is allowed and the **amino acids** allowed at
  each.

It writes the rotamer-selection MILP that encodes those constraints, then
iterates: perturb → choose optimal rotamers → dock → relax → accept or reject
on energy. Out comes a design sequence, the per-iteration energies, and an
ensemble of the best designs.

## Quick start

**One install step, and it is not optional.** Rotamers come only from PyMOL's
mutagenesis wizard, and `pymol-open-source` is conda-only — so `pip install`
alone cannot give you a working design pipeline:

```bash
conda env create -f environment.yml && conda activate ipro-agent
```

Then check what resolved. Everything below runs from the repo root; nothing
needs installing the package itself:

```bash
python3 scripts/doctor.py --require pymol
```

```bash
python3 -m ipro_agent validate examples/ompf_pore/spec_smoke.yaml
```

```bash
python3 -m ipro_agent run examples/ompf_pore/spec_smoke.yaml --run-dir runs/smoke
```

```bash
python3 -m ipro_agent status runs/smoke
```

**What that run actually produces**, so you can tell a working install from a
real result: with the shipped seed it accepts 3 iterations and reports **zero
mutations** — the native residues win at those four positions, because no
reference-energy term offsets the packing bias against large side chains (the
run says so in its `notes`). That is a successful smoke test, not a design. For
a run that does mutate, see `examples/ompf_pore/spec_full.yaml` (25 positions,
~85 s/iteration) or set `goal_options.reference_energies`.

## For agents

Clone the repo, create the conda environment above, open it with an agent, and
ask in plain language — "redesign this thioesterase to prefer C12 over C14, you
may change residues 110, 113 and 117" — and the skill at
[.claude/skills/ipro-agent/SKILL.md](.claude/skills/ipro-agent/SKILL.md) loads
itself from its description. Beyond that one environment step there is nothing
for the agent to configure: it should run `scripts/doctor.py` first and report
the mode it finds.

To read it yourself, it is the operating manual: goal selection, spec format,
encoding user constraints into the integer program, the degraded modes, and a
validation checklist to work through before reporting a design.

## Documentation

| File | Contents |
|---|---|
| [.claude/skills/ipro-agent/SKILL.md](.claude/skills/ipro-agent/SKILL.md) | how an agent drives this tool (auto-loaded) |
| [docs/DESIGN.md](docs/DESIGN.md) | every capability traced to its paper and equation, and every deviation |
| [docs/MILP.md](docs/MILP.md) | the integer program in full |
| [docs/INSTALL.md](docs/INSTALL.md) | dependencies and what each one unlocks |
| [examples/ompf_pore/](examples/ompf_pore/) | worked case, with an honest `EXPECTED.md` |

## Dependencies

**Required:** Python ≥ 3.10, numpy, Biopython, PyYAML — and **PyMOL**, which
has no fallback. Rotamers come only from its mutagenesis wizard, there is no
built-in rotamer builder, and `backends.mutate: builtin` is refused by the
config. Without it `validate`, `positions`, `milp`, `score` and `goals` work and
`run` cannot produce a design. `conda env create -f environment.yml` installs
all five.

Genuinely optional — each resolved at run time with a documented fallback:

| Tool | Used for | Without it |
|---|---|---|
| **PyRosetta** | backbone relaxation, scoring | loop runs `fixed-backbone`; energies come from a screening function |
| **AutoDock Vina** | re-docking partners after mutation | input poses are kept |
| **MILP solver** (CPLEX, Gurobi, CBC, HiGHS, GLPK) | proving the optimal packing | a heuristic that does not prove optimality |

`scripts/doctor.py` reports what resolved and what a run will actually use.

## Status and honesty

This is **research code and an independent reimplementation**. It was written
from the published formulations; the released upstream source was read for
behaviour the papers leave ambiguous, and every value taken from it is cited as
such in `docs/DESIGN.md` §2 and `docs/MILP.md`. **No upstream IPRO code is
copied.** It substitutes an open stack for the original's CHARMM, GAMS/CPLEX and
in-house Monte-Carlo docking, so **energies are not comparable to the numbers in
the IPRO papers** and identical designs should not be expected.

**What it has actually been run on.** Every end-to-end run in this repo has been
on one chain of PDB **2OMF**, with **1CRN** as a test fixture. No other protein
has been tried here, and **no design from this code has been synthesised or
tested experimentally.** The checks listed below are self-consistency checks of
the optimizer, not validation against experiment.

What has been verified here: the MILP reproduces exhaustive enumeration exactly
on a test case (including with a class-cardinality constraint), the integer cuts
provably prevent revisiting a sequence, and the Metropolis temperature
reproduces IPRO's published calibration. The PyRosetta and AutoDock Vina paths
**have now been executed** — both were installed deliberately to do so, and
between them that found seven defects, because neither path had ever run (see
`docs/DESIGN.md` §7e). What remains unexercised is listed in
`ipro_agent/energy/pyrosetta_backend.UNEXERCISED`, which a test keeps honest.

**One known defect is open, and it can produce a wrong answer that looks
right.** The channel axis is fitted through the centroid of the design
positions, so it only follows the channel when those positions *ring* the lumen.
When they do not — four positions along one wall, in the shipped smoke example —
the axis runs inside the protein and the reported "bottleneck" is a gap beside a
wall, which then **passes** PoreDesigner's published sub-4 Å filter. The run
says so in its notes and reports the axis clearance, but any pore radius from a
spec whose positions do not surround the channel must be treated as meaningless;
give `goal_options.pore_axis` explicitly instead. `docs/DESIGN.md` §7d has the
measurements and the three candidate fixes that were tried and rejected.

Objectives of this kind reproduce the rank and sign of a change far better than
its magnitude; published hit rates for this family of methods are roughly
24–43%. Every design is a hypothesis for the bench.

## Responsible use

This is a research tool that proposes mutations. It does not know what it is
being pointed at, and a coding agent driving it knows even less.

- **Research use, on targets you are authorised to work on.** Every design is a
  hypothesis for the bench, not a validated construct — published hit rates for
  this family of methods are roughly 24–43%.
- **Some targets need a human in the loop before anything is run**, not after
  the result arrives: toxins, virulence factors and select agents; anything
  intended to change host range, transmissibility, immune escape or drug
  resistance; and any sequence heading for synthesis. If the goal is to make a
  protein *better at harming*, that is not a use this repo supports.
- **An agent must not relax the user's stated constraints to make a failing run
  succeed.** The troubleshooting tables offer knobs — widen `allowed`, loosen a
  catalytic-distance restraint, raise `max_self_energy` — and every one of them
  changes the scientific question. Report the change; do not absorb it.
- **Report the mode and the notes with every design** (a house rule in
  `AGENTS.md`). A run in a degraded mode, or one where nothing was accepted, is
  not a design and must not be presented as one.
- Screening energies from the built-in backend are **not** kcal/mol comparable
  to the papers, and no design from this code has been tested experimentally.

## Citing

For the **implementation**, cite this repository — `CITATION.cff` carries the
metadata, and GitHub's "Cite this repository" button renders it.

For the **method**, cite the original work — see `docs/DESIGN.md` for the full
list with DOIs, chiefly:

- Saraf, Moore, Goodey, Cao, Benkovic, Maranas. *Biophys J* **90**:4167 (2006) — IPRO.
- Pantazes, Grisewood, Li, Gifford, Maranas. *J Comput Chem* **36**:251 (2015) — the IPRO suite.
- Chowdhury et al. *Nat Commun* **9**:3661 (2018) — PoreDesigner.
- Chowdhury et al. *Structure* **28**:1344 (2020) — IPRO+/−, insertions and deletions.
- Grisewood et al. *ACS Catal* **7**:3837 (2017) — specificity redesign.
- Grisewood et al. *PLoS ONE* **8**:e75358 (2013) — OptZyme.

## Licence

**MIT** — see [`LICENSE`](LICENSE).

That covers *this implementation only*: the code, tests and documentation
written here. Three things in this tree are deliberately **not** covered, and
the licence file says so in its own words:

| | |
|---|---|
| **The method** | Not ours and not licensed by us. Cite the papers above for the method and this repo only for the implementation. |
| **`refs/`** | Publisher PDFs and clones of upstream repositories, kept locally for provenance and **git-ignored** — they are not in this repository and are not redistributed. |
| **Optional dependencies** | PyMOL, AutoDock Vina, Open Babel, PyRosetta and any MILP solver are separately licensed and separately installed. **PyRosetta requires its own licence**; see [`docs/INSTALL.md`](docs/INSTALL.md). |

**No upstream IPRO code is copied here**, which is what makes publishing this
defensible: the upstream releases carry no `LICENSE` file at all, and the
Maranas group's pages describe the suite as freely available to academic users.
This is an *independent* reimplementation rather than a fork — written from the
papers, with the released source read only for behaviour the papers leave
ambiguous (chiefly a handful of numeric defaults), every such value cited as
coming from the source in `docs/DESIGN.md` §2 and `docs/MILP.md`.
`docs/DESIGN.md` §6 maps every capability back to the paper and equation it came
from, so the provenance of each piece is checkable.
