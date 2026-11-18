# ipro-agent

A generalized, agent-drivable implementation of **IPRO** — Iterative Protein
Redesign and Optimization (Maranas group, Penn State).

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

```bash
python3 scripts/doctor.py
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

## For agents

Read [SKILL.md](SKILL.md). It is the operating manual: goal selection, spec
format, encoding user constraints into the integer program, the degraded modes,
and a validation checklist to work through before reporting a design.

## Documentation

| File | Contents |
|---|---|
| [SKILL.md](SKILL.md) | how an agent drives this tool |
| [docs/DESIGN.md](docs/DESIGN.md) | every capability traced to its paper and equation, and every deviation |
| [docs/MILP.md](docs/MILP.md) | the integer program in full |
| [docs/INSTALL.md](docs/INSTALL.md) | dependencies and what each one unlocks |
| [examples/ompf_pore/](examples/ompf_pore/) | worked case, with an honest `EXPECTED.md` |

## Dependencies

Required: Python ≥ 3.10, numpy, Biopython, PyYAML.

Recommended, each resolved at run time with a documented fallback:

| Tool | Used for | Without it |
|---|---|---|
| **PyMOL** | side-chain rotamers and mutagenesis | no rotamers — effectively required |
| **PyRosetta** | backbone relaxation, scoring | loop runs `fixed-backbone`; energies come from a screening function |
| **AutoDock Vina** | re-docking partners after mutation | input poses are kept |
| **MILP solver** (CPLEX, Gurobi, CBC, HiGHS, GLPK) | proving the optimal packing | a heuristic that does not prove optimality |

`scripts/doctor.py` reports what resolved and what a run will actually use.

## Status and honesty

This is **research code and a clean-room reimplementation**, written from the
published formulations rather than from the Maranas group's source. It
substitutes an open stack for the original's CHARMM, GAMS/CPLEX and in-house
Monte-Carlo docking, so **energies are not comparable to the numbers in the
IPRO papers** and identical designs should not be expected.

What has been verified here: the MILP reproduces exhaustive enumeration exactly
on a test case (including with a class-cardinality constraint), the integer cuts
provably prevent revisiting a sequence, and the Metropolis temperature
reproduces IPRO's published calibration. What has **not** been verified: the
PyRosetta and Vina paths, since neither is installed in this environment.

Objectives of this kind reproduce the rank and sign of a change far better than
its magnitude; published hit rates for this family of methods are roughly
24–43%. Every design is a hypothesis for the bench.

## Citing

Cite the original work for the method — see `docs/DESIGN.md` for the full list
with DOIs, chiefly:

- Saraf, Moore, Goodey, Cao, Benkovic, Maranas. *Biophys J* **90**:4167 (2006) — IPRO.
- Pantazes, Grisewood, Li, Gifford, Maranas. *J Comput Chem* **36**:251 (2015) — the IPRO suite.
- Chowdhury et al. *Nat Commun* **9**:3661 (2018) — PoreDesigner.
- Chowdhury et al. *Structure* **28**:1344 (2020) — IPRO+/−, insertions and deletions.
- Grisewood et al. *ACS Catal* **7**:3837 (2017) — specificity redesign.
- Grisewood et al. *PLoS ONE* **8**:e75358 (2013) — OptZyme.

## Licence

Not yet chosen — add one before sharing this repo. The upstream IPRO releases
carry no `LICENSE` file; the Maranas group's pages describe the suite as freely
available to academic users. No upstream code is reproduced here.
