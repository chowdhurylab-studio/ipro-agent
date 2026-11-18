# ipro-agent — router

This repo is one tool: a generalized, agent-drivable implementation of IPRO
(Iterative Protein Redesign and Optimization).

- **Redesigning a protein toward a stated goal** — changing specificity,
  reshaping a channel or pore, improving interface affinity, tuning catalytic
  activity, inserting or deleting residues, or producing a mutant sequence from
  a structure: **read `SKILL.md` first.** It covers choosing the goal, writing
  the spec, encoding the user's position/residue constraints into the MILP,
  running the loop, and the degraded modes that change how a result must be
  reported.

- **Before anything else**, run `python3 scripts/doctor.py`. The protocol
  degrades in documented ways when PyRosetta, AutoDock Vina or a MILP solver is
  absent, and the mode must be stated when reporting a design.

- **Reading the method, or checking a claim against its source**:
  `docs/DESIGN.md` maps each capability to the paper and equation it came from
  and lists every deviation; `docs/MILP.md` gives the formulation in full.

- **Installing dependencies**: `docs/INSTALL.md`.

- **A worked example**: `examples/ompf_pore/`, with `EXPECTED.md` stating what
  is and is not reproducible.

## House rules for this repo

- Energies from the built-in backend are a screening scale, not kcal/mol
  comparable to the IPRO papers. Never present them as published quantities.
- Never report a design without also reporting the run's `mode` and its `notes`.
- `iterations.jsonl` grows without limit; read progress with
  `python3 -m ipro_agent status <run-dir>`, never by loading the file.
- This is a clean-room reimplementation. Cite the original papers for the
  method, this repo only for the implementation.
