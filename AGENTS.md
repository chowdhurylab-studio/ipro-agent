# ipro-agent: router

An agent opening this repo normally loads the skill by itself, from its
`description` in `.claude/skills/ipro-agent/SKILL.md`. This file is the fallback
for tools that read `AGENTS.md` instead, and the place the house rules live.

This repo is one tool: a generalized, agent-drivable implementation of IPRO
(Iterative Protein Redesign and Optimization).

- **Redesigning a protein toward a stated goal**, whether that is changing
  specificity, reshaping a channel or pore, improving interface affinity,
  tuning catalytic activity, inserting or deleting residues, or producing a
  mutant sequence from a structure: read `.claude/skills/ipro-agent/SKILL.md`
  first. It covers choosing the goal, writing the spec, encoding the user's
  position/residue constraints into the MILP, running the loop, and the
  degraded modes that change how a result must be reported.

- **Before anything else**, run `python3 scripts/doctor.py --require pymol`
  (equivalently `python3 -m ipro_agent.doctor --require pymol`).
  PyMOL has no fallback, since rotamers come only from its mutagenesis wizard,
  so without it `run` cannot produce a design; `--require pymol` makes that a
  non-zero exit instead of something to discover mid-run. The protocol also
  degrades in documented ways when PyRosetta, AutoDock Vina or a MILP solver is
  absent, and the mode must be stated when reporting a design.

- **Reading the method, or checking a claim against its source**:
  `docs/DESIGN.md` maps each capability to the paper and equation it came from
  and lists every deviation; `docs/MILP.md` gives the formulation in full.

- **Installing dependencies**: `bash scripts/bootstrap.sh` does the whole
  thing in one command; `docs/INSTALL.md` explains what each one unlocks.

- **Handing this to someone else**: `docs/HANDOFF.md`.

- **Worked requests, end to end**: `docs/EXAMPLE_PROMPTS.md`, five real
  prompts with what each produces and what must not be concluded from it.

- **Driving this from another agent, without cloning it**: there is an MCP
  server (`python3 -m ipro_agent.mcp_server`, or the `ipro-agent-mcp` console
  script) exposing fifteen typed tools. See README.md, "Use it from another agent
  (MCP)". If you are reading this file you have the repo, so prefer the skill:
  it carries the degraded modes and the reporting rules that tool schemas
  cannot.

- **A worked example**: `examples/ompf_pore/`, with `EXPECTED.md` stating what
  is and is not reproducible.

## House rules for this repo

- Energies from the built-in backend are a screening scale, not kcal/mol
  comparable to the IPRO papers. Never present them as published quantities.
- Never report a design without also reporting the run's `mode` and its `notes`.
- A run with `accepted: 0` designed nothing. Its "best design" is the input, so
  report it as "the search found nothing", never as a design.
- Never quietly weaken what the user specified to make a failing run succeed.
  Loosening a restraint, widening `allowed`, raising `max_self_energy` or
  adding positions each change the scientific question: report the change, or
  ask first.
- Some targets and goals need a human decision before anything runs, not after
  the result arrives; see README.md, "Responsible use".
- `iterations.jsonl` grows without limit; read progress with
  `python3 -m ipro_agent status <run-dir>`, never by loading the file.
- This is an independent reimplementation: no upstream IPRO code is copied.
  Cite the original papers for the method, and this repo only for the
  implementation.
