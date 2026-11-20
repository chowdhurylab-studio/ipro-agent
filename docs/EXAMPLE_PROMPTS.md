# Example prompts

> These five are also a page you can send someone: [`docs/index.html`](index.html),
> with each prompt on a copy button and the caveats beside it. This file is the
> same content in the form the repository reads.

Five prompts you can paste at an agent that has this tool, from "I only know
the protein's name" to "explain the run I just did". Each one was executed
against this repository, and what it produced is written down, including where
it stops being a result and starts being a caveat.

Setup is one command (`bash scripts/bootstrap.sh`) or the MCP block in
[HANDOFF.md](HANDOFF.md). To follow these you need PyMOL, which the environment
installs; `ipro_fetch_structure` additionally needs the network, and nothing
else here does.

> Read this before quoting any number from these runs. Energies from the
> built-in backend are a screening scale, not kcal/mol comparable to the IPRO
> papers. Every run reports a `mode` and a list of `notes`, and both belong
> with any number you repeat. A run that reports `accepted: 0` designed nothing.
> Published hit rates across this family of methods are roughly 24–43%: treat
> every design as a hypothesis for the bench rather than a result.

---

## 1. Narrow a pore, starting from nothing but the protein's name

Uses `ipro_fetch_structure`, `ipro_inspect_structure`, `ipro_validate` and
`ipro_run`: the full path from a name to a design, including getting the
structure.

> Narrow the OmpF porin pore below 4 Å. Fetch the structure yourself — it's PDB
> 2OMF, use chain A. You may change residues 18, 20, 38 and 40. Tell me what's
> at those positions before you design anything.

What this shows: the agent downloads 2OMF from the RCSB (274 KB), writes a
chain-A copy, and *inspects it before writing a spec*, which is what makes
"residues 18, 20, 38 and 40" mean anything. Those four resolve to V18, L20,
M38 and Y40 of chain A, and the agent should tell you so and hand back spec
lines you can read. A bare residue number does not say which chain it belongs
to, and a number resolved against the wrong chain silently designs a different
residue, a defect that shipped here once already (DESIGN.md §7b).

It is also the published case: PoreDesigner's tested designs mutate exactly
these positions to tryptophan (V18W L20W M38W Y40W), so "did Trp come out at
any of the four" is a real score against the paper; see
`examples/ompf_pore/EXPECTED.md` for which comparisons are meaningful and which
are not available at all.

The caveat here is not a small one. Four positions along one wall do not
*ring* the lumen, and the channel axis is
fitted through their centroid, so it runs inside the protein and the reported
"bottleneck" is a gap beside a wall. Measured on the input structure: 0.30 Å of
clearance where a water needs 1.4 Å, giving a 1.53 Å reading that *passes* a
sub-4 Å filter. The run now says this outright, "THE PORE RADIUS HERE IS NOT A
PORE RADIUS", and names the fix. Do not read a pore radius from this prompt.
For a real measurement, give `goal_options.pore_axis` as two atoms spanning the
channel, or use all 25 positions (`examples/ompf_pore/spec_full.yaml`, whose
axis clears 2.80 Å and whose wild-type eyelet measures 8.07 Å). DESIGN.md §7h
has the full account, including why the clearance has to be measured on the
input structure and not on a candidate.

---

## 2. Switch substrate specificity, with the catalysis kept intact

Uses `ipro_goals`, `ipro_inspect_structure`, `ipro_score` and `ipro_run`: the
objective that needs two molecules and a restraint.

> I want this thioesterase to prefer the C12 substrate over C14. Keep the
> catalytic geometry — don't let the design break it. I can give you both
> substrates as separate PDBs. Which objective should this be, and what do you
> need from me?

What this shows: the agent has to *ask* here before it can write a spec.
`switch_specificity` minimises IE(target) − w·IE(anti-target), so it needs
the wanted substrate with role `target` and the rejected one with role
`anti_target`. With only one molecule it refuses rather than optimising plain
affinity and calling it specificity. "Keep the catalysis" has to become an
explicit `restraints:` entry (a `distance` between the catalytic atoms, the
mechanism the ACS Catal 2017 work used, holding catalytic distances to ±0.2 Å);
the optimizer preserves nothing on its own, and the agent should say so instead
of hoping.

Signs are the trap here. The two thioesterase papers report interaction-energy
signs in opposite directions; this implementation normalises so that every goal
is minimised, which is why the number you see will not match a figure in either
paper even when the ranking agrees. DESIGN.md §5.

---

## 3. Find the positions yourself

Uses `ipro_positions` and `ipro_inspect_structure`: for when the user knows
the interface matters but not which residues it is.

> Here's a protein and its binding partner. I don't know which residues to
> design — find the interface and propose positions, then show me the list
> before you write a spec.

What this shows: `ipro_positions` returns the residues of the design molecule
within a cutoff (default 5 Å) of a named partner, which is the operational
definition of an interface used throughout the IPRO case studies. It needs a
partner molecule in the spec or `--near`, and it says so plainly rather than
guessing: on a spec with no target it answers *"spec has no target molecule;
pass --near or add one"*.

Read the list before the spec is written. The proposed set is geometry rather
than biology: it will happily include a catalytic residue, a disulfide partner
or a metal ligand, and the optimizer will mutate any of them unless you say
otherwise. Anything that must survive the redesign belongs in `restraints:`.

---

## 4. Check what you're about to get before spending the CPU

Uses `ipro_doctor`, `ipro_validate`, `ipro_milp` and `ipro_score`: the four
things that cost nothing and change what a result means.

> Before we run anything: what's installed, what mode will this run in, is my
> spec valid, and are my constraints actually in the optimisation? Also give me
> the wild-type baseline.

What this shows: four cheap answers that prevent an expensive wrong one.
`ipro_doctor` reports `can_run`, `can_design` and which backends are degraded.
Without PyRosetta the loop is `fixed-backbone` and energies come from the
screening function; without a MILP solver the packing is a heuristic that
proves nothing about optimality. `ipro_validate` resolves every position
against the structure and reports the design-space size, so you learn that a
typo widened your alphabet from 1 residue to 18 *now* rather than from the
results. `ipro_milp` writes the integer program as an LP file, which is how you
check that "at least half the positions must be tryptophan" is a constraint the
solver honours rather than a hope. `ipro_score` measures the input as supplied,
which is also how you get `goal_options.reference_interaction` for a
constrained run.

Ask about the notes as well: a validation pass that reports `spec is usable`
can still hide a scientific problem, such as the native residue not being an
allowed choice at a position, which the run reports as a note rather than an
error.

---

## 5. Explain the run honestly

Uses `ipro_status` and the `ipro://house-rules` resource: turning numbers into
something you can put in a slide.

> The run in runs/case1 finished. Give me the honest version: what changed,
> what the score means, what you can't conclude from it, and what I'd need to
> do to trust it.

What this shows: `ipro_status` carries the mutation string, the score change,
the `mode`, the `notes` and a `designed_nothing` flag, and the
`ipro://house-rules` resource says what must travel with any design. This
prompt exists because the failure mode is social rather than computational: the
numbers look like results, and the caveats are the result.

A good answer states the mode, every note, that screening energies are not
publication kcal/mol, and, when `accepted` is 0, that nothing was designed and
the reported "best design" is the starting point. It should also say what would
raise confidence: reference energies set, a real MILP solver so optimality is
proven, PyRosetta so the backbone actually relaxes, a pore axis given
explicitly, and ultimately an experiment.

<details>
<summary><b>Walk through prompt 1 in detail</b></summary>

**What the agent does**

1. `ipro_doctor` confirms `can_design: true`. If PyMOL has not resolved it
   stops here and calls `ipro_setup`, rather than starting a run that dies at
   the first rotamer harvest.
2. `ipro_fetch_structure(pdb_id="2OMF", chains=["A"])`: 274 KB from
   `files.rcsb.org`, plus a chain-A copy.
3. `ipro_inspect_structure(structure=..., positions=["18","20","38","40"])`
   reports chain A, 340 residues spanning A1..A340, the sequence, and
   `18 → A18 is VAL`, `20 → A20 is LEU`, `38 → A38 is MET`, `40 → A40 is TYR`.
4. Writes a spec: the design molecule, the four positions, `goal:
   reshape_channel`, `direction: narrow`, and your words in `goal_statement`.
5. `ipro_validate` resolves the positions and reports the design-space size.
6. `ipro_run(..., max_minutes=...)` → a run id, then `ipro_status` until done.
7. Reports with the mode, the notes, and the axis caveat above.

**What is happening under the hood**

Each iteration perturbs the backbone (when enabled), then solves a
mixed-integer linear program that chooses one rotamer per position. That is the
reason your constraints are honoured exactly rather than approximately: "at
least half these positions must be long residues" is a linear constraint, and a
simulated-annealing packer cannot promise it. The chosen design is then
accepted or rejected on energy by a Metropolis test at IPRO's published
temperature. `docs/MILP.md` has the formulation.

**Explore the result**

Ask: *"show me the LP file for one iteration and point out where my
constraint on positions 18, 20, 38 and 40 appears"*. That turns a claim about
the optimisation into something you can read.

</details>

---

## What not to ask it

- To fold or predict a structure. It redesigns a structure you already have. A
  model is acceptable input, but then say so in the report: every energy is
  only as good as the coordinates.
- For a ΔΔG or an affinity in nM. These objectives reproduce the rank and sign
  of a change far better than its magnitude.
- For a library of thousands of variants. It produces a small ensemble
  (10 by default), not a deep mutational scan.
- For an insertion or deletion in the middle of a chain. Only terminal indels
  are supported; an internal one needs loop closure. It refuses with that
  reason rather than producing a broken backbone.
- For expression, solubility or immunogenicity. There is no objective for a
  property with no structural handle.
- Anything that needs a human decision first. Toxins, virulence factors and
  select agents; anything aimed at host range, transmissibility, immune escape
  or drug resistance; anything heading for synthesis. See
  [Responsible use](../README.md#responsible-use).
