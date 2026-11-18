---
name: ipro-agent
description: Redesign a protein toward a stated goal using the IPRO procedure (Iterative Protein Redesign and Optimization) — change substrate specificity, widen or narrow a channel or pore, improve the affinity of a protein–protein or protein–peptide interface, tune catalytic activity, or insert/delete residues. Writes the rotamer-selection MILP that encodes the user's permitted positions and residues, then iterates mutation, docking, relaxation and energy-based accept/reject. Use when the task mentions protein design or redesign, IPRO, OptZyme, PoreDesigner, OptCDR, mutating specific residue positions, allowed amino acids at a position, rotamer selection, design positions, interaction energy, binding specificity between two ligands, pore or channel size, enzyme activity (kcat, KM), or producing a mutant sequence from a structure.
---

# IPRO: iterative protein redesign, driven by an agent

## Mental model

IPRO answers one question: **given a structure and a list of positions the user
will let you touch, which amino acids should go there?**

It is a loop, not a one-shot prediction. Each iteration proposes a small change,
re-packs the side chains optimally around it by solving an integer program,
re-scores, and keeps or discards the change by a Metropolis test. Over many
iterations it walks through sequence space, never revisiting a sequence.

Three things are yours to get right, and the tool cannot guess any of them:

1. **The goal** — which energy actually encodes what the user wants.
2. **The molecules and their roles** — what is being redesigned, what it should
   bind, and what it should *stop* binding.
3. **The design space** — which positions may change, and to which residues.

Everything else (rotamers, the MILP, the loop, the bookkeeping) is mechanical.
Spend your effort on 1–3 and on reading the result honestly.

## When NOT to use this

- **You want a structure prediction.** This redesigns a structure you already
  have; it does not fold or predict one.
- **You have no structure.** A model is acceptable, but then say so in the
  report — every energy here is only as good as the coordinates.
- **The user wants a ranked library of thousands of variants.** IPRO produces a
  small ensemble of designs (10 by default), not a deep mutational scan.
- **The user wants predicted ΔΔG or an affinity in nM.** These objectives
  reproduce the *rank* and *sign* of a change far better than its magnitude.
  Published hit rates for this family of methods are roughly 24–43%. Treat every
  design as a hypothesis to test at the bench.
- **The goal is a property with no structural handle** (expression,
  solubility, immunogenicity). There is no objective here for those.

## Workflow

Run `python3 scripts/doctor.py` once at the start. It reports which backends
resolved and what the run will actually use. **Read its output before promising
the user anything**, because the protocol degrades in a specific, documented way
when PyRosetta or Vina is absent (see *Degraded modes*).

### 1. Turn the request into a goal

`python3 -m ipro_agent goals` lists every goal, what it optimizes, and which
paper justifies it. Pick deliberately; do not rely on keyword matching.

| The user says | Goal | What gets minimized |
|---|---|---|
| "bind X more tightly", "improve this interface" | `bind_tighter` | IE(design, X) |
| "prefer X over Y", "stop it binding Y", "change its specificity" | `switch_specificity` | IE(X) − w·IE(Y) |
| "widen the channel", "narrow the pore", "let bigger substrates through" | `reshape_channel` | ±IE(probe), plus an optional radius target |
| "make it a better catalyst", "improve kcat/KM", "lower KM" | `improve_catalysis` | IE with a transition-state analogue, or with the substrate |
| "make it more stable" | `stabilize` | energy of the design molecule alone |

Two things to get right here:

- **`switch_specificity` needs an anti-target.** The molecule to select
  *against* gets `role: anti_target`. Without one, specificity collapses into
  plain affinity and you are not answering the question asked.
- **`improve_catalysis` with `parameter: kcat_over_km` or `kcat` needs a
  transition-state analogue** (`kind: transition_state`), not the substrate.
  If the user has no TSA structure, use `parameter: km`, and say that you did.

Always copy the user's own words into `goal_statement`. It is carried into every
report, so a design can always be traced back to what was actually asked.

### 2. Write the spec

One YAML file is the whole problem statement. Minimal shape:

```yaml
name: my_design
goal: switch_specificity
goal_statement: "make it prefer the C12 substrate over C14"

molecules:
  - name: enzyme
    path: enzyme.pdb
    kind: protein
    role: design          # the only molecule that gets mutated
    chains: [A]
  - name: C12
    path: c12.pdb
    kind: ligand
    role: target          # reward binding this
  - name: C14
    path: c14.pdb
    kind: ligand
    role: anti_target     # penalize binding this

positions:
  - {position: A115, allowed: [hydrophobic]}
  - {position: A117, allowed: [W, F, Y, M]}
  - {position: A119, allowed: [ALA, SER, THR]}

loop:
  iterations: 200
  random_seed: 1
```

`allowed` takes 1-letter codes, 3-letter codes, or group names (`all`,
`standard`, `hydrophobic`, `polar`, `charged`, `aromatic`, `small`, `large`,
`nonpolar_nogly`). A misspelled key is **rejected**, not ignored — a typo must
not silently change the design problem.

**If the user has not named positions**, propose them by proximity:

```bash
python3 -m ipro_agent positions spec.yaml --near C12 --cutoff 5 --json
```

This prints the residues within the cutoff of that partner, ready to paste as
spec YAML. **It is a proposal, not an answer.** The published protocol sorts
residues by distance and then takes *the eight closest that are not excluded* —
and the exclusions are what a distance rule cannot see. Walk the user through
them before accepting a position list (Grisewood et al., ACS Catal 2017,
Table S4 excludes a residue for any of these):

- it belongs to the catalytic triad, or forms the oxyanion hole;
- mutagenesis has shown it matters for function;
- it is within 4.5 Å of the reactive atom (there, the thioester sulfur) — i.e.
  it is at the wrong end of the ligand to change specificity;
- its Cα is more than 0.75 Å *closer* than its Cβ to the discriminating atom of
  the ligand, so the side chain points away from the feature being designed;
- it has ≥40% conservation across the family;
- the family alignment has a gap there;
- it sits in a flexible loop;
- an earlier round showed mutating it abolished activity.

Distance alone is cheap to compute and is the one criterion this tool
implements. The rest need the user's knowledge of the system, a family
alignment, or a prior round's results. Ask; do not guess, and never silently
design away a catalytic residue.

### 2a. Use assemblies when one sequence must work in several complexes

By default every molecule sits in one complex. When the design has to work in
more than one binding topology — bound to C12 *and* to C14, or with and without
a cofactor — declare them, and the loop will design **one sequence** that packs
separately in each, requiring all of them to pass acceptance:

```yaml
assemblies:
  - {name: with_C12, molecules: [C12]}
  - {name: with_C14, molecules: [C14], weight: 0.5}
```

The design molecule is implicit in every assembly. `weight` scales an
assembly's contribution to the combined objective; `must_pass: false` lets an
assembly be scored without gating acceptance.

Three things to understand before reaching for this:

- **Identity is shared, packing is not.** That is the point: the same residue
  may adopt different rotamers in different complexes, which is what makes the
  sequence a genuine compromise rather than an average.
- **Assemblies do not express "bind A, not B".** Every assembly's packing is
  minimized, because that is what the protein will do when it meets that
  partner. Selecting *against* a partner is `goal: switch_specificity` with
  `role: anti_target`. Use assemblies when the sequence must work in several
  contexts; use the anti-target role when it must fail in one.
- **It may not change the answer.** On a buried site the intrinsic packing
  differences between rotamers can dwarf the partner's contribution, so every
  assembly may pick the same rotamer anyway. That is a real result, not a bug.

Every molecule must belong to an assembly — one left out would be silently
unscored, so the spec is rejected instead.

### 2b. Add restraints when geometry must be preserved

This is the capability that makes enzyme work possible: **"redesign this pocket
but keep the catalysis intact"** is a restraint, not an objective. The published
thioesterase redesign held its catalytic distances to ±0.2 Å while changing
specificity — without that, the optimizer will happily destroy the active site
to improve a binding score.

```yaml
restraints:
  # keep a catalytic contact: the published form
  - {kind: distance, atoms: ["A10:OG", "LIG1:C1"], target: 3.2, tolerance: 0.2,
     label: catalytic_Ser_to_carbonyl}
  # keep two side chains apart
  - {kind: distance, atoms: ["A115:CZ", "A119:CZ"], minimum: 4.0}
  # let a side chain move no more than half an Angstrom from the input
  - {kind: tether, atoms: ["A145:OH"], tolerance: 0.5}
  # pin an atom
  - {kind: fix, atoms: ["A145:OH"]}
  # hold a dihedral (target + tolerance only; a one-sided bound is refused)
  - {kind: dihedral, atoms: ["A1:N","A1:CA","A1:CB","A1:CG"], target: -60, tolerance: 20}
```

`validate` tells you how each one will be enforced before anything expensive
runs. Four things to get right:

- **Restrain a side-chain tip, not Cβ.** Cβ is pinned by the backbone — it moves
  ~0.01 Å across a position's rotamers, so a Cβ restraint does nothing.
- **Naming a side-chain atom restricts the residue types.** `A20:CG` cannot be
  satisfied by ALA or GLY, so those are vetoed. That may be what you want; know
  that it is happening.
- **A restraint over three or more design positions cannot be enforced.** It is
  reported as `RESTRAINT NOT ENFORCED` in the run notes — read them.
- **If a restraint forbids everything**, the error names how many rotamers
  dead-end elimination had already pruned. A restraint satisfiable only by a
  clashing rotamer is a real conflict between the restraint and the structure;
  report that to the user rather than quietly raising `max_self_energy` until it
  goes away.

A restraint the *input structure* already violates is reported as
`NATIVE VIOLATION` and the run continues — no choice of rotamer can fix it, so
it is the structure's problem, not the design's. Pass `strict: true` on the
restraint to make that a hard failure instead.

### 3. Validate before spending time

```bash
python3 -m ipro_agent validate spec.yaml --json
```

Checks that every structure loads, every design position exists in the file
(with its native identity), the goal's requirements are met, and reports the
size of the design space. **Always run this.** A position that does not exist,
or a partner whose chains are missing, is caught here in a second rather than
after an hour.

### 4. Inspect the MILP you are about to solve

```bash
python3 -m ipro_agent milp spec.yaml -o model.lp --json
```

This writes the formulation and stops. The LP file is commented with the
positions, their rotamer counts, and every constraint with the equation number
it corresponds to. **Read it back to the user when they have stated a
constraint** — it is the artifact that proves their requirement was encoded as
they meant it.

User constraints that the integer program can express (this is the reason it is
a MILP and not a packer — a simulated-annealing packer cannot honour any of
them):

| User's requirement | Constraint |
|---|---|
| "change at most 5 positions" | `max_mutations` |
| "at least half must be bulky/aromatic" | `min_class` with a fraction |
| "no more than two charged residues" | `max_class` |
| "position 117 must be a tryptophan" | `require` |
| "never put a proline there" | `forbid` |
| "do not destabilize past this energy" | `max_energy` |

`min_class` is not hypothetical: PoreDesigner required at least 50% of its
designed positions to be long residues (W/F/Y/M), and `reshape_channel` adds
that constraint automatically when narrowing.

### 5. Decide how partners get re-placed

Mark a partner `redock: true` and the loop re-places it every `redock_every`
iterations (default 3, as published). **You choose the program**, because the
two available do different jobs:

```yaml
backends:
  dock: auto     # auto | vina | builtin | none
```

| Choice | Use it when | Cost |
|---|---|---|
| `builtin` | the pose is roughly right and should be *refined* — IPRO's own local rigid-body Monte Carlo, scored with the same energy as rotamer selection | ~0.2–0.4 s per 500 moves |
| `vina` | there is no usable starting pose, or mutations have moved the pocket so much the old one is meaningless — a global search of the box | seconds to minutes per ligand, plus prep and file I/O |
| `auto` | Vina if installed, else `builtin` (it says which in the notes) | — |
| `none` | the input poses are what you want scored | free |

An explicit `dock: vina` with no Vina installed **fails** rather than quietly
using the cheaper option — they answer different questions, so a silent
substitution would change the science.

**Vina's affinity is never added to the objective.** The pose is taken; the
score stays the energy backend's. Report Vina's number separately if at all,
and say it is on its own scale.

You can also place a partner once, outside the loop:

```bash
python3 -m ipro_agent dock spec.yaml --out-dir runs/dock --seed 1 --json
```

The box is centred on the design positions by default; `--center X Y Z` and
`--size X Y Z` override it, and a box much larger than the site turns a redock
into a blind dock. `--dry-run` prints the exact Vina command without running
it, which is how to check the box before spending the time.

**Never add a Vina affinity to an objective value.** Vina reports kcal/mol on
its own scale, unrelated to this tool's score. Take the *pose*, then re-score
with `ipro-agent score`.

### 6. Run

```bash
python3 -m ipro_agent run spec.yaml --run-dir runs/case1 --json
```

For a long run, launch it detached and poll:

```bash
nohup python3 -m ipro_agent run spec.yaml --run-dir runs/case1 > runs/case1.out 2>&1 &
```

```bash
python3 -m ipro_agent status runs/case1 --json
```

`status` is cheap and its output is bounded however long the run has gone on,
so it is safe to poll. **Never** read `iterations.jsonl` in full into your
context — it grows without limit. Use `status`.

`run` accepts `--max-minutes`, which stops cleanly at a checkpoint. Use it when
you do not know how long a case will take.

### 7. Report honestly

The run directory has a fixed layout:

| Path | What it holds |
|---|---|
| `best/design.json` | the design: identities, mutations as `D115W`, full sequence, score breakdown |
| `ensemble/ensemble.json` | the best N designs, not just the single best |
| `state.json` | status, best score, mode |
| `iterations.jsonl` | one record per iteration (append-only; do not bulk-read) |
| `logs/iter*.lp` | the MILP as actually solved, for the first few iterations |
| `spec.json` | the resolved spec, as actually run |

In your report, give the user: the **mutation string**, the **score change**,
the **mode** the loop ran in, and **every note** from `notes`. The notes are
where the tool records what it could not do — a skipped relaxation step, a
native residue that was not an allowed choice, a missing reference-energy
table. Suppressing them turns a qualified result into an overclaim.

## Degraded modes — check these before interpreting any number

The loop reports `mode` in `state.json` and in the run summary.

- **`full`** — all steps, including backbone perturbation and relaxation. Needs
  PyRosetta.
- **`fixed-backbone`** — the backbone never moves. The search is over sequence
  and side-chain packing only, driven by the MILP and the integer cuts. This is
  a real and widely used protocol but it is **not** the published IPRO
  protocol. Say so when reporting.

Two traps specific to the fallback stack:

1. **Without reference energies, nothing corrects for unfolded-state
   propensity**, so identities are decided by packing alone and large side
   chains are disfavoured at buried positions. The loop emits a note when this
   applies. Fix it by setting `goal_options.reference_energies`, or by using the
   PyRosetta backend (whose ref2015 score function includes the term). If a run
   returns only small residues at buried positions, treat that as the missing
   term rather than as a result.
2. **Without a MILP solver, the fallback is a heuristic that does not prove
   optimality.** Check `proven_optimal` in the ensemble entries. If it is false,
   say the packing is heuristic.

## Validation checklist

Before reporting a design, confirm:

- [ ] `doctor.py` output was read, and the mode is stated in the report
- [ ] `validate` passed with no problems
- [ ] the goal matches what the user asked for, and `goal_statement` is their wording
- [ ] every user-stated constraint appears in the LP file
- [ ] the native residue was an allowed choice at every position (or the note is reported)
- [ ] `proven_optimal` is true, or the heuristic packing is disclosed
- [ ] the score *improved* (`best_score` < `start_score`); if not, say the search found nothing
- [ ] reference energies were set, or the all-small-residue caveat is reported
- [ ] if the design must work in more than one complex, assemblies are declared (otherwise every partner is scored in one combined complex, which is a different problem)
- [ ] any geometry that must survive the redesign is stated as a restraint (a catalytic contact, a metal site, a disulfide) — the optimizer will not preserve it on its own
- [ ] no restraint came back `RESTRAINT NOT ENFORCED` or `NATIVE VIOLATION` unreported
- [ ] every entry in `notes` is passed on to the user

## Common problems

| Symptom | Cause | Fix |
|---|---|---|
| `no PyMOL available` | PyMOL not importable and not found in a conda env | `conda install -c conda-forge pymol-open-source`, or set `IPRO_PYMOL_PYTHON` to an interpreter that can import it |
| `design position(s) not found` | wrong chain, or author vs. sequential numbering | check `validate` output for the chain's residue range; PDB author numbering is what the spec uses |
| `no rotamers could be built at [...]` | PyMOL's wizard refused that residue type | check `errors` in the harvest; PRO and GLY at constrained backbones are the usual cases |
| `goal 'switch_specificity' needs ... 'anti_target'` | only a target was supplied | add the molecule to select against, or switch to `bind_tighter` |
| run stops early, note says "no unvisited assignment remained" | the reachable design space was exhausted by the integer cuts | expected for a small design space; widen `allowed`, add positions, or accept the enumeration is complete |
| every iteration rejected | the acceptance temperature is low relative to the score scale, or the energy units differ from kcal/mol | check `loop.temperature_k` (IPRO's default 3640 K ⇒ kT ≈ 7.2 kcal/mol) against the score magnitudes in `iterations.jsonl` |
| interaction energy ≈ 0 | the partner is not in contact with the design molecule | the structures are not in a common frame; superpose them first, or dock |
| design drifts to small residues | no reference energies, so nothing corrects for unfolded-state propensity (see *Degraded modes*) | set `goal_options.reference_energies` or use PyRosetta |
| `the restraints forbid every rotamer at X` | the restraint conflicts with the structure, often after clash pruning removed the rotamers that would satisfy it | read the rest of the message: it says how many were pruned. Loosen the tolerance, widen `allowed`, or raise `goal_options.max_self_energy` — but report the conflict, it is a finding |
| a restraint seems to do nothing | it names Cβ (pinned by the backbone), or its atoms are all fixed | restrain a side-chain tip atom; check `validate`'s `restraint_notes` |
| MILP huge / slow | design space too large | fewer positions, smaller `allowed` sets, or raise `pair_eps` to prune weak pair terms |

## Reference index

| File | Read it for |
|---|---|
| `docs/DESIGN.md` | every capability mapped to the paper and equation it came from, and where this implementation deviates |
| `docs/MILP.md` | the formulation in full, with the published equation numbers |
| `docs/INSTALL.md` | dependencies, what each unlocks, and what its absence degrades to |
| `examples/ompf_pore/` | a worked case based on PoreDesigner, with `EXPECTED.md` stating what is and is not reproducible |
| `ipro_agent/goals.py` | the objective for each goal, with its citation in the docstring |
| `ipro_agent/milp.py` | the formulation, the linearization, and the pruning rules |

## What this is not

This is a clean-room reimplementation of the published IPRO procedure, not the
Maranas group's code. It substitutes an open stack (PyMOL, AutoDock Vina,
PyRosetta, an open MILP solver) for the original's CHARMM, GAMS/CPLEX and
in-house Monte-Carlo docking. Energies are therefore **not** comparable to
numbers in the IPRO papers, and identical designs should not be expected.
`docs/DESIGN.md` lists every substitution. Cite the original papers for the
method; cite this repo only for the implementation.
