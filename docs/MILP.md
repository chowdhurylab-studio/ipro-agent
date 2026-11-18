# The rotamer-selection MILP

The integer program at the centre of IPRO, as implemented in
[`ipro_agent/milp.py`](../ipro_agent/milp.py), with the correspondence to the
published equations and to the behaviour of the original source.

## Sets and data

| Symbol | Meaning | In the code |
|---|---|---|
| `i, j ∈ P` | design positions | `EnergyTables.positions` (labels like `A115`) |
| `r ∈ R_i` | a rotamer at position `i` — fixes both the **identity** and the **conformation** | `EnergyTables.rotamers[i]`, a flat list over allowed residues × PyMOL states |
| `aa(i,r)` | the amino acid of rotamer `r` | `Rotamer.aa` |
| `E_self(i,r)` | rotamer `r`'s interaction with everything held fixed — backbone, non-design side chains, target and anti-target molecules — plus optional internal strain and a reference energy | `EnergyTables.self_energy` |
| `E_pair(i,r,j,s)` | interaction between two chosen rotamers | `EnergyTables.pair_energy` |

The template's own internal energy is **deliberately excluded**: it is identical
for every assignment, so it is a constant that cannot affect the argmin.
Absolute energies come from the loop's whole-structure scoring instead.

## Variables

```
y[i,r] ∈ {0,1}        rotamer r is chosen at position i
w[i,r,j,s] ∈ [0,1]    rotamers r at i and s at j are both chosen   (i < j)
```

`w` is **continuous**, not binary. This is not an approximation — see
*Why the linearization is exact* below. It is also what the original does: the
GAMS path declares the pair variable continuous on `[0,1]`.

## The program

**Objective (1)**

```
min  Σ_{i∈P} Σ_{r∈R_i} E_self(i,r) · y[i,r]
   + Σ_{i<j}  Σ_{r,s}   E_pair(i,r,j,s) · w[i,r,j,s]
```

**Assignment (2)** — exactly one rotamer per design position

```
Σ_{r∈R_i} y[i,r] = 1                      ∀ i ∈ P
```

**Linearization (3)** — written for both orientations of each surviving pair

```
Σ_{s∈R_j} w[i,r,j,s] = y[i,r]             ∀ i, r, and every j ≠ i in a kept pair
```

**Bounds (4)**

```
0 ≤ w[i,r,j,s] ≤ 1
```

### Why the linearization is exact

The product `y[i,r]·y[j,s]` is replaced by `w[i,r,j,s]` and tied down by the two
marginalization equalities (3) — one summing over `R_j`, one over `R_i` — rather
than by big-M or McCormick inequalities. Given (2) and binary `y`, the feasible
set for `w` at fixed `y` is the transportation polytope between two unit
marginals, one of which is a vertex of the simplex. Its only feasible point is
the product, so `w[i,r,j,s] = y[i,r]·y[j,s]` at every optimum, and `w` needs no
integrality constraint of its own.

That is what keeps the problem tractable: the integer variables number
`Σ_i |R_i|` (hundreds), not the square of that. The original solved instances of
roughly 1300 rotamers this way.

## Constraints this implementation adds

These are the reason the inner problem is an integer program rather than a
simulated-annealing packer: a packer solves (1)–(2) well but cannot express any
of them. Each is linear in `y` (or, for the energy cap, in `y` and `w`).

| Code | Form | Source |
|---|---|---|
| `max_mutations` | `Σ_{i} Σ_{r: aa(i,r) ≠ native(i)} y[i,r] ≤ M` | IPRO's mutation-count constraint; the suite draws `M` per iteration |
| `min_class` / `max_class` | `Σ_{i} Σ_{r: aa(i,r) ∈ S} y[i,r] ≥ K` (or `≤ K`) | PoreDesigner's requirement that ≥ 50% of designed positions take a long residue (W/F/Y/M) |
| `require` / `forbid` | `Σ_{r: aa(i,r) ∈ S} y[i,r] = 1` (or `= 0`) at position `i` | IPRO's permitted-amino-acid enforcement |
| `max_energy` | `objective(1) ≤ E_cutoff` | IPRO's stability constraint, `Σ X(E_rb+E_rs) + Σ Z·E_rr + E_sb ≤ E_cutoff` in the 2006 formulation |
| integer cuts | `Σ_{i} Σ_{r: aa(i,r) = A(i)} y[i,r] ≤ |P| − 1` for each previously evaluated sequence `A` | IPRO's "no design is ever revisited" cut (`≤ len−1` in the source) |

The integer cut forbids a **sequence**, not a rotamer assignment: every rotamer
of the named identity contributes to the sum. So a sequence cannot be
re-evaluated, while the optimizer stays free to repack any other sequence
differently.

`min_class` accepts a `fraction` instead of a `count`, rounded up — with 25
positions, `fraction: 0.5` becomes `≥ 13`.

## Size control

Pair variables grow as `(Σ_i |R_i|)²`, so three prunings run before the program
is written. All three counts appear in `EnergyTables.stats` and in
`ipro-agent milp --json`.

1. **Generalized dead-end elimination.** A rotamer whose self-energy exceeds
   `max_self_energy` clashes irrecoverably with fixed atoms and cannot appear in
   an optimal assignment, so it is dropped. This is IPRO's
   `X_ir = 0 if E_rs(i,r) > δ_i`. Two safeguards the bare rule lacks: a position
   never loses its **last** rotamer (which would make the program infeasible),
   and the **native** identity is never pruned, so "do not mutate" stays
   reachable and a mutation can be rejected.
2. **Position-pair pruning.** If the rotamer clouds at `i` and `j` cannot reach
   each other — centroid separation beyond the sum of their bounding radii plus
   the energy cutoff — the pair contributes no variables at all.
3. **Pair-term pruning.** Individual `|E_pair| < pair_eps` are dropped.

Repulsion is capped per atom pair by the energy backend (default 10 kcal/mol).
Rotamer libraries routinely produce hard clashes whose raw `1/r¹²` values would
dominate the objective and leave the program badly conditioned; the original
clamps objective coefficients at 10000 kcal/mol for the same reason.

## Solving

`RotamerMILP.to_lp()` emits CPLEX LP format with the positions, their rotamer
counts and every constraint commented. **This file is the artifact that records
what was actually solved** — the deliverable when a user has stated a
constraint and wants to see it encoded. `ipro-agent milp spec.yaml -o model.lp`
writes it without running anything.

`solve()` tries, in order:

1. **PuLP**, with whichever solver it has (CPLEX, Gurobi, CBC, HiGHS, SCIP, GLPK);
2. **a solver binary via the LP file** — `glpsol` or `cbc`, whose solution
   formats are parsed here;
3. **a built-in greedy heuristic** — iterated best-improvement over
   single-position reassignments, which respects (2) by construction and checks
   every user constraint, but **does not prove optimality**. `proven_optimal`
   stays `False`, and the solver string says so.

Two failure modes are kept distinct, because conflating them corrupts results:

* a solver that is **missing or broken** → try the next one;
* a solver that **proves infeasibility** → raise `Infeasible` immediately and
  never retry. Substituting another solver, or the heuristic, would return an
  assignment that violates the user's constraints or re-visits a sequence an
  integer cut forbade.

A returned solution is also checked structurally: the selected weight at each
position must be 1. `glpsol` writes a solution file of zeros even when it proves
infeasibility, and without that check the code would hand back the first rotamer
at every position as though the solver had chosen it.

## Verification

`tests/test_ipro.py` checks the formulation against exhaustive enumeration —
on randomized toy tables and on real PyMOL rotamers with the real energy
function — unconstrained and with a class-cardinality constraint. Both match to
`1e-6`. It also checks that an integer cut excludes the sequence just found,
that an infeasible program raises rather than returning something, and that the
LP file's objective agrees with the objective used by the accept/reject test.

## Known deviations from the original

| Here | Original | Consequence |
|---|---|---|
| Rotamers from PyMOL's mutagenesis library | backbone-dependent Dunbrack library in an external directory, not shipped with either release | different rotamer sets, so different optima; PyMOL's is not backbone-dependent in the same sense |
| `E_self`/`E_pair` from PyRosetta or a built-in screening function | CHARMM vdW + electrostatics + Lazaridis–Karplus solvation, with a softened LJ and `−ln(p)` rotamer-probability terms | energies are **not** comparable to the papers' |
| Reference energies optional and user-supplied | a hardcoded 20-entry amino-acid reference-energy table | without them a rigid-backbone search drifts to the smallest allowed residue; the loop warns |
| Open solver via an LP file, or PuLP | CPLEX through GAMS (CPLEX directly for OptMAVEn) | same formulation, same optimum |
| One MILP over all design positions, with non-window positions pinned | MILP over a 10–15 residue window around the perturbed segment | comparable locality, different exact scope |

## Sources

- Saraf, Moore, Goodey, Cao, Benkovic, Maranas. *Biophys J* **90**:4167–4180 (2006). The original formulation; equations 3–10 (single-structure) and 11–21 (library).
- Pantazes, Grisewood, Li, Gifford, Maranas. *J Comput Chem* **36**:251–263 (2015). The IPRO suite.
- Chowdhury et al. *Nat Commun* **9**:3661 (2018), supplementary information. The formulation reproduced in full, CC-BY, including the ≥50%-long constraint and the no-good cut.
- The IPRO Suite source (`ROTAMERS.py`, `GAMS.py`, `CPLEX.py`), for the variable declarations, the coefficient clamp and the cut forms.
