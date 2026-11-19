## [1] BLOCKER The backbone perturbation is not anchored on a design position, so perturbation and redesign are statistically independent

**file:** /Users/supantha/Documents/test123/ipro-agent/ipro_agent/perturb.py:112-153 (choose_segment), called from /Users/supantha/Documents/test123/ipro-agent/ipro_agent/loop.py:628-648

**source:** Biophys J 2006, Fig. 5 caption and step iii ("Side-chain/sequence optimization"); IPRO Suite JCC 2015, step 1; released source IPRO_FUNCTIONS.py:102-110 + 151-176

**detail:** In IPRO the perturbed segment is *built around a design position*: the Suite paper's step 1 "begins by randomly selecting a DP", then draws the region size (1-5) and the DP's relative position inside it; the released source does exactly this (standard_position_selection picks a DP uniformly, standard_perturbation_selection splits randint(0,4) into before/after around it), and only DPs inside that region get permission to mutate. The 2006 paper states the same coupling: design positions *within the perturbation region* may change amino acid type (Fig. 5). Here choose_segment picks a uniformly random proline-free run anywhere in the chain, and _choose_window (loop.py:1858) independently picks 1-3 design positions from the sorted design-position list. The two never meet. Because there is also no neighbour repacking (see the perturbation_flank/neighbour_cutoff finding), a perturbation that misses every design position changes energy in a region the MILP has no variables for, and the Metropolis test then accepts or rejects on that change alone. Secondary: _perturb_once only ever perturbs (dm.chains or chain_ids())[0], so in a multi-chain design molecule later chains are never perturbed.

**evidence:** Paper (Fig. 5): design positions "within the perturbation region" may change amino acid type; Suite step 1: "begins by randomly selecting a DP" and "Only DPs within the perturbed region are permitted to mutate". Measured with the repo's own full example (examples/ompf_pore/spec_full.yaml: 25 design positions, chain A of 340 residues), 20,000 draws from choose_segment(st, 'A', (1,5)): P(segment touches any design position) = 0.1244. In IPRO that probability is 1.0 by construction. So ~88% of perturbation moves are decoupled from the design. docs/MILP.md:189 claims "comparable locality" for this deviation; 12.4% overlap is not comparable locality.

**fix:** Pick the perturbation spot from spec.positions (as the source does), then grow the 1-5 residue segment around it with the before/after split, and derive the mutable window from the design positions that fall inside the perturbed region instead of drawing it independently. If the current behaviour is kept deliberately, say so in docs/DESIGN.md s4 and correct "comparable locality" in docs/MILP.md:189.

---

## [2] BLOCKER Backbone relaxation runs before the MILP, not after it, contradicting both papers and DESIGN.md's own stated order

**file:** /Users/supantha/Documents/test123/ipro-agent/ipro_agent/loop.py:648 (and _relax_once at 650-692)

**source:** Biophys J 2006, Fig. 4 step iv and section "iv. Backbone relaxation"; IPRO Suite JCC 2015 ("The fourth and fifth steps ... energy minimization and interaction energy calculation")

**detail:** The 2006 paper puts relaxation at step iv, after the side-chain/sequence optimization, and says why: the MILP's new residues/rotamers "may no longer be optimally interacting with the previous backbone", so the backbone torsions are minimized over the same 10-15 residue window with the side chains held fixed. The Suite makes minimization step 4, after rotamer selection and docking. Here _perturb_once returns self._relax_once(candidate), so relaxation happens on the perturbed backbone carrying the OLD side chains, and the rotamer tables are then rebuilt from the relaxed file (loop.py:1427-1430). backend.relax has exactly one call site (loop.py:672), so nothing relaxes after the MILP. The backbone therefore never adapts to the chosen mutation, which is the entire purpose of the published step, and the structure that is scored and written out is an unminimized packing of newly placed side chains. Related: the relaxation is a whole-pose FastRelax rather than the paper's 10-15 residue window with side chains fixed.

**evidence:** docs/DESIGN.md:18 states the implemented order as "perturb -> optimal rotamers -> redock -> relax -> score". The code's order is perturb -> relax -> rotamers/MILP -> redock -> score. loop.py's own module docstring (lines 11-12) also lists "5. relax the whole complex and compute its energy" as a separate step that does not exist. _relax_once's docstring says "The published protocol minimizes the structure after perturbing the backbone", which is only half of step iv.

**fix:** Move the relax call after the MILP solve (and after the redock, to match the Suite's step 4), re-score from the relaxed coordinates, and keep the perturbation's own strain relief as a separate restrained minimization if that is wanted. Otherwise correct docs/DESIGN.md:18 and the loop.py docstring to state the order actually run.

---

## [3] NICE-TO-HAVE perturbation_flank and neighbour_cutoff are dead settings: no flanking or neighbour repacking exists anywhere

**file:** /Users/supantha/Documents/test123/ipro-agent/ipro_agent/config.py:636 and :662 (declared, never read); /Users/supantha/Documents/test123/ipro-agent/ipro_agent/structure.py:263 (neighbours(), no callers)

**source:** Biophys J 2006, step iii and Fig. 5 (flanking +/-5 positions may change rotamer); IPRO Suite JCC 2015, steps 1-2 (4.5 A repack shell)

**detail:** The 2006 paper's step iii optimizes a 10-15 residue window: design positions inside the perturbed region change identity, and the five flanking residues on either side "can only change rotamers but not the residue type". The Suite adds a 4.5 A repack shell around the perturbed region. Neither is implemented. grep over ipro_agent/, tests/ and scripts/ finds perturbation_flank only at config.py:662 and neighbour_cutoff only at config.py:636; Structure.neighbours() (whose docstring says it decides "which non-design residues must be repacked alongside a mutation") has no caller. The MILP's variables are exactly the user's design positions, so after a perturbation the surrounding native side chains keep their input rotamers and cannot relieve the strain the perturbation introduces. docs/DESIGN.md s4 ("What is not implemented") does not list this.

**evidence:** docs/DESIGN.md:473-474 lists both parameters as live with provenance ("IPRO Suite | +/-5 residues around the segment -> an 11-residue window", "source | Packing/repacking shell"). examples/ompf_pore/spec_full.yaml:188,192 annotate them "[paper] 11-residue window" and "[paper] repack within 4.5 A of the perturbation", so a user setting them believes they do something. loop.py's module docstring (line 9) also promises the MILP covers "the chosen positions and their neighbourhood".

**fix:** Either add the flanking/neighbour positions to the MILP as rotamer-only variables (identity pinned to native, which the existing require-constraint already expresses), or remove the two settings from LoopSettings and the example specs and record the omission in docs/DESIGN.md s4.

---

## [4] SHOULD-FIX The 40 kcal/mol / 5% hard-reject guard is applied to the goal objective instead of the complex energy, so there is no default stability safeguard

**file:** /Users/supantha/Documents/test123/ipro-agent/ipro_agent/loop.py:1550-1559

**source:** Biophys J 2006, Eq. 4 and the paragraph introducing it; released source IPRO_FUNCTIONS.py:742-751 (standard_decision)

**detail:** In the released source the two tests are deliberately on different quantities: the Metropolis window is compared against the *interaction* energy, while the 40 kcal/mol-or-5% cap is compared against `Energies[group]['Complex']` -- its comment says the goal of better binding must not come at the expense of the complex energy. That is the runtime form of the 2006 paper's Eq. 4, which the paper introduces because the binding objective "does not encode information about the interactions in the entire protein". Here `delta = score - current_score` is the goal objective's change, and hard_cap = max(40, 0.05*|current_score|) is computed from the same objective, so both the Metropolis test and the hard reject police the same number and nothing watches the complex energy. ScoreContext.total already holds the complex energy (loop.py:412-416) and is tracked as current_asm_energy, but only the multi-assembly branch uses it. The only stability protection left is goal_options.max_total_energy, which is off unless the user sets an absolute number, and the optional MILP max_energy constraint.

**evidence:** Source: `if energies[...]['Complex'] > complex + value` with `value = max(40.0, 0.05*|complex|)` -> keep = False. Code: `hard_cap = max(spec.loop.hard_reject_kcal, spec.loop.hard_reject_fraction * abs(current_score))`; `if delta > hard_cap`. docs/DESIGN.md:29 and :479 record the 40/5% numbers as coming from the source but do not say which energy they apply to, so the row reads as agreement.

**fix:** Compare ctx.total (per assembly) against the previous iteration's total with the max(40, 5%) rule, keeping the Metropolis test on the objective, as the source does; or state in docs/DESIGN.md that the guard was re-pointed at the objective and that Eq. 4's safeguard has no default counterpart.

---

## [5] SHOULD-FIX phi is perturbed at the N-terminal residue, where it is undefined; the result is a rigid-body rotation of the whole molecule

**file:** /Users/supantha/Documents/test123/ipro-agent/ipro_agent/perturb.py:193-204

**source:** Biophys J 2006, step i (phi/psi of the perturbed positions); released source IPRO_FUNCTIONS.py:206-212 (generate_angles)

**detail:** phi(i) is the dihedral C(i-1)-N(i)-CA(i)-C(i) and does not exist for the first residue. The released source guards exactly this (`if residue.name != molecule[0].name` before generating phi, and the mirror guard for psi at the last residue). perturb_backbone applies phi whenever the residue has N and CA, so at residue 1 it rotates every atom except N about the N->CA axis -- which changes no internal coordinate at all and only re-orients the entire design molecule relative to the fixed partners. The iteration's energy change is then pure docking artefact, and if the Metropolis test accepts it the molecule stays rotated. Low frequency in a long chain, but the probability scales as (segment length)/(chain length), so for a short peptide design molecule it is a large fraction of all moves.

**evidence:** Measured on examples/ompf_pore/2omf_A.pdb with a forced phi = +5.0 deg, psi = 0: at residue 1 the maximum change in any internal interatomic distance is 2.1e-14 A (machine zero) while atoms move up to 3.058 A; at interior residue 151 the same perturbation changes internal distances by up to 3.483 A. P(segment includes residue 1) = 0.0027 over 20,000 draws on this 340-residue chain.

**fix:** Skip phi for the first residue of the chain and psi for the last, as the source does.

---

## [6] SHOULD-FIX The dead-end-elimination prune uses one global 50 kcal/mol threshold, not the paper's per-position delta_i, so it is not optimality-preserving

**file:** /Users/supantha/Documents/test123/ipro-agent/ipro_agent/milp.py:287-337 (prune) and :68-76 (docstring)

**source:** Biophys J 2006, Eq. 6 (eq6_M14.gif) with the delta_i threshold (eqinl_M16.gif) and the Looger & Hellinga background-optimization citation

**detail:** Eq. 6 is `X_ir = 0 for all i,r such that E^rs_ir > delta_i` -- a threshold *per position*, derived (the paper cites Looger & Hellinga's background optimization) so that the eliminated rotamers are "guaranteed not to be part of the optimal solution". build_tables instead drops any rotamer whose self energy exceeds a single absolute max_self_energy (default 50.0), keeping only the best rotamer and the native identity as safeguards. With an absolute threshold the guarantee does not hold: pair energies are unbounded below in aggregate (each atom pair is bounded to -10 kcal/mol but a block sums over many pairs), so a rotamer just above the threshold can still be in the optimum. The two safeguards do not restore exactness -- they only stop infeasibility.

**evidence:** Constructed the smallest counterexample and solved both tables with the repo's own RotamerMILP (glpsol): with self energies {A1 r0: 0.0, A1 r1: 51.0}, pair {(A1,r1,A2,r0): -60.0}, the full table's optimum is -9.000 choosing r1; after the >50 prune the optimum is +0.000 choosing r0. docs/DESIGN.md:20 and docs/MILP.md "Size control" item 1 both assert the pruned rotamer "cannot appear in an optimal assignment".

**fix:** Make the threshold position-relative (e.g. min_r E_self(i,r) + tolerance, or a true DEE/Goldstein criterion using the pair-energy extrema), or downgrade the claim in milp.py:68-76, docs/MILP.md and docs/DESIGN.md:20 from dead-end elimination to a heuristic clash filter and report it as such in EnergyTables.stats.

---

## [7] SHOULD-FIX The 2006 paper's hybrid-library formulation (HLDP, Eqs 11-21) is absent and is not listed in section 4

**file:** /Users/supantha/Documents/test123/ipro-agent/docs/DESIGN.md:511-523 (section 4) and :19

**source:** Biophys J 2006, title and Abstract; HLDP formulation Eqs 11-21; Methods steps a-b

**detail:** The 2006 paper's subject is redesigning "an entire combinatorial protein library in one step": it defines SSDP (Eqs 3-10) for a single sequence and HLDP (Eqs 11-21) for the library, where amino acid identity is tied across hybrids via the parental-sequence link parameters while rotamers stay free per hybrid, and the paper's worked result is the 16-member DHFR library. None of that is implemented -- no hybrid generation, no parental splicing, no per-hybrid rotamer sets with shared identities -- and docs/DESIGN.md section 4 ("What is not implemented") does not mention it, while section 1 cites the 2006 paper as the source of the MILP. The multi-assembly machinery is a different thing (one sequence in several binding topologies), not the library problem. A reader checking the tool against Saraf 2006 has nothing telling them the paper's central formulation is out of scope.

**evidence:** grep for hybrid/HLDP/SSDP/parental/combinatorial over docs/DESIGN.md and README.md returns nothing; the only mention anywhere is docs/MILP.md:193, a bibliography line reading "equations 3-10 (single-structure) and 11-21 (library)", which does not say the library half is unimplemented. The paper itself frames single-sequence design as the limiting case, not the main case.

**fix:** Add a row to docs/DESIGN.md section 4 stating that only the single-sequence formulation (Eqs 3-10) is implemented and that the hybrid-library problem (Eqs 11-21) and its structure-generation steps are out of scope, and narrow the citation at DESIGN.md:19 accordingly.

---

## [8] BLOCKER design_per_iteration = (1,3) is attributed to the IPRO Suite, which states no such rule

**file:** /Users/supantha/Documents/test123/ipro-agent/docs/DESIGN.md:472 (design_per_iteration row); /Users/supantha/Documents/test123/ipro-agent/ipro_agent/config.py:643

**source:** IPRO Suite JCC 2015, step 1-2; Biophys J 2006, step iii; released source IPRO_FUNCTIONS.py:151-176, 228-256

**detail:** DESIGN.md:472 says "IPRO Suite | 1-3 contiguous positions re-designed per iteration" and config.py:642 repeats "The IPRO Suite paper describes 1-3 contiguous residues". The Suite paper says nothing of the kind: it specifies a perturbed region of one to five residues positioned around a randomly chosen DP, and then that only the DPs inside that region may mutate -- so the count of positions redesigned per iteration is an outcome of where the perturbation landed, never a 1-3 draw. The released source likewise has no mutation-count draw (its only randint calls are the DP choice and the before/after split of randint(0,4)). The number of positions redesigned here is therefore both the wrong rule and, per the first finding, drawn independently of the perturbation.

**evidence:** No occurrence of a 1-3 / one-to-three rule anywhere in refs/papers/user-downloads/jcc_suite.txt; the relevant sentences are "the total size of the perturbed region (one to five residues)" and "Only DPs within the perturbed region are permitted to mutate". The 2006 paper's Fig. 5 states the same coupling.

**fix:** Derive the redesign window from the perturbed region (see the first finding), or mark design_per_iteration in DESIGN.md section 2 as this repo's choice rather than citing the Suite.

---

## [9] SHOULD-FIX The paper's rotamer-reduction rule is relative to the best rotamer at each position; ours is a global absolute threshold, and the difference is not listed in §4

**file:** /Users/supantha/Documents/test123/ipro-agent/ipro_agent/milp.py:333

**source:** J Comput Chem 36:251-263 (2015), p. 255 col. 2 (rotamer reduction); implemented upstream in ROTAMERS.py:998-1119 `sort_rotamers`, STANDARDS.py:107 `defaultRotamerWindow = 40.0`

**detail:** The Suite paper specifies the reduction precisely: keep the three lowest-RNR-energy rotamers of each permitted amino-acid kind at each position, plus every rotamer within 40 kcal/mol of the lowest-energy rotamer *at that position*, tightening both until the total is under the cap. `build_tables` instead drops any rotamer whose self energy exceeds a single global `max_self_energy` (default 50.0), with keep-best and keep-native patches. An absolute threshold is not scale-free: at a position whose whole rotamer set sits above the threshold, the design space collapses to one rotamer. Neither the per-kind floor, the relative window, nor the tightening is implemented, and §4's "what is not implemented" does not mention any of it. MILP.md:103-109 presents the absolute rule as "IPRO's X_ir = 0 if E_rs(i,r) > δ_i" and asserts the dropped rotamer "cannot appear in an optimal assignment" — the 2006 paper's δ_i is derived from the Looger-Hellinga background-optimization bound so elimination is provably safe; a fixed 50 kcal/mol constant carries no such guarantee, as the code's own keep-best safeguard concedes.

**evidence:** Measured (scratch script /private/tmp/claude-501/-Users-supantha-Documents-test123/a4581fba-d9ac-4b77-93bd-2ae3a83c7cd7/scratchpad/paper-jcc-milp/check_prune.py): six rotamers at one position with self energies 60.0, 62.0, 70.0, 75.0, 90.0, 99.0 (a 39 kcal/mol band, i.e. inside IPRO's 40 kcal/mol window). `build_tables(..., max_self_energy=50.0)` keeps **1 of 6** — only the native ALA, via the keep-native safeguard. IPRO's published rule (lowest + 40) keeps **6 of 6**. Upstream code confirms the rule: ROTAMERS.py:1053 `N = 3`; :1089-1095 `elif sorted[n][i][1] <= sorted[n][0][1] + window`; :1108 `N -= 1`; :1112 `window = 0.95*window`. DESIGN.md §7d defect 25 records this exact failure mode biting once already ("94 of 160" rotamers pruned, TRP/PHE/TYR/PRO deleted from A18, so the published V18W answer left the design space silently).

**fix:** Either make the cutoff relative — `min_r E_self(i,r) + window` with a per-amino-acid retention floor — or add a §4 row stating that IPRO's relative 40 kcal/mol window and 3-per-kind floor are replaced here by one absolute threshold, and soften MILP.md:104-105 from "cannot appear in an optimal assignment" to what the code actually guarantees.

---

## [10] SHOULD-FIX Exceeding `max_rotamers` is documented as triggering tighter pruning, but nothing in the code tightens anything

**file:** /Users/supantha/Documents/test123/ipro-agent/ipro_agent/loop.py:370

**source:** J Comput Chem 36:251-263 (2015), p. 255 col. 2 ("progressively become more stringent until the number of rotamers kept is below the threshold"); upstream ROTAMERS.py:1058-1112

**detail:** `config.py:629-630` states "Exceeding it triggers tighter pruning rather than an unsolvable program", and `loop.py:371-373` appends the run note "pruning will be tighter than usual". No pruning parameter is read from or adjusted by `max_rotamers` anywhere. A user who sees that note in a run's `notes` will believe an automatic size-control step ran; it did not, and the MILP is handed the full oversized rotamer set. This is the paper's progressive-tightening step, specified explicitly, implemented upstream, absent here and not listed in §4.

**evidence:** `grep -n max_rotamers ipro_agent/*.py` returns exactly two sites: config.py:631 (the dataclass default) and loop.py:370/372 (the comparison and the note). Nothing else consumes it; `_tables_for` (loop.py:1183-1185) reads `max_self_energy` from `goal_options` with a fixed 50.0 default, independent of the rotamer count. Upstream does implement the loop: ROTAMERS.py `while total > MAX:` with `window = 0.95*window` (:1112) and `N -= 1` once `window < 0.15*WINDOW` (:1107-1109), raising only after 300 s of failing to shrink.

**fix:** Either implement the shrink loop (re-run `build_tables` with a tightened threshold until the kept count is under the cap), or change both the config comment and the run note to say the cap is advisory and only reported — a note that claims an action the code never takes is worse than no note.

---

## [11] SHOULD-FIX §1b still says the tool models one binding assembly, contradicting §1d, MILP.md and assemblies.py

**file:** /Users/supantha/Documents/test123/ipro-agent/docs/DESIGN.md:127

**source:** J Comput Chem 36:251-263 (2015), p. 257 ("All BAs must pass the selection criteria"); contradicted by DESIGN.md §1d and docs/MILP.md's own constraint table

**detail:** §1b's prose reads that IPRO supports multiple binding assemblies "this tool models one assembly, so a design that must satisfy several binding topologies at once cannot be expressed". That is stale: multiple assemblies are implemented, with per-assembly packing, identity coupling and conjunctive acceptance. The sentence sits two lines below the same section's own table row "the BAs | `assemblies:` — see §1d", so §1b contradicts itself as well as §1d. A reader checking whether the tool can express the paper's multi-BA requirement gets the wrong answer from the section written to answer that question.

**evidence:** docs/DESIGN.md:190-240 (§1d) describes the virtual-position encoding and the coupling family `sum_{r: aa(r)=a} y[ref::i,r] - sum_{r: aa(r)=a} y[h::i,r] = 0`. ipro_agent/milp.py:490-493 holds `identity_links`; :605-630 emits `idlink<n>` rows; :896-903 adds the same constraints in the PuLP path; ipro_agent/assemblies.py:92-96 and :187-196 build the links and the reference sequence scope. docs/MILP.md's constraint table lists "identity coupling across assemblies" as implemented.

**fix:** Delete or rewrite that clause in §1b to point at §1d, and keep the remaining limit that §1d states accurately (every assembly shares one design backbone).

---

## [12] NICE-TO-HAVE `max_rotamers` = 1300 is attributed only to the source; the paper's threshold is 1400 and the disagreement is nowhere recorded

**file:** /Users/supantha/Documents/test123/ipro-agent/docs/DESIGN.md:480

**source:** J Comput Chem 36:251-263 (2015), p. 255 col. 2

**detail:** §2's header promises that where two sources disagree, both are recorded and the chosen value is stated — as it does for `perturbation_sigma_deg` (1.5 source vs 1.65 paper). The `max_rotamers` row says only "1300 | source" with an empty note. The Suite paper gives 1400. The chosen value is defensible (it matches the released code), but the row as written leaves a reader who checks the paper unable to reconcile it, and MILP.md repeats 1300 twice without mentioning the paper's figure.

**evidence:** Paper, p. 255 col. 2, verified on the PDF page image: the rotamer total is reduced below a threshold of 1400 to keep the MILP solvable in under 10 min and under 4 GB. Source: STANDARDS.py:103 `defaultMaxRotamers = 1300`. docs/MILP.md:71 "roughly 1300 rotamers" and :135 "the ~1300 rotamers the released source caps at" — correctly attributed to the source, but the paper's 1400 appears nowhere in the repo.

**fix:** Fill in the note column: "the Suite paper states 1400; the released source's 1300 was chosen".

---

## [13] BLOCKER Eq 18 divides both terms by ONE RT where the paper uses two RTs that differ by 25x; this is not a scale factor and it flips candidate rankings

**file:** ipro_agent/goals.py:626 (also 541, 547, 560, 618)

**source:** OptZyme, PLoS ONE 2013 (PMC3792102), Eq 18 and the paragraph immediately after it — refs/papers/jcc2014-ipro-suite/surrogate-oa-sources/OptZyme_PLoSONE2013_PMC3792102_fulltext.txt lines 156-159

**detail:** The OptZyme paper IS on disk, so this is resolvable. Its Eq-18 prose gives both constants outright: "(RT)_TSA = 15.3 kJ/mol" and "(RT)_S = 386.7 kJ/mol", self-consistent with the weights it also states (1/(RT)_TSA = 0.06 and 1/(RT)_S = 0.002 mol/kJ), and it names the correlating expression as "IE_TSA/(RT)_TSA-IE_S/(RT)_S". The two RTs differ by a factor of 25.27, so the paper weights IE_S at 3.96% of IE_TSA. Dividing both by one RT sets that relative weight to 1.0 — a 25x over-weighting of the substrate term, which changes the ARGMIN, not just the scale. Dividing by RT is itself correct (the form matches Eq 18), so the form is right and the constants are wrong. Two further points: (i) 298.15 K is the wrong KIND of value — the paper's RTs are fitted correlating pseudo-temperatures (1840 K and ~4.65e4 K) and it says plainly that "experimental and correlating temperatures do not match"; (ii) the paper's own parentheticals are transposed relative to its RT values (R*4.65e4 K = 386.6 kJ/mol, R*1840 K = 15.30 kJ/mol), and the RT values are the self-consistent pair, so do not propagate that typo. Contrary to the task framing, DESIGN.md section 7d says NOTHING about this: the only statement anywhere is the docstring at goals.py:541, which promises `IE_TSA / RT_TSA - IE_S / RT_S`, i.e. the two-RT form the code does not implement; line 547 adds "can be set per term" while OPTION_KEYS (line 560) holds a single scalar `temperature_k`. DESIGN.md:34 asserts Eq 18 with no caveat and section 4 does not list it, so this is a code/paper disagreement where the docs claim agreement.

**evidence:** Paper: (RT)_TSA = 15.3 kJ/mol = 3.6568 kcal/mol; (RT)_S = 386.7 kJ/mol = 92.4235 kcal/mol; ratio 25.27. Checks: 1/15.3 = 0.0654 and 1/386.7 = 0.0026 mol/kJ, matching the paper's stated 0.06 and 0.002. Code: `_rt` = 1.987e-3 * 298.15 = 0.5924 kcal/mol applied to BOTH terms (goals.py:626). Measured rank flip, candidates with both IEs favourable: A (IE_TSA -15.0, IE_S -2.0) vs B (IE_TSA -20.0, IE_S -12.0). Paper Eq 18: A = -4.080, B = -5.339 -> B wins. Repo: A = -21.944, B = -13.504 -> A wins. The repo therefore rejects the candidate that binds the TSA 5 kcal/mol better. Why it survived review: section 7d's clean list claims direction was checked "for all three OptZyme parameters" — direction IS preserved; only the relative weight is wrong. There is also no test for `kcat` or `kcat_over_km` at all (grep over tests/ returns two ImproveCatalysis tests, both about the `km` ambiguity, at tests/test_ipro.py:3123 and 3131). Scratch: /private/tmp/claude-501/-Users-supantha-Documents-test123/a4581fba-d9ac-4b77-93bd-2ae3a83c7cd7/scratchpad/paper-optzyme/eq18.py

**fix:** Give the `kcat` objective two divisors. Add `rt_tsa_kcal`/`rt_s_kcal` (or `temperature_k_tsa`/`temperature_k_substrate`) to OPTION_KEYS and default them to the paper's 15.3 and 386.7 kJ/mol (3.657 and 92.42 kcal/mol), recorded in DESIGN.md section 2 as GUS/pNP-GLU correlation fits that are system-specific rather than physical temperatures. Make a single scalar `temperature_k` an error for `parameter: kcat` (it is meaningful for neither term), or at minimum emit a note stating that one RT weights the two terms equally where the paper weights IE_S at 4%. Fix the docstring at 541/547 to describe what the code does, and add a direction-AND-weighting test for `kcat`.

---

## [14] SHOULD-FIX The paper's own catalytic-distance restraints fall in the "nothing to decide" branch and are then enforced nowhere, although both the perturbation and the redock move them

**file:** ipro_agent/restraints.py:413-430; ipro_agent/loop.py:1432 and 1925-1958

**source:** ACS Catal 2017 (PMC5784442), Materials and Methods — refs/papers/acscatal2017-tesa/PMC5784442_fulltext.txt line 70; SI Table S4 footnotes c and d — refs/papers/user-downloads/si.txt lines 1025 and 1027

**detail:** The paper says restraints were imposed so the "intermolecular catalytic distances (± 0.2 Å) were maintained" at S10, G44, N73, D154 and H157. Those five residues are exactly the ones that same paper EXCLUDES from design (SI footnote c, catalytic triad; footnote d, oxyanion hole), so every one of those restraints has an enzyme atom that is not at a design position and a substrate atom on a partner. `_resolve` (restraints.py:526-528) marks an atom as rotamer-dependent only when it is a side-chain atom AT a design position, so all five land in the `len(positions) == 0` branch: summary "nothing to decide", no veto, `is_empty()` True. The loop re-plans each iteration (loop.py:1432) and the re-plan does detect the violation — but writes it only to `plan.native_violations`, which is read exclusively at loop.py:1330-1337, inside run_loop's setup BEFORE `for it in range(...)`. The multi-assembly branch of `_plan_restraints_for` does not even copy `native_violations` into the merged plan. Meanwhile docking.py, dock.py and perturb.py contain no occurrence of "restraint" at all, so the two steps that actually move an intermolecular distance — backbone perturbation every iteration, and a 500-move rigid-body redock at sigma 0.2 A — run unrestrained. DESIGN.md section 1b claims this exact case is "expressible" ("redesign this site but keep the catalysis intact"), section 1c lists only two limits (>=3 design positions; unnameable molecule), and DESIGN.md quotes IPRO's own "rigorously enforced throughout all functions" twice. So the restraint is expressible as syntax but enforced in substance nowhere for the paper's own atom set.

**evidence:** Constructed the paper's case with the repo's API: a `distance` restraint between a non-design enzyme atom and a partner atom, minimum/maximum = d0 +/- 0.2. plan_restraints returns summary ['catalytic_S10: 0.000 A in the input structure (satisfied); nothing to decide'], rotamer_vetoes set(), pair_vetoes set(), native_violations [], is_empty() True. Then translating the partner by 1.0 A — 5x the paper's tolerance, well under the cumulative drift of 500 redock moves at sigma 0.2 A — the re-plan reports 'catalytic_S10: 1.000 A ... (VIOLATED)' into native_violations, while rotamer_vetoes and pair_vetoes stay empty and is_empty() stays True, so nothing reaches the MILP and nothing rejects the iteration. `check_restraints` says of this case "no design position involved, so no choice of rotamer can change it" — true of rotamer selection, false of the run. Scratch: /private/tmp/claude-501/-Users-supantha-Documents-test123/a4581fba-d9ac-4b77-93bd-2ae3a83c7cd7/scratchpad/paper-optzyme/catdist.py

**fix:** Keep the 0-design-position branch live instead of terminal: record those restraints on the plan, re-measure them after perturbation and after each redock, and either reject the iteration (as `filter_reject` already does for the pore filter and the IPRO+/- interaction floor) or add the deviation as a penalty. Copy `native_violations` and `unsupported` through the multi-assembly merge. Failing that, state explicitly in section 1c and in `check_restraints`' own message that a restraint with no design-position atom is enforced nowhere once the backbone moves or a partner is redocked, and withdraw section 1b's claim that the thioesterase restraint set is expressible.

---

## [15] SHOULD-FIX Section 1a's exclusion list is missing one of the SI's ten criteria, and the omission changes which positions the rule reproduces

**file:** docs/DESIGN.md:60-74 (section 1a exclusion table)

**source:** ACS Catal 2017 SI, Table S4 footnotes c-l — refs/papers/user-downloads/si.txt lines 1025, 1027, 1029, 1031, 1034, 1036, 1039, 1041, 1043, 1046

**detail:** Section 1a presents its table as the SI's exclusion rule ("a residue is excluded if any of the following holds") and says "Two of the nine are geometric and computable". The SI has TEN exclusion footnotes, c through l. The missing one is footnote e, at si.txt:1029: "legacy version of IPRO could not handle mutations from proline". Every other row maps correctly (c triad, d oxyanion hole, f mutagenesis, g 4.5 A sulfur, h Ca/Cb 0.75 A, i R1 abolished activity, j flexible loops, k >=40% conservation, l alignment gap). Worth saying rather than omitting, for two reasons: it is the ONLY one of the ten this tool could detect trivially (is the native residue PRO?), where section 1a's point is that seven of them need outside knowledge; and it is a legacy-tool limitation rather than biology — P110 carries footnote e and is rank 3 in R1/R2, but is rank 1 and UNANNOTATED in the R3-R4 column, i.e. the exclusion had been lifted by the later rounds.

**evidence:** Applying section 1a's nine criteria to the SI's R1 ranked list gives the top-eight unexcluded as Y145, P110, R108, L11, A111, G72, L76, F139. Applying the SI's ten gives Y145, R108, L11, A111, G72, L76, F139, I107 — P110 (rank 3, annotated e) is replaced by I107 (rank 18). So section 1a's list as written does not reproduce the paper's R1/R2 design-position set. Also missing from the row for footnote h: the SI's "(glycines not considered)" qualifier.

**fix:** Add footnote e as a tenth row, marked as a legacy-IPRO limitation that does not apply to this implementation (PyMOL's wizard mutates from PRO), and note that P110 is excluded in R1/R2 but is the top-ranked design position in R3-R4. Change "Two of the nine" to "Two of the ten". Add the "(glycines not considered)" qualifier to the Ca/Cb row.

---

## [16] BLOCKER The "6.24 Å wild-type eyelet" is measured on a 13-substitution structure, not on the wild type

**file:** /Users/supantha/Documents/test123/ipro-agent/docs/DESIGN.md:366

**source:** Nat Commun 2018 (PoreDesigner) SI p.8, "Ellipse fitting method" (WT major/minor 11 Å / 7 Å, area 60.47 Å²)

**detail:** §1h, spec_full.yaml:247/260-262, EXPECTED.md:321/337 and loop.py:561/587 all present 6.24 Å as this tool's measurement of the WILD-TYPE OmpF eyelet and compare it to the published 11 × 7 Å. It is not the wild type. `score` measures `design_fixed + chosen rotamers`, and `IPROLoop._native_choice` substitutes the lowest-self-energy ALLOWED rotamer wherever the native residue is outside the spec's 9-residue alphabet — which is 13 of the 25 positions, 8 of them to Trp. So 6.24 Å is the bottleneck of a partly redesigned, partly Trp-filled pore. Worse, §1h:367 retracts the earlier 8.1 Å figure as "does not reproduce", but 8.1 Å is exactly what the wild type gives: the repo's own regression test (tests/test_ipro.py:1722, which measures the deposited coordinates) reproduces 8.07 Å and asserts 5.5-10 Å. The retraction replaced a correct number with a mutant's number.

**evidence:** `python -m ipro_agent score examples/ompf_pore/spec_full.yaml` -> min_pore_radius 3.1213 (6.243 Å), pore_axis_clearance 1.295 Å; its own note lists the 13 substituted positions (R42W E62W N64W R82W G110W D113W E117M G119L G120L D121V S125W G131W R132L). Same code path with `allowed` pinned to each native identity: 6.799 Å, clearance 3.254 Å. Deposited coordinates through the repo's own test path (axis_from_lining over the 25 CAs, spacing 0.5, max_centre_offset 6.0, no window): 8.071 Å; with the ±4 Å window and the PDB element column: 8.087 Å. Published WT: major 11 Å, minor 7 Å, π·5.5·3.5 = 60.48 Å² (matches the SI's 60.47).

**fix:** Quote the wild-type number from the input structure (≈8.1 Å, i.e. reinstate the retracted figure, or 6.80 Å if the comparison is meant to use PyMOL rotamers at native identities) and relabel 6.24 Å as "the starting design after substitution at the 13 positions whose native residue the alphabet excludes". Update docs/DESIGN.md:366-369 and 391, spec_full.yaml:247/260-262, EXPECTED.md:321/337 and the loop.py note together; ipro_agent/pore.py:267 already says 8.1 Å and needs no change.

---

## [17] SHOULD-FIX The axis-clearance discriminator was abandoned on a non-comparable pair; measured like-for-like it separates the two specs cleanly

**file:** /Users/supantha/Documents/test123/ipro-agent/docs/DESIGN.md:823

**source:** Nat Commun 2018 SI p.7, PoreAnalyzer step 2.1-2.2 ("oriented" structure; per-slice pore centre)

**detail:** §7d's "one confirmed defect left unfixed" rests on the table at :823 — bad axis 0.30 Å vs good axis 1.30 Å, "both below a water's 1.4 Å" — and on that basis a clearance threshold was implemented and reverted (:827). The two numbers are not comparable. The 4-position smoke spec keeps all four native identities (V18 L20 M38 Y40 are all inside the alphabet), so 0.30 Å is a native-geometry clearance. The 25-position figure is taken after 13 substitutions, 8 to Trp, which pack the lumen and pull the clearance down. Measured on identical (native) geometry the same two axes give 0.30 Å and 3.25 Å — a 10× separation that the 1.4 Å water rule splits perfectly. The reason this went unnoticed is that the clearance is intended to validate the AXIS (a fixed property of the input), but it is computed inside `_measure_geometry` against each candidate's packed design, so it drops as the design narrows and stops being an axis diagnostic. Separately, note that the SI gives no axis-fitting algorithm at all: it starts from "the oriented OmpF pdb structure" and re-identifies a pore centre per slice, so the published method could not fail this way and nothing in the SI supports fitting the axis through the design positions.

**evidence:** ipro_agent/loop.py:565-570 computes clearance from `design_atoms` (template + chosen rotamers). Measured via IPROLoop.measure_native: spec_smoke 0.304 Å / 1.528 Å (picked identities VAL LEU MET TYR = all native); spec_full 1.295 Å / 6.243 Å (13 substitutions); spec_full with `allowed` pinned to the natives 3.254 Å / 6.799 Å.

**fix:** Compute the axis-quality clearance once in `prepare()`, from the input structure with its native side chains, and keep it separate from the per-candidate geometry dict. The 1.4 Å rule then passes spec_full (3.25 Å) and flags spec_smoke (0.30 Å), so §7d's open defect can be closed rather than documented.

---

## [18] SHOULD-FIX Stabilize minimizes the design molecule's own energy where DESIGN.md and the docstring claim it minimizes the complex energy

**file:** /Users/supantha/Documents/test123/ipro-agent/ipro_agent/goals.py:705

**source:** Structure 2020 (IPRO+/−), Results, EcTBL/1ERM case: "Enzyme-inhibitor complex energy score was minimized"

**detail:** DESIGN.md:35 records the goal as IPRO+/−'s `min E_complex s.t. IE ≥ 0.75·IE_WT`, and Stabilize's own docstring says it provides "IPRO+/-'s constrained form: minimize complex energy". The code scores `ctx.design_only` (falling back to `ctx.total` only when design_only is exactly 0.0). The complex energy is `ctx.total` (design + interaction + partner internals) — the IPRO Suite paper's "complex energy of the system". Since the 0.75 floor only bounds the interaction term rather than fixing it, the two objectives rank candidates differently inside the feasible region, and the repo can prefer the candidate with the worse complex energy. The interaction floor itself is faithful: the paper's "not worse by more than 25%" matches `ie > 0.75 × reference -> reject` for negative IEs (DESIGN.md's "IE ≥ 0.75·IE_WT" is loose notation, but the code's sense is right).

**evidence:** Constructed two contexts with reference_interaction −20, floor −15. A: design_only −100.0, IE −20.0, E_complex −120.0. B: design_only −103.0, IE −15.0, E_complex −118.0. `filter_reject` returns None for both. `Stabilize.score` gives A −100.000, B −103.000, so the repo picks B; the paper's objective (min E_complex) picks A.

**fix:** Score `ctx.total` when the spec has targets (that is the complex energy the paper minimizes), keeping `ctx.design_only` for the no-partner case; or, if the design-only energy is the intended objective, correct DESIGN.md:35 and the class docstring to say so instead of citing `min E_complex`.

---

## [19] SHOULD-FIX PoreDesigner's anti-occlusion constraint is missing, and the docs describe the objective as occluding the pore — the opposite of the paper

**file:** /Users/supantha/Documents/test123/ipro-agent/ipro_agent/goals.py:459

**source:** Nat Commun 2018, Results: explicit constraint that pore-residue hetero-atoms stay beyond vdW contact of the water-wire oxygen

**detail:** The paper imposes a hard steric constraint so that "the distance between any hetero-atom of the pore constriction residue and the water wire oxygen was greater than the sum of their van der Waals radii", stating its purpose as precluding designs that occlude the pore. Nothing in the repo implements or mentions it (grep for hetero-atom / occlude / vdW sum finds only the inverted gloss), and §4 does not list it as not-implemented. Meanwhile DESIGN.md:33 and goals.py:21/371 say PoreDesigner maximized the probe interaction "to occlude the lumen" / "a worse water interaction means a more occluded lumen". The paper's goal was a narrow but OPEN hydrophobic pore with minimal water-wire interaction; occlusion is what the extra constraint exists to prevent. The consequence is real, not semantic: `direction: narrow` scores −IE with no upper bound, so a side chain driven into the wire scores best.

**evidence:** With spec_full.yaml's own goal_options (direction narrow, pore_diameter_max 4.0, pore_diameter_min null): an occluding candidate (bottleneck 0.8 Å, wire clashing, IE +50) passes `filter_reject` (None) and scores −50.000, while a paper-style compliant design (bottleneck 3.5 Å, IE −0.2) scores +0.200 and an open 4.5 Å pore is rejected by the filter. The built-in pair cap (±10 kcal/mol) bounds each pair but not the sum.

**fix:** Add a §4 row for the constraint, and express it as a rotamer veto (restraints.py already vetoes rotamer choices) when a probe molecule is present: forbid any rotamer heavy atom within r_vdw(i)+r_vdw(j) of a probe atom. Also correct the "to occlude the lumen" wording in DESIGN.md:33 and goals.py:21/371.

---

## DROPPED

- **The MILP objective is Eq. 4's left-hand side (total energy), not Eq. 3 (the binding score DESIGN.md cites)** — The code's objective is the published one; the finding checks it against the wrong equation. (1) Three independent sources state the implemented form verbatim: IPRO Suite (JCC 2015) eq. (1), described one line later as m
- **DESIGN.md says the implementation applies a single 12 A cutoff to the H-bond term; the built-in term is actually zero beyond 4.0 A** — I confirmed both halves of the report and still conclude it is not a finding.

The paper says what is quoted. refs/papers/ipro2006_saraf_fulltext.txt, Step ii, gives per-interaction cutoffs including "hydrogen bond = 3 A
- **DESIGN.md says both temperature calibrations are checked in tests/test_ipro.py; only the 3640 K one is** — Refuted: this is a test-coverage/wording nitpick, not a paper-fidelity defect, and it cannot change a result.

What the paper says, verified directly: refs/papers/user-downloads/jcc_suite.txt:511-512 gives docking start 
- **MILP.md still states the original carries a hardcoded 20-entry amino-acid reference-energy table — a claim DESIGN.md §4 explicitly retracts and the source contradicts** — REFUTED — and the finding is inverted: MILP.md:187 is correct, and the DESIGN.md §4 "correction" it cites as authority is the false statement.

The released IPRO Suite source DOES carry a hardcoded 20-entry per-amino-aci
- **The "constant cannot affect the argmin" argument is applied to the stability constraint, where the dropped constant does change the feasible set** — The source transcription is accurate (eq4_M12.gif does carry +E^sb; eqinl_M7.gif defines it as substrate-backbone energy; milp.py:738-751 emits self+pair only), but every inference drawn from it fails.

1. NOT A SILENT D
- **§4 presents the -ln(p) rotamer-probability term as what IPRO adds; the paper says those terms are excluded by default** — The paper sentence is genuine (verified on the rendered page image of JCC 2015 p.255 col.2, not just OCR), and the source citations are accurate (`probability = True` hardcoded at IPRO_FUNCTIONS.py:587 in all three relea
- **Two defaults marked "source" are stated in the Suite paper, so §6's "defaults the papers leave ambiguous only" does not hold for them** — Refuted on its second half, and the proposed fix is itself wrong against the paper.

ENSEMBLE_SIZE — the paper does NOT state this number. Verified on p.257 (PDF page 7) and jcc_suite.txt:537-545: the paper's "10" is the
- **The w-is-continuous note cites only the GAMS path, while the 2015 paper calls w binary and CPLEX.py declares it binary; the strongest support (the 2006 paper's own declaration) goes uncited** — Every factual claim in the report checks out, but it does not identify a fidelity defect under any of the four reportable categories, and it cannot change a result.

(1) No result can differ. Measured on 12 random instan
- **MILP.md says CPLEX is driven directly only for OptMAVEn; it is one of the two supported paths for ordinary rotamer selection** — Refuted on the code, and already covered in the docs. The paper sentence is real: J Comput Chem 36:251-263, p. 257 col. 1 (PDF page 7, verified on the page image and in refs/papers/user-downloads/jcc_suite.txt:495-497), 
- **The cited identity-coupling equation is transcribed without the hybrid index and the identity-restricted sum** — Refuted on both halves, after reading eq. 17 (refs/papers/eqimg-2006/eq17_M49.gif), eq. 16 (eq16_M48.gif), eq. 18 (eq18_M50.gif) and the surrounding 2006 text.

1. The missing hybrid index is not a slip — it is required,
- **OptZyme's IE_S<0 / IE_TSA<0 run constraints are absent, leaving the `kcat` objective unbounded in the wrong direction** — The quotation is accurate and every number reproduces exactly (single-RT: -5.064 / -168.798 / -329.156; Eq-18 weights: -3.184 / -1.082 / -0.797; RT_TSA/RT_S = 0.039566), and the mechanics are right: OPTION_KEYS = {parame
- **Section 1a answers "yes, given the atom" for two exclusions the shipped `positions` command has no way to be given the atom for** — REFUTED. The code facts are right but the fidelity claim is not; three of the four assertions about the papers and the docs are wrong, and the item cannot change a result.

1) The papers make design positions INPUT to IP
- **The published 4 Å threshold is a major-axis number but is applied to an inscribed-circle (minor-axis-like) measurement** — The paper reading checks out, but the finding's two load-bearing claims — that no doc states the bias direction, and that the bias is "≈1.5×" — are both false, and the third (EXPECTED.md) is self-corrected in the same fi
- **IPRO+/−'s actual indel design mechanism is not listed in §4 as not-implemented** — The paper's facts are reported correctly, but the gap is not silent and is not separable from what §4 already lists. Structure 2020's STAR Methods ("Integration of Loop Closure into IPRO") defines indel-DPs as "the desig
- **Inserted-residue phi/psi defaults (−139/135) have no recorded provenance, and the source samples them instead** — Refuted on four independent grounds; the paper passage says what the colleague quotes, but it does not govern the code path in question.

1. The §2 premise is misread. docs/DESIGN.md:454-456 scopes the claim precisely: "
- **ring_flatness rewards the known-bad axis and penalizes the known-good one, and is vacuous for 3 positions** — Refuted on two independent grounds. (1) The cited source says nothing of the kind. Nat Commun 2018 SI p.7, PoreAnalyzer sub-step 1, takes "the oriented OmpF pdb structure" as an input, and SI §1.2 Step 1 (p.3) orients 2o
- **min_class implements constraint 4's prose (≥50% of DPs) rather than its algebra (#LONG ≥ #SHORT)** — Refuted at the source. I rendered PoreDesigner SI p.5 and p.4 from the PDF (the equations do not survive pdftotext). Constraint (4) as actually printed is `Σ_{i=1..N} y_ir·LONG_r − Σ_{i=1..N} y_ir·SHORT_r ≥ 0, ∀ r ∈ A` —
- **profile_from_atomset drops the AtomSet's element column, re-opening the fixed element-typing bug** — REFUTED. The code observation is factually correct but it is neither a paper-fidelity defect nor a false documentation claim, and it cannot change any result.

1. The paper specifies nothing here. PoreDesigner's SI (refs
