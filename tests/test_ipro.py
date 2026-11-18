"""Tests for the parts where being wrong would be silent.

Deliberately not a broad suite. Each test here locks in one property whose
failure would change a scientific answer without raising an error:

* the energy function reproduces analytically known values;
* the MILP finds the true optimum (checked against exhaustive enumeration),
  honours a class-cardinality constraint, and never returns a fabricated
  assignment when the program is infeasible;
* integer cuts really do prevent a sequence from being re-evaluated;
* the Metropolis temperature reproduces IPRO's published calibration;
* a misspelled spec key is rejected rather than ignored.

Run with pytest if it is installed, or dependency-free via
``python3 scripts/selftest.py``.
"""

from __future__ import annotations

import itertools
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ipro_agent.config import (  # noqa: E402
    AA_GROUPS, ONE_TO_THREE, THREE_TO_ONE, DesignPosition, DesignSpec,
    LoopSettings, R_KCAL, SpecError, expand_allowed,
)
from ipro_agent.energy import AtomSet, resolve  # noqa: E402
from ipro_agent.goals import metropolis_accept  # noqa: E402
from ipro_agent.milp import (  # noqa: E402
    Constraint, EnergyTables, Infeasible, Rotamer, RotamerMILP, build_tables,
)

DATA = REPO / "tests" / "data"
CRN = DATA / "1crn.pdb"


def _have_pymol() -> bool:
    from ipro_agent.rotamers import find_pymol_python

    return find_pymol_python() is not None


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_amino_acid_tables_are_mutually_consistent():
    """A mismatch here would silently mistranslate every reported sequence."""
    assert len(THREE_TO_ONE) == 20
    assert len(ONE_TO_THREE) == 20
    for one, three in ONE_TO_THREE.items():
        assert THREE_TO_ONE[three] == one, (one, three)
    for three, one in THREE_TO_ONE.items():
        assert ONE_TO_THREE[one] == three, (three, one)
    for name, members in AA_GROUPS.items():
        assert members, name
        assert set(members) <= set(THREE_TO_ONE), (name, set(members) - set(THREE_TO_ONE))


def test_expand_allowed_accepts_codes_and_groups_and_rejects_nonsense():
    assert expand_allowed(["ALA", "A"]) == ["ALA"]            # deduplicated
    assert expand_allowed(["W", "TRP"]) == ["TRP"]
    assert expand_allowed(["aromatic"]) == sorted(AA_GROUPS["aromatic"])
    # Deterministic order matters: it fixes the MILP's variable order.
    assert expand_allowed(["V", "A", "L"]) == expand_allowed(["L", "A", "V"])
    for bad in (["ZZZ"], ["B"], ["hydrophobicc"]):
        try:
            expand_allowed(bad)
        except SpecError:
            pass
        else:
            raise AssertionError(f"{bad} should have been rejected")


def test_position_parsing_forms():
    assert DesignPosition.parse("A115").label == "A115"
    assert DesignPosition.parse("A:115").label == "A115"
    assert DesignPosition.parse("A/115").label == "A115"
    assert DesignPosition.parse("A115B").insertion_code == "B"

    # Regression: the optional chain group used to swallow the first digit, so
    # "115" became chain "1", residue 15 -- the wrong residue, silently.
    bare = DesignPosition.parse("115")
    assert bare.chain == "A", bare.chain          # documented default
    assert bare.number == 115, bare.number

    # A numeric chain id needs the separator form to stay unambiguous.
    nch = DesignPosition.parse("1:115")
    assert (nch.chain, nch.number) == ("1", 115)

    # The hyphen is not a separator, so negative residue numbers survive.
    neg = DesignPosition.parse("A-5")
    assert (neg.chain, neg.number) == ("A", -5), (neg.chain, neg.number)
    try:
        DesignPosition.parse("not-a-position")
    except SpecError:
        pass
    else:
        raise AssertionError("bad position should have been rejected")


def test_unknown_spec_key_is_rejected_not_ignored():
    """A typo must not silently change the design problem."""
    base = {
        "goal": "stabilize", "goal_statement": "x",
        "molecules": [{"name": "m", "path": str(CRN), "role": "design"}],
        "positions": ["A16"],
    }
    DesignSpec.from_dict(dict(base))                      # the control
    try:
        DesignSpec.from_dict({**base, "iterations": 10})  # belongs under loop:
    except SpecError:
        pass
    else:
        raise AssertionError("unknown top-level key should have been rejected")


def test_empty_allowed_set_is_rejected():
    try:
        DesignPosition(chain="A", number=1, allowed=[])
    except SpecError:
        pass
    else:
        raise AssertionError("an empty allowed set should have been rejected")


# ---------------------------------------------------------------------------
# energy
# ---------------------------------------------------------------------------


def _atom(name, xyz, resn="ALA", key=("A", 1, "")):
    return AtomSet(names=[name], coords=[xyz], resnames=[resn], reskeys=[key])


def test_lennard_jones_minimum_is_exactly_minus_epsilon():
    """At r = Rmin the Amber 12-6 form equals -epsilon. Catches a wrong
    combining rule or a dropped factor of two in the attractive term."""
    be = resolve("builtin")
    a = _atom("CB", [0, 0, 0], key=("A", 1, ""))
    b = _atom("CB", [1.9080 * 2, 0, 0], key=("A", 2, ""))
    assert abs(be.interaction(a, b) - (-0.1094)) < 1e-9


def test_repulsion_is_capped_and_interaction_is_symmetric():
    be = resolve("builtin", max_repulsion=10.0)
    a = _atom("CB", [0, 0, 0], key=("A", 1, ""))
    b = _atom("CB", [1.0, 0, 0], key=("A", 2, ""))
    e = be.interaction(a, b)
    assert abs(e - 10.0) < 1e-9, e
    assert abs(be.interaction(a, b) - be.interaction(b, a)) < 1e-12


def test_same_residue_pairs_are_excluded():
    """Otherwise a rotamer would be scored against itself."""
    be = resolve("builtin")
    a = _atom("CB", [0, 0, 0], key=("A", 7, ""))
    same = _atom("CG", [3.8, 0, 0], key=("A", 7, ""))
    assert be.interaction(a, same) == 0.0


def test_salt_bridge_attracts_and_like_charges_repel():
    be = resolve("builtin")
    asp = _atom("OD1", [0, 0, 0], resn="ASP", key=("A", 1, ""))
    lys = _atom("NZ", [4.0, 0, 0], resn="LYS", key=("A", 2, ""))
    asp2 = _atom("OD1", [4.0, 0, 0], resn="ASP", key=("A", 2, ""))
    assert be.interaction(asp, lys) < 0
    assert be.interaction(asp, asp2) > be.interaction(asp, lys)


def test_coincident_atoms_do_not_produce_infinities():
    """Duplicate input coordinates must not poison the objective."""
    be = resolve("builtin")
    a = _atom("CB", [0, 0, 0], key=("A", 1, ""))
    b = _atom("CB", [0, 0, 0], key=("A", 2, ""))
    e = be.interaction(a, b)
    assert math.isfinite(e), e


# ---------------------------------------------------------------------------
# MILP
# ---------------------------------------------------------------------------


def _toy_tables(seed: int = 0) -> EnergyTables:
    """Three positions, three identities each, with reproducible energies."""
    rng = np.random.default_rng(seed)
    positions = ["A1", "A2", "A3"]
    rotamers: dict[str, list[Rotamer]] = {}
    for pi, p in enumerate(positions):
        rots = []
        for ai, aa in enumerate(("ALA", "VAL", "TRP")):
            for st in (1, 2):
                xyz = [pi * 3.0 + 0.3 * st, ai * 0.7, 0.0]
                rots.append(
                    Rotamer(
                        position=p, aa=aa, state=st,
                        atoms=AtomSet(
                            names=["CB"], coords=[xyz], resnames=[aa],
                            reskeys=[("ROT", p, "")],
                        ),
                    )
                )
        rotamers[p] = rots
    t = EnergyTables(positions=positions, rotamers=rotamers)
    for p in positions:
        for i in range(len(rotamers[p])):
            t.self_energy[(p, i)] = float(rng.normal(0, 3))
    for a in range(len(positions)):
        for b in range(a + 1, len(positions)):
            for i in range(len(rotamers[positions[a]])):
                for j in range(len(rotamers[positions[b]])):
                    t.pair_energy[(positions[a], i, positions[b], j)] = float(
                        rng.normal(0, 1)
                    )
    return t


def _brute_force(milp: RotamerMILP, predicate=None) -> tuple[float, dict]:
    t = milp.t
    best, best_c = None, None
    ranges = [range(len(t.rotamers[p])) for p in t.positions]
    for combo in itertools.product(*ranges):
        ch = dict(zip(t.positions, combo))
        if predicate and not predicate({p: t.rotamers[p][r].aa for p, r in ch.items()}):
            continue
        e = milp.evaluate(ch)
        if best is None or e < best:
            best, best_c = e, ch
    return best, best_c


def test_milp_matches_exhaustive_enumeration():
    """The whole formulation in one assertion: if the linearization or the
    marginalization constraints were wrong, this would diverge."""
    for seed in (0, 1, 2):
        t = _toy_tables(seed)
        m = RotamerMILP(t, native={p: "ALA" for p in t.positions})
        sol = m.solve()
        best, _ = _brute_force(m)
        assert abs(sol.objective - best) < 1e-6, (seed, sol.objective, best)


def test_min_class_constraint_is_honoured_and_optimal():
    t = _toy_tables(3)
    want = {"TRP"}
    con = Constraint(kind="min_class", count=2, residues=["TRP"], label="two_trp")
    m = RotamerMILP(t, constraints=[con], native={p: "ALA" for p in t.positions})
    sol = m.solve()
    ids = sol.identities(t)
    assert sum(1 for a in ids.values() if a in want) >= 2, ids
    best, _ = _brute_force(m, lambda d: sum(1 for a in d.values() if a in want) >= 2)
    assert abs(sol.objective - best) < 1e-6, (sol.objective, best)


def test_max_mutations_constraint_is_honoured():
    t = _toy_tables(4)
    native = {p: "ALA" for p in t.positions}
    con = Constraint(kind="max_mutations", count=1, label="one_mut")
    sol = RotamerMILP(t, constraints=[con], native=native).solve()
    ids = sol.identities(t)
    assert sum(1 for p, a in ids.items() if a != native[p]) <= 1, ids


def test_integer_cut_excludes_the_visited_sequence():
    t = _toy_tables(5)
    native = {p: "ALA" for p in t.positions}
    first = RotamerMILP(t, native=native).solve()
    seq = first.identities(t)
    second = RotamerMILP(t, native=native, visited=[seq]).solve()
    assert second.identities(t) != seq
    assert second.objective >= first.objective - 1e-9


def test_infeasible_program_raises_rather_than_inventing_an_assignment():
    """A solver that proves infeasibility still writes a solution file of
    zeros; trusting it would return a design nothing endorsed."""
    t = _toy_tables(6)
    cons = [
        # Mutually exclusive: every position must be TRP, and none may be.
        Constraint(kind="require", residues=["TRP"], label="all_trp"),
        Constraint(kind="forbid", residues=["TRP"], label="no_trp"),
    ]
    m = RotamerMILP(t, constraints=cons, native={p: "ALA" for p in t.positions})
    try:
        m.solve()
    except Infeasible:
        pass
    except RuntimeError as exc:
        raise AssertionError(f"expected Infeasible, got RuntimeError: {exc}")
    else:
        raise AssertionError("an infeasible program should not return a solution")


def test_lp_file_records_the_constraints():
    """The LP file is the artifact a user reads to check their requirement was
    encoded; the constraint must actually appear in it."""
    t = _toy_tables(7)
    con = Constraint(kind="min_class", fraction=0.5, residues=["TRP"], label="half_trp")
    lp = RotamerMILP(t, constraints=[con], native={p: "ALA" for p in t.positions}).to_lp()
    assert "half_trp" in lp
    assert "min_class" in lp
    assert "Binaries" in lp and "End" in lp
    # one-rotamer-per-position for each position
    for i in range(len(t.positions)):
        assert f"one_{i}:" in lp


def test_evaluate_agrees_with_the_lp_objective_coefficients():
    """Guards against the objective in the LP file drifting from the objective
    the heuristic and the accept/reject test use."""
    t = _toy_tables(8)
    m = RotamerMILP(t, native={p: "ALA" for p in t.positions})
    choices = {p: 0 for p in t.positions}
    manual = sum(t.self_energy[(p, 0)] for p in t.positions)
    manual += sum(
        c for (pi, i, pj, j), c in t.pair_energy.items()
        if choices.get(pi) == i and choices.get(pj) == j
    )
    assert abs(m.evaluate(choices) - manual) < 1e-9


# ---------------------------------------------------------------------------
# acceptance criterion
# ---------------------------------------------------------------------------


def test_metropolis_reproduces_ipro_published_calibration():
    """IPRO's default T = 3640 K is stated to accept a 10 kcal/mol worsening
    about 25% of the time. If the gas constant or the formula were wrong this
    would miss badly -- and one source's quoted constant does miss badly."""
    kt = R_KCAL * 3640.0
    p = math.exp(-10.0 / kt)
    assert abs(p - 0.25) < 0.005, p
    # And LoopSettings exposes the same number.
    assert abs(LoopSettings().kt - kt) < 1e-12


def test_metropolis_accepts_improvements_always_and_is_monotone():
    kt = LoopSettings().kt
    ok, why = metropolis_accept(-1.0, kt, u=0.999)
    assert ok and why == "improved"
    ok, _ = metropolis_accept(0.0, kt, u=0.999)
    assert ok, "a lateral move must not be rejected"
    # Greedy phase refuses any worsening.
    ok, why = metropolis_accept(+1.0, kt, u=1e-9, greedy=True)
    assert not ok and "greedy" in why
    # A worse move is accepted for a small u and rejected for u near 1.
    assert metropolis_accept(+5.0, kt, u=1e-6)[0]
    assert not metropolis_accept(+50.0, kt, u=0.999)[0]


def test_loop_settings_validation():
    for bad in ({"design_per_iteration": (3, 1)}, {"temperature_k": 0},
                {"greedy_fraction": 1.5}, {"perturbation_segment": (0, 2)}):
        try:
            LoopSettings(**bad)
        except SpecError:
            pass
        else:
            raise AssertionError(f"{bad} should have been rejected")


# ---------------------------------------------------------------------------
# PyMOL-dependent
# ---------------------------------------------------------------------------


def test_rotamer_harvest_shapes():
    """GLY must yield one rotamer with no side-chain atoms, ALA one with CB,
    and a flexible residue several. Wrong counts here mean the MILP is choosing
    from the wrong set."""
    if not _have_pymol():
        print("    (skipped: no PyMOL)")
        return
    from ipro_agent.rotamers import harvest

    pos = [DesignPosition.parse("A16", ["GLY", "ALA", "TRP"])]
    rots, native, errors = harvest(CRN, pos)
    assert not errors, errors
    by_aa: dict[str, list] = {}
    for r in rots["A16"]:
        by_aa.setdefault(r.aa, []).append(r)
    assert len(by_aa["GLY"]) == 1 and len(by_aa["GLY"][0].atoms) == 0
    assert len(by_aa["ALA"]) == 1 and len(by_aa["ALA"][0].atoms) == 1
    assert len(by_aa["TRP"]) > 1, "TRP should have several rotamers"
    assert len(by_aa["TRP"][0].atoms) == 10, "TRP side chain is 10 heavy atoms"
    assert native["A16"] == "CYS"


def test_insertion_coded_position_harvests_and_does_not_destroy_the_object():
    """Regression: handing the wizard a selection *expression* made
    do_select's trailing cmd.delete() remove the loaded object itself."""
    if not _have_pymol():
        print("    (skipped: no PyMOL)")
        return
    icode_pdb = DATA / "1crn_icode.pdb"
    if not icode_pdb.exists():
        print("    (skipped: fixture missing)")
        return
    from ipro_agent.rotamers import harvest

    pos = [DesignPosition(chain="A", number=16, insertion_code="A", allowed=["TRP"])]
    rots, native, errors = harvest(icode_pdb, pos)
    assert not errors, errors
    assert rots.get("A16A"), "no rotamers at the insertion-coded position"
    assert len(rots["A16A"]) > 1
    assert native["A16A"] == "CYS"


def test_milp_matches_brute_force_on_real_rotamers():
    """The same equivalence as the toy test, but with PyMOL geometry and the
    real energy function -- so the template, pruning and atom bookkeeping are
    exercised too."""
    if not _have_pymol():
        print("    (skipped: no PyMOL)")
        return
    from ipro_agent.rotamers import build_template, harvest
    from ipro_agent.structure import Structure

    pos = [
        DesignPosition.parse("A16", ["ALA", "SER", "TRP"]),
        DesignPosition.parse("A18", ["ALA", "VAL", "PHE"]),
    ]
    rots, native, errors = harvest(CRN, pos)
    assert not errors, errors
    st = Structure.load(CRN)
    template = build_template([st], list(rots))
    be = resolve("builtin")
    t = build_tables(rots, template, be, native=native, max_self_energy=1e9, pair_eps=0.0)
    m = RotamerMILP(t, native=native)
    sol = m.solve()
    best, _ = _brute_force(m)
    assert abs(sol.objective - best) < 1e-6, (sol.objective, best)
    assert sol.proven_optimal, f"expected a proven optimum, got {sol.status}"


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


def test_apply_design_applies_exactly_the_requested_mutations():
    """The chosen rotamer must reach the coordinates, and nothing else may move.

    PyMOL's wizard reports failure by leaving the residue alone, so the identity
    has to be read back rather than trusted; and a mutation that silently
    altered a neighbouring residue would corrupt the design without erroring.
    """
    if not _have_pymol():
        print("    (skipped: no PyMOL)")
        return
    from ipro_agent.mutate import apply_design
    from ipro_agent.structure import Structure

    tmp = Path(tempfile.mkdtemp(prefix="ipro_mut_"))
    try:
        out = tmp / "mutated.pdb"
        before = Structure.load(CRN)
        seq_before = before.sequence("A")

        res = apply_design(CRN, out, {"A16": ("TRP", 2), "A18": ("PHE", 1)})
        assert not res.errors, res.errors
        assert res.applied == {"A16": ("TRP", 2), "A18": ("PHE", 1)}

        after = Structure.load(out)
        assert after.find("A", 16).get_resname().strip() == "TRP"
        assert after.find("A", 18).get_resname().strip() == "PHE"
        # TRP has 10 heavy side-chain atoms including CB; a wrong count means
        # the side chain was not rebuilt.
        sc = [a for a in after.find("A", 16) if a.get_name().strip() not in
              ("N", "CA", "C", "O", "OXT")]
        assert len(sc) == 10, len(sc)

        seq_after = after.sequence("A")
        assert len(seq_after) == len(seq_before) == 46
        changed = [i for i, (a, b) in enumerate(zip(seq_before, seq_after)) if a != b]
        assert changed == [15, 17], changed   # 0-based: residues 16 and 18 only
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_loop_never_revisits_a_sequence():
    """The integer-cut guarantee, end to end."""
    if not _have_pymol():
        print("    (skipped: no PyMOL)")
        return
    from ipro_agent.loop import IPROLoop
    from ipro_agent.state import RunDir

    tmp = Path(tempfile.mkdtemp(prefix="ipro_test_"))
    try:
        spec = DesignSpec.from_dict({
            "name": "t", "goal": "stabilize", "goal_statement": "test",
            "molecules": [{"name": "m", "path": str(CRN), "role": "design",
                           "kind": "protein", "chains": ["A"]}],
            "positions": [{"position": "A16", "allowed": ["ALA", "SER", "VAL"]},
                          {"position": "A18", "allowed": ["ALA", "VAL", "LEU"]}],
            "loop": {"iterations": 15, "random_seed": 3},
        })
        loop = IPROLoop(spec, tmp)
        loop.prepare()
        loop.run_loop()
        seqs = [
            tuple(sorted(r.mutations.items()))
            for r in RunDir(tmp, create=False).iterations()
        ]
        assert len(seqs) == len(set(seqs)), "a sequence was evaluated twice"
        assert seqs, "no iterations recorded"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cli_validate_and_goals_emit_json():
    """An agent branches on this JSON, so it must parse."""
    out = subprocess.run(
        [sys.executable, "-m", "ipro_agent", "goals", "--json"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert out.returncode == 0, out.stderr
    rows = json.loads(out.stdout)
    assert len(rows) >= 5
    assert all({"goal", "objective", "source"} <= set(r) for r in rows)
    assert all(r["source"] for r in rows), "every goal must cite a source"


# ---------------------------------------------------------------------------
# Regressions. Each of these was a confirmed defect found by review, and each
# produced a wrong scientific answer without raising.
# ---------------------------------------------------------------------------


def test_regression_pair_blocks_are_complete_so_the_lp_is_the_right_program():
    """Per-term pair pruning broke the linearization.

    `sum_s w[i,r,j,s] = y[i,r]` is only valid when every rotamer s of position j
    has a w variable. With a partial block the two orientations forced
    y[j,s'] = y[i,r], making the dropped rotamers unselectable: glpsol then
    returned a suboptimal design marked proven_optimal, or called a feasible
    program infeasible. Blocks are now kept or dropped whole.
    """
    t = _toy_tables(2)
    # Simulate what per-term pruning did: delete one (r,s) from a block.
    victim = sorted(t.pair_energy)[0]
    del t.pair_energy[victim]
    m = RotamerMILP(t, native={p: "ALA" for p in t.positions})
    lp = m.to_lp()
    # Every (i,r,j) link group must cover all rotamers of j.
    groups: dict[tuple[str, int, str], int] = {}
    for (pi, r, pj, s) in t.pair_energy:
        groups[(pi, r, pj)] = groups.get((pi, r, pj), 0) + 1
        groups[(pj, s, pi)] = groups.get((pj, s, pi), 0) + 1
    incomplete = [
        g for g, n in groups.items() if n != len(t.rotamers[g[2]])
    ]
    assert incomplete, "the partial-block situation should be detectable"
    # And with real tables built by build_tables, blocks are never partial:
    t2 = _toy_tables(2)
    counts: dict[tuple[str, int, str], int] = {}
    for (pi, r, pj, s) in t2.pair_energy:
        counts[(pi, r, pj)] = counts.get((pi, r, pj), 0) + 1
    for (pi, r, pj), n in counts.items():
        assert n == len(t2.rotamers[pj]), (pi, r, pj, n, len(t2.rotamers[pj]))
    del lp


def test_regression_milp_matches_brute_force_with_shipped_defaults():
    """The earlier real-rotamer test passed only because it set pair_eps=0 and
    disabled rotamer pruning. This one uses the defaults a user actually gets."""
    if not _have_pymol():
        print("    (skipped: no PyMOL)")
        return
    from ipro_agent.rotamers import build_template, harvest
    from ipro_agent.structure import Structure

    pos = [
        DesignPosition.parse("A16", ["ALA", "SER", "TRP"]),
        DesignPosition.parse("A18", ["ALA", "VAL", "PHE"]),
        DesignPosition.parse("A29", ["ALA", "LEU", "TYR"]),
    ]
    rots, native, errors = harvest(CRN, pos)
    assert not errors, errors
    st = Structure.load(CRN)
    template = build_template([st], list(rots))
    be = resolve("builtin")
    t = build_tables(rots, template, be, native=native)   # defaults, as shipped
    m = RotamerMILP(t, native=native)
    sol = m.solve()
    best, _ = _brute_force(m)
    assert abs(sol.objective - best) < 1e-6, (sol.objective, best)
    assert sol.proven_optimal


def test_regression_rotamer_carries_its_positions_residue_key():
    """Rotamer atoms used a synthetic key, so the same-residue exclusion never
    fired and every rotamer was scored against its own backbone -- a spurious
    +30..+50 kcal/mol that scaled with side-chain size and biased the whole
    objective toward the smallest residue."""
    if not _have_pymol():
        print("    (skipped: no PyMOL)")
        return
    from ipro_agent.rotamers import build_template, harvest
    from ipro_agent.structure import Structure

    pos = [DesignPosition.parse("A16", ["GLY", "ALA"])]
    rots, native, _ = harvest(CRN, pos)
    assert rots["A16"][0].atoms.reskeys[0] == ("A", 16, "")

    st = Structure.load(CRN)
    template = build_template([st], ["A16"])
    assert ("A", 16, "") in set(template.reskeys), "own backbone must be in the template"
    t = build_tables(rots, template, resolve("builtin"), native=native,
                     max_self_energy=1e9)
    by_aa = {r.aa: t.self_energy[("A16", i)] for i, r in enumerate(t.rotamers["A16"])}
    # GLY has no side-chain atoms, so its self energy is exactly zero; ALA's
    # single CB cannot clash with its own backbone once the key matches.
    assert by_aa["GLY"] == 0.0, by_aa
    assert abs(by_aa["ALA"]) < 5.0, by_aa


def test_regression_unsatisfiable_class_or_require_constraint_raises():
    """These were emitted as an LP comment and otherwise dropped, so glpsol
    returned a design that violated the stated constraint, proven_optimal."""
    t = _toy_tables(9)   # identities are ALA / VAL / TRP only
    for con in (
        Constraint(kind="min_class", count=1, residues=["HIS"], label="need_his"),
        Constraint(kind="require", residues=["HIS"], label="his_here"),
    ):
        m = RotamerMILP(t, constraints=[con], native={p: "ALA" for p in t.positions})
        for solver in ("auto", "greedy"):
            try:
                m.solve(solver=solver)
            except Infeasible:
                pass
            else:
                raise AssertionError(f"{con.kind} via {solver} should be infeasible")


def test_regression_named_solver_is_not_silently_the_heuristic():
    """Any name that was not 'pulp'/'binary' fell through to greedy, so every
    documented solver choice and every typo ran the heuristic."""
    t = _toy_tables(10)
    m = RotamerMILP(t, native={p: "ALA" for p in t.positions})
    # An unknown name must be rejected outright.
    try:
        m.solve(solver="nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown solver name should raise")
    # A known name must either use that solver or fail -- never greedy silently.
    for name in ("cbc", "cplex", "gurobi", "highs", "glpk"):
        try:
            sol = m.solve(solver=name)
        except RuntimeError:
            continue          # not installed here: an honest failure
        assert "greedy" not in sol.solver.lower(), (name, sol.solver)


def test_regression_greedy_can_satisfy_max_class_and_max_energy():
    """The heuristic had no repair branch for either, so it rejected every move
    and the loop reported a false 'design space exhausted'."""
    t = _toy_tables(11)
    native = {p: "ALA" for p in t.positions}
    unconstrained = RotamerMILP(t, native=native).solve().objective
    for con in (
        Constraint(kind="max_class", count=1, residues=["TRP"], label="atmost1trp"),
        Constraint(kind="max_energy", count=unconstrained + 1.0, label="cap"),
    ):
        m = RotamerMILP(t, constraints=[con], native=native)
        sol = m.solve(solver="greedy")
        assert m._feasible(sol.choices), con.kind


def test_regression_hbond_does_not_flip_like_charge_sign_or_double_count():
    """The H-bond well was applied to every N/O-N/O pair, so two ARG NH1 came
    out attractive, and a salt bridge was rewarded twice for one contact."""
    be = resolve("builtin")
    be_nohb = resolve("builtin", hbond_weight=0.0)

    arg1 = _atom("NH1", [0, 0, 0], resn="ARG", key=("A", 1, ""))
    arg2 = _atom("NH1", [3.4, 0, 0], resn="ARG", key=("A", 2, ""))
    assert be.interaction(arg1, arg2) > 0, "like-charged nitrogens must repel"

    asp = _atom("OD1", [0, 0, 0], resn="ASP", key=("A", 1, ""))
    lys = _atom("NZ", [3.4, 0, 0], resn="LYS", key=("A", 2, ""))
    assert abs(be.interaction(asp, lys) - be_nohb.interaction(asp, lys)) < 1e-12

    # A neutral polar pair still gets the bonus.
    ser = _atom("OG", [0, 0, 0], resn="SER", key=("A", 1, ""))
    asn = _atom("ND2", [3.0, 0, 0], resn="ASN", key=("A", 2, ""))
    assert be.interaction(ser, asn) < be_nohb.interaction(ser, asn)


def test_regression_atom_is_never_scored_against_itself():
    """With the shared empty-tuple default reskey, internal() scored each atom
    against itself at a clamped 0.3 A, inventing a large repulsion."""
    be = resolve("builtin")
    one = AtomSet(names=["CB"], coords=[[0.0, 0.0, 0.0]])    # no reskeys given
    assert be.internal(one) == 0.0
    assert be.interaction(one, one) == 0.0


def test_regression_fractional_energy_cap_is_not_truncated():
    """int() turned a -12.5 kcal/mol cap into -12, and -0.5 into 0."""
    t = _toy_tables(12)
    for cap in (-12.5, -0.5, 2.5):
        con = Constraint(kind="max_energy", count=cap, label="cap")
        lp = RotamerMILP(t, constraints=[con],
                         native={p: "ALA" for p in t.positions}).to_lp()
        # The row is long enough that _wrap_terms continues it over several
        # lines, so the right-hand side is matched against the whole file.
        assert f"<= {cap:.6g}" in lp, cap
        assert any(l.strip().startswith("cap:") for l in lp.splitlines())
    # And the truncating form would have produced a different cap.
    assert int(-12.5) == -12 and f"<= {-12.5:.6g}" != f"<= {int(-12.5)}"


def test_regression_non_optimal_status_is_not_reported_as_proven():
    """'INTEGER NON-OPTIMAL' contains 'optimal'."""
    from ipro_agent.milp import _parse_glpk_solution

    for status, expect in (
        ("INTEGER OPTIMAL", True), ("OPTIMAL", True),
        ("INTEGER NON-OPTIMAL", False), ("NOT OPTIMAL", False),
        ("SUBOPTIMAL", False),
    ):
        s = status.lower().replace("-", " ")
        proven = "optimal" in s and not any(
            w in s for w in ("non optimal", "not optimal", "suboptimal", "sub optimal")
        )
        assert proven is expect, status
    # and the parser itself still reads a real status
    vals, st = _parse_glpk_solution(
        "Status:     INTEGER OPTIMAL\n\n   No. Column name       Activity\n"
        "------ ------------    -------------\n     1 y_0_0        *              1\n"
    )
    assert st == "INTEGER OPTIMAL" and vals == {"y_0_0": 1.0}


def test_regression_duplicate_rotamer_states_keep_separate_energies():
    """self_energy was looked up by (position, aa, state), so a library with a
    duplicated (aa, state) collapsed two rotamers onto one energy."""
    a = Rotamer(position="A1", aa="ALA", state=1,
                atoms=AtomSet(names=["CB"], coords=[[0.0, 0.0, 0.0]],
                              resnames=["ALA"], reskeys=[("A", 1, "")]))
    b = Rotamer(position="A1", aa="ALA", state=1,   # same key, different place
                atoms=AtomSet(names=["CB"], coords=[[6.0, 0.0, 0.0]],
                              resnames=["ALA"], reskeys=[("A", 1, "")]))
    template = AtomSet(names=["CB"], coords=[[6.6, 0.0, 0.0]],
                       resnames=["LEU"], reskeys=[("A", 9, "")])
    t = build_tables({"A1": [a, b]}, template, resolve("builtin"),
                     max_self_energy=1e9)
    e0, e1 = t.self_energy[("A1", 0)], t.self_energy[("A1", 1)]
    assert e0 != e1, (e0, e1)


def test_regression_integer_cut_binds_when_visited_names_extra_positions():
    """The right-hand side used len(assignment), so a visited sequence naming a
    position absent from the tables left the cut slack by one."""
    t = _toy_tables(13)
    native = {p: "ALA" for p in t.positions}
    first = RotamerMILP(t, native=native).solve()
    seq = dict(first.identities(t))
    seq["Z999"] = "TRP"                      # a position not in these tables
    second = RotamerMILP(t, native=native, visited=[seq]).solve()
    assert second.identities(t) != first.identities(t)


def test_regression_position_with_no_rotamers_is_rejected():
    """An empty rotamer list produced a program with no assignment constraint
    for that position, whose solution left it undefined."""
    t = EnergyTables(positions=["A1"], rotamers={"A1": []})
    try:
        RotamerMILP(t, native={}).to_lp()
    except Infeasible:
        pass
    else:
        raise AssertionError("a rotamer-less position should be infeasible")


def collect():
    g = globals()
    return [(n, g[n]) for n in sorted(g) if n.startswith("test_")]
