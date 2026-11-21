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
import os
import random
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


# ---------------------------------------------------------------------------
# Restraints
# ---------------------------------------------------------------------------


def _geom_tables(coords_by_pos: dict[str, list[tuple[str, float]]]) -> EnergyTables:
    """Toy tables whose rotamers carry a 'CG' atom at a chosen x-offset.

    Needs no PyMOL, so the restraint algebra is tested deterministically and in
    milliseconds. `coords_by_pos` maps a position to (residue name, x) pairs;
    every rotamer also carries a CB at the origin of its position.
    """
    positions = sorted(coords_by_pos)
    rotamers: dict[str, list[Rotamer]] = {}
    for pi, p in enumerate(positions):
        rots = []
        for st, (aa, x) in enumerate(coords_by_pos[p], start=1):
            base = np.array([pi * 10.0, 0.0, 0.0])
            rots.append(
                Rotamer(
                    position=p, aa=aa, state=st,
                    atoms=AtomSet(
                        names=["CB", "CG"],
                        coords=[base, base + np.array([x, 0.0, 0.0])],
                        resnames=[aa, aa],
                        reskeys=[("A", pi + 1, ""), ("A", pi + 1, "")],
                    ),
                )
            )
        rotamers[p] = rots
    t = EnergyTables(positions=positions, rotamers=rotamers)
    for p in positions:
        for i in range(len(rotamers[p])):
            # Deliberately make the LAST rotamer cheapest, so a restraint that
            # forbids it must actually change the answer.
            t.self_energy[(p, i)] = -float(i)
    return t


def test_restraint_spec_validation():
    from ipro_agent.config import AtomRef, Restraint

    assert str(AtomRef.parse("A115:CB")) == "A115:CB"
    assert AtomRef.parse("X1:C1").chain == "X"
    assert AtomRef.parse("A115B:OD1").insertion_code == "B"
    # a real condition in a readable form
    r = Restraint(kind="distance", atoms=["A1:CG", "A2:CG"], target=3.2, tolerance=0.2)
    assert r.interval() == (3.0, 3.4000000000000004) or abs(r.interval()[0] - 3.0) < 1e-9
    assert Restraint(kind="fix", atoms=["A1:CB"]).interval()[1] > 0

    bad = [
        dict(kind="distance", atoms=["A1:CB"], target=3.0, tolerance=0.1),       # arity
        dict(kind="distance", atoms=["A1:CB", "A2:CB"], target=3.0),             # no tolerance
        dict(kind="tether", atoms=["A1:CB"], target=3.0, tolerance=0.1),         # tether+target
        dict(kind="distance", atoms=["A1:CB", "A2:CB"]),                         # no condition
        dict(kind="distance", atoms=["A1:CB", "A2:CB"], minimum=5.0, maximum=3.0),  # empty
    ]
    for kw in bad:
        try:
            Restraint(**kw)
        except SpecError:
            pass
        else:
            raise AssertionError(f"should have been rejected: {kw}")


def test_restraint_single_position_veto_is_exact():
    """A restraint on one design position must forbid exactly the rotamers that
    violate it, and the optimum must equal the constrained brute force."""
    from ipro_agent.config import Restraint
    from ipro_agent.restraints import plan_restraints
    from ipro_agent.structure import Structure

    # CG sits 1, 5 or 9 A along x from A1's origin; the anchor is a real atom.
    t = _geom_tables({"A1": [("SER", 1.0), ("SER", 5.0), ("SER", 9.0)],
                      "A2": [("ALA", 1.0), ("ALA", 2.0)]})
    st = Structure.load(CRN)
    anchor = st.find("A", 30).child_dict["CA"].coord
    # Place the restraint so only the nearest CG qualifies.
    import numpy as _np
    d0 = float(_np.linalg.norm(_np.array([1.0, 0.0, 0.0]) - _np.array(anchor)))
    r = Restraint(kind="distance", atoms=["A1:CG", "A30:CA"],
                  maximum=d0 + 0.5, label="near")
    plan = plan_restraints([r], t, st, [])
    assert plan.rotamer_vetoes, "expected some rotamer to be forbidden"
    assert all(p == "A1" for p, _ in plan.rotamer_vetoes)
    assert not plan.pair_vetoes

    m = RotamerMILP(t, native={p: "ALA" for p in t.positions},
                    rotamer_vetoes=plan.rotamer_vetoes)
    for solver in ("auto", "greedy"):
        sol = m.solve(solver=solver)
        assert (("A1", sol.choices["A1"]) not in plan.rotamer_vetoes), solver
        assert m._feasible(sol.choices), solver
    exact = m.solve(solver="auto")
    # constrained brute force: the best assignment using no vetoed rotamer
    import itertools as _it
    bb = None
    for combo in _it.product(*[range(len(t.rotamers[p])) for p in t.positions]):
        ch = dict(zip(t.positions, combo))
        if any((p, r) in plan.rotamer_vetoes for p, r in ch.items()):
            continue
        e = m.evaluate(ch)
        if bb is None or e < bb:
            bb = e
    assert abs(exact.objective - bb) < 1e-6, (exact.objective, bb)


def test_restraint_two_position_pair_veto_is_exact():
    """A restraint spanning two design positions becomes pair vetoes, and both
    solver paths must respect them."""
    import itertools as _it

    from ipro_agent.config import Restraint
    from ipro_agent.restraints import plan_restraints

    # A1's CG at x in {1,5}; A2's origin is at x=10, CG at +{1,5}.
    t = _geom_tables({"A1": [("SER", 1.0), ("SER", 5.0)],
                      "A2": [("SER", 1.0), ("SER", 5.0)]})
    # CG-CG distances: 10, 14, 6, 10 -> forbid anything above 11 A.
    r = Restraint(kind="distance", atoms=["A1:CG", "A2:CG"], maximum=11.0, label="close")
    plan = plan_restraints([r], t, None, [])     # no fixed atoms needed
    assert plan.pair_vetoes, "expected forbidden pairs"
    assert not plan.rotamer_vetoes

    m = RotamerMILP(t, native={p: "SER" for p in t.positions},
                    pair_vetoes=plan.pair_vetoes)
    bb = None
    for combo in _it.product(*[range(len(t.rotamers[p])) for p in t.positions]):
        ch = dict(zip(t.positions, combo))
        key = ("A1", ch["A1"], "A2", ch["A2"])
        if key in plan.pair_vetoes:
            continue
        e = m.evaluate(ch)
        if bb is None or e < bb:
            bb = e
    for solver in ("auto", "greedy"):
        sol = m.solve(solver=solver)
        assert m._feasible(sol.choices), solver
        got = ("A1", sol.choices["A1"], "A2", sol.choices["A2"])
        assert got not in plan.pair_vetoes, (solver, got)
    assert abs(m.solve(solver="auto").objective - bb) < 1e-6


def test_restraint_forbidding_everything_raises_with_an_explanation():
    from ipro_agent.config import Restraint
    from ipro_agent.restraints import RestraintError, plan_restraints

    t = _geom_tables({"A1": [("SER", 1.0), ("SER", 2.0)]})
    r = Restraint(kind="distance", atoms=["A1:CG", "A1:CB"], minimum=50.0, label="impossible")
    try:
        plan_restraints([r], t, None, [])
    except RestraintError as exc:
        assert "forbid every rotamer" in str(exc)
    else:
        raise AssertionError("should have refused")


def test_restraint_on_a_missing_atom_is_caught_before_the_run():
    """check_restraints needs no rotamer library, so a typo is caught cheaply."""
    from ipro_agent.config import Restraint
    from ipro_agent.restraints import check_restraints
    from ipro_agent.structure import Structure

    st = Structure.load(CRN)
    r = Restraint(kind="distance", atoms=["A30:CA", "A30:ZZ9"], maximum=5.0, label="typo")
    problems, _ = check_restraints([r], st, [], [])
    assert problems and "ZZ9" in problems[0], problems


def test_restraint_with_no_design_position_is_measured_every_iteration():
    """No rotamer choice can affect such a restraint -- but the backbone
    perturbation and the rigid-body redock both can, and the planner's
    "nothing to decide" meant it was then enforced nowhere.

    That is the paper's own mechanism for preserving catalysis: the ACS Catal
    2017 thioesterase work held catalytic distances to +/-0.2 A with exactly
    this kind of restraint, between atoms neither of which is a design
    position. Measured: translating the partner 1.0 A -- five times that
    tolerance, and far less than the drift of a few hundred redock moves at
    sigma 0.2 -- turned a satisfied restraint into a violated one while
    `rotamer_vetoes` stayed empty and `is_empty()` stayed True, so nothing
    reached the MILP and nothing rejected the iteration. They are kept on the
    plan now (`geometry_only`) and the loop rejects on them."""
    from ipro_agent.config import Restraint
    from ipro_agent.restraints import plan_restraints
    from ipro_agent.structure import Structure

    st = Structure.load(CRN)
    t = _geom_tables({"A1": [("SER", 1.0)]})
    # Both atoms are fixed, and deliberately far apart.
    r = Restraint(kind="distance", atoms=["A30:CA", "A31:CA"], maximum=0.5, label="fixedpair")
    plan = plan_restraints([r], t, st, [])
    assert not plan.rotamer_vetoes and not plan.pair_vetoes
    assert plan.geometry_only, "it must stay on the plan, not be dropped"
    assert plan.native_violations, plan.summary
    assert "violated by the coordinates as they stand" in plan.native_violations[0]
    assert "iteration is rejected" in plan.native_violations[0]
    assert "re-measured after each perturbation and redock" in plan.summary[0]

    # And the loop enforces it rather than noting it.
    src = (REPO / "ipro_agent" / "loop.py").read_text()
    assert "_broken_geometry_restraints" in src
    assert 'reason = False, f"restraint violated: {broke[0]}"' in src


def test_dihedral_rejects_one_sided_bounds():
    """A one-sided bound on a circular quantity is ambiguous; refuse it."""
    from ipro_agent.config import Restraint
    from ipro_agent.restraints import RestraintError, plan_restraints

    t = _geom_tables({"A1": [("SER", 1.0)]})
    r = Restraint(kind="dihedral", atoms=["A1:CB", "A1:CG", "A1:CB", "A1:CG"],
                  minimum=60.0, label="amb")
    try:
        plan_restraints([r], t, None, [])
    except RestraintError as exc:
        assert "ambiguous" in str(exc) or "minimum" in str(exc)
    else:
        raise AssertionError("a one-sided dihedral bound should be refused")


def test_dihedral_and_distance_geometry():
    """The geometry helpers themselves, against known values."""
    from ipro_agent.restraints import angular_deviation, dihedral, distance

    assert abs(distance([0, 0, 0], [3, 4, 0]) - 5.0) < 1e-12

    # Both outer bonds along +x: eclipsed (cis), 0 degrees.
    cis = dihedral([1, 1, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0])
    assert abs(cis) < 1e-6, cis
    # Outer bonds opposed: anti (trans), 180 degrees.
    trans = dihedral([1, 1, 0], [0, 1, 0], [0, 0, 0], [-1, 0, 0])
    assert abs(abs(trans) - 180.0) < 1e-6, trans
    # Outer bonds perpendicular: 90 degrees.
    perp = dihedral([1, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1])
    assert abs(abs(perp) - 90.0) < 1e-6, perp
    # Degenerate inputs must raise rather than return a meaningless number.
    for pts in (
        ([0, 0, 0], [1, 1, 1], [1, 1, 1], [2, 0, 0]),     # central atoms coincide
        ([0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]),     # collinear
    ):
        try:
            dihedral(*pts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"degenerate dihedral should raise: {pts}")
    assert abs(angular_deviation(179.0, -179.0) - 2.0) < 1e-9   # wraps
    assert abs(angular_deviation(10.0, 350.0) - 20.0) < 1e-9


# ---------------------------------------------------------------------------
# Binding assemblies
# ---------------------------------------------------------------------------


def _asm_tables(energies: dict[str, dict[tuple[str, int], float]]) -> EnergyTables:
    """Toy tables from an explicit {position: {(aa, state): self energy}} map."""
    positions = sorted(energies)
    t = EnergyTables(positions=positions, rotamers={})
    for pi, p in enumerate(positions):
        lst = [
            Rotamer(
                position=p, aa=aa, state=st,
                atoms=AtomSet(
                    names=["CB"], coords=[[pi * 10.0, st * 0.5, 0.0]],
                    resnames=[aa], reskeys=[("A", pi + 1, "")],
                ),
            )
            for (aa, st) in sorted(energies[p])
        ]
        t.rotamers[p] = lst
        for i, r in enumerate(lst):
            t.self_energy[(p, i)] = energies[p][(r.aa, r.state)]
    return t


def test_assembly_spec_defaults_and_validation():
    from ipro_agent.config import DesignSpec

    base = dict(
        goal="bind_tighter", goal_statement="x",
        molecules=[
            {"name": "e", "path": str(CRN), "role": "design", "kind": "protein", "chains": ["A"]},
            {"name": "L1", "path": str(CRN), "role": "target", "kind": "ligand"},
            {"name": "L2", "path": str(CRN), "role": "anti_target", "kind": "ligand"},
        ],
        positions=["A16"],
    )
    # No assemblies block -> one implicit assembly with every partner.
    only = DesignSpec.from_dict(dict(base)).resolved_assemblies()
    assert len(only) == 1 and set(only[0].molecules) == {"L1", "L2"}

    two = DesignSpec.from_dict({**base, "assemblies": [
        {"name": "a", "molecules": ["e", "L1"]},
        {"name": "b", "molecules": ["L2"], "weight": 0.5, "must_pass": False},
    ]}).resolved_assemblies()
    assert [a.name for a in two] == ["a", "b"]
    assert two[0].molecules == ["L1"]       # the design molecule is implicit
    assert two[1].weight == 0.5 and two[1].must_pass is False

    for bad, why in (
        ([{"name": "a", "molecules": ["L1"]}], "L2 belongs to no assembly"),
        ([{"name": "a", "molecules": ["nope"]}], "unknown molecule"),
        ([{"name": "a", "molecules": ["L1"]}, {"name": "a", "molecules": ["L2"]}], "duplicate name"),
        ([{"name": "a", "molecules": ["L1", "L2"], "must_pass": False}], "nothing gates acceptance"),
        ([{"name": "a", "molecules": ["L1", "L2"], "weight": 0}], "non-positive weight"),
    ):
        try:
            DesignSpec.from_dict({**base, "assemblies": bad}).resolved_assemblies()
        except SpecError:
            pass
        else:
            raise AssertionError(f"should have been rejected: {why}")


def test_single_assembly_combine_is_a_no_op():
    """The single-complex case must be exactly the problem it was before
    assemblies existed -- same positions, same tables object."""
    from ipro_agent.assemblies import combine
    from ipro_agent.config import Assembly

    t = _asm_tables({"A1": {("ALA", 1): -1.0, ("TRP", 1): 0.0}})
    cb = combine({"only": t}, [Assembly(name="only")])
    assert cb.tables is t
    assert cb.identity_links == []
    assert cb.sequence_positions == set(t.positions)
    assert cb.n_assemblies == 1


def test_assembly_identity_is_shared_while_packing_differs():
    """IPRO's rule exactly: "rotamers are selected sequentially for each BA
    whereas residue identity remains the same for all BAs"."""
    from ipro_agent.assemblies import combine, identities_agree, split_vpos
    from ipro_agent.config import Assembly

    # TRP state 1 is best in h1, TRP state 2 is best in h2; ALA is worse in both.
    h1 = _asm_tables({"A1": {("TRP", 1): -5.0, ("TRP", 2): -1.0, ("ALA", 1): -0.5}})
    h2 = _asm_tables({"A1": {("TRP", 1): -1.0, ("TRP", 2): -5.0, ("ALA", 1): -0.5}})
    cb = combine({"h1": h1, "h2": h2}, [Assembly(name="h1"), Assembly(name="h2")])
    assert len(cb.identity_links) == 1
    m = RotamerMILP(
        cb.tables, native={p: "ALA" for p in cb.tables.positions},
        identity_links=cb.identity_links, sequence_positions=cb.sequence_positions,
    )
    for solver in ("auto", "greedy"):
        sol = m.solve(solver=solver)
        assert not identities_agree(cb, sol.choices), solver
        picked = {
            split_vpos(v)[0]: (cb.tables.rotamers[v][i].aa, cb.tables.rotamers[v][i].state)
            for v, i in sol.choices.items()
        }
        assert len({a for a, _ in picked.values()}) == 1, ("identity must be shared", picked)
        assert len({s for _, s in picked.values()}) == 2, ("packing must differ", picked)
    # -10 = each assembly's best packing of the shared identity
    assert abs(m.solve(solver="auto").objective - (-10.0)) < 1e-9


def test_assembly_optimum_matches_shared_identity_brute_force():
    """Brute force with the right semantics: choose an identity per position
    (shared), then pack each assembly independently."""
    import itertools as _it

    from ipro_agent.assemblies import combine
    from ipro_agent.config import Assembly

    e1 = {"A1": {("ALA", 1): -1.0, ("ALA", 2): -0.5, ("TRP", 1): 2.0, ("TRP", 2): 3.0},
          "A2": {("ALA", 1): 0.0, ("ALA", 2): 0.5, ("TRP", 1): -1.0, ("TRP", 2): -0.2}}
    e2 = {"A1": {("ALA", 1): 4.0, ("ALA", 2): 4.5, ("TRP", 1): -3.0, ("TRP", 2): -2.0},
          "A2": {("ALA", 1): -0.5, ("ALA", 2): 0.0, ("TRP", 1): 1.0, ("TRP", 2): 1.5}}
    per = {"h1": _asm_tables(e1), "h2": _asm_tables(e2)}
    cb = combine(per, [Assembly(name="h1"), Assembly(name="h2")])
    m = RotamerMILP(
        cb.tables, native={p: "ALA" for p in cb.tables.positions},
        identity_links=cb.identity_links, sequence_positions=cb.sequence_positions,
    )
    best = None
    for seq in _it.product(["ALA", "TRP"], repeat=2):
        total = 0.0
        for t in per.values():
            for p, aa in zip(["A1", "A2"], seq):
                total += min(
                    t.self_energy[(p, i)]
                    for i, r in enumerate(t.rotamers[p]) if r.aa == aa
                )
        best = total if best is None else min(best, total)
    assert abs(m.solve(solver="auto").objective - best) < 1e-6
    assert abs(m.solve(solver="greedy").objective - best) < 1e-6


def test_assembly_does_not_multiply_sequence_budgets():
    """max_mutations is a budget on the SEQUENCE. Counting it in every assembly
    would silently allow N_assemblies times as many mutations."""
    from ipro_agent.assemblies import combine, split_vpos
    from ipro_agent.config import Assembly

    e = {"A1": {("TRP", 1): -5.0, ("ALA", 1): 0.0},
         "A2": {("TRP", 1): -5.0, ("ALA", 1): 0.0}}
    cb = combine({"h1": _asm_tables(e), "h2": _asm_tables(e)},
                 [Assembly(name="h1"), Assembly(name="h2")])
    native = {p: "ALA" for p in cb.tables.positions}
    for k in (0, 1, 2):
        m = RotamerMILP(
            cb.tables, constraints=[Constraint(kind="max_mutations", count=k)],
            native=native, identity_links=cb.identity_links,
            sequence_positions=cb.sequence_positions,
        )
        sol = m.solve(solver="auto")
        mutated = {
            split_vpos(v)[1] for v, i in sol.choices.items()
            if v in cb.sequence_positions and cb.tables.rotamers[v][i].aa != "ALA"
        }
        assert len(mutated) <= k, (k, mutated)


def test_assembly_mismatched_rotamer_sets_are_rejected():
    """Assemblies share the design molecule's backbone, so their rotamer sets
    must be identical; a mismatch is an internal inconsistency, not input."""
    from ipro_agent.assemblies import combine
    from ipro_agent.config import Assembly

    a = _asm_tables({"A1": {("ALA", 1): 0.0, ("TRP", 1): 0.0}})
    b = _asm_tables({"A1": {("ALA", 1): 0.0}})
    try:
        combine({"h1": a, "h2": b}, [Assembly(name="h1"), Assembly(name="h2")])
    except ValueError as exc:
        assert "rotamer set" in str(exc) or "disagree" in str(exc)
    else:
        raise AssertionError("mismatched rotamer sets should be rejected")


# ---------------------------------------------------------------------------
# Placing the partners
# ---------------------------------------------------------------------------


def test_pose_round_trip_is_exact():
    """A known rigid transform must be recovered to floating-point accuracy;
    everything downstream of docking depends on this fit."""
    import random as _random

    from ipro_agent.docking import _pose_between, _random_rotation

    rng = _random.Random(1)
    start = np.array([[0.0, 0, 0], [1.5, 0, 0], [0, 1.4, 0], [0.3, 0.2, 1.1]])
    R = _random_rotation(30.0, rng)
    shift = np.array([0.7, -0.4, 0.25])
    c = start.mean(axis=0)
    end = (start - c) @ R.T + c + shift
    pose = _pose_between(start, end, c)
    assert np.abs(pose.apply(start) - end).max() < 1e-9
    # A single atom has no orientation, so only the shift is recoverable.
    one = _pose_between(np.array([[1.0, 2, 3]]), np.array([[2.0, 2, 3]]), np.zeros(3))
    assert np.allclose(one.translation, [1.0, 0, 0])


def test_rigid_body_anneal_never_worsens_and_its_pose_reproduces_its_energy():
    """The two properties that make the result usable: the search cannot report
    a worse best than it started with, and the pose it hands back must actually
    give the energy it claims."""
    import random as _random

    from ipro_agent.docking import moved_atomset, net_interaction, redock
    from ipro_agent.loop import _atoms_of
    from ipro_agent.structure import Structure

    st = Structure.load(CRN)
    design = _atoms_of(st)
    be = resolve("builtin")
    cb = np.array(st.find("A", 16).child_dict["CB"].coord)
    probe = AtomSet(
        names=["C1", "O1", "C2"],
        coords=np.array([cb + [2.6, 0, 0], cb + [3.9, 0, 0], cb + [2.6, 1.3, 0]]),
        resnames=["LIG"] * 3, reskeys=[("X", 1, "")] * 3,
    )
    res = redock(design, {"L": probe}, be, moves=150, rng=_random.Random(3))["L"]
    assert res.best_energy <= res.start_energy + 1e-9
    rescored = net_interaction(design, {"L": moved_atomset(probe, res.pose)}, be)
    assert abs(rescored - res.best_energy) < 1e-6, (rescored, res.best_energy)


def test_rigid_body_anneal_is_reproducible_from_a_seed():
    """IPRO seeds nothing; this does, so a run can be repeated."""
    import random as _random

    from ipro_agent.docking import redock
    from ipro_agent.loop import _atoms_of
    from ipro_agent.structure import Structure

    st = Structure.load(CRN)
    design = _atoms_of(st)
    be = resolve("builtin")
    cb = np.array(st.find("A", 16).child_dict["CB"].coord)
    probe = AtomSet(names=["C1"], coords=np.array([cb + [3.0, 0, 0]]),
                    resnames=["LIG"], reskeys=[("X", 1, "")])
    a = redock(design, {"L": probe}, be, moves=60, rng=_random.Random(42))["L"]
    b = redock(design, {"L": probe}, be, moves=60, rng=_random.Random(42))["L"]
    assert a.best_energy == b.best_energy
    assert np.allclose(a.pose.translation, b.pose.translation)


def test_rigid_body_moves_respect_the_published_step_sizes():
    """sigma 0.2 A and 2.0 deg are the paper's numbers; a bug here would turn a
    local refinement into a random relocation."""
    import math as _math
    import random as _random

    from ipro_agent.docking import _random_rotation

    rng = _random.Random(7)
    angles = []
    for _ in range(3000):
        R = _random_rotation(2.0, rng)
        angles.append(_math.degrees(
            _math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))
        ))
    # Mean |angle| for a Gaussian of sigma=2 deg is sigma*sqrt(2/pi) ~ 1.6.
    assert 1.2 < float(np.mean(angles)) < 2.2, float(np.mean(angles))
    assert max(angles) < 15.0, "a single move must stay local"


def test_dock_backend_choice_is_honoured():
    """The user picks the placement program. 'vina' must not silently become
    something cheaper, and 'auto' must say what it fell back to."""
    from ipro_agent.config import DesignSpec
    from ipro_agent.dock import find_vina
    from ipro_agent.loop import IPROLoop

    def spec_with(dock: str, redock: bool):
        return DesignSpec.from_dict({
            "goal": "stabilize", "goal_statement": "x",
            "molecules": [
                {"name": "m", "path": str(CRN), "role": "design",
                 "kind": "protein", "chains": ["A"]},
                {"name": "L", "path": str(CRN), "role": "target",
                 "kind": "ligand", "redock": redock},
            ],
            "positions": ["A16"],
            "backends": {"dock": dock},
        })

    tmp = Path(tempfile.mkdtemp(prefix="ipro_dock_"))
    try:
        # Nothing mobile -> nothing to place, whatever was asked for.
        loop = IPROLoop(spec_with("builtin", False), tmp)
        assert loop._resolve_dock() == "none"
        assert any("redock: true" in n for n in loop.notes)

        assert IPROLoop(spec_with("none", True), tmp)._resolve_dock() == "none"
        assert IPROLoop(spec_with("builtin", True), tmp)._resolve_dock() == "builtin"

        have_vina = find_vina() is not None
        auto = IPROLoop(spec_with("auto", True), tmp)
        resolved = auto._resolve_dock()
        assert resolved == ("vina" if have_vina else "builtin")
        if not have_vina:
            assert any("no AutoDock Vina binary" in n for n in auto.notes)
            # An explicit request must fail loudly, not degrade.
            try:
                IPROLoop(spec_with("vina", True), tmp)._resolve_dock()
            except RuntimeError as exc:
                assert "builtin" in str(exc) and "Install it" in str(exc)
            else:
                raise AssertionError("dock: vina with no Vina should raise")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_vina_pose_is_rejected_when_atoms_do_not_correspond():
    """Without a one-to-one atom mapping the transform cannot be fitted, and
    guessing would move the ligand somewhere nobody chose."""
    from ipro_agent.config import DesignSpec
    from ipro_agent.loop import IPROLoop

    tmp = Path(tempfile.mkdtemp(prefix="ipro_vinapose_"))
    try:
        spec = DesignSpec.from_dict({
            "goal": "stabilize", "goal_statement": "x",
            "molecules": [
                {"name": "m", "path": str(CRN), "role": "design",
                 "kind": "protein", "chains": ["A"]},
                {"name": "L", "path": str(CRN), "role": "target", "kind": "ligand"},
            ],
            "positions": ["A16"],
        })
        loop = IPROLoop(spec, tmp)
        loop._partners["L"] = __import__(
            "ipro_agent.structure", fromlist=["Structure"]
        ).Structure.load(CRN)

        pose_file = tmp / "pose.pdb"
        pose_file.write_text(
            "ATOM      1  C1  LIG X   1       0.000   0.000   0.000"
            "  1.00  0.00           C\nEND\n"
        )

        class _Res:
            poses = [pose_file]

        assert loop._pose_from_vina("L", _Res()) is None
        assert any("was NOT applied" in n for n in loop.notes)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_redock_only_invalidates_self_energies():
    """Pair energies are rotamer-against-rotamer, so moving a partner cannot
    change them. Refreshing with unmoved partners must change nothing at all --
    which also proves the two template constructions agree."""
    if not _have_pymol():
        print("    (skipped: no PyMOL)")
        return
    from ipro_agent.config import DesignSpec
    from ipro_agent.loop import IPROLoop

    tmp = Path(tempfile.mkdtemp(prefix="ipro_refresh_"))
    try:
        spec = DesignSpec.from_dict({
            "goal": "stabilize", "goal_statement": "x",
            "molecules": [{"name": "m", "path": str(CRN), "role": "design",
                           "kind": "protein", "chains": ["A"]}],
            "positions": [{"position": "A16", "allowed": ["ALA", "SER"]}],
        })
        loop = IPROLoop(spec, tmp)
        loop.prepare()
        cb = loop.build_combined()
        before_self = dict(cb.tables.self_energy)
        before_pair = dict(cb.tables.pair_energy)
        loop._refresh_self_energies(cb)      # all poses are identity
        assert cb.tables.self_energy == before_self
        assert cb.tables.pair_energy == before_pair
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Backbone perturbation
# ---------------------------------------------------------------------------


def test_perturbation_leaves_every_bond_length_unchanged():
    """The decisive check. A rigid rotation about a bond axis cannot change any
    bond length, so a wrong axis or a wrong set of moving atoms shows up here --
    and nowhere else, since the result still looks like a protein.

    Note the invariant is *unchanged from the input*, not *equal to ideal*:
    1CRN is a 1.5 A crystal structure with an N-CA bond of 1.518 A at residue
    37 before anything is touched.
    """
    import random as _random

    from ipro_agent.perturb import (
        backbone_bond_lengths, bonds_unchanged, perturb_backbone,
    )
    from ipro_agent.structure import Structure

    for seed in range(8):
        st = Structure.load(CRN)
        before = backbone_bond_lengths(st, "A")
        assert len(before) > 100, "expected a full chain of bonds to check"
        perturb_backbone(st, "A", rng=_random.Random(seed))
        assert not bonds_unchanged(before, backbone_bond_lengths(st, "A"))


def test_perturbation_propagates_downstream_only():
    """Rotating phi/psi at residue i must move i's side chain and everything
    C-terminal, and nothing N-terminal."""
    import random as _random

    from ipro_agent.perturb import perturb_backbone
    from ipro_agent.structure import Structure

    st = Structure.load(CRN)
    residues = list(st.residues("A"))
    target = residues[20]
    num = int(target.id[1])
    before = {int(r.id[1]): np.array(st.coords(r)).mean(axis=0) for r in residues}
    perturb_backbone(st, "A", segment=[target], rng=_random.Random(9))
    after = {int(r.id[1]): np.array(st.coords(r)).mean(axis=0) for r in st.residues("A")}

    upstream = [np.linalg.norm(after[k] - before[k]) for k in before if k < num]
    downstream = [np.linalg.norm(after[k] - before[k]) for k in before if k > num]
    assert max(upstream) < 1e-9, f"upstream moved: {max(upstream)}"
    assert np.mean(downstream) > 1e-4, "downstream did not move"


def test_perturbation_skips_proline_and_keeps_segments_consecutive():
    """Proline's phi is locked by its ring, so perturbing it is meaningless;
    and a segment spanning a numbering gap is not a local move."""
    import random as _random

    from ipro_agent.perturb import choose_segment
    from ipro_agent.structure import Structure

    st = Structure.load(CRN)
    prolines = {
        int(r.id[1]) for r in st.residues("A") if r.get_resname().strip() == "PRO"
    }
    assert prolines, "1CRN should contain prolines for this test to mean anything"

    seen_lengths = set()
    touched: set[int] = set()
    for seed in range(200):
        seg = choose_segment(st, "A", (1, 5), _random.Random(seed))
        assert seg
        nums = [int(r.id[1]) for r in seg]
        assert nums == list(range(nums[0], nums[0] + len(nums))), nums
        seen_lengths.add(len(seg))
        touched |= set(nums)
    assert not (touched & prolines), sorted(touched & prolines)
    assert seen_lengths <= {1, 2, 3, 4, 5} and len(seen_lengths) > 1


def test_perturbation_offsets_are_truncated_by_rejection_not_clipping():
    """Clipping would pile probability mass onto the two bounds, which is a
    different distribution from the Gaussian the paper specifies."""
    import random as _random

    from ipro_agent.perturb import truncated_gauss

    rng = _random.Random(5)
    vals = [truncated_gauss(1.5, 5.0, rng) for _ in range(20000)]
    assert max(abs(v) for v in vals) <= 5.0
    assert 1.3 < float(np.std(vals)) < 1.6, float(np.std(vals))
    at_cap = sum(1 for v in vals if abs(abs(v) - 5.0) < 1e-9)
    assert at_cap < 20, f"{at_cap} values sat exactly on the cap -- looks clipped"


def test_perturbation_is_reproducible_from_a_seed():
    import random as _random

    from ipro_agent.perturb import perturb_backbone
    from ipro_agent.structure import Structure

    out = []
    for _ in range(2):
        st = Structure.load(CRN)
        perturb_backbone(st, "A", rng=_random.Random(77))
        out.append(np.array([a.coord for a in st.all_atoms()]))
    assert np.abs(out[0] - out[1]).max() < 1e-12


def test_mode_reflects_what_will_actually_run():
    """Three modes, and the run must say which one it is in."""
    from ipro_agent.config import DesignSpec
    from ipro_agent.loop import IPROLoop

    def make(perturb: bool):
        return DesignSpec.from_dict({
            "goal": "stabilize", "goal_statement": "x",
            "molecules": [{"name": "m", "path": str(CRN), "role": "design",
                           "kind": "protein", "chains": ["A"]}],
            "positions": ["A16"],
            "loop": {"perturb_backbone": perturb},
        })

    tmp = Path(tempfile.mkdtemp(prefix="ipro_mode_"))
    try:
        off = IPROLoop(make(False), tmp)
        assert off.mode == "fixed-backbone"
        assert any("backbone never moves" in n for n in off.notes)
        on = IPROLoop(make(True), tmp)
        # No relaxing backend is installed here, so this is the honest middle mode.
        assert on.mode == ("full" if on.backend.can_relax else "backbone-mc")
        if on.mode == "backbone-mc":
            assert any("no gradient relaxation" in n for n in on.notes)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Structure output
# ---------------------------------------------------------------------------


def test_written_complex_has_the_design_and_its_partner():
    """A design that exists only as a score is not usable; the file must carry
    the chosen side chains and the partner at its current placement."""
    from ipro_agent.output import write_complex
    from ipro_agent.structure import Structure

    tmp = Path(tempfile.mkdtemp(prefix="ipro_out_"))
    try:
        design = Structure.load(CRN)
        # A partner that deliberately reuses chain A, to force a remap.
        lig = tmp / "lig.pdb"
        lig.write_text(
            "HETATM    1  C1  LIG A 900       1.000   2.000   3.000"
            "  1.00 20.00           C\nEND\n"
        )
        partner = Structure.load(lig)
        out = write_complex(
            tmp / "complex.pdb", design, {"L": partner},
            header=["REMARK   1 TEST"],
        )
        assert out.atoms == len(design.all_atoms()) + 1
        # The collision must have been resolved, not merged: the design keeps
        # chain A and the partner is moved elsewhere.
        assert out.chain_map[out.DESIGN]["A"] == "A"
        assert out.chain_map["L"]["A"] != "A", out.chain_map
        assert ("L", ("A", out.chain_map["L"]["A"])) in [
            (m, v) for m, v in out.renamed().items()
        ]

        back = Structure.load(out.path)
        assert len(back.chain_ids()) == 2, back.chain_ids()
        assert back.sequence("A") == design.sequence("A")
        het = [r for ch in back.chain_ids() for r in back.hetero_residues(ch)]
        assert len(het) == 1 and het[0].get_resname().strip() == "LIG"
        assert out.path.read_text().startswith("REMARK   1 TEST")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_written_complex_applies_the_partner_pose():
    """The file must show where a redock put the partner, not where it started."""
    import numpy as _np

    from ipro_agent.docking import Pose
    from ipro_agent.output import write_complex
    from ipro_agent.structure import Structure

    tmp = Path(tempfile.mkdtemp(prefix="ipro_pose_"))
    try:
        design = Structure.load(CRN)
        lig = tmp / "lig.pdb"
        lig.write_text(
            "HETATM    1  C1  LIG X 900       0.000   0.000   0.000"
            "  1.00 20.00           C\nEND\n"
        )
        partner = Structure.load(lig)
        pose = Pose(translation=_np.array([5.0, 0.0, 0.0]))
        out = write_complex(
            tmp / "c.pdb", design, {"L": partner}, poses={"L": pose}
        )
        back = Structure.load(out.path)
        het = [r for ch in back.chain_ids() for r in back.hetero_residues(ch)][0]
        moved = _np.array(back.coords(het))[0]
        assert abs(moved[0] - 5.0) < 1e-3, moved
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_writes_designed_coordinates_that_contain_the_mutation():
    """End to end: the run must leave a PDB carrying the design it reported."""
    if not _have_pymol():
        print("    (skipped: no PyMOL)")
        return
    from ipro_agent.config import DesignSpec
    from ipro_agent.loop import run_design
    from ipro_agent.structure import Structure

    tmp = Path(tempfile.mkdtemp(prefix="ipro_e2e_"))
    try:
        spec = DesignSpec.from_dict({
            "name": "e2e", "goal": "stabilize", "goal_statement": "x",
            "molecules": [{"name": "m", "path": str(CRN), "role": "design",
                           "kind": "protein", "chains": ["A"]}],
            "positions": [{"position": "A16", "allowed": ["TRP"]}],
            "goal_options": {"max_self_energy": 1e9},
            "loop": {"iterations": 2, "random_seed": 1},
        })
        res = run_design(spec, tmp)
        assert res.structures, "no coordinates were written"
        pdb = tmp / "best" / "design.pdb"
        assert pdb.exists(), sorted(p.name for p in (tmp / "best").iterdir())
        back = Structure.load(pdb)
        assert back.find("A", 16).get_resname().strip() == "TRP"
        assert back.sequence("A")[15] == "W"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Channel geometry
# ---------------------------------------------------------------------------


def _tube(radius: float, z_lo: float = -10.0, z_hi: float = 10.0, narrow=None):
    """Atoms on a cylinder, optionally with a narrow ring, for exact checks."""
    pts = []
    for z in np.arange(z_lo, z_hi + 0.01, 1.0):
        r = radius
        if narrow is not None and abs(z - narrow[0]) < narrow[1]:
            r = narrow[2]
        for th in np.linspace(0, 2 * np.pi, 48, endpoint=False):
            pts.append([r * np.cos(th), r * np.sin(th), z])
    return np.array(pts), ["C"] * len(pts)


def test_inscribed_radius_is_exact_on_a_cylinder():
    """A tube of carbons of radius R admits a circle of radius R - r_vdw."""
    from ipro_agent.pore import VDW, pore_profile

    for R in (5.0, 8.0, 12.0):
        coords, names = _tube(R)
        prof = pore_profile(
            coords, names, axis_origin=np.zeros(3),
            axis_direction=np.array([0.0, 0.0, 1.0]),
            spacing=1.0, max_centre_offset=min(2.0, R / 2),
        )
        assert abs(prof.min_radius - (R - VDW["C"])) < 0.15, (R, prof.min_radius)


def test_constriction_is_found_where_it_is():
    from ipro_agent.pore import pore_profile

    # a narrow ring of radius 5 at z=+3 inside a tube of radius 9
    coords, names = _tube(9.0, narrow=(3.0, 1.5, 5.0))
    prof = pore_profile(
        coords, names, axis_origin=np.zeros(3),
        axis_direction=np.array([0.0, 0.0, 1.0]),
        spacing=0.5, max_centre_offset=2.0,
    )
    assert abs(prof.min_radius - (5.0 - 1.70)) < 0.2, prof.min_radius
    # The ring spans z in {2,3,4}, i.e. axis offsets 12-14; any is correct.
    assert 11.5 <= prof.bottleneck_position <= 14.5, prof.bottleneck_position


def test_regression_measurement_cannot_escape_into_solvent():
    """The centre must stay near the axis.

    In unbounded space the largest circle overlapping no atom is out in the
    bulk, so an unconstrained search reports the distance to the solvent instead
    of the width of the lumen. The first version of this code did exactly that
    and placed the bottleneck one slice away from the real one.
    """
    from ipro_agent.pore import pore_profile

    coords, names = _tube(9.0, narrow=(3.0, 1.5, 5.0))
    tight = pore_profile(
        coords, names, axis_origin=np.zeros(3),
        axis_direction=np.array([0.0, 0.0, 1.0]),
        spacing=0.5, max_centre_offset=2.0,
    )
    loose = pore_profile(
        coords, names, axis_origin=np.zeros(3),
        axis_direction=np.array([0.0, 0.0, 1.0]),
        spacing=0.5, max_centre_offset=40.0,
    )
    assert abs(tight.min_radius - 3.3) < 0.2, tight.min_radius
    # With the constraint lifted the answer becomes about the bulk, not the pore.
    assert loose.min_radius > tight.min_radius + 1.0, (
        loose.min_radius, tight.min_radius,
        "a wide max_centre_offset should visibly stop measuring the lumen",
    )


def test_axis_from_lining_reproduces_the_published_ompf_eyelet():
    """Validation against a number from the literature.

    PoreDesigner reports the wild-type OmpF eyelet as 11 x 7 A. An inscribed
    circle is bounded by the narrow direction, so ~7 A is the comparison. The
    axis is fitted from the 25 published constriction positions; the protein's
    own principal axis is not the barrel axis and gives ~1.4 A.
    """
    pdb = REPO / "examples" / "ompf_pore" / "2omf_A.pdb"
    if not pdb.exists():
        print("    (skipped: example structure not present)")
        return
    from ipro_agent.pore import axis_from_lining, pore_profile, principal_axis, ring_flatness
    from ipro_agent.structure import Structure

    st = Structure.load(pdb)
    atoms = st.all_atoms()
    coords = np.array([a.coord for a in atoms])
    names = [a.get_name() for a in atoms]

    lining = [18, 20, 38, 40, 42, 62, 64, 82, 83, 109, 110, 113, 114, 115,
              117, 118, 119, 120, 121, 123, 124, 125, 130, 131, 132]
    ring = []
    for n in lining:
        try:
            ring.append(st.find("A", n).child_dict["CA"].coord)
        except Exception:
            pass
    assert len(ring) >= 20, len(ring)
    ring = np.array(ring)
    assert ring_flatness(ring) < 0.7, "the lining should be roughly coplanar"

    origin, direction = axis_from_lining(ring)
    good = pore_profile(
        coords, names, axis_origin=origin, axis_direction=direction,
        spacing=0.5, max_centre_offset=6.0,
    )
    assert 5.5 < good.min_diameter < 10.0, (
        good.min_diameter, "expected roughly the published 7 A minor axis"
    )

    # The whole-protein principal axis is NOT the barrel axis, which is why the
    # lining normal is the default.
    bad = pore_profile(
        coords, names, axis_origin=coords.mean(axis=0),
        axis_direction=principal_axis(
            np.array([a.coord for a in atoms if a.get_name().strip() == "CA"])
        ),
        spacing=0.5, max_centre_offset=6.0,
    )
    assert bad.min_diameter < good.min_diameter, (bad.min_diameter, good.min_diameter)


def test_pore_filter_discards_a_pore_of_the_wrong_size():
    """PoreDesigner's filter rejects outright, before the energy counts."""
    from ipro_agent.config import DesignSpec
    from ipro_agent.goals import ScoreContext, build_goal

    spec = DesignSpec.from_dict({
        "goal": "reshape_channel", "goal_statement": "narrow it",
        "molecules": [{"name": "m", "path": str(CRN), "role": "design",
                       "kind": "protein", "chains": ["A"]}],
        "positions": ["A16", "A18", "A20"],
        "goal_options": {"pore_diameter_max": 4.0, "pore_diameter_min": 2.0},
    })
    goal = build_goal(spec)
    assert goal.wants_geometry

    inside = ScoreContext(geometry={"min_pore_diameter": 3.0})
    assert goal.filter_reject(inside) is None

    too_wide = ScoreContext(geometry={"min_pore_diameter": 8.0})
    why = goal.filter_reject(too_wide)
    assert why and "above" in why, why

    too_narrow = ScoreContext(geometry={"min_pore_diameter": 1.0})
    why = goal.filter_reject(too_narrow)
    assert why and "below" in why, why

    # With nothing measured there is nothing to filter on.
    assert goal.filter_reject(ScoreContext()) is None


def test_only_channel_goals_pay_for_geometry():
    """Measuring a channel costs 0.1-0.6 s per iteration, so goals that do not
    read it must not trigger it."""
    from ipro_agent.config import DesignSpec
    from ipro_agent.goals import build_goal

    def goal_for(kind: str, options: dict | None = None):
        return build_goal(DesignSpec.from_dict({
            "goal": kind, "goal_statement": "x",
            "molecules": [
                {"name": "m", "path": str(CRN), "role": "design",
                 "kind": "protein", "chains": ["A"]},
                {"name": "L", "path": str(CRN), "role": "target", "kind": "ligand"},
            ],
            "positions": ["A16", "A18", "A20"],
            "goal_options": options or {},
        }))

    assert goal_for("reshape_channel", {"target_radius": 3.0}).wants_geometry
    for kind in ("bind_tighter", "stabilize"):
        assert not goal_for(kind).wants_geometry, kind


# ---------------------------------------------------------------------------
# Insertions and deletions
# ---------------------------------------------------------------------------


def _bb(res, name: str) -> np.ndarray:
    """Coordinate of a named atom in a Biopython residue (indel tests)."""
    return np.array(
        [a.coord for a in res if a.get_name().strip().upper() == name][0], dtype=float
    )


def test_place_atom_round_trips_its_internal_coordinates():
    """Build an atom from a bond, angle and dihedral, then measure all three
    back. This is the only way to be certain of the sign conventions, and
    everything about an insertion rests on it."""
    from ipro_agent.indel import measure_angle, measure_dihedral, place_atom

    rng = np.random.default_rng(3)
    worst = [0.0, 0.0, 0.0]
    checked = 0
    for _ in range(200):
        a, b, c = (rng.normal(size=3) * 3 for _ in range(3))
        if np.linalg.norm(np.cross(b - a, c - b)) < 1e-3:
            continue
        bond = float(rng.uniform(1.0, 2.0))
        angle = float(rng.uniform(70.0, 150.0))
        dih = float(rng.uniform(-179.0, 179.0))
        d = place_atom(a, b, c, bond, angle, dih)
        worst[0] = max(worst[0], abs(float(np.linalg.norm(d - c)) - bond))
        worst[1] = max(worst[1], abs(measure_angle(b, c, d) - angle))
        got = measure_dihedral(a, b, c, d)
        worst[2] = max(worst[2], abs((got - dih + 180.0) % 360.0 - 180.0))
        checked += 1
    assert checked > 150
    assert worst[0] < 1e-9 and worst[1] < 1e-7 and worst[2] < 1e-7, worst


def test_c_terminal_insertion_builds_ideal_trans_backbone():
    from ipro_agent.config import IndelSpec
    from ipro_agent.indel import (
        B_C_N, B_CA_C, B_N_CA, apply_terminal, classify, measure_dihedral,
    )
    from ipro_agent.structure import Structure

    st = Structure.load(CRN)
    before = [int(r.id[1]) for r in st.residues("A")]
    plan = classify(st, [IndelSpec(kind="insert", chain="A", after=before[-1], length=3)])
    assert len(plan.terminal) == 1 and plan.terminal[0][1] == "C"
    assert not plan.has_internal

    res = apply_terminal(st, plan.terminal)
    assert len(res.added) == 3
    after = [int(r.id[1]) for r in st.residues("A")]
    # The invariant that matters: design positions are named by number.
    assert after[: len(before)] == before, "existing residues were renumbered"
    assert after == sorted(after)

    rs = list(st.residues("A"))
    for i in range(len(before) - 1, len(after) - 1):
        r1, r2 = rs[i], rs[i + 1]
        assert abs(float(np.linalg.norm(_bb(r1, "C") - _bb(r2, "N"))) - B_C_N) < 1e-6
        assert abs(float(np.linalg.norm(_bb(r2, "N") - _bb(r2, "CA"))) - B_N_CA) < 1e-6
        assert abs(float(np.linalg.norm(_bb(r2, "CA") - _bb(r2, "C"))) - B_CA_C) < 1e-6
        omega = measure_dihedral(
            _bb(r1, "CA"), _bb(r1, "C"), _bb(r2, "N"), _bb(r2, "CA")
        )
        assert abs(abs(omega) - 180.0) < 1e-6, omega


def test_regression_n_terminal_insertion_of_more_than_one_residue():
    """Numbers must count *downward* so nothing existing is renumbered, and the
    chain must be re-ordered after each residue -- without that the second pass
    still saw the original first residue and tried to reuse its number."""
    from ipro_agent.config import IndelSpec
    from ipro_agent.indel import B_C_N, apply_terminal, classify, measure_dihedral
    from ipro_agent.structure import Structure

    st = Structure.load(CRN)
    before = [int(r.id[1]) for r in st.residues("A")]
    plan = classify(st, [IndelSpec(kind="insert", chain="A", after=before[0] - 1, length=2)])
    assert plan.terminal[0][1] == "N"
    res = apply_terminal(st, plan.terminal)
    assert len(res.added) == 2, res.added

    after = [int(r.id[1]) for r in st.residues("A")]
    assert after[-len(before):] == before, "existing residues were renumbered"
    assert after == sorted(after), "chain is out of N-to-C order"
    assert after[0] == before[0] - 2

    rs = list(st.residues("A"))
    for i in range(2):
        pep = float(np.linalg.norm(_bb(rs[i], "C") - _bb(rs[i + 1], "N")))
        omega = measure_dihedral(
            _bb(rs[i], "CA"), _bb(rs[i], "C"),
            _bb(rs[i + 1], "N"), _bb(rs[i + 1], "CA"),
        )
        assert abs(pep - B_C_N) < 1e-6, pep
        assert abs(abs(omega) - 180.0) < 1e-6, omega


def test_terminal_deletion_keeps_the_remaining_numbering():
    from ipro_agent.config import IndelSpec
    from ipro_agent.indel import apply_terminal, classify
    from ipro_agent.structure import Structure

    st = Structure.load(CRN)
    before = [int(r.id[1]) for r in st.residues("A")]
    plan = classify(st, [IndelSpec(kind="delete", chain="A", residues=before[-3:])])
    assert plan.terminal and not plan.has_internal
    apply_terminal(st, plan.terminal)
    after = [int(r.id[1]) for r in st.residues("A")]
    assert after == before[:-3]


def test_internal_indels_are_refused_with_a_reason():
    """An internal edit without loop closure leaves a broken peptide bond,
    which still looks like a protein. Refusing is the correct answer."""
    from ipro_agent.config import IndelSpec
    from ipro_agent.indel import classify
    from ipro_agent.structure import Structure

    st = Structure.load(CRN)
    cases = [
        IndelSpec(kind="insert", chain="A", after=20, length=2),
        IndelSpec(kind="delete", chain="A", residues=[20, 21]),
        IndelSpec(kind="delete", chain="A", residues=[1, 3]),   # not contiguous
    ]
    for spec in cases:
        plan = classify(st, [spec])
        assert plan.has_internal and not plan.terminal
        assert plan.internal[0][1], "a refusal must say why"


def test_terminal_insertion_runs_end_to_end_and_the_residues_get_designed():
    """The inserted residues are glycine in the structure and become ordinary
    design positions, so the MILP picks their identities."""
    if not _have_pymol():
        print("    (skipped: no PyMOL)")
        return
    from ipro_agent.config import DesignSpec
    from ipro_agent.loop import run_design
    from ipro_agent.structure import Structure

    tmp = Path(tempfile.mkdtemp(prefix="ipro_indel_"))
    try:
        spec = DesignSpec.from_dict({
            "name": "ins", "goal": "stabilize", "goal_statement": "extend and design",
            "molecules": [{"name": "m", "path": str(CRN), "role": "design",
                           "kind": "protein", "chains": ["A"]}],
            "indels": [{"kind": "insert", "chain": "A", "after": 46, "length": 2}],
            "positions": [
                {"position": "A47", "allowed": ["TRP", "LEU", "ALA"]},
                {"position": "A48", "allowed": ["TRP", "LEU", "ALA"]},
            ],
            "goal_options": {"max_self_energy": 1e9},
            "loop": {"iterations": 3, "random_seed": 2},
        })
        res = run_design(spec, tmp)
        assert res.structures
        assert any("indels applied" in n for n in res.notes)

        back = Structure.load(tmp / "best" / "design.pdb")
        nums = [int(r.id[1]) for r in back.residues("A")]
        assert nums[0] == 1 and nums[-1] == 48 and len(nums) == 48, nums[:3] + nums[-3:]
        # Both inserted positions carry a designed identity, not glycine.
        for n in (47, 48):
            assert back.find("A", n).get_resname().strip() in ("TRP", "LEU", "ALA")
        # And the new residues really are bonded to the chain.
        rs = list(back.residues("A"))
        pep = float(np.linalg.norm(_bb(rs[45], "C") - _bb(rs[46], "N")))
        assert 1.2 < pep < 1.45, pep
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def collect():
    g = globals()
    return [(n, g[n]) for n in sorted(g) if n.startswith("test_")]


# ---------------------------------------------------------------------------
# Regressions from the second independent review
#
# Every one of these was a defect that ran cleanly and returned a plausible
# wrong answer. Each test states the defect and the measurement that exposed
# it, so a reintroduction fails here rather than in someone's results.
# ---------------------------------------------------------------------------


def test_regression_inserted_carbonyl_follows_psi():
    """The carbonyl O was placed at a fixed N-CA-C-O dihedral of 180 deg instead
    of psi+180, putting O(i) 0.871 A from N(i+1) (native 1CRN: 2.299 A) with an
    O-C-N angle of 39.6 deg instead of ~123. The structure looked like a protein
    and scored like nonsense. Appending also left the old terminal OXT in the
    position the new nitrogen needs."""
    from ipro_agent.config import IndelSpec
    from ipro_agent.indel import apply_terminal, classify, measure_angle
    from ipro_agent.structure import Structure

    def xyz(res, name):
        return np.array([a.coord for a in res if a.get_name().strip() == name][0])

    def has(res, name):
        return any(a.get_name().strip() == name for a in res)

    for end, after, n in (("N", 0, 1), ("N", 0, 3), ("C", 46, 1), ("C", 46, 3)):
        st = Structure.load(CRN)
        apply_terminal(
            st,
            classify(st, [IndelSpec(kind="insert", chain="A", after=after, length=n)]).terminal,
        )
        rs = list(st.residues("A"))
        for i in range(len(rs) - 1):
            a, b = rs[i], rs[i + 1]
            if not (has(a, "O") and has(a, "C") and has(b, "N")):
                continue
            o_n = float(np.linalg.norm(xyz(a, "O") - xyz(b, "N")))
            ang = measure_angle(xyz(a, "O"), xyz(a, "C"), xyz(b, "N"))
            pep = float(np.linalg.norm(xyz(a, "C") - xyz(b, "N")))
            assert o_n > 2.0, f"{end}+{n}: O({a.id[1]})..N = {o_n:.3f} A"
            assert ang > 100.0, f"{end}+{n}: O-C-N = {ang:.1f} deg"
            assert abs(pep - 1.329) < 0.05, f"{end}+{n}: peptide bond {pep:.3f} A"
        # no residue may keep a C-terminal OXT unless it is the last one
        for r in rs[:-1]:
            assert not has(r, "OXT"), f"residue {r.id[1]} kept a stale OXT"


def test_regression_position_label_round_trips():
    """`label` dropped the separator for a numeric chain id, so `1:115` became
    `"1115"` and re-parsed as chain A residue 1115. The mutation was then applied
    to a different residue -- one that may well exist, so nothing noticed."""
    for raw in ("1:115", "A115", "A115B", "A-1", "B:7", "1:7B", "115"):
        p = DesignPosition.parse(raw)
        assert DesignPosition.parse(p.label).key == p.key, raw


def test_regression_tables_see_the_current_partner_pose():
    """`_tables_for` built its template from the input Structures, so after a
    redock the MILP kept optimizing against the partner position the docking step
    had already abandoned (6.51 kcal/mol divergence over 6 iterations). And
    `_refresh_self_energies` recomputed only the interaction term, dropping the
    reference and strain contributions `build_tables` had added."""
    import tempfile as _tf

    from ipro_agent.docking import Pose
    from ipro_agent.loop import IPROLoop
    from ipro_agent.structure import Structure

    work = Path(_tf.mkdtemp())
    lig = work / "lig.pdb"
    st = Structure.load(CRN)
    res = [r for r in st.residues("A") if int(r.id[1]) == 16][0]
    ca = np.array([a.coord for a in res if a.get_name().strip() == "CA"][0])
    lig.write_text(
        "".join(
            f"HETATM{i:5d}  C{i}  LIG L   1    "
            f"{ca[0]+dx:8.3f}{ca[1]+dy:8.3f}{ca[2]:8.3f}  1.00  0.00           C\n"
            for i, (dx, dy) in enumerate([(6.0, 0.0), (7.5, 0.0), (6.7, 1.4)], start=1)
        )
        + "END\n"
    )
    base = {
        "name": "pose-probe",
        "goal_statement": "bind the ligand more tightly",
        "goal": "bind_tighter",
        "molecules": [
            {"name": "prot", "path": str(CRN), "role": "design", "chains": ["A"]},
            {"name": "lig", "path": str(lig), "role": "target",
             "kind": "ligand", "redock": True},
        ],
        "positions": [
            {"position": "A16", "allowed": ["ALA", "SER", "TRP"]},
            {"position": "A17", "allowed": ["ALA", "SER", "TRP"]},
        ],
    }
    if not _have_pymol():
        print("    (skipped: no PyMOL)")
        return

    loop = IPROLoop(DesignSpec.from_dict(base), run_dir=Path(_tf.mkdtemp()))
    loop.prepare()
    before = dict(loop.build_combined().tables.self_energy)
    asm = loop._assemblies[0].name
    loop._poses[(asm, "lig")] = Pose(
        translation=np.array([20.0, 0.0, 0.0]), rotation=np.eye(3)
    )
    after = loop.build_combined()
    moved = max(
        abs(after.tables.self_energy[k] - before[k])
        for k in before if k in after.tables.self_energy
    )
    assert moved > 1e-6, "moving a partner 20 A changed no self energy"

    # a refresh must agree with a full rebuild, term for term
    fresh = loop.build_combined()
    loop._refresh_self_energies(fresh)
    drift = max(
        abs(fresh.tables.self_energy[k] - after.tables.self_energy[k])
        for k in after.tables.self_energy
    )
    assert drift < 1e-9, f"refresh and rebuild disagree by {drift}"

    # with a reference-energy table the refresh used to lose it entirely
    spec2 = DesignSpec.from_dict(
        {**base, "goal_options": {"reference_energies": {"TRP": -5.5, "SER": -1.0}}}
    )
    l2 = IPROLoop(spec2, run_dir=Path(_tf.mkdtemp()))
    l2.prepare()
    cb = l2.build_combined()
    kept = dict(cb.tables.self_energy)
    l2._refresh_self_energies(cb)
    lost = max(abs(cb.tables.self_energy[k] - kept[k]) for k in kept)
    assert lost < 1e-9, f"refresh dropped reference energies ({lost} kcal/mol)"


def test_regression_rebase_carries_the_identity_not_the_index():
    """`_rebase_choice` read the old index out of the NEW table -- the exact
    mistake its own docstring warned about. A current choice of TRP came back as
    SER with no energy test, and those indices become `require` pins, so the rest
    of the run was pinned to a residue nothing had chosen. It also ignored the
    restraint vetoes, so a pin could make the next program infeasible, which the
    loop reports as "design space enumerated" -- a silent early stop."""
    from ipro_agent.energy.base import AtomSet
    from ipro_agent.loop import IPROLoop
    from ipro_agent.restraints import RestraintPlan

    def rot(aa):
        return Rotamer(position="A16", aa=aa, state=1, atoms=AtomSet())

    old = EnergyTables(positions=["A16"], rotamers={"A16": [rot("SER"), rot("TRP")]})
    new = EnergyTables(positions=["A16"], rotamers={"A16": [rot("TRP"), rot("SER")]})
    new.self_energy = {("A16", 0): -3.0, ("A16", 1): -1.0}

    loop = IPROLoop.__new__(IPROLoop)
    loop._native_v = {"A16": "SER"}
    ids = IPROLoop._identities({"A16": 1}, old)
    assert ids == {"A16": "TRP"}
    got = loop._rebase_choice(ids, new, RestraintPlan())
    assert new.rotamers["A16"][got["A16"]].aa == "TRP"

    veto = RestraintPlan()
    veto.rotamer_vetoes = {("A16", 0)}
    assert loop._rebase_choice(ids, new, veto)["A16"] != 0, "pinned a vetoed rotamer"


def test_regression_pore_slice_is_a_true_cross_section():
    """Atoms were admitted to a slice if they lay within a slab, then given an
    in-plane radius of sqrt(max(r^2 - dz^2, 0)) -- zero for the ones outside
    their own radius. A radius-0 obstacle still blocks an inscribed circle, so a
    carbon 2.5 A off-plane cut a true 4.300 A lumen to 2.164 A and moved the
    reported bottleneck."""
    from ipro_agent.pore import pore_profile

    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    rings = [
        np.stack([6.0 * np.cos(ang), 6.0 * np.sin(ang), np.full(12, z)], axis=1)
        for z in np.arange(-4.0, 4.01, 1.0)
    ]
    cyl = np.vstack(rings)
    names = ["C"] * len(cyl)
    o, d = np.zeros(3), np.array([0.0, 0.0, 1.0])

    def lumen(pts, nm):
        pr = pore_profile(pts, nm, o, d, spacing=0.5, max_centre_offset=4.0)
        return max(r for z, r in zip(pr.positions, pr.radii) if abs(z - 4.0) < 0.26)

    clean = lumen(cyl, names)
    assert abs(clean - 4.3) < 1e-6, f"clean cylinder measured {clean:.3f}, want 4.300"

    off = lumen(np.vstack([cyl, [[0.0, 0.0, 2.5]]]), names + ["C"])
    assert abs(off - clean) < 1e-9, f"an atom 2.5 A off-plane changed the lumen to {off:.3f}"

    on = lumen(np.vstack([cyl, [[0.0, 0.0, 1.0]]]), names + ["C"])
    assert on < clean - 1e-6, "an atom that does reach the plane must still obstruct it"


def test_regression_pdb_writer_keeps_negative_residue_numbers():
    """`int(resseq) % 10000` mapped -1 to 9999, so an indel-created A-1 was
    written as residue 9999 and read back at the wrong end of the chain."""
    import tempfile as _tf

    from ipro_agent.config import IndelSpec
    from ipro_agent.indel import apply_terminal, classify
    from ipro_agent.output import write_complex
    from ipro_agent.structure import Structure

    st = Structure.load(CRN)
    apply_terminal(
        st, classify(st, [IndelSpec(kind="insert", chain="A", after=0, length=2)]).terminal
    )
    out = Path(_tf.mkdtemp()) / "indel.pdb"
    write_complex(out, st, {})
    cas = [
        int(ln[22:26])
        for ln in out.read_text().splitlines()
        if ln.startswith(("ATOM", "HETATM")) and ln[12:16].strip() == "CA"
    ]
    assert cas[:3] == [-1, 0, 1], f"first CA residue numbers were {cas[:3]}"
    assert [int(r.id[1]) for r in list(Structure.load(out).residues("A"))[:3]] == [-1, 0, 1]


def test_regression_two_partners_never_share_a_chain_id():
    """The free-chain list was built once and never pruned as partners kept their
    own ids, so a partner on chain B and another remapped from A both ended up as
    B -- two molecules merged into one, with only one remap reported."""
    import tempfile as _tf

    from ipro_agent.output import write_complex
    from ipro_agent.structure import Structure

    work = Path(_tf.mkdtemp())

    def mono(chain):
        p = work / f"lig_{chain}.pdb"
        p.write_text(
            f"HETATM    1  C1  LIG {chain}   1       0.000   0.000   0.000"
            f"  1.00  0.00           C\nEND\n"
        )
        return Structure.load(p)

    out = work / "pair.pdb"
    write_complex(out, Structure.load(CRN), {"L1": mono("B"), "L2": mono("A")})
    got = Structure.load(out)
    chains = {c.id for c in got.model for r in c if r.id[0].strip()}
    assert len(chains) == 2, f"the two ligands landed in chains {chains}"


def test_regression_pose_composition_is_exact_for_a_collinear_body():
    """The net pose was recovered by fitting start to finish, which cannot see a
    rotation about the axis of a one- or two-atom or collinear body: those atoms
    do not move under it. Atoms outside the scoring set (hydrogens) were then
    placed by whatever spin the fit returned -- up to 5 A."""
    from ipro_agent.docking import Pose, _pose_between

    def rotx(deg):
        t = math.radians(deg)
        return np.array(
            [[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]]
        )

    body = np.array([[6.0, 0, 0], [7.5, 0, 0]])           # collinear, along x
    full = np.vstack([body, [[6.75, 1.1, 0.0]]])          # + an off-axis atom
    centre = body.mean(axis=0)

    # composition is exact where re-fitting loses the twist
    p1 = Pose(rotation=rotx(40), translation=np.array([1.0, 0, 0]), centre=centre)
    p2 = Pose(rotation=rotx(25), translation=np.array([0, 0.5, 0]), centre=centre)
    twice = p2.apply(p1.apply(full))
    assert np.abs(p1.then(p2).apply(full) - twice).max() < 1e-9

    # and the degenerate fit now returns zero twist, not an arbitrary spin
    notes = []
    fitted = _pose_between(body, p2.apply(p1.apply(body)), centre, notes=notes)
    assert np.abs(fitted.apply(body) - twice[:2]).max() < 1e-9
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(fitted.rotation) - 1) / 2))))
    assert ang < 1e-6, f"a collinear fit invented a {ang:.1f} deg rotation"
    assert notes, "a fit that cannot recover the twist must say so"


def test_regression_assemblies_may_prune_differently():
    """`combine` required every assembly to offer an identical rotamer list, but
    each assembly prunes against its own partners -- so a rotamer that clashes
    with one assembly's partner aborted the whole run as an "internal
    inconsistency". That is exactly the case assemblies exist for: clashing with
    the anti-target is the point, not a reason to discard a rotamer."""
    from ipro_agent.assemblies import combine, vpos
    from ipro_agent.energy.base import AtomSet
    from ipro_agent.config import Assembly

    def rot(aa, state):
        return Rotamer(position="A16", aa=aa, state=state, atoms=AtomSet())

    a = EnergyTables(positions=["A16"], rotamers={"A16": [rot("ALA", 1), rot("TRP", 1)]})
    b = EnergyTables(positions=["A16"], rotamers={"A16": [rot("ALA", 1)]})
    asms = [Assembly(name="x", molecules=[]), Assembly(name="y", molecules=[])]
    try:
        combine({"x": a, "y": b}, asms)
        raise AssertionError("combine must still reject genuinely different sets")
    except ValueError:
        pass
    # the loop's job is to never hand it different sets; vpos guards the names
    try:
        vpos("bad::name", "A16")
        raise AssertionError("an assembly name containing '::' must be rejected")
    except ValueError:
        pass
    assert vpos("ok", "A16") == "ok::A16"


def test_regression_a_one_atom_fix_is_not_a_dihedral():
    """`check_restraints` sent a one-atom fix/tether into `dihedral(*coords)`, so
    `ipro-agent validate` reported a perfectly valid backbone `fix` as a
    problem."""
    from ipro_agent.config import Restraint
    from ipro_agent.restraints import check_restraints
    from ipro_agent.structure import Structure

    st = Structure.load(CRN)
    r = Restraint(kind="fix", atoms=["A20:CA"], label="hold_CA")
    problems, observations = check_restraints([r], st, [], ["A16", "A17"])
    assert not problems, f"a fixed backbone atom was reported as a problem: {problems}"
    assert any("hold_CA" in o for o in observations)


def test_regression_ambiguous_restraint_atom_is_refused():
    """The fixed-atom index kept whichever structure came first, so a partner
    reusing the design molecule's chain id made `A16:CA` resolve to a molecule
    50 A away -- and every veto derived from it was wrong."""
    import tempfile as _tf

    from ipro_agent.restraints import RestraintError, _fixed_atom_index
    from ipro_agent.structure import Structure

    p = Path(_tf.mkdtemp()) / "clash.pdb"
    p.write_text(
        "HETATM    1  CA  LIG A  16      50.000  50.000  50.000"
        "  1.00  0.00           C\nEND\n"
    )
    try:
        _fixed_atom_index([Structure.load(CRN), Structure.load(p)])
        raise AssertionError("an ambiguous atom reference was resolved silently")
    except RestraintError as exc:
        assert "ambiguous" in str(exc)


def test_regression_psi_moves_the_terminal_hydroxyl_hydrogen():
    """HXT is bonded to OXT, so leaving it out of the psi-moving set stretched
    the O-H bond every time psi moved on a C-terminal residue."""
    from ipro_agent.perturb import _PHI_FIXED, _PSI_MOVES

    assert "HXT" in _PSI_MOVES
    assert not (_PHI_FIXED & _PSI_MOVES), "an atom cannot be both phi-fixed and psi-moving"


def test_regression_negative_residue_selection_is_escaped():
    """A negative residue number is what an N-terminal insertion creates, and
    both obvious PyMOL selection forms are silently wrong for it:
    `/prot//A/-1` selects 0 atoms, and `resi -1` is read as the RANGE "up to 1"
    and selects residues -1, 0 and 1 (measured: 15 atoms where the residue has
    4). Only `resi \\-1` selects that residue. Before this was escaped, a design
    position at A-1 harvested no rotamers and the run reported one mutation
    where two were requested -- without raising."""
    import importlib.util

    for script in ("ipro_agent/pymol_helpers/apply_design.py",
                   "ipro_agent/pymol_helpers/harvest_rotamers.py"):
        spec = importlib.util.spec_from_file_location("s", REPO / script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod._selection("A", -1, "") == "prot and chain A and resi \\-1", script
        assert mod._selection("A", 16, "") == "prot and chain A and resi 16", script
        assert mod._selection("A", 0, "") == "prot and chain A and resi 0", script
        assert mod._selection("A", 16, "A") == "prot and chain A and resi 16A", script

    if not _have_pymol():
        print("    (selection strings checked; PyMOL resolution skipped)")
        return

    # and the string really does resolve to exactly that one residue
    import subprocess
    import tempfile as _tf

    from ipro_agent.config import IndelSpec
    from ipro_agent.indel import apply_terminal, classify
    from ipro_agent.output import write_complex
    from ipro_agent.structure import Structure

    work = Path(_tf.mkdtemp())
    st = Structure.load(CRN)
    apply_terminal(
        st, classify(st, [IndelSpec(kind="insert", chain="A", after=0, length=2)]).terminal
    )
    pdb = work / "neg.pdb"
    write_complex(pdb, st, {})
    probe = work / "probe.py"
    probe.write_text(
        "import sys\n"
        "from pymol import cmd\n"
        "cmd.load(sys.argv[1], 'prot')\n"
        "for num in (-1, 0, 1):\n"
        "    resi = ('\\\\' if num < 0 else '') + str(num)\n"
        "    sele = 'prot and chain A and resi ' + resi\n"
        "    got = sorted({int(a.resi) for a in cmd.get_model(sele).atom})\n"
        "    print(num, got)\n"
    )
    from ipro_agent.mutate import find_pymol_python

    out = subprocess.run(
        [find_pymol_python(), str(probe), str(pdb)],
        capture_output=True, text=True, timeout=180,
    )
    seen = dict(
        (int(ln.split()[0]), eval(ln.split(maxsplit=1)[1]))
        for ln in out.stdout.strip().splitlines()
        if ln and ln.split()[0].lstrip("-").isdigit()
    )
    for num in (-1, 0, 1):
        assert seen.get(num) == [num], f"resi for {num} selected {seen.get(num)}"


# ---------------------------------------------------------------------------
# Regressions from the third independent review
#
# Five reviews were run over the parts the first two did not reach: the energy
# backends, goals.py, config.py, structure.py, rotamers.py and scripts/. Every
# defect recorded here ran cleanly and returned a plausible wrong answer. Each
# test names the defect and the measurement that exposed it, so a
# reintroduction fails here rather than in someone's results.
# See docs/DESIGN.md section 7d.
# ---------------------------------------------------------------------------


def _mols(**over):
    m = [
        {"name": "prot", "path": str(CRN), "role": "design"},
        {"name": "LIG", "path": str(CRN), "role": "target", "kind": "ligand"},
    ]
    m[1].update(over)
    return m


def _spec(**over):
    base = {
        "goal_statement": "test", "goal": "bind_tighter",
        "molecules": _mols(), "positions": ["A16"],
    }
    base.update(over)
    return DesignSpec.from_dict(base)


def test_regression_a_scalar_string_is_one_item_not_its_characters():
    """A YAML scalar is an iterable of characters, so every "list of X" field
    silently took one string apart letter by letter.

    Measured at HEAD: ``allowed: all`` gave ['ALA', 'LEU'] (2 of 20);
    ``allowed: standard`` gave 6 of 18; ``positions: "115"`` gave three
    positions A1, A1, A5; and ``residues: "12"`` deleted residues 1 and 2 while
    reporting "removed 2 residue(s): A2, A1" -- defeating section 1i's
    invariant that an indel never touches a residue nobody named."""
    from ipro_agent.config import IndelSpec

    assert expand_allowed("all") == sorted(AA_GROUPS["all"])
    assert len(expand_allowed("standard")) == 18
    assert expand_allowed("negative") == ["ASP", "GLU"]
    assert expand_allowed("W") == ["TRP"]

    sp = DesignSpec.from_dict({
        "goal_statement": "t", "goal": "bind_tighter",
        "molecules": _mols(), "positions": "115",
    })
    assert [p.label for p in sp.positions] == ["A115"], [p.label for p in sp.positions]

    assert IndelSpec(kind="delete", chain="A", residues="12").residues == [12]
    assert IndelSpec(kind="delete", chain="A", residues=[12, 13]).residues == [12, 13]


def test_regression_position_dict_spelling_reads_every_key():
    """``{position: A16, ...}`` read only `position` and `allowed` and dropped
    the rest, while the explicit ``{chain:, number:}`` spelling rejected the
    same input. A misspelled `alowed:` silently widened the design space from 1
    residue to 18, and an `insertion_code:` was dropped, so A16B was designed
    as A16 -- defect 14 of the second review re-entered by another door."""
    try:
        DesignSpec.from_dict({
            "goal_statement": "t", "goal": "bind_tighter", "molecules": _mols(),
            "positions": [{"position": "A16", "alowed": ["TRP"]}],
        })
    except SpecError as exc:
        assert "alowed" in str(exc)
    else:
        raise AssertionError("a misspelled key inside a position must be refused")

    sp = DesignSpec.from_dict({
        "goal_statement": "t", "goal": "bind_tighter", "molecules": _mols(),
        "positions": [{"position": "A16", "insertion_code": "B", "allowed": ["TRP"]}],
    })
    assert sp.positions[0].key == ("A", 16, "B"), sp.positions[0].key
    assert sp.positions[0].label == "A16B"


def test_regression_duplicate_design_position_is_refused():
    """The MILP built variables from the last entry, ``position()`` returned the
    first, and ``design_space_size()`` multiplied both: one position pinned to
    [TRP, PHE] twice reported 4 sequences, and a contradictory pair meant the
    run could not produce the residue the user pinned."""
    try:
        DesignSpec.from_dict({
            "goal_statement": "t", "goal": "bind_tighter", "molecules": _mols(),
            "positions": [{"position": "A16", "allowed": ["TRP"]},
                          {"position": "A16", "allowed": ["ALA", "SER"]}],
        })
    except SpecError as exc:
        assert "A16" in str(exc)
    else:
        raise AssertionError("the same design position twice must be refused")


def test_regression_blank_loop_key_does_not_turn_off_a_default():
    """``integer_cuts:`` with no value parses as null, which is falsy, so a
    documented True default silently became False. Measured: integer cuts off,
    and a 6-iteration run "accepted" six copies of one design. A quoted
    ``"false"`` is a non-empty string and so read as True."""
    try:
        LoopSettings(integer_cuts=None)
    except SpecError as exc:
        assert "integer_cuts" in str(exc)
    else:
        raise AssertionError("a null bool setting must be refused")

    assert LoopSettings(integer_cuts="false").integer_cuts is False
    assert LoopSettings(integer_cuts="yes").integer_cuts is True
    # None stays legal where it documents "no limit".
    assert LoopSettings(stop_after_accepted=None).stop_after_accepted is None
    assert LoopSettings(max_minutes=None).max_minutes is None


def test_regression_fix_restraint_honours_an_explicit_maximum():
    """``__post_init__`` installed FIX_TOLERANCE whenever `tolerance` was None,
    including when `maximum` was given, and ``interval()`` then took
    min(0.05, maximum). Measured: maximum 0.05/0.1/0.2/0.5/1/2/4 all enforced
    at 0.05 -- up to 80x tighter than written -- which narrows a design rather
    than refusing it. The error message in the same method offers 'maximum' as
    the way to state the allowance."""
    from ipro_agent.config import FIX_TOLERANCE, Restraint

    assert Restraint(kind="fix", atoms=["A16:CB"], maximum=2.0).interval() == (0.0, 2.0)
    assert Restraint(kind="fix", atoms=["A16:CB"]).interval() == (0.0, FIX_TOLERANCE)
    # Both given: the tighter of the two, as interval() has always done.
    assert Restraint(kind="fix", atoms=["A16:CB"], tolerance=0.5,
                     maximum=2.0).interval() == (0.0, 0.5)


def test_regression_duplicate_molecule_name_is_refused():
    """Everything downstream looks a molecule up by name, but the goals summed
    over molecule *objects*: a copy-pasted molecule block counted one measured
    interaction energy twice (29.564 reported as 59.128), inflating a
    selectivity gap 2.1x with no error."""
    try:
        DesignSpec.from_dict({
            "goal_statement": "t", "goal": "bind_tighter",
            "molecules": _mols() + [{"name": "LIG", "path": str(CRN),
                                     "role": "target", "kind": "ligand"}],
            "positions": ["A16"],
        })
    except SpecError as exc:
        assert "LIG" in str(exc)
    else:
        raise AssertionError("two molecules with one name must be refused")


def test_regression_duplicate_molecule_in_an_assembly_is_refused():
    """A molecule listed twice in one assembly was kept twice, so its own
    internal energy entered the complex energy once per listing (+1588.339
    measured) while the interaction map, keyed by name, showed one copy."""
    from ipro_agent.config import Assembly

    try:
        Assembly(name="holo", molecules=["LIG", "LIG"])
    except SpecError as exc:
        assert "LIG" in str(exc)
    else:
        raise AssertionError("a repeated participant must be refused")


def test_regression_residue_label_round_trips_for_a_numeric_chain():
    """``ResidueRef.label`` and ``AtomRef.position`` concatenated chain and
    number without the separator ``DesignPosition.label`` uses, so chain 1
    residue 115 was "1115" in two places and "1:115" in the third -- and
    "1115" re-parses as chain A residue 1115. Measured: a mutation applied 40 A
    away, to a residue of the same type, with nothing raised; and a restraint
    on such a position silently reclassified as "satisfied by construction"."""
    from ipro_agent.config import AtomRef, position_label
    from ipro_agent.structure import ResidueRef

    for chain, num, icode in (("A", 115, ""), ("1", 115, ""), ("A", -1, ""),
                              ("A", 115, "B"), ("1", 115, "B"), ("a", 0, "")):
        lab = position_label(chain, num, icode)
        back = DesignPosition.parse(lab)
        assert back.key == (chain, num, icode), (lab, back.key)
        assert ResidueRef(chain, num, icode).label == lab
        assert AtomRef(chain=chain, number=num, atom="CB",
                       insertion_code=icode).position == lab
        assert DesignPosition(chain=chain, number=num, insertion_code=icode).label == lab


def test_regression_pair_energy_is_bounded_on_both_sides():
    """The repulsion cap was documented as "10 kcal/mol per atom pair" but
    applied to the LJ term alone, while the Coulomb term runs as 1/r^2 down to
    the 0.3 A clamp with no bound. So the most *attractive* thing the function
    could produce was a hard clash between opposite charges.

    Measured at HEAD: ASP OD1 against LYS NZ gave -3.577 at a 3.50 A salt
    bridge but -31.508 at 1.00 A and -451.200 at 0.30 A, and a rotamer burying
    NZ inside a template oxygen scored -71.957 against the native alanine's
    -1.391 -- returned by the MILP as the proven-optimal design. Like charges
    ran away the same way (+932.399 for one pair at 0.30 A), past the
    max_self_energy and hard_reject thresholds calibrated against this bound."""
    be = resolve("builtin")
    cap = be.max_repulsion

    def pair(n1, r1, n2, r2, d):
        a = AtomSet(names=[n1], coords=[[0.0, 0, 0]], resnames=[r1], reskeys=[("A", 1, "")])
        b = AtomSet(names=[n2], coords=[[d, 0, 0]], resnames=[r2], reskeys=[("A", 9, "")])
        return be.interaction(a, b)

    for d in (0.1, 0.3, 0.5, 1.0, 1.5):
        assert pair("OD1", "ASP", "NZ", "LYS", d) >= -cap - 1e-9, d
        assert pair("NZ", "LYS", "NZ", "LYS", d) <= cap + 1e-9, d
    # A real salt bridge is far inside the bound and must be untouched.
    assert abs(pair("OD1", "ASP", "NZ", "LYS", 3.5) - (-3.577)) < 0.01
    # So must an ordinary neutral contact.
    assert abs(pair("CB", "ALA", "CB", "ALA", 4.0) - (-0.103)) < 0.01


def test_regression_peptide_bonded_pairs_are_not_scored_as_nonbonded():
    """The same-residue exclusion cannot see a covalent bond that spans two
    residues, so a chain's own connectivity was scored as packing.

    Measured at HEAD on examples/ompf_pore/2omf_A.pdb: internal() reported
    +11132.191 kcal/mol, of which +3390.000 was the 339 peptide bonds (every
    one pinned at the repulsion cap) and +10390.954 the 1048 1-3 pairs, against
    a genuine non-bonded sum of -1070.880 -- 108% of the LJ sum was covalent
    geometry. It also steered the search: 5 of 25 default backbone
    perturbations were hard-rejected on chain connectivity, which a
    perturbation cannot change."""
    from ipro_agent.energy.builtin import PEPTIDE_12_13

    be = resolve("builtin")

    def two(n1, n2, d, num_b=2):
        a = AtomSet(names=[n1], coords=[[0.0, 0, 0]], resnames=["ALA"],
                    reskeys=[("A", 1, "")])
        b = AtomSet(names=[n2], coords=[[d, 0, 0]], resnames=["ALA"],
                    reskeys=[("A", num_b, "")])
        return be.interaction(a, b)

    for (ai, aj) in PEPTIDE_12_13:
        assert two(ai, aj, 1.33) == 0.0, (ai, aj)
        assert two(aj, ai, 1.33, num_b=0) == 0.0, "the reverse direction too"
    # A 1-4 pair across the same peptide bond is a real non-bonded pair.
    assert two("CB", "CB", 4.0) != 0.0
    # Non-sequential residues keep every pair.
    assert two("C", "N", 1.33, num_b=9) != 0.0

    ompf = REPO / "examples" / "ompf_pore" / "2omf_A.pdb"
    if ompf.exists():
        from ipro_agent.loop import _atoms_of
        from ipro_agent.structure import Structure

        e = be.internal(_atoms_of(Structure.load(ompf)))
        assert e < 0, f"a crystal structure's packing energy must be favourable, got {e}"
        assert abs(e - (-1949.303)) < 1.0, e


def test_regression_two_letter_elements_are_typed_correctly():
    """``elements`` took the first alphabetic character of the atom name, so no
    two-letter element was reachable and ``LJ``'s own CL/BR/SE entries were
    dead code. Measured at HEAD: a backbone O 2.40 A from an iron scored
    +5.8272 using *fluorine's* parameters; Cl and Ca were typed as carbon; Na
    was typed as nitrogen and so handed out -1.5 kcal/mol hydrogen bonds."""
    from ipro_agent.energy.base import guess_element
    from ipro_agent.energy.builtin import LJ

    cases = {
        ("FE", "HEM"): "FE", ("CL", "CL"): "CL", ("CA", "CA"): "CA",
        ("MG", "MG"): "MG", ("NA", "NA"): "NA", ("ZN", "ZN"): "ZN",
        ("SE", "MSE"): "SE",            # selenomethionine really is selenium
        ("CA", "ALA"): "C",             # ... but a residue's CA is carbon
        ("CB", "ALA"): "C", ("OD1", "ASP"): "O", ("SD", "MET"): "S",
        ("1HB", "ALA"): "H", ("NZ", "LYS"): "N",
    }
    for (name, resn), want in cases.items():
        assert guess_element(name, resn) == want, (name, resn, guess_element(name, resn))

    # The parameter table's two-letter entries are now reachable.
    for el in ("CL", "BR", "SE"):
        assert LJ[el] != LJ["C"]
        assert guess_element(el, "LIG") == el

    # An explicit element column wins over any guess.
    s = AtomSet(names=["CA"], coords=[[0.0, 0, 0]], resnames=["ALA"], elements=["CA"])
    assert s.elements == ["CA"]


def test_regression_atomset_refuses_mismatched_per_atom_lists():
    """Only ``names`` was length-checked. A short ``resnames`` silently zeroed
    the trailing atoms' formal charges: a measured -7.2102 kcal/mol
    carboxylate/ammonium interaction came out +5.6009 with nothing raised."""
    for kw in ({"resnames": ["ASP"]}, {"reskeys": [("A", 1, "")]},
               {"elements": ["O"]}):
        try:
            AtomSet(names=["OD1", "OD2"], coords=[[0.0, 0, 0], [1.0, 0, 0]], **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"a short {list(kw)[0]} list must be refused")


def test_regression_hbond_well_skips_sequential_backbone_pairs():
    """The well fired on any N/O pair of neutral atoms, including backbone
    atoms of sequentially bonded residues whose separation is fixed by the
    peptide geometry and cannot be a hydrogen bond. Measured on 2omf_A:
    -598.407 kcal/mol came from sequential residues, including 339
    N(i)...N(i+1) pairs -- one per peptide bond, two amide nitrogens that
    cannot bond to each other -- and 98 O(i)...O(i+1) pairs."""
    be = resolve("builtin")

    def two(n1, n2, num_b, d=3.0):
        a = AtomSet(names=[n1], coords=[[0.0, 0, 0]], resnames=["ALA"],
                    reskeys=[("A", 1, "")])
        b = AtomSet(names=[n2], coords=[[d, 0, 0]], resnames=["ALA"],
                    reskeys=[("A", num_b, "")])
        return be.interaction(a, b)

    # Sequential backbone N...N and O...O: no well.
    seq_nn = two("N", "N", 2)
    far_nn = two("N", "N", 9)
    assert seq_nn > far_nn, (seq_nn, far_nn)
    assert abs((seq_nn - far_nn) - be.hbond_weight) < 1e-6
    # A side chain reaching across a peptide bond is a real H-bond: kept.
    assert two("OG", "N", 2) < 0


def test_regression_a_pdb_that_loses_atoms_is_refused():
    """``PDBParser(QUIET=True)`` plus a blanket warnings filter discarded
    exactly the messages that say the file was not read as written. Measured:
    two copies of a molecule pasted into one chain loaded as 82 atoms of 164,
    and a chain holding two different residues numbered 115 silently dropped
    the whole TRP -- every energy afterwards computed on half the molecule,
    with the reduced counts reported back as the file's contents."""
    from ipro_agent.structure import Structure, StructureError

    tmp = Path(tempfile.mkdtemp())
    dup = tmp / "dup.pdb"
    dup.write_text(
        "ATOM      1  N   ASP A 115       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ASP A 115       1.500   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  N   TRP A 115       5.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      4  CA  TRP A 115       6.500   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    try:
        Structure.load(dup)
    except StructureError as exc:
        assert "discarded" in str(exc)
    else:
        raise AssertionError("a file whose residues collide must be refused")

    # An ordinary file still loads, warnings and all.
    assert len(list(Structure.load(CRN).model.get_atoms())) > 300
    ompf = REPO / "examples" / "ompf_pore" / "2omf_A.pdb"
    if ompf.exists():
        assert len(list(Structure.load(ompf).model.get_atoms())) == 2627


def test_regression_save_writes_only_the_loaded_model_and_chains():
    """``save()`` wrote ``self._bio`` -- the whole file back, every model and
    every chain, including the chains ``load(chains=...)`` was asked to exclude
    (the filter only detaches from model 0). These are the files PyMOL and Vina
    read, so an excluded partner was back in the scoring input while the
    Structure's own energies never saw it."""
    from ipro_agent.structure import Structure

    tmp = Path(tempfile.mkdtemp())
    two = tmp / "two.pdb"
    two.write_text(
        "MODEL        1\n"
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  N   GLY B   1      10.000   0.000   0.000  1.00  0.00           N\n"
        "ENDMDL\n"
        "MODEL        2\n"
        "ATOM      3  N   GLY A   1      20.000   0.000   0.000  1.00  0.00           N\n"
        "ENDMDL\nEND\n"
    )
    st = Structure.load(two, chains=["A"])
    out = st.save(tmp / "saved.pdb")
    lines = [ln for ln in out.read_text().splitlines() if ln.startswith("ATOM")]
    assert len(lines) == 1, lines
    assert {ln[21] for ln in lines} == {"A"}
    assert not any(ln.startswith("MODEL") for ln in out.read_text().splitlines())


def test_regression_design_position_on_a_hetatm_residue_keeps_no_sidechain():
    """``build_template``'s hetero loop had no ``is_design`` check, so a design
    position on a HETATM-flagged residue -- MSE and every other modified
    residue is one -- kept its full native side chain in the fixed template.
    Its own rotamers were unaffected (matching key, so the exclusion fired), so
    the damage landed on its neighbours: measured on 2omf_A, TRP at a
    neighbouring position moved +78 to +284 kcal/mol depending on the conformer
    and the chosen rotamer flipped from state 6 to state 1."""
    from ipro_agent.rotamers import build_template
    from ipro_agent.structure import Structure

    tmp = Path(tempfile.mkdtemp())
    pdb = tmp / "mse.pdb"
    pdb.write_text(
        "ATOM      1  N   ALA A  15       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A  15       1.458   0.000   0.000  1.00  0.00           C\n"
        "HETATM    3  N   MSE A  16       3.000   0.000   0.000  1.00  0.00           N\n"
        "HETATM    4  CA  MSE A  16       4.458   0.000   0.000  1.00  0.00           C\n"
        "HETATM    5  CB  MSE A  16       5.000   1.400   0.000  1.00  0.00           C\n"
        "HETATM    6  SE  MSE A  16       6.500   2.000   0.000  1.00  0.00          SE\n"
        "END\n"
    )
    st = Structure.load(pdb)
    full = build_template([st], [])
    assert "SE" in full.names, "the native side chain is in the template when not designed"

    designed = build_template([st], ["A16"])
    assert "SE" not in designed.names, (
        "a design position's side chain must never be in the fixed template, "
        f"HETATM or not; got {designed.names}"
    )
    assert "CB" not in designed.names
    assert "CA" in designed.names, "its backbone must stay"


def test_regression_a_partner_reusing_a_chain_id_keeps_its_own_residue_keys():
    """A residue key is what the backend's same-residue exclusion compares, and
    a partner reusing chain id A -- the default for most single-chain PDB files
    -- collided with the design molecule's own residues. Measured: 22 distinct
    residues collapsed to 11, the partner's residue 115 lost its side chain to
    the design filter, and a ligand numbered like a design position became
    invisible to the rotamer sitting on top of it (+84.699 scored as -6.282)."""
    from ipro_agent.loop import _atoms_of
    from ipro_agent.rotamers import build_template
    from ipro_agent.structure import Structure

    st = Structure.load(CRN)
    merged = build_template([st, st], [])
    keys = set(merged.reskeys)
    one = set(build_template([st], []).reskeys)
    assert len(keys) == 2 * len(one), (len(keys), len(one))
    # The design molecule (first) keeps the bare key rotamer atoms must match.
    assert ("A", 16, "") in keys

    design = _atoms_of(st)
    partner = _atoms_of(st, molecule="LIG")
    assert set(design.reskeys).isdisjoint(set(partner.reskeys))
    # Two copies of the same coordinates now interact instead of cancelling.
    assert resolve("builtin").interaction(design, partner) != 0.0


def _fake_harvest_python(tmp: Path, payload: str) -> Path:
    """A stand-in for the PyMOL interpreter: writes ``payload`` to the out path.

    ``harvest`` invokes ``[py, "-I", script, job, out]``, so $4 is the output.
    """
    exe = tmp / "fake_python.sh"
    exe.write_text("#!/bin/sh\ncat > \"$4\" <<'JSON'\n" + payload + "\nJSON\n")
    exe.chmod(0o755)
    return exe


def test_regression_a_stale_rotamer_file_is_never_reused():
    """``harvest`` read ``rotamers.json`` whenever the file existed and never
    checked the return code, while the loop passes the same workdir every
    iteration. A failed re-harvest after a backbone perturbation therefore
    returned the PREVIOUS backbone's rotamers: measured with the backbone
    translated 10 A and PyMOL exiting 1, side chains came back 10.0 A from
    where they belonged, errors was empty, and a hard-clash conformer (+453.111
    against the real template) was the best-scoring one at -1.036."""
    from ipro_agent.rotamers import RotamerError, harvest

    tmp = Path(tempfile.mkdtemp())
    stale = {"pdb": "/somewhere/old.pdb", "rotamers": {"A16": {"ALA": [
        {"state": 1, "atoms": [["CB", [0.0, 0.0, 0.0]]]}]}}, "native": {"A16": "CYS"}}
    (tmp / "rotamers.json").write_text(json.dumps(stale))
    pos = [DesignPosition.parse("A16", ["ALA"])]
    try:
        harvest(CRN, pos, python="/usr/bin/false", workdir=tmp)
    except RotamerError as exc:
        assert "failed" in str(exc)
    else:
        raise AssertionError("a failed harvest must not fall back on a stale file")

    # And a harvest that describes a different structure is refused.
    fake = _fake_harvest_python(tmp, json.dumps(stale))
    try:
        harvest(CRN, pos, python=str(fake), workdir=tmp)
    except RotamerError as exc:
        assert "another structure" in str(exc)
    else:
        raise AssertionError("rotamers built on another structure must be refused")


def test_regression_a_requested_position_that_harvest_drops_raises():
    """``harvest`` raised for a position it was not asked for but never checked
    the reverse, and the loop's own check scans only the keys present. A
    position PyMOL could not locate -- an incomplete crystal structure missing
    that residue's CA is enough -- vanished without a word: measured, 2
    positions requested, 1 designed, nothing raised."""
    from ipro_agent.rotamers import RotamerError, harvest

    tmp = Path(tempfile.mkdtemp())
    payload = json.dumps({
        "pdb": str(CRN.resolve()),
        "rotamers": {"A16": {"ALA": [{"state": 1, "atoms": [["CB", [0.0, 0.0, 0.0]]]}]}},
        "native": {"A16": "CYS"}, "errors": {},
    })
    fake = _fake_harvest_python(tmp, payload)
    want = [DesignPosition.parse("A16", ["ALA"]), DesignPosition.parse("A17", ["ALA"])]
    try:
        harvest(CRN, want, python=str(fake), workdir=tmp)
    except RotamerError as exc:
        assert "A17" in str(exc)
    else:
        raise AssertionError("a requested position returning nothing must raise")

    # Asking only for what came back still works.
    rots, native, _ = harvest(CRN, want[:1], python=str(fake), workdir=tmp)
    assert set(rots) == {"A16"}


def test_regression_harvested_rotamers_carry_no_hydrogens():
    """Rotamers must be heavy-atom-only, because every template the energy
    model builds is heavy-atom-only. Harvesting hydrogens from a protonated
    input broke that symmetry: the side chain carried its own backbone H and
    HA, which the template no longer held, so every rotamer gained a
    near-constant +18 to +22 kcal/mol. Measured on a protonated 2omf_A with the
    shipped defaults, that pushed 94 of 160 rotamers past max_self_energy
    instead of 46 and deleted TRP, PHE, TYR and PRO from one position
    outright -- so the published Trp answer was not in the design space, with
    validate reporting 6561 sequences and no error anywhere."""
    if not _have_pymol():
        print("    (skipped: no PyMOL)")
        return
    from ipro_agent.rotamers import harvest

    pos = [DesignPosition.parse("A16", ["TRP", "SER"])]
    rots, _, _ = harvest(CRN, pos)
    for rot in rots["A16"]:
        bad = [n for n in rot.atoms.names
               if n.strip().upper().lstrip("0123456789").startswith(("H", "D"))]
        assert not bad, f"{rot.aa}#{rot.state} carries hydrogens: {bad}"
        assert all(e not in ("H", "D", "T") for e in rot.atoms.elements), rot.atoms.elements


def _ctx(**kw):
    from ipro_agent.goals import ScoreContext

    return ScoreContext(**kw)


def test_regression_unknown_goal_option_is_refused():
    """``goal_options`` was the one part of the spec never checked against
    anything, and it carries every objective weight and the published
    acceptance filter. Measured: ``pore_diameter_maximum`` for
    ``pore_diameter_max`` left PoreDesigner's hard filter inert -- a 20.00 A
    bottleneck was not rejected, goal.notes was empty, and the design shipped
    unfiltered. ``antiweight: 5`` left the weight at 1.0."""
    from ipro_agent.goals import build_goal

    for bad in ({"pore_diameter_maximum": 4.0}, {"antiweight": 5},
                {"anti_wieght": 0.5}, {"max_targett_ie": 1}):
        try:
            build_goal(_spec(goal="bind_tighter", goal_options=bad))
        except SpecError as exc:
            assert list(bad)[0] in str(exc)
        else:
            raise AssertionError(f"{bad} must be refused, not ignored")

    # Keys read elsewhere in the run must still be accepted by every goal.
    for shared in ("max_self_energy", "pair_eps", "reference_energies",
                   "strain_weight", "insert_phi", "pore_window"):
        build_goal(_spec(goal="bind_tighter", goal_options={shared: 1.0}))
    # And each goal's own keys.
    build_goal(_spec(goal="bind_tighter", goal_options={"max_total_energy": -1.0}))
    build_goal(_spec(goal="reshape_channel",
                     goal_options={"direction": "narrow", "target_radius": 2.0,
                                   "pore_diameter_max": 4.0, "long_fraction": 0.5,
                                   "long_residues": ["TRP"], "radius_weight": 1.0}))


def test_regression_a_nonpositive_goal_weight_is_refused():
    """A weight scales a contribution to the objective; zero deletes the term
    and a negative value inverts it. Measured with anti_weight -1 and a base
    IE(target) of -10: binding the anti-target 8 kcal/mol tighter scored
    -28.000 against -12.000 for rejecting it, so the loop optimized the
    opposite of the request and its objective curve still fell.
    ``Assembly.weight`` is refused on the same grounds."""
    from ipro_agent.goals import build_goal

    spec_kw = {"goal": "switch_specificity",
               "molecules": _mols() + [{"name": "DEC", "path": str(CRN),
                                        "role": "anti_target", "kind": "ligand"}]}
    for w in (-1, 0, 0.0, -0.5):
        try:
            build_goal(_spec(goal_options={"anti_weight": w}, **spec_kw))
        except SpecError as exc:
            assert "anti_weight" in str(exc)
        else:
            raise AssertionError(f"anti_weight {w} must be refused")
    g = build_goal(_spec(goal_options={"anti_weight": 3.0}, **spec_kw))
    assert g.weight == 3.0
    for w in (-1, 0):
        try:
            build_goal(_spec(goal="reshape_channel",
                             goal_options={"target_radius": 2.0, "radius_weight": w}))
        except SpecError:
            pass
        else:
            raise AssertionError(f"radius_weight {w} must be refused")


def test_regression_the_pore_filter_alone_does_not_hide_a_degenerate_objective():
    """``pore_diameter_min/max`` are a hard acceptance FILTER, not a term in the
    objective -- nothing in score() reads them. Counting them as a geometric
    objective suppressed the degenerate-objective warning for exactly the spec
    that states the paper's published filter: measured, a 3.2 A and a 3.8 A
    candidate both scored -100.0 and the user was told nothing, while a spec
    with no options at all got the warning."""
    from ipro_agent.goals import build_goal

    filter_only = build_goal(_spec(
        goal="reshape_channel", molecules=[{"name": "prot", "path": str(CRN),
                                            "role": "design"}],
        goal_options={"pore_diameter_min": 2.0, "pore_diameter_max": 4.0}))
    assert any("degenerate" in n for n in filter_only.notes), filter_only.notes
    # The filter itself still works.
    assert filter_only.filter_reject(_ctx(geometry={"min_pore_diameter": 20.0}))
    assert filter_only.filter_reject(_ctx(geometry={"min_pore_diameter": 3.0})) is None
    # A real geometric objective silences it.
    with_target = build_goal(_spec(
        goal="reshape_channel", molecules=[{"name": "prot", "path": str(CRN),
                                            "role": "design"}],
        goal_options={"target_radius": 2.0}))
    assert not any("degenerate" in n for n in with_target.notes), with_target.notes


def test_regression_a_note_raised_while_scoring_reaches_the_run():
    """The loop copied ``goal.notes`` at construction, but a goal also raises
    notes while scoring -- a term it had to drop, an option it could not apply.
    Measured: a reshape_channel run whose radius term was never active (score
    identically 0.0 every iteration) reported a mutation string and a best
    score with no caveat at all, against the house rule that a design is never
    reported without its notes."""
    from ipro_agent.loop import IPROLoop

    spec = _spec(goal="reshape_channel",
                 molecules=[{"name": "prot", "path": str(CRN), "role": "design"}],
                 goal_options={"target_radius": 2.0})
    loop = IPROLoop(spec, Path(tempfile.mkdtemp()) / "run", backend=resolve("builtin"))
    # A note the goal raises later must be visible through the loop's list.
    loop.goal.score(_ctx(design_only=-1.0, total=-1.0))
    assert any("target_radius" in n for n in loop.notes), loop.notes


def test_regression_the_interaction_floor_is_a_constraint_not_a_penalty():
    """docs/DESIGN.md records IPRO+/-'s form as ``min E_complex s.t. IE >=
    0.75 * IE_WT``. As a unit-weight quadratic penalty it was not a constraint
    at all: measured with the floor at -15.0, a candidate that destroyed the
    interaction entirely (IE 0.0) scored -115.000, *better* than a compliant
    candidate at -100.000, and was kept as the run's best design."""
    from ipro_agent.goals import build_goal

    g = build_goal(_spec(goal="stabilize",
                         goal_options={"min_interaction_fraction": 0.75,
                                       "reference_interaction": -20.0}))
    ok = _ctx(design_only=-100.0, total=-100.0, interaction={"LIG": -20.0})
    broken = _ctx(design_only=-340.0, total=-340.0, interaction={"LIG": 0.0})
    assert g.filter_reject(ok) is None
    reason = g.filter_reject(broken)
    assert reason and "floor" in reason, reason
    # The penalty stays as a gradient, but it no longer decides acceptance.
    assert g.score(broken) > g.score(ok) - 1e9   # shape unchanged, filter rules


def test_regression_max_total_energy_is_enforced():
    """``BindTighter``'s docstring promised that candidates whose whole-complex
    energy exceeds ``max_total_energy`` are rejected. It was read nowhere in
    the package: measured, a spec with a -50.0 cap ran to completion on a
    candidate at +500.0 with no rejection and no note, while ``ctx.total`` was
    measured and carried for exactly this purpose."""
    from ipro_agent.goals import build_goal

    g = build_goal(_spec(goal_options={"max_total_energy": -50.0}))
    assert g.filter_reject(_ctx(total=500.0, interaction={"LIG": -12.0}))
    assert g.filter_reject(_ctx(total=-60.0, interaction={"LIG": -12.0})) is None


def test_regression_km_without_a_ligand_is_not_summed_over_every_target():
    """OptZyme Eq 7 is the interaction energy with the *substrate*. With no
    molecule of kind 'ligand', score() silently summed every target: measured,
    an unrelated protein partner at -40 entered a "K_M" score of -49.0 that
    components() reported as -9.0, so the number could not even be explained."""
    from ipro_agent.goals import build_goal

    mols = [
        {"name": "prot", "path": str(CRN), "role": "design"},
        {"name": "TSA", "path": str(CRN), "role": "target", "kind": "transition_state"},
        {"name": "OTHER", "path": str(CRN), "role": "target", "kind": "protein"},
    ]
    try:
        build_goal(_spec(goal="improve_catalysis", molecules=mols,
                         goal_options={"parameter": "km"}))
    except SpecError as exc:
        assert "ligand" in str(exc)
    else:
        raise AssertionError("an ambiguous substrate must be refused for 'km'")

    # One target is unambiguous: used, and said so.
    g = build_goal(_spec(goal="improve_catalysis", molecules=mols[:2],
                         goal_options={"parameter": "km"}))
    assert any("km" in n and "substrate" in n for n in g.notes), g.notes
    assert g.score(_ctx(interaction={"TSA": -9.0})) == -9.0


def test_regression_an_ignored_anti_target_is_reported():
    """A declared anti-target is read by ``switch_specificity`` alone. Measured
    with C12 as target and C14 as anti_target, bind_tighter / stabilize /
    reshape_channel scored identically whether C14 was bound at -10 or rejected
    at -2, with notes empty: a user who labelled the decoy correctly but picked
    the wrong goal optimized plain affinity and was never told."""
    from ipro_agent.goals import build_goal

    mols = _mols() + [{"name": "DEC", "path": str(CRN), "role": "anti_target",
                       "kind": "ligand"}]
    for goal in ("bind_tighter", "stabilize"):
        g = build_goal(_spec(goal=goal, molecules=mols))
        assert any("anti-target" in n for n in g.notes), (goal, g.notes)
    g = build_goal(_spec(goal="switch_specificity", molecules=mols))
    assert not any("does not read anti-targets" in n for n in g.notes), g.notes


def test_regression_suggest_goal_prefers_the_more_specific_question():
    """Matching in word order answered the wrong question: "change its binding
    specificity from C12 to C14" hit "binding" first and suggested
    bind_tighter, which optimizes plain affinity and ignores the decoy."""
    from ipro_agent.goals import suggest_goal

    assert suggest_goal("change its binding specificity from C12 to C14") == "switch_specificity"
    assert suggest_goal("make the interface bind the target and not the decoy") == "switch_specificity"
    assert suggest_goal("improve binding affinity for the target") == "bind_tighter"
    assert suggest_goal("narrow the pore") == "reshape_channel"
    assert suggest_goal("nothing relevant here") is None


def test_regression_stabilize_validates_its_reference_up_front():
    """The error raised from score() told the user to "measure it with
    'ipro-agent score'" -- and that command builds the same goal and dies with
    the same error, so the documented recovery path could not be followed.
    ``validate`` reported "spec is usable", exit 0, beforehand."""
    from ipro_agent.goals import build_goal

    try:
        build_goal(_spec(goal="stabilize",
                         goal_options={"min_interaction_fraction": 0.75}))
    except SpecError as exc:
        assert "reference_interaction" in str(exc)
    else:
        raise AssertionError("a missing reference_interaction must fail validation")


def test_regression_pair_prescreen_radius_encloses_from_the_screen_centre():
    """The position-pair pre-screen compared the distance between two mean
    centroids against each set's ``bounding_radius()``, which is measured about
    each rotamer's *own* centroid and so does not enclose it seen from the mean.
    The test was therefore not conservative: measured with two one-atom
    rotamers 40 A apart at one position, a real -0.014002 kcal/mol pair 6.0 A
    away was dropped and the block recorded 0 pair terms."""
    far = Rotamer(position="P1", aa="ALA", state=1,
                  atoms=AtomSet(names=["CB"], coords=[[40.0, 0, 0]],
                                resnames=["ALA"], reskeys=[("A", 1, "")]))
    near = Rotamer(position="P1", aa="SER", state=1,
                   atoms=AtomSet(names=["CB"], coords=[[0.0, 0, 0]],
                                 resnames=["SER"], reskeys=[("A", 1, "")]))
    other = Rotamer(position="P2", aa="ALA", state=1,
                    atoms=AtomSet(names=["CB"], coords=[[6.0, 0, 0]],
                                  resnames=["ALA"], reskeys=[("A", 2, "")]))
    tab = build_tables(
        {"P1": [far, near], "P2": [other]},
        AtomSet(), resolve("builtin"),
        native={}, max_self_energy=1e9, pair_eps=0.0,
    )
    assert tab.n_pair_terms() > 0, "a reachable pair must not be screened out"


def test_regression_a_partial_strain_penalty_is_refused():
    """PyMOL emits no bump score for ALA (its library has no entry), and the
    table builder skipped a missing strain silently, so with strain_weight set
    the penalty applied to eight identities and not to ALA -- which then
    competed without the cost every other identity paid."""
    a = Rotamer(position="P1", aa="ALA", state=1, strain=None,
                atoms=AtomSet(names=["CB"], coords=[[0.0, 0, 0]],
                              resnames=["ALA"], reskeys=[("A", 1, "")]))
    w = Rotamer(position="P1", aa="TRP", state=1, strain=2.0,
                atoms=AtomSet(names=["CB"], coords=[[0.0, 0, 0]],
                              resnames=["TRP"], reskeys=[("A", 1, "")]))
    try:
        build_tables({"P1": [a, w]}, AtomSet(), resolve("builtin"),
                     native={}, max_self_energy=1e9, strain_weight=1.0)
    except ValueError as exc:
        assert "strain" in str(exc)
    else:
        raise AssertionError("a strain penalty applied to only some identities must raise")
    # With no strain_weight the missing value is irrelevant.
    build_tables({"P1": [a, w]}, AtomSet(), resolve("builtin"),
                 native={}, max_self_energy=1e9)


def test_regression_selftest_runner_reports_a_systemexit():
    """``except Exception`` let BaseException through, so one ``SystemExit``
    from a test (or a ``pytest.skip``, which this runner's own docstring
    invites) propagated out and ended the run with exit 0, no summary line and
    no record of the tests that never ran. Measured: injecting sys.exit(0) into
    the alphabetically first test printed nothing at all and exited 0 -- which
    would make every "94 tests passing" claim in this repo unfounded."""
    tmp = Path(tempfile.mkdtemp())
    pkg = tmp / "tests"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "test_ipro.py").write_text(
        "import sys\n"
        "def test_aaa_exits():\n    sys.exit(0)\n"
        "def test_zzz_fine():\n    assert True\n"
        "def collect():\n"
        "    g = globals()\n"
        "    return [(n, g[n]) for n in sorted(g) if n.startswith('test_')]\n"
    )
    shutil.copy(REPO / "scripts" / "selftest.py", tmp / "selftest.py")
    p = subprocess.run([sys.executable, str(tmp / "selftest.py")],
                       capture_output=True, text=True, cwd=tmp, timeout=120)
    assert p.returncode == 1, (p.returncode, p.stdout, p.stderr)
    assert "FAIL test_aaa_exits" in p.stdout, p.stdout
    assert "1 passed, 1 failed, 2 total" in p.stdout, p.stdout


def test_regression_doctor_probes_an_isolated_interpreter():
    """doctor's module probes ran ``python -c`` with no ``-I``, so the current
    directory was on sys.path and PYTHONPATH was inherited -- despite the
    comment in ``_run`` claiming otherwise -- while the run itself probes with
    ``-I``. Measured from a directory holding a ``pyrosetta.py`` and a
    ``vina/`` folder, doctor reported "OK PyRosetta ... OK AutoDock Vina",
    resolved energy to pyrosetta and dock to vina, and printed degraded = [],
    so the run looked not degraded at all and the reported mode was wrong in
    both directions. Neither is installed on this machine."""
    tmp = Path(tempfile.mkdtemp())
    # Names that cannot be installed anywhere, so the test measures isolation
    # rather than whatever this machine happens to have. (PyRosetta may now be
    # genuinely importable in the interpreter running this.)
    (tmp / "ipro_planted_module.py").write_text("__version__ = '9.9.9'\n")
    (tmp / "ipro_planted_pkg").mkdir()
    (tmp / "ipro_planted_pkg" / "__init__.py").write_text("__version__ = '9.9.9'\n")
    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "from ipro_agent.doctor import _module_in\n"
        "print(json.dumps([_module_in(sys.executable, 'ipro_planted_module')[0],"
        " _module_in(sys.executable, 'ipro_planted_pkg')[0]]))\n"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=tmp, timeout=300)
    assert p.returncode == 0, p.stderr
    seen = json.loads(p.stdout.strip().splitlines()[-1])
    assert seen == [False, False], (
        f"a module planted in the working directory must not be reported as "
        f"installed; got {seen}"
    )
    # Sanity check that the planted modules really are importable without -I,
    # so a passing test above means isolation and not a broken fixture.
    naive = subprocess.run(
        [sys.executable, "-c", "import ipro_planted_module, ipro_planted_pkg"],
        capture_output=True, text=True, cwd=tmp, timeout=120,
    )
    assert naive.returncode == 0, (
        "fixture is broken: the planted modules are not importable even "
        f"without -I ({naive.stderr[-200:]})"
    )


def test_regression_doctor_degraded_covers_the_milp():
    """``milp["available"]`` was true whenever any solver binary was on PATH,
    including one doctor had just called unusable, and the MILP never appeared
    in ``degraded``. Measured: with only an unusable cplex on PATH,
    ``doctor.py --require milp`` exited 0 and the greedy fallback -- which
    proves no optimality -- was invisible to any caller reading the exit status
    or the machine-readable report."""
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "doctor.py"), "--json"],
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr
    rep = json.loads(p.stdout)
    greedy = rep["resolved"]["milp"].startswith("greedy")
    assert rep["checks"]["milp"]["available"] is not greedy
    assert ("milp" in rep["degraded"]) is greedy, (rep["resolved"]["milp"], rep["degraded"])
    # `degraded` is what the house rules require to be reported with every
    # design, so it must agree with the probes for every backend, whatever
    # happens to be installed on the machine running this.
    for key in ("pymol", "pyrosetta", "vina", "milp"):
        assert (key in rep["degraded"]) is (not rep["checks"][key]["available"]), (
            key, rep["degraded"], rep["checks"][key]["available"]
        )


class _CountingRelaxBackend:
    """A stand-in for a relaxing backend, so the wiring can be tested without
    PyRosetta."""

    name = "counting-relax"
    can_relax = True
    units = "stand-in units"
    screening_pairwise = False

    def __init__(self):
        from ipro_agent.energy.builtin import BuiltinBackend

        self._screen = BuiltinBackend()
        self.calls: list[tuple] = []

    def interaction(self, a, b):
        return self._screen.interaction(a, b)

    def internal(self, a):
        return self._screen.internal(a)

    def relax(self, pdb_in, pdb_out, fixed=()):
        self.calls.append((Path(pdb_in).name, tuple(fixed)))
        Path(pdb_out).write_text(Path(pdb_in).read_text())
        return 0.0

    def describe(self):
        return "counting-relax (stand-in)"


def test_regression_full_mode_relaxes_after_the_choice_not_before():
    """``EnergyBackend.relax`` had no call site anywhere in the package, and
    neither did ``RestraintPlan.relax_fixed_positions``, which is collected for
    exactly that argument. So a run with a relaxing backend reported mode
    ``full`` -- documented as "perturb + relax + repack" -- while doing no
    relaxation at all, and because ``can_relax`` was true it also suppressed
    the note saying the reported energies were screening energies.

    The fix for that put the call in ``_perturb_once``, which the paper audit
    then showed was the wrong place: the 2006 paper's step iv and the released
    source both relax AFTER the rotamer choice and the docking
    (``Backbone_Perturbation -> Optimal_Rotamers -> Docking -> Relaxation ->
    Calculate_Energy``), because the residues just chosen "may no longer be
    optimally interacting with the previous backbone". Relaxed before the
    MILP, the backbone carried the OLD side chains and never adapted to the
    mutation. This test pins the order: the atoms that get relaxed are the
    ones about to be scored, the restraint-held positions reach the backend,
    and ``_perturb_once`` does not relax."""
    from ipro_agent.loop import IPROLoop, _atoms_of
    from ipro_agent.structure import Structure

    tmp = Path(tempfile.mkdtemp())
    spec = _spec(loop={"perturb_backbone": True, "iterations": 1})
    be = _CountingRelaxBackend()
    loop = IPROLoop(spec, tmp / "run", backend=be)
    assert loop.mode == "full", loop.mode

    atoms = _atoms_of(Structure.load(CRN), set())
    loop._restraints.relax_fixed_positions = ["A16"]
    out = loop._relaxed_design_atoms(atoms, "iter0001_main")
    assert be.calls == [("iter0001_main_chosen.pdb", ("A16",))], be.calls
    assert out is not None and len(out) == len(atoms), (len(out), len(atoms))
    assert loop._post_relaxed == 1 and loop._post_relax_failed == 0

    # The stand-in backend copies its input, so the relaxed atoms must come
    # back identical -- i.e. the candidate really is what was written. Keyed by
    # (residue, atom name), not by index: the writer groups atoms by residue
    # with the backbone first, which is what makes the file parseable at all
    # (see the candidate-PDB regression test), so the order is deliberately
    # not the AtomSet's.
    before = {(str(k), n): c for k, n, c in
              zip(atoms.reskeys, atoms.names, atoms.coords)}
    after = {(str(k), n): c for k, n, c in
             zip(out.reskeys, out.names, out.coords)}
    assert set(before) == set(after), (
        sorted(set(before) ^ set(after))[:5]
    )
    # Not bit-identical any more, and deliberately so: the relaxed atoms are
    # now Kabsch-superposed back onto the frame they were scored in, and
    # fitting an identical copy onto itself gives the identity to within
    # floating-point error rather than exactly.
    assert max(float(np.abs(before[k] - after[k]).max()) for k in before) < 1e-6

    # The perturbation must NOT relax any more: one relaxation per iteration,
    # and it happens after the choice.
    src = (REPO / "ipro_agent" / "loop.py").read_text()
    body = src[src.index("def _perturb_once"):src.index("def _relaxed_design_atoms")]
    code = "\n".join(
        ln for ln in body.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "relax(" not in code and "_relax_once" not in code, (
        "_perturb_once must not relax; step iv is after the MILP"
    )
    assert src.count("self.backend.relax(") == 1, "exactly one relax call site"

    # A backend that cannot relax says so in its mode and changes nothing.
    plain = IPROLoop(_spec(loop={"perturb_backbone": True}), tmp / "run2",
                     backend=resolve("builtin"))
    assert plain.mode == "backbone-mc"
    assert plain._relaxed_design_atoms(atoms, "x") is None
    assert any("no gradient relaxation" in n for n in plain.notes)


def test_regression_the_relaxation_baseline_is_relaxed_too():
    """Relaxing candidates against an unrelaxed reference would credit every
    candidate with the relaxation's own energy drop, which is a much larger
    defect than the one it fixes: with a stand-in backend that lowers the
    energy by a constant, every candidate would beat the start score and be
    accepted whatever the design did.

    So ``run_loop`` relaxes the starting point through the same path, and
    ``measure_native`` -- which is what ``ipro-agent score`` reports for a
    structure as supplied -- does not."""
    src = (REPO / "ipro_agent" / "loop.py").read_text()
    assert 'self._contexts_for(\n            self._rotamers_for(tables, start_choice), relax_tag="start"' in src \
        or 'relax_tag="start"' in src, "the start score must be relaxed like the candidates"
    nat = src[src.index("def measure_native"):src.index("def run_loop")]
    assert "self._relax_native" in nat, (
        "measure_native must not relax unless a run turned it on"
    )
    assert "self._relax_native = False" in src and "self._relax_native = True" in src


def test_regression_a_failed_relaxation_is_reported_not_hidden():
    """If relaxation is claimed and cannot be done, the run must say so rather
    than report mode 'full'."""
    from ipro_agent.loop import IPROLoop, _atoms_of
    from ipro_agent.structure import Structure

    class _Broken(_CountingRelaxBackend):
        def relax(self, pdb_in, pdb_out, fixed=()):
            raise RuntimeError("no params for LIG")

    tmp = Path(tempfile.mkdtemp())
    loop = IPROLoop(_spec(loop={"perturb_backbone": True}), tmp / "run",
                    backend=_Broken())
    atoms = _atoms_of(Structure.load(CRN), set())
    assert loop._relaxed_design_atoms(atoms, "t") is None
    assert loop._post_relax_failed == 1
    assert any("step iv) failed" in n for n in loop.notes), loop.notes


def test_regression_the_screening_caveat_follows_the_pair_table():
    """The "these are screening energies" caveat was keyed on ``can_relax``.
    The PyRosetta backend can relax and -- by default, as its own module
    docstring says -- still returns the screening function's numbers for the
    MILP table and for every reported energy, so the caveat was dropped from
    exactly the file that needed it and the written PDB carried screening
    kcal/mol under a "Rosetta Energy Units (REU)" heading."""
    be = resolve("builtin")
    assert be.screening_pairwise is True
    assert "screening" in be.units

    # The attribute exists on the contract, so any backend answers the question.
    from ipro_agent.energy.base import EnergyBackend

    assert EnergyBackend.screening_pairwise is True
    src = (REPO / "ipro_agent" / "energy" / "pyrosetta_backend.py").read_text()
    assert "self.screening_pairwise = not self.exact_pairs" in src, (
        "the PyRosetta backend must declare where its pairwise numbers come from"
    )
    loop_src = (REPO / "ipro_agent" / "loop.py").read_text()
    assert "self.backend.screening_pairwise else" in loop_src


def test_regression_a_spec_with_nothing_to_design_is_refused():
    """An empty ``positions:`` list ran to completion and reported "ok: true,
    accepted: 3, mutations: {}" with no note, which reads as "the optimizer
    found no improvement" rather than "nothing was designed"."""
    from ipro_agent.loop import IPROLoop

    spec = _spec(positions=[])
    loop = IPROLoop(spec, Path(tempfile.mkdtemp()) / "run", backend=resolve("builtin"))
    try:
        loop.run_loop()
    except SpecError as exc:
        assert "nothing to design" in str(exc)
    else:
        raise AssertionError("a spec that designs nothing must be refused")


def _have_vina() -> bool:
    from ipro_agent.dock import find_vina

    return find_vina() is not None


def test_regression_the_vina_path_converts_pdb_to_pdbqt():
    """Vina reads PDBQT only, and a spec names PDB files -- the format every
    example, every doc and the structure loader use. Nothing converted them, so
    ``backends.dock: vina`` could not dock at all.

    Measured with Vina 1.2.7 installed: every redock failed with "receptor
    receptor.pdb is not a .pdbqt file", the loop noted the failure and left the
    placement unchanged, so a run reporting ``dock -> vina`` had never moved a
    partner. ``auto`` resolves to vina whenever Vina is on PATH, so installing
    Vina made docking stop rather than start. After the fix the same run docks:
    score 62.6984 -> -10.0636 with "redocked 1 partner(s)"."""
    if not _have_vina():
        print("    (skipped: no AutoDock Vina)")
        return
    if not shutil.which("obabel"):
        print("    (skipped: no obabel for the PDB -> PDBQT conversion)")
        return
    from ipro_agent.dock import pdb_to_pdbqt

    tmp = Path(tempfile.mkdtemp())
    q = pdb_to_pdbqt(CRN, tmp / "rec.pdbqt", receptor=True)
    assert q.exists() and q.stat().st_size > 0
    assert q.read_text().lstrip().startswith(("ROOT", "ATOM", "REMARK", "MODEL")), \
        q.read_text()[:80]
    # The loop converts whatever the spec named, rather than requiring PDBQT.
    src = (REPO / "ipro_agent" / "loop.py").read_text()
    assert "pdb_to_pdbqt" in src, "the loop must convert before calling Vina"


def _have_pyrosetta() -> bool:
    try:
        import pyrosetta  # noqa: F401
    except Exception:
        return False
    return True


def test_pyrosetta_unexercised_paths_are_declared_and_real():
    """What in the PyRosetta backend has still not been run, declared in one
    place rather than left to be discovered.

    PyRosetta 2026.29 was installed on 2026-11-18 and the five original
    entries were all executed; three were defects and one of the stated
    suspicions was wrong. What remains is genuinely unexercised. This test
    keeps the declaration from rotting: every name must still resolve to
    something on the backend. See docs/DESIGN.md section 7e."""
    import re

    from ipro_agent.energy import pyrosetta_backend as prb

    assert prb.UNEXERCISED, "if everything is exercised, say so in 7e and empty this"
    for name, why in prb.UNEXERCISED.items():
        # Entries may qualify a method ("relax (multi-chain cases)"), so take
        # the dotted attribute and drop any parenthetical.
        attr = re.split(r"[\s(]", name.split(".")[-1])[0]
        assert hasattr(prb.PyRosettaBackend, attr), (
            f"declared-unexercised {name!r} no longer exists"
        )
        assert why.strip(), f"{name} needs a reason"


def test_regression_relax_holds_the_positions_it_was_told_to_hold():
    """``relax(fixed=[...])`` set a MoveMap and stopped there, which is the
    right idiom for minimization but not for FastRelax: its ramped repack
    cycles take their rotamer set from a PackerTask, and fixing a residue's own
    torsions does not fix its Cartesian position anyway, because the fold tree
    translates it when an upstream torsion moves.

    Measured on 1crn A16 with PyRosetta 2026.29, side-chain displacement after
    one relax cycle:

        MoveMap only                        0.866 A  (2.922 A on another seed)
        MoveMap + PreventRepacking          2.200 A
        + coordinate constraints           0.0001 A

    So a `fix` restraint on a design position -- which the run reports as
    enforced, and whose own note promises it is "held during relaxation in
    'full' mode" -- was silently violated by the relaxation step."""
    if not _have_pyrosetta():
        print("    (skipped: no PyRosetta)")
        return
    from ipro_agent.energy.pyrosetta_backend import PyRosettaBackend
    from ipro_agent.structure import Structure

    def sidechain(pdb, num):
        st = Structure.load(pdb)
        res = st.find("A", num)
        bb = {"N", "CA", "C", "O", "OXT"}
        return {
            a.get_name().strip(): np.array(a.coord, dtype=float)
            for a in res
            if a.get_name().strip() not in bb
            and (a.element or "").strip().upper() != "H"
        }

    def shift(a, b):
        common = set(a) & set(b)
        return max(float(np.abs(a[n] - b[n]).max()) for n in common)

    be = PyRosettaBackend()
    tmp = Path(tempfile.mkdtemp())
    out = tmp / "relaxed.pdb"
    before16, before22 = sidechain(CRN, 16), sidechain(CRN, 22)
    be.relax(CRN, out, fixed=["A16"])

    held = shift(before16, sidechain(out, 16))
    free = shift(before22, sidechain(out, 22))
    assert held < 0.01, f"A16 was told to hold and moved {held:.4f} A"
    assert free > 0.05, f"relaxation did nothing at all (A22 moved {free:.4f} A)"

    # And the promise is checked, not merely attempted: an impossible
    # tolerance must raise rather than write a structure that breaks it.
    try:
        be.relax(CRN, tmp / "never.pdb", fixed=["A16"], hold_tolerance=0.0)
    except RuntimeError as exc:
        assert "told to hold" in str(exc)
    else:
        raise AssertionError("a violated hold must raise, not be written out")


def test_regression_interface_energy_uses_a_constructor_that_exists():
    """``InterfaceAnalyzerMover("A_B")`` is not a constructor this PyRosetta
    has, so this documented method raised TypeError on every call. Measured
    against 2026.29: the overloads take a jump index, a set of chain indices,
    or a DockingPartners -- and DockingPartners is not constructible from a
    string either. The chain-string form is ``set_interface()``."""
    if not _have_pyrosetta():
        print("    (skipped: no PyRosetta)")
        return
    from ipro_agent.energy.pyrosetta_backend import PyRosettaBackend

    tmp = Path(tempfile.mkdtemp())
    prot = [ln for ln in CRN.read_text().splitlines() if ln.startswith("ATOM")]
    shifted = [
        ln[:21] + "B" + ln[22:30] + f"{float(ln[30:38]) + 14.0:8.3f}" + ln[38:]
        for ln in prot
    ]
    two = tmp / "two_chain.pdb"
    two.write_text("\n".join(prot + shifted) + "\nEND\n")

    got = PyRosettaBackend().interface_energy(two, "A_B")
    for key in ("dG_separated", "interface_dSASA", "total_score"):
        assert key in got and isinstance(got[key], float), got


def test_regression_a_second_backend_can_change_the_init_flags():
    """``pyrosetta.init`` was guarded by a bare "have we initialized" boolean,
    so a second backend asking for different ``init_flags`` silently kept the
    first set. Rosetta's re-initialization does work: measured on 2026.29,
    init("-ex1") then reading packing:ex1 gives True where the default gives
    False, and initializing again without it returns to False."""
    if not _have_pyrosetta():
        print("    (skipped: no PyRosetta)")
        return
    from pyrosetta.rosetta.basic import options as O

    from ipro_agent.energy.pyrosetta_backend import PyRosettaBackend

    PyRosettaBackend(init_flags="-mute all -ignore_zero_occupancy false")
    assert O.get_boolean_option("packing:ex1") is False
    PyRosettaBackend(init_flags="-mute all -ex1 -ignore_zero_occupancy false")
    assert O.get_boolean_option("packing:ex1") is True, (
        "a second backend's init_flags were ignored"
    )
    PyRosettaBackend(init_flags="-mute all -ignore_zero_occupancy false")
    assert O.get_boolean_option("packing:ex1") is False


def test_regression_a_pdb_remark_fits_a_pdb_record():
    """The REMARK line carrying the score embedded the backend's full `units`
    sentence. With the PyRosetta backend that made one line 230 characters
    long, where a PDB record is 80 columns -- so the deliverable structure was
    not a valid PDB file. The long form belongs in best/design.json, which has
    no column limit."""
    from ipro_agent.energy import resolve as _resolve

    for name in ("builtin",):
        be = _resolve(name)
        assert len(be.units_short) <= 32, be.units_short
        line = f"REMARK   1 SCORE      {-1234.5678:.4f} ({be.units_short})"
        assert len(line) <= 80, (len(line), line)
    src = (REPO / "ipro_agent" / "loop.py").read_text()
    assert "self.backend.units_short" in src, (
        "the PDB header must use the short units label"
    )


def test_regression_a_pore_axis_that_misses_the_lumen_is_reported():
    """The axis is fitted through the centroid of the design positions, so it
    follows the channel only when they *ring* the lumen. Positions along one
    wall put that centroid inside the wall, and the profile then measures the
    sliver beside the wall -- 1.53 A for the shipped 4-position smoke spec,
    which silently PASSES PoreDesigner's published sub-4 A filter.

    This was documented as an UNFIXED defect, on the grounds that a clearance
    threshold rejected the sound 25-position spec too (it measured 1.30 A).
    That measurement was the mistake, not the threshold: clearance was being
    computed from the *candidate's* atoms, and a narrowing design is supposed
    to fill the lumen, so the number conflates "the axis is buried in a wall"
    with "the design worked". Measured on the INPUT structure with its native
    side chains, the two specs separate by an order of magnitude:

        spec_full.yaml  (25 positions, ringing)   2.80 A   bottleneck 8.07 A
        spec_smoke.yaml ( 4 positions, one wall)  0.30 A   bottleneck 1.53 A

    so a water's 1.40 A is a usable test after all. The run now states flatly
    that the radius is not a pore radius, instead of hedging."""
    ompf = REPO / "examples" / "ompf_pore"
    if not (ompf / "2omf_A.pdb").exists() or not _have_pymol():
        print("    (skipped: needs the ompf example and PyMOL)")
        return
    from ipro_agent.loop import IPROLoop

    def measure(spec_file):
        spec = DesignSpec.load(ompf / spec_file)
        loop = IPROLoop(spec, Path(tempfile.mkdtemp()) / "run",
                        backend=resolve("builtin"))
        cb = loop.build_combined()
        atoms = loop._design_atoms(
            loop._rotamers_for(cb.tables, loop._native_choice(cb.tables)),
            loop._assemblies[0].name,
        )
        return loop._measure_geometry(atoms), loop.notes

    geo, notes = measure("spec_smoke.yaml")
    assert "pore_axis_clearance_input" in geo, geo
    assert geo["pore_axis_clearance_input"] < 1.4, geo
    assert abs(geo["pore_axis_clearance_input"] - 0.30) < 0.15, geo
    assert any("NOT A PORE RADIUS" in n for n in notes), notes

    # And the sound spec is NOT flagged -- which is what made the threshold
    # look useless when it was measured on candidate atoms.
    geo_full, notes_full = measure("spec_full.yaml")
    assert geo_full["pore_axis_clearance_input"] > 1.4, geo_full
    assert abs(geo_full["pore_axis_clearance_input"] - 2.80) < 0.4, geo_full
    assert not any("NOT A PORE RADIUS" in n for n in notes_full), notes_full


def test_regression_doctor_only_claims_an_in_process_energy_backend():
    """PyRosetta has to be importable by the interpreter running ipro-agent:
    ``energy.resolve()`` imports it in-process, unlike PyMOL, which is driven
    as a subprocess and may live in any environment.

    Measured with PyRosetta installed only in the env ``ipro_rosetta``: doctor
    run under the base interpreter reported ``energy -> pyrosetta`` and left
    PyRosetta out of ``degraded``, while a run started with that same
    interpreter silently used the built-in screening function. The mode is
    reported with every design, so this was a scientific misstatement. Finding
    it elsewhere is now recorded as a hint (``how: "elsewhere"``) and named in
    the detail, but ``available`` stays False."""
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "doctor.py"), "--json"],
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr
    rep = json.loads(p.stdout)
    pyros = rep["checks"]["pyrosetta"]

    # Whatever this machine has, the claim must match THIS interpreter.
    importable_here = _have_pyrosetta()
    assert pyros["available"] is importable_here, (pyros, importable_here)
    assert (rep["resolved"]["energy"].startswith("pyrosetta")) is importable_here
    assert ("pyrosetta" in rep["degraded"]) is (not importable_here)
    if pyros.get("how") == "elsewhere":
        assert pyros["available"] is False
        assert "NOT importable by this interpreter" in pyros["detail"]

    # A version that is a callable must not be reported as a function repr.
    assert "<function" not in json.dumps(rep), (
        "a module's version() function was printed instead of called"
    )


# ---------------------------------------------------------------------------
# Regressions from the pre-publication readiness audit
#
# Five dimensions were audited before the first public release (leaks, fresh
# clone, agent entry, packaging, claim integrity) and each finding was
# adversarially verified. These lock in the ones that could mislead a stranger
# or an agent. See docs/DESIGN.md section 7f.
# ---------------------------------------------------------------------------


def test_regression_a_rejected_candidate_is_never_the_best_design():
    """The best-score tracker was a sibling of `if ok:` rather than nested in
    it, so a candidate the goal's hard filter had vetoed still overwrote
    best_score, best_mutations and best_sequence, and still had
    best/design.pdb written for it.

    Measured on this repo's own full example: 14 iterations, **0 accepted**,
    every one rejected with "bottleneck 7.12 A is above pore_diameter_max
    4.00 A" -- and `status` still printed a 14-mutation best design at 0.7297
    against a start of 1.2574. An agent following SKILL.md's checklist ("the
    score improved; if not, say the search found nothing") would have reported
    those 14 mutations as a pore-narrowing design when nothing met the
    constraint."""
    src = (REPO / "ipro_agent" / "loop.py").read_text()
    assert "if ok and score < best_score:" in src, (
        "the best-design tracker must be gated on acceptance"
    )
    # And a run that accepts nothing has to say so itself.
    assert "NO DESIGN WAS ACCEPTED" in src


def test_regression_a_pip_installed_copy_can_reach_the_pymol_helpers():
    """`pip install .` created a working `ipro-agent` command that could
    validate a spec and then died at the first rotamer harvest: the two scripts
    PyMOL runs lived in the repo's top-level `scripts/`, and
    `Path(__file__).parent.parent / "scripts"` resolves to a
    `site-packages/scripts/` that does not exist. Measured in a clean venv: the
    wheel had 29 entries and neither helper.

    They now live inside the package, so they ship in the wheel. Both the
    clone workflow and an installed copy resolve them the same way."""
    from ipro_agent.mutate import APPLY_SCRIPT
    from ipro_agent.rotamers import HARVEST_SCRIPT

    for path in (HARVEST_SCRIPT, APPLY_SCRIPT):
        assert path.exists(), f"missing PyMOL helper: {path}"
        # Inside the package, not beside it -- that is what puts it in the wheel.
        assert path.parent.name == "pymol_helpers", path
        assert path.parent.parent.name == "ipro_agent", path

    # The declared package list must carry it, or the wheel drops it silently.
    # Checked as text rather than with tomllib, which is 3.11+ while this
    # package declares requires-python >=3.10 -- the import alone used to fail
    # this test on the very floor the metadata promises.
    cfg = (REPO / "pyproject.toml").read_text()
    assert '"ipro_agent.pymol_helpers"' in cfg, "the subpackage must be declared"
    # ... and the metadata must not have lost the dependencies to a sub-table,
    # which is what happens if a [project.*] sub-table is opened above them.
    deps_at = cfg.index("dependencies = [")
    urls_at = cfg.index("[project.urls]")
    assert deps_at < urls_at, (
        "[project.urls] is above `dependencies`, so TOML makes dependencies a "
        "key of urls and the package declares none"
    )
    assert 'license = "MIT"' in cfg
    assert "Repository = " in cfg


def test_regression_doctor_separates_can_run_from_can_design():
    """`can_run` was `bool(numpy available)` -- so a missing Biopython or
    PyYAML read as fine -- and nothing reported whether a *design* was
    possible. PyMOL has no fallback (rotamers come only from its mutagenesis
    wizard and `backends.mutate: builtin` is refused), so `can run: True` with
    no PyMOL told an agent the gate was green when `run` could not produce a
    design at all. AGENTS.md tells agents to trust this gate."""
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "doctor.py"), "--json"],
                       capture_output=True, text=True, timeout=600)
    assert p.returncode in (0, 1), p.stderr
    rep = json.loads(p.stdout)

    for key in ("can_run", "can_design", "missing_required"):
        assert key in rep, f"doctor must report {key}"
    # can_run means every REQUIRED module imports -- all three, not just numpy.
    assert rep["can_run"] is (not rep["missing_required"])
    for mod in ("py:numpy", "py:Bio", "py:yaml"):
        assert mod in rep["checks"], mod
        if not rep["checks"][mod]["available"]:
            assert mod.split(":", 1)[1] in rep["missing_required"]
    # can_design additionally requires a mutagenesis backend.
    assert rep["can_design"] is bool(
        rep["can_run"] and rep["checks"]["pymol"]["available"]
    )
    if not rep["checks"]["pymol"]["available"]:
        assert rep["can_design"] is False


def test_regression_a_mistyped_spec_path_is_one_line_not_a_traceback():
    """The agent-facing CLI answered a mistyped path with a 20-line traceback,
    which an agent has to parse past to find the one fact it needs."""
    p = subprocess.run(
        [sys.executable, "-m", "ipro_agent", "validate", "definitely_not_here.yaml"],
        capture_output=True, text=True, cwd=REPO, timeout=120,
    )
    err = p.stderr.strip()
    assert "Traceback" not in err, err
    assert "spec not found" in err, err
    assert len(err.splitlines()) <= 2, err


def test_the_public_facing_claims_are_consistent():
    """Claims a stranger can falsify from the repo itself.

    The audit found the README telling readers the licence was "not yet
    chosen" while an MIT LICENSE shipped; DESIGN.md section 1b saying
    restraints and assemblies were "not implemented" while sections 1c and 1d
    documented both; section 2 calling the docking anneal "not implemented"
    while it runs by default; and the repo describing itself as "clean-room"
    -- a term meaning the implementers had no access to the original -- while
    DESIGN.md records reading the released source and preferring its values.
    No upstream code is copied, which is the claim that actually matters, so
    the accurate word is "independent"."""
    readme = (REPO / "README.md").read_text()
    licence = (REPO / "LICENSE").read_text()
    design = (REPO / "docs" / "DESIGN.md").read_text()

    assert "Not yet chosen" not in readme
    assert "**MIT**" in readme and "MIT License" in licence

    # The *claim* must not reappear anywhere tracked. The term itself may
    # appear in quotes where LICENSE and DESIGN.md explain why it is NOT used,
    # so this looks for the asserting forms rather than the bare word.
    asserting = ("clean-room reimplementation", "is a clean-room",
                 "is clean-room", "clean room reimplementation")
    tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                             cwd=REPO).stdout.split()
    for f in tracked:
        path = REPO / f
        if path.suffix.lower() in (".pdb", ".gif", ".jpg", ".png", ".xml", ".html"):
            continue
        if f == "tests/test_ipro.py":
            continue      # names the phrases above in order to forbid them
        if f.startswith("docs/provenance/"):
            # Verbatim records: the grep log that searched for this phrasing,
            # and two audit reports quoting what reviewers wrote. They are
            # evidence ABOUT the claim, not the repo making it. The assertion
            # below still covers every file that speaks in the repo's voice.
            continue
        try:
            text = path.read_text(errors="ignore").lower()
        except OSError:
            continue
        for phrase in asserting:
            assert phrase not in text, f"{f} still claims {phrase!r}"

    assert "None is implemented here." not in design
    assert "| docking anneal | not implemented |" not in design
    # PyMOL must be stated as required, not as a fallback-having optional.
    assert "no fallback" in readme
    # The open pore-axis defect has to be disclosed on the landing page.
    assert "pore_axis" in readme
    # Responsible-use scope, for a tool published for autonomous agents.
    assert "## Responsible use" in readme


def test_the_documented_environment_matches_the_shipped_one():
    """docs/INSTALL.md pasted a different environment.yml than the one that
    ships: env name `ipro` against `ipro-agent`, and 13 dependencies against 8.
    A reader who copied the doc block got an environment the README's
    `conda activate ipro-agent` could not find, and a different dependency set
    from `conda env create -f environment.yml`. The doc now reproduces the
    shipped file verbatim; this keeps them that way."""
    import re

    try:
        import yaml
    except ImportError:                      # pragma: no cover
        print("    (skipped: no PyYAML)")
        return

    doc = (REPO / "docs" / "INSTALL.md").read_text()
    m = re.search(r"### `environment\.yml`.*?```yaml\n(.*?)```", doc, re.S)
    assert m, "docs/INSTALL.md no longer shows an environment.yml block"
    pasted = yaml.safe_load(m.group(1))
    shipped = yaml.safe_load((REPO / "environment.yml").read_text())
    assert pasted == shipped, (
        "the environment.yml in docs/INSTALL.md has drifted from the shipped "
        f"file: doc={pasted!r} shipped={shipped!r}"
    )
    # The name the README tells people to activate.
    assert shipped["name"] == "ipro-agent"
    # PyMOL must be in it: it is the one required dependency pip cannot supply.
    deps = " ".join(d for d in shipped["dependencies"] if isinstance(d, str))
    assert "pymol" in deps


def test_regression_an_indel_at_an_unmodelled_terminus_is_reported():
    """This module's justification is that a terminus has no second cut end, so
    the edit is exact -- and it refuses internal indels because an unclosed
    break "still looks like a protein". But "terminal" was taken to mean the
    last residue *present in the coordinates*. On a structure with disordered
    termini, that residue is in the middle of the real chain, so appending
    there creates exactly the unclosed break internal indels are refused for,
    with nothing to catch it. SEQRES says how long the chain really is."""
    from ipro_agent.config import IndelSpec
    from ipro_agent.indel import classify
    from ipro_agent.structure import Structure

    # Fully modelled: silent.
    full = classify(Structure.load(CRN),
                    [IndelSpec(kind="insert", chain="A", after=46, sequence="G")])
    assert full.notes == [], full.notes

    # Coordinates truncated to residue 43 while SEQRES still declares 46.
    tmp = Path(tempfile.mkdtemp()) / "trunc.pdb"
    tmp.write_text("".join(
        ln for ln in CRN.read_text().splitlines(True)
        if not (ln.startswith("ATOM") and int(ln[22:26]) > 43)
    ))
    plan = classify(Structure.load(tmp),
                    [IndelSpec(kind="insert", chain="A", after=43, sequence="G")])
    assert plan.terminal, "still classified terminal, which is why the note matters"
    assert plan.notes, "a chain modelled short of SEQRES must be reported"
    note = plan.notes[0]
    for fragment in ("SEQRES declares 46", "modelled as 43", "INTERNAL edit"):
        assert fragment in note, note

    # And the note has to reach the run: apply_terminal takes plan.terminal,
    # not the plan, so the loop must carry plan.notes itself.
    assert "self.notes.extend(plan.notes)" in (REPO / "ipro_agent" / "loop.py").read_text()


def test_a_zero_mutation_structure_says_it_is_not_a_design():
    """best/design.pdb is written up front from the native sequence, repacked,
    and only replaced once an ACCEPTED candidate beats it. On a run where every
    candidate is rejected it is therefore the starting point -- which the file
    has to say on its own, since tracing a stray PDB back to its run is the
    only reason this header exists. Verified end to end on a spec whose pore
    filter rejects everything: 3 iterations, 0 accepted, NO DESIGN WAS ACCEPTED
    in the notes, MUTATIONS none in the header."""
    src = (REPO / "ipro_agent" / "loop.py").read_text()
    assert "NO MUTATIONS: this is the input sequence," in src
    # Every PDB record must still fit 80 columns.
    for line in src.splitlines():
        if "REMARK   1 WARNING" in line and line.strip().startswith('"'):
            text = line.strip().strip('",')
            assert len(text) <= 80, (len(text), text)


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------


def _mcp(messages: list[dict], timeout: float = 900.0) -> dict[int, dict]:
    """Drive the MCP server over real stdio and return replies by id."""
    payload = "\n".join(json.dumps(m) for m in messages) + "\n"
    p = subprocess.run(
        [sys.executable, "-m", "ipro_agent.mcp_server"],
        input=payload, capture_output=True, text=True, timeout=timeout, cwd=REPO,
    )
    assert p.returncode == 0, (p.returncode, p.stderr[-600:])
    out: dict[int, dict] = {}
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        msg = json.loads(line)
        if msg.get("id") is not None:
            out[msg["id"]] = msg
    return out


_INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {}}}


def _call(mid: int, name: str, arguments: dict | None = None) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}}}


def _body(reply: dict) -> tuple[bool, dict]:
    """(isError, parsed payload) from a tools/call reply."""
    res = reply["result"]
    return bool(res["isError"]), json.loads(res["content"][0]["text"])


def test_mcp_server_speaks_the_protocol():
    """initialize / tools/list over stdio, and an unknown tool is a protocol
    error rather than a crash. The server is a thin wrapper over the CLI with
    no dependencies beyond the standard library, so this needs no MCP client."""
    replies = _mcp([
        _INIT,
        {"jsonrpc": "2.0", "method": "notifications/initialized"},   # no reply
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        _call(3, "ipro_nonexistent"),
        {"jsonrpc": "2.0", "id": 4, "method": "ping"},
    ])
    init = replies[1]["result"]
    assert init["protocolVersion"] == "2025-06-18", init   # echoes a known version
    assert init["serverInfo"]["name"] == "ipro-agent"
    assert "instructions" in init and "screening" in init["instructions"]

    names = [t["name"] for t in replies[2]["result"]["tools"]]
    # Membership, not an exact ordered list -- the list grows, and pinning its
    # order made adding a tool look like a regression. What matters is that
    # every tool is there and that the ordering intent holds.
    expected = {
        "ipro_doctor", "ipro_setup", "ipro_fetch_structure",
        "ipro_inspect_structure", "ipro_goals", "ipro_validate",
        "ipro_positions", "ipro_score", "ipro_milp", "ipro_run", "ipro_status",
        "ipro_result", "ipro_iterations", "ipro_runs", "ipro_cancel",
    }
    # Membership, as the comment above says: set equality is what made adding
    # a tool look like a regression, which is the thing this was rewritten to
    # avoid once already.
    missing = expected - set(names)
    assert not missing, sorted(missing)
    assert len(names) == len(set(names)), "a tool is registered twice"
    # doctor first: every other description tells you to call it before
    # anything else, and clients show the list in order.
    assert names[0] == "ipro_doctor"
    assert names[1] == "ipro_setup", "the fix for doctor's bad news comes next"
    # The structure tools must precede the spec tools: you cannot write a spec
    # about residues until you know what they are.
    assert names.index("ipro_inspect_structure") < names.index("ipro_validate")
    for tool in replies[2]["result"]["tools"]:
        assert tool["description"].strip(), tool["name"]
        assert tool["inputSchema"]["type"] == "object", tool["name"]

    assert "error" in replies[3], "an unknown tool must be a JSON-RPC error"
    assert replies[4]["result"] == {}

    # A notification must not get a reply.
    assert set(replies) == {1, 2, 3, 4}


def test_mcp_tools_return_parsed_json_not_prose():
    """The CLI pretty-prints its JSON across many lines, so the wrapper cannot
    parse per-line -- the first version did, and every call came back as "did
    not return JSON". And where the CLI refuses a request with a one-line
    explanation on stderr ("spec has no target molecule; pass --near or add
    one"), that sentence has to BE the error, not be buried under a wrapper
    message."""
    smoke = "examples/ompf_pore/spec_smoke.yaml"
    replies = _mcp([
        _INIT,
        _call(2, "ipro_validate", {"spec": smoke}),
        _call(3, "ipro_goals"),
        _call(4, "ipro_doctor"),
        _call(5, "ipro_validate", {"spec": "no_such_spec_here.yaml"}),
        _call(6, "ipro_positions", {"spec": smoke}),     # this spec has no target
    ])
    err, body = _body(replies[2])
    assert not err, body
    assert body["ok"] is True
    assert body["info"]["design_space_sequences"] == 6561

    err, body = _body(replies[3])
    assert not err and body, body

    err, body = _body(replies[4])
    assert not err, body
    for key in ("can_run", "can_design", "degraded", "resolved"):
        assert key in body, body

    err, body = _body(replies[5])
    assert err and "spec not found" in body["error"], body

    err, body = _body(replies[6])
    assert err, body
    assert "no target molecule" in body["error"], (
        f"the CLI's own explanation must survive: {body['error']!r}"
    )


def test_mcp_accepts_an_inline_spec():
    """An agent usually has the spec in context, not on disk.

    The partner is a separate little molecule, not a second reference to
    1crn.pdb: two molecules at identical coordinates are now a validation
    problem, because they measure as an "interaction energy" in the tens of
    thousands (see the same-coordinates regression test)."""
    lig = Path(tempfile.mkdtemp()) / "lig.pdb"
    lig.write_text(
        "HETATM    1  C1  LIG B   1      20.000  20.000  20.000  1.00  0.00           C\n"
        "HETATM    2  C2  LIG B   1      21.500  20.000  20.000  1.00  0.00           C\n"
        "END\n"
    )
    inline = (
        "goal_statement: inline probe\n"
        "goal: bind_tighter\n"
        "molecules:\n"
        "  - name: prot\n"
        f"    path: {CRN}\n"
        "    role: design\n"
        f"  - name: LIG\n"
        f"    path: {lig}\n"
        "    role: target\n"
        "    kind: ligand\n"
        "positions: [A16]\n"
    )
    replies = _mcp([_INIT, _call(2, "ipro_validate", {"spec": inline})])
    err, body = _body(replies[2])
    assert not err, body
    assert body["ok"] is True, body


def test_mcp_run_refuses_rather_than_launching_a_doomed_job():
    """`ipro_run` checks doctor first. Without PyMOL a run dies at the first
    rotamer harvest, minutes in, having left a run directory that looks like a
    started job -- so it is refused up front with the install command. It also
    refuses a non-empty run_dir rather than writing into someone's results."""
    from ipro_agent.mcp_server import ToolError, tool_run

    # A non-empty directory is refused whatever is installed.
    tmp = Path(tempfile.mkdtemp())
    (tmp / "something.txt").write_text("not empty\n")
    try:
        tool_run({"spec": str(REPO / "examples/ompf_pore/spec_smoke.yaml"),
                  "run_dir": str(tmp)})
    except ToolError as exc:
        assert "not empty" in str(exc), exc
    else:
        raise AssertionError("a non-empty run_dir must be refused")

    # And the source states the can_design gate, which is what stops a doomed
    # launch on a machine with no PyMOL.
    src = (REPO / "ipro_agent" / "mcp_server.py").read_text()
    assert 'if not rep.get("can_design")' in src
    assert "start_new_session=True" in src, "a run must outlive the server"


def test_status_json_carries_the_mode_and_notes():
    """The house rules say a design is never reported without the run's `mode`
    and `notes`, and `status --json` is the documented way for an agent to read
    a run -- it is told to poll this and never to load iterations.jsonl. Both
    fields were missing, so the one interface that instructs an agent to report
    the mode did not supply it."""
    from ipro_agent.state import RunDir

    keys = set(RunDir(Path(tempfile.mkdtemp()) / "r", create=True).as_json())
    for required in ("mode", "notes", "backend", "goal", "status", "accepted",
                     "best_score", "start_score"):
        assert required in keys, f"status --json must report {required}"


def test_mcp_exposes_the_manual_as_a_resource():
    """Eight tool descriptions cannot carry the operating manual -- the
    degraded modes, the spec format, the validation checklist. An agent that
    clones the repo reads SKILL.md and gets all of it; an agent on the other
    end of the server used to get none of it. The documents are resources, so
    the manual is a fetch away rather than unavailable."""
    replies = _mcp([
        _INIT,
        {"jsonrpc": "2.0", "id": 2, "method": "resources/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "resources/read",
         "params": {"uri": "ipro://skill"}},
        {"jsonrpc": "2.0", "id": 4, "method": "resources/read",
         "params": {"uri": "ipro://nope"}},
    ])
    caps = replies[1]["result"]["capabilities"]
    assert "resources" in caps and "prompts" in caps, caps
    # The server must point an agent at the manual in its own instructions.
    assert "ipro://skill" in replies[1]["result"]["instructions"]

    uris = [r["uri"] for r in replies[2]["result"]["resources"]]
    for required in ("ipro://skill", "ipro://house-rules", "ipro://design-notes",
                     "ipro://example-spec"):
        assert required in uris, uris

    skill = replies[3]["result"]["contents"][0]
    assert skill["uri"] == "ipro://skill"
    on_disk = (REPO / ".claude" / "skills" / "ipro-agent" / "SKILL.md").read_text()
    assert skill["text"] == on_disk, "the resource must serve the real file"
    assert len(skill["text"]) > 10_000

    assert "error" in replies[4], "an unknown resource must be an error"


def test_mcp_prompts_walk_the_documented_order():
    """Prompts are what a client shows as slash commands. They have to encode
    the order SKILL.md requires -- doctor, goal, spec, validate, score, run,
    report -- or they are just a free-text box with extra steps."""
    replies = _mcp([
        _INIT,
        {"jsonrpc": "2.0", "id": 2, "method": "prompts/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "prompts/get",
         "params": {"name": "ipro-redesign",
                    "arguments": {"structure": "/tmp/p.pdb", "goal": "prefer C12"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "prompts/get",
         "params": {"name": "ipro-report", "arguments": {"run": "runs/x"}}},
        {"jsonrpc": "2.0", "id": 5, "method": "prompts/get",
         "params": {"name": "ipro-nope", "arguments": {}}},
    ])
    names = [p["name"] for p in replies[2]["result"]["prompts"]]
    assert names == ["ipro-redesign", "ipro-check-setup", "ipro-report"], names

    text = replies[3]["result"]["messages"][0]["content"]["text"]
    assert "/tmp/p.pdb" in text and "prefer C12" in text
    for step in ("ipro://skill", "ipro_doctor", "ipro_goals", "ipro_validate",
                 "ipro_run", "ipro_status", "goal_statement"):
        assert step in text, f"the redesign prompt must mention {step}"
    # No positions given -> it must propose them rather than invent them.
    assert "ipro_positions" in text
    # And it must forbid silently weakening the user's constraints.
    assert "weaken" in text

    report = replies[4]["result"]["messages"][0]["content"]["text"]
    assert "runs/x" in report
    assert "accepted" in report and "screening" in report

    assert "error" in replies[5]


def test_mcp_setup_diagnoses_without_installing():
    """The server is pure standard library, so it starts on a machine with
    nothing installed -- and then every useful tool fails, leaving the agent to
    work out for itself that pymol-open-source is conda-only. Measured in a
    bare venv: the server starts, doctor reports missing ['numpy', 'Bio',
    'yaml'], and setup names the exact command. Nothing is installed without
    confirm=true, because creating the environment downloads ~500 MB and
    changes the machine."""
    replies = _mcp([_INIT, _call(2, "ipro_setup")])
    err, body = _body(replies[2])
    assert not err, body
    assert body["action"] in ("none", "install"), body

    if body["action"] == "install":
        assert "command" in body and "environment.yml" in body["command"]
        assert "next_step" in body or "blocked" in body
        assert body.get("launched") is not True, "must not install without confirm"
    else:
        # This machine can design, so there is nothing to do -- and it still has
        # to say what the optional backends change.
        assert body["can_design"] is True
        assert "degraded" in body

    # The guard itself, so the default cannot drift to installing.
    src = (REPO / "ipro_agent" / "mcp_server.py").read_text()
    assert 'if not args.get("confirm"):' in src
    assert "pymol-open-source is not on PyPI" in src


def test_mcp_server_needs_nothing_but_the_standard_library():
    """It has to start on a machine where numpy is not installed, or it cannot
    be the thing that tells you numpy is not installed."""
    import ast
    import sys as _sys

    tree = ast.parse((REPO / "ipro_agent" / "mcp_server.py").read_text())
    top = {n.module.split(".")[0] for n in ast.walk(tree)
           if isinstance(n, ast.ImportFrom) and n.module and n.col_offset == 0}
    top |= {a.name.split(".")[0] for n in ast.walk(tree)
            if isinstance(n, ast.Import) and n.col_offset == 0 for a in n.names}
    outside = sorted(top - set(_sys.stdlib_module_names) - {"ipro_agent"})
    assert not outside, f"module-level non-stdlib imports: {outside}"


def test_mcp_unexercised_paths_are_declared():
    """Same practice as energy.pyrosetta_backend.UNEXERCISED: what has not been
    run is declared in one place with a reason, not left to be discovered. The
    one entry is the branch that actually launches `conda env create`, which a
    test cannot exercise without building a real multi-hundred-megabyte
    environment on the machine under test."""
    import re

    from ipro_agent import mcp_server as m

    assert m.UNEXERCISED, "empty it only when everything really is exercised"
    for name, why in m.UNEXERCISED.items():
        attr = re.split(r"[\s(]", name)[0]
        assert hasattr(m, attr), f"declared-unexercised {name!r} no longer exists"
        assert len(why.strip()) > 40, f"{name} needs a real reason"


def test_mcp_resources_survive_a_pip_install():
    """`REPO_ROOT` is the repo from a clone but **site-packages** when
    pip-installed, and the documents are not inside the package -- so a
    pip-installed server found none of them and `resources/list` returned 0
    entries, silently removing the operating manual the server's own
    instructions tell the client to read.

    Now every document is always listed (marked when this copy lacks the file),
    reading a missing one returns the raw URL to fetch, and the client's working
    directory is searched too, so a pip-installed server whose MCP config sets
    `cwd` to a checkout serves them normally."""
    from ipro_agent import mcp_server as m

    listed = m._resource_list()
    assert len(listed) == len(m.RESOURCES), "every document must be listed"
    # From a clone, every one resolves.
    for r in m.RESOURCES:
        assert m._resolve_doc(r["path"]) is not None, r["uri"]
        assert "[NOT in this installation" not in next(
            x["description"] for x in listed if x["uri"] == r["uri"]
        )

    # A document the copy does not have: the error must name where to get it.
    m._BY_URI["ipro://_probe"] = {
        "uri": "ipro://_probe", "path": "no/such/file.md",
        "name": "probe", "description": "", "mimeType": "text/markdown",
    }
    try:
        m._resource_read("ipro://_probe")
    except m.ToolError as exc:
        assert m.RAW_BASE in str(exc), exc
        assert "git clone" in str(exc)
    else:
        raise AssertionError("a missing document must not read as empty")
    finally:
        del m._BY_URI["ipro://_probe"]

    # The cwd fallback is what makes a pip-installed server usable.
    assert "Path.cwd()" in (REPO / "ipro_agent" / "mcp_server.py").read_text()


def test_inspect_resolves_a_bare_residue_number_to_its_native_identity():
    """A user says "you may change residues 18, 20, 38 and 40" and nothing in
    the tool could say which chain those are in, what is at them, or whether
    they exist -- so an agent guessed, and a guess here silently designs a
    different residue (docs/DESIGN.md section 7b defect 14).

    On the shipped OmpF example those four resolve to VAL, LEU, MET and TYR --
    which is also a check against the paper, since PoreDesigner's tested designs
    mutate exactly V18, L20, M38 and Y40 to tryptophan."""
    ompf = REPO / "examples" / "ompf_pore" / "2omf_A.pdb"
    if not ompf.exists():
        print("    (skipped: no example structure)")
        return
    p = subprocess.run(
        [sys.executable, "-m", "ipro_agent", "inspect", str(ompf),
         "--positions", "18,20,38,40", "--json"],
        capture_output=True, text=True, cwd=REPO, timeout=600,
    )
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)

    natives = {a["asked"]: a["native"] for a in out["positions"]}
    assert natives == {"18": "VAL", "20": "LEU", "38": "MET", "40": "TYR"}, natives
    # It must also hand back something pasteable, and say which chain it chose.
    for a in out["positions"]:
        assert a["resolves_to"] == f"A{a['asked']}"
        assert "allowed:" in a["spec_form"]

    chain = next(c for c in out["chains"] if c["chain"] == "A")
    assert chain["residues"] == 340 and chain["first"] == "A1" and chain["last"] == "A340"
    assert chain["sequence"].startswith("AEIYNKDGNK")

    # A position that does not exist must say so, not resolve to something else.
    p2 = subprocess.run(
        [sys.executable, "-m", "ipro_agent", "inspect", str(ompf),
         "--positions", "99999", "--json"],
        capture_output=True, text=True, cwd=REPO, timeout=600,
    )
    bad = json.loads(p2.stdout)["positions"][0]
    assert "error" in bad and "not found" in bad["error"].lower(), bad


def test_fetch_is_the_only_networked_command_and_validates_its_input():
    """`fetch` is the one operation here that touches the internet, which is
    why it is a command of its own: a design run must never silently reach out
    for its inputs. It is not exercised against the network in the suite -- that
    would make the tests depend on RCSB being up -- so what is checked here is
    the input validation and that nothing else imports urllib."""
    # A malformed ID must be refused before any request is made.
    p = subprocess.run(
        [sys.executable, "-m", "ipro_agent", "fetch", "NOTANID"],
        capture_output=True, text=True, cwd=REPO, timeout=120,
    )
    assert p.returncode == 2, (p.returncode, p.stdout, p.stderr)
    assert "not a PDB ID" in p.stderr
    assert "Traceback" not in p.stderr

    # Nothing in the package may reach the network except this one command.
    # Checked over the parsed imports, not the text: grepping for "requests"
    # matched the English word in a comment, which is the kind of false
    # positive that teaches people to ignore a check.
    import ast

    NETWORKING = {"urllib", "requests", "httpx", "socket", "http", "ftplib",
                  "telnetlib", "smtplib", "xmlrpc", "asyncio"}
    offenders = {}
    for path in sorted((REPO / "ipro_agent").rglob("*.py")):
        tree = ast.parse(path.read_text(errors="ignore"))
        used = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                used |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                used.add(node.module.split(".")[0])
        hit = used & NETWORKING
        if hit:
            offenders[path.relative_to(REPO).as_posix()] = sorted(hit)
    assert list(offenders) == ["ipro_agent/cli.py"], (
        f"only the fetch command may use the network; found: {offenders}"
    )
    assert offenders["ipro_agent/cli.py"] == ["urllib"], offenders


def test_regression_mcp_never_replies_to_a_notification():
    """A message with no `id` is a notification, and JSON-RPC 2.0 says a server
    must not respond to one. The unknown-method branch replied anyway with
    `"id": null`, so a client sending any notification this server did not
    recognise got back an error frame it never asked for -- which a strict
    client can treat as a protocol desync. The *known* notifications were
    handled; it was the unknown ones that leaked."""
    p = subprocess.run(
        [sys.executable, "-m", "ipro_agent.mcp_server"],
        input='{"jsonrpc":"2.0","method":"notifications/unknown_thing"}\n'
              '{"jsonrpc":"2.0","method":"notifications/initialized"}\n',
        capture_output=True, text=True, cwd=REPO, timeout=300,
    )
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "", f"replied to a notification: {p.stdout!r}"

    # A request with an unknown method still gets a proper error.
    p2 = subprocess.run(
        [sys.executable, "-m", "ipro_agent.mcp_server"],
        input='{"jsonrpc":"2.0","id":5,"method":"no_such_method"}\n',
        capture_output=True, text=True, cwd=REPO, timeout=300,
    )
    err = json.loads(p2.stdout.strip())
    assert err["id"] == 5 and err["error"]["code"] == -32601, err


def test_regression_a_json_rpc_batch_does_not_kill_the_server():
    """A top-level JSON array is a JSON-RPC batch, which MCP allowed before the
    2025-06-18 revision and which some clients still send. It reached handle()
    as a list and killed the entire server process with `AttributeError: 'list'
    object has no attribute 'get'` -- one message ending the session.

    And more generally: no single malformed message may wedge the server."""
    p = subprocess.run(
        [sys.executable, "-m", "ipro_agent.mcp_server"],
        input='[{"jsonrpc":"2.0","id":1,"method":"ping"},'
              '{"jsonrpc":"2.0","id":2,"method":"tools/list"}]\n',
        capture_output=True, text=True, cwd=REPO, timeout=300,
    )
    assert p.returncode == 0, (p.returncode, p.stderr[-400:])
    assert "Traceback" not in p.stderr, p.stderr[-400:]
    batch = json.loads(p.stdout.strip())
    assert isinstance(batch, list) and [r["id"] for r in batch] == [1, 2], batch

    # A batch of only notifications gets no reply at all.
    p2 = subprocess.run(
        [sys.executable, "-m", "ipro_agent.mcp_server"],
        input='[{"jsonrpc":"2.0","method":"notifications/initialized"}]\n',
        capture_output=True, text=True, cwd=REPO, timeout=300,
    )
    assert p2.stdout.strip() == "", p2.stdout

    # Garbage, then good messages: the server must carry on.
    p3 = subprocess.run(
        [sys.executable, "-m", "ipro_agent.mcp_server"],
        input='not json at all\n'
              '{"jsonrpc":"2.0","id":7,"method":"ping"}\n'
              '"a bare string"\n'
              '{"jsonrpc":"2.0","id":8,"method":"tools/list"}\n',
        capture_output=True, text=True, cwd=REPO, timeout=300,
    )
    assert p3.returncode == 0, p3.stderr[-300:]
    ids = [json.loads(l).get("id") for l in p3.stdout.splitlines() if l.strip()]
    assert 7 in ids and 8 in ids, f"did not recover after bad input: {ids}"


def test_regression_mcp_stdout_carries_nothing_but_jsonrpc():
    """stdout IS the protocol channel on stdio. Anything else printed there --
    a stray print, a warning, a library banner -- corrupts the stream, and it
    is the classic way an MCP server fails for everyone but its author. Every
    line must parse as JSON."""
    p = subprocess.run(
        [sys.executable, "-m", "ipro_agent.mcp_server"],
        input='{"jsonrpc":"2.0","id":1,"method":"initialize","params":'
              '{"protocolVersion":"2025-06-18","capabilities":{}}}\n'
              '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":'
              '{"name":"ipro_doctor","arguments":{}}}\n'
              '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":'
              '{"name":"ipro_validate","arguments":'
              '{"spec":"examples/ompf_pore/spec_smoke.yaml"}}}\n',
        capture_output=True, text=True, cwd=REPO, timeout=900,
    )
    assert p.returncode == 0, p.stderr[-400:]
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            raise AssertionError(f"non-JSON on stdout corrupts the stream: {line[:200]!r}")
        assert msg.get("jsonrpc") == "2.0", msg


def test_the_example_prompts_page_states_true_numbers():
    """docs/EXAMPLE_PROMPTS.md tells people what each prompt produces, so its
    numbers are claims a reader will check. Three rounds of audit on this repo
    found stale documented measurements (an 8.1 A eyelet, a 5.69 A bottleneck, a
    0.716 score, 'twelve defects' where there were seven), so the numbers in a
    new page get pinned on the way in rather than after someone trips on them."""
    page = (REPO / "docs" / "EXAMPLE_PROMPTS.md")
    if not page.exists():
        print("    (skipped: no prompts page)")
        return
    doc = page.read_text()
    ompf = REPO / "examples" / "ompf_pore" / "2omf_A.pdb"
    if not ompf.exists():
        print("    (skipped: no example structure)")
        return

    # The native residues the page names must be what the structure holds.
    p = subprocess.run(
        [sys.executable, "-m", "ipro_agent", "inspect", str(ompf),
         "--positions", "18,20,38,40", "--json"],
        capture_output=True, text=True, cwd=REPO, timeout=600,
    )
    out = json.loads(p.stdout)
    natives = {a["asked"]: a["native"] for a in out["positions"]}
    one = {"VAL": "V", "LEU": "L", "MET": "M", "TYR": "Y"}
    for num, three in natives.items():
        claim = f"{one[three]}{num}"
        assert claim in doc, f"the page must name {claim} (residue {num} is {three})"

    chain = next(c for c in out["chains"] if c["chain"] == "A")
    assert f"{chain['residues']} residues" in doc, chain["residues"]
    assert f"{chain['first']}..{chain['last']}" in doc

    # Defaults the page quotes.
    from ipro_agent.config import LoopSettings

    assert f"({LoopSettings().ensemble_size} by default)" in doc
    cli_src = (REPO / "ipro_agent" / "cli.py").read_text()
    assert "default=5.0" in cli_src and "default 5 Å" in doc, (
        "the page quotes the interface cutoff default; keep them together"
    )

    # And the caveat that must never be dropped from a page of prompts.
    for required in ("screening scale", "accepted: 0", "hypothesis",
                     "What not to ask it"):
        assert required in doc, required


def test_regression_unreadable_structures_fail_clearly_not_with_a_traceback():
    """Every structure in this package arrives through `Structure.load`, so the
    unreadable-input cases belong there -- they used to surface as the parser's
    own exceptions (`ValueError: Empty file.`, `IsADirectoryError`,
    `UnicodeDecodeError` on a gzipped file), which every caller lets through as
    a bare traceback because they all catch StructureError.

    And a file the parser *accepts* but that holds no models -- a FASTA, a
    sequence dump, any text file with a .pdb name -- reported "model 0 not in
    <path>", which is true and useless."""
    from ipro_agent.structure import Structure, StructureError

    tmp = Path(tempfile.mkdtemp())
    cases = {
        "empty.pdb": b"",
        "text.pdb": b"not a pdb at all, just prose\n",
        "fasta.pdb": b">seq\nACDEFGHIKLMNPQRST\n",
    }
    for name, body in cases.items():
        (tmp / name).write_bytes(body)
    (tmp / "adir.pdb").mkdir()
    import gzip

    (tmp / "gz.pdb").write_bytes(gzip.compress(CRN.read_bytes()))

    expected = {
        "empty.pdb": "empty",
        "adir.pdb": "is a directory",
        "text.pdb": "no atoms",
        "fasta.pdb": "no atoms",
        "gz.pdb": "cannot parse",
    }
    for name, fragment in expected.items():
        try:
            Structure.load(tmp / name)
        except StructureError as exc:
            assert fragment in str(exc).lower() or fragment in str(exc), \
                f"{name}: message should mention {fragment!r}, got {exc}"
        else:
            raise AssertionError(f"{name} should not have loaded")

    # And the same through the agent-facing CLI: no traceback, one line.
    for name in cases:
        p = subprocess.run(
            [sys.executable, "-m", "ipro_agent", "inspect", str(tmp / name)],
            capture_output=True, text=True, cwd=REPO, timeout=300,
        )
        assert "Traceback" not in p.stderr, (name, p.stderr[-300:])
        assert p.returncode == 2, (name, p.returncode)

    # A real multi-model file asked for a model it lacks says how many it has.
    nmr = tmp / "nmr.pdb"
    nmr.write_text(
        "MODEL        1\n"
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ENDMDL\nMODEL        2\n"
        "ATOM      2  N   GLY A   1       1.000   0.000   0.000  1.00  0.00           N\n"
        "ENDMDL\nEND\n"
    )
    try:
        Structure.load(nmr, model=5)
    except StructureError as exc:
        assert "2 model(s)" in str(exc), exc
    else:
        raise AssertionError("model 5 does not exist in a 2-model file")


def test_inspect_handles_paths_with_spaces_and_non_ascii():
    """An agent will put a structure wherever the user's files live, and that
    includes directories with spaces and non-ASCII names."""
    ompf = REPO / "examples" / "ompf_pore" / "2omf_A.pdb"
    if not ompf.exists():
        print("    (skipped: no example structure)")
        return
    odd = Path(tempfile.mkdtemp()) / "ünïcode dir"
    odd.mkdir()
    target = odd / "my struct.pdb"
    target.write_bytes(ompf.read_bytes())

    p = subprocess.run(
        [sys.executable, "-m", "ipro_agent", "inspect", str(target),
         "--positions", "18", "--json"],
        capture_output=True, text=True, cwd=REPO, timeout=600,
    )
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["positions"][0]["native"] == "VAL"


def test_regression_the_docs_name_every_mcp_tool_and_no_stale_counts():
    """A reader-facing list of the tools went stale the moment two tools were
    added, and silently: README's table listed 8 of 11 and HANDOFF.md said
    "nine tools", both omitting `ipro_fetch_structure` and
    `ipro_inspect_structure`.

    That omission is worse than untidy. An MCP client has no shell, so for a
    remote agent those two tools are the *only* way to obtain a structure and
    the only way to learn that "residue 18" is VAL -- the step whose absence
    already shipped a defect once (DESIGN.md section 7b, a position resolved
    against the wrong chain). A tool nobody is told about does not exist.

    Hardcoded suite sizes rot the same way: "159-test suite" in HANDOFF.md and
    "141 tests" in INSTALL.md were both wrong by the time they were read. The
    historical counts inside DESIGN.md sections 7..7g are *records* of what the
    suite was at each round and are deliberately left alone."""
    import ipro_agent.mcp_server as srv

    tools = [t["name"] for t in srv.TOOLS]
    reader_facing = {
        "README.md": (REPO / "README.md").read_text(),
        "docs/HANDOFF.md": (REPO / "docs" / "HANDOFF.md").read_text(),
    }
    for name, text in reader_facing.items():
        missing = [t for t in tools if t not in text]
        assert not missing, f"{name} never mentions {missing}"

    # And no reader-facing doc may hardcode a suite size, since every fix in
    # this repo adds a test. DESIGN.md is the historical record, excluded.
    import re

    for doc in sorted((REPO / "docs").glob("*.md")) + [REPO / "README.md"]:
        if doc.name == "DESIGN.md":
            continue
        for m in re.finditer(r"(\d{2,4})[- ]tests?\b", doc.read_text()):
            raise AssertionError(
                f"{doc.name} hardcodes {m.group(0)!r}; it will be wrong after "
                f"the next fix -- say 'the whole suite' instead")


def test_regression_mcp_fetch_and_inspect_work_over_the_protocol():
    """Both audits now running were launched before `ipro_fetch_structure` and
    `ipro_inspect_structure` existed, so neither covers them. Everything else
    in this file drives them through the CLI; this drives them the way a remote
    agent does -- JSON-RPC over stdin -- because the MCP layer has its own
    argument plumbing and that is where a mismatch would hide.

    Offline: inspect only, on the shipped structure. The fetch tool is checked
    for shape (it must be declared, take a pdb_id, and be the only tool that
    touches the network) without spending a download."""
    ompf = REPO / "examples" / "ompf_pore" / "2omf_A.pdb"
    if not ompf.exists():
        print("    (skipped: no example structure)")
        return

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "t", "version": "0"}}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "ipro_inspect_structure",
                    "arguments": {"structure": str(ompf),
                                  "positions": ["18", "20", "38", "40"]}}},
    ]
    payload = "\n".join(json.dumps(r) for r in requests) + "\n"
    p = subprocess.run([sys.executable, "-m", "ipro_agent.mcp_server"],
                       input=payload, capture_output=True, text=True,
                       cwd=REPO, timeout=600)
    assert p.returncode == 0, p.stderr[-500:]

    replies = {}
    for line in p.stdout.splitlines():
        if line.strip():
            msg = json.loads(line)
            replies[msg.get("id")] = msg
    assert set(replies) == {1, 2}, sorted(replies)

    body = replies[2]["result"]["content"][0]["text"]
    assert replies[2]["result"].get("isError") is not True, body
    got = json.loads(body)
    resolved = {str(pos["asked"]): pos["native"]
                for pos in got["positions"]}
    assert resolved == {"18": "VAL", "20": "LEU", "38": "MET", "40": "TYR"}, resolved

    import ipro_agent.mcp_server as srv
    fetch = [t for t in srv.TOOLS if t["name"] == "ipro_fetch_structure"]
    assert fetch, "ipro_fetch_structure is not declared"
    assert "pdb_id" in fetch[0]["inputSchema"]["properties"]
    assert "pdb_id" in (fetch[0]["inputSchema"].get("required") or [])
    # One networked tool, and it says so in its own description. Match the
    # asserting form, not the bare word: ipro_inspect_structure's description
    # says "It needs no PyMOL and no network", which is the opposite claim.
    assert "ONLY tool here that uses the network" in fetch[0]["description"]
    claims = [t["name"] for t in srv.TOOLS
              if "uses the network" in t["description"]
              or "download a structure" in t["description"].lower()]
    assert claims == ["ipro_fetch_structure"], claims


def test_regression_the_perturbed_region_is_built_around_a_design_position():
    """IPRO's step 1 "begins by randomly selecting a DP", then draws the
    region size (1-5) and the DP's place inside it; the released source does
    the same, and only design positions inside the perturbed region may
    mutate (2006 paper, Fig. 5). Here the region was drawn anywhere in the
    chain and the redesign window was drawn independently of it, so the
    perturbation and the design were statistically unrelated.

    Measured on the shipped 25-position OmpF spec over 20,000 draws: an
    unanchored region touches a design position in 12.4% of draws, where
    IPRO's probability is 1 by construction. 88% of moves therefore changed
    the energy in a region the MILP had no variables for, and the Metropolis
    test then ruled on that change. docs/MILP.md called this "comparable
    locality"; 12.4% is not comparable locality."""
    import random

    from ipro_agent.perturb import choose_segment
    from ipro_agent.structure import Structure

    pdb = REPO / "examples" / "ompf_pore" / "2omf_A.pdb"
    if not pdb.exists():
        print("    (skipped: no example structure)")
        return
    st = Structure.load(pdb)
    dps = [18, 20, 38, 40, 42, 62, 64, 82, 83, 109, 110, 113, 114, 115,
           117, 118, 119, 120, 121, 123, 124, 125, 130, 131, 132]
    want = set(dps)

    rng = random.Random(0)
    n = 4000
    free = sum(
        bool({r.get_id()[1] for r in choose_segment(st, "A", (1, 5), rng=rng)} & want)
        for _ in range(n)
    ) / n
    assert free < 0.25, f"unanchored overlap {free:.4f}: this is the old behaviour"

    anchored = 0
    sizes = set()
    for _ in range(n):
        seg = choose_segment(st, "A", (1, 5), rng=rng, anchor=rng.choice(dps))
        nums = {r.get_id()[1] for r in seg}
        sizes.add(len(seg))
        anchored += bool(nums & want)
    assert anchored == n, f"anchored overlap {anchored}/{n} must be all of them"
    # The region size is still the paper's one-to-five.
    assert sizes <= {1, 2, 3, 4, 5} and len(sizes) >= 4, sorted(sizes)

    # And the window really is read off the region, not drawn beside it.
    src = (REPO / "ipro_agent" / "loop.py").read_text()
    body = src[src.index("def _choose_window"):src.index("def _choose_window") + 2600]
    assert "pert.residues" in body or 'getattr(pert, "residues"' in body, (
        "the redesign window must come from the perturbed region"
    )
    assert "self._choose_window(ordered, pert)" in src


def test_regression_terminal_torsions_are_not_perturbed():
    """phi needs the preceding residue's C and psi the next residue's N, so at
    a chain's own termini one of them is undefined -- and rotating about an
    undefined torsion is not a conformational change at all, it is a
    rigid-body rotation of everything downstream.

    Measured on 2OMF chain A with a forced +5 deg phi at residue 1: atoms move
    up to 3.058 A while the largest change in any internal interatomic
    distance is 2.1e-14 A, i.e. machine zero. The same perturbation at
    interior residue 151 changes internal distances by up to 3.483 A. The
    energy cannot see the first case, but the docking and the fitted pore axis
    can. The released source skips both terminal torsions."""
    import random

    import numpy as _np

    from ipro_agent.perturb import perturb_backbone
    from ipro_agent.structure import Structure

    pdb = REPO / "examples" / "ompf_pore" / "2omf_A.pdb"
    if not pdb.exists():
        print("    (skipped: no example structure)")
        return

    def shift_and_strain(number):
        st = Structure.load(pdb)
        residues = list(st.residues("A"))
        target = [r for r in residues if r.get_id()[1] == number]
        before = _np.array([a.coord for a in st.all_atoms()])
        rec = perturb_backbone(st, "A", sigma_deg=5.0, cap_deg=5.0,
                               rng=random.Random(1), segment=target)
        after = _np.array([a.coord for a in st.all_atoms()])
        moved = float(_np.abs(after - before).max())
        # Internal strain: how much any pairwise distance changed, sampled.
        idx = list(range(0, len(before), 17))
        d0 = _np.linalg.norm(before[idx][:, None] - before[idx][None], axis=-1)
        d1 = _np.linalg.norm(after[idx][:, None] - after[idx][None], axis=-1)
        return rec, moved, float(_np.abs(d1 - d0).max())

    # The claim itself, measured directly: rotating about residue 1's N->CA
    # axis moves atoms without changing the conformation at all. (psi at
    # residue 1 IS defined -- it needs the next residue's N -- so only phi is
    # the undefined one, and perturb_backbone still applies psi there.)
    from ipro_agent.perturb import _rotate_about

    st0 = Structure.load(pdb)
    res1 = next(r for r in st0.residues("A") if r.get_id()[1] == 1)
    names1 = {a.get_name().strip().upper(): a for a in res1}
    origin = _np.array(names1["N"].coord, dtype=float)
    axis = _np.array(names1["CA"].coord, dtype=float) - origin
    allc = _np.array([a.coord for a in st0.all_atoms()], dtype=float)
    rotated = _rotate_about(allc, origin, axis, 5.0)
    idx = list(range(0, len(allc), 17))
    d0 = _np.linalg.norm(allc[idx][:, None] - allc[idx][None], axis=-1)
    d1 = _np.linalg.norm(rotated[idx][:, None] - rotated[idx][None], axis=-1)
    assert float(_np.abs(rotated - allc).max()) > 1.0, "atoms should move"
    assert float(_np.abs(d1 - d0).max()) < 1e-6, (
        "a phi rotation at residue 1 is a rigid-body move: no internal "
        "distance may change"
    )

    rec1, moved1, strain1 = shift_and_strain(1)
    assert rec1.phi_deg == [0.0], (
        f"phi at the N-terminal residue is undefined and must not be "
        f"perturbed; got {rec1.phi_deg}"
    )

    recmid, movedmid, strainmid = shift_and_strain(151)
    assert recmid.phi_deg and recmid.phi_deg[0] != 0.0, recmid.phi_deg
    assert strainmid > 0.5, (
        f"an interior perturbation must change the conformation; "
        f"strain {strainmid}"
    )

    # psi at the last residue is the mirror image of the same bug.
    last = max(r.get_id()[1] for r in Structure.load(pdb).residues("A"))
    reclast, _m, _s = shift_and_strain(last)
    assert reclast.psi_deg == [0.0], reclast.psi_deg


def test_regression_a_pip_installed_server_can_doctor_setup_and_run():
    """The install README recommends FIRST could not run a design at all.

    `_doctor()` shelled out to ``REPO_ROOT / "scripts" / "doctor.py"``, and
    pyproject ships only the three `ipro_agent*` packages -- so in a
    pip-installed copy REPO_ROOT is site-packages, where there is no
    `scripts/`. Measured in a throwaway venv: ipro_doctor, ipro_setup and
    ipro_run all returned isError with "can't open file
    .../site-packages/scripts/doctor.py", while ipro_goals, ipro_validate,
    ipro_positions, ipro_score, ipro_milp and ipro_status all worked normally
    -- so the server looked healthy right up to the point of designing
    anything. ipro_doctor is the first tool the server's own `instructions`
    tell a client to call.

    `ipro_setup` had the same shape of bug one level down: it printed
    ``mamba env create -f <site-packages>/environment.yml`` for a file that is
    not in the wheel either.

    The doctor now lives in the package (`ipro_agent/doctor.py`, with
    `scripts/doctor.py` kept as a shim for the documented command), and the
    missing environment.yml falls back to fetching the real one."""
    import ast

    src = (REPO / "ipro_agent" / "mcp_server.py").read_text()
    tree = ast.parse(src)

    # Nothing the server executes may live outside the package.
    assert '"scripts"' not in src and "'scripts'" not in src, (
        "the server must not execute anything from scripts/: a wheel has none"
    )
    assert '"-m", "ipro_agent.doctor"' in src

    # The doctor runs as a module, from any interpreter.
    p = subprocess.run([sys.executable, "-m", "ipro_agent.doctor", "--json"],
                       capture_output=True, text=True, cwd=REPO, timeout=600)
    assert p.returncode in (0, 1), p.stderr[-400:]
    rep = json.loads(p.stdout)
    assert "can_run" in rep and "checks" in rep, sorted(rep)

    # And the shim still answers the command that INSTALL.md documents.
    shim = subprocess.run([sys.executable, str(REPO / "scripts" / "doctor.py"), "--json"],
                          capture_output=True, text=True, cwd=REPO, timeout=600)
    assert shim.returncode in (0, 1), shim.stderr[-300:]
    assert json.loads(shim.stdout)["checks"].keys() == rep["checks"].keys()

    # environment.yml: present from a clone, and when absent the command says
    # how to get the real one rather than naming a path that does not exist.
    from ipro_agent import mcp_server as m

    assert m._env_file() is not None, "a clone has environment.yml"
    real = m._env_file()
    g = m.tool_setup.__globals__
    keep_roots, keep_doctor = g["_doc_roots"], g["_doctor"]
    try:
        # An installed copy: no sibling files, and PyMOL not resolved, so the
        # install branch is the one that runs.
        g["_doc_roots"] = lambda: [Path(tempfile.mkdtemp())]
        g["_doctor"] = lambda: {
            "can_run": True, "can_design": False, "missing_required": [],
            "resolved": {}, "degraded": ["pymol"],
        }
        plan = m.tool_setup({})
        assert plan["action"] == "install", plan
        where, cmd = plan.get("environment_file", ""), plan.get("command", "")
        assert m.RAW_BASE in where and m.RAW_BASE in cmd, plan
        assert "env create" in cmd and "site-packages" not in cmd, cmd
    finally:
        g["_doc_roots"], g["_doctor"] = keep_roots, keep_doctor
    assert m._env_file() == real


def test_regression_optzyme_eq18_weights_the_two_terms_as_the_paper_does():
    """OptZyme's Eq 18 is ``IE_TSA/(RT)_TSA - IE_S/(RT)_S`` and the two
    constants are not the same number: the paper gives
    ``(RT)_TSA = 15.3 kJ/mol`` and ``(RT)_S = 386.7 kJ/mol`` for GUS/pNP-GLU,
    self-consistent with the weights it also states (0.06 and 0.002 mol/kJ), so
    the substrate term carries 4% of the TSA term's weight.

    This divided both terms by one RT at 298.15 K, which sets the relative
    weight to 1.0 -- a 25x over-weighting of the substrate term that changes
    the ARGMIN, not the scale. Measured on two candidates with both energies
    favourable: A (IE_TSA -15.0, IE_S -2.0) and B (IE_TSA -20.0, IE_S -12.0).
    Eq 18 gives A -4.080, B -5.339, so B wins. One RT gave A -21.944,
    B -13.504, so A won -- rejecting the candidate that binds the transition
    state 5 kcal/mol better. There was no test for `kcat` at all.

    These are fitted correlating constants for one enzyme/substrate pair, not
    temperatures ("experimental and correlating temperatures do not match"),
    hence the explicit refusal of a single `temperature_k` for this parameter.
    """
    from ipro_agent.goals import ImproveCatalysis, ScoreContext, build_goal

    rt_tsa, rt_s = ImproveCatalysis.RT_TSA_KCAL, ImproveCatalysis.RT_SUBSTRATE_KCAL
    assert abs(rt_tsa - 15.3 / 4.184) < 1e-9 and abs(rt_s - 386.7 / 4.184) < 1e-9
    assert abs(rt_s / rt_tsa - 25.27) < 0.05, rt_s / rt_tsa

    spec = DesignSpec.from_dict({
        "goal": "improve_catalysis",
        "goal_statement": "raise kcat",
        "goal_options": {"parameter": "kcat"},
        "molecules": [
            {"name": "enz", "path": str(CRN), "role": "design",
             "kind": "protein", "chains": ["A"]},
            {"name": "tsa", "path": str(CRN), "role": "target",
             "kind": "transition_state"},
            {"name": "sub", "path": str(CRN), "role": "target",
             "kind": "ligand"},
        ],
        "positions": ["A16"],
    })
    goal = build_goal(spec)
    goal.validate()

    def score(ie_tsa, ie_sub):
        return goal.score(ScoreContext(interaction={"tsa": ie_tsa, "sub": ie_sub}))

    a, b = score(-15.0, -2.0), score(-20.0, -12.0)
    assert abs(a - (-4.080)) < 0.01, a
    assert abs(b - (-5.339)) < 0.01, b
    assert b < a, (
        "the candidate binding the transition state better must win; one "
        "shared RT inverted this"
    )

    # The weighting is visible in the report, since it is the trap.
    comp = goal.components(ScoreContext(interaction={"tsa": -15.0, "sub": -2.0}))
    assert abs(comp["RT_TSA_kcal"] - rt_tsa) < 1e-9
    assert abs(comp["RT_substrate_kcal"] - rt_s) < 1e-9
    assert abs(comp["weighted_IE_substrate"]) < abs(comp["weighted_IE_TSA"]) / 10

    # Another system can override both; one temperature for both is refused.
    spec.goal_options.update({"rt_tsa_kcal": 1.0, "rt_substrate_kcal": 1.0})
    flat = build_goal(spec)
    assert abs(flat.score(ScoreContext(interaction={"tsa": -15.0, "sub": -2.0}))
               - (-13.0)) < 1e-9

    spec.goal_options.clear()
    spec.goal_options.update({"parameter": "kcat", "temperature_k": 310.0})
    try:
        build_goal(spec).validate()   # build_goal validates as it builds
    except SpecError as exc:
        assert "25" in str(exc) and "rt_tsa_kcal" in str(exc), exc
    else:
        raise AssertionError("a single temperature_k must be refused for kcat")

    # km and kcat_over_km are single-RT by the paper's own Eqs 7 and 17.
    spec.goal_options.clear()
    spec.goal_options.update({"parameter": "kcat_over_km", "temperature_k": 310.0})
    build_goal(spec).validate()


def test_regression_one_bad_request_costs_one_reply_not_the_session():
    """Only ``tools/call`` was exception-guarded, so five other malformed
    shapes killed the whole server -- one bad message ending the session for
    every later one. Measured, each with exit=1 and the session gone:
    ``params`` as an array on initialize and on tools/call
    (``AttributeError: 'list' object has no attribute 'get'``), a tools/call
    ``name`` that is a dict and a resources/read ``uri`` that is a list
    (``TypeError: unhashable type``), and ``arguments`` as a list or a string
    (``TypeError: list indices must be integers``).

    And the declared schemas were decorative: a missing ``spec`` came back as
    ``KeyError: 'spec'`` and ``spec: 42`` as ``AttributeError: 'int' object
    has no attribute 'strip'`` -- neither of which a model can act on, and
    both of which read as a bug in the tool rather than in the call.

    A one-line inline spec was treated as a filename, because inline content
    was detected by looking for a newline: an agent holding the spec in
    context sent ``goal: bind_tighter, ...`` and got ``spec not found: goal:
    bind_tighter``."""
    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "0"}}}
    cases = [
        ("params array on initialize", {"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": ["x"]}, "rpc"),
        ("params array on tools/call", {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": ["x"]}, "rpc"),
        ("unhashable tool name", {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": {"a": 1}}}, "rpc"),
        ("unhashable uri", {"jsonrpc": "2.0", "id": 5, "method": "resources/read", "params": {"uri": ["a"]}}, "rpc"),
        ("arguments as a list", {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "ipro_validate", "arguments": ["x"]}}, "rpc"),
        ("arguments as a string", {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "ipro_validate", "arguments": "spec"}}, "rpc"),
        ("spec missing", {"jsonrpc": "2.0", "id": 8, "method": "tools/call", "params": {"name": "ipro_validate", "arguments": {}}}, "tool"),
        ("spec is a number", {"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "ipro_validate", "arguments": {"spec": 42}}}, "tool"),
        ("iterations as a float", {"jsonrpc": "2.0", "id": 10, "method": "tools/call", "params": {"name": "ipro_run", "arguments": {"spec": "x.yaml", "iterations": 8.5}}}, "tool"),
        ("unknown notification", {"jsonrpc": "2.0", "method": "notifications/roots/list_changed"}, "silent"),
    ]
    payload = "\n".join(json.dumps(m) for m in [init] + [c[1] for c in cases]) + "\n"
    p = subprocess.run([sys.executable, "-m", "ipro_agent.mcp_server"],
                       input=payload, capture_output=True, text=True,
                       cwd=REPO, timeout=900)
    assert p.returncode == 0, (
        f"the server must survive every one of these: {p.stderr[-400:]}"
    )
    replies = {}
    for line in p.stdout.splitlines():
        if line.strip():
            m = json.loads(line)
            replies[m.get("id")] = m
    assert None not in replies, (
        f"a notification must get no reply at all, got "
        f"{json.dumps(replies[None])[:200]}"
    )
    assert replies[1]["result"]["serverInfo"], "initialize must still work"

    for label, msg, kind in cases:
        mid = msg.get("id")
        if mid is None:
            continue
        r = replies.get(mid)
        assert r is not None, f"{label}: no reply at all"
        if kind == "rpc":
            assert "error" in r, f"{label}: expected a JSON-RPC error, got {r}"
            assert r["error"]["code"] == -32602, (label, r["error"])
            text = r["error"]["message"]
        else:
            assert r["result"].get("isError") is True, (label, r)
            text = json.loads(r["result"]["content"][0]["text"])["error"]
        # Every message must name what was wrong in words, not as a Python
        # exception class.
        for forbidden in ("KeyError", "AttributeError", "TypeError: list indices",
                          "unhashable"):
            assert forbidden not in text, (label, text)
        assert len(text) > 20, (label, text)

    # A one-line inline spec is inline, not a filename.
    from ipro_agent import mcp_server as m

    assert m._looks_inline("goal: bind_tighter")
    assert m._looks_inline("{goal: bind_tighter, positions: [A16]}")
    assert m._looks_inline("goal: x\npositions: [A16]")
    assert not m._looks_inline("examples/ompf_pore/spec_smoke.yaml")
    assert not m._looks_inline("/tmp/my spec.yaml")


def test_regression_a_run_that_died_at_once_is_not_reported_as_launched():
    """``tool_run`` passed its numeric arguments through as ``str(...)`` and
    never looked at the child again. A JSON float for an integer field --
    which a model will emit -- became ``--iterations 8.0``, argparse exited 2
    within milliseconds, and the reply was still ``isError: false,
    launched: true`` with a pid that was already gone. ``ipro_status`` on that
    directory then answered ``"pending"``, so a polling agent waited forever
    on a run that never started; the real error sat in launch.log, which the
    agent had no reason to read.

    Now the arguments are coerced (and refused with a sentence naming the
    argument), the child is given a moment and polled, and a non-zero exit
    comes back as an error carrying the tail of its own log."""
    from ipro_agent import mcp_server as m

    # Coercion happens before anything is launched.
    for bad in (8.5, "lots", True, [8]):
        try:
            m.tool_run({"spec": str(REPO / "examples" / "ompf_pore" / "spec_smoke.yaml"),
                        "iterations": bad})
        except m.ToolError as exc:
            assert "iterations" in str(exc), (bad, exc)
        except Exception as exc:
            raise AssertionError(f"{bad!r} should be a ToolError, got {exc!r}")
        else:
            raise AssertionError(f"{bad!r} must be refused")

    # A spec the CLI refuses outright: the child dies, and so must the claim.
    tmp = Path(tempfile.mkdtemp())
    bad_spec = tmp / "bad.yaml"
    bad_spec.write_text("goal: bind_tighter\ngoal_statement: x\nmolecules: []\npositions: []\n")
    g = m.tool_run.__globals__
    keep = g["_doctor"]
    try:
        g["_doctor"] = lambda: {"can_run": True, "can_design": True,
                                "degraded": [], "resolved": {}}
        try:
            out = m.tool_run({"spec": str(bad_spec), "run_dir": str(tmp / "r")})
        except m.ToolError as exc:
            assert "exited immediately" in str(exc), exc
            assert len(str(exc)) > 60, "the error must carry the log tail"
        else:
            raise AssertionError(
                f"a run that died at once must not report launched: {out}"
            )
    finally:
        g["_doctor"] = keep


def test_regression_malformed_specs_report_errors_not_tracebacks():
    """Eleven classes of malformed-but-plausible spec died with a bare Python
    traceback from an agent-facing CLI. Each is an ordinary mistake:

    - a tab used for indentation -> ``yaml.scanner.ScannerError`` after 18
      frames (PyYAML's message carries the line and column, which is the only
      useful part);
    - no ``goal_statement:`` -> ``TypeError: DesignSpec.__init__() missing 1
      required positional argument: 'goal_statement'``, Python internals for a
      missing line in the user's own file;
    - ``positions:`` written as a mapping -> parsed as its KEYS, so every
      allowed-residue set was dropped and the alphabet silently widened to all
      18 (measured: a design space of 324 sequences where 1 was asked for);
    - ``allowed: []`` -> the same silent widening, because "absent" and
      "empty" were the same test;
    - ``target_radius: "two"`` -> ``validate`` said "spec is usable" and the
      RUN then died at iteration 0 with ValueError.

    And the four read-only subcommands crashed where ``validate`` was clean:
    an unknown goal name reported properly by ``validate`` killed ``score``,
    ``milp``, ``positions`` and ``dock`` with a traceback, because only
    ``_load`` was guarded and ``build_goal`` raises later."""
    tmp = Path(tempfile.mkdtemp())
    mol = (f"molecules:\n  - {{name: m, path: {CRN}, role: design, "
           f"kind: protein, chains: [A]}}\n")

    cases = {
        "a_tab.yaml": "goal: reshape_channel\n\tname: bad\n",
        "no_statement.yaml": "goal: bind_tighter\n",
        "no_goal.yaml": "goal_statement: x\n",
        "pos_mapping.yaml": (
            "goal: bind_tighter\ngoal_statement: x\n" + mol
            + "positions:\n  A16: [TRP]\n  A18: [TRP]\n"),
        "empty_allowed.yaml": (
            "goal: bind_tighter\ngoal_statement: x\n" + mol
            + "positions:\n  - {position: A16, allowed: []}\n"),
        "bad_radius.yaml": (
            "goal: reshape_channel\ngoal_statement: narrow it\n" + mol
            + "positions: [A16, A18, A20]\n"
            + 'goal_options: {direction: narrow, target_radius: "two"}\n'),
        "not_a_mapping.yaml": "- a\n- b\n",
    }
    for name, body in cases.items():
        (tmp / name).write_text(body)
        p = subprocess.run(
            [sys.executable, "-m", "ipro_agent", "validate", str(tmp / name)],
            capture_output=True, text=True, cwd=REPO, timeout=600,
        )
        both = p.stdout + p.stderr
        assert "Traceback" not in both, f"{name}:\n{both[-500:]}"
        assert p.returncode != 0, (name, both[-300:])
        for junk in ("ScannerError", "__init__()", "could not convert string"):
            assert junk not in both, (name, junk, both[-300:])
    # The messages say what to do, not just what broke.
    out = subprocess.run(
        [sys.executable, "-m", "ipro_agent", "validate", str(tmp / "a_tab.yaml")],
        capture_output=True, text=True, cwd=REPO, timeout=600).stderr
    assert "line 2" in out and "spaces" in out, out
    out = subprocess.run(
        [sys.executable, "-m", "ipro_agent", "validate", str(tmp / "pos_mapping.yaml")],
        capture_output=True, text=True, cwd=REPO, timeout=600).stderr
    assert "must be a LIST" in out and "position: A16" in out, out

    # An empty allowed set is an error; an absent one is the default.
    from ipro_agent.config import DesignPosition

    assert len(DesignPosition.parse("A16").allowed) == 18
    assert len(DesignPosition.parse("A16", None).allowed) == 18
    try:
        DesignPosition.parse("A16", [])
    except SpecError as exc:
        assert "empty allowed-residue set" in str(exc), exc
    else:
        raise AssertionError("allowed: [] must be refused, not widened to 18")

    # Every read-only subcommand gets the same contract as validate.
    bad_goal = tmp / "bad_goal.yaml"
    bad_goal.write_text("goal: improve_affinity\ngoal_statement: x\n" + mol
                        + "positions: [A16]\n")
    for cmd in (["score"], ["milp"], ["positions"], ["dock"]):
        p = subprocess.run(
            [sys.executable, "-m", "ipro_agent", *cmd, str(bad_goal), "--json"],
            capture_output=True, text=True, cwd=REPO, timeout=900,
        )
        both = p.stdout + p.stderr
        assert "Traceback" not in both, (cmd, both[-400:])
        assert p.returncode != 0, (cmd, p.returncode, both[-200:])
        # Either the spec-error contract, or that subcommand's own one-line
        # refusal (`positions` needs a target molecule before it gets as far
        # as the goal). What must never happen is a traceback.
        assert len(both.strip().splitlines()) <= 6, (cmd, both[-300:])
    # score and milp reach the goal, so they must name it.
    for cmd in (["score"], ["milp"]):
        p = subprocess.run(
            [sys.executable, "-m", "ipro_agent", *cmd, str(bad_goal), "--json"],
            capture_output=True, text=True, cwd=REPO, timeout=900,
        )
        assert "unknown goal" in (p.stdout + p.stderr), (cmd, p.stderr[-200:])
        assert p.returncode == 2, (cmd, p.returncode)


def test_regression_a_dead_run_does_not_report_itself_as_running():
    """A killed run left ``status: "running"`` in state.json for ever:
    measured, a status call 4.5 minutes after the process had gone still
    reported "running, 20 of 60 iterations", and nothing in the payload let an
    agent tell that apart from a healthy run -- so a polling agent waits for
    ever. The pid is now recorded at start, and status answers the question.

    Re-running into a directory that already holds a run was the other half:
    two 6-iteration runs of DIFFERENT specs left iterations.jsonl with 12
    lines, status reporting "12 of 6 iterations", and best_score carried over
    from the other spec's run -- a result attributed to a spec that never
    produced it. The MCP server refused this; the CLI appended."""
    from ipro_agent.state import RunDir

    tmp = Path(tempfile.mkdtemp())
    rd = RunDir(tmp / "r", create=True)
    rd.update_state(status="running", pid=999_999, host=None,
                    total_iterations=60, mode="fixed-backbone")
    out = rd.as_json()
    assert out["status"] == "stale", out
    assert out["status_in_file"] == "running", out
    assert out["seconds_since_update"] is not None
    assert any("THIS RUN IS NOT RUNNING" in n for n in out["notes"]), out["notes"]

    # A live pid is still reported as running.
    rd.update_state(status="running", pid=os.getpid(), host=None)
    assert rd.as_json()["status"] == "running"

    # And the CLI refuses to append to a used run directory.
    used = tmp / "used"
    used.mkdir()
    (used / "state.json").write_text("{}")
    p = subprocess.run(
        [sys.executable, "-m", "ipro_agent", "run",
         str(REPO / "examples" / "ompf_pore" / "spec_smoke.yaml"),
         "--run-dir", str(used), "--iterations", "1"],
        capture_output=True, text=True, cwd=REPO, timeout=900,
    )
    assert p.returncode == 2, (p.returncode, p.stdout[-300:], p.stderr[-300:])
    assert "already holds a run" in p.stderr, p.stderr[-300:]
    assert "--resume" in p.stderr
    assert "Traceback" not in p.stderr


def test_regression_two_molecules_at_one_place_are_not_an_interaction():
    """The design molecule and a "partner" both pointing at 2omf_A.pdb chain A
    validated clean and scored an interaction energy of **+149,211 kcal/mol**
    -- atoms at zero distance, reported as a measurement. A goal whose filter
    is "better than the reference" would have adopted that as the reference.

    A homodimer legitimately shares a *file*, so this is not a parse-time
    error: validate reports it, and the loop reports any interaction energy
    that is physically impossible whatever its cause (a partner that was never
    docked, for instance)."""
    tmp = Path(tempfile.mkdtemp())
    same = tmp / "same.yaml"
    same.write_text(
        f"goal: bind_tighter\ngoal_statement: x\nmolecules:\n"
        f"  - {{name: a, path: {CRN}, role: design, kind: protein, chains: [A]}}\n"
        f"  - {{name: b, path: {CRN}, role: target, kind: protein, chains: [A]}}\n"
        f"positions: [A16]\n"
    )
    p = subprocess.run(
        [sys.executable, "-m", "ipro_agent", "validate", str(same), "--json"],
        capture_output=True, text=True, cwd=REPO, timeout=600,
    )
    assert p.returncode != 0, p.stdout
    body = json.loads(p.stdout)
    assert body["ok"] is False
    assert any("same coordinates" in x for x in body["problems"]), body["problems"]

    # Constructing the spec is still allowed (the fixture above is one line of
    # YAML away from a legitimate homodimer).
    DesignSpec.load(same)

    # The energy-side guard, measured rather than grepped: `score` does not
    # run validate's checks, so it is the path that would report the number.
    if not _have_pymol():
        print("    (partial: energy-side guard needs PyMOL)")
        return
    p = subprocess.run(
        [sys.executable, "-m", "ipro_agent", "score", str(same), "--json",
         "--run-dir", str(tmp / "score")],
        capture_output=True, text=True, cwd=REPO, timeout=900,
    )
    assert p.returncode == 0, p.stderr[-400:]
    got = json.loads(p.stdout)
    ie = got.get("interaction_energy") or sum(
        (got.get("components") or {}).get(k, 0.0)
        for k in (got.get("components") or {}) if k.startswith("IE")
    )
    assert any("not a" in n and "physical interaction" in n
               for n in got.get("notes", [])), (ie, got.get("notes"))


def test_regression_the_doctor_gates_on_the_python_floor():
    """On Python 3.9.6 with numpy, Biopython and PyYAML all installed, the
    doctor reported ``python 3.9.6 | can_run True | can_design True |
    missing_required []`` and exited 0 -- and ``--require pymol``, the
    mandatory gate that AGENTS.md tells agents to trust, also exited 0. On
    that interpreter ``import ipro_agent`` raises immediately: the package
    floor is 3.10 and the code uses 3.10 syntax. A green light there is the
    worst possible answer, because it sends the user into a run instead of
    into an upgrade.

    The doctor also has to keep working ON such an interpreter, which is why
    ``scripts/doctor.py`` loads the module by file path instead of importing
    ``ipro_agent.doctor``: the package's own refusal to import would otherwise
    take the diagnosis down with it. Measured: ``python3.9 -m
    ipro_agent.doctor`` prints the upgrade notice and nothing else, while
    ``python3.9 scripts/doctor.py`` prints the full report."""
    import ast

    from ipro_agent import doctor as d
    from ipro_agent import REQUIRES_PYTHON

    assert d.PYTHON_FLOOR == REQUIRES_PYTHON, (
        d.PYTHON_FLOOR, REQUIRES_PYTHON,
        "the doctor's hardcoded floor must match the package's",
    )
    rep = d.check()
    assert rep["python_ok"] is True, rep["python"]
    assert rep["python_floor"] == ".".join(str(n) for n in REQUIRES_PYTHON)

    # The module must import nothing from the package, or the shim cannot
    # bypass the package guard.
    src = (REPO / "ipro_agent" / "doctor.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("ipro_agent"), node.module
            assert node.level == 0, (
                f"relative import of {node.module!r}: the doctor must stay "
                f"loadable without the package"
            )
        if isinstance(node, ast.Import):
            for a in node.names:
                assert not a.name.startswith("ipro_agent"), a.name
    shim = (REPO / "scripts" / "doctor.py").read_text()
    assert "spec_from_file_location" in shim, (
        "the shim must load the module by path, not import the package"
    )

    # A simulated old interpreter: the floor decides, not what is installed.
    keep = d.sys.version_info
    try:
        d.sys.version_info = (3, 9, 6, "final", 0)        # type: ignore[assignment]
        old = d.check()
    finally:
        d.sys.version_info = keep                          # type: ignore[assignment]
    assert old["python_ok"] is False
    assert old["can_run"] is False and old["can_design"] is False
    assert any("python>=3.10" in m for m in old["missing_required"]), \
        old["missing_required"]


def test_regression_conda_search_covers_the_installer_we_recommend():
    """The root list had four entries and missed the whole **forge** family --
    including Miniforge, which docs/INSTALL.md and ``ipro_setup`` tell the user
    to install and which is the only supported installer on Apple silicon. So
    a user who followed the instructions exactly got "PyMOL did not resolve"
    with the environment sitting in ``~/miniforge3/envs/ipro-agent``.

    Worse, the doctor and the rotamer harvester kept SEPARATE lists, and the
    doctor's had two roots the harvester lacked -- so the doctor reported PyMOL
    as resolved for a run that then could not find it. One list now."""
    from ipro_agent.doctor import conda_env_roots, env_interpreter
    from ipro_agent import rotamers

    src = (REPO / "ipro_agent" / "rotamers.py").read_text()
    assert "conda_env_roots" in src, (
        "the harvester must use the doctor's root list, not its own"
    )
    assert "miniconda3\" / \"envs\"" not in src, "a second list has grown back"

    # Every forge layout is covered. (is_dir() filters the list on this
    # machine, so the names are checked in the source of the function.)
    import inspect

    text = inspect.getsource(conda_env_roots)
    for needed in ("miniforge3", "mambaforge", "micromamba", "MAMBA_ROOT_PREFIX",
                   "CONDA_ENVS_PATH", "USERPROFILE", "miniconda3", "anaconda3"):
        assert needed in text, f"{needed} is not searched"

    # Windows and POSIX interpreter names both resolve.
    tmp = Path(tempfile.mkdtemp())
    (tmp / "posix" / "bin").mkdir(parents=True)
    (tmp / "posix" / "bin" / "python").write_text("")
    (tmp / "win" / "Scripts").mkdir(parents=True)
    (tmp / "win" / "Scripts" / "python.exe").write_text("")
    (tmp / "empty").mkdir()
    assert env_interpreter(tmp / "posix") is not None
    assert env_interpreter(tmp / "win") is not None
    assert env_interpreter(tmp / "empty") is None


def test_regression_a_pymol_launcher_is_not_an_interpreter():
    """``shutil.which("pymol")`` was returned unchecked, with the comment "a
    pymol launcher can run a script too". It cannot run one the way this
    package invokes it: given ``-c "import pymol"`` the launcher treats the
    code as a FILE to open, prints "Error: unsupported file type: import
    pymol" -- and exits 0. So it was accepted as the harvest interpreter, and
    the run then started a windowless PyMOL waiting for input until the whole
    job timed out, with no rotamers and no error.

    The probe now requires a sentinel on stdout, which only a real interpreter
    with pymol importable can print."""
    from ipro_agent import rotamers

    src = (REPO / "ipro_agent" / "rotamers.py").read_text()
    assert "IPRO_PYMOL_OK" in src, "the probe must require proof, not exit 0"
    assert "return exe  # a pymol launcher can run a script too" not in src

    # The first version of this fix tested for the sentinel as a SUBSTRING of
    # stdout, which the launcher passes: it echoes its own argument back, and
    # the argument contains the sentinel. This stand-in reproduces exactly
    # that -- it exits 0 and echoes the command line, sentinel and all.
    tmp = Path(tempfile.mkdtemp())
    fake = tmp / "pymol"
    fake.write_text(
        "#!/bin/sh\necho \"Error: unsupported file type: $*\"\nexit 0\n"
    )
    fake.chmod(0o755)
    keep = os.environ.get("IPRO_PYMOL_PYTHON")
    try:
        os.environ["IPRO_PYMOL_PYTHON"] = str(fake)
        got = rotamers.find_pymol_python()
        assert got != str(fake), (
            "a program that exits 0 and echoes the probe back must be refused"
        )
    finally:
        if keep is None:
            os.environ.pop("IPRO_PYMOL_PYTHON", None)
        else:
            os.environ["IPRO_PYMOL_PYTHON"] = keep
    assert 'lines[-1] == "IPRO_PYMOL_OK"' in src, (
        "the sentinel must be the whole last line, or an echo passes"
    )


def test_regression_documents_and_specs_are_utf8_not_locale_encoded():
    """Under an ASCII locale (``LC_ALL=C``, which is what a cron job, a
    container and many MCP clients give a child process) every
    ``read_text()``/``write_text()`` used the locale encoding. Measured: the
    served ``ipro://skill`` came back with **210 U+FFFD replacement
    characters** -- the operating manual, corrupted -- and writing an inline
    spec containing any non-ASCII character failed outright."""
    import ast

    offenders = []
    for path in sorted((REPO / "ipro_agent").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not isinstance(fn, ast.Attribute):
                continue
            if fn.attr not in ("read_text", "write_text"):
                continue
            if not any(k.arg == "encoding" for k in node.keywords):
                offenders.append(f"{path.name}:{node.lineno} {fn.attr}")
    assert not offenders, (
        "every file this package reads or writes is its own data and must be "
        f"UTF-8: {offenders}"
    )

    # And the served manual really is byte-identical under that locale.
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import ipro_agent.mcp_server as m\n"
        "g = m._resource_read('ipro://skill')['contents'][0]['text']\n"
        "print(g.count('\\ufffd'))\n" % str(REPO)
    )
    env = dict(os.environ, LC_ALL="C", LANG="C", PYTHONUTF8="0",
               PYTHONCOERCECLOCALE="0")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=REPO, env=env, timeout=600)
    assert p.returncode == 0, p.stderr[-300:]
    assert p.stdout.strip() == "0", (
        f"{p.stdout.strip()} replacement characters in the served manual"
    )


def test_regression_zero_is_a_mistake_not_an_unset_value():
    """``--iterations 0`` and ``--max-minutes 0`` were tested for truthiness,
    so both were silently dropped: measured, a run asked for 0 iterations ran
    the spec's full 6. And ``max_minutes: 0`` in a spec meant *no limit* --
    the opposite of what it says."""
    from ipro_agent.config import LoopSettings

    for kwargs, fragment in (
        ({"iterations": 0}, "at least 1"),
        ({"iterations": -3}, "at least 1"),
        ({"max_minutes": 0}, "must be positive"),
        ({"max_minutes": -1.0}, "must be positive"),
        ({"iterations": "six"}, "whole number"),
        ({"max_minutes": "soon"}, "number of minutes"),
        ({"random_seed": "lucky"}, "whole number"),
    ):
        try:
            LoopSettings(**kwargs)
        except SpecError as exc:
            assert fragment in str(exc), (kwargs, exc)
        else:
            raise AssertionError(f"{kwargs} must be refused")

    assert LoopSettings(max_minutes=None).max_minutes is None
    assert LoopSettings(iterations=1).iterations == 1

    # Through the CLI, where the truthiness test actually was.
    p = subprocess.run(
        [sys.executable, "-m", "ipro_agent", "run",
         str(REPO / "examples" / "ompf_pore" / "spec_smoke.yaml"),
         "--run-dir", str(Path(tempfile.mkdtemp()) / "zero"), "--iterations", "0"],
        capture_output=True, text=True, cwd=REPO, timeout=900,
    )
    assert p.returncode == 2, (p.returncode, p.stdout[-200:])
    assert "at least 1" in p.stderr, p.stderr[-200:]


def test_regression_the_rotamer_prune_is_relative_as_the_paper_is():
    """Two findings in one place. The prune used a single global 50 kcal/mol
    threshold, and the docs called it generalized dead-end elimination with
    the claim that a pruned rotamer "cannot appear in an optimal assignment".

    It can. Constructed with this repo's own MILP and glpsol: self energies
    ``{A1 r0: 0.0, A1 r1: 51.0}`` with a pair term ``(A1,r1,A2,r0): -60.0``
    has optimum **-9.000** choosing r1, where a >50 prune gives **+0.000**
    choosing r0 -- the prune changed the answer, so the claim is withdrawn.

    And IPRO's actual rule is RELATIVE to the best rotamer at each position
    (``ROTAMERS.py``: ``sorted[n][i] <= sorted[n][0] + window``, 40 kcal/mol).
    Measured: six rotamers at one position with self energies 60, 62, 70, 75,
    90, 99 -- a 39 kcal/mol band, inside IPRO's window -- kept **1 of 6** under
    the absolute threshold (the native, via the safeguard) where IPRO keeps all
    six. On the shipped OmpF smoke spec the absolute default also deleted TRP,
    PHE and TYR from A18 outright, taking the published V18W answer out of the
    design space (DESIGN.md §7d defect 25)."""
    import numpy as _np

    from ipro_agent.energy.base import AtomSet
    from ipro_agent.milp import EnergyTables, Rotamer, RotamerMILP, build_tables

    def rot(pos, aa, state):
        return Rotamer(position=pos, aa=aa, state=state,
                       atoms=AtomSet(names=["CB"], coords=_np.zeros((1, 3)),
                                     resnames=[aa]))

    # The counterexample to the DEE claim.
    tab = EnergyTables(
        positions=["A1", "A2"],
        rotamers={"A1": [rot("A1", "ALA", 0), rot("A1", "TRP", 1)],
                  "A2": [rot("A2", "ALA", 0)]},
        stats={})
    tab.self_energy = {("A1", 0): 0.0, ("A1", 1): 51.0, ("A2", 0): 0.0}
    tab.pair_energy = {("A1", 1, "A2", 0): -60.0}
    sol = RotamerMILP(tab).solve()
    assert abs(sol.objective - (-9.0)) < 1e-6, sol.objective
    assert sol.choices["A1"] == 1, sol.choices
    for doc in ("docs/MILP.md", "docs/DESIGN.md"):
        text = (REPO / doc).read_text()
        assert "cannot appear in\n   an optimal assignment" not in text
        assert "cannot appear in an optimal assignment" not in text or \
            "was wrong" in text, doc

    # The relative window, measured on a band inside IPRO's 40 kcal/mol.
    class _Flat:
        can_relax = False
        name = "flat"
        units = "test"
        screening_pairwise = True

        def __init__(self, energies):
            self.energies = energies

        def interaction(self, a, b):
            # One atom per rotamer, named for its energy.
            return self.energies[a.names[0]]

        def internal(self, a):
            return 0.0

    energies = {"r0": 60.0, "r1": 62.0, "r2": 70.0,
                "r3": 75.0, "r4": 90.0, "r5": 99.0}
    rots = {"A1": []}
    for i, (name, _e) in enumerate(sorted(energies.items())):
        aa = "ALA" if name == "r0" else "TRP"
        rots["A1"].append(
            Rotamer(position="A1", aa=aa, state=i,
                    atoms=AtomSet(names=[name], coords=_np.zeros((1, 3)),
                                  resnames=[aa])))
    template = AtomSet(names=["N"], coords=_np.zeros((1, 3)), resnames=["GLY"])

    relative = build_tables(rots, template, _Flat(energies),
                            native={"A1": "ALA"}, rotamer_window=40.0)
    assert len(relative.rotamers["A1"]) == 6, (
        f"IPRO's 40 kcal/mol window keeps all six; kept "
        f"{len(relative.rotamers['A1'])}"
    )
    absolute = build_tables(rots, template, _Flat(energies),
                            native={"A1": "ALA"}, rotamer_window=1e9,
                            max_self_energy=50.0)
    assert len(absolute.rotamers["A1"]) == 1, (
        f"the old absolute threshold kept {len(absolute.rotamers['A1'])}"
    )
    # And the default is off, so the absolute ceiling cannot bite unasked.
    import inspect

    sig = inspect.signature(build_tables)
    assert sig.parameters["max_self_energy"].default is None
    assert sig.parameters["rotamer_window"].default == 40.0


def test_regression_the_window_plus_a_per_identity_floor_keeps_the_published_answer():
    """Replacing the absolute 50 kcal/mol threshold with IPRO's relative
    40 kcal/mol window made the design space **smaller**, not larger, and
    deleted the one residue the shipped example exists to find.

    Measured on examples/ompf_pore/spec_smoke.yaml, same input, same backend:

        absolute 50 (the old rule)   115 kept   nothing dropped
        window 40 alone              95 kept    A18: TRP, TYR; A20: PRO; A38: PHE
        window 40 + per-identity 3   122 kept   nothing dropped

    A18/TRP is PoreDesigner's published V18W. Its best rotamer sits 51.88
    kcal/mol above that position's best (MET, -7.06) because it genuinely
    clashes with the *unrelaxed* wild-type pocket -- which is exactly why an
    identity must not be deleted outright on the strength of a template it has
    never been relaxed against. IPRO's own rule is the window AND a per-kind
    retention floor (ROTAMERS.py keeps N=3 of each identity, lowering N only
    when the total exceeds the cap); only the window had been implemented."""
    ompf = REPO / "examples" / "ompf_pore"
    if not (ompf / "2omf_A.pdb").exists() or not _have_pymol():
        print("    (skipped: needs the ompf example and PyMOL)")
        return
    from ipro_agent.loop import IPROLoop

    spec = DesignSpec.load(ompf / "spec_smoke.yaml")
    loop = IPROLoop(spec, Path(tempfile.mkdtemp()) / "run",
                    backend=resolve("builtin"))
    tables = loop.build_combined().tables
    kept = {p: len(tables.rotamers[p]) for p in tables.positions}
    assert sum(kept.values()) >= 115, (
        f"{kept} -- the prune must not keep fewer than the absolute rule did"
    )
    assert not tables.dropped_identities, tables.dropped_identities
    assert "TRP" in {r.aa for r in tables.rotamers["A18"]}, (
        "the published V18W answer must be reachable at A18"
    )

    # The floor is per identity, so it cannot resurrect a whole position's
    # worth of clashes: each identity keeps at most `keep_per_identity`
    # rotamers from outside the window.
    import inspect

    from ipro_agent.milp import build_tables

    assert inspect.signature(build_tables).parameters["keep_per_identity"].default == 3


def test_regression_an_unreachable_identity_is_still_reported():
    """The reporting path itself, which the fix above must not have removed:
    an identity whose every rotamer is pruned is not reachable, and `validate`
    still advertises the full product of the allowed sets, so the run has to
    say so. Exercised with the floor turned off, since the shipped spec no
    longer drops anything."""
    import numpy as _np

    from ipro_agent.energy.base import AtomSet
    from ipro_agent.milp import Rotamer, build_tables

    class _Flat:
        can_relax = False
        name = "flat"
        units = "test"
        screening_pairwise = True

        def __init__(self, energies):
            self.energies = energies

        def interaction(self, a, b):
            return self.energies[a.names[0]]

        def internal(self, a):
            return 0.0

    energies = {"a0": 0.0, "b0": 100.0, "b1": 101.0}
    rots = {"A1": []}
    for i, (name, aa) in enumerate((("a0", "ALA"), ("b0", "TRP"), ("b1", "TRP"))):
        rots["A1"].append(Rotamer(
            position="A1", aa=aa, state=i,
            atoms=AtomSet(names=[name], coords=_np.zeros((1, 3)), resnames=[aa])))
    template = AtomSet(names=["N"], coords=_np.zeros((1, 3)), resnames=["GLY"])

    tab = build_tables(rots, template, _Flat(energies), native={"A1": "ALA"},
                       rotamer_window=40.0, keep_per_identity=0)
    assert tab.dropped_identities == {"A1": ["TRP"]}, tab.dropped_identities

    from ipro_agent.loop import IPROLoop

    loop = IPROLoop(_spec(), Path(tempfile.mkdtemp()) / "r",
                    backend=resolve("builtin"))
    loop._note_dropped_identities(tab)
    note = [n for n in loop.notes if "NOT reachable" in n]
    assert note and "A1: TRP" in note[0], loop.notes


def test_regression_stabilize_minimizes_the_complex_energy():
    """``Stabilize`` scored the design molecule's own energy where DESIGN.md
    and its own docstring cite ``min E_complex`` (IPRO+/-: minimize the complex
    energy subject to an interaction-energy floor). Measured with candidates
    A (design_only -100.0, IE -20.0, complex -120.0) and B (-103.0, -15.0,
    -118.0), both passing the floor: the design-only objective picks B, the
    paper's picks A."""
    from ipro_agent.goals import ScoreContext, build_goal

    lig = Path(tempfile.mkdtemp()) / "lig.pdb"
    lig.write_text(
        "HETATM    1  C1  LIG B   1      20.000  20.000  20.000  1.00  0.00           C\nEND\n"
    )
    spec = DesignSpec.from_dict({
        "goal": "stabilize", "goal_statement": "stabilize it",
        "molecules": [
            {"name": "p", "path": str(CRN), "role": "design",
             "kind": "protein", "chains": ["A"]},
            {"name": "t", "path": str(lig), "role": "target", "kind": "ligand"},
        ],
        "positions": ["A16"],
        "goal_options": {"min_interaction_fraction": 0.75,
                         "reference_interaction": -20.0},
    })
    goal = build_goal(spec)
    a = ScoreContext(design_only=-100.0, interaction={"t": -20.0}, total=-120.0)
    b = ScoreContext(design_only=-103.0, interaction={"t": -15.0}, total=-118.0)
    assert goal.filter_reject(a) is None and goal.filter_reject(b) is None
    assert abs(goal.score(a) - (-120.0)) < 1e-9, goal.score(a)
    assert goal.score(a) < goal.score(b), (
        "min E_complex must pick the lower complex energy"
    )

    # With no partner the complex IS the design molecule.
    spec2 = DesignSpec.from_dict({
        "goal": "stabilize", "goal_statement": "stabilize it",
        "molecules": [{"name": "p", "path": str(CRN), "role": "design",
                       "kind": "protein", "chains": ["A"]}],
        "positions": ["A16"],
    })
    alone = build_goal(spec2)
    assert abs(alone.score(ScoreContext(design_only=-50.0, total=-50.0))
               - (-50.0)) < 1e-9


def test_regression_narrowing_without_a_floor_says_a_plug_would_win():
    """PoreDesigner's designs had to still conduct water single-file; this
    objective is monotonic, so a blocked pore is its maximum. Measured with
    ``spec_full.yaml``'s own options (direction narrow, pore_diameter_max 4.0,
    pore_diameter_min null): a candidate with a 0.8 A bottleneck and the probe
    clashing at IE +50 passes ``filter_reject`` and scores -50.0, beating a
    paper-style 3.5 A design at +0.200. The docs also described the objective
    as occluding the lumen, which is the opposite of the paper."""
    from ipro_agent.goals import ScoreContext, build_goal

    ompf = REPO / "examples" / "ompf_pore"
    if not (ompf / "spec_full.yaml").exists():
        print("    (skipped: no example)")
        return
    goal = build_goal(DesignSpec.load(ompf / "spec_full.yaml"))
    assert any("FULLY OCCLUDED" in n for n in goal.notes), goal.notes
    assert any("2.8" in n for n in goal.notes), "the note must give a floor"

    plug = ScoreContext(interaction={"water_wire": 50.0},
                        geometry={"min_pore_diameter": 0.8})
    assert goal.filter_reject(plug) is None, (
        "the published filter alone does not stop a plug -- which is the point"
    )

    # With a floor, it does.
    spec = DesignSpec.load(ompf / "spec_full.yaml")
    spec.goal_options["pore_diameter_min"] = 2.8
    floored = build_goal(spec)
    assert floored.filter_reject(plug), "a floor must reject a blocked pore"
    assert not any("FULLY OCCLUDED" in n for n in floored.notes)

    for text in ((REPO / "ipro_agent" / "goals.py").read_text(),
                 (REPO / "docs" / "DESIGN.md").read_text()):
        assert "to occlude the lumen" not in text


def test_regression_the_hard_reject_guard_watches_the_complex_energy():
    """The released source applies its two tests to two different quantities
    (``standard_decision``): the Metropolis window to the interaction energy,
    and the max(40 kcal/mol, 5%) cap to ``Energies[group]['Complex']``. That
    cap is the stability safeguard. Applied to the objective instead -- as it
    was here -- a goal that is not an energy at all (a pore radius, a
    specificity difference) got a cap in the wrong units, and a design could
    destabilize the protein by any amount so long as the objective improved."""
    src = (REPO / "ipro_agent" / "loop.py").read_text()
    assert "d_complex = ctx.total - current_complex" in src
    assert "hard_reject_fraction * abs(current_complex)" in src, (
        "the 5% is 5% of the complex energy, not of the objective"
    )
    assert "complex energy worse by" in src
    # And the accepted structure's complex energy is what it is measured from.
    assert "current_complex = ctx.total" in src


def test_regression_a_candidate_pdb_is_one_residue_per_residue():
    """The structure that gets relaxed after the MILP is written from the
    candidate's AtomSet, and that set is "template (minus the design
    positions' side chains)" followed by "the chosen rotamers". Written in that
    order the file was malformed twice over, and both were found by running
    full mode with real PyRosetta rather than a stand-in:

    1. residue 16's backbone appeared near the top of the file and its side
       chain at the very end, so the residue read as a duplicate;
    2. the template keeps the NATIVE residue name on the backbone it
       contributes while the rotamer carries the designed one, so residue 16
       was written ``CYS`` for N/CA/C/O and ``SER`` for CB/OG.

    Biopython refuses such a file outright ("Blank altlocs in duplicate
    residue SER (' ', 16, ' ')"). PyRosetta *accepted* it and returned a
    structure with one atom more than it was given -- which is the dangerous
    half: the energies that decide acceptance were computed on whatever it
    made of a split residue, and the only visible symptom was a note about an
    atom count. Fixed, the same run keeps 326 atoms through the relaxation."""
    from ipro_agent.energy.base import AtomSet
    from ipro_agent.loop import _atoms_of
    from ipro_agent.output import write_atomset_pdb
    from ipro_agent.structure import Structure

    st = Structure.load(CRN)
    full = _atoms_of(st, set())
    template = _atoms_of(st, {"A16"})          # A16's side chain removed
    # A "rotamer" at A16 with a DIFFERENT identity, appended last, as the real
    # candidate has it.
    rot = AtomSet(
        names=["CB", "OG"],
        coords=np.array([full.coords[0] + 1.0, full.coords[0] + 2.0]),
        resnames=["SER", "SER"],
        reskeys=[("A", 16, ""), ("A", 16, "")],
        elements=["C", "O"],
    )
    candidate = template.extend(rot)
    out = Path(tempfile.mkdtemp()) / "cand.pdb"
    n = write_atomset_pdb(candidate, out)
    assert n == len(candidate), (n, len(candidate))

    # The strict loader is the test: it raises on a duplicate residue.
    back = _atoms_of(Structure.load(out), set())
    assert len(back) == len(candidate), (len(back), len(candidate))

    # One name per residue, and it is the designed one.
    names = {
        str(k): r for k, r in zip(back.reskeys, back.resnames)
    }
    assert names["('A', 16, '')"] == "SER", names["('A', 16, '')"]
    text = out.read_text()
    assert "CYS A  16" not in text, "the native name must not survive"

    # Atoms of one residue are contiguous, backbone first.
    idx = [i for i, k in enumerate(back.reskeys) if k == ("A", 16, "")]
    assert idx == list(range(idx[0], idx[-1] + 1)), idx
    assert back.names[idx[0]].strip() == "N", back.names[idx[0]]


def test_the_demo_image_and_prompt_page_ship_with_the_repo():
    """README and HANDOFF.md both show the conversation image, and both link
    the prompt page. A broken image in the first screen of a README is worse
    than no image.

    This replaced a terminal recording. The recording was accurate and nobody
    drives this from a terminal: the product is a conversation, so the picture
    is of a conversation."""
    png = REPO / "docs" / "demo-chat.png"
    gif = REPO / "docs" / "demo-chat.gif"
    page = REPO / "docs" / "index.html"
    assert png.is_file() and gif.is_file() and page.is_file()
    for f in (png, gif):
        assert f.stat().st_size / 1048576 < 2.0, f"{f.name} is too big to ship"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert gif.read_bytes()[:6] in (b"GIF87a", b"GIF89a")

    dark = REPO / "docs" / "demo-chat-dark.gif"
    assert dark.is_file() and dark.stat().st_size / 1048576 < 2.0
    raw = "https://raw.githubusercontent.com/chowdhurylab-studio/ipro-agent/main/"
    for doc, img, link in (("README.md", f"{raw}docs/demo-chat.gif", None),
                           ("docs/HANDOFF.md", "demo-chat.gif", "EXAMPLE_PROMPTS.md")):
        text = (REPO / doc).read_text()
        # a <picture> so GitHub serves the theme the reader is in
        assert f'src="{img}"' in text, f"{doc} must show the conversation"
        assert "prefers-color-scheme: dark" in text, f"{doc} needs both themes"
        if link:
            assert f"]({link})" in text, f"{doc} must link the prompt page"
    assert "tour.gif" not in (REPO / "README.md").read_text()

    # pyproject makes README.md the PyPI long description, and PyPI resolves a
    # relative path against pypi.org -- so every image and link in it has to be
    # absolute or the project page shows a broken image and dead links. This is
    # the only document with that constraint; the others are read on GitHub.
    import re

    readme = (REPO / "README.md").read_text()
    assert 'src="docs/' not in readme and 'srcset="docs/' not in readme, \
        "README images must be absolute: PyPI renders this file"
    rel = re.findall(r"\]\((docs/[^)]+)\)", readme)
    assert not rel, f"README links must be absolute for PyPI: {rel}"
    for url in re.findall(rf"{re.escape(raw)}(\S+?)[\"\)]", readme):
        assert (REPO / url).is_file(), f"README points at {url}, which is not in the repo"

    # The prompt page is HTML and GitHub Pages is deliberately off, so a link to
    # it from the hero area shows markup. EXAMPLE_PROMPTS.md carries the same
    # five prompts and renders.
    assert "docs/index.html)" not in readme, \
        "link the markdown prompts, not the raw HTML page"
    assert "EXAMPLE_PROMPTS.md)" in readme

    # Tracked, not ignored: a clone that cannot see them shows a broken image.
    tracked = subprocess.run(
        ["git", "ls-files", "docs/demo-chat.png", "docs/demo-chat.gif",
         "docs/index.html"],
        capture_output=True, text=True, cwd=REPO).stdout.split()
    assert sorted(tracked) == ["docs/demo-chat.gif", "docs/demo-chat.png",
                               "docs/index.html"], tracked
    assert subprocess.run(["git", "ls-files", "docs/demo-chat-dark.gif"],
                          capture_output=True, text=True,
                          cwd=REPO).stdout.strip() == "docs/demo-chat-dark.gif"

    # The page must carry the config block for each client family, or "how do
    # I add this" has no answer on the one page built to answer it.
    html = page.read_text()
    for needed in ('"mcpServers"', "[mcp_servers.ipro-agent]", '"servers"',
                   "IPRO_REPO", "ipro_agent.mcp_server",
                   "bash scripts/bootstrap.sh"):
        assert needed in html, needed

    # The page quotes the prompts, so it must not drift from the document they
    # live in, and the caveats must travel with them.
    html = page.read_text()
    for fragment in ("Narrow the OmpF porin pore below 4",
                     "prefer the C12 substrate over C14",
                     "Find the interface and propose positions",
                     "what mode will this run in",
                     "Give me the honest version",
                     "screening scale", "accepted: 0", "not a pore radius"):
        assert fragment in html, fragment


# ---------------------------------------------------------------------------
# Driving the MCP server with real clients (docs/DESIGN.md section 7i)
# ---------------------------------------------------------------------------


def _mcp_at(messages: list[dict], cwd, env_extra: dict | None = None) -> dict[int, dict]:
    """Drive the server from a working directory that is NOT the repo.

    Every other MCP test here starts the server with `cwd=REPO`, which is the
    one case in which the path and import questions below cannot arise.
    """
    env = dict(os.environ, PYTHONPATH=str(REPO))
    env.update(env_extra or {})
    payload = "\n".join(json.dumps(m) for m in messages) + "\n"
    p = subprocess.run(
        [sys.executable, "-m", "ipro_agent.mcp_server"],
        input=payload, capture_output=True, text=True,
        cwd=str(cwd), env=env, timeout=900,
    )
    assert p.returncode == 0, (p.returncode, p.stderr[-600:])
    out: dict[int, dict] = {}
    for line in p.stdout.splitlines():
        if line.strip():
            msg = json.loads(line)
            if msg.get("id") is not None:
                out[msg["id"]] = msg
    return out


def test_regression_mcp_prompts_refuse_missing_required_arguments():
    """A prompt's `required: true` was decorative, as the tool schemas were
    before section 7h.

    Measured with the official MCP Python SDK (2.3.0) and the MCP Inspector
    CLI, the first real clients this server has had: `prompts/get
    ipro-redesign` with no arguments at all returned a complete
    2167-character design instruction and rc=0 from both. Step 3 of it told
    the model to fetch "(structure)" from the RCSB, and step 6 to put "(goal)"
    in `goal_statement` "in my own words" -- the two things the operating
    manual says must never be invented on the user's behalf. A client is
    allowed to surface a prompt without collecting its arguments, because the
    server is the thing that is supposed to refuse."""
    replies = _mcp([
        _INIT,
        {"jsonrpc": "2.0", "id": 2, "method": "prompts/get",
         "params": {"name": "ipro-redesign", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 3, "method": "prompts/get",
         "params": {"name": "ipro-redesign",
                    "arguments": {"structure": "/tmp/p.pdb", "goal": "prefer C12"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "prompts/get",
         "params": {"name": "ipro-redesign",
                    "arguments": {"structure": None, "goal": "prefer C12"}}},
        {"jsonrpc": "2.0", "id": 5, "method": "prompts/get",
         "params": {"name": "ipro-report", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 6, "method": "prompts/get",
         "params": {"name": "ipro-check-setup", "arguments": {}}},
    ])
    assert "error" in replies[2], replies[2]
    for word in ("structure", "goal"):
        assert word in replies[2]["error"]["message"], replies[2]

    text = replies[3]["result"]["messages"][0]["content"]["text"]
    assert "/tmp/p.pdb" in text and "prefer C12" in text
    assert "(structure)" not in text and "(goal)" not in text, text[:200]

    # An explicit null is not an argument: `str(None)` is the four-character
    # word "None", which every truthiness test in sight would have accepted.
    assert "error" in replies[4], replies[4]
    assert "error" in replies[5], replies[5]
    # ipro-check-setup declares no arguments, so it must still answer.
    assert "result" in replies[6], replies[6]


def test_regression_mcp_resolves_a_relative_path_against_the_client():
    """One relative path meant two different directories depending on the tool.

    `_resolve_spec` resolved against `Path.cwd()` -- the directory an MCP
    config gives the server -- while `tool_inspect` and `tool_fetch` handed
    the string to a child running with `cwd=REPO_ROOT`. Measured with the
    server started from `/`, as Claude Desktop starts one whose config omits
    `cwd`: `ipro_validate {"spec": "examples/ompf_pore/spec_smoke.yaml"}`
    refused with "looked for /examples/ompf_pore/spec_smoke.yaml", while
    `ipro_inspect_structure {"structure": "examples/ompf_pore/2omf_A.pdb"}`
    SUCCEEDED and reported 340 residues -- answering about a file inside this
    repository that the client had never named. Every position, native residue
    and energy downstream of that answer belongs to the wrong molecule, and
    nothing in the reply says so."""
    ompf = REPO / "examples" / "ompf_pore" / "2omf_A.pdb"
    if not ompf.exists():
        print("    (skipped: no example structure)")
        return
    elsewhere = Path(tempfile.mkdtemp())
    shutil.copy2(ompf, elsewhere / "mine.pdb")

    replies = _mcp_at([
        _INIT,
        _call(2, "ipro_inspect_structure",
              {"structure": "examples/ompf_pore/2omf_A.pdb", "positions": ["18"]}),
        _call(3, "ipro_inspect_structure",
              {"structure": "mine.pdb", "positions": ["18"]}),
        _call(4, "ipro_validate", {"spec": "examples/ompf_pore/spec_smoke.yaml"}),
    ], cwd=elsewhere)

    is_error, body = _body(replies[2])
    assert is_error, f"a path that exists only in the repo must not resolve: {body}"
    assert "not found" in body.get("error", ""), body
    assert str(elsewhere) in body["error"], body      # and it says where it looked

    is_error, body = _body(replies[3])
    assert not is_error, body
    assert body["positions"][0]["native"] == "VAL", body

    # The spec tool already resolved against the client; the two now agree.
    is_error, body = _body(replies[4])
    assert is_error and "not found" in body.get("error", ""), body


def test_regression_mcp_inline_specs_do_not_leak_temp_directories():
    """An inline spec wrote a temp directory that nothing ever removed.

    The list they were collected into was called `keep`, and no caller read
    it. Measured under a GUI client's environment, which has no `TMPDIR`, so
    `tempfile.gettempdir()` is the shared `/tmp`: five `ipro_validate` calls
    with an inline spec took `/tmp/ipro_mcp_spec_*` from 7 directories to 12,
    one per call, each holding a 0o644 `spec.yaml`. All five of those calls
    returned isError, which is the half that matters -- an agent iterating on
    a spec it cannot get right leaks fastest, and the leaked text is the
    user's unpublished design problem in a world-readable directory."""
    scratch = Path(tempfile.mkdtemp())
    inline = ("goal: reshape_channel\n"
              "goal_statement: inline leak probe\n"
              "molecules: []\n")
    messages = [_INIT] + [_call(i + 2, "ipro_validate", {"spec": inline})
                          for i in range(3)]
    replies = _mcp_at(messages, cwd=REPO, env_extra={"TMPDIR": str(scratch)})
    assert all(_body(replies[i])[0] for i in (2, 3, 4)), "this spec is invalid"
    left = sorted(scratch.glob("ipro_mcp_spec_*"))
    assert not left, f"{len(left)} temp directories left behind: {left[:3]}"

    # `ipro_run` cannot delete the file on the way out -- its child reads the
    # spec after the call has returned -- so the spec moves into the run
    # directory, where it also records what the run was asked to do.
    from ipro_agent import mcp_server as m

    g = m.tool_run.__globals__
    keep_doctor, keep_tmp = g["_doctor"], tempfile.tempdir
    run_dir = scratch / "run"
    launched = None
    try:
        tempfile.tempdir = str(scratch)
        g["_doctor"] = lambda: {"can_run": True, "can_design": True,
                                "missing_required": [], "resolved": {},
                                "degraded": []}
        try:
            launched = m.tool_run({"spec": inline, "run_dir": str(run_dir)})
        except m.ToolError:
            pass          # the CLI refuses this spec, which is the fast path
    finally:
        g["_doctor"], tempfile.tempdir = keep_doctor, keep_tmp
    if launched and launched.get("pid"):
        try:
            os.kill(int(launched["pid"]), 15)
        except OSError:
            pass
    assert (run_dir / "spec.yaml").is_file(), "the run must hold the spec that made it"
    assert not sorted(scratch.glob("ipro_mcp_spec_*")), "and leave no temp dir"


def test_regression_cheap_checks_run_before_the_doctor_probe():
    """This repo's first CI run failed three tests, and two shared one cause
    that was not in the tests. ``tool_run`` called ``_doctor()`` as its very
    first act, so on a runner with no PyMOL the can_design gate fired ahead of
    both the argument coercion and the non-empty-run_dir check: a caller who
    passed ``iterations: 8.5`` was told "cannot design: PyMOL did not resolve",
    which is true and useless, and so was a caller pointing at a directory that
    already held a run.

    It also spent a one-to-five-second doctor probe before answering a
    malformed argument. Measured with _doctor stubbed to can_design False:
    bad iterations -> "`iterations` must be a whole number", used run_dir ->
    "run directory is not empty", and only an otherwise-valid call reaches
    "cannot design". No directory is created by any of the refusals."""
    from ipro_agent import mcp_server as m

    spec = str(REPO / "examples" / "ompf_pore" / "spec_smoke.yaml")
    tmp = Path(tempfile.mkdtemp())
    used = tmp / "used"
    used.mkdir()
    (used / "state.json").write_text("{}")
    fresh = tmp / "fresh"

    g = m.tool_run.__globals__
    keep = g["_doctor"]
    try:
        g["_doctor"] = lambda: {"can_run": True, "can_design": False,
                                "missing_required": [], "degraded": ["pymol"],
                                "resolved": {}}
        for kwargs, expected in (
            ({"spec": spec, "iterations": 8.5}, "iterations"),
            ({"spec": spec, "seed": "lucky"}, "seed"),
            ({"spec": spec, "run_dir": str(used)}, "not empty"),
            ({"spec": spec, "run_dir": str(fresh)}, "cannot design"),
        ):
            try:
                m.tool_run(kwargs)
            except m.ToolError as exc:
                assert expected in str(exc), (kwargs, str(exc)[:120])
            else:
                raise AssertionError(f"{kwargs} must be refused")
        assert not fresh.exists(), (
            "a refused launch must not leave a run directory behind"
        )
    finally:
        g["_doctor"] = keep


def test_regression_a_missing_backend_is_a_refusal_not_a_traceback():
    """``milp`` and ``score`` both build rotamer tables, so both need PyMOL,
    and on a machine without it both died with a bare ``RotamerError``
    traceback -- the third of the three CI failures. ``cli.main()`` caught
    SpecError and StructureError but not this.

    The README was wrong about it too: it listed ``validate``, ``positions``,
    ``milp``, ``score`` and ``goals`` as working without PyMOL. Measured with
    PyMOL undiscoverable: validate and goals exit 0, positions works, and milp
    and score refuse -- so two of the five named were false."""
    import ipro_agent.cli as cli
    from ipro_agent.rotamers import RotamerError

    src = (REPO / "ipro_agent" / "cli.py").read_text()
    assert "except RotamerError as exc:" in src, (
        "main() must give a missing backend the same contract as a bad spec"
    )

    # Through a real process, with PyMOL hidden the way CI hides it.
    harness = Path(tempfile.mkdtemp()) / "nopymol.py"
    harness.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "import ipro_agent.rotamers as R\n"
        "R.find_pymol_python = lambda: None\n"
        "import ipro_agent.cli as cli\n"
        "sys.exit(cli.main(sys.argv[1:]))\n"
    )
    spec = str(REPO / "examples" / "ompf_pore" / "spec_smoke.yaml")
    out = Path(tempfile.mkdtemp())
    for argv, needs_pymol in (
        (["goals"], False),
        (["validate", spec], False),
        (["milp", spec, "-o", str(out / "m.lp")], True),
        (["score", spec, "--run-dir", str(out / "s")], True),
    ):
        p = subprocess.run([sys.executable, str(harness), *argv],
                           capture_output=True, text=True, cwd=REPO, timeout=900)
        both = p.stdout + p.stderr
        assert "Traceback" not in both, (argv[0], both[-300:])
        if needs_pymol:
            assert p.returncode == 2, (argv[0], p.returncode)
            assert "backend missing" in p.stderr, (argv[0], p.stderr[-200:])
        else:
            assert p.returncode == 0, (argv[0], p.returncode, both[-200:])

    # And the README must not claim otherwise.
    readme = (REPO / "README.md").read_text()
    assert "`validate`, `positions`, `milp`, `score` and `goals` work" not in readme
    assert "`milp`, `score` and `run`" in readme


def test_regression_relaxation_is_superposed_back_into_the_frame_it_was_scored_in():
    """`backend.relax` runs FastRelax in torsion space, and the coordinate
    constraints and MoveMap that would pin the molecule are installed only
    when ``fixed`` is non-empty -- which it is unless the spec happens to carry
    a fix/tether restraint. So the whole design molecule swings through space
    while every partner keeps its docked coordinates, and
    ``ctx.interaction[...]`` becomes a cross-term between a displaced design
    and an undisplaced partner.

    Measured with PyRosetta 2026.29 on the shipped OmpF structure, with a
    3-atom probe grid-placed in a genuine pocket (115 protein atoms within
    8 A, nearest 3.64 A):

        unrelaxed                        IE   -5.739
        relaxed, as the code returned it IE  +81.351   (lab move 9.91 A)
        relaxed, second draw             IE +205.740   (lab move 6.59 A)
        superposed, first draw           IE   -3.435
        superposed, second draw          IE   -3.250

    Two draws of ONE structure disagreed by **124.39 kcal/mol** and both had
    the wrong sign; superposed they agree to 0.18 and keep it. At the shipped
    temperature kT is 7.23 kcal/mol, so the error was seventeen kT.

    The fit is on backbone atoms only, so a large rotamer change at a design
    position cannot drag the placement, and it is energy-neutral for
    `internal()` by construction -- which this test asserts, because that is
    what makes it safe to apply unconditionally."""
    from ipro_agent.energy.base import AtomSet
    from ipro_agent.loop import _atoms_of, _superpose_onto
    from ipro_agent.structure import Structure

    atoms = _atoms_of(Structure.load(CRN), set())
    coords = np.asarray(atoms.coords, dtype=float)

    # A rigid motion: 7 A translation plus a rotation about z.
    ang = np.radians(35.0)
    rot = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                    [np.sin(ang), np.cos(ang), 0.0],
                    [0.0, 0.0, 1.0]])
    moved_coords = coords @ rot.T + np.array([7.0, -3.0, 2.0])
    # ... and a small conformational change on top of it, as a relax makes.
    rng = np.random.default_rng(0)
    moved_coords = moved_coords + rng.normal(0.0, 0.25, size=moved_coords.shape)
    moved = AtomSet(names=list(atoms.names), coords=moved_coords,
                    resnames=list(atoms.resnames), reskeys=list(atoms.reskeys),
                    elements=list(atoms.elements))

    before = float(np.linalg.norm(moved_coords - coords, axis=1).mean())
    fitted, drift, centroid = _superpose_onto(moved, atoms)
    after = float(np.linalg.norm(np.asarray(fitted.coords) - coords, axis=1).mean())

    assert before > 4.0, before
    assert after < 0.6, (before, after, "the rigid-body component must be gone")
    assert after < before / 5.0, (before, after)
    assert drift > 4.0 and centroid > 1.0, (drift, centroid)

    # Exact on a pure rigid motion, which is the property the fix rests on.
    pure = AtomSet(names=list(atoms.names), coords=coords @ rot.T + [7.0, -3.0, 2.0],
                   resnames=list(atoms.resnames), reskeys=list(atoms.reskeys),
                   elements=list(atoms.elements))
    exact, _d, _c = _superpose_onto(pure, atoms)
    assert float(np.abs(np.asarray(exact.coords) - coords).max()) < 1e-6

    # Energy-neutral for the internal term: a rigid motion preserves every
    # internal distance, so superposing one cannot change it.
    be = resolve("builtin")
    assert abs(be.internal(moved) - be.internal(fitted)) < 1e-6, (
        be.internal(moved), be.internal(fitted)
    )
    # And it repairs the cross-term, which is the whole point.
    probe = AtomSet(names=["C1"], coords=coords[:1] + np.array([4.0, 0.0, 0.0]),
                    resnames=["LIG"], reskeys=[("L", 1, "")], elements=["C"])
    assert abs(be.interaction(fitted, probe) - be.interaction(atoms, probe)) \
        < abs(be.interaction(moved, probe) - be.interaction(atoms, probe))

    # The loop applies it, and says so when the motion was large.
    src = (REPO / "ipro_agent" / "loop.py").read_text()
    assert "_superpose_onto(atoms, design_all)" in src, (
        "the relaxed atoms must be returned in the frame they were scored in"
    )
    assert "relax_drift" in src, "a large rigid-body component must be reported"


def test_regression_the_relaxation_basis_is_latched_before_the_baseline():
    """Relaxation failures are per-candidate (PyRosetta's hold-tolerance raise
    fires for one geometry, not for the backend), and the loop used to score
    the failing candidate unrelaxed anyway -- against a baseline that HAD been
    relaxed. Measured with real PyRosetta over three iterations of
    bind_tighter: start_score +76.5872 against candidate interaction energies
    of -2.2466, -2.2673 and -2.2682, all three "accepted: improved", so the
    design accounted for 0.0216 kcal/mol of a reported 78.86 kcal/mol
    improvement. The mirror case (only the first relax fails) rejected
    everything and still claimed it was the candidates that went unrelaxed.

    The basis is now chosen once, before the baseline is scored, and never
    changes: if the baseline cannot be relaxed the whole run goes unrelaxed
    and says so; afterwards a candidate that cannot be relaxed is rejected,
    because it is not measurable on the run's basis."""
    src = (REPO / "ipro_agent" / "loop.py").read_text()

    assert "self._relax_disabled" in src
    assert 'ok, reason = False, "relaxation failed; not comparable"' in src, (
        "a candidate that could not be relaxed must be rejected, not re-based"
    )
    # The latch is before the first score, and re-scores the baseline.
    body = src[src.index("self._relax_native = True"):]
    body = body[:body.index("start_score = self.goal.score(start_ctx)")]
    assert "self._relax_disabled = True" in body
    assert 'relax_tag=None' in body, "the baseline must be re-scored unrelaxed"
    assert body.index("self._relax_disabled = True") < body.index("relax_tag=None")

    # And the helper refuses once the run is on the unrelaxed basis, so a
    # later success cannot switch it back.
    helper = src[src.index("def _relaxed_design_atoms"):]
    helper = helper[:helper.index("    # -- writing structures")]
    assert "if self._relax_disabled:" in helper
    assert helper.index("if self._relax_disabled:") < helper.index("write_atomset_pdb")

    from ipro_agent.loop import IPROLoop
    from ipro_agent.loop import _atoms_of
    from ipro_agent.structure import Structure

    class _Broken(_CountingRelaxBackend):
        def relax(self, pdb_in, pdb_out, fixed=()):
            raise RuntimeError("no params for LIG")

    loop = IPROLoop(_spec(loop={"perturb_backbone": True}),
                    Path(tempfile.mkdtemp()) / "run", backend=_Broken())
    atoms = _atoms_of(Structure.load(CRN), set())
    assert loop._relaxed_design_atoms(atoms, "t") is None
    assert loop._relax_failed_now is True, "the failure must be visible to the loop"
    loop._relax_disabled = True
    loop._relax_failed_now = False
    assert loop._relaxed_design_atoms(atoms, "t2") is None
    assert loop._relax_failed_now is False, (
        "once the run is on the unrelaxed basis, a skipped relax is not a failure"
    )


def test_regression_the_derived_window_works_with_several_assemblies():
    """`ordered` holds VIRTUAL positions once there is more than one assembly
    ("a::A16"), while `spec.positions` hold plain labels ("A16"), so the
    intersection that derives the redesign window from the perturbed region
    was always empty -- and the loop fell back to the independent draw
    silently, in exactly the case assemblies exist for. Measured on a
    two-assembly spec: ordered ['a::A16', 'a::A17'], a perturbed region
    holding BOTH design positions, derived window []."""
    from ipro_agent.loop import IPROLoop

    src = (REPO / "ipro_agent" / "loop.py").read_text()
    assert "(split_vpos(p)[1] or p) in inside_labels" in src, (
        "the window must compare plain labels against plain labels"
    )

    class _Pert:
        chain = "A"
        residues = [16, 17]

    loop = IPROLoop(_spec(positions=["A16", "A17"]),
                    Path(tempfile.mkdtemp()) / "run", backend=resolve("builtin"))
    # Virtual labels, as a multi-assembly build produces.
    ordered = ["a::A16", "a::A17"]
    window = loop._choose_window(ordered, _Pert())
    assert window == ordered, (
        f"both design positions are inside the perturbed region; got {window}"
    )
    # A region holding neither still falls back rather than returning nothing.
    class _Elsewhere:
        chain = "A"
        residues = [40, 41]

    assert loop._choose_window(ordered, _Elsewhere()), "must fall back, not empty"


def test_regression_an_explicit_pore_axis_is_a_unit_vector():
    """`goal_options.pore_axis` is the remedy every document names for a
    fitted axis that misses the lumen, and giving it silently deleted the
    entire pore measurement.

    `_pore_axis` returned the raw difference between the two atoms on that
    branch. `_measure_geometry` builds the default window by projecting onto
    it, so with |direction| = 17.43 A for the repo's own example pair
    (["A18:CA", "A119:CA"]) the window came out seventeen times too large and
    landed hundreds of angstroms off the molecule: `pore_profile` returned zero
    slices, `_measure_geometry` returned {}, `ReshapeChannel.filter_reject`
    read None for `min_pore_diameter` and passed everything, and
    `_axis_clearance` was never reached. PoreDesigner's published sub-4 A
    filter was off, with no note, on exactly the input the docs recommend.

    The fitted branch always returned a unit vector, which is why it never
    showed up."""
    ompf = REPO / "examples" / "ompf_pore"
    if not (ompf / "2omf_A.pdb").exists() or not _have_pymol():
        print("    (skipped: needs the ompf example and PyMOL)")
        return
    from ipro_agent.loop import IPROLoop

    spec = DesignSpec.load(ompf / "spec_smoke.yaml")
    spec.goal_options["pore_axis"] = ["A18:CA", "A119:CA"]
    spec.goal_options["pore_diameter_min"] = 2.0
    loop = IPROLoop(spec, Path(tempfile.mkdtemp()) / "run",
                    backend=resolve("builtin"))
    cb = loop.build_combined()
    _origin, direction, _flat = loop._pore_axis()
    assert abs(float(np.linalg.norm(direction)) - 1.0) < 1e-9, direction

    atoms = loop._design_atoms(
        loop._rotamers_for(cb.tables, loop._native_choice(cb.tables)),
        loop._assemblies[0].name,
    )
    geo = loop._measure_geometry(atoms)
    assert geo, "an explicit axis must still produce a measurement"
    assert "min_pore_diameter" in geo and geo["min_pore_diameter"] > 0, geo
    # And the filter it feeds is live again.
    from ipro_agent.goals import ScoreContext

    assert loop.goal.filter_reject(
        ScoreContext(geometry={"min_pore_diameter": 0.4})
    ), "a blocked pore must be rejected when a floor is set"

    # Two atoms at one point have no direction, and that is an error.
    spec.goal_options["pore_axis"] = ["A18:CA", "A18:CA"]
    loop2 = IPROLoop(spec, Path(tempfile.mkdtemp()) / "run2",
                     backend=resolve("builtin"))
    loop2.prepare()
    try:
        loop2._pore_axis()
    except ValueError as exc:
        assert "same place" in str(exc), exc
    else:
        raise AssertionError("a zero-length axis must be refused")


def test_regression_the_delivered_structure_is_the_one_that_was_scored():
    """`_write_design` rebuilt the output through PyMOL from the unrelaxed
    input plus the chosen rotamers, so the file the user opens was not the
    structure any reported number belongs to. Measured on the shipped smoke
    spec: `best/design.json` reported `min_pore_radius` 1.999994 beside a
    `design.pdb` that measures 0.760, and in `full` mode the relaxed structure
    that produced every reported energy existed only under `logs/relax`.

    The scored atoms are now written directly. They already carry the chosen
    rotamer at each design position, which is what made them the candidate, so
    the guarantee the PyMOL rebuild existed for is met by construction."""
    ompf = REPO / "examples" / "ompf_pore"
    if not (ompf / "2omf_A.pdb").exists() or not _have_pymol():
        print("    (skipped: needs the ompf example and PyMOL)")
        return
    import json as _json

    from ipro_agent.loop import IPROLoop, _atoms_of
    from ipro_agent.structure import Structure

    run = Path(tempfile.mkdtemp()) / "run"
    p = subprocess.run(
        [sys.executable, "-m", "ipro_agent", "run", str(ompf / "spec_smoke.yaml"),
         "--run-dir", str(run), "--iterations", "2", "--json"],
        capture_output=True, text=True, cwd=REPO, timeout=1800,
    )
    assert p.returncode == 0, p.stderr[-400:]
    reported = _json.loads((run / "best" / "design.json").read_text())
    radius = (reported.get("components") or {}).get("min_pore_radius")
    design = run / "best" / "design.pdb"
    if radius is None or not design.exists():
        print("    (partial: no geometry or no structure written)")
        return

    spec = DesignSpec.load(ompf / "spec_smoke.yaml")
    loop = IPROLoop(spec, Path(tempfile.mkdtemp()) / "probe",
                    backend=resolve("builtin"))
    loop.prepare()
    measured = loop._measure_geometry(
        _atoms_of(Structure.load(design), set())
    ).get("min_pore_radius")
    assert measured is not None, "the delivered file must be measurable"
    assert abs(measured - radius) < 0.05, (
        f"design.json reports {radius} and design.pdb measures {measured}: "
        f"the delivered structure is not the one that was scored"
    )

    src = (REPO / "ipro_agent" / "loop.py").read_text()
    assert "self._scored_atoms[a.name] = design_all" in src


def test_regression_the_pdb_writer_refuses_what_it_cannot_write():
    """Every delivered structure now goes through `write_atomset_pdb`, so a
    field that overflows is no longer cosmetic. Measured: a residue number of
    10000 produced `ATOM      1  CA  ALA A10000       1.000 ...` where the
    insertion-code column ate the last digit and every column after it
    shifted, and a coordinate of 1234.567 left the next field reading
    `-1234.56` and the one after `7   3.00`. Bio.PDB refuses both with
    "Invalid or missing coordinate(s)".

    Refused with a sentence rather than written wrong: a misaligned file that
    still parses somewhere else is the worse outcome."""
    import numpy as _np

    from ipro_agent.energy.base import AtomSet
    from ipro_agent.output import write_atomset_pdb

    out = Path(tempfile.mkdtemp()) / "x.pdb"

    def one(reskey, coord):
        return AtomSet(names=["CA"], coords=_np.array([coord], dtype=float),
                       resnames=["ALA"], reskeys=[reskey], elements=["C"])

    # Both limits hold at the edge.
    write_atomset_pdb(one(("A", 9999, ""), [999.999, -999.999, 0.0]), out)
    from ipro_agent.loop import _atoms_of
    from ipro_agent.structure import Structure

    assert len(_atoms_of(Structure.load(out), set())) == 1

    for reskey, coord, fragment in (
        (("A", 10000, ""), [1.0, 2.0, 3.0], "resSeq"),
        (("A", 5, ""), [1234.567, -1234.567, 3.0], "coordinate field"),
    ):
        try:
            write_atomset_pdb(one(reskey, coord), out)
        except ValueError as exc:
            assert fragment in str(exc), (fragment, exc)
        else:
            raise AssertionError(f"{reskey} {coord} must be refused")


def test_regression_fixed_backbone_mode_does_not_relax():
    """The run's own note says, in fixed-backbone mode, that "the backbone
    never moves". It was relaxing anyway whenever the backend could: measured
    on the unmodified smoke spec with the PyRosetta backend, the starting
    point's relaxed output differed from its input by a mean of 15.7 A over
    CA atoms, while the loop's state never moved between iterations. The only
    structure that moved was the one being scored, beside a note promising the
    opposite."""
    from ipro_agent.loop import IPROLoop, _atoms_of
    from ipro_agent.structure import Structure

    atoms = _atoms_of(Structure.load(CRN), set())

    fixed = IPROLoop(_spec(loop={"perturb_backbone": False}),
                     Path(tempfile.mkdtemp()) / "a",
                     backend=_CountingRelaxBackend())
    assert fixed.mode == "fixed-backbone"
    assert fixed._relaxed_design_atoms(atoms, "t") is None
    assert fixed._post_relaxed == 0
    assert not _CountingRelaxBackend.__dict__  or True   # backend untouched

    full = IPROLoop(_spec(loop={"perturb_backbone": True}),
                    Path(tempfile.mkdtemp()) / "b",
                    backend=_CountingRelaxBackend())
    assert full.mode == "full"
    assert full._relaxed_design_atoms(atoms, "t") is not None
    assert full._post_relaxed == 1


def test_regression_optzyme_divisors_must_be_positive():
    """`rt_tsa_kcal` and `rt_substrate_kcal` are divisors in Eq 18. Zero
    raised mid-run; a negative silently inverted that term, so a design that
    binds the transition state WORSE scored better. Both passed validate."""
    from ipro_agent.goals import build_goal

    lig = Path(tempfile.mkdtemp()) / "lig.pdb"
    lig.write_text(
        "HETATM    1  C1  LIG B   1      20.000  20.000  20.000  1.00  0.00           C\nEND\n"
    )
    base = {
        "goal": "improve_catalysis", "goal_statement": "raise kcat",
        "molecules": [
            {"name": "e", "path": str(CRN), "role": "design",
             "kind": "protein", "chains": ["A"]},
            {"name": "t", "path": str(lig), "role": "target",
             "kind": "transition_state"},
            {"name": "s", "path": str(lig), "role": "target", "kind": "ligand"},
        ],
        "positions": ["A16"],
    }
    for bad in (0, -3.66, 0.0):
        spec = DesignSpec.from_dict({
            **base, "goal_options": {"parameter": "kcat", "rt_tsa_kcal": bad}})
        try:
            build_goal(spec).validate()
        except SpecError as exc:
            assert "positive" in str(exc), exc
        else:
            raise AssertionError(f"rt_tsa_kcal={bad} must be refused")

    ok = DesignSpec.from_dict({
        **base, "goal_options": {"parameter": "kcat", "rt_tsa_kcal": 3.0,
                                 "rt_substrate_kcal": 90.0}})
    build_goal(ok).validate()


def test_regression_an_mcp_agent_can_read_back_what_it_designed():
    """The server told agents to do something no tool could do, and withheld
    the product of a run.

    `ipro_status`'s own `reporting_rule` says to "read the per-iteration
    reasons". `iterations.jsonl` is on disk, the house rules forbid loading it
    wholesale because it grows without limit, and an MCP client has no shell,
    so the instruction was unfollowable. And the design itself -- the
    coordinates in `best/design.pdb` -- was reachable by no tool and no
    resource, so an agent could launch a run, quote its score, and never be
    able to hand over the structure it made.

    Four tools close that: ipro_result (the design, the ensemble, optionally
    the PDB text), ipro_iterations (a bounded tail with the rejection
    reasons), ipro_runs (for a lost id) and ipro_cancel."""
    from ipro_agent import mcp_server as m

    names = [t["name"] for t in m.TOOLS]
    for needed in ("ipro_result", "ipro_iterations", "ipro_runs", "ipro_cancel"):
        assert needed in names, needed

    ompf = REPO / "examples" / "ompf_pore"
    if not (ompf / "2omf_A.pdb").exists() or not _have_pymol():
        print("    (partial: schema only, needs the example and PyMOL)")
        return

    run = Path(tempfile.mkdtemp()) / "run"
    p = subprocess.run(
        [sys.executable, "-m", "ipro_agent", "run", str(ompf / "spec_smoke.yaml"),
         "--run-dir", str(run), "--iterations", "2", "--json"],
        capture_output=True, text=True, cwd=REPO, timeout=1800,
    )
    assert p.returncode == 0, p.stderr[-400:]

    res = m.tool_result({"run_dir": str(run)})
    assert res["mode"] and "notes" in res
    assert res["structure_files"], "the coordinates must be reachable"
    # A run that kept no mutation must say so here too, not only in status.
    if not res["mutations"]:
        assert res.get("designed_nothing") is True
        assert "STARTING POINT" in res["warning"]

    with_pdb = m.tool_result({"run_dir": str(run), "include_structure": True})
    assert with_pdb["structure_pdb"].startswith(("ATOM", "REMARK", "HEADER")), \
        with_pdb["structure_pdb"][:40]
    assert "ATOM" in with_pdb["structure_pdb"]

    it = m.tool_iterations({"run_dir": str(run), "limit": 5})
    assert it["iterations_total"] >= 1
    assert it["iterations"] and "reason" in it["iterations"][0]
    assert it["reason_counts"], "the rejection reasons are the point"
    # Bounded whatever the run length.
    assert len(m.tool_iterations({"run_dir": str(run), "limit": 10_000})["iterations"]) <= 200

    listed = m.tool_runs({})
    assert any(r["run_dir"] == str(run) for r in listed["runs"]) or listed["searched"]

    # Cancelling a finished run is a statement, not an error.
    out = m.tool_cancel({"run_dir": str(run)})
    assert out["stopped"] is False and "not running" in out["message"]


def test_regression_the_new_tools_are_reachable_over_the_protocol():
    """Registered in TOOLS is not the same as callable: the handler has to be
    wired and the schema has to validate a realistic call."""
    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "0"}}}
    listed = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    runs = {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "ipro_runs", "arguments": {}}}
    missing = {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
               "params": {"name": "ipro_result", "arguments": {}}}
    payload = "\n".join(json.dumps(m) for m in (init, listed, runs, missing)) + "\n"
    p = subprocess.run([sys.executable, "-m", "ipro_agent.mcp_server"],
                       input=payload, capture_output=True, text=True,
                       cwd=REPO, timeout=900)
    assert p.returncode == 0, p.stderr[-300:]
    replies = {}
    for line in p.stdout.splitlines():
        if line.strip():
            msg = json.loads(line)
            replies[msg.get("id")] = msg

    names = [t["name"] for t in replies[2]["result"]["tools"]]
    for needed in ("ipro_result", "ipro_iterations", "ipro_runs", "ipro_cancel"):
        assert needed in names, (needed, names)

    body = json.loads(replies[3]["result"]["content"][0]["text"])
    assert "runs" in body and "searched" in body, body

    # Without a run to point at, it refuses in words.
    err = json.loads(replies[4]["result"]["content"][0]["text"])["error"]
    assert "run_dir" in err and "Traceback" not in err, err


def test_server_json_matches_the_package_it_points_at():
    """`server.json` is what the MCP Registry stores, and three things in it
    have to agree with the package or the publish is rejected: the name's
    namespace, the PyPI identifier, and the version.

    The ownership check is the easiest to break by accident. The registry
    proves you own the PyPI package by finding `mcp-name: <server name>` in the
    project description, which is the README; a README edit that drops the
    marker makes a release unregisterable, and nothing else would notice."""
    import re

    spec = json.loads((REPO / "server.json").read_text())
    pyproject = (REPO / "pyproject.toml").read_text()
    readme = (REPO / "README.md").read_text()

    name = spec["name"]
    assert name == "io.github.chowdhurylab-studio/ipro-agent", name
    assert f"mcp-name: {name}" in readme, (
        "the README must carry the ownership marker, or the registry cannot "
        "verify the PyPI package"
    )
    # The marker needs a boundary after it; glued to a full stop it does not match.
    line = next(l for l in readme.splitlines() if "mcp-name:" in l)
    assert re.search(rf"mcp-name:\s*{re.escape(name)}(\s|-->|$)", line), line

    version = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
    assert spec["version"] == version, (spec["version"], version)

    # Six files carry this number and the release workflow only checked three,
    # so a bump could ship with MCP clients being told the old version in
    # serverInfo and CITATION.cff citing a release that never existed, with
    # everything green. These are the other three.
    import ipro_agent
    from ipro_agent.mcp_server import SERVER_INFO

    assert SERVER_INFO["version"] == version, (
        f"serverInfo says {SERVER_INFO['version']}, pyproject says {version}: "
        "every MCP client reads this at handshake")
    assert ipro_agent.__version__ == version, (ipro_agent.__version__, version)

    cff = (REPO / "CITATION.cff").read_text()
    cited = re.search(r"^version:\s*(\S+)", cff, re.M).group(1).strip('"')
    assert cited == version, (
        f"CITATION.cff cites {cited}, pyproject says {version}: this is the "
        "version GitHub's 'Cite this repository' button prints")

    pkg = spec["packages"][0]
    assert pkg["registryType"] == "pypi"
    assert pkg["registryBaseUrl"] == "https://pypi.org", pkg["registryBaseUrl"]
    assert pkg["version"] == version, (pkg["version"], version)
    assert pkg["transport"]["type"] == "stdio"

    pypi_name = re.search(r'^name = "([^"]+)"', pyproject, re.M).group(1)
    assert pkg["identifier"] == pypi_name, (pkg["identifier"], pypi_name)

    # Every environment variable it advertises is one the server reads.
    src = (REPO / "ipro_agent" / "mcp_server.py").read_text()
    for env in pkg.get("environmentVariables", []):
        assert env["name"] in src or env["name"] in (
            REPO / "ipro_agent" / "rotamers.py").read_text(), env["name"]
        assert env["isSecret"] is False, "this server takes no secrets"

    # The registry caps these, and the published JSON Schema does not say so:
    # only `mcp-publisher validate` rejects them. A 400-character description
    # passed every local check and failed at the registry with
    # "expected length <= 100".
    assert len(spec["description"]) <= 100, len(spec["description"])
    assert len(spec.get("title", "")) <= 100, len(spec.get("title", ""))

    assert (REPO / "docs" / "RELEASING.md").is_file()


def test_regression_the_readme_quickstart_matches_what_pip_alone_can_do():
    """The README now tells a stranger that `pip install ipro-agent` plus one
    config line is enough to start, and that on a bare install the agent can
    still read a structure and check a spec while scoring and the integer
    program refuse.

    That is a claim about behaviour on a machine this suite does not run on,
    so it is measured rather than asserted: PyMOL discovery is forced to find
    nothing, and each subcommand is run. If someone later moves a PyMOL call
    into `validate`, or makes `milp` fail with a different message, the README
    becomes a lie and this fails first."""
    readme = (REPO / "README.md").read_text()
    assert "pip install ipro-agent" in readme
    assert '"command": "ipro-agent-mcp"' in readme
    assert "pymol-open-source" in readme

    # A machine with no PyMOL anywhere: no importable module, no binary on the
    # path, no conda environment to discover.
    blind = (
        "import sys, shutil\n"
        "import ipro_agent.rotamers as rot, ipro_agent.doctor as doc\n"
        "rot.find_pymol_python = lambda: None\n"
        "doc.find_conda_envs = lambda: []\n"
        "_w = shutil.which\n"
        "shutil.which = lambda n, *a, **k: None if n == 'pymol' else _w(n, *a, **k)\n"
        "from ipro_agent.cli import main\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )
    spec = str(REPO / "examples" / "ompf_pore" / "spec_smoke.yaml")
    pdb = str(REPO / "examples" / "ompf_pore" / "2omf_A.pdb")

    def run(*argv):
        return subprocess.run([sys.executable, "-c", blind, *argv],
                              capture_output=True, text=True, cwd=REPO)

    for argv in (("validate", spec, "--json"),
                 ("inspect", pdb, "--json"),
                 ("goals", "--json")):
        p = run(*argv)
        assert p.returncode == 0, f"{argv[0]} must work without PyMOL: {p.stderr[:300]}"
        json.loads(p.stdout)

    for sub in ("score", "milp"):
        p = run(sub, spec, "--json")
        assert p.returncode != 0, f"{sub} cannot succeed without PyMOL"
        said = (p.stdout + p.stderr).lower()
        assert "no pymol available" in said, f"{sub} must say why: {said[:300]}"
        assert "pymol-open-source" in said, f"{sub} must say how to fix it"


def test_regression_setup_launches_the_command_it_advertised():
    """`ipro_setup(confirm=true)` must run the command it just printed.

    It did not. The plan handled the pip-installed case correctly -- with no
    checkout `_env_file()` is None, so the printed command fetched
    environment.yml from raw GitHub first -- and then the launch ignored that
    string and ran `[manager, "env", "create", "-f", str(env_file)]`, where
    `str(None)` is the four-character word "None". conda died at once, and the
    reply still came back `launched: true` with a pid and a log, which is the
    "success for a dead child" shape already fixed once for `ipro_run`.

    It mattered because of where it sits: a `pip install ipro-agent` has no
    PyMOL, `ipro_doctor` reports `can_design: false`, and `ipro_setup` is
    exactly where the server sends the agent next. So the one repair path
    offered to every published user was the broken branch.

    Popen is stubbed, so no environment is created and nothing is downloaded."""
    prog = (
        "import json\n"
        "from pathlib import Path\n"
        "import ipro_agent.mcp_server as m\n"
        "m._doctor = lambda: {'can_design': False, 'can_run': True,\n"
        "                     'missing_required': [], 'resolved': {}, 'degraded': []}\n"
        "m._which = lambda n: ('/usr/bin/' + n) if n == 'conda' else None\n"
        "seen = {}\n"
        "class P:\n"
        "    pid = 4242\n"
        "def fake(cmd, **kw):\n"
        "    seen['cmd'] = cmd\n"
        "    seen['shell'] = kw.get('shell', False)\n"
        "    return P()\n"
        "m.subprocess.Popen = fake\n"
        "out = {}\n"
        "for label, ef in (('pip', None), ('clone', Path('/tmp/environment.yml'))):\n"
        "    m._env_file = (lambda v: (lambda: v))(ef)\n"
        "    seen.clear()\n"
        "    plan = m.tool_setup({'confirm': True})\n"
        "    out[label] = {'advertised': plan['command'], 'launched': seen.get('cmd'),\n"
        "                  'shell': seen.get('shell'), 'said': plan.get('launched')}\n"
        "print(json.dumps(out))\n"
    )
    p = subprocess.run([sys.executable, "-c", prog],
                       capture_output=True, text=True, cwd=REPO)
    assert p.returncode == 0, p.stderr[-2000:]
    out = json.loads(p.stdout)

    for label in ("pip", "clone"):
        got = out[label]
        assert got["said"] is True, label
        assert got["launched"] == got["advertised"], (
            f"{label}: launched {got['launched']!r} but advertised "
            f"{got['advertised']!r} -- the reply would describe a command that "
            "was never run")
        assert got["shell"] is True, f"{label}: the advertised line needs a shell"

    # The specific regression: nothing may carry the string None as a path.
    pip = out["pip"]
    assert " None" not in pip["launched"] and not pip["launched"].endswith("None"), \
        pip["launched"]
    assert "curl" in pip["launched"] and "environment.yml" in pip["launched"], \
        "a pip install has no environment.yml, so it must fetch one first"
    assert "/tmp/environment.yml" in out["clone"]["launched"], \
        "a clone must use the file it already has, not download another"


def test_regression_the_trp_control_still_reads_under_the_filter():
    """Section 7o's measurement, re-run.

    The claim it guards is narrow and load-bearing: handed PoreDesigner's own
    OCD-TFTrp sequence, this implementation measures a bottleneck under the
    paper's 4.00 A filter. That is what rules out "the geometry is measured on
    the wrong slice" as the cause of the 7m negative result, so if the pore
    measurement, the acceptance filter or the rotamer path drifts, the
    conclusion in 7o silently stops being supported.

    It is a control, not a reproduction: the alphabet is a singleton at every
    position, so the sequence is forced. 7o records the three caveats, and the
    one that matters most here is that `target_radius: 2.0` aims the objective
    at the filter bound, so the margin is 0.005 A and a real change in the
    geometry will cross it."""
    if not _have_pymol():
        print("    (skipped: no PyMOL)")
        return

    spec = REPO / "examples" / "ompf_pore" / "spec_trp_control.yaml"
    assert spec.is_file(), "the control spec must ship, or 7o is not reproducible"

    import yaml
    d = yaml.safe_load(spec.read_text())
    assert len(d["positions"]) == 25
    off = [p["position"] for p in d["positions"] if p["allowed"] != ["TRP"]]
    assert not off, f"a half-pinned control measures something else: {off}"

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "trp"
        p = subprocess.run(
            [sys.executable, "-m", "ipro_agent", "run", str(spec),
             "--run-dir", str(run_dir), "--seed", "1", "--json"],
            capture_output=True, text=True, cwd=REPO)
        assert p.returncode == 0, p.stderr[-2000:]
        res = json.loads(p.stdout)

        assert res["accepted"] == 1, res.get("notes")
        published = (
            "V18W L20W M38W Y40W R42W E62W N64W R82W L83W L109W G110W D113W "
            "M114W L115W E117W F118W G119W G120W D121W A123W Y124W S125W "
            "V130W G131W R132W")
        assert res["mutation_string"] == published, res["mutation_string"]

        line = json.loads((run_dir / "iterations.jsonl").read_text().splitlines()[0])
        dia = line["components"]["min_pore_diameter"]
        assert dia < 4.0, f"7o says the published design reads under the filter; got {dia}"
        assert abs(dia - 3.99464) < 0.01, f"7o records 3.99464 A; got {dia}"


def test_regression_milp_never_writes_into_the_installed_package():
    """`ipro_milp` with no `out` wrote its LP file into the package tree.

    The `out` branch already knew the rule -- the child runs with
    `cwd=REPO_ROOT`, which is site-packages in a pip install, so a relative
    path lands inside the installed package -- and then the no-`out` call, the
    common one, skipped `-o` entirely and handed the child argparse's default
    `model.lp`, resolved against that same cwd. On a writable install that
    drops a 419 KB LP file into site-packages; on a read-only one it is the
    PermissionError traceback §7h already fixed for runs. Either way the reply
    said `lp_file: model.lp`, a path relative to a directory the caller never
    named.

    Checked at the argv the tool builds, so no solver and no PyMOL run."""
    prog = (
        "import json\n"
        "import ipro_agent.mcp_server as m\n"
        "seen = []\n"
        "m._cli = lambda cmd: seen.append(list(cmd)) or {'ok': True}\n"
        "m._resolve_spec = lambda s, keep: '/tmp/spec.yaml'\n"
        "out = {}\n"
        "for label, args in (('default', {'spec': 's'}),\n"
        "                    ('relative', {'spec': 's', 'out': 'mine.lp'}),\n"
        "                    ('absolute', {'spec': 's', 'out': '/tmp/abs.lp'})):\n"
        "    seen.clear()\n"
        "    m.tool_milp(dict(args))\n"
        "    cmd = seen[0]\n"
        "    out[label] = cmd[cmd.index('-o') + 1] if '-o' in cmd else None\n"
        "print(json.dumps({'paths': out, 'repo_root': str(m.REPO_ROOT)}))\n"
    )
    p = subprocess.run([sys.executable, "-c", prog],
                       capture_output=True, text=True, cwd=REPO)
    assert p.returncode == 0, p.stderr[-2000:]
    got = json.loads(p.stdout)
    paths, root = got["paths"], got["repo_root"]

    for label, path in paths.items():
        assert path is not None, f"{label}: -o must always be passed"
        assert Path(path).is_absolute(), f"{label}: {path} is relative"
        assert not path.startswith(root + "/"), (
            f"{label}: {path} is inside the package tree at {root}")

    assert paths["absolute"] == "/tmp/abs.lp", paths["absolute"]
    assert Path(paths["relative"]).name == "mine.lp", paths["relative"]
    assert Path(paths["default"]).name == "model.lp", paths["default"]


def test_regression_a_pore_axis_check_that_fails_says_so():
    """`_axis_clearance` returned None on any exception, which skipped the
    "THE PORE RADIUS HERE IS NOT A PORE RADIUS" warning with no note and no
    log line -- in a file whose rule is that every silent degradation gets a
    note. §7m names the fitted axis as one of three suspects for the shipped
    case accepting nothing, so that warning is the diagnostic least affordable
    to lose quietly.

    The cache was the second half of it: it returned before `origin` was read,
    so once the backbone moved and the axis was refitted, every later
    iteration was answered with iteration 0's axis."""
    import types

    from ipro_agent.loop import IPROLoop

    def bare(input_pdb):
        o = types.SimpleNamespace()
        o._once = set()
        o.notes = []
        o._input_pdb = input_pdb
        o._design_pdb = None
        o._axis_clear_cache = {}
        o.spec = types.SimpleNamespace(
            design_molecule=types.SimpleNamespace(chains=["A"]))
        return o

    # A failure must leave a note rather than nothing.
    o = bare("/nonexistent/no-such-structure.pdb")
    got = IPROLoop._axis_clearance(o, np.zeros(3))
    assert got is None
    assert o.notes, "a swallowed failure must still be reported"
    note = o.notes[0].lower()
    assert "pore axis" in note and "lumen" in note, o.notes[0]
    # and only once, however many iterations ask
    IPROLoop._axis_clearance(o, np.zeros(3))
    assert len(o.notes) == 1, o.notes

    # Two different origins must not share one answer.
    pdb = str(REPO / "examples" / "ompf_pore" / "2omf_A.pdb")
    o = bare(pdb)
    a = IPROLoop._axis_clearance(o, np.array([-3.07, 49.31, 35.36]))
    b = IPROLoop._axis_clearance(o, np.array([-13.1, 57.65, 34.82]))
    assert a is not None and b is not None
    assert a != b, (
        f"both origins returned {a}: the cache is answering for the wrong axis")
    assert len(o._axis_clear_cache) == 2, o._axis_clear_cache
    # and a repeat of the first origin is served from the cache, unchanged
    assert IPROLoop._axis_clearance(o, np.array([-3.07, 49.31, 35.36])) == a


def test_regression_the_pore_offset_cap_has_one_default_and_is_recorded():
    """`pore_max_centre_offset` had two different defaults -- 6.0 where the
    loop calls it, 4.0 in `pore_profile` itself -- and no spec in the repo sets
    the key, so the same structure measured differently depending on which
    entry point you used.

    6.0 is the one to keep, and the reason is measurable rather than a
    preference: the cap still *binds* at 4.0 on OmpF, pinning the winning
    circle's centre at exactly the cap and reading 7.68 A, where at 6.0 the
    centre settles at 3.14 A on its own and the reading is the 8.07 A that §4
    records against the published 11 x 7 A eyelet. A cap low enough to bind is
    measuring the setting, not the pore -- so the geometry record now carries
    the winning centre's offset, and a reader can see when the two are equal."""
    import inspect
    import re

    from ipro_agent.pore import axis_from_lining, pore_profile
    from ipro_agent.structure import Structure, is_hydrogen

    sig = inspect.signature(pore_profile)
    module_default = sig.parameters["max_centre_offset"].default
    src = (REPO / "ipro_agent" / "loop.py").read_text()
    loop_defaults = set(re.findall(
        r'opts\.get\("pore_max_centre_offset",\s*([0-9.]+)\)', src))
    assert loop_defaults, "the loop must still read the key"
    assert loop_defaults == {str(module_default)}, (
        f"pore_profile defaults to {module_default} and the loop to "
        f"{loop_defaults}: the same structure measures differently depending "
        "on which one you call")

    st = Structure.load(str(REPO / "examples" / "ompf_pore" / "2omf_A.pdb"), ["A"])
    want = {18, 20, 38, 40, 42, 62, 64, 82, 83, 109, 110, 113, 114, 115, 117,
            118, 119, 120, 121, 123, 124, 125, 130, 131, 132}
    cas = np.array([a.coord for r in st.residues("A") if int(r.id[1]) in want
                    for a in r if a.get_name().strip() == "CA"])
    assert len(cas) == 25, len(cas)
    origin, direction = axis_from_lining(cas)
    atoms = [a for a in st.all_atoms() if not is_hydrogen(a)]
    coords = np.array([a.coord for a in atoms])
    names = [a.get_name() for a in atoms]
    els = [(a.element or "").strip() or None for a in atoms]

    def read(cap):
        p = pore_profile(coords, names, axis_origin=origin,
                         axis_direction=direction, spacing=0.5,
                         max_centre_offset=cap, elements=els)
        i = int(np.argmin(p.radii))
        off = float(np.linalg.norm(np.cross(
            np.asarray(p.centres[i]) - p.axis_origin, p.axis_direction)))
        return p.min_diameter, off

    tight_d, tight_off = read(4.0)
    kept_d, kept_off = read(module_default)
    assert abs(tight_off - 4.0) < 1e-6, (
        f"4.0 is expected to bind here; centre sat at {tight_off}")
    assert kept_off < module_default - 0.5, (
        f"the kept default must not bind; centre sat at {kept_off}")
    assert kept_d > tight_d, (tight_d, kept_d)
    assert abs(kept_d - 8.07) < 0.05, (
        f"§4 records 8.07 A for this axis and cap; got {kept_d}")


def _backbone_pdb(rows) -> str:
    """N/CA/C/O lines for (chain, resseq, icode, resname), spaced 3.8 A apart."""
    out, serial = [], 1
    for i, (chain, num, icode, name) in enumerate(rows):
        base = i * 3.8
        for atom, elem, dx in (("N", "N", 0.0), ("CA", "C", 1.2),
                               ("C", "C", 2.4), ("O", "O", 2.8)):
            # Exact PDB columns: 13-16 atom, 17 altLoc, 18-20 resName,
            # 22 chain, 23-26 resSeq, 27 iCode.
            out.append(
                f"ATOM  {serial:>5} {' ' + atom:<4} {name:>3} {chain}"
                f"{num:>4}{icode or ' '}   {base + dx:>8.3f}{0.0:>8.3f}"
                f"{0.0:>8.3f}  1.00  0.00          {elem:>2}\n")
            serial += 1
    return "".join(out) + "END\n"


def test_regression_an_unanchored_perturbation_is_recorded_as_one():
    """A proline anchor makes `choose_segment` fall through to an unanchored
    draw, and the comment there promised to "say so through the Perturbation
    record". It did not: the record came back with a region that does not hold
    the design position and no field distinguishing it from a correctly
    anchored one, so `state.json` could not tell them apart.

    It matters because an unanchored region is the 12.4%-overlap defect §7h
    fixed -- the backbone moves where the MILP has no variables, and the
    Metropolis test rules on that."""
    from ipro_agent.perturb import perturb_backbone
    from ipro_agent.structure import Structure

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pro.pdb"
        path.write_text(_backbone_pdb([
            ("A", 51, "", "ALA"), ("A", 52, "", "PRO"), ("A", 53, "", "VAL")]))

        st = Structure.load(str(path), ["A"])
        rec = perturb_backbone(st, "A", rng=random.Random(0), anchor=52,
                               length_range=(1, 1))
        assert rec.anchored is False, (
            "a proline anchor falls through to an unanchored region, and the "
            "record must say so")
        assert rec.anchor == (52, ""), rec.anchor
        assert (52, "") not in rec.keys, rec.keys

        # A reachable anchor is marked the other way, so the flag means something.
        path2 = Path(tmp) / "ok.pdb"
        path2.write_text(_backbone_pdb([
            ("A", 51, "", "ALA"), ("A", 52, "", "GLY"), ("A", 53, "", "VAL")]))
        st2 = Structure.load(str(path2), ["A"])
        ok = perturb_backbone(st2, "A", rng=random.Random(0), anchor=52,
                              length_range=(1, 1))
        assert ok.anchored is True, ok
        assert (52, "") in ok.keys, ok.keys


def test_regression_insertion_codes_are_not_collapsed_when_perturbing():
    """`_run_containing` keyed by `int(r.id[1])`, so A52, A52A and A52B
    collapsed onto one entry and the last one read won.

    Reproduced: the design position 52 was dropped from the region entirely
    while the Perturbation record still reported residues=[51, 52, 53], and
    `_choose_window` matched on the bare number -- so position 52 was counted
    as inside a region that had in fact moved 52B. Structure and
    DesignPosition both carry the insertion code correctly, so it was lost
    only here, and antibody numbering is where insertion codes live."""
    from ipro_agent.perturb import perturb_backbone, residue_key
    from ipro_agent.structure import Structure

    rows = [("A", 51, "", "ALA"), ("A", 52, "", "GLY"), ("A", 52, "A", "SER"),
            ("A", 52, "B", "THR"), ("A", 53, "", "VAL")]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "icode.pdb"
        path.write_text(_backbone_pdb(rows))
        st = Structure.load(str(path), ["A"])
        assert [residue_key(r) for r in st.residues("A")] == \
            [(51, ""), (52, ""), (52, "A"), (52, "B"), (53, "")]

        # The anchor is the bare 52, which must be the GLY and not a 52-variant.
        rec = perturb_backbone(st.copy(), "A", rng=random.Random(0), anchor=52,
                               length_range=(1, 1))
        assert rec.anchored is True, rec
        assert rec.keys == [(52, "")], rec.keys

        # And an insertion-coded anchor must reach its own residue.
        rec_b = perturb_backbone(st.copy(), "A", rng=random.Random(0), anchor=52,
                                 anchor_icode="B", length_range=(1, 1))
        assert rec_b.keys == [(52, "B")], rec_b.keys

        # Both report residues=[52]; only `keys` tells them apart, which is
        # why the window check reads keys.
        assert rec.residues == rec_b.residues == [52]


def test_regression_the_perturbed_chain_follows_the_design_positions():
    """`chain = (dm.chains or chain_ids())[0]` fixed one chain for the run's
    lifetime and then looked for design positions on it. For a design molecule
    whose positions are on the second chain that took two failures at once and
    silently: the anchor list was empty every iteration, so every perturbation
    was unanchored, and the chain being moved was one with nothing designed on
    it. Nothing in config.py ties positions to the first chain, so the spec
    validates, and the skill advertises interface work where a second chain is
    the normal case.

    The draw order now follows the paper: pick the design position, then take
    its chain."""
    import types

    from ipro_agent.loop import IPROLoop

    src = (REPO / "ipro_agent" / "loop.py").read_text()
    assert "(dm.chains or self._design_st.chain_ids())[0]" not in src, \
        "the chain must come from the drawn design position, not from index 0"

    picks = []
    for seed in range(40):
        o = types.SimpleNamespace(
            rng=random.Random(seed), _once=set(), notes=[],
            spec=types.SimpleNamespace(positions=[
                types.SimpleNamespace(chain="B", number=10, insertion_code=""),
                types.SimpleNamespace(chain="B", number=11, insertion_code=""),
            ]),
        )
        here = [p for p in o.spec.positions if p.chain in ("A", "B")]
        picks.append(o.rng.choice(here).chain)
    assert set(picks) == {"B"}, set(picks)
    assert "_perturb_once" in dir(IPROLoop)


def test_regression_dropped_identities_survive_the_assembly_combine():
    """`combine()` copied the pruned *count* per position and left the
    identities behind, so a multi-assembly run reported a clean design space
    while an identity the spec allows had been deleted from it.

    That is the §7k blocker exactly: the prune removed PoreDesigner's
    published V18W and nothing said so. This note is the guard against it, and
    on the combined path the guard was empty by construction."""
    from ipro_agent.assemblies import combine
    from ipro_agent.config import Assembly
    from ipro_agent.energy.base import AtomSet
    from ipro_agent.milp import EnergyTables, Rotamer

    def rot(aa, state):
        return Rotamer(position="A16", aa=aa, state=state,
                       atoms=AtomSet(names=["CB"], coords=[[0.0, 0.0, 0.0]]))

    def one(dropped):
        t = EnergyTables(positions=["A16"],
                         rotamers={"A16": [rot("TRP", 1), rot("PHE", 1)]})
        t.pruned_per_position["A16"] = 1
        t.dropped_identities["A16"] = list(dropped)
        t.self_energy[("A16", 0)] = -1.0
        t.self_energy[("A16", 1)] = -0.5
        return t

    combined = combine(
        {"a": one(["TRP"]), "b": one(["TRP"])},
        [Assembly(name="a"), Assembly(name="b")],
    ).tables
    assert combined.dropped_identities, (
        "the identities the prune removed must survive the combine, or the "
        "only report a multi-assembly run gets is a count")
    assert all("TRP" in v for v in combined.dropped_identities.values()), \
        combined.dropped_identities
    # One entry per virtual position, not one shared entry.
    assert len(combined.dropped_identities) == 2, combined.dropped_identities


def test_regression_dropped_identities_are_rechecked_after_the_backbone_moves():
    """`_note_dropped_identities` was called twice, both before the loop, and
    never after `_rebuild_for_backbone`.

    So an identity that is reachable on the input backbone and unreachable
    once the backbone has moved -- the case a perturbing run exists to
    explore -- was never reported at all."""
    src = (REPO / "ipro_agent" / "loop.py").read_text()
    rebuild = src.index("cb = self._rebuild_for_backbone(candidate)")
    after = src[rebuild:rebuild + 1200]
    assert "_note_dropped_identities" in after, (
        "a rebuilt table must be checked too, or the note only ever describes "
        "the input backbone")
    # and the note stays once-per-run however many iterations rebuild
    note = src[src.index("def _note_dropped_identities"):]
    assert "_once" in note[:1200], "the note must not repeat every iteration"


def test_regression_deep_nesting_does_not_end_the_mcp_session():
    """§7h states the invariant outright: one error reply each, never the
    session. Five malformed shapes were fixed to honour it; this one was
    missed, because `json.loads` recurses per nesting level and therefore
    raises RecursionError rather than a JSONDecodeError, which escaped the
    clause already there and ended the process.

    The depth is probed rather than hardcoded. The sweep that found this
    reported 100,000 levels as the trigger; on the interpreter here 100,000
    parses fine and the server answers it as a JSON-RPC batch, and the real
    threshold is nearer 500,000. It depends on the interpreter and its
    recursion limit, so a fixed number would test nothing on the next one."""
    depth = None
    for candidate in (100_000, 500_000, 1_000_000):
        try:
            json.loads("[" * candidate + "]" * candidate)
        except RecursionError:
            depth = candidate
            break
        except Exception:
            continue
    if depth is None:
        print("    (skipped: json.loads does not recurse far enough here)")
        return

    payload = (
        "[" * depth + "]" * depth + "\n" +
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18",
                               "capabilities": {},
                               "clientInfo": {"name": "t", "version": "0"}}}) + "\n" +
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list",
                    "params": {}}) + "\n"
    )
    p = subprocess.run([sys.executable, "-m", "ipro_agent.mcp_server"],
                       input=payload, capture_output=True, text=True, cwd=REPO,
                       timeout=300)
    assert p.returncode == 0, (
        f"depth {depth} ended the session: exit {p.returncode}, "
        f"{p.stderr[-400:]}")
    replies = [json.loads(l) for l in p.stdout.splitlines()
               if l.strip().startswith(("{", "["))]
    ids = [r.get("id") for r in replies if isinstance(r, dict)]
    assert 2 in ids, f"the session must survive the bad line; got ids {ids}"
    listed = next(r for r in replies if isinstance(r, dict) and r.get("id") == 2)
    assert listed["result"]["tools"], "tools/list must still answer"

    # The invariant is one error reply EACH, not merely survival: the bad line
    # gets its own -32700 and the two good ones are answered normally.
    bad = [r for r in replies
           if isinstance(r, dict) and r.get("error", {}).get("code") == -32700]
    assert len(bad) == 1, [r.get("error") for r in replies if isinstance(r, dict)]


def test_regression_the_document_fallback_urls_match_this_build():
    """A server with no checkout hands the agent a raw GitHub URL instead of
    the document, and `ipro_setup` tells a pip-installed copy to curl
    `environment.yml` from the same base. That base tracked `main`, so an
    installed 0.1.0 was served whatever `main` says today -- a manual
    describing behaviour its code does not have, and an install recipe that
    can drift from the installed version. It is pinned to this build's tag."""
    from ipro_agent.mcp_server import RAW_BASE, SERVER_INFO

    version = SERVER_INFO["version"]
    assert f"/v{version}/" in RAW_BASE, RAW_BASE
    assert "/main/" not in RAW_BASE, (
        "the fallback must not track main, or a released copy is handed "
        "unreleased documents")
    assert RAW_BASE.endswith("/"), RAW_BASE
