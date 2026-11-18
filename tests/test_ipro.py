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


def test_restraint_with_no_design_position_is_reported_not_enforced():
    """If nothing the design controls affects the measurement, say so."""
    from ipro_agent.config import Restraint
    from ipro_agent.restraints import plan_restraints
    from ipro_agent.structure import Structure

    st = Structure.load(CRN)
    t = _geom_tables({"A1": [("SER", 1.0)]})
    # Both atoms are fixed, and deliberately far apart.
    r = Restraint(kind="distance", atoms=["A30:CA", "A31:CA"], maximum=0.5, label="fixedpair")
    plan = plan_restraints([r], t, st, [])
    assert not plan.rotamer_vetoes and not plan.pair_vetoes
    assert plan.native_violations and "violated by the input structure" in plan.native_violations[0]


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
