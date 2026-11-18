"""Command-line interface -- the surface an agent drives.

    python -m ipro_agent validate   spec.yaml
    python -m ipro_agent positions  spec.yaml --near LIG --cutoff 5
    python -m ipro_agent milp       spec.yaml -o model.lp
    python -m ipro_agent run        spec.yaml --run-dir runs/case1
    python -m ipro_agent status     runs/case1 [--json]
    python -m ipro_agent score      spec.yaml
    python -m ipro_agent goals

Design notes for the agent-facing behaviour:

* every subcommand takes ``--json`` and prints a single JSON object, so an agent
  can branch on results without parsing prose;
* ``run`` is the only long-running command; ``status`` is cheap and bounded in
  output size no matter how long the run has gone on, so polling it is safe;
* ``validate`` and ``milp`` do no science -- they exist so an agent can check a
  spec and inspect the formulation *before* spending hours on a run;
* exit status is 0 on success and 1 on a usable-but-failed outcome, so shell
  conditionals work.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import DesignSpec, Role, SpecError
from .goals import GOALS, build_goal, suggest_goal
from .state import RunDir


def _load(path: str) -> DesignSpec:
    try:
        return DesignSpec.load(path)
    except SpecError as exc:
        print(f"spec error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except FileNotFoundError:
        # A mistyped path used to come back as a 20-line traceback from an
        # agent-facing CLI, which is noise an agent has to parse past to find
        # the one fact it needs.
        print(f"spec not found: {path}", file=sys.stderr)
        raise SystemExit(2)
    except IsADirectoryError:
        print(
            f"spec path is a directory, not a file: {path}\n"
            f"  did you mean {path.rstrip('/')}/spec_smoke.yaml ?",
            file=sys.stderr,
        )
        raise SystemExit(2)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def cmd_validate(args) -> int:
    spec = _load(args.spec)
    problems: list[str] = []
    info: dict = {}

    # Structures present and readable, design positions real.
    from .structure import Structure

    try:
        dm = spec.design_molecule
    except SpecError as exc:
        print(f"spec error: {exc}", file=sys.stderr)
        return 2

    try:
        st = Structure.load(dm.path, dm.chains or None)
        info["design_chains"] = st.chain_ids()
        info["design_residues"] = sum(1 for _ in st.residues())
    except Exception as exc:
        problems.append(f"design molecule {dm.path}: {exc}")
        st = None

    for m in spec.targets + spec.anti_targets:
        try:
            s2 = Structure.load(m.path, m.chains or None)
            info.setdefault("partners", {})[m.name] = {
                "chains": s2.chain_ids(),
                "residues": sum(1 for _ in s2.residues()),
                "hetero": sum(1 for _ in s2.hetero_residues()),
            }
        except Exception as exc:
            problems.append(f"{m.role.value} molecule {m.name} ({m.path}): {exc}")

    if st is not None:
        for p in spec.positions:
            try:
                res = st.find(p.chain, p.number, p.insertion_code)
                p.native = res.get_resname().strip()
                if p.native not in p.allowed:
                    info.setdefault("native_not_allowed", []).append(
                        f"{p.label} is {p.native}, which is not in its allowed set"
                    )
            except Exception as exc:
                problems.append(str(exc))

    # Goal consistency.
    try:
        goal = build_goal(spec)
        info["goal"] = goal.describe()
        cons = goal.default_constraints()
        info["implied_constraints"] = [
            f"{c.kind}({c.label or ''}) count={c.count} fraction={c.fraction} "
            f"residues={sorted(c.residue_set()) if c.residues else []}"
            for c in cons
        ]
    except SpecError as exc:
        problems.append(f"goal: {exc}")

    # Restraints: a cheap pre-flight that needs no rotamer library, so a
    # misspelled atom name is caught in a second rather than after a harvest.
    if spec.restraints:
        from .restraints import check_restraints

        partners = []
        for m in spec.targets + spec.anti_targets:
            try:
                partners.append(Structure.load(m.path, m.chains or None))
            except Exception:
                pass
        r_problems, r_notes = check_restraints(
            spec.restraints, st, partners, [p.label for p in spec.positions]
        )
        info["restraints"] = [r.describe() for r in spec.restraints]
        if r_notes:
            info["restraint_notes"] = r_notes
        problems.extend(r_problems)

    info["design_space_sequences"] = spec.design_space_size()
    info["positions"] = len(spec.positions)
    info["iterations"] = spec.loop.iterations
    info["acceptance_kt_kcal"] = round(spec.loop.kt, 4)

    if args.json:
        print(json.dumps({"ok": not problems, "problems": problems, "info": info}, indent=2, default=str))
    else:
        print(spec.summary())
        print()
        for k, v in info.items():
            print(f"{k}: {v}")
        if problems:
            print("\nPROBLEMS:")
            for p in problems:
                print(f"  - {p}")
        else:
            print("\nspec is usable")
    return 1 if problems else 0


# ---------------------------------------------------------------------------
# positions -- propose design positions
# ---------------------------------------------------------------------------


def cmd_positions(args) -> int:
    """Propose design positions by proximity, so the user need not list them.

    This implements the position-selection rule the IPRO case studies use: take
    the residues of the design molecule within a distance cutoff of the partner
    molecule (or of a named ligand). It *proposes*; the user still decides, and
    the papers additionally drop highly conserved and catalytic residues, which
    this cannot know about.
    """
    spec = _load(args.spec)
    from .structure import Structure, interface_residues

    dm = spec.design_molecule
    design = Structure.load(dm.path, dm.chains or None)

    partner = None
    if args.near:
        for m in spec.targets + spec.anti_targets:
            if m.name == args.near:
                partner = Structure.load(m.path, m.chains or None)
                break
        if partner is None:
            print(f"no molecule named {args.near!r} in the spec", file=sys.stderr)
            return 2
    else:
        partners = spec.targets + spec.anti_targets
        if not partners:
            print("spec has no target molecule; pass --near or add one", file=sys.stderr)
            return 2
        partner = Structure.load(partners[0].path, partners[0].chains or None)

    refs = interface_residues(design, partner, cutoff=args.cutoff)
    out = [
        {"position": r.label, "native": r.name}
        for r in refs
    ]
    if args.json:
        print(json.dumps({"cutoff": args.cutoff, "count": len(out), "positions": out}, indent=2))
    else:
        print(f"{len(out)} residues of {dm.name} within {args.cutoff} A of {args.near or 'the first partner'}:")
        for d in out:
            print(f"  {d['position']}  {d['native']}")
        print("\nAs spec YAML:")
        print("positions:")
        for d in out:
            print(f"  - {{position: {d['position']}, allowed: [standard]}}   # native {d['native']}")
        print(
            "\nReview before use: the published protocols also exclude highly "
            "conserved and catalytic residues, which this proximity rule cannot detect."
        )
    return 0


# ---------------------------------------------------------------------------
# milp -- write the formulation without running
# ---------------------------------------------------------------------------


def cmd_milp(args) -> int:
    spec = _load(args.spec)
    from .energy import resolve as resolve_energy
    from .milp import RotamerMILP, build_tables
    from .rotamers import build_template, harvest
    from .structure import Structure

    dm = spec.design_molecule
    design = Structure.load(dm.path, dm.chains or None)
    partners = [
        Structure.load(m.path, m.chains or None)
        for m in spec.targets + spec.anti_targets
    ]
    for p in spec.positions:
        p.native = design.find(p.chain, p.number, p.insertion_code).get_resname().strip()
    native = {p.label: p.native or "" for p in spec.positions}

    rots, pymol_native, errors = harvest(dm.path, spec.positions)
    template = build_template([design, *partners], [p.label for p in spec.positions])
    backend = resolve_energy(spec.backends.energy, cutoff=spec.loop.energy_cutoff)
    tables = build_tables(
        rots, template, backend, native=native,
        energy_cutoff=spec.loop.energy_cutoff,
        reference_energies=spec.goal_options.get("reference_energies"),
    )
    goal = build_goal(spec)
    milp = RotamerMILP(tables, constraints=goal.default_constraints(), native=native)
    lp = milp.to_lp()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(lp)

    payload = {
        "lp_file": str(out),
        "lines": lp.count("\n"),
        "positions": len(tables.positions),
        "binary_variables": sum(len(v) for v in tables.rotamers.values()),
        "pair_variables": len(tables.pair_energy),
        "tables": {k: int(v) for k, v in tables.stats.items()},
        "constraints": [c.kind for c in milp.constraints],
        "rotamer_errors": errors,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"wrote {out}")
        for k, v in payload.items():
            if k != "lp_file":
                print(f"  {k}: {v}")
        print("\nRead the header comments of the LP file to check that your "
              "constraints were encoded as you intended.")
    return 0


# ---------------------------------------------------------------------------
# score -- measure the input structure without designing
# ---------------------------------------------------------------------------


def cmd_score(args) -> int:
    """Energies of the input complex as supplied.

    Use this to get the wild-type reference an objective may need (for example
    ``goal_options.reference_interaction``), and to sanity-check that the
    molecules are positioned relative to each other as intended -- an
    interaction energy near zero usually means the partners are not in contact.
    """
    spec = _load(args.spec)
    from .loop import IPROLoop

    loop = IPROLoop(spec, args.run_dir or "runs/_score")
    loop.prepare()
    ctx, _tables = loop.measure_native()
    goal = loop.goal
    payload = {
        "backend": loop.backend.describe(),
        "mode": loop.mode,
        "goal": goal.describe(),
        "interaction_energies": ctx.interaction,
        "design_only": ctx.design_only,
        "total": ctx.total,
        "score": goal.score(ctx),
        "components": {k: float(v) for k, v in goal.components(ctx).items()},
        "notes": loop.notes,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"backend: {payload['backend']}")
        print(f"goal:    {payload['goal']}")
        for name, ie in ctx.interaction.items():
            print(f"  IE[{name}] = {ie:+.4f}")
        print(f"  design-only energy = {ctx.design_only:+.4f}")
        print(f"  score              = {payload['score']:+.4f}")
        for n in loop.notes:
            print(f"  note: {n}")
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def cmd_run(args) -> int:
    spec = _load(args.spec)
    if args.iterations:
        spec.loop.iterations = args.iterations
    if args.seed is not None:
        spec.loop.random_seed = args.seed
    if args.max_minutes:
        spec.loop.max_minutes = args.max_minutes

    from .loop import run_design

    run_dir = args.run_dir or f"runs/{spec.name}"
    try:
        res = run_design(spec, run_dir)
    except Exception as exc:
        msg = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "run_dir": str(run_dir)}
        print(json.dumps(msg, indent=2) if args.json else f"FAILED: {msg['error']}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({
            "ok": True,
            "run_dir": str(res.run_dir),
            "mode": res.mode,
            "start_score": res.start_score,
            "best_score": res.best_score,
            "mutations": res.best_mutations,
            "mutation_string": res.mutation_string,
            "sequence": res.best_sequence,
            "iterations": res.iterations,
            "accepted": res.accepted,
            "structures": res.structures,
            "notes": res.notes,
        }, indent=2))
    else:
        print(res.summary())
        print(f"\nsequence:\n{res.best_sequence}")
        if res.notes:
            print("\nnotes:")
            for n in res.notes:
                print(f"  - {n}")
        print(f"\nrun directory: {res.run_dir}")
    return 0


# ---------------------------------------------------------------------------
# status / goals
# ---------------------------------------------------------------------------


def cmd_dock(args) -> int:
    """Dock the spec's partner molecules into the design molecule's site.

    An explicit step rather than part of the loop: the iteration does not
    auto-dock (see docs/DESIGN.md), so an agent that needs a partner re-placed
    after mutation calls this, then re-scores. The search box is centred on the
    design positions by default, which is the site that was redesigned.

    Note that Vina's affinity is **not** on the same scale as this tool's
    objective. Use the pose, then score it with ``ipro-agent score``; never add
    a Vina affinity to an objective value.
    """
    spec = _load(args.spec)
    import numpy as np

    from .dock import DockBox, dock, dock_batch, find_vina
    from .structure import Structure

    dm = spec.design_molecule
    design = Structure.load(dm.path, dm.chains or None)

    if args.center:
        center = tuple(float(x) for x in args.center)
    else:
        # Centre on the design positions: the site that was redesigned.
        pts = []
        for p in spec.positions:
            try:
                pts.append(design.coords(design.find(p.chain, p.number, p.insertion_code)))
            except Exception as exc:
                print(f"warning: {p.label}: {exc}", file=sys.stderr)
        if not pts:
            print(
                "cannot place a docking box: no design position resolved and no "
                "--center given",
                file=sys.stderr,
            )
            return 2
        center = tuple(float(v) for v in np.vstack(pts).mean(axis=0))

    box = DockBox(center=center, size=tuple(float(s) for s in args.size))
    ligands = [m for m in spec.molecules if m.role is not Role.DESIGN]
    if args.only:
        ligands = [m for m in ligands if m.name in args.only]
    if not ligands:
        print("no partner molecules to dock", file=sys.stderr)
        return 2

    if find_vina() is None and not args.dry_run:
        print(
            "AutoDock Vina not found. Install it (see docs/INSTALL.md) or pass "
            "--dry-run to see the exact command that would be issued.",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.out_dir)
    results = []
    for m in ligands:
        try:
            res = dock(
                receptor=dm.path, ligand=m.path, box=box, out_dir=out_dir,
                exhaustiveness=args.exhaustiveness, num_modes=args.num_modes,
                seed=args.seed, dry_run=args.dry_run,
            )
            results.append({
                "molecule": m.name,
                "best_score_kcal": res.best_score,
                "scores": res.scores,
                "poses": [str(p) for p in res.poses],
                "command": res.command,
            })
        except Exception as exc:
            results.append({"molecule": m.name, "error": f"{type(exc).__name__}: {exc}"})

    payload = {
        "box": {"center": list(center), "size": list(box.size)},
        "dry_run": bool(args.dry_run),
        "results": results,
        "note": (
            "Vina affinities are not comparable to this tool's objective; "
            "re-score the chosen pose with 'ipro-agent score'"
        ),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"box centre {tuple(round(c, 2) for c in center)} size {box.size}")
        for r in results:
            if "error" in r:
                print(f"  {r['molecule']}: ERROR {r['error']}")
            elif args.dry_run:
                print(f"  {r['molecule']}: would run\n    {' '.join(r['command'])}")
            else:
                print(f"  {r['molecule']}: best {r['best_score_kcal']:+.2f} kcal/mol, "
                      f"{len(r['poses'])} poses")
        print(f"\n{payload['note']}")
    return 0 if all("error" not in r for r in results) else 1


def cmd_status(args) -> int:
    rd = RunDir(args.run_dir, create=False)
    if not rd.path.exists():
        print(f"no such run directory: {rd.path}", file=sys.stderr)
        return 2
    print(json.dumps(rd.as_json(), indent=2) if args.json else rd.summarize(tail=args.tail))
    return 0


def cmd_goals(args) -> int:
    rows = []
    for name, cls in sorted(GOALS.items()):
        rows.append({
            "goal": name,
            "objective": cls.objective_text,
            "source": cls.source,
        })
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            print(f"{r['goal']}")
            print(f"    optimizes: {r['objective']}")
            print(f"    source:    {r['source']}")
    if args.statement:
        g = suggest_goal(args.statement)
        print(f"\nsuggestion for {args.statement!r}: {g or '(no keyword matched; choose deliberately)'}")
    return 0


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ipro-agent",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-v", "--verbose", action="count", default=0)
    sub = ap.add_subparsers(dest="command", required=True)

    def add_json(p):
        p.add_argument("--json", action="store_true", help="print one JSON object")
        return p

    p = add_json(sub.add_parser("validate", help="check a spec without running"))
    p.add_argument("spec")
    p.set_defaults(func=cmd_validate)

    p = add_json(sub.add_parser("positions", help="propose design positions by proximity"))
    p.add_argument("spec")
    p.add_argument("--near", help="name of the partner molecule to measure from")
    p.add_argument("--cutoff", type=float, default=5.0, help="distance cutoff in Angstroms")
    p.set_defaults(func=cmd_positions)

    p = add_json(sub.add_parser("milp", help="write the MILP formulation and stop"))
    p.add_argument("spec")
    p.add_argument("-o", "--output", default="model.lp")
    p.set_defaults(func=cmd_milp)

    p = add_json(sub.add_parser("score", help="energies of the input structure as supplied"))
    p.add_argument("spec")
    p.add_argument("--run-dir")
    p.set_defaults(func=cmd_score)

    p = add_json(sub.add_parser("run", help="run the IPRO loop"))
    p.add_argument("spec")
    p.add_argument("--run-dir")
    p.add_argument("--iterations", type=int, help="override spec loop.iterations")
    p.add_argument("--seed", type=int, help="override spec loop.random_seed")
    p.add_argument("--max-minutes", type=float, help="wall-clock budget; stops cleanly")
    p.set_defaults(func=cmd_run)

    p = add_json(sub.add_parser("dock", help="dock the spec's partners with AutoDock Vina"))
    p.add_argument("spec")
    p.add_argument("--out-dir", default="runs/dock")
    p.add_argument("--center", nargs=3, metavar=("X", "Y", "Z"),
                   help="box centre; default is the centroid of the design positions")
    p.add_argument("--size", nargs=3, default=(22.5, 22.5, 22.5), metavar=("X", "Y", "Z"))
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--num-modes", type=int, default=9)
    p.add_argument("--seed", type=int, help="fix Vina's seed for a reproducible pose")
    p.add_argument("--only", nargs="*", help="dock only these molecule names")
    p.add_argument("--dry-run", action="store_true",
                   help="print the exact Vina command without running it")
    p.set_defaults(func=cmd_dock)

    p = add_json(sub.add_parser("status", help="progress of a run (cheap; safe to poll)"))
    p.add_argument("run_dir")
    p.add_argument("--tail", type=int, default=5)
    p.set_defaults(func=cmd_status)

    p = add_json(sub.add_parser("goals", help="list goals and what each optimizes"))
    p.add_argument("--statement", help="a plain-language goal, to get a suggestion")
    p.set_defaults(func=cmd_goals)

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
