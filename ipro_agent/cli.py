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
from .rotamers import RotamerError
from .structure import StructureError
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

    # Two molecules that are literally the same coordinates. A homodimer
    # docked from one monomer legitimately shares a FILE, so this is not a spec
    # error -- but same file AND same chains, with no pose applied, means two
    # bodies occupying one space. Measured with the design molecule and a
    # "partner" both pointing at 2omf_A.pdb chain A: validate reported ok, and
    # score returned an interaction energy of +149,211 kcal/mol from atoms at
    # zero distance. A number like that is not an interaction, but it was
    # reported as one -- and a goal whose filter is "better than the reference"
    # would have adopted it as the reference.
    bodies: dict[tuple, str] = {}
    for m in spec.molecules:
        key = (str(Path(str(m.path)).resolve()), tuple(sorted(m.chains or ())))
        if key in bodies:
            problems.append(
                f"molecules {bodies[key]!r} and {m.name!r} are the same "
                f"coordinates -- the same file ({Path(str(m.path)).name}) and "
                f"the same chain(s) {list(key[1]) or '(all chains)'}. They "
                f"would be scored as two bodies in one space, which measures "
                f"as an 'interaction energy' in the tens of thousands. For a "
                f"homodimer, give the partner its own chain id, or its own "
                f"file with the second copy's coordinates."
            )
        bodies[key] = m.name

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
    from .goals import build_goal
    from .milp import RotamerMILP, build_tables
    from .rotamers import build_template, harvest
    from .structure import Structure

    # The goal before the harvest, because a spec error is true whatever is
    # installed and a missing backend is not. Measured on CI (no PyMOL): a
    # spec naming a goal that does not exist was answered "backend missing: no
    # PyMOL available", which sends the reader to install something that would
    # not have helped. `validate` had always reported the goal correctly, so
    # the two commands disagreed about the same file.
    build_goal(spec).validate()

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
    out.write_text(lp, encoding="utf-8")

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
    from .goals import build_goal
    from .loop import IPROLoop

    # As in cmd_milp: the goal is checked before anything that needs a backend,
    # so a bad spec is reported as a bad spec on every machine.
    build_goal(spec).validate()

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
    # `is not None`, not truthiness: `--iterations 0` and `--max-minutes 0`
    # were silently ignored, so a request for a zero-length run quietly became
    # the spec's full run. Zero is now refused by LoopSettings, which is the
    # honest answer to a nonsensical budget.
    if args.iterations is not None:
        spec.loop.iterations = args.iterations
    if args.seed is not None:
        spec.loop.random_seed = args.seed
    if args.max_minutes is not None:
        spec.loop.max_minutes = args.max_minutes
    try:
        spec.loop.__post_init__()      # re-validate the overrides
    except SpecError as exc:
        print(f"spec error: {exc}", file=sys.stderr)
        return 2

    from .loop import run_design

    run_dir = args.run_dir or f"runs/{spec.name}"
    # A second run into the same directory APPENDED to the first one's
    # trajectory: measured, two 6-iteration runs of different specs left
    # iterations.jsonl with 12 lines, status reporting "12 of 6 iterations",
    # and best_score carried over from the other spec's run -- a result
    # attributed to a spec that never produced it. The MCP server already
    # refused this; the CLI did not.
    existing = Path(run_dir)
    if existing.exists() and not args.resume:
        used = [
            n for n in ("state.json", "iterations.jsonl", "best", "ensemble")
            if (existing / n).exists()
        ]
        if used:
            print(
                f"run directory already holds a run: {existing} ({', '.join(used)}).\n"
                f"Its trajectory would be appended to, and the status would "
                f"report both runs as one. Pass a fresh --run-dir, read this "
                f"one with `ipro-agent status {existing}`, or pass --resume to "
                f"continue it deliberately.",
                file=sys.stderr,
            )
            return 2
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


# ---------------------------------------------------------------------------
# inspect / fetch -- answering "what is in this structure" and "get me one"
# ---------------------------------------------------------------------------


def cmd_inspect(args) -> int:
    """What is actually in a structure file, before writing a spec about it.

    This existed nowhere, and its absence is why "you may change residues 18,
    20, 38 and 40" was not actionable: nothing could say which chain those
    numbers live in, what the native residues are, or whether they exist at
    all. An agent had to guess, and a guess here silently designs a different
    residue (docs/DESIGN.md section 7b defect 14).
    """
    from .structure import Structure, StructureError

    try:
        st = Structure.load(args.structure, chains=args.chain or None)
    except StructureError as exc:
        print(f"cannot read structure: {exc}", file=sys.stderr)
        return 2

    out: dict = {"structure": str(Path(args.structure)), "chains": []}
    for ch in st.chain_ids():
        residues = list(st.residues(ch))
        if not residues:
            continue
        refs = [st.ref(r) for r in residues]
        out["chains"].append({
            "chain": ch,
            "residues": len(residues),
            "first": refs[0].label,
            "last": refs[-1].label,
            "numbering_is_contiguous": [r.number for r in refs]
                == list(range(refs[0].number, refs[0].number + len(refs))),
            "sequence": st.sequence(ch),
        })
    het = [st.ref(r) for r in st.hetero_residues()]
    out["hetero_residues"] = [
        {"label": r.label, "name": r.name} for r in het[:40]
    ]
    out["hetero_count"] = len(het)

    # The point of the command: what is AT the positions someone named.
    if args.positions:
        from .config import DesignPosition

        asked = []
        for token in args.positions.replace(",", " ").split():
            try:
                pos = DesignPosition.parse(token)
            except SpecError as exc:
                asked.append({"asked": token, "error": str(exc)})
                continue
            try:
                res = st.find(pos.chain, pos.number, pos.insertion_code)
                ref = st.ref(res)
                asked.append({
                    "asked": token, "resolves_to": ref.label,
                    "native": ref.name,
                    "spec_form": f"- {{position: {ref.label}, allowed: [standard]}}",
                })
            except StructureError as exc:
                asked.append({"asked": token, "resolves_to": pos.label,
                              "error": str(exc)})
        out["positions"] = asked

    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    print(f"structure: {out['structure']}")
    for c in out["chains"]:
        gap = "" if c["numbering_is_contiguous"] else "   (numbering has gaps)"
        print(f"  chain {c['chain']}: {c['residues']} residues, "
              f"{c['first']}..{c['last']}{gap}")
        print(f"    {c['sequence']}")
    if out["hetero_count"]:
        names = ", ".join(f"{h['name']}{h['label']}" for h in out["hetero_residues"][:12])
        print(f"  hetero/ligands ({out['hetero_count']}): {names}")
    for a in out.get("positions", []):
        if "error" in a:
            print(f"  {a['asked']}: NOT FOUND -- {a['error']}")
        else:
            print(f"  {a['asked']} -> {a['resolves_to']} is {a['native']}")
            print(f"      {a['spec_form']}")
    if out.get("positions"):
        print("\n  Residue numbers in a spec are per chain. A bare number means "
              "chain A;\n  write '1:115' for chain 1. See docs/DESIGN.md 1b.")
    return 0


#: The only command in this package that touches the network, and the only
#: host it will contact. Everything else is offline by design.
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def cmd_fetch(args) -> int:
    """Download a structure from the RCSB PDB by its ID.

    Deliberately a command of its own rather than something a spec can trigger:
    it is the single networked operation here, and a design run must never
    silently reach the internet for its inputs.
    """
    import re
    import urllib.error
    import urllib.request

    pdb_id = str(args.pdb_id).strip().upper()
    if not re.fullmatch(r"[0-9A-Z]{4}", pdb_id):
        print(
            f"not a PDB ID: {args.pdb_id!r}. Expected four characters, e.g. 2OMF "
            f"or 1CRN. (Entries also have a longer 'pdb_00002omf' form; give the "
            f"short one.)",
            file=sys.stderr,
        )
        return 2

    url = RCSB_URL.format(pdb_id=pdb_id)
    out_path = Path(args.out or f"{pdb_id}.pdb")
    if out_path.exists() and not args.force:
        print(f"{out_path} already exists; pass --force to overwrite",
              file=sys.stderr)
        return 2
    try:
        with urllib.request.urlopen(url, timeout=args.timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        print(f"RCSB returned {exc.code} for {pdb_id}: no such entry?\n  {url}",
              file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"could not reach RCSB ({type(exc).__name__}: {exc}).\n"
              f"  This is the only command that needs the network. Download "
              f"{url}\n  by hand and pass the file directly.", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(body)

    result = {"pdb_id": pdb_id, "url": url, "path": str(out_path),
              "bytes": len(body)}

    if args.chain:
        # Reduce to the requested chain(s), which is what a design usually
        # wants: a spec naming residue numbers is per chain.
        from .structure import Structure, StructureError

        try:
            st = Structure.load(out_path, chains=list(args.chain))
            reduced = out_path.with_name(
                f"{out_path.stem}_{''.join(args.chain)}{out_path.suffix}")
            st.save(reduced)
            result["chain_only"] = str(reduced)
            result["chains_kept"] = list(args.chain)
        except StructureError as exc:
            print(f"downloaded {out_path}, but could not reduce to chain "
                  f"{args.chain}: {exc}", file=sys.stderr)
            result["chain_error"] = str(exc)

    result["next_step"] = (
        f"python3 -m ipro_agent inspect {result.get('chain_only', result['path'])} "
        f"--positions A18,A20"
    )
    result["provenance_note"] = (
        f"Record that this came from {url}. PDB coordinate files carry no "
        f"copyright restriction, but a design is only as good as the structure: "
        f"check the resolution, missing residues and any engineered mutations "
        f"before trusting it."
    )
    print(json.dumps(result, indent=2) if args.json else
          "\n".join(f"{k}: {v}" for k, v in result.items()))
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
    p.add_argument("--resume", action="store_true",
                   help="continue a run directory that already holds one "
                        "(without this, a used directory is refused rather "
                        "than appended to)")
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

    p = add_json(sub.add_parser("inspect", help="what is in a structure file"))
    p.add_argument("structure")
    p.add_argument("--chain", action="append",
                   help="restrict to this chain (repeatable)")
    p.add_argument("--positions",
                   help="residues to report the native identity of, e.g. 'A18,A20,38'")
    p.set_defaults(func=cmd_inspect)

    p = add_json(sub.add_parser(
        "fetch", help="download a structure from the RCSB PDB (the only networked command)"))
    p.add_argument("pdb_id", help="four-character PDB ID, e.g. 2OMF")
    p.add_argument("-o", "--out", help="output path (default <ID>.pdb)")
    p.add_argument("--chain", action="append",
                   help="also write a copy reduced to this chain (repeatable)")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    p.add_argument("--timeout", type=float, default=60.0)
    p.set_defaults(func=cmd_fetch)

    p = add_json(sub.add_parser("goals", help="list goals and what each optimizes"))
    p.add_argument("--statement", help="a plain-language goal, to get a suggestion")
    p.set_defaults(func=cmd_goals)

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except SpecError as exc:
        # `validate` reported these cleanly while `score`, `milp`, `positions`
        # and `dock` died with a 20-frame traceback for the SAME spec -- an
        # unknown goal name, a molecule with no path -- because only _load was
        # guarded and the error came later, from build_goal or from the loop's
        # constructor. One handler here gives every subcommand the contract
        # _load already had.
        print(f"spec error: {exc}", file=sys.stderr)
        return 2
    except StructureError as exc:
        print(f"structure error: {exc}", file=sys.stderr)
        return 2
    except RotamerError as exc:
        # `milp` and `score` both build rotamer tables, so both need PyMOL --
        # and both died with a bare RotamerError traceback on a machine without
        # it. Measured on CI (ubuntu, no conda): two of three failures were
        # this. The message already says what to install; what was missing was
        # a handler, so an agent saw 20 frames instead of one sentence.
        print(f"backend missing: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
