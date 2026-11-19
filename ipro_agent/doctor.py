#!/usr/bin/env python3
"""Report which backends are available, and what a run would actually use.

Run this first, and whenever a run fails for an unclear reason:

    python3 -m ipro_agent.doctor           # human-readable
    python3 -m ipro_agent.doctor --json    # machine-readable
    python3 scripts/doctor.py              # the same thing, from a checkout

It lives **inside the package** rather than in ``scripts/`` because a pip
install gets no ``scripts/`` directory, and the MCP server shells out to the
doctor before it will launch anything. With the doctor outside the package,
``ipro_doctor``, ``ipro_setup`` and ``ipro_run`` all failed in a pip-installed
copy with "can't open file .../site-packages/scripts/doctor.py" -- which is
exactly the install shape README recommends first, and the other eight tools
kept working, so the server looked healthy right up to the point of designing
anything.

Exit status is 0 if the IPRO loop can run at all (the built-in fallbacks are
always available), and 1 only if even that is broken. A missing PyMOL,
PyRosetta or Vina is reported as a downgrade, not a failure -- see
docs/INSTALL.md for what each one buys you.

Stdlib only, on purpose: the doctor has to work in a broken environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

#: The checkout root when running from a clone; inside site-packages
#: otherwise. Only used to look for sibling files (environment.yml), and every
#: such use has to tolerate their absence -- see _repo_file().
HERE = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], timeout: float = 20.0) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            # Do not inherit a parent PYTHONPATH that could shadow modules.
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
        )
        return p.returncode, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, f"{type(exc).__name__}: {exc}"


def _module_in(python: str, module: str, timeout: float = 120.0) -> tuple[bool, str]:
    """Is ``module`` importable by the interpreter at ``python``?

    ``-I`` matches how the run itself probes (``rotamers.py``, ``mutate.py``)
    and how it will import: without it the current directory is on
    ``sys.path`` and ``PYTHONPATH`` is inherited despite the comment in
    ``_run``, so a stray ``pyrosetta.py`` or a ``vina/`` directory in the
    working directory made doctor report both as installed -- with the whole
    "degraded" block gone, so the run looked not degraded at all, and the mode
    line the house rules require was wrong in both directions.

    The timeout is generous because PyRosetta's import genuinely takes tens of
    seconds; at the old 20 s a machine that HAS PyRosetta was reported as not
    having it, while ``backends.energy: auto`` resolved it in-process anyway.
    """
    code = (
        "import importlib,sys\n"
        f"m=importlib.import_module({module!r})\n"
        # `version` may be a function (pyrosetta's is), and printing the
        # function's repr put "<function version at 0x...>" in the mode report.
        "v=getattr(m,'__version__','') or getattr(m,'version','')\n"
        "print(v() if callable(v) else v)\n"
    )
    rc, out = _run([python, "-I", "-c", code], timeout=timeout)
    return rc == 0, out.splitlines()[-1] if (rc == 0 and out) else out


def conda_env_roots() -> list[Path]:
    """Every directory that plausibly holds conda/mamba environments.

    The old list had four entries and missed the **forge** family entirely --
    including Miniforge, which is what ``docs/INSTALL.md`` and ``ipro_setup``
    tell the user to install, and which is the only supported installer on
    Apple silicon. So the tool could not find the environment it had just
    asked for: a user who followed the instructions exactly got "PyMOL did not
    resolve" with the env sitting in ``~/miniforge3/envs/ipro-agent``.

    Kept here, in the one module that imports nothing from the package, so the
    doctor and the rotamer harvester cannot drift apart (they had two separate
    lists, and the doctor's had two roots the harvester's lacked -- so doctor
    said OK and the run then failed).
    """
    roots: list[Path] = []
    home = Path.home()
    for var in ("CONDA_PREFIX", "MAMBA_ROOT_PREFIX"):
        if os.environ.get(var):
            base = Path(os.environ[var])
            roots += [base / "envs", base.parent / "envs"]
    for var in ("CONDA_ENVS_PATH", "CONDA_ENVS_DIRS"):
        for part in (os.environ.get(var) or "").split(os.pathsep):
            if part.strip():
                roots.append(Path(part.strip()))
    roots += [
        home / "miniforge3" / "envs",
        home / "mambaforge" / "envs",
        home / "miniforge" / "envs",
        home / "micromamba" / "envs",
        home / "miniconda3" / "envs",
        home / "anaconda3" / "envs",
        home / "mamba" / "envs",
        Path("/opt/miniforge3/envs"),
        Path("/opt/miniconda3/envs"),
        Path("/opt/anaconda3/envs"),
        Path("/opt/homebrew/Caskroom/miniconda/base/envs"),
        Path("/opt/homebrew/Caskroom/miniforge/base/envs"),
        Path("/usr/share/miniconda/envs"),
        Path("/usr/local/miniconda3/envs"),
    ]
    # Windows layouts, for the same reason: cheap to list, and a wrong guess
    # costs nothing because every root is checked for existence.
    profile = os.environ.get("USERPROFILE")
    if profile:
        for name in ("miniforge3", "mambaforge", "miniconda3", "anaconda3"):
            roots.append(Path(profile) / name / "envs")
    seen: set[str] = set()
    out: list[Path] = []
    for r in roots:
        key = str(r)
        if key in seen:
            continue
        seen.add(key)
        if r.is_dir():
            out.append(r)
    return out


def env_interpreter(env: Path) -> "Path | None":
    """The python inside one environment directory, POSIX or Windows."""
    for rel in (("bin", "python"), ("bin", "python3"),
                ("Scripts", "python.exe"), ("python.exe",)):
        py = env.joinpath(*rel)
        if py.exists():
            return py
    return None


def find_conda_envs() -> list[Path]:
    """Interpreters worth probing for PyMOL/PyRosetta, newest-looking first."""
    cands: list[Path] = []
    for root in conda_env_roots():
        for env in sorted(root.iterdir()):
            py = env_interpreter(env)
            if py is not None:
                cands.append(py)
    for base in (Path("/opt/miniconda3/bin/python"),
                 Path("/opt/anaconda3/bin/python"),
                 Path.home() / "miniforge3" / "bin" / "python"):
        if base.exists():
            cands.append(base)
    return cands


#: Hardcoded, not imported from the package: the doctor must still work when
#: the package cannot be imported at all, which is the case this check is for.
PYTHON_FLOOR = (3, 10)


def check() -> dict:
    report: dict = {"python": sys.version.split()[0], "executable": sys.executable, "checks": {}}
    C = report["checks"]

    # -- PyMOL: mutagenesis ----------------------------------------------
    pymol: dict = {"available": False, "how": None, "detail": ""}
    # IPRO_PYMOL_PYTHON is the documented pin and rotamers.find_pymol_python
    # honours it, so doctor has to as well: otherwise doctor's verdict and the
    # run's behaviour disagree for exactly the user who set it.
    pinned = os.environ.get("IPRO_PYMOL_PYTHON")
    if pinned:
        ok, ver = _module_in(pinned, "pymol")
        if ok:
            pymol.update(
                available=True, how="env", python=pinned,
                detail=f"IPRO_PYMOL_PYTHON={pinned} {ver}".strip(),
            )
        else:
            pymol["detail"] = f"IPRO_PYMOL_PYTHON={pinned} cannot import pymol: {ver}"
    ok, ver = (False, pymol["detail"]) if pymol["available"] else _module_in(sys.executable, "pymol")
    if ok:
        pymol.update(available=True, how="import", python=sys.executable, detail=ver)
    else:
        exe = shutil.which("pymol")
        if exe:
            rc, out = _run([exe, "-cq", "-d", "print('ok')"])
            if rc == 0:
                pymol.update(available=True, how="binary", binary=exe, detail=out[:200])
        if not pymol["available"]:
            for py in find_conda_envs():
                ok, ver = _module_in(str(py), "pymol")
                if ok:
                    pymol.update(
                        available=True, how="subprocess-env", python=str(py),
                        detail=f"found in {py.parent.parent.name}",
                    )
                    break
    C["pymol"] = pymol

    # -- PyRosetta: scoring + relaxation ---------------------------------
    #
    # Unlike PyMOL, PyRosetta has to be importable by **this** interpreter:
    # energy/base.resolve() does `from .pyrosetta_backend import ...`, which
    # imports pyrosetta in-process. Finding it in another conda environment
    # therefore does NOT mean a run would use it -- and reporting it as
    # `energy -> pyrosetta` was wrong in exactly the way the house rules care
    # about, because the mode is reported with every design. Measured: with
    # PyRosetta installed only in the env `ipro_rosetta`, doctor run under the
    # base interpreter claimed `energy -> pyrosetta` while a run under that
    # same interpreter silently used the built-in screening function.
    pyros: dict = {"available": False, "how": None, "detail": ""}
    ok, ver = _module_in(sys.executable, "pyrosetta")
    if ok:
        pyros.update(available=True, how="import", python=sys.executable, detail=ver)
    else:
        for py in find_conda_envs():
            found, ver = _module_in(str(py), "pyrosetta")
            if found:
                # Recorded, but NOT available: a run with this interpreter
                # cannot use it.
                pyros.update(
                    available=False, how="elsewhere", python=str(py),
                    detail=(
                        f"installed in {py.parent.parent.name} ({py}) but NOT "
                        f"importable by this interpreter, so a run started with "
                        f"this python will use the built-in energy. Run "
                        f"ipro-agent with {py} to use it."
                    ),
                )
                break
    C["pyrosetta"] = pyros

    # -- AutoDock Vina: docking ------------------------------------------
    vina: dict = {"available": False, "how": None, "detail": ""}
    # Vina's GitHub releases ship versioned, platform-tagged names (e.g.
    # vina_1.2.7_mac_aarch64), which users often keep as-is on PATH.
    exe = next(
        (
            p
            for p in (
                shutil.which("vina"),
                shutil.which("vina.exe"),
                *(
                    shutil.which(f"vina_1.2.{minor}{suffix}")
                    for minor in range(9, -1, -1)
                    for suffix in ("", "_mac_aarch64", "_mac_x86_64", "_linux_x86_64")
                ),
            )
            if p
        ),
        None,
    )
    # IPRO_VINA is the documented pin, honoured by dock.find_vina.
    env_vina = os.environ.get("IPRO_VINA")
    if env_vina:
        rc, out = _run([env_vina, "--version"])
        vina.update(
            available=rc == 0, how="env", binary=env_vina,
            detail=(out.splitlines()[0] if out else "") + f"  [IPRO_VINA={env_vina}]",
        )
    if exe and not vina["available"]:
        rc, out = _run([exe, "--version"])
        vina.update(available=rc == 0, how="binary", binary=exe, detail=out.splitlines()[0] if out else "")
    if not vina["available"]:
        ok, ver = _module_in(sys.executable, "vina")
        if ok:
            vina.update(available=True, how="python", detail=f"vina python bindings {ver}")
    C["vina"] = vina

    # -- ligand prep ------------------------------------------------------
    prep: dict = {"available": False, "tools": []}
    for name in ("obabel", "mk_prepare_ligand.py", "prepare_ligand4.py", "prepare_receptor4.py"):
        p = shutil.which(name)
        if p:
            prep["tools"].append({"name": name, "path": p})
    ok, _ = _module_in(sys.executable, "meeko")
    if ok:
        prep["tools"].append({"name": "meeko", "path": "python module"})
    prep["available"] = bool(prep["tools"])
    C["ligand_prep"] = prep

    # -- MILP solvers -----------------------------------------------------
    milp: dict = {"available": False, "solvers": [], "preferred": None}
    ok, ver = _module_in(sys.executable, "pulp")
    pulp_ok = ok
    milp["pulp"] = {"available": ok, "version": ver if ok else ""}
    if ok:
        # Ask PuLP which of its solvers are actually usable here.
        rc, out = _run([
            sys.executable, "-c",
            "import pulp,json;print(json.dumps(pulp.listSolvers(onlyAvailable=True)))",
        ])
        if rc == 0:
            try:
                milp["solvers"] = json.loads(out.splitlines()[-1])
            except json.JSONDecodeError:
                pass
    for name, exe in (("glpk", "glpsol"), ("cbc", "cbc"), ("cplex", "cplex"), ("gurobi", "gurobi_cl"),
                      ("highs", "highs"), ("scip", "scip")):
        p = shutil.which(exe)
        if p:
            milp.setdefault("binaries", []).append({"name": name, "path": p})
    # Preference order must match milp.py's, or this reports a solver the code
    # will not actually use. Solvers reachable only through PuLP are listed
    # first (PuLP drives them directly); otherwise only the solver binaries
    # milp.py can parse an LP solution from count.
    for want in (
        "CPLEX_CMD", "GUROBI_CMD", "PULP_CBC_CMD", "COIN_CMD",
        "HiGHS", "HiGHS_CMD", "SCIP_CMD", "GLPK_CMD",
    ):
        if want in milp.get("solvers", []):
            milp["preferred"] = f"{want} (via pulp)"
            break
    if milp["preferred"] is None:
        # Without PuLP, milp.py shells out to an LP file and can only read
        # glpsol's and cbc's solution formats.
        usable = [b for b in milp.get("binaries", []) if b["name"] in ("glpk", "cbc")]
        if usable:
            milp["preferred"] = f"{usable[0]['name']} (via LP file)"
        elif milp.get("binaries"):
            others = ", ".join(b["name"] for b in milp["binaries"])
            milp["preferred"] = (
                f"greedy (built-in; NOT a proof of optimality) -- {others} found but "
                "unusable without pulp: install pulp to drive it"
            )
    if milp["preferred"] is None:
        milp["preferred"] = "greedy (built-in; NOT a proof of optimality)"
    # "available" must mean "a solver this code can actually drive", not "some
    # solver binary is on PATH". It reported True for a binary doctor had just
    # called unusable, so `doctor.py --require milp && run` went ahead with the
    # greedy heuristic -- which proves no optimality -- and the MILP never
    # appeared in `degraded`, so no caller could see it either.
    milp["available"] = not milp["preferred"].startswith("greedy")
    C["milp"] = milp

    # -- structure handling ------------------------------------------------
    for mod in ("numpy", "Bio", "yaml", "scipy"):
        ok, ver = _module_in(sys.executable, mod)
        C[f"py:{mod}"] = {"available": ok, "version": ver if ok else ""}

    # -- resolve what a run would use ------------------------------------
    report["resolved"] = {
        "mutate": "pymol" if pymol["available"] else "builtin (rotamer-library sidechain swap)",
        "energy": "pyrosetta" if pyros["available"] else "builtin (LJ + capped repulsion + Coulomb + H-bond well; NO desolvation term)",
        "dock": "vina" if vina["available"] else "none (input poses kept; set redock:false)",
        "milp": milp["preferred"],
    }
    # Two different questions, kept apart because conflating them misreported
    # the mode. `can_run` is "the package imports at all" and needs all three
    # required modules -- it used to check numpy only, so a missing Biopython
    # or PyYAML read as fine. `can_design` is "a design can actually be
    # produced", which additionally needs PyMOL: rotamers come only from its
    # mutagenesis wizard, there is no built-in rotamer builder, and
    # `backends.mutate: builtin` is refused by the config. Reporting can_run
    # True with no PyMOL told an agent the gate was green when `run` could not
    # produce a design at all -- and AGENTS.md tells agents to trust this gate.
    required_py = ["py:numpy", "py:Bio", "py:yaml"]
    report["missing_required"] = [
        k.split(":", 1)[1] for k in required_py if not C[k]["available"]
    ]
    # The interpreter is a requirement too, and it was not checked. On Python
    # 3.9 with all three modules installed, this reported can_run True,
    # can_design True, missing_required [] and exited 0 -- and `--require
    # pymol`, the mandatory gate, also passed -- for an interpreter on which
    # `import ipro_agent` raises straight away (the package floor is 3.10, and
    # the code uses 3.10 syntax). The green light was the worst possible
    # answer there: it sends the user into a run instead of into an upgrade.
    report["python_ok"] = sys.version_info[:2] >= PYTHON_FLOOR
    report["python_floor"] = ".".join(str(n) for n in PYTHON_FLOOR)
    if not report["python_ok"]:
        report["missing_required"].append(
            f"python>={report['python_floor']} (this is "
            f"{'.'.join(str(n) for n in sys.version_info[:3])})"
        )
    report["can_run"] = not report["missing_required"]
    report["can_design"] = bool(report["can_run"] and pymol["available"])
    report["degraded"] = [
        k for k, v in (("pymol", pymol), ("pyrosetta", pyros), ("vina", vina),
                       ("milp", milp)) if not v["available"]
    ]
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument(
        "--require", nargs="*", default=[],
        metavar="TOOL", help="exit 1 unless each of these is available (pymol pyrosetta vina milp)",
    )
    args = ap.parse_args()

    rep = check()
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        print(f"ipro-agent doctor  (python {rep['python']})")
        print(f"  interpreter: {rep['executable']}")
        print("\nbackends")
        for key, label in (("pymol", "PyMOL (mutagenesis)"),
                           ("pyrosetta", "PyRosetta (scoring + relax)"),
                           ("vina", "AutoDock Vina (docking)")):
            c = rep["checks"][key]
            mark = "OK  " if c["available"] else "--  "
            extra = f"  [{c.get('how')}] {c.get('detail', '')}".rstrip()
            if c["available"]:
                print(f"  {mark}{label}{extra}")
            elif c.get("detail"):
                # Not usable here, but we know something useful about why --
                # "installed in another env" is the case worth printing rather
                # than swallowing behind a bare "(not found)".
                print(f"  {mark}{label}  (not usable here)")
                print(f"        {c['detail']}")
            else:
                print(f"  {mark}{label}  (not found)")
        lp = rep["checks"]["ligand_prep"]
        print(f"  {'OK  ' if lp['available'] else '--  '}ligand prep: "
              + (", ".join(t["name"] for t in lp["tools"]) if lp["tools"] else "none"))
        m = rep["checks"]["milp"]
        print(f"  {'OK  ' if m['available'] else '--  '}MILP: pulp={m['pulp']['available']} "
              f"solvers={m.get('solvers') or []} binaries="
              f"{[b['name'] for b in m.get('binaries', [])]}")
        print("\nwhat a run will use")
        for k, v in rep["resolved"].items():
            print(f"  {k:8s} -> {v}")
        # Name what is missing. This block used to print only "can run: False",
        # leaving the reader to guess which of numpy, Biopython or PyYAML it
        # meant.
        if rep["missing_required"]:
            print("\nREQUIRED and missing: " + ", ".join(rep["missing_required"]))
            print("  conda env create -f environment.yml   (or: pip install -r requirements.txt)")
        optional_missing = [d for d in rep["degraded"] if d != "pymol"]
        if optional_missing:
            print("\nmissing (a run is still possible, with the fallbacks above):")
            print("  " + ", ".join(optional_missing))
            print("  see docs/INSTALL.md for what each one changes")
        if not rep["checks"]["pymol"]["available"]:
            print("\nPyMOL is MISSING and has no fallback: rotamers come only from its")
            print("  mutagenesis wizard, so `run` cannot produce a design. validate,")
            print("  positions, milp, score and goals still work.")
            print("  conda env create -f environment.yml   (pymol-open-source is conda-only)")
        if not rep.get("python_ok", True):
            print(
                f"\nPYTHON TOO OLD: this is {rep['python']}, and ipro-agent "
                f"needs {rep['python_floor']} or newer -- `import ipro_agent` "
                f"fails on anything older, whatever else is installed. Create "
                f"the environment (conda env create -f environment.yml) or use "
                f"a newer interpreter."
            )
        print(f"\ncan run:    {rep['can_run']}")
        # The suffix used to say "no PyMOL" whatever the real cause: with
        # numpy, Biopython and PyYAML all missing but PyMOL resolved, it still
        # blamed PyMOL. Name what is actually missing.
        why = ""
        if not rep["can_design"]:
            if not rep.get("python_ok", True):
                why = "   <- Python too old, see above"
            elif rep["missing_required"]:
                why = ("   <- missing required Python module(s): "
                       + ", ".join(rep["missing_required"]))
            elif not rep["checks"]["pymol"]["available"]:
                why = "   <- no PyMOL, see above"
        print(f"can design: {rep['can_design']}" + why)

    missing = [r for r in args.require if not rep["checks"].get(r, {}).get("available")]
    if missing:
        print(f"\nrequired but missing: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0 if rep["can_run"] else 1


if __name__ == "__main__":
    sys.exit(main())
