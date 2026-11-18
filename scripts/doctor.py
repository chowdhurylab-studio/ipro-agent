#!/usr/bin/env python3
"""Report which backends are available, and what a run would actually use.

Run this first, and whenever a run fails for an unclear reason:

    python3 scripts/doctor.py           # human-readable
    python3 scripts/doctor.py --json    # machine-readable

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


def _module_in(python: str, module: str) -> tuple[bool, str]:
    """Is ``module`` importable by the interpreter at ``python``?"""
    code = (
        "import importlib,sys\n"
        f"m=importlib.import_module({module!r})\n"
        "print(getattr(m,'__version__','') or getattr(m,'version',''))\n"
    )
    rc, out = _run([python, "-c", code])
    return rc == 0, out.splitlines()[-1] if (rc == 0 and out) else out


def find_conda_envs() -> list[Path]:
    """Interpreters worth probing for PyMOL/PyRosetta, newest-looking first."""
    cands: list[Path] = []
    for root in (
        Path(os.environ.get("CONDA_PREFIX", "")).parent / "envs" if os.environ.get("CONDA_PREFIX") else None,
        Path.home() / "miniconda3" / "envs",
        Path.home() / "anaconda3" / "envs",
        Path("/opt/miniconda3/envs"),
        Path("/opt/anaconda3/envs"),
        Path("/opt/homebrew/Caskroom/miniconda/base/envs"),
    ):
        if root and root.is_dir():
            for env in sorted(root.iterdir()):
                py = env / "bin" / "python"
                if py.exists():
                    cands.append(py)
    for base in (Path("/opt/miniconda3/bin/python"), Path("/opt/anaconda3/bin/python")):
        if base.exists():
            cands.append(base)
    return cands


def check() -> dict:
    report: dict = {"python": sys.version.split()[0], "executable": sys.executable, "checks": {}}
    C = report["checks"]

    # -- PyMOL: mutagenesis ----------------------------------------------
    pymol: dict = {"available": False, "how": None, "detail": ""}
    ok, ver = _module_in(sys.executable, "pymol")
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
    pyros: dict = {"available": False, "how": None, "detail": ""}
    ok, ver = _module_in(sys.executable, "pyrosetta")
    if ok:
        pyros.update(available=True, how="import", python=sys.executable, detail=ver)
    else:
        for py in find_conda_envs():
            ok, ver = _module_in(str(py), "pyrosetta")
            if ok:
                pyros.update(
                    available=True, how="subprocess-env", python=str(py),
                    detail=f"found in {py.parent.parent.name}",
                )
                break
    C["pyrosetta"] = pyros

    # -- AutoDock Vina: docking ------------------------------------------
    vina: dict = {"available": False, "how": None, "detail": ""}
    exe = shutil.which("vina") or shutil.which("vina_1.2.5") or shutil.which("vina.exe")
    if exe:
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
    # Preference order: a commercial solver if present (IPRO used CPLEX), then
    # open ones. 'greedy' is the always-available fallback (see milp.py).
    for want in ("CPLEX_CMD", "GUROBI_CMD", "PULP_CBC_CMD", "COIN_CMD", "HiGHS", "HiGHS_CMD", "GLPK_CMD"):
        if want in milp.get("solvers", []):
            milp["preferred"] = want
            break
    if milp["preferred"] is None and milp.get("binaries"):
        milp["preferred"] = f"{milp['binaries'][0]['name']} (via LP file)"
    if milp["preferred"] is None:
        milp["preferred"] = "greedy (built-in; NOT a proof of optimality)"
    milp["available"] = pulp_ok or bool(milp.get("binaries"))
    C["milp"] = milp

    # -- structure handling ------------------------------------------------
    for mod in ("numpy", "Bio", "yaml", "scipy"):
        ok, ver = _module_in(sys.executable, mod)
        C[f"py:{mod}"] = {"available": ok, "version": ver if ok else ""}

    # -- resolve what a run would use ------------------------------------
    report["resolved"] = {
        "mutate": "pymol" if pymol["available"] else "builtin (rotamer-library sidechain swap)",
        "energy": "pyrosetta" if pyros["available"] else "builtin (LJ + Coulomb + implicit-solvent approximation)",
        "dock": "vina" if vina["available"] else "none (input poses kept; set redock:false)",
        "milp": milp["preferred"],
    }
    report["can_run"] = bool(C["py:numpy"]["available"])
    report["degraded"] = [
        k for k, v in (("pymol", pymol), ("pyrosetta", pyros), ("vina", vina)) if not v["available"]
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
            print(f"  {mark}{label}{extra if c['available'] else '  (not found)'}")
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
        if rep["degraded"]:
            print("\nmissing (run is still possible, with the fallbacks above):")
            print("  " + ", ".join(rep["degraded"]))
            print("  see docs/INSTALL.md for what each one changes")
        print(f"\ncan run: {rep['can_run']}")

    missing = [r for r in args.require if not rep["checks"].get(r, {}).get("available")]
    if missing:
        print(f"\nrequired but missing: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0 if rep["can_run"] else 1


if __name__ == "__main__":
    sys.exit(main())
