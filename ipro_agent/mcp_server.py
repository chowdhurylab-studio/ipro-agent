"""MCP server: this tool as typed tools an agent can call, without cloning it.

    python3 -m ipro_agent.mcp_server          # speaks MCP over stdio

Why this exists, and what it is *not*
-------------------------------------
The primary interface is still a clone plus `.claude/skills/ipro-agent/SKILL.md`:
an agent that can read the repo gets the operating manual, the degraded modes
and the reporting rules, which is strictly more than any tool schema conveys.
This server is for the other case -- an agent working in a different project, or
one that cannot read a repo at all -- which otherwise has to shell out blind.

It is a **thin wrapper over the CLI**, deliberately. Every tool shells out to
`python3 -m ipro_agent <subcommand> --json` rather than importing the loop, so
there is exactly one code path to keep correct and the server cannot drift from
the command line. No new science lives here.

It does **not** remove the PyMOL dependency. It is a local stdio server, so the
machine running it still needs the conda environment (`environment.yml`);
`ipro_doctor` reports what resolved and `ipro_run` refuses to launch when a
design is impossible, rather than starting a job that dies minutes later.

Long runs
---------
A real run is minutes to days, which no request/response call can hold open.
`ipro_run` therefore **launches detached** (its own session, output to a log in
the run directory) and returns immediately with a run id; `ipro_status` polls.
That is the same split the CLI and SKILL.md already use, so the shapes match.

Transport
---------
stdio, newline-delimited JSON-RPC 2.0, no dependencies beyond the standard
library -- the same reason `scripts/selftest.py` exists instead of a pytest
requirement. Protocol: MCP `tools/list` + `tools/call`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Protocol revisions this server is known to speak. The client's requested
#: version is echoed back when it is one of these, which is what the spec asks
#: for; otherwise the newest is offered and the client may refuse.
_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

SERVER_INFO = {"name": "ipro-agent", "version": "0.1.0"}

#: Seconds before a *synchronous* tool is abandoned. `ipro_run` is exempt: it
#: does not wait. `score` and `milp` build the full energy tables and can take
#: a couple of minutes on a large design set, hence the generous default.
DEFAULT_TIMEOUT = 900.0


# ---------------------------------------------------------------------------
# Spec handling
# ---------------------------------------------------------------------------

_SPEC_PROPERTY = {
    "spec": {
        "type": "string",
        "description": (
            "Path to a spec YAML/JSON file, OR the spec's YAML content inline. "
            "Inline content is written to a temporary file first. Relative "
            "molecule paths inside a spec resolve against the spec file's own "
            "directory, so inline specs must use absolute structure paths."
        ),
    }
}


def _looks_inline(spec: str) -> bool:
    """Inline YAML rather than a path.

    A path is one line and has no colon-space; a spec always has at least
    ``goal:`` on a line of its own. Checked this way rather than by trying to
    open it, so that a typo'd path is reported as a missing file instead of
    being silently parsed as YAML.
    """
    if "\n" in spec.strip():
        return True
    return False


def _resolve_spec(spec: str, keep: list[Path]) -> str:
    """Return a filesystem path for ``spec``, writing inline content if needed."""
    if not _looks_inline(spec):
        p = Path(spec).expanduser()
        if not p.exists():
            raise ToolError(
                f"spec not found: {spec}\n"
                "Pass a path to a spec file, or the spec's YAML content inline "
                "(inline specs must use absolute paths for their structures)."
            )
        return str(p)
    tmp = Path(tempfile.mkdtemp(prefix="ipro_mcp_spec_")) / "spec.yaml"
    tmp.write_text(spec)
    keep.append(tmp)
    return str(tmp)


class ToolError(RuntimeError):
    """A tool failed in a way the agent should read, not a crash."""


# ---------------------------------------------------------------------------
# Running the CLI
# ---------------------------------------------------------------------------


def _cli(args: list[str], timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Run `python3 -m ipro_agent ... --json` and return the parsed object."""
    cmd = [sys.executable, "-m", "ipro_agent", *args, "--json"]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=str(REPO_ROOT)
        )
    except subprocess.TimeoutExpired:
        raise ToolError(
            f"`{' '.join(args)}` did not finish within {timeout:.0f}s. For a "
            "design, use ipro_run (which launches detached) and poll "
            "ipro_status instead of waiting on a single call."
        )
    out = (proc.stdout or "").strip()
    if not out:
        # The CLI refuses some requests on stderr with a one-line explanation
        # ("spec has no target molecule; pass --near or add one"). That sentence
        # is what the agent needs, so it is the error -- not a wrapper message
        # about there being no output.
        err = (proc.stderr or "").strip()
        raise ToolError(
            err[-800:] if err else
            f"`{' '.join(args)}` produced no output at all (exit {proc.returncode})"
        )
    # The CLI pretty-prints its JSON across many lines, so this cannot be a
    # per-line parse. Try the whole of stdout first; if something has written a
    # banner ahead of it, decode the first complete object instead of guessing
    # at line boundaries.
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    at = out.find("{")
    while at != -1:
        try:
            obj, _end = decoder.raw_decode(out[at:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        at = out.find("{", at + 1)
    # Not JSON at all. The CLI refuses some requests with a one-line
    # explanation rather than a JSON object ("spec has no target molecule; pass
    # --near or add one"), and that sentence is the most useful thing we have --
    # so it is surfaced as the error instead of being buried under a generic
    # "did not return JSON".
    prose = out if len(out) <= 600 else out[:600] + " ..."
    raise ToolError(
        prose if prose else
        f"`{' '.join(args)}` produced no usable output (exit {proc.returncode}); "
        f"stderr: {(proc.stderr or '')[-400:]}"
    )


def _doctor() -> dict:
    p = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "doctor.py"), "--json"],
        capture_output=True, text=True, timeout=600, cwd=str(REPO_ROOT),
    )
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        raise ToolError(f"doctor.py returned no JSON: {(p.stderr or '')[-400:]}")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def tool_doctor(_: dict) -> dict:
    """What is installed, and therefore what a run can and cannot do."""
    rep = _doctor()
    return {
        "can_run": rep.get("can_run"),
        "can_design": rep.get("can_design"),
        "missing_required": rep.get("missing_required", []),
        "degraded": rep.get("degraded", []),
        "resolved": rep.get("resolved", {}),
        "note": (
            "can_design is False without PyMOL: rotamers come only from its "
            "mutagenesis wizard and there is no fallback, so `run` cannot "
            "produce a design. Install with `conda env create -f "
            "environment.yml`. Report `resolved` and the run's notes with any "
            "design -- the mode changes what the numbers mean."
        ),
    }


def tool_goals(_: dict) -> dict:
    return _cli(["goals"])


def tool_validate(args: dict) -> dict:
    keep: list[Path] = []
    return _cli(["validate", _resolve_spec(args["spec"], keep)])


def tool_positions(args: dict) -> dict:
    keep: list[Path] = []
    cmd = ["positions", _resolve_spec(args["spec"], keep)]
    if args.get("near"):
        cmd += ["--near", str(args["near"])]
    if args.get("cutoff") is not None:
        cmd += ["--cutoff", str(args["cutoff"])]
    return _cli(cmd)


def tool_score(args: dict) -> dict:
    keep: list[Path] = []
    return _cli(["score", _resolve_spec(args["spec"], keep)])


def tool_milp(args: dict) -> dict:
    keep: list[Path] = []
    cmd = ["milp", _resolve_spec(args["spec"], keep)]
    if args.get("out"):
        cmd += ["-o", str(args["out"])]
    return _cli(cmd)


def tool_run(args: dict) -> dict:
    """Launch a design run detached and return its id immediately."""
    rep = _doctor()
    if not rep.get("can_run"):
        raise ToolError(
            "required packages are missing: "
            f"{', '.join(rep.get('missing_required', [])) or 'unknown'}. "
            "Install with `conda env create -f environment.yml`."
        )
    if not rep.get("can_design"):
        # Refused rather than launched. A run without PyMOL dies at the first
        # rotamer harvest, minutes in, having written a run directory that
        # looks like a started job.
        raise ToolError(
            "cannot design: PyMOL did not resolve, and it has no fallback -- "
            "rotamers come only from its mutagenesis wizard. `conda env create "
            "-f environment.yml` installs it. Use ipro_score or ipro_validate "
            "to work without it."
        )

    keep: list[Path] = []
    spec = _resolve_spec(args["spec"], keep)
    run_dir = Path(args.get("run_dir") or (REPO_ROOT / "runs" / _run_id(spec)))
    run_dir = run_dir.expanduser()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ToolError(
            f"run directory is not empty: {run_dir}\n"
            "Pass a fresh run_dir, or poll the existing one with ipro_status."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "ipro_agent", "run", spec, "--run-dir", str(run_dir)]
    for flag, key in (("--iterations", "iterations"), ("--seed", "seed"),
                      ("--max-minutes", "max_minutes")):
        if args.get(key) is not None:
            cmd += [flag, str(args[key])]

    log = run_dir / "launch.log"
    with open(log, "wb") as fh:
        proc = subprocess.Popen(
            cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            cwd=str(REPO_ROOT),
            # Its own session, so the run outlives this server and the client
            # that asked for it -- the same reason SKILL.md recommends setsid.
            start_new_session=True,
        )
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "pid": proc.pid,
        "log": str(log),
        "launched": True,
        "mode_note": (
            f"degraded backends: {rep.get('degraded') or 'none'}. "
            "Poll ipro_status; do not wait on this call. Report the run's "
            "`mode` and `notes` with any design."
        ),
    }


def _run_id(spec: str) -> str:
    """A stable-ish directory name from the spec's own name."""
    stem = Path(spec).stem.replace(" ", "_") or "run"
    base = REPO_ROOT / "runs"
    n = 1
    while (base / f"{stem}_{n:03d}").exists():
        n += 1
    return f"{stem}_{n:03d}"


def tool_status(args: dict) -> dict:
    run = args.get("run_dir") or args.get("run_id")
    if not run:
        raise ToolError("give run_dir (a path) or run_id (a directory name)")
    p = Path(run).expanduser()
    if not p.exists():
        p = REPO_ROOT / "runs" / str(run)
    if not p.exists():
        raise ToolError(f"no such run: {run}")
    rep = _cli(["status", str(p)], timeout=120)
    # The house rule, restated where an agent will see it. `mode` and `notes`
    # come from status itself (state.RunDir.as_json).
    rep["reporting_rule"] = (
        "Report `mode` and `notes` with any design. If accepted is 0 nothing "
        "was designed: the reported best_* is the starting point, so say the "
        "search found nothing and read the per-iteration reasons."
    )
    if rep.get("designed_nothing"):
        rep["reporting_rule"] = (
            "NOTHING WAS DESIGNED: every iteration was rejected, so best_score "
            "equals start_score and best_mutations is the input. Report this as "
            "'the search found nothing'. " + rep["reporting_rule"]
        )
    return rep


TOOLS: list[dict[str, Any]] = [
    {
        "name": "ipro_doctor",
        "description": (
            "What is installed and what a run can therefore do. Call this FIRST: "
            "it returns can_run, can_design and the degraded backend list, and "
            "the resolved mode must be reported with any design."
        ),
        "inputSchema": {"type": "object", "properties": {}},
        "handler": tool_doctor,
    },
    {
        "name": "ipro_goals",
        "description": (
            "The design objectives this tool can express (bind_tighter, "
            "switch_specificity, reshape_channel, improve_catalysis, stabilize), "
            "each with the paper it comes from. Use it to pick a goal before "
            "writing a spec."
        ),
        "inputSchema": {"type": "object", "properties": {}},
        "handler": tool_goals,
    },
    {
        "name": "ipro_validate",
        "description": (
            "Check a design spec without spending any CPU: parses it, resolves "
            "positions against the structure, reports the design-space size and "
            "every problem found. Always do this before ipro_run."
        ),
        "inputSchema": {
            "type": "object",
            "properties": dict(_SPEC_PROPERTY),
            "required": ["spec"],
        },
        "handler": tool_validate,
    },
    {
        "name": "ipro_positions",
        "description": (
            "Propose design positions from the structure: residues of the design "
            "molecule within a cutoff of a named partner, i.e. the interface or "
            "pocket. Use it when the user says 'the interface' instead of listing "
            "residue numbers."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                **_SPEC_PROPERTY,
                "near": {"type": "string",
                         "description": "name of the partner molecule to measure from"},
                "cutoff": {"type": "number",
                           "description": "distance cutoff in Angstroms (default 5)"},
            },
            "required": ["spec"],
        },
        "handler": tool_positions,
    },
    {
        "name": "ipro_score",
        "description": (
            "Measure a structure without designing anything: interaction energies "
            "per partner, the goal's score and its components, plus the run mode "
            "and notes. Use it for a wild-type baseline, or to get "
            "goal_options.reference_interaction before a constrained run."
        ),
        "inputSchema": {
            "type": "object",
            "properties": dict(_SPEC_PROPERTY),
            "required": ["spec"],
        },
        "handler": tool_score,
    },
    {
        "name": "ipro_milp",
        "description": (
            "Emit the rotamer-selection integer program as a CPLEX LP file and "
            "report its size. Use it to prove to a user that their stated "
            "constraints really are in the program before a long run."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                **_SPEC_PROPERTY,
                "out": {"type": "string", "description": "where to write the LP file"},
            },
            "required": ["spec"],
        },
        "handler": tool_milp,
    },
    {
        "name": "ipro_run",
        "description": (
            "Launch a design run and return immediately with a run id. The run "
            "is DETACHED and can take minutes to days, so poll ipro_status "
            "rather than waiting. Refuses to launch when a design is impossible "
            "(no PyMOL). Always pass max_minutes for an unattended run."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                **_SPEC_PROPERTY,
                "run_dir": {"type": "string",
                            "description": "output directory; must be empty or new"},
                "iterations": {"type": "integer", "description": "override loop.iterations"},
                "seed": {"type": "integer", "description": "override loop.random_seed"},
                "max_minutes": {"type": "number",
                                "description": "wall-clock budget; stops cleanly at a checkpoint"},
            },
            "required": ["spec"],
        },
        "handler": tool_run,
    },
    {
        "name": "ipro_status",
        "description": (
            "Progress of a run: iterations, how many were ACCEPTED, the best "
            "score and mutations, the mode and the notes. Cheap and bounded in "
            "size however long the run has gone on, so it is safe to poll. "
            "Never read iterations.jsonl directly -- it grows without limit."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_dir": {"type": "string", "description": "the run directory"},
                "run_id": {"type": "string",
                           "description": "or just the id ipro_run returned"},
            },
        },
        "handler": tool_status,
    },
]

_BY_NAME = {t["name"]: t for t in TOOLS}


def _tool_list() -> list[dict]:
    return [{k: t[k] for k in ("name", "description", "inputSchema")} for t in TOOLS]


# ---------------------------------------------------------------------------
# JSON-RPC plumbing
# ---------------------------------------------------------------------------


def handle(message: dict) -> dict | None:
    """Handle one JSON-RPC message; None means "a notification, send nothing"."""
    method = message.get("method")
    mid = message.get("id")
    params = message.get("params") or {}

    if method == "initialize":
        want = params.get("protocolVersion")
        version = want if want in _PROTOCOL_VERSIONS else _PROTOCOL_VERSIONS[0]
        return _ok(mid, {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": (
                "Protein redesign via IPRO. Call ipro_doctor first and report "
                "the mode it gives with any result. Validate a spec before "
                "running one. ipro_run is detached: poll ipro_status. A run "
                "with accepted: 0 designed nothing. Energies from the built-in "
                "backend are a screening scale, not kcal/mol comparable to the "
                "published papers. Every design is a hypothesis for the bench."
            ),
        })

    if method in ("notifications/initialized", "initialized", "notifications/cancelled"):
        return None

    if method == "ping":
        return _ok(mid, {})

    if method == "tools/list":
        return _ok(mid, {"tools": _tool_list()})

    if method == "tools/call":
        name = params.get("name")
        tool = _BY_NAME.get(name)
        if tool is None:
            return _err(mid, -32602, f"unknown tool {name!r}; have {sorted(_BY_NAME)}")
        try:
            result = tool["handler"](params.get("arguments") or {})
            return _ok(mid, _content(result))
        except ToolError as exc:
            # A tool-level failure, reported to the model rather than to the
            # transport, so it can read the message and adjust.
            return _ok(mid, _content({"error": str(exc)}, is_error=True))
        except Exception as exc:                              # pragma: no cover
            return _ok(mid, _content(
                {"error": f"{type(exc).__name__}: {exc}"}, is_error=True))

    if method in ("resources/list", "prompts/list"):
        return _ok(mid, {"resources": [], "prompts": []})

    return _err(mid, -32601, f"method not found: {method}")


def _content(payload: dict, is_error: bool = False) -> dict:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=1, default=str)}],
        "isError": bool(is_error),
    }


def _ok(mid, result) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _err(mid, code, message) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def serve(stdin=None, stdout=None) -> int:
    """Read newline-delimited JSON-RPC from stdin, write replies to stdout."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            stdout.write(json.dumps(_err(None, -32700, "parse error")) + "\n")
            stdout.flush()
            continue
        reply = handle(message)
        if reply is not None:
            stdout.write(json.dumps(reply, default=str) + "\n")
            stdout.flush()
    return 0


def main() -> int:
    if "--list-tools" in sys.argv:
        # A plain-text sanity check that needs no MCP client.
        for t in _tool_list():
            print(f"{t['name']}\n    {t['description']}")
        return 0
    return serve()


if __name__ == "__main__":
    sys.exit(main())
