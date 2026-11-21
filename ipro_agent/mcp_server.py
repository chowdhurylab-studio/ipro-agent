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

import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Paths here that have **not** been executed, in the same spirit as
#: ``energy.pyrosetta_backend.UNEXERCISED``: declared rather than left to be
#: discovered, and checked by a test so the list cannot rot.
UNEXERCISED: dict[str, str] = {
    "tool_setup(confirm=True) -- the conda invocation itself": (
        "The dry run, the manager discovery, the refusal and now the exact "
        "command handed to the shell are all exercised, the last with Popen "
        "stubbed, on both the clone path and the pip-installed path where "
        "there is no environment.yml. What is still not exercised is conda "
        "actually running: that would create a real multi-hundred-megabyte "
        "environment on the machine under test. This entry used to claim the "
        "command had been verified by hand, which is how it hid a defect for "
        "a while -- the launch ignored the command the plan printed and ran "
        "`conda env create -f None` on every pip install. A hand-verified "
        "claim is not a test, and should not be written here as though it "
        "were."
    ),
}

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


#: A line that only a spec would have. Used to tell inline YAML from a path
#: without opening anything, so that a typo'd path is reported as a missing
#: file instead of being silently parsed as YAML.
_SPEC_LINE = re.compile(
    r"^\s*(goal|goal_statement|molecules|positions|restraints|loop|"
    r"goal_options|assemblies|indels)\s*:", re.M
)


def _looks_inline(spec: str) -> bool:
    """Inline YAML rather than a path.

    Multi-line input is inline. A *single* line can be too -- an agent that
    holds the spec in context may well send ``{goal: bind_tighter, ...}`` or
    ``goal: bind_tighter`` with no newline at all, and those were treated as
    filenames and reported as "spec not found: goal: bind_tighter".
    """
    s = spec.strip()
    if "\n" in s:
        return True
    if _SPEC_LINE.search(s):
        return True
    # A YAML flow mapping, which is what a one-line spec looks like.
    return s.startswith("{") and s.endswith("}")


def _resolve_spec(spec: str, keep: list[Path]) -> str:
    """Return a filesystem path for ``spec``, writing inline content if needed."""
    if not isinstance(spec, str) or not spec.strip():
        raise ToolError(
            "`spec` must be a path to a spec file, or the spec's YAML content "
            f"inline; got {type(spec).__name__} "
            f"{'(empty)' if isinstance(spec, str) else repr(spec)[:80]}"
        )
    if not _looks_inline(spec):
        # resolve(), not expanduser() alone: the child runs with cwd=REPO_ROOT,
        # so a relative path the client meant against ITS own directory was
        # checked here and then handed over to be re-resolved somewhere else.
        # Measured: a spec sitting next to the client reported "spec not
        # found" while the file plainly existed.
        p = Path(spec).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if not p.exists():
            raise ToolError(
                f"spec not found: {spec}\n"
                f"(looked for {p}; the server's working directory is "
                f"{Path.cwd()})\n"
                "Pass an absolute path to a spec file, or the spec's YAML "
                "content inline (inline specs must use absolute paths for "
                "their structures)."
            )
        return str(p)
    tmp = Path(tempfile.mkdtemp(prefix="ipro_mcp_spec_")) / "spec.yaml"
    tmp.write_text(spec, encoding="utf-8")
    keep.append(tmp)
    return str(tmp)


@contextlib.contextmanager
def _spec_scratch():
    """Hold the temp files an inline spec needs, and delete them afterwards.

    Inline specs were written to ``tempfile.mkdtemp()`` and never removed. The
    list they were collected into was called ``keep`` and nothing ever read it.
    Measured by driving the server from the official MCP Python SDK under a GUI
    client's environment, where there is no ``TMPDIR`` and ``gettempdir()`` is
    therefore the shared ``/tmp``: five ``ipro_validate`` calls with an inline
    spec took ``/tmp/ipro_mcp_spec_*`` from 7 directories to 12, one per call,
    each holding a 0o644 ``spec.yaml``. All five of those calls *failed*, which
    is the part that matters -- an agent iterating on a spec it cannot get right
    leaks fastest, and the spec's text is the user's unpublished design problem
    sitting in a world-readable directory.

    Only the synchronous tools can use this: ``ipro_run``'s child reads the
    spec after the call returns, so it moves the file into the run directory
    instead (see ``tool_run``).
    """
    keep: list[Path] = []
    try:
        yield keep
    finally:
        for path in keep:
            shutil.rmtree(path.parent, ignore_errors=True)


def _client_path(value: Any, what: str, must_exist: bool = False) -> str:
    """Resolve a path the *client* gave against the *client's* directory.

    A relative path meant two different things depending on which tool received
    it. ``_resolve_spec`` resolved against ``Path.cwd()`` (the server's working
    directory, which an MCP config sets to the client's checkout), while
    ``tool_inspect`` and ``tool_fetch`` handed the string straight to ``_cli``,
    whose child runs with ``cwd=REPO_ROOT``. Measured with the server started
    from ``cwd=/``, as Claude Desktop starts one when its config omits ``cwd``:
    ``ipro_validate {"spec": "examples/ompf_pore/spec_smoke.yaml"}`` refused
    with "looked for /examples/ompf_pore/spec_smoke.yaml", while
    ``ipro_inspect_structure {"structure": "examples/ompf_pore/2omf_A.pdb"}``
    *succeeded* and reported 340 residues -- answering about a file inside this
    repository that the client had never named. A wrong structure that reports
    cleanly is the failure mode section 7 is about: every position, native
    residue and energy downstream of it belongs to the wrong molecule.
    """
    text = str(value).strip()
    if not text:
        raise ToolError(f"`{what}` is empty")
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if must_exist and not p.exists():
        raise ToolError(
            f"{what} not found: {text}\n"
            f"(looked for {p}; the server's working directory is {Path.cwd()})\n"
            "Pass an absolute path, or set the server's `cwd` to the directory "
            "those relative paths belong to."
        )
    return str(p)


def _writable_dir(preferred: "Path | None" = None) -> Path:
    """A directory this process may actually write to.

    Runs and LP files used to default under ``REPO_ROOT``, which is
    *site-packages* in a pip-installed copy: a run then wrote into the
    installed package tree, and on a read-only install failed with a chopped
    PermissionError traceback. Honour ``IPRO_RUN_ROOT``, then
    ``XDG_STATE_HOME``, then the user's home, then a temp directory.
    """
    cands = []
    if preferred is not None:
        cands.append(Path(preferred))
    for var in ("IPRO_RUN_ROOT", "XDG_STATE_HOME"):
        if os.environ.get(var):
            cands.append(Path(os.environ[var]).expanduser() / "ipro-agent")
    cands.append(Path.home() / ".local" / "state" / "ipro-agent")
    cands.append(Path(tempfile.gettempdir()) / "ipro-agent")
    for c in cands:
        try:
            c.mkdir(parents=True, exist_ok=True)
            probe = c / ".write_probe"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
            return c
        except OSError:
            continue
    return Path(tempfile.mkdtemp(prefix="ipro_agent_"))


class ToolError(RuntimeError):
    """A tool failed in a way the agent should read, not a crash."""


def _given(value: Any) -> bool:
    """Was this argument actually supplied?

    ``None`` is not a value: a client that sends ``{"structure": null}`` has
    given nothing, and ``str(None)`` is the four-character string "None",
    which every truthiness test in sight would accept.
    """
    return value is not None and str(value).strip() != ""


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
            _explain_stderr(err, args) if err else
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


#: `ValueError: ...`, `ipro_agent.config.SpecError: ...` -- the line of a
#: traceback that actually says what went wrong.
_EXC_LINE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Warning|Exit):", re.M)


def _explain_stderr(err: str, args: list[str]) -> str:
    """The useful sentence first, the traceback second.

    A tail slice of stderr handed the agent 800 characters that began in the
    middle of a line -- one measured message literally started ``ile "`` --
    with the actual exception last, where a model is least likely to act on it.
    """
    lines = [ln for ln in err.splitlines() if ln.strip()]
    hits = [ln.strip() for ln in lines if _EXC_LINE.match(ln.strip())]
    if not hits:
        return err[-800:]
    message = hits[-1]
    if "Traceback (most recent call last)" not in err:
        return message
    context = err[-400:]
    at = context.find("\n")
    if at != -1:
        context = context[at + 1:]
    return (
        f"{message}\n\n(from `{' '.join(args)}`; traceback tail follows)\n"
        f"{context}"
    )


def _doctor() -> dict:
    p = subprocess.run(
        # -m, not a path: a pip-installed copy has no scripts/ directory, and
        # ipro_doctor / ipro_setup / ipro_run all go through here. With the
        # path form all three failed in an installed copy with "can't open
        # file .../site-packages/scripts/doctor.py" while the other eight tools
        # worked, so the server looked healthy until a design was wanted.
        [sys.executable, "-m", "ipro_agent.doctor", "--json"],
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
    with _spec_scratch() as keep:
        return _cli(["validate", _resolve_spec(args["spec"], keep)])


def tool_positions(args: dict) -> dict:
    with _spec_scratch() as keep:
        cmd = ["positions", _resolve_spec(args["spec"], keep)]
        if args.get("near"):
            cmd += ["--near", str(args["near"])]
        if args.get("cutoff") is not None:
            cmd += ["--cutoff", str(args["cutoff"])]
        return _cli(cmd)


def tool_score(args: dict) -> dict:
    with _spec_scratch() as keep:
        cmd = ["score", _resolve_spec(args["spec"], keep)]
        # `score` writes a small run directory for its log. Defaulted under
        # REPO_ROOT, which is site-packages in a pip-installed copy, that wrote
        # into the installed package tree -- and failed outright on a read-only
        # install, with a PermissionError traceback chopped mid-line.
        cmd += ["--run-dir", str(_writable_dir(args.get("run_dir")) / "score")]
        return _cli(cmd)


def tool_milp(args: dict) -> dict:
    with _spec_scratch() as keep:
        cmd = ["milp", _resolve_spec(args["spec"], keep)]
        # Absolute, because the child runs with cwd=REPO_ROOT: a relative
        # `out` landed inside the installed package and the reported path was
        # relative, so the agent could not find the file it had just asked
        # for. An output rather than an input, so it goes to a directory this
        # process is known to be able to write to, not to wherever the client
        # happens to be standing.
        #
        # `-o` goes on unconditionally. Omitting it did not mean "no file": it
        # handed the child argparse's default, the bare name `model.lp`, which
        # it then resolved against the same REPO_ROOT -- so the no-`out` call,
        # the common one, kept exactly the behaviour the `out` branch existed
        # to prevent, and reported `{"lp_file": "model.lp"}` relative to a
        # directory the caller never named.
        out = args.get("out")
        q = Path(str(out)).expanduser() if out else Path("model.lp")
        if not q.is_absolute():
            q = _writable_dir() / q.name
        cmd += ["-o", str(q)]
        return _cli(cmd)


def tool_run(args: dict) -> dict:
    """Launch a design run detached and return its id immediately.

    The order of the checks is deliberate, and it is not the order this had.
    ``_doctor()`` used to be the very first act, so on a machine without PyMOL
    the can_design gate fired ahead of both the argument coercion and the
    non-empty-run_dir check: a typo in ``iterations`` was answered with
    "cannot design: PyMOL did not resolve", which is true and useless. Two of
    the three failures in this repo's first CI run were exactly that. It also
    spent a one-to-five-second doctor probe before telling a caller their
    argument was malformed.

    So: coerce the arguments, resolve the spec, work out the run directory and
    refuse a used one -- all free -- and only then probe the environment.
    Nothing is created on disk until every refusal has had its chance.
    """
    # 1. Arguments, which cost nothing to check and are what a model gets
    #    wrong. Coerced, not stringified: the schema says `iterations` is an
    #    integer, but nothing enforced it, so a JSON float -- which a model
    #    will happily emit for an integer field -- became `--iterations 8.0`,
    #    argparse exited 2 within milliseconds, and the reply was still
    #    `launched: true` with a pid that was already gone. ipro_status then
    #    answered "pending" for ever.
    flags: list[str] = []
    for flag, key, cast in (("--iterations", "iterations", int),
                            ("--seed", "seed", int),
                            ("--max-minutes", "max_minutes", float)):
        if args.get(key) is None:
            continue
        raw = args[key]
        try:
            value = cast(raw) if not isinstance(raw, bool) else None
            if value is None:
                raise TypeError("bool")
            if cast is int and isinstance(raw, float) and raw != int(raw):
                raise ValueError("not a whole number")
        except (TypeError, ValueError):
            raise ToolError(
                f"`{key}` must be a {'whole number' if cast is int else 'number'}; "
                f"got {type(raw).__name__} {raw!r}. Nothing was launched."
            )
        flags += [flag, str(value)]

    # 2. The spec, and the run directory it implies.
    keep: list[Path] = []
    spec = _resolve_spec(args["spec"], keep)
    try:
        run_dir = Path(args.get("run_dir") or (_writable_dir() / "runs" / _run_id(spec)))
        run_dir = run_dir.expanduser()
        if not run_dir.is_absolute():
            run_dir = (Path.cwd() / run_dir).resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ToolError(
                f"run directory is not empty: {run_dir}\n"
                "Pass a fresh run_dir, or poll the existing one with ipro_status."
            )

        # 3. Only now, the environment.
        rep = _doctor()
        if not rep.get("can_run"):
            raise ToolError(
                "required packages are missing: "
                f"{', '.join(rep.get('missing_required', [])) or 'unknown'}. "
                "Install with `conda env create -f environment.yml`."
            )
        if not rep.get("can_design"):
            # Refused rather than launched. A run without PyMOL dies at the
            # first rotamer harvest, minutes in, having written a run directory
            # that looks like a started job.
            raise ToolError(
                "cannot design: PyMOL did not resolve, and it has no fallback "
                "-- rotamers come only from its mutagenesis wizard. `conda env "
                "create -f environment.yml` installs it. Use ipro_validate or "
                "ipro_positions to work without it."
            )

        run_dir.mkdir(parents=True, exist_ok=True)
    except BaseException:
        # A refusal here -- a bad argument, a used run directory, an
        # unwritable one, a missing backend -- must not leave the inline spec
        # behind in /tmp either.
        for path in keep:
            shutil.rmtree(path.parent, ignore_errors=True)
        raise

    # An inline spec moves into the run directory rather than staying in a temp
    # directory nobody owns. The child outlives this call, so the file cannot
    # simply be deleted on the way out as it is for the synchronous tools
    # (`_spec_scratch`); keeping it beside the run it produced also means the
    # run records the spec that made it, which an inline spec otherwise leaves
    # nowhere. `run` only refuses a directory that already holds state.json,
    # iterations.jsonl, best or ensemble, so a spec.yaml here is not mistaken
    # for a previous run -- launch.log has been written here all along.
    if keep:
        landed = run_dir / "spec.yaml"
        shutil.move(str(keep[0]), str(landed))
        shutil.rmtree(keep[0].parent, ignore_errors=True)
        keep.clear()
        spec = str(landed)

    cmd = [sys.executable, "-m", "ipro_agent", "run", spec,
           "--run-dir", str(run_dir)] + flags

    log = run_dir / "launch.log"
    with open(log, "wb") as fh:
        # Its own session, so the run outlives this server and the client that
        # asked for it -- the same reason SKILL.md recommends setsid.
        # `start_new_session` is silently IGNORED on Windows (CPython names the
        # parameter `unused_start_new_session` there), so the detach needs the
        # creation flags instead. Untested on Windows -- see README's platform
        # note -- but wrong-by-omission is worse than untested.
        detach: dict = {}
        if os.name == "nt":                                # pragma: no cover
            detach["creationflags"] = (
                getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
            )
        else:
            detach["start_new_session"] = True
        proc = subprocess.Popen(
            cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            cwd=str(REPO_ROOT), **detach,
        )
    # Did it survive the first moment? A spec the CLI refuses, or a bad
    # argument, kills the child instantly -- and "launched: true" then sends
    # the agent into an endless poll of a run that never started.
    rc = None
    for _ in range(20):
        rc = proc.poll()
        if rc is not None:
            break
        time.sleep(0.1)
    if rc is not None and rc != 0:
        tail = ""
        try:
            tail = log.read_text(encoding="utf-8", errors="replace").strip()[-600:]
        except OSError:
            pass
        raise ToolError(
            f"the run exited immediately (status {rc}) and nothing is running. "
            f"Its output:\n{tail or '(launch.log is empty)'}"
        )
    return {
        "run_id": str(run_dir),
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
    base = _writable_dir() / "runs"
    n = 1
    while (base / f"{stem}_{n:03d}").exists():
        n += 1
    return f"{stem}_{n:03d}"


def _resolve_run(args: dict) -> Path:
    """The run directory a tool was pointed at, or a readable refusal."""
    run = args.get("run_dir") or args.get("run_id")
    if not run:
        raise ToolError("give run_dir (a path) or run_id (a directory name)")
    p = Path(str(run)).expanduser()
    if not p.exists():
        # `run_id` used to be the directory's *name*, which only resolved for
        # runs started in the default location: a run launched with an explicit
        # run_dir returned an id that ipro_status then rejected as "no such
        # run". It is now the absolute path, and these fallbacks keep the old
        # short form working.
        for base in (_writable_dir() / "runs", REPO_ROOT / "runs", Path.cwd()):
            cand = base / str(run)
            if cand.exists():
                p = cand
                break
    if not p.exists():
        raise ToolError(
            f"no such run: {run}\n"
            f"Pass the `run_dir` that ipro_run returned (an absolute path). "
            f"Looked in {_writable_dir() / 'runs'} and {Path.cwd()}."
        )
    if not (p / "state.json").exists() and not (p / "ipro.log").exists():
        # `status` on a directory that is not a run reported
        # {"status": "pending"}, so a mistyped path polled forever.
        raise ToolError(
            f"{p} is not a run directory: it has no state.json and no "
            f"ipro.log. Check the path -- a run that never started leaves a "
            f"launch.log, and ipro_run reports that case itself now."
        )
    return p


def tool_status(args: dict) -> dict:
    p = _resolve_run(args)
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


def tool_inspect(args: dict) -> dict:
    """What is in a structure, and what is AT the residues someone named.

    This is the step that makes a request like "you may change residues 18, 20,
    38 and 40" actionable at all. Without it nothing says which chain those
    numbers are in, what the native residues are, or whether they exist -- so an
    agent guesses, and a guess here designs a different residue without
    anything noticing (docs/DESIGN.md section 7b, defect 14).
    """
    cmd = ["inspect", _client_path(args["structure"], "structure", must_exist=True)]
    for ch in args.get("chains") or []:
        cmd += ["--chain", str(ch)]
    if args.get("positions"):
        pos = args["positions"]
        cmd += ["--positions", ",".join(str(x) for x in pos)
                if isinstance(pos, list) else str(pos)]
    return _cli(cmd, timeout=300)


def tool_fetch(args: dict) -> dict:
    """Download a structure from the RCSB PDB by its four-character ID.

    The ONLY networked operation in this package. It is a separate, explicit
    tool for that reason: a design run must never silently reach the internet
    for its inputs, and whoever reads the result should know a coordinate file
    arrived from outside.
    """
    cmd = ["fetch", str(args["pdb_id"])]
    if args.get("out"):
        # The same one policy as `structure`: where the client is standing,
        # not where this package happens to live. A relative `out` used to be
        # resolved by the child against REPO_ROOT, so a download the agent
        # asked for "here" landed in the repository (or in site-packages) and
        # the path it was told to use next did not exist.
        cmd += ["-o", _client_path(args["out"], "out")]
    for ch in args.get("chains") or []:
        cmd += ["--chain", str(ch)]
    if args.get("force"):
        cmd += ["--force"]
    return _cli(cmd, timeout=300)


def tool_setup(args: dict) -> dict:
    """Report what is missing and, on confirmation, install it.

    The gap this closes: the server is pure standard library, so it starts
    happily on a machine with nothing installed -- and then every useful tool
    fails, leaving the agent to work out that `pymol-open-source` is conda-only
    and that there is an `environment.yml` for it. That is exactly the
    wheel-reinventing this interface exists to avoid.

    Nothing is installed without ``confirm: true``. Creating a conda
    environment downloads hundreds of megabytes and changes the machine, so the
    default is a dry run that states the command.
    """
    rep = _doctor()
    missing_required = list(rep.get("missing_required") or [])
    needs_pymol = not rep.get("can_design")

    manager = next(
        (m for m in ("mamba", "conda", "micromamba") if _which(m)), None
    ) or "conda"
    # A pip-installed copy has no environment.yml -- it is not package data --
    # so the printed command named a file that does not exist. Fetch it first
    # in that case, the same way an absent document falls back to raw GitHub.
    env_file = _env_file()
    if env_file is not None:
        command = f"{manager} env create -f {env_file}"
        env_where = str(env_file)
    else:
        tmp = "/tmp/ipro-agent-environment.yml"
        command = (
            f"curl -fsSL {RAW_BASE}environment.yml -o {tmp} && "
            f"{manager} env create -f {tmp}"
        )
        env_where = f"{RAW_BASE}environment.yml (not on disk in this install)"

    if not missing_required and not needs_pymol:
        return {
            "action": "none",
            "can_run": True,
            "can_design": True,
            "resolved": rep.get("resolved", {}),
            "degraded": rep.get("degraded", []),
            "message": (
                "Nothing to install: a design can run. Optional backends still "
                f"missing: {rep.get('degraded') or 'none'} -- each has a "
                "documented fallback, and the mode must be reported with any "
                "design. See docs/INSTALL.md to add them."
            ),
        }

    plan = {
        "action": "install",
        "can_run": rep.get("can_run"),
        "can_design": rep.get("can_design"),
        "missing_required_python_modules": missing_required,
        "pymol_missing": needs_pymol,
        "conda_manager_found": manager,
        "command": command,
        "environment_file": env_where,
        "creates_environment": "ipro-agent",
        "approximate_download": "~500 MB (PyMOL dominates it)",
        "why_conda": (
            "pymol-open-source is not on PyPI, so pip alone cannot supply it, "
            "and rotamers come only from PyMOL's mutagenesis wizard -- there is "
            "no fallback. That is why an environment file exists."
        ),
    }
    if not manager:
        plan["blocked"] = (
            "No conda, mamba or micromamba on PATH, so this cannot install "
            "PyMOL for you. Install Miniforge "
            "(https://github.com/conda-forge/miniforge) and call this again, or "
            "run the command above yourself. `pip install -r requirements.txt` "
            "covers the Python modules but NOT PyMOL, so a design still will "
            "not run."
        )
        return plan
    if not args.get("confirm"):
        plan["next_step"] = (
            "This will change the machine: pass confirm=true to run it. The "
            "install is launched detached and takes several minutes; poll the "
            "log it returns, then call ipro_doctor again. Tell the user what "
            "is about to be installed before you confirm on their behalf."
        )
        return plan

    log = Path(tempfile.mkdtemp(prefix="ipro_setup_")) / "setup.log"
    # Launch the command the plan advertises, not a second one assembled here.
    # The two used to differ on every pip install: `_env_file()` returns None
    # when there is no checkout, so the plan printed the curl-then-create line
    # while this launched `conda env create -f None` -- `str(None)` is the
    # four-character string "None" -- which dies immediately, and the reply
    # still came back launched: true with a pid and a log. That is the same
    # "reported success for a dead child" shape already fixed for ipro_run.
    #
    # shell=True is safe here and nowhere else in this file: `command` is built
    # from RAW_BASE, a fixed temp path, and `manager` chosen from a three-item
    # allowlist. No part of it comes from the caller.
    with open(log, "wb") as fh:
        proc = subprocess.Popen(
            command, shell=True,
            stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            cwd=str(REPO_ROOT), start_new_session=True,
        )
    plan.update({
        "launched": True,
        "pid": proc.pid,
        "log": str(log),
        "when_it_finishes": _mcp_config_for_new_env(manager),
        "next_step": (
            f"Installing in the background; follow {log}. When it finishes this "
            "server must be RESTARTED under the new environment's python, or "
            "PyMOL still will not resolve -- a local server sees only what its "
            "own interpreter can import. Give the user the config in "
            "`when_it_finishes`, then call ipro_doctor to confirm."
        ),
    })
    return plan


def _mcp_config_for_new_env(manager: str) -> dict:
    """The exact MCP config block to use once the environment exists.

    Telling someone to "activate the environment" is useless here: the client
    launches this server, not a shell, so `conda activate` never runs. What they
    need is the interpreter path, and it is predictable -- so predict it rather
    than describing it.
    """
    import shutil

    base = Path(shutil.which(manager) or "/").resolve().parent.parent
    candidates = [base / "envs" / "ipro-agent" / "bin" / "python",
                  Path.home() / "miniforge3" / "envs" / "ipro-agent" / "bin" / "python",
                  Path.home() / "miniconda3" / "envs" / "ipro-agent" / "bin" / "python"]
    predicted = next((c for c in candidates if c.parent.parent.parent.exists()),
                     candidates[0])
    return {
        "note": (
            "Put this in the client's MCP config and restart it. `cwd` matters "
            "for two reasons, both measured: in a pip-installed copy it is how "
            "the server finds the documents it serves as resources (ipro://skill "
            "and the rest), which do not ship inside the package -- started "
            "from elsewhere, an installed copy lists all 8 and serves none; and "
            "a relative path in any later call (`spec`, `structure`) is "
            "resolved against it. Run from a clone rather than an install, "
            "`-m ipro_agent.mcp_server` additionally needs `cwd` to be that "
            "clone, or the module cannot be imported at all and the client "
            "sees only a server that disconnected."
        ),
        "python": str(predicted),
        "config": {
            "mcpServers": {
                "ipro-agent": {
                    "command": str(predicted),
                    "args": ["-m", "ipro_agent.mcp_server"],
                    "cwd": str(REPO_ROOT),
                }
            }
        },
        "verify": "then call ipro_doctor; can_design must be true",
    }


def _which(name: str) -> str | None:
    import shutil

    return shutil.which(name)


def tool_result(args: dict) -> dict:
    """The design itself: what changed, and the coordinates it changed into.

    ``ipro_status`` reports progress; this reports the product. An MCP client
    has no shell, so without it an agent can launch a design, quote its score,
    and never be able to hand over the structure it just made.
    """
    run = _resolve_run(args)
    best = run / "best" / "design.json"
    if not best.exists():
        raise ToolError(
            f"{run} has no best/design.json yet. Poll ipro_status: a run that "
            f"has not finished has not written one, and a run that failed "
            f"reports its error there."
        )
    try:
        design = json.loads(best.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ToolError(f"could not read {best}: {exc}")

    out: dict = {
        "run": str(run),
        "mode": design.get("mode"),
        "goal": design.get("goal"),
        "goal_statement": design.get("goal_statement"),
        "mutations": design.get("mutations", {}),
        "mutation_string": design.get("mutation_string"),
        "sequence": design.get("sequence"),
        "start_score": design.get("start_score"),
        "best_score": design.get("best_score"),
        "components": design.get("components", {}),
        "notes": design.get("notes", []),
        "backend": design.get("backend"),
    }
    if not out["mutations"]:
        out["designed_nothing"] = True
        out["warning"] = (
            "No mutation was kept, so the sequence and the structure here are "
            "the STARTING POINT, not a redesign. Report it as 'the search "
            "found nothing'."
        )

    ens = run / "ensemble" / "ensemble.json"
    if ens.exists():
        try:
            rows = json.loads(ens.read_text(encoding="utf-8"))
            out["ensemble"] = [
                {k: r.get(k) for k in
                 ("iteration", "score", "mutations", "accepted", "proven_optimal")}
                for r in rows[:25]
            ]
            out["ensemble_total"] = len(rows)
        except (OSError, json.JSONDecodeError):
            pass

    structures = sorted((run / "best").glob("*.pdb"))
    out["structure_files"] = [str(s) for s in structures]
    if args.get("include_structure") and structures:
        # ~200 KB for a 340-residue chain, so it is opt-in and capped.
        text = structures[0].read_text(encoding="utf-8", errors="replace")
        limit = 400_000
        out["structure_name"] = structures[0].name
        out["structure_pdb"] = text[:limit]
        if len(text) > limit:
            out["structure_truncated"] = (
                f"{len(text)} characters, truncated to {limit}. The whole file "
                f"is at structure_files[0]."
            )
    return out


def tool_iterations(args: dict) -> dict:
    """The per-iteration record, which the reporting rule asks an agent to read.

    ``ipro_status`` tells the caller to "read the per-iteration reasons", and
    until now nothing could: ``iterations.jsonl`` is on disk, the house rules
    forbid loading it wholesale because it grows without limit, and an MCP
    client has no shell. This returns a bounded tail of it.
    """
    run = _resolve_run(args)
    path = run / "iterations.jsonl"
    if not path.exists():
        raise ToolError(f"{run} has no iterations.jsonl yet")
    try:
        limit = int(args.get("limit", 20))
    except (TypeError, ValueError):
        raise ToolError("`limit` must be a whole number")
    limit = max(1, min(limit, 200))
    only_accepted = bool(args.get("only_accepted"))

    rows: list[dict] = []
    total = 0
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if only_accepted and not rec.get("accepted"):
                continue
            rows.append({
                "iteration": rec.get("iteration"),
                "score": rec.get("score"),
                "accepted": rec.get("accepted"),
                "reason": rec.get("reason"),
                "mutations": rec.get("mutations"),
            })
            if len(rows) > limit:
                rows.pop(0)

    reasons: dict[str, int] = {}
    for r in rows:
        key = str(r.get("reason") or "").split("(")[0].strip() or "unknown"
        reasons[key] = reasons.get(key, 0) + 1
    return {
        "run": str(run),
        "iterations_total": total,
        "returned": len(rows),
        "only_accepted": only_accepted,
        "reason_counts": reasons,
        "iterations": rows,
        "note": (
            "The tail of the trajectory, newest last. One reason repeating for "
            "every iteration is itself the finding: a goal filter rejecting "
            "everything usually means the filter window or the design "
            "positions need changing, not more iterations."
        ),
    }


def tool_runs(_args: dict) -> dict:
    """Every run this server can see, for when an id has been lost."""
    seen: list[dict] = []
    roots = [_writable_dir() / "runs", REPO_ROOT / "runs", Path.cwd() / "runs"]
    for root in roots:
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir() or any(x["run_dir"] == str(d) for x in seen):
                continue
            row = {"run_dir": str(d), "status": "unknown"}
            state = d / "state.json"
            if state.exists():
                try:
                    st = json.loads(state.read_text(encoding="utf-8"))
                    row["status"] = st.get("status")
                    row["mode"] = st.get("mode")
                    row["best_score"] = st.get("best_score")
                    row["updated"] = st.get("updated")
                except (OSError, json.JSONDecodeError):
                    row["status"] = "unreadable"
            elif (d / "launch.log").exists():
                row["status"] = "never started (see launch.log)"
            seen.append(row)
    return {
        "runs": seen[:100],
        "searched": [str(r) for r in roots],
        "note": "Poll one with ipro_status; read its design with ipro_result.",
    }


def tool_cancel(args: dict) -> dict:
    """Stop a run this server started. Nothing else is touched."""
    run = _resolve_run(args)
    try:
        st = json.loads((run / "state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ToolError(f"cannot read the run's state: {exc}")
    pid, host = st.get("pid"), st.get("host")
    if st.get("status") != "running":
        return {
            "run": str(run), "stopped": False, "status": st.get("status"),
            "message": f"not running (status {st.get('status')!r})",
        }
    if not isinstance(pid, int) or pid <= 0:
        raise ToolError(
            "this run recorded no pid, so it cannot be stopped from here. It "
            "is an older run, or one started another way."
        )
    import platform

    if host and host != platform.node():
        raise ToolError(
            f"this run was started on {host!r} and this is "
            f"{platform.node()!r}; a pid means nothing across machines."
        )
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return {
            "run": str(run), "stopped": False, "status": "stale",
            "message": "the process is already gone; ipro_status reports this "
                       "run as stale.",
        }
    except PermissionError as exc:
        raise ToolError(f"not permitted to stop pid {pid}: {exc}")
    return {
        "run": str(run), "stopped": True, "pid": pid,
        "message": (
            "SIGTERM sent. The run stops after the iteration it is in and "
            "keeps what it has written: poll ipro_status, then read "
            "ipro_result for the best design it found."
        ),
    }


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
        "name": "ipro_setup",
        "description": (
            "Check whether this machine can actually run a design and, with "
            "confirm=true, install what is missing. Call it when ipro_doctor "
            "reports can_design=false. PyMOL is conda-only and has no "
            "fallback, so pip cannot fix it. Installs nothing without "
            "confirm=true; tell the user what it will do first."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "confirm": {
                    "type": "boolean",
                    "description": (
                        "false (default) reports the plan; true runs the conda "
                        "install, which downloads ~500 MB and changes the machine"
                    ),
                }
            },
        },
        "handler": tool_setup,
    },
    {
        "name": "ipro_fetch_structure",
        "description": (
            "Download a structure from the RCSB PDB by its four-character ID "
            "(e.g. 2OMF), optionally writing a copy reduced to one chain. This "
            "is the ONLY tool here that uses the network. Use it when the user "
            "names a protein or a PDB entry rather than giving you a file. "
            "Always check what you got with ipro_inspect_structure afterwards: "
            "resolution, missing residues and engineered mutations all change "
            "what a design means."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "pdb_id": {"type": "string", "description": "e.g. 2OMF, 1CRN"},
                "out": {"type": "string", "description": "output path (default <ID>.pdb)"},
                "chains": {
                    "type": "array", "items": {"type": "string"},
                    "description": "also write a copy reduced to these chains; a "
                                   "design spec's residue numbers are per chain",
                },
                "force": {"type": "boolean", "description": "overwrite an existing file"},
            },
            "required": ["pdb_id"],
        },
        "handler": tool_fetch,
    },
    {
        "name": "ipro_inspect_structure",
        "description": (
            "What is in a structure file: chains, residue counts and ranges, the "
            "one-letter sequence, ligands and waters -- and, for residues you "
            "name, the NATIVE identity plus a ready-to-paste spec line. Call "
            "this before writing a spec, every time. A user who says 'residues "
            "18, 20, 38 and 40' has not said which chain, and a residue number "
            "that resolves to the wrong chain silently designs a different "
            "residue. It needs no PyMOL and no network."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "structure": {"type": "string", "description": "path to a PDB or mmCIF file"},
                "chains": {"type": "array", "items": {"type": "string"},
                           "description": "restrict the report to these chains"},
                "positions": {
                    "type": "array", "items": {"type": "string"},
                    "description": "residues to report the native identity of, e.g. "
                                   "[\"18\", \"A20\", \"1:115\"]",
                },
            },
            "required": ["structure"],
        },
        "handler": tool_inspect,
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
    {
        "name": "ipro_result",
        "description": (
            "The design a run produced: the mutation string, the sequence, the "
            "score change, the components, the notes, the ensemble of "
            "alternatives, and -- with include_structure -- the designed PDB "
            "itself. Report from this rather than from ipro_status alone: an "
            "MCP client has no shell, so this is the only way to hand over the "
            "coordinates. If no mutation was kept it says so, and the "
            "structure is then the starting point."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_dir": {"type": "string", "description": "the run directory"},
                "run_id": {"type": "string", "description": "or the id ipro_run returned"},
                "include_structure": {
                    "type": "boolean",
                    "description": "include the designed PDB as text (about "
                                   "200 KB for a 340-residue chain); off by default",
                },
            },
        },
        "handler": tool_result,
    },
    {
        "name": "ipro_iterations",
        "description": (
            "The tail of the trajectory: per-iteration score, whether it was "
            "accepted, and why it was rejected. The reporting rule tells you to "
            "read the per-iteration reasons, and this is how. Never read "
            "iterations.jsonl directly; it grows without limit."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_dir": {"type": "string", "description": "the run directory"},
                "run_id": {"type": "string", "description": "or the id ipro_run returned"},
                "limit": {
                    "type": "integer",
                    "description": "how many of the most recent to return (default 20, max 200)",
                },
                "only_accepted": {
                    "type": "boolean",
                    "description": "only the iterations that were accepted",
                },
            },
        },
        "handler": tool_iterations,
    },
    {
        "name": "ipro_runs",
        "description": (
            "Every run this server can see, with its status. For when a run id "
            "has been lost, or to check what is already running before starting "
            "another."
        ),
        "inputSchema": {"type": "object", "properties": {}},
        "handler": tool_runs,
    },
    {
        "name": "ipro_cancel",
        "description": (
            "Stop a running design. It stops after the iteration it is in and "
            "keeps what it has written, so the best design found so far is "
            "still readable with ipro_result. Only runs this server started, on "
            "this machine."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_dir": {"type": "string", "description": "the run directory"},
                "run_id": {"type": "string", "description": "or the id ipro_run returned"},
            },
        },
        "handler": tool_cancel,
    },
]

_BY_NAME = {t["name"]: t for t in TOOLS}


# ---------------------------------------------------------------------------
# Resources: the documents, so a remote agent is not worse informed
# ---------------------------------------------------------------------------
#
# Eleven tool descriptions cannot carry the operating manual -- the degraded
# modes, the reporting rules, the validation checklist. An agent that clones the
# repo reads SKILL.md and gets all of it; an agent on the other end of this
# server used to get none of it. Exposing the documents as resources closes that
# gap: the manual is a fetch away instead of unavailable.

RESOURCES: list[dict[str, str]] = [
    {
        "uri": "ipro://skill",
        "path": ".claude/skills/ipro-agent/SKILL.md",
        "name": "Operating manual (SKILL.md)",
        "description": (
            "READ THIS FIRST for anything beyond a single tool call. Goal "
            "selection, spec format, encoding a user's constraints into the "
            "integer program, the degraded modes, and the validation checklist "
            "to work through before reporting a design. Everything the tool "
            "schemas cannot say."
        ),
        "mimeType": "text/markdown",
    },
    {
        "uri": "ipro://house-rules",
        "path": "AGENTS.md",
        "name": "House rules",
        "description": (
            "The non-negotiables: report the mode and notes with every design, "
            "never present screening energies as published kcal/mol, never "
            "quietly weaken a user's stated constraints, and when to stop and "
            "ask a human."
        ),
        "mimeType": "text/markdown",
    },
    {
        "uri": "ipro://responsible-use",
        "path": "README.md",
        "name": "README, including Responsible use",
        "description": (
            "What the tool is, what it has actually been run on, its known open "
            "defect, and the scope-of-use limits that need a human decision "
            "before anything is run."
        ),
        "mimeType": "text/markdown",
    },
    {
        "uri": "ipro://design-notes",
        "path": "docs/DESIGN.md",
        "name": "Provenance and deviations (DESIGN.md)",
        "description": (
            "Every capability traced to the paper and equation it came from, "
            "every deviation from the published method, and every defect found "
            "by audit with the measurement that exposed it. Use it to answer "
            "'where does this number come from'."
        ),
        "mimeType": "text/markdown",
    },
    {
        "uri": "ipro://milp",
        "path": "docs/MILP.md",
        "name": "The integer program",
        "description": "The rotamer-selection formulation in full, with its constraints.",
        "mimeType": "text/markdown",
    },
    {
        "uri": "ipro://install",
        "path": "docs/INSTALL.md",
        "name": "Dependencies and what each unlocks",
        "description": (
            "What PyMOL, PyRosetta, AutoDock Vina and a MILP solver each buy, "
            "and what their absence costs. Pair with ipro_setup."
        ),
        "mimeType": "text/markdown",
    },
    {
        "uri": "ipro://example-spec",
        "path": "examples/ompf_pore/spec_smoke.yaml",
        "name": "Worked example spec (annotated)",
        "description": (
            "A complete, heavily commented spec to copy from: molecules, "
            "positions, allowed residues, restraints, goal options. The fastest "
            "way to write a correct spec for a new protein."
        ),
        "mimeType": "text/yaml",
    },
    {
        "uri": "ipro://example-expectations",
        "path": "examples/ompf_pore/EXPECTED.md",
        "name": "What the worked example can and cannot reproduce",
        "description": (
            "An honest account of which comparisons against the published paper "
            "are meaningful, and which are not available at all."
        ),
        "mimeType": "text/markdown",
    },
]

_BY_URI = {r["uri"]: r for r in RESOURCES}


#: Where the documents are looked for, in order. ``REPO_ROOT`` is the repo when
#: the server runs from a clone, but **site-packages** when it was pip-installed
#: -- and the documents are not inside the package, so a pip-installed copy
#: found none of them. Measured: `resources/list` returned 0 entries, which
#: silently removed the operating manual that the server's own instructions tell
#: the client to read. The client's working directory is tried as well, so a
#: pip-installed server whose MCP config sets `cwd` to a checkout still works.
def _env_file() -> "Path | None":
    """``environment.yml``, if this copy has one on disk.

    A clone has it; a wheel does not (it is not package data, and adding a
    second copy inside the package would be a file that can drift from the
    real one). Callers must handle None -- see tool_setup.
    """
    for root in _doc_roots():
        cand = root / "environment.yml"
        if cand.is_file():
            return cand
    return None


def _doc_roots() -> list[Path]:
    """Where to look for the documents this server serves as resources.

    ``IPRO_REPO`` comes first because not every MCP client can set a working
    directory for a server it launches. Claude Desktop, Claude Code and Cursor
    take a ``cwd``; the Codex and VS Code configurations are command plus args
    plus environment. Without a way to point at a checkout, a pip-installed
    server serves none of the documents, and `ipro://skill` -- the operating
    manual -- is the one an agent most needs.
    """
    roots: list[Path] = []
    env = os.environ.get("IPRO_REPO")
    if env:
        candidate = Path(env).expanduser()
        if candidate.is_dir():
            roots.append(candidate)
    roots.append(REPO_ROOT)
    try:
        cwd = Path.cwd()
    except OSError:                       # a deleted cwd is survivable
        return roots
    if cwd not in roots:
        roots.append(cwd)
    return roots


#: Canonical location of each document, for a copy that has neither.
#: Where a server with no checkout tells the agent to fetch a document, and
#: where `tool_setup` tells a pip-installed copy to curl `environment.yml`.
#:
#: Pinned to this build's own tag rather than to `main`. It tracked `main`,
#: so an installed 0.1.0 was handed whatever `main` said today -- a manual
#: describing behaviour its code does not have, and, at `tool_setup`, an
#: `SERVER_INFO['version']` is bumped in the same commit the release is tagged
#: from (docs/RELEASING.md) and a test pins it to pyproject, so the tag exists
#: for every published version; the one window where it does not is between a
#: maintainer's bump and their tag, on an unreleased tree.
RAW_BASE = (
    "https://raw.githubusercontent.com/chowdhurylab-studio/ipro-agent/"
    f"v{SERVER_INFO['version']}/"
)


def _resolve_doc(rel: str) -> Path | None:
    for root in _doc_roots():
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


def _resource_list() -> list[dict]:
    """Every document, always -- listed whether or not this copy has the file.

    Listing only what exists hid the manual from a pip-installed server without
    saying so. An agent is better served by a name it can ask for and a clear
    answer about where to get it than by a silently shorter list.
    """
    out = []
    for r in RESOURCES:
        entry = {k: r[k] for k in ("uri", "name", "description", "mimeType")}
        if _resolve_doc(r["path"]) is None:
            entry["description"] += (
                " [NOT in this installation: it ships with the repository, not "
                "the package. Reading it returns the URL to fetch, or run the "
                "server from a clone.]"
            )
        out.append(entry)
    return out


def _resource_read(uri: str) -> dict:
    spec = _BY_URI.get(uri)
    if spec is None:
        raise ToolError(f"unknown resource {uri!r}; have {sorted(_BY_URI)}")
    path = _resolve_doc(spec["path"])
    if path is None:
        raise ToolError(
            f"{uri} is the file {spec['path']}, which ships with the repository "
            f"rather than with the installed package, and this server cannot see "
            f"a checkout (it looked in {', '.join(str(r) for r in _doc_roots())}).\n"
            f"Either fetch it from {RAW_BASE}{spec['path']}\n"
            f"or run the server from a clone, or set `cwd` to a checkout in the "
            f"MCP config -- `git clone "
            f"https://github.com/chowdhurylab-studio/ipro-agent`."
        )
    return {
        "contents": [{
            "uri": uri,
            "mimeType": spec["mimeType"],
            "text": path.read_text(encoding="utf-8", errors="replace"),
        }]
    }


# ---------------------------------------------------------------------------
# Prompts: the client shows these as slash commands
# ---------------------------------------------------------------------------

PROMPTS: list[dict[str, Any]] = [
    {
        "name": "ipro-redesign",
        "description": (
            "Redesign a protein toward a stated goal, start to finish, the way "
            "the operating manual says to."
        ),
        "arguments": [
            {"name": "structure",
             "description": "path to a PDB/mmCIF file, OR a four-character PDB "
                            "ID to fetch (e.g. 2OMF)",
             "required": True},
            {"name": "goal", "description": "what to achieve, in plain words",
             "required": True},
            {"name": "positions",
             "description": "residues that may change, e.g. 'A110, A113, A117' "
                            "(omit to have them proposed from the structure)",
             "required": False},
        ],
    },
    {
        "name": "ipro-check-setup",
        "description": "Check whether this machine can run a design, and fix it if not.",
        "arguments": [],
    },
    {
        "name": "ipro-report",
        "description": (
            "Turn a finished run into an honest report, with the mode, the "
            "notes and the caveats the house rules require."
        ),
        "arguments": [
            {"name": "run", "description": "the run directory or run id",
             "required": True},
        ],
    },
]

_PROMPT_NAMES = {p["name"] for p in PROMPTS}
_PROMPT_BY_NAME = {p["name"]: p for p in PROMPTS}


def _prompt_get(name: str, args: dict) -> dict:
    prompt = _PROMPT_BY_NAME.get(name)
    if prompt is None:
        raise ToolError(f"unknown prompt {name!r}; have {sorted(_PROMPT_NAMES)}")

    # `required: true` was decorative, the same way the tool schemas were before
    # section 7h. Measured by driving the server from the official MCP Python
    # SDK and from the MCP Inspector CLI: `prompts/get ipro-redesign` with no
    # arguments at all returned a complete 2167-character instruction, whose
    # step 3 told the model to fetch "(structure)" from the RCSB and whose step
    # 6 told it to put "(goal)" in `goal_statement` "in my own words". A client
    # that surfaces prompts as slash commands without collecting their
    # arguments -- which the spec allows, because the server is supposed to
    # refuse -- therefore starts a design against a placeholder, and the first
    # thing this repo's own manual insists on is that a position or a goal must
    # never be invented on the user's behalf.
    missing = [
        a["name"] for a in (prompt.get("arguments") or [])
        if a.get("required") and not _given(args.get(a["name"]))
    ]
    if missing:
        detail = "; ".join(
            f"{a['name']}: {a.get('description', '')}".strip(": ")
            for a in (prompt.get("arguments") or []) if a["name"] in missing
        )
        raise ToolError(
            f"prompt {name!r} needs "
            f"{', '.join(repr(m) for m in missing)}, which "
            f"{'was' if len(missing) == 1 else 'were'} not given. {detail}"
        )

    if name == "ipro-check-setup":
        text = (
            "Check whether this machine can actually run a protein design with "
            "ipro-agent.\n\n"
            "1. Call `ipro_doctor`. Report `can_run`, `can_design` and the "
            "degraded backend list in plain words.\n"
            "2. If `can_design` is false, call `ipro_setup` WITHOUT confirm "
            "first, show me exactly what it would install and how large the "
            "download is, and ask before confirming.\n"
            "3. Tell me what the degraded backends change about the numbers I "
            "will get -- read `ipro://install` if you need the detail.\n"
        )
    elif name == "ipro-report":
        run = args.get("run", "(the run)")
        text = (
            f"Report the results of ipro-agent run `{run}` honestly.\n\n"
            "1. Call `ipro_status`.\n"
            "2. Read `ipro://house-rules` and follow it exactly.\n"
            "3. If `accepted` is 0, say the search found nothing and show what "
            "rejected the candidates -- do NOT present the best_* fields as a "
            "design, because they are the starting point.\n"
            "4. Otherwise give the mutation string, the score change, the "
            "`mode`, every entry in `notes`, and where the coordinates are.\n"
            "5. State plainly that energies from the built-in backend are a "
            "screening scale, not kcal/mol comparable to the published papers, "
            "and that the design is a hypothesis for the bench.\n"
        )
    else:
        structure = args.get("structure", "(structure)")
        goal = args.get("goal", "(goal)")
        positions = args.get("positions")
        where = (
            f"The residues I am willing to change are: {positions}."
            if positions else
            "I have not chosen the positions -- propose them with "
            "`ipro_positions` and show me the list before designing."
        )
        text = (
            f"Redesign the protein in `{structure}`. My goal: {goal}\n"
            f"{where}\n\n"
            "Work in this order, and do not skip steps:\n\n"
            "1. Read `ipro://skill` -- it is the operating manual and it covers "
            "goal selection, the spec format and the reporting rules that the "
            "tool descriptions cannot.\n"
            "2. `ipro_doctor`. If a design is impossible, stop and use "
            "`ipro_setup` rather than guessing.\n"
            "3. Get the structure. If what I gave you is a four-character PDB "
            "ID rather than a file, `ipro_fetch_structure` it (and reduce it to "
            "the chain we mean -- a spec's residue numbers are per chain).\n"
            "4. **`ipro_inspect_structure`, with the residues I named.** Tell me "
            "what is at each one before you design anything: a bare residue "
            "number does not say which chain it is in, and one resolved against "
            "the wrong chain silently designs a different residue. If a position "
            "I named does not exist, or is not the residue I seem to expect, "
            "stop and ask.\n"
            "5. `ipro_goals`, then pick the objective that matches my goal, and "
            "tell me which one and why. If the goal needs an anti-target or a "
            "transition-state analogue I have not given you, ask.\n"
            "6. Write the spec, copying the shape from `ipro://example-spec`. "
            "Put my goal in `goal_statement` **in my own words**. Any geometry "
            "that must survive -- a catalytic contact, a metal site, a "
            "disulfide -- has to be a restraint, because the optimizer will not "
            "preserve it on its own.\n"
            "7. `ipro_validate`. Fix every problem it reports before spending "
            "CPU. Show me the design-space size.\n"
            "8. `ipro_score` for the starting baseline.\n"
            "9. `ipro_run` with a `max_minutes` budget, then poll "
            "`ipro_status`. It is detached; do not block.\n"
            "10. Report as `ipro://house-rules` requires: the mode, the notes, "
            "and the fact that every design is a bench hypothesis. If nothing "
            "was accepted, say so.\n\n"
            "If a step fails and the only way past it is to weaken something I "
            "specified -- a restraint, the allowed residues, an energy "
            "threshold -- tell me instead of doing it.\n\n"
            "If the goal is about a pore or channel radius, read the axis "
            "caveat: the radius is only meaningful when the design positions "
            "surround the lumen, and the run reports the axis clearance so you "
            "can tell. Say so rather than quoting a number you cannot stand "
            "behind."
        )
    return {
        "description": f"ipro-agent: {name}",
        "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
    }


def _tool_list() -> list[dict]:
    return [{k: t[k] for k in ("name", "description", "inputSchema")} for t in TOOLS]


# ---------------------------------------------------------------------------
# JSON-RPC plumbing
# ---------------------------------------------------------------------------


def handle(message: dict) -> dict | None:
    """Handle one JSON-RPC message; None means "a notification, send nothing".

    A message with no ``id`` member is a **notification**, and JSON-RPC 2.0 says
    a server must not respond to one. The unknown-method branch at the bottom
    used to reply anyway, with ``"id": null`` -- so a client that sent any
    notification this server did not recognise got back an error frame it had
    not asked for, which a strict client can treat as a protocol desync. The
    known notifications were handled; it was the unknown ones that leaked.
    """
    if not isinstance(message, dict):
        return _err(None, -32600, "invalid request: expected a JSON object")
    method = message.get("method")
    is_notification = "id" not in message
    mid = message.get("id")
    params = message.get("params") or {}
    if not isinstance(params, dict):
        return None if is_notification else _err(
            mid, -32602, f"params must be an object, got {type(params).__name__}"
        )

    if method == "initialize":
        want = params.get("protocolVersion")
        version = want if want in _PROTOCOL_VERSIONS else _PROTOCOL_VERSIONS[0]
        return _ok(mid, {
            "protocolVersion": version,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": SERVER_INFO,
            "instructions": (
                "Protein redesign via IPRO.\n\n"
                "Read the resource `ipro://skill` before anything beyond a "
                "single tool call -- it is the operating manual, and it carries "
                "the degraded modes, the spec format and the validation "
                "checklist that these tool descriptions cannot. "
                "`ipro://house-rules` is the short version.\n\n"
                "Call ipro_doctor first and report the mode it gives with any "
                "result; if it says can_design is false, use ipro_setup rather "
                "than guessing at the install. Validate a spec before running "
                "one. ipro_run is detached: poll ipro_status, never read "
                "iterations.jsonl. A run with accepted: 0 designed nothing -- "
                "its best_* fields are the starting point. Energies from the "
                "built-in backend are a screening scale, not kcal/mol "
                "comparable to the published papers. Never quietly weaken a "
                "constraint the user stated to make a failing run succeed. "
                "Every design is a hypothesis for the bench."
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
        if not isinstance(name, str):
            # An unhashable name (a dict, a list) used to reach _BY_NAME.get
            # and raise `TypeError: unhashable type`, which killed the session.
            return _err(mid, -32602,
                        f"tools/call `name` must be a string, got "
                        f"{type(name).__name__}")
        tool = _BY_NAME.get(name)
        if tool is None:
            return _err(mid, -32602, f"unknown tool {name!r}; have {sorted(_BY_NAME)}")
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return _err(mid, -32602,
                        f"`arguments` must be an object, got "
                        f"{type(arguments).__name__}")
        try:
            problem = _check_args(tool, arguments)
            if problem:
                return _ok(mid, _content({"error": problem}, is_error=True))
            result = tool["handler"](arguments)
            return _ok(mid, _content(result))
        except ToolError as exc:
            # A tool-level failure, reported to the model rather than to the
            # transport, so it can read the message and adjust.
            return _ok(mid, _content({"error": str(exc)}, is_error=True))
        except Exception as exc:                              # pragma: no cover
            return _ok(mid, _content(
                {"error": f"{type(exc).__name__}: {exc}"}, is_error=True))

    if method == "resources/list":
        return _ok(mid, {"resources": _resource_list()})

    if method == "resources/read":
        uri = params.get("uri", "")
        if not isinstance(uri, str):
            return _err(mid, -32602,
                        f"`uri` must be a string, got {type(uri).__name__}")
        try:
            return _ok(mid, _resource_read(uri))
        except ToolError as exc:
            return _err(mid, -32602, str(exc))

    if method == "prompts/list":
        return _ok(mid, {"prompts": PROMPTS})

    if method == "prompts/get":
        pname = params.get("name", "")
        pargs = params.get("arguments") or {}
        if not isinstance(pname, str) or not isinstance(pargs, dict):
            return _err(mid, -32602,
                        "prompts/get needs `name` (a string) and optional "
                        "`arguments` (an object)")
        try:
            return _ok(mid, _prompt_get(pname, pargs))
        except ToolError as exc:
            return _err(mid, -32602, str(exc))

    if method == "resources/templates/list":
        return _ok(mid, {"resourceTemplates": []})

    if is_notification:
        # Unknown, and a notification: silence is the correct response.
        return None
    return _err(mid, -32601, f"method not found: {method}")


#: JSON type names to the Python types a json.loads result can hold.
_JSON_TYPES: dict[str, tuple] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def _check_args(tool: dict, arguments: dict) -> str:
    """Check arguments against the tool's own declared schema.

    The schemas were decorative: a missing ``spec`` reached the handler and
    came back as ``KeyError: 'spec'``, and ``spec: 42`` as
    ``AttributeError: 'int' object has no attribute 'strip'``. A model cannot
    act on either, and both look like a bug in the tool rather than in the
    call. Unknown extra arguments are still ignored, which is correct -- a
    client may send more than the schema names.
    """
    schema = tool.get("inputSchema") or {}
    props = schema.get("properties") or {}
    missing = [
        k for k in (schema.get("required") or [])
        if arguments.get(k) is None
    ]
    if missing:
        names = ", ".join(f"`{k}`" for k in missing)
        detail = "; ".join(
            f"{k}: {(props.get(k) or {}).get('description', '')}".strip(": ")
            for k in missing
        )
        return (
            f"{tool['name']} needs {names}, which "
            f"{'was' if len(missing) == 1 else 'were'} not given. {detail}"
        )
    for key, value in arguments.items():
        want = (props.get(key) or {}).get("type")
        allowed = _JSON_TYPES.get(want)
        if not allowed or value is None:
            continue
        # JSON has no integer type of its own: 8.0 is a whole number.
        if want == "integer" and isinstance(value, float) and value == int(value):
            continue
        if isinstance(value, bool) and want != "boolean":
            return (
                f"{tool['name']}: `{key}` must be {want}, got a boolean"
            )
        if not isinstance(value, allowed):
            return (
                f"{tool['name']}: `{key}` must be {want}, got "
                f"{type(value).__name__} ({json.dumps(value, default=str)[:60]})"
            )
    return ""


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
        except (json.JSONDecodeError, RecursionError):
            # RecursionError belongs here with JSONDecodeError: json.loads
            # recurses per nesting level, so a deeply nested array raises it
            # rather than a decode error, and it escaped this clause and ended
            # the process -- breaking the invariant §7h states outright, one
            # error reply each and never the session. Measured against the
            # real server over stdio: 2,000 levels replied cleanly and the
            # next tools/list was answered; 100,000 levels gave one reply,
            # exit 1, and the queued tools/list was never answered. The
            # threshold is high enough that this is hostile or buggy input
            # rather than normal traffic, which is why it was the last one
            # found, not why it should stay.
            stdout.write(json.dumps(_err(None, -32700, "parse error")) + "\n")
            stdout.flush()
            continue
        # A top-level ARRAY is a JSON-RPC batch, which MCP allowed before the
        # 2025-06-18 revision and which some clients still send. It used to
        # reach handle() as a list and kill the whole server with
        # `AttributeError: 'list' object has no attribute 'get'` -- one message
        # taking down the session.
        try:
            if isinstance(message, list):
                replies = [r for r in (handle(m) for m in message) if r is not None]
                reply = replies or None     # an all-notification batch: silence
            else:
                reply = handle(message)
        except Exception as exc:           # pragma: no cover - belt and braces
            # No single malformed message may end the session. A request gets an
            # error frame; a notification gets silence, as the spec requires.
            mid = message.get("id") if isinstance(message, dict) else None
            reply = None if mid is None else _err(
                mid, -32603, f"internal error: {type(exc).__name__}: {exc}"
            )
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
