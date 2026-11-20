## [1] BLOCKER The MCP install the README documents cannot run a design: ipro_doctor, ipro_setup and ipro_run all fail on a missing scripts/doctor.py inside site-packages

README's MCP section says `pip install git+https://github.com/chowdhurylab-studio/ipro-agent` then `{"mcpServers": {"ipro-agent": {"command": "ipro-agent-mcp"}}}` - no cwd, no module path. pyproject's [tool.setuptools] packages list is ipro_agent, ipro_agent.energy, ipro_agent.pymol_helpers, so scripts/ and environment.yml are not in the wheel, while REPO_ROOT = Path(__file__).resolve().parent.parent resolves to site-packages. _doctor() therefore runs `<venv>/bin/python <site-packages>/scripts/doctor.py --json`, which exits 2 with an empty stdout, and json.loads raises -> ToolError. ipro_doctor is the first tool the server's own initialize `instructions` tell the client to call; ipro_setup and ipro_run both call _doctor() before anything else, so neither can ever succeed in this install shape. ipro_goals/validate/positions/score/milp/status do work (they go through _cli, which only needs the importable package), so the server looks healthy right up to the point where a design is wanted. Commit 8ff7b54 (landed during this audit) fixed the sibling problem for resources/* but did not touch doctor, setup or run; ipro_setup additionally prints a `mamba env create -f <site-packages>/environment.yml` command for a file that is not there.

**evidence**
$ python -m venv venv-install && ./venv-install/bin/python -m pip install --no-deps /Users/supantha/Documents/test123/ipro-agent
$ ./venv-install/bin/python -m pip install numpy biopython pyyaml      # deps present, so this is not a dependency problem
$ printf '<initialize>\n<ipro_setup>\n<ipro_run spec=examples/ompf_pore/spec_smoke.yaml>\n' | ./venv-install/bin/ipro-agent-mcp
1 initialize ok
--- id 4 isError= True
{"error": "doctor.py returned no JSON: .../venv-install/bin/python: can't open file '.../venv-install/lib/python3.14/site-packages/scripts/doctor.py': [Errno 2] No such file or directory\n"}
--- id 5 isError= True
{"error": "doctor.py returned no JSON: ... site-packages/scripts/doctor.py': [Errno 2] No such file or directory\n"}
ipro_doctor gives the identical error; ipro_goals in the same install returns normally (isError=False, 1011 chars).
$ grep -A3 'tool.setuptools' pyproject.toml
packages = ["ipro_agent", "ipro_agent.energy", "ipro_agent.pymol_helpers"]

**fix**
Move the doctor into the package - ipro_agent/doctor.py with a main() - and have _doctor() call it in-process or as `[sys.executable, "-m", "ipro_agent.doctor", "--json"]`; keep scripts/doctor.py as a thin shim for the documented CLI. Ship environment.yml as package data (or, like the new resource fallback, hand back its raw GitHub URL) so ipro_setup's command refers to a file that exists. Then add the pip-installed shape to the test matrix, the same way 8ff7b54 did for resources.

---

## [2] SHOULD-FIX ipro_run reports launched:true for a run that was already dead, and ipro_status then says "pending" forever

tool_run passes iterations/seed/max_minutes through as str(args[key]) with no validation, and never checks that the child survived. The inputSchema says "iterations": {"type": "integer"}, but nothing enforces it, so a JSON float - which a model will emit for an integer field - becomes `--iterations 8.0`, argparse exits 2 within milliseconds, and the MCP reply is still isError:false, launched:true, with a pid that is already gone. The real error sits in launch.log, which the agent has no reason to read because it was told the launch succeeded; ipro_status on that directory answers "status": "pending", so a polling agent waits forever on a run that never started. The same hole covers any instant death of the child (bad spec content, unreadable structure), because nothing between Popen and the reply looks at the process.

**evidence**
$ python -I -c "...ipro_run with iterations=8.0 (a JSON number)"
iterations as a JSON float 8.0  isError=False launched=True pid=22374
  rd_float/launch.log last line: ipro-agent run: error: argument --iterations: invalid int value: '8.0'

Same with a string, and the poll that follows:
  ipro_run(iterations="lots") -> {"run_id":"never","pid":6912,"launched":true}
  never/launch.log            -> ipro-agent run: error: argument --iterations: invalid int value: 'lots'
  ipro_status(run_dir=never)  -> {'status': 'pending', 'iterations_done': 0, 'accepted': 0, 'mode': None}

And the child is not even reaped: ps -o pid,stat 21607 -> "21607 Z <defunct>" while the server is still serving.

**fix**
Coerce the numeric arguments before building the command (int(args["iterations"]) etc., raising ToolError naming the argument on failure), then after Popen give the child a short grace period and proc.poll(); if it has already exited non-zero, return isError with the tail of launch.log instead of launched:true. Reap finished children (signal.signal(SIGCHLD, SIG_IGN) or keep the Popen objects and poll them).

---

## [3] NICE-TO-HAVE The server replies to notifications with an id:null error response, violating JSON-RPC 2.0

handle() detects notifications by method name - only notifications/initialized, initialized and notifications/cancelled return None. Every other notification falls through to the final `return _err(mid, -32601, ...)` with mid None, so the server writes {"jsonrpc":"2.0","id":null,"error":{"code":-32601,...}} for messages it must not answer at all. JSON-RPC 2.0: "The Server MUST NOT reply to a Notification." The notifications that trigger it are ones real clients send: notifications/roots/list_changed (any client that declares the roots capability sends it when the workspace changes), notifications/progress, notifications/message. The response carries an id no client ever sent.

**evidence**
$ python -I -c "...INIT + notification('notifications/roots/list_changed')"
exit=0 replies=2
  {"jsonrpc": "2.0", "id": 1, "result": {...}}
  {"jsonrpc": "2.0", "id": null, "error": {"code": -32601, "message": "method not found: notifications/roots/list_changed"}}
Identical unsolicited reply for notifications/progress, notifications/message and any unknown notification method.

Suspected part, stated as such: how bad this is depends on the client. Driving the server with the official MCP python SDK 2.3.0 in a throwaway venv, the session SURVIVED the stray response - after sending a real RootsListChangedNotification, send_ping() and list_tools() both still worked ("ping after notification -> ok", "list_tools after notification -> 11") and the SDK logged nothing. I could not test Claude Desktop, Cursor or the TypeScript SDK here, and some clients log a protocol error or treat an unmatched response as fatal.

**fix**
Detect notifications structurally, not by name: at the top of handle(), after reading mid, `if mid is None and str(method or "").startswith("notifications/"): return None` - or more simply, return None for any message with no "id" key, which is exactly what JSON-RPC defines a notification to be.

---

## [4] NICE-TO-HAVE Only tools/call is exception-guarded, so four other malformed-parameter shapes kill the whole server

tools/call wraps its handler in `except Exception`, but nothing else does. handle()'s own parameter handling is unguarded, and resources/read and prompts/get catch only ToolError. Measured kills: (a) "params" as a JSON array, which JSON-RPC 2.0 explicitly allows, on any method - `params.get(...)` raises AttributeError; (b) tools/call with "name" a list or object - _BY_NAME.get(unhashable) raises TypeError, before the try block that would have caught it; (c) resources/read with "uri" a list - same TypeError inside _resource_read; (d) prompts/get with "name" a list, or "arguments" a list - TypeError on the set lookup / AttributeError on args.get. Each one exits the process, so the client loses the session rather than one request.

**evidence**
$ python -I -c "..."  (each case: INIT then the bad message, against commit 8ff7b54)
  params as array on initialize   exit=1 replies=0  AttributeError: 'list' object has no attribute 'get'   (handle, `want = params.get("protocolVersion")`)
  params as array on tools/call   exit=1 replies=1  AttributeError: 'list' object has no attribute 'get'   (handle, `name = params.get("name")`)
  tools/call name unhashable      exit=1 replies=1  TypeError: cannot use 'dict' as a dict key (unhashable type: 'dict')   (`tool = _BY_NAME.get(name)`)
  resources/read uri unhashable   exit=1 replies=1  TypeError: cannot use 'list' as a dict key (unhashable type: 'list')   (_resource_read, `spec = _BY_URI.get(uri)`)
  prompts/get name unhashable     exit=1 replies=1  TypeError: cannot use 'list' as a set element (unhashable type: 'list')  (_prompt_get, `if name not in _PROMPT_NAMES`)
  prompts/get arguments is a list exit=1 replies=1  AttributeError: 'list' object has no attribute 'get'   (_prompt_get, `args.get("run", ...)`)

**fix**
Two lines of belt and braces: in handle(), `if params is not None and not isinstance(params, dict): return _err(mid, -32602, "params must be an object")`; and in serve(), wrap `reply = handle(message)` in try/except Exception -> `_err(mid, -32603, f"{type(exc).__name__}: {exc}")`. One bad request should cost one error reply, never the session.

---

## [5] SHOULD-FIX A CLI traceback reaches the agent as an 800-character tail that starts mid-line, with the actual error last

_cli raises ToolError(err[-800:]) when stdout is empty. That is the right instinct for the CLI's one-line refusals, but when the CLI raises instead of refusing, the agent gets the last 800 bytes of a Python traceback - sliced at an arbitrary byte, so the string opens with caret art or half a word, and the one useful sentence is at the very end where a model may truncate or mis-weight it. Both cases I hit were realistic: an inline spec missing goal_statement (an agent writing a spec from context will do this), and a fresh install with a dependency missing (the very situation ipro_doctor/ipro_setup exist to explain in words).

**evidence**
Inline spec whose only problem is a missing goal_statement; the full error string the agent receives:
'       ~~~~~~~~~^^^^^^\n  File ".../ipro_agent/cli.py", line 63, in cmd_validate\n    spec = _load(args.spec)\n ... \nTypeError: DesignSpec.__init__() missing 1 required positional argument: \'goal_statement\''

ipro_goals in a fresh venv without numpy:
{"error": "ile \"/.../site-packages/ipro_agent/cli.py\", line 32, in <module>\n    from .goals import GOALS, build_goal, suggest_goal\n ..."}
   (the message literally begins "ile \"" - a sliced `File "`)

**fix**
Pick the last line matching ^[A-Za-z_.]*(Error|Exception|Warning): from stderr as the message, and append the preceding 400 bytes as context labelled "traceback tail". Keep the existing single-line-refusal path exactly as it is.

---

## [6] NICE-TO-HAVE One slow tool call blocks every other message for up to 900 s, and notifications/cancelled is accepted but cannot interrupt anything

serve() is a single synchronous loop: a request is read, the handler runs to completion, then the next line is read. A ping sent behind ipro_score waits for the whole score. DEFAULT_TIMEOUT is 900 s and the module's own comment says score and milp "can take a couple of minutes on a large design set", so a client's keepalive ping, its tools/list, and anything else it sends can sit unanswered for up to fifteen minutes - long past the point where most clients declare the server unresponsive. notifications/cancelled returns None (correctly silent) but is otherwise discarded, so the request it names keeps running and a user pressing stop cannot stop the subprocess.

**evidence**
$ python -I t5_serial.py    # initialize, ipro_score(spec_smoke), notifications/cancelled{requestId:2}, ping - all written at once
  t=  0.03s  id=1  result   (initialize)
  t=  4.73s  id=2  result   (ipro_score)
  t=  4.73s  id=3  result   (ping)
=== arrival times: {'1': 0.03, '2': 4.73, '3': 4.73}
    ping (id 3) waited behind ipro_score: 4.73s
The cancellation produced no reply (correct) and had no effect (the score ran to completion).
Measured on the 4-position smoke spec; the 900 s ceiling and the "couple of minutes" estimate are from DEFAULT_TIMEOUT and the module docstring, which I did not spend a large design set to reproduce.

**fix**
Run each tools/call handler in a worker thread and keep reading stdin, so ping, tools/list and notifications/cancelled are answered while a tool runs; keep a dict of request id -> Popen so notifications/cancelled can terminate the right subprocess. If that is more machinery than wanted, at least say in the tool descriptions that score and milp block the server, and lower DEFAULT_TIMEOUT to something inside a typical client's patience.

---

## [7] SHOULD-FIX Missing and wrong-typed arguments reach the model as raw Python exception text; the declared required/typed schema is never enforced

inputSchema declares required: ["spec"] and types for every argument, but tools/call passes params["arguments"] straight to the handler, so the schema is documentation only. The broad except turns the resulting Python exception into the error the model reads. It is recoverable - a model can usually guess what KeyError: 'spec' means - but it is the server's one chance to say which argument was wrong, and it spends it on a traceback type name. Confirmed through the real MCP SDK, so this is what a client actually shows the model.

**evidence**
Via the official MCP python SDK 2.3.0:
  call_tool("ipro_validate", {}) -> is_error= True '{\n "error": "KeyError: \'spec\'"\n}'
Over raw stdio, the rest of the matrix (every one isError:true, server survives):
  no "arguments" key / arguments:{} / arguments:null / other keys only -> "KeyError: 'spec'"
  spec: 42 / ["a"] / {"a":1} / null -> "AttributeError: 'int'|'list'|'dict'|'NoneType' object has no attribute 'strip'"
  arguments: ["x"] -> "TypeError: list indices must be integers or slices, not str"
  arguments: "spec" -> "TypeError: string indices must be integers, not 'str'"
  unknown extra argument -> ignored, call succeeds (correct)
  cutoff: "five" -> the CLI's own argparse message, which is clear and useful

**fix**
A dozen lines in tools/call: for each name in inputSchema["required"] that is absent, and each present argument whose JSON type does not match its declared type, return isError with one sentence naming the argument, the expected type and what arrived.

---

## [8] SHOULD-FIX doctor.py reports can_run/can_design True on a Python below the declared floor, and exits 0

doctor.py is the gate the repo tells agents to trust first (AGENTS.md:18 "Before anything else, run `python3 scripts/doctor.py --require pymol`"). But can_run is computed only from the three importable modules (scripts/doctor.py:287-291: required_py = ["py:numpy", "py:Bio", "py:yaml"]; report["can_run"] = not report["missing_required"]), and can_design is that AND pymol. Nothing ever compares sys.version_info with the requires-python = ">=3.10" floor, even though doctor already prints the version in its own header. Measured on Python 3.9.6 with numpy/Bio/yaml present: doctor says python 3.9.6 | can_run True | can_design True | missing_required [] and exits 0, and `--require pymol` also exits 0. On that same interpreter `import ipro_agent` raises RuntimeError from the new floor guard in ipro_agent/__init__.py, so `python -m ipro_agent <anything>`, scripts/selftest.py and `python -m ipro_agent.mcp_server` (rc=1) all refuse to start. The documented first gate therefore says the machine is fully green on an interpreter where no command can run -- and because the guard landed after doctor was written, the two now contradict each other. A stranger whose `python3` is the macOS system 3.9 (very common) or a distro 3.9 reads 'can run: True, can design: True' and then gets a RuntimeError from every subsequent step. doctor is stdlib-only on purpose, so the check needs no imports and no new dependency.

**evidence**
$ /private/tmp/claude-501/-Users-supantha-Documents-test123/a4581fba-d9ac-4b77-93bd-2ae3a83c7cd7/scratchpad/edge-pyversion-deps/venv39/bin/python scripts/doctor.py --json | python -c "..."   # repo at 820212c, venv39 = 3.9.6 + numpy 2.0.2 + Bio 1.85 + yaml 6.0.3
python 3.9.6 | can_run True | can_design True | missing_required []

$ .../venv39/bin/python scripts/doctor.py --require pymol >/dev/null 2>&1; echo rc=$?
rc=0 (0 = gate passed)

$ .../venv39/bin/python scripts/doctor.py --require pymol | tail -3
can run:    True
can design: True

$ .../venv39/bin/python -c "import ipro_agent"
RuntimeError: ipro-agent needs Python 3.10 or newer; this is 3.9.6 at .../venv39/bin/python.

$ .../venv39/bin/python scripts/selftest.py | tail -2   # same commit, same interpreter
RuntimeError: ipro-agent needs Python 3.10 or newer; this is 3.9.6 ...

$ grep -nE "version_info|REQUIRES_PYTHON|requires-python" scripts/doctor.py   # live tree, 2d9813a
NO version-floor check in scripts/doctor.py

$ grep -n "requires-python" pyproject.toml
11:requires-python = ">=3.10"

**fix**
In scripts/doctor.py's report add the interpreter itself as a required check, e.g. report["python_ok"] = sys.version_info >= (3, 10) (hardcode the tuple so doctor stays stdlib-only and still works when the package will not import), append "python<3.10" to missing_required when it fails so can_run and can_design both go False and the exit status becomes 1, and print the floor in the human output next to the version it already shows: `ipro-agent doctor (python 3.9.6 -- BELOW the required 3.10)`. Three lines, no new machinery.

---

## [9] SHOULD-FIX The documented pip-install MCP config is broken: an installed copy has no doctor, no environment.yml and no resources

REPO_ROOT = Path(__file__).resolve().parent.parent (mcp_server.py:47). In a clone that is the repo; in a pip-installed copy it is site-packages, and [tool.setuptools] packages ships only the three Python packages -- no scripts/, no docs/, no environment.yml, no .claude/skills/ipro-agent/SKILL.md, no examples/. So README.md:89-95 (`pip install git+...` plus `{"mcpServers": {"ipro-agent": {"command": "ipro-agent-mcp"}}}`) yields a server where ipro_doctor and ipro_setup both error, resources/list is empty, and ipro://skill is unreadable -- while the server's own `instructions` field tells the client to read ipro://skill before anything beyond a single call and to call ipro_doctor first. ipro_run is dead too, because tool_run calls _doctor() before launching. Only ipro_goals, ipro_validate and ipro_positions work. AGENTS.md's mandatory first step (`python3 scripts/doctor.py --require pymol`) is impossible from an installed copy: there is no scripts/ and no doctor console script. docs/INSTALL.md section 6 lists 'Installed' as one of 'Two supported ways to use it'.

**evidence**
$ venv-installed/bin/python -m pip wheel --no-deps .
$ python -c "import zipfile,glob; n=zipfile.ZipFile(glob.glob('*.whl')[0]).namelist(); print([x for x in n if 'scripts/' in x or 'docs/' in x or x.endswith('.yml') or 'SKILL' in x])"
[]

$ cd /private/tmp/.../neutral-cwd && python -c "import ipro_agent,pathlib; p=pathlib.Path(ipro_agent.__file__).resolve().parent.parent; [print(f'{r:45s}',(p/r).exists()) for r in ('scripts/doctor.py','environment.yml','.claude/skills/ipro-agent/SKILL.md','docs/DESIGN.md','examples/ompf_pore/spec_smoke.yaml')]"
scripts/doctor.py                             False
environment.yml                               False
.claude/skills/ipro-agent/SKILL.md            False
docs/DESIGN.md                                False
examples/ompf_pore/spec_smoke.yaml            False

$ drive.py venv-installed/bin/ipro-agent-mcp   # real stdio, cwd = neutral dir
--- id=2 isError=True   (ipro_doctor)
{"error": "doctor.py returned no JSON: .../venv-installed/bin/python: can't open file
 '.../site-packages/scripts/doctor.py': [Errno 2] No such file or directory\n"}
--- id=3 isError=True   (ipro_setup)  same error
--- id=4  resources/list -> {"resources": []}
--- id=5  resources/read ipro://skill ->
  {"code": -32602, "message": "ipro://skill maps to .claude/skills/ipro-agent/SKILL.md, which is not in this checkout"}
--- id=6 isError=False  (ipro_goals works)

$ ls venv-installed/bin/ | grep -i doctor   ->  (nothing)

**fix**
Move the doctor into the package (ipro_agent/doctor.py, re-exported by scripts/doctor.py for the clone) and add an `ipro-agent-doctor` console script. Ship environment.yml and the eight resource documents as package data ([tool.setuptools.package-data] plus importlib.resources), and make the resource/doctor lookup try the package-data location before REPO_ROOT. At minimum, detect the installed layout and return a ToolError that says 'this is a pip-installed copy; the doctor and the documents are not shipped -- use a clone' rather than a FileNotFoundError.

---

## [10] SHOULD-FIX doctor.py --require pymol exits 0 on Python 3.9, so the mandatory gate green-lights an interpreter that cannot import the package

check() probes numpy/Bio/yaml and the backends but never compares sys.version_info to ipro_agent.REQUIRES_PYTHON. It prints the version in the header and ignores it. AGENTS.md says 'Before anything else, run python3 scripts/doctor.py --require pymol ... --require pymol makes that a non-zero exit instead of something to discover mid-run'. On a Mac whose `python3` is the Xcode 3.9.6 (still the default on macOS 13/14), the user runs doctor, is told 'REQUIRED and missing: numpy, Bio, yaml' with the remedy 'or: pip install -r requirements.txt', does exactly that -- requirements.txt carries no python_requires so pip does not object -- and doctor then reports can_run True, can_design True, exit 0. The tool itself still refuses to import. The one environmental fact that definitely blocks the tool is the one fact the designated gate does not check.

**evidence**
$ /usr/bin/python3 -m venv venv39 && venv39/bin/python -m pip install "numpy>=1.24" "biopython>=1.81" "pyyaml>=6.0"
$ venv39/bin/python -c "import numpy,Bio,yaml,sys;print(sys.version.split()[0],numpy.__version__,Bio.__version__,yaml.__version__)"
3.9.6 2.0.2 1.85 6.0.3

$ venv39/bin/python scripts/doctor.py
ipro-agent doctor  (python 3.9.6)
  OK  PyMOL (mutagenesis)  [subprocess-env] found in ipro-agent
  ...
can run:    True
can design: True
>>> doctor exit=0

$ venv39/bin/python scripts/doctor.py --require pymol >/dev/null 2>&1; echo $?
0                      <-- AGENTS.md's gate is green

$ venv39/bin/python -m ipro_agent goals --json
RuntimeError: ipro-agent needs Python 3.10 or newer; this is 3.9.6 at .../venv39/bin/python.

**fix**
In doctor.check(), add `report['python_ok'] = sys.version_info >= (3, 10)`, append 'python<3.10' to missing_required when it is false, and fold it into can_run so --require and the exit status both fail. Print the floor and the remedy in the text report. Add `python_requires` as a comment is not enough -- also consider marking requirements.txt with a note that it does not enforce the floor.

---

## [11] SHOULD-FIX A PyMOL launcher is accepted as the harvest interpreter; doctor says OK and the run then hangs for an hour with an empty log

find_pymol_python()'s probe is `works(py)` = `[py, '-c', 'import pymol']` with `return p.returncode == 0`. The PyMOL launcher exits 0 even when it cannot do what was asked, so it passes. Its last resort, `exe = shutil.which('pymol'); if exe: return exe`, is not probed at all. harvest() then builds `[py, '-I', HARVEST_SCRIPT, jobs.json, out.json]`, where PyMOL's `-I` means *enable* the internal GUI, not Python's isolated mode: PyMOL opens an OpenGL window, reports 'unsupported file type: json', and sits in its interactive prompt until rotamers.harvest's timeout=3600 (mutate.apply: timeout=1800). Reached whenever the ipro-agent interpreter cannot import pymol and none of the four searched conda roots holds it -- Linux `apt install pymol` plus a venv, Homebrew miniconda on macOS (doctor searches /opt/homebrew/Caskroom/miniconda/base/envs, rotamers does not), Miniforge, mambaforge, micromamba -- or whenever a user reads 'set IPRO_PYMOL_PYTHON to an interpreter that can import pymol' and points it at their `pymol`. DESIGN.md 7g says ipro_run 'refuses rather than launching a doomed job'; here doctor is green, so it launches.

**evidence**
$ python -c "import subprocess,os; p=subprocess.run(['/opt/miniconda3/envs/ipro-agent/bin/pymol','-c','import pymol'],capture_output=True,timeout=60,env={**os.environ,'PYTHONNOUSERSITE':'1'}); print('rc',p.returncode); print(p.stdout[-120:])"
rc 0                     <-- works() accepts it
b" Error: unsupported file type: import pymol\n Error: Argument processing aborted due to exception (above).\n"

$ IPRO_PYMOL_PYTHON=/opt/miniconda3/envs/ipro-agent/bin/pymol \
  /opt/miniconda3/envs/ipro_rosetta/bin/python scripts/doctor.py --require pymol
doctor --require pymol  rc = 0
  OK  PyMOL (mutagenesis)  [env] IPRO_PYMOL_PYTHON=.../bin/pymol  Error: Argument processing aborted due to exception (above).
  mutate   -> pymol
can run:    True
can design: True

$ IPRO_PYMOL_PYTHON=.../bin/pymol python -m ipro_agent run spec_smoke.yaml \
    --run-dir run-launcher-pin --iterations 1 --max-minutes 2
launched pid 21609 (--max-minutes 2)
after 100s  poll() = None  (None = still running)
  PID ELAPSED COMMAND
21609   01:40 python -m ipro_agent run ... --max-minutes 2
21673   01:38 .../pymol/__init__.py -I .../harvest_rotamers.py .../rotamer_jobs.json .../rotamers.json
---- run log tail ----      (empty)

$ python -m ipro_agent status run-launcher-pin --json
{"status": "pending", "mode": null, "iterations_done": 0, "error": null}

(direct check of the same command, bounded at 180 s:)
  stdout: ' Error: unsupported file type: json\n Error: Argument processing aborted due to exception (above).\n'
  TIMED OUT after 180s;  out.json exists: False

**fix**
Make works() prove capability rather than exit status: run `[py, '-I', '-c', "import pymol; print('IPRO_PYMOL_OK')"]` and require the sentinel on stdout. Apply the same probe to the `shutil.which('pymol')` result instead of returning it unchecked, and if the candidate is a launcher rather than an interpreter either invoke it the way doctor does (`-cq -r script`) or refuse it with a message naming IPRO_PYMOL_PYTHON. Also cut rotamers.harvest's 3600 s / mutate.apply's 1800 s timeouts to something a user will not mistake for progress, and have the loop write a 'waiting on PyMOL' line before blocking.

---

## [12] SHOULD-FIX ipro_score, ipro_milp and ipro_run with no run_dir write into the installed package tree, and fail with a chopped traceback when it is read-only

_cli() runs every CLI call with cwd=REPO_ROOT (line 139), and cli.py defaults the run directory to the relative paths 'runs/_score' (line 293) and 'runs/{spec.name}' (line 338); tool_run's own default is REPO_ROOT/'runs'/_run_id(spec) (line 278). For the documented pip install that is site-packages: every ipro_score call drops a run directory into site-packages/runs/_score, a relative ipro_milp `out` lands in site-packages, and a run with no run_dir writes its structures and its unbounded iterations.jsonl there -- where pip uninstall will not remove it and nobody will look. ipro_score has no run_dir in its schema at all, so a client cannot redirect it. On a read-only install (root-owned site-packages, a shared conda env, a container running as non-root) all three fail, and the message the agent receives is err[-800:] of a traceback, so it begins mid-line. ipro_milp also reports its output as the bare relative name it was given, which the agent cannot resolve.

**evidence**
$ cd ro-cwd  (read-only client cwd) && drive.py ... -m ipro_agent.mcp_server
   ipro_score  -> isError=False, "score": 1.5283052080513857
   ipro_milp out="my.lp" -> isError=False, {"lp_file": "my.lp", "lines": 6173, ...}
$ find clone/runs clone/my.lp -maxdepth 2
clone/runs
clone/my.lp
clone/runs/_score
clone/runs/_score/spec.json
clone/runs/_score/ipro.log
...
$ ls -A ro-cwd        ->  (empty: nothing landed where the client was)

$ chmod -R a-w clone && drive.py ... -m ipro_agent.mcp_server
--- ipro_score  isError=True
{"error": " exist_ok=True)\n    ~~~~~~~~~~^^^^^^^^\n  File \"/opt/miniconda3/.../pathlib/__init__.py\", line 1015, in mkdir\n ... PermissionError: [Errno 13] Permission denied: '.../clone/runs'"}
--- ipro_milp   isError=True
{"error": " ... out.write_text(lp) ... PermissionError: [Errno 13] Permission denied: 'my.lp'"}
--- ipro_run (no run_dir) isError=True
{"error": "PermissionError: [Errno 13] Permission denied: '.../clone/runs'"}

**fix**
Default every write to a user-writable location, not the package: use a temp directory or an IPRO_RUN_ROOT / XDG_STATE_HOME base for tool_run's default run_dir, add run_dir to ipro_score's schema and pass --run-dir explicitly, and resolve ipro_milp's `out` to an absolute path and report that absolute path in lp_file. Catch OSError in _cli/tool_run and raise a ToolError naming the directory and asking for a writable run_dir.

---

## [13] SHOULD-FIX A relative spec path is checked in the server's cwd but used in REPO_ROOT, so an existing file is reported as not found

The MCP server's cwd belongs to the client. _resolve_spec (line 108) does Path(spec).expanduser() and tests p.exists() in that cwd, then returns the string unchanged; _cli then runs the CLI with cwd=REPO_ROOT. A relative path that exists where the client is passes the server's own check and is then resolved somewhere else. The agent is told the file does not exist at a path where it does exist. The same mismatch applies to a relative run_dir: tool_run's emptiness guard and mkdir run in the server's cwd while the child writes under REPO_ROOT, so the guard checks one directory, the run fills another, and the run_dir handed back to the agent stays empty.

**evidence**
$ ls clientcwd/     ->  2omf_A.pdb  myspec.yaml
$ cd clientcwd && python -c "import ipro_agent.mcp_server as m; from pathlib import Path;
  print('exists:', Path('myspec.yaml').expanduser().exists());
  print('REPO_ROOT:', m.REPO_ROOT);
  print('resolve ->', m._resolve_spec('myspec.yaml', []))"
exists: True
REPO_ROOT: .../venv-installed/lib/python3.12/site-packages
resolve -> myspec.yaml                     <-- returned relative

$ cd clientcwd && drive.py venv-installed/bin/ipro-agent-mcp
--- ipro_validate {"spec": "myspec.yaml"}   isError=True
{"error": "spec not found: myspec.yaml"}
--- ipro_validate {"spec": "./myspec.yaml"} isError=True
{"error": "spec not found: myspec.yaml"}

(source of that sentence, confirmed:)
$ cd site-packages && python -m ipro_agent validate myspec.yaml --json
spec not found: myspec.yaml   (exit 2)

**fix**
Resolve in _resolve_spec: `p = Path(spec).expanduser().resolve()` and return str(p). Do the same for args['run_dir'] in tool_run, args['out'] in tool_milp and args['run_dir']/args['run_id'] in tool_status, so the path the server checks is the path the child uses.

---

## [14] SHOULD-FIX ipro_status cannot resolve the run_id that ipro_run just returned, whenever a run_dir was given

tool_run returns run_id = run_dir.name (line 303) for any run_dir, but tool_status (line 326) only looks for a bare path in the server's cwd and then REPO_ROOT/'runs'/<id>. A run anywhere else is unreachable by id, although ipro_status's schema advertises run_id as 'or just the id ipro_run returned' and the whole launch/poll split depends on it. The agent has to notice that the id it was handed is useless and fall back to the full path.

**evidence**
$ ipro_run {"spec": ".../spec smoke.yaml", "run_dir": ".../my specs/ünïcode dir/run out ünï", "iterations": 1}
{"run_id": "run out ünï",
 "run_dir": ".../my specs/ünïcode dir/run out ünï", "launched": true}

$ ipro_status {"run_id": "run out ünï"}            isError=True
{"error": "no such run: run out ünï"}

$ ipro_status {"run_dir": ".../run out ünï"}      isError=False
{"status": "done", "mode": "fixed-backbone", "iterations_done": 1, "accepted": 1, ...}

**fix**
Either return the absolute run_dir as the run_id, or keep a small registry (run_id -> absolute path) that tool_run writes and tool_status consults, or state in the schema that run_id works only for runs launched with the default run_dir.

---

## [15] SHOULD-FIX Doctor and ipro_setup blame PyMOL when the real cause is a missing pip dependency, and claim pip will not be enough when it would be

The text report appends '<- no PyMOL, see above' whenever can_design is false, and tool_doctor's note (mcp_server.py:214) and tool_setup's needs_pymol (line 367) both derive from can_design rather than from checks.pymol.available. can_design is `can_run and pymol.available`, so a machine that has PyMOL in another conda env but is missing numpy/Bio/yaml is told its problem is PyMOL. Doctor then contradicts itself on one screen: 'OK PyMOL (mutagenesis) [subprocess-env] found in ipro-agent' at the top and 'can design: False <- no PyMOL, see above' at the bottom. Over MCP it is worse than confusing: ipro_setup reports pymol_missing: true and states that 'pip install -r requirements.txt covers the Python modules but NOT PyMOL, so a design still will not run' -- which is false here, since PyMOL resolved and pip is exactly what is needed. A stranger follows that and goes off to install a 500 MB conda environment they already have.

**evidence**
$ env -i HOME=$HOME PATH=/usr/bin:/bin /opt/miniconda3/envs/bioinfo/bin/python scripts/doctor.py
line  5:   OK  PyMOL (mutagenesis)  [subprocess-env] found in ipro-agent
line 18: REQUIRED and missing: numpy, Bio, yaml
line 25: can run:    False
line 26: can design: False   <- no PyMOL, see above

$ ... scripts/doctor.py --json | jq-equivalent
checks.pymol.available : True   how: subprocess-env
can_run                : False
can_design             : False
missing_required       : ['numpy', 'Bio', 'yaml']

$ ipro_setup (same env, no conda on PATH)
{"pymol_missing": true,
 "blocked": "No conda, mamba or micromamba on PATH, so this cannot install PyMOL for you. ...
             `pip install -r requirements.txt` covers the Python modules but NOT PyMOL,
             so a design still will not run."}
$ ipro_doctor note: "can_design is False without PyMOL: rotamers come only from its
  mutagenesis wizard and there is no fallback, so `run` cannot produce a design..."

**fix**
Drive the PyMOL language off rep['checks']['pymol']['available']: set needs_pymol from it in tool_setup, make tool_doctor's note conditional (and say 'required Python modules are missing: ...' when that is the cause), and print the '<- no PyMOL' suffix and the PyMOL paragraph only when PyMOL really did not resolve.

---

## [16] SHOULD-FIX The conda search roots miss every forge-family layout, including Miniforge -- the installer ipro_setup itself recommends

rotamers.find_pymol_python searches CONDA_PREFIX/../envs, ~/miniconda3/envs, ~/anaconda3/envs, /opt/miniconda3/envs, /opt/anaconda3/envs. doctor.find_conda_envs searches those plus /opt/homebrew/Caskroom/miniconda/base/envs and the two /opt base interpreters. Neither searches ~/miniforge3/envs (Miniforge's installer default prefix), ~/mambaforge/envs, or ~/micromamba/envs -- yet ipro_setup's blocked message tells the user to 'Install Miniforge (https://github.com/conda-forge/miniforge)' and ipro_setup accepts micromamba as a manager. A user who follows that advice, installs Miniforge, creates the environment and then starts the server from any other interpreter gets can_design False again and is told to install it once more. The doctor/rotamers divergence on the Homebrew root is worse: on a Mac with the miniconda cask, doctor reports PyMOL OK via subprocess-env and can_design True, so ipro_run launches, and then find_pymol_python -- which does not search that root -- falls through to shutil.which('pymol') and the hour-long hang above.

**evidence**
$ python - <<'EOF'   # roots enumerated from the two source files
rotamers.find_pymol_python:
    Path(os.environ.get("CONDA_PREFIX", "/nonexistent")).parent / "envs"
    Path.home() / "miniconda3" / "envs"
    Path.home() / "anaconda3" / "envs"
    Path("/opt/miniconda3/envs")
    Path("/opt/anaconda3/envs")
doctor.find_conda_envs:
    ... same four, plus
    Path("/opt/homebrew/Caskroom/miniconda/base/envs")

  searched by rotamers? False  doctor? False  ~/miniforge3/envs    Miniforge (what ipro_setup tells the user to install)
  searched by rotamers? False  doctor? False  ~/mambaforge/envs    Mambaforge
  searched by rotamers? False  doctor? False  ~/micromamba/envs    micromamba (ipro_setup accepts it as a manager)
  searched by rotamers? True   doctor? True   ~/miniconda3/envs    Miniconda
  searched by rotamers? False  doctor? True   /opt/homebrew/Caskroom/miniconda/base/envs   Homebrew cask
EOF

**fix**
Put the root list in one place that both importers use, and add ~/miniforge3/envs, ~/mambaforge/envs, ~/miniforge/envs, ~/micromamba/envs, /usr/share/miniconda/envs and the Windows %USERPROFILE% equivalents, plus MAMBA_ROOT_PREFIX/envs and CONDA_ENVS_PATH/CONDA_ENVS_DIRS when set. Better still, resolve the env by asking the manager (`conda env list --json` / `micromamba env list --json`) when one is on PATH.

---

## [17] SHOULD-FIX Windows: start_new_session is silently ignored, bin/python never matches, and the served documents come back as mojibake

pyproject declares only MacOS and POSIX::Linux classifiers, which is honest, but classifiers do not stop pip, and nothing in README.md or docs/INSTALL.md says Windows is unsupported -- so a Windows user installs successfully and then meets four quiet failures rather than one clear refusal. (1) tool_run and tool_setup pass start_new_session=True, which CPython's Windows _execute_child takes as `unused_start_new_session`: the flag is ignored, so the child is not detached and the run dies with the MCP client, contradicting DESIGN.md 7g ('The child gets its own session, so it outlives both the server and the client that asked for it'). (2) Both env-discovery loops look for `env / 'bin' / 'python'`, which never matches Windows' `Scripts\python.exe`, and all the hard-coded roots are /opt paths. (3) _resource_read and _resolve_spec use the locale encoding, which is cp1252 on Windows for 3.10-3.14. (4) doctor already looks for 'vina.exe', so the code half expects Windows, which makes the silence about it more likely to mislead.

**evidence**
$ python -c "import subprocess,inspect; s=inspect.getsource(subprocess); i=s.find('unused_start_new_session'); print(s[i-260:i+90])"
        def _execute_child(self, args, executable, preexec_fn, close_fds,
                           ...
                           unused_restore_signals,
                           unused_gid, unused_gids, unused_uid,
                           unused_umask,
                           unused_start_new_session, unused_process_group):
            """Execute program (MS Windows version)"""

$ grep -n 'bin" / "python\|"bin" / "python"' ipro_agent/rotamers.py scripts/doctor.py
ipro_agent/rotamers.py:93:        py = env / "bin" / "python"
scripts/doctor.py:77:            py = env / "bin" / "python"

$ python -c "[print(n, repr(c), 'cp1252 ok' if c.encode('cp1252',errors='strict') else '') for n,c in ...]"
  greek mu      'μ'  UnicodeEncodeError under cp1252
  >=            '≥'  UnicodeEncodeError under cp1252
  arrow         '→'  UnicodeEncodeError under cp1252

**fix**
Either support Windows or say it is unsupported. To support it: branch the detach (creationflags=DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP on nt, start_new_session on posix), look for both `bin/python` and `Scripts/python.exe`, add the %USERPROFILE% conda roots, and pass encoding='utf-8' everywhere. To decline it: say so in README.md and docs/INSTALL.md and refuse early in mcp_server with one sentence naming WSL, rather than failing four different ways.

---

## [18] NICE-TO-HAVE Served documents and inline specs use the locale encoding, so the operating manual can arrive corrupted and an inline spec with a non-ASCII character crashes

_resource_read does path.read_text(errors='replace') (line 739) and _resolve_spec does tmp.write_text(spec) (line 120) -- both with no encoding, so both follow locale.getpreferredencoding(). SKILL.md has 56 lines with non-ASCII and DESIGN.md 202, so under a non-UTF-8 locale ipro://skill is served with the characters replaced -- silently, because errors='replace' turns the failure into corruption. On the write side there is no errors= at all, so an inline spec containing an Angstrom sign, a mu or a >= raises UnicodeEncodeError, which is not a ToolError and reaches the agent as a bare exception name. A protein-design goal_statement is exactly the string most likely to contain those characters. On Linux and macOS PEP 538 coercion makes even LC_ALL=C give UTF-8 (measured: utf8mode 1), so the day-to-day exposure here is Windows cp1252 and any deliberately non-UTF-8 locale; the fix is one keyword argument either way.

**evidence**
$ env LC_ALL=C LANG=C PYTHONCOERCECLOCALE=0 PYTHONUTF8=0 python -c "import locale,sys;print(locale.getpreferredencoding(False), sys.flags.utf8_mode)"
US-ASCII 0

$ env LC_ALL=C PYTHONCOERCECLOCALE=0 PYTHONUTF8=0 python -c \
  "import ipro_agent.mcp_server as m; g=m._resource_read('ipro://skill')['contents'][0]['text']; \
   t=open('.../SKILL.md',encoding='utf-8').read(); \
   print('U+FFFD in served text:', g.count('�')); print(len(g),'vs',len(t),'identical:',g==t)"
U+FFFD in served text: 210
29672 vs 29540  identical: False       (DESIGN 7g records the true 29 540)

$ (same locale) ipro_validate with spec = 'goal: stabilize\ngoal_statement: narrow the pore below 4 Å (±5%)\nmolecules: []\n'
--- isError=True
{"error": "UnicodeEncodeError: 'ascii' codec can't encode character '\\xc5' in position 56: ordinal not in range(128)"}

(clean under the default locale: preferred encoding utf-8, stdin utf-8)

**fix**
path.read_text(encoding='utf-8', errors='replace') and tmp.write_text(spec, encoding='utf-8') -- and the same for every other read_text/write_text on a repo document or a spec. Server stdin was measured robust (3 of 3 replies, exit 0, with raw UTF-8 request bytes under an ASCII locale), so only these two need touching.

---

## [19] SHOULD-FIX A one-line inline spec is treated as a path, and the error does not say why

_looks_inline keys only on the presence of a newline (line 103). A compact single-line spec -- which is what json.dumps produces, and the tool's own schema says 'a spec YAML/JSON file, OR the spec's YAML content inline' -- is therefore classified as a path, and the agent is told 'spec not found: {"goal": "stabilize", ...}' with the whole document echoed back as if it were a filename. The docstring's reasoning (a path is one line and has no colon-space) is sound but the check does not implement the colon-space half, and the error does not mention the multi-line requirement.

**evidence**
$ python -c "import ipro_agent.mcp_server as m; ..."
  compact JSON, one line   _looks_inline=False   -> ToolError: spec not found: {"goal": "stabilize", "molecules": []}
  one-line YAML            _looks_inline=False   -> ToolError: spec not found: goal: stabilize
  Windows path             _looks_inline=False   -> ToolError: spec not found: C:\Users\me\spec.yaml
  multi-line YAML          _looks_inline=True    -> /var/folders/.../T/ipro_mcp_spec__3cnakh

**fix**
Treat a value that starts with '{' or contains ': ' as inline as well as one containing a newline, and in the not-found message say 'if this was meant as inline content, inline specs must be multi-line YAML or a JSON object' plus the first 80 characters rather than the whole document.

---

## [20] SHOULD-FIX `validate` dies with a bare Python traceback on ~11 classes of malformed-but-plausible spec

`DesignSpec.load`/`from_dict` let yaml.YAMLError, json.JSONDecodeError, TypeError and ValueError escape, and cli.py `_load` (line 36) only catches SpecError/FileNotFoundError/IsADirectoryError. So the one command whose job is to catch a bad spec before CPU is spent prints a 20-line traceback for inputs an LLM produces constantly. Confirmed, each a full traceback: (1) tab indentation -> yaml.scanner.ScannerError; (2) a .json file holding YAML (load branches on suffix at line 1023 although the docstring says "chosen by content, not suffix") -> json.JSONDecodeError; (3) `molecules:` as a mapping instead of a list -> TypeError ... must be a mapping, not str; (4) `molecules: [<path string>]` -> same; (5) `loop:` as a list -> TypeError; (6) `goal:` missing -> TypeError: DesignSpec.__init__() missing 2 required positional arguments; (7) `goal_statement:` missing -> same, 1 argument; (8) `goal_options: null` -> TypeError: 'NoneType' object is not iterable; (9) `goal_options:` as a list -> ValueError: dictionary update sequence element #0 has length 1; (10) `role: desgin` -> ValueError: 'desgin' is not a valid Role (no list of valid roles); (11) `kind: dna` -> ValueError: 'dna' is not a valid MolKind. Note the contrast: a spec that is valid YAML but not a mapping, a bare string, an empty file and a comments-only file are all handled perfectly (exit 2, "does not contain a mapping at the top level"), so the polish is there -- these paths just are not covered. Over MCP it is worse: ipro_validate returns the last 800 characters of the traceback (mcp_server.py:155), which starts mid-word ("niconda3/envs/...") and cuts off the actual diagnosis line.

**evidence**
S=/private/tmp/claude-501/-Users-supantha-Documents-test123/a4581fba-d9ac-4b77-93bd-2ae3a83c7cd7/scratchpad/edge-agent-inputs
PY=/opt/miniconda3/envs/ipro-agent/bin/python   # 3.14.10, the documented env

$ printf 'goal: reshape_channel\n\tname: bad\n' > $S/specs/a_tab.yaml
$ $PY -m ipro_agent validate $S/specs/a_tab.yaml --json ; echo exit=$?
Traceback (most recent call last):
  File "<frozen runpy>", line 203, in _run_module_as_main
  ... 18 more frames ...
  File ".../yaml/scanner.py", line 258, in fetch_more_tokens
yaml.scanner.ScannerError: while scanning for the next token
found character '\t' that cannot start any token
exit=1

$ for n in k_nogoal k_nostatement k_go_null k_go_list k_badrole k_badkind sh_mol_mapping sh_mol_str sh_loop_list; do
>   $PY -m ipro_agent validate $S/specs/$n.yaml --json >$S/out/v_$n 2>&1
>   printf '%-15s exit=%s traceback=%s :: %s\n' $n $? "$(grep -c '^Traceback' $S/out/v_$n)" "$(tail -1 $S/out/v_$n)"; done
k_nogoal        exit=1 traceback=1 :: TypeError: DesignSpec.__init__() missing 2 required positional arguments: 'goal_statement' and 'goal'
k_nostatement   exit=1 traceback=1 :: TypeError: DesignSpec.__init__() missing 1 required positional argument: 'goal_statement'
k_go_null       exit=1 traceback=1 :: TypeError: 'NoneType' object is not iterable
k_go_list       exit=1 traceback=1 :: ValueError: dictionary update sequence element #0 has length 1; 2 is required
k_badrole       exit=1 traceback=1 :: ValueError: 'desgin' is not a valid Role
k_badkind       exit=1 traceback=1 :: ValueError: 'dna' is not a valid MolKind
sh_mol_mapping  exit=1 traceback=1 :: TypeError: ipro_agent.config.Molecule() argument after ** must be a mapping, not str
sh_mol_str      exit=1 traceback=1 :: TypeError: ipro_agent.config.Molecule() argument after ** must be a mapping, not str
sh_loop_list    exit=1 traceback=1 :: TypeError: ipro_agent.config.LoopSettings() argument after ** must be a mapping, not list

# and over MCP (real stdio session, tools/call ipro_validate on a_tab.yaml):
    isError=True  text:
    {"error": "niconda3/envs/ipro-agent/lib/python3.14/site-packages/yaml/parser.py\", line 428, in parse_block_mapping_key\n ... yaml.scanner.ScannerError: while scanning"}

**fix**
Wrap the parse and the dataclass construction: in `DesignSpec.load`, catch `yaml.YAMLError` and `json.JSONDecodeError` and re-raise as `SpecError(f"{p}: line N: ...")`; in `from_dict`, build each dataclass inside a try that turns TypeError/ValueError into `SpecError` naming the block and the key (`molecules[0]`, `loop`, `goal_options`), and check for the required `goal`/`goal_statement` keys explicitly before `cls(**d)`. Route `.json` through `yaml.safe_load` too (YAML is a JSON superset), which also makes the docstring's "chosen by content, not suffix" true. In mcp_server.`_cli`, prefer the LAST line of stderr (the exception message) over the last 800 characters of the traceback.

---

## [21] BLOCKER In a pip-installed copy -- README's first-recommended MCP setup -- ipro_doctor, ipro_setup and ipro_run are all dead

`REPO_ROOT = Path(__file__).resolve().parent.parent` and `_doctor()` shells out to `REPO_ROOT/scripts/doctor.py`, but pyproject's `[tool.setuptools] packages = [...]` ships only the three Python packages: `scripts/doctor.py` and `environment.yml` are not in the wheel (verified: `find build -name doctor.py -o -name environment.yml` returns nothing). README lines 99-104 tell a stranger to `pip install git+https://github.com/chowdhurylab-studio/ipro-agent` and configure `{"mcpServers": {"ipro-agent": {"command": "ipro-agent-mcp"}}}` -- exactly this layout. Measured in a throwaway venv: initialize and tools/list work, ipro_validate / ipro_goals / prompts work, and ipro_doctor, ipro_setup and ipro_run all fail with "doctor.py returned no JSON: ... can't open file '<site-packages>/scripts/doctor.py'". Since `tool_run` calls `_doctor()` first, **no design can ever be launched from an installed copy**, and `ipro_setup` -- the tool whose whole purpose is to repair a broken install -- is broken the same way. The agent-facing message says nothing an agent can act on. Contrast `_resource_read`, which handles the same situation beautifully ("ipro://skill is the file .claude/skills/ipro-agent/SKILL.md, which ships with the repository rather than with the installed package, and this server cannot see a checkout"). Secondary: even if doctor worked, `_run_id` (line 316) defaults run output to `REPO_ROOT/runs/...`, i.e. inside site-packages.

**evidence**
S=/private/tmp/claude-501/.../scratchpad/edge-agent-inputs
$ /opt/miniconda3/envs/ipro-agent/bin/python -m venv --system-site-packages $S/venv_install
$ $S/venv_install/bin/python -m pip install --no-deps --quiet /Users/supantha/Documents/test123/ipro-agent
$ ls $S/venv_install/bin | grep -i ipro
ipro-agent
ipro-agent-mcp
$ ls $S/venv_install/lib/python3.14/site-packages/scripts
ls: .../site-packages/scripts: No such file or directory

# driving the installed server exactly as README says, cwd = a non-repo directory:
initialize: True
--- tools/list count
    11 tools
--- ipro_doctor
    isError= True
    {"error": "doctor.py returned no JSON: .../venv_install/bin/python: can't open file
       '.../site-packages/scripts/doctor.py': [Errno 2] No such file or directory"}
--- ipro_validate (should work)
    isError= False   {"ok": true, ...}
--- ipro_setup
    isError= True    {"error": "doctor.py returned no JSON: ... scripts/doctor.py ..."}
--- ipro_run
    isError= True    {"error": "doctor.py returned no JSON: ... scripts/doctor.py ..."}
--- resources/read ipro://skill
    rpc error: "ipro://skill is the file .claude/skills/ipro-agent/SKILL.md, which ships with
    the repository rather than with the installed package, and this server cannot see a checkout"
$ find /Users/supantha/Documents/test123/ipro-agent/build -name 'doctor.py' -o -name 'environment.yml'
(no output)

**fix**
Move the doctor into the package as `ipro_agent/doctor.py` (leave `scripts/doctor.py` a two-line wrapper so the documented `python3 scripts/doctor.py` keeps working) and have `_doctor()` call it in-process or as `[sys.executable, "-m", "ipro_agent.doctor", "--json"]`. Ship `environment.yml` as package data for `ipro_setup`. Give `_doctor()` a fallback message in the style of `_resource_read`'s, and default the run directory to the caller's own directory (or require `run_dir`) rather than `REPO_ROOT/runs`.

---

## [22] SHOULD-FIX An empty, null or mapping-shaped `positions`/`allowed` silently becomes the 18-residue `standard` set

`DesignPosition.parse(text, allowed)` ends with `allowed=list(allowed) if allowed else list(AA_GROUPS["standard"])`, so any falsy `allowed` is treated as "not specified" instead of "nothing allowed". Three plausible agent spellings therefore silently replace the user's stated design space with all 18 standard residues, and `validate` says "spec is usable": `allowed: []` -> 18; `allowed:` with no value -> 18; and `positions:` written as a mapping (`A18: [TRP, PHE]` / `A20: [LEU]`) -> `as_sequence()` iterates the dict's keys, the value lists are discarded entirely, and both positions get 18 residues -- 324 sequences instead of the 2 the user asked for. The run then proceeds on that spec (confirmed in the written `runs/.../spec.json`). This is the exact failure class the module already refuses everywhere else: the explicit spelling `{chain: A, number: 18, allowed: []}` raises "position A18 has an empty allowed-residue set", an unknown position key raises "a typo must not silently change the design space", and `loop.<field>:` written with no value gets a four-line lecture (`LoopSettings.__post_init__`). Only `allowed` is exempt.

**evidence**
S=/private/tmp/claude-501/.../scratchpad/edge-agent-inputs
PY=/opt/miniconda3/envs/ipro-agent/bin/python

$ # positions written as a mapping (user asked for 2 x 1 = 2 sequences)
$ $PY -m ipro_agent validate $S/specs/sh_pos_mapping.yaml
design positions (2):
  A18: VAL->18 allowed (A,R,N,D,Q,E,G,H,I,L,K,M,F,S,T,W,Y,V)
  A20: LEU->18 allowed (A,R,N,D,Q,E,G,H,I,L,K,M,F,S,T,W,Y,V)
design space: 324 sequences
spec is usable                       # exit 0, no problems
$ $PY -m ipro_agent run $S/specs/sh_pos_mapping.yaml --run-dir $S/runs/r_posmap --iterations 1 --json >/dev/null
$ python3 -c "import json;[print(p['chain'],p['number'],len(p['allowed']))for p in json.load(open('$S/runs/r_posmap/spec.json'))['positions']]"
A 18 18
A 20 18

$ # allowed: []   and   allowed: (null)
$ $PY -m ipro_agent validate $S/specs/d_empty.yaml --json | grep design_space
    "design_space_sequences": 18,
$ $PY -m ipro_agent validate $S/specs/d_null_allowed.yaml --json | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['ok'], d['info']['design_space_sequences'])"
True 18

$ # the explicit spelling of the SAME input is refused:
$ $PY -m ipro_agent validate $S/specs/d_empty_explicit.yaml --json
spec error: position A18 has an empty allowed-residue set
$ # and the same YAML shape under loop: is refused with an explanation:
$ $PY -m ipro_agent validate $S/specs/d_null_loop.yaml --json
spec error: loop.integer_cuts has no value. A YAML key written with nothing after the colon parses as null, which would silently replace the default (True). Give it a value or remove the key.

**fix**
In `DesignPosition.parse`, distinguish "absent" from "empty": take `allowed: Sequence | None = None` and use the `standard` default only when it is `None`; an empty list or a null value must reach `__post_init__`'s existing "empty allowed-residue set" error. In `from_dict`, reject a `positions:` value that is a mapping with a message naming the list form (`- {position: A18, allowed: [...]}`) -- `as_sequence` over a dict can only ever lose the values.

---

## [23] SHOULD-FIX A killed or crashed run reports `status: "running"` forever, and status carries no timestamp to detect it

`RunDir.as_json`/`summarize` echo `state.json`'s last-written status with no liveness check, and nothing records the run's pid in the run directory (state.json keys: accepted, backend, best_*, finished, goal, mode, notes, start_score, started, status, total_iterations, updated). SKILL.md and the MCP instructions tell an agent to poll `status` and never read iterations.jsonl, so a run killed by a laptop sleep, an OOM, a `kill -9`, a closed SSH session or the MCP client going away is reported as `running` indefinitely with `error: null`, and the polling agent waits forever. `as_json` also drops the `started`/`updated` timestamps that state.json does have, so the agent cannot even apply a staleness heuristic -- while `tool_run` hands the client a pid it has no way to store or re-check.

**evidence**
S=/private/tmp/claude-501/.../scratchpad/edge-agent-inputs
$ # a 60-iteration run was launched and its process killed 4.5 minutes ago
$ ps -eo pid,command | grep -c "[i]pro_agent run"
0
$ /opt/miniconda3/envs/ipro-agent/bin/python -m ipro_agent status $S/runs/r_mm0 --json | python3 -c "import json,sys;d=json.load(sys.stdin);print({k:d.get(k) for k in ('status','iterations_done','total_iterations','accepted','error')})"
{'status': 'running', 'iterations_done': 20, 'total_iterations': 60, 'accepted': 1, 'error': None}
$ /opt/miniconda3/envs/ipro-agent/bin/python -m ipro_agent status $S/runs/r_mm0 | head -2
run: r_mm0  status: running
iterations: 20/60  accepted: 1
$ python3 -c "import json,time;st=json.load(open('$S/runs/r_mm0/state.json'));print('last updated',round(time.time()-st['updated'],1),'s ago; status:',st['status'])"
last updated 273.1 s ago; status: running
$ # the status payload has no timestamp at all:
keys: ['accepted','backend','best_mutations','best_score','best_sequence','designed_nothing','error','goal','iterations_done','mode','notes','run','start_score','status','total_iterations']
$ # same for a SIGKILL mid-iteration (separate run, r_interrupt):
{'status': 'running', 'mode': 'backbone-mc', 'iterations_done': 2, 'accepted': 0, 'error': None}

**fix**
Write `pid: os.getpid()` and `host` into state.json at start, and in `as_json`/`summarize` report `status: "stale"` (or `"abandoned"`) when status is `running` but the pid is gone (`os.kill(pid, 0)` raising ProcessLookupError) or `updated` is older than a few minutes. Include `started`, `updated` and `seconds_since_update` in `as_json` so an agent can judge for itself, and say in the payload what to do next (re-run into a fresh run_dir).

---

## [24] SHOULD-FIX Re-running into an existing run directory appends to the old trajectory: status then reports 12 of 6 iterations and double the accepted count

`RunDir.__init__(create=True)` does `mkdir(exist_ok=True)` and `append_iteration` opens iterations.jsonl in append mode, so the CLI happily starts a second run in a used directory. This is precisely what an agent does after an interrupted run (and after the stale-`running` bug above). Measured: run A (6 iterations, 2 accepted), then a DIFFERENT spec (different seed and goal_statement) into the same `--run-dir` -> iterations.jsonl holds 12 records numbered 1-6 twice, `status --json` reports `iterations_done: 12` with `total_iterations: 6` and `accepted: 4` (the second run accepted 2), and spec.json describes only run B, so half the recorded trajectory is attributed to a spec that did not produce it. Two runs launched concurrently into one directory interleave their appends ([1,1,2,2,3,3,...]) and both exit 0. `best/` and `ensemble/` are overwritten by whichever finishes last. MCP's `tool_run` does refuse a non-empty run_dir (confirmed), so only the CLI/SKILL path -- the primary interface -- is exposed.

**evidence**
S=/private/tmp/claude-501/.../scratchpad/edge-agent-inputs
PY=/opt/miniconda3/envs/ipro-agent/bin/python
$ $PY -m ipro_agent run $S/specs/mix_a.yaml --run-dir $S/runs/r_mix --json | ...
{'iterations': 6, 'accepted': 2, 'best_score': -1942.5143361274788}
$ wc -l < $S/runs/r_mix/iterations.jsonl
6
$ $PY -m ipro_agent run $S/specs/mix_b.yaml --run-dir $S/runs/r_mix --json | ...   # different spec, same dir
{'iterations': 6, 'accepted': 2, 'best_score': -1942.5143361274788}      # exit 0, no refusal
$ wc -l < $S/runs/r_mix/iterations.jsonl
12
$ $PY -m ipro_agent status $S/runs/r_mix --json | ...
{'status': 'done', 'iterations_done': 12, 'total_iterations': 6, 'accepted': 4}
$ python3 -c "import json;print([json.loads(l)['iteration'] for l in open('$S/runs/r_mix/iterations.jsonl')])"
[1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
$ python3 -c "import json;d=json.load(open('$S/runs/r_mix/spec.json'));print(d['name'], d['goal_statement'], d['loop']['random_seed'])"
mix_b narrow the pore (run B) 7
$ # two concurrent runs, same dir, both exit 0:
exit codes: [0, 0]
records: 12 iteration numbers: [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]

**fix**
Refuse a non-empty run directory in `run_design`/`RunDir(create=True)` with the message `tool_run` already uses ("pass a fresh run_dir, or poll the existing one"), and add an explicit `--resume` flag for the one case that should continue (same spec hash, continue numbering from `last_iteration()`). A simple `fcntl.flock` on state.json would also stop two live runs sharing a directory. state.py's docstring already promises resume ("the loop restarts from the last recorded iteration"); `grep -rn resume ipro_agent/` finds no implementation, so either build it or drop the claim.

---

## [25] NICE-TO-HAVE Zero is silently "unset": `--iterations 0` and `--max-minutes 0` are dropped, and `max_minutes: 0` means no limit

`cmd_run` guards its overrides with `if args.iterations:` and `if args.max_minutes:`, and loop.py:1687 tests `if spec.loop.max_minutes and ...`, so every zero is read as "not given". Measured: a spec with `iterations: 6` run with `--iterations 0` ran all 6 (and spec.json records 6); `--max-minutes 0` ran all 6 with max_minutes recorded as `None`, while `--max-minutes 0.0001` correctly stopped after 1 with the note "stopped on max_minutes before the iteration cap" -- so the zero case is a falsy test, not a deliberate rule. In the spec, `loop.max_minutes: 0` is a 0-minute budget that never fires: that run did 20 iterations in 4.5 minutes and was still going. An agent told to bound an unattended run (ipro_run's own description says "Always pass max_minutes") can get an unbounded one. Separately, `iterations: 0` and `iterations: -5` are accepted by validate and by the loop (`range(1, n+1)` is empty), and the run then reports `ok: true` with `mutation_string: "V18L"`, a full best_sequence and `accepted: 0` -- a design-shaped answer from a search that never ran (the "NO DESIGN WAS ACCEPTED" note is present, but no error and no `designed_nothing` flag). `iterations: "10"` passes validate and dies later with `TypeError: can only concatenate str (not "int") to str`, which names neither the field nor the quote.

**evidence**
S=/private/tmp/claude-501/.../scratchpad/edge-agent-inputs
PY=/opt/miniconda3/envs/ipro-agent/bin/python
$ $PY -m ipro_agent run $S/specs/mix_a.yaml --run-dir $S/runs/r_override0 --iterations 0 --json | ...
iterations actually run: 6  accepted: 2
spec.json total_iterations: 6
$ $PY -m ipro_agent run $S/specs/mix_a.yaml --run-dir $S/runs/r_overridemm --max-minutes 0 --json | ...
iterations run: 6
spec.json max_minutes: None
$ $PY -m ipro_agent run $S/specs/mix_a.yaml --run-dir $S/runs/r_overridemm2 --max-minutes 0.0001 --json | ...
iterations run: 1
notes tail: stopped on max_minutes before the iteration cap
$ # loop.max_minutes: 0 in the spec:
$ $PY -m ipro_agent run $S/specs/long_mm0.yaml --run-dir $S/runs/r_mm0 --json
(no result after 120 s; killed at 273 s)
$ python3 -c "import json;print(json.load(open('$S/runs/r_mm0/spec.json'))['loop']['max_minutes'])"
0
$ wc -l < $S/runs/r_mm0/iterations.jsonl
20
$ # iterations: 0 / -5
$ $PY -m ipro_agent run $S/specs/e_iter0.yaml --run-dir $S/runs/r_e_iter0 --json | ...
{'ok': True, 'iterations': 0, 'accepted': 0, 'mutation_string': 'V18L', 'error': None}
$ $PY -m ipro_agent run $S/specs/e_iterstr.yaml --run-dir $S/runs/r_e_iterstr --json | tail -4
{"ok": false, "error": "TypeError: can only concatenate str (not \"int\") to str", ...}

**fix**
Use `is not None` for all three overrides in `cmd_run` and `if spec.loop.max_minutes is not None` in the loop. Validate in `LoopSettings.__post_init__`: coerce `iterations`/`random_seed` with `int(...)` inside a try that raises SpecError naming the field, require `iterations >= 1`, and require `max_minutes` to be `None` or `> 0` (a 0 budget is a mistake, like the nullable-key check already in that method).

---

## [26] SHOULD-FIX `ipro_run` reports `launched: true` for a child that dies instantly; `ipro_status` then says `pending` forever

`tool_run` creates the run directory, Popens the CLI detached and returns `{launched: true, pid, log}` without ever checking that the child survived. When the child dies before writing state.json -- an argparse rejection of a bad override, a missing interpreter, an import error -- the directory holds only launch.log, so `ipro_status` reports `status: "pending"`, `error: null`, `iterations_done: 0` indefinitely, which is indistinguishable from "queued, starting soon". Measured with `iterations: "ten"` (the schema says integer, but the server validates nothing and MCP clients are not required to): `launched: true`, then launch.log contains `ipro-agent run: error: argument --iterations: invalid int value: 'ten'` and status says pending. Nothing in the status payload points at launch.log. With `iterations: -5` the launch "succeeds" and finishes `done` with `total_iterations: -5`, `accepted: 0`, and a `best_mutations`/`best_sequence` pair that an agent will read as the design.

**evidence**
S=/private/tmp/claude-501/.../scratchpad/edge-agent-inputs
# real stdio MCP session against the clone (cwd = repo root)
--- ipro_run with iterations given as a string
    isError=False
    {"run_id": "r_mcp_iterstr", "pid": 31695, "launched": true,
     "log": ".../runs/r_mcp_iterstr/launch.log",
     "mode_note": "degraded backends: ['pyrosetta','vina']. Poll ipro_status; do not wait on this call..."}
$ ls $S/runs/r_mcp_iterstr
launch.log
$ cat $S/runs/r_mcp_iterstr/launch.log
usage: ipro-agent run [-h] [--json] [--run-dir RUN_DIR] [--iterations ITERATIONS] ...
ipro-agent run: error: argument --iterations: invalid int value: 'ten'
--- ipro_status on the launched-but-dead run
    isError=False
    {"status": "pending", "iterations_done": 0, "error": null, ...}
--- ipro_status on the iterations:-5 run
    {"status": "done", "total_iterations": -5, "iterations_done": 0, "accepted": 0,
     "best_mutations": {"A18": "LEU"}, "best_sequence": "AEIYNKDGNKVDLYGKALGLHY..."}

**fix**
After the Popen, wait briefly (e.g. `proc.wait(timeout=2)` / poll for state.json) and if the child has already exited non-zero, return a ToolError carrying launch.log's tail instead of `launched: true`. Validate the three overrides in `tool_run` (int/float coercion with a readable ToolError) rather than passing them through to argparse. Have `tool_status` read `launch.log` when state.json is absent and say "launched but never started; see launch.log" instead of `pending`.

---

## [27] SHOULD-FIX A truncated PDB validates clean and produces a complete, successful-looking design of the fragment

`Structure.load` only raises when Biopython warns about *discarded* atoms (`_LOSSY_WARNINGS`); a file that simply stops early -- an interrupted download, a half-written prep script, a `head`-ed file -- parses as a short but internally consistent structure. Measured on the shipped example cut to its first 300 lines: `validate` says `ok: true`, "spec is usable", with `design_residues: 38` where the intact file has 340, and the design position A18 happens to exist in the fragment. The run then completes normally: `ok: true`, 2 iterations, 2 accepted, `mutation_string: "V18W"`, and a 38-residue "designed sequence" -- every energy computed on 11% of the porin, with no note or warning distinguishing it from a real design. The one available signal (`design_residues`) is a number the agent has nothing to compare against.

**evidence**
S=/private/tmp/claude-501/.../scratchpad/edge-agent-inputs
PDB=/Users/supantha/Documents/test123/ipro-agent/examples/ompf_pore/2omf_A.pdb
$ head -300 $PDB > $S/structs/truncated.pdb
$ /opt/miniconda3/envs/ipro-agent/bin/python -m ipro_agent validate $S/specs/s_truncated.pdb.yaml --json | ...
ok: True problems: []
info: {'design_chains': ['A'], 'design_residues': 38, 'design_space_sequences': 3}
$ /opt/miniconda3/envs/ipro-agent/bin/python -m ipro_agent run $S/specs/s_truncated.pdb.yaml --run-dir $S/runs/r_trunc --json | ...
ok: True | iterations: 2 | accepted: 2 | mutation: V18W
reported designed sequence length: 38
sequence: AEIYNKDGNKVDLYGKAWGLHYFSKGNGENSYGGNGDM
(no note mentions truncation, residue count or chain completeness)
$ # the intact file, same spec shape:
reported designed sequence length: 340

**fix**
After loading, check the file for structural truncation and raise (or add a prominent note to `validate`/`run` notes): no `END`/`ENDMDL` record while ATOM records are present, a final residue missing backbone atoms (N/CA/C/O), or a chain whose residue numbering ends well before its SEQRES length when SEQRES is present. Reporting `design_residues` beside `SEQRES length` in `validate` would make the mismatch visible even without a hard failure.

---

## [28] SHOULD-FIX Two molecules pointing at the same structure file validate clean and score a +149,211 kcal/mol "interaction energy"

`DesignSpec.__post_init__` refuses duplicate molecule *names* (with a long rationale about double-counted interaction energies) but never compares paths or coordinates. An agent that has one structure and a goal phrased as binding -- "improve the affinity of X for Y" with only X on disk, or a duplicated molecules entry -- writes the design molecule in again as the target. `validate` reports `ok: true, problems: []` and cheerfully lists the partner's 340 residues; `score` then returns `interaction_energies: {partner: 149210.99}`, i.e. the protein superimposed on itself, and `bind_tighter` minimizes exactly that quantity. The run's objective is then pure steric overlap and will be "improved" by whichever residue is smallest. `cmd_score`'s docstring warns about the opposite case ("an interaction energy near zero usually means the partners are not in contact") but nothing flags a six-figure clash.

**evidence**
S=/private/tmp/claude-501/.../scratchpad/edge-agent-inputs
PDB=/Users/supantha/Documents/test123/ipro-agent/examples/ompf_pore/2omf_A.pdb
$ cat $S/specs/m_samepath.yaml      # ompf role design + partner role target, same path, same chain A
$ /opt/miniconda3/envs/ipro-agent/bin/python -m ipro_agent validate $S/specs/m_samepath.yaml --json | ...
ok: True problems: []
partners: {'partner': {'chains': ['A'], 'residues': 340, 'hetero': 0}}
$ /opt/miniconda3/envs/ipro-agent/bin/python -m ipro_agent score $S/specs/m_samepath.yaml --run-dir $S/runs/score_samepath --json | ...
{'mode': 'fixed-backbone', 'interaction_energies': {'partner': 149210.98754476177},
 'design_only': -1938.797556166597, 'total': 145322.8865631499, 'score': 149210.98754476177}
(notes: only the standard fixed-backbone and native-not-allowed notes; nothing about the overlap)

**fix**
In `DesignSpec.__post_init__`, refuse two molecules whose resolved path and chain set are identical (same wording as the duplicate-name check: lookups and energies treat them as distinct bodies). In `validate`/`score`, add a clash pre-flight: if a measured interaction energy exceeds, say, +1000 on the screening scale, report it as a problem ("partners overlap; they are probably the same coordinates or were never docked") rather than returning it as a number to optimize.

---

## [29] SHOULD-FIX `status` on anything that is not a run directory reports `pending`, so a mistyped path polls forever

`cmd_status` builds `RunDir(path, create=False)` and only checks `rd.path.exists()`. For any existing path that is not a run directory -- a regular file, the spec YAML, an unrelated folder -- `state_path.exists()` is False, so `read_state()` returns `{"status": "pending"}` and the agent is told the run has not started yet. Measured for a plain file, for a spec file passed by mistake, and for an empty directory, in both `--json` and human form, and identically through MCP `ipro_status` (which adds its reporting_rule on top). The two most likely agent typos here -- passing the spec path instead of the run path, or passing `runs/case1/state.json` -- both land on "pending".

**evidence**
S=/private/tmp/claude-501/.../scratchpad/edge-agent-inputs
PY=/opt/miniconda3/envs/ipro-agent/bin/python
$ touch $S/runs/a_file_not_dir
$ $PY -m ipro_agent status $S/runs/a_file_not_dir --json | head -3
{
  "run": ".../runs/a_file_not_dir",
  "status": "pending",
$ $PY -m ipro_agent status $S/runs/a_file_not_dir
run: a_file_not_dir  status: pending
iterations: 0/?  accepted: 0
$ $PY -m ipro_agent status $S/specs/base.yaml --json | head -3
{
  "run": ".../specs/base.yaml",
  "status": "pending",
# over MCP, ipro_status run_dir=<the file>:  isError=False, "status": "pending", plus reporting_rule

**fix**
In `cmd_status`/`tool_status`, require the path to be a directory and to contain at least one of state.json, spec.json or iterations.jsonl; otherwise exit 2 with "<path> is not a run directory (no state.json); pass the --run-dir you launched with". Reserve `pending` for a directory that a launch actually created.

---

## [30] SHOULD-FIX `score`, `milp`, `positions` and `dock` crash with a bare traceback for spec problems `validate` reports cleanly -- including an invented goal name

Only `_load` is wrapped; `SpecError`s raised later (by `build_goal`, by the goal's own `validate`, by assembly resolution) escape every subcommand except `validate`, which catches them per-section. An invented goal name is probably the single most likely LLM spec error -- the vocabulary is five fixed strings and `improve_affinity`/`increase_binding` are the obvious guesses. `validate` handles it perfectly (`exit 1`, "unknown goal 'improve_affinity'. Valid goals: bind_tighter, improve_catalysis, reshape_channel, stabilize, switch_specificity"); `score` on the same file prints a 20-line traceback, and over MCP that traceback's tail becomes the error string.

**evidence**
S=/private/tmp/claude-501/.../scratchpad/edge-agent-inputs
PY=/opt/miniconda3/envs/ipro-agent/bin/python
$ $PY -m ipro_agent validate $S/specs/m_samepath.yaml --json     # goal: improve_affinity
{"ok": false, "problems": ["goal: unknown goal 'improve_affinity'. Valid goals: bind_tighter, improve_catalysis, reshape_channel, stabilize, switch_specificity"], ...}   # exit 1
$ $PY -m ipro_agent score $S/specs/m_samepath.yaml --run-dir $S/runs/score_samepath --json
Traceback (most recent call last):
  ...
  File "/Users/supantha/Documents/test123/ipro-agent/ipro_agent/loop.py", line 170, in __init__
    self.goal = goal or build_goal(spec)
  File "/Users/supantha/Documents/test123/ipro-agent/ipro_agent/goals.py", line 781, in build_goal
    raise SpecError(
ipro_agent.config.SpecError: unknown goal 'improve_affinity'. Valid goals: bind_tighter, improve_catalysis, reshape_channel, stabilize, switch_specificity

**fix**
Catch `SpecError` (and `StructureError`) in `main()` around `args.func(args)` once, printing `spec error: <message>` to stderr and exiting 2 -- the same contract `_load` already implements -- so every subcommand gets it. Optionally add a did-you-mean on `build_goal` using difflib against the five goal names.

---

## [31] SHOULD-FIX Wrong-typed `goal_options` numbers pass `validate` and kill the run later, bypassing the module's own hardened coercion

`Goal._positive` (line 153) exists precisely to turn a bad numeric option into `SpecError: goal_options.<name> must be a number`, but `target_radius` (line 482) and `long_fraction` (line 494) call bare `float(...)`. Consequences measured: `target_radius: "two"` -> `validate` reports `ok: true, problems: []`, and the run fails with `ValueError: could not convert string to float: 'two'` naming no option, no file and no line (status then reports that string as the run's error); `target_radius: [2.0, 3.0]` likewise passes validate; `long_fraction: "half"` crashes *inside validate* with a bare traceback (`goals.py:494`). A string where a number was wanted is one of the most common LLM slips (quoted numbers, "two", "50%").

**evidence**
S=/private/tmp/claude-501/.../scratchpad/edge-agent-inputs
PY=/opt/miniconda3/envs/ipro-agent/bin/python
$ tail -1 $S/specs/h_radius_str_3pos.yaml
goal_options: {direction: narrow, target_radius: "two"}
$ $PY -m ipro_agent validate $S/specs/h_radius_str_3pos.yaml --json | ...
ok: True problems: []
$ $PY -m ipro_agent run $S/specs/h_radius_str_3pos.yaml --run-dir $S/runs/r_radius_str --json | tail -4
  "ok": false,
  "error": "ValueError: could not convert string to float: 'two'",
$ $PY -m ipro_agent status $S/runs/r_radius_str --json | ...
{'status': 'failed', 'iterations_done': 0, 'accepted': 0, 'error': "ValueError: could not convert string to float: 'two'"}
$ $PY -m ipro_agent validate $S/specs/h_str_fraction.yaml --json   # long_fraction: "half"
Traceback (most recent call last):
  ...
  File "/Users/supantha/Documents/test123/ipro-agent/ipro_agent/goals.py", line 494, in default_constraints
    frac = float(frac)
ValueError: could not convert string to float: 'half'
$ $PY -m ipro_agent validate $S/specs/h_list_for_scalar.yaml --json | ...   # target_radius: [2.0, 3.0]
True []

**fix**
Coerce every numeric goal option through one helper that raises SpecError with the key name (extend `_positive` with a `_number(name, default, allow_zero=...)` sibling and use it for `target_radius`, `long_fraction`, `pore_diameter_min/max`, `reference_interaction`), and call that coercion from each goal's `validate()` so `ipro-agent validate` fails on the spec instead of the run failing an hour in. `direction` already gets exactly this treatment ("must be 'narrow' or 'widen'; got 'sideways'") and is the model to copy.

---

## [32] NICE-TO-HAVE MCP tool arguments are used unchecked: a missing or wrong-typed `spec` surfaces as KeyError/AttributeError, and a spec path containing a newline is silently parsed as an inline spec

Handlers index `args["spec"]` directly and `_looks_inline` decides path-vs-inline on the presence of a newline, with no type or presence check. Measured over real stdio: no `spec` -> `KeyError: 'spec'`; `spec: 123` / `null` / `["a","b"]` -> `AttributeError: 'int'/'NoneType'/'list' object has no attribute 'strip'`. These are Python internals with no mention of which argument is wrong. A path that genuinely contains a newline (and any inline spec written on one line) is misclassified: the newline case is written to a temp file and reported as `spec error: /var/folders/.../ipro_mcp_spec_9cvnn891/spec.yaml does not contain a mapping at the top level`, naming a path the agent never supplied and cannot connect to its input. The single-line inline case is reported as `spec not found: goal: reshape_channel`, which at least echoes the input.

**evidence**
# real stdio MCP session, tools/call ipro_validate:
--- validate: spec arg missing entirely      {"error": "KeyError: 'spec'"}
--- validate: spec is a number               {"error": "AttributeError: 'int' object has no attribute 'strip'"}
--- validate: spec is null                   {"error": "AttributeError: 'NoneType' object has no attribute 'strip'"}
--- validate: spec is a list                 {"error": "AttributeError: 'list' object has no attribute 'strip'"}
--- validate: spec path with a newline in it
    {"error": "spec error: /var/folders/9d/.../T/ipro_mcp_spec_9cvnn891/spec.yaml does not contain a mapping at the top level"}
--- validate: inline ONE-LINE spec (no newline)
    {"error": "spec not found: goal: reshape_channel\nPass a path to a spec file, or the spec's YAML content inline..."}
# (handled well, for contrast:)
--- validate: directory as spec path  -> "spec path is a directory, not a file: ...  did you mean .../spec_smoke.yaml ?"
--- unknown tool name                 -> rpc error "unknown tool 'ipro_desgin'; have [...]"
--- positions: cutoff is a word       -> "ipro-agent positions: error: argument --cutoff: invalid float value: 'five'"

**fix**
Add a two-line argument check at the top of each handler (`spec = args.get("spec"); if not isinstance(spec, str) or not spec.strip(): raise ToolError("ipro_validate needs `spec`: a path to a spec file, or the spec's YAML content inline")`), and make `_looks_inline` test for spec-shaped content (a line matching `^\s*(goal|goal_statement|molecules|positions)\s*:`) rather than for a newline. When an inline spec fails to parse, echo the first line of what was received alongside the temp path.

---

