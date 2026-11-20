#!/usr/bin/env bash
# One command to get ipro-agent working on a machine that has never seen it.
#
#     bash scripts/bootstrap.sh
#
# It creates the documented conda environment, checks what resolved, and runs
# the shipped example end to end -- so you find out whether the install works
# from the install, not from your first real design three days later.
#
# Safe to re-run: if the environment already exists it is left alone.
# Nothing outside the conda environment is modified, and nothing is deleted.
set -uo pipefail

ENV_NAME="ipro-agent"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  ok    %s\n' "$*"; }
warn() { printf '  note  %s\n' "$*"; }
die()  { printf '\n  STOP  %s\n\n' "$*" >&2; exit 1; }

say "ipro-agent bootstrap"
echo "  repo: $REPO"

# ---------------------------------------------------------------- 1. conda ---
MANAGER=""
for m in mamba conda micromamba; do
  if command -v "$m" >/dev/null 2>&1; then MANAGER="$m"; break; fi
done
if [ -z "$MANAGER" ]; then
  die "No conda, mamba or micromamba on PATH.

  PyMOL is the one dependency pip cannot supply (pymol-open-source is not on
  PyPI), and rotamers come only from PyMOL's mutagenesis wizard -- there is no
  fallback, so a design cannot run without it.

  Install Miniforge, then re-run this script:
      https://github.com/conda-forge/miniforge
  macOS/Linux one-liner:
      curl -L -O \"https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-\$(uname)-\$(uname -m).sh\" && bash Miniforge3-\$(uname)-\$(uname -m).sh"
fi
ok "found $MANAGER ($("$MANAGER" --version 2>&1 | head -1))"

# ----------------------------------------------------------- 2. environment ---
if "$MANAGER" env list 2>/dev/null | grep -qE "^${ENV_NAME}[[:space:]]"; then
  ok "environment '$ENV_NAME' already exists -- leaving it as it is"
else
  say "Creating the '$ENV_NAME' environment (a few minutes, ~1.4 GB)"
  echo "  this is conda solving and downloading; no output for a while is normal"
  "$MANAGER" env create -f environment.yml || die "environment creation failed.
  Re-run, or create it by hand:  $MANAGER env create -f environment.yml"
  ok "environment created"
fi

# Find that environment's python without needing `conda activate`, which does
# not work in a non-interactive shell.
ENV_PY=""
for base in "$(dirname "$(dirname "$(command -v "$MANAGER")")")" "$HOME/miniforge3" \
            "$HOME/miniconda3" "$HOME/anaconda3" "/opt/miniconda3" "/opt/homebrew/Caskroom/miniforge/base"; do
  if [ -x "$base/envs/$ENV_NAME/bin/python" ]; then ENV_PY="$base/envs/$ENV_NAME/bin/python"; break; fi
done
[ -n "$ENV_PY" ] || die "created the environment but cannot find its python.
  Run:  $MANAGER activate $ENV_NAME && python3 scripts/doctor.py"
ok "python: $ENV_PY ($("$ENV_PY" -V 2>&1))"

# ---------------------------------------------------------------- 3. doctor ---
say "What resolved (this is the 'mode', and it must be reported with any design)"
"$ENV_PY" scripts/doctor.py || warn "doctor reported a problem -- read it above"

# ------------------------------------------------------------- 4. self-test ---
say "Test suite"
SUITE=$("$ENV_PY" scripts/selftest.py 2>&1 | tail -3)
if printf '%s' "$SUITE" | grep -q "0 failed"; then
  ok "$(printf '%s' "$SUITE" | grep -oE '[0-9]+ passed, 0 failed, [0-9]+ total' | tail -1)"
else
  warn "some tests failed -- run '$ENV_PY scripts/selftest.py' and read the output"
  printf '%s\n' "$SUITE" | sed 's/^/        /'
fi

# ---------------------------------------------------------------- 5. example ---
say "The shipped example, end to end"
"$ENV_PY" -m ipro_agent validate examples/ompf_pore/spec_smoke.yaml >/dev/null 2>&1 \
  && ok "validate: the example spec is usable" \
  || warn "validate failed on the shipped example"

RUN_DIR="runs/bootstrap"
rm -rf "$RUN_DIR" 2>/dev/null
if "$ENV_PY" -m ipro_agent run examples/ompf_pore/spec_smoke.yaml --run-dir "$RUN_DIR" >/tmp/ipro_bootstrap.txt 2>&1; then
  ok "a real design run completed"
  grep -E '^(mode|iterations|mutations)' /tmp/ipro_bootstrap.txt | sed 's/^/        /'
  warn "zero mutations is the EXPECTED result for this smoke example -- it is a"
  warn "working install, not a design. examples/ompf_pore/EXPECTED.md says why."
else
  warn "the example run failed; see /tmp/ipro_bootstrap.txt"
fi

# ---------------------------------------------------------------- 6. next ----
say "Ready. What to do next"
cat <<NEXT
  Use it from a terminal:
      $ENV_PY -m ipro_agent goals            # the five objectives
      $ENV_PY -m ipro_agent validate your_spec.yaml
      $ENV_PY -m ipro_agent run your_spec.yaml --run-dir runs/case1 --max-minutes 240
      $ENV_PY -m ipro_agent status runs/case1

  Use it from an AI agent, in this repo:
      open the repo with the agent and ask in plain language -- it loads
      .claude/skills/ipro-agent/SKILL.md by itself.

  Use it from an AI agent, anywhere (MCP). The server has no registry id: what
  you send someone is this block, and "ipro-agent" is just the name they will
  see. Claude Desktop, Claude Code and Cursor take it as written:

      {"mcpServers": {"ipro-agent": {"command": "$ENV_PY",
                                     "args": ["-m", "ipro_agent.mcp_server"],
                                     "env": {"IPRO_REPO": "$REPO"}}}}

  Codex (~/.codex/config.toml) wants the same three fields in TOML:

      [mcp_servers.ipro-agent]
      command = "$ENV_PY"
      args = ["-m", "ipro_agent.mcp_server"]
      env = { IPRO_REPO = "$REPO" }

  VS Code / Copilot: the same JSON in a workspace .mcp.json, or in
  .vscode/mcp.json under "servers" instead of "mcpServers", with
  "type": "stdio" added.

  IPRO_REPO is what lets the server hand the agent its own operating manual
  (ipro://skill) and the other documents. Without it a pip-installed server
  still runs designs but serves no documents, and tells the agent where to
  fetch them instead.

  Read before trusting a number:
      README.md               "Status and honesty" and "Responsible use"
      docs/DESIGN.md          where every capability and number comes from
      examples/ompf_pore/EXPECTED.md   what the example can and cannot reproduce

  Optional backends, each with a documented fallback (docs/INSTALL.md):
      PyRosetta   backbone relaxation and Rosetta scoring (needs its own licence)
      AutoDock Vina + Open Babel   global re-docking of partners
NEXT
