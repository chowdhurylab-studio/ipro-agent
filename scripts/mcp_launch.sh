#!/usr/bin/env sh
# Start the ipro-agent MCP server under the interpreter the user chose.
#
# Why this file exists rather than naming the interpreter directly in
# plugin-mcp.json: the plugin directory's validator blocks an MCP `command`
# that contains any variable other than ${CLAUDE_PLUGIN_ROOT}, and the
# interpreter here cannot be known in advance -- PyMOL is conda-only, so the
# python that can import it is wherever the user built their environment.
# The command is therefore this file, a fixed path under the plugin root, and
# the interpreter arrives in the environment instead.
#
# A validator that follows shell scripts will still flag the $IPRO_PYTHON
# expansion below and hold the version for a human reviewer. That is the
# honest outcome for a tool whose whole job is to drive a native application
# the user installed themselves, and a hold is not a rejection.
set -eu

if [ -z "${IPRO_PYTHON:-}" ]; then
    echo "ipro-agent: no interpreter configured." >&2
    echo "  Set this plugin's \"Python interpreter\" option to the python of a" >&2
    echo "  conda environment that has pymol-open-source AND ipro-agent:" >&2
    echo "    conda create -n ipro-agent -c conda-forge pymol-open-source python=3.12" >&2
    echo "    conda activate ipro-agent && pip install ipro-agent && which python" >&2
    exit 1
fi

if [ ! -x "$IPRO_PYTHON" ]; then
    echo "ipro-agent: \"$IPRO_PYTHON\" is not an executable file." >&2
    exit 1
fi

exec "$IPRO_PYTHON" -m ipro_agent.mcp_server
