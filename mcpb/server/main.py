"""Launcher for the ipro-agent MCP server inside a Claude Desktop extension.

This bundle deliberately ships no Python dependencies. MCPB's `python` server
type expects them vendored under `server/lib` or `server/venv`, and that cannot
work here: the one dependency that decides whether a design is possible at all
is PyMOL, which is conda-only and native, so vendoring the pip half would give
a bundle that imports cleanly and still cannot produce a rotamer.

So the bundle asks for an interpreter instead, and this file is what that
interpreter runs. Its whole job is to turn the most likely setup mistake --
pointing the picker at a python that does not have ipro-agent installed --
into a sentence rather than a traceback.
"""

import sys


def _die(message: str) -> int:
    sys.stderr.write("ipro-agent: " + message + "\n")
    return 1


def main() -> int:
    try:
        from ipro_agent.mcp_server import main as serve
    except ImportError as exc:
        return _die(
            f"this interpreter cannot import ipro_agent ({exc}).\n"
            f"  interpreter: {sys.executable}\n"
            f"  Install it into the SAME environment that has PyMOL:\n"
            f"    conda activate ipro-agent && pip install ipro-agent\n"
            f"  then set this extension's interpreter to that environment's "
            f"python, which `conda activate ipro-agent && which python` prints."
        )
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
