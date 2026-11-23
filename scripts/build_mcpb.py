#!/usr/bin/env python3
"""Pack the Claude Desktop extension (.mcpb).

A .mcpb is a zip holding manifest.json plus whatever the server needs. This one
bundles no Python dependencies -- see mcpb/server/main.py for why -- but it
does bundle the eight documents the server offers as MCP resources, which ship
with the repository rather than inside the wheel. They total about 300 KB, so
the bundle is self-contained for documents and needs no checkout.

The document list is read out of ipro_agent/mcp_server.py rather than written
here, so a document added to the server cannot silently go missing from the
bundle, and a document the server names but the repo does not have is an error
rather than a resource that 404s on the user's machine.

Writes the zip entry by entry, with no staging directory, so there is nothing
to clean up afterwards.
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def documents() -> list[str]:
    src = (REPO / "ipro_agent" / "mcp_server.py").read_text()
    return list(dict.fromkeys(re.findall(r'"path": "([^"]+)"', src)))


def build(out: Path) -> int:
    manifest = REPO / "mcpb" / "manifest.json"
    launcher = REPO / "mcpb" / "server" / "main.py"
    for f in (manifest, launcher):
        if not f.is_file():
            sys.exit(f"missing {f.relative_to(REPO)}")

    docs = documents()
    if not docs:
        sys.exit("no documents found in mcp_server.py")
    missing = [d for d in docs if not (REPO / d).is_file()]
    if missing:
        sys.exit(f"the server offers {missing}, which the repo does not have")

    out.parent.mkdir(parents=True, exist_ok=True)
    # "w" truncates, so a rebuild never inherits a stale entry.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(manifest, "manifest.json")
        z.write(launcher, "server/main.py")
        for rel in docs:
            z.write(REPO / rel, f"repo/{rel}")
    return len(docs)


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "dist" / "ipro-agent.mcpb"
    n = build(target)
    print(f"built {target}")
    print(f"  {n} documents bundled, {target.stat().st_size // 1024} KB")
