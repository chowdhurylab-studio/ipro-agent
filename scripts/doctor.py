#!/usr/bin/env python3
"""Thin shim: the doctor lives in the package, so a pip install also has it.

    python3 scripts/doctor.py --json

is exactly ``python3 -m ipro_agent.doctor --json``. Kept because it is the
command in INSTALL.md, README.md and SKILL.md, and because running it from a
clone must work before anything is installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ipro_agent.doctor import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
