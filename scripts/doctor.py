#!/usr/bin/env python3
"""Thin shim: the doctor lives in the package, so a pip install also has it.

    python3 scripts/doctor.py --json

is the same report as ``python3 -m ipro_agent.doctor --json``.

**It loads the module by file path, deliberately**, instead of importing
``ipro_agent.doctor``. The package's ``__init__`` refuses to import on a Python
below the floor -- correctly, since the code uses 3.10 syntax -- and that
refusal would take the doctor down with it on exactly the interpreter whose
diagnosis matters most. Measured on Python 3.9.6 with every dependency
installed: ``python -m ipro_agent.doctor`` prints the package's upgrade notice
and nothing else, where this prints the full report with PYTHON TOO OLD at the
top. ``ipro_agent/doctor.py`` imports nothing from the package, which is what
makes this possible; the test suite checks that it stays that way.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE = Path(__file__).resolve().parent.parent / "ipro_agent" / "doctor.py"

if not _MODULE.is_file():                                  # pragma: no cover
    sys.stderr.write(f"cannot find the doctor at {_MODULE}\n")
    raise SystemExit(1)

_spec = importlib.util.spec_from_file_location("_ipro_doctor", _MODULE)
_doctor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_doctor)

if __name__ == "__main__":
    raise SystemExit(_doctor.main())
