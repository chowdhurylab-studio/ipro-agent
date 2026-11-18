#!/usr/bin/env python3
"""Run the test suite without pytest, and exit non-zero on failure.

    python3 scripts/selftest.py            # everything
    python3 scripts/selftest.py -k milp    # only tests whose name matches

pytest is not a dependency of this repo, so this runner exists to make the
tests usable in a bare environment. If pytest *is* installed, `pytest tests/`
works too and gives better output.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_ipro import collect  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", dest="pattern", default="", help="substring filter on test name")
    ap.add_argument("-q", dest="quiet", action="store_true")
    args = ap.parse_args()

    tests = [(n, f) for n, f in collect() if args.pattern in n]
    if not tests:
        print(f"no tests matched {args.pattern!r}", file=sys.stderr)
        return 2

    passed, failed = 0, []
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
        except Exception:
            failed.append((name, traceback.format_exc()))
            print(f"FAIL {name}")
        else:
            passed += 1
            if not args.quiet:
                print(f"ok   {name}  ({time.time() - t0:.2f}s)")

    print(f"\n{passed} passed, {len(failed)} failed, {len(tests)} total")
    for name, tb in failed:
        print(f"\n--- {name} ---\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
