"""ipro-agent: agent-drivable Iterative Protein Redesign and Optimization.

A generalized re-implementation of the IPRO procedure (Maranas group) that
takes a design goal stated in plain language, a set of structures, and the
positions/residues the user permits, and runs the IPRO iteration loop to
produce a design sequence.

See .claude/skills/ipro-agent/SKILL.md for the agent-facing interface and docs/DESIGN.md for the mapping
from each capability here to the paper and equation it came from.
"""

import sys as _sys

#: The floor `pyproject.toml` declares. Checked here as well, because
#: `requires-python` only stops **pip**: someone who clones the repo and runs
#: `python3 -m ipro_agent` never goes through pip at all. Measured on 3.9.6,
#: `import ipro_agent.config` succeeds, so without this the run gets well past
#: the gate and then fails somewhere unrelated and confusing. One clear sentence
#: at import time is worth more than a traceback ten modules deep.
REQUIRES_PYTHON = (3, 10)

if _sys.version_info < REQUIRES_PYTHON:
    raise RuntimeError(
        f"ipro-agent needs Python "
        f"{REQUIRES_PYTHON[0]}.{REQUIRES_PYTHON[1]} or newer; this is "
        f"{_sys.version.split()[0]} at {_sys.executable}.\n"
        f"The documented environment installs a supported Python:\n"
        f"    conda env create -f environment.yml && conda activate ipro-agent\n"
        f"See docs/INSTALL.md."
    )

__version__ = "0.1.0"
