"""Scripts run by a *separate* PyMOL interpreter, not imported by this package.

PyMOL usually lives in its own conda environment, so these two are executed as
``[pymol_python, "-I", <this file>, job.json, out.json]`` rather than imported
(see :mod:`ipro_agent.rotamers` and :mod:`ipro_agent.mutate`). They live inside
the package so that they ship in the wheel: when they sat in the repo's
top-level ``scripts/`` directory, ``pip install .`` produced an ``ipro-agent``
command that could validate a spec and then died at the first rotamer harvest,
because the path resolved to a ``site-packages/scripts/`` that does not exist.

Do not import them from library code -- they ``import pymol`` at module level.
"""
