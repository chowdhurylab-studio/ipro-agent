# Publishing a release

Two separate publications, in this order. The MCP Registry stores metadata
only, so the package has to exist on PyPI before the registry will accept the
`server.json` that points at it.

Both steps need credentials, so they are run by a person, not by an agent.

## Decide first

Publishing to PyPI is public and effectively permanent. A version number cannot
be reused, and the project name is taken from the moment the first upload
succeeds. A release can be yanked but not unpublished.

Say the thing a `pip install` cannot do, because people will try it:
`pip install ipro-agent` gives a working CLI and a working MCP server, and it
cannot produce a design. Rotamers come only from PyMOL's mutagenesis wizard,
`pymol-open-source` is conda-only, and there is no fallback. The README states
this and the README is what PyPI shows as the project description.

## 1. PyPI

Build from a clean checkout and check the artifacts before uploading:

```bash
python3 -m build
python3 -m twine check dist/*
```

The wheel must contain `ipro_agent/doctor.py`, `ipro_agent/mcp_server.py` and
both files under `ipro_agent/pymol_helpers/`. Without the helpers an installed
copy validates a spec and then dies at the first rotamer harvest; without the
doctor, `ipro_doctor`, `ipro_setup` and `ipro_run` all fail in an installed
copy while the other tools keep working, which is how that defect hid
(`docs/DESIGN.md` §7h).

Upload with a token, or configure PyPI Trusted Publishing for this repository
and let a tagged GitHub Actions run do it without any token at all:

```bash
python3 -m twine upload dist/*
```

Then check the published page renders the README, and that it contains the line
`mcp-name: io.github.chowdhurylab-studio/ipro-agent`. The MCP Registry reads
that string out of the PyPI description to prove you own the package, so a
release whose description lost it cannot be registered.

## 2. The MCP Registry

`server.json` at the repository root is already written and validates against
the official schema. Its `version` and its `packages[0].version` must both
match the version just uploaded to PyPI.

Validate first. It checks against the live registry, needs no credentials, and
catches limits the published JSON Schema does not carry: `description` is
capped at 100 characters, and a 400-character one passed every local check
before failing here.

```bash
mcp-publisher validate
mcp-publisher login github
mcp-publisher publish
```

The GitHub login proves you own the `io.github.chowdhurylab-studio` namespace,
which is why the server name begins with it.

After that the server is discoverable by name in clients that read the
registry, and the manual configuration block in `README.md` keeps working for
clients that do not.

## 3. Afterwards

- Tag the commit that was released: `git tag v0.1.0 && git push --tags`.
- Bump `version` in `pyproject.toml` and in both places in `server.json`
  together. A mismatch is rejected at publish time.
- `CITATION.cff` carries the version people will cite. Bump it too.
