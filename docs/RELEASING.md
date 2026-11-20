# Publishing a release

A release is one `git push` of a tag. `.github/workflows/release.yml` does both
publications, and neither of them uses a secret: PyPI through Trusted
Publishing and the MCP Registry through `mcp-publisher login github-oidc`, both
authenticated by the workflow's OIDC token. There is no token to create, store,
paste or leak.

The order is fixed, and the workflow enforces it with job dependencies. The MCP
Registry stores metadata only, so the package has to exist on PyPI before the
registry will accept the `server.json` that points at it.

## Decide first

Publishing to PyPI is public and effectively permanent. A version number cannot
be reused, and the project name is taken from the moment the first upload
succeeds. A release can be yanked but not unpublished.

Say the thing a `pip install` cannot do, because people will try it:
`pip install ipro-agent` gives a working CLI and a working MCP server, and it
cannot produce a design. Rotamers come only from PyMOL's mutagenesis wizard,
`pymol-open-source` is conda-only, and there is no fallback. The README states
this and the README is what PyPI shows as the project description.

## Releasing

Bump the version everywhere first, then tag. The tag is the trigger, and the
`check` job reads the version out of the tagged tree, so a tag pushed before
the bump releases the old number.

```bash
# 1. bump, in all six places
#      pyproject.toml            version
#      server.json               version AND packages[0].version
#      ipro_agent/__init__.py    __version__
#      ipro_agent/mcp_server.py  SERVER_INFO
#      CITATION.cff              version, and date-released
#
# 2. commit, then tag that commit
git tag v0.1.1 && git push origin v0.1.1
```

`test_server_json_matches_the_package_it_points_at` checks all six agree, so
run the suite before tagging and a mismatch fails locally rather than mid
release.

The workflow then runs three jobs:

| job | what it does | what stops it |
|---|---|---|
| `check` | tag, `pyproject.toml` and both `server.json` versions must agree; the README must still carry the `mcp-name:` marker | a mismatched tag, a dropped marker |
| `pypi` | `build`, `twine check`, a wheel-contents check, then Trusted Publishing | a wheel missing the files below |
| `registry` | `mcp-publisher validate`, then `login github-oidc` and `publish` | a `description` over 100 characters |

The wheel must contain `ipro_agent/doctor.py`, `ipro_agent/mcp_server.py` and
both files under `ipro_agent/pymol_helpers/`, and the `pypi` job asserts it.
Without the helpers an installed copy validates a spec and then dies at the
first rotamer harvest; without the doctor, `ipro_doctor`, `ipro_setup` and
`ipro_run` all fail in an installed copy while the other tools keep working,
which is how that defect hid (`docs/DESIGN.md` §7h).

`mcp-publisher validate` catches a limit the published JSON Schema does not
carry: `description` is capped at 100 characters, and a 400-character one
passed every local check before failing there.

## One-time setup, done once and easy to forget

Trusted Publishing has to be registered on PyPI before the first release, and
it is the one piece that does not live in this repository. Without it the
`pypi` job fails with `invalid-publisher: valid token, but no corresponding
publisher`, which is confusing because the token is valid and the workflow is
correct. Linking GitHub to a PyPI account as a sign-in method is a different
thing and does not count.

At <https://pypi.org/manage/account/publishing/>, under "Add a new **pending**
publisher" while the project does not exist yet, or under the project's own
Publishing tab afterwards:

| field | value |
|---|---|
| PyPI project name | `ipro-agent` |
| Owner | `chowdhurylab-studio` |
| Repository name | `ipro-agent` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

These must match the OIDC claims exactly. A failed run prints the claims it
sent, which is the fastest way to see which field disagrees.

## Afterwards

- Check the published page renders the README, including the images and the
  links. The README is the long description, PyPI resolves a relative path
  against `pypi.org`, and a test now requires every image and link in it to be
  absolute for that reason.
- Check the page still contains `mcp-name: io.github.chowdhurylab-studio/ipro-agent`.
  The registry reads that string out of the PyPI description to prove you own
  the package, so a release whose description lost it cannot be registered.
- Create a GitHub Release for the tag. The workflow does not, so without this
  the Releases page is empty for a package that is live.

## If the workflow cannot be used

The manual route still works and needs credentials, which is the reason to
prefer the workflow:

```bash
python3 -m build && python3 -m twine check dist/* && python3 -m twine upload dist/*
mcp-publisher validate && mcp-publisher login github && mcp-publisher publish
```
