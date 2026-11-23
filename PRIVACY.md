# Privacy

ipro-agent runs entirely on your machine. There is no account, no server
operated by the author, and no telemetry. Nothing you design, load or type is
sent anywhere by this software, and the author receives no data about you or
your work at all.

That is a statement about the code, so here is the code it is based on.

## What touches the network

The package makes a network request in exactly two places, both of which you
start deliberately:

- **`ipro_fetch_structure` / `ipro-agent fetch`** downloads a structure from
  the RCSB Protein Data Bank (`https://files.rcsb.org/download/`). This tells
  RCSB which PDB entry you asked for, and your IP address, exactly as opening
  that page in a browser would. It does not tell them anything else, and it
  happens only when you ask for an entry by ID.
- **`ipro_setup` with `confirm: true`**, on an installed copy that cannot see
  a checkout, fetches `environment.yml` from this project's own GitHub
  repository before creating a conda environment. The command is printed in
  full by the same tool without `confirm`, so you can read it first.

Nothing else reaches out. `grep` for `urlopen` in the package finds one call
site, and the other URLs in the source are text the tool prints for you to
read, not addresses it contacts.

## What stays on your machine

Structures, specs, run directories, logs, designed coordinates, and the
energy tables are files on your disk, written where you point the tool. They
are never uploaded. A run directory is yours to keep or delete; this software
never deletes one.

## What your AI client does

ipro-agent is an MCP server, so an AI client starts it and exchanges messages
with it on your machine over standard input and output. What that client then
sends to its own provider is governed by that client's privacy policy, not
this one. If you ask an assistant to design a protein, the sequence and the
numbers it reports back to you travel through that assistant in the ordinary
way. This project has no visibility into that and no agreement with you about
it: read your client's policy.

## Third-party components

PyMOL, PyRosetta, AutoDock Vina and the MILP solvers are separately licensed
and separately governed. ipro-agent runs them locally as subprocesses and
sends them nothing beyond the structure and the task at hand. See
`docs/INSTALL.md` for what each one is and `NOTICE.md` for licensing.

## Changes

This file is versioned with the software. Its history is the record of any
change, and there is no other copy.

## Contact

Open an issue: https://github.com/chowdhurylab-studio/ipro-agent/issues
