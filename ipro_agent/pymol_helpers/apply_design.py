#!/usr/bin/env python3
"""Apply a chosen design to a structure with PyMOL's mutagenesis wizard.

Runs *inside* a PyMOL-capable interpreter (often not the interpreter running
ipro-agent -- see :mod:`ipro_agent.mutate` for the dispatch). Reads a job file,
mutates one position at a time, and writes the resulting structure plus a JSON
report.

    pymol-python apply_design.py jobs.json out.json

Job file::

    {"pdb": "/path/in.pdb", "out": "/path/out.pdb",
     "jobs": [{"chain": "A", "number": 16, "icode": "", "aa": "TRP", "state": 2}]}

Output::

    {"pdb": ..., "out": ..., "pymol_version": ...,
     "applied": {"A16": ["TRP", 2]},
     "native":  {"A16": "CYS"},
     "sequences": {"A": "TTCCPSIVARSNFNVWRF..."},
     "errors":  {"A16": "reason"}}

Why the same wizard that harvested the rotamers: a rotamer is identified
downstream by a bare integer ``state``, which means nothing on its own. It
indexes PyMOL's rotamer set for that residue at that position, in the order
PyMOL enumerates it. Committing the choice with any other builder -- even one
using the same library -- would silently apply a different conformation than
the MILP scored. ``ipro_agent/pymol_helpers/harvest_rotamers.py`` and this script are therefore
two halves of one contract.

Mutations are applied one at a time against the progressively mutated object,
because that is the structure the next side chain must be built into; the
wizard is re-created per position because it retains the previous selection
and mode.

Picking the state is the subtle part. ``wizard.do_state(n)`` does **not** choose
the rotamer to apply: it is the hook PyMOL's GUI calls *after* the viewer has
moved to state ``n``, and on its own it only prints the strain and redraws
polar contacts. ``wizard.apply()`` copies the rotamer at ``cmd.get_state()``
(mutagenesis.py: ``src_frame = cmd.get_state()`` feeding ``cmd.create``), and
selecting a residue leaves the viewer on the *best-scoring* rotamer rather than
the first (``do_library`` ends with ``cmd.frame(state_best)``). So the state is
set with ``cmd.frame(n)``, and the applied coordinates are checked against the
requested state afterwards. Verified on PyMOL 3.1.0 with TRP at A16 of
tests/data/1crn.pdb: without ``cmd.frame``, asking for state 1 or 5 silently
applied state 2 (that residue's lowest-strain rotamer).
"""

from __future__ import annotations

import json
import sys
from typing import Any

# Supplied by the structure, not by the rotamer: excluded when checking that
# the applied side chain is the one that was asked for. OT1/OT2/OT are the
# CHARMM/AMBER spellings of the C-terminal carboxylate.
BACKBONE = {"N", "CA", "C", "O", "OXT", "OT1", "OT2", "OT"}


def _is_hydrogen(atom) -> bool:
    """True for a hydrogen, by element where PyMOL gives one, else by name.

    The state check below compares the applied side chain against the rotamer
    the MILP scored, and the rotamers are heavy-atom-only (see
    ``ipro_agent/pymol_helpers/harvest_rotamers.py``). Including hydrogens here made the check
    fail on an atom that is not part of the rotamer at all: on a protonated
    input it reported "applied side chain is not rotamer state 4: 1 atom(s)
    differ (H)", the whole structure write was suppressed, and the run blamed
    the user's PyMOL build while doctor.py reported PyMOL working.
    """
    sym = (getattr(atom, "symbol", "") or "").strip().upper()
    if sym:
        return sym in ("H", "D", "T")
    name = (getattr(atom, "name", "") or "").strip().upper().lstrip("0123456789")
    return name[:1] in ("H", "D")

#: The wizard is always handed a named selection, never a selection expression,
#: because ``do_select`` finishes with ``cmd.delete(selection)``: given the
#: expression ``prot and chain A and resi 16A`` that call matches and deletes
#: the object ``prot`` itself, leaving nothing to mutate (verified on PyMOL
#: 3.1.0: 327 atoms to 0, then "Mutagenesis: object not found" from apply()).
#: With a name, the same call removes only this temporary selection.
POS_SELE = "_ipro_mut_pos"


def _selection(chain: str, number: int, icode: str, obj: str = "prot") -> str:
    r"""PyMOL selection for one residue.

    Always the ``resi`` form, with a negative number escaped. Both of the
    obvious alternatives are silently wrong for the negative residue numbers
    an N-terminal insertion creates (it numbers downward so that nothing
    existing is renumbered):

    * ``/prot//A/-1`` selects **0 atoms** -- the slash form does not take a
      leading minus at all;
    * ``resi -1`` selects **residues -1, 0 and 1** -- PyMOL reads ``-`` as its
      range operator, so this is the range "up to 1" (measured: 15 atoms where
      the residue has 4).

    ``resi \-1`` is the one form that selects exactly that residue. Insertion
    codes append to the number, e.g. ``resi 16A``.
    """
    resi = f"{number}{icode}".strip()
    if number < 0:
        resi = "\\" + resi
    return f"{obj} and chain {chain} and resi {resi}"


def _sequence(cmd, chain: str, obj: str = "prot") -> str:
    """One-letter sequence of ``chain``, using PyMOL's own residue mapping.

    ``get_fastastr`` returns a FASTA record (header plus wrapped lines), so the
    header is dropped and the wrapping removed.
    """
    text = cmd.get_fastastr(f"{obj} and chain {chain} and polymer")
    return "".join(
        line.strip() for line in text.splitlines() if not line.startswith(">")
    )


def apply_design(job_path: str, out_path: str) -> int:
    with open(job_path) as fh:
        job = json.load(fh)

    import pymol
    from pymol import cmd

    pymol.finish_launching(["pymol", "-cq"])
    # Quieten per-rotamer chatter; the strain of the applied state is reported
    # here instead.
    try:
        cmd.feedback("disable", "executive", "actions")
    except Exception:
        pass

    cmd.load(job["pdb"], "prot")
    result: dict[str, Any] = {
        "pdb": job["pdb"],
        "out": job["out"],
        "pymol_version": str(cmd.get_version()[0]),
        "applied": {},
        "native": {},
        "sequences": {},
        "errors": {},
    }

    for spec in job["jobs"]:
        chain = str(spec["chain"])
        number = int(spec["number"])
        icode = str(spec.get("icode", "") or "")
        aa = str(spec["aa"]).upper()
        state = int(spec["state"])
        label = f"{chain}{number}{icode}"
        sel = _selection(chain, number, icode)

        try:
            native = cmd.get_model(f"({sel}) and name CA").atom[0].resn
        except Exception as exc:
            result["errors"][label] = f"residue not found in structure ({exc})"
            continue
        result["native"][label] = native

        try:
            # A fresh wizard per position: it keeps the previous mode and
            # selection, and reusing one silently mutates the wrong residue.
            cmd.wizard("mutagenesis")
            w = cmd.get_wizard()
            cmd.refresh_wizard()
            w.set_mode(aa)
            cmd.select(POS_SELE, sel)
            w.do_select(POS_SELE)
            cmd.refresh_wizard()

            n_states = cmd.count_states("mutation")
            if n_states < 1:
                raise RuntimeError(f"PyMOL offered no {aa} rotamers here")
            if not 1 <= state <= n_states:
                raise RuntimeError(
                    f"state {state} out of range: PyMOL has {n_states} {aa} "
                    f"rotamer(s) at this position (states are 1-based)"
                )

            # The coordinates the MILP scored for this state, kept so the
            # applied side chain can be checked against them.
            want = {
                a.name: tuple(a.coord)
                for a in cmd.get_model("mutation", state=state).atom
                if a.name not in BACKBONE and not _is_hydrogen(a)
            }

            cmd.frame(state)
            w.do_state(state)
            w.apply()
            cmd.set_wizard()

            # The wizard reports failures by leaving the residue alone, so read
            # the identity back rather than trusting apply().
            ca = cmd.get_model(f"({sel}) and name CA").atom
            if not ca:
                raise RuntimeError(
                    f"the residue is not selectable after apply(): "
                    f"{sel!r} matches nothing"
                )
            got = ca[0].resn
            if got != aa:
                raise RuntimeError(
                    f"wizard left {got} in place of the requested {aa}"
                )

            # And the right identity is not enough: a different rotamer of the
            # right residue would be a structure the MILP never scored.
            have = {
                a.name: tuple(a.coord)
                for a in cmd.get_model(sel).atom
                if not _is_hydrogen(a)
            }
            off = [
                n for n, c in want.items()
                if n not in have
                or max(abs(have[n][i] - c[i]) for i in range(3)) > 1e-3
            ]
            if off:
                raise RuntimeError(
                    f"applied side chain is not rotamer state {state}: "
                    f"{len(off)} atom(s) differ ({', '.join(sorted(off)[:5])}). "
                    f"This PyMOL build may choose the applied state "
                    f"differently; see the module docstring of "
                    f"ipro_agent/pymol_helpers/apply_design.py"
                )
            result["applied"][label] = [aa, state]
        except Exception as exc:  # one bad position must not kill the design
            result["errors"][label] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                cmd.set_wizard()
                cmd.delete("mutation")
            except Exception:
                pass

    for chain in dict.fromkeys(str(s["chain"]) for s in job["jobs"]):
        try:
            result["sequences"][chain] = _sequence(cmd, chain)
        except Exception as exc:
            result["errors"].setdefault(f"chain {chain}", f"sequence unreadable ({exc})")

    # Saved even when some positions failed: the caller needs to be able to look
    # at what was produced, and ``errors`` says it is not the requested design.
    cmd.save(job["out"], "prot")

    with open(out_path, "w") as fh:
        json.dump(result, fh)

    muts = ", ".join(
        f"{result['native'].get(k, '?')}{k}{v[0]}"
        for k, v in sorted(result["applied"].items())
    )
    print(
        f"applied {len(result['applied'])}/{len(job['jobs'])} mutations"
        + (f": {muts}" if muts else "")
        + (f" ({len(result['errors'])} errors)" if result["errors"] else ""),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(apply_design(sys.argv[1], sys.argv[2]))
