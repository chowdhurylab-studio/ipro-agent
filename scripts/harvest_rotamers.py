#!/usr/bin/env python3
"""Enumerate side-chain rotamers at design positions, using PyMOL's own library.

Runs *inside* a PyMOL-capable interpreter (which is often not the interpreter
running ipro-agent -- see ``ipro_agent.rotamers`` for the dispatch). Reads a
job file and writes a rotamer table as JSON.

    pymol-python harvest_rotamers.py jobs.json out.json

Job file::

    {"pdb": "/path/in.pdb",
     "jobs": [{"chain": "A", "number": 115, "icode": "", "allowed": ["TRP","PHE"]}]}

Output::

    {"pdb": ..., "pymol_version": ...,
     "rotamers": {"A115": {"TRP": [{"state": 1, "strain": 65.25,
                                    "atoms": [["N", [x,y,z]], ...]}, ...]}},
     "errors": {"A115": {"PRO": "reason"}}}

Why PyMOL and not a chi-angle table: PyMOL's mutagenesis wizard already ships a
rotamer library and builds ideal side-chain geometry for each rotamer, giving
coordinates directly. That is both the dependency the user asked for and a way
to avoid transcribing a rotamer library by hand. Backbone atoms come out
identical across states; only the side chain moves (verified -- see
docs/DESIGN.md).
"""

from __future__ import annotations

import json
import sys
from typing import Any

# Backbone atoms are shared by every rotamer at a position; the side chain is
# what the MILP chooses. CB is listed as side chain (GLY has none). OT1/OT2/OT
# are the CHARMM/AMBER spellings of the C-terminal carboxylate.
BACKBONE = {"N", "CA", "C", "O", "OXT", "OT1", "OT2", "OT"}


def _is_hydrogen(atom) -> bool:
    """True for a hydrogen, by element where PyMOL gives one, else by name.

    Rotamers must be heavy-atom-only, because every template the energy model
    builds is heavy-atom-only (``rotamers.build_template`` drops H). Harvesting
    hydrogens from a *protonated* input broke that symmetry: the side chain
    carried its own backbone H and HA, which the template no longer held, so
    every rotamer at a position gained a near-constant +18 to +22 kcal/mol.
    Measured on a protonated 2omf_A with the shipped defaults, that offset
    pushed 94 of 160 rotamers past ``max_self_energy`` instead of 46, and
    deleted TRP, PHE, TYR and PRO from position A18 outright -- so the
    published V18W answer was not in the design space, with ``validate``
    reporting 6561 sequences and no error anywhere. The side-chain hydrogens
    additionally re-ranked the conformers.

    PyMOL's chempy atoms carry ``symbol``; fall back to the PDB naming rule
    (a leading digit then H, as in ``1HB``, or a leading H).
    """
    sym = (getattr(atom, "symbol", "") or "").strip().upper()
    if sym:
        return sym in ("H", "D", "T")
    name = (getattr(atom, "name", "") or "").strip().upper().lstrip("0123456789")
    return name[:1] in ("H", "D")

#: The wizard is always handed a *named* selection, never a selection
#: expression, because ``do_select`` finishes with ``cmd.delete(selection)``.
#: Given the expression ``prot and chain A and resi 16A`` that call matches and
#: deletes the object ``prot`` itself, leaving nothing to harvest (verified on
#: PyMOL 3.1.0: 327 atoms to 0). With a name, the same call removes only this
#: temporary selection.
POS_SELE = "_ipro_harvest_pos"


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


def harvest(job_path: str, out_path: str) -> int:
    with open(job_path) as fh:
        job = json.load(fh)

    import pymol
    from pymol import cmd

    pymol.finish_launching(["pymol", "-cq"])
    # Quieten per-rotamer chatter; we read strain from the wizard instead.
    try:
        cmd.feedback("disable", "executive", "actions")
    except Exception:
        pass

    cmd.load(job["pdb"], "prot")
    result: dict[str, Any] = {
        "pdb": job["pdb"],
        "pymol_version": str(cmd.get_version()[0]),
        "rotamers": {},
        "errors": {},
    }

    for spec in job["jobs"]:
        chain = str(spec["chain"])
        number = int(spec["number"])
        icode = str(spec.get("icode", "") or "")
        label = f"{chain}{number}{icode}"
        sel = _selection(chain, number, icode)

        # Native identity, for reporting mutations as D115W.
        try:
            native = cmd.get_model(f"({sel}) and name CA").atom[0].resn
        except Exception as exc:
            result["errors"].setdefault(label, {})["_position"] = (
                f"residue not found in structure ({exc})"
            )
            continue

        per_pos: dict[str, list[dict[str, Any]]] = {}
        for aa in spec["allowed"]:
            aa = aa.upper()
            try:
                # A fresh wizard per (position, residue): the wizard keeps state
                # and reusing one across positions silently mis-selects.
                cmd.wizard("mutagenesis")
                w = cmd.get_wizard()
                cmd.refresh_wizard()
                w.set_mode(aa)
                cmd.select(POS_SELE, sel)
                w.do_select(POS_SELE)
                cmd.refresh_wizard()

                n_states = cmd.count_states("mutation")
                if n_states < 1:
                    raise RuntimeError("PyMOL offered no rotamers")

                # PyMOL's wizard exposes per-rotamer strain as bump_scores in
                # recent versions; treat as optional.
                strains = getattr(w, "bump_scores", None) or []

                rots: list[dict[str, Any]] = []
                for s in range(1, n_states + 1):
                    model = cmd.get_model("mutation", state=s)
                    atoms = [
                        [a.name, [round(float(c), 4) for c in a.coord]]
                        for a in model.atom
                        if a.name not in BACKBONE and not _is_hydrogen(a)
                    ]
                    if not atoms and aa != "GLY":
                        # No side-chain atoms for a non-GLY residue means the
                        # wizard did not build it; better to report than to
                        # silently contribute an empty rotamer to the MILP.
                        raise RuntimeError(f"state {s} produced no side-chain atoms")
                    rec: dict[str, Any] = {"state": s, "atoms": atoms}
                    if s - 1 < len(strains):
                        try:
                            rec["strain"] = round(float(strains[s - 1]), 4)
                        except (TypeError, ValueError):
                            pass
                    rots.append(rec)
                per_pos[aa] = rots
            except Exception as exc:  # one bad residue type must not kill the run
                result["errors"].setdefault(label, {})[aa] = f"{type(exc).__name__}: {exc}"
            finally:
                try:
                    cmd.set_wizard()
                    cmd.delete("mutation")
                    cmd.delete(POS_SELE)
                except Exception:
                    pass

        result["rotamers"][label] = per_pos
        result.setdefault("native", {})[label] = native

    with open(out_path, "w") as fh:
        json.dump(result, fh)

    n_rot = sum(len(r) for pos in result["rotamers"].values() for r in pos.values())
    n_err = sum(len(v) for v in result["errors"].values())
    print(
        f"harvested {n_rot} rotamers at {len(result['rotamers'])} positions"
        + (f" ({n_err} residue-type errors)" if n_err else ""),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(harvest(sys.argv[1], sys.argv[2]))
