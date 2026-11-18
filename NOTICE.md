# Scope of the MIT licence in this repository

`LICENSE` is the MIT licence, verbatim and unmodified, so that automated
licence detection reports it correctly. It covers **this implementation only**:
the code, tests and documentation written here.

Six things in this tree are deliberately **not** covered by it.

* **No upstream IPRO code is copied here.** This is an *independent*
  reimplementation, written from the published formulations. The released
  upstream source was read for behaviour the papers leave ambiguous -- chiefly
  a handful of numeric defaults -- and every value taken from it is cited as
  coming from the source in docs/DESIGN.md section 2 and docs/MILP.md. It is
  deliberately NOT described as "clean-room": that term means the implementers
  had no access to the original, which is not what happened here, and the
  weaker, accurate claim is the one this licence relies on. See
  docs/DESIGN.md section 6 for the sources and what each supplied.

* **Short quotations from the cited papers remain their publishers'
  copyright** and are not licensed by this file. Where the code or docs quote a
  paper to explain what is being implemented, the quotation is marked as such
  and attributed; the permissions granted above cover this repository's own
  text, not the quoted material.

* **The methods are not ours and are not licensed by this file.** Cite the
  original papers for the method and this repository only for the
  implementation: docs/DESIGN.md section 6 ("Citations") lists them.

* **The papers and reference material under `refs/` are not covered.** Those
  files are third-party copyrighted works retained for provenance checking and
  carry their publishers' own terms. They are not redistributed under the MIT
  licence above.

* **The example structure** `examples/ompf_pore/2omf_A.pdb` is derived from PDB
  entry 2OMF; see `examples/ompf_pore/PROVENANCE.md`.

* **The optional dependencies** (PyMOL, AutoDock Vina, Open Babel, PyRosetta,
  and any MILP solver) are separately licensed and separately installed.
  PyRosetta in particular requires its own licence; see docs/INSTALL.md.

---

See also: `README.md` ("Licence"), `docs/DESIGN.md` §6 for what each source
supplied, and `CITATION.cff` for how to cite the implementation versus the
method.
