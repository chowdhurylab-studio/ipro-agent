# Installing ipro-agent's dependencies

**Versions and install routes in this file were checked on 2026-11-17** against the
upstream package indexes (conda-forge, PyPI, GitHub releases) and, where noted,
against this machine (macOS 26.3.1, darwin 25.3.0, Apple Silicon). Package
indexes move; re-check anything that fails rather than assuming this file is
current. Items I could not verify are marked **[unverified]**.

ipro-agent is a clean-room reimplementation of IPRO (Iterative Protein Redesign
and Optimization, Maranas group). The original suite needs CHARMM (licensed),
GAMS or CPLEX, and prebuilt Linux x86-64 C++ binaries, and it is Python 2. None
of that is required here; the substitutions are:

| IPRO's original component | here |
|---|---|
| CHARMM (mutation, minimization, energy) | PyMOL mutagenesis wizard + rotamer library; PyRosetta for relax/score; built-in screening function as fallback |
| CPLEX / GAMS (rotamer-selection MILP) | CPLEX-LP file + any solver (CPLEX, Gurobi, CBC, GLPK); built-in heuristic as fallback |
| prebuilt Linux x86-64 C++ docking binaries | AutoDock Vina |
| Python 2 driver scripts | Python >= 3.10 package |

Nothing here is redistributed with the repo. PyRosetta in particular is
licensed, not vendored (see [PyRosetta](#pyrosetta)).

---

## 1. Dependency tiers

`scripts/doctor.py` is the authoritative probe; this table says what each probe
buys and what its absence costs.

### Required

| Package | Why | If absent |
|---|---|---|
| Python >= 3.10 | the package itself | — |
| `numpy` | all geometry and energy arrays | `doctor.py` reports `can_run: False`; nothing runs |
| `biopython` (`Bio`) | the only PDB/mmCIF reader (`ipro_agent/structure.py`) | `StructureError: Biopython is required` on any structure load |
| `pyyaml` | **only** to read a YAML design spec | JSON specs still work; a YAML spec raises a message telling you to install PyYAML |

`scipy` is probed by `doctor.py` for information only — nothing imports it
(`structure.py` does its own broadcasting instead of `scipy.spatial.cdist`).

### Recommended, and what each unlocks

| Backend | Unlocks | Degradation if absent |
|---|---|---|
| **PyMOL** (`pymol-open-source`) | both halves of the mutation contract: rotamer enumeration at every design position (`ipro_agent/rotamers.py` -> `scripts/harvest_rotamers.py`) and committing the chosen design to coordinates (`ipro_agent/mutate.py` -> `scripts/apply_design.py`), using PyMOL's mutagenesis wizard and its rotamer library, with per-rotamer strain from the wizard's `bump_scores` | **Effectively fatal.** `ipro_agent.rotamers.harvest()` raises `RotamerError`. `backends.mutate: builtin` is accepted by the config and `doctor.py` prints a `builtin (rotamer-library sidechain swap)` label, but no built-in rotamer builder exists in the tree — PyMOL is the only rotamer source. It is also not substitutable after the fact: a rotamer is identified downstream by a bare integer index into *PyMOL's* enumeration for that residue, so rebuilding the side chain with another library would write coordinates the MILP never scored. Treat PyMOL as required. |
| **PyRosetta** | (a) backbone relaxation, (b) whole-structure and interface scoring in Rosetta Energy Units — and with it the **full** IPRO protocol: backbone perturbation + restrained minimization, then whole-complex relaxation | No relaxation at all: `EnergyBackend.relax()` raises `NotImplementedError` and `can_relax` stays `False`, so `loop.py` sets `mode = "fixed-backbone"` (`self.mode = "full" if self.backend.can_relax else "fixed-backbone"`) and runs sequence + side-chain packing over a frozen backbone, dropping the perturbation and relaxation steps. That mode is recorded in `state.json` as `mode: fixed-backbone`; it is a real protocol but **not** the published IPRO one, and the file says so rather than letting the two be confused. Energies then come from `ipro_agent/energy/builtin.py` (`builtin-screen`): Amber parm94/99 Lennard-Jones 12-6 at the **element** level, capped repulsion, Coulomb on formally charged side-chain atoms with a distance-dependent dielectric eps(r) = 4r, and a distance-only H-bond well. Units are `kcal/mol (screening scale, not comparable to Rosetta energy units)`. A screening function, not a publication-grade force field. |
| **AutoDock Vina** | rigid-body re-docking of target molecules (`loop.redock_every`, per-molecule `redock: true`, `backends.dock`) | `doctor.py` resolves `dock -> none (input poses kept; set redock:false)`. Nothing is re-docked; input poses are carried through unchanged for the whole run. Set `backends.dock: none` and `redock: false` so the spec records that honestly. **Dated caveat (file state 2026-11-17):** no module in the tree consumes `backends.dock` yet — `grep -rn "backends.dock" ipro_agent` returns nothing, and the redock step is listed in `loop.py`'s docstring but not implemented — so installing Vina currently only changes what the doctor reports. Re-run that grep before assuming the step is live. |
| **A MILP solver** (CPLEX, Gurobi, CBC, GLPK) | a *proven optimal* rotamer assignment for the emitted integer program, with your linear constraints enforced | `MILP.solve()` falls through to `_solve_greedy()`: iterated best-improvement over single-position reassignments. It respects one-rotamer-per-position by construction and checks user constraints, but `proven_optimal=False` and `status="heuristic"`. Quality is a lower bound, not the optimum. |
| **Ligand prep** (Meeko, or Open Babel) | PDBQT receptor/ligand inputs for Vina | Only matters if Vina is installed; without prep tools you must supply PDBQT yourself. |

Even with PyRosetta installed, the MILP's self/pair energy tables are built from
the decomposable screening terms by default (`PyRosettaBackend(exact_pairs=False)`),
because Rosetta scores poses, not arbitrary atom sets. PyRosetta is used for
relaxation and for scoring whole structures. `exact_pairs=True` forces
Rosetta-scored pairs and is slow; only worth it for tiny design spaces.

---

## 2. PyMOL

Two different products share the name:

| | open-source build | Schrödinger build |
|---|---|---|
| package | conda-forge `pymol-open-source` **3.1.0** (builds for py39–py314; `linux-64`, `linux-aarch64`, `linux-ppc64le`, `osx-64`, `osx-arm64`, `win-64`) | installers / `pymol-bundle` from pymol.org, currently **3.1.9** (updated 2026-10-07) |
| cost | free | paid licence (licence file required) |
| mutagenesis wizard | **yes** — verified on this machine: `.../pymol_env/lib/python3.9/site-packages/pymol/wizard/mutagenesis.py`, `pymol-open-source 3.1.0 py39hddd9dd7_0` | present in the same codebase, but I did not verify it in a licensed build **[unverified]** |

Use the open-source build. ipro-agent only ever drives PyMOL headless
(`pymol -cq`) through `scripts/harvest_rotamers.py`, which calls
`cmd.wizard("mutagenesis")`; nothing needs the GUI or a licence file.

```bash
conda create -n pymol_env -c conda-forge pymol-open-source
```

### Pointing ipro-agent at a PyMOL in another environment

PyMOL usually lives in its own conda env with its own Python, so ipro-agent does
not import it — `ipro_agent/rotamers.py` dispatches a subprocess. The PyMOL
interpreter therefore does **not** have to match the one running ipro-agent
(here: PyMOL on 3.9.18, ipro-agent on 3.12.12).

`find_pymol_python()` tries, in order:

1. the current interpreter,
2. `$IPRO_PYMOL_PYTHON`,
3. conda envs under `$CONDA_PREFIX/../envs`, `~/miniconda3/envs`, `~/anaconda3/envs`, `/opt/miniconda3/envs`, `/opt/anaconda3/envs` — env names containing `pymol` are probed first,
4. a `pymol` launcher on `PATH`.

Pin it rather than relying on discovery:

```bash
export IPRO_PYMOL_PYTHON=/opt/miniconda3/envs/pymol_env/bin/python
```

If that interpreter cannot `import pymol`, you get a logged warning
(`IPRO_PYMOL_PYTHON=... cannot import pymol`) and discovery continues — so a
typo degrades silently into whatever else is found. Verify:

```bash
"$IPRO_PYMOL_PYTHON" -c "import pymol; print(pymol.__file__)"
"$IPRO_PYMOL_PYTHON" -c "import pymol, sys; sys.path; from pymol import cmd; print('wizard ok')"
python3 scripts/doctor.py | grep -i pymol
```

The harvest subprocess runs with `-I` (isolated) and `PYTHONNOUSERSITE=1`, so a
stray module in the input PDB's directory or in your user site-packages cannot
shadow PyMOL's. Timeout is 3600 s for the whole harvest.

---

## 3. PyRosetta

**Licence first.** PyRosetta is distributed under the *PyRosetta Software
Non-Commercial License Agreement*
(<https://github.com/RosettaCommons/rosetta/blob/main/LICENSE.PyRosetta.md>).
Free of charge for "Non-Commercial Users" — employees of not-for-profit research
institutions, government laboratories and universities — excluding commercial
service and contract research whose IP is owned by a for-profit company. The
agreement states that use of PyRosetta "requires a valid concurrent
non-commercial license of Rosetta", and obliges you to refer other
non-commercial users to GitHub to download it themselves. Commercial use
requires a paid licence from the University of Washington CoMotion office
(<https://els2.comotion.uw.edu/product/pyrosetta>, `license@uw.edu`).

Consequences for this repo: **do not vendor PyRosetta, commit its wheels, or
bake it into an image you share.** Each user installs it under their own
licence. The repo's test environment deliberately has no PyRosetta, which is why
`ipro_agent/energy/pyrosetta_backend.py` is written against the documented API
and exercised only when it resolves.

Install routes, as published on <https://www.pyrosetta.org/downloads>:

```bash
# (a) quarterly release via pip -- reproducible; use this for work you will publish
pip install pyrosetta --find-links https://west.rosettacommons.org/pyrosetta/quarterly/release
# east-coast mirror:
# pip install pyrosetta --find-links https://graylab.jhu.edu/download/PyRosetta4/archive/release-quarterly/release

# (b) weekly "devel" builds via the pip helper (pyrosetta-installer 0.1.2)
pip install pyrosetta-installer
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'

# (c) conda channel (weekly builds)
conda install -y -c https://conda.rosettacommons.org -c conda-forge pyrosetta
```

Which is current: **all three are**, and the quarterly pip channel is the one
the downloads page marks recommended. Prefer (a): weekly `devel` builds are not
retained indefinitely, so they cannot anchor a reproducible result. The
`pyrosetta-installer` helper fetches the weekly build, not the quarterly one.

The quarterly index today lists `2026.29+releasequarterly.80a0635615` with
wheels for cp39–cp314 on `linux_x86_64`, `macosx_12_0_arm64` and
`macosx_12_0_x86_64` — i.e. Apple Silicon is a first-class target, no Rosetta 2
needed. No Windows wheels.

Verify:

```bash
python3 -c "import pyrosetta; print(pyrosetta.__file__)"
python3 scripts/doctor.py | grep -i pyrosetta
```

Whether the download still requires a username/password: the current downloads
page does not mention one, and these commands are password-free as published,
but I did not run them here **[unverified]**.

---

## 4. AutoDock Vina

Current release: **v1.2.7**, published 2025-02-26
(<https://github.com/ccsb-scripps/AutoDock-Vina/releases>). Assets are prebuilt,
per-platform executables, including a **native Apple Silicon** build:

| asset | platform |
|---|---|
| `vina_1.2.7_mac_aarch64`, `vina_split_1.2.7_mac_aarch64` | macOS arm64 (native) |
| `vina_1.2.7_mac_x86_64` | macOS Intel (or arm64 under Rosetta 2) |
| `vina_1.2.7_linux_x86_64`, `vina_1.2.7_linux_aarch64` | Linux |
| `vina_1.2.7_win.exe` | Windows |

### Easiest route: conda-forge (CLI + Python bindings in one package)

```bash
conda install -c conda-forge vina        # 1.2.7; linux-64, linux-aarch64, linux-ppc64le, osx-64, osx-arm64
```

The conda-forge feedstock's own test runs `vina --help`, `vina_split --help` and
`import vina`, so this single package provides both the executables and the
Python bindings.

### GitHub release binary

`doctor.py` looks for `vina`, `vina_1.2.5` or `vina.exe` on `PATH` — so install
the 1.2.7 binary **under the name `vina`**, or it will not be found:

```bash
# macOS, Apple Silicon
curl -L -o ~/bin/vina https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_mac_aarch64
# Linux x86-64:
# curl -L -o ~/bin/vina https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_linux_x86_64
chmod +x ~/bin/vina
xattr -dr com.apple.quarantine ~/bin/vina 2>/dev/null   # macOS only; see troubleshooting
vina --version
```

### Python bindings via pip: expect a source build

PyPI `vina` 1.2.7 ships wheels only for `manylinux_2_17_x86_64` and
`musllinux_1_2_x86_64`, plus an sdist. There is **no macOS wheel and no
Linux aarch64 wheel**, so `pip install vina` compiles from source on those
platforms and needs Boost and SWIG:

```bash
brew install boost swig      # macOS
pip install vina
```

Known pain, and why conda-forge is the recommendation: the conda-forge recipe
pins `libboost-devel <=1.86` and carries a patch, i.e. the sdist is not known to
build against newer Boost. If `pip install vina` fails on a Boost header or a
SWIG invocation, install the conda package instead of fighting it. ipro-agent
only needs the `vina` executable; the Python bindings are an alternative the
doctor accepts, not a requirement.

### Receptor / ligand preparation

`doctor.py` accepts any of `obabel`, `mk_prepare_ligand.py`,
`prepare_ligand4.py`, `prepare_receptor4.py`, or an importable `meeko`.

```bash
# preferred: Meeko 0.8.0 (Forli lab, developed alongside Vina)
conda install -c conda-forge meeko      # or: pip install meeko
mk_prepare_ligand.py  -i ligand.sdf -o ligand.pdbqt
mk_prepare_receptor.py -i receptor.cif -o receptor -j -p      # writes receptor.pdbqt + receptor.json
mk_export.py vina_out.pdbqt -j receptor.json -s docked.sdf -p receptor_docked.pdb

# fallback: Open Babel (verified here: `obabel -L formats` lists "pdbqt -- AutoDock PDBQT format")
conda install -c conda-forge openbabel  # or: brew install open-babel
obabel ligand.sdf -O ligand.pdbqt -p 7.4 --partialcharge gasteiger
obabel receptor.pdb -O receptor.pdbqt -xr      # -xr = rigid receptor
```

Open Babel PDBQT output is adequate for screening but does not do Meeko's
typing, macrocycle handling or flexible-side-chain bookkeeping; prefer Meeko
when the docking result matters. `prepare_ligand4.py` / `prepare_receptor4.py`
come from AutoDockTools / the ADFR suite; `doctor.py` will use them if present,
but I did not test an ADFR install on Apple Silicon **[unverified]**.

### Do not bother with Homebrew for Vina

`brew info autodock-vina` fails on this machine (Homebrew 6.0.21):

```
Error: Refusing to load formula brewsci/bio/autodock-vina from untrusted tap brewsci/bio.
Run `brew trust --formula brewsci/bio/autodock-vina` or `brew trust brewsci/bio` to trust it.
```

Homebrew 6.0.0 (2026-06-11) made non-official taps opt-in: loading a third-party
tap's formula runs its Ruby with your privileges, so it must be trusted first.
`brewsci/bio` is third-party. You *can* trust it —
`brew trust brewsci/bio` (recorded in `~/.homebrew/trust.json`, or
`$XDG_CONFIG_HOME/homebrew/trust.json`) — but there is no reason to: use
`conda install -c conda-forge vina` or the official GitHub release binary above.
Both avoid the tap entirely. (`HOMEBREW_NO_REQUIRE_TAP_TRUST` is deprecated;
don't reach for it.)

---

## 5. MILP solver

`ipro_agent/milp.py` emits the rotamer-selection program as a **CPLEX-LP file**
(`MILP.to_lp()`, commented, written to `lp_path` so the formulation you actually
solved is an artifact you can read). `solve(solver="auto")` then tries, in order:

1. **PuLP**, with the first of `CPLEX_CMD`, `GUROBI_CMD`, `PULP_CBC_CMD`, `COIN_CMD`, `GLPK_CMD` that `pulp.listSolvers(onlyAvailable=True)` reports;
2. **a solver binary on the LP file** — `glpsol` if present, else `cbc`;
3. **the built-in greedy heuristic**.

Preference order to install, best first:

| Solver | Route | Notes |
|---|---|---|
| CPLEX | IBM Academic Initiative (<https://academic.ibm.com>): free, no model-size limit, registration + eligibility check | IPRO's original solver. `doctor.py` probes a `cplex` binary. Users report portal friction (Download Director vs HTTP, SkillsBuild redirects); I did not verify the current download path **[unverified]** |
| Gurobi | free academic named-user licence (<https://www.gurobi.com/academia/academic-program-and-licenses/>) | faculty/staff/students at degree-granting institutions, research and teaching only, one user + one machine, not usable in containers (use WLS there). `doctor.py` probes `gurobi_cl`, so install the full distribution, not just `gurobipy`. Licence term is stated inconsistently across Gurobi's own pages (6 months vs 1 year) **[unverified]** |
| CBC | `conda install -c conda-forge coin-or-cbc` (2.10.13), or `pip install 'pulp[cbc]'` | gives a `cbc` on `PATH`, which satisfies both PuLP's `COIN_CMD` and milp.py's binary path |
| HiGHS | `conda install -c conda-forge highs` (1.16.0) / `pip install 'pulp[highs]'` | **see the gap below** |
| GLPK | `conda install -c conda-forge glpk` (5.0) / `brew install glpk` | `glpsol`. Verified here: GLPK 5.0 parses the LP dialect `to_lp()` emits and reports `Status: INTEGER OPTIMAL`, which is what milp.py keys `proven_optimal` on |
| (none) | — | greedy heuristic; `proven_optimal=False` |

**PuLP no longer bundles CBC.** PuLP 4.2.0 (2026-10-22, requires Python >= 3.12)
removed the bundled binary and the `PULP_CBC_CMD` API; the project's own
description says so explicitly. So:

```bash
pip install 'pulp[cbc]'     # pulls cbcbox, which installs a `cbc` executable into the env's bin/
```

On Python 3.10–3.11 pip resolves to PuLP 3.3.2 (requires >= 3.10), which still
ships CBC as `PULP_CBC_CMD`. Either way the result is a working open MIP solver;
just do not expect a bare `pip install pulp` on 3.12+ to bring a solver with it.
Other PuLP extras: `[highs]` -> `highspy`, `[highs-cmd]` -> `highsbox`,
`[glpk]` -> `swiglpk`, `[gurobi]` -> `gurobipy`, `[scip]` -> `pyscipopt`.

**Known gap — HiGHS alone is not enough.** `doctor.py` lists `HiGHS` and
`HiGHS_CMD` in its preference display, but `milp.py` never selects them:
`_solve_pulp()` checks only `CPLEX_CMD, GUROBI_CMD, PULP_CBC_CMD, COIN_CMD,
GLPK_CMD`, and `_solve_binary()` knows only `glpsol` and `cbc`. A HiGHS-only
install therefore makes the doctor report `milp -> HiGHS` while an actual run
falls through to the greedy heuristic. Install CBC or GLPK alongside it.

A solver is strongly recommended, not optional-in-spirit: the point of the
integer program here is that *your* linear constraints ("at most M mutations
this iteration", "no more than k charged residues at these positions") are
enforced and the assignment is provably optimal under the energy tables. The
heuristic checks the constraints but proves nothing, and `MILPSolution.solver`
will say so: `greedy (built-in heuristic; optimality NOT proven)`.

---

## 6. Recipes

There is no `pyproject.toml` in the repo yet, so these install dependencies and
run from the repo root (`python3 scripts/doctor.py`, `import ipro_agent` with the
repo root as cwd or on `PYTHONPATH`).

### (a) Minimal — runs with every fallback engaged, no licences

```bash
conda create -y -n ipro-min -c conda-forge python=3.12 numpy biopython pyyaml
conda activate ipro-min
conda install -y -c conda-forge pymol-open-source glpk   # PyMOL is effectively required; glpk gives a real solver
cd /path/to/ipro-agent
python3 scripts/doctor.py
# expect: pymol OK, pyrosetta --, vina --, milp -> glpk
# energy = builtin screening function, no backbone relaxation, no re-docking
```

### (b) Full open stack — PyMOL + Vina + CBC, still no licences

```bash
conda create -y -n ipro -c conda-forge \
  python=3.12 numpy biopython pyyaml scipy \
  pymol-open-source vina meeko openbabel \
  coin-or-cbc glpk pulp
conda activate ipro
cd /path/to/ipro-agent
python3 scripts/doctor.py --require pymol vina milp
# expect: pymol OK, vina OK, ligand prep = obabel, meeko; milp -> COIN_CMD (or PULP_CBC_CMD)
```

### (c) Full, with PyRosetta (academic licence required)

```bash
conda create -y -n ipro-rosetta -c conda-forge \
  python=3.12 numpy biopython pyyaml scipy \
  pymol-open-source vina meeko openbabel coin-or-cbc glpk pulp
conda activate ipro-rosetta
# PyRosetta: read and accept the non-commercial licence first (section 3). Do not redistribute.
pip install pyrosetta --find-links https://west.rosettacommons.org/pyrosetta/quarterly/release
cd /path/to/ipro-agent
python3 scripts/doctor.py --require pymol pyrosetta vina milp
# expect: energy -> pyrosetta (ref2015), relaxation available, dock -> vina
```

If PyMOL and PyRosetta will not co-resolve in one environment (different Python
floors, conflicting Qt/boost pins), keep PyMOL in its own env and point at it —
that is exactly what the subprocess dispatch is for:

```bash
export IPRO_PYMOL_PYTHON=/opt/miniconda3/envs/pymol_env/bin/python
```

### `environment.yml`

Save as `environment.yml`, then `conda env create -f environment.yml`. PyRosetta
is deliberately left commented out: it is licensed per user, so putting it in a
shared env file invites redistribution.

```yaml
name: ipro
channels:
  - conda-forge
dependencies:
  - python=3.12
  # --- required ---
  - numpy
  - biopython
  - pyyaml
  # --- recommended backends ---
  - pymol-open-source=3.1.0      # mutagenesis wizard = rotamers
  - vina=1.2.7                   # CLI + python bindings
  - meeko                        # mk_prepare_ligand.py / mk_prepare_receptor.py
  - openbabel                    # obabel, fallback ligand prep
  - coin-or-cbc                  # cbc: MILP
  - glpk                         # glpsol: MILP (LP-file path)
  - pulp                         # modelling layer milp.py tries first
  # --- optional ---
  - scipy                        # probed by doctor.py; not imported by the package
  - pip
  # - pip:
  #   # PyRosetta: academic/non-commercial licence required, not redistributable.
  #   # Install it yourself, under your own licence:
  #   #   pip install pyrosetta --find-links https://west.rosettacommons.org/pyrosetta/quarterly/release
```

---

## 7. Verification

```bash
python3 scripts/doctor.py            # human-readable
python3 scripts/doctor.py --json     # machine-readable
python3 scripts/doctor.py --require pymol vina milp    # exit 1 unless all present
```

Exit status is 0 if the loop can run at all (numpy present), 1 if even that is
broken or a `--require` name is missing. `--require` accepts `pymol`,
`pyrosetta`, `vina`, `milp` (and any other key under `checks`). Use it to gate a
run whose result you intend to report: `--require pyrosetta` is how you assert
"this run did real relaxation".

### Worked example: this machine, as measured 2026-11-17

macOS 26.3.1 (darwin 25.3.0), Apple Silicon; conda at `/opt/miniconda3`, base
Python 3.12.12; PyMOL 3.1.0 only inside the `pymol_env` conda env (Python
3.9.18); Homebrew `glpk` 5.0 and `open-babel`; no PyRosetta, no Vina, no
CPLEX/Gurobi, no PuLP.

```
ipro-agent doctor  (python 3.12.12)
  interpreter: /opt/miniconda3/bin/python3

backends
  OK  PyMOL (mutagenesis)  [subprocess-env] found in pymol_env
  --  PyRosetta (scoring + relax)  (not found)
  --  AutoDock Vina (docking)  (not found)
  OK  ligand prep: obabel
  OK  MILP: pulp=False solvers=[] binaries=['glpk']

what a run will use
  mutate   -> pymol
  energy   -> builtin (LJ + Coulomb + implicit-solvent approximation)
  dock     -> none (input poses kept; set redock:false)
  milp     -> glpk (via LP file)

missing (run is still possible, with the fallbacks above):
  pyrosetta, vina
  see docs/INSTALL.md for what each one changes
```

Read that as: rotamers are real (PyMOL found in another env and driven by
subprocess — `[subprocess-env]` is the normal, healthy case on macOS, not a
warning); the MILP is solved to proven optimality by `glpsol` via the LP file;
energies are the screening function and there is **no backbone relaxation**, so
the loop will run in `fixed-backbone` mode and write `mode: fixed-backbone` into
`state.json`; and target poses are whatever the input file contained.
Good output is `OK` on PyMOL and on MILP with a named solver — `milp -> greedy
(built-in; NOT a proof of optimality)` is the line to fix before trusting a
design.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `import pymol` works but `cmd.wizard("mutagenesis")` fails / no rotamers come back | PyMOL installed without the wizard package, or a stub/partial install (e.g. `pymol` from a non-conda-forge source, or only `pymol-open-source`'s python shim) | Check the wizard file exists: `ls $(dirname $(python -c "import pymol,os;print(pymol.__file__)"))/wizard/mutagenesis.py`. If absent, reinstall from conda-forge: `conda install -c conda-forge pymol-open-source` |
| `RotamerError: no PyMOL available` / `find_pymol_python()` returns nothing | no env on the search path can `import pymol`; or `$IPRO_PYMOL_PYTHON` points at the wrong interpreter (a logged warning, then discovery continues); or PyMOL lives somewhere not scanned (venv, `/usr/local`, a non-standard conda root) | `export IPRO_PYMOL_PYTHON=/path/to/env/bin/python` and confirm `"$IPRO_PYMOL_PYTHON" -c "import pymol"` exits 0. Note the probe runs with `PYTHONNOUSERSITE=1`, so a user-site PyMOL is invisible on purpose |
| `rotamer harvest produced no output (rc=...)` | the harvest subprocess died — usually a PyMOL that cannot start headless, or a PDB it refuses | Read the `stderr tail` in the exception, then run it by hand: `"$IPRO_PYMOL_PYTHON" -I scripts/harvest_rotamers.py jobs.json out.json` |
| `ImportError: No module named pyrosetta`, or `energy backend 'pyrosetta' requested but unavailable` | not installed, or installed in a different env | Install per section 3 and check `python3 -c "import pyrosetta"`. With `backends.energy: auto` this is a silent downgrade to the screening function — use `backends.energy: pyrosetta` to make it a hard error, or gate with `doctor.py --require pyrosetta` |
| PyRosetta import succeeds but a licence/EULA error appears, or `pip` cannot find a wheel | wrong find-links index for your Python/platform, or an expired weekly `devel` build that has been removed from the server | Use the quarterly index (section 3); wheels exist for cp39–cp314 on `linux_x86_64`, `macosx_12_0_arm64`, `macosx_12_0_x86_64`. Commercial use needs a paid UW CoMotion licence — that is a licence problem, not an install problem |
| macOS: `vina` is "damaged", "cannot be opened", or is killed immediately | Gatekeeper. Browser downloads get a `com.apple.quarantine` xattr, and the release binaries are unsigned/un-notarized (`curl` does not set the flag, Safari/Chrome do) | `chmod +x ./vina && xattr -dr com.apple.quarantine ./vina`, then run it from the shell (not Finder). Or sidestep it: `conda install -c conda-forge vina` |
| `vina --version` works but `doctor.py` still shows Vina missing | the binary is not named `vina` (the doctor probes `vina`, `vina_1.2.5`, `vina.exe`), or it is not on the `PATH` of the interpreter running the doctor | Install it as `vina` on `PATH`; re-check with `which vina` using the same shell |
| `pip install vina` fails compiling (Boost headers, SWIG) | no macOS or Linux-aarch64 wheel on PyPI for 1.2.7, so pip builds the sdist; the conda recipe pins `libboost-devel <=1.86` plus a patch, so newer Boost is not known to work | `conda install -c conda-forge vina`, or install the release binary and skip the Python bindings — ipro-agent only needs the executable |
| `brew info autodock-vina` → `Refusing to load formula ... from untrusted tap brewsci/bio` | Homebrew >= 6.0.0 requires explicit trust for non-official taps | Don't use the tap: `conda install -c conda-forge vina`, or the GitHub release binary. If you insist: `brew trust brewsci/bio` |
| `RuntimeError: no MILP solver succeeded` / `milp -> greedy (built-in; NOT a proof of optimality)` | no CPLEX/Gurobi/CBC/GLPK found, and (on Python >= 3.12) `pip install pulp` no longer brings CBC with it | `pip install 'pulp[cbc]'` or `conda install -c conda-forge coin-or-cbc glpk`; confirm with `python -c "import pulp;print(pulp.listSolvers(onlyAvailable=True))"` and `which cbc glpsol` |
| doctor says `milp -> HiGHS` but runs report the greedy heuristic | `milp.py` does not select HiGHS (see section 5) | Install CBC or GLPK as well |
| `pulp has no available solver` although PuLP imports | PuLP 4.x without the CBC extra and nothing else on `PATH` | `pip install 'pulp[cbc]'`, or put `cbc`/`glpsol` on `PATH` |
| `StructureError: Biopython is required` | `biopython` missing in the interpreter running ipro-agent (easy to install it into the *PyMOL* env by mistake) | `pip install biopython` / `conda install -c conda-forge biopython` in the ipro-agent env; `python3 scripts/doctor.py` prints which interpreter it probed |
| "reading a YAML spec needs PyYAML" | PyYAML absent | `pip install pyyaml`, or hand it the JSON form of the spec |
| Trying to run the original IPRO instead: its binaries will not start on a Mac | the upstream IPRO distribution ships prebuilt **Linux x86-64** C++ binaries, and depends on CHARMM (licensed), GAMS or CPLEX, and Python 2 | Nothing to fix — that platform restriction is the reason this reimplementation exists. The stack here (PyMOL + Vina + PyRosetta + an open MILP solver) is cross-platform, and every piece above has a native macOS arm64 build: `pymol-open-source` (`osx-arm64`), Vina (`vina_1.2.7_mac_aarch64`), PyRosetta (`macosx_12_0_arm64`), CBC/GLPK/HiGHS (`osx-arm64`) |

---

## 9. What I could not verify

- Whether the licensed Schrödinger PyMOL build ships the mutagenesis wizard in the same importable form. The open-source build does (verified on this machine); the paid build was not available to test.
- Whether the PyRosetta download endpoints still require credentials in any circumstance, and whether the published install commands succeed end-to-end — no PyRosetta was installed here. The file listing on the quarterly index was read directly and is reliable.
- Current IBM CPLEX academic download path, and the exact term of a free Gurobi academic licence (Gurobi's own pages state 6 months in one place and 1 year in another).
- ADFR suite / AutoDockTools (`prepare_ligand4.py`, `prepare_receptor4.py`) on Apple Silicon: `doctor.py` probes these names, but no ADFR install was tested.
- Vina's Python bindings on macOS: only the absence of a macOS wheel on PyPI and the conda recipe's Boost pin were verified, not the exact compiler error you will hit.
