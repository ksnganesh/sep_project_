# sep_project: reproduction of *Device-independent quantum digital signatures under realistic conditions*

Fan, Bian, Li, Wen, Yin, Chen, **Phys. Rev. A 114, 032617 (2026)**, DOI 10.1103/tyjx-bczz.

This repository re-implements the paper from its equations. It covers the signing and verification of messages
(Part 1), device-independent key distribution with its finite-size key length (Part 2), and every result figure of
Sec. IV, Figs. 3–7 (Part 3). All Python code lives in `QIQDS_parts1-2/`.

---

## 1. Repository layout

```
sep_project/
├── pyproject.toml          # project + dependencies (used by uv)
├── uv.lock                 # exact, locked versions of every dependency
├── .python-version         # 3.13
├── requirements.in         # same dependencies, for plain pip
├── src/sep_project/        # placeholder left by `uv init`; not used
└── QIQDS_parts1-2/
    ├── diqds/              # the library (installed into the environment as package `diqds`)
    │   ├── part1_messaging/     # Sec. II.B, III, App. A: LFSR-Toeplitz hash, sign/verify, security bounds
    │   ├── part2_distribution/  # Sec. II.A, App. B: DIQKD protocol, finite-size key length Eq. (9)
    │   └── part3_figures/       # Sec. IV, App. C-D: signature-rate sweeps, Refs. [73-75], photonic model
    ├── scripts/            # the three runnable programs
    │   ├── part1_messaging_demo.py
    │   ├── part2_distribution_demo.py
    │   └── part3_figures.py
    ├── tests/              # pytest suites (83 tests)
    ├── results/            # figures and cached numbers written by the scripts
    └── PART1_NOTES.md, PART2_NOTES.md, PART3_NOTES.md   # paper -> code mapping, choices, results
```

---

## 2. Create the environment with uv

### 2.1 Install uv (once per machine)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# or, with any existing Python
pip install uv
```

Check it with `uv --version`.

### 2.2 Get the code and create the environment

```bash
git clone https://github.com/ksnganesh/sep_project.git
cd sep_project
uv sync
```

`uv sync` does everything in one step:

* reads `.python-version` and uses Python **3.13**, downloading it automatically if it is not installed;
* creates a virtual environment in `./.venv`;
* installs the exact versions pinned in `uv.lock`: NumPy, SciPy, Matplotlib, pytest, QuTiP, Qiskit, Qiskit Aer, SeQUeNCe and
  their dependencies (about 130 packages);
* installs this project's library `diqds` in editable mode, so changes to the code take effect without reinstalling.

The first run downloads the packages (a few minutes). After that it takes a few seconds.

### 2.3 Use the environment

You do **not** need to activate anything. Prefix commands with `uv run` and uv uses `.venv` (and re-syncs it if
`pyproject.toml` changed):

```bash
uv run python --version          # Python 3.13.x
uv run python -c "import diqds, qutip, qiskit_aer; print('environment OK')"
```

If you prefer an activated shell:

```bash
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
python --version                 # now plain `python` / `pytest` use the environment
deactivate                       # leave it
```

All commands below are written with `uv run` from the repository root (`sep_project/`). They also work from inside
`QIQDS_parts1-2/` with the `QIQDS_parts1-2/` prefix dropped, or without `uv run` in an activated shell.

---

## 3. Run every Python file

There are two kinds of Python files:

* **Programs** (`scripts/*.py`) and **tests** (`tests/*.py`): these are meant to be run, as shown below.
* **Library modules** (`diqds/**/*.py`): the programs and tests import these. They have no `main` and do nothing when
  run directly; section 3.4 shows how to check that all of them load.

Times were measured on a 2-core cloud VM with Python 3.13.

### 3.1 Part 1: messaging stage

```bash
uv run python QIQDS_parts1-2/scripts/part1_messaging_demo.py            # ~20 s
uv run python QIQDS_parts1-2/scripts/part1_messaging_demo.py --quick    # ~8 s, fewer Monte-Carlo trials
```

It prints the security budget (n_s = 46 bits per signature segment, ε = 7.84e-11), a full Alice → Bob → Charlie transcript
(Fig. 2), and the forgery / repudiation / robustness attacks with their success rates.
Writes `results/part1_messaging_validation.png` and `results/part1_kat.json` (known-answer test vectors).

### 3.2 Part 2: distribution stage

```bash
uv run python QIQDS_parts1-2/scripts/part2_distribution_demo.py              # ~10 s (uses cached optimisations)
uv run python QIQDS_parts1-2/scripts/part2_distribution_demo.py --quick      # ~8 s, skips the slowest sweeps
uv run python QIQDS_parts1-2/scripts/part2_distribution_demo.py --recompute  # ~3 min, redoes all 48 optimisations
```

It prints Eq. (9) term by term, the minimum visibilities against the paper's printed values, and a full run of the
key-generation protocol on real bit strings whose keys then sign documents with Part 1.
Writes `results/part2_distribution_validation.png` and `results/part2_thresholds.json` (the cache).

### 3.3 Part 3: the paper's figures 3–7

```bash
uv run python QIQDS_parts1-2/scripts/part3_figures.py                    # ~3 s, all five figures from the cache
uv run python QIQDS_parts1-2/scripts/part3_figures.py --figs 3 5         # only some figures (any of 3 4 5 6 7)
uv run python QIQDS_parts1-2/scripts/part3_figures.py --recompute        # ~8 min, redo every optimisation from scratch
uv run python QIQDS_parts1-2/scripts/part3_figures.py --preset rigorous  # alternative finite-size reading (see PART3_NOTES)
```

`--preset` accepts `paper_fit` (default), `rigorous`, `strict` or `loss_only_opt`. A new preset is computed the first time
(several minutes) and cached afterwards. The script prints a JSON table that compares every reproduced number with the
paper.

| Output in `QIQDS_parts1-2/results/` | Paper |
|---|---|
| `fig3_rate_vs_rounds.png` | Fig. 3: signature rate vs number of rounds, V = 1, 0.96, 0.92, 0.88 |
| `fig4_rate_vs_visibility_p.png` | Fig. 4: rate vs visibility at n = 1e10, noisy preprocessing p = 0 … 0.15 |
| `fig5_rate_vs_visibility_n.png` | Fig. 5: rate vs visibility for n = 1e8, 1e9, 1e10, p = 0 and optimised |
| `fig6_matter_experiments.png` | Fig. 6: the ion / atom experiments of Refs. [73–75] |
| `fig7_photonic_distance.png` | Fig. 7: all-photonic scheme with quantum-dot sources, rate vs distance |
| `part3_summary.json` | every reproduced number next to the paper's |
| `part3_curves.json` | cached curve data (delete it or use `--recompute` to regenerate) |

Recompute times per figure: Fig. 3 ≈ 75 s, Fig. 4 ≈ 55 s, Fig. 5 ≈ 85 s, Fig. 6 ≈ 100 s, Fig. 7 ≈ 180 s.

### 3.4 Tests

```bash
uv run pytest -q QIQDS_parts1-2/tests                                   # all 83 tests, ~80 s
uv run pytest -q QIQDS_parts1-2/tests/test_part1_messaging.py           # 36 tests, ~2 s
uv run pytest -q QIQDS_parts1-2/tests/test_part2_distribution.py        # 31 tests, ~35 s
uv run pytest -q QIQDS_parts1-2/tests/test_part3_figures.py             # 16 tests, ~40 s
uv run pytest -q QIQDS_parts1-2/tests -k fig5                           # only tests whose name matches "fig5"
uv run pytest -v QIQDS_parts1-2/tests/test_part3_figures.py             # verbose: one line per test
```

The Part 2 suite includes the QuTiP, Qiskit Aer and SeQUeNCe backends. They are in the environment, so none are skipped.

### 3.5 Library modules (imported, not run)

The modules contain functions only. This command imports all 25 of them, which checks that every file loads and that
all dependencies are present:

```bash
uv run python -c "import importlib, pkgutil, diqds; mods=[m.name for m in pkgutil.walk_packages(diqds.__path__, 'diqds.')]; [importlib.import_module(m) for m in mods]; print(len(mods), 'modules OK')"
```

To use the library interactively:

```bash
uv run python
>>> from diqds.part1_messaging import budget
>>> print(budget())                                   # n_s = 46, eps = 7.84e-11
>>> from diqds.part2_distribution import PAPER_FIT, threshold_visibility
>>> threshold_visibility(10**9, PAPER_FIT, p=0.0)     # 0.8685  (paper: 0.8686)
>>> from diqds.part3_figures import photonic as ph
>>> op = ph.optimise_angles(ph.HeraldedSource.build(ph.IDEAL_SOURCE, L=0.0))
>>> op.chsh, op.qber                                  # (2.631, 0.0348)
```

Module overview:

| Package | Module | Contents |
|---|---|---|
| `part1_messaging` | `bits.py` | bit-string conventions, document encoding |
| | `gf2poly.py` | GF(2)[x] arithmetic, Rabin irreducibility test |
| | `polygen.py` | polynomial generator p = f(Z) (Ref. [32], App. B.1) |
| | `lfsr_toeplitz.py` | LFSR-based Toeplitz hash, App. A (three equivalent implementations) |
| | `keys.py` | key segments {X, Y, Z} and their correlations |
| | `protocol.py` | signing, verification, certification (Fig. 2) |
| | `attacks.py` | forging strategies of Sec. III.A |
| | `security.py` | Eqs. (4)–(8), Table I |
| | `kat.py` | known-answer test vectors |
| `part2_distribution` | `finitekey.py` | entropy bounds (B1–B3), Eq. (9)/(B12) term by term, (B10), (B11) |
| | `optimize.py` | maximises the key length over γ, t, ε split, α, α′, p, β |
| | `presets.py` | named readings of the choices the paper leaves open |
| | `devices.py` | Werner-state devices and the CHSH game |
| | `kgp.py` | protocol steps 1–7 on real bit strings |
| | `privacy.py` | FFT Toeplitz hashing (verification tag, privacy amplification) |
| | `heralding.py` | heralded-round timeline (NumPy and SeQUeNCe) |
| | `quantum_qutip.py`, `quantum_aer.py` | QuTiP and Qiskit Aer versions of the devices |
| `part3_figures` | `sweeps.py` | signature rate vs n or V with the finite-size threshold resolved |
| | `matter.py` | Refs. [73–75] parameters and the KGP of Ref. [73] (Fig. 6) |
| | `photonic.py` | exact multi-photon model of the heralded photonic link (Fig. 1b, App. C–D, Fig. 7) |
| | `plotting.py` | figure styling |

### 3.6 Everything in one go

```bash
uv sync
uv run pytest -q QIQDS_parts1-2/tests
uv run python QIQDS_parts1-2/scripts/part1_messaging_demo.py
uv run python QIQDS_parts1-2/scripts/part2_distribution_demo.py
uv run python QIQDS_parts1-2/scripts/part3_figures.py
```

About 2 minutes in total once the environment exists. Add `--recompute` to the last two to rebuild every cached number
from scratch (about 11 minutes more).

---

## 4. Troubleshooting

| Problem | Fix |
|---|---|
| `uv: command not found` after installing | open a new terminal, or run `source $HOME/.local/bin/env` (Linux/macOS) |
| uv picks the wrong Python | `uv python install 3.13`, then `uv sync --python 3.13` |
| `ModuleNotFoundError: diqds` | run through `uv run …` (or activate `.venv`); `uv sync` installs `diqds` |
| want a fresh environment | delete `.venv` and run `uv sync` again |
| a dependency was added to `pyproject.toml` | `uv lock` then `uv sync` (or `uv add <package>`, which does both) |
| figures look stale after changing code | run the script with `--recompute` (the caches are in `results/*.json`) |
| no uv available | `python3.13 -m venv .venv && source .venv/bin/activate && pip install -r requirements.in && pip install -e .` |

The scripts use Matplotlib's non-interactive `Agg` backend. They only write PNG files, so they also run on servers
without a display.

---

## 5. How close the reproduction is

| Figure | Agreement with the paper (paper values in brackets) |
|---|---|
| Fig. 3 | onsets and rates within 5 % of the published curves |
| Fig. 4 | V_min 0.8621 (0.8622); with optimised p 0.8505 (0.8508) |
| Fig. 5 | all six printed thresholds within 3e-4 |
| Fig. 6 | rates at n = 1e10 within 3 %; onsets ~1.3× later than in the paper |
| Fig. 7 | ideal source 111.2 / 126.3 km (111.46 / 126.53); realistic 17.8 km (fitted) / 29.7 km (28.87) |

The paper leaves some choices open: the free parameter β, which form of Eq. (B11) applies to which figure, the herald
pattern, and how g² enters Eq. (C13). Each one is a switchable option in the code, and `QIQDS_parts1-2/PART*_NOTES.md`
explains it.
