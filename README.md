<p align="center">
  <img src="./logo.png" alt="UACPY logo" width="350">
</p>

# 🌊 Underwater Acoustic Propagation for Python 🌊

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+"></a>
  <a href="https://www.gnu.org/licenses/gpl-3.0">  <img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="GPLv3 License"></a>
  <a href="#"><img src="https://img.shields.io/badge/Status-Alpha-orange.svg" alt="Alpha"></a>
</p>

## 🚀 Vision & Motivation

For decades, underwater acoustic propagation models have been
implemented in highly optimized Fortran/C code. For many years, wrapping 
these models in MATLAB was the natural solution adopted by the scientific 
community. As Python has become a dominant language in scientific computing, 
a noticeable gap has emerged. Despite multiple efforts to wrap or re-implement 
these models, Python users still lack a unified, comprehensive, and up-to-date 
solution.

**UACPY is an attempt to close that gap.**\
It was created for researchers, engineers, oceanographers, and acousticians 
who need underwater acoustic modeling to be more open, consistent, 
transparent, and reproducible. It builds on decades of pioneering work in 
the field and aims to provide a shared foundation for comparing models, 
validating results, running experiments, and developing new ideas.

This project began as an AI-assisted (Claude Code with Sonnet 4.5, Opus 4.6 
and 4.7) initiative to reduce early development time, but starting with the 
first release, it will be maintained manually by its author—without autonomous 
AI-driven modifications.

Community feedback, verification, and contributions are warmly 
encouraged. The project’s success depends on collective effort; the 
codebase is far too large and complex for one person to maintain alone 
in their spare time. The goal is for this module to be truly 
community-driven.


> **⚠️ Note:** UACPY is *not* production‑ready. Expect missing features,
> inconsistencies, and the need for validation.


## 🔍 What Is UACPY?

UACPY provides:

-   A unified Python API for major underwater acoustic propagation
    models
-   High‑level tools for configuring environments, sources, receivers,
    media, bathymetry, and boundaries
-   Propagation modeling outputs (TL grids, eigenrays, mode fields, PE
    fields, arrivals, reflection coefficients)
-   Signal processing toolbox (waveform generation, matched filtering,
    beamforming, spectral analysis, correlation)
-   Ambient noise modeling (Wenz curves, wind, shipping, thermal noise)
-   Visualization helpers for rays, TL maps, modes, fields, and
    comparisons
-   Modular architecture that allows adding new models or backends

### Current Goals

-   Provide clean, high‑level Python access to classical models
-   Standardize I/O, parameter structures, and environmental
    descriptions
-   Lower the barrier for experimentation, benchmarking & teaching
-   Promote transparent, repeatable acoustic modeling workflows


## 🧭 Supported Propagation Models

### ✔️ Implemented or In Progress

-   **Bellhop** --- Ray/beam tracing
-   **Kraken** --- Normal modes
-   **Scooter** --- Fast‑field / normal modes
-   **SPARC** --- PE fast field
-   **RAM** (mpiramS) --- Parabolic equation (broadband + TL)
-   **OASES** --- Full suite: OAST (TL), OASN (noise), OASR (reflection), OASP (transfer function)
-   **Bounce** --- Reflection coefficients

## 📦 Installation

Linux is currently the primary supported platform.\
Windows and macOS should work with similar steps, though compilation
requires toolchain adjustments.

### 1. Install dependencies

-   Fortran compiler
-   C/C++ compiler
-   (Optional) CUDA toolkit
-   (Windows) MSYS2 or WSL

### 2. Create a virtual environment

``` bash
python -m venv uacpy_venv
source uacpy_venv/bin/activate
```

### 3. Clone and install

``` bash
git clone https://github.com/ErVuL/uacpy.git
cd uacpy
pip install -e .
./install.sh        # Linux / macOS
# or
install.bat         # Windows
```

The installer compiles OALIB, OASES, BellhopCUDA, and other required
binaries, then places them inside UACPY's internal directory for API
access.

### Uninstall

``` bash
pip uninstall uacpy
rm -rf uacpy
```

## ▶ Simplest example

A minimal "hello world": transmission loss in a 100 m Pekeris waveguide with
Bellhop, at 1000 Hz, out to 5 km.

``` python
import numpy as np
import matplotlib.pyplot as plt

import uacpy
from uacpy.models import Bellhop, RunMode
from uacpy.core.environment import BoundaryProperties
from uacpy.visualization.plots import plot_transmission_loss

# 1. Environment — isovelocity water over a fluid half-space bottom
env = uacpy.Environment(
    name="Pekeris Waveguide",
    depth=100.0,
    sound_speed=1500.0,
    ssp_type='isovelocity',
    bottom=BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1600.0,
        density=1.5,
        attenuation=0.5,
    ),
)

# 2. Source — 1000 Hz, mid water column
source = uacpy.Source(depth=50.0, frequency=1000.0)

# 3. Receiver grid — 200 depths × 5000 ranges out to 5 km
receiver = uacpy.Receiver(
    depths=np.linspace(0, 100, 200),
    ranges=np.linspace(0, 5000, 5000),
)

# 4. Run Bellhop in coherent-TL mode
result = Bellhop(beam_type='B', n_beams=300, alpha=(-80, 80)).run(
    env, source, receiver, run_mode=RunMode.COHERENT_TL,
)

# 5. Plot the TL field
fig, ax = plt.subplots(figsize=(8, 4))
plot_transmission_loss(result, env, ax=ax, show_colorbar=True)
plt.tight_layout()
plt.show()
```

<p align="center">
  <img src="./docs/bellhop_tl_example.png" alt="Bellhop TL field — Pekeris waveguide, 1 kHz" width="720">
</p>

## 📚 Documentation & Examples

The full API reference lives in a single file:
[`DOCUMENTATION.md`](./DOCUMENTATION.md) — quick start, environment setup,
per-model signatures, visualization, signal processing, noise, units, and
troubleshooting.

Inside `uacpy/examples/` you will find 25+ example scripts covering:

-   Transmission loss (TL) computations
-   Ray tracing & eigenray extraction
-   Normal‑mode fields
-   Parabolic‑equation comparisons
-   BellhopCUDA demos
-   Reflection coefficients (Bounce)
-   OASES suite examples
-   Time-domain propagation (SPARC)
-   Signal processing (waveform generation, chirps, filtering)
-   Ambient noise modeling (Wenz curves, wind, shipping)
-   Visualization tools

More examples and tutorials are planned.


## 🧪 Testing

UACPY uses **pytest** with custom markers for categorizing tests.

### Run all tests

``` bash

cd uacpy
pytest uacpy/tests/

```

### Run a specific test file

``` bash

pytest uacpy/tests/test_models.py

```

### Run a single test

``` bash
pytest uacpy/tests/test_models.py::TestClassName::test_method -v

```

### Test markers

Tests use custom markers to allow selective execution:

- `slow` -- Long-running tests (broadband, large grids)
- `requires_binary` -- Tests that need compiled native binaries (Fortran/C)
- `requires_oases` -- Tests that need compiled OASES binaries
- `integration` -- End-to-end integration tests

``` bash

# Skip slow tests

pytest uacpy/tests/ -m "not slow"



# Run only integration tests

pytest uacpy/tests/ -m "integration"



# Run only tests that don't need compiled binaries

pytest uacpy/tests/ -m "not requires_binary"



# Skip OASES tests (if OASES is not installed)

pytest uacpy/tests/ -m "not requires_oases"

```

## 🗺️ Roadmap

Because the initial codebase was bootstrapped with an LLM, *auditing* comes
before new features. The lists below are contributor checklists — please
open an issue or PR for anything you investigate.

### 🛠️ Hardening & validation (priority)

**🧱 Architecture & API audit**
-   Review `PropagationModel` and per‑model overrides for consistency
    (signatures, return types, `_UNSET` usage, `run()` vs `compute_tl()`).
-   Spot‑check the capability matrix in `DOCUMENTATION.md` — it’s a
    claim, not a proof.
-   Audit naming across `core/`, `models/`, `io/`, `visualization/` for
    drifted conventions and inconsistent units (km vs m).
-   Identify over‑engineered abstractions and under‑engineered gaps.

**🔬 Native model re‑validation** — every in‑tree modification is a
potential source of silent numerical drift.
-   **mpiramS (RAM):** rerun the original published test cases against
    an unmodified upstream build and diff the outputs. See
    `uacpy/third_party/MODIFICATIONS.md` for the list of changes
    (OpenMP fix, NaN‑safe init, double‑precision promotion,
    range‑dependent sediment, I/O rewrite).
-   **KrakenField:** confirm the `field.f90` out‑of‑bounds sentinel fix
    doesn’t alter mode amplitudes in previously‑working regimes.
-   **bellhopcuda:** smoke‑test the widened CUDA arch detection
    (`SetupCUDA.cmake`) against a CPU Bellhop run; verify the `tl.cpp`
    SHDFIL field‑width fix round‑trips `.shd` files through both uacpy’s
    `read_shd_bin` and the Matlab `read_shd_bin.m`.
-   **UACPY RAM TL formula**
    (`TL = -20·log10(|psif|·4π) + 10·log10(r)`): validate against
    shallow‑ and deep‑water benchmarks (e.g. ASA 1990).
-   Cross‑model regression: the same environment through Bellhop,
    Kraken (field), Scooter, RAM, and OASES should agree within
    tolerance — build a benchmark suite that fails loudly on drift.

**🐍 Python‑side code review**
-   **Dead code / hallucinated features:** grep for unused functions,
    unreachable branches, and parameters that never make it into the
    generated `.env` / `.flp` / OASES input files.
-   **Doc ↔ code drift:** every signature and default in
    `DOCUMENTATION.md` should match the code.
-   **Error handling:** missing executable, failed subprocess, malformed
    output, NaN TL should raise clean documented exceptions — not bare
    `RuntimeError`.
-   **Security of `subprocess` + file I/O:** audit for command
    injection, path traversal, unbounded reads on large outputs, and
    temp‑file cleanup.
-   **Magic numbers:** trace `cmin`, `cmax`, `n_mesh`, absorbing‑layer
    widths, etc. to a reference or mark as tunable heuristics.

**📊 Visualization review** — `plots.py` / `quickplot.py` / `style.py`
are easy to write plausibly but hard to get correct.
-   Axes, units, orientation: m vs km, Hz vs kHz, depth increasing
    downward on TL/ray plots, colorbars labelled in dB.
-   Colormaps and `tl_min`/`tl_max` clipping: confirm defaults don’t
    hide low‑TL structure across shallow vs deep regimes.
-   Overlays (bathymetry, SSP, markers, layered bottoms, altimetry)
    must share the field’s coordinate frame — off‑by‑one on range
    grids is a classic LLM bug.
-   Ray coloring (`color_by_bounces`) and mode plots: normalization,
    sign convention, and ordering should match what `Kraken` / `OASN`
    return and the metadata wavenumbers.
-   `compare_models`, `compare_range_cuts`, and statistics helpers
    interpolate across grids — confirm disagreements aren’t hidden by
    resampling.
-   Prune unused plot functions; document the `style.py` rcParams
    contract and audit per‑plot overrides that may leak.

**🧪 Test suite audit**
-   Distinguish smoke tests (“does it run?”) from validation tests
    (“does it give the right answer?”) — many are closer to the former.
-   Add reference‑case regressions with fixed tolerances: ASA 1990,
    Pekeris, Munk, Jensen & Kuperman.
-   Audit the `slow` / `requires_binary` / `requires_oases` /
    `integration` markers — a silently skipped test is worse than a
    failing one.
-   Verify every script in `uacpy/examples/` runs end‑to‑end and
    produces a sensible plot.

**📦 Build, install, packaging**
-   Reproduce install on a clean Linux VM, macOS, and WSL — gfortran
    version drift is a common silent‑miscompilation source.
-   Verify `install.sh` and `install.bat` agree on binary names and
    locations.
-   Confirm the OASES download URL and archive hash are still current.
-   Pin a known‑good numpy / scipy / matplotlib combination.

**🔁 CI / CD** — currently none; has to change before tagging a release.
-   Lint (+ optional `mypy`) on every push.
-   Non‑binary tests (`pytest -m "not requires_binary"`) on every push,
    Python 3.8 → 3.13.
-   Full suite nightly / on release: build binaries via `install.sh`
    on a clean runner; this is where Fortran / MPI / CUDA drift hides.
-   Matrix build: Ubuntu, macOS, Windows (WSL at minimum) — `install.sh`
    vs `install.bat` must not diverge.
-   Release automation: on tag, run the full suite → build sdist →
    publish to PyPI.
-   Benchmark regression job: canonical scenarios with TL / arrival
    tolerances; any PR that moves a value beyond tolerance fails loudly.

**🌍 Community & process**
-   Issue template for benchmark deviations (model, scenario, expected
    vs observed, reproducer).
-   Solicit targeted reviews from domain experts per model (rays,
    modes, PE) rather than a single full‑project review — underwater
    acoustics expertise is rarely breadth‑first.

> **If you are evaluating UACPY for a project: do not trust any specific
> number produced by it until at least the re‑validation bullets above
> have been independently verified for the model and regime you care
> about.**

### 🔮 Future scope

**➕ Model‑level improvements**
-   Support for *all* features of each native model
-   GPU acceleration for more models
-   Full 3‑D propagation support (multiple approaches)

**➕ Environmental data integration**
-   Global bathymetry (GEBCO, SRTM)
-   NOAA / IOOS / CMEMS oceanographic fields (temperature, salinity,
    sound speed)
-   On‑the‑fly extraction, caching, and mesh generation

**➕ Framework & tools**
-   Scenario‑based batch simulations
-   Reproducible experiment containers
-   Interactive dashboards for TL / modes visualization


## 🙏 Acknowledgments

UACPY would not exist without decades of prior work by the underwater
acoustics community. Every propagation model shipped here was designed,
implemented, and validated elsewhere --- UACPY only provides a unified
Python interface around them. Each vendored or adapted codebase is
credited below with its origin, what UACPY uses from it, and whether
the source has been modified. When modifications were made, full diffs
are available in [MODIFICATIONS.md](./uacpy/third_party/MODIFICATIONS.md).

### Acoustics Toolbox --- Bellhop, Kraken, KrakenField, Scooter, SPARC, Bounce

Michael B. Porter --- http://oalib.hlsresearch.com/AcousticsToolbox/
- Porter, *The BELLHOP Manual and User's Guide*, 2011
- Porter, *The KRAKEN Normal Mode Program*, 1992

Porter's Acoustics Toolbox provides Bellhop (ray/beam tracing), Kraken
and KrakenField (normal modes and range-dependent mode fields), Scooter
(fast field), SPARC (time-domain PE), and Bounce (reflection
coefficients). UACPY ships the Fortran sources and compiles them
in-tree via `install.sh`.

**Modifications:** one out-of-bounds sentinel fix in
`KrakenField/field.f90`. See MODIFICATIONS.md.

### BellhopCUDA

C. S. Schmid, D. F. Schmidt, A. E. Hodgson --- https://github.com/A-New-BellHope/bellhopcuda
- *BellhopCUDA: High-Performance Acoustical Ray Tracing on GPUs*, 2020

A C++/CUDA port of BELLHOP. UACPY ships the sources and compiles them
in-tree for GPU-accelerated ray tracing.

**Modifications:** (1) CUDA arch detection in
`config/cuda/SetupCUDA.cmake` (widened `CUDA_ARCH_OVERRIDE` validation
range and extended the hardcoded GPU-name table to cover laptop variants
and modern desktop cards); (2) SHDFIL field widths in `src/mode/tl.cpp`
corrected to match the Fortran Acoustics-Toolbox spec (`Sx`, `Sy`, `Rr`,
`theta`, `freqVec`, `freq0`, `atten` written and read as `REAL*8` instead
of upstream's 4-byte values). See MODIFICATIONS.md.

### RAM

Michael D. Collins (Naval Research Laboratory)
- Collins, "A split-step Padé solution for the parabolic equation
  method," *JASA*, 1993

Collins' RAM is the original split-step Padé parabolic-equation
algorithm that underpins UACPY's PE model. UACPY does not ship Collins'
original Fortran; the implementation actually built is Dushaw's
mpiramS (below), which implements the same algorithm.

### mpiramS

Brian D. Dushaw --- https://zenodo.org/records/10818570

mpiramS is Dushaw's MPI-parallel, broadband Fortran implementation of
Collins' RAM algorithm. UACPY ships the Fortran sources and compiles
them in-tree.

**Modifications:** extensive --- OpenMP race-condition fix, NaN-safe
complex initialization, double-precision promotion, configurable
sediment depth points, range-dependent sediment support, multi-range
output, and an I/O rewrite. Full diffs in MODIFICATIONS.md.

### OASES --- OAST, OASN, OASR, OASP

Henrik Schmidt (Massachusetts Institute of Technology) --- https://acoustics.mit.edu/faculty/henrik/oases.html

OASES provides OAST (transmission loss), OASN (modes), OASR
(reflection), and OASP (PE). Because OASES is not redistributable,
UACPY does **not** bundle the sources; `install.sh` downloads them
directly from MIT at install time.

**Modifications:** none --- used as-is.

### arlpy

Mandar Chitre (Acoustic Research Lab, National University of Singapore) --- https://github.com/org-arl/arlpy

A small number of domain-utility functions from `arlpy.uwa`
(sound speed, absorption, density --- Mackenzie and Francois-Garrison
formulas) and `arlpy.signal` (signal-processing helpers) have been
adapted into `uacpy/core/acoustics.py` and
`uacpy/acoustic_signal/advanced.py`. Each adapted file preserves
Mandar Chitre's 2016 copyright header and cites arlpy as the source.
The scientific formulas are unchanged; only Python-level formatting
(type hints, docstrings) differs from upstream.


## 📄 Licensing

UACPY aggregates code from multiple projects, each under its own
license. Downstream users are responsible for respecting each license
when redistributing or modifying UACPY or its outputs.

| Component                  | Location                           | How it ships                                     | License                                          |
|----------------------------|------------------------------------|--------------------------------------------------|--------------------------------------------------|
| UACPY wrapper              | this repository                    | source + Python package                          | GPL-3.0                                          |
| Acoustics Toolbox (Porter) | `third_party/Acoustics-Toolbox/`   | vendored Fortran sources, **modified**           | GPL-3.0                                          |
| bellhopcuda (Schmid et al.)| `third_party/bellhopcuda/`         | vendored C++/CUDA sources, **modified**          | GPL-3.0                                          |
| mpiramS (Dushaw)           | `third_party/mpiramS/`             | vendored Fortran sources, **modified**           | Creative Commons Attribution 4.0 International   |
| arlpy utilities (Chitre)   | `uacpy/core/`, `uacpy/acoustic_signal/` | adapted (ported into UACPY sources, unmodified scientifically) | BSD-3-Clause                    |
| OASES (Schmidt, MIT)       | `third_party/oases/` (gitignored)  | downloaded at install time, **not redistributed**| Academic license --- see Henrik Schmidt's terms  |


## 📬 Contact

Questions, bug reports, and contributions are welcome. For matters not
suited to a GitHub issue (collaboration proposals, private questions,
etc.), the maintainer can be reached at:

**ervul.github@gmail.com**


## 📖 Citation

``` bibtex
@software{uacpy2026,
  title   = {UACPY: Underwater ACoustics for PYthon},
  author  = {ErVuL and UACPY Contributors},
  year    = {2026},
  url     = {https://github.com/ErVuL/uacpy}
}
```


## Other interesting projects

- https://github.com/hunterakins/pykrak
- https://github.com/signetlabdei/WOSS?tab=readme-ov-file
- https://github.com/nposdalj/PropaMod
- https://github.com/marcuskd/pyram
- https://github.com/org-arl/UnderwaterAcoustics.jl



















