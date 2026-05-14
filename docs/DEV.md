# Developer Guide

This document explains how UACPY is wired internally — the model
contract, the I/O layer, the shared support systems, and the rules for
extending or modifying any of them. It is a complement to:

- `README.md` — user-facing intro + quick start.
- `DOCUMENTATION.md` — public API reference (signatures, kwargs, units).
- `CLAUDE.md` — high-density architectural notes for AI assistants.
- `AUDIT.md` / `AUDIT_SCIENCE.md` — historical findings from the
  code/science audits.

If you want to add a model, hook a new I/O format, or change shared
plumbing, start here.

---

## 1. Repository layout

```
uacpy/
├── docs/                    PDFs, screenshots, this file
├── install.sh               Native-binary build script (Fortran/C/CUDA)
├── pyproject.toml           Package + pytest config (default `-n logical`)
├── DOCUMENTATION.md         Public API reference
├── AUDIT.md, AUDIT_SCIENCE.md  Audit logs
└── uacpy/
    ├── core/                Physics-agnostic dataclasses + invariants
    ├── models/              One PropagationModel subclass per engine
    ├── io/                  File-format readers/writers + FileManager
    ├── acoustic_signal/     PPSD, PSD, SEL, FRF, FKTransform, Spectrogram
    ├── noise/               Wenz curves, wind noise, ship noise
    ├── visualization/       plot_field / plot_rays / plot_modes / …
    ├── tests/               pytest suite (markers: slow, requires_binary, …)
    ├── examples/            25 numbered example scripts
    ├── third_party/         Vendored Fortran/C sources (see §9)
    ├── bin/                 Gitignored; populated by install.sh
    ├── _log.py              Single log channel + warning formatter
    └── _stack.py            One-shot RLIMIT_STACK bump on import
```

`uacpy/` (the source package) is installed editable via
`pip install -e ".[dev]"`. The native binaries (`bin/oalib/`,
`bin/bellhopcuda/`, `bin/mpirams/`, `bin/ramsurf/`, `bin/oases/`) are
built separately by `install.sh` — see the README.

---

## 2. The model contract

Every wrapper is a subclass of `models.base.PropagationModel`. The base
class enforces a tight API contract; bend it only when you must.

### 2.1 Run signature

```python
result = Model(...).run(env, source, receiver, run_mode=None, **kwargs)
```

The first four positional parameters are **fixed** and **shared** by
every model. Model configuration is **constructor-only** —
`RAM(dr=2.0, dz=0.5, np_pade=8)`, `Bellhop(beam_type='B', n_beams=500)`.
There is no `set_params()`. To sweep, build one instance per parameter
set; `model.copy(**overrides)` short-circuits the boilerplate.

`run()` returns one `core.results.Result` subclass — `Field`,
`Arrivals`, `Rays`, `Modes`, `Covariance`, `Replicas`, or
`ReflectionCoefficient`. The physical meaning of a `Field` is encoded
in its `dtype` + which keys live in `Field.coords` (e.g. `complex` plus
`{depth, range, frequency}` ≡ broadband `H(d, r, f)`).

### 2.2 RunMode enum

`models.base.RunMode` is the single source of truth for run modes:

```
COHERENT_TL / INCOHERENT_TL / SEMICOHERENT_TL
RAYS / EIGENRAYS / ARRIVALS
MODES                                  # Kraken eigenfunctions
COVARIANCE / REPLICA                   # OASN frequency-domain array products
TIME_SERIES                            # p(t) at receivers
BROADBAND                              # H(f) complex transfer function
REFLECTION                             # Plane-wave coefficients (Bounce, OASR)
```

A model advertises its supported subset in `self._supported_modes` and
the base class refuses anything else with `UnsupportedFeatureError`.

`_SINGLE_FREQUENCY_MODES` (in `base.py`) refuses a multi-frequency
`Source` for `COHERENT_TL` / `RAYS` / `MODES` / etc. — you must pick
BROADBAND, TIME_SERIES, or one of the OASES sweep modes for those.

### 2.3 Capability flags

Each model declares which **env shapes** it consumes natively:

```python
self._supports_altimetry                       = False
self._supports_range_dependent_bathymetry      = True
self._supports_range_dependent_ssp             = True
self._supports_range_dependent_bottom          = True
self._supports_layered_bottom                  = False
self._supports_range_dependent_layered_bottom  = False
self._supports_elastic_media                   = False
self._supports_multi_source_depth              = False
```

Flip True for each axis the model handles natively. Anything left False
that appears in `env` on `run()` is **collapsed** by
`_project_environment()` and triggers one `UserWarning` per dropped
feature.

**The flag list is intentionally bounded.** Add a flag ONLY for a
question of the form "does this env shape work with this model?".
Numerical-method requirements (specific SSP interp scheme, 3-D-vs-2-D,
volume-attenuation formula) belong in `run()`-time asserts, not flags.

### 2.4 Collapse policy

`DEFAULT_COLLAPSE` in `base.py` maps each collapsible feature to a
default reduction method:

```
'bathymetry'        : 'max'
'ssp'               : 'r0'
'bottom'            : 'r0'
'layered'           : 'halfspace'
'rd_layered_range'  : 'median'
'rd_layered_layers' : 'halfspace'
'altimetry'         : 'drop'
'elastic'           : 'fluid'
```

`VALID_COLLAPSE_METHODS` enumerates the allowed values per key and is
asserted at module import. User overrides via
`Model(collapse={'bathymetry': 'min', ...})` always win.

Per-model physics-aware defaults go in `_set_collapse_defaults({...})`
inside `__init__`. The base merges user overrides over those.

### 2.5 Result construction helpers

`base.py` provides two helpers every concrete `run()` should use:

- `self._result_kwargs(source, *, backend=..., frequencies=..., ...)`
  populates the cross-model identification block (`model`, `backend`,
  `source_depths`, `frequencies`, `phase_reference`) that every
  `Result` carries as direct attributes.
- `self._attach_output_paths(result, work_dir, base_name,
  primary_files=(('shd_file', '.shd'), ...))` attaches per-file metadata
  keys (`'shd_file'`, `'arr_file'`, `'brc_file'`, …) to
  `result.metadata` so downstream consumers can find what the model
  wrote.

Model-specific extras (`'brc_file'` for Bounce, `'c0'`/`'dr'`/`'dz'`
for RAM, …) go in `result.metadata`. Cross-model identification fields
are direct attributes on `Result`, **not** metadata.

---

## 3. Adding a new model

1. Subclass `PropagationModel`. In `__init__`:
   - call `super().__init__(...)` first;
   - set `self._supported_modes`;
   - flip the relevant `self._supports_*` flags;
   - install model-specific collapse defaults via
     `self._set_collapse_defaults({...})`;
   - store every constructor argument as `self.<name>` so
     `model.copy()` can introspect them.
2. Implement `run(self, env, source, receiver, run_mode=None, **kwargs)`:
   - call `self._resolve_run_mode(run_mode)` first;
   - call `env = self._project_environment(env)` to apply the collapse
     policy;
   - validate via `self.validate_inputs(env, source, receiver,
     run_mode=...)`;
   - allocate a working dir through `self._setup_file_manager()`;
   - write the model's input file(s) via the appropriate `io/` writer
     (don't roll a new format inline);
   - invoke the native binary via `self._run_executable(...)` or the
     subprocess helpers in `base.py`;
   - read outputs through the matching `io/` reader;
   - return a `Result` built from `self._result_kwargs(...)` +
     `self._attach_output_paths(...)`.
3. Register the model in `uacpy/models/__init__.py`.
4. Add a test file under `uacpy/tests/`. Use the marker that fits:
   - `slow` for broadband / large grids;
   - `requires_binary` if a native binary must be present;
   - `requires_oases` for OASES-only tests.

---

## 4. The I/O layer

`uacpy/io/` is the **only** module that touches file formats. Models
call its readers/writers and never `open()` a `.env` / `.shd` / `.mod`
file directly.

### 4.1 Map of the I/O modules

```
oalib_writer.py / oalib_reader.py   Acoustics-Toolbox (.env / .shd /
                                    .arr / .ray / .flp / …)
bellhop_writer.py                   Bellhop-specific knobs (beam types,
                                    run types) — Bellhop's writer is
                                    split out because its env-file
                                    options diverge from the rest of the
                                    AT family
oases_writer.py / oases_reader.py   OAST / OASN / OASP / OASR (.dat
                                    inputs, .trf / .xsm / .rpo / .trc
                                    outputs)
mpirams_writer.py / mpirams_reader.py   mpiramS env + TL grids
ramsurf_writer.py / ramsurf_reader.py   ramsurf1.5 env + TL grids
modes_reader.py                     Kraken .mod / .moA binary mode files
grn_reader.py                       Scooter / SPARC .grn Green's-
                                    function snapshots
refl_io.py                          .brc / .trc / .irc reflection-
                                    coefficient files
bathy_io.py                         .bty / .ati bathymetry / altimetry
file_manager.py                     FileManager — see §6.1
units.py                            km_to_m, m_to_km, deg_to_rad,
                                    rad_to_deg (USE THESE at file
                                    boundaries)
_fortran_helpers.py                 detect_endian, read_fortran_record,
                                    read_vector — Fortran unformatted
                                    direct-access helpers
utils.py                            misc reader/writer-shared utilities
```

### 4.2 Rules for I/O code

- **Units at boundaries.** Public API is metres everywhere except
  attributes carrying an explicit suffix (`_km`, `_cm`). OASES /
  Acoustics-Toolbox formats want km on disk — every writer that hits
  a km-using format converts via `m_to_km(...)` from `io/units.py`.
  Same for radians vs degrees.
- **Endian detection.** Fortran unformatted binary files (`.shd`,
  `.mod`, `.grn`) can be either-endian. Use `detect_endian(...)` from
  `_fortran_helpers.py` to auto-detect; do not hard-code `<i` / `<d`.
- **Reader-side translation.** When a reader returns a dict with keys
  the model wrapper passes into `Result.metadata`, rename to the
  documented schema (`Nsam → n_samples`, `cmin → c_min`,
  `bw → bandwidth_hz`, `df → df_hz`, `n_pts → n_points`).
- **Third-party formats are upstream contracts.** Before touching any
  reader/writer for `.shd`, `.mod`, `.trf`, `.dat`, …, consult the
  upstream documentation (`uacpy/third_party/.../doc/*.tex` for OASES,
  the PDFs in `docs/` for AT, the source comments for RAM). The format
  doc is authoritative; the existing code may have bugs (the audit
  found several).

---

## 5. Core dataclasses (`uacpy/core/`)

These are the physics-agnostic primitives every model consumes:

- `environment.py` — `Environment`, `BoundaryProperties`,
  `SedimentLayer`, `LayeredBottom`, `RangeDependentBottom`,
  `RangeDependentLayeredBottom`, `SoundSpeedProfile`.
- `source.py` / `receiver.py` — `Source(depths, frequencies)`,
  `Receiver(depths, ranges)`.
- `results.py` — `Result` base + `Field`, `Arrivals`, `Rays`, `Modes`,
  `Covariance`, `Replicas`, `ReflectionCoefficient`, plus
  `ResultStack`. Defines `PhaseReference` enum (`'travelling_wave'` /
  `'time_domain_native'`).
- `absorption.py` — `Thorp`, `FrancoisGarrison`, `Biological`,
  `ConstantAbsorption`. All implement `alpha_db_per_m(f, z)` and
  `topopt_code()`. Models read `env.absorption` and emit the right AT
  `TopOpt[4]` letter automatically.
- `acoustics.py` — user-helper sound-speed / density / pekeris-root
  / SPL utilities. **Not** imported by the model wrappers; safe to
  use from notebooks. Some functions are arlpy-adapted; see
  `third_party/arlpy/NOTICE`.
- `materials.py` — named-material presets (`SAND`, `MUD`, `BASALT`,
  `ICE`, …) for `BoundaryProperties`.
- `metrics.py` — cross-model comparison helpers (TL bias, residual,
  band-averaged TL).
- `constants.py` — `DEFAULT_SOUND_SPEED`, `TL_MAX_DB`, `PRESSURE_FLOOR`,
  the broadband-grid defaults (`DEFAULT_BROADBAND_N_FREQS`,
  `DEFAULT_BROADBAND_BANDWIDTH_FACTOR`), and the phase-speed search
  factors (`C_LOW_FACTOR` for FFP solvers, `C_LOW_FACTOR_KRAKEN` for
  the modal solver, `C_HIGH_FACTOR`). Promote any new "magic number"
  to this module rather than embedding it.
- `exceptions.py` — `ConfigurationError`, `ExecutableNotFoundError`,
  `ModelExecutionError`, `UnsupportedFeatureError`. Use these instead
  of bare `ValueError` / `TypeError`.

Public API attribute names: distances in **metres**, sound speeds in
**m/s**, densities in **g/cm³**, attenuations in **dB/wavelength**,
frequencies in **Hz**. **Depth is positive downward**; altimetry
height is positive upward.

---

## 6. Support systems

### 6.1 `FileManager`

`io/file_manager.py` allocates per-run scratch directories. Every
model `run()` should pull one via `self._setup_file_manager()`. Pass
`use_tmpfs=True` on construction to use `/dev/shm` when available
(faster I/O for grid-heavy runs).

`tests/conftest.py` rewires `tempfile.gettempdir()` to the per-test
`tmp_path` so scratch dirs from one xdist worker don't bleed into
another's `/dev/shm`.

### 6.2 Logging — `uacpy/_log.py`

Single output channel: `log_message(source, message, level='info')`.
**Do not** use `print()` inside the package.

Verbose gate semantics (string OR bool, accepted by every model
constructor and reader):

```
False | None | 'off' | 'silent'   →  WARN + ERROR only
True  | 'info'                    →  + INFO
'debug'                           →  + DEBUG
```

Warnings go through the standard `warnings.warn(...)` machinery; uacpy
installs a custom formatter at import (see `_uacpy_format_warning`) so
they render compactly.

### 6.3 Stack-size bootstrapping — `uacpy/_stack.py`

Side-effect-on-import: raises `RLIMIT_STACK` to the hard limit before
any binary runs. SPARC-class solvers blow the default 8 MiB stack on
first large allocation; subprocesses inherit the larger value. Do not
remove or guard the side effect.

### 6.4 Exceptions

Use the typed exception hierarchy from `core/exceptions.py`. The base
class auto-formats a clean message including the model name; models
attach `.stdout` / `.stderr` / `.return_code` on
`ModelExecutionError` for post-mortem.

---

## 7. Shared processing — `acoustic_signal/`, `noise/`,
   `visualization/`

These are orthogonal to the model layer. They consume `Result`
objects (typically `Field`) or raw arrays.

- `acoustic_signal/analysis.py` — `PPSD`, `Spectrogram`, `SEL` (sound
  exposure level), `PSD`, `FRF` (frequency-response function),
  `FKTransform`. Each has `compute(...)` + `plot(...)`.
- `acoustic_signal/processing.py` — beamforming, fourier synthesis,
  shift-to-max-correlation.
- `acoustic_signal/generation.py` — source-waveform synthesis (Ricker,
  Gaussian, M-wave, Hann sine, …) — uses the same alphabet as AT
  `cans.f90` where possible.
- `noise/noise.py` — `compute_windnoise`, Wenz curves, ship noise.
- `visualization/plots.py` — single-entry `plot_result(result, env=…)`
  plus per-result-type helpers (`plot_field`, `plot_rays`,
  `plot_arrivals`, `plot_modes_heatmap`, `plot_covariance`, …).
- `visualization/style.py` — colour palette + font/sizing presets.
  Touch this if you want to change the package look-and-feel globally.

Convention: each result-type plotting function takes the `Result`
positionally + an optional `env=` for seafloor / surface overlays +
optional axis-control kwargs.

---

## 8. Tests (`uacpy/tests/`)

```bash
pytest                         # full suite, -n logical via xdist (pyproject default)
pytest -n 0                    # single-process for debugging
pytest -m "not slow"           # fast subset
pytest uacpy/tests/test_bellhop.py::TestX::test_y -v
```

Markers (registered in `pyproject.toml`):

- `slow` — long broadband or large-grid runs.
- `requires_binary` — needs a compiled native binary under `uacpy/bin/`.
- `requires_oases` — needs OASES binaries (separate install).
- `integration` — multi-subsystem end-to-end.

`tests/conftest.py` autouse fixtures: force `matplotlib.use("Agg")`,
seed `numpy.random` to `0xACED`, close all figures after each test,
rewrite `tempfile.gettempdir()` to the per-test `tmp_path`.

Lint (CI parity — real-bug subset only):

```bash
flake8 uacpy/ --exclude=uacpy/third_party,uacpy/uacpy/third_party \
       --count --select=E9,F63,F7,F82 --show-source --statistics
```

CI runs on Ubuntu + Python 3.12 + `--bellhop cxx --oases yes`. macOS,
WSL, Python 3.10/3.11/3.13, the CUDA build, and the no-OASES partial
install are advertised but not validated by CI — test locally before
submitting patches that touch those paths.

---

## 9. Vendored Fortran/C sources (`uacpy/third_party/`)

UACPY vendors:

- `Acoustics-Toolbox/` — Bellhop, Kraken, KrakenC, Scooter, SPARC,
  Bounce (Porter, NRL/HLS).
- `oases/` — Schmidt's OASES family. Academic license, **not**
  redistributable; `install.sh --oases yes` downloads it on demand.
- `mpiramS/` — Lytaev's MPI-parallel RAM-S branch.
- `rams0.5/`, `ramsurf1.5/` — Collins's elastic + variable-surface RAM
  variants.
- `arlpy/` — partial vendor of arlpy.uwa (BSD-3-Clause). See
  `third_party/arlpy/NOTICE` for the list of adapted functions.

### 9.1 Rules

Every modification to a vendored source must:

1. Be documented with an exact diff in
   `uacpy/third_party/MODIFICATIONS.md`.
2. Be re-validated against upstream behaviour for the regime affected
   (Pekeris / Munk / canonical case agreement within tolerance). The
   README roadmap calls this out — silent numerical drift in vendored
   sources is the single biggest correctness risk in the project.

Touching the vendored sources is a re-validation event, **not** a
refactor.

### 9.2 install.sh

Worth knowing:

- `-y` / `--yes` — non-interactive.
- `--bellhop fortran|cxx|cuda` — Fortran always built; `cxx` adds the
  C++ port; `cuda` adds the CUDA build (hard-errors if `nvcc` is
  absent).
- `--oases yes|no` — downloads from acoustics.mit.edu when `yes`.
- `--force` — full clean rebuild of every selected component.

---

## 10. Coding conventions

- Public API uses metres / Hz / m/s / g/cm³ / dB/wavelength. Suffix
  the attribute name (`_km`, `_cm`) when not metres / cm.
- Constructor-only model configuration — no `set_params()`.
- Use the typed exception hierarchy (`core/exceptions.py`), not bare
  `ValueError`.
- Promote any new "magic number" to `core/constants.py`.
- Default to writing no comments. Only add one when the *why* is
  non-obvious (a hidden invariant, a workaround for a specific bug,
  behavior that would surprise a reader). Do **not** comment on code
  evolution ("this replaces the old…", "after the fix…"). Do **not**
  pin to current line numbers in nearby files; cite source-of-truth
  files (`AttenMod.f90:78`) instead.
- No backwards-compatibility shims. Change code directly; uacpy is
  pre-1.0 and explicitly LLM-bootstrapped per the README roadmap.
- One PR = one logical change. Mention which physics regime / file
  format the change targets in the title.

---

