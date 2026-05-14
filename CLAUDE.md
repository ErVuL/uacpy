# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

UACPY is a unified Python interface over classical underwater-acoustic propagation
models implemented in Fortran/C/C++. The Python layer wraps native binaries via
subprocess + file I/O; the heavy numerics live in `uacpy/third_party/`.

The full API reference is `DOCUMENTATION.md` (single file). The roadmap in
`README.md` explicitly flags this codebase as LLM-bootstrapped and **not
production-ready**: auditing the wrappers and re-validating the vendored native
sources against canonical cases is the project's first priority. Treat any
specific number with skepticism until the regime has been validated.

## Build & install

The Python package installs editable; native binaries are built separately by
`install.sh` and dropped into `uacpy/bin/{oalib,bellhopcuda,mpirams,ramsurf,oases}/`.

```bash
pip install -e ".[dev]"     # python deps incl. pytest, black, flake8, pytest-cov
./install.sh -y             # build every native model non-interactively
```

`install.sh` flags worth knowing:

- `-y` / `--yes` — non-interactive
- `--bellhop fortran|cxx|cuda` — Fortran is always built; `cxx`/`cuda` build the
  C++/CUDA variant in addition. `--bellhop cuda` hard-errors if `nvcc` is absent
  (no silent downgrade).
- `--oases yes|no` — OASES is academic-licensed; `install.sh` downloads from
  acoustics.mit.edu and is **never** redistributed in the wheel/sdist
  (intentional — see the `package-data` comment in `pyproject.toml`).
- `--force` — full clean rebuild of every selected component.

`install.sh` verifies toolchain prerequisites and aborts with a clear message
rather than installing system packages itself.

CI (`.github/workflows/ci.yml`) runs only `ubuntu-latest` + Python 3.12 +
`--bellhop cxx --oases yes`. macOS, WSL, Python 3.10/3.11/3.13, the CUDA build,
the Fortran-only fallback, and the no-OASES partial install are advertised but
not validated by CI — test them locally before submitting patches that touch
those paths.

## Tests

```bash
pytest uacpy/tests/                                  # full suite
pytest uacpy/tests/test_bellhop.py                   # one file
pytest uacpy/tests/test_models.py::TestX::test_y -v  # one test
```

`pyproject.toml` injects `-n logical --dist=worksteal` by default, so the suite
runs parallel via pytest-xdist. Custom markers (see `pyproject.toml`):

- `slow` — long-running (broadband, large grids)
- `requires_binary` — needs a compiled native binary in `uacpy/bin/`
- `requires_oases` — needs OASES binaries (separate `install.sh --oases yes`)
- `integration` — multi-subsystem end-to-end

Common filters: `-m "not slow"`, `-m "not requires_binary"`,
`-m "not requires_oases"`.

`tests/conftest.py` autouse fixtures: force `matplotlib.use("Agg")`, seed
numpy.random to `0xACED`, close all figures after each test, and rewrite
`tempfile.gettempdir()` to the per-test `tmp_path` so `FileManager` scratch dirs
are reaped and don't leak into `/dev/shm` under xdist.

Lint (CI parity, real-bug subset only):

```bash
flake8 uacpy/ --exclude=uacpy/third_party,uacpy/uacpy/third_party \
       --count --select=E9,F63,F7,F82 --show-source --statistics
```

## Architecture

### Top-level package layout (`uacpy/uacpy/`)

- `core/` — physics-agnostic dataclasses: `Environment`, `Source`, `Receiver`,
  `Result` hierarchy, `Absorption` models, `BoundaryProperties`, `constants`,
  `materials`, `metrics` (cross-model comparison), `acoustics` (adapted from
  arlpy).
- `models/` — one wrapper per propagation engine, all subclassing
  `models.base.PropagationModel`. `models.base` defines `RunMode` and the
  collapse/capability machinery (see below).
- `io/` — readers/writers per native file format. `oalib_{reader,writer}`
  handles all Acoustics-Toolbox formats (`.env`, `.shd`, `.arr`, `.ray`,
  `.flp`, ...); Bellhop has its own writer because its run-type/beam knobs
  diverge. `mpirams_*`, `ramsurf_*`, `oases_*`, `modes_reader`, `grn_reader`,
  `bathy_io`, `refl_io`, `units` (km↔m, deg↔rad helpers used by every
  writer/reader at file boundaries). `FileManager` manages per-run tempdirs
  (optionally tmpfs).
- `visualization/`, `acoustic_signal/`, `noise/` — orthogonal helpers.
- `third_party/` — vendored Fortran/C sources. **Several are modified
  in-tree**; full diffs are kept in `third_party/MODIFICATIONS.md`. Touching
  the vendored sources is a re-validation event, not a refactor.
- `bin/` — gitignored; populated by `install.sh`. `oalib/`, `bellhopcuda/`,
  `mpirams/`, `ramsurf/`, `oases/`.
- `examples/` — 25 scripts numbered `example_01_*.py` through
  `example_25_*.py`. The index in `DOCUMENTATION.md` (section 12) is the
  source of truth for what each one demonstrates.
- `_stack.py` — side-effect-on-import raises `RLIMIT_STACK` to the hard limit
  before any binary runs (SPARC-class binaries blow the 8 MiB default on first
  large allocation; subprocesses inherit the larger value).
- `_log.py` — single output channel `log_message(...)` plus a custom
  `warnings.formatwarning` installed at import. `verbose` gate semantics:
  `False`/`None`/`'off'`/`'silent'` → WARN+ERROR only; `True`/`'info'` →
  +INFO; `'debug'` → everything.

### The model contract (`models/base.py`)

Every wrapper takes the **same** first four `run()` parameters in the same
order:

```python
result = Model(...).run(env, source, receiver, run_mode=None, **kwargs)
```

Model configuration is **constructor-only** (`RAM(dr=2.0, dz=0.5, np_pade=8)`,
`Bellhop(beam_type='B', n_beams=500)`). To sweep parameters, instantiate one
model per parameter set — there is no `set_params()`. Unrecognised `**kwargs`
to `run()` are silently ignored (uacpy convention) and otherwise threaded
into the env-file writer for model-specific quirks.

Mode-specific kwarg conventions are fixed across the family:

- `TIME_SERIES`-capable models accept `source_waveform=` + `sample_rate=` on
  `run()` (Bellhop, Scooter, KrakenField, OASP, RAM). SPARC computes p(t)
  from its native source pulse and rejects them.
- Broadband-transfer-function models accept `frequencies=` as an override
  for `source.frequencies`.
- `Bounce` takes a required `output_dir=` for persisting `.brc`/`.irc`.

### `RunMode` enum

Standard run modes (each model declares a subset in `self._supported_modes`):
`COHERENT_TL`, `INCOHERENT_TL`, `SEMICOHERENT_TL`, `RAYS`, `EIGENRAYS`,
`ARRIVALS`, `MODES` (Kraken depth eigenfunctions), `COVARIANCE`/`REPLICA`
(OASN), `TIME_SERIES`, `BROADBAND`, `REFLECTION` (Bounce / OASR).

`_SINGLE_FREQUENCY_MODES` (in `base.py`) explicitly refuses a multi-frequency
`Source` for COHERENT_TL / RAYS / MODES / etc. The user must pick
BROADBAND/TIME_SERIES or one of the OASES sweep modes.

### Capability flags + collapse policy

Each model declares which environment shapes it natively supports via a
bounded set of `_supports_*` flags (altimetry, range-dependent bathymetry,
range-dependent SSP, range-dependent bottom, layered bottom, range-dependent
layered bottom, elastic media, multi-source-depth). Anything an env carries
that the model doesn't support is **collapsed** by `_project_environment`
with one `UserWarning` per dropped feature.

The collapse strategy per feature is a `dict` keyed by feature name; values
are validated at construction against `VALID_COLLAPSE_METHODS` in
`base.py`. User overrides go through `Model(collapse={'bathymetry': 'min', ...})`
and always win over `_set_collapse_defaults()` (the per-model physics-aware
default hook).

**The flag list is intentionally bounded.** Add a new flag only for "does
this env shape work with this model?" questions. Niche numerical knobs
(specific SSP interp scheme, 3-D-vs-2-D, volume-attenuation formula) belong
in `run()`-time asserts, not capability flags.

### `Result` typing (`core/results.py`)

`Result` subclasses are: `Field` (all gridded outputs — pressure / TL / H(f) /
p(t) / single-point trace), `Arrivals`, `Rays`, `Modes`, `Covariance`,
`Replicas`, `ReflectionCoefficient`. The physical meaning of a `Field` is
encoded in its dtype + which keys are in `coords` (e.g. `complex` +
`{depth, range, frequency}` = broadband `H(d, r, f)`). `Field.at`/`isel`/`max`
collapse axes by dropping them from `coords` and recording the chosen sample
in `Field.pinned`. Cross-model identification (`model`, `backend`,
`source_depths`, `frequencies`, `phase_reference`) is on every `Result` as
direct attributes — *not* via `Result.metadata`, which is reserved for
model-specific extras (`'brc_file'` for Bounce, `'c0'`/`'dr'`/`'dz'` for RAM,
etc.).

### RAM dispatcher

`models.ram.RAM` is a façade over three Fortran backends and auto-picks one
based on the environment: `mpiramS` (fluid broadband), `rams0.5` (elastic
bottoms, broadband via Python frequency loop), `ramsurf1.5` (variable
surface, broadband via Python frequency loop). All three support
`COHERENT_TL` / `BROADBAND` / `TIME_SERIES`. The selection rule is in
`models/ram.py`; tests live in `tests/test_ram_backends.py`.

## Units & conventions

Distances are **metres** unless the attribute/argument name carries an
explicit suffix (`_km`, `_cm`). Sound speeds m/s, densities g/cm³,
attenuations dB/wavelength, frequencies Hz. **Depth is positive
downward**; sea-surface altimetry height is positive upward (`z=0` at the
mean sea surface). Section 10 of `DOCUMENTATION.md` is the source of
truth.

## When touching vendored native sources

Every modification to a Fortran/C source under `third_party/` must:

1. Be documented with an exact diff in `third_party/MODIFICATIONS.md`.
2. Be re-validated against the upstream behaviour for the regime affected
   (Pekeris / Munk / canonical case agreement within tolerance). The README
   roadmap calls this out explicitly — silent numerical drift in the
   vendored sources is the single biggest correctness risk in the project.
