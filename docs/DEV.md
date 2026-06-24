# Developer Guide

This document explains how UACPY is wired internally — the model
contract, the I/O layer, the shared support systems, and the rules for
extending or modifying any of them. It is a complement to:

- `README.md` — user-facing intro + quick start.
- `DOCUMENTATION.md` — public API reference (signatures, kwargs, units).
- `CLAUDE.md` — high-density architectural notes for AI assistants.

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
└── uacpy/
    ├── core/                Physics-agnostic dataclasses + invariants
    ├── models/              One PropagationModel subclass per engine
    ├── io/                  File-format readers/writers + FileManager
    ├── acoustic_signal/     psd/ppsd/sel, fk_transform/taup/radon, spectrogram, FRF
    ├── noise/               Wenz curves, wind noise, ship noise
    ├── visualization/       plot_field / plot_rays / plot_modes / …
    ├── tests/               pytest suite (markers: slow, requires_binary, …)
    ├── examples/            38 numbered example scripts (01–38)
    ├── third_party/         Vendored Fortran/C sources (see §9)
    ├── bin/                 Gitignored; populated by install.sh
    ├── parallel.py          run_parallel / Job — parallel batch runner
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
result = Model(...).run(env, source, receiver, run_mode=None, *,
                        frequencies=None, source_waveform=None,
                        sample_rate=None, output_duration=None)
```

The signature is **fixed and minimal** — no `**kwargs` anywhere, so an
unknown keyword raises Python's standard `TypeError` at the call site.
The only sanctioned extensions are `n_modes=` on Kraken and
`output_duration=` on the broadband
synthesizers (Bellhop, RAM, Scooter, Kraken, OASP). Model configuration is
**constructor-only** —
`RAM(dr=2.0, dz=0.5, np_pade=8)`, `Bellhop(beam_type='B', n_beams=500)`.
There is no `set_params()`. To sweep, build one instance per parameter
set; `model.copy(**overrides)` short-circuits the boilerplate. Run a
batch of independent runs in parallel with `uacpy.run_parallel` over
self-contained `Job`s (a process pool; `uacpy/parallel.py`).

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
self._supports_range_dependent_surface         = False
self._supports_range_dependent_bathymetry      = True
self._supports_range_dependent_ssp             = True
self._supports_range_dependent_bottom          = True
self._supports_layered_bottom                  = False
self._supports_range_dependent_layered_bottom  = False
self._supports_elastic_media                   = False
self._supports_multi_source_depth              = False
```

`_supports_range_dependent_surface` is `False` for **every** model: the AT
solvers carry a single global top boundary (only the SSP varies with range),
so — exactly like a range-dependent *bottom* in Kraken — a range-dependent
`Surface` is collapsed, not honoured. The `Surface` carrier still exists to
build / fetch / plot a marginal ice zone.

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
'bottom_range'      : 'r0'
'bottom_layers'     : 'halfspace'
'altimetry'         : 'drop'
'surface'           : 'r0'      # r0 / rmax / mean / median (single boundary type)
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
2. Implement `run(self, env, source, receiver, run_mode=None, *,
   frequencies=None, source_waveform=None, sample_rate=None)` — the
   fixed signature, no `**kwargs`:
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

- `environment.py` — `Environment` (re-exports the carriers below for stable
  import paths). The shape/property carriers live in their own modules:
  `ssp.py` (`SoundSpeedProfile`, `generate_sea_surface`), `bathymetry.py`
  (`Bathymetry`), `altimetry.py` (`Altimetry`), `bottom.py`
  (`SedimentLayer`, `SeabedColumn`, `Bottom`, `BoundaryProperties`),
  `surface.py` (`Surface`). `env.bathymetry` / `ssp` / `altimetry` / `bottom`
  / `surface` are always these carriers — a scalar / pairs / single
  `BoundaryProperties` is coerced at construction.
  - **Grid-library contract** (`core/_grid.py`): the gridded carriers share
    `at(...)` (nearest, never fabricates), `isel(...)` (positional), and
    `eval(..., method=)` (interpolate: `linear`/`nearest`/`cubic`). *Shape*
    carriers (`Bathymetry`, `Altimetry`, `SoundSpeedProfile`) interpolate;
    *property* carriers (`Bottom`, `SeabedColumn`, `Surface`) are select-only
    (`at`/`isel` — boundary/material types cannot be blended, so no `eval`).
    A uniform `Surface` delegates `BoundaryProperties` attribute reads to its
    one node, so it stands in for a single boundary everywhere.
- `source.py` / `receiver.py` — `Source(depths, frequencies)`,
  `Receiver(depths, ranges)` (input param carriers; no grid-library slicing).
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
- `materials.py` — named-material presets for `BoundaryProperties`,
  keyed (case-insensitively) in the `MATERIALS` dict and looked up via
  `get_material(name)` / enumerated via `list_materials()`. Keys are
  seafloor classes: `'clay'`, `'silt'`, `'sand'`, `'gravel'`, `'moraine'`,
  `'chalk'`, `'limestone'`, `'basalt'`, `'granite'`. There are no
  uppercase module-level constants and no `'mud'`/`'ice'` entries (sea-ice
  lives as `SEA_ICE_*` constants in `constants.py`).
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

- `acoustic_signal/analysis.py` — `psd`, `ppsd` (→ `PPSDResult`), `sel` (sound
  exposure level). Pure functions returning arrays; `system_id.py` keeps the
  `FRF` class (it holds fitted state). Transforms (`fk_transform`,
  `taup_transform`, `radon_transform`, `spectrogram`, `cwt`, `wigner_ville`,
  `cepstrum`) are likewise functions with `inverse_*` where meaningful. **All
  plotting lives in `uacpy.visualization`** (`plot_psd`, `plot_fk`, …) — the
  `acoustic_signal`/`comms` modules import no matplotlib.
- `acoustic_signal/arrays.py` — beamforming / steering vectors;
  `active.py` — matched filter, pulse compression, ambiguity.
- `acoustic_signal/waveforms.py` — source-waveform synthesis (Ricker,
  Gaussian, M-wave, Hann sine, …) — uses the same alphabet as AT
  `cans.f90` where possible; `sequences.py` — m-sequences / coded probes.
- `acoustic_signal/constant_q.py` — constant-Q transform family (Brown 1991:
  transform / PSD / spectrogram / probabilistic), `spectrum`/`density` scaling.
- `acoustic_signal/bands.py` — decidecade (ISO 18405) band levels;
  `timefreq.py` — Hilbert, spectrogram, CWT, Wigner-Ville, cepstrum;
  `channel.py`, `modal.py`, `noise_synthesis.py`, `system_id.py`.
- `noise/noise.py` — `compute_windnoise`, Wenz curves, ship noise.
- `visualization/plots.py` — single-entry `plot_result(result, env=…)`
  plus per-result-type helpers (`plot_field`, `plot_rays`,
  `plot_arrivals`, `plot_modes_heatmap`, `plot_covariance`, …).
- `visualization/style.py` — colour palette + font/sizing presets.
  Touch this if you want to change the package look-and-feel globally.

Convention: each result-type plotting function takes the `Result`
positionally + an optional `env=` for seafloor / surface overlays +
optional axis-control kwargs.

### 7.1 On-demand data layer (`uacpy/data/`)

Builds an `Environment` from GPS coordinates (+ date) by fetching from public
ocean databases. Sits *upstream* of the models — it produces carrier inputs,
never touches model internals. **Data is fetched on demand, never bundled or
redistributed** (same rule as OASES, §9).

Module map (one per concern):

- `_http.py` — `http_get(url, …)`, the only network call (stdlib `urllib`),
  wraps failures in `DataFetchError`. No third-party HTTP dep.
- `bathymetry.py` — GEBCO via OpenTopoData (`fetch_bathy`, `fetch_bathy_transect`).
- `sound_speed.py` — WOA23 (`fetch_ssp`, `fetch_ssp_transect`, `fetch_ts_profile`)
  + the shared `assemble_range_dependent(columns, ranges)` helper.
- `copernicus.py` — Copernicus Marine operational SSP (`copernicusmarine` core
  dep, lazy-imported; needs a free Copernicus account + `copernicusmarine login`).
- `sediment.py` — pure ϕ→geoacoustic conversion + `range_dependent_bottom_along`.
- `seabed.py` / `lithology.py` — EMODnet (CC-BY) and Dutkiewicz (CC-BY-NC) bottom.
- `sources.py` — the **data-source catalogue** (`SOURCES`, licences, citations).
- `environment.py` — `fetch_environment(...)`, the orchestrator + dispatch.

Conventions: every fetcher shares the `(base_url, timeout, verbose)` trio,
raises `DataFetchError`/`ConfigurationError` only, logs via `log_message`, and
emits user notices via `warnings.warn`.

**Adding a source.** Write a module exposing a fetcher with the standard trio,
then:

- *Sediment* → add one row to `environment._bottom_registry()`
  (`id → (point_fetcher, transect_fetcher)`) and, to join `'auto'`, to
  `_AUTO_BOTTOM_ORDER` / `_BOTTOM_SOURCE_KEYWORDS`.
- *Sound speed* → add a branch to `environment._fetch_ssp` (and the
  range-dependent block) keyed on `ssp_source`.
- Add a `DataSource` row to `sources.SOURCES` (id, licence, attribution,
  citation, `commercial_use`) and map the dispatch keyword to that id so
  provenance is recorded automatically.

**Provenance — two levels, one container type.** `sources.py` holds two frozen
dataclasses: `DataSource` (the static **catalogue entry** — one per dataset,
holding identity/licence/citation; `SOURCES` is the catalogue) and
`DataProvenance` (one **fetch instance** — a reference to a `DataSource` via
`.source`, plus the *actual* `data_date` and `data_point=(lat, lon)` that fetch
returned, with `offset_km` derived from the requested point). Read the dataset's
identity/citation through `prov.source`; there is **no** attribute delegation.

Every carrier carries provenance **uniformly as a tuple of `DataProvenance`** in
`carrier.data_sources` — leaf carriers (`SoundSpeedProfile`/`Bathymetry`/
`BoundaryProperties`) as a validated field (`_coerce_data_sources` rejects a
non-`DataProvenance`), container carriers (`Bottom`/`SeabedColumn`/`Surface`) as
an aggregating property. A fetcher stamps its carrier with a real
`DataProvenance` (e.g. WOA23 → snapped cell centre + climatology period; Argo →
cast time + float position); `Environment` aggregates the union across its
carriers (`_aggregate_data_sources`, dedup by `r.source.id`) into
`env.data_sources`, and `fetch_environment`'s `_record_provenance` wraps any
un-stamped layer's bare catalogue id in `DataProvenance(source=…)` so the tuple
stays uniform. `uacpy.data.citations(env)` (or a carrier, id, `DataSource`, or
`DataProvenance`) renders the licence/attribution/citation plus the fetched
date/coords. Non-commercial sources (Dutkiewicz) emit a runtime `UserWarning`
whenever fetched, so they are never returned silently.

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

