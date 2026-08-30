# Developer Guide

This document explains how UACPY is wired internally — the model
contract, the I/O layer, the shared support systems, and the rules for
extending or modifying any of them. It is a complement to:

- `README.md` — user-facing intro + quick start.
- `DOCUMENTATION.md` — public API reference (signatures, kwargs, units).

If you want to add a model, hook a new I/O format, or change shared
plumbing, start here.

---

## 1. Repository layout

```
uacpy/
├── docs/                    Guided pages (guide/, models/), figure_scripts/,
│                            doc checkers (check_links.py, check_structure.py),
│                            generate_model_figures.py, this file
├── install.sh               Native-binary build script (Fortran/C/CUDA)
├── pyproject.toml           Package + pytest config (default `-n logical`)
├── DOCUMENTATION.md         Public API reference
└── uacpy/
    ├── core/                Physics-agnostic dataclasses + invariants
    ├── models/              One PropagationModel subclass per engine
    ├── io/                  File-format readers/writers + FileManager
    ├── data/                External-data fetch layer (GPS → Environment)
    ├── comms/               Underwater communications (modem PHY, JANUS)
    ├── sonar/               Sonar equation, reverberation, detection, MFP
    ├── acoustic_signal/     psd/ppsd/sel, fk_transform/taup/radon, spectrogram, FRF
    ├── noise/               Wenz curves, wind noise, ship noise
    ├── visualization/       plot_field / plot_bottom_properties / … (+ result.plot())
    ├── tests/               pytest suite (markers: slow, requires_binary, …)
    ├── examples/            39 numbered example scripts (01–39)
    ├── third_party/         Vendored Fortran/C sources (see §9)
    ├── bin/                 Gitignored; populated by install.sh
    ├── parallel.py          run_parallel / Job — parallel batch runner
    ├── metrics.py           tl_rmse / tl_max_error / tl_bias (core.metrics shim)
    ├── _log.py              Single log channel + warning formatter
    ├── _stack.py            One-shot RLIMIT_STACK bump on import
    └── _version.py          `__version__`; pyproject reads it via `dynamic`
```

`uacpy/` (the source package) is installed editable via
`pip install -e ".[dev]"`. The native binaries (`bin/oalib/`,
`bin/bellhopcuda/`, `bin/mpirams/`, `bin/ramsurf/`, `bin/ramgeo/`,
`bin/oases/`) are built separately by `install.sh` — see the README.

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
unknown keyword raises Python's standard `TypeError` at the call site. Every
model takes exactly these parameters and no others; `output_duration=` is part
of that signature and is simply ignored by models with no broadband path. The
one sanctioned extension is the keyword-only `c_low`/`c_high`/`rmax` of
`Bellhop.run_with_bounce` — a *different method*, which tabulates the BOUNCE
reflection table a single call consumes. Model configuration is
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
REVERBERATION                          # Reverberation level vs range (OASS)
```

A model declares its supported subset in `spec.modes` (§2.3); the base copies
that to `self._supported_modes` and refuses anything else with
`UnsupportedFeatureError`. The first entry is the default when
`run_mode=None`.

`_SINGLE_FREQUENCY_MODES` (in `base.py`) refuses a multi-frequency
`Source` for `COHERENT_TL` / `RAYS` / `MODES` / etc. — you must pick
BROADBAND, TIME_SERIES, or one of the OASES sweep modes for those.

### 2.3 Capability flags

Each model declares which **env shapes** it consumes natively, along with its
run modes, source geometries and collapse defaults, in one `ModelSpec` class
attribute. The base validates it at *class-definition* time
(`PropagationModel.__init_subclass__` in `base.py`) and applies it in
`__init__`, so a malformed spec fails on import rather than on the first run:

```python
class Scooter(PropagationModel):
    spec = ModelSpec(
        modes=(RunMode.COHERENT_TL, RunMode.BROADBAND, RunMode.TIME_SERIES),
        supports={'layered_bottom', 'elastic_media', 'rough_surface'},
        source_types=frozenset({'point', 'line', 'scaled'}),
        collapse={'ssp': 'mean', 'bottom_range': 'median'},
    )
    source = 'acoustics_toolbox'          # MODEL_SOURCES id, not a spec field
```

`supports` names a subset of these ten flags; every flag not named defaults
False, and the base sets the matching `self._supports_<name>` attribute from
it:

```
altimetry                       range_dependent_bathymetry
range_dependent_ssp             range_dependent_bottom
layered_bottom                  elastic_media
multi_source_depth              source_beam_pattern
rough_surface                   rough_bottom
```

A subclass with no `spec` keeps the base defaults and sets
`_supported_modes` / `_supports_*` / collapse by hand in `__init__` — the
fallback path, used by none of the twelve shipped wrappers.

There is no `range_dependent_surface` flag: no engine consumes a
range-dependent surface deck (the AT solvers carry one global top boundary,
RAM one attenuator, OASES one top half-space), so a range-dependent `Surface`
is collapsed **unconditionally** in `_project_environment()` —
`collapse['surface']` picks the reduction method. The `Surface` carrier still
exists to build / fetch / plot a marginal ice zone.

Flip True for each axis the model handles natively. Anything left False
that appears in `env` on `run()` is **collapsed** by
`_project_environment()` and triggers one `UserWarning` per dropped
feature.

**The flag list is intentionally bounded.** Add a flag ONLY for a
question of the form "does this env shape work with this model?".
Numerical-method requirements (specific SSP interp scheme, 3-D-vs-2-D,
volume-attenuation formula) belong in `run()`-time asserts, not flags.

**Reading the flags back.** `spec.supports` is the *class-level declaration*
— what the author wrote. Ask an **instance** instead, through the two public
accessors that mirror `supported_modes` / `supports_mode`:

```python
model = Bellhop(interp_ssp='c-linear')
model.supported_features              # sorted list of flag names
model.supports_feature('range_dependent_ssp')   # False for this instance
Bellhop.spec.supports                 # the declaration; says nothing about interp_ssp
```

The two can differ on purpose: a model may resolve a flag from its own
constructor arguments in `__init__` (Bellhop turns `range_dependent_ssp` off
when `interp_ssp` cannot carry a 2-D profile), and the class-level
declaration cannot know that. `supports_feature` raises `ValueError` on a
name outside the ten — a typo must not answer "no".

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

Per-model physics-aware defaults go in `spec.collapse` (§2.3), which the base
feeds to `_set_collapse_defaults({...})`; a spec-less subclass calls that
helper itself in `__init__`. Either way the base merges user overrides over
them.

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

1. Subclass `PropagationModel` and declare a `ModelSpec` (§2.3) as a class
   attribute — run modes, capability flags, source geometries and collapse
   defaults — plus a `source` id from `models/sources.py`. Then in `__init__`:
   - call `super().__init__(...)` first, which applies the spec;
   - resolve the binary into `self._exe` (deliberately not a spec field:
     multi-binary models pick theirs at `run()` time);
   - store every constructor argument as `self.<name>` so
     `model.copy()` can introspect them.
2. Implement `run(self, env, source, receiver, run_mode=None, *,
   frequencies=None, source_waveform=None, sample_rate=None,
   output_duration=None)` — the fixed signature, no `**kwargs`:
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
3. Register the model in `uacpy/models/__init__.py` — both the import and
   `__all__`.
4. Add it to `_LAZY_ATTRS` in `uacpy/__init__.py`. That table is what makes
   `uacpy.NewModel` resolve at the top level; without the entry the wrapper is
   importable as `uacpy.models.NewModel` and raises `AttributeError` as
   `uacpy.NewModel`. `tests/test_lazy_imports.py` gates both directions.
5. Register every key the wrapper writes into `result.metadata` in
   `_DOCUMENTED_METADATA` (`uacpy/core/results/_base.py`), and add a drift case
   in `tests/test_metadata_file_paths.py::_drift_cases` — `list_metadata()` is
   public API and reads that registry.
6. Add the model to the `alternatives=[...]` list of every
   `compute_*` method in `uacpy/models/base.py` whose run mode it supports.
   Those lists are the "try one of these models instead" text a user gets from
   `UnsupportedFeatureError`.
7. Add a row to the `DOCUMENTATION.md` §7 capability matrix and to
   `docs/models/README.md`'s run-mode matrix. Both are gated
   (`tests/test_documentation.py`), against the model set derived from
   `uacpy.models.__all__`.
8. Add a test file under `uacpy/tests/`. Use the marker that fits:
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
units.py                            km_to_m, m_to_km, deg_to_rad
                                    (USE THESE at file boundaries)
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
- `results/` — `Result` base (`_base.py`) + `Field`, `Arrivals`, `Rays`, `Modes`,
  `Covariance`, `Replicas`, `ReflectionCoefficient`, plus
  `ResultStack`. Defines `PhaseReference` enum (`'travelling_wave'` /
  `'time_domain_native'`).
- `absorption.py` — `Thorp`, `FrancoisGarrison`, `Biological`,
  `ConstantAbsorption`. Callers use `alpha_db_per_m(f, z)`, which the base
  class implements: it validates the frequency once for every model and
  dispatches to `_alpha_db_per_m(f, z)`, the method a new subclass
  overrides. All implement `topopt_code()`. Models read `env.absorption` and emit the right AT
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
- `metrics.py` — cross-model TL agreement helpers over a pair of 2-D
  `Field`s: `tl_rmse`, `tl_max_error`, `tl_bias`. Re-exported at
  `uacpy.metrics` by the top-level `metrics.py` shim.
- `sediment.py` — grain size (Wentworth ϕ) → bulk geoacoustics. Distinct from
  `data/sediment.py` (§7.1), which fetches ϕ; this converts it. It lives in
  `core/` so `BoundaryProperties.from_grain_size` works without importing
  `uacpy.data`.
- `_beamforming.py` — the single Bartlett/MVDR numerical core behind both
  `acoustic_signal.bartlett_spectrum` / `mvdr_spectrum` (steering vectors) and
  `sonar.bartlett` / `mvdr` (replica banks). Change the algebra here, not in
  either caller.
- `_carrier_validate.py` — the shared input validators the carriers above call
  (`bottom.py`, `ssp.py`, `environment.py`), so one wrong-shape message reads
  the same wherever it comes from.
- `_warn_frames.py` — `USER_FRAME_SKIP`, the tuple of package path prefixes
  every `warnings.warn` in uacpy passes as `skip_file_prefixes` (§6.2). The
  most widely imported module in `core/`: every subpackage warns through it.
  Its docstring carries the trailing-separator and dropped-`.py` mechanics and
  why `stacklevel` must not be combined with the skip walk.
- `constants.py` — `DEFAULT_SOUND_SPEED`, `TL_MAX_DB`, `PRESSURE_FLOOR`,
  the broadband-grid defaults (`DEFAULT_BROADBAND_N_FREQS`,
  `DEFAULT_BROADBAND_BANDWIDTH_FACTOR`), and the phase-speed search
  factors (`C_LOW_FACTOR` for FFP solvers, `C_HIGH_FACTOR`). Promote any
  new "magic number" to this module rather than embedding it.
- `exceptions.py` — `UACPYError` (the base every other one derives
  from, so `except UACPYError` is the catch-all) plus
  `ConfigurationError`, `ExecutableNotFoundError`, `ModelExecutionError`,
  `UnsupportedFeatureError`, `InvalidDepthError`, `FileFormatError`,
  `DataFetchError`. Use these instead of bare `ValueError` /
  `TypeError`; see DOCUMENTATION §4 for which one each situation calls
  for.

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

Single output channel:
`log_message(source, message, *, verbose=False, level='info')` — `verbose` is
the caller's gate setting (below), `level` the severity of this message.
**Do not** use `print()` inside the package: `_log.py` holds the only one.

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

Every warn site attributes itself to the **user's** call line, never to the
uacpy frame that raised it — a warning that names a line inside the package
tells the caller to change a knob without saying which of their own lines set
it, and a `-W` filter keyed on their module never matches. Two forms do that,
and every site in the package uses one of them:

```python
from uacpy.core._warn_frames import USER_FRAME_SKIP
warnings.warn(msg, UserWarning, skip_file_prefixes=USER_FRAME_SKIP)   # preferred
warnings.warn(msg, UserWarning, stacklevel=2)                         # fixed depth
```

`skip_file_prefixes` walks out to the first frame outside the package, so it
survives a helper being inserted between the public function and the warn; a
hand-counted `stacklevel` does not, and is for the sites whose depth is fixed
by construction. **Never pass both** — the skip walk starts from the frame
`stacklevel` already selected, and the two compose into a frame further out
than either intends. A bare `warnings.warn(msg)` blames this package.
`tests/test_warning_attribution.py` gates all three rules.

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
- `noise/noise.py` — `compute_windnoise`, `WenzNoise`, and the per-mechanism
  submodels (wind, shipping, rain, turbulence, thermal) the composite selects
  between; `ship_radiated_noise.py` — ISO 17208 RNL and equivalent monopole
  source level from a measured pass-by, a different quantity from the
  shipping *ambient* term above; `marine_mammal.py` — Southall et al. 2019
  auditory weighting.
- `visualization/plots/` — every result/carrier plots via `.plot()`,
  which dispatches through `plot_result(result, env=…)` to the private
  per-type renderers (`_plot_rays`, `_plot_arrivals`, `_plot_mode_functions`,
  `_plot_environment`, `_plot_ssp`, …). Public free functions remain for the
  grid/flexible renderers (`plot_field`, `plot_absorption`), alternate views
  (`plot_bottom_properties`, `plot_mode_wavenumbers`, `plot_modes_heatmap`),
  composition (`compare`, `compare_models`, `plot_overview`, maps), and the
  raw-array DSP/comms plotters.
- `visualization/style.py` — colour palette: field colormaps, sediment
  fill/hatch styles and source/receiver marker styles. No font or sizing
  presets — importing it leaves `rcParams` untouched. Touch this if you want
  to change the package look-and-feel globally.

Convention: each result-type plotting function takes the `Result`
positionally + an optional `env=` for seafloor / surface overlays +
optional axis-control kwargs.

**`core` reaches up into `visualization`, from inside function bodies.**
`visualization/plots/` imports `uacpy.core` at module scope (it needs
`Environment`, `Field` and the rest to render them), and `.plot()` on a core
carrier or result calls back into it — eight edges, from `core/_grid.py`,
`core/absorption.py`, `core/environment.py`, `core/ssp.py`,
`core/results/_base.py` and `core/results/field.py`. Four of the eight name a
private renderer (`_plot_range_profile`, `_plot_environment`, `_plot_ssp`,
`_draw_result_credit`), so those names are depended on across a package
boundary despite the underscore.

Every one of those imports sits **inside the method that uses it**, and each
carries a comment saying so. Hoisting one to file scope makes `import uacpy`
raise `ImportError` from a partially initialised module, because
`uacpy/__init__` eagerly loads `uacpy.core`; the failure is immediate and
loud, and `tests/test_lazy_imports.py` catches it as well, since that test
asserts a cold `import uacpy` leaves matplotlib out of `sys.modules`.

Two consequences to keep in mind when touching either side. Renaming one of
the four private renderers means editing its `core` caller in the same change
(`tests/test_carrier_plot_methods.py` and `tests/test_visualization.py` fail
otherwise). And `core` cannot be read, split or vendored without the plotting
stack, even though nothing else in `core` needs matplotlib. Removing the eight
edges would make `core` a sink but would *not* make the package graph acyclic:
a 22-module SCC inside `uacpy.data` (counting every import edge, function-local
imports included — the same edge definition as the eight edges above; on
top-level imports alone it is 11 modules) and a 3-module one across
`core.results` / `core.results.field` / `core.results.modes` remain either
way.

### 7.1 On-demand data layer (`uacpy/data/`)

Builds an `Environment` from GPS coordinates (+ date) by fetching from public
ocean databases. Sits *upstream* of the models — it produces carrier inputs,
never touches model internals. **Data is fetched on demand, never bundled or
redistributed** (same rule as OASES, §9).

Module map. The core modules carry the shared machinery; alongside them sits
one file per dataset, named by suffix — `*_local.py` reads a cached grid
downloaded by `install.sh --data`, `*_live.py` calls a web service:

- `_http.py` — `http_get(url, …)`, the only network call (stdlib `urllib`),
  wraps failures in `DataFetchError`. No third-party HTTP dep.
- `_cache.py`, `_geo.py`, `_netcdf.py`, `_time.py` — cache dir resolution,
  great-circle / transect geometry, netCDF read helpers, date normalisation.
- `bathymetry.py` — GEBCO via OpenTopoData (`fetch_bathy`, `fetch_bathy_transect`).
- `sound_speed.py` — WOA23 (`fetch_ssp`, `fetch_ssp_transect`, `fetch_ts_profile`)
  + the shared `assemble_range_dependent(columns, ranges)` helper.
- `copernicus.py` — Copernicus Marine operational SSP (`copernicusmarine` is an
  optional extra, `pip install -e ".[copernicus]"`, lazy-imported; needs a free
  Copernicus account + `copernicusmarine login`).
- `sediment.py` — pure ϕ→geoacoustic conversion + `range_dependent_bottom_along`.
- `seabed.py` — EMODnet (CC-BY) and Diesing (CC-BY) seabed substrate.
- `sources.py` — the **data-source catalogue** (`SOURCES`, licences, citations).
- `environment.py` — `fetch_environment(...)`, the orchestrator + dispatch.
- Per-dataset modules — bathymetry `gebco_local`, `gmrt_live`,
  `emodnet_bathy_live`; seabed `emodnet_local`, `diesing_local`,
  `globsed_local`, `crust1_local`, `graw_local`, `mars`, `pelagic`,
  `sediment_db`; water column `woa23_local`, `argo`, `glodap_local`;
  surface `sea_surface`, `seaice_local`, `wind_local`, `wind_live`,
  `waves`, `ww3_live`; plus `absorption.py` (site Francois–Garrison).

Conventions: every fetcher shares the `(base_url, timeout, verbose)` trio,
raises `DataFetchError`/`ConfigurationError` only, logs via `log_message`, and
emits user notices via `warnings.warn`.

**Adding a source.** Write a module exposing a fetcher with the standard trio,
then:

- *Sediment* → write a `_<name>_pair(cached)` resolver returning
  `(point_fetcher, transect_fetcher)` and add one `_BottomProvider` row to
  `environment._BOTTOM_PROVIDERS` (its `id` doubles as the source keyword;
  `in_auto=True` joins the `'auto'` chain, `in_cache_auto=True` the `'local'`
  one). The accepted keywords, fallback order and provenance id all derive from
  that list.
- *Sound speed* → add a branch to `environment._fetch_ssp` (and the
  range-dependent block) keyed on `ssp_source`.
- *Absorption pH* → `environment._fetch_ph` is pH-source aware: it prefers the
  operational Copernicus BGC `ph` field on the Copernicus SSP branch
  (`copernicus.fetch_ph_operational`), else the cached GLODAP grid
  (`glodap_local.py`), else the model-default constant, then feeds
  `build_francois_garrison` under `with_absorption` (cache/best-effort, silent
  fallback).
- *Offline grid* → register it in `_cache.DATASETS` and add an `install.sh`
  `download_<name>` (mirror `download_globsed`/`download_glodap`).
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
date/coords. Non-commercial / unlicensed sources (currently CRUST1.0) emit a
runtime `UserWarning` whenever fetched, so they are never returned silently.

---

## 8. Tests (`uacpy/tests/`)

```bash
pytest                         # full suite, -n logical via xdist (pyproject default)
pytest -n 0                    # single-process for debugging
pytest -m "not slow"           # fast subset
pytest -m "not requires_binary and not slow"   # pure-Python dev tier
pytest uacpy/tests/test_bellhop.py::TestX::test_y -v
```

Markers (registered in `pyproject.toml`):

- `slow` — long broadband or large-grid runs.
- `requires_binary` — needs a compiled native binary under `uacpy/bin/`.
- `requires_oases` — needs OASES binaries (separate install).
- `requires_network` — hits a live external service (`uacpy.data`
  fetchers); deselected by default via `addopts`.
- `benchmark` — validates output against a closed-form analytic or
  canonical published reference.
- `convention` — pins repo conventions rather than runtime behaviour
  (docstring prose, source-convention sweeps, repr snapshots); a failure
  signals doc/convention drift, not a runtime defect.

The composed dev tier `-m "not requires_binary and not slow"` is the fast
pure-Python loop: 3,941 of 5,339 test functions, ≥5,280 collected cases
(static AST count as of 2026-08-29; `requires_oases` tests count as
`requires_binary` because `conftest.pytest_collection_modifyitems`
auto-attaches that marker, which is also what makes the conjunction
exclude OASES tests). It is a development loop, not a gate: the full
suite (default `pytest` invocation) must pass before a change lands.
Gate runs pass `-rs --durations=50` so every skip is identified in the
summary — a missing binary degrades marker-less guarded tests to skips,
and only the `-rs` report makes that visible — and the 50 slowest tests
are recorded for the next audit to measure against.

**`match=` anchoring.** Tests that pin error or warning wording via
`pytest.raises(..., match=...)` / `pytest.warns(..., match=...)` match
the load-bearing fragment only — the clause that carries the contract
(the offending name, the limit, the unit), never the full sentence.
`match=` is an unanchored `re.search`, so a fragment pattern survives
rewording around it, while a fully anchored pattern turns every wording
change into edits across hundreds of tests. Escape any regex
metacharacters inside the fragment (`re.escape` or `\(`-style escapes).

Future maintenance: the mechanical sibling clusters — 139–185 test
functions (2.7–3.5% of the suite, same-file siblings identical once
constants are masked, concentrated in `test_io_functions.py` and
`test_oass.py`) — are `@pytest.mark.parametrize` candidates, to be
folded file-by-file with per-file collected-case parity as the gate.
The `convention` marker is registered but not yet applied to the
meta/convention tests scattered through the mixed test modules
(doc-prose, source-sweep and repr-snapshot pins); marking those
class-by-class is the other standing maintenance item.

`test_documentation.py` is the docs gate. It imports `docs/check_links.py` and
`docs/check_structure.py` and runs them over `docs/` — dead link, dead section
anchor, unbalanced fence, unparseable sample — and then holds the prose to the
code: the §7 capability matrix and §18 defaults in `DOCUMENTATION.md`, the
model pages' own `| Name | Default |` tables, the §17 examples index, the
`models/README.md` run-mode matrix, worked-example ↔ figure-script
containment, and every `uacpy.…` name and keyword argument used by a
documented sample or an example script. Pure-Python, a few seconds, no marker.
Figure regeneration (`python docs/generate_model_figures.py`) stays manual —
it needs the native binaries and runs for minutes. Regenerating into a scratch
directory and pixel-diffing against the committed PNGs is how figure staleness
gets measured; mtime says nothing.

`tests/conftest.py` pins `matplotlib.use("Agg")` at import — before any test
can import matplotlib — and adds three **autouse** fixtures: reseed
`numpy.random` to `0xACED` before each test (worksteal means files do not run
in order), close all figures after each test, and rewrite
`tempfile.gettempdir()` to the per-test `tmp_path`.

Lint (CI parity — real-bug subset only):

```bash
flake8 uacpy/ --exclude=uacpy/third_party,uacpy/uacpy/third_party \
       --count --select=E9,F63,F7,F82,F401,F811 --show-source --statistics
```

CI runs on Ubuntu + Python 3.12 + `--bellhop cxx --oases yes`. macOS,
WSL, Python 3.13, the CUDA build, and the no-OASES partial install are
advertised but not validated by CI — test locally before submitting
patches that touch those paths. (`requires-python` is `>=3.12`; 3.10 and
3.11 are not supported.)

---

## 9. Vendored Fortran/C sources (`uacpy/third_party/`)

UACPY vendors:

- `Acoustics-Toolbox/` — Bellhop, Kraken, KrakenC, Scooter, SPARC,
  Bounce (Porter, NRL/HLS).
- `oases/` — Schmidt's OASES family. Academic license, **not**
  redistributable; `install.sh --oases yes` downloads it on demand.
- `mpiramS/` — Dushaw's MPI-parallel broadband RAM (CC-BY-4.0).
- `ramsurf/` — Collins's RAM family: `rams0.5.f` (elastic) and
  `ramsurf1.5.f` (variable sea surface).
- `ramgeo/` — Collins's RAMGEO range-dependent layered-fluid PE.
- `bellhopcuda/` — git submodule pinned to upstream `v1.5`, unmodified.
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
3. Be followed by a re-resolution of every citation into the patched file.
   An inserted or deleted line shifts every `file.f90:NNN` address below it,
   and the citation gates cannot see that: they check that a target carries
   code, not that it carries the *right* code, so a drifted address that
   lands on any other line of code passes silently. The only drift they can
   read without understanding the claim is a target that is blank, past the
   file's last line, or nothing but the end of a block (`end if`,
   `continue`) — measured on one such patch, they saw 2 of its 26 shifted
   addresses. Re-resolve each one from what the citing comment **claims** —
   read the sentence, find the Fortran that supports it, cite that — rather
   than by adding the offset or by quoting whatever now sits at the old
   address; on an already-drifted pin both of those launder the drift into a
   form nothing can detect.
   `command grep -rn 'file\.f90:[0-9]' uacpy --include='*.py'` enumerates
   them; mind the bare `:NNN` continuations beside a full citation, which the
   gate counts as skipped rather than checked.

Touching the vendored sources is a re-validation event, **not** a
refactor.

### 9.2 install.sh

Worth knowing:

- `-y` / `--yes` — non-interactive.
- `--bellhop fortran|cxx|cuda` — Fortran always built; `cxx` adds the
  C++ port; `cuda` adds the CUDA build (hard-errors if `nvcc` is
  absent).
- `--oases yes|no` — downloads from acoustics.mit.edu when `yes`.
- `--data LIST` — fills `./data_cache` for the offline `*_local.py` fetchers
  (§7.1); `LIST` is a comma list of dataset ids (`gebco`, `woa23`, `sediment`,
  `emodnet`, `coastline`, `globsed`, `crust1`, `diesing`, `seaice`, `glodap`,
  `wind`, `graw`) or `all`. Several fetchers' remediation messages name this
  flag, so it is the one to know when a cached-grid test skips.
- `--no-models` (`--data-only`) — skip every native build; pure-Python
  install, no compilers needed.
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
  - A uacpy file moves every time anyone edits above the citation, so
    name the symbol (`write_ssp_section`, `Bellhop.run`) rather than
    the line. `test_no_comment_pins_a_line_number_in_another_python_file`
    enforces this across the package **and** `uacpy/tests/`.
  - A line under `uacpy/third_party/` is a stable address only while
    the patch set above it is unchanged. That tree is **not** pristine:
    `third_party/MODIFICATIONS.md` documents patches uacpy applies
    (a 12-line `BLOCK` inserted in `KrakenField/field.f90`, the
    `misc/interpolation.f90` rewrite, RAM kind promotions and enlarged
    array dimensions). Adding or dropping a patch shifts every citation
    below its insertion point in that file, so re-check them alongside
    the patch — a re-vendor is not the only event that invalidates a
    line number.
  - Source that is *not* vendored here cannot be checked by anything in
    the repo, so mark it: prefix the address with `external:`
    (`external:rx.c:413` for the CMRE janus-c reference `comms/janus.py`
    transcribes). `test_vendored_citations_resolve_and_single_line_targets_carry_code`
    fails on an unmarked address that resolves to no vendored file, on a
    marked one that *does* resolve, and on a single-line target that is
    blank; it reads every element of a comma-continued address
    (`RefCoef.f90:139-140,146-147`) separately, and reports how many
    citations it skipped as external or as ambiguous (a bare basename
    shipping twice, e.g. `sspMod.f90`) so the gate's coverage stays
    visible.
- A leading underscore means **not public API** — it does not mean
  module-private. Underscore-prefixed names are imported across *package*
  boundaries in several directions, and each such import is a deliberate
  internal dependency rather than an accident. Which ones, and why each is
  not public, is recorded in one place: `_CROSS_PACKAGE_PRIVATES` in
  `tests/test_packaging.py`, enforced by
  `test_no_undocumented_private_name_crosses_a_package_boundary`. Read the
  list there rather than a summary here — a count or a set of directions
  written into this prose is a second copy that nothing checks, and it drifts.
  To add a cross-package private, add it to that list with its reason; the
  gate fails in both directions, so an entry that stops being imported has to
  be dropped too. Renaming a private that appears in the list is a
  cross-package change and needs its consumers updated in the same edit. The
  gate reads `from … import _name` only — a private reached as an attribute
  (`module._helper()`) does not appear in it.
- No backwards-compatibility shims. Change code directly; uacpy is
  pre-1.0 and explicitly LLM-bootstrapped per the README roadmap.
- One PR = one logical change. Mention which physics regime / file
  format the change targets in the title.

---

