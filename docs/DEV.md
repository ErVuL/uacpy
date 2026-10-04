# Developer Guide

This document explains how UACPY is wired internally — the package layout,
the run protocol every engine follows, the data objects and their export
protocol, the I/O layer, the shared support systems, the gates that hold the
structure, and the recipes for extending any of it. It is a complement to:

- `README.md` — user-facing intro + quick start.
- `DOCUMENTATION.md` — public API reference (signatures, kwargs, units).

If you want to add a model, hook a new I/O format, or change shared
plumbing, start here; the table below says which section to read.

---

## Where to find what

| Looking for | Where |
|---|---|
| An engine's knobs, spec and stage hooks | `uacpy/models/<engine>/_model.py` (OASES: `uacpy/models/oases/<program>.py`) |
| What one run resolved | the engine's `_settings.py` record, inside `RunSettings` (`uacpy/core/run_settings.py`) |
| The run protocol | `PropagationModel.run` in `uacpy/models/base.py` (§2) |
| The list of engines | `ENGINES` in `uacpy/models/_registry.py` |
| Run modes, output declarations | `RunMode`, `OutputSpec` in `uacpy/core/run_settings.py` |
| Capability flags, engine traits | `ModelSpec`, `EngineTraits` in `uacpy/models/_spec.py` |
| Collapse methods and defaults | `uacpy/models/_projection.py` (§2.6) |
| Carriers (`Environment`, `Source`, ...) | `uacpy/core/` (§5) |
| Results (`Field`, `Arrivals`, ...) | `uacpy/core/results/` (§5) |
| Save / load / xarray / NetCDF | the export protocol, `uacpy/core/_export.py` (§5.1) |
| A file format | `uacpy/io/` (§4) |
| Plotters | `uacpy/visualization/plots/`, the `_PLOTTERS` table (§7) |
| Errors and warning classes | `uacpy/core/exceptions.py` (§6.5) |
| How an object prints (`repr`) | `uacpy/core/_repr.py` (§5.2) |
| What a metadata key means | `_DOCUMENTED_METADATA` in `uacpy/core/results/_base.py` |
| Licences and citations of the engines | `MODEL_PROVENANCE` in `uacpy/models/provenance.py` |
| Licences and citations of the datasets | `SOURCES` in `uacpy/data/sources.py` (§7.1) |
| Binary lookup and launch | `uacpy/models/_launch.py` (§6.4) |
| Who may import whom | §12 |
| Why a convention test failed | §8.1 |
| How to add an engine / knob / result / reader / warning | §3, §13 |

---

## 1. Repository layout

```
uacpy/
├── docs/                    Guide pages (guide/, models/), figure_scripts/,
│                            doc checkers (check_links.py, check_structure.py),
│                            generate_model_figures.py, this file
├── install.sh               Native-binary build script (Fortran/C/CUDA)
├── pyproject.toml           Package + pytest config (default `-n logical`)
├── DOCUMENTATION.md         Public API reference
└── uacpy/
    ├── core/                Carriers, run records, physics helpers, invariants
    │   ├── results/         Result classes (Field, Arrivals, Rays, Modes, ...)
    │   └── acoustics/       User-level physics helpers, one module per subject
    ├── models/              The run protocol and one package per engine
    │   ├── base.py          PropagationModel: the staged run (§2)
    │   ├── bellhop/         Bellhop (+ BOUNCE route, synthesis from arrivals)
    │   ├── bounce/          BOUNCE reflection coefficients
    │   ├── kraken/          Kraken / KrakenC normal modes + field.exe
    │   ├── ram/             RAM: mpiramS, rams0.5, ramsurf1.5, ramgeo
    │   ├── scooter/         Scooter wavenumber integration
    │   ├── sparc/           SPARC time-marched wavenumber integration
    │   └── oases/           OAST / OASN / OASP / OASR / OASS / OASSP
    ├── io/                  File-format readers and writers (§4)
    ├── data/                External-data fetch layer (GPS → Environment, §7.1)
    ├── acoustic_signal/     generate, estimate, arrays, detect, channel
    ├── comms/               modulate, link, receive, janus — the modem chain
    ├── sonar/               Sonar equation, reverberation, detection, MFP
    ├── noise/               Wenz curves, wind noise, ship noise
    ├── visualization/       plot_* functions and the .plot() dispatch table
    ├── tests/               pytest suite (§8)
    ├── examples/            numbered example scripts (example_NN_*.py)
    ├── third_party/         Vendored Fortran/C sources (§9)
    ├── bin/                 Gitignored; populated by install.sh
    ├── parallel.py          run_parallel / Job — parallel batch runner
    ├── metrics.py           tl_rmse / tl_max_error / tl_bias (core.metrics shim)
    ├── _log.py              Single log channel + warning formatter
    ├── _stack.py            Child-only RLIMIT_STACK raise for model binaries
    └── _version.py          `__version__`; pyproject reads it via `dynamic`
```

`uacpy/` (the source package) is installed editable via
`pip install -e ".[dev]"`. The native binaries (`bin/oalib/`,
`bin/bellhopcuda/`, `bin/mpirams/`, `bin/ramsurf/`, `bin/ramgeo/`,
`bin/oases/`) are built separately by `install.sh` — see the README.

**Inside an engine package.** Every engine package follows one layout; an
engine has only the modules it needs:

| Module | Holds |
|---|---|
| `__init__.py` | re-exports the engine class |
| `_model.py` | the engine class: constructor knobs, `spec`, `provenance_id`, `outputs`, stage hooks |
| `_settings.py` | the frozen `<Engine>Settings` record one run resolves (§2.3) |
| `_plan.py` | what one deck resolves to before it is written (grids, windows, memory) |
| `_checks.py` | refusals of knobs and carriers the binary cannot run |
| `_extract.py` / `_output.py` | the binary's output read into the package's conventions, the result assembled |
| `_launch.py` | engine-specific launch handling (the generic launcher is `models/_launch.py`) |

OASES differs: `oases/_base.py` holds the abstract `OASES` base, and each
program is one public module (`oast.py`, `oasn.py`, `oasp.py`, `oasr.py`,
`oass.py`, `oassp.py`) holding its class and its Settings record, with shared
helpers in `_common.py`, `_sampling.py` and `_mean_field.py`.

**Beside the engine packages** in `uacpy/models/`, one concern per module:
`_spec.py` (`ModelSpec`, `EngineTraits`, the capability-flag vocabulary),
`_registry.py` (`ENGINES`), `_knobs.py` (knob validators, §2.3),
`_replay.py` (`from_run_settings`), `_projection.py` (collapse policy),
`_checks.py` (carrier and geometry checks of stages 1-2), `_band.py` (the
frequency grid of a BROADBAND / TIME_SERIES run), `_window.py` (the
Acoustics-Toolbox phase-speed window), `_budget.py` (the memory a run will
hold), `_workspace.py` (`FileManager`, §6.1), `_launch.py` (§6.4),
`_stacking.py` (the source-depth loop), `_extract.py` (stages 5-6),
`_notices.py` (once-per-run notices), `_conventions.py` (cross-model level
conventions), `_defaults.py` (what an unset knob resolves to),
`_introspect.py` (constructor introspection behind `copy()` and `__repr__`),
and the public `pe_grid.py` (PE grid optimiser) and `provenance.py`.

---

## 2. The run protocol

Every engine is a subclass of `models.base.PropagationModel`. The base class
owns `run()`; an engine supplies declarations and stage hooks.

### 2.1 Run signature

```python
result = Model(...).run(env, source, receiver, run_mode=None, *,
                        frequencies=None, source_waveform=None,
                        sample_rate=None, output_duration=None,
                        t_start=None)
```

The signature is **fixed and minimal** — no `**kwargs` anywhere, so an
unknown keyword raises Python's standard `TypeError` at the call site. Every
engine takes exactly these parameters; a keyword a run mode does not read is
refused or warned about by the base, not silently dropped. The one sanctioned
extension is the keyword-only `c_low`/`c_high`/`rmax` of
`Bellhop.run_with_bounce` — a *different method*, which tabulates the BOUNCE
reflection table a single call consumes.

Model configuration is **constructor-only** — `RAM(dr=2.0, dz=0.5,
n_pade=8)`, `Bellhop(beam_type='B', n_beams=500)`. There is no
`set_params()`; assigning an attribute the engine does not have raises. To
sweep, build one instance per parameter set; `model.copy(**overrides)`
short-circuits the boilerplate. Run a batch of independent runs in parallel
with `uacpy.run_parallel` over self-contained `Job`s (a process pool;
`uacpy/parallel.py`).

`run()` returns one `core.results.Result` subclass (`Field`, `Arrivals`,
`Rays`, `Modes`, `Covariance`, `Replicas`, `ReflectionCoefficient`), or a
`ResultStack` for a multi-depth source the engine runs one depth at a time.

### 2.2 The six stages

`PropagationModel.run` is a template method. It runs these stages in order;
the names in *italics* are the hooks an engine overrides.

| Stage | What happens | Engine hooks |
|---|---|---|
| 1 normalise | argument and keyword checks; the environment projected onto what the engine reads (`spec.collapse`, §2.6) | *`_normalise_env`*, *`_project_environment`*, *`_check_time_series_request`* |
| 2 validate | carrier types, geometry, source type against `spec.source_types`; the knobs re-checked (an attribute may have been reassigned) | *`_check_knobs`*, *`_validate_engine`* |
| 3 resolve | one `RunSettings`: run mode, frequency grid, source-depth loop, time grid, output declaration; the engine's own record; each `Notice` of that record warned once | *`_requested_frequencies`*, *`_resolve_engine_settings`*, *`_marched_frequencies`* |
| 4 execute | per source depth, per launch: write the deck, launch the binary, read its output | *`_n_launches`*, *`_prepare_launches`*, **`_write_input`**, **`_launch`**, **`_read_output`** |
| 5 extract | the result built, then checked against `outputs[run_mode]` | **`_to_result`** |
| 6 stamp | `run_mode` and the `RunSettings` recorded on the result | — |

The four **bold** hooks are abstract: an engine class that lacks one cannot be
instantiated. Stages 1-3 never launch anything, so two public methods stop
there: `run_settings()` returns the `RunSettings` a `run()` with the same
arguments would use (and emits the same warnings), and `validate_inputs()`
runs the same refusals and announces nothing. A refusal therefore reads the
same from all three entry points.

An engine that another engine runs first (BOUNCE for Bellhop's seabed, the
OASES mean field for OASS / OASSP) is reached through the producer hooks
`_producer_settings`, `_write_producer_deck` and `_launch_producer`, so the
producer runs the same projection and checks as a direct run.

### 2.3 Knobs and the run record

A **knob** is a keyword-only constructor parameter, stored as the public
attribute of the same name. There is no central knob table: introspection
(`models/_introspect.py`) finds the knobs for `copy()`, `__repr__` and the
record. After construction, assigning an attribute that is not a knob is
refused with the closest knob name (`refuse_unknown_knob` in
`models/_knobs.py`).

Each engine validates its knobs in `_check_knobs`, which runs at construction
and again in stage 2, using the shared validators in `models/_knobs.py`
(`positive_finite`, `whole_count`), which raise `ConfigurationError` naming
the knob and the value.

Stage 3 records the knobs. `RunSettings.engine` is the engine's
`<Engine>Settings` record (an `EngineSettings` subclass): its `knobs` field
holds every knob **as given** (`None` where the run derived it), and a knob
the run derived is the field of the **same name** holding the derived value.
Host knobs that cannot change a result (`work_dir`, `verbose`, `timeout`,
`executable`, ...) are not recorded (`_HOST_KNOBS` in `models/base.py`).

`result.run_settings` returns that record, and
`PropagationModel.from_run_settings(record)` rebuilds the engine that re-runs
it (`models/_replay.py`): every knob as given, every derived value pinned as
an explicit knob. Two kinds of value are not pinned:

- values the run takes from the call or composes from other knobs, listed in
  the engine's `_UNPINNED_FIELDS`;
- values derived once per launch (per range segment for Kraken, per
  frequency for RAM), declared by `_per_launch_knob_values`. When they differ
  between launches, the rebuilt model keeps the recorded values, and its first
  run warns (`ProvenanceWarning`) for each one it now derives differently.

`EngineSettings.to_dataframe()` tabulates the per-launch records of an engine
that has them.

### 2.4 RunMode enum

`core.run_settings.RunMode` is the single source of truth for run modes (it
sits in `core` beside the `RunSettings` record, so a result checks and
rebuilds its run mode without loading a model):

```
COHERENT_TL / INCOHERENT_TL / SEMICOHERENT_TL
RAYS / EIGENRAYS / ARRIVALS
MODES                                  # Kraken eigenfunctions
COVARIANCE / REPLICA                   # OASN frequency-domain array products
TIME_SERIES                            # p(t) at receivers
BROADBAND                              # H(f) complex transfer function
REFLECTION                             # Plane-wave coefficients (Bounce, OASR)
REVERBERATION                          # Reverberation loss vs range (OASS)
```

An engine declares its supported subset in `spec.modes` (§2.5) and refuses
anything else with `UnsupportedFeatureError`. The first entry is the default
when `run_mode=None`. A multi-frequency `Source` in a single-frequency mode
(`COHERENT_TL`, `RAYS`, `MODES`, ...) is refused; BROADBAND, TIME_SERIES or an
OASES sweep mode takes it.

What each mode returns is declared per engine in `outputs`, a mapping of
`RunMode` to `OutputSpec(result_type, kind=, unit=, phase_reference=,
coherent=)`. Stage 5 compares the built result with it and raises
`OutputContractError` on a mismatch, so a declaration and its engine cannot
drift.

### 2.5 The engine declaration: `ModelSpec`

Each engine declares, in class attributes checked when the class is defined
(`PropagationModel.__init_subclass__`, so a malformed declaration fails on
import):

- `spec = ModelSpec(...)` — run modes, capability flags, source geometries,
  collapse defaults and engine traits;
- `provenance_id` — the key of its entry in `MODEL_PROVENANCE`
  (`models/provenance.py`), which gives the licence and the citation;
- `outputs` — what each run mode returns (§2.4).

```python
class Scooter(PropagationModel):
    spec = ModelSpec(
        modes=(RunMode.COHERENT_TL, RunMode.BROADBAND, RunMode.TIME_SERIES),
        supports={'layered_bottom', 'elastic_media', 'rough_surface'},
        source_types=frozenset({'point', 'line', 'scaled'}),
        collapse={'ssp': 'mean', 'bottom_range': 'median'},
        traits=EngineTraits(
            consumes_volume_absorption=True,
            native_multi_depth_modes=frozenset({RunMode.COHERENT_TL}),
        ),
    )
    provenance_id = 'acoustics_toolbox'
```

`supports` names a subset of these capability flags; every flag not named
defaults to False:

```
altimetry                       range_dependent_bathymetry
range_dependent_ssp             range_dependent_bottom
layered_bottom                  elastic_media
source_beam_pattern             rough_surface
rough_bottom
```

`multi_source_depth` and `volume_attenuation` are derived, never declared:
the first from `traits.native_multi_depth_modes`, the second from
`traits.consumes_volume_absorption`. `EngineTraits` holds the remaining
per-engine facts the base reads (which modes stack source depths in one
launch, which ignore the frequency argument in TIME_SERIES, which Fortran
fatal messages are benign, ...); its docstring lists them.

There is no `range_dependent_surface` flag: no engine consumes a
range-dependent surface deck (the AT solvers carry one global top boundary,
RAM one attenuator, OASES one top half-space), so a range-dependent `Surface`
is collapsed **unconditionally** in `_project_environment()` —
`collapse['surface']` picks the reduction method. The `Surface` carrier still
exists to build / fetch / plot a marginal ice zone.

Anything in `env` that a flag left False is **collapsed** in stage 1 and
triggers one `FallbackWarning` per dropped feature.

**The flag list is intentionally bounded.** Add a flag ONLY for a question of
the form "does this env shape work with this model?". Numerical-method
requirements (specific SSP interp scheme, 3-D-vs-2-D, volume-attenuation
formula) belong in `_validate_engine`, not in flags.

**Reading the flags back.** `spec.supports` is the class-level declaration.
Ask an **instance** instead, through the two public accessors that mirror
`supported_modes` / `supports_mode`:

```python
from uacpy.models import Bellhop

model = Bellhop(interp_ssp='linear')
model.supported_features              # sorted list of flag names
model.supports_feature('range_dependent_ssp')   # False for this instance
Bellhop.spec.supports                 # the declaration; says nothing about interp_ssp
```

The two can differ on purpose: an engine may resolve a flag from its own
constructor arguments in `__init__` (Bellhop turns `range_dependent_ssp` off
when `interp_ssp` cannot carry a 2-D profile). `supports_feature` raises
`ValueError` on a name that is not a flag, so a typo does not answer "no".

### 2.6 Collapse policy

`DEFAULT_COLLAPSE` in `models/_projection.py` maps each collapsible feature
to a default reduction method:

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
asserted at module import. Engine defaults go in `spec.collapse`, and user
overrides via `Model(collapse={'bathymetry': 'min', ...})` win over both.

### 2.7 Building the result

Stages 5-6 live in `models/_extract.py`. An engine's `_to_result` builds the
result with `result_kwargs(...)`, which fills the cross-model identification
block every `Result` carries as attributes (`model`, `backend`,
`source_depths`, `frequencies`, `phase_reference`), and attaches the files it
wrote through `attach_output_paths(...)` (`'shd_file'`, `'arr_file'`, ... in
`result.metadata`). Engine-specific extras go in `result.metadata`, and every
key is registered in `_DOCUMENTED_METADATA` (§3, step 8). Package code never
reads a metadata key back: metadata is provenance for the user, not an input.

---

## 3. Adding a new model

Every step names the test that fails when it is skipped.

1. **The package.** Create `uacpy/models/<engine>/` with the modules of §1
   it needs, and add `uacpy.models.<engine>` to `_EXPECTED_PACKAGES` in
   `tests/test_packaging.py`, in its sorted position: the list is compared
   whole (`test_discovery_yields_exactly_the_shipped_packages`).
2. **The class.** Subclass `PropagationModel` in `_model.py` and declare
   `spec`, `provenance_id` and `outputs` (§2.5). A missing `spec` or
   `provenance_id` is a `TypeError` at import; an unknown `provenance_id`
   needs its `MODEL_PROVENANCE` entry first. Implement the four abstract
   stage hooks, `_check_knobs` (required:
   `test_every_engine_re_checks_its_knobs_in_stage_2`), and as needed
   `_validate_engine` and `_resolve_engine_settings`, which returns the
   engine's `<Engine>Settings` from `_settings.py`.
3. **The knobs.** Keyword-only, stored as `self.<name>`, each documented in the
   engine's one Parameters section (§13.1).
4. **The registry row.** Add an `EngineEntry` to `ENGINES` in
   `uacpy/models/_registry.py`: the module and class name, the constructor
   keywords a small example run needs (`example_kwargs`), the seabed it runs
   on if it refuses the default one (`example_bottom`), and the installs it
   needs (`requires`). That row is what exports the engine:
   `uacpy.models.NewModel` and `uacpy.NewModel` (`_LAZY_ATTRS`) derive from
   it, as do the `alternatives` a `compute_*` refusal suggests, and the
   conformance suite (`tests/test_engine_conformance.py`) and every test
   parametrised with `conftest.engine_params()` read it. Restate the class in
   the `TYPE_CHECKING` blocks of `uacpy/__init__.py` and
   `uacpy/models/__init__.py`, which type checkers read instead of the lazy
   tables (`tests/test_lazy_imports.py`).
5. **The per-engine test tables.** Each of these is keyed by engine name and
   fails, naming this step, until it has the engine's row: `_EXPECTED` in
   `tests/test_supported_modes.py`; `_EXPECTED`, `_EXPECTED_SOURCE_TYPES`,
   `_EXPECTED_ROUGH_SURFACE`, `_EXPECTED_MULTI_SOURCE_DEPTH`,
   `_EXPECTED_ROUGH_BOTTOM`, `_VOLUME_ATTENUATION` and
   `_MULTI_DEPTH_IN_DEFAULT_MODE` in `tests/test_capability_flags.py`;
   `_PER_MODEL_DEFAULTS` in `tests/test_collapse_policy.py`; `_drift_cases()`
   in `tests/test_metadata_file_paths.py`. The engine also needs a saved
   run-settings record, the form a result saved to disk carries:
   `python -m uacpy.tests._save_run_settings <key>` writes it into
   `tests/data/run_settings_by_engine.json` once
   (`test_a_saved_run_settings_loads_from_the_class_path_it_names`).
6. **The deck writer and reader.** Through `uacpy/io/` (§4, §13.4), never a
   format rolled inline. A new writer's parameters spell each knob as the
   engine does, and the writer gets a row in `_WRITER_ENGINES` in
   `tests/test_architecture.py` (one engine per writer: an engine that reuses
   another's writer adds none). An engine that imports a private io helper
   needs a row in two allow-lists: `_PRIVATE_NAMES_ACROSS_LAYERS` in
   `tests/test_architecture.py` (`test_layering`, §12) and
   `_CROSS_PACKAGE_PRIVATES` in `tests/test_packaging.py` (§10); a public
   helper needs neither. The binary launches through `models/_launch.py`
   (§6.4) and weighs its memory through `models/_budget.py`.
   `tests/test_architecture.py` also refuses a vocabulary (a file suffix, a
   letter code) declared in two places, so a second engine reuses the first
   one's constant rather than restating it.
7. **The docs.** A row in the `DOCUMENTATION.md` §7 capability matrix, a
   column in `docs/models/README.md`'s run-mode matrix, and a model page
   `docs/models/<engine>.md` whose Default column matches the constructor,
   named in `_PAGE_CLASSES` in `tests/test_documentation.py` with the
   classes it documents. `tests/test_documentation.py` holds all three
   against the code, and fails while a page or an engine is missing from
   `_PAGE_CLASSES`. Update `docs/models/README.md`'s feature matrix (range
   dependence, layered and elastic seabeds) too: no test reads it.
8. **The metadata keys.** Declare every key the engine writes into
   `result.metadata` as a `('<ModelName>', '<key>')` row of
   `_DOCUMENTED_METADATA` in `uacpy/core/results/_base.py`
   (`test_metadata_keys_are_all_documented`; a row nothing writes fails
   `test_documented_metadata_has_no_dead_rows`).
9. **The tests.** Add a test file under `uacpy/tests/` with the marker that
   fits (`slow`, `requires_binary`, `requires_oases`, §8). Name it for the
   subject it pins, never for a session, a date or a kind of code
   (`tests/test_packaging.py` gates test file names).

---

## 4. The I/O layer

`uacpy/io/` is the **only** module that touches file formats. Models
call its readers/writers and never `open()` a `.env` / `.shd` / `.mod`
file directly.

### 4.1 Map of the I/O modules

```
oalib_writer.py / oalib_reader.py   Acoustics-Toolbox (.env / .shd /
                                    .ssp / .flp / .rts / …)
bellhop_writer.py                   Bellhop-specific deck sections (beam
                                    types, run types), split out because
                                    its options diverge from the rest of
                                    the AT family
env_reader.py                       read_env: an AT .env deck back into
                                    carriers (the inverse of the writers)
at_codes.py                         the AT deck letters, one table per
                                    .env option field, shared by the
                                    writers and read_env
oases_writer.py / oases_reader.py   OAST / OASN / OASP / OASR / OASS /
                                    OASSP (.dat inputs, .trf / .xsm /
                                    .rpo / .trc / .rhs outputs)
mpirams_writer.py / mpirams_reader.py   mpiramS env + TL grids
ramsurf_writer.py / ramsurf_reader.py   ramsurf1.5 / rams0.5 env + TL grids
modes_reader.py                     Kraken .mod / .moA binary mode files
grn_reader.py                       Scooter / SPARC .grn Green's-
                                    function files -> GreensFunction
refl_io.py                          .brc / .trc / .irc reflection-
                                    coefficient and .sbp beam-pattern files
bathy_io.py                         .bty / .ati bathymetry / altimetry
audio_io.py                         write_wav / read_wav /
                                    read_wav_metadata — WAV export and
                                    import of signals
_parsers.py                         the raw output parsers the engines
                                    read (a user reads the public record
                                    built from the same file)
_fortran_helpers.py                 detect_endian, read_fortran_record,
                                    DirectAccessFile — Fortran unformatted
                                    and direct-access helpers
input_checks.py                     what a reader/writer checks before it
                                    trusts its input
```

A public reader returns an `ExportRecord` (§5.1), a carrier or a result, never
a bare dict of arrays. Unit conversion helpers (`km_to_m`, `m_to_km`,
`deg_to_rad`) live in `core/units.py`, not under `io/`; every reader and
writer imports them from there at its file boundary.

### 4.2 Rules for I/O code

- **Units at boundaries.** Public API is metres everywhere except
  attributes carrying an explicit suffix (`_km`). OASES /
  Acoustics-Toolbox formats want km on disk — every writer that hits
  a km-using format converts via `m_to_km(...)` from `core/units.py`.
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
  `ssp.py` (`SoundSpeedProfile`), `bathymetry.py` (`Bathymetry`),
  `altimetry.py` (`Altimetry`, and the Pierson–Moskowitz generator
  `generate_sea_surface` behind `Altimetry.from_sea_state`), `boundary.py`
  (`BoundaryType`, `BoundaryProperties`, `SedimentLayer`: the boundary nodes
  the seabed and the surface share), `bottom.py` (`SeabedColumn`, `Bottom`),
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
- `run_settings.py` — the run records: `RunMode`, `RunSettings`,
  `EngineSettings`, `OutputSpec`, `TimeSettings`, `Notice` (§2). `_records.py`
  holds their `FrozenRecord` base.
- `results/` — `Result` base (`_base.py`) + `Field`, `Arrivals`, `Rays`, `Modes`,
  `GreensFunction`, `Covariance`, `Replicas`, `ReflectionCoefficient`, plus
  `ResultStack`. Defines `PhaseReference` enum (`'travelling_wave'` /
  `'time_domain_native'`).
- `absorption.py` — `Thorp`, `FrancoisGarrison` (one water row, or a T/S
  profile, each property a number, an array on `depths=` or `(depth, value)`
  pairs on its own axis; no depth of its own — every evaluation takes the
  depth evaluated), `Biological`, `ConstantAbsorption`. **One object
  per absorption model: the law.** It is what a user builds, an
  `Environment` holds, a file saves and loads; `law.table(frequencies,
  depths=, units=)` evaluates it into an `AbsorptionCoefficient` (α with its
  axes, `.model`, `.parameters`) to look at, plot or export, and there is no
  second, function-form spelling of a law. The base class validates the
  frequency once for every model and dispatches to `_alpha_dB_per_m(f, z)`,
  the method a new subclass overrides; the plain-array formulas
  (`absorption_<model>`, dB/km, `core/acoustics/attenuation.py`) are the computation it
  calls. Formulas compute numbers; objects are what models take, plot, save and cite; the object calls the formula. `Environment(absorption=...)` takes a law, or a measured
  `AbsorptionCoefficient` with no law behind it (`model=None`), which
  becomes the private `_TabulatedAbsorption`; a table a law computed is
  refused (pass the law) — no second public type, and no second path to a
  law.
  The engines read a law through generic methods, never
  its type: `alpha_dB_per_m`, `alpha_dB_per_wavelength(f, z, c)` (what an AT
  `alphaI` row and a RAM water block carry), and the private hooks
  `_breakpoint_depths` (where α bends: an engine samples there),
  `_scales_by_frequency_ratio` (Bellhop's one-trace band synthesis scales by
  the surface ratio, and the band check measures that with `by_ratio=True`),
  `_table_depths` (what `table` evaluates at given no depths) and
  `_needs_node_sound_speed`. The AT writers look up the `TopOpt(4)`
  letter of `env.absorption` in `io/at_codes.py`
  (`VOLUME_ATTENUATION_CODES`), where a new law with a deck formula adds its
  row; a law with no letter (Francois-Garrison, whose `'F'` would freeze it
  at one deck `z_bar`; a table) goes into the water SSP rows' `alphaI` at the
  deck frequency — except one Francois-Garrison row on a deck covering
  several frequencies, written as `'F'` at mid-water column
  (`writes_francois_garrison_letter`, `multi_frequency=`)
  (`writes_alpha_per_ssp_row`), so a new law runs on every engine without
  one.
- `acoustics/` — user-helper physics, one module per subject:
  `seawater.py` (four sound-speed equations, density, Doppler),
  `boundaries.py` (`reflection_coeff`, `bottom_loss_curve`,
  `pekeris_root`), `bubbles.py` (resonance, bubbly-water speed, surface
  loss), `levels.py` (volts → Pa → dB), `wavenumber.py` (Hankel transform,
  alias period) and `modal.py` (mode shapes at depths, modal sum). Every
  public name is re-exported from the package (and by the root
  `uacpy/acoustics.py` module), so callers write
  `uacpy.acoustics.sound_speed_mackenzie` and never name a sub-module. **Not**
  imported by the model wrappers; safe to use from notebooks. Some
  functions are arlpy-adapted; see `third_party/arlpy/NOTICE`.
- `materials.py` — named-material presets for `BoundaryProperties`,
  keyed (case-insensitively) in the `MATERIALS` dict and looked up via
  `get_material(name)` / enumerated via `list_materials()`. Keys are
  seafloor classes: `'clay'`, `'silt'`, `'sand'`, `'gravel'`, `'moraine'`,
  `'chalk'`, `'limestone'`, `'basalt'`, `'granite'`. There are no
  uppercase module-level constants and no `'mud'`/`'ice'` entries (sea-ice
  lives as `SEA_ICE_*` constants in `data/seaice_local.py`, their one
  reader).
- `metrics.py` — cross-model TL agreement helpers over `(field, reference)`,
  two `Field`s on one depth/range grid or two plain dB arrays: `tl_rmse`,
  `tl_max_error`, `tl_bias`, and
  `tl_rmse_on_shared_ranges` for two grids that share only some ranges.
  Re-exported at
  `uacpy.metrics` by the top-level `metrics.py` shim.
- `sediment.py` — grain size (Wentworth ϕ) → bulk geoacoustics. Distinct from
  `data/sediment.py` (§7.1), which fetches ϕ; this converts it. It lives in
  `core/` so `BoundaryProperties.from_grain_size` works without importing
  `uacpy.data`.
- `_beamforming.py` — the single Bartlett/MVDR numerical core behind
  `acoustic_signal.bartlett` / `mvdr`, which every matched-field surface
  (`sonar.bartlett` / `mvdr`, `Covariance.bartlett` / `mvdr`) wraps, and the
  snapshot covariance behind `sample_covariance` and `sonar.csdm`. Change the
  algebra here, not in a caller.
- `_repr.py` — the one-line repr formatting every public class shares
  (§5.2): the axis, quantity and record helpers and the `FieldsRepr` /
  `SettingsRepr` mixins.
- `_validate.py` — the input guards every layer calls (carriers, results, io,
  the signal estimators), so one wrong-shape message reads the same wherever it
  comes from. Beside it, `collapse.py` holds the public collapse vocabularies
  (`RANGE_COLLAPSE_METHODS`, `DEPTH_COLLAPSE_METHODS`,
  `COLUMN_COLLAPSE_METHODS`), `_provenance.py` the `data_sources` checks and
  merge, and `_carrier.py` the `copy()` and revalidate-on-assignment mixins.
- `_warn_frames.py` — `USER_FRAME_SKIP`, the tuple of package path prefixes
  every `warnings.warn` in uacpy passes as `skip_file_prefixes` (§6.2). The
  most widely imported module in `core/`: every subpackage warns through it.
  Its docstring carries the trailing-separator and dropped-`.py` mechanics and
  why `stacklevel` must not be combined with the skip walk.
- `constants.py` — the physics reference every layer reads: the reference
  sea water (`REFERENCE_*`, `DEFAULT_SOUND_SPEED`,
  `DEFAULT_WATER_DENSITY_G_CM3`), `PRESSURE_FLOOR`/`NO_ENERGY_DB`, the dB
  reference pressures, `NEPER_TO_DB` and `EARTH_RADIUS_M`. A constant lives
  with its owner: the Acoustics-Toolbox deck limits the carriers enforce in
  `core/deck_limits.py`, `BoundaryType` in `core/boundary.py`, model defaults in
  `models/_defaults.py` (`DEFAULT_C_MIN`, `DEFAULT_BROADBAND_N_FREQS`,
  `DEFAULT_BROADBAND_BANDWIDTH_FACTOR`), the phase-speed search factors
  (`C_LOW_FACTOR`, `C_HIGH_FACTOR`, `DEFAULT_C_MAX_UNBOUNDED`) beside their
  one resolver in `models/_window.py`, and the AT deck letters
  (`AttenuationUnits`, `parse_boundary_type`) in `io/at_codes.py`.
- `exceptions.py` — `UACPYError` (the base every other one derives
  from, so `except UACPYError` is the catch-all) plus
  `ConfigurationError`, `ExecutableNotFoundError`, `ModelExecutionError`,
  `UnsupportedFeatureError`, `InvalidDepthError`, `FileFormatError`,
  `DataFetchError`, `OutputContractError` (an engine's result broke its
  declared output contract). Use these instead of bare `ValueError` /
  `TypeError`; see DOCUMENTATION §4 for which one each situation calls
  for. The warnings are `UACPYWarning` subclasses by cause
  (`NumericsWarning`, `ValidityWarning`, `FallbackWarning`,
  `ProvenanceWarning`, `IOWarning`); a `warnings.warn` passes the one
  that names its cause, never a bare `UserWarning`.

Public API attribute names: distances in **metres**, sound speeds in
**m/s**, densities in **g/cm³**, attenuations in **dB/wavelength**,
frequencies in **Hz**. **Depth is positive downward**; altimetry
height is positive upward.

### 5.1 Carriers, records and the export protocol

Three kinds of data object cross the package, each with one base:

- A **carrier** is an input the user builds and may change: the environment
  carriers, `Source`, `Receiver`, the absorption models. It is a `@carrier`
  dataclass (`core/_carrier.py`) inheriting `RevalidateOnAssignMixin`,
  `DeepCopyMixin` and `CarrierExport`. Its checks live in `__post_init__`, and
  assigning a field rebuilds the carrier through its constructor, so the
  checks run again: a carrier cannot be put into a state its constructor
  refuses. A carrier whose fields are coupled overrides
  `_fields_for_assignment`; `Environment`, which holds the other carriers,
  checks and completes an assignment in its own `__setattr__` instead of the
  mixin. `@carrier`'s `__init__` counts as a package
  frame, so a warning raised while building one names the user's line (§6.2).
- A **record** is a frozen value nothing changes after it is built:
  `FrozenRecord` (`core/_records.py`) for the run records (`RunSettings`,
  `EngineSettings` and the per-engine Settings, `OutputSpec`, `TimeSettings`)
  and `ExportRecord` (`core/_export.py`) for what a reader or a data fetcher
  returns. Array fields named in `_ARRAY_FIELDS` are stored as read-only
  copies.
- A **result** is what a run or a computation returns: `Result`
  (`core/results/_base.py`) and its subclasses. A result carries no carriers;
  it carries its coordinates, its `run_settings` and `metadata`.

All three share one export protocol (`core/_export.py`). A type implements
the hooks:

```
_payload()      {name: (array, dims, unit)}, the primary array first
_coords()       {name: (values, unit[, dim])}
_table()        {column: 1-D array}               optional, tabular types
_export_attrs() scalar attributes to keep
_from_export(arrays, attrs)   classmethod: rebuild from the above
```

and gets, from `Exportable`: `values(name=None)` (a read-only view),
`to_dataframe()` (tabular types only), `to_xarray()` / `from_xarray()` (CF
`units` attributes) and `to_netcdf(path)`. `to_dict()` / `from_dict()` give
the plain-type form, which is also the `.npz` route:
`np.savez(path, **x.to_dict())`, then
`cls.from_dict(dict(np.load(path, allow_pickle=True)))`. A carrier's
`to_dict` stores every constructor field under its public class path, and
`from_dict` rebuilds it through the constructor, so the checks run on load.
xarray and pandas are an optional extra (`uacpy[xarray]`), imported where they
are used, never at module level.

### 5.2 One-line reprs

Every public class prints as one line, `ClassName(...)`, built with the
helpers in `core/_repr.py` rather than by hand:

- A quantity is its number in `:g` form, a space and its unit: `20 m`,
  `200 Hz`, `cp=1650 m/s`.
- An axis is its values when there are at most four (`depth 20 m`,
  `depths [10, 50] m`), else its count and extent (`12 depths 5–95 m`); a
  data array is its shape (`power 129×63 Pa²/Hz`). Any other list follows
  the same four-value cut.
- A nested object is described in its parent's words, never as a second
  constructor: `Bottom('sand' half-space, cp=1650 m/s, ...)`, not
  `Bottom(SeabedColumn(...))`; an object field shows its class name.
- The physics a user must not miss is in the line: the frequency, the
  seabed, the absorption (or `no absorption`) and the water density of an
  `Environment`, the ocean values of an `AbsorptionCoefficient`.
- A model or a configured tool shows only the constructor arguments that
  differ from their defaults (`Bellhop(beam_type='B')`); a record leaves
  out a field that is `None` or at its default.
- Aim for at most about 120 characters.

`axis`, `qty`, `extent` and `build` format the pieces; a dataclass record
inherits `FieldsRepr` and names its shown fields and their units in
`_REPR_FIELDS` / `_REPR_UNITS` (`FrozenRecord`, `ExportRecord` and every
`ResultTuple` already do); a tool inherits `SettingsRepr`. The run records
are the exception: `RunSettings` and an engine's `EngineSettings` print an
aligned multi-line block. `tests/test_reprs.py` pins the exact strings and
holds every public class to one line with no nested constructor.

---

## 6. Support systems

### 6.1 `FileManager`

`models/_workspace.py` holds `FileManager`, which allocates per-run scratch
directories. Stage 4 of the base's `run()` takes one per run
(`_setup_file_manager`); an engine's hooks receive it as `inputs.work_dir`. Pass
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
warnings.warn(msg, NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)   # preferred
warnings.warn(msg, NumericsWarning, stacklevel=2)                         # fixed depth
```

`skip_file_prefixes` walks out to the first frame outside the package, so it
survives a helper being inserted between the public function and the warn; a
hand-counted `stacklevel` does not, and is for the sites whose depth is fixed
by construction. **Never pass both** — the skip walk starts from the frame
`stacklevel` already selected, and the two compose into a frame further out
than either intends. A bare `warnings.warn(msg)` blames this package.
`tests/test_warning_attribution.py` gates all three rules.

### 6.3 Stack limit for the binaries — `uacpy/_stack.py`

SPARC-class solvers blow the default 8 MiB stack on first large
allocation. `_run_subprocess` prefixes every launch with
`stack_limit_prefix()` — `sh -c 'ulimit -s <hard>; exec "$0" "$@"'` — so
the child alone runs at the hard limit and the importing process is never
changed. A `preexec_fn` would avoid the shell hop but is unsafe once threads
exist, and models run from threads. Because the shell `exec`s the binary,
`_check_executable` re-creates the `OSError` a direct exec would raise (missing
file, no execute bit, unrecognised format) before the launch, so a broken
install still arrives as `ExecutableNotFoundError`. `UACPY_NO_STACK_RAISE=1`
drops the prefix.

The same module supplies the outer wrapper, `parent_death_prefix()` —
`setpriv --pdeathsig TERM --` on Linux, checked once per process — so the argv
is setpriv → the stack-limit shell → the binary. Binaries run in their own
session (`start_new_session`, so a timeout can reap the whole tree), which
means a Python process killed outright (SIGTERM with the default handler,
SIGKILL, a notebook kernel restart, a scheduler's walltime) runs no reaper and
the binary kept running under init; the parent-death signal ends it with its
launcher. The signal fires when the forking *thread* exits, and that thread
waits in `communicate` for the binary, so it cannot fire early. Where setpriv
is missing or too old (macOS, busybox, util-linux < 2.33) the prefix is empty
and the launch is exactly as before. Pinned by
`test_a_binary_dies_with_the_python_process_that_launched_it`.

### 6.4 Launching a binary — `uacpy/models/_launch.py`

Every engine binary runs as a `Launch` record through `run_launch`
(`PropagationModel._launch_binary`). The record names the command line, the work
directory, the child environment, the stale outputs removed before the launch,
the `.prt` the binary writes (removed first, its tail attached to a failure),
an optional `tolerate_exit` for a failure that is read anyway (Kraken's
field.exe teardown), and the checks run in order on the completed process: the
engine's fatal scans and warning passes over its marker tables
(`_STDOUT_FATAL_MARKERS` for OASES, `_COLLINS_STOP_MARKERS` for RAM's Collins
codes, `_FIELD_COMPLETION_MARKER` / `_FIELD_FATAL_MARKER` for field.exe) and
the outputs the run must leave. `_prt_launch` builds the record of a binary
that writes `<base>.prt` (the Acoustics Toolbox, OASES). `_run_subprocess` is
reached only through the runner (`TestEveryLaunchPassesOneRunner`).

Threads follow one rule, `_launch_single_threaded()`: in a `run_parallel`
worker Bellhop's C++ port gets `-1` and an OpenMP binary (RAM's mpiramS)
`OMP_NUM_THREADS=1`, the pool supplying the parallelism; outside a pool both
use the cores. An exported `OMP_NUM_THREADS` is passed through as given
(`openmp_thread_env`).

### 6.5 Exceptions and warning classes

Every class lives in `core/exceptions.py` and is re-exported at `uacpy` and
`uacpy.core`. Raise and warn with these, never a bare `ValueError` /
`TypeError` / `UserWarning`.

| Exception | Raised when |
|---|---|
| `UACPYError` | base of all of them (`except UACPYError` is the catch-all); takes a `remediation=` rendered as a "How to fix:" paragraph |
| `ConfigurationError` | the user's input is wrong: a value, a combination, a malformed environment, a file the user named |
| `UnsupportedFeatureError` | a legal request this engine cannot satisfy; `alternatives` lists what can |
| `InvalidDepthError` | a source or receiver depth outside what the engine resolves |
| `ExecutableNotFoundError` | a binary is absent or present but not runnable (`reason=`) |
| `ModelExecutionError` | a binary failed, timed out or never launched; keeps `stdout` / `stderr` / `return_code` |
| `FileFormatError` | a file an engine should have written is absent, truncated or malformed |
| `OutputContractError` | an engine's result is not what its `outputs` declaration states: a defect in the engine wrapper, never in the call |
| `DataFetchError` | the data layer cannot supply a value at a location, whatever the cause |

| Warning (all `UACPYWarning`) | Warned when |
|---|---|
| `NumericsWarning` | the discretisation limits the answer: aliasing, wrap-around, truncation, non-convergence, over budget. The remedy is a finer, longer or smaller computation |
| `ValidityWarning` | outside where a formula or fit holds; the value is still computed |
| `FallbackWarning` | something other than what was asked was used: ignored, clamped, defaulted, collapsed. The message names what was used |
| `ProvenanceWarning` | licence, attribution or calibration status; a replay that derives a value differently |
| `IOWarning` | an irregular file, engine log or service reply that was still read |

`UACPYWarning` itself is never warned: a site passes the class that names its
cause (`test_every_warning_names_its_cause`). A refusal message ends with a
full stop, and a refusal a user can reach raises a `UACPYError` subclass
(`tests/test_harmonisation_gates.py`). An engine's stage-3 conditions are not
warned where they are found: the resolver records a `Notice` on its Settings
record, and the base warns each once (§2.2), so `run_settings()` and `run()`
warn alike and `validate_inputs()` stays silent.

---

## 7. Shared processing — `acoustic_signal/`, `noise/`,
   `visualization/`

These are orthogonal to the model layer. They consume `Result`
objects (typically `Field`) or raw arrays.

**Five modules, one question each.** The package re-exports every name flat
(`uacpy.acoustic_signal.welch`), so these boundaries are for maintainers, not
callers:

- `acoustic_signal/generate.py` — *give me a signal*: parametric waveforms
  (Ricker, Gaussian, M-wave, chirps — the same alphabet as AT `cans.f90`
  where possible), coded sequences (m-sequences, BPSK probes) and noise built
  to a target spectrum.
- `acoustic_signal/spectral.py` — *measure this signal*: one estimator per
  statistic — `welch` and `constant_q` (→ `SpectralEstimate`,
  `scaling='density'` or `'spectrum'`), `sound_exposure` (the ISO 18405 band
  energy, which takes no window or overlap because a band sum needs every bin
  counted once and whole), and a histogram twin of each (→
  `ProbabilisticSpectralEstimate`). `spectral` imports `bands` and
  `cqt` inside the two estimator cores, since both import it at load
  time.
- `acoustic_signal/cqt.py` — the constant-Q transform and spectrogram,
  and the kernels `constant_q` resolves frequency with.
- `acoustic_signal/bands.py` — the band ladders (`decidecade_bands`,
  `octave_bands`, `standard_bands`, all from one builder) and the band integration `band_levels` and `sound_exposure` are
  written on.
- `acoustic_signal/timefreq.py` — the time-resolved views (`spectrogram`,
  `cwt`, `wigner_ville`, the analytic signal, cepstra).
- `acoustic_signal/spectrum_at.py` — `tone_phasor` and `waveform_spectrum_at`:
  a transform evaluated at the frequency asked for.
- `acoustic_signal/beamforming.py` — *what does this array see*: steering
  vectors, covariance, conventional and adaptive beamforming.
- `acoustic_signal/gathers.py` — the gather transforms (`fk_transform`,
  `taup_transform`, `radon_transform`, each with an inverse). Every one takes
  the receiver spacing `dx`, which is what separates them from the
  single-channel estimators.
- `acoustic_signal/windows.py` — the spectral windows the estimators and
  syntheses share.
- `acoustic_signal/detect.py` — *is my transmission in there*: matched filter,
  pulse compression, processing gain, ambiguity function.
- `acoustic_signal/frf.py` — *what did the channel do to it*: `frf_welch`,
  `etfe`, `periodic_etfe`, `lsfir` → `FRFResult`, and `FRF`, the class that
  holds one of them configured.
- `acoustic_signal/channel.py` — channel simulation, the transfer-function
  operations, and the arrival-list receptions (`simulate_arrival_reception`,
  and `simulate_arrival_grid` over a receiver grid on one clock).
- `acoustic_signal/delay_profile.py` — the power-delay-profile statistics and
  the channel regime.
- `acoustic_signal/dispersion.py` — modal group velocity with warping.
- `acoustic_signal/_synthesis.py` and `_results.py` — package-internal: the
  array half of the IFFT synthesis, and the behaviour every result tuple
  shares.

Everything is a pure function returning arrays or a small namedtuple. **All
plotting lives in `uacpy.visualization.plots`**, reached as `uacpy.plot`
(`plot_psd`, `plot_fk`, …) — the
`acoustic_signal`/`comms` modules import no matplotlib.
- `noise/ambient.py` — `wind_noise_level`, `WenzNoise`, and the per-mechanism
  submodels (wind, shipping, rain, turbulence, thermal) the composite selects
  between; `ship_radiated_noise.py` — ISO 17208 RNL and equivalent monopole
  source level from a measured pass-by, a different quantity from the
  shipping *ambient* term above; `marine_mammal.py` — Southall et al. 2019
  auditory weighting.
- `visualization/plots/` — every result/carrier plots via `.plot()`,
  which dispatches through one table (`_PLOTTERS`): `plot_result(result,
  env=…)` for a result, to the private per-type renderers (`_plot_rays`,
  `_plot_arrivals`, `_plot_mode_functions`, …), and `plot_carrier(carrier)`
  for a carrier, to the public `plot_environment`, `plot_ssp`,
  `plot_range_profile` and `plot_absorption`. Public free functions remain
  for the grid/flexible renderers (`plot_field`, `plot_transfer_function`,
  `plot_impulse_response`), alternate views
  (`plot_bottom_properties`, `plot_bottom_loss`, `plot_mode_wavenumbers`,
  `plot_modes_heatmap`, `plot_beam_power`),
  composition (`compare`, `compare_models`, `plot_overview`, maps), and the
  raw-array DSP/comms plotters.
- `visualization/style.py` — colour palette: field colormaps, sediment
  fill/hatch styles and source/receiver marker styles. No font or sizing
  presets — importing it leaves `rcParams` untouched. Touch this if you want
  to change the package look-and-feel globally.

**One public path per object: the facades.** `uacpy.plot`, `uacpy.acoustics`
and `uacpy.metrics` are THE documented paths of the plotters, the acoustics
helpers and the TL metrics, and docs, examples and the README name them.
`uacpy.visualization.plots`, `uacpy.core.acoustics` (with the absorption
formulas of `uacpy.core.absorption`) and `uacpy.core.metrics` are the
implementation modules those facades expose; package code imports the
implementation modules (layering: `core` must not import a root facade), and a
facade adds no name of its own. `uacpy.visualization` re-exports no plotter;
it holds `style` and the coastline backdrop (`land_polygons`,
`download_coastline`).

Convention: each result-type plotting function takes the `Result`
positionally + an optional `env=` for seafloor / surface overlays +
optional axis-control kwargs.

**`core` reaches up into `visualization`, through one function.**
`visualization/plots/` imports `uacpy.core` at module scope (it needs
`Environment`, `Field` and the rest to render them), and `.plot()` on a
carrier or a result calls back into it. Every such call, from `core`,
`core.results`, `acoustic_signal`, `comms` and `noise` alike, goes through
`uacpy.core._plotting.plotter(name)`, which imports `uacpy.visualization.plots`
and returns its public plotter of that name: `plot_carrier` for a carrier,
`plot_result` for a result, and the plotter a signal result declares.

That import sits **inside the function**, with a comment saying so.
Hoisting it to file scope makes
`import uacpy` raise `ImportError` from a partially initialised module,
because `uacpy/__init__` eagerly loads the core carriers and
`visualization/plots/` imports them back; the failure is immediate and
loud, and `tests/test_lazy_imports.py` catches it as well, since that test
asserts a cold `import uacpy` leaves matplotlib out of `sys.modules`.

Two consequences to keep in mind when touching either side. A plotter is
named by a string in the method that draws through it, so renaming one means
editing that string in the same change (`tests/test_carrier_plot_methods.py`
fails otherwise). And `core` cannot be read, split or vendored without the
plotting stack, even though nothing else in `core` needs matplotlib.
Removing the edge would make `core` a sink but would *not* make the package
graph acyclic: a 22-module SCC inside `uacpy.data` (counting every import
edge, function-local imports included — the same edge definition as the
edge above; on
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
- `_cache.py`, `_geo.py`, `_netcdf.py` — cache dir resolution,
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
- `_provenance_notice.py` — `one_provenance_notice`, the decorator that folds
  every provenance notice one fetch gives into one `ProvenanceWarning`; on
  `fetch_environment` and every transect fetcher that samples a point
  fetcher per waypoint.
- Per-dataset modules — bathymetry `gebco_local`, `gmrt_live`,
  `emodnet_bathy_live`; seabed `emodnet_local`, `diesing_local`,
  `globsed_local`, `crust1_local`, `graw_local`, `mars`, `pelagic`,
  `sediment_db`; water column `woa23_local`, `argo`, `glodap_local`;
  surface `sea_surface`, `seaice_local`, `wind_local`, `wind_live`,
  `waves`, `ww3_live`; plus `absorption.py` (site Francois–Garrison).

Conventions: a fetcher takes `base_url` where its endpoint is configurable,
`timeout` / `verbose` where it can reach the network, and none of them when it
is a pure lookup into an installed dataset (e.g. `fetch_seabed_density`); every
fetcher raises `DataFetchError`/`ConfigurationError` only, logs via `log_message`, and
emits user notices via `warnings.warn`.

**Adding a source.** Write a module exposing a fetcher with the standard trio,
then:

- *Sediment* → write the source's point fetcher, a `_<name>_pair(cached)`
  resolver returning `_along_track(point_fetcher, label)` (the transect is the
  point fetcher sampled along the track, `sediment.transect_fetcher`; a layered
  source such as CRUST1.0 returns its own transect), and add one
  `_BottomProvider` row to
  `environment._BOTTOM_PROVIDERS` (its `id` doubles as the source keyword;
  `in_auto=True` joins the `'auto'` chain, `in_cache_auto=True` the `'local'`
  one). The accepted keywords, fallback order and provenance id all derive from
  that list.
- *Sound speed* / *bathymetry* → add one `SourceProvider` row (id, backends
  cached twin first, point fetcher, transect fetcher or `None`) to
  `environment._SSP_CHAIN` / `bathymetry._BATHY_CHAIN` (`data/_chain.py`), and
  its id to the chain's `auto` order if `'auto'` should try it. The accepted
  keywords, the cache-first attempts and the fallback order derive from that
  row.
- *Absorption pH* → `environment._fetch_ph` is pH-source aware: it prefers the
  operational Copernicus BGC `ph` field on the Copernicus SSP branch
  (`copernicus.fetch_ph_operational`), else the cached GLODAP grid
  (`glodap_local.py`), else the model-default constant, then feeds
  `FrancoisGarrison.from_temperature_salinity` under `with_absorption` (GLODAP
  as `(depth, pH)` pairs; cache/best-effort, silent
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
runtime `ProvenanceWarning` whenever fetched, so they are never returned silently.

---

## 8. Tests (`uacpy/tests/`)

```bash
pytest                         # full suite, -n logical via xdist (pyproject default)
pytest -n 0                    # single-process for debugging
pytest -m "not slow and not requires_network"           # fast subset
pytest -m "not requires_binary and not slow and not requires_network"   # pure-Python dev tier
pytest uacpy/tests/test_bellhop.py::TestX::test_y -v
```

Markers (registered in `pyproject.toml`):

- `slow` — long broadband or large-grid runs.
- `requires_binary` — needs a compiled native binary under `uacpy/bin/`.
- `requires_oases` — needs the OASES binaries or the OASES sources under
  `uacpy/third_party/oases/` (both from `./install.sh --oases yes`).
- `requires_network` — hits a live external service (`uacpy.data`
  fetchers); deselected by default via `addopts`. A command-line `-m`
  replaces the `addopts` one, so each `-m` expression above ends in
  `and not requires_network`.
- `benchmark` — validates output against a closed-form analytic solution,
  a canonical published reference, or an independent reference solution
  (another engine only when the test says so).
- `convention` — pins repo conventions rather than runtime behaviour
  (docstring prose, source-convention sweeps, repr snapshots); a failure
  signals doc/convention drift, not a runtime defect.

The composed dev tier `-m "not requires_binary and not slow and not requires_network"` is the fast
pure-Python loop. The count moves every round, so it is not quoted here;
the collected-case total comes from

```bash
uacpy_venv/bin/python -m pytest uacpy/tests --collect-only -q -n 0 | tail -1
```

and adding `-m "not requires_binary and not slow and not requires_network"` to that command gives
the tier's share (`requires_oases` tests count as
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
constants are masked, concentrated in the `test_io_*.py` files and
`test_oass.py`) — are `@pytest.mark.parametrize` candidates, to be
folded file-by-file with per-file collected-case parity as the gate.
The `convention` marker selects the four modules that pin repo conventions
rather than runtime behaviour: `test_packaging.py`, `test_documentation.py`,
`test_architecture.py` and `test_lazy_imports.py` each set it as
`pytestmark`. The doc-prose, source-sweep and repr-snapshot pins scattered
through the mixed test modules are unmarked, so deselecting `convention`
removes those four modules and not the whole category. Marking the rest
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

It also holds the one engine list and the run seams tests share:

- `engine_params(value=)` gives one `pytest.param` per registered engine,
  with its markers; `build_engine(name, **kw)` builds one with its registry
  `example_kwargs`; `engine_names()` and `engine_entry(name)` answer the
  rest. A table of expected values per engine asserts it covers
  `engine_names()`.
- `launch_spy(target, then=None)` replaces the method that starts a binary
  and raises `LaunchReached` (or calls `then`); `stop_at(target, stage)`
  stops a run as it enters a stage (`'project'`, `'settings'`, `'write'`,
  `'launch'`, `'read'`, `'result'`) with `StageReached`; `stage_spy`
  records the calls of a stage hook. They are the only place tests name those
  hooks, so renaming one is one edit there.

Moving a function to another module re-binds every name it reads from its
module to the destination's. `python -m uacpy.tests._free_names
OLD.py:QUALNAME NEW.py:QUALNAME` lists the names that resolve differently
(exit 1) or none (exit 0); keep the old file aside to run it.

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

### 8.1 The convention gates and how to fix each

These tests pin the structure, not the physics. Each one's failure message
names what to change; the table says where to look first.

| Gate | Fails when | Fix |
|---|---|---|
| `test_engine_conformance.py` | a registered engine breaks the run contract: a positional knob, a knob not stored as `self.<name>`, no stage-2 `_check_knobs`, a refusal that differs between `validate_inputs` / `run_settings` / `run`, a result unlike its `outputs` declaration (`OutputContractError`), a knob missing from the record | follow §2 and §3; the test name says which rule |
| `test_supported_modes.py`, `test_capability_flags.py` | an engine's modes or flags differ from the expected tables, or a table misses a registered engine | change the `spec`, or the table when the change is intended (§3, step 5) |
| `test_collapse_policy.py`, `test_engine_conformance.py` (saved records) | a per-engine collapse row or a saved run-settings record is missing; a record saved before a new Settings field no longer loads | §3, step 5; §13.1, step 3 |
| `test_exceptions.py` | a public reader in neither absent-file list; a warning class missing from `CAUSES` or from the §6.5 table | §13.4, step 3; §13.5 |
| `test_metadata_file_paths.py` | a metadata key nothing documents, or a documented key nothing writes | add or drop the `_DOCUMENTED_METADATA` row |
| `test_lazy_imports.py` | `import uacpy` loads scipy / matplotlib; a lazy table entry fails to resolve; a `TYPE_CHECKING` block differs from its lazy table; `core` imports `visualization` outside a function body | restate the name in both tables; keep heavy imports inside functions (§7) |
| `test_architecture.py::test_layering` | an import points up the layer order, a new import cycle, a private name crossing layers | §12 |
| `test_architecture.py::test_every_warning_names_its_cause` | a `warnings.warn` passes `UserWarning`, `UACPYWarning` or no category | pass the cause class (§6.5) |
| `test_warning_attribution.py` | a warn site with neither `skip_file_prefixes=USER_FRAME_SKIP` nor `stacklevel`, or with both | §6.2 |
| `test_architecture.py`, plotter tables | a result class with no row, or two rows, in `_PLOTTERS`; a plotter that branches on a model name | add the row (§13.3) |
| `test_architecture.py::test_every_refusal_pin_names_its_refusal` | a `pytest.raises` with no `match=` | match the load-bearing fragment (§8) |
| `test_architecture.py::test_every_deck_writer_names_a_knob_as_its_model_does` | a writer parameter spelled as the deck spells it rather than as the knob | rename the parameter to the knob name |
| `test_harmonisation_gates.py` | API spelling: `max_range` instead of `range_max`, two keywords for one argument, a dB reference not called `ref`, a writer default the model resolves, a refusal without a full stop, a user error that is not a `UACPYError`, package code reading a metadata key | rename or reword as the message says |
| `test_packaging.py` | an unexpected or missing package (`_EXPECTED_PACKAGES`), the CI lint, a comment pinning a line number in a uacpy file, a vendored citation that no longer resolves, a private module or name crossing a package boundary undeclared, a name stating history (`still`, `previously`, ...) or a kind of code (`utils`, `helpers`, ...), a test file named after a session | §10; the message names the list to update |
| `test_documentation.py` | a dead link or anchor, a sample that does not parse or names something the package lacks, a capability matrix or a documented default that differs from the code, sections out of order | fix the page; the code is the reference |
| `test_export_protocol.py` | a result that does not round-trip, a record with an array field outside `_ARRAY_FIELDS` | §5.1, §13.3 |
| `test_methods_delegate_to_array_level_computation.py` | a method that computes something generic inline instead of wrapping an exported array-level function | move the arithmetic into a function and call it |

The `convention` marker selects the four modules that pin conventions
wholesale (`test_packaging.py`, `test_documentation.py`,
`test_architecture.py`, `test_lazy_imports.py`); the other rows above are
unmarked and run in every tier.

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
- `bellhopcuda/` — git submodule pinned to a commit on uacpy's fork
  (`ErVuL/bellhopcuda`): upstream `v1.5` plus the Francois-Garrison fix,
  offered upstream as a pull request. `install.sh` pins the SHA
  (`BELLHOPCUDA_COMMIT_SHA`); bump it and the submodule pointer together.
  The SHA in `install.sh` is the pin and the gitlink follows it:
  `test_packaging.py` compares the two (`git ls-tree HEAD` against the
  parsed SHA) and fails until they agree. To move the gitlink onto a new
  pin: `git -C uacpy/third_party/bellhopcuda fetch origin && git -C
  uacpy/third_party/bellhopcuda checkout <sha> && git add
  uacpy/third_party/bellhopcuda`.
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
- `UACPY_FORTRAN_ARCH_FLAGS` (environment variable) — the architecture flags
  for every Fortran build except OASES. Unset, they are `-march=native
  -mtune=native` (`-mcpu=native` on aarch64), so the OALIB, mpiramS, ramsurf
  and ramgeo binaries run only on CPUs with the build host's instruction set;
  a tree shared across machines, copied into a container image, or built on a
  newer CPU than it runs on stops with an illegal-instruction signal inside the
  binary. `UACPY_FORTRAN_ARCH_FLAGS="-march=x86-64-v3" ./install.sh --force`
  builds for a CPU family instead. The flags used are written to
  `uacpy/bin/BUILD_INFO.txt` (`arch` line).

---

## 10. Coding conventions

- Public API uses metres / Hz / m/s / g/cm³ / dB/wavelength. Suffix
  the attribute name (`_km`) when not metres.
- Constructor-only model configuration — no `set_params()`.
- Use the typed exception hierarchy (`core/exceptions.py`), not bare
  `ValueError`.
- Promote any new "magic number" to a named constant in the module of its
  owner (`core/constants.py` only for the physics reference every layer
  reads).
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
  `test_no_undocumented_private_name_crosses_a_package_boundary`. An
  underscore-prefixed *module* imported from another package is declared,
  with its reason, in `PACKAGE_INTERNAL_MODULES`
  (`tests/_internal_modules.py`), enforced by
  `test_a_private_module_crosses_a_package_boundary_only_when_declared`. Read
  the lists there rather than a summary here — a count or a set of directions
  written into this prose is a second copy that nothing checks, and it drifts.
  To add a cross-package private, add it to that list with its reason; the
  gate fails in both directions, so an entry that stops being imported has to
  be dropped too. Renaming a private that appears in the list is a
  cross-package change and needs its consumers updated in the same edit. The
  gate reads `from … import _name` only — a private reached as an attribute
  (`module._helper()`) does not appear in it.
- A subpackage's `__all__` lists functions, classes and constants, never its
  submodules; a submodule stays reachable as an attribute
  (`uacpy.io.oalib_reader`). `tests/test_architecture.py` gates the
  subpackages it names.
- No backwards-compatibility shims. Change code directly; uacpy is
  pre-1.0 and explicitly LLM-bootstrapped per the README roadmap.
- One PR = one logical change. Mention which physics regime / file
  format the change targets in the title.

---

## 11. Cutting a release

uacpy is distributed from GitHub only: the `Private :: Do Not Upload`
classifier in `pyproject.toml` keeps it off PyPI, because no wheel can carry
the model binaries `install.sh` compiles. A release is a tagged commit that a
fresh clone can install, and these steps get it there:

1. **Version.** Set `__version__` in `uacpy/_version.py`; `pyproject.toml`
   reads it from there.
2. **bellhopcuda pin.** The submodule gitlink and `BELLHOPCUDA_COMMIT_SHA` in
   `install.sh` name the same commit
   (`test_the_submodule_gitlink_records_the_sha_install_sh_pins`). Move the
   gitlink with `git -C uacpy/third_party/bellhopcuda checkout <sha>` and
   `git add uacpy/third_party/bellhopcuda`.
3. **Stage every file the code needs.** `git commit -a` records edits and
   deletions but not new files. Before committing, `git status --short` lists
   each untracked (`??`) module, example and test: add every one that tracked
   code imports, and `git rm` the deleted ones. Working notes and scratch
   directories stay out of the commit.
4. **Gate the committed tree.** Run the full suite on the commit, not on the
   working tree it came from, then check that a fresh clone imports:
   `git clone --recurse-submodules <repo> <dir> && cd <dir> &&
   python -m venv v && v/bin/pip install -e . && v/bin/python -c "import uacpy"`.
5. **Artifacts.** Build an sdist or wheel only from a fresh clone, or after
   `rm -rf build/`: setuptools copies sources into `build/lib` and never
   deletes what is already there, so a wheel built next to an old `build/`
   ships every module removed since that tree was made.
6. **Tag.** `git tag v<version>` on the gated commit, and push the tag.

---

## 12. Layering

A module's layer is its subpackage (`core.results` and `core.acoustics`
counted apart from `core`). A layer imports only its own rank or below:

```
1  core, core.acoustics, _log, _stack, _version
2  acoustic_signal
3  comms, noise
4  core.results
5  sonar, analytic
6  io, data
7  models
8  parallel
9  visualization
10 the top-level modules: uacpy/__init__.py, acoustics, metrics, plot
```

`tests/test_architecture.py::test_layering` enforces it (`_LAYER_ORDER`) with
three more rules: no module-level import cycle; `core` imports nothing from
`models`, `io`, `visualization` or `data` at module level; and every import
that points up the order, and every private name imported across a layer, is
in an allow-list (`_UPWARD_IMPORTS`, `_PRIVATE_NAMES_ACROSS_LAYERS`). The
allow-lists only shrink: a new entry fails the test, and so does an entry
whose import has gone.

When the gate fails, move the code down to the layer that needs it, or pass
the value in as an argument. The one sanctioned upward seam is `.plot()`:
`core` reaches `visualization` through `plotter()` in `core/_plotting.py`,
imported inside the function (§7). A package boundary is a separate rule
(§10): a private module or name crossing one is declared in
`PACKAGE_INTERNAL_MODULES` or `_CROSS_PACKAGE_PRIVATES`.

---

## 13. Recipes

Adding an engine is §3. Each recipe below names the test that fails when a
step is skipped.

### 13.1 A new knob

1. Add a keyword-only constructor parameter, store it as `self.<name>`, and
   validate it in the engine's `_check_knobs` with the helpers in
   `models/_knobs.py` (`test_copy_carries_every_constructor_knob`,
   `test_every_constructor_knob_is_keyword_only`).
2. Document it, with its default, in the engine's one Parameters section:
   the class docstring or the `__init__` docstring, whichever the engine
   uses (`test_every_model_constructor_parameter_is_documented` reads both).
   Then add its row to the Default column of
   `docs/models/<engine>.md`. `tests/test_documentation.py` checks every row
   the page has against the constructor, and cannot see a row that is
   absent.
3. If the run derives it when it is `None`, resolve it in
   `_resolve_engine_settings` and add a field of the **same name** to the
   engine's Settings record: the record then states the derived value, and
   `from_run_settings` pins it. If it is derived per launch, declare it in
   `_per_launch_knob_values`; if the run takes it from the call or composes
   it from other knobs, list it in `_UNPINNED_FIELDS`. A knob that cannot
   change a result (scratch, logging) belongs in `_HOST_KNOBS`
   (`test_the_record_states_every_knob_as_given`).

   A new Settings field takes a **default**, and the default is the value
   that reproduces what the run computed before the field existed: a record
   saved before then has no such key, and loads with the field at its
   default. A field with no default makes every earlier saved record fail to
   load, which `test_a_saved_run_settings_loads_from_the_class_path_it_names`
   catches on the saved records in `tests/data/run_settings_by_engine.json`.
   Never rewrite a saved record to make that test pass: it stands for the
   files users saved.
4. Spell it by the house rules: `<quantity>_max` / `_min`, `dB` never `db`,
   `_mps` for m/s and `_s` for seconds, a default equal to a package constant
   written as that constant (`tests/test_harmonisation_gates.py`,
   `tests/test_architecture.py`). The deck writer that carries it takes a
   parameter of the same name and no default of its own.

### 13.2 A new run mode for an existing engine

Add the mode to `spec.modes` and an `OutputSpec` to `outputs`, handle it in
the stage hooks, and update the engine's row in `_EXPECTED`
(`tests/test_supported_modes.py`) and the two capability matrices (§3,
step 7). Then check the per-mode facts the base reads: whether the mode
stacks source depths in one launch (`traits.native_multi_depth_modes`, and
`_EXPECTED_MULTI_SOURCE_DEPTH`), whether TIME_SERIES reads the frequency
argument, and whether the mode returns a field (`_FIELD_MODES` in
`models/base.py`, which decides whether a multi-depth source is looped or
refused). The `compute_*` method for that mode suggests the engine from the
registry with no further edit; `compute_tl` asks for `COHERENT_TL`, so an
engine that adds only `INCOHERENT_TL` is not offered there. Making the new
mode the engine's default (the first entry of `spec.modes`) changes the
record a default run saves, so the engine's saved record (§3, step 5) no
longer matches it: that is a change to what users' saved files mean, and
needs its own decision.

### 13.3 A new result class

1. Subclass `Result` in `core/results/<name>.py`, implement the export hooks
   (§5.1) plus `to_dict` / `from_dict`, and carry no carriers.
   `ReflectionCoefficient` (`core/results/reflection.py`) is the smallest
   template. The identity block every result shares goes through the base's
   `_identity_dict` / `_identity_from_dict` in `to_dict` / `from_dict`, and
   `_identity_from_attrs` in `_from_export`.
2. Export it from `core/results/__init__.py`, and add it to the
   `uacpy.core.results` block of `_EXPORTS` in `core/__init__.py` and to that
   file's `TYPE_CHECKING` block (`__all__` derives from `_EXPORTS`). `uacpy`
   picks it up from `uacpy.core.__all__`, and its own `TYPE_CHECKING` block
   restates it (`tests/test_lazy_imports.py`).
3. Give it one row in `_PLOTTERS` (`visualization/plots/__init__.py`):
   `(ResultClass, plotter, draws_env)` (`TestEveryResultTypeHasExactlyOnePlotterRow`).
   The plotter is a function in the `visualization/plots/` module of its
   subject, `_plot_<name>(result, ax=None, *, figsize=..., title=None)`
   returning `(fig, ax)`, imported into `plots/__init__.py` with the result
   class. `draws_env` is `True` only when the plotter takes `env=` and draws
   the environment under the result; `.plot(env=...)` on any other type is
   refused.
4. Add a maker to `MAKERS` in `tests/test_export_protocol.py`. Nothing fails
   if this step is skipped: the round-trip test covers only the classes that
   table names.
5. An engine that returns it names it in `OutputSpec(result_type=...)`.

### 13.4 A new reader

1. Put the public reader in `io/<format>_reader.py`; a raw parser the engines
   share goes in `io/_parsers.py`.
2. Return an `ExportRecord` subclass (`@dataclass(frozen=True, eq=False)`)
   listing every array field in `_ARRAY_FIELDS` and any tabular ones in
   `_TABLE_FIELDS`. `TestEveryRecordFreezesEveryArrayField` finds every
   record subclass by itself.
3. A file an engine should have written and did not is a `FileFormatError`; a
   file the user named is a `ConfigurationError`. Decorate the reader with
   `@typed_format_error` and start it with `require_model_output(path,
   name)` or `require_user_input(path, name)` (all three in
   `io/_fortran_helpers.py`), which raise the right one. Add the reader to
   `_model_output_readers()` or `_user_input_readers()` in
   `tests/test_exceptions.py`;
   `test_every_public_reader_states_what_an_absent_file_is` fails until every
   public `read_*` is in one of them. A readable but irregular file is an
   `IOWarning`.
   The record also implements the export hooks (`_payload`, `_coords`, §5.1);
   nothing checks that a record exports.
4. Export it from `io/__init__.py`, and update `docs/guide/io.md`: the
   public-name count in its subtitle line and one reference-table row per
   new public name (the reader *and* its record class).
   `tests/test_documentation.py` holds both to the export list.
5. Cite the format's vendored source as `file.f90:NNN` (checked); never a line
   of a uacpy file (§10).

### 13.5 A new warning or exception class

1. Subclass `UACPYWarning` (or `UACPYError`) in `core/exceptions.py`, with a
   docstring stating the cause, and add it to `__all__`.
2. Add it to the `uacpy.core.exceptions` block of `_EXPORTS` in
   `core/__init__.py` and to that file's `TYPE_CHECKING` block (`__all__`
   derives from `_EXPORTS`), and restate it in `uacpy/__init__.py`'s
   `TYPE_CHECKING` block.
3. For a warning, add it to `CAUSES` in
   `tests/test_exceptions.py::TestWarningClasses`, which fails until it lists
   every warning class `core/exceptions.py` defines. The cause sweep
   (`test_every_warning_names_its_cause`) reads the classes from
   `core/exceptions.py` itself. For an exception, add an instance to the
   pickling list in `tests/test_exceptions.py`, so `run_parallel` returns it
   intact.
4. Add its row to the tables in §6.5; for a warning,
   `test_dev_md_tabulates_every_warning_class` fails until the row is there.
