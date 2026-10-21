# uacpy Documentation

**uacpy** is a unified Python interface to the standard underwater-acoustics
propagation models (Bellhop, Kraken, Scooter, RAM, SPARC, OASES) plus toolkits
for signal processing, sonar, communications, and ambient noise, and a
real-world-data layer that builds an `Environment` from GPS coordinates.

This guide is example-driven: each concept is introduced briefly, then shown
with a minimal runnable snippet. Every public class and function also carries a
docstring (`help(uacpy.Bellhop)`), `uacpy/examples/` holds the complete scripts
listed in §17,
and the internals are documented in `docs/DEV.md`.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Quick Start](#3-quick-start)
4. [Core Concepts](#4-core-concepts)
5. [Environment](#5-environment)
6. [Source and Receiver](#6-source-and-receiver)
7. [Propagation Models](#7-propagation-models)
8. [Results](#8-results)
9. [Visualization](#9-visualization)
10. [Signal Processing](#10-signal-processing)
11. [Sonar Performance](#11-sonar-performance)
12. [Digital Communications](#12-digital-communications)
13. [Ambient Noise](#13-ambient-noise)
14. [Standards-Based Metrics](#14-standards-based-metrics)
15. [Units & Conventions](#15-units--conventions)
16. [Troubleshooting](#16-troubleshooting)
17. [Examples Index](#17-examples-index)
18. [Parameter Reference](#18-parameter-reference)

## 1. Introduction

uacpy is a unified Python interface to the standard underwater-acoustics
propagation codes — Bellhop, Kraken, Scooter, RAM, SPARC, and the OASES
suite — wrapping their vendored Fortran/C/CUDA solvers behind one
consistent, NumPy-native API. You describe the ocean once (an
`Environment`, a `Source`, a `Receiver`), then hand it to whichever model
suits the physics; every model returns the same typed `Result` objects, so
swapping a ray tracer for a normal-mode or parabolic-equation solver is a
one-line change. Around the propagation core sit orthogonal toolkits for
signal processing, sonar, communications, and ambient noise, plus a
real-world-data layer that builds an `Environment` straight from GPS
coordinates.

What it covers:

- **Propagation models** — ray (Bellhop), normal modes (Kraken),
  wavenumber-integration (Scooter/OASES), parabolic equation (RAM),
  time-marched FFP (SPARC), and plane-wave reflection (Bounce/OASR).
- **DSP & sonar** (`acoustic_signal`, `sonar`) — waveforms, beamforming,
  matched filtering, the sonar equation, matched-field localization.
- **Communications** (`comms`) — modems including a bit-exact NATO JANUS.
- **Ambient noise & metrics** (`noise`, `metrics`) — Wenz spectra, plus
  ISO/UNESCO/Southall standards-based metrics.
- **Real-world data** (`data`) — bathymetry, sound-speed, and seabed
  fetched from public datasets for a given location and date.

Conventions throughout: metres, Hz, m/s, g/cm³, attenuation in
dB/wavelength. Depth is positive **down**; sea-surface altimetry is
positive **up** (z = 0 at the mean surface).

## 2. Installation

uacpy is a Python package plus a set of native solver binaries you compile
once. From the repository root:

```bash
pip install -e .           # editable install
./install.sh -y           # compile native binaries into uacpy/bin/ (gitignored)
```

The `test` and `dev` extras (`pip install -e ".[dev]"`) add the test and lint
toolchain; a user does not need them.

`install.sh` builds Bellhop, Kraken, Scooter, SPARC, Bounce and the RAM family
(mpiramS, rams0.5, ramsurf1.5, ramgeo1.5), and (by default) downloads and
builds the OASES suite from MIT. Useful flags:

- `--bellhop fortran|cxx|cuda` — pick the Bellhop backend to build
  (`cuda` requires a CUDA toolkit; the Fortran build is the safe default).
- `--oases yes|no` — include or skip OASES (academically licensed, fetched
  at install time, never redistributed).
- `--force` — rebuild even if binaries already exist.
- `--no-models` — pure-Python install, skipping all native builds.

Run `./install.sh --help` for the full flag list. Always use the project
virtualenv for execution. `uacpy.__version__` reports the installed package
version.

Check what got built with `python -m uacpy`: it prints the version and one
row per binary each engine can launch (every backend, and Kraken's
`field.exe`) — whether it was found and where, and the backend a construction
with no `backend=` resolves — then exits 0 when every engine has a binary and
1 otherwise. An optional install not made (OASES, `./install.sh --oases no`)
is named but does not fail the check. Found means on disk and executable: no
binary is run, so a found build can still fail on the machine (`bellhopcuda`
on a host with no CUDA device, which a `backend=None` run replaces with the
next installed build). From Python the table is
`uacpy.models.engine_status()`, an `EngineStatus`: `to_dataframe()` gives one
row per binary, and `missing_engines(*, optional_installs=False)` names the
engines with no binary found (those of an optional install not made only with
`optional_installs=True`). A model whose `backend=` was left `None` names the
build it resolved in its repr (`Bellhop(backend='cuda')`); Kraken and RAM
choose their backend per environment at run time, and `result.backend` names
the one that ran.

## 3. Quick Start

A complete transmission-loss calculation: build the environment, place a
source and a receiver grid, run a model, plot the field.

```python
import numpy as np
import matplotlib.pyplot as plt
import uacpy

# A Pekeris waveguide: 100 m of isovelocity water over a fluid seabed.
env = uacpy.Environment(
    name="Pekeris",
    bathymetry=100.0,          # flat bottom at 100 m (positive down)
    ssp=1500.0,                # isovelocity sound-speed profile (m/s)
    bottom=uacpy.BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1600.0,    # m/s
        density=1.5,           # g/cm³
        attenuation=0.5,       # dB/wavelength
    ),
)

source = uacpy.Source(depths=50.0, frequencies=100.0)   # 50 m deep, 100 Hz
receiver = uacpy.Receiver(
    depths=np.linspace(0, 100, 101),       # depth axis (m)
    ranges=np.linspace(100, 10_000, 200),  # range axis (m)
)

field = uacpy.Bellhop().run(env, source, receiver)

field.plot(env=env, source=source)
plt.show()

# field is a Field of COMPLEX PRESSURE in Pa, on a (depth × range) grid.
# .dB is what turns it into transmission loss (.tl is the same view,
# defined only on pressure fields); .p keeps the phase.
print(field.dB.shape)                 # (101, 200)
print(field.at(depth=50.0).dB.shape)  # (200,) — TL vs range at source depth
```

`compute_field` is the convenience wrapper for the pressure field; it is
exactly `model.run(env, source, receiver, run_mode=RunMode.COHERENT_TL)`.

It returns the **coherent** complex pressure field, `unit='Pa'`, phase intact —
`.dB` (`-20·log10|p|`) is the step that makes it a loss. Two consequences. You
can beamform or matched-filter the result, because the phase is still there
([§8](#8-results)). And a coherent field crosses any threshold
at every interference null, so it is the **wrong** input to a detection-range
budget — pass `run_mode=RunMode.INCOHERENT_TL` for those, and see the
[sonar guide's gotchas](docs/guide/sonar.md#10-gotchas) for how far apart the
two answers land.

The model classes are re-exported at the top level (`uacpy.Bellhop`, …);
the plotters live on `uacpy.plot`, the acoustics helpers on `uacpy.acoustics`
and the TL metrics on `uacpy.metrics`. `dir(uacpy)` is the full index.

### Which call for which task

Several tasks have more than one route; each pair below differs in what it
returns, and the left column is the one to reach for first.

| Task | Use | Not the same as |
|---|---|---|
| A TL map | `field = Model().run(env, source, receiver)`, then `field.plot(env=env)`; `field.dB` is the loss | `compute_field` is the same call; it returns complex pressure too |
| TL for a detection budget | `run(..., run_mode=RunMode.INCOHERENT_TL)`; on a ray model, check `sonar.detection_annuli` too, since a caustic spike is a crossing as well ([sonar §10](docs/guide/sonar.md#10-gotchas)) | the coherent default, whose interference nulls push detection ranges out |
| A pulse at one receiver | `run(..., run_mode=RunMode.TIME_SERIES, source_waveform=w, sample_rate=fs)`; from an `H(f)` you already hold, `H.to_time_trace(depth=z, range=r, source_waveform=w, sample_rate=fs)` | `H.to_time_trace(depth=z, range=r, sample_rate=fs)` with no waveform, which is that cell's band-limited impulse response |
| A pulse at every receiver | `H.synthesize_time_series(source_waveform=w, sample_rate=fs)` on a `Field`; `arr.synthesize_time_series(source_waveform=w, sample_rate=fs)` on `Arrivals` — both return a `Field` at `fs` | `H.synthesize_time_series(..., t_start='reduced')`, each range on its own record, for a section whose travel times outspan `1/Δf` |
| An impulse response | `H.impulse_response(fs)` on a `Field`; `arr.impulse_response(fs)` on `Arrivals` (the received amplitudes, absorption at every frequency) — every cell's `h(t)` at `fs`, each range on its reduced-time record | `H.to_time_trace(depth=z, range=r, sample_rate=fs)` with no waveform, one cell of it on absolute time; `arr.amplitudes`, the geometric column, which carries no volume absorption |
| A long signal (a recording) through a modelled channel | `ir = H.impulse_response(fs)` (or `arr.impulse_response(fs)`), then `ir.at(depth=z, range=r).apply(source_waveform=x, sample_rate=fs)`; blocks too long to hold: `cell.apply_blocks(blocks, sample_rate=fs)` on that cell, whose output starts at `cell.stream_start` | `synthesize_time_series`, whose `1/Δf` record must hold the whole signal (refused otherwise) |
| A modem channel from arrivals | `arr.channel_taps(symbol_rate, fc=fc)` | — |
| SEL over a grid | `H.sound_exposure_level(source_waveform=w, sample_rate=fs)`, in dB | `acoustic_signal.sound_exposure`, which is per band and linear (Pa²·s) |
| Weighted SEL and the range a criterion is reached | `sel = H.sound_exposure_level(source_waveform=w, sample_rate=fs, weighting='VHF')`, then `sonar.detection_ranges(sel.ranges, signal_excess_dB=sel.at(depth=z).data - EXPOSURE_THRESHOLDS['VHF']['impulsive']['TTS']['SEL'])`: the outermost range the map reaches the criterion (`inf` past the grid, `nan` never); the whole recipe is [noise guide §8](docs/guide/noise.md#a-noise-impact-assessment-end-to-end) | the unweighted map, which goes with the peak criterion (`H.peak_sound_pressure_level`), not the SEL one |
| Moving-source exposure | a single-event SEL map's `sel.track_exposure(positions, closest_approach=d)` (`along=` for the plan view, `.plot()`); a continuous source is one event per `Δt`: its received-level map (`H.broadband_loss(source_spectrum=S).at_source_level(SL)`) takes `level.track_exposure(..., event_duration=Δt)`, each event's SEL the level plus `10·log10(Δt)` ([results §6b](docs/guide/results.md#6b-weighted-accumulated-and-moving-source-exposure)) | `acoustics.sum_levels_dB`, which adds events heard at fixed ranges |
| Detection range | `sonar.detection_ranges_field(se)` on a signal-excess `Field` (a float for a cut, one per depth for a map); `sonar.detection_ranges(ranges, signal_excess_dB=se)` on arrays | — |
| Read a `.shd` file | `io.read_shd_file(path)` → `Field` | `io.read_shd_bin` / `read_shd_asc`, which return the raw file record |
| Compare models on a plot | `plot.compare(fields)` overlays 1-D cuts; `plot.compare_models(fields)` draws 2-D maps side by side | — |
| Beamform a field | `acoustic_signal.beamform_field(p, x, angles_deg, f)` → a `BeamformedField`, beam power with `.plot()` | its `.snr_dB(source_level_dB, noise_level_dB)`, the SNR per look of a sonar budget |
| Matched-field processing | `sonar.bartlett(R, replicas)` / `sonar.mvdr(R, replicas)` | `acoustic_signal.bartlett` / `mvdr(R, weights)`, the same estimators over rows of element weights; a star import of both packages rebinds one pair |

## 4. Core Concepts

The whole library is organised around one pipeline:

```
Environment + Source + Receiver  →  Model.run()  →  Result
```

- **`Environment`** — the ocean: bathymetry, sound-speed profile, surface,
  bottom, volume absorption (§Environment).
- **`Source`** — emitter depth(s) and frequency(ies).
- **`Receiver`** — a grid (depth × range) of hydrophones; paired samples (a
  track) are indexed out of a grid run (§6).
- **`Result`** — a typed object carrying the output (§Results).

### The model contract

Every model subclasses `PropagationModel` and shares one fixed `run`
signature:

```python
def run(self, env, source, receiver, run_mode=None, *,
        frequencies=None,        # broadband sweep override for source.frequencies
        source_waveform=None,    # real source pulse, for TIME_SERIES
        sample_rate=None,        # Hz, for TIME_SERIES
        output_duration=None,    # TIME_SERIES record length (s)
        t_start=None):           # TIME_SERIES first sample (s after emission)
    ...

result = model.run(env, source, receiver, run_mode=RunMode.COHERENT_TL)
```

There is **no `**kwargs`** — an unrecognised keyword raises `TypeError` at
the call site. Every model takes exactly these parameters and no others.
The one sanctioned extra belongs to a *different method*:
`Bellhop.run_with_bounce`'s BOUNCE window (§7 — the reflection table is
tabulated per call, not per Bellhop instance). Everything that *tunes* a model
— beam type, PE step sizes, Padé order, array geometry — is a **constructor**
argument, not a `run()` argument (see *Constructor-only configuration* below).

### The `compute_*` convenience family

`run(run_mode=…)` reaches every mode; each of the eleven `compute_*` methods
is `run()` with its mode fixed. It first refuses a model that does not
declare the mode — `UnsupportedFeatureError` naming the engines that do —
then forwards the call to `run()`. Settings stay on the model
(`Kraken(n_modes=…)`, `Bellhop(n_beams=…)`): no wrapper takes a per-call
override of one.

| Wrapper | Runs | Returns | Models | Extra keywords |
|---|---|---|---|---|
| `compute_field` | `COHERENT_TL`, or `INCOHERENT_TL` / `SEMICOHERENT_TL` via `run_mode=` | `Field`: the coherent complex pressure (`unit='Pa'`; `.dB` is the TL); real dB in the incoherent modes and from OAST | Bellhop, Kraken, OASP, OAST, RAM, Scooter (`INCOHERENT_TL`: Bellhop, Kraken; `SEMICOHERENT_TL`: Bellhop) | `run_mode=` |
| `compute_rays` | `RAYS` | `Rays`: the traced ray fan | Bellhop | — |
| `compute_eigenrays` | `EIGENRAYS` | `Rays`: the rays that reach the receivers | Bellhop | — |
| `compute_arrivals` | `ARRIVALS` | `Arrivals`: delay, amplitude and phase of each path at each receiver | Bellhop | — |
| `compute_modes` | `MODES` | `Modes`: wavenumbers `k` and depth eigenfunctions `phi` | Kraken | takes no `receiver` |
| `compute_reflection` | `REFLECTION` | `ReflectionCoefficient`: the plane-wave seabed reflection table | Bounce, OASR | — |
| `compute_time_series` | `TIME_SERIES` | `Field` of real p(t) on a `time` axis | Bellhop, Kraken, OASP, OASSP, RAM, SPARC, Scooter | `source_waveform=`, `sample_rate=`, `output_duration=` (record length; on the IFFT engines — Kraken, Scooter, RAM, OASP, OASSP — one period of a circular record: a trace still ringing at its end, a record ending before the slowest arrival or a receiver range span longer than the record gives one warning that the later response folded onto it), `t_start=` (first sample, s after emission; `None` = a tenth of the record (or `4/bandwidth`, whichever is longer) before the estimated earliest arrival on the IFFT engines, a tenth of the pulse before it on Bellhop's arrival reception; a value overrides it on every engine but SPARC, which warns and ignores it because its record starts at emission), `frequencies=` |
| `compute_transfer_function` | `BROADBAND` | `Field` of complex H(f) on a `frequency` axis | Bellhop, Kraken, OASP, OASSP, RAM, Scooter | `frequencies=` |
| `compute_covariance` | `COVARIANCE` | `Covariance`: the hydrophone cross-spectral matrix C(f, i, j) | OASN (noise field), OASS (reverberant field) | — |
| `compute_replicas` | `REPLICA` | `Replicas`: the field at the array per candidate source position | OASN | — |
| `compute_reverberation` | `REVERBERATION` | `Field` of `kind='reverberation'`, a loss in dB scattered from a rough interface | OASS | — |

All take `(env, source, receiver)` positionally (except `compute_modes`,
which omits `receiver` — modes are receiver-independent depth
eigenfunctions). A multi-depth `Source` gives one `Field` over a leading
`source_depth` axis from the wrappers that return a `Field`, and a
`ResultStack` of one slab per depth from `compute_rays`,
`compute_eigenrays` and `compute_arrivals` (§8, *The field of several
sources*).

### `RunMode`

`RunMode` (`uacpy.RunMode`) is the single enumeration of output kinds:

| Mode | Output |
|---|---|
| `COHERENT_TL` / `INCOHERENT_TL` / `SEMICOHERENT_TL` | transmission-loss field |
| `RAYS` / `EIGENRAYS` / `ARRIVALS` | ray paths / eigenrays / arrival structure |
| `MODES` | normal-mode depth eigenfunctions |
| `BROADBAND` | complex transfer function H(f) |
| `TIME_SERIES` | time-domain pressure p(t) |
| `COVARIANCE` / `REPLICA` | OASN array covariance / matched-field replicas |
| `REVERBERATION` | OASS reverberation loss scattered from a rough interface (a loss: `-10·log10 E[\|p_scat\|²]`) |
| `REFLECTION` | plane-wave reflection coefficients |

A model advertises what it supports via `model.supported_modes` /
`model.supports_mode(...)`; anything else raises `UnsupportedFeatureError`.
The single-frequency modes (TL, rays, modes) reject a multi-frequency
`Source` — use `BROADBAND` or `TIME_SERIES` for a band.

### Constructor-only configuration

Model knobs are set **once, at construction** — there is no `set_params()`
and no per-call override. To sweep a parameter, build one instance per
configuration; `model.copy(**overrides)` derives a new instance from an
existing one without re-typing every argument:

```python
base = uacpy.RAM(dr=2.0, dz=0.5, n_pade=8)
for dr in (1.0, 2.0, 4.0):
    field = base.copy(dr=dr).compute_field(env, source, receiver)
```

For many independent runs, batch them with `uacpy.run_parallel` over
`uacpy.Job` objects rather than looping in Python.

### Model provenance & licence

Every model wraps a third-party engine with its own authorship and licence.
That metadata lives in a catalogue (`uacpy.models.provenance.MODEL_PROVENANCE`) and
is surfaced per instance:

```python
m = uacpy.Kraken()
m.provenance_id     # 'acoustics_toolbox' — catalogue id (declared on the class)
m.provenance        # the ModelProvenance looked up from that id in MODEL_PROVENANCE
m.provenance.name   # 'Acoustics Toolbox'  (.authors / .license / .url / .note too)
m.citation          # bibliographic string to cite in a write-up
```

Two more instance-level queries answer "will this run?" without running it:
`m.supported_modes` / `m.supports_mode(RunMode.RAYS)` list the products the
model can produce, and `m.validate_inputs(env, source, receiver,
run_mode=…)` runs the carrier-level checks `run()` runs — the run mode
(resolved as `run()` resolves it, so `None` and the value string `'coherent_tl'`
are checked as the default and `COHERENT_TL`), the source frequencies against
that mode, the source geometry and beam pattern, the source/receiver depths and
the bottom type — and raises the exception a real run would for those, so you
can gate a batch up front. It also resolves the run's settings as `run()`
does, on the environment the engine projects from `env`, so the refusals that
need the computational grid (RAM's depth-cell and domain limits, for example)
come from it too; only the solver itself does not run.

The catalogue flags drive policy, mirroring the `uacpy.data` source catalogue:
constructing an engine whose licence forbids commercial use — only
**OASES** (academic, non-redistributable) — emits a one-time `ProvenanceWarning`, so
a licence-restricted result is never produced silently. GPL / public-domain
engines (Acoustics Toolbox; the Collins RAM family) stay quiet.

### Results

Every `run()` returns a typed `Result` subclass chosen by the run mode:
`Field` (TL / H(f) / p(t) — one unified array type whose physical meaning
follows from its dtype and coordinate axes), `Rays`, `Modes`, `Arrivals`,
`Covariance`/`Replicas`, `ReflectionCoefficient`; `uacpy.io.read_grn_file`
reads a Scooter/SPARC `.grn` into a `GreensFunction`, whose methods transform
it to a `Field`. `Field` exposes `.dB` (and
`.tl`, the same values under the quantity's name, pressure fields only),
the `.depths`/`.ranges` axes, and `.at(...)`/`.isel(...)`/`.max(...)` to
slice a dimension away. Details are in the Results section.

### Working directory & cleanup

Each run executes the native binary in a scratch directory. By default
this is a fresh temp directory that is deleted after the run. Pin one to
keep the inputs/outputs around — and to chain models that read each
other's files:

```python
model = uacpy.Bellhop(work_dir='./run_out', cleanup=False)
```

`cleanup` defaults to `True` only when uacpy owns the directory (i.e.
`work_dir=None`); a user-pinned `work_dir` is kept unless you set
`cleanup=True`. Pass `use_tmpfs=True` for RAM-backed scratch I/O.

### Exceptions

All uacpy errors inherit from `UACPYError`, so you can catch the whole
family with one `except`. The typed subclasses are
`ExecutableNotFoundError` (binary not built/found),
`ModelExecutionError` (solver returned non-zero / timed out),
`InvalidDepthError` (source below the resolvable depth),
`UnsupportedFeatureError` (mode/feature the model lacks),
`ConfigurationError` (bad parameter value; its subclass
`models.GridInfeasibleError` is what `models.optimize_grid` / `grid_error` raise
when no PE grid meets the accuracy budget, and `MissingFileError`, which
every reader raises for a path that names no file and which is a
`FileNotFoundError` too), `FileFormatError` (I/O parse
failure), `DataFetchError` (the data layer), and `OutputContractError` (a
model returned another result than its settings declare: a uacpy defect
to report). `supports_feature` on a name that is not a capability flag
raises `ConfigurationError` too; `UACPYError` does not derive from
`ValueError`, so catch it as `ConfigurationError`. Every uacpy check raises a
uacpy error — apart from the Pythonic container
protocols: `.p`, `.magnitude` and `.phase` raise `AttributeError` on a real
field, `.dB` and `.tl` raise it on a time trace (§8), and an out-of-range
`.isel` raises `IndexError`.

### Warnings

Every warning uacpy emits is a subclass of `UACPYWarning`, itself a
`UserWarning`, named for its cause:

| Class | Means |
|---|---|
| `NumericsWarning` | the discretisation or the numerical solution limits the answer: a grid, step, window or record that cannot resolve what is asked (aliasing, wrap-around, truncation, an under-resolved axis), a solve that did not converge, a run sized past its budget |
| `ValidityWarning` | the problem lies outside where the physics, formula, fit or approximation holds, or the quantity is undefined there |
| `FallbackWarning` | uacpy used something other than what was asked: an input ignored, clamped, rounded or replaced, a default or the nearest value taken, a feature the model cannot carry collapsed |
| `ProvenanceWarning` | where fetched data come from (source, distance, age, a value a data fetch substituted from a neighbour) and a licence, attribution or calibration status you must know |
| `IOWarning` | a file, an engine's own log or a data service delivered something irregular |

A filter on `UserWarning` still catches all of them;
`warnings.filterwarnings('ignore', category=uacpy.FallbackWarning)` silences
one cause and keeps the others (inside one data fetch, the substitution notes
travel in that fetch's one `ProvenanceWarning`), and `simplefilter('error', uacpy.NumericsWarning)`
stops a run on an under-resolved grid only. Under `run_parallel` the workers do
not inherit the filter: each job runs to the end, and the filter acts as its
warnings are re-emitted, failing that job (its result dropped, not stacked).
`import uacpy` renders uacpy's own warnings as
`[timestamp] [WARN ValidityWarning] [source:line] message` (the class printed);
every other warning of the process prints as Python prints it.
`UACPY_NO_WARNING_FORMAT=1` keeps Python's rendering for uacpy's too, and a
formatter your application installed first is kept
([utilities guide](docs/guide/utilities.md)).

### Status output, logging & parallelism

Models are quiet by default, printing only warnings and errors. Pass
`verbose=True` (or `'info'`) to see progress, `verbose='debug'` for
subprocess command lines and grid-resolution choices. Status goes through
`uacpy._log` (never `print`); user-facing notices use `warnings.warn`.

Each model instance owns its own scratch directory, so independent
instances are safe to run concurrently — use `uacpy.run_parallel` to fan
runs out across processes.

**Model binaries run with a raised stack limit; your process keeps its own.**
SPARC-class binaries can overflow the default 8 MiB stack on their first large
allocation, so every model subprocess is launched through a short `sh` step
that raises its soft `RLIMIT_STACK` to the hard limit (`ulimit -s`) and then
`exec`s the binary. Importing uacpy leaves the importing process's own stack
limit as it was — a Jupyter kernel or a host service keeps its limit. Set `UACPY_NO_STACK_RAISE=1`
to launch the binaries directly with the inherited limit; SPARC-class models
may then segfault on large problems.

## 5. Environment

The `Environment` is the physics-agnostic carrier that describes the ocean: it
holds the bathymetry, sound-speed profile, boundaries, and volume absorption.
Every propagation model consumes the *same* `Environment` — the object knows
nothing about which solver runs over it. Build it once, reuse it everywhere.

```text
Environment(bathymetry, ssp, altimetry, bottom, surface, absorption,
            *, name, location, transect, date, water_density,
            extra_data_sources)
```

All arguments except `bathymetry` are optional. Each is a small typed carrier
(below); most also accept a shorthand the constructor coerces (a scalar SSP, a
preset-name bottom, …). `env.depth` is a **read-only** property — the maximum
water depth, derived from `bathymetry`. Convention throughout: metres, Hz, m/s,
g/cm³, dB-per-wavelength; depth positive **down**, altimetry positive **up**.
`water_density` (g/cm³; default `None`, the package's
`DEFAULT_WATER_DENSITY` = 1.027; any positive value, one above 20 refused as kg/m³) is the sea-water density every deck writes for the water column;
pass `1.0` for a textbook ρ_w = 1 benchmark.

### Bathymetry and altimetry

`bathymetry` is the only required argument. Pass a scalar for a flat bottom, or
a list of `(range, depth)` pairs for a sloping/range-dependent seafloor:

```python
env = Environment(bathymetry=100)                       # flat, 100 m
env = Environment(bathymetry=[(0, 100), (10_000, 200)])  # wedge, 100→200 m
```

Arrays held as two columns go through `np.column_stack([ranges, depths])` (or
`Bathymetry(ranges=, depths=)`): `bathymetry=(ranges, depths)` is refused with
that hint. Two numpy arrays, or a tuple of two lists, are read as columns at any
length, two points included; a list of tuples, a 2 × 2 nested list or a 2 × 2 array is
read as two pairs.
`altimetry=` and `ssp=` follow the same rule.

`altimetry` describes a non-flat sea surface as `(range, height)` pairs (height
positive up). Build a realistic rough surface from a Pierson–Moskowitz spectrum
with `generate_sea_surface`, which returns an array shaped for `altimetry=`
(`Altimetry.from_sea_state` takes the same arguments and returns the carrier):

```python
import numpy as np
from uacpy import generate_sea_surface

surf = generate_sea_surface(rmax=10_000, wind_speed_19_5m_kn=20,
                            rng=np.random.default_rng(0))
env = Environment(bathymetry=100, altimetry=surf)
```

The range grid must resolve the spectral peak: 8 samples per nominal
Pierson–Moskowitz peak wavelength `λ_p = 2π U²/g` (64 m at 10 m/s), i.e.
`n_points ≥ 8 * rmax / λ_p + 1` (1250 points over 10 km). Left at its default
(`None`) it is chosen that way — at least 500, at most 200 000 (past which it is
capped and warned about rather than allocated), and 500 for a calm sea
(U ≤ 0.5 m/s, whose millimetre ripples stand in for a flat surface). An explicit
coarser grid warns and returns a surface flatter than the wind's Hs.
`uacpy.data.fetch_sea_surface` sizes its realization by the same rule.

`env.bathymetry` and `env.altimetry` are **carriers** (`Bathymetry` / `Altimetry`),
the seafloor- and surface-*shape* analogues of `SoundSpeedProfile`. They expose the
same grid-library trio — `at(range=)` (nearest), `isel(range=)` (positional),
`eval(range=, method=)` (interpolate) — returning the depth / height there:

```python
env.bathymetry.eval(range=4_000)     # interpolated seafloor depth (m)
env.altimetry.at(range=4_000)        # nearest surface height (m, up)
```

### Surface boundary (uniform or range-dependent)

`surface=` is the top *boundary* (a `BoundaryProperties`: vacuum / pressure-release
by default, or an elastic ice half-space). `env.surface` is a `Surface` carrier —
the top-properties analogue of `env.bottom` — and a single `BoundaryProperties` is
coerced to a uniform one. A **range-dependent** surface (e.g. a marginal ice zone,
open water → pack ice → open water) is built from `(range, BoundaryProperties)`
nodes and selected with `at`/`isel` (no `eval` — boundary types can't blend, just
like `Bottom`):

```python
from uacpy import Surface, BoundaryProperties
ice  = BoundaryProperties(acoustic_type='half-space', sound_speed=3500,
                          density=0.9, shear_speed=1800)
surf = Surface.coerce([(0, BoundaryProperties(acoustic_type='vacuum')),
                       (5_000, ice)])               # open water → ice at 5 km
env = Environment(bathymetry=200, ssp=1500, surface=surf)
```

The propagation solvers each carry a **single global top boundary** (only the
SSP varies with range), so — exactly like a range-dependent *bottom* — **no
model honours range-dependent surface properties**: every model collapses it to
one boundary with a `FallbackWarning` (`collapse={'surface': 'r0'|'rmax'|'mean'|'median'}`).
The `Surface` carrier still lets you build, fetch, and **plot** the marginal ice
zone (`env.plot()` draws it from `env.surface`). Kraken runs an elastic ice
surface: `krakenc.f90:220-222` folds the top half-space shear speed into `cMin`,
so uacpy pins the phase-speed floor at the minimum compressional speed rather
than letting the solver drop to 0.84x the ice shear speed and chase the
ice/water Scholte mode. The data layer can build a marginal-ice-zone surface
straight from sea-ice climatology with
`uacpy.data.sea_ice_surface_transect(start, end)`.

### Sound-speed profile

The SSP **shape** (the profile data) lives on the environment; the
*interpolation scheme* used between samples is a model knob, not an env field
(`interp_ssp=`: `'linear'`, `'n2linear'`, `'pchip'`, `'spline'`, or Bellhop's
`'quad'`; `None` picks `'quad'` for a range-dependent profile and `'linear'`
otherwise, and any other value is refused when the run writes its deck). Pass
`ssp=` in any of these forms — the constructor coerces them to a
`SoundSpeedProfile`:

```python
from uacpy import Environment, SoundSpeedProfile

Environment(bathymetry=100, ssp=1500)                       # isovelocity
Environment(bathymetry=200, ssp=[(0, 1520), (200, 1480)])    # (depth, c) pairs
Environment(bathymetry=5000, ssp=SoundSpeedProfile.from_munk(5000))
```

`SoundSpeedProfile` factories cover the common cases:

```python
SoundSpeedProfile.from_isovelocity(depth_max=100, sound_speed=1500)
SoundSpeedProfile.from_pairs([(0, 1520), (200, 1480)])
SoundSpeedProfile.from_munk(depth_max=5000)                  # canonical Munk
SoundSpeedProfile.from_temperature_salinity(depths, temperature, salinity)  # from T/S
```

A **range-dependent** (2-D) profile carries a `sound_speed[n_depth, n_range]` matrix
plus a `ranges` axis (metres). The depth axis is shared across ranges:

```python
import numpy as np
ssp = SoundSpeedProfile.from_2d(
    depths=[0, 200],
    ranges=[0, 5000],                      # metres
    matrix=np.array([[1520, 1500],         # rows = depth, cols = range
                     [1480, 1470]]),
)
env = Environment(bathymetry=200, ssp=ssp)
```

Casts measured on their own depth grids — a 60 m shelf cast beside an 800 m
slope cast, or a raw down/up cast with repeated depths — go through
`from_casts`: each cast is sorted, repeated depths are averaged, and all are
interpolated onto the union of their depths, a short cast holding its deepest
value:

```python
ssp = SoundSpeedProfile.from_casts(
    [0, 5000],                                   # range of each cast (m)
    [[(0, 1500), (60, 1495)],                    # (depth, c) rows per cast
     [(0, 1502), (100, 1490), (800, 1485)]])
```

If a profile is shallower than the seafloor, the constructor extends its deepest
sample down to `env.depth` automatically, holding that speed constant. A hold
long enough for pressure alone to add more than 5 m/s (about 307 m) gives a
`FallbackWarning`; `uacpy.data.extend_ssp_below_data` continues the profile
along its deep gradient instead.

### Bottom

The seabed is one `Bottom` carrier — a list of `SeabedColumn`s with an optional
`ranges` axis, mirroring `SoundSpeedProfile`. A `SeabedColumn` is *sediment
layers over a half-space*; an empty layer list is a pure half-space. `bottom=`
auto-coerces several shorthands:

```python
Environment(bathymetry=100, bottom='sand')   # material preset (half-space)
Environment(bathymetry=100, bottom=1800)     # scalar → half-space sound speed
Environment(bathymetry=100, bottom=BoundaryProperties(
    sound_speed=1700, density=1.8, attenuation=0.6))   # explicit half-space
```

The default (no `bottom=`) is a generic fluid half-space (1600 m/s, 1.5 g/cm³,
0.5 dB/λ), uacpy guidance: no published source gives this triple
(Jensen's 1984 upslope wedge, in Etter §5.3.3.1, uses 1600 m/s and 1.5 g/cm³
with 0.2 dB/λ). It is none of the presets
— `sand` is 1650 m/s, 1.95 g/cm³, 0.8 dB/λ — so pass `bottom=` with a preset or
measured values when the seabed matters. For range-dependent or layered seabeds, build a `Bottom` explicitly.

Range-dependent **half-space** from parallel property arrays:

```python
from uacpy import Bottom
bottom = Bottom.from_halfspaces(
    ranges=[0, 10_000],
    sound_speed=[1600, 1800], density=[1.6, 2.0], attenuation=[0.5, 0.8],
)
```

**Stratified** (layered) column, then a range-dependent stack of columns:

```python
from uacpy import (Environment, Bottom, SeabedColumn, SedimentLayer,
                   BoundaryProperties)

near_col = SeabedColumn(
    layers=[
        SedimentLayer(thickness=10, sound_speed=1600, density=1.7, attenuation=0.8),
        SedimentLayer(thickness=30, sound_speed=1750, density=2.0, attenuation=0.5),
    ],
    halfspace=BoundaryProperties.from_preset('limestone'),
)
Environment(bathymetry=100, bottom=near_col)                 # layered, range-indep

far_col = SeabedColumn.from_presets(layers=[('clay', 25)], halfspace='chalk')
rdl = Bottom.from_columns([near_col, far_col], ranges=[0, 10_000])  # range-dep layered
```

`SeabedColumn.from_presets` builds a layered column straight from material names:

```python
col = SeabedColumn.from_presets(
    layers=[('sand', 10), ('clay', 30)],     # (preset, thickness)
    halfspace='limestone',
)
```

### BoundaryProperties

`BoundaryProperties` describes a single fluid/elastic boundary — used for the
surface (`surface=`) and for bottom half-spaces. Acoustic properties only;
geometry comes from `bathymetry`/`altimetry`.

```python
BoundaryProperties(
    sound_speed=1700,        # compressional c_p, m/s
    density=1.8,             # g/cm³
    attenuation=0.6,         # compressional, dB/wavelength
    shear_speed=0.0,         # c_s, m/s (>0 = elastic bottom)
    shear_attenuation=0.0,   # dB/wavelength
    roughness=0.0,           # RMS interface roughness, m
    acoustic_type=None,      # inferred: 'half-space' / 'vacuum' / 'file' / ...
)
```

`acoustic_type` is usually inferred: any *explicitly passed* acoustic property
→ `'half-space'` (even a value equal to the documented default — passing
`sound_speed=1600` always means a 1600 m/s half-space), a `reflection_file` →
`'file'`, nothing → `'vacuum'` (pressure-release surface, the default). Pass it
explicitly only for the parameter-free `'rigid'` model. The default `surface`
is a vacuum (pressure-release) boundary.

The accepted strings are the values of the `BoundaryType` enum (`'vacuum'`,
`'rigid'`, `'half-space'`, `'file'`, `'precalc'`), exported at top level for
comparisons that should not hard-code the string. The Acoustics-Toolbox
attenuation-unit letters (`'W'` dB/λ, the one uacpy writes, plus `'N'`, `'F'`,
`'M'`, `'Q'`, `'L'`) are a deck-format vocabulary:
`uacpy.io.at_codes.AttenuationUnits`.

### Materials catalog

`uacpy.materials` provides class-typical geoacoustics for common seabeds
(`clay`, `silt`, `sand`, `gravel`, `moraine`, `chalk`, `limestone`, `basalt`,
`granite`):

```python
from uacpy import MATERIALS, list_materials, get_material, BoundaryProperties

list_materials()                 # ['basalt', 'chalk', 'clay', ...]
get_material('sand')             # {'sound_speed': 1650.0, 'density': 1.9513, ...}
                                 # (Table 1.3's ratio 1.9 × the 1.027 g/cm³ water)
bp = BoundaryProperties.from_preset('sand')                 # fluid half-space
bp = BoundaryProperties.from_preset('limestone', elastic=True)   # keep shear
```

Presets are **fluid by default** (shear dropped) so they work with every model;
pass `elastic=True` to keep the preset's shear speed/attenuation (needed only
for elastic-capable solvers). `SedimentLayer.from_preset(name, thickness=...)`
does the same for a layer.

The acoustic properties are keyword-only on both carriers: only
`BoundaryProperties`' `acoustic_type` and `SedimentLayer`'s `thickness` may be
given positionally, so a sound speed and a density cannot trade places.

### Volume absorption

`env.absorption` carries the water-column volume-attenuation model. Pick one;
each deck writer carries it automatically (a `TopOpt` letter for Thorp and
Biological, the water rows' `alphaI` for the others):

```python
from uacpy import Thorp, FrancoisGarrison, Biological, ConstantAbsorption

Environment(bathymetry=100, absorption=Thorp())             # frequency-only
Environment(bathymetry=100, absorption=FrancoisGarrison(    # T/S/pH, at each depth
    temperature=10, salinity=35, pH=8))   # pH on NBS scale
Environment(bathymetry=100, absorption=FrancoisGarrison(    # a GLODAP / BGC pH
    temperature=10, salinity=35, pH=7.9, ph_scale='total'))
Environment(bathymetry=100, absorption=ConstantAbsorption(0.1))   # flat dB/λ
Environment(bathymetry=100, absorption=Biological(          # fish-bladder layers
    [(0, 50, 1000, 5, 0.5)]))   # (z_top, z_bottom, f0, Q, a0)
Environment(bathymetry=100, absorption=FrancoisGarrison(    # a T/S profile,
    temperature=[(0, 20), (100, 12)],                       # (depth, value)
    salinity=[(0, 35), (100, 34.8)], pH=8))                 # pairs
Environment(bathymetry=100,                                 # the same, from
    absorption=FrancoisGarrison.from_temperature_salinity(  # arrays on depths
        [0, 100], [20, 12], [35, 34.8]))
FrancoisGarrison().table([1e3, 1e4])          # the law's α, to look at
```

Francois-Garrison evaluates the formula at each depth with the water there
(one row or a profile; the formula's depth term takes the depth evaluated, so
the law has no depth of its own); the Acoustics-Toolbox decks carry it in
every water SSP row's `alphaI` (dB/λ at the deck frequency) under a blank
`TopOpt(4)`, so every engine sees the same α(z); a broadband Kraken/Scooter
deck carries one water row as AT's `'F'` at mid-water column instead (exact
in frequency, warned when the one depth costs 0.05 dB/km or more). **One object per law:** the
environment takes the law; `law.table(f)` is its α for looking at and is
refused as an absorption (pass the law). The one table an environment takes
is a measured α(f, z) with no law behind it,
`AbsorptionCoefficient(frequencies=, data=, units=, depths=)` (`model=None`), or from `(frequency, value)` pairs with `AbsorptionCoefficient.from_pairs([...])` / `from_pairs({depth: [...]})`,
used as tabulated — linear in depth and in log f, refusing a frequency
outside it ([environment guide](docs/guide/environment.md#absorption-from-real-water-a-profile-or-a-table)).

Each water property (`temperature`, `salinity`, `pH`) is a single value, an
array of exactly the shape of the depth evaluated, or `(depth, value)` pairs
as an `(M, 2)` array ([utilities guide](docs/guide/utilities.md#sound-speed)).

`fetch_environment(..., with_absorption=True)` builds a `FrancoisGarrison`
profile from the site's fetched temperature/salinity column; on a transect, the
column at the track's deepest seafloor sample continues the start's column
below its deepest level ([data guide](docs/guide/data.md#4-sound-speed-and-the-rest-of-the-water-column)). Its pH is pH-source
aware: on the Copernicus SSP branch (`ssp_sources='copernicus'`) it prefers the
date-specific Copernicus biogeochemistry `ph` field (one value at the
column's mid-depth), else the cached GLODAP climatology when installed
(`install.sh --data glodap`, its column as `(depth, pH)` pairs), else the
reference-water default (8.0). Both sources report the total scale; the built model carries
`ph_scale='total'` and converts to the NBS scale the equation was fitted on
(about +0.1, +20 % absorption below 1 kHz). The 8.0 default stays on NBS.
The law keeps where its water came from in `env.absorption.data_sources` (the
T/S and pH records; every law and `AbsorptionCoefficient` has the keyword-only
field, `()` for water typed by hand): its `.table(f)` carries them and the
table's plot credits them.

All four subclass `Absorption`, so `isinstance(env.absorption, Absorption)` is
the type test for a law; an environment given a measured table holds that
`AbsorptionCoefficient` itself. `Biological` takes its layers either as
`BiologicalLayer(z_top, z_bottom, f0, Q, a0)` objects or as the bare
5-tuples above — the constructor coerces.

Default is `None` (no explicit volume absorption). To evaluate a law
rather than attach one, call its `table`: `Thorp().table(frequencies)`,
`FrancoisGarrison(temperature=…, salinity=…, pH=…).table(frequencies, depths=…)`
(reference water when omitted, the surface when `depths` is), `Biological(layers=…).table(…)` or
`ConstantAbsorption(value=…).table(…)`. Each returns an
`AbsorptionCoefficient` — α (`.data`) over frequency, and over depth when
`depths=` is a sequence — carrying its own units and the law that made it
(`.model`, `.parameters`), with `.plot()`, `.to_units()` and the export
protocol (`to_dict`, `to_xarray`). The formulas the laws are written
on take and return plain arrays in dB/km, named `absorption_<model>` like
`sound_speed_mackenzie`:
`uacpy.acoustics.absorption_thorp(frequency, depth=None)`,
`uacpy.acoustics.absorption_francois_garrison(frequency, temperature, salinity,
pH, depth)` and `uacpy.acoustics.absorption_biological(frequency, f0, Q, a0)` (one
layer's resonance), beside `uacpy.acoustics.ph_to_nbs`. The unit converter for
a bare number or array is exported beside them too (dB/km ↔ dB/m ↔
dB/wavelength ↔ Np/m ↔ Q ↔ L); its `alpha`, `frequency` and `sound_speed`
broadcast together, so a spectrum α(f) converts in one call:

```python
uacpy.acoustics.convert_attenuation_units(0.5, 1000.0, 'dB/km', 'dB/wavelength')
f = np.array([100.0, 1000.0, 10_000.0])                 # Hz
alpha = uacpy.acoustics.absorption_thorp(f)              # dB/km, one per f
uacpy.acoustics.convert_attenuation_units(alpha, f, 'dB/km', 'dB/wavelength')
```

### Real-world environments

`uacpy.data.fetch_environment` assembles a ready-to-run `Environment` for a
GPS point from public datasets: GEBCO bathymetry, WOA23/Copernicus sound speed,
and grain-size → bottom geoacoustics.

```python
import uacpy.data as data

env = data.fetch_environment((43.2, 7.5), date='2026-06-14', bottom='sand')
```

Each axis can be a **literal** (`ssp=`, `bathymetry=`, `bottom=`, as
`Environment` takes them, with one difference: a `bottom=` number is a mean
grain size ϕ, not a sound speed, and a number with |ϕ| > 20 is refused as a
likely sound speed) and/or fetched from one or more `*_sources` tried in
order. `transect_to=(lat, lon)` samples bathymetry (and optionally SSP/bottom)
along a great-circle path for a range-dependent environment;
`with_absorption=True` attaches a site-specific Francois–Garrison model;
`bottom_model='apl-uw'` swaps the grain-size → geoacoustics relations every
ϕ-reporting seabed source goes through from Hamilton's low-frequency table to
the APL-UW TR 9407 high-frequency ones.
Fetching is cache-first **within each source**: an installed local dataset is
sampled before that source's own network call. It is not cache-first across the
chain — `'auto'` is ordered by data quality, not by what happens to be
installed, so it can reach the network for a better source before falling
through to an installed global grid. `*_sources='local'` keeps only the cached
backends and never touches the network, which is what an offline install
(`install.sh --data`) is for. See `help(data.fetch_environment)` and
`example_37` for the long tail of options.

**Per-layer fetchers.** `fetch_environment` is the capstone; every layer it
assembles is also a public function you can call on its own, so you can fetch
one piece and build the rest by hand. Each returns the carrier (or the raw
arrays) for a GPS point; where a `*_transect` twin is listed it samples a
great-circle path instead, for a range-dependent environment.

| Layer | Point / transect fetchers |
|---|---|
| Bathymetry | `fetch_bathy`, `fetch_bathy_transect`, `fetch_bathy_grid` (a lat/lon map for `plot_bathymetry_map`), `transect_length`, `transect_waypoints` (the fetchers' geodesic, for sampling your own data on it) |
| Sound speed (climatology) | `fetch_ssp`, `fetch_ssp_transect`, `fetch_ts_profile` (your own casts stack into a range-dependent profile with `SoundSpeedProfile.from_casts`) |
| Sound speed (operational) | `fetch_ssp_operational`, `fetch_ssp_transect_operational`, `fetch_ts_profile_operational` |
| Sound speed (in-situ) | `fetch_argo_profile`, `fetch_ssp_argo` |
| Sound speed (deep extension) | `extend_ssp_below_data` (one profile, along its own deep gradient), `extend_column_to_seafloor` (the transect form), `deepen_with_climatology` (a cast of your own continued by the WOA23 column at its point and month, blended over its deepest 500 m) |
| Seabed — any source | `fetch_bottom`, `fetch_bottom_transect` (`source=` one of `'emodnet'`, `'grainsize'`, `'diesing'`, `'mars'`, `'graw'`, `'crust1'`, `'pelagic'`, or the chains `'auto'` / `'local'` that `fetch_environment(bottom_sources=)` takes; a chain on a transect is resolved at each waypoint) |
| Seabed — grain size | the ϕ conversion `grain_size_to_geoacoustics` (one ϕ or an array; inverse `grain_size_from_density`), which `BoundaryProperties.from_grain_size` wraps into a half-space (`BoundaryProperties.from_preset` for a named class); `bottom_from_samples` (your own samples' ϕ → a range-dependent bottom along a transect, each waypoint taking its nearest sample) |
| Seabed — source records | `fetch_emodnet_substrate` (EMODnet; `source='emodnet'` installed polygons then the live WFS, `'local'` installed only), `fetch_sediment_sample` (NCEI / DECK41 samples), `fetch_seafloor_lithology` (Diesing), `fetch_mars_sediment`, `fetch_seabed_density`, `fetch_seabed_density_transect` (Graw), `pelagic_lithology`, `pelagic_grain_size` |
| Seabed — thickness / crust | `fetch_sediment_thickness`, `fetch_sediment_thickness_transect`, `fetch_crust1_profile` (→ `Crust1Profile`) |
| Sea surface | `fetch_sea_surface` (an `Altimetry` carrying its provenance), `fetch_wind`, `fetch_wind_transect`, `fetch_waves`, `fetch_waves_operational`, `hs_to_pm_wind` (significant wave height → Pierson-Moskowitz wind, knots at 19.5 m) |
| Sea ice | `fetch_sea_ice_concentration`, `fetch_sea_ice_concentration_transect`, `sea_ice_grid` (a `SeaIceGrid`: the hemisphere's concentration with `hemi`, `month`, the cell coordinates `x`/`y`/`lats`/`lons`, `provenance`; `np.asarray(grid)` is the concentration, `.plot()` the polar map), `sea_ice_pixel`, `sea_ice_surface`, `fetch_sea_ice_surface`, `sea_ice_surface_transect`, `SEA_ICE_TYPICAL_ROUGHNESS` (a starting under-ice RMS roughness in m for `sea_ice_surface(..., roughness=)`) |
| Absorption inputs | `fetch_ph`, `fetch_ph_profile`, `fetch_ph_operational` (the law: `FrancoisGarrison.from_temperature_salinity`) |
| Provenance | `SOURCES`, `DataSource`, `DataProvenance`, `citations` |
| Cache layout | `cache_root`, `dataset_root`, `is_installed` (where an offline dataset lives, and whether it does) |

**Offline caches.** `install.sh --data <keyword>` calls the matching
`download_*_db` function, which is also public: `download_emodnet_db`,
`download_globsed_db`, `download_crust1_db`, `download_diesing_db`,
`download_graw_db`, `download_sediment_db`, `download_glodap_db`,
`download_seaice_db`, `download_wind_db`, and — for the coastline the map
plotters draw — `uacpy.visualization.download_coastline`. Once a database is
cached the corresponding `*_sources='local'` path runs with no network. Each one takes the
address it downloads from — `url=` for the fetchers that pull one file,
`base_url=` for the two that page through a service (`emodnet`, `seaice`) — so
a moved publisher or a dead host is a keyword away from a mirror; only
`http`/`https` addresses are accepted.

**Provenance & citations.** Every fetched layer records where it came from. A
`SeabedSample` (`fetch_sediment_sample`, …), an `ArgoProfile` or a
`SeaStateRecord` (`fetch_waves`) holds its record under `.provenance`, as a `TSProfile` (`fetch_ts_profile`) does. A raw value
(`fetch_bathy*`, the scalar transects) holds none and names the
carrier-level fetcher that does. Each carrier carries `carrier.data_sources` — a tuple of `data.DataProvenance`
records, each pairing the catalogue `.source` (`data.DataSource`: name, licence,
attribution, citation) with the **actual** `data_date` and `data_point=(lat,
lon)` returned (which may differ from what was requested — a climatology snaps to
a grid cell, an Argo float is the nearest cast). `Environment` aggregates the
union into `env.data_sources`. Render the required attribution text with
`data.citations(env)` (or pass a carrier, a record a fetcher returned, a source
id, or a `DataProvenance`);
non-commercial sources also emit a `ProvenanceWarning` when fetched.

## 6. Source and Receiver

`Source` and `Receiver` carry only **geometry and spectrum** — solver knobs
(launch angles, beam counts, …) live on the model.

`Source(depths, frequencies)` places one or more sources. Depths are positive
down; multiple depths must be strictly increasing (outputs are indexed by source
depth). Frequencies may be a scalar or array. `weights=` gives each source a
complex amplitude (one per depth, a scalar broadcasts, default all ones): the
engines never read it — a multi-depth run returns one `Field` whose leading
`source_depth` axis holds the field of each unit-amplitude source — and
`field.superpose()` adds them with those weights (§8, *The field of several
sources*). A **single-depth** source with a non-unit
weight is the n = 1 case of the same sum: `run()` returns its `Field` already
scaled by the weight and records it in its `superposed_sources`. The
same rule bounds both: a field that has lost its phase (a dB-only result such
as Kraken's `INCOHERENT_TL` or OAST's TL) refuses a weight, and a real
time-domain trace refuses a complex one; outside the field modes (rays,
arrivals, modes, reflection tables) the weights are not applied and `run()`
warns. A multi-depth field read through `.dB` / `.tl` / `.p` / `.plot()`
without `superpose()` shows its unit-amplitude sources and, when the `Source`
carried other weights, warns that they were not applied.

```python
from uacpy import Source

Source(depths=50, frequencies=100)              # single source, single tone
Source(depths=[10, 20, 30], frequencies=200)    # vertical source array
Source(depths=50, frequencies=[100, 200, 400])  # multi-frequency
Source(depths=[40, 60], frequencies=200, weights=[1, -1])   # antiphase pair
Source(depths=50, frequencies=100, source_level_dB=180)     # driven at 180 dB
```

`source_level_dB=` is how hard a unit-weight element is driven (dB re 1 µPa
at 1 m). The engines never read it — like `weights` it is applied to the
field afterwards — but every result carries it, so
`field.at_source_level()` turns the loss into the absolute level a
hydrophone at that cell would read, with nothing repeated at the call site:

```python
field = model.run(env, Source(depths=50, frequencies=100,
                              source_level_dB=180), receiver)
field.dB                       # transmission loss, re a unit source at 1 m
field.at_source_level().dB     # received level, dB re 1 µPa
```

A multi-depth `Source` is an **array**, and three angular quantities describe
it — the product theorem `P(θ) = f(θ)·A(θ)` as Butler & Sherman state it for
underwater arrays (*Transducers and Arrays for Underwater Sound*, §7.1.1):

```python
src.element_directivity(angles)   # f(θ) — the .sbp as linear amplitude,
                                  #        interpolated the way engines read it
src.array_factor(angles)          # A(θ) — the geometry alone: the pattern of
                                  #        the array with POINT elements
src.array_beam_pattern(angles)    # P(θ) = f·A — what the array radiates
```

`array_factor` is the one the theorem isolates, so it is unchanged when the
elements are swapped and is **not** what a shaded array radiates; compare
`array_beam_pattern` against a measurement. The first two are methods over
plain-array functions, `uacpy.acoustics.array_factor(positions, weights,
angles_deg, frequency, sound_speed)` and
`uacpy.acoustics.element_directivity(beam_pattern, angles_deg)`, for an array
described by numbers rather than a `Source`. The theorem needs identical,
co-aligned elements — which a `Source` satisfies, since one `beam_pattern`
and one `source_type` describe every depth.

All three are **free-field**. In a waveguide the array is a *mode filter*
(Medwin & Clay §11.3.1): it sets each mode's amplitude, which is what
`Modes.excitation` computes and what actually reaches the receiver. The two
descriptions agree only while the pattern is symmetric in ±θ — steering is
precisely what breaks that, because a trapped mode is a standing wave with
equal up- and down-going halves. `modes.excitation_pattern(source,
sound_speed=…)` returns both on one angle axis for that reason, and its
`.plot()` (`plot_mode_excitation`) draws them. Example 40 works through all of it.

The level is its own `kind`, so it is captioned "Level (dB re 1 µPa)" rather
than "TL (dB)" and a 1-D cut through it reads upward — more of a level is
louder, where more of a loss is quieter. A multi-source total works the same
way: `field.superpose(coherent=False).at_source_level()`.

`Receiver(depths, ranges)` places the field-evaluation points. Depths are
positive down, ranges are outward from the source (metres); like `Source`'s
depths, both axes must be **strictly increasing** (minimum steps
`uacpy.core.deck_limits.DECK_DEPTH_RESOLUTION` and `DECK_RANGE_RESOLUTION`):

```python
import numpy as np
from uacpy import Receiver

Receiver(depths=50, ranges=1000)                              # single point
Receiver(depths=np.linspace(10, 90, 9), ranges=5000)          # vertical array
Receiver(depths=np.linspace(0, 100, 51),
         ranges=np.linspace(50, 10_000, 200))                 # full grid, past r = 0
```

A receiver is a **grid**: the field is evaluated on the full depth × range
cross-product.

Paired samples (depths and ranges paired point-by-point, e.g. a glider track
or tilted array) come from a grid run: every model returns the full
cross-product. A track's depths
need not be monotonic (a glider yo-yos), so run the grid over the track's
distinct depths and ranges and index the pairs (on Kraken a tilted vertical
array can also go into the deck itself: `Kraken(range_offsets=[...])`, one
offset per depth, field.exe's `Rro`):

```python
track_depths = np.array([10, 50, 20, 60])          # yo-yo in depth
track_ranges = np.array([100, 200, 300, 400])
zu, iz = np.unique(track_depths, return_inverse=True)
ru, ir = np.unique(track_ranges, return_inverse=True)
tl = model.run(env, src, Receiver(depths=zu, ranges=ru)).dB
paired = tl[iz, ir]      # (track_depths[k], track_ranges[k]) for each k
```

## 7. Propagation Models

A propagation model takes the `Environment` / `Source` / `Receiver` triple and
computes the acoustic field. Every model subclasses `PropagationModel` and
shares one contract:

- **Constructor-only configuration.** Every solver knob is a constructor
  argument (`Bellhop(beam_type='B', n_beams=500)`, `RAM(dr=2.0, n_pade=8)`).
  To sweep a parameter, build one model per setting — `model.copy(**overrides)`
  clones an instance with one knob changed.
- **Fixed run signature.** `run(env, source, receiver, run_mode=None, *,
  frequencies=None, source_waveform=None, sample_rate=None, output_duration=None,
  t_start=None)`.
  Identical across every model. No `**kwargs` — an unexpected keyword
  raises `TypeError`.
- **`compute_*` convenience family.** Thin wrappers over `run()`, one per run
  mode: `compute_field`, `compute_rays`, `compute_eigenrays`, `compute_arrivals`,
  `compute_modes`, `compute_reflection`, `compute_time_series`,
  `compute_transfer_function`, `compute_covariance`, `compute_replicas`,
  `compute_reverberation`. Each capability-checks first, then forwards its mode's kwargs.
- **Units:** metres, Hz, m/s, g/cm³, dB-per-wavelength. Depth is positive down.

The sections below are usage-oriented. For the **complete per-parameter tables**
(every constructor knob with *unit · default · meaning*) see
[§18 Parameter Reference](#18-parameter-reference).

### Choosing a model

| Reach for | When |
|---|---|
| **Bellhop** | High-frequency ray/beam tracing. The only source of ray paths, eigenrays, and arrival structure; broadband time series from a single trace. Honours range-dependent bathymetry/SSP and altimetry. |
| **Kraken** | Normal modes. Want the modal decomposition itself (`k`, `φ(z)`) or modal-sum TL; range-dependent via adiabatic/coupled modes. Elastic media via `krakenc`. |
| **Scooter** | Wavenumber integration (FFP). Exact range-independent field including layered/elastic bottoms and leaky energy that modes miss. |
| **RAM** | Parabolic equation. The workhorse for strongly range-dependent environments (sloping bathymetry, fronts) at low-to-mid frequency. |
| **SPARC** | Time-marched FFP (wavenumber integration, Porter 1990) — direct synthesis of the transient pressure pulse `p(t)` without an IFFT step. |
| **OASES** | Wavenumber integration for arrays and elastic seismo-acoustics: TL (OAST), array covariance/replicas (OASN), reflection coefficients (OASR), pulse/broadband (OASP), reverberation from rough interfaces (OASS), broadband scattered-field realizations (OASSP). |
| **Bounce** | Not a propagation model — tabulates a bottom reflection coefficient `R(θ)` to a `.brc`/`.irc` file for the others to consume. |

### Capability matrix

Verified against each model's own flags, read from a constructed instance:
`model.supported_modes` / `model.supports_mode(mode)` for the run modes,
`model.supported_features` / `model.supports_feature(name)` for the env
shapes. Ask the instance, not `spec.supports`: some flags are resolved
from constructor arguments (see the Bellhop row).

| Model | Run modes | Range-dep. | Elastic | Altimetry | Broadband |
|---|---|---|---|---|---|
| **Bellhop** | TL (coh/incoh/semicoh), RAYS, EIGENRAYS, ARRIVALS, BROADBAND, TIME_SERIES | bathy + SSP (`interp_ssp` `None`/`'quad'`) + bottom | yes | yes | yes |
| **Kraken** | COHERENT_TL, MODES, INCOHERENT_TL, BROADBAND, TIME_SERIES | bathy + SSP + bottom + layers (field path; MODES reduces to one profile by the collapse policy: deepest bathymetry, median seabed, SSP r = 0, unless `collapse=` sets the quantity) | yes (`krakenc`) | no | yes |
| **Scooter** | COHERENT_TL, BROADBAND, TIME_SERIES | no (collapsed) | yes | no | yes |
| **RAM** | COHERENT_TL, BROADBAND, TIME_SERIES | bathy + SSP + bottom + layers | yes (`rams`) | yes (`ramsurf`) | yes |
| **SPARC** | TIME_SERIES | no (collapsed) | no | no | (native pulse) |
| **OAST** | COHERENT_TL | no (collapsed) | yes | no | no |
| **OASN** | COVARIANCE, REPLICA | no | yes | no | (multi-freq sweep) |
| **OASR** | REFLECTION | no | yes | no | (multi-freq sweep) |
| **OASP** | COHERENT_TL, BROADBAND, TIME_SERIES | no | yes | no | yes |
| **OASS** | REVERBERATION, COVARIANCE | no | yes | no | no |
| **OASSP** | BROADBAND, TIME_SERIES | no | yes | no | yes |
| **Bounce** | REFLECTION | no | yes | no | no |

All range-independent models support a layered seabed
(`supports_feature('layered_bottom')`) natively; "Range-dep." above is the horizontal axis only.

### Environment feature support and collapse

Each model declares, per axis of `Environment` shape, whether it handles that
feature natively. These answer one question only: *does this env shape work
with this model?* Query them with `supports_feature(name)`, or list them all
with `supported_features`:

```python
Bellhop().supports_feature('range_dependent_ssp')                 # True
Bellhop(interp_ssp='linear').supports_feature('range_dependent_ssp')  # False
Scooter().supported_features
# ['elastic_media', 'layered_bottom', 'multi_source_depth', 'rough_surface', 'volume_attenuation']
```

The names are `altimetry`, `range_dependent_bathymetry`,
`range_dependent_ssp`, `range_dependent_bottom`, `layered_bottom`,
`elastic_media`, `multi_source_depth`, `source_beam_pattern`,
`rough_surface`, `rough_bottom` and `volume_attenuation`; anything else raises
`ConfigurationError`, so a typo cannot come back as a quiet `False`.
`volume_attenuation` is declarative only: it answers whether the engine
honours `env.absorption` along a propagation path (OASR does not — its water
is a lossless half-space; Bounce writes it into its deck, where AT adds it to the sediment media it
tabulates, but answers `False` because a reflection table has no path for the
lossless-water notice to describe), drives no collapse, and `env.absorption`
passes through untouched either way.

`multi_source_depth` says whether the engine writes every source depth into
one deck (Bellhop in its three TL modes, `RAYS` and `ARRIVALS`; Kraken in its
two TL modes and `MODES`; Scooter in `COHERENT_TL` — see each model's
`spec.traits.native_multi_depth_modes`). It does not gate what a caller may pass: in the field
modes (`COHERENT_TL` / `INCOHERENT_TL` / `SEMICOHERENT_TL` / `BROADBAND` /
`TIME_SERIES`) **every model accepts a multi-depth `Source` and returns one
`Field` with a leading `source_depth` axis** — `PropagationModel.run` runs the
engine once per depth with the same environment, receiver and keywords (or
the engine carries every depth in one deck) and joins the fields on that
axis. Outside the field modes only Bellhop takes several depths (`RAYS` /
`ARRIVALS` / `EIGENRAYS`, one `ResultStack` slab per depth) and Kraken's
`MODES` (one mode set tabulated at every depth); the other engines raise
`ConfigurationError`. Whether the field then **superposes** coherently
depends on what it holds: complex pressure (the coherent engines, and OASP /
OASSP among the OASES family) adds, while a dB-only engine such as OAST cannot
add its sources coherently — `field.superpose()` refuses and says so, and
`superpose(coherent=False)` adds them in intensity.

When you hand a model an environment richer than it supports, the collapse
step **collapses** the unsupported feature to something the model can run,
emitting one `FallbackWarning` per dropped feature so nothing
happens silently. For example, Scooter has no range-dependent SSP, so a 2-D SSP
is reduced to one profile (its per-model default: the range-mean).

The reduction is governed by a per-feature policy you can override with
`Model(collapse={...})`:

```python
# Force Scooter to use the near-source SSP instead of the range-mean,
# and the maximum depth for a sloping bottom.
scooter = Scooter(collapse={'ssp': 'r0', 'bathymetry': 'max'})
```

Keys and values: `bathymetry` (`max`/`median`/`mean`/`min`/`r0`/`rmax`), `ssp`
(`r0`/`rmax`/`mean`/`median`), `bottom_range` (`r0`/`rmax`/`mean`/`median`),
`bottom_layers` (`halfspace`/`top_layer`/`volume_average` — the last is the
thickness-weighted *arithmetic* mean of c, ρ and dB/λ over the layers and the
half-space, the half-space weighted by the deepest layer's thickness: 1 m of
1500 m/s mud over 5250 m/s basalt collapses to 3375 m/s), `surface`
(`r0`/`rmax`/`mean`/`median`), `altimetry` (`drop`, on the engines that cannot
read a sea-surface shape — Kraken, Scooter, SPARC, Bounce, OASES; Bellhop and
RAM's `ramsurf` read altimetry natively), `elastic` (`fluid`/`vacuum`).

Warn versus raise: an unknown key or an invalid method raises
`ConfigurationError` at construction, not deep inside a writer at `run()`
time, and a reduction with no defined answer raises too — averaging seabed
columns that carry *different* reflection files, since tables cannot be
blended. A collapse the policy *can* perform only ever warns. A key whose
feature the environment carries and the model reads natively (RAM's or
Bellhop's range-dependent bathymetry, RAM's layers) reduces nothing in that
model's deck, and the run says so with a `FallbackWarning`. The full
policy — the seven keys in detail, the per-model default overrides, the
order the seabed's two axes collapse in, and why the design is a reduction
rather than a refusal — is the environment guide's
[§7 Collapse policy](docs/guide/environment.md#7-collapse-policy).

---

### Bellhop

Gaussian beam / ray tracing with three backends. `backend=None` (default)
auto-selects the fastest installed binary (CUDA → C++ → Fortran); an explicitly
requested backend that isn't installed raises `ExecutableNotFoundError` naming
the `./install.sh --bellhop …` flag that builds it. They are not the same code — `cxx`/`cuda` port `A-New-BellHope`,
a bug-fixed fork of the Fortran BELLHOP that `backend='fortran'` runs. Measured
`fortran` against `cxx`, |ΔTL| is 0.002 dB at p99 (0.03 dB max) on the ASA
wedge at 25 Hz, and 0.096 dB at p99 on the 100 Hz sloping shelf of the
[Kraken page](docs/models/kraken.md) (100 to 400 m over 20 km), where 16 of 14 634 cells differ by
more than 1 dB (10.4 dB at most); the mean difference is 0.002 dB, no bias.

```python
import numpy as np, uacpy
from uacpy import Bellhop, RunMode

env = uacpy.Environment(
    bathymetry=200,
    ssp=uacpy.SoundSpeedProfile.from_pairs([(0, 1500), (200, 1520)]),
    bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=1600, density=1.8, attenuation=0.5),
)
source = uacpy.Source(depths=50, frequencies=200)
receiver = uacpy.Receiver(depths=np.linspace(1, 199, 60),
                          ranges=np.linspace(100, 10_000, 200))

bellhop = Bellhop(backend='cxx', n_beams=500, launch_angles_deg=(-80, 80))
field = bellhop.compute_field(env, source, receiver)           # Field
rays = bellhop.compute_rays(env, source, receiver)          # Rays
arr  = bellhop.compute_arrivals(env, source, receiver)      # Arrivals

# Eigenrays to a single point:
target = uacpy.Receiver(depths=[100.0], ranges=[5000.0])
eig = bellhop.compute_eigenrays(env, source, target)
```

Ray travel times are frequency-independent, so a single trace yields a
broadband result. Pass a real source pulse to synthesise `p(t)`:

```python
ts = bellhop.compute_time_series(env, source, receiver,
                                 source_waveform=pulse, sample_rate=fs)
```

Layered bottoms auto-route through BOUNCE (a `.brc` reflection table) unless
`Bellhop(auto_bounce=False)`; a non-layered elastic half-space runs natively —
bellhop.f90 evaluates its exact acousto-elastic reflection coefficient, per
range node on a range-dependent bottom. Range-dependent SSP needs
`Bellhop(interp_ssp='quad')` (the default auto-picks it).

To pin the BOUNCE tabulation itself rather than let `auto_bounce` size it, call
`run_with_bounce` — the one sanctioned per-call escape hatch, because the
phase-velocity window belongs to the reflection table, not to Bellhop:

```python
tl = bellhop.run_with_bounce(env, source, receiver,
                             c_low=1400.0, rmax=20_000.0)
```

Everything after `receiver` is keyword-only, `run_mode=` included; it otherwise
takes the same call arguments as `run()` and returns the same `Result`. Left at
`c_low=None`, the minimum phase velocity is resolved by `Bounce` as
`min(1400, min(env.ssp))` — bounce.htm's "lowest speed in the problem" — so a
cold or brackish column resolves below 1400 m/s, and the reflection table
starts at that lower phase speed. The seafloor water speed alone does not set
it: a slow layer anywhere in the column does.

### Kraken

Normal-mode solver. `backend='kraken'` (real) or `'krakenc'` (complex: elastic
media, attenuation, leaky modes); `backend=None` auto-picks `krakenc` when the
env carries shear or `leaky_modes=True`, and `kraken.exe` otherwise. Above a
fluid seabed's speed `kraken.exe` solves the modes as if the seabed were rigid
(`Kraken/BCImpedanceMod.f90:83-88`), and `backend='krakenc'` computes them as
leaky modes (`kraken.htm`). The default window reaches above that speed and a
default run says nothing of it; `backend='kraken'`, or a `c_high` above the
seabed speed, gives one `ValidityWarning`, and `run_settings(...).engine.backend`
records the program that ran. The choice is yours (see
[Kraken §3](docs/models/kraken.md#real-or-complex-arithmetic-kraken-vs-krakenc)). `field.exe` runs **only** for field
modes — `compute_modes` stops at the modes binary:

```python
from uacpy import Environment, Kraken

kraken = Kraken(n_modes=30)                             # the mode cap is a model setting
modes = kraken.compute_modes(env, source)               # Modes (k, phi); no receiver
print(modes.k.shape, modes.phi.shape)

field = kraken.compute_field(env, source, receiver)        # modes -> field.exe -> Field

# Range-dependent field via coupled modes:
env_rd = Environment(bathymetry=[(0, 100), (5_000, 160)],   # a wedge
                     ssp=env.ssp, bottom=env.bottom)
field_rd = Kraken(mode_coupling='coupled', n_segments=20).compute_field(env_rd, source, receiver)
```

`run()` defaults to a field mode (`COHERENT_TL`, or `BROADBAND` when the
call itself passes `run(..., frequencies=[...])` with more than one element; a
`Source` listing several frequencies does not switch it, and is refused in the
narrowband default) — call `compute_modes` for the modes
themselves. `RunMode.INCOHERENT_TL` sums the modes in power instead of
amplitude; it cannot be combined with `mode_coupling='coupled'` on a
range-dependent env (the solver rejects that pairing, so `run()` raises
`ConfigurationError` up front) — use `'adiabatic'` for an incoherent
range-dependent field. `'coupled'` is not the more exact of the two:
`field.exe` marches it forward only and projects only the pressure onto each
new segment's modes, so it does not conserve energy and moves with
`n_segments` instead of converging — on the shelf of
[kraken.md §6.6](docs/models/kraken.md) it sits 7.4 dB above a refined RAM
run in the upper 100 m past 15 km, where `'adiabatic'` sits 0.15 dB below.
Check either against RAM on a slope. Kraken segments range-dependent bathymetry, SSP and
bottom natively (each profile carries the seabed column at its own range,
layers included; `compute_modes` reduces to one profile by the collapse policy — deepest bathymetry, median seabed, SSP and surface at r = 0 — unless `collapse=` sets the quantity);
altimetry is not supported (Bellhop and RAM's `ramsurf` backend support it).

### Scooter

Finite-element FFP (wavenumber integration), range-independent. Computes the
exact field — including layered/elastic bottoms and leaky energy — then
Hankel-transforms the Green's function to range.

```python
from uacpy import Scooter

scooter = Scooter()
field = scooter.compute_field(env, source, receiver)           # Field
hf = scooter.compute_transfer_function(env, source, receiver,
                                       frequencies=np.linspace(180, 220, 64))
```

`c_low` defaults to 0.90 × the slowest wave of the problem — water,
sediment, shear speeds, half-spaces, and every seabed Scholte
wave, solved from Brekhovskikh & Lysanov eq. 3.4.29 (`scooter.f90:258`;
Jensen et al. ch. 4 App. 1, "the lowest wave speed in the problem"); a
pinned `c_low` above a Scholte wave the geometry reaches gives a
`NumericsWarning`. `c_high` defaults to the unbounded `1e9`
(`k_min = 0`, Jensen et al. ch. 4 App. 1) over every bottom. The
spectral `RMax` is set from `receiver.range_max × rmax_factor` (2.0 in
every run mode, the toolbox's Scooter decks). `RMax` sets the wavenumber sampling
`Δk ≈ π/(2·RMax)`; the k→r step is a direct trapezoidal DFT, periodic in range
with period `2π/Δk ≈ 4·RMax`, so the multiplier keeps that wrap-around replica
beyond the receivers.

### RAM

Parabolic-equation solver and a **multi-backend dispatcher**: it routes to
`mpirams` (Dushaw's mpiramS: fluid, flat surface — the broadband-native default), `rams` (elastic
bottom), `ramsurf` (rough surface / altimetry), or `ramgeo` by environment
shape. Inspect the choice with `select_backend(env)`; force one with
`RAM(backend=...)`.

```python
from uacpy import RAM

ram = RAM(n_pade=6)
print(ram.select_backend(env))                              # e.g. 'mpirams'
tl = ram.run(env, source, receiver)                         # COHERENT_TL -> Field
```

Broadband works on every backend (the Collins binaries loop in Python, mpiramS
sweeps natively). The band is derived from `(fc, q_factor, record_duration)` (mpiramS's `(fc, Q, T)`): half-bandwidth `fc/q_factor` (the sweep
spans `fc ± fc/q_factor`, full width `2·fc/q_factor`), `Δf = 1/record_duration`. `q_factor`/`record_duration` default to narrowband `(1e6, 1.0)` for `COHERENT_TL`; on
the broadband paths they are resolved from the source (a bare single frequency
expands to the same default band as on Bellhop, Kraken and Scooter — 128 bins
over `fc·(1 ± 0.25)` — and an explicit `run(frequencies=[fc])` is one bin; see
the knob table in `docs/models/ram.md` §7). Pass a
multi-element `frequencies=` array to set
the band explicitly — the returned `H(f)` carries exactly the requested bins
(the internal sweep is trimmed onto them). On narrowband run modes a
`frequencies=` argument raises `ConfigurationError` (the frequency comes from
the `Source`):

```python
ts = ram.compute_time_series(env, source, receiver,
                             source_waveform=pulse, sample_rate=fs)
```

`env.absorption` reaches every RAM backend through **uacpy-patched binaries**:
Collins' codes assume a lossless water column, so the vendored ramgeo, rams and
ramsurf read an optional water-column attenuation block per range section and
mpiramS a depth × frequency table (`uacpy/third_party/MODIFICATIONS.md`,
*water-column attenuation*). A stock build misreads a deck carrying the block,
so rebuild with `install.sh` after updating; see `docs/models/ram.md` §9.

### SPARC

Time-marched FFP (Porter 1990) — wavenumber integration marched in time to
produce `p(t)` directly, without the per-frequency synthesis the other models
use. Not a parabolic equation: it makes no one-way approximation. The pulse comes from
the constructor `pulse_type` (a 4-character AT code); `output_mode` selects the
geometry: `'R'` horizontal array, `'D'` vertical array, `'S'` snapshot.

```python
import numpy as np
from uacpy import SPARC, Receiver

# SPARC's deck carries only vacuum / rigid floors; a half-space bottom
# raises ConfigurationError, so the floor is chosen explicitly.
env_rigid = uacpy.Environment(bathymetry=200, ssp=env.ssp,
                              bottom=uacpy.BoundaryProperties(acoustic_type='rigid'))
sparc = SPARC(output_mode='R', n_time_samples=1024)
rcv_sparc = Receiver(depths=50.0,                          # one march per depth
                     ranges=np.linspace(200, 1_200, 24))
ts = sparc.compute_time_series(env_rigid, source, rcv_sparc,
                               output_duration=1.0)    # Field of p(t), 1 s record
```

A half-space bottom is refused rather than replaced: a rigid floor in its
place is a different waveguide (7–10 dB louder over 1–9 km on the default
seabed, up to 34 dB on clay, measured with Scooter). Use Scooter for a
half-space.

`'R'` marches one subprocess per receiver depth, so a dense depth axis queues
one solve per depth; `SPARC(max_launches=N)` caps it (a denser receiver then
raises `UnsupportedFeatureError`) — use a coarser depth axis here than the
grids the other models take.

SPARC marches the `source_waveform` a run is handed (the `'FN+B'` STSFIL route)
and takes `output_duration` as its `time_max`, the end of its record, unless
`pulse_type` is pinned; with no waveform it builds the canned `'PN+B'` pulse.
Range-independent: range-dependent envs collapse.

### Bounce

Reflection-coefficient writer, not a propagation model. It tabulates `R(θ)`
over the phase-velocity band `[c_low, c_high]` and emits a `.brc` (and `.irc`)
file the other models consume. Pin `work_dir` (with `cleanup=False`) so the
files outlive the call:

```python
import tempfile
from uacpy import Bounce

with tempfile.TemporaryDirectory() as work_dir:
    bounce = Bounce(c_low=1400, work_dir=work_dir, cleanup=False)
    rc = bounce.compute_reflection(env, source, receiver)   # ReflectionCoefficient
    brc_path = rc.metadata['brc_file']

    # Chain into another model while the table is still on disk:
    env2 = env.copy()
    env2.bottom = uacpy.Bottom.from_halfspace(
        uacpy.BoundaryProperties(acoustic_type='file', reflection_file=brc_path))
    tl = Scooter().compute_field(env2, source, receiver)
```

Bellhop does this transparently for *layered* bottoms via `auto_bounce`; use
Bounce directly when you want control over the table or to feed
Scooter/krakenc.

### OASES

OASES is the abstract base for the wavenumber-integration sub-models —
instantiate the one for the run mode (docs/models/oases.md §2). All OASES knobs
are constructor-only.

```python
from uacpy import OAST, OASN, OASR, OASP, OASS, OASSP, RunMode

# OAST — transmission loss
tl = OAST().run(env, source, receiver)                      # Field

# OASN — array covariance C(f, i, j) and matched-field replicas
oasn = OASN(surface_noise_level_dB=70.0)
cov  = oasn.compute_covariance(env, source, receiver)       # Covariance
rep  = oasn.compute_replicas(env, source, receiver)         # Replicas

# OASR — plane-wave reflection coefficients
refl = OASR(angles_deg=np.linspace(0, 90, 91)).run(env, source, receiver)

# OASP — broadband pulse / transfer function
oasp = OASP(n_time_samples=256, freq_max=500)
probe = uacpy.Receiver(depths=[50.0, 150.0], ranges=[1000.0, 5000.0])
hf   = oasp.run(env, source, probe, run_mode=RunMode.BROADBAND)   # a few points: a pulse
                                                                   # on the 60×200 grid takes minutes

# OASS / OASSP scatter from a rough interface, and the roughness belongs to
# the environment: a smooth env is refused before any binary runs, and
# rms_roughness= only overrides the scattering deck's copy of it.
rough_env = uacpy.Environment(
    bathymetry=200,
    ssp=uacpy.SoundSpeedProfile.from_pairs([(0, 1500), (200, 1520)]),
    bottom=uacpy.BoundaryProperties(acoustic_type='half-space', sound_speed=1600,
                                    density=1.8, attenuation=0.5, roughness=0.5),
)
scat_rcv = uacpy.Receiver(depths=[50, 150], ranges=[500, 1000, 1500])

# OASS — reverberation from the rough seabed (runs OAST first, then oass2)
oass = OASS(correlation_length=10.0)
reverb = oass.run(rough_env, source, scat_rcv)              # Field, kind='reverberation'

# OASSP — one realization of the scattered field (runs OASP first, then oassp2).
# n_time_samples=512 keeps oassp2 at 2813 wavenumbers here (~40 s); the default
# time window would need 5204, past the 4096 its arrays hold, and is refused.
oassp = OASSP(correlation_length=5.0, realization=0, n_time_samples=512)
h_scat = oassp.run(rough_env, source, scat_rcv)             # Field, complex H(f)
```

OASES models handle layered and elastic seabeds natively (their strength for
seismo-acoustics), but are range-independent — range-dependent envs collapse.

One output caveat: **OAST is the only TL engine that returns a real dB
`Field`** — its `.plt` output carries `|p|` in dB and no phase, so `field.p`
raises and off-grid receiver ranges are interpolated *in dB* (with a warning:
that smears sharp interference nulls). Every other coherent-TL path — Bellhop,
Kraken, Scooter, RAM, and OASP's `COHERENT_TL` — returns complex pressure in
Pa, from which `.dB` derives the same TL (§8). Use OASP when you need the
phase or exact null depths from an OASES run.

Two more payload notes. `INCOHERENT_TL` (and Bellhop's `SEMICOHERENT_TL`)
store **real dB TL** on every engine: a magnitude sum has no phase, so the
result claims none — `.dB`/`.tl` read it and `.phase` refuses. And across the complex-pressure engines the **phase convention is
uniform**: every coherent complex `Field` carries
`result.phase_reference == PhaseReference.TRAVELLING_WAVE` (`'travelling_wave'`) — the outgoing
travelling-wave convention, verified engine-by-engine against Scooter — with
a line source additionally carrying the 2-D Green's function's `e^{−iπ/4}` and, like the point source, normalised to unit amplitude at 1 m in free space on every engine (see the source-receiver guide).

### Running in parallel

Model runs are independent subprocess computations, so batches are
embarrassingly parallel. Build a `Job` per run — each carries its own model,
scenario, and run mode, so cross-model comparisons and parameter sweeps are the
same operation — and hand the list to `run_parallel`:

```python
from uacpy import Bellhop, Kraken, RAM, run_parallel, Job, RunMode

# In a .py script the `if __name__ == '__main__':` guard is required, not
# stylistic: workers start with 'spawn' and re-import the script's __main__, so
# an unguarded module-level run_parallel call re-enters itself in every worker
# and the pool dies. A REPL, Jupyter and `python -c` need no guard; code piped
# on stdin cannot run a pool at all (the workers cannot re-read it).
if __name__ == '__main__':
    jobs = [
        Job(Bellhop(), env, source, receiver, run_mode=RunMode.COHERENT_TL, label='bellhop'),
        Job(Kraken(),  env, source, receiver, run_mode=RunMode.COHERENT_TL, label='kraken'),
        Job(RAM(), env, source, receiver, run_mode=RunMode.COHERENT_TL, label='ram'),
    ]
    batch = run_parallel(jobs, n_workers=3)                 # ParallelResult
    for label, result in zip(['bellhop', 'kraken', 'ram'], batch):
        print(label, result.dB.min())

    # Single-model sweep -> one Field with a leading 'n_pade' axis:
    sweep = [Job(RAM().copy(n_pade=p), env, source, receiver, label=p)
             for p in (4, 6, 8)]
    swept = run_parallel(sweep, coordinate_name='n_pade').stack()
```

`ParallelResult` is indexable/iterable (`.results`, `.errors`, `.ok`);
`.to_dataframe()` is one row per job (`label`, `ok`, `error_type`, `n_warnings`);
`.stack()` bundles successful same-model results along one new axis — the
leading one, except a `frequency` or `time` sweep, whose axis goes last as in
every producer's layout: `Field` results into one `Field` (they must share a
grid), other results
(rays, arrivals, modes) into a `ResultStack`. A joined `Field` whose jobs'
run records differ (a knob sweep) has `run_settings=None`, each job's record
staying on `results[i].run_settings`; a sweep that leaves the records equal
keeps the one record its jobs share (attenuation on Kraken, Scooter, Bellhop
or OAST; RAM's record holds the grid it resolves from the environment, so a
sweep that changes that grid — attenuation, which sets its absorbing layer's
depth — drops it, and a bottom-density sweep keeps it). Jobs on
different waveguide sound speeds do not join (`stack()` refuses them):
iterate `results`.
Workers
that own their scratch dir (`cleanup=True`, the default) drop on-disk artifacts —
pin a unique `work_dir` per job with `cleanup=False` to keep them. Example 50
runs a frequency sweep this way.

## 8. Results

Every `model.run(...)` returns a typed `Result`. The concrete type is chosen by
the run mode, not by a flag you pass — it carries exactly the payload that mode
produces, plus the slicing and convenience methods that make sense for it.

| Run mode | Result type | Payload |
|---|---|---|
| `*_TL`, `TIME_SERIES`, `BROADBAND` | `Field` | gridded pressure / TL / H(f) / p(t) |
| `ARRIVALS` | `Arrivals` | flat list of arrival events |
| `RAYS` / `EIGENRAYS` | `Rays` | ray polylines |
| `MODES` | `Modes` | depth eigenfunctions + wavenumbers |
| `COVARIANCE` | `Covariance` | OASN hydrophone covariance |
| `REPLICA` | `Replicas` | OASN MFP replica field |
| `REFLECTION` | `ReflectionCoefficient` | R(θ[, f]) |
| `REVERBERATION` | `Field` | OASS reverberation loss (`kind='reverberation'`, dB) |

All of them subclass `Result`, so every result carries the same identification
fields: `result.model`, `result.backend`, `result.source_depths`,
`result.frequencies` (1-D, length-1 for narrowband; `result.f0` is the scalar
shortcut), `result.phase_reference`, `result.run_mode` (the `RunMode` the model
ran; `RunMode` members are strings, so `result.run_mode == 'coherent_tl'` also
holds), `result.run_settings` (every setting the run resolved), the source's
`result.source_level_dB` and `result.source_weights`, the engine's
`result.model_source` provenance, the `result.components` it was built from,
and a `result.metadata` dict of what no setting states.

A result carries its own identity and provenance — never the carriers it was run
against. The `Environment`, `Source` and `Receiver` stay yours to keep, and the
plotters take them explicitly:

```python
tl = Bellhop().run(env, src, rcv, run_mode=RunMode.COHERENT_TL)
tl.plot(env=env, source=src, receiver=rcv, title="Transmission Loss")
```

Pass `env=` whenever you want the depth axis to span the full water column with
the seabed drawn: on its own, `tl.plot()` can only span the receiver grid it was
sampled on, which stops short of the seafloor whenever your receivers do.

### Field — one container described on three axes

`Field` is the **single** gridded result type. What it means is not a
constructor argument — it is described on three independent axes, derived from
the data unless a model tags otherwise:

| axis | question | ask it when you need |
|---|---|---|
| `.kind` | *what* is this? | to know whether comparing two fields means anything |
| `.unit` | what is it *measured in*? | to know whether the data is in dB (which way is louder in dB is the kind's: `is_loss`) |
| `.data.dtype` | how is it *stored*? | to know whether there is phase to work with |

There is no `Field.dtype`; the storage axis is read off `.data.dtype`, or as the
boolean `.is_complex`.

| `.data` dtype | `.coords` keys | `.kind` | `.unit` | meaning |
|---|---|---|---|---|
| real | `{depth, range}` | `pressure` | `dB` | transmission loss |
| complex | `{depth, range}` | `pressure` | `Pa` | narrowband p(d, r) |
| complex | `{depth, range, frequency}` | `pressure` | `Pa` | broadband H(d, r, f) |
| real | `{time}` or `{…, time}` | `pressure` | `Pa` | p(t) |
| real | `{time}` or `{…, time}` | `impulse_response` | `1/s` | band-limited impulse response h(t) (no source waveform) |
| any of the above | `{source_depth, …}` | `pressure` | as above | a multi-depth run: one plane per source depth, summed by `.superpose()` |

Transmission loss is not a separate kind: `-20·log10|p|` is the same pressure
field written in dB, which is what `.unit` records. That is why a RAM TL field
and a Kraken complex field compare on one colour scale. Models producing a
different *quantity* tag it — OASS reverberation, or `signal_excess` and
`probability_of_detection` from the [sonar equation](docs/guide/sonar.md).

Two dB quantities run backwards: transmission loss and OASS reverberation are
*losses* (OASES writes reverberation as `-10·log10 E|p_scat|²`), so less is
louder. Signal excess and an ambiguity surface share their dB unit but are
**levels**, where more is more. Which way is louder is declared once,
`quantities.LOSS_KINDS` (read through `is_loss(kind)`), which is why `.max()`
consults the kind as well as the unit, never the unit alone.

`.data.shape` matches the insertion order of `.coords` (canonical order
`source_depth → depth → range → frequency|time`).

```python
field.dB          # dB; -20·log10(|data|) for complex, data as-is if already real (raises AttributeError on a time trace)
field.p           # complex pressure / H(f) (raises if data is real — phase gone)
field.magnitude   # |data|;  field.phase  → angle in rad  (complex only)
field.data        # raw ndarray
field.coords['range']           # axis vectors, always 1-D, metres / Hz / s
field.ranges, field.depths, field.times    # shorthand accessors
field.kind        # 'pressure' (or a model-tagged quantity, e.g. 'reverberation')
field.unit        # 'Pa' | 'dB'          -- is the data in dB
field.data.dtype  # real | complex        -- is there phase to work with
field.is_complex  # whether phase survives
field.n_depths, field.n_ranges, field.n_frequencies, field.n_times   # axis lengths
```

`.kind` is the physical quantity; the result's class (`Field` / `Rays` /
`Modes` / …) is its *category*, and is what tells the plot dispatcher which
view to build.

Derived views return new `Field`s and never re-run the solver (`to_dB()` on
a field that is already real returns that field itself):

```python
field.to_dB()                              # → the same kind in unit='dB'
field.mask_below_seafloor(env.bathymetry)  # NaN out the sub-seafloor cells
field.resample_to(depths=np.linspace(0, 100, 60),
                  ranges=np.linspace(100, 5_000, 200))   # onto another grid
                                           # (any other axis carried; plain
                                           # arrays: acoustics.resample_depth_range)
Field.from_dict(field.to_dict())           # round-trip the raw (data, coords) pair
da = field.to_xarray()                     # labelled xarray.DataArray (extra: uacpy[xarray])
Field.from_xarray(da)                      # and back: coords, pinned, kind/unit, identity
field.to_netcdf('field.nc')                # NetCDF; complex data as real/imag parts
field.view('level')                        # any plotted view by name, read-only
stack.to_netcdf('rays.nc')                 # a ResultStack: one NetCDF group per slab
Arrivals.from_dict(arr.to_dict())          # arrivals as plain per-arrival columns
arr.to_dataframe()                         # one row per arrival (also Rays, Modes,
                                           # ReflectionCoefficient)
```

`np.savez(path, **x.to_dict())` saves any of them, and
`from_dict(dict(np.load(path, allow_pickle=True)))` reads it back — for files
you wrote only: `allow_pickle=True` unpickles, which runs whatever code the
file carries. Exchange files through `to_netcdf` / `from_netcdf`, whose class
paths are checked before anything is imported. A reader handed a file of
another class (`Result.from_netcdf` on an Environment's file,
`Environment.from_netcdf` on a result's) refuses, naming what the file holds
and the call that reads it.

Two more need a `time` axis — they read a `TIME_SERIES` field back into the
frequency domain: `field.to_transfer_function()` is the spectrum on the grid
the trace was synthesised from (on the record's bins for any other trace) —
`H(f)` in `'Pa'` from an impulse response, and from a pressure trace (a
`TIME_SERIES` field) the received spectrum `S(f)·H(f)`, a density in `'Pa·s'`
(`.broadband_loss()` refuses it: its band mean carries the source spectrum)
— and `field.extract_tone(frequency=200)` pulls the steady-state
complex pressure at one frequency back out, dropping only the time axis (a
`{depth, range, time}` field gives a `{depth, range}` map, a single trace one
phasor).

### Slicing — `.at`, `.isel`, `.max` drop axes into `.pinned`

Selecting along an axis **removes** that axis from `coords` and records the
chosen coordinate value in `.pinned`. The result is still a `Field`, so you can
chain and then plot.

```python
H = bellhop.run(env, src, rcv, run_mode=RunMode.BROADBAND, frequencies=freqs)
H.kind, H.unit               # 'pressure', 'Pa'; coords = {depth, range, frequency}

narrow = H.at(frequency=200)        # nearest-label select
narrow.coords                       # {'depth': ..., 'range': ...}
narrow.pinned                       # {'frequency': 199.6}  (nearest sample)
narrow.kind                         # 'pressure'

line = narrow.at(depth=50)          # coords = {'range': ...}, a 1-D cut
peak = H.max()                      # loudest point (argmax |p|; min finite dB for TL) → scalar Field
peak.pinned                         # {'depth': 50.0, 'range': 3000.0, 'frequency': 59.6}
```

`.at(**labels)` picks the nearest grid value (never fabricates); `.isel(**indices)`
takes a positional index; `.eval(**labels, method='linear')` *interpolates*
(`'linear'` / `'nearest'` / `'cubic'`) when you want a value between samples;
`.max()` collapses every axis at the loudest cell: the largest `|data|` for
linear data, the least value of a dB loss (TL, reverberation), the largest of a
dB level (signal excess). `.loudest_along('depth')` applies the same rule
along one axis only, point by point — per range, the worst-case depth of an
exposure map — and keeps a Field over the other axes. Slicing takes the
place of per-shape result types: you slice a `Field` down to 1-D/2-D, then
hand it to one plotter.

Two methods narrow an axis instead of collapsing it: `.window(time=(lo, hi))`
drops the samples outside an inclusive label range and **keeps** the axis
(`None` for either end leaves it alone; a window that keeps nothing raises),
and `.shift(time=delta)` translates a coordinate without touching the data.
Together they put several models on one display axis — move an IFFT model's
emission to `t=0`, then cut every trace to the same window. `.remove_delay(delay=)` /
`.remove_delay(sound_speed=)` is the frequency-domain twin: it advances `H(f)`
by a delay you name, or by `r/c` taken from the field's own range (per range if
the axis survives). There is no default — a `Field` carries `r` but not `c`. The
magnitude is untouched. `.removed_delay` records the delay, and every synthesis
from the advanced `H` puts it back first, so its traces sit on the true time.

`at` / `isel` is the **grid-library convention** shared across the whole API
(`at` = nearest, `isel` = positional): `Field` (a multi-depth run's field
along its `source_depth` axis too), `ResultStack`, `ReflectionCoefficient`,
and the input carriers `SoundSpeedProfile`,
`Bathymetry`, `Altimetry`, `Bottom` and `Surface` all provide the pair. `eval`
(= interpolate) exists only where interpolating between samples is meaningful —
`Field`, `ReflectionCoefficient`, `SoundSpeedProfile`, `Bathymetry`,
`Altimetry`; `Bottom`, `Surface` and `ResultStack` hold discrete columns/slabs
that cannot be blended, so they stop at `at`/`isel`. For example
`env.ssp.at(depth=50)` is the nearest stored sample and
`env.ssp.eval(depth=50)` interpolates.

### Metadata and output-file paths

`result.metadata` holds model-specific extras that no setting states (reader
outputs such as Bounce's `n_points`, OAST's `oast_grid_shape`); the
settings a run resolved — RAM's grid and Padé point, Kraken's phase-speed
window and `mode_coupling` — are in `result.run_settings.engine`, once. When you construct a model
with a **pinned** `work_dir`, the solver's on-disk outputs are kept and their
paths are recorded in the metadata:

```python
ram = RAM(work_dir='/tmp/run1')
r = ram.run(env, src, rcv)
r.metadata['psif_file']              # '/tmp/run1/psif.dat' — the PE field the run wrote
r.list_metadata()['psif_file']
                                     # {'value_type': 'str', 'documented_type': 'str',
                                     #  'description': 'psif.dat broadband output ...'}
r.run_settings.engine.c0             # Padé expansion point c0 the PE was initialised at
```

Names say what the number *is*, not just where it came from: RAM's Padé
point is the setting `c0` (an algorithmic expansion point, not a medium
speed), while the physical speeds a Field states are named members of `Field.speeds`
(a `SoundSpeeds` record). Bellhop's `surface` is
the sea-surface sound speed of the first profile.

Which keys appear depends on the model: only the Acoustics-Toolbox binaries
(Bellhop, Kraken, Scooter, SPARC, Bounce) write a diagnostic `.prt` log, so
`prt_file` exists on their results and not on RAM's or OASES'. Rather than
assume, read the keys back — `result.list_metadata()` describes every key
attached (runtime type, documented type, one-line meaning) so you do
not have to grep the source.

### Phase reference — why IFFT needs it

A broadband `Field` tags how its complex `H(f)` is phased, via
`result.phase_reference` (a `PhaseReference`):

- **`travelling_wave`** — `H(f)` carries the engineering propagator
  `exp(-i k0 r)`, so `2·Re[ifft(H)]` lands the causal arrival at `t = r/c0`.
  Used by Bellhop, Scooter, OASES, Kraken, and RAM.
- **`time_domain_native`** — SPARC already solved in the time domain; `H(f)` is
  just the FFT of an existing `p(t)`. To get a time series, read the
  `TIME_SERIES` `Field` directly instead of inverse-transforming.

This is the one bit of provenance the time-series helpers consult before
inverting — `time_domain_native` fields refuse `impulse_response`,
`to_time_trace` and `synthesize_time_series` and tell you to read the native
trace instead.

### Broadband → time series

Two paths, both starting from a broadband `Field` (`coords = {depth, range,
frequency}`):

**(a) Helpers (recommended).** They honour `phase_reference`, resolve bin
resolution internally, and land the arrival at the right time even on a coarse
frequency grid.

```python
# one cell's impulse response h(t) → time-domain Field, coords = {'time': ...}
trace = H.to_time_trace(depth=50, range=5000, sample_rate=fs)

# convolve every cell with a source waveform → Field, coords = {depth, range, time}
ts = H.synthesize_time_series(source_waveform=p_src, sample_rate=fs)
p = ts.at(depth=50, range=5000).data        # the received pulse
t = ts.times                                # seconds
```

The record is `1/Δf` long and sampled at exactly `sample_rate`: `fs/Δf`
samples when that is whole, the next whole number above it otherwise, every
sample the band's sum at its own instant (an inverse FFT on the lattice, a
chirp-z off it). A waveform whose support is longer than `1/Δf` is refused
(`ConfigurationError`): the synthesis would return it folded onto itself.
`H.impulse_response(fs)` and `.apply(source_waveform=..., sample_rate=fs)`
convolve instead, linearly, so the record has to hold only the channel; on a
range grid `t_start='reduced'` gives each range its own record on the reduced
time `t − r/c` (JKPS eq. 8.15), `.reduction_speed` recording `c`.

**The level a signal with bandwidth or duration reaches** is a different
question from either, and has two named answers:

```python
# the transmission loss OF A SIGNAL — hand it the waveform
# (Ainslie Eq. 11.46 / total path loss; Abraham's pulse L_p)
loss = H.broadband_loss(source_waveform=my_chirp, sample_rate=fs)

# or of a plain band, with no particular waveform in mind
loss = H.window(frequency=(f0 - B / 2, f0 + B / 2)).broadband_loss()

# sound exposure level — the time integral of p(t)² (Abraham sect. 3.2.1.5,
# ISO 18405). No source-level argument: the level rides on the amplitude, so
# the map carries no source_level_dB; it records ref**2 as sel.reference.
sel = H.sound_exposure_level(source_waveform=unit_rms * 1e-6 * 10 ** (SL / 20),
                             sample_rate=fs)
```

`broadband_loss` is **not** `RunMode.INCOHERENT_TL`, which sums intensities
over paths instead of averaging over the band, and which only Bellhop and
Kraken offer (Scooter refuses it). Bellhop's sums its ray arrivals: Ainslie's
Eq. 11.47, incoherent "in the ray sense", which he marks invalid when the sonar
or target is within a few wavelengths of the sea surface. Kraken's sums its
modes (Jensen et al. Eq. 5.18) — the "mode sense", which Eq. 11.47 is not.
Neither method gates: both keep a late path's energy and remove only its interference. See
[guide/results.md §6a](docs/guide/results.md) and example 45.

To **plot** one cell directly, `H.at(depth=50, range=5000).plot_transfer_function()`
draws stacked modulus-in-dB (`20·log10|H|`, top) and phase (radians, bottom)
panels on a log frequency axis — `plot_frf`'s convention for a measured FRF —
(reduce-then-plot, so a single-receiver field needs no `.at()`), and
`trace.plot_impulse_response()` draws the impulse response `h(t)` (1/s) it is
handed — the `trace` above; an `H(f)` is refused, since the plotter computes no
synthesis. Use `synthesize_time_series` for the response to a specific source
pulse.

**(b) Manual `2·Re[ifft(H)]`** for `travelling_wave` `H(f)`, when you want full
control. A DFT of spacing `Δf` holds only multiples of `Δf`, so place the first
model frequency at bin `round(f[0]/Δf)` and each next one a bin later: the band
then sits on the lattice offset by one common `offset = f[0] − b0·Δf`, which a
rotation by `exp(2πi·offset·t)` takes back out. Every default BROADBAND grid
(`linspace(0.75 fc, 1.25 fc, 128)`) starts half a bin off the lattice, and
rounding each frequency to its own bin without the rotation synthesises the
right envelope on a carrier shifted by `Δf/2`. The window length `1/Δf` must
exceed the arrival time `r/c0`, so the grid must be fine enough:

```python
import numpy as np
f = H.coords['frequency']
spec1d = H.at(depth=50, range=5000).data       # complex H(f) at the cell
df = f[1] - f[0]                                # need 1/df > r/c0
b0 = int(np.round(f[0] / df))                   # the first sample's bin
offset = f[0] - b0 * df                         # the band's common offset
nfft = 1 << int(np.ceil(np.log2(2 * (b0 + f.size) + 2)))
buf = np.zeros(nfft, complex)
buf[b0 + np.arange(f.size)] = spec1d
t = np.arange(nfft) / (nfft * df)
pt = 2.0 * np.real(np.fft.ifft(buf) * np.exp(2j * np.pi * offset * t)) \
    * (nfft * df)                               # Riemann sum of ∫H df
#  → peak of pt sits at t ≈ r/c0
```

This is `H.to_time_trace(depth=50, range=5000, sample_rate=nfft * df,
t_start=0.0)` written out
(equal to within 1.1e-14 of the peak); `acoustic_signal.synthesize_time_series` is the
same synthesis of a waveform through an `H` held as a plain array.

For a unit-amplitude phase-only `H = exp(-i 2π f r/c0)` with `r = 3000 m`,
`c0 = 1500 m/s`, this lands the impulse exactly at `t = 2.0 s`.

### Filtering helpers (sparse results)

The non-grid results return new copies — no solver re-run:

```python
cell = (receiver.depths[30], receiver.ranges[99])   # one receiver of the grid
one = arr.at_receiver(receiver=cell)         # Arrivals of that one cell
arr.filter_by_bounces(kind='surface')        # Arrivals: by multipath class
arr.window(delay=(2.0, None)).top_n_by_amplitude(5)
arr.sorted_by_amplitude()                    # loudest first
arr.delays, arr.amplitudes, arr.phases       # the ray-arrival triple
arr.received_amplitudes                      # what ARRIVES: absorption + phase
arr.rms_delay_spread(receiver=cell)          # pulse smearing, energy-weighted; one cell (receiver= on a grid, nothing on one cell)
one.synthesis_band(bandwidth=100.0, energy_fraction=1.0)  # grid whose record holds every arrival
arr.channel_taps(50.0, fc=200.0, sps=1, pulse=None, rolloff=0.25, span=8, receiver=cell, normalize=False)  # ChannelTaps: baseband modem channel at a symbol rate; pulse None = 'rc' (raised cosine, the channel at the decision instants) at sps=1, 'rrc' above, or 'nearest'
arr.coherence_bandwidth(convention=None, factor=None, receiver=cell)  # 1/(k·τ_rms), k=1 unset ('rappaport_0.5' k=5, 'rappaport_0.9' k=50, or factor=k; not both)
arr.channel_regime(50.0, convention=None, factor=None, rolloff=0.0, receiver=cell)  # ChannelRegime: whether that symbol rate sees the channel flat
rays.window(launch_angle_deg=(-10, 10))         # Rays: pure data subsets
rays.top_n_by_miss(20, target_depth=50, target_range=5000)
rays.filter_by_miss_distance(50.0, target_depth=50, target_range=5000)
rays.sorted_by_miss(target_depth=50, target_range=5000)
rays.first_n(10)                       # first N traced rays
rays.truncate_at_receiver(target_depth=50, target_range=5000)
modes.first_n(10).phase_speeds     # Modes: trim + derive v_p
modes2 = kraken.compute_modes(env, uacpy.Source(depths=50, frequencies=205))
modes.group_speeds_between(modes2)         # dω/dk — needs a 2nd frequency's Modes
modes.with_attenuation(alpha_dB_per_m=0.01,  # perturbational modal attenuation
                       sound_speed_z=env.ssp.sound_speed_at(modes.depths),
                                            # c(z) is required: α_m scales as 1/c
                       bottom=env.bottom)    # bottom= runs the normalisation
                                            # into the half-space; without it
                                            # the .mod's recorded fluid
                                            # half-space is used
modes.modal_pressure_field(source_depth=50, receiver_depths=receiver.depths,
                             ranges=receiver.ranges)   # → complex pressure; .dB is TL
modes.excitation(source, sound_speed=1500.0) # what a source ARRAY drives:
                                            # Σₙ wₙ·φₘ(zₙ), one amplitude per
                                            # mode — the mode filter. Exact in
                                            # the waveguide, unlike a free-field
                                            # pattern. sound_speed= is required
                                            # when the Source is directional:
                                            # the excitation is shaded by
                                            # f(θₘ), θₘ against that speed
                                            # (Kraken's field.exe uses 1500,
                                            # and S(0°) below it)
rc.at(angle=30)                              # ReflectionCoefficient: nearest sample
rc.is_broadband                              # is there a frequency axis to select on?
```

The two sparse Bellhop results name a place differently because they ask
different questions. An `Arrivals` method's `receiver=(depth, range)` picks an
existing cell of the result's receiver grid (and may be omitted when the run
had one receiver). A `Rays` method's `target_depth=` / `target_range=` is a
free point the miss distance is measured to, on or off the grid; `None` is the
receiver the result was built for.

`rc.at(angle=30, frequency=200)` needs a **broadband** reflection coefficient —
the 2-D `(angle, frequency)` table `OASR` produces. Anything `Bounce` returns is
single-frequency by construction, so `frequency=` raises `ConfigurationError`
there; `rc.is_broadband` is the guard.

### The field of several sources

A run with a **multi-depth `Source`** in a field mode (the TL modes,
`BROADBAND`, `TIME_SERIES`) returns **one `Field`** on every model, its
leading `source_depth` axis holding the field of each unit-amplitude source:
`(source_depth, depth, range)` for TL, `(source_depth, depth, range,
frequency)` for `H(f)`, `(source_depth, depth, range, time)` for traces (every
depth on one time axis: an engine that opens each depth's record at its own
earliest arrival runs every depth again at the earliest of those starts, and
`run_settings.notes` says so — pass `t_start=` to choose it yourself).
`source_depths` lists the depths and `source_weights` the `Source` weights.
Both follow a selection: `window(source_depth=…)` keeps the weight of each
depth it keeps, so its `superpose()` sums those planes, and one plane (`at`,
`isel`, `eval` on `source_depth`) is one source's field, with no weights.

```python
field = model.run(env, Source(depths=[40, 60], frequencies=200,
                              weights=[1, -1]), receiver)
field.coords['source_depth']          # [40., 60.]
field.at(source_depth=40)             # one source's field (nearest label)
field.isel(source_depth=1)            # by position
field.data[0]                         # one source's raw plane
field.plot()                          # one panel per source depth
field.superpose()                     # Σ wᵢ·pᵢ with the Source's weights → one Field
field.superpose([1, 1j])              # the same with weights given here
field.superpose(coherent=False)       # √Σ|wᵢ·pᵢ|² → one real-dB Field
uacpy.stack_fields([f40, f60], [40, 60])  # one-source Fields run apart → the same Field
```

`.dB`, `.tl`, `.p` and `.plot()` read the unit-amplitude planes and warn
when the `Source` carried other weights. A metadata entry the depths
disagree on — each depth's own `*_file` scratch paths when the run keeps
its work directory — holds one value per depth. `superpose` records what it
added in the Field's `superposed_sources` attribute (written by `to_dict` /
`to_xarray` and read back), and `coherent=` says **how the
sources combine** — a statement about the sources, not about the arithmetic:

* `coherent=True` (default) adds complex pressure, `Σ wᵢ·pᵢ`: the sources are
  driven together with a fixed relative phase, as the elements of one array
  are. The sum keeps the field's `phase_reference` (time-domain traces add
  with real weights). A field of real dB values has lost its phase and
  refuses here — superpose the complex field the run returned, or add the
  intensities instead.
* `coherent=False` adds intensity, `√Σ|wᵢ·pᵢ|²`: the sources are mutually
  incoherent (separate platforms, unrelated tones, random relative phase), so
  their phases carry no information and only `|w|` is read. N identical
  sources give `10·log10(N)` where a coherent sum gives `20·log10(N)`. The
  result has no phase, so it comes back the way every engine returns its own
  incoherent mode — real dB, `phase_reference` cleared — and a **dB-only
  field combines this way**, since magnitudes are all it needs (the power sum
  is `uacpy.acoustics.sum_levels_dB`'s). Time-domain traces refuse: intensity
  does not add sample by sample. A field of *levels* adds as levels, not as
  losses, so two incoherent 120 dB sources give 123.01 dB.

### ResultStack

The sparse results cannot be one array — each source's rays or arrivals
number differently — so a multi-depth Bellhop `RAYS` / `ARRIVALS` /
`EIGENRAYS` run returns a `ResultStack`: one slab per source depth, every
slab the same result type. `run_parallel(...).stack()` bundles non-Field
results the same way. A `Field` is never a slab.

```python
for src_depth, rays in stack: ...     # iterate (coordinate, slab) pairs
stack.at(source_depth=20)             # nearest-label slab
stack[0], stack.isel(source_depth=1)  # positional slab
stack.n_slabs, stack.slab_type        # how many slabs, and of what result type
stack.to_dataframe()                  # every slab's rows, the coordinate a column
stack.to_netcdf('rays.nc')            # one NetCDF group per slab; Result.from_netcdf reads it
```

A `ResultStack` is a `Result`: its identity is the one its slabs share —
they must agree on `model`, `backend`, `run_mode`, `phase_reference`,
`source_level_dB` and `source_weights`, as the fields `stack_fields` joins
must; `run_settings` is the shared record, or `None` when the slabs' records
differ (a knob sweep), each slab keeping its own — and it
saves and loads as every result does (`to_dict` / `from_dict`, npz,
`to_netcdf` / `from_netcdf`). Its slabs share no grid, so it is no single
xarray object — `to_xarray()` refuses and names `stack[i].to_xarray()`.

A weight therefore refuses at one place and waits at another, and the
difference is *when* it is applied rather than whether it could be. A
**single-depth** `Source(weights=…)` is applied by `run()` itself, which
refuses a field whose phase is gone at that moment. A **multi-depth** one is
applied by `superpose`, which has not been called yet and may be called
either way — so a dB-only multi-depth field is returned, and only a complex
weight (nothing could rotate a phase that is not there) is refused up front.

Note what the level means. TL is referenced to a unit source at 1 m, and a
superposed field keeps that reference, so the array's gain is *inside* the
number: three in-phase sources read several dB "less TL" than one. That is
the total propagation loss re one unit source at 1 m (plots caption it so),
which is normally what an array
calls for — it is not the TL of a single source. Because the array gain is
inside it, it can fall below a single source's TL, and it goes negative where
the gain exceeds the loss; subtract it from a source level, never add it. Scale the weights (e.g.
`Σ|wᵢ|² = 1`) to normalise the array as a source, or give the `Source` a
`source_level_dB` and read `superpose(...).at_source_level()`, which is an
absolute level and so carries no such ambiguity. This is also why
`field.superpose(coherent=False)` and an engine's own `INCOHERENT_TL` are
different quantities: the engine sums the intensities of one source's own
paths (Bellhop's ray arrivals, Kraken's modes), this sums *sources*.

### File I/O (`uacpy.io`)

The models talk to their binaries through `uacpy.io`, and the same
readers/writers are public: point them at a file another tool produced, or at
the outputs a pinned `work_dir` left behind. `import uacpy` exposes them as
`uacpy.io.*`.

One rule governs the whole subpackage: **every public reader and writer speaks
metres and Hz at the Python boundary, degrees for angles, named so
(`source_angles_deg`, `bearings_deg`, `rams_rotation_angle_deg`), and radians for phase
everywhere.** The km and on-disk angle axes the
file formats want are converted inside, by the public `uacpy.units` helpers. So
`write_ssp(filepath, ranges, sound_speed)` takes metres even though the `.ssp` format stores
km, and `read_ssp_2d` hands its `ranges` back in metres.

| Family | Readers | Writers |
|---|---|---|
| AT field / rays / arrivals | `read_shd_file` (`read_shd_bin`, `read_shd_asc`), `read_ray_file`, `read_arr_file` | — |
| AT modes | `read_modes` (`.mod` or `.moa` → `Modes`) | `write_fieldflp` |
| AT env (`.env`) | `read_env` (→ Environment, Source, Receiver, options), `read_prt`, `read_flp` | `write_bellhop_env_file`, `write_kraken_env_file`, `write_scooter_env_file`, `write_sparc_env_file`, `write_bounce_input_file`, `write_multi_profile_env` |
| AT env building blocks (`uacpy.io.oalib_writer`; staging in `uacpy.io.refl_io`) | — | `write_header`, `write_ssp_section`, `write_layer_sections`, `write_bottom_section`, `writable_layers`, `write_source_depths`, `write_receiver_depths`, `write_receiver_ranges`, `write_phase_speed_and_rmax`, `write_absorption_block`, `write_fg_params`, `write_bio_layers`, `write_broadband_freqs`, `resolve_ssp_interp`, `resolve_ssp_topopt` |
| Boundaries (`.bty`/`.ati`/`.ssp`/`.brc`/`.trc`/`.irc`/`.sbp`) | `read_bathymetry`, `read_altimetry`, `read_ssp_2d`, `read_reflection_coefficient` (returns a `ReflectionCoefficient`), `read_source_beam_pattern` (→ `BeamPatternTable`: `angles_deg`, `levels_dB`); the BELLHOP3D grid `read_boundary_3d` (→ `Boundary3dFile`: `x`, `y`, `depths`) | `write_bty_file`, `write_bty_long_format`, `write_ati_file`, `write_ssp`, `write_reflection_coefficient`, `write_source_beam_pattern`; staging in `uacpy.io.refl_io`: `stage_reflection_file`, `stage_source_beam_pattern`, `dedupe_reflection_file` |
| Scooter / SPARC (`.grn`, `.rts`, `STSFIL`) | `read_grn_file` (→ `GreensFunction`: `to_field`, `to_transfer_function`, `snapshot_to_field`, `snapshot_to_time_field`, `wavenumbers`), `read_rts_file` (one frequency's pressure from it: `uacpy.models.sparc.rts_to_pressure`) | `write_sparc_source_time_series` |
| OASES (`.dat`, `.trf`, `.rhs`) | `read_oast_tl` (→ a list of TL `Field`s, one per frequency), `read_oasp_trf` (→ a pressure `Field`), `read_oasr_reflection_coefficients` (→ `ReflectionCoefficient`), `read_oasn_covariance` (→ `Covariance`), `read_oasn_replicas` (→ `Replicas`), `read_oases_rhs_header` | `write_oast_input`, `write_oasp_input`, `write_oasr_input`, `write_oasn_input`, `write_oass_input`, `write_oassp_input` |
| RAM (`in.pe`/`ram.in`, `psif.dat`, `tl.line`, `tl.grid`, `pcomplex.bin`) | `read_psif`, `read_tl_line`, `read_tl_grid`, `read_pcomplex_grid` | `write_mpirams_inpe`, `write_ram_in`, `write_mpirams_ssp`, `write_mpirams_bathymetry`, `write_mpirams_ranges`, `write_mpirams_sediment`, `write_water_attenuation_file` |
| Audio (`.wav`) | `read_wav` (`start=` / `frames=` read one span), `read_wav_info` (→ `WavInfo`: `encoding`, `bits`, `n_channels`, `sample_rate`, `n_frames`), `read_wav_metadata` | `write_wav` |

Anything malformed raises `FileFormatError` (§4), never a bare `ValueError`.

```python
import uacpy

bellhop = uacpy.Bellhop(work_dir='/tmp/run1', cleanup=False)
r = bellhop.compute_field(env, source, receiver)
shd = uacpy.io.read_shd_file(r.metadata['shd_file'])   # the raw .shd behind the Field
```

## 9. Visualization

The convention is uniform: **every result and every *shape* carrier plots
itself** — results (`tl.plot()`, `rays.plot()`, `arrivals.plot()`,
`modes.plot()`, …) and the carriers that reduce to a curve (`env.plot()`,
`env.ssp.plot()`, `env.bathymetry.plot()`, `env.altimetry.plot()`,
`Thorp().table(freqs).plot()`). The *property* carriers `Bottom` and
`Surface` have
no `.plot()`: a stack of boundary types is not a single curve, and drawing it
needs the seafloor an `Environment` supplies — use `plot_bottom_properties(env)`
or `env.plot()` instead.

The free `plot_*` functions below are the remaining public surface: the
type-dispatcher, the grid/flexible renderers, alternate views, composition
helpers, geographic maps, animation, and the raw-array DSP/comms plotters.
`uacpy.plot` is their one public path: every plotter is reachable, after
`import uacpy`, as an attribute of the `uacpy.plot` module
(`uacpy.plot.compare`, `uacpy.plot.plot_psd`, …), or by
`from uacpy.plot import …`. The two coastline calls
`land_polygons` and `download_coastline` are not plotters and live on
`uacpy.visualization` only.

| Function | Use |
|---|---|
| `result.plot(...)` / carrier `.plot()` | preferred — the object plots itself (results dispatch via `plot_result`, carriers via `plot_carrier`) |
| `plot_result(result, env=…)` | type-dispatch — the function a result's `.plot()` calls |
| `plot_carrier(carrier)` | type-dispatch — the function a carrier's `.plot()` calls |
| `plot_rays(rays)` / `plot_arrivals(arrivals)` / `plot_mode_shapes(modes)` / `plot_reflection_coefficient(rc)` / `plot_covariance(covariance)` / `plot_replicas(replicas)` / `plot_source_depth_panels(field)` | the view each result's `.plot()` draws, public so `help()` lists its keywords; `plot_source_depth_panels` is a multi-depth run's field, one panel per source depth |
| `plot_field(field, env=…, source=…, receiver=…)` | auto-shape a (sliced) Field: 1 surviving axis → line, 2 → heatmap; `source=`/`receiver=` mark the geometry on a (depth, range) cross-section |
| `H.plot_transfer_function()` / `plot_transfer_function(H)` | stacked modulus-in-dB + phase at one receiver cell (reduce with `.at(depth=, range=)` first; a single receiver plots directly) |
| `ir.plot_impulse_response()` / `plot_impulse_response(ir)` | an impulse response `h(t)` (1/s) at one cell, as `H.to_time_trace(depth=…, range=…, sample_rate=fs)` or `H.impulse_response(fs).at(depth=…, range=…)` returns it; an `H(f)` is refused |
| `plot.compare(fields, labels)` | overlay several 1-D sliced fields on one axes (`uacpy.plot.compare`) |
| `compare_models(fields, labels, env=…, title=…)` | side-by-side heatmaps, one shared colourbar; `title=` sets the figure title above the panels |
| `metrics.tl_difference(field, reference).plot(env=…, vmax=…)` / `plot.plot_field_difference(difference)` | `field - reference` in dB (a `Field`, `kind='difference'`), diverging map symmetric about zero; positive means `field` carries the higher loss. `tl_difference` refuses two fields that are not on one grid (`metrics.same_grid`) |
| `metrics.field_statistics(fields, labels, depth=…).plot()` / `plot.plot_field_statistics(statistics)` | mean and standard deviation per field at one depth, drawn as two separate bars (not error bars), plus the pairwise RMS-difference matrix; `None` entries (a model that did not run) are dropped |
| `plot.shared_colorbar(fig, axes, label=…)` | one colorbar for panels drawn with `show_colorbar=False`, taken from their own mappable; refuses panels that are not on one colour scale |
| `env.plot()` / `plot_environment(env)` | SSP + seafloor cross-section, optional `source=`/`receiver=` markers |
| `ssp.plot()` / `env.ssp.plot()` / `plot_ssp(ssp)` / `plot_ssp(c, depths=z)` | sound-speed profile `c(z)` as a depth-down line (one per range if range-dependent); bare speeds take `depths=` (and `ranges=` for a 2-D array). `label=` / `color=` overlay several profiles on one `ax=`: an explicit colour draws the whole profile in it and drops the range colourbar, and the label names the profile rather than each column. `show_legend=` forces the legend on or off |
| `bathymetry.plot()` / `altimetry.plot()` / `plot_range_profile(profile)` | seafloor depth / sea-surface height vs range — the shape carriers |
| `Thorp().table(f).plot()` / `plot_absorption(coefficient)` / `plot_absorption(alpha, frequencies=f)` | volume absorption `α(f)` (dB/km, log-log), or an α(f, z) heatmap when the carrier has a depth axis. The plotter draws a carrier or a bare dB/km array (`frequencies=`, and `depths=` for a 2-D one) and computes nothing: a law is refused, so build a carrier with the law's `.table(...)`. Every plotter draws what it is handed: the weighting curve `plot_weighting(f, auditory_weighting(f, group))`, the ROC `sonar.roc_curve(d).plot()`, the bottom loss `bottom_loss_curve(material).plot(value='loss')`, the BER theory `comms.ber_theory_curve(scheme, ebn0_dB).plot(ax=…)` |
| `plot_bottom_properties(env)` | seabed `c` / `ρ` / `α` vs depth, per layer stack |
| `acoustics.bottom_loss_curve(material, water_sound_speed=…).plot(value='loss', label=…, mark_critical=True, ax=…)` | plane-wave bottom loss vs grazing angle of a fluid seabed (a preset name, a property dict, a seabed carrier or an unlayered `Bottom`), one call per seabed on one `ax=` against one `water_sound_speed` — the critical angle is the ratio of the two speeds, so curves drawn against different water are not comparable. `mark_critical=True` rules the critical angle the table carries; a seabed slower than the water has none (an angle of intromission instead) and gets no rule |
| `modes.excitation_pattern(source, sound_speed=…).plot()` / `plot_mode_excitation(pattern)` | what a source array drives, both descriptions on one angle axis: a stem per mode at its grazing angle, height `|Σₙ wₙ·φₘ(zₙ)|` — the modal excitation, which is what sets the field in a waveguide — over the same array's free-field pattern: `Source.array_beam_pattern` gives `P(θ)=f(θ)·A(θ)`, the product theorem's element pattern times `Source.array_factor`'s point-source factor (Butler & Sherman §7.1.1). The two agree while the pattern is symmetric in ±θ and diverge once steering breaks that, so both are drawn |
| `source.plot_beam_pattern()` / `plot_beam_pattern(pattern)` | source directivity from a `BeamPatternTable` (`read_source_beam_pattern`) or an `(N, 2)` `(angle, level)` array; polar by default, oriented like the field (0° = increasing range, +angle downward) and spanning the propagating half-plane. `polar=False` gives level-vs-angle, `mirror=True` reflects a half-defined table |
| `beams.plot()` / `plot_beam_power(beams)` / `plot_beam_power(angles_deg, power)` | a scanned beam's power against **look** angle, from the `BeamformedField` `beamform_field` returns or an angular spectrum's arrays (Bartlett, MVDR, MUSIC). The receive dual of `plot_beam_pattern`, which is a *launch* fan and labels its axis so; `at=` selects the grid point or frequency bin, `normalize=False` keeps the absolute level, `label=` overlays processors |
| `plot_mode_wavenumbers(modes)` / `plot_modes_heatmap(modes, mode_numbers=…)` / `plot_mode_attenuation(modes)` | Re and Im of the modal wavenumbers against mode index (twin axes) · mode shapes as a heatmap (chosen modes by 1-based number) · `|Im k_m|` against phase speed with the seabed speed ruled |
| `plot_mode_speeds(modes)` / `uacpy.Modes.dispersion(modes_by_frequency).plot()` (`plot_dispersion`) | phase speed per mode index, with the group speed where the backend filled it and the seabed speed the result records (or `seabed_sound_speed=`) marking the trapped count · the same two speeds against frequency, computed from a **sequence** of `Modes` (one per frequency, sorted by each result's own `f0`) as a `ModeDispersion` |
| `plot_greens_function(grn, modes=…)` / `acoustics.wavenumber_sampling(f, c_low=…, c_high=…, delta_k=…, rmax=…).plot()` (`plot_wavenumber_sampling(ws)`) | `|G(k_r, z)|` from a Scooter `.grn`, with Kraken's eigenvalues marked on it · the wavenumber axis that transform is sampled on — the phase-speed window against the water and seabed `ω/c`, and whether the receivers fit inside the alias period `2π/Δk` |
| `plot_signal_excess(field)` / `plot_detection_probability(field)` / `sonar.roc_curve(d).plot()` (`plot_roc(curve)`) | `uacpy.sonar` field maps and the ROC curve; `draw_contours(ax, pd, levels=[0.5], label='Pd = 0.5')` overlays the contours of a second Field on a map |
| `plot_track_exposure(field, ax=None, *, positions=None, depth=None, contours=None)` | the plan view of a moving source's cumulative SEL (the map `Field.track_exposure(..., along=)` returns), with its line, its events and labelled contours (an exposure criterion, say); that map's `.plot()` draws it |
| `plot_bathymetry_map(grid)` or `plot_bathymetry_map(lats, lons, depth)` / `plot_sea_ice_map(grid)` | geographic maps (also the pluggable `map_fn=` of `plot_overview`); the `BathyGrid` `fetch_bathy_grid` returns and the `SeaIceGrid` `sea_ice_grid` returns are taken whole and credit their source (`grid.plot()` is the same call) |
| `land_polygons(resolution='50m')` / `download_coastline(cache_dir=None)` | the Natural Earth land rings those maps draw behind the data, and the call that caches them offline — both take `url=` for a mirror |
| `plot_overview(env, map_data, tl=…, title=…)` | three-panel map + TL + environment composite; `map_title`/`tl_title`/`env_title` name the panels, `title=` the figure |
| `animate_field(field)` / `save_animation(field, path)` / `plot_time_snapshots(fields, times)` | time-domain animation and its still-frame grid |

**Credits follow the result.** A plotter writes the model that produced what
it draws (`Model: Bellhop (fortran) — Michael B. Porter`)
and the data sources of a fetched `env` it draws into the grey footnote of the
figure it draws in — its own, or yours when you pass `ax=` / `fig=`. The
footnote holds everything drawn into the figure, each model and source once;
a result with no model provenance and bare arrays add nothing
([plotting guide §5](docs/guide/plotting.md#5-composition-with-ax)).

**Branch-relevant kwargs only.** `plot_field` picks its rendering branch from
the surviving coords and then **rejects** any keyword that branch cannot use,
raising `ConfigurationError` that names the coords it saw — it is never silently
ignored. `vmin`/`vmax`/`cmap`/`show_colorbar`/`contours` belong to the 2-D
heatmap and `env`/`source`/`receiver`/`seabed` to a `(depth, range)`
cross-section (`seabed='line'` draws the `env` seafloor as a line, so values
below it stay visible; the default fills the sediment over them);
`label=` belongs to the 1-D line cut and `stacked`/`stack_offset`/`normalize`
to the stacked traces of a 2-D field with a `time` axis (`normalize='trace'`,
the default, scales each trace to its peak; `'global'` keeps true amplitudes). So `plot_field(tl, env=env)` is right
and `plot_field(tl.at(depth=50), env=env)` raises — slice first, then drop the
heatmap-only kwargs. The same rule holds for the views that are not fields:
`env=` reaches `Field` and `Rays`, and passing it to `Modes` / `Arrivals` /
`ReflectionCoefficient` / `Covariance` / `Replicas` / `GreensFunction` is an
error rather than a no-op.

**DSP / comms plotters.** All signal-processing and communications plotting also
lives here (the `acoustic_signal` and `comms` modules are pure computation and
import no matplotlib). Each takes the arrays a transform/estimator returns as
its leading positional arguments, then an optional `ax` — always passable by
keyword, and best passed that way, because its position follows the number of
data arrays (`plot_psd(frequencies, psd_linear, ax)` but
`plot_spectrogram(frequencies, times, Sxx, ax)`). Most also take the result
carrier itself in place of its arrays (`plot_psd(welch(x, fs))`). All return
`(fig, ax)`, except `plot_frf`, which draws magnitude over phase and returns
`fig` and the `(ax_mag, ax_phase)` pair, and `plot_lsfir_diagnostics(fit)` (an `lsfir`
result, or its `information_matrix`, `information_vector`, `impulse_response`),
which takes no `ax` — it builds its own three-panel figure, or fills a `fig=`
handed to it, and returns `fig` and its three panels; every multi-panel
plotter returns its panels as a numpy array of Axes (`fig, (ax_mag, ax_phase) = …`
unpacks it).

- **Waveforms:** `plot_waveform` — a bare 1-D signal and its sample rate
  against time; `value='envelope'` / `'envelope_dB'` draw
  `acoustic_signal.envelope` instead of the carrier, and `t0=` states the
  first sample's time. A gridded result plots itself: `Field({'time'}).plot()`,
  and one trace's `.plot(value='envelope')` / `'envelope_dB'` draws the same
  envelope views.
- **Spectra / levels:** `plot_psd`, `plot_ppsd`, `plot_spectrogram`,
  `plot_band_levels` (band levels and banded exposure),
  `plot_cumulative_exposure` (the running SEL of a record).
- **Gather transforms:** `plot_fk`, `plot_radon`, `plot_taup` (+ `draw_sound_cone`,
  `draw_slowness_line` overlays).
- **Time-frequency:** `plot_cwt`, `plot_wigner_ville`, `plot_cepstrum`.
- **Constant-Q:** `plot_constant_q_transform`, `plot_constant_q_spectrogram` (a `constant_q` estimate
  and its histogram are drawn by `plot_psd` / `plot_ppsd`).
- **Arrays / active / system-ID:** `plot_beam_power`, `plot_ambiguity`
  (`value='dB'` for the sidelobes, which sit tens of dB down), `plot_matched_field`
  (the ambiguity `Field` of `sonar.bartlett` / `sonar.mvdr` over a
  `replica_bank` or of `Covariance.bartlett` / `.mvdr`, dB re its own peak;
  example 38), `plot_frf`, `plot_coherence`, `plot_lsfir_diagnostics`.
- **Comms:** `plot_channel`, `plot_channel_response`, `plot_constellation`,
  `plot_received_symbols`, `plot_eye_diagram`, `plot_ber_curve`, `plot_convergence`,
  `plot_sync_metric`, `plot_doppler_ambiguity`, `plot_subcarriers`.

Slicing replaces specialised plotters: there is no separate "TL line" vs "TL
heatmap" function — you slice a `Field` to 1 or 2 axes and `plot_field`
auto-selects the shape. Plotting a field with 3+ surviving axes raises and tells
you to slice first.

**Fixed TL colour scale.** Every TL heatmap defaults to `vmin=20`, `vmax=120` dB
so figures are directly comparable across runs and models. Pass
explicit `vmin=`/`vmax=` to override.

### (a) Single TL field with seafloor + contours

```python
import uacpy
tl = bellhop.compute_field(env, src, rcv)     # Field, coords = {depth, range}; .dB → dB
fig, ax = uacpy.plot.plot_field(
    tl, env=env,            # env= overlays the seafloor on the (depth, range) heatmap
    contours=[60, 80, 100], # dB contour lines
)
```

`plot_field` puts range (km) on X, depth (down) on Y, applies the TL colormap on
the fixed 20–120 dB scale, and draws the seafloor from `env`. Slice first to get
a line cut instead:

```python
uacpy.plot.plot_field(tl.at(depth=50))          # TL vs range, 1-D
uacpy.plot.plot_field(tl.at(range=5000))        # TL vs depth, depth down
```

### (b) Multi-model comparison, one shared colourbar

```python
fields = {
    'Bellhop': bellhop.compute_field(env, src, rcv),
    'Kraken':  kraken.compute_field(env, src, rcv),
    'RAM':     ram.compute_field(env, src, rcv),
}
fig, axes = uacpy.plot.compare_models(fields, env=env, contours=[80, 100])
```

All panels share the fixed 20–120 dB scale and a single right-hand colourbar, so
differences between models read directly. `compare_models` warns if the depth or
range axes differ between fields (a shared colourbar would mix sample grids). For
1-D overlays (e.g. TL vs range from several models at one depth) slice each field
and use `uacpy.plot.compare([f.at(depth=50) for f in ...], labels=[...])`.

## 10. Signal Processing

A processing toolbox for the waveforms that propagation models emit or
consume. Named `acoustic_signal` so it never shadows the stdlib `signal`.
Transforms and estimators are **pure functions** returning arrays (or a small
named tuple such as `SpectralEstimate`); `FRF` is the one retained class, and
it holds a configuration of the `frf_welch`/`etfe`/`periodic_etfe`/`lsfir`
functions, not a result. Every plotter is reached through `uacpy.plot`
(`uacpy.plot.plot_psd`, `uacpy.plot.plot_fk`, …), not through the computation
modules.

| Area | Public names |
|------|--------------|
| Waveforms | `lfm_chirp`, `hfm_chirp`, `tone_burst`, `gaussian_pulse`, `ricker_wavelet`, `sparc_pulse`, `nwave` |
| Coded probes | `m_sequence` (also `uacpy.comms.m_sequence`, the same function), `make_mseq_probe`, `bpsk_modulate` |
| Noise synthesis | `make_bandlimited_noise` (unit-RMS, or in Pa at an absolute in-band density `psd_level_dB`), `synthesize_noise_from_psd` (any spectrum, at its absolute level); noise relative to a signal's power is `uacpy.comms.awgn` |
| Spectral / levels | One estimator per statistic: `welch` and `constant_q` (both `scaling='density'` or `'spectrum'`), `sound_exposure` (the ISO 18405 band energy), and a histogram twin of each — `probabilistic_welch`, `probabilistic_constant_q`, `probabilistic_sound_exposure` (→ `SpectralEstimate` / `ProbabilisticSpectralEstimate`, whose Welch and exposure forms keep `.segment_levels_dB`, `.segment_times` and `.percentiles(q)`), and on plain arrays `level_histogram` (the dB-level histogram every histogram twin builds) and `level_percentiles`. Each takes only the parameters its own statistic can honour: `sound_exposure` has no window, overlap, detrending or averaging, because a band sum is the band's energy only when every bin is counted once and whole, and asking a bin estimator for `scaling='exposure'` raises and names it. `freq_min`, `freq_max`, `integration_time` and `axis` mean the same thing in all six: `axis` is the time axis of a multichannel record, along which `welch`, `constant_q` and `sound_exposure` return one spectrum (one band row) per channel, frequency last, and the histogram twins pool the channels. `.level_dB(ref)` on a `SpectralEstimate` (and on a `SpectrogramResult` or a `CQSpectrogramResult`) is its power as a level in dB re `ref` (1 µPa by default), `power_to_dB` of it. `BAND_TYPES` names the ladders (`'decidecade'` by default, also `'octave'` — both IEC 61260-1 base-10 — and `'linear'`). Two result types cover the family: a name that fixes one scaling would say "density" over band power |
| Band ladders | `decidecade_bands` (ISO 18405) and `octave_bands` (every third decidecade; `base=2` for exact octaves), both IEC 61260-1 base-10, `standard_bands` (the bands `sound_exposure` reports a range on, per `band_type`); `band_levels` integrates a PSD (one spectrum, or one per channel along `axis=`) onto either ladder, returning a `BandLevels` (`centers`, `levels` with the bands last, the band edges, `band_type` and `ref`; `.plot()` draws it) |
| Arrays | `steering_vectors`, `snapshots` (→ `Snapshots`, whose `.covariance()` is `uacpy.sonar.csdm` of its data), `bartlett` and `mvdr` (the Bartlett / MVDR power of any rows of element weights — steering vectors or matched-field replicas — with `normalize=`), `music_spectrum` (the same `weights` rows and `normalize='max'`), `shading_taper`, `beamform_field` (→ `BeamformedField`, whose `.snr_dB(source_level_dB, noise_level_dB)` gives the SNR in dB) |
| Arrays over a field | `beamform_field` (→ `BeamformedField`: beam power over a whole `(n_elements, *grid)` field, with `.best`, `.best_angle_deg`, `.array_gain()` and the named `grid_coords` it was given), `realised_array_gain` (what `.array_gain()` returns, on plain arrays), `plane_wave_array_gain` (`|Σw|²/‖w‖²`), `matched_replica_gain` (the `10log10(N)` ceiling), `independent_beams` (orthogonal looks in a scanned sector — pair with `per_look_false_alarm`) |
| Active / pulse compression | `matched_filter`, `pulse_compression`, `processing_gain_dB`, `ambiguity_function` (→ `AmbiguityResult`) |
| Time-frequency | `spectrogram` (the short-time PSD), `stft` (the complex short-time transform on the same segments), `analytic_signal`, `envelope`, `instantaneous_frequency`, `wigner_ville`, `cwt`, `inverse_cwt`, `cepstrum`, `complex_cepstrum`, `inverse_complex_cepstrum` (→ `SpectrogramResult`/`STFTResult`/`WignerVilleResult`/`CWTResult` with its `times`/`Cepstrum` with its `quefrencies`/`ComplexCepstrum` with its `.quefrencies`); `axis=` names the time axis of a multichannel record on the analytic-signal family, `cwt` and `cepstrum` |
| Constant-Q (Brown 1991) | `constant_q_transform`, `constant_q_spectrogram`, and `constant_q` / `probabilistic_constant_q` (→ `CQTResult`/`SpectralEstimate`/`CQSpectrogramResult`/`ProbabilisticSpectralEstimate`) |
| Gather transforms | `fk_transform`, `taup_transform`, `radon_transform`, `inverse_fk`, `inverse_taup`, `inverse_radon` (→ `FKResult`/`TauPResult`/`RadonResult`). `FKResult.scaling` says whether `power` is a calibrated density per Hz·rad/m (`scaling='density'`, the default) or the raw unnormalised `\|FK\|²` (`scaling='raw'`), and `plot_fk` labels the panel from it; `radon_transform` scans the curved (parabolic, hyperbolic) moveouts, the linear one being `taup_transform`'s; `axis=` names the time axis of a `(nx, nt)` gather; each inverse takes its forward result whole (`inverse_taup(r)`, a filtered panel as `r._replace(panel=…)`), whose geometry rides on it |
| System ID / channel | `frf_welch`, `etfe`, `periodic_etfe`, `lsfir` (→ `FRFResult`), `FRF` (one of them configured), `impulse_response`, `channel_response` (a `ChannelResponse`, `.plot()`), `impulse_response_from_transfer_function`, `transfer_function_from_impulse_response`, `synthesize_time_series` (a waveform through `H(f)` on its `1/Δf` record at exactly `sample_rate`, density convention; `t_start` required), `simulate_reception`, `apply_impulse_response` (a waveform through sampled taps, linear FFT overlap-add), `apply_impulse_response_blocks` (the same over a signal in blocks, `O(block + M)` memory), `fractional_delay_taps` |
| Channel statistics (arrays) | `arrival_transfer_function`, `received_amplitudes`, `remove_delay`, `spectral_centroid`, `synthesis_band`, `broadband_loss`, `truncate_response`, `rms_delay_spread`, `energy_support`, `coherence_bandwidth`, `COHERENCE_BANDWIDTH_FACTORS`, `channel_regime`, `ChannelRegime`, `uniform_frequency_step`, `tone_phasor`, `waveform_spectrum_at`, `simulate_arrival_reception`, `simulate_arrival_grid`, `arrival_grid_transfer_function`, `arrival_impulse_response`, `arrival_grid_impulse_response`, `delayandsum` |
| Modal acoustics (arrays) | `acoustics.modal_attenuation`, `acoustics.mode_shapes_at`, `acoustics.modal_field`, `acoustics.modal_phase_speeds`, `acoustics.modal_grazing_angles`, `acoustics.modal_excitation`, `acoustics.medium_density_at` |
| Ray geometry (arrays) | `acoustics.polyline_miss_distance` |
| Seafloor mask (arrays) | `acoustics.mask_below_seafloor` (`Field.mask_below_seafloor` wraps it) |
| Wavenumber domain (arrays) | `acoustics.hankel_transform`, `acoustics.wavenumber_taper`, `alias_period`, `ranges_fit_alias_period`, `wavenumber_sampling` (the window, the alias period and the verdict as a `WavenumberSampling` record, `.plot()`), `wavenumbers_from_phase_speeds`, `snapshot_frequency_component` (the methods of `GreensFunction` call these) |
| Boundary / waveguide scalars (arrays) | `acoustics.reflection_coeff`, `critical_angle`, `bottom_loss_curve`, `scholte_wave_speed` (the Scholte wave of a liquid on a solid, Brekhovskikh & Lysanov eq. 3.4.29; `nan` where the solid has no root), `pekeris_root` |
| Seawater & bubbles (arrays) | `acoustics.sound_speed_mackenzie` / `_delgrosso` / `_teos10` / `_unesco`, `density`, `doppler`, `bubble_sound_speed`, `bubble_resonance`, `bubble_surface_loss` |
| Model agreement | `metrics.tl_rmse` (same grid, `metrics.same_grid`), `metrics.tl_rmse_on_shared_ranges` (resampling), `metrics.tl_difference` (the difference map, a Field), `metrics.field_statistics` (per-field mean/std and the pairwise RMS table) |
| Levels (arrays) | `acoustics.pressure_from_recording` / `recording_from_pressure` (volts or counts ↔ Pa), `acoustics.spl`, `peak_level`, `sound_exposure_level`, `cumulative_sound_exposure_level` (a `CumulativeExposure`, `.plot()`), `track_exposure`, `power_to_dB`, `transmission_loss_dB`, `received_level_dB`, `sum_levels_dB`, `integrate_psd`, `band_level`, `no_energy_mask` |
| Frequency weighting (arrays) | `acoustics.weighted_pressure`, `frequency_weighting`, `auditory_weighting`; every SEL entry point takes `weighting=` (a hearing group, a `W(f)` function or a `(f, W_dB)` table) |
| PE grid quality (arrays; module `models.pe_grid`) | `models.optimize_grid`, `grid_error`, `optimal_c0`, `numerov_error`, `combined_error`, `rams_stable_dr`, `rams_stable_theta`, `rams_growth_margin`, `rams_dz_shear_cap`, `rotated_pade_coefficients`, `rotated_cn_growth`, `rotated_growth_floor`, `seabed_leak_rate`; results `GridChoice` (of `optimize_grid`) and `GrowthMargin` (of `rams_growth_margin`), named tuples (`GridInfeasibleError` when no grid meets `eps`) |
| Modal / dispersion | `warp_signal` (→ `Warped(times, data)`, carrying the forward call's settings), `unwarp_signal` (takes the `Warped` whole), `modal_group_speeds` |

Frequency-response estimation is four functions of one input/output record
pair, each returning an `FRFResult` — the `(frequencies, transfer_function)`
tuple, with the rest on attributes: `frf_welch` (H1/H2, with `.coherence`),
`etfe`, `periodic_etfe` and `lsfir`, a least-squares FIR fit with
`.impulse_response`, `.order`, `.criterion` and `.rcond`. An integer `order`
pins the FIR order; one of the criterion strings `'AIC'`, `'BIC'`, `'FPE'` or
`'CP'` (Akaike 1974, Schwarz 1978, Akaike 1969 and Mallows 1973, in their
published forms) searches instead, up to `max_order`.
`FRF(method=…, estimator=…, order=…, max_order=…, stop_count=…, …)` is the
configuration of one of them — every setting on the constructor, unset ones
derived at the run, and one the chosen method does not read refused — run by
`.compute(x, y, sample_rate)` on a record pair or averaged over a 2-D block of
measurements;
`method='periodic_etfe'` also takes `period=`, the excitation period in samples:

```python
from uacpy.acoustic_signal import lsfir

fit = lsfir(x, y, sample_rate=1_000.0, order='AIC', max_order=64)
fit.order                 # the FIR order AIC picked
fit.impulse_response      # its impulse response
```

Waveform builders come in two families. Those that take a `duration` (or a
cycle count) and a `sample_rate` build their own time axis and return
`(time, signal)`, with `round(duration × sample_rate)` samples: `lfm_chirp`,
`hfm_chirp`, `tone_burst`, `make_bandlimited_noise` and
`synthesize_noise_from_psd` (whose `sample_rate` defaults to twice the last
frequency of its spectrum). Those that take a time axis you built evaluate the
pulse on it and return the signal alone: `ricker_wavelet(times, …)`,
`gaussian_pulse(times, …)`, `nwave(times, …)`, and `sparc_pulse(times, frequency, pulse_type)`, which
returns `(signal, name)`, takes its characteristic frequency in Hz and forms
ω = 2πf itself. Build that axis as `np.arange(n) / sample_rate`,
the step the first family uses. `make_mseq_probe(freq_min, freq_max, *, sample_rate,
duration)` returns the signal alone. Build a chirp, push it through a delay, and
pulse-compress:

```python
import numpy as np
from uacpy.acoustic_signal import lfm_chirp, matched_filter

t, sig = lfm_chirp(freq_start=2_000, freq_end=6_000, duration=0.02, sample_rate=48_000)
rx = np.concatenate([np.zeros(500), sig, np.zeros(500)])   # echo at 500 samples
mf = matched_filter(rx, sig)            # peak marks the arrival
```

See examples 10 (signal tour), 28 (matched filter), 29 (arrays),
30 (time-frequency / f-k / tau-p).

## 11. Sonar Performance

`uacpy.sonar` — the sonar equation and detection theory, layered on TL fields
and noise spectra. Levels are in dB; the equation terms are the usual SL, TL, NL, DI,
TS, RL. The `*_field` helpers map the equation over a model TL
`Field` to produce signal-excess / detection-probability maps over
`(depth, range)`.

| Area | Public names |
|------|--------------|
| Sonar equation | `passive_signal_excess`, `active_signal_excess`, `echo_level`, `figure_of_merit`, `noise_background`, `SonarBudget` (one budget's terms: `.signal_excess`, `.figure_of_merit`, `.signal_excess_field`, `.to_dict` / `.from_dict`), `transition_probability`, `detection_ranges`, `detection_ranges_field`, `detection_annuli` |
| Field maps | `passive_signal_excess_field`, `active_signal_excess_field`, `transition_probability_field` (`.sonar_budget` is the `SonarBudget.to_dict()` of the map) |
| Detection theory | `albersheim_snr`, `probability_of_detection`, `roc_curve` (a `RocCurve`, `.plot()`), `detection_index`, `deflection_coefficient`, `detection_threshold_energy`, `per_look_false_alarm`, `scan_false_alarm` (a scan of N independent beams false-alarms at the scan's rate, not one beam's) |
| Target strength | `ts_sphere`, `ts_cylinder`, `ts_plate`, `ts_ellipsoid`, `ts_convex` |
| Scattering / reverb | `lambert_bottom`, `LAMBERT_MU_DB`, `chapman_harris_surface`, `column_scattering_strength`, `boundary_reverberation`, `volume_reverberation` (components add through `acoustics.sum_levels_dB`) |
| High-frequency boundary scattering (APL-UW TR 9407, 10–100 kHz) | `apl_uw_bottom_backscatter`, `apl_uw_bottom_loss`, `apl_uw_surface_backscatter`, `BottomParameters` (`.from_sediment` / `.from_grain_size` / `.from_geoacoustics` / `.from_bottom` / `.from_environment`), `APL_UW_SEDIMENTS`, `TABLE_WATER_SOUND_SPEED` |
| Interface roughness (coherent reflection) | `rayleigh_parameter`, `coherent_reflection_factor`, `perturbative_grazing_limit` |
| Matched-field localization | `synthesize_replica`, `replica_bank`, `replica_bank_from_field` (→ `Replicas`), `csdm`, `bartlett`, `mvdr` (→ an ambiguity `Field` in dB re max, whose `reference` is the peak power); `ambiguity_field` builds that Field from an array-level surface |

`rayleigh_parameter` is the RMS phase deviation a rough interface imposes,
`P = 2kσ·sin θ` with θ the **grazing** angle (JKPS Sect. 1.7, where it is `Γ`):
`P ≪ 1` is an acoustically smooth boundary, `P ≳ 1` one that scatters most of
what reaches it. `coherent_reflection_factor` is the `exp(−P²/2)` that a
smooth-interface reflection coefficient is multiplied by (JKPS eq. 1.79) — the
energy it removes has gone into the incoherent field, which is reverberation,
so it is a redistribution and not a loss. Because that factor is itself a
small-roughness result it decays far faster than reality once `P` passes 1;
`perturbative_grazing_limit` solves `P = 1` for θ and returns the steepest
grazing angle at which the treatment still has a claim (90° when `2kσ ≤ 1`,
i.e. perturbative everywhere). That last one is the form a full-field run
needs, since such a run has no single grazing angle but a spectrum of them —
it is what the OASES rough-interface warning reports.

`apl_uw_bottom_backscatter` and `apl_uw_surface_backscatter` are the handbook's
seabed (Kirchhoff + composite roughness + large roughness + sediment volume)
and sea-surface (bubble layer + Bragg ripples + facets) models for 10–100 kHz,
where Lambert and Chapman–Harris are out of band; `BottomParameters` carries the
six seabed inputs from a sediment name, a grain size or measured geoacoustics,
or from a uacpy seabed — `from_environment(env)` reads a fetched
`Environment`'s bottom (grain size first, geoacoustics against the water at
the seafloor otherwise), `from_bottom` a bare `Bottom` / `SeabedColumn` /
`BoundaryProperties` — and `APL_UW_SEDIMENTS` is the handbook's Table 2.
`TABLE_WATER_SOUND_SPEED` (1528.0 m/s) is the water speed the report's
Table 3 and Figure 2 were computed with; pass it as the water sound speed to
reproduce them.

`LAMBERT_MU_DB` (float, −27.0) is `lambert_bottom`'s default backscattering
constant 10·log10(μ) in dB — Mackenzie's (1961) deep-water value; pass
`mu_dB=` to `lambert_bottom` to pick another point in the empirical −25 to
−35 dB spread for unconsolidated sediments.

```python
from uacpy.sonar import figure_of_merit, albersheim_snr

dt  = albersheim_snr(pd=0.5, pf=1e-4)         # required SNR (dB) for Pd/Pfa
fom = figure_of_merit(source_level_dB=180, noise_level_dB=60,
                      directivity_index_dB=15, detection_threshold_dB=dt)
# fom is the max allowable one-way TL — cross it with a TL field for range.
```

**Matched-field processing (MFP).** Localize a source in range and depth by
correlating measured array data against *replicas* — the modeled pressure at
the sensors for each candidate position. Replicas are synthesized directly
from a KRAKEN `Modes` set (the far-field modal sum, validated against
`field.exe` to a normalized correlation of 1.0), so the modes are computed once
and every grid point is a cheap re-sum. This path is self-contained (KRAKEN +
numpy); it does not require OASES/OASN.

`replica_bank_from_field(field, array_depths=...)` is the model-agnostic
counterpart of `replica_bank`: it assembles the same `Replicas` bank from a
coherent pressure `Field` produced by *any* forward model — run with the
receivers at the array element depths and a multi-depth `Source` over the
candidate depths, whose field carries them on its `source_depth` axis — so PE
or ray replicas drop into the same `bartlett` / `mvdr` processors.

```python
import numpy as np
from uacpy.models import Kraken
from uacpy.sonar import synthesize_replica, replica_bank, csdm, bartlett, mvdr

modes = Kraken().compute_modes(env, source)     # eigenpairs (k_m, phi_m), once
bank  = replica_bank(modes, array_depths=array_depths, candidate_depths=cand_depths,
                     candidate_ranges=cand_ranges)  # Replicas on (depth, range)
K     = csdm(snapshots)                          # (N, L) snapshots -> (N, N) CSDM
amb_b = bartlett(K, bank)                        # robust, broad-lobed
amb_m = mvdr(K, bank, diagonal_loading=1e-2)     # Capon: sharp, mismatch-sensitive
best  = amb_m.max().pinned                       # {'depth': ..., 'range': ...}
```

`mvdr`'s `diagonal_loading` trades resolution for robustness: small values give sharp
Capon peaks, larger values flatten the surface toward Bartlett under
environmental mismatch — the dominant error source in MFP.

See example 27 (sonar equation, reverberation, detection-range maps) and
example 38 (matched-field localization with KRAKEN replicas).

## 12. Digital Communications

`uacpy.comms` — a coherent underwater link in two tiers: Tier 1 is the essential chain
(modulation, fading channels, DFE/LMS/RLS equalization with an optional
carrier PLL, Doppler estimation, framing, BER/EVM); Tier 2 adds OFDM,
convolutional+Viterbi FEC, DSSS, and NATO **JANUS** (STANAG 4748,
verified bit-exact against CMRE janus-c).

| Area | Public names |
|------|--------------|
| Modulation | `SCHEMES` (scheme name → `(family, order)`), `psk_qam_modulate`, `psk_qam_demodulate` (and `Modulator`, which wraps them for one scheme), `constellation`, `dpsk_modulate`, `dpsk_demodulate`, `fsk_modulate`, `fsk_demodulate` |
| Channel | `awgn`, `ebn0_to_snr_dB`, `snr_to_ebn0_dB`, `multipath_channel`, `pulse_shaped_taps`, `apply_channel`, `apply_fading_channel`, `fading_taps` |
| Equalization | `DFE`, `lms_equalizer`, `rls_equalizer`, `mmse_equalizer`, `slicer`, `complex_gain` (the least-squares one-tap gain against a known sequence) |
| Doppler / sync | `estimate_doppler_scale` (→ a `DopplerEstimate` record; `.plot()` draws its ambiguity curve), `compensate_doppler`, `doppler_from_speed`, `detect_preamble`, `detect_frames`, `matched_filter_metric` |
| Passband PHY | `rrc_filter`, `rrc_pulse`, `rc_pulse`, `pulse_shape`, `rrc_matched_filter`, `upconvert`, `downconvert`, `symbol_sync` |
| Framing | `pack_frame`, `unpack_frame`, `bytes_to_bits`, `bits_to_bytes` |
| Transceivers | `Transmitter`, `CommsReceiver`, `ReceiverDiagnostics` (what `receive(..., return_diagnostics=True)` returns on both receivers; the OFDM pair is in its own row) |
| Channel estimation | `ls_estimate`, `omp_estimate` |
| Link harness | `simulate_link`, `ber_sweep`, `ofdm_ber_sweep` (the OFDM pair's curve), `BerCurve` (what `ber_sweep` returns: `(ebn0_dB, ber)` with the sweep's settings, `.plot()`), `LinkResult`, `ChannelTaps` (from `Arrivals.channel_taps(symbol_rate, fc=…)`; hand it to the link harness as its `channel=`) |
| Metrics | `bit_error_rate`, `symbol_error_rate`, `evm`, `ber_theory`, `ber_theory_curve` (a `BerTheory`, `.plot(ax=)`) |
| Coding / spread | `ConvCode`, `conv_encode`, `viterbi_decode`, `viterbi_from_branch_metrics`, `interleave`, `deinterleave`, `m_sequence`, `spread`, `despread`, `spreading_gain_dB` |
| OFDM | `ofdm_modulate`, `ofdm_demodulate`, `ofdm_symbol`, `subcarrier_response`, `equalize_subcarriers`, `schmidl_cox_preamble`, `schmidl_cox_metric`, `schmidl_cox_sync`, `remove_cfo`, `estimate_channel`, `OFDMTransmitter`, `OFDMReceiver` |
| JANUS | `janus_encode`, `janus_decode`, `janus_modulate`, `janus_demodulate`, `janus_detect`, `janus_transmit`, `janus_receive`, `JanusPacket`, `JanusReception` (what `janus_demodulate` returns: `(bits, crc_ok)` with how the packet was found; `bits=None`, `crc_ok=False` when no complete frame was in the recording); `janus_receive` returns `(packet, reception)`; `sample_rate` is required on every JANUS function |

`simulate_link` composes transmit → channel → receive and measures BER;
`ber_theory` gives the AWGN bound for the same scheme:

```python
import numpy as np
from uacpy.comms import simulate_link, ber_theory

rng = np.random.default_rng(0xACED)
for ebn0 in (0, 4, 8, 12):
    res = simulate_link("qpsk", ebn0_dB=ebn0, n_bits=20_000, rng=rng)
    print(ebn0, res.ber, ber_theory("qpsk", ebn0))
```

`simulate_link`, `ber_sweep`, `ofdm_ber_sweep`, `awgn` and `fading_taps` each take an `rng` and
default to an **unseeded** `np.random.default_rng()`, so a call without one
returns a different BER every run. Pass a seeded generator for any number you
intend to quote or compare.

See examples 31 (comms tour), 32 (text→wav modem), 33 (OFDM), 34 (JANUS beacon).

## 13. Ambient Noise

`uacpy.noise` — Wenz-style spectra plus ship radiated-noise (ISO 17208) and marine-mammal
auditory weighting (Southall 2019). Spectra are in dB re 1 µPa²/Hz.

| Area | Public names |
|------|--------------|
| Wenz spectrum | `WenzNoise` (band levels: `.band_level(freq_min, freq_max)`, `.decidecade_levels(...)`), `NoiseComponents` (its per-mechanism levels), `wind_noise_level`; the submodel registries `WIND_MODELS`, `SHIPPING_MODELS`, `RAIN_MODELS`, `THERMAL_MODELS`, `TURBULENCE_MODELS` (read-only; `register_noise_model(kind, name, fn)` adds one) |
| Ship radiated noise (ISO 17208) | `radiated_noise_level`, `monopole_source_level`, `nominal_source_depth`, `lloyd_mirror_correction`, `RNL_UNCERTAINTY_DB`; without a measurement, `ship_source_level_wales_heitmeyer` (the Wales & Heitmeyer 2002 mean merchant-ship monopole spectrum, Ainslie Eq. 8.195: measured 30–1200 Hz, continued to 15 kHz with Ainslie's −23 dB/decade gradient (sect. 8.3.1.4), evaluated outside 30 Hz–15 kHz with a `ValidityWarning`; an ensemble mean with no speed term) |
| Mammal weighting (Southall 2019) | `auditory_weighting`, `apply_weighting`, `weighted_level`, `HEARING_GROUPS`, `WEIGHTING_PARAMS`, `EXPOSURE_THRESHOLDS` (Southall Tables 6-7 TTS/PTS onset; weighted SEL of a record or a map: `weighting=` on `sound_exposure_level`, see §8) |

The matching plotters `plot_source_level` and `plot_weighting` live on
`uacpy.plot` (§9), not on `uacpy.noise`.

```python
import numpy as np
from uacpy.noise import WenzNoise

f = np.logspace(0, 5, 500)
wenz = WenzNoise(f, wind_speed_kn=15, water_depth="deep",
                shipping_level="medium", rain_rate="moderate")
psd = wenz.as_psd()             # linear SI Pa²/Hz (levels stay dB re 1 µPa²/Hz); uacpy.plot.plot_wenz(wenz) for the dB spectrum
```

Hearing groups: `LF, HF, VHF, SI, PCW, OCW, PCA, OCA`. See examples 9, 35, 36.

## 14. Standards-Based Metrics

These standards-grounded helpers live in their natural packages rather than a
single module — each is sourced to its standard:

- **`uacpy.acoustics.sound_speed_unesco`** — seawater sound speed, UNESCO
  (Chen & Millero 1977 / UNESCO 1983), `c(T, S, *, depth= | pressure_dbar=)`.
- **`uacpy.acoustics.sound_speed_delgrosso`** — Del Grosso 1974 alternative,
  preferred at high pressure / deep water.
- **`uacpy.acoustics.sound_speed_teos10`** — TEOS-10 (IOC/SCOR/IAPSO 2010),
  Eqn. (2.17.1) on the IAPWS-08/09 Gibbs function; agrees with Del Grosso at depth.
- **`uacpy.acoustic_signal.decidecade_bands`** / `band_levels` —
  decidecade bands, ISO 18405 / IEC 61260-1. Base-ten (`1000·10^(n/10)`), which
  is what the standards specify, as are the base-ten octaves
  (`band_type='octave'`, `octave_bands`). `standard_bands` and `band_levels`
  give the same ladders by `band_type` name.
- **`uacpy.noise.monopole_source_level`** / `radiated_noise_level` — ship
  monopole source level, ISO 17208, from a measurement;
  `ship_source_level_wales_heitmeyer` — the mean merchant-ship monopole
  spectrum (Wales & Heitmeyer 2002), without one.
- **`uacpy.noise.auditory_weighting`** — marine-mammal frequency weighting,
  Southall et al. 2019.

`uacpy.metrics` itself holds the cross-model TL-agreement helpers
(`tl_rmse`, `tl_max_error`, `tl_bias`), each taking `(field, reference)` as
two `Field`s on one depth/range grid (2-D or a 1-D cut) or two plain dB
arrays, and `tl_rmse_on_shared_ranges(field, reference, depth=…)` for two
grids that share only some ranges (or two `(ranges, tl_dB)` curves, no depth);
`tl_difference(field, reference)` returns the difference map as a `Field`
(`.plot()`), `field_statistics(fields, depth=…)` the per-field statistics and
pairwise RMS table as a `FieldStatistics` (`.plot()`), and
`same_grid(field, reference)` the one same-grid rule they all apply.
`uacpy.analytic` supplies the exact fields to score an engine against:
`free_field`, `lloyd_mirror`, `ideal_waveguide(env, source, receiver)` and
`pekeris(env, source, receiver)`, with `pekeris_modes(env, frequency)` the
trapped modes themselves as a `Modes` (see the utilities guide, §2); example 49
scores Bellhop, Kraken and Scooter against them.

```python
from uacpy.acoustics import (
    sound_speed_unesco, sound_speed_delgrosso, sound_speed_teos10,
)

sound_speed_unesco(15, 35, depth=0.0)      # 1506.675 m/s  (T °C, S PSU, depth m)
sound_speed_delgrosso(15, 35, depth=0.0)   # 1506.678 m/s
sound_speed_teos10(15, 35, depth=0.0)      # 1506.674 m/s
```

Example 35 chains site sound speed → decidecade bands → ship SL → weighted level.

## 15. Units & Conventions

uacpy is SI throughout; underwater levels reference **1 µPa**. The exceptions
are laws published in other units, and they convert inside, behind an
argument that names its unit: wind speed is **knots** everywhere
(`WenzNoise(wind_speed_kn=)`, `sonar.chapman_harris_surface(wind_speed_kn=)`,
`generate_sea_surface(wind_speed_19_5m_kn=)`,
`sonar.apl_uw_surface_backscatter(wind_speed_kn=)`, and what
`data.fetch_wind` returns), the unit the Wenz and Chapman-Harris fits are
stated in; the fits stated in m/s convert inside, and a reading in m/s converts
with `uacpy.units.ms_to_knots`. The height differs: the noise and scattering
laws and `data.fetch_wind` use the 10 m wind (`wind_speed_kn`),
`generate_sea_surface` the Pierson-Moskowitz wind at 19.5 m
(`wind_speed_19_5m_kn`) — divide a 10 m reading by 0.94 first
(Ainslie 2010 Eq. 8.12; [noise guide §5](docs/guide/noise.md)). Arguments named `*_deg` are degrees,
`acoustics.reflection_coeff(grazing_deg, …)` included.

A name carries a unit suffix only when its unit is not SI, and then the
suffix is the unit: `_km` kilometres, `_deg` degrees, `_kn` knots, `_dbar`
decibars, `_dB` a level in decibels (`max_distance_km`, `grazing_deg`,
`angles_deg`, `wind_speed_kn`, `source_level_dB`). An SI unit goes unsaid: a
name with no suffix is in metres, seconds, Hz, m/s or radians (`ranges`,
`rmax`, `delays`, `bandwidth`, `speed`, `phases`). Six quantities keep plain
names in their stated units: density in g/cm³ and attenuation in dB per
wavelength (`density`, `water_density`, `attenuation`), latitude and longitude
in degrees (`lat`, `lon`, `latitude`), temperature in °C and salinity in PSU
(`temperature`, `salinity`). Constants follow the same rule
(`DEFAULT_WATER_DENSITY`, `EARTH_RADIUS`, `DEFAULT_RUN_TIMEOUT`, but
`EARTH_RADIUS_KM`). `TestNoPublicNameCarriesAnSIUnitSuffix` in
`tests/test_architecture.py` holds every public parameter, attribute and
constant to it.

One idea has one spelling: `fc` for a centre or carrier frequency
(`OASP(fc=)`, `Arrivals.channel_taps(fc=)`, `synthesis_band(fc=)`, the `fc`
metadata key), `centers` for band centres (`BandLevels.centers`),
`n_frequencies` for a count of frequencies (`cwt(n_frequencies=)`,
`Field.n_frequencies`), `rng` for a random generator (`awgn(rng=)`,
`schmidl_cox_preamble(rng=)`), `group_speeds` for modal group speeds
(`Modes.group_speeds`, `acoustic_signal.modal_group_speeds`) and `c_low` /
`c_high` for a phase-speed window (`Scooter(c_low=)`,
`GreensFunction.to_field(c_low=)`, `acoustics.wavenumber_taper`). `test_one_spelling_per_idea` in
`tests/test_harmonisation_gates.py` holds every public parameter, result field,
settings-record field and documented metadata key to them.

A sample rate goes where the samples are: a function handed samples takes
their rate positionally right after them (`welch(x, fs)`,
`janus_receive(waveform, fs)`, `upconvert(baseband, fs, fc)`), and a function
that creates samples takes `sample_rate=` keyword-only (`lfm_chirp`,
`bpsk_modulate`, `janus_modulate`, `fsk_modulate`,
`Transmitter.transmit_passband(bits, sample_rate=fs, fc=fc)`).

### Objects wrap functions

A computation that makes sense on data from anywhere else — another model, a
file, a measurement, an array you typed in — lives in a **function over plain
arrays**, and the carrier's method is a wrapper around it. The method
contributes what only an object can: axis and coordinate bookkeeping,
arguments read off its own identity (band, sample rate, geometry), units and
provenance, re-wrapping the result, and refusals only it can raise.

```python
# the same computation, two ways in
spread = arrivals.rms_delay_spread()                  # from a model result
spread = rms_delay_spread(tau_measured, power_measured)   # from anywhere else
```

This is a correctness rule, not a tidiness one. Two implementations of one
computation drift, and the drift is invisible to the tests you would naturally
write: two inverses of `h → H` with different normalisations differ by a factor
of `fs` **with the phase exact to 1e-16**, so every phase assertion passes, and
two arrival sums that differ only in taking `abs` of a negative amplitude
differ by up to 10.7 dB per bin. Each computation therefore has one
implementation, and every route to it calls that one.

Three things make delegation possible, and their absence is what forces a
method to re-implement:

* the function takes `axis=` (or is otherwise N-D), so a `(depth, range, …)`
  grid needs no loop;
* the function has a private labelled form, `_name(..., who=)`, which the
  method calls with its own name, so a refusal names the call the user made
  rather than an internal function they have never heard of — and the public
  signature carries physics only;
* one **stated** convention per transform pair, pinned by a test on the
  **ratio** — the only quantity a normalisation error moves.

Where two conventions legitimately coexist, both are written down. The
array-level pair `impulse_response_from_transfer_function` /
`transfer_function_from_impulse_response` is **unscaled** (`irfft` one way,
`rfft` the other); `Field.impulse_response` / `Field.to_transfer_function` use
the **density** convention (`taps·fs`, `rfft·dt`) because a model's `H` is a
density. Each pair round-trips to floating point; mixing halves is off by
`fs`. `acoustic_signal.synthesize_time_series` is the density route on plain
arrays — the core `Field.synthesize_time_series` runs. `Field.impulse_response`
stores the taps of `impulse_response_from_transfer_function` in the density
convention (`taps·fs`, 1/s, the unit of every impulse-response Field),
and `Field.apply` takes them back to taps (`h·dt`) before
`apply_impulse_response` convolves.

Every carrier is also exported and constructible from plain arrays, so a
method could in principle be reached by building the carrier; the function
form spares a caller the canonical coord order, the `kind`/`unit` tags and the
metadata keys the method reads, a coupling that arithmetic needs none of.

### Units

| Quantity | Unit | Note |
|----------|------|------|
| Length / depth / range | metres | no suffix; `_km` when kilometres (`m_to_km` at IO boundaries) |
| Frequency | Hz | |
| Time | seconds | |
| Angle | degrees | `_deg`: launch angles, grazing/incidence angles, wedge angles; phase is in radians, unsuffixed |
| Sound speed | m/s | |
| Density | g/cm³ | acoustic inputs (bottom/sediment, water) and the `uacpy.acoustics` formulas alike — see *Sound speed & density* below |
| Attenuation (geoacoustic) | dB per wavelength | models emit the matching `AT` TopOpt letter |
| Attenuation (volume) | dB/km | `FrancoisGarrison(...).table(f)`, `Thorp().table(f)` |
| Pressure | Pa (µPa for levels) | a recording enters through `uacpy.acoustics.pressure_from_recording`, which turns volts (or ADC counts) into Pa given the hydrophone sensitivity in dB re 1 V/µPa; `uacpy.acoustics.spl` takes the waveform from there to dB re 1 µPa |
| Pressure level / SPL | dB re 1 µPa | air would be dB re 20 µPa |
| Noise spectral level | dB re 1 µPa²/Hz | |

### Geometry & coordinate system

- **Depth `z` is positive down** from the sea surface (`z = 0`); **altimetry is
  positive up** (surface elevation above the mean).
- Range `r` is horizontal distance **from the source** (`r = 0` at the source);
  range-dependent carriers (`Bathymetry`/`ssp`/`Bottom`) index their `ranges`
  axis from the source too.
- The field is **range–depth (2-D / N×2-D)**: azimuthal symmetry is assumed, so
  a point source spreads **cylindrically** (`∝ 1/√r`) in a waveguide.

### Fields, pressure normalization & transmission loss

- A `Field` with **complex** `data` holds the **acoustic pressure normalized to
  a unit point source at 1 m** — the transmission-loss pressure `P = p/p₀` of
  Jensen et al. §2.3.5, `p₀` the free-field pressure at 1 m from the same
  source. A free point source is therefore stored as `e^{-i k₀ r}/r` (|p| = 1
  at 1 m; the free-space Green's function `e^{-i k₀ r}/(4π r)` times 4π), in
  the `e^{+iωt}` form the Acoustics-Toolbox
  binaries use (`KrakenField/EvaluateMod.f90:42`: "use e{ i( w t - k r ) }
  form") and `PhaseReference.TRAVELLING_WAVE` records. Hence transmission loss is simply
  **`TL = −20·log₁₀|p|`** (`Field.dB`), in dB re 1 m. Real `data` is already in
  dB and returned as-is.
- This is the 3-D point-source (**spherical-spreading**) convention, and it is
  **the same for every model** (Bellhop, Kraken, Scooter, RAM, SPARC) so that
  TL is directly comparable across them. Each native binary's own normalisation
  is bridged to it in the **model** layer, after the raw reader: Kraken's
  `field.exe` sign (−1, and `e^{-iπ/4}·√k0` for a line source) and `1/ρ(z_s)`,
  Bellhop's sign and line-source factor, Scooter's line-source `√k0`. So a raw
  reader (`uacpy.io.read_shd_file`, `GreensFunction.to_field`) returns the engine's own
  normalisation. SPARC's `'D'` and `'S'` outputs are harmonised onto its `'D'`
  normalisation — the branch that carries the true inverse-Hankel weight
  `Δk·√(2k/(π·r))`. `'R'` is that weight without the `1/√π` (~4.97 dB hot) and
  is divided back in the SPARC model; `'S'` is scaled by `−2` inside
  `GreensFunction.snapshot_to_time_field`, so all three carry the
  unit-source level (measured against the rigid-guide image sum,
  `docs/models/validation.md` §1.3).
- A line source's 1 m reference is applied **per frequency** on BROADBAND and
  TIME_SERIES too: each bin of its `H(f)` is relative to its own free-field
  level at 1 m, so a synthesised pulse is the one whose 1 m free-field spectrum
  is the `source_waveform`'s. JKPS §5.2.2 warns that this "scale factor is
  frequency dependent and if it is not removed during the frequency synthesis,
  the pulse will be distorted": against a pulse built from an un-normalised
  line field, a uacpy line-source pulse is tilted by `√k0`, 3.01 dB per octave.
  Bellhop's line-source factor does not depend on frequency, so all three
  engines share this convention.
- A 2-D (line-source) analytic solution differs by the spreading factor
  `|G₃D/G₂D| = √(k/2πr)` — relevant only when comparing against closed-form 2-D
  references (see the ideal-wedge benchmark in `tests/test_benchmarks_analytic.py`).
- The spectral estimators and `spectrogram` are pure functions returning arrays;
  their plots (`plot_band_levels`, `plot_psd`, `plot_ppsd`, `plot_spectrogram`, and the
  gather/transform/comms plotters) are reached through `uacpy.plot`, not
  through the computation modules. Levels are formed through
  `acoustics.power_to_dB` (`10·log₁₀(power/ref²)`), which floors `power/ref²` at
  `PRESSURE_FLOOR²` (1e-60; `PRESSURE_FLOOR` is in `uacpy.core.constants`) before the log so a silent sample
  yields −600 dB for every `ref`, the level view of the no-energy marker `no_energy_mask` finds, rather than
  `−inf`. The band integrators (`band_levels`,
  `acoustics.band_level`, `noise.weighted_level`, `WenzNoise.band_level`) do
  not floor: a band that carries no power is `−inf` there.

### Phase / time convention

- Harmonic time dependence and the sign of the imaginary part **follow the
  underlying Acoustics-Toolbox binaries** (KRAKEN, e.g., is implemented with the
  `e^{+iωt}` convention). Magnitudes and TL are convention-independent; **phase**
  is not — this matters for matched-field processing and for broadband IFFT,
  where the `phase_reference` carried on a `Result` fixes the reference (see §8,
  *Phase reference — why IFFT needs it*).
- Broadband bands are defined per model: RAM uses half-bandwidth `fc/q_factor` (the sweep spans `fc ± fc/q_factor`), `Δf = 1/record_duration`;
  the other engines expand a lone `fc` to 128 bins over `0.5·fc` — a full width, so the same band is
  `Q = 2/0.5 = 4`. Passing the band as a `frequencies=` array — or as a multi-element
  `Source.frequencies` — means the same band on every engine. OASP computes it on its own
  `1/(DT·NX)` bin ladder spanning that band, so its bins need not be the ones listed (it warns
  when they differ).

### Boundaries & angles

- **Surface** is pressure-release by default; **bottom** `acoustic_type` ∈
  `{'half-space', 'vacuum' (pressure-release), 'rigid', 'file', …}`.
- **Geoacoustic attenuation is dB per wavelength** (the AT default); the writer
  sets the corresponding TopOpt letter. Volume absorption is dB/km.
- Reflection coefficients (`Bounce`, OASR) are tabulated over **grazing angle**
  (degrees) by default; `OASR(angle_type='incidence')` converts via
  `grazing = 90 − incidence`. `ReflectionCoefficient` carries `|R|` and phase
  (radians). The critical grazing angle for `c₂>c₁` is `cos θ_c = c₁/c₂`.

### Sound speed & density

- **One reference water.** Every default that means "sea water" describes one
  state, exported at the top level: `REFERENCE_TEMPERATURE` = 10 °C,
  `REFERENCE_SALINITY` = 35, `REFERENCE_DEPTH` = 0 (the surface),
  `REFERENCE_PH` = 8.0. Its density is `DEFAULT_WATER_DENSITY` =
  1.027 g/cm³ (IES-80), the one water density of the package — what every
  deck writes, what every standalone default reads (`reflection_coeff`,
  `modal_field`, `bubble_resonance`, …) and what a published seabed density
  *ratio* (Jensen et al. Table 1.3 presets, Hamilton, APL-UW) is multiplied by,
  so a default run sees exactly the published ratio. The seawater formulas
  (`sound_speed_*`, `density`) default to the reference state. The water
  **sound speed** default is instead the field's nominal
  `DEFAULT_SOUND_SPEED` = 1500 m/s (the reference state itself gives
  1489.8 m/s), read by every function that needs "a" water sound speed,
  `doppler` included. For a textbook ρ_w = 1 benchmark pass
  `Environment(water_density=1.0)` **and** give seabed densities as the
  published ratios.
- Sound-speed helpers: **`sound_speed_teos10`** (T °C, S PSU, `depth=` in m
  converted to pressure at `latitude=`, or `pressure_dbar=`) is the default `formula` of `SoundSpeedProfile.from_temperature_salinity`
  and of the data routes; **`sound_speed_mackenzie` = Mackenzie (1981)** (T °C,
  S ppt, depth m) is `formula='mackenzie'`, with **`sound_speed_unesco` = Chen &
  Millero / UNESCO** and `sound_speed_delgrosso` as the other alternatives.
  A cast held in depth reaches any of the four through
  **`sound_speed_at_depth(temperature, salinity, depth, formula=, latitude=)`**,
  which converts depth to pressure at the cast's latitude for the three
  pressure equations; the conversion is public as
  `depth_to_pressure_dbar(depth, latitude)` and its inverse
  `pressure_dbar_to_depth(pressure_dbar, latitude)` (Leroy & Parthiot 1998
  standard ocean), and `insitu_from_potential(*, salinity, theta, pressure_dbar)`
  turns an ocean model's potential temperature into the in-situ temperature the
  equations take (UNESCO 44 adiabatic lapse rate).
- **Density is g/cm³ in both of its roles**:
  - *Acoustic input* density (bottom/sediment, and the water column on disk) is
    **g/cm³**. Every deck writes `env.water_density` (default
    `DEFAULT_WATER_DENSITY` = **1.027 g/cm³**) for the water column: the
    AT and OASES writers put it in the water rows, and the engines that fix
    the water at 1 (the RAM family and BOUNCE) receive each seabed density
    divided by it — so what every propagation model sees is the same
    impedance contrast. Pass `Environment(water_density=1.0)` to reproduce a
    textbook ρ_w = 1 benchmark.
  - *Formula-level* density: `density()` (the oceanographic equation of
    state) returns g/cm³ and `bubble_resonance` takes `water_density` in
    g/cm³, as every density argument does. `bubble_surface_loss`
    takes `grazing_deg` like the scattering functions. `reflection_coeff` and
    `bottom_loss_curve` take the geoacoustic units — grazing degrees, g/cm³,
    dB/λ — and default their water to the same 1.027 g/cm³ and 1500 m/s the
    decks write, so their curves match the engines. A value moves between
    the two roles as it is: `Environment(water_density=density(T, S))`.

### Modal & numerical conventions

- Kraken modes are depth-normalized so `∫ Z_m²/ρ dz = 1`; horizontal
  wavenumbers `k_m` are returned in descending order (mode 1 first), real for
  trapped modes with `k₂ < k_m < k₁`.
- SSP *shape* (the profile data) lives on `Environment`; the SSP *interpolation
  scheme* is a model knob (`Model(interp_ssp=...)`).
- Out-of-range positional indexing (`.isel`) raises `IndexError`; invalid
  configuration raises the typed `core.exceptions` hierarchy.
- Status logging goes through `uacpy._log.log_message` (never `print`);
  user-facing notices go through `warnings.warn` (a custom formatter is
  installed at import).

> These conventions are exercised by the **benchmark suite**
> (`pytest -m benchmark`), every test of which compares uacpy output against a
> published or closed-form reference: the TL/pressure normalization against the
> analytic Pekeris modal-sum and ideal-wedge solutions, the reflection-angle
> convention against the Rayleigh coefficient, and the sound-speed equations
> against their published check values.

### Reproducibility

What comes back byte-identical on a repeat run, and what you pin to get it.
Every row is measured on one machine, one build, one NumPy.

| Layer | Repeat run | Pin |
|-------|------------|-----|
| Analysis entry points (`acoustic_signal`, `sonar`, `comms`, `noise`, `metrics`) | byte-identical | — (58 entry points; holds across processes, `PYTHONHASHSEED`, locales, timezones, and 1,014 concurrent calls in an 8-thread pool) |
| The functions that draw: `make_bandlimited_noise`, `synthesize_noise_from_psd`, `awgn`, `fading_taps`, `simulate_link`, `ber_sweep`, `ofdm_ber_sweep`, `generate_sea_surface`, `data.fetch_sea_surface`, and `data.fetch_environment` when it builds a sea surface | byte-identical per seed; a fresh draw when omitted | `rng=np.random.default_rng(seed)` (`altimetry_rng=` on `fetch_environment`) |
| `Kraken` (`kraken`, `krakenc`), `Scooter`, `SPARC`, `Bounce`, `RAM` (all four backends), `OAST`/`OASP`/`OASR`/`OASN`/`OASS` | numeric output byte-identical | `work_dir=` |
| `OASSP` | byte-identical at a fixed realization | `realization=` — the index *is* the seed; different values are different draws by design |
| `Bellhop()`, default backend | **not** bit-reproducible — the auto-picked multithreaded backend accumulates beams in completion order (~1 ULP, ≤ 1.53e-05 dB) | `Bellhop(backend='fortran')`, whose `.shd` is byte-identical |
| Beamforming, MVDR, MUSIC, and anything else ending in a large matrix product | last bits follow the BLAS thread count (1e-16 to 1e-13 relative); every discrete output tested — peak bearings, and the JANUS detection index across 750 real detections — is unchanged | `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, set before NumPy is imported |

- **No uacpy function draws from a global random state** — neither Python's
  `random` nor NumPy's legacy `RandomState`. Randomness enters only through an
  explicit `rng`. Every set-to-output path is `sorted()`, so no result depends
  on `PYTHONHASHSEED`. Both are gated by source scans in
  `uacpy/tests/test_determinism.py`.
- **Written decks are byte-identical** across 15 configurations (processes,
  hash seeds, locales, timezones, work-dir paths), so a deck diff means a real
  input difference. Calling one model twice in one process into one pinned
  `work_dir` also returns byte-identical results.
- **The `.prt` listing is the exception, and only the listing.** Exactly four
  binaries stamp `CPU Time = …` into it — Kraken, Scooter, SPARC and Bellhop's
  fortran backend — so for those that one file is not byte-identical; **no
  numeric output depends on it**. Everything else matches throughout: Bounce is
  an Acoustics Toolbox model but writes no stamp, so its `.prt` is identical
  alongside its `.brc` and `.irc`, and RAM and the six OASES models write no
  such file at all. Hash the payload, not the directory.
- **Not measured:** `run_parallel` across worker counts; environments richer
  than the small test scenarios (range-dependent bathymetry, layered seabeds,
  broadband sweeps); cross-machine and cross-NumPy-version reproducibility.

Full detail, with a checklist for a reproducible run:
[reproducibility guide](docs/guide/reproducibility.md). The two moving parts
have their own measurements —
[Bellhop backends](docs/models/bellhop.md#7-gotchas) and
[BLAS threading](docs/guide/arrays.md#10-gotchas).

## 16. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `binary not found` / model won't run | native binaries aren't built — run `./install.sh -y` (lands in `uacpy/bin/`, gitignored); pick a backend with `--bellhop fortran\|cxx\|cuda`. `python -m uacpy` lists which engine binaries were found and where. |
| `UnsupportedFeatureError` | the model can't honour that `RunMode` or env axis. Check `model.supports_mode(...)`; unsupported env *shapes* are reduced by the collapse step (one `FallbackWarning` per dropped feature) — override the policy with `Model(collapse={...})`. |
| TL is `NaN` in places | `NaN` marks no-data cells. Every engine NaNs the `r ≤ 0` columns of a point-source field (the `1/√r` spreading is singular there; Bellhop never evaluates `r = 0` for a line source either, except with `beam_type='S'`). A Bellhop cell no beam reached is not `NaN`: the engine's empty beam sum is an exact zero, reported at the 600 dB no-energy marker. Below the seafloor it is per-engine: Bellhop NaNs receivers below the local seafloor; RAM returns its PE field down to the top of its artificial absorbing layer and NaNs the samples inside it (Collins & Siegmann §1.2.5), with a warning counting them; Scooter and SPARC compute through the sediment layers and NaN only below the deepest modelled interface; Kraken and the OASES models return the physical transmitted / evanescent field (Kraken NaNs only receivers in an *elastic* sub-bottom, which `field.exe` cannot evaluate). A solver that **wrote no number** there also NaNs: RAM marks the samples its march wrote as NaN/inf (and every receiver cell that interpolates one; a finite sample is the march's own value, however far below 0 dB), `Modes.with_attenuation` marks modes whose perturbation has no answer, and a trace synthesised from a spectrum with unsolved bins is NaN rather than carrying a notch (the bins where a normal-mode model found no mode, Kraken's `no_mode_bins`, are the exception: the modal sum there is empty, and every synthesis takes them as zero) — uacpy never writes a level over a sample the model did not produce, and a cell that genuinely carries no energy reports the 600 dB `PRESSURE_FLOOR` instead. The solver cases are announced by a warning naming the cause: a `NumericsWarning` for non-finite PE samples or unsolved bins, a `ValidityWarning` for a mode whose attenuation has no answer. Reductions and `uacpy.metrics` exclude NaNs via `np.isfinite`; plots leave them blank. |
| OASES tests skipped / `requires_oases` | OASES is academic-licensed and not bundled; fetch it via `install.sh --oases yes`. Run only the rest with `pytest -m "not requires_oases and not requires_network"`. |
| `Bellhop()` warns that `bellhopcuda` found no CUDA device | the auto-pick found the CUDA build but no usable GPU (none installed, `CUDA_VISIBLE_DEVICES=""`, a container without `--gpus`), so the run moved to the next installed binary for the rest of the process. Pass `Bellhop(backend='cxx')` to choose it directly; `result.backend` names the engine that ran. An explicit `backend='cuda'` raises instead. |
| A solver keeps running after its Python process was killed | binaries run in their own session so a timeout can reap them; on Linux with util-linux `setpriv` (2.33+) they are also ended when the launching Python process dies. Without it — macOS, busybox-based images, older util-linux — killing Python outright (a scheduler's walltime, a notebook kernel restart, an outer `timeout`) leaves the binary running. Pass the model's `timeout=` so uacpy reaps it itself, and stop a leftover by its PID (`ps -eo pid,etime,args \| grep <binary>`, then `kill <pid>`). |
| Wrong Python / missing deps | activate the project venv (`source uacpy_venv/bin/activate`), not system Python; `pip install -e ".[dev]"`. |
| `numpy.ndarray size changed, may indicate binary incompatibility` on a NetCDF load | a compiled dependency the NetCDF route imports (`netCDF4`, `cftime`) was built against an older numpy than the one installed; the message is that module's, the load completes and the arrays come back bit-identical. numpy registers an ignore filter for this exact message when it is imported, so it shows only after the warning filters are reset (`python -W default`, `warnings.simplefilter('always')`). Escalated to an error (`warnings.simplefilter('error', RuntimeWarning)` after numpy is imported, or a test suite's `filterwarnings = error`), a cached-grid read raises it as it is: the cached file is not at fault, and no live service is asked in its place. Reinstall those wheels against the installed numpy to remove it: `pip install --force-reinstall --no-cache-dir netCDF4 cftime`. |
| Two runs give slightly different numbers | expected in three places only, all bounded: the default `Bellhop()` backend (~1 ULP), the BLAS thread count (1e-16 to 1e-13 relative), and a `.prt` CPU-time stamp that no numeric output reads. Everything else is byte-identical — see [§15 Reproducibility](#reproducibility) for what to pin. |

## 17. Examples Index

All 50 runnable scripts live in `uacpy/examples/`. Run them **by script
path** from the repo root — `python uacpy/examples/example_01_basic_shallow_water.py`
— the form `run_all_examples.py` and the test suite use. The module form
(`python -m uacpy.examples.example_01_…`) also works, from a source checkout;
the examples are not part of the wheel.

Each script is a linear read: no `main()` and no cross-file helpers, and the
uacpy calls it demonstrates are named in its own docstring.

| # | Topic |
|---|-------|
| 01 | Basic shallow-water propagation — Pekeris waveguide |
| 02 | Sound-speed profiles — Munk, Pekeris, thermocline |
| 03 | Five models on one thermocline environment at a single reference frequency (needs OASES, `./install.sh --oases yes`; runs the other models without it) |
| 04 | Bellhop advanced — all-features showcase, a multi-depth `Field` over `source_depth` summed as a phased array with `superpose(weights=)` |
| 05 | RAM (mpiramS) — range-dependent bottom and bathymetry |
| 06 | Kraken — adiabatic modes over a continental shelf; an antiphase source pair (`Source(weights=[1, -1])` → `Field.superpose()`) |
| 07 | All models — comprehensive comparison (needs OASES, `./install.sh --oases yes`; runs the other models without it) |
| 08 | Deep-water SOFAR channel — long-range propagation in the Munk channel (needs OASES, `./install.sh --oases yes`; runs the other models without it) |
| 09 | Ambient noise (Wenz) + PSD→time-series synthesis + PPSD check |
| 10 | Signal-processing tour |
| 11 | Bellhop run modes — comprehensive |
| 12 | Attenuation models comparison |
| 13 | OASES suite — comprehensive |
| 14 | Plotting features — stacked time series, mode heatmaps |
| 15 | Elastic boundaries — both workflows compared |
| 16 | Bellhop + BOUNCE integration, layered / range-dependent bottom |
| 17 | Boundary conditions — top BC and layered bottoms |
| 18 | Range-dependent bottom — adiabatic vs coupled modes vs RAM |
| 19 | Broadband — time series and transfer functions across models (needs OASES, `./install.sh --oases yes`; runs the other models without it) |
| 20 | RAM multi-backend dispatch — mpiramS, RAMS (elastic), RAMSurf (rough) |
| 21 | Bellhop vs RAM(ramsurf) on identical altimetry env |
| 22 | RAM Padé-error grid optimizer (Lytaev 2023) |
| 23 | Per-feature collapse-method API |
| 24 | Synthesize a time series from H(f) |
| 25 | Canonical SSP shapes and bottom-loss curves |
| 26 | Animated wave propagation — wave-equation solvers vs a ray solver |
| 27 | Sonar equation, reverberation, detection range |
| 28 | Matched filtering, pulse compression, ambiguity function |
| 29 | Adaptive & high-resolution array processing |
| 30 | Time-frequency, wavenumber & slowness transforms tour |
| 31 | Underwater acoustic communications tour, incl. a modelled channel (`Arrivals.channel_taps` → `simulate_link`) |
| 32 | Real-data underwater modem (text → .wav → text) |
| 33 | OFDM underwater modem (text → .wav → text) |
| 34 | JANUS standard beacon (NATO STANAG 4748) |
| 35 | Underwater noise impact assessment — standards chain |
| 36 | Modeled noise impact — ship SL through a real TL field |
| 37 | Real-world environment — map · transmission loss · section |
| 38 | Matched-field source localization — KRAKEN replicas, Bartlett vs MVDR |
| 39 | OASS reverberation from a rough seabed — the mean-field → scattered-field chain |
| 40 | A vertical source array — slabs · coherent vs incoherent total · received level · array beam pattern vs mode filter |
| 41 | Beamforming a modelled field on a vertical receive array — the trapped-mode fan · shading · the λ/2 spacing limit |
| 42 | Propagation → array processing → detection, end to end — sources on the array elements, the realised array gain as a grid in the sonar equation, detection maps against the constant plane-wave gain |
| 43 | Which end of the path carries the sources — reciprocity on a range-dependent section, the cost of the mirror geometry against the solver's own residual, and `Source(beam_pattern=)` as per-element receive directivity |
| 44 | Two propagation models through one detection chain — maps that agree where headline ranges do not, and the beam's broadband reception of a tone burst |
| 45 | The level a signal with bandwidth or duration reaches, as a map — broadband propagation loss against the continuous-wave answer and against the incoherent stand-in, the sound exposure level and peak sound pressure level of one burst at a stated source level, and a received chirp |
| 46 | The same acoustics on data that never came from a model — a measured power delay profile, a recorded waveform and a hand-written mode set, through the array-level functions the carriers wrap |
| 47 | The exposure a moving source delivers — single-shot SEL maps unweighted and weighted for two hearing groups, the cumulative SEL of a survey line against distance with the distance to TTS onset, the plan view, and one receiver's record accumulating to the same number |
| 48 | Model files in, labelled data out — Bellhop's `.shd` and `.arr` read back with `uacpy.io.read_shd_file` / `read_arr_file`, and the results exported with `to_netcdf` / `from_netcdf`, `to_xarray` and `to_dataframe` (the export calls need `pip install "uacpy[xarray]"`) |
| 49 | Three engines against closed-form fields — Bellhop against `analytic.lloyd_mirror` (hat and Gaussian beams), Kraken and Scooter against `analytic.ideal_waveguide`, the agreement measured with `uacpy.metrics` |
| 50 | A frequency sweep in parallel, with the PE grid each frequency needs — `models.optimize_grid` before the runs, `run_parallel` with one `Job` per frequency, and `grid_error` on the grid RAM marched |

## 18. Parameter Reference

Every constructor knob with **unit · default · meaning**, grounded in the
constructor docstrings (the authoritative source). Configuration is
constructor-only (§4); to sweep a value, build one instance per setting or use
`model.copy(**overrides)`. Unit **`—`** marks a non-dimensional knob (enum code,
flag, count, path, factor); enum choices are spelled out under *Meaning*. `λ` =
wavelength.

### Common plumbing (every model)

These seven appear on **all** models with identical meaning and are omitted from
the per-model tables below.

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `executable` | path | `None` | Explicit binary path; `None` auto-detects the binary `install.sh` built. |
| `use_tmpfs` | — | `False` | Run scratch files in `/dev/shm` (tmpfs) instead of `$TMPDIR`. |
| `verbose` | — | `False` | Status-logging gate: `False`/`'silent'` → warnings+errors only; `True`/`'info'`; `'debug'`. |
| `work_dir` | path | `None` | Fixed scratch directory; `None` → a fresh temp dir per run. |
| `cleanup` | — | `None` | Delete scratch after the run; `None` → `True` unless `work_dir` is pinned. |
| `timeout` | s | `600.0` | Per-subprocess wall-clock limit (every model, SPARC included), positive and finite, or `None` for no limit; uacpy guidance: a guard on the host, no engine reads a time limit. |
| `collapse` | dict | `None` | Per-feature range/layer collapse-policy overrides (see §7, "Environment feature support and collapse"). |

### `run()` call arguments (every model)

Passed at call time, not construction — the fixed no-`**kwargs` signature (§4):

| Argument | Unit | Default | Meaning |
|---|---|---|---|
| `run_mode` | `RunMode` | `None` | Which product to compute; `None` → the model's default mode. |
| `frequencies` | Hz | `None` | Explicit frequency vector for `BROADBAND` / `TIME_SERIES`, overriding `source.frequencies` for the call, and refused like a `Source`'s own frequencies (positive, finite, each value once). On a single-frequency run mode it raises `ConfigurationError`, since the frequency comes from the `Source`; OASP is the exception, its solver always runs a frequency sweep and the keyword pins it. Kraken defaults `run_mode=None` to `BROADBAND` when the vector is multi-element; RAM never switches run modes on it. |
| `source_waveform` | array | `None` | Source pressure pulse for `TIME_SERIES` synthesis. |
| `sample_rate` | Hz | `None` | Sample rate of `source_waveform`, and of the returned trace on every engine: Bellhop places each arrival on its samples, and Kraken, Scooter, RAM and OASP sum their band at `t_start + n/sample_rate` exactly (`sample_rate/Δf` samples a record, the next whole number above it when that is not whole). |
| `output_duration` | s | `None` | Time-series record length, overriding the auto/constructor value. |
| `t_start` | s | `None` | Time of the record's first sample, after emission, on every engine that can place its record (Bellhop, Kraken, Scooter, RAM, OASP, OASSP; SPARC's always starts at emission and warns). `None` opens it just before the estimated earliest arrival (a tenth of the record or `4/bandwidth`, whichever is longer, on the IFFT engines; a tenth of the pulse on Bellhop); a value replaces that placement. |

### Source / Receiver

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `Source.depths` | m | *required* | Source depth(s), positive down. |
| `Source.frequencies` | Hz | *required* | Source frequency or frequencies. |
| `Source.source_type` | — | `'point'` | `'point'` (cylindrical spreading), `'line'` (Cartesian), `'scaled'` (point with cylindrical spreading removed). Support: Bellhop `point/line`; Kraken, Scooter, Bounce and `SPARC(output_mode='S')` all three; RAM and OASES `point` only, except OASR (reflection only, any source type — nothing in its deck reads the source). |
| `Source.beam_pattern` | deg / dB | `None` | Source directivity: `(N, 2)` `[angle_deg, level_dB]` array with strictly increasing angles, the `BeamPatternTable` `read_source_beam_pattern` returns, or a `.sbp` path. `None` = omnidirectional. Read by Bellhop and Kraken. |
| `Receiver.depths` | m | *required* | Receiver depth(s), positive down. |
| `Receiver.ranges` | m | `None` | Receiver range(s); `None` → a single point at 0 m. |

Interface roughness lives on the boundary carriers, not on the models:
`env.surface.roughness` is the sea-surface RMS roughness (AT `sigma(1)`, the
water column's mesh line) and `env.bottom`'s `roughness` is the seabed
(`sigma(NMedia+1)`, the bottom half-space line). Kraken and the OASES models
consume both; Scooter consumes the sea surface only, and only when that surface
is pressure-release (vacuum). Every other model drops what it cannot carry,
with a warning naming the value it dropped.

*(Environment and its carriers — bathymetry, SSP, bottom, surface, altimetry — are documented in §5; their units follow §15: metres / m/s / g/cm³ / dB-per-λ.)*

### Bellhop parameters

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `backend` | — | `None` | Binary variant: `'fortran'`/`'cxx'`/`'cuda'`; `None` auto-selects CUDA > C++ > Fortran (uacpy guidance: fastest first). |
| `dimensionality` | — | `'2D'` | Only `'2D'` is supported; `'3D'` raises. |
| `beam_type` | — | `'G'` | `G` geometric hat (Cartesian; BELLHOP's own default, `ReadEnvironmentBell.f90:393-395`), `B` geometric Gaussian (Cartesian), `g` geometric hat (ray-centred), `S` simple Gaussian, `C`/`R` Červený in Cartesian / ray-centred coordinates. `'b'` raises: `bellhop.f90:403` `ERROUT`s on it and the C++/CUDA ports silently substitute the Cartesian beam. |
| `n_beams` | count | `0` | Number of beams; `0` defers to Bellhop's auto-selection, as `doc/bellhop.htm` advises (`angleMod.f90:38-51`). |
| `single_beam` | index | `None` | Trace and sum only this beam of the fan (1-based): `TopOpt(6:6) = 'I'` (`angleMod.f90:32-33`; Bellhop User Guide §6.2). `None` sums every beam; refused past a pinned `n_beams`. |
| `launch_angles_deg` | deg | `(-80, 80)` | Launch-angle limits `(min, max)`; uacpy guidance (the manual names no fan). |
| `ray_step` | m | `None` | Ray step size, written as given; `None` = `env.depth / 50` (uacpy guidance), or `2 × r_box / MaxN` when coarser (the factor 2 is uacpy guidance), so a ray crosses the box within half the selected build's per-ray step limit (100 000 on bellhopcuda / cxx, 1 000 000 on Fortran); `0` asks BELLHOP for its own `depth/10` (`bellhop.f90:170-174`). |
| `z_box` | m | `None` | Max depth of the ray box; `None` = 1.2 × max depth (uacpy guidance, measured; the manual's "say 1%" is the documented alternative). BELLHOP drops a ray leaving the box (`bellhop.f90:571-572`); a pinned box short of the seafloor runs with a `ValidityWarning` (not on `RAYS`). |
| `r_box` | m | `None` | Max range of the ray box; `None` = 1.2 × max range, uacpy guidance, measured (room for paths that turn back from beyond the farthest receiver; [Bellhop](docs/models/bellhop.md)). A pinned box short of the farthest receiver runs with a `ValidityWarning` (not on `RAYS`). |
| `grid_type` | — | `'R'` | Receiver grid: `'R'` rectilinear (BELLHOP's default, `doc/bellhop.htm`), `'I'` irregular (sorted depth *i* pairs with sorted range *i*: a monotone diagonal, never arbitrary points — see `docs/models/bellhop.md`). |
| `interp_ssp` | — | `None` | SSP scheme; `None` auto (`'quad'` if RD-SSP else `'linear'`). Accepted: `'linear'` (AT `C`, piecewise linear), `'n2linear'`, `'pchip'`, `'spline'`, `'quad'` (2-D `.ssp`; on a range-independent profile it runs as `'Q'` with the one profile written). `'analytic'` (AT `A`) is refused. |
| `interp_bathymetry` | — | `'linear'` | `.bty` interpolation: `'linear'` (as `doc/ATI_BTY_File.htm` advises) or `'curvilinear'`. |
| `interp_altimetry` | — | `'linear'` | `.ati` interpolation: `'linear'` or `'curvilinear'`. |
| `beam_width_type` | — | `None` | Cerveny width (`ReadEnvironmentBell.f90:178-181`): `'F'` space-filling, `'M'` minimum width, `'W'` WKB (used for `beam_type` ∈ C/R). `None` is `'F'`, uacpy guidance (measured; BELLHOP has no built-in width). |
| `beam_curvature` | — | `None` | `'D'` doubling, `'S'` standard, `'Z'` zeroing (`ReadEnvironmentBell.f90:202-210`). `None` is `'S'`, BELLHOP's built-in value (`bellhopMod.f90:25`). |
| `eps_multiplier` | factor | `None` | Beam-width epsilon multiplier; `None` is `1.0`, BELLHOP's built-in value (`bellhopMod.f90:23`). |
| `r_loop` | m | `None` | Range at which the beam width is chosen, read only by the `'M'` width (`bellhop.f90:378-380`); `None` is 1000 m, uacpy guidance (BELLHOP has no built-in value; the toolbox's calib decks write 1 km, its 100 km decks 25-100 km). |
| `n_image` | count | `None` | Number of images; a whole number ≥ 0 (read as an INTEGER, `ReadEnvironmentBell.f90:217`). `None` is `1`, BELLHOP's built-in (`bellhopMod.f90:22`). |
| `ib_win` | — | `None` | Beam-windowing parameter; a whole number ≥ 0 (read as an INTEGER, `ReadEnvironmentBell.f90:217`). `None` is `5`, BELLHOP's built-in (`bellhopMod.f90:22`). The six Cerveny knobs set on another beam type are not used, with a notice. |
| `component` | — | `'P'` | **Reserved: only `'P'` (pressure) returns a field.** The Cerveny ray-centred field component: `'P'` pressure, `'V'` vertical, `'H'` horizontal particle velocity (`influence.f90:120-130`); any other letter raises `ConfigurationError` (`validate_beam_shape` in `io/bellhop_writer.py`). Only `beam_type='R'` reads it — `InfluenceCervenyRayCen` is the sole `Beam%Component` site — and there `'V'`/`'H'` raise `UnsupportedFeatureError`, since a particle-velocity `.shd` has no `Field` kind; every other beam type ignores the letter with a warning. `'P'` is BELLHOP's built-in value (`bellhopMod.f90:24`). |
| `beam_shift` | — | `False` | Enable beam-shift on boundary reflections; off is BELLHOP's own default (`ReadEnvironmentBell.f90:159-165`). |
| `auto_bounce` | — | `True` | Auto-route *layered* bottoms through BOUNCE (elastic half-spaces run natively); `False` collapses the stack to a half-space (one warning). BOUNCE is the manual's route for elastic boundaries (`doc/bellhop.htm`). |

### Kraken parameters

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `backend` | — | `None` | Modes binary: `'kraken'`/`'krakenc'`; `None` auto-picks `krakenc` for elastic/leaky media and a reflection table, `kraken` otherwise. Over a fluid seabed `kraken`'s real arithmetic solves the seabed above its speed as rigid (`BCImpedanceMod.f90:83-88`), where `krakenc` computes leaky modes (`doc/kraken.htm`); a default run says nothing of it, and `'kraken'` or a `c_high` above the seabed speed gives one `ValidityWarning`. `run_settings(...).engine.backend` records the program that ran. |
| `mode_coupling` | — | `None` | RD mode transitions: `'adiabatic'` or `'coupled'`; `None` = `'adiabatic'`, `field.exe`'s own default (`doc/field.htm`). A set value a range-independent or `MODES` run does not use gives a `FallbackWarning`. |
| `n_segments` | count | `None` | Range segments for RD. `None` puts edges at the bathymetry / SSP / bottom change points and splits so no segment exceeds 2 km, a seafloor depth change of a quarter of the shortest water wavelength, or (range-dependent SSP) a sound-speed change of `0.5·c/f`, at the run's highest frequency; the automatic count is capped at 200 profiles, thinned evenly with a `NumericsWarning`. An int (≥ 2) forces a uniform split. uacpy guidance (the manuals name no segment spacing). |
| `mode_points_per_meter` | pts/m | `None` (derived) | Mode-depth grid density; `None` → 10 points per wavelength, the manual's coupled-mode sampling (`doc/kraken.htm`), floor 1.5 pts/m (uacpy guidance). |
| `field_executable` | path | `None` | `field.exe` path; auto-detected if `None`. |
| `c_low` | m/s | `None` | Lower phase-speed limit of the modal solver; `None` → `0` for a fluid environment, KRAKEN's automatic floor (`doc/kraken.htm`), or the slowest compressional speed when the bottom or the surface carries shear (uacpy guidance). |
| `c_high` | m/s | `None` | Upper phase-speed limit; `None` = 1.05 × max(SSP, bottom) over a fluid or elastic half-space; 10 × the fastest water speed over a `'file'`/`'precalc'` reflection-table seabed (and on `krakenc` over a rigid or vacuum floor); unbounded (`1e9`) over a rigid or vacuum floor on `kraken`; with `leaky_modes=True`, 10 × the fastest speed in the profile, water or seabed. `1e9` is the toolbox's "no upper limit" (`doc/bounce.htm`); the `1.05 ×` and `10 ×` factors are uacpy guidance (`doc/kraken.htm` leaves CHIGH to experience). |
| `n_mesh` | count | `None` | Mesh points per medium (one count, or one per medium on a range-independent run), at the deck's frequency line; `0` = KRAKEN's automatic mesh (`doc/EnvironmentalFile.htm`), 20 points per wavelength, the shear wavelength where a shear speed is set (`ReadEnvironmentMod.f90:101-104`). `None` = `0`, except that a band deck writes, on a medium that would otherwise fall under AT's 10 points at the band's lowest frequency, the least count that keeps them. |
| `interp_ssp` | — | `None` | `TopOpt(1)` sample-connection scheme; unset → `'linear'` (AT `'C'`, the manual's choice when unsure, `doc/EnvironmentalFile.htm`); `'quad'` is Bellhop-only and raises. |
| `n_modes` | count | `None` | Cap on the modes `field.exe` propagates (FLP `MLimit`), and the truncation applied to a `Modes` result; `None` uses every mode found, as `doc/field.htm` does with a large count. |
| `range_offsets` | m | `None` | Receiver range offsets, one per receiver depth (FLP `Rro`): the receiver at depth `z_i` reads the field at `r + ro_i`, a tilted or distorted array (`doc/field.htm`). `None` = zeros, a vertical array. Must not decrease with depth (`field.exe` sorts them, `SourceReceiverPositions.f90:267-268`); read by the range-independent sum only (notice otherwise). |
| `leaky_modes` | — | `False` | Include leaky modes: `c_high` → 10 × the fastest speed in each profile (uacpy guidance, measured), and forces `krakenc`. |
| `top_reflection_file` | path | `None` | Top-boundary reflection-coefficient file. |
| `rmax` | m | `None` | The Acoustics Toolbox's `RMax`: set it to the farthest range you will compute the field at. KRAKEN doubles its mesh until the eigenvalues are accurate at that range (`kraken.f90:80`, `Error*1000*RMax < 1`), so a **larger** value is a **tighter** mesh tolerance; `field.exe` never reads it. `None` = the farthest receiver range, the manual's value (`doc/kraken.htm`); 100 km without a positive receiver range (`compute_modes`), uacpy guidance, measured against the manual's `0`. A set `0` runs (no mesh doubling, with a `NumericsWarning`); a negative value is refused. |
| `mode_depths` | array | `None` | Explicit mode-depth output grid. |

### Scooter parameters

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `c_low` | m/s | `None` | Lower phase-speed limit; `None` = 0.90 × the slowest wave of the problem: every medium's speed (shear speeds and half-spaces included) and every Scholte wave (Brekhovskikh & Lysanov eq. 3.4.29) — `scooter.f90:258`'s margin. |
| `c_high` | m/s | `None` | Upper phase-speed limit; `None` = the toolbox's unbounded `1e9` (`doc/bounce.htm`) over every bottom, so the integral starts at `k_min ≈ 0` — Jensen et al. (*Computational Ocean Acoustics*, ch. 4 App. 1): "it is recommended to initially include these components by choosing k min = 0". A set value is written as given; one that drops a direct or surface-reflected path the receivers see warns. |
| `n_mesh` | count | `None` | Total FE mesh points **per medium** (AT `NG`): one count, or one per medium (water first), each a whole number ≥ 0 written as given; `None` = `0`, the manual's automatic mesh, scaled with frequency and floored at 100 points (`scooter.f90:106-110`). Not points-per-wavelength. |
| `rmax_factor` | factor | `None` | Wavenumber-resolution range multiplier; `None` → 2.0 in every run mode, the ratio of the toolbox's own Scooter decks, narrowband and broadband (`tests/Munk/MunkS.env`, `tests/BroadBand/MunkS.env`: RMax 200 km for receivers to 100 km); `doc/scooter.htm` asks only that RMax reach the farthest receiver. |
| `interp_ssp` | — | `None` | `TopOpt(1)` sample-connection scheme. `'quad'` is Bellhop's external `.ssp` scheme, which the shared AT `EvaluateSSP` cannot read, so it raises. Unset → `'linear'` (AT `'C'`, the manual's choice when unsure, `doc/EnvironmentalFile.htm`). |
| `wavenumber_spectrum` | — | `'positive'` | FLP Opt(2): `'positive'`/`'negative'`/`'both'` wavenumber spectrum; `'positive'` is the manual's recommendation (`doc/fields.htm`). |
| `taper` | fraction | `0.0` | Hanning roll-off over this fraction of the wavenumber span at **each** edge of the window before the Hankel transform (`fieldsco.m:taper`); `0` = rectangular window, the AT reference; `0 <= taper < 0.5`, and a positive taper is refused on a Green's function of fewer than four phase speeds. |
| `stabilizing_attenuation_off` | — | `False` | Disable Scooter's stabilising attenuation (TopOpt pos 7 = `'0'`); `False` keeps SCOOTER's own default (`scooter.f90:129-130`). |

### SPARC parameters

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `c_low` | m/s | `None` | Lower phase-speed limit; `None` = 0.90 × the slowest speed of every medium (shear collapsed to fluid first), Scooter's rule. |
| `c_high` | m/s | `None` | Upper phase-speed limit; `None` = `1e7`, the "no upper limit" every toolbox SPARC deck writes (`tests/iso.env:9`), so `k_min = 0` (Jensen et al., ch. 4 App. 1). Not `1e9`: the per-wavenumber band-pass index (`sparc.f90:398`, `tslib/bandpassc.f90:16`) overflows there on a long source series and silences the source; a deck whose index would overflow is refused. |
| `n_mesh` | count | `None` | Total mesh points **per medium** (AT `NG`): one count, or one per medium (water first); `None` = 20 points per wavelength at the band top `freq_max`; a set `0` is the toolbox's automatic mesh at the deck frequency (`ReadEnvironmentMod.f90:101-106`). Not points-per-wavelength. |
| `interp_ssp` | — | `None` | `TopOpt(1)` sample-connection scheme. `'quad'` is Bellhop's external `.ssp` scheme, which the shared AT `EvaluateSSP` cannot read, so it raises. Unset → `'linear'` (AT `'C'`, the manual's choice when unsure, `doc/EnvironmentalFile.htm`). |
| `output_mode` | — | `'R'` | Which SPARC output geometry to march: `'R'` horizontal array (one run per receiver depth), `'D'` vertical array (one run per receiver range), `'S'` snapshot. All three return a `TIME_SERIES` `Field` on a shared time grid, at the unit-source level (`docs/models/validation.md` §1.3). `'R'` is uacpy guidance: `doc/sparc.htm` names no default. See *Fields, pressure normalization & transmission loss* (§15) and `SPARC.run`. |
| `pulse_type` | — | `None` | AT 4-character pulse-type code; `None` marches the run's `source_waveform` (`'FN+B'`) and falls back to `'PN+B'` without one. The `'B'` band-pass is not modelled by the CW source-spectrum deconvolution, so it adds a few dB to the `'R'` absolute level; `'PN+N'` (no band-pass) calibrates tighter (~±1.5 dB vs Kraken). |
| `n_time_samples` | count | `None` | Number of output time samples over `[0, time_max]`; `None` → the least count with more than eight samples per period at `freq_max`, `floor(8·freq_max·time_max) + 2` (Jensen et al., *Computational Ocean Acoustics*, Sect. 8.2.1.3: "Δt = T/N < 1/(8 f max)"; the manual gives no count). A set value is used verbatim; `run()` warns (naming that count) when its Nyquist sits below the source band, because the p(t) would alias silently. |
| `march_start` | s | `-0.1` | Time the march starts (not the output window, which is `[0, time_max]`); the TSTART of `doc/sparc.htm`'s sample deck. |
| `courant_factor` | factor | `0.999` | Integration time multiplier (TMULT); `1.0` is the maximum stable step (`doc/sparc.htm`), `0.999` the TMULT of the toolbox's `tests/iso.env`. |
| `max_launches` | count | `None` | Cap on the looped axis — receiver depths for `output_mode='R'`, receiver ranges for `'D'` (`'S'` runs once). `None` runs every element; a set cap refuses a grid past it with `UnsupportedFeatureError`, since SPARC marches one subprocess per element. |
| `rmax_factor` | factor | `None` | RMax multiplier on the receiver max range; `None` → `RMax = r + 1.2 × c_fast × time_max`. SPARC synthesises range inline as a direct `deltak` sum (`sparc.f90:595,622`), periodic with period `2π/dk ≈ RMax` (`sparc.f90:116`) and folded mirrored, so receiver `r` sees the replica from `RMax − r`; the default keeps it out of the record — Jensen et al.'s criterion (Sect. 8.3.2) applied to that fold, `1.2` uacpy's measured margin (`4 × r` on the auto window). |
| `freq_min` | Hz | `None` | Pulse-band lower edge; `None` → the span of `run(frequencies=)`, else the band of a marched waveform, else `0` for a canned pulse (its spectral support starts at 0 Hz, `tslib/cans.f90`). |
| `freq_max` | Hz | `None` | Pulse-band upper edge; `None` → as `freq_min`, and for a canned pulse on `f = source.frequencies[0]` the top of the support `tslib/cans.f90` states for its shape: `3f` for the default `'P'` ("support [0, 3F]"), `2f` `'R'`, `2.5f` `'A'`, `3f` `'N'`, and `3f` for a shape stated unbounded (uacpy guidance, the toolbox's SPARC decks). |
| `window_sound_speed` | m/s | `None` | Reference speed for the travel-time window when `run(output_duration=)` is not given (the record end `time_max` is then 2.5 × the travel time to the farthest receiver at this speed — uacpy guidance: the manual gives no window; Jensen et al. §8.2.1.3 size it from the slowest modal group speed); `None` → the slowest speed in `env.ssp` (uacpy guidance), 1500 m/s only when the profile carries no usable speed. |

### RAM parameters

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `backend` | — | `None` | RAM-family backend: `'mpirams'`/`'ramgeo'`/`'rams'`/`'ramsurf'`; `None` auto-dispatches by what each code can represent (uacpy guidance; `ramgeo1.5.f:3-4`, `ramsurf1.5.f:3-11`). |
| `dr` | m | `None` | Range step; `None` → Lytaev optimiser (Lytaev 2023 §4.1; the RAM guide asks for convergence tests rather than a value, `ram.pdf` p.8). |
| `dz` | m | `None` | Depth step; `None` → Lytaev optimiser, floored at `λ/15` (the Acoustics Toolbox RAM writers' step), refined for the trapped modes, with the shallowest seafloor placed `SEAFLOOR_CELL_OFFSET` (a quarter cell) below a node on the fluid backends. |
| `zmax` | m | `None` | PE domain depth; `None` = seafloor + sediment stack + a pad of real seabed (the depth the bottom field reaches over the track) + absorbing layer. |
| `n_pade` | count | `8` | Number of Padé coefficients (2–10); 8 is Collins' sample deck, Lytaev's `[7/8]` and the Acoustics Toolbox RAM writer's order. |
| `n_stability` | count | `None` | Number of stability terms, written as given (`0` turns them off); `None` → 1 (ram.pdf p.3: "ns = 1 or 2 is effective for most problems"), Collins' sample deck's `ns`. rams0.5 takes `ns = 1` itself (`rams0.5.f:176`), so a value set there gives a notice. |
| `stability_range` | m | `None` | Range past which the stability terms are switched off; `None` keeps them at every range (mpiramS writes the max output range, the Collins codes 0 → `2 × rmax`), the RAM guide's default (`ram.pdf` p.7). |
| `q_factor` | — | `None` | `fc / half-bandwidth`: the sweep spans `fc ± fc/q_factor`. `None` is resolved from the source: a single frequency with `record_duration` also unset → the default band (128 bins over `fc·(1 ± 0.25)`, as every broadband engine), `1e6` (one bin) for an explicit `run(frequencies=[fc])`; a single frequency with `T` pinned → `4.0`, the default band's half-width; a multi-frequency source → the array's own half-width, `fc / ((n//2 + ½)·Δf)`. |
| `record_duration` | s | `None` | Time-window width, `Δf = 1/record_duration`. `None` is resolved alongside `q_factor`: a single frequency with `Q` also unset → the default band's spacing (`1.0` for an explicit `run(frequencies=[fc])`); a single frequency with `Q` pinned → `127/(fc·0.5)`, the default band's spacing; a multi-frequency source → `1/Δf` from the array's spacing. |
| `depth_decimation` | factor | `1` | Output depth-decimation factor; `1` = none, as Collins' sample deck (`ram.pdf` p.12-13). |
| `earth_curvature` | — | `False` | Apply the flat-earth (Earth-curvature) transformation; off so RAM matches the other engines' flat Earth, on for long-range work (beyond ~56 km, Etter §5.5.2); uacpy guidance. |
| `absorber_width_wavelengths` | λ | `20.0` | Absorbing-layer width below the seafloor (in wavelengths); uacpy guidance, measured on lossless seabeds — far thicker than the RAM guide's "lower few wavelengths" (`ram.pdf` p.7). |
| `absorber_attenuation` | dB/λ | `None` | Attenuation at the absorbing-layer floor (ramped from the sediment's). `None` = 10, the floor of Collins' sample ramp (`ram.pdf` p.12), raised to the seabed's own where higher so the ramp never decreases (`ram.pdf` p.7); a set value is written as given, with a notice when it is below the seabed half-space's. |
| `n_sediment_points` | count | `None` | Sediment depth control points for the mpiramS profile, written as given (≥ 4); `None` → one per depth cell over the sediment span, at least 1000 (uacpy guidance). |
| `c0` | m/s | `None` | PE reference (expansion) speed — algorithmic, not physical; `None` → Lytaev Eq. (15) (Collins' sample uses 1600 m/s for 1500–1700 m/s phase speeds, `ram.pdf` p.8). |
| `accuracy` | — | `None` | Lytaev optimiser accuracy budget (max \|τ·n_steps\|). `None` → the default chooser's 1e-3 target, relaxed, floored at `λ/15` and capped, each reported as status; a pinned value is met as given (no relaxation, floor or cap) or refused when no grid meets it. |
| `angle_max_deg` | deg | `None` | Max propagation angle bounding the PE spectrum. `None` → Lytaev (2023) §5.5's `max(30°, steepest bathymetry slope)`; a set value is used as given and never narrowed. |
| `rams_rotation_angle_deg` | deg | `None` | `rams` backend rotated-Padé angle (Milinazzo-Zala-Brooke): `None` is 45° unless the stability rule needs less, then the widest stable angle per frequency (warned); a float pins it, marched as given (warned when the rule finds it unstable). 45° is uacpy's measured tuning, in line with Milinazzo, Zala & Brooke (1997) Fig. 11. |
| `rams_rotation` | — | `None` | `rams` Padé branch-cut rotation; `None` = on, uacpy guidance (the 45° tuning and the stability rule were measured with rotation on); a set value on another backend gets the not-used notice. |
| `rams_dr_factor` | factor | `None` | Divisor of the Lytaev `dr` for the `rams` backend, used as given; `None` → 5.0 (uacpy guidance), and 1.0 leaves the λ/5 cap and the stability rule alone. It divides the automatic `dr` only: beside a pinned `dr`, or on another backend, it gives a notice that it is not used. The automatic `rams` step is also held under a stability rule — the rotated Crank-Nicolson step's growth on the steepest propagating components against the seabed's leak of them — which binds above a few kHz and in deep water; a pinned `dr` the rule predicts to diverge warns before the march. |

### Bounce parameters

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `c_low` | m/s | `None` | Min phase velocity for tabulation (must be > 0). `None` → `min(1400, min(env.ssp))`, AT `bounce.htm`'s "lowest speed in the problem"; a fixed 1400 loses the grazing wedge in water slower than that. |
| `c_high` | m/s | `None` | Max phase velocity; `None` is unbounded (`1e9`, the manual's value, `doc/bounce.htm`), tabulating the full 0–90° grazing span. A finite value truncates the table at `acos(c0 / c_high)` and must be > `c_low`. |
| `rmax` | m | `None` | Max range for angular sampling; `None` auto-derives from `receiver.range_max` (10000 m fallback). Refused beside `n_angles`, which sets it. |
| `n_angles` | count | `None` | Exact number of angular samples (a count the deck's RMax cannot give is refused, naming the one it gives); refused beside `rmax`. `None` = bounce computes it from `rmax`. |
| `n_mesh` | count | `None` | Mesh points per sediment medium (`NPTS`), one count or one per medium, written as given; `0` = the reader's 20 points per wavelength. `None` = 100 points per wavelength, at least 100 (`doc/bounce.htm`'s suggestion). Under the reader's floor (`ReadEnvironmentMod.f90:110-112`) refused. |
| `interp_ssp` | — | `None` | `TopOpt(1)` sample-connection scheme. A BOUNCE deck has no water column, so `'quad'` raises. Unset → `'linear'` (AT `'C'`, the manual's choice when unsure, `doc/EnvironmentalFile.htm`). |

### OASES — OAST (transmission loss)

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `compute_contour` | — | `None` | `True` is refused: `'C'` writes range-depth contour files (units 28/29) holding, decimated, the TL the Field already holds, and uacpy reads none of them (`unoast31.f:332-334`, `:776-797`); unset or `False` adds no `'C'`. |
| `compute_depth_average` | — | `None` | `True` adds `'A'`: the depth-averaged TL, `-10·log10` of `|p|²` averaged over every receiver depth (`oasfun22.f:209-242`, `oast.tex:144-146`), returned as `result.components['depth_average']` (a real dB Field on `range`, OAST's native grid); unset → `False`. |
| `complex_contour` | — | `None` | `'J'` option (complex integration contour); unset → `True`, as the manual's worked OAST deck (`oast.tex:476`); the derived line is `K T J`, the bulk pressure `K` the output parameter. |
| `options` | — | `None` | Raw OASES options string, written verbatim; `None` derives it from the three flags above. The string replaces the whole option line, so passing it **together with** any of those flags raises `ConfigurationError` rather than discarding them — that is why the three default to `None` and not to their effective values. A string without `'T'` is refused: OAST writes the TL table uacpy reads only under `'T'` (`unoast31.f:1036-1038`). |
| `integration_offset` | dB/λ | `0.0` | Wavenumber-integration contour offset; `0` asks OASES for its own, the manual's recommended default (`oast.tex:293-303`). |
| `n_wavenumbers` | count | `None` | Number of wavenumber samples; `None` = OASES's automatic sampling (`NW = -1`, `oast.tex:543-550`). A value set must be a power of two (`oast.tex:73`), else it is refused naming the two neighbouring powers. |
| `integration_window` | samples | `None` | `(IC1, IC2)`: the first and last wavenumber samples integrated (`oast.tex:528-538`), with a pinned `n_wavenumbers`; `None` = the whole window. Under automatic sampling it has no effect, which the run says. |
| `c_low` / `c_high` | m/s | `None` | `CMIN`/`CMAX`, Block VII's phase-speed window (`unoast31.f:213`); `None` derives `c_low` as every wavenumber-integration engine does — 0.90 × the slowest wave of the problem, every medium's speed (shear speeds and half-spaces included) and every Scholte wave (Brekhovskikh & Lysanov eq. 3.4.29) (`scooter.f90:258`; Jensen et al., ch. 4 App. 1), so an elastic seabed's shear and interface branches are integrated — and `c_high` as `1e8`, the CMAX of the manual's worked decks (`oast.tex:487`). |
| `range_min` | m | `None` | First range of OAST's native TL grid (`XLEFT`); `None` → 0, the manual's worked deck (`oast.tex:489`). |
| `range_max` | m | `None` | Last range of OAST's native TL grid (`XRIGHT`); `None` → `receiver.range_max`, the range OAST's automatic sampling sizes its wavenumber step against (`unoast31.f:427-430`). |
| `vrec` | m/s | `0.0` | Source/receiver velocity `V` of option `'d'` (Block III, `oast.tex:37`): source and receiver move at the same speed, `V` its projection on the line joining them, so there is no Doppler shift (`oast.tex:215-222`); read only under a raw `options` carrying `'d'`, refused non-zero otherwise. `0` is uacpy guidance (stationary). |
| `dip_angle_deg` | deg | `None` | Fault dip angle of the dip-slip moment source that option `'4'` selects (`unoast31.f:1117-1122`). `None` writes 0 when `'4'` is present, and raises when it is not. |

### OASES — OASP (broadband pulse / transfer function)

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `n_time_samples` | count | `None` | Power-of-two FFT length (samples per receiver trace). `None`: 4096 for the default sweep (uacpy guidance: OASES requires only NT = 2^M; its worked decks use 2048, `oasp.tex:583`, and 1024, `oasp.tex:773`); for a requested band (`frequencies=`, a TIME_SERIES pulse, COHERENT_TL's source frequency) the smallest power of two whose bins are as fine as the requested spacing. A value set must be a power of two (`oasp.tex:129`) and is written as given; when its bins are coarser than the requested spacing a `NumericsWarning` names the count that reaches it. |
| `freq_max` | Hz | `None` | Sweep upper edge; `None` → 2.5 × centre frequency, as the manual's worked OASP deck (`oasp.tex:572`, `:583`). |
| `freq_min` | Hz | `None` | Sweep lower edge; `None` → `0`, as the manual's worked OASP deck (`oasp.tex:583`). A pinned edge that differs from a band the call names is refused; beside a derived band it gets a not-used notice. |
| `fc` | Hz | `None` | Pulse carrier frequency; `None` → the midpoint of the run's frequency band (a single `source.frequencies` entry is its own centre). The manual's FRC (`oasp.tex:84`), which OASP only records (`oasp.tex:278-280`). |
| `integrand_plot_step` | factor | `None` | Integrand-**plot** frequency decimation (INTF, `unoasp22.f:382,591`); `None` → 0, the manual's worked deck (`oasp.tex:582`), no plots. The `.trf` always carries every bin `LXP1..MX`. |
| `options` | — | `None` | Raw OASES options string; `None` → `'K J'` (the bulk pressure, `oasp.tex:178-183`). |
| `integration_offset` | dB/λ | `0.0` | Wavenumber-contour offset; `0` asks OASES for its own, the manual's recommended default (`oast.tex:293-303`). |
| `n_wavenumbers` | count | `None` | Wavenumber sample count; `None` = auto. |
| `integration_window` | samples | `None` | As OAST (`oasp.tex:652-660`). |
| `c_low` / `c_high` | m/s | `None` | `CMIN`/`CMAX`, Block VII's phase-speed window; `None` derives the pair as OAST does (0.90 × the slowest wave of the problem, Scholte waves included); the derived `c_high` is `1e8`, as the manual's worked OASP decks write (`oasp.tex:581`). |
| `dip_angle_deg` | deg | `None` | Fault dip angle of the dip-slip moment source that option `'4'` selects, as OAST. |

### OASES — OASR (reflection coefficients)

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `angles_deg` | deg | `None` | **Uniformly spaced** angle grid; `None` → `linspace(0, 90, 181)`, the grid of the manual's worked OASR deck (`oasr.tex:373`). OASR regenerates the axis from `(min, max, count)`, so a non-uniform array raises. |
| `angle_type` | — | `'grazing'` | `'grazing'` (native, OASR's default axis, `oasr.tex:151-152`) or `'incidence'` (grazing = 90 − incidence). |
| `reflection_type` | — | `None` | `'P-P'` (unset → this, OASR's default, `oasr.tex:138-140`) or `'transmission'`. `'P-SV'` and `'P-Slow'` raise `UnsupportedFeatureError`: the water column the wave arrives through carries no SV wave, and no uacpy carrier expresses the poro-elastic medium the Biot slow wave needs. |
| `options` | — | `None` | Raw OASES options string; `None` derives from `reflection_type`. |
| `plot_angle_step` | factor | `None` | Angle-axis **plot** decimation (`unoasr21.f:167`); the `.rco`/`.trc` tables always carry every angle. A whole number ≥ 0; `0` leaves the plot out. |
| `plot_frequency_step` | factor | `None` | Frequency-axis plot decimation, symmetric with `plot_angle_step` (a whole number ≥ 0). |
| `interface_roughness` | m | `None` | Per-interface RMS roughness (top → bottom). |

### OASES — OASN (noise covariance / replicas)

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `options` | — | `None` | OASES options string; `None` → `'N'` (covariance) / `'R'` (replica). |
| `surface_noise_level_dB` | dB re 1 µPa²/Hz | `0.0` | Surface-generated noise spectral level. Written exactly as given; disabled when `abs(level) < 0.01`, tested at single precision as OASN reads it (`oasnun22.f:183`); a **negative** level at or beyond −0.01 is a spectrum-file unit number to OASN, not a quieter surface (`oasnun22.f:191-193`), and is refused. Requires a vacuum or air `env.surface` — OASES stops on an upper half-space faster than 500 m/s. |
| `white_noise_level_dB` | dB re 1 µPa²/Hz | `None` | Uncorrelated per-hydrophone white-noise level, added to every covariance diagonal. OASES has no off switch for it (`oasnun22.f:228`), so an explicit `0.0` is a literal 0 dB re 1 µPa²/Hz (1e-12 Pa²/Hz on the covariance diagonal) per sensor; `None` writes −200 dB, numerically nil (uacpy guidance: the manual gives no level). |
| `deep_noise_level_dB` | dB re 1 µPa²/Hz | `0.0` | Deep broad-area source spectral level. Disabled when `level ≤ 0.01`, including any negative value (`oasnun22.f:233`) — deliberately not the same rule as `surface_noise_level_dB`. OASES's deep loop keeps two defects uacpy does not correct (a plane-geometry division with `ICDR` unset, `oasnun22.f:705`, and a stale taper window, `:713-714`): a deep-only level reads `10·log10(c/f)` dB high and off in depth shape — see the OASES page. |
| `deep_source_depth` | m | `None` | Depth of the deep noise sheet; `None` → half the water depth (uacpy guidance). |
| `discrete_sources` | list | `None` | Point sources: dicts with `depth`/`x`/`y` (m) and `level_dB` (dB). OASES carries no per-source phase; any other key raises. The deck lists them farthest first: OASN's automatic sampling sizes its wavenumber step from the first record only (`oasnun22.f:426-428`). |
| `receiver_gains_dB` | dB | `None` | Per-element array gain, one value or one per element; `None` → 0 dB. Applied to the signal and the ambient noise, not to the white noise (`oasn.tex:242-244`). |
| `replica_xmin` / `replica_xmax` | m | `None` | Replica grid x-bounds; `None` → 100 / 10000 (uacpy guidance; OASN reads the grid from the deck and has no default, `unoasn22.f:184-186`). |
| `replica_nx` | count | `None` | Replica grid points in x; `None` → 50 (uacpy guidance). A pin is a whole number ≥ 1, as `replica_ny` / `replica_nz`; a replica knob pinned on a covariance run is noticed as unused. |
| `replica_ymin` / `replica_ymax` | m | `None` | Replica grid y-bounds; `None` → 0 / 0, as the manual's worked deck (`oasn.tex:491`). |
| `replica_ny` | count | `None` | Replica grid points in y; `None` → 1, as the manual's worked deck (`oasn.tex:491`). |
| `replica_zmin` / `replica_zmax` | m | `None` | Replica grid depth-bounds; `None` → 10 / `env.depth − 10` (uacpy guidance). |
| `replica_nz` | count | `None` | Replica grid points in depth; `None` → 20 (uacpy guidance). |
| `c_low` / `c_high` | m/s | `None` | Phase-speed bounds for the wavenumber integrations; `None` → 0.90 × the slowest wave of the problem, Scholte waves included (`scooter.f90:258`) / 1e8, which the manual asks for surface noise (`oasn.tex:283-285`). |
| `integration_offset` | dB/λ | `0.0` | Wavenumber-integration contour offset; `0` asks OASES for its own default (`oasn.tex:212-213`). |
| `n_wavenumbers` | count | `None` | Wavenumber sample count. `None`: OASES's automatic sampling for the replicas; the surface- and deep-noise blocks have none, so `None` writes 1600 / 63836 / 100 (continuous / discrete / evanescent) for surface noise (uacpy guidance, measured) and the manual's worked-deck 400 / 400 / 100 for deep noise (`oasn.tex:480`); a pinned `n` writes `n`, `n`, `max(1, n // 4)`, with a notice below 10 discrete samples (uacpy guidance). |
| `integration_window` | samples | `None` | As OAST, on the discrete-source and replica integrations (`oasn.tex:103-105`). |

### OASES — OASS (reverberation / scattered field)

OASS is a post-processor: `run()` drives the mean-field producer (OAST or OASR
with option `'s'`) first, then `oass2` over the `.rhs` it leaves behind. Both
decks share one work dir, and the producer's `Result` comes back as
`components['mean_field']`.

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `correlation_length` | m | — | **Required.** `CL` of the roughness power spectrum. OASS does not adopt it from the mean-field run (`oass.tex:182-183`). Zero, NaN and infinity raise `ConfigurationError`; a negative value raises `UnsupportedFeatureError` (volume scattering). |
| `spectral_exponent` | — | `None` | `M` of the roughness spectrum; must exceed 1.5 or the spectrum is not integrable. `None` is 2.5, OASES's own Goff-Jordan exponent (`oaseun31.f:103-104`), the "very smooth variation" end of the manual's range (`oassp.tex:356-358`); a pin under the Gaussian spectrum gets a not-used notice. |
| `roughness_spectrum` | — | `'gaussian'` | `'gaussian'` (OASES's own default, `oass.tex:136-137`) or `'goff-jordan'` (option `'g'`). |
| `rms_roughness` | m | `None` | `|RG|` at the scattering interface **in the OASS deck**; `None` → the environment's own. The mean-field deck always uses the environment's value, so the scattering interface must be rough **in `env`** (a smooth one raises `ConfigurationError` before any binary runs); this knob cannot supply the roughness. |
| `interface` | count | `None` | `INTFC`, the deck-layer index of the scattering interface (a whole number ≥ 1); `None` → the seafloor. Must be rough in the environment, or the mean field writes an empty `.rhs`. |
| `multiple_scattering` | — | `False` | Option `'p'`: perturbed boundary operator, a lower bound on the reverberation level. |
| `mean_field` | — | `None` | The producer model; `None` → `OAST(options='K J T s')`. A supplied one keeps its own `verbose` and `timeout`; only `work_dir`, `cleanup` and `use_tmpfs` follow this run. Only OAST qualifies, and the option line must carry `'s'`. An OASR producer is refused: it stamps every `.rhs` record with interface 2 and samples the wavenumber axis in `cos θ`, so OASS cannot recover a uniform `DLWVNO` from it. |
| `c_low` / `c_high` | m/s | `None` | `CMIN`/`CMAX`. `c_low` is physically significant, not a tuning knob: raising it above the mean field's truncates the scattering integral (measured 30 dB), lowering it is inert. `None` → the mean field's own bound; a different value warns. |
| `receiver_gains_dB` | dB | `None` | Per-element gain, Block VI column 5. |
| `options` | — | `None` | Raw OASES options string; `None` → the run mode's product letter (`'r'` / `'a'`) plus the flags above. Mutually exclusive with them. |

OASS takes no `integration_offset` or `n_wavenumbers`: Block III's `COFF` is
unreachable (`unoass21.f:596`) and `NWVNO` is re-derived from the `.rhs`
(`unoass21.f:209-215`), so both are set on `mean_field`.

`RunMode.REVERBERATION` returns a real dB `Field` tagged `kind='reverberation'`
(`-10·log10 E[|p_scat|²]`, **not** transmission loss, so it does not compare
against a TL field). `RunMode.COVARIANCE` returns a `Covariance` shaped
`(1, n_rcv, n_rcv)` — the same carrier and file format OASN produces, which is
why the OASES manual ships an `addcov` utility to sum the two. One product per
run: `'a'` sharing a deck with `'r'` returns a silently zero covariance, so the
run mode picks exactly one letter.

### OASES — OASSP (scattered-field realizations)

OASSP is OASS's time-domain counterpart and the same shape of post-processor:
`run()` drives an `OASP` mean-field run with option `'s'` first, then `oassp2`
over the `.rhs`/`.vol` pair it leaves behind, in one shared work dir. Where
OASS returns the *expectation* of the scattered field, OASSP returns one
**realization** of it, in OASP's own `.trf` format and read by the same reader.
The producer's `Result` comes back as `components['mean_field']`. OASSP
takes no `multiple_scattering` (OASS does): the vendor disabled re-scattering
in OASSP's integration path, so option `'p'` would be parsed and do nothing.

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `correlation_length` | m | — | **Required.** `CL` of the roughness power spectrum; OASSP reads it from this deck, not from the mean field. Zero, NaN and infinity raise `ConfigurationError` before the mean field runs. A **negative** value is OASES' switch to the 12-token volume-scattering record and raises `UnsupportedFeatureError` — see below. |
| `spectral_exponent` | — | `None` | `M` of the roughness spectrum; must exceed 1.5 or the spectrum is not integrable. `None` is 2.5, OASES's own Goff-Jordan exponent (`oaseun31.f:103-104`), the "very smooth variation" end of the manual's range (`oassp.tex:356-358`); a pin under the Gaussian spectrum gets a not-used notice. |
| `roughness_spectrum` | — | `'gaussian'` | `'gaussian'` (OASES's own default, `oass.tex:136-137`) or `'goff-jordan'` (option `'g'`). |
| `rms_roughness` | m | `None` | `\|RG\|` at the scattering interface **in the OASSP deck**; `None` → the environment's own. The scattering interface must be rough **in `env`** — the mean field reads only the environment, and a smooth one raises `ConfigurationError`. The scattered field is linear in it (`pow = \|ROUGH(INTFCE)\|`, `unoassp30.f:601, :613`). |
| `interface` | count | `None` | Cross-check only. OASSP reads the scattering interface out of the `.rhs` (`unoassp30.f:548-549`); a value that disagrees with the file raises rather than attaching the spectrum to a layer the binary never looks at. |
| `realization` | count | `0` | Realization index `k` (a whole number ≥ 0); the OASES seed is `-123 - k` (`unoassp30.f:170`, `:535`), so a given `k` is reproducible and different `k` are different draws. Under cylindrical geometry it also shifts the full-Bessel tabulation window by `k/2` in `kr` (`:171`, `:639-640`), which OASP pins to 0 — so a large-`k` ensemble is not *purely* statistical. |
| `scattered_only` | — | `True` | Option `'s'`: zero the source arrays so the `.trf` holds the scattered field alone (`unoassp30.f:628-635`), as both of the manual's worked OASSP decks (`oassp.tex:592`, `:653`); the program's own default is the total field (`oassp.tex:231-233`). |
| `mean_field` | — | `None` | The producer model; `None` → an `OASP` built from the FFT-grid knobs below; a supplied one keeps its own `verbose` and `timeout`. Mutually exclusive with them, since the mean field owns the grid. Its option line must carry `'s'` (e.g. `OASP(options='K J s')`) — without it OASP writes no `.rhs` at all — or it is refused. |
| `n_time_samples`, `freq_min`, `freq_max`, `fc` | — | `None` | Passed to the mean-field `OASP`. **Not** OASSP's own Block VIII — see below. |
| `c_low` / `c_high` | m/s | `None` | `CMIN`/`CMAX`. `c_high` needs plane geometry, i.e. `Source(source_type='line')`: a point Source runs cylindrical, where OASSP forces `CMAX = 1e12` and a full Hankel transform (`unoassp30.f:205-217`). |
| `n_wavenumbers`, `integration_offset` | — | — | As for `OASP`. |
| `integration_window` | samples | `None` | As for `OASP`, on the OASSP deck. |
| `options` | — | `None` | Raw OASES options string; `None` → `'K J s'` plus the flags above. Mutually exclusive with them. Validated by the wrapper, because OASSP's `GETOPT` closes with an empty `ELSE` (`unoassp30.f:1050`) and discards an unknown letter with no diagnostic. |

**Block VIII is not the user's to set.** OASSP replaces its deck's
`NT`/`FR1`/`FR2`/`DT` with the `.rhs`'s own values but warns on only two of the
four (`unoassp30.f:181-188`), so a band that differs from the mean field's is
substituted with no message at all. The wrapper reads those four straight out
of the `.rhs` and writes them back, which makes the substitution a no-op — ask
for a different band by configuring the mean field, or by passing
`run(frequencies=…)`, which is forwarded to it. `Field.coords['frequency']` is
therefore always the mean field's axis.

`RunMode.BROADBAND` (default) returns a complex `Field` of the scattered `H(f)`
on `(depth, range, frequency)`, tagged `phase_reference='travelling_wave'`;
`RunMode.TIME_SERIES` returns real `p(t)` and needs `source_waveform` +
`sample_rate`.

**Not supported.** Volume scattering (`oassp.tex` ≥ 2.2) needs `SKW`, `M`,
`RMS` and `GAM` on the layer record, and no uacpy carrier has a field for them,
so it raises rather than writing zeros into three of twelve columns. Option
`'U'` (wave-field decomposition) writes five separate `.trf` files and is
refused; for the record, its file↔component order is the source's
(`unoassp30.f:48-52`: `trf` total, `trfdc` down-going compressional, `trfds`
down-going shear, `trfuc` up-going compressional, `trfus` up-going shear),
verified through `KERDEC` — `oassp.tex:185-189` has `ds` and `uc` swapped.
Option `'b'` (bistatic) needs an undocumented third input file. The scattering
interface must be a **bottom** interface: the water-layer records carry no
nine-token form, so a rough sea surface cannot be the scatterer (the same limit
as OASS).
