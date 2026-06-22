# uacpy Documentation

**uacpy** is a unified Python interface to the standard underwater-acoustics
propagation models (Bellhop, Kraken, Scooter, RAM, SPARC, OASES) plus toolkits
for signal processing, sonar, communications, and ambient noise, and a
real-world-data layer that builds an `Environment` from GPS coordinates.

This guide is example-driven: each concept is introduced briefly, then shown
with a minimal runnable snippet. Every public class and function also carries a
docstring (`help(uacpy.Bellhop)`), `uacpy/examples/` holds 37 complete scripts,
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
  time-domain FDTD (SPARC), and plane-wave reflection (Bounce/OASR).
- **DSP & sonar** (`acoustic_signal`, `sonar`) — waveforms, beamforming,
  matched filtering, the sonar equation.
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
pip install -e ".[dev]"   # editable install + dev/test dependencies
./install.sh -y           # compile native binaries into uacpy/bin/ (gitignored)
```

`install.sh` builds Bellhop, Kraken, Scooter, RAM, and SPARC, and (by
default) downloads and builds the OASES suite from MIT. Useful flags:

- `--bellhop fortran|cxx|cuda` — pick the Bellhop backend to build
  (`cuda` requires a CUDA toolkit; the Fortran build is the safe default).
- `--oases yes|no` — include or skip OASES (academically licensed, fetched
  at install time, never redistributed).
- `--force` — rebuild even if binaries already exist.
- `--no-models` — pure-Python install, skipping all native builds.

Run `./install.sh --help` for the full flag list. Always use the project
virtualenv for execution.

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

field = uacpy.Bellhop().compute_tl(env, source, receiver)

uacpy.plot_field(field, env=env)
plt.show()

# field is a Field: TL as a (depth × range) array, plus the axes.
print(field.tl.shape)                 # (101, 200)
print(field.at(depth=50.0).tl.shape)  # (200,) — TL vs range at source depth
```

`compute_tl` is the convenience wrapper for transmission loss; it is
exactly `model.run(env, source, receiver, run_mode=RunMode.COHERENT_TL)`.

The model classes and headline plotters are re-exported at the top level
(`uacpy.Bellhop`, `uacpy.plot_field`, …); `dir(uacpy)` is the full index.

## 4. Core Concepts

The whole library is organised around one pipeline:

```
Environment + Source + Receiver  →  Model.run()  →  Result
```

- **`Environment`** — the ocean: bathymetry, sound-speed profile, surface,
  bottom, volume absorption (§Environment).
- **`Source`** — emitter depth(s) and frequency(ies).
- **`Receiver`** — a grid (depth × range) or a paired line of hydrophones.
- **`Result`** — a typed object carrying the output (§Results).

### The model contract

Every model subclasses `PropagationModel` and shares one fixed `run`
signature:

```python
def run(self, env, source, receiver, run_mode=None, *,
        frequencies=None,        # broadband sweep override for source.frequencies
        source_waveform=None,    # real source pulse, for TIME_SERIES
        sample_rate=None,        # Hz, for TIME_SERIES
        output_duration=None):   # desired output window (s), broadband synth
    ...

result = model.run(env, source, receiver, run_mode=RunMode.COHERENT_TL)
```

There is **no `**kwargs`** — an unrecognised keyword raises `TypeError` at
the call site. The only sanctioned per-model extras are `n_modes=` (Kraken
family, to cap the modal set) and `output_duration=` (broadband
synthesizers). Everything that *tunes* a model — beam type, PE step sizes,
Padé order, array geometry — is a **constructor** argument, not a `run()`
argument (see *Constructor-only configuration* below).

### The `compute_*` convenience family

Each `RunMode` has a thin wrapper that capability-checks the model and
forwards to `run()` with the right mode and kwargs:

| Wrapper | Run mode | Extra kwargs |
|---|---|---|
| `compute_tl` | `COHERENT_TL` (also `INCOHERENT_TL`/`SEMICOHERENT_TL` via `run_mode=`) | — |
| `compute_rays` / `compute_eigenrays` / `compute_arrivals` | `RAYS` / `EIGENRAYS` / `ARRIVALS` | — |
| `compute_modes` | `MODES` | `n_modes=` (and takes no `receiver`) |
| `compute_reflection` | `REFLECTION` | — |
| `compute_time_series` | `TIME_SERIES` | `source_waveform=`, `sample_rate=`, `output_duration=` |
| `compute_transfer_function` | `BROADBAND` | `frequencies=` |
| `compute_covariance` / `compute_replicas` | `COVARIANCE` / `REPLICA` | — |

All take `(env, source, receiver)` positionally (except `compute_modes`,
which omits `receiver` — modes are receiver-independent depth
eigenfunctions). Calling a wrapper a model doesn't support raises
`UnsupportedFeatureError` listing models that do.

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
base = uacpy.RAM(dr=2.0, dz=0.5, np_pade=8)
for dr in (1.0, 2.0, 4.0):
    field = base.copy(dr=dr).compute_tl(env, source, receiver)
```

For many independent runs, batch them with `uacpy.run_parallel` over
`uacpy.Job` objects rather than looping in Python.

### Model provenance & licence

Every model wraps a third-party engine with its own authorship and licence.
That metadata lives in a catalogue (`uacpy.models.sources.MODEL_SOURCES`) and
is surfaced per instance:

```python
m = uacpy.Kraken()
m.source            # 'acoustics_toolbox' — catalogue id (declared on the class)
m.provenance.name   # 'Acoustics Toolbox'  (.authors / .license / .url / .note too)
m.citation          # bibliographic string to cite in a write-up
```

The catalogue flags drive policy, mirroring the `uacpy.data` source catalogue:
constructing an engine whose licence forbids commercial use — currently only
**OASES** (academic, non-redistributable) — emits a one-time `UserWarning`, so
a licence-restricted result is never produced silently. GPL / public-domain
engines (Acoustics Toolbox; the Collins RAM family) stay quiet.

### Results

Every `run()` returns a typed `Result` subclass chosen by the run mode:
`Field` (TL / H(f) / p(t) — one unified array type whose physical meaning
follows from its dtype and coordinate axes), `Rays`, `Modes`, `Arrivals`,
`Covariance`/`Replicas`, `ReflectionCoefficient`. `Field` exposes `.tl`,
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
`ConfigurationError` (bad parameter value), `FileFormatError` (I/O parse
failure), and `DataFetchError` (the data layer). uacpy never raises a bare
`ValueError` for its own checks.

### Status output, logging & parallelism

Models are quiet by default, printing only warnings and errors. Pass
`verbose=True` (or `'info'`) to see progress, `verbose='debug'` for
subprocess command lines and grid-resolution choices. Status goes through
`uacpy._log` (never `print`); user-facing notices use `warnings.warn`.

Each model instance owns its own scratch directory, so independent
instances are safe to run concurrently — use `uacpy.run_parallel` to fan
runs out across processes.

## 5. Environment

The `Environment` is the physics-agnostic carrier that describes the ocean: it
holds the bathymetry, sound-speed profile, boundaries, and volume absorption.
Every propagation model consumes the *same* `Environment` — the object knows
nothing about which solver runs over it. Build it once, reuse it everywhere.

```python
from uacpy import Environment

env = Environment(bathymetry, ssp, altimetry, bottom, surface, absorption)
```

All arguments except `bathymetry` are optional. Each is a small typed carrier
(below); most also accept a shorthand the constructor coerces (a scalar SSP, a
preset-name bottom, …). `env.depth` is a **read-only** property — the maximum
water depth, derived from `bathymetry`. Convention throughout: metres, Hz, m/s,
g/cm³, dB-per-wavelength; depth positive **down**, altimetry positive **up**.

### Bathymetry and altimetry

`bathymetry` is the only required argument. Pass a scalar for a flat bottom, or
a list of `(range, depth)` pairs for a sloping/range-dependent seafloor:

```python
env = Environment(bathymetry=100)                       # flat, 100 m
env = Environment(bathymetry=[(0, 100), (10_000, 200)])  # wedge, 100→200 m
```

`altimetry` describes a non-flat sea surface as `(range, height)` pairs (height
positive up). Build a realistic rough surface from a Pierson–Moskowitz spectrum
with `generate_sea_surface`, which returns an array shaped for `altimetry=`:

```python
from uacpy import generate_sea_surface

surf = generate_sea_surface(max_range=10_000, wind_speed_ms=10, seed=0)
env = Environment(bathymetry=100, altimetry=surf)
```

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
one boundary with a `UserWarning` (`collapse={'surface': 'r0'|'rmax'|'mean'|'median'}`).
The `Surface` carrier still lets you build, fetch, and **plot** the marginal ice
zone (`plot_environment` draws it from `env.surface`). An elastic ice surface is
best run with **Bellhop** (Kraken's `krakenc` aborts on an elastic top). The data
layer can build a marginal-ice-zone surface straight from sea-ice climatology with
`uacpy.data.sea_ice_surface_transect(start, end)`.

### Sound-speed profile

The SSP **shape** (the profile data) lives on the environment; the
*interpolation scheme* used between samples is a model knob, not an env field
(e.g. `Bellhop(interp_ssp='cubic')`). Pass `ssp=` in any of these forms — the
constructor coerces them to a `SoundSpeedProfile`:

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
SoundSpeedProfile.from_mackenzie(depths, temperature_c, salinity_psu)  # from T/S
```

A **range-dependent** (2-D) profile carries a `data[n_depth, n_range]` matrix
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

If a profile is shallower than the seafloor, the constructor extends its deepest
sample down to `env.depth` automatically.

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

The default (no `bottom=`) is a fluid sand-like half-space (1600 m/s, 1.5 g/cm³,
0.5 dB/λ). For range-dependent or layered seabeds, build a `Bottom` explicitly.

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
from uacpy import SeabedColumn, SedimentLayer, BoundaryProperties

col = SeabedColumn(
    layers=[
        SedimentLayer(thickness=10, sound_speed=1600, density=1.7, attenuation=0.8),
        SedimentLayer(thickness=30, sound_speed=1750, density=2.0, attenuation=0.5),
    ],
    halfspace=BoundaryProperties.from_preset('limestone'),
)
Environment(bathymetry=100, bottom=col)                      # layered, range-indep

rdl = Bottom.from_columns([near_col, far_col], ranges=[0, 10_000])  # range-dep layered
```

`SeabedColumn.from_presets` builds a layered column straight from material names:

```python
col = SeabedColumn.from_presets(
    layers=[('sand', 10), ('clay', 30)],     # (preset, thickness_m)
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

`acoustic_type` is usually inferred: any non-default property → `'half-space'`,
a `reflection_file` → `'file'`, nothing → `'vacuum'` (pressure-release surface,
the default). Pass it explicitly only for the distinct models `'rigid'` and
`'grain-size'`. The default `surface` is a vacuum (pressure-release) boundary.

### Materials catalog

`uacpy.materials` provides class-typical geoacoustics for common seabeds
(`clay`, `silt`, `sand`, `gravel`, `moraine`, `chalk`, `limestone`, `basalt`,
`granite`):

```python
from uacpy import MATERIALS, list_materials, get_material, BoundaryProperties

list_materials()                 # ['basalt', 'chalk', 'clay', ...]
get_material('sand')             # {'sound_speed': 1650.0, 'density': 1.9, ...}
bp = BoundaryProperties.from_preset('sand')                 # fluid half-space
bp = BoundaryProperties.from_preset('limestone', elastic=True)   # keep shear
```

Presets are **fluid by default** (shear dropped) so they work with every model;
pass `elastic=True` to keep the preset's shear speed/attenuation (needed only
for elastic-capable solvers). `SedimentLayer.from_preset(name, thickness=...)`
does the same for a layer.

### Volume absorption

`env.absorption` carries the water-column volume-attenuation model. Pick one;
each model emits the right Acoustics-Toolbox `TopOpt` character automatically:

```python
from uacpy import Thorp, FrancoisGarrison, Biological, ConstantAbsorption

Environment(bathymetry=100, absorption=Thorp())             # frequency-only
Environment(bathymetry=100, absorption=FrancoisGarrison(    # T/S/pH/depth model
    temperature_c=10, salinity_psu=35, pH=8, z_bar_m=1000))
Environment(bathymetry=100, absorption=ConstantAbsorption(0.1))   # flat dB/λ
Environment(bathymetry=100, absorption=Biological(          # fish-bladder layers
    [(0, 50, 1000, 5, 0.5)]))   # (z_top, z_bottom, f0, Q, a0)
```

Default is `None` (no explicit volume absorption). The bare formulas
`thorp_db_per_km(f)` and `francois_garrison_db_per_km(f, ...)` are available
for plotting attenuation curves directly.

### Real-world environments

`uacpy.data.fetch_environment` assembles a ready-to-run `Environment` for a
GPS point from public datasets: GEBCO bathymetry, WOA23/Copernicus sound speed,
and grain-size → bottom geoacoustics.

```python
import uacpy.data as data

env = data.fetch_environment((43.2, 7.5), date='2026-06-14', bottom='sand')
```

Each axis can be a **literal** (`ssp=`, `bathymetry=`, `bottom=`, exactly as
`Environment` takes them) and/or fetched from one or more `*_sources` tried in
order. `transect_to=(lat, lon)` samples bathymetry (and optionally SSP/bottom)
along a great-circle path for a range-dependent environment;
`with_absorption=True` attaches a site-specific Francois–Garrison model.
Fetching is cache-first; an offline install (`install.sh --data`) lets
`*_sources='local'` run with no network. See `help(data.fetch_environment)` and
`example_37` for the long tail of options.

## 6. Source and Receiver

`Source` and `Receiver` carry only **geometry and spectrum** — solver knobs
(launch angles, beam counts, …) were purged from them and now live on the model.

`Source(depths, frequencies)` places one or more sources. Depths are positive
down; multiple depths must be strictly increasing (outputs are indexed by source
depth). Frequencies may be a scalar or array.

```python
from uacpy import Source

Source(depths=50, frequencies=100)              # single source, single tone
Source(depths=[10, 20, 30], frequencies=200)    # vertical source array
Source(depths=50, frequencies=[100, 200, 400])  # multi-frequency
```

`Receiver(depths, ranges)` places the field-evaluation points. Depths are
positive down, ranges are outward from the source (metres):

```python
import numpy as np
from uacpy import Receiver

Receiver(depths=50, ranges=1000)                              # single point
Receiver(depths=np.linspace(10, 90, 9), ranges=5000)          # vertical array
Receiver(depths=np.linspace(0, 100, 51),
         ranges=np.linspace(0, 10_000, 201))                  # full grid
```

By default a receiver is a **grid**: the field is evaluated on the full
depth × range cross-product. Pass `receiver_type='line'` to pair depths and
ranges point-by-point instead (e.g. a glider track or tilted array), requiring
equal-length `depths` and `ranges` (or one scalar):

```python
Receiver(depths=[10, 20, 30], ranges=[100, 200, 300], receiver_type='line')
```

## 7. Propagation Models

A propagation model takes the `Environment` / `Source` / `Receiver` triple and
computes the acoustic field. Every model subclasses `PropagationModel` and
shares one contract:

- **Constructor-only configuration.** Every solver knob is a constructor
  argument (`Bellhop(beam_type='B', n_beams=500)`, `RAM(dr=2.0, np_pade=8)`).
  To sweep a parameter, build one model per setting — `model.copy(**overrides)`
  clones an instance with one knob changed.
- **Fixed run signature.** `run(env, source, receiver, run_mode=None, *,
  frequencies=None, source_waveform=None, sample_rate=None, output_duration=None)`.
  The Kraken family adds `n_modes=`. No `**kwargs` — an unexpected keyword
  raises `TypeError`.
- **`compute_*` convenience family.** Thin wrappers over `run()`, one per run
  mode: `compute_tl`, `compute_rays`, `compute_eigenrays`, `compute_arrivals`,
  `compute_modes`, `compute_reflection`, `compute_time_series`,
  `compute_transfer_function`, `compute_covariance`, `compute_replicas`. Each
  capability-checks first, then forwards its mode's kwargs.
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
| **SPARC** | Time-domain PE. Direct synthesis of the transient pressure pulse `p(t)` without an IFFT step. |
| **OASES** | Wavenumber integration for arrays and elastic seismo-acoustics: TL (OAST), array covariance/replicas (OASN), reflection coefficients (OASR), pulse/broadband (OASP). |
| **Bounce** | Not a propagation model — tabulates a bottom reflection coefficient `R(θ)` to a `.brc`/`.irc` file for the others to consume. |

### Capability matrix

Verified against each model's `_supported_modes` / `_supports_*` flags.

| Model | Run modes | Range-dep. | Elastic | Altimetry | Broadband |
|---|---|---|---|---|---|
| **Bellhop** | TL (coh/incoh/semicoh), RAYS, EIGENRAYS, ARRIVALS, BROADBAND, TIME_SERIES | bathy + SSP + bottom | yes | yes | yes |
| **Kraken** | MODES, COHERENT_TL, BROADBAND, TIME_SERIES | bathy + SSP (modes) | yes (`krakenc`) | no | yes |
| **Scooter** | COHERENT_TL, BROADBAND, TIME_SERIES | no (collapsed) | yes | no | yes |
| **RAM** | COHERENT_TL, BROADBAND, TIME_SERIES | bathy + SSP + bottom + layers | yes (`rams`) | yes (`ramsurf`) | yes |
| **SPARC** | COHERENT_TL, TIME_SERIES | no (collapsed) | no | no | (native pulse) |
| **OAST** | COHERENT_TL | no (collapsed) | yes | no | no |
| **OASN** | COVARIANCE, REPLICA | no | yes | no | (multi-freq sweep) |
| **OASR** | REFLECTION | no | yes | no | (multi-freq sweep) |
| **OASP** | COHERENT_TL, BROADBAND, TIME_SERIES | no | yes | no | yes |
| **Bounce** | REFLECTION | no | yes | no | no |

All range-independent models support a layered seabed (`_supports_layered_bottom`)
natively; "Range-dep." above is the horizontal axis only.

### Environment feature support and collapse

Each model declares, per axis of `Environment` shape, whether it handles that
feature natively (`_supports_altimetry`, `_supports_range_dependent_bathymetry`,
`_supports_range_dependent_ssp`, `_supports_range_dependent_bottom`,
`_supports_layered_bottom`, `_supports_elastic_media`). These answer one
question only: *does this env shape work with this model?*

When you hand a model an environment richer than it supports,
`_project_environment` **collapses** the unsupported feature to something the
model can run, emitting one `UserWarning` per dropped feature so nothing
happens silently. For example, Scooter has no range-dependent SSP, so a 2-D SSP
is reduced to one profile (its per-model default: the range-mean).

The reduction is governed by a per-feature policy you can override with
`Model(collapse={...})`:

```python
# Force Scooter to use the near-source SSP instead of the range-mean,
# and the maximum depth for a sloping bottom.
scooter = Scooter(collapse={'ssp': 'r0', 'bathymetry': 'max'})
```

Keys and values: `bathymetry` (`max`/`median`/`mean`/`min`/`initial`), `ssp`
(`r0`/`rmax`/`mean`/`median`), `bottom_range` (`r0`/`rmax`/`mean`/`median`),
`bottom_layers` (`halfspace`/`top_layer`/`volume_average`), `altimetry`
(`drop`), `elastic` (`fluid`/`vacuum`).

---

### Bellhop

Gaussian beam / ray tracing with three interchangeable backends. `backend=None`
(default) auto-selects the fastest installed binary (CUDA → C++ → Fortran); an
explicitly requested backend that isn't installed falls back to Fortran with a
`UserWarning`.

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

bellhop = Bellhop(backend='cxx', n_beams=500, alpha=(-80, 80))
tl   = bellhop.compute_tl(env, source, receiver)            # Field
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

Layered or elastic bottoms auto-route through BOUNCE (a `.brc` reflection table)
unless `Bellhop(auto_bounce=False)`. Range-dependent SSP needs
`Bellhop(interp_ssp='quad')` (the default auto-picks it).

### Kraken

Normal-mode solver. `backend='kraken'` (real) or `'krakenc'` (complex: elastic
media, attenuation, leaky modes); `backend=None` auto-picks `krakenc` when the
env carries shear or `leaky_modes=True`. `field.exe` runs **only** for field
modes — `compute_modes` stops at the modes binary:

```python
from uacpy import Kraken

kraken = Kraken()
modes = kraken.compute_modes(env, source, n_modes=30)   # Modes (k, phi); no receiver
print(modes.k.shape, modes.phi.shape)

tl = kraken.compute_tl(env, source, receiver)           # modes -> field.exe -> Field

# Range-dependent field via coupled modes:
tl_rd = Kraken(mode_coupling='coupled', n_segments=20).compute_tl(env_rd, source, receiver)
```

`run()` defaults to a field mode (`COHERENT_TL`, or `BROADBAND` when a
multi-element `frequencies=` is supplied) — call `compute_modes` for the modes
themselves. Kraken segments range-dependent bathymetry/SSP natively; altimetry
is not supported (only Bellhop is).

### Scooter

Finite-element FFP (wavenumber integration), range-independent. Computes the
exact field — including layered/elastic bottoms and leaky energy — then
Hankel-transforms the Green's function to range.

```python
from uacpy import Scooter

scooter = Scooter()
tl = scooter.compute_tl(env, source, receiver)              # Field
hf = scooter.compute_transfer_function(env, source, receiver,
                                       frequencies=np.linspace(180, 220, 64))
```

`c_low`/`c_high` default to `0.95 × min(SSP)` / `1.05 × max(SSP, bottom)`. The
spectral `RMax` is set from `receiver.range_max × rmax_multiplier` (2.0
narrowband, 3.0 broadband) to keep the FFT-Hankel alias clear of the receivers.

### RAM

Parabolic-equation solver and a **multi-backend dispatcher**: it routes to
`mpiramS` (fluid, flat surface — the broadband-native default), `rams` (elastic
bottom), `ramsurf` (rough surface / altimetry), or `ramgeo` by environment
shape. Inspect the choice with `select_backend(env)`; force one with
`RAM(backend=...)`.

```python
from uacpy import RAM

ram = RAM(np_pade=6, accuracy=1e-1)
print(ram.select_backend(env))                              # e.g. 'mpiramS'
tl = ram.run(env, source, receiver)                         # COHERENT_TL -> Field
```

Broadband works on every backend (the Collins binaries loop in Python, mpiramS
sweeps natively). The band is derived from `(fc, Q, T)`: `bandwidth = fc/Q`,
`Δf = 1/T`. `Q`/`T` default to narrowband `(1e6, 1.0)` for `COHERENT_TL` and
`(2.0, 10.0)` for broadband; pass a multi-element `frequencies=` array to set
the band explicitly:

```python
ts = ram.compute_time_series(env, source, receiver,
                             source_waveform=pulse, sample_rate=fs)
```

### SPARC

Time-domain PE — marches the wave equation in time to produce `p(t)` directly,
without the per-frequency synthesis the other models use. The pulse comes from
the constructor `pulse_type` (a 4-character AT code); `output_mode` selects the
geometry: `'R'` horizontal array, `'D'` vertical array, `'S'` snapshot.

```python
from uacpy import SPARC

sparc = SPARC(output_mode='R', pulse_type='PN+B', n_t_out=512)
ts = sparc.compute_time_series(env, source, receiver)       # Field of p(t)
```

SPARC builds its own pulse, so `source_waveform`/`sample_rate` are ignored (it
warns if you pass them). Range-independent: range-dependent envs collapse.

### Bounce

Reflection-coefficient writer, not a propagation model. It tabulates `R(θ)`
over the phase-velocity band `[c_low, c_high]` and emits a `.brc` (and `.irc`)
file the other models consume. Pin `work_dir` (with `cleanup=False`) so the
files outlive the call:

```python
from uacpy import Bounce

bounce = Bounce(c_low=1400, c_high=10000, work_dir='./bounce_out', cleanup=False)
rc = bounce.compute_reflection(env, source, receiver)       # ReflectionCoefficient
brc_path = rc.metadata['brc_file']

# Chain into another model:
env2 = env.copy()
env2.bottom = uacpy.Bottom.from_halfspace(
    uacpy.BoundaryProperties(acoustic_type='file', reflection_file=brc_path))
tl = Scooter().compute_tl(env2, source, receiver)
```

Bellhop does this transparently via `auto_bounce`; use Bounce directly when you
want control over the table or to feed Scooter/krakenc.

### OASES

OASES is the abstract base for four wavenumber-integration sub-models — pick one
directly, or use the `OASES.for_mode(run_mode=...)` factory. All OASES knobs are
constructor-only.

```python
from uacpy import OAST, OASN, OASR, OASP, OASES, RunMode

# OAST — transmission loss
tl = OAST().run(env, source, receiver)                      # Field

# OASN — array covariance C(f, i, j) and matched-field replicas
oasn = OASN(surface_noise_level=70.0)
cov  = oasn.compute_covariance(env, source, receiver)       # Covariance
rep  = oasn.compute_replicas(env, source, receiver)         # Replicas

# OASR — plane-wave reflection coefficients
refl = OASR(angles=np.linspace(0, 90, 91)).run(env, source, receiver)

# OASP — broadband pulse / transfer function
oasp = OASP(n_time_samples=256, freq_max=120)
hf   = oasp.run(env, source, receiver, run_mode=RunMode.BROADBAND)

# Or dispatch by mode:
model = OASES.for_mode(RunMode.REFLECTION, angles=np.linspace(0, 90, 91))
```

OASES models handle layered and elastic seabeds natively (their strength for
seismo-acoustics), but are range-independent — range-dependent envs collapse.

### Running in parallel

Model runs are independent subprocess computations, so batches are
embarrassingly parallel. Build a `Job` per run — each carries its own model,
scenario, and run mode, so cross-model comparisons and parameter sweeps are the
same operation — and hand the list to `run_parallel`:

```python
from uacpy import Bellhop, Kraken, RAM, run_parallel, Job, RunMode

jobs = [
    Job(Bellhop(), env, source, receiver, run_mode=RunMode.COHERENT_TL, label='bellhop'),
    Job(Kraken(),  env, source, receiver, run_mode=RunMode.COHERENT_TL, label='kraken'),
    Job(RAM(accuracy=1e-1), env, source, receiver, run_mode=RunMode.COHERENT_TL, label='ram'),
]
batch = run_parallel(jobs, n_workers=3)                     # ParallelResult
for label, result in zip(['bellhop', 'kraken', 'ram'], batch):
    print(label, result.tl.min())

# Single-model sweep -> stack into one ResultStack:
sweep = [Job(RAM(accuracy=1e-1).copy(np_pade=p), env, source, receiver, label=p)
         for p in (4, 6, 8)]
stack = run_parallel(sweep).stack(coordinate_name='np_pade')
```

`ParallelResult` is indexable/iterable (`.results`, `.errors`, `.ok`);
`.stack()` bundles successful same-model results into a `ResultStack`. Workers
that own their scratch dir (`cleanup=True`, the default) drop on-disk artifacts —
pin a unique `work_dir` per job with `cleanup=False` to keep them.

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

All of them subclass `Result`, so every result carries the same identification
fields: `result.model`, `result.backend`, `result.source_depths`,
`result.frequencies` (1-D, length-1 for narrowband; `result.f0` is the scalar
shortcut), `result.phase_reference`, and a free-form `result.metadata` dict.

### Field — one container, meaning derived from (dtype, coords)

`Field` is the **single** gridded result type. Its physical meaning is not a
constructor argument — it is *derived* from the dtype of `.data` plus the keys
in `.coords`, and read back through `.kind`:

| `.data` dtype | `.coords` keys | `.kind` | meaning |
|---|---|---|---|
| real | `{depth, range}` | `tl` | transmission loss (dB) |
| complex | `{depth, range}` | `pressure` | narrowband p(d, r) |
| complex | `{depth, range, frequency}` | `transfer_function` | broadband H(d, r, f) |
| real | `{time}` or `{…, time}` | `time_series` | p(t) |
| complex | `{source_depth, depth, range}` | `pressure` | multi-source pressure |

`data.shape` matches the insertion order of `coords` (canonical order
`source_depth → depth → range → frequency|time`).

```python
field.tl          # dB; -20·log10(|data|) for complex, data as-is if already real
field.p           # complex pressure / H(f) (raises if data is real — phase gone)
field.magnitude   # |data|;  field.phase  → angle in rad  (complex only)
field.data        # raw ndarray
field.coords['range']           # axis vectors, always 1-D, metres / Hz / s
field.ranges, field.depths, field.times    # shorthand accessors
field.kind        # 'tl' | 'pressure' | 'transfer_function' | 'time_series'
```

### Slicing — `.at`, `.isel`, `.max` drop axes into `.pinned`

Selecting along an axis **removes** that axis from `coords` and records the
chosen coordinate value in `.pinned`. The result is still a `Field`, so you can
chain and then plot.

```python
H = bellhop.run(env, src, rcv, run_mode=RunMode.BROADBAND, frequencies=freqs)
H.kind                       # 'transfer_function', coords = {depth, range, frequency}

narrow = H.at(frequency=200)        # nearest-label select
narrow.coords                       # {'depth': ..., 'range': ...}
narrow.pinned                       # {'frequency': 199.6}  (nearest sample)
narrow.kind                         # 'pressure'

line = narrow.at(depth=50)          # now coords = {'range': ...}, a 1-D cut
peak = H.max()                      # argmax of |data| → scalar Field, coords = {}
peak.pinned                         # {'depth': 50.0, 'range': 3000.0, 'frequency': 59.6}
```

`.at(**labels)` picks the nearest grid value (never fabricates); `.isel(**indices)`
takes a positional index; `.eval(**labels, method='linear')` *interpolates*
(`'linear'` / `'nearest'` / `'cubic'`) when you want a value between samples;
`.max()` collapses every axis at the global `argmax(|data|)`. Slicing replaces
the old per-shape result subclasses: you slice a `Field` down to 1-D/2-D, then
hand it to one plotter.

This `at` / `isel` / `eval` trio is the **grid-library convention** shared
across the whole API — `Field`, `ResultStack`, `ReflectionCoefficient`, and the
input carriers `SoundSpeedProfile` and `Bottom` all use it (`at` = nearest,
`isel` = positional, `eval` = interpolate). For example `env.ssp.at(depth=50)`
is the nearest stored sample and `env.ssp.eval(depth=50)` interpolates.

### Metadata and output-file paths

`result.metadata` holds model-specific extras (`c0`, `Q`, `dr`, …). When you
construct a model with a **pinned** `work_dir`, the solver's on-disk outputs are
kept and their paths are recorded there:

```python
bellhop = Bellhop(work_dir='/tmp/run1')
r = bellhop.run(env, src, rcv)
r.metadata['shd_file']      # '/tmp/run1/.../FIELD.shd'
r.metadata['prt_file']      # diagnostic .prt log
r.list_metadata()['c0']     # {'value_type': 'float', 'documented_type': 'float',
                            #  'description': 'Reference sound speed (m/s) ...'}
```

`result.list_metadata()` describes every key currently attached (runtime type,
documented type, one-line meaning) so you do not have to grep the source.

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
inverting — `time_domain_native` fields refuse `to_time_trace` and tell you to
read the native trace instead.

### Broadband → time series

Two paths, both starting from a broadband `Field` (`coords = {depth, range,
frequency}`):

**(a) Helpers (recommended).** They honour `phase_reference`, resolve bin
resolution internally, and land the arrival at the right time even on a coarse
frequency grid.

```python
# one trace at a chosen cell → time-domain Field, coords = {'time': ...}
trace = H.to_time_trace(depth=50, range=5000)

# convolve every cell with a source waveform → Field, coords = {depth, range, time}
ts = H.synthesize_time_series(source_waveform=p_src, sample_rate=fs)
p = ts.at(depth=50, range=5000).data        # the received pulse
t = ts.times                                # seconds
```

A `UserWarning` fires if `1/Δf` (the DFT period) is shorter than the waveform
duration — refine the frequency grid so the late response does not wrap.

**(b) Manual `2·Re[ifft(H)]`** for `travelling_wave` `H(f)`, when you want full
control. Place each model frequency at bin `round(f/Δf)`; the window length
`1/Δf` must exceed the arrival time `r/c0`, so the grid must be fine enough:

```python
import numpy as np
f = H.coords['frequency']
spec1d = H.at(depth=50, range=5000).data       # complex H(f) at the cell
df = f[1] - f[0]                                # need 1/df > r/c0
nfft = 1 << int(np.ceil(np.log2(2 * round(f[-1] / df) + 2)))
buf = np.zeros(nfft, complex)
buf[np.round(f / df).astype(int)] = spec1d
pt = 2.0 * np.real(np.fft.ifft(buf)) * (nfft * df)   # Riemann sum of ∫H df
t = np.arange(nfft) / (nfft * df)
#  → peak of pt sits at t ≈ r/c0
```

For a unit-amplitude phase-only `H = exp(-i 2π f r/c0)` with `r = 3000 m`,
`c0 = 1500 m/s`, this lands the impulse exactly at `t = 2.0 s`.

### Filtering helpers (sparse results)

The non-grid results return new copies — no solver re-run:

```python
arr.filter_by_bounces(kind='surface')        # Arrivals: by multipath class
arr.in_delay_window(t_min=2.0).top_n_by_amplitude(5)
rays.filter_by_launch_angle(-10, 10)         # Rays: pure data subsets
rays.top_n_by_miss(20, target_range_m=5000, target_depth_m=50)
modes.first_n(10).compute_phase_speeds()     # Modes: trim + derive v_p
rc.at(angle=30, frequency=200)               # ReflectionCoefficient: nearest sample
```

### ResultStack

A `ResultStack` is a sequence of same-typed slabs that share every axis except a
stacking coordinate. You get one two ways:

* a **single run with a multi-source-depth `Source`** on a model that supports it
  — `Bellhop` returns a `ResultStack` stacked over `source_depth`;
* `run_parallel(...).stack(coordinate_name=...)` over a parameter sweep.

Models that don't support multiple source depths (e.g. `Kraken`) raise a
`ConfigurationError` for a multi-depth `Source` — loop one `Source` per depth.

```python
for src_depth, slab in stack: ...     # iterate (coordinate, slab) pairs
stack.at(source_depth=20)             # nearest-label slab → a Field
stack.tl                              # stacked TL, shape (n_slabs, *slab.tl.shape)
stack.plot()                          # panel grid (Field slabs)
```

Index a single slab with `stack[i]` or `stack.at(source_depth=…)` (each is a
`Field`).

## 9. Visualization

A small, composable helper set. The convention is uniform: **the result is the
positional argument; `env=` adds overlays.** All are exposed at top level
(`uacpy.plot_field`, `uacpy.plot_result`, …) and under `uacpy.plot.*`.

| Function | Use |
|---|---|
| `plot_result(result, env=…)` | type-dispatch — calls the right plotter for any result |
| `plot_field(field, env=…)` | auto-shape: 1 surviving axis → line, 2 → heatmap |
| `result.plot()` | shorthand for `plot_result(result)` |
| `plot.compare(fields, labels)` | overlay several 1-D sliced fields on one axes (`uacpy.plot.compare`) |
| `compare_models(fields, labels, env=…)` | side-by-side heatmaps, one shared colourbar |
| `plot_environment(env)` | SSP + seafloor cross-section, optional source/receiver markers |
| `plot_ssp(env_or_ssp)` | sound-speed profile `c(z)` as a depth-down line (one per range if range-dependent) |
| `plot_overview(env, map_args, tl=…)` | three-panel map + TL + environment composite |

**DSP / comms plotters.** All signal-processing and communications plotting also
lives here (the `acoustic_signal` and `comms` modules are pure computation and
import no matplotlib). Each takes the arrays a transform/estimator returns, the
target `ax` as its second positional argument, and returns `(fig, ax)` — the same
convention as `plot_field`:

- **Spectra / levels:** `plot_psd`, `plot_ppsd`, `plot_sel`, `plot_spectrogram`,
  `plot_band_levels`.
- **Gather transforms:** `plot_fk`, `plot_radon`, `plot_taup` (+ `draw_sound_cone`,
  `draw_slowness_line` overlays).
- **Time-frequency:** `plot_cwt`, `plot_wigner_ville`, `plot_cepstrum`.
- **Arrays / active / system-ID:** `plot_angular_spectrum`, `plot_ambiguity`,
  `plot_frf`, `plot_coherence`, `plot_impulse_response_info`.
- **Comms:** `plot_channel`, `plot_constellation`, `plot_scatter`,
  `plot_eye_diagram`, `plot_ber_curve`, `plot_convergence`, `plot_sync_metric`,
  `plot_doppler_ambiguity`, `plot_subcarriers`.

Slicing replaces specialised plotters: there is no separate "TL line" vs "TL
heatmap" function — you slice a `Field` to 1 or 2 axes and `plot_field`
auto-selects the shape. Plotting a field with 3+ surviving axes raises and tells
you to slice first.

**Fixed TL colour scale.** Every TL heatmap defaults to `vmin=20`, `vmax=120` dB
(`_TL_LIMITS`) so figures are directly comparable across runs and models. Pass
explicit `vmin=`/`vmax=` to override.

### (a) Single TL field with seafloor + contours

```python
import uacpy
tl = bellhop.compute_tl(env, src, rcv)     # Field, coords = {depth, range}; .tl → dB
fig, ax = uacpy.plot_field(
    tl, env=env,            # env= overlays the seafloor on the (depth, range) heatmap
    contours=[60, 80, 100], # dB contour lines
)
```

`plot_field` puts range (km) on X, depth (down) on Y, applies the TL colormap on
the fixed 20–120 dB scale, and draws the seafloor from `env`. Slice first to get
a line cut instead:

```python
uacpy.plot_field(tl.at(depth=50))          # TL vs range, 1-D
uacpy.plot_field(tl.at(range=5000))        # TL vs depth, depth down
```

### (b) Multi-model comparison, one shared colourbar

```python
fields = {
    'Bellhop': bellhop.compute_tl(env, src, rcv),
    'Kraken':  kraken.compute_tl(env, src, rcv),
    'RAM':     ram.compute_tl(env, src, rcv),
}
fig, axes = uacpy.compare_models(fields, env=env, contours=[80, 100])
```

All panels share the fixed 20–120 dB scale and a single right-hand colourbar, so
differences between models read directly. `compare_models` warns if the depth or
range axes differ between fields (a shared colourbar would mix sample grids). For
1-D overlays (e.g. TL vs range from several models at one depth) slice each field
and use `uacpy.plot.compare([f.at(depth=50) for f in ...], labels=[...])`.

## 10. Signal Processing

A processing toolbox for the waveforms that propagation models emit or
consume. Named `acoustic_signal` so it never shadows the stdlib `signal`.
Most transforms ship as **both** a CapWords class (a reusable, configured
estimator) and a lower-case function (one-shot on a single array) — pick by
use.

| Area | Public names |
|------|--------------|
| Waveforms | `lfm_chirp`, `hfm_chirp`, `tone_burst`, `gaussian_pulse`, `ricker_wavelet`, `sparc_pulse`, `nwave` |
| Coded probes | `mseq`, `make_mseq_probe`, `bpsk_modulate` |
| Noise synthesis | `make_noise_waveform`, `make_bandlimited_noise`, `synthesize_noise_from_psd`, `fourier_synthesis`, `add_noise` |
| Spectral / levels | `psd`, `ppsd` (→ `PPSDResult`), `sel` |
| Decidecade (ISO 18405) | `decidecade_bands`, `decidecade_band_levels` |
| Arrays | `steering_vectors`, `beamform`, `sample_covariance`, `bartlett_spectrum`, `mvdr_spectrum`, `music_spectrum`, `shading_taper` |
| Active / pulse compression | `matched_filter`, `pulse_compression`, `processing_gain`, `ambiguity_function` |
| Time-frequency | `spectrogram`, `analytic_signal`, `envelope`, `instantaneous_frequency`, `wigner_ville`, `cwt` |
| Gather transforms | `fk_transform`, `taup_transform`, `radon_transform` (+ `inverse_*`) |
| System ID / channel | `FRF`, `impulse_response`, `simulate_reception` |
| Modal / dispersion | `warp_signal`, `unwarp_signal`, `modal_group_velocity` |

Waveform builders return `(signal, time)`; lengths follow `duration ×
sample_rate`. Build a chirp, push it through a delay, and pulse-compress:

```python
import numpy as np
from uacpy.acoustic_signal import lfm_chirp, matched_filter

sig, t = lfm_chirp(fmin=2_000, fmax=6_000, T=0.02, sample_rate=48_000)
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
| Sonar equation | `passive_signal_excess`, `active_signal_excess`, `echo_level`, `figure_of_merit`, `noise_background`, `detection_range`, `detection_range_by_depth` |
| Field maps | `passive_signal_excess_field`, `active_signal_excess_field`, `probability_of_detection_field` |
| Detection theory | `albersheim_snr`, `probability_of_detection`, `roc_curve`, `detection_index`, `deflection_coefficient`, `detection_threshold_energy` |
| Target strength | `ts_sphere`, `ts_cylinder`, `ts_plate`, `ts_ellipsoid`, `ts_convex` |
| Scattering / reverb | `lambert_bottom`, `chapman_harris_surface`, `column_scattering_strength`, `boundary_reverberation`, `volume_reverberation`, `total_reverberation` |

```python
from uacpy.sonar import figure_of_merit, albersheim_snr

dt  = albersheim_snr(pd=0.5, pf=1e-4)         # required SNR (dB) for Pd/Pfa
fom = figure_of_merit(source_level=180, noise_level=60,
                      directivity_index=15, detection_threshold=dt)
# fom is the max allowable one-way TL — cross it with a TL field for range.
```

See example 27 (sonar equation, reverberation, detection-range maps).

## 12. Digital Communications

`uacpy.comms` — a coherent underwater link in two tiers: Tier 1 is the essential chain
(modulation, fading channels, DFE/LMS/RLS equalization with an optional
carrier PLL, Doppler estimation, framing, BER/EVM); Tier 2 adds OFDM,
convolutional+Viterbi FEC, DSSS, and NATO **JANUS** (STANAG 4748,
verified bit-exact against CMRE janus-c).

| Area | Public names |
|------|--------------|
| Modulation | `Modulator`, `constellation`, `dpsk_modulate`, `fsk_modulate` |
| Channel | `awgn`, `multipath_channel`, `apply_fading_channel`, `fading_taps` |
| Equalization | `DFE`, `lms_equalizer`, `rls_equalizer`, `mmse_equalizer` |
| Doppler / sync | `estimate_doppler_scale`, `compensate_doppler`, `detect_preamble`, `detect_frames` |
| Link harness | `simulate_link`, `ber_sweep`, `LinkResult` |
| Metrics | `bit_error_rate`, `symbol_error_rate`, `evm`, `ber_theory`, `eye_diagram` |
| Coding / spread | `ConvCode`, `conv_encode`, `viterbi_decode`, `interleave`, `spread`, `despread` |
| OFDM | `ofdm_modulate`, `ofdm_demodulate`, `schmidl_cox_sync`, `OFDMTransmitter`, `OFDMReceiver` |
| JANUS | `janus_encode`, `janus_decode`, `janus_modulate`, `janus_detect`, `JanusPacket` |

`simulate_link` composes transmit → channel → receive and measures BER;
`ber_theory` gives the AWGN bound for the same scheme:

```python
from uacpy.comms import simulate_link, ber_theory

for ebn0 in (0, 4, 8, 12):
    res = simulate_link("qpsk", ebn0_db=ebn0, n_bits=20_000)
    print(ebn0, res.ber, ber_theory("qpsk", ebn0))
```

See examples 31 (comms tour), 32 (text→wav modem), 33 (OFDM), 34 (JANUS beacon).

## 13. Ambient Noise

`uacpy.noise` — Wenz-style spectra plus ship radiated-noise (ISO 17208) and marine-mammal
auditory weighting (Southall 2019). Spectra are in dB re 1 µPa²/Hz.

| Area | Public names |
|------|--------------|
| Wenz spectrum | `WenzNoise`, `compute_windnoise` |
| Ship radiated noise (ISO 17208) | `radiated_noise_level`, `monopole_source_level`, `nominal_source_depth`, `lloyd_mirror_correction`, `plot_source_level` |
| Mammal weighting (Southall 2019) | `auditory_weighting`, `apply_weighting`, `weighted_level`, `plot_weighting`, `HEARING_GROUPS`, `WEIGHTING_PARAMS` |

```python
import numpy as np
from uacpy.noise import WenzNoise

f = np.logspace(0, 5, 500)
wenz = WenzNoise(f, wind_speed=15, water_depth="deep",
                shipping_level="medium", rain_rate="moderate")
psd = wenz.as_psd(ref=1)        # linear Pa²/Hz; wenz.plot() for dB spectrum
```

Hearing groups: `LF, HF, VHF, SI, PCW, OCW, PCA, OCA`. See examples 9, 35, 36.

## 14. Standards-Based Metrics

These standards-grounded helpers live in their natural packages rather than a
single module — each is sourced to its standard:

- **`uacpy.core.acoustics.soundspeed_unesco`** — seawater sound speed, UNESCO
  (Chen & Millero 1977 / UNESCO 1983), `c(T, S, pressure)`.
- **`uacpy.core.acoustics.soundspeed_delgrosso`** — Del Grosso 1974 alternative,
  preferred at high pressure / deep water.
- **`uacpy.acoustic_signal.decidecade_bands`** / `decidecade_band_levels` —
  one-third-octave (decidecade) bands, ISO 18405 / IEC 61260-1.
- **`uacpy.noise.monopole_source_level`** / `radiated_noise_level` — ship
  monopole source level, ISO 17208.
- **`uacpy.noise.auditory_weighting`** — marine-mammal frequency weighting,
  Southall et al. 2019.

`uacpy.metrics` itself holds the cross-model TL-agreement helpers
(`tl_rmse`, `tl_max_error`, `tl_bias`), each taking two 2-D `Field`s.

```python
from uacpy.core.acoustics import soundspeed_unesco, soundspeed_delgrosso

soundspeed_unesco(15, 35, 0)      # 1506.675 m/s  (T °C, S PSU, pressure dbar)
soundspeed_delgrosso(15, 35, 0)   # 1506.667 m/s
```

Example 35 chains site sound speed → decidecade bands → ship SL → weighted level.

## 15. Units & Conventions

uacpy is SI throughout; underwater levels reference **1 µPa**.

### Units

| Quantity | Unit | Note |
|----------|------|------|
| Length / depth / range | metres | suffix `_km` only when kilometres (`m_to_km` at IO boundaries) |
| Frequency | Hz | |
| Time | seconds | |
| Angle | degrees | launch angles, grazing/incidence angles, wedge angles (radians only inside formulas) |
| Sound speed | m/s | |
| Density | g/cm³ | **acoustic inputs** (bottom/sediment). SI `kg/m³` only for the intensity constant — see *Density* below |
| Attenuation (geoacoustic) | dB per wavelength | models emit the matching `AT` TopOpt letter |
| Attenuation (volume) | dB/km | `francois_garrison_db_per_km`, `thorp_db_per_km` |
| Pressure | Pa (µPa for levels) | |
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
  a unit point source at 1 m** — i.e. referenced to the free-field
  `p₀(r) = e^{i k₀ r}/(4π r)`. Hence transmission loss is simply
  **`TL = −20·log₁₀|p|`** (`Field.tl`), in dB re 1 m. Real `data` is already in
  dB and returned as-is.
- This is the 3-D point-source convention. A 2-D (line-source) analytic
  solution differs by the spreading factor `|G₃D/G₂D| = √(k/2πr)` — relevant
  only when comparing against closed-form 2-D references (see the ideal-wedge
  benchmark in `tests/test_benchmarks_analytic.py`).
- `sel`, `psd`, `ppsd` and `spectrogram` are pure functions returning arrays;
  their plots (`plot_sel`, `plot_psd`, `plot_ppsd`, `plot_spectrogram`, and the
  gather/transform/comms plotters) live in `uacpy.visualization`, not in the
  computation modules. Levels are formed through
  `core.acoustics.power_to_db` (`10·log₁₀(power/ref²)`), which floors `power` at
  `PRESSURE_FLOOR` before the log so a silent sample yields a finite, very
  negative level rather than `−inf`.

### Phase / time convention

- Harmonic time dependence and the sign of the imaginary part **follow the
  underlying Acoustics-Toolbox binaries** (KRAKEN, e.g., is implemented with the
  `e^{+iωt}` convention). Magnitudes and TL are convention-independent; **phase**
  is not — this matters for matched-field processing and for broadband IFFT,
  where the `phase_reference` carried on a `Result` fixes the reference (see §8,
  *Phase reference — why IFFT needs it*).
- Broadband bands are defined per model: RAM uses `bandwidth = fc/Q`, `Δf = 1/T`;
  Bellhop synthesizes `n_freqs` bins over `bandwidth_factor·fc` (§18).

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

- Sound-speed helpers: **`soundspeed` = Mackenzie (1981)** (T °C, S ppt, depth m)
  is the default; **`soundspeed_unesco` = Chen & Millero / UNESCO** (T °C, S PSU,
  **pressure in dbar**) is the international-standard algorithm.
- **Density has two distinct roles** — do not conflate them:
  - *Acoustic input* density (bottom/sediment, and the water column on disk) is
    **g/cm³**; when a water density is not written, the AT binaries use their
    default **ρ_water = 1.0 g/cm³** (this is what the propagation models see).
  - `constants.DENSITY_SEAWATER = 1027 kg/m³` is used **only** for SI
    particle-velocity / acoustic-intensity physics, never as a model input.

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

## 16. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `binary not found` / model won't run | native binaries aren't built — run `./install.sh -y` (lands in `uacpy/bin/`, gitignored); pick a backend with `--bellhop fortran\|cxx\|cuda`. |
| `UnsupportedFeatureError` | the model can't honour that `RunMode` or env axis. Check `model.supports_mode(...)`; unsupported env *shapes* are reduced by `_project_environment()` (one `UserWarning` per dropped feature) — override the policy with `Model(collapse={...})`. |
| TL is `inf`/`NaN` at the source | `r = 0` and very-near-field cells are sentinel/NaN by construction (TL undefined at zero range); start receiver ranges past the first cell. |
| OASES tests skipped / `requires_oases` | OASES is academic-licensed and not bundled; fetch it via `install.sh --oases yes`. Run only the rest with `pytest -m "not requires_oases"`. |
| CUDA backend silently slow | driver/toolkit mismatch falls back to Fortran with a warning — check the emitted backend; pin with `Bellhop(backend="fortran")`. |
| Wrong Python / missing deps | use the workspace venv (`../pyenv/bin/python`), not system Python; `pip install -e ".[dev]"`. |

## 17. Examples Index

All 37 runnable scripts live in `uacpy/examples/`.

| # | Topic |
|---|-------|
| 01 | Basic shallow-water propagation — Pekeris waveguide |
| 02 | Sound-speed profiles — Munk, Pekeris, thermocline |
| 03 | Multi-frequency / broadband propagation |
| 04 | Bellhop advanced — all-features showcase |
| 05 | RAM (mpiramS) — range-dependent SSP and bottom |
| 06 | Kraken / KrakenC — coupled-mode theory |
| 07 | All models — comprehensive comparison |
| 08 | Deep-water long range — 100+ km propagation |
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
| 19 | Broadband — time series and transfer functions across models |
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
| 31 | Underwater acoustic communications tour |
| 32 | Real-data underwater modem (text → .wav → text) |
| 33 | OFDM underwater modem (text → .wav → text) |
| 34 | JANUS standard beacon (NATO STANAG 4748) |
| 35 | Underwater noise impact assessment — standards chain |
| 36 | Modeled noise impact — ship SL through a real TL field |
| 37 | Real-world environment — map · transmission loss · section |

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
| `timeout` | s | `600.0` | Per-subprocess wall-clock limit (SPARC default `180.0`). |
| `collapse` | dict | `None` | Per-feature range/layer collapse-policy overrides (see §7, "Environment feature support and collapse"). |

### `run()` call arguments (every model)

Passed at call time, not construction — the fixed no-`**kwargs` signature (§4):

| Argument | Unit | Default | Meaning |
|---|---|---|---|
| `run_mode` | `RunMode` | `None` | Which product to compute; `None` → the model's default mode. |
| `frequencies` | Hz | `None` | Explicit frequency or band; a multi-element array selects a broadband run. |
| `source_waveform` | array | `None` | Source pressure pulse for `TIME_SERIES` synthesis. |
| `sample_rate` | Hz | `None` | Sample rate of `source_waveform` / the synthesised output. |
| `output_duration` | s | `None` | Time-series window length, overriding the auto/constructor value. |
| `n_modes` *(Kraken only)* | count | — | Number of modes to retain. |

### Source / Receiver

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `Source.depths` | m | *required* | Source depth(s), positive down. |
| `Source.frequencies` | Hz | *required* | Source frequency or frequencies. |
| `Source.source_type` | — | `'point'` | `'point'` or `'line'` (physical line-source geometry). |
| `Receiver.depths` | m | *required* | Receiver depth(s), positive down. |
| `Receiver.ranges` | m | `None` | Receiver range(s); `None` → a single point at 0 m. |
| `Receiver.receiver_type` | — | `'grid'` | `'grid'` (depth×range cross-product) or `'line'` (depths/ranges paired point-by-point, `len(depths)==len(ranges)`). |

*(Environment and its carriers — bathymetry, SSP, bottom, surface, altimetry — are documented in §5; their units follow §15: metres / m/s / g/cm³ / dB-per-λ.)*

### Bellhop

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `backend` | — | `None` | Binary variant: `'fortran'`/`'cxx'`/`'cuda'`; `None` auto-selects CUDA > C++ > Fortran. |
| `dimensionality` | — | `'2D'` | Only `'2D'` is supported; `'3D'` raises. |
| `beam_type` | — | `'B'` | `B` Gaussian, `R` ray-centered, `C` Cartesian, `b` geometric Gaussian, `g` geometric hat, `G` geometric hat Cartesian, `S` simple Gaussian. |
| `n_beams` | count | `0` | Number of beams; `0` defers to Bellhop's auto-selection. |
| `alpha` | deg | `(-80, 80)` | Launch-angle limits `(min, max)`. |
| `step` | m | `0.0` | Ray step size; `0` = automatic. |
| `z_box` | m | `None` | Max depth of the ray box; `None` = 1.2 × max depth. |
| `r_box` | m | `None` | Max range of the ray box; `None` = 1.2 × max range. |
| `source_type` | — | `'R'` | `'R'` point (cylindrical), `'X'` line (Cartesian). |
| `grid_type` | — | `'R'` | Receiver grid: `'R'` rectilinear, `'I'` irregular. |
| `interp_ssp` | — | `None` | SSP scheme; `None` auto (`'quad'` if RD-SSP else `'linear'`); also `'linear'`/`'pchip'`/`'cubic'`/`'quad'`/`'n2linear'`/`'analytic'`. |
| `interp_bathymetry` | — | `'linear'` | `.bty` interpolation: `'linear'` or `'curvilinear'`. |
| `interp_altimetry` | — | `'linear'` | `.ati` interpolation: `'linear'` or `'curvilinear'`. |
| `source_beam_pattern_file` | path/array | `None` | `.sbp` path or `(angle_deg, level_dB)` pairs; `None` = omnidirectional. |
| `arrivals_format` | — | `'ascii'` | `ARRIVALS` output format. Use `'ascii'`; `'binary'` (`'a'`, Fortran unformatted) is accepted at construction but **not yet parseable** — `compute_arrivals` raises `ConfigurationError` on read. |
| `beam_width_type` | — | `'F'` | Cerveny width: `'F'` filling, `'M'` match, `'W'` waveguide (used for `beam_type` ∈ C/R). |
| `beam_curvature` | — | `'D'` | `'D'` double, `'S'` single, `'Z'` zero. |
| `eps_multiplier` | factor | `1.0` | Beam-width epsilon multiplier. |
| `r_loop` | m | `1000.0` | Range at which the beam width is chosen. |
| `n_image` | count | `1` | Number of images. |
| `ib_win` | — | `4` | Beam-windowing parameter. |
| `component` | — | `'P'` | Displacement-receiver output: `'P'` pressure, `'D'` displacement. |
| `beam_shift` | — | `False` | Enable beam-shift on boundary reflections. |
| `n_freqs` | count | `128` | Frequency bins for BROADBAND/TIME_SERIES synthesis from a single centre frequency. |
| `bandwidth_factor` | factor | `0.5` | Fractional bandwidth of the synthesised band around the centre frequency. |
| `time_window` | s | `None` | TIME_SERIES window length; `None` auto-derives. |
| `t_start` | s | `None` | TIME_SERIES start time; `None` auto-derives. |
| `auto_bounce` | — | `True` | Auto-route layered/elastic bottoms through BOUNCE; `False` collapses to fluid (one warning). |

### Kraken

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `backend` | — | `None` | Modes binary: `'kraken'`/`'krakenc'`; `None` auto-picks `krakenc` for elastic/leaky media. |
| `mode_coupling` | — | `'adiabatic'` | RD mode transitions: `'adiabatic'` or `'coupled'`. |
| `coherent` | — | `True` | Coherent mode addition; `'coupled'` + `coherent=False` is rejected. |
| `n_segments` | count | `None` | Range segments for RD; `None` auto-picks from change-points (gaps > 2 km split). |
| `mode_points_per_meter` | pts/m | `1.5` | Mode-depth grid density. |
| `source_type` | — | `'R'` | `field.exe` Opt(1): `'R'` cylindrical, `'X'` Cartesian line, `'S'` scaled-cylindrical. |
| `source_beam_pattern_file` | path | `None` | `.sbp` file; sets `field.exe` Opt(3)=`'*'`. |
| `field_executable` | path | `None` | `field.exe` path; auto-detected if `None`. |
| `c_low` | m/s | `None` | Lower phase-speed limit of the modal solver. |
| `c_high` | m/s | `None` | Upper phase-speed limit. |
| `n_mesh` | count | `0` | Mesh points per medium; `0` = auto. |
| `roughness` | m | `0.0` | Bottom RMS roughness. |
| `interp_ssp` | — | `None` | SSP interpolation scheme (as Bellhop). |
| `leaky_modes` | — | `False` | Include leaky modes (forces `krakenc`). |
| `top_reflection_file` | path | `None` | Top-boundary reflection-coefficient file. |
| `rmax_m` | m | `None` | Max range for `field.exe`. |
| `mode_depth_grid` | array | `None` | Explicit mode-depth output grid. |

### Scooter

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `c_low` | m/s | `None` | Lower phase-speed limit; `None` = 0.95 × min SSP. |
| `c_high` | m/s | `None` | Upper phase-speed limit; `None` = 1.05 × max(SSP, bottom). |
| `n_mesh` | count | `0` | Total FE mesh points **per medium** (AT `NG`); `0` = auto. Not points-per-wavelength. |
| `roughness` | m | `0.0` | Bottom RMS roughness. |
| `rmax_multiplier` | factor | `None` | Wavenumber-resolution range multiplier; `None` → 2.0 narrowband / 3.0 broadband. |
| `interp_ssp` | — | `None` | SSP interpolation scheme. |
| `source_type` | — | `'R'` | FLP Opt(1): `'R'` cylindrical, `'X'` Cartesian. |
| `spectrum` | — | `'positive'` | FLP Opt(2): `'positive'`/`'negative'`/`'both'` wavenumber spectrum. |
| `stabilizing_attenuation_off` | — | `False` | Disable Scooter's stabilising attenuation (TopOpt pos 7 = `'0'`). |
| `field_interp` | — | `'O'` | FLP Opt(3): `'O'` polynomial, `'P'` Padé. |

### SPARC

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `c_low` | m/s | `None` | Lower phase-speed limit; `None` = auto. |
| `c_high` | m/s | `None` | Upper phase-speed limit; `None` = auto. |
| `n_mesh` | count | `0` | Mesh points per wavelength; `0` = auto. |
| `roughness` | m | `0.0` | Bottom RMS roughness. |
| `interp_ssp` | — | `None` | SSP interpolation scheme. |
| `output_mode` | — | `'R'` | `'R'` horizontal array, `'D'` vertical array, `'S'` snapshot. **`'S'` is experimental: its absolute TL is uncalibrated (~30 dB offset vs Kraken/Scooter; emits a `UserWarning`) — use `'R'` or another model for absolute levels.** |
| `pulse_type` | — | `'PN+B'` | AT 4-character pulse-type code. |
| `n_t_out` | count | `512` | Number of output time samples. |
| `t_max` | s | `None` | Max time; `None` = auto (2.5 × travel time). |
| `t_start` | s | `-0.1` | Integration start time. |
| `t_mult` | factor | `0.999` | Integration time multiplier. |
| `max_depths` | count | `20` | Max number of depths before a warning. |
| `rmax_safety_margin` | factor | `None` | RMax multiplier on receiver max range; `None` → 1.0001 (COHERENT_TL) / 3.0 (TIME_SERIES). |
| `f_min` | Hz | `None` | Pulse-band lower edge; `None` → one octave around source freq. |
| `f_max` | Hz | `None` | Pulse-band upper edge; `None` → one octave around source freq. |
| `sound_speed` | m/s | `None` | Reference speed for the travel-time window when `t_max` is auto; `None` → default. |

### RAM

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `backend` | — | `None` | RAM-family backend: `'mpiramS'`/`'ramgeo'`/`'rams'`/`'ramsurf'`; `None` auto-dispatches by env shape. |
| `dr` | m | `None` | Range step; `None` → Lytaev optimiser. |
| `dz` | m | `None` | Depth step; `None` → Lytaev optimiser (snapped to integer depth count). |
| `zmax` | m | `None` | PE domain depth; `None` = seafloor + absorbing layer. |
| `np_pade` | count | `6` | Number of Padé coefficients (2–8). |
| `ns_stability` | count | `1` | Number of stability terms. |
| `rs_stability` | m | `None` | Stability range; `None` = max output range. |
| `Q` | — | `None` | Broadband bandwidth factor (bandwidth = fc/Q); `None` → 2.0 broadband / 1e6 COHERENT_TL. |
| `T` | s | `None` | Time-window width; `None` → 10.0 broadband / 1.0 COHERENT_TL. |
| `depth_decimation` | factor | `1` | Output depth-decimation factor. |
| `flat_earth` | — | `True` | Apply the flat-earth transformation. |
| `absorbing_layer_width` | λ | `20.0` | Absorbing-layer width below the seafloor (in wavelengths). |
| `absorbing_layer_attn` | dB/λ | `10.0` | Attenuation at the absorbing-layer floor (ramped from sediment attn). |
| `n_sed_points` | count | `50` | Sediment depth control points for the mpiramS profile. |
| `c0` | m/s | `None` | PE reference (expansion) speed — algorithmic, not physical; `None` → Lytaev Eq. (15). |
| `accuracy` | — | `0.001` | Lytaev optimiser accuracy budget (max \|τ·n_steps\|). |
| `theta_max` | deg | `30.0` | Max propagation angle bounding the PE spectrum (Lytaev). |
| `rams_theta` | deg | `45.0` | `rams` backend rotated-Padé angle (Milinazzo-Zala-Brooke). |
| `rams_irot` | — | `1` | `rams` rotation flag. |
| `rams_dr_safety_factor` | factor | `5.0` | Tightening factor on the Lytaev `dr` for the `rams` backend (1.0 disables). |

### Bounce

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `c_low` | m/s | `1400.0` | Min phase velocity for tabulation (must be > 0). |
| `c_high` | m/s | `10000.0` | Max phase velocity (`1e9` ≈ full 90° coverage; must be > `c_low`). |
| `rmax` | m | `None` | Max range for angular sampling; `None` = 10000. Ignored when `n_angles` is set. |
| `n_angles` | count | `None` | Explicit number of angular samples; `None` = bounce computes it from `rmax`. |
| `interp_ssp` | — | `None` | SSP interpolation scheme. |

### OASES — OAST (transmission loss)

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `compute_contour` | — | `False` | Add `'C'` option (range-depth contour plot). |
| `compute_depth_average` | — | `False` | Add `'A'` option (depth-averaged TL). |
| `complex_contour` | — | `True` | `'J'` option (complex integration contour). |
| `options` | — | `None` | Raw OASES options string; `None` derives it from the three flags above. |
| `integration_offset` | dB/λ | `0.0` | Wavenumber-integration contour offset. |
| `nw_samples` | count | `-1` | Number of wavenumber samples; `-1` = OASES auto. |
| `plot_rmin` | m | `None` | TL plot range-axis min; `None` → 0. |
| `plot_rmax` | m | `None` | TL plot range-axis max; `None` → `receiver.range_max`. |

### OASES — OASP (broadband pulse / transfer function)

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `n_time_samples` | count | `4096` | Power-of-two FFT length (samples per receiver trace). |
| `freq_max` | Hz | `None` | Sweep upper edge; `None` → 2.5 × centre frequency. |
| `freq_min` | Hz | `0.0` | Sweep lower edge. |
| `center_frequency` | Hz | `None` | Pulse carrier frequency; `None` → `source.frequencies[0]`. |
| `freq_output_increment` | factor | `None` | `.trf` frequency-axis decimation; `None` → `max(1, n_freq // 10)`. |
| `options` | — | `None` | Raw OASES options string; `None` → `'N J'`. |
| `range_start` | m | `None` | First receiver range; `None` → `receiver.ranges.min()`. |
| `integration_offset` | dB/λ | `0.0` | Wavenumber-contour offset. |
| `nw_samples` | count | `-1` | Wavenumber sample count; `-1` = auto. |

### OASES — OASR (reflection coefficients)

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `angles` | deg | `None` | Angle grid; `None` → `linspace(0, 90, 181)`. |
| `angle_type` | — | `'grazing'` | `'grazing'` (native) or `'incidence'` (grazing = 90 − incidence). |
| `reflection_type` | — | `'P-P'` | `'P-P'`, `'P-SV'`, `'P-Slow'` (Biot), or `'transmission'`. |
| `options` | — | `None` | Raw OASES options string; `None` derives from `reflection_type`. |
| `angle_output_increment` | factor | `None` | Output angle-table decimation; `None` keeps every sample. |
| `interface_roughness` | m | `None` | Per-interface RMS roughness (top → bottom). |

### OASES — OASN (noise covariance / replicas)

| Parameter | Unit | Default | Meaning |
|---|---|---|---|
| `options` | — | `None` | OASES options string; `None` → `'N'` (covariance) / `'R'` (replica). |
| `surface_noise_level` | dB re 1 µPa²/Hz | `0.0` | Surface-generated noise spectral level (0 disables). |
| `white_noise_level` | dB re 1 µPa²/Hz | `0.0` | Uncorrelated per-hydrophone white-noise level (0 disables). |
| `deep_noise_level` | dB re 1 µPa²/Hz | `0.0` | Deep broad-area source spectral level (0 disables). |
| `deep_source_depth` | m | `None` | Depth of the deep noise sheet; `None` → half the water depth. |
| `discrete_sources` | list | `None` | Point sources: dicts with `depth`/`x`/`y` (m), `level` (dB), `phase` (rad). |
| `xmin` / `xmax` | m | `None` | Replica grid x-bounds; `None` → 100 / 10000. |
| `nx` | count | `50` | Replica grid points in x. |
| `ymin` / `ymax` | m | `None` | Replica grid y-bounds; `None` → 0 / 0. |
| `ny` | count | `1` | Replica grid points in y. |
| `zmin` / `zmax` | m | `None` | Replica grid depth-bounds; `None` → 10 / `env.depth − 10`. |
| `nz` | count | `20` | Replica grid points in depth. |
| `cmin` / `cmax` | m/s | `None` | Phase-speed bounds for the wavenumber integrations; `None` → 0.95 × min(c_water) / 1e8. |
| `integration_offset` | dB/λ | `0.0` | Wavenumber-integration contour offset. |
| `nw_samples` | count | `-1` | Wavenumber sample count; `-1` = auto. |
| `plot_rmin` / `plot_rmax` | m | `None` | TL plot range-axis bounds. |
| `vrec` | m/s | `0.0` | Vertical receiver velocity (for Doppler). |
| `offdb` | dB | `None` | Single-mode horizontal offset. |
