# UACPY Documentation

Underwater Acoustic Propagation for Python. This is the complete reference:
concepts, API signatures, model-by-model notes, visualization, signal
processing, noise, and troubleshooting.

> **Status: Beta.** Most APIs are stable; signatures and defaults reflect the
> current codebase. Refer to source for anything not documented here.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Core Concepts](#2-core-concepts)
3. [Environment](#3-environment)
4. [Source & Receiver](#4-source--receiver)
5. [Propagation Models](#5-propagation-models)
6. [Results — typed hierarchy](#6-results--typed-hierarchy)
7. [Visualization](#7-visualization)
8. [Signal Processing](#8-signal-processing)
9. [Ambient Noise](#9-ambient-noise)
10. [Units & Conventions](#10-units--conventions)
11. [Troubleshooting](#11-troubleshooting)
12. [Examples Index](#12-examples-index)

---

## 1. Quick Start

### Install

```bash
git clone --recurse-submodules https://github.com/ErVuL/uacpy.git
cd uacpy
python -m venv uacpy_venv
source uacpy_venv/bin/activate
pip install -e .
./install.sh         # compiles Fortran/C/CUDA binaries into uacpy/bin/
```

`--recurse-submodules` is required: bellhopcuda is vendored as a submodule and
the build will skip it otherwise.

### Minimal example — transmission loss with Bellhop

```python
import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.models import Bellhop
from uacpy.visualization import plot_transmission_loss

env = uacpy.Environment(name="shallow", depth=100.0, sound_speed=1500.0)
source = uacpy.Source(depths=50.0, frequencies=100.0)
receiver = uacpy.Receiver(
    depths=np.linspace(0, 100, 101),
    ranges=np.linspace(100, 10_000, 200),
)

field = Bellhop().compute_tl(env, source, receiver)
plot_transmission_loss(field, env=env)
plt.show()
```

### Imports you will actually use

```python
# Core
from uacpy import (
    Environment, Source, Receiver,
    SoundSpeedProfile,
    BoundaryProperties, RangeDependentBottom,
    SedimentLayer, LayeredBottom, RangeDependentLayeredBottom,
    generate_sea_surface,
)
# Result types (used for type checks in user code)
from uacpy import (
    Result, TLField, PressureField, TransferFunction,
    TimeSeriesField, TimeTrace,
    Arrivals, Rays, Modes,
    Covariance, Replicas, ReflectionCoefficient,
)

# Models
from uacpy.models import (
    Bellhop, BellhopCUDA,
    Kraken, KrakenC, KrakenField,
    Scooter, SPARC, RAM, Bounce,
    OAST, OASN, OASR, OASP, OASES,
)
from uacpy.models.base import RunMode  # when you need explicit run_mode

# Visualization, signal, noise
from uacpy.visualization import (
    plot_transmission_loss, plot_rays, plot_modes,
    plot_environment, plot_environment_advanced,
    plot_ssp, plot_ssp_2d, plot_bathymetry,
    plot_layered_bottom, plot_rd_layered_bottom, plot_bottom_properties,
    plot_arrivals, plot_time_series, plot_reflection_coefficient,
    compare_models,
)
import uacpy
sig = uacpy.signal     # signal processing. The on-disk package is
                       # `acoustic_signal` (avoids shadowing stdlib
                       # `signal`); the alias is only an attribute on
                       # `uacpy`, so `import uacpy.signal` would fail —
                       # go through the parent module as shown.
from uacpy.noise import WenzNoise
```

---

## 2. Core Concepts

Every simulation goes through the same four objects:

```
Environment + Source + Receiver  →  Model.run()  →  Result
```

- **`Environment`** — the water column: depth, SSP, boundaries, bathymetry,
  sediment.
- **`Source`** — depth(s), frequency(ies), launch angles (for rays), beam
  pattern, optional position.
- **`Receiver`** — grid or line of hydrophones at specified depths/ranges.
- **`Result`** — returned by every model. Concrete typed subclass
  (`TLField`, `PressureField`, `TransferFunction`, `TimeSeriesField`,
  `Arrivals`, `Rays`, `Modes`, `Covariance`, `Replicas`,
  `ReflectionCoefficient`) carries the data appropriate to the run mode.

All models inherit from `PropagationModel` (`uacpy.models.base`) and expose:

| Method                         | What it does                                  |
|--------------------------------|-----------------------------------------------|
| `run(env, source, receiver, **kw)`                          | Full-control entry point (required) |
| `compute_tl(env, source, receiver, **kw)`                   | Coherent-TL convenience wrapper |
| `compute_rays(env, source, receiver, **kw)`                 | Ray-paths convenience wrapper |
| `compute_eigenrays(env, source, receiver=None, range_m=, depth_m=, tolerance_m=, max_rays=, truncate=, **kw)` | Eigenrays — single-point or multi-receiver |
| `compute_modes(env, source, n_modes=None, **kw)`            | Normal-modes convenience wrapper |
| `compute_arrivals(env, source, receiver, **kw)`             | Arrivals convenience wrapper |
| `supports_mode(RunMode.X)`                                  | Capability check |
| `supported_modes`                                           | Property returning the list of supported `RunMode`s |

Every model defines the full `compute_*` family. They delegate through
`supports_mode()` and raise `UnsupportedFeatureError` if the underlying
solver doesn't implement that mode (e.g. `Bounce.compute_tl()` raises
because Bounce only emits reflection coefficients). Use
`model.supports_mode(RunMode.X)` or the capability matrix below before
calling. `**kwargs` on every `compute_*` method forwards to `run()`, so
the full per-model configuration surface is always reachable from the
convenience methods.

### Run modes (`uacpy.models.base.RunMode`)

```python
RunMode.COHERENT_TL        # phase-coherent TL
RunMode.INCOHERENT_TL      # intensity-averaged TL
RunMode.SEMICOHERENT_TL    # Lloyd-mirror only (Bellhop)
RunMode.RAYS               # ray paths
RunMode.EIGENRAYS          # rays reaching each receiver
RunMode.ARRIVALS           # amplitude–delay pairs
RunMode.MODES              # normal modes (Kraken / KrakenC / KrakenField)
RunMode.COVARIANCE         # OASN spatial covariance C(f, i, j)
RunMode.REPLICA            # OASN MFP replica fields G(z_src, x_src, y_src; f) at array elements
RunMode.TIME_SERIES        # time-domain output
RunMode.BROADBAND          # complex broadband H(f) (KrakenField, Scooter, RAM, OASP, Bellhop)
RunMode.REFLECTION         # plane-wave reflection coefficients (Bounce, OASR)
```

Which modes each model supports is listed in [Section 5](#5-propagation-models).

### Model constructor options (shared)

```python
Model(
    use_tmpfs=False,                          # tmpfs scratch I/O (Linux, faster)
    verbose=False,                            # print per-step progress
    work_dir=None,                            # pin scratch dir as-is; None ⇒ temp
    cleanup=None,                             # None ⇒ True when uacpy owns the
                                              #   work dir, False when the user
                                              #   supplied work_dir.
    timeout=600.0,                            # subprocess timeout (s) per binary run
    # Per-feature collapse policies — applied by ``_project_environment``
    # at the top of ``run()`` when env contains a feature this model does
    # not natively support.
    bathymetry_collapse_method='max',         # max|median|mean|min|initial
    ssp_collapse_method='r0',                 # r0|rmax|mean|median
    bottom_collapse_method='r0',              # r0|rmax|mean|median
    layered_collapse_method='halfspace',      # halfspace|top_layer|volume_average
    rd_layered_collapse_method='halfspace',   # same set as layered
    altimetry_collapse_method='drop',         # drop
    elastic_collapse_method='fluid',          # fluid (zero shear) | vacuum
)
```

`Source(frequencies=[…])` with more than one frequency on a single-
frequency `RunMode` (`COHERENT_TL`, `RAYS`, `MODES`, `ARRIVALS`, etc.)
raises `ConfigurationError`. Use `RunMode.BROADBAND` for `H(f)` or
`RunMode.TIME_SERIES` for `p(t)`. Multi-depth `Source` raises on every
wrapper except Bellhop (which natively writes the full source-depth
array). Bad `executable=` paths raise `ExecutableNotFoundError` at
construction time.

Each unsupported feature emits one `UserWarning` per `run()` citing the
chosen method and an alternative-model hint. See `_project_environment`
in `models/base.py`.

Most models add their own tuning parameters on top of these (e.g. `dr`, `dz`,
`beam_type`, `cmin`, `cmax`, `volume_attenuation`). Every constructor
parameter is also accepted as a per-call override on `run()` via the
`_UNSET` sentinel pattern: pass `n_beams=600` (or any other constructor
attr) directly to `model.run(...)` and it is temporarily applied for one
call only, then restored on exit. The mechanism lives in `models/base.py`:
the `_resolve_overrides(self, **overrides)` context manager handles the
common case; a per-arg `self._resolve(value, attr)` helper exists for
writers that need the *resolved* value before the context is entered.

**Mode-specific kwargs.** Every `RunMode.TIME_SERIES`-capable wrapper
(Bellhop, Scooter, KrakenField, OASP, RAM) accepts `source_waveform=` and
`sample_rate=` as **explicit keyword arguments on `run()`** — same name,
same role across models. Models with a broadband H(f) path also accept
`frequencies=` for an explicit override of `source.frequencies`. SPARC
computes p(t) from its native source pulse and ignores both.

**Irrelevant kwargs are silently ignored.** Any kwarg passed to `run()`
that doesn't match an explicit signature arg, a constructor attribute,
or a documented writer pass-through is dropped without a warning — same
contract as Python's standard `dict()` constructor with an unknown
keyword. A typo like `Bellhop().run(env, src, rcv, n_beam=10)` (missing
the `s`) silently uses the default `n_beams`.

**Typed exceptions.** Constructor and `run()` failures raise:

| Exception | When |
|---|---|
| `ConfigurationError` | bad input parameters (out-of-range, wrong type, mode/freq mismatch) |
| `UnsupportedFeatureError` | model can't honour a requested `RunMode` or `Environment` axis |
| `ExecutableNotFoundError` | Fortran/C binary missing at the resolved path |
| `ModelExecutionError` | binary ran but failed (non-zero exit, empty output, truncated file). The captured `.prt` log tail is appended to the message for Acoustics-Toolbox models. |

All four inherit from `uacpy.core.exceptions.UACPYError`; catch the base
class to handle "anything went wrong with this model" uniformly.

**Thread safety.** Model instances mutate `self.file_manager` per
`run()` and are **not safe to share across threads**. For parameter
sweeps, instantiate one model per worker (or use `concurrent.futures`
with `ProcessPoolExecutor`).

---

## 3. Environment

### Signature

```python
uacpy.Environment(
    name: str,
    depth: float,                          # max water depth (m)
    ssp = None,                            # SoundSpeedProfile (None → isovelocity)
    sound_speed: float = 1500.0,           # default isovelocity speed when ssp=None
    bathymetry = None,                     # list[(r_m, z_m)] or ndarray (range-dep)
    altimetry  = None,                     # list[(r_m, h_m)] sea surface (positive up)
    bottom   = None,                       # Boundary / RD / Layered / RDLayered
                                            # (default: fluid half-space
                                            #  c=1600 m/s, ρ=1.5 g/cm³, α=0.5 dB/λ)
    surface  = None,                       # BoundaryProperties (default: vacuum)
    volume_attenuation: float = 0.0,       # water-column volume attenuation (dB/λ)
)
```

### Sound-speed profile (`SoundSpeedProfile`)

The water-column speed lives in a single object that handles both the
1-D and 2-D (range-dependent) forms uniformly. Construct via the
classmethods; the `interp` field carries the writer hint
(`'isovelocity' | 'linear' | 'bilinear' | 'munk' | 'pchip' | 'cubic' |
'analytic' | 'n2linear' | 'quad'`).

```python
from uacpy import SoundSpeedProfile

# 1. Constant — also the default when Environment(ssp=None)
env = uacpy.Environment(name="iso", depth=100, sound_speed=1500.0)

# 2. Tabulated 1-D profile
profile = [(0, 1540), (50, 1520), (100, 1510), (200, 1505)]
env = uacpy.Environment(
    name="tab", depth=200,
    ssp=SoundSpeedProfile.from_pairs(profile, interp='linear'),
)

# 3. Analytic (Munk)
env = uacpy.Environment(
    name="munk", depth=5000,
    ssp=SoundSpeedProfile.from_munk(5000),
)

# 4. Range-dependent — c(depth, range)
ranges_m  = np.array([0, 10000, 20000, 30000])  # metres
depths_m  = np.linspace(0, 1000, 50)
ssp_2d    = np.zeros((50, 4))  # fill in…
env = uacpy.Environment(
    name="rd", depth=1000,
    ssp=SoundSpeedProfile.from_2d(
        depths=depths_m, ranges=ranges_m, matrix=ssp_2d,
        interp='quad',  # required for Bellhop to honour the 2-D form
    ),
)
```

The class exposes `at_range(r_m)` (1-D slice), `collapse(method)` (used
by the per-feature collapse pipeline), `to_pairs()` (legacy `(N, 2)` view
of the range-0 column), and `extend_to(z_max)` (constant-extrapolation
to a deeper bathymetry level).

### BoundaryProperties (surface or bottom)

```python
from uacpy import BoundaryProperties

BoundaryProperties(
    acoustic_type='vacuum',        # 'vacuum'|'rigid'|'half-space'|'grain-size'|'file'
    density=1.5,                   # g/cm³
    sound_speed=1600.0,            # m/s — compressional
    attenuation=0.5,               # dB/λ  — compressional
    roughness=0.0,                 # RMS (m)
    shear_speed=0.0,               # m/s (0 = fluid bottom)
    shear_attenuation=0.0,         # dB/λ
    grain_size_phi=1.0,            # when acoustic_type='grain-size'
    reflection_file=None,          # path to .brc / .trc  (when acoustic_type='file')
    reflection_cmin=1400.0,        # m/s — tabulation range for reflection file
    reflection_cmax=10000.0,
    reflection_rmax_m=10000.0,
)
```

Common presets:

```python
# Pressure-release surface (default when surface=None)
surf = BoundaryProperties(acoustic_type='vacuum')

# Rigid bottom (hard rock)
bot  = BoundaryProperties(acoustic_type='rigid')

# Standard fluid sediment
bot  = BoundaryProperties(acoustic_type='half-space',
                          sound_speed=1700.0, density=1.5, attenuation=0.5)

# Elastic bottom (supports shear)
bot  = BoundaryProperties(acoustic_type='half-space',
                          sound_speed=1800.0, shear_speed=400.0,
                          density=2.0, attenuation=0.5, shear_attenuation=1.0)

# Pre-computed reflection coefficients (from BOUNCE or OASR)
bot  = BoundaryProperties(acoustic_type='file', reflection_file='bottom.brc')
```

### Range-dependent bottom (varies with range only)

```python
from uacpy import RangeDependentBottom

bot = RangeDependentBottom(
    ranges           = np.array([0, 5000, 10000, 15000]),  # metres
    depths           = np.array([100, 150, 200, 180]),
    sound_speed      = np.array([1600, 1650, 1700, 1750]),
    density          = np.array([1.5, 1.7, 1.9, 2.1]),
    attenuation      = np.array([0.8, 0.6, 0.4, 0.3]),
    shear_speed      = np.zeros(4),
    shear_attenuation= np.zeros(4),
    acoustic_type    = 'half-space',         # default is 'vacuum'
)
```

Bellhop honours range-dependent bottom geoacoustics natively through the
long `.bty` format (per-range `cp / ρ / α / cs` written to the
`'XL'`-typed bathymetry file and consumed by `BOTY(:)` in
`ReadEnvironmentBell.f90`). RAM dispatches to mpiramS, which threads the
RD-bottom through the Fortran natively. The Collins backends (rams0.5,
ramsurf1.5) use the range-0 profile only and warn on the dropped data.
Models that don't natively support RD bottoms collapse via
`bottom_collapse_method` (default `'r0'` — see §5.2).

### Layered bottom (varies with depth only)

```python
from uacpy import SedimentLayer, LayeredBottom, BoundaryProperties

bottom = LayeredBottom(
    layers=[
        SedimentLayer(thickness=5.0,  sound_speed=1550, density=1.5,
                      attenuation=0.3, shear_speed=100.0),
        SedimentLayer(thickness=10.0, sound_speed=1650, density=1.7,
                      attenuation=0.5),
        SedimentLayer(thickness=20.0, sound_speed=1800, density=2.0,
                      attenuation=0.8),
    ],
    halfspace=BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=2000, density=2.2, attenuation=0.1,
    ),
)

env = uacpy.Environment(name="layered", depth=200,
                        sound_speed=1500, bottom=bottom)

env.has_layered_bottom()          # True
bottom.total_thickness()          # 35.0 m
bottom.layer_depths(env.depth)    # [(200,205), (205,215), (215,235)]
```

Kraken, Scooter, SPARC, and OASES handle `LayeredBottom` natively (via
`NMEDIA > 1` in the Acoustics-Toolbox format). Bellhop and RAM collapse
layers to a single halfspace and warn.

### Range + depth dependent layered bottom

```python
from uacpy import RangeDependentLayeredBottom

rdl = RangeDependentLayeredBottom(
    ranges  = np.array([0, 20000]),          # metres
    profiles=[near_profile, far_profile],    # each is a LayeredBottom
)

# Bathymetry lives on the Environment, not on the RDL.
bathy = np.array([[0, 100], [20000, 300]])   # (range_m, seafloor_depth_m)
env = uacpy.Environment(name="shelf", depth=300, sound_speed=1500,
                        bathymetry=bathy, bottom=rdl)
```

Only RAM supports this natively (via its 4-point per-range sediment
profile). Everything else collapses it. The RDL carries only the
sediment-stack range axis; if you need a sloped seafloor, supply
`bathymetry=` to `Environment` — models that need the seafloor depth
at one of the RDL ranges interpolate it from `env.bathymetry`.

### Bathymetry & altimetry

```python
bathy = np.array([[0, 100], [5000, 150], [10000, 200]])  # (range_m, depth_m)
env = uacpy.Environment(name="slope", depth=200, sound_speed=1500,
                        bathymetry=bathy)

# Rough sea surface (Bellhop and RAM ramsurf backend — others drop it)
# generate_sea_surface returns (range_m, surface_height_m) pairs from a
# Pierson-Moskowitz spectrum parameterised by 10-m wind speed.
alt = uacpy.generate_sea_surface(
    max_range_m=10_000, wind_speed_ms=10.0, n_points=500, seed=0xACED,
)
env = uacpy.Environment(name="rough", depth=100, sound_speed=1500,
                        altimetry=alt)
```

### Environment introspection

```python
env.is_range_dependent                         # any axis varies with range
env.has_range_dependent_bathymetry()
env.has_range_dependent_ssp()
env.has_range_dependent_bottom()
env.has_layered_bottom()
env.has_range_dependent_layered_bottom()

# Per-feature collapse is the canonical path now: pass *_collapse_method
# kwargs to the model constructor (see §2). The whole-env shortcut below
# still exists but only collapses bathymetry + truncates SSP — it does
# not honour the per-feature methods.
env_ri = env.get_range_independent_approximation(method='median')  # legacy
```

---

## 4. Source & Receiver

### Source

```python
uacpy.Source(
    depths,                     # float or array — positive, down from surface (m)
    frequencies,                # float or array — Hz (multiple values give a broadband sweep)
    angles=None,                # launch angles (deg). Default: linspace(-80, 80, 361)
    source_type='point',        # 'point' | 'line'
)
```

For broadband simulations, pass a frequency vector (`frequencies=np.linspace(50, 250, 41)`)
rather than a separate bandwidth parameter.

Useful properties: `source.n_sources`, `source.n_frequencies`,
`source.n_angles`.

### Receiver

```python
uacpy.Receiver(
    depths,                     # required: array or scalar (m)
    ranges=None,                # array or scalar (m); default 0.0
    receiver_type='grid',       # 'grid' (meshgrid) | 'line' (paired)
)
```

- `'grid'` is a full `depths × ranges` mesh (default).
- `'line'` pairs `depths[i]` with `ranges[i]` (same length, or one of the two
  may be a single scalar that is broadcast).
- There is no `'point'` receiver type — a single hydrophone is just a 1×1
  grid: `Receiver(depths=[d], ranges=[r])`.
- For time-series / broadband outputs, models always return the full
  receiver grid. Extract a single trace with
  `TransferFunction.to_time_trace(depth=, range_m=)` or
  `synthesize_time_series(depth=, range_m=)` on the returned typed
  result; see §6 *Results — typed hierarchy*.

Useful properties: `n_depths`, `n_ranges`, `depth_min/max`, `range_min/max`.

```python
# Full TL grid
rx = uacpy.Receiver(depths=np.linspace(0, 100, 101),
                    ranges=np.linspace(100, 10_000, 200))

# Vertical line array at 5 km
rx = uacpy.Receiver(depths=np.linspace(10, 90, 9), ranges=[5000])

# Paired line
rx = uacpy.Receiver(depths=[20, 40, 60], ranges=[1000, 2000, 3000],
                    receiver_type='line')
```

---

## 5. Propagation Models

### 5.1 Model capability matrix

| Model        | Coh TL | Incoh TL | Rays | Eigen | Arrivals | Modes | Time series | Transfer fn | Reflection | Altimetry | Notes |
|--------------|:------:|:--------:|:----:|:-----:|:--------:|:-----:|:-----------:|:-----------:|:----------:|:---------:|-------|
| Bellhop      | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ | — | ✓ | Ray/beam tracing |
| BellhopCUDA  | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ | — | ✓ | C++/CUDA port |
| Kraken       | — | — | — | — | — | ✓ | — | — | — | — | Real modes |
| KrakenC      | — | — | — | — | — | ✓ | — | — | — | — | Complex modes |
| KrakenField  | ✓ | — | — | — | — | — | ✓ | ✓ | — | — | kraken→field pipeline |
| Scooter      | ✓ | — | — | — | — | — | ✓ | ✓ | — | — | Wavenumber integration |
| RAM          | ✓ | — | — | — | — | — | ✓ | ✓ | — | — | Split-step Padé PE; dispatcher → mpiramS / rams0.5 / ramsurf1.5 |
| SPARC        | ✓ | — | — | — | — | — | ✓ | — | — | — | Time-domain PE |
| OAST         | ✓ | — | — | — | — | — | — | — | — | — | OASES TL |
| OASN         | — | — | — | — | — | — | — | — | — | — | OASES covariance / replicas (RunMode.COVARIANCE / RunMode.REPLICA) |
| OASR         | — | — | — | — | — | — | — | — | ✓ | — | OASES reflection coefficients |
| OASP         | ✓ | — | — | — | — | — | ✓ | ✓ | — | — | OASES wideband (wavenumber-int) |
| Bounce       | — | — | — | — | — | — | — | — | ✓ | — | Writes .brc / .irc reflection files |

`Time series` lists models whose output is genuinely a time-domain
waveform (SPARC, OASP), or that can synthesize one from arrivals
(Bellhop's `delayandsum`) or from a broadband transfer function
(KrakenField, Scooter, RAM). `Transfer fn` is the complex-valued H(f)
representation reachable through `RunMode.BROADBAND`.

### 5.2 Environment feature support

| Environment feature             | Bellhop | RAM | Kraken / KrakenC | KrakenField | Scooter | SPARC | OASES | Bounce |
|---------------------------------|:-------:|:---:|:----------------:|:-----------:|:-------:|:-----:|:-----:|:------:|
| 1-D SSP                         | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2-D (range-dep) SSP             | ✓ (with `ssp.interp='quad'`) | ✓ via mpiramS / warn+collapse via rams0.5 / ramsurf1.5 | warn+collapse | **native** (segments) | warn+collapse | warn+collapse | warn+collapse | warn+collapse |
| Range-dep bathymetry            | ✓ | ✓ (all three RAM backends honour multi-point bathymetry) | warn+collapse | **native** (segments) | warn+collapse | warn+collapse | warn+collapse | warn+collapse |
| Halfspace bottom                | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `RangeDependentBottom`          | **native** (long .bty geoacoustics) | **native** via mpiramS / warn+collapse via rams0.5 / ramsurf1.5 | warn+collapse | warn+collapse | warn+collapse | warn+collapse | warn+collapse | warn+collapse |
| `LayeredBottom` (multi-layer)   | warn+halfspace | **native** (mpiramS samples nzs depths; Collins backends use `to_piecewise_breakpoints`) | **native** | **native** | **native** | **native** | **native** | **native** |
| `RangeDependentLayeredBottom`   | warn+halfspace | **native** via mpiramS / warn+collapse via rams0.5 / ramsurf1.5 | warn+collapse | warn+collapse | warn+collapse | warn+collapse | warn+collapse | warn+collapse |
| Elastic bottom (shear)          | ✓ (or ``run_with_bounce()``) | ✓ (auto → rams0.5) | ✓ (Kraken auto-routes to krakenc.exe; KrakenC native) | ✓ (via krakenc.exe) | ✓ | warn+rigid | ✓ | ✓ |
| Reflection file (.brc / .trc)   | ✓ | — | ✓ | ✓ | ✓ | — | — | **output** |
| Altimetry (rough surface)       | ✓ | ✓ (auto → ramsurf1.5) | — | — | — | — | — | — |

> **Note on RAM range-dependence**: the dispatcher routes by env shape. Pure
> fluid + flat-surface envs go to `mpiramS`, which threads range-dependent
> SSP / bottom / layered bottom through the Fortran natively. Once the env
> needs `rams0.5` (any shear) or `ramsurf1.5` (altimetry), uacpy's writer
> currently emits a single SSP / bottom segment at range 0 — so range
> variation in those fields is **dropped with a `UserWarning`**, while
> range-dependent **bathymetry** (and altimetry, for ramsurf) is still
> honoured. To keep range-dependent SSP / bottom plumbed all the way
> through, choose an env that routes to mpiramS.

`warn+collapse` = uacpy emits a `UserWarning` and collapses each
unsupported feature axis via the matching `*_collapse_method`
constructor kwarg (`bathymetry_collapse_method`, `ssp_collapse_method`,
`bottom_collapse_method`, `layered_collapse_method`,
`rd_layered_collapse_method`, `altimetry_collapse_method`). The default
`bathymetry_collapse_method='max'` keeps source/receiver depths above the
seafloor; other valid values are `'min'`, `'median'`, `'mean'`,
`'initial'`. SSP/bottom defaults are `'r0'` (range-0 column / sample);
layered defaults are `'halfspace'`. See
`PropagationModel._project_environment` (`models/base.py`).

The matrix above is driven by six boolean capability flags each model
sets in its `__init__`:

```python
self._supports_altimetry
self._supports_range_dependent_bathymetry
self._supports_range_dependent_ssp
self._supports_range_dependent_bottom
self._supports_layered_bottom
self._supports_range_dependent_layered_bottom
self._supports_elastic_media
```

The flag list is intentionally bounded: each entry answers a distinct
*env-shape* question. Niche numerical-method requirements (3-D,
broadband, specific SSP interp scheme, volume-attenuation formula)
belong in `run()`-time asserts, not in this matrix.

Per-feature alternative-model hints emitted in the warnings come from
`self._unsupported_env_alternatives: Dict[str, str]` keyed by the
feature name (e.g. `'layered_bottom'`). Bellhop overrides
`'layered_bottom'` and `'range_dependent_layered_bottom'` to point users
at `Bellhop.run_with_bounce()`.

### 5.3 Bellhop — ray/beam tracing

```python
from uacpy.models import Bellhop

bh = Bellhop(
    executable=None,                 # auto-detected from uacpy/bin/
    prefer_cuda=True,                # auto-pick CUDA → cxx → fortran
    beam_type='B',                   # see beam code table below
    n_beams=0,                       # 0 = let Bellhop auto-pick (NBEAMS<=0)
    alpha=(-80, 80),                 # launch-angle limits (deg)
    step=0.0,                        # ray step (m); 0 = auto
    z_box=None,                      # max trace depth (m); None = 1.2 × depth
    r_box=None,                      # max trace range (m); None = 1.2 × max range
    source_type='R',                 # 'R' (point, cylindrical) | 'X' (line, Cartesian)
    grid_type='R',                   # 'R' (rectilinear) | 'I' (irregular)
    volume_attenuation=None,         # None | 'T' Thorp | 'F' Francois–Garrison | 'B' Biological
    attenuation_unit=AttenuationUnits.DB_PER_WAVELENGTH,
                                      # AttenuationUnits enum or letter — 'W' dB/λ (default),
                                      # 'N' Np/m, 'F' dB/kmHz, 'M' dB/m, 'Q' Q-factor, 'L' loss-tangent
    francois_garrison_params=None,   # required when volume_attenuation='F': (T, S, pH, z_bar)
    bio_layers=None,                 # required when volume_attenuation='B': [(Z1, Z2, f0, Q, a0), ...]
    bty_interp_type='L',             # '.bty' / '.ati' interpolation: 'L' (linear) | 'C' (curvilinear)
    source_beam_pattern_file=None,   # .sbp path or (angle_deg, level_dB) array; sets RunType(3)='*'
    arrivals_format='ascii',         # 'ascii' → 'A', 'binary' → 'a' (Fortran unformatted)
    # Cerveny / simple-Gaussian advanced beam knobs (used when beam_type ∈ {C, R, S}):
    beam_width_type='F',             # 'F' filling | 'M' match | 'W' waveguide
    beam_curvature='D',              # 'D' double | 'S' single | 'Z' zero
    eps_multiplier=1.0,
    r_loop=1.0,                      # km
    n_image=1, ib_win=4,
    component='P',                   # 'P' pressure | 'D' displacement
    use_tmpfs=False, verbose=False, work_dir=None,
)

from uacpy.models import RunMode
field = bh.run(env, source, receiver, run_mode=RunMode.COHERENT_TL, **per_call_overrides)
```

Every constructor parameter is also accepted as a per-call override on
`run()` (the `_UNSET` sentinel pattern). Pass `n_beams=600` directly to
`bh.run(...)` to override the constructor default for one call only.

**`run_type` (Bellhop-native letter code, passed through to Fortran):**

| Code | Meaning                                 |
|:----:|-----------------------------------------|
| `C`  | Coherent TL (default)                   |
| `I`  | Incoherent TL                           |
| `S`  | Semi-coherent (Lloyd mirror only)       |
| `R`  | Ray trace                               |
| `E`  | Eigenrays                               |
| `A`  | Arrivals (ASCII)                        |
| `a`  | Arrivals (binary) — reader not yet supported; call with `arrivals_format='ascii'` |

**`beam_type`:** `'B'` Gaussian (default), `'R'` ray-centered,
`'C'` Cartesian, `'b'` geometric Gaussian, `'g'` geometric hat (ray),
`'G'` geometric hat (Cartesian), `'S'` simple Gaussian. Case matters:
lowercase variants are ray-centered (distinct numerical method).

**Unsupported knobs (raise `ConfigurationError`):**
`attenuation_unit='m'` (power-law) is rejected because uacpy has no
environment field for the BETA exponent. Grid-type `'I'` requires
`len(receiver.depths) == len(receiver.ranges)` and is pre-validated.

**Time series via `RunMode.TIME_SERIES`.** Bellhop runs in arrivals
mode (`'A'`) under the hood and then convolves the per-receiver
arrivals with a user-supplied `source_waveform`. The full receiver
grid is populated, matching the `(n_d, n_r, n_t)` shape that RAM /
Scooter / KrakenField / OASP return:

```python
ts = bh.run(
    env, source, receiver,
    run_mode=RunMode.TIME_SERIES,
    source_waveform=s,           # 1-D ndarray (the transmitted pulse)
    sample_rate=fs,              # required when source_waveform is given
    time_window=None, t_start=None,
)
trace = ts.get_trace(depth=50.0, range_m=2000.0)   # → TimeTrace at one cell
```

Without `source_waveform` you get a `TransferFunction` covering the full
receiver grid; call `tf.to_time_trace(depth=, range_m=)` to extract a
`TimeTrace` at one cell.

**Bellhop3D:** a stub `Bellhop3D` class exists in `uacpy.models` but its
constructor raises `NotImplementedError`. Partial 3D support is already
in-tree (`write_bty_3d` and `read_boundary_3d` in `io/boundary_io.py`),
but the `.env` writer is 2D-only and 3D arrivals parsing is not
implemented (`io/oalib_reader.py:769` raises `NotImplementedError`).
Wiring up Bellhop3D therefore requires finishing the `.env` writer
(NSx/NSy source grid, Ntheta/Nbeta bearing fan, 3D SSP) and the 3D
arrivals reader. See
`third_party/Acoustics-Toolbox/doc/bellhop3d.htm` for the file format.

**Bellhop + BOUNCE for elastic bottoms.** When the bottom has significant
shear, run BOUNCE first to produce a reflection coefficient file, then feed
it to Bellhop:

```python
field = bh.run_with_bounce(
    env, source, receiver,
    run_mode=RunMode.COHERENT_TL,
    c_low=1400.0, c_high=10000.0, rmax_m=10000.0,
)
```

### 5.4 BellhopCUDA — C++/CUDA ray tracing

C++/CUDA port of BELLHOP, same run types as Bellhop. `install.sh`
builds either `bellhopcuda` (with `--bellhop cuda`, GPU) or `bellhopcxx`
(with `--bellhop cxx`, CPU) — never both — and the wrapper auto-picks
whichever is present. Fortran `bellhop.exe` is always installed alongside
but is **not** used by this wrapper; reach for `Bellhop(prefer_cuda=True)`
if you want full cuda → cxx → fortran fallback.

`BellhopCUDA` is a thin subclass of `Bellhop` whose only own parameters
are `executable` and `dimensionality`; every other constructor kwarg is
forwarded to `Bellhop.__init__` (so the defaults — `beam_type='B'`,
`n_beams=0`, `alpha=(-80, 80)`, etc. — are inherited unchanged).

```python
from uacpy.models import BellhopCUDA

bhc = BellhopCUDA(
    executable=None,          # auto-detect (cuda → cxx)
    dimensionality='2D',      # CLI flag '--2D'. '3D' is plumbed through to
                               # the binary, but the .env writer is still 2D
                               # (see Bellhop3D note in 5.3) — full 3D end-to-end
                               # is NOT supported yet.
    # forwarded to Bellhop:
    # beam_type='B', n_beams=0, alpha=(-80, 80), step=0.0,
    # z_box=None, r_box=None, source_type='R', grid_type='R',
    # volume_attenuation=None, attenuation_unit='W',
    # francois_garrison_params=None, bio_layers=None,
    # bty_interp_type='L', source_beam_pattern_file=None,
    # arrivals_format='ascii', use_tmpfs=False, verbose=False, work_dir=None,
)
field = bhc.run(env, source, receiver, run_mode=RunMode.COHERENT_TL)
```

GPU builds require the CUDA toolkit at install time (`./install.sh --bellhop cuda`).

### 5.5 Kraken & KrakenC — normal modes

Both inherit from the same base (`_KrakenBase`) and only differ in which
executable they call — Kraken (real arithmetic) vs KrakenC (complex,
required for elastic/attenuating bottoms).

```python
from uacpy.models import Kraken, KrakenC

kr = Kraken(
    c_low=None,                  # m/s — None auto: 0.95 × min SSP speed
    c_high=None,                 # m/s — None auto: 1.05 × max SSP/bottom speed
    n_mesh=0,                    # mesh points per wavelength (0 = auto)
    roughness=0.0,               # bottom RMS (m)
    volume_attenuation=None,     # 'T' | 'F' | 'B' | None
    francois_garrison_params=None,  # required when volume_attenuation='F'
    bio_layers=None,             # required when volume_attenuation='B'
    leaky_modes=False,           # if True, override c_high to 1e9 (kraken doc:
                                  # large CHIGH attempts leaky modes)
    top_reflection_file=None,    # path to .trc — sets surface BC TopOpt(2)='F'
    use_tmpfs=False, verbose=False, work_dir=None,
)
modes = kr.run(env, source, receiver, n_modes=None)
# modes is a Modes Result; access via .k, .phi, .depths (also mirrored in metadata)

krc = KrakenC(...)               # identical signature
modes = krc.run(env_with_shear_bottom, source, receiver)
```

For genuinely range-dependent modal propagation use `KrakenField`. When
`Kraken` / `KrakenC` are given a range-dependent env, the wrapper emits a
`UserWarning` per unsupported feature and collapses each via the
matching `*_collapse_method` (bathymetry default `'max'`, SSP/bottom
default `'r0'`).

Note: real Kraken cannot handle elastic media (`shear_speed > 0`); the
wrapper rejects such environments. KrakenField also rejects the `'Q'`
(quadrilateral) SSP interpolation type — that variant is Bellhop-only.

### 5.6 KrakenField — modes + field pipeline

Runs `kraken.exe` (or `krakenc.exe` if the bottom is elastic), then
`field.exe`. Produces TL grids using adiabatic or coupled-mode theory,
or a complex broadband transfer function via `RunMode.BROADBAND`
/ `RunMode.TIME_SERIES`.

```python
from uacpy.models import KrakenField

kf = KrakenField(
    mode_points_per_meter=1.5,
    mode_coupling='adiabatic',     # 'adiabatic' | 'coupled'
    coherent=True,                 # `coupled` + `coherent=False` is rejected up
                                    # front (field.exe has no coupled-incoherent)
    n_segments=10,                 # range segments for RD scenarios
    source_beam_pattern_file=None, # .sbp; sets field.exe Opt(3)='*'
    source_type='R',               # 'R' cylindrical (default) | 'X' Cartesian | 'S' scaled
    use_tmpfs=False, verbose=False, work_dir=None,
)
field = kf.run(env, source, receiver, run_mode=None)   # default COHERENT_TL
```

Supported run modes: `COHERENT_TL`, `BROADBAND`, `TIME_SERIES`.
Range-dependent environments are handled via segmentation. Sea-surface
altimetry is not supported (Bellhop is the only model that consumes
`env.altimetry`).

### 5.7 Scooter — wavenumber integration (fast field)

```python
from uacpy.models import Scooter

sc = Scooter(
    c_low=None, c_high=None,
    n_mesh=0, roughness=0.0,
    rmax_multiplier=2.0,            # multiply max receiver range for k-resolution
    volume_attenuation=None,        # 'T' | 'F' | 'B' | None
    attenuation_unit=AttenuationUnits.DB_PER_WAVELENGTH,
                                     # AttenuationUnits enum or letter (W/N/F/M/Q/L)
    francois_garrison_params=None,
    bio_layers=None,
    source_type='R',                # FLP Option(1): 'R' cylindrical (default) | 'X' Cartesian
    spectrum='positive',            # FLP Option(2): 'positive' (fast) | 'negative' | 'both'
    stabilizing_attenuation_off=False,  # set True only if you know what you're doing
                                        # (the stabiliser prevents pole-on-contour blow-ups)
    field_interp='O',               # FLP Option(3): 'O' polynomial (default) | 'P' Padé
    use_fields_exe=False,           # default: in-tree Python Hankel on .grn.
                                     # Upstream Acoustic Toolbox discontinued
                                     # fields.exe in 2020; install.sh no longer
                                     # builds it. Set True to opt into a legacy
                                     # local install (silently falls back to
                                     # Python if the binary is missing).
    use_tmpfs=False, verbose=False, work_dir=None,
)
field = sc.run(env, source, receiver)
```

Range-independent only. Supports `LayeredBottom`. Supported run modes:
`COHERENT_TL`, `BROADBAND`, `TIME_SERIES`. The Green's-function `.grn` is
always converted to range-domain TL via the in-tree Python Hankel transform
(`uacpy.io.grn_reader`).

### 5.8 RAM — parabolic equation (multi-backend dispatcher)

`RAM` is a façade that auto-selects one of three vendored Collins-family
PE binaries based on the environment:

| `env` shape                              | `select_backend()` returns | Binary       | Source                         |
|------------------------------------------|----------------------------|--------------|--------------------------------|
| fluid bottom + flat surface              | `'mpiramS'`                | `s_mpiram`   | Dushaw broadband Fortran 95 PE |
| any `shear_speed > 0` in the bottom      | `'rams'`                   | `rams0.5`    | Collins elastic PE             |
| `env.altimetry is not None`              | `'ramsurf'`                | `ramsurf1.5` | Collins variable-surface PE    |
| elastic + altimetry                      | `NotImplementedError` (no published Collins PE) |

Inspect the choice without running via `RAM(...).select_backend(env)`.
Every returned `Result` carries `metadata['backend']` so callers can see
which binary actually ran. The `metadata` keys `model`, `backend`,
`frequency`, `source_depth`, `dr`, `dz` are present on every backend's
output for portability.

```python
from uacpy.models import RAM

ram = RAM(
    dr=None,                    # range step (m); None ⇒ Lytaev optimizer
    dz=None,                    # depth step (m); None ⇒ Lytaev optimizer (snapped to env.depth)
    zmax=None,                  # PE-domain depth (m); None ⇒ auto = seafloor + absorbing layer
    np_pade=6,                  # Padé coefficients (2–8)
    ns_stability=1,             # mpiramS / ramsurf1.5 only — ignored by rams0.5
    rs_stability=None,
    Q=None,                     # broadband bandwidth f_c / Q. None resolves to
                                #   Q=1e6, T=1.0 (single-bin) for COHERENT_TL,
                                #   Q=2.0, T=10.0 for BROADBAND/TIME_SERIES.
    T=None,                     # time-window width (s); see Q above.
    depth_decimation=1,
    flat_earth=True,            # apply flat-earth transformation (mpiramS only)
    absorbing_layer_width=20.0, # wavelengths below seafloor (mpiramS only)
    absorbing_layer_attn=10.0,  # dB/λ at domain floor      (mpiramS only)
    n_sed_points=50,            # sediment profile control points (mpiramS only)
    c0=None,                    # PE reference speed (m/s); None ⇒ Lytaev Eq. (15)
    accuracy=1e-3,              # Lytaev per-run accuracy budget
    theta_max=30.0,             # max propagation angle for Lytaev spectrum bound (°)
    rams_theta=45.0,            # rams0.5 Padé rotation angle (°); tuned vs KrakenC
    rams_irot=1,                # rams0.5 rotation flag
    timeout=600.0,              # subprocess timeout per run (s)
)
# Grid selection: (dr, dz) come from the user when set, otherwise from the
# Lytaev (2023) Padé-error optimizer that picks the coarsest grid keeping
# cumulative single-step error below ``accuracy`` over the marched range.
# c₀ defaults to Lytaev Eq. (15) — the PE reference speed that centres
# the spectrum [ξ_min, ξ_max] around 0 and minimises the Padé error.
# All three backends (mpiramS, rams, ramsurf) honour the resolved c₀.
# In the Collins broadband path one (dr, dz, zmax) is picked for the
# whole Q/T-derived band: dr at the lowest freq (largest λ → coarsest
# acceptable step), dz at the highest freq (smallest λ_min → finest
# required step). When kind == 'rams' the optimizer's dr is further
# tightened by 5× because the rotated Padé operator is much less stable.
# Reference: Lytaev, M.S. (2023). Mesh Optimization for the Acoustic
# Parabolic Equation. J. Mar. Sci. Eng. 11(3), 496.
# https://doi.org/10.3390/jmse11030496
field = ram.run(env, source, receiver)
```

Every constructor knob — including the Lytaev tuning pair `accuracy` and
`theta_max` — is also accepted as a per-call override on `run()`. Pass
e.g. `ram.run(env, source, receiver, accuracy=5e-4, theta_max=45.0)` to
tighten the grid for one call without rebuilding the wrapper.

A `λ_p/16` acoustic dz floor is applied to **all three backends** when
the Lytaev optimizer's choice would otherwise drive `dz` below it. For
`rams`/`ramsurf` this is a stability constraint (the Collins finite-
difference march destabilises below `λ_p/16`); for `mpiramS` it is a
runtime cap (mpiramS is stable at any `dz` but cost grows linearly with
the depth-grid count). When the floor activates the wrapper emits a
`UserWarning` citing the kind-specific reason — the Lytaev accuracy
budget is no longer met. Override with explicit `dr=`/`dz=` to bypass.

Supported run modes per backend:

| Backend       | `COHERENT_TL` | `BROADBAND` | `TIME_SERIES` |
|---------------|:-:|:-:|:-:|
| mpiramS       | ✓ | ✓ | ✓ (needs `source_waveform`+`sample_rate`) |
| rams0.5       | ✓ | ✓ (Python freq loop) | ✓ (needs `source_waveform`+`sample_rate`) |
| ramsurf1.5    | ✓ | ✓ (Python freq loop) | ✓ (needs `source_waveform`+`sample_rate`) |

mpiramS does the broadband frequency loop in-process (Dushaw's Fortran
loop with shared setup) and is the fastest broadband backend for the
fluid + flat-surface case. The Collins backends produce broadband by
running the binary once per frequency on the Python side; uacpy's local
Fortran patch (see ``third_party/MODIFICATIONS.md``) makes them dump the
complex PE envelope alongside ``tl.grid`` so the per-frequency results
can be assembled into a transfer function. The phase convention is the
same as mpiramS (``phase_reference='psif_envelope'``), so
``TransferFunction.synthesize_time_series`` works uniformly.

For elastic broadband, ``rams_theta`` may be a callable
``theta_fn(freq_hz) -> float`` to vary the elastic stability angle
across the band — RAMS occasionally needs different ``theta`` at low
vs. high frequencies on the same problem.

#### Surface convention (env.altimetry)

`env.altimetry` follows uacpy's documented convention: `(range_m,
height_m)` with **height positive UP from sea level** (matches
Bellhop/.ati). The RAM dispatcher converts to `ramsurf1.5`'s native
`(range, zsrf)` convention (`zsrf` = depth below z=0, must be ≥ 0)
internally. Wave crests (`height > 0`) are clamped to z=0 with a
`UserWarning`, since `ramsurf1.5` only models surface depressions / ice
keels — for two-sided wave fields use Bellhop. The same `env.altimetry`
fed to Bellhop and RAM produces the same physical scenario; this is
verified by the `altimetry-consistency-bellhop-vs-ramsurf` scenario in
`tests/test_cross_model_agreement.py`.

#### Range dependence

mpiramS supports range-dependent SSP, bathymetry, and layered bottom
(via `RangeDependentLayeredBottom`) natively — RAM is the only model
with full RD-layered support. The Collins backends (rams0.5, ramsurf1.5)
currently use the range-0 SSP and bottom only; passing a range-dependent
SSP or `bottom_rd_layered` to an env that routes to a Collins backend
emits a `UserWarning` flagging the dropped data.

#### Broadband output (mpiramS)

`run_mode=RunMode.BROADBAND` returns a `TransferFunction` shaped
`(n_d, n_r, n_f)` — a complex broadband field across the full receiver
grid. To get a time-domain waveform at one `(depth, range)`, call
`tf.to_time_trace(depth=…, range_m=…)` (single trace → `TimeTrace`)
or `tf.synthesize_time_series(source_waveform=…, sample_rate=…)`
(full grid → `TimeSeriesField`).

Phase convention gotcha (mpiramS only): `tf.phase_reference == 'psif_envelope'`.
The travelling-wave factor `exp(+i·k0·r)` is **not** re-applied by RAM, so a
naive IFFT does not align arrivals at `t = r/c` without re-adding it;
`synthesize_time_series` and `to_time_trace`
handle this correctly.

**TL formula inside UACPY's RAM wrapper (mpiramS):**
`TL = -20·log10(|psif| · 4π) + 10·log10(r)`. Keep this in mind when
comparing against other implementations.

### 5.9 SPARC — time-domain PE

```python
from uacpy.models import SPARC

sp = SPARC(
    c_low=None, c_high=None,
    n_mesh=0, roughness=0.0,
    output_mode='R',                # 'R' horizontal array | 'D' vertical array | 'S' snapshot
    pulse_type='PN+B',
    n_t_out=501,
    t_max=None,                     # None auto: 2.5 × travel time
    t_start=-0.1,
    t_mult=0.999,
    max_depths=20,                  # warning threshold
    rmax_multiplier=1.0001,         # margin so SPARC's RMax > max receiver range
    volume_attenuation=None,        # 'T' | 'F' | 'B' | None
    francois_garrison_params=None,
    bio_layers=None,
    timeout=180.0,                  # subprocess timeout per run (s)
    use_tmpfs=False, verbose=False, work_dir=None,
)
field = sp.run(env, source, receiver)
```

Output is genuinely time-domain pressure. Supported run modes:
`COHERENT_TL`, `TIME_SERIES`. Unlike RAM/OASP, SPARC's `TIME_SERIES`
returns a :class:`TimeSeriesField` directly with the same shape contract
as ``TransferFunction.synthesize_time_series``:

- `data` shape `(n_d, n_r, n_t)` — single- and multi-depth cases share
  the layout (single-depth is `(1, n_r, n_t)`).
- `metadata['time']`, `metadata['dt']`, `metadata['fs']`, `metadata['nt']`,
  `metadata['t_start']` give the shared time axis.

There is no run-time receiver-picking kwarg — the output covers every
receiver in the supplied grid.

#### `output_mode='S'` (snapshot) caveat

SPARC's snapshot file holds the *time evolution* of the wavenumber-domain
Green's function `G(itout, irz, ik)` (`sparc.f90:283-289`). Steady-state
TL at the source frequency is recovered by FFT'ing along the snapshot
time axis (`uacpy.io.grn_reader.sparc_snapshot_to_field`), picking the
bin closest to the source frequency, then Hankel-transforming to range.
This requires `n_t_out` to be large enough that the source frequency
stays below the snapshot Nyquist (`0.5/dt`); the wrapper raises a
`ValueError` with a remediation hint if not. For most acoustic problems
the default `n_t_out=501` is sufficient; bump it for high-frequency or
long-range cases.

### 5.10 Bounce — reflection coefficients

```python
from uacpy.models import Bounce

bn = Bounce(
    c_low=1400.0, c_high=10000.0,  # phase-velocity bounds for tabulation (m/s).
                                    # c_low must be > 0 (kx = ω/c). c_high = 1e9 is
                                    # a valid recommendation for ~full 90° coverage.
    rmax_m=10000.0,                # max range (m) used to derive angular sampling.
                                    # Ignored when `n_angles` is provided.
    volume_attenuation=None,    # 'T' | 'F' | 'B' | None
    n_angles=None,              # explicit override of NkTab; uacpy back-derives
                                 # rmax_m so bounce yields ~n_angles samples
)
field = bn.run(env, source, receiver, output_brc=True, output_irc=True)

# Reflection coefficients + path to .brc are in field.metadata
```

Bounce only emits `RunMode.REFLECTION`; calling `bn.compute_tl(...)` raises
`UnsupportedFeatureError`. Use the resulting `.brc` file via
`BoundaryProperties(acoustic_type='file', reflection_file=...)` when
feeding Kraken, Scooter, or Bellhop with an elastic-bottom reflection
table.

### 5.11 OASES — OAST / OASN / OASR / OASP

Individual model classes mirror the OASES Fortran utilities. All four
accept the standard `volume_attenuation` / `francois_garrison_params` /
`bio_layers` triple and the usual `use_tmpfs` / `verbose` / `work_dir`
plumbing kwargs.

```python
from uacpy.models import OAST, OASN, OASR, OASP
from uacpy.models.base import RunMode

# Transmission loss (wavenumber integration) — RunMode.COHERENT_TL
oast = OAST(
    volume_attenuation=None,
    francois_garrison_params=None, bio_layers=None,
    compute_contour=False,         # add 'C' option (range-depth contour plot)
    compute_depth_average=False,   # add 'A' option (depth-averaged TL)
    complex_contour=True,          # 'J' option (complex integration contour)
)
field = oast.run(env, source, receiver)

# Spatial covariance + matched-field replicas — RunMode.COVARIANCE / RunMode.REPLICA
oasn = OASN(volume_attenuation=None)
cov = oasn.compute_covariance(env, source, receiver)   # Covariance result
# Replicas need Block-X grid kwargs (replica_zmin/zmax/nz, …):
rep = oasn.compute_replicas(
    env, source, receiver,
    replica_zmin=10.0,  replica_zmax=90.0,    replica_nz=20,   # depths in m
    replica_xmin=500.0, replica_xmax=10000.0, replica_nx=40,   # ranges in m
)                                                       # Replicas result

# Reflection coefficients — RunMode.REFLECTION
oasr = OASR(
    angles=None,                   # default linspace(0, 90, 181)
    angle_type='grazing',          # 'grazing' (OASES native) | 'incidence' (90 - x)
    reflection_type='P-P',         # 'P-P' (default) | 'P-SV' | 'P-Slow' | 'transmission'
    volume_attenuation=None,
    francois_garrison_params=None, bio_layers=None,
)
refl = oasr.run(env, source, receiver)     # ReflectionCoefficient
# Broadband sweep: pass freq_min / freq_max / n_frequencies via kwargs:
broad = oasr.run(env, source, receiver,
                 freq_min=50.0, freq_max=200.0, n_frequencies=16)
assert broad.is_broadband                           # R/phi shape (n_angles, n_freq)

# Broadband / pulse transfer function (NOT parabolic-equation — see note)
oasp = OASP(
    n_time_samples=4096,
    freq_max=250.0,
    volume_attenuation=None,
    francois_garrison_params=None, bio_layers=None,
)
tf = oasp.run(env, source, receiver, run_mode=RunMode.BROADBAND)
```

OASP is the **wideband wavenumber-integration / pulse-synthesis** branch
of OASES. Use it for broadband TRF or for range-dependent problems where
OAST's range-independent kernel is inappropriate. For a narrowband RD
problem, OAST warns and uses maximum bathymetry depth.

`OASP.run(run_mode=BROADBAND)` returns a `TransferFunction` shaped
`(n_d, n_r, n_f)`. With `run_mode=TIME_SERIES` plus `source_waveform`
and `sample_rate`, the wrapper internally calls
`tf.synthesize_time_series(...)` and returns a `TimeSeriesField`. For a
single-cell trace use `tf.to_time_trace(depth, range_m)` → `TimeTrace`.

OASN produces covariance matrices (`.xsm`) and matched-field replicas
(`.rpo`), exposed via `RunMode.COVARIANCE` / `RunMode.REPLICA` and
packaged as the typed `Covariance` / `Replicas` results. A covariance is
a hydrophone × hydrophone correlation; replicas are frequency-domain
Green's-function templates for matched-field processing.

### 5.12 OASES unified façade

```python
from uacpy.models import OASES
from uacpy.models.base import RunMode

oases = OASES(use_tmpfs=False, verbose=False)

field  = oases.compute_tl(env, source, receiver)                      # → OAST
field  = oases.compute_tl(env, source, receiver, broadband=True)      # → OASP
refl   = oases.compute_reflection(env, source, receiver)              # → OASR
trf    = oases.compute_transfer_function(env, source, receiver)       # → OASP
```

For maximum control use the individual classes; the façade is for
convenience. Pass `broadband=True` to route a TL request through OASP
(wideband transfer function) instead of OAST — needed for
range-dependent environments where OAST's range-independent kernel is
inappropriate.

OASES does not compute explicit normal-mode eigenfunctions; for those
use `Kraken` or `KrakenC`. Calling `OASES.run(run_mode=RunMode.MODES)`
raises `UnsupportedFeatureError` and points at OASN's hydrophone-array
products: `RunMode.COVARIANCE` (covariance matrix) or `RunMode.REPLICA`
(replica field at array elements).

---

## 6. Results — typed hierarchy

Every `model.run(...)` returns an instance of a typed `Result` subclass.
Concrete types disambiguate shape, methods, and convention. Test the
type with `isinstance`:

```python
from uacpy.core.results import (
    Result, PhaseReference,
    TLField, PressureField, TransferFunction,
    TimeSeriesField, TimeTrace,
    Arrivals, Rays, Modes,
    Covariance, Replicas, ReflectionCoefficient,
)

result = bellhop.run(env, source, receiver)
assert isinstance(result, TLField)
```

Hierarchy:

```
Result                              identification + metadata
├── TLField                         (n_d, n_r) or (n_d, n_r, n_f), dB
├── PressureField                   (n_d, n_r) complex
├── TransferFunction                (n_d, n_r, n_f) complex
├── TimeSeriesField                 (n_d, n_r, n_t) real, p(t) on a grid
├── TimeTrace                       (n_t,) real, p(t) at one (d, r)
├── Arrivals                        per-(isd, ird, irr) arrival lists
├── Rays                            list of ray paths
├── Modes                           Kraken normal modes (k, phi, z)
├── Covariance                      OASN spatial covariance C(f, i, j)
├── Replicas                        OASN MFP replica fields (n_f, n_z, n_x, n_y, n_rcv)
└── ReflectionCoefficient           theta + R/phi shape (n_angles,) or (n_angles, n_freq)
```

Every ``Result`` carries a ``tag(model=…, backend=…, source_depths=…,
frequency=…, phase_reference=…, **extra)`` method used by model wrappers
to attach harmonised identification onto a result returned by a reader.
Mutates the typed attributes and the mirrored ``metadata`` dict in lockstep.

### Rays helpers

`Rays` is a pure data container — a list of ray polylines plus the
geometric context of the run (`source_depths`, `receiver_depths`,
`receiver_ranges`, `is_eigen`). Filtering helpers return new `Rays`
objects and never call back into a solver.

```python
result = bellhop.run(env, source, receiver, run_mode=RunMode.RAYS)
result.is_eigen           # False — regular ray fan (RunType='R')
for ray in result.rays:
    r = ray['r']                    # range array (m)
    z = ray['z']                    # depth array (m)
    alpha = ray['alpha']            # launch angle (deg)
    n_top = ray['num_top_bounces']  # surface reflections
    n_bot = ray['num_bottom_bounces']

# Pure-data subsets. ``is_eigen`` is preserved (a subset of a fan stays
# a fan; a subset of eigenrays stays eigenrays).
direct   = result.filter_by_bounces(kind='direct')       # 0 surface, 0 bottom
no_bot   = result.filter_by_bounces(bot=0)               # never touches seafloor
two_bot  = result.filter_by_bounces(bot=2)               # exactly 2 bottom bounces
deep     = result.filter_by_bounces(bot=(1, None))       # at least one bottom bounce
narrow   = result.filter_by_launch_angle(-5, 5)          # ±5° fan
custom   = result.filter(lambda r: r['alpha'] > 0)       # generic predicate
```

For "rays at a receiver" — i.e. the eigenray solver — use
`compute_eigenrays`. It runs the model's eigenray solver
(`RunType='E'`) and, for a single (range, depth) query, sorts the
returned eigenrays by closest-approach miss distance, drops any beyond
`tolerance_m` (default: one acoustic wavelength), caps to `max_rays`,
and (when `truncate=True`) trims each kept polyline at its closest-
approach index for clean display. `**kwargs` forwards to `run()` so the
full per-model configuration surface (beam type, step size, etc.) is
available:

```python
# Single-point query (most common case).
eig = bellhop.compute_eigenrays(env, source,
                                 range_m=2000, depth_m=30,
                                 tolerance_m=15, max_rays=8)
eig.is_eigen               # True
for ray in eig.rays:
    print(ray['miss_distance_m'], ray['num_top_bounces'],
          ray['num_bottom_bounces'])

# Multi-receiver run — pass a Receiver, skip post-filter.
eig_multi = bellhop.compute_eigenrays(env, source, receiver=rcv_array)
```

The single-point form internally builds a 1-point `Receiver` from
`(range_m, depth_m)` and applies the cosmetic post-filter. The multi-
receiver form skips the post-filter (no single anchor) and returns the
raw solver output, with `is_eigen=True` and `receiver_depths` /
`receiver_ranges` populated so `result.plot(env=env)` renders source
and receiver markers without re-passing them.

### Arrivals helpers

Bellhop's `.arr` payload is nested as
`arrivals_data[src][depth][range] -> dict`; with single-point receivers
the leading dimensions all have length 1. Two flat accessors avoid
walking the structure manually:

```python
result = bellhop.run(env, source, receiver, run_mode=RunMode.ARRIVALS)

# Flat dict for one (src, depth, range) cell:
d = result.at()                          # default cell (all indices 0)
d = result.at(src_idx=0, depth_idx=2, range_idx=5)
delays = d['delays']
amps = d['amplitudes']
n_top = d['n_top_bounces']

# Per-arrival list of dicts (already classified by bounce kind):
for arr in result.to_table():
    print(arr['delay'], arr['amplitude'],
          arr['kind'],          # 'direct' | 'surface' | 'bottom' | 'both'
          arr['src_angle'], arr['rcv_angle'],
          arr['n_top_bounces'], arr['n_bot_bounces'])
```

### Trailing-axis convention

The variable axis (frequency or time) is always **trailing**. So:

| Result | Shape |
|---|---|
| `TLField` (narrowband) | `(n_d, n_r)` |
| `TLField` (broadband) | `(n_d, n_r, n_f)` |
| `PressureField` | `(n_d, n_r)` complex |
| `TransferFunction` | `(n_d, n_r, n_f)` complex |
| `TimeSeriesField` | `(n_d, n_r, n_t)` real |
| `TimeTrace` | `(n_t,)` real |

This matches numpy's FFT default (`axis=-1`) so `np.fft.ifft(H)` directly
produces the time-domain counterpart without an axis argument or a
`moveaxis`. It also makes spatial slicing natural —
`tf.data[d_idx, r_idx, :]` is the spectrum at one (depth, range), and
`tf.data[..., k]` is a 2-D snapshot at one frequency.

### Common metadata keys

Every model populates these on every Result:

| Key | Type | Meaning |
|---|---|---|
| `model` | `str` | Wrapper class name (`'Bellhop'`, `'KrakenField'`, `'RAM'`, …). |
| `backend` | `str` | Concrete binary that actually ran (`'bellhop'`, `'kraken.exe'`, `'field.exe'`, `'mpiramS'`, `'rams0.5'`, `'ramsurf1.5'`, `'oasp'`, …). When the wrapper is not a dispatcher, `backend == model` (lower-cased). |
| `source_depths` | `ndarray` | Every source depth in the run, even if a single scalar was passed in. |
| `frequencies` | `ndarray` | Always 1-D, plural — even single-frequency results store a 1-element array. Use `result.f0` for the convenience scalar. |
| `phase_reference` | `str`, optional | Describes the phase convention of complex H(f) outputs — see below. |

### Phase convention (`TransferFunction.phase_reference`)

`TransferFunction.phase_reference` is a typed `PhaseReference` enum
member (subclass of `str`, so existing string comparisons still work).
The constructor coerces a raw string via
`PhaseReference(phase_reference)` and raises on unknown values, so a
typo can no longer silently corrupt the IFFT path. `synthesize_time_series`
and `to_time_trace` honour each value transparently.

| Value | Models that emit it | Meaning |
|---|---|---|
| `'travelling_wave'` | Bellhop (broadband), Scooter, OASP, KrakenField (broadband — emitted after the wrapper negates `field.exe`'s output to match the Scooter/Bellhop/RAM polarity convention) | H(f) carries the full propagation phase $e^{-i 2\pi f r/c}$. A direct IFFT aligns arrivals at $t = r/c$. |
| `'psif_envelope'` | RAM (mpiramS broadband, rams0.5 broadband, ramsurf1.5 broadband) | Each broadband path hands `TransferFunction.synthesize_time_series` the canonical quantity $\mathrm{conj}(\psi)\,e^{-i k_0 r}$ (modulo amplitude factors) so the downstream IFFT pipeline lands the peak at physical time $r/c_0$. The conjugation flips numpy's $e^{+i\omega t}$ IFFT convention to physics $e^{-i\omega t}$ per JKPS *Computational Ocean Acoustics* §8.2 eq. (8.1)–(8.4); the negative carrier is restored to $+i k_0 r$ by the `t_start` spectrum-shift in `field.py`. The three Collins-derived backends store DIFFERENT raw quantities, so each needs its own conversion: (1) **mpiramS** stores $\psi\,e^{+i(k_0 r+\pi/4)}/(4\pi)$ — carrier embedded — and applies `np.conj(psif) * 4π·e^{-iπ/4}/\sqrt{r}` (`ram.py:1673`). (2) **rams0.5** has `solve(... g0)` multiply by $g_0=e^{i k_0 \Delta r}$ at every range step (`rams0.5.f:830-831`), so its $u$ accumulates the full $e^{+i k_0 r}$ carrier — the broadband path applies just `np.conj(H)` (no extra carrier removal). (3) **ramsurf1.5** has no `g0` argument in `solve` (`ramsurf1.5.f:310`); the carrier is subtracted out of the matrix coefficients via the operator-function `g(x) = (1-νx)²·exp(α·log(1+x) + i σ (\sqrt{1+x}-1))` (`ramsurf1.5.f:564`), so $u$ is the bare envelope and the broadband path applies `np.conj(H) * exp(-i k_0(ω) r)`. After this convention bookkeeping all three backends align with Bellhop's geometric arrival to within ~20 ms on a Pekeris reference (real waveguide modal dispersion), with no calibration constants required. |
| `'time_domain_native'` | SPARC | Result data is real pressure $p(t)$ produced directly by the solver; no phase reconstruction needed. |

Common access patterns:

```python
# TL grid (Bellhop / KrakenField / RAM / Scooter / SPARC / OAST / OASP)
result = bellhop.run(env, source, receiver, run_mode=RunMode.COHERENT_TL)
assert isinstance(result, TLField)
tl   = result.data            # shape (n_d, n_r), dB
rngs = result.ranges          # m
zs   = result.depths          # m

# Rays — Bellhop run_type='R' / compute_rays returns a Rays Result.
rays = bellhop.run(env, source, receiver, run_mode=RunMode.RAYS)
for ray in rays.rays:         # list of ray dicts
    ...

# Arrivals — Bellhop run_type='A' / compute_arrivals returns Arrivals.
arr = bellhop.run(env, source, receiver, run_mode=RunMode.ARRIVALS)
arr.by_receiver[isd][ird][irr]   # {amplitudes, phases, delays, …}

# Modes (Kraken / KrakenC) — typed Modes Result.
modes = kraken.run(env, source, receiver, run_mode=RunMode.MODES)
modes.k                          # complex wavenumbers, shape (M,)
modes.phi                        # mode shapes, shape (nz, M)
modes.depths                     # depth grid

# OASN spatial covariance — Covariance Result.
cov = oasn.compute_covariance(env, source, receiver)
cov.covariance                   # shape (n_freq, n_rcv, n_rcv) complex
cov.frequencies                  # (n_freq,) Hz
cov.receiver_positions           # (n_rcv, 3) (x, y, z) in metres

# OASN matched-field replicas — Replicas Result.
rep = oasn.compute_replicas(
    env, source, receiver,
    replica_zmin=10.0,  replica_zmax=90.0,    replica_nz=20,   # depths in m
    replica_xmin=500.0, replica_xmax=10000.0, replica_nx=40,   # ranges in m
)
rep.replicas                     # shape (n_freq, n_z, n_x, n_y, n_rcv) complex
rep.replica_z, rep.replica_x, rep.replica_y          # all in metres

# Transfer function (KrakenField / Scooter / RAM / OASP / Bellhop broadband)
tf = bellhop.run(env, source, receiver, run_mode=RunMode.BROADBAND)
assert isinstance(tf, TransferFunction)
H     = tf.data                  # complex, shape (n_d, n_r, n_f)
freqs = tf.frequencies
trace = tf.to_time_trace(depth=50.0, range_m=2000.0)   # → TimeTrace
ts    = tf.synthesize_time_series(src_pulse, fs)       # → TimeSeriesField

# Reflection coefficients — unified Bounce / OASR.
rc = bounce.run(env, source, receiver, run_mode=RunMode.REFLECTION)
assert isinstance(rc, ReflectionCoefficient)
rc.theta                         # grazing angles, deg
rc.R                             # magnitude
rc.phi                           # phase, rad
rc.metadata.get('brc_file')      # path, when written

# Time series:
#   • SPARC native           → TimeSeriesField, shape (n_d, n_r, n_t)
#   • RAM run_mode=TIME_SERIES → TimeSeriesField (same shape)
#   • TransferFunction.synthesize_time_series(...) → TimeSeriesField
#   • TransferFunction.to_time_trace(d, r)        → TimeTrace, shape (n_t,)
ts = sparc.run(env, source, receiver, run_mode=RunMode.TIME_SERIES)
assert isinstance(ts, TimeSeriesField)
ts.data         # (n_d, n_r, n_t)
ts.time         # (n_t,) seconds
ts.fs           # sample rate, Hz
trace = ts.get_trace(depth=50.0, range_m=1000.0)   # → TimeTrace
freqs, X = ts.get_spectrum()                       # rfft along time axis
                                                    # X shape (n_d, n_r, n_freq)
```

There is no separate "point" receiver type. To get a single-position
time series, either:

- Build a 1-element receiver grid (`Receiver(depths=[d], ranges=[r])`),
  which gives a degenerate 1×1 grid; or
- Pass `depth=…` / `range_m=…` to `TransferFunction.to_time_trace(...)`
  or `TransferFunction.synthesize_time_series(...)` on the returned
  typed result — both pick the nearest (depth, range) cell from the
  receiver grid by `argmin`.

---

## 7. Visualization

All plotting functions live in `uacpy.visualization` (also exposed as
`uacpy.plot`). They accept a `Result` and optionally an `Environment` and
return matplotlib objects so you can further customize.

The uacpy rcParams (grid, fonts, colors) are applied automatically when
`uacpy.visualization` is imported. If you tweak `matplotlib.rcParams`
yourself and want to snap back to the uacpy defaults, call:

```python
from uacpy.visualization.style import apply_professional_style
apply_professional_style()
```

To revert entirely to matplotlib's defaults, use `mpl.rcdefaults()`.

### Transmission loss

```python
from uacpy.visualization import plot_transmission_loss

fig, ax, im = plot_transmission_loss(
    field, env=None,
    vmin=None, vmax=None,        # dB color-scale limits. Auto if both None:
                                  # vmax = round(median + 0.75·std, tl_round dB);
                                  # vmin = vmax − tl_span.
    cmap=None,                   # None → 'jet_r' (Acoustics-Toolbox standard)
    figsize=(12, 7),
    show_bathymetry=True,
    show_colorbar=True,
    contours=None,               # list of dB contour values, e.g. [70, 80, 90]
    ax=None,
    tl_span=50.0,                # dB span used for auto vmin
    tl_round=10,                 # round vmax to nearest multiple
    seafloor_alpha=1.0,          # 0 transparent ↔ 1 opaque
    frequency=None,              # for broadband fields, pick a frequency slice
)
```

Polar variant: `plot_transmission_loss_polar(field, ...)`.

### Rays

```python
from uacpy.visualization import plot_rays

fig, ax = plot_rays(
    field, env=None,
    source=None, receiver=None,   # optional markers; if omitted, falls
                                  # back to source_depths /
                                  # receiver_depths / receiver_ranges
                                  # stored on the Rays result
    max_rays=None,                # cap rays plotted (sampled or sorted)
    figsize=(12, 6), ax=None,
    color_by_bounces=True,        # red=direct, green=surface,
                                  # blue=bottom, black=both bdys
    ray_colors=None,              # dict override
    seafloor_alpha=1.0,
    linewidth=1.0, alpha=0.55,
    xlim=None, ylim=None,         # km / m, for zoom views
    title=None, show_legend=True,
    truncate_at_receiver=None,    # default: True when field.is_eigen
    closest_approach_threshold_m=None,  # drop rays whose closest approach
                                  # to receiver is > threshold (m)
    sort_by_miss_distance=False,  # combine with max_rays to keep N best
)
```

`field.plot(env=env)` is equivalent. Source position is auto-pulled
from the result. Receiver auto-pull is gated on `field.is_eigen`: for
eigenray results the markers (and `truncate_at_receiver`) fire from
`receiver_depths` / `receiver_ranges` stored on the result; for a plain
ray fan, pass `receiver=...` explicitly if you want the markers.

For one-call eigenray extraction at a single receiver point:

```python
from uacpy.models import Bellhop

bellhop = Bellhop(verbose=False, alpha=(-20, 20), n_beams=51)

# Runs the eigenray solver (RunType='E') targeting (range_m, depth_m),
# then sorts the returned rays by closest-approach miss distance, drops
# any beyond `tolerance_m` (default: one acoustic wavelength), caps to
# `max_rays`, and truncates each kept polyline at its closest-approach
# index so it visibly terminates on the receiver. `**kwargs` forwards
# to `run()` so any per-model setting (beam type, step size, etc.) is
# accessible from this one call.
eig = bellhop.compute_eigenrays(env, source,
                                 range_m=2000, depth_m=30,
                                 tolerance_m=15, max_rays=8)
for ray in eig.rays:
    print(ray['miss_distance_m'], ray['num_top_bounces'],
          ray['num_bottom_bounces'])
```

For multi-receiver eigenray runs (rays to many points in one solve),
pass a `Receiver` directly: `bellhop.compute_eigenrays(env, source,
receiver=rcv_array)` — the post-filter is skipped (no single anchor)
and `is_eigen=True` plus `receiver_depths` / `receiver_ranges` are set
on the result. To narrow such a result onto a single point afterwards,
just call `compute_eigenrays` again for that point; Bellhop runs are
cheap.

### Modes

```python
from uacpy.visualization import plot_modes, plot_mode_functions, \
    plot_modes_heatmap, plot_mode_wavenumbers, plot_dispersion_curves

plot_modes(modes, mode_numbers=[1, 2, 3, 5, 10], normalize=True)
plot_mode_wavenumbers(modes)

# plot_dispersion_curves accepts {freq: Modes_instance, ...}:
modes_50  = kraken.compute_modes(env, source_50)
modes_100 = kraken.compute_modes(env, source_100)
plot_dispersion_curves({50: modes_50, 100: modes_100})

# Group velocity v_g = dω/dk between two Modes at adjacent frequencies:
v_g = modes_50.group_velocity(modes_100)            # ndarray (n_modes,)
```

### Environment / bathymetry / SSP

```python
from uacpy.visualization import (
    plot_environment, plot_environment_advanced,
    plot_ssp, plot_ssp_2d, plot_bathymetry,
    plot_bottom_properties, plot_rd_bottom,
    plot_layered_bottom, plot_rd_layered_bottom,
)

plot_environment_advanced(env, source=source, receiver=receiver,
                          figsize=(16, 12), seafloor_alpha=1.0)
# Auto-adapts to range-dependent SSP, range-dependent bottom, bathymetry,
# and source/receiver overlays — there are no per-panel toggles.

# Requires the corresponding env.has_*() predicate
plot_bottom_properties(env)      # 4-panel summary (depth/cp/ρ/α curves)
plot_rd_bottom(env)              # RangeDependentBottom geological cross-section
                                  # (water column, seafloor, sound-speed-coloured
                                  # half-space, hatched halfspace, profile labels)
plot_layered_bottom(env)         # LayeredBottom (single column)
plot_rd_layered_bottom(env)      # RangeDependentLayeredBottom — piecewise
                                  # cross-section with discrete colour bands
                                  # parallel to the seafloor (matches how RAM
                                  # and KrakenField apply piecewise envs)
plot_ssp_2d(env)                 # range-dependent SSP heatmap
```

### Other result types

```python
from uacpy.visualization import (
    plot_arrivals, plot_time_series, plot_time_trace,
    plot_reflection_coefficient, plot_reflection_coefficient_heatmap,
    plot_transfer_function, plot_transfer_function_slice, plot_phase_field,
    plot_covariance, plot_replicas,
    plot_range_cut, plot_depth_cut, plot_tl_difference,
)

# Arrivals — coloured by bounce class (direct/surface/bottom/both),
# walks any nesting of arrivals_data automatically:
plot_arrivals(arrivals_field)

# Transfer function — 1-D spectrum overlay (optionally with phase):
plot_transfer_function({'Bellhop': bh, 'RAM': ram, 'Scooter': sc},
                        depth_idx=0, range_idx=0,
                        show_phase=True, unwrap_phase=False)

# 2-D |H(f₀)| heatmap on the (depth, range) grid:
plot_transfer_function_slice(tf, frequency=120.0)

# Phase heatmap on the (depth, range) grid for any complex Result:
plot_phase_field(pressure_field)                    # PressureField
plot_phase_field(tf, frequency=120.0)               # TransferFunction slice

# Reflection coefficient. Single-frequency → 1-D R(θ). Broadband → 2-D
# heatmap |R(θ, f)|; pass ``show_phase=True`` to add a phase panel.
plot_reflection_coefficient(rc_single)
plot_reflection_coefficient_heatmap(rc_broadband, show_phase=True)
# `result.plot()` auto-dispatches to whichever is appropriate.

# OASN spatial covariance and MFP replicas:
plot_covariance(cov, freq_index=0)                   # |C(i,j)| heatmap
plot_replicas(rep, freq_index=0, receiver_index=0)   # |G(z, x)| heatmap

# Signed TL difference (a − b) on a diverging colormap, with the
# bathy floor matched to plot_transmission_loss for clean side-by-side:
plot_tl_difference(field_a, field_b, env=env,
                   label='RAM − Bellhop',
                   diff_vmax=10,         # auto if None
                   show_colorbar=True)
```

### Comparison helpers

```python
from uacpy.visualization import (
    compare_models, compare_range_cuts,
    plot_model_statistics, plot_model_comparison_matrix,
    plot_comparison_curves,
)

# `compare_models` accepts a dict of {name: field} and renders one TL
# panel per model with a single shared colorbar — guaranteeing equal
# panel widths so brown bathy floors align across panels.
compare_models(
    {'Bellhop': f_bh, 'RAM': f_ram, 'KrakenField': f_kf},
    env=env,
    ncols=None,                     # row layout if None
    vmin=30, vmax=90,
    contours=None,                  # e.g. [70, 90] for overlay isolines
    suptitle='Model Comparison',
)
```

### Quick plotting

For one-liners during exploration. Functions in `uacpy.visualization.quickplot`
are prefixed with `quick_` (TL, rays, environment, modes, range/depth cuts):

```python
from uacpy.visualization import quickplot

quickplot.quick_tl(field, env=env)               # TL map
quickplot.quick_rays(field, env=env)             # ray trace
quickplot.quick_env(env, source=source)          # environment overview
quickplot.quick_modes(field, n_modes=6)          # mode shapes
quickplot.quick_cut(field, depth=50)             # range cut at depth
quickplot.quick_compare({'Bellhop': f1, 'RAM': f2}, env=env)
```

---

## 8. Signal Processing

Reachable as `uacpy.signal` (the on-disk package is `acoustic_signal` to
avoid shadowing the stdlib `signal` module). The alias is set as an
**attribute** of the `uacpy` package, so use:

```python
import uacpy
sig = uacpy.signal           # OK — attribute access
```

`import uacpy.signal as sig` does **not** work: `signal` is not a real
submodule.

### Waveform generation (`uacpy.signal.generation`)

The two main families are `tone_burst` / `lfm_chirp` / `hfm_chirp`
(MATLAB-style: build a fixed-duration signal and return its time vector
alongside) and `cw` / `sweep` (arlpy-style: return just the signal,
duration is implied by `n / fs`).

```python
# Windowed tone burst — returns (signal, time)
s, t = sig.tone_burst(frequency=1000.0, n_cycles=5,
                      sample_rate=48000, window=True)

# Linear-frequency chirp from fmin → fmax over T seconds — returns (s, t)
s, t = sig.lfm_chirp(fmin=100.0, fmax=1000.0, T=1.0, sample_rate=10000)

# Hyperbolic-frequency chirp (linear period modulation) — returns (s, t)
s, t = sig.hfm_chirp(fmin=100.0, fmax=1000.0, T=1.0, sample_rate=10000)

# Plain CW (no time vector returned)
s = sig.cw(fc=1000.0, duration=0.5, fs=48000,
           window=None, complex_output=False)

# Frequency sweep (linear / quadratic / logarithmic / hyperbolic)
s = sig.sweep(f1=100.0, f2=1000.0, duration=0.5, fs=48000,
              method='linear', window=None)

# Pulse waveforms (driven by an external time vector)
s = sig.gaussian_pulse(time=t, delay=0.05, duration=0.01)
s = sig.ricker_wavelet(time=t, F=500.0)

# BPSK on a ±1 chip sequence
s = sig.bpsk_modulate(s_bipolar=bits, fc=12000.0, fs=48000.0,
                      chips_per_sec=3000.0)

# Bandlimited Gaussian noise centred at fc with given bandwidth
n = sig.make_bandlimited_noise(fc=1000.0, bandwidth=200.0,
                               duration=1.0, sample_rate=48000)

# Spectral Synthesis of Random Processes — generate a time series realising
# a target PSD. Reachable only via the submodule (not re-exported at top level):
from uacpy.acoustic_signal.generation import ssrp
t, x, fs = ssrp(Pxx, Fxx, duration=1.0, scale=1.0)
```

### Processing (`uacpy.signal.processing`)

```python
# Plane-wave array replicas (steering vectors)
rep = sig.planewave_rep(phone_coords, angles, freq,
                        c=1500.0, window=False)

# Conventional plane-wave beamformer — returns (power_dB, angles, peak_dB)
power, angles_out, peak = sig.beamform(
    pressure, phone_coords, freq=100.0,
    angles=None,        # default: -90..90 in 1° steps
    SL=150.0, NL=0.0, c=1500.0,
)

# Add Gaussian noise sized for a given SL/NL pair around carrier (fc, BW)
y = sig.add_noise(timeseries, sample_rate=fs,
                  source_level_db=150.0, noise_level_db=80.0,
                  fc=1000.0, bandwidth=200.0)

# Fourier-synthesize a time series from a frequency-domain pressure response.
# `pressure_freq` shape (..., n_freqs); `freq_vec` is the frequency axis (Hz).
# `source_spectrum` lets you weight by an arbitrary source spectrum; Tstart
# shifts the time origin. Returns (time, signal).
t, s = sig.fourier_synthesis(pressure_freq, freq_vec,
                             source_spectrum=None, Tstart=0.0)
```

### Advanced (`uacpy.signal.advanced` — adapted from arlpy)

```python
# Time vector
t = sig.time(n, fs)                # n samples or len(array) at fs Hz

# Baseband ↔ passband conversion
pb = sig.bb2pb(x, fd, fc, fs=None, axis=-1)   # baseband (rate fd) → passband
bb = sig.pb2bb(x, fs, fc, fd=None, flen=127, axis=-1)  # passband → baseband

# Zero-phase filtering, periodic cross-correlation
y = sig.lfilter0(b, a, x, axis=-1)
r = sig.correlate_periodic(a, v=None)

# Single-bin DFT (Goertzel) — extract amplitude/phase at frequency f
X = sig.goertzel(f, x, fs=2.0, filter=False)

# Numerically-controlled oscillators (whole-array and generator forms)
phi = sig.nco(fc, fs=2.0, phase0=0, wrap=2*np.pi, func=None)
gen = sig.nco_gen(fc, fs=2.0, phase0=0, wrap=2*np.pi, func=None)

# Generator-based IIR/FIR filter for streaming data
filt = sig.lfilter_gen(b, a)

# m-sequence (max-length pseudo-random sequence). `spec` is either the
# register length (int) or an explicit polynomial (list).
seq = sig.mseq(spec, n=None)

# Polyphase resample (up/down by integer ratio)
y = sig.resample(data, up_factor, down_factor)
```

### Spectral analysis (`uacpy.signal.analysis`)

Class-based API — each estimator is instantiated with reference level and
(for Welch-based tools) spectral-estimation parameters, then `.compute(data, fs)`
is called, and finally `.plot(...)` renders the result.

```python
from uacpy.acoustic_signal.analysis import (
    PSD, PPSD, Spectrogram, SEL, FRF, FKTransform,
)
# (or, if you've already done `import uacpy`:  `uacpy.signal.analysis.PSD`, etc.)

# --- Power Spectral Density (Welch) ---
psd = PSD(ref=1e-6, nperseg=8192, noverlap=4096, window="hann")
freqs, Pxx = psd.compute(data, fs)          # Pxx in Pa^2/Hz (linear)
fig, ax = psd.plot(title="Site A", ymin=40, ymax=120)
psd.add_to_plot(ax, label="Site B")         # overlay another curve

# --- Probabilistic Power Spectral Density (PPSD) ---
# Segments the signal, computes a Welch PSD per segment, then builds a 2-D
# histogram (PDF) of spectral levels at each frequency bin — useful for
# characterising ambient noise statistics over long recording
p = PPSD(ref=1e-6, seg_duration=1.0, overlap_pct=50, ddB=1.0,
         lvlmin=0, lvlmax=150, nperseg=8192)
p.compute(data, fs)                          # accepts 1-D, 2-D, or list of signals
fig, ax = p.plot(title="PPSD", ymin=0, ymax=200, vmin=0, vmax=None)

# --- Spectrogram ---
sg = Spectrogram(ref=1e-6, nperseg=2048, noverlap=1024, window="hann")
sg.compute(data, fs)
fig, ax = sg.plot(title="Event", ymin=1, ymax=None, vmin=0, vmax=200)

# --- Sound Exposure Level (SEL) per band ---
sel = SEL(fmin=8.9125, fmax=22387, band_type="third_octave",
          num_bands=30, ref=1e-6, integration_time=None)
sel.compute(data, fs, chunk_size=262144, nfft=None)
fig, ax = sel.plot(title="Pile driving", ylim=(0, 200))

# --- Frequency Response Function (FRF) ---
# Estimate a transfer function x -> y via Welch (H1/H2), ETFE, periodic
# ETFE, or least-squares FIR; with impulse response and coherence.
# method ∈ {'welch', 'ls_fir', 'etfe', 'p_etfe'};  estimator ∈ {'H1', 'H2'}.
frf = FRF(method="welch", estimator="H1", m=512, nperseg=4096)
frf.compute(x, y, fs)                         # 1-D or 2-D (multi-measurement)
fig, ax = frf.plot(title="Channel", ymin=-60, ymax=60)
frf.plot_coh(title="Coherence")
frf.plot_impulse_info(title="LSFIR diagnostics")   # only for method="ls_fir"

# --- f-k transform (array data) ---
fk = FKTransform(ref=1e-6)
fk.compute(data, fs, dx)                      # data shape (n_sensors, n_samples)
fig, ax = fk.plot(title="f-k", vmin=-60, vmax=20)
x_back = fk.inverse()                         # round-trip to time domain
```

Conventions: all `plot()` methods return `(fig, ax)` so you can overlay
additional content. Reference pressure defaults to **1 µPa** (water); pass
`ref=2e-5` for air. dB scales in plots are dB re `ref²/Hz` for PSD-like
quantities and dB re `ref²·s` for SEL. PSDs are stored in linear units
(`Pa²/Hz`) on the object — conversion to dB happens in `.plot()`.

Scientific content in `advanced.py` and `uacpy.core.acoustics` is adapted
from arlpy (BSD-3-Clause) — see `uacpy/third_party/arlpy/LICENSE` and
`NOTICE`.

---

## 9. Ambient Noise

`uacpy.noise` packages a Tollefsen / Pecknold-style Wenz model
(wind / shipping / rain / thermal / turbulence). The user-facing API
is the :class:`WenzNoise` class plus the standalone
:func:`compute_windnoise` helper.

```python
import numpy as np
from uacpy.noise import WenzNoise

f    = np.logspace(0, 5, 1000)                     # 1 Hz – 100 kHz
wenz = WenzNoise(f, wind_speed=15,
                  rain_rate='moderate',
                  water_depth='deep',
                  shipping_level='medium')

# Per-component spectral levels (dB re 1 µPa²/Hz):
wenz.total          # incoherent sum
wenz.shipping       # Wenz 1962
wenz.wind           # Merklinger 1979 + Piggott 1964 shallow correction
wenz.rain           # Torres & Costa 2019
wenz.thermal        # Mellen 1952
wenz.turbulence     # Nichols & Bradley 2016
wenz.components     # (N, 6) ndarray: [total, ship, wind, rain, therm, turb]

fig, ax = wenz.plot()                              # all components
fig, ax = wenz.plot(show_components=False)         # total only
```

`shipping_level` ∈ `{'no','low','medium','high'}`, `rain_rate` ∈
`{'no','light','moderate','heavy','veryheavy'}`, `water_depth` ∈
`{'deep','shallow'}`. `wind_speed` is in **knots**.

To synthesise a time-domain noise realisation from the spectrum and
verify the round-trip, pair `WenzNoise.as_psd()` with
:func:`uacpy.signal.ssrp` (spectral synthesis of random processes) and
:class:`uacpy.signal.PPSD` (probability density of Welch-PSD levels):

```python
import uacpy
Pxx = wenz.as_psd()                                # Pa² / Hz (linear)
t, x, fs = uacpy.signal.ssrp(Pxx, wenz.frequencies, duration=30.0)
ppsd = uacpy.signal.PPSD(ref=1e-6, seg_duration=1.0)
ppsd.compute(x, fs)
fig, ax = ppsd.plot()
```

See `examples/example_09_ambient_noise.py` for the full pipeline.

References: Tollefsen & Pecknold, *Wenz curves for predicting ambient
noise*, DRDC-RDDC-2022-D051 (Annex A: canonical MATLAB implementation —
see `docs/WenzCurves.pdf`); Wenz (1962); Mellen (1952); Piggott (1964);
Merklinger (1979); Torres & Costa (2019); Nichols & Bradley (2016).

---

## 10. Units & Conventions

All quantities are SI-based. Cheat sheet:

| Quantity                | Unit                   | Notes                                       |
|-------------------------|------------------------|---------------------------------------------|
| Depth, range, altitude  | meters (m)             | Depth positive down from surface            |
| Sound speed             | m/s                    | Water 1400–1550; sediment 1500–2000         |
| Shear speed             | m/s                    | 0 for fluid; 100–500+ for elastic solids    |
| Frequency / sample rate | Hz                     |                                             |
| Time                    | s                      |                                             |
| Launch / grazing angle  | degrees                | Negative = upward, 0 = horizontal           |
| Attenuation             | dB / wavelength (dB/λ) | Used throughout the Acoustics Toolbox       |
| Transmission loss       | dB                     | TL = −20·log₁₀(\|p\|/\|p₀\|)                |
| Density                 | g/cm³                  | Water ≈ 1.0; sediment 1.2–2.5               |
| Source level            | dB re 1 μPa @ 1 m      |                                             |
| Noise spectral density  | dB re 1 μPa²/Hz        |                                             |

### Internal ↔ file-format units

UACPY stores ranges internally in meters and converts at the I/O boundary
when writing model input files:

| Model              | Input file expects                                  | UACPY converts? |
|--------------------|-----------------------------------------------------|:---------------:|
| Bellhop `.env`     | ranges in km                                        | ✓               |
| Bellhop `.bty`     | ranges in km                                        | ✓               |
| Kraken / KrakenField `.env` / `.flp` | ranges in km                      | ✓               |
| RAM (mpiramS backend) | receiver-range list in m; SSP / bottom profile headers in km | ✓   |
| RAM (Collins backends — rams0.5 / ramsurf1.5) | meters throughout (Collins `ram.in` format) | —   |
| OASES              | meters throughout                                   | —               |

You always pass **meters** to `Receiver(ranges=…)` and
`bathymetry=[[range_m, depth_m], …]`. The wrapper handles per-model
conversions; you should never need to convert manually.

---

## 11. Troubleshooting

**Executable not found.** You forgot to run `./install.sh`, or the
compilation failed for one of the Fortran/C binaries. `uacpy/bin/` should
contain `oalib/`, `mpirams/`, `bellhopcuda/`, and — after downloading —
`oases/`. Rerun `./install.sh` and read its stderr. OASES is not
redistributable; the installer fetches it from MIT on demand.

**"Source/Receiver depth exceeds environment depth".** The model's
`validate_inputs()` checks sources and receivers against `env.depth` (or
the maximum of `env.bathymetry`). Trim your receiver grid or deepen the
environment.

**"Kraken does not support range-dependent environments".** Use
`KrakenField` (adiabatic or coupled modes) or switch to Bellhop/RAM.

**A model dropped my range-dependent SSP / bottom / layered bottom.**
Each unsupported feature triggers one `UserWarning` per `run()` and is
collapsed via the matching `*_collapse_method` constructor kwarg.
Either pick a model that supports the feature (see the matrix in §5.2)
or change the collapse policy — e.g.
`Kraken(ssp_collapse_method='mean', bottom_collapse_method='median')`.

**`LayeredBottom` in Bellhop collapses to a halfspace.** Bellhop natively
honours range-dependent bathymetry and range-dependent bottom
geoacoustics (long `.bty`), but does not handle stratified sediment.
For layered bottoms with Bellhop, run `Bellhop.run_with_bounce()` to
generate a reflection-coefficient table from BOUNCE and feed it back via
`BoundaryProperties(acoustic_type='file', reflection_file=…)`.
Otherwise use Kraken, KrakenC, Scooter, SPARC, or OASES.

**OASN normal modes.** OASN produces covariance matrices (`.xsm`) and
matched-field replicas (`.rpo`), exposed via `oasn.compute_covariance(...)`
→ `Covariance` and `oasn.compute_replicas(...)` → `Replicas`. For
depth-eigenfunction normal modes use Kraken or KrakenC.

**Bellhop ray box too small.** Increase `r_box` / `z_box` or let them
default (1.2 × receiver extent).

**Kraken not converging.** Raise `n_modes` or widen `c_low`/`c_high` — the
defaults (0.95 × min SSP, 1.05 × max SSP/bottom) sometimes miss high-speed
bottom-trapped modes.

**TL contains NaNs at long range (RAM).** Check `zmax` and
`absorbing_layer_width`; spurious reflections from the PE domain bottom
commonly show up as NaNs. Increasing `absorbing_layer_width` to ~30–40
wavelengths usually fixes it.

**Bellhop time series.** Bellhop supports `RunMode.TIME_SERIES`: the
wrapper runs Bellhop in arrivals mode (`'A'`), then calls `delayandsum`
(`uacpy.models.bellhop.delayandsum`) to convolve a source waveform with
the per-receiver impulse response. For time-domain propagation you can
also use SPARC (native time-domain PE) or OASP (broadband
wavenumber-integration / pulse synthesis). To work with raw Bellhop
arrivals directly, call with `run_mode=RunMode.ARRIVALS` and consume
`field.metadata['arrivals_by_receiver']` yourself.

**Elastic bottoms in Bellhop.** Bellhop is a fluid-only ray tracer. Use
`Bellhop.run_with_bounce()` or route through BOUNCE → `.brc` →
Kraken/Scooter.

---

## 12. Examples Index

All scripts live in `uacpy/uacpy/examples/` and are runnable as-is once
`./install.sh` has completed.

The 24 examples are numbered sequentially. Every example carries the
`@pytest.mark.slow` marker in `tests/test_examples_integration.py` (so a
default `pytest -m "not slow"` skips them all); the *Long* column flags
the four examples that need a longer subprocess timeout (see
`_LONG_TIMEOUT_STEMS` in the integration test — 240 s instead of 120 s).

| # | File | Demonstrates | Long? |
|---|------|--------------|:-----:|
| 01 | `example_01_basic_shallow_water.py`            | Minimal TL — start here | |
| 02 | `example_02_sound_speed_profiles.py`           | SSP types (linear, Munk, cubic, …) — Munk + Pekeris + thermocline | ✓ |
| 03 | `example_03_multi_frequency.py`                | Sweep over frequencies | |
| 04 | `example_04_bellhop_advanced.py`               | Beam types, source patterns, advanced rays | |
| 05 | `example_05_ram_advanced.py`                   | RAM (mpiramS) with sloping shelf + RD bottom | |
| 06 | `example_06_kraken_advanced.py`                | Modal analysis with Kraken | |
| 07 | `example_07_all_models_comparison.py`          | All models side by side, `compare_models` + `plot_rd_bottom` | |
| 08 | `example_08_long_range.py`                     | Convergence-zone propagation | |
| 09 | `example_09_ambient_noise.py`                  | Wenz ambient noise + ssrp synthesis + PPSD verification | |
| 10 | `example_10_signal_processing.py`              | CW, chirps, matched filtering | |
| 11 | `example_11_bellhop_run_modes.py`              | Every Bellhop run mode + `compute_eigenrays` API | |
| 12 | `example_12_attenuation_models.py`             | Thorp / Francois-Garrison / biological | |
| 13 | `example_13_oases_suite.py`                    | OAST / OASN / OASR / OASP | |
| 14 | `example_14_new_plotting_features.py`          | Visualization tour | |
| 15 | `example_15_elastic_boundaries_comparison.py`  | Fluid vs elastic bottoms | |
| 16 | `example_16_bellhop_bounce_integration.py`     | `Bellhop.run_with_bounce()` + `LayeredBottom` | |
| 17 | `example_17_boundary_conditions_layered.py`    | Surface BC + layered bottoms (RD layered) | ✓ |
| 18 | `example_18_rd_bottom_krakenfield_vs_ram.py`   | RD layered bottom: adiabatic vs coupled vs RAM | |
| 19 | `example_19_broadband_comparison.py`           | Multi-model H(f) + p(t) (Bellhop / RAM / Scooter / OASP / SPARC) | ✓ |
| 20 | `example_20_ram_backends.py`                   | RAM dispatcher: mpiramS / rams / ramsurf | |
| 21 | `example_21_bellhop_vs_ramsurf.py`             | Bellhop vs ramsurf with rough surface | |
| 22 | `example_22_ram_lytaev_grid.py`                | RAM Lytaev (2023) Padé grid optimizer | ✓ |
| 23 | `example_23_collapse_methods.py`               | Same RD env collapsed four ways via `*_collapse_method` kwargs | |
| 24 | `example_24_synthesize_time_series.py`         | Bellhop BROADBAND H(f) → IFFT → p(t) via `TransferFunction.synthesize_time_series` | |

Smoke test:

```bash
python uacpy/uacpy/examples/example_01_basic_shallow_water.py
```

---

## Reference

- Source:  https://github.com/ErVuL/uacpy
- Issues:  https://github.com/ErVuL/uacpy/issues
- Contact: ervul.github@gmail.com
- Citation / licensing / acknowledgements: see `README.md`
