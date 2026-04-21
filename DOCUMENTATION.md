# UACPY Documentation

Underwater Acoustic Propagation for Python. This is the complete reference:
concepts, API signatures, model-by-model notes, visualization, signal
processing, noise, and troubleshooting.

> **Status: Alpha.** Expect rough edges. Signatures and defaults reflect the
> current codebase; refer to source for anything not documented here.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Core Concepts](#2-core-concepts)
3. [Environment](#3-environment)
4. [Source & Receiver](#4-source--receiver)
5. [Propagation Models](#5-propagation-models)
6. [Field & Results](#6-field--results)
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
git clone https://github.com/ErVuL/uacpy.git
cd uacpy
python -m venv uacpy_venv
source uacpy_venv/bin/activate
pip install -e .
./install.sh         # compiles Fortran/C/CUDA binaries into uacpy/bin/
```

### Minimal example — transmission loss with Bellhop

```python
import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.models import Bellhop
from uacpy.visualization import plot_transmission_loss

env = uacpy.Environment(name="shallow", depth=100.0, sound_speed=1500.0)
source = uacpy.Source(depth=50.0, frequency=100.0)
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
    Environment, Source, Receiver, Field,
    BoundaryProperties, RangeDependentBottom,
    SedimentLayer, LayeredBottom, RangeDependentLayeredBottom,
    generate_sea_surface,
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
import uacpy.signal as sig               # signal processing (acoustic_signal)
from uacpy.noise import AmbientNoiseSimulator
```

---

## 2. Core Concepts

Every simulation goes through the same four objects:

```
Environment + Source + Receiver  →  Model.run()  →  Field
```

- **`Environment`** — the water column: depth, SSP, boundaries, bathymetry,
  sediment.
- **`Source`** — depth(s), frequency(ies), launch angles (for rays), beam
  pattern, optional position.
- **`Receiver`** — grid or line of hydrophones at specified depths/ranges.
- **`Field`** — returned by every model. Holds TL, pressure, rays, modes,
  arrivals, reflection coefficients, or time series depending on the run.

All models inherit from `PropagationModel` (`uacpy.models.base`) and expose:

| Method                         | What it does                                  |
|--------------------------------|-----------------------------------------------|
| `run(env, source, receiver, **kw)` | Full-control entry point (required)       |
| `compute_tl(env, source, receiver)` | Convenience: coherent TL with defaults   |
| `compute_rays(env, source)`    | Convenience: ray paths (Bellhop only)         |
| `compute_modes(env, source, n_modes)` | Convenience: modes (Kraken/KrakenC/OASN) |
| `compute_arrivals(env, source, receiver)` | Convenience: arrivals (Bellhop)   |
| `supports_mode(RunMode.X)`     | Capability check                              |

### Run modes (`uacpy.models.base.RunMode`)

```python
RunMode.COHERENT_TL      # phase-coherent TL
RunMode.INCOHERENT_TL    # intensity-averaged TL
RunMode.SEMICOHERENT_TL  # Lloyd-mirror only (Bellhop)
RunMode.RAYS             # ray paths
RunMode.EIGENRAYS        # rays reaching each receiver
RunMode.ARRIVALS         # amplitude–delay pairs
RunMode.MODES            # normal modes
RunMode.MODES_FIELD      # field built from modes
RunMode.TIME_SERIES      # time-domain output
```

Which modes each model supports is listed in [Section 5](#5-propagation-models).

### Model constructor options (shared)

```python
Model(
    use_tmpfs=False,   # Use RAM-backed tmpfs for scratch I/O (Linux, faster)
    verbose=False,     # Print per-step progress
    work_dir=None,     # Pin scratch directory instead of using a temp dir
)
```

Most models add their own tuning parameters on top of these (e.g. `dr`, `dz`,
`beam_type`, `cmin`, `cmax`, `volume_attenuation`). Every constructor
parameter is also accepted as a per-call override on `run()` via `_UNSET`
sentinel pattern — see each model for specifics.

---

## 3. Environment

### Signature

```python
uacpy.Environment(
    name: str,
    depth: float,                                     # max water depth (m)
    ssp_type: str = 'isovelocity',                    # see table below
    ssp_data = None,                                  # list[(z, c)] or ndarray
    ssp_2d_ranges = None,                             # range-dep SSP: km
    ssp_2d_matrix = None,                             # range-dep SSP: (nz, nr)
    sound_speed: float = 1500.0,                      # used when ssp_type='isovelocity'
    bathymetry = None,                                # list[(r_m, z_m)] or ndarray
    altimetry  = None,                                # list[(r_m, z_m)] sea surface
    bottom   = None,                                  # Boundary / RD / Layered / RDLayered
    surface  = None,                                  # BoundaryProperties (default: vacuum)
    attenuation: float = 0.0,                         # water volume attenuation (dB/λ)
)
```

### Sound-speed profiles

Valid `ssp_type` values (lowercased automatically): `'isovelocity'`,
`'linear'`, `'bilinear'`, `'munk'`, `'pchip'`, `'cubic'`, `'analytic'`,
`'n2linear'`, `'quad'`.

```python
# 1. Constant
env = uacpy.Environment(name="iso", depth=100, sound_speed=1500.0,
                        ssp_type='isovelocity')

# 2. Tabulated profile — ssp_data is a list of (depth, speed)
profile = [(0, 1540), (50, 1520), (100, 1510), (200, 1505)]
env = uacpy.Environment(name="tab", depth=200, ssp_type='linear',
                        ssp_data=profile)

# 3. Analytic (Munk) — self-generating
env = uacpy.Environment(name="munk", depth=5000, ssp_type='munk')

# 4. Range-dependent SSP — ssp_2d_matrix shape is (n_depths, n_ranges)
ranges_km = np.array([0, 10, 20, 30])
depths_m  = np.linspace(0, 1000, 50)
ssp_2d    = np.zeros((50, 4))  # fill in...
env = uacpy.Environment(name="rd", depth=1000, ssp_type='linear',
                        ssp_data=[(d, 1500) for d in depths_m],
                        ssp_2d_ranges=ranges_km,
                        ssp_2d_matrix=ssp_2d)
```

### BoundaryProperties (surface or bottom)

```python
from uacpy import BoundaryProperties

BoundaryProperties(
    acoustic_type='vacuum',        # 'vacuum'|'rigid'|'half-space'|'grain-size'|'file'|'precalc'
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
    reflection_rmax_km=10.0,
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
    ranges_km    = np.array([0, 5, 10, 15]),
    depths       = np.array([100, 150, 200, 180]),
    sound_speed  = np.array([1600, 1650, 1700, 1750]),
    density      = np.array([1.5, 1.7, 1.9, 2.1]),
    attenuation  = np.array([0.8, 0.6, 0.4, 0.3]),
    shear_speed  = np.zeros(4),
    acoustic_type='half-space',
)
```

Bellhop will take the **median** of range-dependent bottom properties and
warn; RAM (mpiramS) handles this natively.

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
                        ssp_type='isovelocity', sound_speed=1500,
                        bottom=bottom)

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
    ranges_km=np.array([0, 20]),
    depths   =np.array([100, 300]),
    profiles =[near_profile, far_profile],   # each is a LayeredBottom
)

env = uacpy.Environment(name="shelf", depth=300, sound_speed=1500,
                        bottom=rdl)
```

Only RAM supports this natively (via its 4-point per-range sediment
profile). Everything else collapses it.

### Bathymetry & altimetry

```python
bathy = np.array([[0, 100], [5000, 150], [10000, 200]])  # (range_m, depth_m)
env = uacpy.Environment(name="slope", depth=200, sound_speed=1500,
                        bathymetry=bathy)

# Rough sea surface (Bellhop only — other models warn and ignore)
alt = uacpy.generate_sea_surface(length=10_000, rms_height=0.5, n_points=201)
env = uacpy.Environment(name="rough", depth=100, sound_speed=1500,
                        altimetry=alt)
```

### Environment introspection

```python
env.is_range_dependent                         # bathymetry or RD bottom present
env.has_range_dependent_ssp()
env.has_range_dependent_bottom()
env.has_layered_bottom()
env.has_range_dependent_layered_bottom()

# Collapse to range-independent
env_ri = env.get_range_independent_approximation(method='median')  # 'max'|'min'|'mean'
```

---

## 4. Source & Receiver

### Source

```python
uacpy.Source(
    depth,                      # float or array — positive, down from surface (m)
    frequency,                  # float or array — Hz
    position=(0.0, 0.0),        # (x, y) in m
    angles=None,                # launch angles (deg). Default: linspace(-80, 80, 361)
    source_type='point',        # 'point' | 'line' | 'array'
    beam_pattern='omni',        # 'omni' or a callable(angle_deg) -> amplitude
    power=0.0,                  # dB re 1 μPa @ 1 m
    phase=0.0,                  # radians
)
```

Useful properties: `source.n_sources`, `source.n_frequencies`,
`source.n_angles`, `source.wavelength` (uses 1500 m/s nominal — for accurate
wavelengths pass through the environment).

### Receiver

```python
uacpy.Receiver(
    depths=None,                # array or scalar (m)
    ranges=None,                # array or scalar (m)
    positions=None,             # list[(x,y)] or list[(x,y,z)] — overrides depths/ranges
    receiver_type='grid',       # 'grid' (meshgrid) | 'line' (paired) | 'point'
)
```

- `'grid'` is a full `depths × ranges` mesh (default).
- `'line'` pairs `depths[i]` with `ranges[i]` (same length or one scalar).
- `positions=[(x,y,z), ...]` is taken verbatim for 3D configurations.

Useful properties: `n_receivers`, `n_depths`, `n_ranges`, `depth_min/max`,
`range_min/max`, `get_positions()`, `get_cylindrical_positions()`,
`subset(depth_range=, range_range=)`.

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

| Model        | Coh TL | Incoh TL | Rays | Eigen | Arrivals | Modes | Time series | Altimetry | Notes |
|--------------|:------:|:--------:|:----:|:-----:|:--------:|:-----:|:-----------:|:---------:|-------|
| Bellhop      | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ | Ray/beam tracing |
| BellhopCUDA  | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ | C++/CUDA port |
| Kraken       | — | — | — | — | — | ✓ | — | — | Real modes |
| KrakenC      | — | — | — | — | — | ✓ | — | — | Complex modes |
| KrakenField  | ✓ | — | — | — | — | — | ✓ | — | kraken→field pipeline |
| Scooter      | ✓ | — | — | — | — | — | ✓ | — | Wavenumber integration |
| RAM (mpiramS)| ✓ | — | — | — | — | — | ✓ | — | Split-step Padé PE |
| SPARC        | ✓ | — | — | — | — | — | ✓ | — | Time-domain PE |
| OAST         | ✓ | — | — | — | — | — | — | — | OASES TL |
| OASN         | — | — | — | — | — | ✓ | — | — | OASES modes (experimental) |
| OASR         | ✓* | — | — | — | — | — | — | — | OASES reflection (data in metadata) |
| OASP         | ✓ | — | — | — | — | — | ✓ | — | OASES PE |
| Bounce       | ✓* | — | — | — | — | — | — | — | Writes .brc; data in metadata |

`*` the model is declared `COHERENT_TL`-capable internally but its output
is a reflection coefficient table, not a TL grid.

### 5.2 Environment feature support

| Environment feature             | Bellhop | RAM | Kraken/Field | Scooter | SPARC | OASES | Bounce |
|---------------------------------|:-------:|:---:|:------------:|:-------:|:-----:|:-----:|:------:|
| 1-D SSP                         | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2-D (range-dep) SSP             | ✓ | ✓ | warn+approx | warn | warn | warn | — |
| Range-dep bathymetry            | ✓ | ✓ | warn+approx | warn | warn | — | — |
| Halfspace bottom                | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `RangeDependentBottom`          | warn+median | **native** | warn+approx | warn | warn | warn | — |
| `LayeredBottom` (multi-layer)   | warn+halfspace | warn+halfspace | **native** | **native** | **native** | **native** | — |
| `RangeDependentLayeredBottom`   | warn+halfspace | **native** | warn+approx | warn | warn | warn | — |
| Elastic bottom (shear)          | via BOUNCE | — | ✓ (KrakenC) | ✓ | ✓ | ✓ | ✓ |
| Reflection file (.brc)          | ✓ | — | ✓ | ✓ | — | — | output |
| Altimetry (rough surface)       | ✓ | — | — | — | — | — | — |

### 5.3 Bellhop — ray/beam tracing

```python
from uacpy.models import Bellhop

bh = Bellhop(
    executable=None,                 # auto-detected
    prefer_cuda=True,                # pick BellhopCUDA/cxx if present
    beam_type='B',                   # see beam code table below
    n_beams=0,                       # 0 = use source.n_angles
    alpha=(-80, 80),                 # launch-angle limits (deg)
    step=0.0,                        # ray step (m); 0 = auto
    z_box=None,                      # max trace depth (m); None = 1.2 × depth
    r_box=None,                      # max trace range (m); None = 1.2 × max range
    source_type='R',                 # 'R' (point) | 'X' (line)
    grid_type='R',                   # 'R' (rectilinear) | 'I' (irregular)
    beam_shift=False,
    volume_attenuation=None,         # None | 'T' (Thorp) | 'F' (Francois-Garrison) | 'B' (Biological)
    use_tmpfs=False, verbose=False, work_dir=None,
)

field = bh.run(env, source, receiver, run_type='C', **per_call_overrides)
```

**`run_type` (Bellhop-native letter code, passed through to Fortran):**

| Code | Meaning                                 |
|:----:|-----------------------------------------|
| `C`  | Coherent TL (default)                   |
| `I`  | Incoherent TL                           |
| `S`  | Semi-coherent (Lloyd mirror only)       |
| `R`  | Ray trace                               |
| `E`  | Eigenrays                               |
| `A`  | Arrivals (ASCII)                        |
| `a`  | Arrivals (binary)                       |

**`beam_type`:** `'B'` Gaussian (default), `'R'` ray-centered,
`'C'` Cartesian, `'b'` geometric Gaussian, `'g'` geometric hat (ray),
`'G'` geometric hat (Cartesian), `'S'` simple Gaussian.

**Bellhop + BOUNCE for elastic bottoms.** When the bottom has significant
shear, run BOUNCE first to produce a reflection coefficient file, then feed
it to Bellhop:

```python
field = bh.run_with_bounce(
    env, source, receiver,
    run_type='C',
    cmin=1400.0, cmax=10000.0, rmax_km=10.0,
)
```

### 5.4 BellhopCUDA — GPU ray tracing

C++/CUDA port of BELLHOP, same run types as Bellhop. API is nearly
identical; the key difference is `use_gpu` and the auto-selected
executable:

```python
from uacpy.models import BellhopCUDA

bhc = BellhopCUDA(
    use_gpu=True,             # False → use bellhopcxx (CPU) binary
    beam_type='R',            # default differs from Bellhop Fortran ('B')
    n_beams=0, alpha=(-80, 80),
    step=0.0, z_box=None, r_box=None,
    source_type='R', grid_type='R',
    beam_shift=False, volume_attenuation=None,
    dimensionality='2D',      # '2D' only in current build
    use_tmpfs=False, verbose=False,
)
field = bhc.run(env, source, receiver, run_type='C')
```

Requires CUDA toolkit; falls back to the CPU `bellhopcxx` build if `use_gpu=False`.

### 5.5 Kraken & KrakenC — normal modes

Both inherit from the same base (`_KrakenBase`) and only differ in which
executable they call — Kraken (real arithmetic) vs KrakenC (complex,
required for elastic/attenuating bottoms).

```python
from uacpy.models import Kraken, KrakenC

kr = Kraken(
    c_low=None,                # m/s — None auto: 0.95 × min SSP speed
    c_high=None,               # m/s — None auto: 1.05 × max SSP/bottom speed
    n_mesh=0,                  # mesh points per wavelength (0 = auto)
    roughness=0.0,             # bottom RMS (m)
    volume_attenuation=None,   # 'T' | 'F' | 'B' | None
    use_tmpfs=False, verbose=False, work_dir=None,
)
modes = kr.run(env, source, receiver, n_modes=None)
# modes is a Field(field_type='modes')
# modes.metadata['k'], modes.metadata['phi'], modes.metadata['z']

krc = KrakenC(...)               # identical signature
modes = krc.run(env_with_shear_bottom, source, receiver)
```

The environment must be range-independent. For range-dependent modal
propagation use `KrakenField`.

### 5.6 KrakenField — modes + field pipeline

Runs `kraken.exe` (or `krakenc.exe` if the bottom is elastic), then
`field.exe`. Produces TL grids using adiabatic or coupled-mode theory.

```python
from uacpy.models import KrakenField

kf = KrakenField(
    mode_points_per_meter=1.5,
    mode_coupling='adiabatic',   # 'adiabatic' | 'coupled'
    coherent=True,
    n_segments=10,
)
field = kf.run(env, source, receiver, run_mode=None)   # default COHERENT_TL
```

Range-dependent environments are supported via segmentation.

### 5.7 Scooter — wavenumber integration (fast field)

```python
from uacpy.models import Scooter

sc = Scooter(
    c_low=None, c_high=None,
    n_mesh=0, roughness=0.0,
    rmax_multiplier=2.0,         # multiply max receiver range for k-resolution
    volume_attenuation=None,
)
field = sc.run(env, source, receiver)
```

Range-independent only. Supports `LayeredBottom`.

### 5.8 RAM (mpiramS) — parabolic equation

Split-step Padé PE; the backend is the modified `s_mpiram` Fortran binary.

```python
from uacpy.models import RAM

ram = RAM(
    dr=None,                    # range step (m); None auto ≈ 1 wavelength, capped at 500
    dz=0.5,                     # depth step (m)
    zmax=None,                  # PE-domain depth (m); None auto = seafloor + absorbing layer
    np_pade=6,                  # Padé coefficients (2–8)
    ns_stability=1,
    rs_stability=None,
    Q=2.0,                      # broadband bandwidth f_c / Q
    T=10.0,                     # time-window width (s)
    depth_decimation=1,
    flat_earth=True,            # apply flat-earth transformation
    absorbing_layer_width=20.0, # wavelengths below seafloor
    absorbing_layer_attn=10.0,  # dB/λ at domain floor
    n_sed_points=50,            # sediment profile control points
)
field = ram.run(env, source, receiver)
```

RAM is the only model with native `RangeDependentLayeredBottom` support.

**TL formula inside UACPY's RAM wrapper:**
`TL = -20·log10(|psif| · 4π) + 10·log10(r)`. Keep this in mind when
comparing against other implementations.

### 5.9 SPARC — time-domain PE

```python
from uacpy.models import SPARC

sp = SPARC(
    c_low=None, c_high=None,
    n_mesh=0, roughness=0.0,
    output_mode='R',            # 'R' horizontal array | 'D' vertical array | 'S' snapshot
    pulse_type='PN+B',
    n_t_out=501,
    t_max=None,                 # None auto: 2.5 × travel time
    t_start=-0.1,
    t_mult=0.999,
    max_depths=20,              # warning threshold
    volume_attenuation=None,
)
field = sp.run(env, source, receiver)
```

Output is time-series. Not a coherent-TL model in the usual sense.

### 5.10 Bounce — reflection coefficients

```python
from uacpy.models import Bounce

bn = Bounce(
    cmin=1400.0, cmax=10000.0,  # phase-velocity bounds for tabulation (m/s)
    rmax_km=10.0,               # max range used to derive angular sampling
    volume_attenuation=None,
)
field = bn.run(env, source, receiver, output_brc=True, output_irc=True)

# Reflection coefficients + path to .brc are in field.metadata
```

Use the resulting `.brc` file via `BoundaryProperties(acoustic_type='file',
reflection_file=...)` when feeding Kraken, Scooter, or Bellhop with an
elastic-bottom reflection table.

### 5.11 OASES — OAST / OASN / OASR / OASP

Individual model classes mirror the OASES Fortran utilities:

```python
from uacpy.models import OAST, OASN, OASR, OASP

# Transmission loss (wavenumber integration)
oast = OAST(volume_attenuation=None)
field = oast.run(env, source, receiver)

# Normal modes (experimental — prefer Kraken for production)
oasn = OASN(volume_attenuation=None)
modes = oasn.run(env, source, receiver, n_modes=None)

# Reflection coefficients
oasr = OASR(
    angles=None,                   # default linspace(0, 90, 181)
    angle_type='grazing',          # 'grazing' | 'incidence'
    volume_attenuation=None,
)
refl = oasr.run(env, source, receiver)     # field_type='reflection_coefficients'

# Parabolic equation (broadband)
oasp = OASP(
    n_time_samples=4096,
    freq_max=250.0,
    volume_attenuation=None,
)
field = oasp.run(env, source, receiver)
```

OAST warns and uses maximum bathymetry depth for range-dependent cases —
for proper range-dependent OASES runs, use OASP.

### 5.12 OASES unified façade

```python
from uacpy.models import OASES
from uacpy.models.base import RunMode

oases = OASES(use_tmpfs=False, verbose=False)

field  = oases.compute_tl(env, source, receiver)          # → OAST
modes  = oases.compute_modes(env, source, n_modes=30)     # → OASN
field  = oases.run(env, source, receiver,
                   run_mode=RunMode.COHERENT_TL,
                   use_pe=True)                           # RD + use_pe → OASP
```

For maximum control use the individual classes; the façade is for
convenience.

---

## 6. Field & Results

`Field` is the uniform result object returned by every `.run()`.

```python
Field(
    field_type,   # 'tl' | 'pressure' | 'rays' | 'arrivals' | 'modes' |
                  # 'eigenrays' | 'reflection_coefficients' | 'transfer_function' |
                  # 'time_series'
    data,         # main numeric array — semantics depend on field_type
    ranges=None,  # ndarray (m)  — for TL grids etc.
    depths=None,  # ndarray (m)
    frequencies=None,
    metadata={},  # model-specific extras
)
```

Common access patterns:

```python
# TL grid
tl  = field.data            # shape (n_depths, n_ranges), dB
rngs = field.ranges          # m
zs   = field.depths          # m

# Rays (Bellhop, run_type='R' or compute_rays)
rays = field.data            # list of ray tuples
                             # each: (range_m[N], depth_m[N], n_top_bounces, n_bot_bounces, …)

# Arrivals (Bellhop, run_type='A' or compute_arrivals)
arrivals = field.data        # structured per-receiver arrival lists

# Modes (Kraken/OASN)
k     = field.metadata['k']     # complex wavenumbers
phi   = field.metadata['phi']   # mode shapes
z     = field.metadata['z']     # depth grid

# Reflection coefficients (OASR, Bounce)
metadata = field.metadata    # 'angles', 'reflection_coeff', 'brc_file', etc.

# Time series (SPARC / RAM / OASP with TIME_SERIES)
ts = field.data              # shape (n_receivers, n_time_samples)
```

---

## 7. Visualization

All plotting functions live in `uacpy.visualization` (also exposed as
`uacpy.plot`). They accept a `Field` and optionally an `Environment` and
return matplotlib objects so you can further customize.

### Transmission loss

```python
from uacpy.visualization import plot_transmission_loss

fig, ax, im = plot_transmission_loss(
    field, env=env,
    show_bathymetry=True,
    show_ssp=False,
    show_colorbar=True,
    tl_min=None, tl_max=None,
    contours=None, contour_labels=True,
    cmap='jet_r',               # default: blue = low loss
    figsize=(12, 6), ax=None,
)
```

Polar variant: `plot_transmission_loss_polar(field, ...)`.

### Rays

```python
from uacpy.visualization import plot_rays

fig, ax = plot_rays(
    field, env=env,
    color_by_bounces=True,    # red=direct, green=surface, blue=bottom, black=both
    ray_colors=None,          # dict override
    max_rays=1000,
    linewidth=0.5, alpha=0.7,
    figsize=(12, 6),
)
```

### Modes

```python
from uacpy.visualization import plot_modes, plot_mode_functions, \
    plot_modes_heatmap, plot_mode_wavenumbers, plot_dispersion_curves

plot_modes(field, mode_numbers=[1, 2, 3, 5, 10], normalize=True)
plot_mode_wavenumbers(field)
plot_dispersion_curves(...)
```

### Environment / bathymetry / SSP

```python
from uacpy.visualization import (
    plot_environment, plot_environment_advanced,
    plot_ssp, plot_ssp_2d, plot_bathymetry,
    plot_bottom_properties, plot_layered_bottom, plot_rd_layered_bottom,
)

plot_environment_advanced(env, source=source, receiver=receiver,
                          show_grid=True, show_ssp=True)

# Requires the corresponding env.has_*() predicate
plot_bottom_properties(env)      # RangeDependentBottom
plot_layered_bottom(env)         # LayeredBottom
plot_rd_layered_bottom(env)      # RangeDependentLayeredBottom
plot_ssp_2d(env)                 # range-dependent SSP heatmap
```

### Other result types

```python
from uacpy.visualization import (
    plot_arrivals, plot_time_series, plot_reflection_coefficient,
    plot_transfer_function, plot_range_cut, plot_depth_cut,
)
```

### Comparison helpers

```python
from uacpy.visualization import (
    compare_models, compare_range_cuts,
    plot_model_statistics, plot_model_comparison_matrix,
    plot_comparison_curves,
)

compare_models([('Bellhop', f_bh), ('RAM', f_ram), ('Kraken', f_kr)],
               env=env, layout='grid', share_colorbar=True)
```

### Quick plotting

For one-liners during exploration:

```python
from uacpy.visualization import quickplot

quickplot.tl(field)
quickplot.rays(field)
# etc.
```

---

## 8. Signal Processing

Imported as `uacpy.signal` at the top level (the package is actually named
`acoustic_signal` on disk to avoid shadowing Python's stdlib `signal`).

### Waveform generation (`uacpy.signal.generation`)

```python
import uacpy.signal as sig

# Continuous-wave pulse
s = sig.cw_pulse(fc, fs, duration, window=None)

# Linear-frequency chirp
s = sig.lfm_chirp(fs, duration, f_start, f_end, window=None)

# Hyperbolic-frequency chirp
s = sig.hfm_chirp(fs, duration, f_start, f_end, window=None)

# Digital modulations
s = sig.mfsk_signal(symbols, fs, symbol_rate, f0, df)
s = sig.psk_signal(symbols, fs, symbol_rate, fc, M=2)

# Noise-like and composite signals
s = sig.noise_signal(fs, duration, bandwidth=None)
s = sig.composite_signal([...])

# Spectral Synthesis of Random Processes — generate a time series realising
# a target PSD (useful for feeding noise models into time-domain simulators).
t, x, fs = sig.ssrp(Pxx, Fxx, duration=1.0, scale=1.0)
```

### Processing (`uacpy.signal.processing`)

```python
# Plane-wave array replicas
rep = sig.planewave_rep(array_geometry, f, c, direction_deg)

# Beamforming (method-agnostic Bartlett/Capon/MUSIC selector inside)
out = sig.beamform(array_data, sensor_depths, angles_deg,
                   frequency=100.0, sound_speed=1500.0, method='bartlett')

# Add coloured/white noise at a target SNR
y = sig.add_noise(signal, snr_db=10)

# Bandlimited noise
n = sig.make_bandlimited_noise(fs, duration, f_low, f_high)

# Fourier-synthesize a broadband response from frequency samples
t, s = sig.fourier_synthesis(freqs, complex_amplitudes, fs)
```

### Advanced (`uacpy.signal.advanced` — adapted from arlpy)

```python
# Time vectors / CW / sweep helpers
sig.time(n, fs); sig.cw(fc, duration, fs); sig.sweep(f1, f2, duration, fs)

# Baseband/passband conversion
bb = sig.bb2pb(x, fs, fc); pb = sig.pb2bb(x, fs, fc)

# Zero-phase filtering, periodic cross-correlation
y = sig.lfilter0(b, a, x)
r = sig.correlate_periodic(x, y)

# Single-bin DFT (Goertzel), numerically-controlled oscillators, m-sequences
sig.goertzel(x, fc, fs); sig.nco(...); sig.nco_gen(...); sig.lfilter_gen(...)
sig.mseq(n)
sig.resample(x, p, q)
```

### Spectral analysis (`uacpy.signal.analysis`)

Class-based API — each estimator is instantiated with reference level and
(for Welch-based tools) spectral-estimation parameters, then `.compute(data, fs)`
is called, and finally `.plot(...)` renders the result.

```python
from uacpy.signal.analysis import PSD, PPSD, Spectrogram, SEL, FRF, FKTransform

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
# Estimate a transfer function x -> y via Welch H1/H2/Hv, ETFE, periodic
# ETFE, or least-squares FIR; with impulse response and coherence.
frf = FRF(method="welch", estimator="H1", m=512, nperseg=4096)
frf.compute(x, y, fs)                         # 1-D or 2-D (multi-measurement)
fig, ax = frf.plot(title="Channel", ymin=-60, ymax=60)
frf.plot_coh(title="Coherence")
frf.plot_impulse_info(title="LSFIR diagnostics")  # for method="lsfir"

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

```python
from uacpy.noise import AmbientNoiseSimulator

sim = AmbientNoiseSimulator(freq=np.logspace(0, 5, 1000))   # 1 Hz – 100 kHz

# Add contributions. Each helper takes a model name + free-form kwargs.
sim.add_wind('knudsen', wind_speed=10)
sim.add_shipping('wenz', shipping_level=5)
sim.add_rain('ma_nystuen', rainfall=10)       # mm/hr
sim.add_turbulence('wenz')
sim.add_thermal('mellen')
sim.add_biological('snapping_shrimp')
sim.add_seismic('webb')
sim.add_explosion('chapman', charge_kg=1, range_km=5)

# Global parameters passed to every model that accepts them
sim.set_global_params(wind_speed=10)

components, total = sim.compute()
#   components : {label: ndarray}  — per-model spectra (dB re 1 μPa²/Hz)
#   total      : ndarray           — incoherent sum

fig, ax = sim.plot(title="Ambient noise", show_total=True)
```

Exact sets of `name` strings available per category are registered in
`uacpy.noise.noise.MODEL_REGISTRY`. Each model documents its own
parameters.

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

| Model         | Input file expects | UACPY converts? |
|---------------|--------------------|:---------------:|
| Bellhop `.env`| ranges in km       | ✓ (auto)        |
| Bellhop `.bty`| ranges in km       | ✓               |
| KrakenField `.flp` | ranges in km  | ✓               |
| RAM           | meters throughout  | —               |
| OASES         | meters throughout  | —               |

You always pass **meters** to `Receiver(ranges=…)` and
`bathymetry=[[range_m, depth_m], …]`.

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

**`RangeDependentBottom` in Bellhop silently medians the properties.** This
is by design but produces a warning. If you need true range-dependent
geoacoustics, use RAM.

**`LayeredBottom` in RAM/Bellhop collapses to a halfspace.** Use Kraken,
Scooter, SPARC, or OASES for layered sediment.

**OASN modes return empty.** Mode extraction from the OASES XSM format is
experimental. Use Kraken/KrakenC for production modal analysis.

**Bellhop ray box too small.** Increase `r_box` / `z_box` or let them
default (1.2 × receiver extent).

**Kraken not converging.** Raise `n_modes` or widen `c_low`/`c_high` — the
defaults (0.95 × min SSP, 1.05 × max SSP/bottom) sometimes miss high-speed
bottom-trapped modes.

**TL contains NaNs at long range (RAM).** Check `zmax` and
`absorbing_layer_width`; spurious reflections from the PE domain bottom
commonly show up as NaNs. Increasing `absorbing_layer_width` to ~30–40
wavelengths usually fixes it.

**Bellhop time-series pipeline.** The arrivals → waveform convolution step
is not yet wired up inside UACPY. For time-domain propagation use SPARC
(native time-domain PE) or OASP. To work with Bellhop arrivals, request
`run_type='A'` and post-process the returned arrivals yourself.

**Elastic bottoms in Bellhop.** Bellhop is a fluid-only ray tracer. Use
`Bellhop.run_with_bounce()` or route through BOUNCE → `.brc` →
Kraken/Scooter.

---

## 12. Examples Index

All scripts live in `uacpy/uacpy/examples/` and are runnable as-is once
`./install.sh` has completed.

| # | File | Demonstrates |
|---|------|--------------|
| 01 | `example_01_basic_shallow_water.py`          | Minimal TL — start here |
| 02 | `example_02_sound_speed_profiles.py`         | SSP types (linear, Munk, cubic, …) |
| 03 | `example_03_range_dependent_bathymetry.py`   | Sloping / varying bottoms |
| 04 | `example_04_complex_bathymetry.py`           | Complex seamount / canyon |
| 05 | `example_05_range_dependent_full.py`         | Combined RD scenario |
| 06 | `example_06_multi_frequency.py`              | Sweep over frequencies |
| 07 | `example_07_bellhop_advanced.py`             | Beam types, eigenrays, arrivals |
| 08 | `example_08_ram_advanced.py`                 | RAM PE tuning parameters |
| 09 | `example_09_kraken_advanced.py`              | Modal analysis with Kraken |
| 10 | `example_10_thermal_fronts.py`               | Oceanographic fronts |
| 11 | `example_11_all_models_comparison.py`        | All models side by side |
| 12 | `example_12_maximum_complexity.py`           | Stress test |
| 13 | `example_13_long_range.py`                   | Convergence-zone propagation |
| 14 | `example_14_ambient_noise.py`                | `AmbientNoiseSimulator` |
| 15 | `example_15_signal_processing.py`            | CW, chirps, matched filtering |
| 16 | `example_16_bellhop_run_modes.py`            | Every Bellhop run type |
| 17 | `example_17_attenuation_models.py`           | Thorp / Francois-Garrison / Biological |
| 18 | `example_18_boundary_conditions.py`          | Vacuum / rigid / halfspace |
| 19 | `example_19_oases_suite.py`                  | OAST / OASN / OASR / OASP |
| 20 | `example_20_sparc_time_domain.py`            | SPARC seismo-acoustic PE |
| 21 | `example_21_plotting_features.py`            | Basic plotting tour |
| 22 | `example_22_new_plotting_features.py`        | Advanced visualization |
| 23 | `example_23_bounce_reflection_coefficients.py` | Bounce `.brc` output |
| 24 | `example_24_elastic_boundaries_comparison.py` | Fluid vs elastic bottoms |
| 25 | `example_25_bellhop_bounce_integration.py`   | `Bellhop.run_with_bounce()` |
| 25b| `example_25_broadband_comparison.py`         | Broadband propagation |
| 26 | `example_26_boundary_conditions_layered.py`  | `LayeredBottom` across models |
| 27 | `example_27_rd_bottom_krakenfield_vs_ram.py` | RD bottom: KrakenField vs RAM |

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
