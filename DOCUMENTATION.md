# UACPY Documentation

Underwater acoustic propagation for Python — concepts, API surface, and
the gotchas that aren't obvious from the docstrings. Per-class detail
lives on the constructor docstrings (`help(Bellhop)`, `help(KrakenField)`,
…); this doc is the **reference for shapes, conventions, model
selection, and cross-cutting behaviour**.

> **Status: Beta.** APIs are stable; signatures and defaults reflect the
> current code. Source is the ground truth for anything not covered here.

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

```bash
git clone --recurse-submodules https://github.com/ErVuL/uacpy.git
cd uacpy
python -m venv uacpy_venv && source uacpy_venv/bin/activate
pip install -e .
./install.sh                  # compile Fortran/C/CUDA binaries into uacpy/bin/
```

`--recurse-submodules` is required (bellhopcuda is vendored as a
submodule). OASES is fetched from MIT on demand by `install.sh` —
it is not redistributable.

```python
import numpy as np, matplotlib.pyplot as plt
import uacpy
from uacpy.models import Bellhop
from uacpy.visualization import plot_transmission_loss

env      = uacpy.Environment(name='shallow', bathymetry=100.0, ssp=1500.0)
source   = uacpy.Source(depths=50.0, frequencies=100.0)
receiver = uacpy.Receiver(depths=np.linspace(0, 100, 101),
                          ranges=np.linspace(100, 10_000, 200))

field = Bellhop().compute_tl(env, source, receiver)
plot_transmission_loss(field, env=env)
plt.show()
```

The full public surface lives on `uacpy.*` — `dir(uacpy)` is the index.
Models are in `uacpy.models`, visualization in `uacpy.visualization`,
signal processing in `uacpy.signal` (alias for the on-disk
`uacpy.acoustic_signal` package — pulled in to dodge stdlib `signal`).

---

## 2. Core Concepts

```
Environment + Source + Receiver  →  Model.run()  →  Result
```

- **`Environment`** — water column: bathymetry, SSP, surface, bottom.
- **`Source`** — depth(s), frequency(ies), launch angles, beam pattern.
- **`Receiver`** — grid or paired line of hydrophones.
- **`Result`** — concrete typed subclass, one per output kind (see §6).

### The shared model surface

Every model inherits from `PropagationModel` (`uacpy.models.base`):

| Method | Purpose |
|---|---|
| `run(env, src, rcv, run_mode=…, **kw)` | Full-control entry point. |
| `compute_tl / rays / eigenrays / modes / arrivals(...)` | Convenience wrappers — forward `**kw` to `run()`. |
| `supports_mode(RunMode.X)` / `supported_modes` | Capability check. |

Convenience wrappers raise `UnsupportedFeatureError` when the underlying
solver lacks the mode (e.g. `Bounce.compute_tl()` raises — Bounce only
emits reflection coefficients).

### `RunMode` (`uacpy.models.base.RunMode`)

| Mode | Purpose |
|---|---|
| `COHERENT_TL` / `INCOHERENT_TL` / `SEMICOHERENT_TL` | TL field |
| `RAYS` / `EIGENRAYS` / `ARRIVALS` | Bellhop ray products |
| `MODES` | Kraken / KrakenC eigenfunctions |
| `BROADBAND` | complex `H(f)` |
| `TIME_SERIES` | real `p(t)` |
| `COVARIANCE` / `REPLICA` | OASN array products |
| `REFLECTION` | Bounce / OASR plane-wave reflection coef. |

§5.1 lists which modes each model supports.

### Constructor surface (shared)

```python
Model(
    use_tmpfs=False,          # tmpfs scratch I/O (Linux, faster)
    verbose=False,            # bool | 'off'/'silent' | 'info' | 'debug'.
                              #   False ⇒ WARN+ERROR only; True/'info' ⇒ +INFO;
                              #   'debug' ⇒ +DEBUG (subprocess command lines,
                              #   per-frequency grid resolution, etc.)
    work_dir=None,            # pin scratch dir; None ⇒ temp dir
    cleanup=None,             # None ⇒ True iff uacpy owns the work dir
    timeout=600.0,            # subprocess timeout (s) per binary call
    collapse={                # see §5.2 — global defaults shown
        'bathymetry':        'max',
        'ssp':               'r0',
        'bottom':            'r0',
        'layered':           'halfspace',
        'rd_layered_range':  'median',
        'rd_layered_layers': 'halfspace',
        'altimetry':         'drop',
        'elastic':           'fluid',
    },
)
```

**Configuration is constructor-only.** Every tuning knob (beam_type, dr,
…) is set when you build the wrapper. To sweep parameters, construct one
model per parameter set. `run()` accepts only the standard
`(env, source, receiver, run_mode, frequencies, source_waveform,
sample_rate)` plus narrow per-model writer pass-through.

Volume absorption is environmental, not model-level: attach a
`Thorp()` / `FrancoisGarrison(...)` / `Biological(...)` /
`ConstantAbsorption(...)` to `env.absorption` once and every model
inspects it to write TopOpt(4) and the supporting per-formula lines.
See *Volume Absorption* below.

**Mode-specific kwargs.** Every `RunMode.TIME_SERIES`-capable wrapper
(Bellhop, Scooter, KrakenField, OASP, RAM) takes `source_waveform=` +
`sample_rate=` on `run()`. Models with a broadband path also accept
`frequencies=` for an explicit override of `source.frequencies`. SPARC
computes p(t) from its native source pulse and ignores both silently.

**Irrelevant kwargs are silently ignored** (per uacpy convention — same
as Python's `dict()` constructor with an unknown keyword). A typo like
`Bellhop().run(env, src, rcv, n_beam=10)` (missing the `s`) silently
uses the default `n_beams`.

### Persisting output files (`work_dir` + `cleanup`)

Every model writes its binary inputs and outputs into a **work
directory**. The same two constructor knobs govern persistence on
every model — there are no per-model special cases.

```python
# Default — uacpy creates a tempdir, runs the binary, wipes the dir.
# `result` carries the in-memory data; nothing on disk afterwards.
m = Model()
result = m.run(env, src, rcv)

# Pin the work dir — files persist there for inspection / chaining.
m = Model(work_dir='./out')             # cleanup defaults to False
result = m.run(env, src, rcv)
result.metadata['shd_file']             # './out/...shd' — valid path

# Pin AND clean — files exist briefly during run(), gone after.
m = Model(work_dir='./out', cleanup=True)
```

The rule:

| `work_dir` | `cleanup` default | `cleanup` effect |
|---|---|---|
| `None` (uacpy owns it) | `True` | tempdir created → wiped after `run()` |
| pinned by user | `False` | dir survives `run()` |

When `cleanup=False` (files persist), every model attaches the
on-disk paths of its primary outputs to `result.metadata` (`shd_file`,
`mod_file`, `brc_file`, `psif_file`, `prt_file`, … — see §6 for the
per-model key table). When `cleanup=True`, those keys are absent —
the absence is the signal that the dir was wiped. **Read with
`.get()`** so missing keys don't raise:

```python
brc = result.metadata.get('brc_file')   # None if cleanup wiped the dir
prt = result.metadata.get('prt_file')   # debug log path, when persisted
```

The most common reason to pin `work_dir` is **chaining models**:
`Bounce.run(...)` → `result.metadata['brc_file']` → consumer model's
`BoundaryProperties(acoustic_type='file', reflection_file=…)` (Bellhop /
Scooter / Kraken / KrakenC). Bellhop's auto-route through BOUNCE
handles its own work-dir lifecycle internally; the user never needs
to manage it.

### Status output

Status text from every uacpy module — models, writers, readers — flows
through a single `uacpy._log.log_message(source, message, *, verbose,
level)` helper. `WARN` / `ERROR` always print; `INFO` / `DEBUG` print
only when `verbose` opts in. Format:

```
[YYYY/MM/DD HH:MM:SS UTC] [LEVEL] [source] message
```

User-facing problems still go through `warnings.warn(UserWarning, ...)`
so `pytest.warns(...)`, `simplefilter('error')`, and
`@pytest.mark.filterwarnings(...)` keep working. uacpy installs a custom
`warnings.formatwarning` at import time that renders warnings in the
same `[ts] [WARN] [module:lineno] message` shape as `log_message`, so
your run-log reads uniformly without losing any of Python's warning
machinery.

### Typed exceptions (all inherit `UACPYError`)

| Exception | Raised when |
|---|---|
| `ConfigurationError` | bad input — out-of-range, wrong type, mode/freq mismatch |
| `UnsupportedFeatureError` | model can't honour the requested mode or env axis |
| `ExecutableNotFoundError` | binary missing at construction time |
| `ModelExecutionError` | binary ran but failed (non-zero exit, empty output). The captured `.prt` log tail is appended for AT models. |

### Thread safety

Model instances mutate `self.file_manager` per `run()` and are **not
safe to share across threads**. For sweeps, instantiate one model per
worker (or use `ProcessPoolExecutor`).

---

## 3. Environment

```python
uacpy.Environment(
    bathymetry,                 # float (flat) or (N,2) ndarray (range_m, depth_m)
    ssp = None,                 # SoundSpeedProfile or scalar (None ⇒ isovelocity)
    sound_speed = 1500.0,       # default speed when ssp=None
    altimetry = None,           # (N,2) sea-surface (range_m, height_m, +up)
    bottom = None,              # BoundaryProperties / RD / Layered / RDL
                                #   (default: fluid half-space c=1600, ρ=1.5, α=0.5)
    surface = None,             # BoundaryProperties (default: vacuum / pressure-release)
    absorption = None,          # Absorption: Thorp / FrancoisGarrison /
                                #   Biological / ConstantAbsorption (default: None)
    *, name = 'unnamed',
)
```

`env.depth` is a read-only property = `bathymetry[:,1].max()`.

### `SoundSpeedProfile`

Handles both 1-D and 2-D (range-dependent) forms in one class. Build via
the classmethod factories:

```python
from uacpy import SoundSpeedProfile

SoundSpeedProfile.from_isovelocity(depth_max=2000.0, sound_speed=1500.0)
SoundSpeedProfile.from_pairs([(0,1540),(50,1520),(200,1505)])
SoundSpeedProfile.from_munk(depth_max=5000)
SoundSpeedProfile.from_mackenzie(depths=z, temperature_c=T, salinity_psu=S)
SoundSpeedProfile.from_2d(depths=z, ranges=r, matrix=ssp_2d)
```

The SSP carries a **shape** declaration (env-level metadata): one of
`'measured'` (default), `'isovelocity'`, `'munk'`, `'analytic'`,
`'n2linear'`. The factory methods set this automatically
(``from_isovelocity`` → ``'isovelocity'``, ``from_munk`` → ``'munk'``,
others → ``'measured'``).

The **sample-connection scheme** is a model-level kwarg
``Model(interp_ssp='linear'|'pchip'|'cubic'|'quad'|'n2linear'|...)``.
Each AT-family wrapper exposes it; values map onto AT's ``TopOpt(1)``
character. When ``env.ssp.shape`` implies a code (``'munk'`` →
``'A'`` for Bellhop's native Munk path, ``'n2linear'`` → ``'N'``,
``'isovelocity'`` → ``'C'``, ``'analytic'`` → ``'A'``), the shape wins
over the model's ``interp_ssp``.

**Bellhop honours range-dependent SSP only when
``Bellhop(interp_ssp='quad')``** (writes the external ``.ssp`` file);
any other ``interp_ssp`` collapses the SSP to 1-D via the model's
``collapse['ssp']`` policy with one tailored warning.

Useful methods:
- `eval(range=…, depth=…, interp='linear'|'nearest')` — label-based
  interpolated read.
- `collapse(method)` — `r0`/`rmax`/`mean`/`median` → 1-D profile (used by
  the projection pipeline).
- `extend_to(z_max)` — bidirectional alignment to env depth (extends by
  constant extrapolation, or truncates with linear interpolation).

### `BoundaryProperties` (surface or bottom halfspace)

```python
BoundaryProperties(
    acoustic_type='vacuum',      # vacuum | rigid | half-space | grain-size | file
    sound_speed=1600.0,          # m/s (compressional)
    density=1.5,                 # g/cm³
    attenuation=0.5,             # dB/λ (compressional)
    shear_speed=0.0,             # 0 = fluid; >0 makes the boundary elastic
    shear_attenuation=0.0,       # dB/λ
    grain_size_phi=1.0,          # acoustic_type='grain-size'
    reflection_file=None,        # acoustic_type='file' → .brc / .trc path
    reflection_cmin=1400.0,      # tabulation bounds for reflection_file
    reflection_cmax=10000.0,
    reflection_rmax=10000.0,
    roughness=0.0,               # RMS (m)
)
```

### Materials catalog

`uacpy.materials` ships class-typical geoacoustic presets. Use them
directly via `from_preset`:

```python
import uacpy
uacpy.materials.list_materials()
# ['basalt','chalk','clay','granite','gravel','limestone','moraine','sand','silt']

bottom = uacpy.BoundaryProperties.from_preset('sand')
bottom = uacpy.BoundaryProperties.from_preset('sand', attenuation=0.5)  # tweak
layer  = uacpy.SedimentLayer.from_preset('silt', thickness=10.0)
column = uacpy.LayeredBottom.from_presets(
    layers=[('clay', 5), ('silt', 15, {'attenuation': 1.5}), ('sand', 30)],
    halfspace='limestone',
    halfspace_overrides={'attenuation': 0.05},
)
```

| Material | c_p (m/s) | c_s (m/s) | ρ (g/cm³) | α_p (dB/λ) | α_s (dB/λ) |
|---|---|---|---|---|---|
| clay | 1500 | 80 | 1.5 | 0.2 | 1.0 |
| silt | 1575 | 80 | 1.7 | 1.0 | 1.5 |
| sand | 1650 | 110 | 1.9 | 0.8 | 2.5 |
| gravel | 1800 | 180 | 2.0 | 0.6 | 1.5 |
| moraine | 1950 | 600 | 2.1 | 0.4 | 1.0 |
| chalk | 2400 | 1000 | 2.2 | 0.2 | 0.5 |
| limestone | 3000 | 1500 | 2.4 | 0.1 | 0.2 |
| basalt | 5250 | 2500 | 2.7 | 0.1 | 0.2 |
| granite | 5500 | 3000 | 2.7 | 0.1 | 0.2 |

Values are class-typical. Shear speeds for unconsolidated sediments
grow with depth below the seabed (`c_s ≈ k·z^0.3`); the table reports
the `z = 1 m` value. `core.acoustics.bottom_loss_curve(name)` returns
`(grazing_angles_deg, loss_db)` for plane-wave fluid–fluid Rayleigh
reflection.

### Range-dependent and layered bottoms

uacpy carries four bottom flavours, all stored on `env.bottom`:

| Class | What it represents |
|---|---|
| `BoundaryProperties` | flat halfspace (the default). |
| `RangeDependentBottom` | per-range halfspace properties (interpolated). |
| `LayeredBottom` | depth-dependent stack of `SedimentLayer`s + halfspace. |
| `RangeDependentLayeredBottom` | per-range list of `LayeredBottom` profiles. |

Predicates: `env.has_layered_bottom()`,
`env.has_range_dependent_bottom()`,
`env.has_range_dependent_layered_bottom()`,
`env.has_elastic_bottom()`, `env.has_elastic_surface()`.

Slicing: `env.bottom_at_range(r)` returns `LayeredBottom` for RDLB,
`BoundaryProperties` otherwise. `env.halfspace_at_range(r)` always
returns a flat `BoundaryProperties` (digs into `.halfspace` for layered
structures) — used by env-file writers that emit a single bottom row.

```python
from uacpy import RangeDependentBottom, LayeredBottom, SedimentLayer, RangeDependentLayeredBottom

# RD halfspace — geoacoustics that vary with range
bot_rd = RangeDependentBottom(
    ranges=np.array([0, 5000, 10000, 15000]),       # metres
    sound_speed=np.array([1600, 1650, 1700, 1750]),
    density=np.array([1.5, 1.7, 1.9, 2.1]),
    attenuation=np.array([0.8, 0.6, 0.4, 0.3]),
    shear_speed=np.zeros(4),
    acoustic_type='half-space',
)

# Stratified column (depth-dependent)
bot_layered = LayeredBottom(
    layers=[
        SedimentLayer(thickness=5,  sound_speed=1550, density=1.5, attenuation=0.3),
        SedimentLayer(thickness=20, sound_speed=1800, density=2.0, attenuation=0.8),
    ],
    halfspace=uacpy.BoundaryProperties(acoustic_type='half-space',
                                       sound_speed=2000, density=2.2, attenuation=0.1),
)

# RD layered — list of profiles along range
rdlb = RangeDependentLayeredBottom(
    ranges=np.array([0, 20000]),
    profiles=[near_layered, far_layered],
)
```

The bathymetry lives on `env.bathymetry` regardless of bottom flavour;
RDLB only carries the sediment-stack range axis. See §5.2 for which
model honours which flavour natively.

### Bathymetry & altimetry

```python
bathy = np.array([[0, 100], [5000, 150], [10000, 200]])   # (range_m, depth_m)
env = uacpy.Environment(name='slope', ssp=1500, bathymetry=bathy)

# Rough sea surface (Bellhop and RAM ramsurf only — others drop it)
alt = uacpy.generate_sea_surface(max_range=10_000, wind_speed_ms=10.0,
                                 n_points=500, seed=0xACED)
env = uacpy.Environment(name='rough', bathymetry=100, ssp=1500, altimetry=alt)
```

Per-feature collapse is the canonical path: pass `collapse={…}` on the
model constructor (see §5.2). uacpy auto-collapses any axis the chosen
model can't honour natively, with one `UserWarning` per axis.

### Volume absorption

Water-column absorption is environmental: attach an `Absorption`
subclass to `env.absorption`. Every AT-family model reads it to set
TopOpt position 4 and write the supporting per-formula lines.

```python
from uacpy import (
    Thorp, FrancoisGarrison, Biological, BiologicalLayer,
    ConstantAbsorption,
)

env = uacpy.Environment(bathymetry=100, ssp=1500,
                        absorption=Thorp())                 # TopOpt(4)='T'

env = uacpy.Environment(bathymetry=100, ssp=1500,
                        absorption=FrancoisGarrison(        # TopOpt(4)='F'
                            temperature_c=10, salinity_psu=35,
                            pH=8.0, z_bar_m=100))

env = uacpy.Environment(bathymetry=100, ssp=1500,
                        absorption=Biological([             # TopOpt(4)='B'
                            BiologicalLayer(z_top_m=10, z_bottom_m=60,
                                            f0_hz=400.0, Q=5.0, a0=0.6),
                        ]))

env = uacpy.Environment(bathymetry=100, ssp=1500,
                        absorption=ConstantAbsorption(0.3)) # alphaI baseline
```

`None` (default) means no volume absorption. `ConstantAbsorption(v)`
writes the value into every SSP row's `alphaI` column (dB/wavelength) —
useful when you have a calibrated value that isn't Thorp/FG. The choice
is read by Bellhop, Kraken, KrakenC, KrakenField, Scooter, SPARC, Bounce,
OAST, OASN, OASR, OASP. RAM uses Q/T attenuation envelopes internally
and ignores this field.

For modal-perturbation post-processing, `Modes.with_attenuation` accepts
either `alpha_db_per_m=` (a scalar or `α(z)` array) or
`absorption=Thorp()` / `absorption=FrancoisGarrison(...)`. Other
Absorption subclasses are not supported by the modal kernel.

---

## 4. Source & Receiver

```python
uacpy.Source(
    depths,                # float or array — positive, down from surface (m)
    frequencies,           # float or array — Hz (multiple ⇒ broadband sweep)
    angles=None,           # launch angles (deg). Default linspace(-80, 80, 361)
    source_type='point',   # 'point' | 'line'
)

uacpy.Receiver(
    depths,                # required: array or scalar (m)
    ranges=None,           # array or scalar (m); default 0.0
    receiver_type='grid',  # 'grid' (meshgrid) | 'line' (paired)
)
```

- `'grid'` builds a full `depths × ranges` mesh.
- `'line'` pairs `depths[i]` with `ranges[i]` (same length, or one is
  scalar broadcast).
- A single hydrophone is just `Receiver(depths=[d], ranges=[r])` — there
  is no `'point'` receiver type.
- Time-series / broadband models always populate the full grid; extract
  a trace via `TransferFunction.to_time_trace(depth=, range=)` or
  `synthesize_time_series(depth=, range=)` on the returned typed result.

`Source(frequencies=[…])` with more than one frequency on a
single-frequency `RunMode` raises `ConfigurationError`. Multi-depth
`Source` raises on every wrapper except Bellhop (which natively writes
the full source-depth array).

---

## 5. Propagation Models

### 5.1 Capability matrix

| Model | Coh TL | Incoh TL | Rays | Eigen | Arr. | Modes | Time series | Trans. fn | Refl. | Altim. |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Bellhop | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | – | ✓ |
| BellhopCUDA | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | – | ✓ |
| Kraken | – | – | – | – | – | ✓ | – | – | – | – |
| KrakenC | – | – | – | – | – | ✓ | – | – | – | – |
| KrakenField | ✓ | – | – | – | – | – | ✓ | ✓ | – | – |
| Scooter | ✓ | – | – | – | – | – | ✓ | ✓ | – | – |
| RAM (mpiramS / rams0.5 / ramsurf1.5) | ✓ | – | – | – | – | – | ✓ | ✓ | – | ✓ |
| SPARC | ✓ | – | – | – | – | – | ✓ | – | – | – |
| OAST | ✓ | – | – | – | – | – | – | – | – | – |
| OASN | – | – | – | – | – | – | – | – | – | – |
| OASR | – | – | – | – | – | – | – | – | ✓ | – |
| OASP | ✓ | – | – | – | – | – | ✓ | ✓ | – | – |
| Bounce | – | – | – | – | – | – | – | – | ✓ | – |

OASN-only modes: `RunMode.COVARIANCE`, `RunMode.REPLICA`. `Time series`
is genuine time-domain output (SPARC, OASP) **or** synthesised from
arrivals (Bellhop) / a broadband transfer function (KrakenField,
Scooter, RAM).

### 5.2 Environment feature support + collapse

| Env feature | Bellhop | RAM | Kraken / KrakenC | KrakenField | Scooter | SPARC | OASES | Bounce |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1-D SSP | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2-D SSP | quad-only | mpiramS | warn | **segments** | warn | warn | warn | warn |
| RD bathymetry | ✓ | ✓ | warn | **segments** | warn | warn | warn | warn |
| `RangeDependentBottom` | **native** (long .bty) | mpiramS | warn | warn | warn | warn | warn | warn |
| `LayeredBottom` | warn → BOUNCE | **native** | **native** | **native** | **native** | **native** | **native** | **native** |
| `RangeDependentLayeredBottom` | warn → BOUNCE | **native** (mpiramS) | warn | warn | warn | warn | warn | warn |
| Elastic bottom | auto → BOUNCE | auto → rams0.5 | Kraken→KrakenC; KrakenC native | krakenc.exe | ✓ | warn → rigid | ✓ | ✓ |
| Reflection file (.brc / .trc) | ✓ | – | ✓ | ✓ | ✓ | – | – | **output** |
| Altimetry | ✓ | auto → ramsurf1.5 | – | – | – | – | – | – |

Each model declares its env-shape capability via 7 boolean flags
(`_supports_*`) on `__init__`; `_project_environment` walks every axis
on `run()` and reduces unsupported ones via the matching collapse key.

**Collapse keys (one per axis):**

| Key | Values | Default | What it does |
|---|---|---|---|
| `bathymetry` | `max\|median\|mean\|min\|initial` | `max` | RD bathy → flat depth |
| `ssp` | `r0\|rmax\|mean\|median` | `r0` | 2-D SSP → 1-D |
| `bottom` | `r0\|rmax\|mean\|median` | `r0` | RD halfspace → flat halfspace |
| `layered` | `halfspace\|top_layer\|volume_average` | `halfspace` | layer stack → halfspace |
| `rd_layered_range` | `r0\|rmax\|median` | `median` | RDLB → which range to sample |
| `rd_layered_layers` | `preserve\|halfspace\|top_layer\|volume_average` | `halfspace` | flatten layers (or `preserve` to keep them — requires layered support) |
| `altimetry` | `drop` | `drop` | flat surface |
| `elastic` | `fluid\|vacuum` | `fluid` | drop shear, keep cp/ρ/α |

**Per-model defaults override the global table.** Single-shot
range-independent solvers (Kraken/C, Scooter, SPARC, OAST, OASN, OASP,
Bounce) default to `'bottom': 'median'` and `'rd_layered_layers':
'preserve'` (when they natively consume `LayeredBottom`); the
single-spectrum solvers also default `'ssp': 'mean'`. KrakenField
defaults `'bottom': 'median'` and `'rd_layered_layers': 'preserve'`
since it segments bathy/SSP natively but still needs to collapse the
bottom axis. Bellhop, BellhopCUDA, OASR (SSP irrelevant), and RAM use
the global defaults.

`Bellhop.run(env, …)` **auto-routes through BOUNCE** for any env with a
`LayeredBottom`, `RangeDependentLayeredBottom`, elastic halfspace, or
`RangeDependentBottom` with non-zero `shear_speed` anywhere along range
(emits one `UserWarning`). The user's `Bellhop(collapse={…})` dict is
forwarded to the spawned Bounce, so the same constructor controls how a
range-dependent env reduces to BOUNCE's single BRC. The Bellhop result
carries the in-memory bounce result on
`result.metadata['bounce_result']` (a typed `ReflectionCoefficient`
with `.theta`/`.R`/`.phi`) so the user can inspect or plot the BRC
without re-running BOUNCE. Use `Bellhop.run_with_bounce(...)` for
explicit control over BOUNCE parameters (`c_low`, `c_high`, `rmax`),
or pass `auto_bounce=False` to skip the auto-route entirely.

### 5.3 Bellhop / BellhopCUDA — Gaussian beam / ray tracing

Ray/beam tracer (`bellhop.exe`); `BellhopCUDA` is the CUDA / C++ port
auto-picked by `Bellhop(prefer_cuda=True)`. Supports every ray product
and broadband synthesis via the arrivals → `H(f)` → IFFT pipeline.

```python
from uacpy.models import Bellhop, RunMode
bh = Bellhop(beam_type='B',          # B Gaussian | R ray-centered | C Cartesian | g/G geometric
             alpha=(-80, 80),         # launch-angle limits (deg)
             arrivals_format='ascii') # 'binary' (Fortran unformatted) is not parseable
field = bh.run(env, source, receiver, run_mode=RunMode.COHERENT_TL)

# Time series: per-receiver delay-and-sum via arrivals
ts = bh.run(env, source, receiver,
            run_mode=RunMode.TIME_SERIES,
            source_waveform=s, sample_rate=fs)
trace = ts.get_trace(depth=50.0, range=2000.0)        # → TimeTrace
```

**Caveats:**
- Range-dependent SSP is honoured only when `Bellhop(interp_ssp='quad')`;
  any other ``interp_ssp`` collapses the 2-D profile to 1-D via
  ``collapse['ssp']`` with a tailored warning.
- Layered / RDLB / elastic bottoms auto-route through BOUNCE (§5.2).
- `Bellhop3D` is a stub — the env writer is 2-D only, 3-D arrivals
  parsing is not implemented.
- Bellhop is the only model that natively writes a multi-source-depth
  grid; everyone else loops in Python.

Examples: 01, 04, 11, 16, 24.

### 5.4 Kraken / KrakenC — normal modes

Real (Kraken) and complex (KrakenC) normal-modes solvers. Both share
`_KrakenBase`; KrakenC handles elastic / leaky cases via complex
arithmetic. **Kraken auto-routes to `krakenc.exe`** when it sees
`shear_speed > 0` anywhere or `leaky_modes=True`, with a `UserWarning`.

```python
from uacpy.models import Kraken, KrakenC
modes = Kraken(c_low=None, c_high=None,   # None ⇒ 0.95 × min SSP / 1.05 × max SSP+bottom
               n_mesh=0,                   # 0 ⇒ auto
               leaky_modes=False) \
        .run(env, source, receiver)
modes.k          # complex wavenumbers, shape (M,)
modes.phi        # (n_z, M) eigenfunctions
```

**Caveats:**
- Modes solve an eigenproblem of the **whole** water column; no
  source-side. Per-model defaults: `'ssp': 'mean'`, `'bottom': 'median'`
  (RD inputs collapse to representative samples).
- KrakenField — not Kraken — handles range-dependent envs (segments).
- Real Kraken raises on elastic media; you'll be auto-routed to
  KrakenC.

Example: 06.

### 5.5 KrakenField — modes + field pipeline

Runs `kraken.exe` (or `krakenc.exe` for elastic) then `field.exe`.
Produces TL grids via adiabatic or coupled-mode theory, plus broadband
`H(f)` and `p(t)` through `RunMode.BROADBAND` / `RunMode.TIME_SERIES`.

```python
from uacpy.models import KrakenField
kf = KrakenField(mode_coupling='adiabatic',    # 'adiabatic' | 'coupled'
                 coherent=True,                  # coupled+coherent=False is rejected
                 n_segments=10)                  # range segments for RD
field = kf.run(env, source, receiver)            # default RunMode.COHERENT_TL
```

**Caveats:**
- Range-dependent bathymetry and SSP are honoured natively (segments).
- Range-dependent / layered bottoms still collapse — defaults
  `'bottom': 'median'`, `'rd_layered_layers': 'preserve'`.
- Rejects `'Q'` SSP interp (Bellhop-only).
- No altimetry support.

Examples: 18, 19.

### 5.6 Scooter — wavenumber integration (FFP)

Range-independent finite-element FFP (frequency domain). The `.grn`
output is converted to range-domain TL via the in-tree Python Hankel
transform (`uacpy.io.grn_reader`).

```python
from uacpy.models import Scooter
sc = Scooter(rmax_multiplier=2.0,           # padding for k-resolution
             source_type='R',                # 'R' cylindrical | 'X' Cartesian
             spectrum='positive',            # 'positive' (fast) | 'negative' | 'both'
             field_interp='O')               # 'O' polynomial | 'P' Padé
field = sc.run(env, source, receiver)
```

Supported run modes: `COHERENT_TL`, `BROADBAND`, `TIME_SERIES`. Supports
elastic + layered bottoms natively. Per-model defaults: `'ssp': 'mean'`,
`'bottom': 'median'`, `'rd_layered_layers': 'preserve'`.

Example: 19.

### 5.7 RAM — parabolic equation (multi-backend dispatcher)

Façade that auto-picks one of three vendored Collins-family PE binaries:

| Env shape | Backend | Source |
|---|---|---|
| fluid bottom + flat surface | `mpiramS` | Dushaw broadband Fortran 95 PE |
| any `shear_speed > 0` | `rams0.5` | Collins elastic PE |
| `env.altimetry is not None` | `ramsurf1.5` | Collins variable-surface PE |
| elastic + altimetry | `NotImplementedError` | (no published Collins PE) |

Inspect via `RAM(...).select_backend(env)`. Every result carries
`result.backend` so you can audit what the dispatcher chose, plus
`dr`/`dz`/`c0`/`zmax`/`Q`/`T` on `result.metadata`.

```python
from uacpy.models import RAM
ram = RAM(dr=None, dz=None, zmax=None,        # None ⇒ Lytaev (2023) Padé optimizer
          accuracy=1e-3, theta_max=30.0,       # optimizer tuning
          c0=None,                             # None ⇒ Lytaev Eq. (15)
          Q=None, T=None,                      # None ⇒ context-dep. (1e6/1.0 narrowband, 2.0/10.0 broadband)
          np_pade=6,
          rams_theta=45.0)                     # rams0.5 Padé rotation angle (°)
field = ram.run(env, source, receiver)
```

Every constructor knob also accepts a per-call override on `run()`
(e.g. `ram.run(env, src, rcv, accuracy=5e-4)`). `rams_theta=` may be a
callable `theta_fn(freq_hz) -> float` to vary the elastic stability
angle across a band.

**RAM-specific notes:**
- `λ_p/16` acoustic dz floor is applied to all three backends (stability
  for Collins, runtime cap for mpiramS); the wrapper warns when it
  triggers because the Lytaev accuracy budget is no longer met.
- `env.altimetry` is positive-up (uacpy convention); the dispatcher
  converts to ramsurf1.5's positive-down convention internally and
  clamps wave crests to `z=0` with a warning (ramsurf only models
  surface depressions).
- mpiramS supports RD-SSP / RD-bathy / RDLB natively. Collins backends
  (rams0.5, ramsurf1.5) use the range-0 SSP and bottom only and warn on
  dropped data.
- `phase_reference='travelling_wave'` on every broadband result —
  consistent across the three backends despite different raw quantities
  (mpiramS stores ψ·exp(+i k0 r), Collins backends use g0
  multiplication or matrix substitution; the wrapper bakes in the
  carrier and applies `conj(·)·exp(−i k0 r)/√r` so IFFT'd peaks align
  at `t = r/c`).
- TL formula (mpiramS): `TL = -20·log10(|psif|·4π) + 10·log10(r)`.

Examples: 05, 18, 19, 20, 22.

### 5.8 SPARC — time-domain PE

Range-independent time-marched FFP. Native time-domain — `RunMode.TIME_SERIES`
returns a `TimeSeriesField` directly, shape `(n_d, n_r, n_t)`.

```python
from uacpy.models import SPARC
sp = SPARC(output_mode='R',          # 'R' horizontal array | 'D' vertical | 'S' snapshot
           pulse_type='PN+B',         # 4-char source pulse code
           n_t_out=501,
           t_max=None,                # None ⇒ 2.5 × travel time
           rmax_safety_margin=1.0001)
field = sp.run(env, source, receiver)
```

**`pulse_type` is a 4-character code** (per SPARC's `sourceMod.f90`):
position 1 = pulse shape (`PRASHNGFBMTC`), 2 = post-process (`H`
pre-envelope, `Q` Hilbert, others none), 3 = sign (`-` invert), 4 =
filter (`L`/`H`/`B` band-pass, `N` none). Common combos: `'PN+B'`
(pseudo-Gaussian), `'RN+ '` (Ricker), `'GH- '` (inverted Gaussian
envelope).

**Caveats:**
- Source pulse is constructor-driven (`pulse_type`); `source_waveform=`
  / `sample_rate=` on `run()` are silently ignored for API uniformity.
- `output_mode='S'` (snapshot) FFTs the snapshot's tout axis — the
  source frequency must stay below the snapshot Nyquist (`0.5/dt`); the
  wrapper raises with a remediation hint if not.
- Only `Vacuum` / `Rigid` bottom interfaces (writer auto-converts).
- Per-model defaults: `'ssp': 'mean'`, `'bottom': 'median'`,
  `'rd_layered_layers': 'preserve'`.

Example: 19.

### 5.9 Bounce — reflection-coefficient writer

Writes `.brc` (bottom) and `.irc` (internal) reflection-coefficient
files for use by Bellhop / Scooter / KrakenC (`.brc`) or Kraken
(`.irc`). Only emits `RunMode.REFLECTION`.

```python
from uacpy.models import Bounce
# In-memory only — read theta / R / phi off the typed result.
res = Bounce(c_low=1400.0, c_high=10000.0, rmax=10000.0).run(env, source, receiver)
res.theta, res.R, res.phi                     # always there

# Chain to a consumer model — pin work_dir so the .brc/.irc persist.
res = Bounce(c_low=1400.0, c_high=10000.0, rmax=10000.0,
             work_dir=Path('./bounce_out')).run(env, source, receiver)
res.metadata['brc_file']                      # ./bounce_out/...brc — valid path
res.metadata['irc_file']                      # Kraken consumer
```

Bounce uses the same uniform ``(work_dir, cleanup)`` rule as every
other model: a user-pinned ``work_dir`` defaults ``cleanup=False`` and
the ``.brc`` / ``.irc`` files persist; an unpinned ``work_dir``
defaults ``cleanup=True`` and the temp dir is wiped when ``run()``
returns. In the latter case ``result.metadata`` omits the (now stale)
file-path keys, so the absence is the signal — try chaining to a
consumer model and you'll see ``KeyError`` immediately.

Bounce overrides defaults to `'bottom': 'median'` and
`'rd_layered_layers': 'preserve'` because it produces ONE BRC consumed
across the whole receiver-range axis. Bellhop's auto-route forwards the
parent's `Bellhop(collapse={…})` overrides to the spawned Bounce.

Examples: 15, 16.

### 5.10 OASES — OAST / OASN / OASR / OASP

Four sub-models mirror the OASES Fortran utilities. Volume absorption is
read from `env.absorption` like every other AT-family model; the chosen
formula's single-letter marker is appended to the OASES options string
(OASES uses empirical Skretting–Leroy internally, so the marker is
informational rather than a physics switch).

```python
from uacpy.models import OAST, OASN, OASR, OASP, OASES
from uacpy.models.base import RunMode

# Wavenumber-integration TL — RunMode.COHERENT_TL
oast = OAST(compute_contour=False,         # add 'C' (range-depth contour)
            compute_depth_average=False,    # add 'A'
            complex_contour=True)           # 'J' option
field = oast.run(env, source, receiver)

# Spatial covariance + matched-field replicas — RunMode.COVARIANCE / RunMode.REPLICA
oasn = OASN()
cov = oasn.compute_covariance(env, source, receiver)
rep = oasn.compute_replicas(env, source, receiver,
                            replica_zmin=10, replica_zmax=90, replica_nz=20,
                            replica_xmin=500, replica_xmax=10000, replica_nx=40)

# Reflection coefficients — RunMode.REFLECTION
oasr = OASR(angles=None,                   # default linspace(0, 90, 181)
            angle_type='grazing',           # 'grazing' | 'incidence' (90 - x)
            reflection_type='P-P')          # 'P-P' | 'P-SV' | 'P-Slow' | 'transmission'
refl = oasr.run(env, source, receiver)
broad = oasr.run(env, source, receiver,
                 freq_min=50, freq_max=200, n_frequencies=16)

# Broadband / pulse synthesis — RunMode.BROADBAND / RunMode.TIME_SERIES
oasp = OASP(n_time_samples=4096, freq_max=250.0)
tf = oasp.run(env, source, receiver, run_mode=RunMode.BROADBAND)

# One-line factory — pick the right sub-class by run_mode
m = OASES(run_mode=RunMode.COVARIANCE)     # → OASN
m = OASES(broadband=True)                   # → OASP (instead of default OAST)
```

OASP is wideband wavenumber-integration / pulse synthesis (NOT a
parabolic equation). Per-model defaults for all four:
`'bottom': 'median'`, `'rd_layered_layers': 'preserve'`. OAST/OASN/OASP
also default `'ssp': 'mean'`; OASR keeps `'ssp': 'r0'` (the SSP
boundary speed is essentially irrelevant to the reflection coefficient).

Examples: 13, 19.

---

## 6. Results — typed hierarchy

Every `model.run(...)` returns an instance of a typed `Result`
subclass:

```
Result                              identification + metadata
├── PressureField                   (n_d, n_r) or (n_d, n_r, n_f); units='complex' (default) or 'dB'
│                                   .tl always works; .p only when complex
├── TransferFunction                (n_d, n_r, n_f) complex; sibling of PressureField
│                                   carries phase_reference + .synthesize_time_series / .to_time_trace / .to_tl
├── TimeSeriesField                 (n_d, n_r, n_t) real, p(t) on a grid
├── TimeTrace                       (n_t,) real, p(t) at one (depth, range)
├── Arrivals                        flat list of arrival events
├── Rays                            list of ray polylines
├── Modes                           Kraken normal modes (.k complex wavenumbers, .phi eigenfunctions)
├── Covariance                      OASN spatial covariance C(f, i, j)
├── Replicas                        OASN MFP replica fields (n_f, n_z, n_x, n_y, n_rcv)
└── ReflectionCoefficient           .theta + .R / .phi (n_θ,) or (n_θ, n_f)
```

### Trailing-axis convention

The variable axis is always trailing — `np.fft.ifft(H)` works without an
axis argument:

| Result | Shape |
|---|---|
| `PressureField` (narrowband) | `(n_d, n_r)` |
| `PressureField` (broadband) / `TransferFunction` | `(n_d, n_r, n_f)` |
| `TimeSeriesField` | `(n_d, n_r, n_t)` |
| `TimeTrace` | `(n_t,)` |
| `Covariance` | `(n_f, n_rcv, n_rcv)` |
| `Replicas` | `(n_f, n_z, n_x, n_y, n_rcv)` |

### Slicing — `.at()` / `.max()` / `.data`

| Method | Selector | Behaviour |
|---|---|---|
| `.at(depth=, range=, frequency=, time=)` | label (m / Hz / s) | nearest grid point via `argmin(|axis - value|)`; returns a typed slice |
| `.max()` | none | global `argmax(|data|)` across all axes |
| `.data[…]` | numpy slicing | raw ndarray (no axis labels, no metadata) |

**Auto-squeeze on slices.** A full grid preserves shape
(`field.tl.shape == field.data.shape`); a *sliced* field is
`SlicedPressureField` and squeezes singleton axes on `.tl` / `.p`:

```python
field.at(depth=50).tl                 # (n_r,) — plt-ready
field.at(range=2000, depth=50).tl     # 0-D scalar
field.at(frequency=200).tl            # (n_d, n_r) for a broadband field
field.max().tl                        # 0-D scalar at global argmax(|data|)
```

**`.tl` vs `.data`.** Use `.tl` for plotting (always dB, always
plot-friendly shape). Use `.data` inside numerical code that pairs the
field with a 2-D mask — auto-squeeze on a sliced `.tl` can drop axis
identity. `_complex_to_db` from `core.results` converts when needed.

`SoundSpeedProfile` and `RangeDependentBottom` split nearest vs.
interpolation: `.at(...)` is nearest, `.eval(...)` interpolates.

### Common metadata

Every `Result` carries:

| Attribute | Type | Meaning |
|---|---|---|
| `model` | `str` | wrapper class name (`'Bellhop'`, `'KrakenField'`, …) |
| `backend` | `str` | concrete binary that ran (`'mpiramS'`, `'kraken.exe'`, …); `== model` (lowercased) when no dispatcher |
| `source_depths` | `ndarray` | every source depth in the run |
| `frequencies` | `ndarray` | always 1-D, plural; `result.f0` is the convenience scalar |
| `phase_reference` | `'travelling_wave'` / `'time_domain_native'` | complex `H(f)` phase convention (see below) |

Free-form `result.metadata` dict carries genuinely model-specific
outputs (OAST: `oast_native_ranges`; RAM: `dr` / `dz` / `c0` / `zmax`
/ `Q` / `T`). Typed attrs are NOT mirrored into metadata — read them
as `result.model`, `result.backend`, etc.

### Output-file paths in metadata

Every model attaches its primary on-disk output paths (and the binary's
`.prt` log) to `result.metadata` **when `cleanup=False`** — i.e. when
the files outlive `run()`. The convention is **uniform**: paths in
`metadata` are valid when present; absence means the work dir was
cleaned up, no signal needed.

| Model | Keys (when present) |
|---|---|
| Bellhop / BellhopCUDA | `shd_file` / `arr_file` / `ray_file` (run-mode dependent), `prt_file` |
| Kraken / KrakenC | `mod_file`, `prt_file` |
| KrakenField | `shd_file`, `mod_file`, `prt_file` |
| Scooter | `grn_file`, `prt_file` |
| SPARC | `grn_file` / `rts_file` (output-mode dependent), `prt_file` |
| OAST | `plt_file`, `prt_file` |
| OASN | `xsm_file` (Covariance) / `rpo_file` (Replicas), `prt_file` |
| OASR | `trc_file`, `rco_file`, `prt_file` |
| OASP | `trf_file`, `prt_file` |
| RAM (mpiramS) | `psif_file`, `prt_file` |
| RAM (Collins) | `tl_grid_file`, `pcomplex_file`, `in_file` (per-call work-dir; broadband attaches the last frequency's paths) |
| Bounce | `brc_file`, `irc_file`, `prt_file` |

Read with `.get()` so absence is graceful:
```python
result = bh.run(env, src, rcv)               # default cleanup=True ⇒ no paths
result.metadata.get('prt_file')              # → None

result = Bellhop(work_dir='./out').run(...)   # work_dir pinned ⇒ paths present
result.metadata['shd_file']                   # → './out/bellhop_run.shd'
```

### Phase reference

`TransferFunction.phase_reference` is a `PhaseReference` enum (subclass
of `str`). Two values today:

- **`'travelling_wave'`** — Bellhop / Scooter / OASP / KrakenField /
  RAM (all backends) / SPARC steady-state modes (`output_mode='R'/'D'/'S'`,
  recovered via time-FFT + Hankel transform of the native pressure).
  `H(f)` carries the engineering propagator `exp(-i 2πf r/c)`; a direct
  IFFT aligns arrivals at `t = r/c`. The wrappers normalise across
  solvers (negate KrakenField field.exe to match Scooter polarity; bake
  the carrier into RAM mpiramS / rams0.5 / ramsurf1.5 outputs which
  each store a different raw quantity).
- **`'time_domain_native'`** — SPARC `RunMode.TIME_SERIES`. Result data
  is real `p(t)` direct from the solver; no phase reconstruction needed.

`TransferFunction.synthesize_time_series(...)` and `.to_time_trace(...)`
honour both transparently. Cross-model agreement is verified in
`tests/test_cross_model_broadband.py` (~6 dB of Scooter on a Pekeris
reference; IFFT envelope peaks within 100 ms).

### Filtering helpers

`Rays` and `Arrivals` are pure data containers — filtering helpers
return new instances (closed under filter):

```python
result = bellhop.run(env, source, receiver, run_mode=RunMode.RAYS)
result.filter_by_bounces(kind='direct')              # 0 surface, 0 bottom
result.filter_by_bounces(bot=(1, None))              # ≥1 bottom bounce
result.filter_by_launch_angle(-5, 5)                 # ±5° fan
result.filter(lambda r: r['alpha'] > 0)              # generic predicate

arr = bellhop.run(env, source, receiver, run_mode=RunMode.ARRIVALS)
arr.in_delay_window(0.05, 0.15)                       # delay range
arr.top_n_by_amplitude(8)
```

For "rays at a receiver", use `compute_eigenrays`. It runs Bellhop's
eigenray solver and returns the raw `Rays`; filtering / sorting /
truncation lives on `Rays` itself so the convenience method stays
small:

```python
rays = bellhop.compute_eigenrays(env, source, range=2000, depth=30)
close = rays.top_n_by_miss(8).truncate_at_receiver()
within = rays.filter_by_miss_distance(max_miss=15.0)
direct = rays.filter_by_bounces(kind='direct')
```

For multi-receiver eigenray runs, pass a `Receiver` instead of
`range=`/`depth=`; the result has `is_eigen=True` and carries
`receiver_depths` / `receiver_ranges`.

---

## 7. Visualization

`uacpy.visualization` ships ~30 plot helpers. Every result type has a
`.plot(env=env)` method that auto-dispatches to the appropriate helper
— that's the canonical entry point:

```python
field = bellhop.run(env, source, receiver)
fig, ax, im = field.plot(env=env)               # → plot_transmission_loss
modes.plot(n_modes=6)                            # → plot_modes
arrivals.plot()                                  # → plot_arrivals
rays.plot(env=env)                               # → plot_rays
tf.plot()                                        # → plot_transfer_function or _slice
```

The uacpy rcParams (grid, fonts, colours) are applied automatically when
`uacpy.visualization` is imported. To re-apply after fiddling with
matplotlib defaults:

```python
from uacpy.visualization.style import apply_professional_style
apply_professional_style()
```

### Helpers by category

| Category | Helpers |
|---|---|
| **TL** | `plot_transmission_loss`, `plot_transmission_loss_polar`, `plot_range_cut`, `plot_depth_cut`, `plot_tl_difference` |
| **Rays** | `plot_rays` (auto-truncates eigenrays at closest approach) |
| **Modes** | `plot_modes`, `plot_mode_functions`, `plot_modes_heatmap`, `plot_mode_wavenumbers`, `plot_dispersion_curves` |
| **Environment** | `plot_environment`, `plot_environment_advanced`, `plot_ssp`, `plot_ssp_2d`, `plot_bathymetry`, `plot_bottom_properties`, `plot_rd_bottom`, `plot_layered_bottom`, `plot_rd_layered_bottom`, `plot_bottom_loss` |
| **Time / spectrum** | `plot_time_series`, `plot_time_trace`, `plot_arrivals`, `plot_transfer_function`, `plot_transfer_function_slice`, `plot_phase_field` |
| **Other results** | `plot_reflection_coefficient`, `plot_reflection_coefficient_heatmap`, `plot_covariance`, `plot_replicas` |
| **Multi-model** | `compare_models`, `compare_range_cuts`, `plot_model_comparison_matrix` |

Two canonical patterns:

```python
# TL with shared colourbar across models
from uacpy.visualization import plot_transmission_loss, compare_models
fig, ax, im = plot_transmission_loss(field, env=env,
                                     vmin=None, vmax=None,    # auto: median + 0.75·std rounded
                                     contours=[70, 80, 90])
compare_models({'Bellhop': f_bh, 'RAM': f_ram, 'KrakenField': f_kf},
               env=env, vmin=30, vmax=90)

# Signed TL difference (a − b) on diverging cmap, with matched bathy floor
from uacpy.visualization import plot_tl_difference
plot_tl_difference(f_ram, f_bh, env=env, label='RAM − Bellhop', diff_vmax=10)
```

For per-helper kwargs, read the docstrings (`help(plot_rays)`).
Examples 14 and 17 tour the visualization surface end-to-end.

---

## 8. Signal Processing

Reachable as `uacpy.signal` (alias for the on-disk `acoustic_signal`
package; the alias is an attribute of `uacpy`, so `import uacpy.signal`
does not work — use `import uacpy; sig = uacpy.signal`).

Three sub-modules:

| Module | Purpose | Reach |
|---|---|---|
| `uacpy.signal.generation` | Waveforms (CW, chirps, Ricker, BPSK, bandlimited noise, SSRP) | `tone_burst`, `lfm_chirp`, `hfm_chirp`, `cw`, `sweep`, `gaussian_pulse`, `ricker_wavelet`, `bpsk_modulate`, `make_bandlimited_noise`, `ssrp` |
| `uacpy.signal.processing` | Beamforming, source/noise scaling, Fourier synthesis | `planewave_rep`, `beamform`, `add_noise`, `fourier_synthesis` |
| `uacpy.signal.analysis` | Class-based estimators | `PSD`, `PPSD`, `Spectrogram`, `SEL`, `FRF`, `FKTransform` |

Three canonical patterns:

```python
import uacpy
sig = uacpy.signal

# 1. Build a waveform — chirps return (signal, time); cw/sweep return signal only
s, t = sig.lfm_chirp(fmin=100.0, fmax=1000.0, T=1.0, sample_rate=10000)

# 2. Add Gaussian noise sized for SL/NL around carrier (fc, BW)
y = sig.add_noise(s, sample_rate=10000, source_level_db=150.0,
                  noise_level_db=80.0, fc=550.0, bandwidth=900.0)

# 3. Class-based estimator — instantiate with reference + window settings,
#    .compute(data, fs), then .plot(...) to render
psd = sig.analysis.PSD(ref=1e-6, nperseg=8192, noverlap=4096, window='hann')
freqs, Pxx = psd.compute(data, fs)
fig, ax = psd.plot(title='Site A', ymin=40, ymax=120)
```

Conventions: reference pressure defaults to **1 µPa** (water; pass
`ref=2e-5` for air). dB scales in `.plot()` are dB re `ref²/Hz` for
PSD-like quantities and dB re `ref²·s` for SEL. PSDs are stored linear
(`Pa²/Hz`); conversion to dB happens in `.plot()`.

For per-class kwargs / methods, read the docstrings
(`help(uacpy.signal.analysis.FRF)`). Examples 09, 10 walk through the
common workflows.

Scientific content in `uacpy.core.acoustics` is adapted from arlpy
(BSD-3-Clause) — see `uacpy/third_party/arlpy/LICENSE` and `NOTICE`.

---

## 9. Ambient Noise

`uacpy.noise` packages a Tollefsen / Pecknold-style Wenz model
(wind / shipping / rain / thermal / turbulence). The user-facing API
is the `WenzNoise` class plus `compute_windnoise`:

```python
import numpy as np
from uacpy.noise import WenzNoise

f = np.logspace(0, 5, 1000)            # 1 Hz – 100 kHz
wenz = WenzNoise(f, wind_speed=15,      # knots
                 rain_rate='moderate',
                 water_depth='deep',
                 shipping_level='medium')

wenz.total          # incoherent sum (dB re 1 µPa²/Hz)
wenz.shipping       # per-component levels
wenz.wind; wenz.rain; wenz.thermal; wenz.turbulence
wenz.components     # (N, 6) ndarray: [total, ship, wind, rain, therm, turb]
fig, ax = wenz.plot()                    # stacked components
```

`shipping_level ∈ {no, low, medium, high}`,
`rain_rate ∈ {no, light, moderate, heavy, veryheavy}`,
`water_depth ∈ {deep, shallow}`. Wind speed in **knots**.

Round-trip a noise spectrum to a time-domain realisation with
`uacpy.signal.ssrp` and verify with `PPSD`:

```python
Pxx = wenz.as_psd()                                  # Pa²/Hz
t, x, fs = uacpy.signal.ssrp(Pxx, wenz.frequencies, duration=30.0)
ppsd = uacpy.signal.PPSD(ref=1e-6, seg_duration=1.0)
ppsd.compute(x, fs); ppsd.plot()
```

References: Tollefsen & Pecknold, *Wenz curves for predicting ambient
noise*, DRDC-RDDC-2022-D051; Wenz (1962); Mellen (1952); Piggott (1964);
Merklinger (1979); Torres & Costa (2019); Nichols & Bradley (2016).

Example: 09.

---

## 10. Units & Conventions

| Quantity | Unit | Notes |
|---|---|---|
| Depth, range, altitude | metres | depth positive down from surface |
| Sound / shear speed | m/s | water 1400–1550; sediment 1500–2000; shear 0 (fluid) or 100–500+ |
| Frequency / sample rate | Hz | |
| Time | s | |
| Launch / grazing angle | degrees | negative = upward, 0 = horizontal |
| Attenuation | dB / wavelength (dB/λ) | uacpy stores **every** attenuation field in this unit. AT TopOpt(3) is hard-wired to ``'W'``. |
| Transmission loss | dB | TL = −20·log₁₀(\|p\|/\|p₀\|) |
| Density | g/cm³ | water ≈ 1.0; sediment 1.2–2.5 |
| Source level | dB re 1 µPa @ 1 m | |
| Noise spectral density | dB re 1 µPa²/Hz | |

uacpy stores ranges internally in **metres** and converts at the I/O
boundary when writing model input files (Bellhop / Kraken `.env` /
`.bty` use km; OASES + RAM Collins use metres). You always pass metres
to `Receiver(ranges=…)` and `bathymetry`.

---

## 11. Troubleshooting

**Executable not found.** You forgot to run `./install.sh`, or the build
failed for one of the binaries. `uacpy/bin/` should contain `oalib/`,
`mpirams/`, `bellhopcuda/`, and (after fetch) `oases/`. Rerun
`./install.sh` and read its stderr.

**"Source/Receiver depth exceeds environment depth".** The model's
`validate_inputs()` checks against `env.depth`. Trim your receiver grid
or deepen the env.

**Kraken says "does not support range-dependent environments".** Use
`KrakenField` (adiabatic / coupled modes) or switch to Bellhop / RAM.

**A model dropped my range-dependent SSP / bottom / layered bottom.**
Each unsupported feature triggers one `UserWarning` per `run()`. Either
pick a model that supports it (§5.2) or change the collapse policy —
`Kraken(collapse={'ssp': 'mean', 'bottom': 'median'})`.

**`LayeredBottom` / RDLB / elastic / shear-RD bottom in Bellhop.**
Auto-routed through BOUNCE — generates a `.brc` table and re-runs
Bellhop against `acoustic_type='file'`, with one `UserWarning`. Use
`Bellhop.run_with_bounce(...)` for explicit BOUNCE parameters, or pick
a natively-layered model (Kraken / KrakenC / Scooter / SPARC / OASES).

**OASN normal modes.** OASN does NOT produce eigenfunctions; it produces
covariance matrices (`.xsm` → `Covariance`) and matched-field replicas
(`.rpo` → `Replicas`). For eigenfunctions, use Kraken / KrakenC.

**Bellhop ray box too small.** Increase `r_box` / `z_box`, or let them
default (1.2 × receiver extent).

**Kraken not converging.** Raise `n_modes` or widen `c_low` / `c_high` —
the defaults (0.95 × min SSP, 1.05 × max SSP+bottom) sometimes miss
high-speed bottom-trapped modes.

**TL contains NaNs at long range (RAM).** Check `zmax` and
`absorbing_layer_width`; spurious reflections from the PE-domain bottom
typically show up as NaNs. Bumping `absorbing_layer_width` to ~30–40
wavelengths usually fixes it.

**Bellhop time series.** Bellhop's `RunMode.TIME_SERIES` runs in
arrivals mode under the hood and convolves the per-receiver impulse
response with `source_waveform`. For native time-domain propagation,
use SPARC (time-marched PE) or OASP (broadband wavenumber-integration /
pulse synthesis).

**OASES factory rejects a kwarg.** `OASES(...)` forwards kwargs to the
chosen sub-class; mistypes surface as `TypeError` instead of being
silently dropped. Check the targeted class's signature
(`help(OASN)` / `help(OASR)` / etc.).

---

## 12. Examples Index

All scripts live in `uacpy/uacpy/examples/` and run as-is once
`./install.sh` has completed. They are all `@pytest.mark.slow`-marked
in `tests/test_examples_integration.py`; the *Long* column flags those
needing a longer subprocess timeout (240 s instead of 120 s).

| # | File | Demonstrates | Long? |
|---|------|--------------|:-----:|
| 01 | `example_01_basic_shallow_water.py` | Minimal TL — start here | |
| 02 | `example_02_sound_speed_profiles.py` | SSP types (linear, Munk, cubic …) | ✓ |
| 03 | `example_03_multi_frequency.py` | Sweep over frequencies | |
| 04 | `example_04_bellhop_advanced.py` | Beam types, source patterns, advanced rays | |
| 05 | `example_05_ram_advanced.py` | RAM (mpiramS) with sloping shelf + RD bottom | |
| 06 | `example_06_kraken_advanced.py` | Modal analysis with Kraken | |
| 07 | `example_07_all_models_comparison.py` | All models side by side, `compare_models` + `plot_rd_bottom` | |
| 08 | `example_08_long_range.py` | Convergence-zone propagation | |
| 09 | `example_09_ambient_noise.py` | Wenz noise + ssrp synthesis + PPSD verification | |
| 10 | `example_10_signal_processing.py` | CW, chirps, matched filtering | |
| 11 | `example_11_bellhop_run_modes.py` | Every Bellhop run mode + `compute_eigenrays` | |
| 12 | `example_12_attenuation_models.py` | Thorp / Francois-Garrison / biological | |
| 13 | `example_13_oases_suite.py` | OAST / OASN / OASR / OASP | |
| 14 | `example_14_new_plotting_features.py` | Visualization tour | |
| 15 | `example_15_elastic_boundaries_comparison.py` | Fluid vs elastic bottoms | |
| 16 | `example_16_bellhop_bounce_integration.py` | `Bellhop.run_with_bounce()` + `LayeredBottom` | |
| 17 | `example_17_boundary_conditions_layered.py` | Surface BC + layered bottoms (RD layered, preset stacks) | ✓ |
| 18 | `example_18_rd_bottom_krakenfield_vs_ram.py` | RD layered bottom: adiabatic vs coupled vs RAM | |
| 19 | `example_19_broadband_comparison.py` | Multi-model `H(f)` + `p(t)` (Bellhop / RAM / Scooter / OASP / SPARC) | ✓ |
| 20 | `example_20_ram_backends.py` | RAM dispatcher: mpiramS / rams / ramsurf | |
| 21 | `example_21_bellhop_vs_ramsurf.py` | Bellhop vs ramsurf with rough surface | |
| 22 | `example_22_ram_lytaev_grid.py` | RAM Lytaev (2023) Padé grid optimizer | ✓ |
| 23 | `example_23_collapse_methods.py` | Same RD env collapsed multiple ways via `collapse={…}` | |
| 24 | `example_24_synthesize_time_series.py` | Bellhop `H(f)` → IFFT → `p(t)` via `TransferFunction.synthesize_time_series` | |
| 25 | `example_25_canonical_presets.py` | Parametric SSPs + plane-wave bottom-loss overlay | |

Smoke test:

```bash
python uacpy/uacpy/examples/example_01_basic_shallow_water.py
```

---

## Reference

- Source: https://github.com/ErVuL/uacpy
- Issues: https://github.com/ErVuL/uacpy/issues
- Contact: ervul.github@gmail.com
- Citation / licensing / acknowledgements: see `README.md`
