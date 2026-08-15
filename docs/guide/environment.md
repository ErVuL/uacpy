# Environment — describing the ocean to uacpy

> `uacpy.Environment` and the carriers it holds · the input every model consumes

You pick a model for the physics. You write an `Environment` for the water.
This page is the second thing to read: it covers every carrier that describes
the ocean, and the **collapse policy** that decides what happens when the model
you picked cannot consume something you wrote.

One object goes into every model in the package:

```python
result = Model(**knobs).run(env, source, receiver, run_mode=...)
```

`env` is always an `Environment`. `source` and `receiver` are covered in
[source and receiver](source-receiver.md).

---

## 1. The `Environment` carrier

`Environment` is a container of six independent carriers, each owning one
aspect of the ocean:

| Field | Type | What it owns | Default |
|---|---|---|---|
| `bathymetry` | `Bathymetry` | seafloor depth vs range | **required** |
| `ssp` | `SoundSpeedProfile` | water sound speed `c(z)` or `c(z, r)` | isovelocity 1500 m/s |
| `bottom` | `Bottom` | seabed acoustic properties | 1600 m/s / 1.5 g/cm³ / 0.5 dB/λ half-space |
| `surface` | `Surface` | top-boundary acoustic properties | vacuum (pressure release) |
| `altimetry` | `Altimetry` or `None` | sea-surface **shape** vs range | `None` (flat, z = 0) |
| `absorption` | `Absorption` or `None` | water-column volume attenuation | `None` |

Only `bathymetry` is required. Every other argument has a physically sensible
default, and each accepts a shorthand that is coerced to the real carrier — so
you rarely construct the carriers by hand:

```python
import uacpy

env = uacpy.Environment(
    name='Shallow-water channel',
    bathymetry=100.0,                                  # → Bathymetry
    ssp=[(0.0, 1500.0), (30.0, 1495.0), (100.0, 1490.0)],  # → SoundSpeedProfile
    bottom='sand',                                     # → Bottom (preset half-space)
)
```

`Environment` also carries optional provenance — `name`, `location`,
`transect`, `date` and `data_sources`. These are keyword-only, default to
`None`/empty for a hand-built environment, and are stamped for you by
`uacpy.data.fetch_environment`; see [external data](data.md).

### The six carriers, drawn at once

Everything on this page comes from
[`docs/figure_scripts/environment.py`](../figure_scripts/environment.py) — the
code below **is** that code, so it cannot drift from what you see.

```python
import numpy as np
import uacpy

env = uacpy.Environment(
    name='Shelf break',
    bathymetry=[(0.0, 90.0), (6000.0, 110.0),
                (12000.0, 210.0), (20000.0, 380.0)],
    ssp=[(0.0, 1522.0), (20.0, 1521.0), (60.0, 1502.0),
         (150.0, 1492.0), (380.0, 1488.0)],
    bottom=uacpy.Bottom.from_columns(
        [uacpy.SeabedColumn.from_presets(
            layers=[('sand', 12.0)], halfspace='limestone'),
         uacpy.SeabedColumn.from_presets(
            layers=[('silt', 30.0)], halfspace='chalk')],
        ranges=[0.0, 20_000.0],
    ),
    altimetry=uacpy.generate_sea_surface(
        20_000.0, wind_speed_ms=15.0, n_points=1200, seed=0),
)
source = uacpy.Source(depths=40.0, frequencies=150.0)
receiver = uacpy.Receiver(depths=np.linspace(40.0, 320.0, 8), ranges=18_000.0)

env.plot(source=source, receiver=receiver)
```

![Environment overview](figures/env_overview.png)

A shelf-break transect: the seafloor falls from 90 m to 380 m over 20 km, a
summer thermocline sits at 60 m, and the seabed turns from sand-over-limestone
to silt-over-chalk past the break. The two colour scales are independent —
**blues** are water sound speed, **browns** are seabed compressional speed —
because they span different ranges and sharing one would flatten both. `P1` and
`P2` mark the two seabed columns; the dashed line is the boundary between them,
drawn midway between the range nodes. The red star is the source, the green
dots the receiver array.

`env.plot()` is the carrier counterpart of `result.plot()`. Every uacpy object
you would plot on its own has a `.plot()` — see [plotting](plotting.md).

### `depth` is derived, not stored

```python
env.depth        # 380.0 — read-only, maximum of env.bathymetry.depths
env.max_range    # 20000.0 — largest range coordinate across every carrier
```

**`bathymetry` is the sole seafloor source.** No other carrier declares where
the seabed is: not `bottom` (which carries acoustic properties only), not
`ssp` (whose depth axis may extend past the seafloor). Assigning `env.depth`
raises `AttributeError`.

Consequences worth knowing:

- If the SSP stops **above** the deepest bathymetry, `Environment.__init__`
  extends it to that depth by holding the deepest sound speed flat — the
  constant-extrapolation convention the Acoustics-Toolbox writers require.
- If the SSP extends **below** the seafloor, it is left alone. `env.depth`
  still comes from the bathymetry.

```python
uacpy.Environment(bathymetry=100.0, ssp=[(0, 1500), (50, 1490)]).ssp.depths
# array([  0.,  50., 100.])   ← extended, c held at 1490 m/s

uacpy.Environment(bathymetry=100.0, ssp=[(0, 1500), (200, 1480)]).depth
# 100.0                       ← bathymetry wins
```

### Querying an environment

| Call | Returns |
|---|---|
| `env.depth` | max seafloor depth (m) |
| `env.max_range` | range extent across all carriers (m) |
| `env.get_sound_speed(depth, range=0.0)` | `c` at depth(s), linear in depth |
| `env.get_representative_depth(method)` | `'max'`/`'median'`/`'mean'`/`'min'`/`'initial'` |
| `env.is_range_dependent` | does *anything* vary with range? |
| `env.has_range_dependent_bathymetry` / `..._ssp` / `..._bottom` | per-axis |
| `env.has_layered_bottom` / `has_range_dependent_layered_bottom` | bottom shape |
| `env.has_elastic_bottom` / `has_elastic_surface` | is there shear? |
| `env.copy()` | deep copy — every carrier duplicated |

`get_sound_speed` is always **linear** in depth regardless of the interpolation
scheme a model will use, and warns if you ask outside the profile (the value is
held flat at the nearest endpoint, never fabricated).

The `has_*` predicates are the same ones the collapse machinery reads in
[§7](#7-collapse-policy), so they tell you in advance what a model will complain
about.

---

## 2. `Bathymetry` and `Altimetry` — the two shape carriers

Both are 1-D `value(range)` profiles with the same API. They differ in one
thing: **sign convention**.

| | `Bathymetry` | `Altimetry` |
|---|---|---|
| Value vector | `.depths` | `.heights` |
| Sign | positive **down** | positive **up** |
| Zero | sea surface | mean sea level |
| Constraint | strictly positive | any finite value |
| `None` allowed | no — required | yes (flat surface) |

```python
bathy = uacpy.Bathymetry.coerce([(0.0, 100.0), (5000.0, 200.0)])

bathy.at(range=2500.0)     # 100.0  — nearest stored node, never fabricates
bathy.eval(range=2500.0)   # 150.0  — linear interpolation
bathy.isel(range=-1)       # 200.0  — positional
bathy.depth                # 200.0  — the deepest point
bathy.range_max            # 5000.0
bathy.to_pairs()           # (N, 2) array of (range, depth)
```

`at` / `eval` / `isel` are the shared **grid-library** selectors used across
uacpy ([results](results.md) has the same three on `Field`). `at` is always
nearest; `eval` takes `method='linear'` (default), `'nearest'` or `'cubic'`,
with constant extrapolation past the ends. Because these carriers have a single
axis, the selectors collapse it and hand back the **value** directly — a float
for a scalar range, an array for an array of ranges.

Ranges are metres everywhere in the API. Kilometres appear only on plot axes
and inside the native model file formats.

### Generating a sea surface

`generate_sea_surface` draws a Pierson–Moskowitz realisation and returns an
`(n_points, 2)` array of `(range, height)`, ready to hand to
`Environment(altimetry=...)`:

```python
from figure_scripts._common import sloping_shelf   # 100 m shelf → 400 m over 20 km

env, _, _ = sloping_shelf()
altimetry = uacpy.Altimetry.coerce(uacpy.generate_sea_surface(
    2000.0, wind_speed_ms=12.0, n_points=800, seed=7))

altimetry.plot(title='Altimetry — sea-surface height, positive up')
env.bathymetry.plot(title='Bathymetry — seafloor depth, positive down')
```

![Bathymetry and altimetry](figures/env_range_profiles.png)

The two panels are the same kind of object with opposite sign conventions:
the altimetry axis points up and crosses zero, the bathymetry axis points down
and cannot. `wind_speed_ms` is the wind at 19.5 m (the Pierson–Moskowitz
convention); significant wave height is `Hs ≈ 0.021·U²`, so 12 m/s gives
`Hs ≈ 3 m` — consistent with the ±2 m excursions above. Pass `seed=` for a
reproducible realisation.

**Altimetry is surface *shape*, `Surface` is surface *properties*.** A rough
sea surface that is still pressure-release is `altimetry=`; an ice canopy with
its own sound speed and shear is `surface=`. They are independent, and models
support them independently — [Bellhop](../models/bellhop.md) and
[RAM](../models/ram.md) are the only two that take altimetry natively.

---

## 3. `SoundSpeedProfile`

The water column. One class covers 1-D and 2-D profiles: internally the data is
always a `(n_depth, n_range)` matrix, with `ranges=None` and one column for the
range-independent case.

### Building one

| Factory | Use |
|---|---|
| `SoundSpeedProfile.from_pairs([(z, c), …])` | measured cast — the common case |
| `SoundSpeedProfile.from_isovelocity(depth_max, c=1500.0)` | constant column |
| `SoundSpeedProfile.from_munk(depth_max, n_points=101)` | deep-water canonical profile |
| `SoundSpeedProfile.from_mackenzie(depths, T, S)` | from in-situ `T(z)` and `S(z)` |
| `SoundSpeedProfile.from_2d(depths, ranges, matrix)` | range-dependent `c(z, r)` |

`Environment(ssp=…)` coerces the shorthands for you: `None` → isovelocity at
1500 m/s spanning the water column, a scalar → isovelocity at that speed, a
list of `(depth, c)` pairs → `from_pairs`, and a `SoundSpeedProfile` is used
as-is.

```python
shallow = {
    'Isovelocity': uacpy.SoundSpeedProfile.from_isovelocity(200.0, 1500.0),
    'Summer thermocline': uacpy.SoundSpeedProfile.from_pairs(
        [(0.0, 1523.0), (15.0, 1522.0), (30.0, 1512.0),
         (60.0, 1497.0), (120.0, 1492.0), (200.0, 1493.0)]),
    'Winter (isothermal)': uacpy.SoundSpeedProfile.from_pairs(
        [(0.0, 1487.0), (50.0, 1487.8), (120.0, 1489.0), (200.0, 1490.3)]),
    'Surface duct': uacpy.SoundSpeedProfile.from_pairs(
        [(0.0, 1496.0), (40.0, 1499.0), (55.0, 1494.0),
         (120.0, 1490.0), (200.0, 1489.0)]),
}
munk = uacpy.SoundSpeedProfile.from_munk(5000.0)
```

![Canonical SSP shapes](figures/env_ssp_shapes.png)

Four shallow-water shapes and the deep-water canonical one. The **summer
thermocline** bends every ray downward and strips energy out of the surface
layer. The **winter** profile is very slightly upward-refracting, so energy
stays near the surface and range dependence matters less. The **surface duct**
has a sound-speed maximum at 40 m, trapping energy above it — at the price of
putting every trapped path onto the sea surface once per cycle, so a duct's
reach falls away with sea state and frequency (§2). **Munk** is the
canonical deep-water profile with its axis at 1300 m (dashed) and `c_min` =
1500 m/s — the sound channel that carries energy across ocean basins.

`from_munk` is analytic, so those numbers are fixed by the formula, not by the
`depth_max` you request; `n_points` only sets the sampling.

### Shape declares, the model interpolates

This split matters and is easy to trip over:

- The **carrier** declares what its samples *are*, via
  `SoundSpeedProfile(shape=…)`: `'measured'` (default), `'isovelocity'`,
  `'munk'`, `'analytic'` or `'n2linear'`.
- The **model** decides how to *connect* those samples, via
  `Model(interp_ssp=…)`: `'linear'`, `'pchip'`, `'cubic'`, `'quad'`,
  `'n2linear'`, `'analytic'`, or `None` for auto.

`shape` is informational metadata with exactly one exception:
`shape='isovelocity'` forces constant interpolation, because any connection
scheme over constant data is constant anyway. Every other value leaves the
choice to the model.

`interp_ssp=None` (the default on every model that has it) resolves to
`'quad'` when the environment carries a range-dependent SSP, and `'linear'`
otherwise.

The reason for the split: the same measured cast is a legitimate input to seven
solvers that discretise it differently. Baking an interpolation scheme into the
carrier would mean re-writing the environment to change solver — which is the
thing this whole design exists to avoid.

### Range-dependent (2-D) profiles

```python
depths = np.linspace(0.0, 300.0, 61)
ranges = np.linspace(0.0, 30_000.0, 13)
# A front: the thermocline deepens from 40 m to 110 m across the transect.
z_therm = np.linspace(40.0, 110.0, ranges.size)
matrix = np.empty((depths.size, ranges.size))
for j, zt in enumerate(z_therm):
    matrix[:, j] = 1492.0 + 30.0 / (1.0 + np.exp((depths - zt) / 12.0))

ssp = uacpy.SoundSpeedProfile.from_2d(depths, ranges, matrix)
env = uacpy.Environment(name='Frontal transect', bathymetry=300.0, ssp=ssp,
                        bottom='sand')

ssp.plot(title='One cast per range node')
env.plot(title='The same profile in the environment')
```

![Range-dependent SSP](figures/env_ssp_range_dependent.png)

The same carrier, two views. On the left `ssp.plot()` draws one line per range
column, coloured by range — the fan of casts you would actually have measured.
On the right the environment renders it as a field, which is where the
structure reads: a warm eddy or shelf-slope front pushing the thermocline from
40 m down to 110 m across 30 km.

Slicing follows the grid-library rules, with one guard specific to 2-D
profiles:

```python
ssp.at(depth=50.0, range=10_000.0)   # nearest sample on both axes
ssp.eval(depth=50.0, range=10_000.0) # interpolated, method='linear'|'nearest'|'cubic'
ssp.isel(depth=0, range=-1)          # positional
ssp.collapse('mean')                 # 2-D → 1-D: 'r0' | 'rmax' | 'mean' | 'median'

ssp.at(depth=50.0)                   # ConfigurationError on a 2-D profile
```

A depth-only slice of a range-dependent profile is ambiguous, and silently
returning the `r = 0` column would be wrong physics on exactly the profiles the
2-D carrier exists for. Pin the range as well, or collapse the range axis
first.

---

## 4. The seabed

Four classes, in increasing scope:

| Class | Is | Carries |
|---|---|---|
| `BoundaryProperties` | one boundary | `sound_speed`, `density`, `attenuation`, `shear_speed`, `shear_attenuation`, `roughness` |
| `SedimentLayer` | one finite layer | the above plus `thickness` |
| `SeabedColumn` | one column at one range | `layers` (possibly empty) over a `halfspace` |
| `Bottom` | the seabed | one or more `SeabedColumn`s plus an optional `ranges` axis |

`env.bottom` is **always** a `Bottom`. Everything else is coerced:

```python
uacpy.Environment(bathymetry=100.0, bottom='sand')      # preset half-space
uacpy.Environment(bathymetry=100.0, bottom=1800.0)      # half-space at 1800 m/s
uacpy.Environment(bathymetry=100.0, bottom=bp)          # a BoundaryProperties
uacpy.Environment(bathymetry=100.0, bottom=column)      # a SeabedColumn
uacpy.Environment(bathymetry=100.0, bottom=bottom)      # a Bottom, used as-is
```

Note what the seabed classes do **not** carry: any notion of where the seafloor
is. That is `env.bathymetry`, always.

### The four shapes a `Bottom` can take

The two axes — range and layering — are independent, giving four combinations:

```python
# Half-space
uacpy.BoundaryProperties.from_preset('sand')

# Layered
uacpy.SeabedColumn.from_presets(
    layers=[('sand', 6.0), ('silt', 14.0)], halfspace='limestone')

# Range-dependent half-space
uacpy.Bottom.from_halfspaces(
    [0.0, 4000.0, 8000.0], sound_speed=[1550.0, 1650.0, 1800.0],
    density=[1.6, 1.9, 2.0], attenuation=[0.3, 0.8, 0.6])

# Range-dependent layered
uacpy.Bottom.from_columns(
    [uacpy.SeabedColumn.from_presets(
        layers=[('sand', 8.0)], halfspace='limestone'),
     uacpy.SeabedColumn.from_presets(
        layers=[('silt', 20.0)], halfspace='chalk')],
    ranges=[0.0, 8000.0])
```

![The four bottom shapes](figures/env_bottom_shapes.png)

The same 120 m water column in all four panels, so the only thing that changes
is the seabed carrier. The half-space is a single hatched fill; the layered
column adds solid per-layer bands above it; the range-dependent half-space
steps through three property sets at the range breaks; the range-dependent
layered case is a geological cross-section with one column per range node
(`P1`, `P2`) and a dashed boundary between them.

`from_halfspaces` accepts a scalar or a per-range array for each property, so
`density=1.8` with three ranges means the same density everywhere.

Predicates tell you which of the four you have: `bottom.is_layered`,
`bottom.is_range_dependent`, `bottom.is_elastic`.

### Presets and the `uacpy.materials` catalogue

Nine class-typical materials, keyed by name:

| | `c_p` (m/s) | `ρ` (g/cm³) | `α_p` (dB/λ) | `c_s` (m/s) | `α_s` (dB/λ) | `ϕ` |
|---|--:|--:|--:|--:|--:|--:|
| `clay` | 1500 | 1.5 | 0.2 | 80 | 1.0 | 8.8 |
| `silt` | 1575 | 1.7 | 1.0 | 80 | 1.5 | 5.4 |
| `sand` | 1650 | 1.9 | 0.8 | 110 | 2.5 | 3.34 |
| `gravel` | 1800 | 2.0 | 0.6 | 180 | 1.5 | −1.5 |
| `moraine` | 1950 | 2.1 | 0.4 | 600 | 1.0 | — |
| `chalk` | 2400 | 2.2 | 0.2 | 1000 | 0.5 | — |
| `limestone` | 3000 | 2.4 | 0.1 | 1500 | 0.2 | — |
| `basalt` | 5250 | 2.7 | 0.1 | 2500 | 0.2 | — |
| `granite` | 5500 | 2.7 | 0.1 | 3000 | 0.2 | — |

Each entry also carries `porosity` (%) and `roughness` (m). `list_materials()`
returns the names, `get_material(name)` a copy of the dict. `ϕ` is the mean
Wentworth grain size, undefined for consolidated rocks. Shear speeds for the
unconsolidated sediments are near-surface (1 m) values — `c_s` grows with depth
below the seabed, so pass an explicit `shear_speed` if you need another depth.

Three constructors read the catalogue:

```python
uacpy.BoundaryProperties.from_preset('sand')                    # half-space
uacpy.SedimentLayer.from_preset('sand', thickness=10.0)         # one layer
uacpy.SeabedColumn.from_presets(layers=[('sand', 10.0), ('silt', 25.0)],
                                halfspace='limestone')          # a stack
```

All three are **fluid by default** — the preset's shear properties are dropped
so the result works in every model. Pass `elastic=True` to keep them. Any extra
keyword overrides a preset field for site-specific tuning, and layer entries may
be `(name, thickness, overrides)` triples.

Two other routes to a half-space: `BoundaryProperties.from_grain_size(phi)`
converts a Wentworth ϕ to explicit `c_p` / `ρ` / `α` via Hamilton (default) or
APL-UW, and `BoundaryProperties(acoustic_type='rigid')` gives a perfectly
reflecting boundary. A bare `grain_size_phi=` is rejected: it carries no
geoacoustics until converted, so inferring a bottom from it would silently use
the default properties.

### What `shear_speed` changes

`shear_speed > 0` anywhere in a column makes the seabed **elastic**, and that
single number changes which models can take your environment honestly:

| Model | With shear |
|---|---|
| [OASES](../models/oases.md), [Scooter](../models/scooter.md) | full elastic physics, natively |
| [Kraken](../models/kraken.md) | auto-routes to `backend='krakenc'` (complex modes) |
| [Bounce](../models/bounce.md) | exact plane-wave `R(θ)` for the elastic stack |
| [Bellhop](../models/bellhop.md) | runs Bounce first, then uses its `.brc` reflection table |
| [RAM](../models/ram.md) | routes to the `rams` backend |
| [SPARC](../models/sparc.md) | shear zeroed, with a warning — see [§7](#7-collapse-policy) |

Physically, shear opens a loss channel that a fluid seabed does not have:
energy converts to shear waves at the interface and leaves the water column, so
treating an elastic seabed as fluid **under-predicts** bottom loss. How much
depends on where `c_s` sits against the water speed. Clay through gravel
(`c_s` 80–180 m/s) cost under 0.25 dB, which you can ignore. The error grows as
`c_s` approaches the water speed and is worst for chalk and limestone, which
lose an extra 11–16 dB near 20–30° grazing. Above `c_s = 1500 m/s` the shear
wave is evanescent below the **shear critical angle** `arccos(1500/c_s)`, so a
basalt or granite basement loses nothing there and a few dB only at steeper
angles.

### Inspecting a seabed

```python
env = uacpy.Environment(
    name='Sand over limestone',
    bathymetry=[(0.0, 100.0), (10_000.0, 160.0)],
    ssp=[(0.0, 1505.0), (160.0, 1492.0)],
    bottom=uacpy.SeabedColumn.from_presets(
        layers=[('sand', 10.0), ('silt', 25.0)],
        halfspace='limestone', elastic=True),
)
uacpy.plot.plot_bottom_properties(env)
```

![Seabed properties](figures/env_bottom_properties.png)

`Bottom` and `Surface` are the two carriers **without** a `.plot()`: placing a
sub-bottom depth axis needs a seafloor depth, which lives on the bathymetry, so
the plotter takes the whole environment. Five panels, one per geoacoustic
property — `c_p`, `c_s`, `ρ`, `α_p`, `α_s` — each drawn against range and
sub-bottom depth, so the layer stack is visible as horizontal bands tracking
the sloping seafloor. The `c_s` panel is the one to check when you are deciding
whether a fluid approximation is defensible: here the limestone half-space is
at 1500 m/s while the sediments are near 100 m/s.

Programmatic access:

```python
column = bottom.at(range=5000.0)          # nearest SeabedColumn — never blends stacks
column.at(depth=5.0)                      # material at 5 m sub-bottom (step lookup)
column.isel(layer=0)                      # the SedimentLayer itself
column.total_thickness()                  # 35.0
column.layer_depths(seafloor_depth=100.0) # [(100.0, 110.0), (110.0, 135.0)]

bottom.halfspace_at(range=2000.0)         # interpolates — half-space bottoms only
bottom.max_total_thickness()
bottom.all_sound_speeds()                 # every real c_p in the seabed
```

`Bottom` and `SeabedColumn` have `at` and `isel` but **no** `eval`: distinct
materials cannot be linearly blended, so selection is always nearest.
`halfspace_at` is the one exception — when every column is a pure half-space,
blending properties is well defined, and it interpolates by default.

---

## 5. `Surface` — the top boundary

`env.surface` carries the acoustic properties of the top boundary, one
`BoundaryProperties` per range node. The default is a vacuum (pressure-release)
surface, which is right for open water.

```python
# Uniform ice canopy
uacpy.Environment(bathymetry=200.0,
                  surface=uacpy.BoundaryProperties.from_preset('chalk', elastic=True))

# Marginal ice zone: open water inshore, ice past 5 km
uacpy.Environment(bathymetry=200.0, surface=[
    (0.0, uacpy.BoundaryProperties(acoustic_type='vacuum')),
    (5000.0, uacpy.BoundaryProperties.from_preset('chalk', elastic=True)),
])
```

A uniform `Surface` delegates attribute reads to its single node, so
`env.surface.acoustic_type` and friends work without indexing. Like `Bottom`,
it has `at` / `isel` but no `eval`.

Interfacial roughness lives on `BoundaryProperties.roughness` (RMS, metres) and
applies to **any** boundary type — a rough but still pressure-release sea
surface is `BoundaryProperties(roughness=2.0)`, which is why roughness never
triggers the half-space inference.

Which models consume it differs by boundary. [Kraken](../models/kraken.md) and
the [OASES](../models/oases.md) family take it at both the sea surface and the
seabed. [Scooter](../models/scooter.md) takes it at the sea surface **only when
that surface is a vacuum** — its solver reads the value through the
pressure-release impedance branch and nowhere else — and does not take it at
the seabed at all. Every other model, and every combination not listed, drops
it with a warning naming what was dropped.

What `Surface` and `altimetry` between them do **not** carry is a bubble layer.
Wind-driven bubbles change the sound speed in the top few metres — a void
fraction of only 1e-6 drops `bubble_soundspeed` by 15.5 m/s (1539.1 → 1523.6 at
its default reference) — and add an excess attenuation no boundary property
reproduces. `uacpy.core.acoustics` has `bubble_soundspeed`, `bubble_resonance`
and `bubble_surface_loss` for quantifying this by hand, but none of the three
feeds an `Environment`: they are calculators, not carriers. Note also that
`bubble_surface_loss` returns a per-bounce amplitude multiplier in `(0, 1]` and
takes its angle in **radians**, unlike `bottom_loss_curve`.

---

## 6. `Absorption` — volume attenuation

Volume attenuation is a property of the water, not of the solver: the same
column absorbs the same energy whichever model runs over it. So it lives on
`env.absorption`, and each model writer reads it to emit the right native
parameters.

| Class | Parameters | Depth-dependent |
|---|---|---|
| `Thorp()` | none | no |
| `FrancoisGarrison(temperature_c, salinity_psu, pH, z_bar_m)` | four, all required | yes |
| `Biological(layers=[(z_top, z_bottom, f0, Q, a0), …])` | per-layer resonance | yes, by layer |
| `ConstantAbsorption(value_db_per_wavelength)` | one | no |

```python
freqs = np.logspace(1, 5.7, 400)

uacpy.Thorp().plot(freqs, label='Thorp')
uacpy.FrancoisGarrison(temperature_c=20.0, salinity_psu=35.0,
                       pH=8.0, z_bar_m=50.0).plot(freqs, depth=50.0)
uacpy.FrancoisGarrison(temperature_c=4.0, salinity_psu=35.0,
                       pH=8.0, z_bar_m=3000.0).plot(freqs, depth=3000.0)
uacpy.ConstantAbsorption(value_db_per_wavelength=1.0e-4).plot(freqs)
uacpy.Biological(layers=[(20.0, 80.0, 1500.0, 4.0, 0.02)]).plot(freqs, depth=50.0)
```

![Absorption models](figures/env_absorption.png)

`Absorption.plot()` requires `frequencies` because absorption *is* a function
of frequency, and takes `depth=` for the depth-dependent models.

Reading the curves:

- **Thorp** implements *Computational Ocean Acoustics* Eq. (1.47) — four terms,
  one per regime: a 3.3×10⁻³ dB/km constant, the boric-acid (1 kHz) and
  magnesium-sulphate (64 kHz) relaxations, and viscous absorption in seawater,
  which does not overtake the magnesium-sulphate term until ~380 kHz.
  Frequency only, no parameters to get wrong, which is why it is the usual
  default. Two things the curve does not tell you. That constant is **not** an
  absorption mechanism — Jensen attributes the regime to leakage out of the deep
  sound channel, not to anything the water does — so below ~50 Hz Thorp and
  Francois–Garrison diverge hard: at 10 Hz Thorp gives 3.3×10⁻³ dB/km against
  FG's 1.2×10⁻⁵, and only FG is modelling absorption. And Eq. (1.47) is fitted at
  4 °C, 35 ppt, pH 8.0 and ~1000 m — the conditions of the 4 °C / 3000 m curve,
  not of the warm shallow one.
- **Francois–Garrison** is the same physics with the T/S/pH/depth dependence
  kept. The two curves stay within ~20 % of each other from 1 to 30 kHz and
  then separate: at 100 kHz the 20 °C / 50 m water absorbs about **twice** as
  much as the 4 °C / 3000 m water (37.8 vs 18.3 dB/km). They part at the other
  end too — below ~300 Hz the cold water absorbs 1.6–1.7× more, almost entirely
  a temperature effect — but both stay under 0.01 dB/km there, so the gap is a
  fraction of a dB even over 100 km. For high-frequency work in cold or deep
  water the choice matters a lot. Of the four parameters `pH` is the one to get
  right at low frequency: it scales the boric-acid term, and moving 8.0 (North
  Atlantic) to 7.7 (North Pacific) takes about a third off the absorption below
  1 kHz.
- **ConstantAbsorption** is flat in dB/**wavelength**, which is a slope-1 line
  in dB/km on log-log: `α[dB/m] = α[dB/λ]·f/c`. It is a calibration knob, not a
  physical model.
- **Biological** is a Lorentzian fish-bladder resonance per layer, applied only
  within `[z_top, z_bottom]`. The peak at `f0` is `a0·Q²` (here
  0.02 × 4² = 0.32 dB/km) and the tail settles at `a0`.

Units throughout the public API are **dB/wavelength** for material attenuation
and dB/km for the plotted volume curves; `uacpy.core.absorption.convert_attenuation_units`
converts between dB/km, dB/m, dB/λ, Nepers/m, `Q` and `L`.

---

## 7. Collapse policy

This is the mechanism every model page links back to.

### The problem

Seven solvers make seven different approximations, and the approximation
decides what each can represent. [Scooter](../models/scooter.md) integrates over
a horizontally stratified medium, so a sloping seafloor is not something it can
express. [Bellhop](../models/bellhop.md) is a fluid ray tracer with no layered
seabed. [RAM](../models/ram.md) is one-way, so it cannot take a rough surface.

A library has three options when you hand a model something it cannot consume:
refuse, silently ignore, or reduce it under a stated policy. uacpy takes the
third.

### The mechanism

Before writing any input file, every model runs your environment through a
projection step that checks each feature against its own capability flags and
**collapses** anything it cannot take natively — emitting one `UserWarning` per
dropped feature, naming the feature, the method used, and the keyword that
overrides it.

```
UserWarning: Scooter does not support range-dependent bathymetry;
collapsed to 300.0 m (method='max', range 100.0–300.0 m).
Override via `collapse={'bathymetry': 'min'|'median'|'mean'|'max'|'initial'}`.
```

Your `env` is never mutated — the model works on a copy.

### The seven keys

| Key | Applies when | Methods | Default |
|---|---|---|---|
| `bathymetry` | seafloor varies with range | `max`, `median`, `mean`, `min`, `initial` | `max` |
| `ssp` | 2-D sound-speed profile | `r0`, `rmax`, `mean`, `median` | `r0` |
| `bottom_range` | seabed varies with range | `r0`, `rmax`, `mean`, `median` | `r0` |
| `bottom_layers` | seabed has layers | `halfspace`, `top_layer`, `volume_average` | `halfspace` |
| `altimetry` | rough/sloped sea surface | `drop` | `drop` |
| `surface` | top properties vary with range | `r0`, `rmax`, `mean`, `median` | `r0` |
| `elastic` | any shear present | `fluid`, `vacuum` | `fluid` |

Set them on the constructor:

```python
from uacpy.models import Scooter

Scooter(collapse={'bathymetry': 'median', 'bottom_layers': 'volume_average'})
```

Unknown keys and invalid methods raise `ConfigurationError` at construction,
not deep inside a writer at `run()` time.

Two behaviours are not `collapse` keys because they have no meaningful choice:
interfacial **roughness** is zeroed for models whose solver has nowhere to put
it, and `altimetry` only supports `'drop'`.

### Order matters for the seabed

The bottom has two orthogonal axes and they collapse in a fixed order: the
range axis first (to one column, layers intact), then the layer axis if needed.
For a model that takes range dependence but not layers — Bellhop — that leaves
a *range-dependent half-space* bottom rather than a single column, which is
strictly more information than collapsing the other way round.

### Some defaults are per model

`DEFAULT_COLLAPSE` above is the project-wide baseline; a model may override it
where its physics prefers something else, and your `collapse={…}` always wins
over both.

| Model | Overrides |
|---|---|
| [Kraken](../models/kraken.md), [Scooter](../models/scooter.md), [SPARC](../models/sparc.md), [OASES](../models/oases.md) | `ssp='mean'`, `bottom_range='median'` |
| [Bounce](../models/bounce.md) | `bottom_range='median'` |
| [Bellhop](../models/bellhop.md), [RAM](../models/ram.md) | none — they take range dependence natively |

The pattern: a modal or spectral solver reduces a 2-D profile by averaging
rather than by picking the source-end cast, because its answer depends on the
whole path.

### What it looks like

```python
env, source, _ = shelf_break()   # the §1 environment: RD layered bottom, rough surface

# Bellhop takes range dependence and the rough surface natively. It has no
# layered bottom, so every column is flattened to its half-space.
bellhop_view = env.copy()
bellhop_view.bottom = env.bottom.collapse(layers='halfspace')

# Scooter is a stratified solver: the bathymetry, the bottom's range axis and
# the sea surface all go, but the layer stack survives intact.
scooter_view = env.copy()
scooter_view.bathymetry = uacpy.Bathymetry.coerce(
    env.get_representative_depth('max'))
scooter_view.bottom = env.bottom.select_range('median')
scooter_view.altimetry = None
```

![Collapse](figures/env_collapse.png)

One environment, three views. The reduced panels are built with the **public**
carrier reductions — `Bottom.collapse(layers=…)`, `Bottom.select_range(…)`,
`Environment.get_representative_depth(…)` — at each model's documented default
method, so each panel reproduces exactly what that model hands its writer.

Bellhop keeps the slope, the two seabed columns and the rough surface, and
loses only the sand and silt layers: past 10 km the fill turns to the pale
chalk half-space that was underneath. Scooter keeps the layers and loses
everything range-dependent — a flat 380 m column (`bathymetry='max'`) carrying
the silt-over-chalk stack picked by `bottom_range='median'`, under a flat
surface.

Neither is a bug, and neither is the model being unhelpful. They are two
different approximations of the same ocean, and the warnings tell you which
parts of your description survived into each.

### Why the design exists

**You can hand the same `Environment` to every model.** That is the whole
point. Cross-model comparison — the most useful validation you can do — only
means something if both runs started from the same description of the water.
Maintaining seven environment shapes, one per solver, would make that
comparison impossible to trust.

**Silence would lie.** A model that quietly ignores your 2-D SSP produces a
plausible-looking field that answers a different question. The warning is the
feature: it names what was dropped, so you can decide whether that mattered.

**Refusing would be worse.** A first look at a shelf-break problem with Kraken
is genuinely useful even though Kraken must collapse the seabed's range axis to
get there. Turning that into a hard error would force you to hand-build a
reduced environment — exactly the reduction the library just did for you, only
undocumented and unwarned.

**The reduction is yours to choose.** `max` is the safe default for bathymetry
because it never puts a receiver below the seafloor, but if you are modelling
the shallow end, `initial` or `min` is the better answer. The policy is a
keyword, not a hard-coded assumption.

When the collapse is not acceptable, that is the signal to change model, not to
fight the policy. The [capability matrix](../models/README.md#capability-matrix)
shows what each model takes natively.

---

## 8. Conventions and gotchas

**Units are SI, without exception.** Metres for depth, range and thickness;
m/s for every sound speed; g/cm³ for density; dB/wavelength for material
attenuation; Hz for frequency. Kilometres exist only on plot axes and inside
native file formats.

**Depth is positive down, altimetry positive up.** Range is measured from the
source, which sits at `r = 0`.

**A result carries no environment.** `Result` holds its own identity and
provenance — the model, backend, frequencies and file paths — never the
`Environment`, `Source` or `Receiver` it ran against. Plotters take them
explicitly:

```python
tl.plot(env=env, source=source)
```

Without `env=`, a TL plot spans only the receiver grid and draws no seabed.
That is correct behaviour, not a missing feature: the result genuinely does not
know what water it came from. See [results](results.md).

**Carriers are stored by reference, not copied.** `Environment` does not
deep-copy the `ssp` / `bottom` / `surface` you pass. Models copy the whole
environment before touching anything, so your object is safe from *them* — but
if you mutate `env.ssp` in place yourself, you mutate the original. Use
`env.copy()` first.

**`at` never fabricates, `eval` interpolates.** Across every uacpy carrier and
result: `at` returns the nearest stored sample, `eval` interpolates, `isel`
indexes positionally. When a carrier has only `at`, it is because interpolating
that quantity is not well defined — layer stacks and boundary types cannot be
blended.

**Validation is at construction.** Non-monotonic axes, negative depths,
mismatched array lengths and unknown preset names all raise
`ConfigurationError` when you build the carrier, not when a solver chokes on
the file it generated.

**`bottom='sand'` is a half-space, not a layer.** The string and scalar
shorthands both build a pure half-space. For a finite sand layer over something
else you need `SeabedColumn.from_presets(layers=[('sand', 10.0)],
halfspace='limestone')`.

---

## 9. Where to go next

- **[Source and receiver](source-receiver.md)** — the other two carriers:
  geometry, frequencies, beam patterns, array layouts.
- **[External data](data.md)** — build an `Environment` from GPS coordinates
  instead of by hand: `fetch_environment` populates bathymetry, sound speed and
  seabed from public datasets and stamps the provenance.
- **[Model index](../models/README.md)** — the capability matrix, and which
  approximation breaks when.
- **[Results](results.md)** · **[Plotting](plotting.md)** — what comes back and
  how to draw it.
- **[Utilities](utilities.md)** — material presets, sound-speed and density
  helpers.
- **[File I/O](io.md)** — the readers and writers that turn these carriers into
  native model input.

---

**See also:** [documentation index](../README.md) ·
[source and receiver](source-receiver.md) · [results](results.md) ·
[plotting](plotting.md) · [external data](data.md) ·
[model index](../models/README.md)
