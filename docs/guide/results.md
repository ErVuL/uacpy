# Results — what a model gives you back

> `uacpy.core.results` · every public name re-exported on `uacpy.*`
> · `Field` · `Rays` · `Arrivals` · `Modes` · `Covariance` · `Replicas`
> · `ReflectionCoefficient` · `ResultStack`

Every model in the package has the same shape: three carriers in
([environment](environment.md), [source and receiver](source-receiver.md)),
one **result** out. This page is about that result — what type you get for
which `RunMode`, how to slice it down to the number you actually wanted, and
what it knows about the run that produced it.

The organising idea is that a result is **data plus identity, and nothing
else**. It carries its values, its axes, and the provenance of the run. It does
not carry the environment, the source or the receiver — see
[§8](#8-identity-and-provenance) for why that is deliberate.

---

## 1. The eight result types

```python
from uacpy.models import Bellhop, RunMode
result = Bellhop().run(env, source, receiver, run_mode=RunMode.COHERENT_TL)
```

| Type | `RunMode` that produces it | Payload | Models |
|---|---|---|---|
| `Field` | `COHERENT_TL`, `INCOHERENT_TL`, `SEMICOHERENT_TL`, `BROADBAND`, `TIME_SERIES` | gridded values + named axes | [Bellhop](../models/bellhop.md), [Kraken](../models/kraken.md), [Scooter](../models/scooter.md), [RAM](../models/ram.md), [SPARC](../models/sparc.md), [OAST / OASP](../models/oases.md) |
| `Rays` | `RAYS`, `EIGENRAYS` | list of ray polylines | [Bellhop](../models/bellhop.md) |
| `Arrivals` | `ARRIVALS` | flat list of arrival events | [Bellhop](../models/bellhop.md) |
| `Modes` | `MODES` | `k` (wavenumbers) + `phi` (mode shapes) | [Kraken](../models/kraken.md) |
| `ReflectionCoefficient` | `REFLECTION` | `R(θ)` magnitude + phase | [Bounce](../models/bounce.md), [OASR](../models/oases.md) |
| `Covariance` | `COVARIANCE` | `C(f, i, j)` across an array | [OASN](../models/oases.md) |
| `Replicas` | `REPLICA` | Green's functions per candidate position | [OASN](../models/oases.md) |
| `ResultStack` | any of the above, with several source depths | a list of slabs + the coordinate they vary along | every field model, [Bellhop](../models/bellhop.md) for rays/arrivals, [`run_parallel`](utilities.md) |

Not every model supports every mode; ask the model rather than guessing:

```python
>>> [m.name for m in Bellhop().supported_modes]
['COHERENT_TL', 'INCOHERENT_TL', 'SEMICOHERENT_TL', 'RAYS', 'EIGENRAYS',
 'ARRIVALS', 'BROADBAND', 'TIME_SERIES']
>>> Bellhop().supports_mode(RunMode.MODES)
False
```

Every one of these types prints a one-line summary of itself, which is the
fastest way to see what you are holding:

```
Field(kind='pressure', unit='Pa', model='Bellhop', f=200 Hz, axes=(depth, range))
Rays(model='Bellhop', f=200 Hz, n_eigenrays=2639)
Arrivals(model='Bellhop', f=200 Hz, n_arrivals=2639)
Modes(model='Kraken', f=200 Hz, n_modes=14, n_z=101)
ReflectionCoefficient(model='Bounce', f=200 Hz, n_θ=571, narrowband)
ResultStack[Field](n_slabs=3, source_depth=[15.0, 50.0, 85.0])
```

---

## 2. `Field` — one container described on three axes

There is exactly one gridded result class. Transmission loss, complex
pressure, a broadband transfer function and a time series are **not** four
types: they are one class described on three independent axes, each derived
from the data rather than read off a flag someone set.

| axis | question it answers | ask it when you need |
|---|---|---|
| `.kind` | *what* is this? | to know whether comparing two fields means anything |
| `.unit` | what is it *measured in*? | to know which direction is louder |
| `.data.dtype` | how is it *stored*? | to know whether there is phase to work with |

There is no `Field.dtype` — the storage axis is read off `.data.dtype`, or as
the boolean `.is_complex`.

Bellhop's `COHERENT_TL` hands back complex pressure — `unit='Pa'`, as the
one-line summaries above show. `.to_dB()` is the step that moves it onto the dB
side of the `.unit` axis:

```python
>>> tl_dB = tl.to_dB()
>>> tl_dB
Field(kind='pressure', unit='dB', model='Bellhop', f=200 Hz, axes=(depth, range))
```

`.kind` is `'pressure'` unless a model tags something else — reverberation
level, for instance. Transmission loss is deliberately **not** its own kind:
`-20·log10|p|` is the same pressure field written in dB, and that difference
is the `.unit` axis's job. The **domain** is not an axis either; it is already
in `.coords`, as a `time` or `frequency` entry.

| `.data` dtype | `.coords` contains | `.kind` | `.unit` | Physical meaning |
|---|---|---|---|---|
| complex | a `frequency` axis | `'pressure'` | `'Pa'` | `H(f)` |
| real | a `time` axis | `'pressure'` | `'Pa'` | `p(t)` |
| complex | neither | `'pressure'` | `'Pa'` | complex pressure `p` |
| real | neither | `'pressure'` | `'dB'` | transmission loss, dB |

Real data alone does not mean dB — a time trace is linear pressure, which is
why `.unit` consults the axes too.

Nothing about dimensionality enters into it. `tl.at(depth=20)` is a 1-D range
cut and still dB; `tl.max()` is a scalar and still dB.

![One Field, four states](figures/results_field_kinds.png)

All four panels above are the same class:

```python
env, source, receiver = shallow_water()
pressure = Bellhop(n_beams=3000).run(env, source, receiver)

source_bb = uacpy.Source(depths=25.0,
                         frequencies=np.linspace(150.0, 450.0, 192))
point = uacpy.Receiver(depths=60.0, ranges=3000.0)
H = Bellhop(n_beams=3000).run(env, source_bb, point,
                              run_mode=RunMode.BROADBAND)

spectrum = H.isel(depth=0, range=0)
trace = H.to_time_trace()

pressure.to_dB().plot(env=env)          # real    + {depth, range} → dB
pressure.plot(env=env, value='phase')   # complex + {depth, range} → Pa
spectrum.plot(value='mag_dB')           # complex + {frequency}    → H(f)
trace.plot()                            # real    + {time}         → p(t)
```

### Why the axes are kept apart

Collapsing them is not hypothetical — each collapse has produced a bug:

- `Field.max` inferred dB-ness from the quantity. Introducing a reverberation
  kind made it return the *quietest* cell of a dB grid instead of the loudest.
- `compare_models` keyed on representation, so it refused a RAM TL field
  against a Kraken complex one — the ordinary cross-model comparison, since
  those are one quantity written two ways.
- `replica_bank_from_field` asked for a quantity when what matched-field
  processing actually needs is **phase**, i.e. the dtype. A real TL field
  answers `kind='pressure'` and would have sailed past a kind-only guard.

So: compare on `.kind`, decide loudness on `.unit`, and require phase on the
dtype.

### Adding a quantity

Every quantity is registered once, in
[`core/results/quantities.py`](../../uacpy/core/results/quantities.py), with
the units it may carry and the label for each pairing:

```python
Quantity('reverberation', {'dB': 'Reverberation loss (dB re unit source)'}),
```

A model then tags it — `metadata['kind'] = 'reverberation'` — and the label,
the validation and the colour scale follow. An **unregistered** kind is
refused when the `Field` is constructed rather than defaulting silently, since
a typo'd tag otherwise resurfaces much later as a wrong colour scale or a
wrong `.max()` direction with nothing pointing back at the model that set it.

Colormaps are deliberately *not* in that registry — they are a rendering
choice and live in `visualization/style.py`, which `core/` must not depend on.

Two things it deliberately does not model: unit conversion, and a per-unit
"which way is louder" flag. Two quantities are inverted — transmission loss,
and OASS reverberation, which OASES writes as `-10·log10 E[|p_scat|²]` — so
two documented special cases in `Field.max` beat a field that would read `+1`
in every row but two. Both read as losses everywhere: `Field.max` returns the
smallest cell, and a 1-D cut of either draws its value axis downward.

The consequence worth internalising: **operations that change the dtype or the
axes change what the field is**. `pressure.to_dB()` moves `Pa` to `dB`.
`H.at(frequency=300)` drops the frequency axis. `H.to_time_trace()` returns to
the time domain. You never declare any of it.

### Canonical layouts

`data.shape` follows the insertion order of `coords`, and every uacpy producer
uses the same order: `source_depth → depth → range → frequency` (or `time`).

| `coords` | `.unit` | Produced by |
|---|---|---|
| `{depth, range}`, complex | `Pa` | `COHERENT_TL` from every field model |
| `{depth, range}`, real | `dB` | Kraken `INCOHERENT_TL`, OAST `COHERENT_TL`, any `.to_dB()` |
| `{depth, range, frequency}` | `Pa` | `BROADBAND` |
| `{depth, range, time}` | `Pa` | `TIME_SERIES` (natively from [SPARC](../models/sparc.md)) |
| `{time}` | `Pa` | `to_time_trace()` on one cell |
| `{source_depth, depth, range}` | `Pa` | a replica bank for [matched-field processing](sonar.md) |

Every row is `kind='pressure'`; a model producing another quantity (OASS
reverberation) tags its own `.kind`.

Depths and ranges are **metres** throughout; kilometres appear only on plot
axes. See [units and conventions](../README.md#conventions).

---

## 3. Slicing — `at`, `isel`, `eval`, `max`, and `.pinned`

A model hands you a grid; you almost always want a cut through it. All four
slicers share one rule:

> **A collapsed axis is dropped from `coords` and its value is recorded in
> `pinned`.**

That is the whole mechanism. `coords` is what remains — the axes the field is
still a function of. `pinned` is the running record of where you are standing.

| Call | Selection | Fabricates values? |
|---|---|---|
| `.at(depth=60.0)` | nearest **stored sample** to the label | no |
| `.isel(depth=0)` | positional index (negatives allowed) | no |
| `.eval(depth=60.0, method='linear')` | interpolated onto the label | **yes** |
| `.max()` | the loudest cell, every axis at once | no |

![Slicing a Field](figures/results_slicing.png)

```python
tl = Bellhop(n_beams=3000).run(env, source, receiver).to_dB()
loudest = tl.max()

tl.plot(env=env, source=source)     # coords = {depth, range}  → heatmap
tl.at(depth=60.0).plot()            # coords = {range}         → range cut
tl.at(range=3000.0).plot()          # coords = {depth}         → depth cut
```

The two cut panels were given **no** title. The titles you see — `Depth =
60.4 m` and `Range = 2.99 km` — are the plotter rendering `field.pinned`,
and they are the honest answer to "what did I actually get?":

```python
>>> cut = tl.at(depth=60.0)
>>> cut
Field(kind='pressure', unit='dB', model='Bellhop', f=200 Hz, axes=(range))
>>> cut.coords.keys()
dict_keys(['range'])
>>> cut.pinned
{'depth': 60.3939393939394}           # nearest receiver depth, not 60.0
```

### Narrowing an axis instead of collapsing it — `window` and `shift`

The four slicers above all *collapse*: one sample survives and the axis is
gone. Two more methods leave the axis in place.

| Call | Effect | Axis survives? |
|---|---|---|
| `.window(time=(0.0, 0.18))` | drop samples outside an inclusive label range | **yes** |
| `.shift(time=-0.02)` | translate the coordinate, data untouched | **yes** |

`window` takes `None` for either end, so `window(time=(0.0, None))` trims a
pre-roll and nothing else, and it **raises when the window keeps no sample** —
an empty axis is not a smaller field but a field with nothing in it, and every
later slice of it would fail somewhere less obvious.

Together they put several models on one display axis, which is what comparing
them takes:

```python
# A time-marching solver integrates from a negative pre-roll; an IFFT one
# starts at zero and carries the source waveform's own peak offset. Move the
# emission to t=0, then cut both to the same window.
aligned = synthesised.shift(time=-waveform_peak).window(time=(0.0, t_max))
```

Both go through `id_kwargs()` (§8), so the result keeps the whole identity
surface — model, backend, frequencies, source depths, phase reference,
`model_source` and `metadata`. Rebuilding a `Field` by hand to do this is how
`metadata` gets dropped, and `metadata` is where `kind` lives, so a tagged
quantity silently becomes an untagged one.

### The same move in the frequency domain — `remove_delay`

Shifting a time axis by `-τ` and removing a delay `τ` from `H(f)` are one
operation on two representations, so `remove_delay` is `shift`'s frequency-domain
twin:

```python
H.remove_delay(seconds=tau)          # a delay you already know
H.remove_delay(sound_speed=1500.0)   # take r/c from the field's own range
```

**There is no default τ.** The delay worth removing is the geometric travel time
`r/c`, and a `Field` carries `r` but not `c` — the sound speed belongs to the
Environment that produced it, and guessing 1500 m/s would be the package
inventing a number you did not supply. Give it `sound_speed` and it does the
division itself; on a field that still has a range axis, **each range is
advanced by its own `r/c`** (the reduced-time convention).

Why you need it: the phase of a delay wraps at `1/τ` in frequency, so on a grid
of spacing `Δf` it is unambiguous only for `τ < 1/(2·Δf)`. A 3.3 s travel time
on a 1 Hz grid is aliased beyond reading — and two models sampled on *different*
grids alias differently and appear to disagree when they do not. Read the delay
back off the unwrapped phase slope of one such field and the coarse grid says
+340 ms while a finer one says −60 ms, for a true 3340 ms; compensated, both say
the 7 ms residual they can actually resolve. The magnitude is untouched
throughout — it is a unit-modulus factor — so `|H|` and any TL from it are
unchanged.

`at` asked for 60 m and got 60.394 m, because that is where a receiver
actually is. Nothing was interpolated and nothing was invented. Slicing
composes, and `pinned` accumulates:

```python
>>> tl.at(depth=60.0).at(range=3000.0)
Field(kind='pressure', unit='dB', model='Bellhop', f=200 Hz, axes=(scalar))
>>> _.pinned
{'depth': 60.3939393939394, 'range': 2992.169}
>>> _.data                       # 0-D array — a single number
array(66.18529739)
```

`.max()` does the same thing for every axis in one step (the white star on the
figure). For a complex or time-domain field it takes the global argmax of
`|data|`; for real dB it takes the **minimum** finite TL, because smaller dB is
louder. `NaN` no-data cells are skipped:

```python
>>> tl.max().pinned
{'depth': 21.78787878787879, 'range': 50.0}
```

Pinning an identity-bearing axis narrows the identity too, so a slice never
misreports itself: `H.at(frequency=300)` comes back with
`frequencies == [299.2]` and an `f0` to match.

### How slicing decides what a plot looks like

[`plotting.md`](plotting.md) owns rendering, but the branch it takes is chosen
by the `coords` **you** left behind:

| surviving axes | render branch |
|---|---|
| 2 | heatmap (or stacked traces with `stacked=True` when one axis is `time`) |
| 1 | line plot |
| 3 or more | `ConfigurationError` — slice it first |

```python
>>> H.plot()
ConfigurationError: plot_field: cannot plot a 3-axis field (coords
['depth', 'range', 'frequency']); slice it first with .at(...) / .isel(...)
so 1 or 2 axes remain.
```

So `.at` / `.isel` are not just data reduction — they are how you choose the
view. [`plotting.md`](plotting.md) covers what each branch then draws.

---

## 4. Derived views and grid operations

None of these mutate the field; each returns a fresh array or a fresh `Field`.

| Accessor | Returns | Notes |
|---|---|---|
| `.dB` | ndarray, dB | `-20·log10\|data\|` for complex data; a **read-only view** when data is already dB. Raises for a time-domain field. |
| `.tl` | ndarray, dB | transmission loss: exactly `.dB`, on **pressure-kind fields only** — any other kind raises and points at `.dB` |
| `.p` | ndarray, complex | read-only view; raises when data is real (the phase is gone) |
| `.magnitude` | ndarray | `\|data\|`, complex only |
| `.phase` | ndarray, radians | `angle(data)`, complex only |
| `.to_dB()` | `Field` | the dB counterpart of this field; a no-op when already real |
| `.shape`, `.axes`, `.is_complex` | — | shape, axis names, dtype test |
| `.depths`, `.ranges`, `.times` | ndarray or `None` | the coord vectors by name |
| `.dt`, `.sample_rate` | float | time-axis spacing; `0.0` when not time-resolved |

`.dB` and `.p` hand back read-only views on purpose: the array *is* the
result's payload, and `p = field.p; p *= 2` would otherwise silently corrupt
it. Copy first if you need to modify.

Grid-level operations, all requiring the canonical `['depth', 'range']` layout
(or `['depth', 'range', 'time']` where noted):

| Method | What it does |
|---|---|
| `.resample_to(depths=…, ranges=…)` | interpolate onto a new grid; out-of-bounds is `NaN`. Keyword-only and depth-first. |
| `.mask_below_seafloor(env)` | `NaN` out samples under the bathymetry |
| `.extract_tone(f)` | steady-state complex pressure at one frequency from a `{depth, range, time}` field |
| `.get_spectrum()` | `(freqs, X)` — real FFT along the time axis |
| `.to_dict()` / `Field.from_dict(d)` | round-trip to plain arrays for caching or `np.savez` |
| `.copy()` | deep copy, symmetric with the carriers |

`to_dict` is the supported way to transform a field's values and keep it a
field — [SPARC's record section](../models/sparc.md) uses exactly that to
apply a `√r` gain before plotting.

---

## 5. `ResultStack` — one run, several source depths

A `Source` with several depths runs on every field model in a single call
and returns a `ResultStack` — a list of slabs plus the coordinate they vary
along, one slab per source. Bellhop writes every depth into one deck, and
so do Kraken in its TL modes and Scooter in `COHERENT_TL`; the other engines
run once per depth inside `run()`.

`stack.superpose()` adds the slabs' complex pressure — the elements of one
array, driven with a fixed relative phase — and `superpose(coherent=False)`
adds their intensity instead, for sources that are mutually incoherent. N
identical sources give `20·log10(N)` one way and `10·log10(N)` the other, so
the choice is a statement about the sources rather than about the arithmetic.

Either total keeps transmission loss's reference — one unit source at 1 m —
so the array's gain sits inside the number and the map is captioned *Total
level*, not TL. Give the `Source` a `source_level_dB=` and
`total.at_source_level()` returns an absolute received level instead, in dB
re 1 µPa, which carries no such ambiguity. See example 40.

![A ResultStack of Field slabs](figures/results_stack.png)

```python
source = uacpy.Source(depths=[15.0, 50.0, 85.0], frequencies=200.0)
stack = Bellhop(n_beams=3000).run(env, source, receiver)
stack.plot(env=env, title='ResultStack[Field] — one slab per source depth')
```

Every slab is a full result of the same concrete type, sharing `model`,
`backend` and every identity axis except the stacking one — `ResultStack`
validates that at construction rather than letting a mismatched bundle through.

| Access | Gives you |
|---|---|
| `stack[i]` | the `i`-th slab |
| `stack.at(source_depth=50.0)` | the slab nearest a label |
| `stack.isel(source_depth=1)` | the slab at a position |
| `for depth, slab in stack:` | `(coordinate, slab)` pairs |
| `len(stack)`, `stack.n_slabs` | slab count |
| `stack.dB` | one dense array, shape `(n_slabs, *slab.dB.shape)` |
| `stack.tl` | `stack.dB` for pressure slabs; any other kind raises |
| `stack.model`, `.backend`, `.frequencies`, `.source_depths` | the identity every slab agrees on |
| `stack.superpose()` | one `Field`: `Σ wᵢ·pᵢ` over the slabs with the `Source`'s `weights` |
| `stack.superpose([w1, w2, …])` | the same sum with weights given here |

`stack.dB` exists so generic code can read `result.dB` whether one or many
source depths were asked for.

`superpose` is how several sources are driven together: each slab is the
complex field of one unit-amplitude source, the engines are linear in the
source amplitude, so the weighted sum on the shared grid is the field of the
weighted array. The sum keeps the slabs' grid and `phase_reference`, widens
`source_depths` to every depth it added and records the depths and weights
in `metadata['superposed_sources']`. A `TIME_SERIES` stack sums its real
traces with real weights. A stack of real dB values has lost its phase and
refuses — superpose the complex field the run returned, before `to_dB()`.

```python
pair = uacpy.Source(depths=[40.0, 60.0], frequencies=200.0, weights=[1, -1])
stack = Kraken().run(env, pair, receiver)       # ResultStack[Field], 2 slabs
field = stack.superpose()                       # p(40 m) − p(60 m)
same = stack.superpose([1, -1])                 # weights given at the call
```

Bellhop's `RAYS` / `ARRIVALS` / `EIGENRAYS` stack the same way and give
`ResultStack[Rays]` / `ResultStack[Arrivals]`; those slabs are not fields and
do not superpose. Every other non-field mode (`MODES`, `REFLECTION`,
`COVARIANCE`, `REPLICA`, `REVERBERATION`) takes one source depth per run and
says so:

```python
>>> Kraken().run(env, uacpy.Source(depths=[20.0, 60.0], frequencies=200.0),
...              receiver, run_mode=RunMode.MODES)
ConfigurationError: Kraken takes a single source depth per MODES run; got 2:
[20.0, 60.0]. A multi-depth Source stacks only in the
field modes (COHERENT_TL / INCOHERENT_TL / SEMICOHERENT_TL / BROADBAND /
TIME_SERIES); for MODES loop over single-depth Sources externally.
```

A sweep over any other parameter goes through [`run_parallel`](utilities.md);
`.stack()` on the outcome gives the same `ResultStack`, along whatever
coordinate you varied.

---

## 6. From `H(f)` to `p(t)`

A `BROADBAND` run gives you a complex transfer function on a `frequency` axis.
Two methods turn it into time:

| Method | Input | Output |
|---|---|---|
| `.to_time_trace(depth=…, range=…)` | one `(depth, range)` cell | `Field`, `coords={'time'}` — the band-limited impulse response |
| `.synthesize_time_series(waveform, sample_rate)` | a source waveform | `Field`, `coords={'depth', 'range', 'time'}` — every cell convolved |
| `.truncate_response(duration, origin='peak', window='boxcar')` | a pulse length | `Field`, same coords — `H(f)` with its impulse response cut to what a pulse that long can overlap. `H(f)` as a model returns it is the **continuous-wave** answer, every path at once; two copies of a `T`-long pulse interfere only where they overlap, so arrivals more than `T` apart arrive separately and do not add. The window reaches `duration` **either side** of the origin. Works on any model's `H`, which is the point — a wave model has no paths to drop, so it has to be found in the response. Warns when the `1/Δf` record has already folded the response, because a fold cannot be told from an early arrival |

Both require the canonical `['depth', 'range', 'frequency']` layout. With no
arguments, `to_time_trace` takes the middle depth and the first range.

**Channel simulation — one signal, one receiver.** There are two routes and
the model-level one came first: `RunMode.TIME_SERIES` with `source_waveform=`
and `sample_rate=` runs the solver straight to `p(t)` on Bellhop, Kraken,
Scooter, RAM, SPARC and OASP — see
[example 19](../../uacpy/examples/example_19_broadband_comparison.py), which
reconstructs a chirp at one receiver across eight solvers.

Use `to_time_trace` when you already hold an `H(f)` — from a `BROADBAND` run
you made for the level maps above, say — and want a receiver's waveform out
of it without a second model run:

```python
trace = H.to_time_trace(depth=60, range=2000, waveform=tx, sample_rate=fs)
```

Use `synthesize_time_series` when you want every cell; this when you want a
receiver. A `depth` or `range` outside the grid warns — the match is to the
nearest stored coordinate, so a receiver past the panel's edge would
otherwise become the edge cell and return a perfectly ordinary trace of the
wrong place.

**Size the frequency grid from the arrivals, not by eye.** The record is
`1/Δf` long and circular, so a channel whose multipath outlasts it folds onto
its own early part and reads as extra arrivals. `Arrivals.synthesis_band`
picks Δf for you:

```python
f = arrivals.synthesis_band(bandwidth=1600.0, centre=3000.0,
                            energy_fraction=0.999)
H = model.run(env, Source(depths=30.0, frequencies=f), rcv,
              run_mode=RunMode.BROADBAND, frequencies=f)
```

On a 2 km Pekeris path with 41 ms of rms delay spread it chose Δf = 4.35 Hz —
a 230 ms record for 191 ms of arrival energy. A hand-picked 10 Hz would have
given 100 ms and wrapped half the channel.

**Add the pulse yourself.** That default budgets the *arrivals* only, so a
20 ms pulse through a 2.7 ms channel can be handed a 5 ms record and wrap
completely — `synthesis_band` stays silent because the arrivals do fit. State
the whole budget with `record=`, which bypasses the helper's own `margin`, so
supply headroom too:

```python
span = arrivals.energy_support(0.999) + len(waveform) / fs
f = arrivals.synthesis_band(bandwidth=B, centre=f0, record=6.0 * span)
```

Several times over, not a token factor: the pulse has to fit *and* the band
edge's precursor needs somewhere to sit. Measured on one geometry, energy
arriving before the first path could ran 21.2 % at 1.5×, 0.19 % at 4×, and
0.19 % at 60× — so it converges around 4×, and that floor is the band edge,
not a fold.

**Two routes to a received signal, and they are not identical.** The wave
route above keeps `H(f)` across the band. The path route —
`Arrivals.channel_taps` / `acoustic_signal.simulate_reception` — freezes the
arrival amplitudes at one carrier, which is the tap model's narrowband
assumption. On the case above the two peak within 2.8 ms of each other out of
a 1339 ms travel time, with envelopes correlating +0.70 unaligned. For modem
work `Arrivals.channel_regime(symbol_rate)` says which regime you are in
(that channel: frequency-selective, 41 ms of spread = 82 symbols at 2 kBd).

![Time-series synthesis](figures/results_time_synthesis.png)

**And back again.** `Field.to_transfer_function()` is the way from a
time-domain `Field` to `H(f)`, as a carrier rather than as raw arrays:

```python
trace = H.to_time_trace(depth=50.0, range=5e3, window='none')
back  = trace.to_transfer_function()     # same band, same numbers
```

The band is **restricted, not extended**. An `rfft` of an `N`-sample record
returns every bin up to Nyquist, but a trace synthesised from a 100–995 Hz
field supports nothing outside that — the rest are the synthesis's own edges,
and returning them would invent data. The band comes from the Field's
identity, or from `band=`.

The result is a complex `['…', 'frequency']` Field, so
`plot_transfer_function`, `broadband_loss` and `truncate_response` all take
it. Note that `window='hann'` (the default of `to_time_trace`) is a real
modification of the signal: the round trip closes to 7e-16 with
`window='none'` and differs visibly with the taper on, which is the taper
working, not the transform failing.

The method is a wrapper: the transform, the `t0` rotation and the band cut are
[`transfer_function_from_impulse_response`](signal.md#61-going-back-h--h),
which you can call on your own arrays. What the Field adds is the axis
bookkeeping, the band read off its identity, the metadata, and the `dt` that
carries the result into the density convention — see that section for why
there are two conventions and what mixing them costs.

For the two narrower tools: `get_spectrum` is the raw `rfft` (every bin, no
rotation, arrays not a Field), and `extract_tone` is the careful
single-frequency answer, evaluated AT the frequency rather than at the
nearest bin.

---

## 6a. The level a signal with bandwidth or duration reaches

A model's TL is the **continuous-wave** answer. Two quantities say what a real
signal reaches, and they answer different questions:

**Both start from a `RunMode.BROADBAND` run.** They need a complex `H(f)` with
a `frequency` axis, and that is the only run mode that produces one —
`COHERENT_TL` refuses more than one source frequency by construction. Bellhop,
Kraken, Scooter and RAM all offer it:

```python
f = np.arange(800.0, 1201.0, 20.0)
H = model.run(env, Source(depths=20.0, frequencies=f), rcv,
              run_mode=RunMode.BROADBAND, frequencies=f)
```

`Δf` sets the record length `1/Δf` that the exposure integral and any time
trace live in, so size it from the channel — `Arrivals.synthesis_band` does
that (see §6).

| Method | Returns | Question |
|---|---|---|
| `.broadband_loss(spectrum=None)` | `Field`, frequency axis gone, `kind='pressure'` `unit='dB'` — so `.tl` and `.plot()` treat it as the TL map it is | how much of the continuous-wave interference does my **bandwidth** survive? |
| `.sound_exposure_level(waveform, sample_rate)` | `Field`, `kind='sound_exposure'` — a **level**, so it reads upward and never shares a colorbar with TL | how much **energy** does one transmission deliver? |
| `.peak_sound_pressure_level(waveform, sample_rate)` | `Field`, `kind='peak_pressure'` — a level of its own kind, distinct from `'level'` | how loud is its **single loudest excursion**? |

`broadband_loss` is the frequency average of the **coherent** `|H|²`, weighted
by `|spectrum|²`:

```python
# the simplest form: hand it the signal, get that signal's TL map
loss = H.broadband_loss(waveform=my_chirp, sample_rate=fs)
loss.plot(env=env)

# or a plain band, when you want the band rather than a specific waveform
band = H.window(frequency=(40e3 - 25, 40e3 + 25))    # a 20 ms burst's 1/T
band.broadband_loss().plot(env=env)
```

`waveform=` evaluates the spectrum on the field's axis with the same DTFT
`synthesize_time_series` uses, so `SEL = ESL − TPL` closes exactly. Do **not**
`np.interp` an `rfft` onto the axis and pass it as `spectrum=` — the two grids
rarely coincide and interpolation is a triangular-kernel convolution, not a
resampling; measured at 0.13 dB of error on a plain tone burst.

This **is** the transmission loss of a transient signal, in the ordinary
sense of the word: Ainslie's broadband propagation loss (§11.3.3, Eq. 11.46),
which is Abraham's pulse propagation loss `L_p = ∫|U|²df / ∫|H|²|U|²df`
(§3.2.4.2) once a source spectrum weights it, and Ainslie's total path loss
(§3.3.2.1) in energy form. Received level is `SL − TPL` for mean square and
`SEL = ESL − TPL` for energy, exactly as a CW TL is used.

Two properties make it a generalisation of TL rather than a lookalike:

* **It converges to the CW loss as the band narrows.** On a two-path channel
  at a near-cancellation frequency: 12.19 dB of error at B = 200 Hz, 4.12 at
  50, 0.295 at 10, 0.0166 at 2, and exactly 0 at one bin.
* **It is a functional of `|S(f)|` on the field's own axis.** Phase does not
  enter at that point, so passing `spectrum=` and `spectrum=` with any phase
  gives the same map. It does **not** follow that two waveforms sharing a
  nominal band share a loss: `waveform=` evaluates the continuous DTFT on the
  field's axis, and off a waveform's own DFT grid that is not fixed by its
  `|rfft|`. Whether phase matters at all is decided by **grid alignment**:
  `|DTFT|` equals `|rfft|` only on the waveform's own DFT bins, spaced
  `1/T`. When `T·Δf` is an integer the field's axis is a subset of those
  bins, phase cannot reach the answer, and two same-`|rfft|` waveforms give
  **exactly** the same loss. Off that alignment it can. On a two-path
  channel over a 25 Hz axis, 500 Hz bursts of 20 and 40 cycles (`T·Δf` = 1
  and 2) spread 0.0000 dB, while 19, 21 and 41 cycles (0.95, 1.05, 2.05)
  spread 0.066, 0.055 and 0.043 — and 21 cycles is *narrower* than 20, so
  this is not a bandwidth effect. Either way different signals get
  different losses, which is the point of passing the waveform.

It is **not** `RunMode.INCOHERENT_TL` (Ainslie's Eq. 11.47), which drops the
arrivals' relative phase — Ainslie marks that approximation invalid within a
few wavelengths of a boundary, which is every Lloyd mirror. Jensen's
semicoherent loss (§3.3.5.4) is a third thing again, and he calls that family
"somewhat informal and partially empirically based".

`sound_exposure_level` integrates `p(t)²` from `synthesize_time_series` —
Abraham's energy flux density integral (§3.2.1.5), ISO 18405's sound exposure:

```python
sel = H.sound_exposure_level(tone_burst(40e3, 800, fs)[1], fs)   # dB re 1 µPa²s
```

There is **no source-level argument** — the level rides on the waveform's
amplitude, because `synthesize_time_series` reproduces the source waveform
where `H` is unity. For a source level `SL` in dB re 1 µPa at 1 m:

```python
unit = waveform / np.sqrt(np.mean(waveform ** 2))
sel  = H.sound_exposure_level(unit * 1e-6 * 10 ** (SL / 20), fs)
```

Pass the unit waveform instead and you get the propagation term alone, with
the source level added afterwards as an **energy** source level,
`ESL = SL + 10log10(T)` (Ainslie Eq. 3.155). The two agree exactly: `SL` =
190 dB at 1 m, a 10 ms burst, 100 m of spherical spreading gives
190 − 40 + 10log10(0.01) = **130 dB re 1 µPa²s** either way.

The last two are the **dual metric** impulsive-exposure criteria are written
in — "frequency-weighted SEL and unweighted peak sound pressure level", either
one exceeding its threshold being sufficient (Southall et al. 2019). Neither
substitutes for the other, and a peak is a property of the waveform in *time*,
so no reduction of `|H(f)|` yields it. One asymmetry worth knowing: SEL is
exactly invariant to the synthesis length (Parseval), while the peak is not —
a maximum is a sample. Sweeping `nfft` from 320 to 65536 moved the peak
**0.0615 dB** and the SEL **0.0000 dB**; the peak is converged by
`nfft ≈ 4096`.

Its band window defaults to `'none'` rather than `synthesize_time_series`'s
`'hann'`, because a taper removes energy the integral is defined to count: a
5-cycle 500 Hz burst over a 25 Hz–4 kHz band reads 52.675 dB flat and
35.676 dB tapered, 17 dB light, purely because 500 Hz sits low in the band.

**Neither one gates.** Both keep a late path's energy; only its *interference*
goes. Discarding the path itself is `.truncate_response`, which answers a
receiver-side question about one cell.

```python
from uacpy.acoustic_signal import lfm_chirp

source = uacpy.Source(depths=25.0, frequencies=np.arange(150.0, 450.1, 0.5))
receiver = uacpy.Receiver(depths=60.0, ranges=np.linspace(1000.0, 3000.0, 9))
H = Bellhop(n_beams=3000).run(env, source, receiver, run_mode=RunMode.BROADBAND)

sample_rate = 4000.0
t_src, waveform = lfm_chirp(150.0, 450.0, 0.04, sample_rate)
series = H.synthesize_time_series(waveform, sample_rate)

series.isel(depth=0).plot(stacked=True)
```

The moveout across the nine traces is the travel time to each range — the
figure is a record section, and it comes out of the same `Field` container as
everything else.

**The frequency grid sets the record length.** The synthesis places each model
frequency at bin `round(f / Δf)`, so the trace it can represent is exactly
`1/Δf` long. That is why the run above uses `Δf = 0.5 Hz` (a 2 s window, enough
to hold the spread of arrivals out to 3 km) rather than a coarser grid: a
longer record needs **more model frequencies**, not a bigger `nfft`. uacpy
warns rather than aliasing silently when the receiver span outruns the window.

Other knobs: `window=` tapers the band edges before the IFFT (`'hann'` by
default — a hard band edge rings), `t_start=` moves the window, and `nfft=`
overrides the auto-sizing. Both methods tag their output
`phase_reference='time_domain_native'`, because from there on the payload is
`p(t)` whatever convention `H(f)` carried.

---

## 7. The other result types

### Rays and Arrivals — the sparse pair

![Eigenrays and arrivals](figures/results_rays_arrivals.png)

```python
point = uacpy.Receiver(depths=60.0, ranges=3000.0)
model = Bellhop(n_beams=4000, alpha=(-45.0, 45.0))
eig = model.run(env, source, point, run_mode=RunMode.EIGENRAYS)
arr = model.run(env, source, point, run_mode=RunMode.ARRIVALS)

eig.top_n_by_miss(12).plot(env=env)
arr.plot()
```

Both are pure data containers whose filters return new objects and never call
back into a solver, so they chain freely:

| `Rays` | |
|---|---|
| `.rays` | list of dicts: `r`, `z` (metres), `alpha`, `n_top_bounces`, `n_bot_bounces` |
| `.is_eigen` | `True` for an eigenray solve, set from the run type |
| `.filter_by_bounces(kind=…, top=…, bot=…)` | `'direct'` / `'surface'` / `'bottom'` / `'both'`, or exact counts / `(lo, hi)` ranges |
| `.filter_by_launch_angle(min_deg, max_deg)`, `.filter_nfirst(n)`, `.filter(predicate)` | subsets |
| `.sorted_by_miss()`, `.top_n_by_miss(n)`, `.filter_by_miss_distance(max_miss)` | closest approach to the receiver; each kept ray gains `miss_distance_m`. Measured to the polyline's **segments**, not its vertices, so the number is the ray's geometry and not the ray step — a ray passing through the receiver misses by zero however coarsely it was sampled |
| `.truncate_at_receiver()` | clip each polyline at its closest approach |

`Rays`'s miss-distance helpers default their target to the receiver the run
was aimed at, so `top_n_by_miss(12)` needs no coordinates when the receiver is
a single point.

| `Arrivals` | |
|---|---|
| `.arrivals` | list of dicts: `delay`, `amplitude`, `phase`, bounce counts, `src_angle`, `rcv_angle`, `kind`, cell indices |
| `.delays`, `.amplitudes`, `.phases` | bulk ndarray views — `phases` is converted to **radians** |
| `.received_amplitudes` | complex amplitude that actually **arrives**: `A·exp(ω·Im τ)·exp(i·phase)`. Use this to compare, sum or synthesise. `.amplitudes` is Bellhop's *geometric* column and carries no volume absorption — Bellhop keeps that in the imaginary travel time, so on that column a long absorbed path stands at its lossless height |
| `.filter_by_bounces(…)`, `.in_delay_window(t_min, t_max)`, `.filter(predicate)` | subsets |
| `.sorted_by_amplitude()`, `.top_n_by_amplitude(n)` | rank by strength |
| `.rms_delay_spread()` | energy-weighted width of the arrival pattern (s) — how much the multipath smears a pulse, and far less tail-driven than `ptp(delays)` |
| `.energy_support(fraction=0.999)` | delay span holding that share of the energy (s) — the span a synthesis window has to cover, unmoved by a faint straggler the way `ptp(delays)` is |
| `.synthesis_band(bandwidth=…, record=…)` | frequency grid to synthesise these arrivals on — a record is `1/Δf` long, so the window, not the bandwidth, decides the spacing. State `record` (seconds) or let it come from `energy_support`; anything left outside folds back onto the early trace, and it says so |
| `.transfer_function(frequencies, receiver=None)` | `H(f) = Σ aᵢ(f)·e^{iφᵢ}·e^{−i2πfτᵢ}` as a single-cell broadband `Field`, with `aᵢ(f)` the received amplitude at each `f` so the absorption in `Im τ` is applied per frequency — the same expression `RunMode.BROADBAND` evaluates, reproduced to floating point. It exists separately because these arrivals can be **filtered first**: `arr.in_delay_window(t0, t0 + T).transfer_function(grid)` is the channel a `T`-long pulse sees, and which paths belong in the sum is a question about the signal, not about the channel |
| `.channel_taps(symbol_rate, carrier=…, sps=1, pulse=None)` | the arrivals as a modem's baseband channel: `ChannelTaps` whose `.taps[k]` is `Σ aᵢ·e^{iφᵢ}·e^{−i2πf_cτᵢ}·g(kT − τᵢ)` with `aᵢ` the received amplitude at the carrier and `g` the pulse — by default the raised cosine at `sps=1` (transmit pulse times matched filter: the channel at the decision instants) and the root-raised-cosine transmit half above it; `pulse='nearest'` bins to the nearest sample, exactly `comms.multipath_channel`. `comms.simulate_link(..., channel=taps)` takes the `sps=1` set; a multi-cell result needs `receiver=(depth, range)` |
| `.coherence_bandwidth(convention='inverse_spread', factor=None)`, `.channel_regime(symbol_rate, convention=…, factor=…)` | `1/(k·τ_rms)` with `k = 1` by default — the convention the corpus states (APL-UW TR 9407 §II.7.b: the inverse of the delay spread measures the coherence bandwidth; Abraham §8.7: `W < 1/σ_t`). Rappaport's 0.5- and 0.9-correlation rules are the named options `'rappaport_0.5'` (`k = 5`) and `'rappaport_0.9'` (`k = 50`), and `factor=k` sets any other. The regime is the verdict for one symbol rate: signal bandwidth, coherence bandwidth, ISI in symbols, the convention used and `frequency_selective` — true when the symbol band is wider than the coherence bandwidth |
| `len(arr)`, `for a in arr:` | count and iterate |

`Arrivals` is the channel impulse response that
[`uacpy.comms`](comms.md) simulates a modem link over.

### Modes and reflection coefficients

![Modes and reflection coefficient](figures/results_modes_reflection.png)

```python
modes = Kraken().run(env, source, receiver, run_mode=RunMode.MODES)
refl = Bounce().run(env_el, source_el, receiver_el, run_mode=RunMode.REFLECTION)

modes.plot(n_modes=6)
refl.plot(show_phase=True)
```

**The modal physics takes plain arrays.** `Modes.with_attenuation` and
`Modes.modal_propagation_loss` are wrappers over
`uacpy.core.acoustics.modal_attenuation` (JKPS Eq. 5.169, the first-order
perturbation) and `uacpy.core.acoustics.modal_field` (the asymptotic modal
sum), so a mode set from another solver, a file or an analytic waveguide
reaches the same code:

```python
from uacpy.core.acoustics import modal_attenuation, modal_field

alpha_m = modal_attenuation(k, psi, z, 0.01, frequency=100.0)   # Np/m per mode
p = modal_field(k, psi[z_s_index], psi_at_receivers, ranges_m)  # complex Pa
```

`modal_field` takes the shapes **already evaluated** at the source and
receiver depths: how you get there from a tabulation — interpolating,
refusing to extrapolate, masking what lies outside — is the caller's
question, and the method answers it one way.

`uacpy.core.acoustics.transmission_loss_dB` is the conversion behind every
TL view in the package — `-20·log10(max(|p|, floor))`, note the **minus**,
so a quiet cell is a large number. It takes the pressure, not its square,
which is what separates it from `power_to_dB`.

Similarly `uacpy.core.acoustics.hankel_transform` turns a wavenumber-domain
`G(k)` into a range-domain field (the direct trapezoidal DFT of
`fieldsco.m`, not an FFT), and `wavenumber_taper` builds the `c_min`/`c_max`
phase-speed window that decides which physics a spectral run keeps. Both
were private inside the `.grn` reader; they work on any `G(k)`.
`alias_period(Δk)` gives the range at which such a transform wraps
(`2π/Δk`), and `ranges_fit_alias_period(Δk, r_max)` answers whether the
ranges you want sit inside it — necessary, not sufficient, and the one
question nothing downstream of `G(k)` can answer for you, because folded
energy is indistinguishable from real energy once it has landed and makes
the field too **loud**.

`Modes` carries `k` (complex horizontal wavenumbers, shape `(n_modes,)`),
`phi` (mode shapes, `(n_depths, n_modes)`) and `depths`. `n_modes` is derived
from `len(k)`, so it can never desync. Beyond plotting it does real work:
`first_n(n)` trims `k` and `phi` together, `compute_phase_speeds()` gives
`ω/Re(k)`, `compute_group_velocity(other)` differences two nearby frequencies,
`with_attenuation(...)` applies a first-order perturbation — pass `bottom=`
so the mode normalisation can run into the half-space, since without it the
evanescent tail is missing from the denominator and the returned attenuation
is an upper bound (it warns) — and
`modal_propagation_loss(...)` sums the modes into a complex `Field` — a
`Modes` result that becomes a `Field`. See [Kraken](../models/kraken.md).

`ReflectionCoefficient` holds `theta` (grazing angles, degrees), `R`
(magnitude) and `phi` (phase, radians), either 1-D or frequency-resolved
(`is_broadband`). Its `.at` / `.isel` / `.eval` deliberately differ from
`Field`'s: selecting one **frequency** collapses to a narrowband `R(theta)`,
but selecting one **angle** keeps `theta` as a length-1 axis, because `theta`
is this type's permanent abscissa. See [Bounce](../models/bounce.md).

### Covariance and Replicas

[OASN](../models/oases.md) produces the two array-processing types.
`Covariance` holds `C(f, i, j)` across the hydrophones; `Replicas` holds
Green's-function samples at every element for every candidate source position
`(z, x, y)`. They meet in the ambiguity surface:

```python
surface = cov.bartlett(replicas)             # (n_freq, n_zr, n_xr, n_yr)
sharper = cov.mvdr(replicas, diagonal_loading=1e-6)
```

Both return a plain ndarray — argmax over the last three axes is the
localisation estimate. [`sonar.md`](sonar.md) covers matched-field processing
properly, including building a replica bank out of ordinary `Field` results.

---

## 8. Identity and provenance

Every result, gridded or sparse, carries the same identity surface:

| Attribute | Type | What it records |
|---|---|---|
| `model` | `str` | wrapper class that produced it — `'Bellhop'`, `'Kraken'`, `'RAM'` |
| `backend` | `str` | the concrete binary that ran — `'cuda'`, `'krakenc'`, `'mpiramS'` |
| `model_source` | `ModelSource` or `None` | engine provenance: authors, licence, citation, URL. Drawn as the credit line on plots. |
| `source_depths` | ndarray | source depths of the run, metres |
| `frequencies` | ndarray or `None` | plural-only: 1-D, length ≥ 1. `f0` gives the scalar; `n_frequencies` the count. |
| `phase_reference` | `str` or `None` | phase convention of a complex payload — below |
| `metadata` | `dict` | model-specific extras |

Plus `copy()` (deep), `id_kwargs()` (the identity as a kwargs dict, for
spawning a derived result with the provenance intact) and `list_metadata()`.

**`backend` is worth reading after a run.** `Bellhop(backend='cuda')` without a
usable GPU falls back to Fortran with a warning; `result.backend` is what
actually executed.

### `phase_reference` — the contract for complex payloads

A complex `H(f)` is meaningless to a consumer that does not know which phase
convention it is in, and every model's native convention differs. uacpy
normalises at the wrapper boundary and records the answer:

| Value | Meaning | Who tags it |
|---|---|---|
| `'travelling_wave'` | `H(f)` carries the engineering propagator `exp(-i k₀ r)`, so `2·Re[ifft(H)]` lands the causal arrival at `t = r/c₀` | every frequency-domain producer: Bellhop, Kraken, Scooter, RAM, OASES |
| `'time_domain_native'` | the payload is already `p(t)` | [SPARC](../models/sparc.md), and everything `to_time_trace` / `synthesize_time_series` returns |

This is what lets one IFFT path serve every model. It also lets that path
refuse work it cannot do: handing a `'time_domain_native'` field to the
synthesis raises rather than inverting a spectrum that was never a travelling
wave. `PhaseReference` subclasses `str`, so `result.phase_reference ==
'travelling_wave'` just works.

### `metadata` and `list_metadata()`

`metadata` is a free-form bag — output-file paths under a pinned `work_dir`,
solver settings the wrapper resolved for you, intermediate results.
`list_metadata()` describes what is actually in it, so you do not have to grep
the source:

```python
>>> p = SPARC(t_max=0.8).run(env, source, point)
>>> p.list_metadata()['dt']
{'value_type': 'float',
 'documented_type': 'float',
 'description': 'Time-sample step (s) for TIME_SERIES output.'}
```

Undocumented keys still appear, with `documented_type=None` and
`description=None`, so nothing a wrapper attached is hidden from you. Some
entries are whole results: a Bellhop broadband run keeps the `Arrivals` it was
synthesised from under `metadata['arrivals_field']`, so you can re-synthesise
with a different waveform without re-running the model.

### Why a result carries no `Environment`

A `Field` knows it came from Bellhop at 200 Hz with the source at 25 m. It does
**not** hold the `Environment`, `Source` or `Receiver` it ran against, and that
is a design decision rather than an omission.

The inputs live in the carriers, and carriers are mutable. If a result kept a
reference to the environment, then editing that environment after the run —
deepening the bathymetry, swapping the seabed — would leave a result whose
attached environment no longer describes the water it was computed in. Every
plot drawn from it afterwards would assert something false, and nothing in the
code could detect the disagreement. Copying the environment into the result
instead just moves the problem: now you have two environments that look
authoritative and quietly differ.

So results carry provenance, not inputs. The cost is one keyword at the plot
call:

```python
tl.plot(env=env, source=source, receiver=receiver)
```

`env=` draws the seabed and spans the full water column; `source=` and
`receiver=` overlay the geometry. Without them the plot spans exactly the
receiver grid, which is **correct** — it is drawing the only depth axis the
result has. If your TL image stops at 99 m over a 100 m seabed, you did not hit
a bug; you did not pass `env=`. [`plotting.md`](plotting.md) has the full
overlay story.

---

## 9. Gotchas

**`.at()` never interpolates.** It returns the nearest stored sample and tells
you which one in `pinned`. Ask for 3000 m on a grid that samples 2992 m and you
get 2992 m. Use `.eval()` when you genuinely want an interpolated value — and
prefer to interpolate complex pressure, not dB: interpolating in dB smooths
sharp interference nulls into something that never existed.

**A fully-collapsed `Field` is a number, not a plot.** `tl.max()` has empty
`coords` and 0-D `data`; read `.data` and `.pinned`, do not try to plot it.

**An incoherent field has no phase, whatever its dtype says.** Kraken's
`INCOHERENT_TL` and OAST's `COHERENT_TL` return `unit='dB'` — real dB, and no
path to time-series synthesis. Bellhop writes its incoherent sum into the same
complex `.shd` container, so it stays complex with an **identically zero**
imaginary part. The phase is meaningless in both cases; only the dtype
differs.

**No-data cells are `NaN`, not zero.** Where no ray reached, TL is `NaN`, and
so is a cell the solver could not solve — a diverged PE march, a mode whose
attenuation has no answer, a trace synthesised from a spectrum with unsolved
bins. uacpy never writes a level over a sample the model did not produce, so
`NaN` means *no data* and is never a quiet result you could mistake for one; a
cell that genuinely carries **no energy** is a different thing and reports the
600 dB `PRESSURE_FLOOR`, past anything a real field reaches. Each case is
announced by a `UserWarning`. `.max()` skips NaNs; your own reductions should
use `np.nanmedian` and friends.

**Slicing narrows identity.** After `H.at(frequency=300)` the result's
`frequencies` is `[299.2]`, not the original 192-element grid. That is the point
— the slice reports what it is — but do not expect the full sweep back from a
slice. Keep the parent if you need it.

**A synthesised record is `1/Δf` long.** More `nfft` does not buy more time; a
finer frequency grid does, at the cost of more model runs.

**`.p` — and `.dB` on an already-real field — hand back read-only views.**
They look at the result's own buffer, so numpy refuses in-place edits rather
than let you corrupt it. Copy first. (`.dB` on complex data computes a fresh
array, which is writable; do not rely on the difference.)

---

## 10. Where this connects

- **Rendering** — [`plotting.md`](plotting.md). You shape `coords` here; that
  page turns them into a picture.
- **Inputs** — [environment](environment.md),
  [source and receiver](source-receiver.md), and
  [external data](data.md) if you want an environment built from GPS.
- **Downstream analysis** — [signal processing](signal.md),
  [array processing](arrays.md), [communications](comms.md),
  [noise](noise.md), [sonar and matched-field](sonar.md).
- **Persistence** — [file I/O](io.md) for the native formats behind
  `metadata`'s file paths; [utilities](utilities.md) for TL metrics and
  parallel sweeps.
- **Per-model detail** — [Bellhop](../models/bellhop.md) ·
  [Kraken](../models/kraken.md) · [Scooter](../models/scooter.md) ·
  [SPARC](../models/sparc.md) · [RAM](../models/ram.md) ·
  [Bounce](../models/bounce.md) · [OASES](../models/oases.md) ·
  [model index](../models/README.md)

Every figure on this page is generated by
[`docs/figure_scripts/results.py`](../figure_scripts/results.py); the snippets
above are condensed from it, so the script is the authoritative figure code.

---

**See also:** [guide index](../README.md) · [plotting](plotting.md) ·
[reference](../../DOCUMENTATION.md)
