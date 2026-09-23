# Plotting — one convention, one workhorse

> `uacpy.plot` · 63 public plotters · every result and every drawable carrier
> renders itself with `.plot()`

There are two halves to the plotting surface. Anything that is a uacpy object
draws itself — `result.plot()`, `env.plot()`, `env.ssp.plot()` — so you never
have to remember which function goes with which type. Anything that is *not* an
object, because it is a pair of NumPy arrays coming out of a DSP routine, gets a
free plotter: `plot_spectrogram(f, t, Sxx)`.

Every plotter returns `(fig, ax)` — `(fig, axes)` for the multi-panel ones —
takes `title=`, and, if it draws on a single axes, takes `ax=`, so any of them
can be dropped into a panel of a figure you built yourself. (The two animation
helpers are the exceptions: `animate_field` returns a `FuncAnimation` and
`save_animation` returns the path it wrote.)

---

## 1. The convention: everything plots itself

```python
import uacpy
```

`uacpy.plot` is the whole surface. It is an attribute alias for
`uacpy.visualization.plots`, so use `uacpy.plot.plot_field(...)` after
`import uacpy`, or import the name directly from
`uacpy.visualization` — `from uacpy.plot import plot_field` raises
`ModuleNotFoundError`, because `uacpy.plot` is not a module path. Four names are
also re-exported at the top level for convenience: `uacpy.plot_result`,
`uacpy.plot_field`, `uacpy.plot_overview`, `uacpy.compare_models`.

![Carriers and results both plot themselves](figures/plot_dispatch.png)

```python
env, source, receiver = shallow_water()
tl = Bellhop(n_beams=3000).run(env, source, receiver).to_dB()
rays = Bellhop(n_beams=25, alpha=(-12.0, 12.0)).run(
    env, source, receiver, run_mode=RunMode.RAYS)

fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
env.plot(ax=axes[0][0], title='env.plot()  —  Environment')
env.ssp.plot(ax=axes[0][1], title='env.ssp.plot()  —  SoundSpeedProfile')
tl.plot(env=env, ax=axes[1][0], title='tl.plot(env=env)  —  Field')
rays.plot(env=env, ax=axes[1][1], show_receivers=False, show_legend=False,
          title='rays.plot(env=env)  —  Rays')
```

### The contract

`result = compute(...)` then `fig, ax = result.plot()` holds for **every**
measurement uacpy returns — the propagation results, the medium carriers, the
spectral estimators and every transform:

```
sig.welch(x, fs).plot()                       # SpectralEstimate
sig.spectrogram(x, fs).plot()                 # SpectrogramResult
sig.cwt(x, fs).plot(sample_rate=fs)           # CWTResult
sig.wigner_ville(x, fs).plot()                # WignerVilleResult
sig.complex_cepstrum(x).plot()                # ComplexCepstrum
sig.constant_q_transform(x, fs).plot()        # CQTResult
sig.ambiguity_function(x, fs).plot()          # AmbiguityResult
sig.fk_transform(panel, fs, dx).plot()        # FKResult
sig.taup_transform(panel, fs, dx).plot()      # TauPResult
sig.radon_transform(panel, fs, dx, p).plot()  # RadonResult
uacpy.absorption_thorp(f).plot()              # AbsorptionCoefficient
WenzNoise(f, wind_speed_kn=15).plot()         # WenzNoise
arrivals.channel_taps(...).plot()             # ChannelTaps
env.ssp.plot()                                # SoundSpeedProfile
```

Every one returns `(fig, ax)`, and every one takes `ax=` so it composes.

`.plot()` is the one obvious way, not the only way. Each delegates to the free
plotter named in its docstring, and **that plotter takes either spelling** —
the carrier, or the bare arrays:

```
plot_spectrogram(result)                                  # the carrier
plot_spectrogram(result.frequencies, result.times, result.power)
```

Both draw the same thing, which `test_architecture.py` checks by comparing
what actually lands on the axes. The array form is what you want when the
numbers did not come from uacpy. The carrier form is recognised only when the
other positional arguments are left out, so passing arrays always means
arrays, and an incomplete call is refused by name rather than failing inside
matplotlib:

```
plot_spectrogram(f, t)
ConfigurationError: plot_spectrogram: pass every array, or one
SpectrogramResult.
```

Carriers come in two shapes, chosen rather than defaulted. A result that *is*
a few arrays you would unpack stays a `namedtuple`, so `f, t, S =
spectrogram(x, fs)` keeps working; metadata a plot needs but an unpack should
not see rides as an attribute beside the tuple (`SpectralEstimate.scaling`,
`FKResult.scaling`), which is why the dB axis can label itself `Pa²/Hz` or
`Pa²` without being told. A result you would *ask something* is a full class —
`Field.at()`, `Rays.filter_by_bounces()`, `Modes.excitation()`.

`CWTResult.plot` is the single exception that asks for an argument. A
scalogram's time axis is drawn from the sample rate and the carrier does not
hold one, so it cannot be defaulted without inventing the axis.

A carrier that carries everything its plotter needs derives the rest rather
than asking: `ChannelTaps.plot()` works out its sample rate as
`symbol_rate * sps`.

`test_architecture.py` enforces this — a new result carrier cannot ship
without a `.plot()` unless it is recorded there as having no plotter at all.

### Results

`Result.plot(**kwargs)` forwards to **`plot_result`**, which dispatches on type
and forwards the rest of your keywords to the plotter it picked:

| Result | `.plot()` draws | Underlying plotter |
|---|---|---|
| `Field` | heatmap / line cut / stacked traces | `plot_field` (public) |
| `ResultStack[Field]` | one titled panel per slab | `_plot_field_stack` |
| `Rays` | the ray fan, coloured by boundary interaction | `_plot_rays` |
| `Arrivals` | amplitude-vs-delay stems; `dB=True` draws the level axis instead, bounded `dynamic_range` dB (default 60) under the loudest arrival | `_plot_arrivals` |
| `Modes` | the depth eigenfunctions ψ(z) | `_plot_mode_functions` |
| `ReflectionCoefficient` | \|R(θ)\| (and phase with `show_phase=True`); broadband draws \|R(θ,f)\| as a heatmap, with `angle_on_x=True` to share an angle axis with a narrowband panel, `frequency_unit='Hz'`, `vmin`/`vmax`, `cmap` and `show_colorbar` | `_plot_reflection_coefficient` |
| `Covariance` | the CSDM as an image | `_plot_covariance` |
| `Replicas` | the replica field | `_plot_replicas` |

Only `plot_field` is public in that list. The other seven are single-view
renderers with exactly one caller each, so they live behind the `.plot()` they
implement; two *alternate* views of `Modes` that `.plot()` does not give you —
`plot_mode_wavenumbers` and `plot_modes_heatmap` — are public, because there is
no other way to reach them.

### Carriers

Carriers are not results, but the drawable ones follow the same convention:

| Carrier | `.plot()` draws |
|---|---|
| `Environment` | water column + seafloor cross-section, with optional `source=` / `receiver=` |
| `SoundSpeedProfile` | c(z), one line per range column |
| `Bathymetry` | seafloor depth vs range (axis pointing down) |
| `Altimetry` | sea-surface height vs range (axis pointing up) |
| `Absorption` | α(f) in dB/km, log-log — **takes `frequencies` as a required argument**, because absorption *is* a function of frequency |

`Bottom.plot()` and `Surface.plot()` deliberately **do not exist**. A seabed
cross-section needs to know where the seafloor is, and that lives in
`env.bathymetry`, not in the `Bottom` — a `Bottom` on its own has nothing to
draw itself against. Use [`plot_bottom_properties(env)`](environment.md)
instead, which gives you a small-multiples panel per geoacoustic property
(cp, cs, ρ, αp, αs) — strictly more than `env.plot()`'s cp-only view.

`Source` and `Receiver` have no `.plot()` for the same reason: they are geometry
to be drawn *over* something. Pass them as `source=` / `receiver=` to a plotter
that has a cross-section (§3).

A source's *directivity*, though, does stand on its own — it is a table, not a
position — so it gets a named method rather than the bare one:
`source.plot_beam_pattern()`. The name says which of the two views you asked
for, which `.plot()` could not.

---

## 2. `plot_field` in depth

`plot_field` is the workhorse — every TL image, every range cut, every waterfall
in this documentation goes through it.

```python
plot_field(field, ax=None, *, env=None, source=None, receiver=None,
           value=None, vmin=None, vmax=None, cmap=None, title=None,
           label=None, figsize=(10, 5), stacked=False, stack_offset=None,
           show_colorbar=None, contours=None, **mpl_kw)
```

### 2.1 Three render branches, chosen from `coords`

`plot_field` does not ask what you want drawn; it reads `field.coords` and
picks:

| Surviving axes | Branch | Looks like |
|---|---|---|
| 2 | **heatmap** | `pcolormesh`; `(depth, range)` puts range on x and depth down the y axis |
| 1 | **line cut** | a `depth` axis goes on y pointing down; anything else on x, and a TL y axis is inverted so louder is up |
| 2, one of them `time`, `stacked=True` | **stacked traces** | the seismic waterfall: one offset trace per row |

So the way to control the picture is to slice the field first —
`.at()` / `.isel()` / `.max()` drop an axis into `.pinned`, which is what the
[results page](results.md) is about. One field, three branches:

![One Field, three render branches](figures/plot_branches.png)

```python
def _time_series():
    """``p(depth, range, time)`` — the field every render branch is drawn from."""
    env, _, _ = shallow_water()
    source = uacpy.Source(depths=25.0,
                          frequencies=np.arange(150.0, 450.1, 0.5))
    receiver = uacpy.Receiver(depths=CUT_DEPTH,
                              ranges=np.linspace(1000.0, 3000.0, 9))
    H = Bellhop(n_beams=3000).run(env, source, receiver,
                                  run_mode=RunMode.BROADBAND)
    _, waveform = lfm_chirp(150.0, 450.0, 0.04, 4000.0)
    return H.synthesize_time_series(waveform, 4000.0)

series = _time_series()
panel = series.isel(depth=0)     # coords {range, time}

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
panel.plot(ax=axes[0],
           title="2 axes  →  heatmap")
panel.at(range=CUT_RANGE).plot(ax=axes[1],
                               title="1 axis  →  line cut")
panel.plot(stacked=True, ax=axes[2],
           title="stacked=True  →  offset traces")
```

A field with **three or more** surviving axes has no picture, and says so:

```
ConfigurationError: plot_field: cannot plot a 3-axis field (coords
['depth', 'range', 'frequency']); slice it first with .at(...) / .isel(...)
so 1 or 2 axes remain.
```

`stacked=True` on anything but a 2-D field carrying a `time` axis is rejected
the same way.

### 2.2 Keywords the selected branch cannot use are **rejected**

This is the part that surprises people, and it is deliberate.

```python
>>> tl.at(depth=60.0).plot(vmin=20)
ConfigurationError: plot_field: vmin= has no effect on a 1-D line cut
(coords ['range']). vmin=, vmax=, cmap=, show_colorbar=, contours= apply to
the 2-D heatmap, env=, source=, receiver= to a (depth, range) cross-section,
label= to the 1-D line cut, stack_offset= to stacked=True.
```

A colour limit on a line plot is not a harmless no-op — it means you believe you
are looking at a heatmap and you are not. Silently dropping it would let that
misunderstanding survive all the way into a figure you publish. So every knob is
owned by a branch, and passing one to a branch that cannot read it raises
`ConfigurationError` before any figure is created:

| Keyword | Valid on |
|---|---|
| `vmin`, `vmax`, `cmap`, `show_colorbar`, `contours` | the 2-D heatmap |
| `env`, `source`, `receiver` | a `(depth, range)` heatmap only — not *any* heatmap |
| `label` | the 1-D line cut (it is the legend entry) |
| `stack_offset` | `stacked=True` |
| `value`, `title`, `figsize`, `ax` | every branch |

Note the second row. `env=` keys on the **axes**, not just on the branch: a
2-D heatmap of `(range, time)` is still a heatmap, but there is no depth axis to
hang a seafloor on, so it rejects too, with a message naming the distinction:

```
ConfigurationError: plot_field: env= has no effect on a heatmap that is not a
(depth, range) cross-section (coords ['range', 'time']). …
```

### 2.3 `value=` — which number gets drawn

`value=` applies to every branch and picks what the field's complex pressure is
reduced to. It defaults to `'real'` for a time-series field and `'dB'`
otherwise.

| `value` | Draws | Auto colour treatment (heatmap) |
|---|---|---|
| `'dB'` | the dB view (TL for a pressure field) | depends on the quantity: fixed 20–120 dB (§4) for pressure, symmetric about 0 dB for signal excess, autoscaled for reverberation |
| `'mag_dB'` | 20·log10\|H\| = −TL, dB | the TL colormap REVERSED (`jet`), autoscaled |
| `'mag'` | \|p\|, linear (complex fields only) | linear colormap (`seismic`), anchored at zero |
| `'phase'` | arg(p), radians (complex fields only) | `twilight`, fixed ±π |
| `'real'`, `'imag'` | Re(p) / Im(p) (`'imag'` complex only) | linear colormap (`seismic`), symmetric about zero; real time-domain data is clipped to ±RMS |

![The heatmap-only knobs](figures/plot_heatmap_knobs.png)

```python
env, source, receiver = deep_water()
model = Bellhop(n_beams=3000)
p = model.run(env, source, receiver)
incoherent = model.run(env, source, receiver,
                       run_mode=RunMode.INCOHERENT_TL)

p.plot(env=env, ax=axes[0][0],
       title='default — fixed 20-120 dB TL scale')
p.plot(env=env, ax=axes[0][1], vmin=70.0, vmax=110.0, cmap='viridis',
       title="vmin=70, vmax=110, cmap='viridis'")
p.plot(env=env, ax=axes[1][0], value='phase',
       title="value='phase' — twilight, ±π")
incoherent.plot(env=env, ax=axes[1][1], contours=(80.0, 90.0, 100.0),
                title='contours=(80, 90, 100) — on the incoherent run')
```

Two things worth reading off that figure. Narrowing `vmin`/`vmax` to the range
the data actually occupies is how you see convergence-zone structure that the
fixed scale compresses — it is the legitimate reason to override it. And
`contours=` is drawn on the **coherent** field only if you enjoy contouring
speckle: interference nulls put a 90 dB level everywhere. Contours read on a
smooth field, which in practice means an incoherent run, so that is the panel
they are shown on.

### The phase panel needs a grid the level panel does not

`value='phase'`, and equally `'real'` and `'imag'`, draw the **carrier**
rather than its envelope, and a wrapped phase resolves the carrier only while
it turns less than half a cycle between neighbouring samples. That is
Nyquist, so the bound is the half wavelength — on **both** spatial axes, and
at the highest frequency the field carries. Past it the panel does not go
noisy, it goes *smooth*: an aliased phase draws broad, confident-looking
bands that belong to the grid and not to the field, which is why uacpy warns
rather than trusting you to notice.

The bottom-left panel above is that trap, not an illustration of good
practice. At 50 Hz the half wavelength is 15 m, while that receiver grid is
42 m in depth and 331 m in range, so the carrier turns eleven times between
neighbouring samples:

```
plot_field(value='phase'): depth samples are 42.0168 m apart and range samples
are 331.104 m apart, at or over the 15 m half wavelength at 50 Hz (nominal
c=1500 m/s), so the carrier turns up to 11.04 cycles between neighbouring
samples and this view is aliased. ...
```

The panel is kept because it is the honest shape of the mistake — a deep-water
field over 100 km simply has no readable phase map, since resolving it would
take some 6600 range samples and the fringes would still fall below a pixel.
Read phase on a window you can afford to sample, and read `dB` everywhere
else: `|p|` varies on the interference scale rather than on the carrier, so
the level panels on this same grid are coarse but honest, and they are left
unguarded for that reason.

This is a different bound from the one [`resample_to` and
`eval`](results.md) enforce. Those interpolate *between* stored samples, and
keeping the interpolant off the opposite-phase lobe takes a **quarter**
wavelength; this one asks only whether the stored samples resolve the carrier
at all. Between the two a phase map is coarse but unambiguous.

---

## 3. Overlays: `env=`, `source=`, `receiver=`

**A `Result` carries no `Environment`, `Source` or `Receiver`.** It knows its
model, its backend, its provenance and its frequencies, and nothing about the
water it ran through. That is a deliberate design choice — a result is a value,
not a scene graph — and it has one visible consequence: a TL plot with no `env=`
spans exactly the receiver grid and draws no seabed. That is correct, not a bug.

![What env= and the geometry add](figures/plot_overlays.png)

```python
env, source, _ = shallow_water()
receiver = uacpy.Receiver(depths=np.linspace(1.0, 60.0, 80),
                          ranges=np.linspace(50.0, 5000.0, 250))
tl = Bellhop(n_beams=3000).run(env, source, receiver).to_dB()

tl.plot(ax=axes[0], show_colorbar=False,
        title='tl.plot()  —  depth axis spans the receiver grid')
tl.plot(env=env, ax=axes[1], show_colorbar=False,
        title='tl.plot(env=env)  —  seafloor drawn, axis extended')
tl.plot(env=env, source=source, receiver=receiver, ax=axes[2],
        show_colorbar=False,
        title='+ source=, receiver=  —  the run geometry')
```

The receiver grid stops at 60 m in a 100 m channel, which makes the difference
plain:

- **top** — the depth axis runs 1–60 m. Everything drawn is data.
- **`env=`** — the axis is extended past the seafloor (to 105 m, 5 % headroom
  below it), the seabed line is drawn and the sediment below it filled. A
  range-dependent bathymetry is clipped to the plotted range span, so a long
  environment does not stretch a short plot.
- **`source=` / `receiver=`** — the run geometry. The receiver lattice is
  decimated (≤ 20 range dots × 10 depth dots) so a dense grid does not paint the
  panel solid. The source is drawn at **r = 0** by the package convention that
  range is measured from it, and the x axis widens by the marker's own half
  width to keep it whole on screen — which is why the last panel starts just
  left of 0 km while the data starts at 50 m.

All three apply to a `(depth, range)` cross-section only (§2.2). `env=` is
accepted by every view that has one — `plot_field`, `plot_signal_excess`,
`plot_detection_probability`, `compare_models`, `animate_field`,
`plot_time_snapshots` and the ray plotter behind `rays.plot()`. `source=` /
`receiver=` are `plot_field`'s, `env.plot()`'s, `compare_models`'s and
`plot_overview`'s — `compare_models` draws the same markers on **every**
panel, so two models of one scene carry one geometry. The ray plotter draws
both by default instead, and turns them off with `show_source=False` /
`show_receivers=False`. Ask for an overlay anywhere else —
`arrivals.plot(env=env)`, a `(range, time)` heatmap — and you get a
`ConfigurationError` saying so rather than a silently ignored keyword.

### Drawing the source somewhere other than the origin

The source marker sits at `r = 0`, because range in uacpy is measured *from*
the source. For a scene built around a fixed receive array that is the wrong
end to anchor: the array is the object that stays put, and the source is the
thing out at range. `env.plot()` therefore accepts
`source_marker_range_m=5000.0`, which moves the **star** and nothing else.

It is a drawing coordinate, not a property of the `Source` and not an input
to any model — no TL changes, no deck is rewritten. (It is named for the
marker for that reason: a parameter called "source range" would read as a
contradiction on a range axis that starts at the source.) The default `0.0`
is bit-identical to omitting it.

It matters most on a range-dependent bottom, where drawing the source at the
origin does not merely mislabel the picture but shows the mirror-image
geometry, with the array standing on the wrong part of the section — see
`example_42_detection_chain`, which measures what that costs.

---

## 4. The fixed TL colour scale

TL heatmaps default to **`vmin=20`, `vmax=120` dB** — a fixed scale, not one
derived from the data. `_TL_LIMITS` in
[`uacpy/visualization/plots/_common.py`](../../uacpy/visualization/plots/_common.py)
is the single definition; `plot_field(value='dB')` and `compare_models` both
read it.

![Why the TL scale is fixed](figures/plot_tl_scale.png)

```python
env_s, src_s, rcv_s = shallow_water()
env_d, src_d, rcv_d = deep_water()
shallow = Bellhop(n_beams=3000).run(env_s, src_s, rcv_s).to_dB()
deep = Bellhop(n_beams=3000).run(env_d, src_d, rcv_d).to_dB()

for col, (field, env, name) in enumerate(
        [(shallow, env_s, 'Shallow, 200 Hz, 5 km'),
         (deep, env_d, 'Deep (Munk), 50 Hz, 100 km')]):
    field.plot(env=env, ax=axes[0][col],
               title=f'{name} — default scale')
    lo, hi = np.nanpercentile(field.data, [2.0, 98.0])
    field.plot(env=env, ax=axes[1][col], vmin=lo, vmax=hi,
               title=f'{name} — vmin/vmax per panel')
```

The top row says something true: the deep-water field at 100 km is roughly
28 dB quieter than the shallow-water field at 5 km, and you can read that
straight off the colour. The bottom row autoscales each panel to its own 2nd–98th
percentile. Both panels now look equally colourful, the level difference has
vanished, and the two colourbars are the only warning — which nobody reads.
A colour scale that changes per panel turns a comparison into a decoration.

The corollaries:

- **The default is comparable across models, frequencies and runs.** Two TL
  images produced a week apart can be put side by side.
- **`compare_models` shares one colourbar** across its grid, and warns if the
  fields' depth or range axes do not match, because a shared scale over
  different sample grids is misleading in a subtler way.
- **Override for structure, not for prettiness.** Narrow the window when the
  variation you are after occupies a fraction of the 100 dB range, as in §2.3 —
  and then say so on the figure.
- **No-data cells are `NaN`, not 120 dB.** Bellhop cells no ray reached render
  as the axes background, so they read as absent rather than as very quiet.

---

## 5. Composition with `ax=`

Every single-axes plotter takes `ax=`. Hand it one and it draws into your axes
and returns `(fig, ax)` for that figure; hand it nothing and it makes its own of
`figsize=`.

![Composing carriers, results and comparisons](figures/plot_composition.png)

```python
env, source, receiver = shallow_water()
bellhop = Bellhop(n_beams=3000).run(env, source, receiver).to_dB()
kraken = Kraken().run(env, source, receiver).to_dB()

fig = plt.figure(figsize=(11.0, 6.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 2.2],
                      hspace=0.38, wspace=0.24)

env.ssp.plot(ax=fig.add_subplot(gs[:, 0]), title='env.ssp.plot()')
ax_tl = fig.add_subplot(gs[0, 1])
bellhop.plot(env=env, source=source, ax=ax_tl, title='Bellhop TL')
ax_tl.axhline(CUT_DEPTH, color='white', lw=1.2, ls='--')

ax_cut = fig.add_subplot(gs[1, 1])
uacpy.plot.compare(
    [bellhop.at(depth=CUT_DEPTH), kraken.at(depth=CUT_DEPTH)],
    labels=['Bellhop', 'Kraken'], ax=ax_cut,
    title=f'compare() — TL at {CUT_DEPTH:g} m')

# The cut is the map's own row, so the two range axes have to line up — and
# `bellhop.plot` took its colourbar out of the map's width alone.
fig.canvas.draw()
b_map, b_cut = ax_tl.get_position(), ax_cut.get_position()
ax_cut.set_position([b_map.x0, b_cut.y0, b_map.width, b_cut.height])
```

Three rules make this work:

**Depth-axis inversion is idempotent.** `Axes.invert_yaxis` *toggles*, so a
plotter that called it unconditionally would flip the axis back to
increasing-upward the second time you drew into the same axes. Every uacpy
plotter goes through one helper that inverts only if the axis is not already
inverted, so overlaying two fields — or a field and an SSP — on one `ax` is
safe in any order and any number of times.

**`figsize=` is ignored when you pass `ax=`.** The figure already exists. Size it
yourself in `plt.subplots(figsize=…)` or `plt.figure(figsize=…)`.

**Figure furniture is only drawn by whoever owns the figure.** The grey
provenance footnote — `Model: Bellhop — Michael B. Porter, Acoustics Toolbox`,
plus a `Data:` block listing the sources of a fetched `env` — is added only when
the plotter created the figure. In a composed figure it is your call where the
credit goes, so nothing is stamped on top of your layout. (Colorbars are *not*
figure furniture: they still get drawn per axes. Pass `show_colorbar=False` on
the panels that should share one.)

A handful of plotters are **figure-level** and take no `ax=`, because one axes
is not enough to hold what they draw: `compare_models`,
`plot_bottom_properties`, `plot_overview`, `plot_lsfir_diagnostics`,
`plot_time_snapshots` (multi-panel), `plot_result` (it forwards to whichever
plotter fits) and `save_animation` (it writes a file). Two plotters take a
**2-tuple** of axes instead: `plot_frf` and `plot_channel`, which are inherently
two stacked panels. `Field.plot_transfer_function` does the same via
`axes=(ax_mag, ax_phase)`.

### `Field`'s two rendering shortcuts

Beyond `.plot()`, a broadband `Field` carries two methods that render a
reduce-then-plot view of one receiver cell:

| Method | Draws |
|---|---|
| `H.plot_transfer_function(axes=None, …)` | 20·log10\|H\| over arg(H), two stacked panels sharing the frequency axis |
| `H.plot_impulse_response(ax=None, window='hann', …)` | the band-limited `p(t)` that `H` inverts to |

Both squeeze singleton axes and require the field to reduce to a single
`(depth, range)` cell — `H.at(depth=…, range=…).plot_transfer_function()`.

---

## 6. Free plotters: arrays in, no object

The DSP, comms and noise routines return plain arrays or small result tuples,
not uacpy objects — there is nothing to hang a `.plot()` on. Those get a free
plotter that consumes exactly what the analysis function produced.

![Four free plotters](figures/plot_dsp.png)

```python
rng = np.random.default_rng(0)
sample_rate = 4000.0
trace = _time_series().isel(depth=0).at(range=1000.0)
# Model pressure is referenced to a unit source at 1 m; scale it to a
# 170 dB re 1 µPa @ 1 m projector so the dB axes read physically.
p_t = 316.0 * np.asarray(trace.data, dtype=float)

f_s, t_s, S = spectrogram(p_t, sample_rate, nperseg=256)
f_p, P = welch(p_t, sample_rate, nperseg=1024)

mod = Modulator('16qam')
bits = rng.integers(0, 2, size=4 * 600)
symbols = awgn(mod.modulate(bits), 18.0, rng=rng)

uacpy.plot.plot_spectrogram(f_s, t_s, S, ax=axes[0][0], vmin=30, vmax=85,
                            ymax=800.0,
                            title='plot_spectrogram — chirp at 1 km')
uacpy.plot.plot_psd(f_p, P, ax=axes[0][1], ymin=0, ymax=80,
                    title='plot_psd — same trace')
uacpy.plot.plot_constellation(constellation('16qam'), ax=axes[1][0],
                              scheme='16qam',
                              title='plot_constellation — 16-QAM map')
uacpy.plot.plot_scatter(symbols, ax=axes[1][1],
                        ideal=constellation('16qam'),
                        title='plot_scatter — received, 18 dB SNR')
```

The pairing is one-to-one and mechanical: `spectrogram` → `plot_spectrogram`,
`welch` → `plot_psd`, `cwt` → `plot_cwt`, `fk_transform` → `plot_fk`,
`ambiguity_function` → `plot_ambiguity`. Unpack the result and pass it through.

The spectral estimators also carry the pairing themselves: a
`SpectralEstimate` or a `ProbabilisticSpectralEstimate` has `.plot()`, which
picks the plotter from the `method` the estimate was computed with and labels
the axis from its `scaling` and `ref` — see
[signal processing](signal.md#3-spectra-levels-and-bands). Like every plotter
here it returns `(fig, ax)`.

Two conventions to know before reading the levels:

- **dB references are the plotter's, not the data's.** The acoustic plotters
  take `ref=1e-6` (1 µPa) and their default axis limits assume ocean-ambient
  levels. A model field is pressure for a *unit* source at 1 m, not the pressure
  at a real projector — multiply by the source amplitude in Pa (above, 316 Pa
  ≈ 170 dB re 1 µPa @ 1 m) before reading a dB axis, or the trace sits below
  `ymin` and the panel looks empty.
- **`draw_sound_cone` and `draw_slowness_line` are overlays, not plots.** They
  take an existing `ax` as their first positional argument and annotate an f-k
  or τ-p panel you already drew.

The science behind these lives on the pages that own it:
[signal processing](signal.md) for the time-frequency and transform plotters,
[array processing](arrays.md) for `plot_angular_spectrum`,
[communications](comms.md) for the constellation/eye/BER family,
[noise](noise.md) for Wenz and weighting, [sonar](sonar.md) for signal excess,
detection probability and the ROC.

---

## 7. Reference — every public plotter

All 63 plotters in `uacpy.plot.__all__`, plus the two coastline calls they
draw land with — the 8 remaining names in `__all__` are the submodules
themselves. **ax** marks a single-axes plotter you can
compose with. Every entry takes `title=` except `plot_result` (it forwards
yours), `shared_colorbar` (a colorbar on an existing figure) and the two
`draw_*` overlays; every entry takes `figsize=` except `plot_result`,
`shared_colorbar`, `animate_field`, the two `draw_*` overlays and
`plot_time_snapshots`, which sizes itself from `figsize_per_panel=` instead.

### Fields and results

| Plotter | ax | Draws |
|---|---|---|
| `plot_result(result, env=None, **kw)` | — | type-dispatcher behind every `Result.plot()` |
| `plot_field(field, ax=None, …)` | ✓ | the workhorse — §2 |
| `compare(fields, labels=None, ax=None, value=None)` | ✓ | overlay several 1-D sliced fields on one axes |
| `compare_models(fields, labels=None, env=None, ncols=None, contours=None)` | — | side-by-side heatmap grid, one shared colourbar |
| `plot_field_difference(field, reference, ax=None, env=None, diff_vmax=None)` | ✓ | `field - reference` in dB on a diverging map, symmetric about zero; positive means `field` is the quieter one |
| `plot_field_statistics(fields, labels=None, *, depth)` | — | mean ± std per field at one depth, plus the pairwise RMS-difference matrix |
| `shared_colorbar(fig, axes, *, label=None, **kw)` | — | one colorbar for a row or grid of panels drawn with `show_colorbar=False`, taken from their own mappable. Refuses panels that are not on one colour scale — a single bar over two scales describes one and mislabels the rest |
| `plot_signal_excess(field, ax=None, env=None, …)` | ✓ | diverging SE heatmap + the SE = 0 detection boundary → [sonar](sonar.md) |
| `plot_detection_probability(field, ax=None, env=None, …)` | ✓ | `P_D` on a fixed [0, 1] scale with labelled contours → [sonar](sonar.md) |
| `animate_field(field, env=None, fps=30, …)` | ✓ | a `FuncAnimation` sweeping the time axis |
| `save_animation(field, path, fps=20, …)` | — | render that animation to GIF/MP4 (writer from the suffix) |
| `plot_time_snapshots(fields, times_s, env=None, …)` | — | per-model rows × per-time columns of `p(d, r, t)` |

### Rays and modes

| Plotter | ax | Draws |
|---|---|---|
| `plot_mode_wavenumbers(modes, ax=None)` | ✓ | `Re(k_m)` vs mode index, with `Im(k_m)` when non-zero → [Kraken](../models/kraken.md) |
| `plot_modes_heatmap(modes, n_modes=None, ax=None, …)` | ✓ | ψ_m(z) as a (depth, mode index) image |
| `plot_mode_speeds(modes, ax=None, c_bottom=None, …)` | ✓ | phase speed per mode index, plus group speed when the result carries it |
| `plot_dispersion(modes_by_frequency, ax=None, n_modes=3, …)` | ✓ | phase and group speed vs frequency — the dispersion diagram |
| `plot_greens_function(grn, ax=None, frequency_index=0, depth=None, modes=None, vmin_dB=-60, …)` | ✓ | \|G(k_r, z)\| from a Scooter `.grn`; `modes=` marks the trapped eigenvalues on it |
| `plot_wavenumber_sampling(frequency, c_low, c_high, delta_k, ax=None, r_max=None, …)` | ✓ | the k_r axis a Hankel transform is sampled on, with the wrap-around limit `r_max` implies |

Ray fans, arrival stems, mode functions, covariance, replicas and reflection
coefficients have no public free plotter — they are reached through
`result.plot()` (§1). `plot_dispersion` is the exception that needs several
results at once: it takes a sequence of `Modes`, one per frequency — it sorts
them by each result's own `f0`, so the caller need not — because a dispersion
curve is not a property of any single run. Group speed is only
drawn where the backend filled it — `krakenc` does, `kraken` prints zeros
(see [Kraken](../models/kraken.md)).

### Source

| Plotter | ax | Draws |
|---|---|---|
| `plot_detection_probability(field, env=None, source=None, receiver=None, contour_levels=(0.1,0.5,0.9))` / `plot_signal_excess(...)` | ✓ | the two sonar-equation panels. Both take `source=`/`receiver=` like `plot_field`: a detection map answers "would this array hear that target", so the two things it is about belong on it |
| `plot_result_stack(stack, env=None, ncols=None)` | ✓ | one TL panel per slab. A grid stacked over `source_depth` marks **each panel's own source**, since that is what one panel shows; pass `source=` to override |
| `plot_mode_excitation(modes, source, ax=None, sound_speed=1500.0, show_array_factor=True, floor_dB=-40.0)` | ✓ | what a source array drives, both ways on one angle axis: a stem per mode at its grazing angle, height `|Σₙ wₙ·φₘ(zₙ)|` (the waveguide's answer — Medwin & Clay §11.3.1, *mode filters*), over the free-field pattern of the same array — its **array beam pattern** `P(θ)=f(θ)·A(θ)` when the elements are directional, the bare array factor `A(θ)` when not (Butler & Sherman §7.1.1). They agree while the pattern is symmetric in ±θ and part company once steering breaks it, which is why both are drawn |
| `plot_beam_pattern(pattern=None, ax=None, polar=True, mirror=False, fill=True, rmin=None)` | ✓ | the `.sbp` directivity table, on polar axes oriented like the field: 0° along increasing range, positive angles downward. `source.plot_beam_pattern()` is the object form; `None` draws the flat 0 dB circle Bellhop substitutes for an omni source |
| `plot_beam_power(beams, ax=None, at=None, normalise=True)` | ✓ | a scanned beam's power against **look** angle, from the `BeamformedField` that `beamform_field` returns; `beams.plot()` is the object form. The receive dual of `plot_beam_pattern` — that one is a *launch* fan and labels itself so. `at=` picks the point when the beamformer ran over a grid, or the bin when it ran over a band |

The polar orientation is not cosmetic. The `.sbp` angle axis *is* Bellhop's
launch declination `alpha` — `Bellhop._check_beam_pattern_spans_the_fan`
compares the two directly — and `ray2D(1)%t = [COS(alpha), SIN(alpha)]/c`
(`Bellhop/bellhop.f90:453`) over a depth axis that is positive downward sends
`alpha > 0` deeper. A lobe drawn below the horizontal is therefore a lobe that
ensonifies the water below the source in the TL image beside it.

**The sector is always ±90°, in both renderings.** A launch steeper than
±90° has `COS(alpha) < 0`, so it traces to *negative* range and never enters
the `r > 0` the field is evaluated on — traced directly, `alpha = ±127.5°`
gives `r` in `[-6000, 0]` while `alpha = ±42.5°` reaches `+6000`. That makes
`[-90, 90]` the whole of what a `.sbp` has to say, so there is nothing to
choose between: every pattern lands on the same axes, and two of them can be
compared by eye. `polar=False` takes the identical limit as its `xlim`.

It is an axes limit, not a filter — the line always carries every row of the
table — and a table whose strongest level falls outside the fan says so,
because Bellhop would not launch into it either.

![Polar and rectilinear share one sector](figures/plot_beam_pattern.png)

`mirror=` is **off** by default: no engine mirrors a half-defined table.
`ReadPat` (`misc/beampattern.f90`) reads it verbatim, and `bellhop.f90:269-274`
interpolates with the index clamped but the weight unclamped, so the angles a
0-180° table omits are *extrapolated*, not reflected. A table that does not
cover ±90° warns, and `mirror=True` reflects it through 0° when you know it is
symmetric. `example_04_bellhop_advanced.py` runs a directional source
against an omni one.

### Environment

| Plotter | ax | Draws |
|---|---|---|
| `plot_bottom_properties(env, properties=None, n_range=240, n_depth=200)` | — | small-multiples seabed cross-sections, one panel per property → [environment](environment.md) |
| `plot_bottom_loss(materials, ax=None, water_speed=1500.0, mark_critical=False)` | ✓ | plane-wave bottom loss vs grazing angle, one curve per seabed — a preset name, a property dict, a sequence, or a `{label: material}` mapping. Draws what `core.acoustics.bottom_loss_curve` computes; `mark_critical=True` rules each **faster-than-water** seabed's critical angle (a slower one has none) |
| `plot_absorption(coefficient, ax=None, label=None)` | ✓ | draws an `AbsorptionCoefficient`: α(f) log-log, or α(f, z) as a heatmap. `absorption_thorp(f).plot()` is the object form |

### Maps

| Plotter | ax | Draws |
|---|---|---|
| `plot_bathymetry_map(lats, lons, depth, transect=None, relief=True, …)` | ✓ | a fetched bathymetry grid as a geographic map → [external data](data.md) |
| `plot_overview(env, map_args, tl=None, source=None, receiver=None, …)` | — | one-call composite: map + TL + environment cross-section |
| `plot_sea_ice_map(grid, hemi='N', transect=None, …)` | ✓ | sea-ice concentration on a polar map |

Every map above draws land from the same source, and the two calls behind it
are public so a map of your own can use it:

| Call | Returns |
|---|---|
| `land_polygons(resolution='50m', *, url=None, …)` | Natural Earth land rings as `(N, 2)` `(lon, lat)` arrays — the backdrop the map plotters draw, or `None` when no source is reachable (the map then renders sea only) |
| `download_coastline(cache_dir=None, *, url=None, resolutions=…)` | Caches those rings for offline use and returns the written paths — the coastline's counterpart to `uacpy.data`'s `download_*_db` fetchers, and it takes the same `url=` mirror override |

Natural Earth is public domain, so neither call carries an attribution
requirement. `./install.sh --data coastline` runs `download_coastline` for you.

### Signal processing

Every one consumes the output of the same-named routine in
`uacpy.acoustic_signal` → [signal processing](signal.md).

| Plotter | ax | Consumes |
|---|---|---|
| `plot_psd(frequencies, psd_linear, ax=None, ref=1e-6, freq_scale='log', …)` | ✓ | a `SpectralEstimate` from `welch` or `constant_q` — the spectrum in dB, labelled from its own scaling (also `.plot()`) |
| `plot_ppsd(result, ax=None, …)` | ✓ | a `ProbabilisticSpectralEstimate` from `probabilistic_welch` or `probabilistic_sound_exposure` — 2-D histogram of levels (also `.plot()`) |
| `plot_sel(result, ax=None, band_type='decidecade', …)` | ✓ | a banded `SpectralEstimate`, i.e. `sound_exposure` output — band levels as bars, labelled from its scaling; also `.plot()` |
| `plot_band_levels(centers, levels, ax=None, …)` | ✓ | `decidecade_band_levels` — bar plot |
| `plot_spectrogram(frequencies, times, Sxx, ax=None, ymin=1, vmin=0, vmax=200, …)` | ✓ | `spectrogram` |
| `plot_constant_q_transform(frequencies, coefficients, ax=None, label=None, …)` | ✓ | `constant_q_transform` — one frame's \|X_cq\|, linear |
| `plot_constant_q_spectrogram(frequencies, times, power, ax=None, scaling='spectrum', …)` | ✓ | `constant_q_spectrogram` (log frequency) |
| `plot_constant_q_psd(frequencies, power, ax=None, scaling='spectrum', …)` | ✓ | `constant_q(...)` |
| `plot_constant_q_ppsd(result, ax=None, scaling='spectrum', …)` | ✓ | a `probabilistic_constant_q` estimate — the same histogram on geometric bins (also `.plot()`) |
| `plot_cwt(frequencies, W, sample_rate, ax=None, …)` | ✓ | `cwt` — scalogram \|W\| |
| `plot_wigner_ville(frequencies, times, W, ax=None, …)` | ✓ | `wigner_ville` |
| `plot_cepstrum(c, ax=None, sample_rate=None, …)` | ✓ | `cepstrum` vs quefrency |
| `plot_fk(frequencies, wavenumbers, power, ax=None, scaling=None, wavenumber_unit='rad/m', sound_speed=None, …)` | ✓ | `fk_transform` — f-k panel in dB, labelled from the result's scaling: `plot_fk(result)` draws a density (`normalize=True`) as PSD in Pa²·m/(Hz·rad), or Pa²·m/Hz with `wavenumber_unit='cycles/m'` (axis `ν = k/2π`, panel ×2π), and the raw `|FK|²` as unnormalised power; bare arrays need `scaling='density'` or `'power'`, and a `scaling=` that contradicts the result raises |
| `plot_taup(slownesses, taus, taup, ax=None, sound_speed=None, …)` | ✓ | `taup_transform` |
| `plot_radon(moveout, taus, R, ax=None, kind='linear', …)` | ✓ | `radon_transform` |
| `draw_sound_cone(ax, f_max, k_max, sound_speed, wavenumber_unit='rad/m', …)` | overlay | the `f = c·k/2π` cone on an f-k axis (`f = c·ν` over cycles/m) |
| `draw_slowness_line(ax, tau_max, sound_speed, …)` | overlay | `p = ±1/c` on a τ-p axis |
| `plot_ambiguity(delays_s, doppler_hz, chi, ax=None, dB=False, dynamic_range=40, …)` | ✓ | `ambiguity_function` — range-Doppler surface; `dB=True` shows it re its peak, where the sidelobes are |
| `plot_matched_field(x_m, z_m, surface, ax=None, dynamic_range=20, true_position=None, …)` | ✓ | a matched-field ambiguity surface over a replica grid (`Covariance.bartlett` / `.mvdr`). Draws one (z, x) plane: those return `(n_frequencies, n_zr, n_xr, n_yr)`, so index the frequency and y axes yourself when either is longer than 1 |
| `plot_angular_spectrum(angles_deg, spectrum, ax=None, dB=True, …)` | ✓ | a Bartlett / MVDR / MUSIC spectrum → [arrays](arrays.md) |
| `plot_frf(frequencies, tf, ax=None, tag='', …)` | 2-tuple | `FRF` — magnitude (dB) over phase (deg) |
| `plot_coherence(frequencies, coh, ax=None, …)` | ✓ | `FRF` coherence vs frequency |
| `plot_lsfir_diagnostics(Minfo, Vinfo, g)` | — | LS-FIR diagnostics: information matrix, vector, impulse response |

### Communications

→ [communications](comms.md).

| Plotter | ax | Draws |
|---|---|---|
| `plot_channel(h, sample_rate, ax=None, …)` | 2-tuple | \|h[n]\| and \|H(f)\| side by side |
| `plot_subcarriers(channel, n_subcarriers, ax=None, …)` | ✓ | channel magnitude across the OFDM subcarriers |
| `plot_constellation(constellation, ax=None, scheme='', annotate=True, …)` | ✓ | the ideal Gray-labelled constellation |
| `plot_scatter(symbols, ax=None, ideal=None, …)` | ✓ | received symbols, optionally over the ideal points |
| `plot_eye_diagram(signal, samples_per_symbol, ax=None, n_symbols=2, …)` | ✓ | overlaid symbol windows |
| `plot_convergence(mse, ax=None, label=None, …)` | ✓ | equaliser learning curve (MSE vs symbol, dB) |
| `plot_sync_metric(metric, ax=None, threshold=None, …)` | ✓ | synchronisation metric vs sample index |
| `plot_doppler_ambiguity(scales, peak_metric, ax=None, …)` | ✓ | peak correlation vs Doppler scale |
| `plot_ber_curve(ebn0_dB, ber_measured, ax=None, scheme=None, …)` | ✓ | measured BER vs Eb/N0, with the theory curve when `scheme=` is given |

### Noise and sonar

| Plotter | ax | Draws |
|---|---|---|
| `plot_wenz(wenz, ax=None, show_components=True, …)` | ✓ | a Wenz ambient-noise spectrum and its components → [noise](noise.md) |
| `plot_weighting(group, ax=None, frequency=None, …)` | ✓ | marine-mammal auditory weighting curve(s) |
| `plot_source_level(frequency, level_dB, ax=None, label=None, …)` | ✓ | a ship source-level spectrum |
| `plot_roc(deflection=None, ax=None, pfa=None, pd=None, …)` | ✓ | the ROC curve, `P_D` vs log `P_F` → [sonar](sonar.md) |

---

## 8. Gotchas

**`from uacpy.plot import …` does not work.** `uacpy.plot` is an attribute
alias for `uacpy.visualization.plots`, not an importable path. Use
`import uacpy` then `uacpy.plot.plot_field(...)`, or
`from uacpy.visualization import plot_field`.

**Slice, then plot.** There is no `depth=`/`range=` selection keyword on
`plot_field`. The field decides its own picture from `coords`, so the way to ask
for a cut is `field.at(depth=60.0).plot()`, not a plotter argument.

**A rejected keyword leaves no figure behind.** Branch validation happens before
anything is created, and the `plot_*` functions are wrapped in a guard that
closes any figure a failed call opened. Nothing half-drawn survives in
`plt.get_fignums()`.

**`ConfigurationError` is what a bad plot call raises**, including for degenerate
input (empty or mismatched-length arrays, wrong shapes, a missing arrival key)
that would otherwise leak a bare `IndexError`, `KeyError` or `ValueError` from
matplotlib. That same guard does the relabelling. A genuine wrong-type call still raises `TypeError` — that is a bug
in the caller, not bad input.

**Importing `uacpy.visualization` does not touch `matplotlib.rcParams`.** Your
own style sheet survives. Anything you want globally, set yourself.

**`env=` extends the depth axis, it does not clip the data.** If your receiver
grid reaches below the seafloor, those values are still drawn — the seabed fill
just covers them. That is a sign of a receiver grid that needs fixing, not a
plotting artefact.

**Every figure on this page is generated by committed code.** It is
[`docs/figure_scripts/plotting.py`](../figure_scripts/plotting.py); the snippets
above are that code. Regenerate with:

```bash
python docs/generate_model_figures.py plotting
```

---

**See also:** [Results](results.md) — the slicing that produces each render
branch · [Environment](environment.md) · [Source and receiver](source-receiver.md) ·
[Signal processing](signal.md) · [Array processing](arrays.md) ·
[Communications](comms.md) · [Noise](noise.md) · [Sonar](sonar.md) ·
[External data](data.md) · [documentation index](../README.md)
