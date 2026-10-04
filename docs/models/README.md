# Propagation models

uacpy wraps seven acoustic engines behind one API — six propagators plus
[Bounce](bounce.md), which tabulates a boundary rather than propagating. They
differ in the approximation they make, and that approximation is what decides
which one is right for your problem — not speed, and not convenience.

Every model takes the same three carriers and returns the same result types:

```python
result = Model(**knobs).run(env, source, receiver, run_mode=...)
```

---

## Pick a model

| Model | Method | Regime it owns |
|---|---|---|
| **[Bellhop](bellhop.md)** | Gaussian-beam ray tracing | High frequency, range-dependent, and the **only** source of rays, eigenrays and arrivals |
| **[Kraken](kraken.md)** | Normal modes | Low frequency, shallow water; gives you the **modes** themselves |
| **[Scooter](scooter.md)** | Wavenumber integration (FFP) | **Reference-grade** exact solution; range-independent — over the whole spectrum only with `c_high=1e9` (the default cut drops paths steeper than `arccos(c/c_high)`, and warns when a receiver sees one) |
| **[SPARC](sparc.md)** | Time-domain FFP | **Transient** propagation — watch a pulse evolve; vacuum or rigid seabed only |
| **[RAM](ram.md)** | Parabolic equation | **Strongly range-dependent** environments, long ranges |
| **[Bounce](bounce.md)** | Plane-wave reflection | Not a propagator — computes seabed `R(θ)` |
| **[OASES](oases.md)** | Seismo-acoustic wavenumber integration | Full **elastic** seabed physics; arrays and MFP |

### By question

- *"What paths connect my source and receiver?"* → **[Bellhop](bellhop.md)** (eigenrays, arrivals)
- *"It's 50 Hz in 80 m of water."* → **[Kraken](kraken.md)** — the ray approximation is invalid here
- *"Is my answer right?"* → **[Scooter](scooter.md)** (with `c_high=1e9` and a small `taper`) or **[OASES](oases.md)** — no ray or one-way approximation
- *"The seabed slopes from 100 m to 2 km."* → **[RAM](ram.md)**
- *"I need the impulse response of a pulse."* → **[SPARC](sparc.md)** over a vacuum or rigid seabed; over a half-space, Scooter's or Bellhop's `TIME_SERIES` / `BROADBAND`
- *"My seabed has shear."* → **[OASES](oases.md)**, or **[Bounce](bounce.md)** for the boundary alone
- *"I have a hydrophone array."* → **[OASES](oases.md)** (OASN) + [array processing](../guide/arrays.md)

---

## Capability matrix

What each model consumes **natively**. An *environment* feature marked ✗ is
*collapsed* to something the model can take, with a `FallbackWarning` naming what
was dropped — see [collapse policy](../guide/environment.md). The source row
is the exception: a ✗ model reads one depth per deck, and instead of
collapsing, `run()` loops over the depths in the field modes (TL,
`BROADBAND`, `TIME_SERIES`) — each depth in its own `source_depth_<z>m`
subdirectory of a pinned `work_dir` — and returns a `ResultStack`; in any
other mode a multi-depth `Source` raises `ConfigurationError`.

| | Bellhop | Kraken | Scooter | SPARC | RAM | Bounce | OASES |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Range-dep. bathymetry | ✅ | ✅ | ✗ | ✗ | ✅ | ✗ | ✗ |
| Range-dep. SSP | ✅ | ✅ | ✗ | ✗ | ✅ | ✗ | ✗ |
| Range-dep. bottom | ✅ | ✅ | ✗ | ✗ | ✅ | ✗ | ✗ |
| Sea-surface altimetry | ✅ | ✗ | ✗ | ✗ | ✅⁹ | ✗ | ✗ |
| Layered bottom | ✅¹ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Rough surface/bottom (`sigma`) | ✗ | ✅ | ✅⁴ | ✗ | ✗ | ✗ | ✅ |
| Elastic media (shear) | ✅¹ | ✅² | ✅ | ✗ | ✅³ | ✅ | ✅ |
| Multiple source depths | ✅ | ✅⁷ | ✅⁷ | ✗ | ✗ | ✗ | ✗ |
| Water-column absorption (`env.absorption`) | ✅ | ✅ | ✅ | ✗⁸ | ✅⁵ | ✗ | ✅⁶ |

¹ an elastic *half-space* is native (bellhop.f90 applies its exact
acousto-elastic `R(θ)`); a layered bottom goes via the auto-BOUNCE reflection
table — the layer stack is kept and `R(θ)` is exact, but BOUNCE is
range-independent, so the seabed collapses to one column.
² requires `backend='krakenc'`; auto-selected.
³ routes to the `rams` backend.
⁴ sea surface only, and only under a pressure-release (vacuum) surface —
Scooter drops seabed roughness, and a rough rigid/elastic surface, with a
warning. Kraken and OASES carry both interfaces unconditionally.
⁵ as a dB-per-wavelength profile on the water wavenumber in every backend,
per bin on a broadband sweep (uacpy-patched binaries). Bounce tabulates a
reflection coefficient at an interface, so volume loss has no path to act on.
⁶ as each water layer's attenuation in dB/wavelength, exact at the deck
frequency; a multi-frequency deck (OASP, OASSP, an OAST/OASN sweep) carries
one value per layer for the whole band, evaluated at the frequency whose
line departs least from the law across the band and the water column
(`uacpy.core.absorption.minimax_anchor_frequency`; the band centre for a law
already linear in frequency), and OASES re-applies it at
every frequency (`oaseun31.f:1522`), so across the band the loss grows
linearly in frequency; a run whose band and water column put that more than
0.05 dB/km off the law warns. With `env.absorption=None` the water is lossless.
OASR's water is a lossless half-space.
⁷ TL modes only. One Scooter wavenumber sweep writes every depth into the
`.grn`, and one Kraken mode solve serves every depth of the `.flp`, so each
launches its binary once (measured 4.4× for Kraken at eight depths);
`BROADBAND` / `TIME_SERIES` loop per depth on both.
⁸ SPARC's march is lossless: `sparc.f90:221` keeps the real part of the
sound speed only, so water absorption and seabed attenuation are ignored, and
a run with either warns (see [SPARC](sparc.md#the-march-is-lossless)).
⁹ depressions only, through the `ramsurf` backend: a crest above `z = 0` is
clamped to the mean sea level with a `FallbackWarning`, which flattens about
half of a two-sided sea surface such as `generate_sea_surface`'s. A two-sided
wave field needs Bellhop.

`OASES` is an abstract base: instantiate **OAST** (TL), **OASN** (covariance,
replicas), **OASR** (reflection), **OASP** (pulse/broadband), **OASS**
(reverberation) or **OASSP** (scattered pulse) directly, or let
`OASES.for_mode(run_mode=...)` pick. Its columns here are the union across the
six.

## Run modes

| | Bellhop | Kraken | Scooter | SPARC | RAM | Bounce | OASES |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `COHERENT_TL` | ✅ | ✅ | ✅ | ✗ | ✅ | ✗ | ✅ |
| `INCOHERENT_TL` | ✅ | ✅ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `SEMICOHERENT_TL` | ✅ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `RAYS` / `EIGENRAYS` | ✅ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `ARRIVALS` | ✅ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `MODES` | ✗ | ✅ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `BROADBAND` | ✅ | ✅ | ✅ | ✗ | ✅ | ✗ | ✅ |
| `TIME_SERIES` | ✅ | ✅ | ✅ | ✅ | ✅ | ✗ | ✅ |
| `REFLECTION` | ✗ | ✗ | ✗ | ✗ | ✗ | ✅ | ✅ |
| `COVARIANCE` / `REPLICA` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✅ |
| `REVERBERATION` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✅ |

Generated from `model.supported_modes` — see each page for the authoritative list.

---

## Validity: which approximation breaks when

The most useful single number is **`D/λ`**, the water depth in wavelengths — a
first filter on the depth scale, not a criterion. The underlying requirement is
that the wavelength be small compared to *every* physical scale in the problem,
duct thickness and bathymetric relief included.

| `D/λ` | What works |
|---|---|
| `≲ 5` | Modal ([Kraken](kraken.md)) or exact ([Scooter](scooter.md), [OASES](oases.md)). Rays are meaningless. |
| `5 – 20` | Transition. Cross-check a ray answer against a modal one. |
| `≳ 20` | [Bellhop](bellhop.md) is accurate and far cheaper. Modes become too numerous to be useful. |

Independently of frequency:

- **Range dependence** rules out Scooter, SPARC, Bounce and OASES (all
  stratified solvers) unless you accept a collapse. [RAM](ram.md) and
  [Bellhop](bellhop.md) are built for it; [Kraken](kraken.md) segments.
- **Backscatter** rules out [RAM](ram.md) — the parabolic equation is one-way.
- **Shear** rules out the fluid solvers. Over a range-independent elastic
  seabed [Scooter](scooter.md), [Kraken](kraken.md) (`krakenc`) and
  [OASES](oases.md) are exact and agree — on limestone and chalk
  half-spaces under 100 m of water at 200 Hz, range-averaged levels over
  1–9 km sit within 0.01 dB (Scooter and Kraken) and 0.11 dB (OAST) of
  each other; where the seabed varies in range,
  [RAM](ram.md)'s `rams` backend.

To check an engine rather than compare two, score it against a closed form:
`uacpy.analytic` gives the free field, the Lloyd mirror, the ideal waveguide
and the Pekeris waveguide as engine-shaped `Field`s
([utilities](../guide/utilities.md) §2).
The [validation page](validation.md) gives each engine's measured agreement
with those closed forms and with the published range-dependent benchmarks.

---

## Reproducing the figures

Every figure in this section is generated by committed code:

```bash
python docs/generate_model_figures.py            # all pages
python docs/generate_model_figures.py bellhop    # one page
python docs/generate_model_figures.py --list     # what exists
```

Each page's worked example **is** its figure code
([`docs/figure_scripts/`](../figure_scripts/)), so a snippet cannot drift from
the image it claims to produce. The model pages share a small set of canonical
environments from [`_common.py`](../figure_scripts/_common.py) — so when you
compare Bellhop's TL plot with Kraken's, you are comparing the same water. A
snippet's `WIDE` / `TALL` figure sizes and its scenario come from there too:
run it with `docs/` on `sys.path` and
`from figure_scripts._common import shallow_water, WIDE, TALL`.

Model pages need the native binaries (`./install.sh`); OASES additionally
needs `--oases yes`.

---

**See also:** [documentation index](../README.md) ·
[environment carriers](../guide/environment.md) ·
[results and slicing](../guide/results.md) · [plotting](../guide/plotting.md)
