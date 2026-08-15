# uacpy documentation

A unified Python interface to the standard underwater-acoustic propagation
codes — plus the signal processing, communications, noise and sonar tooling
you need around them.

These pages are the **teaching material**: what each piece does, when to reach
for it, and worked examples with figures. For a terse API reference — every
signature, every keyword, every unit — see
[`DOCUMENTATION.md`](../DOCUMENTATION.md). For internals and how to extend the
package, see [`DEV.md`](DEV.md).

---

## Start here

```python
import numpy as np
import uacpy
from uacpy.models import Bellhop, RunMode

env = uacpy.Environment(
    bathymetry=100.0,
    ssp=[(0.0, 1500.0), (100.0, 1490.0)],
    bottom=uacpy.BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1650.0, density=1.8, attenuation=0.6,
    ),
)
source = uacpy.Source(depths=25.0, frequencies=200.0)
receiver = uacpy.Receiver(depths=np.linspace(1, 99, 100),
                          ranges=np.linspace(50, 5000, 250))

tl = Bellhop().run(env, source, receiver, run_mode=RunMode.COHERENT_TL)
tl.plot(env=env, source=source)
```

Three carriers in — [environment](guide/environment.md),
[source and receiver](guide/source-receiver.md) — one
[result](guide/results.md) out, which [plots itself](guide/plotting.md).
Every model in the package follows that shape.

---

## Propagation models

**→ [Model index and capability matrix](models/README.md)** — start here if you
are choosing a model.

| | | |
|---|---|---|
| [Bellhop](models/bellhop.md) | Gaussian-beam rays | rays, eigenrays, arrivals; high frequency |
| [Kraken](models/kraken.md) | Normal modes | low frequency; the modes themselves |
| [Scooter](models/scooter.md) | Wavenumber integration | reference-grade, exact |
| [SPARC](models/sparc.md) | Time-domain FFP | transient pulses |
| [RAM](models/ram.md) | Parabolic equation | strong range dependence, long range |
| [Bounce](models/bounce.md) | Plane-wave reflection | seabed `R(θ)` |
| [OASES](models/oases.md) | Seismo-acoustic | full elastic seabed; arrays, MFP |

---

## Guide

### Describing the problem
- **[Environment](guide/environment.md)** — bathymetry, sound-speed profiles,
  seabeds, absorption, and the collapse policy that governs what a model can
  actually consume.
- **[Source and receiver](guide/source-receiver.md)** — geometry, frequencies,
  beam patterns, array layouts.

### Working with what comes back
- **[Results](guide/results.md)** — `Field`, `Rays`, `Modes` and friends; how
  `Field.kind` / `.unit` / `.dtype` describe one container, and how `.at` / `.isel` /
  `.max` slice an axis into `.pinned`.
- **[Plotting](guide/plotting.md)** — the `.plot()` convention, `plot_field`'s
  three render branches, overlays and composition.

### Analysis
- **[Signal processing](guide/signal.md)** — waveforms, time-frequency
  (spectrogram, CWT, Wigner-Ville), FK/τ-p/Radon transforms, matched filtering.
- **[Array processing](guide/arrays.md)** — steering vectors, conventional and
  MVDR beamforming, MUSIC.
- **[Communications](guide/comms.md)** — modulation, OFDM, coding, equalisation,
  Doppler, synchronisation, DSSS and JANUS.
- **[Noise](guide/noise.md)** — Wenz curves, wind and shipping noise, ship
  radiated noise, marine-mammal auditory weighting.
- **[Sonar](guide/sonar.md)** — the sonar equation, detection theory,
  reverberation, scattering, matched-field processing.

### Data and plumbing
- **[External data](guide/data.md)** — build an `Environment` from GPS
  coordinates: bathymetry, sound speed, seabed, and the provenance uacpy
  attaches.
- **[File I/O](guide/io.md)** — the readers and writers behind every model.
- **[Utilities](guide/utilities.md)** — material presets, TL metrics,
  sound-speed and density helpers, parallel batch runs.

---

## How these pages stay honest

Every figure is produced by committed code under
[`docs/figure_scripts/`](figure_scripts/), one module per page, driven by:

```bash
python docs/generate_model_figures.py
```

The worked example shown on a page **is** the code that generated its figures —
not a re-typed approximation of it. The model pages also share a small set of
canonical environments, so comparing two models' plots compares the same water
rather than two different setups.

If a documented call stops working, the figure fails to generate and the
script exits non-zero.

Cross-references and page structure are gated by the test suite, not by that
script: `uacpy/tests/test_documentation.py` runs
[`check_links.py`](check_links.py) and [`check_structure.py`](check_structure.py)
over this tree on every `pytest` run — a dead link, an unbalanced fence, a
duplicated block or a code sample that no longer parses fails CI. Both also run
standalone:

```bash
python docs/check_links.py
python docs/check_structure.py
```

---

## Conventions

- **Units are SI throughout**: metres, m/s, g/cm³, Hz, dB re 1 µPa.
  Range and depth are metres — kilometres appear only on plot axes and inside
  the native file formats.
- **Depth is positive down**; sea-surface altimetry is positive up.
- **Range is measured from the source**, which sits at `r = 0`.
- **Results carry no carriers.** A result knows its model, backend, provenance
  and frequencies — never the `Environment` it ran against. Pass `env=` to a
  plotter when you want the seabed drawn.

---

**Reference:** [`DOCUMENTATION.md`](../DOCUMENTATION.md) ·
**Internals:** [`DEV.md`](DEV.md) ·
**Examples:** [`uacpy/examples/`](../uacpy/examples/) (39 runnable scripts)
