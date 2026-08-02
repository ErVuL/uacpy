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
| **[Scooter](scooter.md)** | Wavenumber integration (FFP) | **Reference-grade** exact solution; range-independent |
| **[SPARC](sparc.md)** | Time-domain FFP | **Transient** propagation — watch a pulse evolve |
| **[RAM](ram.md)** | Parabolic equation | **Strongly range-dependent** environments, long ranges |
| **[Bounce](bounce.md)** | Plane-wave reflection | Not a propagator — computes seabed `R(θ)` |
| **[OASES](oases.md)** | Seismo-acoustic wavenumber integration | Full **elastic** seabed physics; arrays and MFP |

### By question

- *"What paths connect my source and receiver?"* → **[Bellhop](bellhop.md)** (eigenrays, arrivals)
- *"It's 50 Hz in 80 m of water."* → **[Kraken](kraken.md)** — the ray approximation is invalid here
- *"Is my answer right?"* → **[Scooter](scooter.md)** or **[OASES](oases.md)** — no ray or one-way approximation
- *"The seabed slopes from 100 m to 2 km."* → **[RAM](ram.md)**
- *"I need the impulse response of a pulse."* → **[SPARC](sparc.md)**, or Bellhop's `BROADBAND`
- *"My seabed has shear."* → **[OASES](oases.md)**, or **[Bounce](bounce.md)** for the boundary alone
- *"I have a hydrophone array."* → **[OASES](oases.md)** (OASN) + [array processing](../guide/arrays.md)

---

## Capability matrix

What each model consumes **natively**. Anything marked ✗ is *collapsed* to
something the model can take, with a `UserWarning` naming what was dropped —
see [collapse policy](../guide/environment.md).

| | Bellhop | Kraken | Scooter | SPARC | RAM | Bounce | OASES |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Range-dep. bathymetry | ✅ | ✅ | ✗ | ✗ | ✅ | ✗ | ✗ |
| Range-dep. SSP | ✅ | ✅ | ✗ | ✗ | ✅ | ✗ | ✗ |
| Range-dep. bottom | ✅ | ✗ | ✗ | ✗ | ✅ | ✗ | ✗ |
| Sea-surface altimetry | ✅ | ✗ | ✗ | ✗ | ✅ | ✗ | ✗ |
| Layered bottom | ✅¹ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Rough surface/bottom (`sigma`) | ✗ | ✅ | ✅ | ✗ | ✗ | ✗ | ✗ |
| Elastic media (shear) | ✅¹ | ✅² | ✅ | ✗ | ✅³ | ✅ | ✅ |
| Multiple source depths | ✅ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

¹ via the auto-BOUNCE reflection table — the layer stack is kept and `R(θ)` is
exact, but BOUNCE is range-independent, so the seabed collapses to one column.
² requires `backend='krakenc'`; auto-selected.
³ routes to the `rams` backend.

`OASES` is an abstract base: instantiate **OAST** (TL), **OASN** (covariance,
replicas), **OASR** (reflection) or **OASP** (pulse/broadband) directly, or let
`OASES.for_mode(run_mode=...)` pick. Its columns here are the union across the
four.

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
- **Shear** rules out the fluid solvers. [OASES](oases.md) is the honest answer.

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
compare Bellhop's TL plot with Kraken's, you are comparing the same water.

Model pages need the native binaries (`./install.sh`); OASES additionally
needs `--oases yes`.

---

**See also:** [documentation index](../README.md) ·
[environment carriers](../guide/environment.md) ·
[results and slicing](../guide/results.md) · [plotting](../guide/plotting.md)
