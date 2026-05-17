"""
EXAMPLE 26: Animated wave propagation (SPARC vs RAM)
====================================================

Watch a pulse propagate through a Pekeris waveguide and bounce off the
seafloor — two wave-equation solvers shown on the same grid, same
physics target.

* **SPARC** — native time-marched FFP. The TIME_SERIES output is the
  pressure field the solver produced directly. The range domain has an
  implicit periodic boundary from the wavenumber-FFT method, so
  ``t_max`` must end *before* the wave reaches the far range edge
  (otherwise the field wraps back and the late-time animation shows
  aliasing, not propagation). Here ``RMax = 5 km``, ``c ≈ 1500 m/s``,
  so ``t_max = 2.5 s`` keeps the simulation in the physics-correct
  regime.
* **RAM via synthesize_time_series** — broadband PE H(f) → IFFT with a
  windowed Gaussian source. PE is a wave-equation solver too; the
  Fourier-domain time axis is what governs validity, and the IFFT
  window has a soft periodicity but the source-pulse envelope keeps
  late-time content small.

Bellhop is included for contrast: its TIME_SERIES output is a
*per-receiver* delay-and-sum of arrivals — not a wave-equation
solution. The 2-D animation is then a grid of independent time-series,
not a coherent wavefield, and the visual "wave" is an illusion
stitched from neighbour-receiver arrivals. At fc=200 Hz / 50 m water
the eigenray sum still tracks the modal field reasonably; at lower
frequencies or shorter ranges it falls apart.

The script saves:

* ``example_26_wave_propagation.png`` — 6-frame snapshot grid, SPARC + RAM.
* ``example_26_sparc.gif``           — SPARC animation.
* ``example_26_ram_ifft.gif``        — RAM animation.

ENVIRONMENT
    Pekeris guide, 100 m deep, fluid half-space bottom. Source at 20 m,
    pulse centred at 200 Hz. Both models share a 5-depth × 40-range
    receiver grid so the snapshot panels are directly comparable.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import uacpy  # noqa: E402
from uacpy.core.environment import BoundaryProperties  # noqa: E402
from uacpy.models import (  # noqa: E402
    RAM, SPARC, Scooter, KrakenField, Bellhop, RunMode,
)
from uacpy.visualization import (  # noqa: E402
    save_animation, plot_time_snapshots,
)

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


# Shared simulation parameters — 50 m water × 200 m of range, fine
# grid. Constraints kept in mind:
#   * RMax / safety-margin / interpolation-edge knobs (SPARC's
#     ``rmax_safety_margin``, Scooter's ``rmax_multiplier``,
#     KrakenField's ``rmax_m``) are now auto-widened to 3× receiver_max
#     in TIME_SERIES mode — no need to compute them by hand here.
#   * Range sampling resolves λ_min ≈ 4.3 m at f_max=350 Hz; 2 m
#     spacing → 100 range bins over 200 m.
BATHYMETRY = 50.0                                # water depth (m)
RECEIVER_DEPTHS = np.linspace(1, 49, 50)         # 1 m vertical spacing
RECEIVER_RANGES = np.linspace(2, 200, 100)       # 2 m horizontal spacing
T_MAX = 0.18           # seconds — wave at 270 m past array, shows reflections
F_CENTER = 200.0       # Hz, source-pulse centre frequency
F_MIN = 50.0           # Hz, pulse-band lower edge (SPARC only)
F_MAX = 350.0          # Hz, pulse-band upper edge (SPARC only)
SIGMA_T = 0.003        # s, Gaussian-pulse 1-σ width (~3 ms ≈ 4.5 m wide)
FS = 8000.0            # Hz, time-series sample rate (>= 2× f_max)


def _build_env_source():
    """Pekeris guide with a fluid half-space bottom + a 200 Hz source."""
    bottom = BoundaryProperties(
        acoustic_type='half-space', sound_speed=1700.0,
        density=1.5, attenuation=0.5,
    )
    env = uacpy.Environment(
        name='Pekeris (animation)', bathymetry=BATHYMETRY, ssp=1500.0,
        bottom=bottom,
    )
    # Source at mid-depth so the visualisation is roughly symmetric.
    source = uacpy.Source(depths=BATHYMETRY / 2.0, frequencies=F_CENTER)
    return env, source


def _gaussian_pulse(f_center: float, sigma_t: float, fs: float,
                    n_periods: int = 8):
    """Gaussian-windowed cosine. Returns ``(waveform, peak_time_s)``.

    The pulse peak sits at ``duration/2`` so the early samples are
    identically zero (causality), keeping the spectrum narrow around
    ``f_center``. ``peak_time_s`` is reported so the caller can shift
    the IFFT output's time axis to put the emission peak at t=0
    (matching SPARC's native convention).
    """
    duration = max(n_periods / f_center, 6 * sigma_t)
    t = np.arange(0, duration, 1.0 / fs)
    peak_time = duration / 2.0
    t_centered = t - peak_time
    envelope = np.exp(-0.5 * (t_centered / sigma_t) ** 2)
    return envelope * np.cos(2 * np.pi * f_center * t_centered), peak_time


def _clip_to_window(field, t_max):
    """Return ``field`` with its time axis clipped to ``0 ≤ t ≤ t_max``
    so every model shares the same display window. SPARC's integration
    starts at ``t_start = -0.1 s`` (pre-roll); the IFFT models start
    at t=0, so the lower bound drops SPARC's negative-t pre-roll."""
    times = np.asarray(field.coords['time'])
    keep = (times >= 0.0) & (times <= t_max)
    if keep.all():
        return field
    from uacpy.core.results import Field
    data = np.moveaxis(np.asarray(field.data),
                       list(field.coords).index('time'), 2)[:, :, keep]
    return Field(
        data=data,
        coords={'depth': field.coords['depth'],
                'range': field.coords['range'],
                'time': times[keep]},
        model=field.model, backend=field.backend,
        source_depths=field.source_depths,
        frequencies=field.frequencies,
        phase_reference=field.phase_reference,
    )


def _shift_time(field, dt: float):
    """Return ``field`` with its time coord shifted by ``dt`` seconds
    (no data change). Used to put the source-emission peak at t=0 for
    IFFT-based syntheses that otherwise carry the waveform's peak
    offset (``duration/2``) into the output time axis."""
    if dt == 0.0:
        return field
    from uacpy.core.results import Field
    return Field(
        data=field.data,
        coords={'depth': field.coords['depth'],
                'range': field.coords['range'],
                'time': np.asarray(field.coords['time']) + dt},
        model=field.model, backend=field.backend,
        source_depths=field.source_depths,
        frequencies=field.frequencies,
        phase_reference=field.phase_reference,
    )


def _run(name, model, env, source, receiver, waveform=None,
         waveform_peak_t: float = 0.0):
    """Single call site for every solver: TIME_SERIES with output_duration.

    SPARC builds p(t) from its native ``pulse_type``; the IFFT models
    (RAM / Scooter / KrakenField / Bellhop) auto-derive their frequency
    grid from the source-waveform spectrum, zero-pad the waveform
    internally to ``output_duration``, and auto-widen their ``rmax_*``.
    ``waveform_peak_t`` is the time-offset of the source-emission peak
    within the user's waveform array; shifting the output time axis by
    ``-waveform_peak_t`` aligns the emission peak with t=0 so every
    solver's wavefront emerges from the source at t≈0.
    """
    print(f"  Running {name}...", end=' ', flush=True)
    if waveform is None:
        field = model.run(env, source, receiver,
                          run_mode=RunMode.TIME_SERIES)
    else:
        field = model.run(env, source, receiver,
                          run_mode=RunMode.TIME_SERIES,
                          source_waveform=waveform, sample_rate=FS,
                          output_duration=T_MAX + waveform_peak_t)
        field = _shift_time(field, -waveform_peak_t)
    field = _clip_to_window(field, T_MAX)
    print(f"✓  shape={field.data.shape}")
    return field




def main():
    print("\n" + "═" * 80)
    print("EXAMPLE 26: Animated wave propagation (SPARC vs RAM)")
    print("═" * 80)

    env, source = _build_env_source()
    waveform, waveform_peak_t = _gaussian_pulse(F_CENTER, SIGMA_T, FS)
    receiver = uacpy.Receiver(depths=RECEIVER_DEPTHS, ranges=RECEIVER_RANGES)

    # Per-model construction. The wrappers handle every alias / wrap /
    # band-derivation detail for TIME_SERIES at run-time; what each
    # constructor declares below is just the physics / numerics that
    # have to stay pinned per solver:
    #   * SPARC builds p(t) natively — pulse band + n_t_out are SPARC's
    #     own pulse-shaping knobs (not equivalent to source_waveform).
    #   * RAM: dr/dz pinned for upper-band resolution; c0=1500 matches
    #     the physical sound speed so the PE carrier wavelength (c0/fc)
    #     lines up with the other three solvers.
    models = [
        ('SPARC', SPARC(
            verbose=False, n_t_out=400, t_max=T_MAX,
            f_min=F_MIN, f_max=F_MAX,
            max_depths=len(RECEIVER_DEPTHS),
        ), None),
        ('Scooter', Scooter(verbose=False), waveform),
        ('RAM', RAM(verbose=False, dr=1.0, dz=0.5, c0=1500.0), waveform),
        ('KrakenField', KrakenField(verbose=False), waveform),
        # Bellhop is a ray solver — its TIME_SERIES output is the
        # delay-and-sum of arrivals per receiver, not a wave-equation
        # solution. At 200 Hz / 50 m water / 200 m range the
        # eigenray sum still tracks the modal field reasonably, but
        # the visual will look stripier than the three full-wave
        # solvers because each (z, r) cell is an independent trace.
        ('Bellhop', Bellhop(verbose=False), waveform),
    ]

    fields = {}
    for name, model, wf in models:
        try:
            fields[name] = _run(
                name, model, env, source, receiver, wf,
                waveform_peak_t=(waveform_peak_t if wf is not None else 0.0),
            )
        except Exception as exc:
            print(f"  {name} skipped: {exc}")

    if not fields:
        print("\n  No models produced a time-series field — nothing to plot.")
        return

    # Snapshot times spaced across the t_max window, starting from t=0
    # to capture the source emission. Later frames show the wave
    # propagating outward and the first seafloor reflection.
    target_times = np.linspace(0.0, T_MAX, 8)

    print("\n  Building snapshot grid...")
    fig, _ = plot_time_snapshots(
        fields, times_s=tuple(target_times), env=env,
        title=(
            'Pulse propagation snapshots — Pekeris waveguide, fc=200 Hz '
            '(per-row colour scale: each solver has its own absolute '
            'pressure normalisation)'
        ),
    )
    out_png = OUTPUT_DIR / 'example_26_wave_propagation.png'
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {out_png}")

    print("\n  Saving per-model GIFs...")
    for name, field in fields.items():
        safe_name = name.lower().replace(' ', '_')
        out_gif = OUTPUT_DIR / f'example_26_{safe_name}.gif'
        try:
            save_animation(
                field, out_gif, env=env, fps=25,
                aspect=1 / 1000.0,
                title=f"{name} — Pekeris propagation (fc=200 Hz)",
            )
            print(f"  ✓ Saved: {out_gif}")
        except Exception as exc:
            print(f"  {name} animation skipped: {exc}")

    print(f"\n✓ Example 26 complete — outputs under {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
