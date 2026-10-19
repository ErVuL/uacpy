"""Animated wave propagation — wave-equation solvers against a ray solver.

A pulse crossing a Pekeris waveguide and bouncing off the seafloor, computed
five ways on one grid, as a snapshot sheet and as one GIF per solver.

* SPARC marches time natively, over the same water with a rigid floor: its
  deck carries only vacuum and rigid boundaries and it refuses the
  half-space, so its panel shows a different seabed, chosen explicitly
  below. Its range synthesis is a direct sum over a Δk grid, periodic in
  range with period RMax. Left unset, RMax is r + 1.2·c·T_MAX (r the
  farthest receiver, c the fastest speed; 524 m here, reported as
  `run_settings.engine.rmax`), so the folded replica stays outside the
  record whatever T_MAX is. At 1500 m/s, 0.18 s puts the front at 270 m —
  past the array, so reflections are visible.
* RAM, Scooter and Kraken get there through broadband H(f) → IFFT, which is a
  wave-equation solution too.
* Bellhop is the contrast: its TIME_SERIES output is a per-receiver
  delay-and-sum of arrivals, so the 2-D animation is a grid of independent
  traces rather than a coherent wavefield — the "wave" is stitched from
  neighbouring receivers. At 200 Hz over 50 m of water the eigenray sum still
  tracks the modal field; at lower frequencies or shorter ranges it does not.

Uses: RunMode.TIME_SERIES on five solvers · source_waveform= / sample_rate= /
output_duration= · Field.shift / Field.window (one display window across five
solvers) · plot_time_snapshots · save_animation
"""

import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustic_signal import gaussian_pulse

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

# 50 m of water over 200 m of range. Range sampling is 64 bins from 2 to 200 m
# = 3.14 m, against λ_min = 1500/350 = 4.3 m at the top of the band — about 1.4
# samples per minimum wavelength, so the panels render the pulse envelope and
# the modal arrivals but not individual wavefronts up there. Raise the bin
# count if you want those.
T_MAX = 0.18            # s — front at 270 m: past the array
F_CENTER, F_MIN, F_MAX = 200.0, 50.0, 350.0
FS = 8000.0             # ≥ 2× freq_max
receiver = uacpy.Receiver(depths=np.linspace(1, 49, 32),
                          ranges=np.linspace(2, 200, 64))

env = uacpy.Environment(
    name='Pekeris (animation)', bathymetry=50.0, ssp=1500.0,
    bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=1700.0, density=1.5,
                                    attenuation=0.5))
source = uacpy.Source(depths=25.0, frequencies=F_CENTER)   # mid-depth
# SPARC refuses a half-space seabed; its run gets the same water over a rigid
# floor, so its seafloor reflection is total where the others' is partial.
rigid_floor_env = uacpy.Environment(
    name='Pekeris water, rigid floor (SPARC)', bathymetry=50.0, ssp=1500.0,
    bottom=uacpy.BoundaryProperties(acoustic_type='rigid'))

# A Gaussian-windowed cosine whose peak sits at duration/2, so the early
# samples are identically zero (causality) and the spectrum stays narrow around
# F_CENTER. gaussian_pulse's width argument is σ·√2.
sigma_t = 0.003
duration = max(8 / F_CENTER, 6 * sigma_t)
t = np.arange(0, duration, 1.0 / FS)
peak_time = duration / 2.0
waveform = (gaussian_pulse(t, peak_time, sigma_t * np.sqrt(2))
            * np.cos(2 * np.pi * F_CENTER * (t - peak_time)))


def run(name, model, waveform=None, run_env=env):
    """One call site for every solver: TIME_SERIES on the shared window.

    SPARC builds p(t) from its own pulse_type; the IFFT models derive their
    frequency grid from the waveform's spectrum, zero-pad it internally to
    output_duration and auto-widen their rmax. Those carry the waveform's own
    peak offset into the output time axis, so it is shifted back out — every
    solver's wavefront then emerges from the source at t ≈ 0, matching SPARC's
    native convention.
    """
    # The notices belong to what this sheet shows and are printed: the
    # receivers start 2 m from the source, so RAM and Kraken drop the paths
    # steeper than their angular windows keep (Scooter integrates from
    # k = 0 and keeps them), and Bellhop's delay-and-sum drops the echoes
    # that land after the 0.2 s window.
    with warnings.catch_warnings(record=True) as caught:
        if waveform is None:
            field = model.run(run_env, source, receiver,
                              run_mode=uacpy.RunMode.TIME_SERIES,
                              output_duration=T_MAX)
        else:
            field = model.run(run_env, source, receiver,
                              run_mode=uacpy.RunMode.TIME_SERIES,
                              source_waveform=waveform, sample_rate=FS,
                              output_duration=T_MAX + peak_time)
            field = field.shift(time=-peak_time)
    for warning in caught:
        print(f"  noted: {str(warning.message).splitlines()[0]}")
    # Clip to 0 ≤ t ≤ T_MAX. SPARC integrates from t = -0.1 s (pre-roll) while
    # the IFFT models start at 0, so the lower bound drops that pre-roll.
    field = field.window(time=(0.0, T_MAX))
    print(f"  {name:8s} {field.data.shape}")
    return field


# What each constructor declares is the physics or numerics that has to stay
# pinned per solver; the wrappers handle the TIME_SERIES aliases and band
# derivation at run time. SPARC's pulse band and n_time_samples are its own
# pulse-shaping knobs (not equivalent to source_waveform); RAM's dr/dz are
# pinned for upper-band resolution (dz puts the 50 m seafloor a quarter cell
# below a node, where the fluid codes place the interface best), c0=1500
# matches the physical sound speed so its carrier wavelength lines up with the
# others, and n_pade=6 because mpiramS's Padé root finder does not converge
# with 8 terms at dr=1 m. Bellhop's fan reaches ±89.9° so the receivers 2 m
# from the source keep their direct path.
fields = {
    'SPARC': run('SPARC', uacpy.SPARC(n_time_samples=400, freq_min=F_MIN,
                                      freq_max=F_MAX),
                   run_env=rigid_floor_env),
    'Scooter': run('Scooter', uacpy.Scooter(), waveform),
    'RAM': run('RAM', uacpy.RAM(dr=1.0, dz=0.498753, c0=1500.0, n_pade=6),
               waveform),
    'Kraken': run('Kraken', uacpy.Kraken(), waveform),
    'Bellhop': run('Bellhop', uacpy.Bellhop(backend='fortran',
                                            launch_angles_deg=(-89.9, 89.9)),
                   waveform),
}

# Snapshots from t=0 (the emission) through the first seafloor reflection.
fig, _ = uacpy.plot.plot_time_snapshots(
    fields, times=tuple(np.linspace(0.0, T_MAX, 8)), env=env,
    title='Pulse propagation snapshots — Pekeris waveguide, fc=200 Hz '
          '(per-row colour scale: each solver has its own normalisation)')
fig.savefig(OUT / 'example_26_wave_propagation.png', dpi=140,
            bbox_inches='tight')
plt.close(fig)

for name, field in fields.items():
    uacpy.plot.save_animation(
        field, OUT / f'example_26_{name.lower()}.gif', env=env, fps=25,
        aspect=1 / 1000.0,
        title=f"{name} — Pekeris propagation (fc=200 Hz)")
