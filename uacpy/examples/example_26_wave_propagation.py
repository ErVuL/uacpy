"""Animated wave propagation — wave-equation solvers against a ray solver.

A pulse crossing a Pekeris waveguide and bouncing off the seafloor, computed
five ways on one grid, as a snapshot sheet and as one GIF per solver.

* SPARC marches time natively. Its range domain has an implicit periodic
  boundary from the wavenumber-FFT method, so T_MAX must end BEFORE the wave
  reaches the far edge or the late frames show aliasing rather than
  propagation. TIME_SERIES auto-widens the solver domain to 3× the receiver
  span (600 m here), and at 1500 m/s, 0.18 s puts the front at 270 m — past
  the array, so reflections are visible, and well inside the wrap edge. Adapt
  it as T_MAX < 3·max(receiver ranges)/c.
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
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustic_signal.waveforms import gaussian_pulse

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

# 50 m of water over 200 m of range. Range sampling is 64 bins from 2 to 200 m
# = 3.14 m, against λ_min = 1500/350 = 4.3 m at the top of the band — about 1.4
# samples per minimum wavelength, so the panels render the pulse envelope and
# the modal arrivals but not individual wavefronts up there. Raise the bin
# count if you want those.
T_MAX = 0.18            # s — front at 270 m: past the array, inside the wrap
F_CENTER, F_MIN, F_MAX = 200.0, 50.0, 350.0
FS = 8000.0             # ≥ 2× f_max
receiver = uacpy.Receiver(depths=np.linspace(1, 49, 32),
                          ranges=np.linspace(2, 200, 64))

env = uacpy.Environment(
    name='Pekeris (animation)', bathymetry=50.0, ssp=1500.0,
    bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=1700.0, density=1.5,
                                    attenuation=0.5))
source = uacpy.Source(depths=25.0, frequencies=F_CENTER)   # mid-depth

# A Gaussian-windowed cosine whose peak sits at duration/2, so the early
# samples are identically zero (causality) and the spectrum stays narrow around
# F_CENTER. gaussian_pulse's width argument is σ·√2.
sigma_t = 0.003
duration = max(8 / F_CENTER, 6 * sigma_t)
t = np.arange(0, duration, 1.0 / FS)
peak_time = duration / 2.0
waveform = (gaussian_pulse(t, peak_time, sigma_t * np.sqrt(2))
            * np.cos(2 * np.pi * F_CENTER * (t - peak_time)))


def run(name, model, waveform=None):
    """One call site for every solver: TIME_SERIES on the shared window.

    SPARC builds p(t) from its own pulse_type; the IFFT models derive their
    frequency grid from the waveform's spectrum, zero-pad it internally to
    output_duration and auto-widen their rmax. Those carry the waveform's own
    peak offset into the output time axis, so it is shifted back out — every
    solver's wavefront then emerges from the source at t ≈ 0, matching SPARC's
    native convention.
    """
    if waveform is None:
        field = model.run(env, source, receiver,
                          run_mode=uacpy.RunMode.TIME_SERIES)
    else:
        field = model.run(env, source, receiver,
                          run_mode=uacpy.RunMode.TIME_SERIES,
                          source_waveform=waveform, sample_rate=FS,
                          output_duration=T_MAX + peak_time)
        field = field.shift(time=-peak_time)
    # Clip to 0 ≤ t ≤ T_MAX. SPARC integrates from t = -0.1 s (pre-roll) while
    # the IFFT models start at 0, so the lower bound drops that pre-roll.
    field = field.window(time=(0.0, T_MAX))
    print(f"  {name:8s} {field.data.shape}")
    return field


# What each constructor declares is the physics or numerics that has to stay
# pinned per solver; the wrappers handle the TIME_SERIES aliases and band
# derivation at run time. SPARC's pulse band and n_t_out are its own
# pulse-shaping knobs (not equivalent to source_waveform); RAM's dr/dz are
# pinned for upper-band resolution and c0=1500 matches the physical sound speed
# so its carrier wavelength lines up with the others.
fields = {
    'SPARC': run('SPARC', uacpy.SPARC(n_t_out=400, t_max=T_MAX, f_min=F_MIN,
                                      f_max=F_MAX,
                                      max_depths=receiver.depths.size)),
    'Scooter': run('Scooter', uacpy.Scooter(), waveform),
    'RAM': run('RAM', uacpy.RAM(dr=1.0, dz=0.5, c0=1500.0), waveform),
    'Kraken': run('Kraken', uacpy.Kraken(), waveform),
    'Bellhop': run('Bellhop', uacpy.Bellhop(), waveform),
}

# Snapshots from t=0 (the emission) through the first seafloor reflection.
fig, _ = uacpy.plot.plot_time_snapshots(
    fields, times_s=tuple(np.linspace(0.0, T_MAX, 8)), env=env,
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
