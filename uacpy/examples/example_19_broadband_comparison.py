"""Broadband comparison — transfer functions and time series across models.

Eight solvers on one Pekeris waveguide, each producing the broadband quantity
it naturally produces, then all of them turned into time traces at the same
receiver: Bellhop (arrivals → H(f), and delay-and-sum for a chirp), the three
fluid RAM backends (mpiramS, ramgeo, ramsurf1.5), Scooter and Kraken
(multi-frequency FFP and modes), OASP (OASES transient), and SPARC
(time-marched FFP, which returns p(t) directly).

Three things make the comparison fair:

* ONE seabed object, passed to every model. Leaving `bottom=` off would fall
  back to the Environment default for some models while others got an explicit
  half-space — they would then be solving different waveguides.
* ONE frequency grid, 50-150 Hz at df = 1 Hz. Fine df matters twice over: a
  coarse grid produces periodic replicas of the impulse response before the
  geometric arrival when the IFFT window exceeds 1/df, and the Pekeris guide is
  dispersive, so slower modal group velocities arrive after the first arrival
  and build a real coda that df has to resolve. The RAM backends size their
  grid from (Q, T) and the rest from the frequencies array; both land here.
* SPARC is the stated exception: it accepts only vacuum/rigid boundaries, so
  uacpy converts the half-space to rigid and warns. Its trace shows the
  time-marching method, not the same physics.

Two things in the figures are worth reading carefully. Bellhop's |H| sits about
2.7 dB under the six full-wave models: this 100 m guide is only D/λ = 3-10
wavelengths deep over 50-150 Hz, far outside the D/λ ≳ 100 that ray theory
wants, and the gap closes to 0.1 dB by D/λ = 67 and 0.0 dB by D/λ = 200
(measured). The phase panel removes the bulk travel time before plotting,
because the raw phase of a 3.3 s delay is aliased on any grid this coarse —
see the comment at that panel.

The elastic RAM backend rams0.5 is deliberately absent: its rotated-Padé march
is only marginally stable (|G| ≈ 1 for the below-real-line elastic eigenvalues,
Collins & Siegmann §3.3, Milinazzo 1997), and that error compounds across a
wide sweep plus IFFT. It is robust in NARROWBAND TL — within ~0.1 dB of krakenc
on the elastic Pekeris — which is its proper regime.

Uses: RunMode.BROADBAND across six models · RunMode.TIME_SERIES (SPARC native,
Bellhop delay-and-sum with source_waveform=) · RAM(Q=, T=, backend=) ·
Field.to_time_trace · Field.at(frequency=) · plot.compare(value='mag')
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustic_signal.waveforms import lfm_chirp

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

seabed = uacpy.BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1600.0, density=1.5,
                                  attenuation=0.5)
env = uacpy.Environment(name='Pekeris waveguide', bathymetry=100, ssp=1500,
                        bottom=seabed)
source = uacpy.Source(depths=36, frequencies=100)
receiver = uacpy.Receiver(depths=np.linspace(5, 95, 12),
                          ranges=np.array([5000.0]))
frequencies = np.arange(50.0, 150.0 + 0.5, 1.0)
TARGET_DEPTH, TARGET_RANGE = 50.0, 5000.0
print(f"  {env.depth:.0f} m Pekeris, source {source.depths[0]:.0f} m, "
      f"receiver {TARGET_RANGE / 1000:.0f} km, "
      f"{frequencies[0]:.0f}-{frequencies[-1]:.0f} Hz at "
      f"df={frequencies[1] - frequencies[0]:.0f} Hz")

# Six transfer-function models on the shared grid. OASP rebuilds an equispaced
# grid of its own, so its sweep bounds go on the constructor.
fields = {
    name: model.run(env, source, receiver,
                    run_mode=uacpy.RunMode.BROADBAND, frequencies=frequencies)
    for name, model in (
        ('Bellhop', uacpy.Bellhop()),
        ('Scooter', uacpy.Scooter()),
        ('Kraken', uacpy.Kraken()),
        ('OASP', uacpy.OASP(n_time_samples=512,
                            freq_max=float(frequencies[-1]),
                            freq_min=float(frequencies[0]))),
    )
}

# The RAM dispatcher routes by environment: fluid Pekeris + flat surface →
# mpiramS; the same fluid Pekeris carrying a flat z=0 altimetry line →
# ramsurf1.5 (the altimetry only selects the code path); backend='ramgeo'
# forces the third. Only the broadband window (Q, T) is supplied — dr and dz
# come from the Lytaev optimizer. Q=2, T=1 gives fc ± 50 Hz at df = 1 Hz, the
# same grid as the frequencies array above.
flat_env = env
altimetry_env = uacpy.Environment(name='Pekeris-fluid-altimetry',
                                  bathymetry=env.depth, ssp=1500.0,
                                  bottom=seabed,
                                  altimetry=[(0.0, 0.0), (8000.0, 0.0)])
for label, ram_env, kwargs in (
        ('RAM (mpiramS)', flat_env, {}),
        ('RAM (ramgeo)', flat_env, {'backend': 'ramgeo'}),
        ('RAM (ramsurf1.5)', altimetry_env, {})):
    model = uacpy.RAM(Q=2.0, T=1.0, **kwargs)
    print(f"  {label:18s} → backend {model.select_backend(ram_env)}")
    fields[label] = model.run(ram_env, source, receiver,
                              run_mode=uacpy.RunMode.BROADBAND)

for name, field in fields.items():
    print(f"  {name:18s} H{field.data.shape} over "
          f"{field.frequencies[0]:.0f}-{field.frequencies[-1]:.0f} Hz")

# SPARC marches time directly. n_t_out sets the output rate, n_t_out / t_max:
# 4800 over 4 s is 1200 Hz (Nyquist 600 Hz), above the 200 Hz top of the source
# band. 1001 would give 250 Hz and alias p(t).
sparc_receiver = uacpy.Receiver(depths=np.array([50.0]),
                                ranges=np.linspace(500, 5000, 5))
sparc = uacpy.SPARC(n_t_out=4800, t_max=4.0, f_min=50.0, f_max=200.0).run(
    env, source, sparc_receiver, run_mode=uacpy.RunMode.TIME_SERIES)
print(f"  SPARC              p{sparc.data.shape}, dt={sparc.dt * 1e3:.3f} ms "
      f"(rigid bottom — see the module docstring)")

# Bellhop's other time-domain route: delay-and-sum of its arrivals against a
# real source waveform.
fs, chirp_duration = 2000.0, 0.1
t_chirp, chirp = lfm_chirp(50.0, 150.0, chirp_duration, fs)
chirp_response = uacpy.Bellhop().run(
    env, source,
    uacpy.Receiver(depths=np.array([TARGET_DEPTH]),
                   ranges=np.array([TARGET_RANGE])),
    run_mode=uacpy.RunMode.TIME_SERIES, source_waveform=chirp, sample_rate=fs)
print(f"  Bellhop (chirp)    p{chirp_response.data.shape}, "
      f"peak {np.max(np.abs(chirp_response.data)):.3e}")

# ── The transfer functions, magnitude and phase ─────────────────────────────
mid_depth = float(receiver.depths[receiver.depths.size // 2])
spectra = [f.at(depth=mid_depth, range=TARGET_RANGE) for f in fields.values()]
fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                       gridspec_kw={'hspace': 0.25})
uacpy.plot.compare(spectra, labels=list(fields), value='mag', ax=ax_mag,
                   title='Magnitude |H(f)|')
# The RAW phase cannot be read on this grid, for any model. The bulk travel
# time is r/c0 = 3.3 s, and a phase sampled every df Hz is unambiguous only for
# delays under 1/(2·df) — 0.5 s at this 1 Hz grid — so every curve would be
# aliased, each model by its own sampling. (OASP rebuilds a finer grid of its
# own, so it aliases differently and merely LOOKS like different physics.)
# Removing the bulk delay leaves the multipath residual, which the grid does
# resolve: all seven then agree, at about 7 ms.
bulk_delay = TARGET_RANGE / 1500.0
for name, spectrum in zip(fields, spectra):
    spectrum_hz = np.asarray(spectrum.coords['frequency'], dtype=float)
    compensated = (np.asarray(spectrum.data).ravel()
                   * np.exp(2j * np.pi * spectrum_hz * bulk_delay))
    ax_phase.plot(spectrum_hz, np.angle(compensated), label=name)
ax_phase.set_ylabel('Phase (rad)')
ax_phase.legend(fontsize=8)
ax_phase.grid(True, alpha=0.3)
ax_phase.set_title(f'Phase ∠H(f), bulk delay r/c₀ = {bulk_delay * 1e3:.0f} ms '
                   f'removed')
ax_mag.set_xlabel('')
ax_phase.set_xlabel('Frequency (Hz)', fontweight='bold')
fig.suptitle(f'Transfer functions — depth {mid_depth:.0f} m, range '
             f'{TARGET_RANGE / 1000:.0f} km', fontsize=13, fontweight='bold',
             y=0.995)
fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.97)
fig.savefig(OUT / 'example_19_transfer_functions.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

# ── TL against depth at the centre frequency ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for name, field in fields.items():
    # Slice to one (frequency, range) cell, leaving a 1-D vector over depth.
    cut = field.at(frequency=source.frequencies[0], range=TARGET_RANGE).to_dB()
    ax.plot(np.asarray(cut.dB).ravel(), field.depths, label=name, linewidth=1.5)
ax.set_xlabel('Transmission loss (dB)')
ax.set_ylabel('Depth (m)')
ax.set_title(f'TL vs depth at {source.frequencies[0]:.0f} Hz, '
             f'{TARGET_RANGE / 1000:.0f} km')
ax.invert_yaxis()
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'example_19_tl_depth_comparison.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

# ── Every model as a time trace at the same receiver ────────────────────────
arrival_s = TARGET_RANGE / 1500.0
t_start = max(0.0, arrival_s - 0.5)          # half a second of lead-in
traces = {}
for name, field in fields.items():
    trace = field.to_time_trace(depth=TARGET_DEPTH, range=TARGET_RANGE,
                                t_start=t_start)
    traces[name] = (trace.times * 1000, trace.data)
traces['Bellhop (chirp)'] = (
    chirp_response.times * 1000,
    chirp_response.at(depth=TARGET_DEPTH, range=TARGET_RANGE).data)
traces['SPARC'] = (
    sparc.times * 1000,
    sparc.at(depth=float(sparc.depths[0]), range=float(sparc.ranges[-1])).data)
print(f"  first arrival at {arrival_s * 1e3:.0f} ms; {len(traces)} traces")

fig, axes = plt.subplots(len(traces), 1, figsize=(14, 2.5 * len(traces)),
                         squeeze=False)
window = (arrival_s * 1000 - 250, arrival_s * 1000 + 250)
for ax, (name, (times_ms, values)) in zip(axes[:, 0], traces.items()):
    ax.plot(times_ms, values, color=f'C{list(traces).index(name) % 10}', lw=0.8)
    ax.set_ylabel(name, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if name != 'Bellhop (chirp)':     # the chirp trace has its own short axis
        ax.set_xlim(*window)
    ax.text(0.98, 0.92, f'max = {np.max(np.abs(values)):.2e}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            color='gray',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
axes[-1, 0].set_xlabel('Time (ms)')
fig.suptitle(f'Time series — depth {TARGET_DEPTH:.0f} m, range '
             f'{TARGET_RANGE / 1000:.0f} km, fc '
             f'{source.frequencies[0]:.0f} Hz', fontsize=13)
fig.tight_layout()
fig.savefig(OUT / 'example_19_time_series_comparison.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

# ── The chirp in and the chirp out ──────────────────────────────────────────
fig, (ax_tx, ax_rx) = plt.subplots(2, 1, figsize=(12, 6))
ax_tx.plot(t_chirp * 1000, chirp, 'k-', linewidth=0.8)
ax_tx.set_title(f'Source: LFM chirp 50-150 Hz, '
                f'{chirp_duration * 1000:.0f} ms')
ax_rx.plot(*traces['Bellhop (chirp)'], color='b', linewidth=0.8)
ax_rx.set_title(f'Received: Bellhop delay-and-sum at {TARGET_DEPTH:.0f} m, '
                f'{TARGET_RANGE / 1000:.0f} km')
for ax in (ax_tx, ax_rx):
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'example_19_bellhop_chirp.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ── SPARC's own waterfall, one trace per range ──────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
times_ms = sparc.times * 1000
for index, trace in enumerate(sparc.data[0]):          # depth 0 → (n_r, n_t)
    normalised = trace / (np.max(np.abs(trace)) + 1e-30) * 0.8
    ax.plot(times_ms, normalised + index, 'k-', linewidth=0.7)
    ax.text(times_ms[-1] * 1.01, index,
            f'{sparc_receiver.ranges[index] / 1000:.1f} km', fontsize=8,
            va='center')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Range (trace index)')
ax.set_title('SPARC time-domain waveforms')
ax.set_xlim(times_ms[0], times_ms[-1])
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'example_19_sparc_time_series.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
