"""Time-frequency, wavenumber and slowness transforms.

Six transforms, each on the signal it suits:

* f-k — two plane waves in a space-time wavefield, separated by their apparent
  speed, with the 1500 m/s acoustic cone drawn over them.
* τ-p slant stack — two linear events focusing to points in (p, τ).
* Morlet CWT and Wigner-Ville — the same two-component transient, the second
  sharper but carrying a cross-term the first does not.
* Cepstrum — an echo delay recovered as a quefrency peak.
* Hyperbolic Radon — a curved (NMO) event focusing to its velocity.

Uses: acoustic_signal.ricker_wavelet(delay=) · fk_transform · taup_transform ·
cwt · wigner_ville · cepstrum · radon_transform · plot_fk(sound_speed=) ·
plot_taup · plot_cwt · plot_wigner_ville · plot_cepstrum · plot_radon
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustic_signal import (cepstrum, cwt, fk_transform,
                                   radon_transform, taup_transform,
                                   wigner_ville)
from uacpy.acoustic_signal.waveforms import ricker_wavelet

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

fs = 2000.0


# (A) Two plane waves at 1500 and 2500 m/s, tapered, for the f-k transform.
n_times, n_channels, dx = 256, 48, 1.5
t_fk = np.arange(n_times) / fs
x_fk = np.arange(n_channels) * dx
f0 = 200.0
wavefield = (
    np.cos(2 * np.pi * (f0 * t_fk[:, None] - (f0 / 1500.0) * x_fk[None, :]))
    + 0.8 * np.cos(2 * np.pi * (f0 * t_fk[:, None]
                                + (f0 / 2500.0) * x_fk[None, :])))
wavefield *= np.hanning(n_times)[:, None] * np.hanning(n_channels)[None, :]
fk_f, fk_k, fk_power, _ = fk_transform(wavefield, fs, dx)

# (B) Two linear events in a pulsed gather, for the τ-p slant stack.
gather_fs, gather_nt, gather_nx, gather_dx = 1000.0, 512, 48, 10.0
t_gather = np.arange(gather_nt) / gather_fs
x_gather = np.arange(gather_nx) * gather_dx
# delay= broadcasts, so one call lays a pulse on every trace at that
# trace's own arrival time.
gather = sum(ricker_wavelet(t_gather[:, None], 40.0,
                            delay=tau + slowness * x_gather[None, :])
             for slowness, tau in [(1 / 1800.0, 0.04), (-1 / 2500.0, 0.13)])
slownesses, taus, slant_stack = taup_transform(
    gather, gather_fs, gather_dx, p_max=1 / 1200.0, n_slowness=301)

# (C, D) One transient with two components, for the CWT and Wigner-Ville.
t = np.arange(1024) / fs
transient = (
    np.sin(2 * np.pi * 150 * t) * np.exp(-0.5 * ((t - 0.20) / 0.03) ** 2)
    + np.sin(2 * np.pi * 500 * t) * np.exp(-0.5 * ((t - 0.40) / 0.03) ** 2))
cwt_frequencies, scalogram = cwt(transient, fs, wavelet='morlet', n_freqs=140)
wv_f, wv_t, wigner = wigner_ville(transient, fs)

# (E) A broadband pulse and its echo, for the cepstrum.
pulse = (np.exp(-((t - 0.06) / 0.004) ** 2)
         * np.cos(2 * np.pi * 250.0 * (t - 0.06)))
echo_delay = 0.040
shift = int(echo_delay * fs)
with_echo = pulse.copy()
with_echo[shift:] += 0.7 * pulse[:-shift]
cepstral = cepstrum(with_echo)
quefrency = np.arange(cepstral.size) / fs
# Search 8-100 ms of quefrency: below ~8 ms the cepstrum is dominated by the
# pulse's own spectral envelope (its 4 ms Gaussian width), which would outrank
# the echo rahmonic the peak search is after.
low, high = int(0.008 * fs), int(0.1 * fs)

# (F) A hyperbolic (NMO) event, for the Radon transform.
true_velocity = 1500.0
hyperbolic = ricker_wavelet(
    t_gather[:, None], 40.0,
    delay=np.sqrt(0.07 ** 2 + (x_gather[None, :] / true_velocity) ** 2))
velocities = np.linspace(1200.0, 2000.0, 121)
_, radon_taus, radon = radon_transform(hyperbolic, gather_fs, gather_dx,
                                       velocities, kind='hyperbolic')

# slant_stack and radon are (slowness/velocity, τ); a flat argmax floor-divided
# by the column count recovers the first-axis index, i.e. the winning p or v.
print(f"  τ-p strongest event ≈ "
      f"{1 / abs(slownesses[np.argmax(np.abs(slant_stack)) // slant_stack.shape[1]]):.0f} m/s")
print(f"  cepstrum peak {quefrency[low + np.argmax(cepstral[low:high]) ] * 1e3:.1f} ms "
      f"(echo {echo_delay * 1e3:.0f} ms)")
print(f"  hyperbolic Radon focus "
      f"{velocities[np.argmax(np.abs(radon)) // radon.shape[1]]:.0f} m/s "
      f"(true {true_velocity:.0f})")

fig, axes = plt.subplots(3, 2, figsize=(12, 14), constrained_layout=True)

# The gather is synthetic and unit-less, so the f-k power is shown relative to
# its own maximum (ref=1 → 10·log10(p/p_max)).
uacpy.plot.plot_fk(fk_f, fk_k, fk_power / fk_power.max(), ax=axes[0, 0],
                   ref=1.0, vmin=-40, vmax=0, cmap='jet', sound_speed=1500,
                   title='f-k transform + 1500 m/s cone')
axes[0, 0].images[0].colorbar.set_label('Relative power (dB)')
axes[0, 0].set_ylim(0, 400)

uacpy.plot.plot_taup(slownesses, taus, slant_stack, ax=axes[0, 1],
                     sound_speed=1500,
                     title='τ-p slant stack + 1500 m/s')

uacpy.plot.plot_cwt(cwt_frequencies, scalogram, fs, ax=axes[1, 0],
                    title='Morlet CWT scalogram')
axes[1, 0].set_yscale('log')

uacpy.plot.plot_wigner_ville(wv_f, wv_t, wigner, ax=axes[1, 1], vmin=0.0,
                             vmax=wigner.max(),
                             title='Wigner-Ville (cross-term near 325 Hz)')
axes[1, 1].set_ylim(0, 700)

uacpy.plot.plot_cepstrum(cepstral, ax=axes[2, 0], sample_rate=fs,
                         color='#1f77b4', lw=1.2,
                         title='Cepstrum — echo-delay recovery')
axes[2, 0].axvline(echo_delay, color='crimson', ls='--',
                   label=f'echo {echo_delay * 1e3:.0f} ms')
axes[2, 0].axvline(2 * echo_delay, color='crimson', ls=':', alpha=0.6,
                   label=f'2× ({2 * echo_delay * 1e3:.0f} ms)')
axes[2, 0].set_xlim(low / fs, high / fs)
# The value axis follows the search band, not the pulse's own envelope at
# quefrency 0, which is a hundred times the echo rahmonic.
axes[2, 0].set_ylim(1.2 * cepstral[low:high].min(),
                    1.2 * cepstral[low:high].max())
axes[2, 0].legend(loc='upper right')

uacpy.plot.plot_radon(velocities, radon_taus, radon, ax=axes[2, 1],
                      kind='hyperbolic', title='Hyperbolic Radon (velocity)')
axes[2, 1].axvline(true_velocity, color='w', ls='--', lw=1.1)

fig.suptitle('Time-frequency, wavenumber and slowness transforms',
             fontsize=15, fontweight='bold')
fig.savefig(OUT / 'example_30_time_frequency.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
