"""Figures for ``docs/guide/signal.md``.

The code here is the code shown on the page: each builder is the worked
example, so a figure cannot drift from the snippet that claims to produce it.

Where a modelled signal is cheap it is used in preference to a synthetic one —
the matched-filter, f-k and warping figures all run a propagation model first,
so the multipath and the dispersion are real rather than drawn. The
time-frequency figures keep controlled synthetic inputs, because the point
being made there is about the *estimator*, and it needs a known answer to be
measured against.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from figure_scripts._common import WIDE, shallow_water

import uacpy
from uacpy.acoustic_signal import (
    add_noise,
    ambiguity_function,
    analytic_signal,
    cwt,
    decidecade_band_levels,
    envelope,
    fk_transform,
    gaussian_pulse,
    hfm_chirp,
    impulse_response_from_transfer_function,
    instantaneous_frequency,
    inverse_fk,
    lfm_chirp,
    modal_group_velocity,
    mseq,
    nwave,
    processing_gain,
    psd,
    pulse_compression,
    ricker_wavelet,
    simulate_reception,
    sparc_pulse,
    spectrogram,
    synthesize_noise_from_psd,
    tone_burst,
    warp_signal,
    wigner_ville,
)
from uacpy.models import Bellhop, Kraken, RunMode
from uacpy.visualization import draw_sound_cone, plot_band_levels, plot_psd

# Write into docs/guide/figures/ rather than docs/models/figures/.
GUIDE = True


def _db(x, floor=-60.0):
    """Amplitude (real or complex) -> dB re peak, clipped at ``floor``."""
    a = np.abs(np.asarray(x))
    peak = a.max()
    with np.errstate(divide='ignore'):
        d = 20.0 * np.log10(a / peak)
    return np.maximum(d, floor)


def _power_db(p, floor=-60.0):
    """Power -> dB relative to the peak, clipped at ``floor``."""
    a = np.abs(np.asarray(p))
    with np.errstate(divide='ignore'):
        d = 10.0 * np.log10(a / a.max())
    return np.maximum(d, floor)


# ── 1. waveform catalogue ────────────────────────────────────────────────────


def waveform_gallery():
    """Every deterministic generator in the package, drawn at one glance."""
    fs = 8000.0
    t = np.arange(int(0.10 * fs)) / fs

    lfm, t_lfm = lfm_chirp(200.0, 1200.0, 0.10, fs)
    hfm, t_hfm = hfm_chirp(200.0, 1200.0, 0.10, fs)
    burst, t_burst = tone_burst(400.0, 8, fs)
    ricker = ricker_wavelet(t, 200.0)
    gauss = gaussian_pulse(t, delay=0.05, duration=0.012)
    nw = nwave(t - 0.02, 200.0)
    hann4, hann4_title = sparc_pulse(t - 0.02, 2 * np.pi * 200.0, 'H')
    chips = mseq(6)

    panels = [
        (t_lfm, lfm, 'lfm_chirp(200, 1200, 0.1, fs)'),
        (t_hfm, hfm, 'hfm_chirp(200, 1200, 0.1, fs)'),
        (t_burst, burst, 'tone_burst(400, 8, fs)'),
        (t, ricker, 'ricker_wavelet(t, 200)'),
        (t, gauss, 'gaussian_pulse(t, 0.05, 0.012)'),
        (t, nw, 'nwave(t - 0.02, 200)'),
        (t, hann4, f"sparc_pulse(t - 0.02, w, 'H') — {hann4_title}"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 5.2))
    for ax, (tt, sig, label) in zip(axes.ravel(), panels):
        ax.plot(tt * 1e3, sig, lw=0.9, color='#1f4e79')
        ax.set_title(label, fontsize=8.5, fontweight='bold')
        ax.set_xlabel('Time (ms)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    ax = axes.ravel()[-1]
    ax.step(np.arange(chips.size), chips, where='mid', lw=0.9, color='#8c2d04')
    ax.set_title('mseq(6) — 63 chips, ±1', fontsize=8.5, fontweight='bold')
    ax.set_xlabel('Chip index', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_ylim(-1.6, 1.6)
    ax.grid(alpha=0.3)
    fig.suptitle('Source waveforms — one call each',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


# ── 2. the two chirps in time-frequency ──────────────────────────────────────


def chirp_sweep_laws():
    """LFM and HFM side by side: the sweep law each one obeys."""
    fs = 8000.0
    lfm, t_lfm = lfm_chirp(200.0, 1600.0, 0.20, fs)
    hfm, t_hfm = hfm_chirp(200.0, 1600.0, 0.20, fs)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for ax, sig, tt, name in [(axes[0], lfm, t_lfm, 'lfm_chirp — linear sweep'),
                              (axes[1], hfm, t_hfm, 'hfm_chirp — hyperbolic sweep')]:
        f, t_spec, Sxx = spectrogram(sig, fs, nperseg=256, noverlap=240)
        ax.set_facecolor('k')
        ax.pcolormesh(t_spec * 1e3, f, _power_db(Sxx, floor=-35.0),
                      shading='auto', cmap='magma', vmin=-35.0, vmax=0.0)
        f_inst = instantaneous_frequency(sig, fs)
        keep = slice(int(0.015 * fs), -int(0.015 * fs))
        ax.plot(tt[keep] * 1e3, f_inst[keep], color='#7fffd4', lw=1.4,
                label='instantaneous_frequency')
        ax.set_ylim(0.0, 2200.0)
        ax.set_xlabel('Time (ms)')
        ax.set_title(name, fontweight='bold', fontsize=11)
        ax.legend(loc='upper left', fontsize=8)
    axes[0].set_ylabel('Frequency (Hz)')
    fig.suptitle('Two chirps over the same 200–1600 Hz band',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


# ── 3. matched filtering through a modelled channel ──────────────────────────


def _bellhop_arrivals():
    """Bellhop arrival structure at 3 km / 60 m in the shallow-water channel."""
    env, _, _ = shallow_water()
    source = uacpy.Source(depths=25.0, frequencies=3000.0)
    point = uacpy.Receiver(depths=60.0, ranges=3000.0)
    return Bellhop(n_beams=6000, alpha=(-60.0, 60.0)).run(
        env, source, point, run_mode=RunMode.ARRIVALS)


def pulse_compression_gain():
    """Compress an LFM chirp back out of a modelled multipath channel."""
    rng = np.random.default_rng(0)
    fs = 20_000.0
    fmin, fmax, T = 1000.0, 5000.0, 0.05
    tx, _ = lfm_chirp(fmin, fmax, T, fs)

    arr = _bellhop_arrivals()
    taps = arr.amplitudes * np.exp(1j * arr.phases)
    _, rx = simulate_reception(analytic_signal(tx), taps, arr.delays, fs)
    rx = np.real(rx)
    noisy = add_noise(rx, fs, source_level=180.0, noise_level=72.0,
                      fc=3000.0, bandwidth=4000.0, rng=rng)
    t_rx = np.arange(noisy.size) / fs

    lags, comp = pulse_compression(analytic_signal(noisy), analytic_signal(tx), fs)
    gain = processing_gain(fmax - fmin, T)
    span = (arr.delays.min(), arr.delays.max())

    # Input SNR, measured rather than asserted: signal RMS over the arrival
    # window against the RMS of the noise that add_noise actually drew.
    clean = rx * 10.0 ** (180.0 / 20.0)
    inside = (t_rx >= span[0]) & (t_rx <= span[1])
    snr_db = 20.0 * np.log10(np.sqrt(np.mean(clean[inside] ** 2))
                             / np.std(noisy - clean))

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4))
    win = (t_rx > 1.85) & (t_rx < 2.45)
    axes[0].plot(t_rx[win], noisy[win] / 1e6, lw=0.4, color='#9aa5b1')
    axes[0].plot(t_rx[win], envelope(noisy)[win] / 1e6, lw=0.7, color='#c0392b',
                 label='envelope()')
    axes[0].set_title(f'Received — a 50 ms chirp smeared over '
                      f'{1e3 * (span[1] - span[0]):.0f} ms of Bellhop '
                      f'multipath, at {snr_db:+.0f} dB in-band SNR',
                      fontweight='bold', fontsize=10)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Pressure (Pa)')

    keep = (lags > 1.85) & (lags < 2.45)
    axes[1].plot(lags[keep], _db(comp[keep], floor=-40.0), lw=0.6,
                 color='#1f4e79')
    axes[1].axvspan(*span, color='#f0ad4e', alpha=0.18,
                    label='Bellhop arrival window')
    axes[1].set_title(f'Matched-filter output — the channel impulse response, '
                      f'resolved to 1/B = {1e3 / (fmax - fmin):.2f} ms '
                      f'(processing gain 10·log10(BT) = {gain:.0f} dB)',
                      fontweight='bold', fontsize=10)
    axes[1].set_xlabel('Delay (s)')
    axes[1].set_ylabel('Level (dB re peak)')
    axes[1].set_ylim(-40.0, 3.0)
    for ax in axes:
        ax.set_xlim(1.85, 2.45)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle('Pulse compression against a modelled channel',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


# ── 4. waveform resolution: the ambiguity surface ────────────────────────────


def ambiguity_surfaces():
    """LFM against HFM: range-Doppler coupling versus Doppler tolerance."""
    fs = 20_000.0
    fmin, fmax, T = 1000.0, 5000.0, 0.05
    lfm, _ = lfm_chirp(fmin, fmax, T, fs)
    hfm, _ = hfm_chirp(fmin, fmax, T, fs)
    doppler = np.linspace(-300.0, 300.0, 121)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for ax, sig, name in [(axes[0], lfm, 'lfm_chirp'), (axes[1], hfm, 'hfm_chirp')]:
        delays, dop, chi = ambiguity_function(
            analytic_signal(sig), fs, doppler_hz=doppler)
        im = ax.imshow(_db(chi, floor=-30.0), aspect='auto', origin='lower',
                       extent=[delays[0] * 1e3, delays[-1] * 1e3,
                               dop[0], dop[-1]],
                       vmin=-30.0, vmax=0.0, cmap='viridis')
        ax.set_xlim(-6.0, 6.0)
        ax.set_xlabel('Delay τ (ms)')
        ax.set_title(f'{name} — |χ(τ, ν)|', fontweight='bold', fontsize=11)
    axes[0].set_ylabel('Doppler ν (Hz)')
    fig.colorbar(im, ax=axes, label='dB re peak')
    fig.suptitle('Ambiguity surfaces of the same 4 kHz band, 50 ms pulse',
                 fontweight='bold', fontsize=13)
    return fig


# ── 5. the time-frequency resolution trade-off ───────────────────────────────


def _two_tones_and_a_click(fs=2000.0, n=2048):
    """Two close tones burst at one time, plus a short 500 Hz click later.

    Deliberately synthetic: the point is to measure an estimator against a
    known answer. The tones are 30 Hz apart so a short window provably cannot
    separate them; the click is three cycles long so a long window provably
    cannot place it.
    """
    t = np.arange(n) / fs
    tones = (np.sin(2 * np.pi * 100.0 * t) + np.sin(2 * np.pi * 130.0 * t))
    tones = tones * np.exp(-0.5 * ((t - 0.35) / 0.13) ** 2)
    click, _ = tone_burst(500.0, 3, fs)
    signal = tones
    i0 = int(0.72 * fs)
    signal[i0:i0 + click.size] += 2.0 * click
    return t, signal


def resolution_tradeoff():
    """The same signal under a short window, a long window, and a wavelet."""
    fs = 2000.0
    t, sig = _two_tones_and_a_click(fs)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharey=True)
    for ax, nperseg in [(axes[0], 64), (axes[1], 512)]:
        f, t_spec, Sxx = spectrogram(sig, fs, nperseg=nperseg,
                                     noverlap=nperseg - 8)
        ax.pcolormesh(t_spec, f, _power_db(Sxx, floor=-40.0), shading='auto',
                      cmap='magma', vmin=-40.0, vmax=0.0)
        ax.set_title(f'spectrogram(nperseg={nperseg}) — '
                     f'Δf ≈ {fs / nperseg:.0f} Hz, Δt = {nperseg / fs * 1e3:.0f} ms',
                     fontweight='bold', fontsize=10)
        ax.set_xlabel('Time (s)')
    freqs, W = cwt(sig, fs, frequencies=np.logspace(np.log10(20.0),
                                                    np.log10(900.0), 160))
    axes[2].pcolormesh(t, freqs, _db(W, floor=-40.0), shading='auto',
                       cmap='magma', vmin=-40.0, vmax=0.0)
    axes[2].set_title('cwt(wavelet="morlet") — Δf/f constant',
                      fontweight='bold', fontsize=10)
    axes[2].set_xlabel('Time (s)')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_ylim(0.0, 900.0)
    for ax in axes:
        ax.set_facecolor('k')
        ax.set_xlim(0.0, t[-1])
        ax.axhline(100.0, color='#7fffd4', lw=0.6, ls=':')
        ax.axhline(130.0, color='#7fffd4', lw=0.6, ls=':')
        ax.axvline(0.72, color='#7fffd4', lw=0.6, ls=':')
    fig.suptitle('Two tones 30 Hz apart at t = 0.35 s, a three-cycle 500 Hz '
                 'burst at t = 0.72 s', fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


# ── 6. Wigner-Ville and its cross-terms ──────────────────────────────────────


def wigner_ville_cross_terms():
    """The interference term is real, and smoothing is what trades it away."""
    fs = 2000.0
    n = 512
    t = np.arange(n) / fs
    sig = (np.sin(2 * np.pi * 150.0 * t) * np.exp(-0.5 * ((t - 0.07) / 0.018) ** 2)
           + np.sin(2 * np.pi * 500.0 * t) * np.exp(-0.5 * ((t - 0.18) / 0.018) ** 2))

    f_s, t_s, Sxx = spectrogram(sig, fs, nperseg=64, noverlap=60)
    f_w, t_w, W = wigner_ville(sig, fs)
    f_p, t_p, P = wigner_ville(sig, fs, freq_window=63, time_window=25)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharey=True)
    axes[0].pcolormesh(t_s, f_s, _power_db(Sxx, floor=-35.0), shading='auto',
                       cmap='magma', vmin=-35.0, vmax=0.0)
    axes[0].set_title('spectrogram — no cross-terms, blurred',
                      fontweight='bold', fontsize=10)
    for ax, ff, tt, D, name in [
            (axes[1], f_w, t_w, W, 'wigner_ville() — sharp, with a cross-term'),
            (axes[2], f_p, t_p, P,
             'wigner_ville(freq_window=63, time_window=25)')]:
        ax.pcolormesh(tt, ff, _power_db(np.maximum(D, 0.0), floor=-35.0),
                      shading='auto', cmap='magma', vmin=-35.0, vmax=0.0)
        ax.set_title(name, fontweight='bold', fontsize=10)
    for ax in axes:
        ax.set_xlabel('Time (s)')
        ax.set_ylim(0.0, 700.0)
        ax.axhline(325.0, color='#7fffd4', lw=0.7, ls=':')
        ax.axvline(0.125, color='#7fffd4', lw=0.7, ls=':')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[1].annotate('cross-term', xy=(0.125, 325.0), xytext=(0.20, 430.0),
                     color='w', fontsize=9,
                     arrowprops=dict(arrowstyle='->', color='w', lw=1.0))
    fig.suptitle('Two atoms at (0.07 s, 150 Hz) and (0.18 s, 500 Hz); the '
                 'dotted cross marks their midpoint',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


# ── 7. f-k filtering of a modelled array gather ──────────────────────────────


def _vertical_array_gather():
    """Bellhop time series on a 64-element vertical array at 1 km.

    Returns ``(gather, times, depths, sample_rate)`` with the gather shaped
    ``(n_time, n_depth)`` — the ``(nt, nx)`` layout the gather transforms take.
    """
    env, _, _ = shallow_water()
    fs = 4000.0
    pulse = ricker_wavelet(np.arange(int(0.05 * fs)) / fs, 250.0)
    depths = np.linspace(4.0, 96.0, 64)
    source = uacpy.Source(depths=25.0, frequencies=250.0)
    receiver = uacpy.Receiver(depths=depths, ranges=1000.0)
    field = Bellhop(n_beams=6000, alpha=(-80.0, 80.0)).run(
        env, source, receiver, run_mode=RunMode.TIME_SERIES,
        source_waveform=pulse, sample_rate=fs, output_duration=0.85)
    times = np.asarray(field.coords['time'])
    gather = np.real(np.asarray(field.data))[:, 0, :].T
    keep = (times >= 0.62) & (times <= 0.82)
    return gather[keep], times[keep], depths, fs


def fk_round_trip():
    """Forward f-k, mute the up-going quadrants, invert — the whole point of
    keeping the inverse a standalone function on coefficients."""
    gather, times, depths, fs = _vertical_array_gather()
    dz = float(depths[1] - depths[0])

    frequencies, wavenumbers, power, spectrum = fk_transform(gather, fs, dz)
    kk, ff = np.meshgrid(wavenumbers, frequencies)
    # A linear event t = t0 + p*z maps to k = -2*pi*f*p, so the down-going half
    # (p > 0, later at greater depth) is k < 0 for f > 0; muting the complement
    # keeps it.
    mask = np.ones_like(power)
    mask[(ff > 0) & (kk > 0)] = 0.0
    mask[(ff < 0) & (kk < 0)] = 0.0
    down = inverse_fk(spectrum * mask)

    def draw(ax, data, title):
        v = 0.3 * np.abs(gather).max()
        ax.imshow(data, aspect='auto', origin='upper', cmap='RdBu_r',
                  extent=[depths[0], depths[-1], times[-1], times[0]],
                  vmin=-v, vmax=v)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xlabel('Receiver depth (m)')
        ax.set_ylabel('Time (s)')

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    draw(axes[0, 0], gather, 'Bellhop gather — 64-element vertical array, 1 km')
    ax = axes[0, 1]
    ax.imshow(_power_db(power, floor=-40.0), aspect='auto', origin='lower',
              extent=[wavenumbers[0], wavenumbers[-1],
                      frequencies[0], frequencies[-1]],
              vmin=-40.0, vmax=0.0, cmap='magma')
    draw_sound_cone(ax, frequencies[-1], wavenumbers[-1], 1500.0, color='#7fffd4')
    ax.axvspan(0.0, wavenumbers[-1], facecolor='w', alpha=0.22, hatch='//',
               edgecolor='w')
    ax.set_ylim(0.0, 600.0)
    ax.set_xlabel('Vertical wavenumber k (rad/m)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('f-k power (f > 0 half) — hatched quadrant is muted',
                 fontweight='bold', fontsize=10)
    draw(axes[1, 0], down, 'inverse_fk(spectrum · mask) — down-going only')
    draw(axes[1, 1], gather - down, 'Residual — the up-going wavefield')
    fig.suptitle('Filtering between the forward transform and its inverse',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


# ── 8. modal dispersion and warping ──────────────────────────────────────────


def _ideal_waveguide():
    """100 m isovelocity channel with a perfectly rigid seabed.

    The ideal waveguide warping is derived for: mode ``m`` cuts on at
    ``(2m − 1)·c / 4D`` and there is no bottom loss, so every mode survives to
    the receiver.
    """
    return uacpy.Environment(
        name='Ideal waveguide',
        bathymetry=100.0,
        ssp=[(0.0, 1500.0), (100.0, 1500.0)],
        bottom=uacpy.BoundaryProperties(acoustic_type='rigid'),
    )


def modal_warping():
    """Kraken dispersion, the transient it produces, and the warp that undoes it."""
    env = _ideal_waveguide()
    c, depth, range_m = 1500.0, 100.0, 5000.0
    cutoffs = [(2 * m - 1) * c / (4.0 * depth) for m in range(1, 8)]

    sweep = np.linspace(30.0, 70.0, 40)
    kr = [Kraken().compute_modes(env, uacpy.Source(depths=20.0,
                                                   frequencies=float(f))).k
          for f in sweep]
    n_modes = min(len(k) for k in kr)
    k_matrix = np.real(np.array([k[:n_modes] for k in kr]))
    v_group = modal_group_velocity(sweep, k_matrix)

    fs, n = 150.0, 2048
    df = fs / n
    bins = np.arange(int(np.ceil(8.0 / df)), int(np.floor(70.0 / df)) + 1)
    frequencies = bins * df
    H = Kraken().run(env,
                     uacpy.Source(depths=20.0, frequencies=frequencies),
                     uacpy.Receiver(depths=50.0, ranges=range_m),
                     run_mode=RunMode.BROADBAND)
    _, h = impulse_response_from_transfer_function(
        np.asarray(H.data).ravel(), frequencies, fs, n)
    arrival = h[int(round(range_m / c * fs)):]
    warped, t_warp = warp_signal(arrival, fs, range_m, c)
    fs_warp = 1.0 / float(t_warp[1] - t_warp[0])

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    for m in range(n_modes):
        axes[0].plot(sweep, v_group[:, m], lw=1.4, label=f'mode {m + 1}')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Group velocity (m/s)')
    axes[0].set_title('modal_group_velocity(f, kr) from Kraken',
                      fontweight='bold', fontsize=10)
    axes[0].legend(fontsize=8, loc='lower right')
    axes[0].grid(alpha=0.3)

    f_a, t_a, S_a = spectrogram(arrival, fs, nperseg=128, noverlap=124)
    axes[1].pcolormesh(t_a + range_m / c, f_a, _power_db(S_a, floor=-30.0),
                       shading='auto', cmap='magma', vmin=-30.0, vmax=0.0)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_ylim(0.0, 70.0)
    axes[1].set_title('Received transient — modes disperse',
                      fontweight='bold', fontsize=10)

    f_w, t_w, S_w = spectrogram(warped, fs_warp, nperseg=256, noverlap=252)
    axes[2].pcolormesh(t_w, f_w, _power_db(S_w, floor=-30.0), shading='auto',
                       cmap='magma', vmin=-30.0, vmax=0.0)
    axes[2].set_xlabel('Warped time (s)')
    axes[2].set_ylabel('Warped frequency (Hz)')
    axes[2].set_ylim(0.0, 50.0)
    axes[2].set_title('warp_signal(...) — modes become tones',
                      fontweight='bold', fontsize=10)
    for ax in (axes[1], axes[2]):
        for f_c in cutoffs:
            ax.axhline(f_c, color='#7fffd4', lw=0.6, ls=':', alpha=0.8)
    fig.suptitle('Modal dispersion in an ideal 100 m waveguide at 5 km; '
                 'dotted lines are the cutoffs (2m−1)c/4D',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


# ── 9. spectra, levels and bands ─────────────────────────────────────────────


def spectra_and_bands():
    """A synthesised soundscape, its Welch PSD, and its decidecade band levels."""
    rng = np.random.default_rng(0)
    f_target = np.logspace(np.log10(10.0), np.log10(10_000.0), 200)
    level_db = 100.0 - 17.0 * np.log10(f_target / 100.0)
    level_db += 12.0 * np.exp(-0.5 * ((np.log10(f_target / 300.0)) / 0.02) ** 2)
    target = 1e-12 * 10.0 ** (level_db / 10.0)          # Pa²/Hz

    _, x, fs = synthesize_noise_from_psd(
        target, f_target, duration=30.0, sample_rate=25_000,
        n_fft=65536, interp='log', rng=rng)
    frequencies, power = psd(x, fs, nperseg=32768)
    band = (frequencies >= 20.0) & (frequencies <= 11_000.0)
    centers, levels = decidecade_band_levels(power[band], frequencies[band])

    shown = frequencies >= 10.0
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    plot_psd(frequencies[shown], power[shown], ax=axes[0],
             label='psd() of the realisation', ymin=55, ymax=125,
             color='#1f4e79', lw=0.6,
             title='synthesize_noise_from_psd → psd')
    plot_psd(f_target, target, ax=axes[0], label='target PSD', ymin=55,
             ymax=125, color='#f0ad4e', lw=2.0, ls='--')
    plot_band_levels(centers, levels, ax=axes[1], color='#1f4e79',
                     title='decidecade_band_levels(power, frequencies)')
    density_at_center = np.interp(centers, frequencies, 10.0 * np.log10(
        np.maximum(power, 1e-30) / 1e-12))
    axes[1].plot(np.log10(centers), density_at_center, color='#f0ad4e', lw=2.0,
                 label='density level at the same frequency')
    axes[1].set_ylim(55, 125)
    axes[1].legend(fontsize=8, loc='lower left')
    fig.suptitle('Spectral density versus band level',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


FIGURES = {
    'signal_waveforms': waveform_gallery,
    'signal_chirps': chirp_sweep_laws,
    'signal_pulse_compression': pulse_compression_gain,
    'signal_ambiguity': ambiguity_surfaces,
    'signal_resolution': resolution_tradeoff,
    'signal_wigner_ville': wigner_ville_cross_terms,
    'signal_fk_filter': fk_round_trip,
    'signal_warping': modal_warping,
    'signal_levels': spectra_and_bands,
}
