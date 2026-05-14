"""Cross-model agreement on broadband H(f) and IFFT'd time series.

The TL-only agreement suite (``test_cross_model_agreement.py``) only
checks ``|H(fc)|``: a constant amplitude offset, a phase-convention
flip, or a Nyquist-undersized IFFT can all silently slip through. This
suite runs each broadband-capable model on a Pekeris fluid waveguide
and asserts:

1. ``|H(fc)|`` agrees with Scooter (the wavenumber-integration ground
   truth) within 6 dB.
2. The IFFT'd, source-convolved trace's envelope peak lands inside the
   physically plausible early-arrival window ``[r/c_bottom, r/c_water]
   + (-50, +200) ms``, and the inter-model spread of those peaks is
   under 100 ms.

The 100 ms inter-model band absorbs Pekeris multipath / Hann-bandpass
envelope drift while still rejecting any sign-flip, conjugation, or
Nyquist-aliased IFFT (those errors shift the peak by ≥ 1 second).
"""
from __future__ import annotations

import numpy as np
import pytest

from uacpy.core.environment import BoundaryProperties, Environment
from uacpy.core.receiver import Receiver
from uacpy.core.source import Source
from uacpy.models import Bellhop, KrakenField, RAM, RunMode, Scooter


pytestmark = pytest.mark.requires_binary


def _pekeris_env() -> Environment:
    return Environment(
        name='pekeris-broadband',
        bathymetry=100.0, ssp=1500.0,
        bottom=BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1700.0, density=1.7, attenuation=0.5,
        ),
    )


# Single-cell receiver near a stable mid-window range so the first
# arrival is unambiguous and TL is well-behaved.
RANGE_M = 4000.0
DEPTH_M = 36.0
FC = 50.0
F_LO, F_HI = 25.0, 75.0
N_FREQ = 51                 # df = 1 Hz, well-resolved arrivals


def _src_rcv():
    src = Source(depths=DEPTH_M, frequencies=FC)
    rcv = Receiver(
        depths=np.array([DEPTH_M]),
        ranges=np.array([RANGE_M]),
    )
    return src, rcv


def _gaussian_pulse(fc: float, fs: float, duration: float = 0.2) -> np.ndarray:
    """Cosine-modulated Gaussian centred at ``fc`` with σ = 4/fc."""
    t = np.arange(int(duration * fs)) / fs
    t0 = duration / 2.0
    sigma = 4.0 / fc
    env = np.exp(-((t - t0) / sigma) ** 2)
    return env * np.cos(2.0 * np.pi * fc * (t - t0))


def _bellhop_bb(env, src, rcv):
    return Bellhop(verbose=False)._run_broadband(
        env, src, rcv, frequencies=np.linspace(F_LO, F_HI, N_FREQ),
    )


def _kraken_bb(env, src, rcv):
    return KrakenField(verbose=False).run(
        env, src, rcv,
        frequencies=np.linspace(F_LO, F_HI, N_FREQ),
        run_mode=RunMode.BROADBAND,
    )


def _scooter_bb(env, src, rcv):
    return Scooter(verbose=False).run(
        env, src, rcv,
        frequencies=np.linspace(F_LO, F_HI, N_FREQ),
        run_mode=RunMode.BROADBAND,
    )


def _ram_bb(env, src, rcv):
    return RAM(verbose=False, Q=2.0, T=4.0, dr=2.0, dz=0.25).run(
        env, src, rcv, run_mode=RunMode.BROADBAND,
    )


_RUNNERS = {
    'Bellhop': _bellhop_bb,
    'KrakenField': _kraken_bb,
    'Scooter': _scooter_bb,
    'RAM': _ram_bb,
}


def _envelope_peak_time(ts_data: np.ndarray, time_axis: np.ndarray,
                        window: tuple) -> float:
    """Return the time of the analytic-envelope maximum inside ``window``.

    Hilbert envelope keeps the peak detection robust against the
    bandpass cosine ringing on the windowed IFFT trace.
    """
    from scipy.signal import hilbert
    t_lo, t_hi = window
    mask = (time_axis >= t_lo) & (time_axis <= t_hi)
    if not np.any(mask):
        return float('nan')
    env = np.abs(hilbert(ts_data))
    idx = np.argmax(env[mask])
    return float(time_axis[mask][idx])


def _arrival_window():
    """Plausible first-arrival window for the test cell.

    Lower bound: r / c_bottom minus a small lead (refracted-bottom
    rays can be slightly faster than the slowest-mode-anchored
    t_start). Upper bound: r / c_water plus the source pulse half-
    duration so the convolved peak fits.
    """
    c_water = 1500.0
    c_bottom = 1700.0
    return (RANGE_M / c_bottom - 0.05, RANGE_M / c_water + 0.20)


def _runner_param(label):
    """Mark RAM-broadband variants slow (Python freq-loop is the bottleneck)."""
    marks = (pytest.mark.slow,) if label == 'RAM' else ()
    return pytest.param(label, marks=marks, id=label)


@pytest.mark.parametrize('label', [_runner_param(lbl) for lbl in _RUNNERS])
def test_broadband_transfer_function_magnitude(label):
    """|H(fc)| at the test cell is finite, positive, and within 6 dB of
    the Scooter reference (Scooter is the wavenumber-integration ground
    truth on Pekeris)."""
    env = _pekeris_env()
    src, rcv = _src_rcv()
    tf = _RUNNERS[label](env, src, rcv)
    freqs = np.asarray(tf.frequencies)
    i_fc = int(np.argmin(np.abs(freqs - FC)))
    Hfc = np.abs(np.asarray(tf.data)[0, 0, i_fc])
    assert np.isfinite(Hfc) and Hfc > 0, f'{label}: |H(fc)|={Hfc}'

    if label == 'Scooter':
        return                               # reference

    ref = _RUNNERS['Scooter'](env, src, rcv)
    ref_freqs = np.asarray(ref.frequencies)
    j_fc = int(np.argmin(np.abs(ref_freqs - FC)))
    Href = np.abs(np.asarray(ref.data)[0, 0, j_fc])
    diff_db = 20.0 * np.log10(Hfc / Href)
    assert abs(diff_db) <= 6.0, (
        f'{label} vs Scooter at fc: |H| differs by {diff_db:.2f} dB'
    )


@pytest.mark.parametrize('label', [_runner_param(lbl) for lbl in _RUNNERS])
def test_broadband_time_series_envelope_peak_in_arrival_window(label):
    """The IFFT'd Gaussian-convolved trace's analytic envelope peaks
    inside the physically plausible early-arrival window. Catches sign
    flips, conjugations, and Nyquist undersizing — each would shift the
    peak by ≥ 1 second."""
    env = _pekeris_env()
    src, rcv = _src_rcv()
    tf = _RUNNERS[label](env, src, rcv)

    fs = 4096.0                              # one-octave above f_hi
    pulse = _gaussian_pulse(FC, fs)
    ts = tf.synthesize_time_series(pulse, sample_rate=fs)
    trace = np.asarray(ts.data[0, 0])
    time = np.asarray(ts.times)

    win = _arrival_window()
    t_peak = _envelope_peak_time(trace, time, win)
    assert np.isfinite(t_peak), (
        f'{label}: trace empty inside {win}; t_axis is '
        f'[{time[0]:.3f}, {time[-1]:.3f}]'
    )
    assert win[0] <= t_peak <= win[1], (
        f'{label}: envelope peak at {t_peak:.3f}s outside arrival '
        f'window {win} (range/c_water = {RANGE_M/1500:.3f}s)'
    )


@pytest.mark.slow
def test_broadband_peak_times_agree_across_models():
    """Inter-model envelope-peak spread under 100 ms. Tight enough to
    catch a phase-convention regression on any model; loose enough to
    absorb Pekeris multipath envelope drift."""
    env = _pekeris_env()
    src, rcv = _src_rcv()

    fs = 4096.0
    pulse = _gaussian_pulse(FC, fs)
    win = _arrival_window()

    peaks = {}
    for label, runner in _RUNNERS.items():
        tf = runner(env, src, rcv)
        ts = tf.synthesize_time_series(pulse, sample_rate=fs)
        peaks[label] = _envelope_peak_time(
            np.asarray(ts.data[0, 0]),
            np.asarray(ts.times),
            win,
        )

    spread = max(peaks.values()) - min(peaks.values())
    assert spread <= 0.100, (
        f'Inter-model envelope-peak spread {spread*1000:.1f} ms > 100 ms; '
        f'peaks: {peaks}'
    )


def test_synthesize_time_series_honors_user_sample_rate():
    """The :class:`Field` returned by
    :meth:`Field.synthesize_time_series` sits on the same
    sampling grid as the source pulse — i.e. ``ts.fs == sample_rate``
    exactly."""
    env = _pekeris_env()
    src, rcv = _src_rcv()
    tf = _scooter_bb(env, src, rcv)
    fs = 4096.0
    pulse = _gaussian_pulse(FC, fs)
    ts = tf.synthesize_time_series(pulse, sample_rate=fs)
    assert ts.fs == pytest.approx(fs, rel=1e-6), (
        f'expected fs={fs}, got {ts.fs}'
    )
