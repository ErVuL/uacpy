"""Cross-model agreement on broadband H(f) and IFFT'd time series.

The TL-only agreement suite (``test_cross_model_agreement.py``) only
checks ``|H(fc)|``: a constant amplitude offset, a phase-convention
flip, or a Nyquist-undersized IFFT can all silently slip through. This
suite runs each broadband-capable model on a Pekeris fluid waveguide
and asserts:

1. ``|H(fc)|`` agrees with Scooter (the wavenumber-integration ground
   truth) within the per-model bound in ``_H_FC_TOLERANCE_DB`` — 1.5 dB
   for Kraken, 3.5 for RAM, 6.0 for Bellhop, each sized on the measured
   difference at this cell.
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
from uacpy.models import SPARC, Bellhop, Kraken, RAM, RunMode, Scooter
from uacpy.tests.conftest import make_pekeris


pytestmark = pytest.mark.requires_binary


def _pekeris_env() -> Environment:
    return make_pekeris(name='pekeris-broadband', density=1.7)


# Single-cell receiver near a stable mid-window range so the first
# arrival is unambiguous and TL is well-behaved.
RANGE_M = 4000.0
DEPTH_M = 36.0
FC = 50.0
F_LO, F_HI = 25.0, 75.0
# df = 0.5 Hz, so the synthesised record is 1/df = 2 s long. The arrival
# train at 4 km runs from ~2.35 s to past 2.89 s (Bellhop and RAM put the
# envelope maximum at 2.888-2.891 s); a 2 Hz grid gives a 0.5 s record that
# the auto window places at [2.273, 2.773] s, which ends before that
# maximum, so the envelope argmax lands on the truncated record edge.
N_FREQ = 101


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
    return Bellhop(verbose=False).run(
        env, src, rcv,
        run_mode=RunMode.BROADBAND,
        frequencies=np.linspace(F_LO, F_HI, N_FREQ),
    )


def _kraken_bb(env, src, rcv):
    return Kraken(verbose=False).run(
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
    return RAM(verbose=False, q_factor=2.0, record_duration=4.0, dr=2.0, dz=0.25).run(
        env, src, rcv, run_mode=RunMode.BROADBAND,
    )


_RUNNERS = {
    'Bellhop': _bellhop_bb,
    'Kraken': _kraken_bb,
    'Scooter': _scooter_bb,
    'RAM': _ram_bb,
}


# |H(fc)| agreement with Scooter, one bound per model rather than a single
# gate sized for the loosest of them: Kraken agrees to 0.29 dB here and a
# bound shared with Bellhop spends that sensitivity on Bellhop's account.
# Each number below is the difference measured at this cell (RANGE_M,
# DEPTH_M, FC) followed by the multiple of it the bound allows.
_H_FC_TOLERANCE_DB = {
    # Measured -0.286 dB. Modes vs wavenumber integration on a fluid Pekeris
    # is an apples-to-apples comparison, so 1.5 dB is 5.2x the measurement
    # and still catches a factor-2 amplitude convention (6.02 dB) by 3.8x in
    # the nearer of the two directions.
    'Kraken': 1.5,
    # Judged on the band median of |dH| (``_BAND_MEDIAN_JUDGED``), not at
    # fc: fc sits in a Scooter null (-76.3 dB against -64.0 / -61.6 dB at
    # fc -/+ 2 Hz), where Bellhop at D/lambda = 3.33 (100 m water column at
    # 50 Hz, below the D/lambda >= 5 ray-theory floor of uacpy's validity
    # table) reads +6.50 dB and a downward factor-2 error would land at
    # +0.48 dB. Band median measured 1.980 dB with the default hat beams
    # (1.998 / 1.910 at 2000 / 8000 beams; Gaussian beams 1.98-1.99).
    # 3.0 dB is 1.50x the worst measurement and catches a factor-2
    # amplitude convention in both directions: the band median becomes
    # 6.67 dB (x2) or 5.70 dB (/2), 1.9x outside the bound at the nearer.
    'Bellhop': 3.0,
    # Measured +1.415 dB. 3.5 dB is 2.5x it — more headroom than Kraken gets
    # because RAM is the only model here whose accuracy is set by
    # user-facing marching parameters (``_ram_bb`` pins dr=2.0, dz=0.25), so
    # this difference is a function of a grid the caller chooses and the
    # others' are not. Still catches a factor-2 (6.02 dB) by 1.3x in the
    # nearer direction.
    'RAM': 3.5,
}

# Models judged on the median |dH| over the whole band rather than at fc.
# Kraken and RAM agree with Scooter at fc to 0.29 / 1.42 dB despite the null
# there, and their fc bounds catch a factor-2 either way; Bellhop's fc value
# sits on the null.
_BAND_MEDIAN_JUDGED = {'Bellhop'}

# Every model compared against the Scooter reference needs its own bound;
# adding a runner without one must fail here rather than quietly inherit a
# neighbour's number.
assert set(_H_FC_TOLERANCE_DB) == set(_RUNNERS) - {'Scooter'}, (
    f'_H_FC_TOLERANCE_DB {sorted(_H_FC_TOLERANCE_DB)} does not cover the '
    f'non-reference runners {sorted(set(_RUNNERS) - {"Scooter"})}'
)


def test_mpirams_phase_matches_scooter():
    """Anchor the mpiramS phase convention to the exact field.

    ``_pe_phase.py`` converts mpiramS output as ``conj(psif)·4π`` (no
    ``exp(±iπ/4)``): peramx already bakes the Hankel phase into ``psif``. The
    closed-form unit tests in ``test_pe_phase.py`` only prove the helper matches
    that formula — they cannot tell a right convention from a 45°-rotated one.
    This test pins it to ground truth: the narrowband COHERENT_TL complex
    pressure from RAM (mpiramS) agrees in phase with Scooter (wavenumber
    integration) across range. An extra ``exp(±iπ/4)`` would appear as a
    constant ~45° offset and fail the gate; ``|TL|`` is blind to it.
    """
    env = _pekeris_env()
    depth = 36.0
    ranges = np.linspace(2000.0, 6000.0, 9)
    src = Source(depths=depth, frequencies=50.0)
    rcv = Receiver(depths=np.array([depth]), ranges=ranges)

    p_ram = np.asarray(
        RAM(verbose=False, dr=2.0, dz=0.25).run(
            env, src, rcv, run_mode=RunMode.COHERENT_TL).data
    ).ravel()
    p_sco = np.asarray(
        Scooter(verbose=False).run(
            env, src, rcv, run_mode=RunMode.COHERENT_TL).data
    ).ravel()

    ratio = p_ram / p_sco
    ratio = ratio[np.isfinite(ratio)]
    # circular mean of the per-range phase difference: a convention error is a
    # constant offset; per-range modal / numerical jitter averages out.
    mean_phase_deg = np.degrees(np.angle(np.mean(ratio / np.abs(ratio))))
    # 20 deg sits below the 45 deg an unwanted exp(±iπ/4) would impose and well
    # below the 180 deg of a sign flip, while leaving room for the PE's own
    # wide-angle phase error against the exact field.
    assert abs(mean_phase_deg) < 20.0


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
    t_start). Upper bound: r / c_water plus the full 0.2 s source pulse
    (convolution delays the envelope peak by up to the pulse length) plus
    the 0.1 s modal-tail allowance the inter-model spread assertion uses —
    in a Pekeris cell the envelope maximum builds from late high-order
    modes and lands after the first water-speed arrival cluster.
    """
    c_water = 1500.0
    c_bottom = 1700.0
    return (RANGE_M / c_bottom - 0.05, RANGE_M / c_water + 0.20 + 0.10)


def _runner_param(label):
    """Mark RAM-broadband variants slow (Python freq-loop is the bottleneck)."""
    marks = (pytest.mark.slow,) if label == 'RAM' else ()
    return pytest.param(label, marks=marks, id=label)


@pytest.mark.parametrize('label', [_runner_param(lbl) for lbl in _RUNNERS])
def test_broadband_transfer_function_magnitude(label):
    """|H(fc)| at the test cell is finite, positive, and within the model's
    own ``_H_FC_TOLERANCE_DB`` bound of the Scooter reference (Scooter is the
    wavenumber-integration ground truth on Pekeris) — at fc, or as the band
    median of |dH| for the models in ``_BAND_MEDIAN_JUDGED``."""
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
    tolerance_dB = _H_FC_TOLERANCE_DB[label]
    if label in _BAND_MEDIAN_JUDGED:
        np.testing.assert_allclose(freqs, ref_freqs)
        band_dB = 20.0 * np.log10(np.abs(np.asarray(tf.data)[0, 0])
                                  / np.abs(np.asarray(ref.data)[0, 0]))
        median_dB = float(np.median(np.abs(band_dB)))
        assert median_dB <= tolerance_dB, (
            f'{label} vs Scooter over the band: median |dH| {median_dB:.2f} '
            f'dB > {tolerance_dB} dB'
        )
        return
    j_fc = int(np.argmin(np.abs(ref_freqs - FC)))
    Href = np.abs(np.asarray(ref.data)[0, 0, j_fc])
    diff_dB = 20.0 * np.log10(Hfc / Href)
    assert abs(diff_dB) <= tolerance_dB, (
        f'{label} vs Scooter at fc: |H| differs by {diff_dB:.2f} dB '
        f'> {tolerance_dB} dB'
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

    # Nyquist 2048 Hz is far above F_HI = 75 Hz, so the IFFT cannot alias; the
    # point of going this fine is the 0.24 ms time step, which puts the
    # envelope-peak quantisation ~400x below the 100 ms gate.
    fs = 4096.0
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
        times = np.asarray(ts.times)
        peaks[label] = _envelope_peak_time(
            np.asarray(ts.data[0, 0]), times, win)
        # A peak on the record's first or last sample is the truncated
        # (wrapped) record edge, not an arrival.
        assert peaks[label] not in (float(times[0]), float(times[-1])), (
            f'{label}: envelope peak {peaks[label]:.4f} s sits on the record '
            f'edge [{times[0]:.4f}, {times[-1]:.4f}] s')

    spread = max(peaks.values()) - min(peaks.values())
    assert spread <= 0.100, (
        f'Inter-model envelope-peak spread {spread*1000:.1f} ms > 100 ms; '
        f'peaks: {peaks}'
    )


def test_synthesize_time_series_honors_user_sample_rate():
    """The :class:`Field` returned by
    :meth:`Field.synthesize_time_series` sits on the same
    sampling grid as the source pulse — i.e. ``ts.sample_rate == sample_rate``
    exactly."""
    env = _pekeris_env()
    src, rcv = _src_rcv()
    tf = _scooter_bb(env, src, rcv)
    fs = 4096.0
    pulse = _gaussian_pulse(FC, fs)
    ts = tf.synthesize_time_series(pulse, sample_rate=fs)
    assert ts.sample_rate == pytest.approx(fs, rel=1e-6), (
        f'expected sample_rate={fs}, got {ts.sample_rate}'
    )


@pytest.mark.parametrize('model_cls', [Bellhop, Kraken, Scooter])
def test_single_frequency_broadband_auto_expands_the_band(model_cls):
    """source-receiver.md §6 "a single value auto-expands to":
    for BROADBAND a single-element frequency
    is a *centre* frequency, auto-expanded to ``fc·(1 ± bandwidth/2)`` — 128
    uniform bins over ``[0.75·fc, 1.25·fc]`` with the shared defaults
    (``_band.broadband_band``, through ``_requested_frequencies``) — while a multi-element
    vector IS the band, verbatim. Resolver-level, one shared code path per
    engine; nothing runs."""
    from uacpy.models._defaults import (
        DEFAULT_BROADBAND_BANDWIDTH_FACTOR, DEFAULT_BROADBAND_N_FREQS,
    )
    assert DEFAULT_BROADBAND_N_FREQS == 128
    assert DEFAULT_BROADBAND_BANDWIDTH_FACTOR == 0.5
    model = model_cls(verbose=False)
    freqs = model._requested_frequencies(
        RunMode.BROADBAND, Source(depths=DEPTH_M, frequencies=200.0), None,
        None).frequencies
    assert freqs.shape == (128,)
    assert freqs[0] == pytest.approx(200.0 * 0.75)
    assert freqs[-1] == pytest.approx(200.0 * 1.25)
    assert np.allclose(np.diff(freqs), freqs[1] - freqs[0])
    band = model._requested_frequencies(
        RunMode.BROADBAND,
        Source(depths=DEPTH_M, frequencies=np.array([50.0, 60.0, 70.0])),
        None, None).frequencies
    np.testing.assert_array_equal(band, [50.0, 60.0, 70.0])


def _sparc_pseudo_gaussian(t: np.ndarray, f: float) -> np.ndarray:
    """AT's 'P' source pulse (``tslib/cans.f90:26-29``):
    ``s(t) = 0.75 − cos(ωt) + 0.25·cos(2ωt)`` on ``[0, 1/f]``, zero
    elsewhere."""
    w = 2.0 * np.pi * f
    s = 0.75 - np.cos(w * t) + 0.25 * np.cos(2.0 * w * t)
    return np.where((t >= 0.0) & (t <= 1.0 / f), s, 0.0)


@pytest.mark.slow
def test_sparc_pn_n_pulse_deconvolves_onto_kraken_broadband_at_0_dB():
    """With ``pulse_type='PN+N'`` — no per-wavenumber band-pass, which is
    what the scalar deconvolution cannot undo — SPARC's deconvolved p(t)
    reads Kraken's broadband TL cell by cell, with no common gain: both are
    on the package's unit-source level (RA-WAVE-3). SPARC appears in no
    other cross-model comparison, so this is the one place its absolute
    level is tied to another engine.

    The comparison deconvolves the received spectrum by the analytic
    pseudo-Gaussian source spectrum on the same grid:
    ``TL = −20·log10|R(f)/S(f)|`` against Kraken's broadband ``|H|`` at the
    same FFT-bin frequencies. Geometry choices that keep it honest:

    * an explicitly ``'rigid'`` bottom, so SPARC's forced rigidification
      models the same waveguide Kraken solves;
    * comparison bins mid-way between the rigid-guide mode cutoffs
      ``(2m−1)·c/4D`` = 33.75 and 41.25 Hz, so every in-band mode's energy
      (slowest group speed ≈ 594 m/s) has fully arrived inside the record;
    * ``Kraken(c_high=10000)``: near-cutoff rigid-guide modes run to
      ~5.5 km/s phase speed, which the default 1.05× window would discard;
    * ``rmax_factor=7``: the Δk range sum is periodic with period
      RMax, and in a *lossless* rigid guide the nearest periodic image
      (RMax − r) arrives undamped — the margin pushes its first arrival
      (≈ 5.1 s) past the 4 s record instead of into it.

    Measured over the 9 cells: common gain +0.00 dB, |ΔTL| median 0.19,
    max 0.29 dB. The gain bound of ±0.5 dB is 12 times below the 6.02 dB a
    half-pressure field reads; the per-cell bound of 1 dB is 3.4x the
    measured maximum.
    """
    env = Environment(
        name='sparc-vs-kraken', bathymetry=100.0, ssp=1500.0,
        bottom=BoundaryProperties(acoustic_type='rigid'))
    fc = 37.5
    z_src, z_rcv = 20.0, 65.0
    ranges = np.array([800.0, 1000.0, 1200.0])
    time_max = 4.0                       # bins at n/4 Hz — targets land exactly
    ts = SPARC(verbose=False, pulse_type='PN+N', output_mode='R',
               n_time_samples=2048, time_max=time_max, freq_min=5.0, freq_max=75.0,
               rmax_factor=7.0, timeout=600.0).run(
        env, Source(depths=z_src, frequencies=fc),
        Receiver(depths=np.array([z_rcv]), ranges=ranges),
        run_mode=RunMode.TIME_SERIES)
    times = np.asarray(ts.coords['time'], dtype=float)
    dt = float(times[1] - times[0])
    traces = np.asarray(ts.data)[0]               # (n_ranges, n_t)
    assert np.all(np.isfinite(traces))
    spectra = np.fft.rfft(traces, axis=-1)
    freqs = np.fft.rfftfreq(traces.shape[-1], dt)
    targets = np.array([36.75, 37.5, 38.25])
    bins = np.array([int(np.argmin(np.abs(freqs - f))) for f in targets])
    f_bins = freqs[bins]
    source_spectrum = np.fft.rfft(_sparc_pseudo_gaussian(times, fc))[bins]
    assert np.all(np.abs(source_spectrum) > 1.0)   # well off any pulse null
    tl_sparc = -20.0 * np.log10(
        np.abs(spectra[:, bins] / source_spectrum[np.newaxis, :]))

    kraken = Kraken(verbose=False, c_high=10000.0).run(
        env, Source(depths=z_src, frequencies=fc),
        Receiver(depths=np.array([z_rcv]), ranges=ranges),
        run_mode=RunMode.BROADBAND, frequencies=f_bins)
    tl_kraken = np.asarray(kraken.dB)[0]          # (n_ranges, n_bins)

    diff = tl_sparc - tl_kraken
    assert np.all(np.isfinite(diff)), (tl_sparc, tl_kraken)
    gain = np.median(diff)
    assert gain == pytest.approx(0.0, abs=0.5), (
        f"SPARC sits {gain:.2f} dB off Kraken's unit-source level\n"
        f"SPARC:\n{tl_sparc}\nKraken:\n{tl_kraken}")
    assert np.max(np.abs(diff)) <= 1.0, (
        f"max |dTL| = {np.max(np.abs(diff)):.2f} dB\n"
        f"SPARC:\n{tl_sparc}\nKraken:\n{tl_kraken}")


# ── A synthesised record that cuts through the arrival train says so ─────────

_WRAP_NOTICE = 'synthesised record'


def _wrap_notices(fn):
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fn()
    return [str(w.message) for w in caught if _WRAP_NOTICE in str(w.message)]


@pytest.mark.parametrize('n_freq, fires', [(26, True), (51, False)])
def test_a_record_ending_inside_the_arrival_train_is_announced(n_freq, fires):
    """The auto window opens max(T/10, 4/B) before r/c_max = 2.353 s
    (c_max 1700 m/s at 4 km, B = 50 Hz). At Δf = 2 Hz Kraken's 0.5 s record
    opens 0.08 s early, at [2.273, 2.773] s, ending before the 2.887 s main
    arrival: its end holds energy at -1.8 dB and the train wraps. At
    Δf = 1 Hz the 1 s record opens 0.1 s early, at [2.253, 3.253] s, holds
    the train (edges at -55 / -44 dB) and nothing is said."""
    import warnings
    env = _pekeris_env()
    src, rcv = _src_rcv()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tf = Kraken(verbose=False).run(
            env, src, rcv, frequencies=np.linspace(F_LO, F_HI, n_freq),
            run_mode=RunMode.BROADBAND)
    said = _wrap_notices(lambda: tf.synthesize_time_series(
        _gaussian_pulse(FC, 4096.0), sample_rate=4096.0))
    assert bool(said) is fires, said
    if fires:
        assert '[2.273, 2.773] s' in said[0]
        assert '1/Δf = 0.5 s' in said[0] and 't_start=' in said[0]


def test_bellhop_at_its_default_window_is_silent():
    # Bellhop's own TIME_SERIES trace (delay-and-sum, not periodic) and its
    # default BROADBAND grid synthesised through the shared path.
    import warnings
    env = _pekeris_env()
    src, rcv = _src_rcv()
    fs = 4096.0
    pulse = _gaussian_pulse(FC, fs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tf = Bellhop(verbose=False).run(env, src, rcv,
                                        run_mode=RunMode.BROADBAND)
    assert not _wrap_notices(lambda: tf.synthesize_time_series(
        pulse, sample_rate=fs))
    assert not _wrap_notices(lambda: Bellhop(verbose=False).run(
        env, src, rcv, run_mode=RunMode.TIME_SERIES,
        source_waveform=pulse, sample_rate=fs))


class TestRecordEdgeNoticeOnASyntheticArrival:
    """One arrival ``H = e^{-2πifτ}`` at 3 km with ``c_max = 1500`` m/s and
    1/Δf = 4 s over B = 100 Hz: the auto window opens max(T/10, 4/B) =
    0.4 s before r/c_max = 2 s, i.e. [1.6, 5.6) s. An arrival mid-record is
    silent, one straddling the record end is reported, and the same record
    placed by a caller's t_start is not judged."""

    @staticmethod
    def _field(tau):
        from uacpy.core.results import Field, PhaseReference, SoundSpeeds
        freqs = np.arange(50.0, 150.0 + 0.125, 0.25)
        return Field(
            data=np.exp(-2j * np.pi * freqs * tau)[None, None, :],
            coords={'depth': np.array([50.0]), 'range': np.array([3000.0]),
                    'frequency': freqs},
            model='Synthetic', source_depths=np.array([50.0]),
            frequencies=freqs, speeds=SoundSpeeds(water_max=1500.0),
            phase_reference=PhaseReference.TRAVELLING_WAVE)

    def _said(self, tau, **kw):
        return _wrap_notices(lambda: self._field(tau).synthesize_time_series(
            _gaussian_pulse(100.0, 1024.0), sample_rate=1024.0, **kw))

    def test_an_arrival_mid_record_is_silent(self):
        assert not self._said(2.0)

    def test_an_arrival_straddling_the_record_end_is_reported(self):
        # The 0.2 s pulse is centred 0.1 s after tau, so tau = 5.45 s puts
        # its peak 0.05 s before the record end at 5.6 s and its span
        # [5.45, 5.65] s across it.
        said = self._said(5.45)
        assert said and '[1.6, 5.599] s' in said[0], said
        assert '1/Δf = 4 s' in said[0], said

    def test_a_record_placed_by_the_caller_is_not_judged(self):
        assert not self._said(5.45, t_start=1.6)

    @staticmethod
    def _short_record_said(df):
        from uacpy.core.results import Field, PhaseReference, SoundSpeeds
        freqs = np.arange(50.0, 150.0 + df / 2.0, df)
        tf = Field(
            data=np.exp(-2j * np.pi * freqs * 2.1)[None, None, :],
            coords={'depth': np.array([50.0]), 'range': np.array([3000.0]),
                    'frequency': freqs},
            model='Synthetic', source_depths=np.array([50.0]),
            frequencies=freqs, speeds=SoundSpeeds(water_max=1500.0),
            phase_reference=PhaseReference.TRAVELLING_WAVE)
        return _wrap_notices(lambda: tf.synthesize_time_series(
            _gaussian_pulse(100.0, 1024.0), sample_rate=1024.0))

    def test_a_record_a_lone_arrival_already_crosses_is_not_judged(self):
        # Edges are judged when T(1 - 0.05) - lead > the 0.2 s pulse, with
        # lead = max(T/10, 4/B) = 0.04 s here: T > 0.2526 s, Δf < 3.96 Hz.
        # At Δf = 4 Hz (T = 0.25 s) a lone arrival at tau = 2.1 s peaks on
        # the record end, which measures the grid against the pulse (the
        # DFT-period and derived-grid notices' business), not the channel.
        assert not self._short_record_said(4.0)

    def test_a_record_just_longer_than_the_bound_is_judged(self):
        # Δf = 3.75 Hz (T = 0.267 s): the same arrival is reported.
        said = self._short_record_said(3.75)
        assert said and '[1.96, 2.227] s' in said[0], said
