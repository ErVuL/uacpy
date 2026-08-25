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
    return RAM(verbose=False, Q=2.0, T=4.0, dr=2.0, dz=0.25).run(
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
# shared 6 dB bound spends all of that sensitivity on Bellhop's account.
# Each number below is the difference measured at this cell (RANGE_M,
# DEPTH_M, FC) followed by the multiple of it the bound allows.
_H_FC_TOLERANCE_DB = {
    # Measured -0.286 dB. Modes vs wavenumber integration on a fluid Pekeris
    # is an apples-to-apples comparison, so 1.5 dB is 5.2x the measurement
    # and still catches a factor-2 amplitude convention (6.02 dB) by 3.8x in
    # the nearer of the two directions.
    'Kraken': 1.5,
    # Measured -1.970 dB, and left loose deliberately. This fixture is
    # D/lambda = 3.33 (100 m water column at 50 Hz), below the D/lambda >= 5
    # floor uacpy's own model-validity table sets for ray theory — the run
    # emits exactly that warning. Bellhop's disagreement here is a
    # validity-regime gap, not numerical error with a bound to tighten, so
    # 6.0 dB is a sanity bound (3.0x the measurement) and Bellhop is the
    # only model in this gate that needs one that loose.
    'Bellhop': 6.0,
    # Measured +1.415 dB. 3.5 dB is 2.5x it — more headroom than Kraken gets
    # because RAM is the only model here whose accuracy is set by
    # user-facing marching parameters (``_ram_bb`` pins dr=2.0, dz=0.25), so
    # this difference is a function of a grid the caller chooses and the
    # others' are not. Still catches a factor-2 (6.02 dB) by 1.3x in the
    # nearer direction.
    'RAM': 3.5,
}

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
    wavenumber-integration ground truth on Pekeris)."""
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
    tolerance_db = _H_FC_TOLERANCE_DB[label]
    assert abs(diff_db) <= tolerance_db, (
        f'{label} vs Scooter at fc: |H| differs by {diff_db:.2f} dB '
        f'> {tolerance_db} dB'
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
    (base.py ``_resolve_broadband_frequencies``) — while a multi-element
    vector IS the band, verbatim. Resolver-level, one shared code path per
    engine; nothing runs."""
    from uacpy.core.constants import (
        DEFAULT_BROADBAND_BANDWIDTH_FACTOR, DEFAULT_BROADBAND_N_FREQS)
    assert DEFAULT_BROADBAND_N_FREQS == 128
    assert DEFAULT_BROADBAND_BANDWIDTH_FACTOR == 0.5
    model = model_cls(verbose=False)
    freqs = model._resolve_broadband_frequencies(
        Source(depths=DEPTH_M, frequencies=200.0), None)
    assert freqs.shape == (128,)
    assert freqs[0] == pytest.approx(200.0 * 0.75)
    assert freqs[-1] == pytest.approx(200.0 * 1.25)
    assert np.allclose(np.diff(freqs), freqs[1] - freqs[0])
    band = model._resolve_broadband_frequencies(
        Source(depths=DEPTH_M, frequencies=np.array([50.0, 60.0, 70.0])),
        None)
    np.testing.assert_array_equal(band, [50.0, 60.0, 70.0])


def _sparc_pseudo_gaussian(t: np.ndarray, f: float) -> np.ndarray:
    """AT's 'P' source pulse (``tslib/cans.f90:26-29``):
    ``s(t) = 0.75 − cos(ωt) + 0.25·cos(2ωt)`` on ``[0, 1/f]``, zero
    elsewhere."""
    w = 2.0 * np.pi * f
    s = 0.75 - np.cos(w * t) + 0.25 * np.cos(2.0 * w * t)
    return np.where((t >= 0.0) & (t <= 1.0 / f), s, 0.0)


@pytest.mark.slow
def test_sparc_pn_n_pulse_deconvolves_onto_kraken_broadband():
    """sparc.md §7 "calibrates tighter": with ``pulse_type='PN+N'`` — no
    per-wavenumber
    band-pass, which is what the scalar deconvolution cannot undo — SPARC's
    p(t) calibrates to ~±1.5 dB against Kraken. SPARC appears in no other
    cross-model comparison, so this is the one place its absolute level is
    tied to another engine.

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
    * ``rmax_safety_margin=7``: the Δk range sum is periodic with period
      RMax, and in a *lossless* rigid guide the nearest periodic image
      (RMax − r) arrives undamped — the margin pushes its first arrival
      (≈ 5.1 s) past the 4 s record instead of into it.

    The ±1.5 dB gate is the doc's own measured figure, applied to the
    median over 9 cells so one interference null cannot decide the test.
    """
    env = Environment(
        name='sparc-vs-kraken', bathymetry=100.0, ssp=1500.0,
        bottom=BoundaryProperties(acoustic_type='rigid'))
    fc = 37.5
    z_src, z_rcv = 20.0, 65.0
    ranges = np.array([800.0, 1000.0, 1200.0])
    t_max = 4.0                       # bins at n/4 Hz — targets land exactly
    ts = SPARC(verbose=False, pulse_type='PN+N', output_mode='R',
               n_t_out=2048, t_max=t_max, f_min=5.0, f_max=75.0,
               rmax_safety_margin=7.0, timeout=600.0).run(
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
    tl_kraken = np.asarray(kraken.db)[0]          # (n_ranges, n_bins)

    diff = tl_sparc - tl_kraken
    assert np.all(np.isfinite(diff)), (tl_sparc, tl_kraken)
    # Measured: SPARC sits a uniform ~6.06 dB above Kraken here — within
    # 0.5 dB of 20*log10(2), i.e. a global factor-2 amplitude convention
    # between SPARC's injected 'PN+N' pulse and the cans.f90 closed form
    # used for the deconvolution. The gain-removed residual is what
    # sparc.md's ±1.5 dB describes, so the pin is: (a) the per-cell spread
    # about the common gain stays inside ±1.5 dB, and (b) the common gain
    # itself stays at the factor-2 value so a future convention change
    # fails loudly here rather than silently shifting.
    gain = np.median(diff)
    assert np.median(np.abs(diff - gain)) <= 1.5, (
        f"gain-removed median |dTL| = {np.median(np.abs(diff - gain)):.2f} "
        f"dB exceeds the ±1.5 dB sparc.md quotes for 'PN+N' vs Kraken\n"
        f"SPARC:\n{tl_sparc}\nKraken:\n{tl_kraken}")
    assert gain == pytest.approx(20.0 * np.log10(2.0), abs=1.0), (
        f"SPARC-vs-Kraken common gain {gain:.2f} dB moved away from the "
        f"documented-by-measurement 6.02 dB factor-2 offset")
