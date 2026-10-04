"""Cepstrum value contracts: analytic echo series and exact inversion.

The smoke test in ``test_time_frequency.py`` only asserts finiteness; this
module pins what the transforms actually compute:

* **Real cepstrum of a single echo** ``x = δ + a·δ_D``: the spectrum is
  ``1 + a·e^{-iωD}``, so ``log|X|`` expands to
  ``Σ (-1)^{k+1} (a^k/k)·cos(kωD)`` and the (even, two-sided) real cepstrum
  carries **exactly** ``c[kD] = (-1)^{k+1} a^k / (2k)`` with the mirror at
  ``c[N-kD]`` — an analytic closed form, asserted to float precision.
* **Complex cepstrum of the same echo** is the one-sided series
  ``ĉ[kD] = (-1)^{k+1} a^k / k`` (minimum-phase, ``delay == 0``).
* **Round-trip**: ``inverse_complex_cepstrum(complex_cepstrum(x)) == x`` to
  machine precision — the homomorphic transform is exactly reversible
  (``exp`` of the log-spectrum restores magnitude *and* phase regardless of
  the unwrapping branch, and the removed linear-phase ``delay`` is restored).
* **Echo detection through a real signal**: with a broadband carrier, the
  long-passed (``lifter < 0``) cepstrum peaks at the echo quefrency.
"""

from __future__ import annotations

import numpy as np

from uacpy.acoustic_signal import (
    cepstrum, complex_cepstrum, inverse_complex_cepstrum,
)
from uacpy.core.exceptions import ConfigurationError
import pytest


def _delta_echo(n: int, delay: int, amplitude: float) -> np.ndarray:
    x = np.zeros(n)
    x[0] = 1.0
    x[delay] = amplitude
    return x


def test_real_cepstrum_of_single_echo_matches_analytic_series():
    """c[kD] = (-1)^{k+1} a^k / (2k), mirrored at N-kD, zero elsewhere
    (to the truncated-series tail, < a^6)."""
    n, delay, a = 1024, 100, 0.5
    c = cepstrum(_delta_echo(n, delay, a)).cepstrum

    for k in (1, 2, 3):
        expected = (-1.0) ** (k + 1) * a ** k / (2 * k)
        assert np.isclose(c[k * delay], expected, rtol=0, atol=1e-9), (
            f"c[{k}D] = {c[k * delay]} != analytic {expected}"
        )
        # The real cepstrum is even: the same value sits at N - kD.
        assert np.isclose(c[n - k * delay], expected, rtol=0, atol=1e-9)

    # Away from the echo rahmonics the cepstrum is (numerically) zero.
    harmonic_bins = [0] + [k * delay for k in range(1, 6)]
    background = np.abs(np.delete(c[: n // 2], harmonic_bins))
    assert background.max() < a ** 6, (
        f"background {background.max()} above the truncated-series tail"
    )


def test_complex_cepstrum_of_single_echo_is_one_sided_series():
    """The complex cepstrum of a minimum-phase echo (a < 1) is one-sided:
    ĉ[kD] = (-1)^{k+1} a^k / k at positive quefrency, ~0 at negative
    (top-of-buffer) quefrencies, with no linear-phase delay removed."""
    n, delay, a = 1024, 100, 0.5
    cc = complex_cepstrum(_delta_echo(n, delay, a))
    assert cc.delay == 0

    for k in (1, 2, 3):
        expected = (-1.0) ** (k + 1) * a ** k / k
        assert np.isclose(cc.cepstrum[k * delay], expected,
                          rtol=0, atol=1e-9), (
            f"ĉ[{k}D] = {cc.cepstrum[k * delay]} != analytic {expected}"
        )
    # One-sided: the negative-quefrency mirror bins stay ~empty.
    for k in (1, 2, 3):
        assert abs(cc.cepstrum[n - k * delay]) < a ** 6


def test_complex_cepstrum_round_trips_to_machine_precision():
    """inverse_complex_cepstrum(complex_cepstrum(x)) reconstructs x exactly
    (float round-off), for a generic real signal with a full spectrum."""
    rng = np.random.default_rng(0xACED)
    x = rng.standard_normal(512) * np.hanning(512)
    reconstructed = inverse_complex_cepstrum(complex_cepstrum(x))
    np.testing.assert_allclose(reconstructed, x, rtol=0,
                               atol=1e-10 * np.abs(x).max())


def test_complex_cepstrum_round_trips_echoed_signal():
    """Round-trip also holds for a signal with strong echo structure (the
    case the cepstrum exists for), not just for smooth-spectrum noise."""
    rng = np.random.default_rng(0xACED)
    n, delay, a = 2048, 200, 0.5
    base = np.zeros(n)
    base[: n - delay] = rng.standard_normal(n - delay)
    x = base.copy()
    x[delay:] += a * base[: n - delay]
    reconstructed = inverse_complex_cepstrum(complex_cepstrum(x))
    np.testing.assert_allclose(reconstructed, x, rtol=0,
                               atol=1e-9 * np.abs(x).max())


def test_long_pass_lifter_isolates_echo_quefrency_in_noise():
    """Through a broadband carrier, the long-passed real cepstrum
    (lifter < 0 zeroes the low-quefrency spectral envelope) peaks exactly at
    the echo delay — the detection use-case the transform is for."""
    rng = np.random.default_rng(0xACED)
    n, delay, a = 2048, 200, 0.5
    base = np.zeros(n)
    base[: n - delay] = rng.standard_normal(n - delay)
    x = base.copy()
    x[delay:] += a * base[: n - delay]

    c = cepstrum(x, lifter=-100).cepstrum
    assert np.allclose(c[:101], 0.0)          # envelope removed
    peak_quefrency = int(np.argmax(c[: n // 2]))
    assert peak_quefrency == delay, (
        f"cepstral peak at quefrency {peak_quefrency}, echo delay is {delay}"
    )
    # The peak carries ~a/2 (two-sided split of the log-spectrum series).
    assert c[delay] > 0.5 * (a / 2)


def test_inverse_complex_cepstrum_requires_the_namedtuple():
    """A bare cepstrum array carries no delay field, so the linear-phase term
    cannot be restored; the inverse demands the ComplexCepstrum namedtuple."""
    import pytest
    from uacpy.core.exceptions import ConfigurationError
    x = np.roll(np.exp(-np.arange(64) / 8.0), 10)
    cc = complex_cepstrum(x)
    with pytest.raises(ConfigurationError, match="ComplexCepstrum"):
        inverse_complex_cepstrum(cc.cepstrum)
    np.testing.assert_allclose(inverse_complex_cepstrum(cc), x, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# acoustic_signal/timefreq.py — lifter, smoothing windows, cwt, cepstra
# ─────────────────────────────────────────────────────────────────────────────


class TestCepstralLifterMask:
    """Scalar lifter L builds the mask 1 on quefrencies |q| <= |L| (head
    and mirrored tail), inverted for negative L; an array lifter
    multiplies element-wise after an exact shape check."""

    def test_low_pass_mask_exact(self):
        from uacpy.acoustic_signal.timefreq import _apply_lifter
        c = np.arange(1.0, 9.0)
        want = c * np.array([1, 1, 1, 0, 0, 0, 1, 1], dtype=float)
        np.testing.assert_allclose(_apply_lifter(c, 2), want, rtol=0)

    def test_zero_lifter_keeps_only_dc(self):
        from uacpy.acoustic_signal.timefreq import _apply_lifter
        c = np.arange(1.0, 9.0)
        want = np.zeros(8)
        want[0] = 1.0
        np.testing.assert_allclose(_apply_lifter(c, 0), want, rtol=0)

    def test_negative_lifter_is_the_complement(self):
        from uacpy.acoustic_signal.timefreq import _apply_lifter
        c = np.arange(1.0, 9.0)
        np.testing.assert_allclose(
            _apply_lifter(c, 2) + _apply_lifter(c, -2), c, rtol=0)

    def test_array_lifter_multiplies_and_checks_shape(self):
        from uacpy.acoustic_signal.timefreq import _apply_lifter
        c = np.arange(1.0, 9.0)
        np.testing.assert_allclose(_apply_lifter(c, 2.0 * np.ones(8)),
                                   2.0 * c, rtol=0)
        with pytest.raises(ConfigurationError, match="must match"):
            _apply_lifter(c, np.ones(7))


class TestCepstrumWindowConvention:
    """``window=`` multiplies by the *periodic* (fftbins) window before
    the spectrum."""

    def test_matches_prewindowed_signal(self):
        import scipy.signal as _sig
        from uacpy.acoustic_signal.timefreq import cepstrum
        x = np.random.default_rng(7).normal(size=64)
        w = _sig.get_window("hann", 64, fftbins=True)
        np.testing.assert_allclose(cepstrum(x, window="hann").cepstrum,
                                   cepstrum(x * w).cepstrum, rtol=1e-12,
                                   atol=1e-12)


class TestComplexCepstrumDelayEstimator:
    """The linear-phase ramp is rounded to whole samples: a pure delayed
    impulse reports exactly its ramp; degenerate lengths report zero
    without dividing by zero."""

    def test_delayed_impulse_reports_the_ramp(self):
        from uacpy.acoustic_signal.timefreq import complex_cepstrum
        x = np.zeros(31)
        x[3] = 1.0
        r = complex_cepstrum(x)
        assert r.delay == 3            # three samples LATE: positive
        assert np.isfinite(r.cepstrum).all()

    def test_two_sample_signal_computes_the_ramp(self):
        from uacpy.acoustic_signal.timefreq import complex_cepstrum
        # Two samples: the one-sample delay sits at Nyquist, where +1 and -1
        # are the same ramp — only its magnitude is determined.
        assert abs(complex_cepstrum(np.array([0.0, 1.0])).delay) == 1

    def test_single_sample_signal_is_zero_delay(self):
        from uacpy.acoustic_signal.timefreq import complex_cepstrum
        r = complex_cepstrum(np.array([2.0]))
        assert r.delay == 0
        assert np.isfinite(r.cepstrum).all()
