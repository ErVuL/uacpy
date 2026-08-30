"""Cepstrum value contracts: analytic echo series and exact inversion.

The smoke test in ``test_timefreq_funcs.py`` only asserts finiteness; this
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


def _delta_echo(n: int, delay: int, amplitude: float) -> np.ndarray:
    x = np.zeros(n)
    x[0] = 1.0
    x[delay] = amplitude
    return x


def test_real_cepstrum_of_single_echo_matches_analytic_series():
    """c[kD] = (-1)^{k+1} a^k / (2k), mirrored at N-kD, zero elsewhere
    (to the truncated-series tail, < a^6)."""
    n, delay, a = 1024, 100, 0.5
    c = cepstrum(_delta_echo(n, delay, a))

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

    c = cepstrum(x, lifter=-100)
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
