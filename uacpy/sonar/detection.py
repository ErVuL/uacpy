"""Detection-theory utilities linking (P_D, P_F) to required SNR / DT.

Gaussian (equal-variance) binary detection and standard sonar/radar closed
forms used to populate the detection-threshold term of the sonar equation.

References
----------
Urick, R.J. (1983). *Principles of Underwater Sound*, 3rd ed., Ch. 12.
Albersheim, W.J. (1981). A closed-form approximation to Robertson's
    detection characteristics. Proc. IEEE 69(7), 839.
Richards, M.A. *Fundamentals of Radar Signal Processing* (Albersheim eq.).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from uacpy.core.exceptions import ConfigurationError


def _check_prob(p, name: str) -> float:
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ConfigurationError(f"{name} must be in (0, 1), got {p}")
    return p


def deflection_coefficient(pd: float, pf: float) -> float:
    """Gaussian deflection ``d' = Phi^-1(P_D) - Phi^-1(P_F)``.

    The separation (in noise standard deviations) between the signal-present
    and signal-absent decision statistics needed to achieve ``(P_D, P_F)``.
    """
    pd = _check_prob(pd, "pd")
    pf = _check_prob(pf, "pf")
    return float(norm.ppf(pd) - norm.ppf(pf))


def detection_index(pd: float, pf: float) -> float:
    """Urick detection index ``d = (d')^2`` for ``(P_D, P_F)``."""
    return deflection_coefficient(pd, pf) ** 2


def probability_of_detection(deflection, pf):
    """``P_D = Q(Q^-1(P_F) - d')`` for a given deflection and false-alarm rate.

    ``pf`` may be a scalar or array; ``deflection`` broadcasts against it.
    """
    pf = np.asarray(pf, dtype=float)
    if np.any((pf <= 0.0) | (pf >= 1.0)):
        raise ConfigurationError("pf must be in (0, 1)")
    return norm.sf(norm.isf(pf) - np.asarray(deflection, dtype=float))


def roc_curve(deflection: float, n_points: int = 200):
    """ROC ``(P_F, P_D)`` for a Gaussian detector of the given deflection.

    Returns two arrays sampling ``P_F`` logarithmically over ``[1e-6, ~1]``.
    """
    if deflection < 0.0:
        raise ConfigurationError("roc_curve: deflection must be >= 0")
    pf = np.logspace(-6.0, np.log10(0.99), int(n_points))
    pd = probability_of_detection(deflection, pf)
    return pf, pd


def albersheim_snr(pd: float, pf: float, n_pulses: int = 1) -> float:
    """Required per-sample SNR (dB) via Albersheim's equation.

    Non-coherent integration of ``n_pulses`` samples through a linear/square-law
    envelope detector (single sample for ``n_pulses=1``). Accurate to ~0.2 dB
    over ``0.1 <= P_D <= 0.9``, ``1e-7 <= P_F <= 1e-3``.
    """
    pd = _check_prob(pd, "pd")
    pf = _check_prob(pf, "pf")
    n = int(n_pulses)
    if n < 1:
        raise ConfigurationError("albersheim_snr: n_pulses must be >= 1")
    a = np.log(0.62 / pf)
    b = np.log(pd / (1.0 - pd))
    snr_db = (
        -5.0 * np.log10(n)
        + (6.2 + 4.54 / np.sqrt(n + 0.44))
        * np.log10(a + 0.12 * a * b + 1.7 * b)
    )
    return float(snr_db)


def detection_threshold_energy(
    pd: float, pf: float, bandwidth_hz: float, integration_time_s: float
) -> float:
    """Detection threshold (dB) for an incoherent energy detector.

    ``DT = 5*log10(d / (w * t))`` where ``d`` is the detection index and ``w*t``
    is the time-bandwidth product (Urick 1983, Ch. 12, large-``wt`` energy /
    square-law detector). Use as the ``DT`` term of the active/passive sonar
    equation when the receiver integrates signal energy over ``t`` seconds in a
    ``w``-Hz band. The threshold *decreases* by 5 dB per decade of increase in
    the time-bandwidth product ``w*t`` (Abraham §9.2: SNR_d falls 5 dB/decade in
    ``M = T*W``) — more incoherent integration lowers the required SNR.
    """
    if bandwidth_hz <= 0.0 or integration_time_s <= 0.0:
        raise ConfigurationError(
            "detection_threshold_energy: bandwidth_hz and integration_time_s"
            " must be > 0"
        )
    d = detection_index(pd, pf)
    return float(5.0 * np.log10(d / (bandwidth_hz * integration_time_s)))
