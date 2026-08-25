"""Detection-theory utilities linking (P_D, P_F) to required SNR / DT.

Gaussian (equal-variance) binary detection and standard sonar/radar closed
forms used to populate the detection-threshold term of the sonar equation.

References
----------
Urick, R.J. (1983). *Principles of Underwater Sound*, 3rd ed., Ch. 12.
Albersheim, W.J. (1981). A closed-form approximation to Robertson's
    detection characteristics. Proc. IEEE 69(7), 839.
Richards, M.A. (2014). "Alternative Forms of Albersheim's Equation" —
    states eq. (1) and its accuracy/validity ranges verbatim.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import gamma, norm

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

    Gaussian (Neyman-Pearson) detector model: signal-absent and
    signal-present decision statistics are unit-variance Gaussians separated
    by the deflection ``d'``, and the decision threshold is set by the
    false-alarm constraint ``P_F``. ``pf`` may be a scalar or array;
    ``deflection`` broadcasts against it.

    This is **not** the scalar form of
    :func:`uacpy.sonar.sonar_equation.probability_of_detection_field` —
    that function evaluates a different model, Urick's transition curve
    ``P_D = Phi(SE / sigma_db)`` (log-normal signal-excess fluctuation,
    ``P_D = 0.5`` pinned at ``SE = 0``, no ``P_F`` argument).
    """
    pf = np.asarray(pf, dtype=float)
    # Negated admissible interval so NaN is refused: both ``nan <= 0`` and
    # ``nan >= 1`` are False, and a NaN false-alarm rate otherwise returned a
    # silent NaN P_D. (:func:`_check_prob`, used by the Albersheim/Shnidman
    # entry points, is already written this way.)
    admissible = (pf > 0.0) & (pf < 1.0)
    if np.any(~admissible):
        raise ConfigurationError(
            f"pf must be in (0, 1); got "
            f"{np.unique(pf[~admissible]).tolist()}")
    d = np.asarray(deflection, dtype=float)
    # The guard above refuses a NaN ``pf`` because it returned a silent NaN
    # P_D; a NaN ``deflection`` returns that same silent NaN through the other
    # argument, so it is refused at the same door. ``inf`` and ``-inf`` return
    # a perfectly valid-looking probability — exactly 1.0 and 0.0 — which is
    # the degenerate limit of a detector rather than one, and is what
    # :func:`roc_curve`, this function's only in-package caller, already
    # refuses before it gets here.
    if np.any(~np.isfinite(d)):
        raise ConfigurationError(
            f"probability_of_detection: deflection must be finite; got "
            f"{np.unique(d[~np.isfinite(d)]).tolist()}")
    return norm.sf(norm.isf(pf) - d)


def roc_curve(deflection: float, n_points: int = 200):
    """ROC ``(P_F, P_D)`` for a Gaussian detector of the given deflection.

    Returns two arrays sampling ``P_F`` logarithmically over ``[1e-6, ~1]``.
    """
    # Negated admissible condition so a NaN deflection is refused instead of
    # returning an all-NaN curve. ``isfinite`` is the other half of the
    # message's "and finite", and this is the site where its absence was
    # invisible: ``inf >= 0`` is True, and an infinite deflection returns a
    # perfectly FINITE curve — P_D == 1 at every P_F — so nothing downstream
    # looks wrong. It is the degenerate perfect detector, not a ROC, and the
    # only signal that the argument was nonsense is this guard.
    if not np.isfinite(deflection) or not (deflection >= 0.0):
        raise ConfigurationError(
            f"roc_curve: deflection must be >= 0 and finite; got "
            f"{deflection!r}")
    pf = np.logspace(-6.0, np.log10(0.99), int(n_points))
    pd = probability_of_detection(deflection, pf)
    return pf, pd


def albersheim_snr(pd: float, pf: float, n_pulses: int = 1) -> float:
    """Required per-sample SNR (dB) via Albersheim's equation.

    Non-coherent integration of ``n_pulses`` samples through a linear/square-law
    envelope detector (single sample for ``n_pulses=1``).

    ``A = ln(0.62/P_F)``, ``B = ln(P_D/(1-P_D))`` and
    ``SNR_dB = -5·log10(N) + (6.2 + 4.54/sqrt(N+0.44))·log10(A + 0.12·A·B + 1.7·B)``
    (Richards 2014, eq. 1). Accurate to ~0.2 dB over ``0.1 <= P_D <= 0.9``,
    ``1e-7 <= P_F <= 1e-3`` and ``1 <= N <= 8096``; outside that envelope a
    ``UserWarning`` is issued and the value is an unvalidated extrapolation
    of the fit.
    """
    pd = _check_prob(pd, "pd")
    pf = _check_prob(pf, "pf")
    n = int(n_pulses)
    if n < 1:
        raise ConfigurationError(
            f"albersheim_snr: n_pulses must be >= 1; got {n}")
    if not (0.1 <= pd <= 0.9 and 1e-7 <= pf <= 1e-3 and n <= 8096):
        warnings.warn(
            f"albersheim_snr: (pd={pd:g}, pf={pf:g}, n_pulses={n}) is outside "
            f"the envelope Albersheim's equation was fitted over — "
            f"0.1 <= pd <= 0.9, 1e-7 <= pf <= 1e-3, 1 <= N <= 8096 "
            f"(Richards 2014) — so the ~0.2 dB accuracy bound does not apply "
            f"and the value is an extrapolation of an empirical fit.",
            UserWarning, stacklevel=2,
        )
    a = np.log(0.62 / pf)
    b = np.log(pd / (1.0 - pd))
    snr_db = (
        -5.0 * np.log10(n)
        + (6.2 + 4.54 / np.sqrt(n + 0.44))
        * np.log10(a + 0.12 * a * b + 1.7 * b)
    )
    return float(snr_db)


#: How far the shipped large-M approximation may sit from the exact
#: energy-detector threshold before ``detection_threshold_energy`` warns.
#: Abraham's own accuracy statement for eq. (2.77) is quoted in dB, so the
#: envelope is policed in the units it is promised in.
_DT_APPROXIMATION_TOLERANCE_DB = 1.0


def _exact_detection_threshold_db(pd: float, pf: float, m: float) -> float:
    """Exact required per-cell SNR (dB) for the noise-normalised energy detector.

    ``M`` independent cells of unit-mean noise give ``T ~ Gamma(M, 1)`` under
    ``H0``; a Gaussian-fluctuating signal at per-cell SNR ``S`` scales it to
    ``(1 + S)·Gamma(M, 1)`` under ``H1``. The threshold is therefore
    ``h = Ginv(1 - Pf; M)`` and ``h/(1 + S) = Ginv(1 - Pd; M)``, so
    ``S = Ginv(1-Pf; M) / Ginv(1-Pd; M) - 1``. Feeding that ``S`` back through
    the Gamma survival function recovers the requested operating point exactly,
    which is what makes it the benchmark rather than a second approximation.

    Returns a non-finite value where the quantiles do not resolve one; the
    caller falls back rather than suppressing its own check.
    """
    with np.errstate(all="ignore"):
        try:
            snr = gamma.isf(pf, m) / gamma.isf(pd, m) - 1.0
        except (ValueError, ZeroDivisionError):
            return float("nan")
        return float(10.0 * np.log10(snr))


def detection_threshold_energy(
    pd: float, pf: float, bandwidth_hz: float, integration_time_s: float
) -> float:
    """Detection threshold (dB) for an incoherent energy detector.

    ``DT = 5*log10(d / (w * t))`` where ``d`` is the detection index and ``w*t``
    is the processing time-bandwidth product ``M`` (Abraham, *Underwater
    Acoustic Signal Processing*, §9.2.3.1 / §9.2.11). The required SNR
    *decreases* by 5 dB per decade of increase in ``M`` — more incoherent
    integration relaxes the required SNR.

    **Which reference to pair it with.** This ``DT`` is the required ratio
    ``S0/N0`` of signal to noise *power spectral density*. A ratio of two PSDs
    over the same band equals the ratio of the two band powers, so ``DT`` here
    is a **unitless power ratio** (Abraham §2.3.5.5 / the ``DT`` vs ``DT_Hz``
    distinction): it is correct whenever the source and noise levels in the
    sonar equation share a reference — both spectral levels
    (dB re 1 µPa²/Hz), or both band-integrated levels (dB re 1 µPa²). What it
    must *not* be paired with is a mixed pair, e.g. a band-integrated source
    level against a spectral-level noise.

    That mixed case is Urick's form, ``DT = 5*log10(d*w/t)`` — signal band
    power referenced to noise in a 1-Hz band (Abraham's ``DT_Hz``, units
    dB re Hz). The two differ by ``10*log10(w)``, which is 20 dB at a 100 Hz
    bandwidth, so the choice matters.

    **Validity envelope.** This is Abraham's eq. (2.77), the large-``M``
    limit of eq. (2.76): the second line "assumes ``sqrt(M)`` is large
    relative to ``phi_d``", and Abraham adds that "unless ``Pd ~ 0.5``, ``M``
    needs to be quite large for (2.77) to be highly accurate: the error is
    generally more than 1 dB for ``M = 10`` and on the order of 0.5 dB for
    ``M = 100``". Benchmarked against the exact noise-normalised energy
    detector (``H0: T ~ Gamma(M, 1)``, ``H1: T ~ (1+S)·Gamma(M, 1)``, so
    ``S = Ginv(1-Pf; M)/Ginv(1-Pd; M) - 1``) the shipped value is
    **optimistic at every operating point** — it asks for less SNR than the
    detector needs, so it reports signal excess that is not there. Measured
    at ``Pf = 1e-6``: ``-1.07 dB`` at ``(Pd=0.9, M=100)``, ``-3.49 dB`` at
    ``M=10``, ``-13.34 dB`` at ``M=1``, and ``-22.88 dB`` at
    ``(Pd=0.99, M=1)``.

    The envelope is policed **at the operating point you asked for**, not by
    a bound fitted over the ``(Pd, Pf, M)`` space: the exact threshold above
    costs two Gamma quantiles (~100 us), so the value is compared against it
    directly and a ``UserWarning`` is issued when the approximation is off by
    more than the ``1 dB`` it claims — naming the actual error and the exact
    threshold. It therefore cannot warn on a value that is inside the promise
    or stay silent on one that is not. Where the quantiles do not resolve, a
    fitted fallback (``M >= max(10, 7*d)``, measured leak-free over ``Pd`` in
    ``[0.05, 0.9995]`` x ``Pf`` in ``[1e-9, 5e-2]`` x ``M`` in ``[1, 3e4]``)
    keeps the check alive rather than letting it disappear.

    For scale: at ``Pd = 0.5, Pf = 1e-4`` the error is ``-0.53 dB`` at
    ``M = 100`` and ``-0.24 dB`` at ``M = 500`` — both inside the promise and
    both silent — while ``M = 10`` and ``M = 1`` are ``-3.5 dB`` and
    ``-13.3 dB`` at ``Pd = 0.9, Pf = 1e-6`` and both warn.
    """
    # Negated admissible condition so NaN is refused: a NaN bandwidth or
    # integration time otherwise returned a silent NaN threshold.
    # ``isfinite`` is the other half of the message's "and finite": the sign
    # test admits ``+inf``, and an infinite bandwidth or integration time drove
    # ``d/(w*t)`` to zero for a ``DT`` of -inf — "no signal at all is needed",
    # the most dangerous direction for a threshold to be wrong in.
    if (not np.isfinite(bandwidth_hz) or not (bandwidth_hz > 0.0)
            or not np.isfinite(integration_time_s)
            or not (integration_time_s > 0.0)):
        raise ConfigurationError(
            f"detection_threshold_energy: bandwidth_hz and integration_time_s"
            f" must be > 0 and finite; got bandwidth_hz={bandwidth_hz!r}, "
            f"integration_time_s={integration_time_s!r}"
        )
    d = detection_index(pd, pf)
    m = float(bandwidth_hz) * float(integration_time_s)
    value = 5.0 * np.log10(d / m)
    # The envelope is policed against the exact threshold at *this* operating
    # point rather than against a bound fitted over the (Pd, Pf, M) space. Two
    # Gamma quantiles cost ~100 us, and the criterion then is the promise: the
    # warning fires when and only when the approximation is off by more than
    # it claims, so it cannot warn on a correct value or stay silent on a
    # wrong one. The fitted `max(10, 7*d)` backs it up only where the exact
    # quantiles do not resolve, so the check never disappears silently.
    exact = _exact_detection_threshold_db(pd, pf, m)
    with np.errstate(invalid="ignore"):
        error_db = value - exact
    if np.isfinite(error_db):
        outside = abs(error_db) >= _DT_APPROXIMATION_TOLERANCE_DB
        detail = (f"it is optimistic by {-error_db:.2f} dB here (exact "
                  f"threshold {exact:.2f} dB)" if error_db < 0 else
                  f"it is off by {error_db:.2f} dB here (exact threshold "
                  f"{exact:.2f} dB)")
    else:
        outside = m < max(10.0, 7.0 * d)
        detail = (f"the exact threshold did not resolve at this operating "
                  f"point, and M = {m:g} is below the fitted fallback bound "
                  f"max(10, 7*d) = {max(10.0, 7.0 * d):g}")
    if outside:
        warnings.warn(
            f"detection_threshold_energy: DT = 5*log10(d/M) is Abraham's "
            f"large-M approximation (eq. 2.77) and at M = "
            f"bandwidth_hz*integration_time_s = {m:g} (pd={pd:g}, pf={pf:g}) "
            f"{detail} — more than the "
            f"{_DT_APPROXIMATION_TOLERANCE_DB:g} dB it claims. The error is "
            f"optimistic at every operating point, so it reports signal "
            f"excess that is not there. Raise the time-bandwidth product, or "
            f"use the exact threshold.",
            UserWarning, stacklevel=2,
        )
    return float(value)
