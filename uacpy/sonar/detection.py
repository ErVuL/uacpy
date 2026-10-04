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

from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.exceptions import (
    ConfigurationError, NumericsWarning, ValidityWarning,
)


def _check_prob(p, name: str) -> float:
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ConfigurationError(f"{name} must be in (0, 1), got {p}.")
    return p


def _check_operating_point(pd: float, pf: float, who: str) -> None:
    """Refuse ``P_D <= P_F``: a detector at or below chance carries no
    information. The detection index squares the deflection, so a swapped
    ``(pf, pd)`` would otherwise return exactly the threshold of the
    intended ``(pd, pf)``."""
    if not pd > pf:
        raise ConfigurationError(
            f"{who}: pd must exceed pf (a detector at P_D <= P_F does no "
            f"better than chance); got pd={pd:g}, pf={pf:g}. Check the "
            f"argument order: pd comes first.")


def deflection_coefficient(pd: float, pf: float) -> float:
    """Gaussian deflection ``d' = Phi^-1(P_D) - Phi^-1(P_F)``.

    The separation (in noise standard deviations) between the signal-present
    and signal-absent decision statistics needed to achieve ``(P_D, P_F)``.

    Parameters
    ----------
    pd : float
        Probability of detection.
    pf : float
        Probability of false alarm.
    """
    pd = _check_prob(pd, "pd")
    pf = _check_prob(pf, "pf")
    _check_operating_point(pd, pf, "deflection_coefficient")
    return float(norm.ppf(pd) - norm.ppf(pf))


def detection_index(pd: float, pf: float) -> float:
    """Urick detection index ``d = (d')^2`` for ``(P_D, P_F)``.

    The square of the deflection: :func:`probability_of_detection` takes
    ``d'``, so its inverse is :func:`deflection_coefficient`, and a ``d``
    from here goes back as ``probability_of_detection(np.sqrt(d), pf)``.

    Parameters
    ----------
    pd : float
        Probability of detection.
    pf : float
        Probability of false alarm.
    """
    return deflection_coefficient(pd, pf) ** 2


def probability_of_detection(deflection, pf):
    """``P_D = Q(Q^-1(P_F) - d')`` for a given deflection and false-alarm rate.

    Gaussian (Neyman-Pearson) detector model: signal-absent and
    signal-present decision statistics are unit-variance Gaussians separated
    by the deflection ``d'``, and the decision threshold is set by the
    false-alarm constraint ``P_F``. ``pf`` may be a scalar or array;
    ``deflection`` broadcasts against it.

    This is **not** the scalar form of
    :func:`uacpy.sonar.sonar_equation.transition_probability_field` —
    that function evaluates a different model, Urick's transition curve
    ``P_D = Phi(SE / sigma_dB)`` (log-normal signal-excess fluctuation,
    ``P_D = 0.5`` pinned at ``SE = 0``, no ``P_F`` argument).

    The inverse of :func:`deflection_coefficient`. A detection index
    ``d = (d')^2`` from :func:`detection_index` is not a deflection: pass
    ``np.sqrt(d)``.

    Parameters
    ----------
    deflection : float or array_like
        Deflection ``d'`` (not the detection index ``d``).
    pf : float or array_like
        Probability of false alarm.
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
            f"{np.unique(pf[~admissible]).tolist()}.")
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
            f"{np.unique(d[~np.isfinite(d)]).tolist()}.")
    return norm.sf(norm.isf(pf) - d)


def roc_curve(deflection: float, n_points: int = 200):
    """ROC ``(P_F, P_D)`` for a Gaussian detector of the given deflection.

    Returns two arrays sampling ``P_F`` logarithmically over ``[1e-6, ~1]``.

    Parameters
    ----------
    deflection : float
        Deflection ``d'``.
    n_points : int, optional
        Points on the curve. Default 200.
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
            f"{deflection!r}.")
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
    ``ValidityWarning`` is issued and the value is an unvalidated extrapolation
    of the fit.

    Parameters
    ----------
    pd : float
        Probability of detection.
    pf : float
        Probability of false alarm.
    n_pulses : int, optional
        Samples integrated non-coherently. Default 1.
    """
    pd = _check_prob(pd, "pd")
    pf = _check_prob(pf, "pf")
    _check_operating_point(pd, pf, "albersheim_snr")
    n = int(n_pulses)
    if n < 1:
        raise ConfigurationError(
            f"albersheim_snr: n_pulses must be >= 1; got {n}.")
    if not (0.1 <= pd <= 0.9 and 1e-7 <= pf <= 1e-3 and n <= 8096):
        warnings.warn(
            f"albersheim_snr: (pd={pd:g}, pf={pf:g}, n_pulses={n}) is outside "
            f"the envelope Albersheim's equation was fitted over — "
            f"0.1 <= pd <= 0.9, 1e-7 <= pf <= 1e-3, 1 <= N <= 8096 "
            f"(Richards 2014) — so the ~0.2 dB accuracy bound does not apply "
            f"and the value is an extrapolation of an empirical fit.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    a = np.log(0.62 / pf)
    b = np.log(pd / (1.0 - pd))
    snr_dB = (
        -5.0 * np.log10(n)
        + (6.2 + 4.54 / np.sqrt(n + 0.44))
        * np.log10(a + 0.12 * a * b + 1.7 * b)
    )
    return float(snr_dB)


#: How far the shipped large-M approximation may sit from the exact
#: energy-detector threshold before ``detection_threshold_energy`` warns.
#: Abraham's own accuracy statement for eq. (2.77) is quoted in dB, so the
#: envelope is policed in the units it is promised in.
_DT_APPROXIMATION_TOLERANCE_DB = 1.0


def _exact_detection_threshold_dB(pd: float, pf: float, m: float) -> float:
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


def _check_n_looks(n_looks, who: str) -> float:
    """Validate a look count: at least one, finite."""
    n = float(np.asarray(n_looks, dtype=float))
    if not np.isfinite(n) or n < 1.0:
        raise ConfigurationError(
            f"{who}: n_looks must be a finite count of at least 1; got "
            f"{n_looks!r}. It is the number of INDEPENDENT looks the "
            f"detector maximises over — for a beam scan that is the "
            f"resolution cells in the sector (one beam is lambda/L wide in "
            f"sin(theta)), not the number of steering angles computed."
        )
    return n


def per_look_false_alarm(pf_scan: float, n_looks: float) -> float:
    """Per-look ``P_F`` that holds a whole scan's false-alarm rate at ``pf_scan``.

    A detector that forms many looks and reports the largest false-alarms if
    *any* look does, so its rate is the scan's, not one look's. Abraham puts
    it directly: "searching over N independent sonar resolution cells for a
    signal results in an N-fold increase in the single-resolution-cell
    probability of false alarm ... the single-resolution-cell probability of
    false alarm must be set N times smaller than that desired"
    (*Underwater Acoustic Signal Processing*, 8.2.11). Inverting
    :func:`scan_false_alarm`::

        pf_look = 1 - (1 - pf_scan) ** (1 / n_looks)

    ``n_looks`` counts **independent** looks. For a line array scanning a
    sector, one beam is ``lambda/L`` wide in ``sin(theta)``, so the sector
    holds ``sin-span / (lambda/L)`` resolution cells however finely the
    steering grid is sampled. Overlapping (shaded) beams are correlated, so
    using the orthogonal-beam count is the conservative choice: it sets a
    stricter threshold than the true dependence requires.

    Parameters
    ----------
    pf_scan : float
        The scan-level false-alarm probability.
    n_looks : float
        Independent looks in the scan.
    """
    pf = _check_prob(pf_scan, "per_look_false_alarm: pf_scan")
    n = _check_n_looks(n_looks, "per_look_false_alarm")
    # log1p/expm1 rather than the literal formula: at pf=1e-12 over 1e5
    # looks, ``1 - (1 - pf) ** (1/n)`` rounds to exactly 0.0, and
    # scan_false_alarm then raises on this function's own output.
    return float(-np.expm1(np.log1p(-pf) / n))


def scan_false_alarm(pf_look: float, n_looks: float) -> float:
    """False-alarm rate of a max-over-looks detector, ``1 - (1 - pf)**n``.

    The rate a scan actually achieves when each of ``n_looks`` independent
    looks is thresholded at ``pf_look``. This is Abraham's Equation (8.111),
    the probability of one or more false alarms in N statistically
    independent resolution cells,
    ``Pr{at least one FA} = 1 - [1 - P_f]^N ~ N*P_f`` (*Underwater Acoustic
    Signal Processing*, 8.2.11, "False-Alarm Rate"). Inverse of
    :func:`per_look_false_alarm`,
    which is the one to use when setting a threshold from a required
    scan-level rate.

    "Looks" are resolution cells of any kind, not only beams: Ainslie counts
    "100 beams and 1000 frequencies ... 10^5 detection opportunities (and
    therefore also 10^5 false alarm opportunities) every second"
    (*Sonar Performance Modelling*, 7). :func:`independent_beams
    <uacpy.acoustic_signal.independent_beams>` counts only the spatial
    factor.

    Parameters
    ----------
    pf_look : float
        The per-look false-alarm probability.
    n_looks : float
        Independent looks in the scan.
    """
    pf = _check_prob(pf_look, "scan_false_alarm: pf_look")
    n = _check_n_looks(n_looks, "scan_false_alarm")
    return float(-np.expm1(n * np.log1p(-pf)))      # see per_look_false_alarm


def detection_threshold_energy(
    pd: float, pf: float, bandwidth_hz: float, integration_time_s: float,
    *, exact: bool = True,
) -> float:
    """Detection threshold (dB) for an incoherent energy detector.

    By default the exact threshold of the noise-normalised energy detector,
    ``DT = 10*log10(S)`` with ``S = Ginv(1-Pf; M)/Ginv(1-Pd; M) - 1``, where
    ``M = w*t`` is the processing time-bandwidth product and ``Ginv`` the
    Gamma(M) quantile (Abraham, *Underwater Acoustic Signal Processing*,
    eq. 2.76). ``exact=False`` returns Abraham's large-``M`` form, eq. (2.77),
    ``DT = 5*log10(d / M)`` with ``d`` the detection index (§9.2.3.1 /
    §9.2.11): the textbook line, straight at −5 dB per decade of ``M``, which
    the exact threshold approaches from above as ``M`` grows. Either way more
    incoherent integration relaxes the required SNR.

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

    **The large-M form's validity envelope** (``exact=False``). Eq. (2.77) is
    the large-``M`` limit of eq. (2.76). It is optimistic (asks for less SNR than the exact
    noise-normalised energy detector needs) at every operating point, and
    accurate to ~1 dB only for ``M`` of order 100 and above — from
    ``M`` ≈ 20–50 at ``Pd = 0.5``, the least demanding case. At ``Pf = 1e-6``
    and ``M = 1`` (a single-look short pulse) it is optimistic by 6.0 dB at
    ``Pd = 0.5``, 13.3 dB at ``Pd = 0.9`` and 22.9 dB at ``Pd = 0.99``; at
    ``M = 10`` by 2.0, 3.5 and 4.9 dB. With ``exact=False`` the exact
    threshold is still evaluated at the requested ``(Pd, Pf, M)``, and a
    ``ValidityWarning`` names the error when, rounded to the 0.01 dB it is
    printed with, it exceeds 1 dB.

    Both forms are the same Gaussian-fluctuating-signal model and the same
    unitless power ratio. Where the Gamma quantiles do not resolve a
    threshold the exact form returns NaN with a warning rather than falling
    back to the approximation.

    Parameters
    ----------
    pd : float
        Probability of detection.
    pf : float
        Probability of false alarm.
    bandwidth_hz : float
        Processing bandwidth ``w`` (Hz).
    integration_time_s : float
        Integration time ``t`` (s).
    exact : bool, optional
        ``True`` (default): the exact threshold. ``False``: eq. (2.77),
        warning outside its 1 dB envelope.
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
            f"integration_time_s={integration_time_s!r}."
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
    exact_dB = _exact_detection_threshold_dB(pd, pf, m)
    if exact:
        if not np.isfinite(exact_dB):
            warnings.warn(
                f"detection_threshold_energy: the exact threshold did not "
                f"resolve at pd={pd:g}, pf={pf:g}, M={m:g} (the Gamma "
                f"quantiles give no positive SNR); returning NaN.",
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
            return float("nan")
        return float(exact_dB)
    with np.errstate(invalid="ignore"):
        error_dB = value - exact_dB
    if np.isfinite(error_dB):
        # Compared as printed, so a warning never reads "1.00 dB — more
        # than the 1 dB it claims".
        outside = round(abs(error_dB), 2) > _DT_APPROXIMATION_TOLERANCE_DB
        detail = (f"it is optimistic by {-error_dB:.2f} dB here (exact "
                  f"threshold {exact_dB:.2f} dB)" if error_dB < 0 else
                  f"it is off by {error_dB:.2f} dB here (exact threshold "
                  f"{exact_dB:.2f} dB)")
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
            f"pass exact=True for the exact threshold.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    return float(value)
