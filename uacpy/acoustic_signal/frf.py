"""Frequency-response estimation. Four functions estimate a transfer
function from one input/output record pair and return an :class:`FRFResult`:
:func:`frf_welch` (H1/H2 from Welch spectra, with the coherence), :func:`etfe`
(the empirical transfer function estimate), :func:`periodic_etfe` (the ETFE
of the period-averaged records) and :func:`lsfir` (a least-squares FIR fit,
with order selection). :class:`FRF` is the configuration of one of them, run
on a record pair or averaged over a block of measurements.
"""

from __future__ import annotations

import warnings
from collections import namedtuple

import numpy as np
import scipy.signal as _sig
from scipy.linalg import get_lapack_funcs, toeplitz
from uacpy.acoustic_signal._results import PlottedResult
from uacpy.acoustic_signal.windows import _DEFAULT_NPERSEG, _default_noverlap
from uacpy.core.exceptions import (
    ConfigurationError, NumericsWarning, ValidityWarning,
)
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._repr import SettingsRepr
from uacpy.core._validate import require_positive_finite_scalar


# ──────────────────────────────────────────────────────────────────────
# Frequency-response estimation
#
# Functions of one record pair; FRF holds the configuration that runs them.
# ──────────────────────────────────────────────────────────────────────

def _info_matrices(u, y, N, order):
    """Normal equations ``(Minfo, Vinfo)`` of the least-squares FIR fit.

    Assembled from *circular* correlations with the wrap-around terms then
    subtracted, which avoids forming the ``(N - order + 1, order)`` design
    matrix:

    * ``phiuu[i]`` and ``phiuy[i]`` are the circular correlations of ``u`` with
      itself and with ``y`` at lag ``i`` (each shift of ``u_temp`` rotates the
      record by one sample), so ``toeplitz(phiuu)`` is the circular
      autocorrelation matrix.
    * ``W``'s ``order - 1`` rows hold exactly the wrapped segments of ``u``
      that a circular correlation counts and a linear one does not.

    The results are therefore ``X.T @ X`` and ``X.T @ y[order - 1:]`` for the
    covariance-method design matrix ``X[n, k] = u[order - 1 + n - k]``: the fit
    uses only the samples over which the whole filter overlaps the data, and
    assumes no prehistory before ``u[0]``.
    """
    u_temp = u[:N].copy()
    phiuu = np.zeros(order)
    phiuy = np.zeros(order)
    for i in range(order):
        phiuu[i] = np.dot(u[:N], u_temp)
        phiuy[i] = np.dot(y[:N], u_temp)
        u_temp = np.concatenate(([u_temp[-1]], u_temp[:-1]))  # rotate right

    A = toeplitz(phiuu)
    u_flipped = np.flip(u[:N]).copy()
    W = np.zeros((order - 1, order))
    for i in range(order - 1):
        u_flipped = np.concatenate(([u_flipped[-1]], u_flipped[:-1]))
        W[i, :] = u_flipped[:order]

    return A - np.dot(W.T, W), phiuy - np.dot(W.T, y[: order - 1])


#: Reciprocal condition number at which the LU solve of the normal equations
#: stops carrying a correct digit: ``cond(Minfo) * eps >= 1``. The bound is on
#: a *reciprocal condition number*, which is dimensionless and invariant to the
#: amplitude scale of the data — ``rcond(c*Minfo) == rcond(Minfo)`` exactly,
#: because both norms in it scale by the same ``c`` — so the branch below
#: cannot make the fit depend on whether a record is read in Pa or in µPa.
_INFO_RCOND_FLOOR = np.finfo(float).eps


def _solve_info_matrices(Minfo, Vinfo, order):
    """``(g, rcond)`` from the normal equations, minimum-norm when singular.

    ``Minfo`` is ``X.T @ X`` for the design matrix ``X``, so its condition
    number is ``cond(X)**2`` and a band-limited excitation squares its way past
    what float64 can represent: an ordinary 100 Hz - 20 kHz sweep at
    fs = 48 kHz reaches ``cond(X) = 4.2e11`` and ``cond(Minfo) = 6.4e18`` at
    order 512, ``FRF``'s default ``order``, where the LU solve returns an
    impulse response 34x over-scale and a frequency response 5.2 dB wrong
    across the excited band (measured).

    ``rcond`` is LAPACK's 1-norm reciprocal condition estimate, taken from the
    same LU factorization that solves the system, so it costs one extra
    ``O(order**2)`` pass rather than a second factorization. Above
    ``_INFO_RCOND_FLOOR`` the LU solution is returned. That is the same
    algorithm ``np.linalg.solve`` runs — LAPACK ``getrf`` + ``getrs`` — but not
    necessarily the same *build* of it: numpy and scipy ship separate OpenBLAS
    binaries, so the two agree to a few ULP of the solution rather than bit for
    bit. Measured over 60 well-conditioned solves (3 excitations x 10 orders x
    2 record lengths, ``rcond`` down to 3.5e-11): identical in 30 of them and
    never further apart than **2.75 eps of the peak coefficient**. At or below
    ``_INFO_RCOND_FLOOR`` the equations are numerically singular, the LU
    coefficients carry no correct digit, and the *minimum-norm* least-squares
    solution of the same equations is returned instead — the truncation drops
    the directions of ``X`` whose singular values fall below ``sqrt(eps)``
    times the largest, which are exactly the ones the ``X.T @ X`` product
    cannot represent. The user is warned there, because the answer that comes
    back is then a choice of regularization rather than a fit the data
    determines.

    An *exactly* singular ``Minfo`` still raises ``LinAlgError``, the way
    ``np.linalg.solve`` does, so the order-selection loop's skip and the typed
    degenerate-input errors keep firing on an all-zero or constant input.
    """
    getrf, gecon, getrs = get_lapack_funcs(('getrf', 'gecon', 'getrs'),
                                           (Minfo, Vinfo))
    anorm = float(np.max(np.sum(np.abs(Minfo), axis=0))) if Minfo.size else 0.0
    lu, piv, info = getrf(Minfo)
    if info != 0:
        raise np.linalg.LinAlgError("Singular matrix.")
    rcond = float(gecon(lu, anorm, norm='1')[0])

    if rcond > _INFO_RCOND_FLOOR:
        return getrs(lu, piv, Vinfo)[0], rcond

    # numpy's own least-squares cutoff, spelled out rather than left to the
    # ``rcond=None`` default: singular values of Minfo below order*eps times
    # the largest are treated as zero.
    g = np.linalg.lstsq(Minfo, Vinfo,
                        rcond=Minfo.shape[0] * np.finfo(float).eps)[0]
    warnings.warn(
        f"lsfir: the information matrix at FIR order {order} is "
        f"numerically singular (reciprocal condition number {rcond:.2e} <= "
        f"{_INFO_RCOND_FLOOR:.2e}), so its LU solution carries no correct "
        f"digit; the impulse response returned is the minimum-norm "
        f"least-squares solution of the same normal equations, and the "
        f"frequency response is undetermined wherever the input does not "
        f"excite. Lower the FIR order, or excite the whole band up to "
        f"Nyquist.",
        NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )
    return g, rcond


_ETFE_REL_FLOOR = 1e-12          # relative to max|X| over the record


def _etfe_divide(Y, X, who: str, quantity="the transfer function",
                 denominator="input energy"):
    """``Y/X`` with bins whose denominator is numerically zero returned as nan.

    The threshold is relative to ``max|X|`` because a transfer function is a
    ratio: adding an absolute epsilon made the estimate at a numerically empty
    bin a function of the units the caller happened to use — the same signal
    in Pa and in µPa gave answers 1e12 apart, and a bin with no excitation
    came back as a finite ~1/eps number rather than as undefined.
    ``quantity`` and ``denominator`` name the estimate and its denominator
    spectrum in the warning (the Welch H2 and coherence guards divide by
    spectra other than the input's).
    """
    peak = float(np.max(np.abs(X))) if X.size else 0.0
    excited = (np.abs(X) > _ETFE_REL_FLOOR * peak if peak > 0
               else np.zeros(X.shape, bool))
    if not excited.all():
        # Each estimator reaches this divide at its own depth — called
        # directly, or through ``FRF.compute`` — so no single frame count
        # reaches the user: a hand-counted ``stacklevel=3`` named a line in
        # this module (measured). ``skip_file_prefixes`` counts no frames at
        # all — it walks to the first file outside the package.
        warnings.warn(
            f"{who}: {int((~excited).sum())} of {excited.size} frequency "
            f"bins carry no {denominator} (denominator magnitude <= "
            f"{_ETFE_REL_FLOOR:g} of its peak); {quantity} is undefined "
            f"there and is returned as nan. Excite the whole band, or "
            f"restrict the analysis to the excited band.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    return np.where(excited, Y / np.where(excited, X, 1.0), np.nan)


#: Transfer-function estimators :func:`frf_welch` computes.
_FRF_ESTIMATORS = ("H1", "H2")


def _require_estimator(estimator, who):
    """Refuse an estimator name :func:`frf_welch` does not compute, naming the valid
    ones; a silent fall-through to H1 hands an H2 caller the estimate biased
    low by the input noise."""
    if estimator not in _FRF_ESTIMATORS:
        raise ConfigurationError(
            f"{who}: unknown estimator={estimator!r}; valid: "
            f"{', '.join(repr(e) for e in _FRF_ESTIMATORS)} (case-sensitive).")
    return estimator


#: Order-selection criteria :func:`lsfir` accepts in place of an order.
_ORDER_CRITERIA = ("AIC", "BIC", "FPE", "CP")


class FRFResult(PlottedResult,
                namedtuple("FRFResult", "frequencies transfer_function")):
    """A transfer-function estimate: the complex ``transfer_function`` on
    ``frequencies`` (Hz), so ``frequencies, transfer_function = ...``
    unpacks it; :meth:`plot` draws it with
    :func:`~uacpy.plot.plot_frf`.

    What else the estimate holds rides on attributes, each ``None`` where
    its method produces no such value:

    ``method``
        The estimator that ran, by its :class:`FRF` name: ``'welch'``
        (:func:`frf_welch`), ``'etfe'`` (:func:`etfe`), ``'p_etfe'``
        (:func:`periodic_etfe`) or ``'ls_fir'`` (:func:`lsfir`).
    ``estimator``
        ``'H1'`` or ``'H2'`` (``'welch'``).
    ``coherence``
        The magnitude-squared coherence on ``frequencies`` (``'welch'``).
    ``impulse_response``, ``order``, ``criterion``
        The fitted FIR, its order, and the criterion that selected the order
        (``None`` when the order was given) (``'ls_fir'``).
    ``rcond``, ``information_matrix``, ``information_vector``
        The normal equations the FIR was solved from and their reciprocal
        condition number: small means the fit is poorly determined, and at
        or below ``2.2e-16`` it is a minimum-norm choice (``'ls_fir'``).

    From :meth:`FRF.compute` on a 2-D block, ``transfer_function`` and
    ``coherence`` are the means over the measurements, and each ``'ls_fir'``
    attribute is a tuple with one entry per measurement: fits of different
    orders have no mean.
    """

    _attrs = ("method", "estimator", "coherence", "impulse_response",
              "order", "criterion", "rcond", "information_matrix",
              "information_vector")
    _plotter = "plot_frf"
    _plot_fields = ("frequencies", "transfer_function")

    def __new__(cls, frequencies, transfer_function, *, method,
                estimator=None, coherence=None, impulse_response=None,
                order=None, criterion=None, rcond=None,
                information_matrix=None, information_vector=None):
        self = super().__new__(cls, frequencies, transfer_function)
        self.method = str(method)
        self.estimator = estimator
        self.coherence = coherence
        self.impulse_response = impulse_response
        self.order = order
        self.criterion = criterion
        self.rcond = rcond
        self.information_matrix = information_matrix
        self.information_vector = information_vector
        return self

    def _field_units(self):
        # The ratio of an output to an input in the same unit.
        return {"frequencies": "Hz", "transfer_function": ""}


def _paired_records(x, y, who):
    """``x`` and ``y`` as one input/output record pair: 1-D, of one length,
    not empty. Every estimator pairs ``x[n]`` with ``y[n]`` sample by
    sample."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1:
        raise ConfigurationError(
            f"{who}: x and y must be 1-D records; got shapes {x.shape} and "
            f"{y.shape}.",
            remediation="Pass one measurement, or average a block of them "
                        "with FRF().compute(x, y, sample_rate).")
    if x.shape != y.shape:
        raise ConfigurationError(
            f"{who}: x and y must hold records of the same length; "
            f"got {x.size} and {y.size} samples.",
            remediation="Trim both records to a common span before "
                        "computing the FRF.")
    if x.size == 0:
        raise ConfigurationError(f"{who}: x and y hold no samples.")
    return x, y


def frf_welch(x, y, sample_rate, *, estimator="H1", nperseg=_DEFAULT_NPERSEG,
              noverlap=None, **welch_options):
    """The H1 or H2 transfer function, and the coherence, from Welch
    spectra of one input/output record pair.

    Dedicated to stationary signals. Under the scipy convention ``Sxy =
    csd(x, y) = E[X*·Y]``::

        H1(f) = Sxy(f) / Sxx(f)      unbiased by noise on the output
        H2(f) = Syy(f) / Syx(f)      unbiased by noise on the input

    and the coherence ``|Sxy|² / (Sxx·Syy)`` says how much of the output is
    linearly explained by the input at each frequency. Bins where a
    denominator spectrum (``Sxx`` for H1, ``Syx`` for H2, ``Sxx·Syy`` for the
    coherence) is numerically zero relative to its peak are returned as nan
    with a warning, as in :func:`etfe`.

    Parameters
    ----------
    x, y : array_like
        Input (reference) and output records, 1-D, of one length.
    sample_rate : float
        Sampling frequency (Hz).
    estimator : {'H1', 'H2'}
    nperseg : int
        Welch segment length; the frequency grid is its rfft grid.
    noverlap : int, optional
        Unset is the window's overlap, as on
        :func:`uacpy.acoustic_signal.welch`: half a segment under scipy's
        default hann, the flat-top overlap under ``window='flattop'``.
        Overlapping averages about twice as many Hann segments from the same
        record, so the random error of H1 and of the coherence falls (to
        about three quarters of the unoverlapped error on a known FIR
        system).
    **welch_options
        Further :func:`scipy.signal.welch` / :func:`scipy.signal.csd`
        keywords (``window``, ``detrend``, ...). ``fs`` and ``scaling`` are
        refused: the sample rate is ``sample_rate``, and the scaling is
        fixed to ``'density'`` — the transfer function is a spectral ratio,
        so a common scaling cancels.

    Returns
    -------
    FRFResult
        ``method='welch'``, with ``estimator`` and ``coherence``.
    """
    estimator = _require_estimator(estimator, "frf_welch")
    reserved = {"fs", "scaling"} & set(welch_options)
    if reserved:
        raise ConfigurationError(
            f"frf_welch: {sorted(reserved)} cannot be passed as Welch "
            "options — the sample rate is the sample_rate argument, and the "
            "spectral scaling is fixed to 'density' internally (the transfer "
            "function is a spectral ratio, so a common scaling cancels).")
    x, y = _paired_records(x, y, "frf_welch")
    sample_rate = require_positive_finite_scalar(
        sample_rate, "frf_welch", "sample_rate", " Hz")
    params = dict(nperseg=nperseg, noverlap=noverlap, **welch_options)
    if params["noverlap"] is None:
        segment = min(int(params["nperseg"]), x.size)
        params["noverlap"] = _default_noverlap(
            params.get("window", "hann"), segment)
    freqs, Pxx = _sig.welch(x, sample_rate, scaling="density", **params)
    _, Pyy = _sig.welch(y, sample_rate, scaling="density", **params)
    # ``Pxy`` holds csd(y, x) = Syx = conj(Sxy); the H1 expression
    # ``conj(Pxy)/Pxx`` is the textbook ``Sxy/Sxx``.
    _, Pxy = _sig.csd(y, x, sample_rate, scaling="density", **params)
    # Each division masks bins whose denominator spectrum is numerically
    # zero (relative to its own peak) to nan with a warning, the same
    # policy _etfe_divide applies to the ETFE estimators; a zero or
    # constant record yields masked nan rather than dividing through to
    # inf/nan noise.
    if estimator == "H2":
        tf = _etfe_divide(Pyy, Pxy, "frf_welch",
                          denominator="cross-spectral energy")
    else:
        tf = _etfe_divide(np.conj(Pxy), Pxx, "frf_welch")
    coh = _etfe_divide(np.abs(Pxy) ** 2, Pxx * Pyy, "frf_welch",
                       quantity="the coherence",
                       denominator="input or output energy")
    return FRFResult(freqs, tf, method="welch", estimator=estimator,
                     coherence=coh)


def etfe(x, y, sample_rate):
    """The empirical transfer function estimate ``rfft(y) / rfft(x)`` of one
    input/output record pair.

    The ETFE is unbiased but **not consistent**: it spends one complex
    datum per frequency bin, so its variance does not fall as the record
    grows and the estimate stays noisy bin to bin no matter how much data
    is supplied. Measured on a known 4-tap FIR driven by white noise, its
    worst-case error over 100-3500 Hz is ~0.6 where :func:`frf_welch`
    reaches 5e-4. Use it for a quick look or on a clean swept/periodic
    excitation; use :func:`periodic_etfe` (averages over periods),
    :func:`frf_welch` or :func:`lsfir` when the estimate has to be accurate.

    Parameters
    ----------
    x, y : array_like
        Input (reference) and output records, 1-D, of one length.
    sample_rate : float
        Sampling frequency (Hz).

    Returns
    -------
    FRFResult
        ``method='etfe'``, on the rfft grid of the whole record,
        ``k * sample_rate / len(x)`` up to Nyquist. That is finer than the
        ``nperseg`` grid :func:`frf_welch`, :func:`periodic_etfe` and
        :func:`lsfir` share: the ETFE spends one raw rfft bin per frequency,
        so its grid is set by the record length.
    """
    x, y = _paired_records(x, y, "etfe")
    sample_rate = require_positive_finite_scalar(
        sample_rate, "etfe", "sample_rate", " Hz")
    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(x.size, d=1 / sample_rate)
    tf = _etfe_divide(Y, X, "etfe")
    return FRFResult(freqs, tf, method="etfe")


def periodic_etfe(x, y, sample_rate, *, period):
    """The ETFE of one input/output record pair averaged over whole periods
    of a periodic excitation.

    Coherently averaging the *time records* over whole periods (not their
    spectra) is what buys back the consistency the raw :func:`etfe` lacks:
    the periodic part adds in phase while independent noise averages down.
    Samples after the last whole period are not used.

    Parameters
    ----------
    x, y : array_like
        Input (reference) and output records, 1-D, of one length.
    sample_rate : float
        Sampling frequency (Hz).
    period : int
        The excitation's period in samples; the frequency grid is its rfft
        grid, ``k * sample_rate / period`` up to Nyquist.

    Returns
    -------
    FRFResult
        ``method='p_etfe'``.
    """
    x, y = _paired_records(x, y, "periodic_etfe")
    sample_rate = require_positive_finite_scalar(
        sample_rate, "periodic_etfe", "sample_rate", " Hz")
    period = int(period)
    n_periods = x.size // period if period > 0 else 0
    if n_periods < 1:
        raise ConfigurationError(
            f"periodic_etfe: signal length must be at least one period; got "
            f"len(x)={x.size} samples, period={period} samples.")
    x_avg = np.mean(x[: n_periods * period].reshape(n_periods, period),
                    axis=0)
    y_avg = np.mean(y[: n_periods * period].reshape(n_periods, period),
                    axis=0)
    X = np.fft.rfft(x_avg)
    Y = np.fft.rfft(y_avg)
    freqs = np.fft.rfftfreq(period, d=1 / sample_rate)
    tf = _etfe_divide(Y, X, "periodic_etfe")
    return FRFResult(freqs, tf, method="p_etfe")


def lsfir(x, y, sample_rate, *, order, n_samples=None, max_order=4096,
          stop_count=50, nperseg=_DEFAULT_NPERSEG):
    """A least-squares FIR fit of one input/output record pair, with its
    frequency response.

    The fit solves the covariance-method normal equations (information
    matrix and vector) for the FIR taps; given a criterion instead of an
    order it selects the order.

    Parameters
    ----------
    x, y : array_like
        System input (reference) and output, 1-D, of one length.
    sample_rate : float
        Sampling rate in Hz.
    order : int or {'AIC', 'BIC', 'FPE', 'CP'}
        The FIR order, or the criterion that selects it: finite-sample AIC,
        BIC, FPE, or Mallows' Cp.
    n_samples : int, optional
        The leading samples the fit uses (``order <= n_samples``); all of
        them when unset.
    max_order : int
        Highest order a criterion search tries.
    stop_count : int
        A criterion search stops after this many consecutive orders with no
        improvement.
    nperseg : int
        The frequency response is evaluated on the rfft grid of ``nperseg``
        samples, so it lines up bin for bin with :func:`frf_welch` and
        :func:`periodic_etfe`.

    Returns
    -------
    FRFResult
        ``method='ls_fir'``, with ``impulse_response``, ``order``,
        ``criterion``, ``rcond``, ``information_matrix`` and
        ``information_vector``.

    Notes
    -----
    The fit solves the normal equations ``X.T @ X`` of the covariance
    design matrix, whose condition number is ``cond(X)**2``. The result's
    ``rcond`` is the reciprocal condition number of the system the
    returned impulse response came out of; a warning names the order when
    it falls to where the equations are numerically singular, which an
    order longer than the excited band can support reaches easily — a
    100 Hz - 20 kHz sweep at fs = 48 kHz does it at order 512.
    """
    u, y = _paired_records(x, y, "lsfir")
    sample_rate = require_positive_finite_scalar(
        sample_rate, "lsfir", "sample_rate", " Hz")
    N = u.size if n_samples is None else int(n_samples)
    criterion = order if order in _ORDER_CRITERIA else None
    m = order

    if criterion is not None:
        # Model order selection
        m_max = min(max_order, N - 1)
        best_score = np.inf
        best_m = 1
        best_g = None
        count = 0
        # "Numerically exact fit" is judged against the output power, so the
        # decision is invariant to the amplitude scale of the data.
        exact_tol = np.finfo(float).eps * float(np.mean(y[:N] ** 2))

        if m == "CP":
            # Mallows' Cp: σ̂² is the residual variance of a low-bias
            # reference fit (order well above any plausible true order,
            # well below N), not the raw output variance.
            full_model_m = min(m_max, max(2, N // 4))
            # Every Cp score below divides by sigma2, so the reference fit
            # cannot be skipped the way the candidate loop skips a singular
            # order — continuing past it would leave sigma2 unbound. A
            # degenerate input is reported here instead, matching the typed
            # error the other criteria reach after their loop.
            try:
                g_full = np.linalg.solve(
                    *_info_matrices(u, y, N, full_model_m))
            except np.linalg.LinAlgError as exc:
                raise ConfigurationError(
                    f"lsfir: criterion 'CP' needs a reference "
                    f"fit at order {full_model_m} to set its residual "
                    f"variance, and that fit gave a singular information "
                    f"matrix. The input is degenerate (constant, "
                    f"all-zero, or too short); use a persistently exciting "
                    f"input, pass an explicit integer order, or choose "
                    f"'AIC'/'BIC'/'FPE', which select an order without a "
                    f"reference fit.") from exc
            y_hat_full = np.convolve(u[:N], g_full, mode="full")[:N]
            sigma2 = np.sum((y[:N] - y_hat_full) ** 2) / (N - full_model_m)

        for m_candidate in range(1, m_max + 1):
            try:
                Minfo, Vinfo = _info_matrices(u, y, N, m_candidate)
                g = np.linalg.solve(Minfo, Vinfo)
                # np.convolve assumes u is zero before index 0, so the
                # first m_candidate - 1 residuals are start-up transients
                # outside the covariance-method fit window and count toward
                # sse for every candidate order.
                y_hat = np.convolve(u[:N], g, mode="full")[:N]
                residuals = y[:N] - y_hat
                sse = np.sum(residuals**2) / (N - m_candidate)

                if sse <= exact_tol:
                    # Residual at the rounding floor: this order explains
                    # the data exactly, and it is the lowest one that does.
                    best_m = m_candidate
                    best_g = g
                    break

                if m == "AIC":  # AICF
                    # Finite-sample AIC variant on the unbiased residual
                    # variance sse = SSE/(N-m): log(sse) +
                    # (1 + m/(N-m))/(1 - m/(N-m)). Textbook AIC scores
                    # the ML variance SSE/N with penalty 2m/N; since
                    # log(SSE/(N-m)) = log(SSE/N) + m/N + O(m²/N²), this
                    # penalty is heavier by m/N (a mild bias toward
                    # lower orders that vanishes as N >> m).
                    score = np.log(sse) + (1 + m_candidate / (N - m_candidate)) / (
                        1 - m_candidate / (N - m_candidate)
                    )

                elif m == "FPE":  # FPEF
                    # Finite-sample FPE on the same unbiased sse:
                    # sse·(1 + m/(N-m))/(1 - m/(N-m)). Textbook FPE is
                    # (SSE/N)·(1 + m/N)/(1 - m/N); the SSE/(N-m) inside
                    # makes this penalty heavier by m/N, as for AIC.
                    score = (
                        sse
                        * (1 + m_candidate / (N - m_candidate))
                        / (1 - m_candidate / (N - m_candidate))
                    )

                elif m == "CP":  # Mallows' Cp
                    score = (
                        sse * (N - m_candidate) / sigma2 - N + 2 * (m_candidate + 1)
                    )

                elif m == "BIC":  # Bayesian Information Criterion
                    # On the unbiased sse, so heavier than the ML-variance
                    # form log(SSE/N) + m·log(N)/N by m/N, as for AIC.
                    score = np.log(sse) + (m_candidate * np.log(N)) / N

                if score < best_score:
                    best_score = score
                    best_m = m_candidate
                    best_g = g
                    count = 0
                else:
                    count += 1

                if best_g is not None and count >= stop_count:
                    break  # Stop search early

            except np.linalg.LinAlgError:
                continue  # Skip singular matrices

        if best_g is None:
            raise ConfigurationError(
                f"lsfir: no FIR order in 1..{m_max} could be "
                f"fitted with criterion {m!r} — every candidate gave a "
                "singular information matrix. The input is degenerate "
                "(constant, all-zero, or too short); use a persistently "
                "exciting input, or pass an explicit integer order."
            )
        m = best_m
        # Re-solve the normal equations for the order actually selected
        # through the rank-revealing path. The search above compares orders
        # under one solve, and its LU is also its identifiability screen: a
        # candidate whose equations are singular is skipped, and one that is
        # merely near-singular fits worse and scores worse, so neither is
        # selected. The order that *is* selected then gets its coefficients
        # from _solve_info_matrices, which returns the same numbers bit for
        # bit whenever the LU is trustworthy and says so when it is not.
    else:
        m = int(m)
        if m > N:
            raise ConfigurationError(
                f"lsfir: FIR order ({m}) must be <= n_samples "
                f"({N}) — the fit solves for that many coefficients from "
                "n_samples data points. Reduce the order, raise n_samples, "
                "or pass a selection criterion ('AIC', 'BIC', 'FPE', 'CP') "
                "to choose the order automatically.")
    Minfo, Vinfo = _info_matrices(u, y, N, m)
    g, rcond = _solve_info_matrices(Minfo, Vinfo, m)

    # Frequency response on the nperseg rfft grid, so an ls_fir result
    # lines up bin-for-bin with a welch or p_etfe one ('etfe' alone uses
    # the full-record grid instead).
    freqs = np.fft.rfftfreq(int(nperseg), d=1.0 / sample_rate)
    _, h = _sig.freqz(g, worN=freqs, fs=sample_rate)
    return FRFResult(freqs, h, method="ls_fir", impulse_response=g,
                     order=int(m), criterion=criterion, rcond=rcond,
                     information_matrix=Minfo, information_vector=Vinfo)


#: The estimator each :class:`FRF` method runs.
_FRF_METHODS = ("welch", "ls_fir", "etfe", "p_etfe")


class FRF(SettingsRepr):
    """A frequency-response estimator's configuration, set once and run on
    each record pair or block :meth:`compute` is given.

    The estimation itself is :func:`frf_welch`, :func:`etfe`,
    :func:`periodic_etfe` and :func:`lsfir`, which take one record pair and
    return an :class:`FRFResult`; this class holds which one runs and with
    what settings, and averages it over the measurements of a 2-D block. It
    keeps no result: every :meth:`compute` returns its own.
    """

    def __init__(self, method="welch", estimator="H1", order=512, **kwargs):
        """
        Parameters
        ----------
        method : str
            Estimation method. One of:

            - ``'welch'`` -- :func:`frf_welch`, dedicated to stationary
              signals, with coherence.
            - ``'ls_fir'`` -- :func:`lsfir`, a least-squares impulse
              response.
            - ``'etfe'`` -- :func:`etfe` over the whole signal.
            - ``'p_etfe'`` -- :func:`periodic_etfe` over segments of
              ``nperseg`` samples.
        estimator : str
            Estimator type for ``'welch'``. One of:

            - ``'H1'`` -- minimizes the effect of noise introduced at the system output.
            - ``'H2'`` -- minimizes the effect of noise introduced at the system input.
        order : int or str
            FIR order for ``'ls_fir'``, or the criterion that selects it
            (``'AIC'``, ``'BIC'``, ``'FPE'``, ``'CP'``) — :func:`lsfir`'s
            ``order``.
        **kwargs
            Welch options (``nperseg``, ``noverlap``, ``window``, ...), as on
            :func:`frf_welch`; ``nperseg`` is also the ``'ls_fir'`` grid and
            the ``'p_etfe'`` period.
        """
        self.params = {
            "nperseg": _DEFAULT_NPERSEG,
            "noverlap": None,
        }
        # 'fs' and 'scaling' are set internally at every welch/csd call site:
        # the sample rate is compute()'s sample_rate argument, and the spectral
        # scaling is fixed to 'density' — the FRF is a ratio of cross- to
        # auto-spectra, so any common Welch scaling cancels and cannot change
        # the result. Letting either through would collide with the internal
        # keyword and die in scipy with a bare TypeError.
        reserved = {"fs", "scaling"} & set(kwargs)
        if reserved:
            raise ConfigurationError(
                f"FRF: {sorted(reserved)} cannot be passed as Welch options — "
                "the sample rate is the sample_rate argument of compute(), and "
                "the spectral scaling is fixed to 'density' internally (the "
                "transfer function is a spectral ratio, so a common scaling "
                "cancels).")
        import inspect
        from scipy import signal as _scipy_signal
        welch_options = (set(inspect.signature(_scipy_signal.welch).parameters)
                         - {"x", "fs", "scaling", "axis"})
        unknown = sorted(set(kwargs) - welch_options)
        if unknown:
            raise ConfigurationError(
                f"FRF: {unknown} are not Welch options, which is what the "
                f"keyword arguments after method/estimator/order are.",
                remediation=f"Use order= for the 'ls_fir' FIR order; Welch "
                            f"options are {sorted(welch_options)}.")
        self.params.update(kwargs)
        self.method = method
        self.estimator = _require_estimator(estimator, "FRF")
        self.order = order

    def compute(
        self,
        x,
        y,
        sample_rate,
        order=None,
        method=None,
        estimator=None,
        nperseg=None,
        noverlap=None,
        max_order=4096,
        stop_count=50,
    ):
        """
        Estimate the frequency response of one record pair, or the mean over
        the measurements (rows) of a 2-D block.

        ``order``, ``method``, ``estimator``, ``nperseg`` and ``noverlap`` apply to
        this call alone: they override the constructor's values for the run and
        leave the object's own settings as the constructor set them, so two
        results from one ``FRF`` are comparable unless the caller says
        otherwise on each call.

        Parameters
        ----------
        x : array_like
            Input signal array (reference) as 1D (single measurement) or 2D (rows = measurements).
        y : array_like
            Output signal array, the shape of ``x``.
        sample_rate : float
            Sampling frequency (Hz).
        order : int or str, optional
            FIR order for ``'ls_fir'``, or an order-selection criterion
            (``'AIC'``, ``'BIC'``, ``'FPE'``, ``'CP'``); the constructor's
            ``order`` when omitted.
        method : str, optional
            Method for this call ('welch', 'ls_fir', 'etfe', 'p_etfe');
            the constructor's ``method`` when omitted.
        estimator : str, optional
            Estimator for the Welch method for this call ('H1', 'H2');
            the constructor's ``estimator`` when omitted.
        nperseg : int, optional
            Segment length for this call; the constructor's
            ``params['nperseg']`` when omitted.
        noverlap : int, optional
            Overlap for Welch for this call; the constructor's
            ``params['noverlap']`` when omitted.
        max_order : int
            Highest order a criterion search tries. Default 4096, as on
            :func:`lsfir`.
        stop_count : int
            A criterion search stops after this many consecutive orders with
            no improvement. Default 50, as on :func:`lsfir`.

        Returns
        -------
        FRFResult
            The one record pair's estimate; for a 2-D block, the mean
            transfer function (and coherence) over the measurements, with
            each ``'ls_fir'`` attribute a tuple of one entry per measurement.
        """
        # Per-call arguments are resolved into locals and left there: the
        # constructor's method, estimator, order and Welch parameters are what
        # the next call with no arguments uses, so one `compute(method='etfe')`
        # cannot move a later plain `compute()` onto a different estimator or
        # a different frequency grid.
        method = self.method if method is None else method
        estimator = _require_estimator(
            self.estimator if estimator is None else estimator, "FRF.compute")
        order = self.order if order is None else order
        params = dict(self.params)
        if nperseg is not None:
            params["nperseg"] = nperseg
        if noverlap is not None:
            params["noverlap"] = noverlap
        if method not in _FRF_METHODS:
            raise ConfigurationError(
                f"FRF.compute: unknown method={method!r}; "
                "valid: 'welch', 'ls_fir', 'etfe', 'p_etfe'")

        # Convert inputs to 2D arrays (rows = measurements)
        x = np.asarray(x)
        y = np.asarray(y)
        single_measurement = x.ndim == 1
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        if x.shape[0] != y.shape[0]:
            raise ConfigurationError(
                f"FRF.compute: x and y must have the same number of measurements; "
                f"got x.shape[0]={x.shape[0]}, y.shape[0]={y.shape[0]}."
            )
        if x.shape[0] == 0:
            raise ConfigurationError(
                "FRF.compute: x and y hold no measurements (zero rows)")
        # Every estimator pairs x[n] with y[n] sample by sample, so the
        # records must match: refused here once, for the whole block.
        if x.shape != y.shape:
            raise ConfigurationError(
                f"FRF.compute: x and y must hold records of the same length; "
                f"got {x.shape[1]} and {y.shape[1]} samples.",
                remediation="Trim both records to a common span before "
                            "computing the FRF.")
        sample_rate = require_positive_finite_scalar(
            sample_rate, "FRF.compute", "sample_rate", " Hz")

        results = []
        for x_i, y_i in zip(x, y):
            if method == "welch":
                results.append(frf_welch(x_i, y_i, sample_rate,
                                         estimator=estimator, **params))
            elif method == "ls_fir":
                results.append(lsfir(
                    x_i, y_i, sample_rate, order=order, n_samples=x_i.size,
                    max_order=max_order, stop_count=stop_count,
                    nperseg=params["nperseg"]))
            elif method == "etfe":
                results.append(etfe(x_i, y_i, sample_rate))
            else:
                results.append(periodic_etfe(x_i, y_i, sample_rate,
                                             period=params["nperseg"]))
        if single_measurement:
            return results[0]

        # A block's measurements share one record length, so every row
        # lands on the same grid: the nperseg rfft grid for 'welch',
        # 'ls_fir' and 'p_etfe', the full-record one for 'etfe'.
        first = results[0]
        per_fit = {}
        if method == "ls_fir":
            per_fit = {name: tuple(getattr(r, name) for r in results)
                       for name in ("impulse_response", "order", "rcond",
                                    "information_matrix",
                                    "information_vector")}
            per_fit["criterion"] = first.criterion
        coherence = (np.mean([r.coherence for r in results], axis=0)
                     if method == "welch" else None)
        return FRFResult(
            first.frequencies,
            np.mean([r.transfer_function for r in results], axis=0),
            method=method, estimator=first.estimator, coherence=coherence,
            **per_fit)
