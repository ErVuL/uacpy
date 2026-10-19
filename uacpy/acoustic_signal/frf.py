"""Frequency-response estimation. Four functions estimate a transfer
function from one input/output record pair and return an :class:`FRFResult`:
:func:`frf_welch` (H1/H2 from Welch spectra, with the coherence), :func:`etfe`
(the empirical transfer function estimate), :func:`periodic_etfe` (the ETFE
of the period-averaged records) and :func:`lsfir` (a least-squares FIR fit,
with order selection). :class:`FRF` is the configuration of one of them, run
on a record pair or averaged over a block of measurements.
"""

from __future__ import annotations

import numbers
import warnings
from collections import namedtuple

import numpy as np
import scipy.signal as _sig
from scipy.linalg import get_lapack_funcs, toeplitz
from uacpy.acoustic_signal._results import PlottedResult
from uacpy.acoustic_signal.windows import (
    _DEFAULT_NPERSEG, _default_noverlap, _welch_nperseg,
)
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
        # reaches the user: a hand-counted ``stacklevel=3`` names a line in
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
        (:func:`frf_welch`), ``'etfe'`` (:func:`etfe`), ``'periodic_etfe'``
        (:func:`periodic_etfe`) or ``'lsfir'`` (:func:`lsfir`).
    ``estimator``
        ``'H1'`` or ``'H2'`` (``'welch'``).
    ``coherence``
        The magnitude-squared coherence on ``frequencies`` (``'welch'``).
    ``impulse_response``, ``order``, ``criterion``
        The fitted FIR, its order, and the criterion that selected the order
        (``None`` when the order was given) (``'lsfir'``).
    ``rcond``, ``information_matrix``, ``information_vector``
        The normal equations the FIR was solved from and their reciprocal
        condition number: small means the fit is poorly determined, and at
        or below ``2.2e-16`` it is a minimum-norm choice (``'lsfir'``).

    From :meth:`FRF.compute` on a 2-D block, ``transfer_function`` and
    ``coherence`` come from the measurements' spectra averaged first
    (``Σ X*·Y / Σ |X|²`` for ``'etfe'`` and ``'periodic_etfe'``, the averaged Welch
    spectra for ``'welch'``); for ``'lsfir'`` the transfer function is the
    mean of the measurements' FIR responses, and each ``'lsfir'`` attribute
    is a tuple with one entry per measurement: fits of different orders have
    no mean.
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
    """``x`` and ``y`` as one input/output record pair: real, 1-D, of one
    length, not empty. Every estimator pairs ``x[n]`` with ``y[n]`` sample
    by sample, on the one-sided frequency grid up to Nyquist that a real
    record has."""
    x = np.asarray(x)
    y = np.asarray(y)
    if np.iscomplexobj(x) or np.iscomplexobj(y):
        raise ConfigurationError(
            f"{who}: x and y must be real passband records; got "
            f"{x.dtype} and {y.dtype}. The FRF family works on the one-sided "
            f"grid up to Nyquist, which a complex (baseband) record does not "
            f"have.",
            remediation="Upconvert a baseband pair to passband first "
                        "(uacpy.comms.upconvert), then identify the "
                        "passband channel.")
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


def frf_welch(x, y, sample_rate, *, estimator="H1", nperseg=None,
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
    nperseg : int, optional
        Welch segment length; the frequency grid is its rfft grid. Unset, it
        is 8192 samples (uacpy guidance), or the whole record when that is
        shorter, or an array ``window``'s own length, as on
        :func:`uacpy.acoustic_signal.welch`; a given one is used as given,
        and refused when longer than the record.
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
    return _welch_frf(x, y, sample_rate, estimator, params, "frf_welch")


def _welch_frf(x, y, sample_rate, estimator, params, who):
    """:func:`frf_welch`'s estimate from one record pair (1-D) or from the
    rows of a block (2-D, rows = measurements).

    A block's Welch spectra ``Sxx``, ``Syy`` and ``Sxy`` are averaged over
    the rows first, and H1, H2 and the coherence formed from the averages:
    a two-channel analyser "averages corresponding values of |X|², |Y|² and
    X*Y, from which estimates of Sxx, Syy and Sxy are determined. In turn
    the frequency response … can be calculated" (Jacobsen & Juhl,
    *Fundamentals of General Linear Acoustics*, App. B.7.3, Eqs. B.74, B.75,
    B.82)."""
    params = dict(params)
    window = params.get("window", "hann")
    params["nperseg"] = _welch_nperseg(params.get("nperseg"), window,
                                       x.shape[-1], who)
    if params.get("noverlap") is None:
        params["noverlap"] = _default_noverlap(window, params["nperseg"])
    freqs, Pxx = _sig.welch(x, sample_rate, scaling="density", **params)
    _, Pyy = _sig.welch(y, sample_rate, scaling="density", **params)
    # ``Pxy`` holds csd(y, x) = Syx = conj(Sxy); the H1 expression
    # ``conj(Pxy)/Pxx`` is the textbook ``Sxy/Sxx``.
    _, Pxy = _sig.csd(y, x, sample_rate, scaling="density", **params)
    if x.ndim == 2:
        Pxx, Pyy, Pxy = (np.mean(P, axis=0) for P in (Pxx, Pyy, Pxy))
    # Each division masks bins whose denominator spectrum is numerically
    # zero (relative to its own peak) to nan with a warning, the same
    # policy _etfe_divide applies to the ETFE estimators; a zero or
    # constant record yields masked nan rather than dividing through to
    # inf/nan noise.
    if estimator == "H2":
        tf = _etfe_divide(Pyy, Pxy, who,
                          denominator="cross-spectral energy")
    else:
        tf = _etfe_divide(np.conj(Pxy), Pxx, who)
    coh = _etfe_divide(np.abs(Pxy) ** 2, Pxx * Pyy, who,
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
        ``nperseg`` grid :func:`frf_welch` and :func:`lsfir` share (and
        :func:`periodic_etfe`'s at ``period=nperseg``): the ETFE spends one
        raw rfft bin per frequency, so its grid is set by the record length.
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
        The excitation's period in samples, a whole number of at least 1
        (2.5, ``True`` or ``'64'`` is refused); the frequency grid is its
        rfft grid, ``k * sample_rate / period`` up to Nyquist.

    Returns
    -------
    FRFResult
        ``method='periodic_etfe'``.
    """
    x, y = _paired_records(x, y, "periodic_etfe")
    sample_rate = require_positive_finite_scalar(
        sample_rate, "periodic_etfe", "sample_rate", " Hz")
    period = _whole_period(period, "periodic_etfe")
    X = np.fft.rfft(_period_average(x, period, "periodic_etfe"))
    Y = np.fft.rfft(_period_average(y, period, "periodic_etfe"))
    freqs = np.fft.rfftfreq(period, d=1 / sample_rate)
    tf = _etfe_divide(Y, X, "periodic_etfe")
    return FRFResult(freqs, tf, method="periodic_etfe")


def _period_average(records, period, who):
    """The mean of the whole ``period``-sample periods along the last axis of
    ``records``; samples after the last whole period are not used."""
    n_samples = records.shape[-1]
    n_periods = n_samples // period
    if n_periods < 1:
        raise ConfigurationError(
            f"{who}: signal length must be at least one period; got "
            f"len(x)={n_samples} samples, period={period} samples.")
    whole = records[..., : n_periods * period]
    return np.mean(whole.reshape(*records.shape[:-1], n_periods, period),
                   axis=-2)


def _pooled_etfe(X, Y, who):
    """``Σ X_i*·Y_i / Σ |X_i|²`` over the rows of the input and output
    spectra ``X`` and ``Y`` (rows = measurements).

    This is the averaged transient estimate of Jacobsen & Juhl,
    *Fundamentals of General Linear Acoustics*, App. B.6.2, which "will
    approach the true frequency response function as m increases, since the
    phase spectrum of the different noise sequences can be expected to vary
    randomly from one record to another"; the input transients need not be
    identical. The mean of the rows' ratios ``Y_i/X_i`` is not that
    estimate: on a 4-tap FIR, 40 white-noise records of 4096 samples and
    output noise of standard deviation 0.5, its worst error over 100-3500 Hz
    was 3.4 to 9.8 across five seeds where this one's was 0.22 to 0.26
    (measured).

    A bin is masked where the root of the summed input power is numerically
    zero, so one row is masked exactly where :func:`etfe` masks ``Y/X``.
    """
    root_power = np.sqrt(np.sum(np.abs(X) ** 2, axis=0))
    cross = np.sum(np.conj(X) * Y, axis=0)
    return _etfe_divide(cross / np.where(root_power > 0, root_power, 1.0),
                        root_power, who)


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
        The FIR order, or the criterion that selects it: Akaike's AIC,
        Schwarz's BIC, Akaike's FPE, or Mallows' Cp (see Notes).
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
        samples, so it lines up bin for bin with :func:`frf_welch`, and
        with :func:`periodic_etfe` at ``period=nperseg``.

    Returns
    -------
    FRFResult
        ``method='lsfir'``, with ``impulse_response``, ``order``,
        ``criterion``, ``rcond``, ``information_matrix`` and
        ``information_vector``.

    Notes
    -----
    The criteria score each order ``m`` (the number of taps) on its fit's
    loss ``V = SSE/N``, the mean squared residual over the ``N`` output
    samples, in their published forms:

    * AIC ``log(V) + 2m/N`` (Akaike, "A new look at the statistical model
      identification", *IEEE Trans. Automat. Control* 19, 716-723, 1974);
    * FPE ``V·(1 + m/N)/(1 - m/N)`` (Akaike, "Fitting autoregressive models
      for prediction", *Ann. Inst. Statist. Math.* 21, 243-247, 1969);
    * BIC ``log(V) + m·log(N)/N`` (Schwarz, "Estimating the dimension of a
      model", *Ann. Statist.* 6, 461-464, 1978);

    the three as Ljung gives them for a model of ``m`` parameters fitted to
    ``N`` data (*System Identification: Theory for the User*, 2nd ed.,
    1999, §16.4); and Mallows' Cp ``SSE/σ² - N + 2(m + 1)`` (Mallows, "Some
    comments on Cp", *Technometrics* 15, 661-675, 1973), with ``σ²`` the
    residual variance of a reference fit at order ``max(2, N // 4)``
    (capped at ``max_order`` and ``N - 1``; the reference order is uacpy
    guidance — Mallows takes the full set of regressors).

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
    if not 1 <= N <= u.size:
        raise ConfigurationError(
            f"lsfir: n_samples ({N}) must be in [1, {u.size}] — it counts "
            f"the leading samples of the records the fit uses.")
    if isinstance(order, str) and order not in _ORDER_CRITERIA:
        raise ConfigurationError(
            f"lsfir: order={order!r} is neither an FIR order nor a selection "
            f"criterion.",
            remediation=f"Pass an integer order, or one of "
                        f"{', '.join(repr(c) for c in _ORDER_CRITERIA)} "
                        f"(upper case).")
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
                sse = float(np.sum(residuals**2))
                loss = sse / N              # Ljung's V, the mean squared residual

                if loss <= exact_tol:
                    # Residual at the rounding floor: this order explains
                    # the data exactly, and it is the lowest one that does.
                    best_m = m_candidate
                    best_g = g
                    break

                # The published forms (Notes): Akaike's AIC and FPE,
                # Schwarz's BIC, Mallows' Cp.
                if m == "AIC":
                    score = np.log(loss) + 2.0 * m_candidate / N
                elif m == "FPE":
                    score = loss * (N + m_candidate) / (N - m_candidate)
                elif m == "CP":
                    score = sse / sigma2 - N + 2 * (m_candidate + 1)
                elif m == "BIC":
                    score = np.log(loss) + m_candidate * np.log(N) / N

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
        if m < 1:
            raise ConfigurationError(
                f"lsfir: FIR order must be at least 1; got {m}. Pass a "
                f"positive order, or a selection criterion ('AIC', 'BIC', "
                f"'FPE', 'CP').")
        if m > N:
            raise ConfigurationError(
                f"lsfir: FIR order ({m}) must be <= n_samples "
                f"({N}) — the fit solves for that many coefficients from "
                "n_samples data points. Reduce the order, raise n_samples, "
                "or pass a selection criterion ('AIC', 'BIC', 'FPE', 'CP') "
                "to choose the order automatically.")
    Minfo, Vinfo = _info_matrices(u, y, N, m)
    g, rcond = _solve_info_matrices(Minfo, Vinfo, m)

    # Frequency response on the nperseg rfft grid, so an lsfir result
    # lines up bin-for-bin with a welch one, and with a periodic_etfe one whose
    # period is nperseg ('etfe' uses the full-record grid instead).
    freqs = np.fft.rfftfreq(int(nperseg), d=1.0 / sample_rate)
    _, h = _sig.freqz(g, worN=freqs, fs=sample_rate)
    return FRFResult(freqs, h, method="lsfir", impulse_response=g,
                     order=int(m), criterion=criterion, rcond=rcond,
                     information_matrix=Minfo, information_vector=Vinfo)


#: The estimator each :class:`FRF` method runs.
_FRF_METHODS = ("welch", "lsfir", "etfe", "periodic_etfe")

#: The FIR order :class:`FRF` fits under ``method='lsfir'`` when ``order``
#: is unset (uacpy guidance).
_FRF_DEFAULT_ORDER = 512

#: The methods that read each :class:`FRF` setting; a Welch option not
#: listed is read by ``'welch'`` alone.
_FRF_READERS = {"estimator": ("welch",), "order": ("lsfir",),
                "max_order": ("lsfir",), "stop_count": ("lsfir",),
                "nperseg": ("welch", "lsfir")}


def _refuse_unread_settings(method, order, settings, who):
    """Refuse every setting in ``settings`` (name → value, ``None`` = unset)
    that ``method`` does not read, naming the method that does: a setting
    the run drops would read as one that shaped the estimate. ``max_order``
    and ``stop_count`` bound a criterion search, so an integer ``order``
    (512 when unset) does not read them either."""
    def readers(name):
        return _FRF_READERS.get(name, ("welch",))

    unread = [name for name, value in settings.items()
              if value is not None and method not in readers(name)]
    fit_order = _FRF_DEFAULT_ORDER if order is None else order
    by_order = [name for name in ("max_order", "stop_count")
                if method == "lsfir" and fit_order not in _ORDER_CRITERIA
                and settings.get(name) is not None]
    unread += by_order
    if not unread:
        return
    where = "; ".join(
        f"{name}= by " + " or ".join(f"method={m!r}" for m in readers(name))
        + (" with an order criterion" if name in ("max_order", "stop_count")
           else "")
        for name in unread)
    them = "it" if len(unread) == 1 else "them"
    raise ConfigurationError(
        f"{who}: method={method!r}"
        + (f" with order={fit_order!r}" + (" (unset)" if order is None else "")
           if by_order else "")
        + f" does not read {', '.join(n + '=' for n in unread)}.",
        remediation=f"Drop {them}, or use the method that reads {them}: "
                    f"{where}.")


def _resolve_period(method, period, who):
    """The ``'periodic_etfe'`` excitation period as a whole number of samples, or
    ``None`` for another method; refused where the method does not match
    it, since only ``'periodic_etfe'`` reads a period and it cannot run without
    one."""
    if method != "periodic_etfe":
        if period is not None:
            raise ConfigurationError(
                f"{who}: period= is the excitation period of method='periodic_etfe'; "
                f"method={method!r} does not read it.",
                remediation="Drop period=, or use method='periodic_etfe'.")
        return None
    if period is None:
        raise ConfigurationError(
            f"{who}: method='periodic_etfe' averages whole periods of a periodic "
            f"excitation, and no period= was given.",
            remediation="Pass period=, the excitation's period in samples.")
    return _whole_period(period, who)


def _whole_period(period, who):
    """``period`` as an ``int`` of samples, refused unless it is a real
    whole number of at least 1: a ``bool``, a string or 2.5 is not a period,
    and truncating one would average over a period nobody asked for."""
    is_number = (isinstance(period, numbers.Real)
                 and not isinstance(period, bool))
    value = float(period) if is_number else float("nan")
    if not (np.isfinite(value) and value >= 1 and value.is_integer()):
        raise ConfigurationError(
            f"{who}: period={period!r} is not a whole number of samples "
            f"of at least 1.",
            remediation="Pass the excitation's period in samples as a "
                        "positive integer.")
    return int(value)


class FRF(SettingsRepr):
    """A frequency-response estimator's configuration, set once and run on
    each record pair or block :meth:`compute` is given.

    The estimation itself is :func:`frf_welch`, :func:`etfe`,
    :func:`periodic_etfe` and :func:`lsfir`, which take one record pair and
    return an :class:`FRFResult`; this class holds which one runs and with
    what settings, and averages it over the measurements of a 2-D block. It
    keeps no result: every :meth:`compute` returns its own.
    """

    def __init__(self, method="welch", estimator=None, order=None, *,
                 period=None, max_order=None, stop_count=None, **kwargs):
        """
        Parameters
        ----------
        method : str
            Estimation method. One of:

            - ``'welch'`` -- :func:`frf_welch`, dedicated to stationary
              signals, with coherence.
            - ``'lsfir'`` -- :func:`lsfir`, a least-squares impulse
              response.
            - ``'etfe'`` -- :func:`etfe` over the whole signal.
            - ``'periodic_etfe'`` -- :func:`periodic_etfe` over whole periods of
              ``period`` samples.
        estimator : str, optional
            Estimator type for ``'welch'``. One of:

            - ``'H1'`` -- minimizes the effect of noise introduced at the system output.
            - ``'H2'`` -- minimizes the effect of noise introduced at the system input.

            Unset, ``'H1'`` (:func:`frf_welch`'s default).
        order : int or str, optional
            FIR order for ``'lsfir'``, or the criterion that selects it
            (``'AIC'``, ``'BIC'``, ``'FPE'``, ``'CP'``) — :func:`lsfir`'s
            ``order``. Unset, 512 (uacpy guidance).
        period : int, optional
            The excitation's period in samples, which ``'periodic_etfe'`` requires
            and averages the records over (:func:`periodic_etfe`'s
            ``period``); its rfft grid is the result's. Refused under
            another method.
        max_order : int, optional
            Highest order an ``'lsfir'`` criterion search tries
            (:func:`lsfir`'s ``max_order``). Unset, :func:`lsfir`'s 4096.
        stop_count : int, optional
            An ``'lsfir'`` criterion search stops after this many
            consecutive orders with no improvement (:func:`lsfir`'s
            ``stop_count``). Unset, :func:`lsfir`'s 50.
        **kwargs
            Welch options (``nperseg``, ``noverlap``, ``window``, ...), as on
            :func:`frf_welch`, with its defaults; ``nperseg`` is also the
            ``'lsfir'`` grid (unset, :func:`lsfir`'s 8192).

        A setting the chosen method does not read is refused, naming the
        method that reads it: ``estimator`` and the Welch options are
        ``'welch'``'s (``nperseg`` also ``'lsfir'``'s), ``order``,
        ``max_order`` and ``stop_count`` are ``'lsfir'``'s (the last two
        with an order criterion only), ``period`` is ``'periodic_etfe'``'s.
        """
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
                remediation=f"Use order= for the 'lsfir' FIR order; Welch "
                            f"options are {sorted(welch_options)}.")
        if method not in _FRF_METHODS:
            raise ConfigurationError(
                f"FRF: unknown method={method!r}; "
                "valid: 'welch', 'lsfir', 'etfe', 'periodic_etfe'.")
        _refuse_unread_settings(
            method, order, {"estimator": estimator, "order": order,
                            "max_order": max_order, "stop_count": stop_count,
                            **kwargs}, "FRF")
        self.params = dict(kwargs)
        self.method = method
        self.estimator = (None if estimator is None
                          else _require_estimator(estimator, "FRF"))
        self.order = order
        self.period = _resolve_period(method, period, "FRF")
        self.max_order = max_order
        self.stop_count = stop_count

    def compute(self, x, y, sample_rate):
        """
        Estimate the frequency response of one record pair, or the average
        over the measurements (rows) of a 2-D block, with the estimator and
        settings the constructor set.

        Parameters
        ----------
        x : array_like
            Input signal array (reference) as 1D (single measurement) or 2D (rows = measurements).
        y : array_like
            Output signal array, the shape of ``x``.
        sample_rate : float
            Sampling frequency (Hz).

        Returns
        -------
        FRFResult
            The one record pair's estimate. For a 2-D block, the estimate
            from the measurements' spectra averaged first, as a two-channel
            analyser averages them (Jacobsen & Juhl, App. B.6.2 and B.7.3):
            ``Σ X*·Y / Σ |X|²`` for ``'etfe'`` and ``'periodic_etfe'`` (each row
            period-averaged first), H1 or H2 and the coherence from the
            averaged Welch spectra for ``'welch'``. For ``'lsfir'``, the
            mean of the measurements' FIR responses, with each ``'lsfir'``
            attribute a tuple of one entry per measurement.
        """
        method, period, params = self.method, self.period, dict(self.params)
        estimator = "H1" if self.estimator is None else self.estimator
        order = _FRF_DEFAULT_ORDER if self.order is None else self.order
        # lsfir's own defaults for what is unset.
        fit_options = {name: value for name, value in (
            ("max_order", self.max_order), ("stop_count", self.stop_count),
            ("nperseg", params.get("nperseg"))) if value is not None}

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

        if single_measurement:
            x_1, y_1 = x[0], y[0]
            if method == "welch":
                return frf_welch(x_1, y_1, sample_rate, estimator=estimator,
                                 **params)
            if method == "etfe":
                return etfe(x_1, y_1, sample_rate)
            if method == "periodic_etfe":
                return periodic_etfe(x_1, y_1, sample_rate, period=period)

        if method == "lsfir":
            results = [lsfir(x_i, y_i, sample_rate, order=order,
                             n_samples=x_i.size, **fit_options)
                       for x_i, y_i in zip(x, y)]
            if single_measurement:
                return results[0]
            # Each row's FIR response is linear in its output record, so the
            # block's response is the mean of the rows' responses, all on the
            # nperseg rfft grid.
            per_fit = {name: tuple(getattr(r, name) for r in results)
                       for name in ("impulse_response", "order", "rcond",
                                    "information_matrix",
                                    "information_vector")}
            return FRFResult(
                results[0].frequencies,
                np.mean([r.transfer_function for r in results], axis=0),
                method=method, criterion=results[0].criterion, **per_fit)

        # A block: the measurements' spectra are averaged first and the
        # transfer function formed from the averages (_welch_frf,
        # _pooled_etfe), on the grid every row shares — the nperseg rfft grid
        # for 'welch', the period's for 'periodic_etfe', the full-record one for
        # 'etfe'.
        for x_i, y_i in zip(x, y):
            _paired_records(x_i, y_i, "FRF.compute")
        if method == "welch":
            return _welch_frf(x, y, sample_rate, estimator, params,
                              "FRF.compute")
        if method == "periodic_etfe":
            x = _period_average(x, period, "FRF.compute")
            y = _period_average(y, period, "FRF.compute")
        freqs = np.fft.rfftfreq(x.shape[1], d=1 / sample_rate)
        tf = _pooled_etfe(np.fft.rfft(x, axis=-1), np.fft.rfft(y, axis=-1),
                          "FRF.compute")
        return FRFResult(freqs, tf, method=method)
