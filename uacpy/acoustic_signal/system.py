"""The medium between transmitter and receiver: identify it, simulate it,
or read its dispersion.

One question — *what did the channel do to my signal* — answered three ways:
:class:`FRF` fits a transfer function to measured input/output pairs;
:func:`impulse_response` and :func:`simulate_reception` play a signal through
a known channel; and :func:`modal_group_velocity` with
:func:`warp_signal` / :func:`unwarp_signal` read and undo waveguide
dispersion.

Three more families answer that question on plain arrays, for a channel
that did not come from a uacpy model: the power-delay-profile statistics
(:func:`rms_delay_spread` and its three companions), the
transfer-function operations (:func:`arrival_transfer_function`,
:func:`broadband_propagation_loss`, :func:`gate_transfer_function`), and
:func:`simulate_arrival_reception`, which places each arrival as a
phase-rotated, absorbed copy of the transmitted waveform.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
import numpy as np
import scipy.signal as _sig
from scipy.linalg import get_lapack_funcs, toeplitz
from uacpy.core.exceptions import ConfigurationError
from uacpy.core._warn_frames import USER_FRAME_SKIP
from typing import Optional, Tuple
from uacpy.acoustic_signal._signal_validate import (
    require_increasing_axis, require_positive_finite_scalar)
from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core._finite_difference import warn_if_storage_under_resolves


# ──────────────────────────────────────────────────────────────────────
# Frequency-response estimation
#
# The one class in the package,
# because it holds a fitted model.
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
    the shipped default order ``m = 512``, where the LU solve returns an
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
        raise np.linalg.LinAlgError("Singular matrix")
    rcond = float(gecon(lu, anorm, norm='1')[0])

    if rcond > _INFO_RCOND_FLOOR:
        return getrs(lu, piv, Vinfo)[0], rcond

    # numpy's own least-squares cutoff, spelled out rather than left to the
    # ``rcond=None`` default: singular values of Minfo below order*eps times
    # the largest are treated as zero.
    g = np.linalg.lstsq(Minfo, Vinfo,
                        rcond=Minfo.shape[0] * np.finfo(float).eps)[0]
    warnings.warn(
        f"FRF.compute_lsfir: the information matrix at FIR order {order} is "
        f"numerically singular (reciprocal condition number {rcond:.2e} <= "
        f"{_INFO_RCOND_FLOOR:.2e}), so its LU solution carries no correct "
        f"digit; the impulse response returned is the minimum-norm "
        f"least-squares solution of the same normal equations, and the "
        f"frequency response is undetermined wherever the input does not "
        f"excite. Lower the FIR order, or excite the whole band up to "
        f"Nyquist.",
        UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )
    return g, rcond


_ETFE_REL_FLOOR = 1e-12          # relative to max|X| over the record


def _etfe_divide(Y, X, caller: str, quantity="the transfer function",
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
        # Each estimator method reaches this divide through its own branch of
        # ``FRF.compute``, so no single frame count reaches the user: a
        # hand-counted ``stacklevel=3`` named the branch line in this module
        # for welch, etfe and p_etfe alike (measured). ``skip_file_prefixes``
        # counts no frames at all — it walks to the first file outside the
        # package.
        warnings.warn(
            f"{caller}: {int((~excited).sum())} of {excited.size} frequency "
            f"bins carry no {denominator} (denominator magnitude <= "
            f"{_ETFE_REL_FLOOR:g} of its peak); {quantity} is undefined "
            f"there and is returned as nan. Excite the whole band, or "
            f"restrict the analysis to the excited band.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    return np.where(excited, Y / np.where(excited, X, 1.0), np.nan)


class FRF:
    """Frequency Response Function (FRF) computation.

    Computation only: the drawing is :func:`uacpy.plot.plot_frf`, and the
    LS-FIR diagnostics are :func:`uacpy.plot.plot_lsfir_diagnostics`. This
    class keeps state — the estimator and its settings — so one configured
    instance answers several signal pairs; it has no ``.plot()``.
    """

    def __init__(self, method="welch", estimator="H1", m=512, **kwargs):
        """
        Transfer Function (Frequency Response Function, FRF) computation.

        Parameters
        ----------
        method : str
            Estimation method. One of:

            - ``'welch'`` -- Welch periodogram for PSD estimate, dedicated to stationary signals.
            - ``'ls_fir'`` -- least-squares impulse response method.
            - ``'etfe'`` -- ETFE method over the whole signal.
            - ``'p_etfe'`` -- Periodic ETFE that computes average signal over segments.
        estimator : str
            Estimator type. One of:

            - ``'H1'`` -- minimizes the effect of noise introduced at the system output.
            - ``'H2'`` -- minimizes the effect of noise introduced at the system input.
        m : int
            Length of the impulse response in samples for the ls_fir method.
        **kwargs
            Additional keyword arguments (e.g., nperseg, noverlap).

        Notes
        -----
        The Transfer Function (FRF) is a complex function that relates the
        input and output of a linear time-invariant (LTI) system in the
        frequency domain. Under the scipy convention ``Sxy = csd(x, y)
        = E[X*·Y]`` it is defined as::

            H1(f) = Sxy(f) / Sxx(f)
            H2(f) = Syy(f) / Syx(f)

        where ``Sxx`` is the input PSD and ``Sxy`` the input/output CPSD.
        Internally the code stores ``csd(y, x) = Syx = conj(Sxy)`` in
        the variable named ``Pxy``; the H1 expression
        ``conj(Pxy)/Pxx`` recovers the textbook ``Sxy/Sxx``.
        """
        # Default parameters, overridden by kwargs if provided
        self.params = {
            "nperseg": 8192,
            "noverlap": 0,
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
        self.params.update(kwargs)
        self.method = method
        self.estimator = estimator
        self.Minfo = np.array([[0]])
        self.Vinfo = np.array([[0]])
        self.m = m
        self.g = 0  # Impulse response
        # FIR order(s) chosen by ls_fir order selection: an int for 1-D
        # input, a per-measurement list for 2-D input, None when m is an
        # explicit order (nothing was selected) or the method is not ls_fir.
        self.selected_order = None
        # Reciprocal condition number of the information matrix the returned
        # impulse response was solved from, ls_fir only: small means the fit
        # is poorly determined, at or below _INFO_RCOND_FLOOR it is not
        # determined at all and the coefficients are a minimum-norm choice.
        # None when the method is not ls_fir; for 2-D input it carries the
        # last measurement's fit, the row `g` also comes from.
        self.info_rcond = None
        self.coh = None  # Coherence, welch only
        self.frequencies = None
        self.tf = None

    def compute(
        self,
        x,
        y,
        sample_rate,
        m=None,
        method=None,
        estimator=None,
        nperseg=None,
        noverlap=None,
        m_max=4096,
        stop_count=None,
    ):
        """
        Compute the Frequency Response Function (FRF), supporting both 1D and 2D inputs.

        If inputs are 2D, average results are computed over all measurements.

        ``m``, ``method``, ``estimator``, ``nperseg`` and ``noverlap`` apply to
        this call alone: they override the constructor's values for the run and
        leave the object's own settings as the constructor set them, so two
        results from one ``FRF`` are comparable unless the caller says
        otherwise on each call. The *result* attributes (``frequencies``,
        ``tf``, ``coh``, ``g``, ``selected_order``) are rewritten by every run.

        Parameters
        ----------
        x : array_like
            Input signal array (reference) as 1D (single measurement) or 2D (rows = measurements).
        y : array_like
            Output signal array as 1D (single measurement) or 2D (rows = measurements).
        sample_rate : float
            Sampling frequency (Hz).
        m : int or str, optional
            Impulse response length (for TF methods), or an automatic
            order-selection criterion for ``'ls_fir'``: ``'AIC'``,
            ``'BIC'``, ``'FPE'``, or ``'CP'``. The order the criterion picks is
            published on ``self.selected_order`` — an int for 1-D input, a
            per-measurement list for 2-D input; ``self.selected_order`` is
            ``None`` when ``m`` is an explicit order.
        method : str, optional
            Method for this call ('welch', 'ls_fir', 'etfe', 'p_etfe');
            the constructor's ``method`` when omitted.
        estimator : str, optional
            Estimator for the Welch method for this call ('H1', 'H2');
            the constructor's ``estimator`` when omitted.
        nperseg : int, optional
            Segment length for Welch for this call; the constructor's
            ``params['nperseg']`` when omitted.
        noverlap : int, optional
            Overlap for Welch for this call; the constructor's
            ``params['noverlap']`` when omitted.
        m_max : int
            Maximum impulse response length.
        stop_count : int, optional
            Stop AIC search after this many consecutive non-improvements.

        Returns
        -------
        freqs : ndarray
            Frequency array (Hz).
        tf : ndarray
            Transfer function (complex-valued).
        """
        # Per-call arguments are resolved into locals and left there: the
        # constructor's method, estimator, order and Welch parameters are what
        # the next call with no arguments uses, so one `compute(method='etfe')`
        # or `compute_periodic_etfe(nperseg=256)` cannot move a later plain
        # `compute()` onto a different estimator or a different frequency grid.
        method = self.method if method is None else method
        estimator = self.estimator if estimator is None else estimator
        m = self.m if m is None else m
        params = dict(self.params)
        if nperseg is not None:
            params["nperseg"] = nperseg
        if noverlap is not None:
            params["noverlap"] = noverlap
        if stop_count is None:
            # early-stop after 50 consecutive orders with no score improvement
            # (compute_lsfir's documented default); m_max is the hard order cap.
            stop_count = 50

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
                f"got x.shape[0]={x.shape[0]}, y.shape[0]={y.shape[0]}"
            )
        n_meas = x.shape[0]
        if n_meas == 0:
            raise ConfigurationError(
                "FRF.compute: x and y hold no measurements (zero rows)")
        m_list, tf_list, coh_list = [], [], []

        for i in range(n_meas):
            x_i = x[i, :].ravel()
            y_i = y[i, :].ravel()
            if method == "welch":
                freqs_i, tf_i, coh_i = self.compute_welch(
                    x_i, y_i, sample_rate, params=params, estimator=estimator)
                coh_list.append(coh_i)
            elif method == "ls_fir":
                # compute_lsfir takes (output, input) — y before x, unlike the
                # other three estimators.
                freqs_i, tf_i, g_i = self.compute_lsfir(
                    y_i, x_i, sample_rate, m, len(x_i), m_max=m_max,
                    stop_count=stop_count, nperseg=params["nperseg"]
                )
                m_list.append(len(g_i))
            elif method == "etfe":
                freqs_i, tf_i = self.compute_etfe(x_i, y_i, sample_rate)
            elif method == "p_etfe":
                freqs_i, tf_i = self.compute_periodic_etfe(
                    x_i, y_i, sample_rate, nperseg=params["nperseg"])
            else:
                raise ConfigurationError(
                    f"FRF.compute: unknown method={method!r}; "
                    "valid: 'welch', 'ls_fir', 'etfe', 'p_etfe'"
                )

            tf_list.append(tf_i)

        # Average across measurements. 'welch', 'ls_fir' and 'p_etfe' share
        # the nperseg rfft grid (k*fs/nperseg); 'etfe' returns the full-record
        # rfft grid (k*fs/len(x)). Within one call all rows have the same
        # length, so every measurement lands on the same grid either way.
        freqs = freqs_i
        tf = np.mean(tf_list, axis=0)

        # Update object state; every run rewrites the method-specific
        # attributes so a reused FRF cannot report a previous method's result.
        self.frequencies = freqs
        self.tf = tf
        self.coh = np.mean(coh_list, axis=0) if method == "welch" else None
        if method == "ls_fir":
            self.g = g_i  # For 2D inputs, uses last channel's impulse response
            if m in ("AIC", "BIC", "FPE", "CP"):
                # Criterion-selected order(s): the int for 1-D input, the
                # per-measurement list for 2-D input (a mean of the rows'
                # orders is an order no row selected).
                self.selected_order = (m_list[0] if single_measurement
                                       else m_list)
            else:
                # m was an explicit order: nothing was selected.
                self.selected_order = None
        else:
            self.g = 0
            self.selected_order = None
            self.info_rcond = None

        return freqs, tf

    def compute_welch(self, x, y, sample_rate, *, params=None,
                      estimator=None):
        """
        Compute the Frequency Response Function (FRF) using Welch's method.

        This method is dedicated to stationary signals. Coherence indicates
        the degree of linear dependency between input (x) and output (y) at
        each frequency. Bins where the estimator's denominator spectrum
        (``Pxx`` for H1, ``Syx`` for H2, ``Pxx*Pyy`` for the coherence) is
        numerically zero relative to its peak are returned as nan with a
        warning, as in the ETFE estimators.

        Parameters
        ----------
        x : array_like
            Input signal array (reference).
        y : array_like
            Output signal array.
        sample_rate : float
            Sampling frequency of the signals (Hz).
        params : dict, optional
            Welch/CSD keyword arguments for this call; the constructor's
            ``params`` when omitted.
        estimator : str, optional
            ``'H1'`` or ``'H2'`` for this call; the constructor's
            ``estimator`` when omitted.

        Returns
        -------
        freqs : ndarray
            Array of frequencies (Hz).
        tf : ndarray
            Complex transfer function.
        coh : ndarray
            Array of coherence values.
        """
        params = self.params if params is None else params
        estimator = self.estimator if estimator is None else estimator
        freqs, Pxx = _sig.welch(x, sample_rate, scaling="density", **params)
        _, Pyy = _sig.welch(y, sample_rate, scaling="density", **params)
        _, Pxy = _sig.csd(y, x, sample_rate, scaling="density", **params)
        # Each division masks bins whose denominator spectrum is numerically
        # zero (relative to its own peak) to nan with a warning, the same
        # policy _etfe_divide applies to the ETFE estimators; a zero or
        # constant record yields masked nan rather than dividing through to
        # inf/nan noise.
        if estimator == "H2":
            tf = _etfe_divide(Pyy, Pxy, "FRF.compute_welch",
                              denominator="cross-spectral energy")
        else:  # Default to H1
            tf = _etfe_divide(np.conj(Pxy), Pxx, "FRF.compute_welch")
        coh = _etfe_divide(np.abs(Pxy) ** 2, Pxx * Pyy, "FRF.compute_welch",
                           quantity="the coherence",
                           denominator="input or output energy")

        return freqs, tf, coh

    def compute_periodic_etfe(self, x, y, sample_rate, nperseg=None):
        """
        Compute ETFE for periodic data.

        Coherently averaging the *time records* over whole periods of a
        periodic excitation (not their spectra) is what buys back the
        consistency the raw :meth:`compute_etfe` lacks: the periodic part adds
        in phase while independent noise averages down.

        Parameters
        ----------
        x : array_like
            Input signal.
        y : array_like
            Output signal.
        sample_rate : float
            Sampling frequency.
        nperseg : int, optional
            Segment length of period in samples.

        Returns
        -------
        freqs : ndarray
            Frequencies.
        tf : ndarray
            Complex transfer function.
        """

        # Frequency grid is the rfft grid of one period, k*sample_rate/period
        # in Hz, up to Nyquist. ``nperseg`` applies to this call only — the
        # constructor's value is what a later call with no ``nperseg`` uses.
        period = int(nperseg) if nperseg else self.params["nperseg"]
        n_periods = len(x) // period

        if n_periods < 1:
            raise ConfigurationError(
                f"FRF.compute_periodic_etfe: signal length must be at least one "
                f"period; got len(x)={len(x)} samples, period={period} samples"
            )

        # Extract a whole number of periods
        x = x[: n_periods * period]
        y = y[: n_periods * period]

        x_reshaped = x.reshape(n_periods, period)
        y_reshaped = y.reshape(n_periods, period)

        # Average over periods to reduce noise
        x_avg = np.mean(x_reshaped, axis=0)
        y_avg = np.mean(y_reshaped, axis=0)
        X = np.fft.rfft(x_avg)
        Y = np.fft.rfft(y_avg)
        freqs = np.fft.rfftfreq(period, d=1 / sample_rate)
        tf = _etfe_divide(Y, X, 'FRF.compute_periodic_etfe')

        return freqs, tf

    def compute_etfe(self, x, y, sample_rate):
        """
        Compute the Empirical Transfer Function Estimate (ETFE).

        This method directly estimates the transfer function by dividing the
        output Fourier transform by the input Fourier transform.

        The ETFE is unbiased but **not consistent**: it spends one complex
        datum per frequency bin, so its variance does not fall as the record
        grows and the estimate stays noisy bin to bin no matter how much data
        is supplied. Measured on a known 4-tap FIR driven by white noise, its
        worst-case error over 100-3500 Hz is ~0.6 where ``'welch'`` reaches
        5e-4. Use it for a quick look or on a clean swept/periodic excitation;
        use ``'p_etfe'`` (averages over periods), ``'welch'`` or ``'ls_fir'``
        when the estimate has to be accurate.

        Parameters
        ----------
        x : array_like
            Input signal array (reference).
        y : array_like
            Output signal array.
        sample_rate : float
            Sampling frequency of the signals (Hz).

        Returns
        -------
        freqs : ndarray
            Array of frequencies (Hz): the rfft grid of the whole record,
            ``k * sample_rate / len(x)`` up to Nyquist. This is finer than
            the ``nperseg`` grid that 'welch', 'ls_fir' and 'p_etfe' share —
            the ETFE spends one raw rfft bin per frequency, so its grid is
            set by the record length, not by ``nperseg``.
        tf : ndarray
            Complex transfer function.
        """

        # Ensure signals are the same length
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
        X = np.fft.rfft(x)
        Y = np.fft.rfft(y)

        # Determine frequency grid based on n_freqs
        n_fft = min_len
        freqs = np.fft.rfftfreq(n_fft, d=1 / sample_rate)
        tf = _etfe_divide(Y, X, 'FRF.compute_etfe')

        return freqs, tf

    def compute_lsfir(self, y, u, sample_rate, m, N, m_max=4096, stop_count=50, nperseg=None):
        """
        Compute the finite impulse response estimation using an information matrix/vector method.
        Supports model order selection using AIC, BIC, FPE, or Mallows' Cp.

        Parameters
        ----------
        y : array_like
            System output. Note the ``(output, input)`` argument order, the
            reverse of the ``(x, y)`` used by the other estimators.
        u : array_like
            System input.
        sample_rate : float
            Sampling rate in Hz.
        m : int or str
            Model order or selection criterion ('AIC', 'BIC', 'FPE', 'CP').
        N : int
            Number of data points to consider (N >= m).
        m_max : int
            Maximum model order for automatic selection.
        stop_count : int
            Stop search after stop_count consecutive steps with no improvement.
        nperseg : int, optional
            Frequency axis will be nperseg/2+1 samples between 0 and sample_rate/2.

        Returns
        -------
        freqs : ndarray
            Frequency array (Hz).
        h : ndarray
            Complex frequency response.
        g : ndarray
            Impulse response estimate.

        Notes
        -----
        The fit solves the normal equations ``X.T @ X`` of the covariance
        design matrix, whose condition number is ``cond(X)**2``. The
        reciprocal condition number of the system the returned ``g`` came out
        of is published on ``self.info_rcond``; a warning names the order when
        it falls to where the equations are numerically singular, which an
        order longer than the excited band can support reaches easily — a
        100 Hz - 20 kHz sweep at fs = 48 kHz does it at the default ``m=512``.
        """

        # ``nperseg`` sets this call's frequency grid only; the constructor's
        # value is what a later call with no ``nperseg`` uses.
        grid_nperseg = int(nperseg) if nperseg else self.params["nperseg"]

        y = np.array(y)
        u = np.array(u)

        if m in ["AIC", "FPE", "CP", "BIC"]:
            # Model order selection
            m_max = min(m_max, N - 1)
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
                        f"FRF.compute_lsfir: criterion 'CP' needs a reference "
                        f"fit at order {full_model_m} to set its residual "
                        f"variance, and that fit gave a singular information "
                        f"matrix. The input u is degenerate (constant, "
                        f"all-zero, or too short); use a persistently exciting "
                        f"input, pass an explicit integer m, or choose "
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
                    f"FRF.compute_lsfir: no FIR order in 1..{m_max} could be "
                    f"fitted with criterion {m!r} — every candidate gave a "
                    "singular information matrix. The input u is degenerate "
                    "(constant, all-zero, or too short); use a persistently "
                    "exciting input, or pass an explicit integer m."
                )
            m = best_m

            # Republish the normal equations for the order actually selected,
            # and re-solve them through the rank-revealing path. The search
            # above compares orders under one solve, and its LU is also its
            # identifiability screen: a candidate whose equations are singular
            # is skipped, and one that is merely near-singular fits worse and
            # scores worse, so neither is selected. The order that *is*
            # selected then gets its coefficients from _solve_info_matrices,
            # which returns the same numbers bit for bit whenever the LU is
            # trustworthy and says so when it is not.
            self.Minfo, self.Vinfo = _info_matrices(u, y, N, m)
            g, self.info_rcond = _solve_info_matrices(self.Minfo, self.Vinfo, m)

        else:
            # Given m, compute directly
            m = int(m)
            if m > N:
                raise ConfigurationError(
                    f"FRF.compute_lsfir: FIR order m ({m}) must be <= N "
                    f"({N}) — the fit solves for m coefficients from N data "
                    "points. Reduce m, raise N, or pass a selection "
                    "criterion ('AIC', 'BIC', 'FPE', 'CP') to choose the "
                    "order automatically.")
            self.Minfo, self.Vinfo = _info_matrices(u, y, N, m)
            g, self.info_rcond = _solve_info_matrices(self.Minfo, self.Vinfo, m)

        # Frequency response on the nperseg rfft grid, so an ls_fir result
        # lines up bin-for-bin with a welch or p_etfe one ('etfe' alone uses
        # the full-record grid instead).
        freqs = np.fft.rfftfreq(grid_nperseg, d=1.0 / sample_rate)
        _, h = _sig.freqz(g, worN=freqs, fs=sample_rate)

        return freqs, h, g


# ──────────────────────────────────────────────────────────────────────
# Channel simulation
#
# Playing a signal through a known impulse
# response, and building that response from a transfer function.
# ──────────────────────────────────────────────────────────────────────

# Windowed-sinc fractional-delay kernel: taps each side of the arrival and
# the Kaiser window shape. L=8 is flat to ~0.01 dB below f/fs = 0.35.
_FRAC_DELAY_HALF_LEN = 8
_FRAC_DELAY_KAISER_BETA = 8.0

# Largest impulse response ``impulse_response`` will size for itself off the
# latest arrival (delays.max() * sample_rate taps): 2**26 taps occupy 1 GiB
# as complex128, beyond any physical channel spread (2**26 taps is 699 s of
# channel at 96 kHz). Past it the caller states n_samples rather than having
# one mis-scaled delay allocate that much silently.
_MAX_DEFAULT_TAPS = 1 << 26

# Largest default DFT length ``impulse_response_from_transfer_function`` will
# size for itself (~4 M samples, ~100 MB across the working arrays). Beyond it
# the caller states n_samples rather than having a finely spaced frequency
# vector allocate that much silently.
_MAX_DEFAULT_IR_SAMPLES = 1 << 22


def fractional_delay_taps(frac: float, half_len: int = _FRAC_DELAY_HALF_LEN,
                          beta: float = _FRAC_DELAY_KAISER_BETA):
    """Kaiser-windowed sinc taps that delay a signal by ``frac`` samples.

    Returns ``2*half_len`` taps for offsets ``-half_len+1 .. half_len``
    relative to the sample the delay floors to, normalised to unit DC gain
    (windowing truncates the sinc, so the raw taps sum to a little under
    one — 0.9999784 at worst over ``frac``, 0.99999579 at half a sample —
    and an unnormalised constant would lose up to 1.9e-4 dB).

    Shared with :func:`uacpy.models.bellhop.delayandsum`, which places whole
    waveforms rather than single taps but needs the identical kernel: a
    two-tap linear split is NOT a fractional delay — its response
    ``|(1-frac) + frac*e^{-jw}|`` is a lowpass whose attenuation depends on
    ``frac``, with a full null at Nyquist for ``frac = 0.5``. The windowed
    sinc is flat to ~0.01 dB below ``f/fs = 0.35``.
    """
    offsets = np.arange(-half_len + 1, half_len + 1) - float(frac)
    u = offsets / half_len
    win = (np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - u * u))) / np.i0(beta))
    taps = np.sinc(offsets) * win
    total = taps.sum()
    return taps / total if total else taps


def impulse_response(amplitudes, delays_s, sample_rate: float, *,
                     n_samples: Optional[int] = None, fractional: bool = True):
    """Channel impulse response from discrete arrivals.

    Places each arrival ``amplitudes[i]`` at delay ``delays_s[i]``. With
    ``fractional=True`` the arrival is placed with a windowed-sinc
    fractional-delay kernel; otherwise it is quantised to the nearest sample.

    The kernel matters because a two-tap linear split is **not** a fractional
    delay: its response ``|(1-frac) + frac*e^{-jw}|`` is a lowpass whose
    attenuation depends on ``frac``, with a full null at Nyquist for
    ``frac = 0.5`` (-3.0 dB at ``f/fs = 0.25``, -10.2 dB at 0.40). Two
    arrivals a propagation model reports as equal then came back differing by
    up to 10 dB, decided by the sub-sample part of their travel times — at the
    right time, at the wrong level. The windowed sinc is flat to ~0.01 dB
    over the same band. Group delay was correct either way.

    Parameters
    ----------
    amplitudes : 1-D array
        Complex (or real) arrival amplitudes.
    delays_s : 1-D array
        Arrival travel times (s), >= 0.
    sample_rate : float
        Sample rate (Hz).
    n_samples : int, optional
        Length of the IR. Default: just past the latest arrival; defaults
        above ``2**26`` taps (1 GiB as complex128) raise rather than
        allocate. An explicit ``n_samples`` can end before an arrival: any
        arrival falling entirely outside the window is dropped from ``h``,
        with a ``UserWarning`` counting the drops (on both placement
        paths).
    fractional : bool
        Windowed-sinc fractional-delay placement. ``False`` quantises the
        delay to the nearest sample (+/- 0.5 sample of timing error).

    Returns
    -------
    t : ndarray
        Time axis (s).
    h : ndarray
        Impulse response (complex if amplitudes are complex).
    """
    a = np.asarray(amplitudes)
    d = np.asarray(delays_s, dtype=float)
    if a.shape != d.shape or a.ndim != 1:
        raise ConfigurationError(
            "impulse_response: amplitudes and delays_s must be 1-D, equal "
            f"length; got amplitudes shape {a.shape} and delays_s shape "
            f"{d.shape}")
    if np.any(d < 0):
        raise ConfigurationError(
            f"impulse_response: delays_s must be >= 0; got "
            f"{int(np.count_nonzero(d < 0))} negative value(s), first at "
            f"index {int(np.argmax(d < 0))} ({d[d < 0][0]:g} s)")
    fs = require_positive_finite_scalar(
        sample_rate, "impulse_response", "sample_rate", " Hz")
    pos = d * fs
    L = _FRAC_DELAY_HALF_LEN
    if n_samples is None:
        if pos.size:
            # Non-fractional rounds to the nearest sample, so the last
            # occupied index is round(pos.max()), not floor(pos.max()).
            # Compared as float before the int conversion so an inf/NaN
            # delay hits the typed bound, not a raw OverflowError.
            n_est = float(np.floor(pos.max()) + L + 1 if fractional
                          else np.round(pos.max()) + 1)
            if not n_est <= _MAX_DEFAULT_TAPS:
                raise ConfigurationError(
                    f"impulse_response: the latest delay {d.max():g} s at "
                    f"sample_rate {fs:g} Hz implies a {n_est:.0f}-tap "
                    f"response ({n_est * 16 / 2 ** 30:.3g} GiB as "
                    f"complex128), above the {_MAX_DEFAULT_TAPS}-tap "
                    f"default limit. Check the delay units (delays_s is "
                    f"seconds), or pass n_samples explicitly.")
            n_samples = n_est
        else:
            n_samples = 1
    n_samples = int(n_samples)
    dtype = complex if np.iscomplexobj(a) else float
    h = np.zeros(n_samples, dtype=dtype)
    n_clipped = 0
    n_dropped = 0
    for amp, p in zip(a, pos):
        if not fractional:
            i0 = int(np.round(p))
            if 0 <= i0 < n_samples:
                h[i0] += amp
            else:
                n_dropped += 1
            continue
        i0 = int(np.floor(p))
        k = np.arange(i0 - L + 1, i0 + L + 1)
        g = fractional_delay_taps(p - i0, half_len=L)       # taps on k
        ok = (k >= 0) & (k < n_samples)
        # An integer-delay arrival lives entirely on sample i0 (every other
        # tap is sinc(integer) = 0), so clipping its zero taps loses nothing
        # and it is dropped only when i0 itself falls outside the window. A
        # non-integer arrival with no tap in the window is likewise dropped
        # whole. One with only some taps clipped keeps a truncated kernel —
        # its amplitude changes rather than the whole arrival landing on one
        # sample (an arrival at 10.9 samples would land entirely at 10) —
        # and the change is not one-directional: the sinc tail can sum
        # either way, and the worst case is a GAIN — measured DC gain 1.1274
        # (+1.04 dB) for an arrival 0.5 samples from the start against
        # 0.9862 at 3.5 samples, at fs = 20 kHz over 128 samples.
        if p == i0:
            if not 0 <= i0 < n_samples:
                n_dropped += 1
        elif not ok.any():
            n_dropped += 1
        elif not ok.all():
            n_clipped += 1
        h[k[ok]] += amp * g[ok]
    if n_dropped:
        warnings.warn(
            f"impulse_response: {n_dropped} arrival(s) lie entirely outside "
            f"the {n_samples}-sample response window ({n_samples / fs:g} s "
            f"at {fs:g} Hz) and are dropped from h. Lengthen n_samples, or "
            f"leave it None to size the response just past the latest "
            f"arrival.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
    if n_clipped:
        warnings.warn(
            f"impulse_response: {n_clipped} fractional arrival(s) sit within "
            f"{L} samples of the ends of an {n_samples}-sample response, so "
            f"their interpolation kernel is truncated, so their amplitude "
            f"is wrong in either direction (measured +1.04 dB for an arrival "
            f"half a sample from the end). Lengthen n_samples, or use "
            f"fractional=False to put each such arrival wholly on its "
            f"nearest sample (a ±0.5-sample timing error; a nearest sample "
            f"outside the window is a dropped arrival, which warns).",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
    t = np.arange(n_samples) / fs
    return t, h


def simulate_reception(transmit, amplitudes, delays_s, sample_rate: float):
    """Received signal = ``transmit`` convolved with the channel IR.

    Returns ``(t, received)`` with ``t`` the output time axis (s).
    """
    x = np.asarray(transmit)
    _, h = impulse_response(amplitudes, delays_s, sample_rate)
    y = np.convolve(x, h)
    t = np.arange(y.size) / float(sample_rate)
    return t, y


def channel_response(h, sample_rate: float, *, nfft: Optional[int] = None):
    """Frequency response ``H(f)`` of a baseband impulse response ``h``.

    Returns ``(frequencies, H)`` with ``H`` **complex** and both axes
    centred on 0 Hz: a baseband channel response is two-sided and not
    conjugate symmetric, so the negative half carries information the
    positive half does not, and ``rfft`` rejects complex input outright.

    This is the direction :func:`impulse_response_from_transfer_function`
    does not go, but the two are not exact inverses and composing them is
    not a no-op. That one models a **real** channel from a one-sided
    ``H(f)`` sampled anywhere, so it resamples onto its own DFT grid and
    zeroes every bin outside the band it was handed — including Nyquist,
    which this function's grid reaches and its does not. Measured: a
    40-tap real ``h`` taken here at ``nfft=1024``, half-band sliced and
    passed back, returns with a peak error of 1.8e-4 and the same 1.8e-4
    smeared past the original support.

    ``nfft`` defaults to ``max(1024, 2 * h.size)``. Zero-padding
    **interpolates** between DFT bins — it draws the shape between the
    nulls, it does not resolve anything the ``h.size / sample_rate`` record
    cannot. The floor of 1024 is there so a short tap set still plots as a
    curve rather than a polygon; the factor of two puts a sample between
    every pair of natural bins. Pass ``nfft`` to pin it.

    Magnitude in dB is the caller's: ``20 * log10(abs(H))`` is ``-inf`` at a
    perfect null, so whoever renders it picks the floor that sets how deep a
    null is drawn, rather than inheriting one chosen here.

    Returns ``(frequencies, H)``, each of length ``nfft``.
    """
    ir = np.asarray(h, dtype=complex)
    if ir.ndim != 1:
        raise ConfigurationError(
            f"channel_response: h must be a 1-D impulse response; got shape "
            f"{ir.shape}. Transform one channel at a time.")
    if ir.size == 0:
        raise ConfigurationError(
            "channel_response: h is empty, so there is no channel to "
            "transform.")
    fs = require_positive_finite_scalar(
        sample_rate, "channel_response", "sample_rate", " Hz")
    n = max(1024, 2 * ir.size) if nfft is None else int(nfft)
    if n < ir.size:
        raise ConfigurationError(
            f"channel_response: nfft={n} is shorter than the {ir.size}-tap "
            f"impulse response, which would truncate it — the tail beyond "
            f"the {n}th tap would be dropped, not folded. Pass nfft >= "
            f"{ir.size}.")
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
    H = np.fft.fftshift(np.fft.fft(ir, n))
    return freqs, H


def transfer_function_from_impulse_response(h, sample_rate: float, *,
                                            t0: float = 0.0, band=None,
                                            axis: int = -1, who=None):
    """One-sided ``H(f)`` from a real impulse response — the inverse of
    :func:`impulse_response_from_transfer_function`.

    For an ``h`` whose first sample sits at ``t0``::

        H(f) = rfft(h) * exp(-2j pi f t0)

    Returns ``(frequencies, H)``, both 1-D, with ``frequencies`` running
    from 0 to the Nyquist frequency unless ``band`` narrows it.

    **The rotation is what makes it an inverse.** A record starting at
    ``t0`` carries that offset in every sample, so a bare ``rfft`` returns
    ``H`` multiplied by ``exp(+2j pi f t0)`` — right in magnitude, wrong in
    phase. Checked against a two-path ``H``: with the rotation the round
    trip reproduces it to ``max|err| = 0.0000``; without it, 3.16. Magnitude
    alone looks perfect either way, which is why this is easy to get wrong
    and hard to notice — the error only shows once two such spectra
    interfere.

    **Unscaled, because its counterpart is.**
    :func:`impulse_response_from_transfer_function` is a plain
    ``irfft`` — no ``df``, no ``fs`` — so this is a plain ``rfft`` and the
    pair round-trips to floating point. Do not add a ``1/sample_rate``
    here to make it a spectral density: the first draft of this function
    did, and the round trip came back a factor of ``fs`` short with the
    PHASE still perfect, which is the hardest kind of error to notice.
    ``Field.to_time_trace`` and :meth:`Field.to_transfer_function` use the
    density convention instead, and that pair round-trips too — the two
    conventions each close, and must not be mixed.

    **What it cannot recover.** Only what the record holds: an ``h`` that
    was itself band-limited carries no information outside that band, and
    the bins there come back as whatever the synthesis left — near zero for
    a clean record, and the edge artefacts of the original transform
    otherwise. Pass ``band`` to keep the span the data actually supports.
    Nor can it undo a fold: an arrival later than the record wrapped before
    it ever reached these samples.

    Parameters
    ----------
    h : array_like
        Real impulse response, 1-D. A complex array's real part is used —
        a pressure history is real, and transforming an analytic signal as
        though it were one would double the positive-frequency content.
    sample_rate : float
        Rate (Hz) ``h`` is sampled at, positive and finite.
    t0 : float, default 0.0
        Time (s) of ``h[0]``. Leave at 0 for a response that starts at the
        origin; pass the record's start for one that does not, e.g.
        ``Field.to_time_trace``'s ``times[0]``.
    band : (float, float), optional
        ``(low, high)`` in Hz to keep.
    axis : int, default -1
        Time axis of ``h``. Every other axis is carried through untouched,
        so a ``(depth, range, time)`` block of responses transforms in one
        call — which is how :meth:`~uacpy.Field.to_transfer_function` uses
        this.
    who : str, optional
        Name to put in the refusals. A method that delegates here passes
        its own, so the message names the call the user made rather than
        this function, which they may never have heard of.

    Returns
    -------
    (ndarray, ndarray)
        ``(frequencies, H)``. ``H`` keeps ``h``'s shape with ``axis``
        replaced by the kept bins.

    Raises
    ------
    ConfigurationError
        A zero-dimensional ``h`` or one with fewer than two samples along
        ``axis``, an ``axis`` the array does not have, a non-positive or
        non-finite ``sample_rate``, a non-finite ``t0``, or a ``band``
        keeping no bin.
    """
    who = who or "transfer_function_from_impulse_response"
    samples = np.asarray(h)
    if samples.ndim == 0:
        raise ConfigurationError(
            f"{who}: h must have a time axis; got a scalar.")
    try:
        requested = int(axis)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{who}: axis must be an integer; got {axis!r}.") from exc
    if not -samples.ndim <= requested < samples.ndim:
        raise ConfigurationError(
            f"{who}: axis={requested} is not an axis of an array with "
            f"shape {samples.shape}.")
    axis = requested % samples.ndim
    if samples.shape[axis] < 2:
        raise ConfigurationError(
            f"{who}: h needs at least two samples along axis {axis}; got "
            f"{samples.shape[axis]} (shape {samples.shape}).")
    if np.iscomplexobj(samples):
        samples = samples.real
    samples = samples.astype(float)
    fs = float(sample_rate)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ConfigurationError(
            f"{who}: sample_rate must be positive and finite; got "
            f"{sample_rate!r}.")
    if not np.isfinite(float(t0)):
        raise ConfigurationError(
            f"{who}: t0 must be finite; got {t0!r}.")
    n_time = samples.shape[axis]
    frequencies = np.fft.rfftfreq(n_time, 1.0 / fs)
    H = np.fft.rfft(samples, axis=axis)
    # The rotation multiplies along the frequency axis, which is `axis` in
    # the result; broadcasting it needs that shape, not a bare 1-D vector.
    shape = [1] * H.ndim
    shape[axis] = frequencies.size
    H = H * np.exp(-2j * np.pi * frequencies * float(t0)).reshape(shape)
    if band is not None:
        low, high = float(band[0]), float(band[1])
        # An edge bin sits ON `low` or `high` by construction whenever the
        # band came from the grid that produced h, and a bare `>=`/`<=`
        # then keeps or drops it on floating-point dust: re-inverting a
        # sample rate moved the 995 Hz edge by 1e-13 Hz and cost a bin.
        # A billionth of a bin is far below the df that separates
        # neighbours, so this admits the edge and nothing else.
        tol = 1e-9 * (fs / n_time)
        keep = (frequencies >= low - tol) & (frequencies <= high + tol)
        if not keep.any():
            raise ConfigurationError(
                f"{who}: band=({low:g}, {high:g}) Hz keeps no bin of a "
                f"spectrum spanning {frequencies[0]:g} to "
                f"{frequencies[-1]:g} Hz at {fs / n_time:g} Hz "
                f"spacing.")
        frequencies = frequencies[keep]
        H = np.compress(keep, H, axis=axis)
    return frequencies, H


def impulse_response_from_transfer_function(H, frequencies, sample_rate: float,
                                            n_samples: Optional[int] = None):
    """Real impulse response from a one-sided transfer function ``H(f)``.

    Resamples ``H`` onto a uniform DFT grid ``[0, fs/2]`` and inverse-transforms.
    ``frequencies`` must be non-negative and increasing. Grid bins outside
    ``[frequencies[0], frequencies[-1]]`` are **zero** — a band-limited model
    ``H(f)`` carries no energy out of band (constant extrapolation would
    fabricate a DC/high-frequency plateau in the impulse response).

    The DFT grid spacing ``df`` sets the **unambiguous delay window**
    ``1/df``. An arrival later than that wraps to ``tau mod 1/df`` and is
    then indistinguishable from a genuine early one — measured, ``df = 62.5``
    Hz (a 16 ms window) puts a 20 ms delay at 4.000 ms, and nothing in the
    returned ``h`` reveals it. No check here can catch it, so the caller
    sizes the grid: ``df < 1 / tau_max`` for the longest delay the channel
    can produce (``range_max / c_min`` for a propagation model).
    ``df = sample_rate / n_samples`` when ``n_samples`` is given, else the
    spacing of ``frequencies`` (its smallest spacing if it is non-uniform), so
    the default grid resolves every delay the supplied ``H(f)`` resolves. A
    band-limited ``H`` therefore costs a full-band grid: 100-200 Hz on a 1 Hz
    spacing at ``fs = 10`` kHz returns a 10 000-sample response whose 5 001
    grid bins are zero outside the supplied 101. Pass ``n_samples`` to trade
    that window down. Defaults above ``2**22`` samples raise rather than
    allocate.

    Returns ``(t, h)``.
    """
    f = np.asarray(frequencies, dtype=float)
    Hc = np.asarray(H, dtype=complex)
    if f.ndim != 1 or f.shape != Hc.shape:
        raise ConfigurationError(
            f"impulse_response_from_transfer_function: H and frequencies "
            f"shapes differ — frequencies is {f.shape}, H is {Hc.shape}. "
            f"Both must be the same 1-D shape: one transfer-function sample "
            f"per frequency.")
    if f.size and (np.any(np.diff(f) <= 0) or f[0] < 0):
        raise ConfigurationError(
            f"impulse_response_from_transfer_function: frequencies must be "
            f"non-negative and strictly increasing; got first={f[0]:g} Hz and "
            f"smallest step "
            f"{np.min(np.diff(f)) if f.size > 1 else float('nan'):g} Hz. "
            f"Sort the axis and drop duplicates before calling.")
    require_increasing_axis(
        f, "impulse_response_from_transfer_function: frequencies")
    # The shared guard rather than a bare float(): every check below divides by
    # fs or by fs/2, so an unvalidated rate escaped as an untyped error about
    # the wrong thing — sample_rate=0 with frequencies[0]=0 raised a bare
    # ZeroDivisionError, NaN a "cannot convert float NaN to integer", Inf an
    # OverflowError, and a negative rate a ConfigurationError announcing "the
    # Nyquist frequency -500 Hz", a true statement about the wrong argument.
    fs = require_positive_finite_scalar(
        sample_rate, "impulse_response_from_transfer_function",
        "sample_rate", " Hz")
    # The DFT grid stops at fs/2, and out-of-grid bins are zero (see above), so
    # a band sitting above Nyquist contributes nothing: it came back as an
    # all-zero h with no diagnostic. Partial overlap is legitimate but lossy —
    # a band straddling fs/2 silently loses the half above it — so it warns.
    nyquist = fs / 2.0
    if f.size and f[0] > nyquist:
        raise ConfigurationError(
            f"impulse_response_from_transfer_function: frequencies span "
            f"{f[0]:g}-{f[-1]:g} Hz, entirely above the Nyquist frequency "
            f"sample_rate/2 = {nyquist:g} Hz, so every DFT bin would be zero "
            f"and h all-zero. Raise sample_rate above {2 * f[-1]:g} Hz, or "
            f"pass a band inside [0, sample_rate/2].")
    if f.size and f[-1] > nyquist:
        warnings.warn(
            f"impulse_response_from_transfer_function: frequencies reach "
            f"{f[-1]:g} Hz, above the Nyquist frequency sample_rate/2 = "
            f"{nyquist:g} Hz; the part of H above it is dropped from h. Raise "
            f"sample_rate above {2 * f[-1]:g} Hz to keep the whole band.",
            UserWarning, stacklevel=2)
    if n_samples is None:
        # The grid spacing, not the bin count, is what the caller sizes the
        # delay window with, so the default follows the spacing of
        # `frequencies`: n = fs/df. Sizing it as 2*(f.size - 1) instead — the
        # rfft bin count — agrees with that only when `frequencies` spans the
        # whole of [0, fs/2]; on a band-limited vector it silently gave a much
        # coarser grid than the caller's own spacing (100-200 Hz at 1 Hz
        # spacing, fs = 10 kHz: df = 50 Hz, a 20 ms window, wrapping a 30 ms
        # arrival onto 10 ms). The smallest spacing is used when `frequencies`
        # is non-uniform, so no part of it is under-resolved.
        if f.size < 2:
            n_samples = 2
        else:
            df = float(np.min(np.diff(f)))
            n_samples = max(2, int(round(fs / df)))
            if n_samples > _MAX_DEFAULT_IR_SAMPLES:
                raise ConfigurationError(
                    f"impulse_response_from_transfer_function: the spacing of "
                    f"frequencies ({df:g} Hz) implies a default grid of "
                    f"{n_samples} samples at fs = {fs:g} Hz, above the "
                    f"{_MAX_DEFAULT_IR_SAMPLES}-sample default limit. Pass "
                    f"n_samples explicitly (its 1/df delay window must still "
                    f"exceed the longest arrival), or coarsen frequencies."
                )
    grid = np.fft.rfftfreq(int(n_samples), d=1.0 / fs)
    Hr = (np.interp(grid, f, Hc.real, left=0.0, right=0.0)
          + 1j * np.interp(grid, f, Hc.imag, left=0.0, right=0.0))
    h = np.fft.irfft(Hr, n=n_samples)
    t = np.arange(n_samples) / fs
    return t, h


# ──────────────────────────────────────────────────────────────────────
# Modal dispersion
#
# Group velocity in a waveguide, and the warping
# that straightens a dispersed mode.
# ──────────────────────────────────────────────────────────────────────

def modal_group_velocity(frequencies, k_horizontal):
    """Group velocity ``v_g = d(omega)/d(k_r)`` per mode from the dispersion.

    Parameters
    ----------
    frequencies : 1-D array
        Frequencies (Hz), strictly increasing.
    k_horizontal : array
        Horizontal wavenumber (rad/m). Shape ``(n_freq,)`` for one mode or
        ``(n_freq, n_modes)`` for several. May be complex (KRAKENC's lossy
        modes): the group velocity is then ``d(omega)/d(Re k_r)`` — for
        weakly attenuated modes the propagation speed follows the real part
        of the wavenumber, while ``Im(k_r)`` is the modal attenuation and
        controls amplitude decay, not travel time (Jensen et al., *COA*,
        Sects. 2.4.5 and 5.9.2).

    Returns
    -------
    ndarray
        Group velocity (m/s), same shape as ``k_horizontal``.

    Notes
    -----
    **A finer frequency grid is not always a better derivative.** The centred
    difference's truncation error falls as the frequency step squared, but a
    ``k_horizontal`` read from a model file arrives quantized — KRAKEN's
    ``.mod`` record is ``COMPLEX*8`` — and the difference then carries about
    one storage step of noise however fine the grid is, contributing
    ``spacing(k_r)/|Δk_r|`` relative. On an exact-root ideal-waveguide control,
    refining a 40-point 30-70 Hz sweep to 400 points leaves the float64 answer
    9x more accurate and the float32 one 9x worse.

    A grid whose storage floor exceeds 1e-5 is warned about. Because this
    function is handed the *whole* sweep, the warning carries a measured
    remedy rather than a prescribed one: decimating the grid raises the
    truncation error as the square of the decimation and lowers the floor in
    proportion to it, so the widest spacing whose answer still agrees with its
    doubly-decimated self to within the floor is found by walking, and the
    recommendation is one doubling back from there. Over 30 firings on five
    ideal waveguides that step was never worse than the grid in hand and 3.3x
    better at the median; where the walk cannot widen, the message says so and
    gives the test for whether refining would help instead.

    Raises
    ------
    ConfigurationError
        ``frequencies`` is not 1-D strictly increasing or has fewer than two
        samples; ``k_horizontal``'s leading axis does not match it; or a mode
        column of ``k_horizontal`` does not rise strictly with frequency —
        flat, falling throughout, or doubling back, each named separately.
        One mode's ``k_r`` rises strictly, because ``v_g`` is the
        energy-transport speed and ``d(k_r)/d(omega) = 1/v_g`` is positive
        everywhere; a flat step divides by zero and a falling one returns a
        negative speed. Note that ``v_g`` itself is *not* monotonic — it dips
        through the Airy minimum (Jensen et al., *COA*, Fig. 2.28b) — while
        ``k_r`` still climbs.
    """
    f = np.asarray(frequencies, dtype=float)
    # Complex wavenumbers carry the modal attenuation in the imaginary part;
    # the dispersion (and hence the group velocity) lives in the real part.
    kr = np.real(np.asarray(k_horizontal)).astype(float)
    if f.ndim != 1 or np.any(np.diff(f) <= 0):
        detail = (f" with {int(np.count_nonzero(np.diff(f) <= 0))} "
                  "non-increasing step(s)") if f.ndim == 1 else ""
        raise ConfigurationError(
            "modal_group_velocity: frequencies must be 1-D increasing; got "
            f"shape {f.shape}{detail}")
    require_increasing_axis(f, "modal_group_velocity: frequencies")
    if f.size < 2:
        # np.gradient below needs two samples per axis; one reached it as
        # "Shape of array too small to calculate a numerical gradient",
        # naming no input the caller supplied.
        raise ConfigurationError(
            f"modal_group_velocity: frequencies needs at least 2 samples — "
            f"the group velocity is a numerical derivative d(omega)/d(kr) "
            f"over the axis; got {f.size}.")
    omega = 2.0 * np.pi * f
    if kr.shape[0] != f.size:
        raise ConfigurationError(
            "modal_group_velocity: k_horizontal axis 0 must match freqs; got "
            f"k_horizontal shape {kr.shape} and {f.size} frequencies")
    # Both gradients are taken in index space (unit spacing). Their ratio is
    # (domega/di)/(dkr/di) = domega/dkr, so no frequency spacing is needed and
    # a non-uniform `frequencies` grid is handled without extra care.
    dkr = np.gradient(kr) if kr.ndim == 1 else np.gradient(kr, axis=0)
    # A zero step in kr is the one value the division cannot survive: it
    # returns inf, announced by nothing but numpy's own "divide by zero".
    # Tested exactly rather than against a tolerance, because exact zero is
    # what produces the inf and a *small* step is the physical near-cutoff
    # case (kr flattens in range, not in frequency) that must keep working.
    flat = dkr == 0.0
    if np.any(flat):
        by_frequency = np.any(flat.reshape(flat.shape[0], -1), axis=1)
        first = int(np.flatnonzero(by_frequency)[0])
        raise ConfigurationError(
            f"modal_group_velocity: k_horizontal is flat in frequency at "
            f"{int(np.count_nonzero(by_frequency))} of {f.size} frequency "
            f"sample(s), first at index {first} ({f[first]:g} Hz). The group "
            f"velocity is d(omega)/d(kr), so a flat step divides by zero and "
            f"comes back inf. A propagating mode's horizontal wavenumber "
            f"rises strictly with frequency — check this is one mode's "
            f"dispersion curve sampled across frequency, not a constant "
            f"or several modes "
            f"stacked on the frequency axis.")
    # The rest of the contract: k_r must RISE with frequency. v_g is the speed
    # energy travels at, so it is positive and no larger than the fastest
    # medium speed, and d(kr)/d(omega) = 1/v_g is therefore positive at every
    # frequency. v_g itself is *not* monotonic — it dips through the Airy
    # minimum (Jensen et al., COA Sect. 2.4.4.4 and Fig. 2.28b) — but k_r
    # never falls. A falling step hands back a negative speed, which nothing
    # downstream reads as an error: it is a travel-time denominator.
    #
    # The two ways a column can fall want different remedies, so they are
    # reported separately: a column that falls *throughout* is a curve stored
    # the wrong way round, while one that both rises and falls is a mixed or
    # mode-shifted set. Each mode column is judged on its own.
    columns = dkr.reshape(dkr.shape[0], -1)
    rising = np.any(columns > 0, axis=0)
    falling = np.any(columns < 0, axis=0)
    turning = rising & falling
    if np.any(turning):
        col = columns[:, int(np.flatnonzero(turning)[0])]
        turn = int(np.flatnonzero(np.sign(col[1:]) != np.sign(col[:-1]))[0]) + 1
        raise ConfigurationError(
            f"modal_group_velocity: k_horizontal doubles back in frequency in "
            f"{int(np.count_nonzero(turning))} of {columns.shape[1]} mode "
            f"column(s), first turning at index {turn} ({f[turn]:g} Hz). The "
            f"group velocity is d(omega)/d(kr), the speed energy travels at, "
            f"so a step that reverses sign hands back a negative speed there "
            f"while the rest of the curve looks ordinary. A propagating mode's "
            f"horizontal wavenumber rises strictly with frequency — its phase "
            f"speed is bounded by the medium sound speeds — so a curve that "
            f"turns is not one mode's dispersion: check for several modes "
            f"stacked on the frequency axis, or a mode index that shifted "
            f"where the model's mode list changed length across a cutoff.")
    # Reached only once ``turning`` is empty, so no column mixes the two signs
    # and no step is zero — a column that falls at all therefore falls
    # throughout. That is why this branch has to stay BELOW the one above: run
    # first, it would claim a doubling-back curve falls all the way and send
    # the caller to reverse an axis that is not the problem.
    if np.any(falling):
        raise ConfigurationError(
            f"modal_group_velocity: k_horizontal falls with frequency "
            f"throughout {int(np.count_nonzero(falling))} of "
            f"{columns.shape[1]} mode column(s) — over "
            f"{f[0]:g}-{f[-1]:g} Hz. A propagating mode's horizontal "
            f"wavenumber rises strictly with frequency, because the group "
            f"velocity d(omega)/d(kr) is the speed energy travels at and is "
            f"positive; a falling curve makes every group velocity in the "
            f"column negative. The usual cause is a mode set stored against a "
            f"descending frequency axis: reverse k_horizontal (and "
            f"frequencies with it) so both run low to high.")
    domega = np.gradient(omega)
    v_g = domega / dkr if kr.ndim == 1 else domega[:, None] / dkr
    # The floor is measured on the step between neighbouring frequencies, not
    # on np.gradient's centred half-step, so the same expression covers both
    # this function and Modes.compute_group_velocity, which differences one
    # step directly. dkr is aligned with the FIRST of each pair.
    step = np.diff(kr, axis=0)
    # `grid=` lets the notice prescribe a MEASURED step, which needs a grid
    # it can decimate. Two frequencies cannot separate truncation from
    # storage — that is the case warn_if_storage_under_resolves documents
    # as giving the test instead of the step — so the grid is withheld and
    # the notice falls back to the advice that does not need one.
    warn_if_storage_under_resolves(
        kr[:-1], step, v_g[:-1], "modal_group_velocity",
        grid=(omega, kr) if omega.size >= 3 else None)
    return v_g


#: Elements of the dense sinc matrix :func:`_resample_uniform` builds at a
#: time. Whittaker-Shannon is O(N·K), so the whole matrix for a long record
#: does not fit in memory; 2e6 doubles is 16 MB per block, and the block loop
#: costs nothing next to the matrix product itself.
_SINC_BLOCK_ELEMENTS = 2_000_000

_INTERPOLATIONS = ('linear', 'sinc')


def _resample_uniform(values, t0: float, dt: float, t_query, *, method: str,
                      who: str):
    """``values``, sampled uniformly from ``t0`` at step ``dt``, read at
    ``t_query``.

    ``'linear'`` is :func:`numpy.interp`. ``'sinc'`` is the
    Whittaker-Shannon interpolation of Bonnel et al. (2020) Eq. (C12),
    ``y(t) = sum_n y[n] sinc(t f_s - n)``, which the paper advises over linear
    because linear "sometimes creates high frequency artifacts in the warped
    signal". It is exact for a signal band-limited to the grid it is read
    from — and faithful to the aliased content of one that is not, which is
    why it is not unconditionally better (see :func:`warp_signal`).
    """
    if method not in _INTERPOLATIONS:
        raise ConfigurationError(
            f"{who}: interpolation must be one of "
            f"{', '.join(repr(m) for m in _INTERPOLATIONS)}; got {method!r}.")
    values = np.asarray(values, dtype=float)
    t_query = np.asarray(t_query, dtype=float)
    if method == 'linear':
        t_src = t0 + np.arange(values.size) * dt
        return np.interp(t_query, t_src, values)
    # ``np.sinc`` is the normalised sinc(x) = sin(pi x)/(pi x), which is the
    # kernel Eq. (C12) is written in: the argument is the query's position in
    # units of the source sample index.
    index = (t_query - t0) / dt
    n_src = np.arange(values.size)
    out = np.empty(index.size, dtype=float)
    block = max(1, _SINC_BLOCK_ELEMENTS // max(values.size, 1))
    for start in range(0, index.size, block):
        stop = min(start + block, index.size)
        out[start:stop] = np.sinc(index[start:stop, None] - n_src) @ values
    return out


def _prescribed_warped_length(n: int, fs: float, t_r: float) -> int:
    """``K`` of Bonnel et al. (2020) Eq. (14), for a record of ``n`` samples.

    Eq. (13) sets the warped sampling rate ``f_s^h = 2/dt_N`` with
    ``dt_N = (1/f_s)·t_max/h^-1(t_max)``, i.e.
    ``f_s^h = 2·f_s·h^-1(t_max)/t_max``; Eq. (14) then takes
    ``K = ceil([h^-1(t_max) - h^-1(t_min)]·f_s^h)``, and ``h^-1(t_min) = 0``
    here because the record starts at ``t_r`` (App. C 1, C 2.b).

    The factor 2 is the paper's margin over its own Nyquist bound Eq. (C8),
    ``f_s^h > [h^-1(t_max)/t_max]·f_s``.
    """
    t_max = t_r + (n - 1) / fs
    span = float(np.sqrt(max(t_max ** 2 - t_r ** 2, 0.0)))
    if span <= 0.0:
        return 2
    fs_h = 2.0 * fs * span / t_max
    return max(2, int(np.ceil(span * fs_h)))


def warp_signal(signal, sample_rate: float, range_m: float,
                c: float = DEFAULT_SOUND_SPEED, *,
                oversample: Optional[float] = None,
                interpolation: str = 'linear'):
    """Warp an impulsive shallow-water arrival to linearise ideal-waveguide dispersion.

    Maps original (reduced) time ``t`` to warped time ``t_w = sqrt(t^2 - t_r^2)``
    with ``t_r = range/c``, so each ideal-waveguide mode collapses to a single
    warped frequency. ``signal`` is assumed to start at the direct-wave arrival
    ``t_r``.

    The transform was introduced by Le Touzé et al. (2009); the form and the
    numerical recipe implemented here are Bonnel, Thode, Wright & Chapman
    (2020), *JASA* **147**(3) 1897-1926, p. 1907 — the resampling
    ``h(t) = sqrt(t^2 + t_r^2)`` of Eq. (10), whose inverse
    ``h^-1(t) = sqrt(t^2 - t_r^2)`` (Eq. 11) is the warped axis returned here,
    applied with the ``sqrt(|h'(t)|)`` weight of Eq. (7) that "ensures energy
    conservation" — written discretely as ``sqrt(t_w/h(t_w))`` in Eq. (15),
    which is the form below.

    The operator is derived there for the ideal isovelocity waveguide. On a
    real profile the warped modes are, in that paper's words, "not the
    theoretically predicted pure tones ... but instead are tilted and slightly
    curved" — still separable, but no longer single warped frequencies.

    Parameters
    ----------
    range_m, c : float
        **Trial** parameters, not measurements. They enter only through
        ``t_r = range_m/c``, and warping is "only weakly sensitive to the
        choice of ``t_r``": the paper states that "it is not required to know
        the range nor the water sound speed to apply warping", and warps every
        signal in the tutorial — experimental ones included — at r = 10 km,
        c = 1500 m/s while the true ranges are 5-15 km. Warp followed by
        inverse warp cancels ``t_r`` exactly, so a downstream localisation is
        independent of the value used. What the result *is* sensitive to is
        the **time origin** of ``signal``: see Notes.
    oversample : float, optional
        Length of the warped axis as a multiple of the input length;
        fractional factors are honoured (the length is rounded), and a factor
        below 1 is refused rather than clamped.

        ``None`` (the default) takes the prescription of Eqs. (13)-(14)
        instead: a warped rate ``f_s^h = 2/dt_N`` with
        ``dt_N = (1/f_s)·t_max/h^-1(t_max)``, and
        ``K = ceil(h^-1(t_max)·f_s^h)`` samples. In this function's terms that
        is a factor ``2·(1 + t_r/t_max)``, always in (2, 4), and the factor 2
        in it is the paper's own margin over its Nyquist bound Eq. (C8),
        ``f_s^h > [h^-1(t_max)/t_max]·f_s``.

        That bound is why a fixed factor of 1 is not merely a coarse choice:
        the map is expansive (``dt_w/dt = t/t_w > 1``), so ``oversample=1``
        puts the warped grid **below** Eq. (C8) by exactly ``(1 + t_r/t_max)``
        for every range and rate, and the round trip ``warp -> unwarp`` loses
        the top of the band. Measured on this implementation, relative
        round-trip error on white noise (broadband to Nyquist, so it loses the
        most) over sample rates 2-10 kHz and ranges 0.1-20 km: **46.5-60.5 %
        at ``oversample=1``, 15.6-20.6 % at the prescription**, 5.8-7.9 % at a
        fixed 8. On the band-limited transient the warp exists for — the 40 + 120 Hz
        Hann-windowed pair at 10 kHz over 500 m used by
        ``tests/test_modal_warping.py`` — 0.067 %, 0.019 % and 0.0079 %. At a fixed
        factor the error roughly halves with each doubling.
    interpolation : {'linear', 'sinc'}, optional
        How ``signal`` is read at the off-grid times the warp asks for.
        ``'linear'`` (the default) is :func:`numpy.interp`. ``'sinc'`` is the
        Whittaker-Shannon interpolation of Eq. (C12),
        ``y(t) = sum_n y[n] sinc(t f_s - n)``, which the paper advises because
        linear interpolation "sometimes creates high frequency artifacts in
        the warped signal".

        Two things make it an opt-in rather than the default. It is **only**
        an improvement on a grid that satisfies Eq. (C8): at ``oversample=1``
        it makes the white-noise round trip *worse* — 65 % against linear's
        58 % at 10 kHz over 1 km — because an exact reconstruction faithfully
        reproduces the aliased content that linear interpolation was
        accidentally smoothing away. And it is an O(N·K) dense kernel,
        quadratic in record length: of order 1500-1700x
        :func:`numpy.interp`'s cost, measured at 380-430 ms against 0.24-0.26
        ms for ``n=2048``, ``K=7227`` over two runs. Paired with the default
        grid it makes the round trip essentially exact — 0.0036-0.0067 % on
        the same white-noise grid, and numerically exact on the band-limited
        transient — and it must be passed to :func:`unwarp_signal` as well to
        get that.

    Returns
    -------
    warped : ndarray
        The resampled signal on the warped time grid, Jacobian-weighted so the
        warp is energy-preserving.
    t_warp : ndarray
        The warped time axis (s) ``warped`` lives on, spanning
        ``[0, sqrt(t_end**2 - t_r**2)]`` — ``n * oversample`` samples, or the
        ``K`` of Eq. (14) when ``oversample`` is ``None``.

    Notes
    -----
    **The time origin is the sensitive parameter.** The paper's guidance, and
    the most useful thing it has to say to a caller here: pick an origin as
    close to ``r/c_w`` as possible by an iterative process — read the arrival
    time of the highest frequencies off the original spectrogram (they
    disperse least, so they arrive nearest ``t_r``), warp, inspect the warped
    spectrogram, iterate. The failure is asymmetric. Too early and "the modes
    are definitely not horizontal tones, but span a wider bandwidth across the
    warped spectrum", overlapping and interfering. Too late and "the modes
    become virtually horizontal" and separate *better* — "however, this
    improved separation comes at a price: mode 1 has vanished". In a real
    waveguide energy also arrives *before* ``r/c_w``, through the seabed, and
    ``c_w`` is not uniquely defined unless the water column is isovelocity.


    **Signal first, axis second — deliberately, and against the package's
    usual axis-first convention** (stated at
    :func:`uacpy.acoustic_signal.synthesize_noise_from_psd`, and followed by
    every other function on the ``acoustic_signal`` public surface). This pair
    is ordered to feed :func:`unwarp_signal`, whose first two parameters are
    ``(warped, t_warp)`` in exactly this order, so the round trip is
    ``unwarp_signal(*warp_signal(x, fs, r, oversample=8), fs, r)``. Swapping
    this tuple to match the convention would silently invert that call:
    ``t_warp`` and ``warped``
    have the same length, so the argument-shape checks downstream cannot tell
    them apart and the error surfaces only as a wrong answer.
    """
    x = np.asarray(signal, dtype=float)
    fs = require_positive_finite_scalar(sample_rate, "warp_signal",
                                        "sample_rate", " Hz")
    n = x.size
    range_m = require_positive_finite_scalar(range_m, "warp_signal",
                                             "range_m", " m")
    c = require_positive_finite_scalar(c, "warp_signal", "c", " m/s")
    t_r = range_m / c
    # The record is taken to start at the direct arrival: t_min = [t_r]+, the
    # convention of App. C 1, under which the warped domain is Eq. (C1)'s
    # [0, sqrt(t_max^2 - t_r^2)] — the axis returned below.
    t = t_r + np.arange(n) / fs
    t_w = np.sqrt(np.maximum(t ** 2 - t_r ** 2, 0.0))
    if oversample is None:
        n_w = _prescribed_warped_length(n, fs, t_r)
    else:
        # Scale first and round after, so a fractional factor lengthens the
        # axis instead of truncating to the integer below it (int(1.5) == 1
        # makes the accuracy knob a no-op for every non-integer value). A
        # factor below 1 would shorten the warped axis, which is the opposite
        # of what the argument is for, so it is refused rather than clamped.
        try:
            os_factor = float(oversample)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"warp_signal: oversample must be a number >= 1; got "
                f"{oversample!r}.") from exc
        if not (os_factor >= 1.0):
            raise ConfigurationError(
                f"warp_signal: oversample must be >= 1 — it is the warped "
                f"axis' length as a multiple of the input's, and the warp is "
                f"expansive; got {oversample!r}.")
        n_w = max(2, int(round(n * os_factor)))
    tw_axis = np.linspace(t_w[0], t_w[-1], n_w)
    t_orig = np.sqrt(tw_axis ** 2 + t_r ** 2)
    warped = _resample_uniform(x, t[0], 1.0 / fs, t_orig,
                               method=interpolation, who='warp_signal')
    # Unitary Jacobian weighting — ``sqrt(|h'|)``, Bonnel et al. (2020)
    # Eq. (7). App. C 2.d works it through to the form used here: "Because
    # h'(t) = t/h(t), this factor is given as sqrt(|h'[k/f_s^h]|) =
    # sqrt(t_k/h(t_k)) for the kth sample of the warped signal", which is
    # Eq. (15)'s weight. So with h(t_w) = t = sqrt(t_w^2 + t_r^2) the
    # energy-preserving weight is sqrt(t_w / t) —
    # verified numerically against np.gradient(t, t_w), and by the resulting
    # E_warp/E_in being range-INDEPENDENT (the reciprocal inflates it by ~30x
    # at 20 km and grows with range). t_w = 0 at the direct arrival t = t_r, so
    # the numerator is floored at one sample.
    warped = warped * np.sqrt(np.maximum(tw_axis, 1.0 / fs) / t_orig)
    return warped, tw_axis


def unwarp_signal(warped, t_warp, sample_rate: float, range_m: float,
                  c: float = DEFAULT_SOUND_SPEED, *,
                  interpolation: str = 'linear'):
    """Inverse of :func:`warp_signal`; returns ``(t, signal)`` on the original grid.

    ``(warped, t_warp)`` are taken in the order :func:`warp_signal` returns
    them, so ``unwarp_signal(*warp_signal(x, fs, r), fs, r)`` round-trips. The
    **return** here is axis-first ``(t, signal)``, the package convention,
    because nothing consumes it positionally.

    Bonnel et al. (2020) Eq. (16), ``y_u[n] = sqrt(t_n/h^-1(t_n)) y_w[h^-1(t_n)]``,
    with the output rate and sample count "already known: they are the same as
    for the original signal" — which is what the grid below reconstructs from
    the warped axis' own extent.

    ``interpolation`` is :func:`warp_signal`'s, and must match the forward
    call for the round trip to be exact: ``'sinc'`` (Eq. C12) buys its
    accuracy only when both directions use it.
    """
    w = np.asarray(warped, dtype=float)
    tw = np.asarray(t_warp, dtype=float)
    fs = float(sample_rate)
    t_r = float(range_m) / float(c)
    # The output grid follows the warped axis' own extent, not ``w.size``:
    # ``warp_signal(oversample=k)`` returns k times as many samples over the
    # same warped span, and reading the length off the array would unwarp onto
    # a record k times too long, the tail of it extrapolated.
    t_end = float(np.sqrt(tw[-1] ** 2 + t_r ** 2))
    n = max(2, int(round((t_end - t_r) * fs)) + 1)
    t = t_r + np.arange(n) / fs
    t_w_of_t = np.sqrt(np.maximum(t ** 2 - t_r ** 2, 0.0))
    # Divide out the forward weight sqrt(t_w / t) applied by ``warp_signal``.
    w_unweighted = w / np.sqrt(np.maximum(tw, 1.0 / fs)
                               / np.sqrt(tw ** 2 + t_r ** 2))
    dt_w = (tw[-1] - tw[0]) / max(tw.size - 1, 1)
    signal = _resample_uniform(w_unweighted, tw[0], dt_w, t_w_of_t,
                               method=interpolation, who='unwarp_signal')
    return t, signal


# ──────────────────────────────────────────────────────────────────────
# Power delay profile
#
# A power delay profile is a list of delays and the power arriving at each.
# It comes from a model's arrival list, from a chirp sounding, from a
# measured channel in a file — these functions ask nothing about which.
# `Arrivals` supplies its own delays and absorption-corrected powers and
# wraps each of them.
# ──────────────────────────────────────────────────────────────────────

#: ``k`` of ``1 / (k tau_rms)``. ``inverse_spread`` is the corpus's own
#: statement (APL-UW TR 9407 sect. II.7.b p. II-32; Abraham sect. 8.7);
#: the Rappaport correlation rules (2nd ed. sect. 5.4.3, eqs 5.39-5.40)
#: come from outside the corpus, so they are options and not the default.
COHERENCE_BANDWIDTH_FACTORS = {'inverse_spread': 1.0,
                               'rappaport_0.5': 5.0,
                               'rappaport_0.9': 50.0}


@dataclass(frozen=True)
class ChannelRegime:
    """Verdict of :func:`channel_regime` for one symbol rate.

    ``frequency_selective`` is
    ``signal_bandwidth_hz > coherence_bandwidth_hz``:
    the symbol band spans more than one fade of the channel, so the symbols
    overlap their neighbours (``isi_symbols`` of them, the rms delay spread
    in symbol periods) and a flat gain cannot describe the link.
    """
    coherence_bandwidth_hz: float
    signal_bandwidth_hz: float
    rms_delay_spread_s: float
    symbol_duration_s: float
    frequency_selective: bool
    isi_symbols: float
    convention: str

    def __str__(self) -> str:
        verdict = ("frequency-selective" if self.frequency_selective
                   else "frequency-flat")
        sign = ">" if self.frequency_selective else "<="
        return (f"{verdict}: signal {self.signal_bandwidth_hz:g} Hz {sign} "
                f"coherence {self.coherence_bandwidth_hz:g} Hz "
                f"[{self.convention}] (rms delay spread "
                f"{self.rms_delay_spread_s:g} s = {self.isi_symbols:g} "
                f"symbols of {self.symbol_duration_s:g} s)")


def _profile(delays_s, powers, who):
    """The two arrays every function below takes, checked once."""
    delays = np.asarray(delays_s, dtype=float).ravel()
    power = np.asarray(powers, dtype=float).ravel()
    if delays.size != power.size:
        raise ConfigurationError(
            f"{who}: delays_s and powers must have the same length; got "
            f"{delays.size} and {power.size}.")
    if np.any(power < 0.0):
        raise ConfigurationError(
            f"{who}: powers must be non-negative — this is a POWER delay "
            f"profile, so pass |a|**2, not the complex amplitudes.")
    return delays, power


def rms_delay_spread(delays_s, powers, *,
                     who: str = "rms_delay_spread") -> float:
    """Energy-weighted spread of a power delay profile, in seconds.

    The second central moment of the profile: delays weighted by ``powers``,
    about their weighted mean. It measures how much the arrival pattern
    smears a pulse in time, so it bounds the time resolution any processing
    of the channel can have — the smearing of a transmitted pulse, the
    length a replica or matched filter has to cover, the interval a symbol
    would have to exceed to avoid overlapping its neighbour.

    Prefer it to the peak-to-peak spread ``ptp(delays)``, which is set by
    whichever path arrives last no matter how faint: on a 1 km
    bottom-to-bottom path in 1000 m of water at 40 kHz the two differ by
    more than two orders of magnitude, because a path tens of dB down lands
    seconds late while almost all the energy arrives within a millisecond
    of the first.

    Parameters
    ----------
    delays_s : array_like
        Arrival delays (s). Order does not matter.
    powers : array_like
        Power at each delay, same length. Non-negative: pass ``|a|**2``.
        Any consistent scaling works — the result is scale-invariant.

    Returns
    -------
    float
        Seconds. ``0.0`` for a single arrival and for a profile carrying no
        energy at all; ``nan`` if a delay or power is non-finite, rather
        than a spread computed from whatever else was finite.

    Notes
    -----
    Its reciprocal is the frequency scale over which the transfer function
    decorrelates — see :func:`coherence_bandwidth`, which states the
    constant relating the two.
    """
    delays, power = _profile(delays_s, powers, who)
    total = float(power.sum())
    if delays.size < 2 or total <= 0.0:
        return 0.0
    weights = power / total
    mean = float((weights * delays).sum())
    return float(np.sqrt((weights * (delays - mean) ** 2).sum()))


def energy_support(delays_s, powers, fraction: float = 0.999, *,
                   who: str = "energy_support") -> float:
    """Delay span holding ``fraction`` of a profile's energy, in seconds.

    Measured from the first arrival to the one by which ``fraction`` of the
    energy has arrived. It answers the question a synthesis window asks —
    how long does the response have to be? — which neither of the other two
    measures does: ``ptp(delays)`` is an extremum, moved by one faint
    straggler however little it carries, and :func:`rms_delay_spread` is a
    second moment, a width rather than a span the energy fits inside.

    Parameters
    ----------
    delays_s, powers : array_like
        The profile, as for :func:`rms_delay_spread`.
    fraction : float, default 0.999
        Share of the total energy the span must hold, in ``(0, 1]``. ``1.0``
        is the peak-to-peak span. The default leaves a thousandth of the
        energy — 30 dB down — outside.

    Returns
    -------
    float
        Seconds. ``0.0`` for a single arrival and for a profile carrying no
        energy at all; ``nan`` if a delay or power is non-finite.
    """
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ConfigurationError(
            f"{who}: fraction={fraction:g} is not a share of the energy. "
            f"Pass 0 < fraction <= 1 (1.0 spans every arrival, i.e. the "
            f"peak-to-peak delay).")
    delays, power = _profile(delays_s, powers, who)
    if delays.size < 2:
        return 0.0
    if not (np.all(np.isfinite(delays)) and np.all(np.isfinite(power))):
        return float('nan')
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    order = np.argsort(delays)
    delays = delays[order]
    # The cumulative share is monotone, so the first entry at or above the
    # target is the last arrival that has to fit. Rounding can leave the
    # final entry a hair under 1.0, which would put the index one past the
    # end, so clamp it.
    cumulative = np.cumsum(power[order]) / total
    cut = min(int(np.searchsorted(cumulative, fraction, side='left')),
              delays.size - 1)
    return float(delays[cut] - delays[0])


def coherence_factor(convention: str = 'inverse_spread', factor=None, *,
                     who: str = "coherence_factor") -> float:
    """The ``k`` of ``1 / (k tau_rms)``: ``factor`` when given (any finite
    ``k > 0``), else the named convention's — see
    :data:`COHERENCE_BANDWIDTH_FACTORS`."""
    if factor is not None:
        factor = float(factor)
        if not (np.isfinite(factor) and factor > 0.0):
            raise ConfigurationError(
                f"{who}: factor must be a finite number > 0 (the k of "
                f"1 / (k * tau_rms)); got {factor!r}. Leave it out to "
                f"use convention={convention!r}.")
        return factor
    try:
        return COHERENCE_BANDWIDTH_FACTORS[convention]
    except KeyError:
        raise ConfigurationError(
            f"{who}: convention must be one of "
            f"{sorted(COHERENCE_BANDWIDTH_FACTORS)} (1/tau_rms, "
            f"1/(5 tau_rms), 1/(50 tau_rms)), or pass factor=k for "
            f"1/(k tau_rms); got {convention!r}.") from None


def coherence_bandwidth(delays_s, powers, *,
                        convention: str = 'inverse_spread',
                        factor=None, who: str = "coherence_bandwidth"
                        ) -> float:
    """Bandwidth over which a channel's transfer function stays correlated,
    in Hz, as ``1 / (k * tau_rms)`` with ``tau_rms`` the
    :func:`rms_delay_spread` of the profile.

    The default ``k = 1`` is the convention the corpus states: "the inverse
    [of the elongation time] in hertz is a measure of the coherence
    bandwidth of the channel" (APL-UW TR 9407, sect. II.7.b, p. II-32) and
    ``W < 1 / sigma_t = W_c`` (Abraham, *Underwater Acoustic Signal
    Processing*, sect. 8.7, Fig. 8.34: 33 ms of spreading gives 30 Hz). The
    named options ``'rappaport_0.5'`` (``k = 5``) and ``'rappaport_0.9'``
    (``k = 50``) are the 0.5- and 0.9-correlation rules of Rappaport,
    *Wireless Communications*, 2nd ed., sect. 5.4.3, eqs 5.39-5.40 — a
    source outside the corpus, so they are options and not the default.

    Returns ``inf`` for a single arrival (no spread, a flat channel), and
    ``nan`` when the spread is.
    """
    k = coherence_factor(convention, factor, who=who)
    spread = rms_delay_spread(delays_s, powers, who=who)
    if not np.isfinite(spread):
        return float('nan')
    if spread <= 0.0:
        return float('inf')
    return 1.0 / (k * spread)


def channel_regime(delays_s, powers, symbol_rate: float, *,
                   convention: str = 'inverse_spread', factor=None,
                   rolloff: float = 0.0,
                   who: str = "channel_regime") -> ChannelRegime:
    """Whether a modem at ``symbol_rate`` sees this profile as flat or
    frequency-selective.

    Compares the signal bandwidth ``(1 + rolloff) * symbol_rate`` with
    :func:`coherence_bandwidth` under ``convention`` / ``factor``:
    selective when the signal is the wider (Proakis, *Digital
    Communications*, 4th ed., sect. 14.1.2; Stojanovic and Preisig 2009,
    sect. II). ``isi_symbols`` is the rms delay spread in symbol periods,
    the number of neighbours each symbol overlaps. The result records the
    convention it was judged under (``factor=k`` is recorded as
    ``'factor=k'``).

    Parameters
    ----------
    delays_s, powers : array_like
        The profile, as for :func:`rms_delay_spread`.
    symbol_rate : float
        Symbol rate (Bd).
    convention, factor
        As on :func:`coherence_bandwidth`.
    rolloff : float, default 0.0
        Excess bandwidth of the pulse; ``0`` takes the Nyquist bandwidth
        equal to the symbol rate.
    """
    symbol_rate = float(symbol_rate)
    if not (np.isfinite(symbol_rate) and symbol_rate > 0.0):
        raise ConfigurationError(
            f"{who}: symbol_rate must be positive and finite (Bd); got "
            f"{symbol_rate!r}.")
    rolloff = float(rolloff)
    if not 0.0 <= rolloff <= 1.0:
        raise ConfigurationError(
            f"{who}: rolloff must be in [0, 1]; got {rolloff!r}.")
    k = coherence_factor(convention, factor, who=who)
    coherence = coherence_bandwidth(delays_s, powers, factor=k, who=who)
    spread = rms_delay_spread(delays_s, powers, who=who)
    signal = (1.0 + rolloff) * symbol_rate
    return ChannelRegime(
        coherence_bandwidth_hz=coherence,
        signal_bandwidth_hz=signal,
        rms_delay_spread_s=spread,
        symbol_duration_s=1.0 / symbol_rate,
        frequency_selective=bool(signal > coherence),
        isi_symbols=spread * symbol_rate,
        convention=(str(convention) if factor is None
                    else f"factor={float(factor):g}"),
    )


# ──────────────────────────────────────────────────────────────────────
# Transfer-function operations
#
# Building an H(f) from a path list, averaging one over a band, and
# cutting one down to the paths a pulse of a given length can overlap.
# ──────────────────────────────────────────────────────────────────────


def arrival_transfer_function(frequencies, amplitudes, delays_s, *,
                              delays_imag_s=None, phases_rad=None,
                              phase_offset: float = 0.0,
                              who: str = "arrival_transfer_function"):
    """``H(f)`` of a discrete multipath arrival list.

    .. math::
        H(f) = \\sum_i A_i\\,e^{i\\varphi_i}\\,
               e^{\\omega\\,\\mathrm{Im}\\tau_i}\\,
               e^{-i\\omega\\,\\mathrm{Re}\\tau_i}

    The absorption term is separate because ray codes put volume absorption
    in the **imaginary travel time**, not in the amplitude: Bellhop writes
    ``delay_imag`` as its own field (``ArrMod.f90:118-125``), so the
    received amplitude is ``A·exp(ω·Im τ)`` and grows more negative with
    frequency across a band. Scoring on ``A`` alone treats a late, heavily
    absorbed path as though the water were lossless.

    Parameters
    ----------
    frequencies : array_like
        Frequencies (Hz) to evaluate at.
    amplitudes : array_like
        Arrival amplitudes, as **non-negative magnitudes**. A sign belongs
        in ``phases_rad`` as ``pi``, not here — see below.
    delays_s : array_like
        Real travel times (s), same length as ``amplitudes``.
    delays_imag_s : array_like, optional
        Imaginary travel times (s). ``None`` means a lossless list, i.e.
        zeros — which is what a hand-built arrival set without the field
        carries.
    phases_rad : array_like, optional
        Per-arrival phase (radians). ``None`` means zeros.
    phase_offset : float, default 0
        Added to every ``phases_rad``.
    who : str, optional
        Name to put in the refusals, for a method that delegates here.

    Returns
    -------
    ndarray
        Complex ``H(f)``, one entry per frequency.

    Raises
    ------
    ConfigurationError
        Mismatched lengths, a complex or negative amplitude, or a
        non-finite delay.

    Notes
    -----
    **Why a negative amplitude is refused rather than accepted.** This
    package had two implementations of this sum, and they disagreed on
    exactly that input: one took ``abs`` of the amplitude and the other did
    not, so a negative entry silently produced two different spectra — up
    to 10.7 dB apart per bin — with no error on either side. An amplitude
    here is a magnitude and the phase column carries the sign, so a
    negative value means the caller has mixed the two conventions. Saying
    so is the only answer that cannot be silently wrong.
    """
    freqs = np.asarray(frequencies, dtype=float).ravel()
    amp = np.asarray(amplitudes).ravel()
    delays = np.asarray(delays_s, dtype=float).ravel()
    if np.iscomplexobj(amp):
        raise ConfigurationError(
            f"{who}: amplitudes must be real magnitudes; got a complex "
            f"array. Pass |A| here and its argument in phases_rad.")
    amp = amp.astype(float)
    if amp.size != delays.size:
        raise ConfigurationError(
            f"{who}: amplitudes and delays_s must have the same length; "
            f"got {amp.size} and {delays.size}.")
    if np.any(amp < 0.0):
        n = int(np.count_nonzero(amp < 0.0))
        raise ConfigurationError(
            f"{who}: {n} of {amp.size} amplitudes are negative. An arrival "
            f"amplitude is a magnitude and its sign lives in the phase, so "
            f"pass abs(A) with pi added to phases_rad for each one. "
            f"Accepting it silently is what made two paths in this package "
            f"disagree by up to 10.7 dB per bin.")
    if not np.all(np.isfinite(delays)):
        raise ConfigurationError(
            f"{who}: an arrival has a non-finite delay, so its phase term "
            f"is undefined at every frequency.")
    imag = (np.zeros_like(delays) if delays_imag_s is None
            else np.asarray(delays_imag_s, dtype=float).ravel())
    phase = (np.zeros_like(delays) if phases_rad is None
             else np.asarray(phases_rad, dtype=float).ravel())
    for name, arr in (('delays_imag_s', imag), ('phases_rad', phase)):
        if arr.size != delays.size:
            raise ConfigurationError(
                f"{who}: {name} must have one entry per arrival "
                f"({delays.size}); got {arr.size}.")
    omega = 2.0 * np.pi * freqs
    gains = amp * np.exp(1j * (phase + float(phase_offset)))
    with np.errstate(over='ignore'):
        contrib = gains[:, None] * np.exp(np.outer(imag, omega)
                                          - 1j * np.outer(delays, omega))
    return contrib.sum(axis=0)


def broadband_propagation_loss(H, weights=None, *, axis: int = -1,
                               who: str = "broadband_propagation_loss"):
    """Propagation loss of a signal with bandwidth — Ainslie Eq. 11.46.

    .. math::
        \\mathrm{PL} = 10\\log_{10}
        \\frac{\\sum_f w(f)}{\\sum_f w(f)\\,|H(f)|^2}

    with ``w(f) = |S(f)|²`` the source's power spectrum. This is **the**
    transmission loss of a transient: the frequency average of the
    *coherent* ``|H|²``, weighted by the spectrum the signal actually puts
    on each bin (Ainslie, *Sonar Performance Modeling*, sect. 11.3.3,
    Eq. 11.46 and its footnote 13 for the coloured-spectrum form). The
    same quantity appears as Abraham's pulse loss
    ``∫|U|²df / ∫|H|²|U|²df`` (sect. 3.2.4.2) and as Ainslie's total path
    loss (sect. 3.3.2.1).

    Do not confuse it with **incoherent** TL (Eq. 11.47), the average of
    ``|H|²`` over paths rather than over frequency: that one is a cheap
    stand-in the KRAKEN manual endorses, and Ainslie marks it invalid
    within a few wavelengths of a boundary.

    Parameters
    ----------
    H : array_like
        Complex transfer function. Any shape; ``axis`` is the frequency
        axis, and the others are carried through, so a
        ``(depth, range, frequency)`` grid returns a map.
    weights : array_like, optional
        ``w(f) = |S(f)|²`` on the same grid, one per frequency. ``None``
        weights every bin equally — the flat-spectrum case, which is the
        band average of ``|H|²``.
    axis : int, default -1
        Frequency axis of ``H``.
    who : str, optional
        Name to put in the refusals, for a method that delegates here.

    Returns
    -------
    ndarray
        Loss in dB, ``H``'s shape without ``axis``. A cell with a NaN at
        any weighted frequency stays NaN: a no-data bin must not average
        away into a finite level.

    Raises
    ------
    ConfigurationError
        A ``weights`` of the wrong length, non-finite, or carrying no
        energy at all.

    Notes
    -----
    Accumulated one frequency at a time rather than as
    ``sum(w * |H|**2, axis)``: that spelling materialises a temporary the
    size of the whole broadband grid, on top of the grid itself. Measured
    on 120 x 400 cells over 401 frequencies (a 0.29 GiB grid), peak extra
    allocation is 0.001 GiB against 0.287 GiB for the one temporary, and
    the two agree to 1e-14 dB.
    """
    data = np.asarray(H)
    if data.ndim == 0:
        raise ConfigurationError(
            f"{who}: H must have a frequency axis; got a scalar.")
    try:
        requested = int(axis)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{who}: axis must be an integer; got {axis!r}.") from exc
    if not -data.ndim <= requested < data.ndim:
        raise ConfigurationError(
            f"{who}: axis={requested} is not an axis of an array with "
            f"shape {data.shape}.")
    axis = requested % data.ndim
    n_f = data.shape[axis]
    if weights is None:
        w = np.ones(n_f, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1:
            raise ConfigurationError(
                f"{who}: weights must be one value per frequency; got "
                f"shape {w.shape}. It is the source power spectrum ON "
                f"this frequency axis.")
        if w.size != n_f:
            raise ConfigurationError(
                f"{who}: weights has {w.size} samples but the frequency "
                f"axis has {n_f}.")
        if not np.all(np.isfinite(w)):
            raise ConfigurationError(f"{who}: weights must be finite.")
        if w.sum() <= 0.0:
            raise ConfigurationError(
                f"{who}: weights carry no energy, so the weighted average "
                f"is undefined.")
    moved = np.moveaxis(data, axis, 0)
    power = np.zeros(moved.shape[1:], dtype=float)
    for i, weight in enumerate(w):
        if weight:
            power += weight * np.abs(moved[i]) ** 2
        else:
            # A zero-weight bin contributes nothing but must still carry a
            # no-data cell forward: skipping it silently would let a NaN
            # column average to a finite level.
            power += np.where(np.isnan(moved[i]), np.nan, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        return 10.0 * np.log10(w.sum() / power)


def uniform_frequency_step(frequencies, who: str = "uniform_frequency_step"
                           ) -> float:
    """The step of a uniformly-spaced ascending frequency axis, or a refusal.

    Bin placement presumes such a grid: off one, frequencies land at the
    wrong bins and can collide (the later value overwrites). Every routine
    that turns an ``H(f)`` into a time record asks this first.
    """
    freqs = np.asarray(frequencies, dtype=float)
    df = float(freqs[1] - freqs[0])
    spacings = np.diff(freqs)
    if df <= 0 or not np.allclose(spacings, df, rtol=1e-6, atol=0.0):
        raise ConfigurationError(
            f"{who}: the frequency axis must be uniformly spaced and "
            f"ascending; spacing runs from {spacings.min():.6g} Hz to "
            f"{spacings.max():.6g} Hz against a leading Δf of {df:.6g} Hz. "
            f"Resample H(f) onto an equispaced grid before synthesising.",
            remediation="Run the model on an equispaced frequencies= array.",
        )
    return df


def _warn_if_response_wraps(h, offset, record, who: str) -> None:
    """Warn when the impulse response is still live where the window is not,
    at the far side of the circular record.

    The two cases a cut cannot tell apart are an arrival that is genuinely
    late and one that was later than ``1/df`` and folded back. This measures
    the only thing visible from inside: how much of the response sits in the
    half of the record furthest from the window. A decayed response leaves
    that part empty; a folding one does not, and then the energy the window
    removed is not the energy the pulse would resolve.

    It is deliberately NOT the whole complement of the window. Energy
    immediately outside the window is what a cut exists to remove — a
    genuine late path — so counting it warns on channels that cannot fold:
    two equal paths 100 ms apart in a 500 ms record read 50 %.

    **It is a heuristic with a blind spot, not a test.** What it detects is
    a response that has not decayed where a well-sized record would be
    empty. A fold that lands NEAR the origin — a path just past the
    record's end, which is the likeliest kind — arrives in the near half
    and is never counted at all. Size the grid from the arrivals; this only
    catches the loud, obvious case.
    """
    power = np.abs(h) ** 2
    total = power.sum(axis=-1)
    # The FAR HALF of the record from the window, not the whole complement.
    # Energy just outside the window is what a cut is FOR — a genuine late
    # path being removed — and counting it made this warn on channels that
    # cannot fold at all: two equal paths 100 ms apart in a 500 ms record
    # read 50% and were told to refine the grid.
    #
    # ``offset`` is the caller's own circular distance from the window
    # centre, passed in rather than re-derived. Recovering it here as
    # ``argmax(taper)`` gave a BOXCAR's left edge, not its centre — the mask
    # came out rotated by the window's half-width, so the same channel and
    # cut warned under 'boxcar' and stayed quiet under 'hann'.
    far = offset > record / 4.0
    live = np.where(total > 0.0,
                    power.sum(axis=-1, where=far)
                    / np.where(total > 0.0, total, 1.0), 0.0)
    worst = float(np.max(live)) if live.size else 0.0
    # 0.02, chosen from a 48-case sweep rather than inherited. The old 0.33
    # was calibrated against a DIFFERENT measure (the whole complement of
    # the window) and was never re-derived when this became the far half:
    # against the far half it missed 77.8 % of folds. At 0.02 the sweep
    # missed 11.1 % and false-alarmed on none of 21 clean channels. The
    # residual 11 % is the blind spot, not a tuning failure.
    if worst > 0.02:
        warnings.warn(
            f"{who}: {100.0 * worst:.0f}% of the response sits in the "
            f"half of the record furthest from the window. A decayed "
            f"response leaves that empty, so either a path is arriving "
            f"that late, or one later than the 1/df record has folded "
            f"onto it — from H(f) alone these are the same measurement. "
            f"If the arrivals are known to fit the record, ignore this; "
            f"otherwise size the grid from them "
            f"(Arrivals.synthesis_band).",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)


def gate_transfer_function(H, frequencies, duration: float, *,
                           origin='peak', window: str = 'boxcar',
                           axis: int = -1,
                           who: str = "gate_transfer_function"):
    """Keep only the part of a channel's impulse response within ``duration``
    of its centre, and return the ``H(f)`` of what is left.

    Transform the band to its own baseband response, gate it, transform
    back. It answers "what would this channel look like if only the paths
    arriving within ±``duration`` existed?" — the separability question a
    pulse of that length asks, since a pulse only interferes with the paths
    it can overlap.

    The record the gate lives in is ``1/df`` long and wraps, so the window
    is applied on **circular** distance: one centred near an end reaches
    round to the other, which is where a path near the record edge actually
    sits.

    Parameters
    ----------
    H : array_like
        Complex transfer function on a uniform ascending frequency grid.
    frequencies : array_like
        That grid (Hz). Its spacing sets the record length ``1/df``.
    duration : float
        Half-width of the gate (s). It reaches ``duration`` either side of
        the centre, so ``2*duration`` must fit inside the record.
    origin : {'peak'} or float
        ``'peak'`` centres the gate on each cell's own ``argmax|h|``; a
        number centres every cell on that fixed time (s) from the start of
        the record, which is what comparing cells requires.
    window : {'boxcar', 'hann'}
        ``'boxcar'`` is the separability criterion stated plainly;
        ``'hann'`` is the same cut, tapered.
    axis : int, default -1
        Frequency axis of ``H``.
    who : str, optional
        Name to put in the refusals and the wrap warning.

    Returns
    -------
    ndarray
        Gated ``H(f)``, same shape as ``H``.

    Warns
    -----
    UserWarning
        When the response is still live in the half of the record furthest
        from the window — either a genuinely late path or a fold, which
        ``H(f)`` alone cannot distinguish.
    """
    data = np.asarray(H)
    freqs = np.asarray(frequencies, dtype=float).ravel()
    if freqs.size < 2:
        raise ConfigurationError(
            f"{who}: needs at least 2 frequencies to define a record "
            f"length; got {freqs.size}.")
    df = uniform_frequency_step(freqs, who)
    record = 1.0 / df
    if window not in ('boxcar', 'hann'):
        raise ConfigurationError(
            f"{who}: window must be 'boxcar' (the separability criterion "
            f"stated plainly) or 'hann' (the same cut, tapered); got "
            f"{window!r}.")
    duration = float(duration)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ConfigurationError(
            f"{who}: duration must be a positive number of seconds; got "
            f"{duration!r}.")
    if 2.0 * duration >= record:
        raise ConfigurationError(
            f"{who}: the window reaches {duration:g} s either side of the "
            f"origin, which is not inside the {record:g} s record the grid "
            f"defines (1/df, df = {df:g} Hz), so it keeps everything and "
            f"the call would be a no-op. Refine the grid or shorten the "
            f"pulse.")
    try:
        requested = int(axis)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{who}: axis must be an integer; got {axis!r}.") from exc
    if not -data.ndim <= requested < data.ndim:
        raise ConfigurationError(
            f"{who}: axis={requested} is not an axis of an array with "
            f"shape {data.shape}.")
    axis = requested % data.ndim
    n_f = freqs.size
    if data.shape[axis] != n_f:
        raise ConfigurationError(
            f"{who}: H has {data.shape[axis]} samples along axis {axis} "
            f"but frequencies has {n_f}.")
    moved = np.moveaxis(data, axis, -1)
    flat = moved.reshape(-1, n_f)
    # The band's own baseband response: one period is 1/df and the step is
    # 1/(n_f*df), which is the resolution the band itself has.
    h = np.fft.ifft(flat, axis=-1)
    times = np.arange(n_f) * (record / n_f)
    if isinstance(origin, str) and origin == 'peak':
        centres = times[np.argmax(np.abs(h), axis=-1)]
    else:
        try:
            fixed = float(origin)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"{who}: origin must be 'peak' or a time in seconds from "
                f"the start of the record; got {origin!r}.") from exc
        if not np.isfinite(fixed) or not 0.0 <= fixed < record:
            raise ConfigurationError(
                f"{who}: origin {fixed:g} s is outside the "
                f"[0, {record:g}) s record this grid defines.")
        centres = np.full(flat.shape[0], fixed)
    # Circular distance: the record wraps, so a window centred near one end
    # reaches round to the other, which is where the response of a path near
    # the record edge actually sits.
    offset = np.abs(times[None, :] - centres[:, None])
    offset = np.minimum(offset, record - offset)
    if window == 'boxcar':
        taper = (offset <= duration).astype(float)
    else:
        taper = np.where(offset <= duration,
                         0.5 * (1.0 + np.cos(np.pi * offset / duration)),
                         0.0)
    _warn_if_response_wraps(h, offset, record, who)
    out = np.fft.fft(h * taper, axis=-1).reshape(moved.shape)
    return np.moveaxis(out, -1, axis)


# ──────────────────────────────────────────────────────────────────────
# Reception synthesis
#
# Placing each arrival as a phase-rotated, absorbed copy of the
# transmitted waveform, and reporting what the output window left out.
# ──────────────────────────────────────────────────────────────────────


#: Length of the silent trace an empty arrival record returns, when no
#: time_window says otherwise.
_EMPTY_TRACE_SECONDS = 0.1


def _echo_window_counts(starts, ends, powers, n_samples: int) -> dict:
    """Count the placed echoes a ``[0, n_samples)`` record omits or clips.

    ``starts``/``ends`` are each echo's first and one-past-last sample on
    the record, ``powers`` the received power each carries. An echo with no
    sample inside the record is omitted outright — a delay-and-sum drops an
    echo, it does not fold it the way an inverse FFT does — and one that
    straddles an edge is clipped there: at the start it loses its leading
    edge, the first samples of the waveform, so a chirp arrives as a
    different signal; at the end it loses its tail.
    """
    starts = np.asarray(starts, dtype=int)
    ends = np.asarray(ends, dtype=int)
    powers = np.asarray(powers, dtype=float)
    omitted = (ends <= 0) | (starts >= n_samples)
    return {
        'omitted': int(omitted.sum()),
        'clipped_start': int((~omitted & (starts < 0)).sum()),
        'clipped_end': int((~omitted & (ends > n_samples)).sum()),
        'omitted_power': float(powers[omitted].sum()),
        'total_power': float(powers.sum()),
        # First sample of the earliest echo, relative to the record start;
        # negative when it lands before the window. Merged by ``min``.
        'earliest_start': int(starts.min()),
    }


def _echo_window_notice(counts: dict, t_start: float, time_window: float,
                        who: str) -> Optional[str]:
    """The text for a window that does not hold every echo, or ``None``."""
    if not (counts.get('omitted') or counts.get('clipped_start')
            or counts.get('clipped_end')):
        return None
    parts = []
    if counts['omitted']:
        total = counts['total_power']
        share = counts['omitted_power'] / total if total > 0.0 else 0.0
        if share >= 1.0 - 1e-12:
            level = "all of the received energy"
        elif share > 0.0:
            level = f"{10.0 * np.log10(share):.0f} dB of the received energy"
        else:
            level = "no measurable energy"
        parts.append(f"{counts['omitted']} echo(es) fall entirely outside it "
                     f"and are omitted — a delay-and-sum drops an echo rather "
                     f"than folding it — carrying {level}")
    if counts['clipped_start']:
        parts.append(f"{counts['clipped_start']} echo(es) begin before it and "
                     f"lose their leading edge, the first samples of the "
                     f"waveform, so a chirp arrives as a different signal")
    if counts['clipped_end']:
        parts.append(f"{counts['clipped_end']} echo(es) run past its end and "
                     f"lose their tail")
    if 'earliest_s' in counts:
        parts.append(
            f"the earliest echo arrives at {counts['earliest_s']:g} s")
    return (f"{who}: the [{t_start:g}, {t_start + time_window:g}] s window "
            f"does not hold every echo: " + "; ".join(parts) + ". Widen "
            f"time_window= or move t_start= (on run(): output_duration= and "
            f"t_start=), or leave both unset to size the window from the "
            f"arrivals.")


def simulate_arrival_reception(
    source_timeseries: np.ndarray,
    amplitudes,
    delays_s,
    sample_rate: float,
    fc: float,
    *,
    delays_imag_s=None,
    phases_rad=None,
    time_window: Optional[float] = None,
    t_start: Optional[float] = None,
    phase_offset: float = 0.0,
    fractional: bool = True,
    report: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Received waveform from a sparse arrival list, with carrier and absorption.

    Places a phase-shifted, amplitude-scaled copy of the source waveform at
    each arrival time.  Uses the Hilbert transform (analytic signal) to apply
    arbitrary phase rotations from caustics and boundary reflections, as
    described in the Bellhop User Guide (Sec. 9.3).

    Parameters
    ----------
    amplitudes, delays_s : array_like
        Arrival magnitudes and real travel times (s).
    delays_imag_s : array_like, optional
        Imaginary travel times (s), carrying the volume absorption as
        ``exp(omega * Im tau)``. ``None`` is a lossless list.
    phases_rad : array_like, optional
        Per-arrival phase (radians) from caustics and boundary reflections.
        ``None`` is zeros.
    source_timeseries : ndarray
        Source waveform (1-D), used as-is.
    sample_rate : float
        Sample rate in Hz.
    fc : float
        Center frequency in Hz (for volume-attenuation scaling).
    time_window : float, optional
        Output time window in seconds.  If *None*, estimated from the
        latest arrival plus the source waveform duration plus margin.
    t_start : float, optional
        Start time for the output.  If *None*, set to just before the
        earliest arrival.
    phase_offset : float, optional
        Constant phase (radians) added to every arrival, applied on the
        analytic signal. ``0.0`` (default): a :class:`Bellhop` ARRIVALS
        result already carries the line-source ``exp(-i*pi/4)`` in its
        ``phases``.
    fractional : bool, optional
        Place each echo at its exact delay with a windowed-sinc kernel
        (default). ``False`` rounds every delay to the nearest sample, which
        is what this did before: an error up to half a sample, which is tens
        of degrees of carrier phase at any frequency a modem uses, summed
        coherently into the interference pattern. Continuous placement is
        what the formulation asks for — COA eq. 8.30 shifts the waveform by
        ``t - tau(s)``, not by a whole number of samples.
    report : dict, optional
        Collect, instead of warning, what the window omits or clips: the
        counts from :func:`_echo_window_counts` are ADDED into it, so one
        dict passed across every receiver cell of a run totals the grid and
        the run says it once. Left ``None``, this warns itself when an echo
        falls outside the window or straddles one of its edges — a
        delay-and-sum drops such an echo silently otherwise, and the record
        reads as complete.

    Returns
    -------
    time_vector : ndarray
        Time of each sample (s).
    rts : ndarray
        Received time series, shape ``(n_samples,)``.

    The axis comes FIRST, as it does from ``impulse_response``,
    ``simulate_reception`` and the two transform pairs. Both are real
    arrays of the same length, so the wrong order is silent — which is
    why the package has one order rather than two.
    (``Bellhop``'s ``delayandsum`` keeps its own historical
    ``(signal, time)`` and swaps on the way out.)

    References
    ----------
    Bellhop User Guide, Section 9.3
    Original MATLAB code: delayandsum.m by M. B. Porter, 8/96
    """
    amps = np.atleast_1d(np.asarray(amplitudes, dtype=float)).ravel()
    delays = np.atleast_1d(np.asarray(delays_s, dtype=float)).ravel()
    if amps.size != delays.size:
        raise ConfigurationError(
            f"simulate_arrival_reception: amplitudes and delays_s must have "
            f"the same length; got {amps.size} and {delays.size}.")
    delays_imag = (np.zeros_like(delays) if delays_imag_s is None
                   else np.atleast_1d(
                       np.asarray(delays_imag_s, dtype=float)).ravel())
    phases = (np.zeros_like(delays) if phases_rad is None
              else np.atleast_1d(np.asarray(phases_rad, dtype=float)).ravel())
    for name, arr in (('delays_imag_s', delays_imag), ('phases_rad', phases)):
        if arr.size != delays.size:
            raise ConfigurationError(
                f"simulate_arrival_reception: {name} must have one entry per "
                f"arrival ({delays.size}); got {arr.size}.")
    n_arr = delays.size
    if n_arr == 0:
        if time_window is not None:
            nrts = int(np.ceil(time_window * sample_rate))
        else:
            nrts = int(_EMPTY_TRACE_SECONDS * sample_rate)
        t0 = 0.0 if t_start is None else float(t_start)
        return t0 + np.arange(nrts) / sample_rate, np.zeros(nrts)

    sts = np.asarray(source_timeseries, dtype=float)
    nsts = len(sts)

    # Compute analytic signal via Hilbert transform
    sts_analytic = _sig.hilbert(sts)

    deltat = 1.0 / sample_rate
    src_duration = nsts * deltat

    # Determine time window
    min_delay = float(np.min(delays))
    max_delay = float(np.max(delays))

    # Every arrival places a whole copy of the source waveform starting at its
    # own delay, so the window must reach ``max_delay + src_duration`` for the
    # last one to fit; ``2 *`` leaves a further source duration of tail. The
    # lead-in keeps the earliest arrival's leading edge inside the window, and
    # the max(0, ...) stops the clock from starting before source emission.
    if t_start is None:
        t_start = max(0.0, min_delay - 0.1 * src_duration)

    if time_window is None:
        time_window = (max_delay - t_start) + 2.0 * src_duration

    nrts = int(np.ceil(time_window * sample_rate))
    rts = np.zeros(nrts)

    omega_c = 2.0 * np.pi * fc
    # Where each echo's waveform copy lands on the record — first sample and
    # one past its last — and the power it carries, for the window report.
    starts, ends, powers = [], [], []
    for ia in range(n_arr):
        phase_rad = phases[ia] + phase_offset
        phase_factor = np.exp(1j * phase_rad)

        # ``delays_imag`` is Im(tau) in seconds; volume-attenuation factor
        # is exp(omega * Im(tau)) per delayandsum.m:134.
        atten = np.exp(omega_c * delays_imag[ia])

        scaled_amp = amps[ia] * atten

        delay_samples = (delays[ia] - t_start) / deltat

        # Add this arrival's shifted, scaled copy of the source signal as a
        # single clipped slice-add (vectorised over the source samples).
        contrib = scaled_amp * np.real(sts_analytic * phase_factor)
        if fractional:
            # Resolve the sub-sample part of the delay with the same
            # windowed-sinc kernel channel.impulse_response uses. Convolving
            # by taps centred on offset 0 delays by (half_len - 1) + frac
            # samples, so the placement index backs off by that integer part
            # and the echo lands at delay_samples exactly.
            i_start = int(np.floor(delay_samples))
            nominal_start = i_start
            taps = fractional_delay_taps(delay_samples - i_start)
            placed = np.convolve(contrib, taps)
            i_start -= taps.size // 2 - 1
        else:
            i_start = int(np.round(delay_samples))
            nominal_start = i_start
            placed = contrib
        lo = max(0, i_start)
        hi = min(nrts, i_start + placed.size)
        if lo < hi:
            rts[lo:hi] += placed[lo - i_start:hi - i_start]
        # The report reads the WAVEFORM's extent, not the kernel's: the
        # sinc taps ring a few samples ahead of the echo, and losing that
        # pre-ring at the window's start is not a cut leading edge.
        starts.append(nominal_start)
        ends.append(nominal_start + nsts)
        powers.append(float(scaled_amp) ** 2)

    counts = _echo_window_counts(starts, ends, powers, nrts)
    earliest = counts.pop('earliest_start')
    counts['earliest_s'] = t_start + earliest * deltat
    if report is not None:
        for key, value in counts.items():
            if key == 'earliest_s':
                report[key] = min(report.get(key, np.inf), value)
            else:
                report[key] = report.get(key, 0) + value
    else:
        notice = _echo_window_notice(counts, t_start, time_window,
                                     who="delayandsum")
        if notice is not None:
            warnings.warn(notice, UserWarning,
                          skip_file_prefixes=USER_FRAME_SKIP)

    time_vector = t_start + np.arange(nrts) * deltat
    return time_vector, rts
