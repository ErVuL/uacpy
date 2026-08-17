"""Time-frequency and cepstral primitives.

Analytic-signal envelope/instantaneous-frequency, the Wigner-Ville
distribution, and (real/complex) cepstra — useful for transient analysis,
multipath delay estimation and sub-bottom layer picking.

References
----------
Cohen, L. *Time-Frequency Analysis* (Wigner-Ville).
Oppenheim & Schafer. *Discrete-Time Signal Processing* (cepstrum).
"""

from __future__ import annotations

import math
from collections import namedtuple
from functools import lru_cache

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import hilbert
from scipy.special import gamma
import scipy.signal as _sig

from uacpy.core.exceptions import ConfigurationError
from uacpy.acoustic_signal._signal_validate import require_finite_signal


WignerVilleResult = namedtuple("WignerVilleResult",
                               "frequencies times distribution")
CWTResult = namedtuple("CWTResult", "frequencies coefficients")
SpectrogramResult = namedtuple("SpectrogramResult", "frequencies times power")


def analytic_signal(data):
    """Analytic signal ``data + j*Hilbert(data)`` of a real signal."""
    xa = np.asarray(data)
    if np.iscomplexobj(xa):
        raise ConfigurationError(
            "analytic_signal: data must be real; the analytic/Hilbert "
            "representation is only defined for a real signal (got complex input)."
        )
    xr = xa.astype(float)
    if xr.ndim != 1:
        raise ConfigurationError("analytic_signal: data must be 1-D")
    require_finite_signal(xr, "analytic_signal")
    return hilbert(xr)


def envelope(data):
    """Instantaneous amplitude envelope ``|analytic_signal(data)|``."""
    return np.abs(analytic_signal(data))


def instantaneous_frequency(data, sample_rate: float):
    """Instantaneous frequency (Hz) from the analytic-signal phase derivative.

    Returns an array of length ``len(data)`` (centred differences of the
    unwrapped phase via :func:`numpy.gradient`, time-aligned with ``data``).
    """
    phase = np.unwrap(np.angle(analytic_signal(data)))
    return np.gradient(phase) / (2.0 * np.pi) * float(sample_rate)


def _smoothing_window(spec, name):
    """Centered, odd-length smoothing window from a ``None`` / int / array spec.

    ``None`` -> no smoothing. An int ``L`` -> a symmetric Hann window of odd length (``L-1`` for even ``L``). A 1-D
    array is used verbatim. Returns ``(w, half)`` with ``w`` of length
    ``2*half+1`` centered at index ``half`` (so ``w[half+k]`` weights offset
    ``k``); ``(None, 0)`` for no smoothing.
    """
    if spec is None:
        return None, 0
    if np.isscalar(spec):
        L = int(spec)
        if L < 1:
            raise ConfigurationError(f"wigner_ville: {name} length must be >= 1")
        # Generate an odd length directly: trimming an even symmetric
        # window (hann(6)[:-1]) puts the peak off-centre, which breaks the
        # acc(-tau) = conj(acc(tau)) symmetry the transform's .real relies
        # on (measured up to 59 % error at L = 4).
        w = _sig.get_window("hann", L - 1 if L % 2 == 0 else L,
                            fftbins=False).astype(float)
    else:
        w = np.asarray(spec, dtype=float)
        if w.ndim != 1 or w.size < 1:
            raise ConfigurationError(f"wigner_ville: {name} must be a 1-D window")
        if w.size % 2 == 0:
            w = w[:-1]
    return w, w.size // 2


def wigner_ville(data, sample_rate: float, *, analytic: bool = True,
                 freq_window=None, time_window=None, nfft=None):
    """Discrete (smoothed-pseudo-) Wigner-Ville distribution of a 1-D signal.

    Returns a :class:`WignerVilleResult` ``(frequencies, times, distribution)``
    with the distribution real, shape ``(NF, n)``; ``f`` spans
    ``[0, fs/2)``. The kernel ``z(t+tau)z*(t-tau)`` doubles the apparent
    frequency, so the physical frequency axis is ``k*fs/(2*NF)``.

    A quadratic energy distribution — there is no routine inverse (like a
    spectrogram, it maps a signal to a 2-D density, not reversibly).

    Parameters
    ----------
    data : 1-D array
        Real or complex signal.
    sample_rate : float
        Sample rate (Hz).
    analytic : bool
        Use the analytic signal for real input (default), suppressing
        cross-terms with the negative spectrum. ``False`` runs the raw signal.
        Ignored when ``data`` is already complex.
    freq_window : None, int, or 1-D array
        Lag-domain smoothing window ``h(tau)`` — the *pseudo*-WVD. Smooths
        along frequency and limits the lag extent (shorter window -> more
        cross-term suppression, coarser frequency resolution). ``None`` is the
        full-lag WVD. An int gives a symmetric Hann window of odd length (L-1 for even L).
    time_window : None, int, or 1-D array
        Time-domain smoothing window ``g`` — the *smoothed*-pseudo-WVD. Averages
        the instantaneous autocorrelation over neighbouring times (more
        cross-term suppression, coarser time resolution). ``None`` disables it.
    nfft : int, optional
        Zero-pad the lag FFT to ``nfft >= n`` bins (finer frequency spacing).
        ``None`` uses ``n``.

    Returns
    -------
    WignerVilleResult
        ``(frequencies, times, distribution)``: frequency axis (Hz), time axis
        (s), and the distribution ``(NF, n)`` — the same axis order as
        :class:`SpectrogramResult`.
    """
    xc = np.asarray(data)
    if xc.ndim != 1:
        raise ConfigurationError("wigner_ville: data must be 1-D")
    require_finite_signal(xc, "wigner_ville")
    if np.iscomplexobj(xc):
        z = xc
    elif analytic:
        z = analytic_signal(xc)
    else:
        z = xc.astype(complex)
    n = z.size
    fs = float(sample_rate)
    NF = n if nfft is None else int(nfft)
    if NF < n:
        raise ConfigurationError(
            f"wigner_ville: nfft ({NF}) must be >= n ({n}) (zero-pad only)")
    hv, Lh = _smoothing_window(freq_window, "freq_window")
    gv, Lg = _smoothing_window(time_window, "time_window")
    lag_cap = n - 1 if hv is None else Lh
    W = np.zeros((NF, n))
    for ti in range(n):
        taumax = min(ti, n - 1 - ti, lag_cap)
        taus = np.arange(-taumax, taumax + 1)
        if gv is None:
            acc = z[ti + taus] * np.conj(z[ti - taus])
        else:
            acc = np.empty(taus.size, dtype=complex)
            for a, tau in enumerate(taus):
                mmax = min(Lg, ti + tau, n - 1 - ti - tau,
                           ti - tau, n - 1 - ti + tau)
                ms = np.arange(-mmax, mmax + 1)
                gw = gv[Lg + ms]
                acc[a] = (np.sum(gw * z[ti + tau + ms]
                                 * np.conj(z[ti - tau + ms])) / np.sum(gw))
        if hv is not None:
            acc = acc * hv[Lh + taus]
        kernel = np.zeros(NF, dtype=complex)
        # Lag axis in FFT order: lag 0 at index 0, negative lags wrapped to the
        # top of the buffer, zeros in the unfilled middle (the zero-padding).
        kernel[(taus + NF) % NF] = acc
        # acc(-tau) = conj(acc(tau)) — a symmetric smoothing window preserves
        # that — so the transform is real and `.real` drops only rounding
        # error, not signal. An asymmetric `freq_window` array breaks the
        # symmetry and the discarded imaginary part is then meaningful.
        W[:, ti] = np.real(np.fft.fft(kernel))
    f = np.arange(NF) * fs / (2.0 * NF)
    t = np.arange(n) / fs
    return WignerVilleResult(f, t, W)


def _wavelet_fourier(wavelet, s, omega, w0, order):
    """Fourier-domain daughter wavelet ``psi_hat(s*omega)`` and the Fourier
    frequency<->scale factor ``f = factor*fs/s`` (Torrence & Compo 1998, Table 1).
    ``omega`` is in rad/sample."""
    so = s * omega
    pos = omega > 0.0
    if wavelet == "morlet":
        psi = (np.pi ** -0.25) * np.exp(-0.5 * (so - w0) ** 2) * pos
        factor = (w0 + np.sqrt(2.0 + w0 ** 2)) / (4.0 * np.pi)
    elif wavelet == "paul":
        m = order
        norm = 2.0 ** m / np.sqrt(m * math.factorial(2 * m - 1))
        psi = np.zeros(omega.shape, dtype=complex)
        psi[pos] = norm * (so[pos] ** m) * np.exp(-so[pos])
        factor = (2.0 * m + 1.0) / (4.0 * np.pi)
    elif wavelet == "dog":
        m = order
        norm = -(1j ** m) / np.sqrt(gamma(m + 0.5))
        psi = norm * (so ** m) * np.exp(-0.5 * so ** 2)  # real wavelet: all omega
        factor = np.sqrt(m + 0.5) / (2.0 * np.pi)
    else:
        raise ConfigurationError(
            f"cwt: unknown wavelet {wavelet!r}; choose 'morlet', 'paul', or 'dog'"
        )
    return psi, factor


@lru_cache(maxsize=None)
def _reconstruction_constants(wavelet, w0, order):
    """``(C_delta, psi0(0))`` for the Torrence & Compo (1998) eq.-11 inverse.

    ``psi0(0)``, the wavelet at zero lag, is ``(2*pi)**-0.5 * int psihat0(w) dw``.
    ``C_delta`` is the delta-function calibration of T&C eq. 13-14 taken in the
    continuous-scale limit: for a unit impulse, ``Re{W(s)}/sqrt(s)`` collapses to
    ``psi0(0)*R(pi*s)/s``, where ``R(a)`` is the fraction of the wavelet's
    spectral mass inside the discrete band ``|w| <= pi`` rad/sample. The
    calibration is therefore a one-dimensional quadrature over scale, and it is
    a property of the wavelet alone — not of the scale set the caller analysed
    with, so a band-limited scale set still reconstructs only its own band.

    Against T&C's Table 2 (whose caption calls those factors "empirically
    derived"): ``psi0(0)`` agrees to every tabulated digit for Morlet ``w0=6``,
    Paul ``m=4`` and DOG ``m=2``/``m=6``; ``C_delta`` agrees to 0.3 % for
    Morlet and Paul but differs by ~2 % for the DOG pair (0.776/1.132/3.541/
    1.966 tabulated against 0.778/1.133/3.616/1.929 here). Extends the table to
    any admissible ``w0`` / ``order``.
    """
    u_max = 60.0 + w0 + 6.0 * order
    u = np.linspace(-u_max, u_max, 100001)
    psi_hat, factor = _wavelet_fourier(wavelet, 1.0, u, w0, order)
    mass = cumulative_trapezoid(np.real(psi_hat), u, initial=0.0)
    total = mass[-1] - mass[0]
    psi0_zero = total / np.sqrt(2.0 * np.pi)
    if abs(psi0_zero) < 1e-9:
        raise ConfigurationError(
            f"inverse_cwt: the {wavelet!r} wavelet of order {order} is an odd "
            "function, so psi0(0) = 0 and the Torrence & Compo eq.-11 "
            "reconstruction is undefined for it. Use an even order.")
    dj = 0.02
    s_min = 2.0 * factor / 256.0            # well below the Nyquist-cut scale
    n_scale = int(np.log2(2.0e6 / s_min) / dj) + 1
    scales = s_min * 2.0 ** (np.arange(n_scale) * dj)
    a = np.minimum(np.pi * scales, u_max)
    in_band = (np.interp(a, u, mass) - np.interp(-a, u, mass)) / total
    return float(np.sum(in_band / scales) * dj), float(psi0_zero)


def cwt(data, sample_rate, frequencies=None, wavelet="morlet", *, w0=6.0,
        order=None, n_freqs=64):
    """Continuous wavelet transform with a selectable wavelet (FFT-based).

    At each scale the signal is filtered by the chosen analysing wavelet
    (Mallat, *A Wavelet Tour of Signal Processing*, Ch. 4; scale<->frequency
    factors from Torrence & Compo 1998). Linear in time, logarithmic in
    frequency — well suited to dispersive / transient ocean-acoustic arrivals.

    Parameters
    ----------
    data : 1-D array
        Real signal.
    sample_rate : float
        Sample rate (Hz).
    frequencies : array, optional
        Frequencies (Hz) to analyse. Default: ``n_freqs`` log-spaced points
        from ``4*fs/N`` to ``fs/2``.
    wavelet : {'morlet', 'paul', 'dog'}
        Analysing wavelet. ``'morlet'`` (complex, best frequency resolution),
        ``'paul'`` (complex, best time resolution), ``'dog'`` (real Derivative
        Of Gaussian; ``order=2`` is the Mexican-hat / Ricker wavelet).
    w0 : float
        Morlet central (non-dimensional) frequency; ``>= 5`` keeps it
        admissible. Default 6.
    order : int, optional
        Wavelet order ``m``. Default 4 for ``'paul'``, 2 for ``'dog'``; ignored
        for ``'morlet'``.
    n_freqs : int
        Number of log-spaced frequencies when ``frequencies`` is None.

    Returns
    -------
    CWTResult
        ``(frequencies, coefficients)``: the analysis frequencies (Hz) and the
        complex CWT coefficients, shape ``(n_freqs, len(data))``;
        ``abs(coefficients)`` is the scalogram.
    """
    xr = np.asarray(data, dtype=float)
    if xr.ndim != 1:
        raise ConfigurationError("cwt: data must be 1-D")
    require_finite_signal(xr, "cwt")
    if order is None:
        order = 4 if wavelet == "paul" else 2
    n = xr.size
    fs = float(sample_rate)
    if frequencies is None:
        f_lo = 4.0 * fs / n  # lowest default frequency = 4 cycles per record
        if f_lo >= fs / 2.0:
            raise ConfigurationError(
                f"cwt: signal too short (n={n}) for the default frequency "
                f"range — the lowest analysis frequency {f_lo:.1f} Hz already "
                f"exceeds Nyquist {fs / 2.0:.1f} Hz. Pass an explicit "
                "`frequencies=` below Nyquist, or use a longer signal.")
        frequencies = np.logspace(np.log10(f_lo), np.log10(fs / 2.0),
                                  int(n_freqs))
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
    if np.any(frequencies <= 0):
        raise ConfigurationError("cwt: frequencies must be > 0")
    omega = 2.0 * np.pi * np.fft.fftfreq(n)  # rad/sample
    Xf = np.fft.fft(xr)
    # One scale gives the factor; scales follow from f = factor*fs/s.
    _, factor = _wavelet_fourier(wavelet, 1.0, omega, w0, order)
    scales = factor * fs / frequencies  # scale in samples
    W = np.empty((frequencies.size, n), dtype=complex)
    for i, s in enumerate(scales):
        psi_hat, _ = _wavelet_fourier(wavelet, s, omega, w0, order)
        # Torrence & Compo eq. 6 normalisation sqrt(2*pi*s/dt), with dt = 1
        # because `omega` is rad/sample: equal energy at every scale.
        psi_hat = np.sqrt(2.0 * np.pi * s) * psi_hat
        W[i] = np.fft.ifft(Xf * np.conj(psi_hat))
    return CWTResult(frequencies, W)


def inverse_cwt(W, frequencies, sample_rate, wavelet="morlet", *, w0=6.0,
                order=None):
    """Inverse CWT (Torrence & Compo 1998, eq. 11).

    ``x_n = dj/(C_delta*psi0(0)) * sum_j Re(W_n(s_j))/sqrt(s_j)`` with ``dj``
    the log2 scale spacing. The reconstruction constants ``C_delta`` and
    ``psi0(0)`` are derived for the wavelet order actually in use
    (:func:`_reconstruction_constants`), so non-default ``w0`` / ``order``
    reconstruct at the right amplitude. Pass the same ``frequencies`` /
    ``wavelet`` / ``w0`` / ``order`` used in :func:`cwt`; amplitude is
    recovered to the accuracy of the scale coverage (a band-limited scale set
    reconstructs only the band it spans).

    Parameters
    ----------
    W : ndarray
        CWT coefficients ``(n_freqs, n_time)`` from :func:`cwt`.
    frequencies : array
        The analysis frequencies returned by :func:`cwt`.
    sample_rate : float
        Sample rate (Hz).

    Returns
    -------
    ndarray
        Reconstructed 1-D signal.
    """
    Wc = np.asarray(W)
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
    if Wc.ndim != 2 or Wc.shape[0] != frequencies.size:
        raise ConfigurationError(
            "inverse_cwt: W must be (n_freqs, n_time) matching frequencies")
    if order is None:
        order = 4 if wavelet == "paul" else 2
    fs = float(sample_rate)
    _, factor = _wavelet_fourier(wavelet, 1.0, np.array([1.0]), w0, order)
    scales = factor * fs / frequencies
    dj = float(np.mean(np.abs(np.diff(np.log2(scales))))) if scales.size > 1 else 1.0
    c_delta, psi0_zero = _reconstruction_constants(wavelet, float(w0), int(order))
    return (np.sum(np.real(Wc) / np.sqrt(scales)[:, None], axis=0)
            * dj / (c_delta * psi0_zero))


def _apply_lifter(c, lifter):
    """Quefrency-domain liftering of a real cepstrum ``c`` (length ``NF``).

    ``lifter`` is an int cutoff or a 1-D weight array. A positive int ``L``
    short-passes (keeps ``|quefrency| <= L`` -> spectral envelope); a negative
    int long-passes (zeros ``|quefrency| <= |L|`` -> excitation / echo). An
    array multiplies the cepstrum element-wise (symmetric weighting is on you).
    """
    nf = c.size
    if np.isscalar(lifter):
        L = int(lifter)
        w = np.zeros(nf)
        keep = abs(L)
        w[:keep + 1] = 1.0
        if keep:
            # Negative quefrencies live at the top of the buffer. max(1, ...)
            # is what makes an over-long cutoff (keep >= nf) keep everything:
            # a bare nf - keep would go negative and w[-2:] would set two
            # elements instead of the whole tail.
            w[max(1, nf - keep):] = 1.0
        if L < 0:
            w = 1.0 - w
        return c * w
    w = np.asarray(lifter, dtype=float)
    if w.shape != c.shape:
        raise ConfigurationError(
            f"cepstrum: lifter array {w.shape} must match cepstrum {c.shape}")
    return c * w


def cepstrum(data, *, window=None, nfft=None, lifter=None):
    """Real cepstrum ``irfft(log|rfft(data)|)``.

    Not invertible: discards phase. Use :func:`complex_cepstrum` /
    :func:`inverse_complex_cepstrum` for a reversible homomorphic transform.

    Parameters
    ----------
    data : 1-D array
        Real signal.
    window : None, str, or tuple
        :func:`scipy.signal.get_window` spec applied before the FFT to curb
        spectral leakage. ``None`` is rectangular.
    nfft : int, optional
        Zero-pad the FFT to ``nfft >= len(data)`` bins (finer quefrency
        spacing). ``None`` uses ``len(data)``.
    lifter : None, int, or 1-D array
        Quefrency liftering — see :func:`_apply_lifter`. ``None`` returns the
        raw cepstrum; a positive int keeps low quefrencies (spectral envelope),
        a negative int keeps high quefrencies (pitch / echo structure).
    """
    xa = np.asarray(data)
    if np.iscomplexobj(xa):
        raise ConfigurationError(
            "cepstrum: data must be real (got complex input); the real cepstrum "
            "irfft(log|rfft(x)|) is defined for a real signal. For a complex "
            "spectrum use complex_cepstrum.")
    xr = xa.astype(float)
    if xr.ndim != 1:
        raise ConfigurationError("cepstrum: data must be 1-D")
    require_finite_signal(xr, "cepstrum")
    n = xr.size
    NF = n if nfft is None else int(nfft)
    if NF < n:
        raise ConfigurationError(
            f"cepstrum: nfft ({NF}) must be >= len(x) ({n}) (zero-pad only)")
    if window is not None:
        xr = xr * _sig.get_window(window, n, fftbins=True).astype(float)
    spectrum = np.abs(np.fft.rfft(xr, n=NF))
    spectrum = np.maximum(spectrum, np.finfo(float).tiny)
    c = np.fft.irfft(np.log(spectrum), n=NF)
    if lifter is not None:
        c = _apply_lifter(c, lifter)
    return c


ComplexCepstrum = namedtuple("ComplexCepstrum", "cepstrum delay")


def complex_cepstrum(data):
    """Complex cepstrum with the linear-phase (rotation) term removed.

    Returns ``ComplexCepstrum(cepstrum, delay)``. ``delay`` is the integer
    number of samples of linear phase taken out;
    :func:`inverse_complex_cepstrum` needs it to reconstruct ``data``.

    Without the removal the unwrapped phase carries a ramp whose inverse
    transform is a ``1/q`` tail that swamps the echo structure the cepstrum
    exists to show, and makes the result depend on the signal's absolute
    arrival time rather than on its echo delays.

    The cepstrum stays **complex**: unwrapping breaks the Hermitian symmetry
    of ``log(fft(x))``, and the imaginary part is what makes the homomorphic
    transform reversible.
    """
    xa = np.asarray(data)
    if np.iscomplexobj(xa):
        raise ConfigurationError(
            "complex_cepstrum: data must be real (got complex input); the "
            "homomorphic cepstrum is defined for a real signal.")
    xr = xa.astype(float)
    if xr.ndim != 1:
        raise ConfigurationError("complex_cepstrum: data must be 1-D")
    require_finite_signal(xr, "complex_cepstrum")
    spectrum = np.fft.fft(xr)
    n = xr.size
    mag = np.maximum(np.abs(spectrum), np.finfo(float).tiny)
    phase = np.unwrap(np.angle(spectrum))
    # Remove the linear-phase term: round the end-to-end ramp to a whole
    # number of samples and subtract it, so the residual phase carries only
    # the echo structure.
    delay = int(np.round(phase[n // 2] * n / (2.0 * np.pi * (n // 2)))) if n > 1 else 0
    phase = phase - 2.0 * np.pi * delay * np.arange(n) / n
    log_spectrum = np.log(mag) + 1j * phase
    return ComplexCepstrum(np.fft.ifft(log_spectrum), delay)


def inverse_complex_cepstrum(c):
    """Invert :func:`complex_cepstrum`: ``x = real(ifft(exp(fft(c))))``.

    Takes the complex cepstrum :func:`complex_cepstrum` returns (the
    imaginary part is significant — see there) and reconstructs the real
    signal ``x``."""
    delay = 0
    if isinstance(c, ComplexCepstrum):
        c, delay = c.cepstrum, int(c.delay)
    cr = np.asarray(c, dtype=complex)
    if cr.ndim != 1:
        raise ConfigurationError("inverse_complex_cepstrum: c must be 1-D")
    n = cr.size
    log_spectrum = np.fft.fft(cr)
    # Restore the linear-phase term complex_cepstrum took out.
    log_spectrum = log_spectrum + 1j * 2.0 * np.pi * delay * np.arange(n) / n
    return np.real(np.fft.ifft(np.exp(log_spectrum)))


def spectrogram(data, sample_rate, *, window="hann", nperseg=8192,
                noverlap=None, nfft=None, scaling="density", mode="psd"):
    """Short-time spectrogram. Returns a :class:`SpectrogramResult`
    ``(frequencies, times, power)`` (Pa²/Hz).

    ``noverlap=None`` (default) lets scipy derive the overlap (``nperseg // 8``)
    and clamp ``nperseg`` to the input length, so short signals don't raise; pass
    an int to override. ``nfft`` (zero-pad length) mirrors
    :func:`uacpy.acoustic_signal.psd`. For logarithmic / constant-Q frequency
    resolution, see :func:`uacpy.acoustic_signal.constant_q_spectrogram`."""
    data = require_finite_signal(data, "spectrogram")
    f, t, Sxx = _sig.spectrogram(data, sample_rate, window=window,
                                 nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                                 scaling=scaling, mode=mode)
    return SpectrogramResult(f, t, Sxx)
