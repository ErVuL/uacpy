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

import numpy as np
from scipy.signal import hilbert
from scipy.special import gamma
import scipy.signal as _sig
import matplotlib.pyplot as plt

from uacpy.core.exceptions import ConfigurationError
from uacpy.core.constants import (REFERENCE_PRESSURE_AIR,
                                  REFERENCE_PRESSURE_WATER)
from uacpy.core.acoustics import power_to_db


def analytic_signal(x):
    """Analytic signal ``x + j*Hilbert(x)`` of a real signal."""
    xa = np.asarray(x)
    if np.iscomplexobj(xa):
        raise ConfigurationError(
            "analytic_signal: x must be real; the analytic/Hilbert representation "
            "is only defined for a real signal (got complex input)."
        )
    xr = xa.astype(float)
    if xr.ndim != 1:
        raise ConfigurationError("analytic_signal: x must be 1-D")
    return hilbert(xr)


def envelope(x):
    """Instantaneous amplitude envelope ``|analytic_signal(x)|``."""
    return np.abs(analytic_signal(x))


def instantaneous_frequency(x, sample_rate: float):
    """Instantaneous frequency (Hz) from the analytic-signal phase derivative.

    Returns an array of length ``len(x) - 1`` (centred differences of the
    unwrapped phase).
    """
    phase = np.unwrap(np.angle(analytic_signal(x)))
    return np.diff(phase) / (2.0 * np.pi) * float(sample_rate)


def _smoothing_window(spec, name):
    """Centered, odd-length smoothing window from a ``None`` / int / array spec.

    ``None`` -> no smoothing. An int ``L`` -> a length-``L`` Hann window. A 1-D
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
        w = _sig.get_window("hann", L, fftbins=False).astype(float)
    else:
        w = np.asarray(spec, dtype=float)
        if w.ndim != 1 or w.size < 1:
            raise ConfigurationError(f"wigner_ville: {name} must be a 1-D window")
    if w.size % 2 == 0:
        w = w[:-1]
    return w, w.size // 2


def wigner_ville(x, sample_rate: float, *, analytic: bool = True,
                 freq_window=None, time_window=None, nfft=None):
    """Discrete (smoothed-pseudo-) Wigner-Ville distribution of a 1-D signal.

    Returns ``(t, f, W)`` with ``W`` real, shape ``(NF, n)``; ``f`` spans
    ``[0, fs/2)``. The kernel ``z(t+tau)z*(t-tau)`` doubles the apparent
    frequency, so the physical frequency axis is ``k*fs/(2*NF)``.

    A quadratic energy distribution — there is no routine inverse (like a
    spectrogram, it maps a signal to a 2-D density, not reversibly).

    Parameters
    ----------
    x : 1-D array
        Real or complex signal.
    sample_rate : float
        Sample rate (Hz).
    analytic : bool
        Use the analytic signal for real input (default), suppressing
        cross-terms with the negative spectrum. ``False`` runs the raw signal.
        Ignored when ``x`` is already complex.
    freq_window : None, int, or 1-D array
        Lag-domain smoothing window ``h(tau)`` — the *pseudo*-WVD. Smooths
        along frequency and limits the lag extent (shorter window -> more
        cross-term suppression, coarser frequency resolution). ``None`` is the
        full-lag WVD. An int gives a Hann window of that length.
    time_window : None, int, or 1-D array
        Time-domain smoothing window ``g`` — the *smoothed*-pseudo-WVD. Averages
        the instantaneous autocorrelation over neighbouring times (more
        cross-term suppression, coarser time resolution). ``None`` disables it.
    nfft : int, optional
        Zero-pad the lag FFT to ``nfft >= n`` bins (finer frequency spacing).
        ``None`` uses ``n``.

    Returns
    -------
    t, f, W : ndarray
        Time axis (s), frequency axis (Hz), and the distribution ``(NF, n)``.
    """
    xc = np.asarray(x)
    if xc.ndim != 1:
        raise ConfigurationError("wigner_ville: x must be 1-D")
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
        kernel[(taus + NF) % NF] = acc
        W[:, ti] = np.real(np.fft.fft(kernel))
    f = np.arange(NF) * fs / (2.0 * NF)
    t = np.arange(n) / fs
    return t, f, W


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


def cwt(x, sample_rate, freqs=None, wavelet="morlet", *, w0=6.0, order=None,
        n_freqs=64):
    """Continuous wavelet transform with a selectable wavelet (FFT-based).

    At each scale the signal is filtered by the chosen analysing wavelet
    (Mallat, *A Wavelet Tour of Signal Processing*, Ch. 4; scale<->frequency
    factors from Torrence & Compo 1998). Linear in time, logarithmic in
    frequency — well suited to dispersive / transient ocean-acoustic arrivals.

    Parameters
    ----------
    x : 1-D array
        Real signal.
    sample_rate : float
        Sample rate (Hz).
    freqs : array, optional
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
        Number of log-spaced frequencies when ``freqs`` is None.

    Returns
    -------
    freqs : ndarray
        Analysis frequencies (Hz).
    W : ndarray
        Complex CWT coefficients, shape ``(n_freqs, len(x))``. ``abs(W)`` is the
        scalogram.
    """
    xr = np.asarray(x, dtype=float)
    if xr.ndim != 1:
        raise ConfigurationError("cwt: x must be 1-D")
    if order is None:
        order = 4 if wavelet == "paul" else 2
    n = xr.size
    fs = float(sample_rate)
    if freqs is None:
        freqs = np.logspace(np.log10(4.0 * fs / n), np.log10(fs / 2.0),
                            int(n_freqs))
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    if np.any(freqs <= 0):
        raise ConfigurationError("cwt: freqs must be > 0")
    omega = 2.0 * np.pi * np.fft.fftfreq(n)  # rad/sample
    Xf = np.fft.fft(xr)
    # One scale gives the factor; scales follow from f = factor*fs/s.
    _, factor = _wavelet_fourier(wavelet, 1.0, omega, w0, order)
    scales = factor * fs / freqs  # scale in samples
    W = np.empty((freqs.size, n), dtype=complex)
    for i, s in enumerate(scales):
        psi_hat, _ = _wavelet_fourier(wavelet, s, omega, w0, order)
        psi_hat = np.sqrt(2.0 * np.pi * s) * psi_hat  # unit-energy per scale
        W[i] = np.fft.ifft(Xf * np.conj(psi_hat))
    return freqs, W


def inverse_cwt(W, freqs, sample_rate, wavelet="morlet", *, w0=6.0, order=None):
    """Approximate inverse CWT (Torrence & Compo 1998 reconstruction).

    Recovers the signal by summing ``Re(W)`` over scale. The waveform *shape* is
    recovered faithfully; absolute amplitude is approximate (it carries the
    wavelet's reconstruction constant). Pass the same ``freqs`` / ``wavelet`` /
    ``w0`` / ``order`` used in :func:`cwt`.

    Parameters
    ----------
    W : ndarray
        CWT coefficients ``(n_freqs, n_time)`` from :func:`cwt`.
    freqs : array
        The analysis frequencies returned by :func:`cwt`.
    sample_rate : float
        Sample rate (Hz).

    Returns
    -------
    ndarray
        Reconstructed 1-D signal.
    """
    Wc = np.asarray(W)
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    if Wc.ndim != 2 or Wc.shape[0] != freqs.size:
        raise ConfigurationError("inverse_cwt: W must be (n_freqs, n_time) matching freqs")
    if order is None:
        order = 4 if wavelet == "paul" else 2
    fs = float(sample_rate)
    _, factor = _wavelet_fourier(wavelet, 1.0, np.array([1.0]), w0, order)
    scales = factor * fs / freqs
    dln = float(np.mean(np.abs(np.diff(np.log(scales))))) if scales.size > 1 else 1.0
    c_delta = {"morlet": 0.776, "paul": 1.132, "dog": 3.541}.get(wavelet, 1.0)
    return np.sum(np.real(Wc) / np.sqrt(scales)[:, None], axis=0) * dln / c_delta


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
            w[max(1, nf - keep):] = 1.0
        if L < 0:
            w = 1.0 - w
        return c * w
    w = np.asarray(lifter, dtype=float)
    if w.shape != c.shape:
        raise ConfigurationError(
            f"cepstrum: lifter array {w.shape} must match cepstrum {c.shape}")
    return c * w


def cepstrum(x, *, window=None, nfft=None, lifter=None):
    """Real cepstrum ``irfft(log|rfft(x)|)``.

    Not invertible: discards phase. Use :func:`complex_cepstrum` /
    :func:`inverse_complex_cepstrum` for a reversible homomorphic transform.

    Parameters
    ----------
    x : 1-D array
        Real signal.
    window : None, str, or tuple
        :func:`scipy.signal.get_window` spec applied before the FFT to curb
        spectral leakage. ``None`` is rectangular.
    nfft : int, optional
        Zero-pad the FFT to ``nfft >= len(x)`` bins (finer quefrency spacing).
        ``None`` uses ``len(x)``.
    lifter : None, int, or 1-D array
        Quefrency liftering — see :func:`_apply_lifter`. ``None`` returns the
        raw cepstrum; a positive int keeps low quefrencies (spectral envelope),
        a negative int keeps high quefrencies (pitch / echo structure).
    """
    xr = np.asarray(x, dtype=float)
    if xr.ndim != 1:
        raise ConfigurationError("cepstrum: x must be 1-D")
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


def complex_cepstrum(x):
    """Complex cepstrum ``ifft(log(fft(x)))`` with phase unwrapping.

    Reversible via :func:`inverse_complex_cepstrum`.
    """
    xr = np.asarray(x, dtype=float)
    if xr.ndim != 1:
        raise ConfigurationError("complex_cepstrum: x must be 1-D")
    spectrum = np.fft.fft(xr)
    mag = np.abs(spectrum)
    mag = np.maximum(mag, np.finfo(float).tiny)
    log_spectrum = np.log(mag) + 1j * np.unwrap(np.angle(spectrum))
    return np.real(np.fft.ifft(log_spectrum))


def inverse_complex_cepstrum(c):
    """Invert :func:`complex_cepstrum`: ``x = real(ifft(exp(fft(c))))``."""
    cr = np.asarray(c, dtype=float)
    if cr.ndim != 1:
        raise ConfigurationError("inverse_complex_cepstrum: c must be 1-D")
    return np.real(np.fft.ifft(np.exp(np.fft.fft(cr))))


class Spectrogram:
    """Spectrogram computation and visualization."""

    def __init__(self, ref=REFERENCE_PRESSURE_WATER, **kwargs):
        """
        Spectrogram computation and visualization class.

        Parameters
        ----------
        ref : float
            Reference level for dB scaling.
        **kwargs
            Additional arguments for scipy.signal.spectrogram.
        """
        self.ref = ref

        # Default spectrogram parameters, overridden by kwargs if provided
        self.spec_params = {
            "nperseg": 8192,
            "noverlap": 4096,
            "window": "hann",
        }
        self.spec_params.update(kwargs)

    def compute(self, data, fs):
        """
        Compute the spectrogram using scipy.signal.spectrogram.

        Parameters
        ----------
        data : array_like
            Input signal array (Pa).
        fs : float
            Sampling frequency of the signal (Hz).

        Returns
        -------
        freqs : ndarray
            Array of frequencies (Hz).
        times : ndarray
            Array of time points (s).
        Sxx : ndarray
            2D array of spectrogram values.
        """
        freqs, times, Sxx = _sig.spectrogram(
            data, fs, scaling="density", mode="psd", **self.spec_params
        )

        self.frequencies = freqs
        self.times = times
        self.Sxx = Sxx

        return freqs, times, Sxx

    def plot(self, title="", ymin=1, ymax=None, vmin=0, vmax=200):
        """
        Plot the computed spectrogram as a colormap.

        Parameters
        ----------
        title : str
            Plot title.
        ymin : float
            Minimum frequency to display (Hz).
        ymax : float
            Maximum frequency to display (Hz).
        vmin : float
            Minimum value for color scaling (dB).
        vmax : float
            Maximum value for color scaling (dB).
        """
        if (
            not hasattr(self, "frequencies")
            or not hasattr(self, "times")
            or not hasattr(self, "Sxx")
        ):
            raise ConfigurationError(
                "Spectrogram.plot: compute() must be called before plotting"
            )
        Sxx_db = power_to_db(self.Sxx, self.ref)

        fig, ax = plt.subplots(figsize=(10, 6))
        pcm = ax.pcolormesh(
            self.times,
            self.frequencies,
            Sxx_db,
            cmap="jet",
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        cbar = fig.colorbar(pcm, ax=ax)

        if self.ref == REFERENCE_PRESSURE_WATER:
            ref = "1µ"
        elif self.ref == REFERENCE_PRESSURE_AIR:
            ref = "20µ"
        else:
            ref = f"{self.ref:02e}"
        cbar.set_label(f"Level [dB re {ref}Pa²/Hz]")
        ax.set_title(f"[Spectrogram] {title}", loc="left")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")

        if ymax is None:
            ymax = self.frequencies[-1]

        ax.set_ylim((ymin, ymax))
        ax.grid(which="both", alpha=0.25, color="black")

        return fig, ax
