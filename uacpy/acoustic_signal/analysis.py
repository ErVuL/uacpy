"""
Signal analysis tools for underwater acoustics.

Provides classes for computing power spectral densities (PSD), PSD
probability density functions, spectrograms, sound exposure levels (SEL),
and frequency response functions (FRF) with multiple estimation methods
(Welch, ETFE, least-squares FIR).
"""

import numpy as _np
import scipy.signal as _sig
import matplotlib.pyplot as plt
import math
from scipy.linalg import toeplitz
from scipy import fft as _fft
from matplotlib.gridspec import GridSpec


class PSD_PDF:
    """Compute the probability density function of PSD levels.

    Segments input signals, computes Welch PSD for each segment, and
    builds a histogram (PDF) of spectral levels across time segments.

    Parameters
    ----------
    ref : float
        Reference pressure for dB conversion (default 1e-6 Pa for water).
    seg_duration : float
        Duration of each time segment in seconds.
    overlap_pct : float
        Overlap percentage between segments.
    ddB : float
        Bin width in dB for the level histogram.
    lvlmin, lvlmax : float
        Minimum and maximum dB levels for histogram range.
    **kwargs
        Additional keyword arguments passed to ``scipy.signal.welch``.
    """

    def __init__(
        self,
        ref=1e-6,
        seg_duration=1,
        overlap_pct=50,
        ddB=1.0,
        lvlmin=0,
        lvlmax=150,
        **kwargs,
    ):
        self.seg_duration = seg_duration
        self.overlap_pct = overlap_pct
        self.ref = ref
        self.ddB = ddB
        self.lvlmin = lvlmin
        self.lvlmax = lvlmax

        self.welch_params = {
            "nperseg": 8192,
            "noverlap": 4096,
            "window": "hann",
            "scaling": "density",
        }
        self.welch_params.update(kwargs)

    def compute(self, data, fs):
        """Compute PSD PDF from 1D, list, or 2D signals."""
        # Normalize input
        if isinstance(data, list):
            signals = data
        else:
            data = _np.asarray(data)
            if data.ndim == 1:
                signals = [data]
            elif data.ndim == 2:
                if data.shape[0] < data.shape[1]:
                    signals = [data[i, :] for i in range(data.shape[0])]
                else:
                    signals = [data[:, i] for i in range(data.shape[1])]
            else:
                raise ValueError("Data must be 1D, 2D, or list of arrays")

        chunk_size = int(self.seg_duration * fs)
        overlap_samples = int(chunk_size * self.overlap_pct / 100)
        step = chunk_size - overlap_samples

        levels = _np.arange(self.lvlmin, self.lvlmax + self.ddB, self.ddB)
        psd_list = []

        # --- Loop over signals ---
        for sig in signals:
            nperseg = self.welch_params.get("nperseg", 8192)
            if chunk_size < nperseg:
                self.welch_params["nperseg"] = chunk_size
                self.welch_params["noverlap"] = int(chunk_size * self.overlap_pct / 100)

            for i in range(0, len(sig) - chunk_size + 1, step):
                chunk = sig[i : i + chunk_size]
                freqs, psd = _sig.welch(chunk, fs, **self.welch_params)
                psd_list.append(psd)

        if len(psd_list) == 0:
            raise ValueError(
                f"No PSD segments computed. "
                f"Check segment duration ({self.seg_duration}s) vs signal length ({len(sig)/fs:.2f}s)."
            )

        psd_array = _np.array(psd_list)

        # --- Compute mean and std in linear power ---
        mean_psd_linear = _np.mean(psd_array, axis=0)
        std_psd_linear = _np.std(psd_array, axis=0)

        # Convert mean to dB
        self.mean_psd = 10 * _np.log10(mean_psd_linear / self.ref**2)
        # Convert std to dB relative to mean
        self.std_psd = 10 * _np.log10((mean_psd_linear + std_psd_linear)/self.ref**2) - self.mean_psd

        # --- PDF in dB ---
        psd_segments_dB = 10 * _np.log10(psd_array / self.ref**2)
        pdf_matrix = _np.zeros((len(levels)-1, len(freqs)))
        for i in range(len(freqs)):
            hist, _ = _np.histogram(psd_segments_dB[:, i], bins=levels, density=True)
            pdf_matrix[:, i] = hist
        pdf_matrix[pdf_matrix == 0] = _np.nan

        self.binwidth_dB = self.ddB
        self.freqs = freqs
        self.levels = 10 ** (levels/10) * self.ref**2
        self.pdf = pdf_matrix

        return freqs, levels, pdf_matrix

    def plot(self, title="", ymin=0, ymax=200, vmin=0, vmax=None):
        if vmax is None:
            vmax = 1 / self.binwidth_dB

        fig, ax = plt.subplots(figsize=(10,6))
        align_ybins = self.binwidth_dB / 2

        pcm = ax.pcolormesh(
            self.freqs,
            10 * _np.log10(self.levels[:-1]/self.ref**2) + align_ybins,
            self.pdf,
            cmap="jet",
            shading="auto",
            vmin=vmin,
            vmax=vmax
        )

        fig.colorbar(
            pcm,
            ax=ax,
            label=f"Probability Density [{self.binwidth_dB:.1f} dB/bin]"
        )

        ax.plot(self.freqs, self.mean_psd, "k-", label="Mean level", linewidth=1.5)
        ax.plot(self.freqs, self.mean_psd + self.std_psd, "k--", label="Mean level ± STD")
        ax.plot(self.freqs, self.mean_psd - self.std_psd, "k--")

        ax.set_title(f"[PSD-PDF {self.seg_duration}s] {title}", loc="left")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Level [dB]")
        ax.set_xscale("log")
        ax.set_xlim((_np.max((self.freqs[0],1)), self.freqs[-1]))
        ax.set_ylim((ymin, ymax))
        ax.grid(which="both", alpha=0.5)
        ax.legend(loc="upper right")
        return fig, ax


class Spectrogram:
    def __init__(self, ref=1e-6, **kwargs):
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

        self.freqs = freqs
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
            not hasattr(self, "freqs")
            or not hasattr(self, "times")
            or not hasattr(self, "Sxx")
        ):
            raise RuntimeError("You must compute the spectrogram before plotting it.")

        # Convert to dB scale
        Sxx_db = 10 * _np.log10(self.Sxx / (self.ref**2))

        fig, ax = plt.subplots(figsize=(10, 6))
        pcm = ax.pcolormesh(
            self.times,
            self.freqs,
            Sxx_db,
            cmap="jet",
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        cbar = fig.colorbar(pcm, ax=ax)

        if self.ref == 1e-6:
            ref = "1µ"
        elif self.ref == 2e-5:
            ref = "20µ"
        else:
            ref = f"{self.ref:02e}"
        cbar.set_label(f"Level [dB re {ref}Pa²/Hz]")
        ax.set_title(f"[Spectrogram] {title}", loc="left")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        # ax.set_yscale("log")

        if ymax is None:
            ymax = self.freqs[-1]

        ax.set_ylim((ymin, ymax))
        ax.grid(which="both", alpha=0.25, color="black")

        return fig, ax


class SEL:
    def __init__(
        self,
        fmin=8.9125,
        fmax=22387,
        band_type="third_octave",
        num_bands=30,
        ref=1e-6,
        integration_time=None,
    ):
        """
        Initialize SEL calculator.

        Parameters
        ----------
        fmin : float
            Minimum frequency in Hz.
        fmax : float
            Maximum frequency in Hz.
        band_type : str
            Type of frequency bands ('octave', 'third_octave', or 'linear').
        num_bands : int
            Number of bands for linear band_type.
        ref : float
            Reference pressure level in Pa.
        integration_time : float or None
            Integration time in seconds (if None, uses full signal length).
        """
        self.fmin = fmin
        self.fmax = fmax
        self.band_type = band_type
        self.num_bands = num_bands
        self.duration = None
        self.ref = ref  # Store the reference level as an attribute
        self.integration_time = integration_time

    def _adjust_fmin_fmax(self, fs):
        """
        Adjust minimum and maximum frequencies to match band boundaries.

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz.
        """
        if self.band_type == "octave":
            self.fmin = 2 ** _np.floor(math.log2(self.fmin))
            self.fmax = 2 ** _np.ceil(math.log2(self.fmax))
            if self.fmax > fs / 2:
                self.fmax = 2 ** _np.floor(math.log2(self.fmax))
        elif self.band_type == "third_octave":
            base = math.pow(2, 1 / 6)
            self.fmin = base ** _np.floor(math.log(self.fmin, base))
            self.fmax = base ** _np.ceil(math.log(self.fmax, base))
            if self.fmax > fs / 2:
                self.fmax = base ** _np.floor(math.log(self.fmax, base))

    def _generate_frequency_bands(self, fs):
        """
        Generate frequency bands based on specified band_type.

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz.

        Returns
        -------
        bands : list of tuple
            List of tuples containing (low, center, high) frequencies for each band.
        """
        if self.fmin <= 0 or self.fmax <= self.fmin:
            raise ValueError("fmin must be > 0 and fmax must be greater than fmin.")

        if self.band_type in ["octave", "third_octave"]:
            self._adjust_fmin_fmax(fs)

        bands = []

        if self.band_type == "octave":
            base = math.sqrt(2)
            f_center = self.fmin
            while f_center < self.fmax:
                f_low = f_center / base
                f_high = f_center * base
                bands.append((f_low, f_center, f_high))
                f_center *= 2
            if bands and bands[-1][2] > self.fmax:
                bands[-1] = (bands[-1][0], bands[-1][1], self.fmax)

        elif self.band_type == "third_octave":
            base = math.pow(2, 1 / 6)
            f_center = self.fmin
            while f_center < self.fmax:
                f_low = f_center / base
                f_high = f_center * base
                bands.append((f_low, f_center, f_high))
                f_center *= math.pow(2, 1 / 3)
            if bands and bands[-1][2] > self.fmax:
                bands[-1] = (bands[-1][0], bands[-1][1], self.fmax)

        elif self.band_type == "linear":
            if self.num_bands <= 0:
                raise ValueError(
                    "num_bands must be a positive integer for linear bands."
                )
            band_width = (self.fmax - self.fmin) / self.num_bands
            f_low = self.fmin
            for _ in range(self.num_bands):
                f_high = f_low + band_width
                f_center = (f_low + f_high) / 2
                bands.append((f_low, f_center, f_high))
                f_low = f_high
            if bands and bands[-1][2] > self.fmax:
                bands[-1] = (bands[-1][0], bands[-1][1], self.fmax)

        else:
            raise ValueError(
                "Invalid band_type. Choose 'octave', 'third_octave', or 'linear'."
            )

        return bands

    def compute(self, data, fs, chunk_size=262144, nfft=None):
        """
        Compute Sound Exposure Level for each frequency band.

        Parameters
        ----------
        data : array_like
            Input time series data in Pa.
        fs : float
            Sampling frequency in Hz.
        chunk_size : int
            Number of samples per processing chunk.
        nfft : int, optional
            Number of FFT points.

        Returns
        -------
        sel : ndarray
            SEL values in Pa^2*s.
        bands : list of tuple
            Frequency bands as (low, center, high) tuples.
        """
        # Determine how much data to process based on integration_time
        if self.integration_time is not None:
            samples_to_process = min(int(self.integration_time * fs), len(data))
            data = data[:samples_to_process]

        self.bands = self._generate_frequency_bands(fs)
        self.duration = len(data) / fs
        if chunk_size > len(data):
            self.chunk_size = len(data)
        else:
            self.chunk_size = chunk_size

        if nfft is None:
            nfft = fs

        window = _sig.windows.hann(nfft)

        # Initialize frequency axis to determine band indices
        f = _np.fft.rfftfreq(nfft, d=1 / fs)

        # Initialize band indices
        band_indices = []
        for low, center, high in self.bands:
            idx = _np.logical_and(f >= low, f < high)
            band_indices.append(idx)

        # Initialize SEL accumulator
        self.sel = _np.zeros(len(self.bands))

        # Process data in chunks
        for i in range(0, len(data), chunk_size):
            chunk = data[i : min(i + chunk_size, len(data))]

            if len(chunk) < nfft:
                chunk = _np.pad(chunk, (0, nfft - len(chunk)))

            # Compute spectrogram for chunk
            f, t, Sxx = _sig.spectrogram(
                chunk, fs, window=window, noverlap=0, nfft=nfft, scaling="spectrum"
            )
            Sxx_sum = _np.sum(Sxx, axis=1)

            # Accumulate SEL in each band
            for k, idx in enumerate(band_indices):
                self.sel[k] += _np.sum(Sxx_sum[idx])

        return self.sel, self.bands

    def plot(self, title="", ylim=(0, 200)):
        """
        Plot Sound Exposure Level spectrum.

        Parameters
        ----------
        title : str
            Plot title.
        ylim : tuple
            Y-axis limits as (min, max).

        Returns
        -------
        fig : Figure
            Matplotlib figure.
        ax : Axes
            Matplotlib axes.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        Fedges = [low for low, _, _ in self.bands] + [self.bands[-1][2]]
        width = [Fedges[i + 1] - Fedges[i] for i in range(len(Fedges) - 1)]
        ax.bar(
            Fedges[:-1],
            10 * _np.log10(self.sel / (self.ref**2)),
            width=width,
            align="edge",
            edgecolor="black",
        )

        # If the duration is provided, include it in the title
        ax.set_title(f"[SEL {self.duration}s] {title}", loc="left")

        if self.ref == 1e-6:
            ref = "1µ"
        elif self.ref == 2e-5:
            ref = "20µ"
        else:
            ref = f"{self.ref:02e}"
        ax.set_ylabel(f"Level [dB re {ref}Pa²·s]")
        if self.band_type != "linear":
            ax.set_xscale("log")
        ax.set_xlabel(f"Frequency ({self.band_type}) [Hz]")
        ax.set_ylim(ylim)
        ax.grid(which="both", alpha=0.75)
        ax.set_axisbelow(True)
        return fig, ax


def shift_to_max_correlation(x, y):
    """
    Shift two signals based on the maximum cross-correlation.

    Computes the cross-correlation between the signals `x` and `y`,
    extracts the lag that maximizes the correlation, and shifts one of
    the signals accordingly.

    Parameters
    ----------
    x : ndarray
        First signal to be shifted.
    y : ndarray
        Second signal to be shifted.

    Returns
    -------
    x : ndarray
        Shifted first signal.
    y : ndarray
        Shifted second signal.
    """

    correlation = _sig.correlate(x, y, mode="full")
    lags = _sig.correlation_lags(x.size, y.size, mode="full")
    lag = lags[_np.argmax(correlation)]

    if lag < 0:
        y = y[-lag:]
        x = x[:lag]
    else:
        x = x[lag:]
        y = y[:-lag]

    return x, y


class PSD:
    """Power Spectral Density (PSD) computation and visualization.

    Parameters
    ----------
    ref : float
        Reference pressure for dB conversion (default 1e-6 Pa for water).
    **kwargs
        Additional keyword arguments passed to ``scipy.signal.welch``
        (e.g., nperseg, noverlap, window).
    """

    def __init__(self, ref=1e-6, **kwargs):
        """Initialize PSD with reference level and Welch parameters."""
        self.ref = ref

        # Default Welch parameters, overridden by kwargs if provided
        self.welch_params = {
            "nperseg": 8192,
            "noverlap": 4096,
            "window": "hann",
            "scaling": "density",
        }
        self.welch_params.update(kwargs)

    def compute(self, data, fs):
        """
        Compute the Power Spectral Density using Welch's method.

        Parameters
        ----------
        data : array_like
            Input signal array (Pa).
        fs : float
            Sampling frequency in Hz.

        Returns
        -------
        freqs : ndarray
            Frequency array in Hz.
        psd : ndarray
            PSD values in linear scale (Pa^2/Hz).
        """
        # Compute Welch periodogram
        freqs, Pxx = _sig.welch(data, fs, **self.welch_params)

        # Store frequencies and PSD values
        self.freqs = freqs
        self.psd = Pxx
        return freqs, Pxx

    def plot(self, title="", label="", ymin=0, ymax=150, **kwargs):
        """
        Plot the computed PSD as a line plot.

        Parameters
        ----------
        title : str
            Plot title.
        label : str
            Line label for legend.
        ymin, ymax : float
            Y-axis limits in dB.
        **kwargs
            Additional keyword arguments passed to ``ax.semilogx``.
        """
        if not hasattr(self, "freqs") or not hasattr(self, "psd"):
            raise RuntimeError("You must compute the PSD before plotting it.")

        # Convert PSD to dB scale
        psd_db = 10 * _np.log10(self.psd / (self.ref**2))

        # Plot PSD
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogx(self.freqs, psd_db, label=label, **kwargs)

        # Customize plot appearance
        ax.set_title(f"[PSD] {title}", loc="left")
        ax.set_xlabel("Frequency [Hz]")
        if self.ref == 1e-6:
            ref = "1µ"
        elif self.ref == 2e-5:
            ref = "20µ"
        else:
            ref = f"{self.ref:02e}"
        ax.set_ylabel(f"Level [dB re {ref}Pa²/Hz]")
        ax.set_ylim((ymin, ymax))
        ax.set_xlim((_np.max((self.freqs[0], 1)), self.freqs[-1]))
        ax.grid(which="both", alpha=0.75)
        plt.tight_layout()
        if label != "":
            ax.legend()

        return fig, ax

    def add_to_plot(self, ax, Fxx=None, Pxx=None, ref=None, label="", **kwargs):
        if Fxx is None and Pxx is None:
            Fxx = self.freqs
            Pxx = self.psd
        if ref is None:
            ref = self.ref

        psd_db = 10 * _np.log10(Pxx / (ref**2))
        ax.plot(Fxx, psd_db, label=label, **kwargs)
        if label != "":
            ax.legend()

        return ax


class FRF:
    def __init__(self, method="welch", estimator="H1", m=512, **kwargs):
        """
        Transfer Function (Frequency Response Function, FRF) computation and visualization class.

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
        frequency domain. It is defined as::

            H1(f) = Pyx(f) / Pxx(f)
            H2(f) = Pyy(f) / Pxy(f)

        where ``Pxx(f)`` is the Power Spectral Density (PSD) of the input
        signal (x) and ``Pxy(f)`` is the Cross-Power Spectral Density (CPSD)
        between input (x) and output (y).
        """
        import numpy as _np

        # Default parameters, overridden by kwargs if provided
        self.params = {
            "nperseg": 8192,
            "noverlap": 0,
        }
        self.params.update(kwargs)
        self.method = method
        self.estimator = estimator
        self.Minfo = _np.array([[0]])
        self.Vinfo = _np.array([[0]])
        self.m = m
        self.g = 0  # Impulse response

    def compute(
        self,
        x,
        y,
        fs,
        m=None,
        method=None,
        estimator=None,
        nperseg=None,
        noverlap=None,
        wavelet=None,
        scales=None,
        m_max=4096,
        stop_count=None,
    ):
        """
        Compute the Frequency Response Function (FRF), supporting both 1D and 2D inputs.

        If inputs are 2D, average results are computed over all measurements.

        Parameters
        ----------
        x : array_like
            Input signal array (reference) as 1D (single measurement) or 2D (rows = measurements).
        y : array_like
            Output signal array as 1D (single measurement) or 2D (rows = measurements).
        fs : float
            Sampling frequency (Hz).
        m : int or str, optional
            Impulse response length (for TF methods).
        method : str, optional
            Method to use ('welch', 'ls_fir', 'etfe', 'p_etfe').
        estimator : str, optional
            Estimator for Welch method ('H1', 'H2').
        nperseg : int, optional
            Segment length for Welch.
        noverlap : int, optional
            Overlap for Welch.
        wavelet : optional
            Wavelet parameter (reserved).
        scales : optional
            Scales parameter (reserved).
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
        # Update parameters
        if method is not None:
            self.method = method
        if nperseg is not None:
            self.params["nperseg"] = nperseg
        if noverlap is not None:
            self.params["noverlap"] = noverlap
        if estimator is not None:
            self.estimator = estimator
        if scales is not None:
            self.scales = scales
        if m is not None:
            self.m = m
        if stop_count is None:
            self.stop_count = m_max

        # Convert inputs to 2D arrays (rows = measurements)
        x = _np.asarray(x)
        y = _np.asarray(y)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of measurements")
        n_meas = x.shape[0]

        # Initialize lists to store results from all measurements
        m_list, freqs_list, tf_list, coh_list = [], [], [], []

        for i in range(n_meas):
            # Extract the i-th measurement
            x_i = x[i, :].ravel()
            y_i = y[i, :].ravel()

            # Compute FRF for this measurement
            if self.method == "welch":
                freqs_i, tf_i, coh_i = self.compute_welch(x_i, y_i, fs)
                coh_list.append(coh_i)
            elif self.method == "ls_fir":
                freqs_i, tf_i, g_i = self.compute_lsfir(
                    y_i, x_i, fs, self.m, len(x_i), m_max=m_max, stop_count=stop_count
                )
                m_list.append(len(g_i))
            elif self.method == "etfe":
                freqs_i, tf_i = self.compute_etfe(x_i, y_i, fs)
            elif self.method == "p_etfe":
                freqs_i, tf_i = self.compute_periodic_etfe(x_i, y_i, fs)
            else:
                raise ValueError(f"Unsupported method: {self.method}")

            # Append results
            tf_list.append(tf_i)

        # Average results
        freqs = freqs_i
        tf = _np.mean(tf_list, axis=0)

        # Update object state
        self.freqs = freqs
        self.tf = tf
        if self.method == "welch":
            self.coh = (
                _np.mean(coh_list, axis=0)
                if all(c is not None for c in coh_list)
                else None
            )
        if self.method == "ls_fir":
            self.g = g_i  # For 2D inputs, uses last channel's impulse response
            self.m = int(
                _np.mean(m_list) if all(mi is not None for mi in m_list) else None
            )

        return freqs, tf

    def compute_welch(self, x, y, fs):
        """
        Compute the Frequency Response Function (FRF) using Welch's method.

        This method is dedicated to stationary signals. Coherence indicates
        the degree of linear dependency between input (x) and output (y) at
        each frequency.

        Parameters
        ----------
        x : array_like
            Input signal array (reference).
        y : array_like
            Output signal array.
        fs : float
            Sampling frequency of the signals (Hz).

        Returns
        -------
        freqs : ndarray
            Array of frequencies (Hz).
        tf : ndarray
            Complex transfer function.
        coh : ndarray
            Array of coherence values.
        """

        # Compute cross-spectral densities
        freqs, Pxx = _sig.welch(x, fs, scaling="density", **self.params)
        _, Pyy = _sig.welch(y, fs, scaling="density", **self.params)
        _, Pxy = _sig.csd(y, x, fs, scaling="density", **self.params)

        # Compute transfer function based on estimator choice
        if self.estimator == "H2":
            tf = Pyy / Pxy
        else:  # Default to H1
            tf = _np.conj(Pxy) / Pxx

        # Compute coherence
        coh = abs(Pxy) ** 2 / (Pxx * Pyy)

        return freqs, tf, coh

    def compute_periodic_etfe(self, x, y, fs, nperseg=None):
        """
        Compute ETFE for periodic data.

        Parameters
        ----------
        x : array_like
            Input signal.
        y : array_like
            Output signal.
        fs : float
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

        if nperseg:
            self.params["nperseg"] = nperseg

        # For periodic data, we compute at frequencies k*2*pi/period/Ts
        # up to the Nyquist frequency
        period = self.params["nperseg"]
        n_periods = len(x) // period

        if n_periods < 1:
            raise ValueError("Signal length must be at least one period.")

        # Extract a whole number of periods
        x = x[: n_periods * period]
        y = y[: n_periods * period]

        # Reshape to n_periods rows of period columns
        x_reshaped = x.reshape(n_periods, period)
        y_reshaped = y.reshape(n_periods, period)

        # Average over periods to reduce noise
        x_avg = _np.mean(x_reshaped, axis=0)
        y_avg = _np.mean(y_reshaped, axis=0)

        # Compute FFT of averaged signals
        X = _np.fft.rfft(x_avg) + _np.finfo(float).eps
        Y = _np.fft.rfft(y_avg) + _np.finfo(float).eps

        # Compute frequencies
        freqs = _np.fft.rfftfreq(period, d=1 / fs)

        # Initialize transfer function
        tf = _np.zeros_like(X, dtype=complex)

        # Compute transfer function where input has significant energy
        tf = Y / X

        return freqs, tf

    def compute_etfe(self, x, y, fs):
        """
        Compute the Empirical Transfer Function Estimate (ETFE).

        This method directly estimates the transfer function by dividing the
        output Fourier transform by the input Fourier transform.

        Parameters
        ----------
        x : array_like
            Input signal array (reference).
        y : array_like
            Output signal array.
        fs : float
            Sampling frequency of the signals (Hz).

        Returns
        -------
        freqs : ndarray
            Array of frequencies (Hz).
        tf : ndarray
            Complex transfer function.
        """

        # Ensure signals are the same length
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]

        # Compute FFTs
        X = _np.fft.rfft(x) + _np.finfo(float).eps
        Y = _np.fft.rfft(y)

        # Determine frequency grid based on n_freqs
        n_fft = min_len
        freqs = _np.fft.rfftfreq(n_fft, d=1 / fs)

        # Initialize transfer function
        tf = _np.zeros_like(X, dtype=complex)

        # Compute transfer function
        tf = Y / X

        return freqs, tf

    def compute_lsfir(self, y, u, fs, m, N, m_max=4096, stop_count=50, nperseg=None):
        """
        Compute the finite impulse response estimation using an information matrix/vector method.
        Supports model order selection using AIC, BIC, FPE, or Mallows' Cp.

        Parameters
        ----------
        y : array_like
            System output.
        u : array_like
            System input.
        m : int or str
            Model order or selection criterion ('AIC', 'BIC', 'FPE', 'CP').
        N : int
            Number of data points to consider (N >= m).
        fs : float
            Sampling rate in Hz.
        m_max : int
            Maximum model order for automatic selection.
        stop_count : int
            Stop search after stop_count consecutive steps with no improvement.
        nperseg : int, optional
            Frequency axis will be nperseg/2+1 samples between 0 and fs/2.

        Returns
        -------
        freqs : ndarray
            Frequency array (Hz).
        h : ndarray
            Complex frequency response.
        g : ndarray
            Impulse response estimate.
        """

        if nperseg:
            self.params["nperseg"] = nperseg

        y = _np.array(y)
        u = _np.array(u)

        if m in ["AIC", "FPE", "CP", "BIC"]:
            # Model order selection
            m_max = min(m_max, N - 1)
            best_score = _np.inf
            best_m = 1
            best_g = None
            count = 0

            if m == "CP":
                # Estimation of noise variance (for Cp)
                full_model_m = min(m_max, N - 1)
                u_temp = u[:N].copy()
                phiuu_full = _np.zeros(full_model_m)
                phiuy_full = _np.zeros(full_model_m)

                for i in range(full_model_m):
                    phiuu_full[i] = _np.dot(u[:N], u_temp)
                    phiuy_full[i] = _np.dot(y[:N], u_temp)
                    u_temp = _np.concatenate(([u_temp[-1]], u_temp[:-1]))

                sigma2 = _np.sum((y[:N] - _np.mean(y[:N])) ** 2) / (N - full_model_m)

            for m_candidate in range(1, m_max + 1):
                try:
                    # Compute information matrix and vector
                    u_temp = u[:N].copy()
                    phiuu = _np.zeros(m_candidate)
                    phiuy = _np.zeros(m_candidate)

                    for i in range(m_candidate):
                        phiuu[i] = _np.dot(u[:N], u_temp)
                        phiuy[i] = _np.dot(y[:N], u_temp)
                        u_temp = _np.concatenate(
                            ([u_temp[-1]], u_temp[:-1])
                        )  # Shift right

                    A = toeplitz(phiuu)
                    u_flipped = _np.flip(u[:N]).copy()
                    W = _np.zeros((m_candidate - 1, m_candidate))

                    for i in range(m_candidate - 1):
                        u_flipped = _np.concatenate(([u_flipped[-1]], u_flipped[:-1]))
                        W[i, :] = u_flipped[:m_candidate]

                    Minfo = A - _np.dot(W.T, W)
                    Vinfo = phiuy - _np.dot(W.T, y[: m_candidate - 1])

                    g = _np.linalg.solve(Minfo, Vinfo)

                    # Compute residuals
                    y_hat = _np.convolve(u[:N], g, mode="full")[:N]
                    residuals = y[:N] - y_hat
                    sse = _np.sum(residuals**2) / (N - m_candidate)

                    if sse < 1e-9:
                        continue  # Avoid log issues

                    if m == "AIC":  # AICF
                        # AIC: score = _np.log(sse) + 2*m_candidate/N
                        score = _np.log(sse) + (1 + m_candidate / (N - m_candidate)) / (
                            1 - m_candidate / (N - m_candidate)
                        )

                    elif m == "FPE":  # FPEF
                        # FPE: score = sse * (1 + m_candidate/N) / (1 - m_candidate/N)
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
                        score = _np.log(sse) + (m_candidate * _np.log(N)) / N

                    if score < best_score:
                        best_score = score
                        best_m = m_candidate
                        best_g = g
                        count = 0
                    else:
                        count += 1

                    if count >= stop_count:
                        break  # Stop search early

                except _np.linalg.LinAlgError:
                    continue  # Skip singular matrices

            m = best_m

            # Recompute Minfo and Vinfo for the best m
            u_temp = u[:N].copy()
            phiuu = _np.zeros(m)
            phiuy = _np.zeros(m)

            for i in range(m):
                phiuu[i] = _np.dot(u[:N], u_temp)
                phiuy[i] = _np.dot(y[:N], u_temp)
                u_temp = _np.concatenate(([u_temp[-1]], u_temp[:-1]))

            A = toeplitz(phiuu)
            u_flipped = _np.flip(u[:N]).copy()
            W = _np.zeros((m - 1, m))

            for i in range(m - 1):
                u_flipped = _np.concatenate(([u_flipped[-1]], u_flipped[:-1]))
                W[i, :] = u_flipped[:m]

            self.Minfo = A - _np.dot(W.T, W)
            self.Vinfo = phiuy - _np.dot(W.T, y[: m - 1])
            g = best_g

        else:
            # Given m, compute directly
            u_temp = u[:N].copy()
            phiuu = _np.zeros(m)
            phiuy = _np.zeros(m)

            for i in range(m):
                phiuu[i] = _np.dot(u[:N], u_temp)
                phiuy[i] = _np.dot(y[:N], u_temp)
                u_temp = _np.concatenate(([u_temp[-1]], u_temp[:-1]))

            A = toeplitz(phiuu)
            u_flipped = _np.flip(u[:N]).copy()
            W = _np.zeros((m - 1, m))

            for i in range(m - 1):
                u_flipped = _np.concatenate(([u_flipped[-1]], u_flipped[:-1]))
                W[i, :] = u_flipped[:m]

            self.Minfo = A - _np.dot(W.T, W)
            self.Vinfo = phiuy - _np.dot(W.T, y[: m - 1])
            g = _np.linalg.solve(self.Minfo, self.Vinfo)

        # Frequency response
        w_imp, h = _sig.freqz(g, worN=int(self.params["nperseg"] / 2 + 1))
        freqs = w_imp * fs / (2 * _np.pi)

        return freqs, h, g

    def plot_impulse_info(self, title="", figsize=(12, 8), **kwargs):
        """
        Plot the LS-FIR estimation diagnostics.

        Shows the information matrix (Minfo), information vector (Vinfo),
        and the estimated impulse response (g) in a 2x2 grid layout.
        Only available after calling ``compute()`` with ``method='ls_fir'``.

        Parameters
        ----------
        title : str
            Title prefix for each subplot.
        figsize : tuple
            Figure size (width, height) in inches.

        Returns
        -------
        fig : Figure
            Matplotlib figure.
        axes : list of Axes
            List of [matrix_ax, vector_ax, impulse_ax].
        """

        # Create figure and gridspec
        fig = plt.figure(figsize=figsize)

        # Define a 2x2 grid with adjusted height ratios
        gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])

        # Plot Minfo as heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        im = ax1.imshow(self.Minfo, cmap="viridis", aspect="equal")
        ax1.set_title(f"[Information Matrix] {title}", loc="left")
        ax1.set_xlabel("Index j")
        ax1.set_ylabel("Index i")

        # Add colorbar to Minfo plot
        cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
        cbar.set_label("Correlation Value")

        # Plot Vinfo as bar chart
        ax2 = fig.add_subplot(gs[0, 1])
        indices = _np.arange(len(self.Vinfo))
        ax2.bar(indices, self.Vinfo, color="skyblue", edgecolor="navy")
        ax2.set_title(f"[Information Vector] {title}", loc="left")
        ax2.set_xlabel("Index i")
        ax2.set_ylabel("Cross-correlation Value")

        # Plot impulse response (self.g)
        ax3 = fig.add_subplot(gs[1, :])  # Span both columns in the second row
        ax3.plot(self.g, color="red", linestyle="-", marker="o", markersize=4)
        ax3.set_title(f"[Impulse Response] {title}", loc="left")
        ax3.set_xlabel("Time Index")
        ax3.set_ylabel("Amplitude")
        ax3.grid(True)

        plt.tight_layout()

        return fig, [ax1, ax2, ax3]

    def plot_coh(self, title="", label="", **kwargs):
        """
        Plot the coherence function.

        Only available after calling ``compute()`` with ``method='welch'``.

        Parameters
        ----------
        title : str
            Plot title.
        label : str
            Line label for legend.
        **kwargs
            Additional keyword arguments passed to ``ax.plot``.

        Returns
        -------
        fig : Figure
            Matplotlib figure.
        ax : Axes
            Matplotlib axes.
        """
        fig, ax = plt.subplots(1, 1)

        if label != "":
            addstr = f"[{self.method}-{self.estimator}] "
            label = addstr.upper() + label

        # Coherence plot
        ax.plot(self.freqs, self.coh, label=label, **kwargs)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Coherence")
        ax.set_xscale("log")
        ax.set_ylim((0.75, 1.01))
        ax.set_xlim((_np.max((self.freqs[0], 1)), self.freqs[-1]))
        ax.grid(which="major", alpha=0.75)
        ax.grid(which="minor", alpha=0.25)
        ax.tick_params(axis="x", which="both")
        ax.set_title(f"[Coherence] {title}", loc="left")

        if label != "":
            ax.legend()

        return fig, ax

    def add_coherence_to_plot(self, axes, title="", label="", **kwargs):
        ax = axes

        if label != "":
            addstr = f"[{self.method}-{self.estimator}] "
            label = addstr.upper() + label

        # Coherence plot
        ax.plot(self.freqs, self.coh, label=label, **kwargs)

        if label != "":
            ax.legend()

        return ax

    def plot(self, title="", label="", ymin=-60, ymax=60, **kwargs):
        """
        Plot the computed Transfer Function as magnitude and phase plots.

        Parameters
        ----------
        title : str
            Plot title.
        label : str
            Legend label.
        ymin, ymax : float
            Y-axis limits for magnitude plot (dB).
        **kwargs
            Additional plotting arguments.

        Notes
        -----
        The magnitude (in dB) is computed as::

            20 * log10(|H(f)|)

        Phase is given in degrees.
        Coherence is plotted to assess the reliability of the FRF.
        """

        if not hasattr(self, "freqs") or not hasattr(self, "tf"):
            raise RuntimeError(
                "You must compute the Transfer Function before plotting it."
            )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
        ax1.set_title(f"[FRF] {title}", loc="left")

        if label != "":
            if self.method == "welch":
                addstr = f"[{self.method}-{self.estimator}] "
                label = addstr.upper() + label
            elif self.method == "p_etfe":
                addstr = f"[{self.method}-{self.params['nperseg']}] "
                label = addstr.upper() + label
            elif self.method == "ls_fir":
                addstr = f"[{self.method}-{self.m}] "
                label = addstr.upper() + label
            else:
                addstr = f"[{self.method}] "
                label = addstr.upper() + label

        # Magnitude plot
        mag_db = 20 * _np.log10(_np.abs(self.tf))
        ax1.plot(self.freqs, mag_db, label=label, **kwargs)
        ax1.set_ylabel("Magnitude [dB]")
        ax1.set_xscale("log")
        ax1.set_ylim((ymin, ymax))
        ax1.set_xlim((_np.max((self.freqs[0], 1)), self.freqs[-1]))
        ax1.grid(which="major", alpha=0.75)
        ax1.grid(which="minor", alpha=0.25)
        ax1.set_xticklabels([])
        ax1.tick_params(axis="x", which="both", bottom=False)

        # Phase plot
        phase_deg = _np.angle(self.tf, deg=True)
        ax2.plot(self.freqs, phase_deg, label=label, **kwargs)
        ax2.set_ylabel("Phase [degrees]")
        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_xscale("log")
        ax2.set_ylim((-180, 180))
        ax2.set_xlim((_np.max((self.freqs[0], 1)), self.freqs[-1]))
        ax2.grid(which="major", alpha=0.75)
        ax2.grid(which="minor", alpha=0.25)
        ax2.tick_params(axis="x", which="both", bottom=True)

        if label != "":
            ax1.legend()
            ax2.legend()

        plt.tight_layout()

        return fig, (ax1, ax2)

    def add_to_plot(
        self,
        axes,
        freqs=None,
        mag=None,
        phase=None,
        method=None,
        estimator=None,
        label="",
        **kwargs,
    ):
        """
        Add transfer function data to existing plots.

        Parameters
        ----------
        axes : tuple
            Tuple of (magnitude_axis, phase_axis).
        freqs : ndarray, optional
            Frequency array. If None, uses stored values.
        mag : ndarray, optional
            Magnitude array. If None, uses stored values.
        phase : ndarray, optional
            Phase array. If None, uses stored values.
        method : str, optional
            Method name override.
        estimator : str, optional
            Estimator name override.
        label : str
            Legend label.
        **kwargs
            Additional plotting arguments.
        """

        ax1, ax2 = axes

        if estimator is None:
            estimator = self.estimator
        if method is None:
            method = self.method

        if label != "":
            if self.method == "welch":
                addstr = f"[{self.method}-{self.estimator}] "
                label = addstr.upper() + label
            elif self.method == "p_etfe":
                addstr = f"[{self.method}-{self.params['nperseg']}] "
                label = addstr.upper() + label
            elif self.method == "ls_fir":
                addstr = f"[{self.method}-{self.m}] "
                label = addstr.upper() + label
            else:
                addstr = f"[{self.method}] "
                label = addstr.upper() + label

        if freqs is None or mag is None:
            ax1.plot(
                self.freqs, 20 * _np.log10(_np.abs(self.tf)), label=label, **kwargs
            )
        else:
            ax1.plot(freqs, 20 * _np.log10(mag), label=label, **kwargs)

        if freqs is None or phase is None:
            ax2.plot(self.freqs, _np.angle(self.tf, deg=True), label=label, **kwargs)
        else:
            ax2.plot(freqs, phase, label=label, **kwargs)

        if label != "":
            ax1.legend()
            ax2.legend()

        return axes

class FKTransform:
    
    def __init__(self, ref=1e-6, **kwargs):
        """
        Frequency-Wavenumber (f-k) transform computation and visualization class.

        Parameters
        ----------
        ref : float
            Reference level for dB scaling.
        **kwargs
            Additional keyword arguments.
        """
        self.ref = ref

    def compute(self, data, fs, dx):
        """
        Compute the frequency-wavenumber (f-k) spectrum using a 2D FFT.

        Parameters
        ----------
        data : array_like
            2D array with shape (nt, nx).
        fs : float
            Temporal sampling frequency (Hz).
        dx : float
            Spatial sampling interval (m).

        Returns
        -------
        freqs : ndarray
            Frequency axis (Hz).
        wavenumbers : ndarray
            Wavenumber axis (1/m).
        fk : ndarray
            2D f-k spectrum (linear scale).
        """
        data = _np.asarray(data)
        nt, nx = data.shape

        # Forward 2D FFT
        FK = _np.fft.fftshift(_np.fft.fft2(data), axes=(0, 1))

        # Power spectrum
        FKp = _np.abs(FK) ** 2

        # Axes
        freqs = _np.fft.fftshift(_np.fft.fftfreq(nt, d=1 / fs))
        wavenumbers = _np.fft.fftshift(_np.fft.fftfreq(nx, d=dx))

        # Store results
        self.freqs = freqs
        self.wavenumbers = wavenumbers
        self.FK = FK
        self.fk = FKp

        return freqs, wavenumbers, FKp

    def inverse(self, FK=None):
        """
        Inverse frequency-wavenumber (f-k) transform.

        Parameters
        ----------
        FK : ndarray, optional
            Complex f-k spectrum. If None, uses internally stored FK.

        Returns
        -------
        data_rec : ndarray
            Reconstructed time-space signal.
        """
        if FK is None:
            if not hasattr(self, "FK"):
                raise RuntimeError("No f–k data available for inverse transform.")
            FK = self.FK

        # Undo fftshift before inverse FFT
        FK_unshifted = _np.fft.ifftshift(FK, axes=(0, 1))

        # Inverse 2D FFT
        data_rec = _np.fft.ifft2(FK_unshifted)

        # Return real-valued signal if applicable
        data_rec = _np.real(data_rec)

        return data_rec

    def plot(self, title="", vmin=-60, vmax=20, **kwargs):
        """
        Plot the computed f-k spectrum as an image.

        Parameters
        ----------
        title : str
            Plot title.
        vmin, vmax : float
            Color scale limits (dB).
        **kwargs
            Additional keyword arguments passed to ``ax.imshow``.
        """
        if not hasattr(self, "freqs") or not hasattr(self, "wavenumbers") or not hasattr(self, "fk"):
            raise RuntimeError("You must compute the f–k transform before plotting it.")

        fk_db = 10 * _np.log10(self.fk / (self.ref ** 2))

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(
            fk_db,
            extent=[
                self.wavenumbers[0],
                self.wavenumbers[-1],
                self.freqs[0],
                self.freqs[-1],
            ],
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

        ax.set_title(f"[f–k] {title}", loc="left")
        ax.set_xlabel("Wavenumber [1/m]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlim((self.wavenumbers[0], self.wavenumbers[-1]))
        ax.set_ylim((0, self.freqs[-1]))
        ax.grid(alpha=0.3)
        
        if self.ref == 1e-6:
            ref = "1µ"
        elif self.ref == 2e-5:
            ref = "20µ"
        else:
            ref = f"{self.ref:02e}"

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f"Level [dB re {ref}Pa².m/Hz]")

        plt.tight_layout()
        return fig, ax

