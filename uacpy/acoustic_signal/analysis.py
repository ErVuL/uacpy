"""Spectral and level estimators: PSD, PPSD (Welch density-scaled) and
SEL (band-integrated sound exposure level). dB references default to 1 uPa.
"""

import math

import numpy as np
import scipy.signal as _sig
import matplotlib.pyplot as plt

from uacpy.core.exceptions import ConfigurationError

from uacpy.core.constants import REFERENCE_PRESSURE_AIR, REFERENCE_PRESSURE_WATER
from uacpy.core.acoustics import power_to_db


class PPSD:
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
        ref=REFERENCE_PRESSURE_WATER,
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
        """Compute PSD PDF from 1D, list, or 2D signals.

        2-D input is interpreted with the *longer* axis as time: an
        ``(n_signals, n_samples)`` array with more samples than signals
        iterates rows, otherwise columns. For arrays where that
        heuristic is wrong (more channels than samples), pass an
        explicit list of 1-D signals instead.
        """
        # Normalize input
        if isinstance(data, list):
            signals = data
        else:
            data = np.asarray(data)
            if data.ndim == 1:
                signals = [data]
            elif data.ndim == 2:
                if data.shape[0] < data.shape[1]:
                    signals = [data[i, :] for i in range(data.shape[0])]
                else:
                    signals = [data[:, i] for i in range(data.shape[1])]
            else:
                raise ConfigurationError(
                    "PPSD.compute: data must be 1-D, 2-D, or a list of 1-D arrays; "
                    f"got ndim={data.ndim}"
                )

        chunk_size = int(self.seg_duration * fs)
        overlap_samples = int(chunk_size * self.overlap_pct / 100)
        step = chunk_size - overlap_samples
        if step <= 0:
            raise ConfigurationError(
                f"PPSD.compute: overlap_pct ({self.overlap_pct}) too high — "
                "chunks never advance; require overlap_pct < 100."
            )

        levels = np.arange(self.lvlmin, self.lvlmax + self.ddB, self.ddB)
        psd_list = []

        # --- Loop over signals ---
        for sig in signals:
            # Local per-signal Welch params: a short signal shrinks nperseg for
            # itself only, without lowering resolution for later signals.
            welch_params = dict(self.welch_params)
            nperseg = welch_params.get("nperseg", 8192)
            if chunk_size < nperseg:
                welch_params["nperseg"] = chunk_size
                welch_params["noverlap"] = int(chunk_size * self.overlap_pct / 100)

            for i in range(0, len(sig) - chunk_size + 1, step):
                chunk = sig[i: i + chunk_size]
                freqs, psd = _sig.welch(chunk, fs, **welch_params)
                psd_list.append(psd)

        if len(psd_list) == 0:
            raise ConfigurationError(
                f"PPSD.compute: no PSD segments computed; "
                f"seg_duration={self.seg_duration}s vs signal length={len(sig)/fs:.2f}s"
            )

        psd_array = np.array(psd_list)

        # Convert to dB once; mean/std/percentiles all live in dB-space so
        # they line up with the histogram (and with how users read PPSD).
        psd_segments_dB = power_to_db(psd_array, self.ref)

        self.mean_psd = np.mean(psd_segments_dB, axis=0)
        self.std_psd = np.std(psd_segments_dB, axis=0)

        pdf_matrix = np.zeros((len(levels)-1, len(freqs)))
        for i in range(len(freqs)):
            hist, _ = np.histogram(psd_segments_dB[:, i], bins=levels, density=True)
            pdf_matrix[:, i] = hist
        pdf_matrix[pdf_matrix == 0] = np.nan

        self.binwidth_dB = self.ddB
        self.frequencies = freqs
        self.levels = levels          # dB bin edges, same unit compute() returns
        self.pdf = pdf_matrix

        return freqs, levels, pdf_matrix

    def plot(self, title="", ymin=0, ymax=200, vmin=0, vmax=None):
        """Plot the computed PSD PDF as a 2-D histogram over frequency/level."""
        if vmax is None:
            vmax = 1 / self.binwidth_dB

        fig, ax = plt.subplots(figsize=(10, 6))
        align_ybins = self.binwidth_dB / 2

        pcm = ax.pcolormesh(
            self.frequencies,
            self.levels[:-1] + align_ybins,
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

        ax.plot(self.frequencies, self.mean_psd, "k-", label="Mean level", linewidth=1.5)
        ax.plot(self.frequencies, self.mean_psd + self.std_psd, "k--", label="Mean level ± STD")
        ax.plot(self.frequencies, self.mean_psd - self.std_psd, "k--")

        ax.set_title(f"[PPSD {self.seg_duration}s] {title}", loc="left")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Level [dB]")
        ax.set_xscale("log")
        ax.set_xlim((np.max((self.frequencies[0], 1)), self.frequencies[-1]))
        ax.set_ylim((ymin, ymax))
        ax.grid(which="both", alpha=0.5)
        ax.legend(loc="upper right")
        return fig, ax


class SEL:
    """Sound Exposure Level (SEL) computation in configurable frequency bands."""

    def __init__(
        self,
        fmin=8.9125,
        fmax=22387,
        band_type="third_octave",
        num_bands=30,
        ref=REFERENCE_PRESSURE_WATER,
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
        Snap the configured band edges to band boundaries for this ``fs``.

        Returns the adjusted ``(fmin, fmax)`` without mutating the
        configured ``self.fmin`` / ``self.fmax``, so ``compute()`` calls
        with different sampling rates don't drift the configured band.
        """
        fmin, fmax = self.fmin, self.fmax
        if self.band_type == "octave":
            fmin = 2 ** np.floor(math.log2(fmin))
            fmax = 2 ** np.ceil(math.log2(fmax))
            if fmax > fs / 2:
                fmax = 2 ** np.floor(math.log2(fmax))
        elif self.band_type == "third_octave":
            base = math.pow(2, 1 / 6)
            fmin = base ** np.floor(math.log(fmin, base))
            fmax = base ** np.ceil(math.log(fmax, base))
            if fmax > fs / 2:
                fmax = base ** np.floor(math.log(fmax, base))
        return fmin, fmax

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
            raise ConfigurationError(
                f"SEL._generate_frequency_bands: require fmin > 0 and fmax > fmin; "
                f"got fmin={self.fmin}, fmax={self.fmax}"
            )

        fmin, fmax = self.fmin, self.fmax
        if self.band_type in ["octave", "third_octave"]:
            fmin, fmax = self._adjust_fmin_fmax(fs)

        bands = []

        if self.band_type == "octave":
            base = math.sqrt(2)
            f_center = fmin
            while f_center < fmax:
                f_low = f_center / base
                f_high = f_center * base
                bands.append((f_low, f_center, f_high))
                f_center *= 2
            if bands and bands[-1][2] > fmax:
                bands[-1] = (bands[-1][0], bands[-1][1], fmax)

        elif self.band_type == "third_octave":
            base = math.pow(2, 1 / 6)
            f_center = fmin
            while f_center < fmax:
                f_low = f_center / base
                f_high = f_center * base
                bands.append((f_low, f_center, f_high))
                f_center *= math.pow(2, 1 / 3)
            if bands and bands[-1][2] > fmax:
                bands[-1] = (bands[-1][0], bands[-1][1], fmax)

        elif self.band_type == "linear":
            if self.num_bands <= 0:
                raise ConfigurationError(
                    f"SEL._generate_frequency_bands: num_bands must be a "
                    f"positive integer for linear bands; got {self.num_bands}"
                )
            band_width = (fmax - fmin) / self.num_bands
            f_low = fmin
            for _ in range(self.num_bands):
                f_high = f_low + band_width
                f_center = (f_low + f_high) / 2
                bands.append((f_low, f_center, f_high))
                f_low = f_high
            if bands and bands[-1][2] > fmax:
                bands[-1] = (bands[-1][0], bands[-1][1], fmax)

        else:
            raise ConfigurationError(
                f"SEL._generate_frequency_bands: unknown band_type={self.band_type!r}; "
                "valid: 'octave', 'third_octave', 'linear'"
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

        Notes
        -----
        Each chunk is split into rectangular (boxcar) segments of length
        ``nfft`` with ``noverlap=0`` and no detrending.
        ``scipy.signal.spectrogram`` with ``scaling="density"`` returns the
        PSD in Pa²/Hz; summing it over a band's bins gives that segment's
        mean-square pressure (Parseval), and ``Δf·(nfft/fs)=1`` so the
        summed PSD is directly the band exposure in Pa²·s. The total over
        all bands equals the discrete ``∫p²dt`` to within FFT band-edge
        leakage.
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
        nfft = int(nfft)

        window = _sig.windows.boxcar(nfft)
        f = np.fft.rfftfreq(nfft, d=1 / fs)
        # Assign each FFT bin to exactly one band via the (contiguous) band
        # edges, instead of independent [low, high) masks. At coarse FFT
        # resolution several sub-bin-wide low-frequency bands would otherwise
        # map to the same 1-Hz bin (double-counted) or to none (dropped); a
        # single digitize keeps every bin in exactly one band, so per-band
        # levels and the total stay consistent.
        edges = np.array([b[0] for b in self.bands] + [self.bands[-1][2]])
        # np.digitize: bin index in [1, len(edges)-1] for f inside the band span,
        # 0 below the first edge, len(edges) at/above the last. Map to band idx.
        bin_band = np.digitize(f, edges) - 1
        nb = len(self.bands)
        band_bins = [np.where(bin_band == k)[0] for k in range(nb)]
        self.sel = np.zeros(len(self.bands))

        # Process data in chunks
        for i in range(0, len(data), chunk_size):
            chunk = data[i: min(i + chunk_size, len(data))]

            # Pad to a whole number of segments so spectrogram keeps the remainder.
            n_seg = max(1, -(-len(chunk) // nfft))
            pad = n_seg * nfft - len(chunk)
            if pad:
                chunk = np.pad(chunk, (0, pad))
            f, t, Sxx = _sig.spectrogram(
                chunk, fs, window=window, noverlap=0, nfft=nfft,
                detrend=False, scaling="density",
            )
            Sxx_sum = np.sum(Sxx, axis=1)

            # Accumulate SEL in each band
            for k, idx in enumerate(band_bins):
                self.sel[k] += np.sum(Sxx_sum[idx])

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
            power_to_db(self.sel, self.ref),
            width=width,
            align="edge",
            edgecolor="black",
        )

        # If the duration is provided, include it in the title
        ax.set_title(f"[SEL {self.duration}s] {title}", loc="left")

        if self.ref == REFERENCE_PRESSURE_WATER:
            ref = "1µ"
        elif self.ref == REFERENCE_PRESSURE_AIR:
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

    def __init__(self, ref=REFERENCE_PRESSURE_WATER, **kwargs):
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
        freqs, Pxx = _sig.welch(data, fs, **self.welch_params)

        # Store frequencies and PSD values
        self.frequencies = freqs
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
        if not hasattr(self, "frequencies") or not hasattr(self, "psd"):
            raise ConfigurationError(
                "PSD.plot: compute() must be called before plotting")
        psd_db = power_to_db(self.psd, self.ref)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogx(self.frequencies, psd_db, label=label, **kwargs)

        # Customize plot appearance
        ax.set_title(f"[PSD] {title}", loc="left")
        ax.set_xlabel("Frequency [Hz]")
        if self.ref == REFERENCE_PRESSURE_WATER:
            ref = "1µ"
        elif self.ref == REFERENCE_PRESSURE_AIR:
            ref = "20µ"
        else:
            ref = f"{self.ref:02e}"
        ax.set_ylabel(f"Level [dB re {ref}Pa²/Hz]")
        ax.set_ylim((ymin, ymax))
        ax.set_xlim((np.max((self.frequencies[0], 1)), self.frequencies[-1]))
        ax.grid(which="both", alpha=0.75)
        plt.tight_layout()
        if label != "":
            ax.legend()

        return fig, ax

    def add_to_plot(self, ax, Fxx=None, Pxx=None, ref=None, label="", **kwargs):
        """Overlay a PSD curve on an existing axes (defaults to this instance's data)."""
        if Fxx is None and Pxx is None:
            Fxx = self.frequencies
            Pxx = self.psd
        if ref is None:
            ref = self.ref

        psd_db = power_to_db(Pxx, ref)
        ax.plot(Fxx, psd_db, label=label, **kwargs)
        if label != "":
            ax.legend()

        return ax
