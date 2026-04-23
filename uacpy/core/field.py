"""
Field class for storing and manipulating acoustic field results
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple

from uacpy.core.constants import DEFAULT_SOUND_SPEED, PRESSURE_FLOOR


class Field:
    """
    Acoustic field results

    Stores results from acoustic propagation models including transmission loss,
    complex pressure, ray paths, arrivals, etc.

    Parameters
    ----------
    field_type : str
        Type of field: 'tl' (transmission loss), 'pressure', 'rays',
        'arrivals', 'modes'
    data : ndarray
        Main field data array
    ranges : ndarray, optional
        Range coordinates in meters
    depths : ndarray, optional
        Depth coordinates in meters
    frequencies : ndarray, optional
        Frequency coordinates in Hz
    metadata : dict, optional
        Additional metadata about the simulation

    Attributes
    ----------
    field_type : str
        Field type
    data : ndarray
        Field data
    ranges : ndarray
        Range grid
    depths : ndarray
        Depth grid
    frequencies : ndarray
        Frequencies
    metadata : dict
        Metadata dictionary

    Examples
    --------
    Access transmission loss field:

    >>> tl = field.data  # Shape: (n_depths, n_ranges) for 2D,
    ...                  # (n_depths, n_freqs, n_ranges) for broadband
    >>> tl_at_1km = field.get_at_range(1000)

    Get field at specific location:

    >>> value = field.get_value(range=5000, depth=50)
    """

    def __init__(
        self,
        field_type: str,
        data: np.ndarray,
        ranges: Optional[np.ndarray] = None,
        depths: Optional[np.ndarray] = None,
        frequencies: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.field_type = field_type.lower()
        self.data = data
        self.ranges = ranges if ranges is not None else np.array([0.0])
        self.depths = depths if depths is not None else np.array([0.0])
        self.frequencies = frequencies if frequencies is not None else np.array([0.0])
        self.metadata = metadata if metadata is not None else {}

        # Validate field type
        valid_types = ['tl', 'pressure', 'rays', 'arrivals', 'modes', 'eigenrays', 'reflection_coefficients', 'transfer_function', 'time_series']
        if self.field_type not in valid_types:
            raise ValueError(f"field_type must be one of {valid_types}")

    @property
    def shape(self) -> Tuple:
        """Shape of data array"""
        return self.data.shape

    @property
    def n_ranges(self) -> int:
        """Number of range points"""
        return len(self.ranges)

    @property
    def n_depths(self) -> int:
        """Number of depth points"""
        return len(self.depths)

    @property
    def n_frequencies(self) -> int:
        """Number of frequencies"""
        return len(self.frequencies)

    @property
    def ray_data(self):
        """Convenience property to access ray data from metadata"""
        return self.metadata.get('rays', None)

    @property
    def arrivals_data(self):
        """Convenience property to access arrivals data from metadata"""
        return self.metadata.get('arrivals', None)

    def get_value(self, range_m: float, depth: float,
                  frequency: Optional[float] = None) -> float:
        """
        Get field value at specified location

        Parameters
        ----------
        range_m : float
            Range in meters
        depth : float
            Depth in meters
        frequency : float, optional
            Frequency in Hz (for multi-frequency fields)

        Returns
        -------
        value : float
            Interpolated field value
        """
        if self.field_type in ['rays', 'arrivals', 'eigenrays', 'time_series']:
            raise ValueError(f"get_value not supported for field_type='{self.field_type}'")

        # Find nearest indices
        r_idx = np.argmin(np.abs(self.ranges - range_m))
        d_idx = np.argmin(np.abs(self.depths - depth))

        if len(self.data.shape) == 2:
            return self.data[d_idx, r_idx]
        elif len(self.data.shape) == 3 and frequency is not None:
            f_idx = np.argmin(np.abs(self.frequencies - frequency))
            # data shape is (n_depths, n_freqs, n_ranges) per class docstring
            return self.data[d_idx, f_idx, r_idx]
        else:
            raise ValueError("Frequency required for multi-frequency field")

    def get_at_range(self, range_m: float) -> np.ndarray:
        """
        Get field values at specified range

        Parameters
        ----------
        range_m : float
            Range in meters

        Returns
        -------
        values : ndarray
            Field values at all depths at this range.
            Shape: (n_depths,) for 2D fields; (n_depths, n_freqs) for 3D
            broadband fields.
        """
        r_idx = np.argmin(np.abs(self.ranges - range_m))

        if len(self.data.shape) == 2:
            return self.data[:, r_idx]
        elif len(self.data.shape) == 3:
            # data shape is (n_depths, n_freqs, n_ranges)
            return self.data[:, :, r_idx]
        else:
            raise ValueError("Unsupported data shape")

    def get_at_depth(self, depth: float) -> np.ndarray:
        """
        Get field values at specified depth

        Parameters
        ----------
        depth : float
            Depth in meters

        Returns
        -------
        values : ndarray
            Field values at all ranges at this depth.
            Shape: (n_ranges,) for 2D fields; (n_freqs, n_ranges) for 3D
            broadband fields.
        """
        d_idx = np.argmin(np.abs(self.depths - depth))

        if len(self.data.shape) == 2:
            return self.data[d_idx, :]
        elif len(self.data.shape) == 3:
            # data shape is (n_depths, n_freqs, n_ranges)
            return self.data[d_idx, :, :]
        else:
            raise ValueError("Unsupported data shape")

    def to_db(self) -> 'Field':
        """
        Convert to dB scale (if not already)

        Returns
        -------
        field_db : Field
            New field with data in dB
        """
        if self.field_type == 'tl':
            # Already in dB typically
            return self

        if self.field_type == 'pressure':
            # Convert complex pressure to dB with proper handling of shadow zones
            p_abs = np.abs(self.data)
            # Set floor to prevent extreme TL values in shadow zones
            p_abs = np.maximum(p_abs, PRESSURE_FLOOR)
            data_db = 20 * np.log10(p_abs)
            # Note: This is pressure level in dB, not TL
            # For TL (transmission loss), use negative: TL = -data_db
            # But this method is to_db() for general pressure-to-dB conversion
            return Field(
                field_type='tl',
                data=data_db,
                ranges=self.ranges,
                depths=self.depths,
                frequencies=self.frequencies,
                metadata=self.metadata.copy()
            )

        raise ValueError(f"to_db not supported for field_type='{self.field_type}'")

    def to_time_domain(
        self,
        depth: Optional[float] = None,
        range_m: Optional[float] = None,
        source_spectrum: Optional[np.ndarray] = None,
        window: str = 'hann',
        nfft: Optional[int] = None,
        t_start: Optional[float] = None,
    ) -> 'Field':
        """
        Convert transfer function to time-domain via IFFT.

        Extracts the transfer function at a specific (depth, range) and
        returns the time-domain impulse response or convolved signal.

        Parameters
        ----------
        depth : float, optional
            Receiver depth in meters. If None, uses the middle depth.
        range_m : float, optional
            Receiver range in meters. If None, uses the first range.
        source_spectrum : ndarray, optional
            Source spectrum S(f) to convolve with the transfer function.
            Shape: (n_freq,). If None, computes the impulse response.
        window : str, optional
            Window applied before IFFT to reduce spectral leakage.
            'hann' (default), 'hamming', 'blackman', 'tukey', or 'none'.
        nfft : int, optional
            FFT length (zero-padded). If None, uses next power of 2 above
            4 * n_freq for finer time resolution.
        t_start : float, optional
            Start time in seconds. If None, estimated from range and
            sound speed as t_start = r/c - margin.

        Returns
        -------
        field : Field
            time_series field with:
            - data: real pressure array, shape (nt,)
            - metadata['time']: time vector in seconds
            - metadata['dt']: time step
            - metadata['fs']: sampling rate
            - metadata['nt']: number of time samples
            - metadata['depth']: receiver depth
            - metadata['range']: receiver range

        Notes
        -----
        Implements frequency-aware IFFT following the mpiramS plotram.m
        convention and standard Fourier synthesis:

        1. Apply frequency window (Hann by default, matching plotram.m)
        2. For mpiramS output, conjugate(H) before IFFT (plotram.m convention)
        3. Optionally multiply by source spectrum
        4. Phase-rotate for time shift: H * exp(i*2*pi*f*t_start)
        5. Place spectrum at correct FFT bins (bin_k = round(f_k / df))
        6. IFFT → real part, scaled by 2 for one-sided spectrum

        The bin placement at step 5 is critical: the spectrum covers
        [f_min, f_max], NOT starting from DC. Each frequency must map
        to its correct FFT bin index. See plotram.m lines 94-98 which
        does an equivalent fftshift-like rearrangement.

        Examples
        --------
        >>> # Impulse response at 50 m depth, 5 km range
        >>> ts = transfer_func.to_time_domain(depth=50, range_m=5000)
        >>> plt.plot(ts.metadata['time'] * 1000, ts.data)

        >>> # With Gaussian source pulse
        >>> src = np.exp(-((freqs - 100)**2) / (2*20**2))
        >>> ts = transfer_func.to_time_domain(depth=50, range_m=5000,
        ...                                    source_spectrum=src)
        """
        if self.field_type != 'transfer_function':
            raise ValueError(
                f"to_time_domain() requires field_type='transfer_function', "
                f"got '{self.field_type}'"
            )

        data = self.data  # (n_depths, n_freq, n_ranges)
        freqs = self.frequencies
        n_depths, n_freq, n_ranges = data.shape

        if n_freq < 2:
            raise ValueError("Need at least 2 frequencies for IFFT")

        # Select depth/range indices
        if depth is not None:
            d_idx = int(np.argmin(np.abs(self.depths - depth)))
        else:
            d_idx = n_depths // 2
        if range_m is not None:
            r_idx = int(np.argmin(np.abs(self.ranges - range_m)))
        else:
            r_idx = 0

        actual_depth = float(self.depths[d_idx])
        actual_range = float(self.ranges[r_idx])

        # Extract single spectrum
        spectrum = data[d_idx, :, r_idx].copy()  # (n_freq,)

        # Replace NaN with 0 (depths outside PE domain, etc.)
        spectrum = np.nan_to_num(spectrum, nan=0.0)

        df_data = float(freqs[1] - freqs[0])

        # Use fine FFT bin spacing for adequate time window.
        # The IFFT time span is 1/df_fft. With coarse frequency data
        # (e.g., 32 points over 150 Hz → df=4.8 Hz → 207ms window),
        # the signal may fall outside the window. Cap df_fft at 1 Hz
        # to guarantee at least a 1-second time window.
        df = min(df_data, 1.0)

        # Compute FFT bin index for each frequency:
        # bin_k = round(f_k / df) maps each frequency to its correct
        # position in the FFT array (bin 0 = DC, bin k = k*df Hz)
        bin_indices = np.round(freqs / df).astype(int)
        max_bin = int(bin_indices[-1])

        # Check if this is mpiramS data (for time-window estimation)
        is_ram = 'Nsam' in self.metadata

        # FFT size: must cover highest frequency bin AND provide
        # enough time resolution
        if nfft is None:
            if is_ram:
                # Match mpiramS Nsam for consistency with plotram.m
                nfft_min = int(self.metadata.get('Nsam', 4 * n_freq))
            else:
                nfft_min = 4 * n_freq
            nfft = max(nfft_min, max_bin + 1)
            # Round up to power of 2 for FFT efficiency
            nfft_pow2 = 1
            while nfft_pow2 < nfft:
                nfft_pow2 *= 2
            nfft = nfft_pow2

        # Window function
        if window == 'hann':
            win = np.hanning(n_freq)
        elif window == 'hamming':
            win = np.hamming(n_freq)
        elif window == 'blackman':
            win = np.blackman(n_freq)
        elif window == 'tukey':
            from scipy.signal import windows
            win = windows.tukey(n_freq, alpha=0.5)
        elif window == 'none':
            win = np.ones(n_freq)
        else:
            raise ValueError(f"Unknown window: {window}")

        # Time parameters
        dt = 1.0 / (nfft * df)
        fs = 1.0 / dt

        # Estimate start time
        if t_start is None:
            if is_ram:
                # plotram.m: tdelay = rout/cmin - T - 0.5
                cmin = self.metadata.get('cmin', DEFAULT_SOUND_SPEED)
                T_window = self.metadata.get('Nsam', nfft) / self.metadata.get('fs', fs)
                t_start = max(0.0, actual_range / cmin - T_window - 0.5)
            else:
                c0 = self.metadata.get('c0', DEFAULT_SOUND_SPEED)
                t_start = max(0.0, actual_range / c0 - 2.0 / df)

        # RAM conjugation and pressure scaling are handled by each model's
        # _run_broadband() method before reaching this point.

        # Apply window and source spectrum
        spectrum = spectrum * win

        if source_spectrum is not None:
            spectrum = spectrum * np.asarray(source_spectrum)

        # Place spectrum at correct FFT bins (frequency-aware placement).
        # Each frequency f_k goes to bin round(f_k/df), NOT to bin k.
        #
        # When df_fft < df_data (i.e., we use finer FFT bins than the data
        # spacing), we must interpolate the spectrum to fill ALL bins in
        # the frequency range — not just the bins at data frequencies.
        # Leaving gaps creates periodic replicas (ghost echoes) in the
        # IFFT output at intervals of 1/df_data.
        padded = np.zeros(nfft, dtype=complex)
        min_bin = int(bin_indices[0])
        max_bin_fill = int(bin_indices[-1])

        if df < df_data * 0.99 and n_freq >= 4:
            # Interpolation of a complex spectrum with rapid phase variation
            # (e.g., 100+ rad between adjacent samples) requires removing
            # the dominant group delay first. After demodulation, the
            # residual phase varies slowly and interpolates well.
            c0 = self.metadata.get('c0', DEFAULT_SOUND_SPEED)
            t_demod = actual_range / c0  # estimated group delay
            demod = np.exp(1j * 2.0 * np.pi * freqs * t_demod)
            spec_demod = spectrum * demod  # remove propagation delay

            from scipy.interpolate import interp1d
            fill_bins = np.arange(min_bin, min(max_bin_fill + 1, nfft))
            fill_freqs = fill_bins * df

            interp_re = interp1d(freqs, spec_demod.real, kind='linear',
                                 bounds_error=False, fill_value=0.0)
            interp_im = interp1d(freqs, spec_demod.imag, kind='linear',
                                 bounds_error=False, fill_value=0.0)
            spec_interp = interp_re(fill_freqs) + 1j * interp_im(fill_freqs)

            # Remodulate: add back propagation delay and apply t_start shift
            remod = np.exp(1j * 2.0 * np.pi * fill_freqs * (t_start - t_demod))
            padded[fill_bins] = spec_interp * remod
        else:
            # Dense enough — no interpolation needed
            spectrum = spectrum * np.exp(1j * 2.0 * np.pi * freqs * t_start)
            valid = (bin_indices >= 0) & (bin_indices < nfft)
            padded[bin_indices[valid]] = spectrum[valid]

        # IFFT → real part, factor 2 for one-sided spectrum
        # (only positive frequency bins are filled; the negative-frequency
        # conjugate is implicit in taking Re())
        result = 2.0 * np.real(np.fft.ifft(padded))

        # Time vector
        time = t_start + np.arange(nfft) * dt

        return Field(
            field_type='time_series',
            data=result,
            ranges=np.array([actual_range]),
            depths=np.array([actual_depth]),
            metadata={
                'time': time,
                'dt': dt,
                'fs': fs,
                'nt': nfft,
                't_start': t_start,
                'depth': actual_depth,
                'range': actual_range,
                'window': window,
                'source_model': self.metadata.get('model', 'unknown'),
            },
        )

    def get_max(self) -> Tuple[float, float, float]:
        """
        Get location and value of maximum field

        Returns
        -------
        max_value : float
            Maximum field value
        max_range : float
            Range of maximum
        max_depth : float
            Depth of maximum
        """
        if self.field_type in ['rays', 'arrivals', 'eigenrays', 'time_series']:
            raise ValueError(f"get_max not supported for field_type='{self.field_type}'")

        if len(self.data.shape) == 2:
            d_idx, r_idx = np.unravel_index(np.argmax(self.data), self.data.shape)
            return self.data[d_idx, r_idx], self.ranges[r_idx], self.depths[d_idx]
        else:
            raise ValueError("get_max only supported for 2D fields")

    def extract_rays(self) -> Dict[str, np.ndarray]:
        """
        Extract ray path data (for ray-type fields)

        Returns
        -------
        rays : dict
            Dictionary with ray data. Keys depend on model output.
        """
        if self.field_type not in ['rays', 'eigenrays']:
            raise ValueError(f"extract_rays only for ray-type fields")

        return self.metadata.get('rays', {})

    def extract_arrivals(self) -> Dict[str, np.ndarray]:
        """
        Extract arrival data (for arrivals-type fields)

        Returns
        -------
        arrivals : dict
            Dictionary with arrival data (times, amplitudes, angles, etc.)
        """
        if self.field_type != 'arrivals':
            raise ValueError("extract_arrivals only for arrivals-type fields")

        return self.metadata.get('arrivals', {})

    def __repr__(self) -> str:
        shape_str = 'x'.join(map(str, self.shape))
        return (f"Field(type='{self.field_type}', shape={shape_str}, "
                f"{self.n_ranges} ranges, {self.n_depths} depths)")

    def copy(self):
        """Create a deep copy of the field"""
        return Field(
            field_type=self.field_type,
            data=self.data.copy(),
            ranges=self.ranges.copy(),
            depths=self.depths.copy(),
            frequencies=self.frequencies.copy(),
            metadata=self.metadata.copy(),
        )

    def plot(self, env=None, **kwargs):
        """
        Plot the field (automatically detects type and calls appropriate function)

        This is the simple, unified plotting interface. The method automatically
        detects the field type and calls the appropriate plotting function.

        Parameters
        ----------
        env : Environment, optional
            Environment for bathymetry overlay (recommended for TL plots)
        **kwargs
            Additional plotting parameters passed to underlying plot function:

            For TL fields:
                - vmin, vmax : float - Color scale limits in dB
                - cmap : str - Colormap name (default: 'viridis')
                - figsize : tuple - Figure size (default: (12, 6))
                - show_bathymetry : bool - Show bathymetry overlay (default: True)
                - ax : Axes - Matplotlib axes to plot on

            For ray fields:
                - max_rays : int - Maximum number of rays to plot
                - figsize : tuple - Figure size (default: (12, 6))
                - ax : Axes - Matplotlib axes

            For mode fields:
                - n_modes : int - Number of modes to plot (default: 6)
                - figsize : tuple - Figure size (default: (14, 6))

            For arrival fields:
                - figsize : tuple - Figure size (default: (10, 6))
                - ax : Axes - Matplotlib axes

        Returns
        -------
        fig : Figure
            Matplotlib figure
        ax : Axes or tuple of Axes
            Matplotlib axes

        Examples
        --------
        >>> # Simple TL plot
        >>> result = bellhop.compute_tl(env, source, max_range=10000)
        >>> fig, ax = result.plot(env=env)
        >>> plt.show()

        >>> # Ray plot
        >>> rays = bellhop.compute_rays(env, source)
        >>> fig, ax = rays.plot(env=env, max_rays=50)
        >>> plt.show()

        >>> # Mode plot
        >>> modes = kraken.compute_modes(env, source, n_modes=20)
        >>> fig, axes = modes.plot(n_modes=10)
        >>> plt.show()

        >>> # Custom styling
        >>> fig, ax = result.plot(env=env, vmin=40, vmax=100, cmap='jet')
        >>> plt.show()

        Notes
        -----
        This method provides a simple, unified interface to plotting. For more
        control, you can still use the dedicated plotting functions from
        uacpy.visualization.plots directly.
        """
        # Import here to avoid circular dependency
        from uacpy.visualization import plots

        if self.field_type == 'tl':
            fig, ax, _cbar = plots.plot_transmission_loss(self, env=env, **kwargs)
            return fig, ax

        elif self.field_type in ['rays', 'eigenrays']:
            return plots.plot_rays(self, env=env, **kwargs)

        elif self.field_type == 'arrivals':
            return plots.plot_arrivals(self, **kwargs)

        elif self.field_type == 'modes':
            # plot_modes() now accepts Field objects directly
            return plots.plot_modes(self, **kwargs)

        elif self.field_type == 'pressure':
            # Convert to TL first
            field_tl = self.to_db()
            return plots.plot_transmission_loss(field_tl, env=env, **kwargs)

        elif self.field_type == 'transfer_function':
            return plots.plot_transfer_function(self, **kwargs)

        elif self.field_type == 'time_series':
            return plots.plot_time_series(self, **kwargs)

        else:
            raise ValueError(f"plot() not supported for field_type='{self.field_type}'")

    @staticmethod
    def plot_comparison(results, env=None, **kwargs):
        """
        Compare multiple model results in a single plot

        Parameters
        ----------
        results : dict
            Dictionary with model names as keys and Field objects as values.
            Example: {'Bellhop': field1, 'RAM': field2, 'KrakenField': field3}
        env : Environment, optional
            Environment for bathymetry overlay
        **kwargs
            Additional parameters:
                - vmin, vmax : float - TL color scale limits
                - figsize : tuple - Figure size
                - depth : float - Depth for range cuts (for plot_compare_range_cuts)

        Returns
        -------
        fig : Figure
            Matplotlib figure
        axes : array of Axes
            Matplotlib axes array

        Examples
        --------
        >>> # Run multiple models
        >>> results = {
        ...     'Bellhop': bellhop.compute_tl(env, source, max_range=10000),
        ...     'RAM': ram.compute_tl(env, source, max_range=10000),
        ...     'KrakenField': krakenfield.compute_tl(env, source, max_range=10000),
        ... }

        >>> # Simple comparison plot
        >>> fig, axes = Field.plot_comparison(results, env=env)
        >>> plt.show()

        >>> # With custom styling
        >>> fig, axes = Field.plot_comparison(results, env=env, vmin=40, vmax=100)
        >>> plt.show()

        >>> # Range cut comparison at specific depth
        >>> from uacpy.visualization.plots import compare_range_cuts
        >>> fig, ax = compare_range_cuts(results, depth=50.0)
        >>> plt.show()

        Notes
        -----
        This is a convenience method for comparing multiple model results.
        For more detailed comparisons, use the comparison functions from
        uacpy.visualization.plots directly (compare_models, compare_range_cuts,
        plot_model_statistics, etc.)
        """
        # Import here to avoid circular dependency
        from uacpy.visualization import plots

        # Use the appropriate comparison function
        if 'depth' in kwargs:
            # Range cut comparison at specific depth
            return plots.compare_range_cuts(results, **kwargs)
        else:
            # Full field comparison
            return plots.compare_models(results, env=env, **kwargs)
