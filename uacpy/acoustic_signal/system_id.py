"""Frequency Response Function (FRF) estimation: Welch / ETFE / LS-FIR."""

import numpy as np
import scipy.signal as _sig
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from matplotlib.gridspec import GridSpec


class FRF:
    """Frequency Response Function (FRF) computation and visualization."""

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
        frequency domain. Under the scipy convention ``Sxy = csd(x, y)
        = E[X*·Y]`` it is defined as::

            H1(f) = Sxy(f) / Sxx(f)
            H2(f) = Syy(f) / Syx(f)

        where ``Sxx`` is the input PSD and ``Sxy`` the input/output CPSD.
        Internally the code stores ``csd(y, x) = Syx = conj(Sxy)`` in
        the variable named ``Pxy``; the H1 expression
        ``conj(Pxy)/Pxx`` recovers the textbook ``Sxy/Sxx``.
        """
        import numpy as np

        # Default parameters, overridden by kwargs if provided
        self.params = {
            "nperseg": 8192,
            "noverlap": 0,
        }
        self.params.update(kwargs)
        self.method = method
        self.estimator = estimator
        self.Minfo = np.array([[0]])
        self.Vinfo = np.array([[0]])
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
            Impulse response length (for TF methods), or an automatic
            order-selection criterion for ``'ls_fir'``: ``'AIC'``,
            ``'BIC'``, ``'FPE'``, or ``'CP'``.
        method : str, optional
            Method to use ('welch', 'ls_fir', 'etfe', 'p_etfe').
        estimator : str, optional
            Estimator for Welch method ('H1', 'H2').
        nperseg : int, optional
            Segment length for Welch.
        noverlap : int, optional
            Overlap for Welch.
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
        if m is not None:
            self.m = m
        if stop_count is None:
            # early-stop after 50 consecutive orders with no score improvement
            # (compute_lsfir's documented default); m_max is the hard order cap.
            stop_count = 50

        # Convert inputs to 2D arrays (rows = measurements)
        x = np.asarray(x)
        y = np.asarray(y)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"FRF.compute: x and y must have the same number of measurements; "
                f"got x.shape[0]={x.shape[0]}, y.shape[0]={y.shape[0]}"
            )
        n_meas = x.shape[0]
        m_list, tf_list, coh_list = [], [], []

        for i in range(n_meas):
            # Extract the i-th measurement
            x_i = x[i, :].ravel()
            y_i = y[i, :].ravel()
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
                raise ValueError(
                    f"FRF.compute: unknown method={self.method!r}; "
                    "valid: 'welch', 'ls_fir', 'etfe', 'p_etfe'"
                )

            # Append results
            tf_list.append(tf_i)

        # Average results
        freqs = freqs_i
        tf = np.mean(tf_list, axis=0)

        # Update object state
        self.frequencies = freqs
        self.tf = tf
        if self.method == "welch":
            self.coh = (
                np.mean(coh_list, axis=0)
                if all(c is not None for c in coh_list)
                else None
            )
        if self.method == "ls_fir":
            self.g = g_i  # For 2D inputs, uses last channel's impulse response
            self.m = (
                int(np.mean(m_list))
                if all(mi is not None for mi in m_list) else None
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
        freqs, Pxx = _sig.welch(x, fs, scaling="density", **self.params)
        _, Pyy = _sig.welch(y, fs, scaling="density", **self.params)
        _, Pxy = _sig.csd(y, x, fs, scaling="density", **self.params)
        if self.estimator == "H2":
            tf = Pyy / Pxy
        else:  # Default to H1
            tf = np.conj(Pxy) / Pxx
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
            raise ValueError(
                f"FRF.compute_periodic_etfe: signal length must be at least one "
                f"period; got len(x)={len(x)} samples, period={period} samples"
            )

        # Extract a whole number of periods
        x = x[: n_periods * period]
        y = y[: n_periods * period]

        # Reshape to n_periods rows of period columns
        x_reshaped = x.reshape(n_periods, period)
        y_reshaped = y.reshape(n_periods, period)

        # Average over periods to reduce noise
        x_avg = np.mean(x_reshaped, axis=0)
        y_avg = np.mean(y_reshaped, axis=0)
        X = np.fft.rfft(x_avg) + np.finfo(float).eps
        Y = np.fft.rfft(y_avg) + np.finfo(float).eps
        freqs = np.fft.rfftfreq(period, d=1 / fs)
        tf = np.zeros_like(X, dtype=complex)
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
        X = np.fft.rfft(x) + np.finfo(float).eps
        Y = np.fft.rfft(y)

        # Determine frequency grid based on n_freqs
        n_fft = min_len
        freqs = np.fft.rfftfreq(n_fft, d=1 / fs)
        tf = np.zeros_like(X, dtype=complex)
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

        y = np.array(y)
        u = np.array(u)

        if m in ["AIC", "FPE", "CP", "BIC"]:
            # Model order selection
            m_max = min(m_max, N - 1)
            best_score = np.inf
            best_m = 1
            best_g = None
            count = 0

            if m == "CP":
                # Mallows' Cp: σ̂² is the residual variance of a low-bias
                # reference fit (order well above any plausible true order,
                # well below N), not the raw output variance.
                full_model_m = min(m_max, max(2, N // 4))
                u_temp = u[:N].copy()
                phiuu_full = np.zeros(full_model_m)
                phiuy_full = np.zeros(full_model_m)
                for i in range(full_model_m):
                    phiuu_full[i] = np.dot(u[:N], u_temp)
                    phiuy_full[i] = np.dot(y[:N], u_temp)
                    u_temp = np.concatenate(([u_temp[-1]], u_temp[:-1]))
                A_full = toeplitz(phiuu_full)
                u_flipped = np.flip(u[:N]).copy()
                W_full = np.zeros((full_model_m - 1, full_model_m))
                for i in range(full_model_m - 1):
                    u_flipped = np.concatenate(([u_flipped[-1]], u_flipped[:-1]))
                    W_full[i, :] = u_flipped[:full_model_m]
                g_full = np.linalg.solve(
                    A_full - np.dot(W_full.T, W_full),
                    phiuy_full - np.dot(W_full.T, y[: full_model_m - 1]),
                )
                y_hat_full = np.convolve(u[:N], g_full, mode="full")[:N]
                sigma2 = np.sum((y[:N] - y_hat_full) ** 2) / (N - full_model_m)

            for m_candidate in range(1, m_max + 1):
                try:
                    u_temp = u[:N].copy()
                    phiuu = np.zeros(m_candidate)
                    phiuy = np.zeros(m_candidate)

                    for i in range(m_candidate):
                        phiuu[i] = np.dot(u[:N], u_temp)
                        phiuy[i] = np.dot(y[:N], u_temp)
                        u_temp = np.concatenate(
                            ([u_temp[-1]], u_temp[:-1])
                        )  # Shift right

                    A = toeplitz(phiuu)
                    u_flipped = np.flip(u[:N]).copy()
                    W = np.zeros((m_candidate - 1, m_candidate))

                    for i in range(m_candidate - 1):
                        u_flipped = np.concatenate(([u_flipped[-1]], u_flipped[:-1]))
                        W[i, :] = u_flipped[:m_candidate]

                    Minfo = A - np.dot(W.T, W)
                    Vinfo = phiuy - np.dot(W.T, y[: m_candidate - 1])

                    g = np.linalg.solve(Minfo, Vinfo)
                    y_hat = np.convolve(u[:N], g, mode="full")[:N]
                    residuals = y[:N] - y_hat
                    sse = np.sum(residuals**2) / (N - m_candidate)

                    if sse < 1e-9:
                        continue  # Avoid log issues

                    if m == "AIC":  # AICF
                        # Finite-sample AIC variant: log(sse) scaled by
                        # (1 + m/(N-m))/(1 - m/(N-m)), not the textbook
                        # log(sse) + 2m/N (they agree as N >> m).
                        score = np.log(sse) + (1 + m_candidate / (N - m_candidate)) / (
                            1 - m_candidate / (N - m_candidate)
                        )

                    elif m == "FPE":  # FPEF
                        # Finite-sample FPE: sse·(1 + m/(N-m))/(1 - m/(N-m))
                        # (textbook uses m/N; same N >> m limit).
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
                        score = np.log(sse) + (m_candidate * np.log(N)) / N

                    if score < best_score:
                        best_score = score
                        best_m = m_candidate
                        best_g = g
                        count = 0
                    else:
                        count += 1

                    if count >= stop_count:
                        break  # Stop search early

                except np.linalg.LinAlgError:
                    continue  # Skip singular matrices

            m = best_m

            # Recompute Minfo and Vinfo for the best m
            u_temp = u[:N].copy()
            phiuu = np.zeros(m)
            phiuy = np.zeros(m)

            for i in range(m):
                phiuu[i] = np.dot(u[:N], u_temp)
                phiuy[i] = np.dot(y[:N], u_temp)
                u_temp = np.concatenate(([u_temp[-1]], u_temp[:-1]))

            A = toeplitz(phiuu)
            u_flipped = np.flip(u[:N]).copy()
            W = np.zeros((m - 1, m))

            for i in range(m - 1):
                u_flipped = np.concatenate(([u_flipped[-1]], u_flipped[:-1]))
                W[i, :] = u_flipped[:m]

            self.Minfo = A - np.dot(W.T, W)
            self.Vinfo = phiuy - np.dot(W.T, y[: m - 1])
            g = best_g

        else:
            # Given m, compute directly
            u_temp = u[:N].copy()
            phiuu = np.zeros(m)
            phiuy = np.zeros(m)

            for i in range(m):
                phiuu[i] = np.dot(u[:N], u_temp)
                phiuy[i] = np.dot(y[:N], u_temp)
                u_temp = np.concatenate(([u_temp[-1]], u_temp[:-1]))

            A = toeplitz(phiuu)
            u_flipped = np.flip(u[:N]).copy()
            W = np.zeros((m - 1, m))

            for i in range(m - 1):
                u_flipped = np.concatenate(([u_flipped[-1]], u_flipped[:-1]))
                W[i, :] = u_flipped[:m]

            self.Minfo = A - np.dot(W.T, W)
            self.Vinfo = phiuy - np.dot(W.T, y[: m - 1])
            g = np.linalg.solve(self.Minfo, self.Vinfo)

        # Frequency response
        w_imp, h = _sig.freqz(g, worN=int(self.params["nperseg"] / 2 + 1))
        freqs = w_imp * fs / (2 * np.pi)

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
        fig = plt.figure(figsize=figsize)

        # Define a 2x2 grid with adjusted height ratios
        gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        im = ax1.imshow(self.Minfo, cmap="viridis", aspect="equal")
        ax1.set_title(f"[Information Matrix] {title}", loc="left")
        ax1.set_xlabel("Index j")
        ax1.set_ylabel("Index i")
        cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
        cbar.set_label("Correlation Value")
        ax2 = fig.add_subplot(gs[0, 1])
        indices = np.arange(len(self.Vinfo))
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
        ax.plot(self.frequencies, self.coh, label=label, **kwargs)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Coherence")
        ax.set_xscale("log")
        ax.set_ylim((0.75, 1.01))
        ax.set_xlim((np.max((self.frequencies[0], 1)), self.frequencies[-1]))
        ax.grid(which="major", alpha=0.75)
        ax.grid(which="minor", alpha=0.25)
        ax.tick_params(axis="x", which="both")
        ax.set_title(f"[Coherence] {title}", loc="left")

        if label != "":
            ax.legend()

        return fig, ax

    def add_coherence_to_plot(self, axes, title="", label="", **kwargs):
        """Overlay this instance's coherence curve on an existing axes."""
        ax = axes

        if label != "":
            addstr = f"[{self.method}-{self.estimator}] "
            label = addstr.upper() + label

        # Coherence plot
        ax.plot(self.frequencies, self.coh, label=label, **kwargs)

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

        if not hasattr(self, "frequencies") or not hasattr(self, "tf"):
            raise RuntimeError(
                "FRF.plot: compute() must be called before plotting"
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
        mag_db = 20 * np.log10(np.abs(self.tf))
        ax1.plot(self.frequencies, mag_db, label=label, **kwargs)
        ax1.set_ylabel("Magnitude [dB]")
        ax1.set_xscale("log")
        ax1.set_ylim((ymin, ymax))
        ax1.set_xlim((np.max((self.frequencies[0], 1)), self.frequencies[-1]))
        ax1.grid(which="major", alpha=0.75)
        ax1.grid(which="minor", alpha=0.25)
        ax1.set_xticklabels([])
        ax1.tick_params(axis="x", which="both", bottom=False)

        # Phase plot
        phase_deg = np.angle(self.tf, deg=True)
        ax2.plot(self.frequencies, phase_deg, label=label, **kwargs)
        ax2.set_ylabel("Phase [degrees]")
        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_xscale("log")
        ax2.set_ylim((-180, 180))
        ax2.set_xlim((np.max((self.frequencies[0], 1)), self.frequencies[-1]))
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
                self.frequencies, 20 * np.log10(np.abs(self.tf)), label=label, **kwargs
            )
        else:
            ax1.plot(freqs, 20 * np.log10(mag), label=label, **kwargs)

        if freqs is None or phase is None:
            ax2.plot(self.frequencies, np.angle(self.tf, deg=True), label=label, **kwargs)
        else:
            ax2.plot(freqs, phase, label=label, **kwargs)

        if label != "":
            ax1.legend()
            ax2.legend()

        return axes


