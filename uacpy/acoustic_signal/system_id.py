"""Frequency Response Function (FRF) estimation: Welch / ETFE / LS-FIR."""

import numpy as np
import scipy.signal as _sig
from scipy.linalg import toeplitz

from uacpy.core.exceptions import ConfigurationError


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
        self.selected_order = None  # FIR order chosen by ls_fir order selection
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
            published on ``self.selected_order``.
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
            # Extract the i-th measurement
            x_i = x[i, :].ravel()
            y_i = y[i, :].ravel()
            if self.method == "welch":
                freqs_i, tf_i, coh_i = self.compute_welch(x_i, y_i, sample_rate)
                coh_list.append(coh_i)
            elif self.method == "ls_fir":
                freqs_i, tf_i, g_i = self.compute_lsfir(
                    y_i, x_i, sample_rate, self.m, len(x_i), m_max=m_max, stop_count=stop_count
                )
                m_list.append(len(g_i))
            elif self.method == "etfe":
                freqs_i, tf_i = self.compute_etfe(x_i, y_i, sample_rate)
            elif self.method == "p_etfe":
                freqs_i, tf_i = self.compute_periodic_etfe(x_i, y_i, sample_rate)
            else:
                raise ConfigurationError(
                    f"FRF.compute: unknown method={self.method!r}; "
                    "valid: 'welch', 'ls_fir', 'etfe', 'p_etfe'"
                )

            # Append results
            tf_list.append(tf_i)

        # Average results
        freqs = freqs_i
        tf = np.mean(tf_list, axis=0)

        # Update object state; every run rewrites the method-specific
        # attributes so a reused FRF cannot report a previous method's result.
        self.frequencies = freqs
        self.tf = tf
        self.coh = np.mean(coh_list, axis=0) if self.method == "welch" else None
        if self.method == "ls_fir":
            self.g = g_i  # For 2D inputs, uses last channel's impulse response
            self.selected_order = int(np.mean(m_list))
        else:
            self.g = 0
            self.selected_order = None

        return freqs, tf

    def compute_welch(self, x, y, sample_rate):
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
        sample_rate : float
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
        freqs, Pxx = _sig.welch(x, sample_rate, scaling="density", **self.params)
        _, Pyy = _sig.welch(y, sample_rate, scaling="density", **self.params)
        _, Pxy = _sig.csd(y, x, sample_rate, scaling="density", **self.params)
        if self.estimator == "H2":
            tf = Pyy / Pxy
        else:  # Default to H1
            tf = np.conj(Pxy) / Pxx
        coh = abs(Pxy) ** 2 / (Pxx * Pyy)

        return freqs, tf, coh

    def compute_periodic_etfe(self, x, y, sample_rate, nperseg=None):
        """
        Compute ETFE for periodic data.

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

        if nperseg:
            self.params["nperseg"] = nperseg

        # For periodic data, we compute at frequencies k*2*pi/period/Ts
        # up to the Nyquist frequency
        period = self.params["nperseg"]
        n_periods = len(x) // period

        if n_periods < 1:
            raise ConfigurationError(
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
        Y = np.fft.rfft(y_avg)
        freqs = np.fft.rfftfreq(period, d=1 / sample_rate)
        tf = Y / X

        return freqs, tf

    def compute_etfe(self, x, y, sample_rate):
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
        sample_rate : float
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
        freqs = np.fft.rfftfreq(n_fft, d=1 / sample_rate)
        tf = Y / X

        return freqs, tf

    def compute_lsfir(self, y, u, sample_rate, m, N, m_max=4096, stop_count=50, nperseg=None):
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
        sample_rate : float
            Sampling rate in Hz.
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
            # "Numerically exact fit" is judged against the output power, so the
            # decision is invariant to the amplitude scale of the data.
            exact_tol = np.finfo(float).eps * float(np.mean(y[:N] ** 2))

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

                    if sse <= exact_tol:
                        # Residual at the rounding floor: this order explains
                        # the data exactly, and it is the lowest one that does.
                        best_m = m_candidate
                        best_g = g
                        break

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

        # Frequency response on the same rfft grid the other methods return,
        # so an ls_fir result lines up bin-for-bin with a welch/etfe one.
        freqs = np.fft.rfftfreq(int(self.params["nperseg"]), d=1.0 / sample_rate)
        _, h = _sig.freqz(g, worN=freqs, fs=sample_rate)

        return freqs, h, g
