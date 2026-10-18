"""Frequency-response estimation in ``uacpy.acoustic_signal.frf``: the
functions ``frf_welch``, ``etfe``, ``periodic_etfe`` and ``lsfir``, the
``FRFResult`` they return, and the ``FRF`` configuration that runs them.

* **Which estimator ran.** ``H1``/``H2``/``ls_fir`` answer differently on the
  same record, and the result carries the method, the selected model order
  and the conditioning of the fit rather than only the response.
* **Per-call keywords stay per call.** A keyword passed to one
  ``FRF.compute`` does not survive into the next, so two calls on one object
  are two independent fits, and ``FRF`` keeps no result.
* **A singular fit is reported, not smoothed over.** ``ls_fir`` falls back to
  the minimum-norm solution exactly at the reciprocal-condition floor, and
  both sides of that threshold are reached by construction.
* **Refusals.** A record whose rate, axis or shape the fit cannot use is
  refused by name.

The waveform generators and the package-wide entry-point guards live in
``test_signal.py``; the channel this response describes is in
``test_channel_response.py``.
"""

import warnings

import numpy as np
import pytest

from uacpy.acoustic_signal.frf import (
    FRF, FRFResult, etfe, frf_welch, lsfir, periodic_etfe,
)
from uacpy.core.exceptions import ConfigurationError


class TestFRF:
    """FRF automatic FIR-order selection (m='AIC'|'BIC'|'FPE'|'CP') must run,
    not crash with 'count >= None' from an un-defaulted stop_count."""

    @pytest.mark.parametrize("criterion", ['AIC', 'BIC', 'FPE', 'CP'])
    def test_auto_order_runs_and_recovers_order(self, criterion):
        from uacpy.acoustic_signal.frf import FRF
        rng = np.random.default_rng(1)
        u = rng.standard_normal(2000)
        g = np.array([1.0, -0.5, 0.25])                  # order-3 FIR
        y = np.convolve(u, g)[:u.size] + 0.01 * rng.standard_normal(2000)
        frf = FRF(method='ls_fir', order=criterion)
        result = frf.compute(u, y, 1000.0)
        assert np.isfinite(result.transfer_function).all()
        # every criterion recovers the true order-3 FIR at this SNR; the
        # chosen order is on the result, and the object keeps the criterion
        # the constructor set.
        assert result.order == 3 and result.criterion == criterion
        assert frf.order == criterion

    def test_cp_recovers_order_six_fir(self):
        """Mallows' Cp recovers the true order-6 FIR at moderate SNR. Cp scales
        the residual sum of squares by σ̂², the residual variance of a low-bias
        reference fit; this higher-order case exercises that estimate (order 3
        at high SNR above is too easy to constrain it).
        """
        r = np.random.default_rng(2)
        N, order = 3000, 6
        u = r.standard_normal(N)
        g = r.standard_normal(order)
        g = g / np.linalg.norm(g)
        clean = np.convolve(u, g)[:N]
        y = clean + 0.1 * np.std(clean) * r.standard_normal(N)
        fit = lsfir(u, y, 1000.0, order='CP')
        assert np.isfinite(fit.transfer_function).all()
        assert fit.order == order

    @pytest.mark.parametrize("criterion", ['AIC', 'BIC', 'FPE', 'CP'])
    def test_order_selection_is_amplitude_scale_invariant(self, criterion):
        """The selected order must depend on the data, not on its units: a
        pressure record in Pa and the same record in MPa must give the same
        FIR order. All four criteria compare log(sse) or sse ratios, so only
        the exact-fit cutoff can break the invariance."""
        rng = np.random.default_rng(0)
        u = rng.standard_normal(400)
        g = np.array([1.0, 0.5, -0.3])
        y = np.convolve(u, g)[:u.size] + 0.01 * rng.standard_normal(400)
        orders = []
        for scale in (1.0, 1e-3, 1e-6, 1e-9):
            orders.append(lsfir(scale * u, scale * y, 1000.0,
                                order=criterion, max_order=60).order)
        assert orders == [3, 3, 3, 3]

    @pytest.mark.parametrize("criterion", ['AIC', 'BIC', 'FPE', 'CP'])
    def test_exact_fit_selects_lowest_explaining_order(self, criterion):
        """A pure-gain loopback y = 2*u is fitted exactly at order 1; the
        search must return that order instead of discarding every candidate."""
        rng = np.random.default_rng(3)
        u = rng.standard_normal(400)
        fit = lsfir(u, 2.0 * u, 1000.0, order=criterion, max_order=60)
        assert np.isfinite(fit.transfer_function).all()
        assert fit.order == 1
        assert fit.impulse_response == pytest.approx([2.0])

    def test_unfittable_input_raises_configurationerror(self):
        """An all-zero input is singular at every order: typed error, not a
        ValueError out of scipy.signal.freqz on a None filter."""
        from uacpy.acoustic_signal.frf import FRF
        from uacpy.core.exceptions import ConfigurationError
        rng = np.random.default_rng(4)
        y = rng.standard_normal(300)
        with pytest.raises(ConfigurationError, match='no FIR order in'):
            FRF(method='ls_fir', order='AIC', max_order=40).compute(
                np.zeros(300), y, 1000.0)

    def test_zero_row_input_raises_configurationerror(self):
        """A 2-D input with no measurement rows must not fall through the
        per-measurement loop and hit an UnboundLocalError on the frequency
        axis."""
        from uacpy.acoustic_signal.frf import FRF
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match='hold no measurements'):
            FRF(method='ls_fir', order=4).compute(
                np.zeros((0, 100)), np.zeros((0, 100)), 1000.0)

    def test_each_result_carries_only_its_own_methods_values(self):
        """The order and impulse response are ls_fir-only and the coherence
        welch-only: each result holds its own method's values, and ``FRF``
        keeps none of them between runs."""
        rng = np.random.default_rng(11)
        u = rng.standard_normal(4096)
        y = np.convolve(u, [1.0, 0.5, -0.3])[:u.size]
        frf = FRF(method='ls_fir', order='AIC', max_order=40)
        fit = frf.compute(u, y, 1000.0)
        assert fit.method == 'ls_fir' and fit.order is not None
        assert fit.coherence is None and fit.estimator is None
        welch = FRF(nperseg=512).compute(u, y, 1000.0)
        assert welch.method == 'welch' and welch.estimator == 'H1'
        assert welch.order is None and welch.impulse_response is None
        assert welch.coherence.shape == welch.frequencies.shape
        for stale in ('tf', 'coh', 'g', 'selected_order', 'info_rcond',
                      'frequencies', 'Minfo', 'Vinfo'):
            assert not hasattr(frf, stale)


class TestFRFEstimators:
    """H1 and H2 differ only in which noise they reject (Bendat & Piersol):
    ``H1 = Sxy/Sxx`` is unbiased when the noise is on the output, ``H2 =
    Syy/Syx`` when it is on the input. Both recover the plant when there is no
    noise at all."""

    FS, N = 8000.0, 200_000
    H = np.array([1.0, -0.7, 0.35, -0.1])

    @classmethod
    def _truth(cls, f):
        n = np.arange(cls.H.size)
        return (cls.H[None, :] * np.exp(-2j * np.pi * np.outer(f, n) / cls.FS)
                ).sum(axis=1)

    @classmethod
    def _err(cls, estimator, x, y):
        result = frf_welch(x, y, cls.FS, estimator=estimator, nperseg=4096)
        f, tf, coh = result.frequencies, result.transfer_function, \
            result.coherence
        band = (f > 100) & (f < 3500)
        return (float(np.abs(tf[band] - cls._truth(f)[band]).max()),
                float(coh[band].mean()))

    def _signals(self, seed):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(self.N)
        return rng, x, np.convolve(x, self.H)[: self.N]

    def test_both_recover_the_plant_without_noise(self):
        # Noise-free, so the residual is Welch segmentation/leakage only:
        # measured max |tf - truth| is 4.6e-4 and the coherence is 1 - 4e-7.
        # 1e-2 / 1e-3 are floors an order of magnitude above that.
        _, x, y = self._signals(5)
        for est in ('H1', 'H2'):
            err, coh = self._err(est, x, y)
            assert err < 1e-2 and coh == pytest.approx(1.0, abs=1e-3)

    def test_h1_beats_h2_on_output_noise(self):
        # H2 is biased UP by output noise: measured errors are H1 0.21 vs
        # H2 0.91, a factor 4.3, so the required factor 3 leaves ~40 % margin
        # on this seed.
        rng, x, y = self._signals(6)
        yn = y + 0.5 * rng.standard_normal(self.N)
        assert self._err('H1', x, yn)[0] < self._err('H2', x, yn)[0] / 3

    def test_h2_beats_h1_on_input_noise(self):
        # The mirror case is weaker: measured H1 0.61 vs H2 0.39, a factor
        # 1.57 against the required 1.5 — only ~5 % margin, so this assertion
        # is seed-sensitive and the factor cannot be tightened.
        rng, x, y = self._signals(7)
        xn = x + 0.5 * rng.standard_normal(self.N)
        assert self._err('H2', xn, y)[0] < self._err('H1', xn, y)[0] / 1.5

    @pytest.mark.parametrize('bad', ['h2', 'bogus', None, 2])
    def test_an_unknown_estimator_is_refused_not_run_as_h1(self, bad):
        """``'h2'`` or a typo is refused rather than run as H1, whose
        estimate is biased by input noise the caller asked H2 to reject."""
        x = np.random.default_rng(0).standard_normal(4096)
        with pytest.raises(ConfigurationError, match="FRF: unknown estimator"):
            FRF(estimator=bad)
        with pytest.raises(ConfigurationError,
                           match="frf_welch: unknown estimator"):
            frf_welch(x, x, 1000.0, estimator=bad)


class TestFRFReservedKwargs:
    """Welch options that the FRF sets internally are rejected typed, not
    left to die in scipy as a bare TypeError."""

    def test_scaling_and_fs_raise_configurationerror(self):
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.acoustic_signal.frf import FRF
        with pytest.raises(ConfigurationError, match="scaling"):
            FRF(method="welch", scaling="spectrum")
        with pytest.raises(ConfigurationError, match="fs"):
            FRF(method="welch", fs=48_000.0)
        x = np.ones(64)
        with pytest.raises(ConfigurationError, match="frf_welch.*scaling"):
            frf_welch(x, x, 1000.0, scaling="spectrum")

    def test_legitimate_welch_kwargs_pass_through(self):
        from uacpy.acoustic_signal.frf import FRF
        rng = np.random.default_rng(2)
        x = rng.standard_normal(8192)
        y = np.convolve(x, [1.0, 0.5], mode="same")
        freqs, tf = FRF(method="welch", nperseg=1024,
                        window="hamming").compute(x, y, 8000.0)
        assert freqs.size == 513 and np.all(np.isfinite(tf))

    def test_etfe_grid_is_the_full_record_grid(self):
        from uacpy.acoustic_signal.frf import FRF
        rng = np.random.default_rng(3)
        x = rng.standard_normal(16384)
        y = np.convolve(x, [1.0, 0.5], mode="same")
        f_etfe, _ = FRF(method="etfe").compute(x, y, 8000.0)
        f_welch, _ = FRF(method="welch").compute(x, y, 8000.0)
        np.testing.assert_allclose(
            f_etfe, np.fft.rfftfreq(x.size, d=1 / 8000.0))
        assert f_welch.size == 8192 // 2 + 1      # the nperseg grid


class TestLsFirCpReferenceFitReportsDegenerateInput:
    N = 256

    def _degenerate(self, criterion):
        return lsfir(np.ones(self.N), np.zeros(self.N), 1000.0,
                     order=criterion)

    def test_cp_raises_the_typed_error_the_other_criteria_reach(self):
        # The Cp reference fit runs before the candidate loop's LinAlgError
        # handling, so a constant input escaped as a raw LinAlgError.
        with pytest.raises(ConfigurationError, match="criterion 'CP'"):
            self._degenerate('CP')

    @pytest.mark.parametrize("criterion", ['AIC', 'BIC', 'FPE'])
    def test_the_other_criteria_return_a_non_empty_result_on_the_same_input(self, criterion):
        assert self._degenerate(criterion).frequencies.size > 0


class TestLsFirSolvesSingularNormalEquationsByMinimumNorm:
    """``lsfir`` fits through ``X.T @ X``, whose condition number is
    ``cond(X)**2``, so a probe that leaves part of the Nyquist band unexcited
    makes the system numerically singular at any FIR order longer than the
    excited band supports. ``FRF``'s default ``order=512`` reaches it on an
    ordinary 100 Hz - 20 kHz sweep at fs = 48 kHz (``cond(X) = 4.2e11``,
    reciprocal condition number of the information matrix 1e-20): the LU
    solve of that system carries no correct digit, and the coefficients come
    back from a rank-revealing least-squares solve of the same equations
    instead, with the order named in a warning.
    """

    fs = 48000.0
    N = 8000
    order = 512
    h_true = np.array([1.0, -0.6, 0.3, 0.1])

    def _fit(self):
        import scipy.signal as sig
        rng = np.random.default_rng(11)
        u = sig.chirp(np.arange(self.N) / self.fs, 100.0,
                      self.N / self.fs, 20000.0)
        y = (np.convolve(u, self.h_true)[:self.N]
             + 1e-8 * rng.standard_normal(self.N))
        return lsfir(u, y, self.fs, order=self.order, nperseg=2048)

    def _in_band_dB_error(self, freqs, h):
        import scipy.signal as sig
        _, ht = sig.freqz(self.h_true, worN=freqs, fs=self.fs)
        band = (freqs >= 100.0) & (freqs <= 20000.0)
        return float(np.max(np.abs(20 * np.log10(np.abs(h[band]))
                                   - 20 * np.log10(np.abs(ht[band])))))

    def test_the_impulse_response_keeps_the_scale_of_the_channel_it_fits(self):
        # The true channel peaks at 1.0; the LU solve of the same equations
        # returns a peak of 16.9 on this record. The warning is pinned on its
        # own below, so this asserts the coefficients and nothing else.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            g = self._fit().impulse_response
        assert np.max(np.abs(g)) == pytest.approx(1.0, abs=0.05)

    def test_the_frequency_response_matches_the_channel_across_the_swept_band(self):
        # The LU solve of the same equations is 46.6 dB out here.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            freqs, h = self._fit()
        assert self._in_band_dB_error(freqs, h) < 0.05

    def test_the_warning_names_the_order_and_the_condition_estimate(self):
        with pytest.warns(UserWarning, match=r"FIR order 512 is numerically "
                                             r"singular \(reciprocal condition "
                                             r"number "):
            fit = self._fit()
        assert fit.rcond < np.finfo(float).eps

    #: Agreement demanded between the LU branch and a direct
    #: ``np.linalg.solve`` on a well-conditioned system, as a multiple of eps
    #: times the peak coefficient. numpy and scipy ship separate OpenBLAS
    #: builds, so the two run the same LAPACK algorithm from different
    #: binaries and bit-equality is a property of one machine's pairing, not
    #: of this code: measured over 60 well-conditioned solves it holds in 30
    #: and the worst disagreement is 2.75 eps of the peak. The fixtures below
    #: keep ``rcond`` above 0.05, where the LU's own backward error bounds the
    #: disagreement at roughly 20 eps, so this leaves 6x over the theory and
    #: 46x over the measurement.
    LU_AGREEMENT_EPS = 128.0

    @pytest.mark.parametrize("order", [64, 128, 256])
    def test_a_white_probe_takes_the_lu_branch_and_warns_about_nothing(self, order):
        rng = np.random.default_rng(11)
        u = rng.standard_normal(self.N)
        y = (np.convolve(u, self.h_true)[:self.N]
             + 1e-8 * rng.standard_normal(self.N))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            fit = lsfir(u, y, self.fs, order=order, nperseg=2048)
        g = fit.impulse_response
        assert fit.rcond > 0.05
        # A well-conditioned fit is the LU solution of the normal equations,
        # to the tolerance two builds of the same LAPACK routine can differ by.
        g_lu = np.linalg.solve(fit.information_matrix, fit.information_vector)
        tol = self.LU_AGREEMENT_EPS * np.finfo(float).eps * np.max(np.abs(g_lu))
        assert np.max(np.abs(g - g_lu)) <= tol


class TestLsFirInfoRcondFloorBoundary:
    """``_solve_info_matrices`` switches to the minimum-norm solution exactly
    at ``rcond <= _INFO_RCOND_FLOOR``. The fixtures are diagonal, where
    LAPACK's 1-norm reciprocal condition estimate is the smallest diagonal
    entry exactly, so the two sides of the threshold are reached by
    construction rather than by a fit that happens to land there.
    """

    n = 8

    def _system(self, delta):
        d = np.ones(self.n)
        d[-1] = delta
        return np.diag(d), np.ones(self.n)

    def test_above_the_floor_the_lu_coefficients_are_returned(self):
        from uacpy.acoustic_signal.frf import (
            _INFO_RCOND_FLOOR, _solve_info_matrices,
        )
        delta = 2.0 * _INFO_RCOND_FLOOR
        minfo, vinfo = self._system(delta)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            g, rcond = _solve_info_matrices(minfo, vinfo, self.n)
        assert rcond == pytest.approx(delta, rel=1e-12)
        assert rcond > _INFO_RCOND_FLOOR
        assert g[-1] == pytest.approx(1.0 / delta, rel=1e-12)

    def test_at_the_floor_the_ill_conditioned_direction_is_dropped(self):
        from uacpy.acoustic_signal.frf import (
            _INFO_RCOND_FLOOR, _solve_info_matrices,
        )
        delta = _INFO_RCOND_FLOOR
        minfo, vinfo = self._system(delta)
        with pytest.warns(UserWarning, match="numerically singular"):
            g, rcond = _solve_info_matrices(minfo, vinfo, self.n)
        assert rcond == pytest.approx(delta, rel=1e-12)
        assert rcond <= _INFO_RCOND_FLOOR
        assert g[-1] == 0.0
        # The directions the product can still represent are untouched.
        assert g[:-1] == pytest.approx(np.ones(self.n - 1))

    def test_below_the_floor_the_ill_conditioned_direction_is_dropped(self):
        from uacpy.acoustic_signal.frf import (
            _INFO_RCOND_FLOOR, _solve_info_matrices,
        )
        minfo, vinfo = self._system(0.5 * _INFO_RCOND_FLOOR)
        with pytest.warns(UserWarning, match="numerically singular"):
            g, rcond = _solve_info_matrices(minfo, vinfo, self.n)
        assert rcond < _INFO_RCOND_FLOOR
        assert g[-1] == 0.0

    @pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
    def test_the_branch_does_not_move_with_the_amplitude_scale(self, scale):
        """A record in Pa and the same record in µPa must be fitted the same
        way: the threshold is on a reciprocal condition number, which both
        norms scale out of."""
        from uacpy.acoustic_signal.frf import (
            _INFO_RCOND_FLOOR, _solve_info_matrices,
        )
        minfo, vinfo = self._system(2.0 * _INFO_RCOND_FLOOR)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _, rcond = _solve_info_matrices(scale * minfo, scale * vinfo,
                                            self.n)
        assert rcond == pytest.approx(2.0 * _INFO_RCOND_FLOOR, rel=1e-12)

    def test_the_floor_sits_where_the_lu_error_bound_reaches_the_answer(self):
        """``cond(Minfo) * eps >= 1`` is the point at which the LU solution's
        error bound is the size of the answer itself, so the floor is
        float64's eps and not a tuned constant. The fixtures above move with
        the constant; this pins where the constant is."""
        from uacpy.acoustic_signal.frf import _INFO_RCOND_FLOOR
        assert _INFO_RCOND_FLOOR == np.finfo(float).eps

    def test_an_exactly_singular_system_raises_the_error_the_order_search_skips_on(self):
        from uacpy.acoustic_signal.frf import _solve_info_matrices
        with pytest.raises(np.linalg.LinAlgError):
            _solve_info_matrices(np.zeros((4, 4)), np.zeros(4), 4)


class TestAnLsFirResultCarriesTheConditioningOfItsFit:
    def test_rcond_is_on_an_ls_fir_result_and_none_on_a_welch_one(self):
        rng = np.random.default_rng(11)
        u = rng.standard_normal(4096)
        y = np.convolve(u, [1.0, 0.5, -0.3])[:u.size]
        assert 0.0 < FRF(method='ls_fir', order=8).compute(
            u, y, 1000.0).rcond <= 1.0
        assert FRF(nperseg=512).compute(u, y, 1000.0).rcond is None


class TestTheOrderOnAnLsFirResult:
    def _fir_records(self, rows, seed=1):
        rng = np.random.default_rng(seed)
        u = rng.standard_normal((rows, 600))
        g = np.array([1.0, -0.5, 0.25])
        y = np.stack([np.convolve(u[i], g)[:600] for i in range(rows)])
        return u, y + 0.01 * rng.standard_normal(y.shape)

    def test_a_block_carries_one_fit_per_row(self):
        """Fits of different orders have no mean, so a 2-D block's result
        holds each row's order, impulse response and conditioning, and each
        row's fit is the one ``lsfir`` returns for that row alone."""
        u, y = self._fir_records(rows=3)
        result = FRF(method="ls_fir", order="AIC", max_order=40).compute(
            u, y, 1000.0)
        assert result.order == (3, 3, 3) and result.criterion == "AIC"
        assert len(result.impulse_response) == len(result.rcond) == 3
        rows = [lsfir(u[i], y[i], 1000.0, order="AIC", max_order=40)
                for i in range(3)]
        for row, g in zip(rows, result.impulse_response):
            np.testing.assert_array_equal(g, row.impulse_response)
        np.testing.assert_array_equal(
            result.transfer_function,
            np.mean([row.transfer_function for row in rows], axis=0))

    def test_a_criterion_on_one_record_gives_an_int_order(self):
        u, y = self._fir_records(rows=1)
        fit = FRF(method="ls_fir", order="BIC", max_order=40).compute(
            u[0], y[0], 1000.0)
        assert isinstance(fit.order, int)
        assert fit.order == 3 and fit.criterion == "BIC"

    def test_an_explicit_order_is_the_order_with_no_criterion(self):
        u, y = self._fir_records(rows=1)
        fit = lsfir(u[0], y[0], 1000.0, order=5)
        assert fit.order == 5 and fit.criterion is None
        assert len(fit.impulse_response) == 5


class TestFRFComputeTakesTheRecordsOnly:
    """Every setting is the constructor's: ``compute(x, y, sample_rate)``
    takes the records and their rate, so every result one ``FRF`` returns
    comes from the same estimator and settings."""

    @staticmethod
    def _signals(n=16384):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(n)
        return x, np.convolve(x, np.ones(5) / 5)[:n]

    def test_compute_has_no_per_call_setting(self):
        import inspect
        assert list(inspect.signature(FRF.compute).parameters) == [
            'self', 'x', 'y', 'sample_rate']
        x, y = self._signals()
        for key, value in (('method', 'etfe'), ('nperseg', 1024),
                           ('order', 'AIC'), ('max_order', 40)):
            with pytest.raises(TypeError, match=key):
                FRF().compute(x, y, 8000.0, **{key: value})

    def test_the_constructors_settings_reach_the_run(self):
        x, y = self._signals()
        assert FRF().compute(x, y, 8000.0).frequencies.size == 4097
        assert FRF(nperseg=1024).compute(x, y, 8000.0).frequencies.size == 513
        whole = FRF(method='etfe').compute(x, y, 8000.0)
        assert whole.coherence is None
        assert whole.frequencies.size == x.size // 2 + 1
        assert FRF(estimator='H2').compute(x, y, 8000.0).estimator == 'H2'

    def test_an_unknown_method_is_refused_by_the_constructor(self):
        with pytest.raises(ConfigurationError,
                           match=r"^FRF: unknown method='bogus'"):
            FRF(method='bogus')

    def test_lsfir_takes_input_first_like_its_siblings(self):
        """``(x, y)`` — input, then output — fits the system itself: a
        3-tap FIR comes back tap for tap. Reading the arguments the other
        way round fits the inverse system, whose taps are nothing like it."""
        rng = np.random.default_rng(4)
        u = rng.standard_normal(600)
        h = np.array([1.0, -0.5, 0.25])
        y = np.convolve(u, h)[:u.size]
        g = lsfir(u, y, 1000.0, order=3).impulse_response
        np.testing.assert_allclose(g, h, atol=1e-9)

    def test_lsfir_order_and_length_are_keyword_only(self):
        """A positional call in an ``(output, input, fs, m, N)`` shape
        raises instead of silently fitting the inverse system."""
        u = np.random.default_rng(4).standard_normal(64)
        with pytest.raises(TypeError,
                           match='takes 3 positional arguments but'):
            lsfir(u, u, 1000.0, 3, 64)


class TestThePeriodicEtfeTakesItsOwnPeriod:
    """``FRF(method='p_etfe', period=)``: the excitation period is its own
    keyword, required by 'p_etfe' and refused by every other method; the
    Welch segment length ``nperseg`` plays no part in it."""

    FS = 8000.0

    @staticmethod
    def _signals(n=16384):
        rng = np.random.default_rng(0)
        x = np.tile(rng.standard_normal(256), n // 256)
        return x, np.convolve(x, np.ones(5) / 5)[:n]

    def test_the_grid_is_the_periods_whatever_nperseg_says(self):
        x, y = self._signals()
        got = FRF(method='p_etfe', period=256, nperseg=4096).compute(
            x, y, self.FS)
        want = periodic_etfe(x, y, self.FS, period=256)
        assert got.frequencies.size == 129
        np.testing.assert_array_equal(got.transfer_function,
                                      want.transfer_function)

    def test_the_period_sets_the_grid(self):
        x, y = self._signals()
        got = FRF(method='p_etfe', period=128).compute(x, y, self.FS)
        assert got.frequencies.size == 65
        assert repr(FRF(method='p_etfe', period=256)) == \
            "FRF(method='p_etfe', period=256)"

    def test_p_etfe_without_a_period_is_refused(self):
        with pytest.raises(ConfigurationError,
                           match=r"method='p_etfe' averages whole periods"):
            FRF(method='p_etfe')

    @pytest.mark.parametrize('method', ['welch', 'etfe', 'ls_fir'])
    def test_a_period_under_another_method_is_refused(self, method):
        with pytest.raises(ConfigurationError,
                           match=rf"^FRF: period= .*method='{method}'"):
            FRF(method=method, period=256)

    @pytest.mark.parametrize('period, accepted', [
        (1, True), (256.0, True), (np.int64(64), True),
        (0, False), (-4, False), (2.5, False), (True, False),
        ('256', False), (np.nan, False), (np.inf, False), (None, False),
    ])
    def test_a_period_is_a_whole_number_of_samples(self, period, accepted):
        if accepted:
            assert FRF(method='p_etfe', period=period).period == int(
                float(period))
        else:
            with pytest.raises(ConfigurationError, match=r"^FRF: "):
                FRF(method='p_etfe', period=period)


def test_the_signal_guide_states_the_constructor_frf_contract():
    """``docs/guide/signal.md``'s FRF section states that every setting is
    the constructor's, and shows the order a criterion selects read off the
    result while the object keeps the criterion; this pins that sentence and
    the readback it demonstrates."""
    import pathlib
    page = (pathlib.Path(__file__).resolve().parents[2]
            / 'docs' / 'guide' / 'signal.md')
    if not page.is_file():
        pytest.skip('docs/ is not present (source checkout only)')
    text = page.read_text(encoding='utf-8')
    assert "**Every setting is the constructor's.**" in text
    assert '**`m` holds the criterion' not in text
    # And the readback the page does show has to be what the code returns.
    from uacpy.acoustic_signal.frf import FRF
    rng = np.random.default_rng(2)
    n = 3000
    u = rng.standard_normal(n)
    g = rng.standard_normal(6)
    g = g / np.linalg.norm(g)
    clean = np.convolve(u, g)[:n]
    y = clean + 0.1 * np.std(clean) * rng.standard_normal(n)
    frf = FRF(method='ls_fir', order='CP')
    assert frf.compute(u, y, 1000.0).order == 6
    assert frf.order == 'CP' and FRF(method='ls_fir').order == 512


class TestTheFunctionsAreTheEstimatorsFRFRuns:
    """``FRF`` configures and runs the four functions: on one record pair its
    result is theirs, value for value, and on a block the measurements'
    spectra are averaged before the ratio is formed."""

    FS = 1000.0

    @staticmethod
    def _pair(n=8192, seed=21):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(n)
        return x, (np.convolve(x, [1.0, -0.4, 0.2])[:n]
                   + 0.01 * rng.standard_normal(n))

    @pytest.mark.parametrize('method, call', [
        ('welch', lambda x, y: frf_welch(x, y, 1000.0, nperseg=1024)),
        ('etfe', lambda x, y: etfe(x, y, 1000.0)),
        ('p_etfe', lambda x, y: periodic_etfe(x, y, 1000.0, period=1024)),
        ('ls_fir', lambda x, y: lsfir(x, y, 1000.0, order=6, nperseg=1024)),
    ])
    def test_one_record_pair_gives_the_functions_result(self, method, call):
        x, y = self._pair()
        period = {'period': 1024} if method == 'p_etfe' else {}
        got = FRF(method=method, order=6, nperseg=1024,
                  **period).compute(x, y, self.FS)
        want = call(x, y)
        assert isinstance(got, FRFResult) and got.method == method
        for name in got._fields + got._attrs:
            a, b = getattr(got, name), getattr(want, name)
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)
            else:
                assert a == b, name

    def test_a_welch_block_is_h1_of_its_rows_averaged_spectra(self):
        """A two-channel analyser averages |X|², |Y|² and X*Y over the
        records, then forms H1 and the coherence (Jacobsen & Juhl App.
        B.7.3); the expectation is built from scipy's spectra, not from
        ``frf_welch``."""
        import scipy.signal as sig
        rows = [self._pair(seed=s) for s in (1, 2, 3)]
        x = np.stack([r[0] for r in rows])
        y = np.stack([r[1] for r in rows])
        block = FRF(nperseg=1024).compute(x, y, self.FS)
        opts = dict(nperseg=1024, noverlap=512, scaling='density')
        sxx = np.mean([sig.welch(r[0], self.FS, **opts)[1] for r in rows], 0)
        syy = np.mean([sig.welch(r[1], self.FS, **opts)[1] for r in rows], 0)
        sxy = np.mean([sig.csd(r[0], r[1], self.FS, **opts)[1] for r in rows],
                      0)
        np.testing.assert_allclose(block.transfer_function, sxy / sxx,
                                   rtol=1e-12)
        np.testing.assert_allclose(block.coherence,
                                   np.abs(sxy) ** 2 / (sxx * syy), rtol=1e-12)
        assert block.estimator == 'H1' and block.order is None

    def test_an_etfe_block_averages_the_spectra_not_the_ratios(self):
        """``Σ X*·Y / Σ |X|²`` (Jacobsen & Juhl App. B.6.2) settles on the
        system as records are added; the mean of the records' ``Y/X`` does
        not. On this fixture (4-tap FIR, 40 white-noise records, output noise
        σ 0.5) the ratio mean errs by 9.76 at its worst bin over 100-3500 Hz
        and the averaged spectra by 0.263, so the 0.5 bound separates them."""
        fs, n, h = 8000.0, 4096, np.array([1.0, 0.5, -0.3, 0.2])
        rng = np.random.default_rng(1)
        x = rng.standard_normal((40, n))
        y = (np.stack([np.convolve(row, h)[:n] for row in x])
             + 0.5 * rng.standard_normal((40, n)))
        block = FRF(method='etfe').compute(x, y, fs)
        X, Y = np.fft.rfft(x, axis=-1), np.fft.rfft(y, axis=-1)
        np.testing.assert_allclose(
            block.transfer_function,
            np.sum(np.conj(X) * Y, 0) / np.sum(np.abs(X) ** 2, 0),
            rtol=1e-12)
        f = np.fft.rfftfreq(n, 1 / fs)
        band = (f >= 100) & (f <= 3500)
        error = np.abs(block.transfer_function - np.fft.rfft(h, n))[band]
        assert error.max() < 0.5

    def test_a_periodic_etfe_block_pools_the_period_averaged_spectra(self):
        """Each row is averaged over its whole periods first, then the rows'
        spectra are pooled, ``Σ Xa*·Ya / Σ |Xa|²`` with ``Xa``/``Ya`` the
        rfft of each row's period mean — not the mean of the rows'
        ``periodic_etfe`` ratios, which this fixture (10 rows of 8 periods
        of 512, output noise σ 0.5) tells apart."""
        fs, period, h = 8000.0, 512, np.array([1.0, 0.5, -0.3, 0.2])
        rng = np.random.default_rng(3)
        x = np.stack([np.tile(rng.standard_normal(period), 8)
                      for _ in range(10)])
        y = (np.stack([np.convolve(row, h)[:row.size] for row in x])
             + 0.5 * rng.standard_normal(x.shape))
        block = FRF(method='p_etfe', period=period).compute(x, y, fs)
        Xa = np.fft.rfft(x.reshape(10, 8, period).mean(axis=1), axis=-1)
        Ya = np.fft.rfft(y.reshape(10, 8, period).mean(axis=1), axis=-1)
        pooled = np.sum(np.conj(Xa) * Ya, 0) / np.sum(np.abs(Xa) ** 2, 0)
        np.testing.assert_allclose(block.transfer_function, pooled,
                                   rtol=1e-12)
        ratios = np.mean([periodic_etfe(xr, yr, fs, period=period)
                          .transfer_function for xr, yr in zip(x, y)], 0)
        assert np.max(np.abs(ratios - pooled)) > 1e-3

    @pytest.mark.parametrize('method, period', [
        ('etfe', None), ('p_etfe', 512), ('welch', None)])
    def test_a_one_row_block_is_the_functions_estimate(self, method, period):
        x, y = self._pair()
        frf = FRF(method=method, nperseg=1024, period=period)
        block = frf.compute(x[np.newaxis], y[np.newaxis], self.FS)
        np.testing.assert_allclose(block.transfer_function,
                                   frf.compute(x, y, self.FS).transfer_function,
                                   rtol=1e-12)

    def test_the_result_unpacks_and_keeps_its_attributes_through_a_copy(self):
        import copy
        import pickle
        result = frf_welch(*self._pair(), self.FS, nperseg=1024,
                           estimator='H2')
        frequencies, transfer_function = result
        assert frequencies is result.frequencies
        for twin in (copy.deepcopy(result), pickle.loads(pickle.dumps(result)),
                     result._replace(transfer_function=transfer_function)):
            assert twin.estimator == 'H2' and twin.method == 'welch'
            np.testing.assert_array_equal(twin.coherence, result.coherence)
        assert result.units == {'frequencies': 'Hz', 'transfer_function': ''}

    def test_the_result_draws_its_response_through_plot_frf(self):
        import matplotlib.pyplot as plt
        result = etfe(*self._pair(), self.FS)
        fig, (ax_mag, ax_phase) = result.plot()
        try:
            np.testing.assert_array_equal(ax_mag.lines[0].get_xdata(),
                                          result.frequencies)
            np.testing.assert_allclose(
                ax_phase.lines[0].get_ydata(),
                np.angle(result.transfer_function))
        finally:
            plt.close(fig)

    @pytest.mark.parametrize('function', [
        lambda x, y: frf_welch(x, y, 1000.0), lambda x, y: etfe(x, y, 1000.0),
        lambda x, y: periodic_etfe(x, y, 1000.0, period=16),
        lambda x, y: lsfir(x, y, 1000.0, order=4)])
    def test_a_function_takes_one_record_pair_of_one_length(self, function):
        x = np.random.default_rng(0).standard_normal(256)
        with pytest.raises(ConfigurationError, match='must be 1-D records'):
            function(np.stack([x, x]), np.stack([x, x]))
        with pytest.raises(ConfigurationError, match='same length'):
            function(x, x[:-1])
        with pytest.raises(ConfigurationError, match='no samples'):
            function(x[:0], x[:0])

    def test_a_period_the_record_does_not_hold_once_is_refused(self):
        x = np.random.default_rng(0).standard_normal(256)
        with pytest.raises(ConfigurationError, match='at least one period'):
            periodic_etfe(x, x, 1000.0, period=300)

    @pytest.mark.parametrize('period', [0, -4, 2.5, True, '64', np.nan])
    def test_a_period_that_is_not_a_whole_count_is_refused_by_name(
            self, period):
        """The plain function refuses what FRF(method='p_etfe') refuses,
        rather than truncating 2.5 to 2, True to 1 or '64' to 64."""
        x = np.random.default_rng(0).standard_normal(256)
        with pytest.raises(ConfigurationError,
                           match=r'^periodic_etfe: period=.* whole number'):
            periodic_etfe(x, x, 1000.0, period=period)

    @pytest.mark.parametrize('period', [64, 64.0, np.int64(64)])
    def test_a_whole_period_of_any_numeric_type_runs(self, period):
        x = np.random.default_rng(0).standard_normal(256)
        out = periodic_etfe(x, x, 1000.0, period=period)
        assert out.frequencies.size == 33


def test_frf_names_its_fir_settings_as_lsfir_does():
    """The constructor takes ``order``, ``max_order`` and ``stop_count``
    under ``lsfir``'s names and defaults."""
    import inspect
    from uacpy.acoustic_signal import lsfir
    ctor = inspect.signature(FRF.__init__).parameters
    fit = inspect.signature(lsfir).parameters
    assert 'order' in ctor and 'm' not in ctor
    for name in ('max_order', 'stop_count'):
        assert ctor[name].default == fit[name].default


@pytest.mark.parametrize('kwargs, refused', [({'m': 4}, True),
                                            ({'nperseg': 512}, False),
                                            ({'window': 'hann'}, False)])
def test_frf_refuses_a_keyword_that_is_not_a_welch_option(kwargs, refused):
    if refused:
        with pytest.raises(ConfigurationError,
                           match=r"\['m'\] are not Welch options"):
            FRF(**kwargs)
    else:
        FRF(**kwargs)


class TestTheFrfFamilyTakesRealRecords:
    """Every FRF estimator works on the one-sided grid of a real record, so a
    complex (baseband) pair is refused by name instead of answering wrong
    (lsfir) or failing in numpy (etfe)."""

    @staticmethod
    def _pair(complex_side):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(2048)
        y = np.convolve(x, [1.0, 0.5, -0.25, 0.1])[:x.size]
        if complex_side == 'x':
            x = x * (1.0 + 0.5j)
        elif complex_side == 'y':
            y = y * (1.0 + 0.5j)
        return x, y

    @pytest.mark.parametrize('side', ['x', 'y'])
    @pytest.mark.parametrize('estimate', [
        lambda x, y: frf_welch(x, y, 1000.0, nperseg=256),
        lambda x, y: etfe(x, y, 1000.0),
        lambda x, y: periodic_etfe(x, y, 1000.0, period=256),
        lambda x, y: lsfir(x, y, 1000.0, order=4),
    ], ids=['frf_welch', 'etfe', 'periodic_etfe', 'lsfir'])
    def test_a_complex_record_is_refused(self, estimate, side):
        x, y = self._pair(side)
        with pytest.raises(ConfigurationError, match='real passband records'):
            estimate(x, y)

    def test_the_real_pair_identifies_the_filter(self):
        x, y = self._pair(None)
        np.testing.assert_allclose(
            lsfir(x, y, 1000.0, order=4).impulse_response,
            [1.0, 0.5, -0.25, 0.1], atol=1e-10)


class TestLsfirOrderAndSampleGuards:
    """Both sides of each bound: order 0 / 1, n_samples 0 / 1 and
    len / len + 1."""

    @staticmethod
    def _pair():
        rng = np.random.default_rng(8)
        x = rng.standard_normal(500)
        return x, np.convolve(x, [1.0, 0.5])[:x.size]

    @pytest.mark.parametrize('order, refused', [(0, True), (-2, True),
                                                (1, False)])
    def test_an_order_below_one_is_refused(self, order, refused):
        x, y = self._pair()
        if refused:
            with pytest.raises(ConfigurationError,
                               match='FIR order must be at least 1'):
                lsfir(x, y, 1000.0, order=order)
        else:
            assert lsfir(x, y, 1000.0, order=order).impulse_response.size == 1

    @pytest.mark.parametrize('n_samples, refused', [(0, True), (1, False),
                                                    (500, False), (501, True)])
    def test_n_samples_must_lie_within_the_record(self, n_samples, refused):
        x, y = self._pair()
        if refused:
            with pytest.raises(ConfigurationError,
                               match=r'n_samples \(\d+\) must be in \[1, 500\]'):
                lsfir(x, y, 1000.0, order=1, n_samples=n_samples)
        else:
            lsfir(x, y, 1000.0, order=1, n_samples=n_samples)


class TestTheOrderCriteriaAreThePublishedForms:
    """``lsfir``'s criteria score each order ``m`` on its fit's loss
    ``V = SSE/N``: Akaike's AIC ``log V + 2m/N``, Schwarz's BIC
    ``log V + m log N/N``, Akaike's FPE ``V (N+m)/(N-m)`` and Mallows' Cp
    ``SSE/σ² - N + 2(m+1)``. Each selected order is the argmin of the form
    computed here from the fits at every explicit order. On this fixture
    (N = 120, a decaying 10-tap FIR, 30 % output noise) the AIC, BIC and FPE
    forms on ``SSE/(N - m)`` would select 6, 5 and 6 taps, where the
    published ones select 8, 6 and 8."""

    N, M_MAX = 120, 40

    def _fixture(self):
        rng = np.random.default_rng(13)
        u = rng.standard_normal(self.N)
        g = rng.standard_normal(10) * np.exp(-0.25 * np.arange(10))
        clean = np.convolve(u, g)[:self.N]
        return u, clean + 0.3 * np.std(clean) * rng.standard_normal(self.N)

    def _sse(self, u, y, m):
        g = lsfir(u, y, 1000.0, order=m).impulse_response
        return float(np.sum((y - np.convolve(u, g)[:self.N]) ** 2))

    @pytest.mark.parametrize('criterion, expected', [
        ('AIC', 8), ('BIC', 6), ('FPE', 8), ('CP', None)])
    def test_the_selected_order_minimises_the_published_score(
            self, criterion, expected):
        u, y = self._fixture()
        n = self.N
        orders = np.arange(1, self.M_MAX + 1)
        sse = np.array([self._sse(u, y, m) for m in orders])
        v = sse / n
        reference = min(self.M_MAX, max(2, n // 4))     # 30 taps here
        sigma2 = self._sse(u, y, reference) / (n - reference)
        score = {'AIC': np.log(v) + 2 * orders / n,
                 'BIC': np.log(v) + orders * np.log(n) / n,
                 'FPE': v * (n + orders) / (n - orders),
                 'CP': sse / sigma2 - n + 2 * (orders + 1)}[criterion]
        fit = lsfir(u, y, 1000.0, order=criterion, max_order=self.M_MAX,
                    stop_count=self.M_MAX)
        assert fit.order == int(orders[np.argmin(score)])
        if expected is not None:
            assert fit.order == expected
