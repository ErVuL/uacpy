"""The ``FRF`` frequency-response estimator in ``uacpy.acoustic_signal.system``.

``FRF`` is the one class on the ``acoustic_signal`` surface — it carries the
fit it made — and that state is what these tests are about:

* **Which estimator ran.** ``H1``/``H2``/``ls_fir`` answer differently on the
  same record, and the class publishes the method, the selected model order
  and the conditioning of the fit rather than only the response.
* **Per-call keywords stay per call.** A keyword passed to one ``compute``
  does not survive into the next, so two calls on one object are two
  independent fits.
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

from uacpy.acoustic_signal.system import FRF
from uacpy.core.exceptions import ConfigurationError


class TestFRF:
    """FRF automatic FIR-order selection (m='AIC'|'BIC'|'FPE'|'CP') must run,
    not crash with 'count >= None' from an un-defaulted stop_count."""

    @pytest.mark.parametrize("criterion", ['AIC', 'BIC', 'FPE', 'CP'])
    def test_auto_order_runs_and_recovers_order(self, criterion):
        from uacpy.acoustic_signal.system import FRF
        rng = np.random.default_rng(1)
        u = rng.standard_normal(2000)
        g = np.array([1.0, -0.5, 0.25])                  # order-3 FIR
        y = np.convolve(u, g)[:u.size] + 0.01 * rng.standard_normal(2000)
        frf = FRF()
        _, tf = frf.compute(u, y, 1000.0, method='ls_fir', m=criterion)
        assert np.isfinite(tf).all()
        # every criterion recovers the true order-3 FIR at this SNR; the
        # chosen order is published on .selected_order so a reused FRF
        # re-selects instead of pinning, and the per-call criterion leaves
        # the object's own .m as the constructor set it.
        assert frf.selected_order == 3
        assert frf.m == FRF().m

    def test_cp_recovers_order_six_fir(self):
        """Mallows' Cp recovers the true order-6 FIR at moderate SNR. Cp scales
        the residual sum of squares by σ̂², the residual variance of a low-bias
        reference fit; this higher-order case exercises that estimate (order 3
        at high SNR above is too easy to constrain it).
        """
        from uacpy.acoustic_signal.system import FRF
        r = np.random.default_rng(2)
        N, order = 3000, 6
        u = r.standard_normal(N)
        g = r.standard_normal(order)
        g = g / np.linalg.norm(g)
        clean = np.convolve(u, g)[:N]
        y = clean + 0.1 * np.std(clean) * r.standard_normal(N)
        frf = FRF()
        _, tf = frf.compute(u, y, 1000.0, method='ls_fir', m='CP')
        assert np.isfinite(tf).all()
        assert frf.selected_order == order
        assert frf.m == FRF().m

    @pytest.mark.parametrize("criterion", ['AIC', 'BIC', 'FPE', 'CP'])
    def test_order_selection_is_amplitude_scale_invariant(self, criterion):
        """The selected order must depend on the data, not on its units: a
        pressure record in Pa and the same record in MPa must give the same
        FIR order. All four criteria compare log(sse) or sse ratios, so only
        the exact-fit cutoff can break the invariance."""
        from uacpy.acoustic_signal.system import FRF
        rng = np.random.default_rng(0)
        u = rng.standard_normal(400)
        g = np.array([1.0, 0.5, -0.3])
        y = np.convolve(u, g)[:u.size] + 0.01 * rng.standard_normal(400)
        orders = []
        for scale in (1.0, 1e-3, 1e-6, 1e-9):
            frf = FRF()
            frf.compute(scale * u, scale * y, 1000.0, method='ls_fir',
                        m=criterion, m_max=60)
            orders.append(frf.selected_order)
        assert orders == [3, 3, 3, 3]

    @pytest.mark.parametrize("criterion", ['AIC', 'BIC', 'FPE', 'CP'])
    def test_exact_fit_selects_lowest_explaining_order(self, criterion):
        """A pure-gain loopback y = 2*u is fitted exactly at order 1; the
        search must return that order instead of discarding every candidate."""
        from uacpy.acoustic_signal.system import FRF
        rng = np.random.default_rng(3)
        u = rng.standard_normal(400)
        frf = FRF()
        _, tf = frf.compute(u, 2.0 * u, 1000.0, method='ls_fir', m=criterion,
                            m_max=60)
        assert np.isfinite(tf).all()
        assert frf.selected_order == 1
        assert np.asarray(frf.g) == pytest.approx([2.0])

    def test_unfittable_input_raises_configurationerror(self):
        """An all-zero input is singular at every order: typed error, not a
        ValueError out of scipy.signal.freqz on a None filter."""
        from uacpy.acoustic_signal.system import FRF
        from uacpy.core.exceptions import ConfigurationError
        rng = np.random.default_rng(4)
        y = rng.standard_normal(300)
        with pytest.raises(ConfigurationError):
            FRF().compute(np.zeros(300), y, 1000.0, method='ls_fir', m='AIC',
                          m_max=40)

    def test_zero_row_input_raises_configurationerror(self):
        """A 2-D input with no measurement rows must not fall through the
        per-measurement loop and hit an UnboundLocalError on the frequency
        axis."""
        from uacpy.acoustic_signal.system import FRF
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError):
            FRF().compute(np.zeros((0, 100)), np.zeros((0, 100)), 1000.0,
                          method='ls_fir', m=4)

    def test_method_switch_clears_ls_fir_state(self):
        """``selected_order``/``g`` are ls_fir-only and ``coh`` is welch-only;
        a reused FRF must not report the previous method's values."""
        from uacpy.acoustic_signal.system import FRF
        rng = np.random.default_rng(11)
        u = rng.standard_normal(4096)
        y = np.convolve(u, [1.0, 0.5, -0.3])[:u.size]
        frf = FRF()
        frf.compute(u, y, 1000.0, method='ls_fir', m='AIC', m_max=40)
        assert frf.selected_order is not None
        assert frf.coh is None
        frf.compute(u, y, 1000.0, method='welch', nperseg=512)
        assert frf.selected_order is None and frf.g == 0
        assert frf.coh is not None and frf.coh.shape == frf.frequencies.shape


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
        from uacpy.acoustic_signal.system import FRF
        f, tf, coh = FRF(estimator=estimator,
                         nperseg=4096).compute_welch(x, y, cls.FS)
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


class TestFRFReservedKwargs:
    """Welch options that the FRF sets internally are rejected typed, not
    left to die in scipy as a bare TypeError."""

    def test_scaling_and_fs_raise_configurationerror(self):
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.acoustic_signal.system import FRF
        with pytest.raises(ConfigurationError, match="scaling"):
            FRF(method="welch", scaling="spectrum")
        with pytest.raises(ConfigurationError, match="fs"):
            FRF(method="welch", fs=48_000.0)

    def test_legitimate_welch_kwargs_pass_through(self):
        from uacpy.acoustic_signal.system import FRF
        rng = np.random.default_rng(2)
        x = rng.standard_normal(8192)
        y = np.convolve(x, [1.0, 0.5], mode="same")
        freqs, tf = FRF(method="welch", nperseg=1024,
                        window="hamming").compute(x, y, 8000.0)
        assert freqs.size == 513 and np.all(np.isfinite(tf))

    def test_etfe_grid_is_the_full_record_grid(self):
        from uacpy.acoustic_signal.system import FRF
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
        return FRF(method='ls_fir').compute_lsfir(
            np.zeros(self.N), np.ones(self.N), 1000.0, criterion, self.N)

    def test_cp_raises_the_typed_error_the_other_criteria_reach(self):
        # The Cp reference fit runs before the candidate loop's LinAlgError
        # handling, so a constant input escaped as a raw LinAlgError.
        with pytest.raises(ConfigurationError, match="criterion 'CP'"):
            self._degenerate('CP')

    @pytest.mark.parametrize("criterion", ['AIC', 'BIC', 'FPE'])
    def test_the_other_criteria_return_a_non_empty_result_on_the_same_input(self, criterion):
        assert self._degenerate(criterion)[0].size > 0


class TestLsFirSolvesSingularNormalEquationsByMinimumNorm:
    """``compute_lsfir`` fits through ``X.T @ X``, whose condition number is
    ``cond(X)**2``, so a probe that leaves part of the Nyquist band unexcited
    makes the system numerically singular at any FIR order longer than the
    excited band supports. The shipped default ``m=512`` reaches it on an
    ordinary 100 Hz - 20 kHz sweep at fs = 48 kHz (``cond(X) = 4.2e11``,
    reciprocal condition number of the information matrix 1e-20): the LU
    solve of that system carries no correct digit, and the coefficients come
    back from a rank-revealing least-squares solve of the same equations
    instead, with the order named in a warning.
    """

    fs = 48000.0
    N = 8000
    m = 512
    h_true = np.array([1.0, -0.6, 0.3, 0.1])

    def _fit(self):
        import scipy.signal as sig
        rng = np.random.default_rng(11)
        u = sig.chirp(np.arange(self.N) / self.fs, 100.0,
                      self.N / self.fs, 20000.0)
        y = (np.convolve(u, self.h_true)[:self.N]
             + 1e-8 * rng.standard_normal(self.N))
        frf = FRF(method='ls_fir')
        freqs, h, g = frf.compute_lsfir(y, u, self.fs, self.m, self.N,
                                        nperseg=2048)
        return frf, freqs, h, g

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
            _, _, _, g = self._fit()
        assert np.max(np.abs(g)) == pytest.approx(1.0, abs=0.05)

    def test_the_frequency_response_matches_the_channel_across_the_swept_band(self):
        # The LU solve of the same equations is 46.6 dB out here.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, freqs, h, _ = self._fit()
        assert self._in_band_dB_error(freqs, h) < 0.05

    def test_the_warning_names_the_order_and_the_condition_estimate(self):
        with pytest.warns(UserWarning, match=r"FIR order 512 is numerically "
                                             r"singular \(reciprocal condition "
                                             r"number "):
            frf, _, _, _ = self._fit()
        assert frf.info_rcond < np.finfo(float).eps

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
        frf = FRF(method='ls_fir')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _, _, g = frf.compute_lsfir(y, u, self.fs, order, self.N,
                                        nperseg=2048)
        assert frf.info_rcond > 0.05
        # A well-conditioned fit is the LU solution of the normal equations,
        # to the tolerance two builds of the same LAPACK routine can differ by.
        g_lu = np.linalg.solve(frf.Minfo, frf.Vinfo)
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
        from uacpy.acoustic_signal.system import (
            _INFO_RCOND_FLOOR, _solve_info_matrices)
        delta = 2.0 * _INFO_RCOND_FLOOR
        minfo, vinfo = self._system(delta)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            g, rcond = _solve_info_matrices(minfo, vinfo, self.n)
        assert rcond == pytest.approx(delta, rel=1e-12)
        assert rcond > _INFO_RCOND_FLOOR
        assert g[-1] == pytest.approx(1.0 / delta, rel=1e-12)

    def test_at_the_floor_the_ill_conditioned_direction_is_dropped(self):
        from uacpy.acoustic_signal.system import (
            _INFO_RCOND_FLOOR, _solve_info_matrices)
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
        from uacpy.acoustic_signal.system import (
            _INFO_RCOND_FLOOR, _solve_info_matrices)
        minfo, vinfo = self._system(0.5 * _INFO_RCOND_FLOOR)
        with pytest.warns(UserWarning, match="numerically singular"):
            g, rcond = _solve_info_matrices(minfo, vinfo, self.n)
        assert rcond < _INFO_RCOND_FLOOR
        assert g[-1] == 0.0

    @pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
    def test_the_branch_does_not_move_with_the_amplitude_scale(self, scale):
        """A record in Pa and the same record in uPa must be fitted the same
        way: the threshold is on a reciprocal condition number, which both
        norms scale out of."""
        from uacpy.acoustic_signal.system import (
            _INFO_RCOND_FLOOR, _solve_info_matrices)
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
        from uacpy.acoustic_signal.system import _INFO_RCOND_FLOOR
        assert _INFO_RCOND_FLOOR == np.finfo(float).eps

    def test_an_exactly_singular_system_raises_the_error_the_order_search_skips_on(self):
        from uacpy.acoustic_signal.system import _solve_info_matrices
        with pytest.raises(np.linalg.LinAlgError):
            _solve_info_matrices(np.zeros((4, 4)), np.zeros(4), 4)


class TestFrfPublishesTheConditioningOfTheFitItReturns:
    def test_info_rcond_is_none_until_an_ls_fir_run_and_after_a_welch_one(self):
        rng = np.random.default_rng(11)
        u = rng.standard_normal(4096)
        y = np.convolve(u, [1.0, 0.5, -0.3])[:u.size]
        frf = FRF()
        assert frf.info_rcond is None
        frf.compute(u, y, 1000.0, method='ls_fir', m=8)
        assert 0.0 < frf.info_rcond <= 1.0
        frf.compute(u, y, 1000.0, method='welch', nperseg=512)
        assert frf.info_rcond is None


class TestSelectedOrderContract:
    def _fir_records(self, rows, seed=1):
        rng = np.random.default_rng(seed)
        u = rng.standard_normal((rows, 600))
        g = np.array([1.0, -0.5, 0.25])
        y = np.stack([np.convolve(u[i], g)[:600] for i in range(rows)])
        return u, y + 0.01 * rng.standard_normal(y.shape)

    def test_two_dimensional_input_publishes_one_order_per_row(self):
        u, y = self._fir_records(rows=3)
        frf = FRF()
        frf.compute(u, y, 1000.0, method="ls_fir", m="AIC", m_max=40)
        assert frf.selected_order == [3, 3, 3]

    def test_one_dimensional_criterion_input_publishes_an_int(self):
        u, y = self._fir_records(rows=1)
        frf = FRF()
        frf.compute(u[0], y[0], 1000.0, method="ls_fir", m="BIC", m_max=40)
        assert isinstance(frf.selected_order, int)
        assert frf.selected_order == 3

    def test_explicit_order_publishes_no_selected_order(self):
        u, y = self._fir_records(rows=1)
        frf = FRF()
        frf.compute(u[0], y[0], 1000.0, method="ls_fir", m=5)
        assert frf.selected_order is None
        assert len(np.asarray(frf.g)) == 5


class TestFRFComputeKeywordsApplyToOneCallOnly:
    """A per-call ``method=`` / ``estimator=`` / ``nperseg=`` configures that
    run and nothing after it.

    The sharpest face is the frequency grid: one
    ``compute_periodic_etfe(nperseg=256)`` on a default ``FRF`` would move
    every later plain ``compute()`` onto a 129-bin axis instead of 4097 — a
    32x change in the axis two results are compared on, from a call that
    returned its own result and looked finished.
    """

    @staticmethod
    def _signals(n=16384):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(n)
        return x, np.convolve(x, np.ones(5) / 5)[:n]

    def test_a_method_override_does_not_stick(self):
        x, y = self._signals()
        frf = FRF()
        freqs_default, _ = frf.compute(x, y, 8000.0)
        frf.compute(x, y, 8000.0, method='etfe')
        assert frf.method == 'welch'
        freqs_after, _ = frf.compute(x, y, 8000.0)
        assert freqs_after.size == freqs_default.size

    def test_an_estimator_override_does_not_stick(self):
        x, y = self._signals()
        frf = FRF()
        frf.compute(x, y, 8000.0, estimator='H2')
        assert frf.estimator == 'H1'

    @pytest.mark.parametrize('key, value', [('nperseg', 1024),
                                            ('noverlap', 64)])
    def test_a_welch_parameter_override_does_not_stick(self, key, value):
        x, y = self._signals()
        frf = FRF()
        frf.compute(x, y, 8000.0, **{key: value})
        assert frf.params[key] == FRF().params[key]

    def test_compute_periodic_etfe_does_not_move_the_shared_grid(self):
        x, y = self._signals()
        frf = FRF()
        frf.compute_periodic_etfe(x, y, 8000.0, nperseg=256)
        assert frf.params['nperseg'] == FRF().params['nperseg']
        freqs, _ = frf.compute(x, y, 8000.0)
        assert freqs.size == FRF().params['nperseg'] // 2 + 1

    def test_compute_lsfir_does_not_move_the_shared_grid(self):
        rng = np.random.default_rng(4)
        u = rng.standard_normal(600)
        y = np.convolve(u, np.array([1.0, -0.5, 0.25]))[:u.size]
        frf = FRF()
        frf.compute_lsfir(y, u, 1000.0, m=8, N=600, nperseg=128)
        assert frf.params['nperseg'] == FRF().params['nperseg']

    def test_the_override_reaches_the_run_it_was_given_to(self):
        """The negative half: a per-call keyword must change *this* result."""
        x, y = self._signals()
        frf = FRF()
        freqs_default, _ = frf.compute(x, y, 8000.0)
        freqs_override, _ = frf.compute(x, y, 8000.0, nperseg=1024)
        assert freqs_default.size == 4097
        assert freqs_override.size == 513

    def test_the_result_attributes_are_rewritten_by_every_run(self):
        x, y = self._signals()
        frf = FRF()
        frf.compute(x, y, 8000.0)
        assert frf.coh is not None
        frf.compute(x, y, 8000.0, method='etfe')
        assert frf.coh is None
        assert frf.frequencies.size == x.size // 2 + 1


class TestTransferFunctionImpulseResponseValidatesItsSampleRate:
    """Every sibling entry point routes ``sample_rate`` through the shared
    positive-finite-scalar guard; this one did ``float(sample_rate)``.

    Measured before the fix, with ``frequencies`` spanning 0-500 Hz: a rate of
    0 raised a bare ``ZeroDivisionError``, NaN a ``ValueError`` about
    converting NaN to an integer, Inf an ``OverflowError``, a non-number a
    ``ValueError`` from ``float()`` — and a negative rate reached the Nyquist
    check and raised a ``ConfigurationError`` announcing "the Nyquist
    frequency -500 Hz", a true statement about the wrong argument.
    """

    @staticmethod
    def _call(rate):
        from uacpy.acoustic_signal.system import (
            impulse_response_from_transfer_function)
        f = np.arange(0.0, 501.0, 1.0)
        return impulse_response_from_transfer_function(
            np.ones(f.size, dtype=complex), f, rate)

    @pytest.mark.parametrize('rate', [0.0, -1000.0, float('nan'),
                                      float('inf'), -float('inf')])
    def test_a_non_positive_or_non_finite_rate_names_sample_rate(self, rate):
        with pytest.raises(ConfigurationError) as exc:
            self._call(rate)
        message = str(exc.value)
        assert 'impulse_response_from_transfer_function' in message
        assert 'sample_rate' in message and '> 0 Hz' in message
        assert 'Nyquist' not in message, (
            f"a bad sample_rate must not be reported as a statement about the "
            f"frequency axis: {message}")

    def test_a_non_numeric_rate_names_sample_rate_too(self):
        with pytest.raises(ConfigurationError, match='sample_rate must be a '
                                                     'scalar number'):
            self._call('fast')

    def test_a_positive_finite_rate_returns_a_finite_response(self):
        t, h = self._call(2000.0)
        assert t.size == h.size > 0 and np.all(np.isfinite(h))


def test_the_signal_guide_states_the_per_call_frf_contract():
    """``docs/guide/signal.md``'s FRF section documented ``frf.m`` reading back
    as the criterion string, which is the behaviour the per-call resolution
    removed. A prose page cannot be pinned wholesale, so this pins the one
    sentence that went stale and the readback it demonstrated."""
    import pathlib
    page = (pathlib.Path(__file__).resolve().parents[2]
            / 'docs' / 'guide' / 'signal.md')
    if not page.is_file():
        pytest.skip('docs/ is not present (source checkout only)')
    text = page.read_text(encoding='utf-8')
    assert 'Every `compute` argument applies to that call alone' in text
    assert '**`m` holds the criterion' not in text
    # The snippet must not show the criterion coming back off the object.
    assert ">>> frf.m\n'CP'" not in text
    # And the readback the page does show has to be what the code returns.
    from uacpy.acoustic_signal.system import FRF
    rng = np.random.default_rng(2)
    n = 3000
    u = rng.standard_normal(n)
    g = rng.standard_normal(6)
    g = g / np.linalg.norm(g)
    clean = np.convolve(u, g)[:n]
    y = clean + 0.1 * np.std(clean) * rng.standard_normal(n)
    frf = FRF(method='ls_fir')
    frf.compute(u, y, 1000.0, m='CP')
    assert frf.selected_order == 6
    assert frf.m == FRF(method='ls_fir').m == 512
