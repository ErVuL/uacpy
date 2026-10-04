"""Tests for the MFP processors on :class:`Covariance`."""

import warnings

import numpy as np
import pytest
from uacpy.core.exceptions import ConfigurationError

from uacpy.core.results import Covariance, Replicas


def _linear(field):
    """The linear surface an ambiguity Field is in dB re the peak of."""
    return field.reference * 10 ** (np.asarray(field.data) / 10)


def _synthetic_mfp(n_rcv=8, n_zr=5, n_xr=4, n_yr=1, source_idx=(2, 2, 0),
                   noise_level=0.05, freq=200.0):
    """Build a covariance + replica pair where the source sits at
    ``source_idx``.

    The replica bank is complex Gaussian, one independent vector per
    candidate (z, x, y) point; both processors normalise each vector to unit
    length themselves, so only its direction matters. The seed is fixed
    because the peak-location assertions hold for this draw, not for every
    draw — a different seed can put two candidate points near-degenerate.
    """
    rng = np.random.default_rng(0)
    # Make replica vectors random but unitary per-(z,x,y)
    replicas = (rng.normal(size=(1, n_zr, n_xr, n_yr, n_rcv))
                + 1j * rng.normal(size=(1, n_zr, n_xr, n_yr, n_rcv)))
    # Inject the true source at source_idx with bigger magnitude
    truth = replicas[0, source_idx[0], source_idx[1], source_idx[2]]
    truth = truth / np.linalg.norm(truth)
    # Covariance: rank-one signal plus full-rank noise. Adding a Hermitian
    # *perturbation* instead would leave C indefinite, which no measured
    # covariance is — and an indefinite C makes wᴴC⁻¹w negative, i.e. a
    # negative Capon power.
    noise = (rng.normal(size=(n_rcv, n_rcv))
             + 1j * rng.normal(size=(n_rcv, n_rcv)))
    C = (np.outer(truth, truth.conj())
         + noise_level * (noise @ noise.conj().T) / n_rcv)
    cov = Covariance(
        covariance=C[np.newaxis],
        model='Test', frequencies=freq,
    )
    rep = Replicas(
        replicas=replicas,
        candidates={'depth': np.linspace(20.0, 80.0, n_zr),
                    'x': np.linspace(500.0, 2000.0, n_xr),
                    'y': np.linspace(0.0, 0.0, n_yr)},
        model='Test', frequencies=freq,
    )
    return cov, rep, source_idx


class TestBartlett:
    def test_peaks_at_true_source(self):
        cov, rep, src = _synthetic_mfp(noise_level=0.01)
        amb = cov.bartlett(rep)
        assert amb.shape == (1, 5, 4, 1)
        assert list(amb.coords) == ['frequency', 'depth', 'x', 'y']
        peak = np.unravel_index(np.nanargmax(amb.data[0]), amb.shape[1:])
        assert peak == src

    def test_real_valued_output(self):
        cov, rep, _ = _synthetic_mfp()
        amb = cov.bartlett(rep)
        assert amb.data.dtype == float
        assert np.all(np.isfinite(amb.data))


class TestMVDR:
    def test_peaks_at_true_source(self):
        cov, rep, src = _synthetic_mfp(noise_level=0.01)
        amb = cov.mvdr(rep, diagonal_loading=1e-2)
        peak = np.unravel_index(np.nanargmax(amb.data[0]), amb.shape[1:])
        assert peak == src

    def test_diagonal_loading_smooths(self):
        cov, rep, _ = _synthetic_mfp(noise_level=0.1)
        # Heavier loading flattens the surface. Compare standard
        # deviation normalised by mean (coefficient of variation) so
        # the metric is independent of overall scale.
        loose = _linear(cov.mvdr(rep, diagonal_loading=1e-3))
        loaded = _linear(cov.mvdr(rep, diagonal_loading=10.0))
        loose_cv = loose.std() / abs(loose.mean())
        loaded_cv = loaded.std() / abs(loaded.mean())
        assert loaded_cv < loose_cv


class TestMVDRHeavyLoading:
    def test_peaks_at_true_source(self):
        cov, rep, src = _synthetic_mfp(noise_level=0.05)
        amb = cov.mvdr(rep, diagonal_loading=0.05)
        peak = np.unravel_index(np.nanargmax(amb.data[0]), amb.shape[1:])
        assert peak == src

    def test_heavy_loading_correlates_with_bartlett(self):
        # Heavy diagonal loading collapses the MVDR surface onto the Bartlett
        # surface: as delta grows, (C + delta*I)^-1 -> (I - C/delta)/delta, so
        # 1/(w^H (C+delta*I)^-1 w) -> delta + w^H C w, an affine map of
        # Bartlett with correlation 1. delta = 100 is short of that limit, so
        # the threshold is "nearly collinear", not exact.
        cov, rep, _ = _synthetic_mfp(noise_level=0.1)
        bart = _linear(cov.bartlett(rep))[0].ravel()
        loaded = _linear(cov.mvdr(rep, diagonal_loading=100.0))[0].ravel()
        r = np.corrcoef(bart, loaded)[0, 1]
        assert r > 0.95


class TestShapeChecks:
    def test_freq_mismatch_raises(self):
        cov, rep, _ = _synthetic_mfp()
        bad = Replicas(
            replicas=np.zeros((2, 5, 4, 1, 8), dtype=complex),
            candidates=rep.candidates, model='Test',
            frequencies=np.array([200.0, 400.0]),
        )
        with pytest.raises(ConfigurationError, match="frequency mismatch"):
            cov.bartlett(bad)

    def test_receiver_count_mismatch_raises(self):
        cov, rep, _ = _synthetic_mfp(n_rcv=8)
        bad = Replicas(
            replicas=np.zeros((1, 5, 4, 1, 6), dtype=complex),
            candidates=rep.candidates, model='Test',
            frequencies=200.0,
        )
        with pytest.raises(ConfigurationError, match="receiver-count mismatch"):
            cov.mvdr(bad)

    def test_a_frequency_axis_neither_records_is_refused(self):
        """The ambiguity Field stands on the frequencies; with neither the
        covariance nor the replicas recording them there is no axis."""
        cov, rep, _ = _synthetic_mfp()
        bare_cov = Covariance(covariance=cov.covariance)
        bare_rep = Replicas(replicas=rep.replicas, candidates=rep.candidates)
        with pytest.raises(ConfigurationError, match="frequency axis"):
            bare_cov.bartlett(bare_rep)
        # Either one recording them is enough.
        assert bare_cov.bartlett(rep).coords['frequency'][0] == 200.0


class TestTheTwoMfpEntryPointsAgree:
    """``Covariance.bartlett``/``.mvdr`` and ``uacpy.sonar``'s free functions
    are the same two processors over different input conventions (OASN
    covariance per frequency vs one measured CSDM; unnormalised vs
    normalised). They must stay numerically identical up to the
    normalisation each documents — a drift in either implementation is a
    real defect.
    """

    N_RCV, N_PTS = 6, 5

    def _rig(self):
        rng = np.random.default_rng(3)
        snaps = (rng.normal(size=(self.N_RCV, 4))
                 + 1j * rng.normal(size=(self.N_RCV, 4)))
        K = (snaps @ snaps.conj().T) / 4
        E = (rng.normal(size=(self.N_RCV, self.N_PTS))
             + 1j * rng.normal(size=(self.N_RCV, self.N_PTS)))
        cov = Covariance(covariance=K[None], model='OASN', frequencies=200.0)
        rep = Replicas(
            replicas=E.T.reshape(1, self.N_PTS, 1, 1, self.N_RCV),
            candidates={'depth': np.arange(self.N_PTS, dtype=float),
                        'x': np.array([0.0]), 'y': np.array([0.0])},
            model='OASN', frequencies=200.0,
        )
        return K, cov, rep

    def test_bartlett_differs_by_exactly_the_covariance_trace(self):
        from uacpy.sonar.matched_field import bartlett
        K, cov, rep = self._rig()
        core = _linear(cov.bartlett(rep)).ravel()
        sonar = _linear(bartlett(K, rep)).ravel()
        np.testing.assert_allclose(core, sonar * np.real(np.trace(K)),
                                   rtol=1e-12)

    @pytest.mark.parametrize('loading', [1e-6, 1e-3, 1e-2])
    def test_mvdr_differs_by_exactly_the_surface_max(self, loading):
        from uacpy.sonar.matched_field import mvdr
        K, cov, rep = self._rig()
        core = _linear(cov.mvdr(rep, diagonal_loading=loading)).ravel()
        sonar = _linear(mvdr(K, rep, diagonal_loading=loading)).ravel()
        np.testing.assert_allclose(core / core.max(), sonar, rtol=1e-10)

    @pytest.mark.parametrize('loading', [1e-6, 1e-3, 1e-2])
    def test_a_covariance_in_pascal_squared_scales_the_surface_not_its_peak(
            self, loading):
        """OASN's covariance is 1e-12 times the µPa²/Hz its binary writes.
        Bartlett is linear in C and MVDR's loading is relative to
        tr(C)/N, so both surfaces scale by that same 1e-12 and keep their
        argmax: the dB-re-peak values are unchanged and the scale lands on
        the reference. Measured: Bartlett 5.6e-16 relative; MVDR 6.8e-11 at
        loading 1e-6 (the loaded inverse's conditioning), <= 4.7e-14 at
        1e-3 and 1e-2."""
        K, cov, rep = self._rig()
        pa = Covariance(covariance=(K * 1e-12)[None], model='OASN',
                        frequencies=200.0)
        np.testing.assert_allclose(_linear(pa.bartlett(rep)),
                                   1e-12 * _linear(cov.bartlett(rep)),
                                   rtol=1e-14, atol=0)
        upa = _linear(cov.mvdr(rep, diagonal_loading=loading)).ravel()
        si = _linear(pa.mvdr(rep, diagonal_loading=loading)).ravel()
        np.testing.assert_allclose(si, 1e-12 * upa, rtol=1e-9, atol=0)
        assert si.argmax() == upa.argmax()

    def test_the_two_defaults_are_the_documented_pair(self):
        """Different on purpose — OASN's covariance is full rank, a measured
        few-snapshot CSDM is not. Pinned so a change has to be deliberate."""
        import inspect
        from uacpy.sonar.matched_field import mvdr
        core_default = inspect.signature(
            Covariance.mvdr).parameters['diagonal_loading'].default
        sonar_default = inspect.signature(
            mvdr).parameters['diagonal_loading'].default
        assert (core_default, sonar_default) == (1e-6, 1e-2)

    def _blank(self, rep):
        """``rep`` with candidate point 1 emptied: an unpopulated ``.rpo``
        cell."""
        replicas = np.array(rep.replicas)
        replicas[0, 1] = 0.0
        return Replicas(replicas=replicas, candidates=rep.candidates,
                        model='OASN', frequencies=200.0)

    def test_an_empty_replica_cell_is_no_data_not_a_unit_peak(self):
        """An unpopulated ``.rpo`` cell has no energy, so ``wᴴC⁻¹w`` is 0.
        Reporting a finite power there puts a fabricated peak in the surface;
        1.0 is the global maximum of this one."""
        K, _cov, rep = self._rig()
        cov = Covariance(covariance=K[None], model='OASN', frequencies=200.0)
        out = cov.mvdr(self._blank(rep)).data.ravel()
        assert np.isnan(out[1]), f"empty replica returned {out[1]!r}"
        assert np.isfinite(np.delete(out, 1)).all()

    def test_both_entry_points_call_an_empty_cell_no_data(self):
        from uacpy.sonar.matched_field import mvdr
        K, _cov, rep = self._rig()
        blank = self._blank(rep)
        cov = Covariance(covariance=K[None], model='OASN', frequencies=200.0)
        assert np.isnan(cov.mvdr(blank).data.ravel()[1])
        assert np.isnan(mvdr(K, blank).data.ravel()[1])


class TestTheArrayFunctionsScaleEachRowToUnitNorm:
    """``bartlett`` and ``mvdr`` score the DIRECTION of each weight row: a row
    and three times it score alike, and a row of zeros — a candidate a
    forward model put no energy at — scores zero (Bartlett) or is undefined
    (MVDR), never a NaN from 0/0 or a finite peak."""

    def _rig(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(5, 40)) + 1j * rng.normal(size=(5, 40))
        K = x @ x.conj().T / 40
        rows = rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5))
        return K, rows

    def test_a_scaled_row_scores_as_the_row(self):
        from uacpy.acoustic_signal import bartlett, mvdr
        K, rows = self._rig()
        for fn in (bartlett, mvdr):
            np.testing.assert_allclose(fn(K, 3.0 * rows), fn(K, rows),
                                       rtol=1e-12)

    def test_a_row_of_zeros_scores_zero_or_undefined(self):
        from uacpy.acoustic_signal import bartlett, mvdr
        K, rows = self._rig()
        rows[2] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert bartlett(K, rows)[2] == 0.0
            assert np.isnan(mvdr(K, rows)[2])
            assert np.isfinite(np.delete(mvdr(K, rows), 2)).all()


class TestTheAmbiguityFieldKeepsItsReference:
    """The wrappers return dB re the surface's peak and keep that peak as
    ``reference``, in ``reference_unit``, so the linear surface of the
    array-level function comes back from the result alone."""

    def test_the_covariance_surfaces_are_the_array_functions(self):
        from uacpy.acoustic_signal import bartlett, mvdr
        _cov, rep, _ = _synthetic_mfp()
        cov = Covariance(covariance=_cov.covariance, frequencies=200.0,
                         unit='Pa²/Hz')
        for field, linear in (
                (cov.bartlett(rep), bartlett(cov.covariance, rep.replicas)),
                (cov.mvdr(rep), mvdr(cov.covariance, rep.replicas))):
            assert field.reference_unit == 'Pa²/Hz'
            assert field.reference == pytest.approx(np.nanmax(linear),
                                                    rel=1e-15)
            np.testing.assert_allclose(_linear(field), linear, rtol=1e-12)

    def test_the_sonar_surfaces_are_the_array_functions(self):
        from uacpy import sonar
        from uacpy.acoustic_signal import bartlett, mvdr
        cov, rep, _ = _synthetic_mfp()
        K, rows = cov.covariance[0], rep.replicas[0]
        b, m = sonar.bartlett(K, rep), sonar.mvdr(K, rep)
        assert (b.reference_unit, m.reference_unit) == ('1', '1')
        np.testing.assert_allclose(
            _linear(b), bartlett(K, rows, normalize='trace'), rtol=1e-12)
        np.testing.assert_allclose(
            _linear(m), mvdr(K, rows, diagonal_loading=1e-2,
                             normalize='max'), rtol=1e-12)
        assert m.reference == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Wave-5 mutation-campaign killing tests (2026-08-18 rerun): each class
# below pins a contract a surviving mutant showed to be untested.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# core/_beamforming.py — MVDR diagonal-loading magnitude
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadedInverseLoadingMagnitude:
    """``loaded_inverse`` adds ``loading`` *fractions of the average
    eigenvalue* ``tr(R)/N`` to the diagonal: for R = diag(2, 4) and
    loading = 0.5 the average eigenvalue is 3, so the loaded matrix is
    diag(3.5, 5.5) exactly."""

    def test_loading_is_a_fraction_of_the_average_eigenvalue(self):
        from uacpy.core._beamforming import loaded_inverse
        R = np.diag([2.0, 4.0]).astype(complex)
        got = loaded_inverse(R, loading=0.5)
        np.testing.assert_allclose(
            got, np.diag([1.0 / 3.5, 1.0 / 5.5]), rtol=1e-12)

    def test_zero_loading_is_the_plain_inverse(self):
        from uacpy.core._beamforming import loaded_inverse
        R = np.array([[2.0, 0.5], [0.5, 1.0]], dtype=complex)
        np.testing.assert_allclose(
            loaded_inverse(R, loading=0.0) @ R, np.eye(2), atol=1e-12)
