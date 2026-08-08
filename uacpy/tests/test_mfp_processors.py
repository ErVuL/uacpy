"""Tests for the MFP processors on :class:`Covariance`."""

import numpy as np
import pytest
from uacpy.core.exceptions import ConfigurationError

from uacpy.core.results import Covariance, Replicas


def _synthetic_mfp(n_rcv=8, n_zr=5, n_xr=4, n_yr=1, source_idx=(2, 2, 0),
                   noise_level=0.05, freq=200.0):
    """Build a covariance + replica pair where the source sits at
    ``source_idx``. Each receiver gets a complex unit-magnitude weight
    that varies across the candidate (z, x, y) grid."""
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
        replica_z=np.linspace(20.0, 80.0, n_zr),
        replica_x=np.linspace(500.0, 2000.0, n_xr),
        replica_y=np.linspace(0.0, 0.0, n_yr),
        model='Test', frequencies=freq,
    )
    return cov, rep, source_idx


class TestBartlett:
    def test_peaks_at_true_source(self):
        cov, rep, src = _synthetic_mfp(noise_level=0.01)
        amb = cov.bartlett(rep)
        assert amb.shape == (1, 5, 4, 1)
        peak = np.unravel_index(np.argmax(amb[0]), amb.shape[1:])
        assert peak == src

    def test_real_valued_output(self):
        cov, rep, _ = _synthetic_mfp()
        amb = cov.bartlett(rep)
        assert amb.dtype == float
        assert np.all(np.isfinite(amb))


class TestMVDR:
    def test_peaks_at_true_source(self):
        cov, rep, src = _synthetic_mfp(noise_level=0.01)
        amb = cov.mvdr(rep, diagonal_loading=1e-2)
        peak = np.unravel_index(np.argmax(amb[0]), amb.shape[1:])
        assert peak == src

    def test_diagonal_loading_smooths(self):
        cov, rep, _ = _synthetic_mfp(noise_level=0.1)
        # Heavier loading flattens the surface. Compare standard
        # deviation normalised by mean (coefficient of variation) so
        # the metric is independent of overall scale.
        loose = cov.mvdr(rep, diagonal_loading=1e-3)
        loaded = cov.mvdr(rep, diagonal_loading=10.0)
        loose_cv = loose.std() / abs(loose.mean())
        loaded_cv = loaded.std() / abs(loaded.mean())
        assert loaded_cv < loose_cv


class TestMVDRHeavyLoading:
    def test_peaks_at_true_source(self):
        cov, rep, src = _synthetic_mfp(noise_level=0.05)
        amb = cov.mvdr(rep, diagonal_loading=0.05)
        peak = np.unravel_index(np.argmax(amb[0]), amb.shape[1:])
        assert peak == src

    def test_heavy_loading_correlates_with_bartlett(self):
        # Heavy diagonal loading collapses the MVDR surface onto the
        # Bartlett surface (the constraint becomes inactive).
        cov, rep, _ = _synthetic_mfp(noise_level=0.1)
        bart = cov.bartlett(rep)[0].ravel()
        loaded = cov.mvdr(rep, diagonal_loading=100.0)[0].ravel()
        r = np.corrcoef(bart, loaded)[0, 1]
        assert r > 0.95


class TestShapeChecks:
    def test_freq_mismatch_raises(self):
        cov, rep, _ = _synthetic_mfp()
        bad = Replicas(
            replicas=np.zeros((2, 5, 4, 1, 8), dtype=complex),
            replica_z=rep.replica_z, replica_x=rep.replica_x,
            replica_y=rep.replica_y, model='Test',
            frequencies=np.array([200.0, 400.0]),
        )
        with pytest.raises(ConfigurationError, match="frequency mismatch"):
            cov.bartlett(bad)

    def test_receiver_count_mismatch_raises(self):
        cov, rep, _ = _synthetic_mfp(n_rcv=8)
        bad = Replicas(
            replicas=np.zeros((1, 5, 4, 1, 6), dtype=complex),
            replica_z=rep.replica_z, replica_x=rep.replica_x,
            replica_y=rep.replica_y, model='Test',
            frequencies=200.0,
        )
        with pytest.raises(ConfigurationError, match="receiver-count mismatch"):
            cov.mvdr(bad)


class TestTheTwoMfpEntryPointsAgree:
    """``Covariance.bartlett``/``.mvdr`` and ``uacpy.sonar``'s free functions
    are the same two processors over different input conventions (OASN
    covariance vs measured CSDM; replica index last vs first; unnormalised vs
    normalised). They must stay numerically identical up to the normalisation
    each documents — a drift in either implementation is a real defect.
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
            replica_z=np.arange(self.N_PTS, dtype=float),
            replica_x=np.array([0.0]), replica_y=np.array([0.0]),
            model='OASN', frequencies=200.0,
        )
        return K, E.reshape(self.N_RCV, self.N_PTS, 1, 1), cov, rep

    def test_bartlett_differs_by_exactly_the_covariance_trace(self):
        from uacpy.sonar.matched_field import bartlett
        K, bank, cov, rep = self._rig()
        core = cov.bartlett(rep).ravel()
        sonar = bartlett(K, bank).ravel()
        np.testing.assert_allclose(core, sonar * np.real(np.trace(K)),
                                   rtol=1e-12)

    @pytest.mark.parametrize('loading', [1e-6, 1e-3, 1e-2])
    def test_mvdr_differs_by_exactly_the_surface_max(self, loading):
        from uacpy.sonar.matched_field import mvdr
        K, bank, cov, rep = self._rig()
        core = cov.mvdr(rep, diagonal_loading=loading).ravel()
        sonar = mvdr(K, bank, loading=loading).ravel()
        np.testing.assert_allclose(core / core.max(), sonar, rtol=1e-10)

    def test_the_two_defaults_are_the_documented_pair(self):
        """Different on purpose — OASN's covariance is full rank, a measured
        few-snapshot CSDM is not. Pinned so a change has to be deliberate."""
        import inspect
        from uacpy.sonar.matched_field import mvdr
        core_default = inspect.signature(
            Covariance.mvdr).parameters['diagonal_loading'].default
        sonar_default = inspect.signature(mvdr).parameters['loading'].default
        assert (core_default, sonar_default) == (1e-6, 1e-2)

    def test_an_empty_replica_cell_is_no_data_not_a_unit_peak(self):
        """An unpopulated ``.rpo`` cell has no energy, so ``wᴴC⁻¹w`` is 0.
        Reporting a finite power there puts a fabricated peak in the surface —
        and 1.0 would have been the global maximum of this one."""
        K, _bank, _cov, rep = self._rig()
        replicas = np.array(rep.replicas)
        replicas[0, 1] = 0.0                        # candidate point 1: no data
        blank = Replicas(
            replicas=replicas, replica_z=rep.replica_z,
            replica_x=rep.replica_x, replica_y=rep.replica_y,
            model='OASN', frequencies=200.0,
        )
        cov = Covariance(covariance=K[None], model='OASN', frequencies=200.0)
        out = cov.mvdr(blank).ravel()
        assert np.isnan(out[1]), f"empty replica returned {out[1]!r}"
        assert np.isfinite(np.delete(out, 1)).all()

    def test_both_entry_points_call_an_empty_cell_no_data(self):
        from uacpy.sonar.matched_field import mvdr
        K, bank, _cov, rep = self._rig()
        bank = bank.copy()
        bank[:, 1] = 0.0
        replicas = np.array(rep.replicas)
        replicas[0, 1] = 0.0
        cov = Covariance(covariance=K[None], model='OASN', frequencies=200.0)
        blank = Replicas(
            replicas=replicas, replica_z=rep.replica_z,
            replica_x=rep.replica_x, replica_y=rep.replica_y,
            model='OASN', frequencies=200.0,
        )
        assert np.isnan(cov.mvdr(blank).ravel()[1])
        assert np.isnan(mvdr(K, bank).ravel()[1])
