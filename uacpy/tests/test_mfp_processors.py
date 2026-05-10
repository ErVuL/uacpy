"""Tests for the MFP processors on :class:`Covariance`."""

import numpy as np
import pytest

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
    # Covariance: outer product of truth + noise
    C = np.outer(truth, truth.conj())
    C = C + noise_level * (
        rng.normal(size=(n_rcv, n_rcv))
        + 1j * rng.normal(size=(n_rcv, n_rcv))
    )
    C = C + C.conj().T  # symmetrise (Hermitian)
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
        with pytest.raises(ValueError, match="frequency mismatch"):
            cov.bartlett(bad)

    def test_receiver_count_mismatch_raises(self):
        cov, rep, _ = _synthetic_mfp(n_rcv=8)
        bad = Replicas(
            replicas=np.zeros((1, 5, 4, 1, 6), dtype=complex),
            replica_z=rep.replica_z, replica_x=rep.replica_x,
            replica_y=rep.replica_y, model='Test',
            frequencies=200.0,
        )
        with pytest.raises(ValueError, match="receiver-count mismatch"):
            cov.mvdr(bad)
