"""Tests for ``uacpy.acoustic_signal.transforms`` — tau-p / Radon / f-k
transforms and their inverses (functional API)."""

import numpy as np
import pytest

from uacpy.acoustic_signal import (
    fk_transform,
    inverse_fk,
    inverse_radon,
    inverse_taup,
    radon_transform,
    taup_transform,
)
from uacpy.core.exceptions import ConfigurationError

FS = 1000.0
NT, NX, DX = 512, 48, 10.0
_t = np.arange(NT) / FS
_x = np.arange(NX) * DX


def _ricker(tc, F=40.0):
    u = 2 * np.pi * F * (_t - tc)
    return (1 - 0.5 * u ** 2) * np.exp(-0.25 * u ** 2)


def _linear_gather(p0, tau0):
    g = np.zeros((NT, NX))
    for ix in range(NX):
        g[:, ix] += _ricker(tau0 + p0 * _x[ix])
    return g


def _hyperbolic_gather(v0, tau0):
    g = np.zeros((NT, NX))
    for ix in range(NX):
        g[:, ix] += _ricker(np.sqrt(tau0 ** 2 + (_x[ix] / v0) ** 2))
    return g


class TestTauP:
    def test_focuses_linear_event(self):
        p0, tau0 = 1 / 1500.0, 0.05
        p, tau, U = taup_transform(_linear_gather(p0, tau0), FS, DX,
                                   p_max=1 / 1000.0, n_slowness=241)
        i, j = np.unravel_index(np.argmax(np.abs(U)), U.shape)
        assert p[i] == pytest.approx(p0, abs=p[1] - p[0])
        assert tau[j] == pytest.approx(tau0, abs=2 / FS)

    def test_compute_then_standalone_inverse(self):
        g = _linear_gather(1 / 1500.0, 0.05)
        slow, tau, U = taup_transform(g, FS, DX, p_max=1 / 1000.0, n_slowness=241)
        rec = inverse_taup(U, slow, FS, DX, NX)
        assert rec.shape == g.shape
        # inverse_taup is the adjoint slant stack, not an exact inverse: a
        # finite slowness fan and 48 traces leave amplitude taper and aliasing
        # fringes. 0.85 asks that the event be recovered in shape and position,
        # which is what the adjoint promises; it is not a round-trip bound.
        # The same figure governs the two other reconstruction checks below.
        assert np.corrcoef(g.ravel(), rec.ravel())[0, 1] > 0.85

    def test_requires_2d(self):
        with pytest.raises(ConfigurationError):
            taup_transform(np.zeros(10), FS, DX)

    def test_zero_pad_and_window_focus(self):
        p0, tau0 = 1 / 1500.0, 0.05
        p, tau, U = taup_transform(_linear_gather(p0, tau0), FS, DX,
                                   p_max=1 / 1000.0, n_slowness=241,
                                   window="hann", nfft=1024)
        assert U.shape[1] == 1024 and tau.size == 1024
        i, j = np.unravel_index(np.argmax(np.abs(U)), U.shape)
        assert p[i] == pytest.approx(p0, abs=p[1] - p[0])
        assert tau[j] == pytest.approx(tau0, abs=2 / FS)

    def test_nfft_truncation_raises(self):
        with pytest.raises(ConfigurationError):
            taup_transform(np.zeros((512, 8)), FS, DX, nfft=256)


class TestRadon:
    def test_linear_peak_slowness(self):
        p0 = 1 / 1500.0
        mv = np.linspace(-1e-3, 1e-3, 241)
        m, tau, R = radon_transform(_linear_gather(p0, 0.05), FS, DX, mv,
                                    kind='linear')
        assert m[np.argmax(np.abs(R)) // R.shape[1]] == pytest.approx(
            p0, abs=mv[1] - mv[0])

    def test_hyperbolic_peak_velocity(self):
        v0, tau0 = 1500.0, 0.08
        vels = np.linspace(1200, 2000, 81)
        m, tau, R = radon_transform(_hyperbolic_gather(v0, tau0), FS, DX, vels,
                                    kind='hyperbolic')
        i, j = np.unravel_index(np.argmax(np.abs(R)), R.shape)
        # Two velocity nodes and three time samples of slack: the hyperbolic
        # apex is flat in v near the true value, so the focus straddles nodes
        # more than the linear case above (one node, two samples) does.
        assert m[i] == pytest.approx(v0, abs=2 * (vels[1] - vels[0]))
        assert tau[j] == pytest.approx(tau0, abs=3 / FS)

    def test_parabolic_runs(self):
        g = _linear_gather(1 / 2000.0, 0.05)
        q = np.linspace(-1e-6, 1e-6, 41)
        m, tau, R = radon_transform(g, FS, DX, q, kind='parabolic')
        assert R.shape == (41, NT)

    def test_inverse_reconstructs(self):
        g = _linear_gather(1 / 1500.0, 0.05)
        mv = np.linspace(-1e-3, 1e-3, 241)
        _, _, R = radon_transform(g, FS, DX, mv, kind='linear')
        back = inverse_radon(R, FS, DX, mv, NX, kind='linear')
        assert back.shape == g.shape
        assert np.corrcoef(g.ravel(), back.ravel())[0, 1] > 0.85

    def test_bad_kind_raises(self):
        with pytest.raises(ConfigurationError):
            radon_transform(np.zeros((NT, NX)), FS, DX, [1.0], kind='cubic')


class TestFKInverse:
    def test_exact_round_trip(self):
        g = _linear_gather(1 / 1500.0, 0.05)
        _, _, _, spectrum = fk_transform(g, FS, DX)
        assert np.max(np.abs(inverse_fk(spectrum) - g)) < 1e-9

    def test_calibrated_psd_parseval(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((256, 64))
        f, k, psd, _ = fk_transform(data, FS, DX, normalize=True)
        lhs = psd.sum() * (f[1] - f[0]) * (k[1] - k[0])
        assert lhs == pytest.approx(np.mean(data ** 2), rel=1e-9)


class TestFKWindowing:
    def test_windowed_psd_parseval(self):
        from scipy.signal import get_window
        rng = np.random.default_rng(0)
        data = rng.standard_normal((256, 64))
        wt = get_window("hann", 256, fftbins=True)
        wx = get_window("hann", 64, fftbins=True)
        f, k, psd, _ = fk_transform(data, FS, DX, normalize=True, window="hann")
        lhs = psd.sum() * (f[1] - f[0]) * (k[1] - k[0])
        expected = (np.sum(data ** 2 * wt[:, None] ** 2 * wx[None, :] ** 2)
                    / (np.sum(wt ** 2) * np.sum(wx ** 2)))
        assert lhs == pytest.approx(expected, rel=1e-9)

    def test_per_axis_window_list(self):
        data = _linear_gather(1 / 1500.0, 0.05)
        f, k, panel, _ = fk_transform(data, FS, DX,
                                      window=["hann", ("kaiser", 8)])
        assert panel.shape == data.shape

    def test_bad_window_list_raises(self):
        with pytest.raises(ConfigurationError):
            fk_transform(np.zeros((32, 8)), FS, DX, window=["hann"])

    def test_zero_pad_grows_axes_pad_independent(self):
        rng = np.random.default_rng(1)
        data = rng.standard_normal((256, 64))
        f, k, psd, _ = fk_transform(data, FS, DX, normalize=True, nfft=(512, 128))
        assert psd.shape == (512, 128)
        assert f.size == 512 and k.size == 128
        lhs = psd.sum() * (f[1] - f[0]) * (k[1] - k[0])
        assert lhs == pytest.approx(np.mean(data ** 2), rel=1e-9)

    def test_nfft_truncation_raises(self):
        with pytest.raises(ConfigurationError):
            fk_transform(np.zeros((256, 64)), FS, DX, nfft=(128, 32))

    def test_compute_requires_2d(self):
        with pytest.raises(ConfigurationError):
            fk_transform(np.zeros(10), FS, DX)


class TestBringYourOwn:
    """Inverse a transform panel obtained elsewhere (no prior forward call)."""

    def test_taup_functional_round_trip(self):
        g = _linear_gather(1 / 1500.0, 0.05)
        slow, tau, U = taup_transform(g, FS, DX, p_max=1 / 1000.0, n_slowness=241)
        rec = inverse_taup(U, slow, FS, DX, NX)
        assert rec.shape == g.shape
        assert np.corrcoef(g.ravel(), rec.ravel())[0, 1] > 0.85

    def test_fk_inverse_standalone(self):
        g = _linear_gather(1 / 1500.0, 0.05)
        # A hand-built panel must use fk_transform's layout: fftshifted, and
        # with the spatial axis on the k = ω/c sign — the negative of
        # np.fft.fft2's exp(-i2πνx), so column ν holds -ν.
        F = np.fft.fft2(g)
        my_FK = np.fft.fftshift(np.roll(F[:, ::-1], 1, axis=1), axes=(0, 1))
        assert np.max(np.abs(inverse_fk(my_FK) - g)) < 1e-9

    def test_inverse_taup_bad_shape_raises(self):
        with pytest.raises(ConfigurationError):
            inverse_taup(np.zeros((3, 10)), np.array([1.0, 2.0]), FS, DX, NX)


class TestRadonHyperbolicMoveoutValidation:
    def test_forward_rejects_nonpositive_velocity(self):
        with pytest.raises(ConfigurationError, match="> 0"):
            radon_transform(np.zeros((NT, NX)), FS, DX, [0.0, 1500.0],
                            kind='hyperbolic')

    def test_inverse_rejects_nonpositive_velocity(self):
        with pytest.raises(ConfigurationError, match="> 0"):
            inverse_radon(np.zeros((2, NT)), FS, DX, [-1.0, 1500.0], NX,
                          kind='hyperbolic')

    def test_positive_velocities_yield_finite_panel(self):
        m, tau, R = radon_transform(_hyperbolic_gather(1500.0, 0.08), FS, DX,
                                    np.linspace(1200, 2000, 5),
                                    kind='hyperbolic')
        assert np.all(np.isfinite(R))


class TestRadonAdjointness:
    """``inverse_radon`` is the exact transpose of ``radon_transform``.

    It scatters each Radon sample along the moveout curve with the forward
    interpolation's own weights. Gathering instead — reading the curve back
    with ``np.interp(taus, tt, ...)`` — only agrees with the transpose when the
    moveout is a pure time shift; a hyperbolic curve compresses near ``tau=0``
    and returns early samples at a fraction of their forward weight.
    """

    NT_S, NX_S, FS_S, DX_S = 32, 4, 100.0, 10.0

    KINDS = [('linear', np.array([-1e-3, 0.0, 1e-3])),
             ('parabolic', np.array([-1e-5, 1e-5])),
             ('hyperbolic', np.array([900.0, 1500.0]))]

    def _operator(self, kind, mv):
        """Explicit forward matrix ``L`` and adjoint matrix ``A``."""
        nt, nx = self.NT_S, self.NX_S
        nm = mv.size
        L = np.zeros((nt * nm, nt * nx))
        for j in range(nt * nx):
            e = np.zeros(nt * nx)
            e[j] = 1.0
            L[:, j] = radon_transform(e.reshape(nt, nx), self.FS_S, self.DX_S,
                                      mv, kind=kind).panel.ravel()
        A = np.zeros((nt * nx, nt * nm))
        for j in range(nt * nm):
            e = np.zeros(nt * nm)
            e[j] = 1.0
            A[:, j] = inverse_radon(e.reshape(nm, nt), self.FS_S, self.DX_S,
                                    mv, nx, kind=kind).ravel()
        return L, A

    @pytest.mark.parametrize("kind,mv", KINDS)
    def test_adjoint_matrix_equals_forward_transpose(self, kind, mv):
        L, A = self._operator(kind, mv)
        assert np.max(np.abs(A - L.T)) < 1e-12

    @pytest.mark.parametrize("kind,mv", KINDS)
    def test_dot_product_test(self, kind, mv):
        """``<L x, y> == <x, A y>`` on random data — what an iterative
        least-squares (sparse Radon) solver needs of the pair."""
        rng = np.random.default_rng(11)
        x = rng.standard_normal((self.NT_S, self.NX_S))
        y = rng.standard_normal((mv.size, self.NT_S))
        lhs = np.sum(radon_transform(x, self.FS_S, self.DX_S, mv,
                                     kind=kind).panel * y)
        rhs = np.sum(x * inverse_radon(y, self.FS_S, self.DX_S, mv,
                                       self.NX_S, kind=kind))
        assert lhs == pytest.approx(rhs, rel=1e-12)

    def test_hyperbolic_column_sums_match_the_forward_weights(self):
        """Every output sample receives exactly the weight the forward
        transform took from it. The gather form left these between 1.0 and
        3.2 instead of the nx = 4 the forward operator spreads."""
        mv = np.array([1500.0])
        L, A = self._operator('hyperbolic', mv)
        assert np.allclose(A.sum(axis=0), L.sum(axis=1))

    def test_scatter_keeps_the_tau_zero_contribution(self):
        """``tau = 0`` survives a moveout that lands a few ULPs off the grid.

        With ``fs = 10`` and ``p·x = 0.1·3 = 0.30000000000000004`` the moveout
        time sits just above ``taus[3] = 0.3``, which the gather form's
        ``np.interp(..., left=0.0)`` read as out of range and zeroed.
        """
        fs, dx, nt, nx = 10.0, 3.0, 8, 2
        assert 0.1 * dx > 3 / fs                     # the ULP overshoot
        R = np.zeros((1, nt))
        R[0, 0] = 1.0                                # tau = 0 only
        out = inverse_radon(R, fs, dx, np.array([0.1]), nx, kind='linear')
        assert out[3, 1] == pytest.approx(1.0, rel=1e-12)


class TestInverseArgumentOrder:
    """``inverse_taup`` takes its slowness axis second, ``inverse_radon`` its
    moveout axis fourth; passing one order to the other is caught by name."""

    def test_radon_order_into_inverse_taup_raises(self):
        with pytest.raises(ConfigurationError, match="must be a scalar"):
            inverse_taup(np.zeros((3, 64)), FS, DX,
                         np.array([-1e-3, 0.0, 1e-3]), NX)

    def test_taup_order_into_inverse_radon_raises(self):
        with pytest.raises(ConfigurationError, match="must be a scalar"):
            inverse_radon(np.zeros((3, 64)), np.array([-1e-3, 0.0, 1e-3]),
                          FS, DX, NX)

    def test_mismatched_axis_length_names_the_signature(self):
        with pytest.raises(ConfigurationError, match="comes second"):
            inverse_taup(np.zeros((3, 10)), np.array([1.0, 2.0]), FS, DX, NX)
        with pytest.raises(ConfigurationError, match="comes fourth"):
            inverse_radon(np.zeros((3, 10)), FS, DX, np.array([1.0, 2.0]), NX)


class TestReferenceOffset:
    """``taup_transform`` walks the same ``t = tau + p·(x - x0)`` moveout curve
    as ``radon_transform(kind='linear')``, so it takes the same ``x0``."""

    def test_taup_x0_matches_radon_x0(self):
        g = _linear_gather(1 / 1500.0, 0.05)
        ps = np.linspace(-1e-3, 1e-3, 121)
        tp = taup_transform(g, FS, DX, ps, x0=100.0).panel
        matched = radon_transform(g, FS, DX, ps, kind='linear', x0=100.0).panel
        mismatched = radon_transform(g, FS, DX, ps, kind='linear').panel
        # The two use different interpolators (frequency-domain sinc vs linear),
        # so they correlate strongly rather than agreeing sample-by-sample; the
        # point is that x0 has to match for them to track at all.
        assert np.corrcoef(tp.ravel(), matched.ravel())[0, 1] > 0.85
        assert np.corrcoef(tp.ravel(), mismatched.ravel())[0, 1] < 0.5

    def test_taup_x0_shifts_the_panel_by_p_times_x0(self):
        g = _linear_gather(1 / 1500.0, 0.05)
        ps = np.array([1e-3])
        base = taup_transform(g, FS, DX, ps).panel[0]
        moved = taup_transform(g, FS, DX, ps, x0=50.0).panel[0]
        assert np.argmax(moved) - np.argmax(base) == pytest.approx(
            round(1e-3 * 50.0 * FS), abs=1)

    def test_inverse_taup_stays_adjoint_under_x0(self):
        rng = np.random.default_rng(7)
        nt, nx = 64, 8
        d = rng.standard_normal((nt, nx))
        ps = np.linspace(-1e-3, 1e-3, 9)
        y = rng.standard_normal((ps.size, nt))
        lhs = np.sum(taup_transform(d, FS, DX, ps, x0=20.0).panel * y)
        rhs = np.sum(d * inverse_taup(y, ps, FS, DX, nx, x0=20.0))
        assert lhs == pytest.approx(rhs, rel=1e-10)


def _adjoint_delay_kernel(nt, m):
    """One output trace of ``inverse_taup`` for a unit tau-p sample delayed by
    ``m`` samples, evaluated from the transform's definition.

    ``inverse_taup`` takes the rfft of the tau axis, multiplies bin ``k`` by
    ``exp(-i·omega_k·p·x)`` and irffts. A unit sample at ``tau0`` has rfft
    ``exp(-i·omega_k·tau0)``, so the spectrum handed to the irfft is
    ``exp(-2πi·k·m/nt)`` with ``m = (tau0 + p·x)·fs`` — a pure delay of ``m``
    samples. Undoing that half-spectrum by hand for even ``nt`` (bins ``1 …
    nt/2-1`` conjugate-mirrored, DC and Nyquist taken once) gives

        w[n] = (1 + 2·Σ_{k=1}^{nt/2-1} cos(k·φ) + cos(nt·φ/2)) / nt ,
        φ = 2π(n - m)/nt

    which is the Dirichlet kernel ``sin(πd) / (nt·tan(πd/nt))``, ``d = n - m``:
    unit amplitude at ``d = 0``, zero at every other integer, periodic in
    ``nt``. The cosine sum is used rather than that closed form only because it
    is finite at ``d = 0``; both are written from the definition above, not
    read off the implementation.
    """
    phi = 2.0 * np.pi * (np.arange(nt) - m) / nt
    k = np.arange(1, nt // 2)
    return (1.0 + 2.0 * np.cos(np.outer(phi, k)).sum(axis=1)
            + np.cos(nt * phi / 2.0)) / nt


class TestTauPAbsoluteScale:
    """The absolute gain of the tau-p pair, which adjointness alone leaves free.

    ``TestReferenceOffset::test_inverse_taup_stays_adjoint_under_x0`` pins the
    forward transform only *against* the inverse: multiplying both by the same
    constant preserves ``<L d, y> == <d, A y>`` exactly, so the pair's joint
    scale passes it unchallenged. These two fix the adjoint's gain against the
    closed form above, and the adjointness test then carries that to the
    forward. A Parseval check would not do the job — the adjoint slant stack
    sums ``n_slowness`` moveout lines into each trace and is not
    energy-preserving.
    """

    NT_S, FS_S, P = 64, 1000.0, 1e-3

    def test_unit_sample_scatters_unit_weight_along_its_moveout(self):
        """On-grid geometry: ``p·(x - x0)·fs`` is a whole number of samples for
        every trace, where the kernel is an exact delta, so each trace gets
        exactly 1.0 at its moveout sample and 0 everywhere else."""
        nx, dx, x0, j0 = 3, 20.0, 20.0, 20
        u = np.zeros((1, self.NT_S))
        u[0, j0] = 1.0
        out = inverse_taup(u, np.array([self.P]), self.FS_S, dx, nx, x0=x0)
        for ix in range(nx):
            m = j0 + self.P * (ix * dx - x0) * self.FS_S
            assert m == round(m)                     # 0, 20, 40 samples
            want = np.zeros(self.NT_S)
            want[round(m)] = 1.0
            # The input sample is exactly 1.0, so 1e-12 is measured against the
            # unit weight the adjoint is being pinned to.
            assert np.max(np.abs(out[:, ix] - want)) < 1e-12

    def test_off_grid_moveout_scatters_the_dirichlet_kernel(self):
        """Fractional geometry: no trace's delay is a whole sample, so the
        whole trace — not just its peak — carries the band-limited kernel, and
        an interpolation change moves it as surely as a gain change does."""
        nx, dx, x0, j0 = 4, 6.4, 1.6, 20
        u = np.zeros((1, self.NT_S))
        u[0, j0] = 1.0
        out = inverse_taup(u, np.array([self.P]), self.FS_S, dx, nx, x0=x0)
        for ix in range(nx):
            m = j0 + self.P * (ix * dx - x0) * self.FS_S
            assert m != round(m)                     # 18.4, 24.8, 31.2, 37.6
            assert np.max(np.abs(out[:, ix]
                                 - _adjoint_delay_kernel(self.NT_S, m))) < 1e-12
