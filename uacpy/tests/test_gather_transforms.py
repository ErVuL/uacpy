"""Gather transforms in ``uacpy.acoustic_signal.arrays`` — tau-p, Radon, f-k.

The three ways a multichannel gather is re-expressed: slowness (tau-p),
moveout curvature (Radon) and frequency-wavenumber (f-k), each with its
inverse. What is pinned here is what a caller depends on and cannot see:

* **Round trips.** Every forward/inverse pair returns the gather it was given,
  which is the only statement that covers the scaling, the axis order and the
  wavenumber sign at once.
* **The sign convention.** ``k = omega / c`` for a wave travelling towards
  increasing ``x``, the same direction tau-p and Radon call positive, so a
  panel read from one transform can be compared with another.
* **Absolute scale.** tau-p is checked against the amplitude a hand-summed
  slant stack gives, not merely against itself.
* **Refusals.** A gather whose spacing, shape or wavenumber axis the transform
  cannot represent is refused by name.

The figures these produce are exercised in ``test_transform_plots.py``.
"""

import warnings

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

# ── f-k: the single-segment transform, its averaging, and its sign ──────────

def _gather(seed=0):
    rng = np.random.default_rng(seed)
    nt, nx = 128, 32
    t = np.arange(nt)
    x = np.arange(nx)
    return (np.cos(2 * np.pi * (0.1 * t[:, None] - 0.05 * x[None, :]))
            + rng.standard_normal((nt, nx)))


def _fft2_on_fk_convention(d, shape=None):
    """``np.fft.fft2`` re-indexed onto ``fk_transform``'s wavenumber sign.

    ``fft2``'s ``exp(-i2πνx)`` kernel puts a +x-travelling wave on ``ω = -c·k``;
    ``fk_transform`` reports the package-wide ``k = ω/c``, so its spatial axis
    is the negative of ``fft2``'s — column ``ν`` holds ``-ν``.
    """
    F = np.fft.fft2(d, s=shape)
    return np.fft.fftshift(np.roll(F[:, ::-1], 1, axis=1), axes=(0, 1))


def test_fk_single_segment_matches_direct_fft():
    d = _gather()
    fs, dx = 1000.0, 5.0
    nt, nx = d.shape
    FKc = _fft2_on_fk_convention(d)
    p0 = np.abs(FKc) ** 2
    f0 = np.fft.fftshift(np.fft.fftfreq(nt, d=1.0 / fs))
    # Angular wavenumber k = 2π·ν rad/m (fk_transform's convention; ν = fftfreq).
    k0 = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    f, k, power, spectrum = fk_transform(d, sample_rate=fs, dx=dx)
    assert np.allclose(f, f0) and np.allclose(k, k0)
    assert np.allclose(power, p0)
    assert np.allclose(spectrum, FKc)
    assert spectrum is not None


def test_fk_averaging_reduces_variance():
    # nperseg=32 with no overlap splits the 128-sample record into 4
    # independent segments, so Welch averaging should cut the coefficient of
    # variation of the noise floor by 1/sqrt(4) = 0.5. The 0.6 bound is that
    # theoretical factor with slack for the finite 8-seed sample.
    cv = []
    for kw in ({}, dict(nperseg=32, noverlap=0)):
        floors = []
        for s in range(8):
            _, _, p, _ = fk_transform(_gather(s), 1000.0, 5.0, **kw)
            floors.append(p[:6, :6])
        floors = np.array(floors)
        cv.append(floors.std() / floors.mean())
    assert cv[1] < cv[0] * 0.6


def test_fk_average_panel_not_invertible():
    _, _, _, spec = fk_transform(_gather(), 1000.0, 5.0, nperseg=32)
    assert spec is None


def test_inverse_fk_roundtrip_and_none_guard():
    d = _gather()
    _, _, _, spec = fk_transform(d, 1000.0, 5.0)
    rec = inverse_fk(spec)
    # A single-segment forward/inverse FFT pair is algebraically exact, so
    # 1e-10 is float round-off headroom, not a physical tolerance.
    assert np.linalg.norm(rec - d) / np.linalg.norm(d) < 1e-10
    with pytest.raises(ConfigurationError):
        inverse_fk(None)


@pytest.mark.parametrize("kw", [dict(nperseg=999),
                                dict(nperseg=32, noverlap=32),
                                dict(nperseg=32, nfft=8)])
def test_fk_validation(kw):
    with pytest.raises(ConfigurationError):
        fk_transform(_gather(), 1000.0, 5.0, **kw)


@pytest.mark.parametrize("direction", [+1, -1])
def test_fk_wavenumber_sign_is_omega_over_c(direction):
    """A plane wave of speed ``c`` towards ``+x`` peaks at ``k = +ω/c``.

    This is the ``k = ω/c`` convention the docstring states and the models use,
    and the sign directional f-k muting depends on; raw ``np.fft.fft2`` would
    put the same wave on ``ω = -c·k``.
    """
    fs, dx, nt, nx, c, f0 = 200.0, 2.0, 256, 64, 1000.0, 25.0
    t = np.arange(nt) / fs
    x = np.arange(nx) * dx
    d = np.sin(2 * np.pi * f0 * (t[:, None] - direction * x[None, :] / c))
    f, k, power, _ = fk_transform(d, fs, dx)
    pos = f > 0
    i, j = np.unravel_index(np.argmax(power[pos]), power[pos].shape)
    assert f[pos][i] == pytest.approx(f0, abs=f[1] - f[0])
    # Half a wavenumber bin of slack: the k grid steps by 2π/(nx·dx) =
    # 0.049 rad/m and the event sits at |k| = 2π·25/1000 = 0.157 rad/m.
    assert k[j] == pytest.approx(direction * 2 * np.pi * f0 / c,
                                 abs=0.5 * (k[1] - k[0]))
    assert np.sign(k[j]) == direction


def test_fk_wavenumber_sign_agrees_with_taup_and_radon():
    """The three gather transforms report the same sign for one event.

    ``taup_transform``/``radon_transform`` give a +x-travelling wave the
    slowness ``p = +1/c``; ``fk_transform`` must therefore place it at
    ``k = +ω/c``, not at ``-ω/c``.
    """
    from uacpy.acoustic_signal.arrays import (radon_transform,
                                                  taup_transform)
    fs, dx, nt, nx, c, f0 = 200.0, 2.0, 256, 64, 1000.0, 25.0
    t = np.arange(nt) / fs
    x = np.arange(nx) * dx
    d = np.sin(2 * np.pi * f0 * (t[:, None] - x[None, :] / c))

    f, k, power, _ = fk_transform(d, fs, dx)
    pos = f > 0
    _, j = np.unravel_index(np.argmax(power[pos]), power[pos].shape)
    assert k[j] > 0

    tp = taup_transform(d, fs, dx)
    assert tp.slownesses[np.argmax(np.abs(tp.panel).max(axis=1))] > 0
    ps = np.linspace(-2e-3, 2e-3, 201)
    rd = radon_transform(d, fs, dx, ps, kind="linear")
    assert rd.moveout[np.argmax(np.abs(rd.panel).max(axis=1))] > 0


@pytest.mark.parametrize("nx", [31, 32])
@pytest.mark.parametrize("pad", [False, True])
def test_inverse_fk_undoes_the_wavenumber_flip(nx, pad):
    """The spatial re-indexing is its own inverse, for odd and even ``nx``
    and with zero padding, so the round trip stays exact."""
    rng = np.random.default_rng(4)
    nt = 33
    d = rng.standard_normal((nt, nx))
    nfft = (nt + 7, nx + 5) if pad else None
    _, _, _, spec = fk_transform(d, 500.0, 2.0, nfft=nfft)
    rec = inverse_fk(spec)
    assert np.max(np.abs(rec[:nt, :nx] - d)) < 1e-10


class TestTauPSpatialAliasing:
    """The array-spacing bound on the slowness axis, and the warning for it.

    A slant stack reads the moveout only at the sensors, so between adjacent
    traces it sees ``2*pi*f*p*dx`` modulo a turn. Past half a turn ``p`` and
    ``p -/+ 1/(f*dx)`` are the same measurement. Nothing downstream can undo
    that — the wavenumber was undersampled before the transform ran — so the
    transform says so at the point where it still means something.
    """

    FS, DX, NT, NXX = 2000.0, 2.0, 1024, 24

    def _tone(self, f, p0=4e-4, dx=None):
        dx = self.DX if dx is None else dx
        t = np.arange(self.NT) / self.FS
        x = np.arange(self.NXX) * dx
        return np.cos(2 * np.pi * f * (t[:, None] - p0 * x[None, :]))

    def _warns(self, d, dx=None, **kw):
        dx = self.DX if dx is None else dx
        with pytest.warns(UserWarning, match="aliases beyond"):
            taup_transform(d, self.FS, dx, **kw)
        return True

    def _quiet(self, d, dx=None, **kw):
        dx = self.DX if dx is None else dx
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            taup_transform(d, self.FS, dx, **kw)
        return [w for w in caught if issubclass(w.category, UserWarning)] == []

    def test_the_aliasing_it_warns_about_is_real(self):
        """Not just that it fires — that the answer really is wrong.

        A monochromatic event puts EQUAL peaks at ``p0 + n/(f*dx)`` for every
        integer n (measured: the pair here agree to 1%), so which one a global
        argmax returns is a numerical coin flip and pins nothing — tau-p and
        radon disagree on it for this very gather. What is not a coin flip is
        which peak falls inside the unaliased band ``|p| <= 1/(2*f*dx)``,
        because that is the band a caller who respects the bound will scan.
        At 900 Hz p0 = +4e-4 s/m is OUTSIDE it and the in-band representative
        sits at p0 - 1/(f*dx): a scan that is correctly restricted returns a
        NEGATIVE slowness for a wave travelling towards +x, with nothing in
        the panel to say so.
        """
        f, p0 = 900.0, 4e-4
        band = 1.0 / (2.0 * f * self.DX)
        assert p0 > band                       # the event is out of band
        ps = np.linspace(-band, band, 1001)    # scan the band, as one should
        r = taup_transform(self._tone(f, p0), self.FS, self.DX, ps)
        peak = r.slownesses[np.argmax(np.abs(r.panel).max(axis=1))]
        assert peak == pytest.approx(p0 - 1.0 / (f * self.DX),
                                     abs=2 * (ps[1] - ps[0]))
        assert peak < 0.0 < p0                 # direction of travel reversed

    def test_below_the_bound_the_same_event_is_placed_correctly(self):
        """The companion: at 600 Hz, p0 = +4e-4 is INSIDE the band, so the
        in-band peak is the true one. Same gather, same fan, one variable —
        the frequency — moved across the bound."""
        f, p0 = 600.0, 4e-4
        band = 1.0 / (2.0 * f * self.DX)
        assert p0 < band                       # the event is in band
        ps = np.linspace(-band, band, 1001)
        r = taup_transform(self._tone(f, p0), self.FS, self.DX, ps)
        peak = r.slownesses[np.argmax(np.abs(r.panel).max(axis=1))]
        assert peak == pytest.approx(p0, abs=2 * (ps[1] - ps[0]))

    def test_warns_above_the_bound_and_is_quiet_below_it(self):
        """Both sides of the threshold, one variable moving. A 200 Hz tone
        passes at the 1e-3 default (no false alarm on ordinary narrowband
        data); the same gather at 900 Hz trips it."""
        assert self._quiet(self._tone(200.0))
        assert self._warns(self._tone(900.0))

    def test_the_threshold_itself_is_pinned(self):
        """Straddling p_alias on the same gather, so the boundary is the only
        thing that changes. Far-from-boundary values would pass against a
        guard placed anywhere in between."""
        d = self._tone(200.0)
        D = np.fft.rfft(d, axis=0)
        f = np.fft.rfftfreq(self.NT, 1.0 / self.FS)
        P = (np.abs(D) ** 2).sum(axis=1)
        f_edge = f[np.searchsorted(np.cumsum(P) / P.sum(), 0.99)]
        p_alias = 1.0 / (2.0 * f_edge * self.DX)
        assert self._quiet(d, p_max=p_alias * 0.999)
        assert self._warns(d, p_max=p_alias * 1.001)

    def test_every_remedy_the_message_prints_actually_clears_it(self):
        """A remedy quoted to 3 figures must work when typed back verbatim,
        so each number is rounded the safe way. Parsed out of the message
        rather than recomputed, which is what a reader does."""
        import re
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            taup_transform(self._tone(900.0), self.FS, self.DX)
        msg = str(caught[0].message)
        p_cap = float(re.search(r"Cap the slowness range at ([\d.e+-]+)",
                                msg).group(1))
        dx_fine = float(re.search(r"dx = ([\d.e+-]+) m or finer",
                                  msg).group(1))
        f_cut = float(re.search(r"low-pass the gather below ([\d.e+-]+)",
                                msg).group(1))
        assert self._quiet(self._tone(900.0), p_max=p_cap)
        assert self._quiet(self._tone(900.0, dx=dx_fine), dx=dx_fine)
        # The low-pass corner is quoted for band-limited data, which is what
        # a caller can actually produce; f99 tracks a real cut rather than a
        # single tone's leakage tail.
        rng = np.random.default_rng(3)
        w = rng.standard_normal((self.NT, self.NXX))
        W = np.fft.rfft(w, axis=0)
        W[np.fft.rfftfreq(self.NT, 1.0 / self.FS) > f_cut] = 0
        assert self._quiet(np.fft.irfft(W, n=self.NT, axis=0))

    @pytest.mark.parametrize("gather,label", [
        (np.zeros((64, 8)), "silent"),
        (np.ones((64, 8)), "DC only"),
    ])
    def test_degenerate_gathers_neither_warn_nor_raise(self, gather, label):
        """A silent gather has no energy to alias and an all-DC one has no
        frequency to alias at; both divide by zero in the naive form of the
        bound, so both are returned quietly rather than warned about."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            taup_transform(gather, self.FS, self.DX)
        assert [w for w in caught
                if issubclass(w.category, UserWarning)] == [], label

    def test_a_zero_only_slowness_axis_cannot_alias(self):
        """p = 0 stacks the traces flat, which no spacing can undersample."""
        assert self._quiet(self._tone(900.0), slownesses=np.array([0.0]))


def test_taup_has_no_spatial_padding_because_it_would_do_nothing():
    """Why there is no spatial ``nfft`` here while ``fk_transform`` has one.

    tau-p SUMS over sensors, so zero traces add zero terms and the panel is
    unchanged bit for bit. f-k TRANSFORMS across them, so the spatial FFT
    length sets ``dk = 2*pi/(NX*dx)`` and padding genuinely interpolates the
    wavenumber axis. Pinned so that a future "for symmetry" spatial nfft on
    tau-p has to confront the fact that it cannot change an answer.
    """
    fs, dx, nt, nx = 2000.0, 2.0, 256, 16
    d = np.random.default_rng(0).standard_normal((nt, nx))
    padded = np.hstack([d, np.zeros((nt, 3 * nx))])
    plain = taup_transform(d, fs, dx, p_max=2e-4)
    with_zeros = taup_transform(padded, fs, dx, p_max=2e-4)
    assert np.array_equal(plain.panel, with_zeros.panel)

    # f-k, the contrast: the same padding moves the wavenumber grid.
    k_plain = fk_transform(d, fs, dx).wavenumbers
    k_padded = fk_transform(d, fs, dx, nfft=(nt, 4 * nx)).wavenumbers
    assert k_padded.size == 4 * k_plain.size
    assert np.diff(k_padded)[0] == pytest.approx(np.diff(k_plain)[0] / 4)
