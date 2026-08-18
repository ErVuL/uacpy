"""Tests for the Padé-error grid optimiser.

Reference: Lytaev, M.S. (2023). *Mesh Optimization for the Acoustic
Parabolic Equation.* J. Mar. Sci. Eng. 11(3), 496.
https://doi.org/10.3390/jmse11030496
"""

import numpy as np
import pytest

from uacpy.models._pade_optimizer import (
    combined_error,
    grid_error,
    optimal_c0,
    numerov_error,
    optimize_grid,
    rams_dz_shear_cap,
)


class TestOptimalC0:
    """Lytaev (2023), Eq. (15)."""

    def test_water_only_30deg(self):
        """Homogeneous water + 30° angle → ≈ 1591 m/s per Table 1."""
        c0 = optimal_c0(1500.0, 1500.0, 30.0)
        assert 1590 < c0 < 1592

    def test_pekeris_water_sediment_30deg(self):
        """Eq. (15) closed form:
        ``c_min·c_max·sqrt((2 + sin²θ)/(c_min² + c_max²))`` at
        (1500, 1700, 30°) → 1687.14 m/s — between the two speeds, and pinned
        to ±1 m/s like the siblings so a formula change fails loudly."""
        c0 = optimal_c0(1500.0, 1700.0, 30.0)
        assert 1500 < c0 < 1700
        assert 1686 < c0 < 1688

    def test_inhomogeneous_30deg(self):
        """Range [1500, 1550] @ 30° → ≈ 1616 m/s per Table 4."""
        c0 = optimal_c0(1500.0, 1550.0, 30.0)
        assert 1614 < c0 < 1618

    def test_eq15_centres_the_spectrum_on_the_pade_sweet_spot(self):
        """Eq. (15) is derived "Based on equality ξ_min = -ξ_max" (§4.2) —
        the c₀ that makes the spectrum interval symmetric about ξ = 0, where
        Fig. 2 shows the Padé error smallest. Asserted through the interval
        ``optimize_grid`` itself reports, so the pin holds against the code
        path RAM consumes, not a re-derived formula."""
        c0 = optimal_c0(1480.0, 1750.0, 35.0)
        res = optimize_grid(
            freq=100.0, c_min=1480.0, c_max=1750.0, x_max=2000.0,
            c0=c0, theta_max=35.0, eps=5e-2, p=4, alpha=0.0,
        )
        assert res['xi_min'] == pytest.approx(-res['xi_max'], abs=1e-12)
        assert res['xi_max'] > 0


class TestPadeError:
    """``R(Δx, ξ)`` from Section 4.1, exercised through ``combined_error``.

    ``combined_error`` is τ(Δx, Δz) — the same Padé functional widened by the
    discretisation spread ``Δξ = h(Δz)/k₀²``. Driving Δz to a value where
    ``Δξ`` is negligible isolates the Padé term, so these assert the Section
    4.1 properties on the code path the optimiser actually calls.
    """

    # Δξ = h(Δz)/k₀² ≈ 2e-12 at k₀ = 2 — six orders below the ±1e-6 ξ window,
    # the tightest used here, so what is left is the Padé term alone.
    TINY_DZ = 1e-5
    THETA = np.deg2rad(30.0)

    def _pade_only(self, **kwargs):
        return combined_error(dz=self.TINY_DZ, theta_max=self.THETA, **kwargs)

    def test_zero_at_origin(self):
        """The Padé approximation is exact at ξ = 0."""
        e = self._pade_only(dx=10.0, k0=2.0, p=8, xi_min=-1e-6, xi_max=1e-6)
        assert e < 1e-10

    def test_grows_with_dx(self):
        """Padé error grows monotonically with the range step."""
        k0 = 2 * np.pi * 100 / 1500
        errs = [
            self._pade_only(dx=dx, k0=k0, p=6, xi_min=-0.25, xi_max=0.0)
            for dx in [1.0, 5.0, 20.0, 100.0]
        ]
        assert all(errs[i] <= errs[i + 1] + 1e-12 for i in range(len(errs) - 1))

    def test_higher_order_is_more_accurate(self):
        """Higher Padé order gives smaller error at the same Δx."""
        k0 = 2 * np.pi * 100 / 1500
        e2 = self._pade_only(dx=20.0, k0=k0, p=2, xi_min=-0.25, xi_max=0.0)
        e8 = self._pade_only(dx=20.0, k0=k0, p=8, xi_min=-0.25, xi_max=0.0)
        assert e8 < e2


class TestNumerovError:
    """``h(Δz)`` from Eq. (13)."""

    def test_4th_order_beats_2nd_order(self):
        """At small Δz the Numerov correction (α=1/12) is much more accurate."""
        k0 = 2 * np.pi * 500 / 1500
        h2 = numerov_error(dz=0.1, k0=k0, theta_max=np.deg2rad(30), alpha=0.0)
        h4 = numerov_error(dz=0.1, k0=k0, theta_max=np.deg2rad(30), alpha=1 / 12)
        assert h4 < h2 / 100  # 4th-order is at least 100× tighter at this dz

    def test_grows_with_dz(self):
        k0 = 2 * np.pi * 500 / 1500
        errs = [
            numerov_error(dz=dz, k0=k0, theta_max=np.deg2rad(30), alpha=0.0)
            for dz in [0.05, 0.1, 0.5, 1.0]
        ]
        assert all(errs[i] < errs[i + 1] for i in range(len(errs) - 1))


class TestOptimizeGrid:
    """End-to-end optimiser checks against paper Tables 1-4."""

    def test_table1_5km_default_c0(self):
        """Paper Table 1, x_max=5 km, c0=1500: dx≈10, dz≈0.08 (within ladder)."""
        res = optimize_grid(
            freq=500.0, c_min=1500.0, c_max=1500.0, x_max=5000.0,
            c0=1500.0, theta_max=30.0, eps=1e-3, p=8, alpha=1 / 12,
        )
        # Allow some slack — geometric ladder won't land exactly on paper values.
        assert 8 <= res['dr'] <= 30
        assert 0.04 <= res['dz'] <= 0.15
        assert res['predicted_error'] < 1e-3

    def test_optimal_c0_gives_better_grid(self):
        """Eq. (15) c₀ permits a coarser dx than the suboptimal c0=1500."""
        kw = dict(freq=500.0, c_min=1500.0, c_max=1500.0, x_max=5000.0,
                  theta_max=30.0, eps=1e-3, p=8, alpha=1 / 12)
        res_default = optimize_grid(c0=1500.0, **kw)
        res_optimal = optimize_grid(c0=optimal_c0(1500.0, 1500.0, 30.0), **kw)
        assert res_optimal['dr'] >= res_default['dr']

    def test_user_c0_is_honoured(self):
        """``c0`` is echoed back unchanged."""
        for c0 in [1480.0, 1500.0, 1545.0, 1600.0]:
            res = optimize_grid(
                freq=500.0, c_min=1500.0, c_max=1500.0, x_max=2000.0,
                c0=c0, theta_max=30.0, eps=1e-2, p=6, alpha=1 / 12,
            )
            assert res['c0'] == c0

    def test_the_search_sees_only_the_error_model(self):
        """``optimize_grid`` takes the Lytaev parameters and nothing else.

        Stability floors and array caps are RAM's, applied to the returned
        Δz — see ``TestRamAppliesTheStabilityFloor`` in
        ``test_ram_backends.py``. A knob here would let the two disagree
        about which grid is actually marched.
        """
        import inspect
        params = set(inspect.signature(optimize_grid).parameters)
        assert params == {'freq', 'c_min', 'c_max', 'x_max', 'c0',
                          'theta_max', 'eps', 'p', 'alpha'}

    def test_infeasible_raises(self):
        """No grid satisfies ε at this combination of inputs."""
        with pytest.raises(RuntimeError, match="No"):
            optimize_grid(
                freq=10000.0, c_min=1500.0, c_max=1500.0, x_max=1000000.0,
                c0=1500.0, theta_max=60.0, eps=1e-12, p=2,
                alpha=0.0,
            )

    def test_the_depth_ladder_spans_the_module_bounds(self):
        """The Δz ladder runs between ``DZ_MIN`` and ``DZ_MAX``, so it is
        never empty and the search cannot fail for want of a candidate."""
        from uacpy.models._pade_optimizer import DZ_MIN, DZ_MAX, _ladder
        rungs = _ladder(DZ_MIN, DZ_MAX)
        assert rungs
        assert min(rungs) == pytest.approx(DZ_MIN)
        assert max(rungs) == pytest.approx(DZ_MAX)
        assert rungs == sorted(rungs, reverse=True)


class TestGridError:
    """``grid_error`` describes any grid, including one the optimiser did
    not choose — the adjustments RAM applies afterwards (stability floors,
    array caps, seafloor snapping) leave ``predicted_error`` stale."""

    _KW = dict(freq=50.0, c_min=1500.0, c_max=1700.0, x_max=8000.0,
               c0=1600.0, theta_max=30.0, p=6)

    def test_reproduces_the_optimizers_own_number(self):
        res = optimize_grid(eps=1e-3, **self._KW)
        assert grid_error(dr=res['dr'], dz=res['dz'], **self._KW) == \
            pytest.approx(res['predicted_error'], rel=1e-12)

    def test_reports_a_bound_adjusted_grid_honestly(self):
        """``grid_error`` must describe the grid actually marched.

        RAM bounds the optimizer's Δz afterwards, so the optimizer's own
        ``predicted_error`` no longer applies and has to be recomputed.
        """
        res = optimize_grid(eps=1e-3, **self._KW)
        cap = rams_dz_shear_cap(c_shear_min=400.0, freq=50.0)
        shipped = grid_error(dr=res['dr'], dz=cap, **self._KW)
        assert cap != pytest.approx(res['dz'])
        assert shipped != pytest.approx(res['predicted_error'], rel=1e-6)


class TestInfeasibleGridRelaxationLadder:
    """RAM's ``_optimize_grid_relaxing`` (ram.py) wraps the optimiser in two
    nested fallbacks (ram.md §5): ε is tripled until it passes 0.5, then
    θ_max steps 30° → 20° → 15° restarting the ε ladder, and only then a
    ``ConfigurationError`` is raised. The optimiser itself is stubbed, so no
    grid search (and no binary) runs."""

    _KW = dict(freq=100.0, c_min=1500.0, c_max=1700.0, max_range=5000.0,
               c0_pe=1600.0, eps0=1e-2, theta0=30.0, kind='ramgeo')

    @staticmethod
    def _model():
        from uacpy.models import RAM
        return RAM(verbose=False)

    def _record_calls(self, monkeypatch, succeed_when):
        """Stub ``optimize_grid`` to record every (θ_max, ε) trial and
        succeed only when ``succeed_when(theta, eps)`` says so."""
        import uacpy.models._pade_optimizer as popt
        calls = []

        def fake(**kw):
            calls.append((kw['theta_max'], kw['eps']))
            if succeed_when(kw['theta_max'], kw['eps']):
                return {'dr': 10.0, 'dz': 1.0, 'c0': kw['c0'],
                        'predicted_error': kw['eps']}
            raise RuntimeError('infeasible')

        monkeypatch.setattr(popt, 'optimize_grid', fake)
        return calls

    def test_fully_infeasible_walks_the_whole_ladder_then_raises(
            self, monkeypatch):
        from uacpy.core.exceptions import ConfigurationError
        calls = self._record_calls(monkeypatch, lambda t, e: False)
        with pytest.raises(ConfigurationError, match=r'0\.5'):
            self._model()._optimize_grid_relaxing(**self._KW)
        # ε triples from 1e-2 until it passes 0.5 (4 trials), restarting at
        # each θ rung 30 → 20 → 15.
        eps_ladder = [1e-2, 3e-2, 9e-2, 0.27]
        expected = [(t, e) for t in (30.0, 20.0, 15.0) for e in eps_ladder]
        assert [(t, pytest.approx(e)) for t, e in expected] == calls

    def test_epsilon_relaxation_is_tried_before_narrowing_theta(
            self, monkeypatch):
        self._record_calls(monkeypatch,
                           lambda t, e: e >= 0.1)
        res, eps_used, theta_used = self._model()._optimize_grid_relaxing(
            **self._KW)
        assert theta_used == 30.0
        assert eps_used == pytest.approx(0.27)
        assert res['dr'] == 10.0

    def test_theta_steps_down_only_after_the_eps_ladder_fails(
            self, monkeypatch):
        self._record_calls(monkeypatch,
                           lambda t, e: t <= 20.0)
        res, eps_used, theta_used = self._model()._optimize_grid_relaxing(
            **self._KW)
        assert theta_used == 20.0
        assert eps_used == pytest.approx(1e-2)

    def test_a_narrow_caller_aperture_is_never_widened(self, monkeypatch):
        calls = self._record_calls(monkeypatch, lambda t, e: True)
        kw = dict(self._KW, theta0=18.0)
        _, _, theta_used = self._model()._optimize_grid_relaxing(**kw)
        assert theta_used == 18.0
        assert calls == [(18.0, pytest.approx(1e-2))]


class TestRamsDzShearCap:

    def test_zero_for_fluid(self):
        assert rams_dz_shear_cap(c_shear_min=0.0, freq=100.0) == 0.0

    def test_resolves_lambda_s(self):
        """Cap = λ_s / 14 = c_s / (14 f), Collins (1991)'s coarsest own grid."""
        dz = rams_dz_shear_cap(c_shear_min=400.0, freq=100.0)
        assert abs(dz - 4.0 / 14.0) < 1e-9

    def test_is_far_finer_than_a_fraction_of_the_shear_wavelength(self):
        """Direction guard. Collins (1991) resolves λ_s at 14-85 points per
        wavelength in every worked example, so the bound must sit well below
        λ_s — a grid at 0.55 λ_s makes the rams0.5 march diverge (measured
        134 dB against OASES on Collins' own example D)."""
        cs, freq = 800.0, 25.0
        lam_s = cs / freq
        assert rams_dz_shear_cap(cs, freq) <= lam_s / 14.0 * (1.0 + 1e-9)
        assert rams_dz_shear_cap(cs, freq) < 0.55 * lam_s / 4.0
