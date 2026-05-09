"""Tests for ``uacpy.core.metrics`` — TL-pair agreement helpers
(``tl_rmse``, ``tl_max_error``, ``tl_bias``).

All tests synthesize :class:`PressureField` instances directly; no model
binary is involved.
"""

import numpy as np
import pytest

import uacpy
from uacpy.core.metrics import tl_bias, tl_max_error, tl_rmse
from uacpy.core.results import PressureField


# ─── Basic dB-pair behaviour ──────────────────────────────────────────────

class TestTLRmseBasic:
    """``tl_rmse`` on dB-units :class:`PressureField` pairs."""

    def test_identical_fields_zero_rmse(self):
        d = np.linspace(5, 95, 10)
        r = np.linspace(100, 5000, 20)
        data = 60 + 10 * np.log10(np.maximum(r, 1.0)[None, :])
        data = np.broadcast_to(data, (10, 20)).copy()
        a = PressureField(units="dB", data=data, depths=d, ranges=r, model='A')
        b = PressureField(units="dB", data=data.copy(), depths=d, ranges=r, model='B')
        assert uacpy.tl_rmse(a, b) == pytest.approx(0.0)

    def test_constant_offset(self):
        d = np.linspace(5, 95, 10)
        r = np.linspace(100, 5000, 20)
        base = np.zeros((10, 20))
        a = PressureField(units="dB", data=base, depths=d, ranges=r)
        b = PressureField(units="dB", data=base + 3.0, depths=d, ranges=r)
        assert uacpy.tl_rmse(a, b) == pytest.approx(3.0)

    def test_window_selects_subregion(self):
        d = np.linspace(5, 95, 10)
        r = np.linspace(100, 5000, 20)
        a = PressureField(units="dB", data=np.zeros((10, 20)), depths=d, ranges=r)
        b_data = np.zeros((10, 20))
        b_data[:, :5] = 10.0
        b = PressureField(units="dB", data=b_data, depths=d, ranges=r)
        # Inside the noisy band RMSE = 10; outside it RMSE = 0.
        assert tl_rmse(a, b, range_window=(r[0], r[4])) == pytest.approx(10.0)
        assert tl_rmse(a, b, range_window=(r[5], r[-1])) == pytest.approx(0.0)

    def test_type_error_on_non_pressurefield(self):
        a = PressureField(
            units="dB", data=np.zeros((4, 4)),
            depths=np.arange(4), ranges=np.arange(4),
        )
        with pytest.raises(TypeError):
            uacpy.tl_rmse(a, object())


# ─── Unit-mode flexibility ────────────────────────────────────────────────

class TestTLMetricsUnits:
    """The metrics pull TL via the local ``_to_db`` helper, so any mix of
    ``units='complex'`` and ``units='dB'`` must round-trip."""

    def _pair(self, *, both_complex=False):
        rng = np.random.default_rng(0)
        a_db = 50.0 + 5.0 * rng.standard_normal((4, 5))
        b_db = a_db + 1.0     # 1-dB shift everywhere
        if both_complex:
            a = 10 ** (-a_db / 20.0) * np.exp(1j * rng.standard_normal((4, 5)))
            b = 10 ** (-b_db / 20.0) * np.exp(1j * rng.standard_normal((4, 5)))
            units_a = units_b = 'complex'
        else:
            a, b = a_db, b_db
            units_a = units_b = 'dB'
        common = dict(
            depths=np.linspace(10, 90, 4),
            ranges=np.linspace(100, 1000, 5),
            model='Test', frequencies=100.0,
        )
        return (
            PressureField(data=a, units=units_a, **common),
            PressureField(data=b, units=units_b, **common),
        )

    def test_rmse_db_pair(self):
        a, b = self._pair(both_complex=False)
        assert tl_rmse(a, b) == pytest.approx(1.0, abs=1e-9)

    def test_rmse_complex_pair_works(self):
        a, b = self._pair(both_complex=True)
        v = tl_rmse(a, b)
        assert v >= 0.0
        assert np.isfinite(v)

    def test_rmse_mixed_units(self):
        """Same TL stored once as complex and once as dB → RMSE ≈ 0."""
        a_cplx, b_cplx = self._pair(both_complex=True)
        a_db = PressureField(
            data=-20.0 * np.log10(np.maximum(np.abs(a_cplx.data), 1e-50)),
            depths=a_cplx.depths, ranges=a_cplx.ranges,
            units='dB', model='Test', frequencies=100.0,
        )
        assert tl_rmse(a_cplx, a_db) == pytest.approx(0.0, abs=1e-9)

    def test_max_error_and_bias_companions(self):
        a, b = self._pair(both_complex=False)
        assert tl_max_error(a, b) == pytest.approx(1.0, abs=1e-9)
        assert tl_bias(a, b) == pytest.approx(-1.0, abs=1e-9)
