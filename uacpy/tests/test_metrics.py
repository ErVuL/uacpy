"""Tests for ``uacpy.core.metrics`` — TL-pair agreement helpers
(``tl_rmse``, ``tl_max_error``, ``tl_bias``).

All tests synthesize :class:`Field` instances directly; no model
binary is involved.
"""

import numpy as np
import pytest

import uacpy
from uacpy.core.metrics import tl_bias, tl_max_error, tl_rmse
from uacpy.core.results import Field


def _tl_field(data, depths, ranges, **kw):
    return Field(
        data=np.asarray(data),
        coords={'depth': np.asarray(depths), 'range': np.asarray(ranges)},
        **kw,
    )


class TestTLRmseBasic:
    """``tl_rmse`` on real-dB :class:`Field` pairs."""

    def test_identical_fields_zero_rmse(self):
        d = np.linspace(5, 95, 10)
        r = np.linspace(100, 5000, 20)
        data = 60 + 10 * np.log10(np.maximum(r, 1.0)[None, :])
        data = np.broadcast_to(data, (10, 20)).copy()
        a = _tl_field(data, d, r, model='A')
        b = _tl_field(data.copy(), d, r, model='B')
        assert uacpy.tl_rmse(a, b) == pytest.approx(0.0)

    def test_constant_offset(self):
        d = np.linspace(5, 95, 10)
        r = np.linspace(100, 5000, 20)
        base = np.zeros((10, 20))
        a = _tl_field(base, d, r)
        b = _tl_field(base + 3.0, d, r)
        assert uacpy.tl_rmse(a, b) == pytest.approx(3.0)

    def test_window_selects_subregion(self):
        d = np.linspace(5, 95, 10)
        r = np.linspace(100, 5000, 20)
        a = _tl_field(np.zeros((10, 20)), d, r)
        b_data = np.zeros((10, 20))
        b_data[:, :5] = 10.0
        b = _tl_field(b_data, d, r)
        assert tl_rmse(a, b, range_window=(r[0], r[4])) == pytest.approx(10.0)
        assert tl_rmse(a, b, range_window=(r[5], r[-1])) == pytest.approx(0.0)

    def test_type_error_on_non_field(self):
        a = _tl_field(np.zeros((4, 4)), np.arange(4), np.arange(4))
        with pytest.raises(TypeError):
            uacpy.tl_rmse(a, object())


class TestTLMetricsUnits:
    """Metrics pull TL via :attr:`Field.tl`, so a complex-pressure field
    and an equivalent real-dB field round-trip."""

    def _pair(self, *, both_complex=False):
        rng = np.random.default_rng(0)
        a_db = 50.0 + 5.0 * rng.standard_normal((4, 5))
        b_db = a_db + 1.0     # 1-dB shift everywhere
        depths = np.linspace(10, 90, 4)
        ranges = np.linspace(100, 1000, 5)
        if both_complex:
            a = 10 ** (-a_db / 20.0) * np.exp(1j * rng.standard_normal((4, 5)))
            b = 10 ** (-b_db / 20.0) * np.exp(1j * rng.standard_normal((4, 5)))
        else:
            a, b = a_db, b_db
        return (
            _tl_field(a, depths, ranges, model='Test', frequencies=100.0),
            _tl_field(b, depths, ranges, model='Test', frequencies=100.0),
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
        """Same TL stored once as complex and once as real-dB → RMSE ≈ 0."""
        a_cplx, _ = self._pair(both_complex=True)
        depths = a_cplx.coords['depth']
        ranges = a_cplx.coords['range']
        a_db = _tl_field(
            -20.0 * np.log10(np.maximum(np.abs(a_cplx.data), 1e-50)),
            depths, ranges, model='Test', frequencies=100.0,
        )
        assert tl_rmse(a_cplx, a_db) == pytest.approx(0.0, abs=1e-9)

    def test_max_error_and_bias_companions(self):
        a, b = self._pair(both_complex=False)
        assert tl_max_error(a, b) == pytest.approx(1.0, abs=1e-9)
        assert tl_bias(a, b) == pytest.approx(-1.0, abs=1e-9)
