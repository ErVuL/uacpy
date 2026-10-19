"""Tests for ``uacpy.core.results.metrics`` — TL-pair agreement helpers
(``tl_rmse``, ``tl_max_error``, ``tl_bias``, ``tl_difference``,
``field_statistics``, ``same_grid``).

All tests synthesize :class:`Field` instances directly; no model
binary is involved.
"""

import warnings

import numpy as np
import pytest
from uacpy.core.exceptions import ConfigurationError

import uacpy
from uacpy.core.results.metrics import tl_bias, tl_max_error, tl_rmse
from uacpy.core.results import Field
from uacpy.tests.conftest import recorded_warnings


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
        assert uacpy.metrics.tl_rmse(a, b) == pytest.approx(0.0)

    def test_constant_offset(self):
        d = np.linspace(5, 95, 10)
        r = np.linspace(100, 5000, 20)
        base = np.zeros((10, 20))
        a = _tl_field(base, d, r)
        b = _tl_field(base + 3.0, d, r)
        assert uacpy.metrics.tl_rmse(a, b) == pytest.approx(3.0)

    def test_window_selects_subregion(self):
        d = np.linspace(5, 95, 10)
        r = np.linspace(100, 5000, 20)
        a = _tl_field(np.zeros((10, 20)), d, r)
        b_data = np.zeros((10, 20))
        b_data[:, :5] = 10.0
        b = _tl_field(b_data, d, r)
        assert tl_rmse(a, b, range_window=(r[0], r[4])) == pytest.approx(10.0)
        assert tl_rmse(a, b, range_window=(r[5], r[-1])) == pytest.approx(0.0)

    def test_window_selecting_no_finite_cells_raises(self):
        """A window past the grid's edge (or one that lands only on NaN
        no-data cells) selects nothing: an empty comparison is a bug, not a
        zero (docs/guide/utilities.md)."""
        d = np.linspace(5, 95, 10)
        r = np.linspace(100, 5000, 20)
        a = _tl_field(np.zeros((10, 20)), d, r)
        b = _tl_field(np.zeros((10, 20)), d, r)
        with pytest.raises(ConfigurationError, match='no finite cells'):
            tl_rmse(a, b, range_window=(6000.0, 9000.0))
        # All-NaN cells inside an otherwise valid window select nothing too.
        nan_a = _tl_field(np.full((10, 20), np.nan), d, r)
        nan_b = _tl_field(np.full((10, 20), np.nan), d, r)
        with pytest.raises(ConfigurationError, match='no finite cells'):
            tl_rmse(nan_a, nan_b)

    def test_type_error_on_non_field(self):
        a = _tl_field(np.zeros((4, 4)), np.arange(4), np.arange(4))
        with pytest.raises(ConfigurationError,
                           match='must both be Fields or both be dB arrays'):
            uacpy.metrics.tl_rmse(a, object())

    def test_nan_no_data_cells_excluded(self):
        # NaN marks a no-data cell (e.g. Bellhop's r = 0 column); it
        # must be excluded from every statistic, not read as a value.
        d = np.array([10.0, 20.0])
        r = np.array([100.0, 200.0])
        a = _tl_field(np.array([[60.0, np.nan], [62.0, 64.0]]), d, r)
        b = _tl_field(np.array([[61.0, 70.0], [63.0, 65.0]]), d, r)
        assert tl_rmse(a, b) == pytest.approx(1.0)
        assert tl_max_error(a, b) == pytest.approx(1.0)
        assert tl_bias(a, b) == pytest.approx(-1.0)


class TestGridAlignment:
    """Grids agreeing to sub-millimetre rounding compare directly (models
    interpolate onto the requested receiver grid); genuinely different grids
    raise. The gate is :func:`uacpy.metrics.same_grid`: every coordinate
    within a thousandth of its axis' smallest cell (686 m here, so 0.69 m),
    which the 0.4 mm case clears and the 1 m case does not."""

    @pytest.mark.parametrize('fraction, agree', [(0.9e-3, True),
                                                 (1.1e-3, False)])
    def test_two_grids_agree_within_a_thousandth_of_their_own_cell(
            self, fraction, agree):
        """Both sides of the tolerance, on a 41.6 m range grid far from
        zero: the offset is measured against the cell, not the coordinate's
        magnitude (1.1e-3 of a cell is 4.6 cm, 9 parts per million of the
        last range)."""
        d = np.linspace(5.0, 95.0, 4)
        r = np.linspace(50.0, 5000.0, 120)
        cell = r[1] - r[0]
        a = _tl_field(np.zeros((4, 120)), d, r)
        b = _tl_field(np.zeros((4, 120)), d, r + fraction * cell)
        assert uacpy.metrics.same_grid(a, b) is agree
        if agree:
            assert tl_rmse(a, b) == pytest.approx(0.0)
        else:
            with pytest.raises(ConfigurationError, match='range axes differ'):
                tl_rmse(a, b)

    def test_axes_in_another_order_are_another_grid(self):
        d = np.linspace(5.0, 95.0, 4)
        r = np.linspace(100.0, 400.0, 4)
        a = Field(data=np.zeros((4, 4)), coords={'depth': d, 'range': r})
        b = Field(data=np.zeros((4, 4)), coords={'range': r, 'depth': d})
        assert uacpy.metrics.same_grid(a, a) is True
        assert uacpy.metrics.same_grid(a, b) is False

    def test_submillimetre_offset_compares(self):
        d = np.linspace(5, 95, 10)
        r = np.linspace(100, 20_000, 30)
        data = np.zeros((10, 30))
        a = _tl_field(data, d, r)
        b = _tl_field(data.copy(), d, r + 4e-4)        # 0.4 mm shift
        assert tl_rmse(a, b) == pytest.approx(0.0)

    def test_metre_scale_offset_raises(self):
        d = np.linspace(5, 95, 10)
        r = np.linspace(100, 20_000, 30)
        data = np.zeros((10, 30))
        a = _tl_field(data, d, r)
        b = _tl_field(data.copy(), d, r + 1.0)         # 1 m shift
        with pytest.raises(ConfigurationError, match="range axes differ"):
            tl_rmse(a, b)

    def test_two_grids_of_different_shape_point_at_the_resampling_metric(self):
        d = np.linspace(5, 95, 10)
        a = _tl_field(np.zeros((10, 19)), d, np.linspace(100, 2000, 19))
        b = _tl_field(np.zeros((10, 10)), d, np.linspace(100, 2000, 10))
        with pytest.raises(ConfigurationError,
                           match="shape mismatch(?s:.*)tl_rmse_on_shared_ranges"):
            tl_rmse(a, b)


class TestTLMetricsUnits:
    """Metrics pull TL via :attr:`Field.dB`, so a complex-pressure field
    and an equivalent real-dB field round-trip."""

    def _pair(self, *, both_complex=False):
        rng = np.random.default_rng(0)
        a_dB = 50.0 + 5.0 * rng.standard_normal((4, 5))
        b_dB = a_dB + 1.0     # 1-dB shift everywhere
        depths = np.linspace(10, 90, 4)
        ranges = np.linspace(100, 1000, 5)
        if both_complex:
            a = 10 ** (-a_dB / 20.0) * np.exp(1j * rng.standard_normal((4, 5)))
            b = 10 ** (-b_dB / 20.0) * np.exp(1j * rng.standard_normal((4, 5)))
        else:
            a, b = a_dB, b_dB
        return (
            _tl_field(a, depths, ranges, model='Test', frequencies=100.0),
            _tl_field(b, depths, ranges, model='Test', frequencies=100.0),
        )

    def test_rmse_dB_pair(self):
        a, b = self._pair(both_complex=False)
        assert tl_rmse(a, b) == pytest.approx(1.0, abs=1e-9)

    def test_rmse_complex_pair_recovers_dB_offset(self):
        # The complex pair stores |p| = 10^(-TL/20) with independent random
        # phases; .dB discards the phases, so the built-in 1-dB offset comes
        # back exactly — the same pin as the real-dB pair above.
        a, b = self._pair(both_complex=True)
        assert tl_rmse(a, b) == pytest.approx(1.0, abs=1e-9)

    def test_rmse_mixed_units(self):
        """Same TL stored once as complex and once as real-dB → RMSE ≈ 0."""
        a_cplx, _ = self._pair(both_complex=True)
        depths = a_cplx.coords['depth']
        ranges = a_cplx.coords['range']
        a_dB = _tl_field(
            -20.0 * np.log10(np.maximum(np.abs(a_cplx.data), 1e-50)),
            depths, ranges, model='Test', frequencies=100.0,
        )
        assert tl_rmse(a_cplx, a_dB) == pytest.approx(0.0, abs=1e-9)

    def test_max_error_and_bias_companions(self):
        a, b = self._pair(both_complex=False)
        assert tl_max_error(a, b) == pytest.approx(1.0, abs=1e-9)
        assert tl_bias(a, b) == pytest.approx(-1.0, abs=1e-9)


class TestKindsMustMatch:
    """The metrics compare the QUANTITY, not the representation. Complex
    pressure and real TL are one quantity written two ways (the class above
    pins that they round-trip); reverberation shares TL's dB representation
    exactly and is a different quantity, so subtracting the two produces a
    number with no meaning. ``compare_models`` refuses the same pairing
    before it puts both on one colour scale."""

    def _pair(self, kind_b):
        d = np.linspace(10, 90, 4)
        r = np.linspace(100, 1000, 5)
        data = 60.0 + np.zeros((4, 5))
        return (
            _tl_field(data, d, r, model='A',
                      kind='pressure', unit='dB'),
            _tl_field(data + 1.0, d, r, model='B',
                      kind=kind_b, unit='dB'),
        )

    @pytest.mark.parametrize('fn', [tl_rmse, tl_max_error, tl_bias])
    def test_pressure_against_reverberation_raises(self, fn):
        a, b = self._pair('reverberation')
        with pytest.raises(ConfigurationError,
                           match='different physical quantities'):
            fn(a, b)

    @pytest.mark.parametrize('fn', [tl_rmse, tl_max_error, tl_bias])
    def test_matching_kinds_compare(self, fn):
        a, b = self._pair('pressure')
        assert np.isfinite(fn(a, b))

    def test_two_reverberation_fields_compare(self):
        a, b = self._pair('reverberation')
        a = _tl_field(np.asarray(a.data), a.coords['depth'], a.coords['range'],
                      model='A', kind='reverberation', unit='dB')
        assert tl_rmse(a, b) == pytest.approx(1.0, abs=1e-9)


def _uniform_tl_field(value):
    return Field(data=np.full((2, 3), value),
                 coords={'depth': np.array([10.0, 20.0]),
                         'range': np.array([100.0, 200.0, 300.0])})


def _probability_field(value):
    return Field(data=np.full((2, 3), value),
                 coords={'depth': np.array([10.0, 20.0]),
                         'range': np.array([100.0, 200.0, 300.0])},
                 kind='probability_of_detection')


class TestTlMetricsPreCheckTheUnit:
    """Matching ``kind`` is not enough to make a pair a TL pair: a
    probability-of-detection field passes it and then reaches ``Field.dB``,
    whose refusal is an ``AttributeError``."""

    @pytest.mark.parametrize('metric', [tl_rmse, tl_max_error, tl_bias])
    def test_a_non_dB_pair_raises_a_typed_error(self, metric):
        with pytest.raises(ConfigurationError, match="not dB"):
            metric(_probability_field(0.4), _probability_field(0.6))

    @pytest.mark.parametrize('metric', [tl_rmse, tl_max_error, tl_bias])
    def test_the_error_names_the_offending_argument(self, metric):
        with pytest.raises(
                ConfigurationError,
                match='these are different physical quantities') as exc:
            metric(_uniform_tl_field(10.0), _probability_field(0.6))
        assert 'reference is' in str(exc.value)

    def test_two_dB_fields_compute_the_metrics(self):
        a, b = _uniform_tl_field(10.0), _uniform_tl_field(20.0)
        assert tl_rmse(a, b) == pytest.approx(10.0)
        assert tl_max_error(a, b) == pytest.approx(10.0)
        assert tl_bias(a, b) == pytest.approx(-10.0)

    def test_a_complex_field_derives_its_dB_view_and_compares(self):
        complex_field = Field(
            data=np.full((2, 3), 1e-3 + 0j),
            coords={'depth': np.array([10.0, 20.0]),
                    'range': np.array([100.0, 200.0, 300.0])})
        assert tl_rmse(complex_field, _uniform_tl_field(60.0)) == pytest.approx(0.0)


# ── The no-energy marker is no data, not a 600 dB loss ───────────────────────
# One exactly-zero cell in 50 used to turn a 0.83 dB agreement into 75 dB.

def _complex_pair(zero_col=None):
    depths = np.array([10.0, 20.0])
    ranges = np.linspace(100.0, 5000.0, 50)
    a = (1.0 / ranges)[None, :] * np.ones((2, 1)) + 0j
    b = 1.1 * a
    if zero_col is not None:
        b = b.copy()
        b[:, zero_col] = 0.0
    mk = lambda d: Field(data=d, coords={'depth': depths, 'range': ranges},
                         model='Synthetic')
    return mk(a), mk(b)


class TestNoEnergyCellsAreLeftOutOfEveryMetric:
    def test_a_zero_cell_does_not_move_the_rmse(self):
        a, b = _complex_pair()
        clean = tl_rmse(a, b)
        a2, b2 = _complex_pair(zero_col=7)
        with pytest.warns(UserWarning, match="2 cell"):
            got = tl_rmse(a2, b2)
        assert got == pytest.approx(clean, abs=1e-9)
        with pytest.warns(UserWarning):
            assert tl_max_error(a2, b2) == pytest.approx(
                tl_max_error(a, b), abs=1e-9)
        with pytest.warns(UserWarning):
            assert tl_bias(a2, b2) == pytest.approx(tl_bias(a, b), abs=1e-9)

    def test_a_deep_real_null_is_kept(self):
        a, b = _complex_pair()
        b.data[0, 3] = 1e-15            # 300 dB of real loss, not the marker
        assert tl_max_error(a, b) > 200.0

    def test_shared_ranges_leaves_the_marker_out_too(self):
        from uacpy.core.results.metrics import tl_rmse_on_shared_ranges
        a, b = _complex_pair()
        clean = tl_rmse_on_shared_ranges(a, b, depth=10.0)
        a2, b2 = _complex_pair(zero_col=7)
        with pytest.warns(UserWarning, match="no energy"):
            got = tl_rmse_on_shared_ranges(a2, b2, depth=10.0)
        assert got == pytest.approx(clean, abs=0.05)


class TestSharedRangesStatesWhatItCompares:
    def _field(self, depths, ranges, scale=1.0):
        data = scale * (1.0 / np.asarray(ranges))[None, :] \
            * np.ones((len(depths), 1)) + 0j
        return Field(data=data, coords={'depth': np.asarray(depths, float),
                                        'range': np.asarray(ranges, float)},
                     model='Synthetic')

    def test_two_different_nearest_depths_are_named(self):
        from uacpy.core.results.metrics import tl_rmse_on_shared_ranges
        a = self._field([10.0, 20.0], np.linspace(100, 1000, 10))
        b = self._field([12.0, 22.0], np.linspace(100, 1000, 10))
        with pytest.warns(UserWarning, match="10 m in `field` and 12 m"):
            tl_rmse_on_shared_ranges(a, b, depth=10.0)

    def test_the_common_axis_is_the_coarser_spacing_not_the_shorter_axis(self):
        """50 points at 20 m against 100 at 100 m: the coarser grid is the
        100 m one, although it has more points. Read on it, 1/r agrees
        exactly at every node both fields hold, so the RMS is 0; read on the
        20 m axis, the coarse field is interpolated between its nodes and
        is not."""
        from uacpy.core.results.metrics import tl_rmse_on_shared_ranges
        fine = self._field([10.0], np.linspace(20.0, 1000.0, 50))
        coarse = self._field([10.0], np.linspace(100.0, 10000.0, 100))
        assert tl_rmse_on_shared_ranges(fine, coarse, depth=10.0) == \
            pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize('depth, refused', [
        (95.0 + 5.0 - 1e-6, False), (95.0 + 5.0 + 1e-6, True),
        (5.0 - 5.0 + 1e-6, False), (5.0 - 5.0 - 1e-6, True),
        (500.0, True)])
    def test_a_depth_beyond_half_a_step_outside_the_axis_is_refused(
            self, depth, refused):
        """5..95 m every 10 m: a depth within half a step of an end reads
        that end, a depth past it (500 m, a km-for-m slip) is refused
        rather than compared at 95 m."""
        from uacpy.core.results.metrics import tl_rmse_on_shared_ranges
        z = np.linspace(5.0, 95.0, 10)
        a = self._field(z, np.linspace(100, 1000, 10))
        b = self._field(z, np.linspace(100, 1000, 10), scale=2.0)
        if refused:
            with pytest.raises(ConfigurationError,
                               match=r"outside field's depth axis \[5, 95\]"):
                tl_rmse_on_shared_ranges(a, b, depth=depth)
        else:
            assert tl_rmse_on_shared_ranges(a, b, depth=depth) == \
                pytest.approx(20 * np.log10(2.0))

    def test_two_range_cuts_compare_as_the_grids_they_came_from(self):
        """``field.at(depth=z)`` is a Field read at one depth: it compares
        with no depth=, to the number the grids give at that depth, and the
        depth it was cut at is the one the mismatch warning names."""
        from uacpy.core.results.metrics import tl_rmse_on_shared_ranges
        a = self._field([10.0, 20.0], np.linspace(100, 1000, 10))
        b = self._field([10.0, 20.0], np.linspace(100, 1000, 7), scale=2.0)
        on_grids = tl_rmse_on_shared_ranges(a, b, depth=10.0)
        assert np.isfinite(on_grids) and on_grids > 5.0
        cut_a, cut_b = a.at(depth=10.0), b.at(depth=10.0)
        assert tl_rmse_on_shared_ranges(cut_a, cut_b) == on_grids
        assert tl_rmse_on_shared_ranges(a, cut_b, depth=10.0) == on_grids
        with pytest.warns(UserWarning, match="10 m in `field` and 20 m"):
            tl_rmse_on_shared_ranges(cut_a, b.at(depth=20.0))
        with pytest.raises(ConfigurationError,
                           match='two range cuts are one depth'):
            tl_rmse_on_shared_ranges(cut_a, cut_b, depth=10.0)

    def test_a_field_without_a_range_axis_is_a_typed_refusal(self):
        from uacpy.core.results.metrics import tl_rmse_on_shared_ranges
        a = self._field([10.0, 20.0], np.linspace(100, 1000, 10))
        with pytest.raises(ConfigurationError, match='has no range axis'):
            tl_rmse_on_shared_ranges(a.at(range=500.0), a.at(range=500.0),
                                     depth=10.0)

    @pytest.mark.parametrize('metric', ['tl_rmse_on_shared_ranges',
                                        'tl_rmse'])
    def test_both_field_forms_name_the_same_first_defect(self, metric):
        """A pressure field against a probability-of-detection field is two
        defects (kind and unit): both forms report the kind first."""
        import uacpy.core.results.metrics as metrics
        a = self._field([10.0], np.linspace(100, 1000, 10))
        pd = Field(data=np.full((1, 10), 0.5),
                   coords={'depth': np.array([10.0]),
                           'range': np.linspace(100, 1000, 10)},
                   model='Synthetic', kind='probability_of_detection')
        kwargs = {'depth': 10.0} if metric == 'tl_rmse_on_shared_ranges' \
            else {}
        with pytest.raises(ConfigurationError,
                           match='different physical quantities'):
            getattr(metrics, metric)(a, pd, **kwargs)

    def test_a_real_field_not_in_dB_is_refused(self):
        from uacpy.core.results.metrics import tl_rmse_on_shared_ranges
        a = self._field([10.0], np.linspace(100, 1000, 10))
        pd = Field(data=np.full((1, 10), 0.5),
                   coords={'depth': np.array([10.0]),
                           'range': np.linspace(100, 1000, 10)},
                   model='Synthetic', unit='Pa')
        with pytest.raises(ConfigurationError, match="not dB"):
            tl_rmse_on_shared_ranges(pd, a, depth=10.0)

    @pytest.mark.parametrize('metric', ['tl_rmse_on_shared_ranges',
                                        'tl_rmse'])
    def test_a_multi_depth_field_on_either_side_names_one_source_or_the_sum(
            self, metric):
        """A field over source depths is one TL map per source: either
        side is refused typed, naming at(source_depth=...) and superpose();
        one plane compares."""
        import uacpy.core.results.metrics as metrics
        call = getattr(metrics, metric)
        a = self._field([10.0, 20.0], np.linspace(100, 1000, 10))
        pair = Field(data=np.stack([a.data, 2.0 * a.data]),
                     coords={'source_depth': np.array([5.0, 15.0]),
                             **a.coords},
                     model='Synthetic', source_depths=[5.0, 15.0])
        kwargs = {'depth': 10.0} if metric == 'tl_rmse_on_shared_ranges' \
            else {}
        for field, reference in ((pair, a), (a, pair)):
            with pytest.raises(ConfigurationError,
                               match=rf"{metric}: .*got coords "
                                     rf"\['source_depth'") as err:
                call(field, reference, **kwargs)
            assert 'at(source_depth=...)' in str(err.value)
            assert 'superpose()' in str(err.value)
        assert call(pair.at(source_depth=5.0), a, **kwargs) == \
            pytest.approx(0.0, abs=1e-9)


# ── (field, reference) on plain dB arrays and on 1-D cuts ────────────────────

class TestSharedRangesTakesTwoPlainCurves:
    """``tl_rmse_on_shared_ranges`` runs on two ``(ranges, tl_dB)`` curves
    with no Field built, to the number the Field form gives for the same
    curves; depth= belongs to Fields only, the two forms do not mix, and a
    curve whose ranges cannot be interpolated on is refused."""

    R_FINE = np.linspace(1000.0, 5000.0, 65)
    R_COARSE = np.linspace(1500.0, 6000.0, 10)

    def _curves(self):
        tl_fine = 60.0 + 10.0 * np.log10(self.R_FINE) + np.sin(
            self.R_FINE / 300.0)
        tl_coarse = 61.0 + 10.0 * np.log10(self.R_COARSE)
        return (self.R_FINE, tl_fine), (self.R_COARSE, tl_coarse)

    def test_two_curves_give_the_field_forms_number(self):
        from uacpy.core.results.metrics import tl_rmse_on_shared_ranges
        (r_a, tl_a), (r_b, tl_b) = self._curves()
        plain = tl_rmse_on_shared_ranges((r_a, tl_a), (r_b, tl_b))
        fields = [Field(data=tl[None, :], coords={'depth': np.array([50.0]),
                                                  'range': r},
                        unit='dB', kind='pressure')
                  for r, tl in ((r_a, tl_a), (r_b, tl_b))]
        assert plain == pytest.approx(
            tl_rmse_on_shared_ranges(*fields, depth=50.0), rel=1e-12)
        assert np.isfinite(plain) and plain > 0.5

    def test_depth_with_two_curves_is_refused(self):
        from uacpy.core.results.metrics import tl_rmse_on_shared_ranges
        with pytest.raises(ConfigurationError, match='one depth'):
            tl_rmse_on_shared_ranges(*self._curves(), depth=50.0)

    def test_a_field_and_a_curve_together_are_refused(self):
        from uacpy.core.results.metrics import tl_rmse_on_shared_ranges
        (r_a, tl_a), curve = self._curves()
        field = Field(data=tl_a[None, :], coords={'depth': np.array([50.0]),
                                                  'range': r_a},
                      unit='dB', kind='pressure')
        with pytest.raises(ConfigurationError, match='both be Fields'):
            tl_rmse_on_shared_ranges(field, curve, depth=50.0)

    def test_two_fields_without_a_depth_are_refused(self):
        from uacpy.core.results.metrics import tl_rmse_on_shared_ranges
        (r, tl), _ = self._curves()
        field = Field(data=tl[None, :], coords={'depth': np.array([50.0]),
                                                'range': r},
                      unit='dB', kind='pressure')
        with pytest.raises(ConfigurationError, match='pass depth='):
            tl_rmse_on_shared_ranges(field, field)

    def test_ranges_that_do_not_increase_are_refused(self):
        from uacpy.core.results.metrics import tl_rmse_on_shared_ranges
        (r, tl), other = self._curves()
        with pytest.raises(ConfigurationError, match='strictly increasing'):
            tl_rmse_on_shared_ranges((r[::-1], tl), other)


class TestMetricsTakeFieldAndReference:
    """The four metrics spell their pair ``(field, reference)`` and the
    signed difference is ``field - reference``."""

    def test_keywords_are_field_and_reference(self):
        a, b = _uniform_tl_field(10.0), _uniform_tl_field(20.0)
        assert tl_bias(field=a, reference=b) == pytest.approx(-10.0)
        assert tl_rmse(field=a, reference=b) == pytest.approx(10.0)
        assert tl_max_error(field=a, reference=b) == pytest.approx(10.0)

    def test_bias_is_field_minus_reference(self):
        a, b = _uniform_tl_field(23.0), _uniform_tl_field(20.0)
        assert tl_bias(a, b) == pytest.approx(3.0)
        assert tl_bias(b, a) == pytest.approx(-3.0)


class TestMetricsOnPlainDbArrays:
    """A measured TL curve or a table from another code compares without
    building a Field; the Field form runs the same computation."""

    def test_arrays_give_the_numbers_their_fields_give(self):
        rng = np.random.default_rng(3)
        tl = 60.0 + 5.0 * rng.standard_normal((4, 6))
        ref = tl + rng.standard_normal((4, 6))
        d, r = np.linspace(10, 40, 4), np.linspace(100, 600, 6)
        fa, fb = _tl_field(tl, d, r), _tl_field(ref, d, r)
        for metric in (tl_rmse, tl_max_error, tl_bias):
            assert metric(tl, ref) == metric(fa, fb)

    def test_a_list_curve_compares(self):
        assert tl_bias([60.0, 62.0, 64.0], [59.0, 61.0, 63.0]) == \
            pytest.approx(1.0)
        assert tl_rmse([60.0, 62.0], [60.0, 58.0]) == \
            pytest.approx(np.sqrt(8.0))

    def test_nan_and_the_no_energy_marker_are_left_out_of_arrays(self):
        tl = np.array([60.0, np.nan, 600.0, 70.0])
        ref = np.array([61.0, 50.0, 60.0, 71.0])
        with pytest.warns(UserWarning, match="1 cell"):
            assert tl_rmse(tl, ref) == pytest.approx(1.0)

    def test_shape_mismatch_is_refused(self):
        with pytest.raises(ConfigurationError, match="shape mismatch"):
            tl_rmse(np.zeros(3), np.zeros(4))

    def test_a_complex_array_is_refused(self):
        with pytest.raises(ConfigurationError, match="complex array"):
            tl_rmse(np.ones(3) + 0j, np.ones(3))

    def test_a_window_on_arrays_is_refused(self):
        with pytest.raises(ConfigurationError, match="no depth or range axis"):
            tl_rmse(np.zeros(3), np.ones(3), range_window=(0.0, 1.0))

    def test_a_field_and_an_array_together_are_refused(self):
        with pytest.raises(ConfigurationError, match="both be Fields"):
            tl_rmse(_uniform_tl_field(10.0), np.full((2, 3), 10.0))


class TestMetricsOnOneDimensionalCuts:
    """``field.at(depth=z)`` — the TL-vs-range curve a comparison most
    often wants — is a Field on one ``range`` axis."""

    def _grid(self, offset):
        d = np.array([10.0, 20.0, 30.0])
        r = np.linspace(100.0, 1000.0, 10)
        data = 60.0 + np.arange(30.0).reshape(3, 10) + offset
        return _tl_field(data, d, r)

    def test_a_range_cut_compares(self):
        a, b = self._grid(2.0).at(depth=20.0), self._grid(0.0).at(depth=20.0)
        assert list(a.coords) == ['range']
        assert tl_bias(a, b) == pytest.approx(2.0)

    def test_the_range_window_applies_to_a_range_cut(self):
        a = self._grid(0.0).at(depth=20.0)
        data = np.asarray(a.data, dtype=float).copy()
        data[:5] += 4.0
        b = _tl_field(data[None, :], [20.0], a.coords['range']).at(depth=20.0)
        r = a.coords['range']
        assert tl_rmse(b, a, range_window=(r[0], r[4])) == pytest.approx(4.0)
        assert tl_rmse(b, a, range_window=(r[5], r[-1])) == pytest.approx(0.0)

    def test_a_depth_window_on_a_range_cut_is_refused(self):
        a = self._grid(0.0).at(depth=20.0)
        with pytest.raises(ConfigurationError, match="no depth axis"):
            tl_rmse(a, a, depth_window=(0.0, 50.0))

    def test_a_cut_against_a_grid_is_refused(self):
        with pytest.raises(ConfigurationError, match="do not correspond"):
            tl_rmse(self._grid(0.0).at(depth=20.0), self._grid(0.0))

    def test_a_frequency_axis_is_refused(self):
        f = Field(data=np.ones((2, 3)) + 0j,
                  coords={'frequency': np.array([100.0, 200.0]),
                          'range': np.array([100.0, 200.0, 300.0])})
        with pytest.raises(ConfigurationError, match="depth and/or range"):
            tl_rmse(f, f)


class TestTheDifferenceMapIsAField:
    """``tl_difference`` returns ``field - reference`` as a Field of kind
    ``'difference'``, under the metrics' own pair and cell rules."""

    def test_it_is_the_signed_difference_on_the_fields_grid(self):
        d = np.linspace(5, 95, 4)
        r = np.linspace(100, 1000, 5)
        a = _tl_field(np.full((4, 5), 70.0), d, r, model='A', unit='dB',
                      pinned={'frequency': 250.0})
        b = _tl_field(np.full((4, 5), 64.0), d, r, model='B', unit='dB')
        diff = uacpy.metrics.tl_difference(a, b)
        assert isinstance(diff, Field)
        assert (diff.kind, diff.unit) == ('difference', 'dB')
        np.testing.assert_array_equal(diff.data, np.full((4, 5), 6.0))
        np.testing.assert_array_equal(diff.coords['range'], r)
        assert diff.pinned == {'frequency': 250.0}

    def test_a_no_energy_cell_has_no_difference(self):
        """The 600 dB marker is no level: the cell is NaN, as the other
        metrics leave it out, and the map shows it blank with no warning."""
        a, b = _complex_pair(zero_col=2)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            diff = uacpy.metrics.tl_difference(a, b)
        assert np.all(np.isnan(diff.data[:, 2]))
        assert np.all(np.isfinite(np.delete(diff.data, 2, axis=1)))

    def test_two_grids_and_two_quantities_are_refused(self):
        d = np.linspace(5, 95, 4)
        a = _tl_field(np.zeros((4, 5)), d, np.linspace(100, 1000, 5),
                      unit='dB')
        shifted = _tl_field(np.zeros((4, 5)), d, np.linspace(200, 1100, 5),
                            unit='dB')
        with pytest.raises(ConfigurationError, match='range axes differ'):
            uacpy.metrics.tl_difference(a, shifted)
        with pytest.raises(ConfigurationError,
                           match="'pressure' field but reference is "
                                 "'reverberation'"):
            uacpy.metrics.tl_difference(a, a.replace(kind='reverberation'))
        with pytest.raises(ConfigurationError, match='both be Fields'):
            uacpy.metrics.tl_difference(a, np.zeros((4, 5)))


class TestFieldStatisticsIsARecord:
    """``field_statistics`` returns the numbers ``plot_field_statistics``
    draws: means, standard deviations and the pairwise RMS table."""

    @staticmethod
    def _field(level_dB, kind='pressure'):
        r = np.linspace(100.0, 1000.0, 10)
        return Field(data=np.full((1, r.size), level_dB),
                     coords={'depth': np.array([50.0]), 'range': r},
                     model='T', frequencies=100.0, kind=kind, unit='dB')

    def test_the_numbers_are_the_metrics(self):
        stats = uacpy.metrics.field_statistics(
            {'a': self._field(60.0), 'b': self._field(66.0), 'c': None},
            depth=50.0)
        assert stats.labels == ('a', 'b')
        np.testing.assert_allclose(stats.mean_dB, [60.0, 66.0])
        np.testing.assert_allclose(stats.std_dB, [0.0, 0.0])
        np.testing.assert_allclose(stats.rms_dB, [[0.0, 6.0], [6.0, 0.0]])
        assert stats.kinds == ('pressure', 'pressure')

    def test_two_quantities_score_nan_with_one_warning(self):
        """A reverberation loss against a TL is not an agreement metric:
        the pair's entries are NaN (not a number on the colormap's green),
        and one warning names the function and the pair."""
        with recorded_warnings() as caught:
            stats = uacpy.metrics.field_statistics(
                [self._field(60.0), self._field(60.0, 'reverberation')],
                depth=50.0)
        assert np.isnan(stats.rms_dB[0, 1]) and np.isnan(stats.rms_dB[1, 0])
        kind_warnings = [str(w.message) for w in caught
                         if 'agreement metric' in str(w.message)]
        assert len(kind_warnings) == 1
        assert kind_warnings[0].startswith('field_statistics:')
        # The like-for-like pair still scores.
        same = uacpy.metrics.field_statistics(
            [self._field(60.0), self._field(60.0)], depth=50.0)
        assert same.rms_dB[0, 1] == 0.0

    def test_it_exports_and_reads_back(self):
        stats = uacpy.metrics.field_statistics(
            [self._field(60.0), self._field(63.0)], depth=50.0)
        again = type(stats).from_xarray(stats.to_xarray())
        assert again.labels == stats.labels
        np.testing.assert_array_equal(again.rms_dB, stats.rms_dB)
        again = type(stats).from_dict(stats.to_dict())
        np.testing.assert_array_equal(again.mean_dB, stats.mean_dB)
