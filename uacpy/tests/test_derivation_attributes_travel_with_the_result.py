"""What a reader acts on travels as a typed attribute of the result:
the Field's derivation record (band, synthesis_window, no_mode_bins,
sonar_budget, sigma_dB, reduction_speed), Arrivals.absorption and
ReflectionCoefficient.reflection_type. Each is carried to a derived result,
round-trips through to_dict and the xarray export, and is refused as a
metadata key — and a dict keeping it in metadata is refused on load."""
import numpy as np
import pytest

import uacpy
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Arrivals, Field, ReflectionCoefficient

BUDGET = {'mode': 'passive', 'source_level_dB': 140.0,
          'detection_threshold_dB': 3.0, 'noise_level_dB': 60.0,
          'target_strength_dB': None, 'reverberation_level_dB': None,
          'directivity_index_dB': 15.0, 'array_gain_dB': [1.0, 2.0],
          'processing_loss_dB': 0.0}

#: tag -> (value, its metadata spelling)
DERIVATION = {
    'band': ((100.0, 300.0), 'band'),
    'synthesis_window': ('hann', 'window'),
    'no_mode_bins': ((0, 2), 'no_mode_bins'),
    'sonar_budget': (BUDGET, 'sonar_budget'),
    'sigma_dB': (5.6, 'sigma_dB'),
    'reduction_speed': (1490.0, 'reduction_speed'),
}


def _field(**kw):
    return Field(data=np.full((2, 3, 3), 60.0),
                 coords={'depth': np.array([10.0, 20.0]),
                         'range': np.array([100.0, 200.0, 300.0]),
                         'frequency': np.array([50.0, 60.0, 70.0])},
                 kind='pressure', unit='dB', **kw)


@pytest.mark.parametrize('speed, ok', [
    (1500.0, True), (1e-9, True), (0.0, False), (-1500.0, False),
    (np.nan, False), (np.inf, False)])
def test_a_reduction_speed_is_a_positive_finite_speed(speed, ok):
    if ok:
        assert _field(reduction_speed=speed).reduction_speed == speed
    else:
        with pytest.raises(ConfigurationError, match="reduction_speed"):
            _field(reduction_speed=speed)


@pytest.mark.parametrize('tag', sorted(DERIVATION))
class TestFieldDerivationRecord:

    def test_the_constructor_keyword_is_the_attribute(self, tag):
        value, _ = DERIVATION[tag]
        assert getattr(_field(**{tag: value}), tag) == value
        assert getattr(_field(), tag) is None

    def test_a_derived_field_carries_it(self, tag):
        value, _ = DERIVATION[tag]
        sliced = _field(**{tag: value}).at(range=200.0)
        assert getattr(sliced, tag) == value

    def test_to_dict_round_trips_it(self, tag):
        value, _ = DERIVATION[tag]
        back = Field.from_dict(_field(**{tag: value}).to_dict())
        assert getattr(back, tag) == value
        assert not back.metadata

    def test_the_xarray_export_round_trips_it(self, tag):
        pytest.importorskip('xarray')
        value, _ = DERIVATION[tag]
        back = Field.from_xarray(_field(**{tag: value}).to_xarray())
        assert getattr(back, tag) == value
        assert not back.metadata

    def test_a_saved_dict_keeping_it_in_metadata_is_refused(self, tag):
        """Only the current layout is read: the tag is an attribute, so a
        dict that holds it in metadata is refused, not read."""
        value, spelling = DERIVATION[tag]
        d = _field().to_dict()
        d['metadata'] = {spelling: value, 'title': 'kept'}
        with pytest.raises(ConfigurationError, match=spelling):
            Field.from_dict(d)

    def test_a_metadata_entry_naming_it_is_refused(self, tag):
        value, spelling = DERIVATION[tag]
        with pytest.raises(ConfigurationError, match=spelling):
            _field(metadata={spelling: value})


@pytest.mark.parametrize('bins, axis', [
    ((3,), True), ((-1,), True), ((0.5,), True), ((0,), False)])
def test_no_mode_bins_index_the_frequency_axis(bins, axis):
    field = _field() if axis else _field().at(frequency=60.0)
    with pytest.raises(ConfigurationError, match='no_mode_bins'):
        field.replace(no_mode_bins=bins)


def test_no_mode_bins_are_sorted_distinct_and_none_when_empty():
    assert _field(no_mode_bins=[2, 0, 2]).no_mode_bins == (0, 2)
    assert _field(no_mode_bins=()).no_mode_bins is None


def test_an_array_narrowed_along_frequency_keeps_the_bins_it_holds():
    """The attrs index the band they list; an array cut from it keeps the
    no-mode bins of the frequencies it still holds, at their new index."""
    pytest.importorskip('xarray')
    band = np.array([50.0, 60.0, 70.0])
    da = _field(no_mode_bins=(0, 2), frequencies=band).to_xarray()
    assert Field.from_xarray(da.isel(frequency=slice(1, None))
                             ).no_mode_bins == (1,)
    assert Field.from_xarray(da.isel(frequency=[1])).no_mode_bins is None
    assert Field.from_xarray(da.isel(frequency=0)).no_mode_bins is None
    # With no band in the attrs, bins the narrowed axis cannot place go.
    bare = _field(no_mode_bins=(0, 2)).to_xarray()
    assert Field.from_xarray(bare.isel(frequency=slice(1, None))
                             ).no_mode_bins is None


def test_no_mode_bins_round_trip_through_netcdf(tmp_path):
    pytest.importorskip('xarray')
    path = tmp_path / 'band.nc'
    _field(no_mode_bins=(0, 2),
           frequencies=np.array([50.0, 60.0, 70.0])).to_netcdf(path)
    assert Field.from_netcdf(path).no_mode_bins == (0, 2)


def test_a_band_that_is_not_a_pair_is_refused():
    with pytest.raises(ConfigurationError, match='band'):
        _field(band=(100.0, 200.0, 300.0))


def test_the_budget_property_is_a_copy():
    f = _field(sonar_budget=BUDGET)
    f.sonar_budget['mode'] = 'active'
    assert f.sonar_budget['mode'] == 'passive'


def test_a_windowed_trace_warns_on_inversion_and_an_unwindowed_one_does_not():
    import warnings
    t = np.arange(64) / 1000.0
    trace = Field(data=np.sin(2 * np.pi * 50.0 * t), coords={'time': t},
                  synthesis_window='hann')
    with pytest.warns(uacpy.NumericsWarning, match="'hann' band window"):
        trace.to_transfer_function()
    with warnings.catch_warnings():
        warnings.simplefilter('error', uacpy.NumericsWarning)
        trace.replace(synthesis_window=None).to_transfer_function()


def _arrivals(**kw):
    cell = [{'delay': 0.1, 'delay_imag': -1e-5, 'amplitude': 1.0,
             'phase': 0.0, 'n_top_bounces': 0, 'n_bot_bounces': 0,
             'source_angle': 1.0, 'receiver_angle': 1.0}]
    return Arrivals(by_receiver=[[[cell]]], receiver_depths=[20.0],
                    receiver_ranges=[1000.0], frequencies=[1000.0], **kw)


LAW = uacpy.Thorp()


class TestArrivalsAbsorption:

    def test_the_constructor_keyword_is_the_attribute(self):
        assert _arrivals(absorption=LAW).absorption is LAW
        assert _arrivals().absorption is None

    def test_a_filtered_set_carries_it(self):
        kept = _arrivals(absorption=LAW).filter(lambda a: True)
        assert kept.absorption is LAW

    def test_to_dict_round_trips_it(self):
        back = Arrivals.from_dict(_arrivals(absorption=LAW).to_dict())
        assert type(back.absorption) is type(LAW)
        assert back.absorption.to_dict() == LAW.to_dict()

    def test_the_xarray_export_round_trips_it(self):
        pytest.importorskip('xarray')
        back = Arrivals.from_xarray(_arrivals(absorption=LAW).to_xarray())
        assert back.absorption.to_dict() == LAW.to_dict()
        assert 'absorption' not in back.metadata

    def test_a_measured_table_round_trips_as_the_table(self):
        # A traced environment's measured table is what the arrivals hold
        # (env.absorption as it is), and what both exports bring back.
        from uacpy.core.absorption import AbsorptionCoefficient
        from uacpy.tests.conftest import measured_absorption_table
        table = measured_absorption_table()
        arrivals = _arrivals(absorption=table)
        backs = [Arrivals.from_dict(arrivals.to_dict())]
        try:
            import xarray  # noqa: F401
            backs.append(Arrivals.from_xarray(arrivals.to_xarray()))
        except ImportError:
            pass
        for back in backs:
            assert type(back.absorption) is AbsorptionCoefficient
            np.testing.assert_array_equal(back.absorption.data, table.data)
            np.testing.assert_array_equal(back.absorption.depths,
                                          table.depths)
            np.testing.assert_array_equal(
                back.received_amplitudes, arrivals.received_amplitudes)

    def test_a_saved_dict_keeping_it_in_metadata_is_refused(self):
        d = _arrivals().to_dict()
        d['metadata'] = {'absorption': LAW}
        with pytest.raises(ConfigurationError, match='absorption'):
            Arrivals.from_dict(d)

    def test_a_metadata_entry_naming_it_is_refused(self):
        with pytest.raises(ConfigurationError, match='absorption'):
            _arrivals(metadata={'absorption': LAW})


def _table(**kw):
    return ReflectionCoefficient(angles_deg=[10.0, 20.0, 30.0],
                                 magnitude=[0.9, 1.1, 0.8],
                                 phase=[0.0, 0.1, 0.2], **kw)


class TestReflectionType:

    def test_the_constructor_keyword_is_the_attribute(self):
        assert _table(reflection_type='transmission').reflection_type == \
            'transmission'
        assert _table().reflection_type is None

    def test_a_slice_carries_it(self):
        sliced = _table(reflection_type='transmission').at(angle=20.0)
        assert sliced.reflection_type == 'transmission'

    def test_to_dict_round_trips_it(self):
        back = ReflectionCoefficient.from_dict(
            _table(reflection_type='transmission').to_dict())
        assert back.reflection_type == 'transmission'

    def test_the_xarray_export_round_trips_it(self):
        pytest.importorskip('xarray')
        back = ReflectionCoefficient.from_xarray(
            _table(reflection_type='transmission').to_xarray())
        assert back.reflection_type == 'transmission'
        assert 'reflection_type' not in back.metadata

    def test_a_saved_dict_keeping_it_in_metadata_is_refused(self):
        d = _table().to_dict()
        d['metadata'] = {'reflection_type': 'transmission'}
        with pytest.raises(ConfigurationError, match='reflection_type'):
            ReflectionCoefficient.from_dict(d)

    def test_a_metadata_entry_naming_it_is_refused(self):
        with pytest.raises(ConfigurationError, match='reflection_type'):
            _table(metadata={'reflection_type': 'transmission'})
