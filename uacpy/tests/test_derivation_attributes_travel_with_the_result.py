"""What a reader acts on travels as a typed attribute of the result:
the Field's derivation record (band_hz, synthesis_window, sub_cutoff_bins,
sonar_budget, sigma_dB), Arrivals.absorption and
ReflectionCoefficient.reflection_type. Each is carried to a derived result,
round-trips through to_dict and the xarray export, loads from the metadata
spelling an older file kept it under, and is refused as a metadata key."""
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
    'band_hz': ((100.0, 300.0), 'band_hz'),
    'synthesis_window': ('hann', 'window'),
    'sub_cutoff_bins': (2, 'sub_cutoff_bins'),
    'sonar_budget': (BUDGET, 'sonar_budget'),
    'sigma_dB': (5.6, 'sigma_dB'),
}


def _field(**kw):
    return Field(data=np.full((2, 3), 60.0),
                 coords={'depth': np.array([10.0, 20.0]),
                         'range': np.array([100.0, 200.0, 300.0])},
                 kind='pressure', unit='dB', **kw)


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

    def test_an_older_file_keeping_it_in_metadata_loads_it(self, tag):
        value, spelling = DERIVATION[tag]
        d = _field().to_dict()
        d['metadata'] = {spelling: value, 'title': 'kept'}
        back = Field.from_dict(d)
        assert getattr(back, tag) == value
        assert back.metadata == {'title': 'kept'}

    def test_a_metadata_entry_naming_it_is_refused(self, tag):
        value, spelling = DERIVATION[tag]
        with pytest.raises(ConfigurationError, match=spelling):
            _field(metadata={spelling: value})


def test_a_band_that_is_not_a_pair_is_refused():
    with pytest.raises(ConfigurationError, match='band_hz'):
        _field(band_hz=(100.0, 200.0, 300.0))


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

    def test_an_older_file_keeping_it_in_metadata_loads_it(self):
        d = _arrivals().to_dict()
        d['metadata'] = {'absorption': LAW}
        back = Arrivals.from_dict(d)
        assert back.absorption is LAW
        assert not back.metadata

    def test_a_metadata_entry_naming_it_is_refused(self):
        with pytest.raises(ConfigurationError, match='absorption'):
            _arrivals(metadata={'absorption': LAW})


def _table(**kw):
    return ReflectionCoefficient(angles=[10.0, 20.0, 30.0],
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

    def test_an_older_file_keeping_it_in_metadata_loads_it(self):
        d = _table().to_dict()
        d['metadata'] = {'reflection_type': 'transmission'}
        assert ReflectionCoefficient.from_dict(d).reflection_type == \
            'transmission'

    def test_a_metadata_entry_naming_it_is_refused(self):
        with pytest.raises(ConfigurationError, match='reflection_type'):
            _table(metadata={'reflection_type': 'transmission'})
