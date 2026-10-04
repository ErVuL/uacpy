"""Plausibility warnings on the seabed carriers and Francois-Garrison.

A value the deck accepts but no ocean has: a density typed in kg/m³ into a
g/cm³ field, a shear speed above the compressional speed, a Francois-Garrison
row or frequency outside the envelope the equation was fitted over. Each is a
``UserWarning`` — never an error, since none of them breaks a run — and each
threshold is pinned on both sides here: the last value inside stays silent,
the first value outside warns, and the remedy the message names is typed back
and shown to be silent.

Every warning has to name the *caller's* line, from a direct constructor call
and from the in-package doors alike (``from_preset``, the fetched-row builder,
a model's absorption accessor): ``warnings`` keys its once-per-location dedup
on the attributed file and line, so a warning that named a uacpy line would be
shown for the first caller in a program and swallowed for every other.
"""

import dataclasses
import inspect
import warnings
from pathlib import Path

import numpy as np
import pytest

from uacpy.core.absorption import FrancoisGarrison, Thorp
from uacpy.core.boundary import (
    BoundaryProperties, SedimentLayer, _DENSITY_UNITS_SUSPECT_G_CM3,
)
from uacpy.core.bottom import SeabedColumn
from uacpy.core.exceptions import ValidityWarning
from uacpy.core._validate import SPEED_UNITS_SUSPECT_M_S
from uacpy.core.ssp import SoundSpeedProfile
from uacpy.core.materials import MATERIALS
from uacpy.tests.conftest import recorded_warnings

_THIS_FILE = Path(__file__).resolve()


def _recorded(call):
    """The warnings ``call()`` emits, under the always-show filter."""
    with recorded_warnings() as record:
        call()
    return record


def _only(record):
    assert len(record) == 1, [str(w.message) for w in record]
    assert record[0].category is ValidityWarning
    return record[0]


def _silent(record):
    assert [str(w.message) for w in record] == []


# --------------------------------------------------------------------------
# Seabed carriers: density in the wrong unit, shear above compressional
# --------------------------------------------------------------------------

def _layer(**kw):
    base = dict(thickness=5.0, sound_speed=1600.0, density=1.8)
    base.update(kw)
    return SedimentLayer(**base)


def _halfspace(**kw):
    base = dict(sound_speed=1600.0, density=1.8)
    base.update(kw)
    return BoundaryProperties(**base)


@pytest.mark.parametrize('build', [_layer, _halfspace],
                         ids=['SedimentLayer', 'BoundaryProperties'])
class TestDensityTypedInKgPerCubicMetre:

    def test_a_kg_per_m3_value_warns_and_names_the_g_per_cm3_it_means(
            self, build):
        warning = _only(_recorded(lambda: build(density=1500.0)))
        message = str(warning.message)
        assert 'density=1500 looks like kg/m³' in message
        assert 'uacpy takes g/cm³ (1.5)' in message

    def test_the_first_value_over_the_bound_warns(self, build):
        assert _DENSITY_UNITS_SUSPECT_G_CM3 == 20.0
        _only(_recorded(lambda: build(density=20.001)))

    def test_the_bound_itself_and_everything_under_it_stay_silent(
            self, build):
        _silent(_recorded(lambda: build(density=20.0)))
        _silent(_recorded(lambda: build(density=19.99)))

    def test_the_remedy_typed_back_is_silent(self, build):
        # The message quotes 1.5 for 1500: the value re-entered as g/cm³.
        _silent(_recorded(lambda: build(density=1.5)))

    def test_the_value_is_stored_as_given_not_corrected(self, build):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert build(density=1500.0).density == 1500.0


@pytest.mark.parametrize('build', [_layer, _halfspace],
                         ids=['SedimentLayer', 'BoundaryProperties'])
class TestShearSpeedAboveCompressional:

    def test_shear_over_compressional_warns_naming_both_numbers(self, build):
        warning = _only(_recorded(
            lambda: build(sound_speed=1600.0, shear_speed=1700.0)))
        message = str(warning.message)
        assert 'shear_speed=1700 m/s' in message
        assert 'sound_speed=1600 m/s' in message
        assert 'sqrt(2)' in message

    def test_the_first_value_over_the_bound_warns(self, build):
        _only(_recorded(
            lambda: build(sound_speed=1600.0, shear_speed=1600.001)))

    def test_equal_speeds_and_everything_under_stay_silent(self, build):
        _silent(_recorded(
            lambda: build(sound_speed=1600.0, shear_speed=1600.0)))
        _silent(_recorded(
            lambda: build(sound_speed=1600.0, shear_speed=1599.0)))

    def test_the_remedy_typed_back_is_silent(self, build):
        # The message asks whether the two were swapped; swapped back, the
        # same pair is an ordinary elastic rock.
        _silent(_recorded(
            lambda: build(sound_speed=1700.0, shear_speed=1600.0)))

    def test_a_fluid_layer_never_trips_it(self, build):
        _silent(_recorded(lambda: build(sound_speed=1600.0, shear_speed=0.0)))


def _ssp(speed):
    return SoundSpeedProfile(depths=[0.0, 100.0], sound_speed=[1500.0, speed])


@pytest.mark.parametrize('build', [
    lambda c: _layer(sound_speed=c), lambda c: _halfspace(sound_speed=c),
    _ssp,
], ids=['SedimentLayer', 'BoundaryProperties', 'SoundSpeedProfile'])
class TestSoundSpeedTypedInKmPerSecond:

    def test_a_km_per_s_value_warns_and_names_the_m_per_s_it_means(
            self, build):
        warning = _only(_recorded(lambda: build(1.49)))
        message = str(warning.message)
        assert '1.49 m/s looks like km/s' in message
        assert 'uacpy takes m/s (1490)' in message

    def test_the_first_value_under_the_bound_warns(self, build):
        assert SPEED_UNITS_SUSPECT_M_S == 10.0
        _only(_recorded(lambda: build(9.999)))

    def test_the_bound_itself_and_everything_over_it_stay_silent(
            self, build):
        _silent(_recorded(lambda: build(10.0)))
        _silent(_recorded(lambda: build(10.001)))

    def test_the_remedy_typed_back_is_silent(self, build):
        _silent(_recorded(lambda: build(1490.0)))


class TestSeabedWarningsStayOffTheOrdinaryPaths:

    @pytest.mark.parametrize('name', sorted(MATERIALS))
    def test_every_catalogue_preset_is_silent_as_a_fluid_and_as_a_solid(
            self, name):
        _silent(_recorded(lambda: BoundaryProperties.from_preset(name)))
        _silent(_recorded(
            lambda: BoundaryProperties.from_preset(name, elastic=True)))
        _silent(_recorded(
            lambda: SedimentLayer.from_preset(name, thickness=5.0,
                                              elastic=True)))

    @pytest.mark.parametrize('acoustic_type', ['vacuum', 'rigid'])
    def test_the_parameter_free_boundaries_are_silent(self, acoustic_type):
        _silent(_recorded(
            lambda: BoundaryProperties(acoustic_type=acoustic_type)))

    def test_a_file_boundary_is_silent(self, tmp_path):
        brc = tmp_path / 'table.brc'
        brc.write_text('1\n0.0 1.0 0.0\n', encoding='utf-8')
        _silent(_recorded(lambda: BoundaryProperties(
            acoustic_type='file', reflection_file=str(brc))))

    def test_a_refusal_wins_over_a_warning(self):
        """A density that is also non-positive is refused, not warned about."""
        from uacpy.core.exceptions import ConfigurationError
        with recorded_warnings() as record:
            with pytest.raises(ConfigurationError,
                               match='density must be positive'):
                SedimentLayer(thickness=1.0, sound_speed=1600.0, density=-1500.0)
        _silent(record)


class TestSeabedWarningsNameTheCallersLine:

    def test_a_direct_constructor_call_lands_here(self):
        for build in (_layer, _halfspace):
            warning = _only(_recorded(lambda: build(density=1500.0)))
            assert warning.filename != '<string>'
            assert Path(warning.filename).resolve() == _THIS_FILE

    def test_from_preset_with_an_override_lands_here(self):
        """The in-package door: ``from_preset`` builds the object one
        ``bottom.py`` frame down from the user, which the skip walk steps
        over — and which a generated ``__init__``'s ``<string>`` frame would
        have stopped on."""
        warning = _only(_recorded(
            lambda: BoundaryProperties.from_preset('sand', density=1900.0)))
        assert Path(warning.filename).resolve() == _THIS_FILE

    def test_a_layer_collapsed_onto_a_half_space_lands_here(self):
        """``SeabedColumn.collapse_layers('top_layer')`` rebuilds a
        ``BoundaryProperties`` from the layer's numbers — a second door two
        frames down."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            column = SeabedColumn(
                layers=[SedimentLayer(thickness=2.0, sound_speed=1600.0,
                                      density=1800.0)],
                halfspace=BoundaryProperties(sound_speed=1700.0, density=1.9))
        warning = _only(_recorded(lambda: column.collapse_layers('top_layer')))
        assert Path(warning.filename).resolve() == _THIS_FILE

    def test_two_call_sites_each_warn_under_the_default_filter(self):
        """What the attribution buys: the once-per-location registry keys on
        the caller's line, so two lines get two warnings."""
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('default')
            SedimentLayer(thickness=5.0, sound_speed=1600.0, density=1500.0)
            SedimentLayer(thickness=5.0, sound_speed=1600.0, density=1500.0)
        assert len(record) == 2


@pytest.mark.parametrize('cls', [SedimentLayer, BoundaryProperties],
                         ids=['SedimentLayer', 'BoundaryProperties'])
def test_the_carrier_init_takes_exactly_the_dataclass_fields(cls):
    """``@carrier`` builds each carrier's ``__init__`` from its dataclass
    fields, as an ordinary package function, so no ``<string>`` frame sits
    between the warnings above and the user. Its signature lists the
    fields in order, so ``help()`` and ``inspect`` describe the constructor
    Python runs."""
    parameters = list(inspect.signature(cls.__init__).parameters)[1:]
    assert parameters == [f.name for f in dataclasses.fields(cls)]


def test_the_carriers_compare_copy_and_repr_as_dataclasses():
    a = _halfspace(shear_speed=300.0)
    assert a == a.copy() and a is not a.copy()
    assert a != _halfspace(shear_speed=301.0)
    assert repr(a).startswith('BoundaryProperties(')
    layer = _layer(name='mud')
    assert layer == layer.copy()
    assert repr(layer).startswith("SedimentLayer('mud'")


# --------------------------------------------------------------------------
# Francois-Garrison: the fitted envelope
# --------------------------------------------------------------------------

def _fg(**kw):
    base = dict(temperature=10.0, salinity=35.0, pH=8.0)
    base.update(kw)
    return FrancoisGarrison(**base)


class TestFrancoisGarrisonRowEnvelope:
    """Francois & Garrison (1982) Part II §III: the boric-acid term was fitted
    at 34-41 ‰, 2-22 °C; Table IV tabulates -1.8 to 30 °C at 30 and 35 ‰; the
    MgSO4 field data span 30-35 ‰ (APL-UW TR 9407 §I.B), so the salinity
    range is their union, 30-41 ‰; seawater pH runs 7.7-8.3 (Mellen et al.
    1987). The bounds here are inclusive."""

    @pytest.mark.parametrize('field,inside,outside,text', [
        ('temperature', -2.0, -2.001, 'temperature=-2.001 is outside -2..30 °C'),
        ('temperature', 30.0, 30.001, 'temperature=30.001 is outside -2..30 °C'),
        ('salinity', 30.0, 29.99, 'salinity=29.99 is outside 30..41 PSU'),
        ('salinity', 41.0, 41.01, 'salinity=41.01 is outside 30..41 PSU'),
        ('pH', 7.7, 7.699, 'pH=7.699 is outside 7.7..8.3'),
        ('pH', 8.3, 8.301, 'pH=8.301 is outside 7.7..8.3'),
    ])
    def test_each_bound_is_silent_on_it_and_warns_just_past_it(
            self, field, inside, outside, text):
        _silent(_recorded(lambda: _fg(**{field: inside})))
        warning = _only(_recorded(lambda: _fg(**{field: outside})))
        message = str(warning.message)
        assert text in message
        assert 'Francois & Garrison 1982 Part II' in message
        assert 'used as given' in message

    def test_a_salinity_warning_names_the_terms_its_bounds_come_from(self):
        """A 42 ‰ is outside both salinity terms' data, so the message
        names each term's range and their source and claims no accuracy
        figure for outside."""
        message = str(_only(_recorded(
            lambda: _fg(salinity=42.0))).message)
        assert 'boric acid at 34-41, MgSO4 at 30-35' in message
        assert 'TR 9407 §I.B' in message
        assert 'none outside' not in message

    def test_a_salinity_inside_the_boric_acid_data_only_is_silent(self):
        # A Mediterranean 38.5 ‰ is past the MgSO4 term's 30-35 ‰ but
        # inside the boric-acid term's 34-41 ‰, so inside the union.
        _silent(_recorded(lambda: _fg(salinity=38.5)))

    def test_a_row_out_on_every_axis_is_reported_in_one_warning(self):
        warning = _only(_recorded(
            lambda: _fg(temperature=-5.0, salinity=7.0, pH=8.5)))
        message = str(warning.message)
        assert 'temperature=-5' in message
        assert 'salinity=7' in message
        assert 'pH=8.5' in message

    def test_the_remedy_typed_back_is_silent(self):
        # A Baltic row (7 ‰) warns; the value the message ranges name, 30,
        # re-entered, does not.
        _only(_recorded(lambda: _fg(salinity=7.0)))
        _silent(_recorded(lambda: _fg(salinity=30.0)))

    def test_the_value_is_used_as_given(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fg = _fg(salinity=7.0)
        assert fg.salinity == 7.0
        assert np.isfinite(fg.alpha_dB_per_m(1000.0, 0.0)).all()

    def test_ph_is_compared_on_the_nbs_scale_and_the_message_says_so(self):
        """A total-scale 8.3 is about 8.43 on NBS — the scale the equation
        was fitted on — so it is outside, and the message shows both."""
        warning = _only(_recorded(lambda: _fg(pH=8.3, ph_scale='total')))
        message = str(warning.message)
        assert "from 8.3 'total'" in message
        assert 'on the NBS scale' in message
        # And a total-scale value that converts to inside stays silent.
        _silent(_recorded(lambda: _fg(pH=7.9, ph_scale='total')))

    def test_a_refusal_wins_over_the_envelope_warning(self):
        from uacpy.core.exceptions import ConfigurationError
        with recorded_warnings() as record:
            with pytest.raises(ConfigurationError,
                               match='salinity must be non-negative'):
                _fg(salinity=-1.0)
        _silent(record)


class TestFrancoisGarrisonFrequencyEnvelope:
    """Part II §III: "the equation may not hold below 200 Hz"; Table IV stops
    at 1000 kHz."""

    def test_the_bounds_are_silent_and_just_past_them_warns(self):
        fg = _fg()
        _silent(_recorded(lambda: fg.alpha_dB_per_m(200.0, 0.0)))
        _silent(_recorded(lambda: fg.alpha_dB_per_m(1.0e6, 0.0)))
        low = _only(_recorded(lambda: fg.alpha_dB_per_m(199.99, 0.0)))
        assert '1 of 1 frequency (199.99 Hz) is below 200 Hz' in str(
            low.message)
        assert 'outside the 200 Hz..1e+06 Hz' in str(low.message)
        assert 'may not hold below 200 Hz' in str(low.message)
        high = _only(_recorded(lambda: fg.alpha_dB_per_m(1.0e6 + 1.0, 0.0)))
        assert '(1000001 Hz) is above 1e+06 Hz' in str(high.message)

    def test_a_frequency_vector_gets_one_notice_with_its_counts(self):
        """Hundreds of lines for one ``alpha`` call over a wide band was the
        defect: the notice fires once, naming how many samples fall outside
        on each side and the span they cover."""
        fg = _fg()
        f = np.logspace(1.0, 6.2, 500)
        below, above = f[f < 200.0], f[f > 1.0e6]
        warning = _only(_recorded(lambda: fg.table(f)))
        text = str(warning.message)
        assert (f"{below.size} of 500 frequencies ({below.min():.10g}-"
                f"{below.max():.10g} Hz) are below 200 Hz") in text
        assert (f"{above.size} of 500 frequencies ({above.min():.10g}-"
                f"{above.max():.10g} Hz) are above 1e+06 Hz") in text

    def test_an_in_band_vector_is_silent_and_200_hz_exactly_is_in_band(self):
        fg = _fg()
        _silent(_recorded(lambda: fg.table(np.linspace(200.0, 1.0e6, 50))))
        edge = _only(_recorded(lambda: fg.table(
            np.array([199.99, 200.0, 5000.0]))))
        assert '1 of 3 frequencies (199.99 Hz) is below' in str(edge.message)

    def test_the_value_is_the_polynomial_as_given(self):
        fg = _fg()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            below = fg.alpha_dB_per_m(100.0, np.array([0.0, 500.0]))
        assert np.all(np.isfinite(below)) and np.all(below > 0.0)

    def test_thorp_has_no_envelope_notice(self):
        _silent(_recorded(lambda: Thorp().alpha_dB_per_m(50.0, 0.0)))


class TestFrancoisGarrisonWarningsNameTheCallersLine:

    def test_a_direct_constructor_call_lands_here(self):
        warning = _only(_recorded(lambda: _fg(salinity=7.0)))
        assert warning.filename != '<string>'
        assert Path(warning.filename).resolve() == _THIS_FILE

    def test_the_fetched_row_builder_lands_here(self):
        """``FrancoisGarrison.from_temperature_salinity`` constructs the
        model one package frame down — the door a Baltic or Arctic fetch
        comes through."""
        warning = _only(_recorded(lambda: FrancoisGarrison.from_temperature_salinity(
            depths=[0.0, 50.0], temperature=[5.0, 4.0], salinity=[7.0, 7.5])))
        assert Path(warning.filename).resolve() == _THIS_FILE

    def test_the_frequency_notice_through_the_public_accessor_lands_here(self):
        fg = _fg()
        warning = _only(_recorded(lambda: fg.alpha_dB_per_m(100.0, 0.0)))
        assert Path(warning.filename).resolve() == _THIS_FILE

    def test_two_call_sites_each_warn_under_the_default_filter(self):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('default')
            FrancoisGarrison(10.0, 7.0, 8.0)
            FrancoisGarrison(10.0, 7.0, 8.0)
        assert len(record) == 2


def test_the_francois_garrison_init_takes_exactly_the_dataclass_fields():
    parameters = list(inspect.signature(FrancoisGarrison.__init__).parameters)[1:]
    assert parameters == [f.name for f in dataclasses.fields(FrancoisGarrison)]
    # Positional construction, the form the docstrings use, works.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert FrancoisGarrison(10, 35, 8, 'nbs') == _fg(ph_scale='nbs')
