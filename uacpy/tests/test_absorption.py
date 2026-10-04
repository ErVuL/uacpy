"""Volume absorption (``uacpy.core.absorption``) and the Francois-Garrison
builder.

``FrancoisGarrison.from_temperature_salinity`` builds the law from a
measured temperature/salinity column, whole or collapsed to one row.

``uacpy.core.absorption`` answers for the rows its formulas cover.
Francois-Garrison is an empirical fit with a stated domain, and outside it
the expression goes complex or divides by zero. A row that evaluates to NaN
is refused by name rather than returned, and the boundary values of the
domain are accepted — both sides pinned, so the guard cannot be loosened or
tightened without a test moving.

The Thorp and Francois-Garrison formulas are checked against values
evaluated from the published expressions, and ``convert_attenuation_units``
against its closed forms. The biological layers, ``ConstantAbsorption`` and
the shared frequency guard are pinned at their boundaries.

No binary runs here; this is arithmetic and validation.
"""

import dataclasses
import inspect
import re
import warnings

import numpy as np
import pytest

from uacpy.core.absorption import Biological, BiologicalLayer, FrancoisGarrison
from uacpy.core.acoustics.attenuation import (
    ph_to_nbs, convert_attenuation_units, absorption_francois_garrison,
)
from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.deck_limits import MAX_ATTENUATION_DB_PER_WAVELENGTH
from uacpy.core.exceptions import ConfigurationError
from uacpy.io.at_codes import volume_attenuation_code
import math
from uacpy.tests.conftest import recorded_warnings
from uacpy.core import Environment
from uacpy.core.absorption import ConstantAbsorption, Thorp
from uacpy.core.exceptions import ProvenanceWarning, ValidityWarning
from uacpy.io.at_codes import writes_alpha_per_ssp_row
from uacpy.tests.conftest import (
    TWO_LAYER_DEPTHS, measured_absorption_table, two_layer_absorption,
)


def test_builds_the_whole_column_as_a_profile_by_default():
    """Every sample of the column is kept, so every depth is evaluated with
    its own water."""
    fg = FrancoisGarrison.from_temperature_salinity([0.0, 50.0, 100.0], [18.0, 16.0, 13.0],
                                 [36.0, 36.1, 36.2])
    assert isinstance(fg, FrancoisGarrison)
    assert fg.is_profile
    np.testing.assert_array_equal(fg.profile_depths, [0.0, 50.0, 100.0])
    np.testing.assert_array_equal(fg.temperature, [[0.0, 18.0], [50.0, 16.0],
                                                   [100.0, 13.0]])
    np.testing.assert_array_equal(fg.salinity[:, 1], [36.0, 36.1, 36.2])
    assert fg.depths is None              # consumed into each property
    assert fg.pH == 8.0                   # the reference-water default


def test_collapse_to_depth_keeps_the_nearest_row():
    fg = FrancoisGarrison.from_temperature_salinity([0.0, 50.0, 100.0], [18.0, 16.0, 13.0],
                                 [36.0, 36.1, 36.2], collapse_to_depth=60.0,
                                 pH=7.9)
    assert not fg.is_profile
    assert fg.temperature == 16.0       # the 50 m row, nearest to 60 m
    assert fg.salinity == 36.1
    assert fg.pH == 7.9


def test_mismatched_or_empty_raises():
    with pytest.raises(ConfigurationError,
                       match='must be non-empty and equal length'):
        FrancoisGarrison.from_temperature_salinity([], [], [])
    with pytest.raises(ConfigurationError,
                       match='must be non-empty and equal length'):
        FrancoisGarrison.from_temperature_salinity([0.0, 10.0], [18.0], [36.0])


def test_a_uniform_profile_is_the_one_row_law_exactly():
    """A profile whose water is the same at every row is the one-row law,
    bit for bit, at its rows, between them and beyond them: both evaluate
    the formula's depth term at the depth asked."""
    row = FrancoisGarrison(temperature=6.5, salinity=34.2, pH=7.95)
    profile = FrancoisGarrison(temperature=[6.5, 6.5, 6.5],
                               salinity=[34.2, 34.2, 34.2], pH=7.95,
                               depths=[0.0, 300.0, 1200.0])
    z = np.array([0.0, 150.0, 300.0, 777.0, 1200.0, 3000.0])
    for f in (300.0, 3e3, 3e4):
        np.testing.assert_array_equal(profile.alpha_dB_per_m(f, z),
                                      row.alpha_dB_per_m(f, z))
    assert np.all(np.diff(row.alpha_dB_per_m(3e4, z)) != 0.0)


#: Temperature and salinity on depths of their own, pH a number.
_T_PAIRS = [(0.0, 22.0), (30.0, 21.0), (60.0, 12.0), (200.0, 10.0)]
_S_PAIRS = [(0.0, 35.0), (150.0, 34.6)]


class TestEachWaterPropertyHasItsOwnDepths:
    """``temperature``, ``salinity`` and ``pH`` each take a number, a 1-D
    array on the shared ``depths=``, or ``(depth, value)`` pairs on depths of
    their own; each is interpolated on its own axis at the depth
    evaluated."""

    def test_each_property_is_interpolated_on_its_own_axis(self):
        law = FrancoisGarrison(temperature=_T_PAIRS, salinity=_S_PAIRS,
                               pH=8.0)
        z = np.array([0.0, 15.0, 45.0, 100.0, 175.0, 400.0])
        t = np.interp(z, *np.array(_T_PAIRS).T)
        s = np.interp(z, *np.array(_S_PAIRS).T)
        want = absorption_francois_garrison(1e4, t, s, 8.0, z) / 1000.0
        np.testing.assert_allclose(law.alpha_dB_per_m(1e4, z), want,
                                   rtol=1e-13)
        np.testing.assert_array_equal(law.profile_depths,
                                      [0.0, 30.0, 60.0, 150.0, 200.0])

    def test_the_three_forms_mix_and_agree(self):
        # A 1-D array on depths= is the same pairs on those depths.
        z = [0.0, 30.0, 60.0, 200.0]
        mixed = FrancoisGarrison(temperature=[22.0, 21.0, 12.0, 10.0],
                                 salinity=_S_PAIRS, pH=8.0, depths=z)
        pairs = FrancoisGarrison(temperature=_T_PAIRS, salinity=_S_PAIRS,
                                 pH=8.0)
        assert mixed == pairs and mixed.depths is None
        assert FrancoisGarrison(**pairs.table([1e3]).parameters) == pairs

    @pytest.mark.parametrize('kwargs, match', [
        (dict(temperature=[10.0, 9.0]), 'needs the depths='),
        (dict(depths=[0.0, 10.0]), 'none is'),
        (dict(temperature=[(0.0, 10.0), (0.0, 9.0)]), 'strictly increasing'),
        (dict(salinity=[(10.0, 35.0), (-1.0, 35.0)]), 'strictly increasing'),
        (dict(pH=np.ones((2, 3))), r'shape \(N, 2\)'),
        (dict(temperature=[10.0, 9.0], depths=[0.0, 5.0, 9.0]),
         '2 values for 3 depths')])
    def test_a_malformed_property_is_refused(self, kwargs, match):
        with pytest.raises(ConfigurationError, match=match):
            FrancoisGarrison(**kwargs)

    def test_the_envelope_warns_once_for_the_law(self):
        with recorded_warnings() as rec:
            FrancoisGarrison(temperature=[(0.0, 31.0), (50.0, 20.0)],
                             salinity=[(0.0, 42.0), (10.0, 34.0),
                                       (20.0, 34.0)], pH=8.0)
        (msg,) = [str(w.message) for w in rec
                  if issubclass(w.category, ValidityWarning)]
        assert 'temperature=31 at 1 of 2 depths' in msg
        assert 'salinity=42 at 1 of 3 depths' in msg

    def test_the_repr_shows_each_span_and_depth_count(self):
        law = FrancoisGarrison(temperature=_T_PAIRS, salinity=_S_PAIRS,
                               pH=8.0)
        assert repr(law) == ('FrancoisGarrison(T 10–22 °C (4 depths), '
                             'S 34.6–35 psu (2 depths), pH 8)')

    def test_from_temperature_salinity_takes_ph_pairs(self):
        ph = np.array([(0.0, 8.1), (500.0, 8.0)])
        law = FrancoisGarrison.from_temperature_salinity(
            [0.0, 100.0], [12.0, 8.0], [35.0, 35.0], pH=ph, ph_scale='total')
        np.testing.assert_array_equal(law.pH, ph)
        row = FrancoisGarrison.from_temperature_salinity(
            [0.0, 100.0], [12.0, 8.0], [35.0, 35.0], pH=ph,
            collapse_to_depth=100.0)
        assert row.pH == pytest.approx(8.08) and row.temperature == 8.0


class TestTheColumnIsKeptWhole:
    """A single T/S row governs the absorption of the whole column: on a
    mid-latitude column (22 °C surface, 4 °C at 2 km) its temperature is
    wrong everywhere but at its own depth. The default keeps the column, so
    each depth gets its own water; ``collapse_to_depth=`` still takes one
    row."""

    Z = np.array([0.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0])
    T = np.array([22.0, 20.0, 16.0, 12.0, 8.0, 5.0, 4.0])
    S = np.full(7, 35.0)

    def _fg(self, ref=None):
        return FrancoisGarrison.from_temperature_salinity(self.Z, self.T, self.S,
                                       collapse_to_depth=ref)

    def test_default_evaluates_each_depth_with_its_own_water(self):
        fg = self._fg()
        alpha = fg.alpha_dB_per_m(1e4, self.Z)
        for z, t, a in zip(self.Z, self.T, alpha):
            expected = absorption_francois_garrison(1e4, t, 35.0, 8.0, z)
            assert a == pytest.approx(float(expected) / 1000.0, rel=1e-14)

    def test_a_collapse_depth_takes_one_row(self):
        assert self._fg(ref=0.0).temperature == pytest.approx(22.0)
        assert self._fg(ref=1000.0).temperature == pytest.approx(5.0)

    def test_one_surface_row_understates_the_column_at_depth(self):
        # The size of what the profile fixes, in dB: the surface row at
        # 500 m carries 22 °C water where the column holds 8 °C.
        zq = np.array([500.0])
        a_col = float(np.ravel(self._fg().alpha_dB_per_m(1e4, zq))[0])
        a_surf = float(np.ravel(self._fg(ref=0.0).alpha_dB_per_m(1e4, zq))[0])
        assert a_surf < a_col
        assert (a_col - a_surf) / a_col == pytest.approx(0.30, abs=0.02)

    def test_isothermal_column_matches_its_one_row_exactly(self):
        # Where there is no stratification the profile and any one row are
        # the same law, bit for bit.
        z = np.array([0.0, 100.0, 500.0])
        t = np.full(3, 12.0)
        s = np.full(3, 35.0)
        row = FrancoisGarrison.from_temperature_salinity(z, t, s, collapse_to_depth=0.0)
        column = FrancoisGarrison.from_temperature_salinity(z, t, s)
        zq = np.linspace(0.0, 500.0, 11)
        np.testing.assert_array_equal(column.alpha_dB_per_m(3e3, zq),
                                      row.alpha_dB_per_m(3e3, zq))


# (kwargs, the fragment of the message that names the offending field)
_REFUSED_ROWS = [
    (dict(temperature=10.0, salinity=-1e-12, pH=8.0),
     'salinity must'),
    (dict(temperature=-273.0, salinity=35.0, pH=8.0),
     'temperature must'),
    (dict(temperature=10.0, salinity=35.0, pH=-1e-12),
     'pH'),
    (dict(temperature=10.0, salinity=35.0, pH=float('nan')),
     'pH'),
    (dict(temperature=10.0, salinity=35.0, pH=-0.1),
     'which no water reaches'),
    (dict(temperature=10.0, salinity=35.0, pH=14.1),
     'which no water reaches'),
    # -0.1 on the total scale is about +0.05 on NBS: the input itself is
    # what no water has.
    (dict(temperature=25.0, salinity=35.0, pH=-0.1,
          ph_scale='total'),
     'which no water reaches'),
    # A total-scale 13.9 is 13.9 + 0.147 on the NBS scale at 25 °C / 35.
    (dict(temperature=25.0, salinity=35.0, pH=13.9,
          ph_scale='total'),
     'on the NBS scale'),
    # The row from the report: every field out of range at once.
    (dict(temperature=-500.0, salinity=-10.0, pH=-3.0),
     'salinity must'),
]


@pytest.mark.parametrize('kwargs,names', _REFUSED_ROWS)
def test_francois_garrison_refuses_rows_that_evaluate_to_nan(kwargs, names):
    with pytest.raises(ConfigurationError, match=re.escape(names)):
        FrancoisGarrison(**kwargs)


# The other side of each threshold: the last value that is still a number.
_ACCEPTED_ROWS = [
    dict(temperature=10.0, salinity=0.0, pH=8.0),
    dict(temperature=-272.999, salinity=35.0, pH=8.0),
    dict(temperature=10.0, salinity=35.0, pH=0.0),
    dict(temperature=10.0, salinity=35.0, pH=14.0),
    dict(temperature=25.0, salinity=35.0, pH=13.8,
         ph_scale='total'),
    # The ordinary mid-latitude row every other test in the suite uses.
    dict(temperature=10.0, salinity=35.0, pH=8.0),
]


@pytest.mark.parametrize('kwargs', _ACCEPTED_ROWS)
def test_francois_garrison_accepts_the_boundary_values(kwargs):
    absorption = FrancoisGarrison(**kwargs)
    assert writes_alpha_per_ssp_row(absorption)


@pytest.mark.parametrize('bad', [np.inf, -np.inf, np.nan])
@pytest.mark.parametrize(
    'name', ['temperature', 'salinity', 'pH'])
def test_francois_garrison_refuses_a_non_finite_field_naming_it(name, bad):
    """inf passes every range guard: inf T or S evaluate to a NaN alpha,
    inf pH to an inf alpha. The finiteness check runs ahead of the range
    guards so the message names the field, not the sound speed it feeds."""
    kwargs = dict(temperature=10.0, salinity=35.0, pH=8.0)
    kwargs[name] = bad
    with pytest.raises(ConfigurationError,
                       match=re.escape(name) + '.*must be finite'):
        FrancoisGarrison(**kwargs)


def test_the_bare_formula_answers_an_out_of_domain_row_with_nan_only():
    """The module-level formula keeps its no-validation contract — but the
    NaN comes back without numpy's raw ``RuntimeWarning``, which would be the
    one warning uacpy emits that is not a ``UserWarning``."""
    with recorded_warnings() as record:
        alpha = absorption_francois_garrison(
            1000.0, temperature=-500.0, salinity=-10.0, pH=-3.0, depth=50.0)
    assert np.isnan(alpha)
    assert record == [], [str(w.message) for w in record]


def test_an_in_domain_row_is_unchanged_by_the_errstate_guard():
    """Silencing the invalid flag must not touch the numbers."""
    alpha = absorption_francois_garrison(
        10_000.0, temperature=10.0, salinity=35.0, pH=8.0, depth=1000.0)
    assert 0.0 < float(alpha) < 10.0


def _biological_layer(**kw):
    """A 0-50 m BiologicalLayer at f0 = 100 Hz, Q = 10, a0 = 1; keywords override."""
    args = {'z_top_m': 0.0, 'z_bottom_m': 50.0, 'f0_hz': 100.0,
            'Q': 10.0, 'a0': 1.0}
    args.update(kw)
    return BiologicalLayer(**args)


class TestBiologicalLayerMeetsTheCrciCeiling:
    """``AttenMod.f90``'s ``'B'`` branch (:105-106) adds ``a/8685.8896``
    Nepers/m, :113 scales by ``c²/ω`` and :116 aborts once the result passes
    ``c``. The Lorentzian peaks at ``f = f0``, where its denominator is
    ``1/Q²``, so the layer presents at most ``a0·Q²`` dB/km and aborts every
    AT solver above ``8685.8896·2πf0/c`` — 3638 dB/km at 100 Hz in 1500 m/s
    water. The three sibling attenuation carriers are all held to this
    ceiling via ``_require_attenuation_in_range``; this one warns instead of
    raising, because the peak is only reached with the run frequency on
    ``f0`` and the ceiling scales with the true ``c(z)`` over the layer."""

    @staticmethod
    def _ceiling_dB_km(f0):
        return (MAX_ATTENUATION_DB_PER_WAVELENGTH * 1000.0 * f0
                / DEFAULT_SOUND_SPEED)

    def test_the_documented_at_threshold_is_3638_dB_per_km_at_100hz(self):
        assert self._ceiling_dB_km(100.0) == pytest.approx(3638.34, abs=0.01)

    def test_a_peak_over_the_ceiling_warns(self):
        with pytest.warns(UserWarning, match='CRCI'):
            _biological_layer(Q=61.0, a0=1.0)      # a0·Q² = 3721 dB/km

    def test_a_peak_under_the_ceiling_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _biological_layer(Q=60.0, a0=1.0)      # a0·Q² = 3600 dB/km

    def test_the_warning_names_the_peak_and_the_ceiling(self):
        with pytest.warns(UserWarning) as rec:
            _biological_layer(Q=61.0, a0=1.0)
        message = str(rec[0].message)
        assert '3721' in message
        assert '3638' in message

    def test_the_ceiling_scales_with_the_resonance_frequency(self):
        """The bound is on Nepers/m against ω/c, so ten times the resonance
        frequency buys ten times the dB/km — the same layer that warns at
        100 Hz is comfortable at 1000 Hz."""
        with pytest.warns(UserWarning, match='CRCI'):
            _biological_layer(f0_hz=100.0, Q=61.0, a0=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _biological_layer(f0_hz=1000.0, Q=61.0, a0=1.0)

    def test_the_layer_is_built_and_computes(self):
        """A warning, not a refusal: the object exists and its Lorentzian is
        untouched."""
        from uacpy.core.absorption import Biological
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            bio = Biological(layers=[(0.0, 50.0, 100.0, 61.0, 1.0)])
        got = float(bio.alpha_dB_per_m(100.0, [25.0])[0]) * 1000.0
        assert got == pytest.approx(1.0 * 61.0 ** 2, rel=1e-12)

    @pytest.mark.parametrize('kwargs, match', [
        (dict(f0_hz=0.0), 'f0_hz'), (dict(Q=0.0), 'Q'), (dict(a0=0.0), 'a0')])
    def test_the_existing_refusals_run_before_the_ceiling_check(self, kwargs,
                                                                match):
        """f0 = 0 divides in the ceiling formula, so the ordering matters."""
        with pytest.raises(ConfigurationError, match=match):
            _biological_layer(**kwargs)

    def test_the_class_docstring_names_the_pair_that_straddles_the_ceiling(self):
        """The docstring quotes a worked example either side of the 3638 dB/km
        bound. It read ``Q = 61 clears it`` while 61 gives 3721 and warns —
        the two tests above already had it the other way round. Pinned against
        the computed ceiling so the prose cannot drift off the arithmetic
        again."""
        ceiling = self._ceiling_dB_km(100.0)
        assert 1.0 * 60.0 ** 2 < ceiling < 1.0 * 61.0 ** 2
        doc = ' '.join(BiologicalLayer.__doc__.split())
        assert '``a0 = 1, Q = 60`` gives 3600 and clears it' in doc
        assert '``Q = 61`` gives 3721 and warns' in doc

    def test_a_layer_built_from_a_tuple_warns_once(self):
        """The nested path raises the same single warning the direct one
        does. Where that warning *lands* is pinned in
        ``test_warning_attribution.py``; this pins that the redesign which
        moved it there did not turn it into two, or none."""
        with pytest.warns(UserWarning, match='CRCI') as rec:
            Biological(layers=[(0.0, 50.0, 100.0, 61.0, 1.0)])
        assert len(rec) == 1, [str(w.message) for w in rec]


class TestBiologicalLayerRefusesNonFiniteAndNegativeDepths:
    """Every one of the five fields is held to finiteness, and the two depths
    additionally to ``>= 0``, before the ceiling arithmetic runs.

    The sign tests further down are bare ``<=`` comparisons, which NaN answers
    False to and which ``inf`` passes for ``a0`` and ``Q``. Unguarded, all six
    of the inputs below constructed: the NaN ones silently, and the two
    infinities with a ceiling warning that reported its own limit as
    ``nan dB/km``. A layer that gets through reaches the Acoustics-Toolbox
    deck via ``io.at_codes.biological_records``, where ``AttenMod.f90``'s band test
    ``z >= Z1 .AND. z <= Z2`` (:104) is False at every depth for a NaN or a
    negative bound — so the layer is written to the file and then contributes
    nothing, which is the failure mode the typed refusal replaces."""

    @pytest.mark.parametrize('kwargs, match', [
        (dict(z_top_m=float('nan')), 'z_top_m must be finite'),
        (dict(z_bottom_m=float('nan')), 'z_bottom_m must be finite'),
        (dict(z_top_m=-500.0, z_bottom_m=-100.0),
         'z_top_m must be non-negative'),
        (dict(a0=float('nan')), 'a0 must be finite'),
        (dict(Q=float('nan')), 'Q must be finite'),
        (dict(a0=float('inf')), 'a0 must be finite'),
        (dict(Q=float('inf')), 'Q must be finite'),
        (dict(f0_hz=float('nan')), r'BiologicalLayer\.f0_hz must be finite'),
        (dict(f0_hz=float('inf')), r'BiologicalLayer\.f0_hz must be finite'),
    ], ids=['z_top_nan', 'z_bottom_nan', 'negative_depths', 'a0_nan', 'Q_nan',
            'a0_inf', 'Q_inf', 'f0_nan', 'f0_inf'])
    def test_a_non_finite_or_negative_input_is_refused_by_name(self, kwargs,
                                                               match):
        with warnings.catch_warnings():
            # An escaped warning would mean the ceiling arithmetic ran, which
            # is exactly what these guards are placed in front of.
            warnings.simplefilter('error')
            with pytest.raises(ConfigurationError, match=match):
                _biological_layer(**kwargs)

    def test_a_non_finite_f0_is_named_by_the_layer_not_the_unit_converter(self):
        """``f0_hz`` was the one field that already failed, but from inside
        ``convert_attenuation_units`` in the ceiling formula — the message
        named that function and its frequency argument, not the layer field
        the user wrote."""
        with pytest.raises(ConfigurationError,
                           match='f0_hz must be finite') as excinfo:
            _biological_layer(f0_hz=float('nan'))
        assert 'convert_attenuation_units' not in str(excinfo.value)

    def test_a_finite_zero_reaches_the_sign_check_not_the_finiteness_guard(self):
        """The finiteness guards run first but do not take over the sign
        verdicts: zero is finite, so ``f0_hz = 0`` still fails as a sign
        error, keeping the ``"must be positive"`` phrase those refusals own."""
        with pytest.raises(ConfigurationError, match='f0_hz must be positive'):
            _biological_layer(f0_hz=0.0)

    def test_a_layer_at_zero_depth_is_accepted(self):
        """The depth bound is non-negative, not positive: a layer whose top
        sits at the sea surface is an ordinary configuration."""
        assert _biological_layer(z_top_m=0.0, z_bottom_m=50.0).z_top_m == 0.0


@pytest.mark.parametrize('cls', [BiologicalLayer, Biological, FrancoisGarrison],
                         ids=['BiologicalLayer', 'Biological', 'FrancoisGarrison'])
def test_the_written_out_init_takes_exactly_the_dataclass_fields(cls):
    """All three classes hand-write the ``__init__`` the decorator would
    generate, so that no ``<string>`` frame sits between a construction-time
    warning (the layer ceiling, the fitted envelope) and the user (see the
    class docstrings). The cost is that the field list now
    exists twice, and the drift is silent in one direction: an annotation
    added below still shapes ``repr`` / ``__eq__`` / ``fields()`` while the
    constructor has no way to set it, so instances carry the attribute only
    when something else assigns it. Pinned in the order the 5-tuple
    ``io.at_codes.biological_records`` writes, which is the same order."""
    parameters = list(inspect.signature(cls.__init__).parameters)[1:]
    assert parameters == [f.name for f in dataclasses.fields(cls)]


class TestTheTwoAbsorptionRoutesDivergeByTheDocumentedAmount:
    """One absorption model, two documented routes, different answers.

    A constant absorption is converted at 1500 m/s by the Python accessor
    and at each SSP row's own ``c`` by the Acoustics-Toolbox deck
    (``AttenMod.f90:73``). Neither side is wrong and neither is changed;
    ``Modes.with_attenuation`` says so where it sends the user from one to
    the other. These are the numbers its documentation quotes. (A
    Francois-Garrison law has no such gap: the deck rows carry the accessor's
    own α at each node, see ``test_io_oalib.py``.)"""

    @pytest.mark.parametrize('sound_speed, expected', [
        (1450.0, -3.33), (1500.0, 0.0), (1550.0, 3.33)])
    def test_a_constant_absorption_diverges_with_the_ssp_sound_speed(
            self, sound_speed, expected):
        from uacpy.core.absorption import ConstantAbsorption
        python = float(np.ravel(
            ConstantAbsorption(0.5).alpha_dB_per_m(1e3, [0.0]))[0])
        deck = float(convert_attenuation_units(
            0.5, 1e3, 'dB/wavelength', 'dB/m', sound_speed=sound_speed))
        assert 100.0 * (python - deck) / deck == pytest.approx(expected,
                                                               abs=0.01)

    def test_with_attenuation_warns_the_reader_that_the_routes_differ(self):
        from uacpy.core.results.modes import Modes
        doc = ' '.join(Modes.with_attenuation.__doc__.split())
        assert 'need not be the same number the solver used' in doc
        assert 'AttenMod.f90:73' in doc
        assert '±3.3 %' in doc

    def test_the_absorption_classes_state_where_the_routes_meet(self):
        from uacpy.core.absorption import ConstantAbsorption
        fg_doc = ' '.join(FrancoisGarrison.__doc__.split())
        assert 'Every engine sees the same α(z)' in fg_doc
        assert 'z_bar' not in fg_doc
        ca_doc = ' '.join(ConstantAbsorption.__doc__.split())
        assert 'misc/AttenMod.f90:73' in ca_doc
        assert '±3.3 %' in ca_doc


class TestPhToNbs:
    """``ph_to_nbs`` moves a measured pH onto the NBS scale Francois–Garrison
    was fitted on (Brewer & Hester 2009: "the sound absorption equations are
    based on the old NBS scale"; Uzhansky et al. 2025 read it the same way).

    The conversion is the Takahashi et al. (1982, GEOSECS) activity
    coefficient ``fH(T, S)`` that CO2SYS uses: ``pH_NBS = pH_SWS −
    log10(fH)``, exact for the seawater scale; the total scale differs from
    the seawater scale by the fluoride term, ≈ 0.01, which is neglected.
    Expected offsets are that formula: +0.100 at 4 °C / 35, +0.147 at 25 °C
    / 35 (S = 35 — the fit's own quadratic in S makes it +0.133 at 10 °C /
    20).
    """

    def test_seawater_scale_shifts_up_by_minus_log10_fH(self):
        assert ph_to_nbs(8.0, 'seawater', temperature=4.0,
                         salinity=35.0) == pytest.approx(8.1001, abs=1e-3)
        assert ph_to_nbs(8.0, 'seawater', temperature=25.0,
                         salinity=35.0) == pytest.approx(8.1467, abs=1e-3)

    def test_total_scale_is_treated_as_the_seawater_scale(self):
        assert ph_to_nbs(7.9, 'total', temperature=10.0, salinity=20.0) \
            == pytest.approx(8.0334, abs=1e-3)

    def test_nbs_passes_through_unchanged(self):
        assert ph_to_nbs(8.0, 'nbs', temperature=25.0, salinity=35.0) \
            == 8.0

    def test_an_unknown_scale_is_refused_naming_the_choices(self):
        with pytest.raises(ConfigurationError, match="'nbs'.*'total'.*'seawater'"):
            ph_to_nbs(8.0, 'free', temperature=4.0, salinity=35.0)

    def test_the_water_is_named_as_the_formulas_name_it(self):
        with pytest.raises(TypeError, match='temperature_c'):
            ph_to_nbs(8.0, 'total', temperature_c=4.0, salinity=35.0)

    def test_broadcasts_over_arrays(self):
        out = ph_to_nbs(np.array([7.8, 8.0]), 'total',
                        temperature=np.array([4.0, 25.0]), salinity=35.0)
        assert out == pytest.approx([7.9001, 8.1467], abs=1e-3)


class TestFrancoisGarrisonPhScale:
    """``FrancoisGarrison`` takes the scale its ``pH`` is on and converts to
    NBS once, for the formula every engine evaluates (the AT deck rows carry
    its α, so they are handed the same pH)."""

    def _pair(self, scale):
        return FrancoisGarrison(temperature=4.0, salinity=35.0, pH=8.0,
                                ph_scale=scale)

    def test_the_default_scale_is_nbs_and_leaves_every_number_alone(self):
        fg = FrancoisGarrison(temperature=4.0, salinity=35.0, pH=8.0)
        assert fg.ph_scale == 'nbs'
        assert fg.ph_nbs == 8.0

    def test_total_scale_converts_before_the_boric_term(self):
        total = self._pair('total')
        converted = ph_to_nbs(8.0, 'total', temperature=4.0, salinity=35.0)
        nbs = FrancoisGarrison(temperature=4.0, salinity=35.0, pH=converted)
        assert total.ph_nbs == pytest.approx(converted)
        for f in (100.0, 500.0, 1000.0, 10000.0):
            assert total.alpha_dB_per_m(f, [0.0, 1000.0]) == pytest.approx(
                nbs.alpha_dB_per_m(f, [0.0, 1000.0]))

    def test_total_scale_raises_low_frequency_absorption_by_about_a_fifth(self):
        """+0.10 on the boric term's ``10**(0.78·pH)`` is ×1.20; at 300 Hz
        that term is nearly all of the absorption, at 10 kHz almost none."""
        total, nbs = self._pair('total'), self._pair('nbs')
        low = float(total.alpha_dB_per_m(300.0, [1000.0])[0]
                    / nbs.alpha_dB_per_m(300.0, [1000.0])[0])
        high = float(total.alpha_dB_per_m(20000.0, [1000.0])[0]
                     / nbs.alpha_dB_per_m(20000.0, [1000.0])[0])
        assert 1.15 < low < 1.22
        assert 1.0 < high < 1.02

    def test_an_unknown_scale_is_refused_at_construction(self):
        with pytest.raises(ConfigurationError, match='ph_scale'):
            self._pair('free')

    def test_the_builder_forwards_the_scale(self):
        fg = FrancoisGarrison.from_temperature_salinity([0.0, 100.0], [10.0, 8.0], [35.0, 35.0],
                                     pH=7.9, ph_scale='total')
        assert fg.ph_scale == 'total'
        assert fg.pH == 7.9
        assert np.all(fg.ph_nbs > 7.9)
        assert FrancoisGarrison.from_temperature_salinity([0.0, 100.0], [10.0, 8.0],
                                       [35.0, 35.0]).ph_scale == 'nbs'


class TestAlphaIsEvaluatedOnBothAxesFromOneEvaluator:
    """alpha(f, z) through one door, in whichever units are asked for.

    Before this, the package exposed the same formula twice and the two were
    transposes of each other: ``absorption_francois_garrison`` vectorised over
    frequency with a scalar depth, ``FrancoisGarrison.alpha_dB_per_m``
    vectorised over depth with a scalar frequency, and an array frequency into
    the second raised ``TypeError: only 0-dimensional arrays can be converted
    to Python scalars``. Neither could answer alpha(f, z).
    """

    FG = dict(temperature=10.0, salinity=35.0, pH=8.0)
    F = np.array([1e3, 1e4, 1e5])

    def test_an_array_of_frequencies_is_evaluated_elementwise(self):
        """One value per frequency, from one call."""
        a = Thorp().table(self.F)
        assert a.data.shape == (3,)

    def test_both_axes_together_give_a_depth_by_frequency_grid(self):
        """Depth-first, the shape convention the package uses everywhere."""
        z = np.array([0.0, 50.0, 100.0, 200.0])
        a = FrancoisGarrison(**self.FG).table(self.F, depths=z)
        assert a.data.shape == (z.size, self.F.size)
        assert a.n_depths == 4 and a.n_frequencies == 3
        assert a.is_depth_dependent

    def test_the_grid_agrees_with_both_old_doors_elementwise(self):
        """The row-and-column check that would have caught the transpose:
        every cell must equal what the frequency-vectorised formula and the
        depth-vectorised method each return for that cell."""
        from uacpy.core.acoustics.attenuation import (
            absorption_francois_garrison,
        )
        z = np.array([0.0, 100.0, 250.0])
        grid = FrancoisGarrison(**self.FG).table(self.F, depths=z).data
        for j, f in enumerate(self.F):                     # column <- method
            col = FrancoisGarrison(**self.FG).alpha_dB_per_m(f, z) * 1000.0
            np.testing.assert_allclose(grid[:, j], col, rtol=1e-12)
        for i, zz in enumerate(z):                         # row <- formula
            row = absorption_francois_garrison(
                self.F, temperature=self.FG['temperature'],
                salinity=self.FG['salinity'], pH=self.FG['pH'], depth=zz)
            np.testing.assert_allclose(grid[i, :], row, rtol=1e-12)

    def test_without_depths_one_water_row_is_drawn_at_the_surface(self):
        """The law has no depth of its own: with no depth argument a water
        row is evaluated at 0 m, and says so on the curve."""
        flat = FrancoisGarrison(**self.FG).table(self.F)
        at_surface = FrancoisGarrison(**self.FG).table(self.F, depths=[0.0])
        at_depth = FrancoisGarrison(**self.FG).table(self.F, depths=[1000.0])
        assert not flat.is_depth_dependent
        assert flat.depth_m == 0.0
        np.testing.assert_array_equal(flat.data, at_surface.data[0])
        assert np.all(flat.data != at_depth.data[0])

    def test_without_depths_a_profile_is_drawn_on_its_own_depths(self):
        law = two_layer_absorption()
        table = law.table(self.F)
        assert table.is_depth_dependent
        np.testing.assert_array_equal(table.depths, TWO_LAYER_DEPTHS)
        np.testing.assert_array_equal(
            table.data, law.table(self.F, depths=TWO_LAYER_DEPTHS).data)

    def test_the_biological_table_takes_depths_units_and_sound_speed(self):
        """One depth inside the layer, one below it, in a unit that reads the
        sound speed: the table is the law's dB/m converted at that speed."""
        layers = [(10.0, 50.0, 2e3, 5.0, 0.1)]
        z = [30.0, 80.0]
        law = Biological(layers=layers)
        got = law.table(self.F, depths=z, units='dB/wavelength',
                        sound_speed=1480.0)
        want = np.stack([law.alpha_dB_per_m(f, z) * 1480.0 / f
                         for f in self.F], axis=1)
        np.testing.assert_allclose(got.data, want, rtol=1e-12)
        assert got.model == 'biological'
        assert got.units == 'dB/wavelength'
        assert np.all(got.data[0] > 0.0) and np.all(got.data[1] == 0.0)

    def test_the_constant_table_converts_at_the_sound_speed(self):
        """The per-wavelength value, the units and the sound speed that
        converts it reach ``ConstantAbsorption.table``."""
        from uacpy.core.constants import DEFAULT_SOUND_SPEED
        law = ConstantAbsorption(value_dB_per_wavelength=0.25)
        got = law.table(self.F, units='dB/m', sound_speed=1480.0)
        np.testing.assert_allclose(got.data, 0.25 * self.F / 1480.0,
                                   rtol=1e-12)
        assert got.model == 'constant'
        # 0.25 dB per wavelength is 0.25 * f / c dB/m, at the default c.
        flat = law.table(self.F)
        np.testing.assert_allclose(
            flat.data, 0.25 * self.F / DEFAULT_SOUND_SPEED * 1e3, rtol=1e-12)

    @pytest.mark.parametrize('sound_speed', [1480.0, 1500.0])
    def test_a_per_wavelength_value_round_trips_at_any_sound_speed(
            self, sound_speed):
        """The ``sound_speed`` that converts the output also converts the
        per-wavelength value on the way in, so asking for dB/wavelength
        back returns the value itself — at 1480 m/s as at the 1500 m/s
        default (REL-NEW-ABS: it read 0.24667 at 1480)."""
        got = ConstantAbsorption(value_dB_per_wavelength=0.25).table(
            [1e3], units='dB/wavelength', sound_speed=sound_speed)
        np.testing.assert_allclose(got.data, [0.25], rtol=1e-12)

    def test_the_carrier_records_which_formula_made_it(self):
        """Provenance, mirroring ``SoundSpeedProfile.formula``."""
        assert Thorp().table(self.F).model == 'thorp'
        assert FrancoisGarrison(**self.FG).table(
            self.F).model == 'francois_garrison'

    def test_units_are_carried_not_baked_into_a_name(self):
        a = Thorp().table(self.F)
        assert a.units == 'dB/km'
        np.testing.assert_allclose(a.to_units('dB/m').data,
                                   a.data / 1000.0, rtol=1e-12)
        assert a.to_units('dB/m').units == 'dB/m'

    def test_a_frequency_dependent_unit_needs_the_axis_the_carrier_keeps(self):
        """The structural reason for a carrier rather than a bare array:
        dB/wavelength, Q and L cannot be evaluated once the frequency axis is
        gone. Converting to one must use each frequency, not a single value."""
        from uacpy.core.acoustics.attenuation import convert_attenuation_units
        a = Thorp().table(self.F)
        got = a.to_units('dB/wavelength', sound_speed=1500.0).data
        want = [float(convert_attenuation_units(v, f, 'dB/km', 'dB/wavelength',
                                                sound_speed=1500.0))
                for v, f in zip(a.data, self.F)]
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_francois_garrison_without_an_ocean_is_the_reference_water(self):
        """``FrancoisGarrison()`` is the reference water (10 °C, 35 PSU,
        pH 8), the formula's own defaults, and its table records those
        values; with no depths it is drawn at the surface, the formula's own
        default depth."""
        from uacpy.core.acoustics.attenuation import (
            absorption_francois_garrison,
        )
        a = FrancoisGarrison().table(self.F)
        assert a.parameters == {'temperature': 10.0, 'salinity': 35.0,
                                'pH': 8.0, 'ph_scale': 'nbs'}
        assert a.depth_m == 0.0
        np.testing.assert_allclose(a.data, absorption_francois_garrison(self.F),
                                   rtol=1e-12)

    @pytest.mark.parametrize('ocean, depth', [
        (dict(temperature=10.0, salinity=35.0, pH=8.0), 0.0),
        (dict(temperature=2.0, salinity=34.7, pH=7.9), 4000.0),
        (dict(temperature=25.0, salinity=38.0, pH=8.2), 50.0),
        (dict(temperature=4.0, salinity=8.0, pH=7.7), 1000.0),
    ])
    def test_the_law_and_the_formula_agree_elementwise(self, ocean, depth):
        """Same names, same values, element by element, at the depth asked;
        the parameters are the law's fields, so they build it again."""
        from uacpy.core.acoustics.attenuation import (
            absorption_francois_garrison,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ValidityWarning)
            law = FrancoisGarrison(**ocean)
            a = law.table(self.F, depths=depth)
            assert FrancoisGarrison(**a.parameters) == law
        assert a.parameters == {**ocean, 'ph_scale': 'nbs'}
        assert a.depth_m == depth
        np.testing.assert_allclose(
            a.data, absorption_francois_garrison(self.F, **ocean, depth=depth),
            rtol=1e-12)

    def test_the_recorded_ocean_survives_unit_conversion_and_export(self):
        from uacpy.core.absorption import AbsorptionCoefficient
        a = FrancoisGarrison(**self.FG).table(self.F)
        assert a.to_units('dB/m').parameters == a.parameters
        assert AbsorptionCoefficient.from_dict(a.to_dict()).parameters == \
            a.parameters
        assert Thorp().table(self.F).parameters is None


def test_every_absorption_model_is_one_object():
    """One public object per absorption model, the law: every law evaluates
    itself with ``.table``, and no function returns a law's table. The
    ``absorption_<model>`` names of ``uacpy.acoustics`` are the formulas on
    plain arrays (dB/km), not a second spelling of a law. The tabulated law
    is private: its spelling is the measured ``AbsorptionCoefficient`` an
    environment is given."""
    import inspect
    import uacpy
    import uacpy.core.absorption as A
    concrete = {c for c in vars(A).values()
                if inspect.isclass(c) and issubclass(c, A.Absorption)
                and c is not A.Absorption and not c.__name__.startswith('_')}
    assert {c.__name__ for c in concrete} == {
        'Thorp', 'FrancoisGarrison', 'Biological', 'ConstantAbsorption'}
    for c in concrete:
        law = c(**{f.name: _SAMPLE[f.name] for f in dataclasses.fields(c)
                   if f.name in _SAMPLE})
        assert law.table([1e3]).model == law._model_name()
    assert not [n for n in dir(uacpy) if n.startswith('absorption_')]
    formulas = {n for n in dir(uacpy.acoustics) if n.startswith('absorption_')}
    assert formulas == {'absorption_thorp', 'absorption_francois_garrison',
                        'absorption_biological'}
    for n in formulas:
        out = getattr(uacpy.acoustics, n)(
            np.array([1e3, 1e4]), **({'f0_hz': 1e3, 'Q': 4.0, 'a0': 0.1}
                                     if n == 'absorption_biological' else {}))
        assert isinstance(out, np.ndarray) and out.shape == (2,), n


#: Minimal constructor arguments for the models the one-object test builds.
_SAMPLE = {'layers': [(20.0, 80.0, 1500.0, 4.0, 0.02)],
           'value_dB_per_wavelength': 1e-4}


def test_the_two_formula_families_are_reached_the_same_way():
    """Sound speed and absorption are the same shape — a family of named
    equations on plain arrays — so they must import the same way: both under
    ``uacpy.acoustics`` and its ``__all__``, neither at the top level, where
    the absorption *laws* (``Thorp``, ...) live.

    `density` is deliberately NOT promoted: a bare `uacpy.density` would be
    seawater density sitting next to `BoundaryProperties(density=)`, which is
    the seabed's.
    """
    import uacpy
    import uacpy.acoustics
    from uacpy.core.acoustics.seawater import SOUND_SPEED_FORMULAS

    speeds = {f'sound_speed_{name}' for name in SOUND_SPEED_FORMULAS}
    absorptions = {'absorption_thorp', 'absorption_francois_garrison',
                   'absorption_biological'}
    for family, names in (('sound speed', speeds), ('absorption', absorptions)):
        missing = {n for n in names if not hasattr(uacpy.acoustics, n)}
        assert not missing, (
            f'{family}: {sorted(missing)} is not reachable as '
            f'uacpy.acoustics.<name>, while the other family is')
        assert names <= set(uacpy.acoustics.__all__), (
            f'{family}: reachable but absent from uacpy.acoustics.__all__')
        assert not names & set(uacpy.__all__), (
            f'{family}: a second public path at the top level')

    assert not hasattr(uacpy, 'density'), (
        'uacpy.density would collide in meaning with '
        'BoundaryProperties(density=), which is the seabed"s')


def test_the_module_docstring_points_only_at_public_names():
    """``help(uacpy.core.absorption)`` names only public objects; the
    private dB/km kernels the laws are written on are not a second way to
    ask. ``convert_attenuation_units`` has one public path, the
    ``uacpy.acoustics`` facade, as the same object."""
    import uacpy
    import uacpy.core.absorption as A
    named = re.findall(r':(?:func|class|mod):`~?([\w.]+)`', A.__doc__)
    assert named, 'the module docstring names nothing'
    assert not [n for n in named if n.rpartition('.')[2].startswith('_')], \
        named
    assert uacpy.acoustics.convert_attenuation_units is A.convert_attenuation_units
    assert 'convert_attenuation_units' in uacpy.acoustics.__all__
    assert 'convert_attenuation_units' not in uacpy.__all__


def test_two_absorption_coefficients_compare_by_identity():
    """A dataclass ``__eq__`` over ndarray fields raises on a two-element
    comparison; the carrier compares by identity, like the other carriers
    with array fields."""
    a = Thorp().table([1000.0, 2000.0])
    b = Thorp().table([1000.0, 2000.0])
    assert (a == b) is False and (a == a) is True


class TestBiologicalLayerValidation:
    """:class:`BiologicalLayer` rejects impossible inputs at construction,
    matching the validation pattern on :class:`SedimentLayer`."""

    def test_valid_biological_layer(self):
        from uacpy.core.absorption import BiologicalLayer
        layer = BiologicalLayer(
            z_top_m=10.0, z_bottom_m=50.0, f0_hz=200.0, Q=20.0, a0=0.5,
        )
        assert layer.f0_hz == 200.0

    @pytest.mark.parametrize("kwargs,match", [
        (dict(z_top_m=50.0, z_bottom_m=10.0, f0_hz=200.0, Q=20.0, a0=0.5),
         "z_bottom_m"),
        (dict(z_top_m=10.0, z_bottom_m=10.0, f0_hz=200.0, Q=20.0, a0=0.5),
         "z_bottom_m"),
        (dict(z_top_m=10.0, z_bottom_m=50.0, f0_hz=-1.0, Q=20.0, a0=0.5),
         "f0_hz"),
        (dict(z_top_m=10.0, z_bottom_m=50.0, f0_hz=200.0, Q=0.0, a0=0.5),
         "Q"),
        (dict(z_top_m=10.0, z_bottom_m=50.0, f0_hz=200.0, Q=20.0, a0=-0.1),
         "a0"),
    ])
    def test_biological_layer_rejects_invalid(self, kwargs, match):
        from uacpy.core.absorption import BiologicalLayer
        with pytest.raises(ConfigurationError, match=match):
            BiologicalLayer(**kwargs)


class TestBiologicalBoundaryContributions:
    """Each layer is tested independently over its inclusive
    ``[z_top, z_bottom]`` span and the contributions summed, matching the
    AttenMod.f90:102-109 loop (``z >= Z1 .AND. z <= Z2`` per layer) — so a
    depth exactly on a boundary two stacked layers share receives both
    layers' contributions, and the outer edges of the stack stay
    inclusive."""

    @staticmethod
    def _stack():
        from uacpy.core.absorption import Biological
        return Biological(layers=[(0.0, 10.0, 100.0, 5.0, 10.0),
                                  (10.0, 20.0, 100.0, 5.0, 10.0)])

    def test_shared_boundary_sums_both_layers(self):
        a = self._stack().alpha_dB_per_m(100.0, [5.0, 10.0, 15.0])
        assert a[1] == pytest.approx(a[0] + a[2])
        # At f = f0 each layer peaks at a0·Q² = 10·25 = 250 dB/km, so the
        # shared depth carries 500 dB/km (the AttenMod.f90 sum).
        assert a[0] * 1000.0 == pytest.approx(250.0)
        assert a[1] * 1000.0 == pytest.approx(500.0)

    def test_outer_edges_are_inclusive(self):
        a = self._stack().alpha_dB_per_m(100.0, [0.0, 20.0, 25.0])
        assert a[0] == pytest.approx(a[1])
        assert a[0] > 0.0
        assert a[2] == 0.0


class TestAbsorptionFormulaOutputShapes:
    """The bare formulas and the unit converter shape their output after
    their input: 0-d for a scalar, unchanged for an array — a 1-element
    array stays 1-D and indexable."""

    def test_one_element_array_stays_indexable(self):
        from uacpy.core.acoustics.attenuation import (
            absorption_thorp, absorption_francois_garrison,
            convert_attenuation_units,
        )
        assert absorption_thorp(np.array([100.0])).shape == (1,)
        assert float(absorption_thorp(np.array([100.0]))[0]) > 0
        assert absorption_francois_garrison(np.array([100.0])).shape == (1,)
        assert convert_attenuation_units(
            np.array([1.0]), 100.0, 'dB/km', 'dB/m').shape == (1,)

    def test_scalar_input_yields_0d(self):
        from uacpy.core.acoustics.attenuation import (
            absorption_thorp, absorption_francois_garrison,
            convert_attenuation_units,
        )
        assert np.ndim(absorption_thorp(100.0)) == 0
        assert np.ndim(absorption_francois_garrison(100.0)) == 0
        assert np.ndim(
            convert_attenuation_units(1.0, 100.0, 'dB/km', 'dB/m')) == 0

    def test_n_element_array_keeps_shape(self):
        from uacpy.core.acoustics.attenuation import absorption_thorp
        assert absorption_thorp(np.array([100.0, 200.0, 300.0])).shape == (3,)


class TestConvertAttenuationUnitsFromQ:
    """Q sits in the denominator of the from-'Q' path, so a non-positive
    quality factor raises instead of dividing to inf."""

    @pytest.mark.parametrize("bad_q", [0.0, -5.0])
    def test_non_positive_q_raises(self, bad_q):
        from uacpy.core.acoustics.attenuation import convert_attenuation_units
        with pytest.raises(ConfigurationError, match="from_unit='Q'"):
            convert_attenuation_units(bad_q, 100.0, 'Q', 'dB/m')

    def test_positive_q_round_trips(self):
        from uacpy.core.acoustics.attenuation import convert_attenuation_units
        q = 50.0
        dB_m = convert_attenuation_units(q, 100.0, 'Q', 'dB/m')
        back = convert_attenuation_units(float(dB_m), 100.0, 'dB/m', 'Q')
        assert float(back) == pytest.approx(q)


class TestConvertAttenuationUnitsToQ:
    """The mirror direction is deliberately *not* symmetric. Q = 0 is not the
    limit of anything representable (it is α → ∞) and raises; α = 0 is the
    lossless limit and Q → ∞ is its exact value, so it is answered with
    ``inf`` — quietly, without numpy's bare divide-by-zero RuntimeWarning —
    and converts straight back to zero."""

    def test_zero_attenuation_gives_the_lossless_limit(self):
        from uacpy.core.acoustics.attenuation import convert_attenuation_units
        with warnings.catch_warnings():
            warnings.simplefilter('error')      # no bare RuntimeWarning
            q = convert_attenuation_units(0.0, 100.0, 'dB/m', 'Q',
                                          sound_speed=1500.0)
        assert np.isinf(float(q)) and float(q) > 0

    def test_the_lossless_limit_converts_back_to_zero(self):
        from uacpy.core.acoustics.attenuation import convert_attenuation_units
        q = convert_attenuation_units(0.0, 100.0, 'dB/m', 'Q',
                                      sound_speed=1500.0)
        back = convert_attenuation_units(float(q), 100.0, 'Q', 'dB/m',
                                         sound_speed=1500.0)
        assert float(back) == pytest.approx(0.0)

    def test_an_array_keeps_its_finite_entries(self):
        from uacpy.core.acoustics.attenuation import convert_attenuation_units
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            out = convert_attenuation_units(np.array([0.0, 0.5]), 100.0,
                                            'dB/m', 'Q', sound_speed=1500.0)
        assert np.isinf(out[0])
        assert out[1] == pytest.approx(3.63833694, rel=1e-6)


class TestAbsorptionFrequencyGuardIsShared:
    """``α(f, z)`` has no value at or below zero for any of the four models,
    so the guard sits on the public ``alpha_dB_per_m`` ahead of the dispatch
    rather than in each subclass. Thorp and Francois-Garrison are polynomials
    that had no guard and kept evaluating: both returned a *positive*
    attenuation at f = 0 and for a negative frequency."""

    def _models(self):
        from uacpy.core.absorption import (
            Thorp, FrancoisGarrison, Biological, ConstantAbsorption)
        return [
            Thorp(),
            FrancoisGarrison(temperature=15.0, salinity=35.0, pH=8.1),
            Biological(layers=[(0.0, 100.0, 100.0, 10.0, 1.0)]),
            ConstantAbsorption(value_dB_per_wavelength=0.5),
        ]

    @pytest.mark.parametrize('freq', [0.0, -100.0, float('nan')])
    def test_every_model_rejects_a_non_positive_frequency(self, freq):
        z = np.array([0.0, 50.0])
        for model in self._models():
            with pytest.raises(ConfigurationError,
                               match='frequency must be > 0'):
                model.alpha_dB_per_m(freq, z)

    def test_the_message_names_the_model(self):
        from uacpy.core.absorption import Thorp
        with pytest.raises(ConfigurationError, match='Thorp.alpha_dB_per_m'):
            Thorp().alpha_dB_per_m(0.0, np.array([0.0]))

    def test_a_positive_frequency_evaluates(self):
        z = np.array([0.0, 50.0])
        for model in self._models():
            a = np.asarray(model.alpha_dB_per_m(1000.0, z))
            assert a.shape == z.shape
            assert np.isfinite(a).all() and (a >= 0).all()

    def test_sub_hertz_is_legal(self):
        """Only f <= 0 has no wavelength; infrasonic frequencies convert."""
        from uacpy.core.absorption import ConstantAbsorption
        out = ConstantAbsorption(value_dB_per_wavelength=0.5).alpha_dB_per_m(
            0.5, np.array([10.0]))
        assert np.isfinite(out).all()


class TestConstantAbsorptionCeiling:
    """ConstantAbsorption enforces the same attenuation ceiling as the seabed
    carriers (above it every AT solver aborts in AttenMod.f90's CRCI)."""

    def test_above_ceiling_raises(self):
        from uacpy.core.absorption import ConstantAbsorption
        from uacpy.core.deck_limits import MAX_ATTENUATION_DB_PER_WAVELENGTH
        with pytest.raises(ConfigurationError, match="dB/wavelength exceeds"):
            ConstantAbsorption(
                value_dB_per_wavelength=MAX_ATTENUATION_DB_PER_WAVELENGTH + 1.0)

    def test_at_ceiling_constructs(self):
        from uacpy.core.absorption import ConstantAbsorption
        from uacpy.core.deck_limits import MAX_ATTENUATION_DB_PER_WAVELENGTH
        c = ConstantAbsorption(
            value_dB_per_wavelength=MAX_ATTENUATION_DB_PER_WAVELENGTH)
        assert c.value_dB_per_wavelength == pytest.approx(
            MAX_ATTENUATION_DB_PER_WAVELENGTH)


class TestConvertAttenuationUnitsNeedsAFrequency:
    """``dB/wavelength``, ``Q`` and ``L`` are all written against the
    frequency — λ = c/f, and ω = 2πf for the other two — so ``frequency=0``
    reached the arithmetic as a bare ``ZeroDivisionError`` on the wavelength
    paths and as a silent 0 or inf on the Q and L ones. The rest of the table
    is a pure scaling and converts at any frequency."""

    FREQUENCY_DEPENDENT = [('dB/wavelength', 'dB/m'), ('dB/m', 'dB/wavelength'),
                           ('Q', 'dB/m'), ('dB/m', 'Q'),
                           ('L', 'dB/m'), ('dB/m', 'L')]
    FREQUENCY_FREE = [('dB/m', 'dB/km'), ('dB/km', 'Nepers/m'),
                      ('Nepers/m', 'dB/m')]

    @pytest.mark.parametrize('from_unit, to_unit', FREQUENCY_DEPENDENT)
    @pytest.mark.parametrize('frequency', [0.0, -10.0, float('nan'),
                                           float('inf')])
    def test_a_frequency_bearing_path_refuses_it(self, from_unit, to_unit,
                                                 frequency):
        with pytest.raises(ConfigurationError, match='positive finite'):
            convert_attenuation_units(1.0, frequency, from_unit, to_unit)

    @pytest.mark.parametrize('from_unit, to_unit', FREQUENCY_FREE)
    def test_a_frequency_free_path_converts_at_zero(self, from_unit, to_unit):
        at_zero = convert_attenuation_units(1.0, 0.0, from_unit, to_unit)
        at_100 = convert_attenuation_units(1.0, 100.0, from_unit, to_unit)
        assert float(at_zero) == pytest.approx(float(at_100), rel=1e-12)

    @pytest.mark.parametrize('from_unit, to_unit',
                             FREQUENCY_DEPENDENT + FREQUENCY_FREE)
    def test_a_real_frequency_is_unaffected(self, from_unit, to_unit):
        got = convert_attenuation_units(1.0, 100.0, from_unit, to_unit)
        assert np.isfinite(float(got))

    def test_the_message_names_the_unit_that_needed_it(self):
        with pytest.raises(ConfigurationError, match='dB/wavelength'):
            convert_attenuation_units(1.0, 0.0, 'dB/wavelength', 'dB/km')


# ─────────────────────────────────────────────────────────────────────────────
# core/absorption.py — bare formulas
# ─────────────────────────────────────────────────────────────────────────────


def _thorp_dB_per_km_published(f_hz):
    """Thorp attenuation in dB/km, transcribed from the published polynomial.

    JKPS *Computational Ocean Acoustics* 2nd ed. Eq. (1.47) — four terms in
    ``f`` measured in **kHz**:

        alpha = 3.3e-3 + 0.11·f²/(1 + f²) + 44·f²/(4100 + f²) + 3e-4·f²

    AT's ``misc/AttenMod.f90`` carries the same four terms character for
    character (its comment numbers it Eq. 1.34, the 1st-edition numbering for
    the same expression). Written out here so a coefficient mis-transcribed
    into ``uacpy.core.absorption`` disagrees with something, rather than being
    frozen by the check values below — those were evaluated from these same
    coefficients and so cannot flag the transcription itself.
    """
    f = f_hz / 1000.0
    return (3.3e-3
            + 0.11 * f ** 2 / (1.0 + f ** 2)
            + 44.0 * f ** 2 / (4100.0 + f ** 2)
            + 3.0e-4 * f ** 2)


def _francois_garrison_dB_per_km_published(f_hz, T, S, pH, z):
    """Francois–Garrison attenuation in dB/km, transcribed from the published
    formulas: Francois & Garrison (1982), JASA 72(6) 1879–1890, in the form AT
    codes in ``misc/AttenMod.f90``. ``f`` in **kHz**, ``T`` in °C, ``S`` in
    psu, ``z`` in m. Two chemical relaxations of the form
    ``A·P·f_r·f²/(f_r² + f²)`` plus pure-water viscosity, which has no
    relaxation and so enters as a plain ``f²``:

        c  = 1412 + 3.21·T + 1.19·S + 0.0167·z
        A1 = 8.86/c · 10^(0.78·pH − 5)              P1 = 1
        f1 = 2.8·sqrt(S/35) · 10^(4 − 1245/(T + 273))
        A2 = 21.44·S/c · (1 + 0.025·T)              P2 = 1 − 1.37e-4·z + 6.2e-9·z²
        f2 = 8.17·10^(8 − 1990/(T + 273)) / (1 + 0.0018·(S − 35))
                                                    P3 = 1 − 3.83e-5·z + 4.9e-10·z²
        A3 = 4.937e-4 − 2.59e-5·T + 9.11e-7·T² − 1.5e-8·T³      (T < 20)
             3.964e-4 − 1.146e-5·T + 1.45e-7·T² − 6.5e-10·T³    (otherwise)

        alpha = A1·P1·f1·f²/(f1² + f²) + A2·P2·f2·f²/(f2² + f²) + A3·P3·f²

    The published statement gives the two A3 fits for "T < 20" and "T > 20"
    and says nothing about T = 20 exactly; ``AttenMod.f90`` writes
    ``if (T < 20)`` with the warm fit in its ``else``, so T = 20.0 takes the
    **warm** branch there. This transcription writes the branch the same way,
    which is the behaviour uacpy matches.

    Same purpose as :func:`_thorp_dB_per_km_published`: the 16-digit check
    values below were produced from these coefficients, so only an independent
    statement of the coefficients can catch a slip in them.
    """
    f = f_hz / 1000.0
    c = 1412.0 + 3.21 * T + 1.19 * S + 0.0167 * z

    A1 = 8.86 / c * 10.0 ** (0.78 * pH - 5.0)
    P1 = 1.0
    f1 = 2.8 * math.sqrt(S / 35.0) * 10.0 ** (4.0 - 1245.0 / (T + 273.0))

    A2 = 21.44 * S / c * (1.0 + 0.025 * T)
    P2 = 1.0 - 1.37e-4 * z + 6.2e-9 * z ** 2
    f2 = 8.17 * 10.0 ** (8.0 - 1990.0 / (T + 273.0)) / (1.0 + 0.0018 * (S - 35.0))

    P3 = 1.0 - 3.83e-5 * z + 4.9e-10 * z ** 2
    if T < 20.0:
        A3 = 4.937e-4 - 2.59e-5 * T + 9.11e-7 * T ** 2 - 1.5e-8 * T ** 3
    else:
        A3 = 3.964e-4 - 1.146e-5 * T + 1.45e-7 * T ** 2 - 6.5e-10 * T ** 3

    return (A1 * P1 * f1 * f ** 2 / (f1 ** 2 + f ** 2)
            + A2 * P2 * f2 * f ** 2 / (f2 ** 2 + f ** 2)
            + A3 * P3 * f ** 2)


class TestThorpReferenceValues:
    """Thorp's Eq. (1.47) is a fixed four-term polynomial in f²; these check
    values were evaluated from the published coefficients and agree with the
    textbook curve (~0.07 dB/km at 1 kHz, ~1.2 dB/km at 10 kHz)."""

    @pytest.mark.parametrize("f_hz, a_dB_km", [
        (100.0, 0.004499425722313501),
        (1e3, 0.06932909046574005),
        (1e4, 1.1898299387081566),
        (5e4, 17.52992268425963),
    ])
    def test_matches_published_curve(self, f_hz, a_dB_km):
        from uacpy.core.acoustics.attenuation import absorption_thorp
        assert float(absorption_thorp(f_hz)) == pytest.approx(
            a_dB_km, rel=1e-12)

    @pytest.mark.parametrize("f_hz", [
        10.0, 100.0, 1e3, 1e4, 5e4, 1e5, 3e5,
    ])
    def test_matches_the_published_polynomial(self, f_hz):
        """The implementation against :func:`_thorp_dB_per_km_published`, which
        states the coefficients independently of it. Evaluated on the four
        pinned frequencies plus three more, so a coefficient change that
        happened to leave the pinned values alone still has nowhere to hide.
        The 1e-12 is for float association, not for the algebra: the two
        expressions are the same polynomial and agree to a few ulp."""
        from uacpy.core.acoustics.attenuation import absorption_thorp
        assert float(absorption_thorp(f_hz)) == pytest.approx(
            _thorp_dB_per_km_published(f_hz), rel=1e-12)

    def test_class_converts_dB_per_km_to_dB_per_m(self):
        """Thorp.alpha_dB_per_m is the bare formula divided by 1000, flat in
        depth."""
        from uacpy.core.absorption import Thorp
        from uacpy.core.acoustics.attenuation import absorption_thorp
        z = np.array([0.0, 500.0, 5000.0])
        a = Thorp().alpha_dB_per_m(1e4, z)
        assert a.shape == z.shape
        np.testing.assert_allclose(
            a, float(absorption_thorp(1e4)) / 1000.0, rtol=1e-12)


class TestFrancoisGarrisonReferenceValues:
    """Francois & Garrison (1982) check values evaluated from the published
    coefficients (the AT AttenMod.f90 transcription); the set spans the
    boric-acid, MgSO4 and viscosity regimes, a deep cold-water case for the
    pressure corrections, and both sides of the 20 °C viscosity fit."""

    @pytest.mark.parametrize("f_hz, T, S, pH, z, a_dB_km", [
        (1e3, 10.0, 35.0, 8.0, 0.0, 0.060126063391378846),
        (1e4, 10.0, 35.0, 8.0, 0.0, 0.962637291817115),
        (1e5, 10.0, 35.0, 8.0, 0.0, 33.63031641787127),
        (1e4, 4.0, 34.0, 7.9, 2000.0, 0.8372956055122321),
        # 20 °C sits on the A3 piecewise break: the warm fit applies there.
        (5e5, 20.0, 35.0, 8.0, 0.0, 146.64062876899342),
        (5e5, 10.0, 35.0, 8.0, 0.0, 124.67276253563452),
        (5e5, 25.0, 35.0, 8.0, 0.0, 169.7876059853789),
    ])
    def test_matches_published_curve(self, f_hz, T, S, pH, z, a_dB_km):
        from uacpy.core.acoustics.attenuation import (
            absorption_francois_garrison,
        )
        got = float(absorption_francois_garrison(f_hz, T, S, pH, z))
        assert got == pytest.approx(a_dB_km, rel=1e-12)

    @pytest.mark.parametrize("f_hz, T, S, pH, z", [
        # Every point the check values above pin, so the two sets are tied
        # together rather than each standing alone.
        (1e3, 10.0, 35.0, 8.0, 0.0),
        (1e4, 10.0, 35.0, 8.0, 0.0),
        (1e5, 10.0, 35.0, 8.0, 0.0),
        (1e4, 4.0, 34.0, 7.9, 2000.0),
        (5e5, 20.0, 35.0, 8.0, 0.0),
        (5e5, 10.0, 35.0, 8.0, 0.0),
        (5e5, 25.0, 35.0, 8.0, 0.0),
        # The 63-kHz depth pair of test_depth_correction_attenuates_mgso4_term.
        (6.3e4, 10.0, 35.0, 8.0, 0.0),
        (6.3e4, 10.0, 35.0, 8.0, 4000.0),
        # Off the pinned grid: either side of the A3 break to a tenth of a
        # degree, the pH and salinity terms away from their nominal values, and
        # a deep cold case that leans on P2/P3.
        (2e5, 19.9, 35.0, 8.0, 0.0),
        (2e5, 20.1, 35.0, 8.0, 0.0),
        (3e3, 12.0, 35.0, 7.4, 0.0),
        (3e4, 12.0, 8.0, 8.2, 0.0),
        (1e5, 2.0, 34.7, 8.1, 5000.0),
    ])
    def test_matches_the_published_formula(self, f_hz, T, S, pH, z):
        """The implementation against
        :func:`_francois_garrison_dB_per_km_published`, which states every
        coefficient independently of it. Same role as the Thorp transcription
        test, and the same reason for 1e-12: the two expressions differ only in
        how the products associate."""
        from uacpy.core.acoustics.attenuation import (
            absorption_francois_garrison,
        )
        got = float(absorption_francois_garrison(f_hz, T, S, pH, z))
        assert got == pytest.approx(
            _francois_garrison_dB_per_km_published(f_hz, T, S, pH, z),
            rel=1e-12)

    def test_the_a3_branch_at_exactly_20_degrees_is_the_warm_fit(self):
        """T = 20.0 exactly. The publication states the two A3 fits for
        "T < 20" and "T > 20" and says nothing about 20; ``AttenMod.f90``
        writes ``if (T < 20)`` with the warm fit in its ``else``, so AT takes
        the warm branch there and uacpy matches AT. Both the implementation and
        the transcription are held to that here, so the choice cannot drift on
        one side only.

        The two fits are built to nearly meet at the break — 2.2000e-4 against
        2.2010e-4, 4.5e-4 apart — which moves α at 500 kHz by only 1.7e-4
        relative. That is 8 orders above the 1e-12 the tests compare at and
        wholly invisible to a percent-level check, so the branch is worth
        pinning and only worth pinning tightly.
        """
        from uacpy.core.acoustics.attenuation import (
            absorption_francois_garrison,
        )
        f_khz, T, z = 500.0, 20.0, 0.0
        A3_warm = 3.964e-4 - 1.146e-5 * T + 1.45e-7 * T ** 2 - 6.5e-10 * T ** 3
        A3_cold = 4.937e-4 - 2.59e-5 * T + 9.11e-7 * T ** 2 - 1.5e-8 * T ** 3
        P3 = 1.0 - 3.83e-5 * z + 4.9e-10 * z ** 2

        warm = _francois_garrison_dB_per_km_published(f_khz * 1e3, T, 35.0, 8.0, z)
        # A3 is the only thing the branch changes, so swapping it is the whole
        # of the difference between the two readings at this T.
        cold = warm + (A3_cold - A3_warm) * P3 * f_khz ** 2
        assert abs(cold - warm) / warm > 1e-5

        assert float(absorption_francois_garrison(f_khz * 1e3, T, 35.0, 8.0, z)) \
            == pytest.approx(warm, rel=1e-12)

    def test_depth_correction_attenuates_mgso4_term(self):
        """P2 = 1 − 1.37e-4·z + 6.2e-9·z² cuts the 63-kHz (MgSO4-dominated)
        absorption to ~0.549 of its surface value at 4000 m."""
        from uacpy.core.acoustics.attenuation import (
            absorption_francois_garrison,
        )
        a_surf = float(absorption_francois_garrison(6.3e4, 10., 35., 8., 0.))
        a_deep = float(absorption_francois_garrison(6.3e4, 10., 35., 8., 4000.))
        assert a_deep / a_surf == pytest.approx(0.54917617566523, rel=1e-10)

    def test_the_class_evaluates_the_formula_at_each_depth_asked(self):
        """FrancoisGarrison.alpha_dB_per_m evaluates the formula at each
        depth asked (dB/m = dB/km / 1000)."""
        from uacpy.core.absorption import FrancoisGarrison
        from uacpy.core.acoustics.attenuation import (
            absorption_francois_garrison,
        )
        fg = FrancoisGarrison(temperature=10.0, salinity=35.0, pH=8.0)
        z = np.array([0.0, 2000.0])
        got = fg.alpha_dB_per_m(1e4, z)
        want = absorption_francois_garrison(
            1e4, temperature=10.0, salinity=35.0, pH=8.0, depth=z) / 1000.0
        np.testing.assert_allclose(got, want, rtol=1e-12)
        assert float(got[0]) == pytest.approx(0.000962637291817115, rel=1e-12)


class TestConvertAttenuationUnitsClosedForms:
    """Every unit is defined against the nepers/m attenuation ``a`` of
    ``exp(-a·x)`` at ``omega = 2πf``: dB/m = a·20/ln10, dB/wavelength =
    dB/m·(c/f), Q gives a = omega/(2cQ), L gives a = L·omega/c. Each path is
    checked against that definition written out independently, plus the
    round-trip back."""

    F = 100.0
    C = 1480.0
    NEPER_DB = 20.0 / np.log(10.0)

    def _conv(self, alpha, frm, to):
        from uacpy.core.acoustics.attenuation import convert_attenuation_units
        return float(convert_attenuation_units(
            alpha, self.F, frm, to, sound_speed=self.C))

    def test_dB_km_is_dB_m_times_1000(self):
        assert self._conv(3.0, 'dB/km', 'dB/m') == pytest.approx(
            3.0e-3, rel=1e-12)
        assert self._conv(3.0e-3, 'dB/m', 'dB/km') == pytest.approx(
            3.0, rel=1e-12)

    def test_dB_per_wavelength_uses_lambda_c_over_f(self):
        lam = self.C / self.F
        assert self._conv(0.5, 'dB/wavelength', 'dB/m') == pytest.approx(
            0.5 / lam, rel=1e-12)
        assert self._conv(0.5 / lam, 'dB/m', 'dB/wavelength'
                          ) == pytest.approx(0.5, rel=1e-12)

    def test_nepers_use_20_over_ln10(self):
        assert self._conv(1.0, 'Nepers/m', 'dB/m') == pytest.approx(
            self.NEPER_DB, rel=1e-12)
        assert self._conv(self.NEPER_DB, 'dB/m', 'Nepers/m'
                          ) == pytest.approx(1.0, rel=1e-12)

    def test_quality_factor_definition(self):
        # a = omega/(2cQ) nepers/m, with omega = 2·pi·f, so a = pi·f/(c·Q).
        Q = 50.0
        a_nepers = np.pi * self.F / (self.C * Q)
        assert self._conv(Q, 'Q', 'dB/m') == pytest.approx(
            a_nepers * self.NEPER_DB, rel=1e-12)
        assert self._conv(a_nepers * self.NEPER_DB, 'dB/m', 'Q'
                          ) == pytest.approx(Q, rel=1e-12)

    def test_loss_tangent_definition(self):
        # a = L·omega/c nepers/m.
        L = 2e-3
        a_nepers = L * 2.0 * np.pi * self.F / self.C
        assert self._conv(L, 'L', 'dB/m') == pytest.approx(
            a_nepers * self.NEPER_DB, rel=1e-12)
        assert self._conv(a_nepers * self.NEPER_DB, 'dB/m', 'L'
                          ) == pytest.approx(L, rel=1e-12)

    def test_unknown_units_raise_in_both_positions(self):
        from uacpy.core.acoustics.attenuation import convert_attenuation_units
        with pytest.raises(ConfigurationError, match="unknown unit from_unit"):
            convert_attenuation_units(1.0, self.F, 'furlongs', 'dB/m')
        with pytest.raises(ConfigurationError, match="unknown unit to_unit"):
            convert_attenuation_units(1.0, self.F, 'dB/m', 'furlongs')

    def test_an_unknown_unit_is_told_the_valid_ones(self):
        from uacpy.core.acoustics.attenuation import convert_attenuation_units
        with pytest.raises(ConfigurationError,
                           match="unknown unit to_unit='db/m'") as exc:
            convert_attenuation_units(1.0, self.F, 'dB/km', 'db/m')
        text = str(exc.value)
        for unit in ('dB/km', 'dB/m', 'dB/wavelength', 'Nepers/m', "'Q'",
                     "'L'"):
            assert unit in text


def test_an_empty_frequency_axis_is_refused_by_name():
    from uacpy.core.absorption import Thorp
    with pytest.raises(ConfigurationError, match="frequencies is empty"):
        Thorp().table([])


class TestBiologicalLorentzian:
    """The layer resonance is ``a0 / ((1 − f0²/f²)² + 1/Q²)`` dB/km
    (AttenMod.f90): the peak at f = f0 is exactly a0·Q², and one octave
    below, (1 − f0²/f²) = −3 so the denominator is 9 + 1/Q²."""

    def _bio(self):
        from uacpy.core.absorption import Biological
        return Biological(layers=[(10.0, 60.0, 1000.0, 5.0, 10.0)])

    def test_peak_is_a0_q_squared(self):
        a = self._bio().alpha_dB_per_m(1000.0, np.array([30.0]))
        assert float(a[0]) == pytest.approx(10.0 * 25.0 / 1000.0, rel=1e-12)

    def test_off_resonance_denominator(self):
        a = self._bio().alpha_dB_per_m(500.0, np.array([30.0]))
        want = 10.0 / (9.0 + 1.0 / 25.0) / 1000.0
        assert float(a[0]) == pytest.approx(want, rel=1e-12)

    def test_zero_frequency_raises_typed_error(self):
        """f = 0 must raise ConfigurationError, not divide by zero."""
        with pytest.raises(ConfigurationError, match="frequency must be > 0"):
            self._bio().alpha_dB_per_m(0.0, np.array([30.0]))

    def test_zero_resonance_frequency_rejected(self):
        from uacpy.core.absorption import BiologicalLayer
        with pytest.raises(ConfigurationError, match="f0_hz must be positive"):
            BiologicalLayer(z_top_m=0.0, z_bottom_m=10.0, f0_hz=0.0,
                            Q=5.0, a0=1.0)


class TestConstantAbsorptionContract:
    """A zero baseline is the valid lossless limit; f = 0 has no wavelength
    to convert through and must raise the typed error."""

    def test_zero_value_is_a_valid_lossless_baseline(self):
        from uacpy.core.absorption import ConstantAbsorption
        c = ConstantAbsorption(value_dB_per_wavelength=0.0)
        np.testing.assert_allclose(
            c.alpha_dB_per_m(100.0, np.array([0.0, 50.0])), 0.0)

    def test_zero_frequency_raises_typed_error(self):
        from uacpy.core.absorption import ConstantAbsorption
        c = ConstantAbsorption(value_dB_per_wavelength=0.5)
        with pytest.raises(ConfigurationError, match="frequency must be > 0"):
            c.alpha_dB_per_m(0.0, np.array([10.0]))


# ─────────────────────────────────────────────────────────────────────────────
# core/absorption.py — untested guard boundaries (wave-5 survivors)
# ─────────────────────────────────────────────────────────────────────────────


class TestConvertFromQualityFactorBoundaries:
    """Q in (0, 1] is a legal (heavily damped) quality factor; only
    Q <= 0 has no attenuation to convert."""

    def test_sub_unity_q_round_trips(self):
        from uacpy.core.acoustics.attenuation import convert_attenuation_units
        dB_m = convert_attenuation_units(0.5, 100.0, 'Q', 'dB/m',
                                         sound_speed=1480.0)
        back = convert_attenuation_units(dB_m, 100.0, 'dB/m', 'Q',
                                         sound_speed=1480.0)
        assert float(back) == pytest.approx(0.5, rel=1e-12)

    def test_zero_q_raises(self):
        from uacpy.core.acoustics.attenuation import convert_attenuation_units
        with pytest.raises(ConfigurationError, match="positive quality"):
            convert_attenuation_units(0.0, 100.0, 'Q', 'dB/m',
                                      sound_speed=1480.0)


class TestBiologicalLayerValidatorBoundaries:
    """Zero-thickness layers and exactly-zero Q/a0 are rejected;
    parameters in (0, 1] are legal."""

    def test_zero_thickness_layer_raises(self):
        from uacpy.core.absorption import BiologicalLayer
        with pytest.raises(
                ConfigurationError,
                match='z_bottom_m must be strictly greater than z_top_m'):
            BiologicalLayer(z_top_m=10.0, z_bottom_m=10.0, f0_hz=100.0,
                            Q=5.0, a0=1.0)

    def test_zero_q_raises(self):
        from uacpy.core.absorption import BiologicalLayer
        with pytest.raises(ConfigurationError, match='Q must be positive'):
            BiologicalLayer(z_top_m=0.0, z_bottom_m=10.0, f0_hz=100.0,
                            Q=0.0, a0=1.0)

    def test_zero_a0_raises(self):
        from uacpy.core.absorption import BiologicalLayer
        with pytest.raises(ConfigurationError, match='a0 must be positive'):
            BiologicalLayer(z_top_m=0.0, z_bottom_m=10.0, f0_hz=100.0,
                            Q=5.0, a0=0.0)

    def test_sub_unity_parameters_are_legal(self):
        from uacpy.core.absorption import Biological
        b = Biological(layers=[(0.0, 10.0, 0.5, 0.5, 0.5)])
        a = b.alpha_dB_per_m(0.5, np.array([5.0]))
        assert np.isfinite(a).all()


class TestSubHertzFrequenciesAreLegal:
    """Only f <= 0 has no wavelength; infrasonic f in (0, 1] converts."""

    def test_constant_absorption_at_half_a_hertz(self):
        from uacpy.core.absorption import ConstantAbsorption
        c = ConstantAbsorption(value_dB_per_wavelength=0.5)
        assert np.isfinite(c.alpha_dB_per_m(0.5, np.array([10.0]))).all()


class TestArrivalAbsorptionExponent:
    """``omega * Im tau = -integral of alpha ds`` along a ray (Jensen et al.,
    *Computational Ocean Acoustics*, §3.6.2), so an arrival traced at
    ``f_t`` is absorbed at ``f`` by ``2 pi f_t Im tau * alpha(f)/alpha(f_t)``
    for a law that separates in frequency and depth, and linearly in ``f``
    for everything else."""

    IMAG = np.array([-1.0e-5, -3.0e-6, 0.0])
    FREQS = np.array([20000.0, 40000.0, 60000.0])

    def test_a_separable_law_scales_by_its_own_ratio(self):
        from uacpy.core.absorption import Thorp, arrival_absorption_exponent
        from uacpy.core.acoustics.attenuation import absorption_thorp
        got = arrival_absorption_exponent(
            self.IMAG, self.FREQS, trace_frequency=40000.0,
            absorption=Thorp())
        ratio = absorption_thorp(self.FREQS) / absorption_thorp(40000.0)
        want = np.outer(self.IMAG, 2.0 * np.pi * 40000.0 * ratio)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)
        # The Thorp exponent is not the linear one: 20 kHz is 36 % weaker.
        assert got[0, 0] / (2.0 * np.pi * 20000.0 * self.IMAG[0]) < 0.7

    @pytest.mark.parametrize('law', ['none', 'constant', 'biological',
                                     'no_trace_frequency'])
    def test_other_laws_scale_linearly_bit_for_bit(self, law):
        from uacpy.core.absorption import (
            Biological, ConstantAbsorption, Thorp, arrival_absorption_exponent)
        absorption, f_t = {
            'none': (None, 40000.0),
            'constant': (ConstantAbsorption(0.1), 40000.0),
            'biological': (Biological(layers=[(0.0, 50.0, 3e4, 4.0, 0.1)]),
                           40000.0),
            'no_trace_frequency': (Thorp(), None),
        }[law]
        got = arrival_absorption_exponent(
            self.IMAG, self.FREQS, trace_frequency=f_t, absorption=absorption)
        assert np.array_equal(got, np.outer(self.IMAG,
                                            2.0 * np.pi * self.FREQS))

    def test_one_francois_garrison_row_scales_by_its_surface_ratio(self):
        """The law has no depth of its own, so the ratio is the one at the
        surface (``table`` with no depths); its bend with depth is measured
        by ``band_absorption_error_dB_per_km(..., by_ratio=True)``."""
        from uacpy.core.absorption import arrival_absorption_exponent
        from uacpy.core.acoustics.attenuation import (
            absorption_francois_garrison,
        )
        fg = FrancoisGarrison(temperature=10.0, salinity=35.0, pH=8.0)
        got = arrival_absorption_exponent(
            self.IMAG, self.FREQS, trace_frequency=40000.0, absorption=fg)
        ratio = (absorption_francois_garrison(self.FREQS, 10.0, 35.0, 8.0, 0.0)
                 / absorption_francois_garrison(40000.0, 10.0, 35.0, 8.0, 0.0))
        np.testing.assert_allclose(
            got, np.outer(self.IMAG, 2.0 * np.pi * 40000.0 * ratio),
            rtol=1e-12, atol=0.0)

    def test_at_the_trace_frequency_the_law_is_not_consulted(self):
        """Francois-Garrison is fitted from 200 Hz: a 100 Hz trace read back
        at 100 Hz returns Im tau's own exponent, with no out-of-band
        notice."""
        from uacpy.core.absorption import arrival_absorption_exponent
        fg = FrancoisGarrison(temperature=10.0, salinity=35.0, pH=8.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            got = arrival_absorption_exponent(
                self.IMAG, [100.0], trace_frequency=100.0, absorption=fg)
        assert np.array_equal(got, np.outer(self.IMAG, [2.0 * np.pi * 100.0]))


class TestBandAbsorptionError:
    """The error of a law frozen at one frequency and scaled as
    ``(f/f_a)**power``, scanned over the whole band and the water column."""

    F = np.linspace(5e3, 15e3, 11)
    Z = np.linspace(0.0, 5000.0, 65)

    def test_by_ratio_is_zero_for_thorp(self):
        """Thorp has no depth term: its surface ratio is its ratio at every
        depth, so the Bellhop band check reads zero and stays silent."""
        from uacpy.core.absorption import band_absorption_error_dB_per_km
        assert band_absorption_error_dB_per_km(
            Thorp(), self.F, 1e4, depths=self.Z, by_ratio=True) < 1e-12

    def test_by_ratio_measures_the_pressure_bend_of_francois_garrison(self):
        """One Francois-Garrison row over a 5 km column, 5-15 kHz: the
        surface ratio misses the deep water by 0.058 dB/km, the linear line
        by 0.54 dB/km, and at the surface the ratio is exact."""
        from uacpy.core.absorption import band_absorption_error_dB_per_km
        fg = FrancoisGarrison()
        assert band_absorption_error_dB_per_km(
            fg, self.F, 1e4, depths=self.Z, by_ratio=True) == pytest.approx(
                0.05835, abs=1e-5)
        assert band_absorption_error_dB_per_km(
            fg, self.F, 1e4, depths=self.Z) == pytest.approx(0.54154,
                                                             abs=1e-5)
        assert band_absorption_error_dB_per_km(
            fg, self.F, 1e4, depths=[0.0], by_ratio=True) == 0.0

    def test_a_thorp_band_edge_error_is_the_gap_to_the_line(self):
        from uacpy.core.absorption import (
            Thorp, band_absorption_error_dB_per_km,
        )
        from uacpy.core.acoustics.attenuation import absorption_thorp
        f = np.array([20000.0, 60000.0])
        got = band_absorption_error_dB_per_km(Thorp(), f, 40000.0,
                                              depths=[0.0])
        line = absorption_thorp(40000.0) * f / 40000.0
        assert got == pytest.approx(np.max(np.abs(absorption_thorp(f) - line)),
                                    rel=1e-12)
        assert got == pytest.approx(2.349, abs=1e-3)   # P1: -2.349 dB at 1 km

    def test_the_square_law_is_measured_from_the_same_anchor(self):
        from uacpy.core.absorption import (
            Thorp, band_absorption_error_dB_per_km,
        )
        from uacpy.core.acoustics.attenuation import absorption_thorp
        f = np.array([20000.0, 60000.0])
        got = band_absorption_error_dB_per_km(Thorp(), f, 40000.0,
                                              depths=[0.0], power=2.0)
        square = absorption_thorp(40000.0) * (f / 40000.0) ** 2
        assert got == pytest.approx(
            np.max(np.abs(absorption_thorp(f) - square)), rel=1e-12)

    def test_a_resonance_inside_the_band_and_below_the_surface_is_seen(self):
        from uacpy.core.absorption import band_absorption_error_dB_per_km
        bio = Biological(layers=[(1000.0, 2000.0, 2500.0, 3.0, 0.1)])
        f = np.linspace(1000.0, 4000.0, 13)
        at_surface = band_absorption_error_dB_per_km(bio, f, 1000.0,
                                                     depths=[0.0])
        in_column = band_absorption_error_dB_per_km(
            bio, f, 1000.0, depths=np.linspace(0.0, 3000.0, 65))
        assert at_surface == 0.0
        # The 0.9 dB/km peak at 2500 Hz, against a line through 1000 Hz.
        assert in_column == pytest.approx(0.9 - 0.1 / ((1 - 6.25) ** 2 + 1 / 9)
                                          * 2.5, rel=1e-6)


class TestMinimaxAnchorFrequency:
    """The anchor whose frozen, linearly scaled law departs least, at its
    worst over the band, from the law itself."""

    BAND = np.linspace(1000.0, 4000.0, 2001)

    def _error(self, law, anchor):
        from uacpy.core.absorption import band_absorption_error_dB_per_km
        return band_absorption_error_dB_per_km(law, self.BAND, anchor,
                                               depths=[0.0])

    def test_thorp_over_1_to_4_khz_misses_by_less_than_from_the_centre(self):
        from uacpy.core.absorption import Thorp, minimax_anchor_frequency
        anchor = minimax_anchor_frequency(Thorp(), 1000.0, 4000.0,
                                          depths=[0.0])
        assert self._error(Thorp(), anchor) == pytest.approx(0.00629,
                                                             abs=2e-5)
        assert self._error(Thorp(), 2500.0) == pytest.approx(0.01551,
                                                             abs=2e-5)

    def test_no_frequency_in_the_band_is_a_better_anchor(self):
        from uacpy.core.absorption import Thorp, minimax_anchor_frequency
        anchor = minimax_anchor_frequency(Thorp(), 1000.0, 4000.0,
                                          depths=[0.0])
        best = self._error(Thorp(), anchor)
        others = [self._error(Thorp(), f)
                  for f in np.linspace(1000.0, 4000.0, 301)]
        assert min(others) >= best - 1e-6

    def test_a_law_linear_in_frequency_anchors_at_the_band_centre(self):
        from uacpy.core.absorption import (ConstantAbsorption,
                                           minimax_anchor_frequency)
        assert minimax_anchor_frequency(ConstantAbsorption(0.01), 1000.0,
                                        4000.0, depths=[0.0, 50.0]) == 2500.0

    def test_the_search_leaves_the_out_of_band_notice_to_the_caller(self):
        """The trial grids are the search's own; a Francois-Garrison band
        reaching below its 200 Hz fit is announced by the caller's own
        evaluations (the band warning, the deck's AC), not per trial."""
        from uacpy.core.absorption import minimax_anchor_frequency
        fg = FrancoisGarrison(temperature=10.0, salinity=35.0, pH=8.0)
        with recorded_warnings() as caught:
            anchor = minimax_anchor_frequency(fg, 100.0, 600.0,
                                              depths=[0.0, 50.0])
        assert 100.0 < anchor < 600.0
        assert not [w for w in caught if 'fitted over' in str(w.message)]

    def test_no_law_and_a_single_frequency_anchor_at_the_band_centre(self):
        from uacpy.core.absorption import Thorp, minimax_anchor_frequency
        assert minimax_anchor_frequency(None, 1000.0, 4000.0,
                                        depths=[0.0]) == 2500.0
        assert minimax_anchor_frequency(Thorp(), 1500.0, 1500.0,
                                        depths=[0.0]) == 1500.0


class TestTheModelsAreWrittenOnTheArrayFormulas:
    """Each absorption model evaluates the formula ``uacpy.acoustics`` holds
    on plain arrays, so the model and the array function are one computation:
    replacing the formula changes the model's answer."""

    def test_thorp_evaluates_absorption_thorp(self, monkeypatch):
        import uacpy.core.absorption as absorption
        monkeypatch.setattr(absorption, 'absorption_thorp',
                            lambda f, depth: np.full(np.shape(depth), 7.0))
        assert absorption.Thorp().alpha_dB_per_m(1000.0, [0.0, 50.0]) == \
            pytest.approx([7.0e-3, 7.0e-3])

    def test_francois_garrison_evaluates_its_array_formula(self, monkeypatch):
        import uacpy.core.absorption as absorption
        monkeypatch.setattr(
            absorption, 'absorption_francois_garrison',
            lambda **kw: np.full(np.shape(kw['depth']), 5.0))
        fg = absorption.FrancoisGarrison(temperature=10.0, salinity=35.0,
                                         pH=8.0)
        assert fg.alpha_dB_per_m(1000.0, [0.0, 50.0]) == \
            pytest.approx([5.0e-3, 5.0e-3])

    def test_biological_sums_biological_dB_per_km_over_its_layers(
            self, monkeypatch):
        import uacpy.core.absorption as absorption
        monkeypatch.setattr(absorption, 'absorption_biological',
                            lambda f, f0_hz, Q, a0: a0)
        bio = absorption.Biological(layers=[(0.0, 100.0, 1000.0, 4.0, 2.0),
                                            (50.0, 200.0, 800.0, 3.0, 3.0)])
        # 25 m is in the first layer only, 75 m in both, 150 m in the second.
        assert bio.alpha_dB_per_m(1000.0, [25.0, 75.0, 150.0]) == \
            pytest.approx([2.0e-3, 5.0e-3, 3.0e-3])


# ── absorption from real water: profiles and tables (FEATURE_IDEAS §10) ──


_TABLE_FREQS = np.array([500.0, 1000.0, 4000.0, 10000.0])


def _quiet(fn, *args, **kwargs):
    """``fn(*args, **kwargs)`` with the fitted-envelope notices silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ValidityWarning)
        return fn(*args, **kwargs)


#: ``absorption_thorp`` / ``absorption_francois_garrison`` /
#: ``absorption_biological`` / ``absorption_constant`` outputs, captured on
#: 2026-10-04 before those functions were removed (round-trip ``repr`` of each
#: float), with the call that produced each. ``law.table`` must return them
#: element for element.
_CAPTURED_F = [100.0, 1000.0, 10000.0, 100000.0]
_CAPTURED = [
    ('thorp', lambda: Thorp().table(_CAPTURED_F), 'dB/km',
     [0.004499425722313501, 0.06932909046574005, 1.1898299387081566,
      34.31896275996514]),
    ('fg_row', lambda: FrancoisGarrison(
        temperature=4.0, salinity=34.5, pH=7.9).table(
            _CAPTURED_F, depths=800.0), 'dB/km',
     [0.0010607628868378968, 0.05277449222948564, 1.0041500861843837,
      24.457828436118625]),
    ('fg_row_total', lambda: FrancoisGarrison(
        temperature=4.0, salinity=34.5, pH=7.9,
        ph_scale='total').table(_CAPTURED_F, depths=800.0,
                                units='dB/wavelength',
                                sound_speed=1490.0), 'dB/wavelength',
     [1.866831858607032e-05, 9.144975249397685e-05, 0.00015189729324640726,
      0.0003646513241696668]),
    ('fg_profile', lambda: FrancoisGarrison(
        temperature=[20.0, 20.0, 8.0, 8.0], salinity=[35.0, 35.0, 34.0, 34.0],
        depths=[0.0, 40.0, 60.0, 100.0]).table(
            [1000.0, 10000.0], depths=[0.0, 40.0, 60.0, 100.0]), 'dB/km',
     [[0.05170849527344906, 0.7369468868638465],
      [0.05165478151664996, 0.733543368095234],
      [0.06101932207729469, 0.9859416083488592],
      [0.06094398360027162, 0.9808171739134799]]),
    ('biological', lambda: Biological(layers=[(30, 45, 1000, 5, 1)]).table(
        [500.0, 1000.0, 2000.0], depths=[0.0, 40.0, 90.0]), 'dB/km',
     [[0.0, 0.0, 0.0], [0.11061946902654868, 25.0, 1.6597510373443982],
      [0.0, 0.0, 0.0]]),
    ('constant', lambda: ConstantAbsorption(
        value_dB_per_wavelength=0.02).table([1000.0, 10000.0], units='dB/m'),
     'dB/m', [0.013333333333333334, 0.13333333333333333]),
]


class TestTheLawIsTheOneObject:
    """One public object per absorption model: the law. ``law.table(f)``
    evaluates it; an environment takes the law, never its table."""

    @pytest.mark.parametrize('make, units, values', [c[1:] for c in _CAPTURED],
                             ids=[c[0] for c in _CAPTURED])
    def test_the_table_is_the_removed_functions_output(self, make, units,
                                                       values):
        table = _quiet(make)
        assert table.units == units
        np.testing.assert_array_equal(np.asarray(table.data),
                                      np.asarray(values))

    @pytest.mark.parametrize('law', [
        Thorp(), FrancoisGarrison(4.0, 34.5, 7.9),
        Biological(layers=[(30, 45, 1000, 5, 1)]), ConstantAbsorption(0.02)],
        ids=['thorp', 'fg-row', 'biological', 'constant'])
    def test_a_laws_table_is_refused_naming_the_law(self, law):
        table = _quiet(law.table, _TABLE_FREQS)
        with pytest.raises(ConfigurationError,
                           match='Pass the law itself') as caught:
            Environment(bathymetry=100.0, absorption=table)
        assert repr(table.model) in str(caught.value)
        # The law itself is taken, and a measured table (no model) too.
        assert Environment(bathymetry=100.0, absorption=law).absorption == law
        Environment(bathymetry=100.0, absorption=measured_absorption_table())

    def test_thorp_keeps_its_letter_and_francois_garrison_goes_into_the_rows(
            self):
        # Thorp has no depth term, so its TopOpt(4) letter is exact. AT's
        # 'F' evaluates Francois-Garrison at one z_bar for the whole column,
        # so on a one-frequency deck every Francois-Garrison law goes into
        # the SSP rows' alphaI under a blank letter; only one water row on a
        # deck covering several frequencies takes 'F' (exact in frequency).
        assert volume_attenuation_code(Thorp()) == 'T'
        assert not writes_alpha_per_ssp_row(Thorp())
        for law in (FrancoisGarrison(), two_layer_absorption()):
            assert volume_attenuation_code(law) == ' '
            assert writes_alpha_per_ssp_row(law)
        assert volume_attenuation_code(FrancoisGarrison(),
                                       multi_frequency=True) == 'F'
        assert not writes_alpha_per_ssp_row(FrancoisGarrison(),
                                            multi_frequency=True)
        assert volume_attenuation_code(two_layer_absorption(),
                                       multi_frequency=True) == ' '
        assert writes_alpha_per_ssp_row(two_layer_absorption(),
                                        multi_frequency=True)


class TestAMeasuredTable:
    """A table with no law behind it is used as tabulated."""

    def test_it_is_linear_in_depth_and_in_log_frequency(self):
        env = Environment(bathymetry=100.0,
                          absorption=measured_absorption_table())
        mid_f = float(np.sqrt(1000.0 * 10000.0))
        a = env.absorption.table([1000.0, mid_f, 10000.0],
                                 depths=[0.0, 50.0, 100.0]).data
        np.testing.assert_allclose(a[:, 0], [0.06, 0.05, 0.04], rtol=1e-13)
        np.testing.assert_allclose(a[:, 2], [0.90, 0.70, 0.50], rtol=1e-13)
        np.testing.assert_allclose(a[1, 1], 0.25 * (0.06 + 0.9 + 0.04 + 0.5),
                                   rtol=1e-13)

    def test_it_refuses_a_frequency_outside_it(self):
        env = Environment(bathymetry=100.0,
                          absorption=measured_absorption_table())
        with pytest.raises(ConfigurationError, match='outside the table'):
            env.absorption.table([20000.0])
        with pytest.raises(ConfigurationError, match='outside the table'):
            env.absorption.table([999.0])
        # Both ends are inside.
        env.absorption.table([1000.0, 10000.0])

    def test_a_short_table_is_held_and_announced(self):
        with recorded_warnings() as rec:
            Environment(bathymetry=100.0,
                        absorption=measured_absorption_table((0.0, 60.0)))
        notes = [str(w.message) for w in rec
                 if issubclass(w.category, ValidityWarning)]
        assert notes and 'below 60 m it takes the 60 m row' in notes[0]
        with recorded_warnings() as rec:
            Environment(bathymetry=100.0,
                        absorption=measured_absorption_table((0.0, 100.0)))
        assert not [w for w in rec if issubclass(w.category, ValidityWarning)]


def test_a_profile_prints_its_ocean():
    assert repr(two_layer_absorption()) == (
        'FrancoisGarrison(T 8–20 °C (4 depths), S 34–35 psu (4 depths), '
        'pH 8)')
    table = two_layer_absorption().table([1000.0], depths=TWO_LAYER_DEPTHS)
    assert repr(table) == (
        'AbsorptionCoefficient(francois_garrison, T 8–20 °C (4 depths), '
        'S 34–35 psu (4 depths), pH 8, depths [0, 40, 60, 100] m, '
        'frequency 1000 Hz, dB/km)')
    env = Environment(bathymetry=100.0, absorption=measured_absorption_table())
    assert 'absorption=tabulated' in repr(env)


class TestReferenceWaterReplacingTheEnvironmentsOwn:
    """A Francois-Garrison on the reference water assigned over the
    environment's own Francois-Garrison water is announced once the two
    differ by more than 2 °C or 1 psu."""

    @staticmethod
    def _replace(t, s):
        env = Environment(bathymetry=100.0, absorption=FrancoisGarrison(
            [t, t], [s, s], 8.0, depths=[0.0, 100.0]))
        with recorded_warnings() as rec:
            env.absorption = FrancoisGarrison(10.0, 35.0, 8.0)
        return [str(w.message) for w in rec
                if issubclass(w.category, ProvenanceWarning)]

    def test_it_warns_past_two_degrees(self):
        assert self._replace(12.0, 35.0) == []
        (msg,) = self._replace(12.01, 35.0)
        assert 'reference water' in msg and 'T 12' in msg
        assert 'FrancoisGarrison(temperature=T' in msg

    def test_it_warns_past_one_psu(self):
        assert self._replace(10.0, 34.0) == []
        assert len(self._replace(10.0, 33.99)) == 1

    def test_no_water_to_compare_means_no_warning(self):
        env = Environment(bathymetry=100.0)
        with recorded_warnings() as rec:
            env.absorption = FrancoisGarrison(10.0, 35.0, 8.0)
        assert not [w for w in rec
                    if issubclass(w.category, ProvenanceWarning)]


class TestAProfileOutsideTheEnvelopeWarnsOnce:
    """A T/S/pH profile outside the fitted envelope gives one warning per
    law, naming the span of the values outside and how many depths carry
    them, not one warning per value."""

    DEPTHS = np.linspace(0.0, 390.0, 40)

    def _salinity(self, deep):
        salinity = np.full(40, 34.0)
        salinity[28:] = deep
        return salinity

    def _validity(self, **profile):
        with recorded_warnings() as rec:
            FrancoisGarrison(depths=self.DEPTHS, **profile)
        return [str(w.message) for w in rec
                if issubclass(w.category, ValidityWarning)]

    def test_twelve_depths_outside_give_one_warning_naming_span_and_count(self):
        (msg,) = self._validity(
            salinity=self._salinity(np.linspace(41.1, 42.4, 12)))
        assert 'salinity=41.1–42.4 at 12 of 40 depths is outside 30..41 PSU' \
            in msg

    def test_a_profile_on_the_bound_is_silent(self):
        assert self._validity(salinity=self._salinity(41.0)) == []

    def test_one_depth_just_past_the_bound_is_counted_alone(self):
        salinity = self._salinity(41.0)
        salinity[-1] = 41.01
        (msg,) = self._validity(salinity=salinity)
        assert 'salinity=41 at 1 of 40 depths is outside 30..41 PSU' in msg

    def test_two_fields_outside_share_the_one_warning(self):
        temperature = np.full(40, 10.0)
        temperature[:5] = 31.0
        (msg,) = self._validity(
            temperature=temperature,
            salinity=self._salinity(np.linspace(41.1, 42.4, 12)))
        assert 'temperature=31 at 5 of 40 depths' in msg
        assert 'salinity=41.1–42.4 at 12 of 40 depths' in msg
