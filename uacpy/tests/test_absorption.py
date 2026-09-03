"""Francois-Garrison volume absorption: the builder and the formula's domain.

The first half covers ``uacpy.data.absorption.build_francois_garrison`` — how
a measured profile is reduced to the single representative row the models use.

The second half covers ``uacpy.core.absorption`` itself: which rows the
formula will answer for. Francois-Garrison is an empirical fit with a stated
domain, and outside it the expression goes complex or divides by zero. A row
that evaluates to NaN is refused by name rather than returned, and the
boundary values of the domain are accepted — both sides pinned, so the guard
cannot be loosened or tightened without a test moving.

No binary runs here; this is arithmetic and validation.
"""

import dataclasses
import inspect
import re
import warnings

import numpy as np
import pytest

from uacpy.core.absorption import (
    Biological, BiologicalLayer, FrancoisGarrison,
    convert_attenuation_units, francois_garrison_db_per_km,
)
from uacpy.core.constants import (
    DEFAULT_SOUND_SPEED, MAX_ATTENUATION_DB_PER_WAVELENGTH,
)
from uacpy.core.exceptions import ConfigurationError
from uacpy.data.absorption import build_francois_garrison


def test_builds_from_the_mid_depth_row_by_default():
    """The single row this picks sets the temperature the models use for the
    whole column (they vary only depth), so the representative choice is the
    middle of the profile, not its shallowest sample. See
    TestNominalRowIsColumnRepresentative for the dB consequence."""
    fg = build_francois_garrison([0.0, 50.0, 100.0], [18.0, 16.0, 13.0],
                                 [36.0, 36.1, 36.2])
    assert isinstance(fg, FrancoisGarrison)
    assert fg.temperature_c == 16.0      # mid-depth sample, not the surface
    assert fg.salinity_psu == 36.1
    assert fg.z_bar_m == 50.0
    assert fg.pH == 8.1                   # default open-ocean


def test_reference_depth_picks_nearest_level():
    fg = build_francois_garrison([0.0, 50.0, 100.0], [18.0, 16.0, 13.0],
                                 [36.0, 36.1, 36.2], reference_depth=60.0, pH=7.9)
    assert fg.z_bar_m == 50.0             # nearest to 60 m
    assert fg.temperature_c == 16.0
    assert fg.pH == 7.9


def test_mismatched_or_empty_raises():
    with pytest.raises(ConfigurationError):
        build_francois_garrison([], [], [])
    with pytest.raises(ConfigurationError):
        build_francois_garrison([0.0, 10.0], [18.0], [36.0])


class TestNominalRowIsColumnRepresentative:
    """The models receive a **single** T/S row and re-evaluate the formula in
    depth only, so that row's temperature governs absorption for the whole
    column. Taking it at the surface carried the warmest water down the entire
    profile: on a mid-latitude column (22 C surface, 4 C at 2 km) that
    understated absorption by 34 % at 10 kHz and 20 % at 1 kHz against a
    mid-column reference."""

    Z = np.array([0.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0])
    T = np.array([22.0, 20.0, 16.0, 12.0, 8.0, 5.0, 4.0])
    S = np.full(7, 35.0)

    def _fg(self, ref=None):
        return build_francois_garrison(self.Z, self.T, self.S,
                                       reference_depth=ref)

    def test_default_takes_the_mid_depth_row(self):
        fg = self._fg()
        assert fg.z_bar_m == pytest.approx(1000.0)
        assert fg.temperature_c == pytest.approx(5.0)

    def test_explicit_reference_wins(self):
        # The discriminating counterpart: the default changed, the override
        # did not.
        assert self._fg(ref=0.0).temperature_c == pytest.approx(22.0)
        assert self._fg(ref=1000.0).temperature_c == pytest.approx(5.0)

    def test_surface_reference_understates_high_frequency_absorption(self):
        # Pins the reason the default moved, in dB rather than in degrees.
        zq = np.array([500.0])
        a_mid = float(np.ravel(self._fg().alpha_db_per_m(1e4, zq))[0])
        a_surf = float(np.ravel(self._fg(ref=0.0).alpha_db_per_m(1e4, zq))[0])
        assert a_surf < a_mid
        assert (a_mid - a_surf) / a_mid == pytest.approx(0.34, abs=0.05)

    def test_isothermal_column_is_unaffected_by_the_change(self):
        # Where there is no stratification there is nothing to choose, so the
        # mid-column default must agree with reference_depth=0.0 exactly.
        z = np.array([0.0, 100.0, 500.0])
        t = np.full(3, 12.0)
        s = np.full(3, 35.0)
        top = build_francois_garrison(z, t, s, reference_depth=0.0)
        mid = build_francois_garrison(z, t, s)
        assert mid.temperature_c == pytest.approx(top.temperature_c)


# (kwargs, the fragment of the message that names the offending field)
_REFUSED_ROWS = [
    (dict(temperature_c=10.0, salinity_psu=-1e-12, pH=8.0, z_bar_m=0.0),
     'salinity_psu'),
    (dict(temperature_c=-273.0, salinity_psu=35.0, pH=8.0, z_bar_m=0.0),
     'temperature_c'),
    (dict(temperature_c=10.0, salinity_psu=35.0, pH=-1e-12, z_bar_m=0.0),
     'pH'),
    # c = 1412 + 3.21·T + 1.19·S + 0.0167·z = -15288 m/s: every mechanism
    # divides by it. Only a synthetic depth reaches this with the other
    # three fields in range, which is why it is checked separately.
    (dict(temperature_c=0.0, salinity_psu=0.0, pH=8.0, z_bar_m=-1e6),
     'sound speed'),
    # The row from the report: all four fields out of range at once.
    (dict(temperature_c=-500.0, salinity_psu=-10.0, pH=-3.0, z_bar_m=-99.0),
     'salinity_psu'),
]


@pytest.mark.parametrize('kwargs,names', _REFUSED_ROWS)
def test_francois_garrison_refuses_rows_that_evaluate_to_nan(kwargs, names):
    with pytest.raises(ConfigurationError, match=re.escape(names)):
        FrancoisGarrison(**kwargs)


# The other side of each threshold: the last value that is still a number.
_ACCEPTED_ROWS = [
    dict(temperature_c=10.0, salinity_psu=0.0, pH=8.0, z_bar_m=0.0),
    dict(temperature_c=-272.999, salinity_psu=35.0, pH=8.0, z_bar_m=0.0),
    dict(temperature_c=10.0, salinity_psu=35.0, pH=0.0, z_bar_m=0.0),
    # c = 9.2 m/s — absurd, but positive, so nothing here refuses it.
    dict(temperature_c=0.0, salinity_psu=0.0, pH=8.0, z_bar_m=-84000.0),
    # The ordinary mid-latitude row every other test in the suite uses.
    dict(temperature_c=10.0, salinity_psu=35.0, pH=8.0, z_bar_m=1000.0),
]


@pytest.mark.parametrize('kwargs', _ACCEPTED_ROWS)
def test_francois_garrison_accepts_the_boundary_values(kwargs):
    absorption = FrancoisGarrison(**kwargs)
    assert absorption.topopt_code() == 'F'


def test_the_bare_formula_answers_an_out_of_domain_row_with_nan_only():
    """The module-level formula keeps its no-validation contract — but the
    NaN comes back without numpy's raw ``RuntimeWarning``, which would be the
    one warning uacpy emits that is not a ``UserWarning``."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        alpha = francois_garrison_db_per_km(
            1000.0, temperature=-500.0, salinity=-10.0, pH=-3.0, depth=50.0)
    assert np.isnan(alpha)
    assert record == [], [str(w.message) for w in record]


def test_an_in_domain_row_is_unchanged_by_the_errstate_guard():
    """Silencing the invalid flag must not touch the numbers."""
    alpha = francois_garrison_db_per_km(
        10_000.0, temperature=10.0, salinity=35.0, pH=8.0, depth=1000.0)
    assert 0.0 < float(alpha) < 10.0


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
    def _ceiling_db_km(f0):
        return (MAX_ATTENUATION_DB_PER_WAVELENGTH * 1000.0 * f0
                / DEFAULT_SOUND_SPEED)

    @staticmethod
    def _layer(**kw):
        args = {'z_top_m': 0.0, 'z_bottom_m': 50.0, 'f0_hz': 100.0,
                'Q': 10.0, 'a0': 1.0}
        args.update(kw)
        return BiologicalLayer(**args)

    def test_the_documented_at_threshold_is_3638_db_per_km_at_100hz(self):
        assert self._ceiling_db_km(100.0) == pytest.approx(3638.34, abs=0.01)

    def test_a_peak_over_the_ceiling_warns(self):
        with pytest.warns(UserWarning, match='CRCI'):
            self._layer(Q=61.0, a0=1.0)      # a0·Q² = 3721 dB/km

    def test_a_peak_under_the_ceiling_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self._layer(Q=60.0, a0=1.0)      # a0·Q² = 3600 dB/km

    def test_the_warning_names_the_peak_and_the_ceiling(self):
        with pytest.warns(UserWarning) as rec:
            self._layer(Q=61.0, a0=1.0)
        message = str(rec[0].message)
        assert '3721' in message
        assert '3638' in message

    def test_the_ceiling_scales_with_the_resonance_frequency(self):
        """The bound is on Nepers/m against ω/c, so ten times the resonance
        frequency buys ten times the dB/km — the same layer that warns at
        100 Hz is comfortable at 1000 Hz."""
        with pytest.warns(UserWarning, match='CRCI'):
            self._layer(f0_hz=100.0, Q=61.0, a0=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self._layer(f0_hz=1000.0, Q=61.0, a0=1.0)

    def test_the_layer_is_built_and_computes(self):
        """A warning, not a refusal: the object exists and its Lorentzian is
        untouched."""
        from uacpy.core.absorption import Biological
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            bio = Biological(layers=[(0.0, 50.0, 100.0, 61.0, 1.0)])
        got = float(bio.alpha_db_per_m(100.0, [25.0])[0]) * 1000.0
        assert got == pytest.approx(1.0 * 61.0 ** 2, rel=1e-12)

    @pytest.mark.parametrize('kwargs, match', [
        (dict(f0_hz=0.0), 'f0_hz'), (dict(Q=0.0), 'Q'), (dict(a0=0.0), 'a0')])
    def test_the_existing_refusals_run_before_the_ceiling_check(self, kwargs,
                                                                match):
        """f0 = 0 divides in the ceiling formula, so the ordering matters."""
        with pytest.raises(ConfigurationError, match=match):
            self._layer(**kwargs)

    def test_the_class_docstring_names_the_pair_that_straddles_the_ceiling(self):
        """The docstring quotes a worked example either side of the 3638 dB/km
        bound. It read ``Q = 61 clears it`` while 61 gives 3721 and warns —
        the two tests above already had it the other way round. Pinned against
        the computed ceiling so the prose cannot drift off the arithmetic
        again."""
        ceiling = self._ceiling_db_km(100.0)
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
    deck via ``as_at_tuples``, where ``AttenMod.f90``'s band test
    ``z >= Z1 .AND. z <= Z2`` (:104) is False at every depth for a NaN or a
    negative bound — so the layer is written to the file and then contributes
    nothing, which is the failure mode the typed refusal replaces."""

    @staticmethod
    def _layer(**kw):
        args = {'z_top_m': 0.0, 'z_bottom_m': 50.0, 'f0_hz': 100.0,
                'Q': 10.0, 'a0': 1.0}
        args.update(kw)
        return BiologicalLayer(**args)

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
                self._layer(**kwargs)

    def test_a_non_finite_f0_is_named_by_the_layer_not_the_unit_converter(self):
        """``f0_hz`` was the one field that already failed, but from inside
        ``convert_attenuation_units`` in the ceiling formula — the message
        named that function and its frequency argument, not the layer field
        the user wrote."""
        with pytest.raises(ConfigurationError) as excinfo:
            self._layer(f0_hz=float('nan'))
        assert 'convert_attenuation_units' not in str(excinfo.value)

    def test_a_finite_zero_reaches_the_sign_check_not_the_finiteness_guard(self):
        """The finiteness guards run first but do not take over the sign
        verdicts: zero is finite, so ``f0_hz = 0`` still fails as a sign
        error, keeping the ``"must be positive"`` phrase those refusals own."""
        with pytest.raises(ConfigurationError, match='f0_hz must be positive'):
            self._layer(f0_hz=0.0)

    def test_a_layer_at_zero_depth_is_accepted(self):
        """The depth bound is non-negative, not positive: a layer whose top
        sits at the sea surface is an ordinary configuration."""
        assert self._layer(z_top_m=0.0, z_bottom_m=50.0).z_top_m == 0.0


@pytest.mark.parametrize('cls', [BiologicalLayer, Biological],
                         ids=['BiologicalLayer', 'Biological'])
def test_the_written_out_init_takes_exactly_the_dataclass_fields(cls):
    """Both classes hand-write the ``__init__`` the decorator would generate,
    so that no ``<string>`` frame sits between the layer ceiling warning and
    the user (see the class docstrings). The cost is that the field list now
    exists twice, and the drift is silent in one direction: an annotation
    added below still shapes ``repr`` / ``__eq__`` / ``fields()`` while the
    constructor has no way to set it, so instances carry the attribute only
    when something else assigns it. Pinned in the order the 5-tuple
    ``as_at_tuples`` writes, which is the same order."""
    parameters = list(inspect.signature(cls.__init__).parameters)[1:]
    assert parameters == [f.name for f in dataclasses.fields(cls)]


class TestTheTwoAbsorptionRoutesDivergeByTheDocumentedAmount:
    """One absorption model, two documented routes, different answers.

    The Python accessor and the Acoustics-Toolbox deck evaluate the same
    formula at different arguments: Francois-Garrison per caller depth here
    against one module-level ``z_bar`` there (``misc/AttenMod.f90:148-160``),
    and a constant absorption converted at 1500 m/s here against each SSP
    row's own ``c`` there (``AttenMod.f90:73``). Neither side is wrong and
    neither is changed — ``test_absorption.py`` above and
    ``test_modes_perturbation.py`` pin the per-depth behaviour deliberately.
    What was missing is that ``Modes.with_attenuation`` sends the user from
    one to the other without saying so. These are the numbers its
    documentation now quotes."""

    @staticmethod
    def _fg():
        return FrancoisGarrison(10, 35, 8, z_bar_m=1000)

    @classmethod
    def _percent_over_deck(cls, frequency, depth):
        fg = cls._fg()
        python = float(np.ravel(fg.alpha_db_per_m(frequency, [depth]))[0]) * 1000.0
        deck = float(francois_garrison_db_per_km(
            frequency, fg.temperature_c, fg.salinity_psu, fg.pH, fg.z_bar_m))
        return 100.0 * (python - deck) / deck

    @pytest.mark.parametrize('frequency, expected', [
        (1e3, 3.0), (1e4, 13.9), (3e4, 15.5), (1e5, 15.0)])
    def test_the_surface_gap_is_what_with_attenuation_documents(
            self, frequency, expected):
        assert self._percent_over_deck(frequency, 0.0) == pytest.approx(
            expected, abs=0.05)

    def test_the_two_routes_agree_exactly_at_z_bar(self):
        # The pivot the whole divergence turns on, and the discriminating
        # check that the gap is the depth argument and nothing else.
        assert self._percent_over_deck(1e4, 1000.0) == pytest.approx(0.0,
                                                                    abs=1e-9)

    def test_the_gap_reverses_sign_below_z_bar(self):
        # Not a bias: the accessor is under the deck as far as it is over it.
        assert self._percent_over_deck(1e4, 2000.0) == pytest.approx(-12.4,
                                                                     abs=0.05)

    @pytest.mark.parametrize('sound_speed, expected', [
        (1450.0, -3.33), (1500.0, 0.0), (1550.0, 3.33)])
    def test_a_constant_absorption_diverges_with_the_ssp_sound_speed(
            self, sound_speed, expected):
        from uacpy.core.absorption import ConstantAbsorption
        python = float(np.ravel(
            ConstantAbsorption(0.5).alpha_db_per_m(1e3, [0.0]))[0])
        deck = float(convert_attenuation_units(
            0.5, 1e3, 'dB/wavelength', 'dB/m', sound_speed=sound_speed))
        assert 100.0 * (python - deck) / deck == pytest.approx(expected,
                                                               abs=0.01)

    def test_with_attenuation_warns_the_reader_that_the_routes_differ(self):
        from uacpy.core.results.modes import Modes
        doc = ' '.join(Modes.with_attenuation.__doc__.split())
        assert 'not the same number the solver used' in doc
        assert 'misc/AttenMod.f90:148-160' in doc
        assert 'AttenMod.f90:73' in doc
        assert '+13.9 %' in doc and '±3.3 %' in doc

    def test_both_absorption_classes_cross_reference_the_divergence(self):
        from uacpy.core.absorption import ConstantAbsorption
        fg_doc = ' '.join(FrancoisGarrison.__doc__.split())
        assert 'The deck does not do what the accessor does' in fg_doc
        assert 'misc/AttenMod.f90:148-160' in fg_doc
        ca_doc = ' '.join(ConstantAbsorption.__doc__.split())
        assert 'misc/AttenMod.f90:73' in ca_doc
        assert '±3.3 %' in ca_doc
