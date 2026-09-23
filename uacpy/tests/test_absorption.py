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
    ph_to_nbs,
    Biological, BiologicalLayer, FrancoisGarrison,
    convert_attenuation_units, francois_garrison_dB_per_km,
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
        a_mid = float(np.ravel(self._fg().alpha_dB_per_m(1e4, zq))[0])
        a_surf = float(np.ravel(self._fg(ref=0.0).alpha_dB_per_m(1e4, zq))[0])
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


@pytest.mark.parametrize('bad', [np.inf, -np.inf, np.nan])
@pytest.mark.parametrize(
    'name', ['temperature_c', 'salinity_psu', 'pH', 'z_bar_m'])
def test_francois_garrison_refuses_a_non_finite_field_naming_it(name, bad):
    """inf passes every range guard: inf T or S evaluate to a NaN alpha,
    inf pH to an inf alpha, and inf z_bar_m to a finite alpha whose deck
    record reads "inf". The finiteness check runs ahead of the range guards
    so the message names the field, not the sound speed it feeds."""
    kwargs = dict(temperature_c=10.0, salinity_psu=35.0, pH=8.0,
                  z_bar_m=1000.0)
    kwargs[name] = bad
    with pytest.raises(ConfigurationError,
                       match=re.escape(name) + '.*must be finite'):
        FrancoisGarrison(**kwargs)


def test_the_bare_formula_answers_an_out_of_domain_row_with_nan_only():
    """The module-level formula keeps its no-validation contract — but the
    NaN comes back without numpy's raw ``RuntimeWarning``, which would be the
    one warning uacpy emits that is not a ``UserWarning``."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        alpha = francois_garrison_dB_per_km(
            1000.0, temperature=-500.0, salinity=-10.0, pH=-3.0, depth=50.0)
    assert np.isnan(alpha)
    assert record == [], [str(w.message) for w in record]


def test_an_in_domain_row_is_unchanged_by_the_errstate_guard():
    """Silencing the invalid flag must not touch the numbers."""
    alpha = francois_garrison_dB_per_km(
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
    def _ceiling_dB_km(f0):
        return (MAX_ATTENUATION_DB_PER_WAVELENGTH * 1000.0 * f0
                / DEFAULT_SOUND_SPEED)

    @staticmethod
    def _layer(**kw):
        args = {'z_top_m': 0.0, 'z_bottom_m': 50.0, 'f0_hz': 100.0,
                'Q': 10.0, 'a0': 1.0}
        args.update(kw)
        return BiologicalLayer(**args)

    def test_the_documented_at_threshold_is_3638_dB_per_km_at_100hz(self):
        assert self._ceiling_dB_km(100.0) == pytest.approx(3638.34, abs=0.01)

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
        got = float(bio.alpha_dB_per_m(100.0, [25.0])[0]) * 1000.0
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
        python = float(np.ravel(fg.alpha_dB_per_m(frequency, [depth]))[0]) * 1000.0
        deck = float(francois_garrison_dB_per_km(
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
            ConstantAbsorption(0.5).alpha_dB_per_m(1e3, [0.0]))[0])
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
        assert ph_to_nbs(8.0, 'seawater', temperature_c=4.0,
                         salinity_psu=35.0) == pytest.approx(8.1001, abs=1e-3)
        assert ph_to_nbs(8.0, 'seawater', temperature_c=25.0,
                         salinity_psu=35.0) == pytest.approx(8.1467, abs=1e-3)

    def test_total_scale_is_treated_as_the_seawater_scale(self):
        assert ph_to_nbs(7.9, 'total', temperature_c=10.0, salinity_psu=20.0) \
            == pytest.approx(8.0334, abs=1e-3)

    def test_nbs_passes_through_unchanged(self):
        assert ph_to_nbs(8.0, 'nbs', temperature_c=25.0, salinity_psu=35.0) \
            == 8.0

    def test_an_unknown_scale_is_refused_naming_the_choices(self):
        with pytest.raises(ConfigurationError, match="'nbs'.*'total'.*'seawater'"):
            ph_to_nbs(8.0, 'free', temperature_c=4.0, salinity_psu=35.0)

    def test_broadcasts_over_arrays(self):
        out = ph_to_nbs(np.array([7.8, 8.0]), 'total',
                        temperature_c=np.array([4.0, 25.0]), salinity_psu=35.0)
        assert out == pytest.approx([7.9001, 8.1467], abs=1e-3)


class TestFrancoisGarrisonPhScale:
    """``FrancoisGarrison`` takes the scale its ``pH`` is on and converts to
    NBS once, for both the in-Python formula and the tuple the AT deck gets —
    the solver evaluates the same equation on whatever number it is written,
    so the two routes must be handed the same pH."""

    def _pair(self, scale):
        return FrancoisGarrison(temperature_c=4.0, salinity_psu=35.0, pH=8.0,
                                z_bar_m=1000.0, ph_scale=scale)

    def test_the_default_scale_is_nbs_and_leaves_every_number_alone(self):
        fg = FrancoisGarrison(temperature_c=4.0, salinity_psu=35.0, pH=8.0,
                              z_bar_m=1000.0)
        assert fg.ph_scale == 'nbs'
        assert fg.ph_nbs == 8.0
        assert fg.as_at_tuple()[2] == 8.0

    def test_total_scale_converts_before_the_boric_term_and_in_the_deck_tuple(self):
        total = self._pair('total')
        converted = ph_to_nbs(8.0, 'total', temperature_c=4.0, salinity_psu=35.0)
        nbs = FrancoisGarrison(temperature_c=4.0, salinity_psu=35.0,
                               pH=converted, z_bar_m=1000.0)
        assert total.ph_nbs == pytest.approx(converted)
        assert total.as_at_tuple()[2] == pytest.approx(converted)
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
        fg = build_francois_garrison([0.0, 100.0], [10.0, 8.0], [35.0, 35.0],
                                     pH=7.9, ph_scale='total')
        assert fg.ph_scale == 'total'
        assert fg.pH == 7.9
        assert fg.ph_nbs > 7.9
        assert build_francois_garrison([0.0, 100.0], [10.0, 8.0],
                                       [35.0, 35.0]).ph_scale == 'nbs'


class TestAlphaIsEvaluatedOnBothAxesFromOneEvaluator:
    """alpha(f, z) through one door, in whichever units are asked for.

    Before this, the package exposed the same formula twice and the two were
    transposes of each other: ``francois_garrison_dB_per_km`` vectorised over
    frequency with a scalar depth, ``FrancoisGarrison.alpha_dB_per_m``
    vectorised over depth with a scalar frequency, and an array frequency into
    the second raised ``TypeError: only 0-dimensional arrays can be converted
    to Python scalars``. Neither could answer alpha(f, z).
    """

    FG = dict(temperature_c=10.0, salinity_psu=35.0, pH=8.0, z_bar_m=100.0)
    F = np.array([1e3, 1e4, 1e5])

    def test_an_array_of_frequencies_is_evaluated_elementwise(self):
        """One value per frequency, from one call."""
        from uacpy.core.absorption import Thorp
        a = Thorp().alpha(self.F)
        assert a.values.shape == (3,)

    def test_both_axes_together_give_a_depth_by_frequency_grid(self):
        """Depth-first, the shape convention the package uses everywhere."""
        from uacpy.core.absorption import absorption_francois_garrison
        z = np.array([0.0, 50.0, 100.0, 200.0])
        a = absorption_francois_garrison(self.F, depths=z, **self.FG)
        assert a.values.shape == (z.size, self.F.size)
        assert a.n_depths == 4 and a.n_frequencies == 3
        assert a.is_depth_dependent

    def test_the_grid_agrees_with_both_old_doors_elementwise(self):
        """The row-and-column check that would have caught the transpose:
        every cell must equal what the frequency-vectorised free function and
        the depth-vectorised method each return for that cell."""
        from uacpy.core.absorption import (absorption_francois_garrison,
                                           francois_garrison_dB_per_km,
                                           FrancoisGarrison)
        z = np.array([0.0, 100.0, 250.0])
        grid = absorption_francois_garrison(self.F, depths=z, **self.FG).values
        for j, f in enumerate(self.F):                     # column <- old method
            col = FrancoisGarrison(**self.FG).alpha_dB_per_m(f, z) * 1000.0
            np.testing.assert_allclose(grid[:, j], col, rtol=1e-12)
        for i, zz in enumerate(z):                         # row <- old function
            row = francois_garrison_dB_per_km(
                self.F, temperature=self.FG['temperature_c'],
                salinity=self.FG['salinity_psu'], pH=self.FG['pH'], depth=zz)
            np.testing.assert_allclose(grid[i, :], row, rtol=1e-12)

    def test_without_depths_francois_garrison_sits_at_its_own_z_bar(self):
        """``z_bar_m`` is the model's own depth; a caller-supplied axis
        overrides it (the class docstring says so), and no axis means the
        model's own value rather than an invented surface."""
        from uacpy.core.absorption import absorption_francois_garrison
        flat = absorption_francois_garrison(self.F, **self.FG)
        at_zbar = absorption_francois_garrison(
            self.F, depths=[self.FG['z_bar_m']], **self.FG)
        assert not flat.is_depth_dependent
        np.testing.assert_allclose(flat.values, at_zbar.values[0], rtol=1e-12)

    def test_the_two_spellings_are_one_evaluator(self):
        from uacpy.core.absorption import absorption_thorp, Thorp
        np.testing.assert_array_equal(absorption_thorp(self.F).values,
                                      Thorp().alpha(self.F).values)

    def test_the_carrier_records_which_formula_made_it(self):
        """Provenance, mirroring ``SoundSpeedProfile.formula``."""
        from uacpy.core.absorption import (absorption_thorp,
                                           absorption_francois_garrison)
        assert absorption_thorp(self.F).model == 'thorp'
        assert absorption_francois_garrison(
            self.F, **self.FG).model == 'francois_garrison'

    def test_units_are_carried_not_baked_into_a_name(self):
        from uacpy.core.absorption import absorption_thorp
        a = absorption_thorp(self.F)
        assert a.units == 'dB/km'
        np.testing.assert_allclose(a.to_units('dB/m').values,
                                   a.values / 1000.0, rtol=1e-12)
        assert a.to_units('dB/m').units == 'dB/m'

    def test_a_frequency_dependent_unit_needs_the_axis_the_carrier_keeps(self):
        """The structural reason for a carrier rather than a bare array:
        dB/wavelength, Q and L cannot be evaluated once the frequency axis is
        gone. Converting to one must use each frequency, not a single value."""
        from uacpy.core.absorption import absorption_thorp
        from uacpy.core.absorption import convert_attenuation_units
        a = absorption_thorp(self.F)
        got = a.to_units('dB/wavelength', sound_speed=1500.0).values
        want = [float(convert_attenuation_units(v, f, 'dB/km', 'dB/wavelength',
                                                sound_speed=1500.0))
                for v, f in zip(a.values, self.F)]
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_francois_garrison_refuses_to_invent_an_ocean(self):
        """The kernel defaults T/S/pH/depth; the function does not. F&G's
        answer is a statement about a particular ocean."""
        from uacpy.core.absorption import absorption_francois_garrison
        with pytest.raises(TypeError):
            absorption_francois_garrison(self.F)


def test_every_absorption_model_has_a_function_spelling():
    """Four models, four functions — no model you have to remember is special.

    The first draft gave `Thorp` and `FrancoisGarrison` a function form and
    left `Biological` and `ConstantAbsorption` reachable only as classes, on
    the reasoning that a layer list and a scalar "are not a frequency-grid
    call". They are ordinary parameters; the asymmetry was a leftover from a
    registry design that no longer exists, and it cost a reader one special
    case to memorise. This fails if a fifth model arrives without one.
    """
    import inspect
    import uacpy.core.absorption as A
    concrete = {c for c in vars(A).values()
                if inspect.isclass(c) and issubclass(c, A.Absorption)
                and c is not A.Absorption}
    have = {name[len('absorption_'):] for name in vars(A)
            if name.startswith('absorption_') and callable(getattr(A, name))}
    want = {c(**{f.name: _SAMPLE[f.name] for f in dataclasses.fields(c)
                 if f.name in _SAMPLE})._model_name()
            if c is not A.FrancoisGarrison else 'francois_garrison'
            for c in concrete}
    assert want <= have, f"models with no absorption_* function: {want - have}"


#: Minimal constructor arguments for the models the symmetry test builds.
_SAMPLE = {'temperature_c': 10.0, 'salinity_psu': 35.0, 'pH': 8.0,
           'z_bar_m': 100.0, 'layers': [(20.0, 80.0, 1500.0, 4.0, 0.02)],
           'value_dB_per_wavelength': 1e-4}


def test_the_two_formula_families_are_reached_the_same_way():
    """Sound speed and absorption are the same shape — a family of named
    equations, a registry, a carrier — so they must import the same way.

    They did not. `uacpy.absorption_thorp` resolved while
    `uacpy.sound_speed_teos10` did not, because the absorption functions were
    exported beside `Thorp`/`FrancoisGarrison` and the sound-speed ones were
    left under `uacpy.acoustics`. Same shape, two import paths, and nothing
    noticed until someone asked where a file was.

    `density` is deliberately NOT promoted: a bare `uacpy.density` would be
    seawater density sitting next to `BoundaryProperties(density=)`, which is
    the seabed's.
    """
    import uacpy
    from uacpy.core.acoustics.seawater import SOUND_SPEED_FORMULAS
    import uacpy.core.absorption as absorption_module

    speeds = {f'sound_speed_{name}' for name in SOUND_SPEED_FORMULAS}
    absorptions = {n for n in dir(absorption_module)
                   if n.startswith('absorption_')
                   and callable(getattr(absorption_module, n))}

    for family, names in (('sound speed', speeds), ('absorption', absorptions)):
        missing = {n for n in names if not hasattr(uacpy, n)}
        assert not missing, (
            f'{family}: {sorted(missing)} is not reachable as uacpy.<name>, '
            f'while the other family is')
        assert names <= set(uacpy.__all__), (
            f'{family}: reachable but absent from uacpy.__all__')

    assert not hasattr(uacpy, 'density'), (
        'uacpy.density would collide in meaning with '
        'BoundaryProperties(density=), which is the seabed"s')
