"""Tests for the core uacpy carriers and result types.

``Environment``, ``Source``, ``Receiver`` and ``Bathymetry`` on the input
side; ``Field``, ``ResultStack``, ``PhaseReference`` and
``ReflectionCoefficient`` on the output side; plus the enums and unit
conversions in ``uacpy.core.constants`` that both sides share.

A recurring subject is the label query. ``at()`` and its siblings pick a
node with ``argmin(|axis - label|)``, which ranks nothing when every
distance is NaN and hands back index 0 — a real node, so the caller sees a
plausible answer to an unanswerable question. Those guards are pinned on
both sides: the label that must be refused, and the legitimate one next to
it that must still work.
"""

import re
import warnings

import pytest
import numpy as np
import uacpy
from uacpy.core.absorption import convert_attenuation_units
from uacpy.core.altimetry import Altimetry
from uacpy.core.bathymetry import Bathymetry
from uacpy.core.bottom import Bottom, BoundaryProperties
from uacpy.core.constants import (
    AttenuationUnits, BoundaryType, SBP_ANGLE_RESOLUTION_DEG,
)
from uacpy.core.environment import Environment
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import (Arrivals, Field, PhaseReference,
                                ReflectionCoefficient)
from uacpy.core.results._base import Result
from uacpy.core.results.field import ResultStack
from uacpy.core.source import Source
from uacpy.core.ssp import SoundSpeedProfile
from uacpy.core.surface import Surface


class TestEnvironment:
    """Tests for Environment class."""

    def test_create_simple_environment(self, simple_env):
        """Test creating a simple isovelocity environment."""
        assert simple_env.name == "Test Environment"
        assert simple_env.depth == 100.0
        assert float(simple_env.ssp.data[0, 0]) == 1500.0
        assert simple_env.ssp.shape == 'isovelocity'
        assert not simple_env.is_range_dependent

    def test_create_parabolic_ssp_environment(self, parabolic_ssp_env):
        """Construction of a 100-m parabolic-SSP env."""
        assert parabolic_ssp_env.name == "Parabolic SSP"
        assert parabolic_ssp_env.depth == 100.0
        assert parabolic_ssp_env.ssp.n_depths == 21
        assert parabolic_ssp_env.ssp.shape == 'measured'

    def test_create_munk_environment(self, munk_env):
        """Construction of a deep-water Munk env using from_munk()."""
        assert munk_env.name == "Munk Profile"
        assert munk_env.depth == 5000.0
        assert munk_env.ssp.shape == 'munk'

    def test_range_dependent_environment(self, range_dependent_env):
        """Test range-dependent environment."""
        assert range_dependent_env.is_range_dependent
        assert range_dependent_env.bathymetry.n_ranges == 11
        assert range_dependent_env.bathymetry.depths[0] == 80.0
        assert range_dependent_env.bathymetry.depths[-1] == 120.0

    def test_max_range(self, simple_env, range_dependent_env):
        """max_range is the range extent (0 when range-independent), symmetric
        with depth and matching the bathymetry range axis."""
        assert simple_env.max_range == 0.0
        assert range_dependent_env.max_range == pytest.approx(
            float(range_dependent_env.bathymetry.ranges.max()))

    def test_max_range_covers_single_node_ranged_carriers(self):
        """A carrier whose ranged axis holds one node still marks a range
        coordinate, even though it is not range-*dependent* (that is a
        more-than-one-node test); max_range must include it."""
        from uacpy.core.environment import (
            Bottom, SeabedColumn, BoundaryProperties, SoundSpeedProfile,
            Surface,
        )
        surf = Surface(properties=[BoundaryProperties(acoustic_type='vacuum')],
                       ranges=[5000.0])
        assert uacpy.Environment(bathymetry=100.0,
                                 surface=surf).max_range == 5000.0
        ssp = SoundSpeedProfile(depths=[0.0, 100.0],
                                data=[[1500.0], [1500.0]], ranges=[7000.0])
        assert uacpy.Environment(bathymetry=100.0,
                                 ssp=ssp).max_range == 7000.0
        bot = Bottom(columns=[SeabedColumn(
            layers=[], halfspace=BoundaryProperties(sound_speed=1700.0))],
            ranges=[9000.0])
        assert uacpy.Environment(bathymetry=100.0,
                                 bottom=bot).max_range == 9000.0

    def test_ssp_pairs_shape(self, simple_env, parabolic_ssp_env):
        """SSP pairs view always has shape (N, 2)."""
        assert simple_env.ssp.to_pairs().shape[1] == 2
        assert parabolic_ssp_env.ssp.to_pairs().shape[1] == 2

    def test_get_representative_depth(self, range_dependent_env):
        """Median depth of the fixture's 11-node 80..120 m linspace is its
        middle node — exactly 100.0."""
        assert range_dependent_env.get_representative_depth('median') == 100.0

    def test_depth_is_read_only(self, simple_env):
        """``env.depth`` is derived from bathymetry (a getter-only property);
        assigning to it raises rather than silently shadowing the
        bathymetry."""
        with pytest.raises(AttributeError):
            simple_env.depth = 200.0

    def test_invalid_depth(self):
        """Test that negative depth raises error."""
        with pytest.raises(ConfigurationError):
            uacpy.Environment(name="Test", bathymetry=-10, ssp=1500)

    def test_bathymetry_rejects_negative_range(self):
        """Bathymetry ranges are measured from the source; they cannot
        be negative."""
        with pytest.raises(ConfigurationError, match="ranges must be non-negative"):
            uacpy.Environment(
                name="Test",
                bathymetry=[[-100.0, 80.0], [5000.0, 90.0]],
                ssp=1500,
            )


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
        a = self._stack().alpha_db_per_m(100.0, [5.0, 10.0, 15.0])
        assert a[1] == pytest.approx(a[0] + a[2])
        # At f = f0 each layer peaks at a0·Q² = 10·25 = 250 dB/km, so the
        # shared depth carries 500 dB/km (the AttenMod.f90 sum).
        assert a[0] * 1000.0 == pytest.approx(250.0)
        assert a[1] * 1000.0 == pytest.approx(500.0)

    def test_outer_edges_are_inclusive(self):
        a = self._stack().alpha_db_per_m(100.0, [0.0, 20.0, 25.0])
        assert a[0] == pytest.approx(a[1])
        assert a[0] > 0.0
        assert a[2] == 0.0


class TestAbsorptionFormulaOutputShapes:
    """The bare formulas and the unit converter shape their output after
    their input: 0-d for a scalar, unchanged for an array — a 1-element
    array stays 1-D and indexable."""

    def test_one_element_array_stays_indexable(self):
        from uacpy.core.absorption import (
            thorp_db_per_km, francois_garrison_db_per_km,
            convert_attenuation_units)
        assert thorp_db_per_km(np.array([100.0])).shape == (1,)
        assert float(thorp_db_per_km(np.array([100.0]))[0]) > 0
        assert francois_garrison_db_per_km(np.array([100.0])).shape == (1,)
        assert convert_attenuation_units(
            np.array([1.0]), 100.0, 'dB/km', 'dB/m').shape == (1,)

    def test_scalar_input_yields_0d(self):
        from uacpy.core.absorption import (
            thorp_db_per_km, francois_garrison_db_per_km,
            convert_attenuation_units)
        assert np.ndim(thorp_db_per_km(100.0)) == 0
        assert np.ndim(francois_garrison_db_per_km(100.0)) == 0
        assert np.ndim(
            convert_attenuation_units(1.0, 100.0, 'dB/km', 'dB/m')) == 0

    def test_n_element_array_keeps_shape(self):
        from uacpy.core.absorption import thorp_db_per_km
        assert thorp_db_per_km(np.array([100.0, 200.0, 300.0])).shape == (3,)


class TestConvertAttenuationUnitsFromQ:
    """Q sits in the denominator of the from-'Q' path, so a non-positive
    quality factor raises instead of dividing to inf."""

    @pytest.mark.parametrize("bad_q", [0.0, -5.0])
    def test_non_positive_q_raises(self, bad_q):
        from uacpy.core.absorption import convert_attenuation_units
        with pytest.raises(ConfigurationError, match="from_unit='Q'"):
            convert_attenuation_units(bad_q, 100.0, 'Q', 'dB/m')

    def test_positive_q_round_trips(self):
        from uacpy.core.absorption import convert_attenuation_units
        q = 50.0
        db_m = convert_attenuation_units(q, 100.0, 'Q', 'dB/m')
        back = convert_attenuation_units(float(db_m), 100.0, 'dB/m', 'Q')
        assert float(back) == pytest.approx(q)


class TestConvertAttenuationUnitsToQ:
    """The mirror direction is deliberately *not* symmetric. Q = 0 is not the
    limit of anything representable (it is α → ∞) and raises; α = 0 is the
    lossless limit and Q → ∞ is its exact value, so it is answered with
    ``inf`` — quietly, without numpy's bare divide-by-zero RuntimeWarning —
    and converts straight back to zero."""

    def test_zero_attenuation_gives_the_lossless_limit(self):
        from uacpy.core.absorption import convert_attenuation_units
        with warnings.catch_warnings():
            warnings.simplefilter('error')      # no bare RuntimeWarning
            q = convert_attenuation_units(0.0, 100.0, 'dB/m', 'Q',
                                          sound_speed=1500.0)
        assert np.isinf(float(q)) and float(q) > 0

    def test_the_lossless_limit_converts_back_to_zero(self):
        from uacpy.core.absorption import convert_attenuation_units
        q = convert_attenuation_units(0.0, 100.0, 'dB/m', 'Q',
                                      sound_speed=1500.0)
        back = convert_attenuation_units(float(q), 100.0, 'Q', 'dB/m',
                                         sound_speed=1500.0)
        assert float(back) == pytest.approx(0.0)

    def test_an_array_keeps_its_finite_entries(self):
        from uacpy.core.absorption import convert_attenuation_units
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            out = convert_attenuation_units(np.array([0.0, 0.5]), 100.0,
                                            'dB/m', 'Q', sound_speed=1500.0)
        assert np.isinf(out[0])
        assert out[1] == pytest.approx(3.63833694, rel=1e-6)


class TestAbsorptionFrequencyGuardIsShared:
    """``α(f, z)`` has no value at or below zero for any of the four models,
    so the guard sits on the public ``alpha_db_per_m`` ahead of the dispatch
    rather than in each subclass. Thorp and Francois-Garrison are polynomials
    that had no guard and kept evaluating: both returned a *positive*
    attenuation at f = 0 and for a negative frequency."""

    def _models(self):
        from uacpy.core.absorption import (
            Thorp, FrancoisGarrison, Biological, ConstantAbsorption)
        return [
            Thorp(),
            FrancoisGarrison(temperature_c=15.0, salinity_psu=35.0,
                             pH=8.1, z_bar_m=50.0),
            Biological(layers=[(0.0, 100.0, 100.0, 10.0, 1.0)]),
            ConstantAbsorption(value_db_per_wavelength=0.5),
        ]

    @pytest.mark.parametrize('freq', [0.0, -100.0, float('nan')])
    def test_every_model_rejects_a_non_positive_frequency(self, freq):
        z = np.array([0.0, 50.0])
        for model in self._models():
            with pytest.raises(ConfigurationError,
                               match='frequency must be > 0'):
                model.alpha_db_per_m(freq, z)

    def test_the_message_names_the_model(self):
        from uacpy.core.absorption import Thorp
        with pytest.raises(ConfigurationError, match='Thorp.alpha_db_per_m'):
            Thorp().alpha_db_per_m(0.0, np.array([0.0]))

    def test_a_positive_frequency_evaluates(self):
        z = np.array([0.0, 50.0])
        for model in self._models():
            a = np.asarray(model.alpha_db_per_m(1000.0, z))
            assert a.shape == z.shape
            assert np.isfinite(a).all() and (a >= 0).all()

    def test_sub_hertz_is_legal(self):
        """Only f <= 0 has no wavelength; infrasonic frequencies convert."""
        from uacpy.core.absorption import ConstantAbsorption
        out = ConstantAbsorption(value_db_per_wavelength=0.5).alpha_db_per_m(
            0.5, np.array([10.0]))
        assert np.isfinite(out).all()


class TestConstantAbsorptionCeiling:
    """ConstantAbsorption enforces the same attenuation ceiling as the seabed
    carriers (above it every AT solver aborts in AttenMod.f90's CRCI)."""

    def test_above_ceiling_raises(self):
        from uacpy.core.absorption import ConstantAbsorption
        from uacpy.core.constants import MAX_ATTENUATION_DB_PER_WAVELENGTH
        with pytest.raises(ConfigurationError, match="dB/wavelength exceeds"):
            ConstantAbsorption(
                value_db_per_wavelength=MAX_ATTENUATION_DB_PER_WAVELENGTH + 1.0)

    def test_at_ceiling_constructs(self):
        from uacpy.core.absorption import ConstantAbsorption
        from uacpy.core.constants import MAX_ATTENUATION_DB_PER_WAVELENGTH
        c = ConstantAbsorption(
            value_db_per_wavelength=MAX_ATTENUATION_DB_PER_WAVELENGTH)
        assert c.value_db_per_wavelength == pytest.approx(
            MAX_ATTENUATION_DB_PER_WAVELENGTH)


class TestGenerateSeaSurfaceSynthesis:
    """The realisation is the inverse DFT of the Pierson-Moskowitz
    random-phase spectrum, so it must equal the direct cosine sum it is
    defined by and stay seed-reproducible."""

    def test_matches_the_direct_cosine_sum(self):
        from uacpy.core.ssp import generate_sea_surface
        n, max_range, wind, seed = 257, 4000.0, 10.0, 12345
        out = generate_sea_surface(max_range, wind, n, seed)
        ranges, surface = out[:, 0], out[:, 1]
        dx = ranges[1] - ranges[0]
        dk = 1.0 / (n * dx)
        k = np.arange(1, n // 2 + 1) * dk
        g = 9.81
        omega = np.sqrt(g * 2 * np.pi * k)
        omega_p = g / wind
        S_omega = (8.1e-3 * g ** 2 / omega ** 5) * np.exp(
            -0.74 * (omega_p / omega) ** 4)
        S_k = S_omega * (np.pi * g / omega)
        amplitude = np.sqrt(2 * S_k * dk)
        phase = np.random.default_rng(seed).uniform(0, 2 * np.pi, len(k))
        direct = (amplitude[None, :]
                  * np.cos(2 * np.pi * ranges[:, None] * k[None, :]
                           + phase[None, :])).sum(axis=1)
        assert float(np.max(np.abs(surface - direct))) < 1e-9

    def test_seeded_runs_are_reproducible(self):
        from uacpy.core.ssp import generate_sea_surface
        a = generate_sea_surface(2000.0, 10.0, 128, 7)
        b = generate_sea_surface(2000.0, 10.0, 128, 7)
        assert np.array_equal(a, b)


def _public_named_surfaces():
    """Every name a caller of ``uacpy``, ``uacpy.acoustic_signal``,
    ``uacpy.comms`` or ``uacpy.sonar`` types: each export's call parameters
    (``mod.callable(param)``) and each exported class's public attributes
    (``mod.Class.attr``).

    Both halves, because a unit suffix has to hold wherever the name is
    read: ``AmbiguityResult`` is a NamedTuple, so ``delays_s`` is a keyword
    at construction *and* an attribute afterwards. Sweeping the surfaces
    together is what makes a suffix rule checkable everywhere at once rather
    than at the one site an audit happened to open."""
    import inspect
    import uacpy.acoustic_signal
    import uacpy.comms
    import uacpy.sonar

    sites = set()
    for module in (uacpy, uacpy.acoustic_signal, uacpy.comms, uacpy.sonar):
        for name in getattr(module, '__all__', ()):
            obj = getattr(module, name, None)
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            try:
                signature = inspect.signature(obj)
            except (TypeError, ValueError):
                signature = None
            if signature is not None:
                for param in signature.parameters:
                    sites.add(f"{module.__name__}.{name}({param})")
            if inspect.isclass(obj):
                for attr in vars(obj):
                    if not attr.startswith('_'):
                        sites.add(f"{module.__name__}.{name}.{attr}")
    return sites


def _sites_with_suffix(suffix):
    """The public surfaces whose *name* ends in ``suffix`` — the trailing
    ``)`` of a parameter site is stripped before matching."""
    return sorted(s for s in _public_named_surfaces()
                  if s.rstrip(')').endswith(suffix))


class TestTheSecondsSuffixIsNotSpentOnMetresPerSecond:
    """``_s`` is this package's suffix for seconds — eleven public sites use
    it that way (``delays_s``, ``pulse_length_s``, ``integration_time_s``, …)
    — so a public ``_ms`` reads as milliseconds, which is what ``_ms``
    already means at every private site that has one (``t_ms``,
    ``delays_ms``, ``time_ms`` are all milliseconds). Metres per second is
    spelled ``_mps``.
    """

    def test_the_sea_surface_generator_takes_wind_speed_mps(self):
        import inspect
        params = inspect.signature(uacpy.generate_sea_surface).parameters
        assert 'wind_speed_mps' in params
        assert params['wind_speed_mps'].default == 10.0

    def test_the_wind_speed_mps_keyword_sets_the_wave_height(self):
        # The name is pinned on the live signature above; this pins that it
        # is the wind speed, so the gate cannot be satisfied by a parameter
        # that merely spells itself right.
        calm = uacpy.generate_sea_surface(
            2000.0, wind_speed_mps=5.0, n_points=256, seed=3)
        blow = uacpy.generate_sea_surface(
            2000.0, wind_speed_mps=15.0, n_points=256, seed=3)
        # Pierson-Moskowitz: Hs = 0.021*U^2, so 15 m/s is 9x the 5 m/s sea.
        assert float(np.std(blow[:, 1])) > 5.0 * float(np.std(calm[:, 1]))

    def test_the_seconds_spelling_of_the_wind_speed_keyword_is_not_accepted(self):
        # The other side of the same boundary: one spelling reaches the
        # generator and the other is a TypeError, so the two cannot both be
        # live at once.
        with pytest.raises(TypeError, match='wind_speed_ms'):
            uacpy.generate_sea_surface(2000.0, wind_speed_ms=5.0, n_points=64)

    def test_the_seconds_suffix_is_already_spoken_for_on_eleven_surfaces(self):
        """The premise the ``_mps`` spelling rests on, measured rather than
        asserted. Ten call parameters plus one attribute — the eleventh is
        ``AmbiguityResult.delays_s``, readable as well as passable, which is
        why the sweep covers attributes too. If this count moves the
        convention has changed and the rule below has to be restated."""
        seconds = [s for s in _sites_with_suffix('_s')
                   if not s.rstrip(')').endswith('_ms')]
        assert len(seconds) == 11, seconds
        assert 'uacpy.acoustic_signal.AmbiguityResult.delays_s' in seconds

    def test_no_public_surface_spells_metres_per_second_as_ms(self):
        offenders = _sites_with_suffix('_ms')
        assert not offenders, (
            "public name(s) ending in `_ms`, which reads as milliseconds "
            "next to the `_s`-for-seconds sites:\n" + "\n".join(offenders))

    def test_the_mps_spelling_is_the_one_the_sweep_finds(self):
        """The far side of the previous gate: silence there must mean the
        sweep looked and found nothing, not that it sees no speed at all.

        ``uacpy.comms.doppler_from_speed`` spells both of its speeds
        ``_mps`` already, so the sea-surface generator joins a convention
        rather than starting one."""
        assert _sites_with_suffix('_mps') == [
            'uacpy.comms.doppler_from_speed(sound_speed_mps)',
            'uacpy.comms.doppler_from_speed(speed_mps)',
            'uacpy.generate_sea_surface(wind_speed_mps)',
        ]


class TestSource:
    """Tests for Source class."""

    def test_create_source(self, source):
        """Test creating a source."""
        assert source.depths[0] == 50.0
        assert source.frequencies[0] == 100.0

    def test_source_array_conversion(self):
        """Test that single values are converted to arrays."""
        source = uacpy.Source(depths=30.0, frequencies=200.0)
        assert isinstance(source.depths, np.ndarray)
        assert isinstance(source.frequencies, np.ndarray)
        assert len(source.depths) == 1
        assert len(source.frequencies) == 1

    def test_multiple_sources(self):
        """Test multiple source depths."""
        source = uacpy.Source(depths=[10.0, 20.0, 30.0], frequencies=100.0)
        assert len(source.depths) == 3
        assert np.allclose(source.depths, [10, 20, 30])

    def test_multiple_frequencies(self):
        """Test multiple frequencies."""
        source = uacpy.Source(depths=50.0, frequencies=[50.0, 100.0, 200.0])
        assert len(source.frequencies) == 3

    def test_source_depths_must_be_strictly_increasing(self):
        """Multi-element source depths must be sorted (matches Receiver), so
        output rows indexed by source depth stay unambiguous across models."""
        with pytest.raises(ConfigurationError, match="strictly increasing"):
            uacpy.Source(depths=[30.0, 10.0, 20.0], frequencies=100.0)


@pytest.mark.parametrize("ctor,kwargs", [
    # Source / Receiver reject NaN or inf in any
    # ``depths``/``frequencies``/``ranges`` array.
    (uacpy.Source, dict(depths=[10, np.nan], frequencies=100)),
    (uacpy.Source, dict(depths=10, frequencies=[100, np.nan])),
    (uacpy.Receiver, dict(depths=[10, 20], ranges=[100, np.nan])),
    (uacpy.Receiver, dict(depths=[np.nan], ranges=[100])),
    (uacpy.Source, dict(depths=[10, np.inf], frequencies=100)),
    (uacpy.Receiver, dict(depths=[10], ranges=[np.inf])),
])
def test_source_receiver_reject_non_finite(ctor, kwargs):
    """Source and Receiver reject NaN / inf at construction so
    non-finite values cannot leak into env-file writers."""
    with pytest.raises(ConfigurationError, match="finite"):
        ctor(**kwargs)


class TestReceiver:
    """Tests for Receiver class."""

    def test_create_receiver_grid(self, receiver_grid):
        """Test creating receiver grid."""
        assert len(receiver_grid.depths) == 9
        assert len(receiver_grid.ranges) == 11
        assert receiver_grid.receiver_type == 'grid'
        assert receiver_grid.depth_min == 10.0
        assert receiver_grid.depth_max == 90.0
        assert receiver_grid.range_min == 100.0
        assert receiver_grid.range_max == 5000.0

    def test_small_receiver_grid(self, receiver_small):
        """Test small receiver grid."""
        assert len(receiver_small.depths) == 3
        assert len(receiver_small.ranges) == 3

    def test_receiver_line_array(self):
        """Test line array receiver."""
        receiver = uacpy.Receiver(
            depths=[50.0],
            ranges=np.linspace(1000, 10000, 100)
        )
        assert len(receiver.depths) == 1
        assert len(receiver.ranges) == 100

    def test_receiver_type_rejects_unknown(self):
        """Unknown ``receiver_type`` strings raise
        :class:`ConfigurationError`, mirroring the validation on
        ``Source.source_type``."""
        with pytest.raises(ConfigurationError, match="receiver_type"):
            uacpy.Receiver(depths=50, ranges=1000, receiver_type='gird')

    def test_receiver_type_accepts_grid_rejects_line(self):
        rx = uacpy.Receiver(depths=50, ranges=1000, receiver_type='grid')
        assert rx.receiver_type == 'grid'
        with pytest.raises(ConfigurationError, match="receiver_type='line'"):
            uacpy.Receiver(depths=50, ranges=1000, receiver_type='line')

    def test_omitted_ranges_default_to_source_point_with_warning(self):
        """``Receiver(depths=50)`` defaults ranges to a single point at 0 m
        (the source location) and warns, because r=0 is singular for
        TL/pressure runs."""
        with pytest.warns(UserWarning, match="ranges not given"):
            rx = uacpy.Receiver(depths=50.0)
        np.testing.assert_array_equal(rx.ranges, [0.0])
        np.testing.assert_array_equal(rx.depths, [50.0])


class TestField:
    """Tests for the unified :class:`~uacpy.Field` container."""

    @staticmethod
    def _tl_field(data, ranges, depths, **kw):
        return Field(
            data=data,
            coords={'depth': depths, 'range': ranges},
            model=kw.pop('model', 'Test'),
            frequencies=kw.pop('frequencies', 100.0),
            **kw,
        )

    def test_create_tl_field(self):
        from uacpy.core.results import Field
        data = np.random.rand(10, 20) * 50 + 40  # dB
        ranges = np.linspace(100, 5000, 20)
        depths = np.linspace(10, 90, 10)
        field = self._tl_field(data, ranges, depths)
        assert isinstance(field, Field)
        assert field.shape == (10, 20)
        assert field.n_ranges == 20
        assert field.n_depths == 10
        assert not field.is_complex

    def test_to_dict_roundtrip_preserves_model_source(self):
        from uacpy.models.sources import model_source
        src = model_source('acoustics_toolbox')
        field = self._tl_field(
            np.zeros((2, 2)), np.array([100.0, 200.0]),
            np.array([10.0, 20.0]), model_source=src)
        rt = Field.from_dict(field.to_dict())
        assert rt.model_source is src

    # data[d, r] = 10*d + r in the three tests below, so each value names
    # its own cell and a depth/range transpose (data[r, d]) is caught
    # exactly, which the previous 44-55 band assertions admitted.

    def test_at_point_returns_nearest_cell_value(self):
        data = np.arange(100).reshape(10, 10).astype(float)
        ranges = np.linspace(0, 9000, 10)   # 1000 m spacing
        depths = np.linspace(0, 90, 10)     # 10 m spacing
        field = self._tl_field(data, ranges, depths)
        # Off-centre query: range=4200 → index 4, depth=68 → index 7,
        # so the nearest cell is data[7, 4] = 74 (a transpose reads 47).
        assert float(field.at(range=4200.0, depth=68.0).db) == 74.0

    def test_at_range_returns_nearest_cell_values(self):
        data = np.arange(100).reshape(10, 10).astype(float)
        ranges = np.linspace(0, 9000, 10)
        depths = np.linspace(0, 90, 10)
        field = self._tl_field(data, ranges, depths)
        values = field.at(range=4200.0).db
        # Nearest range sample is index 4 (4000 m): the depth column
        # 10*d + 4. A transposed field would return 40..49 instead.
        np.testing.assert_array_equal(values, np.arange(10) * 10.0 + 4.0)

    def test_at_depth_returns_nearest_cell_values(self):
        data = np.arange(100).reshape(10, 10).astype(float)
        ranges = np.linspace(0, 9000, 10)
        depths = np.linspace(0, 90, 10)
        field = self._tl_field(data, ranges, depths)
        values = field.at(depth=68.0).db
        # Nearest depth sample is index 7 (70 m): the range row 70..79.
        # A transposed field would return 8, 18, ..., 98 instead.
        np.testing.assert_array_equal(values, np.arange(10) + 70.0)

    def test_field_deepcopy(self):
        import copy as _copy
        data = np.random.rand(10, 20)
        ranges = np.linspace(100, 5000, 20)
        depths = np.linspace(10, 90, 10)
        field = self._tl_field(data, ranges, depths)
        field_copy = _copy.deepcopy(field)
        assert type(field_copy) is type(field)
        assert np.array_equal(field_copy.data, field.data)
        assert field_copy is not field
        assert field_copy.data is not field.data

    def test_field_repr(self):
        data = np.random.rand(10, 20)
        ranges = np.linspace(100, 5000, 20)
        depths = np.linspace(10, 90, 10)
        field = self._tl_field(data, ranges, depths)
        repr_str = repr(field)
        assert 'Field' in repr_str
        assert field.shape == (10, 20)

    def test_resample_to_is_keyword_only_and_depth_first(self):
        """The two axis vectors are interchangeable in type and only
        distinguishable by name, so a positional call that swapped them would
        silently resample onto a transposed, mostly-NaN grid instead of
        raising. Keyword-only makes that unrepresentable."""
        ranges = np.linspace(0.0, 1000.0, 5)
        depths = np.linspace(0.0, 100.0, 3)
        # value == depth, so a transposed result is obvious in the numbers.
        data = np.repeat(depths[:, None], ranges.size, axis=1)
        field = self._tl_field(data, ranges, depths)

        with pytest.raises(TypeError):
            field.resample_to(ranges, depths)

        out = field.resample_to(depths=[25.0, 75.0], ranges=[250.0, 750.0])
        assert list(out.coords) == ['depth', 'range']
        assert out.shape == (2, 2)
        np.testing.assert_allclose(out.data, [[25.0, 25.0], [75.0, 75.0]])

    @staticmethod
    def _coherent_field(dr, dz=1.0, f0=200.0, c=1500.0):
        """Complex pressure with carrier e^{ikr}, grid spaced ``dz`` x ``dr``."""
        ranges = np.arange(1000.0, 1500.0 + dr, dr)
        depths = np.arange(40.0, 60.0 + dz, dz)
        k = 2.0 * np.pi * f0 / c
        data = np.exp(1j * k * ranges)[None, :] / ranges[None, :]
        return Field(
            data=np.repeat(data, depths.size, axis=0),
            coords={'depth': depths, 'range': ranges},
            model='Test', frequencies=f0,
        )

    @staticmethod
    def _midpoints(field):
        r = field.coords['range']
        z = field.coords['depth']
        return dict(depths=z[:-1] + np.diff(z) / 2, ranges=r[:-1] + np.diff(r) / 2)

    # Quarter wavelength at 200 Hz with c = 1500 m/s. Each axis is bracketed
    # against it independently: a guard that fires on two axes needs the
    # coarse and the resolved case *per axis*, or a regression re-opening the
    # depth half (measured +1.36 dB, silent) passes on the range cases alone.
    QUARTER = 1.875

    @pytest.mark.parametrize('dr,dz,named,silent_axis', [
        (2.0, 1.8, 'range', 'depth'),      # range just over, depth just under
        (1.8, 2.0, 'depth', 'range'),      # depth just over, range just under
    ])
    def test_resample_to_brackets_each_axis_independently(self, dr, dz, named,
                                                          silent_axis):
        """Each axis is checked against the quarter wavelength on its own.
        The pair brackets 1.875 m tightly from both sides *per axis*: the
        named one is coarse at 2.0 m, the other resolved at 1.8 m, so the
        message must name exactly one."""
        field = self._coherent_field(dr, dz)
        with pytest.warns(UserWarning, match=f'{named} samples are') as rec:
            field.resample_to(**self._midpoints(field))
        assert f'{silent_axis} samples are' not in str(rec[0].message)

    @pytest.mark.parametrize('dr,dz,named', [
        (25.0, 1.0, 'range'),        # range alone   — measured +2.6 dB
        (1.0, 5.0, 'depth'),         # depth alone   — measured +1.4 dB, was silent
        (25.0, 5.0, 'depth'),        # both          — measured +4.9 dB
    ])
    def test_resample_to_warns_and_names_the_coarse_axis(self, dr, dz, named):
        """Interpolating a coherent field across opposite-phase lobes biases
        the level upward. Both axes matter and their biases compound, and
        ``resample_to`` always interpolates both — so a guard that reported
        only the range axis stayed silent on a pure depth-axis error and
        misattributed the mixed one."""
        field = self._coherent_field(dr, dz)
        with pytest.warns(UserWarning, match=f'{named} samples are'):
            field.resample_to(**self._midpoints(field))

    @pytest.mark.parametrize('dr,dz', [
        (0.5, 0.5), (1.0, 1.0), (1.8, 1.8),   # both resolved
        (1.8, 0.5), (0.5, 1.8),               # each axis at the bound in turn
    ])
    def test_resample_to_is_silent_when_both_axes_are_resolved(self, dr, dz):
        # The discriminating half: the guard must not fire below the quarter
        # wavelength, or it would be noise on every well-sampled field.
        assert max(dr, dz) < self.QUARTER
        field = self._coherent_field(dr, dz)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            field.resample_to(**self._midpoints(field))

    def test_resample_to_admits_it_cannot_check_a_field_with_no_frequency(self):
        """A wrapped phase step cannot stand in for the frequency: a grid
        spaced a whole wavelength aliases to 0.000 rad and reads as perfectly
        sampled while being maximally undersampled (it misses 47.8 % of coarse
        grids overall). So a Field with no f0 is reported as unverifiable
        rather than silently passing — a silence that reads as a pass is worse
        than an admission."""
        field = self._coherent_field(7.5, dz=1.0)         # dr == one wavelength
        blind = Field(data=field.data, coords=dict(field.coords),
                      model='Test', frequencies=None)
        with pytest.warns(UserWarning, match='carries no frequency'):
            blind.resample_to(**self._midpoints(blind))

    def test_resample_to_never_warns_for_a_real_field(self):
        # A TL field carries no carrier, so it interpolates freely however
        # coarse the grid — keying the guard on dtype rather than on grid
        # spacing alone is what keeps this quiet.
        ranges = np.arange(1000.0, 5000.0, 250.0)
        depths = np.array([50.0, 100.0])
        field = self._tl_field(np.zeros((depths.size, ranges.size)), ranges, depths,
                               frequencies=200.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            field.resample_to(depths=[50.0], ranges=ranges[:-1] + 125.0)


class TestFieldValueAccessorsAreWriteGuarded:
    """``Field`` copies on ingest, so no accessor may hand back a writable
    alias of ``data``: ``p = field.p; p *= k`` would otherwise corrupt the
    stored result. Real ``.db`` is the common path (RAM / OAST /
    Bellhop-incoherent all return real dB)."""

    @staticmethod
    def _real():
        return Field(data=np.array([[60.0, 70.0], [80.0, 90.0]]),
                     coords={'depth': [0.0, 1.0], 'range': [0.0, 1.0]})

    @staticmethod
    def _complex():
        return Field(data=np.array([[1 + 1j, 2 + 0j], [0 + 3j, 4 - 1j]]),
                     coords={'depth': [0.0, 1.0], 'range': [0.0, 1.0]})

    def test_real_tl_is_read_only(self):
        f = self._real()
        tl = f.db
        assert not tl.flags.writeable
        with pytest.raises(ValueError):
            tl[0, 0] = -999.0
        np.testing.assert_array_equal(f.data, [[60.0, 70.0], [80.0, 90.0]])

    def test_real_tl_reads_the_stored_dB_values(self):
        f = self._real()
        np.testing.assert_array_equal(np.asarray(f.db), f.data)

    def test_scalar_tl_is_read_only_and_castable(self):
        f = self._real().at(depth=0.0, range=1.0)
        assert not f.db.flags.writeable
        assert float(f.db) == 70.0

    def test_complex_tl_is_a_fresh_array(self):
        f = self._complex()
        tl = f.db
        assert not np.shares_memory(tl, f.data)
        tl[0, 0] = -999.0                       # derived array: safe to write
        assert f.data[0, 0] == 1 + 1j

    def test_p_is_read_only(self):
        f = self._complex()
        with pytest.raises(ValueError):
            f.p[0, 0] = 5.0
        assert f.data[0, 0] == 1 + 1j

    def test_magnitude_and_phase_are_fresh_arrays(self):
        f = self._complex()
        mag, ph = f.magnitude, f.phase
        assert not np.shares_memory(mag, f.data)
        assert not np.shares_memory(ph, f.data)
        mag[0, 0] = -1.0
        ph[0, 0] = -1.0
        assert f.data[0, 0] == 1 + 1j
        np.testing.assert_allclose(f.magnitude, np.abs(f.data))
        np.testing.assert_allclose(f.phase, np.angle(f.data))

    def test_data_is_the_writeable_buffer_the_read_only_views_share(self):
        """The other half of the guard, stated because ``.p``'s read-only flag
        reads like a promise about the field: it is on the *view*, and
        :attr:`data` is the same memory, writeable."""
        f = self._complex()
        assert f.data.flags.writeable
        assert np.shares_memory(f.data, f.p)
        f.data *= 1e6
        assert f.p[0, 0] == (1 + 1j) * 1e6

    def test_the_docs_say_data_is_writeable_and_shares_the_buffer(self):
        # A caller who has internalised ``.p``'s read-only guarantee has no
        # way to learn from the code that ``field.data *= k`` defeats it.
        prose = (Field.__doc__ or '') + '\n' + (Field.p.__doc__ or '')
        assert 'writeable' in prose
        assert 'read-only' in prose
        assert 'same buffer' in prose or 'same memory' in prose

    @pytest.mark.parametrize('dtype', ['float64', 'float32', 'int64'])
    def test_real_db_aliases_data_in_every_real_dtype(self, dtype):
        """The real branch hands back ``data`` itself, so which engine
        produced the field decides nothing about the contract.

        float32 is the case a user meets: ``to_db()`` of a ``.shd``-backed
        complex64 Field is float32. float64 is what an in-memory Field and
        the RAM/OAST readers carry. Both alias, and the array taken before a
        write reads the value written after it."""
        f = Field(data=np.ones((2, 2), dtype=dtype),
                  coords={'depth': [0.0, 1.0], 'range': [0.0, 1.0]},
                  metadata={'unit': 'dB'})
        view = f.db
        assert np.shares_memory(view, f.data)
        assert not view.flags.writeable
        f.data[0, 0] = 999
        assert view[0, 0] == 999

    @pytest.mark.parametrize('dtype', ['float64', 'float32', 'int64'])
    def test_real_db_carries_the_fields_own_dtype(self, dtype):
        # The accepted cost of aliasing: no upcast, so the caller reads the
        # stored precision and asks for float64 explicitly if it needs it.
        f = Field(data=np.ones((2, 2), dtype=dtype),
                  coords={'depth': [0.0, 1.0], 'range': [0.0, 1.0]},
                  metadata={'unit': 'dB'})
        assert f.db.dtype == np.dtype(dtype)
        assert np.asarray(f.db, dtype=float).dtype == np.dtype('float64')

    def test_to_db_of_a_complex64_field_gives_a_float32_field_whose_db_aliases(self):
        # The in-package route to a non-float64 real field: read_shd_bin
        # returns complex64, so to_db() of it is float32.
        f = Field(data=np.ones((2, 2), dtype='complex64'),
                  coords={'depth': [0.0, 1.0], 'range': [0.0, 1.0]},
                  metadata={'unit': 'Pa'})
        real = f.to_db()
        assert real.data.dtype == np.dtype('float32')
        assert not real.is_complex
        assert np.shares_memory(real.db, real.data)
        assert real.db.dtype == np.dtype('float32')

    def test_the_docstring_states_in_prose_that_db_follows_the_fields_dtype(self):
        """The contract a caller reads before deciding whether to cast.

        Literal spans are stripped first: ``dtype`` also occurs inside the
        ``np.asarray(..., dtype=float)`` example, which shows how to opt out
        of the stored precision rather than stating what ``.db`` returns.
        Matching it there passes on a docstring that never makes the claim.
        """
        prose = re.sub(r'``[^`]*``', ' ',
                       ' '.join((Field.db.__doc__ or '').split()))
        assert 'dtype' in prose
        assert 'read-only view' in prose
        assert 'alias' in prose

    @pytest.mark.parametrize('dtype', ['complex128', 'complex64'])
    def test_complex_db_is_a_fresh_array_in_every_complex_dtype(self, dtype):
        """The other side of the same boundary: the complex branch computes
        ``-20·log10|data|``, so there is nothing of ``data`` to alias and the
        caller owns what it gets."""
        f = Field(data=np.full((2, 2), 1 + 1j, dtype=dtype),
                  coords={'depth': [0.0, 1.0], 'range': [0.0, 1.0]},
                  metadata={'unit': 'Pa'})
        tl = f.db
        assert not np.shares_memory(tl, f.data)
        tl[0, 0] = -999.0
        assert f.data[0, 0] == 1 + 1j

    def test_to_dict_does_not_alias_the_field(self):
        f = self._real()
        d = f.to_dict()
        d['data'][1, 1] = -1.0
        d['coords']['depth'][0] = 99.0
        assert f.data[1, 1] == 90.0
        assert f.coords['depth'][0] == 0.0


class TestFieldMaxComplexData:
    """max() ranks complex data by magnitude whatever unit the field is
    tagged with, so no float cast of complex values occurs."""

    def test_complex_db_tagged_field_uses_magnitude(self):
        f = Field(
            data=np.array([[1 + 1j, 3 + 4j]]),
            coords={'depth': np.array([1.0]),
                    'range': np.array([10.0, 20.0])},
            metadata={'kind': 'reverberation', 'unit': 'dB'},
            frequencies=np.array([100.0]))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m = f.max()
        assert complex(m.data) == 3 + 4j
        assert m.pinned['range'] == 20.0


class TestPublicReexports:
    """Public namespace contract."""

    def test_soundspeedprofile_at_top_level(self):
        from uacpy import SoundSpeedProfile
        assert SoundSpeedProfile is uacpy.core.environment.SoundSpeedProfile

    def test_environment_helpers_at_core(self):
        from uacpy.core import SoundSpeedProfile, generate_sea_surface
        assert SoundSpeedProfile is uacpy.core.environment.SoundSpeedProfile
        assert generate_sea_surface is uacpy.core.environment.generate_sea_surface

    def test_acoustic_signal_is_importable_submodule(self):
        import uacpy.acoustic_signal as sig
        assert sig is uacpy.acoustic_signal

    def test_signal_analysis_classes_reachable(self):
        sig = uacpy.acoustic_signal
        # Estimators/transforms are free functions; FRF is a class.
        for name in ('ppsd', 'psd', 'FRF', 'sel', 'fk_transform', 'spectrogram'):
            assert hasattr(sig, name), f"uacpy.acoustic_signal.{name} not reachable"
        assert 'psd' in sig.__all__
        assert 'FRF' in sig.__all__
        assert 'sel' in sig.__all__
        assert 'fk_transform' in sig.__all__

    def test_metrics_is_importable_submodule(self):
        import uacpy.metrics as m
        assert m is uacpy.metrics
        assert hasattr(m, 'tl_rmse')
        assert hasattr(m, 'tl_max_error')
        assert hasattr(m, 'tl_bias')


class TestModesFieldType:
    """Single canonical ``field_type`` declaration on Modes."""

    def test_modes_field_type_value(self):
        from uacpy.core.results import Modes
        assert Modes.field_type == "modes"


class TestModesComputePhaseSpeeds:
    """``Modes.compute_phase_speeds`` requires a frequency context;
    without one it raises :class:`ValueError`."""

    def _build_modes(self, *, frequencies):
        from uacpy.core.results import Modes
        depths = np.linspace(0, 100, 11)
        # Two trivial modes; numbers don't matter.
        k = np.array([0.4 + 0.0j, 0.3 + 0.0j])
        phi = np.zeros((len(depths), 2))
        return Modes(
            k=k, phi=phi, depths=depths,
            model='Test', frequencies=frequencies,
        )

    def test_phase_speeds_raises_without_frequency(self):
        modes = self._build_modes(frequencies=None)
        with pytest.raises(ConfigurationError, match='requires frequencies'):
            modes.compute_phase_speeds()

    def test_phase_speeds_with_frequency_is_omega_over_k(self):
        modes = self._build_modes(frequencies=100.0)
        v_p = modes.compute_phase_speeds()
        omega = 2.0 * np.pi * 100.0
        expected = omega / np.array([0.4, 0.3])
        np.testing.assert_allclose(v_p, expected, rtol=1e-12)


class TestRaysMissDistanceUnits:
    """Ray polylines are in metres; ``Rays._miss_distance_to`` consumes
    them verbatim with no unit rescaling."""

    def test_short_polyline_in_metres_is_not_rescaled(self):
        from uacpy.core.results import Rays
        r_m = np.linspace(0.0, 10.0, 11)
        z_m = np.linspace(0.0, 5.0, 11)
        rays = Rays(
            rays=[{
                'r': r_m, 'z': z_m,
                'alpha': 0.0,
                'n_top_bounces': 0, 'n_bot_bounces': 0,
            }],
            is_eigen=False,
            receiver_depths=np.array([5.0]),
            receiver_ranges=np.array([10.0]),
            model='Test', frequencies=100.0,
        )
        # The polyline passes exactly through (r=10, z=5), so miss == 0
        # in metres. (No km->m rescale: that would blow the miss up to
        # ~10 km.)
        miss, _ = rays._miss_distance_to(rays.rays[0], 10.0, 5.0)
        assert miss == pytest.approx(0.0, abs=1e-12)


class TestArrivalsFlatListKeys:
    """``Arrivals.arrivals`` flat list carries writer-aligned bounce keys."""

    def test_arrivals_flat_list_uses_n_bounce_keys(self):
        from uacpy.core.results import Arrivals
        # Build a minimal payload with the canonical IO key naming.
        payload = [[[{
            "delays": np.array([0.1, 0.2]),
            "amplitudes": np.array([1.0, 0.5]),
            "phases": np.array([0.0, 0.1]),
            "n_top_bounces": np.array([0, 1], dtype=int),
            "n_bot_bounces": np.array([1, 2], dtype=int),
            "src_angles": np.array([0.0, 5.0]),
            "rcv_angles": np.array([0.0, -5.0]),
            "n_arrivals": 2,
        }]]]
        arr = Arrivals(
            by_receiver=payload,
            receiver_depths=np.array([50.0]),
            receiver_ranges=np.array([1000.0]),
            model='Test',
            frequencies=100.0,
        )
        table = arr.arrivals
        assert len(table) == 2
        for row in table:
            assert 'n_top_bounces' in row
            assert 'n_bot_bounces' in row
        assert table[0]['kind'] == 'bottom'
        assert table[1]['kind'] == 'both'
        assert table[0]['n_bot_bounces'] == 1
        assert table[1]['n_top_bounces'] == 1


class TestArrivalsFilterChain:
    """Arrivals exposes a Rays-style filter chain over a flat list of
    arrival events; no continuous-axis ``at(...)`` slicer."""

    def _arrivals(self):
        from uacpy.core.results import Arrivals
        # Two cells (1 src × 1 depth × 2 ranges) with a mix of bounce kinds.
        cell0 = {
            "delays": np.array([0.1, 0.2, 0.3]),
            "amplitudes": np.array([1.0, 0.5, 0.2]),
            "phases": np.array([0.0, 0.0, 0.0]),
            "n_top_bounces": np.array([0, 1, 1], dtype=int),
            "n_bot_bounces": np.array([0, 0, 2], dtype=int),
            "src_angles": np.array([0.0, 5.0, 10.0]),
            "rcv_angles": np.array([0.0, -5.0, -10.0]),
        }
        cell1 = {
            "delays": np.array([0.4]),
            "amplitudes": np.array([0.8]),
            "phases": np.array([0.0]),
            "n_top_bounces": np.array([0], dtype=int),
            "n_bot_bounces": np.array([1], dtype=int),
            "src_angles": np.array([2.0]),
            "rcv_angles": np.array([-2.0]),
        }
        return Arrivals(
            by_receiver=[[[cell0, cell1]]],
            receiver_depths=np.array([50.0]),
            receiver_ranges=np.array([1000.0, 2000.0]),
            model='Test', frequencies=100.0,
        )

    def test_flat_list_length(self):
        a = self._arrivals()
        assert len(a) == 4    # 3 from cell0 + 1 from cell1

    def test_phases_returns_radians_from_degree_store(self):
        # The .arr file stores phase in degrees (ArrMod.f90 writes RadDeg*Phase);
        # the public .phases accessor must return radians for exp(1j*phase).
        from uacpy.core.results import Arrivals
        cell = {
            "delays": np.array([0.1]), "amplitudes": np.array([1.0]),
            "phases": np.array([90.0]),   # degrees as read from the file
            "n_top_bounces": np.array([0], dtype=int),
            "n_bot_bounces": np.array([0], dtype=int),
            "src_angles": np.array([0.0]), "rcv_angles": np.array([0.0]),
        }
        a = Arrivals(by_receiver=[[[cell]]], receiver_depths=np.array([50.0]),
                     receiver_ranges=np.array([1000.0]),
                     model='Test', frequencies=100.0)
        assert a.phases[0] == pytest.approx(np.pi / 2)

    def test_filter_by_bounces_kind(self):
        a = self._arrivals()
        direct = a.filter_by_bounces(kind='direct')
        assert len(direct) == 1
        assert direct.arrivals[0]['delay'] == pytest.approx(0.1)
        bottom = a.filter_by_bounces(kind='bottom')
        assert len(bottom) == 1
        assert bottom.arrivals[0]['delay'] == pytest.approx(0.4)

    def test_filter_by_bounces_top_low_high(self):
        a = self._arrivals()
        # 0-1 surface bounces — keeps everything but the 'both' arrival
        # has top=1 too, so all four pass; instead use bot=(1, None).
        few_bot = a.filter_by_bounces(bot=(1, None))
        assert len(few_bot) == 2     # cell0 last + cell1 last
        exact_top = a.filter_by_bounces(top=1)
        assert len(exact_top) == 2   # cell0 idx 1 and 2

    def test_in_delay_window(self):
        a = self._arrivals()
        mid = a.in_delay_window(0.15, 0.35)
        assert len(mid) == 2
        assert all(0.15 <= x['delay'] <= 0.35 for x in mid.arrivals)

    def test_top_n_by_amplitude(self):
        a = self._arrivals()
        top2 = a.top_n_by_amplitude(2)
        assert len(top2) == 2
        amps = [x['amplitude'] for x in top2.arrivals]
        assert amps == sorted(amps, reverse=True)
        assert amps[0] == 1.0   # cell0 first

    def test_filter_chain_returns_arrivals(self):
        from uacpy.core.results import Arrivals
        a = self._arrivals()
        chained = a.filter(lambda x: x['n_bot_bounces'] >= 1).top_n_by_amplitude(1)
        assert isinstance(chained, Arrivals)
        assert len(chained) == 1


class TestSpawnedArrivalCellsMatchTheReadersRecord:
    """``Arrivals._rebuild_by_receiver`` rebuilds the per-cell record that
    ``io/oalib_reader.read_arr_file`` produces, so a filtered or sorted
    ``Arrivals`` feeds the same consumers as a freshly-read one.

    ``models.bellhop.delayandsum`` reads ``cell['n_arrivals']`` as its first
    statement and indexes the columns from there, so a rebuilt cell missing
    that key raises ``KeyError`` on a public object. Types are asserted
    alongside keys: a 0-d array or a wider integer would make the key sets
    agree while the values diverged, which is the same defect one layer
    down."""

    #: The record ``read_arr_file`` builds for a cell holding two arrivals,
    #: key for key and dtype for dtype (``io/oalib_reader.py``).
    READER_CELL = {
        "amplitudes": np.array([1.0, 0.5], dtype='float64'),
        "phases": np.array([0.0, 30.0], dtype='float64'),
        "delays": np.array([0.1, 0.2], dtype='float64'),
        "delays_imag": np.array([0.0, 0.0], dtype='float64'),
        "src_angles": np.array([0.0, 5.0], dtype='float64'),
        "rcv_angles": np.array([0.0, -5.0], dtype='float64'),
        "n_top_bounces": np.array([0, 1], dtype='int32'),
        "n_bot_bounces": np.array([1, 2], dtype='int32'),
        "n_arrivals": 2,
    }

    def _arrivals(self):
        from uacpy.core.results import Arrivals
        empty = {k: (np.array([], dtype=v.dtype) if isinstance(v, np.ndarray)
                     else 0)
                 for k, v in self.READER_CELL.items()}
        return Arrivals(
            by_receiver=[[[dict(self.READER_CELL), empty]]],
            receiver_depths=np.array([50.0]),
            receiver_ranges=np.array([1000.0, 2000.0]),
            model='Test', frequencies=100.0,
        )

    @staticmethod
    def _signature(cell):
        return {k: (v.dtype.str if isinstance(v, np.ndarray) else type(v))
                for k, v in sorted(cell.items())}

    def test_a_spawned_cell_carries_the_readers_keys_and_types(self):
        spawned = self._arrivals().filter(lambda a: True)
        assert self._signature(spawned.by_receiver[0][0][0]) == \
            self._signature(self.READER_CELL)

    def test_an_emptied_cell_carries_the_readers_keys_and_types(self):
        # The other side of the count boundary: a predicate that keeps
        # nothing still has to produce the reader's record, with zero rows.
        spawned = self._arrivals().filter(lambda a: False)
        cell = spawned.by_receiver[0][0][0]
        assert self._signature(cell) == self._signature(self.READER_CELL)
        assert cell['n_arrivals'] == 0
        assert cell['delays'].size == 0

    def test_n_arrivals_counts_the_rows_that_survived(self):
        spawned = self._arrivals().filter(
            lambda a: a['delay'] < 0.15)
        cell = spawned.by_receiver[0][0][0]
        assert cell['n_arrivals'] == 1
        assert cell['n_arrivals'] == len(cell['delays'])
        assert type(cell['n_arrivals']) is int

    def test_delayandsum_accepts_a_spawned_cell(self):
        from uacpy.models.bellhop import delayandsum
        source = np.ones(8)
        fresh, _ = delayandsum(dict(self.READER_CELL), source, 10000.0, 1000.0)
        spawned_cell = self._arrivals().filter(
            lambda a: True).by_receiver[0][0][0]
        rebuilt, _ = delayandsum(spawned_cell, source, 10000.0, 1000.0)
        assert rebuilt.shape == fresh.shape
        assert np.allclose(rebuilt, fresh)


class TestTimeFieldChainAccessors:
    """A time-domain :class:`Field` carries ``coords={'depth', 'range',
    'time'}``. Slicing follows the same axis-drop rule as 2-D fields."""

    @staticmethod
    def _ts():
        rng = np.random.default_rng(0)
        data = rng.standard_normal((3, 4, 50))
        return Field(
            data=data,
            coords={
                'depth': np.linspace(10, 90, 3),
                'range': np.linspace(100, 1000, 4),
                'time': np.linspace(0, 0.49, 50),
            },
            model='Test', frequencies=100.0,
        )

    def test_at_partial_keeps_remaining_axes(self):
        ts = self._ts()
        sliced = ts.at(depth=50.0)
        assert list(sliced.coords) == ['range', 'time']
        assert sliced.data.shape == (4, 50)
        assert sliced.pinned['depth'] == 50.0

    def test_at_both_spatial_drops_to_trace(self):
        ts = self._ts()
        trace = ts.at(depth=50.0, range=500.0)
        assert list(trace.coords) == ['time']
        assert trace.data.shape == (50,)
        assert set(trace.pinned) == {'depth', 'range'}

    def test_max_records_all_axes_in_pinned(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((3, 4, 50))
        data[2, 1, 30] = 100.0
        ts = Field(
            data=data,
            coords={
                'depth': np.linspace(10, 90, 3),
                'range': np.linspace(100, 1000, 4),
                'time': np.linspace(0, 0.49, 50),
            },
            model='Test', frequencies=100.0,
        )
        m = ts.max()
        assert list(m.coords) == []
        assert float(m.data) == pytest.approx(100.0)
        assert m.pinned['depth'] == ts.coords['depth'][2]
        assert m.pinned['range'] == ts.coords['range'][1]
        assert m.pinned['time'] == ts.coords['time'][30]


class TestReflectionCoefficientChainAccessors:
    """``ReflectionCoefficient.at`` — label slicing of the angle and
    frequency axes; broadband-only kwargs raise on narrowband instances."""

    def _broadband_rc(self):
        from uacpy.core.results import ReflectionCoefficient
        theta = np.linspace(0, 90, 91)            # 91 angles
        freqs = np.array([50.0, 100.0, 200.0])    # 3 frequencies
        R = np.outer(np.cos(np.deg2rad(theta)), np.ones(3)) ** 2
        phi = np.zeros_like(R)
        return ReflectionCoefficient(
            theta=theta, R=R, phi=phi,
            frequencies=freqs, model='Test',
        )

    def test_at_frequency_returns_narrowband(self):
        from uacpy.core.results import ReflectionCoefficient
        rc = self._broadband_rc()
        sliced = rc.at(frequency=100.0)
        assert isinstance(sliced, ReflectionCoefficient)
        assert not sliced.is_broadband
        assert sliced.R.shape == (91,)

    def test_at_angle_keeps_broadband(self):
        from uacpy.core.results import ReflectionCoefficient
        rc = self._broadband_rc()
        sliced = rc.at(angle=45.0)
        assert isinstance(sliced, ReflectionCoefficient)
        assert sliced.theta.shape == (1,)
        assert sliced.R.shape == (1, 3)

    def test_at_both_collapses_to_single_value(self):
        rc = self._broadband_rc()
        sliced = rc.at(angle=45.0, frequency=100.0)
        assert sliced.theta.shape == (1,)
        assert sliced.R.shape == (1,)

    def test_at_frequency_on_narrowband_raises(self):
        from uacpy.core.results import ReflectionCoefficient
        rc = ReflectionCoefficient(
            theta=np.linspace(0, 90, 5),
            R=np.linspace(0, 1, 5),
            phi=np.zeros(5),
            model='Test', frequencies=100.0,
        )
        with pytest.raises(ConfigurationError, match="broadband"):
            rc.at(frequency=100.0)

    @pytest.mark.parametrize('slice_call', [
        lambda rc: rc.at(frequency=100.0),
        lambda rc: rc.at(angle=45.0),
        lambda rc: rc.isel(frequency=1),
        lambda rc: rc.isel(angle=10),
        lambda rc: rc.eval(frequency=120.0),
    ])
    def test_slicing_keeps_provenance(self, slice_call):
        """Every slice carries ``model_source`` (the plot credit) and
        ``phase_reference`` — the same invariant ``Field.id_kwargs``
        enforces."""
        from uacpy.models.sources import model_source
        src = model_source('acoustics_toolbox')
        rc = self._broadband_rc()
        rc.model_source = src
        rc.phase_reference = 'travelling_wave'
        sliced = slice_call(rc)
        assert sliced.model_source is src
        assert sliced.phase_reference == 'travelling_wave'

    def test_at_theta_is_synonym_for_angle(self):
        rc = self._broadband_rc()
        by_angle = rc.at(angle=45.0)
        by_theta = rc.at(theta=45.0)
        assert by_theta.theta.shape == (1,)
        assert np.array_equal(by_theta.theta, by_angle.theta)

    def test_at_unknown_axis_raises(self):
        # Generic Field.at-style form rejects axes it doesn't have.
        rc = self._broadband_rc()
        with pytest.raises(ConfigurationError, match="unknown axis"):
            rc.at(depth=5.0)

    def test_at_angle_and_theta_together_raises(self):
        rc = self._broadband_rc()
        with pytest.raises(ConfigurationError, match="synonyms"):
            rc.at(angle=45.0, theta=45.0)

    def test_isel_positional_angle(self):
        rc = self._broadband_rc()
        s = rc.isel(angle=2)
        assert s.theta.shape == (1,) and s.theta[0] == rc.theta[2]
        assert s.R.shape == (1, 3)

    def test_isel_oob_raises_indexerror(self):
        with pytest.raises(IndexError):
            self._broadband_rc().isel(angle=999)

    def test_eval_interpolates_off_grid_angle(self):
        rc = self._broadband_rc()              # 1° grid → 30.5° is off-grid
        s = rc.eval(angle=30.5)
        assert s.theta[0] == pytest.approx(30.5)
        expect = 0.5 * (np.cos(np.deg2rad(30)) ** 2
                        + np.cos(np.deg2rad(31)) ** 2)   # linear of cos^2
        assert s.R[0, 0] == pytest.approx(expect, abs=1e-6)

    def test_eval_method_cubic(self):
        rc = self._broadband_rc()
        s = rc.eval(angle=30.5, method='cubic')
        assert s.R[0, 0] == pytest.approx(np.cos(np.deg2rad(30.5)) ** 2, abs=1e-3)


class TestSoundSpeedProfileNearestVsInterp:
    """``SoundSpeedProfile.at(...)`` is **nearest** (never fabricates),
    ``.eval(...)`` is **linear**, ``.isel(...)`` is **positional** — the
    grid-library invariant shared with ``Field`` et al."""

    def _ssp(self):
        from uacpy.core.environment import SoundSpeedProfile
        return SoundSpeedProfile(
            depths=np.array([0.0, 100.0, 200.0]),
            data=np.array([[1500.0], [1490.0], [1480.0]])
        )

    def test_at_picks_nearest_depth(self):
        ssp = self._ssp()
        # depth=51 is closer to 100 than to 0 → returns the 100m sample
        sliced = ssp.at(depth=51.0)
        assert sliced.depths[0] == 100.0
        assert sliced.value == 1490.0

    def test_eval_interpolates_linear(self):
        ssp = self._ssp()
        # depth=50 is halfway between (0, 1500) and (100, 1490) → 1495
        sliced = ssp.eval(depth=50.0)
        assert sliced.depths[0] == 50.0
        assert sliced.value == pytest.approx(1495.0)

    def test_isel_positional_depth(self):
        ssp = self._ssp()
        sliced = ssp.isel(depth=1)
        assert sliced.depths[0] == 100.0 and sliced.value == 1490.0
        with pytest.raises(IndexError):
            ssp.isel(depth=9)


class TestSoundSpeedProfileDepthOnlySliceOf2D:
    """A depth-only slice of a range-dependent profile is ambiguous.

    Silently returning the r = 0 column is wrong physics on exactly the
    profiles the 2-D carrier exists for, so ``at`` / ``eval`` / ``isel``
    require the range to be pinned (or the range axis collapsed first).
    """

    def _ssp2d(self):
        from uacpy.core.environment import SoundSpeedProfile
        return SoundSpeedProfile.from_2d(
            depths=[0.0, 100.0], ranges=[0.0, 5000.0],
            matrix=[[1500.0, 1400.0], [1490.0, 1390.0]])

    @pytest.mark.parametrize('call', [
        lambda s: s.at(depth=50.0),
        lambda s: s.eval(depth=50.0),
        lambda s: s.isel(depth=0),
    ])
    def test_depth_only_slice_raises(self, call):
        with pytest.raises(ConfigurationError, match='range-dependent'):
            call(self._ssp2d())

    def test_pinning_the_range_works(self):
        s = self._ssp2d()
        assert s.at(depth=0.0, range=5000.0).value == pytest.approx(1400.0)
        assert s.eval(depth=0.0, range=2500.0).value == pytest.approx(1450.0)
        assert s.isel(depth=0, range=1).value == pytest.approx(1400.0)

    def test_collapsing_the_range_axis_works(self):
        assert self._ssp2d().collapse('mean').at(depth=0.0).value == \
            pytest.approx(1450.0)

    def test_range_only_and_1d_paths_unaffected(self):
        from uacpy.core.environment import SoundSpeedProfile
        assert self._ssp2d().eval(range=5000.0).data.shape == (2, 1)
        flat = SoundSpeedProfile.from_isovelocity(100.0, 1500.0)
        assert flat.at(depth=50.0).value == pytest.approx(1500.0)

    @pytest.mark.parametrize('call', [
        lambda s: s.eval(range=0.0, method='bogus'),
        lambda s: s.eval(depth=50.0, method='bogus'),
    ])
    def test_interp_method_validated_on_every_path(self, call):
        """The membership check runs on entry, so the range-independent
        shortcut cannot swallow a bad method name."""
        from uacpy.core.environment import SoundSpeedProfile
        flat = SoundSpeedProfile.from_isovelocity(100.0, 1500.0)
        with pytest.raises(ConfigurationError, match='interpolation method'):
            call(flat)


class TestSoundSpeedProfileExtendTo:
    """``SoundSpeedProfile.extend_to(z_max)`` is the canonical alignment
    hook used by every env writer. Must extend OR truncate so that
    ``ssp.depths[-1] == z_max`` exactly."""

    def _profile(self, depths, speeds):
        from uacpy.core.environment import SoundSpeedProfile
        return SoundSpeedProfile(
            depths=np.asarray(depths, dtype=float),
            data=np.asarray(speeds, dtype=float).reshape(-1, 1)
        )

    def test_noop_when_depth_max_equals_deepest(self):
        ssp = self._profile([0, 100, 200], [1500, 1490, 1485])
        assert ssp.extend_to(200.0) is ssp

    def test_extend_with_constant_extrapolation(self):
        out = self._profile([0, 100], [1500, 1490]).extend_to(300.0)
        assert out.depths[-1] == 300.0
        assert out.data[-1, 0] == 1490.0

    def test_truncate_with_linear_interpolation(self):
        out = self._profile([0, 100, 200], [1500, 1490, 1480]).extend_to(150.0)
        assert out.depths[-1] == 150.0
        assert out.data[-1, 0] == pytest.approx(1485.0)
        assert (out.depths <= 150.0).all()

    def test_truncate_then_extend_round_trip(self):
        ssp = self._profile([0, 100, 200], [1500, 1490, 1480])
        out = ssp.extend_to(150.0).extend_to(150.0)
        assert out.depths[-1] == 150.0
        assert out.data[-1, 0] == pytest.approx(1485.0)

    def test_noop_under_floating_point_drift(self):
        """``extend_to`` is a no-op when the requested depth matches the
        deepest sample to within a small relative tolerance — a 1-ulp
        drift (from e.g. a round trip through I/O) must not rewrite the
        bottom sample."""
        ssp = self._profile([0, 100, 200], [1500, 1490, 1485])
        # Smallest perturbation that survives a few arithmetic ops:
        perturbed = 200.0 + 1e-12
        out = ssp.extend_to(perturbed)
        assert out is ssp

    def test_truncation_snaps_a_sample_inside_the_readers_last_point_window(self):
        """``misc/sspMod.f90:353`` ends a medium's SSP block at the first
        sample within ``AT_LAST_SSP_POINT_EPS_M`` (1.19e-5 m) of the declared
        medium depth. Truncating beside such a sample emits two rows metres
        apart in index and microns apart in depth; the reader takes the first
        as the end of the block and consumes the second as the bottom-option
        record. The sample has to *move* onto the target instead — the same
        rule the near-miss branch applies when no truncation is needed."""
        ssp = self._profile([0, 50, 99.999995, 150, 200],
                            [1500, 1495, 1490, 1485, 1480])
        out = ssp.extend_to(100.0)
        assert out.depths.tolist() == [0.0, 50.0, 100.0]
        assert out.data[-1, 0] == pytest.approx(1490.0)
        assert float(np.min(np.diff(out.depths))) > 1.1920929e-05

    def test_the_written_deck_ends_the_block_once(self):
        """The consequence the snap exists for, read off the deck the AT
        writer produces: one last SSP row at the medium depth, not two."""
        import io
        from uacpy.io.oalib_writer import write_ssp_section
        ssp = self._profile([0, 50, 99.999995, 150, 200],
                            [1500, 1495, 1490, 1485, 1480])
        env = uacpy.Environment(name='snap', bathymetry=100.0, ssp=ssp,
                                bottom=1800.0)
        buf = io.StringIO()
        write_ssp_section(buf, env, 100.0)
        depths = [float(line.split()[0]) for line in
                  buf.getvalue().splitlines()[1:] if line.strip()]
        assert depths == [0.0, 50.0, 100.0]

    def test_an_ordinary_truncation_interpolates_a_new_sample(self):
        """The discriminating half: when the deepest surviving sample is a
        real distance above the target, the final row is still interpolated
        rather than dragged down onto it."""
        out = self._profile([0, 100, 200], [1500, 1490, 1480]).extend_to(150.0)
        assert out.depths.tolist() == [0.0, 100.0, 150.0]
        assert out.data[-1, 0] == pytest.approx(1485.0)

    def test_the_source_profile_is_untouched(self):
        ssp = self._profile([0, 50, 99.999995, 150, 200],
                            [1500, 1495, 1490, 1485, 1480])
        ssp.extend_to(100.0)
        assert ssp.depths[2] == pytest.approx(99.999995)
        assert ssp.depths.size == 5

    def test_a_range_dependent_profile_snaps_every_column(self):
        from uacpy.core.environment import SoundSpeedProfile
        ssp = SoundSpeedProfile(
            depths=np.array([0.0, 50.0, 99.999995, 150.0]),
            data=np.array([[1500.0, 1502.0], [1495.0, 1497.0],
                           [1490.0, 1492.0], [1485.0, 1487.0]]),
            ranges=np.array([0.0, 5000.0]))
        out = ssp.extend_to(100.0)
        assert out.depths.tolist() == [0.0, 50.0, 100.0]
        assert out.data[-1].tolist() == [1490.0, 1492.0]
        assert out.ranges.tolist() == [0.0, 5000.0]


class TestFieldSlicing:
    """:meth:`Field.at` / :meth:`Field.isel` drop the named axis from
    ``coords`` and record the selected sample in :attr:`pinned`.
    :meth:`Field.max` does the same for every axis."""

    @staticmethod
    def _full_grid(complex_data: bool = True):
        from uacpy.core.results import Field
        if complex_data:
            data = (np.arange(20).reshape(4, 5) + 1j).astype(complex)
        else:
            data = np.arange(20, dtype=float).reshape(4, 5) + 30.0
        return Field(
            data=data,
            coords={
                'depth': np.linspace(10, 90, 4),
                'range': np.linspace(100, 1000, 5),
            },
            model='Test', frequencies=100.0,
        )

    @staticmethod
    def _tf():
        from uacpy.core.results import Field
        data = (np.arange(24).reshape(2, 3, 4) + 1j).astype(complex)
        return Field(
            data=data,
            coords={
                'depth': np.array([10., 20.]),
                'range': np.array([100., 200., 300.]),
                'frequency': np.array([100., 200., 300., 400.]),
            },
            phase_reference='travelling_wave',
            model='Test',
        )

    def test_full_grid_tl_preserves_data_shape(self):
        f = self._full_grid(complex_data=True)
        assert f.db.shape == f.data.shape == (4, 5)
        assert f.p.shape == f.data.shape

    def test_eval_interpolates_and_differs_from_neighbours(self):
        f = self._full_grid(complex_data=False)
        # on-grid eval matches at; off-grid midpoint differs from both neighbours
        assert np.allclose(f.eval(depth=10.0).data, f.at(depth=10.0).data)
        d0, d1 = float(f.coords['depth'][0]), float(f.coords['depth'][1])
        mid = f.eval(depth=(d0 + d1) / 2)
        assert not np.allclose(mid.data, f.isel(depth=0).data)
        assert not np.allclose(mid.data, f.isel(depth=1).data)

    def test_eval_method_nearest_matches_at(self):
        f = self._full_grid(complex_data=False)
        d0, d1 = float(f.coords['depth'][0]), float(f.coords['depth'][1])
        mid = (d0 + d1) / 2
        assert np.allclose(f.eval(depth=mid, method='nearest').data,
                           f.at(depth=mid).data)

    def test_eval_two_axes_to_scalar(self):
        f = self._full_grid(complex_data=False)
        s = f.eval(depth=55.0, range=550.0)
        assert s.data.shape == () and set(s.pinned) == {'depth', 'range'}

    def test_eval_bad_method_and_unknown_axis_raise(self):
        f = self._full_grid(complex_data=False)
        with pytest.raises(ConfigurationError):
            f.eval(depth=50.0, method='spline')
        with pytest.raises(ConfigurationError, match='unknown axis'):
            f.eval(frequency=200.0)

    def test_p_raises_on_real_data(self):
        f = self._full_grid(complex_data=False)
        with pytest.raises(AttributeError):
            _ = f.p

    def test_at_depth_drops_axis_and_records_pinned(self):
        f = self._full_grid()
        sliced = f.at(depth=50.0)
        assert list(sliced.coords) == ['range']
        assert sliced.data.shape == (5,)
        assert 'depth' in sliced.pinned

    def test_at_range_drops_axis(self):
        f = self._full_grid()
        sliced = f.at(range=500.0)
        assert list(sliced.coords) == ['depth']
        assert sliced.data.shape == (4,)
        assert 'range' in sliced.pinned

    def test_at_point_collapses_to_scalar(self):
        f = self._full_grid()
        point = f.at(range=500.0, depth=50.0)
        assert list(point.coords) == []
        assert point.data.shape == ()
        assert isinstance(float(point.db), float)

    def test_max_records_every_axis_in_pinned(self):
        f = self._full_grid()
        m = f.max()
        assert list(m.coords) == []
        assert set(m.pinned) == {'depth', 'range'}
        flat = int(np.argmax(np.abs(f.data)))
        d_idx, r_idx = np.unravel_index(flat, f.data.shape)
        assert m.pinned['depth'] == float(f.coords['depth'][d_idx])
        assert m.pinned['range'] == float(f.coords['range'][r_idx])

    def _tl_grid(self, data):
        from uacpy.core.results import Field
        d = np.asarray(data, dtype=float)
        return Field(
            data=d,
            coords={'depth': np.arange(d.shape[0], dtype=float) * 10 + 10,
                    'range': np.arange(d.shape[1], dtype=float) * 100 + 100},
            model='Test', frequencies=100.0,
        )

    def test_max_on_tl_returns_loudest(self):
        # unit='dB': loudest = smallest dB; a NaN no-data cell and an 80 dB
        # cell must lose to 35 dB.
        f = self._tl_grid([[40.0, np.nan], [35.0, 80.0]])
        m = f.max()
        assert float(m.data) == pytest.approx(35.0)
        assert m.pinned['depth'] == 20.0 and m.pinned['range'] == 100.0

    def test_max_on_tl_skips_nan(self):
        f = self._tl_grid([[np.nan, 50.0], [45.0, 60.0]])
        assert float(f.max().data) == pytest.approx(45.0)

    def test_max_all_nan_raises(self):
        from uacpy.core.exceptions import ConfigurationError
        f = self._tl_grid([[np.nan, np.nan]])
        with pytest.raises(ConfigurationError, match='finite'):
            f.max()

    def test_max_of_a_complex_field_picks_the_largest_magnitude(self):
        f = self._full_grid(complex_data=True)   # |data| argmax, not dB path
        m = f.max()
        assert abs(complex(m.data)) == pytest.approx(np.max(np.abs(f.data)))

    def test_tf_at_frequency_drops_frequency_axis(self):
        tf = self._tf()
        narrow = tf.at(frequency=300.0)
        assert list(narrow.coords) == ['depth', 'range']
        assert narrow.data.shape == (2, 3)
        assert narrow.pinned['frequency'] == 300.0

    def test_tf_at_spatial_keeps_frequency_axis(self):
        tf = self._tf()
        spec = tf.at(depth=15.0, range=200.0)
        assert list(spec.coords) == ['frequency']
        assert spec.data.shape == (4,)
        # ``depth=15`` is equidistant from samples 10 and 20; argmin picks
        # the first → 10.0.
        assert spec.pinned['depth'] == 10.0
        assert spec.pinned['range'] == 200.0

    def test_tf_at_frequency_narrows_identity(self):
        from uacpy.core.results import Field
        # A broadband field carries both a frequency coord and a frequencies
        # identity (as a wrapper emits). Pinning one frequency narrows the
        # identity so f0 / n_frequencies / repr reflect the pinned value.
        freqs = np.array([100., 200., 300., 400.])
        tf = Field(
            data=(np.arange(24).reshape(2, 3, 4) + 1j).astype(complex),
            coords={'depth': np.array([10., 20.]),
                    'range': np.array([100., 200., 300.]),
                    'frequency': freqs},
            model='Test', frequencies=freqs,
        )
        assert tf.n_frequencies == 4 and tf.f0 == 100.0
        narrow = tf.at(frequency=300.0)
        assert narrow.n_frequencies == 1
        assert narrow.f0 == 300.0
        assert 'f=300 Hz' in repr(narrow)
        # a non-frequency slice keeps the full identity
        assert tf.at(depth=10.0).n_frequencies == 4

    def test_tf_to_tl_returns_real_field(self):
        tf = self._tf()
        tl = tf.to_db()
        assert not tl.is_complex
        assert tl.data.shape == tf.data.shape

    def test_tf_to_tl_is_minus_20log10_magnitude(self):
        """``to_db`` is exactly ``-20·log10(|data|)`` (every |data| here is
        far above the PRESSURE_FLOOR clamp, so the clamp is inert)."""
        tf = self._tf()
        tl = tf.to_db()
        np.testing.assert_allclose(
            tl.data, -20.0 * np.log10(np.abs(tf.data)), rtol=1e-12)
        # One hand-checked value: data flat index 3 is 3+1j, |3+1j|² = 10,
        # so -20·log10(√10) = -10 dB exactly.
        k = np.unravel_index(3, tf.data.shape)
        assert tl.data[k] == pytest.approx(-10.0, abs=1e-12)


class TestResultStackInvariants:
    """:class:`ResultStack` is a thin composition wrapper. The
    constructor enforces uniform slab type, uniform model / backend /
    frequencies, and matching ``len(slabs) == len(source_depths)`` so
    the stack's read-through properties (``stack.model``,
    ``stack.frequencies``) never silently disagree with a slab."""

    @staticmethod
    def _slab(*, depths=2, ranges=3, frequencies=100.0, model='Test',
              source_depth=50.0, model_source=None, phase_reference=None):
        from uacpy.core.results import Field
        return Field(
            data=np.ones((depths, ranges), dtype=complex),
            coords={
                'depth': np.arange(depths, dtype=float),
                'range': np.arange(ranges, dtype=float) * 100.0,
            },
            model=model,
            frequencies=frequencies,
            source_depths=np.array([float(source_depth)]),
            model_source=model_source,
            phase_reference=phase_reference,
        )

    def test_requires_at_least_one_slab(self):
        from uacpy.core.results import ResultStack
        with pytest.raises(ConfigurationError, match="at least one slab"):
            ResultStack(slabs=[], coordinate=[])

    def test_rejects_length_mismatch(self):
        from uacpy.core.results import ResultStack
        with pytest.raises(ConfigurationError, match="coordinate length"):
            ResultStack(slabs=[self._slab(source_depth=10.0)],
                        coordinate=[10.0, 20.0])

    def test_rejects_mixed_slab_types(self):
        from uacpy.core.results import Rays, ResultStack
        pf = self._slab(source_depth=10.0)
        ry = Rays(rays=[], model='Test', backend='')
        with pytest.raises(ConfigurationError, match="same concrete type"):
            ResultStack(slabs=[pf, ry], coordinate=[10.0, 20.0])

    def test_rejects_disagreeing_frequencies(self):
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0, frequencies=100.0)
        b = self._slab(source_depth=20.0, frequencies=200.0)
        with pytest.raises(ConfigurationError, match="frequencies"):
            ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])

    def test_rejects_disagreeing_model(self):
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0, model='Bellhop')
        b = self._slab(source_depth=20.0, model='Kraken')
        with pytest.raises(ConfigurationError, match="model"):
            ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])

    def test_accepts_uniform_slabs(self):
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0)
        b = self._slab(source_depth=20.0)
        stack = ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])
        assert stack.slab_type is Field
        assert stack.coordinate_name == 'source_depth'
        assert stack.n_slabs == 2
        assert len(stack) == 2
        # Universally-shared metadata reads through from slab[0].
        assert stack.model == 'Test'
        np.testing.assert_array_equal(
            stack.coordinate, np.array([10.0, 20.0]))

    def test_db_stacks_slab_views_into_a_dense_array(self):
        """``stack.db`` is one dense ``(n_slabs, *slab.shape)`` ndarray, so
        generic code can read ``result.db`` whether one or many source
        depths were requested."""
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0)               # |p| = 1 → 0 dB
        b = self._slab(source_depth=20.0)
        b.data[...] = 10.0 + 0j                          # |p| = 10 → -20 dB
        stack = ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])
        db = stack.db
        assert isinstance(db, np.ndarray)
        assert db.shape == (2, 2, 3)                     # (n_slabs, z, r)
        np.testing.assert_allclose(db[0], 0.0)
        np.testing.assert_allclose(db[1], -20.0)

    def test_iteration_and_label_select_share_slab_identity(self):
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0)
        b = self._slab(source_depth=20.0)
        stack = ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])
        # __getitem__ returns the same object stored in slabs[i].
        assert stack[0] is a
        assert stack[1] is b
        # at(source_depth=z) routes to the nearest slab by label.
        assert stack.at(source_depth=20.0) is b
        # Iteration yields (source_depth, slab) pairs.
        pairs = list(stack)
        assert pairs == [(10.0, a), (20.0, b)]

    def test_frequency_axis_stack(self):
        """Stacking along ``frequency`` is just a coordinate-name swap.
        Slabs legitimately differ on ``frequencies`` (the stacking axis)
        while sharing ``source_depths`` and ``model``."""
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=50.0, frequencies=100.0)
        b = self._slab(source_depth=50.0, frequencies=200.0)
        stack = ResultStack(slabs=[a, b], coordinate=[100.0, 200.0],
                            coordinate_name='frequency')
        assert stack.coordinate_name == 'frequency'
        assert stack.at(frequency=200.0) is b
        # A kwarg that is not the stacking axis names an axis the stack
        # does not have.
        with pytest.raises(ConfigurationError, match="frequency"):
            stack.at(source_depth=200.0)

    def test_frequency_axis_rejects_disagreeing_source_depths(self):
        """When stacking by ``frequency`` the slabs must still agree on
        ``source_depths`` — ``frequency`` is the varying axis, not depth."""
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0, frequencies=100.0)
        b = self._slab(source_depth=99.0, frequencies=200.0)
        with pytest.raises(ConfigurationError, match="source_depths"):
            ResultStack(slabs=[a, b], coordinate=[100.0, 200.0],
                        coordinate_name='frequency')

    def test_external_coordinate_axis(self):
        """An external coordinate (e.g. wind speed) requires both
        ``frequencies`` and ``source_depths`` to agree across slabs;
        ``at(<coordinate_name>=…)`` keys off the custom name."""
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=50.0, frequencies=100.0)
        b = self._slab(source_depth=50.0, frequencies=100.0)
        stack = ResultStack(slabs=[a, b], coordinate=[5.0, 15.0],
                            coordinate_name='wind_speed')
        assert stack.coordinate_name == 'wind_speed'
        assert stack.at(wind_speed=15.0) is b
        # An external coordinate requires both internal axes to agree, so a
        # disagreeing source_depth is rejected.
        c = self._slab(source_depth=99.0, frequencies=100.0)
        with pytest.raises(ConfigurationError, match="source_depths"):
            ResultStack(slabs=[a, c], coordinate=[5.0, 15.0],
                        coordinate_name='wind_speed')

    def test_forwards_the_whole_identity_surface(self):
        """A stack forwards every identity field a slab carries, not just
        model/backend — the plotters read ``model_source`` for the
        model-credit footnote."""
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0, model_source='engine-provenance',
                       phase_reference='travelling_wave')
        b = self._slab(source_depth=20.0, model_source='engine-provenance',
                       phase_reference='travelling_wave')
        stack = ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])
        # Every field of the shared identity surface, so a field added to
        # ``Result.__init__`` cannot silently stop at the stack boundary.
        missing = [k for k in a.id_kwargs() if not hasattr(stack, k)]
        assert not missing, f"ResultStack does not forward {missing}"
        assert stack.model_source == 'engine-provenance'
        assert stack.phase_reference == 'travelling_wave'
        # The stacking axis reads back as the stack coordinate; the other
        # axis reads through from the (verified identical) slabs.
        np.testing.assert_array_equal(stack.source_depths, [10.0, 20.0])
        np.testing.assert_array_equal(stack.frequencies, [100.0])

    def test_frequency_stack_forwards_the_frequency_axis(self):
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=50.0, frequencies=100.0)
        b = self._slab(source_depth=50.0, frequencies=200.0)
        stack = ResultStack(slabs=[a, b], coordinate=[100.0, 200.0],
                            coordinate_name='frequency')
        np.testing.assert_array_equal(stack.frequencies, [100.0, 200.0])
        np.testing.assert_array_equal(stack.source_depths, [50.0])


class TestResultIngestCopiesArrays:
    """Every ``Result`` copies on ingest, and ``Field.to_dict`` is a real
    snapshot — a caller mutating their source array can never reach inside a
    result, and a cached dict never aliases the field it came from."""

    @staticmethod
    def _field():
        from uacpy.core.results import Field
        return Field(
            data=np.ones((2, 3), dtype=complex),
            coords={'depth': np.array([10.0, 20.0]),
                    'range': np.array([100.0, 200.0, 300.0])},
            model='Test', source_depths=np.array([50.0]),
            frequencies=np.array([100.0]))

    def test_result_copies_identity_arrays_on_ingest(self):
        from uacpy.core.results import Field
        sd = np.array([50.0])
        fr = np.array([100.0])
        f = Field(data=np.ones((1, 1)), coords={'depth': [0.0], 'range': [0.0]},
                  source_depths=sd, frequencies=fr)
        sd[0] = 999.0
        fr[0] = 999.0
        assert f.source_depths[0] == 50.0
        assert f.frequencies[0] == 100.0

    def test_to_dict_is_a_snapshot(self):
        f = self._field()
        d = f.to_dict()
        for key, live in (('data', f.data), ('frequencies', f.frequencies),
                          ('source_depths', f.source_depths)):
            assert not np.shares_memory(d[key], live), key
        for name, vec in f.coords.items():
            assert not np.shares_memory(d['coords'][name], vec), name

    def test_covariance_and_replicas_copy_on_ingest(self):
        from uacpy.core.results import Covariance, Replicas
        cov_src = np.zeros((1, 2, 2), dtype=complex)
        cov = Covariance(covariance=cov_src, model='OASN')
        cov_src[0, 0, 0] = 99.0
        assert cov.covariance[0, 0, 0] == 0.0

        rep_src = np.zeros((1, 1, 1, 1, 2), dtype=complex)
        rep = Replicas(replicas=rep_src, replica_z=[0.0], replica_x=[0.0],
                       replica_y=[0.0], model='OASN')
        rep_src[0, 0, 0, 0, 0] = 7.0
        assert rep.replicas[0, 0, 0, 0, 0] == 0.0


class TestCopyAndGeolocation:
    """`.copy()` is universal across carriers + results; Environment carries
    optional geolocation/date provenance that survives copy."""

    def test_copy_symmetry_across_carriers_and_io(self):
        import uacpy
        from uacpy import (Bathymetry, Altimetry, Surface, Bottom,
                           SoundSpeedProfile, BoundaryProperties)
        from uacpy.core.bottom import SeabedColumn, SedimentLayer
        objs = [
            Bathymetry(ranges=[0, 1000.], depths=[100, 90.]),
            Altimetry(ranges=[0, 1000.], heights=[0, -2.]),
            Surface.coerce(BoundaryProperties(acoustic_type='vacuum')),
            Bottom.from_halfspace(BoundaryProperties()),
            SeabedColumn(layers=[], halfspace=BoundaryProperties()),
            SedimentLayer(thickness=5.0, sound_speed=1600.0, density=1.8,
                          attenuation=0.2),
            BoundaryProperties(),
            SoundSpeedProfile.from_pairs([(0, 1500), (100, 1490.)]),
            uacpy.Source(depths=50., frequencies=120.),
            uacpy.Receiver(depths=[100.], ranges=[2000.]),
            uacpy.Environment(bathymetry=200., ssp=1500.),
        ]
        for o in objs:
            c = o.copy()
            assert type(c) is type(o) and c is not o
            # Deep copy: no top-level array is shared, so mutating the
            # copy can never reach the original.
            for name, val in vars(o).items():
                if isinstance(val, np.ndarray):
                    assert not np.shares_memory(getattr(c, name), val), (
                        type(o).__name__, name)

    def test_every_carrier_docs_call_a_carrier_has_copy(self):
        """``copy()``'s docstring says "symmetric with the other carriers" at
        nine sites, and docs/DEV.md section 5 names the carrier set. Two of
        the classes it lists — both components of ``SeabedColumn``, which has
        ``copy()`` — reached that claim without the method."""
        import uacpy
        from uacpy.core.bottom import (Bottom, BoundaryProperties,
                                       SeabedColumn, SedimentLayer)
        from uacpy.core.surface import Surface
        carriers = (uacpy.SoundSpeedProfile, uacpy.Bathymetry,
                    uacpy.Altimetry, SedimentLayer, SeabedColumn, Bottom,
                    BoundaryProperties, Surface, uacpy.Source,
                    uacpy.Receiver, uacpy.Environment)
        missing = sorted(c.__name__ for c in carriers
                         if not callable(getattr(c, 'copy', None)))
        assert not missing, (
            f"carrier(s) {missing} are named in docs/DEV.md section 5 but "
            f"have no copy()")

    def test_the_bottom_components_copy_deeply(self):
        """The two additions, driven: a mutation of the copy must not reach
        the original, which is the whole content of "deep"."""
        from uacpy.core.bottom import BoundaryProperties, SedimentLayer
        layer = SedimentLayer(thickness=5.0, sound_speed=1600.0, density=1.8,
                              attenuation=0.2)
        clone = layer.copy()
        assert clone is not layer and clone == layer
        clone.sound_speed = 1900.0
        assert layer.sound_speed == 1600.0

        props = BoundaryProperties(sound_speed=1700.0, density=1.9)
        twin = props.copy()
        assert twin is not props and twin == props
        twin.density = 2.5
        assert props.density == 1.9

    def test_source_and_receiver_copy_the_whole_attribute_surface(self):
        """``Source`` / ``Receiver`` deep-copy like every other carrier, so an
        attribute added to either is carried across without editing
        ``copy()``, and no array is shared with the original."""
        import uacpy
        objs = [
            uacpy.Source(depths=[10., 50.], frequencies=[100., 200.],
                         source_type='line',
                         beam_pattern=np.array([[-90., -20.], [90., 0.]])),
            uacpy.Receiver(depths=[100., 200.], ranges=[1000., 2000.]),
        ]
        for o in objs:
            c = o.copy()
            assert vars(c).keys() == vars(o).keys(), type(o).__name__
            for name, val in vars(o).items():
                new = getattr(c, name)
                if isinstance(val, np.ndarray):
                    np.testing.assert_array_equal(new, val)
                    assert not np.shares_memory(new, val), name
                else:
                    assert new == val, name

    def test_result_copy_is_independent(self):
        import uacpy
        env = uacpy.Environment(bathymetry=300., ssp=1500.)
        f = uacpy.Bellhop().compute_tl(
            env, uacpy.Source(depths=50., frequencies=150.),
            uacpy.Receiver(depths=[100., 200.], ranges=[2000., 4000.]))
        c = f.copy()
        assert type(c) is type(f) and c is not f
        # Mutating the copy's payload must not reach the original.
        assert not np.shares_memory(c.data, f.data)
        baseline = f.data.copy()
        c.data[...] = 999.0
        np.testing.assert_array_equal(f.data, baseline)

    def test_environment_geolocation_and_date(self):
        import datetime
        import uacpy
        e = uacpy.Environment(bathymetry=200., ssp=1500.,
                              location=(75., 12.5), date='2026-03-15')
        assert e.location == (75., 12.5)
        assert e.date == datetime.date(2026, 3, 15)
        assert e.transect is None
        # transect → location defaults to the midpoint; explicit overrides it
        t = uacpy.Environment(bathymetry=[(0, 200), (10000, 300)], ssp=1500.,
                              transect=((75., 0.), (77., 40.)))
        assert t.location == (76.0, 20.0)
        o = uacpy.Environment(bathymetry=200., ssp=1500., location=(75., 0.),
                              transect=((75., 0.), (77., 40.)))
        assert o.location == (75., 0.)
        # survives a deep copy
        c = t.copy()
        assert c.location == (76., 20.) and c.transect == ((75., 0.), (77., 40.))
        # hand-built env carries none of it
        plain = uacpy.Environment(bathymetry=100., ssp=1500.)
        assert plain.location is None and plain.transect is None and plain.date is None

    @pytest.mark.parametrize("bad", [(91.0, 0.0), ('a', 'b'), 'not-a-date'])
    def test_environment_geolocation_typed_errors(self, bad):
        import uacpy
        with pytest.raises(ConfigurationError):
            if isinstance(bad, str):
                uacpy.Environment(bathymetry=100., ssp=1500., date=bad)
            else:
                uacpy.Environment(bathymetry=100., ssp=1500., location=bad)


class TestSourceGeometryAndBeamPattern:
    """Source owns source geometry and directivity (spec 2026-07-25)."""

    def test_scaled_is_a_valid_source_type(self):
        src = uacpy.Source(depths=50, frequencies=100, source_type='scaled')
        assert src.source_type == 'scaled'

    def test_unknown_source_type_raises(self):
        with pytest.raises(ConfigurationError, match="source_type"):
            uacpy.Source(depths=50, frequencies=100, source_type='Z')

    def test_beam_pattern_defaults_to_none(self):
        assert uacpy.Source(depths=50, frequencies=100).beam_pattern is None

    def test_beam_pattern_accepts_angle_level_array(self):
        pat = np.array([[-90.0, -20.0], [0.0, 0.0], [90.0, -20.0]])
        src = uacpy.Source(depths=50, frequencies=100, beam_pattern=pat)
        assert src.beam_pattern.shape == (3, 2)

    def test_beam_pattern_wrong_shape_raises(self):
        with pytest.raises(ConfigurationError, match=r"N, 2"):
            uacpy.Source(depths=50, frequencies=100,
                         beam_pattern=np.array([1.0, 2.0, 3.0]))

    def test_beam_pattern_non_monotonic_angles_raise(self):
        # misc/beampattern.f90:56-57 rejects this with ERROUT, which gfortran
        # exits 0 on; catching it here is the point of validating in Python.
        pat = np.array([[0.0, 0.0], [-90.0, -20.0], [90.0, -20.0]])
        with pytest.raises(ConfigurationError):
            uacpy.Source(depths=50, frequencies=100, beam_pattern=pat)

    def test_beam_pattern_single_row_raises(self):
        # bellhop.f90:273 interpolates between rows IBP and IBP+1 after clamping
        # IBP to NSBPPts-1, so one row makes it read below the bound allocated at
        # misc/beampattern.f90:36 and return an all-NaN field with exit code 0.
        # misc/monotonicMod.f90:20 returns .TRUE. for N==1, so the engine's own
        # guard cannot catch it either.
        with pytest.raises(ConfigurationError, match="at least 2"):
            uacpy.Source(depths=50, frequencies=100,
                         beam_pattern=np.array([[0.0, 0.0]]))

    def test_beam_pattern_path_is_stored_as_path(self, tmp_path):
        from pathlib import Path
        sbp = tmp_path / 'pattern.sbp'
        sbp.write_text("2\n-90.0 0.0\n90.0 0.0\n")
        src = uacpy.Source(depths=50, frequencies=100, beam_pattern=sbp)
        assert isinstance(src.beam_pattern, Path)

    def test_copy_carries_geometry_and_pattern(self):
        pat = np.array([[-90.0, -20.0], [90.0, 0.0]])
        src = uacpy.Source(depths=50, frequencies=100,
                           source_type='line', beam_pattern=pat)
        dup = src.copy()
        assert dup.source_type == 'line'
        np.testing.assert_array_equal(dup.beam_pattern, pat)


class TestEnvironmentPredicatesAreProperties:
    """``has_*`` must be properties, not methods.

    As bound methods they are always truthy, so ``if env.has_layered_bottom:``
    silently takes the True branch for every environment. They sit alongside
    ``is_range_dependent``, which is likewise a property.
    """

    _NAMES = (
        'has_range_dependent_bathymetry', 'has_range_dependent_ssp',
        'has_range_dependent_bottom', 'has_layered_bottom',
        'has_range_dependent_layered_bottom', 'has_elastic_bottom',
        'has_elastic_surface',
    )

    @pytest.mark.parametrize('name', _NAMES)
    def test_is_a_property(self, name):
        import inspect
        from uacpy.core.environment import Environment
        assert isinstance(inspect.getattr_static(Environment, name), property), (
            f"Environment.{name} is a {type(inspect.getattr_static(Environment, name)).__name__}; "
            f"as a method it is always truthy in a boolean test")

    def test_flat_environment_reports_false_not_truthy_method(self):
        env = uacpy.Environment(bathymetry=100.0, ssp=1500.0)
        for name in self._NAMES:
            assert getattr(env, name) is False, f"{name} should be False"


class TestReflectionFileDedupe:
    """``dedupe_reflection_file`` is a .brc/.trc rewriter, not a .irc one.

    BOUNCE writes the .irc with a title/frequency header and
    ``(5G15.7, I5)`` records (``Kraken/bounce.f90:225-228``), read back by
    the same fixed format at ``misc/RefCoef.f90:98-107``. The 3-column
    angle dedupe would strip the header and four of the columns, so an
    .irc must be rejected rather than rewritten.
    """

    def _irc(self, tmp_path):
        p = tmp_path / 't.irc'
        p.write_text(
            " ' BOUNCE test '  100.0\n 2\n"
            "     0.1000000     1.0000000     0.0000000"
            "     0.5000000     0.1000000    0\n"
            "     0.2000000     0.9000000     0.1000000"
            "     0.4000000     0.2000000    1\n")
        return p

    def test_irc_is_rejected_and_left_untouched(self, tmp_path):
        from uacpy.io.refl_io import dedupe_reflection_file
        from uacpy.core.exceptions import FileFormatError
        irc = self._irc(tmp_path)
        before = irc.read_text()
        with pytest.raises(FileFormatError, match='row count'):
            dedupe_reflection_file(irc)
        assert irc.read_text() == before

    def test_brc_duplicate_angles_are_collapsed(self, tmp_path):
        from uacpy.io.refl_io import dedupe_reflection_file
        brc = tmp_path / 't.brc'
        brc.write_text("   4\n  0.0 1.0 0.0\n  0.0 1.0 0.0\n"
                       "  30.0 0.8 10.0\n  60.0 0.5 20.0\n")
        dedupe_reflection_file(brc)
        rows = [ln.split() for ln in brc.read_text().splitlines() if ln.strip()]
        assert int(rows[0][0]) == 3
        assert [float(r[0]) for r in rows[1:]] == [0.0, 30.0, 60.0]


class TestFieldEvalSamplingGuard:
    """``Field.eval`` interpolates the same coherent field as
    ``resample_to`` and carried the same +2.3 dB level bias with no warning:
    the guard existed on one public interpolation path and not the other.
    ``eval`` is checked per **requested axis** — interpolating along range
    says nothing about the depth spacing — and ``method='nearest'`` is exempt
    because it fabricates nothing."""

    @staticmethod
    def _field(dr, dz, f0=200.0, c=1500.0):
        ranges = np.arange(1000.0, 1100.0 + dr, dr)
        depths = np.arange(40.0, 60.0 + dz, dz)
        k = 2.0 * np.pi * f0 / c
        row = np.exp(1j * k * ranges)[None, :] / ranges[None, :]
        return Field(data=np.repeat(row, depths.size, axis=0),
                     coords={'depth': depths, 'range': ranges},
                     model='Test', frequencies=f0)

    @pytest.mark.parametrize('axis,dr,dz', [('range', 25.0, 1.0),
                                            ('depth', 1.0, 5.0)])
    def test_eval_warns_on_the_axis_it_interpolates(self, axis, dr, dz):
        field = self._field(dr, dz)
        with pytest.warns(UserWarning, match=f'{axis} samples are'):
            field.eval(**{axis: float(field.coords[axis][0]) + 0.5})

    def test_eval_ignores_a_coarse_axis_it_is_not_interpolating(self):
        # The discriminating half: a coarse depth axis is irrelevant to
        # eval(range=...), and warning about it would be noise.
        field = self._field(dr=1.0, dz=5.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            field.eval(range=1000.5)

    def test_eval_is_silent_on_a_resolved_axis(self):
        field = self._field(dr=1.0, dz=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            field.eval(depth=40.5, range=1000.5)

    def test_nearest_never_warns(self):
        field = self._field(dr=25.0, dz=5.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            field.eval(method='nearest', depth=42.5, range=1012.5)

    def test_resample_to_nearest_never_warns_either(self):
        """The exemption is a property of ``method='nearest'``, not of
        ``eval``: it returns a stored sample on both paths, so there is no
        phase to corrupt and nothing to announce."""
        field = self._field(dr=25.0, dz=5.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            field.resample_to(depths=field.coords['depth'],
                              ranges=field.coords['range'], method='nearest')

    def test_resample_to_warns_when_it_interpolates(self):
        field = self._field(dr=25.0, dz=5.0)
        with pytest.warns(UserWarning, match='samples are'):
            field.resample_to(depths=field.coords['depth'],
                              ranges=field.coords['range'], method='linear')


class TestReflectionCoefficientExtrapolationIsAnnounced:
    """``eval``/``at`` hold the end value outside the tabulated angles, like
    every other carrier. A solver reading the same table does not:
    ``misc/RefCoef.f90:139-140,146-147`` sets R = 0 and phi = 0 outside the
    range, killing the ray. The number is deliberately unchanged — returning 0
    would make this the only carrier with a different extrapolation rule, and 0
    is AT's kill convention, not a claim about R at that angle — but the
    disagreement is named in the warning."""

    @staticmethod
    def _rc():
        th = np.arange(10.0, 41.0, 5.0)
        return ReflectionCoefficient(
            theta=th, R=np.linspace(0.9, 0.3, th.size),
            phi=np.zeros(th.size), model='Test', frequencies=100.0)

    @pytest.mark.parametrize('angle', [5.0, 80.0])
    def test_out_of_range_angle_warns_and_cites_the_solver(self, angle):
        with pytest.warns(UserWarning, match='RefCoef.f90'):
            self._rc().eval(angle=angle)

    def test_in_range_angle_is_silent_and_interpolates(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            out = self._rc().eval(angle=25.0)
        assert float(np.asarray(out.R).ravel()[0]) == pytest.approx(0.6)

    def test_isel_takes_an_index_and_must_not_warn(self):
        # The discriminating case: isel's `angle` is a positional index, so
        # comparing it against degrees is meaningless. A guard on the shared
        # funnel that forgets this fires a bogus warning on every isel.
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self._rc().isel(angle=2)


class TestKindUnitAndDtypeAreIndependentAxes:
    """A Field is described on three axes that must not be collapsed into one:
    ``kind`` (what it is), ``unit`` (what it is measured in) and ``dtype`` (how
    it is stored). Each has a distinct consumer, and each collapse has already
    produced a bug:

    * ``Field.max`` inferred dB-ness from the quantity, so introducing
      ``'reverberation'`` made it return the *quietest* cell of a dB grid.
    * ``compare_models`` keyed on representation, so it refused the ordinary
      RAM-vs-Kraken comparison of one quantity written two ways.
    * ``replica_bank_from_field`` asked for a quantity when what matched-field
      processing actually needs is phase, i.e. the dtype.
    """

    Z = np.array([10.0, 20.0])
    R = np.array([100.0, 200.0])
    DB = np.array([[40.0, 90.0], [70.0, 60.0]])      # 40 dB is the loudest

    def _field(self, data=None, metadata=None, coords=None):
        return Field(data=self.DB if data is None else data,
                     coords=coords or {'depth': self.Z, 'range': self.R},
                     model='Test', frequencies=100.0, metadata=metadata)

    # ── kind: the quantity ────────────────────────────────────────────
    def test_transmission_loss_is_not_a_separate_kind(self):
        # TL is pressure in dB, so it must share pressure's kind and differ
        # only on the unit axis. This is what lets example_07 compare a RAM
        # TL field against a Kraken complex one.
        tl, px = self._field(), self._field(data=self.DB.astype(complex))
        assert tl.kind == px.kind == 'pressure'
        assert (tl.unit, px.unit) == ('dB', 'Pa')

    def test_a_model_tags_a_different_quantity(self):
        assert self._field(metadata={'kind': 'reverberation'}).kind == \
            'reverberation'

    # ── unit: which way is louder ─────────────────────────────────────
    def test_transmission_loss_is_the_one_quantity_that_inverts(self):
        # TL is a *loss*, so the least of it is the loudest.
        f = self._field()
        assert (f.kind, f.unit) == ('pressure', 'dB')
        assert float(f.max().data) == pytest.approx(40.0)

    @pytest.mark.parametrize('kind', ['reverberation', 'signal_excess'])
    def test_a_db_level_is_not_inverted(self, kind):
        # Reverberation and signal excess share TL's dB unit but are levels,
        # not losses: more is more. Deciding direction from the unit alone
        # reports the *weakest* cell of a level grid as the strongest.
        f = self._field(metadata={'kind': kind})
        assert f.unit == 'dB'
        assert float(f.max().data) == pytest.approx(90.0)

    def test_a_time_trace_is_linear_not_db(self):
        # Real data alone does not mean dB — a time trace is Pa, and treating
        # it as a level would make max() return the trace's *trough*.
        t = np.array([0.0, 1.0, 2.0, 3.0])
        f = self._field(data=np.array([1.0, -5.0, 2.0, 0.5]),
                        coords={'time': t})
        assert f.kind == 'pressure' and f.unit == 'Pa'
        assert float(f.max().data) == pytest.approx(-5.0)   # largest |p|

    def test_a_model_may_pin_the_unit(self):
        # probability_of_detection is dimensionless, so it cannot be derived
        # from the storage the way pressure's Pa/dB split is.
        f = self._field(metadata={'kind': 'probability_of_detection',
                                  'unit': '1'})
        assert (f.kind, f.unit) == ('probability_of_detection', '1')

    def test_an_unregistered_quantity_is_refused_at_construction(self):
        # The guard is on the funnel: a typo'd tag that survived construction
        # would resurface as a wrong colour scale or a wrong argmax direction
        # with nothing pointing back at the model that set it.
        with pytest.raises(ConfigurationError, match='unknown Field kind'):
            self._field(metadata={'kind': 'transmission_loss'})
        with pytest.raises(ConfigurationError, match='not measured in'):
            self._field(metadata={'kind': 'reverberation', 'unit': 'Pa'})

    def test_the_untagged_path_never_consults_the_registry(self):
        # Every slice builds a Field; validation must not cost the common
        # case, so an untagged field carries no 'kind'/'unit' metadata at all.
        f = self._field()
        assert 'kind' not in (f.metadata or {})
        assert (f.kind, f.unit) == ('pressure', 'dB')

    # ── dtype: is there phase to work with ────────────────────────────
    def test_matched_field_rejects_a_real_field_of_the_right_kind(self):
        from uacpy.sonar.matched_field import replica_bank_from_field
        tl = self._field()                      # kind='pressure', but real
        assert tl.kind == 'pressure'            # would pass a kind-only guard
        with pytest.raises(ConfigurationError, match='complex'):
            replica_bank_from_field(tl)

    # ── the axes stay independent ─────────────────────────────────────
    def test_compare_models_keys_on_kind_not_representation(self):
        import matplotlib
        matplotlib.use('Agg')
        from uacpy import compare_models
        tl = self._field()
        px = self._field(data=self.DB.astype(complex))
        rv = self._field(metadata={'kind': 'reverberation'})
        # One quantity, two representations: the ordinary comparison.
        compare_models([tl, px])
        # Same representation, different quantity: refused.
        with pytest.raises(ConfigurationError, match='different physical'):
            compare_models([tl, rv])

    def test_all_three_axes_round_trip(self):
        f = self._field(metadata={'kind': 'reverberation'})
        d = f.to_dict()
        assert (d['kind'], d['unit']) == ('reverberation', 'dB')
        back = Field.from_dict(d)
        assert (back.kind, back.unit, back.data.dtype) == \
            (f.kind, f.unit, f.data.dtype)


class TestMaskBelowSeafloor:
    """:meth:`Field.mask_below_seafloor` NaN-masks samples strictly below
    the (range-interpolated) seafloor and returns a copy; a sample exactly
    on the seafloor is kept."""

    @staticmethod
    def _field():
        return Field(
            data=np.ones((3, 2)),
            coords={'depth': np.array([50.0, 120.0, 150.0]),
                    'range': np.array([0.0, 1000.0])},
            model='Test', frequencies=100.0)

    def test_masks_only_cells_below_the_local_seafloor(self):
        f = self._field()
        # Sloping seafloor: 100 m at r=0, 140 m at r=1000. Column r=0
        # loses 120 and 150 m; column r=1000 loses only 150 m.
        masked = f.mask_below_seafloor([(0.0, 100.0), (1000.0, 140.0)])
        expected = np.array([[1.0, 1.0],
                             [np.nan, 1.0],
                             [np.nan, np.nan]])
        np.testing.assert_array_equal(masked.data, expected)
        # The parent field is untouched (mask returns a copy).
        assert np.isfinite(f.data).all()

    def test_boundary_cell_on_the_seafloor_is_kept(self):
        # The mask is strict (depth > seafloor), so a receiver exactly on
        # the interface keeps its value.
        f = Field(
            data=np.ones((2, 1)),
            coords={'depth': np.array([100.0, 100.5]),
                    'range': np.array([500.0])},
            model='Test', frequencies=100.0)
        masked = f.mask_below_seafloor([(0.0, 100.0), (1000.0, 100.0)])
        assert masked.data[0, 0] == 1.0          # exactly on the interface
        assert np.isnan(masked.data[1, 0])       # half a metre below

    def test_requires_canonical_depth_range_layout(self):
        trace = Field(data=np.zeros(4),
                      coords={'time': np.arange(4) * 0.1}, model='Test')
        with pytest.raises(ConfigurationError, match='canonical'):
            trace.mask_below_seafloor([(0.0, 100.0), (1000.0, 100.0)])


class TestSpectrumAndToneExtraction:
    """`get_spectrum` (bare rFFT) and `extract_tone` (windowed phasor
    estimate) on a synthetic pure-tone time Field."""

    A0 = 2.5
    PHI0 = 0.7
    F0 = 32.0
    N = 256

    def _tone_field(self):
        # dt = 1/256 s puts every integer frequency exactly on an rFFT bin.
        t = np.arange(self.N) / 256.0
        p = self.A0 * np.cos(2 * np.pi * self.F0 * t + self.PHI0)
        return Field(
            data=p.reshape(1, 1, self.N),
            coords={'depth': np.array([50.0]),
                    'range': np.array([1000.0]),
                    'time': t},
            model='Test')

    def test_get_spectrum_peaks_at_the_tone_bin(self):
        freqs, X = self._tone_field().get_spectrum()
        assert X.shape == (1, 1, self.N // 2 + 1)
        k = int(np.argmax(np.abs(X[0, 0])))
        assert freqs[k] == self.F0                       # exact-bin tone
        # rFFT of A0·cos(2πft+φ0) at the tone bin is (N/2)·A0·e^{iφ0}.
        assert X[0, 0, k] == pytest.approx(
            (self.N / 2) * self.A0 * np.exp(1j * self.PHI0), rel=1e-9)

    def test_extract_tone_recovers_the_complex_amplitude(self):
        tone = self._tone_field().extract_tone(self.F0)
        assert list(tone.coords) == ['depth', 'range']
        assert tone.pinned['frequency'] == self.F0
        np.testing.assert_array_equal(tone.frequencies, [self.F0])
        # Phasor convention p(t) = Re{A·e^{+2πift}} → A = A0·e^{+iφ0}.
        # rel=1e-5: the symmetric (non-periodic) np.hanning taper leaks
        # ~8e-7 of the negative-frequency image into the tone bin
        # (measured); a periodic window would make this exact.
        assert complex(tone.data[0, 0]) == pytest.approx(
            self.A0 * np.exp(1j * self.PHI0), rel=1e-5)


class TestFieldDomainAccessorContracts:
    """The documented unit/domain guards on the value accessors
    (docs/guide/results.md §4): ``.db`` refuses a time-domain trace,
    ``.p`` refuses real data, and ``.dt``/``.sample_rate`` read 0.0 when
    no time axis exists."""

    def test_db_raises_on_a_time_domain_field(self):
        trace = Field(data=np.zeros((1, 1, 8)),
                      coords={'depth': np.array([10.0]),
                              'range': np.array([100.0]),
                              'time': np.arange(8) * 0.01},
                      model='Test')
        with pytest.raises(AttributeError, match='time-domain'):
            trace.db

    def test_p_raises_on_a_real_db_field(self):
        tl = Field(data=np.array([[60.0]]),
                   coords={'depth': np.array([10.0]),
                           'range': np.array([100.0])},
                   model='Test', frequencies=100.0)
        with pytest.raises(AttributeError, match='data is real'):
            tl.p

    def test_dt_and_sample_rate_are_zero_without_a_time_axis(self):
        tl = Field(data=np.array([[60.0]]),
                   coords={'depth': np.array([10.0]),
                           'range': np.array([100.0])},
                   model='Test', frequencies=100.0)
        assert tl.dt == 0.0
        assert tl.sample_rate == 0.0

    def test_dt_and_sample_rate_read_the_time_axis(self):
        trace = Field(data=np.zeros((1, 1, 50)),
                      coords={'depth': np.array([10.0]),
                              'range': np.array([100.0]),
                              'time': np.arange(50) * 0.01},
                      model='Test')
        assert trace.dt == pytest.approx(0.01)
        assert trace.sample_rate == pytest.approx(100.0)


class TestResultReprSnapshots:
    """Pins the stable ``__repr__`` format of the sparse results and
    :class:`ResultStack`: ``Cls(model=..., f=... | n_f=..., <size extra>)``.
    Each object below is fully hand-built, so the whole string is
    deterministic."""

    def test_rays_repr(self):
        from uacpy.core.results import Rays
        fan = Rays(rays=[{'r': [0.0], 'z': [0.0]}] * 2,
                   model='Bellhop', frequencies=100.0)
        assert repr(fan) == "Rays(model='Bellhop', f=100 Hz, n_rays=2)"

    def test_eigenrays_repr_names_eigenrays(self):
        from uacpy.core.results import Rays
        eig = Rays(rays=[{'r': [0.0], 'z': [0.0]}], is_eigen=True,
                   model='Bellhop', frequencies=100.0)
        assert repr(eig) == "Rays(model='Bellhop', f=100 Hz, n_eigenrays=1)"

    def test_arrivals_repr(self):
        from uacpy.core.results import Arrivals
        arr = Arrivals(arrivals=[{'delay': 0.1}, {'delay': 0.2}],
                       receiver_depths=np.array([50.0]),
                       receiver_ranges=np.array([1000.0]),
                       model='Bellhop', frequencies=100.0)
        assert repr(arr) == "Arrivals(model='Bellhop', f=100 Hz, n_arrivals=2)"

    def test_modes_repr(self):
        from uacpy.core.results import Modes
        m = Modes(k=np.array([0.1 + 0j, 0.2 + 0j]),
                  phi=np.zeros((3, 2)),
                  depths=np.array([0.0, 50.0, 100.0]),
                  model='Kraken', frequencies=25.0)
        assert repr(m) == "Modes(model='Kraken', f=25 Hz, n_modes=2, n_z=3)"

    def test_reflection_coefficient_repr_narrowband(self):
        rc = ReflectionCoefficient(
            theta=np.array([0.0, 45.0, 90.0]),
            R=np.array([1.0, 0.5, 0.2]),
            phi=np.zeros(3),
            model='Bounce', frequencies=50.0)
        assert repr(rc) == (
            "ReflectionCoefficient(model='Bounce', f=50 Hz, "
            "n_θ=3, narrowband)")

    def test_reflection_coefficient_repr_broadband_counts_frequencies(self):
        rc = ReflectionCoefficient(
            theta=np.array([0.0, 45.0, 90.0]),
            R=np.full((3, 2), 0.5),
            phi=np.zeros((3, 2)),
            model='OASR', frequencies=np.array([50.0, 100.0]))
        assert repr(rc) == (
            "ReflectionCoefficient(model='OASR', n_f=2, n_θ=3, broadband)")

    def test_result_stack_repr(self):
        from uacpy.core.results import ResultStack
        def slab(z):
            return Field(data=np.ones((1, 1), dtype=complex),
                         coords={'depth': np.array([10.0]),
                                 'range': np.array([100.0])},
                         model='Test', frequencies=100.0,
                         source_depths=np.array([z]))
        stack = ResultStack(slabs=[slab(10.0), slab(20.0)],
                            coordinate=[10.0, 20.0])
        assert repr(stack) == (
            "ResultStack[Field](n_slabs=2, source_depth=[10.0, 20.0])")


class TestRaysFilterAndSortHelpers:
    """Pure-data filtering/sorting on hand-built ray fans (no solver):
    ``filter_by_miss_distance`` / ``sorted_by_miss`` / ``filter_nfirst``,
    plus ``Arrivals.sorted_by_amplitude``."""

    @staticmethod
    def _fan(**kwargs):
        from uacpy.core.results import Rays
        # Each polyline has a vertex exactly at r=1000, so the closest
        # approach to the (1000, 50) target is the plain depth offset:
        # 3 m, 10 m and 30 m for alpha 1, 2 and 3 respectively.
        def ray(alpha, z_end):
            return {'r': np.array([0.0, 500.0, 1000.0]),
                    'z': np.array([10.0, 30.0, z_end]),
                    'alpha': alpha, 'n_top_bounces': 0, 'n_bot_bounces': 0}
        return Rays(rays=[ray(1.0, 47.0), ray(2.0, 60.0), ray(3.0, 80.0)],
                    model='Bellhop', frequencies=100.0, **kwargs)

    def test_filter_by_miss_distance_keeps_and_annotates(self):
        kept = self._fan().filter_by_miss_distance(
            5.0, target_range_m=1000.0, target_depth_m=50.0)
        assert [r['alpha'] for r in kept.rays] == [1.0]
        assert kept.rays[0]['miss_distance_m'] == pytest.approx(3.0,
                                                                abs=1e-12)
        # The parent's ray dicts are not annotated in place.
        assert 'miss_distance_m' not in self._fan().rays[0]

    def test_sorted_by_miss_orders_ascending(self):
        fan = self._fan()
        # Shuffle so the sort has work to do.
        fan.rays.reverse()
        ordered = fan.sorted_by_miss(target_range_m=1000.0,
                                     target_depth_m=50.0)
        assert [r['alpha'] for r in ordered.rays] == [1.0, 2.0, 3.0]
        np.testing.assert_allclose(
            [r['miss_distance_m'] for r in ordered.rays], [3.0, 10.0, 30.0],
            atol=1e-12)

    def test_sorted_by_miss_defaults_to_single_point_receiver_context(self):
        fan = self._fan(receiver_depths=np.array([50.0]),
                        receiver_ranges=np.array([1000.0]))
        ordered = fan.sorted_by_miss()
        assert [r['alpha'] for r in ordered.rays] == [1.0, 2.0, 3.0]

    def test_miss_helpers_without_target_or_context_raise(self):
        with pytest.raises(ConfigurationError, match='target_range_m'):
            self._fan().sorted_by_miss()

    def test_filter_nfirst_keeps_the_first_n_in_order(self):
        first_two = self._fan().filter_nfirst(2)
        assert [r['alpha'] for r in first_two.rays] == [1.0, 2.0]
        assert isinstance(first_two, type(self._fan()))
        # Composes with the sorters: the 2 closest rays.
        closest_two = self._fan().sorted_by_miss(
            target_range_m=1000.0, target_depth_m=50.0).filter_nfirst(2)
        assert [r['alpha'] for r in closest_two.rays] == [1.0, 2.0]

    def test_arrivals_sorted_by_amplitude_both_directions(self):
        from uacpy.core.results import Arrivals
        arr = Arrivals(
            arrivals=[{'delay': 0.1, 'amplitude': 0.5},
                      {'delay': 0.2, 'amplitude': 1.0},
                      {'delay': 0.3, 'amplitude': 0.2}],
            receiver_depths=np.array([50.0]),
            receiver_ranges=np.array([1000.0]),
            model='Bellhop', frequencies=100.0)
        down = arr.sorted_by_amplitude()
        assert [a['amplitude'] for a in down.arrivals] == [1.0, 0.5, 0.2]
        up = arr.sorted_by_amplitude(descending=False)
        assert [a['amplitude'] for a in up.arrivals] == [0.2, 0.5, 1.0]
        # The original order is untouched (sorts return copies).
        assert [a['amplitude'] for a in arr.arrivals] == [0.5, 1.0, 0.2]


class TestInterpolatedSlicesRequireAMonotonicAxis:
    """``eval`` brackets its query by binary search over the coordinate
    vector, which is only meaningful on a monotonic axis: ascending and
    descending axes interpolate, an interleaved one is refused by name."""

    def _field(self, depths):
        depths = np.asarray(depths, dtype=float)
        ranges = np.array([0.0, 100.0])
        data = np.repeat(depths[:, None], ranges.size, axis=1)
        return Field(data=data, coords={'depth': depths, 'range': ranges},
                     model='Test')

    def test_a_non_monotonic_depth_axis_refuses_an_interpolated_slice(self):
        f = self._field([0.0, 50.0, 25.0, 75.0])
        with pytest.raises(ConfigurationError, match="'depth'"):
            f.eval(depth=30.0)

    def test_an_ascending_axis_interpolates(self):
        f = self._field([0.0, 25.0, 50.0, 75.0])
        got = f.eval(depth=37.5)
        np.testing.assert_allclose(got.data, 37.5)

    def test_a_descending_axis_interpolates(self):
        f = self._field([75.0, 50.0, 25.0, 0.0])
        got = f.eval(depth=37.5)
        np.testing.assert_allclose(got.data, 37.5)


class TestFromStringRefusesNonStrings:
    """``BoundaryType.from_string`` / ``AttenuationUnits.from_string`` refuse
    a non-string, non-enum input as a ``ConfigurationError`` naming the
    received type, and the carrier-level ``acoustic_type`` message keeps its
    own wording."""

    @pytest.mark.parametrize("bad", [2, None])
    def test_boundary_type_names_the_received_type(self, bad):
        with pytest.raises(ConfigurationError,
                           match=type(bad).__name__):
            BoundaryType.from_string(bad)

    @pytest.mark.parametrize("bad", [2, None])
    def test_attenuation_units_names_the_received_type(self, bad):
        with pytest.raises(ConfigurationError,
                           match=type(bad).__name__):
            AttenuationUnits.from_string(bad)

    def test_the_carrier_message_names_the_bad_acoustic_type(self):
        with pytest.raises(ConfigurationError,
                           match=r"acoustic_type=2 is not recognized"):
            BoundaryProperties(acoustic_type=2)

    def test_the_lowercase_m_is_refused_before_the_uppercasing_lookup(self):
        """``'m'`` and ``'M'`` are two different AT units, and the lookup
        below upper-cases — so only an explicit guard keeps ``'m'`` from
        silently becoming ``DB_PER_M``. Both sides of that case boundary."""
        with pytest.raises(ConfigurationError, match="has no enum member"):
            AttenuationUnits.from_string('m')
        assert AttenuationUnits.from_string('M') is AttenuationUnits.DB_PER_M

    def test_from_string_is_documented_as_having_no_in_package_caller(self):
        """The reader's question at an unreferenced public parser is "what
        calls this?", and the answer — nothing, because every writer hardwires
        ``TOPOPT(3:3)='W'`` — is the useful one. A ``grep`` that finds only
        this test is otherwise indistinguishable from dead code."""
        import ast
        from pathlib import Path

        import uacpy
        package = Path(uacpy.__file__).resolve().parent
        callers = []
        for path in sorted(package.rglob('*.py')):
            if 'third_party' in path.parts or 'tests' in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == 'from_string'
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == 'AttenuationUnits'):
                    callers.append(f"{path.relative_to(package)}:{node.lineno}")
        prose = AttenuationUnits.from_string.__doc__ or ''
        if callers:
            assert 'No uacpy API takes an attenuation unit' not in prose, (
                f"the docstring says nothing calls this, but {callers} do")
        else:
            assert 'No uacpy API takes an attenuation unit' in prose


class TestEnvironmentLongitudeNormalisation:
    """A geolocation longitude is accepted in either sign convention up to
    one full wrap and stored normalised into [-180, 180); beyond a full wrap
    it is refused."""

    def test_a_lon_beyond_180_stores_its_wrapped_equivalent(self):
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          location=(10.0, 250.0))
        assert env.location[0] == pytest.approx(10.0)
        assert env.location[1] == pytest.approx(-110.0)

    def test_a_transect_across_the_antimeridian_gets_the_wrapped_midpoint(self):
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          transect=((0.0, 170.0), (0.0, -160.0)))
        assert env.location[0] == pytest.approx(0.0)
        assert env.location[1] == pytest.approx(-175.0)

    def test_an_in_range_longitude_is_stored_bit_exactly(self):
        # ==, not approx: ((lon + 180) % 360) - 180 returns -6.199999999999989
        # for -6.2, so the wrap runs only on values outside [-180, 180) and a
        # stored coordinate compares equal to the pair the caller passed.
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          location=(10.0, -6.2))
        assert env.location[1] == -6.2

    def test_a_lon_beyond_a_full_wrap_is_rejected(self):
        with pytest.raises(ConfigurationError, match="longitude"):
            Environment(bathymetry=100.0, ssp=1500.0,
                        location=(10.0, 9999.0))


class TestFieldCoordVectorsMustBeFinite:
    """A coord vector is rejected at construction if any element is NaN or
    inf: a non-finite coordinate turns every ``|axis - label|`` distance at
    that sample into NaN, so ``at()``'s argmin can land on it and return a
    sample no label ever named."""

    def _coords(self, depth_axis):
        return {'depth': np.asarray(depth_axis, dtype=float),
                'range': np.array([100.0, 200.0])}

    def test_nan_coordinate_is_rejected_at_construction(self):
        with pytest.raises(ConfigurationError,
                           match=r"Field\.coords\['depth'\] must be finite"):
            Field(data=np.zeros((3, 2)),
                  coords=self._coords([0.0, np.nan, 20.0]))

    def test_inf_coordinate_is_rejected_at_construction(self):
        with pytest.raises(ConfigurationError,
                           match=r"Field\.coords\['depth'\] must be finite"):
            Field(data=np.zeros((3, 2)),
                  coords=self._coords([0.0, np.inf, 20.0]))

    def test_finite_coords_construct_and_at_picks_the_nearest_sample(self):
        f = Field(data=np.arange(6.0).reshape(3, 2),
                  coords=self._coords([0.0, 10.0, 20.0]))
        assert f.at(depth=19.0).pinned['depth'] == pytest.approx(20.0)


class TestEnvironmentCoerceDispatchesRejectBool:
    """``True``/``False`` reach the scalar dispatch arm as numbers (bool is
    an int subclass; ``np.ndim(True) == 0``), where they would mean a 1 or
    0 m/s ocean, a 1 or 0 m/s half-space, or a 1 m deep seafloor. All three
    coerce dispatches refuse bools with a typed error instead."""

    @pytest.mark.parametrize("value", [True, False])
    def test_ssp_bool_raises_a_typed_error(self, value):
        with pytest.raises(ConfigurationError, match="bool"):
            SoundSpeedProfile.coerce(value, depth_max=100.0)

    @pytest.mark.parametrize("value", [True, False])
    def test_bottom_bool_raises_a_typed_error(self, value):
        with pytest.raises(ConfigurationError, match="bool"):
            Environment._coerce_bottom(value)

    @pytest.mark.parametrize("value", [True, False, np.True_])
    def test_bathymetry_bool_raises_a_typed_error(self, value):
        with pytest.raises(ConfigurationError, match="bool"):
            Bathymetry.coerce(value)

    def test_numeric_scalars_coerce_on_all_three_axes(self):
        assert SoundSpeedProfile.coerce(
            1480, depth_max=50.0).data[0, 0] == pytest.approx(1480.0)
        bottom = Environment._coerce_bottom(1700)
        assert bottom.halfspace_at(
            range=0.0).sound_speed == pytest.approx(1700.0)
        assert Bathymetry.coerce(120).depth == pytest.approx(120.0)

    @pytest.mark.parametrize('spelling', [
        1500.0, 1500, np.float64(1500.0), np.int64(1500),
        np.array(1500.0), np.array(1500),
    ], ids=['float', 'int', 'np.float64', 'np.int64', '0d-float-array',
            '0d-int-array'])
    def test_every_scalar_spelling_reaches_isovelocity(self, spelling):
        """``Environment(ssp=np.array(1500.0))`` failed while
        ``Environment(bathymetry=np.array(200.0))`` succeeded in the same
        constructor call: a 0-d ndarray matches no ``isinstance`` in the
        scalar chain and fell through to ``from_pairs``, which complained
        about an ``(N, 2)`` shape the caller never asked for."""
        profile = SoundSpeedProfile.coerce(spelling, depth_max=50.0)
        assert profile.data[0, 0] == pytest.approx(1500.0)
        env = uacpy.Environment(bathymetry=200.0, ssp=spelling)
        assert env.ssp.data[0, 0] == pytest.approx(1500.0)

    @pytest.mark.parametrize('spelling', [
        np.array('abc'), np.array(object(), dtype=object),
    ], ids=['0d-string-array', '0d-object-array'])
    def test_a_non_numeric_zero_d_array_raises_a_typed_error(self, spelling):
        # The other side of the same branch: admitting the numeric 0-d
        # spelling must not route a string or an object into float().
        with pytest.raises(ConfigurationError, match='0-d array'):
            SoundSpeedProfile.coerce(spelling, depth_max=50.0)

    def test_a_zero_d_bool_array_is_refused_like_a_bool(self):
        with pytest.raises(ConfigurationError, match='is a bool'):
            SoundSpeedProfile.coerce(np.array(True), depth_max=50.0)

    def test_a_one_d_array_goes_to_from_pairs(self):
        # The dimension boundary: 0-d is a scalar, 1-d and above are pairs.
        with pytest.raises(ConfigurationError, match=r'shape \(N, 2\)'):
            SoundSpeedProfile.coerce(np.array([1500.0, 1490.0]),
                                     depth_max=50.0)
        profile = SoundSpeedProfile.coerce(
            np.array([[0.0, 1500.0], [50.0, 1490.0]]), depth_max=50.0)
        assert profile.data[0, 0] == pytest.approx(1500.0)


class TestResultStackDbRefusesNonDbSlabs:
    """``ResultStack.db`` raises the stack's typed error for slabs whose
    real data is not a level (unit other than ``'dB'``), while complex
    slabs — whose dB view is derived — still stack."""

    def _slab(self, data, meta=None):
        return Field(data=data,
                     coords={'depth': np.array([1.0, 2.0]),
                             'range': np.array([10.0, 20.0, 30.0])},
                     metadata=meta)

    def _stack(self, slabs):
        return ResultStack(slabs, np.array([5.0, 10.0]),
                           coordinate_name='source_depth')

    def test_dimensionless_slabs_raise_the_stack_typed_error(self):
        pd = self._slab(np.full((2, 3), 0.5),
                        meta={'kind': 'probability_of_detection',
                              'unit': '1'})
        with pytest.raises(ConfigurationError,
                           match=r"ResultStack\.db: slabs are in '1', not dB"):
            self._stack([pd, pd]).db

    def test_complex_pressure_slabs_stack_to_a_db_view(self):
        stack = self._stack([self._slab(np.full((2, 3), 1j)),
                             self._slab(np.full((2, 3), 1j))])
        assert stack.db.shape == (2, 2, 3)
        assert np.allclose(stack.db, 0.0)

    def test_time_domain_slabs_raise_the_stack_typed_error(self):
        trace = Field(data=np.zeros(4),
                      coords={'time': np.arange(4.0)})
        with pytest.raises(ConfigurationError, match="time-domain slabs"):
            ResultStack([trace, trace], np.array([5.0, 10.0]),
                        coordinate_name='source_depth').db


class TestSynthesisWarnsWhenNoSpeedStamped:
    """A synthesis window anchored with no producer-stamped speed
    (neither ``'c_max'`` nor ``'c0'`` in metadata) warns that the
    1500 m/s default is the anchor whenever the anchor delays the window
    start. The geometry here keeps the fast/slow-spread heuristic below
    its own threshold (5 % of the travel time is under the half-window
    lead), so the warning is pinned to the unstamped-speed rule alone."""

    def _broadband(self, meta):
        freqs = np.linspace(50.0, 59.0, 10)
        return Field(data=np.ones((1, 1, 10), dtype=complex),
                     coords={'depth': np.array([50.0]),
                             'range': np.array([3000.0]),
                             'frequency': freqs},
                     metadata=meta)

    def _trace_warnings(self, meta, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trace = self._broadband(meta).to_time_trace(
                depth=50.0, range=3000.0, **kwargs)
        return trace, [str(w.message) for w in caught
                       if 'to_time_trace' in str(w.message)]

    def test_unstamped_speed_warns_of_the_default_anchor(self):
        _, messages = self._trace_warnings({})
        assert len(messages) == 1
        assert "stamped no sound speed" in messages[0]
        assert "1500" in messages[0]

    def test_stamped_c_max_anchors_silently(self):
        trace, messages = self._trace_warnings({'c_max': 3000.0})
        assert messages == []
        assert float(trace.coords['time'][0]) == pytest.approx(0.5)

    def test_stamped_c0_carries_no_unstamped_speed_warning(self):
        _, messages = self._trace_warnings({'c0': 1500.0})
        assert all("stamped no sound speed" not in m for m in messages)

    def test_explicit_t_start_silences_the_anchor_warning(self):
        trace, messages = self._trace_warnings({}, t_start=1.0)
        assert messages == []
        assert float(trace.coords['time'][0]) == pytest.approx(1.0)


def _profile_1d():
    return SoundSpeedProfile.from_pairs([[0.0, 1500.0], [100.0, 1490.0]])


class TestSliceLabelsMustBeFiniteScalars:
    """``collapse_axis`` rejects NaN/inf and array-valued labels with a
    typed error, on every carrier that routes ``at``/``eval`` through it.
    A NaN label makes every ``|axis - label|`` distance NaN, so a nearest
    lookup would fall to index 0 — a real sample — and an interpolated
    slice would propagate NaN into the result, whose own validators then
    blame the data."""

    def test_ssp_at_nan_depth_raises_a_typed_label_error(self):
        with pytest.raises(ConfigurationError,
                           match="depth=nan is not a finite label"):
            _profile_1d().at(depth=np.nan)

    def test_ssp_eval_nan_depth_blames_the_label_not_the_data(self):
        with pytest.raises(ConfigurationError) as exc:
            _profile_1d().eval(depth=float('nan'))
        assert "not a finite label" in str(exc.value)
        assert "sound speeds must be finite" not in str(exc.value)

    def test_ssp_eval_array_depth_raises_a_typed_scalar_label_error(self):
        with pytest.raises(ConfigurationError,
                           match="is not a scalar label"):
            _profile_1d().eval(depth=[10.0, 20.0])

    def test_ssp_eval_nan_range_on_a_1d_profile_raises(self):
        with pytest.raises(ConfigurationError,
                           match="range=nan is not a finite label"):
            _profile_1d().eval(range=np.nan)

    def test_field_eval_inf_label_raises_a_typed_label_error(self):
        f = Field(data=np.arange(6.0).reshape(3, 2),
                  coords={'depth': np.array([0.0, 10.0, 20.0]),
                          'range': np.array([100.0, 200.0])})
        with pytest.raises(ConfigurationError,
                           match="depth=inf is not a finite label"):
            f.eval(depth=np.inf)

    def test_reflection_at_nan_angle_raises_a_typed_label_error(self):
        rc = ReflectionCoefficient(
            theta=np.linspace(0.0, 90.0, 5),
            R=np.linspace(0.0, 1.0, 5),
            phi=np.zeros(5),
            model='Test', frequencies=100.0,
        )
        with pytest.raises(ConfigurationError,
                           match="angle=nan is not a finite label"):
            rc.at(angle=np.nan)

    def test_finite_scalar_labels_slice_nearest_and_interpolated(self):
        ssp = _profile_1d()
        assert float(ssp.at(depth=99.0).depths[0]) == pytest.approx(100.0)
        assert ssp.eval(depth=50.0).value == pytest.approx(1495.0)


# Every coordinate array a core carrier casts with
# ``np.array(..., dtype=float)``, as
# ``(message label, ctor, field, complex samples, real samples, siblings)``.
# The four carriers are ``Source``, ``Receiver``, ``SoundSpeedProfile`` and
# ``_RangeProfile`` — the last through both of its concrete subclasses, since
# the guard reads ``_VALUE_FIELD`` and so builds a different label for each.
_COMPLEX_COORDINATE_FIELDS = [
    ("source depths", uacpy.Source, 'depths',
     [50.0 + 2.0j], [50.0], dict(frequencies=100.0)),
    ("source frequencies", uacpy.Source, 'frequencies',
     [100.0 + 5.0j], [100.0], dict(depths=50.0)),
    ("receiver depths", uacpy.Receiver, 'depths',
     [50.0 + 2.0j], [50.0], dict(ranges=[1000.0])),
    ("receiver ranges", uacpy.Receiver, 'ranges',
     [1000.0 + 5.0j], [1000.0], dict(depths=[50.0])),
    ("SoundSpeedProfile.depths", SoundSpeedProfile, 'depths',
     [0.0, 100.0 + 1.0j], [0.0, 100.0], dict(data=[1500.0, 1490.0])),
    ("SoundSpeedProfile sound speeds", SoundSpeedProfile, 'data',
     [1500.0 + 2.0j, 1490.0], [1500.0, 1490.0], dict(depths=[0.0, 100.0])),
    ("SoundSpeedProfile.ranges", SoundSpeedProfile, 'ranges',
     [0.0 + 1.0j, 1000.0], [0.0, 1000.0],
     dict(depths=[0.0, 100.0],
          data=[[1500.0, 1500.0], [1490.0, 1490.0]])),
    ("Bathymetry ranges", Bathymetry, 'ranges',
     [0.0 + 1.0j, 1000.0], [0.0, 1000.0], dict(depths=[100.0, 120.0])),
    ("Bathymetry depths", Bathymetry, 'depths',
     [100.0 + 7.0j, 120.0], [100.0, 120.0], dict(ranges=[0.0, 1000.0])),
    ("Altimetry ranges", Altimetry, 'ranges',
     [0.0 + 1.0j, 1000.0], [0.0, 1000.0], dict(heights=[0.5, -0.5])),
    ("Altimetry heights", Altimetry, 'heights',
     [0.5 + 1.0j, -0.5], [0.5, -0.5], dict(ranges=[0.0, 1000.0])),
]

_COMPLEX_FIELD_IDS = [case[0] for case in _COMPLEX_COORDINATE_FIELDS]


class TestEveryCoreCarrierRejectsComplexCoordinates:
    """One typed ``ConfigurationError`` for a complex coordinate, on every
    carrier and in every container spelling.

    The float64 cast each carrier applies destroys a complex input two
    different ways — an ndarray keeps only the real part under a
    ``ComplexWarning`` that the suite's ``ignore::UserWarning`` filter hides,
    a scalar or list raises a bare ``TypeError`` from ``float()`` naming no
    field — so the guard runs ahead of the cast and both spellings are pinned
    per field, together with the real construction it lets through.
    """

    @pytest.mark.parametrize("label,ctor,field,bad,good,siblings",
                             _COMPLEX_COORDINATE_FIELDS,
                             ids=_COMPLEX_FIELD_IDS)
    def test_a_complex_ndarray_raises_a_typed_error_naming_the_field(
            self, label, ctor, field, bad, good, siblings):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with pytest.raises(ConfigurationError,
                               match=re.escape(f"{label} must be real numbers")):
                ctor(**{field: np.array(bad)}, **siblings)

    @pytest.mark.parametrize("label,ctor,field,bad,good,siblings",
                             _COMPLEX_COORDINATE_FIELDS,
                             ids=_COMPLEX_FIELD_IDS)
    def test_a_complex_list_raises_the_same_typed_error_as_an_ndarray(
            self, label, ctor, field, bad, good, siblings):
        with pytest.raises(ConfigurationError,
                           match=re.escape(f"{label} must be real numbers")):
            ctor(**{field: bad}, **siblings)

    @pytest.mark.parametrize("label,ctor,field,bad,good,siblings",
                             _COMPLEX_COORDINATE_FIELDS,
                             ids=_COMPLEX_FIELD_IDS)
    def test_a_complex_dtype_carrying_a_zero_imaginary_part_is_refused(
            self, label, ctor, field, bad, good, siblings):
        # The near side of the boundary: dtype is the criterion, because the
        # cast emits its ComplexWarning for any complex dtype regardless of
        # what the imaginary parts hold (measured on numpy 2.5).
        with pytest.raises(ConfigurationError,
                           match=re.escape(f"{label} must be real numbers")):
            ctor(**{field: np.array(good, dtype=complex)}, **siblings)

    @pytest.mark.parametrize("label,ctor,field,bad,good,siblings",
                             _COMPLEX_COORDINATE_FIELDS,
                             ids=_COMPLEX_FIELD_IDS)
    def test_a_real_float_array_constructs_and_is_stored_as_float64(
            self, label, ctor, field, bad, good, siblings):
        # The far side of the same boundary, and the guard's cost: a real
        # array reaches the cast untouched and raises no warning of any kind.
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            carrier = ctor(**{field: np.array(good, dtype=float)}, **siblings)
        assert np.asarray(getattr(carrier, field)).dtype == np.float64

    def test_a_complex_scalar_raises_a_typed_error(self):
        # The bare-scalar spelling, which only the single-value fields take.
        with pytest.raises(ConfigurationError,
                           match="source frequencies must be real"):
            Source(depths=50.0, frequencies=100.0 + 5.0j)

    def test_the_error_names_the_first_element_carrying_an_imaginary_part(self):
        # Not flat index 0: one complex element promotes the whole array, so
        # the leading sample prints as ``(10+0j)`` and names nothing the
        # caller can act on.
        with pytest.raises(ConfigurationError) as exc:
            Source(depths=[10.0, 50.0 + 2.0j, 90.0], frequencies=100.0)
        assert '(50+2j) at flat index 1 of 3 value(s)' in str(exc.value)

    def test_an_empty_complex_array_reports_its_dtype(self):
        # No element to point at, so the offence is the dtype itself; the
        # alternative is an IndexError out of the guard.
        with pytest.raises(ConfigurationError,
                           match="an empty array of dtype complex"):
            Source(depths=np.array([], dtype=complex), frequencies=100.0)


def _bathymetry():
    return Bathymetry(ranges=[0.0, 1000.0, 2000.0],
                      depths=[100.0, 200.0, 300.0])


def _tl_field(value):
    return Field(data=np.full((2, 3), value),
                 coords={'depth': np.array([10.0, 20.0]),
                         'range': np.array([100.0, 200.0, 300.0])})


def _probability_field(value):
    return Field(data=np.full((2, 3), value),
                 coords={'depth': np.array([10.0, 20.0]),
                         'range': np.array([100.0, 200.0, 300.0])},
                 metadata={'kind': 'probability_of_detection'})


class TestRangeProfileQueriesMustBeFinite:
    """``Bathymetry``/``Altimetry`` query a profile by searchsorted rather
    than through ``collapse_axis``, so a NaN or inf range walked to an end
    node and returned that node's stored value as a successful lookup. The
    query stays array-capable — ``PropagationModel`` hands it the whole
    receiver range axis — so the check is element-wise."""

    def test_bathymetry_at_nan_range_raises_a_typed_label_error(self):
        with pytest.raises(ConfigurationError,
                           match="range=nan is not a finite label"):
            _bathymetry().at(range=np.nan)

    @pytest.mark.parametrize('bad', [np.inf, -np.inf])
    def test_bathymetry_at_infinite_range_raises(self, bad):
        with pytest.raises(ConfigurationError, match="not a finite label"):
            _bathymetry().at(range=bad)

    def test_bathymetry_eval_nan_range_raises(self):
        with pytest.raises(ConfigurationError, match="not a finite label"):
            _bathymetry().eval(range=float('nan'))

    def test_bathymetry_eval_rejects_an_array_containing_one_nan(self):
        with pytest.raises(ConfigurationError) as exc:
            _bathymetry().eval(range=np.array([0.0, np.nan, 2000.0]))
        assert "1 of 3 query ranges" in str(exc.value)

    def test_bathymetry_eval_accepts_a_finite_array_of_ranges(self):
        out = _bathymetry().eval(range=np.array([0.0, 1500.0]))
        assert isinstance(out, np.ndarray)
        assert np.allclose(out, [100.0, 250.0])

    def test_bathymetry_at_a_finite_range_returns_the_nearest_depth(self):
        assert _bathymetry().at(range=900.0) == 200.0

    def test_altimetry_at_nan_range_raises(self):
        alt = Altimetry(ranges=[0.0, 1000.0], heights=[0.0, 1.0])
        with pytest.raises(ConfigurationError, match="not a finite label"):
            alt.at(range=np.nan)


class TestResultStackAndTraceLabelsMustBeFinite:
    """``ResultStack.at`` and the single-cell IFFT pick their slab / cell by
    ``argmin``, where ``Field.at`` routes through ``collapse_axis`` and
    already raised — the same query answered two different ways."""

    def _stack(self):
        return ResultStack(slabs=[_tl_field(10.0), _tl_field(20.0)],
                           coordinate=np.array([5.0, 50.0]),
                           coordinate_name='source_depth')

    def test_stack_at_nan_coordinate_raises_naming_the_stacking_axis(self):
        with pytest.raises(ConfigurationError,
                           match="source_depth=nan is not a finite label"):
            self._stack().at(source_depth=np.nan)

    def test_stack_at_inf_coordinate_raises(self):
        with pytest.raises(ConfigurationError, match="not a finite label"):
            self._stack().at(source_depth=np.inf)

    def test_stack_at_a_finite_coordinate_returns_the_nearest_slab(self):
        stack = self._stack()
        assert stack.at(source_depth=40.0) is stack.slabs[1]

    def _broadband(self):
        n_freq = 64
        data = np.zeros((3, 2, n_freq), dtype=complex)
        data[1, 0, :] = 1.0
        return Field(data=data,
                     coords={'depth': np.array([10.0, 20.0, 30.0]),
                             'range': np.array([1000.0, 2000.0]),
                             'frequency': np.linspace(100.0, 500.0, n_freq)})

    def test_to_time_trace_nan_depth_raises_a_typed_label_error(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ConfigurationError,
                               match="depth=nan is not a finite label"):
                self._broadband().to_time_trace(depth=np.nan)

    def test_to_time_trace_inf_range_raises(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ConfigurationError, match="not a finite label"):
                self._broadband().to_time_trace(range=np.inf)

    def test_to_time_trace_pins_the_nearest_cell_for_a_finite_label(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            trace = self._broadband().to_time_trace(depth=19.0, range=1900.0)
        assert trace.pinned == {'depth': 20.0, 'range': 2000.0}

    def test_to_time_trace_without_labels_takes_the_middle_depth(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            trace = self._broadband().to_time_trace()
        assert trace.pinned == {'depth': 20.0, 'range': 1000.0}


class TestPhaseReferenceIsValidatedAtIngest:
    """A value outside :class:`PhaseReference` compares unequal to
    ``'time_domain_native'`` and so walks straight through the IFFT guard
    that reads it. The value is checked for membership but stored exactly as
    passed, so a plain string stays a plain string."""

    def test_an_unknown_phase_reference_raises_a_typed_error(self):
        with pytest.raises(ConfigurationError,
                           match="not a known phase convention"):
            Result(model='X', phase_reference='travelling_wve')

    def test_the_error_names_the_accepted_values(self):
        with pytest.raises(ConfigurationError) as exc:
            Field(data=np.zeros((2, 3)),
                  coords={'depth': np.array([10.0, 20.0]),
                          'range': np.array([1.0, 2.0, 3.0])},
                  phase_reference='travelling')
        assert 'travelling_wave' in str(exc.value)
        assert 'time_domain_native' in str(exc.value)

    def test_a_plain_string_is_stored_without_coercion(self):
        result = Result(model='X', phase_reference='travelling_wave')
        assert type(result.phase_reference) is str
        assert str(result.phase_reference).endswith('travelling_wave')

    def test_an_enum_member_is_stored_without_conversion(self):
        result = Result(model='X',
                        phase_reference=PhaseReference.TRAVELLING_WAVE)
        assert result.phase_reference is PhaseReference.TRAVELLING_WAVE

    def test_none_is_accepted(self):
        assert Result(model='X').phase_reference is None

    def test_a_stored_value_survives_the_field_dict_round_trip(self):
        field = Field(data=np.zeros((2, 3)),
                      coords={'depth': np.array([10.0, 20.0]),
                              'range': np.array([1.0, 2.0, 3.0])},
                      phase_reference='time_domain_native')
        assert (Field.from_dict(field.to_dict()).phase_reference
                == 'time_domain_native')


class TestBeamPatternAnglesCarryTheSbpResolution:
    """``write_source_beam_pattern`` prints the angle column at ``%12.6f``
    and refuses a pair closer than 1e-6 deg; the carrier now refuses the
    same pattern at construction rather than at write."""

    def test_angles_closer_than_the_sbp_resolution_are_rejected(self):
        with pytest.raises(ConfigurationError,
                           match="must increase by more than"):
            Source(depths=[10.0], frequencies=[100.0],
                   beam_pattern=[[0.0, 0.0], [1e-9, -3.0], [10.0, -6.0]])

    def test_the_angle_step_is_reported_in_degrees(self):
        with pytest.raises(ConfigurationError) as exc:
            Source(depths=[10.0], frequencies=[100.0],
                   beam_pattern=[[0.0, 0.0], [1e-9, -3.0]])
        assert 'deg apart' in str(exc.value)
        assert 'm apart' not in str(exc.value)

    def test_a_metre_axis_keeps_reporting_metres(self):
        with pytest.raises(ConfigurationError) as exc:
            Source(depths=[25.0, 25.0 + 1e-7], frequencies=[200.0])
        assert 'm apart' in str(exc.value)

    def test_a_resolvable_pattern_is_accepted(self):
        src = Source(depths=[10.0], frequencies=[100.0],
                     beam_pattern=[[-90.0, -20.0], [0.0, 0.0], [90.0, -20.0]])
        assert src.beam_pattern.shape == (3, 2)

    def test_the_constant_matches_the_sbp_writers_bound(self):
        assert SBP_ANGLE_RESOLUTION_DEG == 1e-6

    def test_a_step_of_exactly_the_resolution_is_rejected_by_the_carrier(self):
        # Source.__post_init__ asks _require_strictly_increasing for steps
        # greater than the resolution, so the carrier refuses the one step
        # write_source_beam_pattern accepts (its own guard is `< resolution`).
        # The two conventions differ by exactly this value, deliberately: the
        # carrier is the stricter of the two.
        with pytest.raises(ConfigurationError,
                           match="must increase by more than"):
            Source(depths=[10.0], frequencies=[100.0],
                   beam_pattern=[[0.0, 0.0], [SBP_ANGLE_RESOLUTION_DEG, -3.0]])

    def test_a_step_just_above_the_resolution_is_accepted(self):
        src = Source(depths=[10.0], frequencies=[100.0],
                     beam_pattern=[[0.0, 0.0],
                                   [2 * SBP_ANGLE_RESOLUTION_DEG, -3.0]])
        assert src.beam_pattern[1, 0] == pytest.approx(2e-6)


def _field(data=None, **meta):
    depths = np.array([0.0, 10.0, 20.0, 30.0])
    ranges = np.array([100.0, 200.0, 300.0])
    if data is None:
        data = np.arange(12.0).reshape(4, 3) + 1.0
    return Field(data=data, coords={'depth': depths, 'range': ranges},
                 model='Test', **meta)


class TestMaskBelowSeafloorValidatesTheRangeAxis:
    """``mask_below_seafloor`` hands its bathymetry to ``np.interp``, which
    takes ``xp`` on trust. A raw ``(N, 2)`` array skipped the check the
    ``Bathymetry`` form gets, so a profile whose range column does not
    increase interpolated against a broken axis and masked the wrong cells
    with no error: the same two-point profile masked 24 cells sorted and 28
    reversed."""

    def test_a_reversed_range_column_is_refused(self):
        rows = np.array([[4000.0, 150.0], [0.0, 100.0]])
        with pytest.raises(ConfigurationError, match='strictly increasing'):
            _field().mask_below_seafloor(rows)

    def test_a_repeated_range_is_refused(self):
        rows = np.array([[0.0, 100.0], [0.0, 150.0]])
        with pytest.raises(ConfigurationError, match='strictly increasing'):
            _field().mask_below_seafloor(rows)

    def test_the_shape_error_comes_from_the_field_method(self):
        """The (N, 2) shape check stays where the caller can see which
        argument it is about, ahead of the bathymetry axis check."""
        with pytest.raises(ConfigurationError, match='mask_below_seafloor'):
            _field().mask_below_seafloor(np.ones((3, 3)))

    def test_a_sorted_profile_masks_what_it_did_before(self):
        f = _field()
        rows = [(0.0, 15.0), (300.0, 25.0)]
        masked = np.asarray(f.mask_below_seafloor(rows).data)
        seafloor = np.interp(f.coords['range'], [0.0, 300.0], [15.0, 25.0])
        want = f.coords['depth'][:, None] > seafloor[None, :]
        np.testing.assert_array_equal(np.isnan(masked), want)

    def test_the_bathymetry_carrier_form_masks_the_same_cells_as_an_array(self):
        from uacpy.core.bathymetry import Bathymetry
        rows = np.array([[0.0, 15.0], [300.0, 25.0]])
        via_array = np.asarray(_field().mask_below_seafloor(rows).data)
        via_carrier = np.asarray(
            _field().mask_below_seafloor(Bathymetry.coerce(rows)).data)
        np.testing.assert_array_equal(np.isnan(via_array),
                                      np.isnan(via_carrier))


class TestFieldAtRejectsALabelItCannotRank:
    """``Field.at`` picks ``argmin(|coord - label|)``. A NaN or inf label
    makes every distance NaN or inf and argmin falls to index 0; a label
    large enough to absorb the whole axis (``|z - 1e300|`` rounds to 1e300
    for every z) ties every sample and does the same. Index 0 is a real
    sample, so both read as a successful slice at the first coordinate."""

    def test_a_nan_label_is_refused(self):
        with pytest.raises(ConfigurationError, match='not a finite label'):
            _field().at(depth=float('nan'))

    @pytest.mark.parametrize('label', [float('inf'), float('-inf')])
    def test_an_infinite_label_is_refused(self, label):
        with pytest.raises(ConfigurationError, match='not a finite label'):
            _field().at(range=label)

    @pytest.mark.parametrize('label', [1e300, -1e300])
    def test_a_label_that_absorbs_the_axis_is_refused(self, label):
        with pytest.raises(ConfigurationError, match='same distance'):
            _field().at(depth=label)

    def test_an_ordinary_out_of_range_label_clamps_to_the_nearest(self):
        """Outside the axis but still rankable is the documented
        nearest-sample behaviour and stays."""
        assert _field().at(depth=-5.0).pinned['depth'] == pytest.approx(0.0)
        assert _field().at(depth=999.0).pinned['depth'] == pytest.approx(30.0)

    def test_a_genuine_midpoint_tie_is_answered(self):
        """Two samples exactly equidistant is a real tie, not a lost axis:
        the first wins, as argmin has always done."""
        f = Field(data=np.ones((2, 3)),
                  coords={'depth': np.array([0.0, 10.0]),
                          'range': np.array([100.0, 200.0, 300.0])},
                  model='Test')
        assert f.at(depth=5.0).pinned['depth'] == pytest.approx(0.0)

    def test_a_single_sample_axis_takes_any_finite_label(self):
        """One sample cannot tie with another, so there is nothing to lose."""
        f = Field(data=np.ones((1, 3)),
                  coords={'depth': np.array([7.0]),
                          'range': np.array([100.0, 200.0, 300.0])},
                  model='Test')
        assert f.at(depth=1e300).pinned['depth'] == pytest.approx(7.0)


class TestToDbRewritesTheUnitTag:
    """``metadata['unit']`` describes the data, and ``to_db`` replaces the
    data. Carrying a ``'Pa'`` tag onto ``-20·log10|p|`` left a dB field
    reporting Pa, which sends ``Field.max`` down its linear branch: it then
    ranks by ``|dB|``, where the largest magnitude is the *quietest* sample
    rather than the loudest."""

    AMP = np.array([[10.0, 0.5, 2.0], [3.0, 0.2, 8.0],
                    [1.0, 4.0, 0.1], [6.0, 0.05, 20.0]])

    def _tagged(self):
        return _field(data=self.AMP * (1.0 + 0j), metadata={'unit': 'Pa'})

    def test_the_tag_follows_the_data(self):
        assert self._tagged().to_db().unit == 'dB'
        assert self._tagged().to_db().metadata['unit'] == 'dB'

    def test_the_source_field_keeps_its_own_tag(self):
        f = self._tagged()
        f.to_db()
        assert f.metadata['unit'] == 'Pa'

    def test_max_finds_the_same_point_tagged_or_not(self):
        tagged = self._tagged().to_db().max().pinned
        untagged = _field(data=self.AMP * (1.0 + 0j)).to_db().max().pinned
        assert tagged == untagged

    def test_max_finds_the_loudest_sample_and_not_the_quietest(self):
        loudest = self._tagged().to_db().max().pinned
        i, j = np.unravel_index(int(np.argmax(self.AMP)), self.AMP.shape)
        assert loudest['depth'] == pytest.approx(
            float(_field().coords['depth'][i]))
        assert loudest['range'] == pytest.approx(
            float(_field().coords['range'][j]))

    def test_an_untagged_field_gains_no_tag(self):
        out = _field(data=self.AMP * (1.0 + 0j)).to_db()
        assert 'unit' not in out.metadata
        assert out.unit == 'dB'

    def test_a_real_field_is_returned_unchanged(self):
        f = _field(metadata={'unit': 'dB'})
        assert f.to_db() is f


class TestDbRefusalOffersAnActionableRoute:
    """``Field.db``'s unit guard is reachable only for real data — the complex
    branch returns first — and ``to_db()`` returns ``self`` for every real
    field. So the set of fields that can see this message is exactly the set
    on which ``to_db()`` does nothing, and naming it as the remedy sends the
    reader in a circle."""

    def _dimensionless(self):
        return _field(data=np.full((4, 3), 0.5),
                      metadata={'kind': 'probability_of_detection',
                                'unit': '1'})

    def _linear_pressure(self):
        return _field(data=np.full((4, 3), 2.0),
                      metadata={'kind': 'pressure', 'unit': 'Pa'})

    @pytest.mark.parametrize('name', ['_dimensionless', '_linear_pressure'])
    def test_to_db_is_the_identity_on_every_field_that_reaches_the_guard(
            self, name):
        f = getattr(self, name)()
        assert not f.is_complex
        assert f.unit != 'dB'
        assert f.to_db() is f

    @pytest.mark.parametrize('name', ['_dimensionless', '_linear_pressure'])
    def test_the_message_names_an_operation_that_changes_the_values(self, name):
        f = getattr(self, name)()
        with pytest.raises(AttributeError) as excinfo:
            f.db
        message = str(excinfo.value)
        assert 'not dB' in message
        assert 'log10' in message
        # Naming to_db() is only honest alongside the fact that it is the
        # identity here.
        if 'to_db()' in message:
            assert 'unchanged' in message

    def test_a_db_tagged_real_field_is_the_other_side_of_the_guard(self):
        f = _field(data=np.full((4, 3), -60.0), metadata={'unit': 'dB'})
        assert np.allclose(f.db, -60.0)

    def test_the_stack_message_names_an_operation_that_changes_the_values(self):
        slab = self._dimensionless()
        stack = ResultStack([slab, slab], np.array([5.0, 10.0]),
                            coordinate_name='source_depth')
        with pytest.raises(ConfigurationError) as excinfo:
            stack.db
        message = str(excinfo.value)
        assert 'not dB' in message
        assert 'log10' in message
        if 'to_db()' in message:
            assert 'unchanged' in message


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


# ── the two roles of a dataclass field annotation ───────────────────────────
#
# ``Source``, ``Receiver`` and ``BoundaryProperties`` all take a wide input and
# normalize it in ``__post_init__``, so the constructor parameter and the
# attribute have different types. A dataclass writes one annotation for both
# roles, and the wider of the two used to win: ``s.depths`` was declared
# ``float | list[float] | ndarray``, so ``len(s.depths)``, ``s.depths.shape``
# and ``for d in r.depths`` were reported as errors in downstream code that
# runs correctly — on a package that ships ``py.typed`` and therefore has its
# annotations believed. The field now declares what the attribute holds, and an
# ``if TYPE_CHECKING:`` ``__init__`` carries the input union.
#
# (class, field, attribute type, constructor annotation, constructor kwargs)
_SPLIT_ANNOTATION_FIELDS = [
    (Source, 'depths', np.ndarray,
     'Union[float, List[float], np.ndarray]',
     dict(depths=50.0, frequencies=100.0)),
    (Source, 'frequencies', np.ndarray,
     'Union[float, List[float], np.ndarray]',
     dict(depths=50.0, frequencies=[100.0, 200.0])),
    (uacpy.Receiver, 'depths', np.ndarray,
     'Union[float, List[float], np.ndarray]',
     dict(depths=[10.0, 20.0], ranges=1000.0)),
    (uacpy.Receiver, 'ranges', np.ndarray,
     'Optional[Union[float, List[float], np.ndarray]]',
     dict(depths=10.0, ranges=[1000.0, 2000.0])),
    (BoundaryProperties, 'acoustic_type', str,
     'Optional[str]',
     dict(sound_speed=1600.0, density=1.8)),
]

_SPLIT_IDS = [f'{cls.__name__}.{field}'
              for cls, field, _, _, _ in _SPLIT_ANNOTATION_FIELDS]


def _type_checking_init(cls):
    """The ``__init__`` the class declares inside ``if TYPE_CHECKING:``.

    Read from source: the ``__init__`` the dataclass decorator compiles at
    runtime carries the *field* annotations, so ``inspect.signature`` shows
    the attribute half and can never see the constructor half."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(inspect.getmodule(cls)))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == cls.__name__):
            continue
        for child in node.body:
            if not isinstance(child, ast.If):
                continue
            test = child.test
            guard = (test.id if isinstance(test, ast.Name)
                     else test.attr if isinstance(test, ast.Attribute) else '')
            if not guard.endswith('TYPE_CHECKING'):
                continue
            for inner in child.body:
                if (isinstance(inner, ast.FunctionDef)
                        and inner.name == '__init__'):
                    return inner
    return None


@pytest.mark.parametrize(
    'cls, field, attribute_type, constructor_annotation, kwargs',
    _SPLIT_ANNOTATION_FIELDS, ids=_SPLIT_IDS)
def test_a_normalized_field_is_annotated_as_what_the_attribute_holds(
        cls, field, attribute_type, constructor_annotation, kwargs):
    """Both halves of the split, so neither can drift back into the other.

    The attribute annotation has to name the normalized type *and* the
    constructed object has to carry it — an annotation narrowed without the
    normalization behind it is the same defect pointing the other way."""
    import typing

    declared = typing.get_type_hints(cls)[field]
    assert declared is attribute_type, (
        f"{cls.__name__}.{field} is annotated {declared!r}, not "
        f"{attribute_type!r}: every downstream read of the attribute is "
        f"checked against the constructor's input type")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        carrier = cls(**kwargs)
    assert isinstance(getattr(carrier, field), attribute_type), (
        f"{cls.__name__}.{field} is annotated {attribute_type!r} but holds "
        f"{type(getattr(carrier, field))!r} after construction")


_SPLIT_CLASSES = [Source, uacpy.Receiver, BoundaryProperties]


@pytest.mark.parametrize('cls', _SPLIT_CLASSES,
                         ids=[c.__name__ for c in _SPLIT_CLASSES])
def test_the_type_checking_constructor_matches_the_one_python_compiles(cls):
    """The guarded ``__init__`` and the one the decorator compiles have to be
    the same function, parameter for parameter.

    Two ways they come apart, and this sees both. A field added to the
    dataclass and not to the block silently drops out of every static caller's
    view of the constructor. And the compiled ``__init__`` takes its
    annotations from the *fields*, so without the ``__annotations__`` update
    after each class ``inspect.signature`` and ``help()`` advertise a default
    the annotation refuses — ``ranges: np.ndarray = None``.
    """
    import ast
    import inspect

    initializer = _type_checking_init(cls)
    assert initializer is not None, cls.__name__
    guarded = [a.arg for a in initializer.args.args if a.arg != 'self']
    compiled = list(inspect.signature(cls).parameters)
    assert guarded == compiled, (
        f"{cls.__name__}'s `if TYPE_CHECKING:` __init__ takes {guarded} and "
        f"the compiled one takes {compiled}")

    namespace = vars(inspect.getmodule(cls))
    padding = [None] * (len(initializer.args.args)
                        - len(initializer.args.defaults))
    declared_defaults = padding + list(initializer.args.defaults)
    for argument, default in zip(initializer.args.args, declared_defaults):
        if argument.arg == 'self':
            continue
        parameter = inspect.signature(cls).parameters[argument.arg]
        assert parameter.annotation == eval(  # noqa: S307 — our own source
            ast.unparse(argument.annotation), namespace), (
            f"{cls.__name__}.__init__ advertises "
            f"{parameter.annotation!r} for {argument.arg}, not the "
            f"{ast.unparse(argument.annotation)!r} the guarded block declares")
        if default is None:
            assert parameter.default is inspect.Parameter.empty, (
                f"{cls.__name__}.__init__ gives {argument.arg} a default the "
                f"guarded block does not")
        else:
            assert repr(parameter.default) == ast.unparse(default), (
                f"{cls.__name__}.__init__ defaults {argument.arg} to "
                f"{parameter.default!r}, not {ast.unparse(default)}")


@pytest.mark.parametrize(
    'cls, field, attribute_type, constructor_annotation, kwargs',
    _SPLIT_ANNOTATION_FIELDS, ids=_SPLIT_IDS)
def test_the_constructor_declares_the_input_union_the_docstring_documents(
        cls, field, attribute_type, constructor_annotation, kwargs):
    """The other half. Narrowing the field annotation without this block would
    make ``Source(depths=50.0)`` — the spelling every docstring example and
    the whole test suite uses — a type error for a downstream caller."""
    import ast

    initializer = _type_checking_init(cls)
    assert initializer is not None, (
        f"{cls.__name__} declares no `if TYPE_CHECKING:` __init__, so its "
        f"constructor is typed from the fields and rejects the input union "
        f"its Parameters section documents")
    annotations = {arg.arg: ast.unparse(arg.annotation)
                   for arg in initializer.args.args if arg.annotation}
    assert annotations.get(field) == constructor_annotation, (
        f"{cls.__name__}.__init__ annotates {field} as "
        f"{annotations.get(field)!r}, not {constructor_annotation!r}")


def _nearest_probe_field():
    return Field(data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                 coords={'depth': [0.0, 50.0],
                         'range': [0.0, 500.0, 1000.0]},
                 model='Test')


def _bottom():
    return Bottom.from_halfspaces([0.0, 1000.0, 2000.0],
                                  sound_speed=[1600.0, 1700.0, 1800.0],
                                  density=1.8, attenuation=0.5)


class TestSharedNearestPathRefusesAnAxisAbsorbingLabel:
    """``_grid.py``'s argmin-based nearest lookups refuse a finite label so
    far outside the axis that every ``|axis - label|`` rounds to the same
    value — the refusal ``Field.at`` makes on its own argmin path. A label
    that still ranks (1e9 against a metre-scale axis) answers with the
    correct end node."""

    def test_bottom_at_refuses_the_absorbing_label(self):
        with pytest.raises(ConfigurationError, match='same distance'):
            _bottom().at(range=1e300)

    def test_bottom_at_answers_a_rankable_out_of_span_label(self):
        assert _bottom().at(range=1e9).halfspace.sound_speed == \
            pytest.approx(1800.0, rel=1e-12)

    def test_bottom_at_matches_field_at_on_the_absorbing_label(self):
        with pytest.raises(ConfigurationError) as exc_field:
            _nearest_probe_field().at(depth=1e300)
        with pytest.raises(ConfigurationError) as exc_bottom:
            _bottom().at(range=1e300)
        assert 'same distance' in str(exc_field.value)
        assert 'same distance' in str(exc_bottom.value)

    def test_surface_at_refuses_the_absorbing_label(self):
        s = Surface.coerce(
            [(0.0, BoundaryProperties(acoustic_type='vacuum')),
             (1000.0, BoundaryProperties(acoustic_type='half-space',
                                         sound_speed=340.0, density=0.0012,
                                         attenuation=0.0))])
        with pytest.raises(ConfigurationError, match='same distance'):
            s.at(range=1e300)

    def test_ssp_at_refuses_the_absorbing_depth_label(self):
        ssp = SoundSpeedProfile(depths=[0.0, 50.0, 100.0],
                                data=[1500.0, 1490.0, 1495.0])
        with pytest.raises(ConfigurationError, match='same distance'):
            ssp.at(depth=1e300)

    def test_field_eval_nearest_matches_field_at_refusal(self):
        with pytest.raises(ConfigurationError, match='same distance'):
            _nearest_probe_field().eval(depth=1e300, method='nearest')

    def test_bathymetry_searchsorted_path_answers_the_far_node(self):
        """``Bathymetry.at`` brackets by searchsorted + midpoint compare,
        which is cancellation-immune: the huge label resolves to the correct
        end node on both sides."""
        bath = Bathymetry(ranges=[0.0, 1000.0, 2000.0],
                          depths=[100.0, 200.0, 300.0])
        assert bath.at(range=1e300) == pytest.approx(300.0, rel=1e-12)
        assert bath.at(range=-1e300) == pytest.approx(100.0, rel=1e-12)

    def test_a_single_node_axis_takes_any_finite_label(self):
        """One node cannot tie with another, so there is nothing to lose."""
        b1 = Bottom.from_halfspaces([5000.0], sound_speed=[1600.0],
                                    density=1.8, attenuation=0.5)
        assert b1.at(range=1e300).halfspace.sound_speed == \
            pytest.approx(1600.0, rel=1e-12)


class TestSingleNodeRangesTravelsThroughSspSlicing:
    """A single-node ``ranges`` is a coordinate at that range —
    ``env.max_range`` reads it — so every SSP slice of a single-column
    profile carries it, the rule ``Bottom.select_range`` states for the
    same case."""

    def _ssp(self):
        return SoundSpeedProfile(depths=[0.0, 50.0, 100.0],
                                 data=[[1500.0], [1490.0], [1495.0]],
                                 ranges=[5000.0])

    def test_a_depth_only_slice_keeps_env_max_range(self):
        env = Environment(ssp=self._ssp().at(depth=50.0), bathymetry=100.0)
        assert env.max_range == pytest.approx(5000.0, rel=1e-12)

    @pytest.mark.parametrize('slicer', [
        lambda s: s.at(depth=50.0),
        lambda s: s.at(range=5000.0),
        lambda s: s.eval(depth=25.0),
        lambda s: s.isel(depth=0),
        lambda s: s.isel(range=0),
    ], ids=['at_depth', 'at_range', 'eval_depth', 'isel_depth',
            'isel_range'])
    def test_each_slice_keeps_the_single_node_ranges(self, slicer):
        out = slicer(self._ssp())
        assert out.ranges is not None
        assert float(out.ranges[0]) == pytest.approx(5000.0, rel=1e-12)

    def test_collapsing_a_range_dependent_profile_drops_ranges(self):
        """Pinning the range axis of a multi-column profile collapses it, so
        the result carries no ranges — the ``Bottom.select_range('r0')``
        counterpart."""
        rd = SoundSpeedProfile(depths=[0.0, 100.0],
                               data=[[1500.0, 1510.0], [1490.0, 1505.0]],
                               ranges=[0.0, 5000.0])
        assert rd.at(range=5000.0).ranges is None
        assert rd.isel(range=1).ranges is None


class TestArrivalDictPhaseUnit:
    """The per-arrival dict carries ``'phase'`` in degrees (the ``.arr``
    reader's unit, preserved for ``by_receiver`` parity); the class
    docstring key list says so, and the ``phases`` accessor converts to
    radians."""

    def _arrivals(self):
        cell = {'delays': [0.1], 'amplitudes': [1.0], 'phases': [180.0],
                'n_top_bounces': [0], 'n_bot_bounces': [0],
                'src_angles': [0.0], 'rcv_angles': [0.0]}
        return Arrivals(by_receiver=[[[cell]]], receiver_depths=[10.0],
                        receiver_ranges=[100.0], model='Test',
                        frequencies=100.0)

    def test_dict_phase_is_degrees_and_accessor_radians(self):
        arr = self._arrivals()
        assert arr.arrivals[0]['phase'] == pytest.approx(180.0, rel=1e-12)
        assert arr.phases[0] == pytest.approx(np.pi, rel=1e-12)

    def test_class_docstring_names_the_degree_unit_for_phase(self):
        doc = Arrivals.__doc__
        segment = doc.split('``phase``', 1)[1].split('``n_top_bounces``')[0]
        assert 'degrees' in segment


class TestTlIsTheDbViewRestrictedToPressureFields:
    """``Field.tl`` answers with exactly ``Field.db``'s values on a
    pressure-kind field — the quantity's literature name for the same
    array — and refuses any other kind, whose level view stays ``.db``."""

    @staticmethod
    def _pressure_field():
        from uacpy.core.results.field import Field
        return Field(data=np.array([[0.01 + 0.001j, 0.002 + 0.0j]]),
                     coords={'depth': [10.0], 'range': [100.0, 200.0]},
                     model='Test')

    @staticmethod
    def _reverberation_field():
        from uacpy.core.results.field import Field
        return Field(data=np.array([[35.0, 30.0]]),
                     coords={'depth': [10.0], 'range': [100.0, 200.0]},
                     model='Test',
                     metadata={'kind': 'reverberation', 'unit': 'dB'})

    def test_tl_of_complex_pressure_equals_db_and_is_a_positive_loss(self):
        f = self._pressure_field()
        assert np.array_equal(f.tl, f.db)
        assert (f.tl > 0).all()

    def test_tl_of_a_real_db_pressure_field_is_the_same_readonly_view(self):
        f = self._pressure_field()
        real = f.to_db()
        assert real.tl.base is real.data or np.shares_memory(real.tl,
                                                             real.data)
        assert not real.tl.flags.writeable

    def test_tl_of_a_reverberation_field_refuses_and_names_db(self):
        with pytest.raises(AttributeError, match="'reverberation'.*\\.db"):
            self._reverberation_field().tl

    def test_stack_tl_matches_stack_db_on_pressure_slabs(self):
        from uacpy.core.results.field import ResultStack
        f = self._pressure_field()
        st = ResultStack(slabs=[f, f], coordinate=np.array([10.0, 20.0]),
                         coordinate_name='source_depth')
        assert np.array_equal(st.tl, st.db)
        assert st.tl.shape == (2, 1, 2)

    def test_stack_tl_of_reverberation_slabs_refuses_and_names_stack_db(self):
        from uacpy.core.results.field import ResultStack
        rl = self._reverberation_field()
        st = ResultStack(slabs=[rl, rl], coordinate=np.array([10.0, 20.0]),
                         coordinate_name='source_depth')
        with pytest.raises(ConfigurationError, match='stack\\.db'):
            st.tl
