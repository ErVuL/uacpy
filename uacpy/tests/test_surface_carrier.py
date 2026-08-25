"""Unit tests for the Surface / Bathymetry / Altimetry shape & property carriers."""

import warnings

import numpy as np
import pytest

from uacpy.core.bottom import BoundaryProperties
from uacpy.core.surface import Surface
from uacpy.core.bathymetry import Bathymetry
from uacpy.core.altimetry import Altimetry
from uacpy.core.exceptions import ConfigurationError


def _ice():
    return BoundaryProperties(acoustic_type='half-space', sound_speed=3500.0,
                              density=0.9, shear_speed=1800.0, attenuation=1.0)


class TestSurfaceCarrier:
    def test_coerce_uniform_delegates_like_boundaryproperties(self):
        s = Surface.coerce(_ice())
        assert not s.is_range_dependent
        # delegated BoundaryProperties reads
        assert s.acoustic_type == 'half-space'
        assert s.sound_speed == 3500.0 and s.shear_speed == 1800.0

    def test_coerce_none_is_vacuum(self):
        s = Surface.coerce(None)
        assert s.acoustic_type == 'vacuum' and not s.is_range_dependent

    def test_range_dependent_at_isel_no_eval(self):
        openw = BoundaryProperties(acoustic_type='vacuum')
        s = Surface.coerce([(0.0, openw), (5000.0, _ice())])
        assert s.is_range_dependent and s.n_ranges == 2
        assert s.at(range=1000).acoustic_type == 'vacuum'      # nearest 0
        assert s.at(range=6000).acoustic_type == 'half-space'  # nearest 5000
        assert s.isel(range=1).shear_speed == 1800.0
        assert not hasattr(s, 'eval')                          # property carrier

    def test_collapse_r0_rmax_mean(self):
        s = Surface.coerce([(0.0, _ice()), (5000.0,
                            BoundaryProperties(acoustic_type='half-space',
                                               sound_speed=3700.0, density=0.92,
                                               shear_speed=2000.0))])
        assert s.collapse('r0').sound_speed == 3500.0
        assert s.collapse('rmax').sound_speed == 3700.0
        assert s.collapse('mean').sound_speed == pytest.approx(3600.0)

    def test_mismatched_ranges_raises(self):
        with pytest.raises(ConfigurationError):
            Surface(properties=[_ice(), _ice()], ranges=[0.0])

    def test_isel_out_of_range_is_typed(self):
        s = Surface.coerce([(0.0, _ice()), (5000.0, _ice())])
        with pytest.raises(IndexError, match="Surface.isel"):
            s.isel(range=99)
        with pytest.raises(IndexError, match="Surface.isel"):
            s.isel(range=-99)

    def test_collapse_mean_carries_roughness(self):
        a = BoundaryProperties(acoustic_type='half-space', sound_speed=3500.0,
                               density=0.9, shear_speed=1800.0, roughness=1.5)
        b = BoundaryProperties(acoustic_type='half-space', sound_speed=3500.0,
                               density=0.9, shear_speed=1800.0, roughness=2.5)
        s = Surface(properties=[a, b], ranges=[0.0, 5000.0])
        assert s.collapse('mean').properties[0].roughness == pytest.approx(2.0)
        assert s.collapse('median').properties[0].roughness == pytest.approx(2.0)


class TestAltimetryCarrier:
    """Heights are metres about the mean surface, **positive up** — the
    opposite sign to bathymetry depths, which are positive down. A negative
    height is a trough, not a deeper seabed."""

    def test_at_eval_isel(self):
        a = Altimetry(ranges=[0.0, 10000.0], heights=[0.0, -8.0])
        assert a.eval(range=5000) == pytest.approx(-4.0)       # linear
        assert a.at(range=4000) == 0.0                         # nearest node 0
        assert a.isel(range=1) == -8.0
        assert a.is_range_dependent

    def test_heights_any_sign(self):
        a = Altimetry(ranges=[0.0, 5000.0], heights=[2.0, -3.0])  # crest + trough
        assert float(np.max(a.heights)) == 2.0 and a.eval(range=0) == 2.0

    def test_isel_out_of_range_is_typed(self):
        a = Altimetry(ranges=[0.0, 10000.0], heights=[0.0, -8.0])
        with pytest.raises(IndexError, match="Altimetry.isel"):
            a.isel(range=99)


class TestBathymetryCarrier:
    def test_at_eval_isel(self):
        b = Bathymetry(ranges=[0.0, 8000.0], depths=[100.0, 60.0])
        assert b.eval(range=4000) == pytest.approx(80.0)
        assert b.at(range=1000) == 100.0
        assert b.isel(range=1) == 60.0
        assert b.depth == 100.0 and b.is_range_dependent

    def test_positive_depth_enforced(self):
        with pytest.raises(ConfigurationError):
            Bathymetry(ranges=[0.0], depths=[-5.0])

    def test_isel_out_of_range_is_typed(self):
        b = Bathymetry(ranges=[0.0, 8000.0], depths=[100.0, 60.0])
        with pytest.raises(IndexError, match="Bathymetry.isel"):
            b.isel(range=99)


class TestSurfaceModelBehaviour:
    """How a ``Surface`` reaches a model: it counts towards
    ``env.is_range_dependent``, and Kraken reads the *collapsed* surface
    rather than the raw carrier when deciding what the deck carries."""

    def _ice(self):
        return BoundaryProperties(acoustic_type='half-space', sound_speed=3500.0,
                                  density=0.9, shear_speed=1800.0,
                                  attenuation=0.5, shear_attenuation=1.0)

    def test_env_is_range_dependent_includes_surface(self):
        import uacpy
        env = uacpy.Environment(
            bathymetry=300.0, ssp=1500.0,
            surface=Surface.coerce([(0.0, BoundaryProperties(acoustic_type='vacuum')),
                                    (5000.0, self._ice())]))
        # surface-only range-dependence must register (else it is silently dropped)
        assert env.is_range_dependent
        assert env.max_range == 5000.0

    @pytest.mark.requires_binary
    def test_kraken_runs_under_an_ice_canopy(self):
        """An elastic top half-space is supported.

        ``krakenc.f90:220-222`` folds ``HSTop%cS`` into ``cMin``
        symmetrically with the bottom at ``:210-212``, so an ice canopy sets
        ``ElasticFlag`` and drags the automatic search floor to 0.84x the ice
        shear speed; the solver then chases the ice/water Scholte mode and
        fails at 1 kHz and 2 kHz. ``_c_low_for`` pins the floor at the minimum
        compressional speed instead, and the band runs: measured 64.82 dB at
        50 Hz, 65.03 at 120, 56.70 at 400, 58.17 at 1 kHz and 60.24 at 2 kHz
        on a 300 m column under a 3500/1800 m/s canopy.
        """
        import numpy as np
        import uacpy
        env = uacpy.Environment(bathymetry=300.0, ssp=1500.0, surface=self._ice())
        src = uacpy.Source(depths=50.0, frequencies=120.0)
        rcv = uacpy.Receiver(depths=[100.0], ranges=[2000.0])
        tl = np.asarray(uacpy.Kraken(verbose=False).compute_tl(env, src, rcv).db,
                        dtype=float)
        assert np.isfinite(tl).all()

    def test_an_ice_canopy_pins_the_phase_speed_floor(self):
        """The floor is the minimum compressional speed, not KRAKEN's own
        automatic choice — which an elastic surface would drag below the
        waterborne modes."""
        import uacpy
        from uacpy.models.kraken import Kraken
        iced = uacpy.Environment(bathymetry=300.0, ssp=1500.0, surface=self._ice())
        fluid = uacpy.Environment(bathymetry=300.0, ssp=1500.0)
        assert Kraken(verbose=False)._c_low_for(iced) == pytest.approx(1500.0)
        # A wholly fluid environment still hands the choice to KRAKEN.
        assert Kraken(verbose=False)._c_low_for(fluid) == 0.0

    def test_the_phase_speed_floor_reads_the_collapsed_surface(self):
        import uacpy
        # Kraken carries a single global top, so a range-dependent surface is
        # collapsed first; open water at r=0 collapses to vacuum, leaving no
        # elastic top and so no reason to pin the floor.
        env = uacpy.Environment(
            bathymetry=300.0, ssp=1500.0,
            surface=Surface.coerce([(0.0, BoundaryProperties(acoustic_type='vacuum')),
                                    (5000.0, self._ice())]))
        from uacpy.models.kraken import Kraken
        model = Kraken(verbose=False)
        assert model._has_elastic_surface(model._project_environment(env)) is False

    def test_compute_modes_rejects_receiver_arg(self):
        # A mode solve has no receiver grid, so ``compute_modes`` takes none:
        # its third positional is ``n_modes`` (``PropagationModel.compute_modes``).
        # Passing a Receiver there must name the real parameter in the error
        # rather than fail deep inside the solver on a non-integer count.
        import uacpy
        env = uacpy.Environment(bathymetry=300.0, ssp=1500.0)
        src = uacpy.Source(depths=50.0, frequencies=120.0)
        rcv = uacpy.Receiver(depths=[100.0], ranges=[2000.0])
        with pytest.raises(uacpy.ConfigurationError, match="n_modes"):
            uacpy.Kraken().compute_modes(env, src, rcv)


class TestSurfaceValidation:
    """A mean over mixed boundary types has no meaning, so it must raise
    rather than average a vacuum against a half-space."""

    def test_mixed_type_mean_collapse_clear_error(self):
        s = Surface.coerce([(0.0, BoundaryProperties(acoustic_type='vacuum')),
                            (5000.0, BoundaryProperties(acoustic_type='half-space',
                                                        sound_speed=3500.0,
                                                        density=0.9, shear_speed=1800.0))])
        with pytest.raises(ConfigurationError, match="single boundary type"):
            s.collapse('mean')


class TestRangeAxisFiniteness:
    """Every range axis goes through the shared ``_require_non_negative``
    validator, which checks finiteness first — ``+inf`` passes a bare
    ``< 0`` test and a strictly-increasing check, and NaN passes both."""

    @pytest.mark.parametrize('bad', [np.inf, np.nan])
    def test_surface_ranges_reject_non_finite(self, bad):
        with pytest.raises(ConfigurationError, match='finite'):
            Surface(properties=[BoundaryProperties(), BoundaryProperties()],
                    ranges=[0.0, bad])

    def test_surface_ranges_reject_negative(self):
        with pytest.raises(ConfigurationError, match='non-negative'):
            Surface(properties=[BoundaryProperties(), BoundaryProperties()],
                    ranges=[-1.0, 5.0])


class TestAltimetryProvenance:
    """``Altimetry`` carries ``data_sources`` like every other shape carrier,
    so a fetched sea surface keeps the date/coords the fetcher returned."""

    @staticmethod
    def _record():
        from uacpy.data.sources import DataProvenance, DataSource
        return DataProvenance(source=DataSource(
            id='test-altimetry', name='Test', used_for='altimetry',
            license='x', attribution='x', citation='x', url='x',
            commercial_use=True))

    def test_altimetry_carries_data_sources(self):
        alt = Altimetry(ranges=[0.0, 100.0], heights=[0.1, -0.2],
                        data_sources=(self._record(),))
        assert [r.source.id for r in alt.data_sources] == ['test-altimetry']

    def test_env_aggregates_altimetry_provenance(self):
        import uacpy
        alt = Altimetry(ranges=[0.0, 100.0], heights=[0.1, -0.2],
                        data_sources=(self._record(),))
        env = uacpy.Environment(name='t', bathymetry=100.0, ssp=1500.0,
                                altimetry=alt)
        assert [r.source.id for r in env.data_sources] == ['test-altimetry']

    def test_altimetry_rejects_bare_data_source(self):
        from uacpy.data.sources import DataSource
        bare = DataSource(id='x', name='x', used_for='x', license='x',
                          attribution='x', citation='x', url='x',
                          commercial_use=True)
        with pytest.raises(ConfigurationError, match='DataProvenance'):
            Altimetry(ranges=[0.0], heights=[0.0], data_sources=(bare,))

    def test_bathymetry_sibling_aggregates(self):
        import uacpy
        bathy = Bathymetry(ranges=[0.0, 100.0], depths=[100.0, 90.0],
                           data_sources=(self._record(),))
        env = uacpy.Environment(name='t', bathymetry=bathy, ssp=1500.0)
        assert [r.source.id for r in env.data_sources] == ['test-altimetry']


class TestDelegatedWritesAreValidated:
    """Writes through the Surface proxy obey the ``BoundaryProperties``
    construction rules, so the proxy cannot store a value on the nodes that
    their constructor would refuse."""

    @staticmethod
    def _halfspace_surface():
        return Surface.coerce(BoundaryProperties(sound_speed=1700.0))

    def test_type_fields_are_not_assignable(self):
        s = self._halfspace_surface()
        for name, value in (('acoustic_type', 'rigid'),
                            ('reflection_file', 'top.trc')):
            with pytest.raises(ConfigurationError, match='cannot be assigned'):
                setattr(s, name, value)

    def test_numeric_rules_mirror_the_constructor(self):
        s = self._halfspace_surface()
        with pytest.raises(ConfigurationError, match='must be positive'):
            s.density = -3.0
        with pytest.raises(ConfigurationError, match='must be positive'):
            s.sound_speed = 0.0
        with pytest.raises(ConfigurationError, match='exceeds'):
            s.attenuation = 500.0
        with pytest.raises(ConfigurationError, match='non-negative'):
            s.roughness = -1.0

    def test_vacuum_nodes_reject_halfspace_params(self):
        s = Surface.coerce(None)
        with pytest.raises(ConfigurationError, match='vacuum'):
            s.sound_speed = 1700.0
        s.roughness = 0.5
        assert s.properties[0].roughness == 0.5

    def test_valid_writes_reach_every_node(self):
        s = Surface.coerce([(0.0, _ice()), (5000.0, _ice())])
        s.attenuation = 0.7
        assert [p.attenuation for p in s.properties] == [0.7, 0.7]
        assert s.attenuation == 0.7


class TestCollapseFileNodes:
    """'mean'/'median' over uniform 'file' nodes keeps the shared table and
    reduces only the roughness; distinct tables refuse to blend."""

    @staticmethod
    def _file_node(roughness):
        return BoundaryProperties(acoustic_type='file',
                                  reflection_file='top.trc',
                                  roughness=roughness)

    def test_shared_table_collapses_to_it_with_reduced_roughness(self):
        s = Surface.coerce([(0.0, self._file_node(0.1)),
                            (5000.0, self._file_node(0.3))])
        c = s.collapse('mean')
        node = c.properties[0]
        assert node.acoustic_type == 'file'
        assert node.reflection_file == 'top.trc'
        assert node.roughness == pytest.approx(0.2)

    def test_distinct_tables_refuse_to_blend(self):
        a = self._file_node(0.1)
        b = BoundaryProperties(acoustic_type='file',
                               reflection_file='other.trc')
        s = Surface.coerce([(0.0, a), (5000.0, b)])
        with pytest.raises(ConfigurationError, match='different reflection'):
            s.collapse('median')


class TestSurfaceAccessorsReturnCopies:
    """``Surface.at`` / ``Surface.isel`` hand back a copy, matching `Bottom`
    and ``Bottom.halfspace_at``. Returning the stored node from one accessor
    and a copy from another left a caller no way to tell which results were
    safe to mutate. The delegated attributes (``surface.roughness`` …) stay
    the in-place route: reads come from the r = 0 node, writes broadcast to
    every node, and ``.properties[i]`` addresses one node."""

    def _surface(self):
        return Surface(
            properties=[
                BoundaryProperties(acoustic_type='vacuum', roughness=1.0),
                BoundaryProperties(acoustic_type='vacuum', roughness=2.0),
            ],
            ranges=np.array([0.0, 5000.0]))

    def test_at_result_is_not_the_stored_node(self):
        s = self._surface()
        got = s.at(range=0.0)
        got.roughness = 99.0
        assert s.properties[0].roughness == pytest.approx(1.0)

    def test_isel_result_is_not_the_stored_node(self):
        s = self._surface()
        got = s.isel(range=1)
        got.roughness = 99.0
        assert s.properties[1].roughness == pytest.approx(2.0)

    def test_the_copies_carry_the_stored_values(self):
        s = self._surface()
        assert s.at(range=100.0).roughness == pytest.approx(1.0)
        assert s.at(range=4900.0).roughness == pytest.approx(2.0)
        assert s.isel(range=-1).acoustic_type == 'vacuum'

    def test_delegated_attribute_writes_reach_the_node(self):
        s = self._surface()
        s.roughness = 3.0
        assert s.properties[0].roughness == pytest.approx(3.0)
        assert s.at(range=0.0).roughness == pytest.approx(3.0)


class TestSurfaceDelegatedWriteBroadcasts:
    """A delegated write (``surface.roughness = …``) propagates to every
    range node — a uniform broadcast. On a multi-node surface it warns,
    because it flattens any range dependence; ``.properties[i]`` is the
    single-node route. On a single-node surface the broadcast and the one
    node are the same thing, so it is silent."""

    def _multi_node(self):
        return Surface(
            properties=[
                BoundaryProperties(acoustic_type='vacuum', roughness=1.0),
                BoundaryProperties(acoustic_type='vacuum', roughness=2.0),
            ],
            ranges=np.array([0.0, 5000.0]))

    def test_a_multi_node_write_warns_and_reaches_every_node(self):
        s = self._multi_node()
        with pytest.warns(UserWarning, match=r"sets all 2 range nodes"):
            s.roughness = 3.0
        assert all(p.roughness == pytest.approx(3.0) for p in s.properties)

    def test_the_multi_node_warning_points_at_properties(self):
        s = self._multi_node()
        with pytest.warns(UserWarning, match=r"\.properties\[i\]"):
            s.roughness = 3.0

    def test_a_single_node_write_is_silent_and_writes_through(self):
        s = Surface(properties=[
            BoundaryProperties(acoustic_type='vacuum', roughness=1.0)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            s.roughness = 5.0
        assert not [w for w in caught if issubclass(w.category, UserWarning)]
        assert s.properties[0].roughness == pytest.approx(5.0)
        assert s.roughness == pytest.approx(5.0)


class TestSurfaceValidatesGrainSizePhi:
    """The delegated-write validator applies the ``BoundaryProperties``
    construction rules to ``surface.<field> = value``, and ``grain_size_phi``
    was the one field it stepped over entirely: a NaN or inf ϕ was stored
    where every sibling field rejects one, and ``None`` — the field's own
    unset value — died in ``float(None)`` with a bare ``TypeError``. The ϕ
    *range* stays unchecked here, as at construction: ϕ = −log₂(d/mm) is
    signed, and ``grain_size_to_geoacoustics`` warns naming each model's
    valid interval at the point of use."""

    @staticmethod
    def _surface():
        return Surface(properties=[BoundaryProperties(
            acoustic_type='half-space', sound_speed=1600.0, density=1.8,
            attenuation=0.5)])

    @pytest.mark.parametrize('bad', [float('nan'), float('inf'),
                                     float('-inf')])
    def test_a_non_finite_phi_is_refused(self, bad):
        with pytest.raises(ConfigurationError, match='grain_size_phi'):
            self._surface().grain_size_phi = bad

    def test_the_node_keeps_its_previous_value_after_a_refusal(self):
        s = self._surface()
        s.grain_size_phi = 3.5
        with pytest.raises(ConfigurationError):
            s.grain_size_phi = float('nan')
        assert s.properties[0].grain_size_phi == pytest.approx(3.5)

    @pytest.mark.parametrize('phi', [3.5, 0.0, -2.0])
    def test_a_signed_finite_phi_is_stored(self, phi):
        s = self._surface()
        s.grain_size_phi = phi
        assert s.grain_size_phi == pytest.approx(phi)
        assert s.properties[0].grain_size_phi == pytest.approx(phi)

    def test_none_clears_it(self):
        s = self._surface()
        s.grain_size_phi = 3.5
        s.grain_size_phi = None
        assert s.grain_size_phi is None
        assert s.properties[0].grain_size_phi is None

    def test_the_write_reaches_every_node(self):
        s = Surface(properties=[
            BoundaryProperties(acoustic_type='half-space', sound_speed=1600.0,
                               density=1.8, attenuation=0.5),
            BoundaryProperties(acoustic_type='half-space', sound_speed=1700.0,
                               density=1.9, attenuation=0.5)],
            ranges=np.array([0.0, 1000.0]))
        s.grain_size_phi = 4.0
        assert [p.grain_size_phi for p in s.properties] == [4.0, 4.0]

    def test_a_parameter_free_node_refuses_it_first(self):
        s = Surface(properties=[BoundaryProperties(acoustic_type='vacuum')])
        with pytest.raises(ConfigurationError, match='vacuum'):
            s.grain_size_phi = 3.5

    @pytest.mark.parametrize('field, bad', [
        ('density', -1.0), ('sound_speed', 0.0), ('roughness', -1.0),
        ('attenuation', -1.0), ('shear_speed', -1.0)])
    def test_every_sibling_field_refuses_its_own_bad_value(self, field, bad):
        with pytest.raises(ConfigurationError):
            setattr(self._surface(), field, bad)
