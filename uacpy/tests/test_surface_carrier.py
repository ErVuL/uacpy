"""Unit tests for the Surface / Bathymetry / Altimetry shape & property carriers."""

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
    ``env.is_range_dependent``, and Kraken's elastic-top rejection reads the
    *collapsed* surface rather than the raw carrier."""

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

    def test_kraken_rejects_elastic_surface_clearly(self):
        import uacpy
        env = uacpy.Environment(bathymetry=300.0, ssp=1500.0, surface=self._ice())
        src = uacpy.Source(depths=50.0, frequencies=120.0)
        rcv = uacpy.Receiver(depths=[100.0], ranges=[2000.0])
        # krakenc.exe aborts on an elastic top — must be a clear typed error
        with pytest.raises(uacpy.UnsupportedFeatureError, match="ice|elastic"):
            uacpy.Kraken().compute_tl(env, src, rcv)

    def test_kraken_elastic_surface_guard_is_collapse_aware(self):
        import uacpy
        # open water at r=0 collapses to vacuum → no elastic top → no guard
        env = uacpy.Environment(
            bathymetry=300.0, ssp=1500.0,
            surface=Surface.coerce([(0.0, BoundaryProperties(acoustic_type='vacuum')),
                                    (5000.0, self._ice())]))
        # guard checks the *collapsed* surface; r0=vacuum so it must NOT fire here
        from uacpy.models.kraken import Kraken
        Kraken()._reject_elastic_surface(Kraken()._project_environment(env))

    def test_compute_modes_rejects_receiver_arg(self):
        # A mode solve has no receiver grid, so ``compute_modes`` takes none:
        # its third positional is ``n_modes`` (``models/base.py:1471-1475``).
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

    def test_bathymetry_sibling_still_aggregates(self):
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
