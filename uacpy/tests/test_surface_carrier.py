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


class TestAltimetryCarrier:
    def test_at_eval_isel(self):
        a = Altimetry(ranges=[0.0, 10000.0], heights=[0.0, -8.0])
        assert a.eval(range=5000) == pytest.approx(-4.0)       # linear
        assert a.at(range=4000) == 0.0                         # nearest node 0
        assert a.isel(range=1) == -8.0
        assert a.is_range_dependent

    def test_heights_any_sign(self):
        a = Altimetry(ranges=[0.0, 5000.0], heights=[2.0, -3.0])  # crest + trough
        assert float(np.max(a.heights)) == 2.0 and a.eval(range=0) == 2.0


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


class TestSurfaceModelBehaviour:
    """Regression locks for bugs found in manual testing of the Surface feature."""

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
        import uacpy
        env = uacpy.Environment(bathymetry=300.0, ssp=1500.0)
        src = uacpy.Source(depths=50.0, frequencies=120.0)
        rcv = uacpy.Receiver(depths=[100.0], ranges=[2000.0])
        with pytest.raises(uacpy.ConfigurationError, match="n_modes"):
            uacpy.Kraken().compute_modes(env, src, rcv)


class TestSurfaceValidation:
    """Locks for edge-case bugs found in manual testing (round 6)."""

    def test_mixed_type_mean_collapse_clear_error(self):
        s = Surface.coerce([(0.0, BoundaryProperties(acoustic_type='vacuum')),
                            (5000.0, BoundaryProperties(acoustic_type='half-space',
                                                        sound_speed=3500.0,
                                                        density=0.9, shear_speed=1800.0))])
        with pytest.raises(ConfigurationError, match="single boundary type"):
            s.collapse('mean')
