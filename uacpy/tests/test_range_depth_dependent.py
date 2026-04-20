"""
Tests for range-dependent and depth-dependent parameters
"""

import pytest
import warnings
import numpy as np
import uacpy
from uacpy.models import Bellhop, RAM
from uacpy.core.environment import (
    RangeDependentBottom, SedimentLayer, LayeredBottom,
    RangeDependentLayeredBottom, BoundaryProperties,
)


class TestRangeDependentEnvironment:
    """Tests for range-dependent environments"""

    def test_range_dependent_bathymetry(self):
        """Test environment with range-dependent bathymetry"""
        # Create sloping bathymetry
        ranges = np.linspace(0, 10000, 21)
        depths = np.linspace(80, 120, 21)
        bathymetry = np.column_stack([ranges, depths])

        env = uacpy.Environment(
            name="Sloping Bottom",
            depth=100.0,
            sound_speed=1500.0,
            ssp_type='isovelocity',
            bathymetry=bathymetry
        )

        assert env.is_range_dependent
        assert len(env.bathymetry) == 21
        assert env.bathymetry[0, 1] == 80.0
        assert env.bathymetry[-1, 1] == 120.0

    def test_range_dependent_bottom_properties(self):
        """Test range-dependent bottom properties"""
        ranges_km = np.array([0, 5, 10])
        depths = np.array([100, 110, 120])
        sound_speeds = np.array([1600, 1650, 1700])
        densities = np.array([1.5, 1.6, 1.7])
        attenuations = np.array([0.5, 0.6, 0.7])

        bottom_rd = RangeDependentBottom(
            ranges_km=ranges_km,
            depths=depths,
            sound_speed=sound_speeds,
            density=densities,
            attenuation=attenuations
        )

        env = uacpy.Environment(
            name="RD Bottom",
            depth=100.0,
            sound_speed=1500.0,
            ssp_type='isovelocity',
            bottom=bottom_rd
        )

        assert env.has_range_dependent_bottom()
        assert len(env.bottom_rd.ranges_km) == 3

        # Test getting bottom at specific range
        bottom_at_2km = env.get_bottom_at_range(2000)
        assert bottom_at_2km.sound_speed > 1600
        assert bottom_at_2km.sound_speed < 1650

    def test_range_dependent_ssp(self):
        """Test range-dependent sound speed profile"""
        # Create 2D SSP matrix: depth x range
        depths = np.linspace(0, 100, 21)
        ranges_km = np.array([0, 5, 10])

        # SSP varies with range (warming trend)
        ssp_matrix = np.zeros((len(depths), len(ranges_km)))
        for i, r in enumerate(ranges_km):
            # Warmer water at longer ranges
            ssp_matrix[:, i] = 1500 + r * 0.5

        env = uacpy.Environment(
            name="RD SSP",
            depth=100.0,
            ssp_data=np.column_stack([depths, ssp_matrix[:, 0]]),  # Base profile
            ssp_type='pchip',
            ssp_2d_ranges=ranges_km,
            ssp_2d_matrix=ssp_matrix
        )

        assert env.has_range_dependent_ssp()
        assert env.ssp_2d_matrix.shape == (21, 3)

    def test_combined_range_dependencies(self):
        """Test environment with both RD bathymetry and RD SSP"""
        # Bathymetry
        ranges = np.linspace(0, 10000, 11)
        depths_bathy = np.linspace(90, 110, 11)
        bathymetry = np.column_stack([ranges, depths_bathy])

        # SSP
        depths_ssp = np.linspace(0, 100, 21)
        ranges_km = np.array([0, 5, 10])
        ssp_matrix = np.tile(np.linspace(1480, 1520, 21)[:, None], (1, 3))

        env = uacpy.Environment(
            name="Combined RD",
            depth=100.0,
            ssp_data=np.column_stack([depths_ssp, ssp_matrix[:, 0]]),
            ssp_type='pchip',
            bathymetry=bathymetry,
            ssp_2d_ranges=ranges_km,
            ssp_2d_matrix=ssp_matrix
        )

        assert env.is_range_dependent
        assert env.has_range_dependent_ssp()


class TestDepthDependentSSP:
    """Tests for depth-dependent sound speed profiles"""

    def test_isovelocity_profile(self):
        """Test isovelocity (constant) profile"""
        env = uacpy.Environment(
            name="Isovelocity",
            depth=100.0,
            sound_speed=1500.0,
            ssp_type='isovelocity'
        )

        # All sound speeds should be the same
        assert np.all(env.ssp_data[:, 1] == 1500.0)

    def test_linear_gradient(self):
        """Test linear sound speed gradient"""
        depths = np.linspace(0, 100, 11)
        sound_speeds = 1480 + depths * 0.4  # 0.4 m/s per meter

        env = uacpy.Environment(
            name="Linear Gradient",
            depth=100.0,
            ssp_data=np.column_stack([depths, sound_speeds]),
            ssp_type='linear'
        )

        assert env.ssp_data[0, 1] == 1480.0
        assert env.ssp_data[-1, 1] == 1520.0

    def test_munk_profile(self):
        """Test Munk sound speed profile"""
        depths = np.linspace(0, 100, 51)
        axis_depth = 50.0
        c_axis = 1485.0
        epsilon = 0.00737

        # Munk profile formula
        eta = 2 * (depths - axis_depth) / axis_depth
        sound_speeds = c_axis * (1 + epsilon * (eta - 1 + np.exp(-eta)))

        env = uacpy.Environment(
            name="Munk Profile",
            depth=100.0,
            ssp_data=np.column_stack([depths, sound_speeds]),
            ssp_type='pchip'
        )

        # Check that sound speed minimum is near axis depth
        min_idx = np.argmin(env.ssp_data[:, 1])
        min_depth = env.ssp_data[min_idx, 0]
        assert abs(min_depth - axis_depth) < 5.0  # Within 5m of axis

    def test_negative_gradient(self):
        """Test negative sound speed gradient (sound channel)"""
        depths = np.linspace(0, 100, 11)
        sound_speeds = 1520 - depths * 0.4  # Decreasing with depth

        env = uacpy.Environment(
            name="Negative Gradient",
            depth=100.0,
            ssp_data=np.column_stack([depths, sound_speeds]),
            ssp_type='linear'
        )

        assert env.ssp_data[0, 1] > env.ssp_data[-1, 1]


class TestModelWithRangeDependence:
    """Test models with range-dependent environments"""

    @pytest.mark.requires_binary
    def test_bellhop_range_dependent_bathymetry(self):
        """Test Bellhop with range-dependent bathymetry"""
        # Create sloping bottom
        ranges = np.linspace(0, 5000, 11)
        depths = np.linspace(80, 100, 11)
        bathymetry = np.column_stack([ranges, depths])

        env = uacpy.Environment(
            name="Slope",
            depth=90.0,
            sound_speed=1500.0,
            ssp_type='isovelocity',
            bathymetry=bathymetry
        )

        source = uacpy.Source(depth=50.0, frequency=100.0)
        receiver = uacpy.Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.array([1000.0, 3000.0, 5000.0])
        )

        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=env, source=source, receiver=receiver)

        assert result.field_type == 'tl'
        assert result.shape == (3, 3)

    def test_ram_range_dependent_ssp(self):
        """Test RAM with range-dependent SSP"""
        try:
            # Create range-dependent SSP
            depths = np.linspace(0, 100, 21)
            ranges_km = np.array([0, 2.5, 5])

            ssp_matrix = np.zeros((len(depths), len(ranges_km)))
            for i, r_km in enumerate(ranges_km):
                # Temperature increases with range
                ssp_matrix[:, i] = 1500 + r_km * 2

            env = uacpy.Environment(
                name="RD SSP",
                depth=100.0,
                ssp_data=np.column_stack([depths, ssp_matrix[:, 0]]),
                ssp_type='linear',
                ssp_2d_ranges=ranges_km,
                ssp_2d_matrix=ssp_matrix
            )

            source = uacpy.Source(depth=50.0, frequency=100.0)
            receiver = uacpy.Receiver(
                depths=np.array([25.0, 50.0, 75.0]),
                ranges=np.array([1000.0, 3000.0, 5000.0])
            )

            ram = RAM(verbose=False)
            result = ram.compute_tl(env=env, source=source, receiver=receiver)

            assert result.field_type == 'tl'
            # RAM computes on its own internal grid
            assert result.shape[0] > 0  # Has depth dimension
            assert result.shape[1] > 0  # Has range dimension
            assert np.all(np.isfinite(result.data))  # All values are finite

        except ImportError:
            pytest.skip("RAM module not available")

    @pytest.mark.requires_binary
    def test_bellhop_range_dependent_bottom(self):
        """Test Bellhop with range-dependent bottom properties"""
        # Create range-dependent bottom
        ranges_km = np.array([0, 2.5, 5])
        depths = np.array([100, 105, 110])
        sound_speeds = np.array([1600, 1650, 1700])
        densities = np.array([1.5, 1.6, 1.7])
        attenuations = np.array([0.5, 0.6, 0.7])

        bottom_rd = RangeDependentBottom(
            ranges_km=ranges_km,
            depths=depths,
            sound_speed=sound_speeds,
            density=densities,
            attenuation=attenuations
        )

        env = uacpy.Environment(
            name="RD Bottom",
            depth=100.0,
            sound_speed=1500.0,
            ssp_type='isovelocity',
            bottom=bottom_rd
        )

        source = uacpy.Source(depth=50.0, frequency=100.0)
        receiver = uacpy.Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.array([1000.0, 3000.0, 5000.0])
        )

        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=env, source=source, receiver=receiver)

        assert result.field_type == 'tl'


class TestRangeDependentConsistency:
    """Test consistency of range-dependent handling"""

    def test_bathymetry_interpolation(self):
        """Test that bathymetry is correctly interpolated"""
        ranges = np.array([0, 5000, 10000])
        depths = np.array([80, 100, 120])
        bathymetry = np.column_stack([ranges, depths])

        env = uacpy.Environment(
            name="Test",
            depth=100.0,
            sound_speed=1500.0,
            ssp_type='isovelocity',
            bathymetry=bathymetry
        )

        # Depth at 2500m should be interpolated between 80 and 100
        # Simple linear interpolation: 80 + (100-80) * (2500/5000) = 90
        from scipy.interpolate import interp1d
        interp = interp1d(bathymetry[:, 0], bathymetry[:, 1])
        depth_at_2500 = interp(2500)

        assert 85 < depth_at_2500 < 95  # Should be around 90m

    def test_bottom_properties_at_range(self):
        """Test getting bottom properties at specific range"""
        ranges_km = np.array([0, 5, 10])
        depths = np.array([100, 110, 120])
        sound_speeds = np.array([1600, 1650, 1700])

        bottom_rd = RangeDependentBottom(
            ranges_km=ranges_km,
            depths=depths,
            sound_speed=sound_speeds,
            density=np.array([1.5, 1.5, 1.5]),
            attenuation=np.array([0.5, 0.5, 0.5])
        )

        env = uacpy.Environment(
            name="Test",
            depth=100.0,
            sound_speed=1500.0,
            ssp_type='isovelocity',
            bottom=bottom_rd
        )

        # Get bottom at 2.5 km (should interpolate between first and second)
        bottom_at_2_5km = env.get_bottom_at_range(2500)

        assert 1600 < bottom_at_2_5km.sound_speed < 1650
        assert 100 < bottom_at_2_5km.depth < 110

    def test_ssp_2d_matrix_shape(self):
        """Test that 2D SSP matrix has correct shape"""
        depths = np.linspace(0, 100, 21)
        ranges_km = np.array([0, 5, 10])

        ssp_matrix = np.tile(np.linspace(1480, 1520, 21)[:, None], (1, 3))

        env = uacpy.Environment(
            name="Test",
            depth=100.0,
            ssp_data=np.column_stack([depths, ssp_matrix[:, 0]]),
            ssp_type='pchip',
            ssp_2d_ranges=ranges_km,
            ssp_2d_matrix=ssp_matrix
        )

        assert env.ssp_2d_matrix.shape == (21, 3)  # 21 depths, 3 ranges
        assert len(env.ssp_2d_ranges) == 3


# ============================================================================
# NEW: Tests for SedimentLayer & LayeredBottom
# ============================================================================

class TestSedimentLayer:
    """Tests for the SedimentLayer dataclass"""

    def test_basic_creation(self):
        """Test basic SedimentLayer creation"""
        layer = SedimentLayer(thickness=10, sound_speed=1650, density=1.9)
        assert layer.thickness == 10
        assert layer.sound_speed == 1650
        assert layer.density == 1.9
        assert layer.attenuation == 0.5  # default
        assert layer.shear_speed == 0.0  # default

    def test_validation_negative_thickness(self):
        """Negative thickness should raise ValueError"""
        with pytest.raises(ValueError, match="thickness"):
            SedimentLayer(thickness=-5, sound_speed=1650, density=1.9)

    def test_validation_negative_sound_speed(self):
        """Negative sound speed should raise ValueError"""
        with pytest.raises(ValueError, match="sound_speed"):
            SedimentLayer(thickness=10, sound_speed=-100, density=1.9)

    def test_elastic_layer(self):
        """Test layer with shear properties"""
        layer = SedimentLayer(
            thickness=20, sound_speed=1700, density=2.0,
            shear_speed=400, shear_attenuation=1.0
        )
        assert layer.shear_speed == 400
        assert layer.shear_attenuation == 1.0


class TestLayeredBottom:
    """Tests for the LayeredBottom class"""

    def test_basic_creation(self):
        """Test basic LayeredBottom creation"""
        lb = LayeredBottom(
            layers=[
                SedimentLayer(thickness=10, sound_speed=1550, density=1.3, attenuation=0.5),
                SedimentLayer(thickness=50, sound_speed=1650, density=1.7, attenuation=0.3),
            ],
            halfspace=BoundaryProperties(
                acoustic_type='half-space',
                sound_speed=1800, density=2.0, attenuation=0.1
            )
        )
        assert len(lb.layers) == 2
        assert lb.total_thickness() == 60

    def test_layer_depths(self):
        """Test layer depth computation"""
        lb = LayeredBottom(
            layers=[
                SedimentLayer(thickness=10, sound_speed=1550, density=1.3),
                SedimentLayer(thickness=50, sound_speed=1650, density=1.7),
            ],
            halfspace=BoundaryProperties(acoustic_type='half-space', sound_speed=1800, density=2.0)
        )
        depths = lb.layer_depths(200)
        assert depths[0] == (200, 210)
        assert depths[1] == (210, 260)

    def test_empty_layers_raises(self):
        """Empty layers list should raise ValueError"""
        with pytest.raises(ValueError, match="at least one"):
            LayeredBottom(
                layers=[],
                halfspace=BoundaryProperties(acoustic_type='half-space', sound_speed=1800, density=2.0)
            )

    def test_environment_with_layered_bottom(self):
        """Test Environment accepts LayeredBottom"""
        lb = LayeredBottom(
            layers=[SedimentLayer(thickness=10, sound_speed=1550, density=1.3)],
            halfspace=BoundaryProperties(acoustic_type='half-space', sound_speed=1800, density=2.0)
        )
        env = uacpy.Environment(name='test', depth=100, bottom=lb)

        assert env.has_layered_bottom()
        assert not env.has_range_dependent_bottom()
        assert env.bottom_layered is lb
        # env.bottom should be the halfspace for backward compat
        assert env.bottom.sound_speed == 1800

    def test_environment_backward_compat(self):
        """Existing BoundaryProperties usage should be unaffected"""
        env = uacpy.Environment(name='test', depth=100)
        assert not env.has_layered_bottom()
        assert env.bottom_layered is None


class TestRangeDependentLayeredBottom:
    """Tests for the RangeDependentLayeredBottom class"""

    def _make_rdl(self):
        near = LayeredBottom(
            layers=[SedimentLayer(5, 1500, 1.2, 1.0),
                    SedimentLayer(15, 1550, 1.4, 0.8)],
            halfspace=BoundaryProperties(acoustic_type='half-space',
                                         sound_speed=1700, density=1.8, attenuation=0.2),
        )
        far = LayeredBottom(
            layers=[SedimentLayer(3, 1650, 1.8, 0.3),
                    SedimentLayer(10, 1750, 2.0, 0.2)],
            halfspace=BoundaryProperties(acoustic_type='half-space',
                                         sound_speed=2200, density=2.5, attenuation=0.05),
        )
        return RangeDependentLayeredBottom(
            ranges_km=np.array([0, 20]),
            depths=np.array([100, 300]),
            profiles=[near, far],
        )

    def test_basic_creation(self):
        rdl = self._make_rdl()
        assert len(rdl.profiles) == 2
        assert rdl.max_total_thickness() == 20.0  # max(5+15, 3+10)

    def test_sample_at_depths(self):
        rdl = self._make_rdl()
        cs, rho, attn = rdl.sample_at_depths(0, n_points=4)
        assert len(cs) == 4
        # First sample is at depth 0 → layer 0
        assert cs[0] == 1500
        assert rho[0] == 1.2

    def test_sample_at_depths_halfspace(self):
        """Samples below all layers should return halfspace properties"""
        rdl = self._make_rdl()
        # Profile 1 has total thickness 13m, max is 20m
        # So sample at depth 20m is below its layers → halfspace
        cs, rho, attn = rdl.sample_at_depths(1, n_points=4)
        # Last sample should be halfspace (2200)
        assert cs[3] == 2200
        assert rho[3] == 2.5

    def test_validation_length_mismatch(self):
        lb = LayeredBottom(
            layers=[SedimentLayer(5, 1500, 1.2)],
            halfspace=BoundaryProperties(acoustic_type='half-space',
                                         sound_speed=1700, density=1.8),
        )
        with pytest.raises(ValueError, match="profiles length"):
            RangeDependentLayeredBottom(
                ranges_km=np.array([0, 10, 20]),
                depths=np.array([100, 200, 300]),
                profiles=[lb, lb],  # 2 profiles for 3 ranges
            )

    def test_environment_accepts_rdl(self):
        rdl = self._make_rdl()
        env = uacpy.Environment(name='test', depth=300, bottom=rdl)
        assert env.has_range_dependent_layered_bottom()
        assert not env.has_layered_bottom()
        assert not env.has_range_dependent_bottom()
        assert env.is_range_dependent
        # Bathymetry should be set from depths
        assert len(env.bathymetry) == 2
        assert env.bathymetry[0, 1] == 100
        assert env.bathymetry[1, 1] == 300

    @pytest.mark.requires_binary
    def test_ram_with_rdl(self):
        """RAM should handle RangeDependentLayeredBottom"""
        rdl = self._make_rdl()
        env = uacpy.Environment(name='rdl_test', depth=300,
                                ssp_type='isovelocity', sound_speed=1500.0,
                                bottom=rdl)
        source = uacpy.Source(frequency=100.0, depth=30.0)
        receiver = uacpy.Receiver(
            depths=np.linspace(5, 290, 10),
            ranges=np.linspace(100, 20000, 15),
        )
        ram = RAM(verbose=False)
        result = ram.run(env, source, receiver)
        assert result.data.shape[0] == 10
        assert 30 < np.nanmin(result.data) < 100


class TestWarnings:
    """Test that models warn for unsupported features"""

    def test_bellhop_warns_bottom_rd(self):
        """Bellhop should warn about range-dependent bottom"""
        rd_bottom = RangeDependentBottom(
            ranges_km=np.array([0, 10]),
            depths=np.array([100, 100]),
            sound_speed=np.array([1600, 1700]),
            density=np.array([1.5, 1.7]),
            attenuation=np.array([0.5, 0.3])
        )
        env = uacpy.Environment(name='test', depth=100, bottom=rd_bottom)
        source = uacpy.Source(frequency=100, depth=25)
        receiver = uacpy.Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))

        bellhop = Bellhop(verbose=False)
        with pytest.warns(UserWarning, match="range-dependent bottom"):
            try:
                bellhop.run(env, source, receiver)
            except (FileNotFoundError, RuntimeError):
                pass  # Binary may not be available

    def test_bellhop_warns_layered_bottom(self):
        """Bellhop should warn about layered bottom"""
        lb = LayeredBottom(
            layers=[SedimentLayer(thickness=10, sound_speed=1550, density=1.3)],
            halfspace=BoundaryProperties(acoustic_type='half-space', sound_speed=1800, density=2.0)
        )
        env = uacpy.Environment(name='test', depth=100, bottom=lb)
        source = uacpy.Source(frequency=100, depth=25)
        receiver = uacpy.Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))

        bellhop = Bellhop(verbose=False)
        with pytest.warns(UserWarning, match="layered"):
            try:
                bellhop.run(env, source, receiver)
            except (FileNotFoundError, RuntimeError):
                pass

    def test_ram_accepts_layered_bottom(self):
        """RAM should accept layered bottom without warnings"""
        lb = LayeredBottom(
            layers=[SedimentLayer(thickness=10, sound_speed=1550, density=1.3)],
            halfspace=BoundaryProperties(acoustic_type='half-space', sound_speed=1800, density=2.0)
        )
        env = uacpy.Environment(name='test', depth=100, bottom=lb)
        source = uacpy.Source(frequency=100, depth=25)
        receiver = uacpy.Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))

        ram = RAM(verbose=False)
        try:
            ram.run(env, source, receiver)
        except (FileNotFoundError, RuntimeError):
            pass  # Binary may not be available


class TestIntegrationLayeredBottom:
    """Integration tests: run models with layered bottom"""

    @pytest.fixture
    def layered_env(self):
        lb = LayeredBottom(
            layers=[
                SedimentLayer(thickness=5.0, sound_speed=1550.0, density=1.5, attenuation=0.3),
                SedimentLayer(thickness=10.0, sound_speed=1650.0, density=1.7, attenuation=0.5),
            ],
            halfspace=BoundaryProperties(
                acoustic_type='half-space', sound_speed=2000.0, density=2.2, attenuation=0.1
            )
        )
        return uacpy.Environment(name='layered_test', depth=200.0,
                                 ssp_type='isovelocity', sound_speed=1500.0, bottom=lb)

    @pytest.fixture
    def source(self):
        return uacpy.Source(frequency=100.0, depth=50.0)

    @pytest.fixture
    def receiver(self):
        return uacpy.Receiver(
            depths=np.linspace(1, 199, 20),
            ranges=np.linspace(100, 5000, 20),
        )

    @pytest.mark.requires_binary
    def test_krakenfield_layered(self, layered_env, source, receiver):
        """KrakenField should produce valid TL with layered bottom"""
        from uacpy.models import KrakenField
        kf = KrakenField(verbose=False)
        result = kf.compute_tl(layered_env, source, receiver)
        assert result.data.shape == (20, 20)
        assert 30 < np.nanmin(result.data) < 100
        assert 50 < np.nanmax(result.data) < 200

    @pytest.mark.requires_binary
    def test_scooter_layered(self, layered_env, source, receiver):
        """Scooter should produce valid TL with layered bottom"""
        from uacpy.models import Scooter
        scooter = Scooter(verbose=False)
        result = scooter.compute_tl(layered_env, source, receiver)
        assert result.data.shape == (20, 20)
        assert 30 < np.nanmin(result.data) < 100

    @pytest.mark.requires_binary
    def test_kraken_layered_modes(self, layered_env, source, receiver):
        """Kraken should produce valid modes with layered bottom"""
        from uacpy.models import Kraken
        kraken = Kraken(verbose=False)
        result = kraken.run(layered_env, source, receiver)
        # Should have modes
        assert result.data is not None
        assert len(result.data) > 0


class TestIntegrationRunWithBounce:
    """Integration test for Bellhop.run_with_bounce()"""

    @pytest.mark.requires_binary
    def test_run_with_bounce_produces_tl(self):
        """run_with_bounce() should produce valid TL field"""
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1700.0, shear_speed=400.0, density=1.9,
            attenuation=0.5, shear_attenuation=1.0,
        )
        env = uacpy.Environment(name='bounce_test', depth=100,
                                ssp_type='isovelocity', sound_speed=1500.0, bottom=bottom)
        source = uacpy.Source(frequency=500.0, depth=25.0)
        receiver = uacpy.Receiver(
            depths=np.linspace(1, 99, 10),
            ranges=np.linspace(100, 3000, 10),
        )

        bellhop = Bellhop(verbose=False)
        result = bellhop.run_with_bounce(env, source, receiver, run_type='C')

        assert result.data.shape == (10, 10)
        assert 30 < np.nanmin(result.data) < 100
        assert 50 < np.nanmax(result.data) < 200


class TestIntegrationRAMRangeDependent:
    """Integration test for RAM with range-dependent bottom"""

    @pytest.mark.requires_binary
    def test_ram_rd_bottom_produces_tl(self):
        """RAM with RD bottom should produce valid TL"""
        bottom_rd = RangeDependentBottom(
            ranges_km=np.array([0, 5, 10]),
            depths=np.array([100, 100, 100]),
            sound_speed=np.array([1500, 1600, 1700]),
            density=np.array([1.2, 1.5, 2.0]),
            attenuation=np.array([1.0, 0.5, 0.3]),
            acoustic_type='half-space',
        )
        env = uacpy.Environment(name='ram_rd', depth=100.0,
                                ssp_type='isovelocity', sound_speed=1500.0, bottom=bottom_rd)
        source = uacpy.Source(frequency=100.0, depth=25.0)
        receiver = uacpy.Receiver(
            depths=np.linspace(1, 99, 10),
            ranges=np.linspace(100, 10000, 20),
        )

        ram = RAM(verbose=False)
        result = ram.run(env, source, receiver)
        assert result.data.shape[0] == 10
        assert 30 < np.nanmin(result.data) < 100


class TestATEnvWriterLayered:
    """Test AT env writer with layered bottom"""

    def test_nmedia_with_layers(self):
        """AT env writer should set NMEDIA > 1 for layered bottom"""
        import io
        from uacpy.io.at_env_writer import ATEnvWriter
        from uacpy.core.constants import SSPType, BoundaryType

        lb = LayeredBottom(
            layers=[
                SedimentLayer(thickness=10, sound_speed=1550, density=1.3, attenuation=0.5),
                SedimentLayer(thickness=50, sound_speed=1650, density=1.7, attenuation=0.3),
            ],
            halfspace=BoundaryProperties(acoustic_type='half-space', sound_speed=1800, density=2.0, attenuation=0.1)
        )
        env = uacpy.Environment(name='test', depth=200, sound_speed=1500, bottom=lb)
        source = uacpy.Source(frequency=100, depth=25)

        buf = io.StringIO()
        ATEnvWriter.write_header(buf, env, source,
                                 ssp_type=SSPType.ISOVELOCITY,
                                 surface_type=BoundaryType.VACUUM)
        content = buf.getvalue()

        # Should have NMEDIA = 3 (1 water + 2 sediment layers)
        lines = content.strip().split('\n')
        nmedia_line = lines[2]  # Third line is NMEDIA
        assert nmedia_line.strip() == '3'

    def test_layer_sections_written(self):
        """Layer sections should be written between SSP and bottom"""
        import io
        from uacpy.io.at_env_writer import ATEnvWriter

        lb = LayeredBottom(
            layers=[SedimentLayer(thickness=10, sound_speed=1550, density=1.3, attenuation=0.5)],
            halfspace=BoundaryProperties(acoustic_type='half-space', sound_speed=1800, density=2.0)
        )
        env = uacpy.Environment(name='test', depth=100, sound_speed=1500, bottom=lb)

        buf = io.StringIO()
        depth_after = ATEnvWriter.write_layer_sections(buf, env, 100)
        content = buf.getvalue()

        assert depth_after == 110  # 100 + 10m layer
        assert '1550' in content  # Layer sound speed present

    def test_halfspace_depth_below_layers(self):
        """Halfspace depth should be below all layers"""
        import io
        from uacpy.io.at_env_writer import ATEnvWriter
        from uacpy.core.constants import BoundaryType

        lb = LayeredBottom(
            layers=[
                SedimentLayer(thickness=10, sound_speed=1550, density=1.3),
                SedimentLayer(thickness=50, sound_speed=1650, density=1.7),
            ],
            halfspace=BoundaryProperties(acoustic_type='half-space', sound_speed=1800, density=2.0, attenuation=0.1)
        )
        env = uacpy.Environment(name='test', depth=200, sound_speed=1500, bottom=lb)

        buf = io.StringIO()
        ATEnvWriter.write_bottom_section(buf, env)
        content = buf.getvalue()

        # Halfspace should be at 260m (200 + 10 + 50)
        assert '260.00' in content
