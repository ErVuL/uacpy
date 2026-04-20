"""
Tests for top boundary conditions and layered bottoms across models.

Covers:
  - Top BC: vacuum (default), elastic/ice surface
  - Bottom: halfspace, single-layer, multi-layer (LayeredBottom)
  - Range-dependent: RangeDependentLayeredBottom
  - Combined: ice surface + layered bottom
  - OASES writer surface property propagation
"""

import numpy as np
import pytest
import uacpy
from uacpy.core.environment import (
    BoundaryProperties, SedimentLayer, LayeredBottom,
    RangeDependentLayeredBottom,
)
from uacpy.models.ram import RAM
from uacpy.models.kraken import Kraken
from uacpy.models.scooter import Scooter


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def source():
    return uacpy.Source(frequency=200, depth=25)


@pytest.fixture
def receiver():
    return uacpy.Receiver(
        depths=np.linspace(5, 95, 20),
        ranges=np.array([2000.0, 5000.0, 10000.0]),
    )


@pytest.fixture
def vacuum_surface():
    return BoundaryProperties(acoustic_type='vacuum', depth=0.0)


@pytest.fixture
def ice_surface():
    return BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=3500.0, shear_speed=1800.0, density=0.9,
        attenuation=0.1, shear_attenuation=0.2,
    )


@pytest.fixture
def halfspace_bottom():
    return BoundaryProperties(
        acoustic_type='half-space', sound_speed=1600,
        density=1.5, attenuation=0.5,
    )


@pytest.fixture
def single_layer_bottom():
    return LayeredBottom(
        layers=[
            SedimentLayer(thickness=10.0, sound_speed=1550, density=1.3,
                          attenuation=0.8),
        ],
        halfspace=BoundaryProperties(
            acoustic_type='half-space', sound_speed=2500,
            density=2.5, attenuation=0.1,
        ),
    )


@pytest.fixture
def multi_layer_bottom():
    return LayeredBottom(
        layers=[
            SedimentLayer(thickness=5.0, sound_speed=1550, density=1.3,
                          attenuation=0.8),
            SedimentLayer(thickness=15.0, sound_speed=1650, density=1.7,
                          attenuation=0.4),
            SedimentLayer(thickness=30.0, sound_speed=1800, density=2.0,
                          attenuation=0.2),
        ],
        halfspace=BoundaryProperties(
            acoustic_type='half-space', sound_speed=2500,
            density=2.5, attenuation=0.1,
        ),
    )


@pytest.fixture
def rd_layered_bottom():
    near = LayeredBottom(
        layers=[
            SedimentLayer(thickness=8.0, sound_speed=1500, density=1.2,
                          attenuation=1.0),
            SedimentLayer(thickness=20.0, sound_speed=1580, density=1.5,
                          attenuation=0.6),
        ],
        halfspace=BoundaryProperties(
            acoustic_type='half-space', sound_speed=1800,
            density=2.0, attenuation=0.2,
        ),
    )
    far = LayeredBottom(
        layers=[
            SedimentLayer(thickness=3.0, sound_speed=1650, density=1.8,
                          attenuation=0.3),
        ],
        halfspace=BoundaryProperties(
            acoustic_type='half-space', sound_speed=2500,
            density=2.5, attenuation=0.05,
        ),
    )
    return RangeDependentLayeredBottom(
        ranges_km=np.array([0.0, 10.0]),
        depths=np.array([100.0, 100.0]),
        profiles=[near, far],
    )


def _make_env(bottom, surface=None, name='test'):
    return uacpy.Environment(
        name=name, depth=100,
        ssp_data=[(0, 1500), (100, 1500)],
        bottom=bottom,
        surface=surface,
    )


# ── Top Boundary Condition Tests ─────────────────────────────────────────────

class TestTopBoundary:
    """Verify surface properties propagate to env and writers."""

    def test_default_is_vacuum(self):
        env = _make_env(BoundaryProperties(acoustic_type='half-space',
                                           sound_speed=1600, density=1.5))
        assert env.surface.acoustic_type == 'vacuum'

    def test_ice_surface_stored(self, ice_surface, halfspace_bottom):
        env = _make_env(halfspace_bottom, surface=ice_surface)
        assert env.surface.sound_speed == 3500.0
        assert env.surface.shear_speed == 1800.0
        assert env.surface.density == 0.9

    def test_oases_writer_vacuum(self, halfspace_bottom):
        """OAST writer produces all-zero upper halfspace for vacuum."""
        from uacpy.io.oases_writer import _format_upper_halfspace
        env = _make_env(halfspace_bottom)
        line = _format_upper_halfspace(env)
        assert line == "0 0 0 0 0 0 0"

    def test_oases_writer_ice(self, ice_surface, halfspace_bottom):
        """OAST writer includes ice properties in upper halfspace."""
        from uacpy.io.oases_writer import _format_upper_halfspace
        env = _make_env(halfspace_bottom, surface=ice_surface)
        line = _format_upper_halfspace(env)
        assert "3500" in line
        assert "1800" in line
        assert "0.90" in line

    def test_oast_file_has_surface_props(self, ice_surface, halfspace_bottom,
                                         source, receiver, tmp_path):
        """Full OAST file includes ice surface in layer block."""
        from uacpy.io.oases_writer import write_oast_input
        env = _make_env(halfspace_bottom, surface=ice_surface)
        fpath = tmp_path / "test.dat"
        write_oast_input(fpath, env, source, receiver)
        content = fpath.read_text()
        lines = content.strip().split('\n')
        # The upper halfspace line is after the layer count line
        # Find the line count, then check the next line
        for i, line in enumerate(lines):
            if line.strip().isdigit() and int(line.strip()) >= 3:
                upper_hs = lines[i + 1]
                assert "3500" in upper_hs, f"Ice cp not in upper halfspace: {upper_hs}"
                break

    def test_oasn_file_has_surface_props(self, ice_surface, halfspace_bottom,
                                          source, receiver, tmp_path):
        """Full OASN file includes ice surface."""
        from uacpy.io.oases_writer import write_oasn_input
        env = _make_env(halfspace_bottom, surface=ice_surface)
        fpath = tmp_path / "test.dat"
        write_oasn_input(fpath, env, source, receiver)
        content = fpath.read_text()
        assert "3500" in content


# ── RAM: Top + Bottom Combinations ──────────────────────────────────────────

class TestRAMBoundaries:
    """RAM with various top and bottom boundary combinations."""

    @pytest.mark.requires_binary
    def test_ram_vacuum_halfspace(self, source, receiver, halfspace_bottom):
        env = _make_env(halfspace_bottom)
        field = RAM(verbose=False).run(env, source, receiver)
        tl = field.data
        assert tl.shape == (20, 3)
        assert np.all(np.isfinite(tl[5:-5, :]))
        assert 40 < np.nanmin(tl) < 80

    @pytest.mark.requires_binary
    def test_ram_single_layer(self, source, receiver, single_layer_bottom):
        env = _make_env(single_layer_bottom)
        field = RAM(verbose=False).run(env, source, receiver)
        tl = field.data
        assert tl.shape == (20, 3)
        assert np.all(np.isfinite(tl[5:-5, :]))

    @pytest.mark.requires_binary
    def test_ram_multi_layer(self, source, receiver, multi_layer_bottom):
        env = _make_env(multi_layer_bottom)
        field = RAM(verbose=False).run(env, source, receiver)
        tl = field.data
        assert tl.shape == (20, 3)
        assert np.all(np.isfinite(tl[5:-5, :]))

    @pytest.mark.requires_binary
    def test_ram_rd_layered(self, source, receiver, rd_layered_bottom):
        env = uacpy.Environment(
            name='rd_layered', depth=100,
            ssp_data=[(0, 1500), (100, 1500)],
            bottom=rd_layered_bottom,
        )
        field = RAM(verbose=False).run(env, source, receiver)
        tl = field.data
        assert tl.shape == (20, 3)
        assert np.all(np.isfinite(tl[5:-5, :]))

    @pytest.mark.requires_binary
    def test_ram_ice_and_layered(self, source, receiver,
                                 ice_surface, multi_layer_bottom):
        """RAM with ice surface + layered bottom (ice ignored by RAM,
        layered bottom handled)."""
        env = _make_env(multi_layer_bottom, surface=ice_surface)
        field = RAM(verbose=False).run(env, source, receiver)
        tl = field.data
        assert tl.shape == (20, 3)

    @pytest.mark.requires_binary
    def test_ram_halfspace_vs_layered_differ(self, source, receiver,
                                             halfspace_bottom,
                                             multi_layer_bottom):
        """Halfspace and layered bottoms should produce different TL."""
        env1 = _make_env(halfspace_bottom)
        env2 = _make_env(multi_layer_bottom)
        tl1 = RAM(verbose=False).run(env1, source, receiver).data
        tl2 = RAM(verbose=False).run(env2, source, receiver).data
        # They should not be identical
        assert not np.allclose(tl1, tl2, atol=0.5)

    @pytest.mark.requires_binary
    def test_ram_n_sed_points(self, source, receiver, multi_layer_bottom):
        """Custom n_sed_points works."""
        env = _make_env(multi_layer_bottom)
        field = RAM(verbose=False, n_sed_points=20).run(env, source, receiver)
        assert field.data.shape == (20, 3)


# ── Kraken/Scooter: Layered Bottom ──────────────────────────────────────────

class TestATModelsBoundaries:
    """Kraken and Scooter with layered bottoms."""

    def test_kraken_env_file_has_layers(self, source, receiver,
                                         multi_layer_bottom, tmp_path):
        """Kraken .env file includes sediment layer sections."""
        from uacpy.io.at_env_writer import ATEnvWriter
        env = _make_env(multi_layer_bottom)
        fpath = tmp_path / "test.env"
        with open(fpath, 'w') as f:
            ATEnvWriter.write_layer_sections(f, env, env.depth)
        content = fpath.read_text()
        # Should have layer entries (3 layers = 3 blocks)
        assert content.strip() != ''
        # Each layer writes 3 lines: mesh, top, bottom
        lines = [l for l in content.strip().split('\n') if l.strip()]
        assert len(lines) == 9  # 3 layers * 3 lines each

    def test_scooter_env_file_has_layers(self, source, receiver,
                                          single_layer_bottom, tmp_path):
        """Scooter env file includes single sediment layer."""
        from uacpy.io.at_env_writer import ATEnvWriter
        env = _make_env(single_layer_bottom)
        fpath = tmp_path / "test.env"
        with open(fpath, 'w') as f:
            ATEnvWriter.write_layer_sections(f, env, env.depth)
        content = fpath.read_text()
        lines = [l for l in content.strip().split('\n') if l.strip()]
        assert len(lines) == 3  # 1 layer * 3 lines


# ── KrakenField: Segmentation with Layered Bottom ───────────────────────────

class TestKrakenFieldSegmentation:
    """KrakenField segmentation preserves layered bottom."""

    def test_segment_preserves_layered_bottom(self, multi_layer_bottom):
        """Segmentation keeps LayeredBottom in each segment."""
        from uacpy.models.coupled_modes import segment_environment_by_range
        env = uacpy.Environment(
            name='test', depth=200,
            ssp_data=[(0, 1500), (200, 1500)],
            bathymetry=np.array([[0, 100], [20000, 200]]),
            bottom=multi_layer_bottom,
        )
        segments = segment_environment_by_range(env, n_segments=3)
        for _, seg_env in segments:
            assert seg_env.has_layered_bottom()
            assert len(seg_env.bottom_layered.layers) == 3

    def test_segment_preserves_rd_layered(self, rd_layered_bottom):
        """Segmentation selects nearest LayeredBottom profile."""
        from uacpy.models.coupled_modes import segment_environment_by_range
        env = uacpy.Environment(
            name='test', depth=100,
            ssp_data=[(0, 1500), (100, 1500)],
            bottom=rd_layered_bottom,
        )
        segments = segment_environment_by_range(env, n_segments=4)
        # Near segments should have 2 layers, far should have 1
        near_layers = segments[0][1].bottom_layered.layers
        far_layers = segments[-1][1].bottom_layered.layers
        assert len(near_layers) == 2
        assert len(far_layers) == 1


# ── get_bottom_at_range ──────────────────────────────────────────────────────

class TestGetBottomAtRange:
    """Environment.get_bottom_at_range returns correct type."""

    def test_halfspace_returns_boundary(self, halfspace_bottom):
        env = _make_env(halfspace_bottom)
        b = env.get_bottom_at_range(5000)
        assert isinstance(b, BoundaryProperties)

    def test_layered_returns_layered(self, multi_layer_bottom):
        env = _make_env(multi_layer_bottom)
        b = env.get_bottom_at_range(5000)
        assert isinstance(b, LayeredBottom)
        assert len(b.layers) == 3

    def test_rd_layered_nearest_profile(self, rd_layered_bottom):
        env = uacpy.Environment(
            name='test', depth=100,
            ssp_data=[(0, 1500), (100, 1500)],
            bottom=rd_layered_bottom,
        )
        # Near range: profile[0] has 2 layers
        b_near = env.get_bottom_at_range(1000)
        assert isinstance(b_near, LayeredBottom)
        assert len(b_near.layers) == 2

        # Far range: profile[1] has 1 layer
        b_far = env.get_bottom_at_range(9000)
        assert isinstance(b_far, LayeredBottom)
        assert len(b_far.layers) == 1


# ── OASES Writer Integration ────────────────────────────────────────────────

class TestOASESWriterIntegration:
    """Full OASES input files with various boundary conditions."""

    def test_oast_vacuum_halfspace(self, source, receiver,
                                    halfspace_bottom, tmp_path):
        from uacpy.io.oases_writer import write_oast_input
        env = _make_env(halfspace_bottom)
        fpath = tmp_path / "test.dat"
        write_oast_input(fpath, env, source, receiver)
        content = fpath.read_text()
        # Should have vacuum upper halfspace (all zeros)
        assert "0 0 0 0 0 0 0" in content

    def test_oast_layered_bottom(self, source, receiver,
                                  multi_layer_bottom, tmp_path):
        from uacpy.io.oases_writer import write_oast_input
        env = _make_env(multi_layer_bottom)
        fpath = tmp_path / "test.dat"
        write_oast_input(fpath, env, source, receiver)
        content = fpath.read_text()
        # Should include sediment layer sound speeds
        assert "1550" in content
        assert "1650" in content
        assert "1800" in content

    def test_oast_ice_surface(self, source, receiver,
                               ice_surface, halfspace_bottom, tmp_path):
        from uacpy.io.oases_writer import write_oast_input
        env = _make_env(halfspace_bottom, surface=ice_surface)
        fpath = tmp_path / "test.dat"
        write_oast_input(fpath, env, source, receiver)
        content = fpath.read_text()
        # Ice properties should appear in the upper halfspace line
        assert "3500" in content
        assert "1800" in content

    def test_oasp_ice_surface(self, source, receiver,
                               ice_surface, halfspace_bottom, tmp_path):
        from uacpy.io.oases_writer import write_oasp_input
        env = _make_env(halfspace_bottom, surface=ice_surface)
        fpath = tmp_path / "test.dat"
        write_oasp_input(fpath, env, source, receiver)
        content = fpath.read_text()
        assert "3500" in content
