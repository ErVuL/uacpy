"""Smoke tests for the canonical visualization surface.

Each canonical plot function is exercised with synthetic data to confirm
it produces a figure without raising. Numerical correctness of the plots
is not asserted — that's :mod:`uacpy.core.metrics` and the model
regression tests' job.
"""

import numpy as np
import matplotlib  # noqa: E402 ordering — must precede pyplot
matplotlib.use('Agg')

import pytest  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import uacpy  # noqa: E402
from uacpy.core.exceptions import ConfigurationError  # noqa: E402
from uacpy.core.results import (  # noqa: E402
    Field, Modes, Arrivals, Rays, ReflectionCoefficient,
)
from uacpy.visualization import plots  # noqa: E402


@pytest.fixture
def env():
    return uacpy.Environment(name='Test', bathymetry=100.0, ssp=1500.0)


@pytest.fixture
def tl_field():
    d = np.linspace(5, 95, 12)
    r = np.linspace(100, 5000, 40)
    data = 50.0 + 10.0 * np.log10(np.maximum(r, 1.0)[None, :])
    return Field(
        data=np.broadcast_to(data, (12, 40)).copy(),
        coords={'depth': d, 'range': r},
        model='Synth', frequencies=100.0,
    )


@pytest.fixture
def complex_field():
    d = np.linspace(5, 95, 12)
    r = np.linspace(100, 5000, 40)
    rng = np.random.default_rng(0)
    p = (rng.standard_normal((12, 40)) + 1j * rng.standard_normal((12, 40))) * 1e-3
    return Field(
        data=p, coords={'depth': d, 'range': r},
        model='Synth', frequencies=100.0,
    )


@pytest.fixture
def broadband_field():
    d = np.linspace(5, 95, 8)
    r = np.linspace(100, 5000, 20)
    f = np.linspace(50, 500, 10)
    rng = np.random.default_rng(1)
    data = (rng.standard_normal((8, 20, 10))
            + 1j * rng.standard_normal((8, 20, 10))) * 1e-3
    return Field(
        data=data,
        coords={'depth': d, 'range': r, 'frequency': f},
        phase_reference='travelling_wave',
        model='Synth', frequencies=f,
    )


class TestPlotField:
    """``plot_field`` auto-shapes based on what survives in
    :attr:`Field.coords` after slicing."""

    def test_2d_heatmap_complex(self, complex_field, env):
        fig, ax = plots.plot_field(complex_field, env=env)
        assert fig is not None and ax is not None
        plt.close(fig)

    def test_2d_heatmap_real_dB(self, tl_field, env):
        fig, ax = plots.plot_field(tl_field, env=env)
        plt.close(fig)

    def test_1d_range_cut_via_at(self, tl_field):
        fig, ax = plots.plot_field(tl_field.at(depth=50.0))
        plt.close(fig)

    def test_1d_depth_cut_via_at(self, tl_field):
        fig, ax = plots.plot_field(tl_field.at(range=2000.0))
        plt.close(fig)

    def test_broadband_at_frequency_drops_to_2d(self, broadband_field):
        narrow = broadband_field.at(frequency=200.0)
        assert list(narrow.coords) == ['depth', 'range']
        fig, ax = plots.plot_field(narrow)
        plt.close(fig)

    def test_broadband_spectrum_1d(self, broadband_field):
        spec = broadband_field.at(depth=50.0, range=2500.0)
        assert list(spec.coords) == ['frequency']
        fig, ax = plots.plot_field(spec)
        plt.close(fig)

    def test_phase_value(self, complex_field):
        fig, ax = plots.plot_field(complex_field, value='phase')
        plt.close(fig)


class TestCompare:
    """``compare`` overlays 1-D sliced fields on one axes."""

    def test_overlay_two_range_cuts(self, tl_field):
        a = tl_field.at(depth=50.0)
        b = Field(
            data=tl_field.data + 2.0,
            coords=dict(tl_field.coords),
        ).at(depth=50.0)
        fig, ax = plots.compare([a, b], labels=['A', 'B'])
        plt.close(fig)

    def test_rejects_2d_input(self, tl_field):
        with pytest.raises(ConfigurationError, match="1 surviving axis"):
            plots.compare([tl_field, tl_field])


class TestCompareModels:
    def test_grid_of_heatmaps(self, tl_field, env):
        fig, axes = plots.compare_models(
            [tl_field, Field(data=tl_field.data + 1.0,
                             coords=dict(tl_field.coords),
                             model='B')],
            labels=['A', 'B'], env=env,
        )
        plt.close(fig)


class TestPlotRays:
    def test_basic_ray_fan(self, env):
        rays = Rays(
            rays=[
                {'r': np.linspace(0, 5000, 50),
                 'z': 50 + 20 * np.sin(np.linspace(0, 5, 50)),
                 'alpha': 0.0, 'n_top_bounces': 0, 'n_bot_bounces': 0},
                {'r': np.linspace(0, 5000, 50),
                 'z': 50 + 30 * np.cos(np.linspace(0, 6, 50)),
                 'alpha': 5.0, 'n_top_bounces': 1, 'n_bot_bounces': 0},
            ],
            receiver_depths=np.array([50.0]),
            receiver_ranges=np.array([2000.0]),
            model='Bellhop',
        )
        fig, ax = plots.plot_rays(rays, env=env)
        plt.close(fig)


class TestPlotArrivals:
    def test_stem_plot(self):
        arr = Arrivals(
            arrivals=[
                {'delay': 0.5, 'amplitude': 1.0, 'phase': 0.0,
                 'n_top_bounces': 0, 'n_bot_bounces': 0, 'src_angle': 0,
                 'rcv_angle': 0, 'kind': 'direct',
                 'src_idx': 0, 'depth_idx': 0, 'range_idx': 0},
                {'delay': 0.7, 'amplitude': 0.5, 'phase': 1.0,
                 'n_top_bounces': 1, 'n_bot_bounces': 0, 'src_angle': 0,
                 'rcv_angle': 0, 'kind': 'surface',
                 'src_idx': 0, 'depth_idx': 0, 'range_idx': 0},
            ],
            receiver_depths=np.array([50.0]),
            receiver_ranges=np.array([2000.0]),
            model='Bellhop',
        )
        fig, ax = plots.plot_arrivals(arr)
        plt.close(fig)


class TestPlotEnvironment:
    def test_flat_env(self, env):
        fig, _ = plots.plot_environment(env)
        plt.close(fig)


class TestPlotModes:
    @pytest.fixture
    def modes(self):
        depths = np.linspace(0, 100, 21)
        n = 5
        phi = np.zeros((21, n))
        k = np.empty(n, dtype=complex)
        for m in range(n):
            kz = (m + 0.5) * np.pi / 100.0
            phi[:, m] = np.sin(kz * depths)
            k[m] = np.sqrt((2 * np.pi * 100 / 1500) ** 2 - kz ** 2 + 0j)
        return Modes(k=k, phi=phi, depths=depths, model='Test', frequencies=100.0)

    def test_plot_mode_functions(self, modes):
        fig, _ = plots.plot_mode_functions(modes)
        plt.close(fig)

    def test_plot_mode_wavenumbers(self, modes):
        fig, _ = plots.plot_mode_wavenumbers(modes)
        plt.close(fig)

    def test_plot_modes_heatmap(self, modes):
        fig, _ = plots.plot_modes_heatmap(modes)
        plt.close(fig)


class TestPlotReflectionCoefficient:
    def test_narrowband(self):
        rc = ReflectionCoefficient(
            theta=np.linspace(0, 90, 91),
            R=np.linspace(1.0, 0.0, 91),
            phi=np.zeros(91),
            model='Bounce',
        )
        fig, _ = plots.plot_reflection_coefficient(rc)
        plt.close(fig)

    def test_broadband(self):
        theta = np.linspace(0, 90, 31)
        freqs = np.linspace(50, 500, 10)
        R = np.tile(np.linspace(1.0, 0.0, 31)[:, None], (1, 10))
        rc = ReflectionCoefficient(
            theta=theta, R=R, phi=np.zeros_like(R),
            frequencies=freqs, model='Bounce',
        )
        fig, _ = plots.plot_reflection_coefficient(rc)
        plt.close(fig)


class TestAutoTLLimits:
    """The internal helper used by ``plot_field`` / ``compare_models`` clips
    Bellhop's TL sentinel out of the auto-scale window."""

    def test_sentinel_removed(self):
        from uacpy.visualization.plots import _auto_tl_limits
        rng = np.random.default_rng(0)
        body = 50.0 + 10.0 * rng.standard_normal((30, 30))
        data = np.full((40, 40), 600.0)
        data[:30, :30] = body
        vmin, vmax = _auto_tl_limits(data)
        assert vmax < 200.0
        assert vmin < vmax

    def test_no_finite_falls_back_to_default(self):
        from uacpy.visualization.plots import _auto_tl_limits
        vmin, vmax = _auto_tl_limits(np.full((4, 4), np.nan))
        assert (vmin, vmax) == (30.0, 80.0)


class TestBathymetryMap:
    """plot_bathymetry_map — plain, coastline (mocked), and unreachable paths."""

    @staticmethod
    def _grid():
        lats = np.linspace(36, 44, 8)
        lons = np.linspace(0, 10, 10)
        depth = np.random.default_rng(0).uniform(500, 3000, (8, 10))
        depth[0, 0] = np.nan                 # a land cell
        return lats, lons, depth

    def test_plain_lonlat(self):
        lats, lons, depth = self._grid()
        fig, ax = plots.plot_bathymetry_map(
            lats, lons, depth, basemap=None,
            transect=((42, 4), (38.3, 6)), title='t')
        assert fig is not None and ax.has_data()
        assert ax.get_xlabel().startswith('Longitude')
        plt.close(fig)

    def test_coastline_default(self, monkeypatch):
        # Default backdrop = Natural Earth coastlines (public domain); stub fetch.
        ring = np.array([[2, 40], [4, 40], [4, 42], [2, 42], [2, 40]], float)
        monkeypatch.setattr('uacpy.visualization.basemap.land_polygons',
                            lambda *a, **k: [ring])
        lats, lons, depth = self._grid()
        fig, ax = plots.plot_bathymetry_map(lats, lons, depth)   # default 'coastline'
        assert any('°E' in t.get_text() for t in ax.get_xticklabels())
        plt.close(fig)

    def test_coastline_unreachable_still_draws(self, monkeypatch):
        monkeypatch.setattr('uacpy.visualization.basemap.land_polygons',
                            lambda *a, **k: None)
        lats, lons, depth = self._grid()
        fig, ax = plots.plot_bathymetry_map(lats, lons, depth)   # sea only, no crash
        assert ax.has_data()
        plt.close(fig)

    def test_coastline_with_transect(self, monkeypatch):
        ring = np.array([[2, 40], [4, 40], [4, 42], [2, 42], [2, 40]], float)
        monkeypatch.setattr('uacpy.visualization.basemap.land_polygons',
                            lambda *a, **k: [ring])
        lats, lons, depth = self._grid()
        fig, ax = plots.plot_bathymetry_map(
            lats, lons, depth, transect=((42, 4), (38.3, 6)))
        assert ax.get_legend() is not None        # transect labelled
        plt.close(fig)


class TestOverview:
    """plot_overview — the one-call map · TL · environment composite."""

    @staticmethod
    def _grid():
        lats = np.linspace(36, 44, 8)
        lons = np.linspace(0, 10, 10)
        depth = np.random.default_rng(0).uniform(500, 3000, (8, 10))
        return lats, lons, depth

    def test_full_composite(self, env, tl_field):
        src = uacpy.Source(depths=50.0, frequencies=100.0)
        fig, axes = plots.plot_overview(
            env, self._grid(), transect=((42, 4), (38.3, 6)),
            tl=tl_field, source=src, suptitle='ov')
        assert fig is not None and len(axes) == 3
        ax_map, ax_tl, ax_env = axes
        assert ax_map.has_data() and ax_tl.has_data() and ax_env.has_data()
        # accessible as a first-class library function
        assert uacpy.plot.plot_overview is plots.plot_overview
        plt.close(fig)

    def test_without_tl_leaves_placeholder(self, env):
        fig, (ax_map, ax_tl, ax_env) = plots.plot_overview(env, self._grid())
        assert any('no TL' in t.get_text() for t in ax_tl.texts)
        plt.close(fig)


class TestSeaIce:
    """plot_sea_ice_map (mirrors the depth map) + plot_environment(sea_ice=)."""

    def test_sea_ice_map_notation(self, monkeypatch):
        import uacpy.data.seaice_local as sil
        # mock the reprojection so no cache/pyproj is needed
        monkeypatch.setattr(sil, 'sea_ice_pixel', lambda pt, hemi='N': (2, 3))
        grid = np.random.default_rng(0).uniform(0, 1, (8, 10))
        grid[0, 0] = np.nan                            # land cell
        fig, ax = plots.plot_sea_ice_map(
            grid, transect=((88, 0), (79, -3)), source=(88, 0), title='ice')
        assert ax.has_data()
        assert ax.get_legend() is not None             # 'transect' legend, as on the map
        assert uacpy.plot.plot_sea_ice_map is plots.plot_sea_ice_map
        plt.close(fig)

    def test_environment_shows_ice(self, env):
        fig0, ax0 = plots.plot_environment(env)            # no ice
        n0 = len(ax0.collections)
        plt.close(fig0)
        fig, ax = plots.plot_environment(env, sea_ice=0.8)
        assert len(ax.collections) > n0                    # ice band added at surface
        plt.close(fig)
