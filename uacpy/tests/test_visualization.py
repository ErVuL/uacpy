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
        # Physical contract of the 2-D TL heatmap (not just "a fig came back"):
        # depth increases downward, the y-label names depth, and the default TL
        # colour scale is the fixed _TL_LIMITS so panels stay comparable.
        assert ax.yaxis_inverted()
        assert ax.get_ylabel() == 'Depth (m)'
        mesh = ax.collections[0]
        assert mesh.get_clim() == (20.0, 120.0)
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

    def test_rejects_empty_input(self):
        with pytest.raises(ConfigurationError, match="empty fields list"):
            plots.compare([])


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
        fig, ax = rays.plot(env=env)
        plt.close(fig)

    def test_edge_receiver_visible_via_xlim_margin(self, env):
        # A single receiver at the max range must be visible — not clipped to
        # the spine. Solved by an x-axis right margin (NOT clip_on=False, which
        # would paint out-of-view receivers across a later zoom). Markers keep
        # default clipping so a zoom hides them correctly.
        from uacpy.visualization.plots._common import (
            ZORDER_RECEIVERS, ZORDER_SOURCE,
        )
        rays = Rays(
            rays=[{'r': np.linspace(0, 2000, 50),
                   'z': 50 + 10 * np.sin(np.linspace(0, 5, 50)),
                   'alpha': 0.0, 'n_top_bounces': 0, 'n_bot_bounces': 0}],
            receiver_depths=np.array([50.0]),
            receiver_ranges=np.array([2000.0]),
            source_depths=np.array([10.0]),
            model='Bellhop',
        )
        fig, ax = rays.plot(env=env)
        by_z = {line.get_zorder(): line for line in ax.lines}
        # zoom-safe: default clipping on both markers
        assert by_z[ZORDER_RECEIVERS].get_clip_on() is True
        assert by_z[ZORDER_SOURCE].get_clip_on() is True
        # edge receiver at 2.0 km sits inside the axis (right margin past it)
        assert ax.get_xlim()[1] > 2.0
        plt.close(fig)

    def test_markers_clipped_on_user_zoom(self, env):
        # Regression (example_11b): a wide receiver grid must NOT paint
        # markers outside a user-set zoom window — they carry clip_on so
        # matplotlib clips them to the axes.
        from uacpy.visualization.plots._common import ZORDER_RECEIVERS
        rays = Rays(
            rays=[{'r': np.linspace(0, 100000, 60),
                   'z': 500 + 100 * np.sin(np.linspace(0, 8, 60)),
                   'alpha': 0.0, 'n_top_bounces': 0, 'n_bot_bounces': 0}],
            receiver_depths=np.array([500.0]),
            receiver_ranges=np.linspace(0.0, 100000.0, 21),  # 0–100 km
            source_depths=np.array([50.0]),
            model='Bellhop',
        )
        fig, ax = rays.plot(env=env)
        ax.set_xlim(20, 40)                       # user zoom, as in example_11b
        rcv = next(ln for ln in ax.lines
                   if ln.get_zorder() == ZORDER_RECEIVERS)
        assert rcv.get_clip_on() is True          # → far markers stay hidden
        plt.close(fig)

    def test_marker_sizes_visible(self, env):
        # Source/receiver markers are bumped in ray plots (receiver more) so
        # they read clearly against the ray fan.
        from uacpy.visualization.plots._common import (
            ZORDER_RECEIVERS, ZORDER_SOURCE,
        )
        rays = Rays(
            rays=[{'r': np.linspace(0, 2000, 50),
                   'z': 50 + 10 * np.sin(np.linspace(0, 5, 50)),
                   'alpha': 0.0, 'n_top_bounces': 0, 'n_bot_bounces': 0}],
            receiver_depths=np.array([50.0]),
            receiver_ranges=np.array([2000.0]),
            source_depths=np.array([10.0]),
            model='Bellhop',
        )
        fig, ax = rays.plot(env=env)
        by_z = {line.get_zorder(): line for line in ax.lines}
        assert by_z[ZORDER_RECEIVERS].get_markersize() >= 6
        assert by_z[ZORDER_SOURCE].get_markersize() >= 16
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
        fig, ax = arr.plot()
        plt.close(fig)


class TestPlotEnvironment:
    def test_flat_env(self, env):
        fig, _ = env.plot()
        plt.close(fig)


class TestBottomTitles:
    """The default bottom titles must name the bottom's *shape*. A class name
    no longer discriminates — ``core.bottom`` declares one ``Bottom``."""

    @staticmethod
    def _layer(**kw):
        from uacpy.core.environment import SedimentLayer
        return SedimentLayer(thickness=15, sound_speed=1650, density=1.6,
                             attenuation=0.4, **kw)

    @staticmethod
    def _column(layers):
        from uacpy.core import BoundaryProperties
        from uacpy.core.environment import SeabedColumn
        return SeabedColumn(
            layers=layers,
            halfspace=BoundaryProperties(acoustic_type='half-space',
                                         sound_speed=2000, density=2.0,
                                         attenuation=0.2))

    def _env(self, bottom):
        return uacpy.Environment(name='bt', bathymetry=[(0, 100), (5000, 120)],
                                 ssp=[(0, 1500), (120, 1490)], bottom=bottom)

    @pytest.fixture
    def bottoms(self):
        from uacpy.core.bottom import Bottom
        return {
            'half-space': Bottom.from_column(self._column([])),
            'layered': Bottom.from_column(self._column([self._layer()])),
            'range-dependent half-space, 2 ranges': Bottom.from_columns(
                [self._column([]), self._column([])], ranges=[0.0, 5000.0]),
            'range-dependent layered, 2 ranges': Bottom.from_columns(
                [self._column([self._layer()]), self._column([self._layer()])],
                ranges=[0.0, 5000.0]),
        }

    def test_environment_title_names_the_shape(self, bottoms):
        for expected, bottom in bottoms.items():
            fig, ax = self._env(bottom).plot(data_source=None)
            assert ax.get_title() == f"Bottom — {expected}"
            plt.close(fig)

    def test_bottom_properties_title_names_the_shape(self, bottoms):
        for expected, bottom in bottoms.items():
            fig, _ = plots.plot_bottom_properties(self._env(bottom),
                                                  data_source=None)
            assert fig._suptitle.get_text() == f"Seabed properties — {expected}"
            plt.close(fig)

    def test_explicit_title_still_wins(self, env):
        fig, ax = env.plot(title='Custom')
        assert ax.get_title() == 'Custom'
        plt.close(fig)


class TestPlotRangeProfile:
    def test_title_distinguishes_bathymetry_from_altimetry(self):
        bathy = uacpy.Bathymetry(ranges=[0.0, 1000.0], depths=[100.0, 120.0])
        alti = uacpy.Altimetry(ranges=[0.0, 1000.0], heights=[0.0, 1.0])
        fig, ax = bathy.plot()
        assert ax.get_title() == 'Bathymetry profile'
        plt.close(fig)
        fig, ax = alti.plot()
        assert ax.get_title() == 'Altimetry profile'
        plt.close(fig)


class TestPlotSSP:
    def test_profile_input_depth_down_single_line(self):
        ssp = uacpy.SoundSpeedProfile.from_pairs([(0, 1520), (100, 1490),
                                                  (200, 1480)])
        fig, ax = ssp.plot()
        assert ax.get_xlabel() == 'Sound speed (m/s)'
        assert ax.get_ylabel() == 'Depth (m)'
        assert ax.yaxis_inverted()          # depth positive down
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_environment_input(self, env):
        fig, ax = env.ssp.plot()             # isovelocity env
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_range_dependent_one_line_per_range(self):
        ssp = uacpy.SoundSpeedProfile.from_2d(
            depths=[0, 100, 200], ranges=[0, 5000, 10000],
            matrix=np.array([[1520, 1510, 1500],
                             [1500, 1495, 1490],
                             [1480, 1478, 1475]]))
        fig, ax = ssp.plot()
        assert len(ax.lines) == 3            # one per range column
        assert len(fig.axes) > 1             # range colorbar
        plt.close(fig)

    def test_into_existing_axis_not_double_inverted(self):
        fig, ax = plt.subplots()
        ax.invert_yaxis()                    # already depth-down
        ssp = uacpy.SoundSpeedProfile.from_pairs([(0, 1500), (100, 1490)])
        _, ax_out = ssp.plot(ax=ax)
        assert ax_out is ax and ax.yaxis_inverted()
        plt.close(fig)

    def test_bad_input_raises(self):
        from uacpy.visualization.plots.environment import _plot_ssp
        with pytest.raises(ConfigurationError):
            _plot_ssp(42)


class TestPlotBottomProperties:
    def _env(self, bottom, bathy=None):
        from uacpy.core import Environment
        return Environment(name='bp', bathymetry=bathy or [(0, 100), (5000, 120)],
                           ssp=[(0, 1500), (120, 1490)], bottom=bottom)

    def test_elastic_layered_shows_five_panels(self):
        from uacpy.core import BoundaryProperties
        from uacpy.core.environment import SeabedColumn, SedimentLayer
        lay = SeabedColumn(
            layers=[SedimentLayer(thickness=15, sound_speed=1650, density=1.6,
                                  attenuation=0.4, shear_speed=300,
                                  shear_attenuation=0.3)],
            halfspace=BoundaryProperties(acoustic_type='half-space',
                                         sound_speed=2000, density=2.0,
                                         attenuation=0.2, shear_speed=600,
                                         shear_attenuation=0.5))
        fig, axes = plots.plot_bottom_properties(self._env(lay), data_source=None)
        # cp, cs, ρ, αp, αs all present and non-zero → 5 visible panels.
        assert sum(a.get_visible() for a in axes.ravel()) == 5
        plt.close(fig)

    def test_fluid_halfspace_skips_shear(self):
        from uacpy.core import BoundaryProperties
        hs = BoundaryProperties(acoustic_type='half-space', sound_speed=1800,
                                density=1.8, attenuation=0.3)
        fig, axes = plots.plot_bottom_properties(self._env(hs), data_source=None)
        # No shear → cp, ρ, αp only.
        assert sum(a.get_visible() for a in axes.ravel()) == 3
        plt.close(fig)

    def test_properties_filter(self):
        from uacpy.core import BoundaryProperties
        hs = BoundaryProperties(acoustic_type='half-space', sound_speed=1800,
                                density=1.8, attenuation=0.3)
        fig, axes = plots.plot_bottom_properties(
            self._env(hs), properties=['cp'], data_source=None)
        assert sum(a.get_visible() for a in axes.ravel()) == 1
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
        fig, _ = modes.plot()
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
        fig, _ = rc.plot()
        plt.close(fig)

    def test_broadband(self):
        theta = np.linspace(0, 90, 31)
        freqs = np.linspace(50, 500, 10)
        R = np.tile(np.linspace(1.0, 0.0, 31)[:, None], (1, 10))
        rc = ReflectionCoefficient(
            theta=theta, R=R, phi=np.zeros_like(R),
            frequencies=freqs, model='Bounce',
        )
        fig, _ = rc.plot()
        plt.close(fig)


class TestTLLimits:
    """The fixed TL colour scale used everywhere TL is drawn."""

    def test_fixed_limits(self):
        from uacpy.visualization.plots._common import _TL_LIMITS
        assert _TL_LIMITS == (20.0, 120.0)


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
            lats, lons, depth, basemap=False,
            transect=((42, 4), (38.3, 6)), title='t')
        assert fig is not None and ax.has_data()
        assert ax.get_xlabel().startswith('Longitude')
        plt.close(fig)

    def test_relief_orientation_invariant_to_lat_order(self):
        # The shaded-relief path uses imshow(origin='lower'), which assumes
        # row 0 of the array is the southernmost. A descending lat axis must be
        # flipped to that canonical order so the relief image matches the true
        # geography (and the flat pcolormesh path) — not render upside down.
        lons = np.linspace(0, 10, 10)
        lats_up = np.linspace(36, 44, 8)            # ascending S→N
        rng = np.random.default_rng(0)
        depth_up = rng.uniform(500, 3000, (8, 10))
        depth_up[-1, 0] = np.nan                    # land at the NORTH-WEST corner

        def relief_rgba(lats, depth):
            fig, ax = plots.plot_bathymetry_map(
                lats, lons, depth, basemap=False, relief=True, graticule=None)
            im = ax.get_images()[0]                 # the relief AxesImage
            rgba = np.asarray(im.get_array())
            plt.close(fig)
            return rgba

        up = relief_rgba(lats_up, depth_up)
        # Same geography, lat axis reversed (descending N→S) and rows flipped
        # to match — the rendered image must be identical.
        down = relief_rgba(lats_up[::-1], depth_up[::-1, :])
        assert np.array_equal(up, down, equal_nan=True)
        # imshow(origin='lower') ⇒ display row -1 is the northern edge; the land
        # cell sits there (top-left), transparent (alpha == 0).
        assert up[-1, 0, 3] == 0.0

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


class TestReliefLighting:
    """The shaded relief must light real seafloor *relief* from the requested
    azimuth.

    ``LightSource.hillshade`` negates ``dy`` (matplotlib ``colors.py``: "most
    image and raster GIS data has the first row in the array as the top of the
    image ... dy is implicitly negative"), so an array whose row index grows
    northward under ``origin='lower'`` has to pass ``-dy`` to mean what it
    says. And the field is water depth, positive *down*, so the shader must be
    given ``-depth`` or a seamount lights like a pit.

    The four probes below sit at equal radius from the feature's centre, so
    they share a depth and therefore a base colour: any luminance difference
    between them is the hillshade alone.
    """

    # cos(40.5 deg) ~ 0.76, so this longitude span makes the cells square in
    # metres and the four probes symmetric under the shader's own metric.
    _LATS = np.linspace(40.0, 41.0, 61)
    _LONS = np.linspace(0.0, 1.316, 61)

    def _feature(self, sign):
        lo, la = np.meshgrid(self._LONS, self._LATS)
        r2 = (((lo - self._LONS.mean()) / np.ptp(self._LONS)) ** 2
              + ((la - self._LATS.mean()) / np.ptp(self._LATS)) ** 2)
        return 3000.0 - sign * 1000.0 * np.exp(-r2 / (2 * 0.12 ** 2))

    def _flank_luminance(self, depth):
        """Rendered luminance at the NW / NE / SW / SE flanks."""
        fig, ax = plots.plot_bathymetry_map(
            self._LATS, self._LONS, depth, basemap=False, relief=True,
            graticule=None)
        rgba = np.asarray(ax.get_images()[0].get_array())
        plt.close(fig)
        lum = rgba[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
        mid, off = 30, 12          # origin='lower': row grows northward
        return {'NW': lum[mid + off, mid - off], 'NE': lum[mid + off, mid + off],
                'SW': lum[mid - off, mid - off], 'SE': lum[mid - off, mid + off]}

    def test_a_seamount_is_lit_from_the_northwest(self):
        lum = self._flank_luminance(self._feature(+1))
        assert lum['NW'] == max(lum.values()), (
            f"azdeg=315 requested, brightest flank is "
            f"{max(lum, key=lum.get)}: {lum}")
        assert lum['SE'] == min(lum.values()), lum

    def test_a_basin_lights_the_opposite_flank(self):
        """Same geometry, inverted: the wall a depression turns toward the
        light is its far (SE) one. Depth-as-elevation would not flip."""
        lum = self._flank_luminance(self._feature(-1))
        assert lum['SE'] == max(lum.values()), (
            f"a basin lit like a seamount: {lum}")
        assert lum['NW'] == min(lum.values()), lum

    def test_cell_size_and_dy_sign_reach_the_shader(self, monkeypatch):
        """n samples span n-1 intervals, and the row order is south→north."""
        from matplotlib.colors import LightSource
        seen = {}
        real = LightSource.shade_rgb

        def spy(self, rgb, elevation, **kw):
            seen.update(kw)
            seen['elevation_sign'] = float(np.sign(np.mean(elevation)))
            return real(self, rgb, elevation, **kw)

        monkeypatch.setattr(LightSource, 'shade_rgb', spy)
        self._flank_luminance(self._feature(+1))

        deg_m = 111320.0
        assert seen['dy'] == pytest.approx(
            -np.ptp(self._LATS) / (self._LATS.size - 1) * deg_m)
        assert seen['dx'] == pytest.approx(
            np.ptp(self._LONS) / (self._LONS.size - 1) * deg_m
            * np.cos(np.radians(self._LATS.mean())))
        assert seen['elevation_sign'] == -1.0      # height, not depth


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
            tl=tl_field, source=src, title='ov')
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
    """plot_sea_ice_map (mirrors the depth map) + env.plot(sea_ice=)."""

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
        fig0, ax0 = env.plot()            # no ice
        n0 = len(ax0.collections)
        plt.close(fig0)
        fig, ax = env.plot(sea_ice=0.8)
        assert len(ax.collections) > n0                    # ice band added at surface
        plt.close(fig)


def test_compare_models_label_length_validation(tl_field):
    with pytest.raises(ConfigurationError, match="match"):
        uacpy.plot.compare_models([tl_field, tl_field, tl_field],
                                  labels=['only-one'])


def test_plot_environment_rejects_non_environment():
    from uacpy.visualization.plots.environment import _plot_environment
    with pytest.raises(ConfigurationError, match="Environment"):
        _plot_environment("not an environment")
