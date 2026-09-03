"""Smoke tests for the canonical visualization surface.

Each canonical plot function is exercised with synthetic data to confirm
it produces a figure without raising. Numerical correctness of the plots
is not asserted — that's :mod:`uacpy.core.metrics` and the model
regression tests' job.
"""

import inspect
import warnings
from pathlib import Path

import numpy as np
import pytest
import matplotlib.collections as mcoll
import matplotlib.figure as mfig
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

import uacpy
from uacpy import (Bathymetry, Bottom, BoundaryProperties, Environment,
                   SeabedColumn, SedimentLayer)
from uacpy.core.bottom import Bottom as _Bottom
from uacpy.core.constants import REFERENCE_PRESSURE_WATER
from uacpy.core.exceptions import ConfigurationError
from uacpy.visualization.plots.environment import _seabed_property_grid
from uacpy.core.results import (
    Field, Modes, Arrivals, Rays, ReflectionCoefficient,
    Covariance, Replicas, ResultStack,
)
from uacpy.acoustic_signal.analysis import PPSDResult
from uacpy.acoustic_signal.constant_q import CQPPSDResult
from uacpy.noise import WenzNoise
from uacpy.visualization import plots
from uacpy.visualization.plots import plot_beam_pattern, plot_field
from uacpy.visualization.plots import fields as _fields
from uacpy.visualization.plots.fields import (compare,
                                              plot_detection_probability,
                                              plot_signal_excess)
from uacpy.data._geo import lon_linspace
from uacpy.visualization.plots.maps import (_lon_label_value,
                                            _unwrap_lon_axis,
                                            plot_overview)
from uacpy.visualization.plots.noise import plot_source_level, plot_wenz
from uacpy.visualization.plots.rays_modes import _plot_rays, plot_modes_heatmap
from uacpy.visualization.plots.signal import (_clamped_freq_limits,
                                              _log_freq_xlim, _ref_label,
                                              plot_constant_q_ppsd,
                                              plot_constant_q_psd,
                                              plot_constant_q_spectrogram,
                                              plot_frf, plot_ppsd, plot_psd,
                                              plot_sel, plot_spectrogram)
from uacpy.visualization.plots.fields import compare_models


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


@pytest.fixture
def time_field():
    """Real 2-D ``(range, time)`` field — the stacked-traces / time-heatmap input."""
    r = np.linspace(100.0, 2000.0, 6)
    t = np.linspace(0.0, 0.1, 5)
    data = np.random.default_rng(2).standard_normal((6, 5))
    return Field(
        data=data, coords={'range': r, 'time': t},
        model='Synth', frequencies=100.0,
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
        # Phase heatmaps draw on the cyclic 'twilight' colormap with the
        # fixed ±π colour scale.
        mesh = ax.collections[0]
        assert mesh.get_cmap().name == 'twilight'
        assert mesh.get_clim() == pytest.approx((-np.pi, np.pi))
        plt.close(fig)

    def test_three_axis_field_raises_slice_first(self, broadband_field):
        # No axis of broadband_field is singleton, so nothing auto-squeezes:
        # three surviving axes have no picture and the error says to slice.
        with pytest.raises(ConfigurationError,
                           match=r"cannot plot a 3-axis field .*slice it "
                                 r"first"):
            broadband_field.plot()


class TestValueModeDefaults:
    """``value=`` defaults to ``'real'`` if and only if the field carries a
    ``time`` axis, and ``'db'`` otherwise; each default brings its own colour
    treatment."""

    def test_time_field_defaults_to_real_with_seismic_rms_clim(self, time_field):
        fig, ax = plots.plot_field(time_field)
        mesh = ax.collections[0]
        # The drawn array is the raw real data — the 'real' default.
        assert np.allclose(np.asarray(mesh.get_array()).ravel(),
                           time_field.data.ravel())
        # Real time-domain pressure gets the diverging 'seismic' colormap
        # clipped symmetrically to ±RMS.
        assert mesh.get_cmap().name == 'seismic'
        rms = float(np.sqrt(np.mean(time_field.data ** 2)))
        assert mesh.get_clim() == pytest.approx((-rms, rms))
        plt.close(fig)

    def test_non_time_field_defaults_to_db(self, complex_field):
        fig, ax = plots.plot_field(complex_field)
        mesh = ax.collections[0]
        # The drawn array is the dB view, on the fixed TL scale.
        assert np.allclose(np.asarray(mesh.get_array()).ravel(),
                           complex_field.db.ravel())
        assert mesh.get_clim() == (20.0, 120.0)
        plt.close(fig)


class TestStackedTraces:
    """``stacked=True`` draws one vertically offset trace per row of a 2-D
    field carrying a ``time`` axis, and is rejected anywhere else."""

    def test_rejected_on_a_field_without_a_time_axis(self, tl_field):
        with pytest.raises(ConfigurationError,
                           match=r"stacked=True.*requires a 2-D field with "
                                 r"a 'time' axis"):
            plots.plot_field(tl_field, stacked=True)

    def test_per_trace_vertical_offsets(self, time_field):
        fig, ax = plots.plot_field(time_field, stacked=True, stack_offset=1.5)
        # One Line2D per range row, each lifted by i · stack_offset.
        assert len(ax.lines) == time_field.coords['range'].size
        for i, line in enumerate(ax.lines):
            np.testing.assert_allclose(
                line.get_ydata() - time_field.data[i], i * 1.5)
        plt.close(fig)


class TestKwargOwnershipRejections:
    """Every ``plot_field`` knob is owned by one render branch; a knob handed
    to a branch that cannot read it raises before any figure exists."""

    @pytest.mark.parametrize('case, match', [
        ('vmin_on_line_cut',
         r"vmin= has no effect on a 1-D line cut"),
        ('label_on_heatmap',
         r"label= has no effect on a 2-D heatmap"),
        ('env_on_time_heatmap',
         r"env= has no effect on a heatmap that is not a "
         r"\(depth, range\) cross-section"),
    ])
    def test_foreign_kwarg_rejected(self, case, match, tl_field, time_field,
                                    env):
        calls = {
            'vmin_on_line_cut': (tl_field.at(depth=50.0), {'vmin': 20.0}),
            'label_on_heatmap': (tl_field, {'label': 'cut'}),
            'env_on_time_heatmap': (time_field, {'env': env}),
        }
        field, kwargs = calls[case]
        with pytest.raises(ConfigurationError, match=match):
            plots.plot_field(field, **kwargs)
        assert not plt.get_fignums()


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

    def test_grid_mismatch_warns(self, tl_field):
        # Two fields on different range grids share one colourbar, so the
        # mismatch is announced as a UserWarning naming the odd axis.
        shifted = Field(
            data=tl_field.data.copy(),
            coords={'depth': tl_field.coords['depth'],
                    'range': tl_field.coords['range'] + 250.0},
            model='B', frequencies=100.0,
        )
        with pytest.warns(UserWarning,
                          match=r"'B' range axis differs from 'A'"):
            fig, _ = plots.compare_models([tl_field, shifted],
                                          labels=['A', 'B'])
        plt.close(fig)


class TestResultStackPlot:
    """``stack.plot()`` renders one titled heatmap panel per slab."""

    @staticmethod
    def _slab(amp):
        d = np.linspace(10.0, 90.0, 5)
        r = np.linspace(100.0, 3000.0, 8)
        return Field(data=np.full((5, 8), amp),
                     coords={'depth': d, 'range': r},
                     model='Synth', frequencies=100.0)

    def test_panel_grid_one_axes_per_slab(self):
        stack = ResultStack(
            [self._slab(40.0), self._slab(60.0), self._slab(80.0)],
            [10.0, 20.0, 30.0], coordinate_name='source_depth')
        fig, axes = stack.plot()
        axes = np.asarray(axes)
        assert axes.shape == (1, 3)
        # Every grid cell holds a drawn mesh — panel count == slab count.
        drawn = [a for a in axes.ravel()
                 if any(hasattr(c, 'get_coordinates') for c in a.collections)]
        assert len(drawn) == stack.n_slabs
        assert [a.get_title() for a in axes.ravel()] == [
            'source_depth=10', 'source_depth=20', 'source_depth=30']
        plt.close(fig)


class TestPlotRays:
    def test_ray_fan_plot_smoke(self, env):
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

    def test_receiver_lattice_decimated(self):
        # A 40-range × 30-depth lattice draws 20 × 10 markers: each axis is
        # decimated by step size // cap (caps 20 range dots, 10 depth dots).
        from uacpy.visualization.plots._common import ZORDER_RECEIVERS
        rays = Rays(
            rays=[{'r': np.linspace(0, 2000, 10),
                   'z': np.linspace(10, 60, 10),
                   'alpha': 0.0, 'n_top_bounces': 0, 'n_bot_bounces': 0}],
            receiver_depths=np.linspace(5.0, 25.0, 30),
            receiver_ranges=np.linspace(100.0, 2000.0, 40),
            model='Bellhop',
        )
        fig, ax = rays.plot()
        rcv = next(ln for ln in ax.lines
                   if ln.get_zorder() == ZORDER_RECEIVERS)
        assert len(rcv.get_xdata()) == 20 * 10
        plt.close(fig)

    @staticmethod
    def _classed_rays(classes):
        def ray(z_peak, n_top, n_bot):
            return {'r': np.linspace(0, 1500, 10),
                    'z': np.linspace(10, z_peak, 10),
                    'alpha': 0.0, 'n_top_bounces': n_top,
                    'n_bot_bounces': n_bot}
        bounces = {'direct': (0, 0), 'surface': (1, 0),
                   'bottom': (0, 1), 'both': (2, 3)}
        return Rays(rays=[ray(40.0 + 10.0 * i, *bounces[k])
                          for i, k in enumerate(classes)],
                    receiver_depths=np.array([50.0]),
                    receiver_ranges=np.array([1500.0]),
                    model='Bellhop')

    def test_ray_class_colours_and_legend_counts(self):
        # Multipath class mapping: direct = red, surface-reflected = green,
        # bottom-reflected = blue, both = black; the legend counts each class.
        from uacpy.visualization.plots._common import ZORDER_RAYS
        rays = self._classed_rays(['direct', 'surface', 'bottom', 'both'])
        fig, ax = rays.plot()
        ray_lines = [ln for ln in ax.lines if ln.get_zorder() == ZORDER_RAYS]
        assert [ln.get_color() for ln in ray_lines] == [
            '#e53935', '#43a047', '#1e88e5', '#000000']
        assert [t.get_text() for t in ax.get_legend().get_texts()] == [
            'direct (1)', 'surface (1)', 'bottom (1)', 'both (1)']
        plt.close(fig)

    def test_rays_draw_over_the_seafloor_fill_and_line(self, env):
        # A ray 0.5 m above a 100 m seabed lies inside the seafloor line's
        # own drawn width, so it shows only if it is painted after that line
        # and the sediment fill; it must still sit under the receiver marker.
        rays = Rays(
            rays=[{'r': np.linspace(0, 1500, 10),
                   'z': np.full(10, 99.5),
                   'alpha': 0.0, 'n_top_bounces': 0, 'n_bot_bounces': 0}],
            receiver_depths=np.array([99.5]),
            receiver_ranges=np.array([1500.0]),
            model='Bellhop',
        )
        fig, ax = rays.plot(env=env)
        ray = next(ln for ln in ax.lines
                   if len(ln.get_ydata()) == 10
                   and np.all(np.asarray(ln.get_ydata()) == 99.5))
        seafloor_line = next(ln for ln in ax.lines
                             if np.array_equal(ln.get_ydata(), [100.0, 100.0]))
        seafloor_fill = next(c for c in ax.collections
                             if isinstance(c, mcoll.PolyCollection))
        receiver = next(ln for ln in ax.lines if len(ln.get_ydata()) == 1)
        assert ray.get_zorder() > seafloor_fill.get_zorder()
        assert ray.get_zorder() > seafloor_line.get_zorder()
        assert ray.get_zorder() < receiver.get_zorder()
        plt.close(fig)

    def test_the_seafloor_stroke_lies_in_the_sediment_not_the_water(self, env):
        # The boundary line is 2 pt wide. Centred on the seabed, 1 pt of it
        # would lie in the water and cover a ray 0.5 m above a 100 m bottom
        # even when the ray is drawn on top. The stroke's upper edge must be
        # the boundary itself, so the ray's stroke clears it entirely.
        rays = Rays(
            rays=[{'r': np.linspace(0, 1500, 10),
                   'z': np.full(10, 99.5),
                   'alpha': 0.0, 'n_top_bounces': 0, 'n_bot_bounces': 0}],
            receiver_depths=np.array([99.5]),
            receiver_ranges=np.array([1500.0]),
            model='Bellhop',
        )
        fig, ax = rays.plot(env=env)
        fig.canvas.draw()
        px_per_pt = fig.dpi / 72.0
        ray = next(ln for ln in ax.lines
                   if len(ln.get_ydata()) == 10
                   and np.all(np.asarray(ln.get_ydata()) == 99.5))
        floor = next(ln for ln in ax.lines
                     if np.array_equal(ln.get_ydata(), [100.0, 100.0]))
        boundary_px = ax.transData.transform((0.0, 100.0))[1]
        # axhline: x in axes fraction, y in data — evaluate the line's own
        # transform at its own y to find where the stroke centre lands.
        floor_centre_px = floor.get_transform().transform((0.5, 100.0))[1]
        floor_top_px = floor_centre_px + floor.get_linewidth() / 2 * px_per_pt
        assert floor_top_px == pytest.approx(boundary_px, abs=1e-6)
        ray_px = ray.get_transform().transform((0.0, 99.5))[1]
        ray_bottom_px = ray_px - ray.get_linewidth() / 2 * px_per_pt
        # Display y grows upward: the ray's lower edge sits on or above the
        # seafloor stroke's upper edge.
        assert ray_bottom_px >= floor_top_px
        plt.close(fig)

        rays = self._classed_rays(['direct', 'surface'])
        fig, ax = rays.plot()
        assert [t.get_text() for t in ax.get_legend().get_texts()] == [
            'direct (1)', 'surface (1)']
        plt.close(fig)

    def test_axis_span_without_env(self):
        # Deepest ray point 80 m, receiver at 2 km. With no env= the depth
        # axis spans the fan itself — 8 % headroom below, 4 % above z = 0 —
        # and the x axis spans the receivers with a 3 % right margin.
        rays = Rays(
            rays=[{'r': np.linspace(0, 1500, 10),
                   'z': np.linspace(10, 80, 10),
                   'alpha': 0.0, 'n_top_bounces': 0, 'n_bot_bounces': 0}],
            receiver_depths=np.array([50.0]),
            receiver_ranges=np.array([2000.0]),
            model='Bellhop',
        )
        fig, ax = rays.plot()
        assert ax.get_ylim() == pytest.approx((80.0 * 1.08, -80.0 * 0.04))
        assert ax.get_xlim() == pytest.approx((0.0, 2.0 * 1.03))
        plt.close(fig)

    @staticmethod
    def _shallow_fan_deep_receiver():
        # Ray fan bottoms out at 80 m; the receiver hangs at 150 m.
        return Rays(
            rays=[{'r': np.linspace(0, 1500, 10),
                   'z': np.linspace(10, 80, 10),
                   'alpha': 0.0, 'n_top_bounces': 0, 'n_bot_bounces': 0}],
            receiver_depths=np.array([150.0]),
            receiver_ranges=np.array([2000.0]),
            model='Bellhop',
        )

    def test_ylim_reaches_a_receiver_below_the_ray_fan(self):
        # With show_receivers on and no env=, drawn receiver markers below
        # the deepest ray must sit inside the depth axis — the same
        # guarantee the x-limit gives an at-max-range receiver.
        fig, ax = self._shallow_fan_deep_receiver().plot()
        assert ax.get_ylim() == pytest.approx((150.0 * 1.08, -150.0 * 0.04))
        plt.close(fig)

    def test_ylim_ignores_receivers_when_not_drawn(self):
        # show_receivers=False draws no markers, so the depth axis spans
        # only the fan itself.
        fig, ax = self._shallow_fan_deep_receiver().plot(show_receivers=False)
        assert ax.get_ylim() == pytest.approx((80.0 * 1.08, -80.0 * 0.04))
        plt.close(fig)

    def test_ylim_with_env_spans_the_water_column(self, env):
        # env= keeps sizing the axis from env.depth when the receivers sit
        # inside the water column.
        rays = Rays(
            rays=[{'r': np.linspace(0, 1500, 10),
                   'z': np.linspace(10, 80, 10),
                   'alpha': 0.0, 'n_top_bounces': 0, 'n_bot_bounces': 0}],
            receiver_depths=np.array([50.0]),
            receiver_ranges=np.array([2000.0]),
            model='Bellhop',
        )
        fig, ax = rays.plot(env=env)
        assert ax.get_ylim() == pytest.approx((100.0 * 1.08, -100.0 * 0.04))
        plt.close(fig)

    def test_ylim_reaches_a_receiver_below_the_seafloor(self, env):
        # A receiver deeper than env.depth is still a drawn marker, so the
        # axis reaches it rather than sizing on the water column alone.
        fig, ax = self._shallow_fan_deep_receiver().plot(env=env)
        assert ax.get_ylim() == pytest.approx((150.0 * 1.08, -150.0 * 0.04))
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


class TestPlotCovariance:
    def test_dispatch_draws_csdm_image(self):
        rng = np.random.default_rng(0)
        C = rng.standard_normal((1, 4, 4)) + 1j * rng.standard_normal((1, 4, 4))
        cov = Covariance(covariance=C, model='OASN', frequencies=200.0)
        fig, ax = cov.plot()
        # One imshow image of |C|, titled with the slice frequency.
        assert len(ax.images) == 1
        assert ax.get_title() == 'Covariance at 200.0 Hz'
        plt.close(fig)


class TestPlotReplicas:
    def test_dispatch_draws_replica_field(self):
        rng = np.random.default_rng(1)
        R = (rng.standard_normal((1, 3, 4, 1, 2))
             + 1j * rng.standard_normal((1, 3, 4, 1, 2)))
        rep = Replicas(replicas=R, replica_z=np.linspace(10.0, 50.0, 3),
                       replica_x=np.linspace(100.0, 400.0, 4),
                       replica_y=[0.0], model='OASN', frequencies=200.0)
        fig, ax = rep.plot()
        # One (z, x) pcolormesh of |R| with the depth axis pointing down.
        meshes = [c for c in ax.collections if hasattr(c, 'get_coordinates')]
        assert len(meshes) == 1
        assert ax.yaxis_inverted()
        plt.close(fig)


class TestPlotEnvironment:
    def test_flat_env(self, env):
        fig, _ = env.plot()
        plt.close(fig)


class TestDataAttributionFootnote:
    """A figure the plotter owns is stamped with the environment's data
    attribution as a ``Data:`` footnote; ``data_source=False`` and ``ax=``
    (composition) both leave the figure unstamped."""

    @pytest.fixture
    def fetched_env(self, env):
        # A fetched env carries data_sources; stamp them onto this
        # hand-built one directly.
        env.data_sources = ('GEBCO Compilation Group, GEBCO Grid',)
        return env

    @staticmethod
    def _data_texts(fig):
        return [t for t in fig.texts if t.get_text().startswith('Data:')]

    def test_footnote_drawn_from_env_data_sources(self, fetched_env):
        fig, _ = fetched_env.plot()
        notes = self._data_texts(fig)
        assert len(notes) == 1
        assert 'GEBCO' in notes[0].get_text()
        plt.close(fig)

    def test_data_source_false_suppresses(self, fetched_env):
        fig, _ = fetched_env.plot(data_source=False)
        assert not self._data_texts(fig)
        plt.close(fig)

    def test_explicit_ax_suppresses(self, fetched_env):
        fig, ax = plt.subplots()
        fetched_env.plot(ax=ax)
        assert not self._data_texts(fig)
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

    def test_explicit_title_wins(self, env):
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


class TestNoiseSonarPlotterSignatures:
    """Documented signatures of the noise / sonar free plotters, each with a
    minimal-call smoke."""

    @pytest.mark.parametrize('name, params, args, kwargs', [
        ('plot_roc', ('deflection', 'ax', 'pfa', 'pd'),
         (), {'deflection': 2.0, 'n_points': 16}),
        ('plot_source_level', ('frequency', 'level_db', 'ax', 'label'),
         (np.array([63.0, 125.0, 250.0]),
          np.array([150.0, 148.0, 145.0])), {}),
        ('plot_weighting', ('group', 'ax', 'frequency'),
         ('LF',), {}),
    ])
    def test_signature_and_smoke(self, name, params, args, kwargs):
        fn = getattr(plots, name)
        sig = inspect.signature(fn)
        assert all(p in sig.parameters for p in params)
        fig, ax = fn(*args, **kwargs)
        assert ax.has_data()
        plt.close(fig)


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

    @pytest.mark.parametrize("kwargs, labelled", [
        ({}, True),                                      # both layers on
        ({'graticule_minor': None}, True),               # fine layer off
        ({'graticule': None}, False),                    # labelled layer off
        ({'graticule': None, 'graticule_minor': None}, False),
    ])
    def test_either_graticule_layer_can_be_switched_off(self, monkeypatch,
                                                        kwargs, labelled):
        """``graticule`` is documented as disable-able but was not Optional, so
        ``graticule=None`` reached ``np.ceil(rng[0] / step)`` and raised a bare
        ``TypeError`` — on the ``basemap=True`` default only, which is why the
        tests passing ``graticule=None`` on the plain branch never saw it."""
        monkeypatch.setattr('uacpy.visualization.basemap.land_polygons',
                            lambda *a, **k: None)
        lats, lons, depth = self._grid()
        fig, ax = plots.plot_bathymetry_map(lats, lons, depth, **kwargs)
        fig.canvas.draw()
        ticks = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        assert bool(ticks) is labelled, ticks
        plt.close(fig)

    def test_coastline_unreachable_draws(self, monkeypatch):
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

    def test_relief_orientation_invariant_to_lon_order(self):
        # The lon counterpart of the lat test above: imshow's extent is
        # anchored to the minimum longitude, so a descending lon axis has to be
        # flipped to west→east or the relief renders mirrored against the flat
        # pcolormesh path and every overlay.
        lats = np.linspace(36, 44, 8)
        lons_east = np.linspace(0, 10, 10)               # ascending W→E
        depth_east = np.random.default_rng(0).uniform(500, 3000, (8, 10))
        depth_east[0, -1] = np.nan                       # land at the EAST edge

        def relief_rgba(lons, depth):
            fig, ax = plots.plot_bathymetry_map(
                lats, lons, depth, basemap=False, relief=True, graticule=None)
            rgba = np.asarray(ax.get_images()[0].get_array())
            plt.close(fig)
            return rgba

        east = relief_rgba(lons_east, depth_east)
        west = relief_rgba(lons_east[::-1], depth_east[:, ::-1])
        assert np.array_equal(east, west, equal_nan=True)
        assert east[0, -1, 3] == 0.0                     # the land cell, transparent


@pytest.mark.parametrize("step, wraps", [
    (-180.0, False),          # exactly half a turn back: a real westward span
    (-180.0 - 1e-9, True),    # the smallest step past it, which is the wrap
    (180.0, False),           # exactly half a turn on: a real eastward span
    (180.0 + 1e-9, True),     # a range crossing the antimeridian westward
    (360.0, False),           # the two ends of a full-globe axis, one meridian
])
def test_only_a_step_past_half_a_turn_reads_as_an_antimeridian_wrap(step, wraps):
    """The unwrap has to fire on the [-180, 180) fold and on nothing else: a
    step of half a turn or less is a real span, and a step of a whole turn is
    the two ends of a full-globe axis sampling the same meridian twice."""
    lons = np.array([0.0, step])
    out, wrapped = _unwrap_lon_axis(lons)
    assert wrapped is wraps
    turn = -np.sign(step) * 360.0 if wraps else 0.0
    assert out == pytest.approx(np.array([0.0, step + turn]))


@pytest.mark.parametrize("lons", [
    np.linspace(0.0, 10.0, 6),        # plainly ascending
    np.linspace(10.0, 0.0, 6),        # plainly descending: the relief flip's case
    np.linspace(-180.0, 180.0, 9),    # the whole globe, no fold
])
def test_a_longitude_axis_without_a_fold_is_returned_untouched(lons):
    out, wrapped = _unwrap_lon_axis(lons)
    assert wrapped is False
    assert np.array_equal(out, lons)


@pytest.mark.parametrize("lon, label", [
    (182.0, -178.0),      # a turn past the antimeridian folds to the west
    (180.0, 180.0),       # the antimeridian itself keeps the sign it was given
    (-180.0, -180.0),
    (-3.5, -3.5),         # an ordinary tick is its own label
    (-182.0, 178.0),
])
def test_a_tick_past_the_antimeridian_takes_its_folded_label(lon, label):
    assert _lon_label_value(lon) == pytest.approx(label)


class TestAntimeridianMap:
    """plot_bathymetry_map on a grid crossing the antimeridian.

    :func:`uacpy.data.fetch_bathy_grid` documents, and
    :func:`uacpy.data._geo.lon_linspace` implements, an eastward range whose end
    lies west of its start. Its longitudes come back folded into [-180, 180), so
    the axis ascends with one fold in it — which read literally is a full-globe
    span running the wrong way.
    """

    _N = 12
    _CELL = 4.0 / (_N - 1)

    @classmethod
    def _crossing(cls):
        """A crossing grid whose depth ramps monotonically west → east."""
        lats = np.linspace(-20.0, -16.0, cls._N)
        lons = lon_linspace(178.0, -178.0, cls._N)
        depth = np.tile(np.linspace(100.0, 5000.0, cls._N), (cls._N, 1))
        return lats, lons, depth

    @staticmethod
    def _relief(lats, lons, depth):
        """``(rgba, extent, xlim)`` from the shaded-relief path."""
        fig, ax = plots.plot_bathymetry_map(lats, lons, depth, basemap=False,
                                            relief=True, graticule=None)
        im = ax.get_images()[0]
        out = (np.asarray(im.get_array()), tuple(im.get_extent()),
               tuple(ax.get_xlim()))
        plt.close(fig)
        return out

    def test_a_crossing_grid_spans_only_its_own_longitudes(self):
        """The fold makes ``lons.min()``/``lons.max()`` a whole-globe pair, and
        the extent built from them stretches a 4° strip across the world."""
        rgba, extent, _ = self._relief(*self._crossing())
        assert extent[:2] == pytest.approx((178.0 - self._CELL / 2,
                                            182.0 + self._CELL / 2))
        assert extent[1] - extent[0] == pytest.approx(4.0 + self._CELL)

    def test_a_crossing_grid_keeps_its_west_to_east_order(self):
        """The fold also satisfies the descending-axis test that mirrors the
        relief, so the strip must reach that test already unwrapped to draw in
        the same column order as an equivalent monotone axis."""
        lats, lons, depth = self._crossing()
        crossing, extent, _ = self._relief(lats, lons, depth)
        control, control_extent, _ = self._relief(
            lats, np.linspace(178.0, 182.0, self._N), depth)
        assert np.array_equal(crossing, control)
        assert extent == pytest.approx(control_extent)

    def test_a_crossing_map_draws_the_land_west_of_the_dateline(self, monkeypatch):
        """Natural Earth cuts its rings at ±180, so the half of an unwrapped
        window beyond 180 is covered only by a copy of the land a turn east."""
        ring = np.array([[-179.5, -19.0], [-178.5, -19.0], [-178.5, -17.0],
                         [-179.5, -17.0], [-179.5, -19.0]])
        monkeypatch.setattr('uacpy.visualization.basemap.land_polygons',
                            lambda *a, **k: [ring])
        fig, ax = plots.plot_bathymetry_map(*self._crossing())
        x0, x1 = ax.get_xlim()
        assert (x0, x1) == pytest.approx((178.0, 182.0))
        drawn = [p.get_xy()[:, 0] for p in ax.patches]
        assert any(x0 <= xs.min() and xs.max() <= x1 for xs in drawn), \
            [(xs.min(), xs.max()) for xs in drawn]
        plt.close(fig)

    def test_a_crossing_map_names_the_hemisphere_on_the_plain_branch_too(self):
        """``basemap=False`` draws no graticule, so its ticks are matplotlib's
        own numbers. Over an unwrapped window those run past 180, where a plain
        number reads 182 for a map that is at 178°W."""
        fig, ax = plots.plot_bathymetry_map(*self._crossing(), basemap=False)
        fig.canvas.draw()
        labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        assert labels, "no x tick labels rendered"
        assert all(label.endswith(('°E', '°W')) for label in labels), labels
        assert any(label.endswith('°W') for label in labels), labels
        assert ax.get_xlabel() == 'Longitude'
        plt.close(fig)

    def test_a_non_crossing_map_keeps_plain_degrees_east_on_the_plain_branch(self):
        """The counterpart: a window that does not cross keeps the numbering
        and the axis label it has always had."""
        lons = np.linspace(0.0, 10.0, self._N)
        lats = np.linspace(36.0, 44.0, self._N)
        depth = np.tile(np.linspace(100.0, 5000.0, self._N), (self._N, 1))
        fig, ax = plots.plot_bathymetry_map(lats, lons, depth, basemap=False)
        fig.canvas.draw()
        labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        assert not any(label.endswith(('°E', '°W')) for label in labels), labels
        assert ax.get_xlabel() == 'Longitude (°E)'
        plt.close(fig)

    def test_a_crossing_map_labels_its_ticks_by_hemisphere(self, monkeypatch):
        monkeypatch.setattr('uacpy.visualization.basemap.land_polygons',
                            lambda *a, **k: None)
        fig, ax = plots.plot_bathymetry_map(*self._crossing(), graticule=1.0,
                                            graticule_minor=None)
        assert [t.get_text() for t in ax.get_xticklabels()] == [
            '178°E', '179°E', '180°E', '179°W', '178°W']
        plt.close(fig)

    def test_overlays_west_of_the_dateline_land_beside_the_grid(self, monkeypatch):
        """A transect end and a source given as -179 belong at 181, next to the
        strip they annotate, not a whole globe away from it."""
        monkeypatch.setattr('uacpy.visualization.basemap.land_polygons',
                            lambda *a, **k: None)
        fig, ax = plots.plot_bathymetry_map(
            *self._crossing(), transect=((-19.0, 179.0), (-17.0, -179.0)),
            source=(-18.0, -178.5))
        x0, x1 = ax.get_xlim()
        transect = [ln for ln in ax.lines if ln.get_label() == 'transect'][0]
        assert transect.get_xdata() == pytest.approx([179.0, 181.0])
        marker = [ln for ln in ax.lines if ln is not transect][0]
        assert float(np.ravel(marker.get_xdata())[0]) == pytest.approx(181.5)
        assert x0 <= 181.5 <= x1
        plt.close(fig)

    def test_a_non_crossing_map_keeps_every_coordinate_as_given(self, monkeypatch):
        """The unwrap must be invisible to an ordinary window: same extent, same
        overlay coordinates, same tick labels."""
        monkeypatch.setattr('uacpy.visualization.basemap.land_polygons',
                            lambda *a, **k: None)
        lats = np.linspace(44.5, 49.5, self._N)
        lons = np.linspace(-11.0, -3.5, self._N)
        depth = np.tile(np.linspace(100.0, 5000.0, self._N), (self._N, 1))
        fig, ax = plots.plot_bathymetry_map(
            lats, lons, depth, transect=((48.2, -8.0), (45.6, -6.2)),
            source=(48.2, -8.0), graticule=1.0, graticule_minor=None)
        assert ax.get_xlim() == pytest.approx((-11.0, -3.5))
        transect = [ln for ln in ax.lines if ln.get_label() == 'transect'][0]
        assert transect.get_xdata() == pytest.approx([-8.0, -6.2])
        assert [t.get_text() for t in ax.get_xticklabels()][0] == '11°W'
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


def test_a_difference_field_is_drawn_as_a_signed_residual_not_a_loss():
    """A dB difference is neither a level nor a loss, and must not be either.

    Subtracting one TL field from another gives a signed residual whose ZERO is
    the meaningful value. Built untagged it inherits ``kind='pressure'``, and
    then three things are wrong at once: the colourbar says "TL (dB)", the
    fixed 20-120 dB transmission-loss window swallows a residual that lives
    near zero, and the loss predicate runs a 1-D cut's value axis downward.
    The registered ``difference`` quantity gives it the diverging map and the
    symmetric window instead, the same treatment signal excess gets.
    """
    from uacpy.core.results import quantities as _q
    from uacpy.core.results import Field
    from uacpy.visualization.plots._common import _TL_LIMITS, _is_loss_view
    from uacpy.visualization.plots.fields import _value_style

    depths = np.linspace(5, 95, 12)
    ranges = np.linspace(100, 5000, 40)
    base = np.broadcast_to(
        50.0 + 10.0 * np.log10(ranges)[None, :], (12, 40)).copy()
    resid = Field(data=base - (base + 2.0),
                  coords={'depth': depths, 'range': ranges})

    # Untagged, it is claimed by the transmission-loss treatment.
    assert resid.kind == 'pressure'
    assert _is_loss_view(resid, 'db') is True

    resid.metadata['kind'] = 'difference'
    assert resid.kind == 'difference'
    assert _is_loss_view(resid, 'db') is False, (
        "a signed residual must not be drawn with its value axis inverted")
    cmap, lo, hi = _value_style(resid, 'db')
    assert (lo, hi) != _TL_LIMITS, "the fixed TL window swallows a residual"
    assert lo == -hi, f"a residual needs a window symmetric about 0, got {lo, hi}"
    assert 'TL' not in _q.label('difference', 'dB')


def test_the_ppsd_level_axis_follows_the_results_own_scaling():
    """The "/Hz" is a claim about the quantity, not decoration.

    ``ppsd`` computes a density or a spectrum; the first is per hertz and the
    second is per band. The plotter used to caption both "/Hz", so a spectrum
    was published under a density's unit. It now reads the scaling the result
    carries.
    """
    from uacpy.acoustic_signal.analysis import ppsd as _ppsd
    from uacpy.visualization.plots.signal import plot_ppsd

    rng = np.random.default_rng(0)
    x = rng.standard_normal(48000)

    fig, ax = plot_ppsd(_ppsd(x, 48000.0, scaling='density', nperseg=1024))
    assert ax.get_ylabel().endswith('Pa²/Hz)'), ax.get_ylabel()
    plt.close(fig)

    fig, ax = plot_ppsd(_ppsd(x, 48000.0, scaling='spectrum', nperseg=1024))
    label = ax.get_ylabel()
    assert label.endswith('Pa²)'), label
    assert '/Hz' not in label, label
    plt.close(fig)


def test_the_compare_models_colourbar_follows_the_credit_margin(tl_field):
    """The figure-level colourbar is placed from the panels' FINAL bottom.

    ``compare_models`` reserves a bottom margin by formula, draws the panels,
    then adds the colourbar as a free axes at that coordinate. The model credit
    is drawn afterwards and reserves its own margin with a second
    ``subplots_adjust``, which moves the panels but not an axes already placed
    at an absolute figure coordinate — so the bar used to hang below the panels,
    level with the x tick labels, whenever a credit was drawn.

    The reserve depends on font metrics, so it is forced here rather than
    conjured with real text: any credit that raises the bottom must carry the
    colourbar with it.
    """
    from uacpy.visualization.plots import fields as _fields

    real = _fields._draw_multi_model_credit

    def reserving(fig, flds):
        real(fig, flds)
        fig.subplots_adjust(bottom=fig.subplotpars.bottom + 0.05)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(_fields, '_draw_multi_model_credit', reserving)
    try:
        fig, axes = _fields.compare_models([tl_field, tl_field, tl_field])
    finally:
        monkey.undo()
    panel_axes = list(np.ravel(axes))
    panel_bottom = min(a.get_position().y0
                       for a in panel_axes if a.get_visible())
    bars = [a.get_position() for a in fig.axes if a not in panel_axes]
    assert bars, "compare_models drew no figure-level colourbar"
    assert bars[0].y0 == pytest.approx(panel_bottom, abs=1e-6), (
        f"colourbar bottom {bars[0].y0:.4f} does not sit at the panels' final "
        f"bottom {panel_bottom:.4f}; it was placed before the credit moved them"
    )
    plt.close(fig)


def test_compare_models_label_length_validation(tl_field):
    with pytest.raises(ConfigurationError, match="match"):
        uacpy.plot.compare_models([tl_field, tl_field, tl_field],
                                  labels=['only-one'])


def test_plot_environment_rejects_non_environment():
    from uacpy.visualization.plots.environment import _plot_environment
    with pytest.raises(ConfigurationError, match="Environment"):
        _plot_environment("not an environment")


class TestPlotFieldSingletonAxes:
    """A length-1 axis has no neighbours for ``shading='nearest'`` to build cell
    edges from, so every heatmap quad collapses to zero extent and the panel
    renders empty — indistinguishable from an all-NaN field. Models pass the
    receiver axes through verbatim and ``Receiver`` keeps a scalar as a length-1
    array, so a single-receiver-depth run reaches this."""

    @staticmethod
    def _field(n_depth, n_range):
        d = np.linspace(5.0, 95.0, n_depth)
        r = np.linspace(100.0, 5000.0, n_range)
        data = 50.0 + 10.0 * np.log10(np.maximum(r, 1.0))[None, :]
        return Field(data=np.broadcast_to(data, (n_depth, n_range)).copy(),
                     coords={'depth': d, 'range': r},
                     model='Synth', frequencies=100.0)

    @staticmethod
    def _drawn_pixels(fig):
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba()).astype(int)
        return int((np.ptp(rgba[..., :3], axis=-1) > 12).sum())

    def test_single_depth_plots_a_range_cut_not_an_empty_mesh(self):
        fig, ax = plots.plot_field(self._field(1, 7))
        assert len(ax.lines) == 1
        assert not [c for c in ax.collections if hasattr(c, 'get_coordinates')]
        assert ax.get_xlabel() == 'Range (km)'
        assert ax.get_ylabel() == 'TL (dB)'
        assert self._drawn_pixels(fig) > 0
        plt.close(fig)

    def test_single_range_plots_a_depth_profile(self):
        fig, ax = plots.plot_field(self._field(5, 1))
        assert len(ax.lines) == 1
        assert ax.get_ylabel() == 'Depth (m)'
        assert ax.yaxis_inverted()
        assert self._drawn_pixels(fig) > 0
        plt.close(fig)

    def test_a_single_cell_is_visible(self):
        """One sample gives a line nothing to join, so it needs a marker."""
        fig, ax = plots.plot_field(self._field(1, 1))
        assert self._drawn_pixels(fig) > 0
        plt.close(fig)

    def test_the_squeezed_axis_is_recorded_in_pinned(self):
        """Dropping an axis must go through ``isel`` so the value is not lost."""
        fig, ax = plots.plot_field(self._field(1, 7))
        plt.close(fig)
        squeezed = self._field(1, 7).isel(depth=0)
        assert 'depth' in squeezed.pinned

    def test_a_full_grid_uses_the_heatmap(self):
        fig, ax = plots.plot_field(self._field(5, 7))
        mesh = [c for c in ax.collections if hasattr(c, 'get_coordinates')]
        assert len(mesh) == 1
        assert not ax.lines
        co = mesh[0].get_coordinates()
        assert float(co[..., 1].max() - co[..., 1].min()) > 0.0
        plt.close(fig)


class TestImshowPanelsAreEdgeAligned:
    """``imshow`` stretches the array onto the OUTER edges of ``extent``, so
    passing the first and last cell *centres* contracts the field by
    ``(N-1)/N`` about its middle — zero error at the centre, half a rendered
    pixel at the edges. That is where every overlay drawn in true coordinates
    (graticule, contours, coastline, markers, the analytic sound cone)
    disagrees with the data it annotates.

    Each case places its feature **off-centre** — a contraction has no error at
    the centre by construction — and uses **non-square** grids so a
    transposition raises instead of silently passing.
    """

    @staticmethod
    def _centres(ax, nx, ny, y_down=False):
        """Coordinates imshow actually gives each pixel centre."""
        e = ax.images[-1].get_extent()
        px = e[0] + (np.arange(nx) + 0.5) * (e[1] - e[0]) / nx
        # origin='upper' puts row 0 at the top of the extent.
        y0, y1 = (e[3], e[2]) if y_down else (e[2], e[3])
        py = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
        return px, py

    def test_relief_map_registers_with_its_own_coordinates(self):
        from uacpy.visualization.plots.maps import _draw_depth
        lons = np.linspace(5.0, 9.0, 41)
        lats = np.linspace(40.0, 44.0, 33)          # non-square on purpose
        LO, LA = np.meshgrid(lons, lats)
        i, j = 2, 3                                  # off-centre, near a corner
        depth = 3000.0 - 2000.0 * np.exp(
            -(((LO - lons[i]) / 0.15) ** 2 + ((LA - lats[j]) / 0.15) ** 2))
        fig, ax = plt.subplots()
        _draw_depth(ax, lons, lats, depth, 'viridis', True, 15.0, 1)
        px, py = self._centres(ax, lons.size, lats.size)
        assert px[i] == pytest.approx(lons[i], abs=1e-9)
        assert py[j] == pytest.approx(lats[j], abs=1e-9)
        plt.close(fig)

    def test_relief_and_flat_branches_agree(self):
        """The two branches of ``_draw_depth`` must place data identically."""
        from uacpy.visualization.plots.maps import _draw_depth
        lons = np.linspace(5.0, 9.0, 41)
        lats = np.linspace(40.0, 44.0, 33)
        depth = np.full((lats.size, lons.size), 2000.0)
        depth[3, 2] = 1000.0
        fig, ax = plt.subplots()
        _draw_depth(ax, lons, lats, depth, 'viridis', False, 15.0, 1)
        co = ax.collections[-1].get_coordinates()
        flat_x = 0.5 * (co[0, :-1, 0] + co[0, 1:, 0])
        plt.close(fig)
        fig, ax = plt.subplots()
        _draw_depth(ax, lons, lats, depth, 'viridis', True, 15.0, 1)
        relief_x, _ = self._centres(ax, lons.size, lats.size)
        np.testing.assert_allclose(relief_x, flat_x, atol=1e-9)
        plt.close(fig)

    def test_fk_panel_registers_against_the_sound_cone(self):
        from uacpy.visualization.plots.signal import plot_fk
        freqs = np.linspace(0.0, 320.0, 33)
        ks = np.linspace(-1.2, 1.2, 25)              # non-square
        power = np.zeros((freqs.size, ks.size))
        power[3, 2] = 1.0
        fig, ax = plot_fk(freqs, ks, power, sound_speed=1500.0)
        px, py = self._centres(ax, ks.size, freqs.size)
        assert px[2] == pytest.approx(ks[2], abs=1e-9)
        assert py[3] == pytest.approx(freqs[3], abs=1e-9)
        plt.close(fig)

    def test_ambiguity_panel_is_edge_aligned(self):
        from uacpy.visualization.plots.signal import plot_ambiguity
        delays = np.linspace(-0.004, 0.004, 17)
        doppler = np.linspace(-60.0, 60.0, 13)
        chi = np.zeros((doppler.size, delays.size))
        chi[2, 3] = 1.0
        fig, ax = plot_ambiguity(delays, doppler, chi)
        px, py = self._centres(ax, delays.size, doppler.size)
        assert px[3] == pytest.approx(delays[3] * 1e3, abs=1e-9)
        assert py[2] == pytest.approx(doppler[2], abs=1e-9)
        plt.close(fig)

    def test_radon_and_taup_panels_are_edge_aligned(self):
        """Both draw ``origin='upper'`` with intercept time increasing downward
        and an abscissa rescaled to s/km, so both conversions must survive."""
        from uacpy.visualization.plots.signal import plot_radon, plot_taup
        taus = np.linspace(0.0, 0.6, 21)
        moveout = np.linspace(-0.002, 0.002, 15)
        R = np.zeros((moveout.size, taus.size))
        R[2, 3] = 1.0
        fig, ax = plot_radon(moveout, taus, R)
        px, py = self._centres(ax, moveout.size, taus.size, y_down=True)
        assert px[2] == pytest.approx(moveout[2] * 1e3, abs=1e-9)
        assert py[3] == pytest.approx(taus[3], abs=1e-9)
        plt.close(fig)

        slow = np.linspace(0.0, 0.0012, 19)
        T = np.zeros((slow.size, taus.size))
        T[4, 5] = 1.0
        fig, ax = plot_taup(slow, taus, T)
        px, py = self._centres(ax, slow.size, taus.size, y_down=True)
        assert px[4] == pytest.approx(slow[4] * 1000.0, abs=1e-9)
        assert py[5] == pytest.approx(taus[5], abs=1e-9)
        plt.close(fig)


class TestLayeredSeabedFollowsTheBathymetry:
    """A layered seabed's layers ride the seafloor. Anchoring them on a scalar
    (the deepest bathymetry point) detaches the stack from a sloping seabed and
    leaves the water colormap painted in the gap below the drawn seafloor line —
    while the range-dependent layered and half-space branches both track it."""

    @staticmethod
    def _env(bottom):
        import uacpy
        from uacpy.core.environment import SoundSpeedProfile
        from uacpy.core.bathymetry import Bathymetry
        return uacpy.Environment(
            bathymetry=Bathymetry(ranges=np.array([0.0, 1500.0, 3300.0]),
                                  depths=np.array([180.0, 210.0, 150.0])),
            ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (210.0, 1490.0)]),
            bottom=bottom)

    @staticmethod
    def _column(cs):
        from uacpy.core.environment import (SeabedColumn, SedimentLayer,
                                            BoundaryProperties)
        return SeabedColumn(
            layers=[SedimentLayer(thickness=20.0, sound_speed=cs, density=1.7,
                                  attenuation=0.5)],
            halfspace=BoundaryProperties(acoustic_type='half-space',
                                         sound_speed=2200.0, density=2.1,
                                         attenuation=0.2))

    @staticmethod
    def _drawn_at(ax, r_km, z_m):
        polys = [c for c in ax.collections
                 if hasattr(c, 'get_paths') and c.get_paths()]
        return any(p.contains_point((r_km, z_m))
                   for c in polys for p in c.get_paths())

    def test_a_range_independent_stack_tracks_a_sloping_seafloor(self):
        from uacpy.core.bottom import Bottom
        env = self._env(Bottom([self._column(1600.0)]))
        out = env.plot()
        fig = out[0] if isinstance(out, tuple) else out
        ax = fig.get_axes()[0]
        # The shallow ends are the discriminating ones: only the range whose
        # seafloor equals the maximum was drawn before.
        for r_km, seafloor in ((0.0, 180.0), (1.5, 210.0), (3.3, 150.0)):
            assert self._drawn_at(ax, r_km, seafloor + 2.0)
            assert self._drawn_at(ax, r_km, seafloor + 15.0)
        plt.close(fig)

    def test_a_range_dependent_stack_covers_the_whole_panel(self):
        """``Bottom.at()`` holds the last column beyond the final profile node,
        so the section beyond it must be drawn, not left bare."""
        from uacpy.core.bottom import Bottom
        env = self._env(Bottom([self._column(1600.0), self._column(1700.0)],
                               ranges=np.array([0.0, 2000.0])))
        assert env.bottom.at(range=3300.0).layers[0].sound_speed == 1700.0
        out = env.plot()
        fig = out[0] if isinstance(out, tuple) else out
        ax = fig.get_axes()[0]
        for r_km, seafloor in ((0.5, 190.0), (2.5, 183.3), (3.2, 155.6)):
            assert self._drawn_at(ax, r_km, seafloor + 5.0), f"bare at {r_km} km"
        plt.close(fig)


class TestAnExampleTitleQuotesTheDeckThatRan:
    """``example_04`` prints the 7-character RunType string in three panel
    titles. Position 6 is the dimensionality: blank is plain 2-D, and a
    literal '2' means Nx2D, which the C++ port warns about and rewrites. The
    titles were quoting a '2' the writer no longer emits, so a reader
    comparing the figure against the deck would find them disagreeing."""

    @staticmethod
    def _titles_in_example():
        import re
        from pathlib import Path
        src = (Path(uacpy.__file__).parent / 'examples'
               / 'example_04_bellhop_advanced.py').read_text(encoding='utf-8')
        return re.findall(r"RunType: '(.{7})'", src)

    @staticmethod
    def _written_run_type(**kwargs):
        """The RunType line the deck writer actually emits for ``kwargs``."""
        import re
        import tempfile
        from pathlib import Path
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        env = uacpy.Environment(bathymetry=100.0, ssp=1500.0, bottom=1650.0)
        src = uacpy.Source(depths=[25.0], frequencies=200.0)
        rcv = uacpy.Receiver(depths=np.linspace(1.0, 99.0, 20),
                             ranges=np.linspace(100.0, 5000.0, 50))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'probe.env'
            write_bellhop_env_file(str(path), env, src, rcv, **kwargs)
            lines = [ln.strip()[1:-1]
                     for ln in path.read_text(encoding='utf-8').splitlines()
                     if re.fullmatch(r"'[A-Za-z].{6}'", ln.strip())]
        assert lines, "no RunType line in the deck"
        return lines[-1]

    def test_all_three_titles_match_the_written_deck(self):
        titles = self._titles_in_example()
        assert len(titles) == 3, titles
        written = [
            self._written_run_type(run_type='C', beam_type='B',
                                   source_type='R', grid_type='R'),
            self._written_run_type(run_type='C', beam_type='B',
                                   source_type='X', grid_type='R'),
            self._written_run_type(run_type='R', beam_type='g',
                                   source_type='R', grid_type='R',
                                   beam_shift=True),
        ]
        assert titles == written, (titles, written)

    def test_no_title_claims_the_nx2d_dimensionality(self):
        """Position 6 read on its own, so the check does not depend on the
        other six characters staying put."""
        for title in self._titles_in_example():
            assert title[5] == ' ', (title, title[5])


class TestATLDifferenceIsNotLabelledAsALevel:
    """``_plot_tl_difference`` builds its residual as a bare ``Field``, which
    inherits ``kind='pressure'``, so the colourbar came back reading 'TL (dB)'
    over a signed difference — and the loss predicate reads the same tag, which
    would run a 1-D cut's value axis downward. Neither is true of a residual."""

    @staticmethod
    def _plotting_utils():
        """``uacpy/examples`` carries no ``__init__.py``, so the shared helper
        is loaded from its path rather than imported by package name."""
        import importlib.util
        from pathlib import Path
        path = (Path(uacpy.__file__).parent / 'examples' / 'plotting_utils.py')
        spec = importlib.util.spec_from_file_location(
            'uacpy_examples_plotting_utils_for_tests', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _pair():
        d = np.linspace(5.0, 95.0, 6)
        r = np.linspace(100.0, 3000.0, 9)
        mk = lambda amp: Field(data=np.full((6, 9), amp, dtype=complex),
                               coords={'depth': d, 'range': r},
                               model='Synth', frequencies=100.0)
        return mk(1e-3), mk(2e-3)

    def test_the_colourbar_names_the_residual_and_its_sign(self):
        a, b = self._pair()
        fig, ax = self._plotting_utils()._plot_tl_difference(a, b)
        label = fig.axes[-1].get_ylabel()
        assert label.startswith('ΔTL (dB)'), label
        assert 'quieter' in label, label
        plt.close(fig)


class TestOnlySomeViewsCarryAFixedColourWindow:
    """The docstring and the guide both said the dB view is "never an
    autoscale". It is one for two of the four registered kinds, and
    ``value='mag_db'`` takes the REVERSED TL map, not the TL map. This table
    is what the prose now describes, so the two cannot drift apart silently."""

    @staticmethod
    def _field(data, metadata=None):
        return Field(data=data,
                     coords={'depth': np.linspace(5.0, 95.0, 4),
                             'range': np.linspace(100.0, 3000.0, 5)},
                     model='Synth', frequencies=100.0, metadata=metadata)

    @pytest.mark.parametrize('kind, unit, value, expected', [
        ('pressure', None, 'db', ('jet_r', 20.0, 120.0)),
        ('signal_excess', 'dB', 'db', ('RdBu_r', -40.0, 40.0)),
        ('reverberation', 'dB', 'db', ('jet_r', None, None)),
        ('probability_of_detection', '1', 'real', ('RdYlGn', 0.0, 1.0)),
        ('pressure', None, 'mag_db', ('jet', None, None)),
    ])
    def test_the_colour_window_each_view_actually_takes(self, kind, unit,
                                                        value, expected):
        from uacpy.visualization.plots.fields import _value_style
        if kind == 'pressure':
            field = self._field(np.ones((4, 5), dtype=complex))
        elif kind == 'signal_excess':
            field = self._field(np.linspace(-40.0, 40.0, 20).reshape(4, 5),
                                {'kind': kind, 'unit': unit})
        elif kind == 'reverberation':
            field = self._field(np.linspace(40.0, 90.0, 20).reshape(4, 5),
                                {'kind': kind, 'unit': unit})
        else:
            field = self._field(np.linspace(0.0, 1.0, 20).reshape(4, 5),
                                {'kind': kind, 'unit': unit})
        assert _value_style(field, value) == expected


class TestTheSourceMarkerClearsTheAxisEdge:
    """Models exclude the singular near field, so a TL grid starts past r = 0
    while the source sits at it, and the plotters widen the x limit to bring
    the marker back on screen. Widening to EXACTLY the marker's x centres it on
    the spine, and markers keep default clipping (a later zoom must hide
    out-of-view receivers), so half the star was cut away."""

    @staticmethod
    def _panel(fig_width, source_range_m):
        import uacpy
        d = np.linspace(1.0, 60.0, 40)
        r = np.linspace(50.0, 5000.0, 120)
        tl = Field(data=np.tile((40.0 + 20 * np.log10(r))[None, :], (40, 1)),
                   coords={'depth': d, 'range': r}, model='Synth',
                   frequencies=200.0,
                   metadata={'kind': 'pressure', 'unit': 'dB'})
        env = uacpy.Environment(bathymetry=100.0, ssp=1500.0, bottom=1650.0)
        rcv = uacpy.Receiver(depths=d, ranges=r)
        fig, ax = plt.subplots(figsize=(fig_width, 2.4))
        tl.plot(env=env, receiver=rcv, ax=ax, show_colorbar=False)
        from uacpy.visualization.plots._common import _draw_geometry
        _draw_geometry(ax, uacpy.Source(depths=[30.0], frequencies=200.0),
                       source_range_m=source_range_m)
        return fig, ax

    @staticmethod
    def _red_pixel_count(fig):
        """Count the star's own pixels on the RENDERED figure. A limit that is
        merely left of the marker's centre still says nothing about whether
        the glyph was drawn whole, which is what the reader sees."""
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int)
        red = ((rgb[:, :, 0] > 170) & (rgb[:, :, 1] < 90)
               & (rgb[:, :, 2] < 90))
        return int(red.sum())

    @pytest.mark.parametrize('fig_width', [3.0, 4.5, 8.6])
    def test_a_source_at_range_zero_is_drawn_whole(self, fig_width):
        """Against a source placed INSIDE the data, which needs no widening at
        all and so is necessarily unclipped."""
        fig_in, _ = self._panel(fig_width, 2500.0)
        whole = self._red_pixel_count(fig_in)
        plt.close(fig_in)
        fig_edge, _ = self._panel(fig_width, 0.0)
        assert self._red_pixel_count(fig_edge) == whole
        plt.close(fig_edge)

    def test_the_pad_is_the_markers_own_size_not_a_share_of_the_span(self):
        """A fixed fraction of the data span is a different number of points
        on every figure size: 1 % cleared the star on a wide panel and still
        clipped it on a 3-inch one."""
        pads = []
        for fig_width in (3.0, 8.6):
            fig, ax = self._panel(fig_width, 0.0)
            lo, hi = sorted(ax.get_xlim())
            pads.append(-lo / (hi - lo))          # pad as a share of the span
            plt.close(fig)
        assert pads[0] > 2.0 * pads[1], pads


class TestAStackedViewLabelsOnlyAsManyTracesAsFit:
    """One tick per trace made the y axis a solid black smear the moment the
    stack was more than a couple of dozen deep — 60 ranges gave 60 overlapping
    labels, and documented receiver grids run to hundreds. The labels are
    strided; every trace is still drawn."""

    @staticmethod
    def _stack(n_other):
        r = np.linspace(100.0, 8000.0, n_other)
        t = np.linspace(0.0, 0.5, 40)
        rng = np.random.default_rng(0)
        f = Field(data=rng.standard_normal((n_other, 40)),
                  coords={'range': r, 'time': t}, model='Synth',
                  frequencies=200.0)
        fig, ax = plot_field(f, stacked=True)
        # The stacked view labels a range axis in km, as every other view of
        # a range axis does, so the expected labels are the km values.
        return fig, ax, r / 1000.0

    @pytest.mark.parametrize('n_other', [1, 5, 12, 23, 24, 60, 137, 250, 600])
    def test_the_tick_count_stays_readable_at_every_stack_depth(self, n_other):
        """23 is the worst case the ~12-label stride admits: below 24 traces
        the stride is 1, so the count is the trace count itself."""
        fig, ax, _ = self._stack(n_other)
        assert len(ax.get_yticks()) <= 23, len(ax.get_yticks())
        plt.close(fig)

    @pytest.mark.parametrize('n_other', [60, 250])
    def test_a_deep_stack_gets_about_a_dozen_labels(self, n_other):
        fig, ax, _ = self._stack(n_other)
        assert 8 <= len(ax.get_yticks()) <= 13, len(ax.get_yticks())
        plt.close(fig)

    def test_a_stack_of_nine_labels_all_nine(self):
        """The stride exists for crowded axes; it must not thin an axis that
        was legible already."""
        fig, ax, r = self._stack(9)
        assert len(ax.get_yticks()) == 9
        assert [t.get_text() for t in ax.get_yticklabels()] == [
            f"{v:.4g}" for v in r]
        plt.close(fig)

    def test_every_label_names_the_trace_it_sits_on(self):
        """Ticks and labels are taken on ONE stride. Striding the positions
        while labelling from the unstrided coordinate renumbers the axis, and
        a renumbered axis reads as perfectly correct."""
        fig, ax, r = self._stack(60)
        assert len(ax.get_lines()) == 60, "every trace is still drawn"
        # Each trace's own offset is the mean of its plotted samples, since
        # the traces are zero-mean noise riding on i * offset.
        trace_offsets = np.array([ln.get_ydata().mean()
                                  for ln in ax.get_lines()])
        for pos, tick in zip(ax.get_yticks(), ax.get_yticklabels()):
            i = int(np.argmin(np.abs(trace_offsets - pos)))
            assert tick.get_text() == f"{r[i]:.4g}", (pos, tick.get_text())
        plt.close(fig)


class TestOnlyABottomWithACpGetsABottomColourbar:
    """A vacuum / rigid / file half-space carries no compressional speed, so
    there is nothing for a 'Bottom cp' bar to show and none is drawn. The
    mappable for it was still built, normalized to the WATER speeds as a
    stand-in — a colour scale for a quantity the panel does not contain."""

    @staticmethod
    def _inset_labels(fig):
        return {c.get_ylabel()
                for ax in fig.axes for c in ax.child_axes}

    @pytest.mark.parametrize('acoustic_type', ['vacuum', 'rigid'])
    def test_a_bottom_without_a_cp_shows_the_water_bar_alone(self,
                                                             acoustic_type):
        from uacpy.core import BoundaryProperties
        from uacpy.visualization.plots.environment import _plot_environment
        env = uacpy.Environment(
            bathymetry=100.0,
            bottom=BoundaryProperties(acoustic_type=acoustic_type))
        fig, _ = _plot_environment(env)
        labels = self._inset_labels(fig)
        assert 'Water c (m/s)' in labels
        assert 'Bottom cp (m/s)' not in labels, labels
        plt.close(fig)

    def test_a_bottom_with_a_cp_gets_both_bars(self):
        from uacpy.visualization.plots.environment import _plot_environment
        fig, _ = _plot_environment(
            uacpy.Environment(bathymetry=100.0, bottom=1650.0))
        assert {'Water c (m/s)', 'Bottom cp (m/s)'} <= self._inset_labels(fig)
        plt.close(fig)


class TestTheEnvironmentPanelShowsTheWholeSeabed:
    """The depth limit is a margin past the deepest seafloor OR the floor of
    what the bottom branch actually painted, whichever is deeper. A 40 m
    sediment stack under 100 m of water needs 152 m of panel; the water-column
    margin alone gives 120 m and clips the second layer, the layer boundary
    and the half-space off the bottom edge."""

    @staticmethod
    def _thick_layered_env():
        import uacpy
        from uacpy.core.environment import (SeabedColumn, SedimentLayer,
                                            BoundaryProperties)
        col = SeabedColumn(
            layers=[SedimentLayer(thickness=10.0, sound_speed=1600.0,
                                  density=1.7, attenuation=0.8),
                    SedimentLayer(thickness=30.0, sound_speed=1750.0,
                                  density=2.0, attenuation=0.5)],
            halfspace=BoundaryProperties.from_preset('limestone'))
        return uacpy.Environment(bathymetry=100.0, bottom=col)

    def test_a_thick_sediment_stack_stays_on_panel(self):
        from uacpy.visualization.plots.environment import _plot_environment
        fig, ax = _plot_environment(self._thick_layered_env())
        deepest = max(ax.get_ylim())
        # Layers end at 140 m; the hatched half-space extends past that.
        assert deepest > 140.0, deepest
        plt.close(fig)

    def test_a_thin_bottom_keeps_the_water_column_margin(self):
        """The seafloor margin still sets the limit whenever it is the deeper
        of the two, so a thin bottom is not given a stretched panel."""
        import uacpy
        from uacpy.visualization.plots.environment import _plot_environment
        fig, ax = _plot_environment(uacpy.Environment(bathymetry=100.0))
        assert max(ax.get_ylim()) == pytest.approx(120.0)
        plt.close(fig)


class TestCompareModelsSharesItsColourScale:
    """``compare_models`` draws ONE figure-level colorbar, so every panel must
    map the same limits. Left to autoscale, each panel maps its own range and
    the single bar annotates the figure with only the last panel's — two fields
    differing by 100x then render identically."""

    @staticmethod
    def _field(amp):
        d = np.linspace(10.0, 90.0, 7)
        r = np.linspace(100.0, 3000.0, 11)
        return Field(data=np.full((7, 11), amp, dtype=complex),
                     coords={'depth': d, 'range': r},
                     model='Synth', frequencies=100.0)

    @pytest.mark.parametrize('value', ['mag', 'real', 'mag_db', 'db'])
    def test_every_value_shares_one_scale(self, value):
        from uacpy.visualization.plots.fields import compare_models
        fig, axes = compare_models([self._field(1.0 + 0j),
                                    self._field(100.0 + 0j)],
                                   ['A', 'B'], value=value)
        clims = []
        for ax in np.asarray(axes).ravel()[:2]:
            meshes = [c for c in ax.collections if hasattr(c, 'get_clim')]
            assert meshes, "panel drew no mesh"
            clims.append(meshes[0].get_clim())
        assert clims[0] == pytest.approx(clims[1])
        plt.close(fig)

    def test_the_shared_scale_spans_both_fields(self):
        """The scale must actually cover both panels' data, so a 100x
        difference is visible rather than flattened."""
        from uacpy.visualization.plots.fields import compare_models
        fig, axes = compare_models([self._field(1.0 + 0j),
                                    self._field(100.0 + 0j)],
                                   ['A', 'B'], value='mag')
        mesh = [c for c in np.asarray(axes).ravel()[0].collections
                if hasattr(c, 'get_clim')][0]
        lo, hi = mesh.get_clim()
        assert lo <= 1.0 and hi >= 100.0
        plt.close(fig)


class TestCrossSectionKnobsKeepTheDepthAxis:
    """``plot_field`` drops singleton axes so a ``(1, n)`` field renders as the
    line cut it is — but ``env=`` / ``source=`` / ``receiver=`` draw over the
    physical (depth, range) plane, which a line cut no longer has. A
    single-receiver-depth run therefore keeps its depth axis whenever one of
    those is supplied, and the row is drawn as a band."""

    @staticmethod
    def _field(n_depth=1, n_range=24, extra=None):
        coords = {'depth': np.linspace(50.0, 50.0 + 10.0 * (n_depth - 1),
                                       n_depth),
                  'range': np.linspace(100.0, 6000.0, n_range)}
        shape = (n_depth, n_range)
        if extra:
            coords.update(extra)
            shape = shape + tuple(v.size for v in extra.values())
        rng = np.random.default_rng(3)
        data = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
        return Field(data=data * 1e-3, coords=coords,
                     model='Synth', frequencies=100.0)

    @staticmethod
    def _mesh(ax):
        meshes = [c for c in ax.collections if hasattr(c, 'get_coordinates')]
        assert meshes, "no heatmap mesh was drawn"
        return meshes[0]

    @pytest.mark.parametrize('knob', ['env', 'source', 'receiver'])
    def test_a_single_receiver_depth_takes_the_overlay(self, knob, env):
        """The mainstream OO path ``result.plot(env=env)`` must not hard-fail
        on a run the user cannot reshape — models pass a single receiver depth
        straight through."""
        arg = {
            'env': env,
            'source': uacpy.Source(depths=10.0, frequencies=100.0),
            'receiver': uacpy.Receiver(depths=50.0,
                                       ranges=np.linspace(100.0, 6000.0, 24)),
        }[knob]
        fig, ax = plots.plot_field(self._field(), **{knob: arg})
        assert ax.get_ylabel() == 'Depth (m)'
        plt.close(fig)

    def test_the_kept_row_is_drawn_with_a_visible_extent(self, env):
        """A length-1 axis has no spacing for ``shading='nearest'`` to work
        from, so the quads must get explicit edges or the panel is empty."""
        fig, ax = plots.plot_field(self._field(), env=env)
        co = self._mesh(ax).get_coordinates()
        assert float(co[..., 1].max() - co[..., 1].min()) > 0.0
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba()).astype(int)
        assert int((np.ptp(rgba[..., :3], axis=-1) > 12).sum()) > 0
        plt.close(fig)

    def test_without_a_cross_section_knob_it_is_a_line_cut(self):
        fig, ax = plots.plot_field(self._field())
        assert len(ax.lines) == 1
        assert not [c for c in ax.collections if hasattr(c, 'get_coordinates')]
        assert ax.get_xlabel() == 'Range (km)'
        plt.close(fig)

    def test_only_the_depth_range_plane_is_kept(self, env):
        """Other singleton axes still collapse: a frequency-pinned slab of a
        broadband run must reach the 2-axis heatmap, not a 3-axis rejection."""
        f = self._field(extra={'frequency': np.array([200.0])})
        fig, ax = plots.plot_field(f, env=env)
        assert ax.get_ylabel() == 'Depth (m)'
        plt.close(fig)

    def test_a_genuine_line_cut_rejects_the_knobs(self, env):
        """A field sliced to one axis by the *user* has no cross-section, and
        the knob must still be refused rather than silently dropped."""
        cut = self._field(n_depth=6).at(depth=50.0)
        with pytest.raises(ConfigurationError,
                           match=r"env= has no effect on a 1-D line cut"):
            plots.plot_field(cut, env=env)

    def test_contours_are_refused_on_a_one_sample_axis(self, env):
        """A contour is interpolated between neighbours; an axis held at one
        sample has none, and matplotlib's own error is an untyped TypeError."""
        with pytest.raises(ConfigurationError,
                           match=r"contours= needs at least 2 samples"):
            plots.plot_field(self._field(), env=env, contours=[60.0])


class TestLinearViewsGetTheLinearColormap:
    """``style.LINEAR_VIEW_COLORMAP`` covers every linear view of any quantity;
    only a dB view takes the quantity's own dB map. Signed pressure on the
    transmission-loss ``jet_r`` with an asymmetric autoscale puts zero at an
    arbitrary colour.

    ``mag_db`` takes that map MIRRORED, because it carries the negated
    quantity (``-field.db``): larger is louder there and larger is quieter
    on the loss view, so sharing one map unmirrored paints the same water
    two different colours — see
    ``test_the_loud_end_is_the_same_colour_in_both_db_views``."""

    @staticmethod
    def _field():
        d = np.linspace(10.0, 90.0, 6)
        r = np.linspace(100.0, 3000.0, 9)
        rng = np.random.default_rng(4)
        p = rng.standard_normal((6, 9)) + 1j * rng.standard_normal((6, 9))
        return Field(data=p, coords={'depth': d, 'range': r},
                     model='Synth', frequencies=100.0)

    @pytest.mark.parametrize('value, expected', [
        ('db', 'jet_r'), ('mag_db', 'jet'),
        ('mag', 'seismic'), ('real', 'seismic'), ('imag', 'seismic'),
    ])
    def test_colormap_per_value_mode(self, value, expected):
        fig, ax = plots.plot_field(self._field(), value=value)
        assert ax.collections[0].get_cmap().name == expected
        plt.close(fig)

    @pytest.mark.parametrize('value', ['real', 'imag'])
    def test_signed_views_are_symmetric_about_zero(self, value):
        fig, ax = plots.plot_field(self._field(), value=value)
        lo, hi = ax.collections[0].get_clim()
        assert lo == pytest.approx(-hi)
        plt.close(fig)

    def test_the_modulus_starts_at_zero(self):
        """|p| is non-negative, so anchoring at 0 keeps the diverging map's
        neutral colour on silence — the same reading as real/imag."""
        fig, ax = plots.plot_field(self._field(), value='mag')
        lo, hi = ax.collections[0].get_clim()
        assert lo == 0.0 and hi > 0.0
        plt.close(fig)

    @staticmethod
    def _pd_field():
        """A detection-probability field: real, dimensionless, in [0, 1]."""
        d = np.linspace(10.0, 90.0, 6)
        r = np.linspace(100.0, 3000.0, 9)
        pd = np.linspace(0.0, 1.0, 54).reshape(6, 9)
        return Field(data=pd, coords={'depth': d, 'range': r},
                     model='Sonar',
                     metadata={'kind': 'probability_of_detection',
                               'unit': '1'})

    def test_a_probability_gets_the_bounded_unsigned_map(self):
        """The signed linear map has half its range below zero, which a
        probability never reaches: the whole field came out in shades of red
        on an autoscaled (-1, 1), with the map's neutral white on P_D = 0."""
        fig, ax = plots.plot_field(self._pd_field())
        mesh = ax.collections[0]
        assert mesh.get_cmap().name == 'RdYlGn'
        assert mesh.get_clim() == pytest.approx((0.0, 1.0))
        plt.close(fig)

    def test_a_probability_renders_alike_through_either_door(self):
        """``Field.plot`` and ``plot_detection_probability`` draw one field, so
        they must agree on the map, the window and the colorbar label."""
        from uacpy.visualization.plots.fields import (
            plot_detection_probability)
        f = self._pd_field()
        fig_a, ax_a = plots.plot_field(f)
        fig_b, ax_b = plot_detection_probability(f)
        mesh_a, mesh_b = ax_a.collections[0], ax_b.collections[0]
        assert mesh_a.get_cmap().name == mesh_b.get_cmap().name
        assert mesh_a.get_clim() == pytest.approx(mesh_b.get_clim())
        assert (fig_a.axes[-1].get_ylabel()
                == fig_b.axes[-1].get_ylabel()
                == 'Probability of detection')
        plt.close('all')

    @pytest.mark.parametrize('value', ['db', 'mag_db', 'mag', 'real'])
    def test_compare_models_picks_the_same_colormap(self, value):
        """One field must not render two ways through the two public entry
        points."""
        from uacpy.visualization.plots.fields import compare_models
        f = self._field()
        fig_one, ax_one = plots.plot_field(f, value=value)
        single = ax_one.collections[0].get_cmap().name
        plt.close(fig_one)
        fig, axes = compare_models([f, f], ['A', 'B'], value=value)
        assert np.asarray(axes).ravel()[0].collections[0].get_cmap().name \
            == single
        plt.close(fig)


class TestSeafloorSpansTheWholePanel:
    """The bathymetry is anchored to both ends of the data range. Without the
    end anchors a profile NARROWER than the field stops at its last sample and
    the panel shows a water column with no seabed under it — while the model
    held that depth out to the end of the run."""

    @staticmethod
    def _field():
        d = np.linspace(10.0, 90.0, 6)
        r = np.linspace(0.0, 10000.0, 12)
        return Field(data=np.full((6, 12), 60.0),
                     coords={'depth': d, 'range': r},
                     model='Synth', frequencies=100.0)

    @pytest.mark.parametrize('bathy, label', [
        ([(2000.0, 100.0), (6000.0, 140.0)], 'narrower than the field'),
        ([(0.0, 100.0), (20000.0, 140.0)], 'wider than the field'),
        ([(0.0, 100.0), (10000.0, 140.0)], 'exactly the field'),
    ])
    def test_the_seabed_covers_the_data_range(self, bathy, label):
        e = uacpy.Environment(name='bt', bathymetry=bathy, ssp=1500.0)
        fig, ax = plots.plot_field(self._field(), env=e)
        seabed = [ln for ln in ax.lines if len(np.atleast_1d(ln.get_xdata())) > 1]
        assert seabed, f"no seafloor line drawn ({label})"
        x = np.concatenate([np.asarray(ln.get_xdata()) for ln in seabed])
        assert x.min() == pytest.approx(0.0)
        assert x.max() == pytest.approx(10.0)
        plt.close(fig)


class TestStackedTracesLabelRangeInKm:
    """Every other view converts a range axis to km via ``_coord_axis``; the
    stacked view labelled the same axis in metres."""

    def test_axis_label_and_ticks_are_km(self):
        r = np.linspace(100.0, 1000.0, 5)
        t = np.linspace(0.0, 0.1, 8)
        f = Field(data=np.random.default_rng(5).standard_normal((5, 8)),
                  coords={'range': r, 'time': t},
                  model='Synth', frequencies=100.0)
        fig, ax = plots.plot_field(f, stacked=True)
        assert ax.get_ylabel() == 'Range (km) (stacked)'
        labels = [tl.get_text() for tl in ax.get_yticklabels()]
        assert labels[0] == '0.1' and labels[-1] == '1'
        # Significant digits, not one decimal: 0.325 km and 0.55 km must not
        # collapse to the same tick label.
        assert len(set(labels)) == len(labels)
        plt.close(fig)


class TestReceiverLatticeFitsThePanel:
    """A count-only cap is blind to the room the panel has: the same 20 x 10
    lattice that reads cleanly on a full-page figure buries the TL heatmap of a
    composite panel a fifth the size."""

    @staticmethod
    def _dots(ax):
        return sum(len(np.atleast_1d(ln.get_xdata()))
                   for ln in ax.lines if ln.get_marker() == 'o')

    def test_a_small_panel_draws_fewer_markers(self):
        d = np.linspace(5.0, 95.0, 50)
        r = np.linspace(100.0, 10000.0, 200)
        rng = np.random.default_rng(6)
        f = Field(data=rng.standard_normal((50, 200)) * 1e-3,
                  coords={'depth': d, 'range': r},
                  model='Synth', frequencies=100.0)
        rec = uacpy.Receiver(depths=d, ranges=r)
        fig_big, ax_big = plots.plot_field(f, receiver=rec, figsize=(10, 5))
        big = self._dots(ax_big)
        plt.close(fig_big)
        fig_small, ax_small = plots.plot_field(f, receiver=rec, figsize=(2, 1.5))
        small = self._dots(ax_small)
        plt.close(fig_small)
        assert small < big, "the lattice ignored the panel size"
        assert small >= 4, "a panel must still show the lattice, not one dot"
        # A full-page panel keeps the documented 20 x 10 ceiling.
        assert big == 200


# ── which mappable the figure-level colorbar takes ──────────────────────────
#
# `compare_models` draws every panel through `plot_field` and then hangs one
# colorbar off the figure, taking its mappable from the last panel's axes. It
# selected `ax.collections[0]`, which is the panel's `QuadMesh` only because
# the mesh happens to be drawn before the contour set and the seafloor fills —
# on matplotlib >= 3.8 a `ContourSet` is itself a `Collection`, so the axes of
# a contoured panel holds three collections and index 0 is a statement about
# draw order, not about which one carries the shared colour scale. (What that
# scale spans is `TestCompareModelsSharesItsColourScale`, above.)
#
# The first test below is the one that bites: it puts a non-mesh collection on
# the axes before the mesh and checks the colorbar still gets the mesh. Through
# the public API alone the two forms agree, so the decoy stands in for any
# later edit that draws something first — the failure mode is silent (a
# colorbar built from a `fill_between` polygon carries the default 0..1 norm
# under a dB label, and matplotlib raises nothing).


_DEPTH = np.array([5.0, 25.0, 60.0])
_RANGE = np.array([0.0, 500.0, 1000.0, 1500.0])


def _grid(kind='pressure'):
    """A complex 2-D ``(depth, range)`` pressure field spanning a full phase
    turn, so the cyclic wrap at ±π is present in the data.

    ``kind`` labels the field as a derived dB quantity; the default leaves it
    a bare pressure field with no metadata.
    """
    mag = np.linspace(1e-4, 1e-2, 12).reshape(3, 4)
    phase = np.linspace(-3.14, 3.14, 12).reshape(3, 4)
    meta = None if kind == 'pressure' else {'kind': kind, 'unit': 'dB'}
    return Field(data=mag * np.exp(1j * phase),
                 coords={'depth': _DEPTH, 'range': _RANGE}, metadata=meta)


@pytest.fixture
def colorbar_mappables(monkeypatch):
    """Every mappable ``compare_models`` hands to ``fig.colorbar``.

    The function keeps no handle on the colorbar it draws, so the mappable is
    read off the call rather than off the returned figure.
    """
    seen = []
    real = mfig.Figure.colorbar

    def record(self, mappable, *args, **kwargs):
        seen.append(mappable)
        return real(self, mappable, *args, **kwargs)

    monkeypatch.setattr(mfig.Figure, 'colorbar', record)
    return seen


def _draw_decoy_then(real_plot_field):
    """A ``plot_field`` that puts a non-mesh collection on the axes first."""
    def wrapper(field, *args, ax=None, **kwargs):
        ax.fill_between([0.0, 1.0], [0.0, 0.0], [1.0, 1.0], color='0.8')
        return real_plot_field(field, *args, ax=ax, **kwargs)
    return wrapper


def test_colorbar_takes_the_mesh_when_another_collection_is_drawn_first(
        monkeypatch, colorbar_mappables):
    """Index 0 is the mesh by draw order alone, so a panel helper that drew
    anything before the ``pcolormesh`` would caption the whole figure with a
    polygon's default 0..1 scale under the field's own dB label."""
    monkeypatch.setattr(_fields, 'plot_field',
                        _draw_decoy_then(_fields.plot_field))
    fig, axes = compare_models([_grid()], labels=['a'], value='db')
    panel = axes[0, 0]
    assert [type(c).__name__ for c in panel.collections][0] != 'QuadMesh'
    assert len(colorbar_mappables) == 1
    assert isinstance(colorbar_mappables[0], mcoll.QuadMesh)
    assert colorbar_mappables[0] is next(
        c for c in panel.collections if isinstance(c, mcoll.QuadMesh))


def test_no_colorbar_when_no_panel_drew_a_mesh(monkeypatch,
                                               colorbar_mappables):
    """The selection has to keep a default: a panel with no mesh must leave
    the figure bare, not raise ``StopIteration`` out of a plotting call and
    not caption the figure with whatever else the axes holds."""
    def mesh_free(field, *args, ax=None, **kwargs):
        ax.fill_between([0.0, 1.0], [0.0, 0.0], [1.0, 1.0], color='0.8')
        return ax.figure, ax

    monkeypatch.setattr(_fields, 'plot_field', mesh_free)
    fig, axes = compare_models([_grid()], labels=['a'], value='db')
    assert axes[0, 0].collections            # the panel is not empty
    assert colorbar_mappables == []


def test_colorbar_maps_the_shared_scale_through_contours_and_seafloor(
        colorbar_mappables):
    """The real figure a contoured, seafloor-overlaid comparison produces: the
    colorbar is the last panel's mesh, at the limits every panel was drawn
    with. Draw order puts the mesh first here, so this one agrees with the
    index form it replaces — it pins the contract, not the selection."""
    env = uacpy.Environment(bathymetry=200.0, ssp=1500.0)
    fig, axes = compare_models([_grid(), _grid()], labels=['a', 'b'], env=env,
                               value='db', contours=[60.0, 80.0], vmin=40.0,
                               vmax=90.0)
    panel = axes[0, -1]
    kinds = [type(c).__name__ for c in panel.collections]
    assert 'QuadContourSet' in kinds and any('Poly' in k for k in kinds)
    assert len(colorbar_mappables) == 1
    assert colorbar_mappables[0] is next(
        c for c in panel.collections if isinstance(c, mcoll.QuadMesh))
    assert np.allclose(colorbar_mappables[0].get_clim(), (40.0, 90.0))


def _range_dependent_env() -> Environment:
    """Three-column layered seabed over sloping bathymetry — the shape whose
    panel paid a deep copy per range node."""
    def column(cp, cs, rho, alpha, thickness):
        return SeabedColumn(
            layers=[SedimentLayer(thickness=thickness, sound_speed=cp,
                                  density=rho, attenuation=alpha,
                                  shear_speed=cs, shear_attenuation=0.5),
                    SedimentLayer(thickness=2 * thickness, sound_speed=cp + 120,
                                  density=rho + 0.2, attenuation=alpha + 0.2,
                                  shear_speed=cs + 40, shear_attenuation=0.7)],
            halfspace=BoundaryProperties("half-space", sound_speed=cp + 400,
                                         density=rho + 0.5,
                                         attenuation=alpha + 0.4))
    return Environment(
        name="rd",
        bathymetry=Bathymetry(ranges=[0.0, 2500.0, 5000.0],
                              depths=[90.0, 130.0, 110.0]),
        ssp=[(0.0, 1500.0), (130.0, 1490.0)],
        bottom=Bottom(columns=[column(1600.0, 180.0, 1.8, 0.6, 8.0),
                               column(1720.0, 240.0, 2.0, 0.9, 14.0),
                               column(1650.0, 200.0, 1.9, 0.7, 11.0)],
                      ranges=[0.0, 2500.0, 5000.0]))


def test_seabed_property_grid_reads_columns_without_copying(monkeypatch) -> None:
    """``_seabed_property_grid`` resolves each range node with
    ``Bottom.column_index_at`` and reads the carrier's live column.

    ``Bottom.at`` deep-copies a whole layer stack to answer a query this loop
    uses only to read layer scalars, and it ran once per range node per panel.
    The values are unchanged — ``column_index_at`` is documented as ``at``'s
    read-only counterpart with the same nearest rule — so this pins the call,
    which is the only thing a revert would alter.
    """
    env = _range_dependent_env()
    calls = []
    original = _Bottom.at
    monkeypatch.setattr(
        _Bottom, "at",
        lambda self, **kwargs: (calls.append(kwargs), original(self, **kwargs))[1])

    r_km = np.linspace(0.0, 5.0, 240)
    z = np.linspace(0.0, 200.0, 200)
    seafloor = np.interp(r_km * 1000.0, env.bathymetry.ranges,
                         env.bathymetry.depths)
    grid = _seabed_property_grid(env.bottom, "sound_speed", r_km, z, seafloor)

    assert not calls, (
        f"_seabed_property_grid deep-copied a column {len(calls)} time(s) via "
        f"Bottom.at; it should index with Bottom.column_index_at"
    )

    # The grid still carries the right values: every cell below the local
    # seafloor holds the property of the layer containing that depth.
    assert np.isfinite(grid).sum() > 0
    for j in (0, 120, 239):
        column = env.bottom.columns[env.bottom.column_index_at(
            range=float(r_km[j] * 1000.0))]
        top = seafloor[j]
        for layer in column.layers:
            inside = (z >= top) & (z < top + layer.thickness)
            if inside.any():
                assert np.allclose(grid[inside, j], layer.sound_speed)
            top += layer.thickness
        deepest = z >= top
        if deepest.any():
            assert np.allclose(grid[deepest, j], column.halfspace.sound_speed)


def test_seabed_property_grid_leaves_the_water_column_blank() -> None:
    """Cells above the range-local seafloor stay NaN, so they render as the
    axes background rather than as a property value."""
    env = _range_dependent_env()
    r_km = np.linspace(0.0, 5.0, 60)
    z = np.linspace(0.0, 200.0, 80)
    seafloor = np.interp(r_km * 1000.0, env.bathymetry.ranges,
                         env.bathymetry.depths)
    grid = _seabed_property_grid(env.bottom, "sound_speed", r_km, z, seafloor)

    for j in range(len(r_km)):
        assert np.all(np.isnan(grid[z < seafloor[j], j]))
        assert np.all(np.isfinite(grid[z >= seafloor[j], j]))


def _range_cut(kind=None, offset=0.0):
    """1-D dB range cut; ``kind`` tags a non-pressure quantity."""
    r = np.linspace(100.0, 5000.0, 30)
    data = offset + 50.0 + 10.0 * np.log10(r)
    metadata = {'kind': kind, 'unit': 'dB'} if kind else None
    return Field(data=data, coords={'range': r}, model='Synth',
                 frequencies=100.0, metadata=metadata)


class TestCompareSharesOneQuantity:
    def test_mixed_kinds_are_refused_in_either_order(self):
        tl = _range_cut()
        reverb = _range_cut(kind='reverberation')
        for pair in ([tl, reverb], [reverb, tl]):
            with pytest.raises(ConfigurationError,
                               match="different physical quantities"):
                plots.compare(pair, labels=['A', 'B'])
        assert not plt.get_fignums()

    def test_same_kind_tl_overlay_draws_both_and_reads_downward(self):
        fig, ax = plots.compare([_range_cut(), _range_cut(offset=2.0)],
                                labels=['A', 'B'])
        assert len(ax.get_lines()) == 2
        assert ax.get_xlabel() == 'Range (km)'
        # A transmission-loss value axis runs downward: loud end on top.
        assert ax.yaxis_inverted()
        plt.close(fig)


class TestEnvironmentPanelSpansToReceivers:
    """The panel's xlim follows the furthest receiver; the water mesh, the
    sediment fill, and the seafloor line hold the last bathymetry value out
    to that edge, as every model does."""

    def _artists_xmax(self, ax):
        from matplotlib.collections import QuadMesh, PolyCollection
        meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
        fills = [c for c in ax.collections
                 if isinstance(c, PolyCollection) and not isinstance(c, QuadMesh)]
        mesh_xmax = max(float(m.get_coordinates()[..., 0].max())
                        for m in meshes) if meshes else -np.inf
        fill_xmax = max(float(p.vertices[:, 0].max())
                        for c in fills for p in c.get_paths()) if fills else -np.inf
        return mesh_xmax, fill_xmax

    def test_bathymetry_short_of_receivers_reaches_the_right_xlim(self):
        ranges = np.linspace(0.0, 10_000.0, 6)
        depths = np.linspace(100.0, 150.0, 6)
        env = uacpy.Environment(
            bathymetry=np.column_stack([ranges, depths]), ssp=1500.0)
        receiver = uacpy.Receiver(depths=[50.0], ranges=[20_000.0])
        fig, ax = env.plot(receiver=receiver)
        x_hi = max(ax.get_xlim())
        assert x_hi == pytest.approx(20.0)

        mesh_xmax, fill_xmax = self._artists_xmax(ax)
        assert mesh_xmax >= x_hi
        assert fill_xmax >= x_hi

        seafloor_lines = [l for l in ax.get_lines()
                          if np.atleast_1d(l.get_xdata()).size > ranges.size]
        assert seafloor_lines, "seafloor line missing its right-edge anchor"
        line = seafloor_lines[0]
        assert float(np.max(line.get_xdata())) == pytest.approx(x_hi)
        # The anchor continues the last bathymetry value, constant.
        assert float(np.asarray(line.get_ydata())[-1]) == pytest.approx(150.0)
        plt.close(fig)

    def test_range_dependent_ssp_mesh_reaches_the_right_xlim(self):
        from uacpy.core.environment import SoundSpeedProfile
        ssp = SoundSpeedProfile(
            depths=[0.0, 100.0],
            data=[[1500.0, 1510.0], [1500.0, 1510.0]],
            ranges=[0.0, 10_000.0],
        )
        env = uacpy.Environment(bathymetry=100.0, ssp=ssp)
        receiver = uacpy.Receiver(depths=[50.0], ranges=[20_000.0])
        fig, ax = env.plot(receiver=receiver)
        x_hi = max(ax.get_xlim())
        assert x_hi == pytest.approx(20.0)
        mesh_xmax, _ = self._artists_xmax(ax)
        assert mesh_xmax >= x_hi
        plt.close(fig)


class TestTimeSnapshotsValidateCoords:
    def test_field_without_time_axis_is_refused_naming_the_axis(self):
        f = Field(
            data=np.zeros((4, 5)),
            coords={'depth': np.linspace(0.0, 100.0, 4),
                    'range': np.linspace(100.0, 2000.0, 5)},
            model='Synth', frequencies=100.0,
        )
        with pytest.raises(ConfigurationError,
                           match=r"missing coord axes.*time"):
            plots.plot_time_snapshots({'M': f}, times_s=[0.01])
        assert not plt.get_fignums()


class TestReferenceLabel:
    def test_named_references_keep_their_micro_pascal_shorthand(self):
        assert _ref_label(1e-6) == "1µ"
        assert _ref_label(20e-6) == "20µ"

    def test_custom_reference_renders_as_a_compact_number(self):
        assert _ref_label(1.0) == "1"
        assert _ref_label(2e-7) == "2e-07"


def test_rays_plot_with_mismatched_ray_arrays_raises_typed_error_and_closes_figures():
    """A ray whose r and z vectors disagree in length raises
    ``ConfigurationError`` (not matplotlib's raw ``ValueError``) and leaves
    pyplot's figure registry empty."""
    rays = Rays(
        rays=[{"r": np.linspace(0.0, 1000.0, 50), "z": np.zeros(49),
               "n_top_bounces": 0, "n_bot_bounces": 0}],
        model="Bellhop",
    )
    with pytest.raises(ConfigurationError, match="invalid plot input"):
        rays.plot()
    assert not plt.get_fignums()


def _field_with_a_no_energy_cell():
    """A 1/r field carrying one no-energy cell and one genuine deep null.

    The two must be treated differently: ``PRESSURE_FLOOR`` marks a cell the
    model reported no energy for, while the -70 dB null is a real
    interference minimum the field computed.
    """
    from uacpy.core.results import Field, PhaseReference
    from uacpy.core.constants import PRESSURE_FLOOR
    depths = np.linspace(0.0, 200.0, 40)
    ranges = np.linspace(10.0, 10000.0, 60)
    R, _ = np.meshgrid(ranges, depths)
    p = (1.0 / R).astype(complex)
    p[5, 5] = PRESSURE_FLOOR
    p[20, 30] = np.abs(p).max() * 10 ** (-70 / 20)
    return Field(data=p, coords={'depth': depths, 'range': ranges},
                 model='Synthetic', source_depths=np.array([50.0]),
                 frequencies=1000.0,
                 phase_reference=PhaseReference.TRAVELLING_WAVE)


def _clim(ax):
    for child in ax.get_children():
        if hasattr(child, 'get_clim') and child.get_clim()[0] is not None:
            return child.get_clim()
    raise AssertionError('no mappable on the axes')


def test_a_no_energy_cell_does_not_set_the_db_colour_limit():
    """600 dB is the marker for "no energy here", not a level the model
    computed, so it must not decide the scale: letting it in stretches the
    bar to 580 dB and paints every real level into the top sixth of it."""
    fig, ax = _field_with_a_no_energy_cell().plot(value='mag_db')
    lo, hi = _clim(ax)
    assert hi - lo < 120.0, f"colour bar spans {hi - lo:.0f} dB"
    plt.close(fig)


def _real_signal_excess_field():
    """A real, already-in-dB field: signal excess, where the SIGN is the
    meaning — SE > 0 is detectable, SE < 0 is not."""
    from uacpy.core.results import Field
    depths = np.linspace(0.0, 200.0, 5)
    ranges = np.linspace(10.0, 1000.0, 6)
    se = np.linspace(-20.0, 40.0, 30).reshape(5, 6)
    return Field(data=se, coords={'depth': depths, 'range': ranges},
                 model='Synth', source_depths=np.array([50.0]),
                 frequencies=1000.0, metadata={'kind': 'signal_excess'})


@pytest.mark.parametrize('value', ['mag', 'mag_db', 'phase', 'imag'])
def test_the_views_of_a_complex_field_are_refused_on_a_real_one(value):
    """``.db`` returns ``-20*log10|data|`` for complex data but the data
    ITSELF for real data, which is already a level. A view derived from the
    complex payload therefore has nothing to read on a real field, and
    ``mag_db`` negating that level silently inverted it: -20 dB of signal
    excess, meaning undetectable, plotted as +20."""
    with pytest.raises(ConfigurationError, match='complex'):
        _real_signal_excess_field().plot(value=value)


def test_the_loud_end_is_the_same_colour_in_both_db_views():
    """``mag_db`` is ``-field.db``: the same water, the opposite sign. The
    colours have to run the opposite way with it, or the two dB views paint
    one cell two different colours — measured, the loudest water came out
    dark red under ``db`` and dark blue under ``mag_db``, while style.py
    states the convention as "LOW TL (loud, near) is red"."""
    from uacpy.core.results import Field, PhaseReference
    depths = np.linspace(0.0, 200.0, 20)
    ranges = np.linspace(10.0, 5000.0, 30)
    R, _ = np.meshgrid(ranges, depths)
    field = Field(data=(1.0 / R).astype(complex),
                  coords={'depth': depths, 'range': ranges},
                  model='Synthetic', source_depths=np.array([50.0]),
                  frequencies=1000.0,
                  phase_reference=PhaseReference.TRAVELLING_WAVE)

    def loud_colour(value):
        fig, ax = field.plot(value=value)
        im = [c for c in ax.get_children()
              if hasattr(c, 'get_clim') and c.get_clim()[0] is not None][0]
        arr = im.get_array().reshape(len(depths), len(ranges))
        rgba = im.cmap(im.norm(arr[10, 0]))       # nearest range = loudest
        plt.close(fig)
        return np.asarray(rgba[:3])

    assert np.allclose(loud_colour('db'), loud_colour('mag_db'), atol=0.1), (
        f"db paints the loudest cell {loud_colour('db')} and mag_db paints "
        f"it {loud_colour('mag_db')}")


def test_the_marker_is_dropped_whichever_sign_the_db_view_carries():
    """``db`` and ``mag_db`` are the same numbers with opposite signs —
    ``mag_db`` is literally ``-field.db`` — so the no-energy marker is -600
    on one and +600 on the other. A filter written for one direction leaves
    the other setting the limit: a loss view then runs to 600 dB and packs
    the real levels into the bottom tenth of the bar."""
    from uacpy.core.results import Field, PhaseReference
    from uacpy.core.constants import PRESSURE_FLOOR
    depths = np.linspace(0.0, 200.0, 20)
    ranges = np.linspace(10.0, 5000.0, 30)
    R, _ = np.meshgrid(ranges, depths)
    p = (1.0 / R).astype(complex)
    p[3, 3] = PRESSURE_FLOOR
    # A kind whose dB view is not transmission loss, so it carries no fixed
    # limits and the auto-limit branch is the one under test.
    field = Field(data=p, coords={'depth': depths, 'range': ranges},
                  model='Synthetic', source_depths=np.array([50.0]),
                  frequencies=1000.0,
                  phase_reference=PhaseReference.TRAVELLING_WAVE,
                  metadata={'kind': 'reverberation'})
    fig, ax = field.plot(value='db')
    lo, hi = _clim(ax)
    assert hi - lo < 120.0, f"loss colour bar spans {hi - lo:.0f} dB"
    plt.close(fig)


def test_a_genuine_deep_null_keeps_its_colour():
    """The remedy has to distinguish a marker from data. A percentile does
    not: the 1st percentile of this field lands at -80 dB and clips the real
    -70 dB interference null, which is exactly the feature the view is for."""
    field = _field_with_a_no_energy_cell()
    fig, ax = field.plot(value='mag_db')
    lo, _ = _clim(ax)
    db = 20 * np.log10(np.abs(np.asarray(field.data)))
    null = np.sort(db.ravel())[1]          # deepest real level, past the marker
    assert lo <= null + 1e-6, (
        f"colour floor {lo:.1f} dB clips the genuine null at {null:.1f} dB")
    plt.close(fig)


def _absorbing_arrivals():
    """A 40 kHz link whose second cluster is heavily absorbed.

    The amplitude COLUMN says the two clusters are within a factor of 1.5;
    the received level says the second is 16 dB down, because Bellhop keeps
    volume absorption in the imaginary travel time and the second path is
    1.24 km longer.
    """
    f0, alpha_db_per_km = 40e3, 12.90
    def dimag(arc_km):
        return -(alpha_db_per_km * arc_km / 8.6858896) / (2 * np.pi * f0)
    cell = {
        "delays": np.array([0.669, 1.496, 8.06]),
        "amplitudes": np.array([2.23e-4, 1.49e-4, 1.0e-8]),
        "phases": np.zeros(3),
        "n_top_bounces": np.array([0, 1, 9]),
        "n_bot_bounces": np.array([0, 1, 9]),
        "src_angles": np.zeros(3), "rcv_angles": np.zeros(3),
        "delays_imag": np.array([dimag(1.0), dimag(2.24), dimag(12.0)]),
    }
    return Arrivals(by_receiver=[[[cell]]],
                    receiver_depths=np.array([999.0]),
                    receiver_ranges=np.array([1000.0]),
                    model='Bellhop', frequencies=f0)


def test_the_stems_are_drawn_at_the_level_that_reaches_the_receiver():
    """Drawing the amplitude column alone puts an absorbed path at its
    lossless height: on this set the second cluster is 16 dB down and would
    be drawn within a factor 1.5 of the direct one."""
    arr = _absorbing_arrivals()
    fig, ax = arr.plot()
    heads = [ln.get_ydata()[0] for ln in ax.lines if ln.get_marker() == 'o']
    heads = sorted(heads, reverse=True)
    assert heads[1] / heads[0] < 0.25, (
        f"second cluster drawn at {heads[1] / heads[0]:.2f} of the direct; "
        f"the received ratio is about 0.16")
    plt.close(fig)


def test_the_delay_axis_follows_the_energy_not_the_last_arrival():
    """The peak-to-peak span is set by whichever ray arrives last however
    faint. Here that is 8.06 s against an energy span under 0.9 s, so a
    peak-to-peak axis spends 90% of its width on arrivals carrying
    nothing."""
    arr = _absorbing_arrivals()
    fig, ax = arr.plot()
    lo, hi = ax.get_xlim()
    assert hi - lo < 2000.0, f"axis spans {hi - lo:.0f} ms"
    assert hi > 1496.0, "the second cluster must stay on the axis"
    plt.close(fig)


def test_arrivals_left_off_the_axis_are_declared():
    """An arrival outside the drawn span vanishes off-axis entirely, unlike
    an outlier on a colour scale, so the plot has to say it is there."""
    arr = _absorbing_arrivals()
    fig, ax = arr.plot()
    text = ' '.join(t.get_text() for t in ax.texts)
    text += ' '.join(t.get_text() for t in ax.get_legend().get_texts()) \
        if ax.get_legend() else ''
    assert '1' in text and ('beyond' in text or 'outside' in text
                            or 'off' in text), text
    plt.close(fig)


def test_arrivals_plot_with_missing_delay_key_raises_typed_error_and_closes_figures():
    """An arrival dict lacking ``'delay'`` raises ``ConfigurationError``
    (not a raw ``KeyError``) and leaves no open figure."""
    arrivals = Arrivals(
        arrivals=[{"kind": "direct", "amplitude": 1.0}],
        receiver_depths=np.array([10.0]),
        receiver_ranges=np.array([100.0]),
        model="Bellhop",
    )
    with pytest.raises(ConfigurationError, match="invalid plot input"):
        arrivals.plot()
    assert not plt.get_fignums()


# Entry points that are not plain decorated plotters: the type dispatcher
# (each of its targets is decorated) and the two axis-annotation overlays,
# which draw onto an axes the caller owns and open no figure of their own.
_UNDECORATED_ENTRY_POINTS = {
    "plot_result", "draw_sound_cone", "draw_slowness_line",
}


# Private renderers reached through ``result.plot()`` / ``env.plot()`` /
# ``env.ssp.plot()`` rather than ``plots.__all__``.
_PRIVATE_RENDERERS = (
    ("rays_modes", "_plot_rays"),
    ("rays_modes", "_plot_arrivals"),
    ("rays_modes", "_plot_mode_functions"),
    ("rays_modes", "_plot_reflection_coefficient"),
    ("rays_modes", "_plot_covariance"),
    ("rays_modes", "_plot_replicas"),
    ("fields", "_plot_field_stack"),
    ("environment", "_plot_environment"),
    ("environment", "_plot_ssp"),
    ("environment", "_plot_range_profile"),
)


def _is_wrapped(fn) -> bool:
    return inspect.unwrap(fn) is not fn


def test_every_plotter_entry_point_is_wrapped_by_typed_plot_error():
    """Every public ``plots.__all__`` plotter and every private per-type
    renderer carries the ``typed_plot_error`` wrapper, so plotting.md's
    "ConfigurationError + no figure left behind" claim holds package-wide."""
    unwrapped = []
    for name in plots.__all__:
        obj = getattr(plots, name)
        if inspect.ismodule(obj) or name in _UNDECORATED_ENTRY_POINTS:
            continue
        if not _is_wrapped(obj):
            unwrapped.append(name)
    for module_name, fn_name in _PRIVATE_RENDERERS:
        fn = getattr(getattr(plots, module_name), fn_name)
        if not _is_wrapped(fn):
            unwrapped.append(f"{module_name}.{fn_name}")
    assert not unwrapped, f"plotters without typed_plot_error: {unwrapped}"


def _range_time_field() -> Field:
    ranges = np.linspace(0.0, 2140.0, 5)
    times = np.linspace(0.0, 1.0, 4)
    rng = np.random.default_rng(0)
    return Field(
        data=rng.standard_normal((ranges.size, times.size)),
        coords={"range": ranges, "time": times},
        model="test",
    )


def test_heatmap_with_range_on_the_y_axis_draws_it_in_km():
    """A ``(range, time)`` heatmap puts range on y in km — same scale and
    label as every other view of a range axis."""
    fig, ax = plot_field(_range_time_field())
    assert ax.get_ylabel() == "Range (km)"
    lo, hi = sorted(ax.get_ylim())
    assert -1.0 < lo and hi < 3.0, (lo, hi)


def test_pinned_range_subtitle_reads_in_km():
    """Slicing at range=2140 m subtitles the cut ``Range = 2.14 km``."""
    fig, ax = plot_field(_range_time_field().at(range=2140.0))
    assert ax.get_title() == "Range = 2.14 km"


def _se_field(n_depth, n_range, *, scale=1.0):
    """Signal excess in dB on an ``(n_depth, n_range)`` grid spanning
    ``scale × (-20 … +40)`` dB, so it crosses the SE = 0 boundary."""
    depths = (np.linspace(10.0, 90.0, n_depth) if n_depth > 1
              else np.array([50.0]))
    ranges = (np.linspace(500.0, 4500.0, n_range) if n_range > 1
              else np.array([1000.0]))
    data = scale * np.linspace(-20.0, 40.0, n_depth * n_range)
    return Field(data=data.reshape(n_depth, n_range),
                 coords={'depth': depths, 'range': ranges},
                 metadata={'kind': 'signal_excess', 'unit': 'dB'},
                 model='test')


def _pd_field(n_depth, n_range):
    """Detection probability on an ``(n_depth, n_range)`` grid spanning 0 … 1,
    so every default contour level falls inside the data."""
    depths = (np.linspace(10.0, 90.0, n_depth) if n_depth > 1
              else np.array([50.0]))
    ranges = (np.linspace(500.0, 4500.0, n_range) if n_range > 1
              else np.array([1000.0]))
    return Field(data=np.linspace(0.0, 1.0, n_depth * n_range).reshape(
                     n_depth, n_range),
                 coords={'depth': depths, 'range': ranges},
                 metadata={'kind': 'probability_of_detection'}, model='test')


def _mesh_extent(ax) -> Bbox:
    """Data-space bounding box of every quad the heatmap drew."""
    mesh = ax.collections[0]
    return Bbox.union([p.get_extents() for p in mesh.get_paths()])


def _contour_sets(ax):
    return [c for c in ax.collections if 'Contour' in type(c).__name__]


def _ppsd_result(level_lo, level_hi, ref=1e-6):
    """A ``PPSDResult`` whose histogram sits between ``level_lo`` and
    ``level_hi`` dB, stated against ``ref``."""
    frequencies = np.array([100.0, 200.0, 400.0])
    level_edges = np.linspace(level_lo, level_hi, 5)
    pdf = np.full((level_edges.size - 1, frequencies.size), 0.25)
    mean_db = np.full(frequencies.size, 0.5 * (level_lo + level_hi))
    return PPSDResult(frequencies, level_edges, pdf, mean_db,
                      np.ones(frequencies.size), 1.0, 1.0, ref)


def _cq_ppsd_result(level_lo, level_hi, ref=1e-6):
    r = _ppsd_result(level_lo, level_hi, ref)
    return CQPPSDResult(r.frequencies, r.level_edges, r.pdf, r.mean_db,
                        r.std_db, r.binwidth_db, ref)


class TestALevelAxisNamesTheReferenceItWasComputedAgainst:
    """These plotters hardcoded "µPa²" while the analyses behind them all take
    ``ref=``. A caller working in Pascals got an axis labelled 120 dB away from
    the numbers printed on it, and nothing anywhere could catch that."""

    @pytest.mark.parametrize('ref, shown', [(1e-6, '1µ'), (1.0, '1'),
                                            (20e-6, '20µ')])
    def test_the_ppsd_level_axis_carries_the_result_reference(self, ref, shown):
        fig, ax = plot_ppsd(_ppsd_result(60.0, 90.0, ref))
        assert ax.get_ylabel() == f"Level (dB re {shown}Pa²/Hz)"
        plt.close(fig)

    @pytest.mark.parametrize('ref, shown', [(1e-6, '1µ'), (1.0, '1'),
                                            (20e-6, '20µ')])
    def test_the_constant_q_ppsd_level_axis_carries_it_too(self, ref, shown):
        fig, ax = plot_constant_q_ppsd(_cq_ppsd_result(60.0, 90.0, ref))
        assert ax.get_ylabel() == f"Level (dB re {shown}Pa²)"
        plt.close(fig)

    @pytest.mark.parametrize('ref, shown', [(1e-6, '1µ'), (1.0, '1')])
    def test_the_fk_colourbar_carries_its_reference(self, ref, shown):
        """``plot_fk`` converts with ``power_to_db(power, ref)``, so its
        colourbar is an absolute level, not a relative one."""
        from uacpy.visualization.plots.signal import plot_fk
        f = np.linspace(0.0, 500.0, 8)
        k = np.linspace(-1.0, 1.0, 9)
        fig, ax = plot_fk(f, k, np.ones((8, 9)), ref=ref)
        assert fig.axes[-1].get_ylabel() == f"Power (dB re {shown}Pa²)"
        plt.close(fig)


def _bands():
    """``sel``-shaped ``(low, centre, high)`` band triples."""
    return [(88.0, 100.0, 112.0), (112.0, 125.0, 141.0)]


def _modes(n=10):
    depths = np.linspace(0.0, 100.0, 21)
    phi = np.sin(np.outer(depths, np.arange(1, n + 1)) * np.pi / 100.0)
    return Modes(k=np.linspace(0.42, 0.30, n) + 0j, phi=phi, depths=depths,
                 model='Test', frequencies=100.0)


def _rays():
    return Rays(rays=[{'r': np.linspace(0.0, 2000.0, 20),
                       'z': np.linspace(0.0, 80.0, 20),
                       'n_top_bounces': 1, 'n_bot_bounces': 0}],
                model='Bellhop')


@pytest.mark.parametrize("shape", [(1, 8), (8, 1)])
def test_signal_excess_renders_an_axis_held_at_one_sample_as_a_band(shape):
    """A single-receiver-depth run (and its transpose, a single range) is a
    mainstream shape. The SE = 0 contour raised a raw ``TypeError`` on it, and
    ``show_boundary=False`` drew a zero-extent mesh — an empty panel."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig, ax = plot_signal_excess(_se_field(*shape))
    extent = _mesh_extent(ax)
    assert extent.width > 0.0 and extent.height > 0.0
    plt.close(fig)


@pytest.mark.parametrize("shape", [(1, 8), (8, 1)])
def test_detection_probability_renders_an_axis_held_at_one_sample_as_a_band(
        shape):
    """Same shape, same crash: ``plot_detection_probability`` drew its ``P_D``
    contours through the same ``contour`` call."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig, ax = plot_detection_probability(_pd_field(*shape))
    extent = _mesh_extent(ax)
    assert extent.width > 0.0 and extent.height > 0.0
    plt.close(fig)


@pytest.mark.parametrize("shape", [(1, 8), (8, 1)])
def test_signal_excess_warns_that_one_sample_leaves_no_boundary_to_trace(shape):
    """A contour is interpolated between neighbouring samples, so the SE = 0
    boundary cannot be drawn — the heatmap can, and the panel says which."""
    with pytest.warns(UserWarning, match="at least 2 samples on both axes"):
        fig, ax = plot_signal_excess(_se_field(*shape))
    assert not _contour_sets(ax)
    plt.close(fig)


@pytest.mark.parametrize("shape", [(1, 8), (8, 1)])
def test_detection_probability_warns_that_one_sample_leaves_no_contour(shape):
    with pytest.warns(UserWarning, match="at least 2 samples on both axes"):
        fig, ax = plot_detection_probability(_pd_field(shape[0], shape[1]))
    assert not _contour_sets(ax)
    plt.close(fig)


def test_signal_excess_draws_the_boundary_when_both_axes_carry_samples():
    """The skip is keyed on the degenerate shape alone: an ordinary grid keeps
    its detection boundary."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fig, ax = plot_signal_excess(_se_field(4, 5))
    assert _contour_sets(ax)
    plt.close(fig)


def test_compare_models_signal_excess_window_spans_every_panel():
    """The symmetric window came from ``fields[0]`` alone, so a wider second
    panel saturated (measured: 70% of its samples) under a colourbar that
    claimed to describe it."""
    narrow, wide = _se_field(4, 5), _se_field(4, 5, scale=4.0)
    fig, axes = compare_models([narrow, wide], value='db')
    lo, hi = axes[0, 0].collections[0].get_clim()
    peak = float(np.max(np.abs(np.asarray(wide.data))))
    assert (lo, hi) == pytest.approx((-peak, peak))
    assert np.all(np.asarray(wide.data) >= lo)
    assert np.all(np.asarray(wide.data) <= hi)
    plt.close(fig)


def test_compare_models_signal_excess_window_ignores_the_panel_order():
    """Order-dependence was the proof the window was one panel's own: reversing
    the list changed the shared colour limits."""
    narrow, wide = _se_field(4, 5), _se_field(4, 5, scale=4.0)
    fig_a, axes_a = compare_models([narrow, wide], value='db')
    fig_b, axes_b = compare_models([wide, narrow], value='db')
    assert (axes_a[0, 0].collections[0].get_clim()
            == axes_b[0, 0].collections[0].get_clim())
    plt.close(fig_a)
    plt.close(fig_b)


def test_compare_models_keeps_signal_excess_zero_on_the_neutral_colour():
    """A diverging map carries its meaning in the neutral colour, and for
    signal excess that colour is the SE = 0 dB detection boundary. The generic
    pooled scale is an asymmetric min/max and would move it."""
    fig, axes = compare_models([_se_field(4, 5), _se_field(4, 5, scale=4.0)],
                               value='db')
    lo, hi = axes[0, 0].collections[0].get_clim()
    assert lo == pytest.approx(-hi)
    plt.close(fig)


def test_compare_models_signal_excess_panels_share_one_window():
    fig, axes = compare_models([_se_field(4, 5), _se_field(4, 5, scale=4.0)],
                               value='db')
    clims = [ax.collections[0].get_clim() for ax in axes.ravel()]
    assert clims[0] == clims[1]
    plt.close(fig)


def _plot_offscreen_ppsd():
    return plot_ppsd(_ppsd_result(300.0, 320.0))


def _plot_offscreen_sel():
    return plot_sel(np.array([1e-30, 1e-30]), _bands(), duration=1.0)


def _plot_offscreen_constant_q_psd():
    return plot_constant_q_psd(np.array([100.0, 200.0, 400.0]),
                               np.array([1e-30, 1e-30, 1e-30]))


def _plot_offscreen_constant_q_ppsd():
    return plot_constant_q_ppsd(_cq_ppsd_result(300.0, 320.0))


def _plot_offscreen_frf():
    return plot_frf(np.array([10.0, 100.0, 1000.0]),
                    np.array([1e-9 + 0j, 1e-9 + 0j, 1e-9 + 0j]))


@pytest.mark.parametrize("call, caller", [
    (_plot_offscreen_ppsd, "plot_ppsd"),
    (_plot_offscreen_sel, "plot_sel"),
    (_plot_offscreen_constant_q_psd, "plot_constant_q_psd"),
    (_plot_offscreen_constant_q_ppsd, "plot_constant_q_ppsd"),
    (_plot_offscreen_frf, "plot_frf"),
])
def test_a_record_outside_the_pinned_level_window_says_the_panel_is_empty(
        call, caller):
    """These plotters pin the ordinate to the range their quantity normally
    occupies, so a record in the wrong units renders as a blank panel that
    looks like "no data". ``plot_psd`` / ``plot_coherence`` warned; these five
    did not."""
    with pytest.warns(UserWarning, match=f"{caller}: every sample"):
        fig, _ = call()
    plt.close(fig)


def test_a_record_inside_the_pinned_level_window_is_not_flagged():
    """The warning fires on an empty panel only — an ordinary record must not
    train the reader to ignore it."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fig, _ = plot_frf(np.array([10.0, 100.0, 1000.0]),
                          np.array([1.0 + 1.0j, 2.0 + 0j, 0.5 - 0.5j]))
        plt.close(fig)
        fig, _ = plot_ppsd(_ppsd_result(60.0, 90.0))
        plt.close(fig)


def test_stacked_traces_take_a_caller_supplied_linewidth():
    """``linewidth`` was hardcoded into the same ``plot`` call ``**mpl_kw``
    lands in, so passing it raised a raw ``TypeError``."""
    field = Field(data=np.random.default_rng(0).standard_normal((3, 6)),
                  coords={'range': np.linspace(0.0, 1000.0, 3),
                          'time': np.linspace(0.0, 1.0, 6)}, model='test')
    fig, ax = plot_field(field, stacked=True, linewidth=2.5)
    assert ax.lines[0].get_linewidth() == pytest.approx(2.5)
    plt.close(fig)
    fig, ax = plot_field(field, stacked=True)
    assert ax.lines[0].get_linewidth() == pytest.approx(0.8)
    plt.close(fig)


@pytest.mark.parametrize("show_components", [True, False])
def test_plot_wenz_total_line_takes_caller_supplied_styles(show_components):
    """``**mpl_kw`` is documented as styling the total-noise line, but colour,
    width and label were all hardcoded into that call."""
    wenz = WenzNoise(np.logspace(1, 5, 40), wind_speed_kn=10.0)
    fig, ax = plot_wenz(wenz, show_components=show_components, color='red',
                        linewidth=3.5, label='mine')
    total = ax.lines[0]
    assert total.get_color() == 'red'
    assert total.get_linewidth() == pytest.approx(3.5)
    assert total.get_label() == 'mine'
    plt.close(fig)


def test_plot_wenz_total_line_keeps_its_own_style_by_default():
    wenz = WenzNoise(np.logspace(1, 5, 40), wind_speed_kn=10.0)
    fig, ax = plot_wenz(wenz)
    total = ax.lines[0]
    assert total.get_color() == 'black'
    assert total.get_linewidth() == pytest.approx(2.0)
    assert total.get_label().startswith('Total noise')
    plt.close(fig)


def test_plot_source_level_takes_a_caller_supplied_marker():
    """``marker`` collided; ``label`` is a named parameter and never did."""
    fig, ax = plot_source_level(np.array([100.0, 200.0]),
                                np.array([120.0, 118.0]), marker='s',
                                label='ship')
    assert ax.lines[0].get_marker() == 's'
    assert ax.lines[0].get_label() == 'ship'
    plt.close(fig)
    fig, ax = plot_source_level(np.array([100.0, 200.0]),
                                np.array([120.0, 118.0]))
    assert ax.lines[0].get_marker() == 'o'
    plt.close(fig)


def test_ray_plot_rejects_an_unknown_colour_mode():
    """A typo'd ``color_by`` fell through to the monochrome branch and dropped
    the per-class legend, so the fan came back looking like a deliberate
    ``color_by=None`` call."""
    with pytest.raises(ConfigurationError, match="not a colouring mode"):
        _plot_rays(_rays(), color_by='bounce')
    assert not plt.get_fignums()


@pytest.mark.parametrize("color_by", ['bounces', None])
def test_ray_plot_accepts_its_two_colour_modes(color_by):
    fig, ax = _plot_rays(_rays(), color_by=color_by)
    assert ax.lines
    plt.close(fig)


@pytest.mark.parametrize("mode_range", [(-3, 4), (-3, 8), (5, 2), (4, 4)])
def test_mode_heatmap_rejects_a_range_that_selects_nothing(mode_range):
    """A negative start wraps under numpy slicing and ``start >= stop`` selects
    no column; ``(-3, 8)`` with ``normalize=False`` reached ``pcolormesh`` as a
    raw ``TypeError`` naming neither knob."""
    for normalize in (True, False):
        with pytest.raises(ConfigurationError, match="0 <= start < stop"):
            plot_modes_heatmap(_modes(), mode_range=mode_range,
                               normalize=normalize)
    assert not plt.get_fignums()


def test_mode_heatmap_rejects_a_range_starting_past_the_last_mode():
    with pytest.raises(ConfigurationError, match="starts past"):
        plot_modes_heatmap(_modes(10), mode_range=(12, 20))
    assert not plt.get_fignums()


def test_mode_heatmap_clamps_a_range_reaching_past_the_last_mode():
    """The open end is a slice bound, not a claim about how many modes exist."""
    fig, ax = plot_modes_heatmap(_modes(10), mode_range=(2, 999))
    assert '8 modes' in ax.get_title()
    plt.close(fig)


@pytest.mark.parametrize('kwargs', [
    {'mode_range': (3, 7)}, {'n_modes': 4}, {'n_modes': 2}, {},
])
def test_the_mode_axis_is_ticked_on_whole_modes_only(kwargs):
    """A mode index is an integer; mode 4.5 does not exist. Over a short span
    the default locator subdivides, and ``mode_range=(3, 7)`` came back with
    3.5, 4.5, 5.5, 6.5 and 7.5 — five of nine ticks naming modes the field
    does not contain."""
    fig, ax = plot_modes_heatmap(_modes(10), **kwargs)
    lo, hi = ax.get_xlim()
    on_axis = [t for t in ax.get_xticks() if lo <= t <= hi]
    assert on_axis, "the mode axis lost every tick"
    assert all(float(t).is_integer() for t in on_axis), on_axis
    plt.close(fig)


def test_mode_heatmap_rejects_n_modes_together_with_mode_range():
    """``mode_range`` took the slice wholesale, so ``n_modes`` was dropped
    without a word."""
    with pytest.raises(ConfigurationError, match="pass one"):
        plot_modes_heatmap(_modes(10), 3, mode_range=(0, 8))
    assert not plt.get_fignums()


def _cut(kind):
    """A 1-D range cut of a dB quantity, ascending with range."""
    data = np.array([4.18, 10.0, 15.82, 20.0])
    if kind == 'pressure':
        return Field(data=data, coords={'range': _RANGE})
    return Field(data=data, coords={'range': _RANGE},
                 metadata={'kind': kind, 'unit': 'dB'})


def _ylim_direction(ax):
    """``'down'`` when the y axis increases downward, else ``'up'``."""
    lo, hi = ax.get_ylim()
    return 'down' if lo > hi else 'up'


@pytest.mark.parametrize("kind, expected", [
    ('pressure', 'down'),            # TL: a loss, so the loud end is the top
    ('signal_excess', 'up'),         # a level: more is more
    # OASES writes -10*log10 E[|p_scat|^2] (oassun26.f:633-637, :853-857), so
    # reverberation is a LOSS like TL and its axis runs the same way.
    ('reverberation', 'down'),
])
def test_db_cut_points_the_same_way_through_either_entry_point(kind, expected):
    """``compare`` keyed the flip on ``value == 'db'`` and ``plot_field`` on the
    quantity, so every dB view except TL came out vertically mirrored between
    the two — and the tick labels stayed truthful, so nothing looked wrong.

    Direction follows the quantity, not the entry point: a loss reads downward
    through both doors, a level upward through both."""
    field = _cut(kind)
    _, ax_single = plot_field(field, value='db')
    _, ax_overlay = compare([field], labels=['a'], value='db')
    assert _ylim_direction(ax_single) == expected
    assert _ylim_direction(ax_overlay) == expected


def test_a_real_tl_grid_and_a_complex_one_agree():
    """TL reaches the plotters two ways — a real dB grid from RAM, complex
    pressure from Kraken — and both are transmission loss."""
    real_tl = Field(data=np.array([40.0, 60.0, 80.0, 100.0]),
                    coords={'range': _RANGE},
                    metadata={'kind': 'pressure', 'unit': 'dB'})
    complex_tl = Field(data=np.array([1e-2, 1e-3, 1e-4, 1e-5]) * (1 + 0j),
                       coords={'range': _RANGE})
    for field in (real_tl, complex_tl):
        _, ax = plot_field(field, value='db')
        assert _ylim_direction(ax) == 'down'
        _, ax = compare([field], labels=['a'], value='db')
        assert _ylim_direction(ax) == 'down'


def test_a_depth_cut_inverts_for_every_quantity():
    """Depth on Y is the oceanographic convention and has nothing to do with
    which way the quantity reads."""
    field = Field(data=np.array([1.0, 2.0, 3.0]), coords={'depth': _DEPTH},
                  metadata={'kind': 'signal_excess', 'unit': 'dB'})
    _, ax = compare([field], labels=['a'], value='db')
    assert _ylim_direction(ax) == 'down'
    assert ax.get_ylabel() == 'Depth (m)'


def _panel(fig_axes):
    fig, axes = fig_axes
    return axes[0, 0].collections[0]


@pytest.mark.parametrize("value", ['db', 'mag_db', 'mag', 'phase',
                                   'real', 'imag'])
def test_one_field_renders_identically_alone_and_in_a_panel(value):
    """``compare_models`` computed one figure-level colormap and colour range
    and pushed them into every panel, overriding the per-value defaults
    ``plot_field`` applies."""
    field = _grid()
    _, ax = plot_field(field, value=value)
    alone = ax.collections[0]
    tiled = _panel(compare_models([field], value=value))
    assert tiled.get_cmap().name == alone.get_cmap().name
    assert np.allclose(tiled.get_clim(), alone.get_clim())


def test_phase_keeps_its_cyclic_colormap_in_a_panel():
    """-π and +π are the same phase. On a non-cyclic map they land at opposite
    ends of the scale and the wrap reads as a real discontinuity."""
    mesh = _panel(compare_models([_grid()], value='phase'))
    assert mesh.get_cmap().name == 'twilight'
    assert np.allclose(mesh.get_clim(), (-np.pi, np.pi))


def test_a_quiet_response_is_not_stretched_to_a_symmetric_scale():
    """``mag_db`` is a level, not a signed quantity: a -80..-20 dB response
    forced out to ±80 occupies half the colormap."""
    mag = 10 ** (np.linspace(-80, -20, 12).reshape(3, 4) / 20)
    field = Field(data=mag * np.exp(1j * 0.1),
                  coords={'depth': _DEPTH, 'range': _RANGE})
    lo, hi = _panel(compare_models([field], value='mag_db')).get_clim()
    assert (lo, hi) == pytest.approx((-80.0, -20.0))


def test_a_signed_view_takes_a_symmetric_scale():
    """Zero has to keep the diverging map's neutral colour."""
    lo, hi = _panel(compare_models([_grid()], value='real')).get_clim()
    assert lo == pytest.approx(-hi)


def test_panels_share_one_pooled_scale():
    """The reason the figure-level limits exist at all: left to autoscale, two
    fields differing by 100x would render identically under one colorbar."""
    loud, quiet = _grid(), _grid()
    quiet = Field(data=quiet.data * 0.01,
                  coords={'depth': _DEPTH, 'range': _RANGE})
    fig, axes = compare_models([loud, quiet], value='mag')
    first = axes[0, 0].collections[0].get_clim()
    second = axes[0, 1].collections[0].get_clim()
    assert first == second
    assert first[1] == pytest.approx(float(np.max(np.abs(loud.data))))


@pytest.mark.parametrize("value, expected", [
    ('db', 'TL (dB)'),
    ('mag_db', '|H| (dB)'),
    ('mag', '|p|'),
    ('phase', 'Phase (rad)'),
    ('real', 'Re(p)'),
    ('imag', 'Im(p)'),
])
def test_the_shared_colorbar_names_the_quantity(value, expected):
    """The colorbar carried the raw ``value`` string for every non-dB view, so
    a phase panel was labelled 'phase' where plot_field writes 'Phase (rad)'."""
    fig, axes = compare_models([_grid()], value=value)
    panels = list(axes.ravel())
    labels = [a.get_ylabel() for a in fig.axes if a not in panels]
    assert expected in labels


def test_the_shared_colorbar_names_the_quantity_not_transmission_loss():
    """A dB view is labelled from the field: signal excess is not TL."""
    fig, axes = compare_models([_grid('signal_excess')], value='db')
    panels = list(axes.ravel())
    labels = [a.get_ylabel() for a in fig.axes if a not in panels]
    assert 'Signal excess (dB)' in labels


def _corner_overview(credits):
    """A transect that starts at the map window's top-left corner: the place
    where the left-aligned title, an 'upper left' legend and the 'A' label
    all used to meet."""
    lats, lons = np.linspace(58.0, 61.0, 40), np.linspace(2.0, 5.0, 50)
    LON, LAT = np.meshgrid(lons, lats)
    depth = 100 + 300 * (LAT - 58) / 3 + 20 * np.sin(LON * 3)
    env = uacpy.Environment(bathymetry=200.0, ssp=1500.0)
    fig, axes = plot_overview(env, (lats, lons, depth),
                              transect=((61.0, 2.0), (58.0, 5.0)),
                              map_kwargs=dict(basemap=False), title='overview',
                              data_source=credits)
    fig.canvas.draw()
    return fig, axes, fig.canvas.get_renderer()


def _end_labels(ax_map):
    return [t for t in ax_map.texts if t.get_text() in ('A', 'B')]


def test_the_transect_labels_stay_inside_the_map():
    """A label outside the axes lands on the title; both ends stay inside."""
    fig, (ax_map, _, _), r = _corner_overview(['GEBCO 2024 Grid'])
    labels = _end_labels(ax_map)
    assert len(labels) == 2
    for label in labels:
        bb = label.get_window_extent(r)
        assert ax_map.bbox.contains(bb.x0, bb.y0), label.get_text()
        assert ax_map.bbox.contains(bb.x1, bb.y1), label.get_text()
    plt.close(fig)


def test_the_legend_leaves_the_transect_ends_clear():
    """Neither end marker (padded by its own radius) nor end label sits under
    the legend box."""
    fig, (ax_map, _, _), r = _corner_overview(['GEBCO 2024 Grid'])
    legend = ax_map.get_legend().get_window_extent(r).padded(8.0)
    line = next(ln for ln in ax_map.lines if ln.get_label() == 'transect')
    for x, y in ax_map.transData.transform(np.column_stack(line.get_data())):
        assert not legend.contains(x, y), (x, y)
    for label in _end_labels(ax_map):
        assert not legend.overlaps(label.get_window_extent(r)), label.get_text()
    plt.close(fig)


@pytest.mark.parametrize("credits", [
    ['GEBCO 2024 Grid'],
    ['GEBCO 2024 Grid', 'World Ocean Atlas 2023, NOAA NCEI'],
])
def test_the_credit_clears_every_axis_label(credits):
    """The footnote's top sits below the lowest tick or axis label, for one
    credit line and for two."""
    fig, axes, r = _corner_overview(credits)
    credit = next(t for t in fig.texts if t.get_text().startswith('Data:'))
    top = credit.get_window_extent(r).y1
    for ax in fig.axes:
        assert ax.get_tightbbox(r).y0 >= top, ax.get_title()
    plt.close(fig)


def _overview(**kwargs):
    env = uacpy.Environment(bathymetry=200.0, ssp=1500.0)
    tl = _grid()
    lats, lons = np.linspace(40.0, 41.0, 4), np.linspace(-1.0, 1.0, 5)
    depth = np.full((4, 5), 1000.0)
    return plot_overview(env=env, tl=tl, map_args=(lats, lons, depth),
                         map_kwargs=dict(basemap=False), **kwargs)


@pytest.mark.parametrize("tl_kwargs, expected", [
    (None, 'TL (dB)'),
    (dict(value='mag'), '|p|'),
    (dict(value='phase'), 'Phase (rad)'),
])
def test_the_overview_colorbar_follows_tl_kwargs(tl_kwargs, expected):
    """The label was hardcoded 'TL (dB)', so ``tl_kwargs=dict(value='mag')``
    captioned linear |p| as a loss in dB."""
    fig, (ax_map, ax_tl, ax_env) = _overview(tl_kwargs=tl_kwargs)
    assert [c.get_ylabel() for c in ax_tl.child_axes] == [expected]


def test_a_panel_grid_warning_points_at_the_caller():
    """``typed_plot_error`` wraps every plotter, so a plotter's own
    ``stacklevel`` lands on the decorator: the user was told to fix a call and
    handed a line in uacpy's _common.py, and a ``-W`` filter keyed on their own
    module never matched."""
    other = Field(data=_grid().data,
                  coords={'depth': _DEPTH, 'range': _RANGE + 1.0})
    with pytest.warns(UserWarning, match='axis differs') as record:
        compare_models([_grid(), other])
    assert Path(record[0].filename).name == Path(__file__).name


def test_an_offscreen_psd_warning_points_at_the_caller():
    """Same offset, one frame deeper: the warning is raised by a helper the
    plotter calls."""
    frequencies = np.array([0.0, 10.0, 100.0, 1000.0])
    with pytest.warns(UserWarning, match='outside the plotted y range') as rec:
        plot_psd(frequencies, np.full(4, 1e-30))
    assert Path(rec[0].filename).name == Path(__file__).name


def test_a_sub_hertz_band_keeps_an_ascending_frequency_axis():
    """The low end is clamped to 1 Hz because a log axis cannot render DC. A
    record whose whole band sits below 1 Hz got lo > hi from that clamp, and
    the axis silently reversed."""
    lo, hi = _log_freq_xlim(np.array([0.0, 0.1, 0.2, 0.5]))
    assert 0 < lo < hi

    _, ax = plot_psd(np.array([0.0, 0.1, 0.2, 0.5]), np.full(4, 1.0))
    left, right = ax.get_xlim()
    assert left < right


def test_an_ordinary_band_starts_at_one_hertz():
    assert _log_freq_xlim(np.array([0.0, 10.0, 100.0])) == (1.0, 100.0)


def _flat_spectrogram(level_db, n_f=6, n_t=5, f_lo=1.0, f_hi=500.0):
    """``(frequencies, times, Sxx)`` whose every cell sits at ``level_db``
    re 1 µPa²/Hz, so the whole record is on one side of a colour window."""
    power = (10.0 ** (level_db / 10.0)) * REFERENCE_PRESSURE_WATER ** 2
    return (np.linspace(f_lo, f_hi, n_f), np.linspace(0.0, 10.0, n_t),
            np.full((n_f, n_t), power))


def test_a_sub_hertz_spectrogram_keeps_an_ascending_frequency_axis():
    """``plot_spectrogram``'s ``ymin=1`` default lands above the whole band of
    a record below 1 Hz, which reverses the y axis and puts the record outside
    its own window. Such a band starts at its first positive bin instead."""
    frequencies = np.linspace(0.01, 0.5, 8)
    fig, ax = plot_spectrogram(frequencies, np.linspace(0.0, 600.0, 5),
                               np.full((8, 5), 1e-12))
    lo, hi = ax.get_ylim()
    assert 0 < lo < hi
    assert not ax.yaxis_inverted()
    assert (lo, hi) == pytest.approx((frequencies[0], frequencies[-1]))
    plt.close(fig)


def test_a_spectrogram_band_above_one_hertz_starts_at_the_clamp():
    """The clamp holds wherever the band can take it, so an ordinary panel
    keeps the published ``ymin=1`` window."""
    fig, ax = plot_spectrogram(np.linspace(0.0, 2000.0, 8),
                               np.linspace(0.0, 600.0, 5),
                               np.full((8, 5), 1e-12), ymax=800.0)
    assert ax.get_ylim() == pytest.approx((1.0, 800.0))
    plt.close(fig)


@pytest.mark.parametrize("top, expected_lo", [
    (1.0, 0.25),            # top == clamp: the window would be empty
    (1.0 + 1e-9, 1.0),      # top above the clamp by the smallest step measured
])
def test_the_frequency_clamp_applies_only_above_the_high_limit(top, expected_lo):
    """Both sides of the clamp boundary: at ``hi == clamp`` the clamped window
    has zero height, so the record's own first positive bin is the low end."""
    frequencies = np.array([0.0, 0.25, 0.5, top])
    assert _clamped_freq_limits(frequencies, 1.0, top)[0] == pytest.approx(expected_lo)


def test_a_spectrogram_outside_the_pinned_colour_window_says_it_is_flat():
    """Both spectrograms pin a 0-200 dB colour window, so a record wholly above
    or below it maps to one end of the colormap in every cell — a flat image
    that reads as a valid featureless record."""
    for plotter, caller in ((plot_spectrogram, "plot_spectrogram"),
                            (plot_constant_q_spectrogram,
                             "plot_constant_q_spectrogram")):
        for level_db in (240.0, -80.0):
            with pytest.warns(UserWarning,
                              match=f"{caller}: every sample.*colour window"):
                fig, _ = plotter(*_flat_spectrogram(level_db))
            plt.close(fig)


@pytest.mark.parametrize("level_db", [0.0, 200.0])
def test_a_spectrogram_on_the_colour_window_edge_is_not_flagged(level_db):
    """Both edges of the window are inside it — the warning fires only when
    every sample is strictly outside, or an ordinary panel trains the reader
    to ignore it."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fig, _ = plot_spectrogram(*_flat_spectrogram(level_db))
        plt.close(fig)


@pytest.mark.parametrize("kwargs, expected", [
    ({}, (1.0, 500.0)),                        # the published default
    ({'ymin': None}, (0.0, 500.0)),            # low end from the record
    ({'ymax': None}, (1.0, 500.0)),            # high end from the record
    ({'ymin': None, 'ymax': None}, (0.0, 500.0)),
    ({'ymin': None, 'ymax': 300.0}, (0.0, 300.0)),
    ({'ymin': 50.0, 'ymax': None}, (50.0, 500.0)),
])
def test_either_frequency_limit_takes_the_records_own_end_when_none(kwargs,
                                                                    expected):
    """``ymin`` and ``ymax`` are symmetric: ``None`` on either takes the
    record's own first / last bin. ``ymax=None`` is the documented spelling and
    ``ymin=None`` is its mirror — reaching the clamp comparison with ``None``
    raises a bare ``TypeError``, which ``typed_plot_error`` does not catch."""
    frequencies = np.linspace(0.0, 500.0, 24)
    fig, ax = plot_spectrogram(frequencies, np.linspace(0.0, 10.0, 12),
                               np.full((24, 12), 1e-12), **kwargs)
    assert ax.get_ylim() == pytest.approx(expected)
    assert not ax.yaxis_inverted()
    plt.close(fig)


def test_a_spectrogram_pinned_off_its_own_band_says_the_panel_is_empty():
    """A caller pinning both ends of the y window away from the record gets an
    empty panel that the clamp rule cannot rescue, so it is named."""
    with pytest.warns(UserWarning,
                      match=r"plot_spectrogram: every sample.*plotted y range"):
        fig, _ = plot_spectrogram(*_flat_spectrogram(140.0),
                                  ymin=800.0, ymax=900.0)
    plt.close(fig)


def test_a_precomputed_absorption_array_rejects_the_model_knobs():
    from uacpy.visualization.plots.environment import plot_absorption
    frequencies = np.array([100.0, 1000.0, 10000.0])
    absorption = np.array([0.001, 0.06, 1.0])
    with pytest.raises(ConfigurationError,
                       match='has no effect on a pre-computed'):
        plot_absorption(frequencies, absorption, model='thorp')
    with pytest.raises(ConfigurationError,
                       match='has no effect on a pre-computed'):
        plot_absorption(frequencies, absorption,
                        model_kwargs={'temperature': 10.0})


def test_a_precomputed_absorption_array_alone_plots():
    from uacpy.visualization.plots.environment import plot_absorption
    frequencies = np.array([100.0, 1000.0, 10000.0])
    absorption = np.array([0.001, 0.06, 1.0])
    _fig, ax = plot_absorption(frequencies, absorption)
    assert ax.get_ylabel() == 'Absorption (dB/km)'


def test_a_source_depth_heatmap_plots_positive_down_like_depth():
    from uacpy.core.results.field import Field
    from uacpy.visualization.plots.fields import plot_field
    depths = np.array([5.0, 20.0, 50.0])
    ranges = np.linspace(100.0, 5000.0, 12)
    data = np.arange(depths.size * ranges.size,
                     dtype=float).reshape(depths.size, ranges.size)
    inverted = {}
    for axis in ('source_depth', 'depth'):
        _fig, ax = plot_field(Field(data=data,
                                    coords={axis: depths, 'range': ranges}))
        inverted[axis] = ax.yaxis_inverted()
    assert inverted == {'source_depth': True, 'depth': True}


# ─────────────────────────────────────────────────────────────────────────────
# plot_beam_pattern — the .sbp source-directivity table
#
# The orientation tests are the load-bearing ones: the polar axes must place a
# positive angle where the *field* plots put it, because the .sbp angle axis is
# Bellhop's launch declination ``alpha`` and ``alpha > 0`` launches downward.
# ``test_bellhop.py`` pins that engine convention itself; these pin that the
# plot follows it.
# ─────────────────────────────────────────────────────────────────────────────


def _beam_full(half_width=15.0, floor=-30.0):
    """A ±``half_width``° main lobe at 0 dB, ``floor`` dB outside."""
    angles = np.linspace(-180.0, 180.0, 361)
    levels = np.where(np.abs(angles) <= half_width, 0.0, floor)
    return np.column_stack([angles, levels])


def _beam_half():
    """A 0-180° table — support that the forward view clips to a quarter."""
    angles = np.linspace(0.0, 180.0, 181)
    return np.column_stack([angles,
                            np.where(np.abs(angles - 30.0) <= 12.0, 0.0, -35.0)])


# ── the chart is covered to its own edges ────────────────────────────────────

def _rays_to_receiver():
    from uacpy.core.results import Rays
    # Carries a SOURCE as well: the source marker is the highest-z-order
    # artist these plots draw, so a fixture without one cannot detect a
    # legend that sits below it.
    return Rays(
        rays=[{'r': np.linspace(0, 1000, 20),
               'z': np.linspace(10, 90, 20),
               'alpha': 0.0, 'n_top_bounces': 0, 'n_bot_bounces': 0}],
        receiver_depths=np.array([50.0]),
        receiver_ranges=np.array([1000.0]),
        source_depths=np.array([10.0]),
        model='Bellhop',
    )


def _fill_extent(ax):
    xs = [np.vstack([pth.vertices for pth in c.get_paths()])[:, 0]
          for c in ax.collections if c.get_paths()]
    return (min(x.min() for x in xs), max(x.max() for x in xs)) if xs else None


def test_the_seafloor_fill_reaches_the_right_spine(env):
    """The fill is anchored on the receiver extent while the x-limit is set
    a margin wider, so the seabed stopped short of the spine and left a bare
    strip under the rays past the receiver — the sliver the anchoring exists
    to avoid."""
    fig, ax = _rays_to_receiver().plot(env=env)
    lo, hi = _fill_extent(ax)
    x_lo, x_hi = ax.get_xlim()
    assert hi >= x_hi - 1e-9, f"fill stops at {hi:g} km, axis ends at {x_hi:g}"
    assert lo <= x_lo + 1e-9
    plt.close(fig)


def test_the_receiver_margin_keeps_the_marker_on_the_axis(env):
    """The margin the fill is anchored to exists to keep a receiver at the
    maximum range off the spine. Without pinning that, the margin could be
    set to anything — including a value that clips the receiver away — and
    the fill tests would still pass, since they only compare the fill against
    whatever limit the margin produced."""
    fig, ax = _rays_to_receiver().plot(env=env)
    rr_km = 1.0
    assert ax.get_xlim()[1] > rr_km, "the receiver sits on or past the spine"
    plt.close(fig)


def test_the_legend_does_not_cover_the_receiver(env):
    """Raising the legend over the geometry is only half the job: the
    receiver sits at the maximum range, which is where a lower-right legend
    lands, so a fixed corner hides the marker it is meant to explain."""
    # A bottom-mounted receiver: at the maximum range AND near the seabed,
    # which is the corner a lower-right legend occupies.
    from uacpy.core.results import Rays
    # All three bounce classes, so the legend is the size a real eigenray
    # plot gives it — a one-entry key is too small to reach the corner.
    rays = Rays(
        rays=[{'r': np.linspace(0, 1000, 20),
               'z': np.linspace(95, 99, 20), 'alpha': float(a),
               'n_top_bounces': top, 'n_bot_bounces': bot}
              for a, top, bot in ((0.0, 0, 0), (1.0, 1, 0), (2.0, 1, 1))],
        receiver_depths=np.array([99.0]),
        receiver_ranges=np.array([1000.0]),
        model='Bellhop')
    fig, ax = rays.plot(env=env)
    fig.canvas.draw()
    legend_box = ax.get_legend().get_window_extent()
    rr, rd = 1.0, 99.0                      # the receiver, in data coords
    x, y = ax.transData.transform((rr, rd))
    assert not (legend_box.x0 <= x <= legend_box.x1
                and legend_box.y0 <= y <= legend_box.y1), \
        "the legend covers the receiver marker"
    plt.close(fig)


def test_the_legend_sits_above_the_geometry(env):
    """Legends default to zorder 5, below this package's own ladder — the
    seabed fill, the seafloor line, the receivers and the source all draw
    over it, so the key ends up underneath the picture it explains."""
    fig, ax = _rays_to_receiver().plot(env=env)
    legend = ax.get_legend()
    assert legend is not None
    over = [type(a).__name__ for a in ax.get_children()
            if hasattr(a, 'get_zorder')
            and a.get_zorder() > legend.get_zorder()]
    assert not over, f"drawn over the legend: {sorted(set(over))}"
    plt.close(fig)


# ── the drawn curve is the table the engine reads ────────────────────────────

def test_a_sparsely_sampled_table_is_drawn_along_its_angles():
    """``interp1`` (misc/interpolation.f90:37-39) reads the table linearly in
    ANGLE, so a two-row 0 dB table is omnidirectional. Handing those two
    samples to a polar axes draws a straight screen chord between them —
    through the origin — which renders an omni source as a deep broadside
    null. The curve has to be sampled along the angle axis instead."""
    omni = np.array([[-90.0, 0.0], [90.0, 0.0]])
    fig, ax = plot_beam_pattern(omni)
    r = np.asarray(ax.lines[0].get_ydata())
    assert r.size > 2, "the pattern was drawn as a chord between two samples"
    np.testing.assert_allclose(r, 0.0, atol=1e-9)
    plt.close(fig)


def test_a_coarse_table_keeps_its_corners_when_densified():
    """The previous check used a 1-degree table, which comes back byte
    identical — it never exercised the resampling at all. A coarse table with
    a real corner does: every tabulated node must keep its own level, and the
    inserted points must interpolate linearly between them rather than round
    the corner off."""
    table = np.array([[-90.0, -30.0], [-20.0, 0.0],
                      [20.0, 0.0], [90.0, -30.0]])
    fig, ax = plot_beam_pattern(table)
    theta = np.degrees(np.asarray(ax.lines[0].get_xdata()))
    level = np.asarray(ax.lines[0].get_ydata())
    assert theta.size > table.shape[0], "a coarse table was not resampled"
    for angle, expected in table:
        assert level[np.argmin(np.abs(theta - angle))] == pytest.approx(
            expected), f"node {angle}deg lost its level"
    # Between nodes the curve must follow the engine, which interpolates the
    # table AFTER converting it to amplitude (``10**(dB/20)``,
    # misc/beampattern.f90:59; Bellhop does it inline at
    # bellhop.f90:267-274). Interpolating in dB instead bows the flank the
    # wrong way — 14.7 dB out at worst on a 0/-35 dB segment.
    mid = level[np.argmin(np.abs(theta - (-55.0)))]
    amp = 0.5 * (10 ** (-30.0 / 20.0) + 10 ** (0.0 / 20.0))
    assert mid == pytest.approx(20.0 * np.log10(amp), abs=0.05)
    plt.close(fig)


def test_both_renderings_draw_the_same_curve():
    """Whichever axes it is drawn on, the reader is looking at one pattern.
    The interpolation is the engine's reading of the table, so it belongs to
    the curve rather than to either axes: applied to one branch only, the two
    views disagreed by more than 11 dB between the tabulated angles."""
    table = np.array([[-90.0, -30.0], [-20.0, 0.0], [20.0, 0.0], [90.0, -30.0]])
    fig_p, ax_p = plot_beam_pattern(table)
    fig_c, ax_c = plot_beam_pattern(table, polar=False)
    angles_p = np.degrees(np.asarray(ax_p.lines[0].get_xdata()))
    angles_c = np.asarray(ax_c.lines[0].get_xdata())
    assert np.allclose(angles_p, angles_c), "the views sample different angles"
    assert np.allclose(ax_p.lines[0].get_ydata(), ax_c.lines[0].get_ydata())
    plt.close(fig_p)
    plt.close(fig_c)


def test_a_degenerate_table_is_refused_on_rectilinear_axes_too():
    """A table that spans no angle cannot be drawn as a pattern in either
    view, so the refusal belongs to the curve, not to the polar axes."""
    with pytest.raises(ConfigurationError, match='spans no angle'):
        plot_beam_pattern(np.array([[0.0, -3.0], [0.0, -40.0]]), polar=False)


def test_a_notch_between_display_samples_keeps_its_depth():
    """Resampling must not round a corner off. A notch tabulated between two
    display samples is the case that would be lost by resampling onto a plain
    grid: the tabulated angles are carried into the grid so the notch keeps
    the depth the table gave it, rather than the shallower value a straight
    line across it would draw."""
    table = np.array([[-90.0, 0.0], [15.4, 0.0], [15.5, -40.0],
                      [15.6, 0.0], [90.0, 0.0]])
    fig, ax = plot_beam_pattern(table)
    drawn = np.degrees(np.asarray(ax.lines[0].get_xdata()))
    levels = np.asarray(ax.lines[0].get_ydata())
    notch = levels[np.isclose(drawn, 15.5, atol=1e-9)]
    assert notch.size == 1, "the tabulated notch angle was dropped"
    assert np.isclose(notch[0], -40.0), f"notch drawn at {notch[0]:.1f} dB"
    plt.close(fig)


def test_a_table_finer_than_the_display_step_keeps_every_row():
    """The rows a caller supplied must survive: a table sampled finer than
    the display step keeps all of them and gains only points between."""
    angles = np.linspace(-90.0, 90.0, 361)          # 0.5 deg
    table = np.column_stack([angles, -20.0 * np.abs(angles) / 90.0])
    fig, ax = plot_beam_pattern(table)
    drawn = np.degrees(np.asarray(ax.lines[0].get_xdata()))
    for angle in angles:
        assert np.isclose(drawn, angle, atol=1e-9).any(), f"{angle} lost"
    plt.close(fig)


# ── shape of the returned axes ───────────────────────────────────────────────

def test_returns_polar_axes_by_default():
    fig, ax = plot_beam_pattern(_beam_full())
    assert isinstance(fig, plt.Figure)
    assert ax.name == 'polar'


def test_polar_false_returns_rectilinear_axes():
    fig, ax = plot_beam_pattern(_beam_full(), polar=False)
    assert ax.name == 'rectilinear'


# ── orientation: must match the field's axes ─────────────────────────────────

def test_positive_angle_points_downward_like_the_field():
    """+90° draws straight down, because +alpha launches downward in Bellhop
    and the field plots depth positive downward."""
    fig, ax = plot_beam_pattern(_beam_full())
    r = ax.get_rmax()
    down = ax.transData.transform((np.pi / 2, r))
    up = ax.transData.transform((-np.pi / 2, r))
    # Display y grows upward, so "down" must sit lower on the canvas.
    assert down[1] < up[1]


def test_zero_degrees_points_along_increasing_range():
    fig, ax = plot_beam_pattern(_beam_full())
    r = ax.get_rmax()
    forward = ax.transData.transform((0.0, r))
    down = ax.transData.transform((np.pi / 2, r))
    up = ax.transData.transform((-np.pi / 2, r))
    assert forward[0] > down[0] and forward[0] > up[0]


def test_cartesian_puts_angle_on_x_and_level_on_y():
    fig, ax = plot_beam_pattern(_beam_full(), polar=False)
    assert 'ngle' in ax.get_xlabel()
    assert 'dB' in ax.get_ylabel()
    x, y = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
    assert np.allclose(x, _beam_full()[:, 0])
    assert np.allclose(y, _beam_full()[:, 1])


# ── the angle axis is labelled in the table's own convention ─────────────────

def test_theta_labels_are_signed_like_the_sbp_table():
    """The upper half reads -90, not 270: a .sbp angle column and a Bellhop
    alpha fan are both spelled signed, so the plot must be too."""
    fig, ax = plot_beam_pattern(_beam_full())
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert any(l.startswith('-90') for l in labels)
    assert any(l.startswith('90') for l in labels)
    assert not any('270' in l for l in labels)


def test_signed_labels_do_not_widen_the_drawn_wedge():
    """Signed tick *labels* must not become signed tick *positions*: a
    negative position drags thetamin to -180, and matplotlib then sweeps 540
    degrees as a distorted part-disc with the pattern squeezed out of it."""
    fig, ax = plot_beam_pattern(_beam_full())
    assert ax.get_thetamax() - ax.get_thetamin() == 180.0


def test_theta_labels_stay_within_plus_minus_180():
    fig, ax = plot_beam_pattern(_beam_full())
    shown = [float(t.get_text().rstrip('\u00b0')) for t in ax.get_xticklabels()
             if t.get_text()]
    assert min(shown) >= -180.0 and max(shown) <= 180.0


# ── the lobe reads as a lobe ─────────────────────────────────────────────────

def test_polar_lobe_is_filled_by_default():
    """A top-hat pattern whose sidelobes sit at the radial floor draws as bare
    radial spokes without a fill."""
    fig, ax = plot_beam_pattern(_beam_full())
    assert len(ax.collections) >= 1


def test_fill_false_leaves_the_outline_alone():
    fig, ax = plot_beam_pattern(_beam_full(), fill=False)
    assert len(ax.collections) == 0
    assert len(ax.lines) == 1


# ── the drawn sector ─────────────────────────────────────────────────────────

def test_the_axes_span_the_launch_fan():
    """Only |alpha| <= 90 propagates into r > 0 — a steeper launch traces to
    negative range and never enters the field — so that is the sector, for
    every table. Fixed rather than fitted to the table, so two patterns drawn
    side by side are drawn on the same axes and can be compared by eye."""
    fig, ax = plot_beam_pattern(_beam_full())
    assert (ax.get_thetamin(), ax.get_thetamax()) == (-90.0, 90.0)


def test_a_table_reaching_past_the_fan_draws_only_the_fan():
    """A 0-180° table is not a reason to draw a sector half of which no
    launch reaches."""
    angles = np.linspace(0.0, 180.0, 181)
    pattern = np.column_stack([angles, np.zeros_like(angles)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        fig, ax = plot_beam_pattern(pattern)
    assert (ax.get_thetamin(), ax.get_thetamax()) == (-90.0, 90.0)


def test_clipping_to_the_fan_does_not_clip_the_drawn_data():
    """The wedge is an axes limit, not a filter: mirror= and the table tests
    keep seeing the whole table."""
    pattern = _beam_full()
    fig, ax = plot_beam_pattern(pattern)
    assert np.allclose(np.rad2deg(ax.lines[0].get_xdata()), pattern[:, 0])


def test_a_peak_outside_the_fan_warns():
    """A back lobe stronger than anything drawn is the one case where the
    launch fan misleads — and it is also a table Bellhop could not use."""
    angles = np.linspace(-180.0, 180.0, 361)
    levels = np.where(np.abs(angles) >= 150.0, 0.0, -30.0)
    with pytest.warns(UserWarning, match='strongest'):
        plot_beam_pattern(np.column_stack([angles, levels]))


def test_a_peak_outside_the_fan_warns_on_the_cartesian_axes_too():
    """The fan clips the angle axis in both renderings, so a main lobe the
    xlim hides has to be reported exactly as the polar wedge reports it."""
    angles = np.linspace(-180.0, 180.0, 361)
    levels = np.where(np.abs(angles) >= 150.0, 0.0, -30.0)
    with pytest.warns(UserWarning, match='strongest'):
        plot_beam_pattern(np.column_stack([angles, levels]), polar=False)


def test_a_peak_inside_the_fan_draws_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        plot_beam_pattern(_beam_full())


def _beam_downward():
    """A 0-90° table — a downward-looking projector, defined only where a
    launch angle propagates."""
    angles = np.linspace(0.0, 90.0, 181)
    return np.column_stack([angles,
                            np.where(np.abs(angles - 30.0) <= 12.0, 0.0, -35.0)])


def test_the_radial_axis_is_not_over_ticked():
    """A polar radius is short and its labels all sit on one spoke, so the
    ~9 ticks a linear dB axis defaults to overprint each other."""
    fig, ax = plot_beam_pattern(_beam_full())
    lo, hi = ax.get_ylim()
    on_axis = [t for t in ax.get_yticks() if lo - 1e-9 <= t <= hi + 1e-9]
    assert len(on_axis) <= 6, f"{len(on_axis)} radial ticks: {on_axis}"


def test_radial_labels_do_not_collide_with_the_title():
    """Measured on the rendered figure, not on ``get_rlabel_position``: once
    the axes is a wedge, matplotlib pins the radial labels to the thetamin
    spoke and ignores that setting entirely, so only the drawn geometry can
    say whether the labels clear the title."""
    for pattern in (_beam_full(), _beam_half()):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            fig, ax = plot_beam_pattern(pattern)
        fig.canvas.draw()
        title = ax.title.get_window_extent()
        for label in ax.yaxis.get_ticklabels():
            box = label.get_window_extent()
            if np.isnan(box.x0) or not label.get_text():
                continue
            assert not title.overlaps(box), (
                f"radial label {label.get_text()!r} overlaps the title")


def test_both_ends_of_the_fan_are_labelled():
    """+/-90 are the two angles a reader checks first — whether the table
    reaches the horizontal and whether it reaches straight down — and a 45°
    step across 180° labels neither."""
    fig, ax = plot_beam_pattern(_beam_full())
    shown = {t.get_text().rstrip('\u00b0') for t in ax.get_xticklabels()
             if t.get_text()}
    assert {'-90', '90'} <= shown, shown


def test_the_cartesian_angle_axis_spans_the_same_fan():
    """Which launch angles exist is a fact about the fan, not about how the
    response is drawn, so both renderings limit the axis the same way."""
    fig, ax = plot_beam_pattern(_beam_full(), polar=False)
    assert ax.get_xlim() == (-90.0, 90.0)


# ── the data actually drawn ──────────────────────────────────────────────────

def test_polar_theta_is_the_table_angle_in_radians():
    pattern = _beam_full()
    fig, ax = plot_beam_pattern(pattern)
    theta = ax.lines[0].get_xdata()
    assert np.allclose(theta, np.deg2rad(pattern[:, 0]))


def test_peak_of_the_main_lobe_reaches_the_radial_maximum():
    fig, ax = plot_beam_pattern(_beam_full())
    levels = ax.lines[0].get_ydata()
    assert np.isclose(levels.max(), 0.0)
    assert np.isclose(levels.min(), -30.0)


# ── omnidirectional ──────────────────────────────────────────────────────────

def test_none_draws_a_flat_zero_db_circle():
    fig, ax = plot_beam_pattern(None)
    levels = ax.lines[0].get_ydata()
    assert np.allclose(levels, 0.0)
    theta = np.rad2deg(ax.lines[0].get_xdata())
    assert np.isclose(theta.min(), -180.0)
    assert np.isclose(theta.max(), 180.0)


def test_none_says_omnidirectional_in_the_title():
    fig, ax = plot_beam_pattern(None)
    assert 'mnidirectional' in ax.get_title()


# ── a half-defined table is not silently completed ───────────────────────────

def test_half_defined_pattern_keeps_its_own_support():
    """0-180° stays 0-180°: Bellhop's ``ReadPat`` (``misc/beampattern.f90``)
    mirrors nothing, so neither does the plot."""
    angles = np.linspace(0.0, 180.0, 181)
    pattern = np.column_stack([angles, np.where(angles <= 15.0, 0.0, -30.0)])
    fig, ax = plot_beam_pattern(pattern)
    theta = np.rad2deg(ax.lines[0].get_xdata())
    assert np.isclose(theta.min(), 0.0)
    assert np.isclose(theta.max(), 180.0)


def test_half_defined_pattern_warns_that_it_leaves_the_fan_uncovered():
    angles = np.linspace(0.0, 180.0, 181)
    pattern = np.column_stack([angles, np.zeros_like(angles)])
    with pytest.warns(UserWarning, match='does not cover'):
        plot_beam_pattern(pattern)


def test_mirror_completes_a_half_defined_pattern():
    angles = np.linspace(0.0, 180.0, 181)
    levels = np.where(angles <= 15.0, 0.0, -30.0)
    fig, ax = plot_beam_pattern(np.column_stack([angles, levels]), mirror=True)
    theta = np.rad2deg(ax.lines[0].get_xdata())
    assert np.isclose(theta.min(), -180.0)
    assert np.isclose(theta.max(), 180.0)


def test_mirror_reflects_the_level_about_zero_degrees():
    angles = np.linspace(0.0, 180.0, 181)
    levels = np.where(angles <= 15.0, 0.0, -30.0)
    fig, ax = plot_beam_pattern(np.column_stack([angles, levels]), mirror=True)
    theta = np.rad2deg(ax.lines[0].get_xdata())
    drawn = ax.lines[0].get_ydata()
    for probe in (10.0, 40.0, 120.0):
        at_plus = drawn[np.argmin(np.abs(theta - probe))]
        at_minus = drawn[np.argmin(np.abs(theta + probe))]
        assert np.isclose(at_plus, at_minus)


def test_mirror_does_not_warn_about_the_gap_it_filled():
    angles = np.linspace(0.0, 180.0, 181)
    pattern = np.column_stack([angles, np.zeros_like(angles)])
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        plot_beam_pattern(pattern, mirror=True)


def test_mirror_warns_when_the_reflection_falls_short_of_the_fan():
    """0-45° mirrors to +/-45°, still short of the +/-90° a launch fan can
    reach, so mirroring does not retire the warning."""
    angles = np.linspace(0.0, 45.0, 46)
    pattern = np.column_stack([angles, np.zeros_like(angles)])
    with pytest.warns(UserWarning, match='does not cover'):
        fig, ax = plot_beam_pattern(pattern, mirror=True)
    theta = np.rad2deg(ax.lines[0].get_xdata())
    assert np.isclose(theta.min(), -45.0)
    assert np.isclose(theta.max(), 45.0)


def test_mirror_to_the_full_fan_does_not_warn():
    """+/-90° covers every launch angle that propagates, so it is complete."""
    angles = np.linspace(0.0, 90.0, 91)
    pattern = np.column_stack([angles, np.zeros_like(angles)])
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        plot_beam_pattern(pattern, mirror=True)


def test_mirror_leaves_a_full_pattern_untouched():
    pattern = _beam_full()
    fig, ax = plot_beam_pattern(pattern, mirror=True)
    theta = np.rad2deg(ax.lines[0].get_xdata())
    assert np.allclose(theta, pattern[:, 0])


# ── reading a .sbp path ──────────────────────────────────────────────────────

def test_accepts_a_sbp_path(tmp_path):
    from uacpy.io import write_source_beam_pattern
    pattern = _beam_full()
    sbp = tmp_path / 'src.sbp'
    write_source_beam_pattern(sbp, pattern[:, 0], pattern[:, 1])
    fig, ax = plot_beam_pattern(sbp)
    assert np.allclose(np.rad2deg(ax.lines[0].get_xdata()), pattern[:, 0])


# ── degenerate input ─────────────────────────────────────────────────────────

def test_empty_table_raises_configuration_error():
    with pytest.raises(ConfigurationError):
        plot_beam_pattern(np.empty((0, 2)))


def test_wrong_width_raises_configuration_error():
    with pytest.raises(ConfigurationError):
        plot_beam_pattern(np.zeros((10, 3)))


def test_rejected_call_leaves_no_figure_behind():
    before = set(plt.get_fignums())
    with pytest.raises(ConfigurationError):
        plot_beam_pattern(np.empty((0, 2)))
    assert set(plt.get_fignums()) == before


class TestThePlottersRefuseAndLabelWhatTheAuditFound:
    """Pins for the visualization findings: the SSP heatmap interpolates like
    the models, a probability defaults to its linear view, empty input and
    mismatched grids are refused with typed errors, one field draws alike
    through either signal-excess door, and snapshot panels keep a readable
    height."""

    def _rgb_at(self, fig, ax, r_km, depth_m):
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        x, y = ax.transData.transform((r_km, depth_m))
        h = buf.shape[0]
        return tuple(int(v) for v in buf[int(round(h - y)), int(round(x)), :3])

    def test_a_two_sample_profile_is_painted_as_a_gradient_not_two_blocks(self):
        # c(0) = 1500, c(100) = 1520: linear between the two samples, as the
        # models read it. 'nearest' painted one flat block per sample, so 25 m
        # and 45 m shared a colour and 55 m jumped.
        env = uacpy.Environment(name='ssp', bathymetry=100.0,
                                ssp=[(0.0, 1500.0), (100.0, 1520.0)])
        fig, ax = env.plot()[:2] if isinstance(env.plot(), tuple) else (None, None)
        out = env.plot()
        fig = out[0] if isinstance(out, tuple) else out
        ax = next(a for a in fig.axes if a.get_ylabel().startswith('Depth'))
        r = 0.5 * float(ax.get_xlim()[1])
        a, b, c = (self._rgb_at(fig, ax, r, z) for z in (25.0, 45.0, 75.0))
        assert a != b, (a, b)                    # varies inside the old block
        assert b != c, (b, c)
        plt.close('all')

    def test_a_probability_field_defaults_to_its_linear_view(self):
        """A real, dimensionless field has no dB view to fall back on, so the
        default view is the raw samples."""
        from uacpy.visualization.plots._common import _default_value

        class Prob:
            coords = {'depth': np.array([1.0]), 'range': np.array([1.0])}
            is_complex = False
            unit = '1'
            kind = 'probability_of_detection'

        assert _default_value(Prob()) == 'real'

    @pytest.mark.parametrize('kind, unit, expected', [
        ('probability_of_detection', '1', 'Probability of detection'),
        ('signal_excess', 'dB', 'Signal excess (dB)'),
    ])
    def test_a_linear_view_is_labelled_from_the_registry(self, kind, unit,
                                                         expected):
        """The label comes from quantities.py, the same source the dB view
        reads. Spelling one out of the ``kind`` tag instead gave a lowercase
        'probability of detection' next to the registry's own capitalised
        colorbar on the dedicated plotter."""
        from uacpy.visualization.plots._common import _value_label

        class Q:
            coords = {'depth': np.array([1.0]), 'range': np.array([1.0])}
            is_complex = False

        q = Q(); q.unit = unit; q.kind = kind
        assert _value_label(q, 'real') == expected

    def test_empty_input_is_refused_by_name(self):
        from uacpy.visualization.plots.comms import plot_constellation
        from uacpy.visualization.plots.signal import plot_cepstrum
        from uacpy.visualization.plots.noise import plot_source_level
        with pytest.raises(ConfigurationError, match='constellation is empty'):
            plot_constellation(np.array([]))
        with pytest.raises(ConfigurationError, match='c is empty'):
            plot_cepstrum(np.array([]))
        with pytest.raises(ConfigurationError, match='level_db is empty'):
            plot_source_level(np.array([100.0]), np.array([]))

    def test_a_mismatched_spectrogram_grid_is_a_typed_error(self):
        from uacpy.visualization.plots.signal import plot_spectrogram
        with pytest.raises(ConfigurationError, match='does not match'):
            plot_spectrogram(np.linspace(0, 1, 5), np.linspace(0, 1, 7),
                             np.ones((7, 5)))
        plt.close('all')

    def test_a_bare_array_is_not_a_wenz_result(self):
        from uacpy.visualization.plots.noise import plot_wenz
        with pytest.raises(ConfigurationError, match='WenzNoise'):
            plot_wenz(np.ones(10))

    def test_signal_excess_draws_alike_through_either_door(self):
        from uacpy.visualization.plots.fields import plot_signal_excess
        from uacpy.sonar.sonar_equation import passive_signal_excess_field
        d = np.linspace(5, 95, 10); r = np.linspace(100, 5000, 40)
        tl = Field(data=np.broadcast_to(50.0 + 10.0 * np.log10(r)[None, :], (10, 40)).copy(),
                   coords={'depth': d, 'range': r}, model='Synth',
                   frequencies=100.0)
        se = passive_signal_excess_field(tl, source_level=180.0, noise_level=60.0)
        fig_a, ax_a = plot_signal_excess(se)
        fig_b, ax_b = se.plot()
        mesh_a = next(c for c in ax_a.collections if hasattr(c, 'get_clim'))
        mesh_b = next(c for c in ax_b.collections if hasattr(c, 'get_clim'))
        assert mesh_a.get_clim() == mesh_b.get_clim()
        assert ax_a.get_title() == ax_b.get_title()
        plt.close('all')

    @staticmethod
    def _snapshot_aspect(depth_m, range_m, *, n_depth=11):
        from uacpy.visualization.plots.animation import plot_time_snapshots
        d = np.linspace(0, depth_m, n_depth); r = np.linspace(0, range_m, 61)
        t = np.linspace(0, 1, 5)
        f = Field(data=np.random.default_rng(0).normal(size=(n_depth, 61, 5)),
                  coords={'depth': d, 'range': r, 'time': t}, model='Synth',
                  frequencies=100.0)
        fig, axes = plot_time_snapshots({'a': f}, [0.5])
        aspect = np.asarray(axes).ravel()[0].get_aspect()
        plt.close('all')
        return aspect

    def test_shallow_long_snapshots_keep_a_readable_panel_height(self):
        # 100 m over 3 km: isotropic would be a 1:30 sliver; the floor is 4:1.
        assert self._snapshot_aspect(100.0, 3000.0) == pytest.approx(
            0.25 / (100.0 / 3000.0) / 1000.0)

    @pytest.mark.parametrize('depth_m, range_m', [
        (100.0, 3000.0), (50.0, 20000.0), (5.0, 400.0), (1000.0, 60000.0),
    ])
    def test_every_long_panel_lands_on_the_same_4_to_1_floor(self, depth_m,
                                                             range_m):
        """The aspect is in km-per-metre, so the panel's rendered height over
        its width is ``aspect * 1000 * depth_span / range_span``. Whatever the
        spans, a long panel comes out a quarter as tall as it is wide."""
        aspect = self._snapshot_aspect(depth_m, range_m)
        rendered = aspect * 1000.0 * depth_m / range_m
        assert rendered == pytest.approx(0.25)

    def test_a_single_receiver_depth_gets_an_auto_aspect(self):
        """A one-depth time series has no depth span to scale the aspect by.
        Dividing the floor through it raised ZeroDivisionError out of the
        plotter, so the whole figure was lost rather than one row's aspect."""
        assert self._snapshot_aspect(0.0, 3000.0, n_depth=1) == 'auto'

    @pytest.mark.parametrize('depth_m, range_m', [
        (100.0, 900.0), (500.0, 3000.0), (2000.0, 5000.0),
    ])
    def test_a_panel_that_is_not_long_is_left_to_matplotlib(self, depth_m,
                                                            range_m):
        """The floor exists for slivers only; anything squarer keeps 'auto',
        so the panel fills the space the figure gave it."""
        assert self._snapshot_aspect(depth_m, range_m) == 'auto'

    def test_a_uniform_bottom_property_gets_a_relative_window(self):
        from uacpy.visualization.plots.environment import plot_bottom_properties
        env = uacpy.Environment(name='u', bathymetry=100.0, ssp=1500.0, bottom=1650.0)
        fig, axes = plot_bottom_properties(env)
        ax = np.asarray(axes).ravel()[0]
        mesh = next(c for c in ax.collections if hasattr(c, 'get_clim'))
        lo, hi = mesh.get_clim()
        assert lo == pytest.approx(1650.0 * 0.95) and hi == pytest.approx(1650.0 * 1.05)
        plt.close('all')

    def test_the_imshow_extent_is_the_flipped_cell_edge_extent(self):
        from uacpy.visualization.plots._common import _imshow_extent
        r = np.array([1000.0, 2000.0, 3000.0]); z = np.array([10.0, 30.0, 50.0])
        assert _imshow_extent(r, z) == pytest.approx((0.5, 3.5, 60.0, 0.0))


def _water_colorbar(fig):
    """The inset colorbar axes are children of the panel, not of the figure."""
    insets = [child for ax in fig.axes for child in ax.child_axes]
    return next(c for c in insets if c.get_ylabel() == 'Water c (m/s)')


def _bottom(kind):
    from uacpy.core import BoundaryProperties
    return None if kind == 'water only' else BoundaryProperties(
        acoustic_type='half-space', sound_speed=1600.0, density=1.5,
        attenuation=0.5)


@pytest.mark.parametrize("bottom", ['water only', 'half-space'])
@pytest.mark.parametrize("ssp", [
    1500.0,
    uacpy.SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.0, 1500.4)]),
], ids=['constant', 'nearly constant'])
def test_a_near_constant_water_colorbar_prints_absolute_speeds(ssp, bottom):
    """No "+1.5e3" offset over 0.0/0.4/0.8: every tick reads in metres per
    second, on the water-only bar and on the stacked water/bottom pair."""
    env = uacpy.Environment(bathymetry=100.0, ssp=ssp, bottom=_bottom(bottom))
    fig, _ = env.plot()
    fig.canvas.draw()
    cax = _water_colorbar(fig)
    assert cax.yaxis.get_offset_text().get_text() == ''
    labels = [t.get_text() for t in cax.get_yticklabels() if t.get_text()]
    assert labels
    assert all(float(label.replace('\u2212', '-')) > 1000.0 for label in labels)
    plt.close(fig)


@pytest.mark.parametrize("bottom", ['water only', 'half-space'])
def test_a_constant_profile_sits_at_the_centre_of_its_water_colorbar(bottom):
    """The window around a constant value is symmetric, so the one speed the
    water has is the bar's middle tick, not its bottom edge."""
    env = uacpy.Environment(bathymetry=100.0, ssp=1500.0, bottom=_bottom(bottom))
    fig, _ = env.plot()
    lo, hi = _water_colorbar(fig).get_ylim()
    assert lo < 1500.0 < hi
    assert (lo + hi) / 2.0 == pytest.approx(1500.0)
    plt.close(fig)
