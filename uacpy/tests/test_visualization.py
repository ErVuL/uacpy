"""
Tests for visualization module including plots and quickplot
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

import uacpy
from uacpy.models import Bellhop, RAM, Kraken, KrakenField
from uacpy.visualization import plots, quickplot

# Visualization tests spawn Bellhop (and other models) via the basic_setup fixture
pytestmark = pytest.mark.requires_binary


@pytest.fixture
def basic_setup():
    """Build env / source / TL result for one test."""
    env = uacpy.Environment(
        name='test',
        bathymetry=100,
        ssp=1500
    )
    source = uacpy.Source(depths=50, frequencies=100)
    receiver = uacpy.Receiver(
        depths=np.linspace(0, 100, 50),
        ranges=np.linspace(100, 5000, 100),
    )

    bellhop = Bellhop(verbose=False)
    result = bellhop.compute_tl(env, source, receiver)

    return env, source, result


class TestPlotTransmissionLoss:
    """Tests for plot_transmission_loss function."""

    def test_plot_transmission_loss_basic(self, basic_setup):
        """Test basic TL plotting."""
        env, source, result = basic_setup

        fig, ax = plots.plot_transmission_loss(result, env)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    @pytest.mark.parametrize("kwargs", [
        dict(vmin=40, vmax=100, cmap='plasma', figsize=(14, 8)),
        dict(show_bathymetry=False),
    ])
    def test_plot_transmission_loss_kwargs(self, basic_setup, kwargs):
        """Optional kwargs forward through to the underlying axes."""
        env, source, result = basic_setup
        fig, ax = plots.plot_transmission_loss(result, env, **kwargs)
        assert fig is not None
        plt.close(fig)

    def test_plot_transmission_loss_custom_ax(self, basic_setup):
        """plot_transmission_loss respects an externally-supplied Axes."""
        env, source, result = basic_setup
        fig, ax = plt.subplots()
        fig2, ax2 = plots.plot_transmission_loss(result, env, ax=ax)
        assert fig2 is fig
        assert ax2 is ax
        plt.close(fig)


class TestPlotRays:
    """Tests for plot_rays function."""

    def test_plot_rays_basic(self, basic_setup):
        """Test basic ray plotting."""
        env, source, _ = basic_setup

        bellhop = Bellhop(verbose=False)
        rays = bellhop.compute_rays(env, source, uacpy.Receiver(depths=[50], ranges=[5000]))

        fig, ax = plots.plot_rays(rays, env)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_rays_filter_nfirst(self, basic_setup):
        """Filtering before plotting via Rays.filter_nfirst."""
        env, source, _ = basic_setup

        bellhop = Bellhop(verbose=False)
        rays = bellhop.compute_rays(env, source, uacpy.Receiver(depths=[50], ranges=[5000]))

        fig, ax = plots.plot_rays(rays.filter_nfirst(20), env)
        assert fig is not None
        plt.close(fig)


class TestPlotEnvironment:
    """Tests for environment plotting functions."""

    def test_plot_ssp(self, basic_setup):
        """Test SSP plotting."""
        env, source, _ = basic_setup

        fig, ax = plots.plot_ssp(env)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_bathymetry(self, basic_setup):
        """Test bathymetry plotting."""
        env, source, _ = basic_setup

        fig, ax = plots.plot_bathymetry(env)
        assert fig is not None
        plt.close(fig)

    def test_plot_environment(self, basic_setup):
        """Test complete environment plotting."""
        env, source, _ = basic_setup

        fig, axes = plots.plot_environment(env, source)
        assert fig is not None
        assert len(axes) >= 2
        plt.close(fig)


class TestCompareModels:
    """Tests for model comparison functions."""

    def test_compare_models_multiple(self, basic_setup):
        """Test comparing multiple models."""
        env, source, _ = basic_setup
        receiver = uacpy.Receiver(
            depths=np.linspace(0, 100, 50),
            ranges=np.linspace(100, 5000, 100),
        )

        bellhop = Bellhop(verbose=False)
        ram = RAM(verbose=False)

        results = {
            'Bellhop': bellhop.compute_tl(env, source, receiver),
            'RAM': ram.compute_tl(env, source, receiver),
        }

        fig, axes = plots.compare_models(results, env)
        assert fig is not None
        assert len(axes) == 2
        plt.close(fig)


class TestRangeDepthCuts:
    """Tests for range and depth cut plotting."""

    def test_plot_range_cut(self, basic_setup):
        """Test range cut plotting."""
        env, source, result = basic_setup

        fig, ax = plots.plot_range_cut(result, depth=50)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_depth_cut(self, basic_setup):
        """Test depth cut plotting."""
        env, source, result = basic_setup

        fig, ax = plots.plot_depth_cut(result, range_m=2500)
        assert fig is not None
        plt.close(fig)

    def test_compare_range_cuts(self, basic_setup):
        """Test comparing range cuts."""
        env, source, result = basic_setup

        results = {'Bellhop': result}
        fig, ax = plots.compare_range_cuts(results, depth=50)
        assert fig is not None
        plt.close(fig)


class TestFieldPlotMethod:
    """Tests for Result.plot() method."""

    def test_field_plot_tl(self, basic_setup):
        """Test Result.plot() for TL field."""
        env, source, result = basic_setup

        fig, ax = result.plot(env=env)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_field_plot_rays(self, basic_setup):
        """Test Result.plot() for ray field."""
        env, source, _ = basic_setup

        bellhop = Bellhop(verbose=False)
        rays = bellhop.compute_rays(env, source, uacpy.Receiver(depths=[50], ranges=[5000]))

        fig, ax = rays.filter_nfirst(20).plot(env=env)
        assert fig is not None
        plt.close(fig)

    def test_field_plot_modes(self, basic_setup):
        """Test Result.plot() for mode field."""
        env, source, _ = basic_setup

        kraken = Kraken(verbose=False)
        modes = kraken.compute_modes(env, source, n_modes=10)

        fig, axes = modes.plot(n_modes=5)
        assert fig is not None
        plt.close(fig)

    def test_field_plot_custom_params(self, basic_setup):
        """Test Result.plot() with custom parameters."""
        env, source, result = basic_setup

        fig, ax = result.plot(env=env, vmin=30, vmax=90, cmap='jet')
        assert fig is not None
        plt.close(fig)


class TestQuickplot:
    """Tests for quickplot convenience functions."""

    def test_quick_tl(self, basic_setup, tmp_path):
        """Test quick_tl function."""
        env, source, result = basic_setup

        save_path = tmp_path / "test_tl.png"
        fig, ax = quickplot.quick_tl(result, env, save=str(save_path))
        assert fig is not None
        assert save_path.exists()
        plt.close(fig)

    def test_quick_compare(self, basic_setup, tmp_path):
        """Test quick_compare function."""
        env, source, result = basic_setup

        results = {'Bellhop': result}
        save_path = tmp_path / "test_compare.png"
        fig, axes = quickplot.quick_compare(results, env, save=str(save_path))
        assert fig is not None
        assert save_path.exists()
        plt.close(fig)

    def test_quick_env(self, basic_setup, tmp_path):
        """Test quick_env function."""
        env, source, _ = basic_setup

        save_path = tmp_path / "test_env.png"
        fig, axes = quickplot.quick_env(env, source, save=str(save_path))
        assert fig is not None
        assert save_path.exists()
        plt.close(fig)

    def test_quick_cut(self, basic_setup, tmp_path):
        """Test quick_cut function."""
        env, source, result = basic_setup

        save_path = tmp_path / "test_cut.png"
        fig, ax = quickplot.quick_cut(result, depth=50, save=str(save_path))
        assert fig is not None
        assert save_path.exists()
        plt.close(fig)

    def test_quick_analysis(self, basic_setup, tmp_path):
        """Test quick_analysis function."""
        env, source, result = basic_setup

        save_prefix = str(tmp_path / "test")
        fig = quickplot.quick_analysis(result, env, save_prefix=save_prefix)
        assert fig is not None
        assert (tmp_path / "test_analysis.png").exists()
        plt.close(fig)

    def test_quick_report(self, basic_setup, tmp_path):
        """Test quick_report function."""
        env, source, result = basic_setup

        results = {'Bellhop': result}
        save_path = tmp_path / "test_report.png"
        fig = quickplot.quick_report(results, env, save=str(save_path))
        assert fig is not None
        assert save_path.exists()
        plt.close(fig)

class TestPlottingEdgeCases:
    """Tests for edge cases and error handling."""

    def test_plot_empty_results(self):
        """Test plotting with empty results dict."""
        env = uacpy.Environment(name='test', bathymetry=100, ssp=1500)

        with pytest.raises((ValueError, KeyError, IndexError)):
            plots.compare_models({}, env)

    def test_plot_invalid_field_type(self, basic_setup):
        """Test Result.plot() with unsupported field type."""
        env, source, _ = basic_setup

        # Create a field with unsupported type
        # plot_result raises TypeError on a non-Result input.
        from uacpy.visualization.plots import plot_result
        with pytest.raises(TypeError):
            plot_result(object())

    def test_plot_with_nan_data(self, basic_setup):
        """Test plotting with NaN values in data."""
        import copy
        env, source, result = basic_setup

        result_with_nan = copy.deepcopy(result)
        result_with_nan.data[0, 0] = np.nan

        fig, ax = plots.plot_transmission_loss(result_with_nan, env)
        assert fig is not None
        plt.close(fig)


class TestPlottingIntegration:
    """Integration tests for plotting workflows."""

    def test_complete_workflow(self, basic_setup, tmp_path):
        """Combined plot/compare/cuts/report workflow on synthetic PressureFields.

        Real model output is not needed to exercise the plotting pipeline; we
        feed three synthetic PressureFields differing by a constant offset so each
        plotting helper has distinct data to render.
        """
        from uacpy.core.results import PressureField

        env, _, _ = basic_setup
        depths = np.linspace(5, 95, 19)
        ranges = np.linspace(100, 5000, 50)
        base = 60 + 10 * np.log10(np.maximum(ranges, 1.0)[None, :])
        base = np.broadcast_to(base, (depths.size, ranges.size)).copy()

        results = {
            name: PressureField(units="dB", 
                data=base + offset, depths=depths, ranges=ranges, model=name,
            )
            for name, offset in (('Bellhop', 0.0), ('RAM', 1.5), ('KrakenField', -1.5))
        }

        for name, result in results.items():
            fig, ax = result.plot(env=env)
            assert fig is not None
            plt.close(fig)

        fig, axes = plots.compare_models(results, env=env)
        assert fig is not None and len(axes) == 3
        plt.close(fig)

        fig, ax = plots.compare_range_cuts(results, depth=50)
        assert fig is not None
        plt.close(fig)

        save_path = tmp_path / "workflow_report.png"
        fig = quickplot.quick_report(results, env, save=str(save_path))
        assert fig is not None and save_path.exists()
        plt.close(fig)


# ─── Synthetic-input tests (no binary needed) ────────────────────────────


class TestPlotTimeSeriesReceiverRanges:
    """Regression tests for plot_time_series receiver_ranges branch."""

    pytestmark = []

    def test_receiver_ranges_without_receiver_depth(self):
        """``receiver_ranges`` present but no ``receiver_depth`` must not NameError."""
        from uacpy.core.results import TimeSeriesField
        from uacpy.visualization.plots import plot_time_series

        n_d, n_r, n_t = 1, 4, 64
        time = np.linspace(0, 0.5, n_t)
        data = np.zeros((n_d, n_r, n_t))
        for ir in range(n_r):
            data[0, ir] = np.sin(2 * np.pi * 50 * time) * np.exp(-time * 5)

        field = TimeSeriesField(
            data=data,
            depths=np.array([30.0]),
            ranges=np.linspace(100.0, 4000.0, n_r),
            time=time,
            model='SPARC',
            metadata={
                # The buggy branch: receiver_ranges set, receiver_depth absent.
                'receiver_ranges': np.linspace(100.0, 4000.0, n_r),
            },
        )

        fig, ax = plot_time_series(field)
        assert fig is not None and ax is not None
        plt.close(fig)


class TestCompareModelsTypeDispatch:
    """A2: compare_models must dispatch on field type, not call .to_db()."""

    pytestmark = []

    def test_pressure_field_routes_via_to_tl(self):
        from uacpy.core.results import PressureField

        depths = np.linspace(5, 95, 10)
        ranges = np.linspace(100, 5000, 20)
        p = (np.random.default_rng(0).standard_normal((10, 20))
             + 1j * np.random.default_rng(1).standard_normal((10, 20))) * 1e-3

        pfield = PressureField(
            data=p, depths=depths, ranges=ranges,
            model='Bellhop', frequencies=100.0,
        )
        env = uacpy.Environment(name='t', bathymetry=100, ssp=1500)
        fig, axes = plots.compare_models({'Bellhop': pfield}, env=env)
        assert fig is not None
        plt.close(fig)

    def test_unsupported_field_type_raises(self):
        env = uacpy.Environment(name='t', bathymetry=100, ssp=1500)
        with pytest.raises(TypeError, match="expected PressureField"):
            plots.compare_models({'bogus': object()}, env=env)


class TestAutoTLLimits:
    """``_auto_tl_limits`` must drop Bellhop's outside-the-fan TL sentinel
    (~600–740 dB) before computing the median+std auto-scale; otherwise
    vmax explodes off-scale and the visible field saturates at vmin."""

    def test_sentinel_removed_from_vmax(self):
        from uacpy.visualization.plots import _auto_tl_limits
        rng = np.random.default_rng(0)
        body = 50.0 + 10.0 * rng.standard_normal((30, 30))
        sentinels = np.full((10, 10), 600.0)
        data = np.empty((40, 40))
        data[:30, :30] = body
        data[30:, 30:] = sentinels
        data[:30, 30:] = body[:, :10]
        data[30:, :30] = body[:10, :]
        vmin, vmax = _auto_tl_limits(data)
        assert vmax < 200.0
        assert vmin < vmax

    def test_no_finite_falls_back_to_default(self):
        from uacpy.visualization.plots import _auto_tl_limits
        vmin, vmax = _auto_tl_limits(np.full((4, 4), np.nan))
        assert (vmin, vmax) == (30.0, 80.0)
