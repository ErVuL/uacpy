"""
Tests for visualization module including plots and quickplot
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

import uacpy
from uacpy.models import Bellhop, RAM, Kraken, KrakenField
from uacpy.visualization import plots, quickplot
from uacpy.core.results import Result as Field  # legacy name → typed base

# Visualization tests spawn Bellhop (and other models) via the basic_setup fixture
pytestmark = pytest.mark.requires_binary


@pytest.fixture(scope='module')
def basic_setup():
    """Create basic environment, source, and result for testing.

    Module-scoped: the Bellhop subprocess runs once per file, not once per test.
    Tests that mutate ``result.data`` must operate on a deep copy.
    """
    env = uacpy.Environment(
        name='test',
        depth=100,
        sound_speed=1500,
        ssp_type='isovelocity'
    )
    source = uacpy.Source(depth=50, frequency=100)

    bellhop = Bellhop(verbose=False)
    result = bellhop.compute_tl(env, source, max_range=5000)

    return env, source, result


class TestPlotTransmissionLoss:
    """Tests for plot_transmission_loss function."""

    def test_plot_transmission_loss_basic(self, basic_setup):
        """Test basic TL plotting."""
        env, source, result = basic_setup

        fig, ax, _ = plots.plot_transmission_loss(result, env)
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
        fig, ax, _ = plots.plot_transmission_loss(result, env, **kwargs)
        assert fig is not None
        plt.close(fig)

    def test_plot_transmission_loss_custom_ax(self, basic_setup):
        """plot_transmission_loss respects an externally-supplied Axes."""
        env, source, result = basic_setup
        fig, ax = plt.subplots()
        fig2, ax2, _ = plots.plot_transmission_loss(result, env, ax=ax)
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

    def test_plot_rays_max_rays(self, basic_setup):
        """Test ray plotting with max_rays limit."""
        env, source, _ = basic_setup

        bellhop = Bellhop(verbose=False)
        rays = bellhop.compute_rays(env, source, uacpy.Receiver(depths=[50], ranges=[5000]))

        fig, ax = plots.plot_rays(rays, env, max_rays=20)
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

    def test_compare_models_single(self, basic_setup):
        """Test comparing single model."""
        env, source, result = basic_setup

        results = {'Bellhop': result}
        fig, axes = plots.compare_models(results, env)
        assert fig is not None
        plt.close(fig)

    def test_compare_models_multiple(self, basic_setup):
        """Test comparing multiple models."""
        env, source, _ = basic_setup

        bellhop = Bellhop(verbose=False)
        ram = RAM(verbose=False)

        results = {
            'Bellhop': bellhop.compute_tl(env, source, max_range=5000),
            'RAM': ram.compute_tl(env, source, max_range=5000),
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
    """Tests for Field.plot() method."""

    def test_field_plot_tl(self, basic_setup):
        """Test Field.plot() for TL field."""
        env, source, result = basic_setup

        fig, ax = result.plot(env=env)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_field_plot_rays(self, basic_setup):
        """Test Field.plot() for ray field."""
        env, source, _ = basic_setup

        bellhop = Bellhop(verbose=False)
        rays = bellhop.compute_rays(env, source, uacpy.Receiver(depths=[50], ranges=[5000]))

        fig, ax = rays.plot(env=env, max_rays=20)
        assert fig is not None
        plt.close(fig)

    def test_field_plot_modes(self, basic_setup):
        """Test Field.plot() for mode field."""
        env, source, _ = basic_setup

        kraken = Kraken(verbose=False)
        modes = kraken.compute_modes(env, source, n_modes=10)

        fig, axes = modes.plot(n_modes=5)
        assert fig is not None
        plt.close(fig)

    def test_field_plot_custom_params(self, basic_setup):
        """Test Field.plot() with custom parameters."""
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
        env = uacpy.Environment(name='test', depth=100, sound_speed=1500, ssp_type='isovelocity')

        with pytest.raises((ValueError, KeyError, IndexError)):
            plots.compare_models({}, env)

    def test_plot_invalid_field_type(self, basic_setup):
        """Test Field.plot() with unsupported field type."""
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

        # Mutate a deep copy — basic_setup is module-scoped and shared.
        result_with_nan = copy.deepcopy(result)
        result_with_nan.data[0, 0] = np.nan

        fig, ax, _ = plots.plot_transmission_loss(result_with_nan, env)
        assert fig is not None
        plt.close(fig)


class TestPlottingIntegration:
    """Integration tests for plotting workflows."""

    def test_complete_workflow(self, basic_setup, tmp_path):
        """Combined plot/compare/cuts/report workflow on synthetic TLFields.

        Real model output is not needed to exercise the plotting pipeline; we
        feed three synthetic TLFields differing by a constant offset so each
        plotting helper has distinct data to render.
        """
        from uacpy.core.results import TLField

        env, _, _ = basic_setup
        depths = np.linspace(5, 95, 19)
        ranges = np.linspace(100, 5000, 50)
        base = 60 + 10 * np.log10(np.maximum(ranges, 1.0)[None, :])
        base = np.broadcast_to(base, (depths.size, ranges.size)).copy()

        results = {
            name: TLField(
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
