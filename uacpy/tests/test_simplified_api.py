"""
Tests for the simplified UACPY API
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from uacpy.models import Bellhop, Kraken
from uacpy import Field
from uacpy.core.results import Modes
from uacpy.visualization import plots
from uacpy.models import RunMode

pytestmark = pytest.mark.requires_binary


class TestComputeAPI:
    """Tests for compute_tl(), compute_modes(), etc."""

    def test_compute_tl_with_explicit_receiver(self, simple_env, source, receiver_small):
        """Test compute_tl with explicit receiver."""
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=simple_env, source=source, receiver=receiver_small)

        assert isinstance(result, Field)
        assert result.n_depths == len(receiver_small.depths)
        assert result.n_ranges == len(receiver_small.ranges)

    def test_compute_modes_returns_field(self, simple_env, source):
        """``compute_modes`` returns :class:`Modes`, a sibling of ``Field``
        under ``Result`` rather than a subclass of it — the mode set is not a
        gridded field and carries ``k``/``phi`` instead of ``data``."""
        kraken = Kraken(verbose=False)
        modes = kraken.compute_modes(env=simple_env, source=source, n_modes=10)

        assert isinstance(modes, Modes)
        assert modes.k is not None
        assert modes.phi is not None

    def test_multiple_models_same_api(self, simple_env, source, receiver_small):
        """Test that multiple models use same API."""
        bellhop = Bellhop(verbose=False)
        kraken = Kraken(verbose=False)

        result_bellhop = bellhop.compute_tl(env=simple_env, source=source, receiver=receiver_small)
        result_kraken = kraken.compute_tl(env=simple_env, source=source, receiver=receiver_small)

        assert isinstance(result_bellhop, Field)
        assert isinstance(result_kraken, Field)


class TestPlottingAPI:
    """Tests for result.plot() and plotting functions."""

    def test_field_plot_method_exists(self, simple_env, source, receiver_small):
        """Test that Field has plot() method."""
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=simple_env, source=source, receiver=receiver_small)

        assert hasattr(result, 'plot')
        assert callable(result.plot)

    def test_plot_tl_field(self, simple_env, source, receiver_small):
        """Test plotting TL field."""
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=simple_env, source=source, receiver=receiver_small)

        fig, ax = result.plot(env=simple_env)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_modes(self, simple_env, source):
        """Test plotting modes — Result.plot() routes Modes to plot_mode_functions."""
        kraken = Kraken(verbose=False)
        modes = kraken.compute_modes(env=simple_env, source=source, n_modes=10)

        fig, ax = modes.plot(n_modes=6)

        assert fig is not None
        assert ax is not None
        # ``plot_mode_functions`` returns a single axes carrying overlaid
        # mode shapes; wavenumbers and the heatmap live on the other two
        # canonical mode plotters.
        plt.close(fig)

    def test_plot_with_custom_parameters(self, simple_env, source, receiver_small):
        """Test plotting with custom parameters."""
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=simple_env, source=source, receiver=receiver_small)

        fig, ax = result.plot(env=simple_env, vmin=40, vmax=100, cmap='jet')

        assert fig is not None
        plt.close(fig)

    def test_plot_comparison(self, simple_env, source, receiver_small):
        """``compare_models`` accepts a name → field mapping and uses the keys
        as panel labels, so no separate ``labels=`` is needed."""
        bellhop = Bellhop(verbose=False)
        kraken = Kraken(verbose=False)

        results = {
            'Bellhop': bellhop.compute_tl(env=simple_env, source=source, receiver=receiver_small),
            'Kraken': kraken.compute_tl(env=simple_env, source=source, receiver=receiver_small),
        }

        fig, axes = plots.compare_models(results, env=simple_env)

        assert fig is not None
        assert axes is not None
        plt.close(fig)


class TestFieldMethods:
    """Tests for Field convenience methods."""

    def test_field_get_methods(self, simple_env, source, receiver_small):
        """``Field.at`` drops each pinned axis: pinning both gives a 0-d
        scalar, pinning one leaves a vector along the axis that survives."""
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=simple_env, source=source, receiver=receiver_small)

        point = result.at(range=3000, depth=50)
        assert isinstance(point, Field)
        assert point.tl.ndim == 0
        assert isinstance(float(point.tl), float)

        values_at_range = result.at(range=3000).tl
        assert len(values_at_range) == len(receiver_small.depths)

        values_at_depth = result.at(depth=50).tl
        assert len(values_at_depth) == len(receiver_small.ranges)

    def test_field_properties(self, simple_env, source, receiver_small):
        """Test Field properties."""
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=simple_env, source=source, receiver=receiver_small)

        assert result.n_depths == len(receiver_small.depths)
        assert result.n_ranges == len(receiver_small.ranges)
        assert result.shape == (len(receiver_small.depths), len(receiver_small.ranges))


class TestRunModeAndComputeTl:
    """run(run_mode=...) and compute_tl() should be interchangeable for TL."""

    def test_run_with_coherent_tl_mode(self, simple_env, source, receiver_small):
        """run(run_mode=RunMode.COHERENT_TL) returns a TL field."""
        bellhop = Bellhop(verbose=False)
        result = bellhop.run(env=simple_env, source=source, receiver=receiver_small,
                             run_mode=RunMode.COHERENT_TL)
        assert isinstance(result, Field)
        assert result.shape == (len(receiver_small.depths), len(receiver_small.ranges))

    def test_run_and_compute_tl_agree(self, simple_env, source, receiver_small):
        """compute_tl and run(run_mode=COHERENT_TL) produce the same field."""
        bellhop = Bellhop(verbose=False)
        a = bellhop.run(env=simple_env, source=source, receiver=receiver_small,
                        run_mode=RunMode.COHERENT_TL)
        b = bellhop.compute_tl(env=simple_env, source=source, receiver=receiver_small)
        # The two calls are separate binary runs over separately-written decks,
        # so the tolerance absorbs deck round-tripping (the .env writes depths
        # and speeds at fixed precision) rather than any modelling difference.
        assert np.allclose(a.data, b.data, rtol=1e-3, atol=1e-3)
