"""
Tests for the simplified UACPY API
"""

import pytest

pytestmark = pytest.mark.requires_binary
import numpy as np
import matplotlib.pyplot as plt

import uacpy
from uacpy.models import Bellhop, Kraken, KrakenField
from uacpy.core.results import Result, TLField, Modes
from uacpy.visualization import plots
from uacpy.models import RunMode


class TestComputeAPI:
    """Tests for compute_tl(), compute_modes(), etc."""

    def test_compute_tl_with_explicit_receiver(self, simple_env, source, receiver_small):
        """Test compute_tl with explicit receiver."""
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=simple_env, source=source, receiver=receiver_small)

        assert result.field_type == 'tl'
        assert result.n_depths == len(receiver_small.depths)
        assert result.n_ranges == len(receiver_small.ranges)

    def test_compute_modes_returns_field(self, simple_env, source):
        """Test that compute_modes returns Field object."""
        kraken = Kraken(verbose=False)
        modes = kraken.compute_modes(env=simple_env, source=source, n_modes=10)

        assert isinstance(modes, Modes)
        assert modes.field_type == 'modes'
        assert 'k' in modes.metadata
        assert 'phi' in modes.metadata

    def test_multiple_models_same_api(self, simple_env, source, receiver_small):
        """Test that multiple models use same API."""
        bellhop = Bellhop(verbose=False)
        krakenfield = KrakenField(verbose=False)

        result_bellhop = bellhop.compute_tl(env=simple_env, source=source, receiver=receiver_small)
        result_kraken = krakenfield.compute_tl(env=simple_env, source=source, receiver=receiver_small)

        assert isinstance(result_bellhop, TLField)
        assert isinstance(result_kraken, TLField)
        assert result_bellhop.field_type == 'tl'
        assert result_kraken.field_type == 'tl'


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
        """Test plotting modes."""
        kraken = Kraken(verbose=False)
        modes = kraken.compute_modes(env=simple_env, source=source, n_modes=10)

        fig, axes = modes.plot(n_modes=6)

        assert fig is not None
        assert axes is not None
        assert len(axes) == 2  # mode shapes and wavenumber spectrum
        plt.close(fig)

    def test_plot_with_custom_parameters(self, simple_env, source, receiver_small):
        """Test plotting with custom parameters."""
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=simple_env, source=source, receiver=receiver_small)

        fig, ax = result.plot(env=simple_env, vmin=40, vmax=100, cmap='jet')

        assert fig is not None
        plt.close(fig)

    def test_plot_comparison(self, simple_env, source, receiver_small):
        """Test Field.plot_comparison() static method."""
        bellhop = Bellhop(verbose=False)
        krakenfield = KrakenField(verbose=False)

        results = {
            'Bellhop': bellhop.compute_tl(env=simple_env, source=source, receiver=receiver_small),
            'KrakenField': krakenfield.compute_tl(env=simple_env, source=source, receiver=receiver_small),
        }

        fig, axes = plots.compare_models(results, env=simple_env)

        assert fig is not None
        assert axes is not None
        plt.close(fig)


class TestFieldMethods:
    """Tests for Field convenience methods."""

    def test_field_get_methods(self, simple_env, source, receiver_small):
        """Test Field get_value, get_at_range, get_at_depth."""
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=simple_env, source=source, receiver=receiver_small)

        # Test get_value
        value = result.get_value(range_m=3000, depth=50)
        assert isinstance(value, (float, np.floating))

        # Test get_at_range
        values_at_range = result.get_at_range(3000)
        assert len(values_at_range) == len(receiver_small.depths)

        # Test get_at_depth
        values_at_depth = result.get_at_depth(50)
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
        assert result.field_type == 'tl'
        assert result.shape == (len(receiver_small.depths), len(receiver_small.ranges))

    def test_run_and_compute_tl_agree(self, simple_env, source, receiver_small):
        """compute_tl and run(run_mode=COHERENT_TL) produce the same field."""
        bellhop = Bellhop(verbose=False)
        a = bellhop.run(env=simple_env, source=source, receiver=receiver_small,
                       run_mode=RunMode.COHERENT_TL)
        b = bellhop.compute_tl(env=simple_env, source=source, receiver=receiver_small)
        # Bellhop has non-deterministic floating-point; compare loosely.
        assert np.allclose(a.data, b.data, rtol=1e-3, atol=1e-3)
