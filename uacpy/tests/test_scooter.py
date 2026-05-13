"""Scooter wavenumber-integration-focused tests."""

import pytest
import numpy as np

from uacpy.core.results import Field, Field
from uacpy.models import Scooter
from uacpy.models.base import RunMode
from uacpy.core import Environment, Source, Receiver
from uacpy.core.exceptions import ConfigurationError

pytestmark = pytest.mark.requires_binary


class TestScooterBasic:
    """Basic tests for Scooter model (wavenumber integration)."""

    @pytest.mark.requires_binary
    def test_scooter_basic_tl(self):
        """Test basic Scooter TL computation."""
        env = Environment(
            name="scooter_test",
            bathymetry=100.0,
            ssp=1500.0
        )
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.array([1000.0, 3000.0])
        )

        scooter = Scooter(verbose=False)
        result = scooter.compute_tl(env=env, source=source, receiver=receiver)

        assert isinstance(result, Field)
        assert result.shape == (len(receiver.depths), len(receiver.ranges))
        assert np.all(np.isfinite(result.data))


class TestScooterBroadband:
    """End-to-end BROADBAND / TIME_SERIES tests for Scooter."""

    @pytest.mark.slow
    def test_scooter_broadband_returns_transfer_function(self):
        """Scooter BROADBAND returns a populated H(f) Field."""
        env = Environment(name="sc_bb", bathymetry=100.0, ssp=1500.0)
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.array([1000.0, 3000.0]),
        )
        frequencies = np.linspace(80.0, 120.0, 5)

        scooter = Scooter(verbose=False)
        result = scooter.run(
            env, source, receiver,
            run_mode=RunMode.BROADBAND,
            frequencies=frequencies,
        )

        assert isinstance(result, Field)
        assert np.iscomplexobj(result.data)
        assert result.data.shape[:2] == (len(receiver.depths), len(receiver.ranges))
        assert result.data.shape[2] > 0

    @pytest.mark.slow
    def test_scooter_time_series_returns_time_series_field(self):
        """Scooter TIME_SERIES with a tonal waveform returns Field."""
        env = Environment(name="sc_ts", bathymetry=100.0, ssp=1500.0)
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(
            depths=np.array([50.0]),
            ranges=np.array([2000.0]),
        )
        fs = 2000.0
        n = 256
        t = np.arange(n) / fs
        waveform = np.sin(2 * np.pi * 100.0 * t) * np.hanning(n)
        frequencies = np.linspace(60.0, 140.0, 9)

        scooter = Scooter(verbose=False)
        result = scooter.run(
            env, source, receiver,
            run_mode=RunMode.TIME_SERIES,
            frequencies=frequencies,
            source_waveform=waveform,
            sample_rate=fs,
        )

        assert isinstance(result, Field)
        assert result.data.shape[0] == len(receiver.depths)
        assert result.data.shape[1] == len(receiver.ranges)
        assert result.data.shape[2] > 0
        assert np.all(np.isfinite(result.data))

    def test_scooter_time_series_requires_waveform(self):
        """Scooter TIME_SERIES without source_waveform must raise."""
        env = Environment(name="sc_ts_err", bathymetry=100.0, ssp=1500.0)
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(
            depths=np.array([50.0]),
            ranges=np.array([2000.0]),
        )
        scooter = Scooter(verbose=False)
        with pytest.raises(ConfigurationError, match="source_waveform"):
            scooter.run(
                env, source, receiver,
                run_mode=RunMode.TIME_SERIES,
            )
