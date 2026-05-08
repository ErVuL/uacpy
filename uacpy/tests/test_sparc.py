"""SPARC time-domain-focused tests."""

import pytest
import numpy as np

import uacpy
from uacpy.core.environment import SoundSpeedProfile
from uacpy.core.results import TimeSeriesField
from uacpy.models import Bellhop, RAM, Kraken, KrakenC, Scooter, SPARC
from uacpy.models.base import RunMode
from uacpy.core import Environment, BoundaryProperties, Source, Receiver
from uacpy.core.exceptions import ExecutableNotFoundError, UnsupportedFeatureError

pytestmark = pytest.mark.requires_binary

class TestSPARCBasic:
    """Basic tests for SPARC model (seismo-acoustic PE)."""

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_sparc_basic_tl(self):
        """Test basic SPARC TL computation."""
        env = Environment(
            name="sparc_test",
            bathymetry=100.0,
            sound_speed=1500.0
        )
        source = Source(depths=50.0, frequencies=50.0)
        receiver = Receiver(
            depths=np.linspace(10, 90, 9),
            ranges=np.linspace(100, 5000, 11)
        )

        sparc = SPARC(verbose=False)
        result = sparc.compute_tl(env=env, source=source, receiver=receiver)

        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))


class TestSPARCTimeSeries:
    """SPARC's primary purpose: native time-domain pressure p(t)."""

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_sparc_time_series_returns_time_series_field(self):
        """SPARC TIME_SERIES returns a real-valued TimeSeriesField."""
        env = Environment(
            name="sparc_ts",
            bathymetry=100.0,
            sound_speed=1500.0,
        )
        source = Source(depths=50.0, frequencies=50.0)
        receiver = Receiver(
            depths=np.linspace(10, 90, 5),
            ranges=np.linspace(500, 5000, 6),
        )

        sparc = SPARC(verbose=False)
        result = sparc.run(
            env, source, receiver,
            run_mode=RunMode.TIME_SERIES,
        )

        assert isinstance(result, TimeSeriesField)
        assert result.data.shape[0] == len(receiver.depths)
        assert result.data.shape[1] == len(receiver.ranges)
        assert result.data.shape[2] > 0
        assert np.isrealobj(result.data)
        assert np.all(np.isfinite(result.data))
        assert 'time' in result.metadata or hasattr(result, 'time')
