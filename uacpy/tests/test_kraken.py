"""Kraken / KrakenC normal-mode-focused tests."""

import pytest
import numpy as np

from uacpy.core.results import TransferFunction, TimeSeriesField
from uacpy.models import Kraken, KrakenC, KrakenField
from uacpy.models.base import RunMode
from uacpy.core import Environment, BoundaryProperties, Source, Receiver

pytestmark = pytest.mark.requires_binary


class TestKrakenFieldBroadband:
    """End-to-end BROADBAND / TIME_SERIES tests for KrakenField."""

    @pytest.mark.slow
    def test_krakenfield_broadband_returns_transfer_function(self):
        """KrakenField BROADBAND returns H(f) on the receiver grid."""
        env = Environment(name="kf_bb", bathymetry=100.0, ssp=1500.0)
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.array([1000.0, 3000.0]),
        )
        frequencies = np.linspace(80.0, 120.0, 5)

        kf = KrakenField(verbose=False)
        result = kf.run(
            env, source, receiver,
            run_mode=RunMode.BROADBAND,
            frequencies=frequencies,
        )

        assert isinstance(result, TransferFunction)
        assert np.iscomplexobj(result.data)
        assert result.data.shape[0] == len(receiver.depths)
        assert result.data.shape[1] == len(receiver.ranges)
        assert result.data.shape[2] > 0

    @pytest.mark.slow
    def test_krakenfield_time_series_returns_time_series_field(self):
        """KrakenField TIME_SERIES with a tonal waveform returns TimeSeriesField."""
        env = Environment(name="kf_ts", bathymetry=100.0, ssp=1500.0)
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

        kf = KrakenField(verbose=False)
        result = kf.run(
            env, source, receiver,
            run_mode=RunMode.TIME_SERIES,
            frequencies=frequencies,
            source_waveform=waveform,
            sample_rate=fs,
        )

        assert isinstance(result, TimeSeriesField)
        assert result.data.shape[0] == len(receiver.depths)
        assert result.data.shape[1] == len(receiver.ranges)
        assert result.data.shape[2] > 0
        assert np.all(np.isfinite(result.data))

    """Test KrakenC for complex modes with elastic bottom."""

    @pytest.fixture
    def elastic_env(self):
        """Create environment with elastic bottom."""
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0,
            shear_speed=400.0,
            density=1.8,
            attenuation=0.2,
            shear_attenuation=0.5
        )
        return Environment(
            name="krakenc_test",
            bathymetry=100.0,
            ssp=1500.0,
            bottom=bottom
        )

    @pytest.fixture
    def receiver(self):
        return Receiver(depths=[25.0, 50.0, 75.0], ranges=[1000.0, 3000.0])

    @pytest.mark.requires_binary
    def test_krakenc_complex_modes(self, elastic_env, source, receiver):
        """Test KrakenC complex mode computation."""
        krakenc = KrakenC(verbose=False)

        modes = krakenc.run(
            env=elastic_env,
            source=source,
            receiver=receiver
        )

        assert modes.field_type == 'modes'
        assert 'k' in modes.metadata
        assert len(modes.metadata['k']) > 0

        # Complex modes should have complex wavenumbers
        k = modes.metadata['k']
        assert np.any(np.imag(k) != 0), "Should have complex wavenumbers for elastic bottom"


class TestKrakenAttenuationUnit:
    """B1: Kraken / KrakenC / KrakenField accept attenuation_unit kwarg."""

    @pytest.mark.parametrize('cls', [Kraken, KrakenC, KrakenField])
    def test_default_is_db_per_wavelength(self, cls):
        from uacpy.core.constants import AttenuationUnits
        m = cls()
        assert m.attenuation_unit == AttenuationUnits.DB_PER_WAVELENGTH

    @pytest.mark.parametrize('cls', [Kraken, KrakenC, KrakenField])
    def test_constructor_kwarg_accepted(self, cls):
        from uacpy.core.constants import AttenuationUnits
        m = cls(attenuation_unit='M')
        assert m.attenuation_unit == AttenuationUnits.from_string('M')

    def test_attenuation_unit_reaches_env_writer(self, tmp_path):
        from uacpy.core.constants import AttenuationUnits
        kraken = Kraken(attenuation_unit='M')
        env = Environment(name='kr', bathymetry=100.0, ssp=1500.0)
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(depths=[25.0, 50.0, 75.0], ranges=[1000.0])
        env_file = tmp_path / 'kraken.env'
        kraken._write_kraken_env(
            env_file, env, source,
            receiver_obj=receiver,
            receiver_depths=receiver.depths,
        )
        # 'M' is the AT code for dB per meter; check the TopOpt line has it.
        text = env_file.read_text()
        # TopOpt is on line 3 (after title + freq + nmedia).
        topopt_line = text.splitlines()[3]
        # Position 3 (0-indexed 2) is the attenuation-units character.
        # AttenuationUnits.from_string('M').value should equal 'M'.
        assert AttenuationUnits.from_string('M').value in topopt_line


class TestKrakenModePointsPerMeter:
    """B6: Kraken / KrakenC expose mode_points_per_meter."""

    @pytest.mark.parametrize('cls', [Kraken, KrakenC])
    def test_default_is_1_5(self, cls):
        m = cls()
        assert m.mode_points_per_meter == 1.5

    @pytest.mark.parametrize('cls', [Kraken, KrakenC])
    def test_density_kwarg_accepted(self, cls):
        m = cls(mode_points_per_meter=3.0)
        assert m.mode_points_per_meter == 3.0

    def test_compute_modes_uses_mode_points_per_meter(self, monkeypatch):
        """The dense mode-depth grid scales with mode_points_per_meter."""
        captured = {}

        def spy_run(self_, env, source, dense_receiver, *args, **kwargs):
            captured['n_depths'] = len(dense_receiver.depths)
            captured['z_max'] = float(np.max(dense_receiver.depths))
            raise RuntimeError('stop after _compute_modes_impl')

        monkeypatch.setattr(Kraken, 'run', spy_run)

        env = Environment(name='kr_modes', bathymetry=200.0, ssp=1500.0)
        source = Source(depths=100.0, frequencies=50.0)
        kraken = Kraken(mode_points_per_meter=5.0)
        with pytest.raises(RuntimeError, match='stop'):
            kraken.compute_modes(env, source, n_modes=3)
        # 200 m * 5 pts/m = 1000 pts (>=100 floor).
        assert captured['n_depths'] == 1000
        assert captured['z_max'] == pytest.approx(200.0)
