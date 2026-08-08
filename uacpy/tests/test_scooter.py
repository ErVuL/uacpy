"""Scooter wavenumber-integration-focused tests."""

import pytest
import numpy as np

from uacpy.core.results import Field
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
        # Δf small enough that 1/Δf ≥ waveform duration (256/2000 = 0.128s)
        # → no DFT-wraparound warning from synthesize_time_series.
        frequencies = np.linspace(60.0, 140.0, 17)

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


def test_scooter_constructor_no_longer_accepts_source_type():
    with pytest.raises(TypeError):
        Scooter(source_type='R')


def test_scooter_constructor_no_longer_accepts_field_interp():
    """``field_interp`` named an FLP option for ``fields.exe``, which uacpy
    never runs — the k→r transform is done in-tree."""
    with pytest.raises(TypeError):
        Scooter(field_interp='P')


def test_grn_transform_method_is_the_direct_dft():
    """The transform is a trapezoidal-rule DFT (``fieldsco.m:5``), not an FFT,
    and ``'fft_hankel'`` is not an accepted name for it."""
    from uacpy.io.grn_reader import grn_to_field

    with pytest.raises(ConfigurationError, match="direct_dft"):
        grn_to_field({}, np.array([1000.0]), method='fft_hankel')


class TestZeroReceiverRange:
    """A receiver at ``r = 0`` sits on the point source's cylindrical-spreading
    singularity (``1/sqrt(r)``). ``fieldsco.m:69`` sidesteps it by moving the
    range to 1 m; uacpy reports no-data instead, and every model must report
    the same thing on the same grid."""

    RANGES = np.array([0.0, 1000.0, 3000.0])

    @staticmethod
    def _env():
        return Environment(name='zero_r', bathymetry=100.0, ssp=1500.0)

    @staticmethod
    def _source():
        return Source(depths=50.0, frequencies=100.0)

    def _receiver(self):
        return Receiver(depths=np.array([25.0, 75.0]), ranges=self.RANGES)

    def test_scooter_zero_range_is_no_data_not_a_huge_number(self):
        receiver = self._receiver()
        with pytest.warns(UserWarning, match="r = 0"):
            result = Scooter(verbose=False).run(
                self._env(), self._source(), receiver)
        data = np.asarray(result.data)
        assert np.all(np.isnan(data[:, 0]))
        assert np.all(np.isfinite(data[:, 1:]))
        # The clamped-denominator form returned |p| ~ 1e152 here (TL ~ -3000 dB),
        # which poisons every colour scale and every max/mean over the grid.
        assert np.nanmax(np.abs(data)) < 1.0

    def test_kraken_and_scooter_agree_on_the_zero_range_cell(self):
        from uacpy.models import Kraken

        env, source = self._env(), self._source()
        with pytest.warns(UserWarning, match="r = 0"):
            scooter_tl = np.asarray(
                Scooter(verbose=False).run(env, source, self._receiver()).tl)
        with pytest.warns(UserWarning, match="r = 0"):
            kraken_tl = np.asarray(
                Kraken(verbose=False).compute_tl(
                    env, source, self._receiver()).tl)

        assert np.all(np.isnan(scooter_tl[:, 0]))
        assert np.all(np.isnan(kraken_tl[:, 0]))
        assert np.all(np.isfinite(scooter_tl[:, 1:]))
        assert np.all(np.isfinite(kraken_tl[:, 1:]))


class TestScooterReceiverDepthAxis:
    """The settled below-domain policy (``PropagationModel.validate_inputs``):
    receivers are outputs, so a receiver below the deepest modelled interface
    is accepted and returns the model's below-domain value. The returned depth
    axis must therefore be the one the caller asked for — moving receivers onto
    the mesh and de-duplicating them silently misaligns every row a caller
    indexes against its own depth array."""

    @staticmethod
    def _env():
        from uacpy.core import BoundaryProperties
        from uacpy.core.bottom import Bottom, SeabedColumn, SedimentLayer
        column = SeabedColumn(
            layers=[SedimentLayer(thickness=20.0, sound_speed=1600.0,
                                  density=1.8, attenuation=0.5)],
            halfspace=BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0,
                density=2.0, attenuation=0.8))
        return Environment(name='media', bathymetry=200.0, ssp=1500.0,
                           bottom=Bottom(columns=[column]))

    def test_requested_depths_are_returned_verbatim(self):
        # 200 m water + 20 m sediment ⇒ resolvable to 220 m; 300 and 400 are
        # below it and would previously have collapsed onto one 217 m row.
        receiver = Receiver(depths=np.array([50.0, 150.0, 300.0, 400.0]),
                            ranges=np.linspace(500.0, 5000.0, 5))
        result = Scooter(verbose=False).run(
            self._env(), Source(depths=50.0, frequencies=100.0), receiver)
        assert result.data.shape[0] == receiver.depths.size
        assert np.asarray(result.coords['depth']) == pytest.approx(
            receiver.depths)

    def test_unresolvable_depths_are_no_data(self):
        """Below the mesh the binary clamps onto the deepest interface; that
        value belongs to a different depth, so it must not be handed back."""
        receiver = Receiver(depths=np.array([50.0, 150.0, 300.0, 400.0]),
                            ranges=np.linspace(500.0, 5000.0, 5))
        data = np.asarray(Scooter(verbose=False).run(
            self._env(), Source(depths=50.0, frequencies=100.0), receiver).data)
        assert np.all(np.isfinite(data[:2]))
        assert np.all(np.isnan(data[2:]))


class TestScooterRejectsATooCoarseMesh:
    """SCOOTER reads its deck through the same ``misc/ReadEnvironmentMod.f90``
    as KRAKEN, so a pinned ``n_mesh`` under the ``Nneeded / 2`` floor at
    ``:110-112`` stops the binary with *Mesh is too coarse*. Guarding only
    KRAKEN left that surfacing as a bare Fortran fatal."""

    @staticmethod
    def _env():
        from uacpy.core import BoundaryProperties
        return Environment(
            name='pekeris', bathymetry=200.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.3))

    def test_a_too_coarse_n_mesh_is_a_configuration_error(self):
        with pytest.raises(ConfigurationError, match='Mesh is too coarse'):
            Scooter(n_mesh=5, verbose=False).run(
                self._env(), Source(depths=50.0, frequencies=100.0),
                Receiver(depths=[100.0], ranges=np.linspace(1000.0, 5000.0, 5)))

    def test_auto_mesh_is_never_rejected(self):
        """``NG = 0`` is AT's automatic sizing and is never range-checked."""
        result = Scooter(verbose=False).run(
            self._env(), Source(depths=50.0, frequencies=100.0),
            Receiver(depths=[100.0], ranges=np.linspace(1000.0, 5000.0, 5)))
        assert np.all(np.isfinite(np.asarray(result.data)))
