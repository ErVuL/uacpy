"""RAM parabolic-equation-focused tests."""

import pytest
import numpy as np

import uacpy
from uacpy.core.environment import SoundSpeedProfile
from uacpy.models import Bellhop, RAM, Kraken, KrakenC, Scooter, SPARC
from uacpy.models.base import RunMode
from uacpy.core import Environment, BoundaryProperties, Source, Receiver
from uacpy.core.exceptions import ExecutableNotFoundError, UnsupportedFeatureError

pytestmark = pytest.mark.requires_binary

class TestRAMAdvancedParameters:
    """Test RAM Pade orders and stability parameters."""

    @pytest.fixture
    def ram_env(self):
        return Environment(
            name="ram_test",
            depth=100.0,
            sound_speed=1500.0
        )

    @pytest.fixture
    def ram_source(self):
        return Source(depths=50.0, frequencies=50.0)

    @pytest.fixture
    def ram_receiver(self):
        return Receiver(
            depths=np.linspace(10, 90, 9),
            ranges=np.linspace(100, 5000, 11)
        )

    @pytest.mark.parametrize('np_pade', [2, 6, 8])
    def test_ram_pade_order(self, ram_env, ram_source, ram_receiver, np_pade):
        """RAM converges across the supported Padé-coefficient counts."""
        ram = RAM(verbose=False)
        result = ram.compute_tl(
            env=ram_env,
            source=ram_source,
            receiver=ram_receiver,
            np_pade=np_pade,
        )
        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))

    def test_ram_stability_parameter(self, ram_env, ram_source, ram_receiver):
        """Test RAM stability parameter."""
        ram = RAM(verbose=False)
        result = ram.compute_tl(
            env=ram_env,
            source=ram_source,
            receiver=ram_receiver,
            ns_stability=1
        )
        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))

    def test_ram_custom_dr_dz(self, ram_env, ram_source, ram_receiver):
        """Test RAM with custom range and depth steps."""
        ram = RAM(verbose=False)
        result = ram.compute_tl(
            env=ram_env,
            source=ram_source,
            receiver=ram_receiver,
            dr=10.0,  # 10m range step
            dz=0.5    # 0.5m depth step
        )
        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))

    def test_ram_tl_honors_constructor_Q_T(
        self, ram_env, ram_source, ram_receiver, monkeypatch,
    ):
        """RAM(Q=…, T=…) values reach the in.pe file written for COHERENT_TL.

        Regression for the hardcoded ``Q_tl=1e6, T_tl=1.0`` in ``_run_tl``.
        """
        from uacpy.io import mpirams_writer as mpw
        from uacpy.models import ram as ram_mod
        captured = {}

        def fake_write_inpe(*args, **kwargs):
            captured['Q'] = kwargs['Q']
            captured['T'] = kwargs['T']
            raise RuntimeError("stop after writing in.pe")

        monkeypatch.setattr(mpw, 'write_inpe', fake_write_inpe)
        monkeypatch.setattr(ram_mod, 'write_inpe', fake_write_inpe)

        ram = RAM(Q=4.0, T=20.0, verbose=False)
        with pytest.raises(RuntimeError, match="stop after writing in.pe"):
            ram.compute_tl(env=ram_env, source=ram_source, receiver=ram_receiver)
        assert captured['Q'] == 4.0
        assert captured['T'] == 20.0

    def test_ram_tl_run_kwargs_override_constructor_Q_T(
        self, ram_env, ram_source, ram_receiver, monkeypatch,
    ):
        """run(Q=…, T=…) overrides constructor values."""
        from uacpy.io import mpirams_writer as mpw
        from uacpy.models import ram as ram_mod
        captured = {}

        def fake_write_inpe(*args, **kwargs):
            captured['Q'] = kwargs['Q']
            captured['T'] = kwargs['T']
            raise RuntimeError("stop after writing in.pe")

        monkeypatch.setattr(mpw, 'write_inpe', fake_write_inpe)
        monkeypatch.setattr(ram_mod, 'write_inpe', fake_write_inpe)

        ram = RAM(Q=4.0, T=20.0, verbose=False)
        with pytest.raises(RuntimeError, match="stop after writing in.pe"):
            ram.run(
                env=ram_env, source=ram_source, receiver=ram_receiver,
                Q=99.0, T=33.0,
            )
        assert captured['Q'] == 99.0
        assert captured['T'] == 33.0
