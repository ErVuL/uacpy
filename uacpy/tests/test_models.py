"""
Tests for all UACPY propagation models
"""

import pytest

import numpy as np
from uacpy.models import (
    Bellhop, RAM, Kraken, KrakenField,
    Bounce, Scooter,
)
from uacpy.models.base import RunMode
from uacpy.core.receiver import Receiver
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Field, Modes, ReflectionCoefficient


@pytest.mark.requires_binary
class TestBellhop:
    """Tests for Bellhop model. Smoke TL coverage on ``simple_env`` lives
    in ``test_simplified_api.TestComputeAPI`` and ``test_bellhop`` —
    only model-specific scenarios live here."""

    def test_bellhop_range_dependent(self, range_dependent_env, source, receiver_small):
        """Test Bellhop with range-dependent environment."""
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(
            env=range_dependent_env,
            source=source,
            receiver=receiver_small
        )

        assert isinstance(result, Field)
        assert result.shape[0] == len(receiver_small.depths)

    def test_bellhopcuda_compute_tl(self, simple_env, source, receiver_small):
        """Smoke test for ``BellhopCUDA``. Skipped when no CUDA/CXX binary
        is available — instantiation will raise ``ExecutableNotFoundError``.
        """
        from uacpy.models import BellhopCUDA
        from uacpy.core.exceptions import ExecutableNotFoundError
        try:
            bhc = BellhopCUDA(verbose=False)
        except ExecutableNotFoundError:
            pytest.skip("bellhopcuda / bellhopcxx binary not installed")
        result = bhc.compute_tl(env=simple_env, source=source, receiver=receiver_small)
        assert isinstance(result, Field)
        assert result.shape == (len(receiver_small.depths), len(receiver_small.ranges))


@pytest.mark.requires_binary
class TestKraken:
    """Tests for Kraken model."""

    def test_kraken_compute_modes(self, simple_env, source):
        """Test Kraken mode computation.

        Note: standalone Kraken does not accept ``n_modes``; that knob
        lives on ``KrakenField.run`` (MLimit in the FLP file).
        """
        kraken = Kraken(verbose=False)
        modes = kraken.compute_modes(env=simple_env, source=source)

        assert isinstance(modes, Modes)
        assert modes.k is not None
        assert modes.phi is not None
        assert len(modes.k) > 0

    def test_kraken_n_modes_clips_output(self, simple_env, source):
        """``n_modes`` caps the number of returned modes from Kraken."""
        kraken = Kraken(verbose=False)
        capped = kraken.compute_modes(env=simple_env, source=source, n_modes=3)
        assert len(capped.k) <= 3
        assert capped.metadata.get('n_modes_requested') == 3

    def test_kraken_modes_have_wavenumbers(self, simple_env, source):
        """Test that computed modes have valid wavenumbers."""
        kraken = Kraken(verbose=False)
        modes = kraken.compute_modes(env=simple_env, source=source)

        k = modes.k
        assert len(k) > 0
        # Real part of wavenumber should be positive for propagating modes
        # Some modes may have k≈0 (non-propagating), which is valid
        k_real = np.real(k)
        propagating_modes = k_real > 1e-6  # Threshold for propagating vs non-propagating
        assert np.any(propagating_modes), "Should have at least one propagating mode"
        # All propagating modes should have positive wavenumbers
        assert np.all(k_real[propagating_modes] > 0)


@pytest.mark.requires_binary
class TestKrakenField:
    """Tests for KrakenField model."""

    def test_krakenfield_compute_tl(self, simple_env, source, receiver_small):
        """Test KrakenField TL computation."""
        kf = KrakenField(verbose=False)
        result = kf.compute_tl(env=simple_env, source=source, receiver=receiver_small)

        assert isinstance(result, Field)
        assert result.shape == (len(receiver_small.depths), len(receiver_small.ranges))


@pytest.mark.requires_binary
class TestBounce:
    """Tests for Bounce model."""

    def test_bounce_compute_reflection_coefficient(self, simple_env, source, receiver_small, tmp_path):
        """Test Bounce reflection coefficient computation.

        Uses ``work_dir`` (with Bounce's default ``cleanup=False``) so
        the .brc/.irc files survive past the call for the consumer model.
        """
        bounce = Bounce(verbose=False, work_dir=tmp_path)

        # Bounce needs an environment with elastic bottom properties
        from uacpy.core import Environment, BoundaryProperties
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600,
            shear_speed=400,
            density=1.8,
            attenuation=0.2,
            shear_attenuation=0.5
        )
        env_elastic = Environment(
            name="elastic_test",
            bathymetry=simple_env.depth,
            ssp=float(simple_env.ssp.data[0, 0]),
            bottom=bottom
        )

        result = bounce.run(
            env=env_elastic,
            source=source,
            receiver=receiver_small,
        )

        assert isinstance(result, ReflectionCoefficient)
        assert 'brc_file' in result.metadata
        assert result.metadata['brc_file'] is not None

        # Check that .brc file persists in work_dir
        import os
        brc_file = result.metadata['brc_file']
        assert os.path.exists(brc_file), f"BRC file should exist: {brc_file}"

        # Check reflection coefficient data
        assert result.R is not None
        assert result.theta is not None
        assert len(result.R) > 0
        assert len(result.theta) > 0

    def test_bounce_compute_reflection_helper(self, simple_env, source, receiver_small, tmp_path):
        """Verify the convenience method ``Bounce.compute_reflection`` runs."""
        from uacpy.core import Environment, BoundaryProperties
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600, shear_speed=400, density=1.8,
            attenuation=0.2, shear_attenuation=0.5,
        )
        env_elastic = Environment(
            name="elastic_test",
            bathymetry=simple_env.depth,
            ssp=float(simple_env.ssp.data[0, 0]),
            bottom=bottom,
        )
        bounce = Bounce(verbose=False, work_dir=tmp_path)
        result = bounce.compute_reflection(
            env=env_elastic, source=source, receiver=receiver_small,
        )
        assert isinstance(result, ReflectionCoefficient)


@pytest.mark.requires_binary
class TestRAM:
    """Tests for RAM model (mpiramS backend)."""

    def test_ram_compute_tl(self, simple_env, source, receiver_small):
        """Test RAM TL computation."""
        ram = RAM(verbose=False, dr=20.0, dz=2.0)
        result = ram.compute_tl(env=simple_env, source=source, receiver=receiver_small)

        assert isinstance(result, Field)
        assert result.shape[0] > 0  # Has depth dimension
        assert result.shape[1] > 0  # Has range dimension
        assert np.all(np.isfinite(result.data))

    def test_ram_broadband_mode(self, simple_env, source):
        """RAM BROADBAND returns the H(f) transfer function."""
        ram = RAM(Q=2.0, T=2.0, dr=20.0, dz=2.0, verbose=False)
        receiver = Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.array([5000.0])
        )
        result = ram.run(
            simple_env, source, receiver,
            run_mode=RunMode.BROADBAND
        )
        assert isinstance(result, Field)
        assert np.iscomplexobj(result.data)
        # Shape: (n_d, n_r, n_f) — trailing axis is the
        # variable dimension (frequency, here).
        assert result.data.shape[0] > 0  # depth
        assert result.data.shape[1] > 0  # range
        assert result.data.shape[2] > 0  # frequency

    def test_ram_time_series_requires_waveform(self, simple_env, source):
        """TIME_SERIES without source_waveform must raise."""
        ram = RAM(Q=2.0, T=2.0, dr=20.0, dz=2.0, verbose=False)
        receiver = Receiver(
            depths=np.array([50.0]),
            ranges=np.array([5000.0])
        )
        with pytest.raises(ConfigurationError, match="source_waveform"):
            ram.run(simple_env, source, receiver,
                    run_mode=RunMode.TIME_SERIES)

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_ram_compute_time_series_helper(self, simple_env, source):
        """Verify the convenience method ``RAM.compute_time_series`` runs.

        The helper takes no ``frequencies=`` so the auto-derive in
        ``_resolve_time_series_frequencies`` fires; the warning is
        expected behaviour, filtered here.
        """
        from uacpy.core.results import Field
        ram = RAM(Q=2.0, T=2.0, dr=20.0, dz=2.0, verbose=False)
        receiver = Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))
        fs = 4000.0
        nt = 64
        t = np.arange(nt) / fs
        sigma = nt / (8.0 * fs)
        f0 = float(np.atleast_1d(source.frequencies)[0])
        wf = (np.sin(2 * np.pi * f0 * (t - t[-1] / 2))
              * np.exp(-((t - t[-1] / 2) ** 2) / (2 * sigma ** 2)))
        result = ram.compute_time_series(
            simple_env, source, receiver,
            source_waveform=wf, sample_rate=fs,
        )
        assert isinstance(result, Field)
        assert result.data.shape[0] == 1
        assert result.data.shape[1] == 1


# OASES instantiation/supported-mode tests live in test_oases_comprehensive.py;
# the cross-model workflow tests below cover Bounce → {Bellhop, Scooter, KrakenC}.


@pytest.mark.requires_binary
class TestModelConsistency:
    """Tests for consistency between different models."""

    # Bellhop ↔ KrakenField TL agreement is covered with tighter
    # tolerance in test_cross_model_agreement.py.

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "downstream",
        [
            pytest.param("Bellhop", id="bellhop"),
            pytest.param(
                "KrakenC",
                id="krakenc",
                marks=pytest.mark.xfail(
                    reason=(
                        "KRAKENC support for .brc files is experimental in the "
                        "Acoustics Toolbox and currently fails with file format "
                        "errors. Use SCOOTER for production .brc workflows."
                    ),
                    strict=True,
                ),
            ),
            pytest.param("Scooter", id="scooter"),
        ],
    )
    def test_bounce_to_downstream_workflow(
        self, simple_env, source, receiver_small, tmp_path, downstream
    ):
        """BOUNCE → downstream model workflow via .brc reflection coefficients.

        Step 1 computes reflection coefficients on an elastic half-space with
        BOUNCE, persisting the .brc file to ``tmp_path``. Step 2 feeds the
        .brc back into the downstream model (Bellhop / KrakenC / Scooter)
        and verifies it produces a valid result.
        """
        import os

        from uacpy.core import Environment, BoundaryProperties
        from uacpy.models import KrakenC

        # Step 1 — BOUNCE on elastic bottom
        bottom_elastic = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600,
            shear_speed=400,
            density=1.8,
            attenuation=0.2,
            shear_attenuation=0.5,
        )
        env_elastic = Environment(
            name="elastic_test",
            bathymetry=simple_env.depth,
            ssp=float(simple_env.ssp.data[0, 0]),
            bottom=bottom_elastic,
        )
        bounce = Bounce(verbose=False, work_dir=tmp_path)
        bounce_result = bounce.run(
            env=env_elastic, source=source, receiver=receiver_small,
        )
        assert 'brc_file' in bounce_result.metadata
        brc_file = bounce_result.metadata['brc_file']
        assert os.path.exists(brc_file), "BRC file should exist"

        # Step 2 — feed .brc into the downstream model
        bottom_with_rc = BoundaryProperties(
            acoustic_type='file',
            reflection_file=brc_file,
            sound_speed=1600,
            density=1.8,
        )
        env_with_rc = Environment(
            name="test_with_rc",
            bathymetry=simple_env.depth,
            ssp=float(simple_env.ssp.data[0, 0]),
            bottom=bottom_with_rc,
        )

        c_low_brc = bounce_result.metadata['c_low']
        c_high_brc = bounce_result.metadata['c_high']

        if downstream == "KrakenC":
            modes = KrakenC(
                verbose=False, c_low=c_low_brc, c_high=c_high_brc,
            ).run(env=env_with_rc, source=source, receiver=receiver_small)
            assert isinstance(modes, Modes)
            assert modes.k is not None and len(modes.k) > 0
            assert modes.phi.shape[1] == len(modes.k)
            assert np.all(np.isfinite(modes.k))
        else:
            model_cls = {"Bellhop": Bellhop, "Scooter": Scooter}[downstream]
            if downstream == "Scooter":
                model = model_cls(
                    verbose=False, c_low=c_low_brc, c_high=c_high_brc,
                )
            else:
                model = model_cls(verbose=False)
            result = model.compute_tl(
                env=env_with_rc, source=source, receiver=receiver_small,
            )
            assert isinstance(result, Field)
            assert result.shape == (
                len(receiver_small.depths), len(receiver_small.ranges)
            )
            assert np.all(np.isfinite(result.data))
