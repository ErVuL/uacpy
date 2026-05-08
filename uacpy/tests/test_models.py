"""
Tests for all UACPY propagation models
"""

import pytest

import numpy as np
import uacpy
from uacpy.models import (
    Bellhop, BellhopCUDA, RAM, Kraken, KrakenField,
    Bounce, Scooter, SPARC,
)
from uacpy.models.base import RunMode
from uacpy.core.receiver import Receiver


@pytest.mark.requires_binary
class TestBellhop:
    """Tests for Bellhop model."""

    def test_bellhop_compute_tl(self, simple_env, source, receiver_small):
        """Test Bellhop TL computation."""
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=simple_env, source=source, receiver=receiver_small)

        assert result.field_type == 'tl'
        assert result.shape == (len(receiver_small.depths), len(receiver_small.ranges))
        assert np.all(np.isfinite(result.data))
        assert np.all(result.data > 0)  # TL should be positive

    def test_bellhop_range_dependent(self, range_dependent_env, source, receiver_small):
        """Test Bellhop with range-dependent environment."""
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(
            env=range_dependent_env,
            source=source,
            receiver=receiver_small
        )

        assert result.field_type == 'tl'
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
        assert result.field_type == 'tl'
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

        assert modes.field_type == 'modes'
        assert 'k' in modes.metadata
        assert 'phi' in modes.metadata
        assert len(modes.metadata['k']) > 0

    def test_kraken_n_modes_clips_output(self, simple_env, source):
        """``n_modes`` caps the number of returned modes from Kraken."""
        kraken = Kraken(verbose=False)
        capped = kraken.compute_modes(env=simple_env, source=source, n_modes=3)
        assert len(capped.metadata['k']) <= 3
        assert capped.metadata.get('n_modes_requested') == 3

    def test_kraken_modes_have_wavenumbers(self, simple_env, source):
        """Test that computed modes have valid wavenumbers."""
        kraken = Kraken(verbose=False)
        modes = kraken.compute_modes(env=simple_env, source=source)

        k = modes.metadata['k']
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

        assert result.field_type == 'tl'
        assert result.shape == (len(receiver_small.depths), len(receiver_small.ranges))


@pytest.mark.requires_binary
class TestBounce:
    """Tests for Bounce model."""

    def test_bounce_compute_reflection_coefficient(self, simple_env, source, receiver_small, tmp_path):
        """Test Bounce reflection coefficient computation.

        Uses ``output_dir`` so the .brc/.irc files survive cleanup.
        """
        bounce = Bounce(verbose=False)

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
            sound_speed=float(simple_env.ssp.data[0, 0]),
            bottom=bottom
        )

        result = bounce.run(
            env=env_elastic,
            source=source,
            receiver=receiver_small,
            output_dir=tmp_path,
        )

        assert result.field_type == 'reflection_coefficients'
        assert 'brc_file' in result.metadata
        assert result.metadata['brc_file'] is not None

        # Check that .brc file persisted via output_dir
        import os
        brc_file = result.metadata['brc_file']
        assert os.path.exists(brc_file), f"BRC file should exist: {brc_file}"

        # Check reflection coefficient data
        assert 'R' in result.metadata
        assert 'theta' in result.metadata
        assert len(result.metadata['R']) > 0
        assert len(result.metadata['theta']) > 0

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
            sound_speed=float(simple_env.ssp.data[0, 0]),
            bottom=bottom,
        )
        bounce = Bounce(verbose=False)
        result = bounce.compute_reflection(
            env=env_elastic, source=source, receiver=receiver_small,
            output_dir=tmp_path,
        )
        assert result.field_type == 'reflection_coefficients'


@pytest.mark.requires_binary
class TestRAM:
    """Tests for RAM model (mpiramS backend)."""

    def test_ram_compute_tl(self, simple_env, source, receiver_small):
        """Test RAM TL computation."""
        ram = RAM(verbose=False)
        result = ram.compute_tl(env=simple_env, source=source, receiver=receiver_small)

        assert result.field_type == 'tl'
        assert result.shape[0] > 0  # Has depth dimension
        assert result.shape[1] > 0  # Has range dimension
        assert np.all(np.isfinite(result.data))

    def test_ram_broadband_mode(self, simple_env, source):
        """RAM BROADBAND returns the H(f) transfer function."""
        ram = RAM(Q=2.0, T=2.0, verbose=False)
        receiver = Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.array([5000.0])
        )
        result = ram.run(
            simple_env, source, receiver,
            run_mode=RunMode.BROADBAND
        )
        assert result.field_type == 'transfer_function'
        assert np.iscomplexobj(result.data)
        # Shape: (n_d, n_r, n_f) — trailing axis is the
        # variable dimension (frequency, here).
        assert result.data.shape[0] > 0  # depth
        assert result.data.shape[1] > 0  # range
        assert result.data.shape[2] > 0  # frequency

    def test_ram_time_series_requires_waveform(self, simple_env, source):
        """TIME_SERIES without source_waveform must raise."""
        ram = RAM(Q=2.0, T=2.0, verbose=False)
        receiver = Receiver(
            depths=np.array([50.0]),
            ranges=np.array([5000.0])
        )
        with pytest.raises(ValueError, match="source_waveform"):
            ram.run(simple_env, source, receiver,
                    run_mode=RunMode.TIME_SERIES)

    @pytest.mark.slow
    def test_ram_compute_time_series_helper(self, simple_env, source):
        """Verify the convenience method ``RAM.compute_time_series`` runs."""
        from uacpy.core.results import TimeSeriesField
        ram = RAM(Q=2.0, T=2.0, verbose=False)
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
        assert isinstance(result, TimeSeriesField)
        assert result.data.shape[0] == 1
        assert result.data.shape[1] == 1


# OASES instantiation/supported-mode tests live in test_oases_comprehensive.py;
# the cross-model workflow tests below cover Bounce → {Bellhop, Scooter, KrakenC}.


@pytest.mark.requires_binary
class TestModelConsistency:
    """Tests for consistency between different models."""

    @pytest.mark.slow
    def test_tl_consistency_bellhop_kraken(self, simple_env, source, receiver_small):
        """Test TL consistency between Bellhop and KrakenField."""
        bellhop = Bellhop(verbose=False)
        krakenfield = KrakenField(verbose=False)

        result_bellhop = bellhop.compute_tl(env=simple_env, source=source, receiver=receiver_small)
        result_kraken = krakenfield.compute_tl(env=simple_env, source=source, receiver=receiver_small)

        # Results should be broadly similar (within ~20 dB in most places)
        # This is a rough consistency check, not an exact match
        valid_bellhop = result_bellhop.data[np.isfinite(result_bellhop.data)]
        valid_kraken = result_kraken.data[np.isfinite(result_kraken.data)]

        if len(valid_bellhop) > 0 and len(valid_kraken) > 0:
            # Check that mean TL is in similar range
            mean_diff = abs(np.mean(valid_bellhop) - np.mean(valid_kraken))
            assert mean_diff < 30, f"Mean TL difference too large: {mean_diff:.1f} dB"

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
            sound_speed=float(simple_env.ssp.data[0, 0]),
            bottom=bottom_elastic,
        )
        bounce = Bounce(verbose=False)
        bounce_result = bounce.run(
            env=env_elastic, source=source, receiver=receiver_small,
            output_dir=tmp_path,
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
            reflection_cmin=bounce_result.metadata['c_low'],
            reflection_cmax=bounce_result.metadata['c_high'],
            reflection_rmax_m=bounce_result.metadata['rmax_m'],
        )
        env_with_rc = Environment(
            name="test_with_rc",
            bathymetry=simple_env.depth,
            sound_speed=float(simple_env.ssp.data[0, 0]),
            bottom=bottom_with_rc,
        )

        if downstream == "KrakenC":
            modes = KrakenC(verbose=False).run(
                env=env_with_rc, source=source, receiver=receiver_small,
            )
            assert modes.field_type == 'modes'
            assert 'k' in modes.metadata and len(modes.metadata['k']) > 0
            assert modes.metadata['phi'].shape[1] == len(modes.metadata['k'])
            assert np.all(np.isfinite(modes.metadata['k']))
        else:
            model_cls = {"Bellhop": Bellhop, "Scooter": Scooter}[downstream]
            result = model_cls(verbose=False).compute_tl(
                env=env_with_rc, source=source, receiver=receiver_small,
            )
            assert result.field_type == 'tl'
            assert result.shape == (
                len(receiver_small.depths), len(receiver_small.ranges)
            )
            assert np.all(np.isfinite(result.data))
