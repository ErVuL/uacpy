"""
Tests for all UACPY propagation models
"""

import pytest
import numpy as np
import uacpy
from uacpy.models import (
    Bellhop, BellhopCUDA, RAM, Kraken, KrakenField,
    Bounce, Scooter, SPARC, OASN, OASR, OASP
)
from uacpy.models.base import RunMode
from uacpy.core.receiver import Receiver


class TestBellhop:
    """Tests for Bellhop model."""

    @pytest.mark.requires_binary
    def test_bellhop_instantiation(self):
        """Test creating Bellhop instance."""
        bellhop = Bellhop(verbose=False)
        assert bellhop.model_name == 'Bellhop'
        assert not bellhop.verbose

    @pytest.mark.requires_binary
    def test_bellhop_supported_modes(self):
        """Test Bellhop supported modes."""
        bellhop = Bellhop(verbose=False)
        assert bellhop.supports_mode(RunMode.COHERENT_TL)
        assert bellhop.supports_mode(RunMode.RAYS)
        assert bellhop.supports_mode(RunMode.ARRIVALS)
        assert not bellhop.supports_mode(RunMode.MODES)

    @pytest.mark.requires_binary
    def test_bellhop_compute_tl(self, simple_env, source, receiver_small):
        """Test Bellhop TL computation."""
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=simple_env, source=source, receiver=receiver_small)

        assert result.field_type == 'tl'
        assert result.shape == (len(receiver_small.depths), len(receiver_small.ranges))
        assert np.all(np.isfinite(result.data))
        assert np.all(result.data > 0)  # TL should be positive

    @pytest.mark.requires_binary
    def test_bellhop_auto_receiver_grid(self, simple_env, source):
        """Test Bellhop with automatic receiver grid generation."""
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=simple_env, source=source, max_range=5000)

        assert result.field_type == 'tl'
        assert result.n_depths > 0
        assert result.n_ranges > 0

    @pytest.mark.requires_binary
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


class TestBellhopCUDA:
    """Tests for BellhopCUDA model.

    The wrapper auto-picks bellhopcuda (GPU) or bellhopcxx (CPU) — whichever
    install.sh built — so these tests run unchanged on GPU and CPU machines.
    """

    @pytest.mark.requires_binary
    def test_bellhopcuda_instantiation(self):
        """Test creating BellhopCUDA instance."""
        bhcuda = BellhopCUDA(verbose=False)
        assert bhcuda.model_name in ('BellhopCUDA', 'bellhopcuda')
        assert not bhcuda.verbose

    @pytest.mark.requires_binary
    def test_bellhopcuda_supported_modes(self):
        """BellhopCUDA should support the same run modes Bellhop does (TL/rays/arrivals)."""
        bhcuda = BellhopCUDA(verbose=False)
        assert bhcuda.supports_mode(RunMode.COHERENT_TL)
        # BellhopCUDA mirrors Bellhop — it does not compute normal modes
        assert not bhcuda.supports_mode(RunMode.MODES)


class TestKraken:
    """Tests for Kraken model."""

    @pytest.mark.requires_binary
    def test_kraken_instantiation(self):
        """Test creating Kraken instance."""
        kraken = Kraken(verbose=False)
        assert kraken.model_name == 'Kraken'

    @pytest.mark.requires_binary
    def test_kraken_supported_modes(self):
        """Test Kraken supported modes."""
        kraken = Kraken(verbose=False)
        assert kraken.supports_mode(RunMode.MODES)
        # Kraken computes modes only, not TL fields - use KrakenField for that
        assert not kraken.supports_mode(RunMode.COHERENT_TL)
        assert not kraken.supports_mode(RunMode.RAYS)

    @pytest.mark.requires_binary
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

    @pytest.mark.requires_binary
    def test_kraken_n_modes_clips_output(self, simple_env, source):
        """``n_modes`` caps the number of returned modes from Kraken."""
        kraken = Kraken(verbose=False)
        capped = kraken.compute_modes(env=simple_env, source=source, n_modes=3)
        assert len(capped.metadata['k']) <= 3
        assert capped.metadata.get('n_modes_requested') == 3

    @pytest.mark.requires_binary
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


class TestKrakenField:
    """Tests for KrakenField model."""

    @pytest.mark.requires_binary
    def test_krakenfield_instantiation(self):
        """Test creating KrakenField instance."""
        kf = KrakenField(verbose=False)
        assert kf.model_name == 'KrakenField'

    @pytest.mark.requires_binary
    def test_krakenfield_supported_modes(self):
        """Test KrakenField supported modes."""
        kf = KrakenField(verbose=False)
        assert kf.supports_mode(RunMode.COHERENT_TL)
        # KrakenField computes TL fields from modes
        assert not kf.supports_mode(RunMode.MODES)  # Use Kraken for mode computation
        assert not kf.supports_mode(RunMode.RAYS)

    @pytest.mark.requires_binary
    def test_krakenfield_compute_tl(self, simple_env, source, receiver_small):
        """Test KrakenField TL computation."""
        kf = KrakenField(verbose=False)
        result = kf.compute_tl(env=simple_env, source=source, receiver=receiver_small)

        assert result.field_type == 'tl'
        assert result.shape == (len(receiver_small.depths), len(receiver_small.ranges))


class TestBounce:
    """Tests for Bounce model."""

    @pytest.mark.requires_binary
    def test_bounce_instantiation(self):
        """Test creating Bounce instance."""
        bounce = Bounce(verbose=False)
        assert bounce.model_name == 'Bounce'

    @pytest.mark.requires_binary
    def test_bounce_supported_modes(self):
        """Test Bounce supported modes."""
        bounce = Bounce(verbose=False)
        # Bounce computes reflection coefficients, not TL or modes
        assert bounce.supports_mode(RunMode.REFLECTION)
        assert not bounce.supports_mode(RunMode.COHERENT_TL)
        assert not bounce.supports_mode(RunMode.RAYS)
        assert not bounce.supports_mode(RunMode.MODES)

    @pytest.mark.requires_binary
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
            depth=simple_env.depth,
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


class TestRAM:
    """Tests for RAM model (mpiramS backend)."""

    @pytest.mark.requires_binary
    def test_ram_instantiation(self):
        """Test creating RAM instance."""
        ram = RAM(verbose=False)
        assert ram.model_name == 'RAM'

    @pytest.mark.requires_binary
    def test_ram_supported_modes(self):
        """Test RAM supported modes."""
        ram = RAM(verbose=False)
        assert ram.supports_mode(RunMode.COHERENT_TL)
        assert ram.supports_mode(RunMode.BROADBAND)
        assert ram.supports_mode(RunMode.BROADBAND)
        assert ram.supports_mode(RunMode.TIME_SERIES)
        assert not ram.supports_mode(RunMode.RAYS)
        assert not ram.supports_mode(RunMode.MODES)

    @pytest.mark.requires_binary
    def test_ram_compute_tl(self, simple_env, source, receiver_small):
        """Test RAM TL computation."""
        ram = RAM(verbose=False)
        result = ram.compute_tl(env=simple_env, source=source, receiver=receiver_small)

        assert result.field_type == 'tl'
        assert result.shape[0] > 0  # Has depth dimension
        assert result.shape[1] > 0  # Has range dimension
        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
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

    @pytest.mark.requires_binary
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


class TestScooter:
    """Tests for Scooter model."""

    @pytest.mark.requires_binary
    def test_scooter_instantiation(self):
        """Test creating Scooter instance."""
        scooter = Scooter(verbose=False)
        assert scooter.model_name == 'Scooter'


class TestSPARC:
    """Tests for SPARC model."""

    @pytest.mark.requires_binary
    def test_sparc_instantiation(self):
        """Test creating SPARC instance."""
        sparc = SPARC(verbose=False)
        assert sparc.model_name == 'SPARC'


# OASES instantiation/supported-mode tests live in test_oases_comprehensive.py;
# the cross-model workflow tests below cover Bounce → {Bellhop, Scooter, KrakenC}.


class TestModelConsistency:
    """Tests for consistency between different models."""

    @pytest.mark.requires_binary
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

    @pytest.mark.requires_binary
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
            depth=simple_env.depth,
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
            depth=100,
            sound_speed=1600,
            density=1.8,
            reflection_cmin=bounce_result.metadata['cmin'],
            reflection_cmax=bounce_result.metadata['cmax'],
            reflection_rmax_km=bounce_result.metadata['rmax_km'],
        )
        env_with_rc = Environment(
            name="test_with_rc",
            depth=simple_env.depth,
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
