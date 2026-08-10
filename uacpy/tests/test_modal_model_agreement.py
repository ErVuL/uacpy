"""
Test that modal models (Kraken, Scooter, OAST) produce similar results.

They do not share a method — Kraken sums normal modes while Scooter and OAST
integrate the wavenumber spectrum. They agree because on a range-independent
environment the modal sum is the residue series of that same integral, closed
in the upper half-plane. The equivalence is not perfect: the contour also
carries a branch-line integral for the components radiating into the bottom,
which a normal-mode code drops, so the two diverge in the near field and
converge as that contribution dies away with range. Range dependence breaks the
equivalence outright, which is why these comparisons are confined to a
range-independent Pekeris guide sampled from 1 km out.
"""

import pytest
import numpy as np

import uacpy
from uacpy.models import Kraken, Scooter, OAST

# All tests in this module spawn model binaries (Kraken, Scooter, OAST)
pytestmark = pytest.mark.requires_binary


class TestModalModelAgreement:
    """Test agreement between different modal models."""

    @pytest.fixture
    def simple_environment(self):
        """Pekeris waveguide with a fluid half-space bottom.

        A fluid half-space rather than the simpler vacuum bottom: the leaky
        branch-line contribution that Kraken drops decays with range only if
        there is a bottom to radiate into. Against a vacuum bottom the trapped
        modes are lossless, nothing damps the disagreement, and the inter-model
        spread stays wide at every range.
        """
        env = uacpy.Environment(
            name='Pekeris',
            bathymetry=100,
            ssp=1500,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space',
                sound_speed=1700.0,
                density=1.8,
                attenuation=0.5,
            ),
        )
        return env

    @pytest.fixture
    def simple_source(self):
        """Create a simple source."""
        return uacpy.Source(depths=[50], frequencies=[100])

    @pytest.fixture
    def single_receiver(self):
        """Create a single receiver at 50m depth, 5km range."""
        return uacpy.Receiver(
            depths=np.array([50]),
            ranges=np.array([5000])
        )

    @pytest.fixture
    def multi_range_receiver(self):
        """Create receivers at multiple ranges."""
        return uacpy.Receiver(
            depths=np.array([50]),
            ranges=np.array([1000, 2000, 3000, 5000, 7000, 10000])
        )

    def test_kraken_modes_valid(self, simple_environment, simple_source):
        """Test that Kraken computes valid modes (wavenumbers non-zero)."""
        # Compute modes (compute_modes derives its own mode-depth grid)
        kraken = Kraken(verbose=False)
        result = kraken.compute_modes(simple_environment, simple_source)

        k = result.k

        # Check that we have modes
        assert len(k) > 0, "No modes computed"

        # Mode 1 must have non-zero wavenumber.
        assert np.abs(k[0]) > 0.1, f"Mode 1 wavenumber is zero or near-zero: {k[0]}"

        # Count valid modes (non-zero with non-positive imaginary part)
        n_valid = sum(1 for k_val in k if np.abs(k_val) >= 1e-10 and np.imag(k_val) <= 0)

        # An ideal (doubly pressure-release) guide holds 2*f*D/c1 half
        # wavelengths across the water column — 13.3 at 100 Hz in 100 m, which
        # Abraham (*Underwater Acoustic Signal Processing*, §3) quotes as 13 for
        # exactly these numbers. A Pekeris bottom traps only the modes steeper
        # than the critical angle, scaling that by
        # sqrt(1 - (c1/c2)^2) = sqrt(1 - (1500/1700)^2) = 0.471, so 6 modes.
        # The +/-1 band admits the marginally-trapped mode nearest cutoff,
        # whose capture depends on the mesh.
        assert n_valid >= 5, f"Expected at least 5 valid modes, got {n_valid}"
        assert n_valid <= 7, f"Expected at most 7 valid modes, got {n_valid}"

    def test_kraken_vs_scooter_single_point(
        self, simple_environment, simple_source, single_receiver
    ):
        """Kraken vs Scooter at a single point.

        These are smoke bounds, not accuracy statements. The two solve the same
        range-independent problem exactly and agree to ~0.1 dB in practice
        (``test_cross_model_agreement.py`` carries the tight RMSE numbers); 5 dB
        here is roughly fifty times looser, sized to survive one solver landing
        on the far side of a deep interference null while still catching a
        wrong deck, a dropped mode set or a units error.
        """
        kf = Kraken(verbose=False)
        kf_result = kf.run(simple_environment, simple_source, single_receiver)
        kf_tl = kf_result.tl[0, 0]

        # Scooter
        scooter = Scooter(verbose=False)
        scooter_result = scooter.run(simple_environment, simple_source, single_receiver)
        scooter_tl = scooter_result.tl[0, 0]

        # Check agreement
        diff = np.abs(kf_tl - scooter_tl)

        assert diff < 5.0, (
            f"Kraken and Scooter disagree by {diff:.2f} dB "
            f"(KF={kf_tl:.2f}, Scooter={scooter_tl:.2f}). "
            "Modal models should agree within 5 dB."
        )

    def test_kraken_vs_scooter_multiple_ranges(
        self, simple_environment, simple_source, multi_range_receiver
    ):
        """Kraken vs Scooter across multiple ranges.

        Averaging over six ranges dilutes any single null misalignment, so the
        mean bound tightens to 2 dB while the per-range peak keeps the 5 dB of
        the single-point check.
        """
        kf = Kraken(verbose=False)
        kf_result = kf.run(simple_environment, simple_source, multi_range_receiver)
        kf_tl = kf_result.tl[0, :]

        # Scooter
        scooter = Scooter(verbose=False)
        scooter_result = scooter.run(simple_environment, simple_source, multi_range_receiver)
        scooter_tl = scooter_result.tl[0, :]

        # Check agreement at each range
        diffs = np.abs(kf_tl - scooter_tl)
        mean_diff = np.mean(diffs)
        max_diff = np.max(diffs)

        assert mean_diff < 2.0, (
            f"Kraken and Scooter mean difference is {mean_diff:.2f} dB. "
            "Modal models should agree with mean difference < 2 dB."
        )

        assert max_diff < 5.0, (
            f"Kraken and Scooter max difference is {max_diff:.2f} dB. "
            "Modal models should agree with max difference < 5 dB."
        )

    @pytest.mark.requires_oases
    @pytest.mark.filterwarnings(
        "ignore:OAST. receiver.ranges do not match:UserWarning")
    def test_all_modal_models_agreement(
        self, simple_environment, simple_source, single_receiver
    ):
        """Test that all modal models (Kraken, Scooter, OAST) agree."""
        # Run all models
        kf = Kraken(verbose=False)
        kf_result = kf.run(simple_environment, simple_source, single_receiver)
        kf_tl = kf_result.tl[0, 0]

        scooter = Scooter(verbose=False)
        scooter_result = scooter.run(simple_environment, simple_source, single_receiver)
        scooter_tl = scooter_result.tl[0, 0]

        oast = OAST(verbose=False)
        oast_result = oast.run(simple_environment, simple_source, single_receiver)
        oast_tl = oast_result.tl[0, 0]

        # Compute pairwise differences
        kf_scooter_diff = np.abs(kf_tl - scooter_tl)
        kf_oast_diff = np.abs(kf_tl - oast_tl)
        scooter_oast_diff = np.abs(scooter_tl - oast_tl)

        # All models should be within reasonable range of each other
        max_diff = max(kf_scooter_diff, kf_oast_diff, scooter_oast_diff)

        assert max_diff < 2.0, (
            f"Modal models disagree on Pekeris waveguide:\n"
            f"  Kraken:      {kf_tl:.2f} dB\n"
            f"  Scooter:     {scooter_tl:.2f} dB\n"
            f"  OAST:        {oast_tl:.2f} dB\n"
            f"  Max difference: {max_diff:.2f} dB\n"
            "All modal models should agree within 2 dB on a fluid-bottom "
            "Pekeris waveguide."
        )

    def test_mode_count_consistency(self, simple_environment, simple_source):
        """Test that mode count is consistent for the environment.

        ``mode_depth_grid`` is only where the eigenfunctions get tabulated —
        the eigenvalue search runs on Kraken's own internal mesh — so the count
        it returns must not move with the output resolution.
        """
        resolutions = [100, 150, 200]
        mode_counts = []

        for n_points in resolutions:
            # 0.999 keeps the deepest tabulation point strictly inside the water
            # column; landing exactly on the seafloor puts it on the interface,
            # where the mode shape is continuous but the medium is not.
            mode_depths = np.linspace(0, simple_environment.depth * 0.999, n_points)
            kraken = Kraken(mode_depth_grid=mode_depths, verbose=False)
            result = kraken.compute_modes(simple_environment, simple_source)

            k = result.k
            n_valid = sum(1 for k_val in k if np.abs(k_val) >= 1e-10 and np.imag(k_val) <= 0)
            mode_counts.append(n_valid)

        # Mode count must be stable across resolutions to within ±1 — a single
        # marginally-trapped mode whose cutoff lies between two grids is
        # allowed to flip in or out, but anything more is a real instability.
        assert max(mode_counts) - min(mode_counts) <= 1, (
            f"Mode count varies more than ±1 across resolutions: {mode_counts}. "
            "Should be stable to within one marginally-trapped mode."
        )
