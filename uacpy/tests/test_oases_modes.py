"""
Tests for OASES normal mode computation

Tests the experimental OASN normal mode computation feature.
"""

import pytest
import numpy as np

pytestmark = pytest.mark.requires_oases

from uacpy import Environment, Source, Receiver
from uacpy.models.oases import OASN


@pytest.fixture
def simple_env():
    """Simple isovelocity environment for testing."""
    return Environment(
        name="Test Environment",
        depth=100.0,
        sound_speed=1500.0
    )


@pytest.fixture
def source_100hz():
    """100 Hz source at 50m depth."""
    return Source(depth=50.0, frequency=100.0)


@pytest.fixture
def receiver_array():
    """Receiver array for covariance computation."""
    depths = np.array([30.0, 50.0, 70.0])
    ranges = np.array([0.0])  # Not used for mode computation
    return Receiver(depths=depths, ranges=ranges)


class TestOASNInitialization:
    """Test OASN model initialization."""

    def test_oasn_instantiation(self):
        """Test that OASN can be instantiated."""
        try:
            oasn = OASN()
            assert oasn is not None
            assert hasattr(oasn, 'executable')
        except FileNotFoundError:
            pytest.skip("OASN executable not found")

    def test_oasn_executable_detection(self):
        """Test OASN executable detection."""
        try:
            oasn = OASN()
            # Check if oasn2_bin was found
            if hasattr(oasn, '_oasn_available'):
                if oasn._oasn_available:
                    assert oasn.oasn_executable is not None
                    assert oasn.oasn_executable.exists()
        except FileNotFoundError:
            pytest.skip("OASES executables not found")


class TestOASNErrorHandling:
    """Test OASN error handling."""

    def test_oasn_missing_executable(self):
        """Test error when OASN executable not found."""
        # Force executable path to non-existent file
        try:
            from pathlib import Path
            oasn = OASN(executable=Path("/nonexistent/oast"))
        except FileNotFoundError as e:
            assert "OASN executable not found" in str(e)

    @pytest.mark.requires_binary
    def test_oasn_requires_executable(self, simple_env, source_100hz):
        """Test that OASN requires oasn2_bin executable."""
        # OASN now directly requires oasn2_bin in __init__
        # If it doesn't exist, __init__ will raise FileNotFoundError
        try:
            oasn = OASN()
            # If we get here, executable exists
            assert oasn.executable.exists()
        except FileNotFoundError as e:
            assert "oasn2_bin" in str(e) or "OASN" in str(e)
            pytest.skip("OASN executable not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
