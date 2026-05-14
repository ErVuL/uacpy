"""
Tests specific to OASN executable resolution and error paths.

OASN instantiation and run-time tests (covariance + replicas) live in
test_oases_comprehensive.py; this file covers only the things unique to
the OASN wrapper:
  - the missing-executable error message
  - the ``_oasn_available`` probe used by the wrapper to decide whether
    to expose its hydrophone-array products.
"""

from pathlib import Path

import pytest

from uacpy.core.exceptions import ExecutableNotFoundError
from uacpy.models.oases import OASN

pytestmark = pytest.mark.requires_oases


class TestOASNExecutableResolution:
    """OASN-specific executable-path handling."""

    @pytest.mark.requires_binary
    def test_oasn_executable_detection(self):
        """If OASN reports availability, the resolved path must actually exist."""
        oasn = OASN()
        if hasattr(oasn, "_oasn_available") and oasn._oasn_available:
            assert oasn.oasn_executable is not None
            assert oasn.oasn_executable.exists()

    def test_oasn_missing_executable(self):
        """Pointing OASN at a nonexistent path raises ExecutableNotFoundError."""
        with pytest.raises(ExecutableNotFoundError, match="OASN executable not found"):
            OASN(executable=Path("/nonexistent/oast"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
