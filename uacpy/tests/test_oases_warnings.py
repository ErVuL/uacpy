"""
OASES wrapper edge cases not covered in test_oases_comprehensive.py.

Per-model smoke (TL, modes, reflection, PE) and cross-model comparisons all
live in test_oases_comprehensive.py. This file holds the bits unique to the
wrapper layer: warning behavior on unsupported configs, and a one-liner
availability summary.
"""

import pytest

from uacpy.models import OAST, OASN, OASR, OASP

pytestmark = pytest.mark.requires_oases


@pytest.mark.requires_binary
class TestOASTWarnings:
    """Tests that the OAST wrapper warns on unsupported inputs."""

    def test_oast_range_dependent_warning(
        self, range_dependent_env, source, receiver_small
    ):
        """OAST warns (and approximates) on range-dependent environments."""
        oast = OAST(verbose=False)

        with pytest.warns(UserWarning, match="does not support range-dependent"):
            result = oast.run(range_dependent_env, source, receiver_small)

        assert result is not None, "OAST failed with range-dependent environment"


def test_all_oases_models_available():
    """Summary test reporting which OASES models are compiled and available."""
    available = []
    for name, cls in [("OAST", OAST), ("OASN", OASN), ("OASR", OASR), ("OASP", OASP)]:
        try:
            cls()
            available.append(name)
        except Exception:
            pass
    if not available:
        pytest.skip("No OASES models compiled")
    assert len(available) > 0, "At least one OASES model should be available"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
