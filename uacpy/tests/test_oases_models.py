"""
Tests for OASES models (OAST, OASN, OASR, OASP)

These tests provide lightweight smoke coverage of the OASES family.
Deeper per-model tests (instantiation, TL/modes/reflection/PE workflows,
cross-model comparisons) live in ``test_oases_comprehensive.py``.
"""

import pytest
import numpy as np

pytestmark = pytest.mark.requires_oases

from uacpy.models import OAST, OASN, OASR, OASP


class TestOAST:
    """Lightweight OAST smoke test covering the basic run path."""

    def test_oast_basic_run(self, simple_env, source, receiver_small):
        """Test basic OAST transmission loss computation"""
        try:
            oast = OAST(verbose=False)
            result = oast.run(simple_env, source, receiver_small)

            # Check result structure
            assert result is not None, "OAST returned None"
            assert hasattr(result, 'data'), "Result missing data attribute"
            assert hasattr(result, 'field_type'), "Result missing field_type"

            # Check data dimensions
            assert result.data.shape[0] == len(receiver_small.depths), \
                f"Wrong depth dimension: {result.data.shape[0]} vs {len(receiver_small.depths)}"
            assert result.data.shape[1] == len(receiver_small.ranges), \
                f"Wrong range dimension: {result.data.shape[1]} vs {len(receiver_small.ranges)}"

            # Check TL values are reasonable
            assert np.all(np.isfinite(result.data)), "OAST produced non-finite values"
            assert np.all(result.data >= 0), "OAST produced negative TL values"
            assert np.all(result.data <= 200), "OAST produced unreasonably high TL values"

        except FileNotFoundError as e:
            pytest.skip(f"OAST not available: {e}")

    @pytest.mark.requires_binary
    def test_oast_range_dependent_warning(self, range_dependent_env, source, receiver_small):
        """Test that OAST warns about range-dependent environments"""
        try:
            oast = OAST(verbose=False)

            with pytest.warns(UserWarning, match="does not support range-dependent"):
                result = oast.run(range_dependent_env, source, receiver_small)

            # Should still produce results (using approximation)
            assert result is not None, "OAST failed with range-dependent environment"

        except FileNotFoundError:
            pytest.skip("OAST not available")


def test_all_oases_models_available():
    """Summary test reporting which OASES models are compiled and available."""
    available = []

    for name, cls in [('OAST', OAST), ('OASN', OASN), ('OASR', OASR), ('OASP', OASP)]:
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
