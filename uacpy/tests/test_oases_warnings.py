"""
OASES wrapper edge cases not covered in test_oases_comprehensive.py.

Per-model smoke (TL, modes, reflection, PE) and cross-model comparisons all
live in test_oases_comprehensive.py. This file holds the bits unique to the
wrapper layer: warning behavior on unsupported configs, and a one-liner
availability summary.
"""

import numpy as np
import pytest

import uacpy
from uacpy.models import OAST, OASN, OASR, OASP
from uacpy.models.base import RunMode
from uacpy.core.exceptions import ConfigurationError

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


@pytest.mark.requires_binary
class TestOASPOptionGuard:
    """OASP rejects option letters its .trf path can't honour. OASES GETOPT
    parses the option line character-by-character, so the guard must match
    on characters — ``'NJO'`` enables 'O' exactly like ``'N J O'``."""

    @staticmethod
    def _run(options):
        env = uacpy.Environment(bathymetry=100.0, ssp=1500.0)
        src = uacpy.Source(depths=20.0, frequencies=100.0)
        rcv = uacpy.Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))
        # Pass explicit frequencies= so the option guard (which raises before
        # any solver runs) isn't preceded by the auto-derived-frequencies
        # warning — keeps this guard test free of incidental warnings.
        OASP(options=options).run(
            env, src, rcv, run_mode=RunMode.TIME_SERIES,
            source_waveform=np.hanning(64), sample_rate=2000.0,
            frequencies=np.array([40.0, 60.0]),
        )

    @pytest.mark.parametrize("options", ['O', 'N J O', 'NJO'])
    def test_option_O_rejected_regardless_of_spacing(self, options):
        with pytest.raises(ConfigurationError, match="option 'O'"):
            self._run(options)

    @pytest.mark.parametrize("options", ['V', 'N J V', 'NJV'])
    def test_multi_axis_option_rejected_regardless_of_spacing(self, options):
        with pytest.raises(ConfigurationError, match="multi-component"):
            self._run(options)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
