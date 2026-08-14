"""
OASES wrapper edge cases not covered in test_oases_comprehensive.py.

Per-model smoke (TL, modes, reflection, PE) and cross-model comparisons all
live in test_oases_comprehensive.py. This file holds the bits unique to the
wrapper layer: warning behavior on unsupported configs, and a one-liner
availability summary.
"""

import numpy as np
import pytest

from uacpy.core.exceptions import UnsupportedFeatureError

import uacpy
from uacpy.models import OAST, OASN, OASR, OASP
from uacpy.models.base import RunMode
from uacpy.core.exceptions import ConfigurationError

pytestmark = pytest.mark.requires_oases


@pytest.mark.requires_binary
class TestOASTWarnings:
    """Tests that the OAST wrapper warns on unsupported inputs."""

    # The off-grid dB-interpolation warning (asserted on its own below) also
    # fires here incidentally; silence it so only the RD warning is under test.
    # The leading '.' matches the literal colon in "OAST:".
    @pytest.mark.filterwarnings(
        "ignore:OAST. receiver.ranges do not match:UserWarning")
    def test_oast_range_dependent_warning(
        self, range_dependent_env, source, receiver_small
    ):
        """OAST warns (and approximates) on range-dependent environments."""
        oast = OAST(verbose=False)

        with pytest.warns(UserWarning, match="does not support range-dependent"):
            result = oast.run(range_dependent_env, source, receiver_small)

        assert result is not None, "OAST failed with range-dependent environment"

    def test_oast_off_grid_range_interpolation_warning(self):
        """OAST warns when receiver ranges miss its native FFT grid and TL is
        therefore interpolated in dB (which smears nulls)."""
        env = uacpy.Environment(
            name='oast_offgrid', bathymetry=100.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1600.0,
                density=1.5, attenuation=0.5))
        src = uacpy.Source(depths=50.0, frequencies=100.0)
        # Deliberately off the internal FFT range grid.
        rcv = uacpy.Receiver(depths=[50.0], ranges=[1234.0, 4321.0])
        oast = OAST(verbose=False)
        with pytest.warns(UserWarning, match="receiver.ranges do not match"):
            oast.run(env, src, rcv)


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


@pytest.mark.requires_binary
class TestRawOptionsVoidTypedKnobs:
    """A raw ``options`` string replaces the whole option line, so a typed
    knob passed alongside it never reaches the deck. Dropping ``'J'`` this
    way silently switches OASES off the complex integration contour."""

    def test_oast_rejects_options_with_a_typed_flag(self):
        with pytest.raises(ConfigurationError, match="complex_contour"):
            OAST(options='N T', complex_contour=True)
        with pytest.raises(ConfigurationError, match="compute_contour"):
            OAST(options='N T', compute_contour=True)

    def test_oast_typed_flags_still_derive_the_option_line(self):
        assert OAST()._resolve_options() == 'N T J'
        assert OAST(complex_contour=False)._resolve_options() == 'N T'
        assert OAST(compute_contour=True,
                    compute_depth_average=True)._resolve_options() == 'N T J C A'
        assert OAST(options='N T')._resolve_options() == 'N T'

    def test_oast_copy_round_trips_the_option_line(self):
        for m in (OAST(), OAST(options='N T'), OAST(complex_contour=False)):
            assert m.copy()._resolve_options() == m._resolve_options()

    def test_oasr_rejects_options_with_reflection_type(self):
        with pytest.raises(ConfigurationError, match="reflection_type"):
            OASR(options='S T', reflection_type='P-P')

    def test_oasr_provenance_follows_the_option_letters(self):
        """``options='S T'`` runs P-SV; recording 'P-P' would make the
        metadata state something the deck never computed."""
        # 'S' returns an all-zero coefficient (the incident medium is the
        # fluid water column), so the raw path warns while still writing the
        # deck verbatim; the named path refuses outright.
        with pytest.warns(UserWarning, match='column of zeros'):
            assert OASR(options='S T')._resolve_reflection_type() == 'P-SV'
        assert OASR(options='t T')._resolve_reflection_type() == 'transmission'
        assert OASR()._resolve_reflection_type() == 'P-P'
        with pytest.raises(UnsupportedFeatureError, match='zeros'):
            OASR(reflection_type='P-SV')
        with pytest.warns(UserWarning, match='column of zeros'):
            assert OASR(options='S T').copy()._resolve_reflection_type() == 'P-SV'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
