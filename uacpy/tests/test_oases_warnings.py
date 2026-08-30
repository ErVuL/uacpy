"""
OASES wrapper edge cases not covered in test_oases_comprehensive.py.

This file holds the bits unique to the wrapper layer: warnings on
unsupported, approximated or off-grid configurations, the option-line
guards, and a one-liner availability summary.
"""

import numpy as np
import pytest

from uacpy.core.exceptions import UnsupportedFeatureError

import uacpy
from uacpy.models import OAST, OASN, OASR, OASP
from uacpy.models.base import RunMode
from uacpy.core.exceptions import ConfigurationError
from uacpy.tests.conftest import make_pekeris

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
            result = oast.run(env, src, rcv)
        # The escape hatch the warning points at (oases.md §11): the native
        # FFT grid rides back on the result so the caller can align to it.
        native = np.asarray(result.metadata['oast_native_ranges'], dtype=float)
        assert native.ndim == 1 and native.size > 1
        assert np.all(np.diff(native) > 0)
        assert result.metadata['interpolated'] is True
        # The off-grid requests fall inside the native hull, or they would
        # have come back NaN rather than interpolated.
        assert native.min() <= 1234.0 and native.max() >= 4321.0


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

    @pytest.mark.parametrize("options", ['N', 'N T', 'NT'])
    def test_custom_options_without_J_under_auto_sampling_raise(self, options):
        """oases.md §11: dropping 'J' under automatic wavenumber sampling
        enables the complex frequency contour (OMEGIM ≠ 0) by the back
        door — the same offset 'O' applies — which the time-series
        synthesis cannot undo. Only the OASSP twin of this guard was
        pinned before."""
        with pytest.raises(ConfigurationError, match=r"without 'J'"):
            self._run(options)

    def test_without_J_but_pinned_sampling_passes_the_guard(self):
        """nw_samples >= 1 disables AUSAMP, so the contour never engages and
        the same option line is legal."""
        assert OASP(options='N', nw_samples=2048)._reject_unreadable_options() \
            is None


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

    def test_oast_typed_flags_derive_the_option_line(self):
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


@pytest.mark.requires_binary
class TestOASPNearestBinSubstitution:
    """OASP's COHERENT_TL Field carries the FFT-ladder bin nearest the
    request, and the ladder rarely lands a bin exactly on it (measured:
    100 Hz asked, 99.97558 Hz returned, a phase error growing to ~36 deg by
    6 km) — so the substitution warns, just as the BROADBAND branch warns
    about its replaced axis."""

    def test_coherent_tl_names_the_substituted_bin(self):
        with pytest.warns(UserWarning, match='nearest bin'):
            field = OASP(verbose=False).run(
                make_pekeris(), uacpy.Source(depths=50.0, frequencies=100.0),
                uacpy.Receiver(depths=[50.0], ranges=[500.0, 1000.0]))
        f = float(np.atleast_1d(field.frequencies)[0])
        assert f != 100.0
        assert abs(f - 100.0) < 0.5    # nearest bin, not a different band


@pytest.mark.requires_binary
class TestOASNDegenerateCovarianceWarns:
    """A COVARIANCE run with no Block VI source returns only the -200 dB
    white-noise floor (diagonal 1e-20, zero cross terms), so it warns and
    names the noise knobs."""

    _RCV = dict(depths=np.array([30.0, 50.0, 70.0]), ranges=np.array([0.0]))

    def test_default_covariance_names_the_noise_knobs(self):
        with pytest.warns(UserWarning, match='no noise source'):
            cov = OASN(verbose=False).run(
                make_pekeris(), uacpy.Source(depths=50.0, frequencies=100.0),
                uacpy.Receiver(**self._RCV), RunMode.COVARIANCE)
        C = np.asarray(cov.covariance)
        assert np.abs(np.diagonal(C[0])) == pytest.approx(1e-20)

    def test_configured_noise_field_is_silent(self, recwarn):
        OASN(verbose=False, surface_noise_level=40.0).run(
            make_pekeris(), uacpy.Source(depths=50.0, frequencies=100.0),
            uacpy.Receiver(**self._RCV), RunMode.COVARIANCE)
        assert not [w for w in recwarn.list
                    if 'no noise source' in str(w.message)]


@pytest.mark.requires_binary
def test_oast_names_ranges_outside_the_native_grid():
    """OAST's native FFT range grid does not start at r = 0, and dB
    interpolation cannot extrapolate below its first sample — the NaN
    columns come with a warning naming the native span and the offending
    ranges, not just the generic dB-interpolation notice."""
    with pytest.warns(UserWarning, match='outside the native FFT range grid'):
        tl = OAST(verbose=False).run(
            make_pekeris(), uacpy.Source(depths=50.0, frequencies=100.0),
            uacpy.Receiver(depths=[50.0], ranges=[0.0, 500.0, 1000.0]))
    assert np.isnan(np.asarray(tl.data)[:, 0]).all()
    assert np.isfinite(np.asarray(tl.data)[:, 1:]).all()


@pytest.mark.requires_binary
def test_oasr_surface_roughness_is_dropped_with_a_warning():
    """The OASR deck has no sea surface — its layer 1 is the water
    half-space the plane wave arrives through, whose RG INENVI discards
    (oaseun31.f:377) — so surface roughness is collapsed by the projection
    with the standard disclosure instead of being silently swallowed."""
    env = make_pekeris()
    env.surface.roughness = 1.0
    with pytest.warns(UserWarning, match='rough sea surface'):
        refl = OASR(verbose=False).run(
            env, uacpy.Source(depths=50.0, frequencies=100.0),
            uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
            RunMode.REFLECTION)
    assert refl is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


@pytest.mark.requires_binary
class TestOasesForwardsItsOwnStdoutWarnings:
    """OASES writes no ``.prt``, so its own non-fatal diagnostics reach the
    user only if uacpy reads stdout.

    The case driven here is the only check of its kind in the pipeline:
    ``oaseun31.f:204-207`` flags ``(cs/cp)**2 > 0.75`` on an elastic
    half-space, and nothing on the Python side tests that ratio — so if the
    line is dropped, an unphysical seabed returns a full TL field with
    nothing said.
    """

    @staticmethod
    def _env(shear_speed, shear_attenuation=0.5):
        return uacpy.Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0,
                shear_speed=shear_speed, density=2.0,
                attenuation=0.2, shear_attenuation=shear_attenuation))

    @staticmethod
    def _run(env):
        import warnings as _w
        model = OAST(verbose=False)
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter('always')
            field = model.run(env,
                              uacpy.Source(depths=25.0, frequencies=100.0),
                              uacpy.Receiver(depths=[50.0],
                                             ranges=[1000.0, 2000.0]))
        return field, [str(x.message) for x in caught
                       if 'on stdout' in str(x.message)]

    def test_an_unphysical_shear_ratio_is_reported_verbatim(self):
        """cs/cp = 1650/1800 gives (cs/cp)^2 = 0.840, above the 0.75 the
        binary refuses to call physical."""
        field, said = self._run(self._env(1650.0))
        assert len(said) == 1
        assert 'UNPHYSICAL SPEED RATIO' in said[0]
        assert np.isfinite(np.asarray(field.db)).all(), (
            "the run must still return a field — this warning is the only "
            "sign anything was wrong, so a failed run would not test it")

    def test_a_physical_shear_ratio_says_nothing(self):
        """The other side of the binary's own two 0.75 thresholds: cs/cp =
        1200/1800 gives (cs/cp)^2 = 0.444, and 0.444 x As = 0.044 stays under
        0.75 x Ap = 0.15, so neither line is emitted to forward."""
        _, said = self._run(self._env(1200.0, shear_attenuation=0.1))
        assert said == []

    def test_the_attenuation_threshold_is_pinned_on_its_own(self):
        """``oaseun31.f:208-210`` is a second, independent check —
        (As/Ap)(cs/cp)^2 > 0.75 — reachable with a perfectly physical speed
        ratio. 0.444 x 0.5 = 0.222 against 0.75 x 0.2 = 0.15."""
        _, said = self._run(self._env(1200.0, shear_attenuation=0.5))
        assert len(said) == 1
        assert 'UNPHYSICAL ATTENUATION' in said[0]

    def test_progress_traces_are_not_forwarded_as_warnings(self):
        """OASES prefixes ordinary traces with the same ``>>>``; only lines
        that say *warning* are passed on."""
        model = OAST(verbose=False)
        import warnings as _w
        import types
        chatter = types.SimpleNamespace(stdout=(
            " >>> Entering SOLDIS <<<\n"
            " >>> Done, CPU=  0.13\n"
            " >>> Max Bessel kr:  1234.5\n"))
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter('always')
            model._warn_on_stdout_warnings(chatter)
        assert not caught

    def test_a_warning_line_in_the_same_stream_is_forwarded(self):
        """...and one that does is, deduplicated."""
        model = OAST(verbose=False)
        import types
        stream = types.SimpleNamespace(stdout=(
            " >>> Entering SOLDIS <<<\n"
            " >>> WARNING: ARRAY OVERFLOW IN RINTPL, ARG=  2.0\n"
            " >>> WARNING: ARRAY OVERFLOW IN RINTPL, ARG=  2.0\n"))
        with pytest.warns(UserWarning, match='ARRAY OVERFLOW') as caught:
            model._warn_on_stdout_warnings(stream)
        assert 'reported 1 non-fatal' in str(caught[0].message)
