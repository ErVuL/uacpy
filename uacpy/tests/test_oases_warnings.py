"""
OASES wrapper edge cases not covered in test_oases_variants.py.

This file holds the bits unique to the wrapper layer: warnings on
unsupported, approximated or off-grid configurations, the option-line
guards, and a one-liner availability summary.
"""

import numpy as np
import pytest

from uacpy.core.exceptions import UnsupportedFeatureError

import uacpy
from uacpy.models import OAST, OASN, OASR, OASP
from uacpy.core.run_settings import RunMode
from uacpy.core.exceptions import ConfigurationError
from uacpy.tests.conftest import make_pekeris
from uacpy.tests.conftest import recorded_warnings

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

    def test_the_default_grid_interpolates_off_grid_ranges_silently(
            self, recwarn):
        """OASES-9: OAST's native FFT range step is ~half a wavelength on the
        default phase-speed window, where dB interpolation onto arbitrary
        receiver ranges moves TL by a median 0.01-0.02 dB, so an off-grid
        request is not warned about; the native grid rides back on the
        result for a caller who wants to align to it."""
        env = uacpy.Environment(
            name='oast_offgrid', bathymetry=100.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1600.0,
                density=1.5, attenuation=0.5))
        src = uacpy.Source(depths=50.0, frequencies=100.0)
        # Deliberately off the internal FFT range grid.
        rcv = uacpy.Receiver(depths=[50.0], ranges=[1234.0, 4321.0])
        result = OAST(verbose=False).run(env, src, rcv)
        assert not [w for w in recwarn.list
                    if 'native FFT range grid' in str(w.message)]
        native = np.asarray(result.metadata['native_ranges'], dtype=float)
        assert native.ndim == 1 and native.size > 1
        assert np.all(np.diff(native) > 0)
        assert np.diff(native).max() < 15.0     # λ = 15 m at 100 Hz
        assert result.metadata['interpolated'] is True
        # The off-grid requests fall inside the native hull, or they would
        # have come back NaN rather than interpolated.
        assert native.min() <= 1234.0 and native.max() >= 4321.0

    def test_a_narrow_phase_speed_window_warns_the_native_step_is_coarse(
            self):
        """A pinned count over a narrow window puts OAST's native samples
        ~10 wavelengths apart (2*pi over the window: 155 m at 100 Hz for
        1450-1600 m/s), where dB interpolation misplaces interference
        structure; the warning names the step and the remedies."""
        rcv = uacpy.Receiver(depths=[50.0], ranges=[1234.0, 4321.0])
        with pytest.warns(UserWarning, match=r'native FFT range grid, whose '
                                             r'step \(15\d\.\d m\)'):
            OAST(verbose=False, n_wavenumbers=4096, c_low=1450.0,
                 c_high=1600.0).run(
                make_pekeris(), uacpy.Source(depths=50.0, frequencies=100.0),
                rcv)


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
        """n_wavenumbers >= 1 disables AUSAMP, so the contour never engages and
        the same option line is legal."""
        from uacpy.models.oases.oasp import _reject_unreadable_oasp_options
        model = OASP(options='N', n_wavenumbers=2048)
        assert _reject_unreadable_oasp_options(model.options,
                                               model.n_wavenumbers) is None


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

    @staticmethod
    def _resolved(model):
        """The engine settings ``model`` resolves for a point source on a
        Pekeris guide."""
        return model.run_settings(
            make_pekeris(), uacpy.Source(depths=50.0, frequencies=100.0),
            uacpy.Receiver(depths=[30.0], ranges=[500.0, 1000.0])).engine

    def test_oast_typed_flags_derive_the_option_line(self):
        assert self._resolved(OAST()).options == 'N T J'
        assert self._resolved(OAST(complex_contour=False)).options == 'N T'
        assert self._resolved(OAST(
            compute_contour=True,
            compute_depth_average=True)).options == 'N T J C A'
        assert self._resolved(OAST(options='N T')).options == 'N T'

    def test_oast_copy_round_trips_the_option_line(self):
        for m in (OAST(), OAST(options='N T'), OAST(complex_contour=False)):
            assert (self._resolved(m.copy()).options
                    == self._resolved(m).options)

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
            assert self._resolved(
                OASR(options='S T')).reflection_type == 'P-SV'
        assert self._resolved(
            OASR(options='t T')).reflection_type == 'transmission'
        assert self._resolved(OASR()).reflection_type == 'P-P'
        with pytest.raises(UnsupportedFeatureError, match='zeros'):
            OASR(reflection_type='P-SV')
        with pytest.warns(UserWarning, match='column of zeros'):
            assert self._resolved(
                OASR(options='S T').copy()).reflection_type == 'P-SV'


@pytest.mark.requires_binary
class TestOASPNearestBinSubstitution:
    """OASP's COHERENT_TL Field carries a bin of its FFT ladder. OASES-1: the
    sweep is narrowed onto the source frequency (``FR1 = f``), so the bin
    lands on it; a ladder bin off the request (measured on the default
    sweep: 100 Hz asked, 99.97558 Hz returned, a phase error growing to ~36
    deg by 6 km) is named in a warning, as the BROADBAND branch names its
    replaced axis."""

    def test_coherent_tl_lands_a_bin_on_the_source_frequency_silently(self):
        with recorded_warnings() as caught:
            field = OASP(verbose=False).run(
                make_pekeris(), uacpy.Source(depths=50.0, frequencies=100.0),
                uacpy.Receiver(depths=[50.0], ranges=[500.0, 1000.0]))
        assert not [w for w in caught if 'nearest bin' in str(w.message)]
        f = float(np.atleast_1d(field.frequencies)[0])
        assert f == pytest.approx(100.0, rel=2e-6)

    def test_the_bin_lands_on_every_source_frequency_of_a_sweep(self):
        """20-2000 Hz in 5 Hz steps (397 frequencies): the one bin is the
        request to REAL*4 (measured |offset| <= 5.9e-8, 0.28 deg at 2 kHz
        and 10 km) and the substituted-bin notice stays silent. A 1e-6
        guard on the Nyquist put it 0.95-1.06e-6 above, past the notice's
        rtol, at 214 of them (20, 40, 65, 75, 80 Hz ...)."""
        model = OASP(verbose=False)
        env = make_pekeris()
        rcv = uacpy.Receiver(depths=[50.0], ranges=[500.0, 1000.0])
        offsets, noticed = [], []
        for f in np.arange(20.0, 2000.1, 5.0):
            with recorded_warnings() as caught:
                settings = model.run_settings(
                    env, uacpy.Source(depths=50.0, frequencies=float(f)), rcv)
            offsets.append(float(settings.frequencies[0]) / f - 1.0)
            if [w for w in caught if 'nearest bin' in str(w.message)]:
                noticed.append(f)
        assert noticed == []
        assert np.max(np.abs(offsets)) < 1.2e-7

    def test_a_substituted_bin_is_named(self):
        from uacpy.models.oases.oasp import _warn_if_trf_grid_replaced_request
        with pytest.warns(UserWarning, match='nearest bin, 99.97558 Hz'):
            _warn_if_trf_grid_replaced_request(100.0, 99.97558)


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
        model = OAST(verbose=False)
        with recorded_warnings() as caught:
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
        assert np.isfinite(np.asarray(field.dB)).all(), (
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
        import types
        chatter = types.SimpleNamespace(stdout=(
            " >>> Entering SOLDIS <<<\n"
            " >>> Done, CPU=  0.13\n"
            " >>> Max Bessel kr:  1234.5\n"))
        with recorded_warnings() as caught:
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


def _quiet_run(call, *args, **kwargs):
    """``call(*args, **kwargs)`` with the licence notice and the other
    warnings muted, for a test that asserts on what it returns or raises."""
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        return call(*args, **kwargs)


def _entry_points(model, *carriers, run_mode=None):
    """``validate_inputs``, ``run_settings`` and ``run`` of ``model`` on
    ``carriers``, as zero-argument calls."""
    return [lambda name=name: getattr(model, name)(*carriers, run_mode)
            for name in ('validate_inputs', 'run_settings', 'run')]


class TestTheCoarseRangeStepBoundary:
    """OASES-9: OAST warns about its dB interpolation only once the native
    range step passes one wavelength of the slowest water speed (15 m at
    1500 m/s and 100 Hz) — both sides of the boundary."""

    @staticmethod
    def _inputs():
        from types import SimpleNamespace
        return SimpleNamespace(
            source=uacpy.Source(depths=50.0, frequencies=100.0),
            env=uacpy.Environment(bathymetry=100.0, ssp=1500.0))

    @pytest.mark.parametrize('wavelengths,warns', [(0.99, False),
                                                   (1.01, True)])
    def test_the_warning_starts_one_wavelength_out(self, wavelengths, warns,
                                                   recwarn):
        native = 15.0 * wavelengths * np.arange(1, 6)
        from uacpy.models.oases.oast import (
            _warn_if_native_range_step_is_coarse)
        inputs = self._inputs()
        _warn_if_native_range_step_is_coarse(native, inputs.source,
                                              inputs.env)
        hits = [w for w in recwarn.list
                if 'native FFT range grid' in str(w.message)]
        assert bool(hits) is warns


class TestOASRWriterRefusesAnOptionLineItCannotRead:
    """OASES-12: OASR prints '>>> UNKNOWN OPTION' and carries on, and writes
    its tables only under 'T', so ``write_oasr_input`` refuses both before
    the binary runs, as every sibling writer refuses an unknown letter."""

    def _write(self, tmp_path, options):
        from uacpy.io.oases_writer import write_oasr_input
        write_oasr_input(tmp_path / 'oasr.dat', make_pekeris(),
                         uacpy.Source(depths=50.0, frequencies=100.0),
                         uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
                         options=options)

    def test_an_unknown_letter_is_refused(self, tmp_path):
        with pytest.raises(ConfigurationError,
                           match=r"\['x'\] are not option letters"):
            self._write(tmp_path, 'N T x')

    def test_a_line_without_the_table_letter_is_refused(self, tmp_path):
        with pytest.raises(ConfigurationError, match="has no 'T'"):
            self._write(tmp_path, 'N L')

    def test_the_letters_oasr_tests_are_written(self, tmp_path):
        self._write(tmp_path, 'N T L P')
        assert (tmp_path / 'oasr.dat').read_text().splitlines()[1] \
            == 'N T L P'


def test_an_oassp2_array_bound_stop_is_raised_with_its_own_words():
    """OASES-13: oassp2 prints '>>> NKMEAN too large in CALSRP<<<' and
    stops with exit 0 (oasvun31.f:62); the stdout scan raises it rather than
    leaving the reader to report a truncated .trf."""
    from types import SimpleNamespace
    from uacpy.core.exceptions import ModelExecutionError
    from uacpy.models.oases import OASES
    model = SimpleNamespace(model_name='OASSP',
                            _STDOUT_FATAL_MARKERS=OASES._STDOUT_FATAL_MARKERS)
    proc = SimpleNamespace(stdout=(" >>> Entering SOLDIS <<<\n"
                                   " >>> NKMEAN too large in CALSRP<<<\n"
                                   " nkr=        9000 max=        8192\n"))
    with pytest.raises(ModelExecutionError,
                       match='NKMEAN too large in CALSRP'):
        OASES._raise_on_stdout_fatal(model, proc)


@pytest.mark.requires_binary
class TestOASNRefusesTheDopplerLetters:
    """OASES-10: in OASN, 'd'/'D' apply a 15 m/s source/receiver Doppler no
    deck field can set (unoasn22.f:702-704, :762), so a raw option line
    carrying either is refused — at construction, and by every entry point
    when ``options`` is reassigned afterwards."""

    @pytest.mark.parametrize('options', ['J d', 'N D'])
    def test_the_constructor_refuses_them(self, options):
        with pytest.raises(ConfigurationError, match='15 m/s'):
            OASN(verbose=False, options=options)

    def test_every_entry_point_refuses_a_reassigned_letter(self):
        model = OASN(verbose=False)
        model.options = 'J d'
        carriers = (make_pekeris(),
                    uacpy.Source(depths=50.0, frequencies=100.0),
                    uacpy.Receiver(depths=[30.0, 70.0], ranges=[0.0]))
        for call in _entry_points(model, *carriers,
                                  run_mode=RunMode.REPLICA):
            with pytest.raises(ConfigurationError, match="Doppler flag"):
                _quiet_run(call)


@pytest.mark.requires_binary
class TestOASNSweepAndLevels:
    """OASES-8 / OASES-2 / OASES-5 on OASN."""

    _RCV = dict(depths=np.array([30.0, 70.0]), ranges=np.array([0.0]))

    def test_a_non_uniform_sweep_is_refused_without_a_resampling_notice(
            self):
        model = OASN(verbose=False, surface_noise_level=60.0)
        carriers = (make_pekeris(),
                    uacpy.Source(depths=50.0,
                                 frequencies=[100.0, 110.0, 130.0]),
                    uacpy.Receiver(**self._RCV))
        for call in _entry_points(model, *carriers,
                                  run_mode=RunMode.COVARIANCE):
            with recorded_warnings() as caught:
                with pytest.raises(ConfigurationError,
                                   match=r'source\.frequencies is not '
                                         r'uniformly spaced'):
                    call()
            assert not [w for w in caught if 'resampled' in str(w.message)]

    def test_a_pinned_count_warns_that_levels_converge_with_it(self):
        with pytest.warns(UserWarning, match='converge as NW grows'):
            OASN(verbose=False, n_wavenumbers=4096)

    def test_the_covariance_keeps_the_noise_levels_it_is_referenced_to(
            self):
        cov = _quiet_run(
            OASN(verbose=False, surface_noise_level=60.0).run,
            make_pekeris(), uacpy.Source(depths=50.0, frequencies=100.0),
            uacpy.Receiver(**self._RCV), RunMode.COVARIANCE)
        assert cov.metadata['surface_noise_level'] == pytest.approx(60.0)
        assert cov.metadata['white_noise_level'] == pytest.approx(-200.0)


@pytest.mark.requires_binary
class TestOASTSourceGeometry:
    """OASES-11 / decision A4: OAST computes a line source natively (option
    'P', plane geometry), so the Source decides the geometry."""

    _CARRIERS = dict(depths=[50.0], ranges=[1000.0, 2000.0])

    def test_a_line_source_puts_P_on_the_option_line(self):
        settings = OAST(verbose=False).run_settings(
            make_pekeris(),
            uacpy.Source(depths=50.0, frequencies=100.0, source_type='line'),
            uacpy.Receiver(**self._CARRIERS))
        assert settings.engine.options == 'N T J P'
        assert settings.source_type == 'line'

    def test_a_point_source_keeps_the_cylindrical_option_line(self):
        settings = OAST(verbose=False).run_settings(
            make_pekeris(), uacpy.Source(depths=50.0, frequencies=100.0),
            uacpy.Receiver(**self._CARRIERS))
        assert settings.engine.options == 'N T J'

    def test_a_raw_P_on_a_point_source_is_refused_by_every_entry_point(
            self):
        model = OAST(verbose=False, options='N J T P')
        for call in _entry_points(
                model, make_pekeris(),
                uacpy.Source(depths=50.0, frequencies=100.0),
                uacpy.Receiver(**self._CARRIERS)):
            with pytest.raises(ConfigurationError, match="carries 'P'"):
                _quiet_run(call)

    def test_oasn_has_no_line_source_geometry(self):
        with pytest.raises(UnsupportedFeatureError,
                           match="source_type='line'"):
            OASN(verbose=False).run_settings(
                make_pekeris(),
                uacpy.Source(depths=50.0, frequencies=100.0,
                             source_type='line'),
                uacpy.Receiver(depths=[30.0, 70.0], ranges=[0.0]),
                RunMode.REPLICA)


@pytest.mark.requires_binary
class TestTheDeckIsWrittenFromTheResolvedSettings:
    """What ``run_settings().engine`` shows is what the deck carries."""

    def test_oast_block_vii_is_the_settings_window(self, tmp_path):
        model = OAST(verbose=False, c_low=1400.0, work_dir=tmp_path)
        carriers = (make_pekeris(),
                    uacpy.Source(depths=50.0, frequencies=100.0),
                    uacpy.Receiver(depths=[50.0], ranges=[1000.0, 2000.0]))
        engine = model.run_settings(*carriers).engine
        assert engine.c_low == 1400.0
        assert engine.c_low_origin == 'OAST(c_low=…)'
        assert engine.c_high_origin == (
            'oases_wavenumber_bounds(water column)')
        _quiet_run(model.run, *carriers)
        deck = (tmp_path / 'oast_run.dat').read_text().splitlines()
        assert f"{engine.c_low:.1f} {engine.c_high:.6g}" in deck
        assert engine.options in deck

    def test_oasr_block_iv_is_the_settings_sweep(self, tmp_path):
        model = OASR(verbose=False, work_dir=tmp_path)
        carriers = (make_pekeris(),
                    uacpy.Source(depths=50.0,
                                 frequencies=[20.0, 30.0, 50.0, 80.0]),
                    uacpy.Receiver(depths=[50.0], ranges=[1000.0]))
        with pytest.warns(UserWarning, match=r'source\.frequencies vector '
                                             r'is non-equispaced'):
            engine = model.run_settings(*carriers).engine
        assert engine.frequency_sweep == (20.0, 80.0, 4)
        refl = _quiet_run(model.run, *carriers)
        np.testing.assert_allclose(refl.frequencies,
                                   np.linspace(20.0, 80.0, 4), rtol=1e-3)


class TestOASPSourceGeometry:
    """Decision A4: OASP computes a line source natively (option 'P',
    plane geometry, unoasp22.f:946-948), so the Source decides the
    geometry."""

    _CARRIERS = dict(depths=[50.0], ranges=[1000.0, 2000.0])

    def test_a_line_source_puts_P_on_the_option_line(self):
        settings = OASP(verbose=False).run_settings(
            make_pekeris(),
            uacpy.Source(depths=50.0, frequencies=100.0, source_type='line'),
            uacpy.Receiver(**self._CARRIERS))
        assert settings.engine.options == 'N J P'
        assert settings.source_type == 'line'

    def test_a_point_source_keeps_the_cylindrical_option_line(self):
        settings = OASP(verbose=False).run_settings(
            make_pekeris(), uacpy.Source(depths=50.0, frequencies=100.0),
            uacpy.Receiver(**self._CARRIERS))
        assert settings.engine.options == 'N J'

    def test_a_raw_P_on_a_point_source_is_refused_by_every_entry_point(
            self):
        model = OASP(verbose=False, options='N J P')
        for call in _entry_points(
                model, make_pekeris(),
                uacpy.Source(depths=50.0, frequencies=100.0),
                uacpy.Receiver(**self._CARRIERS)):
            with pytest.raises(ConfigurationError, match="carries 'P'"):
                _quiet_run(call)


@pytest.mark.requires_binary
class TestTheOaspDeckIsWrittenFromTheResolvedSettings:
    """What ``OASP().run_settings().engine`` shows is what the deck
    carries, and ``run_settings().frequencies`` — the ladder of bins
    ``_oasp_marched_frequencies`` computes from Block VIII as unoasp22.f:194-246
    does, in REAL arithmetic — is the ``.trf`` axis the result carries, bit
    for bit, so the stamp adds no note."""

    @staticmethod
    def _carriers():
        return (make_pekeris(), uacpy.Source(depths=50.0, frequencies=100.0),
                uacpy.Receiver(depths=[20.0, 50.0],
                               ranges=[200.0, 600.0, 1000.0]))

    def test_block_viii_is_the_settings_sweep(self, tmp_path):
        model = OASP(verbose=False, n_time_samples=256, work_dir=tmp_path)
        carriers = self._carriers()
        engine = model.run_settings(*carriers, RunMode.BROADBAND).engine
        _quiet_run(model.run, *carriers, RunMode.BROADBAND)
        deck = (tmp_path / 'oasp_run.dat').read_text().splitlines()
        nx, fr1, fr2, dt = deck[-1].split()[:4]
        assert (int(nx), float(fr1), float(fr2), float(dt)) == (
            engine.n_time_samples, float(f"{engine.freq_min:.9f}"),
            float(f"{engine.freq_max:.9f}"),
            float(f"{engine.time_step:.12g}"))
        assert f"{engine.c_low:.1f} {engine.c_high:.6g}" in deck

    @pytest.mark.parametrize('kw', [
        dict(run_mode=RunMode.BROADBAND),
        dict(run_mode=RunMode.BROADBAND,
             frequencies=np.linspace(30.5, 71.25, 9)),
        dict(),
        dict(run_mode=RunMode.TIME_SERIES,
             source_waveform=np.hanning(40) * np.sin(
                 2 * np.pi * 100.0 * np.arange(40) / 400.0),
             sample_rate=400.0, output_duration=0.5),
    ], ids=['default-sweep', 'band', 'coherent-tl', 'time-series'])
    def test_the_settings_frequencies_are_the_trf_axis(self, kw):
        model = OASP(verbose=False, n_time_samples=64)
        carriers = self._carriers()
        want = _quiet_run(model.run_settings, *carriers, **kw)
        result = _quiet_run(model.run, *carriers, **kw)
        np.testing.assert_array_equal(np.atleast_1d(result.frequencies),
                                      want.frequencies)
        assert result.run_settings == want
        assert not want.notes
