"""OASS / OASSP model classes.

The deck writers are covered by ``test_oass.py``; this file covers the two
wrapper classes — their declared capabilities, the single-config contract, and
the constructor-time refusals that exist because the alternative is a binary
that stops with no output or, worse, returns a plausible wrong answer.
"""

import inspect
import warnings

import numpy as np
import pytest

import uacpy
from uacpy.core.exceptions import ConfigurationError, UnsupportedFeatureError
from uacpy.models import RunMode


def _env():
    return uacpy.Environment(
        bathymetry=100.0,
        ssp=uacpy.SoundSpeedProfile(depths=[0.0, 100.0], data=[1500.0, 1500.0]),
        bottom=uacpy.Bottom.from_halfspace(uacpy.BoundaryProperties(
            sound_speed=1700.0, density=1.8, attenuation=0.5, roughness=0.5)),
    )


_SRC = uacpy.Source(depths=50.0, frequencies=250.0)
_RCV = uacpy.Receiver(depths=[10.0, 50.0, 90.0],
                      ranges=np.linspace(100.0, 5000.0, 20))


class TestDeclaredCapabilities:
    """``spec`` is how this repo declares capability; a class whose declared
    modes drift from what ``run`` accepts is how a user discovers the gap at
    runtime instead of at import."""

    def test_oass_declares_the_scattered_field_products(self):
        assert set(uacpy.OASS.spec.modes) == {RunMode.REVERBERATION,
                                              RunMode.COVARIANCE}

    def test_oassp_declares_the_pulse_products(self):
        assert set(uacpy.OASSP.spec.modes) == {RunMode.BROADBAND,
                                               RunMode.TIME_SERIES}

    def test_both_are_oases_models(self):
        from uacpy.models.oases import OASES
        assert issubclass(uacpy.OASS, OASES)
        assert issubclass(uacpy.OASSP, OASES)

    @pytest.mark.parametrize('cls_name', ['OASS', 'OASSP'])
    def test_both_are_exported_on_the_package(self, cls_name):
        assert hasattr(uacpy, cls_name)


class TestSingleConfigRule:
    """Model knobs live on ``__init__``; ``run()`` has a fixed signature and
    no ``**kwargs``. A ``run(**kwargs)`` is what lets a misspelled knob be
    silently dropped instead of rejected."""

    @pytest.mark.parametrize('cls_name', ['OASS', 'OASSP'])
    def test_run_takes_no_var_keywords(self, cls_name):
        params = inspect.signature(getattr(uacpy, cls_name).run).parameters
        assert not any(p.kind is inspect.Parameter.VAR_KEYWORD
                       for p in params.values())

    @pytest.mark.parametrize('cls_name', ['OASS', 'OASSP'])
    def test_run_takes_no_var_positionals(self, cls_name):
        params = inspect.signature(getattr(uacpy, cls_name).run).parameters
        assert not any(p.kind is inspect.Parameter.VAR_POSITIONAL
                       for p in params.values())


class TestOASSPConstructorRefusals:
    """Each refusal exists because the binary's own failure is silent."""

    def test_correlation_length_is_required(self):
        # oass.tex:182-183 — the roughness spectrum is what OASSP integrates
        # and is NOT adopted from the mean-field run.
        with pytest.raises(ConfigurationError, match='correlation_length'):
            uacpy.OASSP(spectral_exponent=2.0)

    def test_an_invented_spectrum_is_refused(self):
        # GETOPT offers exactly one spectrum letter, 'g' (unoassp30.f:1034).
        with pytest.raises(ConfigurationError, match='spectrum'):
            uacpy.OASSP(correlation_length=10.0, spectrum='von-karman')

    @pytest.mark.parametrize('spectrum', ['gaussian', 'goff-jordan'])
    def test_the_two_real_spectra_are_accepted(self, spectrum):
        # The discriminating counterpart: the guard must not refuse the
        # spectra OASES actually implements.
        uacpy.OASSP(correlation_length=10.0, spectrum=spectrum)

    def test_pinning_the_fft_grid_against_a_mean_field_is_refused(self):
        # unoassp30.f:181-188 reads NT/FR1/FR2/DT back out of the .rhs, so a
        # user-set band would be discarded with no warning.
        with pytest.raises(ConfigurationError, match='discarded silently'):
            uacpy.OASSP(correlation_length=10.0,
                        mean_field=uacpy.OASP(), n_time_samples=1024)

    def test_raw_options_alongside_typed_flags_is_refused(self):
        # options= replaces the whole line, so a typed flag would vanish.
        with pytest.raises(ConfigurationError, match='discarded silently'):
            uacpy.OASSP(correlation_length=10.0, options='N J',
                        plane_geometry=True)


class TestOASSUnsupportedOptions:
    """Options the binary accepts but uacpy cannot read back are refused by
    name. Emitting them would produce output files no reader consumes, or —
    for 'Z' — shift the deck by two records.

    The check takes ``run_mode``, so it belongs to ``run()`` and not to
    ``__init__``: which products are readable depends on what was asked for.
    Tested at the guard rather than through a binary run, so the refusal is
    pinned without paying for two subprocesses."""

    @staticmethod
    def _model():
        return uacpy.OASS(interface=3, correlation_length=10.0,
                          spectral_exponent=2.0)

    @pytest.mark.parametrize('letter', ['C', 'D'])
    def test_contour_options_refused(self, letter):
        # Their output goes to the CONDRW/CONDRB pair on units 28/29
        # (unoass21.f:277-278), which no uacpy reader consumes.
        with pytest.raises(UnsupportedFeatureError):
            self._model()._reject_unreadable_options(
                letter, RunMode.REVERBERATION)

    def test_svp_plot_option_refused_because_it_lengthens_the_deck(self):
        # unoass21.f:319-320 reads two further records after Block IX that
        # write_oass_input does not emit, so the deck would end early.
        with pytest.raises(UnsupportedFeatureError, match='Z'):
            self._model()._reject_unreadable_options(
                'r Z', RunMode.REVERBERATION)

    def test_the_ordinary_reverberation_option_is_accepted(self):
        # The discriminating counterpart: the guard must not refuse the one
        # option the default path emits.
        self._model()._reject_unreadable_options('r', RunMode.REVERBERATION)


class TestOASSRoundTrip:
    """The two-binary chain end to end: the mean-field producer writes a
    ``.rhs`` with option 's', then ``oass2`` integrates the scattered field
    over it. Marked slow — it runs two real binaries."""

    @pytest.mark.slow
    def test_reverberation_run_returns_a_reverberation_field(self, tmp_path):
        model = uacpy.OASS(interface=3, correlation_length=10.0,
                           spectral_exponent=2.0, rms_roughness=0.5,
                           work_dir=tmp_path, cleanup=False, verbose=False)
        result = model.run(_env(), _SRC, _RCV,
                           run_mode=RunMode.REVERBERATION)
        # The quantity is tagged, not derived: reverberation shares TL's dB
        # representation but is a different quantity, so it must not compare
        # against a TL field on one colour scale.
        assert result.kind == 'reverberation'
        assert result.unit == 'dB'
        assert np.isfinite(result.data).any()

    @pytest.mark.slow
    def test_the_mean_field_deck_survives_a_pinned_work_dir(self, tmp_path):
        # The chain is only debuggable if both decks remain; work_dir pinned
        # with cleanup=False is the documented model-chaining idiom.
        model = uacpy.OASS(interface=3, correlation_length=10.0,
                           spectral_exponent=2.0, rms_roughness=0.5,
                           work_dir=tmp_path, cleanup=False, verbose=False)
        model.run(_env(), _SRC, _RCV, run_mode=RunMode.REVERBERATION)
        names = {p.name for p in tmp_path.iterdir()}
        assert any(n.endswith('.045') for n in names), \
            'the mean-field .rhs is this run\'s FOR045 and must survive'
        assert any(n.startswith('oass_run') for n in names)


class TestOASSPDeck:
    """``write_oassp_input`` shares OASP's deck token for token
    (`unoasp22.f` vs `unoassp30.f` read the same sequence with the same
    arities); every difference is pinned here."""

    @staticmethod
    def _deck(tmp_path, **kw):
        from uacpy.io.oases_writer import write_oassp_input
        kw.setdefault('interface', 4)
        kw.setdefault('correlation_length', 10.0)
        kw.setdefault('spectral_exponent', 2.0)
        kw.setdefault('n_time_samples', 1024)
        kw.setdefault('freq_min', 100.0)
        kw.setdefault('freq_max', 400.0)
        kw.setdefault('time_step', 1e-3)
        kw.setdefault('range_start', 100.0)
        kw.setdefault('range_step', 100.0)
        path = tmp_path / 'oassp_run.dat'
        write_oassp_input(path, _env(), _SRC, _RCV, kw.pop('options', 'N J'),
                          **kw)
        return [ln for ln in path.read_text().splitlines() if ln.strip()]

    def test_block_vii_carries_the_realization_as_a_fourth_token(self, tmp_path):
        # OASS's Block VII has three tokens; OASSP's has four, and the 4th is
        # the realization seed (spec §5.5, measured).
        lines = self._deck(tmp_path, realization=7)
        nw = [ln for ln in lines if len(ln.split()) == 4][0]
        assert int(nw.split()[3]) == 7

    def test_the_scattering_interface_carries_the_roughness_spectrum(self, tmp_path):
        # Same silent failure mode as OASS: a POSITIVE RG at the named
        # interface means infinite correlation length, i.e. no back-scatter,
        # from a run that exits 0.
        rough = [ln for ln in self._deck(tmp_path)
                 if len(ln.split()) == 9 and float(ln.split()[6]) < 0.0]
        assert len(rough) == 1, 'exactly one interface carries -|RG| CL M'
        assert (float(rough[0].split()[7]),
                float(rough[0].split()[8])) == (10.0, 2.0)

    def test_a_water_interface_is_refused(self, tmp_path):
        with pytest.raises(ConfigurationError, match='not a bottom interface'):
            self._deck(tmp_path, interface=2)

    def test_the_time_block_carries_the_fft_grid(self, tmp_path):
        # NT FR1 FR2 DT ... — OASSP reads these back out of the .rhs, so the
        # deck must still carry a well-formed record.
        last = self._deck(tmp_path)[-1].split()
        assert int(last[0]) == 1024
        assert (float(last[1]), float(last[2])) == (100.0, 400.0)


class TestScatteringOptionContracts:
    """The OASS/OASSP scattering options map onto what the binaries actually
    honour. Each test pins the *mechanism*, not the symptom."""

    def test_oassp_refuses_the_vendor_disabled_rescattering(self):
        # The letter is parsed and then does nothing: every routine
        # unoassp30.f:677-688 dispatches to has the `if (.not.rescat)` guard
        # commented out with ROUGH2(II)=0 left live (oasvun31.f:76-80 and
        # three siblings). Returning a single-scatter answer under a
        # multiple-scattering label is the failure being prevented.
        with pytest.raises(UnsupportedFeatureError, match='rescat|980402'):
            uacpy.OASSP(correlation_length=10.0, multiple_scattering=True)

    def test_oass_still_honours_multiple_scattering(self):
        # The discriminating counterpart — the asymmetry is real, not a
        # blanket refusal. oassun26.f:491, :685, :901 all test .not.rescat.
        assert uacpy.OASS(interface=3, correlation_length=10.0,
                          multiple_scattering=True).multiple_scattering

    def test_a_kernel_deck_needs_a_positive_phase_speed(self, tmp_path):
        # unoass21.f:342-344 takes the `IF (.not.REVERB)` branch and forms
        # ETA=DSQ/CPHASE; gfortran does not trap the division, so CALSIN runs
        # on Inf rather than failing.
        from uacpy.io.oases_writer import write_oass_input
        with pytest.raises(ConfigurationError, match='divides by zero'):
            write_oass_input(tmp_path / 'k.dat', _env(), _SRC, _RCV, 'I',
                             interface=3, correlation_length=10.0,
                             spectral_exponent=2.0, phase_speed=0.0)

    def test_the_surface_roughness_reaches_both_water_branches(self, tmp_path):
        # ROUGH(M) is the roughness at the TOP of layer M (oaseun31.f:381-383)
        # and ROUGH(1)=ROUGH(2) at :377, so the first water record carries the
        # sea surface. The isovelocity branch used to write a hardcoded 0.0.
        from uacpy.io.oases_writer import write_oass_input
        surface = uacpy.BoundaryProperties(sound_speed=340.0, density=0.0012,
                                           roughness=0.7)
        for iso, interface in ((True, 3), (False, 4)):
            data = [1500.0, 1500.0] if iso else [1500.0, 1490.0]
            env = uacpy.Environment(
                bathymetry=100.0,
                ssp=uacpy.SoundSpeedProfile(depths=[0.0, 100.0], data=data),
                bottom=uacpy.Bottom.from_halfspace(uacpy.BoundaryProperties(
                    sound_speed=1700.0, density=1.8, attenuation=0.5,
                    roughness=0.5)),
                surface=surface)
            path = tmp_path / f'iso{iso}.dat'
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                write_oass_input(path, env, _SRC, _RCV, 'r',
                                 interface=interface, correlation_length=10.0,
                                 spectral_exponent=2.0, rms_roughness=0.5)
            water = path.read_text().splitlines()[5]
            assert float(water.split()[6]) == pytest.approx(0.7), \
                f'isovelocity={iso}: surface roughness lost ({water!r})'
