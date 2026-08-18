"""OASSP (OASES scattered-field realizations) writer, guard and chain tests.

OASSP is a post-processor: it consumes the ``.rhs`` an OASP run with option
``'s'`` writes and returns one realization of the scattered field, in OASP's
own ``.trf`` format. Everything a wrapper can get wrong here fails *silently* —
``GETOPT``'s closing ``ELSE`` is empty (``unoassp30.f:1049``), Block VIII is
overwritten from the ``.rhs`` with a warning covering only two of its four
fields (``:181-188``), and a roughness spectrum attached to the wrong layer
leaves ``CLEN = 0`` on the one that scatters (``oaseun31.f:100``). So the
writer tests below parse the deck off disk rather than re-deriving it, and the
chain tests read the binary's own echo.
"""

import os
import warnings

import numpy as np
import pytest

import uacpy
from uacpy.core.exceptions import ConfigurationError, UnsupportedFeatureError
from uacpy.io.oases_writer import write_oassp_input
from uacpy.io.oases_reader import read_oases_rhs_header
from uacpy.models.base import RunMode
from uacpy.tests.conftest import make_pekeris


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — the §5.4 geometry of the OASS/OASSP spec, kept small
# ─────────────────────────────────────────────────────────────────────────────

def _env(roughness=0.5):
    return make_pekeris(
        bathymetry=128.0,
        ssp=uacpy.SoundSpeedProfile(depths=[0, 128], data=[1500, 1500]),
        sound_speed=2300.0, density=2.65, roughness=roughness)


def _source():
    return uacpy.Source(depths=100.0, frequencies=500.0)


def _receiver(n_depths=4):
    return uacpy.Receiver(depths=np.linspace(60.0, 127.0, n_depths),
                          ranges=np.array([500.0, 1000.0]))


#: Block VIII values a mean-field run would hand over. Only their arrival in
#: the deck is under test here; the chain tests take the real ones.
_BLOCK_VIII = dict(n_time_samples=512, freq_min=400.0, freq_max=600.0,
                   time_step=0.000833)

#: The deck this environment produces has one upper half-space + two water
#: layers, so the seabed record is deck layer 4.
_SEABED_LAYER = 4


def _write(tmp_path, env=None, receiver=None, **kwargs):
    """Write a deck and return its non-empty lines."""
    path = tmp_path / 'oassp_run.dat'
    kw = dict(interface=_SEABED_LAYER, correlation_length=5.0,
              spectral_exponent=2.5, **_BLOCK_VIII)
    kw.update(kwargs)
    write_oassp_input(path, env if env is not None else _env(), _source(),
                      receiver if receiver is not None else _receiver(),
                      kw.pop('options', None), **kw)
    return [ln for ln in path.read_text().splitlines() if ln.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# W3 — deck structure, parsed back off disk
# ─────────────────────────────────────────────────────────────────────────────

class TestDeckStructure:
    """The deck is OASP's token for token except where it is not."""

    def test_block_vii_has_four_tokens_and_the_fourth_is_the_realization(
            self, tmp_path):
        # unoassp30.f:169 READ(1,*) NWVNO,ICW1,ICW2,INTF — four, like OASP and
        # unlike OASS, which reads three (unoass21.f:189). :170 then assigns
        # msft_0=intf and :535 forms ISEED = -123 - msft_0.
        lines = _write(tmp_path, realization=7)
        block_vii = lines[-2].split()
        assert len(block_vii) == 4, block_vii
        assert block_vii[3] == '7'

    def test_block_vi_has_three_tokens_not_oasts_four(self, tmp_path):
        # INREC picks the arity from PROGNM, not from the option line: only
        # 'OASTL' reads a 4th token IDINC (oaseun31.f:1156); OASSP's PROGNM is
        # 'OASP17' (unoassp30.f:1156) so it takes the 3-token form at :1165.
        lines = _write(tmp_path)
        assert len(lines[-4].split()) == 3, lines[-4]

    def test_block_viii_carries_the_mean_fields_grid_verbatim(self, tmp_path):
        lines = _write(tmp_path)
        nt, fr1, fr2, dt, r0, dr, nr = lines[-1].split()
        assert int(nt) == 512
        assert float(fr1) == pytest.approx(400.0)
        assert float(fr2) == pytest.approx(600.0)
        assert float(dt) == pytest.approx(0.000833)
        # R0/DR are km on disk; the receiver asked for 500 m and 1000 m.
        assert float(r0) == pytest.approx(0.5)
        assert float(dr) == pytest.approx(0.5)
        assert int(nr) == 2

    def test_scattering_interface_carries_the_nine_token_form(self, tmp_path):
        # oaseun31.f:72-93: a negative RG makes INENVI backspace and re-read
        # the record as V(1..6) + RG + CL + M. Without it CLEN stays 0 (:100)
        # and unoassp30.f:606 hands PV a zero correlation length.
        lines = _write(tmp_path, correlation_length=5.0, spectral_exponent=2.5)
        seabed = lines[_SEABED_LAYER + 3].split()   # title, opts, freq, NL
        assert len(seabed) == 9, seabed
        assert float(seabed[6]) == pytest.approx(-0.5)   # -|RG|
        assert float(seabed[7]) == pytest.approx(5.0)    # CL
        assert float(seabed[8]) == pytest.approx(2.5)    # M

    def test_water_records_keep_the_seven_token_form(self, tmp_path):
        # Only the scattering interface may go negative; a nine-token water
        # record would shift every read below it.
        lines = _write(tmp_path)
        for row in lines[5:_SEABED_LAYER + 3]:
            assert float(row.split()[6]) >= 0.0, row

    @pytest.mark.parametrize('interface', [4, 5, 6])
    def test_layered_bottom_keys_the_spectrum_on_the_named_deck_layer(
            self, tmp_path, interface):
        # The suffix_fn _emit_bottom_layers calls counts from its own first
        # bottom record, not from deck layer 1, so the offset has to track the
        # water-layer count. Two sediment layers + a halfspace here, on top of
        # one upper half-space and two water records: deck layers 4, 5, 6.
        from uacpy.core.environment import SeabedColumn, SedimentLayer
        env = uacpy.Environment(
            bathymetry=128.0,
            ssp=uacpy.SoundSpeedProfile(depths=[0, 128], data=[1500, 1500]),
            bottom=SeabedColumn(
                layers=[SedimentLayer(thickness=5.0, sound_speed=1600.0,
                                      density=1.6, attenuation=0.2,
                                      roughness=0.3),
                        SedimentLayer(thickness=5.0, sound_speed=1700.0,
                                      density=1.8, attenuation=0.3,
                                      roughness=0.4)],
                halfspace=uacpy.BoundaryProperties(
                    acoustic_type='half-space', sound_speed=2300.0,
                    density=2.65, attenuation=0.5, roughness=0.5)),
        )
        lines = _write(tmp_path, env=env, interface=interface)
        assert int(lines[3]) == 6                     # NL
        # Deck layer L is line L+3 (title, options, frequency, NL).
        bottom = {L: lines[L + 3].split() for L in (4, 5, 6)}
        # Exactly the named record goes negative, i.e. carries CL and M; the
        # other two keep the environment's own RMS on the seven-token form.
        assert [L for L, r in bottom.items() if float(r[6]) < 0.0] == [
            interface], lines
        assert len(bottom[interface]) == 9
        assert float(bottom[interface][7]) == 5.0
        for L, record in bottom.items():
            if L != interface:
                assert len(record) == 7, record

    def test_default_option_line(self, tmp_path):
        # 'N' single component, 'J' holds ICNTIN=1 so automatic sampling takes
        # the complex wavenumber contour rather than OMEGIM = -ln(50)*df
        # (unoassp30.f:281-289), 's' zeroes the sources (:628-635).
        assert set(_write(tmp_path)[1].split()) == {'N', 'J', 's'}


# ─────────────────────────────────────────────────────────────────────────────
# Writer guards
# ─────────────────────────────────────────────────────────────────────────────

class TestWriterGuards:

    def test_unknown_option_letter_raises(self, tmp_path):
        # G9: OASSP's GETOPT closes with an empty ELSE (unoassp30.f:1049-1050),
        # so a typo'd letter produces no diagnostic anywhere.
        with pytest.raises(ConfigurationError, match="not OASSP option"):
            _write(tmp_path, options='N J s W')

    def test_option_letters_needing_unwritten_blocks_raise(self, tmp_path):
        for letter in ('d', 'G', 'T', 'Z', 'l', 'v', 'b'):
            with pytest.raises(ConfigurationError, match="not supported"):
                _write(tmp_path, options=f'N J s {letter}')

    def test_tau_p_raises(self, tmp_path):
        with pytest.raises(ConfigurationError, match="tau-p"):
            _write(tmp_path, options='N J s t')

    def test_decomposition_and_extra_components_raise(self, tmp_path):
        for letter in ('U', 'V', 'H', 'R', 'K', 'S'):
            with pytest.raises(UnsupportedFeatureError,
                               match="multi-component"):
                _write(tmp_path, options=f'N J s {letter}')

    def test_volume_scattering_is_refused_by_name(self, tmp_path):
        # D2: a negative CL is OASES' switch to the twelve-token record
        # (oaseun31.f:76-90); SKW / M / RMS / GAM have no carrier field.
        with pytest.raises(UnsupportedFeatureError) as exc:
            _write(tmp_path, correlation_length=-5.0)
        for name in ('SKW', 'M', 'RMS', 'GAM'):
            assert name in str(exc.value)

    def test_zero_correlation_length_raises(self, tmp_path):
        with pytest.raises(ConfigurationError, match="correlation_length"):
            _write(tmp_path, correlation_length=0.0)

    def test_spectral_exponent_below_the_integrability_bound_raises(
            self, tmp_path):
        # G10: oassp.tex:356-362 — the power spectrum is not integrable.
        with pytest.raises(ConfigurationError, match="spectral_exponent"):
            _write(tmp_path, spectral_exponent=1.2)

    def test_smooth_interface_raises(self, tmp_path):
        # G3: pow = |ROUGH(INTFCE)| comes from this deck (unoassp30.f:601,
        # :615), and SCTRHS skips ROUGH2 < 1e-10 in the producer
        # (oaseun31.f:2310), so the whole chain returns zero.
        with pytest.raises(ConfigurationError, match="nothing to scatter"):
            _write(tmp_path, env=_env(roughness=0.0))

    def test_interface_outside_the_bottom_stack_raises(self, tmp_path):
        # The sea surface is deck layer 2 here: _emit_water_layers cannot emit
        # the nine-token form, so pointing at it must not be silently accepted.
        with pytest.raises(ConfigurationError, match="not a bottom interface"):
            _write(tmp_path, interface=2)
        with pytest.raises(ConfigurationError, match="not a bottom interface"):
            _write(tmp_path, interface=_SEABED_LAYER + 1)

    def test_negative_realization_raises(self, tmp_path):
        with pytest.raises(ConfigurationError, match="realization"):
            _write(tmp_path, realization=-1)

    def test_non_uniform_ranges_raise(self, tmp_path):
        rcv = uacpy.Receiver(depths=[60.0, 100.0],
                             ranges=np.array([500.0, 1000.0, 3000.0]))
        with pytest.raises(ConfigurationError, match="uniformly spaced"):
            _write(tmp_path, receiver=rcv)

    def test_c_high_without_plane_geometry_warns(self, tmp_path):
        # G12: unoassp30.f:205-217 forces CMAXIN = 1e12 in cylindrical
        # geometry, so the deck's value never reaches the integration.
        with pytest.warns(UserWarning, match="Full Hankel"):
            _write(tmp_path, c_high=1.0e5)

    def test_c_high_with_plane_geometry_is_silent_and_reaches_the_deck(
            self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            lines = _write(tmp_path, options='N J s P', c_high=1.0e5)
        assert float(lines[-3].split()[1]) == pytest.approx(1.0e5)

    def test_c_low_reaches_the_deck(self, tmp_path):
        lines = _write(tmp_path, c_low=1200.0)
        assert float(lines[-3].split()[0]) == pytest.approx(1200.0)

    def test_unknown_kwarg_raises(self, tmp_path):
        with pytest.raises(ConfigurationError, match="not read by this deck"):
            _write(tmp_path, phase_speed=1500.0)


# ─────────────────────────────────────────────────────────────────────────────
# The shared body — OASSP must not drift from OASP
# ─────────────────────────────────────────────────────────────────────────────

class TestSharedDeckBody:
    """``grep -n 'READ(1,\\*)' unoasp22.f unoassp30.f`` gives the same record
    sequence with the same arities, so the two decks may differ only in Block
    VII's fourth token and the scattering interface's roughness columns."""

    def test_only_the_two_documented_records_differ_from_oasp(self, tmp_path):
        from uacpy.io.oases_writer import write_oasp_input
        env, src, rcv = _env(), _source(), _receiver()
        oasp = tmp_path / 'oasp.dat'
        write_oasp_input(oasp, env, src, rcv, 'N J s',
                         n_time_samples=512, freq_min=400.0, freq_max=600.0,
                         time_step=0.000833, freq_output_increment=0,
                         integration_offset=0.0)
        a = [ln for ln in oasp.read_text().splitlines() if ln.strip()]
        b = _write(tmp_path, options='N J s', realization=3)
        assert len(a) == len(b)
        differing = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        # The scattering interface's roughness columns, and Block VII's
        # fourth token — nothing else.
        assert differing == [_SEABED_LAYER + 3, len(a) - 2], (differing, a, b)


# ─────────────────────────────────────────────────────────────────────────────
# .rhs header reader
# ─────────────────────────────────────────────────────────────────────────────

class TestRhsHeaderReader:

    def test_empty_file_raises(self, tmp_path):
        # OPFILB opens unit 45 with STATUS='UNKNOWN' (oashun21.f:634), so a
        # producer that wrote nothing still leaves a zero-length file.
        path = tmp_path / 'empty.045'
        path.write_bytes(b'')
        with pytest.raises(uacpy.core.exceptions.FileFormatError,
                           match="empty or truncated"):
            read_oases_rhs_header(path)

    def test_a_header_without_scattering_records_raises(self, tmp_path):
        import struct
        path = tmp_path / 'short.045'
        with open(path, 'wb') as f:
            for payload in (struct.pack('<ifff', 512, 400.0, 600.0, 8.3e-4),
                            struct.pack('<fiiif', 500.0, 2, 2048, 0, 1.0)):
                f.write(struct.pack('<i', len(payload)) + payload
                        + struct.pack('<i', len(payload)))
        with pytest.raises(uacpy.core.exceptions.FileFormatError,
                           match="no 76-byte SCTRHS record"):
            read_oases_rhs_header(path)


# ─────────────────────────────────────────────────────────────────────────────
# Model construction — single-config rule
# ─────────────────────────────────────────────────────────────────────────────

class TestScatteringOptionContracts:
    """The OASS/OASSP scattering options map onto what the binaries actually
    honour. Each test pins the *mechanism*, not the symptom."""

    @pytest.mark.parametrize('spectrum', ['gaussian', 'goff-jordan'])
    def test_the_two_real_spectra_are_accepted(self, spectrum):
        # The discriminating counterpart to test_bad_spectrum_raises: the
        # guard must not refuse the spectra OASES actually implements.
        uacpy.OASSP(correlation_length=10.0, spectrum=spectrum)

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


@pytest.mark.requires_oases
class TestModelConfiguration:

    def test_supported_run_modes(self):
        m = uacpy.OASSP(correlation_length=5.0)
        assert set(m.spec.modes) == {RunMode.BROADBAND, RunMode.TIME_SERIES}
        with pytest.raises(UnsupportedFeatureError):
            m._resolve_run_mode(RunMode.COHERENT_TL)

    def test_correlation_length_is_required(self):
        with pytest.raises(ConfigurationError, match="correlation_length"):
            uacpy.OASSP()

    def test_negative_correlation_length_raises_at_construction(self):
        # OASES reads a negative CL as the volume-scattering record switch,
        # whose extra fields no uacpy carrier holds; fail at construction
        # like every other constructor knob, not at deck-write time.
        with pytest.raises(UnsupportedFeatureError,
                           match="correlation_length"):
            uacpy.OASSP(correlation_length=-5.0)

    def test_zero_correlation_length_raises_at_construction(self):
        with pytest.raises(ConfigurationError, match="correlation_length"):
            uacpy.OASSP(correlation_length=0.0)

    def test_copy_round_trips_every_knob(self):
        m = uacpy.OASSP(correlation_length=5.0, spectral_exponent=2.5,
                        spectrum='goff-jordan', realization=3,
                        plane_geometry=True, n_time_samples=256)
        c = m.copy(realization=4)
        assert (c.realization, c.correlation_length, c.spectral_exponent,
                c.spectrum, c.plane_geometry, c.n_time_samples) == (
                    4, 5.0, 2.5, 'goff-jordan', True, 256)
        # executable stays None so copy() re-resolves rather than re-pinning.
        assert c.executable is None

    def test_options_is_exclusive_with_the_typed_flags(self):
        with pytest.raises(ConfigurationError, match="discarded"):
            uacpy.OASSP(correlation_length=5.0, options='N J s',
                        spectrum='goff-jordan')

    def test_mean_field_is_exclusive_with_the_fft_grid_knobs(self):
        # G11: OASSP reads NT/FR1/FR2/DT back out of the .rhs, so a value set
        # here and a mean field that disagrees cannot both be honoured.
        with pytest.raises(ConfigurationError, match="mean_field"):
            uacpy.OASSP(correlation_length=5.0, n_time_samples=256,
                        mean_field=uacpy.OASP(n_time_samples=512))

    def test_typed_flags_reach_the_option_line(self):
        # 'p' is deliberately absent: OASSP refuses multiple_scattering
        # because the vendor disabled re-scattering in its integration path
        # (oasvun31.f:76-80 and three siblings), so the letter would be
        # parsed and then do nothing.
        m = uacpy.OASSP(correlation_length=5.0, spectrum='goff-jordan',
                        plane_geometry=True)
        assert set(m._resolve_options().split()) == {'N', 'J', 's', 'g', 'P'}
        m = uacpy.OASSP(correlation_length=5.0, scattered_only=False)
        assert 's' not in m._resolve_options()

    def test_mean_field_gets_option_s_added(self):
        # Without 's' OASP never calls opfilb(45) (unoasp22.f:251-254) and no
        # .rhs is written at all.
        m = uacpy.OASSP(correlation_length=5.0,
                        mean_field=uacpy.OASP(options='N J'))
        assert 's' in m._build_mean_field().options
        # An explicit 's' is not doubled.
        m = uacpy.OASSP(correlation_length=5.0,
                        mean_field=uacpy.OASP(options='N J s'))
        assert m._build_mean_field().options == 'N J s'

    def test_bad_spectrum_raises(self):
        with pytest.raises(ConfigurationError, match="spectrum"):
            uacpy.OASSP(correlation_length=5.0, spectrum='von-karman')

    def test_collapse_defaults(self):
        # A single spectral solve over one stack, as for OASP.
        # test_collapse_policy's table constructs every model with no
        # arguments, which OASSP's required correlation_length rules out, so
        # the per-model row lives here.
        from uacpy.models.base import DEFAULT_COLLAPSE
        m = uacpy.OASSP(correlation_length=5.0, verbose=False)
        expected = dict(DEFAULT_COLLAPSE,
                        **{'ssp': 'mean', 'bottom_range': 'median'})
        assert {k: m._collapse[k] for k in expected} == expected


@pytest.mark.requires_oases
class TestScatteringInterfacePreflight:
    """OASSP's ``.rhs`` reader consumes one record per wavenumber with no
    interface filter, taking the scattering interface from the file's first
    record (``oasvun31.f:66-70``, ``unoassp30.f:549``) — so exactly one
    interface may be rough, and it must be a bottom one. Each violation is
    named up front, before a mean-field run is spent producing a ``.rhs``
    the post-processor cannot use (measured: with a rough surface the binary
    scatters from the water record's empty spectrum, derives a zero
    wavenumber step from the interleaved records and stops on
    '>>> ERROR: Frequency mismatch in rhs file <<<')."""

    @staticmethod
    def _model():
        return uacpy.OASSP(correlation_length=5.0, spectral_exponent=2.5,
                           n_time_samples=256, freq_min=400.0,
                           freq_max=600.0)

    @staticmethod
    def _run(model, env):
        return model.run(env, _source(), _receiver(n_depths=2))

    def test_rough_sea_surface_is_refused_by_name(self):
        env = uacpy.Environment(
            bathymetry=128.0,
            ssp=uacpy.SoundSpeedProfile(depths=[0, 128], data=[1500, 1500]),
            surface=uacpy.BoundaryProperties(acoustic_type='vacuum',
                                             roughness=0.5),
            bottom=uacpy.BoundaryProperties(
                sound_speed=2300.0, density=2.65, attenuation=0.5,
                roughness=0.5))
        with pytest.raises(UnsupportedFeatureError,
                           match='rough sea surface'):
            self._run(self._model(), env)

    def test_smooth_environment_is_refused_before_the_mean_field(self,
                                                                 tmp_path):
        # The raise happens before any binary launches: the pinned work dir
        # stays empty rather than holding a spent mean-field run.
        m = uacpy.OASSP(correlation_length=5.0, spectral_exponent=2.5,
                        n_time_samples=256, freq_min=400.0, freq_max=600.0,
                        work_dir=tmp_path, cleanup=False)
        with pytest.raises(ConfigurationError, match='smooth'):
            self._run(m, _env(roughness=0.0))
        assert not any(tmp_path.iterdir())

    def test_rms_roughness_does_not_rescue_a_smooth_environment(self):
        # rms_roughness= overrides only the OASSP deck; the mean field still
        # writes an empty .rhs, so the same refusal applies.
        m = uacpy.OASSP(correlation_length=5.0, spectral_exponent=2.5,
                        rms_roughness=0.5,
                        n_time_samples=256, freq_min=400.0, freq_max=600.0)
        with pytest.raises(ConfigurationError, match='rms_roughness'):
            self._run(m, _env(roughness=0.0))

    def test_two_rough_bottom_interfaces_are_refused(self):
        from uacpy.core.environment import SeabedColumn, SedimentLayer
        env = uacpy.Environment(
            bathymetry=128.0,
            ssp=uacpy.SoundSpeedProfile(depths=[0, 128], data=[1500, 1500]),
            bottom=SeabedColumn(
                layers=[SedimentLayer(thickness=10.0, sound_speed=1700.0,
                                      density=1.8, attenuation=0.5,
                                      roughness=0.3)],
                halfspace=uacpy.BoundaryProperties(
                    acoustic_type='half-space', sound_speed=2300.0,
                    density=2.65, attenuation=0.5, roughness=0.5)))
        with pytest.raises(UnsupportedFeatureError,
                           match='rough bottom interfaces'):
            self._run(self._model(), env)


# ─────────────────────────────────────────────────────────────────────────────
# Round trips against the real binaries
# ─────────────────────────────────────────────────────────────────────────────

def _run(work_dir=None, cleanup=None, run_mode=None, run_kwargs=None,
         **kwargs):
    kw = dict(correlation_length=5.0, spectral_exponent=2.5,
              n_time_samples=256, freq_min=400.0, freq_max=600.0)
    kw.update(kwargs)
    m = uacpy.OASSP(work_dir=work_dir, cleanup=cleanup, **kw)
    return m.run(_env(), _source(), _receiver(n_depths=2), run_mode,
                 **(run_kwargs or {}))


@pytest.mark.slow
@pytest.mark.requires_oases
class TestChain:

    def test_broadband_returns_the_scattered_field_on_the_mean_fields_grid(
            self, tmp_path):
        # R7: whatever band was asked for, the .trf's frequency axis is the
        # mean field's, because OASSP substitutes Block VIII from the .rhs
        # (unoassp30.f:181-188). Asking for the substitution to be a no-op is
        # the whole point of reading the header back.
        result = _run(work_dir=tmp_path, cleanup=False)
        mean = result.metadata['mean_field_result']
        assert result.data.shape == mean.data.shape
        assert np.array_equal(result.coords['frequency'],
                              mean.coords['frequency'])
        assert np.iscomplexobj(result.data)
        assert np.all(np.isfinite(result.data))
        # The scattered field is not the mean field.
        assert not np.array_equal(result.data, mean.data)
        assert result.metadata['interface'] == _SEABED_LAYER
        assert str(result.phase_reference).endswith('travelling_wave')

    def test_the_binary_reports_no_frequency_sampling_mismatch(self, tmp_path):
        # unoassp30.f:182-188 warns on NT/DT only; a band mismatch is silent.
        # Re-run the deck the wrapper wrote and read the binary's own echo.
        import subprocess
        _run(work_dir=tmp_path, cleanup=False)
        model = uacpy.OASSP(correlation_length=5.0)
        env = os.environ | {
            'FOR001': 'oassp_run.dat', 'FOR019': 'oassp_run.plp',
            'FOR020': 'oassp_run.plt', 'FOR045': 'oasp_run.045',
            'FOR046': 'oasp_run.046',
        }
        echo = subprocess.run([str(model._exe)], cwd=tmp_path, env=env,
                              capture_output=True, text=True).stdout
        assert 'Frequency sampling mismatch' not in echo
        # And the roughness spectrum landed on the interface the .rhs names.
        assert 'Roughness scattering Layer:' in echo
        layer = int(echo.split('Roughness scattering Layer:')[1].split()[0])
        assert layer == _SEABED_LAYER
        assert 'volume= F' in echo
        assert 'Random number generator seed:        -123' in echo

    def test_total_field_shares_the_mean_fields_sign_convention(self):
        # scattered_only=False leaves the source arrays live, so the .trf is
        # the total field; with the OASSP deck's rms roughness dialled to
        # near-nothing the realization's contribution vanishes and the total
        # converges to the mean field — the elementwise ratio then exposes
        # the relative sign of the two Fields. Both ride the same
        # travelling-wave convention, so the ratio is +1, not -1.
        result = _run(rms_roughness=0.01, scattered_only=False)
        mean = result.metadata['mean_field_result']
        tot = np.asarray(result.data)
        mf = np.asarray(mean.data)
        sel = np.abs(mf) > np.percentile(np.abs(mf), 60)
        ratio = tot[sel] / mf[sel]
        assert np.median(np.abs(np.angle(ratio, deg=True))) < 20.0, (
            "total/mean angle far from 0 deg — the two .trf payloads no "
            "longer share a sign convention")
        # Magnitude is only a sanity bound: the OASSP deck's 0.01-m rms
        # differs from the mean-field deck's environment value, so the two
        # runs see different coherent scattering loss (measured ratio ~1.35
        # here); the convention discriminator is the angle above.
        assert 0.3 < float(np.median(np.abs(ratio))) < 3.0

    def test_realization_is_reproducible_and_distinct(self, tmp_path):
        # R5/R6. unoassp30.f:535 ISEED = -123 - msft_0.
        a = _run(realization=0)
        b = _run(realization=0)
        c = _run(realization=1)
        assert np.array_equal(a.data, b.data)
        assert not np.array_equal(a.data, c.data)
        assert a.data.shape == c.data.shape

    def test_time_series_mode(self, tmp_path):
        fs = 4000.0
        t = np.arange(int(0.02 * fs)) / fs
        waveform = np.sin(2 * np.pi * 500.0 * t) * np.hanning(len(t))
        result = _run(run_mode=RunMode.TIME_SERIES,
                      run_kwargs=dict(source_waveform=waveform,
                                      sample_rate=fs))
        assert 'time' in result.coords
        assert np.isrealobj(result.data)
        assert np.all(np.isfinite(result.data))

    def test_work_dir_hygiene(self, tmp_path):
        # R8: no fort.68/69/70/71/85, no dum.dum, no .045/.046 survive a
        # cleanup=True run, and the user's own files do.
        keep = tmp_path / 'keepme.txt'
        keep.write_text('x')
        result = _run(work_dir=tmp_path, cleanup=True)
        assert sorted(p.name for p in tmp_path.iterdir()) == ['keepme.txt']
        assert not [k for k in result.metadata if k.endswith('_file')]

    def test_pinned_work_dir_keeps_both_decks(self, tmp_path):
        result = _run(work_dir=tmp_path, cleanup=False)
        names = {p.name for p in tmp_path.iterdir()}
        assert {'oasp_run.dat', 'oasp_run.045', 'oasp_run.046',
                'oassp_run.dat', 'oassp_run.trf'} <= names
        assert result.metadata['rhs_file'].endswith('oasp_run.045')
        assert result.metadata['vol_file'].endswith('oasp_run.046')

    def test_interface_pinned_against_the_rhs(self, tmp_path):
        # Attaching the roughness spectrum to a layer the .rhs does not name
        # leaves CLEN = 0 on the one that scatters (oaseun31.f:100) and the run
        # exits 0 with no back-scatter — so the mismatch has to raise.
        with pytest.raises(ConfigurationError, match="disagrees with the"):
            _run(interface=_SEABED_LAYER - 1)

    def test_rms_roughness_scales_the_scattered_field(self, tmp_path):
        # unoassp30.f:601, :615: pow = |ROUGH(INTFCE)|, and CALSRC scales the
        # realized perturbation by it, so the scattered field is linear in the
        # rms roughness. This exercises carrier → writer → binary → reader.
        base = _run(rms_roughness=0.5)
        doubled = _run(rms_roughness=1.0)
        ratio = np.abs(doubled.data) / np.abs(base.data)
        assert np.allclose(ratio, 2.0, rtol=2e-2), (
            float(ratio.min()), float(ratio.max()))
