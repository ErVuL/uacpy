"""OASSP (OASES scattered-field realizations) writer, guard and chain tests.

OASSP is a post-processor: it consumes the ``.rhs`` an OASP run with option
``'s'`` writes and returns one realization of the scattered field, in OASP's
own ``.trf`` format. Everything a wrapper can get wrong here fails *silently* —
``GETOPT``'s closing ``ELSE`` is empty (``unoassp30.f:1050``), Block VIII is
overwritten from the ``.rhs`` with a warning covering only two of its four
fields (``:181-188``), and a roughness spectrum attached to the wrong layer
leaves ``CLEN = 0`` on the one that scatters (``oaseun31.f:102``). So the
writer tests below parse the deck off disk rather than re-deriving it, and the
chain tests read the binary's own echo.
"""

import os
import warnings

import numpy as np
import pytest

import uacpy
from uacpy.core.exceptions import (
    ConfigurationError, ModelExecutionError, UnsupportedFeatureError,
)
from uacpy.models.oases._common import _reject_wavenumber_overrun
from uacpy.models.oases.oassp import (
    _deck_wavenumber_count, _oassp_options, _oassp_wavenumber_echo_scale,
    _reject_roughness_wavenumber_overflow, _require_single_rough_interface,
)
from uacpy.io.oases_writer import write_oassp_input
from uacpy.io.oases_reader import read_oases_rhs_header
from uacpy.core import BoundaryProperties, Environment
from uacpy.core.environment import SeabedColumn, SedimentLayer
from uacpy.core.surface import Surface
from uacpy.models import OASS, OASSP
from uacpy.core.run_settings import RunMode
from uacpy.tests.conftest import make_pekeris
from uacpy.tests.conftest import recorded_warnings


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — the §5.4 geometry of the OASS/OASSP spec, kept small
# ─────────────────────────────────────────────────────────────────────────────

def _env(roughness=0.5):
    return make_pekeris(
        bathymetry=128.0,
        ssp=uacpy.SoundSpeedProfile(depths=[0, 128], sound_speed=[1500, 1500]),
        sound_speed=2300.0, density=2.65, roughness=roughness)


def _source():
    return uacpy.Source(depths=100.0, frequencies=500.0)


def _line_source():
    return uacpy.Source(depths=100.0, frequencies=500.0, source_type='line')


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
        # the record as V(1..6) + RG + CL + M. Without it CLEN stays 0 (:102)
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
            ssp=uacpy.SoundSpeedProfile(depths=[0, 128], sound_speed=[1500, 1500]),
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
        # G9: OASSP's GETOPT closes with an empty ELSE (unoassp30.f:1050-1051),
        # so a typo'd letter produces no diagnostic anywhere.
        with pytest.raises(ConfigurationError,
                           match="are not option letters this binary tests"):
            _write(tmp_path, options='N J s W')

    def test_option_letters_needing_unwritten_blocks_raise(self, tmp_path):
        # The letter is valid OASES; what is missing is a deck block this
        # writer emits, so the refusal is a capability of the writer rather
        # than an illegal argument — unlike the unknown letter above, which
        # stays a ConfigurationError.
        for letter in ('d', 'G', 'T', 'Z', 'l', 'v', 'b'):
            with pytest.raises(UnsupportedFeatureError,
                               match="produces no such input"):
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
        with pytest.raises(UnsupportedFeatureError,
                           match='does not support: volume scattering') as exc:
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
        # :613), and SCTRHS skips ROUGH2 < 1e-10 in the producer
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

    def test_warn_false_writes_the_same_deck_and_says_none_of_its_warnings(
            self, tmp_path):
        """``warn=False`` (OASSP's stage-3 wavenumber count writes the deck
        for its text alone) reaches every helper that warns: the deck
        below warns from write_oassp_input itself (c_high without 'P'),
        from _check_n_time_samples and from _format_upper_halfspace (a
        rigid surface, three calls down) with ``warn=True``, and says
        nothing with ``warn=False``, byte for byte the same deck."""
        env = make_pekeris(
            bathymetry=128.0,
            ssp=uacpy.SoundSpeedProfile(depths=[0, 128],
                                        sound_speed=[1500, 1500]),
            sound_speed=2300.0, density=2.65, roughness=0.5,
            surface=BoundaryProperties(acoustic_type='rigid'))
        decks, said = {}, {}
        for warn in (True, False):
            folder = tmp_path / str(warn)
            folder.mkdir()
            with recorded_warnings() as caught:
                _write(folder, env=env, warn=warn, c_high=1.0e5,
                       n_time_samples=500)
            decks[warn] = (folder / 'oassp_run.dat').read_bytes()
            said[warn] = [str(w.message) for w in caught]
        assert [m.split(' ')[:2] for m in said[True]] == [
            ['write_oassp_input:', 'n_time_samples=500'],
            ['write_oassp_input:', 'c_high=100000'],
            ['OASES', 'does']]
        assert said[False] == []
        assert decks[False] == decks[True]

    def test_c_low_reaches_the_deck(self, tmp_path):
        lines = _write(tmp_path, c_low=1200.0)
        assert float(lines[-3].split()[0]) == pytest.approx(1200.0)

    def test_a_keyword_the_writer_does_not_take_is_a_type_error(self, tmp_path):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
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
                         time_step=0.000833, integrand_plot_step=0,
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

    @pytest.mark.requires_oases
    @pytest.mark.parametrize('spectrum', ['gaussian', 'goff-jordan'])
    def test_the_two_real_spectra_are_accepted(self, spectrum):
        # The discriminating counterpart to test_bad_spectrum_raises: the
        # guard must not refuse the spectra OASES actually implements.
        uacpy.OASSP(correlation_length=10.0, roughness_spectrum=spectrum)

    def test_oassp_takes_no_multiple_scattering(self):
        # The letter would be parsed and then do nothing: every routine
        # unoassp30.f:677-688 dispatches to has the `if (.not.rescat)` guard
        # commented out with ROUGH2(II)=0 left live (oasvun31.f:76-80 and
        # three siblings), so the constructor does not offer it.
        with pytest.raises(TypeError, match='multiple_scattering'):
            uacpy.OASSP(correlation_length=10.0, multiple_scattering=True)

    @pytest.mark.requires_oases
    def test_oass_honours_multiple_scattering(self):
        # The discriminating counterpart — the asymmetry is real, not a
        # blanket refusal. oassun26.f:491, :685, :901 all test .not.rescat.
        assert uacpy.OASS(interface=3, correlation_length=10.0,
                          multiple_scattering=True).multiple_scattering


@pytest.mark.requires_oases
class TestModelConfiguration:

    def test_supported_run_modes(self):
        m = uacpy.OASSP(correlation_length=5.0)
        assert set(m.spec.modes) == {RunMode.BROADBAND, RunMode.TIME_SERIES}
        with pytest.raises(UnsupportedFeatureError,
                           match='does not support: RunMode.COHERENT_TL'):
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
                        roughness_spectrum='goff-jordan', realization=3,
                        n_time_samples=256)
        c = m.copy(realization=4)
        assert (c.realization, c.correlation_length, c.spectral_exponent,
                c.roughness_spectrum, c.n_time_samples) == (
                    4, 5.0, 2.5, 'goff-jordan', 256)
        # executable stays None so copy() re-resolves rather than re-pinning.
        assert c.executable is None

    def test_options_is_exclusive_with_the_typed_flags(self):
        with pytest.raises(ConfigurationError, match="discarded"):
            uacpy.OASSP(correlation_length=5.0, options='N J s',
                        roughness_spectrum='goff-jordan')

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
        m = uacpy.OASSP(correlation_length=5.0, roughness_spectrum='goff-jordan')
        assert set(_oassp_options(m.options, m.scattered_only,
                                  m.roughness_spectrum).split()) == \
            {'N', 'J', 's', 'g'}
        m = uacpy.OASSP(correlation_length=5.0, scattered_only=False)
        assert 's' not in _oassp_options(m.options, m.scattered_only,
                                         m.roughness_spectrum)

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

    def test_the_bare_spectrum_keyword_is_not_accepted(self):
        with pytest.raises(TypeError, match='spectrum'):
            uacpy.OASSP(correlation_length=5.0, spectrum='gaussian')

    def test_bad_spectrum_raises(self):
        with pytest.raises(ConfigurationError, match="spectrum"):
            uacpy.OASSP(correlation_length=5.0, roughness_spectrum='von-karman')

    def test_collapse_defaults(self):
        # A single spectral solve over one stack, as for OASP.
        # test_collapse_policy's table constructs every model with no
        # arguments, which OASSP's required correlation_length rules out, so
        # the per-model row lives here.
        from uacpy.models._projection import DEFAULT_COLLAPSE
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

    def test_an_interface_that_is_not_the_rough_one_is_refused_first(
            self, monkeypatch):
        """The .rhs names the environment's one rough bottom interface
        (deck layer 4 here); a pinned ``interface=`` that is not it would
        leave that layer with CLEN = 0 (oaseun31.f:102) and the run would
        exit 0 with no back-scatter. Decided from the environment, so every
        entry point refuses it and no mean field is run."""
        model = uacpy.OASSP(interface=_SEABED_LAYER + 3, correlation_length=5.0,
                            spectral_exponent=2.5, n_time_samples=256,
                            freq_min=400.0, freq_max=600.0)
        monkeypatch.setattr(model, '_run_subprocess',
                            lambda *a, **k: pytest.fail('launched'))
        for call in ('validate_inputs', 'run_settings', 'run'):
            with pytest.raises(ConfigurationError,
                               match=r"OASSP\(interface=7\) disagrees .* 4"):
                getattr(model, call)(_env(), _source(), _receiver(n_depths=2))

    def test_the_rough_interface_is_the_one_the_settings_record(self):
        settings = self._model().run_settings(_env(), _source(),
                                              _receiver(n_depths=2))
        assert settings.engine.interface == _SEABED_LAYER

    def test_rough_sea_surface_is_refused_by_name(self):
        env = uacpy.Environment(
            bathymetry=128.0,
            ssp=uacpy.SoundSpeedProfile(depths=[0, 128], sound_speed=[1500, 1500]),
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

    @pytest.mark.parametrize('freq_max, expected', [
        (600.0, 600.0),   # the mean field's band top above the source
        (500.0, 500.0),   # a band that ends at the source frequency
    ])
    def test_rayleigh_check_runs_at_the_band_top_the_deck_computes(
            self, monkeypatch, freq_max, expected):
        # P = 2 k sigma sin(theta) grows with k, so the small-roughness
        # check must see the highest frequency OASSP computes: FR2 of the
        # mean field's .rhs, not the 500 Hz source frequency. It is decided
        # in stage 3, from the grid the mean field's settings give.
        import uacpy.models.oases.oassp as oassp_module
        seen = {}

        def capture(_name, _env, _interface, frequency_hz, *_a, **_k):
            seen['f'] = frequency_hz
            return None

        monkeypatch.setattr(oassp_module, '_roughness_notice', capture)
        m = uacpy.OASSP(correlation_length=5.0, spectral_exponent=2.5,
                        n_time_samples=256, freq_min=400.0,
                        freq_max=freq_max)
        m.run_settings(_env(), _source(), _receiver(n_depths=2))
        assert seen['f'] == pytest.approx(expected)

    def test_the_roughness_notice_is_a_setting_the_preview_shows(self):
        """A roughness past the small-roughness theory at the band top is
        recorded in ``run_settings().engine.notices`` and announced by
        ``run_settings``, not by ``validate_inputs``."""
        m = uacpy.OASSP(correlation_length=5.0, spectral_exponent=2.5,
                        rms_roughness=3.0, n_time_samples=256,
                        freq_min=400.0, freq_max=600.0)
        with recorded_warnings() as caught:
            m.validate_inputs(_env(), _source(), _receiver(n_depths=2))
        assert not [w for w in caught if 'Rayleigh' in str(w.message)]
        with recorded_warnings() as caught:
            settings = m.run_settings(_env(), _source(),
                                      _receiver(n_depths=2))
        said = [str(w.message) for w in caught if 'Rayleigh' in str(w.message)]
        assert len(said) == 1 and said[0] in [n.message for n in settings.engine.notices]

    def test_the_settings_preview_leaves_the_decks_warnings_to_the_launch(
            self):
        """Stage 3 writes the deck for its wavenumber count only, so the
        writer's warnings (c_high without 'P') are not said by
        ``run_settings``; the launch writes the deck again and says them."""
        m = uacpy.OASSP(correlation_length=5.0, spectral_exponent=2.5,
                        c_high=1.0e5, n_time_samples=256, freq_min=400.0,
                        freq_max=600.0)
        with recorded_warnings() as caught:
            m.run_settings(_env(), _source(), _receiver(n_depths=2))
        assert not [w for w in caught if 'Full Hankel' in str(w.message)]

    def test_two_rough_bottom_interfaces_are_refused(self):
        from uacpy.core.environment import SeabedColumn, SedimentLayer
        env = uacpy.Environment(
            bathymetry=128.0,
            ssp=uacpy.SoundSpeedProfile(depths=[0, 128], sound_speed=[1500, 1500]),
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
        mean = result.components['mean_field']
        assert result.data.shape == mean.data.shape
        assert np.array_equal(result.coords['frequency'],
                              mean.coords['frequency'])
        assert np.iscomplexobj(result.data)
        assert np.all(np.isfinite(result.data))
        # The scattered field is not the mean field.
        assert not np.array_equal(result.data, mean.data)
        assert result.run_settings.engine.interface == _SEABED_LAYER
        # Stamped as the enum member, like every other wrapper's result:
        # PropagationModel._result_kwargs coerces, so ``str()`` renders
        # 'PhaseReference.TRAVELLING_WAVE' and the value still compares equal
        # to the plain string.
        from uacpy.core.results import PhaseReference
        assert result.phase_reference is PhaseReference.TRAVELLING_WAVE
        assert result.phase_reference == 'travelling_wave'

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
        mean = result.components['mean_field']
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

    def test_the_decks_warning_is_said_once_per_run(self):
        """The deck is written twice (the stage-3 wavenumber count, then the
        launch) and its warning is said once, by the launch."""
        with recorded_warnings() as caught:
            _run(c_high=1.0e5)
        assert len([w for w in caught
                    if 'Full Hankel' in str(w.message)]) == 1

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

    def test_the_rhs_is_held_to_the_resolved_interface(self, monkeypatch):
        # The deck is written from the settings; a .rhs naming another
        # interface is a resolution this wrapper got wrong, and is said so.
        model = uacpy.OASSP(correlation_length=5.0, spectral_exponent=2.5,
                            n_time_samples=256, freq_min=400.0,
                            freq_max=600.0)
        real = model._read_mean_field

        def other_interface(inputs, deck):
            run = real(inputs, deck)
            import dataclasses
            return run._replace(rhs_header=dataclasses.replace(
                run.rhs_header, interface=_SEABED_LAYER + 1))
        monkeypatch.setattr(model, '_read_mean_field', other_interface)
        with pytest.raises(ModelExecutionError, match='names interface 5'):
            model.run(_env(), _source(), _receiver(n_depths=2))

    def test_rms_roughness_scales_the_scattered_field(self, tmp_path):
        # unoassp30.f:601, :613: pow = |ROUGH(INTFCE)|, and CALSRC scales the
        # realized perturbation by it, so the scattered field is linear in the
        # rms roughness. This exercises carrier → writer → binary → reader.
        base = _run(rms_roughness=0.5)
        doubled = _run(rms_roughness=1.0)
        ratio = np.abs(doubled.data) / np.abs(base.data)
        assert np.allclose(ratio, 2.0, rtol=2e-2), (
            float(ratio.min()), float(ratio.max()))


# ─────────────────────────────────────────────────────────────────────────────
# The NP guard has to count what the binary integrates, not what it echoes
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.requires_oases
class TestPlaneGeometryDoublesTheWavenumberCount:
    """``unoassp30.f:327`` prints ``NO. OF WAVENUMBERS`` straight out of
    AUTSAM, and the plane-geometry branch immediately below (``:337-341``)
    then doubles it — "scattering kernels not symmetric", so the integration
    also runs over negative wavenumbers: ``nwvno = 2*nwvno`` with
    ``ICUT2 = NWVNO``. Reading the echo verbatim under-counted by 2x, so an
    echoed 40000 (80000 integrated, already past ``NP = 2**16``) passed the
    guard and left the meaningless full-size ``.trf`` its own docstring
    warns about. Cylindrical geometry keeps ``icut2 = nwvno`` and is
    unaffected; OASP prints its count after the branch that would have
    doubled it (``unoasp22.f:340-349``) and never doubles at all."""

    class _Proc:
        """Echo from a run whose real grid is 80000 in plane geometry."""
        stdout = ('    >>> Automatic Sampling <<<\n'
                  '    NO. OF WAVENUMBERS:       40000\n')
        stderr = ''
        returncode = 0

    @staticmethod
    def _model(**kw):
        kw.setdefault('correlation_length', 5.0)
        return uacpy.OASSP(**kw)

    def test_the_scale_follows_the_option_letter(self):
        assert _oassp_wavenumber_echo_scale('N J s') == 1
        assert _oassp_wavenumber_echo_scale('N J s P') == 2
        assert _oassp_wavenumber_echo_scale('NJP') == 2

    def test_a_line_source_writes_the_P_the_guard_reads(self):
        # n_wavenumbers pinned under the roughness branch's bound, which the
        # automatic count of this plane-geometry grid passes (stage 3).
        settings = self._model(n_wavenumbers=2048).run_settings(
            _env(), uacpy.Source(depths=100.0, frequencies=500.0,
                                 source_type='line'), _receiver())
        assert _oassp_wavenumber_echo_scale(settings.engine.options) == 2

    def test_plane_geometry_is_caught_at_half_the_echoed_bound(self):
        with pytest.raises(ConfigurationError, match='80000 wavenumbers'):
            _reject_wavenumber_overrun(
                self._model().model_name, self._Proc(),
                _oassp_wavenumber_echo_scale('N J s P'))

    def test_cylindrical_geometry_under_np_is_accepted_silently(self):
        # 40000 is under NP, and in cylindrical geometry that is the real
        # count — the guard must stay silent.
        _reject_wavenumber_overrun(self._model().model_name, self._Proc(),
                                   _oassp_wavenumber_echo_scale('N J s'))

    def test_oasp_never_doubles(self):
        # OASP's launch reads the echo at scale 1, whatever its geometry.
        _reject_wavenumber_overrun(uacpy.OASP().model_name, self._Proc(), 1)


@pytest.mark.requires_oases
def test_oassp_spectrum_spelling_is_normalised_before_the_options_check():
    """``roughness_spectrum`` was lower-cased for validation but compared literally
    against ``'gaussian'`` by the options-exclusivity check, so
    ``OASSP(roughness_spectrum='Gaussian', options=…)`` was rejected as if a
    non-default spectrum had been pinned. Normalising on the way in makes
    both comparisons exact, and matches OASS."""
    assert uacpy.OASSP(correlation_length=5.0,
                       roughness_spectrum='Gaussian').roughness_spectrum == 'gaussian'
    assert uacpy.OASSP(correlation_length=5.0,
                       roughness_spectrum='Goff-Jordan').roughness_spectrum == 'goff-jordan'
    # The default spelled with a capital is still the default.
    model = uacpy.OASSP(correlation_length=5.0, roughness_spectrum='Gaussian',
                        options='N J s')
    assert model.roughness_spectrum == 'gaussian'
    m = uacpy.OASSP(correlation_length=5.0, roughness_spectrum='Goff-Jordan')
    assert 'g' in _oassp_options(m.options, m.scattered_only,
                                 m.roughness_spectrum).split()
    with pytest.raises(ConfigurationError, match='von-karman'):
        uacpy.OASSP(correlation_length=5.0, roughness_spectrum='von-karman')


class TestScatteringSpectralExponent:
    """``M <= 1.5`` leaves the roughness power spectrum unintegrable
    (``oassp.tex:356-362``; the exponent reaches ``amod(m) = fac(3+…)`` at
    ``oaseun31.f:99``). The deck writer refuses it, but the scattering models
    reach their writer only after the mean-field binary has run, so the check
    also belongs on a constructor argument that cannot change in between."""

    @pytest.mark.parametrize('model_cls', [OASS, OASSP])
    @pytest.mark.parametrize('exponent', [1.5, 1.0, 0.0, -1.0])
    def test_an_unintegrable_exponent_is_refused(self, model_cls, exponent):
        with pytest.raises(ConfigurationError, match='spectral_exponent'):
            model_cls(correlation_length=10.0, spectral_exponent=exponent,
                      verbose=False)

    @pytest.mark.requires_oases
    @pytest.mark.parametrize('model_cls', [OASS, OASSP])
    @pytest.mark.parametrize('exponent', [1.51, 2.0, 3.5])
    def test_an_integrable_exponent_is_kept(self, model_cls, exponent):
        model = model_cls(correlation_length=10.0,
                          spectral_exponent=exponent, verbose=False)
        assert model.spectral_exponent == pytest.approx(exponent)

    @pytest.mark.requires_oases
    @pytest.mark.parametrize('model_cls', [OASS, OASSP])
    def test_the_default_exponent_is_accepted(self, model_cls):
        assert model_cls(correlation_length=10.0,
                         verbose=False).spectral_exponent == 2.0


@pytest.mark.requires_oases
class TestOasspTakesExactlyOneRoughBottomInterface:
    """``SCTRHS`` writes one ``.rhs`` record per rough interface per
    wavenumber, the sea surface's first (``oaseun31.f:2306-2310``, ``:2395``),
    and ``oassp2`` reads back one record per wavenumber with **no** interface
    filter, taking its scattering interface from the first record
    (``unoassp30.f:549``, ``oasvun31.f:66-70``) — unlike OASS, whose binary
    filters records by the deck's ``INTFC`` (``oassun26.f:360-362``). So a
    rough surface makes OASSP scatter from the water record's empty spectrum,
    and two rough bottoms interleave records the reader cannot separate. A
    mean field with no rough interface writes no SCTRHS records at all
    (skipped below ``ROUGH2 < 1e-10``, ``oaseun31.f:2310``), which is equally
    unusable.

    The check runs before the mean-field binary, because that is the run whose
    cost it saves.
    """

    @staticmethod
    def _model():
        return OASSP(correlation_length=5.0, spectral_exponent=2.5,
                     verbose=False)

    @staticmethod
    def _env(surface_roughness, *bottom_roughnesses):
        """Deepest roughness is the half-space; any earlier ones are layers."""
        layers = [SedimentLayer(thickness=5.0, sound_speed=1600.0,
                                density=1.6, attenuation=0.2, roughness=r)
                  for r in bottom_roughnesses[:-1]]
        return Environment(
            bathymetry=100.0, ssp=1500.0,
            surface=Surface(nodes=[BoundaryProperties(
                acoustic_type='vacuum', roughness=surface_roughness)]),
            bottom=SeabedColumn(layers=layers, halfspace=BoundaryProperties(
                acoustic_type='half-space', sound_speed=2300.0, density=2.65,
                attenuation=0.5, roughness=bottom_roughnesses[-1])))

    def test_one_rough_bottom_under_a_smooth_surface_is_accepted(self):
        _require_single_rough_interface(self._env(0.0, 0.5))

    def test_a_rough_sea_surface_is_refused(self):
        with pytest.raises(UnsupportedFeatureError, match='rough sea surface'):
            _require_single_rough_interface(self._env(0.5, 0.5))

    def test_two_rough_bottom_interfaces_are_refused(self):
        with pytest.raises(UnsupportedFeatureError,
                           match='2 rough bottom interfaces'):
            _require_single_rough_interface(self._env(0.0, 0.3, 0.5))

    def test_an_entirely_smooth_environment_is_refused(self):
        # No SCTRHS record is written at all, so the .rhs the scattering run
        # reads back has nothing in it.
        with pytest.raises(ConfigurationError, match='smooth'):
            _require_single_rough_interface(self._env(0.0, 0.0))


class TestRoughnessBranchWavenumberBound:
    """oassp2's roughness branch rounds NWVNO up to a power of two
    (``unoassp30.f:542-544``) and PV indexes ``cfk``/``sqrtp(nnkr = 8192)`` up
    to twice that (``oasvun31.f:2273``, ``:2532-2533``), so NWVNO > 4096
    crashes the binary (SIGSEGV in ``pv_``) or corrupts the ``.trf``. The
    automatic count grows with ``min(cref*NX*DT + 2*r_max, 6*r_max)``.

    The fixture is the measured case: 200 m of 1500-1520 m/s water over a
    rough 1600 m/s half-space, receivers at 30/60 m. oassp2 printed
    ``NO. OF WAVENUMBERS: 2243`` for ranges 500-1500 m and crashed for
    500-3000 m."""

    RCV_DEPTHS = [30.0, 60.0]

    @staticmethod
    def _rough_env():
        return uacpy.Environment(
            bathymetry=200.0,
            ssp=uacpy.SoundSpeedProfile.from_pairs([(0, 1500), (200, 1520)]),
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1600.0, density=1.8,
                attenuation=0.5, roughness=0.5))

    def _deck(self, tmp_path, ranges, n_wavenumbers=None):
        rcv = uacpy.Receiver(depths=self.RCV_DEPTHS,
                             ranges=np.asarray(ranges, dtype=float))
        path = tmp_path / 'oassp_run.dat'
        write_oassp_input(
            path, self._rough_env(),
            uacpy.Source(depths=50.0, frequencies=200.0), rcv, 'N J s',
            interface=4, correlation_length=5.0, spectral_exponent=2.0,
            n_time_samples=256, freq_min=0.0, freq_max=500.0,
            time_step=0.0010000000475, c_low=1350.0, n_wavenumbers=n_wavenumbers)
        return path.read_text(), rcv

    @pytest.mark.parametrize('ranges, expected', [
        ([500.0, 1000.0, 1500.0], 2243),        # printed by oassp2 itself
        (np.linspace(500.0, 3000.0, 3), 4216),  # the grid that crashed
    ])
    def test_the_predictor_reproduces_the_binarys_count(self, tmp_path,
                                                        ranges, expected):
        text, rcv = self._deck(tmp_path, ranges)
        assert _deck_wavenumber_count(text, rcv) == expected

    @pytest.mark.requires_oases
    def test_the_crashing_grid_is_refused_before_launch(self, tmp_path):
        text, rcv = self._deck(tmp_path, np.linspace(500.0, 3000.0, 3))
        with pytest.raises(ConfigurationError, match='4216') as err:
            _reject_roughness_wavenumber_overflow(text, rcv)
        assert 'n_wavenumbers=4096' in str(err.value.remediation)

    @pytest.mark.requires_oases
    def test_the_running_grid_passes(self, tmp_path):
        text, rcv = self._deck(tmp_path, [500.0, 1000.0, 1500.0])
        _reject_roughness_wavenumber_overflow(text, rcv)

    @pytest.mark.requires_oases
    @pytest.mark.parametrize('nw, refused', [(4096, False), (4097, True)])
    def test_a_pinned_count_is_judged_at_the_4096_bound(self, tmp_path, nw,
                                                        refused):
        text, rcv = self._deck(tmp_path, np.linspace(500.0, 3000.0, 3),
                               n_wavenumbers=nw)
        guard = _reject_roughness_wavenumber_overflow
        if refused:
            with pytest.raises(ConfigurationError, match='4097'):
                guard(text, rcv)
        else:
            guard(text, rcv)

    def test_the_overflow_is_refused_by_every_entry_point_before_a_launch(
            self, monkeypatch):
        """The count is predicted in stage 3 from the deck the settings
        write, so validate_inputs and run_settings refuse the crashing grid
        as run does, and no mean field is run for it."""
        rcv = uacpy.Receiver(depths=self.RCV_DEPTHS,
                             ranges=np.linspace(500.0, 3000.0, 3))
        src = uacpy.Source(depths=50.0, frequencies=200.0)
        model = OASSP(correlation_length=5.0, n_time_samples=256)
        monkeypatch.setattr(model, '_run_subprocess',
                            lambda *a, **k: pytest.fail('launched'))
        for call in ('validate_inputs', 'run_settings', 'run'):
            with pytest.raises(ConfigurationError, match='wavenumbers'):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    getattr(model, call)(self._rough_env(), src, rcv)

    def test_the_settings_record_the_count(self):
        rcv = uacpy.Receiver(depths=self.RCV_DEPTHS,
                             ranges=[500.0, 1000.0, 1500.0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            engine = OASSP(correlation_length=5.0, n_time_samples=256,
                           n_wavenumbers=4096).run_settings(
                self._rough_env(), uacpy.Source(depths=50.0,
                                                frequencies=200.0),
                rcv).engine
        assert engine.n_integrated_wavenumbers == (4096,)

    def test_each_source_depth_gets_the_count_of_its_own_deck(self):
        """Each source depth is its own deck and launch, and the count
        depends on the source-receiver separation: one count per depth."""
        rcv = uacpy.Receiver(depths=self.RCV_DEPTHS,
                             ranges=[500.0, 1000.0, 1500.0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            engine = OASSP(correlation_length=5.0, n_time_samples=256
                           ).run_settings(
                self._rough_env(), uacpy.Source(depths=[50.0, 120.0],
                                                frequencies=200.0),
                rcv).engine
        assert len(engine.n_integrated_wavenumbers) == 2
        assert engine.n_integrated_wavenumbers[0] == 2243

    @pytest.mark.slow
    @pytest.mark.requires_oases
    def test_the_advertised_remedy_runs_on_the_crashing_grid(self):
        # Without the pin the run is refused before oassp2 launches; with
        # n_wavenumbers=4096 it runs and returns a finite scattered field.
        rcv = uacpy.Receiver(depths=self.RCV_DEPTHS,
                             ranges=np.linspace(500.0, 3000.0, 3))
        src = uacpy.Source(depths=50.0, frequencies=200.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ConfigurationError, match='wavenumbers'):
                OASSP(correlation_length=5.0, n_time_samples=256).run(
                    self._rough_env(), src, rcv)
            field = OASSP(correlation_length=5.0, n_time_samples=256,
                          n_wavenumbers=4096).run(self._rough_env(), src, rcv)
        assert field.data.shape == (2, 3, 127)
        assert np.all(np.isfinite(field.data))
        assert np.nanmax(np.abs(field.data)) > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# OASES-7 — the mean field describes the problem the OASSP deck describes
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.requires_oases
class TestTheMeanFieldDescribesTheScatterersProblem:
    """"The input files for OASSP are virtually identical to the ones used
    for computing the mean field using OASP" (``oassp.tex:36-37``), and the
    manual's worked example (``:559-600``) carries 'P' and the same
    CMIN/CMAX on both decks. oassp2 takes its incident spectrum from the
    mean field's ``.rhs`` alone (``oasvun31.f:313-332``, ``:55-74``), so the
    geometry and the Block VII knobs must reach the mean field. Measured
    (100 m guide over a rough 1600 m/s seabed, 80-120 Hz): a plane-geometry
    OASSP driven by a cylindrical mean field was 11.8 dB off the manual's
    chain, with nothing said."""

    @staticmethod
    def _settings(model, source=None, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return model.run_settings(_env(), source or _source(),
                                      _receiver(n_depths=2), **kw)

    @staticmethod
    def _model(**kw):
        kw.setdefault('correlation_length', 5.0)
        kw.setdefault('spectral_exponent', 2.5)
        return uacpy.OASSP(**kw)

    def test_a_line_source_runs_both_decks_in_plane_geometry(self):
        assert set(OASSP.spec.source_types) == {'point', 'line'}
        # n_wavenumbers pinned under the roughness branch's bound, which the
        # automatic count of this plane-geometry grid passes (stage 3).
        engine = self._settings(self._model(n_time_samples=256,
                                            n_wavenumbers=2048),
                                source=_line_source()).engine
        assert 'P' in engine.options.split()
        assert engine.mean_field.source_type == 'line'
        assert 'P' in engine.mean_field.engine.options.split()

    def test_a_point_source_runs_both_decks_cylindrical(self):
        engine = self._settings(self._model(n_time_samples=256)).engine
        assert 'P' not in engine.options.split()
        mean = engine.mean_field
        assert mean.source_type == 'point'
        assert 'P' not in mean.engine.options.split()

    def test_the_block_vii_knobs_reach_the_default_mean_field(self):
        engine = self._settings(self._model(
            n_time_samples=256, c_low=1400.0, c_high=4000.0,
            n_wavenumbers=2048, integration_offset=0.5)).engine
        mean = engine.mean_field.engine
        assert (mean.c_low, mean.c_high, mean.n_wavenumbers,
                mean.integration_offset) == (1400.0, 4000.0, 2048, 0.5)
        assert engine.c_low == 1400.0

    def test_a_plane_mean_field_under_a_point_source_is_refused(self):
        # The Source decides both decks' geometry: a supplied mean field's
        # raw 'P' with a point Source is refused by the mean field's own
        # check, from every entry point, before anything runs.
        model = self._model(mean_field=uacpy.OASP(options='N J s P'))
        for call in ('validate_inputs', 'run_settings', 'run'):
            with pytest.raises(ConfigurationError,
                               match=r"OASP\(options='N J s P'\) carries 'P'"):
                getattr(model, call)(_env(), _source(), _receiver(n_depths=2))

    def test_a_raw_P_on_a_point_source_is_refused(self):
        model = self._model(options='N J s P')
        for call in ('validate_inputs', 'run_settings', 'run'):
            with pytest.raises(ConfigurationError,
                               match=r"OASSP\(options='N J s P'\) carries 'P'"):
                getattr(model, call)(_env(), _source(), _receiver(n_depths=2))

    @pytest.mark.parametrize('nw, refused', [(8192, False), (8193, True)])
    def test_the_mean_fields_pinned_count_is_bounded_by_nnkr(self, nw,
                                                             refused):
        # OASSP's own count pinned under its roughness bound: the mean
        # field's 4096-sample default window sets 8142 automatically.
        model = self._model(n_wavenumbers=4096,
                            mean_field=uacpy.OASP(options='N J s',
                                                  n_wavenumbers=nw))
        if refused:
            with pytest.raises(ConfigurationError, match='NKMEAN'):
                self._settings(model)
        else:
            assert self._settings(model).engine.mean_field.engine \
                .n_wavenumbers == nw

    def test_a_mean_field_cmin_other_than_the_decks_is_named(self):
        # OASSP's own count pinned under its roughness bound, which the
        # mean field's 4096-sample default window passes (stage 3).
        model = self._model(n_wavenumbers=4096,
                            mean_field=uacpy.OASP(options='N J s',
                                                  c_low=1300.0))
        with pytest.warns(UserWarning, match="differs from the mean field"):
            model.run_settings(_env(), _source(), _receiver(n_depths=2))
        same = self._model(n_wavenumbers=4096,
                           mean_field=uacpy.OASP(options='N J s',
                                                 c_low=1350.0),
                           c_low=1350.0)
        with recorded_warnings() as caught:
            same.run_settings(_env(), _source(), _receiver(n_depths=2))
        assert not [w for w in caught
                    if 'differs from the mean field' in str(w.message)]

    @pytest.mark.slow
    def test_the_mean_field_deck_the_binary_reads_carries_the_geometry(
            self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._model(n_time_samples=256, freq_min=400.0, freq_max=600.0,
                        work_dir=tmp_path, cleanup=False).run(
                _env(), _line_source(), _receiver(n_depths=2))
        option_line = (tmp_path / 'oasp_run.dat').read_text().splitlines()[1]
        assert 'P' in option_line.split()


@pytest.mark.slow
@pytest.mark.requires_oases
def test_oassps_mean_field_records_paths_only_when_the_directory_survives(
        tmp_path):
    """The mean field's files share OASSP's directory, so its result
    records their paths exactly when OASSP's ``cleanup`` keeps them."""
    cleaned = _run().components['mean_field']
    assert not [k for k in cleaned.metadata if k.endswith('_file')]
    kept = _run(work_dir=tmp_path / 'wd', cleanup=False).components[
        'mean_field']
    paths = [v for k, v in kept.metadata.items() if k.endswith('_file')]
    assert paths and all(os.path.exists(p) for p in paths)
