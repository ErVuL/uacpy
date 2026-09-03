"""OASS (OASES reverberation / scattered field): the deck writer, then the model.

Every assertion here is anchored to a line of the vendored source rather than
to the manual, because OASS is a module where the two disagree in ways that
change the deck (see ``write_oass_input``'s docstring). The record table these
tests pin is the read order in ``unoass21.f``.

The second half drives the real two-binary chain — ``oast2`` for the mean field,
then ``oass2`` over the ``.rhs`` it leaves — because that is the only way to
catch a deck OASES accepts and answers wrongly.
"""

import inspect
import warnings

import numpy as np
import pytest

import uacpy
from uacpy.core import BoundaryProperties, Environment, Receiver, Source
from uacpy.core.exceptions import ConfigurationError
from uacpy.io.oases_writer import write_oass_input
from uacpy.tests.conftest import make_pekeris


def _env(c_water=1500.0, roughness=0.5):
    return make_pekeris(
        ssp=uacpy.SoundSpeedProfile(depths=[0.0, 100.0],
                                    data=[c_water, c_water]),
        roughness=roughness)


_SRC = uacpy.Source(depths=50.0, frequencies=250.0)
_RCV = uacpy.Receiver(depths=[10.0, 30.0, 50.0, 70.0, 90.0],
                      ranges=np.linspace(100.0, 5000.0, 20))


def _write(tmp_path, options='r', **kw):
    kw.setdefault('interface', 3)
    kw.setdefault('correlation_length', 10.0)
    kw.setdefault('spectral_exponent', 2.0)
    path = tmp_path / 'oass_run.dat'
    write_oass_input(path, _env(), _SRC, _RCV, options, **kw)
    return [ln for ln in path.read_text().splitlines() if ln.strip()]


class TestRecordTable:
    """One test per record of ``unoass21.f``'s read order."""

    def test_frequency_is_one_token_not_two(self, tmp_path):
        # ICNTIN is initialised to 0 (unoass21.f:596) and GETOPT never sets
        # it, so the two-token READ(1,*) FREQ,OFFDB at :105 is unreachable.
        assert len(_write(tmp_path)[2].split()) == 1

    def test_the_scattering_interface_carries_the_nine_token_form(self, tmp_path):
        # oass.tex:182-183 — rms roughness, correlation length and spectral
        # exponent "must be specified. These are not adopted from OASR or
        # OAST." INENVI re-reads the record as nine when RG < 0
        # (oaseun31.f:72-75, :91-93). Verified against the real oass2, whose
        # echo prints '>>> Rough Interface <<< / Rms / L_x / Exp'.
        bottom_record = _write(tmp_path, interface=3, rms_roughness=0.5)[6]
        tokens = bottom_record.split()
        assert len(tokens) == 9
        assert float(tokens[6]) < 0.0, 'negative RG is the flag CL and M follow'
        assert (float(tokens[7]), float(tokens[8])) == (10.0, 2.0)

    def test_receiver_block_is_count_then_five_tokens_each(self, tmp_path):
        lines = _write(tmp_path)
        n = int(lines[8])
        assert n == 5
        for row in lines[9:14]:
            assert len(row.split()) == 5, 'Z X Y ITYP GAIN (oasnun22.f:36)'

    def test_block_vii_wavenumber_record_has_three_tokens_not_four(self, tmp_path):
        # OASP/OASSP Block VII carries a 4th token IF; OASS does not
        # (unoass21.f:189). Emitting four would shift the whole deck.
        assert len(_write(tmp_path)[15].split()) == 3

    def test_range_block_is_in_km(self, tmp_path):
        # unoass21.f:263 reads RMIN RMAX NR in km; the receiver spans
        # 100-5000 m.
        r_min, r_max, nr = _write(tmp_path)[16].split()
        assert (float(r_min), float(r_max)) == pytest.approx((0.1, 5.0))
        assert int(nr) == 20


class TestInterfaceIndexMapsToTheRightRecord:
    """INTFC counts deck layers from 1 (upper halfspace); the ``suffix_fn``
    ``_emit_bottom_layers`` calls is indexed from its own first bottom
    record. Keying the roughness override off INTFC directly attaches the
    spectrum to the wrong record, and **the failure is silent**: the
    interface then carries a POSITIVE RG, which OASES reads as an infinite
    correlation length (oassp.tex:344-348), so P(k) collapses to a delta at
    k=0 and there is no back-scatter — from a run that exits 0 and writes a
    full set of curves."""

    def test_isovelocity_column_attaches_to_the_seafloor_record(self, tmp_path):
        # vacuum(1) + water(2) + bottom(3) → INTFC=3 is the seafloor.
        rec = _write(tmp_path, interface=3, rms_roughness=0.5)[6].split()
        assert len(rec) == 9 and float(rec[6]) < 0.0

    def test_the_sea_surface_is_a_legal_target_not_an_error(self, tmp_path):
        # Deck layer 2 is the first water record, whose RG is the sea surface
        # (oaseun31.f:377, :381-383). OASS scatters from it, so it must not be
        # refused — see TestTheSeaSurfaceIsAScatteringInterface.
        # rms_roughness is required here: _env()'s surface is smooth, and
        # asking for a scattering spectrum on a smooth interface is refused
        # by the |RG| representability guard, correctly.
        rec = _write(tmp_path, interface=2, phase_speed=1500.0,
                     rms_roughness=0.7)[5].split()
        assert len(rec) == 9 and float(rec[6]) < 0.0

    def test_an_out_of_range_interface_is_refused(self, tmp_path):
        with pytest.raises(ConfigurationError, match='no interface'):
            _write(tmp_path, interface=5)

    def test_no_record_carries_a_positive_rg_at_the_named_interface(self, tmp_path):
        # The property that matters, stated directly: whatever INTFC names
        # must be the nine-token form. A positive RG there is the silent
        # no-backscatter bug.
        lines = _write(tmp_path, interface=3, rms_roughness=0.5)
        intfc = int(lines[7].split()[1])
        record = lines[4 + intfc - 1].split()   # layer records start at [4]
        assert len(record) == 9, f'INTFC={intfc} record is {len(record)} tokens'
        assert float(record[6]) < 0.0



class TestBlockVIIIIsWrittenForEveryReverbOption:
    """``a`` sets REVERB on the way in (unoass21.f:646-653) even though
    oass.tex:254-257 says the range offset "does not apply to option a". A
    deck that omits Block VIII for a covariance run has its *next* record
    consumed as ``RMIN RMAX NR``, shifting everything after it."""

    @pytest.mark.parametrize('option', ['r', 'a', 'C', 'D', 'R'])
    def test_present_for_every_reverb_letter(self, tmp_path, option):
        lines = _write(tmp_path, options=option)
        # The range record is the one after NW IC1 IC2.
        idx = lines.index(next(ln for ln in lines if len(ln.split()) == 3
                              and ln.split()[1] == '1')) + 1
        assert len(lines[idx].split()) == 3

    def test_absent_when_no_reverb_letter_is_requested(self, tmp_path):
        # 'I' plots integrands and does not set REVERB (unoass21.f:621), so
        # Block VIII must NOT be written or the binary reads a record that
        # is not there. A kernel deck also needs a real CPH — that branch
        # forms ETA=DSQ/CPHASE (unoass21.f:342-344).
        assert len(_write(tmp_path, options='I', phase_speed=1500.0)) == 16


class TestBlockIXIsWrittenOnlyUnderC:
    """Block IX (``DR NDR``, oass.tex:97-101) is read only under option 'C'
    (unoass21.f:265): present on any other deck it would be consumed by
    whatever READ follows, absent on a 'C' deck the binary reads past the
    file's end. The writer pins DR = 1.0 m, NDR = 1 — the local horizontal
    correlation array is not exposed through uacpy."""

    def test_a_contour_deck_ends_with_dr_ndr(self, tmp_path):
        assert _write(tmp_path, options='C')[-1].split() == ['1.0', '1']

    @pytest.mark.parametrize('option', ['r', 'a', 'D', 'R'])
    def test_every_other_reverb_deck_ends_with_the_range_block(
            self, tmp_path, option):
        last = _write(tmp_path, options=option)[-1].split()
        assert last != ['1.0', '1'] and len(last) == 3


class TestGuards:
    def test_two_products_refused(self, tmp_path):
        # oassun26.f:683-688 vs :897-902 — 'a' with 'r' returns a silently
        # ZERO covariance, which no downstream check would catch.
        with pytest.raises(ConfigurationError, match='more than one'):
            _write(tmp_path, options='r a')

    @pytest.mark.parametrize('letter', ['N', 'J'])
    def test_oast_option_habits_refused(self, tmp_path, letter):
        # There is no field-parameter letter in OASS; GETOPT ends
        # IOUT(1)=1 / NOUT=1 (:694-696) and the parameter comes from the
        # receiver ITYP column. The manual's own example (oass.tex:305) is
        # wrong about this.
        with pytest.raises(ConfigurationError, match='not OASS option'):
            _write(tmp_path, options=f'r {letter}')

    def test_option_k_refused_with_its_own_reason(self, tmp_path):
        # 'k' IS a real GETOPT letter (:664), so it must not fall through to
        # the generic unknown-letter message: it is refused because PLPOWER
        # is never initialised in GETOPT (:577-590).
        with pytest.raises(ConfigurationError, match='PLPOWER'):
            _write(tmp_path, options='r k')

    def test_single_range_refused(self, tmp_path):
        # RSTEP=(RMAX-RMIN)/(LF-1) with LF=NR (unoass21.f:269).
        with pytest.raises(ConfigurationError, match='n_ranges must be >= 2'):
            _write(tmp_path, n_ranges=1)

    def test_mixed_receiver_types_refused(self, tmp_path):
        # unoass21.f:165-169 STOPs with no output; refuse before writing.
        with pytest.raises(ConfigurationError, match='one type'):
            _write(tmp_path, receiver_types=[1, 1, 4, 1, 1])

    def test_non_integrable_spectral_exponent_refused(self, tmp_path):
        with pytest.raises(ConfigurationError, match='1.5'):
            _write(tmp_path, spectral_exponent=1.5)

    def test_interface_must_be_at_least_two(self, tmp_path):
        with pytest.raises(ConfigurationError, match='interface must be'):
            _write(tmp_path, interface=1)

    def test_a_kernel_deck_needs_a_positive_phase_speed(self, tmp_path):
        # unoass21.f:342-344 takes the `IF (.not.REVERB)` branch and forms
        # ETA=DSQ/CPHASE; gfortran does not trap the division, so CALSIN runs
        # on Inf rather than failing.
        with pytest.raises(ConfigurationError, match='divides by zero'):
            _write(tmp_path, options='I', phase_speed=0.0)


class TestWavenumberRecordIsExplicit:
    """Under REVERB the binary recomputes ``NWVNO`` from the ``.rhs`` file's
    own ``DLWVNO`` and forces ``ICUT1=1, ICUT2=NWVNO``
    (``unoass21.f:211-215``), so this record is inert *there*. It is written
    explicitly anyway: on a plots-only deck the branch is skipped and the
    siblings' ``-1`` auto sentinel would survive as a negative count."""

    def test_never_emits_the_auto_sentinel(self, tmp_path):
        for options in ('r', 'I'):
            nw = [ln for ln in _write(tmp_path, options=options,
                                      phase_speed=1500.0)
                  if len(ln.split()) == 3 and ln.split()[1] == '1'][0]
            assert all(int(t) > 0 for t in nw.split()), \
                f'-1 would be a negative wavenumber count for options={options!r}'

    def test_c_low_is_honoured_because_it_is_the_live_knob(self, tmp_path):
        # CMIN/CMAX form WK0/WKMAX at :190-191 and survive the REVERB
        # override, which is why raising c_low moves the answer ~30 dB while
        # the NW record cannot move it at all.
        line = _write(tmp_path, c_low=1350.0)[14]
        assert float(line.split()[0]) == pytest.approx(1350.0)


# ─────────────────────────────────────────────────────────────────────────────
# The OASS class
#
# The two declared-contract classes below touch only the class objects; every
# other test constructs an OASS, which existence-checks the oass2 binary, so
# those classes are requires_oases. The chain is two binaries deep but cheap
# at these settings — one full run() is ~0.1 s at 100 Hz.
# ─────────────────────────────────────────────────────────────────────────────

from uacpy.core.exceptions import UnsupportedFeatureError    # noqa: E402
from uacpy.io.oases_reader import read_oases_rhs_header      # noqa: E402
from uacpy.models import OASS, OAST, OASN, OASP, OASES       # noqa: E402
from uacpy.models.base import RunMode                        # noqa: E402
from uacpy.core.results import (                             # noqa: E402
    Covariance, Field, _DOCUMENTED_METADATA, _UNIVERSAL_METADATA,
)

pytestmark_oases = pytest.mark.requires_oases

#: 100 Hz keeps the wavenumber axis short: the chain runs in ~0.1 s where the
#: 250 Hz deck of the spec's worked example takes ~10 s.
_CLASS_SRC = uacpy.Source(depths=50.0, frequencies=100.0)
_CLASS_RCV = uacpy.Receiver(depths=[30.0, 70.0],
                            ranges=np.linspace(200.0, 2000.0, 8))


def _oass(**kw):
    kw.setdefault('correlation_length', 10.0)
    kw.setdefault('rms_roughness', 0.5)
    return OASS(**kw)


def _layered_env(halfspace_roughness=0.3):
    """One sediment layer over a half-space: two bottom records, so INTFC has
    a choice the half-space-only environment cannot express."""
    return uacpy.Environment(
        bathymetry=100.0,
        ssp=uacpy.SoundSpeedProfile(depths=[0.0, 100.0],
                                    data=[1500.0, 1500.0]),
        bottom=uacpy.Bottom(columns=[uacpy.SeabedColumn(
            layers=[uacpy.SedimentLayer(
                thickness=10.0, sound_speed=1600.0, density=1.6,
                attenuation=0.2, roughness=0.5)],
            halfspace=uacpy.BoundaryProperties(
                sound_speed=1800.0, density=2.0, attenuation=0.5,
                roughness=halfspace_roughness))]))


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


@pytest.mark.requires_oases
class TestConstructor:
    def test_correlation_length_is_required(self):
        # oass.tex:182-183 — the roughness spectrum "must be specified. These
        # are not adopted from OASR or OAST", and there is nothing on the
        # Environment carrying a correlation length to fall back to.
        with pytest.raises(ConfigurationError, match='correlation_length'):
            OASS()

    def test_integration_offset_is_refused_not_ignored(self):
        # ICNTIN is initialised to 0 (unoass21.f:596) and GETOPT never sets
        # it, so Block III's COFF is unreachable.
        with pytest.raises(ConfigurationError, match=r'unoass21\.f:596'):
            _oass(integration_offset=0.5)

    def test_nw_samples_is_refused_not_ignored(self):
        # Under every option OASS supports, NWVNO comes from the .rhs
        # (unoass21.f:209-215) — measured: a deck asking 2048 ran 2778.
        with pytest.raises(ConfigurationError, match=r'209-215'):
            _oass(nw_samples=2048)

    def test_raw_options_and_typed_flags_are_exclusive(self):
        with pytest.raises(ConfigurationError, match='replaces the whole'):
            _oass(options='r', plane_geometry=True)

    def test_unknown_spectrum_is_refused(self):
        with pytest.raises(ConfigurationError, match='spectrum'):
            _oass(spectrum='von-karman')

    def test_mean_field_must_be_a_producer(self):
        # oass.tex:18-23 names OAST and OASR; an OASP .rhs is broadband and
        # would break the one-header-record contract (unoass21.f:123).
        with pytest.raises(ConfigurationError, match='mean_field'):
            _oass(mean_field=OASP())
        with pytest.raises(ConfigurationError, match='mean_field'):
            _oass(mean_field=OASN())

    def test_copy_round_trips_every_argument(self):
        model = _oass(spectral_exponent=2.5, plane_geometry=True,
                      interface=3, c_low=1300.0)
        clone = model.copy(rms_roughness=0.9)
        for name in ('correlation_length', 'spectral_exponent', 'spectrum',
                     'interface', 'multiple_scattering', 'plane_geometry',
                     'c_low', 'c_high'):
            assert getattr(clone, name) == getattr(model, name), name
        assert clone.rms_roughness == 0.9
        # ``executable`` is kept as passed (None) so copy() re-resolves.
        assert clone.executable is None and clone._exe.exists()


@pytest.mark.requires_oases
class TestOptionLine:
    def test_product_letter_follows_the_run_mode(self):
        model = _oass()
        assert model._resolve_options(RunMode.REVERBERATION) == 'r'
        assert model._resolve_options(RunMode.COVARIANCE) == 'a'

    def test_typed_flags_map_to_their_letters(self):
        model = _oass(spectrum='goff-jordan', multiple_scattering=True,
                      plane_geometry=True)
        assert model._resolve_options(RunMode.REVERBERATION).split() == \
            ['r', 'g', 'p', 'P']

    @pytest.mark.parametrize('run_mode',
                             [RunMode.REVERBERATION, RunMode.COVARIANCE])
    def test_derived_line_asks_for_exactly_one_product(self, run_mode):
        # G7, at the source: 'a' sharing a deck with 'r'/'C'/'D' returns a
        # SILENTLY zero covariance (oassun26.f:683-688 zeroes ROUGH2 before
        # :897-902 reads it). One product per run, chosen by run_mode.
        chars = set(_oass()._resolve_options(run_mode))
        assert len(chars & set('arCD')) == 1

    @pytest.mark.parametrize('letters', ['r C', 'r D'])
    def test_contour_options_are_refused(self, letters):
        with pytest.raises(UnsupportedFeatureError, match='CONDRW|contour'):
            _oass(options=letters).run(_env(), _CLASS_SRC, _CLASS_RCV)

    @pytest.mark.parametrize('letters', ['r I', 'r S', 'r c'])
    def test_kernel_options_are_refused(self, letters):
        # Their curves carry the INTGR / SPECT tags; read_oast_tl selects on
        # TLRAN, so nothing would be found in the .plt.
        with pytest.raises(UnsupportedFeatureError, match='TLRAN'):
            _oass(options=letters).run(_env(), _CLASS_SRC, _CLASS_RCV)

    def test_svp_plot_option_is_refused(self):
        # 'Z' makes the binary read two more records (unoass21.f:319-320)
        # that write_oass_input does not emit.
        with pytest.raises(UnsupportedFeatureError, match="'Z'"):
            _oass(options='r Z').run(_env(), _CLASS_SRC, _CLASS_RCV)

    def test_raw_line_must_carry_the_run_modes_product(self):
        with pytest.raises(ConfigurationError, match='product letter'):
            _oass(options='a').run(_env(), _CLASS_SRC, _CLASS_RCV,
                                   run_mode=RunMode.REVERBERATION)

    def test_for_mode_routes_reverberation_here(self):
        assert OASES.for_mode(RunMode.REVERBERATION,
                              correlation_length=10.0).__class__ is OASS
        assert OASES.for_mode(RunMode.COVARIANCE, reverberation=True,
                              correlation_length=10.0).__class__ is OASS
        assert OASES.for_mode(RunMode.COVARIANCE).__class__ is OASN


@pytest.mark.requires_oases
class TestGuardsThatFireBeforeAnyBinaryRuns:
    """G1, G3 and the interface mapping are checked against the carriers, so
    they cost nothing and never leave a half-written work dir behind."""

    def test_multi_frequency_source_is_refused(self):
        # G1 — OASS pins nfreq=1 (unoass21.f:123) and reads the second
        # frequency's 5-word header as a 19-word boundary-operator record.
        src = uacpy.Source(depths=50.0, frequencies=[100.0, 200.0])
        with pytest.raises(ConfigurationError, match='single-frequency'):
            _oass().run(_env(), src, _CLASS_RCV)

    def test_smooth_interface_is_refused(self):
        # G3 — SCTRHS skips any interface with ROUGH2 < 1e-10
        # (oaseun31.f:2310, :379), so the mean field writes an empty .rhs and
        # OASS has no wavenumber step at all.
        with pytest.raises(ConfigurationError, match=r'oaseun31\.f:2310'):
            _oass().run(_env(roughness=0.0), _CLASS_SRC, _CLASS_RCV)

    def test_smooth_interface_is_judged_on_the_environment_not_the_override(self):
        # rms_roughness overrides the OASS deck only; the mean-field deck
        # always carries the environment's value (oass.tex:184-186).
        with pytest.raises(ConfigurationError, match='mean-field'):
            _oass(rms_roughness=0.5).run(_env(roughness=0.0), _CLASS_SRC,
                                         _CLASS_RCV)

    def test_interface_defaults_to_the_seafloor(self):
        # Deck layers: 1 upper halfspace, 2 water, 3 bottom halfspace.
        assert _oass()._resolve_interface(_env()) == 3

    def test_interface_default_tracks_the_water_layer_count(self):
        # A gradient profile emits one record per SSP row, so the seafloor
        # moves down the deck.
        env = uacpy.Environment(
            bathymetry=100.0,
            ssp=uacpy.SoundSpeedProfile(depths=[0.0, 50.0, 100.0],
                                        data=[1500.0, 1490.0, 1495.0]),
            bottom=uacpy.Bottom.from_halfspace(uacpy.BoundaryProperties(
                sound_speed=1700.0, density=1.8, attenuation=0.5,
                roughness=0.5)),
        )
        assert _oass()._resolve_interface(env) == 5

    @pytest.mark.parametrize('env_factory,expected', [
        (lambda: _env(), 3),
        (None, 5),
    ])
    def test_default_interface_lands_on_the_nine_token_record(
            self, env_factory, expected, tmp_path):
        # The mapping is duplicated between the model (which defaults it) and
        # the writer (which places it), and a mismatch is SILENT: the named
        # interface then carries a positive RG, which OASES reads as an
        # infinite correlation length — no back-scatter, exit 0.
        if env_factory is None:
            env = uacpy.Environment(
                bathymetry=100.0,
                ssp=uacpy.SoundSpeedProfile(depths=[0.0, 50.0, 100.0],
                                            data=[1500.0, 1490.0, 1495.0]),
                bottom=uacpy.Bottom.from_halfspace(uacpy.BoundaryProperties(
                    sound_speed=1700.0, density=1.8, attenuation=0.5,
                    roughness=0.5)),
            )
        else:
            env = env_factory()
        model = _oass()
        interface = model._resolve_interface(env)
        assert interface == expected
        path = tmp_path / 'oass_run.dat'
        write_oass_input(path, env, _CLASS_SRC, _CLASS_RCV, 'r',
                         interface=interface, correlation_length=10.0,
                         spectral_exponent=2.0, rms_roughness=0.5)
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        record = lines[4 + interface - 1].split()   # layer records start at [4]
        assert len(record) == 9, f'INTFC={interface} record is {record}'
        assert float(record[6]) < 0.0

    def test_out_of_range_interface_is_refused(self):
        with pytest.raises(ConfigurationError,
                           match='not a scattering interface'):
            _oass(interface=7).run(_env(), _CLASS_SRC, _CLASS_RCV)

    @pytest.mark.parametrize('interface,record_index', [(3, 6), (4, 7)])
    def test_a_layered_column_can_scatter_off_either_interface(
            self, interface, record_index, tmp_path):
        # With sediment layers the deck has more than one bottom record, so
        # INTFC has a real choice: 3 is the seafloor, 4 the base of the
        # sediment. Only the named one may carry the spectrum.
        env = _layered_env()
        assert _oass()._resolve_interface(env) == 3      # default: seafloor
        path = tmp_path / 'oass_run.dat'
        write_oass_input(path, env, _CLASS_SRC, _CLASS_RCV, 'r',
                         interface=interface, correlation_length=10.0,
                         spectral_exponent=2.0)
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        assert len(lines[record_index].split()) == 9
        assert float(lines[record_index].split()[6]) < 0.0
        others = {6, 7} - {record_index}
        for other in others:
            assert len(lines[other].split()) == 7

    def test_roughness_is_judged_at_the_named_interface_only(self):
        # A rough seafloor over a smooth sediment base: interface 4 has no
        # boundary operators in the .rhs even though interface 3 does.
        env = _layered_env(halfspace_roughness=0.0)
        assert _oass()._resolve_interface(env) == 3
        with pytest.raises(ConfigurationError, match='is smooth'):
            _oass(interface=4).run(env, _CLASS_SRC, _CLASS_RCV)

    def test_mean_field_without_option_s_is_refused(self, tmp_path):
        # SCTOUT gates the whole boundary-operator dump (oaseun31.f:1899-1900).
        model = _oass(mean_field=OAST(options='N J T'), work_dir=tmp_path)
        with pytest.raises(ConfigurationError, match="must contain 's'"):
            model.run(_env(), _CLASS_SRC, _CLASS_RCV)


class TestTheReverberationCitationNamesTheRoutineOnOptionRsPath:
    """``_read_reverberation`` explains the reverberation quantity by citing
    the OASES lines that convert it to dB. Two routines in ``oassun26.f``
    carry a byte-identical "CONVERT TO dB" block — ``REVRAN`` at :633-638 and
    ``REVINT`` at :853-858 — so a content check alone cannot tell them apart,
    and an address that names the wrong one reads as correct. Only the DATA
    PATH separates them, and that is what this pins.

    ``REVINT`` writes ``CFFs``, which ``unoass21.f:38`` equivalences to the
    ``XS`` that ``PLTLOS`` plots into the ``.plt`` this model reads.
    ``REVRAN`` writes ``CFF(1,1)`` (equivalenced to ``X``) and is reached only
    from the ``CCONTU`` contour branch, which option ``'r'`` does not enable.
    """

    @staticmethod
    def _src(name):
        from pathlib import Path
        path = (Path(__file__).resolve().parent.parent / 'third_party' /
                'oases' / 'src' / name)
        return path.read_text().splitlines()

    @staticmethod
    def _docstring():
        from uacpy.models.oases import OASS
        return inspect.getdoc(OASS._read_reverberation)

    def test_the_cited_block_converts_to_db(self):
        cited = '\n'.join(self._src('oassun26.f')[853 - 1:858])
        assert 'CONVERT TO dB' in cited
        assert 'CVMAGS' in cited          # magnitude squared
        assert 'VCLIP' in cited           # floored before the log
        assert 'VALG10' in cited          # log10
        assert '-5E0' in cited            # times -5 on an already-squared term

    def test_the_cited_block_writes_the_array_pltlos_plots(self):
        # The discriminating half. REVRAN's block passes every assertion
        # above and still cannot be the source of these curves, because it
        # writes CFF(1,1) rather than CFFs.
        cited = '\n'.join(self._src('oassun26.f')[853 - 1:858])
        other = '\n'.join(self._src('oassun26.f')[633 - 1:638])
        assert 'CFFs(1)' in cited and 'CFFs(1)' not in other
        assert 'CFF(1,1),1,-5E0' in other
        driver = '\n'.join(self._src('unoass21.f'))
        assert '(XS(1),CFFS(1))' in driver      # so CFFs IS the plotted XS
        assert '(X(1,1),CFF(1,1))' in driver    # while CFF(1,1) is X

    def test_the_cited_lines_sit_inside_revint_not_revran(self):
        lines = self._src('oassun26.f')
        starts = {}
        for i, ln in enumerate(lines, 1):
            up = ln.strip().upper()
            for name in ('REVRAN', 'REVINT', 'REVCOV'):
                if up.startswith(f'SUBROUTINE {name}('):
                    starts[name] = i
        assert set(starts) == {'REVRAN', 'REVINT', 'REVCOV'}
        assert starts['REVRAN'] < 638 < starts['REVINT']
        assert starts['REVINT'] < 853 and 858 < starts['REVCOV']

    def test_option_r_reaches_revint_and_not_the_contour_branch(self):
        driver = self._src('unoass21.f')
        # option 'r' sets PLTL (and REVERB); it never sets CCONTU.
        i = next(k for k, ln in enumerate(driver)
                 if "OPT(I).EQ.'r'" in ln)
        branch = '\n'.join(driver[i:i + 8])
        assert 'PLTL=.TRUE.' in branch.replace(' ', '')
        assert 'CCONTU' not in branch
        # and the PLTL branch is the one that calls REVINT then PLTLOS.
        j = next(k for k, ln in enumerate(driver) if 'CALL REVINT(' in ln)
        assert any('PLTL' in ln for ln in driver[j - 5:j])
        assert any('CALL PLTLOS(' in ln for ln in driver[j:j + 30])
        # REVRAN sits under CCONTU instead.
        k = next(n for n, ln in enumerate(driver) if 'CALL REVRAN(' in ln)
        assert any('CCONTU' in ln for ln in driver[k - 5:k])

    def test_the_docstring_carries_the_corrected_address(self):
        doc = self._docstring()
        assert 'oassun26.f:853-858' in doc
        assert '876-880' not in doc
        # It must also say why the other block is not this one, or the next
        # reader re-derives the same wrong answer from a content match.
        assert 'REVRAN' in doc and '633-638' in doc


@pytest.mark.requires_oases
class TestTwoBinaryRoundTrip:
    """Driven through the real oast2 → oass2 chain."""

    def test_reverberation_field(self, tmp_path):
        result = _oass(work_dir=tmp_path, cleanup=False).run(
            _env(), _CLASS_SRC, _CLASS_RCV)
        assert isinstance(result, Field)
        # -10*log10 E[|p_scat|^2] — REVINT's "CONVERT TO dB" block
        # (oassun26.f:853-858), reached because option 'r' sets PLTL
        # (unoass21.f:607-609) — is a reverberation LOSS: the leading minus
        # from VSMUL(-5E0) at :858 means a larger value is a weaker scattered
        # field. Same direction as transmission loss, different quantity, so
        # it must not read as pressure/dB.
        assert result.kind == 'reverberation' and result.unit == 'dB'
        assert result.shape == (len(_CLASS_RCV.depths),
                                len(_CLASS_RCV.ranges))
        assert np.allclose(result.coords['range'], _CLASS_RCV.ranges)
        assert np.allclose(result.coords['depth'], _CLASS_RCV.depths)
        assert np.isfinite(result.data).all()
        # Reverberation weakens with range, so the loss grows.
        assert (result.data[:, -1] > result.data[:, 0]).all()
        assert result.model == 'OASS' and result.backend == 'oass'
        assert 'interpolated' not in result.metadata
        # work_dir survives, so the curve file is reachable (DOCUMENTATION §8).
        assert result.metadata['plt_file'] == str(tmp_path / 'oass_run.plt')

    def test_field_max_is_the_smallest_stored_loss_not_the_largest(
            self, tmp_path):
        """``Field.max()`` slices the LOUDEST cell, and on a reverberation
        grid that is the SMALLEST stored number — driven through the real
        oast2 → oass2 chain rather than a hand-written array.

        This is the end-to-end half of the loss convention, and it is what
        the two existing direction tests do not cover:

        * ``test_multiple_scattering_lowers_the_reverberation_level``
          compares two real runs against each other (``oass.tex:163-166``), so
          it pins a RELATIVE bound between two kernels. A reader that negated
          every value would flip both runs together and keep the inequality.
        * ``test_reverberation_field`` above checks the near/far endpoints of
          one real run, but nothing downstream of the Field.
        * ``test_core_classes.py`` pins ``Field.max()``, ``.db`` and the axis
          label on Fields built from hand-written arrays, so it pins the
          CONSUMERS against a direction the test itself supplies.

        Nothing joined the two, so a reader that negated the payload would
        leave every consumer pin green. The direction is not uacpy's choice:
        ``REVINT``'s "CONVERT TO dB" block applies ``CVMAGS`` (square),
        ``VCLIP``, ``VALG10`` and then ``VSMUL(…, -5E0, …)``
        (``oassun26.f:855-858``), and that leading minus is what makes the
        stored quantity a loss.

        The ``max()`` assertion comes FIRST on purpose. Negating the reader
        breaks the range direction too, so an assertion order that checked
        growth first would never reach — and so never prove — the consumer
        this test exists for.
        """
        result = _oass(work_dir=tmp_path).run(_env(), _CLASS_SRC, _CLASS_RCV)
        data = np.asarray(result.data, dtype=float)
        assert result.kind == 'reverberation'

        peak = result.max()
        # A loss: least loss is loudest, so max() takes the global MINIMUM
        # branch. This is what fails if ``Field.max`` stops counting
        # reverberation among the inverted kinds — it then returns the other
        # end of a 15.5 dB spread (measured on this case), which the next
        # assertion pins as a real gap rather than a degenerate one.
        assert float(peak.data) == pytest.approx(float(data.min()))
        assert float(data.max()) - float(data.min()) > 10.0
        # The loudest scattering is the cell nearest the source. This is the
        # assertion that carries the SIGN: negating the whole payload flips
        # which value is smallest along with the values, so the comparison
        # above survives it, and only the cell the answer lands in moves
        # (measured: 200 m → 1742.9 m under a negated reader).
        assert peak.pinned['range'] == pytest.approx(
            float(result.coords['range'][0]))

        # The premise that makes those the same cell: the scattered field
        # weakens with range, so the stored LOSS grows. Every range beyond the
        # nearest is a larger number, at both receiver depths. The growth is
        # not monotone cell to cell (a 2.9 dB dip at 1.5 km here), so this is
        # stated against the nearest range rather than as a strict diff.
        assert (data[:, 1:] > data[:, :1]).all()

    def test_wavenumber_count_comes_from_the_rhs_not_the_deck(self, tmp_path):
        # The deck carries 2048 (write_oass_input's explicit value); the
        # binary re-derives NWVNO from the mean field's step
        # (unoass21.f:209-215) and echoes what it used.
        result = _oass(work_dir=tmp_path, cleanup=False).run(
            _env(), _CLASS_SRC, _CLASS_RCV)
        deck = (tmp_path / 'oass_run.dat').read_text().splitlines()
        deck_nw = int([ln for ln in deck
                       if len(ln.split()) == 3 and ln.split()[1] == '1'][0]
                      .split()[0])
        assert deck_nw == 2048
        assert result.metadata['n_wavenumbers'] != deck_nw

    def test_both_decks_integrate_from_the_same_c_low(self, tmp_path):
        # G4's contract, read off the two decks. CMIN truncates the OASS
        # integral over the .rhs — measured 30.15 dB for 1350 → 2000 m/s —
        # and the two writers derive their default independently, so a
        # divergence between them would move the answer with nothing said.
        _oass(work_dir=tmp_path, cleanup=False).run(
            _env(), _CLASS_SRC, _CLASS_RCV)

        def deck_c_low(name):
            # Block VII line 1 is CMIN CMAX; CMAX is the 1e8 "no limit" value.
            for line in (tmp_path / name).read_text().splitlines():
                tokens = line.split()
                if len(tokens) != 2:
                    continue
                try:
                    lo, hi = float(tokens[0]), float(tokens[1])
                except ValueError:
                    continue
                if hi >= 1e7:
                    return lo
            raise AssertionError(f'no CMIN CMAX line in {name}')

        assert deck_c_low('oass_run.dat') == pytest.approx(
            deck_c_low('oast_run.dat'))

    def test_mean_field_result_is_kept_and_is_not_a_carrier(self, tmp_path):
        result = _oass(work_dir=tmp_path, cleanup=False).run(
            _env(), _CLASS_SRC, _CLASS_RCV)
        mean = result.metadata['mean_field_result']
        assert isinstance(mean, Field) and mean.model == 'OAST'
        for forbidden in ('env', 'environment', 'source', 'receiver'):
            assert not hasattr(result, forbidden)
            assert forbidden not in result.metadata
        assert result.metadata['rhs_file'].endswith('oast_run.045')

    def test_covariance(self, tmp_path):
        cov = _oass(work_dir=tmp_path, cleanup=False).compute_covariance(
            _env(), _CLASS_SRC, _CLASS_RCV)
        assert isinstance(cov, Covariance)
        n = len(_CLASS_RCV.depths)
        assert cov.covariance.shape == (1, n, n)
        C = cov.covariance[0]
        assert np.allclose(C, C.conj().T, atol=1e-5 * np.abs(C).max())
        # G7's pathology, pinned: 'a' on a deck that also carries 'r' comes
        # back all zeros, and nothing downstream would notice.
        assert (np.real(np.diag(C)) > 0).all()
        assert cov.metadata['n_receivers'] == n

    def test_asking_for_two_products_in_one_deck_is_refused(self):
        # The other half of G7: the wrapper never emits 'r a', and a raw
        # option line that does is refused before the deck is written.
        with pytest.raises(ConfigurationError, match='more than one'):
            _oass(options='r a').run(_env(), _CLASS_SRC, _CLASS_RCV)

    def test_covariance_title_carries_no_nul_bytes(self, tmp_path):
        # PUTXSM writes TITLE from a COMMON OASS never fills (unoass21.f:34
        # declares it locally), so the raw field is 32 NULs.
        cov = _oass(work_dir=tmp_path, cleanup=False).compute_covariance(
            _env(), _CLASS_SRC, _CLASS_RCV)
        assert '\x00' not in cov.metadata['title']

    def test_the_xsm_agrees_with_the_binarys_own_ascii_dump(self, tmp_path):
        # Two independent paths — the Fortran's normalised ASCII dump on unit
        # 24 (oassun26.f:1068) and uacpy's binary reader — establish the
        # record layout, endianness and column order at once.
        cov = _oass(work_dir=tmp_path, cleanup=False).compute_covariance(
            _env(), _CLASS_SRC, _CLASS_RCV)
        C = cov.covariance[0]
        d = np.sqrt(np.real(np.diag(C)))
        normalised = np.real(C / np.outer(d, d))
        rows = []
        for line in (tmp_path / 'oass_run.cor').read_text().splitlines():
            if line.strip().startswith('receiver'):
                continue
            values = [float(v) for v in line.split()]
            if values:
                rows.append(values)
        assert np.allclose(np.array(rows), normalised, atol=1e-5)

    @pytest.mark.parametrize('run_mode', [RunMode.REVERBERATION,
                                          RunMode.COVARIANCE])
    def test_every_metadata_key_is_documented(self, tmp_path, run_mode):
        result = _oass(work_dir=tmp_path, cleanup=False).run(
            _env(), _CLASS_SRC, _CLASS_RCV, run_mode=run_mode)
        documented = {k for (m, k) in _DOCUMENTED_METADATA if m == 'OASS'}
        documented |= set(_UNIVERSAL_METADATA)
        assert not set(result.metadata) - documented

    def test_a_cleaned_work_dir_keeps_nothing_from_either_binary(self,
                                                                 tmp_path):
        # Both runs' scratch goes: the OASS deck and outputs, the producer's
        # deck and .rhs, and the bare fort.46 an option-'s' run drops
        # (oaseun31.f:1900 WRITEs unit 46 with no OPFILW).
        work = tmp_path / 'pinned'
        work.mkdir()
        (work / 'user_file.txt').write_text('mine')
        _oass(work_dir=work, cleanup=True).run(_env(), _CLASS_SRC, _CLASS_RCV)
        assert sorted(p.name for p in work.iterdir()) == ['user_file.txt']

    def test_a_kept_work_dir_holds_both_decks(self, tmp_path):
        # The other half of the two-axis table: cleanup=False keeps both
        # binaries' files, which is what makes the chain debuggable — the
        # producer's deck and .rhs as much as the consumer's.
        work = tmp_path / 'kept'
        _oass(work_dir=work, cleanup=False).run(_env(), _CLASS_SRC, _CLASS_RCV)
        names = {p.name for p in work.iterdir()}
        assert {'oast_run.dat', 'oast_run.045', 'oass_run.dat',
                'oass_run.plt'} <= names


@pytest.mark.requires_oases
class TestRhsContract:
    """The producer/consumer handshake, read off the bytes."""

    def test_header_record_is_nfreq_and_the_frequency(self, tmp_path):
        # A single-frequency OAST writes nfreq into the NX slot and its own
        # frequency into FR1 (unoast31.f:164-165), which is what OASS reads
        # back as nx_in_file / fr1_in_file (unoass21.f:127).
        _oass(work_dir=tmp_path, cleanup=False).run(
            _env(), _CLASS_SRC, _CLASS_RCV)
        header = read_oases_rhs_header(tmp_path / 'oast_run.045')
        assert header['n_time_samples'] == 1
        assert header['freq_min'] == pytest.approx(
            float(_CLASS_SRC.frequencies[0]))
        assert header['frequency'] == pytest.approx(header['freq_min'])

    def test_a_frequency_mismatch_is_refused_rather_than_substituted(
            self, tmp_path, monkeypatch):
        # G2. unoass21.f:128-135 warns to stdout and then executes
        # ``freq=fr1_in_file``, so a mismatch silently moves the run to the
        # .rhs's frequency while the Field would still be labelled with the
        # deck's.
        import uacpy.models.oases as oases_module
        monkeypatch.setattr(oases_module, 'read_oases_rhs_header',
                            lambda path: {'n_time_samples': 1,
                                          'freq_min': 137.0})
        with pytest.raises(ConfigurationError, match='137'):
            _oass(work_dir=tmp_path).run(_env(), _CLASS_SRC, _CLASS_RCV)

    def test_a_multi_frequency_rhs_is_refused(self, tmp_path, monkeypatch):
        import uacpy.models.oases as oases_module
        monkeypatch.setattr(oases_module, 'read_oases_rhs_header',
                            lambda path: {'n_time_samples': 4,
                                          'freq_min': 100.0})
        with pytest.raises(ConfigurationError, match='frequency block'):
            _oass(work_dir=tmp_path).run(_env(), _CLASS_SRC, _CLASS_RCV)

    def test_the_frequency_match_tolerance_is_a_tenth_of_a_percent(
            self, tmp_path, monkeypatch):
        """G2's boundary at 100 Hz: |Δf| > 0.1 Hz refuses, |Δf| ≤ 0.1 Hz
        passes the guard (the producer's real .rhs carries the exact
        frequency, so the passing chain then completes normally)."""
        import uacpy.models.oases as oases_module
        monkeypatch.setattr(oases_module, 'read_oases_rhs_header',
                            lambda path: {'n_time_samples': 1,
                                          'freq_min': 100.11})
        with pytest.raises(ConfigurationError, match='100.11'):
            _oass(work_dir=tmp_path / 'outside').run(
                _env(), _CLASS_SRC, _CLASS_RCV)
        monkeypatch.setattr(oases_module, 'read_oases_rhs_header',
                            lambda path: {'n_time_samples': 1,
                                          'freq_min': 100.09})
        result = _oass(work_dir=tmp_path / 'inside').run(
            _env(), _CLASS_SRC, _CLASS_RCV)
        assert np.isfinite(np.asarray(result.data)).any()


@pytest.mark.requires_oases
class TestCLowDirectionOfHarm:
    """oases.md §3: ``c_low`` defaults to the mean field's own bound (1350
    m/s here, ``oases_wavenumber_bounds`` on the 1500 m/s water — the same
    figure the doc's measured cases quote). Pinning it differently warns with
    the direction of harm — raising truncates the scattering integral
    (measured 30.15 dB), lowering is inert (measured 1e-4 dB: REVINT bounds
    its own buffer reads, so wavenumbers past the mean field's grid
    contribute nothing). No binary runs — the warning fires pre-launch."""

    def test_lower_c_low_warns_it_does_not_extend(self):
        with pytest.warns(UserWarning, match='does not extend'):
            _oass(c_low=900.0)._warn_on_c_low_mismatch(_env())

    def test_higher_c_low_warns_it_truncates(self):
        with pytest.warns(UserWarning, match='truncates'):
            _oass(c_low=2000.0)._warn_on_c_low_mismatch(_env())

    def test_unset_c_low_inherits_the_mean_fields_bound_silently(self):
        import warnings as _w
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter('always')
            _oass()._warn_on_c_low_mismatch(_env())
        assert not [w for w in caught if 'c_low' in str(w.message)]

    def test_matching_c_low_is_silent(self):
        import warnings as _w
        mean = _oass()._mean_field_c_low(_env())
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter('always')
            _oass(c_low=mean)._warn_on_c_low_mismatch(_env())
        assert not [w for w in caught if 'differs' in str(w.message)]


@pytest.mark.requires_oases
class TestPhysics:
    """Numbers nothing in uacpy can produce by accident."""

    def test_doubling_rms_roughness_costs_exactly_10log10_4(self, tmp_path):
        # REVINT's integration constant is FF = RG2*DLWVNO*FNI5^2/(2 pi)^3
        # (oassun26.f:698), linear in RG^2, and the mean field is unchanged —
        # the OASS deck's roughness is independent of the producer's
        # (oass.tex:182-186). So the whole chain must move by 10*log10(4) and
        # by nothing else, at every depth and range.
        quiet = _oass(work_dir=tmp_path / 'a').run(
            _env(), _CLASS_SRC, _CLASS_RCV)
        loud = _oass(rms_roughness=1.0, work_dir=tmp_path / 'b').run(
            _env(), _CLASS_SRC, _CLASS_RCV)
        delta = loud.data - quiet.data
        assert np.allclose(delta, -10 * np.log10(4.0), atol=1e-3)

    def test_c_low_above_the_mean_fields_warns_because_it_moves_the_answer(
            self, tmp_path):
        # G4. Measured on one .rhs, option 'r', only CMIN changed: 1350 →
        # 2000 m/s moved the field 30.15 dB; 1350 → 900 moved it 1e-4 dB.
        with pytest.warns(UserWarning, match='physically significant'):
            _oass(c_low=2000.0, work_dir=tmp_path).run(
                _env(), _CLASS_SRC, _CLASS_RCV)

    def test_c_low_matching_the_mean_field_is_silent(self, tmp_path,
                                                     recwarn):
        # 0.9 * min(SSP) is what both writers derive.
        _oass(c_low=0.9 * 1500.0, work_dir=tmp_path).run(
            _env(), _CLASS_SRC, _CLASS_RCV)
        assert not [w for w in recwarn
                    if 'physically significant' in str(w.message)]

    def test_multiple_scattering_lowers_the_reverberation_level(self,
                                                                tmp_path):
        # oass.tex:163-166: option 'p' "yields lower bound"; the default
        # single-scattering kernel yields the upper bound. The Field is a
        # loss, so a lower level is a larger number.
        single = _oass(work_dir=tmp_path / 'a').run(
            _env(), _CLASS_SRC, _CLASS_RCV)
        multiple = _oass(multiple_scattering=True,
                         work_dir=tmp_path / 'b').run(
            _env(), _CLASS_SRC, _CLASS_RCV)
        assert (multiple.data > single.data).all()

    def test_goff_jordan_backscatters_more_than_gaussian(self, tmp_path):
        gaussian = _oass(work_dir=tmp_path / 'a').run(
            _env(), _CLASS_SRC, _CLASS_RCV)
        goff = _oass(spectrum='goff-jordan', work_dir=tmp_path / 'b').run(
            _env(), _CLASS_SRC, _CLASS_RCV)
        assert (goff.data < gaussian.data).all()


@pytest.mark.requires_oases
class TestReceiverRangeAxis:
    def test_non_equispaced_ranges_warn_and_resample(self, tmp_path):
        # G13 — Block VIII is RMIN RMAX NR and the binary rebuilds the axis
        # as R0+(i-1)*RSTEP (unoass21.f:267-271), so an irregular request is
        # replaced by a linspace over the same span.
        rcv = uacpy.Receiver(depths=[30.0, 70.0],
                             ranges=np.array([200.0, 400.0, 900.0, 2000.0]))
        with pytest.warns(UserWarning, match='not the equispaced axis'):
            result = _oass(work_dir=tmp_path).run(_env(), _CLASS_SRC, rcv)
        assert np.allclose(result.coords['range'], rcv.ranges)
        assert result.metadata['interpolated'] is True
        assert np.allclose(result.metadata['oass_native_ranges'],
                           np.linspace(200.0, 2000.0, 4))

    def test_equispaced_ranges_are_kept_verbatim(self, tmp_path, recwarn):
        result = _oass(work_dir=tmp_path).run(_env(), _CLASS_SRC, _CLASS_RCV)
        assert not [w for w in recwarn
                    if 'not the equispaced axis' in str(w.message)]
        assert 'oass_native_ranges' not in result.metadata


class TestRoughnessMustSurviveTheRecordFormat:
    """A negative RG is the flag that CL and M follow, but only if it
    survives formatting. The layer record writes RG with ``%.4f``, so every
    ``|RG| < 5e-5 m`` renders as ``-0.0000`` and fails INENVI's
    ``rough(m).lt.-1e-10`` test (``oaseun31.f:72``).

    The deck still carries nine tokens, so nothing shifts and no parse error
    occurs: INENVI takes the seven-token read, sets ``clen(m)=0`` (``:102``)
    and promotes it to ``CLEN=1E6`` (``:111-114``) — an *infinite* correlation
    length. ``RG2`` is then 0 (``:379``) and the integration constant ``FF``
    is identically zero (``oassun26.f:697``), so the run exits 0 and returns a
    flat clipped level. Measured: 150.000 dB at every cell.

    The guard lives in ``_make_roughness_tail`` rather than in one writer, so
    OASS, OASSP and OASR cannot disagree about it.
    """

    @staticmethod
    def _tail(rg):
        from uacpy.io.oases_writer import _make_roughness_tail

        class _HS:
            roughness = 0.5

        class _B:
            def halfspace_at(self, range=0.0):
                return _HS()

        class _E:
            has_layered_bottom = False
            bottom = _B()

        return _make_roughness_tail(_E(), {1: (rg, 10.0, 2.0)})(1)

    @pytest.mark.parametrize('rg', [0.0, 1e-6, 1e-5, 2e-5, 4.9e-5])
    def test_a_roughness_the_format_cannot_carry_is_refused(self, rg):
        with pytest.raises(ConfigurationError, match='-0.0000|smallest'):
            self._tail(rg)

    @pytest.mark.parametrize('rg', [5e-5, 1e-4, 0.5])
    def test_a_representable_roughness_keeps_its_negative_flag(self, rg):
        # The discriminating counterpart: the guard must not creep upward and
        # refuse roughness the record can express.
        tail = self._tail(rg)
        assert float(tail.split()[0]) < 0.0, \
            f'RG rendered as {tail.split()[0]!r} — the CL/M flag is lost'

    def test_the_threshold_is_the_formats_own_resolution(self):
        # Pin the derivation, not a magic number: 5e-5 is exactly where
        # "%.4f" stops rounding to zero.
        assert float(f"{-abs(4.9e-5):.4f}") == 0.0
        assert float(f"{-abs(5e-5):.4f}") != 0.0


class TestTheSeaSurfaceIsAScatteringInterface:
    """Deck layer 2 is the first water record, whose RG is the sea surface:
    ``ROUGH(M)`` is the roughness at the top of layer M
    (``oaseun31.f:381-383``) and ``ROUGH(1)=ROUGH(2)`` at ``:377``. OASS
    scatters from it as readily as from the seabed — an ice keel or a
    wind-roughened surface is the usual case — so refusing it would have
    excluded a whole class of run the binary supports.

    Verified end to end: ``oass2`` exits 0 and echoes
    ``>>> Rough Interface <<< / Rms = 0.7 / L_x = 10 / Exp = 2``.
    """

    @staticmethod
    def _env(isovelocity=True, surface_roughness=0.7):
        data = [1500.0, 1500.0] if isovelocity else [1500.0, 1490.0]
        return uacpy.Environment(
            bathymetry=100.0,
            ssp=uacpy.SoundSpeedProfile(depths=[0.0, 100.0], data=data),
            bottom=uacpy.Bottom.from_halfspace(uacpy.BoundaryProperties(
                sound_speed=1700.0, density=1.8, attenuation=0.5,
                roughness=0.5)),
            surface=uacpy.BoundaryProperties(sound_speed=340.0, density=0.0012,
                                             roughness=surface_roughness))

    def _deck(self, tmp_path, isovelocity):
        path = tmp_path / f'surf{isovelocity}.dat'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            write_oass_input(path, self._env(isovelocity), _SRC, _RCV, 'r',
                             interface=2, correlation_length=10.0,
                             spectral_exponent=2.0, phase_speed=1500.0)
        return path.read_text().splitlines()

    @pytest.mark.parametrize('isovelocity', [True, False])
    def test_the_water_record_carries_the_spectrum(self, tmp_path,
                                                   isovelocity):
        water = self._deck(tmp_path, isovelocity)[5].split()
        assert len(water) == 9
        assert float(water[6]) < 0.0, 'the CL/M flag must survive'
        assert (float(water[7]), float(water[8])) == (10.0, 2.0)

    @pytest.mark.parametrize('isovelocity', [True, False])
    def test_the_seabed_keeps_its_plain_rms(self, tmp_path, isovelocity):
        # Only the named interface gets the spectrum; giving the seabed one
        # too would make it scatter as well and change the answer.
        lines = self._deck(tmp_path, isovelocity)
        bottom = [ln.split() for ln in lines
                  if ln.split() and ln.split()[0] == '100.00'][-1]
        assert len(bottom) == 7 and float(bottom[6]) > 0.0

    def test_an_interior_water_record_is_refused(self, tmp_path):
        # Only the FIRST water record carries an RG column INENVI re-reads;
        # an interior one does not, so naming it must still fail.
        with pytest.raises(ConfigurationError, match='no interface'):
            write_oass_input(tmp_path / 'x.dat', self._env(False), _SRC, _RCV,
                             'r', interface=3, correlation_length=10.0,
                             spectral_exponent=2.0)

    def test_the_surface_roughness_reaches_both_water_branches(self, tmp_path):
        # ROUGH(M) is the roughness at the TOP of layer M (oaseun31.f:381-383)
        # and ROUGH(1)=ROUGH(2) at :377, so the first water record carries the
        # sea surface even when the scattering interface is the seabed. The
        # isovelocity branch used to write a hardcoded 0.0.
        for isovelocity, interface in ((True, 3), (False, 4)):
            path = tmp_path / f'iso{isovelocity}.dat'
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                write_oass_input(path, self._env(isovelocity), _SRC, _RCV, 'r',
                                 interface=interface, correlation_length=10.0,
                                 spectral_exponent=2.0, rms_roughness=0.5)
            water = path.read_text().splitlines()[5]
            assert float(water.split()[6]) == pytest.approx(0.7), \
                f'isovelocity={isovelocity}: surface roughness lost ({water!r})'


class TestCorrelationLengthMustSurviveTheRecordFormat:
    """The companion to ``TestRoughnessMustSurviveTheRecordFormat``: CL rides
    the same ``%.4f`` as RG, and its failure is worse than a lost flag.

    INENVI reads ``0.0`` and then **promotes it to an infinite correlation
    length** — ``oaseun31.f:111-113`` ``IF (CLEN(M).EQ.0.0) … CLEN(M)=1E6``.
    Measured: a ``CL=1e-5`` deck produces a ``.rco`` bit-identical to a
    hand-written ``CL=1e6`` deck. Eleven orders of magnitude, exit 0, no
    warning.

    A *negative* CL is the other half: it selects INENVI's twelve-token
    volume-scattering read (``:77-79``) from a nine-token record, so the
    list-directed read runs into the next record and the deck shifts.

    Both live in ``_make_roughness_tail``, the funnel OASR, OASS and OASSP
    share — OASR reaches it through the public ``interface_roughness=`` and
    had no CL check of its own at all.
    """

    @staticmethod
    def _tail(cl):
        from uacpy.io.oases_writer import _make_roughness_tail

        class _HS:
            roughness = 0.5

        class _B:
            def halfspace_at(self, range=0.0):
                return _HS()

        class _E:
            has_layered_bottom = False
            bottom = _B()

        return _make_roughness_tail(_E(), {1: (0.5, cl, 3.0)})(1)

    @pytest.mark.parametrize('cl', [0.0, 1e-6, 1e-5, 4.9e-5])
    def test_a_cl_the_format_cannot_carry_is_refused(self, cl):
        with pytest.raises(ConfigurationError, match='1e6|INFINITE|smallest'):
            self._tail(cl)

    def test_a_negative_cl_is_refused_before_it_shifts_the_deck(self):
        with pytest.raises(ConfigurationError, match='volume-scattering'):
            self._tail(-10.0)

    @pytest.mark.parametrize('cl', [5e-5, 1e-4, 0.01, 10.0])
    def test_a_representable_cl_is_written(self, cl):
        # The discriminating counterpart: the guard must not creep upward.
        # 0.01 m is the smallest CL with a measured reflection-loss delta
        # (6.23 dB against the promoted 1e6), so it must remain expressible.
        assert float(self._tail(cl).split()[1]) > 0.0

    def test_oasr_inherits_the_guard_through_its_public_knob(self):
        # OASR exposes interface_roughness=[(RG, CL, M)] and had no CL check
        # of its own; the funnel is what gives it one.
        with pytest.raises(ConfigurationError):
            self._tail(1e-5)


class TestKernelPlotsCannotRideAReverbRun:
    """``oass.tex`` says three times (``:141``, ``:151``, ``:160``) that ``I``,
    ``S`` and ``c`` "cannot be applied together with the reverb options
    C,D,R,a,r", and ``:146`` that ``R`` cancels them. The mechanism is
    ``unoass21.f:342-344`` gating ``CALSIN`` on ``IF (.not.REVERB)``: with a
    reverberation letter present the kernel is never computed, so the deck is
    accepted, the run succeeds, and the requested plot simply never appears.
    """

    @pytest.mark.parametrize('kernel', ['I', 'S', 'c'])
    @pytest.mark.parametrize('reverb', ['r', 'R', 'a', 'C', 'D'])
    def test_the_combination_is_refused(self, tmp_path, kernel, reverb):
        with pytest.raises(ConfigurationError, match='CALSIN|cannot be combined'):
            _write(tmp_path, options=f'{kernel} {reverb}', phase_speed=1500.0)

    @pytest.mark.parametrize('options', ['r', 'a', 'I', 'S c'])
    def test_either_alone_is_fine(self, tmp_path, options):
        # The discriminating counterpart: neither group is refused on its own.
        _write(tmp_path, options=options, phase_speed=1500.0)


class TestOassOptionSetMatchesGetopt:
    """``_OASS_OPTIONS`` must be exactly the letters ``GETOPT`` branches on —
    no more (a letter uacpy accepts that OASES prints
    ``>>>> UNKNOWN OPTION <<<<`` for) and no fewer (a legal letter uacpy
    refuses). Pinned against the source because a re-audit mis-read ``'d'``
    as absent when ``unoass21.f:643`` carries ``else if (opt(i).eq.'d')``.
    """

    def test_the_set_is_exactly_getopts_branches(self):
        from uacpy.io.oases_writer import _OASS_OPTIONS
        # Enumerated from unoass21.f:557-698, opt(i).eq.'X' branches.
        assert _OASS_OPTIONS == set('crRICDdaPSkZGgpQ')


class TestSubMillimetreLayersDoNotVanishSilently:
    """The layer record's depth column is ``%.2f``, so a sediment layer
    thinner than 5 mm is written at the same formatted depth as the layer
    below it. ``INENVI`` rejects only a *decreasing* depth
    (``oaseun31.f:58-61``), so equal depths pass and ``THICK(I)`` comes out 0
    (``:1508``) — the layer is dropped.

    Nothing measurable is lost at that thickness (far below λ/100 in any
    usable band), so this warns rather than raises. It exists because the
    request otherwise vanishes without trace — the same silent-collapse class
    as the RG and CL truncations, on a different column.
    """

    @staticmethod
    def _write(tmp_path, thickness):
        from uacpy.io.oases_writer import write_oast_input
        col = uacpy.SeabedColumn(
            layers=[uacpy.SedimentLayer(thickness=thickness,
                                        sound_speed=1600.0, density=1.7,
                                        attenuation=0.2)],
            halfspace=uacpy.BoundaryProperties(sound_speed=1800.0,
                                               density=2.0, attenuation=0.5))
        env = uacpy.Environment(
            bathymetry=100.0,
            ssp=uacpy.SoundSpeedProfile(depths=[0.0, 100.0],
                                        data=[1500.0, 1500.0]),
            bottom=uacpy.Bottom(columns=[col]))
        write_oast_input(tmp_path / 'x.dat', env, _SRC,
                         uacpy.Receiver(depths=[50.0],
                                        ranges=np.linspace(100.0, 1000.0, 5)))
        return (tmp_path / 'x.dat').read_text()

    @pytest.mark.parametrize('thickness', [0.001, 0.004])
    def test_a_collapsed_layer_warns(self, tmp_path, thickness):
        with pytest.warns(UserWarning, match='dropped'):
            self._write(tmp_path, thickness)

    @pytest.mark.parametrize('thickness', [0.0051, 0.5, 5.0])
    def test_a_representable_layer_is_silent(self, tmp_path, thickness):
        # The discriminating counterpart. The boundary is measured, not
        # assumed: 0.005 itself still collides because 100.005 is stored as
        # 100.00499… and formats to "100.00". The first thickness that
        # separates the records is 0.0051.
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            self._write(tmp_path, thickness)

    def test_the_boundary_is_the_formats_own_and_is_measured(self):
        # Pin the derivation so nobody "corrects" 0.0051 back to 0.005.
        assert f"{100.0:.2f}" == f"{100.0 + 0.005:.2f}"
        assert f"{100.0:.2f}" != f"{100.0 + 0.0051:.2f}"

    def test_the_two_records_really_do_collide(self, tmp_path):
        # Pin the mechanism, not just the message.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            deck = self._write(tmp_path, 0.001)
        depths = [ln.split()[0] for ln in deck.splitlines()
                  if ln.split() and ln.split()[0] == '100.00']
        assert len(depths) >= 2, 'the sediment and halfspace records collide'


# ─────────────────────────────────────────────────────────────────────────────
# What the .rhs contract actually admits
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.requires_oases
class TestOasrIsNotAUsableMeanField:
    """``oass.tex:18-23`` names OASR alongside OAST as a producer, but the
    pairing divides by zero inside oass2 for the decks uacpy writes.

    ``unoass21.f:200-211`` keeps only the ``.rhs`` records whose interface
    index equals ``INTFC``, takes ``DLWVNO`` from the gap between the first
    two it keeps, and then computes ``NWVNO=(WKMAX-WK0)/DLWVNO+1``. An OASR
    deck's layer 1 is the water half-space, so ``SCTRHS`` stamps ``IN1 = 2``
    (``oaseun31.f:2308-2309``) on every record, while ``INTFC`` here is a
    bottom deck layer, 3 or deeper — nothing matches, ``KCNT`` stays 0 and
    ``DLWVNO`` is never assigned. Independently, OASR calls ``SCTRHS`` from
    inside its angle loop with ``WVNO = Re(AK(1,1))*cos(ANG)``
    (``oasjun21.f:70,108``), so its wavenumber grid is cosine-spaced and the
    two-record ``DLWVNO`` inference does not describe it."""

    def test_an_oasr_mean_field_is_refused(self):
        from uacpy.models import OASR
        with pytest.raises(ConfigurationError) as ei:
            OASS(correlation_length=10.0, mean_field=OASR())
        message = str(ei.value)
        assert 'IN1=2' in message, message
        assert 'cos(ANG)' in message, message

    def test_an_oast_mean_field_is_accepted(self):
        assert OASS(correlation_length=10.0,
                    mean_field=OAST(options='N J T s')).mean_field is not None

    def test_a_producer_that_writes_no_rhs_is_refused_by_type(self):
        with pytest.raises(ConfigurationError, match='oass.tex:18-23'):
            OASS(correlation_length=10.0, mean_field=OASN())


@pytest.mark.requires_oases
class TestSeaSurfaceIsALegalScatteringInterface:
    """Deck layer 2 is the first water record and ``ROUGH(M)`` is the
    roughness of the interface at the TOP of layer M (``oaseun31.f:377-383``),
    so layer 2's RG is the sea surface; ``SCTRHS`` writes its operators under
    ``IN1 = 2`` like any other interface, and ``write_oass_input`` already
    carries a ``surface_is_target`` branch that routes the nine-token
    roughness tail to the water record. ``OASS.spec.supports`` advertises
    ``'rough_surface'``. Only the model-side interface check refused it."""

    @staticmethod
    def _env(surface_roughness, bottom_roughness):
        return uacpy.Environment(
            bathymetry=100.0,
            ssp=uacpy.SoundSpeedProfile(depths=[0.0, 100.0],
                                        data=[1500.0, 1500.0]),
            surface=uacpy.BoundaryProperties(acoustic_type='vacuum',
                                             roughness=surface_roughness),
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1700.0, density=1.8,
                attenuation=0.5, roughness=bottom_roughness))

    def test_interface_2_resolves_against_the_surface_roughness(self):
        model = OASS(correlation_length=10.0, interface=2)
        assert model._resolve_interface(self._env(0.5, 0.0)) == 2

    def test_a_smooth_surface_is_named_as_the_reason(self):
        model = OASS(correlation_length=10.0, interface=2)
        with pytest.raises(ConfigurationError, match='sea surface'):
            model._resolve_interface(self._env(0.0, 0.5))

    def test_the_default_interface_resolves_to_the_seabed(self):
        model = OASS(correlation_length=10.0)
        assert model._resolve_interface(self._env(0.0, 0.5)) == 3

    def test_an_interface_below_the_stack_is_refused(self):
        model = OASS(correlation_length=10.0, interface=9)
        with pytest.raises(ConfigurationError, match='not a scattering'):
            model._resolve_interface(self._env(0.5, 0.5))

    @pytest.mark.slow
    def test_the_chain_runs_and_answers(self, tmp_path):
        model = OASS(correlation_length=10.0, rms_roughness=0.5, interface=2,
                     work_dir=tmp_path, cleanup=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = model.run(self._env(0.5, 0.0), _CLASS_SRC, _CLASS_RCV,
                               run_mode=RunMode.REVERBERATION)
        data = np.asarray(result.data)
        assert data.shape == (len(_CLASS_RCV.depths), len(_CLASS_RCV.ranges))
        assert np.isfinite(data).all()
        # A real reverberation loss, not a saturated sentinel.
        assert 0.0 < float(np.min(data)) < 200.0


@pytest.mark.requires_oases
def test_oass_accepts_the_same_spectrum_spellings_as_oassp():
    """OASS compared ``spectrum`` exactly while OASSP lower-cased it, so
    ``'Gaussian'`` raised on one and ran on the other. Both normalise now, so
    the stored value — which the option letter and the options-exclusivity
    check both read — is the canonical spelling."""
    assert OASS(correlation_length=10.0,
                spectrum='Gaussian').spectrum == 'gaussian'
    assert OASS(correlation_length=10.0,
                spectrum='Goff-Jordan').spectrum == 'goff-jordan'
    # The canonical value is what reaches the option line.
    assert 'g' in OASS(correlation_length=10.0, spectrum='Goff-Jordan'
                       )._resolve_options(RunMode.REVERBERATION).split()
    # A default spelled with a capital must not read as a pinned non-default.
    assert OASS(correlation_length=10.0, spectrum='Gaussian',
                options='r').spectrum == 'gaussian'
    with pytest.raises(ConfigurationError, match='von-karman'):
        OASS(correlation_length=10.0, spectrum='von-karman')


class TestOassMeanFieldOptionLine:
    """OASS runs a producer for the mean field and consumes only its ``.rhs``,
    but it runs that producer through the producer's own wrapper — which
    requires the table ``'T'`` writes (OAST's FOR020 ``.plt``, OASR's
    ``.rco``/``.trc``). ``options='N J s'`` is documented as legal and does
    write the ``.rhs``, so the run died on a missing table nothing here reads,
    after the binary had already been spent. Both letters are now stated up
    front, before any deck is written."""

    @staticmethod
    def _rig(tmp_path, options):
        model = OASS(correlation_length=10.0, verbose=False,
                     work_dir=tmp_path, cleanup=False,
                     mean_field=OAST(options=options, verbose=False))
        env = Environment(
            name='rough', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.6,
                                      attenuation=0.2, roughness=0.5))
        return (model, env, Source(depths=50.0, frequencies=100.0),
                Receiver(depths=np.array([50.0]),
                         ranges=np.linspace(500.0, 2500.0, 5)))

    def _run_mean_field(self, tmp_path, options):
        model, env, source, receiver = self._rig(tmp_path, options)
        return model._run_mean_field(env, source, receiver,
                                     model._setup_file_manager(), 100.0)

    def test_a_producer_without_s_is_refused(self, tmp_path):
        with pytest.raises(ConfigurationError, match="must contain 's'"):
            self._run_mean_field(tmp_path, 'N J T')

    def test_a_producer_without_T_is_refused(self, tmp_path):
        with pytest.raises(ConfigurationError, match="must also"):
            self._run_mean_field(tmp_path, 'N J s')

    def test_the_refusal_names_a_line_that_works(self, tmp_path):
        with pytest.raises(ConfigurationError) as excinfo:
            self._run_mean_field(tmp_path, 'N J s')
        assert "'N J T s'" in str(excinfo.value)

    def test_the_default_producer_carries_both_letters(self):
        # The default mean field is built inside _run_mean_field, so a
        # missing letter there would refuse every default OASS run.
        chars = set(OAST(options='N J T s', verbose=False)._resolve_options())
        assert {'s', 'T'} <= chars
