"""Pins for OASES manual claims not covered by the other OASES test files.

Each class names the ``third_party/oases/doc`` section it pins (with the
vendored Fortran as ground truth where the two disagree):

- ``oases_install.tex:190-198`` vs ``src/compar.f`` — the array bounds the
  wrapper enforces come from the vendored source; the manual's table is stale.
- ``oases_gen.tex:24-115, :226-317`` — negative values in the layer record's
  CC/CS/AS columns are record-arity *flags* (transverse isotropy, dispersion,
  Biot, continuous-SVP, flow), so the carriers must refuse them outright.
- ``oast.tex:73-75`` / ``oasp.tex:652-664`` — a pinned NW must be written with
  ``IC2 = NW`` or the spectrum's tail is zeroed / Hanning-windowed away.
- ``oasr.tex`` Tables I-II — the plot-axis blocks exist only for
  ``NFOU > 0`` / ``NAOU > 0`` (``unoasr21.f:159-172``).
- ``oasr.tex:262-350`` — the ``.rco``/``.trc`` table format: header code 1 =
  slowness / 2 = angle, and the ``#``-annotated per-frequency header.
- ``oasn.tex:547-618`` — the ``.xsm`` header records and the column-major
  covariance record address ``IREC = 10 + IRCV + (JRCV-1)*NRCV +
  (IFREQ-1)*NRCV**2``.
- ``oasn.tex:621-689`` — the ``.rpo`` header, its mixed units (z in m, x/y in
  km), the receiver-innermost replica loop, and the linear on-disk gain.
- ``oases_graph.tex:155-170`` — a ``.plt`` block tabulates x before y when
  both DX and DY are zero, and a fully parameterised curve writes no block.
- ``oast.tex:281-304`` — option 'J' with COFF = 0 invokes the *binary's own*
  default contour offset (``unoast31.f:497-509``), not a zero offset.
- ``oasn.tex`` Block X — the replica grid cap ``NSMAX = 201``
  (``src/compar.f:51``) is a hard STOP with no output and exit code 0
  (``unoasn22.f:186-188``); the manual states no limit.
"""

import re
import struct
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

import uacpy
from uacpy.core.exceptions import ConfigurationError, FileFormatError
from uacpy.io.oases_writer import (
    write_oasn_input,
    write_oasp_input,
    write_oasr_input,
    write_oast_input,
)
from uacpy.io.oases_reader import (
    read_oasn_covariance,
    read_oasn_replicas,
    read_oasr_reflection_coefficients,
    read_oast_tl,
)
from uacpy.tests.conftest import make_pekeris

_SRC = uacpy.Source(depths=50.0, frequencies=100.0)
_RCV = uacpy.Receiver(depths=np.array([30.0, 60.0]),
                      ranges=np.linspace(100.0, 2000.0, 8))

_OASES_ROOT = Path(uacpy.__file__).parent / 'third_party' / 'oases'


def _deck_lines(path):
    return [ln for ln in Path(path).read_text().splitlines() if ln.strip()]


class TestVendoredArrayBoundsAreTheContract:
    """The limits the wrapper enforces are ``src/compar.f``'s, not the
    manual's: ``oases_install.tex:190-198`` still documents NLA = 200 and
    NRD = 101 from an older build, while the vendored source compiles
    NLA = 1001 and NRD = 501. Parsing both files pins which one the wrapper
    follows — and flags the divergence if either vendored file ever changes."""

    @staticmethod
    def _param(name):
        text = (_OASES_ROOT / 'src' / 'compar.f').read_text()
        return int(re.search(rf'\b{name}\s*=\s*(\d+)', text).group(1))

    def test_writer_layer_budget_is_compars_nla(self):
        from uacpy.io.oases_writer import _OASES_MAX_LAYERS
        assert _OASES_MAX_LAYERS == self._param('NLA') == 1001

    def test_writer_receiver_budget_is_compars_nrd(self):
        from uacpy.io.oases_writer import _OASES_MAX_RECEIVER_DEPTHS
        assert _OASES_MAX_RECEIVER_DEPTHS == self._param('NRD') == 501
        # OASN checks the same bound against its twin NRMAX (oasnun22.f:32-35).
        assert self._param('NRMAX') == self._param('NRD')

    def test_wavenumber_bound_is_two_to_the_npexp(self):
        from uacpy.io.oases_writer import _OASES_MAX_WAVENUMBERS
        from uacpy.models.oases import _OASES_NP
        assert _OASES_MAX_WAVENUMBERS == _OASES_NP == 2 ** self._param('NPEXP')

    def test_the_manuals_install_table_is_stale(self):
        # oases_install.tex:196-198 documents the OLD defaults. If this ever
        # fails, the vendored manual changed and the divergence notes in the
        # writer docstrings need a re-read.
        text = (_OASES_ROOT / 'doc' / 'oases_install.tex').read_text()
        assert re.search(r'NLA\s*&[^&]*&\s*200\b', text)
        assert re.search(r'NRD\s*&[^&]*&\s*101\b', text)
        assert self._param('NLA') != 200 and self._param('NRD') != 101


class TestNegativeFlagValuesCannotReachTheDeck:
    """In the OASES layer record a negative value is not a magnitude but a
    format switch: CC < 0 selects transversely isotropic / dispersive media
    that read EXTRA records after the layer line (oases_gen.tex:34-40, :77-87),
    CC < 0 with CS < 0 selects a Biot layer with a 13-parameter follow-on line
    (:124-137), CS = -999.999 is the continuous-SVP flag (:238-246), and
    AS = -888.888 makes INENVI read a flow-speed line (:283-291). Any of them
    written by mistake would shift every later READ in the deck, so the
    carriers must refuse the negative values at construction."""

    @pytest.mark.parametrize('kwargs', [
        {'sound_speed': -1800.0},               # TI / dispersive / Biot flag
        {'shear_speed': -999.999},              # continuous-SVP flag
        {'shear_attenuation': -888.888},        # stratified-flow flag
    ])
    def test_sediment_layer_refuses_the_flag_values(self, kwargs):
        base = dict(thickness=10.0, sound_speed=1600.0, density=1.8,
                    attenuation=0.5)
        base.update(kwargs)
        with pytest.raises(ConfigurationError, match='must be'):
            uacpy.SedimentLayer(**base)

    @pytest.mark.parametrize('kwargs', [
        {'sound_speed': -1800.0, 'shear_speed': 600.0},   # TI / dispersive
        {'sound_speed': -1800.0, 'shear_speed': -600.0},  # Biot pair
        {'shear_attenuation': -888.888},                  # flow
    ])
    def test_boundary_properties_refuses_the_flag_values(self, kwargs):
        base = dict(acoustic_type='half-space', density=1.8, attenuation=0.5)
        base.update(kwargs)
        with pytest.raises(ConfigurationError, match='must be'):
            uacpy.BoundaryProperties(**base)

    def test_ssp_speeds_cannot_go_negative_either(self):
        # A negative CC on a water record would flag a TI layer mid-column.
        with pytest.raises(ConfigurationError, match='must be positive'):
            uacpy.SoundSpeedProfile(depths=[0.0, 100.0], data=[1500.0, -1490.0])


class TestPinnedWavenumberCountKeepsTheWholeSpectrum:
    """``oast.tex:73-75`` bounds IC2 <= NW and ``oasp.tex:657-660`` Hanning-
    windows the kernel over IC2+1..NW before integration, so a pinned NW must
    be written with IC2 = NW — anything smaller silently tapers the spectrum's
    steep-angle tail away."""

    def test_oast_block_vii_is_nw_one_nw(self, tmp_path):
        path = tmp_path / 'oast_run.dat'
        write_oast_input(path, make_pekeris(), _SRC, _RCV, nw_samples=1024)
        assert '1024 1 1024' in _deck_lines(path)

    def test_oasp_block_vii_is_nw_one_nw_intf(self, tmp_path):
        path = tmp_path / 'oasp_run.dat'
        write_oasp_input(path, make_pekeris(), _SRC, _RCV, nw_samples=1000)
        assert '1000 1 1000 40' in _deck_lines(path)


class TestOasrPlotAxisBlockGates:
    """``oasr.tex`` Tables I-II mark Block VI "(only for NFOU>0)" and Block
    VII "(only for NAOU>0)", and ``unoasr21.f:159-172`` reads them under
    NIPLOT/NAPLOT, non-zero iff the respective increment is. A deck that
    carried the axes anyway would feed them to the *next* option-gated READ."""

    # The two literal axis rows the writer emits, stable by construction:
    # the |R| axis of Block VI and the 0-30 dB loss axis of Block VII.
    _BLOCK_VI_ROW = '0 1 12 0.2'
    _BLOCK_VII_ROW = '0 30 12 5'

    def _deck(self, tmp_path, **kw):
        path = tmp_path / 'oasr_run.dat'
        write_oasr_input(path, make_pekeris(), _SRC, _RCV, **kw)
        return _deck_lines(path)

    def test_positive_increments_write_both_blocks(self, tmp_path):
        lines = self._deck(tmp_path)
        assert self._BLOCK_VI_ROW in lines
        assert self._BLOCK_VII_ROW in lines

    def test_nfou_zero_drops_the_angle_axes_only(self, tmp_path):
        lines = self._deck(tmp_path, freq_output_increment=0)
        assert self._BLOCK_VI_ROW not in lines
        assert self._BLOCK_VII_ROW in lines

    def test_naou_zero_drops_the_frequency_axes_only(self, tmp_path):
        lines = self._deck(tmp_path, angle_output_increment=0)
        assert self._BLOCK_VI_ROW in lines
        assert self._BLOCK_VII_ROW not in lines


class TestRcoTableFollowsTheManual:
    """``oasr.tex:313-350``: the master header's fourth number identifies the
    abscissa — 1 for slowness (s/km), 2 for grazing angle (deg) — and each
    per-frequency header carries a trailing ``# …`` annotation
    (``oasjun21.f:27-30``) that is not data."""

    _ROWS = [(81.373070, 0.339462, -15.996886),
             (79.665359, 0.342714, -15.640528),
             (77.948311, 0.346415, -15.196492)]

    def _table(self, tmp_path, code, name='refl.dat'):
        # Default extension .dat so nothing but the header code can pick
        # the type; a caller-given name exercises the extension paths.
        path = tmp_path / name
        lines = [f"      50.000      50.000   1   {code}",
                 "       50.000     3  # Frequency, # of angles"]
        lines += [f"      {a:.6f}       {m:.6f}     {p:.6f}"
                  for a, m, p in self._ROWS]
        path.write_text('\n'.join(lines) + '\n')
        return path

    def test_code_two_reads_as_an_angle_table(self, tmp_path):
        data = read_oasr_reflection_coefficients(self._table(tmp_path, 2))
        assert data['sampling_type'] == 'angle'
        assert data['n_frequencies'] == 1
        assert data['frequencies'] == [50.0]
        assert np.allclose(data['angles_or_slowness'][0],
                           [r[0] for r in self._ROWS])
        assert np.allclose(data['magnitude'][0], [r[1] for r in self._ROWS])
        # Phase is stored in degrees (oasr.tex:313-314).
        assert np.allclose(data['phase'][0], [r[2] for r in self._ROWS])

    def test_code_one_reads_as_a_slowness_table(self, tmp_path):
        data = read_oasr_reflection_coefficients(self._table(tmp_path, 1))
        assert data['sampling_type'] == 'slowness'

    def test_the_header_code_decides_under_any_file_name(self, tmp_path):
        """A byte-identical slowness table parses as slowness whatever it
        was saved as: the header's fourth field is the code OASR itself
        stamps (unoasr21.f:204-205), while the extension is only the name.
        The matching .rco and a neutral name parse silently."""
        for name in ('t.rco', 't.dat'):
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                data = read_oasr_reflection_coefficients(
                    self._table(tmp_path, 1, name))
            assert data['sampling_type'] == 'slowness', name

    def test_a_disagreeing_extension_warns_and_follows_the_header(
            self, tmp_path):
        """A slowness table renamed .trc keeps its abscissa, with a warning
        naming both the extension and the header code."""
        with pytest.warns(UserWarning, match=r"\.trc.*slowness"):
            data = read_oasr_reflection_coefficients(
                self._table(tmp_path, 1, 't.trc'))
        assert data['sampling_type'] == 'slowness'
        assert np.allclose(data['angles_or_slowness'][0],
                           [r[0] for r in self._ROWS])


class TestXsmLayoutFollowsTheManual:
    """A ``.xsm`` built record for record from the WRITE statements in
    ``oasn.tex:583-618`` must read back with every header field in place and
    ``covariance[ifreq, ircv, jrcv] == COVMAT(IRCV, JRCV, IFREQ)`` — the
    manual's loop is column-major (IRCV fastest), so a reader that skipped the
    transpose would return the conjugate matrix."""

    _NRCV, _NFREQ = 3, 2
    _TITLE = 'FRAM IV environment.'

    @staticmethod
    def _value(ifreq, ircv, jrcv):
        # Asymmetric on purpose: transposing swaps the imaginary part's sign.
        return complex(ifreq * 100 + ircv * 10 + jrcv, ircv - jrcv)

    def _write(self, tmp_path):
        recl = 8
        n = self._NRCV
        records = {}
        title = self._TITLE.ljust(32)
        for k in range(4):                      # REC=1..4: TITLE(1:8)..(25:32)
            records[1 + k] = title[8 * k:8 * (k + 1)].encode('ascii')
        records[5] = struct.pack('<ii', n, self._NFREQ)
        records[6] = struct.pack('<ii', 0, 0)
        records[7] = struct.pack('<ff', 20.0, 30.0)      # FREQ1, FREQ2
        records[8] = struct.pack('<ff', 10.0, 0.0)       # DELFRQ = (F2-F1)/(NF-1)
        records[9] = struct.pack('<ff', 70.0, 50.0)      # SSLEV, WNLEV
        records[10] = struct.pack('<ff', 0.0, 0.0)
        for ifreq in range(1, self._NFREQ + 1):
            for jrcv in range(1, n + 1):
                for ircv in range(1, n + 1):
                    irec = 10 + ircv + (jrcv - 1) * n + (ifreq - 1) * n * n
                    v = self._value(ifreq, ircv, jrcv)
                    records[irec] = struct.pack('<ff', v.real, v.imag)
        path = tmp_path / 'noise.xsm'
        buf = bytearray(recl * max(records))
        for irec, payload in records.items():
            buf[(irec - 1) * recl:(irec - 1) * recl + len(payload)] = payload
        path.write_bytes(bytes(buf))
        return path

    def test_header_and_indexing(self, tmp_path):
        data = read_oasn_covariance(self._write(tmp_path))
        assert data['title'] == self._TITLE
        assert data['n_receivers'] == self._NRCV
        assert data['n_frequencies'] == self._NFREQ
        assert data['freq_min'] == 20.0 and data['freq_max'] == 30.0
        assert data['freq_delta'] == 10.0
        assert data['surface_noise_level'] == 70.0
        assert data['white_noise_level'] == 50.0
        cov = data['covariance']
        assert cov.shape == (self._NFREQ, self._NRCV, self._NRCV)
        for ifreq in range(1, self._NFREQ + 1):
            for ircv in range(1, self._NRCV + 1):
                for jrcv in range(1, self._NRCV + 1):
                    assert cov[ifreq - 1, ircv - 1, jrcv - 1] == (
                        self._value(ifreq, ircv, jrcv))


class TestRpoLayoutFollowsTheManual:
    """A ``.rpo`` built from the header WRITEs and the replica loop of
    ``oasn.tex:651-689`` — IRCV innermost, then IYR, IXR, IZR, IFREQ — must
    read back on the ``(n_freq, n_z, n_x, n_y, n_rcv)`` axes with the grid in
    the deck's own mixed units (z in m, x/y in km, ``oasn.tex:109-111``,
    :365-371) and the on-disk linear gain converted back to the dB the deck
    stated (``oasnun22.f:99``)."""

    _NZ, _NX, _NY, _NRCV = 3, 2, 1, 2

    @staticmethod
    def _rec(payload):
        return struct.pack('<i', len(payload)) + payload + \
            struct.pack('<i', len(payload))

    @staticmethod
    def _value(iz, ix, iy, ircv):
        return complex(iz * 100 + ix * 10 + iy, ircv)

    def _write(self, tmp_path):
        out = bytearray()
        out += self._rec('replica run'.ljust(80).encode('ascii'))
        out += self._rec(struct.pack('<ii', self._NRCV, 1))
        out += self._rec(struct.pack('<fff', 25.0, 25.0, 0.0))
        out += self._rec(struct.pack('<ffi', 10.0, 90.0, self._NZ))   # z in m
        out += self._rec(struct.pack('<ffi', 0.5, 2.5, self._NX))     # x in km
        out += self._rec(struct.pack('<ffi', 0.0, 0.0, self._NY))     # y in km
        # Array element data: X, Y, Z, ITYP, GAIN — GAIN already linear,
        # 10**(dB/20), because INPRCV converts in place before any output.
        out += self._rec(struct.pack('<fffif', 1.5, 0.0, 30.0, 1, 10.0))
        out += self._rec(struct.pack('<fffif', 0.0, 0.0, 60.0, 1, 1.0))
        for iz in range(1, self._NZ + 1):
            for ix in range(1, self._NX + 1):
                for iy in range(1, self._NY + 1):
                    for ircv in range(1, self._NRCV + 1):
                        v = self._value(iz, ix, iy, ircv)
                        out += self._rec(struct.pack('<ff', v.real, v.imag))
        path = tmp_path / 'noise.rpo'
        path.write_bytes(bytes(out))
        return path

    def test_axes_units_and_gain(self, tmp_path):
        data = read_oasn_replicas(self._write(tmp_path))
        assert data['title'] == 'replica run'
        assert (data['n_z'], data['n_x'], data['n_y']) == (3, 2, 1)
        # Grid fields come back in the deck's units: z in m, x/y in km.
        assert (data['z_min'], data['z_max']) == (10.0, 90.0)
        assert (data['x_min'], data['x_max']) == (0.5, 2.5)
        # Receiver rows are X, Y, Z in metres (oasn.tex:47-49, :666-668).
        assert data['receiver_positions'][0].tolist() == [1.5, 0.0, 30.0]
        assert data['receiver_positions'][1].tolist() == [0.0, 0.0, 60.0]
        # The stored linear factor 10.0 is 20 dB; unity is 0 dB.
        assert np.allclose(data['receiver_gains'], [20.0, 0.0])
        rep = data['replicas']
        assert rep.shape == (1, self._NZ, self._NX, self._NY, self._NRCV)
        for iz in range(1, self._NZ + 1):
            for ix in range(1, self._NX + 1):
                for ircv in range(1, self._NRCV + 1):
                    assert rep[0, iz - 1, ix - 1, 0, ircv - 1] == (
                        self._value(iz, ix, 1, ircv))


class TestPltTabulatedAxes:
    """``oases_graph.tex:155-170``: a curve tabulates an axis in the ``.plt``
    only when its increment is 0, and "if both DX and DY are specified as 0
    ... first read all N x-values and then all y-values". PLTWRI mirrors that
    on the write side (``oasgun21.f:658-660``), so a fully parameterised curve
    contributes no block at all and a doubly-tabulated one holds 2N values
    with the ordinate LAST."""

    @staticmethod
    def _plp(tmp_path, curves):
        def rec(value, label):
            return f"{value:<19}{label}"

        lines = [' OAST  MODU']
        for tag, n, xoff, dx, yoff, dy in curves:
            lines += [f' OAST  {tag}', 'ptit', 'title',
                      rec(0, 'NUMBER OF LABELS')]
            lines += [rec(0.0, name) for name in
                      ('XLEN', 'YLEN', 'IGRID', 'XLEFT', 'XRIGHT', 'XINC',
                       'XDIV', 'XTXT', 'XTYP', 'YDOWN', 'YUP', 'YINC',
                       'YDIV', 'YTXT', 'YTYP')]
            lines += [rec(1, 'NC'), rec(n, 'N'), rec(xoff, 'XOFF'),
                      rec(dx, 'DX'), rec(yoff, 'YOFF'), rec(dy, 'DY')]
        lines += [' OAST  PLTEND']
        (tmp_path / 'r.plp').write_text('\n'.join(lines) + '\n')
        return tmp_path / 'r.plp'

    def test_a_doubly_tabulated_curve_reads_the_last_n_as_ordinate(
            self, tmp_path):
        plp = self._plp(tmp_path, [('NTLRAN', 3, 0.0, 0.0, 0.0, 0.0)])
        xs = [10.0, 20.0, 30.0]
        ys = [41.0, 42.0, 43.0]
        (tmp_path / 'r.plt').write_text(
            '\n'.join(f' {v}' for v in xs + ys) + '\n\n')
        out = read_oast_tl(plp, [10.0])
        assert out['tl'].tolist() == [ys]

    def test_a_parameterised_curve_writes_no_block(self, tmp_path):
        # First curve fully parameterised (DX and DY both non-zero): it owns
        # no .plt block, so the TL curve must be matched to block 0.
        plp = self._plp(tmp_path, [
            ('NINTGR', 5, 0.0, 1.0, 0.0, 2.0),
            ('NTLRAN', 3, 1.0, 0.5, 0.0, 0.0),
        ])
        ys = [61.0, 62.0, 63.0]
        (tmp_path / 'r.plt').write_text(
            '\n'.join(f' {v}' for v in ys) + '\n\n')
        out = read_oast_tl(plp, [10.0])
        assert out['tl'].tolist() == [ys]
        assert out['ranges'].tolist() == [1000.0, 1500.0, 2000.0]

    def test_a_curve_that_tabulates_only_x_has_no_ordinate(self, tmp_path):
        # DX = 0 tabulates the abscissa, DY != 0 parameterises the ordinate:
        # the block holds x-values only and there is no TL to return.
        plp = self._plp(tmp_path, [('NTLRAN', 3, 0.0, 0.0, 0.0, 2.0)])
        (tmp_path / 'r.plt').write_text(' 10.0\n 20.0\n 30.0\n\n')
        with pytest.raises(FileFormatError, match='no ordinate'):
            read_oast_tl(plp, [10.0])


@pytest.mark.requires_oases
@pytest.mark.requires_binary
class TestContourOffsetZeroInvokesTheBinarysOwnDefault:
    """``oast.tex:281-304``: "This value is the default which is applied if
    COFF is specified to 0.0" — under option 'J' a zero offset is NOT a zero
    contour; ``unoast31.f:497-509`` substitutes the 60-dB-at-RANMAX value and
    says which branch it took on stdout. Pinned NW keeps the automatic-
    sampling branch (which always zeroes OFFDB first) out of the way, so this
    is the one configuration where the deck's COFF token decides."""

    def _run(self, tmp_path, offset):
        from uacpy.models import OAST
        from uacpy.models.oases import _oases_subprocess_env
        base = 'oast_run'
        write_oast_input(tmp_path / f'{base}.dat', make_pekeris(), _SRC, _RCV,
                         options='N T J', integration_offset=offset,
                         nw_samples=1024)
        proc = subprocess.run(
            [str(OAST(verbose=False)._exe)], cwd=tmp_path,
            env=_oases_subprocess_env(base),
            capture_output=True, text=True, timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-400:]
        return proc.stdout + proc.stderr

    def test_zero_offset_takes_the_binarys_default(self, tmp_path):
        assert 'THE DEFAULT CONTOUR OFFSET IS APPLIED' in self._run(
            tmp_path, 0)

    def test_a_positive_offset_is_the_users(self, tmp_path):
        assert 'THE USER DEFINED CONTOUR OFFSET IS APPLIED' in self._run(
            tmp_path, 1.5)


class TestReplicaGridCapIsRefusedBeforeLaunch:
    """``oasn.tex`` Block X states no limit on the replica grid, but
    ``unoasn22.f:186-188`` STOPs at ``NSMAX = 201`` points per axis
    (``src/compar.f:51``) — a character STOP, so the binary exits 0 having
    written no ``.rpo``. ``write_oasn_input`` therefore refuses the deck
    with a typed error before any launch."""

    def _write(self, tmp_path, **counts):
        write_oasn_input(
            tmp_path / 'oasn_run.dat', make_pekeris(), _SRC,
            uacpy.Receiver(depths=np.linspace(20.0, 90.0, 4), ranges=[0.0]),
            options='R J', nw_samples=256, **counts)

    @pytest.mark.parametrize('axis', ['replica_nz', 'replica_nx',
                                      'replica_ny'])
    def test_202_on_any_axis_raises_before_any_deck_reaches_the_binary(
            self, tmp_path, axis):
        with pytest.raises(ConfigurationError, match='NSMAX = 201'):
            self._write(tmp_path, **{axis: 202})

    def test_201_per_axis_is_the_densest_legal_grid(self, tmp_path):
        self._write(tmp_path, replica_nz=201, replica_nx=201, replica_ny=1)
        deck = (tmp_path / 'oasn_run.dat').read_text()
        assert ' 201\n' in deck


class TestOastLogFrequencyLadderIsRefused:
    """OAST option ``'o'`` makes ``unoast31.f:393`` run the sweep
    LOG-spaced, but the ``.plp`` records no frequency values, so
    ``read_oast_tl`` could only label the curves with the deck's linear
    sweep. Both arms of ``'o'`` are refused with distinct diagnoses: a
    multi-frequency deck for the mislabeling, a single-frequency deck for
    the binary's own ``CONTOURS REQUIRE NRFR>1`` STOP."""

    def _write(self, tmp_path, freqs):
        from uacpy.io.oases_writer import write_oast_input
        write_oast_input(
            tmp_path / 'oast_run.dat', make_pekeris(),
            uacpy.Source(depths=50.0, frequencies=freqs),
            uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
            options='N J T o')

    def test_multi_frequency_sweep_would_be_mislabeled(self, tmp_path):
        with pytest.raises(ConfigurationError, match='log-spaced'):
            self._write(tmp_path, [50.0, 100.0, 150.0])

    def test_single_frequency_hits_the_contour_stop_guard(self, tmp_path):
        with pytest.raises(ConfigurationError, match='REQUIRE NRFR'):
            self._write(tmp_path, 100.0)


@pytest.mark.requires_oases
@pytest.mark.requires_binary
class TestOasrSlownessSamplingIsRefused:
    """OASR raw ``options`` with ``'p'``: the writer always emits Block V
    in grazing degrees, which ``oasjun21.f:36-37`` would reinterpret as
    slownesses in s/km — refused at construction."""

    def test_p_option_raises_at_construction(self):
        from uacpy.models.oases import OASR
        from uacpy.core.exceptions import UnsupportedFeatureError
        with pytest.raises(UnsupportedFeatureError, match='slowness'):
            OASR(options='N p')


class TestOasrTablesAcceptFortranRealSpellings:
    """The ``.rco``/``.trc`` value columns are Fortran real writes; a
    double-precision OASR built elsewhere can emit ``1.0D+00`` and any value
    below ~1e-99 drops the exponent letter. ``fortran_float`` reads both;
    ``float()`` reads neither."""

    @staticmethod
    def _write(tmp_path, magnitude_token):
        p = tmp_path / 'table.trc'
        p.write_text(
            "      10.000     100.000   1   2\n"
            "      10.000      2  # Frequency, # of angles\n"
            f"      10.000000    {magnitude_token}      5.000000\n"
            "      30.000000       0.900000     15.000000\n"
        )
        return p

    def test_a_letterless_three_digit_exponent_parses(self, tmp_path):
        from uacpy.io.oases_reader import read_oasr_reflection_coefficients
        data = read_oasr_reflection_coefficients(
            self._write(tmp_path, '0.123457-118'))
        assert data['magnitude'][0][0] == pytest.approx(
            1.23457e-119, rel=1e-6)

    def test_a_d_exponent_parses(self, tmp_path):
        from uacpy.io.oases_reader import read_oasr_reflection_coefficients
        data = read_oasr_reflection_coefficients(
            self._write(tmp_path, '     1.5D+00'))
        assert data['magnitude'][0][0] == pytest.approx(1.5, rel=1e-12)

    def test_a_d_exponent_header_frequency_parses(self, tmp_path):
        from uacpy.io.oases_reader import read_oasr_reflection_coefficients
        p = tmp_path / 'hdr.trc'
        p.write_text(
            "     1.0D+01     100.000   1   2\n"
            "      10.000      1  # Frequency, # of angles\n"
            "      10.000000       0.500000      5.000000\n"
        )
        data = read_oasr_reflection_coefficients(p)
        assert data['freq_min'] == pytest.approx(10.0, rel=1e-12)
