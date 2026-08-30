"""Three readers that walked their data one record at a time now stride it.

Each fix replaces a per-record loop (or a per-value list scan) with a bulk
read, so every test here is an equivalence test: the same crafted file is read
by the reader and by a reference implementation of the loop the fix removed,
and the two arrays must agree bit for bit. The reference readers below are
that removed code, kept verbatim so a future rewrite is measured against what
the loop actually did rather than against a rounded expectation.

Two of the three had a trap the obvious bulk read walks into:

* an OASP ``.trf`` is single- **or** double-precision, so the Fortran record
  marker the bulk read validates is the detected payload width, not the
  constant ``8`` that the single-precision ``.rpo`` replica reader can hardcode.
  ``test_oasp_trf_double_precision_*`` is what a hardcoded ``8`` fails.
* the ``.plp`` frequency/depth axes are ordered by first appearance, and
  :func:`~uacpy.io.oases_reader._match_label_depths` hands out receiver depths
  greedily in that order, so a ``set`` would reassign curves to other
  receivers instead of raising. ``test_oast_depth_axis_*`` is what a ``set``
  fails.

The synthetic ``.shd`` writer here is also what ``TestMultiSourceShadeFiles``
at the end of the file uses: ``read_shd_bin``'s ``xs_km=``/``ys_km=`` source
selector and its compressed ``'TL'`` source grid decode layouts no uacpy run
produces, so no model test reaches them.
"""

import struct

import numpy as np
import pytest

from uacpy.core.exceptions import FileFormatError
from uacpy.io import oalib_reader
from uacpy.io._fortran_helpers import read_fortran_record
from uacpy.io.oalib_reader import read_shd_bin
from uacpy.io.oases_reader import (
    _match_label_depths,
    _oast_curve_slots,
    _read_oasp_trf_binary,
    _unique_in_order,
)

_E = '<'


# --------------------------------------------------------------------------
# Synthetic file writers
# --------------------------------------------------------------------------

def _fortran_record(payload: bytes) -> bytes:
    marker = struct.pack(_E + 'i', len(payload))
    return marker + payload + marker


def _write_trf(path, values, nplots, nrd, double=False):
    """A minimal single-output OASP ``.trf`` carrying ``values``.

    ``values`` is ``(nf, nplots, nrd, 2)`` of real/imaginary pairs. Header
    layout per ``_read_oasp_trf_binary``'s own record list (oasiun23.f).
    """
    nf = values.shape[0]
    out = bytearray()
    out += _fortran_record(b'PULSETRF')
    out += _fortran_record(b'OASP  ')
    out += _fortran_record(struct.pack(_E + 'i', 1))            # NOUT
    out += _fortran_record(struct.pack(_E + 'i', 1))            # IPARM -> 'N'
    out += _fortran_record(b' ' * 80)                           # TITLE
    out += _fortran_record(b' ')                                # SIGNN
    out += _fortran_record(struct.pack(_E + 'f', 250.0))        # FREQS
    out += _fortran_record(struct.pack(_E + 'f', 50.0))         # SD
    out += _fortran_record(struct.pack(_E + 'ffi', 0.0, 100.0, nrd))
    out += _fortran_record(struct.pack(_E + 'ffi', 1.0, 1.0, nplots))
    out += _fortran_record(struct.pack(_E + 'iiif', 1024, 1, nf, 0.01))
    out += _fortran_record(struct.pack(_E + 'i', 0))            # ICDR
    out += _fortran_record(struct.pack(_E + 'f', 0.0))          # OMEGIM
    out += _fortran_record(struct.pack(_E + 'i', 1))            # MSUFT
    out += _fortran_record(struct.pack(_E + 'i', 1))            # ISROW
    for _ in range(3):                                          # INTTYP, IDUMMY
        out += _fortran_record(struct.pack(_E + 'i', 0))
    for _ in range(5):
        out += _fortran_record(struct.pack(_E + 'f', 0.0))      # DUMMY
    header_bytes = len(out)

    fmt = 'd' if double else 'f'
    for re_im in values.reshape(-1, 2):
        out += _fortran_record(
            struct.pack(_E + f'2{fmt}', float(re_im[0]), float(re_im[1])))
    path.write_bytes(bytes(out))
    return header_bytes


def _write_shd(path, samples, Nfreq, Ntheta, Nsx, Nsy, Nsz, Nrz, Nrr,
               plot_type='rectilin ', recl_floor=0):
    """A synthetic ``.shd`` in the word-based DIRECT layout of RWSHDFile.f90.

    ``samples`` is ``(n_slabs, Ntheta, Nsz, Nrz, Nrr)`` complex64 written in
    the writers' record order. ``recl_floor`` forces extra padding into every
    record without changing any header count.
    """
    recl = max(41, 2 * Nfreq, 2 * Ntheta, 2 * Nsx, 2 * Nsy, Nsz, Nrz, 2 * Nrr,
               recl_floor)
    rec_bytes = 4 * recl

    def pad(payload):
        assert len(payload) <= rec_bytes
        return payload.ljust(rec_bytes, b'\x00')

    out = bytearray()
    out += pad(struct.pack(_E + 'i', recl) + b'round25'.ljust(80, b' '))
    out += pad(plot_type.encode('ascii').ljust(10, b' '))
    out += pad(struct.pack(_E + 'iiiiiii', Nfreq, Ntheta, Nsx, Nsy, Nsz,
                           Nrz, Nrr) + struct.pack(_E + 'dd', 250.0, 0.0))
    out += pad(struct.pack(_E + f'{Nfreq}d',
                           *[100.0 + 50.0 * k for k in range(Nfreq)]))
    out += pad(struct.pack(_E + f'{Ntheta}d',
                           *[10.0 * k for k in range(Ntheta)]))
    out += pad(struct.pack(_E + f'{Nsx}d',
                           *[1000.0 * k for k in range(Nsx)]))
    out += pad(struct.pack(_E + f'{Nsy}d',
                           *[2000.0 * k for k in range(Nsy)]))
    out += pad(struct.pack(_E + f'{Nsz}f', *[10.0 * k for k in range(Nsz)]))
    out += pad(struct.pack(_E + f'{Nrz}f', *[5.0 * k for k in range(Nrz)]))
    out += pad(struct.pack(_E + f'{Nrr}d',
                           *[100.0 * (k + 1) for k in range(Nrr)]))
    for row in samples.reshape(-1, Nrr):
        interleaved = np.empty(2 * Nrr, dtype=np.float32)
        interleaved[0::2] = row.real
        interleaved[1::2] = row.imag
        out += pad(interleaved.tobytes())
    path.write_bytes(bytes(out))
    return recl


def _nonzero_complex64(rng, shape):
    """Samples the reader will not rewrite: an exact 0 on disk decodes NaN."""
    return (rng.standard_normal(shape) + 1.0
            + 1j * (rng.standard_normal(shape) + 1.0)).astype(np.complex64)


# --------------------------------------------------------------------------
# Reference readers: the per-record loops the fixes replaced
# --------------------------------------------------------------------------

def _trf_reference(path, nf, nplots, nrd, double):
    """``.trf`` data block read one ``read_fortran_record`` call at a time."""
    fmt = '2d' if double else '2f'
    out = np.zeros((nf, nplots, nrd),
                   dtype=np.complex128 if double else np.complex64)
    with open(path, 'rb') as f:
        for _ in range(23):                     # header records
            read_fortran_record(f, raw=True, endian=_E)
        for j in range(nf):
            for jrh in range(nplots):
                for jrv in range(nrd):
                    rec = read_fortran_record(f, fmt, endian=_E)
                    out[j, jrh, jrv] = complex(rec[0], rec[1])
    return out


def _shd_reference(path, first_record, Ntheta, Nsz, Nrz_per_range, Nrr, recl):
    """``.shd`` pressure block read with one seek per receiver-depth record."""
    f4 = np.dtype(_E + 'f4')
    out = np.zeros((Ntheta, Nsz, Nrz_per_range, Nrr), dtype=np.complex64)
    with open(path, 'rb') as fid:
        for itheta in range(Ntheta):
            for isz in range(Nsz):
                for irz in range(Nrz_per_range):
                    recnum = first_record + (
                        itheta * Nsz * Nrz_per_range + isz * Nrz_per_range
                        + irz)
                    fid.seek(recnum * 4 * recl, 0)
                    temp = np.fromfile(fid, dtype=f4, count=2 * Nrr)
                    out[itheta, isz, irz, :] = temp[0::2] + 1j * temp[1::2]
    return out


def _unique_in_order_reference(values):
    """First-appearance unique by membership scan — the pre-fix walk."""
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


# ==========================================================================
# S6 — OASP .trf
# ==========================================================================

def test_oasp_trf_single_precision_block_equals_record_at_a_time_read(tmp_path):
    rng = np.random.default_rng(0)
    nf, nplots, nrd = 4, 3, 2
    values = rng.standard_normal((nf, nplots, nrd, 2)).astype(np.float32)
    path = tmp_path / 'single.trf'
    _write_trf(path, values, nplots, nrd, double=False)

    got = _read_oasp_trf_binary(path, np.arange(nrd, dtype=float))
    tf = got['transfer_function']
    assert tf.dtype == np.complex64
    assert np.array_equal(tf, _trf_reference(path, nf, nplots, nrd, False))
    assert np.array_equal(tf, (values[..., 0] + 1j * values[..., 1]
                               ).astype(np.complex64))


def test_oasp_trf_double_precision_block_equals_record_at_a_time_read(tmp_path):
    """The record marker of a COMPLEX*16 ``.trf`` is 16, not the ``.rpo`` 8.

    A bulk read that hardcodes the marker value the single-precision ``.rpo``
    replicas carry rejects this file instead of reading it.
    """
    rng = np.random.default_rng(1)
    nf, nplots, nrd = 4, 3, 2
    values = rng.standard_normal((nf, nplots, nrd, 2)).astype(np.float64)
    path = tmp_path / 'double.trf'
    _write_trf(path, values, nplots, nrd, double=True)

    got = _read_oasp_trf_binary(path, np.arange(nrd, dtype=float))
    tf = got['transfer_function']
    assert tf.dtype == np.complex128
    assert np.array_equal(tf, _trf_reference(path, nf, nplots, nrd, True))
    # Full float64 payload: the values survive a round trip float32 would lose.
    assert np.array_equal(tf, values[..., 0] + 1j * values[..., 1])


def test_oasp_trf_double_precision_marker_is_not_the_single_precision_eight(
        tmp_path):
    """A COMPLEX*16 file whose markers were rewritten to 8 must be rejected."""
    rng = np.random.default_rng(2)
    values = rng.standard_normal((2, 2, 2, 2)).astype(np.float64)
    path = tmp_path / 'bad_marker.trf'
    header_bytes = _write_trf(path, values, 2, 2, double=True)
    raw = bytearray(path.read_bytes())
    raw[header_bytes:header_bytes + 4] = struct.pack(_E + 'i', 8)
    path.write_bytes(bytes(raw))

    with pytest.raises(FileFormatError):
        _read_oasp_trf_binary(path, np.arange(2, dtype=float))


@pytest.mark.parametrize('corrupt', ['truncate_payload', 'truncate_records',
                                     'tail_marker', 'head_marker',
                                     'unreasonable_length'])
def test_oasp_trf_malformed_data_records_raise_file_format_error(
        tmp_path, corrupt):
    """Bulk reading must not soften the per-record validations it replaced.

    ``read_fortran_record`` raised on a short read, on a head/tail marker
    mismatch and on a length no record could have; all three are corruptions
    a strided read would otherwise decode into plausible-looking numbers.
    """
    rng = np.random.default_rng(3)
    values = rng.standard_normal((3, 2, 2, 2)).astype(np.float32)
    path = tmp_path / f'{corrupt}.trf'
    header_bytes = _write_trf(path, values, 2, 2, double=False)
    raw = bytearray(path.read_bytes())
    record_bytes = 4 + 8 + 4

    if corrupt == 'truncate_payload':
        del raw[-3:]
    elif corrupt == 'truncate_records':
        del raw[header_bytes + 2 * record_bytes:]
    elif corrupt == 'tail_marker':
        at = header_bytes + 2 * record_bytes + 4 + 8
        raw[at:at + 4] = struct.pack(_E + 'i', 999)
    elif corrupt == 'head_marker':
        at = header_bytes + record_bytes
        raw[at:at + 4] = struct.pack(_E + 'i', 12345)
    else:
        at = header_bytes + record_bytes
        raw[at:at + 4] = struct.pack(_E + 'i', 1 << 29)
    path.write_bytes(bytes(raw))

    with pytest.raises(FileFormatError):
        _read_oasp_trf_binary(path, np.arange(2, dtype=float))


# ==========================================================================
# S8 — OAST .plp curve -> slot attribution
# ==========================================================================

def test_unique_in_order_keeps_first_appearance_order(tmp_path):
    scrambled = [300.0, 100.0, 300.0, 500.0, 200.0, 100.0, 400.0, 250.0]
    assert _unique_in_order(scrambled) == _unique_in_order_reference(scrambled)
    assert _unique_in_order(scrambled) == [300.0, 100.0, 500.0, 200.0,
                                           400.0, 250.0]


def test_oast_curve_slots_index_the_first_occurrence_of_each_label(tmp_path):
    """Every curve lands on the slot its ``.index()`` scan chose."""
    freqs = [300.0, 100.0, 500.0]
    depths = [17.0, 3.0, 11.0]
    curves = [{'labels': {'Freq': f, 'RD': d}} for f in freqs for d in depths]

    freq_axis, depth_axis, slots = _oast_curve_slots(
        tmp_path / 'x.plp', curves, depths)

    assert freq_axis == freqs and depth_axis == depths
    assert slots == [(freqs.index(c['labels']['Freq']),
                      depths.index(c['labels']['RD'])) for c in curves]


def test_oast_depth_axis_order_decides_which_depth_each_curve_reports():
    """Reordering the depth axis reports curves at other depths, silently.

    :func:`_match_label_depths` walks the axis in order and *consumes* the
    first receiver each label lands within half a print quantum of, so two
    labels competing for one receiver are settled by which comes first. The
    2.25 m receiver below sits inside the 0.05 m window of both the ``2.2``
    and the ``2.3`` label, which is the competition; whichever label is
    walked first takes it and the other falls through to a different answer.

    So a ``set`` — or any other order-losing unique — is not a permutation of
    this result. It is a different depth on the same curve, with nothing
    raised to say so.
    """
    receivers = [2.25, 2.28]
    file_order = [2.3, 2.2]              # the order the .plp's curves appear
    curves = [{'labels': {'Freq': 100.0, 'RD': d}} for d in file_order]

    _, depth_axis, slots = _oast_curve_slots(None, curves, receivers)
    assert depth_axis == file_order
    assert slots == [(0, 0), (0, 1)]

    in_file_order = _match_label_depths(depth_axis, receivers)
    assert {c['labels']['RD']: in_file_order[s[1]]
            for c, s in zip(curves, slots)} == {2.3: 2.25, 2.2: 2.2}

    reordered = sorted(file_order)
    assert reordered != file_order
    reordered_depths = _match_label_depths(reordered, receivers)
    assert {label: reordered_depths[reordered.index(label)]
            for label in file_order} == {2.3: 2.28, 2.2: 2.25}


# ==========================================================================
# S9 — .shd pressure block
# ==========================================================================

_SHD_SHAPES = {
    # recl == 2*Nrr: every record is payload, no padding to skip or move.
    'payload_bound': dict(Nfreq=1, Ntheta=1, Nsx=1, Nsy=1, Nsz=2, Nrz=6,
                          Nrr=40, recl_floor=0),
    # recl set by NRz, so each record trails 60 words the seek loop skipped.
    'padded_by_receiver_depths': dict(Nfreq=1, Ntheta=1, Nsx=1, Nsy=1, Nsz=1,
                                      Nrz=100, Nrr=20, recl_floor=0),
    # recl forced well past every header count: padding dominates the record.
    'padded_beyond_every_count': dict(Nfreq=1, Ntheta=2, Nsx=1, Nsy=1, Nsz=1,
                                      Nrz=3, Nrr=8, recl_floor=300),
    # NRz_per_range == 1: the paired-coordinate BELLHOP layout.
    'irregular': dict(Nfreq=1, Ntheta=1, Nsx=1, Nsy=1, Nsz=3, Nrz=1, Nrr=25,
                      recl_floor=0),
}


@pytest.mark.parametrize('shape', sorted(_SHD_SHAPES))
def test_shd_pressure_block_equals_seek_per_record_read(tmp_path, shape):
    kw = dict(_SHD_SHAPES[shape])
    plot_type = 'irregular ' if shape == 'irregular' else 'rectilin '
    rows = kw['Ntheta'] * kw['Nsz'] * kw['Nrz']
    rng = np.random.default_rng(4)
    samples = _nonzero_complex64(rng, (1, kw['Ntheta'], kw['Nsz'],
                                       kw['Nrz'], kw['Nrr']))
    path = tmp_path / f'{shape}.shd'
    recl = _write_shd(path, samples, plot_type=plot_type, **kw)

    got = read_shd_bin(str(path))['pressure']
    reference = _shd_reference(path, 10, kw['Ntheta'], kw['Nsz'], kw['Nrz'],
                               kw['Nrr'], recl)

    assert got.shape == reference.shape
    assert np.array_equal(got, reference)
    assert np.array_equal(got, samples[0])
    assert rows * recl >= rows * 2 * kw['Nrr']


def test_shd_padding_beyond_the_stride_budget_reads_the_same_samples(
        tmp_path, monkeypatch):
    """The seek loop is kept for records that are mostly padding.

    A strided read moves every padding word the seek loop skips, so past a
    budget the reader falls back. Both routes must decode the same file.
    """
    rng = np.random.default_rng(5)
    samples = _nonzero_complex64(rng, (1, 2, 2, 5, 12))
    path = tmp_path / 'padded.shd'
    recl = _write_shd(path, samples, Nfreq=1, Ntheta=2, Nsx=1, Nsy=1, Nsz=2,
                      Nrz=5, Nrr=12, recl_floor=400)
    reference = _shd_reference(path, 10, 2, 2, 5, 12, recl)

    strided = read_shd_bin(str(path))['pressure']
    monkeypatch.setattr(oalib_reader, '_SHD_STRIDE_PADDING_BUDGET_BYTES', 0)
    fallback = read_shd_bin(str(path))['pressure']

    assert np.array_equal(strided, reference)
    assert np.array_equal(fallback, reference)
    assert np.array_equal(fallback, samples[0])


def test_shd_pressure_block_read_in_slices_equals_one_whole_read(
        tmp_path, monkeypatch):
    """Chunking the strided read must not drop or duplicate a record."""
    rng = np.random.default_rng(6)
    samples = _nonzero_complex64(rng, (1, 3, 2, 7, 16))
    path = tmp_path / 'chunked.shd'
    recl = _write_shd(path, samples, Nfreq=1, Ntheta=3, Nsx=1, Nsy=1, Nsz=2,
                      Nrz=7, Nrr=16)
    whole = read_shd_bin(str(path))['pressure']

    # A budget of one and a half records forces an uneven slice boundary.
    monkeypatch.setattr(oalib_reader, '_SHD_STRIDE_CHUNK_BYTES', 6 * recl)
    sliced = read_shd_bin(str(path))['pressure']

    assert np.array_equal(sliced, whole)
    assert np.array_equal(sliced, samples[0])


def test_shd_frequency_selection_reads_the_slab_of_that_frequency(tmp_path):
    """``frequency=`` seeks straight to its slab rather than walking to it."""
    Nfreq, Ntheta, Nsz, Nrz, Nrr = 5, 1, 2, 4, 20
    rng = np.random.default_rng(7)
    samples = _nonzero_complex64(rng, (Nfreq, Ntheta, Nsz, Nrz, Nrr))
    path = tmp_path / 'broadband.shd'
    recl = _write_shd(path, samples, Nfreq=Nfreq, Ntheta=Ntheta, Nsx=1, Nsy=1,
                      Nsz=Nsz, Nrz=Nrz, Nrr=Nrr)
    rows_per_slab = Ntheta * Nsz * Nrz

    for ifreq in range(Nfreq):
        got = read_shd_bin(str(path), frequency=100.0 + 50.0 * ifreq)
        reference = _shd_reference(path, 10 + ifreq * rows_per_slab, Ntheta,
                                   Nsz, Nrz, Nrr, recl)
        assert got['pressure_freq'] == 100.0 + 50.0 * ifreq
        assert np.array_equal(got['pressure'], reference)
        assert np.array_equal(got['pressure'], samples[ifreq])


def test_shd_source_position_selection_reads_the_slab_of_that_source(tmp_path):
    """The 3-D path's ``(idxX, idxY)`` base record survives the bulk read."""
    Ntheta, Nsz, Nrz, Nrr = 2, 2, 3, 12
    Nsx, Nsy = 3, 2
    rng = np.random.default_rng(8)
    samples = _nonzero_complex64(rng, (Nsx * Nsy, Ntheta, Nsz, Nrz, Nrr))
    path = tmp_path / 'multisource.shd'
    recl = _write_shd(path, samples, Nfreq=1, Ntheta=Ntheta, Nsx=Nsx, Nsy=Nsy,
                      Nsz=Nsz, Nrz=Nrz, Nrr=Nrr)
    rows_per_slab = Ntheta * Nsz * Nrz

    for idx_x in range(Nsx):
        for idx_y in range(Nsy):
            got = read_shd_bin(str(path), xs_km=float(idx_x),
                               ys_km=2.0 * idx_y)['pressure']
            slab = idx_x * Nsy + idx_y
            reference = _shd_reference(path, 10 + slab * rows_per_slab,
                                       Ntheta, Nsz, Nrz, Nrr, recl)
            assert np.array_equal(got, reference)
            assert np.array_equal(got, samples[slab])


@pytest.mark.parametrize('missing_bytes', [3, 200, 4000])
def test_shd_truncated_pressure_block_raises_file_format_error(tmp_path,
                                                               missing_bytes):
    rng = np.random.default_rng(9)
    samples = _nonzero_complex64(rng, (1, 1, 2, 8, 50))
    path = tmp_path / 'truncated.shd'
    _write_shd(path, samples, Nfreq=1, Ntheta=1, Nsx=1, Nsy=1, Nsz=2, Nrz=8,
               Nrr=50)
    path.write_bytes(path.read_bytes()[:-missing_bytes])

    with pytest.raises(FileFormatError):
        read_shd_bin(str(path))


def test_shd_record_shorter_than_its_own_range_row_names_the_disagreement(
        tmp_path):
    """``LRecl >= 2*NRr`` by construction; a file that breaks it is corrupt.

    The strided read builds a record dtype from both numbers, so this has to
    be caught by name — left to numpy it surfaces as an out-of-bounds field.
    """
    rng = np.random.default_rng(10)
    samples = _nonzero_complex64(rng, (1, 1, 1, 4, 10))
    path = tmp_path / 'inconsistent.shd'
    recl = _write_shd(path, samples, Nfreq=1, Ntheta=1, Nsx=1, Nsy=1, Nsz=1,
                      Nrz=4, Nrr=10)
    assert recl == 41                       # the MAX(41, ...) floor
    raw = bytearray(path.read_bytes())
    # NRr sits last in header record 3. 21 ranges need 168 bytes of payload,
    # one word more than the record holds, but stay inside the size bounds
    # the reader checks before it gets there.
    raw[2 * 4 * recl + 24:2 * 4 * recl + 28] = struct.pack(_E + 'i', 21)
    path.write_bytes(bytes(raw))

    with pytest.raises(FileFormatError, match=r'LRecl is at least 2\*NRr'):
        read_shd_bin(str(path))


# --------------------------------------------------------------------------
# .shd multi-source selection and the compressed FIELD3D source grid
# --------------------------------------------------------------------------

class TestMultiSourceShadeFiles:
    """``read_shd_bin``'s ``xs_km=``/``ys_km=`` selector and its ``'TL'``
    source-grid layout decode files no uacpy run produces.

    Every uacpy wrapper writes a single-source 2-D deck, so the model tests
    reach neither branch: the selector's record index
    (``10 + (idxX * Nsy + idxY) * rows_per_slab``) and the compressed
    two-limit source grid are exercised only by a file built here."""

    #: One slab per (source x, source y) slot, x-major — the nesting
    #: ``bellhop3D.f90:407-410`` builds its record index from.
    NSX, NSY = 3, 2
    NTHETA, NSZ, NRZ, NRR = 1, 1, 2, 4

    def _slabs(self):
        """A distinguishable constant per source slot, so a wrong slab is a
        wrong *value*, not a shape mismatch."""
        n_slabs = self.NSX * self.NSY
        shape = (n_slabs, self.NTHETA, self.NSZ, self.NRZ, self.NRR)
        samples = np.empty(shape, dtype=np.complex64)
        for slot in range(n_slabs):
            samples[slot] = np.complex64((slot + 1) + 1j * (slot + 1))
        return samples

    def _write(self, path, plot_type='rectilin '):
        return _write_shd(path, self._slabs(), Nfreq=1, Ntheta=self.NTHETA,
                          Nsx=self.NSX, Nsy=self.NSY, Nsz=self.NSZ,
                          Nrz=self.NRZ, Nrr=self.NRR, plot_type=plot_type)

    @pytest.mark.parametrize('idx_x,idx_y', [(0, 0), (1, 0), (2, 1)],
                             ids=['first', 'middle-x', 'last'])
    def test_the_selector_returns_the_slab_of_the_named_source(
            self, tmp_path, idx_x, idx_y):
        # _write_shd lays the source axes out at 1000 m and 2000 m spacing,
        # and the selector takes km.
        path = tmp_path / 'multi.shd'
        self._write(path)
        data = read_shd_bin(str(path), xs_km=idx_x * 1.0, ys_km=idx_y * 2.0)
        slot = idx_x * self.NSY + idx_y
        expected = (slot + 1) + 1j * (slot + 1)
        assert data['pressure'].shape == (self.NTHETA, self.NSZ, self.NRZ,
                                          self.NRR)
        assert np.all(data['pressure'] == expected)

    def test_the_nearest_source_is_chosen_on_both_sides_of_a_midpoint(
            self, tmp_path):
        # The selector is an argmin, so the boundary is the midpoint between
        # two source x positions: 0 m and 1000 m meet at 0.5 km.
        path = tmp_path / 'nearest.shd'
        self._write(path)
        below = read_shd_bin(str(path), xs_km=0.499, ys_km=0.0)
        above = read_shd_bin(str(path), xs_km=0.501, ys_km=0.0)
        assert np.all(below['pressure'] == 1 + 1j)      # slot (0, 0)
        assert np.all(above['pressure'] == 3 + 3j)      # slot (1, 0)

    def test_xs_km_without_ys_km_raises_a_typed_error(self, tmp_path):
        from uacpy.core.exceptions import ConfigurationError
        path = tmp_path / 'half.shd'
        self._write(path)
        with pytest.raises(ConfigurationError, match='ys_km must be provided'):
            read_shd_bin(str(path), xs_km=0.0)

    def test_no_selector_on_a_multi_source_file_warns_and_takes_slot_zero(
            self, tmp_path):
        import warnings
        path = tmp_path / 'unselected.shd'
        self._write(path)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always')
            data = read_shd_bin(str(path))
        assert any('no xs_km=/ys_km= selector' in str(w.message)
                   for w in record), [str(w.message) for w in record]
        assert np.all(data['pressure'] == 1 + 1j)

    def test_a_single_source_file_selects_without_warning(self, tmp_path):
        # The other side of the Nsx/Nsy boundary: one slot, nothing to choose.
        import warnings
        path = tmp_path / 'single.shd'
        samples = np.full((1, 1, 1, self.NRZ, self.NRR), 7 + 7j,
                          dtype=np.complex64)
        _write_shd(path, samples, Nfreq=1, Ntheta=1, Nsx=1, Nsy=1, Nsz=1,
                   Nrz=self.NRZ, Nrr=self.NRR)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always')
            data = read_shd_bin(str(path))
        assert not [w for w in record if 'selector' in str(w.message)]
        assert np.all(data['pressure'] == 7 + 7j)

    def test_a_tl_plot_type_expands_the_source_grid_from_its_two_limits(
            self, tmp_path):
        """``misc/RWSHDFile.f90:126-127`` writes only the first and last
        source coordinate for a ``'TL'`` file; the grid between them is
        uniform, so the reader has to expand it rather than read Nsx values.
        """
        path = tmp_path / 'compressed.shd'
        recl = self._write(path, plot_type='TL        ')
        raw = bytearray(path.read_bytes())
        # Records 6 and 7 (0-based 5 and 6) carry the x and y limits.
        for record, limits in ((5, (0.0, 4000.0)), (6, (0.0, 3000.0))):
            start = record * 4 * recl
            raw[start:start + 4 * recl] = struct.pack(
                _E + 'dd', *limits).ljust(4 * recl, b'\x00')
        path.write_bytes(bytes(raw))

        data = read_shd_bin(str(path), xs_km=2.0, ys_km=3.0)
        np.testing.assert_allclose(data['Pos']['s']['x'], [0.0, 2000.0, 4000.0])
        np.testing.assert_allclose(data['Pos']['s']['y'], [0.0, 3000.0])
        # x index 1, y index 1 -> slot 1 * NSY + 1 = 3.
        assert np.all(data['pressure'] == 4 + 4j)

    def test_a_non_tl_plot_type_reads_every_source_coordinate(self, tmp_path):
        # The other side of the PlotType branch: the uncompressed layout
        # carries Nsx values, not two limits.
        path = tmp_path / 'uncompressed.shd'
        self._write(path)
        data = read_shd_bin(str(path), xs_km=0.0, ys_km=0.0)
        np.testing.assert_allclose(data['Pos']['s']['x'],
                                   [0.0, 1000.0, 2000.0])
        np.testing.assert_allclose(data['Pos']['s']['y'], [0.0, 2000.0])
