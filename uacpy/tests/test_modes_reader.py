"""Kraken ``.mod`` reader tests, pinned against the vendored ``kraken.exe``.

The mode file is Fortran direct-access binary whose record layout is
declared only in ``Kraken/kraken.f90``; every test here therefore runs a
hand-written deck through the shipped solver and checks the reader against
the eigenvalues the same run printed into its ``.prt``.
"""

import re
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

import uacpy
from uacpy.core.exceptions import ConfigurationError, FileFormatError
from uacpy.io.modes_reader import read_modes, read_modes_bin

_AT_BIN = Path(uacpy.__file__).parent / 'bin' / 'oalib'

# kraken.f90:101-102 prints "mode, k(mode), omega/k, VG" with the complex k
# split across the G18.10 / G10.2 fields, i.e. index, Re(k), Im(k), cp, cg.
_PRT_ROW = re.compile(
    r'^\s*(\d+)\s+([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)\s*$')


def _run_kraken(work_dir, root, deck, exe='kraken.exe'):
    """Write ``<root>.env`` and run an AT mode solver on it."""
    (work_dir / f'{root}.env').write_text(deck)
    subprocess.run([str(_AT_BIN / exe), root], cwd=str(work_dir),
                   capture_output=True, text=True, timeout=300)
    return (work_dir / f'{root}.prt').read_text()


def _prt_eigenvalues(prt_text):
    """Eigenvalues the solver printed, as ``{mode_number: complex}``."""
    rows = {}
    for line in prt_text.splitlines():
        m = _PRT_ROW.match(line)
        if m:
            rows[int(m.group(1))] = complex(float(m.group(2)),
                                            float(m.group(3)))
    return rows


def _record_length_words(mod_file):
    """LRecordLength in 4-byte longwords (kraken.f90:587-589)."""
    with open(mod_file, 'rb') as fid:
        return struct.unpack('<i', fid.read(4))[0]


def _layered_deck(title, n_sediment=10, c_high=20000.0, n_rd=15):
    """Deck with ``1 + n_sediment`` acoustic media.

    ``LRecordLength = MAX(2*Nfreq, 2*NzTab, 32, 3*NAcoustic)``
    (kraken.f90:587); with 11 acoustic media the last term is 33 and wins,
    which is the only way to get an odd record length.
    """
    lines = [f"'{title}'", '300.0', str(1 + n_sediment), "'NVW'",
             '0 0.0 100.0',
             '    0.0 1500.0 0.0 1.0 0.0 /',
             '  100.0 1500.0 /']
    top = 100.0
    for i in range(n_sediment):
        bot = top + 5.0
        c = 1550.0 + 10.0 * i
        lines += [f'0 0.0 {bot:.1f}',
                  f'  {top:.1f} {c:.1f} 0.0 1.6 0.1 /',
                  f'  {bot:.1f} {c:.1f} /']
        top = bot
    lines += ["'A' 0.0",
              f'  {top:.1f} 1800.0 0.0 2.0 0.5 /',
              f'1400.0 {c_high:.1f}', '10.0', '1', '50.0 /',
              str(n_rd), f'5.0 {top - 5.0:.1f} /']
    return '\n'.join(lines) + '\n'


def _isovelocity_deck(title, c=1500.0, freq=100.0, c_low=0.0, c_high=2500.0,
                      n_rd=11):
    return '\n'.join([
        f"'{title}'", f'{freq:.1f}', '1', "'NVW'",
        '0 0.0 100.0',
        f'    0.0 {c:.1f} 0.0 1.0 0.0 /',
        f'  100.0 {c:.1f} /',
        "'A' 0.0",
        '  100.0 1800.0 0.0 1.8 0.5 /',
        f'{c_low:.1f} {c_high:.1f}', '5.0', '1', '  50.0 /',
        str(n_rd), '  0.0 100.0 /',
    ]) + '\n'


@pytest.mark.requires_binary
class TestFoldedEigenvalueRecords:
    """``kraken.f90:108-113`` writes the eigenvalues one *record* at a time,
    ``LRecordLength/2`` complex values per record, each starting on a record
    boundary; ``KrakenField/ReadModes.f90:212-219`` reads them back the same
    way. Fortran truncates ``LRecordLength/2``, so an odd LRecordLength
    leaves one unwritten longword at the tail of every eigenvalue record. A
    single contiguous read absorbs that padding and shifts every value past
    the first record by one float32.
    """

    @pytest.fixture(scope='class')
    def run(self, tmp_path_factory):
        work = tmp_path_factory.mktemp('odd_lrecl')
        prt = _run_kraken(work, 'odd', _layered_deck('odd'))
        return work / 'odd.mod', _prt_eigenvalues(prt)

    def test_deck_produces_an_odd_record_length(self, run):
        mod_file, _ = run
        assert _record_length_words(mod_file) % 2 == 1

    def test_more_modes_than_one_record_holds(self, run):
        mod_file, printed = run
        assert max(printed) > _record_length_words(mod_file) // 2

    def test_wavenumbers_match_the_print_file(self, run):
        mod_file, printed = run
        k = read_modes_bin(str(mod_file))['k']
        assert len(k) == max(printed)
        for mode, expected in printed.items():
            got = k[mode - 1]
            # G18.10 / G10.2: Re(k) is printed to 10 significant digits,
            # Im(k) to 2, so half an ulp of Im(k) is up to ~5% of the value.
            assert np.real(got) == pytest.approx(expected.real, rel=1e-6), (
                f'mode {mode}: {got} != {expected}')
            assert np.imag(got) == pytest.approx(expected.imag, rel=6e-2), (
                f'mode {mode}: {got} != {expected}')

    def test_mode_subset_picks_the_same_wavenumber(self, run):
        mod_file, printed = run
        wanted = _record_length_words(mod_file) // 2 + 1   # second record
        subset = read_modes_bin(str(mod_file), modes=[wanted])
        assert subset['M'] == 1
        assert np.real(subset['k'][0]) == pytest.approx(
            printed[wanted].real, rel=1e-6)


@pytest.mark.requires_binary
def test_fold_that_cannot_hold_every_mode_is_rejected(tmp_path):
    """``kraken.f90:109`` sizes the eigenvalue loop as
    ``1 + (2M-1)/LRecordLength`` while :110 fills only ``LRecordLength/2``
    values per record. For an odd LRecordLength those disagree for some M and
    the trailing eigenvalues are never written, so there is no value on disk
    to return.
    """
    prt = _run_kraken(tmp_path, 'short',
                      _layered_deck('short', c_high=1850.0))
    mod_file = tmp_path / 'short.mod'
    lrecl_words = _record_length_words(mod_file)
    printed = _prt_eigenvalues(prt)
    assert lrecl_words % 2 == 1
    n_records = 1 + (2 * max(printed) - 1) // lrecl_words
    assert n_records * (lrecl_words // 2) < max(printed)

    with pytest.raises(FileFormatError, match='eigenvalue records hold only'):
        read_modes_bin(str(mod_file))


@pytest.mark.requires_binary
class TestProfileSequence:
    """A ``.mod`` holds one mode set per environmental profile —
    ``kraken.f90:42`` loops ``Profile: DO iProf = 1, 9999`` and each profile
    restarts with its own five header records (ReadModes.f90:19-25).
    """

    SPEEDS = (1500.0, 1480.0, 1520.0)

    @pytest.fixture(scope='class')
    def run(self, tmp_path_factory):
        work = tmp_path_factory.mktemp('profiles')
        deck = ''.join(_isovelocity_deck(f'P{i + 1}', c=c)
                       for i, c in enumerate(self.SPEEDS))
        prt = _run_kraken(work, 'p3', deck)
        # kraken restarts the eigenvalue table for every profile, so the
        # mode-1 rows appear in profile order.
        first = [complex(float(m.group(2)), float(m.group(3)))
                 for m in (_PRT_ROW.match(ln) for ln in prt.splitlines())
                 if m and int(m.group(1)) == 1]
        return work / 'p3.mod', first

    @pytest.mark.parametrize('index', [0, 1, 2])
    def test_each_profile_returns_its_own_mode_set(self, run, index):
        mod_file, first_mode = run
        data = read_modes_bin(str(mod_file), profile=index + 1)
        assert data['title'].endswith(f'P{index + 1}')
        assert np.real(data['k'][0]) == pytest.approx(
            first_mode[index].real, rel=1e-6)

    def test_default_profile_is_the_first(self, run):
        mod_file, _ = run
        assert (read_modes_bin(str(mod_file))['k'][0]
                == read_modes_bin(str(mod_file), profile=1)['k'][0])

    def test_profile_past_the_end_raises(self, run):
        mod_file, _ = run
        with pytest.raises(FileFormatError, match='profile header'):
            read_modes_bin(str(mod_file), profile=len(self.SPEEDS) + 1)

    def test_profile_below_one_raises(self, run):
        mod_file, _ = run
        with pytest.raises(ConfigurationError, match='profile must be'):
            read_modes_bin(str(mod_file), profile=0)

    def test_dispatcher_forwards_the_profile(self, run):
        mod_file, first_mode = run
        data = read_modes(str(mod_file), frequency=100.0, profile=3)
        assert data['title'].endswith('P3')
        assert np.real(data['k'][0]) == pytest.approx(
            first_mode[2].real, rel=1e-6)


def test_ascii_reader_has_no_profiles(tmp_path):
    moa = tmp_path / 'x.moa'
    moa.write_text('32\ntitle\n100.0 1 1 1 0\n')
    with pytest.raises(ConfigurationError, match='single mode set'):
        read_modes(str(moa), profile=2)


@pytest.mark.requires_binary
def test_zero_mode_file_keeps_the_documented_phi_shape(tmp_path):
    """``kraken.f90:948-963`` writes a degraded header (NzTab = 0) and M = 0
    when no mode falls in the phase-speed window, then aborts. The file is on
    disk and readable; ``phi`` must keep the ``(ntot, M)`` shape the reader
    documents so the result stays consumable.
    """
    _run_kraken(tmp_path, 'zm',
                _isovelocity_deck('zero modes', freq=20.0,
                                  c_low=1790.0, c_high=1799.0, n_rd=5))
    data = read_modes_bin(str(tmp_path / 'zm.mod'))
    assert data['M'] == 0
    assert data['k'].shape == (0,)
    assert data['phi'].shape == (len(data['z']), 0)

    modes = uacpy.core.results.Modes(
        k=data['k'], phi=data['phi'], depths=data['z'],
        model='Kraken', backend='kraken', frequencies=20.0)
    assert modes.n_modes == 0


@pytest.mark.requires_binary
def test_mode_count_is_bounded_by_the_file_size(tmp_path):
    """``M`` (kraken.f90:106) sizes both the mode index vector and the second
    axis of ``phi``, so it gets the same file-size bound as the record-0
    header counts. A hostile value must be a typed FileFormatError, not a
    multi-GB allocation.
    """
    _run_kraken(tmp_path, 'mb', _isovelocity_deck('bound'))
    mod_file = tmp_path / 'mb.mod'
    raw = bytearray(mod_file.read_bytes())
    lrecl = 4 * _record_length_words(mod_file)
    offset = 5 * lrecl                       # the mode-count record
    struct.pack_into('<i', raw, offset, 0x7ffffff0)
    mod_file.write_bytes(bytes(raw))

    with pytest.raises(FileFormatError, match='mode count M='):
        read_modes_bin(str(mod_file))


@pytest.mark.requires_binary
class TestHalfspaceVerticalWavenumber:
    """``KrakenField/ReadModes.f90:79`` takes the model from ``Title(1:7)``
    and :254-272 branches on it: KRAKENC forms the half-space vertical
    wavenumber from the full complex eigenvalue, KRAKEN from ``Re(k)`` only.
    """

    DECK = _isovelocity_deck('gam')

    def _gamma(self, tmp_path, exe, root):
        _run_kraken(tmp_path, root, self.DECK, exe=exe)
        return read_modes(str(tmp_path / f'{root}.mod'), frequency=100.0)

    def test_kraken_uses_the_real_part_only(self, tmp_path):
        data = self._gamma(tmp_path, 'kraken.exe', 'gk')
        assert data['title'][:7] == 'KRAKEN-'
        expected = np.sqrt(np.real(data['k']) ** 2 - data['Bot']['k2'])
        assert np.allclose(np.abs(data['Bot']['gamma']), np.abs(expected))
        # The imaginary part is non-negligible, so the two branches differ.
        full = np.sqrt(data['k'] ** 2 - data['Bot']['k2'])
        assert not np.allclose(np.abs(data['Bot']['gamma']), np.abs(full))

    def test_krakenc_uses_the_full_complex_eigenvalue(self, tmp_path):
        data = self._gamma(tmp_path, 'krakenc.exe', 'gc')
        assert data['title'][:7] == 'KRAKENC'
        expected = np.sqrt(data['k'] ** 2 - data['Bot']['k2'])
        assert np.allclose(np.abs(data['Bot']['gamma']), np.abs(expected))
