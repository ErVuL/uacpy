"""Kraken ``.mod`` reader tests, pinned against the vendored ``kraken.exe``.

The mode file is Fortran direct-access binary whose record layout is
declared only in ``Kraken/kraken.f90``; every test here therefore runs a
hand-written deck through the shipped solver and checks the reader against
the eigenvalues the same run printed into its ``.prt``.
"""

import re
import struct
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

import uacpy
from uacpy.core import Environment, BoundaryProperties, Source, Receiver
from uacpy.core.exceptions import ConfigurationError, FileFormatError
from uacpy.core.run_settings import RunMode
from uacpy.core.results import Modes
from uacpy.io.modes_reader import _read_modes_payload, read_modes
from uacpy.models import Kraken

_AT_BIN = Path(uacpy.__file__).parent / 'bin' / 'oalib'

# kraken.f90:101-102 prints "mode, k(mode), omega/k, VG" with the complex k
# split across the G18.10 / G10.2 fields, i.e. index, Re(k), Im(k), cp, cg.
_PRT_ROW = re.compile(
    r'^\s*(\d+)\s+([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)\s*$')


def test_a_mode_file_record_length_that_reads_both_ways_is_read_by_its_counts(
        tmp_path):
    """LRecordLength = 65536 little-endian reads 256 big-endian; the four
    counts after the title in record 1 (kraken.f90:593) settle the order."""
    import uacpy.io._fortran_helpers as fh
    from uacpy.io.modes_reader import _mod_header_counts
    record = bytearray(4 * 65536)
    record[0:84] = struct.pack('<i', 65536) + b'mode file'.ljust(80)
    record[84:100] = struct.pack('<4i', 1, 1, 201, 201)
    path = tmp_path / 'lrecl65536.mod'
    path.write_bytes(bytes(record))
    emitted = fh._ENDIAN_WARN_EMITTED
    try:
        fh._ENDIAN_WARN_EMITTED = True
        with open(path, 'rb') as fid:
            daf = fh.DirectAccessFile(fid, 'test',
                                      header_counts=_mod_header_counts)
            assert (daf.endian, daf.record_words) == ('<', 65536)
    finally:
        fh._ENDIAN_WARN_EMITTED = emitted


def test_a_large_mode_file_is_read_in_its_own_order(tmp_path):
    """A ``.mod`` past 16 MiB with one frequency and one medium: the
    record-1 counts byte-swapped (1 → 16 777 216) still lie inside the file,
    so only the mesh count opening record 2 (kraken.f90:594) settles the
    order. A sparse file keeps the disk cost to a few pages."""
    import uacpy.io._fortran_helpers as fh
    from uacpy.io.modes_reader import _mod_header_counts
    lrecl = 65536
    path = tmp_path / 'large.mod'
    with open(path, 'wb') as f:
        f.write(struct.pack('<i', lrecl) + b'mode file'.ljust(80)
                + struct.pack('<4i', 1, 1, 32768, 32768))
        f.seek(4 * lrecl)
        f.write(struct.pack('<i', 200) + b'ACOUSTIC')
        f.truncate(17 * 2 ** 20)
    emitted = fh._ENDIAN_WARN_EMITTED
    try:
        fh._ENDIAN_WARN_EMITTED = True
        with open(path, 'rb') as fid:
            daf = fh.DirectAccessFile(fid, 'test',
                                      header_counts=_mod_header_counts)
            assert (daf.endian, daf.record_words) == ('<', lrecl)
    finally:
        fh._ENDIAN_WARN_EMITTED = emitted


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
        k = _read_modes_payload(str(mod_file))['k']
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
        subset = _read_modes_payload(str(mod_file), modes=[wanted])
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
        _read_modes_payload(str(mod_file))


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
        data = _read_modes_payload(str(mod_file), profile=index + 1)
        assert data['title'].endswith(f'P{index + 1}')
        assert np.real(data['k'][0]) == pytest.approx(
            first_mode[index].real, rel=1e-6)

    def test_default_profile_is_the_first(self, run):
        mod_file, _ = run
        assert (_read_modes_payload(str(mod_file))['k'][0]
                == _read_modes_payload(str(mod_file), profile=1)['k'][0])

    def test_profile_past_the_end_raises(self, run):
        mod_file, _ = run
        with pytest.raises(FileFormatError, match='profile header'):
            _read_modes_payload(str(mod_file), profile=len(self.SPEEDS) + 1)

    def test_profile_below_one_raises(self, run):
        mod_file, _ = run
        with pytest.raises(ConfigurationError, match='profile must be'):
            _read_modes_payload(str(mod_file), profile=0)

    def test_dispatcher_forwards_the_profile(self, run):
        mod_file, first_mode = run
        data = read_modes(str(mod_file), frequency=100.0, profile=3)
        assert data.metadata['title'].endswith('P3')
        assert np.real(data.k[0]) == pytest.approx(
            first_mode[2].real, rel=1e-6)


def test_an_extension_other_than_mod_or_moa_is_a_typed_format_error(
        tmp_path):
    other = tmp_path / 'x.txt'
    other.write_text('32\ntitle\n100.0 1 1 1 0\n')
    with pytest.raises(FileFormatError, match=r'\.mod'):
        read_modes(str(other))


@pytest.mark.requires_binary
def test_the_payload_M_counts_the_modes_returned(tmp_path):
    """``M`` means the same thing in both readers: ``len(k)``. Reporting the
    file's total instead would leave a ``modes=`` subset disagreeing with the
    ``k`` / ``phi`` it is handed back with."""
    from uacpy.models import Kraken

    env = uacpy.Environment(
        name='mcount', bathymetry=100.0, ssp=1500.0,
        bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                        sound_speed=1800.0, density=1.8,
                                        attenuation=0.3))
    kraken = Kraken(verbose=False, work_dir=tmp_path, cleanup=False)
    modes = kraken.compute_modes(env, uacpy.Source(depths=50.0,
                                                   frequencies=100.0))
    assert modes.n_modes > 1

    mod_file = str(tmp_path / 'modes.mod')
    full = _read_modes_payload(mod_file)
    assert full['M'] == len(full['k']) == modes.n_modes

    subset = _read_modes_payload(mod_file, modes=[1])
    assert subset['M'] == 1 == len(subset['k']) == subset['phi'].shape[1]


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
    data = _read_modes_payload(str(tmp_path / 'zm.mod'))
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
        _read_modes_payload(str(mod_file))


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
        bot = data.halfspace_tails['bottom']
        assert data.metadata['title'][:7] == 'KRAKEN-'
        assert data.backend == 'kraken'
        expected = np.sqrt(np.real(data.k) ** 2 - bot['k2'])
        assert np.allclose(np.abs(bot['gamma']), np.abs(expected))
        # The imaginary part is non-negligible, so the two branches differ.
        full = np.sqrt(data.k ** 2 - bot['k2'])
        assert not np.allclose(np.abs(bot['gamma']), np.abs(full))

    def test_krakenc_uses_the_full_complex_eigenvalue(self, tmp_path):
        data = self._gamma(tmp_path, 'krakenc.exe', 'gc')
        bot = data.halfspace_tails['bottom']
        assert data.metadata['title'][:7] == 'KRAKENC'
        assert data.backend == 'krakenc'
        expected = np.sqrt(data.k ** 2 - bot['k2'])
        assert np.allclose(np.abs(bot['gamma']), np.abs(expected))


@pytest.mark.requires_binary
def test_a_rigid_floor_keeps_the_files_density_and_has_no_halfspace_density(
        tmp_path):
    """A rigid or vacuum boundary has no medium: the .mod stores rho = 0.0
    there (misc/ReadEnvironmentMod.f90:280). read_modes reports that value
    as the file holds it and gives MediaTable no half-space density, rather
    than a density of 1.0 the file does not hold."""
    deck = _isovelocity_deck('rig').replace("'A' 0.0\n  100.0 1800.0 0.0 "
                                            "1.8 0.5 /\n", "'R' 0.0\n")
    assert "'R' 0.0" in deck
    _run_kraken(tmp_path, 'rig', deck)
    data = read_modes(str(tmp_path / 'rig.mod'), frequency=100.0)
    bot = data.metadata['bottom_halfspace']
    assert bot['BC'] == 'R'
    assert bot['rho'] == 0.0
    # The deck's top is a vacuum, which has no medium either.
    assert data.metadata['top_halfspace']['BC'] == 'V'
    assert data.metadata['top_halfspace']['rho'] == 0.0
    assert data.media.halfspace_density is None
    # Neither boundary is an acoustic half-space, so neither has a tail.
    assert data.halfspace_tails == {}


@pytest.mark.requires_binary
def test_the_halfspace_tails_leave_metadata_for_their_attribute(tmp_path):
    """metadata keeps only the file's half-space records; the k2, gamma and
    phi read_modes computes (OALIB read_modes.m) are
    ``Modes.halfspace_tails``, sliced by first_n and kept through to_dict
    and NetCDF."""
    _run_kraken(tmp_path, 'tl', _isovelocity_deck('tl'))
    data = read_modes(str(tmp_path / 'tl.mod'), frequency=100.0)
    assert set(data.metadata['bottom_halfspace']) == {
        'BC', 'cp', 'cs', 'rho', 'depth'}
    assert set(data.halfspace_tails) == {'bottom'}
    bot = data.halfspace_tails['bottom']
    assert bot['k2'] == pytest.approx(
        (2 * np.pi * 100.0 / data.metadata['bottom_halfspace']['cp']) ** 2)
    np.testing.assert_array_equal(bot['phi'], data.phi[-1, :])
    assert bot['gamma'].shape == data.k.shape
    assert np.all(np.real(bot['gamma']) >= 0.0)
    two = data.first_n(2)
    np.testing.assert_array_equal(two.halfspace_tails['bottom']['gamma'],
                                  bot['gamma'][:2])
    for back in (Modes.from_dict(data.to_dict()),
                 Modes.from_netcdf(_netcdf_round_trip(data, tmp_path))):
        assert set(back.halfspace_tails) == {'bottom'}
        assert back.halfspace_tails['bottom']['k2'] == pytest.approx(
            bot['k2'])
        np.testing.assert_allclose(back.halfspace_tails['bottom']['gamma'],
                                   bot['gamma'])
        np.testing.assert_allclose(back.halfspace_tails['bottom']['phi'],
                                   bot['phi'])


def _netcdf_round_trip(modes, tmp_path):
    path = tmp_path / 'modes.nc'
    modes.to_netcdf(path)
    return path


def test_a_missing_mod_file_is_named_with_its_resolved_extension(tmp_path):
    """The ``.mod`` extension is appended before the file is looked for, so
    the typed error names the path that was actually opened."""
    from uacpy.core.exceptions import MissingFileError
    with pytest.raises(MissingFileError, match=r'not found.*absent\.mod'):
        _read_modes_payload(str(tmp_path / 'absent'))
    with pytest.raises(MissingFileError, match=r'not found.*absent\.mod'):
        _read_modes_payload(str(tmp_path / 'absent.mod'))
    # A suffix-less Path resolves the same way as its string, through the
    # public reader too.
    with pytest.raises(MissingFileError, match=r'not found.*absent\.mod'):
        _read_modes_payload(tmp_path / 'absent')
    with pytest.raises(MissingFileError, match=r'not found.*absent\.mod'):
        read_modes(tmp_path / 'absent')


def test_a_truncated_mod_file_is_a_typed_parse_error(tmp_path):
    """A record read that runs off the end of the file surfaces as the
    ``typed_format_error`` ``FileFormatError``, never a bare ``struct.error``
    or ``IndexError``."""
    path = tmp_path / 'short.mod'
    path.write_bytes(struct.pack('<i', 32) + b'\x00' * 20)
    with pytest.raises(FileFormatError, match='Invalid mode file'):
        _read_modes_payload(str(path))


@pytest.mark.requires_binary
class TestAMultiFrequencyModeFileNeedsAFrequency:
    """``read_modes`` with ``frequency=None`` reads a single-frequency file
    and refuses to pick a frequency of a broadband one (TopOpt(6) = 'B',
    ``Kraken/kraken.f90:52``), naming what it holds."""

    @staticmethod
    def _broadband(tmp_path):
        deck = _isovelocity_deck('BB').replace("'NVW'", "'NVW  B'")
        _run_kraken(tmp_path, 'bb', deck + '3\n  50.0 100.0 150.0 /\n')
        return tmp_path / 'bb.mod'

    def test_none_on_a_broadband_file_names_its_frequencies(self, tmp_path):
        mod = self._broadband(tmp_path)
        with pytest.raises(ConfigurationError, match='50, 100, 150 Hz'):
            read_modes(str(mod))
        assert read_modes(str(mod), frequency=100.0).frequencies.tolist() == [
            pytest.approx(100.0)]

    def test_none_on_a_single_frequency_file_reads_it(self, tmp_path):
        _run_kraken(tmp_path, 'nb', _isovelocity_deck('NB'))
        data = read_modes(str(tmp_path / 'nb.mod'))
        assert data.frequencies.tolist() == [pytest.approx(100.0)]
        assert data.n_modes > 0


@pytest.mark.requires_binary
class TestAModeFileReadsAsTheModesCarrier:
    """``read_modes`` returns the ``Modes`` carrier ``Kraken`` returns, so a
    ``.mod`` from any run reads into the package's mode tools directly."""

    def test_the_carrier_holds_the_files_modes(self, tmp_path):
        from uacpy.core.results import Modes
        _run_kraken(tmp_path, 'nb', _isovelocity_deck('NB'))
        raw = _read_modes_payload(str(tmp_path / 'nb.mod'))
        modes = read_modes(str(tmp_path / 'nb.mod'), water_density=1.0)
        assert isinstance(modes, Modes)
        np.testing.assert_array_equal(modes.k, raw['k'])
        np.testing.assert_array_equal(modes.phi, raw['phi'])
        np.testing.assert_array_equal(modes.depths, raw['z'])
        assert modes.frequencies.tolist() == [100.0]
        assert modes.backend == 'kraken' and modes.model == ''
        assert modes.media.water_density == 1.0
        assert modes.first_n(2).n_modes == 2

    def test_kraken_run_and_a_re_read_of_its_file_agree(self, tmp_path):
        env = uacpy.Environment(bathymetry=100.0, ssp=1500.0,
                                bottom=uacpy.BoundaryProperties(
                                    acoustic_type='half-space',
                                    sound_speed=1800.0, density=1.8,
                                    attenuation=0.5))
        kraken = uacpy.Kraken(work_dir=tmp_path, cleanup=False)
        run = kraken.compute_modes(env, uacpy.Source(depths=50.0,
                                                     frequencies=100.0))
        again = read_modes(run.metadata['mod_file'])
        np.testing.assert_array_equal(again.k, run.k)
        np.testing.assert_array_equal(again.phi, run.phi)


class TestTheSeabedAttenuationLeavesTheVolumeLawOut:
    """``_seabed_attenuation`` inverts CRCI's ``cp = c + i·alphaT·c²/ω``
    (``misc/AttenMod.f90:114``) and takes the deck's volume law out of
    ``alphaT``: the seabed's own dB/wavelength, 0 for a lossless seabed whose
    record carries the law alone to single precision, ``None`` when the
    deck is not the record's or names a law it does not know."""

    @staticmethod
    def _cp(seabed_dB_per_wavelength, frequency, volume_np_per_m, c=1800.0):
        from uacpy.core.deck_limits import AT_DB_PER_NEPER
        omega = 2.0 * np.pi * frequency
        alpha = (seabed_dB_per_wavelength * frequency / (AT_DB_PER_NEPER * c)
                 + volume_np_per_m)
        return complex(c, alpha * c ** 2 / omega)

    @staticmethod
    def _thorp_np(frequency):
        from uacpy.acoustics import absorption_thorp
        from uacpy.core.deck_limits import AT_DB_PER_NEPER
        return float(absorption_thorp(frequency)) / (AT_DB_PER_NEPER * 1000.0)

    def test_thorp_is_taken_out(self):
        from uacpy.io.modes_reader import _seabed_attenuation
        cp = self._cp(0.5, 4000.0, self._thorp_np(4000.0))
        assert _seabed_attenuation(cp, 4000.0, ('T', None)) == pytest.approx(
            0.5, rel=1e-12)
        assert _seabed_attenuation(cp, 4000.0, (' ', None)) > 0.5

    @pytest.mark.parametrize('short, expected', [(1e-8, 0.0), (1e-5, None)])
    def test_a_record_short_of_the_law_is_zero_or_unknown(self, short,
                                                          expected):
        from uacpy.io.modes_reader import _seabed_attenuation
        cp = self._cp(0.0, 4000.0, self._thorp_np(4000.0) * (1.0 - short))
        assert _seabed_attenuation(cp, 4000.0, ('T', None)) == expected

    @pytest.mark.parametrize('law', [None, ('X', None)])
    def test_an_unknown_law_leaves_it_unknown(self, law):
        from uacpy.io.modes_reader import _seabed_attenuation
        assert _seabed_attenuation(self._cp(0.5, 400.0, 0.0), 400.0,
                                   law) is None


@pytest.mark.requires_binary
class TestTheModFileGivesTheSeabedAttenuation:
    """``read_modes`` inverts the bottom record's complex speed
    ``cp = c + i·α·c²/ω`` (``misc/AttenMod.f90:114``, α in Np/m) to the
    dB/wavelength the deck was given, on kraken and krakenc; the record is
    single precision, so it agrees to about 2e-8. The volume law the deck
    beside the ``.mod`` adds to the half-space (``misc/sspMod.f90:437``) is
    taken out in closed form; without that deck the share is unknown."""

    @pytest.mark.parametrize('leaky_modes', [False, True])
    @pytest.mark.parametrize('attenuation', [0.0, 0.1, 0.5, 0.8])
    def test_it_is_the_attenuation_the_deck_was_given(self, attenuation,
                                                      leaky_modes):
        env = Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=attenuation))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            modes = Kraken(verbose=False, leaky_modes=leaky_modes
                           ).compute_modes(env, Source(depths=36.0,
                                                       frequencies=50.0))
        assert modes.media.halfspace_attenuation == pytest.approx(
            attenuation, rel=1e-7, abs=1e-12)

    @staticmethod
    def _env(absorption, attenuation=0.5):
        from uacpy.core.absorption import Thorp, FrancoisGarrison, Biological
        law = {'none': None, 'thorp': Thorp(),
               'francois_garrison': FrancoisGarrison(10.0, 35.0, 8.0),
               'biological': Biological(
                   layers=[(10.0, 50.0, 3000.0, 5.0, 2.0)])}[absorption]
        return Environment(
            bathymetry=100.0, ssp=1500.0, absorption=law,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=attenuation))

    def _modes(self, absorption, attenuation=0.5, **kraken):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return Kraken(verbose=False, **kraken).compute_modes(
                self._env(absorption, attenuation),
                Source(depths=36.0, frequencies=4000.0))

    @pytest.mark.parametrize('absorption', [
        'none', 'thorp', 'francois_garrison', 'biological'])
    def test_the_volume_law_is_taken_out_of_the_record(self, absorption):
        """Thorp is added to the half-space's attenuation too: its record
        holds 0.500127 dB/λ at 4 kHz, 1.27e-4 of it the deck's Thorp
        (``AttenMod.f90:86-94``). One-frequency Francois-Garrison is a table
        on the water rows and a biological layer acts only between its
        depths, so neither reaches it."""
        media = self._modes(absorption).media
        assert media.halfspace_sound_speed == 1800.0
        assert media.halfspace_attenuation == pytest.approx(0.5, rel=1e-7)

    def test_a_lossless_seabed_under_thorp_reads_zero(self):
        """The record's single precision leaves Im(cp) - Thorp a hair either
        side of zero; the seabed's own is 0."""
        media = self._modes('thorp', attenuation=0.0).media
        assert media.halfspace_attenuation == pytest.approx(0.0, abs=1e-9)
        assert media.halfspace_attenuation >= 0.0

    def test_a_broadband_francois_garrison_deck_is_taken_out_too(
            self, tmp_path):
        """One Francois-Garrison row on a broadband deck is TopOpt(4) 'F',
        added to the half-space with the row's T, S, pH and z_bar
        (``AttenMod.f90:127-171``)."""
        from uacpy.io import read_modes
        frequencies = np.array([2000.0, 3000.0, 4000.0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Kraken(verbose=False, work_dir=tmp_path, cleanup=False).run(
                self._env('francois_garrison'),
                Source(depths=36.0, frequencies=frequencies),
                Receiver(depths=[30.0], ranges=[1000.0, 2000.0]),
                run_mode=RunMode.BROADBAND, frequencies=frequencies)
            deck = next(tmp_path.rglob('kfield.env')).read_text()
            assert deck.splitlines()[3].strip("'")[3] == 'F'
            stored = [read_modes(next(tmp_path.rglob('kfield.mod')),
                                 frequency=f).media.halfspace_attenuation
                      for f in frequencies]
        np.testing.assert_allclose(stored, 0.5, rtol=1e-7)

    def test_a_deck_that_is_not_the_runs_leaves_it_unknown(self, tmp_path):
        """A lossless seabed's .mod read beside a Thorp deck has less in its
        record than the deck's Thorp alone: the deck is not the run's."""
        import shutil
        from uacpy.io import read_modes
        self._modes('none', attenuation=0.0, work_dir=tmp_path / 'run',
                    cleanup=False)
        self._modes('thorp', work_dir=tmp_path / 'other', cleanup=False)
        shutil.copy(next((tmp_path / 'run').rglob('modes.mod')),
                    next((tmp_path / 'other').rglob('modes.mod')))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            media = read_modes(
                next((tmp_path / 'other').rglob('modes.mod'))).media
        assert media.halfspace_attenuation is None

    def test_a_mod_read_without_its_deck_leaves_it_unknown(self, tmp_path):
        import shutil
        from uacpy.io import read_modes
        self._modes('none', work_dir=tmp_path, cleanup=False)
        mod = next(tmp_path.rglob('modes.mod'))
        alone = tmp_path / 'alone'
        alone.mkdir()
        shutil.copy(mod, alone / 'modes.mod')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            beside = read_modes(mod).media
            without = read_modes(alone / 'modes.mod').media
        assert beside.halfspace_attenuation == pytest.approx(0.5, rel=1e-7)
        assert without.halfspace_attenuation is None
        assert without.halfspace_sound_speed == 1800.0
