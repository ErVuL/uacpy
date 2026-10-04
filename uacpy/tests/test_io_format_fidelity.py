"""Format-fidelity contracts of the io layer.

Each class pins one reader/writer behaviour against the vendored Fortran's
own read semantics: list-directed whole-vector READs span records
(``Bellhop/sspMod.f90:417,428``, ``misc/RefCoef.f90:53``), fscanf-style
token streams ignore line breaks (``Matlab/ReadWrite/read_ts.m``), and the
engines' SELECT CASE option parsing is case-sensitive
(``Bellhop/bdryMod.f90:162-165``).
"""

import struct

import numpy as np
import pytest

from uacpy.core.exceptions import (
    ConfigurationError, FileFormatError, UnsupportedFeatureError,
)


class TestExplicitPathsAreNeverShadowed:
    """``read_bathymetry``/``read_altimetry`` read the path they are given;
    the conventional suffix is only a fallback for extensionless roots."""

    def test_explicit_extension_wins_over_sibling_bty(self, tmp_path):
        from uacpy.io.bathy_io import read_bathymetry
        (tmp_path / 'survey.dat').write_text("'L'\n2\n0.0 100.0\n1.0 200.0\n")
        (tmp_path / 'survey.bty').write_text("'L'\n2\n0.0 999.0\n1.0 999.0\n")
        bty = read_bathymetry(tmp_path / 'survey.dat')
        assert bty.depths.tolist() == [100.0, 200.0]

    def test_extensionless_root_resolves_to_bty(self, tmp_path):
        from uacpy.io.bathy_io import read_bathymetry
        (tmp_path / 'survey.bty').write_text("'L'\n2\n0.0 50.0\n1.0 60.0\n")
        bty = read_bathymetry(tmp_path / 'survey')
        assert bty.depths.tolist() == [50.0, 60.0]

    def test_a_missing_explicit_path_is_a_configuration_error(self, tmp_path):
        """A .bty is a deck the caller wrote, so an absent one is a bad
        argument — the provenance split core/exceptions.py states."""
        from uacpy.io.bathy_io import read_bathymetry
        with pytest.raises(ConfigurationError,
                           match='Bathymetry file not found'):
            read_bathymetry(tmp_path / 'absent.dat')


class TestReflectionTableIsOneListDirectedRead:
    """``misc/RefCoef.f90:53`` reads the whole (theta, R, phi) table with a
    single list-directed READ, so records may pack several per line or wrap
    across lines; both the reader and the staged-copy dedupe must accept
    that layout without dropping points."""

    TABLE = ("4\n"
             "0.0 1.0 180.0  10.0 0.9 170.0\n"
             "20.0 0.8 160.0  30.0 0.7 150.0\n")

    def test_reader_takes_packed_records(self, tmp_path):
        from uacpy.io.refl_io import read_reflection_coefficient
        p = tmp_path / 'multi.brc'
        p.write_text(self.TABLE)
        d = read_reflection_coefficient(p)
        assert len(d.angles) == 4
        assert d.angles.tolist() == [0.0, 10.0, 20.0, 30.0]
        assert d.magnitude.tolist() == [1.0, 0.9, 0.8, 0.7]

    def test_reader_takes_wrapped_records(self, tmp_path):
        from uacpy.io.refl_io import read_reflection_coefficient
        p = tmp_path / 'wrap.brc'
        p.write_text("2\n0.0 1.0\n180.0\n10.0 0.9 170.0\n")
        d = read_reflection_coefficient(p)
        assert d.angles.tolist() == [0.0, 10.0]

    def test_dedupe_keeps_every_packed_record(self, tmp_path):
        from uacpy.io.refl_io import (
            dedupe_reflection_file, read_reflection_coefficient)
        p = tmp_path / 'staged.brc'
        p.write_text(self.TABLE)
        dedupe_reflection_file(p)
        d = read_reflection_coefficient(p)
        assert len(d.angles) == 4, "dedupe dropped legally packed records"
        assert d.angles.tolist() == [0.0, 10.0, 20.0, 30.0]

    def test_dedupe_drops_the_evanescent_head(self, tmp_path):
        from uacpy.io.refl_io import (
            dedupe_reflection_file, read_reflection_coefficient)
        p = tmp_path / 'evan.brc'
        p.write_text("4\n0.0 1.0 180.0  0.0 1.0 180.0\n"
                     "0.0 1.0 180.0  10.0 0.9 170.0\n")
        dedupe_reflection_file(p)
        d = read_reflection_coefficient(p)
        assert d.angles.tolist() == [0.0, 10.0]


class TestBoundaryFilesAreListDirected:
    """``bdryMod.f90:71/:171`` read the .ati/.bty counts and ``:98/:195``
    each point with list-directed READs, so an annotated count line and rows
    wrapped across lines are files bellhop.exe runs; ``beampattern.f90:33,44``
    give the .sbp the same semantics, blank lines included. The readers must
    accept every file the binary accepts."""

    def test_bty_annotated_count_and_wrapped_rows(self, tmp_path):
        from uacpy.io.bathy_io import read_bathymetry
        p = tmp_path / 'wrap.bty'
        p.write_text("'L'\n"
                     "3   ! number of bathymetry points\n"
                     "0.0\n100.0\n"
                     "1.0 150.0\n"
                     "2.0\n120.0\n")
        bty = read_bathymetry(p)
        assert bty.interpolation == 'L'
        assert bty.ranges.tolist() == [0.0, 1000.0, 2000.0]
        assert bty.depths.tolist() == [100.0, 150.0, 120.0]

    def test_bty_row_remainder_is_discarded(self, tmp_path):
        """Each point is ONE list-directed READ: tokens past the row's
        n_cols on its final record are the remainder the READ skips."""
        from uacpy.io.bathy_io import read_bathymetry
        p = tmp_path / 'rem.bty'
        p.write_text("'L'\n2\n0.0 100.0 ignored garbage\n1.0 150.0\n")
        bty = read_bathymetry(p)
        assert bty.depths.tolist() == [100.0, 150.0]

    def test_ati_comma_count_line(self, tmp_path):
        from uacpy.io.bathy_io import read_altimetry
        p = tmp_path / 'c.ati'
        p.write_text("'L'\n2, ! npts\n0.0 0.0\n1.0 -2.0\n")
        ati = read_altimetry(p)
        assert ati.depths.tolist() == [0.0, -2.0]

    def test_brc_comma_count_line(self, tmp_path):
        from uacpy.io.refl_io import (
            dedupe_reflection_file, read_reflection_coefficient)
        p = tmp_path / 'c.brc'
        p.write_text("2, ! npts\n0.0 1.0 180.0\n10.0 0.9 170.0\n")
        d = read_reflection_coefficient(p)
        assert len(d.angles) == 2
        assert d.angles.tolist() == [0.0, 10.0]
        # The staging dedupe accepts the same count record the reader does.
        dedupe_reflection_file(p)
        assert len(read_reflection_coefficient(p).angles) == 2

    def test_sbp_blank_lines_and_wrapped_pairs(self, tmp_path):
        from uacpy.io.refl_io import read_source_beam_pattern
        p = tmp_path / 'w.sbp'
        p.write_text("3   ! NSBPPts\n"
                     "-45.0\n-10.0\n"
                     "\n"
                     "0.0 0.0\n"
                     "45.0\n\n-10.0\n")
        pat = read_source_beam_pattern(p)
        assert pat.tolist() == [[-45.0, -10.0], [0.0, 0.0], [45.0, -10.0]]

    def test_truncated_sbp_is_a_typed_error(self, tmp_path):
        from uacpy.io.refl_io import read_source_beam_pattern
        p = tmp_path / 't.sbp'
        p.write_text("2\n-45.0 0.0\n45.0\n")
        with pytest.raises(FileFormatError, match='file ended'):
            read_source_beam_pattern(p)


class TestRepeatCountsReadAsRepeatedValues:
    """A list-directed READ accepts ``r*c`` as ``r`` copies of the constant
    ``c``. gfortran's own list-directed WRITEs never emit the form, but
    ifort's do for consecutive equal values — and ``misc/RefCoef.f90:53``
    reads the .brc, ``Bellhop/ArrMod.f90:99-118`` writes the .arr, and
    SPARC's .rts payload is read as one token stream, so a file from an
    ifort-built engine can carry it. The readers must parse it to the
    values Fortran READ produces; malformed spellings stay typed errors."""

    def test_brc_repeat_count_reads_as_repeated_values(self, tmp_path):
        from uacpy.io.refl_io import read_reflection_coefficient
        p = tmp_path / 'rep.brc'
        # 2*0.0 covers theta and R of the first point (total absorption at
        # grazing), exactly what an ifort WRITE compresses.
        p.write_text("2\n2*0.0 180.0\n10.0 0.9 170.0\n")
        d = read_reflection_coefficient(p)
        assert d.angles.tolist() == [0.0, 10.0]
        assert d.magnitude.tolist() == [0.0, 0.9]

    def test_rts_repeat_count_reads_as_repeated_values(self, tmp_path):
        from uacpy.io.oalib_reader import read_rts_file
        p = tmp_path / 'rep.rts'
        p.write_text("'run'\n2 5000.0 10000.0\n0.0 2*0.5\n")
        d = read_rts_file(p)
        assert d.positions.tolist() == [5000.0, 10000.0]
        assert d.pressure.tolist() == [[0.5, 0.5]]

    def test_ts_repeat_count_reads_as_repeated_values(self, tmp_path):
        from uacpy.io.oalib_reader import read_ts
        p = tmp_path / 'rep.ts'
        p.write_text("t\n2 10.0 20.0\n0.0 2*1.5\n0.1 3.0 4.0\n")
        d = read_ts(p)
        assert d.pressure.tolist() == [[1.5, 1.5], [3.0, 4.0]]

    @pytest.mark.parametrize('token', [
        '2*',      # Fortran's null-value form: stands for values NOT assigned
        'x*0.5',   # no repeat count
        '2*junk',  # constant is not a number
    ])
    def test_a_malformed_repeat_spelling_is_a_typed_error(
            self, token, tmp_path):
        from uacpy.io.refl_io import read_reflection_coefficient
        p = tmp_path / 'bad.brc'
        p.write_text(f"2\n0.0 1.0 180.0\n10.0 {token} 170.0\n")
        with pytest.raises(FileFormatError, match='could not parse'):
            read_reflection_coefficient(p)

    def test_many_small_repeat_groups_stop_at_the_expansion_ceiling(self):
        """A repeat count is a compression device, so a short record can ask
        for an enormous stream: each group here sits under the per-token
        ceiling while their sum runs far past it. The running total is what
        bounds the memory a reader that materialises the stream spends on a
        tiny file."""
        from uacpy.io._fortran_helpers import (
            expand_repeat_counts, _MAX_GENERATED_VECTOR)
        under = _MAX_GENERATED_VECTOR // 2
        stream = expand_repeat_counts([f'{under}*1.0'] * 4)
        with pytest.raises(FileFormatError, match='ceiling'):
            for _ in stream:
                pass

    def test_expansion_under_the_ceiling_is_yielded_in_full(self):
        from uacpy.io._fortran_helpers import expand_repeat_counts
        assert list(expand_repeat_counts(['3*1.5', '2.0'])) == [
            '1.5', '1.5', '1.5', '2.0']

    def test_ray_counts_record_reads_a_repeat_count(self, tmp_path):
        """``WriteRay.f90:41-46`` writes every ray record list-directed, and
        the counts record is ``N2, NumTopBnc, NumBotBnc`` — ``0 0`` for every
        direct path, which is exactly what a writer compresses to ``2*0``."""
        from uacpy.io.oalib_reader import read_ray_file
        p = tmp_path / 'r.ray'
        p.write_text(
            "'title'\n50.0\n1 1 1\n1 1\n0.0\n100.0\n'rz'\n"
            "-15.0\n3 2*0\n0.0 10.0\n500.0 40.0\n1000.0 10.0\n")
        one = read_ray_file(p).rays[0]
        assert one['n_top_bounces'] == 0 and one['n_bot_bounces'] == 0
        assert len(one['r']) == 3

    def test_dedupe_accepts_the_repeat_counts_the_reader_accepts(
            self, tmp_path):
        """``dedupe_reflection_file`` and ``read_reflection_coefficient``
        parse the same table under the same ``RefCoef.f90:53`` list-directed
        READ, so a ``.brc`` one accepts cannot be truncated to the other."""
        from uacpy.io.refl_io import (dedupe_reflection_file,
                                      read_reflection_coefficient)
        p = tmp_path / 'd.brc'
        p.write_text("3\n0.0 1.0 180.0\n10.0 2*0.5\n20.0 0.25 90.0\n")
        assert len(read_reflection_coefficient(p).angles) == 3
        dedupe_reflection_file(p)
        assert len(read_reflection_coefficient(p).angles) == 3


class TestSspRecordsSpanLines:
    """``Bellhop/sspMod.f90:417,428`` read the range vector and each depth
    row with whole-vector list-directed READs, which consume as many lines
    as they need and discard the remainder of their final line."""

    def test_2d_rows_may_wrap(self, tmp_path):
        from uacpy.io.oalib_reader import read_ssp_2d
        p = tmp_path / 'w.ssp'
        p.write_text("3\n0.0 5.0\n10.0\n1500 1501\n1502\n1490 1491 1492\n")
        d = read_ssp_2d(p)
        assert d.ranges.tolist() == [0.0, 5000.0, 10000.0]
        assert d.sound_speed.tolist() == [[1500, 1501, 1502],
                                          [1490, 1491, 1492]]

    def test_2d_row_remainder_is_discarded(self, tmp_path):
        from uacpy.io.oalib_reader import read_ssp_2d
        p = tmp_path / 'p.ssp'
        p.write_text("2\n0.0 5.0\n1500 1501 9e9 9e9\n1490 1491\n")
        d = read_ssp_2d(p)
        assert d.sound_speed.tolist() == [[1500, 1501], [1490, 1491]]

    def test_truncated_row_is_a_typed_error(self, tmp_path):
        from uacpy.io.oalib_reader import read_ssp_2d
        p = tmp_path / 't.ssp'
        p.write_text("3\n0.0 5.0 10.0\n1500 1501\n")
        with pytest.raises(FileFormatError,
                           match='file ended while reading SSP depth row'):
            read_ssp_2d(p)


class TestReadTsTokenStream:
    """``read_ts.m`` reads everything after the title with ``fscanf``, a
    free token stream in which line breaks carry no meaning."""

    def test_wrapped_and_packed_stream(self, tmp_path):
        from uacpy.io.oalib_reader import read_ts
        p = tmp_path / 't.ts'
        p.write_text("my title\n3 10.0\n20.0 30.0\n"
                     "0.0 1 2 3 0.1 4\n5 6\n0.2 7 8 9\n")
        t = read_ts(p)
        assert t.title == 'my title'
        assert t.positions.tolist() == [10.0, 20.0, 30.0]
        assert t.times.tolist() == [0.0, 0.1, 0.2]
        assert t.pressure.tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def test_partial_trailing_block_is_dropped(self, tmp_path):
        from uacpy.io.oalib_reader import read_ts
        p = tmp_path / 't.ts'
        p.write_text("t\n2 10.0 20.0\n0.0 1 2 0.1 3\n")
        t = read_ts(p)
        assert t.times.tolist() == [0.0]

    def test_empty_payload_is_a_typed_error(self, tmp_path):
        from uacpy.io.oalib_reader import read_ts
        p = tmp_path / 't.ts'
        p.write_text("title only\n")
        with pytest.raises(FileFormatError,
                           match='carries no data after the title line'):
            read_ts(p)

    def test_mat_container_is_a_typed_error(self, tmp_path):
        """read_ts parses the ASCII token-stream format only; a .mat path is
        refused with a typed error rather than parsed by guesswork."""
        from uacpy.io.oalib_reader import read_ts
        p = tmp_path / 'ts.mat'
        p.write_bytes(b'MATLAB 5.0 MAT-file' + b'\x00' * 32)
        with pytest.raises(FileFormatError, match=r'\.mat'):
            read_ts(p)


class TestSbpAngleResolution:
    """``.sbp`` angles are written at %.6f, and angles closer than that
    resolution are refused: ``misc/beampattern.f90:56`` aborts the engine on
    a repeated (non-strictly-increasing) angle."""

    def test_fine_pattern_round_trips(self, tmp_path):
        from uacpy.io.refl_io import (
            read_source_beam_pattern, write_source_beam_pattern)
        angles = np.array([-1.0, -0.001, 0.0, 0.001, 1.0])
        p = tmp_path / 'fine.sbp'
        write_source_beam_pattern(p, angles, np.zeros(5))
        back = read_source_beam_pattern(p)
        assert np.array_equal(back[:, 0], angles), \
            "distinct angles collided on the file's angle grid"

    def test_sub_resolution_step_is_refused(self, tmp_path):
        from uacpy.io.refl_io import write_source_beam_pattern
        with pytest.raises(ConfigurationError, match='strictly increasing'):
            write_source_beam_pattern(tmp_path / 'x.sbp',
                                      np.array([0.0, 1e-8, 1.0]), np.zeros(3))


def _plp_tl_pair(tmp_path, curves, n_ranges=4, stem='r'):
    """Write a ``.plp``/``.plt`` pair holding ``curves`` TL-vs-range records.

    ``curves`` is a list of ``(freq, rd, xoff, dx)``; ``freq``/``rd`` go into
    the ``Freq:``/``RD:`` A16 labels PLTLOS writes ahead of the axis block
    (``oasfun22.f:334-337``, ``:368-370``), which is what the reader keys the
    ``(frequency, depth)`` grid off. Curve ``k`` holds the values
    ``k*100 + i``. Pass ``freq=None`` for a label-less block, the shape a
    ``.plp`` written by something other than PLTLOS has.
    """
    def rec(value, label):
        return f"{value:<19}{label}"

    lines = [' OAST  MODU']
    blocks = []
    for curve_id, (freq, rd, xoff, dx) in enumerate(curves):
        lines += [' OAST  NTLRAN', 'ptit', 'title']
        if freq is None:
            lines += [rec(0, 'NUMBER OF LABELS')]
        else:
            lines += [rec(3, 'NUMBER OF LABELS'),
                      f" Freq:{freq:7.1f} Hz$",
                      f" SD:{50.0:9.1f} m$",
                      f" RD:{rd:9.1f} m$"]
        lines += [rec(0.0, name) for name in
                  ('XLEN', 'YLEN', 'IGRID', 'XLEFT', 'XRIGHT', 'XINC',
                   'XDIV', 'XTXT', 'XTYP', 'YDOWN', 'YUP', 'YINC',
                   'YDIV', 'YTXT', 'YTYP')]
        lines += [rec(1, 'NC'),
                  rec(n_ranges, 'N'), rec(xoff, 'XOFF'), rec(dx, 'DX'),
                  rec(0.0, 'YOFF'), rec(0.0, 'DY')]
        blocks.append('\n'.join(
            f' {curve_id * 100 + i}.0' for i in range(n_ranges)))
    lines += [' OAST  PLTEND']
    (tmp_path / f'{stem}.plp').write_text('\n'.join(lines) + '\n')
    (tmp_path / f'{stem}.plt').write_text('\n\n'.join(blocks) + '\n\n')
    return tmp_path / f'{stem}.plp'


class TestOastTlMultiFrequency:
    """OAST writes one TL curve per plotted receiver *per frequency*, and
    each curve carries its own ``Freq:``/``RD:`` labels (oasfun22.f:334-337);
    the reader returns an ``(n_freq, n_depths, n_ranges)`` stack for
    NFREQ > 1."""

    @staticmethod
    def _write_pair(tmp_path, n_freq, n_depths, n_ranges=4):
        # Frequency-major, the order the NFREQ loop (unoast31.f:388) wrapping
        # the receiver loop (:584) produces.
        curves = [(100.0 * (i_f + 1), 10.0 * (i_d + 1), 1.0, 0.5)
                  for i_f in range(n_freq) for i_d in range(n_depths)]
        return _plp_tl_pair(tmp_path, curves, n_ranges=n_ranges)

    def test_single_frequency_keeps_2d_shape(self, tmp_path):
        from uacpy.io._parsers import parse_oast_tl
        plp = self._write_pair(tmp_path, n_freq=1, n_depths=2)
        out = parse_oast_tl(plp, [10.0, 20.0])
        assert out['tl'].shape == (2, 4)
        assert out['metadata']['n_frequencies'] == 1
        assert out['ranges'].tolist() == [1000.0, 1500.0, 2000.0, 2500.0]
        assert out['depths'].tolist() == [10.0, 20.0]

    def test_multi_frequency_returns_a_stack(self, tmp_path):
        from uacpy.io._parsers import parse_oast_tl
        plp = self._write_pair(tmp_path, n_freq=3, n_depths=2)
        out = parse_oast_tl(plp, [10.0, 20.0])
        assert out['tl'].shape == (3, 2, 4)
        assert out['metadata']['n_frequencies'] == 3
        # Frequency-major order: curve (ifreq, idepth) = ifreq*2 + idepth.
        assert out['tl'][1, 1, 0] == 300.0
        assert out['tl'][2, 0, 0] == 400.0
        assert out['metadata']['plotted_frequencies'].tolist() == [
            100.0, 200.0, 300.0]

    def test_non_multiple_curve_count_is_a_typed_error(self, tmp_path):
        """A label-less ``.plp`` falls back to the positional walk, which is
        the one that can only check the count."""
        from uacpy.io._parsers import parse_oast_tl
        curves = [(None, None, 1.0, 0.5)] * 3
        plp = _plp_tl_pair(tmp_path, curves)
        with pytest.raises(FileFormatError, match='whole multiple'):
            parse_oast_tl(plp, [10.0, 20.0])

    def test_a_label_less_plp_warns_before_walking_by_position(self, tmp_path):
        """Nothing but PLTLOS writes a ``…TLRAN`` curve, so a block without
        the labels came from elsewhere; the positional walk still reads it,
        but it is the reading ``IDINC`` breaks."""
        from uacpy.io._parsers import parse_oast_tl
        plp = _plp_tl_pair(tmp_path, [(None, None, 1.0, 0.5)] * 2)
        with pytest.warns(UserWarning, match='by position'):
            out = parse_oast_tl(plp, [10.0, 20.0])
        assert out['tl'].shape == (2, 4)
        assert out['depths'].tolist() == [10.0, 20.0]
        assert 'plotted_frequencies' not in out['metadata']


class TestOastTlIsAlwaysAFrequencyStack:
    """``read_oast_tl`` returns a ResultStack over frequency whatever the
    file holds — length 1 for one frequency — with each slab on its own
    native range grid."""

    def test_one_frequency_is_a_stack_of_one(self, tmp_path):
        from uacpy.core.results import Field, ResultStack
        from uacpy.io import read_oast_tl
        plp = _plp_tl_pair(tmp_path, [(400.0, 10.0, 1.0, 0.5),
                                      (400.0, 20.0, 1.0, 0.5)])
        stack = read_oast_tl(plp, [10.0, 20.0])
        assert type(stack) is ResultStack and len(stack) == 1
        assert stack.coordinate_name == 'frequency'
        assert stack.coordinate.tolist() == [400.0]
        slab = stack[0]
        assert type(slab) is Field
        assert slab.data.shape == (2, 4)
        assert slab.coords['range'].tolist() == [1000.0, 1500.0, 2000.0,
                                                 2500.0]
        assert float(slab.frequencies[0]) == 400.0

    def test_n_frequencies_keep_their_own_range_grids(self, tmp_path):
        from uacpy.io import read_oast_tl
        curves = [(400.0, 10.0, 0.00336845, 0.00336845),
                  (800.0, 10.0, 0.00168423, 0.00168423)]
        plp = _plp_tl_pair(tmp_path, curves, n_ranges=512)
        stack = read_oast_tl(plp, [10.0])
        assert len(stack) == 2
        assert stack.coordinate.tolist() == [400.0, 800.0]
        assert stack[0].coords['range'][-1] == pytest.approx(1724.6464)
        assert stack[1].coords['range'][-1] == pytest.approx(862.32576)
        assert stack.at(frequency=800.0) is stack[1]

    def test_a_label_less_file_needs_its_frequencies(self, tmp_path):
        from uacpy.core.exceptions import FileFormatError
        from uacpy.io import read_oast_tl
        plp = _plp_tl_pair(tmp_path, [(None, None, 1.0, 0.5)] * 2)
        with pytest.warns(UserWarning, match='by position'), \
                pytest.raises(FileFormatError, match="no 'Freq:' labels"):
            read_oast_tl(plp, [10.0, 20.0])

    def test_a_label_less_file_takes_the_frequencies_passed(self, tmp_path):
        from uacpy.io import read_oast_tl
        plp = _plp_tl_pair(tmp_path, [(None, None, 1.0, 0.5)] * 2)
        with pytest.warns(UserWarning, match='by position'):
            stack = read_oast_tl(plp, [10.0, 20.0], frequencies=[250.0])
        assert stack.coordinate.tolist() == [250.0]
        assert float(stack[0].frequencies[0]) == 250.0

    def test_frequencies_one_per_block(self, tmp_path):
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.io import read_oast_tl
        plp = _plp_tl_pair(tmp_path, [(None, None, 1.0, 0.5)] * 2)
        with pytest.warns(UserWarning, match='by position'), \
                pytest.raises(ConfigurationError, match='1 plotted'):
            read_oast_tl(plp, [10.0, 20.0], frequencies=[250.0, 500.0])

    def test_the_labels_win_and_a_disagreeing_frequency_is_refused(
            self, tmp_path):
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.io import read_oast_tl
        plp = _plp_tl_pair(tmp_path, [(400.0, 10.0, 1.0, 0.5),
                                      (400.0, 20.0, 1.0, 0.5)])
        # within the F7.1 label's half step: accepted, the label kept
        stack = read_oast_tl(plp, [10.0, 20.0], frequencies=[400.04])
        assert stack.coordinate.tolist() == [400.0]
        with pytest.raises(ConfigurationError, match="'Freq:' labels"):
            read_oast_tl(plp, [10.0, 20.0], frequencies=[400.06])


class TestOastCurvesAreKeyedByTheirLabels:
    """PLTLOS writes ``Freq:``/``SD:``/``RD:`` into every plot block
    (``oasfun22.f:334-337``) and runs only for ``MOD( NREC-1, INTF ) == 0``
    (``unoast31.f:630``), ``INTF`` being ``IDINC``, the 4th field of the
    OASTL receiver record (``oaseun31.f:1156``). A decimated run therefore
    writes fewer curves than the deck has receivers, and only the labels say
    which receiver each curve belongs to."""

    def test_decimated_receivers_land_on_the_depths_that_were_plotted(
            self, tmp_path):
        from uacpy.io._parsers import parse_oast_tl
        # 'RD1 RD2 4 2' over 10..90 m: OAST plots receivers 1 and 3 only.
        rd = np.linspace(10.0, 90.0, 4)
        curves = [(400.0, 10.0, 0.0, 0.25), (400.0, 63.3, 0.0, 0.25),
                  (800.0, 10.0, 0.0, 0.125), (800.0, 63.3, 0.0, 0.125)]
        plp = _plp_tl_pair(tmp_path, curves)
        out = parse_oast_tl(plp, rd)
        assert out['tl'].shape == (2, 2, 4)
        assert out['metadata']['n_frequencies'] == 2
        # The F9.1 label names the receiver to 0.1 m; the axis the caller
        # passed supplies the full-precision value.
        assert out['depths'] == pytest.approx([10.0, 190.0 / 3.0])
        assert out['metadata']['plotted_frequencies'].tolist() == [400.0, 800.0]

    def test_a_label_off_the_receiver_axis_is_returned_as_printed(
            self, tmp_path):
        from uacpy.io._parsers import parse_oast_tl
        curves = [(100.0, 33.3, 0.0, 0.25)]
        plp = _plp_tl_pair(tmp_path, curves)
        out = parse_oast_tl(plp, [10.0, 20.0])
        assert out['depths'].tolist() == [33.3]

    def test_colliding_labels_fall_back_to_position(self, tmp_path):
        """Two receivers closer than the F9.1 label quantum print the same
        ``RD:``. The grid stops factoring, and the positional walk — right
        for every run that does not decimate — takes over behind a warning."""
        from uacpy.io._parsers import parse_oast_tl
        curves = [(100.0, 10.0, 0.0, 0.25), (100.0, 10.0, 0.0, 0.25)]
        plp = _plp_tl_pair(tmp_path, curves)
        with pytest.warns(UserWarning, match='by position'):
            out = parse_oast_tl(plp, [10.00, 10.04])
        assert out['tl'].shape == (2, 4)
        assert out['depths'].tolist() == [10.0, 10.04]

    def test_without_receiver_depths_the_labels_are_the_depths(
            self, tmp_path):
        """The file names its own depths; the caller's axis only refines
        them to full precision."""
        from uacpy.io._parsers import parse_oast_tl
        curves = [(400.0, 10.0, 0.0, 0.25), (400.0, 63.3, 0.0, 0.25)]
        plp = _plp_tl_pair(tmp_path, curves)
        out = parse_oast_tl(plp)
        assert out['depths'].tolist() == [10.0, 63.3]
        assert out['tl'].shape == (2, 4)

    def test_a_label_less_plp_needs_receiver_depths(self, tmp_path):
        from uacpy.io._parsers import parse_oast_tl
        plp = _plp_tl_pair(tmp_path, [(None, None, 1.0, 0.5)] * 2)
        with pytest.raises(FileFormatError, match='receiver_depths='):
            parse_oast_tl(plp)

    def test_labels_that_neither_factor_nor_divide_are_a_typed_error(
            self, tmp_path):
        from uacpy.io._parsers import parse_oast_tl
        curves = [(400.0, 10.0, 0.0, 0.25), (400.0, 20.0, 0.0, 0.25),
                  (800.0, 10.0, 0.0, 0.25)]
        plp = _plp_tl_pair(tmp_path, curves)
        with pytest.raises(FileFormatError, match='not a whole multiple'):
            parse_oast_tl(plp, [10.0, 20.0])

    def test_a_grid_with_a_hole_is_a_typed_error(self, tmp_path):
        """The grid factors on its totals — 2 frequencies by 2 depths, 4
        curves — while one pair repeats and another is absent, which would
        otherwise return whatever ``np.empty`` allocated."""
        from uacpy.io._parsers import parse_oast_tl
        curves = [(400.0, 10.0, 0.0, 0.25), (400.0, 20.0, 0.0, 0.25),
                  (800.0, 10.0, 0.0, 0.25), (800.0, 10.0, 0.0, 0.25)]
        plp = _plp_tl_pair(tmp_path, curves)
        with pytest.raises(FileFormatError, match='no TL curve for'):
            parse_oast_tl(plp, [10.0, 20.0])


class TestOastRangeAxisIsPerFrequency:
    """``DLRAN = 2*pi/(NWVNO*DLWVNO)`` is recomputed inside the frequency
    loop (``unoast31.f:480``) with ``DLWVNO`` proportional to FREQ, so DX
    halves when the frequency doubles while ``LF`` stays pinned at NWVNO
    (``:491``) — equal point counts, different range axes."""

    def test_each_frequency_gets_its_own_axis(self, tmp_path):
        from uacpy.io._parsers import parse_oast_tl
        curves = [(400.0, 10.0, 0.00336845, 0.00336845),
                  (800.0, 10.0, 0.00168423, 0.00168423)]
        plp = _plp_tl_pair(tmp_path, curves, n_ranges=512)
        out = parse_oast_tl(plp, [10.0])
        assert out['ranges'].shape == (2, 512)
        assert out['ranges'][0, -1] == pytest.approx(1724.6464)
        assert out['ranges'][1, -1] == pytest.approx(862.32576)

    def test_one_frequency_returns_a_1d_axis(self, tmp_path):
        from uacpy.io._parsers import parse_oast_tl
        plp = _plp_tl_pair(tmp_path, [(400.0, 10.0, 1.0, 0.5)])
        out = parse_oast_tl(plp, [10.0])
        assert out['ranges'].shape == (4,)
        assert out['tl'].shape == (1, 4)

    def test_receivers_of_one_frequency_must_share_a_grid(self, tmp_path):
        from uacpy.io._parsers import parse_oast_tl
        curves = [(400.0, 10.0, 1.0, 0.5), (400.0, 20.0, 1.0, 0.25)]
        plp = _plp_tl_pair(tmp_path, curves)
        with pytest.raises(FileFormatError, match='do not share a range grid'):
            parse_oast_tl(plp, [10.0, 20.0])


class TestPlpCountsAreValidated:
    """``NLAB`` and ``NC`` are the file's own DO-loop bounds
    (``oasgun21.f:614``, ``:631``) and the ``.plp`` walk advances by them, so
    a negative count moves the cursor backwards and the walk never reaches
    ``PLTEND``."""

    @staticmethod
    def _truncated(tmp_path, n_lab):
        def rec(value, label):
            return f"{value:<19}{label}"
        lines = [' OAST  MODU', ' OAST  NTLRAN', 'ptit', 'title',
                 rec(n_lab, 'NUMBER OF LABELS')]
        p = tmp_path / 'bad.plp'
        p.write_text('\n'.join(lines) + '\n')
        return p

    def test_a_negative_label_count_is_a_typed_error(self, tmp_path):
        from uacpy.io.oases_reader import _parse_oast_plp
        with pytest.raises(FileFormatError, match='NLAB=-5'):
            _parse_oast_plp(self._truncated(tmp_path, -5))

    def test_a_label_count_past_the_end_is_a_typed_error(self, tmp_path):
        from uacpy.io.oases_reader import _parse_oast_plp
        with pytest.raises(FileFormatError, match='NLAB=9999'):
            _parse_oast_plp(self._truncated(tmp_path, 9999))


class TestOasnWhiteNoiseContract:
    """OASN adds ``10**(WNLEVDB/10)`` to every covariance diagonal with no
    dead band (oasnun22.f:228, :1157), so the writer's default must be a
    level whose linear power is nil, and 0.0 must mean a literal 0 dB."""

    @staticmethod
    def _deck(tmp_path, n_wavenumbers=None, **noise_fields):
        from uacpy import Environment, Receiver, Source
        from uacpy.core.boundary import BoundaryProperties
        from uacpy.io.oases_writer import OasnNoise, write_oasn_input
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=BoundaryProperties(
                              acoustic_type='half-space', sound_speed=1700.0,
                              density=1.8, attenuation=0.5))
        p = tmp_path / 'n.dat'
        write_oasn_input(p, env, Source(depths=10.0, frequencies=100.0),
                         Receiver(depths=[30.0, 50.0], ranges=[0.0]),
                         options='N J', n_wavenumbers=n_wavenumbers,
                         noise=OasnNoise(surface_level=70.0, **noise_fields))
        return p.read_text()

    def test_default_writes_numerically_nil_level(self, tmp_path):
        text = self._deck(tmp_path)
        assert '70.0 -200.0 0.0 0' in text

    def test_explicit_zero_writes_literal_zero_dB(self, tmp_path):
        text = self._deck(tmp_path, white_level=0.0)
        assert '70.0 0.0 0.0 0' in text

    def test_noise_band_total_is_bounded_preflight(self, tmp_path):
        with pytest.raises(ConfigurationError,
                           match='TOO MANY SAMPLING POINTS'):
            self._deck(tmp_path, n_wavenumbers=40000)


class TestOasesInterfaceIndexSpaces:
    """Each family's first-bottom-interface helper matches its own written
    deck for an isovelocity column: OASS collapses the water to one record
    (seafloor = deck layer 3), the OASP family writes one record per SSP
    row (seafloor = 2 + n_rows)."""

    @staticmethod
    def _env():
        from uacpy import Environment
        from uacpy.core.boundary import BoundaryProperties
        return Environment(bathymetry=100.0, ssp=1500.0,
                           bottom=BoundaryProperties(
                               acoustic_type='half-space', sound_speed=1700.0,
                               density=1.8, attenuation=0.5, roughness=0.5))

    def test_oass_deck_layer_3_is_the_seafloor(self, tmp_path):
        from uacpy import Receiver, Source
        from uacpy.io.oases_writer import (
            oass_bottom_interfaces, write_oass_input)
        env = self._env()
        first_bottom, _ = oass_bottom_interfaces(env)
        assert first_bottom == 3
        p = tmp_path / 'o.dat'
        write_oass_input(p, env, Source(depths=10.0, frequencies=100.0),
                         Receiver(depths=[50.0], ranges=[1000.0, 2000.0]),
                         options='r', interface=first_bottom,
                         correlation_length=100.0, spectral_exponent=1.9)
        lines = p.read_text().splitlines()
        n_layers = int(lines[3])
        # Deck layers start on the line after the count; layer `first_bottom`
        # is the seabed record and carries the -|RG| CL M scattering tail.
        seafloor_row = lines[3 + first_bottom].split()
        assert n_layers == 3
        assert float(seafloor_row[0]) == 100.0
        assert float(seafloor_row[6]) == -0.5
        assert float(seafloor_row[7]) == 100.0

    def test_oassp_deck_numbering_matches_its_own_layout(self, tmp_path):
        from uacpy import Receiver, Source
        from uacpy.io.oases_writer import (
            _oasp_layer_geometry, write_oassp_input)
        env = self._env()
        geom = _oasp_layer_geometry(env)
        first_bottom = 1 + geom['n_water_layers'] + 1
        assert first_bottom == 4
        p = tmp_path / 'sp.dat'
        write_oassp_input(p, env, Source(depths=10.0, frequencies=100.0),
                          Receiver(depths=[50.0], ranges=[1000.0, 2000.0]),
                          interface=first_bottom,
                          correlation_length=100.0, spectral_exponent=1.9,
                          n_time_samples=1024, freq_min=50.0, freq_max=150.0,
                          time_step=1e-3)
        lines = p.read_text().splitlines()
        n_layers = int(lines[3])
        seafloor_row = lines[3 + first_bottom].split()
        assert n_layers == 4
        assert float(seafloor_row[0]) == 100.0
        assert float(seafloor_row[6]) == -0.5


class TestBoundaryFileGuards:
    """Writer-side guards against files the engines mis-handle silently."""

    def test_lowercase_type_is_rejected_like_the_fortran(self, tmp_path):
        from uacpy.io.bathy_io import read_bathymetry
        p = tmp_path / 'l.bty'
        p.write_text("'l'\n2\n0.0 100.0\n1.0 200.0\n")
        with pytest.raises(FileFormatError, match='case-sensitive'):
            read_bathymetry(p)

    def test_non_monotonic_ranges_are_rejected(self, tmp_path):
        from uacpy.io.bathy_io import write_bty_file
        with pytest.raises(ConfigurationError, match='strictly increasing'):
            write_bty_file(tmp_path / 'm.bty',
                           np.array([[0.0, 100.0], [500.0, 120.0],
                                     [500.0, 150.0]]))

    def test_nan_range_axis_is_rejected(self, tmp_path):
        from uacpy.io.bathy_io import write_bty_file
        with pytest.raises(ConfigurationError, match='non-finite'):
            write_bty_file(tmp_path / 'n.bty',
                           np.array([[0.0, 100.0], [np.nan, 120.0]]))

    @pytest.mark.parametrize('writer', ['short', 'long'])
    def test_a_transposed_array_is_refused_before_the_file_is_opened(
            self, tmp_path, writer):
        """A (2, N) array (one row per quantity) has a first
        column that passes the axis check; the shape check refuses it and
        leaves no half-written file behind."""
        from uacpy.io.bathy_io import write_bty_file, write_bty_long_format
        p = tmp_path / 't.bty'
        transposed = np.array([[0.0, 5000.0, 10000.0], [100.0, 150.0, 200.0]])
        with pytest.raises(ConfigurationError, match=r'\(N, 2\).*transpose'):
            if writer == 'short':
                write_bty_file(p, transposed)
            else:
                write_bty_long_format(p, transposed, bottom=None)
        assert not p.exists()

    def test_a_one_dimensional_array_is_refused_with_its_shape(self, tmp_path):
        from uacpy.io.bathy_io import write_ati_file
        with pytest.raises(ConfigurationError, match=r'shape \(2,\)'):
            write_ati_file(tmp_path / 'o.ati', np.array([0.0, 100.0]))

    def test_a_read_table_writes_back_unchanged(self, tmp_path):
        from uacpy.io.bathy_io import read_bathymetry, write_bty_file
        rows = np.array([[0.0, 100.0], [5000.0, 150.0], [10000.0, 200.0]])
        write_bty_file(tmp_path / 'a.bty', rows, interp_type='C')
        data = read_bathymetry(tmp_path / 'a.bty')
        write_bty_file(tmp_path / 'b.bty',
                       np.column_stack([data.ranges, data.depths]),
                       interp_type=data.interpolation)
        again = read_bathymetry(tmp_path / 'b.bty')
        assert again.interpolation == 'C'
        assert np.array_equal(again.ranges, data.ranges)
        assert np.array_equal(again.depths, data.depths)


class TestRayFileReadsIfortSpellings:
    """``read_ray_file`` parses the angle and point rows with the shared
    list-directed helpers, so an ifort-written file — repeat counts
    (``2*12.0``) and a three-digit exponent without its ``E`` — reads the
    same numbers as the plain spelling."""

    HEAD = " 'probe'\n 50.0\n 1 1 1\n 1 1\n 0.0\n 200.0\n 'rz'\n"

    def test_the_ifort_spelling_reads_like_the_plain_one(self, tmp_path):
        from uacpy.io.oalib_reader import read_ray_file
        plain = tmp_path / 'plain.ray'
        plain.write_text(self.HEAD + " -20.0\n 2 0 0\n 0.0 10.0\n"
                         " 12.0 12.0\n")
        ifort = tmp_path / 'ifort.ray'
        ifort.write_text(self.HEAD + " -0.2000000+002\n 2 0 0\n 0.0 10.0\n"
                         " 2*12.0\n")
        a, b = read_ray_file(plain).rays, read_ray_file(ifort).rays
        assert len(a) == len(b) == 1
        for key in a[0]:
            np.testing.assert_array_equal(np.asarray(a[0][key]),
                                          np.asarray(b[0][key]))


class TestTruncatedOutputsAreTypedErrors:
    """A run killed mid-write leaves a structurally truncated file; the
    readers raise :class:`FileFormatError` instead of returning a subset."""

    def test_ray_file_truncated_mid_block(self, tmp_path):
        from uacpy.io.oalib_reader import read_ray_file
        p = tmp_path / 't.ray'
        p.write_text(" 'probe'\n 50.0\n 1 1 1\n 2 1\n 0.0\n 200.0\n 'rz'\n"
                     " -20.0\n 3 0 0\n 0.0 10.0\n 5.0 12.0\n")
        with pytest.raises(FileFormatError, match='truncated'):
            read_ray_file(p)

    def test_psif_truncated_is_typed(self, tmp_path):
        from scipy.io import FortranFile
        from uacpy.io.mpirams_reader import read_psif
        with FortranFile(tmp_path / 'psif.dat', 'w') as f:
            f.write_record(np.array([1024., 1, 1, 1, 1500., 1450., 8000., .5]))
            f.write_record(np.array([50.0]))
        with pytest.raises(FileFormatError,
                           match='malformed or truncated mpiramS output'):
            read_psif(tmp_path)


class TestShdExtraAxesAreCapabilityRefusals:
    """``read_shd_file`` returns single-bearing fields for one source (x, y)
    position. A well-formed file carrying any extra axis it has no carrier
    for (Ntheta > 1, Nsx/Nsy > 1) is refused by name — as
    :class:`UnsupportedFeatureError`, a capability limit — rather than
    silently reduced to slot 0 or mislabelled as corruption."""

    @staticmethod
    def _fake_bin(freqVec=(100.0,), sx=(0.0,), theta=(0.0,)):
        from uacpy.io import ShdFile
        return ShdFile(
            title='t', plot_type='rectilin  ', frequencies=np.array(freqVec),
            source_frequency=100.0, stabilizing_attenuation=0.0,
            bearings=np.array(theta), source_x=np.array(sx),
            source_y=np.array([0.0]), source_depths=np.array([5.0]),
            receiver_depths=np.array([10.0, 20.0]),
            receiver_ranges=np.array([100.0, 200.0]),
            pressure=np.ones((len(theta), 1, 2, 2), dtype=complex),
            pressure_frequency=100.0)

    def _patched(self, monkeypatch, **kwargs):
        from uacpy.io import oalib_reader
        monkeypatch.setattr(oalib_reader, 'read_shd_bin',
                            lambda p: self._fake_bin(**kwargs))
        return oalib_reader

    def test_nsx_greater_than_one_raises(self, tmp_path, monkeypatch):
        rdr = self._patched(monkeypatch, sx=(0.0, 1000.0))
        with pytest.raises(UnsupportedFeatureError, match='Nsx=2'):
            rdr.read_shd_file(tmp_path / 'x.shd')

    def test_multi_bearing_raises_unsupported(self, tmp_path, monkeypatch):
        rdr = self._patched(monkeypatch, theta=(0.0, 10.0))
        with pytest.raises(UnsupportedFeatureError, match='2 receiver bearings'):
            rdr.read_shd_file(tmp_path / 'x.shd')

    def test_zero_frequencies_is_corruption(self, tmp_path, monkeypatch):
        """No AT writer emits zero frequency records, so that stays a
        FileFormatError."""
        rdr = self._patched(monkeypatch, freqVec=())
        with pytest.raises(FileFormatError, match='zero frequencies'):
            rdr.read_shd_file(tmp_path / 'x.shd')


class TestShdBroadbandFrequencySlice:
    """A broadband ``.shd`` stacks its pressure records frequency-major
    (``KrakenField/field.f90`` resets iRec to 10 on the first frequency and
    bumps it once per (source depth, receiver depth) inside the frequency
    loop), and ``read_shd_bin``'s record index is arithmetic (io.md §3b) —
    so ``frequency=`` must land on exactly the right slab and tag
    ``pressure_freq`` with the frequency it snapped to."""

    FREQS = (100.0, 200.0, 300.0)

    @staticmethod
    def _re(ifreq, irz, irr):
        return float((ifreq + 1) * 100 + irz * 10 + irr + 1)

    @classmethod
    def _write_bin(cls, path):
        """3 frequencies × 1 bearing × 1 source × 2 receiver depths × 2
        ranges, in the ``misc/RWSHDFile.f90:100-114`` record layout."""
        recl = 41
        rec_bytes = 4 * recl
        header = [
            np.array([recl], '<i4').tobytes() + b'bb'.ljust(80),
            b'rectilin'.ljust(10),
            (np.array([3, 1, 1, 1, 1, 2, 2], '<i4').tobytes()
             + np.array([200.0, 0.0], '<f8').tobytes()),
            np.array(cls.FREQS, '<f8').tobytes(),          # freqVec
            np.array([0.0], '<f8').tobytes(),              # theta
            np.array([0.0], '<f8').tobytes(),              # Sx
            np.array([0.0], '<f8').tobytes(),              # Sy
            np.array([50.0], '<f4').tobytes(),             # Sz
            np.array([10.0, 20.0], '<f4').tobytes(),       # Rz
            np.array([100.0, 200.0], '<f8').tobytes(),     # Rr
        ]
        pressure = []
        for ifreq in range(3):
            for irz in range(2):
                row = []
                for irr in range(2):
                    row += [cls._re(ifreq, irz, irr), float(ifreq + 1)]
                pressure.append(np.array(row, '<f4').tobytes())
        path.write_bytes(b''.join(r.ljust(rec_bytes, b'\x00')
                                  for r in header + pressure))
        return path

    def _expected(self, ifreq):
        return np.array([[self._re(ifreq, irz, irr) + 1j * (ifreq + 1)
                          for irr in range(2)] for irz in range(2)])

    def test_frequency_selects_the_matching_slab(self, tmp_path):
        from uacpy.io.oalib_reader import read_shd_bin
        shd = read_shd_bin(str(self._write_bin(tmp_path / 'bb.shd')),
                           frequency=200.0)
        np.testing.assert_allclose(shd.pressure[0, 0], self._expected(1))
        assert shd.pressure_frequency == pytest.approx(200.0)
        np.testing.assert_allclose(shd.frequencies, self.FREQS)

    def test_default_is_the_first_slab(self, tmp_path):
        from uacpy.io.oalib_reader import read_shd_bin
        shd = read_shd_bin(str(self._write_bin(tmp_path / 'bb.shd')))
        np.testing.assert_allclose(shd.pressure[0, 0], self._expected(0))
        assert shd.pressure_frequency == pytest.approx(100.0)

    def test_off_grid_frequency_snaps_to_the_nearest_slab(self, tmp_path):
        from uacpy.io.oalib_reader import read_shd_bin
        shd = read_shd_bin(str(self._write_bin(tmp_path / 'bb.shd')),
                           frequency=280.0)
        np.testing.assert_allclose(shd.pressure[0, 0], self._expected(2))
        assert shd.pressure_frequency == pytest.approx(300.0)

    def test_read_shd_file_returns_a_broadband_field(self, tmp_path):
        """Every slab lands on its own frequency, on a trailing axis."""
        from uacpy.io.oalib_reader import read_shd_file
        field = read_shd_file(self._write_bin(tmp_path / 'bb.shd'))
        assert field.data.shape == (2, 2, 3)
        np.testing.assert_allclose(field.coords['frequency'], self.FREQS)
        for ifreq in range(3):
            np.testing.assert_allclose(field.data[:, :, ifreq],
                                       self._expected(ifreq))
        assert field.phase_reference is None


class TestReadPsifFullContract:
    """io.md §6: ``read_psif`` takes the directory (or the file), renames the header
    scalars to the metadata schema (``Nsam`` → ``n_samples``, ``cmin`` →
    ``c_min``), returns the frq/rout/zg axes as written (Hz / m), and
    de-interleaves each depth record's real/imag pairs into a complex
    ``psif`` of shape ``(nzo, nf, nr)``."""

    def test_round_trip_of_a_synthetic_file(self, tmp_path):
        from scipy.io import FortranFile
        from uacpy.io.mpirams_reader import read_psif
        nf, nzo, nr = 2, 3, 2
        with FortranFile(tmp_path / 'psif.dat', 'w') as f:
            f.write_record(np.array([1024.0, nf, nzo, nr,
                                     1600.0, 1450.0, 8000.0, 4.0]))
            f.write_record(np.array([50.0, 51.0]))            # frq (Hz)
            f.write_record(np.array([1000.0, 2000.0]))        # rout (m)
            # Depth records: [z, Re_1, Im_1, ..., Re_nf, Im_nf], nzo per
            # range (mpiramS writes range-major).
            for ir in range(nr):
                for iz in range(nzo):
                    rec = [10.0 * (iz + 1)]
                    for jf in range(nf):
                        rec += [100.0 * (ir + 1) + 10.0 * (iz + 1) + jf,
                                -(jf + 1.0)]
                    f.write_record(np.array(rec))
        out = read_psif(tmp_path)
        assert out.n_samples == 1024.0
        assert out.water_min == 1450.0
        assert out.c0 == 1600.0
        assert out.sample_rate == 8000.0
        assert out.q_factor == 4.0
        np.testing.assert_allclose(out.frequencies, [50.0, 51.0])
        np.testing.assert_allclose(out.ranges, [1000.0, 2000.0])
        np.testing.assert_allclose(out.depths, [10.0, 20.0, 30.0])
        assert out.pe_field.shape == (3, 2, 2)
        assert out.pe_field.dtype == np.complex128
        # (iz=1, jf=0, ir=1): Re = 200 + 20 + 0, Im = -1.
        assert out.pe_field[1, 0, 1] == pytest.approx(220.0 - 1.0j)
        # (iz=2, jf=1, ir=0): Re = 100 + 30 + 1, Im = -2.
        assert out.pe_field[2, 1, 0] == pytest.approx(131.0 - 2.0j)
        pytest.importorskip('xarray')
        ds = out.to_xarray()
        assert ds['pe_field'].dims == ('depth', 'frequency', 'range')
        assert complex(ds['pe_field'].sel(depth=30.0, frequency=51.0,
                                          range=1000.0)) == 131.0 - 2.0j

    def test_wrong_header_size_is_a_typed_error(self, tmp_path):
        from scipy.io import FortranFile
        from uacpy.io.mpirams_reader import read_psif
        with FortranFile(tmp_path / 'psif.dat', 'w') as f:
            f.write_record(np.zeros(7))
        with pytest.raises(FileFormatError, match='expected 8'):
            read_psif(tmp_path)

    def test_the_file_path_reads_like_its_directory(self, tmp_path):
        from scipy.io import FortranFile
        from uacpy.io.mpirams_reader import read_psif
        with FortranFile(tmp_path / 'psif.dat', 'w') as f:
            f.write_record(np.array([64.0, 1, 1, 1, 1500., 1450., 8000., 4.]))
            f.write_record(np.array([50.0]))
            f.write_record(np.array([1000.0]))
            f.write_record(np.array([10.0, 3.0, -4.0]))
        by_file = read_psif(tmp_path / 'psif.dat')
        assert by_file.pe_field[0, 0, 0] == pytest.approx(3.0 - 4.0j)
        np.testing.assert_array_equal(by_file.pe_field,
                                      read_psif(tmp_path).pe_field)

    def test_a_stock_direct_access_file_is_named_as_such(self, tmp_path):
        """Stock mpiramS writes a direct-access psif.dat plus recl.dat
        (README.RECL); the refusal names that layout instead of a crash."""
        from uacpy.io.mpirams_reader import read_psif
        recl = 8 * 8
        header = np.array([64.0, 1, 1, 1, 1500., 1450., 8000., 4.])
        (tmp_path / 'psif.dat').write_bytes(
            header.tobytes() + np.array([50.0]).tobytes().ljust(recl, b'\0'))
        (tmp_path / 'recl.dat').write_text(f"{recl}\n")
        with pytest.raises(FileFormatError, match='stock mpiramS direct-access'):
            read_psif(tmp_path)
        # The same bytes without recl.dat keep the generic message.
        (tmp_path / 'recl.dat').unlink()
        with pytest.raises(
                FileFormatError,
                match='malformed or truncated mpiramS output') as excinfo:
            read_psif(tmp_path)
        assert 'recl.dat' not in str(excinfo.value)


def test_an_empty_reflection_table_reads_as_three_empty_float_columns(tmp_path):
    """A count of 0 goes through the same list-directed path as any other
    count: ``read_list_directed_values`` reads nothing and the three columns
    come back as empty float arrays."""
    from uacpy.io.refl_io import read_reflection_coefficient
    path = tmp_path / 'empty.brc'
    path.write_text('0\n')
    table = read_reflection_coefficient(path)
    assert table.n_angles == 0
    for column in ('angles', 'magnitude', 'phase'):
        assert getattr(table, column).shape == (0,)
        assert getattr(table, column).dtype == float


@pytest.mark.parametrize('reader_name, content', [
    # Record 5 (NRCV, NFREQ) is two int32 after a 32-byte title: a file that
    # ends four bytes into it makes struct.unpack raise struct.error.
    ('read_oasn_covariance', b'\x00' * 32 + struct.pack('<i', 12)),
    # A well-framed first record whose decode the patched reader refuses.
    ('read_oasn_replicas',
     struct.pack('<i', 12) + b'\x00' * 12 + struct.pack('<i', 12)),
    # A valid table header, then a per-frequency header whose sample count
    # is not an integer: int() raises ValueError.
    ('read_oasr_reflection_coefficients', b'100.0 200.0 1 1\n100.0 abc\n'),
    ('read_oases_rhs_header',
     struct.pack('<i', 12) + b'\x00' * 12 + struct.pack('<i', 12)),
], ids=['oasn_covariance', 'oasn_replicas', 'oasr', 'rhs_header'])
def test_oases_binary_readers_type_a_raw_parse_error(
        reader_name, content, tmp_path, monkeypatch):
    """A ``struct.error`` / ``ValueError`` / ``IndexError`` escaping a record
    decode surfaces as the ``typed_format_error`` ``FileFormatError``
    ("could not parse"), the same conversion every other ``io`` reader
    applies."""
    import struct
    from uacpy.io import oases_reader

    def short_record(*args, **kwargs):
        raise struct.error('unpack requires a buffer of 12 bytes')

    monkeypatch.setattr(oases_reader, '_read_fortran_record', short_record)
    path = tmp_path / 'out.bin'
    path.write_bytes(content)
    with pytest.raises(FileFormatError, match='could not parse'):
        getattr(oases_reader, reader_name)(path)


class TestOasesReadersRefuseAMalformedFile:
    """Each OASES output reader names what it found where a record belongs,
    instead of returning numbers read out of step."""

    def test_a_plp_record_whose_value_is_not_a_number(self, tmp_path):
        from uacpy.io._parsers import parse_oast_tl
        plp = _plp_tl_pair(tmp_path, [(100.0, 10.0, 1.0, 0.5)])
        text = plp.read_text()
        assert text.count(f"{1:<19}NC") == 1
        plp.write_text(text.replace(f"{1:<19}NC", f"{'x':<19}NC"))
        with pytest.raises(FileFormatError, match='NC field is not numeric'):
            parse_oast_tl(plp, [10.0])

    def test_a_plt_line_that_is_not_one_value(self, tmp_path):
        from uacpy.io._parsers import parse_oast_tl
        plp = _plp_tl_pair(tmp_path, [(100.0, 10.0, 1.0, 0.5)])
        plt = plp.with_suffix('.plt')
        plt.write_text(plt.read_text().replace(' 0.0\n', ' 0.0 1.0\n', 1))
        with pytest.raises(FileFormatError,
                           match='expected a single G13.6 value'):
            parse_oast_tl(plp, [10.0])

    def test_a_trf_shorter_than_one_record_marker(self, tmp_path):
        from uacpy.io.oases_reader import read_oasp_trf
        trf = tmp_path / 'short.trf'
        trf.write_bytes(b'\x00\x01\x02')
        with pytest.raises(FileFormatError, match='TRF: too short'):
            read_oasp_trf(trf, [10.0])

    def test_an_oasr_table_whose_header_has_under_four_fields(self, tmp_path):
        from uacpy.io.oases_reader import read_oasr_reflection_coefficients
        table = tmp_path / 'r.trc'
        table.write_text('100.0 100.0\n')
        with pytest.raises(FileFormatError, match='Invalid header format'):
            read_oasr_reflection_coefficients(table)

    def test_an_oasr_table_that_ends_inside_a_frequency_block(self, tmp_path):
        from uacpy.io.oases_reader import read_oasr_reflection_coefficients
        table = tmp_path / 'r.trc'
        table.write_text('100.0 100.0 1 2\n 100.000     2\n'
                         '10.0 0.9 0.1\n20.0 0.8\n')
        with pytest.raises(FileFormatError,
                           match=r'reflection row 2 of 2 at 100 Hz is short'):
            read_oasr_reflection_coefficients(table)


class TestDeckTitlesAreAscii:
    """The engines cut titles in bytes and echo them into text outputs, so a
    UTF-8 character cut in half became an undecodable byte (measured:
    ``'x' + 'é'*79`` crashed OASP/OASN/OASR with ``UnicodeDecodeError``,
    ``'é'*79`` crashed OAST, Bellhop RAYS and SPARC). Deck titles are folded
    to ASCII; ``Environment.name`` keeps its Unicode."""

    @pytest.mark.parametrize('name, expected', [
        ('é' * 79, 'e' * 79),
        ('Golfe du Lion — été', 'Golfe du Lion ? ete'),
        ('水深', '??'),
        ('plain', 'plain'),
    ])
    def test_deck_title_folds_to_ascii(self, name, expected):
        from uacpy.io._fortran_helpers import deck_title
        assert deck_title(name) == expected

    def test_at_and_oases_decks_carry_the_folded_title(self, tmp_path):
        from uacpy import Environment, Receiver, Source
        from uacpy.io.oalib_writer import quote_fortran_title
        from uacpy.io.oases_writer import write_oasp_input
        env = Environment(name='x' + 'é' * 79, bathymetry=100.0, ssp=1500.0)
        assert env.name == 'x' + 'é' * 79
        assert quote_fortran_title(env.name) == "'x" + 'e' * 79 + "'"
        deck = tmp_path / 'p.dat'
        write_oasp_input(deck, env, Source(depths=50.0, frequencies=100.0),
                         Receiver(depths=[30.0], ranges=[1000.0, 2000.0]))
        first = deck.read_bytes().splitlines()[0]
        assert first.isascii() and len(first) <= 80


class TestTheTransferFunctionReaderReturnsPressure:
    """``read_oasp_trf`` returns the pressure Field of an ``'N'`` file —
    the normal stress negated (``oasp.tex:185``) — and refuses any other
    OASP output parameter rather than label it a pressure."""

    @staticmethod
    def _raw(option):
        tf = np.arange(2 * 3 * 4, dtype=np.complex64).reshape(2, 3, 4) + 1j
        return {'title': 't', 'option': option,
                'freq': np.array([100.0, 110.0]),
                'ranges': np.array([500.0, 1000.0, 1500.0]),
                'depths': np.array([10.0, 20.0, 30.0, 40.0]),
                'transfer_function': tf, 'source_depth': 25.0,
                'center_frequency': 105.0, 'omegim': 0.0}

    def test_normal_stress_becomes_the_pressure(self, monkeypatch):
        import uacpy.io._parsers as parsers
        from uacpy.io import read_oasp_trf
        raw = self._raw('N')
        monkeypatch.setattr(parsers, 'parse_oasp_trf',
                            lambda path, depths: raw)
        field = read_oasp_trf('x.trf', [10.0, 20.0, 30.0, 40.0])
        assert list(field.coords) == ['depth', 'range', 'frequency']
        assert field.data[3, 2, 1] == -raw['transfer_function'][1, 2, 3]
        assert field.data.dtype == np.complex128
        assert field.phase_reference == 'travelling_wave'
        assert field.source_depths.tolist() == [25.0]

    @pytest.mark.parametrize('option', ['V', 'H'])
    def test_another_parameter_is_refused(self, monkeypatch, option):
        import uacpy.io._parsers as parsers
        from uacpy.core.exceptions import UnsupportedFeatureError
        from uacpy.io import read_oasp_trf
        monkeypatch.setattr(parsers, 'parse_oasp_trf',
                            lambda path, depths: self._raw(option))
        with pytest.raises(UnsupportedFeatureError, match=repr(option)):
            read_oasp_trf('x.trf', [10.0])

