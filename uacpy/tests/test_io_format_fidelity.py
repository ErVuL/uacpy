"""Format-fidelity contracts of the io layer.

Each class pins one reader/writer behaviour against the vendored Fortran's
own read semantics: list-directed whole-vector READs span records
(``Bellhop/sspMod.f90:417,428``, ``misc/RefCoef.f90:53``), fscanf-style
token streams ignore line breaks (``Matlab/ReadWrite/read_ts.m``), and the
engines' SELECT CASE option parsing is case-sensitive
(``Bellhop/bdryMod.f90:162-165``).
"""

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
        bty, _ = read_bathymetry(tmp_path / 'survey.dat')
        assert bty[1, 1:-1].tolist() == [100.0, 200.0]

    def test_extensionless_root_resolves_to_bty(self, tmp_path):
        from uacpy.io.bathy_io import read_bathymetry
        (tmp_path / 'survey.bty').write_text("'L'\n2\n0.0 50.0\n1.0 60.0\n")
        bty, _ = read_bathymetry(tmp_path / 'survey')
        assert bty[1, 1:-1].tolist() == [50.0, 60.0]

    def test_a_missing_explicit_path_is_a_configuration_error(self, tmp_path):
        """A .bty is a deck the caller wrote, so an absent one is a bad
        argument — the provenance split core/exceptions.py states."""
        from uacpy.io.bathy_io import read_bathymetry
        with pytest.raises(ConfigurationError):
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
        assert d['n_pts'] == 4
        assert d['theta'].tolist() == [0.0, 10.0, 20.0, 30.0]
        assert d['R'].tolist() == [1.0, 0.9, 0.8, 0.7]

    def test_reader_takes_wrapped_records(self, tmp_path):
        from uacpy.io.refl_io import read_reflection_coefficient
        p = tmp_path / 'wrap.brc'
        p.write_text("2\n0.0 1.0\n180.0\n10.0 0.9 170.0\n")
        d = read_reflection_coefficient(p)
        assert d['theta'].tolist() == [0.0, 10.0]

    def test_dedupe_keeps_every_packed_record(self, tmp_path):
        from uacpy.io.refl_io import (
            dedupe_reflection_file, read_reflection_coefficient)
        p = tmp_path / 'staged.brc'
        p.write_text(self.TABLE)
        dedupe_reflection_file(p)
        d = read_reflection_coefficient(p)
        assert d['n_pts'] == 4, "dedupe dropped legally packed records"
        assert d['theta'].tolist() == [0.0, 10.0, 20.0, 30.0]

    def test_dedupe_drops_the_evanescent_head(self, tmp_path):
        from uacpy.io.refl_io import (
            dedupe_reflection_file, read_reflection_coefficient)
        p = tmp_path / 'evan.brc'
        p.write_text("4\n0.0 1.0 180.0  0.0 1.0 180.0\n"
                     "0.0 1.0 180.0  10.0 0.9 170.0\n")
        dedupe_reflection_file(p)
        d = read_reflection_coefficient(p)
        assert d['theta'].tolist() == [0.0, 10.0]


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
        bty, bty_type = read_bathymetry(p)
        assert bty_type == 'L'
        assert bty[0, 1:-1].tolist() == [0.0, 1000.0, 2000.0]
        assert bty[1, 1:-1].tolist() == [100.0, 150.0, 120.0]

    def test_bty_row_remainder_is_discarded(self, tmp_path):
        """Each point is ONE list-directed READ: tokens past the row's
        n_cols on its final record are the remainder the READ skips."""
        from uacpy.io.bathy_io import read_bathymetry
        p = tmp_path / 'rem.bty'
        p.write_text("'L'\n2\n0.0 100.0 ignored garbage\n1.0 150.0\n")
        bty, _ = read_bathymetry(p)
        assert bty[1, 1:-1].tolist() == [100.0, 150.0]

    def test_ati_comma_count_line(self, tmp_path):
        from uacpy.io.bathy_io import read_altimetry
        p = tmp_path / 'c.ati'
        p.write_text("'L'\n2, ! npts\n0.0 0.0\n1.0 -2.0\n")
        ati, _ = read_altimetry(p)
        assert ati[1, 1:-1].tolist() == [0.0, -2.0]

    def test_brc_comma_count_line(self, tmp_path):
        from uacpy.io.refl_io import (
            dedupe_reflection_file, read_reflection_coefficient)
        p = tmp_path / 'c.brc'
        p.write_text("2, ! npts\n0.0 1.0 180.0\n10.0 0.9 170.0\n")
        d = read_reflection_coefficient(p)
        assert d['n_pts'] == 2
        assert d['theta'].tolist() == [0.0, 10.0]
        # The staging dedupe accepts the same count record the reader does.
        dedupe_reflection_file(p)
        assert read_reflection_coefficient(p)['n_pts'] == 2

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
        assert d['theta'].tolist() == [0.0, 10.0]
        assert d['R'].tolist() == [0.0, 0.9]

    def test_rts_repeat_count_reads_as_repeated_values(self, tmp_path):
        from uacpy.io.oalib_reader import read_rts_file
        p = tmp_path / 'rep.rts'
        p.write_text("'run'\n2 5000.0 10000.0\n0.0 2*0.5\n")
        d = read_rts_file(p)
        assert d['ranges'].tolist() == [5000.0, 10000.0]
        assert d['p'].tolist() == [[0.5, 0.5]]

    def test_ts_repeat_count_reads_as_repeated_values(self, tmp_path):
        from uacpy.io.oalib_reader import read_ts
        p = tmp_path / 'rep.ts'
        p.write_text("t\n2 10.0 20.0\n0.0 2*1.5\n0.1 3.0 4.0\n")
        d = read_ts(p)
        assert d['RTS'].tolist() == [[1.5, 1.5], [3.0, 4.0]]

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
        with pytest.raises(FileFormatError):
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
        assert read_reflection_coefficient(p)['n_pts'] == 3
        dedupe_reflection_file(p)
        assert read_reflection_coefficient(p)['n_pts'] == 3


class TestSspRecordsSpanLines:
    """``Bellhop/sspMod.f90:417,428`` read the range vector and each depth
    row with whole-vector list-directed READs, which consume as many lines
    as they need and discard the remainder of their final line."""

    def test_2d_rows_may_wrap(self, tmp_path):
        from uacpy.io.oalib_reader import read_ssp_2d
        p = tmp_path / 'w.ssp'
        p.write_text("3\n0.0 5.0\n10.0\n1500 1501\n1502\n1490 1491 1492\n")
        d = read_ssp_2d(p)
        assert d['n_prof'] == 3
        assert d['r_prof'].tolist() == [0.0, 5000.0, 10000.0]
        assert d['c_mat'].tolist() == [[1500, 1501, 1502], [1490, 1491, 1492]]

    def test_2d_row_remainder_is_discarded(self, tmp_path):
        from uacpy.io.oalib_reader import read_ssp_2d
        p = tmp_path / 'p.ssp'
        p.write_text("2\n0.0 5.0\n1500 1501 9e9 9e9\n1490 1491\n")
        d = read_ssp_2d(p)
        assert d['c_mat'].tolist() == [[1500, 1501], [1490, 1491]]

    def test_truncated_row_is_a_typed_error(self, tmp_path):
        from uacpy.io.oalib_reader import read_ssp_2d
        p = tmp_path / 't.ssp'
        p.write_text("3\n0.0 5.0 10.0\n1500 1501\n")
        with pytest.raises(FileFormatError):
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
        assert t['PlotTitle'] == 'my title'
        assert t['pos']['r']['z'].tolist() == [10.0, 20.0, 30.0]
        assert t['tout'].tolist() == [0.0, 0.1, 0.2]
        assert t['RTS'].tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def test_partial_trailing_block_is_dropped(self, tmp_path):
        from uacpy.io.oalib_reader import read_ts
        p = tmp_path / 't.ts'
        p.write_text("t\n2 10.0 20.0\n0.0 1 2 0.1 3\n")
        t = read_ts(p)
        assert t['tout'].tolist() == [0.0]

    def test_empty_payload_is_a_typed_error(self, tmp_path):
        from uacpy.io.oalib_reader import read_ts
        p = tmp_path / 't.ts'
        p.write_text("title only\n")
        with pytest.raises(FileFormatError):
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
        from uacpy.io.oases_reader import read_oast_tl
        plp = self._write_pair(tmp_path, n_freq=1, n_depths=2)
        out = read_oast_tl(plp, [10.0, 20.0])
        assert out['tl'].shape == (2, 4)
        assert out['metadata']['n_frequencies'] == 1
        assert out['ranges'].tolist() == [1000.0, 1500.0, 2000.0, 2500.0]
        assert out['depths'].tolist() == [10.0, 20.0]

    def test_multi_frequency_returns_a_stack(self, tmp_path):
        from uacpy.io.oases_reader import read_oast_tl
        plp = self._write_pair(tmp_path, n_freq=3, n_depths=2)
        out = read_oast_tl(plp, [10.0, 20.0])
        assert out['tl'].shape == (3, 2, 4)
        assert out['metadata']['n_frequencies'] == 3
        # Frequency-major order: curve (ifreq, idepth) = ifreq*2 + idepth.
        assert out['tl'][1, 1, 0] == 300.0
        assert out['tl'][2, 0, 0] == 400.0
        assert out['metadata']['frequencies'].tolist() == [100.0, 200.0, 300.0]

    def test_non_multiple_curve_count_is_a_typed_error(self, tmp_path):
        """A label-less ``.plp`` falls back to the positional walk, which is
        the one that can only check the count."""
        from uacpy.io.oases_reader import read_oast_tl
        curves = [(None, None, 1.0, 0.5)] * 3
        plp = _plp_tl_pair(tmp_path, curves)
        with pytest.raises(FileFormatError, match='whole multiple'):
            read_oast_tl(plp, [10.0, 20.0])

    def test_a_label_less_plp_warns_before_walking_by_position(self, tmp_path):
        """Nothing but PLTLOS writes a ``…TLRAN`` curve, so a block without
        the labels came from elsewhere; the positional walk still reads it,
        but it is the reading ``IDINC`` breaks."""
        from uacpy.io.oases_reader import read_oast_tl
        plp = _plp_tl_pair(tmp_path, [(None, None, 1.0, 0.5)] * 2)
        with pytest.warns(UserWarning, match='by position'):
            out = read_oast_tl(plp, [10.0, 20.0])
        assert out['tl'].shape == (2, 4)
        assert out['depths'].tolist() == [10.0, 20.0]
        assert 'frequencies' not in out['metadata']


class TestOastCurvesAreKeyedByTheirLabels:
    """PLTLOS writes ``Freq:``/``SD:``/``RD:`` into every plot block
    (``oasfun22.f:334-337``) and runs only for ``MOD( NREC-1, INTF ) == 0``
    (``unoast31.f:630``), ``INTF`` being ``IDINC``, the 4th field of the
    OASTL receiver record (``oaseun31.f:1156``). A decimated run therefore
    writes fewer curves than the deck has receivers, and only the labels say
    which receiver each curve belongs to."""

    def test_decimated_receivers_land_on_the_depths_that_were_plotted(
            self, tmp_path):
        from uacpy.io.oases_reader import read_oast_tl
        # 'RD1 RD2 4 2' over 10..90 m: OAST plots receivers 1 and 3 only.
        rd = np.linspace(10.0, 90.0, 4)
        curves = [(400.0, 10.0, 0.0, 0.25), (400.0, 63.3, 0.0, 0.25),
                  (800.0, 10.0, 0.0, 0.125), (800.0, 63.3, 0.0, 0.125)]
        plp = _plp_tl_pair(tmp_path, curves)
        out = read_oast_tl(plp, rd)
        assert out['tl'].shape == (2, 2, 4)
        assert out['metadata']['n_frequencies'] == 2
        # The F9.1 label names the receiver to 0.1 m; the axis the caller
        # passed supplies the full-precision value.
        assert out['depths'] == pytest.approx([10.0, 190.0 / 3.0])
        assert out['metadata']['frequencies'].tolist() == [400.0, 800.0]

    def test_a_label_off_the_receiver_axis_is_returned_as_printed(
            self, tmp_path):
        from uacpy.io.oases_reader import read_oast_tl
        curves = [(100.0, 33.3, 0.0, 0.25)]
        plp = _plp_tl_pair(tmp_path, curves)
        out = read_oast_tl(plp, [10.0, 20.0])
        assert out['depths'].tolist() == [33.3]

    def test_colliding_labels_fall_back_to_position(self, tmp_path):
        """Two receivers closer than the F9.1 label quantum print the same
        ``RD:``. The grid stops factoring, and the positional walk — right
        for every run that does not decimate — takes over behind a warning."""
        from uacpy.io.oases_reader import read_oast_tl
        curves = [(100.0, 10.0, 0.0, 0.25), (100.0, 10.0, 0.0, 0.25)]
        plp = _plp_tl_pair(tmp_path, curves)
        with pytest.warns(UserWarning, match='by position'):
            out = read_oast_tl(plp, [10.00, 10.04])
        assert out['tl'].shape == (2, 4)
        assert out['depths'].tolist() == [10.0, 10.04]

    def test_labels_that_neither_factor_nor_divide_are_a_typed_error(
            self, tmp_path):
        from uacpy.io.oases_reader import read_oast_tl
        curves = [(400.0, 10.0, 0.0, 0.25), (400.0, 20.0, 0.0, 0.25),
                  (800.0, 10.0, 0.0, 0.25)]
        plp = _plp_tl_pair(tmp_path, curves)
        with pytest.raises(FileFormatError, match='not a whole multiple'):
            read_oast_tl(plp, [10.0, 20.0])

    def test_a_grid_with_a_hole_is_a_typed_error(self, tmp_path):
        """The grid factors on its totals — 2 frequencies by 2 depths, 4
        curves — while one pair repeats and another is absent, which would
        otherwise return whatever ``np.empty`` allocated."""
        from uacpy.io.oases_reader import read_oast_tl
        curves = [(400.0, 10.0, 0.0, 0.25), (400.0, 20.0, 0.0, 0.25),
                  (800.0, 10.0, 0.0, 0.25), (800.0, 10.0, 0.0, 0.25)]
        plp = _plp_tl_pair(tmp_path, curves)
        with pytest.raises(FileFormatError, match='no TL curve for'):
            read_oast_tl(plp, [10.0, 20.0])


class TestOastRangeAxisIsPerFrequency:
    """``DLRAN = 2*pi/(NWVNO*DLWVNO)`` is recomputed inside the frequency
    loop (``unoast31.f:481``) with ``DLWVNO`` proportional to FREQ, so DX
    halves when the frequency doubles while ``LF`` stays pinned at NWVNO
    (``:492``) — equal point counts, different range axes."""

    def test_each_frequency_gets_its_own_axis(self, tmp_path):
        from uacpy.io.oases_reader import read_oast_tl
        curves = [(400.0, 10.0, 0.00336845, 0.00336845),
                  (800.0, 10.0, 0.00168423, 0.00168423)]
        plp = _plp_tl_pair(tmp_path, curves, n_ranges=512)
        out = read_oast_tl(plp, [10.0])
        assert out['ranges'].shape == (2, 512)
        assert out['ranges'][0, -1] == pytest.approx(1724.6464)
        assert out['ranges'][1, -1] == pytest.approx(862.32576)

    def test_one_frequency_returns_a_1d_axis(self, tmp_path):
        from uacpy.io.oases_reader import read_oast_tl
        plp = _plp_tl_pair(tmp_path, [(400.0, 10.0, 1.0, 0.5)])
        out = read_oast_tl(plp, [10.0])
        assert out['ranges'].shape == (4,)
        assert out['tl'].shape == (1, 4)

    def test_receivers_of_one_frequency_must_share_a_grid(self, tmp_path):
        from uacpy.io.oases_reader import read_oast_tl
        curves = [(400.0, 10.0, 1.0, 0.5), (400.0, 20.0, 1.0, 0.25)]
        plp = _plp_tl_pair(tmp_path, curves)
        with pytest.raises(FileFormatError, match='do not share a range grid'):
            read_oast_tl(plp, [10.0, 20.0])


class TestPlpCountsAreValidated:
    """``NLAB`` and ``NC`` are the file's own DO-loop bounds
    (``oasgun21.f:614``, ``:653``) and the ``.plp`` walk advances by them, so
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
    def _deck(tmp_path, **kw):
        from uacpy import Environment, Receiver, Source
        from uacpy.core.bottom import BoundaryProperties
        from uacpy.io.oases_writer import write_oasn_input
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=BoundaryProperties(
                              acoustic_type='half-space', sound_speed=1700.0,
                              density=1.8, attenuation=0.5))
        p = tmp_path / 'n.dat'
        write_oasn_input(p, env, Source(depths=10.0, frequencies=100.0),
                         Receiver(depths=[30.0, 50.0], ranges=[0.0]),
                         options='N J', surface_noise_level=70.0, **kw)
        return p.read_text()

    def test_default_writes_numerically_nil_level(self, tmp_path):
        text = self._deck(tmp_path)
        assert '70.0 -200.0 0.0 0' in text

    def test_explicit_zero_writes_literal_zero_db(self, tmp_path):
        text = self._deck(tmp_path, white_noise_level=0.0)
        assert '70.0 0.0 0.0 0' in text

    def test_noise_band_total_is_bounded_preflight(self, tmp_path):
        with pytest.raises(ConfigurationError,
                           match='TOO MANY SAMPLING POINTS'):
            self._deck(tmp_path, nw_samples=40000)


class TestOasesInterfaceIndexSpaces:
    """Each family's first-bottom-interface helper matches its own written
    deck for an isovelocity column: OASS collapses the water to one record
    (seafloor = deck layer 3), the OASP family writes one record per SSP
    row (seafloor = 2 + n_rows)."""

    @staticmethod
    def _env():
        from uacpy import Environment
        from uacpy.core.bottom import BoundaryProperties
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
        with pytest.raises(FileFormatError):
            read_psif(tmp_path)


class TestShdExtraAxesAreCapabilityRefusals:
    """``read_shd_file`` returns single-frequency, single-bearing fields for
    one source (x, y) position. A well-formed file carrying any extra axis
    (broadband, Ntheta > 1, Nsx/Nsy > 1) is refused by name — as
    :class:`UnsupportedFeatureError`, a capability limit — rather than
    silently reduced to slot 0 or mislabelled as corruption."""

    @staticmethod
    def _fake_bin(freqVec=(100.0,), sx=(0.0,), theta=(0.0,)):
        return {
            'freqVec': np.array(freqVec),
            'PlotType': 'rectilin  ',
            'Pos': {'theta': np.array(theta),
                    's': {'x': np.array(sx),
                          'y': np.array([0.0]),
                          'z': np.array([5.0])},
                    'r': {'z': np.array([10.0, 20.0]),
                          'r': np.array([100.0, 200.0])}},
            'pressure': np.ones((len(theta), 1, 2, 2), dtype=complex),
            'freq0': 100.0, 'atten': 0.0, 'title': 't',
            'pressure_freq': 100.0,
        }

    def _patched(self, monkeypatch, **kwargs):
        from uacpy.io import oalib_reader
        monkeypatch.setattr(oalib_reader, 'read_shd_bin',
                            lambda p: self._fake_bin(**kwargs))
        return oalib_reader

    def test_nsx_greater_than_one_raises(self, tmp_path, monkeypatch):
        rdr = self._patched(monkeypatch, sx=(0.0, 1000.0))
        with pytest.raises(UnsupportedFeatureError, match='Nsx=2'):
            rdr.read_shd_file(tmp_path / 'x.shd')

    def test_multi_frequency_raises_unsupported(self, tmp_path, monkeypatch):
        rdr = self._patched(monkeypatch, freqVec=(100.0, 200.0))
        with pytest.raises(UnsupportedFeatureError, match='2 frequencies'):
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
        np.testing.assert_allclose(shd['pressure'][0, 0], self._expected(1))
        assert shd['pressure_freq'] == pytest.approx(200.0)
        np.testing.assert_allclose(shd['freqVec'], self.FREQS)

    def test_default_is_the_first_slab(self, tmp_path):
        from uacpy.io.oalib_reader import read_shd_bin
        shd = read_shd_bin(str(self._write_bin(tmp_path / 'bb.shd')))
        np.testing.assert_allclose(shd['pressure'][0, 0], self._expected(0))
        assert shd['pressure_freq'] == pytest.approx(100.0)

    def test_off_grid_frequency_snaps_to_the_nearest_slab(self, tmp_path):
        from uacpy.io.oalib_reader import read_shd_bin
        shd = read_shd_bin(str(self._write_bin(tmp_path / 'bb.shd')),
                           frequency=280.0)
        np.testing.assert_allclose(shd['pressure'][0, 0], self._expected(2))
        assert shd['pressure_freq'] == pytest.approx(300.0)


class TestReadPsifFullContract:
    """io.md §5: ``read_psif`` takes the directory, renames the header
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
        assert out['n_samples'] == 1024.0
        assert out['c_min'] == 1450.0
        assert 'Nsam' not in out and 'cmin' not in out
        assert (out['nf'], out['nzo'], out['nr']) == (2, 3, 2)
        assert out['c0'] == 1600.0
        assert out['fs'] == 8000.0
        assert out['Q'] == 4.0
        np.testing.assert_allclose(out['frq'], [50.0, 51.0])
        np.testing.assert_allclose(out['rout'], [1000.0, 2000.0])
        np.testing.assert_allclose(out['zg'], [10.0, 20.0, 30.0])
        assert out['psif'].shape == (3, 2, 2)
        assert out['psif'].dtype == np.complex128
        # (iz=1, jf=0, ir=1): Re = 200 + 20 + 0, Im = -1.
        assert out['psif'][1, 0, 1] == pytest.approx(220.0 - 1.0j)
        # (iz=2, jf=1, ir=0): Re = 100 + 30 + 1, Im = -2.
        assert out['psif'][2, 1, 0] == pytest.approx(131.0 - 2.0j)

    def test_wrong_header_size_is_a_typed_error(self, tmp_path):
        from scipy.io import FortranFile
        from uacpy.io.mpirams_reader import read_psif
        with FortranFile(tmp_path / 'psif.dat', 'w') as f:
            f.write_record(np.zeros(7))
        with pytest.raises(FileFormatError, match='expected 8'):
            read_psif(tmp_path)
