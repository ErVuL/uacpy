"""Reflection tables and source beam patterns (``uacpy.io.refl_io``):
staging next to the deck, deduplication, angle order, and the ``.sbp``
round trip.
"""

import hashlib
import numpy as np
import pytest
import warnings
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.exceptions import FileFormatError
from uacpy.tests.conftest import recorded_warnings


class TestReflectionFileDedupe:
    """``dedupe_reflection_file`` is a .brc/.trc rewriter, not a .irc one.

    BOUNCE writes the .irc with a title/frequency header and
    ``(5G15.7, I5)`` records (``Kraken/bounce.f90:225-228``), read back by
    the same fixed format at ``misc/RefCoef.f90:98-107``. The 3-column
    angle dedupe would strip the header and four of the columns, so an
    .irc must be rejected rather than rewritten.
    """

    def _irc(self, tmp_path):
        p = tmp_path / 't.irc'
        p.write_text(
            " ' BOUNCE test '  100.0\n 2\n"
            "     0.1000000     1.0000000     0.0000000"
            "     0.5000000     0.1000000    0\n"
            "     0.2000000     0.9000000     0.1000000"
            "     0.4000000     0.2000000    1\n")
        return p

    def test_irc_is_rejected_and_left_untouched(self, tmp_path):
        from uacpy.io.refl_io import dedupe_reflection_file
        from uacpy.core.exceptions import FileFormatError
        irc = self._irc(tmp_path)
        before = irc.read_text()
        with pytest.raises(FileFormatError, match='row count'):
            dedupe_reflection_file(irc)
        assert irc.read_text() == before

    def test_brc_duplicate_angles_are_collapsed(self, tmp_path):
        from uacpy.io.refl_io import dedupe_reflection_file
        brc = tmp_path / 't.brc'
        brc.write_text("   4\n  0.0 1.0 0.0\n  0.0 1.0 0.0\n"
                       "  30.0 0.8 10.0\n  60.0 0.5 20.0\n")
        dedupe_reflection_file(brc)
        rows = [ln.split() for ln in brc.read_text().splitlines() if ln.strip()]
        assert int(rows[0][0]) == 3
        assert [float(r[0]) for r in rows[1:]] == [0.0, 30.0, 60.0]


class TestSourceBeamPatternRoundTrip:
    """``.sbp`` levels are dB on disk (``beampattern.f90:59`` converts to a
    linear amplitude only after reading), and dB is what
    :attr:`uacpy.Source.beam_pattern` carries."""

    def test_write_then_read_returns_the_written_dB(self, tmp_path):
        from uacpy.io.refl_io import (
            read_source_beam_pattern, write_source_beam_pattern)
        angles = np.array([-90.0, -30.0, 0.0, 30.0, 90.0])
        levels = np.array([-20.0, -6.0, 0.0, -6.0, -20.0])
        path = tmp_path / 'beam.sbp'
        write_source_beam_pattern(path, angles, levels)
        back = read_source_beam_pattern(path)
        assert np.allclose(back[:, 0], angles)
        assert np.allclose(back[:, 1], levels)

    def test_root_name_without_extension_resolves(self, tmp_path):
        from uacpy.io.refl_io import (
            read_source_beam_pattern, write_source_beam_pattern)
        write_source_beam_pattern(tmp_path / 'beam.sbp', np.array([-10.0, 10.0]),
                                  np.array([-3.0, -6.0]))
        back = read_source_beam_pattern(tmp_path / 'beam')
        assert np.allclose(back[:, 1], [-3.0, -6.0])


class TestStagedBeamPatternNeedsTwoRows:
    """A degenerate ``.sbp`` passes every guard on both sides.

    ``_require_strictly_increasing`` returns early for ``size <= 1``, and
    ``misc/monotonicMod.f90:30`` pre-sets ``.TRUE.`` then returns for ``N == 1``
    (for ``N == 0`` its ``ANY`` runs over zero-size sections and is also false),
    so ``misc/beampattern.f90:56`` passes it too. The engines then index the
    table as a pair — ``Bellhop/bellhop.f90:270`` clamps ``IBP`` to
    ``NSBPPts - 1`` and reads below the bound allocated at
    ``beampattern.f90:36``, and ``KrakenField/field.f90:203-209`` brackets with
    ``x(iseg + 1)``. Bellhop then yields an all-NaN field and field.exe a finite
    but wrong one, both at exit code 0 with nothing in the print file.

    Staging is the single boundary every pattern crosses, so guarding there
    covers the path form and the array form alike.
    """

    def _stage(self, tmp_path, text):
        from uacpy.io.refl_io import stage_source_beam_pattern
        src = tmp_path / 'pattern.sbp'
        src.write_text(text)
        return stage_source_beam_pattern(src, tmp_path / 'staged.sbp')

    def test_one_row_is_refused(self, tmp_path):
        with pytest.raises(ConfigurationError, match='at least 2'):
            self._stage(tmp_path, "1\n0.00 0.000000\n")

    def test_zero_rows_is_refused(self, tmp_path):
        with pytest.raises(ConfigurationError, match='at least 2'):
            self._stage(tmp_path, "0\n")

    def test_two_rows_is_accepted(self, tmp_path):
        self._stage(tmp_path, "2\n-90.0 0.0\n90.0 0.0\n")
        assert (tmp_path / 'staged.sbp').exists()

    def test_a_repeated_angle_is_refused(self, tmp_path):
        """The strictly-increasing guard must survive alongside the row-count
        one — a repeated abscissa is the division-by-zero case."""
        with pytest.raises(ConfigurationError,
                           match='must be strictly increasing'):
            self._stage(tmp_path, "3\n-90.0 0.0\n0.0 0.0\n0.0 -3.0\n")


class TestReflectionTablesAreNeverEditedInPlace:
    """``stage_reflection_file`` copies a table next to the ``.env`` that
    names it and then normalises the copy. When the source already *is* the
    destination — a pinned ``work_dir`` holding the caller's own ``.brc`` —
    there is no copy, and normalising would edit an input."""

    TABLE = ("4\n 10.0 0.50 -175.0\n 20.0 0.55 -170.0\n"
             " 20.0 0.56 -169.0\n 30.0 0.60 -160.0\n")

    def test_a_table_at_the_destination_is_left_byte_identical(self, tmp_path):
        from uacpy.io.refl_io import stage_reflection_file
        src = tmp_path / 'deck.brc'
        src.write_text(self.TABLE)
        before = hashlib.sha256(src.read_bytes()).hexdigest()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            dest = stage_reflection_file(src, tmp_path / 'deck.env',
                                         boundary='bottom')
        assert dest == src
        assert hashlib.sha256(src.read_bytes()).hexdigest() == before

    def test_a_repeated_angle_at_the_destination_is_reported(self, tmp_path):
        from uacpy.io.refl_io import stage_reflection_file
        src = tmp_path / 'deck.brc'
        src.write_text(self.TABLE)
        with pytest.warns(UserWarning, match='bellhopcuda'):
            stage_reflection_file(src, tmp_path / 'deck.env',
                                  boundary='bottom')

    def test_a_clean_table_at_the_destination_warns_about_nothing(
            self, tmp_path):
        from uacpy.io.refl_io import stage_reflection_file
        src = tmp_path / 'deck.brc'
        src.write_text("2\n 10.0 0.5 -175.0\n 20.0 0.6 -170.0\n")
        with recorded_warnings() as caught:
            stage_reflection_file(src, tmp_path / 'deck.env',
                                  boundary='bottom')
        assert [str(w.message) for w in caught] == []

    @staticmethod
    def _stage_two_row_table(tmp_path, phase_deg_step):
        from uacpy.io.refl_io import stage_reflection_file
        src = tmp_path / 'deck.brc'
        src.write_text(f"2\n 10.0 0.5 0.0\n 20.0 0.6 {phase_deg_step:.1f}\n")
        with recorded_warnings() as caught:
            stage_reflection_file(src, tmp_path / 'deck.env',
                                  boundary='bottom')
        return [str(w.message) for w in caught]

    @pytest.mark.parametrize('step', [180.5, -180.5, 315.6])
    def test_a_phase_step_past_a_half_turn_at_the_destination_is_reported(
            self, tmp_path, step):
        """The engine interpolates phi linearly between bracketing rows
        (misc/RefCoef.f90:119 assumes an unwrapped column), so a wrapped
        step is swept the long way round. The copy path unwraps; the
        in-place path can only say so."""
        messages = self._stage_two_row_table(tmp_path, step)
        assert len(messages) == 1
        assert 'unwrap' in messages[0]
        assert 'staged unmodified' in messages[0]

    @pytest.mark.parametrize('step', [180.0, -180.0, 179.5])
    def test_a_phase_step_within_a_half_turn_is_not_reported(
            self, tmp_path, step):
        assert self._stage_two_row_table(tmp_path, step) == []

    def test_a_copied_table_is_normalised(self, tmp_path):
        """The copy is what earns the right to rewrite."""
        from uacpy.io.refl_io import (
            stage_reflection_file, read_reflection_coefficient)
        src = tmp_path / 'user.brc'
        src.write_text(self.TABLE)
        before = hashlib.sha256(src.read_bytes()).hexdigest()
        dest = stage_reflection_file(src, tmp_path / 'deck.env',
                                     boundary='bottom')
        assert dest != src
        assert hashlib.sha256(src.read_bytes()).hexdigest() == before
        assert len(read_reflection_coefficient(dest).angles) == 3


class TestReflectionTableAngleOrderIsChecked:
    """``dedupe_reflection_file`` keeps the rows whose angle exceeds the last
    kept one, which collapses a descending table to a single row — one
    constant reflection coefficient at every angle, and exit code 0.
    ``misc/RefCoef.f90`` applies no monotonicity test of its own, so such a
    file reaches the engine intact and must be caught here, on the terms
    ``read_reflection_coefficient`` already states."""

    def test_a_descending_table_is_a_typed_error(self, tmp_path):
        from uacpy.io.refl_io import dedupe_reflection_file
        p = tmp_path / 'grazing.brc'
        original = ("5\n 90.0 0.9 -175.0\n 70.0 0.8 -170.0\n"
                    " 50.0 0.7 -160.0\n 30.0 0.6 -150.0\n 10.0 0.5 -140.0\n")
        p.write_text(original)
        with pytest.raises(FileFormatError, match='non-decreasing'):
            dedupe_reflection_file(p)
        assert p.read_text() == original

    def test_the_reader_and_the_dedupe_agree(self, tmp_path):
        from uacpy.io.refl_io import (
            dedupe_reflection_file, read_reflection_coefficient)
        p = tmp_path / 'grazing.brc'
        p.write_text("2\n 90.0 0.9 -175.0\n 10.0 0.5 -140.0\n")
        with pytest.raises(FileFormatError,
                           match='angles must be non-decreasing'):
            read_reflection_coefficient(p)
        with pytest.raises(FileFormatError,
                           match='angles must be non-decreasing'):
            dedupe_reflection_file(p)

    def test_a_repeated_angle_is_collapsed(self, tmp_path):
        from uacpy.io.refl_io import (
            dedupe_reflection_file, read_reflection_coefficient)
        p = tmp_path / 'evan.brc'
        p.write_text("3\n 0.0 1.0 180.0\n 0.0 1.0 180.0\n 10.0 0.9 170.0\n")
        dedupe_reflection_file(p)
        assert read_reflection_coefficient(p).angles.tolist() == [0.0, 10.0]


class TestATruncatedReflectionTableIsNeverNormalisedIntoAValidOne:
    """``dedupe_reflection_file`` runs over every table
    ``stage_reflection_file`` copies, so a shortfall it repairs never reaches
    ``read_reflection_coefficient``, which would have raised. Bellhop reads
    the repaired file without complaint and every grazing angle past the cut
    then falls outside the tabulated domain, where
    ``InterpolateReflectionCoefficient`` returns ``RInt%R = 0`` — a totally
    absorbing bottom — with its warning commented out
    (misc/RefCoef.f90:144-149)."""

    def test_a_short_table_is_refused_rather_than_renumbered(self, tmp_path):
        from uacpy.io.refl_io import dedupe_reflection_file
        p = tmp_path / 'trunc.brc'
        rows = [f"{0.02 * i:.6f} 0.5 180.0" for i in range(3000)]
        p.write_text("4500\n" + "\n".join(rows) + "\n")
        with pytest.raises(FileFormatError, match='4500'):
            dedupe_reflection_file(p)

    def test_the_refused_file_is_left_byte_identical(self, tmp_path):
        from uacpy.io.refl_io import dedupe_reflection_file
        p = tmp_path / 'trunc.brc'
        text = "3\n 10.0 0.5 -175.0\n 20.0 0.6 -170.0\n"
        p.write_text(text)
        with pytest.raises(FileFormatError, match='it is truncated'):
            dedupe_reflection_file(p)
        assert p.read_text() == text

    def test_the_reader_agrees_the_file_is_broken(self, tmp_path):
        """The two must not disagree: before the guard, the same file raised
        on a raw read and passed cleanly after dedupe."""
        from uacpy.io.refl_io import (dedupe_reflection_file,
                                      read_reflection_coefficient)
        p = tmp_path / 'trunc.brc'
        p.write_text("3\n 10.0 0.5 -175.0\n 20.0 0.6 -170.0\n")
        with pytest.raises(
                FileFormatError,
                match=r'file ended while reading 3 \(theta, R, phi\) records'):
            read_reflection_coefficient(p)
        with pytest.raises(FileFormatError, match='it is truncated'):
            dedupe_reflection_file(p)

    def test_tokens_past_the_declared_count_are_ignored(self, tmp_path):
        """``RefCoef.f90:53`` consumes exactly 3*N values and stops, so a
        trailing row is surplus, not corruption."""
        from uacpy.io.refl_io import dedupe_reflection_file
        p = tmp_path / 'extra.brc'
        p.write_text("2\n0.0 1.0 180.0\n1.0 0.5 170.0\n2.0 0.4 160.0\n")
        dedupe_reflection_file(p)
        tokens = p.read_text().split()
        assert tokens[0] == '2'
        assert '2.0' not in tokens

    def test_a_complete_table_dedupes(self, tmp_path):
        from uacpy.io.refl_io import dedupe_reflection_file
        p = tmp_path / 'ok.brc'
        rows = ["0.0 1.0 180.0"] * 3 + [f"{i}.0 0.5 170.0" for i in (1, 2)]
        p.write_text(f"{len(rows)}\n" + "\n".join(rows) + "\n")
        dedupe_reflection_file(p)
        assert int(p.read_text().split()[0]) == 3


class TestSourceBeamPatternNeedsTwoRows:
    """``bellhop.f90:270`` clamps ``IBP`` to ``NSBPPts - 1`` and then reads
    ``SrcBmPat(IBP+1, :)``, so a table shorter than two rows is indexed past
    the allocation at ``beampattern.f90:36`` and the run ends at exit 0 with
    an all-NaN field. ``stage_source_beam_pattern`` already refuses the path
    form of that table; the array form refuses it too, so the guard does
    not depend on how the caller supplied the pattern."""

    def test_a_one_row_pattern_raises_configurationerror(self, tmp_path):
        from uacpy.io.refl_io import write_source_beam_pattern
        with pytest.raises(ConfigurationError, match='at least 2'):
            write_source_beam_pattern(tmp_path / 'one.sbp', np.array([0.0]),
                                      np.array([-3.0]))
        assert not (tmp_path / 'one.sbp').exists()

    def test_an_empty_pattern_raises_configurationerror(self, tmp_path):
        from uacpy.io.refl_io import write_source_beam_pattern
        with pytest.raises(ConfigurationError, match='0 angle'):
            write_source_beam_pattern(tmp_path / 'none.sbp', np.array([]),
                                      np.array([]))

    def test_a_two_row_pattern_round_trips(self, tmp_path):
        from uacpy.io.refl_io import (read_source_beam_pattern,
                                      write_source_beam_pattern)
        p = tmp_path / 'two.sbp'
        write_source_beam_pattern(p, np.array([-10.0, 10.0]),
                                  np.array([-3.0, -6.0]))
        back = read_source_beam_pattern(p)
        assert np.allclose(back[:, 0], [-10.0, 10.0])
        assert np.allclose(back[:, 1], [-3.0, -6.0])

    def test_a_repeated_angle_raises_configurationerror(self, tmp_path):
        from uacpy.io.refl_io import write_source_beam_pattern
        with pytest.raises(ConfigurationError, match='strictly increasing'):
            write_source_beam_pattern(tmp_path / 'dup.sbp',
                                      np.array([0.0, 0.0, 1.0]), np.zeros(3))

    def test_a_step_of_exactly_the_angle_resolution_is_written(self, tmp_path):
        # %12.6f prints 0.000000 and 0.000001 as two distinct tokens, so a
        # step of exactly SBP_ANGLE_RESOLUTION_DEG still reaches Bellhop as a
        # strictly increasing column. The guard rejects steps *below* it.
        from uacpy.io.refl_io import write_source_beam_pattern
        out = tmp_path / 'edge.sbp'
        write_source_beam_pattern(out, np.array([0.0, 1e-6]),
                                  np.array([0.0, -3.0]))
        assert [ln.split()[0] for ln in out.read_text().splitlines()[1:]] == \
            ['0.000000', '0.000001']

    def test_a_step_below_the_angle_resolution_raises_configurationerror(
            self, tmp_path):
        # Whether a sub-resolution pair actually collides depends on where it
        # falls on the grid (1e-7 and 2e-7 both print 0.000000), so the guard
        # rejects the step size itself rather than the printed pair.
        from uacpy.io.refl_io import write_source_beam_pattern
        with pytest.raises(ConfigurationError, match='angle resolution'):
            write_source_beam_pattern(tmp_path / 'sub.sbp',
                                      np.array([0.0, 9e-7]),
                                      np.array([0.0, -3.0]))


class TestSourceBeamPatternRequiresOneLevelPerAngle:
    """The row loop is driven by ``len(angles)``, so an over-long ``pattern``
    would be truncated into a table Bellhop reads happily while it carries a
    different directivity than the caller passed, and a short one would raise
    ``IndexError`` partway through the write with the truncated file already on
    disk. Both columns are refused before ``open``."""

    def test_a_longer_pattern_than_angles_raises_configurationerror(
            self, tmp_path):
        from uacpy.io.refl_io import write_source_beam_pattern
        out = tmp_path / 'long.sbp'
        with pytest.raises(ConfigurationError, match='same shape'):
            write_source_beam_pattern(out, np.array([-90.0, 0.0, 90.0]),
                                      np.array([0.0, -3.0, 0.0, -99.0, -99.0]))
        assert not out.exists()

    def test_a_shorter_pattern_than_angles_raises_configurationerror(
            self, tmp_path):
        from uacpy.io.refl_io import write_source_beam_pattern
        out = tmp_path / 'short.sbp'
        with pytest.raises(ConfigurationError, match='same shape'):
            write_source_beam_pattern(out, np.array([-90.0, 0.0, 90.0]),
                                      np.array([0.0, -3.0]))
        assert not out.exists()

    def test_one_level_per_angle_is_written(self, tmp_path):
        # The boundary either side of the length guard: N and N are written,
        # N and N-1 / N and N+1 are refused by the two tests above.
        from uacpy.io.refl_io import (read_source_beam_pattern,
                                      write_source_beam_pattern)
        out = tmp_path / 'match.sbp'
        write_source_beam_pattern(out, np.array([-90.0, 0.0, 90.0]),
                                  np.array([0.0, -3.0, 0.0]))
        back = read_source_beam_pattern(out)
        assert back.shape == (3, 2)
        assert np.allclose(back[:, 1], [0.0, -3.0, 0.0])

    def test_a_column_shaped_pattern_raises_configurationerror(self, tmp_path):
        # (N, 1) formats as an array in the row loop rather than a float.
        from uacpy.io.refl_io import write_source_beam_pattern
        out = tmp_path / 'col.sbp'
        with pytest.raises(ConfigurationError, match='same shape'):
            write_source_beam_pattern(out, np.array([-90.0, 0.0, 90.0]),
                                      np.array([[0.0], [-3.0], [0.0]]))
        assert not out.exists()

    def test_a_non_numeric_pattern_raises_configurationerror(self, tmp_path):
        from uacpy.io.refl_io import write_source_beam_pattern
        out = tmp_path / 'text.sbp'
        with pytest.raises(ConfigurationError, match='not a numeric array'):
            write_source_beam_pattern(out, np.array([-90.0, 0.0, 90.0]),
                                      ['a', 'b', 'c'])
        assert not out.exists()


class TestTheReaderReturnsTheReflectionResult:
    """``read_reflection_coefficient`` returns a
    :class:`~uacpy.ReflectionCoefficient`, the type Bounce and OASR return,
    so a table read from disk exports and plots as a computed one does."""

    def test_a_written_table_reads_back_as_the_same_result(self, tmp_path):
        from uacpy import ReflectionCoefficient
        from uacpy.io.refl_io import (read_reflection_coefficient,
                                      write_reflection_coefficient)
        table = ReflectionCoefficient(angles=[0.0, 30.0, 60.0, 90.0],
                                      magnitude=[1.0, 0.8, 0.5, 0.25],
                                      phase=[3.0, 2.0, 1.0, 0.5])
        path = tmp_path / 'rt.brc'
        write_reflection_coefficient(path, table.angles, table.coefficient)
        back = read_reflection_coefficient(path)
        assert type(back) is ReflectionCoefficient
        assert np.array_equal(back.angles, table.angles)
        np.testing.assert_allclose(back.magnitude, table.magnitude, atol=1e-6)
        np.testing.assert_allclose(back.phase, table.phase, atol=1e-7)
        assert back.phase_reference == table.phase_reference
        assert back.frequencies is None
        frame = back.to_dataframe()
        assert list(frame.columns) == ['angle', 'magnitude', 'phase']
        assert frame['angle'].tolist() == [0.0, 30.0, 60.0, 90.0]

    def test_a_negative_magnitude_is_a_format_error_naming_the_file(
            self, tmp_path):
        """The result refuses a negative ``magnitude`` as a bad argument; read from
        a file it is the file that is wrong."""
        from uacpy.io.refl_io import read_reflection_coefficient
        p = tmp_path / 'signed.brc'
        p.write_text("2\n0.0 -0.5 0.0\n10.0 0.9 170.0\n")
        with pytest.raises(FileFormatError,
                           match=r'signed\.brc: ReflectionCoefficient: magnitude '
                                 r'holds a negative value'):
            read_reflection_coefficient(p)

    def test_a_zero_magnitude_reads(self, tmp_path):
        from uacpy.io.refl_io import read_reflection_coefficient
        p = tmp_path / 'zero.brc'
        p.write_text("2\n0.0 0.0 0.0\n10.0 0.9 170.0\n")
        assert read_reflection_coefficient(p).magnitude.tolist() == [0.0, 0.9]
