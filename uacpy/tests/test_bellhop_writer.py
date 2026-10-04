"""Bellhop env-writer coverage: what the deck carries, and what it warns about.

Every range-dependent boundary feature needs a file to travel in, and the
``.env`` alone carries none of them. Three invariants: a range-dependent
bottom reaches Bellhop even when the bathymetry is flat (the ``.bty`` is the
only vehicle); the long-format ``.bty`` samples the union of the bathymetry
and property grids, so interior property breaks survive; and a single-sample
(constant-offset) altimetry still produces an ``.ati``.

The last class covers the range-dependent-SSP warning the writer raises. The
writer runs no binary, so these stay in the pure-Python subset — which is why
they live here rather than beside the model tests in ``test_bellhop.py``,
whose module-level ``requires_binary`` mark would take them out of it.
"""

import pathlib
import tempfile
import warnings

import numpy as np
import pytest

import uacpy
from uacpy.core.environment import Bottom, BoundaryProperties, SeabedColumn
from uacpy.core.exceptions import ConfigurationError
from uacpy.io.bellhop_writer import write_bellhop_env_file
from uacpy.core import Environment
from uacpy.io.oalib_writer import write_ssp_section
import io
from uacpy.tests.conftest import warning_messages

_RD_SSP = 'range-dependent SSP'


def _write(tmp_path, env, ranges_max=20000.0):
    src = uacpy.Source(depths=25.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]),
                         ranges=np.linspace(1000.0, ranges_max, 10))
    path = tmp_path / 'case.env'
    write_bellhop_env_file(path, env, src, rcv)
    return path


def _rd_bottom(ranges, speeds):
    return Bottom.from_halfspaces(
        ranges,
        sound_speed=np.asarray(speeds, dtype=float),
        density=np.full(len(ranges), 1.8),
        attenuation=np.full(len(ranges), 0.5),
        shear_speed=np.zeros(len(ranges)),
        shear_attenuation=np.zeros(len(ranges)),
    )


def test_flat_bathy_rd_bottom_writes_long_bty(tmp_path):
    # Flat 200 m bathymetry + sand→mud transition at 5 km: the transition
    # must reach Bellhop via a long-format .bty, not be silently dropped.
    env = uacpy.Environment(
        bathymetry=200.0, ssp=1500.0,
        bottom=_rd_bottom([0.0, 5000.0], [1800.0, 1600.0]))
    path = _write(tmp_path, env)
    bty = path.with_suffix('.bty')
    assert bty.exists()
    text = bty.read_text()
    assert text.splitlines()[0].strip("'").endswith('L')   # long format
    assert '1800.000' in text and '1600.000' in text
    # BOT line carries the '~' bathymetry flag
    env_text = path.read_text()
    assert "'A~'" in env_text or "~'" in env_text


def test_long_bty_preserves_interior_property_breaks(tmp_path):
    # 2-point bathymetry (0, 20 km) + columns at 0, 8 and 16 km: the .bty
    # carries a row at each switch (midway between columns, 4 and 12 km)
    # holding the column to its right, so the interior steps are not lost
    # between the two bathymetry points.
    bathy = [(0.0, 200.0), (20000.0, 200.0)]
    env = uacpy.Environment(
        bathymetry=bathy, ssp=1500.0,
        bottom=_rd_bottom([0.0, 8000.0, 16000.0],
                          [1800.0, 1550.0, 2000.0]))
    path = _write(tmp_path, env)
    rows = [ln.split() for ln in
            path.with_suffix('.bty').read_text().splitlines()[2:] if ln.strip()]
    ranges_km = [float(r[0]) for r in rows]
    cp_by_range = {float(r[0]): float(r[2]) for r in rows}
    assert ranges_km == [0.0, 4.0, 12.0, 20.0]
    assert cp_by_range[0.0] == pytest.approx(1800.0)
    assert cp_by_range[4.0] == pytest.approx(1550.0)
    assert cp_by_range[12.0] == pytest.approx(2000.0)


def test_single_sample_altimetry_writes_ati(tmp_path):
    # A constant -2 m surface offset (one altimetry sample) must produce an
    # .ati (expanded to a 2-point constant profile), not be silently ignored.
    env = uacpy.Environment(bathymetry=200.0, ssp=1500.0,
                            altimetry=[(0.0, -2.0)])
    path = _write(tmp_path, env)
    ati = path.with_suffix('.ati')
    assert ati.exists()
    rows = [ln.split() for ln in ati.read_text().splitlines()[2:] if ln.strip()]
    assert len(rows) >= 2
    # env altimetry is positive-up; .ati is positive-down → +2.0 everywhere.
    assert all(float(r[1]) == pytest.approx(2.0) for r in rows)


def _rd_ssp_env(ssp_range_max):
    ssp = uacpy.SoundSpeedProfile(
        depths=[0.0, 100.0],
        sound_speed=np.array([[1500.0, 1495.0], [1490.0, 1488.0]]),
        ranges=[0.0, ssp_range_max])
    return uacpy.Environment(
        bathymetry=100.0, ssp=ssp,
        bottom=uacpy.BoundaryProperties(
            acoustic_type='half-space', sound_speed=1600.0,
            density=1.5, attenuation=0.5))


def _write_rd_ssp_deck(ssp_range_max, receiver_range_max, work_dir, **kwargs):
    env = _rd_ssp_env(ssp_range_max)
    src = uacpy.Source(depths=25.0, frequencies=200.0)
    rcv = uacpy.Receiver(depths=np.linspace(1.0, 99.0, 20),
                         ranges=np.linspace(100.0, receiver_range_max, 30))
    path = pathlib.Path(work_dir) / 'model.env'
    write_bellhop_env_file(path, env, src, rcv, interp_ssp='quad', **kwargs)
    return path


class TestRangeDependentSSPWarning:
    def test_an_ssp_reaching_the_outermost_receiver_is_accepted(self):
        """The shape README's front-page transect has: the profiles stop
        exactly where the receivers do. Holding the last profile constant
        across the box margin beyond them is not a modelling choice the
        caller made — no receiver sits there."""
        with tempfile.TemporaryDirectory() as td:
            msgs = warning_messages(lambda: _write_rd_ssp_deck(5000.0, 5000.0, td),
                             _RD_SSP)
        assert msgs == []

    def test_a_receiver_one_metre_beyond_the_ssp_warns(self):
        with tempfile.TemporaryDirectory() as td:
            msgs = warning_messages(lambda: _write_rd_ssp_deck(5000.0, 5001.0, td),
                             _RD_SSP)
        assert len(msgs) == 1

    def test_an_ssp_one_metre_beyond_the_outermost_receiver_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            msgs = warning_messages(lambda: _write_rd_ssp_deck(5000.0, 4999.0, td),
                             _RD_SSP)
        assert msgs == []

    def test_the_remediation_names_the_receiver_range_it_asks_for(self):
        """A remediation asking for profiles "out to the receiver range" while
        the check tests against ``r_box`` leaves the warning standing when
        followed. The target named here is the one that silences it."""
        with tempfile.TemporaryDirectory() as td:
            msgs = warning_messages(lambda: _write_rd_ssp_deck(5000.0, 9000.0, td),
                             _RD_SSP)
        assert len(msgs) == 1
        assert 'out to at least 9000 m' in msgs[0]
        assert 'r_box=' in msgs[0]
        assert '10800 m' in msgs[0]

    def test_following_the_remediation_silences_the_warning(self):
        with tempfile.TemporaryDirectory() as td:
            assert len(warning_messages(
                lambda: _write_rd_ssp_deck(5000.0, 9000.0, td), _RD_SSP)) == 1
            assert warning_messages(
                lambda: _write_rd_ssp_deck(9000.0, 9000.0, td), _RD_SSP) == []

    def test_the_ssp_file_extends_past_the_ray_box(self):
        """The padding is a deck requirement (a ray landing on the last SSP
        range is flagged outside the sound-speed box), so suppressing the
        warning must not suppress it. ``.ssp`` ranges are written in km."""
        with tempfile.TemporaryDirectory() as td:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                path = _write_rd_ssp_deck(5000.0, 5000.0, td)
            columns = [float(x) for x
                       in path.with_suffix('.ssp').read_text().splitlines()[1].split()]
        r_box_km = 1.2 * 5000.0 / 1000.0
        assert max(columns) > r_box_km
        assert min(columns) < -r_box_km


class TestBellhopHalfSpaceMatchesItsSiblings:
    """``ReadEnvironmentBell.f90:474`` reads the bottom half-space row with
    a list-directed ``READ``, so no column width is imposed (the ``F10.2`` at
    ``:475`` is the ``.prt`` echo). Writing it at ``.2f`` while every other
    AT writer uses ``.6f`` hands Bellhop and Kraken different seabeds from
    one ``Environment``."""

    @staticmethod
    def _env():
        hs = BoundaryProperties(acoustic_type='half-space',
                                sound_speed=1572.348, shear_speed=112.607,
                                density=1.4449, attenuation=0.23,
                                shear_attenuation=0.11)
        return uacpy.Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=Bottom(columns=[SeabedColumn(layers=[], halfspace=hs)]))

    def test_the_half_space_row_keeps_six_decimals(self, tmp_path):
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        out = tmp_path / 'deck.env'
        write_bellhop_env_file(
            out, self._env(), uacpy.Source(depths=50.0, frequencies=100.0),
            uacpy.Receiver(depths=[20.0, 40.0],
                           ranges=np.linspace(100.0, 1000.0, 5)))
        row = [ln for ln in out.read_text().splitlines()
               if ln.strip().startswith('100.000000  1572')]
        assert row == [' 100.000000  1572.348000 112.607000 1.444900 '
                       '0.230000 0.110000 /']

    def test_the_deck_round_trips_the_boundary_properties_exactly(
            self, tmp_path):
        """The AT bottom section (``write_bottom_section``) is the reference:
        both writers must hand the Fortran the numbers the Environment
        carries, not a rounding of them."""
        import io as _io
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        from uacpy.io.oalib_writer import write_bottom_section
        from uacpy.core.boundary import BoundaryType
        env = self._env()
        bell = tmp_path / 'deck.env'
        write_bellhop_env_file(
            bell, env, uacpy.Source(depths=50.0, frequencies=100.0),
            uacpy.Receiver(depths=[20.0, 40.0],
                           ranges=np.linspace(100.0, 1000.0, 5)))

        buf = _io.StringIO()
        write_bottom_section(buf, env, bottom_type=BoundaryType.HALF_SPACE,
                             filepath=tmp_path / 'k.env',
                             halfspace_depth=100.0)

        def halfspace_numbers(text):
            for line in text.splitlines():
                fields = line.replace('/', '').split()
                if len(fields) == 6 and fields[1].startswith('1572'):
                    return [float(v) for v in fields[1:]]
            raise AssertionError(f'no half-space row in\n{text}')

        hs = env.bottom.halfspace_at(range=0.0)
        expected = [hs.sound_speed, hs.shear_speed, hs.density,
                    hs.attenuation, hs.shear_attenuation]
        assert halfspace_numbers(bell.read_text()) == expected
        assert halfspace_numbers(buf.getvalue()) == expected


class TestOneSeabedDescribesOneSeabed:
    """``write_bty_long_format`` and the ``.env`` writers are handed the same
    ``BoundaryProperties``; ``bdryMod.f90:200-201`` reads the long ``.bty``
    row list-directed, so nothing obliges the two to round it differently."""

    def test_the_long_bty_row_keeps_the_precision_the_env_keeps(self,
                                                                tmp_path):
        from uacpy.io.bathy_io import write_bty_long_format
        hs = BoundaryProperties(sound_speed=1572.348, density=1.4449,
                                attenuation=0.5, shear_speed=112.607,
                                shear_attenuation=0.1)
        bottom = Bottom(columns=[SeabedColumn(layers=[], halfspace=hs),
                                 SeabedColumn(layers=[], halfspace=hs)],
                        ranges=[0.0, 1000.0])
        out = tmp_path / 'x.bty'
        write_bty_long_format(out, np.array([[0.0, 100.0], [1000.0, 120.0]]),
                              bottom)
        row = out.read_text().splitlines()[2].split()
        assert float(row[2]) == pytest.approx(1572.348, abs=5e-7)
        assert float(row[3]) == pytest.approx(112.607, abs=5e-7)
        # 1.4449 g/cm3 is a value a %.3f column would round to 1.445, so it
        # separates the written width from a three-decimal one.
        assert float(row[4]) == pytest.approx(1.4449, abs=5e-7)


class TestAltimetryFilesRoundTripInTheFileConvention:
    """``write_ati_file`` documents its input as positive-DOWN — Bellhop's z
    axis — and ``read_altimetry`` returns the values as the file carries them.
    Neither negates, so the pair is an identity and the negation belongs to
    whoever owns the public positive-up convention (``bellhop_writer`` does
    it for ``Environment(altimetry=…)``)."""

    def test_a_trough_read_back_is_a_trough(self, tmp_path):
        from uacpy.io.bathy_io import read_altimetry, write_ati_file
        p = tmp_path / 's.ati'
        # +2 m is a surface 2 m BELOW MSL in this convention.
        write_ati_file(p, np.array([[0.0, 2.0], [1000.0, -3.0]]))
        ati = read_altimetry(p)
        assert ati.depths[0] == pytest.approx(2.0)
        assert ati.depths[1] == pytest.approx(-3.0)

    def test_the_bellhop_writer_owns_the_negation(self, tmp_path):
        """A crest 2 m above MSL in the public convention has to reach the
        ``.ati`` as -2, or ``bdryMod.f90:113-114`` sees the surface below the
        SSP's first depth instead of above it."""
        from uacpy.io.bathy_io import read_altimetry
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        env = uacpy.Environment(bathymetry=100.0, ssp=1500.0,
                                altimetry=[(0.0, 2.0), (2000.0, 2.0)])
        out = tmp_path / 'r.env'
        write_bellhop_env_file(str(out), env, _source(),
                               uacpy.Receiver(depths=[50.0], ranges=[1000.0]))
        ati = read_altimetry(out.with_suffix('.ati'))
        assert ati.depths[0] == pytest.approx(-2.0)


def _source():
    return uacpy.Source(depths=50.0, frequencies=100.0)


class TestBoundaryFilesRespectTheirOwnKmColumn:
    """The range axis is validated in METRES but written in km at ``%.6f``, so
    the file's own resolution is 1e-6 km = 1 mm. Two metre-domain ranges closer
    than that passed validation and landed on the same token:
    ``[[0,200],[5000.0,190],[5000.0004,180],[10000,200]]`` emitted two
    ``5.000000`` rows and ``bellhop.exe`` stopped with "Bathymetry ranges are
    not monotonically increasing" (``bdryMod.f90:230`` ->
    ``monotonicMod.f90:32``) — at exit 0 with no ``.shd``, because
    ``misc/FatalError.f90:30`` is ``STOP '<string>'``.

    An empty boundary is refused for a related reason: a file declaring 0
    points made ``bellhop.exe`` print "Number of bathymetry points = 0",
    terminate every beam and write an all-zero ``.shd`` at exit 0.
    """

    def test_ranges_a_millimetre_apart_are_written(self, tmp_path):
        from uacpy.io.bathy_io import write_bty_file
        write_bty_file(tmp_path / 'ok.bty',
                       np.array([[0.0, 200.0], [5000.0, 190.0],
                                 [5001.0, 180.0], [10000.0, 200.0]]))
        assert (tmp_path / 'ok.bty').exists()

    def test_ranges_below_the_km_resolution_are_refused(self, tmp_path):
        from uacpy.io.bathy_io import write_bty_file
        with pytest.raises(ConfigurationError, match='5.000000 km'):
            write_bty_file(tmp_path / 'bad.bty',
                           np.array([[0.0, 200.0], [5000.0, 190.0],
                                     [5000.0004, 180.0], [10000.0, 200.0]]))

    def test_an_empty_boundary_is_refused(self, tmp_path):
        from uacpy.io.bathy_io import write_bty_file
        with pytest.raises(ConfigurationError, match='no points'):
            write_bty_file(tmp_path / 'empty.bty', np.zeros((0, 2)))

    def test_a_single_point_boundary_is_written(self, tmp_path):
        # bdryMod.f90:174,224-225 extends a boundary to +/- infinity itself,
        # so one point is a complete description.
        from uacpy.io.bathy_io import write_bty_file
        write_bty_file(tmp_path / 'one.bty', np.array([[0.0, 200.0]]))
        assert (tmp_path / 'one.bty').exists()


class TestTheLongFormatChecksTheAxisItWrites:
    """``write_bty_long_format`` writes the UNION of the bathymetry axis, the
    bottom's first node and the switches between its columns, not the
    bathymetry axis alone — and checks the axis it writes.

    Each carrier enforces its own 1 mm minimum step within itself
    (``Bottom.__post_init__``, ``_grid.py``), but nothing enforces one across
    them, and ``np.union1d`` de-dupes by exact float equality — so the same
    physical range arrived at by different arithmetic survives twice and both
    copies print one ``%.6f`` km token, after which ``bdryMod.f90:230`` ->
    ``monotonicMod.f90:32`` aborts "not monotonically increasing" and
    ``FatalError.f90`` STOPs at exit 0, leaving no ``.shd``. The bottom's
    first node is written as the user gave it; a switch the writer derives
    within 1 mm of an existing row is dropped instead.
    """

    @staticmethod
    def _carriers(bottom_ranges):
        from uacpy.core.bathymetry import Bathymetry
        from uacpy.core.bottom import Bottom
        bathy = Bathymetry(ranges=np.cumsum(np.full(16, 333.3)),
                           depths=np.linspace(200.0, 120.0, 16))
        n = len(bottom_ranges)
        bottom = Bottom.from_halfspaces(
            bottom_ranges, sound_speed=np.full(n, 1700.0),
            density=np.full(n, 1.8), attenuation=np.full(n, 0.5))
        return bathy, bottom

    def test_axes_that_collide_only_after_the_union_are_refused(self,
                                                                tmp_path):
        from uacpy.io.bathy_io import write_bty_long_format
        # The bottom's first node, 6 * 333.3, is the sixth bathymetry node
        # arrived at by a different sum: equal at six decimals, unequal as
        # floats (2.3e-13 m apart), so the union writes the token twice.
        bathy, bottom = self._carriers(np.arange(6, 17) * 333.3)
        with pytest.raises(ConfigurationError, match='merged range axis'):
            write_bty_long_format(tmp_path / 'bad.bty', bathy, bottom)

    def test_axes_that_agree_exactly_write(self, tmp_path):
        from uacpy.io.bathy_io import write_bty_long_format
        bathy, bottom = self._carriers(np.cumsum(np.full(16, 333.3)))
        write_bty_long_format(tmp_path / 'ok.bty', bathy, bottom)
        rows = [ln for ln in (tmp_path / 'ok.bty').read_text().splitlines()
                if ln.strip()]
        # type + count + 16 bathymetry rows + 15 switches between the columns
        assert len(rows) == 2 + 16 + 15


class TestTheOptionLetterAndTheAuxiliaryFileAreDecidedTogether:
    """Bellhop opens the ``.ati`` and the ``.bty`` by base-name convention,
    never by a path in the deck — what tells it to look is a single ``'~'``
    in ``TopOpt(5:5)`` / ``BotOpt(2:2)``.

    So the letter and the file are one decision, and the two failure modes it
    has are silent in opposite directions: a ``'~'`` with no file aborts the
    run in ``bdryMod``, and a file with no ``'~'`` is ignored and the run
    returns a flat-boundary answer that looks right."""

    @staticmethod
    def _quoted_records(path):
        """The deck's quoted records in order: title, TopOpt, BotOpt,
        RunType (``Bellhop/ReadEnvironmentBell.f90:46, :243, :93, :358``)."""
        return [ln.strip().split("'")[1] for ln in path.read_text().splitlines()
                if ln.strip().startswith("'") and ln.strip().count("'") >= 2]

    @classmethod
    def _top_option(cls, path):
        records = cls._quoted_records(path)
        assert len(records) >= 2, records
        return records[1]

    @classmethod
    def _bottom_option(cls, path):
        records = cls._quoted_records(path)
        assert len(records) >= 3, records
        return records[2]

    def _flat(self):
        return uacpy.Environment(bathymetry=200.0, ssp=1500.0)

    def _rd_bathy(self):
        from uacpy.core.bathymetry import Bathymetry
        return uacpy.Environment(
            bathymetry=Bathymetry(ranges=[0.0, 10000.0, 20000.0],
                                  depths=[200.0, 240.0, 180.0]),
            ssp=1500.0)

    def _rd_bottom_env(self):
        return uacpy.Environment(bathymetry=200.0, ssp=1500.0,
                                 bottom=_rd_bottom([0.0, 5000.0],
                                                   [1700.0, 1550.0]))

    @pytest.mark.parametrize('case,expects_bty', [
        ('_flat', False),
        ('_rd_bathy', True),
        ('_rd_bottom_env', True),
    ])
    def test_the_bty_marker_is_present_exactly_when_the_file_is(
            self, tmp_path, case, expects_bty):
        path = _write(tmp_path, getattr(self, case)())
        wrote = path.with_suffix('.bty').exists()
        assert wrote is expects_bty
        assert ('~' in self._bottom_option(path)) is expects_bty, \
            self._bottom_option(path)

    @pytest.mark.parametrize('heights,expects_ati', [
        (None, False),
        ([0.0, -2.0, 1.0], True),
    ], ids=['no-altimetry', 'range-dependent'])
    def test_the_ati_marker_is_present_exactly_when_the_file_is(
            self, tmp_path, heights, expects_ati):
        from uacpy.core.altimetry import Altimetry
        kwargs = {}
        if heights is not None:
            kwargs['altimetry'] = Altimetry(
                ranges=np.linspace(0.0, 20000.0, len(heights)),
                heights=heights)
        env = uacpy.Environment(bathymetry=200.0, ssp=1500.0, **kwargs)
        path = _write(tmp_path, env)
        wrote = path.with_suffix('.ati').exists()
        assert wrote is expects_ati
        assert (self._top_option(path)[4:5] == '~') is expects_ati, \
            self._top_option(path)

    def test_the_helpers_write_what_the_full_deck_writes(self, tmp_path):
        """The two extracted writers are the same code path the deck takes,
        so driving them directly must produce the same bytes."""
        from uacpy.core.altimetry import Altimetry
        from uacpy.io.bellhop_writer import (_write_altimetry_beside,
                                             _write_bathymetry_beside)
        env = uacpy.Environment(
            bathymetry=200.0, ssp=1500.0,
            bottom=_rd_bottom([0.0, 5000.0], [1700.0, 1550.0]),
            altimetry=Altimetry(ranges=[0.0, 10000.0, 20000.0],
                                heights=[0.0, -2.0, 1.0]))
        rcv = uacpy.Receiver(depths=np.array([50.0]),
                             ranges=np.linspace(1000.0, 20000.0, 10))
        from_deck = _write(tmp_path, env)

        direct = tmp_path / 'direct.env'
        _write_altimetry_beside(direct, env, rcv, interp_code='L',
                                verbose=False)
        wrote = _write_bathymetry_beside(direct, env, rcv, z_max=200.0,
                                         interp_code='L')
        assert wrote is True
        for suffix in ('.ati', '.bty'):
            assert (direct.with_suffix(suffix).read_bytes()
                    == from_deck.with_suffix(suffix).read_bytes()), suffix

    def test_a_flat_seabed_writes_no_bty_from_the_helper(self, tmp_path):
        # The other side of the helper's own boundary: it reports False and
        # leaves nothing on disk, which is what keeps the '~' off the deck.
        from uacpy.io.bellhop_writer import _write_bathymetry_beside
        env = self._flat()
        rcv = uacpy.Receiver(depths=np.array([50.0]),
                             ranges=np.linspace(1000.0, 20000.0, 10))
        target = tmp_path / 'flat.env'
        assert _write_bathymetry_beside(target, env, rcv, z_max=200.0,
                                        interp_code='L') is False
        assert not target.with_suffix('.bty').exists()


class TestTheWriterRefusesASingleBeamForInfluenceRuns:
    """bellhop.f90:176-178 leaves Dalpha = 0 for one beam: an empty field,
    empty arrivals or empty eigenrays at exit 0. The deck refuses it for every
    run type that evaluates influence; one beam is a ray-trace request."""

    def _write(self, tmp_path, n_beams, run_type):
        import uacpy
        from uacpy.core.source import Source
        from uacpy.core.receiver import Receiver
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        env = uacpy.Environment(name='w', bathymetry=100.0, ssp=1500.0)
        write_bellhop_env_file(
            filepath=tmp_path / 'w.env', env=env,
            source=Source(depths=50.0, frequencies=200.0),
            receiver=Receiver(depths=[50.0], ranges=[1000.0]),
            run_type=run_type, n_beams=n_beams)

    @pytest.mark.parametrize('run_type', ['C', 'A', 'E'])
    def test_one_beam_is_refused(self, tmp_path, run_type):
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match='single beam'):
            self._write(tmp_path, 1, run_type)

    def test_one_beam_is_allowed_for_a_ray_trace(self, tmp_path):
        self._write(tmp_path, 1, 'R')
        assert (tmp_path / 'w.env').exists()


class TestALongFormatBtyWritesTheNearestColumnStep:
    """A range-dependent half-space is a step function of range: each column
    holds up to the midpoint with its neighbour (``Bottom.halfspace_at``,
    nearest). Bellhop applies a ``.bty`` row's geoacoustics to the whole
    segment to its right (``bellhop.f90:478``), so the writer puts one row at
    every switch, carrying the column to the right of it, and no other row
    beyond the bathymetry nodes."""

    @staticmethod
    def _two_column_bottom():
        from uacpy import Bottom
        return Bottom.from_halfspaces(
            [0.0, 5000.0], sound_speed=[1600.0, 1800.0],
            density=[1.5, 1.9], attenuation=[0.3, 0.5])

    def test_two_columns_switch_at_the_midpoint_row(self, tmp_path):
        from uacpy.io.bathy_io import write_bty_long_format
        out = tmp_path / 'step.bty'
        write_bty_long_format(out, np.array([[0.0, 100.0], [5000.0, 100.0]]),
                              self._two_column_bottom())
        rows = np.loadtxt(out, skiprows=2)
        # Union of the bathymetry nodes and the one switch: no fill rows.
        assert np.allclose(rows[:, 0], [0.0, 2.5, 5.0])
        assert int(out.read_text().splitlines()[1]) == 3
        # Column 1 up to the 2.5 km row, column 2 from it.
        assert np.allclose(rows[:, 2], [1600.0, 1800.0, 1800.0])
        assert np.allclose(rows[:, 4], [1.5, 1.9, 1.9])
        assert np.allclose(rows[:, 5], [0.3, 0.5, 0.5])

    def test_a_bathymetry_node_between_the_columns_carries_its_own_side(
            self, tmp_path):
        from uacpy.io.bathy_io import write_bty_long_format
        out = tmp_path / 'step.bty'
        write_bty_long_format(
            out, np.array([[0.0, 100.0], [2000.0, 110.0], [3000.0, 120.0],
                           [5000.0, 100.0]]),
            self._two_column_bottom())
        rows = np.loadtxt(out, skiprows=2)
        assert np.allclose(rows[:, 0], [0.0, 2.0, 2.5, 3.0, 5.0])
        # Depth ramps between its nodes (Bellhop's 'L'); the seabed steps.
        assert np.allclose(rows[:, 1], [100.0, 110.0, 115.0, 120.0, 100.0])
        assert np.allclose(rows[:, 2], [1600.0, 1600.0, 1800.0, 1800.0, 1800.0])

    def test_range_independent_geoacoustics_keep_the_union_rows_only(self, tmp_path):
        from uacpy import Bottom
        from uacpy.io.bathy_io import write_bty_long_format
        r = np.array([0.0, 5000.0])
        bottom = Bottom.from_halfspaces(r, sound_speed=np.full(2, 1600.0),
                                        density=np.full(2, 1.8), attenuation=np.full(2, 0.5))
        out = tmp_path / 'flat.bty'
        write_bty_long_format(out, np.array([[0.0, 100.0], [2500.0, 110.0], [5000.0, 100.0]]), bottom)
        assert np.loadtxt(out, skiprows=2).shape[0] == 3

    @pytest.mark.requires_binary
    def test_bellhop_and_ram_agree_on_the_step_as_they_do_on_a_flat_seabed(
            self, tmp_path):
        """Both engines place the switch midway between the two columns: the
        Bellhop-vs-RAM spread on the step is no wider than the spread the
        same two engines show on a range-INDEPENDENT seabed of the same
        case, measured here rather than remembered."""
        from uacpy import (Environment, Source, Receiver, Bellhop, RAM,
                           RunMode, Bottom)

        receivers = Receiver(depths=[50.0],
                             ranges=[1000.0, 1500.0, 2000.0,
                                     3000.0, 3500.0, 4000.0])
        source = Source(depths=[30.0], frequencies=300.0)
        bathy = np.array([[0.0, 100.0], [5000.0, 100.0]])

        def spread(bottom, tag):
            env = Environment(bathymetry=bathy, ssp=1500.0, bottom=bottom)
            tl = []
            for model in (Bellhop(n_beams=400, beam_type='B',
                                  work_dir=tmp_path / f'{tag}-bh'),
                          RAM(verbose=False, work_dir=tmp_path / f'{tag}-ram')):
                field = model.run(env, source, receivers, RunMode.COHERENT_TL)
                tl.append(np.asarray(field.dB).ravel())
            diff = tl[0] - tl[1]
            return float(np.sqrt(np.mean(diff ** 2))), float(np.abs(diff).max())

        flat = Bottom.from_halfspaces(
            [0.0, 5000.0], sound_speed=[1600.0, 1600.0],
            density=[1.5, 1.5], attenuation=[0.3, 0.3])
        rms_flat, max_flat = spread(flat, 'flat')
        rms_step, max_step = spread(self._two_column_bottom(), 'step')
        # Cross-engine tolerance is what the flat case measures, with a
        # 2 dB allowance for the ray/PE treatment of the switch itself.
        assert rms_step <= rms_flat + 2.0, (rms_step, rms_flat)
        assert max_step <= max_flat + 3.0, (max_step, max_flat)


class TestBellhopWriterReflectionStaging:
    """``write_bellhop_env_file``'s user-error contracts and its ``.brc``/``.trc``
    staging, which has to tolerate a table already sitting beside the ``.env``."""

    @staticmethod
    def _sr():
        source = uacpy.Source(frequencies=100, depths=25)
        receiver = uacpy.Receiver(depths=np.array([50.0]),
                                  ranges=np.linspace(100.0, 5000.0, 20))
        return source, receiver

    @staticmethod
    def _brc(path):
        theta = np.linspace(0.0, 90.0, 91)
        with open(path, 'w') as fh:
            fh.write(f"{len(theta)}\n")
            for t in theta:
                fh.write(f"{t:12.6f}     0.500000     0.000000\n")
        return path

    @pytest.mark.parametrize('kwargs', [
        {'interp_bathymetry': 'bogus'},
        {'interp_altimetry': 'bogus'},
    ])
    def test_bad_geometry_interp_raises_configurationerror(self, tmp_path, kwargs):
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        env = uacpy.Environment(name='t', bathymetry=100, ssp=1500)
        source, receiver = self._sr()
        with pytest.raises(
                ConfigurationError,
                match="interp_(altimetry|bathymetry) must be 'linear' or 'curvilinear'"):
            write_bellhop_env_file(tmp_path / 'run.env', env, source, receiver,
                                   **kwargs)
        # The guard runs before the file opens, so a refused deck leaves no
        # truncated .env behind.
        assert not (tmp_path / 'run.env').exists()

    def test_refused_irregular_grid_leaves_no_truncated_env(self, tmp_path):
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        env = uacpy.Environment(name='t', bathymetry=100, ssp=1500)
        source, receiver = self._sr()   # 1 depth vs 20 ranges
        with pytest.raises(ConfigurationError, match="grid_type='I'"):
            write_bellhop_env_file(tmp_path / 'run.env', env, source, receiver,
                                   grid_type='I')
        assert not (tmp_path / 'run.env').exists()

    def test_bottom_file_without_reflection_file_raises_configurationerror(
            self, tmp_path):
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        env = uacpy.Environment(
            name='t', bathymetry=100, ssp=1500,
            bottom=uacpy.BoundaryProperties(acoustic_type='file'))
        source, receiver = self._sr()
        with pytest.raises(ConfigurationError, match='reflection_file'):
            write_bellhop_env_file(tmp_path / 'run.env', env, source, receiver)

    def test_brc_already_beside_the_env_is_kept(self, tmp_path):
        """A BOUNCE-produced ``run.brc`` in a pinned work_dir that Bellhop then
        writes ``run.env`` into: source and destination are the same file."""
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        brc = self._brc(tmp_path / 'run.brc')
        env = uacpy.Environment(
            name='t', bathymetry=100, ssp=1500,
            bottom=uacpy.BoundaryProperties(acoustic_type='file',
                                            reflection_file=str(brc)))
        source, receiver = self._sr()
        write_bellhop_env_file(tmp_path / 'run.env', env, source, receiver)
        assert brc.exists() and brc.stat().st_size > 0
        assert "'F'" in (tmp_path / 'run.env').read_text()

    def test_trc_already_beside_the_env_is_kept(self, tmp_path):
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        trc = self._brc(tmp_path / 'run.trc')
        env = uacpy.Environment(
            name='t', bathymetry=100, ssp=1500,
            surface=uacpy.BoundaryProperties(acoustic_type='file',
                                             reflection_file=str(trc)))
        source, receiver = self._sr()
        write_bellhop_env_file(tmp_path / 'run.env', env, source, receiver)
        assert trc.exists() and trc.stat().st_size > 0

    def test_surface_file_without_reflection_file_raises_configurationerror(
            self, tmp_path):
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        env = uacpy.Environment(
            name='t', bathymetry=100, ssp=1500,
            surface=uacpy.BoundaryProperties(acoustic_type='file'))
        source, receiver = self._sr()
        with pytest.raises(ConfigurationError, match='reflection_file'):
            write_bellhop_env_file(tmp_path / 'run.env', env, source, receiver)

    def test_missing_reflection_file_raises_configurationerror(self, tmp_path):
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        env = uacpy.Environment(
            name='t', bathymetry=100, ssp=1500,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='file',
                reflection_file=str(tmp_path / 'nope.brc')))
        source, receiver = self._sr()
        with pytest.raises(ConfigurationError, match='not found'):
            write_bellhop_env_file(tmp_path / 'run.env', env, source, receiver)


class TestATSSPPointLimit:
    """``misc/sspMod.f90:11`` dimensions every profile array to
    ``MaxSSP = 20001``. Its read loop counts per medium but writes at a
    cumulative index (``SSP%Loc`` accumulates at ``:325``), so the clean
    ``ERROUT`` at ``:368`` only fires for a single medium — with sediment
    layers the index runs off the end and gfortran dies inside the READ with
    an unrelated "Bad real number in item 1 of list input". Bellhop's private
    ``Bellhop/sspMod.f90:16`` holds 100001, so the same environment can be
    fine for Bellhop and fatal for Kraken. uacpy already guards the sibling AT
    limits (``MaxNfreq``, ``MaxBioLayers``, the mesh floor); this one was
    missed."""

    BOT = BoundaryProperties(acoustic_type='half-space', sound_speed=1800.0,
                             density=1.8, attenuation=0.5)

    def _env(self, n):
        z = np.linspace(0.0, 200.0, n)
        return Environment(bathymetry=200.0,
                           ssp=list(zip(z, 1500.0 + 0.01 * z)), bottom=self.BOT)

    def test_at_the_limit_writes(self):
        # The discriminating counterpart: 20001 is legal and must not be
        # refused. Off-by-one here would reject a valid environment.
        buf = io.StringIO()
        write_ssp_section(buf, self._env(20001), 200.0, ssp_topopt='C')
        assert buf.getvalue()

    @pytest.mark.parametrize('n', [20002, 25000])
    def test_over_the_limit_is_refused_with_the_reason(self, n):
        with pytest.raises(ConfigurationError, match='sspMod.f90:11'):
            write_ssp_section(io.StringIO(), self._env(n), 200.0,
                              ssp_topopt='C')

    def _submerged_env(self, n):
        z = np.linspace(10.0, 200.0, n)
        return Environment(bathymetry=200.0,
                           ssp=list(zip(z, 1500.0 + 0.01 * z)), bottom=self.BOT)

    def test_the_z0_guard_row_counts_toward_the_limit(self):
        # A profile starting under the surface gains a prepended z = 0 row,
        # so 20001 tabulated samples put 20002 rows in the deck.
        with pytest.raises(ConfigurationError, match='sspMod.f90:11'):
            write_ssp_section(io.StringIO(), self._submerged_env(20001),
                              200.0, ssp_topopt='C')

    def test_a_submerged_profile_at_the_limit_writes_with_its_guard_row(self):
        # The run announces the submerged start (models._checks
        # .warn_on_ssp_start); the writer only adds the guard row.
        buf = io.StringIO()
        write_ssp_section(buf, self._submerged_env(20000), 200.0,
                          ssp_topopt='C')
        rows = [ln for ln in buf.getvalue().splitlines() if ln.endswith('/')]
        assert len(rows) == 20001

    def test_bellhop_is_untouched(self):
        # Bellhop has its own reader and its own 100001 limit, so the guard
        # must not reach its writer.
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        import inspect
        assert 'reject_oversized_at_ssp' not in inspect.getsource(
            write_bellhop_env_file)


class TestBiologicalEdgesReachBothBellhopProfiles:
    """Bellhop evaluates a Biological law at its SSP nodes only
    (``Bellhop/sspMod.f90:906``), and a quad run pairs row ``iz`` of the
    ``.ssp`` matrix with node ``iz`` of the ``.env`` (``:427-431``), so the
    edge node pairs go into both, interpolated linearly in depth as
    ``Quad`` interpolates."""

    def test_the_env_rows_and_the_ssp_matrix_carry_the_same_pairs(
            self, tmp_path):
        env = _rd_ssp_env(20000.0)
        env.absorption = uacpy.Biological(
            layers=[(30.0, 70.0, 300.0, 4.0, 0.125)])
        path = tmp_path / 'model.env'
        src = uacpy.Source(depths=25.0, frequencies=200.0)
        rcv = uacpy.Receiver(depths=np.linspace(1.0, 99.0, 20),
                             ranges=np.linspace(100.0, 15000.0, 30))
        write_bellhop_env_file(path, env, src, rcv, interp_ssp='quad')
        rows = [ln.split() for ln in path.read_text().splitlines()
                if ln.strip().endswith('/') and len(ln.split()) == 7]
        # The SSP block ends at the row on the deck depth; the bottom
        # half-space row after it has the same field count.
        depths = [float(r[0]) for r in rows]
        depths = depths[:depths.index(100.0) + 1]
        assert depths == [0.0, 29.99, 30.0, 70.0, 70.01, 100.0]
        matrix = path.with_suffix('.ssp').read_text().splitlines()[2:]
        assert len(matrix) == len(depths)
        first_column = [float(ln.split()[1]) for ln in matrix]
        np.testing.assert_allclose(
            first_column, np.interp(depths, [0.0, 100.0], [1500.0, 1490.0]),
            rtol=0, atol=1e-6)

    def test_a_pinned_pchip_profile_refuses_an_interior_edge(self, tmp_path):
        env = uacpy.Environment(
            bathymetry=100.0, ssp=[(0.0, 1520.0), (100.0, 1500.0)],
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1600.0,
                density=1.5, attenuation=0.5),
            absorption=uacpy.Biological(
                layers=[(30.0, 70.0, 300.0, 4.0, 0.125)]))
        src = uacpy.Source(depths=25.0, frequencies=200.0)
        rcv = uacpy.Receiver(depths=[50.0], ranges=[1000.0])
        with pytest.raises(ConfigurationError, match="interp_ssp='linear'"):
            write_bellhop_env_file(tmp_path / 'p.env', env, src, rcv,
                                   interp_ssp='pchip')


def test_a_cerveny_deck_with_no_knobs_writes_the_documented_defaults(tmp_path):
    """The two Cerveny lines (ReadEnvironmentBell.f90:197, :217) of a
    ``beam_type='C'`` deck given no beam knobs: space-filling width 'F',
    double curvature 'D', epsilon multiplier 1, RLoop 1 km, one image,
    window 4, pressure — the defaults the Bellhop model declares too."""
    from uacpy import Bellhop
    from uacpy.io.bellhop_writer import CERVENY_DEFAULTS
    src = uacpy.Source(depths=25.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]),
                         ranges=np.linspace(1000.0, 5000.0, 5))
    path = tmp_path / 'cerveny.env'
    write_bellhop_env_file(path, Environment(bathymetry=100.0, ssp=1500.0),
                           src, rcv, beam_type='C')
    assert path.read_text().splitlines()[-2:] == [
        "'FD' 1.000000 1.000000", "1 4 'P'"]
    assert CERVENY_DEFAULTS == {
        'beam_width_type': 'F', 'beam_curvature': 'D', 'component': 'P',
        'eps_multiplier': 1.0, 'r_loop': 1000.0, 'n_image': 1, 'ib_win': 4}
    model = Bellhop()
    assert {name: getattr(model, name) for name in CERVENY_DEFAULTS} == \
        CERVENY_DEFAULTS


class TestARangeDependentProfileHoldsTheRange0Absorption:
    """Under a range-dependent ``'Q'`` profile Bellhop takes the imaginary
    sound speed from the ``.env`` rows alone (``Bellhop/sspMod.f90:520``),
    the range-0 column: a law the rows carry is then the range-0 α(z) at
    every range, and the writer says so — only when the law varies with
    depth and the profile's speeds leave the range-0 column."""

    @staticmethod
    def _notices(absorption, speeds_at_2km):
        from uacpy.core.environment import SoundSpeedProfile
        from uacpy.core.exceptions import FallbackWarning
        from uacpy.tests.conftest import recorded_warnings
        ssp = SoundSpeedProfile(
            depths=np.array([0.0, 100.0]),
            sound_speed=np.array([[1500.0, speeds_at_2km[0]],
                                  [1505.0, speeds_at_2km[1]]]),
            ranges=np.array([0.0, 2000.0]))
        env = Environment(name='rd', bathymetry=100.0, ssp=ssp,
                          bottom='sand', absorption=absorption)
        with tempfile.TemporaryDirectory() as d, recorded_warnings() as rec:
            write_bellhop_env_file(
                pathlib.Path(d) / 'rd.env', env,
                uacpy.Source(depths=30.0, frequencies=4000.0),
                uacpy.Receiver(depths=[20.0], ranges=[1000.0, 3000.0]))
        return [str(w.message) for w in rec
                if issubclass(w.category, FallbackWarning)
                and 'range-0 alpha(z) is used at every range' in str(w.message)]

    def test_a_profile_over_a_changing_column_is_announced(self):
        from uacpy.tests.conftest import two_layer_absorption
        (msg,) = self._notices(two_layer_absorption(), (1490.0, 1495.0))
        assert 'Bellhop/sspMod.f90:520' in msg

    def test_a_column_equal_to_range_0_is_silent(self):
        from uacpy.tests.conftest import two_layer_absorption
        assert self._notices(two_layer_absorption(), (1500.0, 1505.0)) == []

    def test_a_law_with_a_letter_is_silent(self):
        assert self._notices(uacpy.Thorp(), (1490.0, 1495.0)) == []
