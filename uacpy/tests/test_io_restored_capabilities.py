"""Pins for ten ``uacpy.io`` capabilities that were deleted once and restored.

READ THIS BEFORE PROPOSING ANY OF THESE NAMES FOR REMOVAL.

Ten public names were removed in an earlier dead-code round and put back by
the maintainer. Five of them — ``read_boundary_3d``, ``write_bty_3d``,
``read_ssp_3d``, ``read_flp3d`` and ``write_field3dflp`` — are the BELLHOP3D
and FIELD3D file layer. **They are deliberately retained for planned 3-D
support and are not dead code.** They are unreachable from the 2-D public
API *by design*: no uacpy model runs ``bellhop3d`` or ``field3d`` yet
(Bellhop's RunType position 6 is hardwired ``'2'`` and
``Bellhop(dimensionality='3D')`` raises), so a call-graph sweep finds no
caller and never will until 3-D is wired up. Zero callers is the expected
state, not evidence of rot: they are the foundation the 3-D work starts
from, and the 2-D entry points that refuse 3-D input name them in their own
error messages as what a future implementer builds on.

The other five are capability, not groundwork, and are usable today:

* ``read_shd_asc``, ``read_modes_asc``, ``read_tl_line`` read the ASCII
  variants of three formats the package otherwise reads only in binary.
* ``get_component`` extracts one component of the stress-displacement vector
  of an elastic-medium mode set — a regime with little other support here.
* ``write_reflection_coefficient`` closes a round-trip hole: the package
  could read a ``.brc``/``.trc`` reflection table and not write one.

Each test below either exercises the function on a fixture or asserts a
property of the restoration that a re-deletion, or a revert to the
pre-deletion code, would break. The behaviours pinned here that the
pre-deletion versions got *wrong* are called out in the tests that pin them,
so restoring the old body is caught as well as deleting the new one.
"""

import inspect
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import uacpy
import uacpy.io as io_package
from uacpy.core.exceptions import (
    ConfigurationError, FileFormatError, UnsupportedFeatureError,
)
from uacpy.io.bathy_io import read_boundary_3d, write_bty_3d
from uacpy.io.modes_reader import get_component, read_modes_asc
from uacpy.io.oalib_reader import (
    read_arr_file, read_flp3d, read_ray_file, read_shd_asc, read_ssp_3d,
)
from uacpy.io.oalib_writer import write_field3dflp
from uacpy.io.ramsurf_reader import read_tl_line
from uacpy.io.utils import _collapsed_pair_index
from uacpy.io.refl_io import (
    read_reflection_coefficient, write_reflection_coefficient,
)


#: The five BELLHOP3D / FIELD3D names, retained for planned 3-D support.
THREE_D_NAMES = (
    'read_boundary_3d', 'write_bty_3d', 'read_ssp_3d', 'read_flp3d',
    'write_field3dflp',
)

#: The five that work against today's package.
CURRENT_NAMES = (
    'read_shd_asc', 'read_modes_asc', 'read_tl_line', 'get_component',
    'write_reflection_coefficient',
)

RESTORED_NAMES = THREE_D_NAMES + CURRENT_NAMES

#: The shipped Acoustics-Toolbox FIELD3D deck, the one real-world fixture for
#: ``read_flp3d`` in the tree.
_AT_3D_DECK = (Path(uacpy.__file__).resolve().parent / 'third_party' /
               'Acoustics-Toolbox' / 'tests' / '3DAtlantic' / 'lant.flp')


def _write_lines(path, lines):
    """Write ``lines`` as a newline-terminated text file and return the path."""
    path.write_text(''.join(f'{line}\n' for line in lines))
    return path


class TestEveryRestoredNameIsPublic:
    """The ten are importable, exported and documented.

    This is the half a deletion trips first: dropping a function removes its
    ``__all__`` entry and its attribute together.
    """

    @pytest.mark.parametrize('name', RESTORED_NAMES)
    def test_the_name_is_an_attribute_of_uacpy_io(self, name):
        assert hasattr(io_package, name), (
            f"uacpy.io.{name} is gone. It was removed once and restored on "
            f"purpose; read this module's docstring before removing it again.")
        assert callable(getattr(io_package, name))

    @pytest.mark.parametrize('name', RESTORED_NAMES)
    def test_the_name_is_exported(self, name):
        assert name in io_package.__all__, (
            f"uacpy.io.__all__ no longer lists {name}; the ten restored "
            f"capabilities are public API.")

    @pytest.mark.parametrize('name', RESTORED_NAMES)
    def test_the_name_carries_a_docstring(self, name):
        doc = inspect.getdoc(getattr(io_package, name))
        assert doc and len(doc.splitlines()) > 3, (
            f"uacpy.io.{name} lost its docstring.")

    @pytest.mark.parametrize('name', THREE_D_NAMES)
    def test_the_three_d_docstring_says_it_is_retained_on_purpose(self, name):
        """The reason lives beside the code, not only in this test.

        An auditor reads the function before the test suite, so the
        "retained for planned 3-D support" statement has to survive in the
        docstring for the next sweep to meet it there.
        """
        doc = inspect.getdoc(getattr(io_package, name))
        assert 'not dead code' in doc, (
            f"uacpy.io.{name}'s docstring no longer states that it is "
            f"retained for planned 3-D support; that sentence is what stops "
            f"the next dead-code sweep from proposing its removal.")


class TestThreeDBoundaryFilesRoundTrip:
    """``write_bty_3d`` and ``read_boundary_3d`` against each other."""

    X = np.array([0.0, 1000.0, 2000.0])
    Y = np.array([0.0, 1500.0, 4000.0])
    Z = np.array([[100.0, 90.0, 100.0],
                  [110.0, 95.0, 110.0],
                  [120.0, 99.0, 120.0]])

    def test_a_written_grid_reads_back_unchanged(self, tmp_path):
        path = tmp_path / 'seamount.bty'
        write_bty_3d(path, self.X, self.Y, self.Z, interp_type='C')
        x, y, z, n_x, n_y = read_boundary_3d(path)
        assert (n_x, n_y) == (3, 3)
        assert np.array_equal(x, self.X)
        assert np.array_equal(y, self.Y)
        assert np.array_equal(z, self.Z)

    def test_the_axes_are_metres_on_both_sides(self, tmp_path):
        """The file holds km (``bdry3DMod.f90:290-291`` scales by 1000); the
        API is metres either way, per the metres-unless-suffixed rule."""
        path = tmp_path / 'units.bty'
        write_bty_3d(path, self.X, self.Y, self.Z)
        assert '2.000000' in path.read_text()          # 2000 m written as km
        x, _, _, _, _ = read_boundary_3d(path)
        assert x[-1] == 2000.0                          # and read back as m

    def test_the_depth_grid_is_indexed_row_by_y(self, tmp_path):
        """``bdry3DMod.f90:300-301`` reads one row of ``n_x`` depths per y, so
        ``z[iy, ix]``. A transposed grid is refused rather than written."""
        path = tmp_path / 'shape.bty'
        with pytest.raises(ConfigurationError, match=r'ny=2, nx=3'):
            write_bty_3d(path, self.X, np.array([0.0, 1500.0]), self.Z)

    def test_a_non_monotonic_axis_is_refused(self, tmp_path):
        """``bdry3DMod.f90:324,328`` ERROUTs on one, and ERROUT is ``STOP`` at
        exit 0 with no output — a fix carried in on restoration, since the
        pre-deletion writer emitted such a grid without complaint."""
        with pytest.raises(ConfigurationError, match='strictly increasing'):
            write_bty_3d(tmp_path / 'x.bty', np.array([0.0, 2000.0, 1000.0]),
                         self.Y, self.Z)

    def test_an_axis_that_collides_in_the_km_column_is_refused(self, tmp_path):
        """The axis is validated in metres and written at ``%.6f`` km, so two
        coordinates closer than 1 mm pass the axis check and then land on the
        same token."""
        with pytest.raises(ConfigurationError, match='1 mm resolution'):
            write_bty_3d(tmp_path / 'km.bty',
                         np.array([0.0, 5000.0, 5000.0004]), self.Y, self.Z)

    def test_a_non_finite_depth_is_refused_rather_than_zeroed(self, tmp_path):
        """The pre-deletion writer replaced a NaN with 0.0, handing the engine
        a sea surface where the caller had a gap. bellhop3d only *warns* about
        a NaN (``bdry3DMod.f90:310-312``), so the substitution changed the
        seabed silently at exit 0."""
        depth = self.Z.copy()
        depth[1, 1] = np.nan
        with pytest.raises(ConfigurationError, match='non-finite'):
            write_bty_3d(tmp_path / 'nan.bty', self.X, self.Y, depth)

    def test_the_reader_expands_the_subtab_shorthand(self, tmp_path):
        """``bdry3DMod.f90:271`` calls ``SubTab`` on the axis, so ``first
        last /`` under a count of 5 is five evenly spaced values."""
        path = _write_lines(tmp_path / 'sub.bty', [
            "'R'", '5', '0.0 1.0 /', '2', '0.0 1.0',
            '1 2 3 4 5', '6 7 8 9 10'])
        x, _, z, n_x, n_y = read_boundary_3d(path)
        assert np.array_equal(x, [0.0, 250.0, 500.0, 750.0, 1000.0])
        assert (n_x, n_y) == (5, 2)
        assert z.shape == (2, 5)

    def test_a_two_d_interpolation_code_is_refused(self, tmp_path):
        """The 3-D codes are ``'R'``/``'C'`` (``bdry3DMod.f90:241-247``), not
        the 2-D ``'L'``/``'C'`` pair."""
        with pytest.raises(ConfigurationError, match="'R'"):
            write_bty_3d(tmp_path / 'l.bty', self.X, self.Y, self.Z,
                         interp_type='L')
        path = _write_lines(tmp_path / 'l.bty',
                            ["'L'", '2', '0 1', '2', '0 1', '1 2', '3 4'])
        with pytest.raises(FileFormatError, match='regular grid'):
            read_boundary_3d(path)

    def test_a_missing_file_is_a_configuration_error(self, tmp_path):
        """A boundary deck is authored by the user, not written by a model —
        the provenance split ``FileFormatError`` documents."""
        with pytest.raises(ConfigurationError, match='not found'):
            read_boundary_3d(tmp_path / 'absent.bty')


class TestThreeDSoundSpeedFile:
    """``read_ssp_3d`` on the BELLHOP3D hexahedral ``.ssp``."""

    DECK = ['2', '0.0 1.0', '2', '0.0 2.0', '2', '0.0 100.0',
            '1500 1501', '1502 1503', '1510 1511', '1512 1513']

    def test_the_grid_is_indexed_depth_y_x(self, tmp_path):
        """``sspMod.f90:610-612`` loops depth outermost, then y, reading one
        record of ``Nx`` speeds."""
        ssp = read_ssp_3d(_write_lines(tmp_path / 'a.ssp', self.DECK))
        assert (ssp['Nx'], ssp['Ny'], ssp['Nz']) == (2, 2, 2)
        assert ssp['c_mat'].shape == (2, 2, 2)
        assert ssp['c_mat'][0, 0, 0] == 1500.0
        assert ssp['c_mat'][0, 1, 1] == 1503.0
        assert ssp['c_mat'][1, 0, 1] == 1511.0

    def test_the_horizontal_axes_are_metres_and_the_depth_axis_is_too(
            self, tmp_path):
        """``sspMod.f90:621-622`` scales x and y from km and leaves z alone.
        The pre-deletion reader returned the raw km for x and y, against the
        package's metres-unless-suffixed rule — the fix carried in here."""
        ssp = read_ssp_3d(_write_lines(tmp_path / 'u.ssp', self.DECK))
        assert np.array_equal(ssp['Segx'], [0.0, 1000.0])
        assert np.array_equal(ssp['Segy'], [0.0, 2000.0])
        assert np.array_equal(ssp['Segz'], [0.0, 100.0])

    def test_a_record_may_wrap_across_lines(self, tmp_path):
        """Each vector is one list-directed READ, so it keeps consuming
        records until its count is satisfied."""
        ssp = read_ssp_3d(_write_lines(tmp_path / 'w.ssp', [
            '3', '0.0 1.0', '2.0', '2', '0 2', '2', '0 100',
            '1 2 3', '4 5 6', '7 8 9', '10 11 12']))
        assert np.array_equal(ssp['Segx'], [0.0, 1000.0, 2000.0])
        assert ssp['c_mat'].shape == (2, 2, 3)

    def test_a_single_point_axis_is_refused(self, tmp_path):
        """``sspMod.f90:600-601`` ERROUTs below two points on any axis."""
        with pytest.raises(FileFormatError, match='at least two'):
            read_ssp_3d(_write_lines(tmp_path / 'one.ssp',
                                     ['1', '0.0', '2', '0 1', '2', '0 1',
                                      '1', '2']))


class TestThreeDFieldParameterDeck:
    """``write_field3dflp`` and ``read_flp3d`` against each other and against
    the shipped Acoustics-Toolbox deck."""

    POS = {
        's': {'x': np.array([333000.0]), 'y': np.array([315000.0]),
              'z': np.array([400.0])},
        'r': {'z': np.array([400.0]), 'r': np.linspace(0.0, 275000.0, 1001),
              'theta': np.linspace(0.0, 360.0, 181)},
    }
    BATHY = {'X': np.linspace(0.0, 100000.0, 4),
             'Y': np.linspace(0.0, 60000.0, 3),
             'depth': np.full((3, 4), 100.0)}

    def _deck(self, tmp_path, **kwargs):
        path = tmp_path / 'lant.flp'
        write_field3dflp(path, 'STDFM', self.POS, self.BATHY, **kwargs)
        return read_flp3d(path)

    def test_a_written_deck_reads_back_with_every_axis_intact(self, tmp_path):
        deck = self._deck(tmp_path, title='RT', M_limit=60)
        assert deck['title'] == 'RT'
        assert deck['M_limit'] == 60
        assert np.allclose(deck['pos']['s']['x'], self.POS['s']['x'])
        assert np.allclose(deck['pos']['s']['y'], self.POS['s']['y'])
        assert np.allclose(deck['pos']['s']['z'], self.POS['s']['z'])
        assert np.allclose(deck['pos']['r']['z'], self.POS['r']['z'])
        assert np.allclose(deck['pos']['r']['r'], self.POS['r']['r'])
        assert np.allclose(deck['pos']['r']['theta'], self.POS['r']['theta'])

    def test_the_node_table_carries_every_grid_point_in_metres(self, tmp_path):
        deck = self._deck(tmp_path)
        assert np.allclose(deck['nodes']['x'], np.tile(self.BATHY['X'], 3))
        assert np.allclose(deck['nodes']['y'], np.repeat(self.BATHY['Y'], 4))
        assert len(deck['nodes']['mode_file']) == 12

    def test_element_node_indices_start_at_one(self, tmp_path):
        """``field3d.f90:207`` reads them into a Fortran array indexed from 1
        and then does ``x( Node( 1, iElt ) )``. The pre-deletion writer
        numbered from 0, addressing ``x(0)`` on the first row of triangles and
        one past ``x(NNodes)`` on the last — the fix carried in here, and the
        shipped deck ``tests/3DAtlantic/lant.flp`` numbers its 397 nodes
        1..397."""
        deck = self._deck(tmp_path)
        assert deck['elements'].shape == (12, 3)
        assert deck['elements'].min() == 1
        assert deck['elements'].max() == len(deck['nodes']['mode_file'])

    def test_a_dry_node_gets_the_dummy_mode_file(self, tmp_path):
        bathy = dict(self.BATHY, depth=self.BATHY['depth'].copy())
        bathy['depth'][0, 0] = 0.0
        path = tmp_path / 'dry.flp'
        write_field3dflp(path, 'STDFM', self.POS, bathy)
        assert read_flp3d(path)['nodes']['mode_file'][0] == 'DUMMY'

    def test_an_explicit_suffix_is_kept(self, tmp_path):
        """``.flp`` is appended only when the path has none — the convention
        ``write_fieldflp`` was corrected to, and the one ``read_flp3d``
        resolves with."""
        write_field3dflp(tmp_path / 'case.v2', 'STDFM', self.POS, self.BATHY)
        write_field3dflp(tmp_path / 'case', 'STDFM', self.POS, self.BATHY)
        assert (tmp_path / 'case.v2').is_file()
        assert (tmp_path / 'case.flp').is_file()

    def test_the_option_word_is_split_the_way_field3d_reads_it(self, tmp_path):
        """FIELD3D's ``Option`` is ``CHARACTER(LEN=7)``: columns 1-3 select the
        evaluator (``field3d.f90:96``), column 4 the tesselation check
        (``:210``), column 7 the beam-pattern flag (``:54``). There is no
        elastic-component column, so the pre-deletion ``comp`` key — column 3,
        the third letter of ``'STD'`` — named nothing."""
        path = tmp_path / 'opt.flp'
        write_field3dflp(path, 'STDT  *', self.POS, self.BATHY)
        deck = read_flp3d(path)
        assert deck['method'] == 'STD'
        assert deck['tesselation_check'] is True
        assert deck['sbp_flag'] == '*'

    def test_a_grid_too_small_to_triangulate_is_refused(self, tmp_path):
        with pytest.raises(ConfigurationError, match='2 x 2'):
            write_field3dflp(tmp_path / 'thin.flp', 'STDFM', self.POS,
                             {'X': np.array([0.0]), 'Y': np.array([0.0]),
                              'depth': np.full((1, 1), 100.0)})

    @pytest.mark.skipif(not _AT_3D_DECK.is_file(),
                        reason='vendored Acoustics-Toolbox tests/ absent')
    def test_the_shipped_acoustics_toolbox_deck_parses(self):
        """``tests/3DAtlantic/lant.flp`` is the reference FIELD3D deck, and
        ``field3d.exe`` echoes ``NNodes = 397`` / ``NElts = 727`` when it
        loads it. The pre-deletion reader read the records in a different
        order entirely — source x as profile ranges, source y as bearings —
        and could not parse this file at all."""
        deck = read_flp3d(_AT_3D_DECK)
        assert deck['method'] == 'STD'
        assert deck['M_limit'] == 60
        assert len(deck['nodes']['mode_file']) == 397
        assert deck['elements'].shape == (727, 3)
        assert deck['elements'].min() == 1
        assert deck['elements'].max() == 397
        # 333 km / 315 km source, read through ReadVector's km -> m scaling.
        assert deck['pos']['s']['x'][0] == pytest.approx(333000.0)
        assert deck['pos']['s']['y'][0] == pytest.approx(315000.0)
        # '0.0 275.0 /' under a count of 1001, SubTab-expanded.
        assert deck['pos']['r']['r'].size == 1001
        assert deck['pos']['r']['r'][-1] == pytest.approx(275000.0)
        assert deck['pos']['r']['theta'].size == 181


class TestTheTwoDGuardsNameTheThreeDFunctions:
    """The 2-D entry points refuse 3-D input and say where the 3-D work
    starts. That cross-reference is the other half of the anti-deletion pin:
    removing a 3-D function leaves an error message pointing at nothing.
    """

    def test_the_arrivals_guard_names_them(self, tmp_path):
        path = tmp_path / 'a.arr'
        path.write_text("'3D'\n100.0\n")
        with pytest.raises(FileFormatError) as excinfo:
            read_arr_file(path)
        message = str(excinfo.value)
        assert 'not yet available' in message
        assert 'read_boundary_3d' in message and 'read_flp3d' in message

    def test_the_ray_guard_names_them(self, tmp_path):
        path = tmp_path / 'a.ray'
        path.write_text("'title'\n100.0\n1\n1 1\n0.0\n200.0\n'xyz'\n")
        with pytest.raises(FileFormatError) as excinfo:
            read_ray_file(path)
        message = str(excinfo.value)
        assert 'not yet available' in message
        assert 'read_boundary_3d' in message and 'read_ssp_3d' in message

    def test_the_bellhop_dimensionality_guard_names_them(self):
        from uacpy.models.bellhop import Bellhop
        with pytest.raises(UnsupportedFeatureError) as excinfo:
            Bellhop(dimensionality='3D')
        message = str(excinfo.value)
        assert 'not yet available' in message
        assert 'write_bty_3d' in message and 'write_field3dflp' in message


class TestAsciiShadeFile:
    """``read_shd_asc``: the text sibling of ``read_shd_bin``."""

    DECK = ['PEKERIS TEST', 'rectilin',
            '1 1', '1 3 2',            # counts, deliberately split in two
            '100.0', '0.0',            # freq0, atten, likewise
            '100.0',                   # freqVec
            '0.0',                     # theta
            '25.0',                    # source depth
            '0.0 10.0 20.0',           # receiver depths
            '0.0 1000.0',              # receiver ranges
            '1.0 2.0', '3.0 4.0', '5.0 6.0',
            '7.0 8.0', '9.0 10.0', '11.0 12.0']

    def test_the_pressure_block_is_interleaved_real_imaginary(self, tmp_path):
        """``read_shd_asc.m:29-36`` fills ``[2*Nrr, Nrd]`` and splits the odd
        and even rows, i.e. ``Nrd`` groups of ``Nrr`` ``(Re, Im)`` pairs."""
        shd = read_shd_asc(_write_lines(tmp_path / 't.shd.asc', self.DECK))
        assert shd['pressure'].shape == (1, 1, 3, 2)
        assert np.array_equal(shd['pressure'][0, 0],
                              np.array([[1 + 2j, 3 + 4j],
                                        [5 + 6j, 7 + 8j],
                                        [9 + 10j, 11 + 12j]]))

    def test_the_header_is_a_token_stream_not_two_fixed_lines(self, tmp_path):
        """The reference reader takes the seven header numbers with seven
        ``fscanf`` calls, so their line layout carries no meaning. The
        pre-deletion reader read them as one line of five and one of two, and
        refused a file that reads fine everywhere else — the ``DECK`` above
        splits them 2 + 3 and 1 + 1 for exactly that reason."""
        shd = read_shd_asc(_write_lines(tmp_path / 'h.shd.asc', self.DECK))
        assert shd['freq0'] == 100.0
        assert shd['atten'] == 0.0
        assert np.array_equal(shd['freqVec'], [100.0])

    def test_the_dict_keys_match_the_binary_reader(self, tmp_path):
        """A caller switching on extension gets the same shape either way."""
        shd = read_shd_asc(_write_lines(tmp_path / 'k.shd.asc', self.DECK))
        assert set(shd) == {'title', 'PlotType', 'freqVec', 'freq0', 'atten',
                            'Pos', 'pressure'}
        assert np.array_equal(shd['Pos']['r']['z'], [0.0, 10.0, 20.0])
        assert np.array_equal(shd['Pos']['r']['r'], [0.0, 1000.0])
        assert np.array_equal(shd['Pos']['s']['z'], [25.0])

    def test_a_multi_bearing_header_raises_rather_than_returning_one_block(
            self, tmp_path):
        """The format carries a single pressure block after the axes and no
        shipped program writes a multi-block one, so the order of any further
        blocks is not established by a producer. The reference reader returns
        the first block as the whole file; this refuses, the way
        ``read_shd_file`` refuses a multi-bearing binary file."""
        deck = list(self.DECK)
        deck[2] = '1 2'
        with pytest.raises(UnsupportedFeatureError, match='Ntheta=2'):
            read_shd_asc(_write_lines(tmp_path / 'm.shd.asc', deck))

    def test_a_truncated_pressure_block_names_what_it_wanted(self, tmp_path):
        with pytest.raises(FileFormatError, match='interleaved'):
            read_shd_asc(_write_lines(tmp_path / 'c.shd.asc',
                                      self.DECK[:-2]))

    def test_a_missing_file_is_a_format_error(self, tmp_path):
        with pytest.raises(FileFormatError, match='not found'):
            read_shd_asc(tmp_path / 'absent.shd.asc')


class TestAsciiModeFile:
    """``read_modes_asc``: the text sibling of ``read_modes_bin``."""

    DECK = ['32', 'KRAKEN PEKERIS',
            '100.0 1 3 3 2',
            '  50 0.0 100.0',           # one medium record
            "'V' 0 0 0 0 0 0",          # top halfspace, skipped
            "'A' 1600 0 0 0 1.8 100",   # bottom halfspace, skipped
            '',                         # blank line
            '0.0 50.0 100.0',           # depths
            '0.41 -1.0e-5 0.39 -4.0e-5',  # k, interleaved (Re, Im)
            'Mode 1', '1.0 0.1 2.0 0.2 3.0 0.3',
            'Mode 2', '4.0 0.4 5.0 0.5 6.0 0.6']

    def test_complex_records_are_interleaved_pairs(self, tmp_path):
        """``read_modes_asc.m:33,50`` uses ``fscanf( fid, '%f', [ 2, N ] )``,
        which fills a 2-by-N array in column order — interleaved
        ``(Re, Im)``. The pre-deletion reader read a block of ``N`` reals
        followed by a block of ``N`` imaginaries, which silently returns the
        first half of the file's values as the real part of everything."""
        modes = read_modes_asc(_write_lines(tmp_path / 'p.moa', self.DECK))
        assert np.allclose(modes['k'], [0.41 - 1.0e-5j, 0.39 - 4.0e-5j])
        assert np.allclose(modes['phi'][:, 0], [1 + 0.1j, 2 + 0.2j, 3 + 0.3j])
        assert np.allclose(modes['phi'][:, 1], [4 + 0.4j, 5 + 0.5j, 6 + 0.6j])

    def test_the_header_and_axes_are_read(self, tmp_path):
        modes = read_modes_asc(_write_lines(tmp_path / 'h.moa', self.DECK))
        assert modes['pltitl'] == 'KRAKEN PEKERIS'
        assert modes['freq'] == 100.0
        assert (modes['Nmedia'], modes['ntot'], modes['nmat']) == (1, 3, 3)
        assert np.array_equal(modes['z'], [0.0, 50.0, 100.0])

    def test_m_counts_the_modes_returned(self, tmp_path):
        """``M`` means ``len(k)`` in both readers, so a ``modes=`` subset does
        not disagree with the ``k`` and ``phi`` handed back with it."""
        path = _write_lines(tmp_path / 's.moa', self.DECK)
        assert read_modes_asc(path)['M'] == 2
        subset = read_modes_asc(path, modes=[2])
        assert subset['M'] == 1
        assert np.allclose(subset['k'], [0.39 - 4.0e-5j])
        assert np.allclose(subset['phi'][:, 0], [4 + 0.4j, 5 + 0.5j, 6 + 0.6j])

    def test_an_out_of_range_mode_index_is_dropped(self, tmp_path):
        """``read_modes_asc.m:41-43`` filters rather than raising."""
        modes = read_modes_asc(_write_lines(tmp_path / 'o.moa', self.DECK),
                               modes=[1, 999])
        assert modes['M'] == 1

    def test_a_missing_file_is_a_format_error(self, tmp_path):
        with pytest.raises(FileFormatError, match='not found'):
            read_modes_asc(tmp_path / 'absent.moa')

    def test_read_modes_takes_only_the_binary_extension(self, tmp_path):
        """``read_modes`` attaches halfspace terms an ASCII file does not
        carry, so it keeps refusing ``.moa`` — and now names the reader that
        does handle one."""
        from uacpy.io.modes_reader import read_modes
        path = _write_lines(tmp_path / 'x.moa', self.DECK)
        with pytest.raises(FileFormatError) as excinfo:
            read_modes(str(path))
        assert 'read_modes_asc' in str(excinfo.value)


class TestElasticComponentExtraction:
    """``get_component``: one component of the stress-displacement vector."""

    def test_an_acoustic_mode_set_comes_back_unchanged(self):
        """Without KRAKEL, ``Mater`` never holds ``'ELASTIC'`` and this is a
        copy of ``phi`` — which is exactly why a call-graph audit saw nothing
        worth keeping. The elastic case below is the capability."""
        phi = np.arange(12.0).reshape(6, 2)
        modes = {'phi': phi, 'z': np.zeros(6), 'Nmedia': 1,
                 'Mater': ['ACOUSTIC']}
        assert np.array_equal(get_component(modes, 'H'), phi)

    @pytest.mark.parametrize('comp,expected',
                             [('H', [0.0, 4.0]), ('V', [1.0, 5.0]),
                              ('T', [2.0, 6.0]), ('N', [3.0, 7.0])])
    def test_each_component_takes_its_row_of_the_elastic_group(
            self, comp, expected):
        """An elastic medium stacks four rows per depth in the order H, V, T,
        N (``get_component.m:29-41``)."""
        modes = {'phi': np.arange(8.0).reshape(8, 1), 'z': np.zeros(2),
                 'Nmedia': 1, 'Mater': ['ELASTIC']}
        assert np.array_equal(get_component(modes, comp).ravel(), expected)

    def test_an_absent_mater_key_reads_as_acoustic(self):
        """The pre-deletion default was a nested ``[['ACOUSTIC']]``, which
        stringified to ``"['ACOUSTIC']"`` and raised "Unknown material type"
        on every dict that omitted the key."""
        phi = np.arange(6.0).reshape(3, 2)
        assert np.array_equal(
            get_component({'phi': phi, 'z': np.zeros(3)}, 'V'), phi)

    def test_an_unknown_component_is_refused_on_an_acoustic_set_too(self):
        """Validated up front rather than inside the elastic branch, so a typo
        raises whatever the medium stack is."""
        modes = {'phi': np.zeros((2, 1)), 'z': np.zeros(2),
                 'Mater': ['ACOUSTIC']}
        with pytest.raises(ConfigurationError, match='stress-displacement'):
            get_component(modes, 'Q')

    def test_an_unknown_material_is_refused(self):
        modes = {'phi': np.zeros((2, 1)), 'z': np.zeros(2), 'Nmedia': 1,
                 'Mater': ['GLASS']}
        with pytest.raises(ConfigurationError, match='GLASS'):
            get_component(modes, 'H')

    def test_an_empty_mode_set_names_modal_cutoff(self):
        modes = {'phi': np.zeros((0, 0)), 'z': np.zeros(3),
                 'Mater': ['ACOUSTIC']}
        with pytest.raises(FileFormatError, match='M=0'):
            get_component(modes, 'H')


class TestReflectionCoefficientRoundTrip:
    """``write_reflection_coefficient`` closes the read-without-write hole."""

    THETA = np.linspace(0.0, 90.0, 91)
    R = ((0.4 + 0.5 * np.cos(np.radians(np.linspace(0.0, 90.0, 91)))) *
         np.exp(1j * np.radians(-2.0 * np.linspace(0.0, 90.0, 91))))

    def test_a_complex_table_round_trips_through_the_reader(self, tmp_path):
        path = tmp_path / 'sand.brc'
        write_reflection_coefficient(path, self.THETA, self.R)
        table = read_reflection_coefficient(path)
        assert table['n_pts'] == 91
        assert np.array_equal(table['theta'], self.THETA)
        assert np.allclose(table['R'], np.abs(self.R), atol=1e-6)
        assert np.allclose(table['phi'], np.angle(self.R), atol=1e-8)

    def test_phase_is_radians_in_and_degrees_on_disk(self, tmp_path):
        """The direction ``read_reflection_coefficient`` reads back, and the
        one the ``ReflectionCoefficient`` result carries."""
        path = tmp_path / 'phase.brc'
        write_reflection_coefficient(path, np.array([0.0, 90.0]),
                                     np.array([[1.0, np.pi / 2],
                                               [1.0, -np.pi / 2]]))
        assert '90.000000' in path.read_text()
        assert np.allclose(read_reflection_coefficient(path)['phi'],
                           [np.pi / 2, -np.pi / 2])

    def test_a_real_amplitude_column_writes_zero_phase(self, tmp_path):
        path = tmp_path / 'amp.brc'
        write_reflection_coefficient(path, np.array([0.0, 45.0, 90.0]),
                                     np.array([0.9, 0.5, 0.1]))
        table = read_reflection_coefficient(path)
        assert np.allclose(table['R'], [0.9, 0.5, 0.1])
        assert np.array_equal(table['phi'], np.zeros(3))

    def test_mismatched_column_lengths_are_refused_before_the_file_opens(
            self, tmp_path):
        path = tmp_path / 'short.brc'
        with pytest.raises(ConfigurationError, match='exactly one'):
            write_reflection_coefficient(path, np.array([0.0, 45.0, 90.0]),
                                         np.array([0.9, 0.5]))
        assert not path.exists()

    def test_a_decreasing_angle_axis_is_refused(self, tmp_path):
        """``read_reflection_coefficient`` rejects one as malformed, so
        writing it would produce a file uacpy cannot read back."""
        with pytest.raises(ConfigurationError, match='non-decreasing'):
            write_reflection_coefficient(tmp_path / 'down.brc',
                                         np.array([90.0, 45.0, 0.0]),
                                         np.array([0.9, 0.5, 0.1]))

    def test_distinct_angles_that_collide_in_the_column_are_refused(
            self, tmp_path):
        """The angle column is written at 1e-6 degrees. Two distinct angles
        closer than that become a duplicated row: ``bhc::setup()`` aborts on a
        non-strictly-increasing table and Bellhop interpolates across the
        zero-width segment. The pre-deletion writer used ``%8.2f``, 0.01
        degrees, and collapsed a resolved table without a word."""
        with pytest.raises(ConfigurationError, match='angle resolution'):
            write_reflection_coefficient(tmp_path / 'collide.brc',
                                         np.array([0.0, 45.0, 45.0000001]),
                                         np.array([0.9, 0.5, 0.1]))

    def test_equal_angles_are_allowed(self, tmp_path):
        """BOUNCE produces them by construction, which is what
        ``dedupe_reflection_file`` exists to collapse — refusing them here
        would reject the package's own tables."""
        path = tmp_path / 'dup.brc'
        write_reflection_coefficient(path, np.array([0.0, 0.0, 45.0]),
                                     np.array([0.9, 0.9, 0.5]))
        assert read_reflection_coefficient(path)['n_pts'] == 3

    def test_an_empty_table_is_refused(self, tmp_path):
        with pytest.raises(ConfigurationError, match='absorbing'):
            write_reflection_coefficient(tmp_path / 'empty.brc',
                                         np.array([]), np.array([]))


class TestCollinsTlLine:
    """``read_tl_line``: the ASCII single-depth trace of a Collins RAM run."""

    def test_ranges_and_tl_come_back_as_two_columns(self, tmp_path):
        path = _write_lines(tmp_path / 'tl.line',
                            ['  52.586   59.606', ' 105.172   44.938',
                             ' 157.758   45.374'])
        ranges, tl = read_tl_line(path)
        assert np.allclose(ranges, [52.586, 105.172, 157.758])
        assert np.allclose(tl, [59.606, 44.938, 45.374])

    def test_ranges_are_metres(self, tmp_path):
        """``rams0.5.f:253`` writes ``r`` verbatim and RAM marches in metres,
        so no conversion is applied."""
        path = _write_lines(tmp_path / 'tl.line', ['5000.0  70.0'])
        ranges, _ = read_tl_line(path)
        assert ranges[0] == 5000.0

    def test_a_single_row_file_yields_two_one_element_arrays(self, tmp_path):
        ranges, tl = read_tl_line(_write_lines(tmp_path / 'tl.line',
                                               ['100.0  50.0']))
        assert ranges.shape == (1,) and tl.shape == (1,)

    def test_an_empty_file_names_the_run_that_wrote_nothing(self, tmp_path):
        path = tmp_path / 'tl.line'
        path.write_text('')
        with pytest.raises(FileFormatError, match='no rows'):
            read_tl_line(path)

    def test_a_one_column_file_is_refused(self, tmp_path):
        with pytest.raises(FileFormatError, match='column'):
            read_tl_line(_write_lines(tmp_path / 'tl.line', ['1.0', '2.0']))

    def test_a_missing_file_is_a_format_error(self, tmp_path):
        with pytest.raises(FileFormatError, match='not found'):
            read_tl_line(tmp_path / 'absent.line')


class TestWrittenTokenCollisionGuardsShareOneDetector:
    """Each of the six deck-column guards raises its own site-specific
    ``ConfigurationError``, and all six locate the offending pair through
    ``uacpy.io.utils._collapsed_pair_index``."""

    def test_receiver_ranges_that_collide_in_the_km_column_are_refused(self):
        from uacpy.io.oalib_writer import write_receiver_ranges
        # A bare namespace stands in for the carrier: Receiver's own 1 mm
        # step floor refuses this pair before the writer ever sees it, so
        # the writer's token check is exercised directly.
        carrier = SimpleNamespace(ranges=np.array([1000.0, 1000.0004]))
        with pytest.raises(ConfigurationError, match='1 mm resolution'):
            write_receiver_ranges(io.StringIO(), carrier)

    def test_fieldflp_profile_ranges_that_collide_are_refused(self, tmp_path):
        from uacpy.io.oalib_writer import write_fieldflp
        with pytest.raises(ConfigurationError,
                           match='profile axis non-increasing'):
            write_fieldflp(
                tmp_path / 'bad.flp', 'RA',
                {'r': {'r': np.array([1000.0]), 'z': np.array([50.0])},
                 's': {'z': np.array([25.0])}},
                n_profiles=2, profile_ranges_m=np.array([0.0, 0.0004]))

    def test_valid_profile_ranges_write(self, tmp_path):
        from uacpy.io.oalib_writer import write_fieldflp
        out = tmp_path / 'ok.flp'
        write_fieldflp(
            out, 'RA',
            {'r': {'r': np.array([1000.0]), 'z': np.array([50.0])},
             's': {'z': np.array([25.0])}},
            n_profiles=2, profile_ranges_m=np.array([0.0, 5000.0]))
        assert out.exists()

    def test_bty_ranges_that_collide_in_the_km_column_are_refused(
            self, tmp_path):
        from uacpy.io.bathy_io import write_bty_file
        with pytest.raises(ConfigurationError, match='5.000000 km'):
            write_bty_file(
                tmp_path / 'bad.bty',
                np.array([[0.0, 200.0], [5000.0, 190.0],
                          [5000.0004, 180.0], [10000.0, 200.0]]))

    def test_oases_receiver_depth_tokens_that_collide_are_refused(self):
        from uacpy.io.oases_writer import _check_receiver_depth_tokens
        with pytest.raises(ConfigurationError, match='0.01 m resolution'):
            _check_receiver_depth_tokens(
                [50.0, 50.001], ['50.00', '50.00'], what='receiver depths')

    def test_distinct_brc_angles_that_collide_when_rounded_are_refused(
            self, tmp_path):
        from uacpy.io.refl_io import write_reflection_coefficient
        with pytest.raises(ConfigurationError, match='both write 10.000000'):
            write_reflection_coefficient(
                tmp_path / 'bad.brc', np.array([10.0, 10.0000004]),
                np.array([0.5 + 0.0j, 0.6 + 0.0j]))

    def test_a_deliberately_repeated_brc_angle_is_not_a_collision(
            self, tmp_path):
        from uacpy.io.refl_io import write_reflection_coefficient
        out = tmp_path / 'ok.brc'
        write_reflection_coefficient(
            out, np.array([10.0, 10.0, 20.0]),
            np.array([0.5 + 0.0j, 0.5 + 0.0j, 0.6 + 0.0j]))
        assert out.exists()

    def test_sbp_angles_closer_than_the_column_resolution_are_refused(
            self, tmp_path):
        from uacpy.io.refl_io import write_source_beam_pattern
        with pytest.raises(ConfigurationError, match='angle resolution'):
            write_source_beam_pattern(
                tmp_path / 'bad.sbp', np.array([0.0, 5e-7]),
                np.array([0.0, -3.0]))

    def test_an_sbp_step_exactly_at_the_resolution_is_accepted(
            self, tmp_path):
        from uacpy.io.refl_io import write_source_beam_pattern
        out = tmp_path / 'ok.sbp'
        write_source_beam_pattern(
            out, np.array([0.0, 1e-6]), np.array([0.0, -3.0]))
        assert out.exists()

    def test_the_detector_reports_the_collapsed_token_pair(self):
        assert _collapsed_pair_index(['1.000000', '1.000000']) == 0
        assert _collapsed_pair_index(['1.000000', '2.000000']) is None
        assert _collapsed_pair_index(['1.000000']) is None

    def test_the_detector_ignores_a_repeat_the_raw_axis_carries(self):
        written = np.array([10.0, 10.0, 20.0])
        assert _collapsed_pair_index(written, raw=written) is None
        assert _collapsed_pair_index(
            np.array([10.0, 10.0]), raw=np.array([10.0, 10.0000004])) == 0

    def test_the_detector_bounds_step_size_in_min_step_mode(self):
        assert _collapsed_pair_index(
            np.array([0.0, 1e-6]), min_step=1e-6) is None
        assert _collapsed_pair_index(
            np.array([0.0, 5e-7]), min_step=1e-6) == 0
