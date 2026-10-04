"""The bathymetry and altimetry files (``uacpy.io.bathy_io``)."""

import numpy as np
import pytest
from uacpy.core.exceptions import ConfigurationError


class TestBathyIOTypedErrors:
    """``bathy_io``'s failure typing: a malformed file is a parse error, a bad
    user-supplied interpolation code is a configuration error."""

    def test_malformed_bty_row_raises_fileformaterror(self, tmp_path):
        from uacpy.io.bathy_io import read_bathymetry
        from uacpy.core.exceptions import FileFormatError
        bad = tmp_path / 'bad.bty'
        bad.write_text("'L'\n2\n0.0 100.0\nqqq qqq\n")
        with pytest.raises(FileFormatError, match='could not parse'):
            read_bathymetry(str(bad))

    @pytest.mark.parametrize('writer', ['bty', 'bty_long', 'ati'])
    def test_bad_interp_type_raises_configurationerror(self, tmp_path, writer):
        from uacpy.core.bottom import Bottom
        from uacpy.io.bathy_io import (
            write_bty_file, write_bty_long_format, write_ati_file)
        pairs = np.array([[0.0, 100.0], [1000.0, 120.0]])
        path = tmp_path / 'x.bty'
        with pytest.raises(ConfigurationError,
                           match='Invalid interpolation type'):
            if writer == 'bty':
                write_bty_file(path, pairs, interp_type='bogus')
            elif writer == 'ati':
                write_ati_file(path, pairs, interp_type='bogus')
            else:
                rd = Bottom.from_halfspaces(
                    np.array([0.0, 1000.0]),
                    sound_speed=np.array([1600.0, 1700.0]),
                    density=np.array([1.7, 1.9]),
                    attenuation=np.array([0.4, 0.6]))
                write_bty_long_format(path, pairs, rd, interp_type='bogus')

    def test_long_format_bty_round_trips_geoacoustics(self, tmp_path):
        from uacpy.core.bottom import Bottom
        from uacpy.io.bathy_io import write_bty_long_format, read_bathymetry
        bathy = np.array([[0.0, 100.0], [5000.0, 150.0], [10000.0, 120.0]])
        rd = Bottom.from_halfspaces(
            np.array([0.0, 10000.0]),
            sound_speed=np.array([1600.0, 1700.0]),
            density=np.array([1.7, 1.9]),
            attenuation=np.array([0.4, 0.6]))
        path = tmp_path / 'long.bty'
        write_bty_long_format(path, bathy, rd, interp_type='L')
        bty = read_bathymetry(path)
        assert bty.interpolation == 'L'
        assert bty.is_long_format, "long format must return the geoacoustics"
        # Rows: the bathymetry nodes plus the switch midway between the two
        # bottom nodes (here coinciding with the 5 km bathymetry node). The
        # seabed steps there: the first column up to the switch row, the
        # second from it — nothing blended.
        assert np.allclose(bty.ranges, [0.0, 5000.0, 10000.0])
        assert np.allclose(bty.depths, [100.0, 150.0, 120.0])
        assert np.allclose(bty.sound_speed, [1600.0, 1700.0, 1700.0])
        assert np.allclose(bty.density, [1.7, 1.9, 1.9])
        assert np.allclose(bty.attenuation, [0.4, 0.6, 0.6])
        # The parser read_env uses extends to ±infinity, every row constant.
        from uacpy.io.bathy_io import _parse_bathymetry
        padded, _ = _parse_bathymetry(path)
        assert padded[0, 0] == -1e50 and padded[0, -1] == 1e50
        assert padded[2, 0] == padded[2, 1] and padded[2, -1] == padded[2, -2]


class TestAltimetryLongFormat:
    """``ReadATI`` accepts the same long format as ``ReadBTY``
    (``bdryMod.f90:80-110``), so ``read_altimetry`` must not truncate a
    ``TYPE(2:2) == 'L'`` ``.ati`` to two columns."""

    LONG_ATI = (
        "'LL'\n3\n"
        "0.000000 0.000000 3500.000 1800.000 0.900 0.100000 0.200000\n"
        "2.500000 2.000000 3400.000 1750.000 0.910 0.110000 0.210000\n"
        "5.000000 0.000000 3300.000 1700.000 0.920 0.120000 0.220000\n"
    )

    def test_long_format_returns_the_geoacoustic_rows(self, tmp_path):
        from uacpy.io.bathy_io import read_altimetry
        p = tmp_path / 'ice.ati'
        p.write_text(self.LONG_ATI)
        ati = read_altimetry(p)
        assert ati.interpolation == 'L'
        assert ati.ranges.shape == (3,)
        assert np.allclose(ati.ranges, [0.0, 2500.0, 5000.0]), "km -> m"
        assert np.allclose(ati.sound_speed, [3500.0, 3400.0, 3300.0])
        assert np.allclose(ati.shear_speed, [1800.0, 1750.0, 1700.0])
        assert np.allclose(ati.density, [0.9, 0.91, 0.92])
        assert np.allclose(ati.shear_attenuation, [0.2, 0.21, 0.22])

    def test_short_format_yields_two_rows(self, tmp_path):
        from uacpy.io.bathy_io import read_altimetry, write_ati_file
        pairs = np.array([[0.0, 0.0], [2500.0, 2.0], [5000.0, 0.0]])
        p = tmp_path / 'flat.ati'
        write_ati_file(p, pairs, interp_type='C')
        ati = read_altimetry(p)
        assert ati.interpolation == 'C'
        assert not ati.is_long_format and ati.density is None
        assert np.allclose(ati.ranges, pairs[:, 0])
        assert np.allclose(ati.depths, pairs[:, 1])

    def test_truncated_long_row_raises_fileformaterror(self, tmp_path):
        """A long row may wrap across lines (one list-directed READ per
        point), so the typed error fires only when the file ends short of
        the row's 7 values."""
        from uacpy.io.bathy_io import read_altimetry
        from uacpy.core.exceptions import FileFormatError
        p = tmp_path / 'bad.ati'
        p.write_text("'LL'\n1\n0.0 0.0 3500.0\n")
        with pytest.raises(FileFormatError, match='file ended'):
            read_altimetry(p)

    def test_unknown_interp_type_raises_fileformaterror(self, tmp_path):
        from uacpy.io.bathy_io import read_altimetry
        from uacpy.core.exceptions import FileFormatError
        p = tmp_path / 'bad.ati'
        p.write_text("'ZS'\n1\n0.0 0.0\n")
        with pytest.raises(FileFormatError, match='altimetry type'):
            read_altimetry(p)


class TestATableBuildsOnlyTheCarrierItCanHold:
    """``read_bathymetry`` / ``read_altimetry`` keep every file Bellhop reads;
    ``to_bathymetry`` / ``to_altimetry`` refuse, naming the file, whatever
    the carrier cannot hold — never a silent approximation."""

    @staticmethod
    def _bty(tmp_path, body, name='t.bty'):
        p = tmp_path / name
        p.write_text(body)
        return p

    def test_a_linear_table_builds_the_carrier(self, tmp_path):
        from uacpy.core.bathymetry import Bathymetry
        from uacpy.io.bathy_io import read_bathymetry
        p = self._bty(tmp_path, "'L'\n2\n0.0 100.0\n1.0 150.0\n")
        bathy = read_bathymetry(p).to_bathymetry()
        assert type(bathy) is Bathymetry
        assert bathy.ranges.tolist() == [0.0, 1000.0]
        assert bathy.depths.tolist() == [100.0, 150.0]

    def test_curvilinear_interpolation_is_refused_by_name(self, tmp_path):
        from uacpy.io.bathy_io import read_bathymetry
        p = self._bty(tmp_path, "'C'\n2\n0.0 100.0\n1.0 150.0\n", 'curvy.bty')
        table = read_bathymetry(p)
        assert table.interpolation == 'C'
        with pytest.raises(ConfigurationError,
                           match=r"curvy\.bty: interpolation 'C'"):
            table.to_bathymetry()

    def test_a_long_format_table_is_refused(self, tmp_path):
        from uacpy.io.bathy_io import read_bathymetry
        p = self._bty(tmp_path, "'LL'\n2\n0.0 100.0 1600 0 1.8 0.5 0\n"
                                "1.0 150.0 1600 0 1.8 0.5 0\n", 'long.bty')
        with pytest.raises(ConfigurationError, match=r'long\.bty is a '
                                                     r'long-format'):
            read_bathymetry(p).to_bathymetry()

    def test_a_negative_range_is_kept_in_the_table_and_refused_by_the_carrier(
            self, tmp_path):
        from uacpy.io.bathy_io import read_bathymetry
        p = self._bty(tmp_path, "'L'\n2\n-1.0 100.0\n1.0 150.0\n", 'neg.bty')
        table = read_bathymetry(p)
        assert table.ranges.tolist() == [-1000.0, 1000.0]
        with pytest.raises(ConfigurationError, match=r'neg\.bty: '):
            table.to_bathymetry()

    def test_an_altimetry_table_is_not_a_bathymetry(self, tmp_path):
        from uacpy.io.bathy_io import read_altimetry
        p = self._bty(tmp_path, "'L'\n2\n0.0 1.0\n1.0 -2.0\n", 's.ati')
        table = read_altimetry(p)
        with pytest.raises(ConfigurationError, match='altimetry table'):
            table.to_bathymetry()
        alt = table.to_altimetry()
        # The file is positive down; the carrier's heights are positive up.
        assert alt.heights.tolist() == [-1.0, 2.0]

    def test_the_table_is_one_row_per_point(self, tmp_path):
        pytest.importorskip('pandas')
        from uacpy.io.bathy_io import read_bathymetry
        p = self._bty(tmp_path, "'L'\n2\n0.0 100.0\n1.0 150.0\n")
        frame = read_bathymetry(p).to_dataframe()
        assert list(frame.columns) == ['range', 'depth']
        assert frame['depth'].tolist() == [100.0, 150.0]
