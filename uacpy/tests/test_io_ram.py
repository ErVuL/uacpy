"""The RAM-family files: the RAMSurf reader's depth axis and the mpiramS
table writers."""

import numpy as np
import pytest
import warnings
from uacpy.core.exceptions import ConfigurationError


class TestRamsurfReaderDepthAxis:
    """The Collins PE grid maps stored index ``i`` to depth ``(i-1)*dz``.

    The two backends start their output stride at different indices:
    ``ramsurf/ramsurf1.5.f:437`` dumps ``do i = ndz, nzplt, ndz`` while
    ``ramsurf/rams0.5.f:262`` dumps ``do i = 1+ndz, nzplt, ndz``, one grid
    node deeper. ``depth_index_offset`` carries that difference, and both
    backends must land on the same ``(i-1)*dz`` depths.
    """

    @staticmethod
    def _write_grid(path, lz, n_records, dtype='<f8'):
        """Write a synthetic Collins tl.grid (little-endian Fortran records).

        ``f8`` by default: the Collins binaries are built with
        ``-fdefault-real-8`` (third_party/MODIFICATIONS.md), so a uacpy
        ``tl.grid`` carries 8-byte records; ``'<f4'`` is a stock build's.
        """
        import struct
        with open(path, 'wb') as f:
            f.write(struct.pack('<i', 4) + struct.pack('<i', lz) + struct.pack('<i', 4))
            for r in range(n_records):
                payload = np.arange(lz, dtype=dtype).tobytes()
                f.write(struct.pack('<i', len(payload)) + payload
                        + struct.pack('<i', len(payload)))

    def test_ramsurf_first_sample_is_surface_node(self, tmp_path):
        from uacpy.io.ramsurf_reader import read_tl_grid
        p = tmp_path / "tl.grid"
        self._write_grid(p, lz=5, n_records=3)
        depths = read_tl_grid(p, dr=10.0, ndr=1, dz=2.0, ndz=1,
                              depth_index_offset=0).depths
        # ramsurf: i = k*ndz, depth = (i-1)*dz -> first sample at z=0.
        assert depths[0] == 0.0
        assert np.allclose(depths, np.array([0.0, 2.0, 4.0, 6.0, 8.0]))

    def test_rams_skips_surface_node(self, tmp_path):
        from uacpy.io.ramsurf_reader import read_tl_grid
        p = tmp_path / "tl.grid"
        self._write_grid(p, lz=5, n_records=3)
        depths = read_tl_grid(p, dr=10.0, ndr=1, dz=2.0, ndz=1,
                              depth_index_offset=1).depths
        # rams: i = 1 + k*ndz, depth = (i-1)*dz -> first sample at z=dz.
        assert depths[0] == 2.0
        assert np.allclose(depths, np.array([2.0, 4.0, 6.0, 8.0, 10.0]))

    def test_a_single_precision_grid_reads_like_the_double_one(self, tmp_path):
        """A stock build's 4-byte records are recognised by their length."""
        from uacpy.io.ramsurf_reader import read_tl_grid
        self._write_grid(tmp_path / 'd.grid', lz=5, n_records=3)
        self._write_grid(tmp_path / 's.grid', lz=5, n_records=3, dtype='<f4')
        kw = dict(dr=10.0, ndr=1, dz=2.0, ndz=1)
        tl_d = read_tl_grid(tmp_path / 'd.grid', **kw).data
        tl_s = read_tl_grid(tmp_path / 's.grid', **kw).data
        assert tl_s.dtype == np.float64
        np.testing.assert_array_equal(tl_s, tl_d)

    def test_a_partial_trailing_record_warns(self, tmp_path):
        from uacpy.io.ramsurf_reader import read_tl_grid
        p = tmp_path / 'tl.grid'
        self._write_grid(p, lz=5, n_records=3)
        p.write_bytes(p.read_bytes()[:-10])
        with pytest.warns(UserWarning, match='2 range step'):
            grid = read_tl_grid(p, dr=10.0, ndr=1, dz=2.0, ndz=1)
        assert grid.data.shape == (5, 2) and grid.ranges.size == 2

    def test_each_reader_names_the_quantity_it_returns(self, tmp_path):
        """A tl.grid is transmission loss in dB; a pcomplex.bin is the PE
        envelope before the carrier and the Hankel phase, which must never
        read as a pressure."""
        from uacpy.io.ramsurf_reader import read_pcomplex_grid, read_tl_grid
        self._write_grid(tmp_path / 'tl.grid', lz=5, n_records=3)
        self._write_grid(tmp_path / 'pcomplex.bin', lz=5, n_records=3,
                         dtype='<c16')
        kw = dict(dr=10.0, ndr=1, dz=2.0, ndz=1)
        tl = read_tl_grid(tmp_path / 'tl.grid', **kw)
        env = read_pcomplex_grid(tmp_path / 'pcomplex.bin', **kw)
        assert (tl.quantity, tl.unit) == ('transmission_loss', 'dB')
        assert (env.quantity, env.unit) == ('pe_envelope', '')
        assert np.iscomplexobj(env.data)
        pytest.importorskip('xarray')
        ds = env.to_xarray()
        assert ds['data'].dims == ('depth', 'range')
        assert ds['depth'].attrs['units'] == 'm'

    def test_a_quantity_carries_only_its_own_unit(self):
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.io import PeGrid
        with pytest.raises(ConfigurationError, match="'pe_envelope'"):
            PeGrid(quantity='pe_envelope', unit='Pa', ranges=[1.0],
                   depths=None, data=[1j])

    def test_a_whole_grid_reads_without_a_warning(self, tmp_path):
        from uacpy.io.ramsurf_reader import read_tl_grid
        p = tmp_path / 'tl.grid'
        self._write_grid(p, lz=5, n_records=3)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            read_tl_grid(p, dr=10.0, ndr=1, dz=2.0, ndz=1)


class TestRamWritersCheckShapeBeforeWriting:
    """The mpiramS table writers take one row per depth and refuse any
    other shape before the file is opened: a size-matching transpose
    reshaped silently into a row-major scramble, and a short ``speeds``
    crashed mid-write."""

    def test_a_transposed_attenuation_table_is_refused(self, tmp_path):
        from uacpy.io.mpirams_writer import write_water_attenuation_file
        out = tmp_path / 'w.dat'
        with pytest.raises(ConfigurationError, match=r'\(2, 3\)'):
            write_water_attenuation_file(
                out, [0.0, 50.0], [100.0, 200.0, 300.0],
                np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        assert not out.exists()

    def test_the_documented_shapes_are_written(self, tmp_path):
        from uacpy.io.mpirams_writer import write_water_attenuation_file
        out = tmp_path / 'w.dat'
        write_water_attenuation_file(out, [0.0, 50.0], [100.0, 200.0, 300.0],
                                     np.array([[1.0, 2.0, 3.0],
                                               [4.0, 5.0, 6.0]]))
        assert out.read_text().splitlines()[3].split() == ['50', '4', '5', '6']
        # One frequency: a (nz,) vector is the one column.
        write_water_attenuation_file(out, [0.0, 50.0], [100.0], [1.0, 2.0])
        assert out.read_text().splitlines()[3].split() == ['50', '2']

    @pytest.mark.parametrize('speeds', [np.full((2, 2), 1500.0),
                                        np.full((4, 2), 1500.0),
                                        np.full(2, 1500.0)])
    def test_speeds_without_one_row_per_depth_are_refused(
            self, tmp_path, speeds):
        from uacpy.io.mpirams_writer import write_ssp_file
        out = tmp_path / 's.ssp'
        with pytest.raises(ConfigurationError, match='one row per depth|'
                                                     'n_depths'):
            write_ssp_file(out, np.array([0.0, 50.0, 100.0]), speeds,
                           ranges=np.array([0.0, 1000.0]))
        assert not out.exists()
