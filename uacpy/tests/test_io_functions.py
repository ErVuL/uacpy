"""Tests for I/O functions.

Two kinds of contract live here: the readers and writers under ``uacpy/io``
checked against hand-written files, and the record layout of the decks uacpy
emits, checked by running the vendored Acoustics-Toolbox binaries on them
(``@pytest.mark.requires_binary``). Both pin what the Fortran reads, so the
authority for every expectation is the vendored source, cited at the assertion.
"""

import re

import numpy as np
import pytest
from pathlib import Path
import tempfile

import uacpy
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.environment import SoundSpeedProfile
from uacpy.core.results import Field
from uacpy.io.file_manager import FileManager


class TestFileManager:
    """Tests for FileManager class."""

    def test_file_manager_creation(self):
        """Test creating FileManager."""
        fm = FileManager(use_tmpfs=False, base_dir=None, cleanup=True)
        assert fm is not None

    def test_file_manager_work_dir_creation(self):
        """Test creating work directory."""
        fm = FileManager(use_tmpfs=False, base_dir=None, cleanup=True)
        fm.create_work_dir()

        assert fm.work_dir is not None
        assert fm.work_dir.exists()
        assert fm.work_dir.is_dir()

        # Cleanup
        fm.cleanup_work_dir()

    def test_file_manager_get_path(self):
        """Test getting file path."""
        fm = FileManager(use_tmpfs=False, base_dir=None, cleanup=True)
        fm.create_work_dir()

        file_path = fm.get_path('test.txt')
        assert file_path.parent == fm.work_dir
        assert file_path.name == 'test.txt'

        # Cleanup
        fm.cleanup_work_dir()

    def test_file_manager_custom_base_dir(self):
        """Test FileManager with custom base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fm = FileManager(use_tmpfs=False, base_dir=Path(tmpdir), cleanup=False)
            fm.create_work_dir()

            assert fm.work_dir.parent == Path(tmpdir)
            assert fm.work_dir.exists()

    def test_file_manager_cleanup(self):
        """Test FileManager cleanup."""
        fm = FileManager(use_tmpfs=False, base_dir=None, cleanup=True)
        fm.create_work_dir()

        work_dir = fm.work_dir
        assert work_dir.exists()

        fm.cleanup_work_dir()
        assert not work_dir.exists()


class TestEnvironmentIO:
    """Tests for Environment I/O."""

    def test_environment_with_bathymetry_file(self):
        """Test loading environment with bathymetry from file."""
        # Create temporary bathymetry file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("# Range(m) Depth(m)\n")
            f.write("0 80\n")
            f.write("5000 100\n")
            f.write("10000 120\n")
            bathy_file = f.name

        try:
            # Load bathymetry
            bathymetry = np.loadtxt(bathy_file)

            env = uacpy.Environment(
                name="Test",
                ssp=1500.0,
                bathymetry=bathymetry
            )

            assert env.is_range_dependent
            assert env.bathymetry.n_ranges == 3
            assert env.bathymetry.depths[0] == 80
            assert env.bathymetry.depths[-1] == 120

        finally:
            Path(bathy_file).unlink()

    def test_environment_ssp_from_array(self):
        """Test creating environment with SSP from array."""
        depths = np.linspace(0, 100, 11)
        sound_speeds = 1500 + depths * 0.1  # Linear gradient

        ssp_data = np.column_stack([depths, sound_speeds])

        env = uacpy.Environment(
            name="Test",
            bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs(ssp_data)
        )

        assert len(env.ssp.to_pairs()) == 11
        assert np.allclose(env.ssp.to_pairs()[:, 0], depths)


class TestFieldIO:
    """Tests for Field I/O operations."""

    @staticmethod
    def _make_field(**meta):
        return Field(
            data=np.random.rand(10, 20),
            coords={
                'depth': np.linspace(10, 90, 10),
                'range': np.linspace(100, 5000, 20),
            },
            model='Bellhop', frequencies=100.0,
            metadata=meta,
        )

    def test_field_metadata_preservation(self):
        field = self._make_field(source_depth=50.0, custom_param='test_value')
        assert field.model == 'Bellhop'
        assert field.f0 == 100.0
        assert list(field.frequencies) == [100.0]
        assert field.metadata['source_depth'] == 50.0
        assert field.metadata['custom_param'] == 'test_value'

    def test_field_deepcopy_preserves_metadata(self):
        import copy as _copy
        field = self._make_field(test_key='test_value')
        field_copy = _copy.deepcopy(field)
        assert field_copy.metadata['test_key'] == 'test_value'
        assert field_copy.metadata is not field.metadata


# Reference Acoustics-Toolbox SSP files vendored under third_party.
_AT_REF_DIR = (Path(__file__).resolve().parent.parent /
               "third_party" / "Acoustics-Toolbox" / "tests")


class TestSSPReadWriteRoundtrip:
    """Round-trip and canonical-file tests for the AT/Bellhop .ssp readers."""

    def test_read_ssp_2d_canonical_munk_file(self):
        """read_ssp_2d must parse the canonical AT MunkB_geo_rot.ssp layout
        (NProf alone on line 1, range vector on line 2, then one SSP row
        per depth)."""
        from uacpy.io.oalib_reader import read_ssp_2d
        path = _AT_REF_DIR / "Munk" / "MunkB_geo_rot.ssp"
        if not path.exists():
            pytest.skip(f"reference AT file missing: {path}")
        r = read_ssp_2d(path)
        # File header advertises 30 profiles.
        assert r['n_prof'] == 30
        assert r['r_prof'].shape == (30,)
        # First/last ranges on disk are -50 km and 10 km; the reader
        # returns metres (uacpy is SI-internal, km only on disk).
        assert r['r_prof'][0] == -50_000.0
        assert r['r_prof'][-1] == 10_000.0
        # File has 2 depth rows.
        assert r['c_mat'].shape == (2, 30)
        # Spot-check one entry against the file.
        assert r['c_mat'][0, 2] == pytest.approx(1548.52)

    def test_write_then_read_ssp_2d_roundtrip(self, tmp_path):
        """write_ssp followed by read_ssp_2d returns the same matrix."""
        from uacpy.io.oalib_writer import write_ssp
        from uacpy.io.oalib_reader import read_ssp_2d

        ranges_m = np.array([0.0, 5.0, 10.0, 20.0])
        # 5 depths x 4 ranges
        c = np.array([
            [1500.0, 1502.0, 1504.0, 1505.0],
            [1495.0, 1497.0, 1499.0, 1500.5],
            [1490.0, 1492.0, 1494.0, 1495.5],
            [1488.0, 1489.5, 1491.0, 1492.5],
            [1487.0, 1488.0, 1489.0, 1490.0],
        ])
        out = tmp_path / "rt.ssp"
        write_ssp(out, ranges_m, c)
        result = read_ssp_2d(out)
        assert result['n_prof'] == 4
        assert result['c_mat'].shape == (5, 4)
        # The range axis goes to disk as km at ``%.6f`` and comes back in
        # metres, so the quantum on this axis is 1 mm.
        np.testing.assert_allclose(result['r_prof'], ranges_m, atol=1e-3)
        # Speeds go out at ``%8.4f``; the format quantises at 1e-4 m/s, well
        # inside this bound.
        np.testing.assert_allclose(result['c_mat'], c, atol=0.5)

    def test_write_ssp_rejects_mismatched_shape(self, tmp_path):
        """write_ssp must reject a range vector that does not match
        c.shape[1] (otherwise a silently-malformed .ssp would be written)."""
        from uacpy.io.oalib_writer import write_ssp

        c = np.zeros((3, 4))
        out = tmp_path / "bad.ssp"
        with pytest.raises(ConfigurationError, match="does not match"):
            write_ssp(out, np.array([0.0, 5.0]), c)
        with pytest.raises(ConfigurationError, match="2-D"):
            write_ssp(out, np.array([0.0, 5.0]), np.zeros(5))

    def test_read_ssp_3d_canonical_bellhop3d_file(self):
        """read_ssp_3d must parse Bellhop3D's Munk3D.ssp (vectors and
        per-(z,y) SSP rows each on one line with Nx values)."""
        from uacpy.io.oalib_reader import read_ssp_3d
        path = (_AT_REF_DIR / "Bellhop3DTests" / "MunkRot" / "Munk3D.ssp")
        if not path.exists():
            pytest.skip(f"reference Bellhop3D file missing: {path}")
        r = read_ssp_3d(path)
        assert r['Nx'] == 27
        assert r['Ny'] == 3
        assert r['Nz'] == 7
        assert r['Segx'].shape == (27,)
        assert r['Segy'].shape == (3,)
        assert r['Segz'].shape == (7,)
        # Segx/Segy are km on disk and returned in metres; Segz is already m.
        np.testing.assert_allclose(r['Segy'], [0.0, 100_000.0, 200_000.0])
        np.testing.assert_allclose(
            r['Segz'], [0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0]
        )
        assert r['c_mat'].shape == (7, 3, 27)
        # Munk3D.ssp repeats the same SSP at every (z,y); spot-check.
        assert r['c_mat'][0, 0, 0] == pytest.approx(1549.617363)
        assert r['c_mat'][6, 2, 0] == pytest.approx(1549.617363)


class TestArrivalsReaderTokenStream:
    """``read_arr_file`` must tolerate Fortran records that wrap to multiple
    text lines (different compilers wrap list-directed WRITE at different
    column widths)."""

    @staticmethod
    def _write_arr(path, lines):
        with open(path, 'w') as f:
            f.write("'2D'\n")
            for ln in lines:
                f.write(ln + '\n')

    def _expected_arr_lines(self):
        """A single-source, 1-receiver-depth, 2-range, 1-arrival/receiver
        canonical ASCII record stream, ready for write."""
        return [
            "100.0",              # freq
            "1 50.0",             # nsd, sz
            "1 75.0",             # nrd, rz
            "2 500.0 1000.0",     # nrr, rr
            "1",                  # max-narr (per source, unused by reader)
            # rcv (irz=0, irr=0)
            "1",                  # narr
            "0.5 0.0 0.001 0.0 -5.0 5.0 0 1",
            # rcv (irz=0, irr=1)
            "1",                  # narr
            "0.3 0.0 0.0015 0.0 -7.0 7.0 1 2",
        ]

    def test_read_arr_file_canonical_singleline(self, tmp_path):
        from uacpy.io.oalib_reader import read_arr_file
        path = tmp_path / "canon.arr"
        self._write_arr(path, self._expected_arr_lines())
        result = read_arr_file(path)
        assert float(result.frequencies[0]) == pytest.approx(100.0)
        assert result.source_depths.tolist() == [50.0]
        assert result.receiver_depths.tolist() == [75.0]
        assert result.receiver_ranges.tolist() == [500.0, 1000.0]
        a0 = result.by_receiver[0][0][0]
        assert a0['n_arrivals'] == 1
        assert a0['amplitudes'][0] == pytest.approx(0.5)
        assert a0['n_top_bounces'][0] == 0
        assert a0['n_bot_bounces'][0] == 1
        a1 = result.by_receiver[0][0][1]
        assert a1['amplitudes'][0] == pytest.approx(0.3)
        assert a1['n_bot_bounces'][0] == 2

    def test_read_arr_file_with_wrapped_records(self, tmp_path):
        """Simulates an Intel-Fortran-style wrap: the 8-token arrival record
        spans two text lines. The parser must still recover the record."""
        from uacpy.io.oalib_reader import read_arr_file

        # Take the canonical stream, but break the 8-token arrival line
        # in half across two text lines.
        canonical = self._expected_arr_lines()
        # Replace the two arrival lines with wrapped versions.
        wrapped = []
        for ln in canonical:
            tokens = ln.split()
            # Wrap any 8-token arrival line (amp, phase, dr, di, sa, ra, nt, nb).
            if len(tokens) == 8:
                wrapped.append(' '.join(tokens[:4]))
                wrapped.append(' '.join(tokens[4:]))
            else:
                wrapped.append(ln)

        path = tmp_path / "wrapped.arr"
        self._write_arr(path, wrapped)
        result = read_arr_file(path)
        a0 = result.by_receiver[0][0][0]
        assert a0['n_arrivals'] == 1
        assert a0['amplitudes'][0] == pytest.approx(0.5)
        assert a0['n_top_bounces'][0] == 0
        assert a0['n_bot_bounces'][0] == 1
        a1 = result.by_receiver[0][0][1]
        assert a1['amplitudes'][0] == pytest.approx(0.3)
        assert a1['n_bot_bounces'][0] == 2


class TestFieldFlpWriter:
    """Coverage for write_fieldflp's NRro / Rro emission."""

    def test_write_fieldflp_emits_at_subtab_idiom(self, tmp_path):
        """NRro must equal NRz (``KrakenField/field.f90:149-151`` ERROUTs
        otherwise) and the Rro line must be the AT ``0.0 /`` sentinel idiom
        (single value + slash terminator)."""
        from uacpy.io.oalib_writer import write_fieldflp

        out = tmp_path / "test.flp"
        pos = {
            's': {'z': np.array([50.0])},
            'r': {
                'z': np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
                # field.f90 wants r in km internally; the writer converts
                # m -> km by dividing by 1000, so pass meters here.
                'r': np.array([500.0, 1000.0, 1500.0, 2000.0]),
            },
        }
        write_fieldflp(
            filepath=out,
            option='RA  ',
            pos=pos,
            title='unit-test',
            M_limit=999,
        )
        text = out.read_text()
        # NRro must equal NRz (5 receivers).
        nrro_lines = [ln for ln in text.splitlines() if 'NRro' in ln]
        assert len(nrro_lines) == 1
        assert nrro_lines[0].split()[0] == '5'
        # The Rro record (comment starts with 'Rro(') must contain exactly
        # one value followed by ``/``.
        rro_lines = [ln for ln in text.splitlines() if 'Rro(' in ln]
        assert len(rro_lines) == 1
        data_part = rro_lines[0].split('!')[0]
        assert '/' in data_part
        nums = data_part.replace('/', ' ').split()
        assert nums == ['0.0'], f"expected single zero + slash, got {nums!r}"

    @pytest.mark.parametrize('n_rz', [1, 2])
    def test_write_fieldflp_writes_explicit_rro_below_subtab_threshold(
            self, tmp_path, n_rz):
        """With NRz < 3 the Rro vector must be written out in full.

        AT's SubTab only replicates a sentinel-terminated vector when
        ``Nx >= 3`` (misc/subtabulate.f90:24). Below that the ``0.0 /``
        idiom leaves ``x(2)`` at ReadVector's -999.9 pre-fill
        (misc/SourceReceiverPositions.f90:219-221), and the following ``Sort``
        (:224) moves it to Rro(1) — giving the *shallowest* receiver a
        -999.9 m range offset, which ``KrakenField/EvaluateMod.f90:73``
        applies as ``1/sqrt(r + ro)``.
        """
        from uacpy.io.oalib_writer import write_fieldflp

        out = tmp_path / f"n{n_rz}.flp"
        write_fieldflp(
            filepath=out, option='RA  ',
            pos={'s': {'z': np.array([50.0])},
                 'r': {'z': np.linspace(10.0, 40.0, n_rz),
                       'r': np.array([1000.0, 2000.0])}},
            title='unit-test', M_limit=999,
        )
        rro = [ln for ln in out.read_text().splitlines() if 'Rro(' in ln][0]
        values = rro.split('!')[0].replace('/', ' ').split()
        assert len(values) == n_rz, (
            f"NRz={n_rz} needs {n_rz} explicit Rro values (SubTab will not "
            f"replicate below 3); got {values}")
        assert all(float(v) == 0.0 for v in values)


class TestRamsurfReaderDepthAxis:
    """The Collins PE grid maps stored index ``i`` to depth ``(i-1)*dz``.

    The two backends start their output stride at different indices:
    ``ramsurf/ramsurf1.5.f:437`` dumps ``do i = ndz, nzplt, ndz`` while
    ``ramsurf/rams0.5.f:262`` dumps ``do i = 1+ndz, nzplt, ndz``, one grid
    node deeper. ``depth_index_offset`` carries that difference, and both
    backends must land on the same ``(i-1)*dz`` depths.
    """

    @staticmethod
    def _write_grid(path, lz, n_records):
        """Write a synthetic Collins tl.grid (little-endian Fortran records)."""
        import struct
        with open(path, 'wb') as f:
            f.write(struct.pack('<i', 4) + struct.pack('<i', lz) + struct.pack('<i', 4))
            for r in range(n_records):
                payload = np.arange(lz, dtype='<f4').tobytes()
                f.write(struct.pack('<i', len(payload)) + payload
                        + struct.pack('<i', len(payload)))

    def test_ramsurf_first_sample_is_surface_node(self, tmp_path):
        from uacpy.io.ramsurf_reader import read_tl_grid
        p = tmp_path / "tl.grid"
        self._write_grid(p, lz=5, n_records=3)
        _, depths, _ = read_tl_grid(p, dr=10.0, ndr=1, dz=2.0, ndz=1,
                                    depth_index_offset=0)
        # ramsurf: i = k*ndz, depth = (i-1)*dz -> first sample at z=0.
        assert depths[0] == 0.0
        assert np.allclose(depths, np.array([0.0, 2.0, 4.0, 6.0, 8.0]))

    def test_rams_skips_surface_node(self, tmp_path):
        from uacpy.io.ramsurf_reader import read_tl_grid
        p = tmp_path / "tl.grid"
        self._write_grid(p, lz=5, n_records=3)
        _, depths, _ = read_tl_grid(p, dr=10.0, ndr=1, dz=2.0, ndz=1,
                                    depth_index_offset=1)
        # rams: i = 1 + k*ndz, depth = (i-1)*dz -> first sample at z=dz.
        assert depths[0] == 2.0
        assert np.allclose(depths, np.array([2.0, 4.0, 6.0, 8.0, 10.0]))


class TestShdNoDataCells:
    """Exact-zero SHD pressure cells — grid points the engine never wrote
    (Bellhop no-ray cells, an empty KRAKEN modal sum) — surface as NaN,
    uacpy's no-data convention."""

    def test_read_shd_asc_zero_pressure_is_nan(self, tmp_path):
        from uacpy.io.oalib_reader import read_shd_asc
        # 1 freq, 1 theta, 1 source, 2 receiver depths, 2 ranges; the
        # (depth 0, range 0) cell is an exact complex zero = no data.
        header = [
            "'test'", "'rectilin'",
            "1 1 1 2 2", "100.0 0.0",
            "100.0",             # freq_vec
            "0.0",               # theta
            "50.0",              # s_z
            "10.0", "20.0",      # r_z
            "100.0", "200.0",    # r_r
        ]
        pressure_rows = [
            "0.0", "0.0", "0.5", "0.25",     # depth 10 m: no-data, 0.5+0.25j
            "1.0", "0.0", "2.0", "-1.0",     # depth 20 m: 1+0j, 2-1j
        ]
        p = tmp_path / "t.shd.asc"
        p.write_text("\n".join(header + pressure_rows) + "\n")
        pr = read_shd_asc(p)['pressure']
        assert pr.shape == (2, 2)
        assert np.isnan(pr[0, 0])
        assert pr[0, 1] == pytest.approx(0.5 + 0.25j)
        assert pr[1, 0] == pytest.approx(1.0 + 0.0j)
        assert pr[1, 1] == pytest.approx(2.0 - 1.0j)


class TestGrnPhaseSpeedTaper:
    """The cmin/cmax phase-speed taper: valid bands taper the spectrum edges;
    a band with no overlap with the file's phase-speed grid raises a typed
    ConfigurationError instead of a raw broadcast ValueError."""

    def _k(self, freq=100.0, c_lo=1400.0, c_hi=1700.0, nk=64):
        # Wavenumber grid spanning phase speeds [c_lo, c_hi] at ``freq``.
        omega = 2.0 * np.pi * freq
        return np.linspace(omega / c_hi, omega / c_lo, nk)

    def test_interior_band_tapers_edges(self):
        from uacpy.io.grn_reader import _hanning_taper
        win = _hanning_taper(self._k(), 100.0, cmin=1450.0, cmax=1650.0)
        assert win.shape == (64,)
        assert np.all((win >= 0.0) & (win <= 1.0))
        assert win[0] < 1.0 and win[-1] < 1.0     # rolled off at both edges
        assert np.any(win == 1.0)                 # flat in the middle

    def test_generous_bounds_are_a_noop(self):
        from uacpy.io.grn_reader import _hanning_taper
        win = _hanning_taper(self._k(), 100.0, cmin=1000.0, cmax=3000.0)
        assert np.all(win == 1.0)

    @pytest.mark.parametrize("cmin,cmax", [
        (None, 100.0),      # cmax below the grid's slowest phase speed
        (5000.0, None),     # cmin above the grid's fastest phase speed
        (1650.0, 1450.0),   # inverted band (cmin > cmax)
    ])
    def test_no_overlap_band_raises_typed(self, cmin, cmax):
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.io.grn_reader import _hanning_taper
        with pytest.raises(ConfigurationError):
            _hanning_taper(self._k(), 100.0, cmin=cmin, cmax=cmax)

    def test_public_grn_to_field_raises_typed(self):
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.io.grn_reader import grn_to_field
        nk = 32
        freq = 100.0
        cvec = np.linspace(1700.0, 1400.0, nk)     # decreasing, as on disk
        grn = {
            "freq": freq, "freqVec": np.array([freq]), "nfreq": 1,
            "nsd": 1, "nrd": 2, "nk": nk,
            "sd": np.array([50.0]), "rd": np.array([25.0, 75.0]),
            "cVec": cvec, "atten": 0.0,
            "G": np.ones((1, 1, 2, nk), dtype=np.complex64),
            "title": "SCOOTER test", "PlotType": "Green", "is_sparc": False,
        }
        with pytest.raises(ConfigurationError, match="no overlap"):
            grn_to_field(grn, np.array([1000.0]), cmax=100.0)


class TestReaderCorruptFileRaises:
    """The io readers' failure path: a truncated / garbage binary must raise
    the typed :class:`FileFormatError`, not a bare struct/EOF error or a
    self-contradictory ModelExecutionError(return_code=0)."""

    def test_corrupt_mode_file_raises_fileformaterror(self, tmp_path):
        from uacpy.io.modes_reader import read_modes_bin
        from uacpy.core.exceptions import FileFormatError
        bad = tmp_path / "garbage.mod"
        bad.write_bytes(b"\x00\x01\x02not a real mode file\xff\xfe" * 4)
        with pytest.raises(FileFormatError):
            read_modes_bin(str(bad))

    def test_corrupt_shd_file_raises(self, tmp_path):
        from uacpy.io.oalib_reader import read_shd_bin
        from uacpy.core.exceptions import FileFormatError
        bad = tmp_path / "garbage.shd"
        bad.write_bytes(b"\x00" * 12)            # too short for a valid header
        with pytest.raises(FileFormatError, match='cannot resolve byte order'):
            read_shd_bin(str(bad))


class TestOASRFrequencyOverride:
    """An explicit ``frequencies=`` override (passed by OASR.run as
    freq_min/freq_max/n_frequencies) must win over a multi-element
    ``source.frequencies`` — honouring it only for single-frequency sources
    would silently drop it for every sweep."""

    def _freq_line(self, path):
        # The OASR deck writes "<fmin> <fmax> <nfreq> <out_inc>" right after the
        # source line; locate it by the 4-token float/int signature.
        for ln in Path(path).read_text().splitlines():
            toks = ln.split()
            if len(toks) == 4:
                try:
                    return float(toks[0]), float(toks[1]), int(toks[2])
                except ValueError:
                    continue
        raise AssertionError("no frequency line found in OASR deck")

    def test_override_wins_over_multifreq_source(self, tmp_path):
        from uacpy.io.oases_writer import write_oasr_input
        from uacpy.core import Environment, Source, Receiver, BoundaryProperties
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=BoundaryProperties(acoustic_type='half-space',
                                                    sound_speed=1600.0,
                                                    density=1.8,
                                                    attenuation=0.5))
        # Multi-element source: must not hijack the sweep.
        src = Source(depths=50.0, frequencies=[50.0, 100.0, 150.0])
        rcv = Receiver(depths=[50.0], ranges=[1000.0])
        out = tmp_path / "oasr.dat"
        write_oasr_input(str(out), env, src, rcv,
                         angles=np.linspace(0, 90, 91),
                         freq_min=200.0, freq_max=400.0, n_frequencies=5)
        fmin, fmax, nfreq = self._freq_line(out)
        assert (fmin, fmax, nfreq) == (200.0, 400.0, 5)

    def test_source_drives_sweep_without_override(self, tmp_path):
        from uacpy.io.oases_writer import write_oasr_input
        from uacpy.core import Environment, Source, Receiver, BoundaryProperties
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=BoundaryProperties(acoustic_type='half-space',
                                                    sound_speed=1600.0,
                                                    density=1.8,
                                                    attenuation=0.5))
        src = Source(depths=50.0, frequencies=[50.0, 100.0, 150.0])
        rcv = Receiver(depths=[50.0], ranges=[1000.0])
        out = tmp_path / "oasr.dat"
        write_oasr_input(str(out), env, src, rcv,
                         angles=np.linspace(0, 90, 91))
        fmin, fmax, nfreq = self._freq_line(out)
        assert (fmin, fmax, nfreq) == (50.0, 150.0, 3)


class TestHankelScaledCylindrical:
    """fieldsco.m:133 — 'S' is 'R' with cylindrical spreading removed."""

    def test_scaled_equals_point_times_sqrt_r(self):
        from uacpy.io.grn_reader import _hankel_transform
        rng = np.random.default_rng(0)
        k = np.linspace(0.1, 1.0, 64)
        G = rng.normal(size=(2, 64)) + 1j * rng.normal(size=(2, 64))
        ranges = np.array([500.0, 1000.0, 2000.0])
        kw = dict(atten=0.0, spectrum='P')
        p_point = _hankel_transform(G, k, ranges, source_type='R', **kw)
        p_scaled = _hankel_transform(G, k, ranges, source_type='S', **kw)
        np.testing.assert_allclose(
            p_scaled, p_point * np.sqrt(ranges)[None, :], rtol=1e-12)

    def test_unknown_source_type_still_raises(self):
        from uacpy.io.grn_reader import _hankel_transform
        with pytest.raises(ConfigurationError):
            _hankel_transform(
                np.zeros((1, 4), dtype=complex), np.linspace(0.1, 1.0, 4),
                np.array([100.0]), atten=0.0, source_type='Q', spectrum='P')


class TestConstantAbsorptionColumn:
    """A ConstantAbsorption baseline must land in AT's alphaI column.

    ``misc/sspMod.f90:334`` reads each SSP line as
    ``z, alphaR (cp), betaR (cs), rhoR, alphaI, betaI``. Writing the baseline
    third makes it the water column's *shear speed*: Kraken then returns an
    all-NaN field and Scooter segfaults.
    """

    @staticmethod
    def _env(value):
        import uacpy
        from uacpy.core.absorption import ConstantAbsorption
        return uacpy.Environment(
            name='abs', bathymetry=200.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0,
                density=1.8, attenuation=0.5),
            absorption=ConstantAbsorption(value))

    def test_baseline_goes_in_the_attenuation_column(self, tmp_path):
        from uacpy.io.oalib_writer import write_ssp_section
        out = tmp_path / 'ssp.txt'
        with open(out, 'w') as f:
            write_ssp_section(f, self._env(0.5), bottom_depth=200.0)
        rows = [ln for ln in out.read_text().splitlines() if '/' in ln][1:]
        assert rows, "no SSP sample rows written"
        for ln in rows:
            cols = ln.replace('/', ' ').split()
            assert len(cols) >= 5, f"need z cp cs rho alphaI; got {cols}"
            assert float(cols[2]) == 0.0, f"cs must be 0 for water; got {cols[2]}"
            assert float(cols[4]) == pytest.approx(0.5), (
                f"absorption must be in alphaI (col 5); got {cols}")

    def test_zero_absorption_still_pins_all_six_columns(self, tmp_path):
        """The short ``z c /`` form is unsafe even at zero absorption: AT's
        ``/`` terminator leaves the remaining items at their previous value,
        and ``TopBot`` (ReadEnvironmentMod.f90:285) has already loaded the top
        half-space into those module variables. All six columns are pinned."""
        import uacpy
        from uacpy.io.oalib_writer import write_ssp_section
        env = uacpy.Environment(name='p', bathymetry=200.0, ssp=1500.0)
        out = tmp_path / 'ssp2.txt'
        with open(out, 'w') as f:
            write_ssp_section(f, env, bottom_depth=200.0)
        rows = [ln for ln in out.read_text().splitlines() if '/' in ln][1:]
        assert rows, "no SSP sample rows written"
        for ln in rows:
            cols = ln.replace('/', ' ').split()
            assert len(cols) == 6, f"expected all six AT columns; got {cols}"
            assert float(cols[2]) == 0.0 and float(cols[4]) == 0.0


class TestOaspTrfFrequencyAxis:
    """TRF bin indices are 1-based, so bin k is (k-1)*DLFRQ.

    ``oasiun22.f:1256-1261`` sets ``DLFRQP = 1/(DT*NX)`` and
    ``LX = nint(FMIN/DLFRQP + 1)``. Reading bin k as ``k/(dt*nx)`` shifts the
    whole axis up by one bin.
    """

    @staticmethod
    def _axis(lx, mx, nx, dt):
        """The reader's frequency-axis expression, isolated."""
        return np.array([((k - 1) / (dt * nx)) for k in range(lx, mx + 1)],
                        dtype=np.float64)

    def test_round_trips_the_oases_index_formula(self):
        nx, dt = 1024, 1.0 / 4096.0
        dlfrq = 1.0 / (dt * nx)
        f_min, f_max = 100.0, 400.0
        lx = int(round(f_min / dlfrq + 1))       # oasiun22.f:1259
        mx = int(round(f_max / dlfrq + 1))       # oasiun22.f:1261
        axis = self._axis(lx, mx, nx, dt)
        assert axis[0] == pytest.approx(f_min, abs=0.5 * dlfrq)
        assert axis[-1] == pytest.approx(f_max, abs=0.5 * dlfrq)

    def test_first_bin_is_dc_not_one_bin_up(self):
        nx, dt = 512, 1.0 / 2048.0
        assert self._axis(1, 4, nx, dt)[0] == pytest.approx(0.0)

    def test_spacing_is_the_dft_bin_width(self):
        nx, dt = 2048, 1.0 / 8192.0
        axis = self._axis(10, 20, nx, dt)
        np.testing.assert_allclose(np.diff(axis), 1.0 / (dt * nx), rtol=1e-12)


class TestRoughnessGoesToItsOwnInterface:
    """Surface and seabed roughness must come from their own carriers.

    AT reads the medium mesh line as NG, SSP%sigma(Medium), Depth(Medium+1)
    (misc/ReadEnvironmentMod.f90:81-88), so the water column's line is
    sigma(1) — the *sea surface*. The seabed is sigma(NMedia+1), written on
    the bottom halfspace line (:121). The two therefore cannot share one
    model-level knob: each roughness comes from its own carrier.
    """

    @staticmethod
    def _env(surf=0.0, bot=0.0):
        import uacpy
        e = uacpy.Environment(
            name='r', bathymetry=200.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0,
                density=1.8, attenuation=0.5, roughness=bot))
        e.surface.roughness = surf
        return e

    def _mesh_sigma(self, tmp_path, env):
        from uacpy.io.oalib_writer import write_ssp_section
        out = tmp_path / 'ssp.txt'
        with open(out, 'w') as f:
            write_ssp_section(f, env, bottom_depth=200.0)
        return float(out.read_text().splitlines()[0].split()[1])

    def test_surface_carrier_drives_the_water_medium_sigma(self, tmp_path):
        assert self._mesh_sigma(tmp_path, self._env(surf=2.0)) == pytest.approx(2.0)

    def test_bottom_carrier_does_not_touch_the_water_medium_sigma(self, tmp_path):
        assert self._mesh_sigma(tmp_path, self._env(bot=2.0)) == pytest.approx(0.0)

    @pytest.mark.parametrize('model_name', ['Kraken', 'Scooter', 'SPARC'])
    def test_models_take_no_roughness_kwarg(self, model_name):
        import uacpy
        with pytest.raises(TypeError):
            getattr(uacpy, model_name)(roughness=2.0)


class TestSSPLinePinsWaterProperties:
    """AT's ``/`` list-directed terminator leaves unassigned items at their
    previous value, and ``TopBot`` (ReadEnvironmentMod.f90:285) reads the top
    half-space into the very module variables (misc/sspMod.f90:14) that a short
    ``z c /`` SSP line relies on. Every SSP line must therefore pin all six
    columns explicitly, as bellhop_writer.py already does."""

    @staticmethod
    def _env(surface=None):
        from uacpy.core import Environment, BoundaryProperties
        return Environment(
            bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.0, 1500.0)]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.5),
            surface=surface)

    @staticmethod
    def _ssp_rows(path):
        """The water-column SSP rows: lines between the mesh line and the
        bottom-option line, which is quoted."""
        rows = []
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            if parts and parts[-1] == '/' and not line.lstrip().startswith("'"):
                rows.append(parts[:-1])
        return rows

    def _write(self, tmp_path, env, name):
        from uacpy.core import Source, Receiver
        from uacpy.io.oalib_writer import write_kraken_env_file
        from uacpy.core.constants import BoundaryType
        out = tmp_path / f'{name}.env'
        write_kraken_env_file(
            out, env,
            source=Source(depths=25.0, frequencies=100.0),
            receiver=Receiver(depths=[50.0], ranges=[1000.0]),
            ssp_topopt='C',
            surface_type=(BoundaryType.HALF_SPACE if env.surface is not None
                          else BoundaryType.VACUUM),
            bottom_type=BoundaryType.HALF_SPACE,
            frequencies=None, n_mesh=0, rmax_m=5000.0,
            c_low=1400.0, c_high=1e9)
        return out

    @pytest.mark.parametrize('with_surface', [False, True])
    def test_every_ssp_row_carries_all_six_columns(self, tmp_path, with_surface):
        from uacpy.core import BoundaryProperties
        surface = (BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=0.9,
                                      attenuation=1.0)
                   if with_surface else None)
        path = self._write(tmp_path, self._env(surface), 'six')
        rows = self._ssp_rows(path)
        assert rows, "no SSP rows found in the written .env"
        water = [r for r in rows if float(r[0]) <= 100.0 and len(r) >= 2
                 and abs(float(r[1]) - 1500.0) < 1e-6]
        assert water, f"no water-column rows identified in {rows}"
        for row in water:
            assert len(row) == 6, (
                f"SSP row {row} has {len(row)} columns; a short form lets the "
                f"top half-space's cs/rho/alphaI/betaI leak into the water")
            assert float(row[2]) == 0.0, f"water shear speed must be 0: {row}"
            assert float(row[3]) == 1.0, f"water density must be 1.0: {row}"
            assert float(row[5]) == 0.0, f"water shear atten must be 0: {row}"


@pytest.mark.requires_binary
def test_surface_halfspace_does_not_leak_into_the_water_column():
    """End-to-end: a fluid surface half-space must not donate its density and
    attenuation to the water. Leaked alpha=1 dB/lambda costs ~333 dB at 5 km
    (333 wavelengths at 100 Hz)."""
    from uacpy.core import Environment, BoundaryProperties, Source, Receiver
    from uacpy.models import Kraken

    def tl(surface):
        env = Environment(
            bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.0, 1500.0)]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.5),
            surface=surface)
        return np.asarray(Kraken(timeout=300).run(
            env, Source(depths=25.0, frequencies=100.0),
            Receiver(depths=[50.0], ranges=np.linspace(500.0, 5000.0, 10))).tl)

    leaky = tl(BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1600.0, density=0.9,
                                  attenuation=1.0))
    # 120 dB is a gate, not a tolerance: cylindrical spreading alone reaches
    # only 10*log10(5000) = 37 dB at the far receiver, while the leak adds
    # ~333 dB, so no ordinary modal structure can straddle it.
    assert np.nanmax(leaky) < 120.0, (
        f"TL reaches {np.nanmax(leaky):.1f} dB over 5 km in a 100 m isovelocity "
        f"guide — the surface half-space's attenuation leaked into the water")


class TestPhaseSpeedBoundsForParameterFreeBottoms:
    """A vacuum/rigid boundary has no sound speed, but BoundaryProperties still
    carries the constructor default. Capping c_high on that placeholder
    truncates the mode spectrum (3 of 7 modes, 10.6 dB, on a 100 m rigid guide
    at 50 Hz)."""

    @staticmethod
    def _env(bottom):
        import uacpy
        return uacpy.Environment(name='b', bathymetry=100.0, ssp=1500.0,
                                 bottom=bottom)

    @pytest.mark.parametrize('kind', ['rigid', 'vacuum'])
    def test_parameter_free_bottom_is_unbounded_above(self, kind):
        from uacpy.core import BoundaryProperties
        from uacpy.core.constants import DEFAULT_C_MAX_UNBOUNDED
        from uacpy.io.oalib_writer import resolve_phase_speed_bounds
        _, c_high = resolve_phase_speed_bounds(
            self._env(BoundaryProperties(acoustic_type=kind)))
        assert c_high == DEFAULT_C_MAX_UNBOUNDED

    def test_penetrable_bottom_still_caps_on_the_halfspace_speed(self):
        from uacpy.core import BoundaryProperties
        from uacpy.core.constants import C_HIGH_FACTOR
        from uacpy.io.oalib_writer import resolve_phase_speed_bounds
        env = self._env(BoundaryProperties(
            acoustic_type='half-space', sound_speed=1800.0,
            density=1.8, attenuation=0.5))
        _, c_high = resolve_phase_speed_bounds(env)
        assert c_high == pytest.approx(1800.0 * C_HIGH_FACTOR)

    def test_explicit_c_high_always_wins(self):
        from uacpy.core import BoundaryProperties
        from uacpy.io.oalib_writer import resolve_phase_speed_bounds
        _, c_high = resolve_phase_speed_bounds(
            self._env(BoundaryProperties(acoustic_type='rigid')), c_high=1700.0)
        assert c_high == 1700.0


class TestUserWorkDirIsNeverDestroyed:
    """``cleanup`` removes uacpy's scratch. A work_dir the *caller* supplied is
    not uacpy's to delete: neither the directory itself nor anything that was
    already in it."""

    @staticmethod
    def _seed(tmp_path):
        d = tmp_path / 'mine'
        d.mkdir()
        (d / 'PRECIOUS.txt').write_text('do not delete')
        return d

    def test_pinned_work_dir_survives_cleanup_true(self, tmp_path):
        import uacpy
        d = self._seed(tmp_path)
        m = uacpy.Bellhop(work_dir=str(d), cleanup=True, verbose=False)
        fm = m._setup_file_manager()
        fm.get_path('scratch.env').write_text('x')
        fm.cleanup_work_dir()
        assert d.exists(), "uacpy deleted the caller's directory"
        assert (d / 'PRECIOUS.txt').exists(), "uacpy deleted a pre-existing file"
        assert not (d / 'scratch.env').exists(), "uacpy left its own scratch behind"

    def test_copy_onto_a_pinned_work_dir_survives(self, tmp_path):
        """Model(cleanup=True).copy(work_dir=d) — the _cleanup_explicit path,
        where the caller's True rides onto a directory they only named later."""
        import uacpy
        d = self._seed(tmp_path)
        m = uacpy.Bellhop(cleanup=True, verbose=False).copy(work_dir=str(d))
        fm = m._setup_file_manager()
        fm.get_path('scratch.env').write_text('x')
        fm.cleanup_work_dir()
        assert d.exists() and (d / 'PRECIOUS.txt').exists()

    def test_uacpy_owned_temp_dir_is_still_fully_removed(self):
        from uacpy.io.file_manager import FileManager
        fm = FileManager(use_tmpfs=False, base_dir=None, cleanup=True)
        wd = fm.create_work_dir()
        fm.get_path('scratch.env').write_text('x')
        assert wd.exists()
        fm.cleanup_work_dir()
        assert not wd.exists(), "a uacpy-created temp dir must be removed whole"


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
        from uacpy.io import write_reflection_coefficient
        theta = np.linspace(0.0, 90.0, 91)
        coeffs = np.column_stack([0.5 * np.ones(91), np.zeros(91)])
        write_reflection_coefficient(path, theta, coeffs)
        return path

    @pytest.mark.parametrize('kwargs', [
        {'interp_bathymetry': 'bogus'},
        {'interp_altimetry': 'bogus'},
    ])
    def test_bad_geometry_interp_raises_configurationerror(self, tmp_path, kwargs):
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        env = uacpy.Environment(name='t', bathymetry=100, ssp=1500)
        source, receiver = self._sr()
        with pytest.raises(ConfigurationError):
            write_bellhop_env_file(tmp_path / 'run.env', env, source, receiver,
                                   **kwargs)

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


class TestOASNWavenumberSampling:
    """``nw_samples`` reaches every OASN integration block. The replica and
    discrete-source wavenumber lines (``NWSIN``/``NWDIN``, unoasn22.f:227 and
    oasnun22.f:421) are what OASN samples the field on."""

    @staticmethod
    def _env_src_rcv():
        env = uacpy.Environment(name='oasn', bathymetry=100, ssp=1500)
        source = uacpy.Source(frequencies=100, depths=50)
        receiver = uacpy.Receiver(depths=np.array([30.0, 50.0, 70.0]),
                                  ranges=np.array([0.0]))
        return env, source, receiver

    @staticmethod
    def _write(tmp_path, **kwargs):
        import warnings
        from uacpy.io.oases_writer import write_oasn_input
        env, source, receiver = TestOASNWavenumberSampling._env_src_rcv()
        path = tmp_path / 'oasn.dat'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            write_oasn_input(path, env, source, receiver, **kwargs)
        return path.read_text().splitlines()

    def test_replica_block_honours_nw_samples(self, tmp_path):
        lines = self._write(tmp_path, options='N R J', nw_samples=512)
        assert lines[-1].split() == ['512', '1', '512']

    def test_replica_block_defaults_to_automatic(self, tmp_path):
        lines = self._write(tmp_path, options='N R J', nw_samples=-1)
        assert lines[-1].split()[0] == '-1'

    def test_discrete_source_block_honours_nw_samples(self, tmp_path):
        lines = self._write(
            tmp_path, options='N J', nw_samples=256,
            discrete_sources=[{'depth': 40.0, 'x': 1.0, 'y': 0.0,
                               'level': 180.0}])
        assert lines[-1].split() == ['256', '1', '256']


class TestBathyIOTypedErrors:
    """``bathy_io``'s failure typing: a malformed file is a parse error, a bad
    user-supplied interpolation code is a configuration error."""

    def test_read_boundary_3d_malformed_raises_fileformaterror(self, tmp_path):
        from uacpy.io.bathy_io import read_boundary_3d
        from uacpy.core.exceptions import FileFormatError
        bad = tmp_path / 'bad.bty'
        bad.write_text("'R'\n2\n0.0 1.0\n2\n0.0 1.0\nqqq qqq\nqqq qqq\n")
        with pytest.raises(FileFormatError):
            read_boundary_3d(str(bad))

    def test_read_boundary_3d_short_grid_raises_fileformaterror(self, tmp_path):
        from uacpy.io.bathy_io import read_boundary_3d
        from uacpy.core.exceptions import FileFormatError
        bad = tmp_path / 'short.bty'
        bad.write_text("'R'\n3\n0.0 1.0 2.0\n2\n0.0 1.0\n1.0 2.0\n")
        with pytest.raises(FileFormatError):
            read_boundary_3d(str(bad))

    @pytest.mark.parametrize('writer', ['bty', 'bty_long', 'ati', 'bty_3d'])
    def test_bad_interp_type_raises_configurationerror(self, tmp_path, writer):
        from uacpy.core.bottom import Bottom
        from uacpy.io.bathy_io import (
            write_bty_file, write_bty_long_format, write_ati_file, write_bty_3d)
        pairs = np.array([[0.0, 100.0], [1000.0, 120.0]])
        path = tmp_path / 'x.bty'
        with pytest.raises(ConfigurationError):
            if writer == 'bty':
                write_bty_file(path, pairs, interp_type='bogus')
            elif writer == 'ati':
                write_ati_file(path, pairs, interp_type='bogus')
            elif writer == 'bty_long':
                rd = Bottom.from_halfspaces(
                    np.array([0.0, 1000.0]),
                    sound_speed=np.array([1600.0, 1700.0]),
                    density=np.array([1.7, 1.9]),
                    attenuation=np.array([0.4, 0.6]))
                write_bty_long_format(path, pairs, rd, interp_type='bogus')
            else:
                write_bty_3d(path, np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                             np.zeros((2, 2)), interp_type='bogus')

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
        bty, bty_type = read_bathymetry(path)
        assert bty_type == 'L'
        assert bty.shape[0] == 7, "long format must return the geoacoustic rows"
        assert np.allclose(bty[2, 1:-1], [1600.0, 1650.0, 1700.0])
        assert np.allclose(bty[4, 1:-1], [1.7, 1.8, 1.9])
        assert np.allclose(bty[5, 1:-1], [0.4, 0.5, 0.6])
        # ±infinity extension holds every row constant.
        assert bty[0, 0] == -1e50 and bty[0, -1] == 1e50
        assert bty[2, 0] == bty[2, 1] and bty[2, -1] == bty[2, -2]


class TestReadVectorFortranSemantics:
    """``read_vector`` mirrors AT's ``ReadVector`` + ``SubTab``
    (SourceReceiverPositions.f90:221, subtabulate.f90)."""

    @staticmethod
    def _read(text):
        import io as _io
        from uacpy.io._fortran_helpers import read_vector
        return read_vector(_io.StringIO(text))

    def test_explicit_vector_wrapped_across_records(self):
        """A list-directed READ continues across records until Nx values are
        consumed; Fortran-written files wrap at the runtime column width."""
        x, nx = self._read("5\n10.0 20.0 30.0\n40.0 50.0\n")
        assert nx == 5
        assert np.allclose(x, [10.0, 20.0, 30.0, 40.0, 50.0])

    def test_replicate_idiom_does_not_warn(self):
        """``write_fieldflp`` emits ``N`` / ``0.0 /`` for the Rro block."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            x, nx = self._read("9\n0.0 /\n")
        assert nx == 9 and np.allclose(x, np.zeros(9))

    def test_two_value_shorthand_is_equally_spaced(self):
        x, _ = self._read("5\n0 1000 /\n")
        assert np.allclose(x, [0.0, 250.0, 500.0, 750.0, 1000.0])

    def test_truncated_record_raises_fileformaterror(self):
        from uacpy.core.exceptions import FileFormatError
        with pytest.raises(FileFormatError):
            self._read("5\n10 20\n")

    def test_ungeneratable_slash_record_raises_fileformaterror(self):
        """3 values before the '/' — SubTab does not generate for this, and AT
        leaves x(4:Nx) uninitialised."""
        from uacpy.core.exceptions import FileFormatError
        with pytest.raises(FileFormatError):
            self._read("5\n10 20 30 /\n")


class TestMultiProfileEnvKeepsLayerThickness:
    """``write_multi_profile_env`` holds every profile to one total media depth.
    That stretch must land on a transparent halfspace-property pad, never on a
    real ``SedimentLayer`` whose thickness is physical."""

    @staticmethod
    def _segments():
        from uacpy.core.bottom import (
            SeabedColumn, SedimentLayer, BoundaryProperties)
        hs = BoundaryProperties(acoustic_type='half-space', sound_speed=1800,
                                density=2.0, attenuation=0.5)
        layers = [
            SedimentLayer(thickness=5.0, sound_speed=1600, density=1.6,
                          attenuation=0.2),
            SedimentLayer(thickness=6.0, sound_speed=1650, density=1.8,
                          attenuation=0.3),
        ]
        return [
            (0.0, uacpy.Environment(name='deep', bathymetry=200.0, ssp=1500,
                                    bottom=SeabedColumn(layers=[], halfspace=hs))),
            (5000.0, uacpy.Environment(
                name='mid', bathymetry=180.0, ssp=1500,
                bottom=SeabedColumn(layers=layers[:1], halfspace=hs))),
            (10000.0, uacpy.Environment(
                name='shallow', bathymetry=150.0, ssp=1500,
                bottom=SeabedColumn(layers=layers, halfspace=hs))),
        ]

    def test_shallow_segment_layers_keep_their_thickness(self, tmp_path):
        from uacpy.io.oalib_writer import write_multi_profile_env
        env_file = tmp_path / 'multi.env'
        source = uacpy.Source(frequencies=100, depths=50)
        receiver = uacpy.Receiver(depths=np.linspace(10, 140, 14),
                                  ranges=np.array([1000.0]))
        write_multi_profile_env(env_file, self._segments(), source, receiver,
                                n_mesh=500, rmax_m=20000.0)
        text = env_file.read_text()
        shallow = text[text.index("'shallow'"):]
        # Media interfaces: 150 (seafloor) → 155 (5 m layer) → 161 (6 m layer).
        assert '  150.000000 1600.000000' in shallow
        assert '  155.000000 1600.000000' in shallow
        assert '  155.000000 1650.000000' in shallow
        assert '  161.000000 1650.000000' in shallow, \
            "the 6 m sediment layer was stretched to the global max depth"
        # The pad that absorbs the stretch carries halfspace properties.
        assert '  161.000000 1800.000000' in shallow

    def test_every_profile_declares_the_same_nmedia_and_total_depth(self, tmp_path):
        from uacpy.io.oalib_writer import write_multi_profile_env
        env_file = tmp_path / 'multi.env'
        source = uacpy.Source(frequencies=100, depths=50)
        receiver = uacpy.Receiver(depths=np.linspace(10, 140, 14),
                                  ranges=np.array([1000.0]))
        write_multi_profile_env(env_file, self._segments(), source, receiver,
                                n_mesh=500, rmax_m=20000.0)
        lines = env_file.read_text().splitlines()
        n_media = [int(lines[i + 2]) for i, line in enumerate(lines)
                   if line.startswith("'") and line.endswith("'")
                   and line[1:-1] in ('deep', 'mid', 'shallow')]
        assert len(n_media) == 3 and len(set(n_media)) == 1
        # Selected structurally: the half-space row is the line after each
        # profile's bottom-option line. Keying on a depth or sound-speed format
        # would only work while different rows happened to print differently.
        halfspace_depths = [lines[i + 1].split()[0]
                            for i, line in enumerate(lines)
                            if line.startswith("'A'")]
        assert len(halfspace_depths) == 3


class TestSSPRangeAxisPrecision:
    """``write_ssp``'s range axis feeds Bellhop's Quad segment search, which
    needs ``SSP%Seg%r`` strictly increasing (Bellhop/sspMod.f90)."""

    def test_sub_metre_ranges_survive_the_round_trip(self, tmp_path):
        from uacpy.io.oalib_writer import write_ssp
        from uacpy.io.oalib_reader import read_ssp_2d
        ranges = np.array([0.0, 0.4, 0.8, 1.2])
        c = np.tile(np.array([[1500.0], [1490.0]]), (1, 4))
        path = tmp_path / 'fine.ssp'
        write_ssp(path, ranges, c)
        r_back = np.asarray(read_ssp_2d(str(path))['r_prof'])
        assert len(np.unique(r_back)) == 4, "range axis collapsed to duplicates"
        assert np.allclose(r_back, ranges)

    def test_single_profile_ssp_is_rejected(self, tmp_path):
        from uacpy.io.oalib_writer import write_ssp
        with pytest.raises(ConfigurationError, match='at least 2'):
            write_ssp(tmp_path / 'one.ssp', np.array([0.0]),
                      np.array([[1500.0], [1490.0]]))


class TestFlpOptionValidation:
    """``field.exe`` ERROUTs on an option character outside its alphabet
    (field.f90:70-99); catch it before writing a deck that only fails inside
    the Fortran run."""

    @staticmethod
    def _pos():
        return {'s': {'z': np.array([50.0])},
                'r': {'z': np.array([10.0, 20.0, 30.0]),
                      'r': np.linspace(100.0, 1000.0, 5)}}

    @pytest.mark.parametrize('option', ['ZC C', 'RCXC', 'RC X'])
    def test_bad_option_raises_configurationerror(self, tmp_path, option):
        from uacpy.io.oalib_writer import write_fieldflp
        with pytest.raises(ConfigurationError, match='option position'):
            write_fieldflp(tmp_path / 'bad.flp', option, self._pos())

    @pytest.mark.parametrize('option', ['RC C', 'XA*I', 'SC C'])
    def test_valid_options_are_accepted(self, tmp_path, option):
        from uacpy.io.oalib_writer import write_fieldflp
        write_fieldflp(tmp_path / 'good.flp', option, self._pos())
        assert (tmp_path / 'good.flp').exists()

    def test_bad_3d_evaluator_raises_configurationerror(self, tmp_path):
        from uacpy.io.oalib_writer import write_field3dflp
        pos = {'s': {'x': np.array([0.0]), 'y': np.array([0.0]),
                     'z': np.array([50.0])},
               'r': {'z': np.linspace(0, 100, 11),
                     'r': np.linspace(0, 5000, 11),
                     'theta': np.linspace(0, 350, 36)},
               'Nsx': 1, 'Nsy': 1}
        bathy = {'X': np.linspace(0, 10000, 4), 'Y': np.linspace(0, 10000, 3),
                 'depth': 100 * np.ones((3, 4))}
        with pytest.raises(ConfigurationError, match='evaluator'):
            write_field3dflp(tmp_path / 'bad3d.flp', 'ZZZFM', pos, bathy)


class TestShdReaderKeysAgree:
    """``read_shd_bin`` and ``read_shd_asc`` describe the same payload, so a
    caller switching between them must not have to remap keys.

    Both readers are exercised on the SAME field written twice: once in the
    ASCII stream of ``Matlab/ReadWrite/read_shd_asc.m:13-38`` and once in the
    direct-access record layout of ``misc/RWSHDFile.f90:100-114``, whose
    ``LRecl = MAX(41, 2*Nfreq, …)`` is counted in 4-byte words.
    """

    #: One frequency, one bearing, one source, two receiver depths, two ranges.
    TITLE = 'title'
    PLOT_TYPE = 'rectilin'
    FREQ0, ATTEN = 100.0, 0.0
    FREQ_VEC = [100.0]
    THETA = [0.0]
    S_Z = [50.0]
    R_Z = [10.0, 20.0]
    R_R = [1000.0, 2000.0]
    #: (real, imag) per (receiver depth, range). No exact zeros — both readers
    #: map those to NaN as the "engine never wrote this cell" convention.
    ROWS = [[1.0, 0.5, 2.0, 0.25], [3.0, -1.0, 0.75, 2.0]]

    @classmethod
    def _write_asc(cls, path):
        tokens = [
            len(cls.FREQ_VEC), len(cls.THETA), len(cls.S_Z), len(cls.R_Z),
            len(cls.R_R), cls.FREQ0, cls.ATTEN,
            *cls.FREQ_VEC, *cls.THETA, *cls.S_Z, *cls.R_Z, *cls.R_R,
            *[v for row in cls.ROWS for v in row],
        ]
        path.write_text(f"{cls.TITLE}\n{cls.PLOT_TYPE}\n"
                        + ' '.join(str(t) for t in tokens) + "\n")
        return path

    @classmethod
    def _write_bin(cls, path):
        """Emit the same field as a little-endian direct-access ``.shd``."""
        n_rz, n_rr = len(cls.R_Z), len(cls.R_R)
        recl = max(41, 2 * len(cls.FREQ_VEC), 2 * len(cls.THETA), 2, 2,
                   len(cls.S_Z), n_rz, 2 * n_rr)          # RWSHDFile.f90:100
        rec_bytes = 4 * recl
        records = [
            np.array([recl], '<i4').tobytes() + cls.TITLE.encode().ljust(80),
            cls.PLOT_TYPE.encode().ljust(10),
            (np.array([len(cls.FREQ_VEC), len(cls.THETA), 1, 1, len(cls.S_Z),
                       n_rz, n_rr], '<i4').tobytes()
             + np.array([cls.FREQ0, cls.ATTEN], '<f8').tobytes()),
            np.array(cls.FREQ_VEC, '<f8').tobytes(),
            np.array(cls.THETA, '<f8').tobytes(),
            np.array([0.0], '<f8').tobytes(),                       # Sx
            np.array([0.0], '<f8').tobytes(),                       # Sy
            np.array(cls.S_Z, '<f4').tobytes(),      # Sz/Rz are REAL(KIND=4)
            np.array(cls.R_Z, '<f4').tobytes(),
            np.array(cls.R_R, '<f8').tobytes(),
        ] + [np.array(row, '<f4').tobytes() for row in cls.ROWS]
        path.write_bytes(b''.join(r.ljust(rec_bytes, b'\x00') for r in records))
        return path

    def test_both_readers_return_the_same_keys(self, tmp_path):
        from uacpy.io.oalib_reader import read_shd_asc, read_shd_bin
        asc = read_shd_asc(self._write_asc(tmp_path / 'tiny.shd.asc'))
        bin_ = read_shd_bin(str(self._write_bin(tmp_path / 'tiny.shd')))
        # The binary reader adds the slice label; nothing else may differ.
        assert set(bin_) - set(asc) == {'pressure_freq'}
        assert set(asc) - set(bin_) == set()
        assert set(asc['Pos']) == set(bin_['Pos'])
        assert set(asc['Pos']['r']) == set(bin_['Pos']['r']) == {'z', 'r'}
        # The ASCII stream carries no source x/y, so its 's' is a subset.
        assert set(asc['Pos']['s']) <= set(bin_['Pos']['s'])

    def test_both_readers_return_the_same_values(self, tmp_path):
        from uacpy.io.oalib_reader import read_shd_asc, read_shd_bin
        asc = read_shd_asc(self._write_asc(tmp_path / 'tiny.shd.asc'))
        bin_ = read_shd_bin(str(self._write_bin(tmp_path / 'tiny.shd')))
        assert asc['title'] == bin_['title'] == self.TITLE
        assert asc['PlotType'] == bin_['PlotType'].strip() == self.PLOT_TYPE
        assert asc['freq0'] == bin_['freq0'] == self.FREQ0
        assert asc['atten'] == bin_['atten'] == self.ATTEN
        np.testing.assert_allclose(asc['freqVec'], bin_['freqVec'])
        np.testing.assert_allclose(asc['Pos']['theta'], bin_['Pos']['theta'])
        np.testing.assert_allclose(asc['Pos']['s']['z'], bin_['Pos']['s']['z'])
        np.testing.assert_allclose(asc['Pos']['r']['z'], bin_['Pos']['r']['z'])
        np.testing.assert_allclose(asc['Pos']['r']['r'], bin_['Pos']['r']['r'])
        # read_shd_asc is 2-D (one bearing, one source depth); read_shd_bin
        # keeps those axes, so index them away before comparing.
        np.testing.assert_allclose(asc['pressure'], bin_['pressure'][0, 0],
                                   rtol=1e-6)
        assert bin_['pressure_freq'] == self.FREQ_VEC[0]


class TestOASESWriterKnobsReachTheDeck:
    """Every OASES writer takes ``**kwargs``, so a knob no block reads would be
    dropped without a trace and the run would quietly use the default."""

    @staticmethod
    def _args():
        env = uacpy.Environment(name='oases', bathymetry=100, ssp=1500)
        source = uacpy.Source(frequencies=100, depths=50)
        receiver = uacpy.Receiver(depths=np.array([30.0, 50.0, 70.0]),
                                  ranges=np.linspace(100.0, 5000.0, 10))
        return env, source, receiver

    @pytest.mark.parametrize('writer_name,options,bad', [
        ('write_oast_input', None, 'replica_nz'),
        ('write_oasn_input', 'N J', 'vrec'),
        ('write_oasn_input', 'N J', 'plot_rmax'),
        ('write_oasp_input', None, 'nw_sample'),
        ('write_oasr_input', None, 'nw_samples'),
    ])
    def test_unread_parameter_raises(self, tmp_path, writer_name, options, bad):
        import warnings
        from uacpy.io import oases_writer
        writer = getattr(oases_writer, writer_name)
        env, source, receiver = self._args()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with pytest.raises(ConfigurationError, match='not read by this deck'):
                writer(tmp_path / 'x.dat', env, source, receiver,
                       options=options, **{bad: 1.0})

    def test_documented_knobs_are_accepted(self, tmp_path):
        import warnings
        from uacpy.io.oases_writer import write_oasn_input
        env, source, receiver = self._args()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            write_oasn_input(tmp_path / 'ok.dat', env, source, receiver,
                             options='N R J', nw_samples=256, c_low=1400.0,
                             c_high=1.0e8, replica_nz=8, integration_offset=0.0)
        assert (tmp_path / 'ok.dat').exists()


class TestSourceBeamPatternRoundTrip:
    """``.sbp`` levels are dB on disk (``beampattern.f90:59`` converts to a
    linear amplitude only after reading), and dB is what
    :attr:`uacpy.Source.beam_pattern` carries."""

    def test_write_then_read_returns_the_written_db(self, tmp_path):
        from uacpy.io.refl_io import (
            read_source_beam_pattern, write_source_beam_pattern)
        angles = np.array([-90.0, -30.0, 0.0, 30.0, 90.0])
        levels = np.array([-20.0, -6.0, 0.0, -6.0, -20.0])
        path = tmp_path / 'beam.sbp'
        write_source_beam_pattern(path, angles, levels)
        back = read_source_beam_pattern(path, sbp_option='*')
        assert np.allclose(back[:, 0], angles)
        assert np.allclose(back[:, 1], levels)

    def test_omni_default_is_zero_db(self):
        from uacpy.io.refl_io import read_source_beam_pattern
        back = read_source_beam_pattern('unused', sbp_option='O')
        assert np.allclose(back[:, 1], 0.0)

    def test_root_name_without_extension_resolves(self, tmp_path):
        from uacpy.io.refl_io import (
            read_source_beam_pattern, write_source_beam_pattern)
        write_source_beam_pattern(tmp_path / 'beam.sbp', np.array([0.0]),
                                  np.array([-3.0]))
        back = read_source_beam_pattern(tmp_path / 'beam', sbp_option='*')
        assert np.allclose(back[:, 1], [-3.0])


class TestBinaryArrivalsRejected:
    """``read_arr_file`` only parses the ASCII ``.arr``; the binary layout
    (RunType 'a') is a configuration error, not a parse error."""

    def test_binary_arrivals_raises_configurationerror(self, tmp_path):
        from uacpy.io.oalib_reader import read_arr_file
        p = tmp_path / 'run.arr'
        p.write_bytes(b'\x04\x00\x00\x00' + b'\x00' * 32)
        with pytest.raises(ConfigurationError, match="RunType 'A'"):
            read_arr_file(p)


class TestOASNNoiseWavenumberCounts:
    """The surface- and deep-noise blocks have no automatic-sampling branch:
    ``NOIPAR`` reads three explicit counts and sums them into ``NWVNON`` /
    ``NWVNOP`` (oasnun22.f:312, :358). A negative count makes that total
    negative and the block integrates nothing — the covariance silently
    collapses to the white-noise identity."""

    @staticmethod
    def _last_line(tmp_path, **kwargs):
        import warnings
        from uacpy.io.oases_writer import write_oasn_input
        env = uacpy.Environment(name='oasn', bathymetry=100, ssp=1500)
        source = uacpy.Source(frequencies=100, depths=50)
        receiver = uacpy.Receiver(depths=np.array([30.0, 50.0, 70.0]),
                                  ranges=np.array([0.0]))
        path = tmp_path / 'oasn.dat'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            write_oasn_input(path, env, source, receiver, options='N J',
                             **kwargs)
        return path.read_text().splitlines()[-1]

    @pytest.mark.parametrize('block', [
        {'surface_noise_level': 70.0},
        {'deep_noise_level': 70.0},
    ])
    @pytest.mark.parametrize('nw', [-1, 0, None])
    def test_automatic_falls_back_to_positive_counts(self, tmp_path, block, nw):
        counts = [int(v) for v in
                  self._last_line(tmp_path, nw_samples=nw, **block).split()]
        assert len(counts) == 3
        assert sum(counts) > 0, "a non-positive total integrates nothing"

    @pytest.mark.parametrize('block', [
        {'surface_noise_level': 70.0},
        {'deep_noise_level': 70.0},
    ])
    def test_pinned_count_reaches_the_block(self, tmp_path, block):
        line = self._last_line(tmp_path, nw_samples=800, **block)
        assert line.split()[:2] == ['800', '800']


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
        ati, ati_type = read_altimetry(p)
        assert ati_type == 'L'
        assert ati.shape == (7, 5)
        assert np.allclose(ati[0, 1:-1], [0.0, 2500.0, 5000.0]), "km -> m"
        assert np.allclose(ati[2, 1:-1], [3500.0, 3400.0, 3300.0])
        assert np.allclose(ati[3, 1:-1], [1800.0, 1750.0, 1700.0])
        assert np.allclose(ati[4, 1:-1], [0.9, 0.91, 0.92])
        assert ati[0, 0] == -1e50 and ati[0, -1] == 1e50

    def test_short_format_still_two_rows(self, tmp_path):
        from uacpy.io.bathy_io import read_altimetry, write_ati_file
        pairs = np.array([[0.0, 0.0], [2500.0, 2.0], [5000.0, 0.0]])
        p = tmp_path / 'flat.ati'
        write_ati_file(p, pairs, interp_type='C')
        ati, ati_type = read_altimetry(p)
        assert ati_type == 'C'
        assert ati.shape == (2, 5)
        assert np.allclose(ati[0, 1:-1], pairs[:, 0])
        assert np.allclose(ati[1, 1:-1], pairs[:, 1])

    def test_short_row_in_a_long_file_raises_fileformaterror(self, tmp_path):
        from uacpy.io.bathy_io import read_altimetry
        from uacpy.core.exceptions import FileFormatError
        p = tmp_path / 'bad.ati'
        p.write_text("'LL'\n1\n0.0 0.0 3500.0\n")
        with pytest.raises(FileFormatError, match='columns'):
            read_altimetry(p)

    def test_unknown_interp_type_raises_fileformaterror(self, tmp_path):
        from uacpy.io.bathy_io import read_altimetry
        from uacpy.core.exceptions import FileFormatError
        p = tmp_path / 'bad.ati'
        p.write_text("'ZS'\n1\n0.0 0.0\n")
        with pytest.raises(FileFormatError, match='altimetry type'):
            read_altimetry(p)


class TestSedimentLayerColumnOrder:
    """Pin the AT ``.env`` medium-line column order for sediment layers.

    ``misc/sspMod.f90:334`` reads each SSP line as
    ``z, alphaR, betaR, rhoR, alphaI, betaI`` — depth, compressional speed,
    shear speed, density, compressional attenuation, shear attenuation. The
    halfspace line is covered by the analytic Pekeris benchmark, but the
    layered block (``write_layer_sections``, NMEDIA > 1) has no benchmark, so a
    swap there changes every layered result while every test still passes.
    """

    # Chosen so no two fields share a value and none is a plausible default:
    # a swap of any pair moves a number that is checked below.
    CP, CS, RHO, ALPHA_P, ALPHA_S = 1623.0, 0.0, 1.77, 0.33, 0.0
    THICKNESS = 12.0

    def _write(self, tmp_path):
        from uacpy.core.environment import (
            Bottom, SeabedColumn, SedimentLayer, BoundaryProperties)
        from uacpy.io.oalib_writer import write_kraken_env_file
        from uacpy.core.constants import BoundaryType
        layer = SedimentLayer(thickness=self.THICKNESS, sound_speed=self.CP,
                              density=self.RHO, attenuation=self.ALPHA_P)
        # Distinct, non-zero shear values: swapping cs with alpha_s between two
        # zeros would be a no-op and the guard could never fail.
        hs = BoundaryProperties(acoustic_type='half-space', sound_speed=1800.0,
                                density=2.0, attenuation=0.5,
                                shear_speed=600.0, shear_attenuation=0.25)
        env = uacpy.Environment(
            bathymetry=100.0,
            ssp=SoundSpeedProfile(depths=[0.0, 100.0], data=[1500.0, 1500.0]),
            bottom=Bottom(columns=[SeabedColumn(layers=[layer], halfspace=hs)]),
        )
        out = tmp_path / 'layers.env'
        write_kraken_env_file(
            str(out), env,
            uacpy.Source(depths=25.0, frequencies=100.0),
            uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
            ssp_topopt='C', surface_type=BoundaryType.VACUUM,
            bottom_type=BoundaryType.HALF_SPACE, frequencies=[100.0],
            n_mesh=0, rmax_m=1000.0, c_low=0.0, c_high=2000.0,
        )
        return out.read_text().splitlines()

    def test_halfspace_line_fields_are_in_at_order(self, tmp_path):
        """Same column order on the ``'A'`` halfspace line. Swapping the
        elastic columns (cs <-> alpha_s) makes krakenc hang rather than fail,
        so a physical test cannot settle it — the deck is checked directly."""
        lines = self._write(tmp_path)
        idx = next(i for i, ln in enumerate(lines) if ln.strip().startswith("'A'"))
        z, cp, cs, rho, alpha_p, alpha_s = (
            float(v) for v in lines[idx + 1].split()[:6])
        assert cp == pytest.approx(1800.0)
        assert cs == pytest.approx(600.0), "column 3 must be shear speed (betaR)"
        assert rho == pytest.approx(2.0), "column 4 must be density (rhoR)"
        assert alpha_p == pytest.approx(0.5), (
            "column 5 must be compressional attenuation (alphaI)")
        assert alpha_s == pytest.approx(0.25), (
            "column 6 must be shear attenuation (betaI)")

    def test_layer_line_fields_are_in_at_order(self, tmp_path):
        lines = self._write(tmp_path)
        # Keyed on the sediment sound speed: the water SSP rows share the
        # layer rows' depths, and every depth is written in one format, so a
        # depth prefix cannot tell them apart.
        layer_rows = [ln.split() for ln in lines
                      if ln.strip().endswith('/')
                      and len(ln.split()) >= 6
                      and ln.split()[1].startswith('1623')]
        assert len(layer_rows) == 2, f"expected the layer's two rows, got {layer_rows}"
        for row in layer_rows:
            z, cp, cs, rho, alpha_p, alpha_s = (float(v) for v in row[:6])
            assert cp == pytest.approx(self.CP)
            assert cs == pytest.approx(self.CS)
            assert rho == pytest.approx(self.RHO), (
                "column 4 must be density (rhoR), not attenuation")
            assert alpha_p == pytest.approx(self.ALPHA_P), (
                "column 5 must be compressional attenuation (alphaI)")
            assert alpha_s == pytest.approx(self.ALPHA_S)
        assert [float(r[0]) for r in layer_rows] == [100.0, 112.0]


def test_multi_profile_deck_shares_one_bottom_without_slivers(tmp_path):
    """Every profile in a multi-profile ``.env`` ends at the same depth, that
    depth is the one :func:`plan_multi_profile_media` reports, and no profile
    carries a degenerate medium.

    The shared bottom is the load-bearing half.
    ``KrakenField/EvaluateCMMod.f90:312-317`` stops a coupled run unless each
    profile's mode-tabulation grid ends *exactly* on
    ``SSP%Depth( NMedia + 1 )``, so whatever builds that grid has to read the
    bottom off the same planner the deck was written from; an independently
    recomputed bottom lands off that depth and the coupled run dies.

    "Sliver" here means a medium below the deck's own depth resolution. Media
    interfaces are written at ``.1f``, so one quantum is the thinnest medium
    expressible and the comparison has to be made at that resolution: the
    deepest profile's reserve pad is exactly one quantum, and raw subtraction
    reads it as ``0.09999999999999432``. A one-quantum pad is deliberate and
    harmless — KRAKEN meshes it like any other medium (it appears in the
    ``.mod`` media list) and coupled runs project through it correctly.
    """
    import re
    from uacpy.io.oalib_writer import (
        plan_multi_profile_media, _PAD_MEDIUM_THICKNESS_M)
    from uacpy.core.environment import (
        Bathymetry, Bottom, SeabedColumn, SedimentLayer, BoundaryProperties)
    from uacpy.io.oalib_writer import write_multi_profile_env
    from uacpy.core.constants import BoundaryType
    hs = BoundaryProperties(acoustic_type='half-space', sound_speed=1800.0,
                            density=2.0, attenuation=0.5)
    bot = Bottom(columns=[SeabedColumn(
        layers=[SedimentLayer(thickness=3.0, sound_speed=1600.0,
                              density=1.7, attenuation=0.3)], halfspace=hs)])
    segments = []
    for i, wd in enumerate([100.0, 140.0, 180.0, 200.0]):
        segments.append((i * 2.0, uacpy.Environment(
            bathymetry=Bathymetry(ranges=[0.0, 6000.0], depths=[wd, wd]),
            ssp=SoundSpeedProfile(depths=[0.0, wd], data=[1500.0, 1480.0]),
            bottom=bot)))
    out = tmp_path / 'multi.env'
    write_multi_profile_env(
        str(out), segments,
        uacpy.Source(depths=30.0, frequencies=100.0),
        uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
        surface_type=BoundaryType.VACUUM,
        bottom_type=BoundaryType.HALF_SPACE,
        n_mesh=500, rmax_m=6000.0, c_low=0.0, c_high=2000.0,
    )
    mesh = [float(m.group(1)) for m in
            (re.match(r'^\s*\d+\s+[\d.]+\s+([\d.]+)\s*,?\s*$', ln)
             for ln in out.read_text().splitlines()) if m]
    n_media = len(mesh) // len(segments)
    blocks = [mesh[p_i * n_media:(p_i + 1) * n_media]
              for p_i in range(len(segments))]

    bottoms = {block[-1] for block in blocks}
    assert bottoms == {plan_multi_profile_media(segments)[1]}, (
        f"profiles must all end on the planner's shared bottom, got {bottoms}")

    for p_i, block in enumerate(blocks):
        thick = [round(b - a, 1) for a, b in zip([0.0] + block[:-1], block)]
        # The thinnest medium any profile can carry is a transparent pad.
        assert min(thick) >= _PAD_MEDIUM_THICKNESS_M, (
            f"profile {p_i} has a medium thinner than a transparent pad: "
            f"{thick}")


# ---------------------------------------------------------------------------
# Acoustics-Toolbox .env top block: record order and boundary tables.
#
# ``misc/ReadEnvironmentMod.f90`` reads TopOpt (:68 -> ReadTopOpt), then the
# volume-attenuation rows *inside* ReadTopOpt (F-G :215, biological :220-235),
# and only then the top half-space row (:75 -> TopBot :285). ``RefCoef.f90``
# opens ``<root>.trc`` for a top ``'F'`` (:64-76) and ``<root>.irc`` for a
# bottom ``'P'`` (:92-96).
# ---------------------------------------------------------------------------

_AT_BIN = Path(uacpy.__file__).parent / 'bin' / 'oalib'

_ICE = dict(acoustic_type='half-space', sound_speed=3500.0, shear_speed=1800.0,
            density=0.9, attenuation=1.0, shear_attenuation=2.0)


def _fg():
    from uacpy.core.absorption import FrancoisGarrison
    return FrancoisGarrison(temperature_c=10.0, salinity_psu=35.0, pH=8.0,
                            z_bar_m=50.0)


def _bio():
    from uacpy.core.absorption import Biological
    return Biological(layers=[(10.0, 20.0, 400.0, 5.0, 0.1)])


def _top_block_env(absorption=None, surface=None, bathymetry=100.0):
    from uacpy.core import BoundaryProperties
    if surface is None:
        surface = BoundaryProperties(**_ICE)
    return uacpy.Environment(
        name='deck', bathymetry=bathymetry,
        ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.0, 1500.0)]),
        surface=surface,
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1700.0, density=1.5,
                                  attenuation=0.5),
        absorption=absorption)


def _src_rcv():
    return (uacpy.Source(depths=25.0, frequencies=200.0),
            uacpy.Receiver(depths=[50.0], ranges=[1000.0]))


def _write_kraken(path, env, **overrides):
    from uacpy.core.constants import BoundaryType
    from uacpy.io.oalib_writer import write_kraken_env_file
    src, rcv = _src_rcv()
    kwargs = dict(ssp_topopt='C', surface_type=BoundaryType.HALF_SPACE,
                  bottom_type=BoundaryType.HALF_SPACE, frequencies=None,
                  n_mesh=0, rmax_m=5000.0, c_low=1400.0, c_high=2000.0)
    kwargs.update(overrides)
    write_kraken_env_file(path, env, src, rcv, **kwargs)
    return path


def _write_scooter(path, env, **overrides):
    from uacpy.core.constants import BoundaryType
    from uacpy.io.oalib_writer import write_scooter_env_file
    src, rcv = _src_rcv()
    kwargs = dict(ssp_topopt='C', surface_type=BoundaryType.HALF_SPACE,
                  bottom_type=BoundaryType.HALF_SPACE, frequencies=None,
                  topopt_extra='', n_mesh=0, rmax_m=5000.0, c_low=1400.0,
                  c_high=2000.0)
    kwargs.update(overrides)
    write_scooter_env_file(path, env, src, rcv, **kwargs)
    return path


def _run_at(exe, root, work_dir):
    """Run an AT binary on ``<root>.env`` and return its print file.

    ERROUT writes '*** FATAL ERROR ***' into the .prt and then STOPs with a
    zero exit code, so the print file — not the return code — is the verdict.
    """
    import subprocess
    proc = subprocess.run([str(_AT_BIN / exe), root], cwd=str(work_dir),
                          capture_output=True, text=True, timeout=300)
    prt = Path(work_dir) / f'{root}.prt'
    text = prt.read_text() if prt.exists() else ''
    assert '*** FATAL ERROR ***' not in text, (
        f"{exe} rejected the deck:\n{text[-1500:]}")
    assert 'Fortran runtime error' not in proc.stderr, (
        f"{exe} mis-parsed the deck:\n{proc.stderr[-800:]}\n{text[-800:]}")
    return text


def _floats(line):
    return [float(tok) for tok in line.split() if tok != '/']


class TestATEnvTopBlockRecordOrder:
    """The absorption rows belong between the TopOpt line and the top
    half-space row. Emitted the other way round, ReadTopOpt eats the
    half-space row as its F-G parameters and TopBot then reads the F-G row as
    the half-space — the run dies in AttenMod : CRCI."""

    @staticmethod
    def _assert_fg_then_halfspace(text, cp=3500.0):
        lines = text.splitlines()
        assert lines[3].startswith("'C"), f"line 4 is not TopOpt: {lines[3]}"
        assert lines[3][2] == 'A', f"TopOpt(2) is not 'A': {lines[3]}"
        assert lines[3][4] == 'F', f"TopOpt(4) is not 'F': {lines[3]}"
        assert _floats(lines[4]) == [10.0, 35.0, 8.0, 50.0], (
            f"the F-G row must follow TopOpt directly; got {lines[4]!r}")
        hs = _floats(lines[5])
        assert lines[5].rstrip().endswith('/') and len(hs) == 6, (
            f"the top half-space row must follow the F-G row; got {lines[5]!r}")
        assert hs[0] == 0.0 and hs[1] == cp, (
            f"top half-space row carries the wrong medium: {lines[5]!r}")

    def test_kraken_deck_orders_topopt_absorption_halfspace(self, tmp_path):
        text = _write_kraken(tmp_path / 'kr.env',
                             _top_block_env(_fg())).read_text()
        self._assert_fg_then_halfspace(text)

    def test_scooter_deck_orders_topopt_absorption_halfspace(self, tmp_path):
        text = _write_scooter(tmp_path / 'sc.env',
                              _top_block_env(_fg())).read_text()
        self._assert_fg_then_halfspace(text)

    def test_bounce_deck_orders_topopt_absorption_halfspace(self, tmp_path):
        from uacpy.core import BoundaryProperties
        from uacpy.core.constants import BoundaryType
        from uacpy.io.oalib_writer import write_bounce_input_file
        src, _ = _src_rcv()
        out = tmp_path / 'bo.env'
        write_bounce_input_file(
            out, _top_block_env(_fg(),
                                surface=BoundaryProperties(acoustic_type='vacuum')),
            src, ssp_topopt='C', bottom_type=BoundaryType.HALF_SPACE, n_mesh=0, c_low=1400.0,
            c_high=2000.0, rmax=5000.0)
        # BOUNCE's top half-space is the water at the seafloor, not the ice.
        self._assert_fg_then_halfspace(out.read_text(), cp=1500.0)

    def test_multi_profile_deck_orders_every_profile(self, tmp_path):
        from uacpy.io.oalib_writer import write_multi_profile_env
        src, rcv = _src_rcv()
        out = tmp_path / 'multi.env'
        segments = [(0.0, _top_block_env(_fg())),
                    (5000.0, _top_block_env(_fg()))]
        write_multi_profile_env(out, segments, src, rcv, n_mesh=500,
                                rmax_m=20000.0)
        lines = out.read_text().splitlines()
        starts = [i for i, ln in enumerate(lines) if ln.strip() == "'deck'"]
        assert len(starts) == 2
        for i in starts:
            assert _floats(lines[i + 4]) == [10.0, 35.0, 8.0, 50.0], (
                f"profile at line {i} lost the F-G row: {lines[i + 4]!r}")
            assert _floats(lines[i + 5])[1] == 3500.0, (
                f"profile at line {i} lost the top half-space row: "
                f"{lines[i + 5]!r}")

    def test_biological_block_precedes_top_halfspace(self, tmp_path):
        lines = _write_kraken(tmp_path / 'kr.env',
                              _top_block_env(_bio())).read_text().splitlines()
        assert lines[3][4] == 'B', f"TopOpt(4) is not 'B': {lines[3]}"
        assert lines[4].strip() == '1', f"bio layer count missing: {lines[4]!r}"
        assert _floats(lines[5]) == [10.0, 20.0, 400.0, 5.0, 0.1], (
            f"bio layer row missing: {lines[5]!r}")
        assert _floats(lines[6])[1] == 3500.0, (
            f"top half-space row must follow the bio block: {lines[6]!r}")

    @pytest.mark.requires_binary
    @pytest.mark.parametrize('absorption_name', ['fg', 'bio'])
    def test_krakenc_reads_the_deck(self, tmp_path, absorption_name):
        absorption = _fg() if absorption_name == 'fg' else _bio()
        _write_kraken(tmp_path / 'kr.env', _top_block_env(absorption))
        text = _run_at('krakenc.exe', 'kr', tmp_path)
        assert 'ACOUSTO-ELASTIC half-space' in text
        assert '3500.00' in text, (
            "the ice half-space never reached TopBot:\n" + text[:1500])

    @pytest.mark.requires_binary
    @pytest.mark.parametrize('absorption_name', ['fg', 'bio'])
    def test_scooter_reads_the_deck(self, tmp_path, absorption_name):
        absorption = _fg() if absorption_name == 'fg' else _bio()
        _write_scooter(tmp_path / 'sc.env', _top_block_env(absorption))
        text = _run_at('scooter.exe', 'sc', tmp_path)
        assert '3500.00' in text


class TestSparcBoundaryRestriction:
    """``sparc.f90:100-104`` ERROUTs unless both boundaries are vacuum or
    rigid, and SPARC writes no half-space row — declaring 'A' would hand the
    SSP mesh line to TopBot."""

    @staticmethod
    def _write(path, env, surface_type, bottom_type):
        from uacpy.io.oalib_writer import write_sparc_env_file
        src, rcv = _src_rcv()
        write_sparc_env_file(
            path, env, src, rcv, ssp_code='C', surface_type=surface_type,
            bottom_type=bottom_type, output_mode='R', n_mesh=0, rmax_m=5000.0,
            c_low=1400.0, c_high=2000.0, pulse_type='P', f_min=50.0,
            f_max=400.0, n_t_out=20, t_max=1.0, t_start=0.0, t_mult=1.0)

    def test_halfspace_surface_is_refused(self, tmp_path):
        from uacpy.core.constants import BoundaryType
        from uacpy.core.exceptions import UnsupportedFeatureError
        out = tmp_path / 'sp.env'
        with pytest.raises(UnsupportedFeatureError, match='half-space surface'):
            self._write(out, _top_block_env(), BoundaryType.HALF_SPACE,
                        BoundaryType.VACUUM)
        assert not out.exists(), "a rejected deck must not be left behind"

    def test_halfspace_bottom_is_refused(self, tmp_path):
        from uacpy.core import BoundaryProperties
        from uacpy.core.constants import BoundaryType
        from uacpy.core.exceptions import UnsupportedFeatureError
        env = _top_block_env(surface=BoundaryProperties(acoustic_type='vacuum'))
        with pytest.raises(UnsupportedFeatureError, match='half-space bottom'):
            self._write(tmp_path / 'sp.env', env, BoundaryType.VACUUM,
                        BoundaryType.HALF_SPACE)

    def test_vacuum_deck_writes_no_halfspace_row(self, tmp_path):
        from uacpy.core import BoundaryProperties
        from uacpy.core.constants import BoundaryType
        out = tmp_path / 'sp.env'
        self._write(out, _top_block_env(
            _fg(), surface=BoundaryProperties(acoustic_type='vacuum')),
            BoundaryType.VACUUM, BoundaryType.RIGID)
        lines = out.read_text().splitlines()
        assert lines[3][2] == 'V', f"TopOpt(2) is not 'V': {lines[3]}"
        assert _floats(lines[4]) == [10.0, 35.0, 8.0, 50.0]
        # The mesh line comes straight after the absorption block.
        assert _floats(lines[5]) == [0.0, 0.0, 100.0], (
            f"expected the SSP mesh line, got {lines[5]!r}")

    @pytest.mark.requires_binary
    def test_sparc_reads_the_vacuum_rigid_deck(self, tmp_path):
        from uacpy.core import BoundaryProperties
        from uacpy.core.constants import BoundaryType
        self._write(tmp_path / 'sp.env', _top_block_env(
            _fg(), surface=BoundaryProperties(acoustic_type='vacuum')),
            BoundaryType.VACUUM, BoundaryType.RIGID)
        text = _run_at('sparc.exe', 'sp', tmp_path)
        assert 'Francois-Garrison' in text


class TestTopReflectionTableStaging:
    """A ``'file'`` surface makes AT open ``<root>.trc`` (RefCoef.f90:64-76),
    so the table has to be staged beside the .env just like the bottom .brc."""

    @staticmethod
    def _table(tmp_path):
        from uacpy.io.refl_io import write_reflection_coefficient
        path = tmp_path / 'top_table.trc'
        angles = np.linspace(0.0, 90.0, 19)
        write_reflection_coefficient(
            path, angles, np.column_stack([np.full_like(angles, 0.9),
                                           np.zeros_like(angles)]))
        return path

    @staticmethod
    def _env(reflection_file):
        from uacpy.core import BoundaryProperties
        return _top_block_env(surface=BoundaryProperties(
            acoustic_type='file', reflection_file=str(reflection_file)))

    @pytest.mark.parametrize('writer', ['kraken', 'scooter'])
    def test_top_table_is_staged_beside_the_env(self, tmp_path, writer):
        from uacpy.core.constants import BoundaryType
        table = self._table(tmp_path)
        out = tmp_path / f'{writer}.env'
        write = _write_kraken if writer == 'kraken' else _write_scooter
        write(out, self._env(table), surface_type=BoundaryType.FILE)
        assert out.read_text().splitlines()[3][2] == 'F'
        staged = out.with_suffix('.trc')
        assert staged.exists(), (
            "TopOpt(2)='F' was written with no .trc beside the .env")
        # Staging normalises the copy (phase unwrap + duplicate-abscissa
        # removal), so it is the values that must survive, not the bytes:
        # ``misc/RefCoef.f90:119`` requires an unwrapped phi and ``:165``
        # divides by the abscissa gap with no guard.
        assert np.loadtxt(staged, skiprows=1) == pytest.approx(
            np.loadtxt(table, skiprows=1))

    def test_file_surface_without_a_table_raises(self, tmp_path):
        from uacpy.core import BoundaryProperties
        from uacpy.core.constants import BoundaryType
        env = _top_block_env(
            surface=BoundaryProperties(acoustic_type='file'))
        with pytest.raises(ConfigurationError, match='reflection_file'):
            _write_kraken(tmp_path / 'kr.env', env,
                          surface_type=BoundaryType.FILE)

    @pytest.mark.requires_binary
    @pytest.mark.parametrize('exe,writer', [('krakenc.exe', 'kraken'),
                                            ('scooter.exe', 'scooter')])
    def test_binary_reads_the_staged_top_table(self, tmp_path, exe, writer):
        from uacpy.core.constants import BoundaryType
        table = self._table(tmp_path)
        write = _write_kraken if writer == 'kraken' else _write_scooter
        write(tmp_path / 'trc.env', self._env(table),
              surface_type=BoundaryType.FILE)
        text = _run_at(exe, 'trc', tmp_path)
        assert 'tabulated top' in text, (
            "the binary did not read the staged .trc:\n" + text[:1500])


class TestPrecalcBottomStaging:
    """``acoustic_type='precalc'`` writes ``BotOpt='P'``, which makes AT open
    ``<root>.irc`` (RefCoef.f90:92-96) — BOUNCE's internal table, carried by
    ``reflection_file`` and published as ``result.metadata['irc_file']``."""

    @staticmethod
    def _env(reflection_file):
        from uacpy.core import BoundaryProperties
        return uacpy.Environment(
            name='ircdeck', bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.0, 1500.0)]),
            bottom=BoundaryProperties(acoustic_type='precalc',
                                      reflection_file=str(reflection_file)))

    @staticmethod
    def _write(path, env):
        from uacpy.core.constants import BoundaryType
        return _write_kraken(path, env, surface_type=BoundaryType.VACUUM,
                             bottom_type=BoundaryType.PRECALC)

    def test_irc_is_staged_beside_the_env(self, tmp_path):
        table = tmp_path / 'bounce_run.irc'
        table.write_text("'BOUNCE' 200.0\n1\n  0.1 0.2 0.3 0.4 0.5 0\n")
        out = self._write(tmp_path / 'kr.env', self._env(table))
        bot = [ln for ln in out.read_text().splitlines()
               if ln.startswith("'P'")]
        assert bot, "BotOpt='P' was not written"
        staged = out.with_suffix('.irc')
        assert staged.exists(), (
            "BotOpt='P' was written with no .irc beside the .env")
        assert staged.read_text() == table.read_text()

    def test_precalc_without_a_table_raises(self, tmp_path):
        from uacpy.core import BoundaryProperties
        env = uacpy.Environment(
            name='ircdeck', bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.0, 1500.0)]),
            bottom=BoundaryProperties(acoustic_type='precalc'))
        with pytest.raises(ConfigurationError, match='reflection_file'):
            self._write(tmp_path / 'kr.env', env)

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_kraken_reads_a_bounce_irc(self, tmp_path):
        from uacpy.core import BoundaryProperties
        src, rcv = _src_rcv()
        seabed = uacpy.Environment(
            name='seabed', bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.0, 1500.0)]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.5,
                                      attenuation=0.5))
        bounce = uacpy.models.Bounce(verbose=False,
                                     work_dir=tmp_path / 'bounce',
                                     cleanup=False)
        irc = Path(bounce.run(seabed, src, rcv).metadata['irc_file'])
        self._write(tmp_path / 'irc.env', self._env(irc))
        text = _run_at('kraken.exe', 'irc', tmp_path)
        assert 'PRECALCULATED IRC' in text


class TestScooterDeckHasNoReceiverRanges:
    """``scooter.f90:158-176`` (GetPar) stops at ReadfreqVec — it never calls
    ReadRcvrRanges, so the deck must end after the receiver depths."""

    def test_docstring_does_not_promise_a_receiver_range_block(self):
        from uacpy.io.oalib_writer import write_scooter_env_file
        assert 'receiver-range block' not in write_scooter_env_file.__doc__

    def test_deck_ends_after_the_receiver_depths(self, tmp_path):
        from uacpy.core import BoundaryProperties
        out = _write_scooter(
            tmp_path / 'sc.env',
            _top_block_env(surface=BoundaryProperties(acoustic_type='vacuum')))
        lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
        assert _floats(lines[-1]) == [50.0], (
            f"the deck must end on the receiver depths; got {lines[-1]!r}")
        assert lines[-2].strip() == '1'


class TestBroadbandFlagHandlesScalarFrequencies:
    """``write_header`` and ``write_kraken_env_file`` decide TopOpt(6) from the
    same frequency vector, so they must measure it the same way."""

    @pytest.mark.parametrize('frequencies,expected', [
        (None, ' '),
        (np.asarray(200.0), ' '),
        (np.asarray([200.0]), ' '),
        (np.asarray([200.0, 400.0]), 'B'),
    ])
    def test_topopt_broadband_column(self, frequencies, expected):
        import io
        from uacpy.core import BoundaryProperties
        from uacpy.core.constants import BoundaryType
        from uacpy.io.oalib_writer import write_header
        src, _ = _src_rcv()
        buf = io.StringIO()
        write_header(
            buf, _top_block_env(
                surface=BoundaryProperties(acoustic_type='vacuum')),
            src, ssp_topopt='C', surface_type=BoundaryType.VACUUM,
            frequencies=frequencies)
        assert buf.getvalue().splitlines()[3][6] == expected


class TestBottomOptionLineIsSingleCharacter:
    """``misc/ReadEnvironmentMod.f90:121-129`` reads BotOpt(1:8) but uses
    only BotOpt(1:1), as ``HSBot%BC``; the '~' bathymetry flag is Bellhop's
    alone and Bellhop has its own writer."""

    def test_range_dependent_bathymetry_adds_no_flag(self, tmp_path):
        from uacpy.core import BoundaryProperties
        from uacpy.core.environment import Bathymetry
        env = uacpy.Environment(
            name='deck',
            bathymetry=Bathymetry(ranges=[0.0, 5000.0], depths=[100.0, 100.0]),
            ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.0, 1500.0)]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.5,
                                      attenuation=0.5))
        from uacpy.core.constants import BoundaryType
        text = _write_kraken(tmp_path / 'kr.env', env,
                             surface_type=BoundaryType.VACUUM).read_text()
        assert "'A' " in text and '~' not in text


class TestBoundaryTypeIsTheSingleSourceOfTruth:
    """``BoundaryType`` is the only place a boundary letter is decided. A
    silent vacuum fallback for an unrecognised type would model a different
    surface than the one asked for."""

    def test_map_agrees_with_the_enum(self):
        from uacpy.core.constants import BoundaryType
        from uacpy.io.oalib_writer import _BOUNDARY_TYPE_MAP
        for bt in BoundaryType:
            assert _BOUNDARY_TYPE_MAP[bt.value] == bt.to_acoustics_toolbox_code()
        assert _BOUNDARY_TYPE_MAP['halfspace'] == 'A'

    def test_unknown_acoustic_type_raises(self):
        from uacpy.core import BoundaryProperties
        from uacpy.io.oalib_writer import get_top_bc_code
        env = _top_block_env(
            surface=BoundaryProperties(acoustic_type='vacuum'))
        object.__setattr__(env.surface.properties[0], 'acoustic_type', 'mud')
        with pytest.raises(ConfigurationError, match='boundary type'):
            get_top_bc_code(env)


class TestDeckDepthQuantisation:
    """Every interface an AT ``.env`` declares is quantised onto the deck's
    0.1 m depth column. Rounding up keeps the water column at or below the
    physical ``env.depth`` — a source or receiver on the seafloor stays inside
    the mesh instead of being moved up by ReadSzRz — and keeps the AT models
    on the same water column as Bellhop's ``.bty``-clipped mesh.

    ``deck_depth`` delivers that by round-tripping through the one depth format
    every deck writes, so the value uacpy compares is the value the Fortran
    parses. The Acoustics-Toolbox manual states the licence for this outright:
    *"All user input in all modules is read using list-directed I/O. Thus data
    can be typed in free-format"* (``doc/index.htm``)."""

    def test_bellhop_shares_the_one_quantiser(self):
        """Two implementations of the same rule would drift; Bellhop's SSP
        header depth must be the identical function object."""
        from uacpy.io import bellhop_writer
        from uacpy.io.oalib_writer import deck_depth
        assert bellhop_writer.deck_depth is deck_depth

    @pytest.mark.parametrize('depth', [
        100.0, 150.0, 2000.0, 100.04, 100.05, 100.06, 99.999, 0.1, 200.3])
    def test_a_fractional_depth_survives_to_the_deck(self, depth):
        """The requested depth must reach the deck, not a coarsened stand-in.

        Snapping to a 0.1 m grid put ``OAST`` 11.20 dB (max) from the answer for
        the depth the caller built, and made ``Scooter`` insensitive to a 6 cm
        change entirely.
        """
        from uacpy.io.oalib_writer import (deck_depth,
                                           _DECK_DEPTH_RESOLUTION_M)
        got = deck_depth(depth)
        assert abs(got - depth) <= _DECK_DEPTH_RESOLUTION_M

    @pytest.mark.parametrize('depth', [
        100.0, 150.0, 2000.0, 200.3, 100.04, 42.2999960])
    def test_deck_depth_is_idempotent(self, depth):
        """It reports what the deck writes, so re-applying it changes nothing —
        which is what makes the mesh line and the SSP rows agree exactly."""
        from uacpy.io.oalib_writer import deck_depth
        once = deck_depth(depth)
        assert deck_depth(once) == once

    def test_the_resolution_is_finer_than_the_readers_own_tolerance(self):
        """``misc/sspMod.f90:353`` ends a medium on
        ``100 * EPSILON( 1.0e0 )`` = 1.19e-05 m, so the written resolution has to
        sit below that for the mesh line and the SSP rows to count as equal."""
        import numpy as np
        from uacpy.io.oalib_writer import _DECK_DEPTH_RESOLUTION_M
        assert _DECK_DEPTH_RESOLUTION_M < 100.0 * np.finfo(np.float32).eps

    @staticmethod
    def _mesh_line_depth(path):
        """Third field of the first ``NG sigma Depth`` mesh line (Bellhop
        writes the same record comma-separated)."""
        for line in Path(path).read_text().splitlines():
            parts = line.replace(',', ' ').split()
            if (len(parts) == 3 and parts[0].isdigit()
                    and not line.lstrip().startswith("'")):
                return float(parts[2])
        raise AssertionError(f"no mesh line found in {path}")

    @staticmethod
    def _water_ssp_rows(path):
        """SSP rows of medium 1: everything between the mesh line and the next
        quoted option line."""
        rows, started = [], False
        for line in Path(path).read_text().splitlines():
            parts = line.replace(',', ' ').split()
            if not started:
                started = (len(parts) == 3 and parts[0].isdigit()
                           and not line.lstrip().startswith("'"))
                continue
            if line.lstrip().startswith("'"):
                break
            rows.append(parts)
        return rows

    def test_water_column_is_never_shallower_than_env_depth(self, tmp_path):
        from uacpy.core import BoundaryProperties
        from uacpy.core.constants import BoundaryType
        env = uacpy.Environment(
            name='deck', bathymetry=100.04,
            ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.04, 1500.0)]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.5,
                                      attenuation=0.5))
        out = _write_kraken(tmp_path / 'kr.env', env,
                            surface_type=BoundaryType.VACUUM)
        mesh_depth = self._mesh_line_depth(out)
        assert mesh_depth >= 100.04, (
            f"water column ends at {mesh_depth} m, above env.depth=100.04")
        ssp_rows = self._water_ssp_rows(out)
        assert float(ssp_rows[-1][0]) == mesh_depth, (
            "the deepest SSP sample must equal the mesh depth exactly")

    def test_at_and_bellhop_decks_share_the_water_column(self, tmp_path):
        from uacpy.core import BoundaryProperties
        from uacpy.core.constants import BoundaryType
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        src, rcv = _src_rcv()
        env = uacpy.Environment(
            name='deck', bathymetry=100.04,
            ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.04, 1500.0)]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.5,
                                      attenuation=0.5))
        at = _write_kraken(tmp_path / 'kr.env', env,
                           surface_type=BoundaryType.VACUUM)
        bh = tmp_path / 'bh.env'
        write_bellhop_env_file(bh, env, src, rcv)
        assert self._mesh_line_depth(at) == self._mesh_line_depth(bh), (
            "AT and Bellhop model different water columns for the same env")

    def test_layer_interfaces_keep_their_thickness(self, tmp_path):
        from uacpy.core.bottom import (
            SeabedColumn, SedimentLayer, BoundaryProperties)
        from uacpy.core.constants import BoundaryType
        bottom = SeabedColumn(
            layers=[SedimentLayer(thickness=5.0, sound_speed=1600.0,
                                  density=1.6, attenuation=0.2)],
            halfspace=BoundaryProperties(acoustic_type='half-space',
                                         sound_speed=1800.0, density=2.0,
                                         attenuation=0.5))
        env = uacpy.Environment(
            name='deck', bathymetry=100.04,
            ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.04, 1500.0)]),
            bottom=bottom)
        out = _write_kraken(tmp_path / 'kr.env', env,
                            surface_type=BoundaryType.VACUUM)
        text = out.read_text()
        # The interfaces carry the depth the caller asked for, not a value
        # snapped onto a 0.1 m grid: OAST answers 11.20 dB (max) differently for
        # 100.04 m than for 100.1 m, and Scooter was insensitive to the
        # difference entirely because both decks said 100.1.
        assert '  100.040000 1600.000000' in text, (
            f"seafloor interface is not the requested depth:\n{text}")
        assert '  105.040000 1600.000000' in text, (
            f"layer base is not seafloor + thickness:\n{text}")


# ─────────────────────────────────────────────────────────────────────────────
# Irregular receiver grid — Bellhop RunType(5:5) = 'I'
# ─────────────────────────────────────────────────────────────────────────────

_IRREGULAR_DECK_HEAD = """\
'irregular grid probe'
{freq:.1f}
1
'CVW'
0 0.0 {depth:.1f}
0.0 1500.0 /
{depth:.1f} 1500.0 /
'A' 0.0
{depth:.1f} 1800.0 0.0 1.8 0.5 /
1
25.0 /
"""


def _write_bellhop_deck(path, run_type, depths_m, ranges_km,
                        freq=100.0, depth=200.0):
    """Write an isovelocity Bellhop ``.env`` whose receiver depth and range
    vectors are given explicitly, with ``run_type`` in the RunType columns."""
    text = _IRREGULAR_DECK_HEAD.format(freq=freq, depth=depth)
    text += (f"{len(depths_m)}\n"
             + " ".join(f"{d:.1f}" for d in depths_m) + "\n")
    text += (f"{len(ranges_km)}\n"
             + " ".join(f"{r:.4f}" for r in ranges_km) + "\n")
    # Tail records: RunType, NBeams, the take-off fan, then
    # ``deltas Box%z Box%r`` (Bellhop/ReadEnvironmentBell.f90:146,154 — z in
    # m, r in km). deltas=0 asks Bellhop to pick its own step. The box is
    # padded past the deepest receiver and the farthest range because
    # Bellhop/bellhop.f90:571-574 stops a ray the moment it steps outside it.
    text += (f"'{run_type}'\n501\n-45 45 /\n"
             f"0.0 {depth + 50.0:.1f} {ranges_km[-1] * 1.5:.4f}\n")
    path.write_text(text)


# 16 paired receivers: the smallest count for which the .shd header product
# Nrz*Nrr (256) exceeds what the 1804-byte file can hold (225 complex words).
_PAIRED_DEPTHS = [20.0 + 10.0 * i for i in range(16)]
_PAIRED_RANGES_KM = [0.2 + 0.1 * i for i in range(16)]


@pytest.mark.requires_binary
class TestIrregularReceiverGrid:
    """``RunType(5:5) = 'I'`` makes the receivers the paired coordinates
    ``(Rz(i), Rr(i))``: Bellhop writes one row of ``NRr`` pressure records per
    source depth (``Bellhop/bellhop.f90:202-206``, ``:323-326``) and one
    ``.arr`` depth block (``Bellhop/ArrMod.f90:101-102`` with ``Nrd`` bound to
    ``NRz_per_range`` at ``bellhop.f90:329``), while both headers still report
    the full ``Pos%NRz`` (``misc/RWSHDFile.f90:113``,
    ``Bellhop/ReadEnvironmentBell.f90:591``).

    The paired receiver ``i`` is physically the same point as cell ``(i, i)``
    of the rectilinear grid over the same two vectors
    (``Bellhop/influence.f90:460-464``), so every value here is checked
    against that diagonal.
    """

    @staticmethod
    def _run(tmp_path, root, run_type):
        _write_bellhop_deck(tmp_path / f'{root}.env', run_type,
                            _PAIRED_DEPTHS, _PAIRED_RANGES_KM)
        _run_at('bellhop.exe', root, tmp_path)
        return tmp_path / root

    def test_shd_header_bound_admits_paired_receivers(self, tmp_path):
        """The on-disk sample count is Nsz*1*Nrr for an irregular grid, so the
        header sanity bound must not multiply Nrz by Nrr."""
        from uacpy.io.oalib_reader import read_shd_bin

        root = self._run(tmp_path, 'irr', 'CG  I')
        shd = read_shd_bin(str(root.with_suffix('.shd')))
        assert shd['PlotType'].strip() == 'irregular'
        assert shd['pressure'].shape == (1, 1, 1, len(_PAIRED_RANGES_KM))
        # The header still declares the full receiver-depth vector.
        assert len(shd['Pos']['r']['z']) == len(_PAIRED_DEPTHS)

    def test_irregular_shd_is_one_paired_receiver_axis(self, tmp_path):
        from uacpy.io.oalib_reader import read_shd_file

        irr = read_shd_file(self._run(tmp_path, 'irr', 'CG  I')
                            .with_suffix('.shd'))
        rect = read_shd_file(self._run(tmp_path, 'rect', 'CG  R')
                             .with_suffix('.shd'))

        assert irr.axes == ['range']
        assert irr.shape == (len(_PAIRED_RANGES_KM),)
        assert irr.metadata['receiver_depths'].tolist() == _PAIRED_DEPTHS
        assert irr.coords['range'] == pytest.approx(
            [1000.0 * r for r in _PAIRED_RANGES_KM])
        assert np.allclose(irr.data, np.diag(rect.data), equal_nan=True)

    def test_irregular_arr_carries_one_depth_block(self, tmp_path):
        from uacpy.io.oalib_reader import read_arr_file

        irr = read_arr_file(self._run(tmp_path, 'airr', 'A   I')
                            .with_suffix('.arr'), grid_type='I')
        rect = read_arr_file(self._run(tmp_path, 'arect', 'A   R')
                             .with_suffix('.arr'))

        assert len(irr.by_receiver[0]) == 1
        assert len(irr.by_receiver[0][0]) == len(_PAIRED_RANGES_KM)
        assert irr.receiver_depths.tolist() == _PAIRED_DEPTHS
        for i in range(len(_PAIRED_RANGES_KM)):
            paired = irr.by_receiver[0][0][i]
            cell = rect.by_receiver[0][i][i]
            assert paired['n_arrivals'] == cell['n_arrivals'] > 0
            assert paired['delays'] == pytest.approx(cell['delays'])
            assert paired['amplitudes'] == pytest.approx(cell['amplitudes'])

    def test_irregular_arr_read_as_rectilinear_runs_out_of_records(
            self, tmp_path):
        """The header's NRz is not the body's depth-loop bound, so parsing an
        irregular .arr with the header count exhausts the token stream."""
        from uacpy.io.oalib_reader import read_arr_file
        from uacpy.core.exceptions import FileFormatError

        path = self._run(tmp_path, 'airr', 'A   I').with_suffix('.arr')
        with pytest.raises(FileFormatError):
            read_arr_file(path)

    def test_grid_type_I_requires_paired_header_counts(self, tmp_path):
        """Bellhop/ReadEnvironmentBell.f90:414 ERROUTs unless NRz == NRr on an
        irregular deck, so a header with unequal counts is not one."""
        from uacpy.io.oalib_reader import read_arr_file
        from uacpy.core.exceptions import FileFormatError

        _write_bellhop_deck(tmp_path / 'lop.env', 'A   R',
                            _PAIRED_DEPTHS[:3], _PAIRED_RANGES_KM[:5])
        _run_at('bellhop.exe', 'lop', tmp_path)
        with pytest.raises(FileFormatError, match='pairs them one-to-one'):
            read_arr_file(tmp_path / 'lop.arr', grid_type='I')


class TestArrivalsGridTypeValidation:
    def test_unknown_grid_type_rejected(self, tmp_path):
        from uacpy.io.oalib_reader import read_arr_file

        path = tmp_path / 'x.arr'
        path.write_text("'2D'\n")
        with pytest.raises(ConfigurationError, match="grid_type='X'"):
            read_arr_file(path, grid_type='X')


# ─────────────────────────────────────────────────────────────────────────────
# BELLHOP3D outputs: axes and layouts the 2-D readers do not carry
# ─────────────────────────────────────────────────────────────────────────────

_BH3D_DECK = """\
'free space 3D, Hat Cart. Coord'
5.000000
1
'CAF'
0.0 1500.0 /
500 0.0 5000.0
   0.0  1500.0 /
5000.0  1500.0 /
'A'  0.0
5000.0  /
1
0.0 /
1
0.0 /
1
3000.0
4
1000 4000 /
4
1.0 4.0 /
3
0.0 10.0 20.0
'{run_type}'
41
-89 89 /
5
-5  5 /
100.0  10.05 10.05 5000.5
"""


@pytest.mark.requires_binary
class TestBellhop3DOutputsAreNotSilentlyFlattened:
    """A BELLHOP3D run writes axes and coordinate systems the 2-D typed
    readers have no place for: an ``Ntheta`` bearing axis in the ``.shd``
    (``misc/RWSHDFile.f90:105,107``; ``Bellhop/bellhop3D.f90:405-411``), an
    ``'xyz'`` ray layout with a radian take-off angle
    (``Bellhop/ReadEnvironmentBell.f90:564-568``; ``Bellhop/WriteRay.f90:89``
    vs ``:45``; ``Bellhop/bellhop3D.f90:360`` vs ``Bellhop/bellhop.f90:263``),
    and a ten-value ``'3D'`` arrivals record
    (``Bellhop/ArrMod.f90:256-302``). Each must be refused, not reduced."""

    @staticmethod
    def _run(tmp_path, root, run_type):
        (tmp_path / f'{root}.env').write_text(
            _BH3D_DECK.format(run_type=run_type))
        _run_at('bellhop3d.exe', root, tmp_path)
        return tmp_path / root

    def test_read_shd_bin_returns_every_bearing(self, tmp_path):
        from uacpy.io.oalib_reader import read_shd_bin

        shd = read_shd_bin(str(self._run(tmp_path, 'tl3d', 'C^   3')
                               .with_suffix('.shd')))
        assert shd['Pos']['theta'].tolist() == [0.0, 10.0, 20.0]
        assert shd['pressure'].shape[0] == 3

    def test_read_shd_file_refuses_multi_bearing(self, tmp_path):
        from uacpy.io.oalib_reader import read_shd_file
        from uacpy.core.exceptions import FileFormatError

        path = self._run(tmp_path, 'tl3d', 'C^   3').with_suffix('.shd')
        with pytest.raises(FileFormatError, match='3 receiver bearings'):
            read_shd_file(path)

    def test_xyz_ray_file_refused(self, tmp_path):
        from uacpy.io.oalib_reader import read_ray_file
        from uacpy.core.exceptions import FileFormatError

        path = self._run(tmp_path, 'ray3d', 'R^   3').with_suffix('.ray')
        assert "'xyz'" in path.read_text(errors='ignore')[:400]
        with pytest.raises(FileFormatError, match="'xyz'"):
            read_ray_file(path)

    def test_3d_arrivals_refused_inside_the_typed_hierarchy(self, tmp_path):
        from uacpy.io.oalib_reader import read_arr_file
        from uacpy.core.exceptions import UACPYError

        path = self._run(tmp_path, 'arr3d', 'A^   3').with_suffix('.arr')
        with pytest.raises(UACPYError, match='3-D arrivals'):
            read_arr_file(path)


class TestRayFileIsAsciiOnly:
    """The only two ``.ray`` OPENs in the vendored tree,
    ``Bellhop/ReadEnvironmentBell.f90:556`` and
    ``KrakenField/EvaluateGBMod.f90:64``, are both ``FORM = 'FORMATTED'``, so a
    parse failure is an ASCII parse failure and must name the offending
    token."""

    def test_corrupt_token_is_named(self, tmp_path):
        from uacpy.io.oalib_reader import read_ray_file
        from uacpy.core.exceptions import FileFormatError

        path = tmp_path / 'corrupt.ray'
        path.write_text(
            " 'BELLHOP- probe'\n 50.0\n 1 1 1\n 2 1\n 0.0\n 200.0\n 'rz'\n"
            " -20.0\n 5X8 0 0\n"
        )
        with pytest.raises(FileFormatError) as exc:
            read_ray_file(path)
        assert '5X8' in str(exc.value)
        assert 'byte order' not in str(exc.value)


class TestShdAscIsATokenStream:
    """``Matlab/ReadWrite/read_shd_asc.m:15-27`` reads every scalar and vector
    with ``fscanf``, so the file is one whitespace-delimited token stream after
    the two title lines — no per-line layout to honour."""

    _COUNTS = "1 1 1 2 2"
    _SCALARS = "100.0 0.0"
    _VECTORS = ["50.0", "0.0", "10.0", "20.0", "30.0", "100.0", "200.0"]
    _DATA = ["1", "2", "3", "4", "5", "6", "7", "8"]

    def _layouts(self):
        rest = self._VECTORS + self._DATA
        return {
            'counts_and_scalars_share_a_line':
                self._COUNTS + " " + self._SCALARS + "\n" + "\n".join(rest),
            'one_value_per_line':
                self._COUNTS + "\n" + self._SCALARS + "\n" + "\n".join(rest),
            'fully_packed':
                " ".join([self._COUNTS, self._SCALARS] + rest),
        }

    def test_every_whitespace_layout_reads_the_same(self, tmp_path):
        from uacpy.io.oalib_reader import read_shd_asc

        for name, body in self._layouts().items():
            path = tmp_path / f'{name}.asc'
            path.write_text("'probe'\n'rectilin  '\n" + body + "\n")
            shd = read_shd_asc(path)
            assert shd['freq0'] == pytest.approx(100.0), name
            assert shd['atten'] == pytest.approx(0.0), name
            assert shd['Pos']['r']['z'].tolist() == [20.0, 30.0], name
            assert shd['Pos']['r']['r'].tolist() == [100.0, 200.0], name
            assert shd['pressure'].ravel().tolist() == [
                1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], name

    def test_implausible_count_raises_instead_of_allocating(self, tmp_path):
        """A count the file cannot back must surface as a typed parse error,
        never an allocation the token stream can never fill.

        Runs in a subprocess that caps its address space after the import, so
        a header count of 2e9 cannot be satisfied by the allocator and the
        distinction between a typed parse error and ``MemoryError`` is real.
        """
        import subprocess
        import sys
        import textwrap

        path = tmp_path / 'hostile.asc'
        path.write_text("'probe'\n'rectilin  '\n2000000000 1 1 1 1\n100.0 0.0\n")
        script = textwrap.dedent(f"""
            import resource
            from uacpy.io.oalib_reader import read_shd_asc
            from uacpy.core.exceptions import FileFormatError
            resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 ** 3,) * 2)
            try:
                read_shd_asc({str(path)!r})
            except FileFormatError:
                print('TYPED')
        """)
        proc = subprocess.run([sys.executable, '-c', script],
                              capture_output=True, text=True, timeout=300)
        assert 'TYPED' in proc.stdout, (
            f"expected FileFormatError, got:\n{proc.stderr[-800:]}")


class TestInterfaceRoughnessGoesToItsOwnInterface:
    """``SSP%sigma(M)`` is the roughness of the interface at the TOP of medium
    ``M`` — ``ReadEnvironmentMod.f90:88`` reads it on medium ``M``'s mesh line
    and ``kraken.f90:902`` pairs it with the media on either side. So with a
    sediment layer the seafloor is ``sigma(2)``, on the layer's own line, and
    the half-space's roughness stays on the BotOpt line one interface deeper.
    """

    @staticmethod
    def _deck(tmp_path, *, surface, seafloor, halfspace):
        from uacpy.core import Environment, Source, Receiver
        from uacpy.core.bottom import (Bottom, BoundaryProperties,
                                       SeabedColumn, SedimentLayer)
        from uacpy.core.constants import BoundaryType
        from uacpy.core.surface import Surface
        from uacpy.io.oalib_writer import write_kraken_env_file
        env = Environment(
            bathymetry=100.0, ssp=1500.0,
            surface=Surface([BoundaryProperties(acoustic_type='vacuum',
                                                roughness=surface)]),
            bottom=Bottom.from_column(SeabedColumn(
                layers=[SedimentLayer(thickness=30, sound_speed=1600,
                                      density=1.8, attenuation=0.5,
                                      roughness=seafloor)],
                halfspace=BoundaryProperties(
                    acoustic_type='half-space', sound_speed=1800, density=2.0,
                    attenuation=0.6, roughness=halfspace))))
        out = tmp_path / 'rough.env'
        write_kraken_env_file(
            out, env, Source(depths=50, frequencies=100.0),
            Receiver(depths=50, ranges=[5000.0]),
            ssp_topopt='CVW   ', surface_type=BoundaryType.VACUUM,
            bottom_type=BoundaryType.HALF_SPACE, frequencies=[100.0],
            n_mesh=0, rmax_m=5000.0, c_low=0.0, c_high=2000.0)
        return out.read_text().splitlines()

    def test_each_sigma_comes_from_its_own_carrier(self, tmp_path):
        lines = self._deck(tmp_path, surface=1.5, seafloor=2.0, halfspace=0.7)
        mesh = [ln for ln in lines if re.match(r'^\d+\s+[\d.]+\s+[\d.]+$', ln)]
        assert float(mesh[0].split()[1]) == pytest.approx(1.5)   # sea surface
        assert float(mesh[1].split()[1]) == pytest.approx(2.0)   # seafloor
        botopt = next(ln for ln in lines if ln.startswith("'A'"))
        assert float(botopt.split()[1]) == pytest.approx(0.7)    # base of stack

    def test_smooth_layer_writes_a_smooth_interface(self, tmp_path):
        lines = self._deck(tmp_path, surface=0.0, seafloor=0.0, halfspace=0.0)
        mesh = [ln for ln in lines if re.match(r'^\d+\s+[\d.]+\s+[\d.]+$', ln)]
        assert all(float(ln.split()[1]) == 0.0 for ln in mesh[:2])


class TestOasesInterfaceRoughness:
    """OASES reads column 7 of each layer record as that interface's RMS
    roughness (``src/oaseun31.f:54``; ``doc/oast.tex:42,48`` — the record opens
    with "D: Depth of interface" and RG is "RMS value of interface roughness").
    A record's RG therefore belongs to the interface at the TOP of its layer,
    the same convention Kraken uses for ``SSP%sigma``.
    """

    @staticmethod
    def _env(*, surface, seafloor, base):
        from uacpy.core import Environment
        from uacpy.core.bottom import (Bottom, BoundaryProperties,
                                       SeabedColumn, SedimentLayer)
        from uacpy.core.surface import Surface
        return Environment(
            bathymetry=100.0, ssp=1500.0,
            surface=Surface([BoundaryProperties(acoustic_type='vacuum',
                                                roughness=surface)]),
            bottom=Bottom.from_column(SeabedColumn(
                layers=[SedimentLayer(thickness=30, sound_speed=1600,
                                      density=1.8, attenuation=0.5,
                                      roughness=seafloor)],
                halfspace=BoundaryProperties(
                    acoustic_type='half-space', sound_speed=1800, density=2.0,
                    attenuation=0.6, roughness=base))))

    def test_each_record_carries_its_own_interface(self, tmp_path):
        from uacpy.core import Source, Receiver
        from uacpy.io.oases_writer import write_oast_input
        out = tmp_path / 'rg.dat'
        write_oast_input(out, self._env(surface=1.5, seafloor=2.0, base=0.7),
                         Source(depths=50, frequencies=100.0),
                         Receiver(depths=50, ranges=[5000.0]))
        rows = [ln.split() for ln in out.read_text().splitlines()
                if re.match(r'^[\d.]+(\s+[-\d.]+){7}$', ln)]
        by_depth = {r[0]: float(r[6]) for r in rows}
        # Record '0' is the upper half-space (layer 1); '0.00' the first water
        # layer (layer 2). ROUGH(1) is overwritten by ROUGH(2) at
        # oaseun31.f:377 and both manuals call RG(1) dummy (oast.tex:344,
        # oasp.tex:365), so the sea surface can only ride on the water record.
        assert by_depth['0'] == 0.0                       # RG(1), discarded
        assert by_depth['0.00'] == pytest.approx(1.5)     # sea surface
        assert by_depth['100.00'] == pytest.approx(2.0)   # seafloor
        assert by_depth['130.00'] == pytest.approx(0.7)   # base of the stack

    def test_unlayered_column_puts_halfspace_roughness_on_the_seafloor(self, tmp_path):
        from uacpy.core import Environment, Source, Receiver
        from uacpy.core.bottom import Bottom, BoundaryProperties
        from uacpy.io.oases_writer import write_oast_input
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=Bottom.from_halfspace(BoundaryProperties(
                              acoustic_type='half-space', sound_speed=1600,
                              density=1.5, attenuation=0.5, roughness=1.2)))
        out = tmp_path / 'rg1.dat'
        write_oast_input(out, env, Source(depths=50, frequencies=100.0),
                         Receiver(depths=50, ranges=[5000.0]))
        # Selected by the layer record's shape (8 columns), not by a numeric
        # prefix: the frequency record now carries the same digits, and keying on
        # a format is only stable while different records happen to print
        # differently.
        seabed = next(ln for ln in out.read_text().splitlines()
                      if len(ln.split()) == 8
                      and ln.split()[0].startswith('100.'))
        assert float(seabed.split()[6]) == pytest.approx(1.2)


class TestOasesSeaSurfaceRoughnessRecord:
    """``env.surface.roughness`` must land on ROUGH(2), not ROUGH(1).

    INENVI reads column 7 of every layer record into ``ROUGH(M)``
    (``oases/src/oaseun31.f:54``) and then unconditionally overwrites
    ``ROUGH(1)`` with ``ROUGH(2)`` the moment the read loop closes
    (``:376-377``). ``DO 1111 M=2,NUML`` compares ``LAYTYP(M-1)`` with
    ``LAYTYP(M)`` (``:381-383``), which fixes ROUGH(M) as the roughness of
    the interface at the TOP of layer M; ``oasnun22.f:250`` places OASN's
    surface-noise sheet at ``V(2,1)+V(1,2)/(30*FREQ1)+ROUGH(2)`` on the same
    reading. Both manuals state "RG(1) is dummy" (``doc/oast.tex:344``,
    ``doc/oasp.tex:365``). So the sea surface is the FIRST WATER LAYER's RG,
    and anything written to the upper-half-space record is discarded.
    """

    SURFACE_RG = 1.5

    @staticmethod
    def _env(ssp):
        from uacpy.core import Environment
        from uacpy.core.bottom import Bottom, BoundaryProperties
        from uacpy.core.surface import Surface
        return Environment(
            bathymetry=100.0, ssp=ssp,
            surface=Surface([BoundaryProperties(
                acoustic_type='vacuum',
                roughness=TestOasesSeaSurfaceRoughnessRecord.SURFACE_RG)]),
            bottom=Bottom.from_halfspace(BoundaryProperties(
                acoustic_type='half-space', sound_speed=1700, density=1.8,
                attenuation=0.5)))

    @staticmethod
    def _layer_records(text):
        """The deck's NL layer records, in deck order.

        The layer block opens with a bare integer NL (oaseun31.f:43) — the
        first such record in every OASES deck — followed by NL layer records
        (:54).
        """
        lines = text.splitlines()
        i = next(i for i, ln in enumerate(lines) if ln.strip().isdigit())
        n_layers = int(lines[i])
        return [ln.split() for ln in lines[i + 1:i + 1 + n_layers]]

    @pytest.mark.parametrize('writer_name,ssp', [
        ('write_oast_input', 1500.0),
        ('write_oast_input', [(0.0, 1500.0), (50.0, 1490.0), (100.0, 1510.0)]),
        ('write_oasn_input', [(0.0, 1500.0), (50.0, 1490.0), (100.0, 1510.0)]),
        ('write_oasp_input', [(0.0, 1500.0), (50.0, 1490.0), (100.0, 1510.0)]),
    ])
    def test_surface_rg_rides_on_the_first_water_record(
            self, tmp_path, writer_name, ssp):
        import warnings
        from uacpy.core import Source, Receiver
        import uacpy.io.oases_writer as W
        out = tmp_path / f'{writer_name}.dat'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            getattr(W, writer_name)(
                out, self._env(ssp), Source(depths=50, frequencies=100.0),
                Receiver(depths=[20.0, 50.0, 80.0],
                         ranges=np.linspace(100.0, 5000.0, 10)))
        records = self._layer_records(out.read_text())
        assert float(records[0][6]) == 0.0, "RG(1) is the dummy INENVI clobbers"
        assert float(records[1][6]) == pytest.approx(self.SURFACE_RG)
        # Interfaces inside the water column are SSP sample boundaries.
        assert all(float(r[6]) == 0.0 for r in records[2:-1])

    @pytest.mark.parametrize('writer_name,extra', [
        ('write_oast_input', 1), ('write_oasn_input', 1),
        ('write_oasp_input', 2),
    ])
    def test_record_arity_is_unchanged(self, tmp_path, writer_name, extra):
        """INENVI's reads are list-directed but positional, so column 7 has to
        stay column 7 and the inert padding past it has to stay put."""
        import warnings
        from uacpy.core import Source, Receiver
        import uacpy.io.oases_writer as W
        out = tmp_path / f'{writer_name}_arity.dat'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            getattr(W, writer_name)(
                out, self._env([(0.0, 1500.0), (50.0, 1490.0), (100.0, 1510.0)]),
                Source(depths=50, frequencies=100.0),
                Receiver(depths=[20.0, 50.0, 80.0],
                         ranges=np.linspace(100.0, 5000.0, 10)))
        records = self._layer_records(out.read_text())
        # The upper half-space record carries the manual's D CC CS AC AS RO
        # RG IG columns whatever the deck; the water and seabed records carry
        # this program's own trailing padding.
        assert len(records[0]) == 8
        assert all(len(r) == 7 + extra for r in records[1:])

    def test_gradient_first_layer_is_warned_about(self, tmp_path):
        """oaseun31.f:382-388 refuses roughness at an interface bounding a
        LAYTYP 2 (sound-speed-gradient) layer, but its STOP is commented out at
        :388 — so only a uacpy-side warning tells the caller."""
        from uacpy.core import Source, Receiver
        from uacpy.io.oases_writer import write_oast_input
        env = self._env([(0.0, 1500.0), (50.0, 1490.0), (100.0, 1510.0)])
        with pytest.warns(UserWarning, match='sound-speed gradient'):
            write_oast_input(tmp_path / 'grad.dat', env,
                             Source(depths=50, frequencies=100.0),
                             Receiver(depths=[20.0, 50.0, 80.0],
                                      ranges=np.linspace(100.0, 5000.0, 10)))

    def test_isovelocity_column_is_not_warned_about(self, tmp_path, recwarn):
        """An isovelocity first layer is LAYTYP 1, which the test at
        oaseun31.f:383 passes."""
        from uacpy.core import Source, Receiver
        from uacpy.io.oases_writer import write_oast_input
        write_oast_input(tmp_path / 'iso.dat', self._env(1500.0),
                         Source(depths=50, frequencies=100.0),
                         Receiver(depths=[20.0, 50.0, 80.0],
                                  ranges=np.linspace(100.0, 5000.0, 10)))
        assert not [w for w in recwarn
                    if 'sound-speed gradient' in str(w.message)]

    def test_oasr_layer_one_carries_no_surface_roughness(self, tmp_path):
        """OASR's layer 1 IS the water half-space (``SD=V(2,1)-1.0E-3``,
        unoasr21.f:89) — there is no sea surface in the model, and its RG is
        the same dummy. The reflecting interface is ROUGH(2), the seabed."""
        from uacpy.core import Environment, Source, Receiver
        from uacpy.core.bottom import Bottom, BoundaryProperties
        from uacpy.core.surface import Surface
        from uacpy.io.oases_writer import write_oasr_input
        env = Environment(
            bathymetry=100.0, ssp=1500.0,
            surface=Surface([BoundaryProperties(acoustic_type='vacuum',
                                                roughness=self.SURFACE_RG)]),
            bottom=Bottom.from_halfspace(BoundaryProperties(
                acoustic_type='half-space', sound_speed=1700, density=1.8,
                attenuation=0.5, roughness=0.9)))
        out = tmp_path / 'oasr.dat'
        write_oasr_input(out, env, Source(depths=50, frequencies=100.0),
                         Receiver(depths=[50.0], ranges=[1000.0]))
        records = self._layer_records(out.read_text())
        assert float(records[0][6]) == 0.0
        assert float(records[1][6]) == pytest.approx(0.9)   # seabed = ROUGH(2)


class TestOasesNegativeRoughnessRejected:
    """A negative RG changes the layer record's arity, so the writers refuse it.

    ``oaseun31.f:72`` branches on ``rough(m).lt.-1e-10``, backspaces and
    re-reads eight items with CLEN (``:75``); ``:77`` then tests
    ``clen(m).lt.-1e-10``, which is false for the CLEN==0 these decks supply,
    so ``:91`` takes the ``nvol=3`` branch and reads **nine** items from a
    record carrying eight. Every subsequent READ is shifted.

    ``BoundaryProperties`` and ``SedimentLayer`` reject it at construction;
    these cases pin the writer-side guard that catches a value assigned to a
    carrier afterwards, which no ``__post_init__`` sees.
    """

    @staticmethod
    def _env():
        from uacpy.core import Environment
        from uacpy.core.bottom import (Bottom, BoundaryProperties,
                                       SeabedColumn, SedimentLayer)
        from uacpy.core.surface import Surface
        return Environment(
            bathymetry=100.0, ssp=1500.0,
            surface=Surface([BoundaryProperties(acoustic_type='vacuum')]),
            bottom=Bottom.from_column(SeabedColumn(
                layers=[SedimentLayer(thickness=30, sound_speed=1600,
                                      density=1.8, attenuation=0.5)],
                halfspace=BoundaryProperties(
                    acoustic_type='half-space', sound_speed=1800, density=2.0,
                    attenuation=0.6))))

    @staticmethod
    def _write(env, tmp_path):
        from uacpy.core import Source, Receiver
        from uacpy.io.oases_writer import write_oast_input
        write_oast_input(tmp_path / 'neg.dat', env,
                         Source(depths=50, frequencies=100.0),
                         Receiver(depths=[50.0], ranges=[5000.0]))

    def test_negative_surface_roughness(self, tmp_path):
        env = self._env()
        env.surface.properties[0].roughness = -1.0
        with pytest.raises(ConfigurationError, match='oaseun31.f:72-93'):
            self._write(env, tmp_path)

    def test_negative_layer_roughness(self, tmp_path):
        env = self._env()
        env.bottom.columns[0].layers[0].roughness = -1.0
        with pytest.raises(ConfigurationError, match='oaseun31.f:72-93'):
            self._write(env, tmp_path)

    def test_negative_halfspace_roughness(self, tmp_path):
        env = self._env()
        env.bottom.columns[0].halfspace.roughness = -1.0
        with pytest.raises(ConfigurationError, match='oaseun31.f:72-93'):
            self._write(env, tmp_path)

    def test_zero_roughness_writes_normally(self, tmp_path):
        self._write(self._env(), tmp_path)
        assert (tmp_path / 'neg.dat').exists()


class TestRtsToPressureGoertzel:
    """``method='goertzel'`` must agree with ``method='fft'`` and with the
    analytic phasor.

    The Goertzel recurrence leaves ``s1`` aliasing ``s0`` after its final
    shift, so the closing combination has to read ``s2`` — the state one step
    back — and advance by the single sample the recurrence lags. Reading
    ``s1`` instead collapses the result to ``s0*(1 - exp(-i w dt))``, which is
    wrong in magnitude *and* phase, and no cross-mode comparison catches it
    because both modes are not exercised against each other anywhere else.
    """

    @staticmethod
    def _tone(amp, phase, n_t=512, fs=1000.0, bin_index=8):
        """One range column carrying an exactly-on-bin cosine."""
        dt = 1.0 / fs
        freq = fs * bin_index / n_t
        t = np.arange(n_t) * dt
        p = (amp * np.cos(2 * np.pi * freq * t + phase)).reshape(n_t, 1)
        return {'p': p, 'dt': dt, 'ranges': np.array([1000.0]),
                'time': t}, freq

    def test_goertzel_recovers_the_analytic_phasor(self):
        from uacpy.io.oalib_reader import rts_to_pressure
        amp, phase = 3.0, 0.700
        rts, freq = self._tone(amp, phase)
        p, _ = rts_to_pressure(rts, freq, method='goertzel')
        assert abs(p[0]) == pytest.approx(amp, rel=1e-6)
        assert np.angle(p[0]) == pytest.approx(phase, abs=1e-6)

    def test_goertzel_agrees_with_fft(self):
        from uacpy.io.oalib_reader import rts_to_pressure
        rts, freq = self._tone(2.5, -1.1)
        p_g, _ = rts_to_pressure(rts, freq, method='goertzel')
        p_f, _ = rts_to_pressure(rts, freq, method='fft')
        # The FFT path applies a Hann window, so the two agree to the window's
        # own leakage on an on-bin tone rather than to float precision.
        assert abs(p_g[0]) == pytest.approx(abs(p_f[0]), rel=1e-4)
        assert np.angle(p_g[0]) == pytest.approx(np.angle(p_f[0]), abs=1e-4)


class TestStagedBeamPatternNeedsTwoRows:
    """A degenerate ``.sbp`` passes every guard on both sides.

    ``_require_strictly_increasing`` returns early for ``size <= 1``, and
    ``misc/monotonicMod.f90:30`` pre-sets ``.TRUE.`` then returns for ``N == 1``
    (for ``N == 0`` its ``ANY`` runs over zero-size sections and is also false),
    so ``misc/beampattern.f90:56`` passes it too. The engines then index the
    table as a pair — ``Bellhop/bellhop.f90:270`` clamps ``IBP`` to
    ``NSBPPts - 1`` and reads below the bound allocated at
    ``beampattern.f90:36``, and ``KrakenField/field.f90:190-198`` brackets with
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

    def test_a_repeated_angle_is_still_refused(self, tmp_path):
        """The strictly-increasing guard must survive alongside the row-count
        one — a repeated abscissa is the division-by-zero case."""
        with pytest.raises(ConfigurationError):
            self._stage(tmp_path, "3\n-90.0 0.0\n0.0 0.0\n0.0 -3.0\n")


class TestFieldFlpProfileRangeCollapse:
    """``field.exe`` never tests ``rProf`` for monotonicity — unlike
    ``ReadRcvrRanges`` (``misc/SourceReceiverPositions.f90:163-165``), a grep for
    ``monotonic`` in ``KrakenField/field.f90`` is empty — so a collapsed pair
    reaches ``KrakenField/EvaluateADMod.f90:75``, whose
    ``(rProf(iProf+1) - rProf(iProf))`` denominator is unguarded. The 0/0 poisons
    that segment's interpolated wavenumbers and mode functions, and nothing in AT
    reports it, so neither of uacpy's fatal hooks can fire.
    """

    def _pos(self):
        return {'s': {'z': np.array([50.0])},
                'r': {'z': np.array([10.0, 30.0, 50.0]),
                      'r': np.array([1000.0, 2000.0, 3000.0])}}

    def _write(self, tmp_path, ranges_m):
        from uacpy.io.oalib_writer import write_fieldflp
        write_fieldflp(filepath=tmp_path / 'x.flp', option='RA  ',
                       pos=self._pos(), title='t', n_profiles=len(ranges_m),
                       profile_ranges_m=np.asarray(ranges_m, dtype=float))
        return (tmp_path / 'x.flp').read_text()

    def test_a_pair_that_collapses_at_1mm_is_refused(self, tmp_path):
        with pytest.raises(ConfigurationError, match='1 mm'):
            self._write(tmp_path, [0.0, 4000.0, 4000.0000001, 10000.0])

    def test_a_separated_axis_is_written(self, tmp_path):
        text = self._write(tmp_path, [0.0, 4000.0, 6000.0, 10000.0])
        assert '4.000000' in text and '6.000000' in text

    def test_the_quantum_is_the_boundary(self, tmp_path):
        """1 mm apart is exactly representable; anything closer is not."""
        from uacpy.io.oalib_writer import DECK_RANGE_QUANTUM_M
        self._write(tmp_path, [0.0, 4000.0, 4000.0 + 2 * DECK_RANGE_QUANTUM_M,
                               10000.0])
        with pytest.raises(ConfigurationError):
            self._write(tmp_path, [0.0, 4000.0,
                                   4000.0 + DECK_RANGE_QUANTUM_M / 10.0, 10000.0])


class TestOneEnvironmentOneWaterDepth:
    """Every writer must put the caller's depth on its deck.

    Nothing in the Acoustics Toolbox imposes a column width — the manual is
    explicit that *"All user input in all modules is read using list-directed
    I/O"* (``doc/index.htm``), and ``misc/ReadEnvironmentMod.f90:88`` /
    ``misc/sspMod.f90:334`` are both ``READ( ENVFile, * )``. Snapping AT
    interfaces onto a 0.1 m grid therefore modelled a different ocean from the
    one the caller built, and a different one from OASES and RAM, which print
    the depth as given.
    """

    DEPTH = 100.04

    def _env(self):
        from uacpy.core.environment import SoundSpeedProfile, BoundaryProperties
        return uacpy.Environment(
            name='x', bathymetry=self.DEPTH,
            ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0),
                                              (self.DEPTH, 1500.0)]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.7,
                                      attenuation=0.3))

    def test_deck_depth_returns_the_requested_depth(self):
        from uacpy.io.oalib_writer import deck_depth
        assert deck_depth(self.DEPTH) == pytest.approx(self.DEPTH, abs=1e-6)

    @pytest.mark.requires_binary
    def test_every_at_writer_puts_the_same_depth_on_its_deck(self, tmp_path):
        import re
        import warnings
        models = [('Kraken', uacpy.Kraken), ('Bellhop', uacpy.Bellhop),
                  ('Scooter', uacpy.Scooter)]
        src = uacpy.Source(depths=50.0, frequencies=200.0)
        rcv = uacpy.Receiver(depths=[75.0], ranges=[1000.0, 2000.0])
        seen = {}
        for name, model in models:
            work = tmp_path / name
            work.mkdir()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    model(work_dir=work, cleanup=False).compute_tl(
                        self._env(), src, rcv)
                except Exception:
                    pass                      # the deck is what matters here
            values = {round(float(m.group(0)), 6)
                      for f in work.rglob('*.env') if f.is_file()
                      for m in re.finditer(r'100\.0[0-9]*', f.read_text())}
            seen[name] = values
            assert values, f"{name} wrote no water depth"
        pooled = sorted(set().union(*seen.values()))
        assert len(pooled) == 1, \
            f"writers disagree on the water depth: {seen}"
        assert pooled[0] == pytest.approx(self.DEPTH, abs=1e-6), \
            f"the decks carry {pooled[0]}, not the requested {self.DEPTH}"

    @pytest.mark.requires_binary
    def test_a_six_centimetre_depth_change_reaches_the_solver(self):
        """Quantisation made ``Scooter`` answer identically for 100.04 m and
        100.1 m, because both decks said 100.1 — the requested depth was
        discarded. The two must now differ."""
        import warnings
        from uacpy.core.environment import SoundSpeedProfile, BoundaryProperties

        def tl(depth):
            env = uacpy.Environment(
                name='x', bathymetry=depth,
                ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0),
                                                  (depth, 1500.0)]),
                bottom=BoundaryProperties(acoustic_type='half-space',
                                          sound_speed=1600.0, density=1.7,
                                          attenuation=0.3))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return np.asarray(uacpy.Scooter().compute_tl(
                    env, uacpy.Source(depths=50.0, frequencies=200.0),
                    uacpy.Receiver(depths=[75.0],
                                   ranges=np.linspace(1000.0, 10000.0, 91))
                ).tl).ravel()

        assert np.nanmax(np.abs(tl(100.04) - tl(100.1))) > 1.0
