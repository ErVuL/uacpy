"""
Tests for I/O functions
"""

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

        r_km = np.array([0.0, 5.0, 10.0, 20.0])
        # 5 depths x 4 ranges
        c = np.array([
            [1500.0, 1502.0, 1504.0, 1505.0],
            [1495.0, 1497.0, 1499.0, 1500.5],
            [1490.0, 1492.0, 1494.0, 1495.5],
            [1488.0, 1489.5, 1491.0, 1492.5],
            [1487.0, 1488.0, 1489.0, 1490.0],
        ])
        out = tmp_path / "rt.ssp"
        write_ssp(out, r_km, c)
        result = read_ssp_2d(out)
        assert result['n_prof'] == 4
        assert result['c_mat'].shape == (5, 4)
        np.testing.assert_allclose(result['r_prof'], r_km, atol=1e-3)
        # write_ssp truncates speeds to one decimal, so compare loosely.
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
        """NRro must equal NRz and the Rro line must be the AT
        ``0.0 /`` sentinel idiom (single value + slash terminator)."""
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
        (SourceReceiverPositions.f90:219-221), and the following ``Sort``
        moves it to Rro(1) — giving the *shallowest* receiver a -999.9 m
        range offset, which EvaluateMod applies as ``1/sqrt(r + ro)``.
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
    """The Collins PE grid maps stored index i to depth (i-1)*dz.

    Regression for the depth off-by-one: ramsurf1.5 writes from grid index
    ``ndz`` (its first stored sample is the z=0 surface node), rams0.5 from
    ``1+ndz`` (it skips z=0, first sample at z=ndz*dz). Both must land on the
    true (i-1)*dz grid.
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
        bad = tmp_path / "garbage.shd"
        bad.write_bytes(b"\x00" * 12)            # too short for a valid header
        with pytest.raises(Exception):           # FileFormatError or EOF-derived
            read_shd_bin(str(bad))


class TestOASRFrequencyOverride:
    """Audit H3: an explicit ``frequencies=`` override (passed by OASR.run as
    freq_min/freq_max/n_frequencies) must win over a multi-element
    ``source.frequencies``; previously the writer honoured the override only
    for single-frequency sources, silently dropping it otherwise."""

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
        # Multi-element source that previously hijacked the sweep.
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

    ``sspMod.f90:334`` reads each SSP line as
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
    (ReadEnvironmentMod.f90:81-88), so the water column's line is sigma(1) —
    the *sea surface*. The seabed is sigma(NMedia+1), written on the bottom
    halfspace line (:121). A model-level ``roughness=`` kwarg documented as
    "bottom roughness" wrote sigma(1), i.e. the surface.
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
    def test_models_no_longer_take_a_roughness_kwarg(self, model_name):
        import uacpy
        with pytest.raises(TypeError):
            getattr(uacpy, model_name)(roughness=2.0)


class TestSSPLinePinsWaterProperties:
    """AT's ``/`` list-directed terminator leaves unassigned items at their
    previous value, and ``TopBot`` (ReadEnvironmentMod.f90:285) reads the top
    half-space into the very module variables (sspMod.f90:14) that a short
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
    oasnun22.f:291) are what OASN samples the field on."""

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
        assert '  150.0 1600.000000' in shallow
        assert '  155.0 1600.000000' in shallow
        assert '  155.0 1650.000000' in shallow
        assert '  161.0 1650.000000' in shallow, \
            "the 6 m sediment layer was stretched to the global max depth"
        # The pad that absorbs the stretch carries halfspace properties.
        assert '  161.0 1800.000000' in shallow

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
        halfspace_depths = [line.split()[0] for line in lines
                            if line.startswith('  200.30')]
        assert len(halfspace_depths) == 3


class TestSSPRangeAxisPrecision:
    """``write_ssp``'s range axis feeds Bellhop's Quad segment search, which
    needs ``SSP%Seg%r`` strictly increasing (sspMod.f90)."""

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
    caller switching between them must not have to remap keys."""

    def test_ascii_reader_uses_the_binary_reader_keys(self, tmp_path):
        from uacpy.io.oalib_reader import read_shd_asc
        p = tmp_path / 'tiny.shd.asc'
        p.write_text(
            "title\nrectilin\n1 1 1 1 2\n100.0 0.0\n100.0\n0.0\n50.0\n"
            "10.0\n0.0\n1000.0\n1.0\n0.5\n2.0\n0.25\n"
        )
        out = read_shd_asc(p)
        assert set(out) >= {'title', 'PlotType', 'freqVec', 'Pos', 'pressure'}
        assert 'r' in out['Pos'] and 'z' in out['Pos']['r']


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
    ``NWVNOP`` (oasnun22.f:312, :359). A negative count makes that total
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
