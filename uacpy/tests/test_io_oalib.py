"""The Acoustics-Toolbox decks uacpy writes and the files it reads back
(``uacpy.io.oalib_writer`` and ``uacpy.io.oalib_reader``).

Two kinds of contract live here: the readers checked against hand-written
files, and the record layout of the decks uacpy emits, checked by running the
vendored Acoustics-Toolbox binaries on them (``@pytest.mark.requires_binary``).
Both pin what the Fortran reads, so the authority for every expectation is
the vendored source, cited at the assertion.
"""

import io
import numpy as np
import pytest
import re
import uacpy
from pathlib import Path
from uacpy.core import BoundaryProperties
from uacpy.core import Environment
from uacpy.core import Source
from uacpy.core.bathymetry import Bathymetry
from uacpy.core.bottom import Bottom
from uacpy.core.bottom import SeabedColumn
from uacpy.core.boundary import SedimentLayer
from uacpy.core.constants import DEFAULT_WATER_DENSITY_G_CM3
from uacpy.core.environment import SoundSpeedProfile
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.exceptions import FileFormatError
from uacpy.io.oalib_writer import write_bounce_input_file
from uacpy.io.oalib_writer import write_ssp_section
from uacpy.io.oases_writer import _emit_water_layers
from uacpy.tests.conftest import _AT_REF_DIR
from uacpy.tests.conftest import water_density_env
from uacpy.models.ram import mpirams as ram_mpirams, collins as ram_collins
from uacpy.tests.conftest import make_pekeris
from uacpy.tests.conftest import (
    at_deck_water_rows, measured_absorption_table, two_layer_absorption,
    two_layer_dB_per_m,
)


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
        assert r.ranges.shape == (30,)
        # First/last ranges on disk are -50 km and 10 km; the reader
        # returns metres (uacpy is SI-internal, km only on disk).
        assert r.ranges[0] == -50_000.0
        assert r.ranges[-1] == 10_000.0
        # File has 2 depth rows.
        assert r.sound_speed.shape == (2, 30)
        # Spot-check one entry against the file.
        assert r.sound_speed[0, 2] == pytest.approx(1548.52)
        # The carrier starts at range 0, so the table's -50 km profile is
        # refused by name rather than shifted.
        with pytest.raises(ConfigurationError, match='MunkB_geo_rot.ssp'):
            r.to_ssp([0.0, 5000.0])

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
        assert result.sound_speed.shape == (5, 4)
        # The range axis goes to disk as km at ``%.6f`` and comes back in
        # metres, so the quantum on this axis is 1 mm.
        np.testing.assert_allclose(result.ranges, ranges_m, atol=1e-3)
        # Speeds go out at ``%8.4f``; the format quantises at 1e-4 m/s, well
        # inside this bound.
        np.testing.assert_allclose(result.sound_speed, c, atol=0.5)
        depths = np.array([0.0, 50.0, 100.0, 200.0, 500.0])
        ssp = result.to_ssp(depths)
        np.testing.assert_array_equal(ssp.depths, depths)
        np.testing.assert_array_equal(ssp.ranges, result.ranges)
        np.testing.assert_array_equal(ssp.sound_speed, result.sound_speed)

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

    def test_write_ssp_names_its_profiles_sound_speed(self, tmp_path):
        from uacpy.io.oalib_writer import write_ssp
        out = tmp_path / "named.ssp"
        write_ssp(out, ranges=[0.0, 1000.0],
                  sound_speed=np.full((2, 2), 1500.0))
        assert out.read_text().splitlines()[0] == '2'
        with pytest.raises(TypeError, match="unexpected keyword argument 'c'"):
            write_ssp(out, [0.0, 1000.0], c=np.full((2, 2), 1500.0))

    def test_write_ssp_takes_nested_lists(self, tmp_path):
        from uacpy.io.oalib_writer import write_ssp
        out = tmp_path / "list.ssp"
        write_ssp(out, [0.0, 1000.0], [[1500.0, 1500.0], [1510.0, 1510.0]])
        assert out.read_text().splitlines()[1].split() == ['0.000000',
                                                           '1.000000']

    @pytest.mark.parametrize('ranges', [[0.0, 1000.0, 1000.0004],
                                        [0.0, 2000.0, 1000.0],
                                        [0.0, np.nan, 2000.0]])
    def test_write_ssp_refuses_a_range_axis_the_deck_cannot_hold(
            self, tmp_path, ranges):
        """Sub-millimetre, decreasing and NaN profile ranges all fail the
        strictly-increasing km column Bellhop's Quad search needs."""
        from uacpy.io.oalib_writer import write_ssp
        out = tmp_path / "bad.ssp"
        with pytest.raises(ConfigurationError, match='increase strictly'):
            write_ssp(out, np.array(ranges), np.full((2, 3), 1500.0))
        assert not out.exists()

    def test_write_ssp_keeps_ranges_one_millimetre_apart(self, tmp_path):
        from uacpy.io.oalib_writer import write_ssp
        out = tmp_path / "mm.ssp"
        write_ssp(out, np.array([0.0, 1000.0, 1000.001]),
                  np.full((2, 3), 1500.0))
        assert out.read_text().splitlines()[1].split()[-1] == '1.000001'


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

    def test_repeat_count_records_read_as_repeated_values(self, tmp_path):
        """ArrMod.f90:99-118 writes the .arr list-directed, and an
        ifort-built engine compresses consecutive equal values to ``r*c``
        (``3*0.0`` = three zeros); the token stream must expand it."""
        from uacpy.io.oalib_reader import read_arr_file
        lines = self._expected_arr_lines()
        # phase, delay_r, delay_i all zero, as ifort would emit them.
        lines[6] = "0.5 3*0.0 -5.0 5.0 0 1"
        path = tmp_path / "repeat.arr"
        self._write_arr(path, lines)
        result = read_arr_file(path)
        a0 = result.by_receiver[0][0][0]
        assert a0['amplitudes'][0] == pytest.approx(0.5)
        assert a0['phases'][0] == 0.0
        assert a0['delays'][0] == 0.0
        assert a0['delays_imag'][0] == 0.0
        assert a0['source_angles'][0] == pytest.approx(-5.0)
        assert a0['n_bot_bounces'][0] == 1

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
            n_modes=999,
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
            title='unit-test', n_modes=999,
        )
        rro = [ln for ln in out.read_text().splitlines() if 'Rro(' in ln][0]
        values = rro.split('!')[0].replace('/', ' ').split()
        assert len(values) == n_rz, (
            f"NRz={n_rz} needs {n_rz} explicit Rro values (SubTab will not "
            f"replicate below 3); got {values}")
        assert all(float(v) == 0.0 for v in values)


class TestShdNoDataCells:
    """Exact-zero SHD pressure cells — grid points the engine never wrote
    (Bellhop no-ray cells, an empty KRAKEN modal sum) — surface as NaN,
    uacpy's no-data convention."""

    @staticmethod
    def _write_bin(path, rows):
        """Emit a tiny little-endian direct-access ``.shd`` (one frequency,
        one bearing, one source, two receiver depths, two ranges) in the
        record layout of ``misc/RWSHDFile.f90:100-114``, whose
        ``LRecl = MAX(41, 2*Nfreq, …)`` is counted in 4-byte words."""
        recl = 41
        rec_bytes = 4 * recl
        records = [
            np.array([recl], '<i4').tobytes() + b'title'.ljust(80),
            b'rectilin'.ljust(10),
            (np.array([1, 1, 1, 1, 1, 2, 2], '<i4').tobytes()
             + np.array([100.0, 0.0], '<f8').tobytes()),
            np.array([100.0], '<f8').tobytes(),          # freqVec
            np.array([0.0], '<f8').tobytes(),            # theta
            np.array([0.0], '<f8').tobytes(),            # Sx
            np.array([0.0], '<f8').tobytes(),            # Sy
            np.array([50.0], '<f4').tobytes(),           # Sz (REAL(KIND=4))
            np.array([10.0, 20.0], '<f4').tobytes(),     # Rz
            np.array([100.0, 200.0], '<f8').tobytes(),   # Rr
        ] + [np.array(row, '<f4').tobytes() for row in rows]
        path.write_bytes(b''.join(r.ljust(rec_bytes, b'\x00') for r in records))
        return path

    def test_read_shd_bin_returns_the_file_as_a_record(self, tmp_path):
        """Every header record lands on its ``ShdFile`` field."""
        from uacpy.io import ShdFile
        from uacpy.io.oalib_reader import read_shd_bin
        rows = [[1.0, 0.0, 0.5, 0.25], [1.0, 0.0, 2.0, -1.0]]
        shd = read_shd_bin(str(self._write_bin(tmp_path / 'h.shd', rows)))
        assert type(shd) is ShdFile
        assert shd.title == 'title'
        assert shd.plot_type == 'rectilin  '
        assert shd.frequencies.tolist() == [100.0]
        assert shd.source_frequency == 100.0
        assert shd.stabilizing_attenuation == 0.0
        assert shd.bearings.tolist() == [0.0]
        assert shd.source_x.tolist() == [0.0]
        assert shd.source_y.tolist() == [0.0]
        assert shd.source_depths.tolist() == [50.0]
        assert shd.receiver_depths.tolist() == [10.0, 20.0]
        assert shd.receiver_ranges.tolist() == [100.0, 200.0]
        assert shd.pressure.shape == (1, 1, 2, 2)
        assert shd.pressure_frequency == 100.0

    def test_read_shd_bin_zero_pressure_is_nan(self, tmp_path):
        from uacpy.io.oalib_reader import read_shd_bin
        # The (depth 0, range 0) cell is an exact complex zero = no data.
        rows = [
            [0.0, 0.0, 0.5, 0.25],     # depth 10 m: no-data, 0.5+0.25j
            [1.0, 0.0, 2.0, -1.0],     # depth 20 m: 1+0j, 2-1j
        ]
        pr = read_shd_bin(
            str(self._write_bin(tmp_path / 't.shd', rows))).pressure[0, 0]
        assert pr.shape == (2, 2)
        assert np.isnan(pr[0, 0])
        assert pr[0, 1] == pytest.approx(0.5 + 0.25j)
        assert pr[1, 0] == pytest.approx(1.0 + 0.0j)
        assert pr[1, 1] == pytest.approx(2.0 - 1.0j)


class TestReaderCorruptFileRaises:
    """The io readers' failure path: a truncated / garbage binary must raise
    the typed :class:`FileFormatError`, not a bare struct/EOF error or a
    self-contradictory ModelExecutionError(return_code=0)."""

    def test_corrupt_mode_file_raises_fileformaterror(self, tmp_path):
        from uacpy.io.modes_reader import read_modes
        from uacpy.core.exceptions import FileFormatError
        bad = tmp_path / "garbage.mod"
        bad.write_bytes(b"\x00\x01\x02not a real mode file\xff\xfe" * 4)
        with pytest.raises(FileFormatError, match='Invalid mode file'):
            read_modes(str(bad))

    def test_corrupt_shd_file_raises(self, tmp_path):
        from uacpy.io.oalib_reader import read_shd_bin
        from uacpy.core.exceptions import FileFormatError
        bad = tmp_path / "garbage.shd"
        bad.write_bytes(b"\x00" * 12)            # too short for a valid header
        with pytest.raises(FileFormatError, match='cannot resolve byte order'):
            read_shd_bin(str(bad))


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
            write_ssp_section(f, self._env(0.5), bottom_depth=200.0,
                              ssp_topopt='C')
        rows = [ln for ln in out.read_text().splitlines() if '/' in ln][1:]
        assert rows, "no SSP sample rows written"
        for ln in rows:
            cols = ln.replace('/', ' ').split()
            assert len(cols) >= 5, f"need z cp cs rho alphaI; got {cols}"
            assert float(cols[2]) == 0.0, f"cs must be 0 for water; got {cols[2]}"
            assert float(cols[4]) == pytest.approx(0.5), (
                f"absorption must be in alphaI (col 5); got {cols}")

    def test_zero_absorption_pins_all_six_columns(self, tmp_path):
        """The short ``z c /`` form is unsafe even at zero absorption: AT's
        ``/`` terminator leaves the remaining items at their previous value,
        and ``TopBot`` (ReadEnvironmentMod.f90:285) has already loaded the top
        half-space into those module variables. All six columns are pinned."""
        import uacpy
        from uacpy.io.oalib_writer import write_ssp_section
        env = uacpy.Environment(name='p', bathymetry=200.0, ssp=1500.0)
        out = tmp_path / 'ssp2.txt'
        with open(out, 'w') as f:
            write_ssp_section(f, env, bottom_depth=200.0, ssp_topopt='C')
        rows = [ln for ln in out.read_text().splitlines() if '/' in ln][1:]
        assert rows, "no SSP sample rows written"
        for ln in rows:
            cols = ln.replace('/', ' ').split()
            assert len(cols) == 6, f"expected all six AT columns; got {cols}"
            assert float(cols[2]) == 0.0 and float(cols[4]) == 0.0


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
            write_ssp_section(f, env, bottom_depth=200.0, ssp_topopt='C')
        return float(out.read_text().splitlines()[0].split()[1])

    def test_surface_carrier_drives_the_water_medium_sigma(self, tmp_path):
        assert self._mesh_sigma(tmp_path, self._env(surf=2.0)) == pytest.approx(2.0)

    def test_bottom_carrier_does_not_touch_the_water_medium_sigma(self, tmp_path):
        assert self._mesh_sigma(tmp_path, self._env(bot=2.0)) == pytest.approx(0.0)

    @pytest.mark.parametrize('model_name', ['Kraken', 'Scooter', 'SPARC'])
    def test_models_take_no_roughness_kwarg(self, model_name):
        import uacpy
        with pytest.raises(TypeError,
                           match="unexpected keyword argument 'roughness'"):
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
        out = tmp_path / f'{name}.env'
        write_kraken_env_file(
            out, env,
            source=Source(depths=25.0, frequencies=100.0),
            receiver=Receiver(depths=[50.0], ranges=[1000.0]),
            interp_ssp='linear',
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
            assert float(row[3]) == pytest.approx(DEFAULT_WATER_DENSITY_G_CM3), (
                f"water density must be the Environment's (default "
                f"{DEFAULT_WATER_DENSITY_G_CM3} g/cm³): {row}")
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
            Receiver(depths=[50.0], ranges=np.linspace(500.0, 5000.0, 10))).dB)

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
    """A non-geoacoustic boundary (vacuum, rigid, or a reflection table) has
    no physical sound speed, but BoundaryProperties still carries the
    constructor default. Capping c_high on that placeholder truncates the
    mode spectrum (3 of 7 modes, 10.6 dB, on a 100 m rigid guide at 50 Hz;
    a 'file' bottom's 1600 m/s placeholder capped cHigh at 1680 m/s)."""

    @staticmethod
    def _env(bottom):
        import uacpy
        return uacpy.Environment(name='b', bathymetry=100.0, ssp=1500.0,
                                 bottom=bottom)

    @pytest.mark.parametrize('kind', ['rigid', 'vacuum', 'file', 'precalc'])
    def test_parameter_free_bottom_is_unbounded_above(self, kind):
        from uacpy.core import BoundaryProperties
        from uacpy.models._window import DEFAULT_C_MAX_UNBOUNDED
        from uacpy.models._window import resolve_phase_speed_bounds
        _, c_high = resolve_phase_speed_bounds(
            self._env(BoundaryProperties(acoustic_type=kind)))
        assert c_high == DEFAULT_C_MAX_UNBOUNDED

    def test_file_bottom_deck_carries_the_unbounded_c_high(self, tmp_path):
        """The resolved bound must reach the deck itself: a 'file' bottom's
        cLow/cHigh line carries the unbounded value, not a cap derived from
        the 1600 m/s placeholder."""
        import uacpy
        from uacpy.core import BoundaryProperties
        from uacpy.models._window import DEFAULT_C_MAX_UNBOUNDED
        from uacpy.io.oalib_writer import write_kraken_env_file
        from uacpy.models._window import resolve_phase_speed_bounds
        brc = tmp_path / 'table.brc'
        brc.write_text("2\n0.0 1.0 180.0\n90.0 0.5 175.0\n")
        env = self._env(BoundaryProperties(acoustic_type='file',
                                           reflection_file=brc))
        c_low, c_high = resolve_phase_speed_bounds(env)
        out = tmp_path / 'case.env'
        write_kraken_env_file(
            out, env, uacpy.Source(depths=50.0, frequencies=100.0),
            uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
            interp_ssp='linear', frequencies=None,
            n_mesh=0, rmax_m=10000.0, c_low=c_low, c_high=c_high)
        lines = out.read_text().splitlines()
        bounds_line = next(
            ln for ln in lines
            if len(ln.split()) == 2
            and ln.split()[1] == f"{DEFAULT_C_MAX_UNBOUNDED:.1f}")
        assert float(bounds_line.split()[1]) == DEFAULT_C_MAX_UNBOUNDED

    def test_penetrable_bottom_caps_on_the_halfspace_speed(self):
        from uacpy.core import BoundaryProperties
        from uacpy.models._window import C_HIGH_FACTOR
        from uacpy.models._window import resolve_phase_speed_bounds
        env = self._env(BoundaryProperties(
            acoustic_type='half-space', sound_speed=1800.0,
            density=1.8, attenuation=0.5))
        _, c_high = resolve_phase_speed_bounds(env)
        assert c_high == pytest.approx(1800.0 * C_HIGH_FACTOR)

    def test_explicit_c_high_always_wins(self):
        from uacpy.core import BoundaryProperties
        from uacpy.models._window import resolve_phase_speed_bounds
        _, c_high = resolve_phase_speed_bounds(
            self._env(BoundaryProperties(acoustic_type='rigid')), c_high=1700.0)
        assert c_high == 1700.0


class TestMultiProfileEnvKeepsLayerThickness:
    """``write_multi_profile_env`` holds every profile to one total media depth.
    That stretch must land on a transparent halfspace-property pad, never on a
    real ``SedimentLayer`` whose thickness is physical."""

    @staticmethod
    def _segments():
        from uacpy.core.boundary import SedimentLayer, BoundaryProperties
        from uacpy.core.bottom import SeabedColumn
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
                                n_mesh=500, rmax_m=20000.0, c_low=0.0,
                                c_high=2000.0)
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
                                n_mesh=500, rmax_m=20000.0, c_low=0.0,
                                c_high=2000.0)
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
        r_back = np.asarray(read_ssp_2d(str(path)).ranges)
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

    def test_bad_coupling_column_raises_for_multi_profile(self, tmp_path):
        """field.exe gates Opt(2:2) with ERROUT for NProf > 1
        (field.f90:125-136): only 'C' (coupled) and 'A' (adiabatic) run."""
        from uacpy.io.oalib_writer import write_fieldflp
        with pytest.raises(ConfigurationError, match='option position 2'):
            write_fieldflp(
                tmp_path / 'bad2.flp', 'R  C', self._pos(),
                n_profiles=2, profile_ranges=np.array([0.0, 5000.0]))

    def test_blank_coupling_column_is_fine_for_one_profile(self, tmp_path):
        """field.f90 reaches its coupling SELECT CASE only when NProf > 1, so
        a blank column 2 is legal in a single-profile deck."""
        from uacpy.io.oalib_writer import write_fieldflp
        write_fieldflp(tmp_path / 'one.flp', 'R  C', self._pos())
        assert (tmp_path / 'one.flp').exists()

    def test_explicit_suffix_is_kept(self, tmp_path):
        """A caller-supplied suffix is written as given; '.flp' is appended
        only to a bare root (the read_flp resolution convention)."""
        from uacpy.io.oalib_writer import write_fieldflp
        write_fieldflp(tmp_path / 'case.v2', 'RC C', self._pos())
        assert (tmp_path / 'case.v2').exists()
        assert not (tmp_path / 'case.flp').exists()
        write_fieldflp(tmp_path / 'bare', 'RC C', self._pos())
        assert (tmp_path / 'bare.flp').exists()


class TestBinaryArrivalsRejected:
    """``read_arr_file`` only parses the ASCII ``.arr``; the binary layout
    (RunType 'a') is file content this reader cannot parse, so it raises
    the typed :class:`FileFormatError` like every other unreadable file."""

    def test_binary_arrivals_raises_fileformaterror(self, tmp_path):
        from uacpy.io.oalib_reader import read_arr_file
        p = tmp_path / 'run.arr'
        p.write_bytes(b'\x04\x00\x00\x00' + b'\x00' * 32)
        with pytest.raises(FileFormatError, match="RunType 'A'"):
            read_arr_file(p)


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
        layer = SedimentLayer(thickness=self.THICKNESS, sound_speed=self.CP,
                              density=self.RHO, attenuation=self.ALPHA_P)
        # Distinct, non-zero shear values: swapping cs with alpha_s between two
        # zeros would be a no-op and the guard could never fail.
        hs = BoundaryProperties(acoustic_type='half-space', sound_speed=1800.0,
                                density=2.0, attenuation=0.5,
                                shear_speed=600.0, shear_attenuation=0.25)
        env = uacpy.Environment(
            bathymetry=100.0,
            ssp=SoundSpeedProfile(depths=[0.0, 100.0], sound_speed=[1500.0, 1500.0]),
            bottom=Bottom(columns=[SeabedColumn(layers=[layer], halfspace=hs)]),
        )
        out = tmp_path / 'layers.env'
        write_kraken_env_file(
            str(out), env,
            uacpy.Source(depths=25.0, frequencies=100.0),
            uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
            interp_ssp='linear', frequencies=[100.0],
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


class TestATEnvWriterLayered:
    """Test AT env writer with layered bottom."""

    def test_nmedia_with_layers(self):
        """AT env writer should set NMEDIA > 1 for layered bottom."""
        from uacpy.io.oalib_writer import write_header
        from uacpy.core.boundary import BoundaryType
        from uacpy.core.environment import SeabedColumn, SedimentLayer

        lb = SeabedColumn(
            layers=[
                SedimentLayer(thickness=10, sound_speed=1550, density=1.3, attenuation=0.5),
                SedimentLayer(thickness=50, sound_speed=1650, density=1.7, attenuation=0.3),
            ],
            halfspace=BoundaryProperties(acoustic_type='half-space', sound_speed=1800, density=2.0, attenuation=0.1)
        )
        env = uacpy.Environment(name='test', bathymetry=200, ssp=1500, bottom=lb)
        source = uacpy.Source(frequencies=100, depths=25)

        buf = io.StringIO()
        write_header(buf, env, source,
                     ssp_topopt='C',
                     surface_type=BoundaryType.VACUUM)
        content = buf.getvalue()

        # Should have NMEDIA = 3 (1 water + 2 sediment layers)
        lines = content.strip().split('\n')
        nmedia_line = lines[2]  # Third line is NMEDIA
        assert nmedia_line.strip() == '3'

    def test_layer_sections_written(self):
        """Layer sections should be written between SSP and bottom."""
        from uacpy.io.oalib_writer import write_layer_sections
        from uacpy.core.environment import SeabedColumn, SedimentLayer

        lb = SeabedColumn(
            layers=[SedimentLayer(thickness=10, sound_speed=1550, density=1.3, attenuation=0.5)],
            halfspace=BoundaryProperties(acoustic_type='half-space', sound_speed=1800, density=2.0)
        )
        env = uacpy.Environment(name='test', bathymetry=100, ssp=1500, bottom=lb)

        buf = io.StringIO()
        depth_after = write_layer_sections(buf, env, 100)
        content = buf.getvalue()

        assert depth_after == 110  # 100 + 10m layer
        assert '1550' in content  # Layer sound speed present

    def test_halfspace_depth_below_layers(self):
        """Halfspace depth should be below all layers."""
        from uacpy.io.oalib_writer import write_bottom_section
        from uacpy.core.environment import SeabedColumn, SedimentLayer

        lb = SeabedColumn(
            layers=[
                SedimentLayer(thickness=10, sound_speed=1550, density=1.3),
                SedimentLayer(thickness=50, sound_speed=1650, density=1.7),
            ],
            halfspace=BoundaryProperties(acoustic_type='half-space', sound_speed=1800, density=2.0, attenuation=0.1)
        )
        env = uacpy.Environment(name='test', bathymetry=200, ssp=1500, bottom=lb)

        buf = io.StringIO()
        write_bottom_section(buf, env)
        content = buf.getvalue()

        # Halfspace should be at 260m (200 + 10 + 50)
        assert '260.00' in content


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
    hs = BoundaryProperties(acoustic_type='half-space', sound_speed=1800.0,
                            density=2.0, attenuation=0.5)
    bot = Bottom(columns=[SeabedColumn(
        layers=[SedimentLayer(thickness=3.0, sound_speed=1600.0,
                              density=1.7, attenuation=0.3)], halfspace=hs)])
    segments = []
    for i, wd in enumerate([100.0, 140.0, 180.0, 200.0]):
        segments.append((i * 2.0, uacpy.Environment(
            bathymetry=Bathymetry(ranges=[0.0, 6000.0], depths=[wd, wd]),
            ssp=SoundSpeedProfile(depths=[0.0, wd], sound_speed=[1500.0, 1480.0]),
            bottom=bot)))
    out = tmp_path / 'multi.env'
    write_multi_profile_env(
        str(out), segments,
        uacpy.Source(depths=30.0, frequencies=100.0),
        uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
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
# volume-attenuation rows *inside* ReadTopOpt (biological :220-235),
# and only then the top half-space row (:75 -> TopBot :285). ``RefCoef.f90``
# opens ``<root>.trc`` for a top ``'F'`` (:64-76) and ``<root>.irc`` for a
# bottom ``'P'`` (:92-96).
# ---------------------------------------------------------------------------

_AT_BIN = Path(uacpy.__file__).parent / 'bin' / 'oalib'


_ICE = dict(acoustic_type='half-space', sound_speed=3500.0, shear_speed=1800.0,
            density=0.9, attenuation=1.0, shear_attenuation=2.0)


def _fg():
    from uacpy.core.absorption import FrancoisGarrison
    return FrancoisGarrison(temperature=10.0, salinity=35.0, pH=8.0)


def _bio():
    from uacpy.core.absorption import Biological
    return Biological(layers=[(10.0, 20.0, 400.0, 5.0, 0.1)])


def _top_block_env(absorption=None, surface=None, bathymetry=100.0,
                   bottom=None):
    from uacpy.core import BoundaryProperties
    if surface is None:
        surface = BoundaryProperties(**_ICE)
    if bottom is None:
        bottom = BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=1700.0, density=1.5,
                                    attenuation=0.5)
    return uacpy.Environment(
        name='deck', bathymetry=bathymetry,
        ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.0, 1500.0)]),
        surface=surface, bottom=bottom, absorption=absorption)


def _src_rcv():
    return (uacpy.Source(depths=25.0, frequencies=200.0),
            uacpy.Receiver(depths=[50.0], ranges=[1000.0]))


def _write_kraken(path, env, **overrides):
    from uacpy.io.oalib_writer import write_kraken_env_file
    src, rcv = _src_rcv()
    kwargs = dict(interp_ssp='linear', frequencies=None,
                  n_mesh=0, rmax_m=5000.0, c_low=1400.0, c_high=2000.0)
    kwargs.update(overrides)
    write_kraken_env_file(path, env, src, rcv, **kwargs)
    return path


def _write_scooter(path, env, **overrides):
    from uacpy.io.oalib_writer import write_scooter_env_file
    src, rcv = _src_rcv()
    kwargs = dict(interp_ssp='linear', frequencies=None,
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
    half-space row as its layer count and TopBot then reads a bio row as the
    half-space."""

    @staticmethod
    def _assert_bio_then_halfspace(text, cp=3500.0):
        lines = text.splitlines()
        assert lines[3].startswith("'C"), f"line 4 is not TopOpt: {lines[3]}"
        assert lines[3][2] == 'A', f"TopOpt(2) is not 'A': {lines[3]}"
        assert lines[3][4] == 'B', f"TopOpt(4) is not 'B': {lines[3]}"
        assert lines[4].strip() == '1', (
            f"the bio layer count must follow TopOpt directly; got "
            f"{lines[4]!r}")
        assert _floats(lines[5]) == [10.0, 20.0, 400.0, 5.0, 0.1]
        hs = _floats(lines[6])
        assert lines[6].rstrip().endswith('/') and len(hs) == 6, (
            f"the top half-space row must follow the bio rows; got "
            f"{lines[6]!r}")
        assert hs[0] == 0.0 and hs[1] == cp, (
            f"top half-space row carries the wrong medium: {lines[6]!r}")

    def test_kraken_deck_orders_topopt_absorption_halfspace(self, tmp_path):
        text = _write_kraken(tmp_path / 'kr.env',
                             _top_block_env(_bio())).read_text()
        self._assert_bio_then_halfspace(text)

    def test_scooter_deck_orders_topopt_absorption_halfspace(self, tmp_path):
        text = _write_scooter(tmp_path / 'sc.env',
                              _top_block_env(_bio())).read_text()
        self._assert_bio_then_halfspace(text)

    def test_a_broadband_francois_garrison_row_precedes_the_halfspace(
            self, tmp_path):
        """Emitted the other way round, ReadTopOpt eats the half-space row as
        its F-G parameters and TopBot reads the F-G row as the half-space —
        the run dies in AttenMod : CRCI."""
        lines = _write_kraken(tmp_path / 'kr.env', _top_block_env(_fg()),
                              frequencies=np.array([150.0, 200.0, 250.0])
                              ).read_text().splitlines()
        assert lines[3][4] == 'F', f"TopOpt(4) is not 'F': {lines[3]}"
        assert _floats(lines[4]) == [10.0, 35.0, 8.0, 50.0]
        assert _floats(lines[5])[:2] == [0.0, 3500.0], (
            f"the top half-space row must follow the F-G row: {lines[5]!r}")

    @pytest.mark.requires_binary
    def test_krakenc_reads_the_broadband_francois_garrison_deck(
            self, tmp_path):
        _write_kraken(tmp_path / 'kr.env', _top_block_env(_fg()),
                      frequencies=np.array([150.0, 200.0, 250.0]))
        text = _run_at('krakenc.exe', 'kr', tmp_path)
        assert 'Francois-Garrison' in text and '3500.00' in text

    def test_a_francois_garrison_deck_has_no_row_between_topopt_and_halfspace(
            self, tmp_path):
        text = _write_kraken(tmp_path / 'kr.env',
                             _top_block_env(_fg())).read_text()
        lines = text.splitlines()
        assert lines[3][4] == ' ', f"TopOpt(4) is not blank: {lines[3]}"
        assert _floats(lines[4])[:2] == [0.0, 3500.0], (
            f"the top half-space row must follow TopOpt; got {lines[4]!r}")

    def test_bounce_deck_orders_topopt_absorption_halfspace(self, tmp_path):
        from uacpy.core import BoundaryProperties
        from uacpy.io.oalib_writer import write_bounce_input_file
        src, _ = _src_rcv()
        out = tmp_path / 'bo.env'
        write_bounce_input_file(
            out, _top_block_env(_bio(),
                                surface=BoundaryProperties(acoustic_type='vacuum')),
            src, interp_ssp='linear', n_mesh=0, c_low=1400.0,
            c_high=2000.0, rmax_m=5000.0)
        # BOUNCE's top half-space is the water at the seafloor, not the ice.
        self._assert_bio_then_halfspace(out.read_text(), cp=1500.0)

    def test_multi_profile_deck_orders_every_profile(self, tmp_path):
        from uacpy.io.oalib_writer import write_multi_profile_env
        src, rcv = _src_rcv()
        out = tmp_path / 'multi.env'
        segments = [(0.0, _top_block_env(_bio())),
                    (5000.0, _top_block_env(_bio()))]
        write_multi_profile_env(out, segments, src, rcv, n_mesh=500,
                                rmax_m=20000.0, c_low=0.0, c_high=2000.0)
        lines = out.read_text().splitlines()
        starts = [i for i, ln in enumerate(lines) if ln.strip() == "'deck'"]
        assert len(starts) == 2
        for i in starts:
            assert _floats(lines[i + 5]) == [10.0, 20.0, 400.0, 5.0, 0.1], (
                f"profile at line {i} lost the bio row: {lines[i + 5]!r}")
            assert _floats(lines[i + 6])[1] == 3500.0, (
                f"profile at line {i} lost the top half-space row: "
                f"{lines[i + 6]!r}")

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
    def _write(path, env):
        from uacpy.io.oalib_writer import write_sparc_env_file
        src, rcv = _src_rcv()
        write_sparc_env_file(
            path, env, src, rcv, interp_ssp='linear',
            output_mode='R', n_mesh=0, rmax_m=5000.0,
            c_low=1400.0, c_high=2000.0, pulse_type='P', freq_min=50.0,
            freq_max=400.0, n_time_samples=20, time_max=1.0, march_start=0.0, courant_factor=1.0)

    def test_halfspace_surface_is_refused(self, tmp_path):
        from uacpy.core.exceptions import UnsupportedFeatureError
        out = tmp_path / 'sp.env'
        with pytest.raises(UnsupportedFeatureError, match='half-space surface'):
            self._write(out, _top_block_env())
        assert not out.exists(), "a rejected deck must not be left behind"

    def test_halfspace_bottom_is_refused(self, tmp_path):
        from uacpy.core import BoundaryProperties
        from uacpy.core.exceptions import UnsupportedFeatureError
        env = _top_block_env(surface=BoundaryProperties(acoustic_type='vacuum'))
        with pytest.raises(UnsupportedFeatureError, match='half-space bottom'):
            self._write(tmp_path / 'sp.env', env)

    def test_vacuum_deck_writes_no_halfspace_row(self, tmp_path):
        from uacpy.core import BoundaryProperties
        out = tmp_path / 'sp.env'
        self._write(out, _top_block_env(
            _bio(), surface=BoundaryProperties(acoustic_type='vacuum'),
            bottom=BoundaryProperties(acoustic_type='rigid')))
        lines = out.read_text().splitlines()
        assert lines[3][2] == 'V', f"TopOpt(2) is not 'V': {lines[3]}"
        assert _floats(lines[5]) == [10.0, 20.0, 400.0, 5.0, 0.1]
        # The mesh line comes straight after the absorption block.
        assert _floats(lines[6]) == [0.0, 0.0, 100.0], (
            f"expected the SSP mesh line, got {lines[6]!r}")

    @pytest.mark.requires_binary
    def test_sparc_reads_the_vacuum_rigid_deck(self, tmp_path):
        from uacpy.core import BoundaryProperties
        self._write(tmp_path / 'sp.env', _top_block_env(
            _bio(), surface=BoundaryProperties(acoustic_type='vacuum'),
            bottom=BoundaryProperties(acoustic_type='rigid')))
        text = _run_at('sparc.exe', 'sp', tmp_path)
        assert 'Biological attenuation' in text


class TestTopReflectionTableStaging:
    """A ``'file'`` surface makes AT open ``<root>.trc`` (RefCoef.f90:64-76),
    so the table has to be staged beside the .env just like the bottom .brc."""

    @staticmethod
    def _table(tmp_path):
        path = tmp_path / 'top_table.trc'
        angles = np.linspace(0.0, 90.0, 19)
        with open(path, 'w') as fh:
            fh.write(f"{len(angles)}\n")
            for a in angles:
                fh.write(f"{a:12.6f}     0.900000     0.000000\n")
        return path

    @staticmethod
    def _env(reflection_file):
        from uacpy.core import BoundaryProperties
        return _top_block_env(surface=BoundaryProperties(
            acoustic_type='file', reflection_file=str(reflection_file)))

    @pytest.mark.parametrize('writer', ['kraken', 'scooter'])
    def test_top_table_is_staged_beside_the_env(self, tmp_path, writer):
        table = self._table(tmp_path)
        out = tmp_path / f'{writer}.env'
        write = _write_kraken if writer == 'kraken' else _write_scooter
        write(out, self._env(table))
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
        env = _top_block_env(
            surface=BoundaryProperties(acoustic_type='file'))
        with pytest.raises(ConfigurationError, match='reflection_file'):
            _write_kraken(tmp_path / 'kr.env', env)

    @pytest.mark.requires_binary
    @pytest.mark.parametrize('exe,writer', [('krakenc.exe', 'kraken'),
                                            ('scooter.exe', 'scooter')])
    def test_binary_reads_the_staged_top_table(self, tmp_path, exe, writer):
        table = self._table(tmp_path)
        write = _write_kraken if writer == 'kraken' else _write_scooter
        write(tmp_path / 'trc.env', self._env(table))
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
        return _write_kraken(path, env)

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
        from uacpy.core.boundary import BoundaryType
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
        text = _write_kraken(tmp_path / 'kr.env', env).read_text()
        assert "'A' " in text and '~' not in text


class TestBoundaryTypeIsTheSingleSourceOfTruth:
    """``io.at_codes.BOUNDARY_CODES`` is the only place a boundary letter is
    decided, in both directions. A silent vacuum fallback for an unrecognised
    type would model a different surface than the one asked for."""

    def test_every_type_has_one_letter_and_reads_back_as_itself(self):
        from uacpy.core.boundary import BoundaryType
        from uacpy.io.at_codes import (
            BOUNDARY_CODES, BOUNDARY_TYPES_BY_CODE, boundary_code)
        assert {bt: boundary_code(bt.value) for bt in BoundaryType} == {
            BoundaryType.VACUUM: 'V', BoundaryType.RIGID: 'R',
            BoundaryType.HALF_SPACE: 'A', BoundaryType.FILE: 'F',
            BoundaryType.PRECALC: 'P'}
        for bt, code in BOUNDARY_CODES.items():
            assert BOUNDARY_TYPES_BY_CODE[code] is bt
        # Bellhop's grain-size letter is read, never written.
        assert BOUNDARY_TYPES_BY_CODE['G'] is BoundaryType.HALF_SPACE
        assert 'G' not in BOUNDARY_CODES.values()

    def test_the_letter_is_never_an_input_spelling(self):
        from uacpy.io.at_codes import boundary_code, parse_boundary_type
        with pytest.raises(ConfigurationError, match='invalid boundary type'):
            parse_boundary_type('A')
        with pytest.raises(ConfigurationError, match='invalid boundary type'):
            boundary_code('A')

    def test_unknown_acoustic_type_raises(self):
        from uacpy.core import BoundaryProperties
        from uacpy.io.oalib_writer import get_top_bc_code
        env = _top_block_env(
            surface=BoundaryProperties(acoustic_type='vacuum'))
        object.__setattr__(env.surface.nodes[0], 'acoustic_type', 'mud')
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
        from uacpy.core.deck_limits import DECK_DEPTH_RESOLUTION_M
        from uacpy.io.oalib_writer import deck_depth
        got = deck_depth(depth)
        assert abs(got - depth) <= DECK_DEPTH_RESOLUTION_M

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
        from uacpy.core.deck_limits import DECK_DEPTH_RESOLUTION_M
        assert DECK_DEPTH_RESOLUTION_M < 100.0 * np.finfo(np.float32).eps

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
        env = uacpy.Environment(
            name='deck', bathymetry=100.04,
            ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.04, 1500.0)]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.5,
                                      attenuation=0.5))
        out = _write_kraken(tmp_path / 'kr.env', env)
        mesh_depth = self._mesh_line_depth(out)
        assert mesh_depth >= 100.04, (
            f"water column ends at {mesh_depth} m, above env.depth=100.04")
        ssp_rows = self._water_ssp_rows(out)
        assert float(ssp_rows[-1][0]) == mesh_depth, (
            "the deepest SSP sample must equal the mesh depth exactly")

    def test_at_and_bellhop_decks_share_the_water_column(self, tmp_path):
        from uacpy.core import BoundaryProperties
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        src, rcv = _src_rcv()
        env = uacpy.Environment(
            name='deck', bathymetry=100.04,
            ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.04, 1500.0)]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.5,
                                      attenuation=0.5))
        at = _write_kraken(tmp_path / 'kr.env', env)
        bh = tmp_path / 'bh.env'
        write_bellhop_env_file(bh, env, src, rcv)
        assert self._mesh_line_depth(at) == self._mesh_line_depth(bh), (
            "AT and Bellhop model different water columns for the same env")

    def test_layer_interfaces_keep_their_thickness(self, tmp_path):
        from uacpy.core.boundary import SedimentLayer, BoundaryProperties
        from uacpy.core.bottom import SeabedColumn
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
        out = _write_kraken(tmp_path / 'kr.env', env)
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
        assert shd.plot_type.strip() == 'irregular'
        assert shd.pressure.shape == (1, 1, 1, len(_PAIRED_RANGES_KM))
        # The header still declares the full receiver-depth vector.
        assert len(shd.receiver_depths) == len(_PAIRED_DEPTHS)

    def test_irregular_shd_is_one_paired_receiver_axis(self, tmp_path):
        from uacpy.io.oalib_reader import read_shd_file

        irr = read_shd_file(self._run(tmp_path, 'irr', 'CG  I')
                            .with_suffix('.shd'))
        rect = read_shd_file(self._run(tmp_path, 'rect', 'CG  R')
                             .with_suffix('.shd'))

        assert irr.axes == ['range']
        assert irr.shape == (len(_PAIRED_RANGES_KM),)
        assert irr.aux_coords['receiver_depth'][0] == 'range'
        assert irr.aux_coords['receiver_depth'][1].tolist() == _PAIRED_DEPTHS
        assert 'receiver_depths' not in irr.metadata
        exported = irr.to_xarray()
        assert exported.coords['receiver_depth'].dims == ('range',)
        assert exported.coords['receiver_depth'].values.tolist() == _PAIRED_DEPTHS
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
        with pytest.raises(FileFormatError, match='could not parse'):
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
        assert shd.bearings.tolist() == [0.0, 10.0, 20.0]
        assert shd.pressure.shape[0] == 3

    def test_read_shd_file_refuses_multi_bearing(self, tmp_path):
        from uacpy.io.oalib_reader import read_shd_file
        from uacpy.core.exceptions import UnsupportedFeatureError

        # The file is well-formed — the refusal is a capability limit of the
        # single-bearing wrapper, not corruption.
        path = self._run(tmp_path, 'tl3d', 'C^   3').with_suffix('.shd')
        with pytest.raises(UnsupportedFeatureError, match='3 receiver bearings'):
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
        with pytest.raises(FileFormatError, match='could not parse') as exc:
            read_ray_file(path)
        assert '5X8' in str(exc.value)
        assert 'byte order' not in str(exc.value)


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
        from uacpy.core.boundary import BoundaryProperties, SedimentLayer
        from uacpy.core.bottom import Bottom, SeabedColumn
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
            interp_ssp='linear', frequencies=[100.0],
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


class TestRtsToPressureFftOnly:
    """``rts_to_pressure`` accepts only ``method='fft'``; any other
    selector, 'goertzel' included, raises a typed error naming it."""

    @staticmethod
    def _tone(amp, phase, n_t=512, fs=1000.0, bin_index=8):
        """One range column carrying an exactly-on-bin cosine."""
        dt = 1.0 / fs
        freq = fs * bin_index / n_t
        t = np.arange(n_t) * dt
        p = (amp * np.cos(2 * np.pi * freq * t + phase)).reshape(n_t, 1)
        from uacpy.io import RtsFile
        return RtsFile(title='', positions=np.array([1000.0]), times=t,
                       pressure=p), freq

    def test_fft_recovers_the_analytic_phasor(self):
        from uacpy.models.sparc import rts_to_pressure
        amp, phase = 3.0, 0.700
        rts, freq = self._tone(amp, phase)
        p, _ = rts_to_pressure(rts, freq, method='fft')
        assert abs(p[0]) == pytest.approx(amp, rel=1e-4)
        assert np.angle(p[0]) == pytest.approx(phase, abs=1e-4)

    def test_a_goertzel_selector_is_refused(self):
        from uacpy.models.sparc import rts_to_pressure
        rts, freq = self._tone(2.5, -1.1)
        with pytest.raises(ConfigurationError, match='goertzel'):
            rts_to_pressure(rts, freq, method='goertzel')


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
                       profile_ranges=np.asarray(ranges_m, dtype=float))
        return (tmp_path / 'x.flp').read_text()

    def test_a_pair_that_collapses_at_1mm_is_refused(self, tmp_path):
        with pytest.raises(ConfigurationError, match='1 mm'):
            self._write(tmp_path, [0.0, 4000.0, 4000.0000001, 10000.0])

    def test_a_separated_axis_is_written(self, tmp_path):
        text = self._write(tmp_path, [0.0, 4000.0, 6000.0, 10000.0])
        assert '4.000000' in text and '6.000000' in text

    def test_the_quantum_is_the_boundary(self, tmp_path):
        """1 mm apart is exactly representable; anything closer is not."""
        from uacpy.core.deck_limits import DECK_RANGE_RESOLUTION_M
        self._write(tmp_path, [0.0, 4000.0, 4000.0 + 2 * DECK_RANGE_RESOLUTION_M,
                               10000.0])
        with pytest.raises(
                ConfigurationError,
                match="both write as .* at the deck's 1 mm resolution"):
            self._write(tmp_path, [0.0, 4000.0,
                                   4000.0 + DECK_RANGE_RESOLUTION_M / 10.0, 10000.0])


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
                ).dB).ravel()

        assert np.nanmax(np.abs(tl(100.04) - tl(100.1))) > 1.0


class TestReadPrt:
    """io.md §3: AT binaries dump fatal-error detail to ``<base>.prt``, not
    stderr; ``read_prt`` returns the text, ``None`` when the file is absent,
    and only the trailing ``tail_bytes`` when asked for an excerpt."""

    def test_absent_file_returns_none(self, tmp_path):
        from uacpy.io.oalib_reader import read_prt
        assert read_prt(tmp_path / 'missing.prt') is None

    def test_full_text_round_trips(self, tmp_path):
        from uacpy.io.oalib_reader import read_prt
        p = tmp_path / 'run.prt'
        p.write_text('*** FATAL ERROR ***\nMesh is too coarse\n')
        assert read_prt(p) == '*** FATAL ERROR ***\nMesh is too coarse\n'

    def test_tail_bytes_returns_only_the_trailing_excerpt(self, tmp_path):
        from uacpy.io.oalib_reader import read_prt
        p = tmp_path / 'run.prt'
        p.write_text('A' * 90 + 'TAIL_MARKER')
        assert read_prt(p, tail_bytes=11) == 'TAIL_MARKER'

    def test_tail_bytes_past_the_file_size_returns_everything(self, tmp_path):
        from uacpy.io.oalib_reader import read_prt
        p = tmp_path / 'run.prt'
        p.write_text('short log')
        assert read_prt(p, tail_bytes=4096) == 'short log'


def test_read_flp_accepts_list_directed_records_and_sorts(tmp_path):
    """AT's own decks carry trailing commas and bare descriptive text on
    scalar records ('9999,<TAB>! M'), which a list-directed READ accepts —
    the reader must too — and the Fortran Sorts every source/receiver axis
    after SubTab (misc/SourceReceiverPositions.f90:224,268), so axes come
    back ascending whatever order the deck listed."""
    from uacpy.io.oalib_reader import _parse_flp
    import pathlib
    sduct = pathlib.Path('uacpy/third_party/Acoustics-Toolbox/tests/'
                         'sduct/sductK.flp')
    if sduct.exists():
        d = _parse_flp(str(sduct))   # used to raise: invalid literal '9999,'
        assert np.all(np.diff(d['pos']['r']['z']) >= 0)
    deck = tmp_path / 'unsorted.flp'
    deck.write_text(
        "'title'\n'R'\n9999,\t! M\n1\n0.0 /\n3\n20 1 5 km sorted after read\n"
        "2\n100 25 /\n3\n4000 0 2000 /\n1\n0.0 /\n")
    d = _parse_flp(str(deck))
    assert np.all(np.diff(d['pos']['r']['r']) >= 0)
    assert np.all(np.diff(d['pos']['s']['z']) >= 0)
    assert np.all(np.diff(d['pos']['r']['z']) >= 0)


class TestReadFlpBuildsItsDebugPreviewOnlyWhenItCanPrint:
    """``read_flp``'s two axis previews exist solely for a debug message.

    ``verbose`` is public and defaults off, and Python evaluates a call's
    arguments before ``log_message`` can decline them — so an unguarded
    preview is formatted on every read, for a line no one asked for. The
    previews stay (``verbose`` is a documented switch, not dead code);
    only the moment they are built moved.
    """

    DECK = ("'title'\n'R'\n9999,\t! M\n1\n0.0 /\n3\n20 1 5\n"
            "2\n100 25 /\n3\n4000 0 2000 /\n1\n0.0 /\n")

    def _deck(self, tmp_path):
        path = tmp_path / 'preview.flp'
        path.write_text(self.DECK)
        return path

    def _calls(self, tmp_path, **kwargs):
        from unittest.mock import patch
        from uacpy.io.oalib_reader import _parse_flp
        with patch('uacpy.io.oalib_reader.log_message') as logger:
            _parse_flp(str(self._deck(tmp_path)), **kwargs)
        return [call.args[1] for call in logger.call_args_list]

    def test_a_default_read_never_formats_a_preview(self, tmp_path):
        messages = self._calls(tmp_path)
        assert not [m for m in messages if 'rProf (km)' in m]
        assert not [m for m in messages if 'Rro (m)' in m]

    def test_a_verbose_read_gets_both_previews(self, tmp_path):
        """The switch keeps working — this is a lazier preview, not a
        removed one."""
        messages = self._calls(tmp_path, verbose='debug')
        assert [m for m in messages if 'rProf (km)' in m]
        assert [m for m in messages if 'Rro (m)' in m]

    def test_the_preview_lists_a_short_axis_value_by_value(self, tmp_path):
        messages = self._calls(tmp_path, verbose='debug')
        prof = next(m for m in messages if 'rProf (km)' in m)
        assert prof.endswith('0.00')

    def test_the_read_returns_the_same_deck_either_way(self, tmp_path):
        from uacpy.io.oalib_reader import _parse_flp
        path = str(self._deck(tmp_path))
        quiet = _parse_flp(path)
        loud = _parse_flp(path, verbose='debug')
        assert quiet['title'] == loud['title']
        assert np.array_equal(quiet['r_prof'], loud['r_prof'])
        assert np.array_equal(quiet['pos']['r']['ro'],
                              loud['pos']['r']['ro'])


class TestPhaseSpeedBoundsStayContractedToRangeZero:
    """``resolve_phase_speed_bounds`` derives BOTH bounds from
    ``env.ssp.to_pairs()`` — the range-0 column. That is correct, and correct
    for one reason only: the deck it describes is that same column, because
    ``write_ssp_section`` writes ``env.ssp.extend_to(...).to_pairs()`` and the
    half-space is pinned with ``halfspace_at(range=0.0)`` on the very next
    line. It is *not* correct on a multi-profile deck, which is why
    ``kraken._window.c_low_for`` derives its own floor from the full
    ``env.ssp.sound_speed``
    block instead.

    The ``c_high`` half carries the identical range-0 reading, and its safety
    rests entirely on that argument: every caller either supplies ``c_high``
    explicitly or has already collapsed the SSP to one column. The day a model
    that declares ``range_dependent_ssp`` calls it with ``c_high=None``, it
    will silently under-derive ``c_high`` off one column and truncate the top
    of the mode spectrum. These two tests pin the invariant that keeps that
    from happening quietly."""

    _EXPECTED_CALLERS = {
        'io/oalib_writer.py',      # write_phase_speed_and_rmax and its writers
        'models/_window.py',       # resolve_window over the default rule
        'models/kraken/_window.py',
        'models/scooter/_model.py',
        'models/sparc/_model.py',
    }

    def test_only_the_known_modules_derive_phase_speed_bounds(self):
        """A new caller must extend the drive test below, not appear silently."""
        import ast
        import uacpy as _uacpy
        pkg = Path(_uacpy.__file__).parent
        targets = {'resolve_phase_speed_bounds', 'resolve_window',
                   'write_phase_speed_and_rmax'}
        found = set()
        for path in sorted(pkg.rglob('*.py')):
            if 'tests' in path.relative_to(pkg).parts:
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                name = (fn.id if isinstance(fn, ast.Name)
                        else fn.attr if isinstance(fn, ast.Attribute) else None)
                if name in targets:
                    found.add(path.relative_to(pkg).as_posix())
        assert found == self._EXPECTED_CALLERS, (
            f"the set of modules deriving AT phase-speed bounds changed: "
            f"{found ^ self._EXPECTED_CALLERS}. Each caller must either pass "
            f"c_low and c_high explicitly or hand in an SSP already collapsed "
            f"to one range column — add it to "
            f"test_no_driver_derives_a_bound_from_a_two_dimensional_ssp and "
            f"prove it there.")

    @staticmethod
    def _range_dependent_env(bottom=None):
        ssp = SoundSpeedProfile(
            depths=np.array([0.0, 100.0, 200.0]),
            sound_speed=np.array([[1500.0, 1480.0, 1450.0]] * 3),
            ranges=np.array([0.0, 5000.0, 10000.0]))
        return uacpy.Environment(
            name='rd', bathymetry=np.array([[0.0, 200.0], [10000.0, 220.0]]),
            ssp=ssp,
            bottom=bottom if bottom is not None else BoundaryProperties(
                acoustic_type='half-space', sound_speed=1600.0, density=1.8,
                attenuation=0.2))

    @pytest.mark.requires_binary
    def test_no_driver_derives_a_bound_from_a_two_dimensional_ssp(
            self, monkeypatch, tmp_path):
        """Instrument the real resolver and drive every deck writer that reaches
        it on a range-dependent-SSP environment."""
        import warnings as _warnings
        from uacpy.models import _window
        from uacpy.models.kraken import _window as kraken_mod
        from uacpy.models.sparc import _model as sparc_mod
        from uacpy.models.scooter import _model as scooter_mod

        real = _window.resolve_phase_speed_bounds
        real_window = _window.resolve_window
        seen = []

        def spy_in(label):
            def spy(env, c_low=None, c_high=None):
                seen.append((label, env.ssp.n_ranges, c_low is None,
                             c_high is None))
                return real(env, c_low, c_high)
            return spy

        def window_spy_in(label):
            def spy(env, *, c_low, c_high, model_name):
                seen.append((label, env.ssp.n_ranges, c_low is None,
                             c_high is None))
                return real_window(env, c_low=c_low, c_high=c_high,
                                   model_name=model_name)
            return spy

        # Every module whose functions call the resolver, named: a module
        # that stops calling it fails test_architecture's patch-target gate
        # rather than dropping out of this sweep. Scooter and SPARC reach
        # the rule through resolve_window, spied where they call it.
        monkeypatch.setattr(kraken_mod, 'resolve_phase_speed_bounds',
                            spy_in('kraken'))
        for label, mod in {'sparc': sparc_mod,
                           'scooter': scooter_mod}.items():
            monkeypatch.setattr(mod, 'resolve_window', window_spy_in(label))

        env = self._range_dependent_env()
        src = uacpy.Source(depths=100.0, frequencies=200.0)
        rcv = uacpy.Receiver(depths=[50.0, 150.0],
                             ranges=np.linspace(1000.0, 9000.0, 5))

        with _warnings.catch_warnings():
            _warnings.simplefilter('ignore')
            # The multi-profile field deck and the r = 0 MODES deck.
            k = uacpy.Kraken(verbose=False, n_segments=3)
            k.run_settings(env, src, rcv)
            k.run_settings(env, src, rcv, run_mode=uacpy.RunMode.MODES)

            uacpy.Scooter(verbose=False).run_settings(env, src, rcv)

            # SPARC's deck carries only vacuum/rigid boundaries; hand it a
            # rigid seabed so its settings resolver is what gets exercised.
            sp_env = self._range_dependent_env(
                bottom=BoundaryProperties(acoustic_type='rigid'))
            uacpy.SPARC(verbose=False).run_settings(sp_env, src, rcv)

        reached = {label for label, *_ in seen}
        assert {'kraken', 'scooter', 'sparc'} <= reached, (
            f"a driver never reached its resolver — only {sorted(reached)}")
        derived = [rec for rec in seen if rec[2] or rec[3]]
        assert all(n_ranges == 1 for _label, n_ranges, _cl, _ch in derived), (
            f"a driver derived a phase-speed bound from a range-dependent SSP: "
            f"{[rec for rec in derived if rec[1] != 1]} of {seen}")


# omega = 2π·100; AddArr merges at omega·|Δdelay| < 0.05, i.e. complex
# delay distance below ~7.96e-5 s at this frequency (ArrMod.f90:8,44).
_FREQ = 100.0


def _write_arr(path, records, freq=_FREQ):
    """One source (50 m), one receiver depth (75 m), one range (500 m).

    ``records``: 8-tuples ``(amp, phase_deg, delay_s, delay_imag_s,
    source_angle, receiver_angle, n_top, n_bot)`` in the column order of
    ``Bellhop/ArrMod.f90:119-126``. Floats are written with ``repr`` so the
    file round-trips them bit-exactly.
    """
    lines = ["'2D'", repr(float(freq)), "1 50.0", "1 75.0", "1 500.0",
             str(len(records)), str(len(records))]
    for r in records:
        lines.append(' '.join(repr(float(v)) for v in r[:6])
                     + f' {int(r[6])} {int(r[7])}')
    path.write_text('\n'.join(lines) + '\n')
    return path


def _cell(result):
    return result.by_receiver[0][0][0]


def _read_merged(path, merge=True):
    """Read a ``.arr`` with the AddArr merge applied — these tests exercise the
    merge machinery itself, which the reader no longer applies by default."""
    from uacpy.io.oalib_reader import read_arr_file
    return read_arr_file(path, merge=merge)


class TestAddArrPostMerge:

    def test_within_tolerance_pair_merges_amplitude_add_weighted_delay(
            self, tmp_path):
        """omega·|Δdelay| = 0.032 and |Δphase| = 0.035 rad are both inside
        AddArr's 0.05 tolerance: the pair becomes one arrival whose
        amplitude is the sum, whose delay/delays_imag and both angles are
        amplitude-weighted, and whose phase and bounce counts come from the
        record that is first in (source_angle, delay) order — here the record
        listed SECOND in the file."""
        first_by_angle = (0.3, 0.0, 0.001, 0.0, -5.0, 5.0, 0, 1)
        second_by_angle = (0.1, 2.0, 0.00105, -4e-6, -4.9, 5.1, 2, 3)
        path = _write_arr(tmp_path / 'pair.arr',
                          [second_by_angle, first_by_angle])

        cell = _cell(_read_merged(path))
        assert cell['n_arrivals'] == 1
        assert cell['amplitudes'][0] == pytest.approx(0.4)
        assert cell['delays'][0] == pytest.approx(
            0.75 * 0.001 + 0.25 * 0.00105)
        assert cell['delays_imag'][0] == pytest.approx(0.25 * -4e-6)
        assert cell['source_angles'][0] == pytest.approx(
            0.75 * -5.0 + 0.25 * -4.9)
        assert cell['receiver_angles'][0] == pytest.approx(
            0.75 * 5.0 + 0.25 * 5.1)
        assert cell['phases'][0] == 0.0
        assert cell['n_top_bounces'][0] == 0
        assert cell['n_bot_bounces'][0] == 1

    def test_beyond_delay_tolerance_pair_stays_two_arrivals(self, tmp_path):
        """omega·|Δdelay| = 0.31 exceeds the 0.05 tolerance: both records
        survive, sorted by delay."""
        path = _write_arr(tmp_path / 'far.arr', [
            (0.1, 0.0, 0.0015, 0.0, -4.9, 5.1, 2, 3),
            (0.3, 0.0, 0.001, 0.0, -5.0, 5.0, 0, 1),
        ])
        cell = _cell(_read_merged(path))
        assert cell['n_arrivals'] == 2
        assert cell['delays'].tolist() == [0.001, 0.0015]
        assert cell['amplitudes'].tolist() == [0.3, 0.1]

    def test_same_delay_different_phase_stays_two_arrivals(self, tmp_path):
        """|Δphase| = π/2 exceeds the 0.05 rad tolerance — AddArr's phase
        test exists to keep surface-reflected and direct paths apart even
        at equal delay (ArrMod.f90:41)."""
        path = _write_arr(tmp_path / 'phase.arr', [
            (0.3, 0.0, 0.001, 0.0, -5.0, 5.0, 0, 1),
            (0.1, 90.0, 0.001, 0.0, -4.9, 5.1, 1, 1),
        ])
        cell = _cell(_read_merged(path))
        assert cell['n_arrivals'] == 2
        assert sorted(cell['phases'].tolist()) == pytest.approx(
            [0.0, np.pi / 2], abs=1e-12)

    def test_merge_false_returns_records_unmerged_in_file_order(
            self, tmp_path):
        """The raw escape hatch: a within-tolerance pair stays two records
        and the out-of-delay-order file listing is preserved."""
        records = [
            (0.2, 0.0, 0.003, 0.0, 10.0, -10.0, 1, 1),
            (0.3, 0.0, 0.001, 0.0, -5.0, 5.0, 0, 1),
            (0.1, 2.0, 0.00105, -4e-6, -4.9, 5.1, 2, 3),
        ]
        path = _write_arr(tmp_path / 'raw.arr', records)
        cell = _cell(_read_merged(path, merge=False))
        assert cell['n_arrivals'] == 3
        expected = np.array(records)
        # Column 1 is the file's degree phase; the reader stores radians.
        expected[:, 1] = np.deg2rad(expected[:, 1])
        for col, key in enumerate(_FIELD_KEYS):
            assert cell[key].tolist() == expected[:, col].tolist(), key


class TestTotalKeyRecordSort:

    @staticmethod
    def _records():
        """Six pairwise non-merging records (delays 1 ms apart:
        omega·Δ = 0.63) plus one within-tolerance pair, so a shuffle
        exercises the sort and the merge's visit order together."""
        recs = [(0.1 * (k + 1), 10.0 * k, 0.001 * (k + 1), -1e-6 * k,
                 -10.0 + 2.0 * k, 8.0 - 2.0 * k, k, k + 1)
                for k in range(6)]
        recs.append((0.05, 0.5, 0.001 + 5e-5, 0.0, -9.9, 8.1, 3, 4))
        return recs

    def test_shuffled_file_order_reads_identically(self, tmp_path):
        recs = self._records()
        shuffled = [recs[i] for i in
                    np.random.default_rng(0x19F2).permutation(len(recs))]
        assert shuffled != recs

        cell_a = _cell(_read_merged(_write_arr(tmp_path / 'a.arr', recs)))
        cell_b = _cell(_read_merged(
            _write_arr(tmp_path / 'b.arr', shuffled)))

        assert cell_a['n_arrivals'] == cell_b['n_arrivals'] == 6
        for key in _FIELD_KEYS:
            assert np.array_equal(cell_a[key], cell_b[key]), key

    def test_merging_an_already_merged_set_is_identity(self, tmp_path):
        """Re-reading a file that lists exactly the merged, sorted records
        reproduces them bit-for-bit — the merge pass is idempotent."""
        cell_a = _cell(_read_merged(
            _write_arr(tmp_path / 'a.arr', self._records())))

        # The cell holds radians; the file column is degrees, so the
        # round trip writes the phase back in the file's unit.
        remerged = [tuple(np.rad2deg(cell_a[key][i]) if key == 'phases'
                          else cell_a[key][i] for key in _FIELD_KEYS)
                    for i in range(cell_a['n_arrivals'])]
        cell_b = _cell(_read_merged(
            _write_arr(tmp_path / 'b.arr', remerged)))

        assert cell_b['n_arrivals'] == cell_a['n_arrivals']
        for key in _FIELD_KEYS:
            assert np.array_equal(cell_a[key], cell_b[key]), key


def _write_rts(path, body):
    path.write_text("'T'\n" + body)
    return path


class TestReadRtsFileRefusesNonPositiveReceiverCounts:

    def test_negative_receiver_count_raises_file_format_error(self, tmp_path):
        from uacpy.core.exceptions import FileFormatError
        from uacpy.io import read_rts_file
        rts = _write_rts(tmp_path / 'n.rts', '-5 1.0 2.0 3.0\n')
        with pytest.raises(FileFormatError, match='declares -5'):
            read_rts_file(rts)

    def test_zero_receiver_count_raises_file_format_error(self, tmp_path):
        from uacpy.core.exceptions import FileFormatError
        from uacpy.io import read_rts_file
        rts = _write_rts(tmp_path / 'z.rts', '0 1.0 2.0\n')
        with pytest.raises(FileFormatError, match='declares 0'):
            read_rts_file(rts)

    def test_positive_receiver_count_parses_ranges_and_time_steps(
            self, tmp_path):
        from uacpy.io import read_rts_file
        rts = _write_rts(tmp_path / 'p.rts',
                         '2 10.0 20.0\n0.0 1.0 2.0\n0.5 3.0 4.0\n')
        data = read_rts_file(rts)
        assert data.positions.tolist() == [10.0, 20.0]
        assert data.times.tolist() == [0.0, 0.5]
        assert data.pressure.shape == (2, 2)


_FIELD_KEYS = ('amplitudes', 'phases', 'delays', 'delays_imag',
               'source_angles', 'receiver_angles', 'n_top_bounces', 'n_bot_bounces')


class TestBoundaryConditionSeabedGetsNoPadMedia:
    """``plan_multi_profile_media`` equalises the profiles of a
    range-dependent KRAKEN deck with pad media repeating the half-space. A
    ``vacuum`` / ``rigid`` / reflection-table half-space carries no material
    to repeat — the ``sound_speed`` / ``density`` / ``attenuation`` on a
    parameter-free ``BoundaryProperties`` are constructor placeholders, which
    is why ``models._window.resolve_phase_speed_bounds`` consults
    ``_NON_GEOACOUSTIC_TYPES`` before capping cHigh on them. Padding with
    them writes metres of invented sediment under a seabed the caller asked
    to be pressure-release."""

    @staticmethod
    def _segment(water_depth, halfspace, layers=()):
        return uacpy.Environment(
            bathymetry=Bathymetry(ranges=[0.0, 4000.0],
                                  depths=[water_depth, water_depth]),
            ssp=SoundSpeedProfile(depths=[0.0, water_depth],
                                  sound_speed=[1500.0, 1490.0]),
            bottom=Bottom(columns=[SeabedColumn(layers=list(layers),
                                                halfspace=halfspace)]))

    @pytest.mark.parametrize('acoustic_type', ['vacuum', 'rigid'])
    def test_equal_profiles_carry_no_sub_bottom_medium(self, acoustic_type):
        from uacpy.io.oalib_writer import plan_multi_profile_media
        hs = BoundaryProperties(acoustic_type=acoustic_type)
        segments = [(0.0, self._segment(100.0, hs)),
                    (2.0, self._segment(100.0, hs))]
        n_media, bottom, plans = plan_multi_profile_media(segments)
        assert n_media == 1
        assert bottom == 100.0
        assert plans == [[], []]

    def test_the_deck_names_no_invented_sediment(self, tmp_path):
        from uacpy.io.oalib_writer import write_multi_profile_env
        hs = BoundaryProperties(acoustic_type='vacuum')
        segments = [(0.0, self._segment(100.0, hs)),
                    (2.0, self._segment(100.0, hs))]
        out = tmp_path / 'vac.env'
        write_multi_profile_env(
            str(out), segments,
            uacpy.Source(depths=30.0, frequencies=50.0),
            uacpy.Receiver(depths=np.linspace(0.0, 100.0, 101),
                           ranges=[1000.0]),
            rmax_m=4000.0, c_low=0.0, c_high=2000.0)
        text = out.read_text()
        # The placeholder trio BoundaryProperties defaults to.
        assert '1600' not in text
        assert '1.500000' not in text
        # NMedia is written once per profile, right after the frequency.
        assert text.count("\n1\n'") == 2
        assert "'V'" in text

    def test_unequal_bathymetry_is_refused_by_acoustic_type(self):
        from uacpy.io.oalib_writer import plan_multi_profile_media
        hs = BoundaryProperties(acoustic_type='vacuum')
        segments = [(0.0, self._segment(100.0, hs)),
                    (2.0, self._segment(80.0, hs))]
        with pytest.raises(ConfigurationError, match="'vacuum'"):
            plan_multi_profile_media(segments)

    def test_a_geoacoustic_half_space_is_padded(self):
        """The pad is correct there: the half-space *is* that material, so a
        transparent slice of it below the seafloor changes nothing."""
        from uacpy.io.oalib_writer import (
            plan_multi_profile_media, _PAD_MEDIUM_THICKNESS_M)
        hs = BoundaryProperties(acoustic_type='half-space',
                                sound_speed=1800.0, density=2.0,
                                attenuation=0.5)
        segments = [(0.0, self._segment(100.0, hs)),
                    (2.0, self._segment(80.0, hs))]
        n_media, bottom, plans = plan_multi_profile_media(segments)
        assert n_media == 2
        assert bottom == pytest.approx(100.0 + _PAD_MEDIUM_THICKNESS_M)
        assert [len(p) for p in plans] == [1, 1]
        assert plans[1][0][2] == 1800.0

    def test_a_layer_stack_over_a_vacuum_writes_its_own_layers(self):
        from uacpy.io.oalib_writer import plan_multi_profile_media
        hs = BoundaryProperties(acoustic_type='rigid')
        layer = SedimentLayer(thickness=5.0, sound_speed=1600.0,
                              density=1.7, attenuation=0.3)
        segments = [(0.0, self._segment(100.0, hs, [layer])),
                    (2.0, self._segment(100.0, hs, [layer]))]
        n_media, bottom, plans = plan_multi_profile_media(segments)
        assert n_media == 2
        assert bottom == 105.0
        assert [len(p) for p in plans] == [1, 1]
        assert plans[0][0][:3] == (100.0, 105.0, 1600.0)


class TestRtsProjectionNeedsATimeAxis:
    """A run that wrote a single output time has no second sample to
    difference against, and ``rts_to_pressure``'s tone estimate needs at
    least two output times, on both of its branches."""

    @staticmethod
    def _rts(tmp_path, n_steps):
        p = tmp_path / 'sparc.rts'
        rows = ''.join(f"{0.05 * i} 1.5 -0.5\n" for i in range(n_steps))
        p.write_text("'SPARC'\n2\n1000.0 2000.0\n" + rows)
        return p

    def test_a_single_step_file_reads_one_time(self, tmp_path):
        from uacpy.io.oalib_reader import read_rts_file
        data = read_rts_file(self._rts(tmp_path, 1))
        assert data.times.tolist() == [0.0]
        assert data.pressure.shape == (1, 2)

    @pytest.mark.parametrize('pulse_type', [None, 'P'])
    def test_projecting_it_is_a_typed_error_naming_the_two_it_needs(
            self, tmp_path, pulse_type):
        from uacpy.io.oalib_reader import read_rts_file
        from uacpy.models.sparc import rts_to_pressure
        data = read_rts_file(self._rts(tmp_path, 1))
        with pytest.raises(ConfigurationError,
                           match='at least two output times') as err:
            rts_to_pressure(data, 100.0, pulse_type=pulse_type)
        assert 'rfftfreq' not in str(err.value)

    def test_two_steps_project(self, tmp_path):
        from uacpy.io.oalib_reader import read_rts_file
        from uacpy.models.sparc import rts_to_pressure
        data = read_rts_file(self._rts(tmp_path, 8))
        p_at_f, ranges = rts_to_pressure(data, 2.5)
        assert p_at_f.shape == (2,)
        assert ranges.tolist() == [1000.0, 2000.0]


class TestFieldFlpRejectsCoupledIncoherent:
    """``field.f90:126-129`` ERROUTs on coupled modes asked for an incoherent
    sum. It matters more than an ordinary deck error because
    ``misc/FatalError.f90:30`` is ``STOP '<string>'``, so every
    Acoustics-Toolbox fatal error exits with status 0 — the run writes no
    ``.shd`` and a caller trusting the return code reads a stale or missing
    output as success. Column 2 is only read for a range-dependent run
    (``field.f90:122``), so a single-profile deck is unaffected.
    """

    @staticmethod
    def _check(option, n_profiles):
        from uacpy.io.oalib_writer import _validate_flp_option
        _validate_flp_option(option, n_profiles=n_profiles)

    def test_coupled_with_an_incoherent_sum_is_refused(self):
        with pytest.raises(ConfigurationError, match='coupled modes'):
            self._check('RC I', 2)

    @pytest.mark.parametrize('option', ['RC C', 'RA I'])
    def test_the_legal_combinations_are_accepted(self, option):
        self._check(option, 2)

    def test_a_single_profile_deck_never_reads_column_two(self):
        self._check('RC I', 1)


class TestAnIrregularShdIsStridedByItsOwnLayout:
    """``bellhop3D.f90:166`` sets ``NRz_per_range = 1`` for an irregular grid
    and then never uses it: its record index is ``Pos%NRz`` in every term
    (``:405-410``), while 2-D ``bellhop.f90:323`` really does stride by
    ``NRz_per_range``. Both share ``ReadEnvironmentBell``, so both write the
    same ``'irregular '`` PlotType and the header alone cannot tell them apart.

    The two layouts differ in total record count, so the file settles it —
    rather than a guess from ``Ntheta``, since a 3-D run may carry a single
    bearing.
    """

    @staticmethod
    def _build(path, *, three_d, nfreq=1, ntheta=2, nsz=2, nrz=3, nrr=4):
        import struct
        recl = 41
        record_bytes = recl * 4
        buf = bytearray()

        def record(payload):
            assert len(payload) <= record_bytes
            buf.extend(payload.ljust(record_bytes, b'\x00'))

        record(struct.pack('<i', recl) + b'BELLHOP test'.ljust(80, b' '))
        record(b'irregular '.ljust(10, b' '))
        record(struct.pack('<7i', nfreq, ntheta, 1, 1, nsz, nrz, nrr)
               + struct.pack('<dd', 100.0, 0.0))
        record(np.full(nfreq, 100.0, dtype='<f8').tobytes())
        record(np.arange(ntheta, dtype='<f8').tobytes())
        record(np.zeros(1, dtype='<f8').tobytes())
        record(np.zeros(1, dtype='<f8').tobytes())
        record(np.zeros(nsz, dtype='<f4').tobytes())
        record(np.zeros(nrz, dtype='<f4').tobytes())
        record(np.arange(nrr, dtype='<f8').tobytes())
        per_block = nrz if three_d else 1
        n_data = nfreq * ntheta * nsz * per_block
        for i in range(n_data):
            v = np.zeros(2 * nrr, dtype='<f4')
            # Tag from 1, not 0: an all-zero sample is legitimately read back
            # as no-data (NaN), which would make record 0 indistinguishable
            # from a gap.
            v[0::2] = float(i + 1)
            record(v.tobytes())
        path.write_bytes(bytes(buf))
        return n_data

    def test_a_two_d_irregular_file_keeps_one_receiver_per_block(self,
                                                                 tmp_path):
        from uacpy.io.oalib_reader import read_shd_bin
        path = tmp_path / 'flat.shd'
        self._build(path, three_d=False)
        assert read_shd_bin(str(path)).pressure.shape == (2, 2, 1, 4)

    def test_a_three_d_irregular_file_is_read_in_record_order(self, tmp_path):
        from uacpy.io.oalib_reader import read_shd_bin
        path = tmp_path / 'cube.shd'
        n_data = self._build(path, three_d=True)
        pressure = read_shd_bin(str(path)).pressure
        assert pressure.shape == (2, 2, 3, 4)
        # Each record was tagged with its own index; reading them in order is
        # exactly what a correct stride produces.
        tags = np.real(pressure[..., 0]).ravel()
        assert np.array_equal(tags, np.arange(1, n_data + 1, dtype=float))

    def test_read_shd_file_refuses_a_three_d_irregular_file(self, tmp_path):
        """A 3-D irregular cube has Nrz receiver rows per source depth; a
        Field carries one paired receiver axis, so the wrapper refuses
        rather than returning row 0 as the whole file."""
        from uacpy.core.exceptions import UnsupportedFeatureError
        from uacpy.io.oalib_reader import read_shd_bin, read_shd_file
        path = tmp_path / 'cube_single_bearing.shd'
        self._build(path, three_d=True, ntheta=1, nrz=3, nrr=3)
        assert read_shd_bin(str(path)).pressure.shape == (1, 2, 3, 3)
        with pytest.raises(UnsupportedFeatureError, match='receiver rows'):
            read_shd_file(path)

    def test_read_shd_file_returns_the_paired_row_of_a_two_d_irregular_file(
            self, tmp_path):
        from uacpy.io.oalib_reader import read_shd_file
        path = tmp_path / 'flat_single_bearing.shd'
        self._build(path, three_d=False, ntheta=1, nsz=1, nrz=3, nrr=3)
        field = read_shd_file(path)
        assert field.data.shape == (3,)
        assert 'depth' not in field.coords
        assert field.aux_coords['receiver_depth'][1].size == 3


class TestRtsFileParsesEveryFortranRealSpelling:
    """``read_rts_file`` walks the same free token stream ``read_ts`` does, so
    it accepts the same reals. Python's ``float()`` rejects two spellings a
    list-directed READ takes: the 'D' exponent a double-precision WRITE emits,
    and the letterless three-digit exponent a ``Gw.d`` WRITE produces when the
    ``E`` no longer fits (``Scooter/sparc.f90:294``)."""

    @staticmethod
    def _rts(path, body):
        path.write_text("'Title'\n" + body)
        return path

    def test_a_letterless_three_digit_exponent_parses(self, tmp_path):
        from uacpy.io.oalib_reader import read_rts_file
        p = self._rts(tmp_path / 'a.rts',
                      "2\n0.0 100.0\n0.0 0.123457-118 1.0\n")
        assert read_rts_file(p).pressure[0, 0] == pytest.approx(1.23457e-119)

    def test_a_d_exponent_parses(self, tmp_path):
        from uacpy.io.oalib_reader import read_rts_file
        p = self._rts(tmp_path / 'b.rts', "2\n0.0 100.0\n1.0D-03 1.0 2.0\n")
        assert read_rts_file(p).times[0] == pytest.approx(1e-3)

    def test_the_range_column_takes_the_same_spellings(self, tmp_path):
        from uacpy.io.oalib_reader import read_rts_file
        p = self._rts(tmp_path / 'c.rts', "2\n0.0 1.0-120\n0.0 1.0 2.0\n")
        assert read_rts_file(p).positions[1] == pytest.approx(1e-120)

    def test_a_non_numeric_token_raises_fileformaterror(self, tmp_path):
        from uacpy.io.oalib_reader import read_rts_file
        p = self._rts(tmp_path / 'd.rts', "2\n0.0 100.0\n0.0 wrong 1.0\n")
        with pytest.raises(FileFormatError, match='wrong'):
            read_rts_file(p)


class TestTheSparcDeckRefusesRoughInterfaces:
    """sparc.f90:177 stops with 'Rough interfaces not allowed' at exit 0 on a
    non-zero surface or layer roughness; the deck refuses it where it is
    written, with the same reason."""

    def _write(self, tmp_path, env):
        from uacpy.core.source import Source
        from uacpy.core.receiver import Receiver
        from uacpy.io.oalib_writer import write_sparc_env_file
        write_sparc_env_file(
            tmp_path / 's.env', env, Source(depths=50.0, frequencies=100.0),
            Receiver(depths=[50.0], ranges=[1000.0]),
            interp_ssp='linear', output_mode='R', n_mesh=200,
            rmax_m=2000.0, c_low=1400.0, c_high=1700.0, pulse_type='P',
            freq_min=50.0, freq_max=200.0, n_time_samples=256, time_max=2.0, march_start=0.0,
            courant_factor=1.0)

    def test_a_rough_surface_is_refused(self, tmp_path):
        import uacpy
        from uacpy.core.boundary import BoundaryProperties
        from uacpy.core.surface import Surface
        from uacpy.core.exceptions import UnsupportedFeatureError
        env = uacpy.Environment(
            name='r', bathymetry=100.0, ssp=1500.0,
            surface=Surface(nodes=[BoundaryProperties(
                acoustic_type='vacuum', roughness=0.5)]))
        with pytest.raises(UnsupportedFeatureError, match='Rough interfaces'):
            self._write(tmp_path, env)

    def test_a_smooth_deck_is_written(self, tmp_path):
        import uacpy
        from uacpy.core.boundary import BoundaryProperties
        self._write(tmp_path, uacpy.Environment(
            name='s', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='rigid')))
        assert (tmp_path / 's.env').exists()


def test_read_arr_file_accepts_fortran_exponent_tokens(tmp_path):
    """Every field of a ``.arr`` record is a list-directed Fortran REAL
    (``Bellhop/ArrMod.f90:113-127``), so the reader takes the tokens a
    Fortran ``WRITE`` may emit — a ``D`` exponent and a letterless
    three-digit exponent — through ``fortran_float`` like the other AT text
    readers, not through ``float()``."""
    from uacpy.io.oalib_reader import read_arr_file
    path = tmp_path / 'exp.arr'
    path.write_text("'2D'\n2.0D2\n1 5.0E1\n1 75.0\n1 500.0\n1\n1\n"
                    "1.0 0.0 5.0D-1 1.0-101 10.0 -10.0 1 0\n")
    result = read_arr_file(path)
    assert float(result.frequencies[0]) == 200.0
    cell = result.by_receiver[0][0][0]
    assert cell['delays'][0] == 0.5
    assert cell['delays_imag'][0] == 1.0e-101


class TestReadFlpPreviewSwitchesAtTenEntries:
    """``read_flp`` lists an axis value by value while it has fewer than ten
    entries and prints only its two ends from ten on; both branches format
    each value with two decimals."""

    @staticmethod
    def _deck(tmp_path, n_prof):
        path = tmp_path / f'preview{n_prof}.flp'
        path.write_text(f"'title'\n'R'\n9999,\t! M\n{n_prof}\n0.0 {n_prof - 1}.0 /"
                        "\n3\n20 1 5\n2\n100 25 /\n3\n4000 0 2000 /\n"
                        f"{n_prof}\n0.0 0.0 /\n")
        return path

    def _previews(self, tmp_path, n_prof):
        from unittest.mock import patch
        from uacpy.io.oalib_reader import _parse_flp
        with patch('uacpy.io.oalib_reader.log_message') as logger:
            _parse_flp(str(self._deck(tmp_path, n_prof)), verbose='debug')
        messages = [call.args[1] for call in logger.call_args_list]
        return (next(m for m in messages if 'rProf (km)' in m),
                next(m for m in messages if 'Rro (m)' in m))

    def test_nine_entries_are_listed_one_by_one(self, tmp_path):
        prof, offsets = self._previews(tmp_path, 9)
        assert prof.endswith(', '.join(f'{k:.2f}' for k in range(9)))
        assert offsets.endswith(', '.join(['0.00'] * 9))

    def test_ten_entries_show_only_the_two_ends(self, tmp_path):
        prof, offsets = self._previews(tmp_path, 10)
        assert prof.endswith('0.00 … 9.00')
        assert offsets.endswith('0.00 … 0.00')


class TestScooterExtraTopOptCharacter:
    """``write_scooter_env_file(topopt_extra=…)`` appends its character to the
    TopOpt record, inside the quotes after the broadband slot, and nothing
    else in the deck moves; ``write_kraken_env_file`` has no such keyword."""

    def test_the_character_lands_at_the_end_of_the_topopt_record(self, tmp_path):
        env = _top_block_env(_fg())
        plain = _write_scooter(tmp_path / 'plain.env', env).read_text().splitlines()
        extra = _write_scooter(tmp_path / 'extra.env', env,
                               topopt_extra='0').read_text().splitlines()
        assert extra[3] == plain[3][:-1] + "0'"
        assert extra[:3] == plain[:3] and extra[4:] == plain[4:]

    def test_the_kraken_writer_takes_no_extra_character(self, tmp_path):
        with pytest.raises(TypeError, match='topopt_extra'):
            _write_kraken(tmp_path / 'k.env', _top_block_env(_fg()),
                          topopt_extra='0')


class TestDecksCarryTheWaterDensity:
    """Where every deck writes ``env.water_density``: the Acoustics
    Toolbox SSP rows and the OASES water layers carry the value itself;
    Bounce and the RAM codes fix the water at 1 and get seabed densities
    as ratios."""

    @staticmethod
    def _ssp_rows(env):
        buf = io.StringIO()
        write_ssp_section(buf, env, 100.0, ssp_topopt='C')
        return [ln.split() for ln in buf.getvalue().splitlines()
                if ln.strip().endswith('/')]

    def test_acoustics_toolbox_ssp_rows_carry_it(self):
        rows = self._ssp_rows(water_density_env())
        assert rows and all(r[3] == '1.027000' for r in rows)
        rows = self._ssp_rows(water_density_env(water_density=1.0))
        assert all(r[3] == '1.000000' for r in rows)

    def test_bounce_gets_seabed_densities_as_ratios_to_it(self, tmp_path):
        out = tmp_path / 'b.env'
        write_bounce_input_file(
            out, water_density_env(), Source(depths=10.0, frequencies=100.0),
            interp_ssp='linear', n_mesh=0,
            c_low=1400.0, c_high=2000.0, rmax_m=1.0)
        lines = out.read_text().splitlines()
        top = [ln for ln in lines
               if ln.strip().startswith('0.00') and '1500.000000' in ln]
        assert top, out.read_text()
        # bounce.f90 references R to a unit density and never reads this
        # row's, so the water stays at 1 and the seabed becomes a ratio.
        assert top[0].split()[3] == '1.000000'
        seabed = [ln for ln in lines if '1700.000000' in ln]
        assert seabed and seabed[0].split()[3] == f'{1.5 / 1.027:.6f}'

    def test_oases_water_layers_carry_it(self):
        buf = io.StringIO()
        _emit_water_layers(buf, [(0.0, 1500.0), (100.0, 1500.0)],
                           surface_roughness=0.0, extra_columns=1,
                           water_density=1.027,
                           water_ac=lambda zt, zb, c: 1e-8)
        rows = buf.getvalue().splitlines()
        assert rows and all(r.split()[5] == '1.027000' for r in rows)


@pytest.mark.requires_binary  # constructs RAM (resolves its binary)
class TestRamDecksCarryTheWaterDensityRatio:
    """Every RAM code fixes the water at 1 (RAM guide) and reads the seabed
    density relative to it, so the deck carries rho_b / rho_w."""

    def _ram(self):
        from uacpy.models import RAM
        return RAM(verbose=False, earth_curvature=False)

    def test_collins_density_block_is_rho_b_over_rho_w(self):
        model = self._ram()
        seg = ram_collins.collins_range_segments(
            water_density_env(), 'ramgeo', 150.0, 400.0,
            knobs=model._knob_record(), speed_bounds=model._speed_bounds)[0]
        assert all(v == pytest.approx(1.5 / 1.027) for _, v in seg['bottom_rho'])
        model = self._ram()
        seg = ram_collins.collins_range_segments(
            water_density_env(water_density=1.0), 'ramgeo', 150.0, 400.0,
            knobs=model._knob_record(), speed_bounds=model._speed_bounds)[0]
        assert all(v == 1.5 for _, v in seg['bottom_rho'])

    def test_mpirams_density_row_is_rho_b_over_rho_w(self, tmp_path):
        ram = self._ram()
        _, _, _, rho, _, _, _ = ram_mpirams.prepare_bottom_properties(
            water_density_env(), tmp_path, absorber_span=20.0, zmax=150.0,
            dz=0.5, knobs=ram._knob_record(), log=ram._log)
        np.testing.assert_allclose(rho, 1.5 / 1.027)
        _, _, _, rho, _, _, _ = ram_mpirams.prepare_bottom_properties(
            water_density_env(water_density=1.0), tmp_path, absorber_span=20.0, zmax=150.0,
            dz=0.5, knobs=ram._knob_record(), log=ram._log)
        np.testing.assert_allclose(rho, 1.5)


class TestArrPhaseIsRadiansAtTheReaderBoundary:
    """``Bellhop/ArrMod.f90:120`` writes ``SNGL(RadDeg) * Phase``, so the
    ``.arr`` column is degrees; ``read_arr_file`` converts once and every
    consumer downstream — the per-arrival dict, ``Arrivals.phases``,
    ``received_amplitudes`` and the AddArr merge tolerance — sees radians.
    A 180 in the file must therefore read back as pi: a pin on 180 would
    pass a reader that skips the conversion, a pin on pi cannot."""

    @staticmethod
    def _read(path, **kw):
        from uacpy.io.oalib_reader import read_arr_file
        return read_arr_file(path, **kw)

    def test_a_surface_bounce_record_reads_as_pi_not_180(self, tmp_path):
        path = _write_arr(tmp_path / 'top.arr',
                          [(0.3, 180.0, 0.001, 0.0, -5.0, 5.0, 1, 0)])
        result = self._read(path)
        assert _cell(result)['phases'][0] == pytest.approx(np.pi, abs=1e-12)
        assert result.arrivals[0]['phase'] == pytest.approx(np.pi, abs=1e-12)
        assert result.phases[0] == pytest.approx(np.pi, abs=1e-12)
        # exp(1j * pi) flips the sign: the stored value feeds the complex
        # factor with no further conversion.
        assert result.received_amplitudes[0] == pytest.approx(-0.3 + 0j,
                                                              abs=1e-12)

    def test_a_filtered_copy_keeps_the_radian_value(self, tmp_path):
        """``_spawn`` rebuilds ``by_receiver`` from the flat dicts; the
        rebuilt cell must carry the same unit as the freshly-read one."""
        path = _write_arr(tmp_path / 'two.arr', [
            (0.3, 0.0, 0.001, 0.0, -5.0, 5.0, 0, 0),
            (0.1, 180.0, 0.0011, 0.0, -4.9, 5.1, 1, 0)])
        result = self._read(path)
        surface = result.filter_by_bounces(kind='surface')
        assert _cell(surface)['phases'][0] == pytest.approx(np.pi, abs=1e-12)

    @pytest.mark.parametrize('step_deg, n_kept', [(2.0, 1), (4.0, 2)])
    def test_the_merge_tolerance_is_applied_to_the_radian_value(
            self, tmp_path, step_deg, n_kept):
        """AddArr merges when ``|dphase| < 0.05`` rad (``ArrMod.f90:8,45``):
        2 deg = 0.035 rad merges, 4 deg = 0.070 rad does not. Read as
        degrees, both pairs would stay apart."""
        path = _write_arr(tmp_path / 'tol.arr', [
            (0.3, 10.0, 0.001, 0.0, -5.0, 5.0, 0, 1),
            (0.1, 10.0 + step_deg, 0.001, 0.0, -4.9, 5.1, 0, 1)])
        cell = _cell(self._read(path, merge=True))
        assert cell['n_arrivals'] == n_kept

    @pytest.mark.requires_binary
    def test_bellhops_first_surface_bounce_arrives_with_phase_pi(
            self, tmp_path):
        """Isovelocity water, vacuum surface: the one-top-bounce path picks
        up exactly the pi of the pressure-release reflection and nothing
        else (no caustic, no bottom phase). Bellhop writes it as 180 in the
        file; the Python side must hold pi to 1e-6."""
        from uacpy.models import Bellhop, RunMode
        from uacpy.core import Receiver
        env = Environment(
            name='iso', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.5,
                                      attenuation=0.5))
        source = Source(depths=50.0, frequencies=200.0)
        receiver = Receiver(depths=[50.0], ranges=[1000.0])
        result = Bellhop(verbose=False).run(
            env, source, receiver, run_mode=RunMode.ARRIVALS)
        surface_only = [a for a in result.arrivals
                        if a['n_top_bounces'] == 1 and a['n_bot_bounces'] == 0]
        assert surface_only, "no one-top-bounce arrival in the cell"
        first = min(surface_only, key=lambda a: a['delay'])
        assert abs(first['phase'] - np.pi) < 1e-6


class TestDepthBlocksTakeACarrierOrAnArray:
    """``write_source_depths`` and ``write_receiver_depths`` accept the
    carrier or a plain depth array, and write the same record for both."""

    @pytest.mark.parametrize('writer', ['source', 'receiver'])
    def test_the_array_writes_the_carriers_record(self, writer):
        from uacpy.core.receiver import Receiver
        from uacpy.io.oalib_writer import (write_receiver_depths,
                                           write_source_depths)
        depths = np.array([10.0, 25.5])
        if writer == 'source':
            fn, carrier = write_source_depths, Source(depths=depths,
                                                      frequencies=100.0)
        else:
            fn, carrier = write_receiver_depths, Receiver(
                depths=depths, ranges=np.array([100.0, 200.0]))
        by_carrier, by_array = io.StringIO(), io.StringIO()
        fn(by_carrier, carrier)
        fn(by_array, depths)
        assert by_array.getvalue() == by_carrier.getvalue() == (
            "2\n10.000000 25.500000 /\n")


class TestRtsReadersShareOneParser:
    """``read_rts_file`` and ``read_ts`` read the SPARC ``.rts`` through one
    parser, and ``rts_to_pressure`` evaluates on the file's own times."""

    @staticmethod
    def _rts(path, times, p, positions=(1000.0,)):
        rows = [f"{len(positions)}"] + [' '.join(f"{x:.6g}" for x in positions)]
        rows += [' '.join(f"{v:.9g}" for v in (t, *np.atleast_1d(pv)))
                 for t, pv in zip(times, p)]
        path.write_text("'run'\n" + '\n'.join(rows) + '\n')
        return path

    def test_both_readers_return_the_same_arrays(self, tmp_path):
        from uacpy.io import read_rts_file, read_ts
        t = np.arange(8) * 0.01
        path = self._rts(tmp_path / 'a.rts', t, np.sin(t), (1.0, 2.0))
        rts, ts = read_rts_file(path), read_ts(path)
        assert type(rts) is type(ts)
        np.testing.assert_array_equal(rts.times, ts.times)
        np.testing.assert_array_equal(rts.pressure, ts.pressure)
        np.testing.assert_array_equal(rts.positions, ts.positions)
        assert (rts.title, ts.title) == ('run', "'run'")

    def test_read_ts_reads_a_title_cut_inside_a_character(self, tmp_path):
        from uacpy.io import read_ts
        path = tmp_path / 'cut.rts'
        path.write_bytes("'café".encode()[:-1] + b"'\n1\n5.0\n0.0 1.0\n")
        assert read_ts(path).pressure.tolist() == [[1.0]]

    def test_the_tone_keeps_its_phase_when_the_times_start_late(
            self, tmp_path):
        from uacpy.io import read_rts_file
        from uacpy.models.sparc import rts_to_pressure
        f0, phase = 100.3, 0.7
        t = 1.0 + np.arange(4096) / 4096.0
        p = np.cos(2 * np.pi * f0 * t + phase)
        data = read_rts_file(self._rts(tmp_path / 'late.rts', t, p))
        p_f0, _ = rts_to_pressure(data, f0)
        assert np.angle(p_f0[0]) == pytest.approx(phase, abs=1e-3)
        assert abs(p_f0[0]) == pytest.approx(1.0, abs=1e-3)


class TestKrakenFamilyWritersTakeTheirSettingsFromTheEnvironment:
    """The KRAKEN-family deck writers derive what the Environment says and
    default the solver settings, like ``write_bellhop_env_file``."""

    @staticmethod
    def _env(**kw):
        return Environment(
            name='deck', bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.0, 1520.0)]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.5,
                                      attenuation=0.5), **kw)

    @staticmethod
    def _geometry():
        return (Source(depths=25.0, frequencies=200.0),
                uacpy.Receiver(depths=[50.0], ranges=[1000.0, 3000.0]))

    def test_a_kraken_deck_needs_the_carriers_and_the_window(
            self, tmp_path):
        from uacpy.io import read_env, write_kraken_env_file
        env = self._env()
        write_kraken_env_file(tmp_path / 'k.env', env, *self._geometry(),
                              c_low=1425.0, c_high=1785.0)
        got, *_, opt = read_env(tmp_path / 'k.env')
        assert opt['interp_ssp'] == 'linear' and opt['n_mesh'] == [0]
        assert opt['rmax_m'] == 3000.0
        assert (opt['c_low'], opt['c_high']) == (1425.0, 1785.0)
        assert got.bottom.halfspace_at(range=0.0).sound_speed == 1700.0

    def test_the_boundary_letters_are_the_environments(self, tmp_path):
        from uacpy.io import write_scooter_env_file
        env = self._env(surface=BoundaryProperties(acoustic_type='rigid'))
        write_scooter_env_file(tmp_path / 's.env', env, *self._geometry(),
                               c_low=1425.0, c_high=1785.0)
        lines = tmp_path.joinpath('s.env').read_text().splitlines()
        assert lines[3][2] == 'R'
        assert any(ln.startswith("'A'") for ln in lines)

    def test_a_depth_array_receiver_needs_rmax(self, tmp_path):
        from uacpy.io import write_kraken_env_file
        src, _ = self._geometry()
        with pytest.raises(ConfigurationError, match='rmax_m'):
            write_kraken_env_file(tmp_path / 'k.env', self._env(), src,
                                  np.array([50.0]), c_low=1425.0,
                                  c_high=1785.0)

    def test_a_negative_multi_profile_mesh_is_refused(self, tmp_path):
        from uacpy.io import write_multi_profile_env
        with pytest.raises(ConfigurationError, match='n_mesh must be >= 0'):
            write_multi_profile_env(
                tmp_path / 'm.env', [(0.0, self._env()), (2000.0, self._env())],
                *self._geometry(), n_mesh=-1, c_low=1425.0, c_high=1785.0)

    def test_the_multi_profile_rmax_defaults_to_the_farthest_receiver(
            self, tmp_path):
        from uacpy.io import write_multi_profile_env
        out = tmp_path / 'm.env'
        write_multi_profile_env(
            out, [(0.0, self._env()), (2000.0, self._env())],
            *self._geometry(), c_low=1425.0, c_high=1785.0)
        # One RMax record per profile, 3000 m written in km.
        assert out.read_text().splitlines().count('3.000000') == 2

    def test_receiver_ranges_write_the_same_from_an_array(self):
        from uacpy.io.oalib_writer import write_receiver_ranges
        _, rcv = self._geometry()
        a, b = io.StringIO(), io.StringIO()
        write_receiver_ranges(a, rcv)
        write_receiver_ranges(b, np.array([1000.0, 3000.0]))
        assert a.getvalue() == b.getvalue() == "2\n1.000000 3.000000 /\n"


class TestBiologicalLayerEdgesGetANodePair:
    """The Acoustics Toolbox evaluates a Biological law only at SSP nodes
    (``misc/AttenMod.f90:103-104``, per node at ``misc/sspMod.f90:388-393``)
    and interpolates ``Im c`` between them, so a layer holding no node is
    lossless. The writers add a node on each edge and one 0.01 m outside it;
    measured, that brings Kraken within 0.2 dB of RAM (which samples the
    edges itself) where the bare profile applied 0.000 dB."""

    @staticmethod
    def _env(layers, ssp=((0.0, 1520.0), (100.0, 1500.0))):
        from uacpy.core import Biological
        return Environment(name='bio', bathymetry=100.0, ssp=list(ssp),
                           bottom=BoundaryProperties(sound_speed=1700.0,
                                                     density=1.8,
                                                     attenuation=0.5),
                           absorption=Biological(layers=layers))

    @staticmethod
    def _depths(env, code='C'):
        buf = io.StringIO()
        write_ssp_section(buf, env, 100.0, ssp_topopt=code)
        return [float(ln.split()[0]) for ln in buf.getvalue().splitlines()
                if ln.strip().endswith('/')]

    def test_an_edge_between_nodes_gets_the_pair_on_its_outer_side(self):
        env = self._env([(30.0, 70.0, 300.0, 4.0, 0.125)])
        assert self._depths(env) == [0.0, 29.99, 30.0, 70.0, 70.01, 100.0]

    def test_an_edge_on_an_existing_node_gets_only_its_partner(self):
        env = self._env([(30.0, 70.0, 300.0, 4.0, 0.125)],
                        ssp=((0.0, 1520.0), (30.0, 1514.0), (100.0, 1500.0)))
        assert self._depths(env) == [0.0, 29.99, 30.0, 70.0, 70.01, 100.0]

    def test_edges_on_the_surface_and_the_seafloor_add_nothing(self):
        env = self._env([(0.0, 100.0, 300.0, 4.0, 0.125)])
        assert self._depths(env) == [0.0, 100.0]

    def test_the_added_rows_follow_the_deck_interpolation(self):
        from uacpy.io.oalib_writer import biological_edge_nodes
        env = self._env([(30.0, 70.0, 300.0, 4.0, 0.125)])
        z = np.array([0.0, 100.0])
        c = np.array([1520.0, 1500.0])
        zc, cc = biological_edge_nodes(z, c, env.absorption, 'C', who='t')
        np.testing.assert_allclose(cc, np.interp(zc, z, c), rtol=0, atol=1e-9)
        zn, cn = biological_edge_nodes(z, c, env.absorption, 'N', who='t')
        np.testing.assert_allclose(
            cn, 1.0 / np.sqrt(np.interp(zn, z, 1.0 / c ** 2)),
            rtol=0, atol=1e-9)
        assert not np.allclose(cn[1:-1], cc[1:-1], rtol=0, atol=1e-6)

    def test_a_non_biological_law_leaves_the_rows_untouched(self):
        from uacpy.core import Thorp
        env = self._env([(30.0, 70.0, 300.0, 4.0, 0.125)])
        env.absorption = Thorp()
        assert self._depths(env) == [0.0, 100.0]

    @pytest.mark.parametrize('code', ['P', 'S'])
    def test_an_interior_edge_under_pchip_or_spline_is_refused(self, code):
        env = self._env([(30.0, 70.0, 300.0, 4.0, 0.125)])
        with pytest.raises(ConfigurationError, match="interp_ssp='linear'"):
            self._depths(env, code)

    @pytest.mark.parametrize('code', ['P', 'S'])
    def test_column_spanning_layers_are_accepted_under_pchip_or_spline(
            self, code):
        env = self._env([(0.0, 100.0, 300.0, 4.0, 0.125)])
        assert self._depths(env, code) == [0.0, 100.0]

    @pytest.mark.parametrize('interp', ['pchip', 'spline'])
    def test_the_models_refuse_it_before_writing_a_deck(self, interp):
        from uacpy.io.oalib_writer import (
            reject_biological_edges_under_neighbour_interp)
        inside = self._env([(30.0, 70.0, 300.0, 4.0, 0.125)])
        with pytest.raises(ConfigurationError, match=interp):
            reject_biological_edges_under_neighbour_interp(
                'Kraken', inside, interp)
        spanning = self._env([(0.0, 100.0, 300.0, 4.0, 0.125)])
        reject_biological_edges_under_neighbour_interp(
            'Kraken', spanning, interp)
        reject_biological_edges_under_neighbour_interp(
            'Kraken', inside, 'linear')



class TestTheWritersStateTheWindowTheyAreGiven:
    """The deck writers take the resolved phase-speed window; deriving it is
    the engines' (``uacpy.models._window``)."""

    @staticmethod
    def _env():
        return make_pekeris(name='w', density=1.5)

    @staticmethod
    def _geometry():
        return (uacpy.Source(depths=25.0, frequencies=200.0),
                uacpy.Receiver(depths=[50.0], ranges=[1000.0]))

    @pytest.mark.parametrize('missing', ['c_low', 'c_high'])
    def test_a_kraken_deck_needs_both_bounds(self, tmp_path, missing):
        from uacpy.io import write_kraken_env_file
        kw = dict(c_low=1400.0, c_high=1800.0)
        kw.pop(missing)
        with pytest.raises(TypeError, match=missing):
            write_kraken_env_file(tmp_path / 'k.env', self._env(),
                                  *self._geometry(), **kw)

    @pytest.mark.parametrize('missing', ['c_low', 'c_high'])
    def test_a_multi_profile_deck_needs_both_bounds(self, tmp_path,
                                                     missing):
        from uacpy.io import write_multi_profile_env
        kw = dict(c_low=1400.0, c_high=1800.0)
        kw.pop(missing)
        with pytest.raises(TypeError, match=missing):
            write_multi_profile_env(
                tmp_path / 'm.env', [(0.0, self._env()),
                                     (2000.0, self._env())],
                *self._geometry(), **kw)

    def test_the_deck_carries_the_bounds_given(self, tmp_path):
        from uacpy.io import read_env, write_kraken_env_file
        write_kraken_env_file(tmp_path / 'k.env', self._env(),
                              *self._geometry(), c_low=1412.0,
                              c_high=1999.0)
        *_, opt = read_env(tmp_path / 'k.env')
        assert (opt['c_low'], opt['c_high']) == (1412.0, 1999.0)


# ── a depth-varying absorption in the water rows' alphaI ──────────────────

#: AT's ``'F'`` follow-up row ``T S pH z_bar`` as uacpy once wrote it.
_T_S_PH_Z_BAR_ROW = re.compile(
    r'^-?\d+\.\d{4} -?\d+\.\d{4} -?\d+\.\d{4} -?\d+\.\d{4}$', re.M)


class TestARowLawIsWrittenIntoEachWaterRow:
    """Francois-Garrison (one water row or a profile) and a measured table
    have no ``TopOpt(4)`` letter uacpy writes: the deck leaves the letter
    blank and writes α at the deck frequency, in dB per local wavelength,
    into every water row's ``alphaI``."""

    F = 4000.0

    def _env(self, absorption, name='two-layer', ssp=1500.0):
        return Environment(name=name, bathymetry=100.0, ssp=ssp,
                           bottom='sand', absorption=absorption)

    def _write(self, path, env, frequency=F, **kwargs):
        from uacpy.io.oalib_writer import write_kraken_env_file
        write_kraken_env_file(path, env,
                              Source(depths=30.0, frequencies=frequency),
                              uacpy.Receiver(depths=[20.0], ranges=[1000.0]),
                              c_low=1400.0, c_high=2000.0, **kwargs)
        return path.read_text()

    def test_each_row_carries_the_formula_at_its_node(self, tmp_path):
        text = self._write(tmp_path / 'two_layer.env',
                           self._env(two_layer_absorption()))
        topopt = re.search(r"^'(\w\wW.)", text, flags=re.M).group(1)
        assert topopt[3] == ' '                   # no 'F' single-row law
        assert not _T_S_PH_Z_BAR_ROW.search(text)  # and no T S pH z_bar row
        rows = at_deck_water_rows(text)
        assert rows.shape[0] >= 2
        for z, c, alpha in rows:
            hand = two_layer_dB_per_m(self.F, z) * c / self.F
            assert alpha == pytest.approx(hand, rel=1e-8), z

    def test_a_multi_profile_deck_writes_each_profiles_rows(self, tmp_path):
        # Kraken's coupled-mode deck holds one block per profile, each with
        # its own sound speed; every block carries the law in its rows, at
        # its own c.
        from uacpy.io.oalib_writer import write_multi_profile_env
        segments = [(r, self._env(two_layer_absorption(), f'r{int(r)}', c))
                    for r, c in ((0.0, 1500.0), (2000.0, 1490.0))]
        deck = tmp_path / 'rd.env'
        write_multi_profile_env(deck, segments,
                                Source(depths=30.0, frequencies=self.F),
                                uacpy.Receiver(depths=[20.0],
                                               ranges=[3000.0]),
                                c_low=1400.0, c_high=2000.0)
        rows = at_deck_water_rows(deck.read_text())
        # The padded sub-bottom media (sand, 1650 m/s) are not water.
        rows = rows[rows[:, 1] < 1600.0]
        assert set(rows[:, 1]) == {1500.0, 1490.0}
        for z, c, alpha in rows:
            hand = two_layer_dB_per_m(self.F, z) * c / self.F
            assert alpha == pytest.approx(hand, rel=1e-8), (c, z)

    def test_a_table_refuses_a_deck_frequency_outside_it(self, tmp_path):
        env = self._env(measured_absorption_table(), 'measured')
        with pytest.raises(ConfigurationError, match='outside the table'):
            self._write(tmp_path / 'out.env', env, frequency=999.0)

    def _band_warnings(self, tmp_path, law, band, *, bathymetry=100.0,
                       category=None, needle='linear in frequency'):
        from uacpy.core.exceptions import NumericsWarning
        from uacpy.tests.conftest import recorded_warnings
        env = Environment(name='band', bathymetry=bathymetry, ssp=1500.0,
                          bottom='sand', absorption=law)
        with recorded_warnings() as rec:
            text = self._write(tmp_path / 'band.env', env,
                               frequency=10000.0, frequencies=band)
        return text, [str(w.message) for w in rec
                      if issubclass(w.category, category or NumericsWarning)
                      and needle in str(w.message)]

    def test_a_broadband_deck_freezes_a_profile_and_warns(self, tmp_path):
        # The rows hold dB/wavelength at the deck frequency, which the solver
        # re-applies across the band: over 5-15 kHz a profile is not that
        # line. Thorp keeps its letter, evaluated per frequency.
        from uacpy.core.absorption import Thorp
        band = np.linspace(5000.0, 15000.0, 11)
        for law, warns in ((two_layer_absorption(), True), (Thorp(), False)):
            _text, hits = self._band_warnings(tmp_path, law, band)
            assert bool(hits) is warns, law

    def test_a_broadband_deck_writes_one_water_row_as_f_at_mid_column(
            self, tmp_path):
        """One Francois-Garrison row on a deck covering several frequencies
        is AT's 'F', exact in frequency, at z_bar = half the water depth;
        the water rows carry no alphaI, and no frozen-row warning."""
        from uacpy.core.absorption import FrancoisGarrison
        text, hits = self._band_warnings(
            tmp_path, FrancoisGarrison(20.0, 35.0, 8.0),
            np.linspace(5000.0, 15000.0, 11), bathymetry=400.0)
        assert hits == []
        topopt = re.search(r"^'(\w\wW.)", text, flags=re.M).group(1)
        assert topopt[3] == 'F'
        (row,) = _T_S_PH_Z_BAR_ROW.findall(text)
        assert row.split() == ['20.0000', '35.0000', '8.0000', '200.0000']
        assert np.all(at_deck_water_rows(text)[:, 2] == 0.0)

    def test_any_depth_variation_keeps_the_frozen_rows(self, tmp_path):
        """'F' carries one T S pH row: a law with only its pH on depths of
        its own is a profile, written in the rows (c), not as 'F'."""
        from uacpy.core.absorption import FrancoisGarrison
        law = FrancoisGarrison(20.0, 35.0, [(0.0, 8.1), (400.0, 7.9)])
        text, _hits = self._band_warnings(
            tmp_path, law, np.linspace(5000.0, 15000.0, 11),
            bathymetry=400.0)
        assert re.search(r"^'(\w\wW.)", text, flags=re.M).group(1)[3] == ' '
        assert not _T_S_PH_Z_BAR_ROW.search(text)
        assert np.all(at_deck_water_rows(text)[:, 2] > 0.0)

    def test_the_mid_column_row_warns_past_the_band_rule(self, tmp_path):
        # The single depth is judged by the band rule
        # (BAND_ABSORPTION_WARN_DB_PER_KM = 0.05) over the band and the
        # column: 5-15 kHz, 20 °C water, 500 m misses the formula at depth by
        # 0.046 dB/km (silent), 600 m by 0.055 (warns).
        from uacpy.core.absorption import FrancoisGarrison
        from uacpy.core.exceptions import FallbackWarning
        law, band = FrancoisGarrison(20.0, 35.0, 8.0), np.linspace(
            5000.0, 15000.0, 11)
        kw = dict(category=FallbackWarning, needle="as AT's 'F' row")
        assert self._band_warnings(tmp_path, law, band, bathymetry=500.0,
                                   **kw)[1] == []
        (msg,) = self._band_warnings(tmp_path, law, band, bathymetry=600.0,
                                     **kw)[1]
        assert 'z_bar = 300 m' in msg and 'up to 0.055' in msg
        assert 'sediment layers and half-spaces' in msg
