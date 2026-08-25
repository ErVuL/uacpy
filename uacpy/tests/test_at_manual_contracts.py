"""Pins for the Acoustics Toolbox's own documentation, deck-grammar level.

Each test cites the manual section it pins:

- ``doc/EnvironmentalFile.htm`` (vendored AT tree): blocks (4) TopOpt —
  SSP-interpolation letters, attenuation-unit letter, bio-layer attenuation
  formula — and (4b).
- ``doc/kraken.htm`` / KRAKEN manual (2001) §4.2.2 ``kraken.hlp``: blocks
  (7) CLOW/CHIGH, (8) RMAX, (9) source/receiver depths, and the broadband
  frequency vector appended after them (TopOpt(6:6)='B').
- ``doc/index.htm`` (vendored): KRAKENC replaces elastic layers by an
  equivalent reflection coefficient, so fields inside them cannot be
  evaluated. (The same text is KRAKEN manual (2001) §4.2.1 ``notes.hlp``;
  the vendored HTML is cited because the manual PDF is not readable here.)
- ``KrakenField/field.f90:139``: RPROF(1) must be 0.0 — the binary itself
  refuses otherwise, ``ERROUT('FIELD', 'The first profile must be at a
  range of 0 km')``. ``doc/field.htm`` documents the record but not this
  constraint, so the Fortran is the citation.
- ``doc/sparc.htm``: the SPARC-only TopOpt(5:5) output mode and the four
  documented tail blocks (PULSE, FMin/FMax, ranges, output times, time
  integration).
- ``doc/bellhop.htm`` + Bellhop User Guide (2011) §2: NMEDIA is 1, the
  TopOpt(5:5) altimetry flag, the 7-character RunType, and the
  STEP (m) / ZBOX (m) / RBOX (km) record.
- ``doc/ATI_BTY_File.htm``: TYPE string, ranges in km, strictly
  monotonic range axis (same writer for .ati and .bty).
- ``doc/ReflectionCoefficientFile.htm``: the document's own example table.

Everything here is binary-free: the pins run the pure-python writers and
readers directly.
"""

import numpy as np
import pytest

import uacpy
from uacpy.core import Environment, BoundaryProperties
from uacpy.core.environment import SoundSpeedProfile
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.constants import BoundaryType


def _measured_env(**overrides):
    """An Environment whose SSP shape is 'measured', so the model's
    ``interp_ssp`` (not an isovelocity override) picks TopOpt(1)."""
    kwargs = dict(
        name='deck',
        bathymetry=100.0,
        ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.0, 1520.0)]),
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1700.0, density=1.5,
                                  attenuation=0.5),
    )
    kwargs.update(overrides)
    return Environment(**kwargs)


def _src_rcv():
    return (uacpy.Source(depths=25.0, frequencies=200.0),
            uacpy.Receiver(depths=[50.0], ranges=[1000.0, 2000.0]))


def _write_kraken(path, env, **overrides):
    from uacpy.io.oalib_writer import write_kraken_env_file
    src, rcv = _src_rcv()
    kwargs = dict(ssp_topopt='C', surface_type=BoundaryType.VACUUM,
                  bottom_type=BoundaryType.HALF_SPACE, frequencies=None,
                  n_mesh=0, rmax_m=5000.0, c_low=1400.0, c_high=2000.0)
    kwargs.update(overrides)
    write_kraken_env_file(path, env, src, rcv, **kwargs)
    return path


def _floats(line):
    return [float(tok) for tok in line.split() if tok != '/']


def _topopt(line):
    """The option string between the quotes of a TopOpt deck line."""
    return line.split("'")[1]


class TestTopOptInterpolationLetters:
    """``doc/EnvironmentalFile.htm`` block (4), TOPOPT(1:1): 'C' C-linear,
    'N' N2-linear, 'P' PCHIP, 'S' cubic spline, 'Q' quadrilateral (Bellhop).
    ``resolve_ssp_topopt`` is the single map every AT-family deck writes
    through, so each user-facing name is pinned to its letter."""

    @pytest.mark.parametrize('name,letter', [
        ('linear', 'C'),
        ('n2linear', 'N'),
        ('pchip', 'P'),
        ('cubic', 'S'),
        ('spline', 'S'),
        ('quad', 'Q'),
    ])
    def test_each_name_maps_to_its_documented_letter(self, name, letter):
        from uacpy.io.oalib_writer import resolve_ssp_topopt
        assert resolve_ssp_topopt(_measured_env(), name) == letter

    def test_analytic_is_refused(self):
        # 'A' selects the hard-coded Munk curve in misc/munk.f90, which
        # ignores env.ssp entirely, so the writer refuses it.
        from uacpy.io.oalib_writer import resolve_ssp_topopt
        with pytest.raises(ConfigurationError, match='analytic'):
            resolve_ssp_topopt(_measured_env(), 'analytic')


class TestAttenuationUnitLetterIsW:
    """``doc/EnvironmentalFile.htm`` block (4), TOPOPT(3:3): uacpy's
    documented convention is dB/wavelength, letter 'W'. Every AT-family
    deck must carry it in position 3 — a different letter silently
    rescales every attenuation value the deck carries."""

    def test_kraken_deck(self, tmp_path):
        lines = _write_kraken(tmp_path / 'k.env',
                              _measured_env()).read_text().splitlines()
        assert _topopt(lines[3])[2] == 'W'

    def test_scooter_deck(self, tmp_path):
        from uacpy.io.oalib_writer import write_scooter_env_file
        src, rcv = _src_rcv()
        out = tmp_path / 's.env'
        write_scooter_env_file(
            out, _measured_env(), src, rcv, ssp_topopt='C',
            surface_type=BoundaryType.VACUUM,
            bottom_type=BoundaryType.HALF_SPACE, frequencies=None,
            topopt_extra='', n_mesh=0, rmax_m=5000.0, c_low=1400.0,
            c_high=2000.0)
        assert _topopt(out.read_text().splitlines()[3])[2] == 'W'

    def test_sparc_deck(self, tmp_path):
        lines = _write_sparc(tmp_path / 'sp.env').read_text().splitlines()
        assert _topopt(lines[3])[2] == 'W'

    def test_bounce_deck(self, tmp_path):
        from uacpy.io.oalib_writer import write_bounce_input_file
        src, _ = _src_rcv()
        out = tmp_path / 'b.env'
        write_bounce_input_file(
            out, _measured_env(), src, ssp_topopt='C',
            bottom_type=BoundaryType.HALF_SPACE, n_mesh=0,
            c_low=1400.0, c_high=2000.0, rmax=5000.0)
        assert _topopt(out.read_text().splitlines()[3])[2] == 'W'

    def test_bellhop_deck(self, tmp_path):
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        src, rcv = _src_rcv()
        out = tmp_path / 'bh.env'
        write_bellhop_env_file(out, _measured_env(), src, rcv,
                               z_box=150.0, r_box=5000.0)
        assert _topopt(out.read_text().splitlines()[3])[2] == 'W'


class TestBiologicalLorentzianFormula:
    """``doc/EnvironmentalFile.htm`` block (4b): the loss is
    a = a0 / [ (1 - f0^2/f^2)^2 + 1/Q^2 ] dB/km, and the units are dB/km
    regardless of the attenuation-unit letter — the carrier takes no unit
    argument at all."""

    def test_alpha_matches_the_documented_lorentzian(self):
        from uacpy.core.absorption import Biological
        a0, f0, Q, f = 0.1, 400.0, 5.0, 500.0
        bio = Biological(layers=[(10.0, 20.0, f0, Q, a0)])
        expected_db_per_km = a0 / ((1.0 - f0**2 / f**2)**2 + 1.0 / Q**2)
        got = bio.alpha_db_per_m(f, [15.0])[0] * 1000.0
        assert got == pytest.approx(expected_db_per_km, rel=1e-12)

    def test_outside_the_layer_is_zero(self):
        from uacpy.core.absorption import Biological
        bio = Biological(layers=[(10.0, 20.0, 400.0, 5.0, 0.1)])
        assert bio.alpha_db_per_m(500.0, [25.0])[0] == 0.0

    def test_stacked_layers_sum_on_a_shared_boundary(self):
        # AttenMod.f90:102-109 tests z >= Z1 .AND. z <= Z2 per layer and
        # sums, so a depth on the boundary two layers share gets both.
        from uacpy.core.absorption import Biological
        one = Biological(layers=[(10.0, 20.0, 400.0, 5.0, 0.1)])
        two = Biological(layers=[(10.0, 20.0, 400.0, 5.0, 0.1),
                                 (20.0, 30.0, 400.0, 5.0, 0.1)])
        assert (two.alpha_db_per_m(500.0, [20.0])[0]
                == pytest.approx(2.0 * one.alpha_db_per_m(500.0, [20.0])[0]))


class TestKrakenDeckBlockOrder:
    """``doc/kraken.htm`` / manual §4.2.2: after the 6 common blocks the
    KRAKEN deck appends (7) CLOW CHIGH in m/s, (8) RMAX in km, then
    (9) NSD / SD(m) / NRD / RD(m), each depth vector closed by '/'. The
    walk below parses a written deck record by record in that order."""

    def test_blocks_seven_to_nine_in_order_and_units(self, tmp_path):
        lines = _write_kraken(
            tmp_path / 'k.env', _measured_env(),
        ).read_text().splitlines()

        assert lines[0] == "'deck'"
        assert float(lines[1]) == 200.0
        assert int(lines[2]) == 1
        assert _topopt(lines[3])[0] == 'C' and _topopt(lines[3])[1] == 'V'

        mesh = lines[4].split()
        assert int(mesh[0]) == 0 and float(mesh[2]) == 100.0
        # SSP rows run to the declared bottom depth, each closed by '/'.
        i = 5
        while not lines[i].strip().startswith("'"):
            assert lines[i].rstrip().endswith('/')
            i += 1
        assert _floats(lines[i - 1])[0] == 100.0

        # Block (6): BotOpt letter + sigma, then the 6-column 'A' row.
        assert lines[i].startswith("'A'")
        hs = _floats(lines[i + 1])
        assert len(hs) == 6 and hs[:2] == [100.0, 1700.0]

        # Block (7): CLOW CHIGH in m/s.
        assert _floats(lines[i + 2]) == [1400.0, 2000.0]
        # Block (8): RMAX in km — 5000 m arrives as 5.0.
        assert _floats(lines[i + 3]) == [5.0]
        # Block (9): NSD, SD (m) '/', NRD, RD (m) '/'; the deck ends there.
        assert int(lines[i + 4]) == 1
        assert _floats(lines[i + 5]) == [25.0]
        assert lines[i + 5].rstrip().endswith('/')
        assert int(lines[i + 6]) == 1
        assert _floats(lines[i + 7]) == [50.0]
        assert lines[i + 7].rstrip().endswith('/')
        assert i + 8 == len(lines)


class TestBroadbandFrequencyVectorTail:
    """``doc/EnvironmentalFile.htm`` TOPOPT(6:6)='B': the frequencies "are
    given as a vector in the last two lines of the envfil" — a count line
    then the values closed by '/' — read after the source/receiver depth
    blocks (kraken.f90 ReadfreqVec). Broadband is implemented in KRAKEN
    and SCOOTER, so both decks are pinned."""

    FREQS = np.array([50.0, 100.0, 200.0])

    def _assert_tail(self, lines):
        assert _topopt(lines[3])[5] == 'B'
        assert int(lines[-2]) == 3
        assert _floats(lines[-1]) == [50.0, 100.0, 200.0]
        assert lines[-1].rstrip().endswith('/')
        # The receiver-depth block sits directly above the vector.
        assert int(lines[-4]) == 1 and _floats(lines[-3]) == [50.0]

    def test_kraken_deck_ends_with_nfreq_and_vector(self, tmp_path):
        lines = _write_kraken(tmp_path / 'k.env', _measured_env(),
                              frequencies=self.FREQS).read_text().splitlines()
        self._assert_tail(lines)

    def test_scooter_deck_ends_with_nfreq_and_vector(self, tmp_path):
        from uacpy.io.oalib_writer import write_scooter_env_file
        src, rcv = _src_rcv()
        out = tmp_path / 's.env'
        write_scooter_env_file(
            out, _measured_env(), src, rcv, ssp_topopt='C',
            surface_type=BoundaryType.VACUUM,
            bottom_type=BoundaryType.HALF_SPACE, frequencies=self.FREQS,
            topopt_extra='', n_mesh=0, rmax_m=5000.0, c_low=1400.0,
            c_high=2000.0)
        self._assert_tail(out.read_text().splitlines())


class TestFieldFlpFirstProfileRangeMustBeZero:
    """``doc/field.htm`` block (4) / manual §4.3.1: '*** NOTE: RPROF( 1 )
    must be 0.0 ***'. The writer refuses the deck instead of letting
    field.exe misplace every profile."""

    def test_nonzero_first_profile_range_is_refused(self, tmp_path):
        from uacpy.io.oalib_writer import write_fieldflp
        pos = {'s': {'z': np.array([25.0])},
               'r': {'z': np.array([50.0]), 'r': np.array([1000.0])}}
        with pytest.raises(ConfigurationError, match='First profile range'):
            write_fieldflp(tmp_path / 'f.flp', 'RA', pos,
                           n_profiles=2,
                           profile_ranges_m=np.array([1000.0, 5000.0]))


class TestBellhopDeckGrammar:
    """``doc/bellhop.htm`` + Bellhop User Guide (2011) §2: NMEDIA is
    always 1 (line 3), TopOpt(5:5) flags the .ati file ('~', blank for a
    flat surface), RunType is a 7-character string, and the last record is
    STEP (m) then ZBOX (m), RBOX (km)."""

    def _write(self, tmp_path, env, **kwargs):
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        src, rcv = _src_rcv()
        path = tmp_path / 'bh.env'
        write_bellhop_env_file(path, env, src, rcv,
                               z_box=150.0, r_box=5000.0, **kwargs)
        return path.read_text().splitlines()

    def test_nmedia_line_is_the_literal_one(self, tmp_path):
        lines = self._write(tmp_path, _measured_env())
        assert lines[2].strip() == '1'

    def test_altimetry_flag_fills_topopt_position_five(self, tmp_path):
        flat = self._write(tmp_path, _measured_env())
        assert _topopt(flat[3])[4] == ' '
        wavy = self._write(
            tmp_path,
            Environment(bathymetry=200.0, ssp=1500.0,
                        altimetry=[(0.0, -2.0)]))
        assert _topopt(wavy[3])[4] == '~'
        assert (tmp_path / 'bh.ati').exists()

    def test_run_type_is_seven_characters_and_box_units_split(self, tmp_path):
        lines = self._write(tmp_path, _measured_env())
        run_type = next(ln for ln in lines if ln.startswith("'CB"))
        assert run_type == "'CB RR2 '"
        idx = lines.index(run_type)
        assert int(lines[idx + 1]) == 0
        assert _floats(lines[idx + 2]) == [-80.0, 80.0]
        assert lines[idx + 2].rstrip().endswith('/')
        assert _floats(lines[idx + 3]) == [0.0]
        # ZBOX stays in metres; RBOX (5000 m) arrives in km.
        assert _floats(lines[idx + 4]) == [150.0, 5.0]
        assert idx + 5 == len(lines)


def _write_sparc(path):
    from uacpy.io.oalib_writer import write_sparc_env_file
    src, rcv = _src_rcv()
    env = _measured_env(
        surface=BoundaryProperties(acoustic_type='vacuum'),
        bottom=BoundaryProperties(acoustic_type='rigid'))
    write_sparc_env_file(
        path, env, src, rcv, ssp_code='C',
        surface_type=BoundaryType.VACUUM, bottom_type=BoundaryType.RIGID,
        output_mode='S', n_mesh=0, rmax_m=5000.0, c_low=1400.0,
        c_high=2000.0, pulse_type='P', f_min=50.0, f_max=400.0,
        n_t_out=100, t_max=2.0, t_start=-0.1, t_mult=0.9)
    return path


class TestSparcDeckTail:
    """``doc/sparc.htm``: OPT(5:5) selects the calculation type, and the
    deck ends with the documented extra blocks in order — PULSE (quoted),
    FMin FMax, NRr, Rr (km) '/', NTout, Tout '/', then
    TSTART TMULT ALPHA BETA V, with ALPHA = BETA = V = 0 in uacpy's
    fixed scheme (the doc's own sample deck ends the same way)."""

    def test_output_mode_sits_in_topopt_position_five(self, tmp_path):
        lines = _write_sparc(tmp_path / 'sp.env').read_text().splitlines()
        assert _topopt(lines[3])[4] == 'S'

    def test_tail_blocks_in_documented_order(self, tmp_path):
        lines = _write_sparc(tmp_path / 'sp.env').read_text().splitlines()
        assert lines[-7] == "'P'"
        assert _floats(lines[-6]) == [50.0, 400.0]
        assert int(lines[-5]) == 2
        assert _floats(lines[-4]) == [1.0, 2.0]     # ranges in km
        assert lines[-4].rstrip().endswith('/')
        assert int(lines[-3]) == 100
        assert _floats(lines[-2]) == [0.0, 2.0]
        assert lines[-2].rstrip().endswith('/')
        assert _floats(lines[-1]) == [-0.1, 0.9, 0.0, 0.0, 0.0]


class TestReflectionTableDocExample:
    """``doc/ReflectionCoefficientFile.htm``: NTHETA then
    THETA (deg) RMAG RPHASE (deg) rows; angles are grazing angles. The
    document's own example table round-trips through the reader, which
    returns the phase in radians."""

    def test_the_documents_example_reads_back(self, tmp_path):
        from uacpy.io.refl_io import read_reflection_coefficient
        table = tmp_path / 'doc.brc'
        table.write_text("3\n"
                         "0.0   1.00  180.0\n"
                         "45.0  0.95  175.0\n"
                         "90.0  0.90  170.0\n")
        rc = read_reflection_coefficient(table)
        assert rc['n_pts'] == 3
        np.testing.assert_allclose(rc['theta'], [0.0, 45.0, 90.0])
        np.testing.assert_allclose(rc['R'], [1.00, 0.95, 0.90])
        np.testing.assert_allclose(rc['phi'],
                                   np.deg2rad([180.0, 175.0, 170.0]))


class TestAtiFileFollowsTheBtyContract:
    """``doc/ATI_BTY_File.htm``: 'The format is the same for top altimetry
    or bottom bathymetry' — TYPE(2:2) defaults to 'S' (short), R() is in
    km, and the range vector 'must increase strictly monotonically'. The
    .bty side is pinned elsewhere; this pins the .ati writer on the same
    contract.

    The writer takes that default by omitting TYPE(2:2) rather than spelling
    'S', which ``bdryMod.f90:179-181`` accepts as ``CASE ( 'S', '' )`` and
    AT's own decks use (``tests/ParaBot/ParaBot.bty`` is ``'C'``). One
    character is required, not merely permitted: ``atiType``/``btyType`` are
    ``CHARACTER(LEN=2)`` and ``bellhop.f90:535``/``:552`` test the whole
    string against ``'C'``, so a spelled-out ``'CS'`` never selects the
    curvilinear reflection geometry.
    """

    def test_ranges_are_written_in_km_with_short_type(self, tmp_path):
        from uacpy.io.bathy_io import write_ati_file
        path = tmp_path / 'a.ati'
        write_ati_file(path, np.array([[0.0, 2.0], [500.0, 2.0]]))
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        assert lines[0] == "'L'"
        assert int(lines[1]) == 2
        assert _floats(lines[2]) == [0.0, 2.0]
        assert _floats(lines[3]) == [0.5, 2.0]

    def test_non_monotonic_range_axis_is_refused(self, tmp_path):
        from uacpy.io.bathy_io import write_ati_file
        with pytest.raises(ConfigurationError, match='strictly increasing'):
            write_ati_file(tmp_path / 'a.ati',
                           np.array([[0.0, 2.0], [500.0, 2.0],
                                     [500.0, 1.0]]))


class TestElasticSubBottomReceiversArePartitioned:
    """``doc/index.htm`` (vendored AT tree): 'Internally KRAKENC replaces
    elastic layers by an equivalent reflection coefficent. For this
    reason, you cannot use KRAKENC to look at fields within the elastic
    layers.' (the misspelling is the original's; the same text appears as
    KRAKEN manual (2001) §4.2.1 ``notes.hlp``.) uacpy honours this by computing the field only at
    water-column depths and returning NaN (with a warning) below."""

    def _model_env(self):
        from uacpy.models import Kraken
        return Kraken(), _measured_env()

    def test_sub_bottom_depths_are_split_out_with_a_warning(self):
        model, env = self._model_env()
        rcv = uacpy.Receiver(depths=[50.0, 150.0], ranges=[1000.0])
        with pytest.warns(UserWarning, match='elastic sub-bottom'):
            compute_rcv, keep = model._partition_elastic_subbottom(
                env, rcv, True)
        np.testing.assert_array_equal(keep, [True, False])
        np.testing.assert_allclose(compute_rcv.depths, [50.0])

    def test_all_sub_bottom_yields_one_legal_compute_depth(self):
        # misc/SourceReceiverPositions.f90:212 ERROUTs on an empty receiver
        # vector, so the split substitutes a mid-water depth and the keep
        # mask discards its column afterwards.
        model, env = self._model_env()
        rcv = uacpy.Receiver(depths=[150.0], ranges=[1000.0])
        with pytest.warns(UserWarning, match='elastic sub-bottom'):
            compute_rcv, keep = model._partition_elastic_subbottom(
                env, rcv, True)
        assert not keep.any()
        assert len(np.atleast_1d(compute_rcv.depths)) == 1
        assert 0.0 < float(np.atleast_1d(compute_rcv.depths)[0]) < env.depth

    def test_water_column_receivers_pass_through_untouched(self):
        model, env = self._model_env()
        rcv = uacpy.Receiver(depths=[50.0], ranges=[1000.0])
        compute_rcv, keep = model._partition_elastic_subbottom(env, rcv, True)
        assert compute_rcv is rcv and keep is None


class TestFortranTitleQuoting:
    """Deck titles are Fortran quoted character literals read list-directed:
    an interior apostrophe ends the literal early, silently truncating the
    title (and bellhopcxx's C++ parser rejects the Fortran ``''`` escape),
    so apostrophes are STRIPPED, not doubled — at Environment construction
    and again at the write boundary, which closes the post-construction
    ``env.name = ...`` mutation path."""

    def _env(self, name="Pekeris's reef"):
        return Environment(name=name, bathymetry=100.0, ssp=1500.0,
                           bottom=BoundaryProperties(
                               acoustic_type='half-space', sound_speed=1700.0,
                               density=1.8, attenuation=0.5))

    def test_construction_strips_the_apostrophe(self):
        assert self._env().name == "Pekeriss reef"

    def test_at_env_header_re_sanitizes_a_mutated_name(self):
        import io as _io
        from uacpy.io.oalib_writer import write_header
        env = self._env()
        env.name = "O'Brien ridge"
        buf = _io.StringIO()
        write_header(buf, env,
                     uacpy.Source(frequencies=100.0, depths=25.0),
                     ssp_topopt='C', surface_type=BoundaryType.VACUUM)
        assert buf.getvalue().splitlines()[0] == "'OBrien ridge'"

    def test_bellhop_env_re_sanitizes_a_mutated_name(self, tmp_path):
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        env = self._env()
        env.name = "O'Brien ridge"
        path = tmp_path / 'apostrophe.env'
        write_bellhop_env_file(
            path, env,
            uacpy.Source(frequencies=100.0, depths=25.0),
            uacpy.Receiver(depths=[50.0], ranges=[1000.0]))
        assert path.read_text().splitlines()[0] == "'OBrien ridge'"


class TestMediaCountBound:
    """kraken/krakenc/scooter/sparc compile MaxMedium = 500
    (misc/sspMod.f90); a deck with more media dies as a bare Fortran fatal,
    so the env writers refuse first. 500 media (water + 499 layers) is the
    densest legal deck."""

    def _env(self, n_layers):
        from uacpy.core.environment import SeabedColumn, SedimentLayer
        layers = [SedimentLayer(thickness=1.0, sound_speed=1550.0,
                                density=1.5, attenuation=0.3)
                  for _ in range(n_layers)]
        bottom = SeabedColumn(
            layers=layers,
            halfspace=BoundaryProperties(acoustic_type='half-space',
                                         sound_speed=1800.0, density=2.0,
                                         attenuation=0.1))
        return Environment(name='stack', bathymetry=100.0, ssp=1500.0,
                           bottom=bottom)

    def _write(self, n_layers):
        import io as _io
        from uacpy.io.oalib_writer import write_header
        buf = _io.StringIO()
        write_header(buf, self._env(n_layers),
                     uacpy.Source(frequencies=100.0, depths=25.0),
                     ssp_topopt='C', surface_type=BoundaryType.VACUUM)
        return buf.getvalue()

    def test_500_media_is_the_densest_legal_deck(self):
        assert '\n500\n' in self._write(499)

    def test_501_media_raises_before_any_deck_is_written(self):
        with pytest.raises(ConfigurationError, match='MaxMedium = 500'):
            self._write(500)
