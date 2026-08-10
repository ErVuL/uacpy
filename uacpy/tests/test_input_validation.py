"""Input-validation guards, grouped by the G1-G8 labels used in the section
headers below: monotonicity rejection (G1), range-coverage warnings (G3),
per-range receiver-depth checks (G4), the RAM Collins RD-bottom path (G2),
the range=0 SSP fallback in Bellhop's .env block (G6) and the acoustic_type
guard (G8).

Most cases reach only ``validate_inputs`` / ``_project_environment`` / a
writer, so they need no binary; the handful marked ``requires_binary``
construct a model (which resolves its executable) and a few of those do run
one. Whole file: a few seconds.
"""

import warnings

import numpy as np
import pytest

import uacpy
from uacpy.core.exceptions import ConfigurationError
from uacpy.models.bellhop import Bellhop
from uacpy.models.ram import RAM
from uacpy.models.scooter import Scooter
from uacpy.core.environment import (
    BoundaryProperties,
    Environment,
    SeabedColumn,
    Bottom,
    SedimentLayer,
    SoundSpeedProfile,
)


# --- G1 monotonicity -------------------------------------------------------

def test_ssp_depth_must_be_strictly_increasing():
    with pytest.raises(ConfigurationError, match="strictly increasing"):
        SoundSpeedProfile.from_pairs([(0, 1500), (10, 1490), (5, 1495)])


def test_ssp_ranges_must_be_strictly_increasing():
    depths = np.array([0.0, 100.0])
    data = np.array([[1500.0, 1490.0, 1500.0], [1480.0, 1470.0, 1480.0]])
    with pytest.raises(ConfigurationError, match="strictly increasing"):
        SoundSpeedProfile(
            depths=depths, data=data,
            ranges=np.array([0.0, 5000.0, 3000.0]),
        )


def test_ssp_duplicate_depths_rejected():
    with pytest.raises(ConfigurationError, match="strictly increasing"):
        SoundSpeedProfile.from_pairs([(0, 1500), (10, 1490), (10, 1480)])


def test_rd_bottom_ranges_must_be_strictly_increasing():
    with pytest.raises(ConfigurationError, match="strictly increasing"):
        Bottom.from_halfspaces(
            np.array([0.0, 5000.0, 3000.0]),
            sound_speed=np.array([1600.0, 1700.0, 1800.0]),
            density=np.array([1.5, 1.6, 1.7]),
            attenuation=np.array([0.3, 0.4, 0.5]),
            acoustic_type='half-space',
        )


def test_rd_bottom_shear_array_length_validated():
    # A mismatched explicit shear array must raise ConfigurationError at
    # construction (not a bare numpy ValueError later inside at()).
    with pytest.raises(ConfigurationError, match="shear_speed length"):
        Bottom.from_halfspaces(
            np.array([0.0, 1000.0, 2000.0]),
            sound_speed=np.array([1600.0, 1700.0, 1800.0]),
            density=np.array([1.5, 1.6, 1.7]),
            attenuation=np.array([0.3, 0.4, 0.5]),
            shear_speed=np.array([200.0, 300.0]),     # length 2 != 3
        )


def test_sediment_layer_rejects_negative_attenuation():
    with pytest.raises(ConfigurationError, match="attenuation must be .*non-negative"):
        SedimentLayer(thickness=5, sound_speed=1600, density=1.6, attenuation=-0.1)


def test_rd_layered_bottom_ranges_must_be_strictly_increasing():
    layer = SedimentLayer(thickness=5, sound_speed=1600, density=1.6, attenuation=0.4)
    hs = BoundaryProperties(acoustic_type='half-space',
                            sound_speed=1800, density=2.0, attenuation=0.1)
    lb = SeabedColumn(layers=[layer], halfspace=hs)
    with pytest.raises(ConfigurationError, match="strictly increasing"):
        Bottom.from_columns([lb, lb, lb], ranges=np.array([0.0, 1000.0, 500.0]))


def test_bathymetry_must_be_strictly_increasing():
    with pytest.raises(ConfigurationError, match="strictly increasing"):
        Environment(bathymetry=[(0.0, 100.0), (5000.0, 200.0), (3000.0, 150.0)])


def test_altimetry_must_be_strictly_increasing():
    with pytest.raises(ConfigurationError, match="strictly increasing"):
        Environment(
            bathymetry=100.0,
            altimetry=[(0.0, 0.0), (2000.0, 1.0), (1000.0, 0.5)],
        )


def test_receiver_grid_ranges_must_be_increasing():
    with pytest.raises(ConfigurationError, match="strictly increasing"):
        uacpy.Receiver(depths=np.array([10.0, 20.0]),
                       ranges=np.array([1000.0, 500.0]))


def test_receiver_line_rejected_at_construction():
    """``receiver_type='line'`` raises in the carrier, not at run() time.

    No model collapses the depth x range grid to paired samples, so the
    layout is refused as early as possible rather than after a model run.
    """
    with pytest.raises(ConfigurationError, match="receiver_type='line'"):
        uacpy.Receiver(
            depths=np.array([10.0, 50.0, 30.0]),
            ranges=np.array([1000.0, 2000.0, 1500.0]),
            receiver_type='line',
        )


# --- G8 acoustic_type ------------------------------------------------------

def test_acoustic_type_alias_accepted():
    BoundaryProperties(acoustic_type='halfspace')
    BoundaryProperties(acoustic_type='HALF-SPACE')


def test_acoustic_type_typo_rejected():
    with pytest.raises(ConfigurationError, match="not recognized"):
        BoundaryProperties(acoustic_type='vaccum')


def test_rd_bottom_acoustic_type_validated():
    with pytest.raises(ConfigurationError, match="not recognized"):
        Bottom.from_halfspaces(
            np.array([0.0, 1000.0]),
            sound_speed=np.array([1600.0, 1700.0]),
            density=np.array([1.5, 1.6]),
            attenuation=np.array([0.3, 0.4]),
            acoustic_type='spam-eggs',
        )


# --- G4 per-range receiver depth check -------------------------------------

@pytest.mark.requires_binary  # constructs a model (resolves its binary)
def test_per_range_receiver_below_shoaling_seafloor():
    bellhop = Bellhop()
    env = Environment(
        bathymetry=[(0.0, 200.0), (10_000.0, 50.0)],
        ssp=1500.0,
    )
    src = uacpy.Source(depths=10.0, frequencies=100.0)
    rcv = uacpy.Receiver(
        depths=np.array([100.0]),
        ranges=np.array([1000.0, 5000.0, 9000.0]),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bellhop.validate_inputs(env, src, rcv, run_mode=uacpy.RunMode.COHERENT_TL)
    msgs = [str(w.message) for w in caught]
    assert any("below the local seafloor" in m for m in msgs)


@pytest.mark.requires_binary  # constructs a model (resolves its binary)
def test_per_range_receiver_check_passes_when_under_seafloor():
    bellhop = Bellhop()
    env = Environment(
        bathymetry=[(0.0, 200.0), (10_000.0, 50.0)],
        ssp=1500.0,
    )
    src = uacpy.Source(depths=10.0, frequencies=100.0)
    rcv = uacpy.Receiver(
        depths=np.array([30.0]),
        ranges=np.array([1000.0, 5000.0, 9000.0]),
    )
    bellhop.validate_inputs(env, src, rcv, run_mode=uacpy.RunMode.COHERENT_TL)


# --- G3 range coverage warning ---------------------------------------------

@pytest.mark.requires_binary  # constructs a model (resolves its binary)
def test_warn_when_receiver_overruns_bathymetry():
    bellhop = Bellhop()
    env = Environment(
        bathymetry=[(0.0, 100.0), (5_000.0, 200.0)],
        ssp=1500.0,
    )
    src = uacpy.Source(depths=10.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]),
                         ranges=np.array([8_000.0]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bellhop.validate_inputs(env, src, rcv, run_mode=uacpy.RunMode.COHERENT_TL)
    msgs = [str(w.message) for w in caught]
    assert any("env.bathymetry" in m and "constant-extrapolated" in m for m in msgs)


@pytest.mark.requires_binary  # constructs a model (resolves its binary)
def test_warn_when_receiver_overruns_ssp_ranges():
    bellhop = Bellhop()
    ssp = SoundSpeedProfile.from_2d(
        depths=np.array([0.0, 100.0]),
        ranges=np.array([0.0, 2_000.0]),
        matrix=np.array([[1500.0, 1495.0], [1480.0, 1475.0]])
    )
    env = Environment(bathymetry=100.0, ssp=ssp)
    src = uacpy.Source(depths=10.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]),
                         ranges=np.array([5_000.0]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bellhop.validate_inputs(env, src, rcv, run_mode=uacpy.RunMode.COHERENT_TL)
    msgs = [str(w.message) for w in caught]
    assert any("env.ssp.ranges" in m for m in msgs)


# --- G2 RAM Collins RD-bottom warning --------------------------------------

def _elastic_rd_bottom(sound_speed, density, attenuation):
    """Two-column elastic bottom breaking at 2 km. Shear is held constant across
    the break: rams0.5 goes unstable on a 500 m/s shear column at this grid
    whether or not the bottom is range-dependent, which would mask the property
    under test."""
    return Bottom.from_halfspaces(
        np.array([0.0, 2_000.0]),
        sound_speed=np.array(sound_speed),
        density=np.array(density),
        attenuation=np.array(attenuation),
        shear_speed=np.array([400.0, 400.0]),
        acoustic_type='half-space',
    )


@pytest.mark.requires_binary  # constructs RAM (resolves its binary)
def test_ram_collins_threads_rd_bottom(tmp_path):
    """The Collins backends emit one ``ram.in`` profile section per range
    break, so a range-dependent bottom is modelled rather than reduced to its
    r=0 column. No 'range-0' warning may therefore be raised on this env."""
    env = Environment(bathymetry=100.0, ssp=1500.0,
                      bottom=_elastic_rd_bottom([1700.0, 1800.0], [1.7, 1.9],
                                                [0.5, 0.4]))
    assert env.has_elastic_bottom
    ram = RAM(work_dir=str(tmp_path), cleanup=False)
    assert ram.select_backend(env) == 'rams'
    src = uacpy.Source(depths=10.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]),
                         ranges=np.array([2_000.0]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        field = ram.run(env, src, rcv, run_mode=uacpy.RunMode.COHERENT_TL)
    p_rd = complex(np.asarray(field.data).ravel()[0])
    assert np.isfinite(p_rd)
    assert not any("range-0" in str(w.message) for w in caught)

    # rams writes six profile blocks per section, so bathymetry plus two
    # sections is 13 terminators. The section marker is the midpoint of the two
    # breakpoints, inside the 2 km march, and each section carries its own
    # compressional column.
    deck = (tmp_path / 'rams.in').read_text().splitlines()
    assert deck.count('-1 -1') == 1 + 6 * 2
    cut = deck.index('1000.000000')
    head, tail = deck[:cut], deck[cut:]
    assert any('1700.000000' in ln for ln in head)
    assert not any('1800.000000' in ln for ln in head)
    assert any('1800.000000' in ln for ln in tail)
    assert not any('1700.000000' in ln for ln in tail)

    # Reducing the bottom to its r=0 column would reproduce this field exactly.
    p_r0 = complex(np.asarray(RAM().run(
        Environment(bathymetry=100.0, ssp=1500.0,
                    bottom=_elastic_rd_bottom([1700.0, 1700.0], [1.7, 1.7],
                                              [0.5, 0.5])),
        src, rcv, run_mode=uacpy.RunMode.COHERENT_TL).data).ravel()[0])
    assert abs(p_rd - p_r0) > 0.05 * abs(p_r0)


def test_oalib_writer_drops_subresolution_layers():
    """A sediment layer thinner than the .env's .1f depth resolution is dropped,
    so the AT writer never emits a degenerate (top==bottom) zero-thickness
    medium, which makes Kraken/Scooter/Bounce fail. A fetched crustal column
    rescaled to fit a thin sediment cover can produce such a sliver, so the
    guard has to live in the writer rather than in the caller."""
    from uacpy.io.oalib_writer import writable_layers, _MIN_LAYER_THICKNESS_M
    hs = BoundaryProperties(acoustic_type='half-space', sound_speed=1800,
                            density=2.0, attenuation=0.1)
    lb = SeabedColumn(
        layers=[SedimentLayer(thickness=1e-4, sound_speed=1600, density=1.6,
                              attenuation=0.5),          # sub-resolution → dropped
                SedimentLayer(thickness=20.0, sound_speed=1700, density=1.8,
                              attenuation=0.4)],
        halfspace=hs)
    assert _MIN_LAYER_THICKNESS_M == 0.1
    kept = writable_layers(lb)
    assert [round(lyr.thickness, 1) for lyr in kept] == [20.0]


# --- G6 Bellhop .env range=0 SSP fallback ----------------------------------

def test_bellhop_env_ssp_block_uses_range_zero_profile(tmp_path):
    """The .env SSP block is the profile at r=0 even when the 2-D SSP carries
    no r=0 column: the grid here starts at 1000 m, so r=0 is the constant
    back-extrapolation of the first profile."""
    from uacpy.io.bellhop_writer import write_bellhop_env_file

    ssp = SoundSpeedProfile.from_2d(
        depths=np.array([0.0, 100.0]),
        ranges=np.array([1_000.0, 5_000.0]),
        matrix=np.array([[1500.0, 1480.0], [1490.0, 1470.0]])
    )
    env = Environment(bathymetry=100.0, ssp=ssp)
    src = uacpy.Source(depths=10.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]),
                         ranges=np.array([2_500.0]))
    env_path = tmp_path / 'test.env'
    write_bellhop_env_file(env_path, env, src, rcv)

    # AT's SSP block opens with the medium mesh line ``NMESH SIGMA ZMAX,``
    # (trailing comma, e.g. ``2  0.0  100.0,``) and then runs one
    # ``z c ... /`` record per depth, so the comma opens the block and the
    # first line without a ``/`` closes it.
    block = []
    in_ssp = False
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not in_ssp and stripped.endswith(",") and "0.0" in stripped:
            in_ssp = True
            continue
        if in_ssp:
            parts = stripped.split()
            if not parts or "/" not in line:
                break
            block.append((float(parts[0]), float(parts[1])))

    assert len(block) >= 2
    # Surface sound speed of the first profile (r = 1000 m), which constant
    # back-extrapolation carries to r=0 unchanged.
    assert block[0][1] == pytest.approx(1500.0, abs=1e-3)


# --- Positive-path tests: independent grids should reconcile cleanly -------

def test_ssp_eval_interpolates_off_grid_range():
    ssp = SoundSpeedProfile.from_2d(
        depths=np.array([0.0, 100.0]),
        ranges=np.array([0.0, 4_000.0, 10_000.0]),
        matrix=np.array([[1500.0, 1490.0, 1480.0],
                         [1480.0, 1470.0, 1460.0]])
    )
    sliced = ssp.eval(range=2_000.0)
    assert sliced.data[0, 0] == pytest.approx(1495.0)
    assert sliced.data[1, 0] == pytest.approx(1475.0)


def test_ssp_eval_clamps_beyond_last_range():
    ssp = SoundSpeedProfile.from_2d(
        depths=np.array([0.0, 100.0]),
        ranges=np.array([0.0, 4_000.0]),
        matrix=np.array([[1500.0, 1490.0], [1480.0, 1470.0]])
    )
    sliced = ssp.eval(range=10_000.0)
    assert sliced.data[0, 0] == pytest.approx(1490.0)
    assert sliced.data[1, 0] == pytest.approx(1470.0)


def test_rd_bottom_eval_interpolates_off_grid_range():
    rd = Bottom.from_halfspaces(
        np.array([0.0, 5_000.0]),
        sound_speed=np.array([1600.0, 1800.0]),
        density=np.array([1.5, 1.9]),
        attenuation=np.array([0.3, 0.5]),
        acoustic_type='half-space',
    )
    bp = rd.halfspace_at(range=2_500.0)
    assert bp.sound_speed == pytest.approx(1700.0)
    assert bp.density == pytest.approx(1.7)


def test_bathymetry_eval_interpolates_off_grid():
    env = Environment(bathymetry=[(0.0, 100.0), (10_000.0, 200.0)])
    assert env.bathymetry.eval(range=5_000.0) == pytest.approx(150.0)
    # Constant extrapolation past the last range.
    assert env.bathymetry.eval(range=20_000.0) == pytest.approx(200.0)


def test_independent_bathy_ssp_bottom_ranges_compose_ok():
    """Bathymetry, RD-SSP, and RD-bottom each have their own range axis
    of different lengths; the env constructs without complaint and
    everything is reachable via the lookup helpers."""
    bathy = [(0.0, 100.0), (2_000.0, 120.0), (8_000.0, 180.0)]
    ssp = SoundSpeedProfile.from_2d(
        depths=np.array([0.0, 200.0]),
        ranges=np.array([0.0, 5_000.0, 12_000.0]),
        matrix=np.array([[1500.0, 1495.0, 1490.0],
                         [1480.0, 1475.0, 1470.0]])
    )
    rd_bot = Bottom.from_halfspaces(
        np.array([0.0, 3_000.0, 6_000.0, 9_000.0]),
        sound_speed=np.array([1600.0, 1650.0, 1700.0, 1750.0]),
        density=np.array([1.5, 1.6, 1.7, 1.8]),
        attenuation=np.array([0.3, 0.35, 0.4, 0.45]),
        acoustic_type='half-space',
    )
    env = Environment(bathymetry=bathy, ssp=ssp, bottom=rd_bot)
    assert env.is_range_dependent
    assert env.bathymetry.eval(range=4_000.0) == pytest.approx(140.0)
    assert env.ssp.eval(range=4_000.0).data[0, 0] == pytest.approx(1496.0)
    assert env.bottom.halfspace_at(range=4_500.0).sound_speed == pytest.approx(1675.0)


def test_bty_long_format_uses_union_of_range_grids(tmp_path):
    """RD-bottom ranges that differ from the bathymetry ranges are emitted on
    the union grid — the bottom's own breakpoints survive exactly (with depth
    interpolated onto them) instead of being blended onto the bathy grid."""
    from uacpy.io.bathy_io import write_bty_long_format

    bathy = np.array([[0.0, 100.0],
                      [3_000.0, 130.0],
                      [9_000.0, 200.0]])
    rd_bot = Bottom.from_halfspaces(
        np.array([0.0, 6_000.0]),
        sound_speed=np.array([1600.0, 1800.0]),
        density=np.array([1.5, 1.9]),
        attenuation=np.array([0.3, 0.5]),
        acoustic_type='half-space',
    )
    out = tmp_path / 'test.bty'
    write_bty_long_format(out, bathy, rd_bot)
    lines = [ln.split() for ln in out.read_text().splitlines() if ln.strip()
             and not ln.strip().startswith("'")]
    assert int(lines[0][0]) == 4                    # union {0, 3, 6, 9} km
    rows = [list(map(float, row)) for row in lines[1:5]]
    by_range = {r[0]: r for r in rows}
    assert by_range[3.0][2] == pytest.approx(1700.0)    # midway 1600→1800
    assert by_range[6.0][2] == pytest.approx(1800.0)    # bottom's own break
    assert by_range[6.0][1] == pytest.approx(165.0)     # depth interpolated
    assert by_range[9.0][2] == pytest.approx(1800.0)    # constant-extended


@pytest.mark.requires_binary  # constructs Scooter/Kraken/Bellhop (resolves their binaries)
def test_receiver_depth_accepted_across_models_harmonized():
    """A below-seafloor receiver never raises — it is accepted on every
    model, returning that model's below-domain value. Within its resolvable
    depth a model is silent; below it, one harmonized warning fires. Only
    the source (an input that must sit in the medium) is a hard error."""
    from uacpy.models.scooter import Scooter
    from uacpy.core.exceptions import InvalidDepthError

    ssp = SoundSpeedProfile.from_isovelocity(100.0, 1500.0)
    bottom = SeabedColumn(
        layers=[SedimentLayer(thickness=50.0, sound_speed=1600.0,
                              density=1.5, attenuation=0.5)],
        halfspace=BoundaryProperties(acoustic_type='half-space',
                                     sound_speed=1800.0, density=2.0,
                                     attenuation=0.8),
    )
    env = Environment(bathymetry=100.0, ssp=ssp, bottom=bottom)
    src = uacpy.Source(depths=20.0, frequencies=50.0)

    # Solvers that mesh through the sediment (Scooter, the Kraken family)
    # resolve a 130 m receiver in the sediment (env.depth=100, media=150)
    # → accepted, no warning.
    from uacpy.models.kraken import Kraken
    for model in (Scooter(), Kraken()):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.validate_inputs(
                env, src,
                uacpy.Receiver(depths=np.array([50.0, 130.0]),
                               ranges=np.array([500.0, 1000.0])),
                run_mode=uacpy.RunMode.COHERENT_TL,
            )
        assert not any("resolvable depth" in str(w.message) for w in caught)

    # A ray model stops at the seafloor: the same 130 m receiver is still
    # accepted, but warns that the result reflects below-domain behaviour.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Bellhop().validate_inputs(
            env, src,
            uacpy.Receiver(depths=np.array([130.0]), ranges=np.array([500.0])),
        )
    assert any("below the model's resolvable depth" in str(w.message)
               for w in caught)

    # The source, by contrast, must lie within the resolvable medium.
    with pytest.raises(InvalidDepthError):
        Bellhop().validate_inputs(
            env, uacpy.Source(depths=130.0, frequencies=50.0),
            uacpy.Receiver(depths=np.array([50.0]), ranges=np.array([500.0])),
        )


@pytest.mark.requires_binary  # constructs Scooter (resolves its binary)
def test_scooter_collapses_rd_env_with_warning():
    """A model without RD support collapses the env and emits one warning per
    dropped axis; the env returned by ``_project_environment`` is
    range-independent regardless of input shape.

    Scooter is the vehicle because it is a range-independent FFP. Kraken
    segments RD natively via field.exe, so it would not collapse here."""
    scooter = Scooter()
    ssp = SoundSpeedProfile.from_2d(
        depths=np.array([0.0, 100.0]),
        ranges=np.array([0.0, 5_000.0]),
        matrix=np.array([[1500.0, 1480.0], [1490.0, 1470.0]])
    )
    env = Environment(
        bathymetry=[(0.0, 100.0), (5_000.0, 200.0)],
        ssp=ssp,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        projected = scooter._project_environment(env)
    assert not projected.is_range_dependent
    text = " ".join(str(w.message) for w in caught)
    assert "range-dependent bathymetry" in text
    assert "range-dependent SSP" in text


@pytest.mark.requires_binary  # constructs RAM (resolves its binary)
def test_per_range_receiver_below_seafloor_emits_warning_not_error():
    """RAM accepts receivers below the local seafloor; G4 should warn,
    not raise."""
    env = Environment(
        bathymetry=[(0.0, 200.0), (10_000.0, 50.0)],
        ssp=1500.0,
    )
    src = uacpy.Source(depths=10.0, frequencies=100.0)
    rcv = uacpy.Receiver(
        depths=np.array([80.0]),
        ranges=np.array([2_000.0, 9_000.0]),
    )
    ram = RAM()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ram.validate_inputs(env, src, rcv, run_mode=uacpy.RunMode.COHERENT_TL)
    assert any("below the local seafloor" in str(w.message) for w in caught)


def test_kraken_segmentation_unions_distinct_axes():
    """Kraken builds its segment list from the union of bathy / SSP
    / bottom change-points, so a bathy with 3 ranges and an SSP with 5
    ranges should yield at least 5 segments."""
    from uacpy.models._segmentation import segment_environment_by_range

    ssp = SoundSpeedProfile.from_2d(
        depths=np.array([0.0, 200.0]),
        ranges=np.array([0.0, 2_000.0, 4_000.0, 6_000.0, 10_000.0]),
        matrix=np.tile(np.array([[1500.0], [1480.0]]), (1, 5))
    )
    env = Environment(
        bathymetry=[(0.0, 100.0), (5_000.0, 150.0), (10_000.0, 200.0)],
        ssp=ssp,
    )
    segments = segment_environment_by_range(env, max_segment_length=20_000)
    seg_ranges = [r for r, _ in segments]
    for rk in (0.0, 2_000.0, 4_000.0, 5_000.0, 6_000.0, 10_000.0):
        assert any(abs(r - rk) < 1.0 for r in seg_ranges), (
            f"missing union point {rk} in {seg_ranges}"
        )


def test_bellhop_quad_ssp_emits_unchanged_ssp_file(tmp_path):
    """Bellhop should pass ssp.ranges/.data through verbatim to .ssp,
    independent of bathymetry / receiver grids — plus one prepended
    negative-range guard column so back-scattered rays do not trip
    bellhopcuda's BHC_ERR_OUTSIDE_SSP (x < Seg.r[0]) range-box check."""
    from uacpy.io.bellhop_writer import write_bellhop_env_file

    ssp = SoundSpeedProfile.from_2d(
        depths=np.array([0.0, 100.0]),
        ranges=np.array([0.0, 1_000.0, 5_000.0]),
        matrix=np.array([[1500.0, 1495.0, 1485.0],
                         [1480.0, 1475.0, 1465.0]])
    )
    env = Environment(bathymetry=100.0, ssp=ssp)
    src = uacpy.Source(depths=10.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]),
                         ranges=np.array([2_000.0]))
    env_path = tmp_path / 'rdssp.env'
    write_bellhop_env_file(env_path, env, src, rcv, interp_ssp='quad')
    ssp_path = env_path.with_suffix('.ssp')
    assert ssp_path.exists()

    lines = ssp_path.read_text().splitlines()
    # AT/bellhopcuda LDIFile expects Npts and the range vector on
    # separate records (one line each), then one line per depth row.
    # Npts == 3 real columns + 1 prepended guard column.
    assert lines[0].strip() == '4', (
        f"line 1 must contain only Npts; got {lines[0]!r}"
    )
    ranges_km = list(map(float, lines[1].split()))
    # r_box = 1.2 * 2000 m = 2400 m -> guard at -1.1 * r_box = -2.64 km.
    assert ranges_km[0] == pytest.approx(-2.64)
    assert ranges_km[1:] == [0.0, 1.0, 5.0]
    # Two depths -> two SSP rows; the guard column duplicates the first
    # real profile, the rest pass through verbatim.
    assert len(lines) >= 4
    row0 = list(map(float, lines[2].split()))
    row1 = list(map(float, lines[3].split()))
    assert row0 == [1500.0, 1500.0, 1495.0, 1485.0]
    assert row1 == [1480.0, 1480.0, 1475.0, 1465.0]


# ──────────────────────────────────────────────────────────────────────
# Unknown run() kwargs → TypeError (no concrete ``run()`` takes
# ``**kwargs``, so Python rejects them at the call site).
# ──────────────────────────────────────────────────────────────────────


_OASES_MODEL_NAMES = {'OAST', 'OASN', 'OASR', 'OASP'}


def _all_concrete_model_params():
    from uacpy.models.bellhop import Bellhop
    from uacpy.models.bounce import Bounce
    from uacpy.models.kraken import Kraken
    from uacpy.models.oases import OAST, OASN, OASR, OASP
    from uacpy.models.ram import RAM
    from uacpy.models.scooter import Scooter
    from uacpy.models.sparc import SPARC
    classes = [
        Bellhop, Bounce,
        Kraken,
        OAST, OASN, OASR, OASP,
        RAM, Scooter, SPARC,
    ]
    # Every model resolves (and existence-checks) its binary in __init__, so
    # constructing one needs that binary — requires_binary for all, plus
    # requires_oases for the separately-licensed OASES family.
    params = []
    for cls in classes:
        marks = [pytest.mark.requires_binary]
        if cls.__name__ in _OASES_MODEL_NAMES:
            marks.append(pytest.mark.requires_oases)
        params.append(pytest.param(cls, id=cls.__name__, marks=marks))
    return params


@pytest.mark.parametrize("model_cls", _all_concrete_model_params())
def test_run_rejects_unknown_kwarg(model_cls):
    """Every concrete ``run()`` declares its full keyword set; unknown
    names raise :class:`TypeError` at the call site. Constructing the model
    resolves its binary, so this carries ``requires_binary`` (and
    ``requires_oases`` for the OASES models).
    """
    env = uacpy.Environment(name='unknown-kwarg', bathymetry=100.0, ssp=1500.0)
    src = uacpy.Source(depths=50.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=[25.0, 50.0], ranges=[1000.0, 2000.0])

    model = model_cls()
    with pytest.raises(TypeError, match=r"unexpected keyword argument"):
        model.run(env, src, rcv, totally_bogus_kwarg=1)


def test_generate_sea_surface_rejects_nonpositive_range():
    from uacpy import generate_sea_surface
    for bad in (0.0, -100.0):
        with pytest.raises(ConfigurationError, match="max_range"):
            generate_sea_surface(bad)


@pytest.mark.parametrize("bad", ['deep', object(), {'a': 1}])
def test_bathymetry_nonnumeric_is_typed(bad):
    with pytest.raises(ConfigurationError):
        uacpy.Environment(bathymetry=bad, ssp=1500.0)


def test_altimetry_nonnumeric_is_typed():
    with pytest.raises(ConfigurationError):
        uacpy.Environment(bathymetry=100.0, ssp=1500.0, altimetry='wavy')


def test_surface_source_warns_for_field_runs():
    """A source at z=0 sits on the pressure-release surface (field ~0) — a
    field run warns; a positive depth does not."""
    env = uacpy.Environment(
        bathymetry=100.0, ssp=1500.0,
        bottom=uacpy.BoundaryProperties(sound_speed=1600.0, density=1.5,
                                        attenuation=0.5))
    rcv = uacpy.Receiver(depths=[50.0], ranges=[2000.0])
    with pytest.warns(UserWarning, match="pressure-release sea surface"):
        Bellhop().compute_tl(env, uacpy.Source(depths=0.0, frequencies=200.0), rcv)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        Bellhop().compute_tl(env, uacpy.Source(depths=10.0, frequencies=200.0), rcv)


# --- Silent-coercion guards: extrapolation and container types -------------
def test_get_sound_speed_warns_on_extrapolation():
    """Depths outside the (bathymetry-extended) SSP are constant-extrapolated
    with a UserWarning, not silently."""
    ssp = SoundSpeedProfile(depths=[0, 50, 100], data=[1500, 1490, 1480])
    env = Environment(ssp=ssp, bathymetry=200.0)  # SSP extended to 200 m
    with pytest.warns(UserWarning, match="constant-extrapolated"):
        assert float(env.get_sound_speed(250)[0]) == 1480.0
    with pytest.warns(UserWarning, match="constant-extrapolated"):
        assert float(env.get_sound_speed(-10)[0]) == 1500.0
    # in-range query must not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        env.get_sound_speed(75)


def test_francois_garrison_accepts_list_pH():
    """pH as a Python list must not raise a bare TypeError (it is coerced)."""
    from uacpy.core.absorption import francois_garrison_db_per_km
    out = francois_garrison_db_per_km(10000, 10, 35, [8.0, 8.1], 100)
    out = np.atleast_1d(np.asarray(out, dtype=float))
    assert out.shape == (2,) and np.all(out > 0)


@pytest.mark.parametrize('model_name', ['Bellhop', 'Kraken', 'Scooter', 'RAM'])
def test_receiver_type_line_is_rejected_not_silently_gridded(model_name):
    """``receiver_type='line'`` must raise, not return the full grid.

    No model's result assembly collapses depth x range to the paired samples,
    so the carrier refuses the layout outright; the documented ``'grid'``
    workaround still runs on every model.
    """
    import uacpy
    from uacpy.core.exceptions import ConfigurationError

    env = uacpy.Environment(
        name='p', bathymetry=200.0, ssp=1500.0,
        bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                        sound_speed=1800.0, density=1.8,
                                        attenuation=0.5))
    src = uacpy.Source(depths=50.0, frequencies=100.0)
    d = np.array([60.0, 90.0, 120.0])
    r = np.array([1000.0, 2000.0, 3000.0])

    model = getattr(uacpy, model_name)(verbose=False)
    with pytest.raises(ConfigurationError, match="receiver_type='line'"):
        uacpy.Receiver(depths=d, ranges=r, receiver_type='line')
    # The documented workaround still works.
    tl = np.asarray(model.run(env, src,
                              uacpy.Receiver(depths=d, ranges=r)).tl)
    i = np.arange(len(d))
    assert tl[i, i].shape == (3,)
