"""Fast unit tests for the input-validation guards added for G1-G8.

These tests cover monotonicity rejection, range-coverage warnings,
per-range receiver-depth checks, the range=0 SSP fallback in Bellhop's
.env block, the acoustic_type guard, and the RAM Collins RD-bottom
warning. They are kept light (no binary execution) so the full file
runs in under a second.
"""

import warnings

import numpy as np
import pytest

import uacpy
from uacpy.models.bellhop import Bellhop
from uacpy.models.kraken import Kraken
from uacpy.models.ram import RAM
from uacpy.core.environment import (
    BoundaryProperties,
    Environment,
    LayeredBottom,
    RangeDependentBottom,
    RangeDependentLayeredBottom,
    SedimentLayer,
    SoundSpeedProfile,
)


# --- G1 monotonicity -------------------------------------------------------

def test_ssp_depth_must_be_strictly_increasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        SoundSpeedProfile.from_pairs([(0, 1500), (10, 1490), (5, 1495)])


def test_ssp_ranges_must_be_strictly_increasing():
    depths = np.array([0.0, 100.0])
    data = np.array([[1500.0, 1490.0, 1500.0], [1480.0, 1470.0, 1480.0]])
    with pytest.raises(ValueError, match="strictly increasing"):
        SoundSpeedProfile(
            depths=depths, data=data,
            ranges=np.array([0.0, 5000.0, 3000.0]),
        )


def test_ssp_duplicate_depths_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        SoundSpeedProfile.from_pairs([(0, 1500), (10, 1490), (10, 1480)])


def test_rd_bottom_ranges_must_be_strictly_increasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        RangeDependentBottom(
            ranges=np.array([0.0, 5000.0, 3000.0]),
            sound_speed=np.array([1600.0, 1700.0, 1800.0]),
            density=np.array([1.5, 1.6, 1.7]),
            attenuation=np.array([0.3, 0.4, 0.5]),
            acoustic_type='half-space',
        )


def test_rd_layered_bottom_ranges_must_be_strictly_increasing():
    layer = SedimentLayer(thickness=5, sound_speed=1600, density=1.6, attenuation=0.4)
    hs = BoundaryProperties(acoustic_type='half-space',
                            sound_speed=1800, density=2.0, attenuation=0.1)
    lb = LayeredBottom(layers=[layer], halfspace=hs)
    with pytest.raises(ValueError, match="strictly increasing"):
        RangeDependentLayeredBottom(
            ranges=np.array([0.0, 1000.0, 500.0]),
            profiles=[lb, lb, lb],
        )


def test_bathymetry_must_be_strictly_increasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        Environment(bathymetry=[(0.0, 100.0), (5000.0, 200.0), (3000.0, 150.0)])


def test_altimetry_must_be_strictly_increasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        Environment(
            bathymetry=100.0,
            altimetry=[(0.0, 0.0), (2000.0, 1.0), (1000.0, 0.5)],
        )


def test_receiver_grid_ranges_must_be_increasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        uacpy.Receiver(depths=np.array([10.0, 20.0]),
                       ranges=np.array([1000.0, 500.0]))


def test_receiver_line_axes_paired_not_sorted():
    rcv = uacpy.Receiver(
        depths=np.array([10.0, 50.0, 30.0]),
        ranges=np.array([1000.0, 2000.0, 1500.0]),
        receiver_type='line',
    )
    assert rcv.depths.tolist() == [10.0, 50.0, 30.0]


# --- G8 acoustic_type ------------------------------------------------------

def test_acoustic_type_alias_accepted():
    BoundaryProperties(acoustic_type='halfspace')
    BoundaryProperties(acoustic_type='HALF-SPACE')


def test_acoustic_type_typo_rejected():
    with pytest.raises(ValueError, match="not recognized"):
        BoundaryProperties(acoustic_type='vaccum')


def test_rd_bottom_acoustic_type_validated():
    with pytest.raises(ValueError, match="not recognized"):
        RangeDependentBottom(
            ranges=np.array([0.0, 1000.0]),
            sound_speed=np.array([1600.0, 1700.0]),
            density=np.array([1.5, 1.6]),
            attenuation=np.array([0.3, 0.4]),
            acoustic_type='spam-eggs',
        )


# --- G4 per-range receiver depth check -------------------------------------

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

def test_ram_collins_warns_on_rd_bottom():
    bp = BoundaryProperties(acoustic_type='half-space',
                            sound_speed=1700, density=1.7,
                            attenuation=0.5, shear_speed=400.0)
    rd_bot = RangeDependentBottom(
        ranges=np.array([0.0, 5_000.0]),
        sound_speed=np.array([1700.0, 1800.0]),
        density=np.array([1.7, 1.9]),
        attenuation=np.array([0.5, 0.4]),
        shear_speed=np.array([400.0, 500.0]),
        acoustic_type='half-space',
    )
    env = Environment(bathymetry=100.0, ssp=1500.0, bottom=rd_bot)
    assert env.has_elastic_bottom()
    ram = RAM()
    assert ram.select_backend(env) == 'rams'
    src = uacpy.Source(depths=10.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]),
                         ranges=np.array([2_000.0]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            ram.run(env, src, rcv, run_mode=uacpy.RunMode.COHERENT_TL)
        except Exception:
            pass
    msgs = [str(w.message) for w in caught]
    assert any("range-0 bottom geoacoustics" in m for m in msgs)
    # Use the unused halfspace to keep flake8 happy.
    assert bp.sound_speed == 1700


# --- G6 Bellhop .env range=0 SSP fallback ----------------------------------

def test_bellhop_env_ssp_block_uses_range_zero_profile(tmp_path):
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
    expected_top = float(np.interp(0.0, ssp.ranges, ssp.data[0, :]))
    assert abs(block[0][1] - expected_top) < 1e-3


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
    rd = RangeDependentBottom(
        ranges=np.array([0.0, 5_000.0]),
        sound_speed=np.array([1600.0, 1800.0]),
        density=np.array([1.5, 1.9]),
        attenuation=np.array([0.3, 0.5]),
        acoustic_type='half-space',
    )
    bp = rd.eval(range=2_500.0)
    assert bp.sound_speed == pytest.approx(1700.0)
    assert bp.density == pytest.approx(1.7)


def test_bathymetry_at_range_interpolates_off_grid():
    env = Environment(bathymetry=[(0.0, 100.0), (10_000.0, 200.0)])
    assert env.bathymetry_at_range(5_000.0)[0] == pytest.approx(150.0)
    # Constant extrapolation past the last range.
    assert env.bathymetry_at_range(20_000.0)[0] == pytest.approx(200.0)


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
    rd_bot = RangeDependentBottom(
        ranges=np.array([0.0, 3_000.0, 6_000.0, 9_000.0]),
        sound_speed=np.array([1600.0, 1650.0, 1700.0, 1750.0]),
        density=np.array([1.5, 1.6, 1.7, 1.8]),
        attenuation=np.array([0.3, 0.35, 0.4, 0.45]),
        acoustic_type='half-space',
    )
    env = Environment(bathymetry=bathy, ssp=ssp, bottom=rd_bot)
    assert env.is_range_dependent
    assert env.bathymetry_at_range(4_000.0)[0] == pytest.approx(140.0)
    assert env.ssp.eval(range=4_000.0).data[0, 0] == pytest.approx(1496.0)
    assert env.bottom.eval(range=4_500.0).sound_speed == pytest.approx(1675.0)


def test_bty_long_format_interpolates_bottom_onto_bathy_ranges(tmp_path):
    """RD-bottom whose ranges differ from bathymetry ranges should be
    silently resampled onto the bathymetry range grid by the writer."""
    from uacpy.io.bathy_io import write_bty_long_format

    bathy = np.array([[0.0, 100.0],
                      [3_000.0, 130.0],
                      [9_000.0, 200.0]])
    rd_bot = RangeDependentBottom(
        ranges=np.array([0.0, 6_000.0]),
        sound_speed=np.array([1600.0, 1800.0]),
        density=np.array([1.5, 1.9]),
        attenuation=np.array([0.3, 0.5]),
        acoustic_type='half-space',
    )
    out = tmp_path / 'test.bty'
    write_bty_long_format(out, bathy, rd_bot)
    lines = [ln.split() for ln in out.read_text().splitlines() if ln.strip()
             and not ln.strip().startswith("'")]
    assert int(lines[0][0]) == 3
    rows = [list(map(float, row)) for row in lines[1:4]]
    assert rows[0][2] == pytest.approx(1600.0)
    assert rows[1][2] == pytest.approx(1700.0)
    assert rows[2][2] == pytest.approx(1800.0)


def test_kraken_collapses_rd_env_with_warning(simple_env):
    """A model without RD support should collapse the env and emit one
    warning per dropped axis — the env returned by _project_environment
    must be range-independent regardless of the input shape."""
    kraken = Kraken()
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
        projected = kraken._project_environment(env)
    assert not projected.is_range_dependent
    text = " ".join(str(w.message) for w in caught)
    assert "range-dependent bathymetry" in text
    assert "range-dependent SSP" in text
    # the autouse 'simple_env' fixture is referenced to satisfy pytest
    assert simple_env.depth > 0


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
    """KrakenField builds its segment list from the union of bathy / SSP
    / bottom change-points, so a bathy with 3 ranges and an SSP with 5
    ranges should yield at least 5 segments."""
    from uacpy.models.coupled_modes import segment_environment_by_range

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
    independent of bathymetry / receiver grids."""
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
    contents = ssp_path.read_text().split()
    n_profiles = int(contents[0])
    assert n_profiles == 3
    ranges_km = list(map(float, contents[1:1 + n_profiles]))
    assert ranges_km == [0.0, 1.0, 5.0]
