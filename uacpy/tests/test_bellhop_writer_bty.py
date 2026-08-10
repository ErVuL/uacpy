"""Bellhop env-writer boundary-file coverage.

Every range-dependent boundary feature needs a file to travel in, and the
``.env`` alone carries none of them. Three invariants: a range-dependent
bottom reaches Bellhop even when the bathymetry is flat (the ``.bty`` is the
only vehicle); the long-format ``.bty`` samples the union of the bathymetry
and property grids, so interior property breaks survive; and a single-sample
(constant-offset) altimetry still produces an ``.ati``.
"""

import numpy as np
import pytest

import uacpy
from uacpy.core.environment import Bottom
from uacpy.io.bellhop_writer import write_bellhop_env_file


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
    # 2-point bathymetry (0, 20 km) + property breaks at 8 and 16 km: the
    # .bty must carry the union of the grids, not blend the interior away.
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
    assert 8.0 in ranges_km and 16.0 in ranges_km
    assert cp_by_range[8.0] == pytest.approx(1550.0)
    assert cp_by_range[16.0] == pytest.approx(2000.0)


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
