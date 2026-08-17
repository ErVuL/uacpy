"""Tests for the WOA23 sound-speed fetch (uacpy.data.sound_speed).

Unit tests stub the OPeNDAP HTTP layer with canned DAP ``.ascii`` bodies so
they run fully offline; one ``requires_network`` test hits the live NCEI
THREDDS server (auto-skipped offline).
"""

import numpy as np
import pytest

import uacpy
from uacpy.core.environment import SoundSpeedProfile
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import sound_speed as ss

FILL = '9.96921E36'
SEP = '-' * 45


def _axis_body(name, values):
    body = '\n'.join(str(v) for v in [
        "Dataset {", f"    Float32 {name}[{name} = {len(values)}];",
        "} file;", SEP, f"{name}[{len(values)}]",
        ', '.join(str(v) for v in values),
    ])
    return body.encode()


def _data_body(var, values):
    rows = [f"[0][{k}][0], {v}" for k, v in enumerate(values)]
    body = '\n'.join([
        "Dataset {", "} file;", SEP,
        f"{var}.{var}[1][{len(values)}][1][1]", *rows, "",
    ])
    return body.encode()


def _make_fake_http(columns):
    """Build a fake ``http_get`` from ``{(var, period): (depths, values)}``.

    ``var`` is ``'t'``/``'s'``, ``period`` is the WOA code (0 annual, 1-12).
    Dispatches on the URL: ``?depth`` → axis, ``?t_an[``/``?s_an[`` → data.
    """
    def fake_http(url, *, timeout, verbose, source='data'):
        # filename .../woa23_decav_t07_01.nc.ascii?...
        fname = url.split('/')[-1]
        var = 't' if "_t" in fname.split('.')[0] else 's'
        period = int(fname.split('_')[2][1:3])
        depths, values = columns[(var, period)]
        if '.ascii?depth' in url:
            return _axis_body('depth', depths)
        return _data_body(f"{var}_an", values)
    return fake_http


# A short, realistic-ish column: warm surface, cooler at depth.
_Z = [0.0, 50.0, 100.0, 500.0, 1000.0]
_ANNUAL = {
    ('t', 0): (_Z, [18.0, 16.0, 13.0, 8.0, 5.0]),
    ('s', 0): (_Z, [36.0, 36.1, 36.2, 35.5, 35.0]),
}


@pytest.fixture
def annual_http(monkeypatch):
    monkeypatch.setattr(ss, 'http_get', _make_fake_http(_ANNUAL))


def test_fetch_ssp_builds_profile(annual_http):
    profile = ss.fetch_ssp((30.5, -40.5))
    assert isinstance(profile, SoundSpeedProfile)
    np.testing.assert_allclose(profile.depths, _Z)
    # Sound speed is physical and decreases into the cold layer then rises.
    assert np.all((1450 < profile.data[:, 0]) & (profile.data[:, 0] < 1560))
    env = uacpy.Environment(name='woa', bathymetry=1000.0, ssp=profile)
    assert env.ssp.n_depths == 5


def test_unesco_vs_delgrosso_close(annual_http):
    cu = ss.fetch_ssp((30.5, -40.5), formula='unesco').data[:, 0]
    cd = ss.fetch_ssp((30.5, -40.5), formula='delgrosso').data[:, 0]
    assert np.allclose(cu, cd, atol=1.0)  # standard formulas agree ~1 m/s


def test_wet_cell_fallback_stamps_the_cell_actually_used(monkeypatch):
    # The requested cell (30.5, -40.5) is dry and the column comes from its
    # eastern neighbour; provenance must carry that neighbour's centre so
    # offset_km measures the real hop, not 0 to the dry cell.
    wet = (120, 140)                              # cell centre (30.5, -39.5)

    def fake_get_column(source, period, lat_idx, lon_idx, **kw):
        if (lat_idx, lon_idx) == wet:
            return (np.asarray(_Z), np.asarray([18.0, 16.0, 13.0, 8.0, 5.0]),
                    np.asarray([36.0, 36.1, 36.2, 35.5, 35.0]))
        return np.array([]), np.array([]), np.array([])

    monkeypatch.setattr(ss, '_get_column', fake_get_column)
    with pytest.warns(UserWarning, match='closest wet cell'):
        profile = ss.fetch_ssp((30.5, -40.5))
    prov = profile.data_sources[0]
    assert prov.requested_point == (30.5, -40.5)
    assert prov.data_point == (30.5, -39.5)
    assert prov.offset_km == pytest.approx(95.8, abs=0.5)  # 1 deg lon at 30.5N


def test_on_centre_request_stamps_zero_offset(monkeypatch):
    def fake_get_column(source, period, lat_idx, lon_idx, **kw):
        return (np.asarray(_Z), np.asarray([18.0, 16.0, 13.0, 8.0, 5.0]),
                np.asarray([36.0, 36.1, 36.2, 35.5, 35.0]))

    monkeypatch.setattr(ss, '_get_column', fake_get_column)
    prov = ss.fetch_ssp((30.5, -40.5)).data_sources[0]
    assert prov.data_point == (30.5, -40.5)
    assert prov.offset_km == 0.0


def test_land_cell_raises(monkeypatch):
    land = {
        ('t', 0): (_Z, [FILL] * 5),
        ('s', 0): (_Z, [FILL] * 5),
    }
    monkeypatch.setattr(ss, 'http_get', _make_fake_http(land))
    with pytest.raises(DataFetchError, match='no water-column'):
        ss.fetch_ssp((40.0, 0.0))


def test_seafloor_truncation(monkeypatch):
    # Valid to 100 m, filled below → profile keeps 3 levels.
    cols = {
        ('t', 0): (_Z, [18.0, 16.0, 13.0, FILL, FILL]),
        ('s', 0): (_Z, [36.0, 36.1, 36.2, FILL, FILL]),
    }
    monkeypatch.setattr(ss, 'http_get', _make_fake_http(cols))
    profile = ss.fetch_ssp((30.5, -40.5))
    assert profile.depths.tolist() == [0.0, 50.0, 100.0]


def test_month_splices_annual_below_cap(monkeypatch):
    # Monthly fields cap at 1500 m; annual extends to 3000 m. Splice fires
    # because the July column reached the 1500 m cap (deeper water below).
    z_month = [0.0, 50.0, 1500.0]
    z_annual = [0.0, 50.0, 1500.0, 2000.0, 3000.0]
    cols = {
        ('t', 7): (z_month, [20.0, 17.0, 6.0]),
        ('s', 7): (z_month, [36.0, 36.1, 35.0]),
        ('t', 0): (z_annual, [18.0, 16.0, 5.0, 3.0, 2.0]),
        ('s', 0): (z_annual, [36.0, 36.1, 35.0, 34.9, 34.8]),
    }
    monkeypatch.setattr(ss, 'http_get', _make_fake_http(cols))
    profile = ss.fetch_ssp((30.5, -40.5), month=7)
    # Upper part from July, deep part appended from annual (>1500 m).
    assert profile.depths.tolist() == [0.0, 50.0, 1500.0, 2000.0, 3000.0]
    assert np.all(np.diff(profile.depths) > 0)


def test_month_no_splice_when_seafloor_above_cap(monkeypatch):
    # Shallow site: July column ends at 200 m (seafloor), below the 1500 m
    # cap, so no annual splice and no second fetch is needed.
    z_month = [0.0, 50.0, 200.0]
    cols = {
        ('t', 7): (z_month, [20.0, 17.0, 14.0]),
        ('s', 7): (z_month, [36.0, 36.1, 36.0]),
    }
    monkeypatch.setattr(ss, 'http_get', _make_fake_http(cols))
    profile = ss.fetch_ssp((30.5, -40.5), month=7)
    assert profile.depths.tolist() == [0.0, 50.0, 200.0]


def test_date_selects_month(monkeypatch):
    captured = {}
    real = _make_fake_http(_ANNUAL | {
        ('t', 6): (_Z, [19.0, 16.0, 13.0, 8.0, 5.0]),
        ('s', 6): (_Z, [36.0, 36.1, 36.2, 35.5, 35.0]),
    })

    def spy(url, *, timeout, verbose, source='data'):
        captured.setdefault('periods', set()).add(url.split('_')[2])
        return real(url, timeout=timeout, verbose=verbose, source=source)

    monkeypatch.setattr(ss, 'http_get', spy)
    ss.fetch_ssp((30.5, -40.5), date='2026-06-14')
    assert 't06' in captured['periods']  # June file requested


def test_date_and_month_mutually_exclusive():
    with pytest.raises(ConfigurationError, match='not both'):
        ss.fetch_ssp((0.0, 0.0), date='2026-01-01', month=3)


def test_bad_date_raises():
    with pytest.raises(ConfigurationError, match='parse date'):
        ss.fetch_ssp((0.0, 0.0), date='2026-13-99')


@pytest.mark.parametrize('bad', [0, 13, -1])
def test_bad_month_raises(bad):
    with pytest.raises(ConfigurationError, match='month'):
        ss.fetch_ssp((0.0, 0.0), month=bad)


def test_bad_formula_and_resolution_raise():
    with pytest.raises(ConfigurationError, match='formula'):
        ss.fetch_ssp((0.0, 0.0), formula='nope')
    with pytest.raises(ConfigurationError, match='resolution'):
        ss.fetch_ssp((0.0, 0.0), resolution='2.00')


def test_grid_index_and_lon_normalization():
    # 1° grid centres at X.5; lon given in [0,360] must normalize.
    i, j, lat_c, lon_c = ss._grid_index(30.5, 319.5, '1.00')  # 319.5 == -40.5
    assert (lat_c, lon_c) == (30.5, -40.5)
    assert (i, j) == (120, 139)


def test_depth_to_pressure_dbar():
    # ~1 dbar per metre, with the latitude/compressibility correction.
    from uacpy.data._geo import depth_to_pressure_dbar
    p = depth_to_pressure_dbar(np.array([0.0, 1000.0, 5000.0]), 30.0)
    assert p[0] == pytest.approx(0.0, abs=1e-6)
    assert 1000 < p[1] < 1020
    assert 5000 < p[2] < 5120


def test_fetch_ssp_transect_builds_2d(annual_http):
    ssp = ss.fetch_ssp_transect((0.0, 0.0), (1.0, 0.0), n_points=4)
    assert ssp.is_range_dependent
    assert ssp.data.shape == (len(_Z), 4)        # depths × columns
    assert ssp.ranges[0] == 0.0
    assert np.all(np.diff(ssp.ranges) > 0)       # increasing range axis
    # 1° latitude ≈ 111 km total transect length.
    assert ssp.ranges[-1] == pytest.approx(111_195.0, rel=1e-3)
    # Identical fixture columns → every range column equals the 1-D profile.
    assert np.allclose(ssp.data[:, 0], ssp.data[:, -1])


def test_assemble_range_dependent_reorders_unsorted_ranges():
    z = np.array([0.0, 100.0])
    cols = [SoundSpeedProfile(depths=z, data=np.array([[1500.0], [1510.0]])),
            SoundSpeedProfile(depths=z, data=np.array([[1490.0], [1500.0]])),
            SoundSpeedProfile(depths=z, data=np.array([[1480.0], [1490.0]]))]
    # Ranges supplied out of order: the result must come back ascending, with
    # each column following its range.
    ssp = ss.assemble_range_dependent(cols, [2000.0, 0.0, 1000.0])
    assert list(ssp.ranges) == [0.0, 1000.0, 2000.0]
    assert ssp.data[0, 0] == 1490.0              # the 0 m column
    assert ssp.data[0, 1] == 1480.0              # the 1000 m column
    assert ssp.data[0, 2] == 1500.0              # the 2000 m column


@pytest.mark.requires_network
def test_live_woa23_profile():
    try:
        profile = ss.fetch_ssp((30.5, -40.5))
    except DataFetchError as exc:
        pytest.skip(f"WOA23 OPeNDAP unreachable: {exc.message}")
    assert profile.depths[0] == 0.0
    assert profile.depths[-1] > 3000.0           # deep open ocean
    assert np.all((1440 < profile.data) & (profile.data < 1560))
