"""Tests for the WOA23 sound-speed fetch (uacpy.data.sound_speed).

Unit tests stub the OPeNDAP HTTP layer with canned DAP ``.ascii`` bodies so
they run fully offline; one ``requires_network`` test hits the live NCEI
THREDDS server (auto-skipped offline).
"""

import numpy as np
import pytest

import uacpy
from uacpy.core.environment import Bathymetry, SoundSpeedProfile
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


def _provenance_columns():
    """Four columns whose provenance overlaps: two share ``copernicus``, one
    carries no records at all, and the last repeats a record already seen."""
    from uacpy.data.sources import SOURCES, DataProvenance
    p_c1 = DataProvenance(source=SOURCES['copernicus'],
                          requested_point=(30.0, -40.0))
    p_c2 = DataProvenance(source=SOURCES['copernicus'],
                          requested_point=(31.0, -40.0))
    p_w = DataProvenance(source=SOURCES['woa23'], requested_point=(32.0, -40.0))
    p_g = DataProvenance(source=SOURCES['gebco'])
    return [
        SoundSpeedProfile(depths=[0.0, 50.0, 100.0],
                          data=[1500.0, 1495.0, 1490.0],
                          data_sources=(p_c1, p_w)),
        SoundSpeedProfile(depths=[0.0, 120.0], data=[1501.0, 1488.0],
                          data_sources=(p_c2,)),
        SoundSpeedProfile(depths=[0.0, 80.0], data=[1502.0, 1492.0],
                          data_sources=()),
        SoundSpeedProfile(depths=[0.0, 90.0], data=[1503.0, 1491.0],
                          data_sources=(p_g, p_c1)),
    ]


def test_assemble_range_dependent_aggregates_through_the_carrier_deduper(
        monkeypatch):
    """The assembly reaches ``_carrier_validate._dedupe_provenance`` — the
    module that declares itself the single home for this union — rather than
    re-deriving first-seen-order dedupe-by-source-id beside it.

    Monkeypatching the helper is what distinguishes delegation from a local
    copy: a re-implementation ignores the patch and returns the real union.
    """
    from uacpy.core import _carrier_validate
    from uacpy.data.sources import SOURCES, DataProvenance
    # A record no column carries, so the real union can never return it.
    sentinel = (DataProvenance(source=SOURCES['gmrt'],
                               requested_point=(89.0, 179.0)),)
    seen = {}

    def _spy(carriers):
        seen['carriers'] = list(carriers)
        return sentinel

    monkeypatch.setattr(_carrier_validate, '_dedupe_provenance', _spy)
    monkeypatch.setattr(ss, '_dedupe_provenance', _spy)
    out = ss.assemble_range_dependent(_provenance_columns(),
                                      [3000.0, 0.0, 2000.0, 1000.0])
    assert tuple(out.data_sources) == sentinel
    # It is handed the columns in assembled (ascending-range) order, so the
    # first-seen record is the one from the nearest column.
    assert [c.depths.max() for c in seen['carriers']] == [120.0, 90.0, 80.0,
                                                          100.0]


def test_assemble_range_dependent_dedupes_provenance_by_source_id():
    """One record survives per dataset, in first-seen order over the columns
    sorted by range, and a column without ``data_sources`` contributes none."""
    cols = _provenance_columns()
    out = ss.assemble_range_dependent(cols, [3000.0, 0.0, 2000.0, 1000.0])
    assert isinstance(out.data_sources, tuple)
    assert [p.source.id for p in out.data_sources] == ['copernicus', 'gebco',
                                                       'woa23']
    # The surviving copernicus record is the 1000 m column's, not the 3000 m
    # column's: the union runs over the reordered columns.
    assert out.data_sources[0].requested_point == (31.0, -40.0)
    assert out.data_sources[2].requested_point == (32.0, -40.0)
    # The records are the very objects the columns carried, not copies.
    assert out.data_sources[0] is cols[1].data_sources[0]
    assert out.data_sources[2] is cols[0].data_sources[1]


def test_assemble_range_dependent_tolerates_a_column_without_provenance():
    """A carrier lacking the attribute entirely is skipped, not an error."""
    class _NoProvenance:
        depths = np.array([0.0, 10.0])
        data = np.array([[1500.0], [1499.0]])

    cols = _provenance_columns()
    out = ss.assemble_range_dependent([cols[0], _NoProvenance()],
                                      [0.0, 100.0])
    assert [p.source.id for p in out.data_sources] == ['copernicus', 'woa23']


@pytest.mark.requires_network
def test_live_woa23_profile():
    try:
        profile = ss.fetch_ssp((30.5, -40.5))
    except DataFetchError as exc:
        pytest.skip(f"WOA23 OPeNDAP unreachable: {exc.message}")
    assert profile.depths[0] == 0.0
    assert profile.depths[-1] > 3000.0           # deep open ocean
    assert np.all((1440 < profile.data) & (profile.data < 1560))


def _column(depths, speeds):
    return SoundSpeedProfile(depths=np.asarray(depths, dtype=float),
                             data=np.asarray(speeds, dtype=float))


def _bathy(pairs):
    return Bathymetry.coerce(np.asarray(pairs, dtype=float))


def test_column_keeps_its_analysed_levels_under_a_shallower_seafloor():
    """A column reaching below its waypoint's seafloor comes back whole.

    Bathymetry is sampled far more finely than the SSP columns, so the
    seafloor between two waypoints is routinely deeper than at either: a
    column cut back to its own waypoint has lost samples the assembled field
    still interpolates through at those intermediate ranges.
    """
    col = _column([0.0, 1000.0, 3000.0, 4800.0],
                  [1540.0, 1484.0, 1516.0, 1540.8])
    out = ss.extend_column_to_seafloor(col, _bathy([[0.0, 3381.0],
                                                    [1.0e5, 3381.0]]), 0.0)
    assert out is col
    assert float(out.depths[-1]) == 4800.0
    assert float(out.data[-1, 0]) == pytest.approx(1540.8)


def test_column_extends_under_a_deeper_seafloor():
    col = _column([0.0, 1000.0, 5500.0], [1540.0, 1484.0, 1551.05])
    with pytest.warns(UserWarning, match='extrapolated'):
        out = ss.extend_column_to_seafloor(
            col, _bathy([[0.0, 8801.0], [1.0e5, 8801.0]]), 0.0, latitude=29.78)
    assert float(out.depths[-1]) == pytest.approx(8801.0)
    # UNESCO at 8801 m holding the column's own deepest T/S: 1611.93 m/s.
    assert float(out.data[-1, 0]) == pytest.approx(1611.93, abs=0.05)


def test_transect_holds_no_cut_value_inside_the_water_column(monkeypatch):
    """On a ridge whose crest falls between two SSP waypoints, the assembled
    field must still carry the analysed deep sample there.

    The waypoint seafloors (3000 m) sit above the columns' deepest analysed
    level (4500 m) while the seafloor between them reaches 5200 m. Cutting
    each column at its waypoint leaves 1497 m/s flat-held from 3000 m down;
    the analysed value at 4500 m is 1530 m/s.
    """
    deep_z = np.array([0.0, 1000.0, 3000.0, 4500.0])
    deep_c = np.array([1540.0, 1484.0, 1497.0, 1530.0])
    monkeypatch.setattr(ss, 'fetch_ssp',
                        lambda point, **kw: _column(deep_z, deep_c))
    length = 111_195.0                       # 1 degree of latitude
    seafloor = _bathy([[0.0, 3000.0], [length / 2, 5200.0], [length, 3000.0]])
    prof = ss.fetch_ssp_transect((0.0, 0.0), (1.0, 0.0), n_points=2,
                                 seafloor=seafloor)
    assert float(prof.depths[-1]) == pytest.approx(4500.0)
    mid = prof.eval(range=length / 2)
    c_4500 = float(np.interp(4500.0, mid.depths, mid.data[:, 0]))
    assert c_4500 == pytest.approx(1530.0, abs=1e-6), (
        f"{c_4500:.1f} m/s at 4500 m over a 5200 m seafloor — the analysed "
        f"1530.0 was cut and 1497.0 held in its place")


def test_operational_transect_keeps_its_levels_too(monkeypatch):
    """The Copernicus transect fetcher carries the same rule: a 3000 m column
    over a 500 m waypoint seafloor keeps all four of its levels."""
    import sys
    import types
    from uacpy.data import copernicus

    depths = [0.0, 100.0, 1000.0, 3000.0]

    class _DA:                       # minimal xarray.DataArray stand-in
        def __init__(self, values):
            self.values = np.asarray(values, dtype=float)

        def sel(self, **kwargs):
            return self

    class _DS:
        def __init__(self):
            self._vars = {'depth': _DA(depths),
                          'thetao': _DA([22.0, 15.0, 5.0, 3.0]),
                          'so': _DA([36.0, 36.2, 35.0, 34.9])}

        def __getitem__(self, key):
            return self._vars[key]

    fake = types.ModuleType('copernicusmarine')
    fake.open_dataset = lambda **kwargs: _DS()
    monkeypatch.setitem(sys.modules, 'copernicusmarine', fake)
    ssp = copernicus.fetch_ssp_transect_operational(
        (30.0, -40.0), (31.0, -40.0), date='2020-06-15', n_points=4,
        seafloor=_bathy([[0.0, 500.0], [2.0e5, 500.0]]))
    assert ssp.depths.tolist() == depths


class TestTheWetCellHopIsTheNearestToTheRequest:
    """A coastal request snaps to a dry WOA cell; the fallback must take the
    wet cell nearest the REQUESTED point. Ring order ties at d² = 1 break in
    file order and probe the meridional neighbour (111 km away at 44°N)
    before the zonal one (80 km), and never look at where inside the cell
    the request fell."""

    def test_the_zonal_neighbour_wins_when_it_is_nearer(self):
        from uacpy.data.sound_speed import _nearest_wet_column, _grid_index
        lat, lon = 43.8, 7.5
        lat_idx, lon_idx, _c_lat, _c_lon = _grid_index(lat, lon, '1.00')
        wet = {(lat_idx, lon_idx + 1), (lat_idx - 1, lon_idx)}   # (43.5, 8.5) and (42.5, 7.5)
        empty = np.array([])

        def fetch(i, j):
            if (i, j) in wet:
                return np.array([0.0, 10.0]), np.array([10.0, 9.0]), np.array([35.0, 35.0])
            return empty, empty, empty

        *_cols, i, j = _nearest_wet_column(fetch, lat_idx, lon_idx, '1.00',
                                           lat=lat, lon=lon)
        assert (i, j) == (lat_idx, lon_idx + 1)

    def test_a_fractional_month_is_refused_not_truncated(self):
        from uacpy.data.sound_speed import _resolve_period
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match='integer month'):
            _resolve_period(None, 6.7)
        with pytest.raises(ConfigurationError, match='bool'):
            _resolve_period(None, True)
        assert _resolve_period(None, 6.0) == 6


class TestTheDeepExtensionUsesTheProfilesOwnFormula:
    def _profile(self, formula):
        from uacpy.core.ssp import SoundSpeedProfile
        return SoundSpeedProfile(depths=np.array([0.0, 5500.0]),
                                 data=np.array([1500.0, 1551.05]),
                                 formula=formula)

    def test_delgrosso_and_unesco_extend_differently_and_none_means_unesco(self):
        from uacpy.data.sound_speed import extend_ssp_below_data
        deep = {f: float(np.asarray(extend_ssp_below_data(self._profile(f), 8800.0).data)[-1, 0])
                for f in ('unesco', 'delgrosso', None)}
        assert deep[None] == deep['unesco']
        assert abs(deep['delgrosso'] - deep['unesco']) > 0.1     # verifier: +0.33 m/s at 8.8 km

    def test_the_formula_survives_the_carrier_copies(self):
        ssp = self._profile('delgrosso')
        assert ssp.extend_to(6000.0).formula == 'delgrosso'
        assert ssp.copy().formula == 'delgrosso'
