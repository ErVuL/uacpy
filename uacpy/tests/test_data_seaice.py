"""Tests for the NSIDC sea-ice climatology backend (uacpy.data.seaice_local)."""

import warnings

import numpy as np
import pytest

from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import seaice_local


def test_decode_to_fraction():
    a = np.array([[0, 500, 1000], [2510, 2530, 2540]], dtype=np.uint16)
    f = seaice_local._to_fraction(a)
    assert f[0].tolist() == [0.0, 0.5, 1.0]      # concentration ×10 → fraction
    assert f[1, 0] == 1.0                         # pole hole → perennial ice
    assert np.isnan(f[1, 1]) and np.isnan(f[1, 2])   # coast / land


class _FakeTF:
    """Stands in for a ``pyproj.Transformer``, forward and inverse.

    The forward map sends every coordinate to one projected point, so it has
    no true inverse; ``direction='INVERSE'`` returns the seed lon/lat the fake
    was built to represent. That is what ``_cell_center`` calls to name a
    substituted cell, and no test here asserts on the coordinates it reports.
    """

    def __init__(self, x, y, lon=0.0, lat=85.0):
        self._x, self._y = x, y
        self._lon, self._lat = lon, lat

    def transform(self, a, b, direction=None):
        if direction == 'INVERSE':
            return self._lon, self._lat
        return self._x, self._y


@pytest.fixture
def synthetic_model(monkeypatch):
    g = seaice_local._GRID['N']
    # A point that maps to pixel (row=2, col=3) of the North grid.
    x = g['x0'] + 3 * g['px']
    y = g['y0'] - 2 * g['px']
    north = np.zeros((12, 5, 6), dtype=np.float32)
    north[2, 2, 3] = 0.7                          # March, that pixel
    # June: one unobserved pixel amid observed water — NSIDC's coastline class,
    # withheld for land spillover, not because the cell is dry.
    north[5] = 0.8
    north[5, 2, 3] = np.nan
    # September: nothing observed anywhere, which is what genuine land looks
    # like once the climatology has averaged the flag codes away.
    north[8] = np.nan
    model = {'tf': {'N': _FakeTF(x, y), 'S': _FakeTF(0.0, 0.0)},
             'N': north, 'S': np.zeros((12, 5, 6), dtype=np.float32)}
    monkeypatch.setattr(seaice_local, '_model', lambda: model)
    return model


def test_concentration_value(synthetic_model):
    assert seaice_local.fetch_sea_ice_concentration((85.0, 0.0), month=3) == \
        pytest.approx(0.7)


def test_concentration_by_date(synthetic_model):
    assert seaice_local.fetch_sea_ice_concentration((85.0, 0.0),
                                                    date='2026-03-15') == pytest.approx(0.7)


def test_land_raises(synthetic_model):
    """No observed cell within the search radius means the point is inland."""
    with pytest.raises(DataFetchError, match='inland'):
        seaice_local.fetch_sea_ice_concentration((85.0, 0.0), month=9)


def test_a_coastal_spillover_cell_takes_its_observed_neighbour(synthetic_model):
    """NSIDC withholds a concentration at its coastline class because of land
    spillover in the passive-microwave footprint, not because the cell is dry,
    and the averaged climatology cannot tell that class from true land — both are
    NaN. Reading such a cell as open water put a pressure-release surface in the
    middle of pack ice: measured at Tiksi (71.65, 128.90) and Pond Inlet
    (72.70, -77.95), both of which sit in 93-99 % March ice."""
    got = seaice_local.fetch_sea_ice_concentration((85.0, 0.0), month=6)
    assert got == pytest.approx(0.8)


def test_out_of_grid_is_ice_free(monkeypatch):
    far = {'tf': {'N': _FakeTF(1e9, 1e9), 'S': _FakeTF(1e9, 1e9)},
           'N': np.zeros((12, 5, 6), np.float32), 'S': np.zeros((12, 5, 6), np.float32)}
    monkeypatch.setattr(seaice_local, '_model', lambda: far)
    assert seaice_local.fetch_sea_ice_concentration((85.0, 0.0), month=3) == 0.0


def test_date_and_month_rejected(synthetic_model):
    with pytest.raises(ConfigurationError, match='not both'):
        seaice_local.fetch_sea_ice_concentration((85.0, 0.0), date='2026-03-01', month=3)


def test_requires_date_or_month(synthetic_model):
    with pytest.raises(ConfigurationError, match='date.*month'):
        seaice_local.fetch_sea_ice_concentration((85.0, 0.0))


def test_transect_classifies_each_waypoint_from_its_own_cell(monkeypatch):
    # Two latitude bands of the North grid, both in column 3: row 1 (poleward
    # of 85.25°N) holds 0.9 pack ice, row 3 (equatorward) holds 0.05 open
    # water. Waypoints at 86, 85.67, 85.33, 85°N each read their own cell.
    g = seaice_local._GRID['N']

    class _LatBandTF:
        def transform(self, lon, lat):
            row = 1 if lat >= 85.25 else 3
            return g['x0'] + 3.5 * g['px'], g['y0'] - (row + 0.5) * g['px']

    north = np.zeros((12, 5, 6), dtype=np.float32)
    north[2, 1, 3] = 0.9
    north[2, 3, 3] = 0.05
    model = {'tf': {'N': _LatBandTF(), 'S': _LatBandTF()},
             'N': north, 'S': np.zeros((12, 5, 6), dtype=np.float32)}
    monkeypatch.setattr(seaice_local, '_model', lambda: model)

    r, c = seaice_local.fetch_sea_ice_concentration_transect(
        (86.0, 0.0), (85.0, 0.0), month=3, n_points=4)
    assert r.shape == (4,) and c.shape == (4,)
    assert r[0] == 0.0 and np.all(np.diff(r) > 0)
    assert c.tolist() == pytest.approx([0.9, 0.9, 0.9, 0.05])
    # The same per-waypoint zones drive the surface classification: 0.9 ≥ the
    # 15 % ice-edge → elastic canopy, 0.05 < it → open-water vacuum.
    surf = seaice_local.sea_ice_surface_transect(
        (86.0, 0.0), (85.0, 0.0), month=3, n_points=4)
    assert [bp.acoustic_type for bp in surf.properties] == \
        ['half-space', 'half-space', 'half-space', 'vacuum']


def test_sea_ice_surface_transect(synthetic_model):
    # March pixel reads 0.7 ≥ threshold along the whole transect → an elastic
    # Surface carrier with one node per waypoint.
    from uacpy.core.surface import Surface
    surf = seaice_local.sea_ice_surface_transect(
        (85.0, 0.0), (85.5, 0.0), month=3, n_points=4)
    assert isinstance(surf, Surface)
    assert surf.n_ranges == 4 and surf.is_elastic
    assert surf.at(range=0).acoustic_type == 'half-space'


def test_auto_transect_places_ice_edge_at_observed_boundary(monkeypatch):
    """Regression: the 'auto' collapse must keep the probe samples bracketing
    each zone change. The former midpoint collapse anchored a two-run
    (ice → open water) transect at its endpoints only, and the nearest-node
    ``Surface`` then rebuilt the edge at mid-transect — 600+ km off the
    NSIDC March edge on an (85°N, 0°E) → (60°N, 0°E) transect."""
    length_m = 1_000_000.0
    edge_m = 300_000.0                    # ice for r < edge_m, open water beyond
    probe_n = 200
    step_m = length_m / (probe_n - 1)

    def fake_transect(start, end, *, date=None, month=None, n_points=6):
        r = np.linspace(0.0, length_m, n_points)
        return r, np.where(r < edge_m, 0.9, 0.0)

    monkeypatch.setattr(seaice_local, 'fetch_sea_ice_concentration_transect',
                        fake_transect)
    surf = seaice_local.sea_ice_surface_transect(
        (85.0, 0.0), (60.0, 0.0), month=3, max_points=probe_n)
    rr = np.asarray(surf.ranges, dtype=float)
    kinds = [bp.acoustic_type for bp in surf.properties]
    assert rr[0] == 0.0 and rr[-1] == pytest.approx(length_m)
    assert kinds[0] == 'half-space' and kinds[-1] == 'vacuum'
    # Nearest-node reconstruction transitions midway between adjacent kept
    # nodes of different kind; that must land within one probe step of the
    # edge the probe observed.
    transitions = [(rr[i] + rr[i + 1]) / 2.0
                   for i in range(len(kinds) - 1) if kinds[i] != kinds[i + 1]]
    assert len(transitions) == 1
    assert abs(transitions[0] - edge_m) <= step_m
    assert surf.at(range=edge_m - step_m).acoustic_type == 'half-space'
    assert surf.at(range=edge_m + step_m).acoustic_type == 'vacuum'


def test_auto_transect_collapses_a_uniform_zone_to_one_node(monkeypatch):
    """A single run (ice everywhere) still collapses to one range-independent
    node — two identical columns would read as a range-dependent surface."""
    def fake_transect(start, end, *, date=None, month=None, n_points=6):
        return np.linspace(0.0, 1.0e6, n_points), np.full(n_points, 0.9)

    monkeypatch.setattr(seaice_local, 'fetch_sea_ice_concentration_transect',
                        fake_transect)
    surf = seaice_local.sea_ice_surface_transect(
        (85.0, 0.0), (60.0, 0.0), month=3, max_points=50)
    assert surf.n_ranges == 1
    assert surf.at(range=0).acoustic_type == 'half-space'


def test_sea_ice_surface_gates_on_threshold():
    # Below the 15 % ice-edge → open water (no surface override).
    assert seaice_local.sea_ice_surface(0.10) is None
    assert seaice_local.sea_ice_surface(0.30, threshold=0.5) is None
    # At/above → an elastic ice canopy. Jensen, Kuperman, Porter & Schmidt,
    # *Computational Ocean Acoustics*, quote two attenuation pairs for the same
    # 3500/1800 m/s, 900 kg/m³ canopy: 0.4/1.0 dB/λ for the Arctic propagation
    # example and 0.5/1.0 dB/λ elsewhere. uacpy implements 0.4
    # (``SEA_ICE_COMPRESSIONAL_ATTENUATION``), so that is what this pins.
    bp = seaice_local.sea_ice_surface(0.15)
    assert bp is not None and bp.acoustic_type == 'half-space'
    assert (bp.sound_speed, bp.shear_speed, bp.density) == (3500.0, 1800.0, 0.9)
    assert (bp.attenuation, bp.shear_attenuation) == (0.4, 1.0)


def test_sea_ice_surface_nan_is_open_water():
    # A non-finite concentration (NaN land/coast/out-of-grid cell) must be
    # treated as open water, never silently as ice: NaN < threshold is False,
    # so without the isfinite guard a land pixel would get an elastic canopy.
    assert seaice_local.sea_ice_surface(np.nan) is None
    assert seaice_local.sea_ice_surface(float('inf')) is None


def test_fetch_sea_ice_surface(synthetic_model):
    # 0.7 at the March pixel >= threshold -> ice. June's unobserved pixel takes
    # its 0.8 neighbour, so it is ice too. September has nothing observed
    # anywhere -> inland -> raises.
    assert seaice_local.fetch_sea_ice_surface((85.0, 0.0), month=3) is not None
    assert seaice_local.fetch_sea_ice_surface((85.0, 0.0), month=6) is not None
    with pytest.raises(DataFetchError, match='inland'):
        seaice_local.fetch_sea_ice_surface((85.0, 0.0), month=9)


def test_download_builds_climatology(tmp_path, monkeypatch):
    pytest.importorskip('tifffile')             # default dep; guard a stripped-down env
    monkeypatch.setattr(seaice_local, 'http_get', lambda url, **kw: b'TIFF')
    monkeypatch.setattr('tifffile.imread',
                        lambda b: np.full((4, 4), 600, dtype=np.uint16))
    out = seaice_local.download_seaice_db(cache_dir=str(tmp_path), years=[2023])
    assert out.exists()
    with np.load(out, allow_pickle=False) as climo:
        assert climo['N'].shape == (12, 4, 4)
        assert np.allclose(climo['N'], 0.6)      # 600/1000


def test_a_full_climatology_builds_without_a_coverage_warning(tmp_path,
                                                              monkeypatch):
    """Every requested grid decodes, so nothing is skipped and the build is
    silent — the warning below must not fire on a healthy download."""
    pytest.importorskip('tifffile')
    monkeypatch.setattr(seaice_local, 'http_get', lambda url, **kw: b'TIFF')
    monkeypatch.setattr('tifffile.imread',
                        lambda b: np.full((4, 4), 600, dtype=np.uint16))
    with warnings.catch_warnings():
        warnings.simplefilter('error')           # any warning fails the test
        seaice_local.download_seaice_db(cache_dir=str(tmp_path),
                                        years=[2022, 2023])


def test_a_partial_climatology_says_how_thin_it_is(tmp_path, monkeypatch):
    """A grid that will not fetch or decode is skipped, so a month can be
    averaged over fewer years than were requested. Only a month with nothing
    at all raises; a partial one is a valid but thinner climatology, and
    silence would present a one-year mean as the two-year mean asked for."""
    pytest.importorskip('tifffile')
    monkeypatch.setattr(seaice_local, 'http_get', lambda url, **kw: b'TIFF')

    calls = {'n': 0}

    def flaky(buf):
        # Fail March of the first year only: that month averages 1 of 2 years
        # while the other eleven get both, so the shortfall is per-month.
        calls['n'] += 1
        if calls['n'] == 3:
            raise ValueError('truncated TIFF')
        return np.full((4, 4), 600, dtype=np.uint16)

    monkeypatch.setattr('tifffile.imread', flaky)
    with pytest.warns(UserWarning, match=r'3: 1/2'):
        out = seaice_local.download_seaice_db(cache_dir=str(tmp_path),
                                              years=[2022, 2023])
    with np.load(out, allow_pickle=False) as climo:
        assert climo['N'].shape == (12, 4, 4)    # still a usable climatology


def test_sea_ice_surface_transect_warns_once_for_no_data_waypoints(monkeypatch):
    def fake_transect(start, end, *, date=None, month=None, n_points=6):
        conc = np.full(n_points, 0.9)
        conc[:2] = np.nan                      # land clipped along the track
        return np.linspace(0.0, 1.0e5, n_points), conc

    monkeypatch.setattr(seaice_local, 'fetch_sea_ice_concentration_transect',
                        fake_transect)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        surf = seaice_local.sea_ice_surface_transect(
            (85.0, 0.0), (60.0, 0.0), month=3, n_points=8)
    hits = [w for w in rec if 'no NSIDC concentration' in str(w.message)]
    assert len(hits) == 1
    assert '2 of 8' in str(hits[0].message)
    assert 'not measured' in str(hits[0].message)
    # The no-data waypoints classify as open water (vacuum), not as ice.
    assert surf.at(range=0).acoustic_type == 'vacuum'


def test_sea_ice_surface_transect_with_full_coverage_does_not_warn(monkeypatch):
    def fake_transect(start, end, *, date=None, month=None, n_points=6):
        conc = np.where(np.arange(n_points) < n_points // 2, 0.9, 0.0)
        return np.linspace(0.0, 1.0e5, n_points), conc

    monkeypatch.setattr(seaice_local, 'fetch_sea_ice_concentration_transect',
                        fake_transect)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        seaice_local.sea_ice_surface_transect(
            (85.0, 0.0), (60.0, 0.0), month=3, n_points=8)
    assert not [w for w in rec if 'no NSIDC concentration' in str(w.message)]


def test_sea_ice_names_the_cell_it_substituted(monkeypatch):
    """The hop to a neighbouring observed cell is up to 2 cells (50 km) and
    changes the answer, so it warns like ``sound_speed._nearest_wet_column``."""
    pytest.importorskip('pyproj')
    from uacpy.data import seaice_local
    grid = np.full((12, 5, 5), np.nan, dtype=np.float32)
    grid[:, 2, 3] = 0.8                       # one observed cell, east of centre
    model = {'tf': {h: seaice_local._pyproj_transformer(
        seaice_local._GRID[h]['epsg']) for h in ('N', 'S')},
        'N': grid, 'S': grid}
    monkeypatch.setattr(seaice_local, '_model', lambda: model)
    monkeypatch.setattr(seaice_local, '_rowcol',
                        lambda m, hemi, lat, lon: (2, 2))
    with pytest.warns(UserWarning, match='nearest observed cell'):
        conc = seaice_local.fetch_sea_ice_concentration((85.0, 0.0), month=3)
    assert conc == pytest.approx(0.8)


def test_sea_ice_direct_hit_is_silent(monkeypatch):
    pytest.importorskip('pyproj')
    from uacpy.data import seaice_local
    grid = np.full((12, 5, 5), 0.8, dtype=np.float32)
    model = {'tf': {h: seaice_local._pyproj_transformer(
        seaice_local._GRID[h]['epsg']) for h in ('N', 'S')},
        'N': grid, 'S': grid}
    monkeypatch.setattr(seaice_local, '_model', lambda: model)
    monkeypatch.setattr(seaice_local, '_rowcol',
                        lambda m, hemi, lat, lon: (2, 2))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert seaice_local.fetch_sea_ice_concentration(
            (85.0, 0.0), month=3) == pytest.approx(0.8)


def test_sea_ice_canopy_roughness_is_opt_in():
    """The published 0.4/1.0 dB/λ parameters describe a smooth plate, so the
    default stays 0; a caller who wants the scattering passes a roughness."""
    from uacpy.data.seaice_local import (SEA_ICE_TYPICAL_ROUGHNESS_M,
                                         sea_ice_surface)
    smooth = sea_ice_surface(0.9)
    rough = sea_ice_surface(0.9, roughness=SEA_ICE_TYPICAL_ROUGHNESS_M)
    assert not smooth.roughness
    assert rough.roughness == pytest.approx(SEA_ICE_TYPICAL_ROUGHNESS_M)
    for field in ('sound_speed', 'shear_speed', 'density', 'attenuation',
                  'shear_attenuation'):
        assert getattr(smooth, field) == getattr(rough, field)


def test_an_all_failing_first_year_sweep_stops_the_climatology_build(
        monkeypatch, tmp_path):
    """12 of 12 fetch failures in the first year sweep raise the typed
    error before any later year is attempted, so an unreachable server
    costs one sweep of retries, not five."""
    import uacpy.data.seaice_local as si
    calls = []
    def refused(url, **kwargs):
        calls.append(url)
        raise DataFetchError('connection refused')
    monkeypatch.setattr(si, 'http_get', refused)
    with pytest.raises(DataFetchError,
                       match=r'stopping before the remaining 4 year'):
        si.download_seaice_db(cache_dir=tmp_path, years=range(2021, 2026))
    assert len(calls) == 12


class TestTheObservedNeighbourNearestTheRequestIsTaken:
    def test_rank_orders_the_candidates(self):
        from uacpy.data.seaice_local import _observed_at
        grid = np.full((5, 5), np.nan)
        grid[1, 2] = 0.3            # ring-1 meridional neighbour, first in ring order
        grid[2, 3] = 0.7            # ring-1 zonal neighbour, nearer to a request at col 2.9
        value, cell = _observed_at(grid, 2, 2)
        assert cell == (1, 2)       # ring order: the file-order tie
        value, cell = _observed_at(grid, 2, 2, rank=lambda r, c: (r - 2.0) ** 2 + (c - 2.9) ** 2)
        assert cell == (2, 3) and value == 0.7

    def test_a_fractional_month_is_refused(self):
        from uacpy.data.seaice_local import fetch_sea_ice_concentration
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match='integer month'):
            fetch_sea_ice_concentration((72.0, -150.0), month=6.7)
