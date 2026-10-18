"""Wind-reference-height scaling in sea-surface generation (uacpy.data.sea_surface).

``generate_sea_surface`` expects the 19.5 m Pierson-Moskowitz reference wind;
the fetched wind is U10. The wind branch must scale U10 by the 10 → 19.5 m
factor, while the waves branch (Hs-inverted PM wind, already at the reference
height) must pass its wind through unscaled.
"""

import numpy as np
import pytest

from uacpy.core.exceptions import DataFetchError
from uacpy.data import SOURCES, DataProvenance
from uacpy.data import sea_surface, waves as waves_mod, wind_live


def _capture_surface(monkeypatch, seen):
    def fake_surface(rmax, wind_kn, n_points=500, **_draw):
        seen['wind'] = wind_kn
        return np.column_stack([np.linspace(0.0, rmax, n_points),
                                np.zeros(n_points)])
    monkeypatch.setattr(sea_surface, 'generate_sea_surface', fake_surface)


def test_u10_to_pm_reference_is_the_inverse_of_ainslies_v10_over_v20():
    # Ainslie Eq. (8.12): v10/v20 ≈ 0.94 (Dobson 1981).
    assert sea_surface._U10_TO_U195 == pytest.approx(1.0 / 0.94)


def test_wind_branch_scales_u10_to_reference_height(monkeypatch):
    monkeypatch.setattr(
        waves_mod, 'fetch_waves',
        lambda point, **kw: (_ for _ in ()).throw(DataFetchError("no waves")))
    monkeypatch.setattr(wind_live, 'wind_at', lambda point, **kw: (
        10.0, DataProvenance(source=SOURCES['nbs'],
                             requested_point=tuple(point))))
    seen = {}
    _capture_surface(monkeypatch, seen)
    alt = sea_surface.fetch_sea_surface(
        (50.0, 0.0), date='2020-01-01', rmax=5000.0, n_points=32,
        rng=np.random.default_rng(1))
    assert alt.data_sources[0].source.id == 'nbs'
    # the faked wind_at answers 10 kn; the PM wind is that scaled to 19.5 m
    assert seen['wind'] == pytest.approx(10.0 / 0.94)


def test_waves_branch_wind_not_scaled(monkeypatch):
    monkeypatch.setattr(
        waves_mod, 'fetch_waves',
        lambda point, **kw: waves_mod.SeaStateRecord(
            hs=2.1, tp=8.0, provenance=DataProvenance(
                source=SOURCES['waverys'], requested_point=(50.0, 0.0))))
    seen = {}
    _capture_surface(monkeypatch, seen)
    alt = sea_surface.fetch_sea_surface(
        (50.0, 0.0), date='2020-01-01', rmax=5000.0, n_points=32,
        rng=np.random.default_rng(1))
    assert alt.data_sources[0].source.id == 'waverys'
    assert seen['wind'] == pytest.approx(sea_surface.hs_to_pm_wind(2.1))


def test_the_sea_surface_source_matches_in_any_case(monkeypatch):
    monkeypatch.setattr(wind_live, 'wind_at', lambda point, **kw: (
        10.0, DataProvenance(source=SOURCES['nbs'],
                             requested_point=tuple(point))))
    _capture_surface(monkeypatch, {})
    alt = sea_surface.fetch_sea_surface(
        (50.0, 0.0), date='2020-01-01', rmax=5000.0, n_points=32,
        rng=np.random.default_rng(1), source='NBS')
    assert alt.data_sources[0].source.id == 'nbs'
