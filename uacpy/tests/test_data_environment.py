"""Tests for the fetch_environment capstone (uacpy.data.environment)."""

import numpy as np
import pytest

from uacpy.core.environment import BoundaryProperties, Environment, SoundSpeedProfile
from uacpy.core.exceptions import ConfigurationError
from uacpy.data import environment as env_mod


@pytest.fixture
def stub_fetchers(monkeypatch):
    """Replace the network fetchers with deterministic stand-ins.

    Stubs both the point and the transect SSP fetchers — a transect request
    routes through ``fetch_ssp_transect`` (range-dependent SSP), so without that
    stub a transect test would fall through to a live WOA23 OPeNDAP fetch and
    block offline.
    """
    ssp = SoundSpeedProfile(depths=[0.0, 100.0, 2000.0],
                            data=[1500.0, 1490.0, 1510.0])
    rd_ssp = SoundSpeedProfile(depths=[0.0, 100.0, 2000.0],
                               data=[[1500.0, 1502.0], [1490.0, 1492.0],
                                     [1510.0, 1512.0]],
                               ranges=[0.0, 5000.0])
    monkeypatch.setattr(env_mod, 'fetch_bathy',
                        lambda point, **kw: 2000.0)
    monkeypatch.setattr(env_mod, 'fetch_bathy_transect',
                        lambda a, b, **kw: np.array([[0.0, 2000.0], [5000.0, 2200.0]]))
    monkeypatch.setattr(env_mod, 'fetch_ssp',
                        lambda point, **kw: ssp)
    monkeypatch.setattr(env_mod, 'fetch_ssp_transect',
                        lambda start, end, **kw: rd_ssp)
    return ssp


def test_point_environment(stub_fetchers):
    env = env_mod.fetch_environment((43.2, 7.5), date='2026-06-14')
    assert isinstance(env, Environment)
    assert env.depth == 2000.0
    assert env.name == '43.200, 7.500'
    assert not env.has_range_dependent_bathymetry


def test_transect_environment(stub_fetchers):
    env = env_mod.fetch_environment((43.2, 7.5), transect_to=(42.8, 8.1), name='slope')
    assert env.has_range_dependent_bathymetry
    assert env.name == 'slope'


def test_bottom_from_phi(stub_fetchers):
    env = env_mod.fetch_environment((43.2, 7.5), bottom=2.0)  # ϕ → sand-ish
    assert env.bottom is not None
    assert env.bottom.acoustic_type == 'half-space'   # universal across models
    assert env.bottom.columns[0].halfspace.grain_size_phi == 2.0


def test_bottom_from_class_name(stub_fetchers):
    env = env_mod.fetch_environment((43.2, 7.5), bottom='clay')
    assert env.bottom.acoustic_type == 'half-space'


def test_bottom_passthrough(stub_fetchers):
    bp = BoundaryProperties(acoustic_type='half-space', sound_speed=1700, density=1.8)
    env = env_mod.fetch_environment((43.2, 7.5), bottom=bp)
    # A bare BoundaryProperties is coerced into a Bottom, preserving the halfspace.
    assert env.bottom.columns[0].halfspace is bp


def test_literal_ssp_skips_fetch(monkeypatch):
    # A literal ssp= is used verbatim; the SSP fetcher must NOT be called.
    monkeypatch.setattr(env_mod, 'fetch_bathy', lambda point, **kw: 2000.0)
    monkeypatch.setattr(env_mod, 'fetch_ssp',
                        lambda *a, **k: pytest.fail("SSP should not be fetched"))
    prof = SoundSpeedProfile(depths=[0.0, 2000.0], data=[1500.0, 1520.0])
    env = env_mod.fetch_environment((43.2, 7.5), ssp=prof)
    assert env.ssp.data[0] == 1500.0
    assert 'woa23' not in [s.source.id for s in env.data_sources]


def test_literal_bathymetry_skips_fetch(monkeypatch):
    # A literal bathymetry= is used verbatim; the bathy fetcher must NOT run.
    monkeypatch.setattr(env_mod, 'fetch_bathy',
                        lambda *a, **k: pytest.fail("bathy should not be fetched"))
    monkeypatch.setattr(env_mod, 'fetch_ssp',
                        lambda point, **kw: SoundSpeedProfile(depths=[0.0, 50.0],
                                                              data=[1500.0, 1490.0]))
    env = env_mod.fetch_environment((43.2, 7.5), bathymetry=120.0)
    assert env.depth == 120.0
    assert 'gebco' not in [s.source.id for s in env.data_sources]


def test_source_fetch_wins_over_literal(stub_fetchers):
    # Both given + fetch succeeds → the fetched value is used (literal ignored).
    env = env_mod.fetch_environment((43.2, 7.5), bathymetry=999.0,
                                    bathymetry_sources='gebco')
    assert env.depth == 2000.0                       # stub-fetched, not 999.0
    assert 'gebco' in [s.source.id for s in env.data_sources]


def test_bathymetry_sources_auto_prefers_emodnet_dtm(stub_fetchers):
    # 'auto' bathymetry = ('emodnet_dtm', 'gmrt', 'gebco'); the stub fetcher
    # succeeds, so the first (highest-res regional DTM) wins in provenance.
    env = env_mod.fetch_environment((43.2, 7.5), bathymetry_sources='auto')
    assert 'emodnet_dtm' in [s.source.id for s in env.data_sources]


def test_ssp_sources_auto_falls_to_woa23_without_date(stub_fetchers):
    # 'auto' ssp = ('argo','copernicus','woa23'); argo/copernicus need date=, so
    # with none they fall through to WOA23 (the stub).
    env = env_mod.fetch_environment((43.2, 7.5), ssp_sources='auto')
    assert 'woa23' in [s.source.id for s in env.data_sources]
    assert 'argo' not in [s.source.id for s in env.data_sources]


def test_literal_fallback_when_fetch_fails(monkeypatch):
    # Both given + fetch fails → fall back to the user literal (no raise).
    from uacpy.core.exceptions import DataFetchError
    monkeypatch.setattr(env_mod, 'fetch_bathy',
                        lambda *a, **k: (_ for _ in ()).throw(
                            DataFetchError("service down")))
    monkeypatch.setattr(env_mod, 'fetch_ssp',
                        lambda point, **kw: SoundSpeedProfile(depths=[0.0, 50.0],
                                                              data=[1500.0, 1490.0]))
    env = env_mod.fetch_environment((43.2, 7.5), bathymetry=120.0,
                                    bathymetry_sources='gebco')
    assert env.depth == 120.0                        # literal fallback
    assert 'gebco' not in [s.source.id for s in env.data_sources]


def test_range_dependent_ssp(monkeypatch, stub_fetchers):
    rd_ssp = SoundSpeedProfile(depths=[0.0, 100.0, 2000.0],
                               data=[[1500, 1502], [1490, 1492], [1510, 1512]],
                               ranges=[0.0, 5000.0])
    monkeypatch.setattr(env_mod, 'fetch_ssp_transect',
                        lambda start, end, **kw: rd_ssp)
    env = env_mod.fetch_environment((43.2, 7.5), transect_to=(42.8, 8.1),
                                    range_dependent_ssp=True)
    assert env.ssp.is_range_dependent
    assert env.has_range_dependent_bathymetry


def test_range_dependent_ssp_requires_transect(stub_fetchers):
    with pytest.raises(ConfigurationError, match='requires transect_to'):
        env_mod.fetch_environment((43.2, 7.5), range_dependent_ssp=True)


def test_range_dependent_ssp_copernicus(monkeypatch, stub_fetchers):
    import uacpy.data.copernicus as cop_mod
    rd = SoundSpeedProfile(depths=[0.0, 100.0], data=[[1500, 1502], [1490, 1492]],
                           ranges=[0.0, 5000.0])
    monkeypatch.setattr(cop_mod, 'fetch_ssp_transect_operational',
                        lambda start, end, date, **kw: rd)
    env = env_mod.fetch_environment((43.2, 7.5), transect_to=(42.8, 8.1),
                                    range_dependent_ssp=True, ssp_sources='copernicus',
                                    date='2026-01-01')
    assert env.ssp.is_range_dependent


def test_range_dependent_ssp_copernicus_requires_date(stub_fetchers):
    with pytest.raises(ConfigurationError, match='requires date'):
        env_mod.fetch_environment((43.2, 7.5), transect_to=(42.8, 8.1),
                                  range_dependent_ssp=True, ssp_sources='copernicus')


def test_removed_global_keyword_rejected(stub_fetchers):
    # 'global' names no fetch source, so ``bottom='global'`` falls through to
    # the sediment-class lookup and is refused there. uacpy carries no global
    # seabed grid to bind it to: the CC-BY-NC candidate for that role is not
    # redistributable, which is what ``pelagic`` covers instead.
    with pytest.raises(ConfigurationError, match='unknown sediment class'):
        env_mod.fetch_environment((43.2, 7.5), bottom='global')


def test_bottom_auto_falls_back_to_pelagic(tmp_path, monkeypatch, stub_fetchers):
    # 'auto' always resolves: outside measured coverage (and with no DBs
    # installed) it uses the global, commercial-clean pelagic model rather than
    # raising. An empty cache forces the fall-through past EMODnet/Diesing.
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    import uacpy.data.seabed as seabed_mod
    import uacpy.data.pelagic as pelagic_mod
    from uacpy.core.exceptions import DataFetchError

    def no_emodnet(point, **kw):
        raise DataFetchError("European seas only")
    monkeypatch.setattr(seabed_mod, 'fetch_bottom', no_emodnet)
    monkeypatch.setattr(pelagic_mod, '_water_depth', lambda *a, **k: 5000.0)
    env = env_mod.fetch_environment((43.2, 7.5), bottom_sources='auto')
    assert env.bottom is not None
    assert env.data_sources[-1].source.id == 'pelagic'
    # The classification uses the environment's own 2000 m bathymetry (above
    # the CCD → calcareous ooze), not the 5000 m its own lookup would return.
    assert env.bottom.columns[0].halfspace.grain_size_phi == pytest.approx(7.5)


def test_pelagic_never_refetches_the_bathymetry(tmp_path, monkeypatch,
                                                stub_fetchers):
    # fetch_environment already holds the depth, so the pelagic model must be
    # handed it rather than issuing its own GEBCO lookup per waypoint.
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    import uacpy.data.pelagic as pelagic_mod
    monkeypatch.setattr(pelagic_mod, '_water_depth', lambda *a, **k: (
        pytest.fail("pelagic re-fetched the bathymetry")))
    point = env_mod.fetch_environment((43.2, 7.5), bottom_sources='pelagic')
    assert point.bottom.columns[0].halfspace.grain_size_phi == pytest.approx(7.5)
    # A transect crossing the CCD (4500 m) must classify each column at *its
    # own* range: calcareous ooze inshore, pelagic clay beyond the crossing.
    monkeypatch.setattr(
        env_mod, 'fetch_bathy_transect',
        lambda a, b, **kw: np.column_stack([np.linspace(0.0, 50e3, 21),
                                            np.linspace(3000.0, 5500.0, 21)]))
    transect = env_mod.fetch_environment(
        (43.2, 7.5), transect_to=(42.8, 8.1), bottom_sources='pelagic',
        range_dependent_bottom=True, bottom_n_points='auto', max_points=40)
    # Compare densities, not speeds: a column's sound speed is its Hamilton
    # ratio times the *local* water speed, which itself rises with depth, so it
    # does not isolate the lithology. Density is monotone in grain size.
    rho = [c.halfspace.density for c in transect.bottom.columns]
    assert len(rho) == 2 and rho[0] > rho[1]            # ooze, then clay


def test_range_dependent_bottom(monkeypatch, stub_fetchers):
    # 'auto' is cache-first, so EMODnet's local transect fetcher is tried first.
    import uacpy.data.emodnet_local as emodnet_mod
    from uacpy.core.environment import Bottom
    rdb = Bottom.from_halfspaces([0.0, 5000.0], sound_speed=[1650.0, 1600.0],
                               density=[1.9, 1.7], attenuation=[0.8, 1.0])
    monkeypatch.setattr(emodnet_mod, 'fetch_bottom_local_transect',
                        lambda start, end, **kw: rdb)
    env = env_mod.fetch_environment((43.2, 7.5), transect_to=(42.8, 8.1),
                                    range_dependent_bottom=True)
    assert env.bottom is rdb


def test_range_dependent_bottom_requires_transect(stub_fetchers):
    with pytest.raises(ConfigurationError, match='requires transect_to'):
        env_mod.fetch_environment((43.2, 7.5), range_dependent_bottom=True)


def test_with_absorption(monkeypatch, stub_fetchers):
    import uacpy.data.sound_speed as ss_mod
    from uacpy.core.absorption import FrancoisGarrison
    monkeypatch.setattr(ss_mod, 'fetch_ts_profile',
                        lambda point, **kw: (np.array([0.0, 50.0]),
                                                np.array([18.0, 16.0]),
                                                np.array([36.0, 36.1])))
    env = env_mod.fetch_environment((43.2, 7.5), with_absorption=True)
    assert isinstance(env.absorption, FrancoisGarrison)
    assert env.absorption.temperature_c == 18.0


@pytest.mark.parametrize('resolution', ['1.00', '0.25'])
def test_with_absorption_uses_the_requested_woa_grid(monkeypatch, stub_fetchers,
                                                     resolution):
    """The T/S column must come from the same WOA23 grid as the SSP.

    A 0.25° SSP against a 1.00° absorption column is a different cell — up to
    ~50 km away at mid latitudes.
    """
    import uacpy.data.sound_speed as ss_mod
    seen = {}

    def fake_ts(point, **kw):
        seen.update(kw)
        return (np.array([0.0, 50.0]), np.array([18.0, 16.0]),
                np.array([36.0, 36.1]))

    monkeypatch.setattr(ss_mod, 'fetch_ts_profile', fake_ts)
    env_mod.fetch_environment((43.2, 7.5), with_absorption=True,
                              resolution=resolution)
    assert seen['resolution'] == resolution


def test_copernicus_requires_date(stub_fetchers):
    with pytest.raises(ConfigurationError, match='requires date'):
        env_mod.fetch_environment((43.2, 7.5), ssp_sources='copernicus')


def test_unknown_ssp_sources(stub_fetchers):
    with pytest.raises(ConfigurationError, match='unknown ssp source'):
        env_mod.fetch_environment((43.2, 7.5), ssp_sources='nope')


class TestWoaWetCellSearch:
    """A coastal point must not be refused because it snaps onto a land cell.

    The documented quick-start ``fetch_environment((43.2, 7.5))`` is in the
    Ligurian Sea, close enough to shore that the nearest WOA cell can be dry,
    which must not be reported as "on land or outside the analyzed domain".
    """

    def test_ring_offsets_are_nearest_first_and_complete(self):
        from uacpy.data.sound_speed import _ring_offsets
        r1 = _ring_offsets(1)
        assert len(r1) == 8, r1
        assert r1[0] in {(0, -1), (0, 1), (-1, 0), (1, 0)}
        assert len(_ring_offsets(2)) == 16

    def test_dry_nearest_cell_falls_back_to_a_wet_neighbour(self):
        import numpy as np
        from uacpy.data.sound_speed import _nearest_wet_column
        wet = (60, 100)

        def fetch(i, j):
            if (i, j) == wet:
                return (np.array([0.0, 10.0]), np.array([15.0, 14.0]),
                        np.array([38.0, 38.1]))
            return np.array([]), np.array([]), np.array([])

        z, t, s, i, j = _nearest_wet_column(fetch, 60, 99, '1.00')
        assert (i, j) == wet
        assert z.size == 2

    def test_longitude_wraps_during_the_search(self):
        import numpy as np
        from uacpy.data.sound_speed import _nearest_wet_column
        n_lon = 360
        wet = (10, 0)

        def fetch(i, j):
            if (i, j) == wet:
                return np.array([0.0]), np.array([15.0]), np.array([38.0])
            return np.array([]), np.array([]), np.array([])

        z, t, s, i, j = _nearest_wet_column(fetch, 10, n_lon - 1, '1.00')
        assert (i, j) == wet, "search must wrap across the antimeridian"

    def test_land_locked_request_still_fails(self):
        import numpy as np
        from uacpy.data.sound_speed import _nearest_wet_column
        empty = (np.array([]), np.array([]), np.array([]))
        z, t, s, i, j = _nearest_wet_column(lambda i, j: empty, 60, 100, '1.00')
        assert z.size == 0, "a genuinely dry region must not be papered over"


class TestDeepSSPExtension:
    """Analysed T/S products (WOA23, 1 deg) are routinely shallower than
    bathymetry (GEBCO, 15 arc-sec). Holding the last sound speed drops the
    whole pressure term: 61 m/s over the bottom 3.3 km in the Izu-Bonin
    trench."""

    @staticmethod
    def _profile():
        from uacpy.core.environment import SoundSpeedProfile
        # A deep tail at the adiabatic gradient: 1.85 m/s per 100 m.
        z = np.array([0.0, 1000.0, 5300.0, 5400.0, 5500.0])
        c = np.array([1540.0, 1484.0, 1547.34, 1549.20, 1551.05])
        return SoundSpeedProfile.from_pairs(np.column_stack([z, c]))

    def test_extends_along_the_profiles_own_gradient(self):
        import warnings as _w
        from uacpy.data.sound_speed import extend_ssp_below_data
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            out = extend_ssp_below_data(self._profile(), 8801.0, latitude=29.78)
        c_end = float(np.asarray(out.data)[-1, 0])
        # UNESCO at 8801 m holding the deepest T/S gives 1611.93 m/s. The
        # tolerance is tight on purpose: at 2 m/s a fixed 0.0165 s^-1 gradient
        # (6.4 m/s slow here) is only just excluded, and a fixed 0.017 is not.
        assert c_end == pytest.approx(1611.93, abs=0.05), (
            f"deep sound speed {c_end:.2f} m/s — holding the last value would "
            f"give 1551.05, which is 61 m/s slow")
        assert float(np.asarray(out.depths)[-1]) == pytest.approx(8801.0)

    def test_no_single_gradient_can_reproduce_the_extension(self):
        """dc/dz is a function of depth (0.0168 s^-1 at 1 km against 0.0189 at
        8 km, UNESCO), so the same sound speed extended over the same span must
        gain *more* starting deeper. Any fixed-gradient implementation returns
        exactly equal increments here."""
        from uacpy.data.sound_speed import _deep_increment
        shallow = _deep_increment(1500.0, 1500.0, 2500.0, 45.0)
        deep = _deep_increment(1500.0, 5500.0, 6500.0, 45.0)
        assert deep > shallow + 0.5, (
            f"increment over 1000 m is {shallow:.3f} m/s at 1500 m and "
            f"{deep:.3f} m/s at 5500 m — a constant gradient gives both alike")

    def test_a_fixed_canonical_gradient_is_excluded(self):
        """The shipped 0.0165 s^-1 sat below the UNESCO envelope (0.0168-0.0189)
        at every deep condition, so it was slow by 6.4 m/s in the trench."""
        from uacpy.data.sound_speed import _deep_increment
        span = 8801.0 - 5500.0
        got = _deep_increment(1551.05, 5500.0, 8801.0, 29.78)
        assert got - 0.0165 * span > 5.0, (
            f"increment {got:.2f} m/s against {0.0165 * span:.2f} m/s for the "
            f"old constant — the bias has not been removed")
        assert got / span > 0.0168, "below the UNESCO deep-gradient envelope"

    def test_a_noisy_last_segment_cannot_contaminate_the_extension(self):
        """The old gate admitted any measured gradient in (0.005, 0.030), so a
        noisy WOA bottom pair was extrapolated for kilometres: 0.028 s^-1 over
        3.3 km is +92 m/s against a true +61. The increment must depend only on
        the deepest sound speed, never on the segment above it."""
        import warnings as _w
        from uacpy.core.environment import SoundSpeedProfile
        from uacpy.data.sound_speed import extend_ssp_below_data

        def extend(c_second_last):
            p = SoundSpeedProfile.from_pairs(np.array(
                [[0.0, 1540.0], [1000.0, 1484.0],
                 [5400.0, c_second_last], [5500.0, 1551.05]]))
            with _w.catch_warnings():
                _w.simplefilter('ignore')
                out = extend_ssp_below_data(p, 8801.0, latitude=29.78)
            return float(np.asarray(out.data)[-1, 0])

        clean = extend(1549.20)                       # 0.0185 s^-1
        noisy = extend(1548.25)                       # 0.028  s^-1, was accepted
        inverted = extend(1560.00)                    # negative, was rejected
        assert clean == pytest.approx(noisy, abs=1e-9)
        assert clean == pytest.approx(inverted, abs=1e-9)

    def test_an_unphysical_column_clamps_rather_than_diverging(self):
        """A sound speed outside UNESCO's range over the whole -3..35 C bracket
        cannot be inverted; the extension must stay finite and monotone."""
        from uacpy.data.sound_speed import _deep_increment
        for c_absurd in (900.0, 2200.0):
            got = _deep_increment(c_absurd, 1010.0, 3000.0, 45.0)
            assert np.isfinite(got) and got > 0.0, (
                f"c={c_absurd} m/s gave increment {got}")

    def test_warm_deep_basins_are_handled(self):
        """Mediterranean (~13 C) and Red Sea (~21 C) deep water are far off the
        canonical polar values, and dc/dz falls with temperature. The inversion
        recovers it from the column itself."""
        from uacpy.core.acoustics import soundspeed_unesco
        from uacpy.data._geo import depth_to_pressure_dbar
        from uacpy.data.sound_speed import _deep_increment
        for t_true, s_true in ((13.0, 38.5), (21.0, 40.5)):
            p0 = float(depth_to_pressure_dbar(1500.0, 45.0))
            p1 = float(depth_to_pressure_dbar(3000.0, 45.0))
            c0 = soundspeed_unesco(t_true, s_true, p0)
            truth = soundspeed_unesco(t_true, s_true, p1) - c0
            assert _deep_increment(c0, 1500.0, 3000.0, 45.0) == pytest.approx(
                truth, abs=0.15)

    def test_long_extrapolation_warns(self):
        from uacpy.data.sound_speed import extend_ssp_below_data
        with pytest.warns(UserWarning, match="extrapolated"):
            extend_ssp_below_data(self._profile(), 8801.0)

    def test_short_extension_is_quiet(self):
        import warnings as _w
        from uacpy.data.sound_speed import extend_ssp_below_data
        with _w.catch_warnings():
            _w.simplefilter('error')
            extend_ssp_below_data(self._profile(), 5510.0)

    def test_shallower_target_still_trims(self):
        import warnings as _w
        from uacpy.data.sound_speed import extend_ssp_below_data
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            out = extend_ssp_below_data(self._profile(), 2000.0)
        assert float(np.asarray(out.depths)[-1]) == pytest.approx(2000.0)

    def test_a_single_level_profile_still_extends(self):
        """The increment needs only the deepest sound speed, so a profile with
        no segment to measure is no longer a special case."""
        import warnings as _w
        from uacpy.core.environment import SoundSpeedProfile
        from uacpy.data.sound_speed import extend_ssp_below_data, _deep_increment
        p = SoundSpeedProfile.from_pairs(np.array([[1500.0, 1500.0]]))
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            out = extend_ssp_below_data(p, 3000.0, latitude=45.0)
        expected = 1500.0 + _deep_increment(1500.0, 1500.0, 3000.0, 45.0)
        assert float(np.asarray(out.data)[-1, 0]) == pytest.approx(expected)

    def test_each_column_is_extended_from_its_own_deep_value(self):
        """A range-dependent profile carries one column per range; a cold column
        and a warm one must not be given the same increment."""
        import warnings as _w
        from uacpy.core.environment import SoundSpeedProfile
        from uacpy.data.sound_speed import extend_ssp_below_data
        p = SoundSpeedProfile(
            depths=np.array([1000.0, 5500.0]),
            data=np.array([[1484.0, 1500.0], [1551.05, 1575.0]]),
            ranges=np.array([0.0, 50.0]), shape='measured')
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            out = extend_ssp_below_data(p, 8801.0, latitude=29.78)
        cold, warm = np.asarray(out.data)[-1, :]
        assert (cold - 1551.05) - (warm - 1575.0) > 0.5, (
            f"increments {cold - 1551.05:.2f} and {warm - 1575.0:.2f} m/s — "
            f"dc/dz falls with temperature, so the colder column gains more")
