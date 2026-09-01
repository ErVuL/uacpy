"""Tests for the low-frequency seabed backends: GlobSed + CRUST1.0.

Builds a tiny synthetic ``$UACPY_DATA_CACHE`` (1° GlobSed grid + a uniform-ocean
CRUST1.0 column) so the thickness reader and the layered-elastic bottom builder
run fully offline. Skipped where netCDF4 (the grid dependency) is unavailable.
"""

import io
import tarfile
import threading
import warnings

import numpy as np
import pytest

from uacpy.data import _cache

netCDF4 = pytest.importorskip('netCDF4')

from uacpy.core.environment import SeabedColumn, Bottom
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import crust1_local, globsed_local
from uacpy.data import _http
from uacpy.tests._cache_builders import _write_crust1, _write_globsed


@pytest.fixture(scope='module')
def _seis_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('seis_cache')
    _write_globsed(root, marked_cells=True)
    _write_crust1(root, marked_cells=True)
    return root


@pytest.fixture
def cache(_seis_root, monkeypatch):
    monkeypatch.setenv('UACPY_DATA_CACHE', str(_seis_root))
    _cache.invalidate_grids()
    crust1_local._MODEL.clear()
    return _seis_root


# ── GlobSed ──────────────────────────────────────────────────────────────────

def test_globsed_thickness(cache):
    assert globsed_local.fetch_sediment_thickness((30.0, -40.0)) == 500.0


def test_globsed_no_data_raises(cache):
    with pytest.raises(DataFetchError, match='no sediment thickness'):
        globsed_local.fetch_sediment_thickness((-90.0, -180.0))   # the NaN cell


def test_globsed_transect(cache):
    r, thk = globsed_local.fetch_sediment_thickness_transect(
        (30.0, -40.0), (31.0, -40.0), n_points=3)
    assert r.shape == (3,) and thk.shape == (3,) and np.allclose(thk, 500.0)


def test_globsed_0_360_longitude_wraps(cache):
    # (-20, 200) and (-20, -160) are the same physical point; a [0, 360)
    # longitude must wrap onto the stored [-180, 180] gridline axis rather
    # than clip to the +180 edge column.
    assert globsed_local.fetch_sediment_thickness((-20.0, 200.0)) == 149.0
    assert globsed_local.fetch_sediment_thickness((-20.0, -160.0)) == 149.0


def test_globsed_plus_180_reads_minus_180_column(cache):
    # Both meridian ends are stored on the gridline axis; +180 resolves to the
    # -180 column (identical values in the real product; distinct here so the
    # test observes which column was read).
    assert globsed_local.fetch_sediment_thickness((30.0, -180.0)) == 77.0
    assert globsed_local.fetch_sediment_thickness((30.0, 180.0)) == 77.0


# ── CRUST1.0 ─────────────────────────────────────────────────────────────────

def test_crust1_profile(cache):
    p = crust1_local.fetch_crust1_profile((30.0, -40.0))
    assert p['water_depth_m'] == 4000.0
    assert p['sediment_thickness_m'] == 1000.0


def test_crust1_bottom_layered_elastic(cache):
    b = crust1_local.fetch_bottom_crust1((30.0, -40.0))
    assert isinstance(b, SeabedColumn)
    assert b.layers[0].sound_speed == 2000.0       # 2.0 km/s → m/s
    assert b.layers[0].shear_speed == 600.0        # 0.6 km/s → m/s (elastic)
    assert b.halfspace.sound_speed == 5000.0       # upper crystalline crust


def test_crust1_fluid_option(cache):
    b = crust1_local.fetch_bottom_crust1((30.0, -40.0), elastic=False)
    assert all(layer.shear_speed == 0 for layer in b.layers)
    assert b.halfspace.shear_speed == 0


def test_crust1_thickness_rescale(cache):
    b = crust1_local.fetch_bottom_crust1((30.0, -40.0), sediment_thickness=2000.0)
    assert b.total_thickness() == pytest.approx(2000.0)
    assert b.sediment_thickness_source is None      # explicit, not GlobSed


def test_crust1_uses_globsed_by_default(cache):
    # GlobSed (500 m here) rescales CRUST1.0's native 1000 m column by default,
    # and the bottom records that GlobSed supplied the thickness.
    b = crust1_local.fetch_bottom_crust1((30.0, -40.0))
    assert b.total_thickness() == pytest.approx(500.0)
    assert b.sediment_thickness_source == 'globsed'


def test_crust1_globsed_zero_thickness_yields_bare_rock(cache):
    # A genuine GlobSed 0.0 (bare basement) wins over CRUST1.0's own sediment
    # column: the bottom is bare rock, and the stamp still names GlobSed as
    # the thickness source.
    b = crust1_local.fetch_bottom_crust1((10.0, 10.0))
    assert b.sediment_thickness_source == 'globsed'
    assert all(layer.sound_speed >= 5000.0 for layer in b.layers)  # crust, not sediment
    assert b.halfspace.sound_speed == 6500.0        # middle crystalline crust


def test_crust1_zero_sediment_column_warns_and_is_not_stamped_globsed(cache):
    """On a column where CRUST1.0 has no sediment layers there are no sediment
    Vp/Vs/ρ to rescale, so a positive GlobSed thickness cannot be applied: the
    discard is said out loud — naming the thickness, the cell and the reason —
    and the stamp reads ``'globsed-ignored'``, not the ``'globsed'`` it used to
    claim while ignoring GlobSed."""
    with pytest.warns(UserWarning, match='no sediment layers') as rec:
        b = crust1_local.fetch_bottom_crust1((82.5, -56.5))
    assert b.sediment_thickness_source == 'globsed-ignored'
    assert b.layers[0].sound_speed == 5000.0        # upper crystalline crust
    assert b.halfspace.sound_speed == 6500.0        # middle crystalline crust
    msg = next(str(w.message) for w in rec
               if 'no sediment layers' in str(w.message))
    assert '500' in msg and '82.50' in msg and '-56.50' in msg


def test_crust1_zero_sediment_column_warns_on_an_explicit_thickness_too(cache):
    """An explicit ``sediment_thickness`` is discarded on a zero-sediment
    column for the same reason as a GlobSed value, and just as audibly; the
    stamp stays ``None`` (the value never came from GlobSed)."""
    with pytest.warns(UserWarning, match='no sediment layers') as rec:
        b = crust1_local.fetch_bottom_crust1((82.5, -56.5),
                                             sediment_thickness=200.0)
    assert b.sediment_thickness_source is None
    assert b.layers[0].sound_speed == 5000.0
    assert '200' in next(str(w.message) for w in rec
                         if 'no sediment layers' in str(w.message))


def test_crust1_zero_sediment_column_is_quiet_below_the_layer_threshold(cache):
    """Below ``_MIN_SEDIMENT_M`` the answer is bare rock on any column, so
    nothing is discarded by the zero-sediment branch and the notice stays
    quiet — the far side of the threshold the warning fires on."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        b = crust1_local.fetch_bottom_crust1(
            (82.5, -56.5), sediment_thickness=crust1_local._MIN_SEDIMENT_M / 2)
    assert not [w for w in rec if 'no sediment layers' in str(w.message)]
    assert b.layers[0].sound_speed == 5000.0


def test_crust1_sediment_bearing_column_does_not_warn_about_discards(cache):
    """The discard notice belongs to zero-sediment columns only: a normal cell
    rescales to GlobSed silently (bar the licence notice) and keeps the
    ``'globsed'`` stamp."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        b = crust1_local.fetch_bottom_crust1((30.0, -40.0))
    assert b.sediment_thickness_source == 'globsed'
    assert not [w for w in rec if 'no sediment layers' in str(w.message)]


def test_crust1_globsed_fallback_keeps_native(cache):
    # Where GlobSed has no value, CRUST1.0's own thickness is kept (no marker).
    b = crust1_local.fetch_bottom_crust1((-90.0, -180.0))   # GlobSed NaN cell
    assert b.total_thickness() == pytest.approx(1000.0)
    assert b.sediment_thickness_source is None


def test_crust1_use_globsed_false(cache):
    b = crust1_local.fetch_bottom_crust1((30.0, -40.0), use_globsed=False)
    assert b.total_thickness() == pytest.approx(1000.0)
    assert b.sediment_thickness_source is None


def test_crust1_transect(cache):
    rdl = crust1_local.fetch_bottom_crust1_transect(
        (30.0, -40.0), (31.0, -40.0), n_points=3)
    assert isinstance(rdl, Bottom)
    assert len(rdl.columns) == 3 and rdl.ranges[0] == 0.0
    assert rdl.sediment_thickness_source == 'globsed'   # GlobSed applied per point


@pytest.mark.parametrize("point, shape", [
    ((30.0, -40.0), 'sediment over basement'),
    ((10.0, 10.0), 'bare rock'),        # the other column shape the builder makes
])
def test_crust1_roughness_lands_on_the_seafloor_interface(cache, point, shape):
    """``SedimentLayer.roughness`` is the interface at the *top* of its layer,
    so the seafloor is the first layer's. Every other bottom fetcher takes
    ``roughness=``; these two claim signature parity with them."""
    assert crust1_local.fetch_bottom_crust1(point).layers[0].roughness == 0.0
    rough = crust1_local.fetch_bottom_crust1(point, roughness=0.5)
    assert rough.layers[0].roughness == pytest.approx(0.5), shape


def test_crust1_roughness_is_the_top_interface_not_the_deepest(cache):
    """The cached fixture's columns are single-layer, where the seafloor and
    the deepest interface are the same object. A column with three sediment
    layers separates them: only the topmost one is the water/seabed
    interface."""
    n = crust1_local._MID_CRYST + 2
    boundaries = np.array([0.0, -0.3, -0.9, -1.8, -6.0, -12.0, -20.0, -30.0])[:n]
    column = crust1_local._layered_from_column(
        boundaries, np.linspace(1.8, 7.0, n), np.linspace(0.5, 4.0, n),
        np.linspace(1.7, 3.0, n), sediment_attenuation=0.5,
        basement_attenuation=0.1, elastic=True, sediment_thickness=None,
        roughness=0.5)
    assert len(column.layers) > 1, "the synthetic column collapsed to one layer"
    assert [layer.roughness for layer in column.layers] == [0.5] + [0.0] * (
        len(column.layers) - 1)


def test_crust1_roughness_leaves_the_geoacoustics_alone(cache):
    """The knob is an interface property; it must not move Vp, Vs, ρ or the
    sediment column."""
    plain = crust1_local.fetch_bottom_crust1((30.0, -40.0))
    rough = crust1_local.fetch_bottom_crust1((30.0, -40.0), roughness=0.5)
    assert ([(l.thickness, l.sound_speed, l.density, l.shear_speed)
             for l in rough.layers]
            == [(l.thickness, l.sound_speed, l.density, l.shear_speed)
                for l in plain.layers])
    assert rough.halfspace.sound_speed == plain.halfspace.sound_speed
    assert rough.total_thickness() == pytest.approx(plain.total_thickness())


def test_crust1_transect_carries_roughness_to_every_waypoint(cache):
    rdl = crust1_local.fetch_bottom_crust1_transect(
        (30.0, -40.0), (31.0, -40.0), n_points=3, roughness=0.5)
    assert [c.layers[0].roughness for c in rdl.columns] == [0.5, 0.5, 0.5]
    plain = crust1_local.fetch_bottom_crust1_transect(
        (30.0, -40.0), (31.0, -40.0), n_points=3)
    assert [c.layers[0].roughness for c in plain.columns] == [0.0, 0.0, 0.0]


def test_every_bottom_fetcher_takes_a_roughness_argument():
    """The parity ``fetch_bottom_crust1``'s docstring claims with the network
    bottom fetchers, checked across the whole public family."""
    import inspect

    import uacpy.data as data

    missing = [name for name in data.__all__ if name.startswith('fetch_bottom')
               if 'roughness' not in inspect.signature(
                   getattr(data, name)).parameters]
    assert missing == []


def test_crust1_transect_accepts_max_points(cache):
    # environment._fetch_bottom forwards max_points to every transect bottom
    # fetcher; crust1's must accept it and clamp n_points to it.
    rdl = crust1_local.fetch_bottom_crust1_transect(
        (30.0, -40.0), (31.0, -40.0), n_points=6, max_points=3)
    assert isinstance(rdl, Bottom)
    assert len(rdl.columns) == 3                         # clamped to max_points


def test_crust1_emits_commercial_warning(cache):
    # CRUST1.0 is non-commercial (no formal licence); the low-level fetcher must
    # warn so a direct caller is never silent, and the transect fetcher must
    # warn exactly once (not once per waypoint).
    with pytest.warns(UserWarning, match='commercial'):
        crust1_local.fetch_bottom_crust1((30.0, -40.0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        crust1_local.fetch_bottom_crust1_transect(
            (30.0, -40.0), (31.0, -40.0), n_points=4)
    commercial = [w for w in caught
                  if issubclass(w.category, UserWarning)
                  and 'commercial' in str(w.message)]
    assert len(commercial) == 1


def test_crust1_transect_opens_no_process_global_filter_window(cache):
    # The transect buys its once-per-fetch notice from the non-warning builder,
    # not from a ``warnings.catch_warnings()`` window: ``warnings.filters`` is
    # process-global, so such a window mutes every thread for as long as it is
    # held.
    opened = []
    real_catch_warnings = warnings.catch_warnings

    class _CountingCatchWarnings(real_catch_warnings):
        def __enter__(self):
            opened.append(1)
            return super().__enter__()

    warnings.catch_warnings = _CountingCatchWarnings
    try:
        crust1_local.fetch_bottom_crust1_transect(
            (30.0, -40.0), (31.0, -40.0), n_points=3)
    finally:
        warnings.catch_warnings = real_catch_warnings
    assert opened == []


def test_crust1_transect_lets_another_thread_raise_the_same_notice(cache):
    # The transect runs on one thread while another raises the very notice the
    # transect wants quiet for its own waypoints. Only that thread's copy may
    # be silenced. The probe is released from inside a waypoint build, so no
    # timing decides the outcome.
    inside_a_waypoint = threading.Event()
    probe_raised = threading.Event()
    delivered = []
    real_builder = crust1_local._bottom_at_point

    def releasing_builder(*args, **kwargs):
        inside_a_waypoint.set()
        probe_raised.wait(30.0)
        return real_builder(*args, **kwargs)

    def run_the_transect():
        try:
            crust1_local.fetch_bottom_crust1_transect(
                (30.0, -40.0), (31.0, -40.0), n_points=3)
        finally:
            inside_a_waypoint.set()

    with warnings.catch_warnings():
        warnings.simplefilter('always')
        warnings.showwarning = (
            lambda message, *a, **k: delivered.append(str(message)))
        crust1_local._bottom_at_point = releasing_builder
        try:
            worker = threading.Thread(target=run_the_transect)
            worker.start()
            assert inside_a_waypoint.wait(30.0)
            before = delivered.count(crust1_local._COMMERCIAL_WARNING)
            warnings.warn(crust1_local._COMMERCIAL_WARNING, UserWarning)
            after = delivered.count(crust1_local._COMMERCIAL_WARNING)
            probe_raised.set()
            worker.join(30.0)
        finally:
            crust1_local._bottom_at_point = real_builder
    assert not worker.is_alive()
    assert before == 1                  # the transect's own, raised once
    assert after == 2                   # and this thread's, not swallowed
    assert delivered.count(crust1_local._COMMERCIAL_WARNING) == 2


def test_crust1_missing_cache_names_flag(tmp_path, monkeypatch):
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    crust1_local._MODEL.clear()
    with pytest.raises(ConfigurationError, match='install.sh --data crust1'):
        crust1_local.fetch_bottom_crust1((30.0, -40.0))


# ── downloads (mocked) ───────────────────────────────────────────────────────

def test_globsed_download(tmp_path, monkeypatch):
    src = tmp_path / 'src.nc'
    ds = netCDF4.Dataset(src, 'w')
    ds.createDimension('lat', 2); ds.createDimension('lon', 2)
    ds.createVariable('lat', 'f8', ('lat',))[:] = [0, 1]
    ds.createVariable('lon', 'f8', ('lon',))[:] = [0, 1]
    ds.createVariable('z', 'f4', ('lat', 'lon'))[:] = 1.0
    ds.close()
    blob = src.read_bytes()
    # Force the urllib fallback (no real curl to NCEI) and mock the fetch.
    monkeypatch.setattr(globsed_local, 'curl_download', lambda *a, **k: False)
    monkeypatch.setattr(globsed_local, 'http_get', lambda url, **kw: blob)
    out = globsed_local.download_globsed_db(cache_dir=str(tmp_path / 'o'))
    assert out.exists() and out.name == 'GlobSed-v3.nc'


def test_crust1_download(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tf:
        for name in ('crust1.bnds', 'crust1.vp', 'crust1.vs', 'crust1.rho'):
            blob = b'0 0 0 0 0 0 0 0 0\n'
            info = tarfile.TarInfo(name); info.size = len(blob)
            tf.addfile(info, io.BytesIO(blob))
    monkeypatch.setattr(crust1_local, 'http_get', lambda url, **kw: buf.getvalue())
    out = crust1_local.download_crust1_db(cache_dir=str(tmp_path / 'o'))
    assert (out / 'crust1.bnds').exists()


def test_globsed_interrupted_curl_no_final_file(tmp_path, monkeypatch):
    """A curl killed mid-transfer must not poison the cache with a truncated
    grid at the final path."""
    from pathlib import Path
    dest = tmp_path / 'o'

    def fake_run(cmd, **kw):
        Path(cmd[cmd.index('-o') + 1]).write_bytes(b'truncated')
        raise KeyboardInterrupt

    monkeypatch.setattr(_http.shutil, 'which', lambda n: '/usr/bin/curl')
    monkeypatch.setattr(_http.subprocess, 'run', fake_run)
    with pytest.raises(KeyboardInterrupt):
        globsed_local.download_globsed_db(cache_dir=str(dest))
    assert not (dest / globsed_local.GLOBSED_FILE).exists()
