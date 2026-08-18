"""Shared builders for the synthetic ``$UACPY_DATA_CACHE`` stubs.

A plain helper module (not a conftest): the ``test_data_*`` files import the
NetCDF/CSV writers below to populate a per-test cache directory so the
offline backends run with no network. The NetCDF writers need ``netCDF4`` —
their callers run ``pytest.importorskip('netCDF4')`` before importing this
module, and the guarded import below keeps the module importable for the
CSV/network-token users without it.
"""

import numpy as np
import pytest

try:
    import netCDF4
except ImportError:            # the NetCDF writers' callers importorskip first
    netCDF4 = None

_FILL = 9.96921e36                              # WOA23's netCDF _FillValue


def _write_gebco(cache, *, deep=-1500.0, land_at=None):
    """A 1° stand-in for GEBCO_2025.nc: ``elevation(lat, lon)``, positive up.

    The real grid is 43200×86400 at 15″ with cell-centre axes starting at
    ±89.99791666…/±179.99791666…; this one is 1° with axes on whole degrees.
    ``NetcdfGrid`` reads origin and step from the file's own axis variables, so
    it snaps to the nearest stored node under either registration and the
    coarser fixture exercises the same code path. ``land_at`` puts one +25 m
    cell at the named (lat, lon).
    """
    gdir = cache / 'gebco'; gdir.mkdir(parents=True)
    lat = np.arange(-90, 91, 1.0)
    lon = np.arange(-180, 180, 1.0)
    elev = np.full((lat.size, lon.size), deep)
    if land_at is not None:
        i = int(round(land_at[0] + 90)); j = int(round(land_at[1] + 180))
        elev[i, j] = 25.0
    ds = netCDF4.Dataset(gdir / 'GEBCO_2025.nc', 'w')
    ds.createDimension('lat', lat.size); ds.createDimension('lon', lon.size)
    ds.createVariable('lat', 'f8', ('lat',))[:] = lat
    ds.createVariable('lon', 'f8', ('lon',))[:] = lon
    ds.createVariable('elevation', 'f4', ('lat', 'lon'))[:] = elev
    ds.close()


def _write_woa(cache, *, periods=('00',)):
    """WOA23 1° analysed-mean grids: ``t_an``/``s_an`` of shape (time, depth,
    lat, lon) = (1, n_depth, 180, 360), everything but one column set to the
    product's ``_FillValue``. The real annual file carries 102 levels to 5500 m
    and the monthlies 57 to 1500 m; five levels is enough for the readers.

    ``periods`` names the WOA period codes to write: '00' is the annual
    (date-less tests); adding '03' (March) lets date-carrying tests (sea ice)
    resolve WOA from the cache instead of falling through to a live OPeNDAP
    request.
    """
    wdir = cache / 'woa23'; wdir.mkdir(parents=True)
    depth = np.array([0, 50, 100, 500, 1000.0])
    # column for grid_index(30.5, -40.5) == (120, 139) on the 1° grid.
    def mk(var, vals):
        arr = np.full((1, depth.size, 180, 360), _FILL)
        arr[0, :, 120, 139] = vals
        ds = netCDF4.Dataset(wdir / f'woa23_decav_{var}_01.nc', 'w')
        for d, n in [('time', 1), ('depth', depth.size), ('lat', 180), ('lon', 360)]:
            ds.createDimension(d, n)
        ds.createVariable('depth', 'f4', ('depth',))[:] = depth
        name = 't_an' if var[0] == 't' else 's_an'
        ds.createVariable(name, 'f4', ('time', 'depth', 'lat', 'lon'))[:] = arr
        ds.close()
    for period in periods:
        mk(f't{period}', [18, 16, 13, 8, 5.0])
        mk(f's{period}', [36, 36.1, 36.2, 35.5, 35.0])


def _write_sediment(cache, *, deck41=True, ligurian=True):
    """Grain-size sample CSV, plus the DECK41 lithology CSV with ``deck41``.

    The (30.5, -40.5) sample is the point every offline fetch queries;
    ``ligurian`` adds the (43.0, 7.0) Mediterranean sample the offline
    suite's second test point uses.
    """
    sdir = cache / 'sediment'; sdir.mkdir(parents=True)
    rows = 'latitude,longitude,mean_phi\n30.5,-40.5,3.0\n'
    if ligurian:
        rows += '43.0,7.0,2.0\n'
    (sdir / 'grainsize.csv').write_text(rows)
    if deck41:
        (sdir / 'deck41.csv').write_text(
            'latitude,longitude,lithology\n44.0,8.0,Sand\n10.0,20.0,Clay\n')


def _write_globsed(cache, *, marked_cells=False):
    # GlobSed is gridline-registered — its axes hit ±90 / ±180 exactly and both
    # ±180 columns are stored, so the node count is odd (the real v3 grid is
    # 2161×4321 at 5′). Keep that here: 181×361 on whole degrees.
    gdir = cache / 'globsed'; gdir.mkdir(parents=True, exist_ok=True)
    ds = netCDF4.Dataset(gdir / 'GlobSed-v3.nc', 'w')
    ds.createDimension('lat', 181); ds.createDimension('lon', 361)
    ds.createVariable('lat', 'f8', ('lat',))[:] = np.linspace(-90, 90, 181)
    ds.createVariable('lon', 'f8', ('lon',))[:] = np.linspace(-180, 180, 361)
    if marked_cells:
        z = np.full((181, 361), 500.0)
        zv = ds.createVariable('z', 'f4', ('lat', 'lon'), fill_value=np.nan)
        z[0, 0] = np.nan                                   # one no-data cell
        z[70, 20] = 149.0        # (-20, -160): distinct value for the wrap tests
        z[100, 190] = 0.0        # (10, 10): bare basement (zero thickness)
        z[120, 0] = 77.0         # (30, -180): distinct from the +180 end column
        zv[:] = z
    else:
        ds.createVariable('z', 'f4', ('lat', 'lon'))[:] = 500.0  # uniform 500 m
    ds.close()


def _write_crust1(cache):
    cdir = cache / 'crust1'; cdir.mkdir(parents=True, exist_ok=True)
    n = 180 * 360                                  # one whitespace row per 1° cell
    # Nine columns per row, in CRUST1.0's layer order: water, ice, upper/middle/
    # lower sediment, upper/middle/lower crystalline crust, mantle. ``bnds`` is
    # the top of each layer in km (negative down), so this column is 4 km of
    # water over 1 km of sediment (-4 → -5) over crust.
    rows = {
        'crust1.bnds': [0, -4, -4, -5, -5, -5, -10, -20, -30],
        'crust1.vp':   [1.5, 3.8, 2.0, 0, 0, 5.0, 6.5, 7.1, 8.1],
        'crust1.vs':   [0, 1.9, 0.6, 0, 0, 2.7, 3.7, 4.0, 4.5],
        'crust1.rho':  [1.02, 0.9, 1.9, 0, 0, 2.6, 2.8, 3.0, 3.3],
    }
    # Every cell is identical, so repeat one formatted line — far cheaper than
    # np.savetxt over the full 64800×9 grid in this per-test fixture.
    for name, row in rows.items():
        line = ' '.join('%g' % v for v in row) + '\n'
        (cdir / name).write_text(line * n)


_NETWORK_DOWN_TOKENS = ('could not reach', 'timed out', 'timeout', 'connection',
                        'unreachable', 'http 502', 'http 503', 'http 504')


def _skip_or_fail(exc, service):
    """Skip only on network-level failure; a structured server rejection
    (bad variable / constraint / axis) is a real bug and must fail."""
    if any(tok in exc.message.lower() for tok in _NETWORK_DOWN_TOKENS):
        pytest.skip(f"{service} unreachable: {exc.message}")
    pytest.fail(f"{service} rejected the query: {exc.message}")
