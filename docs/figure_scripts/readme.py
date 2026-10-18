"""The README's real-world figure, ``docs/readme_realworld.png``.

Example 37's first composite (``uacpy/examples/example_37_realworld_environment.py``,
its ``example_37_realworld.png``) with the example's own settings: a GEBCO map
grid, a range-dependent North Sea transect (GEBCO seafloor, WOA23 sound speed,
the cached seabed, the site's absorption), Bellhop (Fortran) at 800 Hz, and
``plot_overview``. Every source it names reads the install-time cache first
(``./install.sh --data all``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import uacpy

# Write into docs/, where the README links the figure.
FIGURE_DIR = Path(__file__).resolve().parent.parent

REGION_LAT, REGION_LON = (56.5, 62.0), (-2.0, 9.0)
TRANSECT_START, TRANSECT_END = (61.0, 2.0), (58.0, 5.0)   # shelf → trench


def realworld():
    """Map · transmission loss · section, from one plot_overview call."""
    grid = uacpy.data.fetch_bathy_grid(REGION_LAT, REGION_LON, n_lat=400,
                                       n_lon=400, source='gebco')
    env = uacpy.data.fetch_environment(
        TRANSECT_START, date='2026-01-15', transect_to=TRANSECT_END,
        name="North Sea — Norwegian Trench transect",
        n_points=40, bathymetry_sources='gebco',
        range_dependent_ssp=True, ssp_n_points=5,
        range_dependent_bottom=True, bottom_sources='local',
        bottom_n_points=5, with_absorption=True)
    source = uacpy.Source(depths=100.0, frequencies=800.0)
    receiver = uacpy.Receiver(
        depths=np.linspace(1, env.bathymetry.depth, 150),
        ranges=np.linspace(1000.0, env.bathymetry.range_max, 350))
    tl = uacpy.Bellhop(backend='fortran').run(env, source, receiver)
    fig, _ = uacpy.plot.plot_overview(
        env, grid, transect=(TRANSECT_START, TRANSECT_END),
        tl=tl, source=source, receiver=receiver,
        map_title="North Sea — Norwegian Trench (GEBCO)",
        tl_title="Transmission loss (Bellhop, 800 Hz)",
        env_title="Range-dependent environment A→B",
        title="uacpy — real-world environment from GPS, modelled & plotted",
        map_kwargs=dict(contours=True, aspect=1, coastline_resolution='50m',
                        graticule=2.0, graticule_minor=1.0))
    return fig


FIGURES = {
    'readme_realworld': realworld,
}
