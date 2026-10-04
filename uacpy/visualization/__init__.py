"""Visualization tools for underwater acoustics.

The plotters are reached through :mod:`uacpy.plot` (``uacpy.plot.plot_field``,
``uacpy.plot.plot_overview``, ...), their one public path; they are defined in
:mod:`uacpy.visualization.plots`, the implementation module package code
imports. Every carrier and result also draws itself with ``.plot()``.

This package holds what is not a plotter:

* :mod:`~uacpy.visualization.style` — the colour palette (field colormaps,
  sediment fill/hatch styles, source/receiver markers).
* :func:`land_polygons` — the Natural Earth land rings the map plotters draw
  behind a chart, for a map of your own; :func:`download_coastline` caches
  them for offline use, the way ``uacpy.data``'s ``download_*_db`` fetchers
  cache their grids.

Importing this module does not mutate ``matplotlib.rcParams``.
"""

from uacpy.visualization import style
from uacpy.visualization.basemap import download_coastline, land_polygons

__all__ = [
    # the coastline backdrop the map plotters draw, and its cache
    'land_polygons',
    'download_coastline',
    'style',
]
