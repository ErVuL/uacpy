"""Visualization tools for underwater acoustics.

Canonical surface
-----------------
* :func:`plot_field` — auto-shape plotter for :class:`~uacpy.Field`. Slice
  with :meth:`Field.at` / :meth:`Field.isel` first to control what gets
  drawn (1-D line cut vs 2-D heatmap).
* :func:`compare` — overlay 1-D sliced fields.
* :func:`compare_models` — side-by-side heatmap grid.
* :func:`plot_rays`, :func:`plot_arrivals` — ray fans, arrival stems.
* :func:`plot_environment` — SSP + bathymetry.
* :func:`plot_mode_functions`, :func:`plot_mode_wavenumbers`,
  :func:`plot_modes_heatmap` — three distinct mode views.
* :func:`plot_reflection_coefficient`, :func:`plot_covariance`,
  :func:`plot_replicas` — niche typed results.

Stylesheet (``apply_professional_style``) is opt-in — importing this
module does not mutate ``matplotlib.rcParams``.
"""

from uacpy.visualization import style
from uacpy.visualization.style import apply_professional_style

from uacpy.visualization.plots import (
    plot_result,
    plot_field,
    compare,
    compare_models,
    plot_rays,
    plot_arrivals,
    plot_environment,
    plot_mode_functions,
    plot_mode_wavenumbers,
    plot_modes_heatmap,
    plot_reflection_coefficient,
    plot_covariance,
    plot_replicas,
)

__all__ = [
    'plot_result',
    'plot_field',
    'compare',
    'compare_models',
    'plot_rays',
    'plot_arrivals',
    'plot_environment',
    'plot_mode_functions',
    'plot_mode_wavenumbers',
    'plot_modes_heatmap',
    'plot_reflection_coefficient',
    'plot_covariance',
    'plot_replicas',
    'style',
    'apply_professional_style',
]
