"""Visualization tools for underwater acoustics.

Canonical surface
-----------------
* :func:`plot_field` — auto-shape plotter for :class:`~uacpy.Field`. Slice
  with :meth:`Field.at` / :meth:`Field.isel` first to control what gets
  drawn (1-D line cut vs 2-D heatmap).
* :func:`animate_field` — animate a time-series ``Field`` (returns a
  :class:`~matplotlib.animation.FuncAnimation`).
* :func:`save_animation` — one-liner GIF/MP4 export wrapping
  :func:`animate_field` (writer inferred from suffix).
* :func:`plot_time_snapshots` — time-series analogue of
  :func:`compare_models`: per-model rows × per-time columns of
  ``p(d, r, t)``.
* :func:`compare` — overlay 1-D sliced fields.
* :func:`compare_models` — side-by-side heatmap grid.
* :func:`plot_signal_excess` — diverging SE heatmap with the SE = 0
  detection-boundary contour (fields from
  :func:`uacpy.sonar.passive_signal_excess_field` /
  :func:`uacpy.sonar.active_signal_excess_field`).
* :func:`plot_detection_probability` — ``P_D`` heatmap on [0, 1] with
  labelled probability contours (fields from
  :func:`uacpy.sonar.probability_of_detection_field`).
* :func:`plot_rays`, :func:`plot_arrivals` — ray fans, arrival stems.
* :func:`plot_environment` — single-panel view of the env: water column
  (SSP, Blues cmap) + bottom (every flavour, YlOrBr cmap) + optional
  ``source=``/``receiver=`` markers, with two independent colorbars.
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
    plot_signal_excess,
    plot_detection_probability,
    animate_field,
    save_animation,
    plot_time_snapshots,
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
    'plot_signal_excess',
    'plot_detection_probability',
    'animate_field',
    'save_animation',
    'plot_time_snapshots',
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
