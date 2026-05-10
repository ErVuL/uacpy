"""
Visualization tools for underwater acoustics

Available modules:
- plots: Full-featured plotting functions with extensive customization
- style: Professional matplotlib styling configuration

The uacpy rcParams are an opt-in. To use them, call
``uacpy.visualization.style.apply_professional_style()`` yourself once at
the top of your script. Importing :mod:`uacpy.visualization` does **not**
mutate ``matplotlib.rcParams``.
"""

from uacpy.visualization import style
from uacpy.visualization.style import apply_professional_style

from uacpy.visualization.plots import (
    plot_transmission_loss,
    plot_transmission_loss_polar,
    plot_rays,
    plot_ssp,
    plot_ssp_2d,
    plot_bathymetry,
    plot_arrivals,
    plot_time_series,
    plot_time_trace,
    plot_environment,
    plot_environment_advanced,
    plot_bottom_properties,
    plot_bottom_loss,
    plot_layered_bottom,
    plot_rd_layered_bottom,
    plot_rd_bottom,
    plot_modes,
    plot_mode_functions,
    plot_modes_heatmap,
    compare_models,
    plot_range_cut,
    plot_depth_cut,
    compare_range_cuts,
    plot_model_comparison_matrix,
    plot_reflection_coefficient,
    plot_reflection_coefficient_heatmap,
    plot_mode_wavenumbers,
    plot_dispersion_curves,
    plot_transfer_function,
    plot_transfer_function_slice,
    plot_phase_field,
    plot_covariance,
    plot_replicas,
    plot_tl_difference,
)

__all__ = [
    'plot_transmission_loss',
    'plot_transmission_loss_polar',
    'plot_rays',
    'plot_ssp',
    'plot_ssp_2d',
    'plot_bathymetry',
    'plot_arrivals',
    'plot_time_series',
    'plot_time_trace',
    'plot_environment',
    'plot_environment_advanced',
    'plot_bottom_properties',
    'plot_bottom_loss',
    'plot_layered_bottom',
    'plot_rd_layered_bottom',
    'plot_rd_bottom',
    'plot_modes',
    'plot_mode_functions',
    'plot_modes_heatmap',
    'compare_models',
    'plot_range_cut',
    'plot_depth_cut',
    'compare_range_cuts',
    'plot_model_comparison_matrix',
    'plot_reflection_coefficient',
    'plot_reflection_coefficient_heatmap',
    'plot_mode_wavenumbers',
    'plot_dispersion_curves',
    'plot_transfer_function',
    'plot_transfer_function_slice',
    'plot_phase_field',
    'plot_covariance',
    'plot_replicas',
    'plot_tl_difference',
    'style',
    'apply_professional_style',
]
