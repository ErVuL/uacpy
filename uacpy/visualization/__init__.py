"""
Visualization tools for underwater acoustics

Available modules:
- plots: Full-featured plotting functions with extensive customization
- quickplot: Ultra-simple plotting functions for rapid visualization
- style: Professional matplotlib styling configuration

The uacpy rcParams are applied automatically on import. Call
``uacpy.visualization.style.apply_professional_style()`` again after your
own ``mpl.rcParams`` tweaks to reset to the uacpy defaults, or call
``matplotlib.rcdefaults()`` to revert to matplotlib's defaults.
"""

from uacpy.visualization import style
style.apply_professional_style()

from uacpy.visualization.plots import (
    plot_transmission_loss,
    plot_transmission_loss_polar,
    plot_rays,
    plot_ssp,
    plot_ssp_2d,
    plot_bathymetry,
    plot_arrivals,
    plot_time_series,
    plot_environment,
    plot_environment_advanced,
    plot_bottom_properties,
    plot_layered_bottom,
    plot_rd_layered_bottom,
    plot_modes,
    plot_mode_functions,
    plot_modes_heatmap,
    compare_models,
    plot_range_cut,
    plot_depth_cut,
    compare_range_cuts,
    plot_model_statistics,
    plot_model_comparison_matrix,
    plot_comparison_curves,
    plot_reflection_coefficient,
    plot_mode_wavenumbers,
    plot_dispersion_curves,
    plot_transfer_function,
)

from uacpy.visualization import quickplot

__all__ = [
    'plot_transmission_loss',
    'plot_transmission_loss_polar',
    'plot_rays',
    'plot_ssp',
    'plot_ssp_2d',
    'plot_bathymetry',
    'plot_arrivals',
    'plot_time_series',
    'plot_environment',
    'plot_environment_advanced',
    'plot_bottom_properties',
    'plot_layered_bottom',
    'plot_rd_layered_bottom',
    'plot_modes',
    'plot_mode_functions',
    'plot_modes_heatmap',
    'compare_models',
    'plot_range_cut',
    'plot_depth_cut',
    'compare_range_cuts',
    'plot_model_statistics',
    'plot_model_comparison_matrix',
    'plot_comparison_curves',
    'plot_reflection_coefficient',
    'plot_mode_wavenumbers',
    'plot_dispersion_curves',
    'plot_transfer_function',
    'quickplot',
]
