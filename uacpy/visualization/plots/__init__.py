"""Plotting surface for uacpy results and environments.

A package split by plot kind. Every public ``plot_*`` (and the
``plot_result`` type-dispatcher) is re-exported here, so
``from uacpy.visualization.plots import plot_field`` and
``uacpy.plot.plot_field`` both resolve. Shared primitives live in
:mod:`._common`.
"""

from typing import Optional

from uacpy.core.environment import Environment
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import (
    Field, Arrivals, Rays, Modes,
    Covariance, Replicas, ReflectionCoefficient, ResultStack,
)

from uacpy.visualization.plots.fields import (
    plot_field, plot_signal_excess, plot_detection_probability,
    compare, compare_models,
)
from uacpy.visualization.plots.animation import (
    animate_field, save_animation, plot_time_snapshots,
)
from uacpy.visualization.plots.rays_modes import (
    _plot_rays, _plot_arrivals, _plot_mode_functions, plot_mode_wavenumbers,
    plot_modes_heatmap, _plot_reflection_coefficient, _plot_covariance,
    _plot_replicas,
)
from uacpy.visualization.plots.environment import (
    plot_bottom_properties, plot_absorption,
)
from uacpy.visualization.plots.maps import (
    plot_bathymetry_map, plot_overview, plot_sea_ice_map,
)
from uacpy.visualization.plots.signal import (
    draw_slowness_line, draw_sound_cone, plot_fk, plot_radon, plot_taup,
    plot_psd, plot_ppsd, plot_sel,
    plot_spectrogram, plot_cwt, plot_wigner_ville, plot_cepstrum,
    plot_constant_q_spectrogram, plot_constant_q_psd, plot_constant_q_ppsd,
    plot_band_levels, plot_angular_spectrum, plot_ambiguity,
    plot_frf, plot_coherence, plot_impulse_response_info,
)
from uacpy.visualization.plots.comms import (
    plot_channel, plot_doppler_ambiguity, plot_convergence, plot_sync_metric,
    plot_subcarriers, plot_scatter, plot_constellation, plot_eye_diagram,
    plot_ber_curve,
)
from uacpy.visualization.plots.noise import (
    plot_wenz, plot_weighting, plot_source_level, plot_roc,
)


def plot_result(result, env: Optional[Environment] = None, **kwargs):
    """Type-dispatch to the right plotter. Used by :meth:`Result.plot`."""
    if isinstance(result, ResultStack):
        if issubclass(result.slab_type, Field):
            from uacpy.visualization.plots.fields import plot_field_stack
            return plot_field_stack(result, env=env, **kwargs)
        raise ConfigurationError(
            f"plot_result: this ResultStack holds {result.slab_type.__name__} "
            "slabs — pick one with stack[i] or stack.at(...) before plotting."
        )
    if isinstance(result, Field):
        return plot_field(result, env=env, **kwargs)
    if isinstance(result, Arrivals):
        return _plot_arrivals(result, **kwargs)
    if isinstance(result, Rays):
        return _plot_rays(result, env=env, **kwargs)
    if isinstance(result, Modes):
        return _plot_mode_functions(result, **kwargs)
    if isinstance(result, Covariance):
        return _plot_covariance(result, **kwargs)
    if isinstance(result, Replicas):
        return _plot_replicas(result, **kwargs)
    if isinstance(result, ReflectionCoefficient):
        return _plot_reflection_coefficient(result, **kwargs)
    raise ConfigurationError(
        f"plot_result: no plotter registered for {type(result).__name__}"
    )


__all__ = [
    'plot_result',
    'plot_field',
    'plot_signal_excess',
    'plot_detection_probability',
    'compare',
    'compare_models',
    'plot_bottom_properties',
    'plot_absorption',
    'plot_bathymetry_map',
    'plot_overview',
    'plot_sea_ice_map',
    'plot_mode_wavenumbers',
    'plot_modes_heatmap',
    'plot_fk',
    'plot_radon',
    'plot_taup',
    'draw_sound_cone',
    'draw_slowness_line',
    'plot_psd',
    'plot_ppsd',
    'plot_sel',
    'plot_spectrogram',
    'plot_constant_q_spectrogram',
    'plot_constant_q_psd',
    'plot_constant_q_ppsd',
    'plot_cwt',
    'plot_wigner_ville',
    'plot_cepstrum',
    'plot_band_levels',
    'plot_angular_spectrum',
    'plot_ambiguity',
    'plot_frf',
    'plot_coherence',
    'plot_impulse_response_info',
    'plot_channel',
    'plot_doppler_ambiguity',
    'plot_convergence',
    'plot_sync_metric',
    'plot_subcarriers',
    'plot_scatter',
    'plot_constellation',
    'plot_eye_diagram',
    'plot_ber_curve',
    'plot_wenz',
    'plot_weighting',
    'plot_source_level',
    'plot_roc',
    'animate_field',
    'save_animation',
    'plot_time_snapshots',
]
