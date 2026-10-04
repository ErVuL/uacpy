"""Plotting surface for uacpy results and environments.

A package split by plot kind. Every public ``plot_*`` (and the
``plot_result`` type-dispatcher) is re-exported here, so
``from uacpy.visualization.plots import plot_field`` and
``uacpy.plot.plot_field`` both resolve. Shared primitives live in
:mod:`._common`.
"""

from typing import Optional

from uacpy.core.absorption import AbsorptionCoefficient
from uacpy.core.altimetry import Altimetry
from uacpy.core.bathymetry import Bathymetry
from uacpy.core.environment import Environment
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.ssp import SoundSpeedProfile
from uacpy.core.results import (
    Result, Field, Arrivals, Rays, Modes,
    Covariance, Replicas, ReflectionCoefficient, GreensFunction,
    ResultStack,
)

from uacpy.visualization.plots.fields import (
    plot_field, plot_signal_excess, plot_detection_probability,
    compare, compare_models, plot_field_difference,
    plot_field_statistics, shared_colorbar, _plot_field_stack,
    plot_transfer_function, plot_impulse_response,
)
from uacpy.visualization.plots.animation import (
    animate_field, save_animation, plot_time_snapshots,
)
from uacpy.visualization.plots.rays_modes import (
    _plot_rays, _plot_arrivals, _plot_mode_functions, plot_mode_wavenumbers,
    plot_mode_speeds, plot_dispersion, plot_greens_function,
    plot_wavenumber_sampling,
    plot_modes_heatmap, _plot_reflection_coefficient, _plot_covariance,
    plot_beam_pattern,
    plot_beam_power,
    plot_mode_excitation,
    _plot_replicas,
)
from uacpy.visualization.plots.environment import (
    plot_environment, plot_ssp, plot_range_profile,
    plot_bottom_properties, plot_bottom_loss, plot_absorption,
)
from uacpy.visualization.plots.maps import (
    plot_bathymetry_map, plot_overview, plot_sea_ice_map,
)
from uacpy.visualization.plots.signal import (
    draw_slowness_line, draw_sound_cone, plot_fk, plot_radon, plot_taup,
    plot_waveform,
    plot_psd, plot_ppsd, plot_sel,
    plot_spectrogram, plot_cwt, plot_wigner_ville, plot_cepstrum,
    plot_constant_q_transform, plot_constant_q_spectrogram,
    plot_constant_q_psd, plot_constant_q_ppsd,
    plot_band_levels, plot_angular_spectrum, plot_ambiguity,
    plot_matched_field,
    plot_frf, plot_coherence, plot_lsfir_diagnostics,
)
from uacpy.visualization.plots.comms import (
    plot_channel, plot_doppler_ambiguity, plot_convergence, plot_sync_metric,
    plot_subcarriers, plot_scatter, plot_constellation, plot_eye_diagram,
    plot_ber_curve,
)
from uacpy.visualization.plots.noise import (
    plot_wenz, plot_weighting, plot_source_level, plot_roc,
)


#: Which plotter draws each :class:`~uacpy.core.results.Result` and each
#: carrier that draws itself, and whether that view can draw an environment.
#: One row per type, in isinstance order (no subclass pairs today, so
#: declaration order is free). :func:`plot_result` draws the result rows and
#: :func:`plot_carrier` the carrier rows; every ``.plot()`` reaches one of
#: the two.
#:
#: A table rather than an if/elif ladder, so each result type states both
#: answers on one row: which plotter draws it and whether that view takes
#: ``env=``. Kept in two places — the branches and a separate list of the
#: types that refuse ``env=`` — a type added to one and forgotten in the
#: other would accept ``env=`` and silently ignore it, the defect the
#: refusal exists to prevent.
#:
#: ``ResultStack`` is not here: it dispatches on the type of the slabs it
#: holds, not on its own, so it is handled before the lookup.
_PLOTTERS = (
    # result type             plotter                       draws an env
    (Field,                   plot_field,                   True),
    (Rays,                    _plot_rays,                   True),
    (Arrivals,                _plot_arrivals,               False),
    (Modes,                   _plot_mode_functions,         False),
    (Covariance,              _plot_covariance,             False),
    (Replicas,                _plot_replicas,               False),
    (ReflectionCoefficient,   _plot_reflection_coefficient, False),
    (GreensFunction,          plot_greens_function,         False),
    # carriers: each is drawn on its own, never over an environment
    (Environment,             plot_environment,             False),
    (SoundSpeedProfile,       plot_ssp,                     False),
    (Bathymetry,              plot_range_profile,           False),
    (Altimetry,               plot_range_profile,           False),
    (AbsorptionCoefficient,   plot_absorption,              False),
)


#: The sonar maps a ``(depth, range)`` Field of these kinds is drawn with: the
#: signal-excess map with its SE = 0 dB detection boundary, the
#: detection-probability map with its ``P_D`` contours labelled by value and
#: its title. ``plot_field`` draws neither, so ``se.plot()`` and
#: ``plot_signal_excess(se)`` would otherwise be two different pictures of
#: one field.
_SONAR_MAP_PLOTTERS = {
    'signal_excess': plot_signal_excess,
    'probability_of_detection': plot_detection_probability,
}
#: A ``plot_field`` keyword the sonar map does not take (``value=``,
#: ``stacked=``, ``vmin=``) is refused by that plotter's own signature; call
#: ``plot_field`` directly for those views.


def plot_result(result, env: Optional[Environment] = None, **kwargs):
    """Type-dispatch to the right plotter. Used by :meth:`Result.plot`.

    A result carries no carriers, so ``env`` is only ever what the caller
    passes: supply it to draw the seabed and span the full water column.
    A carrier is drawn by :func:`plot_carrier`.

    Parameters
    ----------
    result : Result or ResultStack
        The result to draw.
    env : Environment, optional
        Draws the seabed and spans the full water column.
    **kwargs
        Keywords of the plotter the result's type selects.
    """
    if not isinstance(result, (Result, ResultStack)):
        raise ConfigurationError(
            f"plot_result: {type(result).__name__} is not a Result; draw "
            f"a carrier with plot_carrier, or call its own .plot()."
        )
    if isinstance(result, ResultStack):
        if issubclass(result.slab_type, Field):
            return _plot_field_stack(result, env=env, **kwargs)
        raise ConfigurationError(
            f"plot_result: this ResultStack holds {result.slab_type.__name__} "
            "slabs — pick one with stack[i] or stack.at(...) before plotting."
        )
    if (isinstance(result, Field)
            and list(result.coords) == ['depth', 'range']
            and result.kind in _SONAR_MAP_PLOTTERS):
        return _SONAR_MAP_PLOTTERS[result.kind](result, env=env, **kwargs)
    for result_type, plotter, draws_env in _PLOTTERS:
        if not isinstance(result, result_type):
            continue
        if draws_env:
            return plotter(result, env=env, **kwargs)
        # Refused rather than dropped: this view has no spatial cross-section
        # to overlay an environment on, and accepting env= silently would
        # look like it had an effect. Reached only once the type is known to
        # be one we render, so an unregistered type still reports that it has
        # no plotter rather than blaming env=.
        if env is not None:
            raise ConfigurationError(
                f"{type(result).__name__}.plot: env= has no effect on this "
                f"view — only Field and Rays plots draw the environment. "
                f"Drop env=."
            )
        return plotter(result, **kwargs)
    raise ConfigurationError(
        f"plot_result: no plotter registered for {type(result).__name__}."
    )


def plot_carrier(carrier, **kwargs):
    """Draw a carrier with the plotter its type is registered with: an
    :class:`~uacpy.core.environment.Environment` with
    :func:`plot_environment`, a
    :class:`~uacpy.core.ssp.SoundSpeedProfile` with :func:`plot_ssp`, a
    :class:`~uacpy.core.bathymetry.Bathymetry` or
    :class:`~uacpy.core.altimetry.Altimetry` with :func:`plot_range_profile`,
    an :class:`~uacpy.core.absorption.AbsorptionCoefficient` with
    :func:`plot_absorption`. Used by each carrier's ``.plot()``; ``kwargs``
    reach the plotter. A result is drawn by :func:`plot_result`.

    Parameters
    ----------
    carrier : Environment, SoundSpeedProfile, Bathymetry, Altimetry or AbsorptionCoefficient
        The carrier to draw.
    **kwargs
        Keywords of the plotter its type is registered with.
    """
    if isinstance(carrier, (Result, ResultStack)):
        raise ConfigurationError(
            f"plot_carrier: {type(carrier).__name__} is a Result; draw it "
            f"with plot_result, or call its own .plot()."
        )
    for carrier_type, plotter, _draws_env in _PLOTTERS:
        if isinstance(carrier, carrier_type):
            return plotter(carrier, **kwargs)
    raise ConfigurationError(
        f"plot_carrier: no plotter registered for {type(carrier).__name__}."
    )


__all__ = [
    'plot_result',
    'plot_carrier',
    'plot_environment',
    'plot_ssp',
    'plot_range_profile',
    'plot_transfer_function',
    'plot_impulse_response',
    'plot_field',
    'plot_signal_excess',
    'plot_detection_probability',
    'compare',
    'compare_models',
    'plot_field_difference',
    'plot_field_statistics',
    'shared_colorbar',
    'plot_bottom_properties',
    'plot_bottom_loss',
    'plot_absorption',
    'plot_bathymetry_map',
    'plot_overview',
    'plot_sea_ice_map',
    'plot_mode_wavenumbers',
    'plot_mode_speeds',
    'plot_greens_function',
    'plot_wavenumber_sampling',
    'plot_dispersion',
    'plot_modes_heatmap',
    'plot_beam_pattern',
    'plot_beam_power',
    'plot_mode_excitation',
    'plot_fk',
    'plot_radon',
    'plot_taup',
    'draw_sound_cone',
    'draw_slowness_line',
    'plot_waveform',
    'plot_psd',
    'plot_ppsd',
    'plot_sel',
    'plot_spectrogram',
    'plot_constant_q_transform',
    'plot_constant_q_spectrogram',
    'plot_constant_q_psd',
    'plot_constant_q_ppsd',
    'plot_cwt',
    'plot_wigner_ville',
    'plot_cepstrum',
    'plot_band_levels',
    'plot_angular_spectrum',
    'plot_ambiguity',
    'plot_matched_field',
    'plot_frf',
    'plot_coherence',
    'plot_lsfir_diagnostics',
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
    # submodules
    'animation',
    'comms',
    'environment',
    'fields',
    'maps',
    'noise',
    'rays_modes',
    'signal',
]
