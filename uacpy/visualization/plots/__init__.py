"""Plotting surface for uacpy results and environments.

A package split by plot kind. Every public ``plot_*`` (and the
``plot_result`` type-dispatcher) is re-exported here, so
``from uacpy.visualization.plots import plot_field`` and
``uacpy.plot.plot_field`` both resolve. Shared primitives live in
:mod:`._common`.
"""

import inspect
from typing import Optional

from uacpy.core._export import CarrierExport
from uacpy.core.environment import Environment
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Result, ResultStack

from uacpy.visualization.plots.fields import (
    plot_field, plot_signal_excess, plot_detection_probability,
    compare, compare_models, plot_field_difference,
    plot_field_statistics, shared_colorbar,
    plot_transfer_function, plot_impulse_response, plot_track_exposure,
)
from uacpy.visualization.plots.animation import (
    animate_field, save_animation, plot_time_snapshots,
)
from uacpy.visualization.plots.rays_modes import (
    plot_mode_wavenumbers,
    plot_mode_speeds, plot_dispersion, plot_greens_function,
    plot_wavenumber_sampling,
    plot_modes_heatmap,
    plot_beam_pattern,
    plot_beam_power,
    plot_mode_excitation,
)
# The private views a result names in ``_plotter`` / ``_plotter_name()``,
# looked up here by name (:func:`_draw_result`).
from uacpy.visualization.plots.fields import (  # noqa: F401
    _plot_field_as, _plot_source_depth_panels,
)
from uacpy.visualization.plots.rays_modes import (  # noqa: F401
    _plot_rays, _plot_arrivals, _plot_mode_functions,
    _plot_reflection_coefficient, _plot_covariance, _plot_replicas,
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


def plot_result(result, env: Optional[Environment] = None, **kwargs):
    """Type-dispatch to the right plotter. Used by :meth:`Result.plot`.

    A result carries no carriers, so ``env`` is only ever what the caller
    passes: supply it to draw the seabed, the depth axis extended down to
    it. A carrier is drawn by :func:`plot_carrier`.

    Parameters
    ----------
    result : Result
        The result to draw. A Field with a ``source_depth`` axis (a
        multi-depth run) is drawn as one panel per source depth when it also
        has a ``depth`` axis (a ``depth``-by-``range`` panel each) or more
        than two axes of several values (a ``range``-by-``frequency`` panel
        each for one receiver depth).
    env : Environment, optional
        Draws the seabed, extending the depth axis down to it.
    **kwargs
        Keywords of the plotter the result's type selects.
    """
    return _draw_result('plot_result', result, env, **kwargs)


def _draw_result(who: str, result, /, env: Optional[Environment] = None,
                 **kwargs):
    """:func:`plot_result`'s drawing, each refusal naming ``who``, the call
    the user made (``plot_result``, ``Field.plot``, ``ResultStack.plot``).
    ``who`` and ``result`` are positional-only, so every keyword of the call
    reaches the plotter.

    The result names its view (``result._plotter_name()``, a plotter in this
    module) and whether that view draws an environment
    (``result._draws_env``): such a view is handed ``env``, and ``who`` too
    when it takes one (a view whose refusals name the call); any other view
    refuses ``env=`` rather than drop it, since accepting it silently would
    look like it had an effect. A ``ResultStack`` is refused before the
    lookup: its slabs are drawn one at a time."""
    if not isinstance(result, Result):
        raise ConfigurationError(
            f"{who}: {type(result).__name__} is not a Result; draw "
            f"a carrier with plot_carrier, or call its own .plot()."
        )
    if isinstance(result, ResultStack):
        raise ConfigurationError(
            f"{who}: this ResultStack holds {result.slab_type.__name__} "
            "slabs — pick one with stack[i] or stack.at(...) before plotting."
        )
    name = result._plotter_name()
    if not name:
        raise ConfigurationError(
            f"{who}: no plotter registered for {type(result).__name__}."
        )
    draw = globals()[name]
    if not result._draws_env:
        if env is not None:
            raise ConfigurationError(result._env_refusal(who))
        return draw(result, **kwargs)
    if 'who' in inspect.signature(draw).parameters:
        return draw(result, env=env, who=who, **kwargs)
    return draw(result, env=env, **kwargs)


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
    if isinstance(carrier, Result):
        raise ConfigurationError(
            f"plot_carrier: {type(carrier).__name__} is a Result; draw it "
            f"with plot_result, or call its own .plot()."
        )
    # A carrier (a CarrierExport) names its plotter in ``_plotter``; a signal
    # result or a noise spectrum names one too but draws itself, and is
    # refused here like any other object.
    name = (getattr(carrier, '_plotter', '')
            if isinstance(carrier, CarrierExport) else '')
    if not name:
        raise ConfigurationError(
            f"plot_carrier: no plotter registered for {type(carrier).__name__}."
        )
    return globals()[name](carrier, **kwargs)


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
    'plot_track_exposure',
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
]
