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
* Ray fans / arrival stems / mode functions / covariance / replicas /
  reflection coefficients and the environment / SSP cross-sections are plotted
  via ``result.plot()`` / ``env.plot()`` / ``ssp.plot()`` — every object that
  renders on its own carries its own ``.plot()`` (dispatched by
  :func:`plot_result`).
* :func:`plot_bottom_properties` — small-multiples seabed cross-sections,
  one panel per property (cp, cs, ρ, αp, αs); shows shear & friends that
  ``env.plot()`` (cp-only) does not.
* :func:`plot_mode_wavenumbers`, :func:`plot_modes_heatmap` — the two
  alternate mode views (the default ``modes.plot()`` is the mode functions).
* :func:`plot_beam_pattern` — source directivity from a ``.sbp`` table or
  an ``(N, 2)`` array, on polar axes oriented like the field
  (``source.plot_beam_pattern()`` is the object-oriented form).
* :func:`plot_absorption` — volume absorption α(f) from a raw dB/km array or a
  model string (``absorption.plot(frequencies)`` is the object-oriented form).

Importing this module does not mutate ``matplotlib.rcParams``.
"""

from uacpy.visualization import style

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
    plot_bottom_properties,
    plot_absorption,
    plot_bathymetry_map,
    plot_overview,
    plot_sea_ice_map,
    plot_mode_wavenumbers,
    plot_modes_heatmap,
    plot_beam_pattern,
    plot_fk,
    plot_radon,
    plot_taup,
    draw_sound_cone,
    draw_slowness_line,
    plot_psd,
    plot_ppsd,
    plot_sel,
    plot_spectrogram,
    plot_constant_q_spectrogram,
    plot_constant_q_psd,
    plot_constant_q_ppsd,
    plot_cwt,
    plot_wigner_ville,
    plot_cepstrum,
    plot_band_levels,
    plot_angular_spectrum,
    plot_ambiguity,
    plot_frf,
    plot_coherence,
    plot_impulse_response_info,
    plot_channel,
    plot_doppler_ambiguity,
    plot_convergence,
    plot_sync_metric,
    plot_subcarriers,
    plot_scatter,
    plot_constellation,
    plot_eye_diagram,
    plot_ber_curve,
    plot_wenz,
    plot_weighting,
    plot_source_level,
    plot_roc,
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
    'plot_bottom_properties',
    'plot_absorption',
    'plot_bathymetry_map',
    'plot_overview',
    'plot_sea_ice_map',
    'plot_mode_wavenumbers',
    'plot_modes_heatmap',
    'plot_beam_pattern',
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
    'style',
]
