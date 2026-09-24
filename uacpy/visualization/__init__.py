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
* :func:`plot_bottom_loss` — plane-wave bottom loss against grazing angle,
  one curve per seabed: preset names, property dicts, or a
  ``{label: material}`` mapping, all against one ``water_speed`` so the
  critical angles are comparable.
* :func:`plot_mode_wavenumbers`, :func:`plot_modes_heatmap` — the two
  alternate mode views (the default ``modes.plot()`` is the mode functions).
* :func:`plot_beam_power` — a scanned beam's power against look angle, from
  a :class:`~uacpy.acoustic_signal.BeamformedField`
  (``beams.plot()`` is the object-oriented form). The receive dual of
  :func:`plot_beam_pattern`, which is a launch fan and labels itself so.
* :func:`plot_beam_pattern` — source directivity from a ``.sbp`` table or
  an ``(N, 2)`` array, on polar axes oriented like the field
  (``source.plot_beam_pattern()`` is the object-oriented form).
* :func:`plot_absorption` — draws an
  :class:`~uacpy.core.absorption.AbsorptionCoefficient`: α(f) on log-log axes,
  or α(f, z) as a heatmap when the carrier has a depth axis. It computes
  nothing; build the carrier with
  :func:`~uacpy.core.absorption.absorption_thorp` or
  :func:`~uacpy.core.absorption.absorption_francois_garrison`, or just call
  ``.plot()`` on it.
* :func:`land_polygons` — the Natural Earth land rings the map plotters draw
  behind a chart, for a map of your own; :func:`download_coastline` caches
  them for offline use, the way ``uacpy.data``'s ``download_*_db`` fetchers
  cache their grids.

Importing this module does not mutate ``matplotlib.rcParams``.
"""

from uacpy.visualization import style
from uacpy.visualization.basemap import download_coastline, land_polygons

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
    plot_field_difference,
    plot_field_statistics,
    shared_colorbar,
    plot_bottom_properties,
    plot_bottom_loss,
    plot_absorption,
    plot_bathymetry_map,
    plot_overview,
    plot_sea_ice_map,
    plot_mode_wavenumbers,
    plot_mode_speeds,
    plot_greens_function,
    plot_wavenumber_sampling,
    plot_dispersion,
    plot_modes_heatmap,
    plot_beam_pattern,
    plot_beam_power,
    plot_mode_excitation,
    plot_fk,
    plot_radon,
    plot_taup,
    draw_sound_cone,
    draw_slowness_line,
    plot_waveform,
    plot_psd,
    plot_ppsd,
    plot_sel,
    plot_spectrogram,
    plot_constant_q_transform,
    plot_constant_q_spectrogram,
    plot_constant_q_psd,
    plot_constant_q_ppsd,
    plot_cwt,
    plot_wigner_ville,
    plot_cepstrum,
    plot_band_levels,
    plot_angular_spectrum,
    plot_ambiguity,
    plot_matched_field,
    plot_frf,
    plot_coherence,
    plot_lsfir_diagnostics,
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
    # the coastline backdrop the map plotters draw, and its cache
    'land_polygons',
    'download_coastline',
    'style',
]
