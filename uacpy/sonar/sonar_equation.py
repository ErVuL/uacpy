"""Passive and active sonar equations (Urick 1983, Ch. 2; Etter Ch. 11).

All terms are in decibels. Sign and grouping follow Urick's table of sonar
parameters (reproduced in Etter, Table 11.1):

* Echo level (active):            ``EL = SL - 2*TL + TS``
* Noise background:               ``NL - DI`` (or ``NL - AG``)
* Passive signal excess:          ``SE = SL - TL - (NL - DI) - DT - L_sp``
* Active, noise-limited:          ``SE = SL - 2*TL + TS - (NL - DI) - DT - L_sp``
* Active, reverberation-limited:  ``SE = SL - 2*TL + TS - RL - DT - L_sp``
* Figure of merit (passive):      ``FOM = SL - (NL - DI) - DT - L_sp``
* Transition curve:               ``P_D = Phi(SE / sigma)`` (Urick Fig. 12.10)

``SE = 0`` means the detector achieves its design ``(P_D, P_F)`` operating
point. ``DT`` is the detection threshold (recognition differential) — see
:mod:`uacpy.sonar.detection`. ``L_sp`` is the optional implementation loss
(``processing_loss_db``); ``AG`` optionally replaces ``DI`` for
non-isotropic noise.

The ``*_field`` variants (:func:`passive_signal_excess_field`,
:func:`active_signal_excess_field`, :func:`probability_of_detection_field`,
:func:`detection_range_by_depth`) evaluate the same budgets over a model TL
:class:`~uacpy.core.results.Field` so performance maps over
``(depth, range)`` come straight from a propagation run.
"""

from __future__ import annotations

import numpy as np

from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Field


def echo_level(source_level, tl, target_strength):
    """Active echo level at the receiver: ``SL - 2*TL + TS`` (dB)."""
    return np.asarray(source_level, float) - 2.0 * np.asarray(tl, float) \
        + np.asarray(target_strength, float)


def noise_background(noise_level, directivity_index=0.0, *, array_gain=None):
    """Noise masking background: ``NL - DI`` (dB), or ``NL - AG``.

    ``array_gain`` replaces the directivity index when given — AG is the
    measured/estimated gain of the receiver against the *actual* noise
    field (AG = DI only for isotropic noise; anisotropic noise or signal
    coherence loss across the array makes AG < DI). Passing both a
    non-zero ``directivity_index`` and ``array_gain`` raises — they are
    alternative parametrisations of the same term.
    """
    if array_gain is not None:
        if np.any(np.asarray(directivity_index, float) != 0.0):
            raise ConfigurationError(
                "noise_background: pass either directivity_index or "
                "array_gain, not both — AG replaces DI."
            )
        return np.asarray(noise_level, float) - np.asarray(array_gain, float)
    return np.asarray(noise_level, float) - np.asarray(directivity_index, float)


def passive_signal_excess(
    source_level, tl, noise_level, directivity_index=0.0,
    detection_threshold=0.0, *, array_gain=None, processing_loss_db=0.0,
):
    """Passive signal excess ``SE = SL - TL - (NL - DI) - DT - L_sp`` (dB).

    ``source_level`` is the *target* radiated level. ``SE >= 0`` means
    the detector achieves its design ``(P_D, P_F)`` operating point.
    ``array_gain`` replaces ``directivity_index`` for non-isotropic
    noise (see :func:`noise_background`); ``processing_loss_db`` is the
    implementation/system loss ``L_sp >= 0`` (windowing, scalloping,
    beam-pattern, integration mismatch) subtracted from the budget.
    """
    return (
        np.asarray(source_level, float)
        - np.asarray(tl, float)
        - noise_background(noise_level, directivity_index,
                           array_gain=array_gain)
        - np.asarray(detection_threshold, float)
        - np.asarray(processing_loss_db, float)
    )


def active_signal_excess(
    source_level,
    tl,
    target_strength,
    *,
    noise_level=None,
    directivity_index=0.0,
    reverberation_level=None,
    detection_threshold=0.0,
    array_gain=None,
    processing_loss_db=0.0,
):
    """Active signal excess (dB), noise- or reverberation-limited.

    Provide ``noise_level`` (noise-limited), ``reverberation_level``
    (reverb-limited), or both — in which case the louder background
    (incoherent sum) is used per range.

        noise-limited:  ``SE = SL - 2*TL + TS - (NL - DI) - DT``
        reverb-limited: ``SE = SL - 2*TL + TS - RL - DT``

    ``directivity_index`` / ``array_gain`` is **not** applied against
    ``RL`` (Urick Ch. 8): a beamformed receiver offers no gain against
    in-beam reverberation, whose level is already beam-limited through
    the scattering-cell size (``horizontal_beamwidth_rad`` in
    :func:`uacpy.sonar.reverberation.boundary_reverberation`).
    ``array_gain`` replaces ``directivity_index`` for non-isotropic
    noise; ``processing_loss_db`` is the implementation loss
    ``L_sp >= 0`` subtracted from the budget.
    """
    if noise_level is None and reverberation_level is None:
        raise ConfigurationError(
            "active_signal_excess: provide noise_level and/or reverberation_level"
        )
    el = echo_level(source_level, tl, target_strength)
    backgrounds = []
    if noise_level is not None:
        backgrounds.append(noise_background(noise_level, directivity_index,
                                            array_gain=array_gain))
    if reverberation_level is not None:
        backgrounds.append(np.asarray(reverberation_level, float))
    bcast = np.broadcast_arrays(*backgrounds)
    background = 10.0 * np.log10(
        np.sum([10.0 ** (b / 10.0) for b in bcast], axis=0)
    )
    return (el - background - np.asarray(detection_threshold, float)
            - np.asarray(processing_loss_db, float))


def figure_of_merit(
    source_level, noise_level, directivity_index=0.0, detection_threshold=0.0,
    *, array_gain=None, processing_loss_db=0.0,
):
    """Figure of merit ``FOM = SL - (NL - DI) - DT - L_sp`` (dB).

    Equals the maximum allowable one-way TL (passive), or two-way TL when
    ``TS = 0`` (active). ``array_gain`` / ``processing_loss_db`` as in
    :func:`passive_signal_excess`.
    """
    return (
        np.asarray(source_level, float)
        - noise_background(noise_level, directivity_index,
                           array_gain=array_gain)
        - np.asarray(detection_threshold, float)
        - np.asarray(processing_loss_db, float)
    )


def _tl_array_from_field(tl_field) -> np.ndarray:
    """One-way TL (dB) at the field's grid, from a real-dB or complex Field."""
    if not isinstance(tl_field, Field):
        raise ConfigurationError(
            f"signal-excess field: expected a Field, got "
            f"{type(tl_field).__name__}"
        )
    if tl_field.kind == 'time_series':
        raise ConfigurationError(
            "signal-excess field: a time-domain Field is not transmission "
            "loss; pass a TL / pressure Field (e.g. from "
            "run_mode=COHERENT_TL)."
        )
    return tl_field.tl


def _per_range_broadcast(values, tl_field, label: str) -> np.ndarray:
    """Broadcast a scalar or per-range 1-D array against the field grid."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return arr
    if arr.ndim != 1:
        raise ConfigurationError(
            f"signal-excess field: {label} must be a scalar or a 1-D "
            f"per-range array; got shape {arr.shape}"
        )
    if 'range' not in tl_field.coords:
        raise ConfigurationError(
            f"signal-excess field: per-range {label} requires the Field to "
            f"carry a 'range' axis; got {list(tl_field.coords)}"
        )
    n_r = tl_field.coords['range'].size
    if arr.size != n_r:
        raise ConfigurationError(
            f"signal-excess field: {label} length ({arr.size}) must match "
            f"the field's range axis ({n_r})"
        )
    shape = [1] * tl_field.data.ndim
    shape[list(tl_field.coords).index('range')] = n_r
    return arr.reshape(shape)


def _spawn_se_field(tl_field, se: np.ndarray, budget: dict) -> Field:
    kwargs = tl_field.id_kwargs()
    kwargs['metadata']['sonar_budget'] = budget
    return Field(
        data=np.asarray(se, dtype=float),
        coords={k: v.copy() for k, v in tl_field.coords.items()},
        pinned=dict(tl_field.pinned),
        **kwargs,
    )


def passive_signal_excess_field(
    tl_field,
    *,
    source_level,
    noise_level,
    directivity_index=0.0,
    detection_threshold=0.0,
    array_gain=None,
    processing_loss_db=0.0,
) -> Field:
    """Passive signal excess over a model TL grid: ``SE = SL - TL - (NL - DI) - DT``.

    Grid counterpart of :func:`passive_signal_excess`: takes the
    :class:`~uacpy.core.results.Field` a propagation model returned
    (real dB TL, or complex pressure — converted via ``Field.tl``) and
    evaluates the sonar equation at every ``(depth, range)`` sample.

    Parameters
    ----------
    tl_field : Field
        One-way TL field from any model run (e.g.
        ``run_mode=RunMode.COHERENT_TL``).
    source_level : float
        Target radiated level SL (dB re 1 µPa @ 1 m).
    noise_level : float
        Ambient noise level NL (dB re 1 µPa²/Hz at the array).
    directivity_index : float, optional
        Receiving directivity index DI (dB). Default 0.
    detection_threshold : float, optional
        Detection threshold DT (dB) — see
        :func:`uacpy.sonar.detection.detection_threshold_energy`. Default 0.
    array_gain : float, optional
        Replaces ``directivity_index`` for non-isotropic noise (see
        :func:`noise_background`).
    processing_loss_db : float, optional
        Implementation/system loss ``L_sp >= 0`` (dB). Default 0.

    Returns
    -------
    Field
        Signal excess (dB) on the same coords/pinned grid; ``SE >= 0``
        marks the detectable region. The budget terms are recorded in
        ``result.metadata['sonar_budget']``. Plot with
        :func:`uacpy.visualization.plots.plot_signal_excess`.
    """
    tl = _tl_array_from_field(tl_field)
    se = passive_signal_excess(
        source_level, tl, noise_level,
        directivity_index=directivity_index,
        detection_threshold=detection_threshold,
        array_gain=array_gain,
        processing_loss_db=processing_loss_db,
    )
    budget = {
        'mode': 'passive',
        'source_level': float(source_level),
        'noise_level': float(noise_level),
        'directivity_index': float(directivity_index),
        'detection_threshold': float(detection_threshold),
        'processing_loss_db': float(processing_loss_db),
    }
    if array_gain is not None:
        budget['array_gain'] = float(array_gain)
    return _spawn_se_field(tl_field, se, budget)


def active_signal_excess_field(
    tl_field,
    *,
    source_level,
    target_strength,
    noise_level=None,
    reverberation_level=None,
    directivity_index=0.0,
    detection_threshold=0.0,
    array_gain=None,
    processing_loss_db=0.0,
) -> Field:
    """Active (monostatic) signal excess over a model TL grid.

    Grid counterpart of :func:`active_signal_excess`, with the same
    noise- / reverberation-limited background handling:

        noise-limited:  ``SE = SL - 2*TL + TS - (NL - DI) - DT``
        reverb-limited: ``SE = SL - 2*TL + TS - RL - DT``

    ``tl_field`` carries the one-way TL; the two-way path assumes a
    monostatic geometry (same TL out and back).

    Parameters
    ----------
    tl_field : Field
        One-way TL field from any model run.
    source_level : float
        Projector source level SL (dB re 1 µPa @ 1 m).
    target_strength : float
        Target strength TS (dB).
    noise_level : float, optional
        Ambient noise level NL (dB). Provide this and/or
        ``reverberation_level``.
    reverberation_level : float or array, optional
        Reverberation level RL (dB) — a scalar, or a 1-D per-range array
        matching the field's ``'range'`` axis (e.g. from
        :func:`uacpy.sonar.reverberation.boundary_reverberation` on
        ``tl_field.coords['range']``).
    directivity_index : float, optional
        Receiving directivity index DI (dB). Default 0.
    detection_threshold : float, optional
        Detection threshold DT (dB). Default 0.
    array_gain : float, optional
        Replaces ``directivity_index`` against the noise background
        (never against ``RL`` — see :func:`active_signal_excess`).
    processing_loss_db : float, optional
        Implementation/system loss ``L_sp >= 0`` (dB). Default 0.

    Returns
    -------
    Field
        Signal excess (dB) on the same coords/pinned grid, with the
        budget terms in ``result.metadata['sonar_budget']``.
    """
    tl = _tl_array_from_field(tl_field)
    rl = (
        _per_range_broadcast(reverberation_level, tl_field,
                             'reverberation_level')
        if reverberation_level is not None else None
    )
    se = active_signal_excess(
        source_level, tl, target_strength,
        noise_level=noise_level,
        directivity_index=directivity_index,
        reverberation_level=rl,
        detection_threshold=detection_threshold,
        array_gain=array_gain,
        processing_loss_db=processing_loss_db,
    )
    budget = {
        'mode': 'active',
        'source_level': float(source_level),
        'target_strength': float(target_strength),
        'directivity_index': float(directivity_index),
        'detection_threshold': float(detection_threshold),
        'processing_loss_db': float(processing_loss_db),
    }
    if array_gain is not None:
        budget['array_gain'] = float(array_gain)
    if noise_level is not None:
        budget['noise_level'] = float(noise_level)
    if reverberation_level is not None:
        budget['reverberation_level'] = np.asarray(
            reverberation_level, dtype=float,
        )
    return _spawn_se_field(tl_field, se, budget)


def probability_of_detection_field(se_field, *, sigma_db) -> Field:
    """Detection-probability field from a signal-excess field.

    Urick's transition curve (Fig. 12.10; Abraham §2.3.5): the detector
    decision statistic is taken log-normal under signal-plus-noise, so
    in dB it is Gaussian with mean ``DT + SE`` and standard deviation
    ``sigma_db``, giving

        ``P_D = Phi(SE / sigma_db)``

    with ``Phi`` the standard normal CDF. ``P_D = 0.5`` exactly on the
    ``SE = 0`` contour — consistent with a detection threshold defined
    at the ``P_D = 0.5`` operating point.

    Parameters
    ----------
    se_field : Field
        Signal excess (dB) from :func:`passive_signal_excess_field` /
        :func:`active_signal_excess_field`. Any coords shape — the
        transform is elementwise.
    sigma_db : float
        Standard deviation of the signal-excess fluctuation (dB).
        Dyer's saturated-multipath result gives ``sigma_db ≈ 5.6``;
        measured one-way totals typically run 5–9 dB. No default —
        it is a physical claim about the channel, not a processing knob.

    Returns
    -------
    Field
        ``P_D`` in [0, 1] on the same coords/pinned grid;
        ``metadata['sigma_db']`` records the fluctuation model. Plot
        with :func:`uacpy.visualization.plots.plot_detection_probability`.

    Notes
    -----
    The log-normal approximation is most accurate near ``SE = 0`` and
    optimistic in the tails; as ``sigma_db → 0`` it degenerates to a
    step at ``SE = 0`` rather than the deterministic-signal ROC
    (Abraham §2.3.5.6). For fluctuation statistics beyond Gaussian-in-dB
    (e.g. the gamma-fluctuating-intensity model), compute ``P_D`` from
    the detector statistics directly via :mod:`uacpy.sonar.detection`.
    """
    from scipy.stats import norm
    if not isinstance(se_field, Field):
        raise ConfigurationError(
            f"probability_of_detection_field: expected a Field, got "
            f"{type(se_field).__name__}"
        )
    if se_field.is_complex:
        raise ConfigurationError(
            "probability_of_detection_field: field must carry real "
            "signal excess in dB — build it with "
            "passive/active_signal_excess_field."
        )
    sigma = float(sigma_db)
    if sigma <= 0.0:
        raise ConfigurationError(
            f"probability_of_detection_field: sigma_db must be positive; "
            f"got {sigma_db}"
        )
    pd = norm.cdf(np.asarray(se_field.data, dtype=float) / sigma)
    kwargs = se_field.id_kwargs()
    kwargs['metadata']['sigma_db'] = sigma
    return Field(
        data=pd,
        coords={k: v.copy() for k, v in se_field.coords.items()},
        pinned=dict(se_field.pinned),
        **kwargs,
    )


def detection_range_by_depth(se_field):
    """Per-depth detection range from a 2-D signal-excess field.

    Applies :func:`detection_range` to each depth row of a canonical
    ``['depth', 'range']`` signal-excess :class:`Field`.

    Returns
    -------
    depths : ndarray, shape ``(n_depths,)``
        The field's depth axis (m).
    ranges : ndarray, shape ``(n_depths,)``
        Detection range (m) at each depth — ``np.inf`` where SE >= 0
        out to the last sample, ``np.nan`` where SE < 0 everywhere.
    """
    if not isinstance(se_field, Field):
        raise ConfigurationError(
            f"detection_range_by_depth: expected a Field, got "
            f"{type(se_field).__name__}"
        )
    if list(se_field.coords) != ['depth', 'range']:
        raise ConfigurationError(
            "detection_range_by_depth: requires canonical "
            f"['depth', 'range'] coords; got {list(se_field.coords)}"
        )
    depths = se_field.coords['depth'].copy()
    r = se_field.coords['range']
    se = np.asarray(se_field.data, dtype=float)
    out = np.array([detection_range(r, se[i, :]) for i in range(depths.size)])
    return depths, out


def detection_range(ranges_m, signal_excess_db):
    """Largest range (m) at which the signal excess is still non-negative.

    Finds the outermost zero-crossing of ``signal_excess_db`` versus range by
    linear interpolation. Returns ``np.inf`` if SE >= 0 everywhere, or
    ``np.nan`` if SE < 0 everywhere.

    Parameters
    ----------
    ranges_m : array
        Monotonically increasing ranges (m).
    signal_excess_db : array
        Signal excess (dB) at each range.
    """
    r = np.asarray(ranges_m, dtype=float)
    se = np.asarray(signal_excess_db, dtype=float)
    if r.shape != se.shape:
        raise ConfigurationError("detection_range: ranges and signal_excess shape mismatch")
    positive = se >= 0.0
    if positive.all():
        return np.inf
    if not positive.any():
        return np.nan
    # Largest range with SE >= 0 is the outermost positive sample — this
    # captures a far-edge recovery (e.g. a convergence zone giving +,-,+).
    last_pos = int(np.where(positive)[0][-1])
    if last_pos == r.size - 1:
        # SE stays/recovers positive at the far edge; detectable out to the
        # last sampled range, with no crossing-down beyond it to interpolate.
        return float(r[last_pos])
    se0, se1 = se[last_pos], se[last_pos + 1]
    frac = se0 / (se0 - se1)
    return float(r[last_pos] + frac * (r[last_pos + 1] - r[last_pos]))
