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

``SL``, ``NL`` and ``RL`` must all share one band reference — see
:func:`noise_background`, which states the rule for the whole module.

The ``*_field`` variants (:func:`passive_signal_excess_field`,
:func:`active_signal_excess_field`, :func:`probability_of_detection_field`,
:func:`detection_range_by_depth`) evaluate the same budgets over a model TL
:class:`~uacpy.core.results.Field` so performance maps over
``(depth, range)`` come straight from a propagation run.
"""

from __future__ import annotations

import warnings

import numpy as np

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.results import Field


def echo_level(source_level, tl, target_strength):
    """Active echo level at the receiver: ``SL - 2*TL + TS`` (dB)."""
    return np.asarray(source_level, float) - 2.0 * np.asarray(tl, float) \
        + np.asarray(target_strength, float)


def noise_background(noise_level, directivity_index=None, *, array_gain=None):
    """Noise masking background: ``NL - DI`` (dB), or ``NL - AG``.

    **Band convention (this module's single statement of it).** ``NL`` has to
    share its reference with the ``SL`` (and ``RL``) it is differenced against:
    either *all* spectral levels (dB re 1 µPa²/Hz, SL dB re 1 µPa²·m²/Hz) or
    *all* band levels over the processing band (dB re 1 µPa², SL dB re
    1 µPa²·m²). The two differ by ``10*log10(w)`` — 20 dB at a 100 Hz band —
    and the ``DT`` from
    :func:`uacpy.sonar.detection.detection_threshold_energy` is a unitless
    power ratio valid only for a matched pair. The two :mod:`uacpy.noise`
    products sit on opposite sides: :attr:`uacpy.noise.WenzNoise.total` is
    spectral, :func:`uacpy.noise.radiated_noise_level` is a decidecade band
    level; convert one before combining them.

    ``array_gain`` replaces the directivity index when given — AG is the
    measured/estimated gain of the receiver against the *actual* noise
    field (AG = DI only for isotropic noise; anisotropic noise or signal
    coherence loss across the array makes AG < DI). ``directivity_index``
    defaults to ``None`` (treated as 0 dB); passing both an explicit
    ``directivity_index`` and ``array_gain`` raises — they are alternative
    parametrisations of the same term. ``None`` distinguishes "not supplied"
    from a legitimate per-angle DI array that happens to contain a 0.
    """
    if array_gain is not None:
        if directivity_index is not None:
            raise ConfigurationError(
                "noise_background: pass either directivity_index or "
                "array_gain, not both — AG replaces DI. Got "
                f"directivity_index={directivity_index!r} and "
                f"array_gain={array_gain!r}."
            )
        return np.asarray(noise_level, float) - np.asarray(array_gain, float)
    di = 0.0 if directivity_index is None else directivity_index
    return np.asarray(noise_level, float) - np.asarray(di, float)


def passive_signal_excess(
    source_level, tl, noise_level, directivity_index=None,
    detection_threshold=0.0, *, array_gain=None, processing_loss_db=0.0,
):
    """Passive signal excess ``SE = SL - TL - (NL - DI) - DT - L_sp`` (dB).

    ``source_level`` is the *target* radiated level, in the same band
    reference as ``noise_level`` (see :func:`noise_background`). ``SE >= 0``
    means the detector achieves its design ``(P_D, P_F)`` operating point.
    ``array_gain`` replaces ``directivity_index`` for non-isotropic
    noise (see :func:`noise_background`); ``processing_loss_db`` is the
    implementation/system loss ``L_sp >= 0`` (windowing, scalloping,
    beam-pattern, integration mismatch) subtracted from the budget.
    """
    _reject_field(tl, 'passive_signal_excess', 'tl',
                  'passive_signal_excess_field')
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
    directivity_index=None,
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
    ``L_sp >= 0`` subtracted from the budget. ``SL``, ``NL`` and ``RL``
    share one band reference — see :func:`noise_background`.
    """
    _reject_field(tl, 'active_signal_excess', 'tl',
                  'active_signal_excess_field')
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
    source_level, noise_level, directivity_index=None, detection_threshold=0.0,
    *, array_gain=None, processing_loss_db=0.0,
):
    """Figure of merit ``FOM = SL - (NL - DI) - DT - L_sp`` (dB).

    Equals the maximum allowable one-way TL (passive), or two-way TL when
    ``TS = 0`` (active). ``array_gain`` / ``processing_loss_db`` as in
    :func:`passive_signal_excess`; ``source_level`` and ``noise_level`` share
    one band reference — see :func:`noise_background`.
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
    if 'time' in tl_field.coords:
        raise ConfigurationError(
            "signal-excess field: a time-domain Field is not transmission "
            "loss; pass a TL / pressure Field (e.g. from "
            f"run_mode=COHERENT_TL). Got axes {list(tl_field.coords)}."
        )
    return tl_field.db


def _reject_field(value, caller: str, label: str, twin: str) -> None:
    """Raise a typed error naming ``twin`` when ``value`` is a ``Field``.

    The scalar sonar-equation functions take arrays and the ``*_field``
    functions take a :class:`~uacpy.core.results.Field`; handing a Field to the
    scalar one reaches ``float()`` and raises
    ``TypeError: float() argument must be … not 'Field'``, which names neither
    the argument nor the function one suffix away.
    """
    if isinstance(value, Field):
        raise ConfigurationError(
            f"{caller}: {label} is a Field; this function takes dB arrays. "
            f"Use {twin}(...) for a Field, or pass {label}.db().data / "
            f"np.asarray({label}.data) to stay here.")


def _require_scalar_db(value, caller: str, label: str) -> float:
    """Validate a sonar-budget term documented as a scalar and return it.

    ``reverberation_level`` is the one term of the budget that may be
    per-range; the rest are scalars, and an array reached ``float()`` at the
    budget dict *after* the signal excess had already been computed, raising
    ``TypeError: only 0-dimensional arrays can be converted to Python
    scalars`` — no function name, no argument name, and the work thrown away.
    """
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 0:
        raise ConfigurationError(
            f"{caller}: {label} must be a scalar dB level; got shape "
            f"{arr.shape}. reverberation_level is the one per-range term of "
            f"this budget — for a range-varying background, pass the profile "
            f"there, or evaluate the field once per {label} value.")
    return float(arr)


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
    """Wrap ``se`` in a Field carrying the TL field's identity and the budget.

    ``id_kwargs`` hands back a fresh ``metadata`` dict, so stamping the budget
    into it leaves the source field's own metadata untouched.

    The ``kind`` tag matters: signal excess is dB but it is not pressure and
    not a loss, so leaving it to derive would both mislabel it and make
    :meth:`Field.max` report the *worst* cell as the best."""
    kwargs = tl_field.id_kwargs()
    kwargs['metadata']['sonar_budget'] = budget
    kwargs['metadata']['kind'] = 'signal_excess'
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
    directivity_index=None,
    detection_threshold=0.0,
    array_gain=None,
    processing_loss_db=0.0,
) -> Field:
    """Passive signal excess over a model TL grid: ``SE = SL - TL - (NL - DI) - DT``.

    Grid counterpart of :func:`passive_signal_excess`: takes the
    :class:`~uacpy.core.results.Field` a propagation model returned
    (real dB TL, or complex pressure — converted via ``Field.db``) and
    evaluates the sonar equation at every ``(depth, range)`` sample.

    Parameters
    ----------
    tl_field : Field
        One-way TL field from any model run (e.g.
        ``run_mode=RunMode.COHERENT_TL``).
    source_level : float
        Target radiated level SL, at 1 m.
    noise_level : float
        Ambient noise level NL at the array, in the same band reference as
        ``source_level`` — see :func:`noise_background`.
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
    noise_level = _require_scalar_db(noise_level,
                                     'passive_signal_excess_field',
                                     'noise_level')
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
        'directivity_index': 0.0 if directivity_index is None else float(directivity_index),
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
    directivity_index=None,
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
        Projector source level SL, at 1 m.
    target_strength : float
        Target strength TS (dB).
    noise_level : float, optional
        Ambient noise level NL, in the same band reference as ``source_level``
        and ``reverberation_level`` (see :func:`noise_background`). Provide
        this and/or ``reverberation_level``.
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
    if noise_level is not None:
        noise_level = _require_scalar_db(noise_level,
                                         'active_signal_excess_field',
                                         'noise_level')
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
        'directivity_index': 0.0 if directivity_index is None else float(directivity_index),
        'detection_threshold': float(detection_threshold),
        'processing_loss_db': float(processing_loss_db),
    }
    if array_gain is not None:
        budget['array_gain'] = float(array_gain)
    if noise_level is not None:
        budget['noise_level'] = float(noise_level)
    if reverberation_level is not None:
        # RL is the one term that may be per-range, so it cannot go through
        # float() like its siblings. ``.tolist()`` keeps the budget dict plain
        # Python all the same — a float for a scalar RL, a list for a per-range
        # one — so the whole of ``metadata['sonar_budget']`` stays comparable
        # and serialisable rather than holding one bare ndarray.
        budget['reverberation_level'] = np.asarray(
            reverberation_level, dtype=float,
        ).tolist()
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

    This is **not** the field form of
    :func:`uacpy.sonar.detection.probability_of_detection` — that function
    evaluates a different model, the Gaussian (Neyman-Pearson) detector
    ``P_D = Q(Q^-1(P_F) - d')``, parameterised by a false-alarm rate and a
    deflection rather than by signal excess and a fluctuation spread.

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
            "passive/active_signal_excess_field. Got dtype "
            f"{se_field.data.dtype}."
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
    # A probability is dimensionless, not dB; inheriting the SE field's tags
    # would put a 0–1 array on a decibel axis.
    kwargs['metadata']['kind'] = 'probability_of_detection'
    kwargs['metadata']['unit'] = '1'
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
        Detection range (m) at each depth — the outermost zero-crossing,
        ``np.inf`` where SE >= 0 at every sampled range, the last sampled
        range where SE recovers positive at the far edge without crossing
        back down, and ``np.nan`` where SE < 0 everywhere.
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
    linear interpolation. Returns ``np.inf`` if SE >= 0 everywhere, the last
    sampled range where SE recovers positive at the far edge without crossing
    back down, and ``np.nan`` if SE < 0 everywhere. When the outermost positive
    sample and the next finite sample are separated by no-data (NaN) cells, the
    positive sample's range is returned as-is — a crossing inside a no-data hole
    has no modeled location to interpolate.

    The far-edge-recovery return is a **lower bound**, not a crossing: the
    outermost crossing lies beyond ``ranges_m``, so the number moves with the
    grid. It is finite, so the ``np.isfinite`` test the sonar guide recommends
    does not separate it from a modeled crossing; a :class:`UserWarning` names
    it instead.

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
        raise ConfigurationError(
            "detection_range: ranges and signal_excess shape mismatch; got "
            f"ranges_m shape {r.shape} and signal_excess_db shape {se.shape}")
    # NaN marks a cell the propagation model never filled (no ray reached it),
    # not a cell where the target is undetectable, so the crossing is sought
    # among the sampled ranges only — the same no-data handling as
    # :meth:`Field.max`.
    known = np.isfinite(se)
    if not known.any():
        return np.nan
    idx = np.where(known)[0]
    r, se = r[idx], se[idx]
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
        # The value is then the grid's own edge rather than a modeled crossing,
        # so it tracks ``receiver.ranges``: on one shelf budget a 20 km grid
        # returned 20000.0 m against 45947 m on a 120 km grid, a factor 2.3.
        # ``positive.all()`` above already took the "SE >= 0 everywhere" case,
        # so reaching here means SE went negative inside the grid and came back.
        warnings.warn(
            f"detection_range: signal excess goes negative inside the grid but "
            f"is back to >= 0 at the outermost sampled range with data "
            f"({float(r[-1]):.6g} m), so no crossing was found beyond it. The "
            f"returned {float(r[-1]):.6g} m is a LOWER BOUND on the detection "
            f"range, not the outermost zero-crossing — widen receiver.ranges "
            f"and re-run to locate the crossing.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
        return float(r[last_pos])
    # ``idx`` keeps each sample's position in the unmasked array: a step > 1
    # between consecutive known samples is a no-data hole, and a crossing
    # inside it has no modeled location to interpolate.
    if idx[last_pos + 1] - idx[last_pos] > 1:
        return float(r[last_pos])
    se0, se1 = se[last_pos], se[last_pos + 1]
    frac = se0 / (se0 - se1)
    return float(r[last_pos] + frac * (r[last_pos + 1] - r[last_pos]))
