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
(``processing_loss_dB``); ``AG`` optionally replaces ``DI`` for
non-isotropic noise.

``SL``, ``NL`` and ``RL`` must all share one band reference — see
:func:`noise_background`, which states the rule for the whole module.

:class:`SonarBudget` holds the terms of one budget (everything but the TL)
and evaluates them through the array functions above; the ``*_field``
variants (:func:`passive_signal_excess_field`,
:func:`active_signal_excess_field`) build one and evaluate it over a model TL
:class:`~uacpy.core.results.Field`, so performance maps over
``(depth, range)`` come straight from a propagation run.
:func:`transition_probability_field` and :func:`detection_ranges_by_depth`
are the Field forms of :func:`transition_probability` and
:func:`detection_ranges`.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Optional, Union

import numpy as np

from uacpy.core.acoustics.levels import no_energy_mask
from uacpy.core.constants import NO_ENERGY_DB
from uacpy.core._repr import FieldsRepr
from uacpy.core.exceptions import (
    ConfigurationError, NumericsWarning, ValidityWarning,
)
from uacpy.core._validate import require_positive_finite_scalar
from uacpy.sonar.reverberation import total_reverberation
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.results import Field
from uacpy.core.results.quantities import is_loss


def echo_level(source_level_dB, tl_dB, target_strength_dB):
    """Active echo level at the receiver: ``SL - 2*TL + TS`` (dB).

    Parameters
    ----------
    source_level_dB : float or array_like
        Source level (dB).
    tl_dB : float or array_like
        One-way transmission loss (dB).
    target_strength_dB : float or array_like
        Target strength (dB).
    """
    return np.asarray(source_level_dB, float) - 2.0 * np.asarray(tl_dB, float) \
        + np.asarray(target_strength_dB, float)


def noise_background(noise_level_dB, directivity_index_dB=None, *, array_gain_dB=None):
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

    ``array_gain_dB`` replaces the directivity index when given — AG is the
    measured/estimated gain of the receiver against the *actual* noise
    field (AG = DI only for isotropic noise; anisotropic noise or signal
    coherence loss across the array makes AG < DI). ``directivity_index_dB``
    defaults to ``None`` (treated as 0 dB); passing both an explicit
    ``directivity_index_dB`` and ``array_gain_dB`` raises — they are alternative
    parametrisations of the same term. ``None`` distinguishes "not supplied"
    from a legitimate per-angle DI array that happens to contain a 0.

    Parameters
    ----------
    noise_level_dB : float or array_like
        Noise level (dB), in the band reference of the source level
        (see :func:`noise_background`).
    directivity_index_dB : float or array_like, optional
        Directivity index (dB); ``None`` is 0 dB.
    array_gain_dB : float or array_like, optional
        Array gain (dB) in place of the directivity index; passing both
        is refused.
    """
    if array_gain_dB is not None:
        if directivity_index_dB is not None:
            raise ConfigurationError(
                "noise_background: pass either directivity_index_dB or "
                "array_gain_dB, not both — AG replaces DI. Got "
                f"directivity_index_dB={directivity_index_dB!r} and "
                f"array_gain_dB={array_gain_dB!r}."
            )
        return np.asarray(noise_level_dB, float) - np.asarray(array_gain_dB, float)
    di = 0.0 if directivity_index_dB is None else directivity_index_dB
    return np.asarray(noise_level_dB, float) - np.asarray(di, float)


def passive_signal_excess(
    source_level_dB, tl_dB, noise_level_dB, directivity_index_dB=None, *,
    detection_threshold_dB, array_gain_dB=None, processing_loss_dB=0.0,
):
    """Passive signal excess ``SE = SL - TL - (NL - DI) - DT - L_sp`` (dB).

    ``source_level_dB`` is the *target* radiated level, in the same band
    reference as ``noise_level_dB`` (see :func:`noise_background`). ``SE >= 0``
    means the detector achieves its design ``(P_D, P_F)`` operating point.
    ``array_gain_dB`` replaces ``directivity_index_dB`` for non-isotropic
    noise (see :func:`noise_background`); ``processing_loss_dB`` is the
    implementation/system loss ``L_sp >= 0`` (windowing, scalloping,
    beam-pattern, integration mismatch) subtracted from the budget.
    ``detection_threshold_dB`` is required here and in every budget of this
    module: it is the detector's own number, and a budget run without it
    reports ``SNR = 0 dB`` rather than any ``(P_D, P_F)``; pass ``0.0`` when
    that bare boundary is what you want.

    Parameters
    ----------
    source_level_dB : float or array_like
        Source level (dB). The target's radiated level.
    tl_dB : float or array_like
        One-way transmission loss (dB).
    noise_level_dB : float or array_like
        Noise level (dB), in the band reference of the source level
        (see :func:`noise_background`).
    directivity_index_dB : float or array_like, optional
        Directivity index (dB); ``None`` is 0 dB.
    detection_threshold_dB : float
        The detector's threshold (dB); ``0.0`` for the bare SNR boundary.
    array_gain_dB : float or array_like, optional
        Array gain (dB) in place of the directivity index; passing both
        is refused.
    processing_loss_dB : float, optional
        Implementation loss ``L_sp >= 0`` (dB). Default 0.

    Examples
    --------
    ``SE = SL − TL − NL + DI − DT``, here with no array (``DI = 0``):

    >>> float(passive_signal_excess(160.0, 70.0, 60.0, detection_threshold_dB=10.0))
    20.0
    """
    _reject_field(tl_dB, 'passive_signal_excess', 'tl_dB',
                  'passive_signal_excess_field')
    return (
        np.asarray(source_level_dB, float)
        - np.asarray(tl_dB, float)
        - noise_background(noise_level_dB, directivity_index_dB,
                           array_gain_dB=array_gain_dB)
        - np.asarray(detection_threshold_dB, float)
        - np.asarray(processing_loss_dB, float)
    )


def active_signal_excess(
    source_level_dB,
    tl_dB,
    target_strength_dB,
    *,
    detection_threshold_dB,
    noise_level_dB=None,
    directivity_index_dB=None,
    reverberation_level_dB=None,
    array_gain_dB=None,
    processing_loss_dB=0.0,
):
    """Active signal excess (dB), noise- or reverberation-limited.

    Provide ``noise_level_dB`` (noise-limited), ``reverberation_level_dB``
    (reverb-limited), or both — in which case the louder background
    (incoherent sum) is used per range.

        noise-limited:  ``SE = SL - 2*TL + TS - (NL - DI) - DT``
        reverb-limited: ``SE = SL - 2*TL + TS - RL - DT``

    ``directivity_index_dB`` / ``array_gain_dB`` is **not** applied against
    ``RL`` (Urick Ch. 8): a beamformed receiver offers no gain against
    in-beam reverberation, whose level is already beam-limited through
    the scattering-cell size (``horizontal_beamwidth_rad`` in
    :func:`uacpy.sonar.reverberation.boundary_reverberation`).
    ``array_gain_dB`` replaces ``directivity_index_dB`` for non-isotropic
    noise; ``processing_loss_dB`` is the implementation loss
    ``L_sp >= 0`` subtracted from the budget. ``SL``, ``NL`` and ``RL``
    share one band reference — see :func:`noise_background`.

    Parameters
    ----------
    source_level_dB : float or array_like
        Source level (dB).
    tl_dB : float or array_like
        One-way transmission loss (dB).
    target_strength_dB : float or array_like
        Target strength (dB).
    detection_threshold_dB : float
        The detector's threshold (dB); ``0.0`` for the bare SNR boundary.
    noise_level_dB : float or array_like, optional
        Noise level (dB), for the noise-limited budget.
    directivity_index_dB : float or array_like, optional
        Directivity index (dB); ``None`` is 0 dB.
    reverberation_level_dB : float or array_like, optional
        Reverberation level (dB), for the reverberation-limited budget.
    array_gain_dB : float or array_like, optional
        Array gain (dB) in place of the directivity index; passing both
        is refused.
    processing_loss_dB : float, optional
        Implementation loss ``L_sp >= 0`` (dB). Default 0.
    """
    _reject_field(tl_dB, 'active_signal_excess', 'tl_dB',
                  'active_signal_excess_field')
    if noise_level_dB is None and reverberation_level_dB is None:
        raise ConfigurationError(
            "active_signal_excess: provide noise_level_dB and/or reverberation_level_dB."
        )
    el = echo_level(source_level_dB, tl_dB, target_strength_dB)
    backgrounds = []
    if noise_level_dB is not None:
        backgrounds.append(noise_background(noise_level_dB, directivity_index_dB,
                                            array_gain_dB=array_gain_dB))
    if reverberation_level_dB is not None:
        backgrounds.append(np.asarray(reverberation_level_dB, float))
    # The incoherent dB sum of the backgrounds is total_reverberation's, and
    # so is its refusal of an empty budget (no noise and no reverberation).
    background = total_reverberation(*backgrounds)
    return (el - background - np.asarray(detection_threshold_dB, float)
            - np.asarray(processing_loss_dB, float))


def figure_of_merit(
    source_level_dB, noise_level_dB, directivity_index_dB=None, *,
    detection_threshold_dB, array_gain_dB=None, processing_loss_dB=0.0,
):
    """Figure of merit ``FOM = SL - (NL - DI) - DT - L_sp`` (dB).

    Equals the maximum allowable one-way TL (passive), or two-way TL when
    ``TS = 0`` (active). ``array_gain_dB`` / ``processing_loss_dB`` as in
    :func:`passive_signal_excess`; ``source_level_dB`` and ``noise_level_dB`` share
    one band reference — see :func:`noise_background`.

    Parameters
    ----------
    source_level_dB : float or array_like
        Source level (dB).
    noise_level_dB : float or array_like
        Noise level (dB), in the band reference of the source level
        (see :func:`noise_background`).
    directivity_index_dB : float or array_like, optional
        Directivity index (dB); ``None`` is 0 dB.
    detection_threshold_dB : float
        The detector's threshold (dB); ``0.0`` for the bare SNR boundary.
    array_gain_dB : float or array_like, optional
        Array gain (dB) in place of the directivity index; passing both
        is refused.
    processing_loss_dB : float, optional
        Implementation loss ``L_sp >= 0`` (dB). Default 0.
    """
    return (
        np.asarray(source_level_dB, float)
        - noise_background(noise_level_dB, directivity_index_dB,
                           array_gain_dB=array_gain_dB)
        - np.asarray(detection_threshold_dB, float)
        - np.asarray(processing_loss_dB, float)
    )


def _tl_array_from_field(tl_field, who: str) -> np.ndarray:
    """One-way TL (dB) at the field's grid, from a real-dB or complex Field.

    The field must carry a propagation loss: ``Field.kind`` names the
    quantity, so a received level (already ``SL - TL``), a signal excess (the
    output of this budget) or a reverberation loss (a different loss) is
    refused rather than read as TL — each is a dB grid that would pass for one.

    A coherent field's TL carries the interference fringes a detection budget
    should not see; it is accepted with a warning naming the incoherent
    routes. Whether it is coherent is :attr:`Field.coherent`'s answer, the
    one decider the package has: it reads a producer's stamp first (an
    incoherent ``superpose`` of coherent slabs says ``False``), then a phase
    reference (a ``BROADBAND`` result sliced at one frequency, or coherent
    pressure after :meth:`~Field.to_dB`) or a ``COHERENT_TL`` run mode
    (OAST's coherent TL, stored as real dB).
    """
    if not isinstance(tl_field, Field):
        raise ConfigurationError(
            f"{who}: expected a Field, got {type(tl_field).__name__}."
        )
    if 'time' in tl_field.coords:
        raise ConfigurationError(
            f"{who}: a time-domain Field is not transmission "
            "loss; pass a TL / pressure Field (e.g. from "
            f"run_mode=COHERENT_TL). Got axes {list(tl_field.coords)}."
        )
    kind = tl_field.kind
    if not is_loss(kind) or kind == 'reverberation':
        raise ConfigurationError(
            f"{who}: tl_field is a {kind!r} field, not transmission loss "
            f"— its dB values would be read as one-way TL.",
            remediation=("Pass the propagation result itself (a model run, "
                         "or Field.broadband_loss()). A 'level' field is "
                         "SL - TL already; a 'signal_excess' field is what "
                         "this function returns."))
    if tl_field.coherent:
        warnings.warn(
            f"{who}: tl_field is coherent TL, so it carries the "
            f"interference fringes, and a detection range read off it can "
            f"land on a constructive fringe past the last reliable crossing. "
            f"Run with run_mode=RunMode.INCOHERENT_TL, or pass "
            f"Field.broadband_loss(), for a budget; filter this warning when "
            f"the coherent field is intended.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    return tl_field.dB


def _require_se_field(se_field, who: str, *, remediation: str) -> None:
    """Raise unless ``se_field`` is a signal-excess :class:`Field`.

    Every dB grid passes for signal excess by its values alone: a TL of
    60 dB read as SE is +60 dB of excess, so a detection range of ``inf``
    and ``P_D = 1``. ``Field.kind`` is what tells them apart.
    """
    if not isinstance(se_field, Field):
        raise ConfigurationError(
            f"{who}: expected a Field, got {type(se_field).__name__}.")
    if se_field.kind != 'signal_excess':
        raise ConfigurationError(
            f"{who}: se_field is a {se_field.kind!r} field, not signal "
            f"excess — its dB values would be read as SE, and a TL of "
            f"60 dB as +60 dB of excess.",
            remediation=remediation)


def _reject_field(value, who: str, label: str, twin: str) -> None:
    """Raise a typed error naming ``twin`` when ``value`` is a ``Field``.

    The scalar sonar-equation functions take arrays and the ``*_field``
    functions take a :class:`~uacpy.core.results.Field`; handing a Field to the
    scalar one reaches ``float()`` and raises
    ``TypeError: float() argument must be … not 'Field'``, which names neither
    the argument nor the function one suffix away.
    """
    if isinstance(value, Field):
        raise ConfigurationError(
            f"{who}: {label} is a Field; this function takes dB arrays. "
            f"Use {twin}(...) for a Field, or pass np.asarray(field.dB, "
            f"float) as {label} to stay here (field.dB is the loss in dB; the "
            f"raw payload is complex pressure).")


def _require_scalar_dB(value, who: str, label: str) -> float:
    """Validate a sonar-budget term documented as a scalar and return it.

    ``reverberation_level_dB`` is the one term of the budget that may be
    per-range; the rest are scalars, and an array reached ``float()`` at the
    budget dict *after* the signal excess had already been computed, raising
    ``TypeError: only 0-dimensional arrays can be converted to Python
    scalars`` — no function name, no argument name, and the work thrown away.
    """
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 0:
        raise ConfigurationError(
            f"{who}: {label} must be a scalar dB level; got shape "
            f"{arr.shape}. reverberation_level_dB is the one per-range term of "
            f"this budget — for a range-varying background, pass the profile "
            f"there, or evaluate the field once per {label} value.")
    return float(arr)


def _require_per_range(values: np.ndarray, tl_field, who: str) -> None:
    """Refuse a 1-D per-range ``reverberation_level_dB`` that does not run
    along ``tl_field``'s range axis."""
    if 'range' not in tl_field.coords:
        raise ConfigurationError(
            f"{who}: per-range reverberation_level_dB requires the Field "
            f"to carry a 'range' axis; got {list(tl_field.coords)}."
        )
    n_r = tl_field.coords['range'].size
    if values.size != n_r:
        raise ConfigurationError(
            f"{who}: reverberation_level_dB length ({values.size}) must "
            f"match the field's range axis ({n_r})"
        )


def _keep_no_energy_marker(se, tl_dB):
    """``se`` with every cell whose one-way loss is the no-energy marker
    (:func:`~uacpy.core.acoustics.levels.no_energy_mask`) set to the level
    view of that marker, ``-NO_ENERGY_DB``. ``SL - TL - …`` would put such a
    cell at a finite excess hundreds of dB down that the mask no longer
    recognises (the rule :func:`~uacpy.core.acoustics.levels.received_level_dB`
    applies to ``SL - TL``). The mask reads the one-way TL, so a two-way
    budget does not turn a genuine deep null past 300 dB into the marker."""
    return np.where(no_energy_mask(tl_dB), -NO_ENERGY_DB, se)


def _spawn_se_field(tl_field, se: np.ndarray, budget: dict) -> Field:
    """Wrap ``se`` in a Field carrying the TL field's identity, the band
    it averaged and its sub-cutoff bin count, and the budget
    (:meth:`SonarBudget.to_dict`) as its ``sonar_budget``.

    The ``kind`` matters: signal excess is dB but it is not pressure and
    not a loss, so leaving it to derive would both mislabel it and make
    :meth:`Field.max` report the *worst* cell as the best."""
    kwargs = tl_field.id_kwargs()
    kwargs['kind'] = 'signal_excess'
    # A complex TL field's unit is Pa; the excess built from it is dB.
    kwargs['unit'] = 'dB'
    return Field(
        data=np.asarray(se, dtype=float),
        coords={k: v.copy() for k, v in tl_field.coords.items()},
        pinned=dict(tl_field.pinned),
        band_hz=tl_field.band_hz,
        sub_cutoff_bins=tl_field.sub_cutoff_bins,
        sonar_budget=budget,
        **kwargs,
    )


def _require_gain_fits(gain: np.ndarray, tl_dB, who: str) -> None:
    """Refuse an array ``array_gain_dB`` that does not broadcast to the TL
    grid.

    A beamformer's realised gain is a grid whenever the signal is not a
    single plane wave: in a waveguide each trapped mode arrives at its own
    grazing angle, so the share of the signal a beam collects changes with
    target depth and range. Such a grid must cover the TL grid it is
    evaluated against.
    """
    try:
        fits = (np.broadcast_shapes(gain.shape, np.shape(tl_dB))
                == np.shape(tl_dB))
    except ValueError:
        fits = False
    if not fits:
        raise ConfigurationError(
            f"{who}: array_gain_dB must be a scalar dB gain, or an array "
            f"broadcasting to the TL grid {np.shape(tl_dB)}; got shape "
            f"{gain.shape}. A per-sample AG is what a beamformer realises on "
            f"a modelled field — see Field.data for the grid it must match."
        )


_BUDGET_MODES = ('passive', 'active')


def _budget_terms(who: str, mode, source_level_dB, detection_threshold_dB,
                  noise_level_dB=None, target_strength_dB=None,
                  reverberation_level_dB=None, directivity_index_dB=None,
                  array_gain_dB=None, processing_loss_dB=0.0) -> dict:
    """The terms of one sonar budget, checked, in the form
    :class:`SonarBudget` holds them; every refusal names ``who``.

    The scalar terms become floats. ``reverberation_level_dB`` becomes a
    float or a read-only 1-D per-range array, ``array_gain_dB`` a float or a
    read-only array of any shape (a grid is checked against the TL grid it
    is evaluated on, which is not known here)."""
    if mode not in _BUDGET_MODES:
        raise ConfigurationError(
            f"{who}: mode must be 'passive' or 'active'; got {mode!r}.")
    if mode == 'passive':
        active_only = [name for name, value in (
            ('target_strength_dB', target_strength_dB),
            ('reverberation_level_dB', reverberation_level_dB))
            if value is not None]
        if active_only:
            one = len(active_only) == 1
            raise ConfigurationError(
                f"{who}: {' and '.join(active_only)} "
                f"{'belongs' if one else 'belong'} to an active (echo) "
                f"budget; a passive budget takes "
                f"{'no target strength or reverberation level' if one else 'neither'}.",
                remediation="Pass mode='active' for an echo budget, or leave "
                            "the active terms out of a passive one.")
    if noise_level_dB is not None:
        noise_level_dB = _require_scalar_dB(noise_level_dB, who,
                                            'noise_level_dB')
    source_level_dB = _require_scalar_dB(source_level_dB, who,
                                         'source_level_dB')
    if target_strength_dB is not None:
        target_strength_dB = _require_scalar_dB(target_strength_dB, who,
                                                'target_strength_dB')
    detection_threshold_dB = _require_scalar_dB(
        detection_threshold_dB, who, 'detection_threshold_dB')
    processing_loss_dB = _require_scalar_dB(processing_loss_dB, who,
                                            'processing_loss_dB')
    if directivity_index_dB is not None:
        directivity_index_dB = _require_scalar_dB(
            directivity_index_dB, who, 'directivity_index_dB')
    if reverberation_level_dB is not None:
        rl = np.array(reverberation_level_dB, dtype=float)
        if rl.ndim > 1:
            raise ConfigurationError(
                f"{who}: reverberation_level_dB must be a scalar or a 1-D "
                f"per-range array; got shape {rl.shape}.")
        if rl.ndim == 0:
            reverberation_level_dB = float(rl)
        else:
            rl.setflags(write=False)
            reverberation_level_dB = rl
    if array_gain_dB is not None:
        ag = np.array(array_gain_dB, dtype=float)
        if ag.ndim == 0:
            array_gain_dB = float(ag)
        else:
            ag.setflags(write=False)
            array_gain_dB = ag
    if mode == 'passive' and noise_level_dB is None:
        raise ConfigurationError(
            f"{who}: a passive budget needs noise_level_dB, the background "
            f"the target's radiated level is detected against.")
    if mode == 'active' and target_strength_dB is None:
        raise ConfigurationError(
            f"{who}: an active budget needs target_strength_dB, the echo "
            f"the target returns.")
    if mode == 'active' and noise_level_dB is None \
            and reverberation_level_dB is None:
        raise ConfigurationError(
            f"{who}: provide noise_level_dB and/or reverberation_level_dB.")
    return dict(mode=mode, source_level_dB=source_level_dB,
                detection_threshold_dB=detection_threshold_dB,
                noise_level_dB=noise_level_dB,
                target_strength_dB=target_strength_dB,
                reverberation_level_dB=reverberation_level_dB,
                directivity_index_dB=directivity_index_dB,
                array_gain_dB=array_gain_dB,
                processing_loss_dB=processing_loss_dB)


def _level_span(values: np.ndarray, unit: str = 'dB') -> str:
    """``'6-14 dB (median 10 dB) over 35 samples'`` for an array term."""
    return (f"{np.nanmin(values):g}-{np.nanmax(values):g} {unit} (median "
            f"{np.nanmedian(values):g} {unit}) over {values.size} samples")


@dataclasses.dataclass(frozen=True, eq=False)
class SonarBudget(FieldsRepr):
    """The terms of one sonar-equation budget: everything but the
    transmission loss it is evaluated against.

    ``mode='passive'`` is ``SE = SL - TL - (NL - DI) - DT - L_sp``;
    ``mode='active'`` is ``SE = SL - 2*TL + TS - B - DT - L_sp`` with ``B``
    the noise background ``NL - DI``, the reverberation level ``RL``, or
    their incoherent sum (see :func:`active_signal_excess`). All terms are dB
    and share one band reference (see :func:`noise_background`).

    Attributes
    ----------
    mode : {'passive', 'active'}
    source_level_dB : float
        SL at 1 m: the target's radiated level (passive) or the projector's
        (active).
    detection_threshold_dB : float
        DT, the detector's own number; ``0.0`` for the bare ``SNR = 0 dB``
        boundary.
    noise_level_dB : float or None
        NL. Required for a passive budget; an active one needs it and/or
        ``reverberation_level_dB``.
    target_strength_dB : float or None
        TS. Required for an active budget, refused by a passive one.
    reverberation_level_dB : float, ndarray or None
        RL, active only: a scalar or a read-only 1-D per-range array.
    directivity_index_dB : float or None
        DI; ``None`` (not supplied) counts as 0 dB.
    array_gain_dB : float, ndarray or None
        AG, replacing DI against noise (never against RL): one number, or a
        read-only grid of the realised gain over the TL grid.
    processing_loss_dB : float
        Implementation loss ``L_sp >= 0``.

    The terms are checked when the budget is built. :meth:`signal_excess`
    evaluates it on TL arrays, :meth:`signal_excess_field` on a model TL
    :class:`~uacpy.core.results.Field`; the Field records
    :meth:`to_dict` as its ``sonar_budget``, and
    :meth:`from_dict` rebuilds the budget from it.

    Examples
    --------
    >>> budget = SonarBudget('passive', source_level_dB=160.0,
    ...                      noise_level_dB=60.0, detection_threshold_dB=10.0)
    >>> float(budget.signal_excess(70.0))
    20.0
    >>> float(budget.figure_of_merit())
    90.0
    """

    mode: str
    source_level_dB: float
    detection_threshold_dB: float
    noise_level_dB: Optional[float] = None
    target_strength_dB: Optional[float] = None
    reverberation_level_dB: Union[float, np.ndarray, None] = None
    directivity_index_dB: Optional[float] = None
    array_gain_dB: Union[float, np.ndarray, None] = None
    processing_loss_dB: float = 0.0

    def __post_init__(self):
        self._set_terms(_budget_terms('SonarBudget', **self._terms()))

    @classmethod
    def _checked(cls, who: str, **terms) -> 'SonarBudget':
        """The budget of ``terms``, its refusals naming ``who`` (the
        public function the terms were passed to)."""
        return cls(**_budget_terms(who, **terms))

    def _terms(self) -> dict:
        return {f.name: getattr(self, f.name)
                for f in dataclasses.fields(self)}

    def _set_terms(self, terms: dict) -> None:
        for name, value in terms.items():
            object.__setattr__(self, name, value)

    def __setstate__(self, state):
        # Unpickling hands arrays back writeable; the terms are read-only.
        self._set_terms(state)
        for value in state.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    def signal_excess(self, tl_dB, *, range_axis=-1):
        """Signal excess (dB) at each one-way TL of ``tl_dB``:
        :func:`passive_signal_excess` or :func:`active_signal_excess` with
        this budget's terms.

        A per-range ``reverberation_level_dB`` runs along ``range_axis`` of
        ``tl_dB``; an ``array_gain_dB`` grid broadcasts against ``tl_dB``.
        """
        _reject_field(tl_dB, 'SonarBudget.signal_excess', 'tl_dB',
                      'SonarBudget.signal_excess_field')
        rl = self.reverberation_level_dB
        ndim = np.ndim(tl_dB)
        if isinstance(rl, np.ndarray) and ndim > 0:
            axis = range_axis + ndim if range_axis < 0 else range_axis
            if not 0 <= axis < ndim:
                raise ConfigurationError(
                    f"SonarBudget.signal_excess: range_axis={range_axis} is "
                    f"outside a {ndim}-D tl_dB.")
            shape = [1] * ndim
            shape[axis] = rl.size
            rl = rl.reshape(shape)
        if self.mode == 'passive':
            return passive_signal_excess(
                self.source_level_dB, tl_dB, self.noise_level_dB,
                directivity_index_dB=self.directivity_index_dB,
                detection_threshold_dB=self.detection_threshold_dB,
                array_gain_dB=self.array_gain_dB,
                processing_loss_dB=self.processing_loss_dB,
            )
        return active_signal_excess(
            self.source_level_dB, tl_dB, self.target_strength_dB,
            noise_level_dB=self.noise_level_dB,
            directivity_index_dB=self.directivity_index_dB,
            reverberation_level_dB=rl,
            detection_threshold_dB=self.detection_threshold_dB,
            array_gain_dB=self.array_gain_dB,
            processing_loss_dB=self.processing_loss_dB,
        )

    def figure_of_merit(self):
        """The largest loss (dB) this budget tolerates: the TL at which
        ``SE = 0``.

        Passive, the one-way TL: :func:`figure_of_merit`. Active, the
        two-way loss ``2*TL``, which is ``SL + TS - B - DT - L_sp``: the
        echo budget at ``TL = 0``, per range when ``reverberation_level_dB``
        is. An ``array_gain_dB`` grid gives a grid.
        """
        if self.mode == 'passive':
            return figure_of_merit(
                self.source_level_dB, self.noise_level_dB,
                self.directivity_index_dB,
                detection_threshold_dB=self.detection_threshold_dB,
                array_gain_dB=self.array_gain_dB,
                processing_loss_dB=self.processing_loss_dB,
            )
        return self.signal_excess(0.0)

    def signal_excess_field(self, tl_field) -> Field:
        """This budget over a model TL :class:`~uacpy.core.results.Field`:
        :func:`passive_signal_excess_field` or
        :func:`active_signal_excess_field` with its terms."""
        terms = self._terms()
        mode = terms.pop('mode')
        if mode == 'passive':
            del terms['target_strength_dB'], terms['reverberation_level_dB']
            return passive_signal_excess_field(tl_field, **terms)
        return active_signal_excess_field(tl_field, **terms)

    def to_dict(self) -> dict:
        """Every term as plain Python: floats, ``None`` for a term not
        supplied, nested lists (shape kept) for an array term."""
        return {name: (value.tolist() if isinstance(value, np.ndarray)
                       else value)
                for name, value in self._terms().items()}

    @classmethod
    def from_dict(cls, d: dict) -> 'SonarBudget':
        """The budget :meth:`to_dict` wrote."""
        return cls(**d)

    def summary(self) -> str:
        """One line naming each supplied term; an array term by its span."""
        labels = (('source_level_dB', 'SL'), ('noise_level_dB', 'NL'),
                  ('target_strength_dB', 'TS'),
                  ('reverberation_level_dB', 'RL'),
                  ('directivity_index_dB', 'DI'), ('array_gain_dB', 'AG'),
                  ('detection_threshold_dB', 'DT'),
                  ('processing_loss_dB', 'L_sp'))
        bits = []
        for name, label in labels:
            value = getattr(self, name)
            if value is None:
                continue
            bits.append(f"{label} {_level_span(value)}"
                        if isinstance(value, np.ndarray)
                        else f"{label} {value:g} dB")
        return f"{self.mode}: " + ', '.join(bits)

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self):
        return hash(repr(self.to_dict()))


def _signal_excess_map(tl_field, tl_dB, budget: SonarBudget,
                       who: str) -> Field:
    """``budget`` evaluated over the TL grid of ``tl_field``, as a
    signal-excess Field recording ``budget.to_dict()``."""
    if isinstance(budget.reverberation_level_dB, np.ndarray):
        _require_per_range(budget.reverberation_level_dB, tl_field, who)
    if isinstance(budget.array_gain_dB, np.ndarray):
        _require_gain_fits(budget.array_gain_dB, tl_dB, who)
    axes = list(tl_field.coords)
    se = budget.signal_excess(
        tl_dB, range_axis=axes.index('range') if 'range' in axes else -1)
    se = _keep_no_energy_marker(se, tl_dB)
    return _spawn_se_field(tl_field, se, budget.to_dict())


def passive_signal_excess_field(
    tl_field,
    *,
    source_level_dB,
    noise_level_dB,
    detection_threshold_dB,
    directivity_index_dB=None,
    array_gain_dB=None,
    processing_loss_dB=0.0,
) -> Field:
    """Passive signal excess over a model TL grid: ``SE = SL - TL - (NL - DI) - DT``.

    Grid counterpart of :func:`passive_signal_excess`: takes the
    :class:`~uacpy.core.results.Field` a propagation model returned
    (real dB TL, or complex pressure — converted via ``Field.dB``) and
    evaluates the sonar equation at every ``(depth, range)`` sample.

    Parameters
    ----------
    tl_field : Field
        One-way TL field from any model run; a budget wants
        ``run_mode=RunMode.INCOHERENT_TL`` or ``Field.broadband_loss()``
        (a coherent field is accepted with a warning, since its fringes
        move the detection range).
    source_level_dB : float
        Target radiated level SL, at 1 m.
    noise_level_dB : float
        Ambient noise level NL at the array, in the same band reference as
        ``source_level_dB`` — see :func:`noise_background`.
    directivity_index_dB : float, optional
        Receiving directivity index DI (dB). Default 0.
    detection_threshold_dB : float
        Detection threshold DT (dB) — see
        :func:`uacpy.sonar.detection.detection_threshold_energy`. Required:
        it is a statement about the detector, and ``SE = 0`` means its
        ``(P_D, P_F)`` operating point only when DT is that detector's.
        Pass ``0.0`` explicitly for a bare ``SNR = 0 dB`` boundary.
    array_gain_dB : float or array_like, optional
        Replaces ``directivity_index_dB`` for non-isotropic noise (see
        :func:`noise_background`). May be one number, or a grid
        broadcasting to the TL field: the gain a beamformer *realises*
        varies over the grid, because a waveguide delivers a sum of modes
        rather than the single plane wave a scalar AG assumes. The budget
        metadata records the grid in full.
    processing_loss_dB : float, optional
        Implementation/system loss ``L_sp >= 0`` (dB). Default 0.

    Returns
    -------
    Field
        Signal excess (dB) on the same coords/pinned grid; ``SE >= 0``
        marks the detectable region. ``result.sonar_budget`` is
        the :meth:`SonarBudget.to_dict` of the budget evaluated. Plot with
        :func:`uacpy.visualization.plots.plot_signal_excess`.
    """
    who = 'passive_signal_excess_field'
    tl_dB = _tl_array_from_field(tl_field, who)
    budget = SonarBudget._checked(
        who, mode='passive', source_level_dB=source_level_dB,
        noise_level_dB=noise_level_dB,
        detection_threshold_dB=detection_threshold_dB,
        directivity_index_dB=directivity_index_dB,
        array_gain_dB=array_gain_dB, processing_loss_dB=processing_loss_dB)
    return _signal_excess_map(tl_field, tl_dB, budget, who)


def active_signal_excess_field(
    tl_field,
    *,
    source_level_dB,
    target_strength_dB,
    detection_threshold_dB,
    noise_level_dB=None,
    reverberation_level_dB=None,
    directivity_index_dB=None,
    array_gain_dB=None,
    processing_loss_dB=0.0,
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
    source_level_dB : float
        Projector source level SL, at 1 m.
    target_strength_dB : float
        Target strength TS (dB).
    noise_level_dB : float, optional
        Ambient noise level NL, in the same band reference as
        ``source_level_dB`` and ``reverberation_level_dB`` (see
        :func:`noise_background`). Provide this and/or
        ``reverberation_level_dB``.
    reverberation_level_dB : float or array, optional
        Reverberation level RL (dB) — a scalar, or a 1-D per-range array
        matching the field's ``'range'`` axis (e.g. from
        :func:`uacpy.sonar.reverberation.boundary_reverberation` on
        ``tl_field.coords['range']``).
    directivity_index_dB : float, optional
        Receiving directivity index DI (dB). Default 0.
    detection_threshold_dB : float
        Detection threshold DT (dB). Required — see
        :func:`passive_signal_excess_field`.
    array_gain_dB : float or array_like, optional
        Replaces ``directivity_index_dB`` against the noise background
        (never against ``RL`` — see :func:`active_signal_excess`). One
        number, or a grid broadcasting to the TL field — see
        :func:`passive_signal_excess_field`.
    processing_loss_dB : float, optional
        Implementation/system loss ``L_sp >= 0`` (dB). Default 0.

    Returns
    -------
    Field
        Signal excess (dB) on the same coords/pinned grid;
        ``result.sonar_budget`` is the
        :meth:`SonarBudget.to_dict` of the budget evaluated.
    """
    who = 'active_signal_excess_field'
    tl_dB = _tl_array_from_field(tl_field, who)
    budget = SonarBudget._checked(
        who, mode='active', source_level_dB=source_level_dB,
        target_strength_dB=target_strength_dB,
        detection_threshold_dB=detection_threshold_dB,
        noise_level_dB=noise_level_dB,
        reverberation_level_dB=reverberation_level_dB,
        directivity_index_dB=directivity_index_dB,
        array_gain_dB=array_gain_dB, processing_loss_dB=processing_loss_dB)
    return _signal_excess_map(tl_field, tl_dB, budget, who)


def transition_probability(signal_excess_dB, *, sigma_dB):
    """Urick's transition curve ``P_D = Phi(SE / sigma_dB)`` on an array of
    signal excess (dB).

    ``Phi`` is the standard normal CDF: the detector statistic is taken
    log-normal under signal-plus-noise, Gaussian in dB with mean ``DT + SE``
    and standard deviation ``sigma_dB`` (Urick Fig. 12.10; Abraham §2.3.7.1).
    ``P_D = 0.5`` on ``SE = 0``. What that curve does and does not model is
    set out at :func:`transition_probability_field`, its Field form.

    Parameters
    ----------
    signal_excess_dB : array_like
        Signal excess (dB); NaN stays NaN.
    sigma_dB : float
        Standard deviation of the signal-excess fluctuation (dB), > 0. No
        default: it is a physical claim about the channel.

    Returns
    -------
    ndarray
        ``P_D`` in [0, 1], the shape of ``signal_excess_dB``.

    Examples
    --------
    >>> float(transition_probability(0.0, sigma_dB=5.6))
    0.5
    """
    from scipy.stats import norm
    if isinstance(signal_excess_dB, Field):
        raise ConfigurationError(
            "transition_probability: signal_excess_dB is a Field; this "
            "function takes dB arrays. Use transition_probability_field(...) "
            "for a signal-excess Field, or pass its .data.")
    sigma = require_positive_finite_scalar(
        sigma_dB, "transition_probability", "sigma_dB", " dB")
    return norm.cdf(np.asarray(signal_excess_dB, dtype=float) / sigma)


def transition_probability_field(se_field, *, sigma_dB) -> Field:
    """Detection-probability field from a signal-excess field: the Field form
    of :func:`transition_probability`.

    Urick's transition curve (Fig. 12.10; Abraham §2.3.7.1): the detector
    decision statistic is taken log-normal under signal-plus-noise, so
    in dB it is Gaussian with mean ``DT + SE`` and standard deviation
    ``sigma_dB``, giving

        ``P_D = Phi(SE / sigma_dB)``

    with ``Phi`` the standard normal CDF. ``P_D = 0.5`` exactly on the
    ``SE = 0`` contour, so the surface is an absolute ``P_D`` only when the
    budget's ``detection_threshold_dB`` was designed at ``P_D = 0.5`` (the
    minimum detectable level). For a DT designed at any other ``P_D`` read
    it as a relative surface: its ``SE = 0`` contour still reports 0.5, and
    shifting the curve by ``Phi^-1(P_D)`` does not repair it, because
    ``sigma_dB`` is the channel's fluctuation, not the width of the
    detector's own transition. For an exact energy detector (``M = 100``,
    ``P_F = 1e-4``) with DT designed at ``P_D = 0.9`` and ``sigma_dB = 5.6``,
    averaging the detector's ``P_D`` over the log-normal channel gives
    0.624 at ``SE = 0``, where this function returns 0.500 and the shifted
    curve 0.900 (measured).

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
    sigma_dB : float
        Standard deviation of the signal-excess fluctuation (dB).
        Dyer's saturated-multipath result gives ``sigma_dB ≈ 5.6``;
        measured one-way totals typically run 5–9 dB. No default —
        it is a physical claim about the channel, not a processing knob.

    Returns
    -------
    Field
        ``P_D`` in [0, 1] on the same coords/pinned grid;
        ``result.sigma_dB`` records the fluctuation model. Plot
        with :func:`uacpy.visualization.plots.plot_detection_probability`.

    Notes
    -----
    The log-normal approximation is most accurate near ``SE = 0`` and
    optimistic in the tails; as ``sigma_dB → 0`` it degenerates to a
    step at ``SE = 0`` rather than the deterministic-signal ROC
    (Abraham §2.3.5.6). For fluctuation statistics beyond Gaussian-in-dB
    (e.g. the gamma-fluctuating-intensity model), compute ``P_D`` from
    the detector statistics directly via :mod:`uacpy.sonar.detection`.
    """
    if isinstance(se_field, Field) and se_field.is_complex:
        raise ConfigurationError(
            "transition_probability_field: field must carry real "
            "signal excess in dB — build it with "
            "passive/active_signal_excess_field. Got dtype "
            f"{se_field.data.dtype}."
        )
    _require_se_field(
        se_field, 'transition_probability_field',
        remediation="Build it with passive_signal_excess_field / "
                    "active_signal_excess_field, which tag their result "
                    "kind='signal_excess'.")
    sigma = require_positive_finite_scalar(
        sigma_dB, "transition_probability_field", "sigma_dB", " dB")
    pd = transition_probability(se_field.data, sigma_dB=sigma)
    kwargs = se_field.id_kwargs()
    # A probability is dimensionless, not dB; inheriting the SE field's unit
    # would put a 0–1 array on a decibel axis.
    kwargs['kind'] = 'probability_of_detection'
    kwargs['unit'] = '1'
    return Field(
        data=pd,
        coords={k: v.copy() for k, v in se_field.coords.items()},
        pinned=dict(se_field.pinned),
        band_hz=se_field.band_hz,
        sub_cutoff_bins=se_field.sub_cutoff_bins,
        sonar_budget=se_field.sonar_budget,
        sigma_dB=sigma,
        **kwargs,
    )


def detection_ranges_by_depth(se_field, *, crossing='outermost'):
    """Per-depth detection range from a 2-D signal-excess field.

    The Field form of :func:`detection_ranges`: applies
    :func:`detection_range` to each depth row of a canonical
    ``['depth', 'range']`` signal-excess :class:`Field`.

    ``crossing`` is passed to :func:`detection_range` (``'outermost'`` or
    ``'first'``); the returned ranges below describe ``'outermost'``.

    Parameters
    ----------
    se_field : Field
        A ``['depth', 'range']`` signal-excess Field.
    crossing : {'outermost', 'first'}, optional
        Which zero-crossing to report (:func:`detection_range`). Default
        ``'outermost'``.

    Returns
    -------
    depths : ndarray, shape ``(n_depths,)``
        The field's depth axis (m).
    ranges : ndarray, shape ``(n_depths,)``
        Detection range (m) at each depth — the outermost zero-crossing,
        ``np.inf`` where SE >= 0 at the last sampled range (the crossing lies
        beyond the grid; one warning per call names how many depths have an
        in-grid shadow before it, and the first), and ``np.nan`` where SE < 0
        everywhere.
    """
    _require_se_field(
        se_field, 'detection_ranges_by_depth',
        remediation="Build it with passive_signal_excess_field / "
                    "active_signal_excess_field, which tag their result "
                    "kind='signal_excess'.")
    if list(se_field.coords) != ['depth', 'range']:
        raise ConfigurationError(
            "detection_ranges_by_depth: requires canonical "
            f"['depth', 'range'] coords; got {list(se_field.coords)}."
        )
    depths = se_field.coords['depth'].copy()
    out = _detection_ranges(se_field.coords['range'], se_field.data, 1,
                            crossing, who='detection_ranges_by_depth',
                            labels=('depth', depths))
    return depths, out


def detection_ranges(ranges_m, *, signal_excess_dB, axis=-1,
                     crossing='outermost'):
    """:func:`detection_range` along one axis of a signal-excess array.

    Each 1-D slice of ``signal_excess_dB`` along ``axis`` is one
    signal-excess-versus-range curve over ``ranges_m``; the result holds its
    detection range, in the shape of ``signal_excess_dB`` without ``axis``.
    ``crossing`` and the ``inf``/``nan`` returns are :func:`detection_range`'s,
    one slice at a time; its warning (an ``inf`` whose curve dips below zero
    inside the grid) is given once per call, with the count of such slices
    and the first one's shadow.

    Parameters
    ----------
    ranges_m : array_like, shape ``(n_r,)``
        Monotonically increasing ranges (m).
    signal_excess_dB : array_like
        Signal excess (dB) with ``n_r`` samples along ``axis``.
    axis : int
        The range axis of ``signal_excess_dB``.
    crossing : {'outermost', 'first'}
        Which crossing to report, as :func:`detection_range`.

    Returns
    -------
    ndarray
        Detection range (m) per slice.

    Examples
    --------
    >>> import numpy as np
    >>> ranges = np.linspace(0.0, 4000.0, 5)
    >>> se = np.array([[3.0, 1.0, -1.0, -2.0, -3.0],
    ...                [5.0, 4.0, 3.0, 2.0, -2.0]])
    >>> detection_ranges(ranges, signal_excess_dB=se).tolist()
    [1500.0, 3500.0]
    """
    return _detection_ranges(ranges_m, signal_excess_dB, axis, crossing,
                             who='detection_ranges')


def _detection_ranges(ranges_m, signal_excess_dB, axis, crossing, *, who,
                      labels=None):
    """:func:`detection_ranges` reporting as ``who``; ``labels``, a
    ``(name, values)`` pair over the slices, names a slice in the warning."""
    r = np.asarray(ranges_m, dtype=float)
    se = np.asarray(signal_excess_dB, dtype=float)
    if r.ndim != 1:
        raise ConfigurationError(
            f"{who}: ranges_m must be 1-D; got shape {r.shape}.")
    if se.ndim == 0:
        raise ConfigurationError(
            f"{who}: signal_excess_dB must have a range axis; got a "
            "scalar.")
    moved = np.moveaxis(se, axis, -1)
    if moved.shape[-1] != r.size:
        raise ConfigurationError(
            f"{who}: signal_excess_dB has {moved.shape[-1]} "
            f"samples along axis {axis}, ranges_m has {r.size}.")
    if crossing not in ('outermost', 'first'):
        raise ConfigurationError(
            f"{who}: crossing must be 'outermost' or 'first'; got "
            f"{crossing!r}.")
    rows = moved.reshape(-1, r.size)
    found = [_detection_range(r, row, crossing) for row in rows]
    shadowed = [(i, shadow) for i, (_, shadow) in enumerate(found)
                if shadow is not None]
    if shadowed:
        i, (edge, start, end) = shadowed[0]
        label = (f"{labels[0]} {labels[1][i]:g}" if labels is not None
                 else f"slice {i}")
        warnings.warn(
            f"{who}: {len(shadowed)} of {len(found)} "
            f"{'rows' if labels is not None else 'slices'} have signal "
            f"excess >= 0 at their outermost sampled range, so their "
            f"outermost crossing lies beyond the grid and inf is returned — "
            f"but SE goes negative inside the grid on each, so the target is "
            f"NOT detectable at every range (at {label}: SE >= 0 at "
            f"{edge:.6g} m, negative first between {start:.6g} and "
            f"{end:.6g} m). {_BEYOND_GRID_REMEDY}",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    out = np.array([value for value, _ in found], dtype=float)
    return out.reshape(moved.shape[:-1])


def detection_range_from_field(se_field, *, crossing='outermost') -> float:
    """Detection range (m) from a 1-D signal-excess :class:`Field` over range.

    The Field door to :func:`detection_range`: ``se.at(depth=z)`` of a
    signal-excess map is a ``['range']`` Field, and this reads its range axis
    and data so neither has to be pulled out by hand. ``crossing``, the return
    values (``inf`` when the outermost crossing lies beyond the grid, ``nan``
    when SE < 0 everywhere) and the no-data handling are
    :func:`detection_range`'s. For the whole ``(depth, range)`` map use
    :func:`detection_ranges_by_depth`.

    Parameters
    ----------
    se_field : Field
        A signal-excess Field over ``range`` alone.
    crossing : {'outermost', 'first'}, optional
        Which zero-crossing to report (:func:`detection_range`). Default
        ``'outermost'``.
    """
    _require_se_field(
        se_field, 'detection_range_from_field',
        remediation="Build it with passive_signal_excess_field / "
                    "active_signal_excess_field and slice one depth "
                    "with .at(depth=...).")
    if list(se_field.coords) != ['range']:
        raise ConfigurationError(
            f"detection_range_from_field: needs a 1-D ['range'] Field; got axes "
            f"{list(se_field.coords)}.",
            remediation="Slice one depth with se.at(depth=...), or use "
                        "detection_ranges_by_depth for the whole map.")
    return detection_range(se_field.coords['range'],
                           signal_excess_dB=np.asarray(se_field.data, dtype=float), crossing=crossing)


def detection_annuli(ranges_m, *, signal_excess_dB):
    """Every range interval (m) over which the signal excess is >= 0.

    A deep-water signal excess is typically + (direct path), - (shadow),
    + (convergence zone), - …, so a detection range is a set of annuli, not
    one number. Returns a list of ``(start_m, end_m)`` pairs, the edges
    placed by linear interpolation between samples (a sample's own range at a
    no-data hole or at the grid's edge). No-data (NaN) samples are skipped,
    as :func:`detection_range` skips them.

    Parameters
    ----------
    ranges_m : array_like
        Increasing ranges (m).
    signal_excess_dB : array_like
        Signal excess (dB) at each range; NaN is no data.
    """
    r = np.asarray(ranges_m, dtype=float)
    se = np.asarray(signal_excess_dB, dtype=float)
    if r.shape != se.shape:
        raise ConfigurationError(
            "detection_annuli: ranges and signal_excess shape mismatch; got "
            f"ranges_m shape {r.shape} and signal_excess_dB shape {se.shape}.")
    known = np.isfinite(se)
    idx = np.where(known)[0]
    r, se = r[idx], se[idx]
    annuli, start = [], None
    for i in range(r.size):
        if se[i] >= 0.0 and start is None:
            if i == 0 or idx[i] - idx[i - 1] > 1:
                start = float(r[i])
            else:
                frac = se[i - 1] / (se[i - 1] - se[i])
                start = float(r[i - 1] + frac * (r[i] - r[i - 1]))
        elif se[i] < 0.0 and start is not None:
            if idx[i] - idx[i - 1] > 1:
                end = float(r[i - 1])
            else:
                frac = se[i - 1] / (se[i - 1] - se[i])
                end = float(r[i - 1] + frac * (r[i] - r[i - 1]))
            annuli.append((start, end))
            start = None
    if start is not None:
        annuli.append((start, float(r[-1])))
    return annuli


def detection_range(ranges_m, *, signal_excess_dB, crossing='outermost'):
    """Detection range (m) from signal excess versus range.

    ``crossing='outermost'`` (default) is the largest range at which the signal
    excess is still non-negative; ``crossing='first'`` is the range of first
    loss of detection — the first downward crossing from the nearest range,
    the operational convention in deep water, where a convergence zone makes
    the outermost range say nothing about the shadow before it.
    :func:`detection_annuli` returns every interval.

    ``'outermost'`` finds the outermost zero-crossing of ``signal_excess_dB``
    versus range by linear interpolation, and returns ``np.nan`` if SE < 0
    everywhere. Whenever that crossing lies **beyond** the sampled ranges —
    SE >= 0 at the last sampled range with data — it returns ``np.inf``: the
    grid holds no crossing to report, and any finite number would be the
    grid's own edge, moving with ``receiver.ranges``. ``np.isfinite`` on the
    result therefore separates a modelled crossing from "beyond the grid".
    When SE >= 0 at the edge but went negative inside the grid (a shadow zone
    before a convergence zone), the ``inf`` comes with a
    ``NumericsWarning`` naming the in-grid shadow, since "beyond the grid"
    is then not "detected everywhere"; :func:`detection_annuli` and
    ``crossing='first'`` give that structure. When the outermost positive sample
    and the next finite sample are separated by no-data (NaN) cells, the
    positive sample's range is returned as-is — a crossing inside a no-data
    hole has no modeled location to interpolate.

    Parameters
    ----------
    ranges_m : array
        Monotonically increasing ranges (m).
    signal_excess_dB : array
        Signal excess (dB) at each range.
    crossing : {'outermost', 'first'}
        Which crossing to report, as above.

    Examples
    --------
    A convergence-zone return at 7 km: the first loss of detection is at
    4.5 km, the outermost at 7.25 km:

    >>> import numpy as np
    >>> ranges = np.linspace(0.0, 10_000.0, 11)
    >>> se = np.array([10, 8, 6, 3, 1, -1, -2, 1, -3, -4, -5.0])
    >>> float(detection_range(ranges, signal_excess_dB=se, crossing='first'))
    4500.0
    >>> float(detection_range(ranges, signal_excess_dB=se))
    7250.0
    """
    if crossing not in ('outermost', 'first'):
        raise ConfigurationError(
            f"detection_range: crossing must be 'outermost' or 'first'; got "
            f"{crossing!r}.")
    r = np.asarray(ranges_m, dtype=float)
    se = np.asarray(signal_excess_dB, dtype=float)
    if r.shape != se.shape:
        raise ConfigurationError(
            "detection_range: ranges and signal_excess shape mismatch; got "
            f"ranges_m shape {r.shape} and signal_excess_dB shape {se.shape}.")
    value, shadow = _detection_range(r, se, crossing)
    if shadow is not None:
        warnings.warn(
            f"detection_range: signal excess is >= 0 at the outermost sampled "
            f"range with data ({shadow[0]:.6g} m), so the outermost "
            f"crossing lies beyond the grid and inf is returned — but SE goes "
            f"negative inside the grid, first between {shadow[1]:.6g} and "
            f"{shadow[2]:.6g} m, so the target is NOT detectable at every "
            f"range. {_BEYOND_GRID_REMEDY}",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    return value


#: What to do about an ``inf`` detection range whose curve dips below zero
#: inside the grid.
_BEYOND_GRID_REMEDY = (
    "detection_annuli(...) lists every detectable interval, crossing='first' "
    "gives the first loss of detection, and a wider receiver.ranges locates "
    "the outermost crossing.")


def _detection_range(r, se, crossing):
    """``(range, shadow)``: :func:`detection_range` of validated 1-D arrays,
    and ``(outermost range with data, shadow start, shadow end)`` when the
    range is ``inf`` although SE goes negative inside the grid (else
    ``None``), for the caller to report."""
    if crossing == 'first':
        annuli = detection_annuli(r, signal_excess_dB=se)
        if not annuli:
            return np.nan, None
        first = annuli[0]
        known = np.isfinite(se)
        if first[0] > float(r[known][0]):
            # The nearest sampled range is already in shadow: there is no
            # first-loss range from the source outward.
            return np.nan, None
        if len(annuli) == 1 and first[1] == float(r[known][-1]) \
                and se[known][-1] >= 0.0:
            return np.inf, None
        return first[1], None
    # NaN marks a cell the propagation model never filled (no ray reached it),
    # not a cell where the target is undetectable, so the crossing is sought
    # among the sampled ranges only — the same no-data handling as
    # :meth:`Field.max`.
    known = np.isfinite(se)
    if not known.any():
        return np.nan, None
    idx = np.where(known)[0]
    full_r, full_se = r, se
    r, se = r[idx], se[idx]
    positive = se >= 0.0
    if positive.all():
        return np.inf, None
    if not positive.any():
        return np.nan, None
    # Largest range with SE >= 0 is the outermost positive sample — this
    # captures a far-edge recovery (e.g. a convergence zone giving +,-,+).
    last_pos = int(np.where(positive)[0][-1])
    if last_pos == r.size - 1:
        # SE is >= 0 at the far edge, so the outermost crossing lies beyond
        # the grid: inf, as for SE >= 0 everywhere. ``positive.all()`` above
        # took that case, so reaching here means SE went negative inside the
        # grid and came back — a shadow zone the inf alone does not show.
        # The first in-grid shadow: before the first detectable interval when
        # the nearest sample is already negative, else between the first two.
        annuli = detection_annuli(full_r, signal_excess_dB=full_se)
        if annuli[0][0] > float(r[0]):
            shadow_start, shadow_end = float(r[0]), annuli[0][0]
        else:
            shadow_start, shadow_end = annuli[0][1], annuli[1][0]
        return np.inf, (float(r[-1]), shadow_start, shadow_end)
    # ``idx`` keeps each sample's position in the unmasked array: a step > 1
    # between consecutive known samples is a no-data hole, and a crossing
    # inside it has no modeled location to interpolate.
    if idx[last_pos + 1] - idx[last_pos] > 1:
        return float(r[last_pos]), None
    se0, se1 = se[last_pos], se[last_pos + 1]
    frac = se0 / (se0 - se1)
    return float(r[last_pos] + frac * (r[last_pos + 1] - r[last_pos])), None
