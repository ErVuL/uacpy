"""The carrier and geometry checks of stages 1-2 that read no model state:
the carrier types, where receivers sit against the resolvable depth and the
local seafloor, how far the range-dependent axes reach, the speed bounds and
the modelled depth of an environment, the ``.irc`` layout of a
``'precalc'`` seabed, and the ``'precalc'`` sea surface no Acoustics Toolbox
binary reads."""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from uacpy._log import log_message
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.environment import Environment
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, UnsupportedFeatureError,
    ValidityWarning,
)
from uacpy.core.receiver import Receiver
from uacpy.core.source import Source


def check_carrier_types(model_name: str, env, source, receiver, *,
                        allow_none_receiver: bool = False) -> None:
    """Raise :class:`ConfigurationError` unless ``run()``'s three
    positional carriers are an :class:`Environment`, a :class:`Source`
    and a :class:`Receiver` (subclasses accepted), in that order.

    Without it a swapped pair — ``run(source, env, receiver)`` —
    surfaces as a raw ``AttributeError`` deep inside deck assembly, far
    from the call that caused it. ``allow_none_receiver`` admits
    ``receiver=None`` for an engine with a mode whose contract accepts it
    (``spec.traits.none_receiver_modes``).
    """
    wrong = [
        f"{name}={type(value).__name__}"
        for name, expected, value in (
            ('env', Environment, env),
            ('source', Source, source),
            ('receiver', Receiver, receiver),
        )
        if not isinstance(value, expected)
        and not (allow_none_receiver
                 and name == 'receiver' and value is None)
    ]
    if wrong:
        raise ConfigurationError(
            f"{model_name}.run takes (env: Environment, "
            f"source: Source, receiver: Receiver, ...) in that "
            f"order; got {', '.join(wrong)}."
        )


def warn_receiver_below_resolvable(
    model_name: str, env: 'Environment', receiver: 'Receiver',
    resolvable_depth: float,
) -> None:
    """Flat-bathymetry counterpart to
    :func:`check_per_range_receiver_depth`: warn — never raise — when a
    receiver lies below the depth this model resolves the field at.
    Such receivers are accepted; what comes back is per-engine:
    Scooter, SPARC and RAM return NaN there (their solvers clamp the
    receiver onto the domain or stop meshing, so no field is evaluated
    at the asked depth); Bellhop returns NaN on its TL, BROADBAND and
    TIME_SERIES routes, an empty cell on ARRIVALS, and on RAYS /
    EIGENRAYS labels the rays it found at the clamped depth with the
    requested one; Kraken and the OASES models compute a physical
    transmitted / evanescent field through the sediment they mesh. The
    range-dependent case is handled per-range by
    :func:`check_per_range_receiver_depth`.
    """
    if env.bathymetry.varies_with_range:
        return
    if receiver.depth_max > resolvable_depth:
        warnings.warn(
            f"{model_name}: receiver depth "
            f"{float(receiver.depth_max):.1f} m is below the model's "
            f"resolvable depth ({resolvable_depth:.1f} m). It is "
            f"accepted; the result there reflects the model's "
            f"below-domain behaviour (a physical transmitted / "
            f"evanescent field from Kraken and OASES; NaN from "
            f"Bellhop, Scooter, SPARC and RAM).",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )


def check_per_range_receiver_depth(
    model_name: str, env: 'Environment', receiver: 'Receiver', *,
    paired: bool, verbose=False,
) -> None:
    """Report receivers below the local seafloor in a range-dependent
    bathymetry. They are accepted, not rejected; the cells come back NaN
    from the engines that evaluate no field there (Scooter, SPARC, RAM, and
    Bellhop's TL / broadband routes — its ARRIVALS cells come back empty,
    its RAYS / EIGENRAYS carry whatever reached the clamped depth) and as a
    physical transmitted field from the ones that mesh the sediment
    (Kraken, OASES). The flat-bathy case is handled by
    :func:`warn_receiver_below_resolvable`.

    A full rectangular (depth x range) grid over a sloping seafloor always
    puts some of its cells under the bed: that is how such a field is drawn,
    so it is an info log line (``verbose``). A ``ValidityWarning`` is kept
    for what no grid explains: a paired deck's own receiver point below its
    seafloor, or a range column with every depth below it.

    Which (depth, range) pairs the deck actually carries is ``paired``
    (the model's ``_receiver_grid_is_paired``).
    """
    if not env.bathymetry.varies_with_range:
        return
    depths = np.atleast_1d(receiver.depths).astype(float)
    ranges = np.atleast_1d(receiver.ranges).astype(float)
    seafloor = np.asarray(env.bathymetry.eval(range=ranges), dtype=float)

    paired = paired and depths.size == ranges.size
    if paired:
        # Paired deck: depths[i] is evaluated only at ranges[i], so the
        # Cartesian product below would report (depth, range) pairs that
        # carry no receiver at all.
        grid_depths, grid_ranges, grid_floors = depths, ranges, seafloor
    else:
        shape = (depths.size, ranges.size)
        grid_depths = np.broadcast_to(depths[:, None], shape)
        grid_ranges = np.broadcast_to(ranges[None, :], shape)
        grid_floors = np.broadcast_to(seafloor[None, :], shape)

    mask = grid_depths > grid_floors
    if not np.any(mask):
        return
    flat = int(np.argmax(mask))
    r = float(grid_ranges.ravel()[flat])
    z = float(grid_depths.ravel()[flat])
    sf = float(grid_floors.ravel()[flat])
    n_below = int(np.count_nonzero(mask))
    message = (
        f"{model_name}: {n_below} receiver point(s) sit below the local "
        f"seafloor, the first at (range={r:.1f} m, depth={z:.1f} m) "
        f"under a {sf:.1f} m seafloor. Results there reflect "
        f"{model_name}'s below-bottom behaviour, not the water column.")
    if paired:
        warnings.warn(message, ValidityWarning,
                      skip_file_prefixes=USER_FRAME_SKIP)
        return
    buried = np.all(mask, axis=0)
    if np.any(buried):
        first = int(np.argmax(buried))
        warnings.warn(
            f"{model_name}: {int(np.count_nonzero(buried))} receiver "
            f"range(s) lie entirely below the local seafloor, the first at "
            f"range={ranges[first]:.1f} m, where the shallowest receiver "
            f"({depths.min():.1f} m) is under a {seafloor[first]:.1f} m "
            f"seafloor. No water-column field is computed there; results "
            f"reflect {model_name}'s below-bottom behaviour.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)
    else:
        log_message(model_name, message, verbose=verbose, level='info')


def warn_on_ssp_start(model_name: str, env: 'Environment') -> None:
    """``FallbackWarning`` when the sound-speed profile starts below the
    sea surface. Every engine holds the shallowest row up to z = 0 — the AT
    and OASES writers prepend it, Bellhop's ``.env`` too, RAM's decks hold
    it — and the field equals the one of the profile written from z = 0
    (measured on Kraken, Bellhop, RAM mpirams / ramgeo and OAST)."""
    z_top = float(env.ssp.depths[0])
    if z_top > 0.0:
        speeds = np.asarray(env.ssp.sound_speed, dtype=float)[0]
        warnings.warn(
            f"{model_name}: the sound-speed profile starts at {z_top:g} m, "
            f"not at the sea surface; its shallowest row "
            f"({np.min(speeds):g}"
            + (f"-{np.max(speeds):g}" if np.ptp(speeds) > 0 else "")
            + f" m/s) is held up to z = 0. Supply a sample at 0 m to "
              f"control the near-surface water.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )


def warn_on_range_coverage(
    model_name: str, env: 'Environment', receiver: 'Receiver',
) -> None:
    """Emit one ``FallbackWarning`` per range-dependent axis whose extent
    falls short of ``receiver.range_max``, and one per axis that starts
    past the source at r = 0. Constant extrapolation is what every
    downstream writer / interpolator does in both cases; this surfaces it
    instead of leaving it silent.
    """
    r_target = float(receiver.range_max)
    if r_target <= 0:
        return

    def _check(axis_name: str, axis_max: float) -> None:
        if axis_max < r_target:
            warnings.warn(
                f"{model_name}: {axis_name} extent "
                f"({axis_max:.1f} m) is shorter than receiver.range_max "
                f"({r_target:.1f} m); values beyond {axis_max:.1f} m are "
                f"constant-extrapolated from the last sample.",
                FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

    def _check_start(axis_name: str, axis_min: float) -> None:
        if axis_min > 0.0:
            warnings.warn(
                f"{model_name}: {axis_name} starts at {axis_min:.1f} m, "
                f"past the source at r = 0; values before {axis_min:.1f} m "
                f"are constant-extrapolated from the first sample.",
                FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

    if env.bathymetry.varies_with_range:
        _check_start("env.bathymetry", float(env.bathymetry.ranges[0]))
        _check("env.bathymetry", float(env.bathymetry.ranges[-1]))
    if env.ssp.is_range_dependent:
        _check_start("env.ssp.ranges", float(env.ssp.ranges[0]))
        _check("env.ssp.ranges", float(env.ssp.ranges[-1]))
    if env.bottom.is_range_dependent:
        _check_start("env.bottom.ranges", float(env.bottom.ranges[0]))
        _check("env.bottom.ranges", float(env.bottom.ranges[-1]))
    if env.altimetry is not None and env.altimetry.n_ranges > 1:
        _check_start("env.altimetry ranges", float(env.altimetry.ranges[0]))
        _check("env.altimetry ranges", float(env.altimetry.ranges[-1]))


def speed_bounds(env: 'Environment'):
    """Slowest / fastest compressional speeds (m/s) anywhere in ``env``.

    Water column plus every bottom layer and half-space that carries
    geoacoustics. An Environment always carries an SSP (``ssp=None``
    resolves to the isovelocity default), so the list is never empty.
    """
    # Every profile, not just the one at r=0: a range-dependent SSP can
    # hold its extremes in any column.
    speeds = [float(c) for c in np.asarray(env.ssp.sound_speed).ravel()
              if np.isfinite(c)]
    speeds.extend(c for c in env.bottom.all_sound_speeds() if c)
    return float(min(speeds)), float(max(speeds))


def total_media_depth(env: 'Environment') -> float:
    """
    Deepest modelled interface (m): the water column plus all sediment
    layer thicknesses. Below this lies the semi-infinite halfspace.

    Full-waveguide spectral solvers (Scooter, SPARC) resolve the field
    through the water and every fluid/elastic sediment layer, so the
    valid receiver range extends to this depth — not merely to
    ``env.depth`` (the seafloor).
    """
    depth = float(env.depth)
    if env.bottom.is_layered:
        depth += env.bottom.total_thickness_max()
    return depth


def reject_malformed_irc_bottom(model_name: str, env: 'Environment') -> None:
    """Refuse a ``'precalc'`` seabed whose table is not in ``.irc`` layout.

    A ``'precalc'`` bottom is staged verbatim as ``<root>.irc``, which
    ``misc/RefCoef.f90:94-107`` reads as: line 1 ``Title freq``, line 2
    the record count ``NkTab``, then ``NkTab`` records written
    ``(5G15.7,I5)`` — tangential wavenumber ``x``, the complex impedance
    pair ``f`` / ``g``, and a power-of-ten exponent: five reals and an
    integer per record. That is a different format from the
    ``.brc`` / ``.trc`` angle tables (a bare count line, then
    ``theta |R| phase`` rows), and the binary answers the mismatch with a
    bare Fortran I/O abort. Checked on the wrappers that stage the file
    (Kraken, Scooter) before any binary is launched; a missing or unset
    ``reflection_file`` is left to the staging step's own typed errors.
    """
    def _numeric(token: str) -> bool:
        try:
            float(token.replace('D', 'E').replace('d', 'e'))
        except ValueError:
            return False
        return True

    def _reason(lines) -> Optional[str]:
        if len(lines) < 3:
            return (f"only {len(lines)} non-blank line(s); the layout is "
                    f"a Title/freq line, a record count, and the records")
        head = lines[0].split()
        if all(_numeric(t) for t in head):
            return ("line 1 is all-numeric — an .irc starts with "
                    "'Title freq', while a bare record count or a "
                    "theta/|R|/phase row starts a .brc/.trc angle table")
        if not _numeric(head[-1]):
            return ("line 1 carries no trailing frequency — an .irc "
                    "starts with 'Title freq'")
        count = lines[1].split()
        if len(count) != 1 or not count[0].lstrip('+-').isdigit() \
                or int(count[0]) < 1:
            return (f"line 2 is {lines[1].strip()!r} where the .irc "
                    f"record count NkTab (a single positive integer) "
                    f"belongs")
        record = lines[2].split()
        if len(record) < 6 or not all(_numeric(t) for t in record[:6]):
            return (f"the first record {lines[2].strip()!r} does not "
                    f"carry the (5G15.7,I5) x/f/g/iPow fields — a "
                    f"3-column row is a theta/|R|/phase angle table")
        return None

    for column in env.bottom.columns:
        hs = column.halfspace
        if hs.acoustic_type != 'precalc':
            continue
        table = getattr(hs, 'reflection_file', None)
        if not table or not Path(table).exists():
            continue
        with open(table, 'r', errors='replace') as fh:
            lines = [ln for ln in (fh.readline() for _ in range(64))
                     if ln.strip()][:3]
        reason = _reason(lines)
        if reason:
            raise ConfigurationError(
                f"{model_name}: acoustic_type='precalc' stages "
                f"{table} as the .irc internal-reflection table, but "
                f"{reason}. An .irc (BOUNCE's f/g impedance format, "
                f"misc/RefCoef.f90:94-107) is not a .brc/.trc "
                f"angle-magnitude-phase table; the binary aborts with a "
                f"bare Fortran backtrace on the mismatch.",
                remediation=(
                    "Pass a BOUNCE result.metadata['irc_file'] as "
                    "reflection_file=, or use acoustic_type='file' for "
                    "a theta/|R|/phase angle table (.brc/.trc)."
                ),
            )


def reject_precalc_surface(env, *, model_name) -> None:
    """A ``'precalc'`` sea surface has no reader in the Acoustics Toolbox.

    ``misc/RefCoef.f90:92`` reads an ``.irc`` only for ``BotRC == 'P'`` —
    there is no top branch — so ``TopOpt(2)='P'`` leaves ``xTab``/``fTab``
    unpopulated. ``kraken.exe`` stops on it at ``Kraken/kraken.f90:47-48``;
    ``krakenc.exe`` and ``scooter.exe`` (``Scooter/scooter.f90:357-358``) run
    ``InterpolateIRC`` over the empty table and die with SIGSEGV.
    """
    if any(p.acoustic_type == 'precalc' for p in env.surface.nodes):
        raise UnsupportedFeatureError(
            model_name,
            "a 'precalc' (.irc) sea surface — the Acoustics Toolbox reads "
            "an internal reflection coefficient for the bottom only "
            "(misc/RefCoef.f90:92), so the top table is never loaded",
            alternatives=[
                "acoustic_type='file' with the .brc/.trc table BOUNCE also "
                "writes",
                'Bellhop',
            ],
        )
