"""The checks that read no model state: the signature every model's own
``run`` keeps (checked when the class is defined), the carrier and geometry
checks of stages 1-2 — the carrier types, where receivers sit against the
resolvable depth and the local seafloor, how far the range-dependent axes
reach, the speed bounds and the modelled depth of an environment, the
``.irc`` layout of a ``'precalc'`` seabed, the ``'precalc'`` sea surface no
Acoustics Toolbox binary reads — and the pulse pair a ``TIME_SERIES`` call
supplies."""

import inspect
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from uacpy._log import log_message
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.environment import Environment
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, InvalidDepthError,
    UnsupportedFeatureError, ValidityWarning,
)
from uacpy.core.receiver import Receiver
from uacpy.core.run_settings import RunMode
from uacpy.core.source import Source
from uacpy.models._registry import engines_supporting
from uacpy.models._band import _real_pulse


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
    Scooter and SPARC return NaN there (their solvers clamp the
    receiver onto the domain or stop meshing, so no field is evaluated
    at the asked depth); RAM returns its PE field down to the top of its
    absorbing layer and NaN inside it; Bellhop returns NaN on its TL,
    BROADBAND and TIME_SERIES routes, an empty cell on ARRIVALS, and on
    RAYS / EIGENRAYS labels the rays it found at the clamped depth with the
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
            f"below-domain behaviour, which docs/guide/source-receiver.md "
            f"lists per engine (a physical transmitted / evanescent field "
            f"or NaN).",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )


def warn_receivers_within_a_wavelength(
    model_name: str, env: 'Environment', frequencies,
    receiver: 'Receiver',
) -> None:
    """``ValidityWarning`` when every receiver range lies within one
    wavelength of the source, at the shortest wavelength the run carries:
    the highest of the run's ``frequencies`` (Hz, its band) in the slowest
    water. The measure is the horizontal range, the ``k r`` of the far-field
    forms below; a receiver far below the source at a short range is
    inside it too.

    Each model sums a far-field form: the mode sum's Hankel asymptote
    (Jensen et al., *Computational Ocean Acoustics*, sect. 5.2.1, eq. 5.14,
    "using the asymptotic approximation to the Hankel function"; its
    large-argument form holds "for k_r r ≫ 1", sect. 2.4.3.3), a ray's
    spreading, a parabolic equation's outgoing wave. Inside one wavelength
    none holds, whatever the units were meant to be; ranges typed in
    kilometres (0.5-5 for 0.5-5 km), or a frequency typed in kHz, land
    there, so the warning names both units. A grid reaching past one
    wavelength, a genuine short-range run included, is not flagged."""
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
    freqs = freqs[np.isfinite(freqs) & (freqs > 0.0)]
    if not freqs.size:
        return
    water = np.asarray(env.ssp.sound_speed, dtype=float)
    water = water[np.isfinite(water) & (water > 0.0)]
    if not water.size:
        return
    wavelength = float(water.min() / freqs.max())
    farthest = float(receiver.range_max)
    if not 0.0 < farthest < wavelength:
        return
    warnings.warn(
        f"{model_name}: every receiver range lies within one wavelength of "
        f"the source (the farthest at {farthest:g} m, the wavelength "
        f"{wavelength:.3g} m at {float(freqs.max()):g} Hz in "
        f"{float(water.min()):g} m/s water), where the far-field sums the "
        f"models evaluate do not hold. Ranges are in metres: "
        f"ranges=np.linspace(0.5, 5, 10) is 0.5-5 m, and kilometres are "
        f"written 1000 times larger; frequencies are in Hz, and kilohertz "
        f"are written 1000 times larger too.",
        ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )


def check_per_range_receiver_depth(
    model_name: str, env: 'Environment', receiver: 'Receiver', *,
    paired: bool, verbose=False,
) -> None:
    """Report receivers below the local seafloor in a range-dependent
    bathymetry. They are accepted, not rejected; the cells come back NaN
    from the engines that evaluate no field there (Scooter, SPARC, and
    Bellhop's TL / broadband routes — its ARRIVALS cells come back empty,
    its RAYS / EIGENRAYS carry whatever reached the clamped depth), as a
    physical transmitted field from the ones that mesh the sediment
    (Kraken, OASES), and as the PE field from RAM down to the top of its
    absorbing layer. The flat-bathy case is handled by
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


def check_run_signature(cls, name: str, body) -> None:
    """Refuse a model class ``cls`` whose own ``run`` (``body``, declared as
    attribute ``name``) breaks the signature every model shares: the four
    carriers by position (``cls._RUN_POSITIONAL``), every extra as a
    keyword, no ``**kwargs`` sink, and no default of its own for
    ``run_mode``."""
    label = f"{cls.__name__}.{name}()"
    params = list(inspect.signature(body).parameters.values())
    for p in params:
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            raise TypeError(
                f"{label} must not use **kwargs: an unknown "
                "keyword has to raise TypeError, not be swallowed. Declare "
                "the accepted extras as keyword-only after a bare '*'."
            )
        # The mirror of the rule below: a *args sink accepts exactly the
        # unknown positional arguments that rule exists to refuse.
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            raise TypeError(
                f"{label} must not use *args: an unknown positional "
                "argument has to raise TypeError, not be swallowed. The "
                "four carriers are the whole positional contract."
            )

    leading = params[:len(cls._RUN_POSITIONAL)]
    names = tuple(p.name for p in leading)
    if names != cls._RUN_POSITIONAL:
        raise TypeError(
            f"{label} must begin with "
            f"{cls._RUN_POSITIONAL!r}; got {names!r}."
        )
    for p in leading:
        if p.kind is inspect.Parameter.KEYWORD_ONLY:
            raise TypeError(
                f"{label} parameter {p.name!r} must be "
                "positional-or-keyword, not keyword-only."
            )
    # The override hands ``run_mode`` to ``super().run()``, which
    # resolves ``None`` through ``_default_run_mode()``: a default of the
    # override's own would be a second decider of the default mode.
    # ``inspect.Parameter.empty`` (no default at all) decides nothing, so
    # only a non-None default is refused.
    run_mode = leading[-1]
    if run_mode.default not in (None, inspect.Parameter.empty):
        raise TypeError(
            f"{label} declares run_mode={run_mode.default!r}; the "
            f"default must be None — run() resolves it through "
            f"_default_run_mode(), which is the place to change it."
        )

    for p in params[len(cls._RUN_POSITIONAL):]:
        if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            raise TypeError(
                f"{label} parameter {p.name!r} must be "
                "keyword-only (place it after a bare '*') so unknown "
                "positional args cannot reach it."
            )


def validate_geometry(model, env, source, receiver, run_mode=None) -> None:
    """The source/receiver geometry of a call to engine ``model`` checked
    against what it resolves (the default of
    :meth:`~uacpy.models.base.PropagationModel._validate_geometry`): the beam
    pattern it reads or not, a multi-depth source on a mode that stacks
    none, the source depths (``model._check_source_depths``), a source on
    the pressure-release surface, and the receiver grid."""
    if (source.beam_pattern is not None
            and not model._supports_source_beam_pattern):
        raise ConfigurationError(
            f"{model.model_name} does not read a source beam pattern; "
            f"drop Source(beam_pattern=...) or use one of "
            f"{', '.join(engines_supporting('source_beam_pattern'))}."
        )

    # Whether a multi-depth Source is legal is a question about the
    # MODE, not about the model: ``_supports_multi_source_depth`` says
    # the engine batches depths into one deck *somewhere*, which is the
    # declarative capability the matrix prints, and would wave through a
    # mode of that same engine which stacks nothing (Kraken MODES).
    n_depths = len(np.atleast_1d(source.depths))
    if n_depths > 1:
        mode = (run_mode if run_mode is not None
                else model._default_run_mode())
        if not model._stacks_source_depths(mode):
            raise ConfigurationError(
                f"{model.model_name} takes a single source depth per "
                f"{mode.name} run; got {n_depths}: "
                f"{np.asarray(source.depths).tolist()}. A multi-depth "
                f"Source stacks "
                f"only in the field modes (COHERENT_TL / INCOHERENT_TL "
                f"/ SEMICOHERENT_TL / BROADBAND / TIME_SERIES); for "
                f"{mode.name} loop over single-depth Sources externally."
            )

    resolvable_depth = model._check_source_depths(env, source)

    # A source exactly at z = 0 sits on the pressure-release sea surface,
    # where the boundary forces p ≈ 0: a field run then returns a
    # degenerate result (a null / saturated-TL sentinel, or — in RAM — an
    # unphysical normalisation) that a valid-looking ``Source(depths=0)``
    # hides. Reflection coefficients and mode shapes don't propagate a
    # source field, so the warning doesn't apply to those.
    if (run_mode not in (RunMode.REFLECTION, RunMode.MODES)
            and np.any(np.asarray(source.depths) == 0.0)):
        warnings.warn(
            f"{model.model_name}: a source at depth 0 m is on the "
            f"pressure-release sea surface, where the field is ~0 — the "
            f"result is degenerate (null / saturated TL, model-dependent). "
            f"Use a small positive depth (e.g. 1 m).",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

    if receiver is None:
        # A mode with no receiver geometry (Kraken MODES builds its own
        # depth grid): nothing below reads one.
        return

    if receiver.depth_min < 0:
        raise ConfigurationError("Receiver depths must be non-negative.")

    # A point source's field carries a 1/sqrt(r) factor that is singular
    # on the source axis, so a grid whose largest range is 0 m has no
    # cell any engine can fill: every wrapper NaNs the r = 0 column, and
    # the PE grid chooser divides by a range step it sizes from r_max.
    # Modes that evaluate nothing on the receiver range axis are exempt
    # (see ``_NO_RANGE_AXIS_MODES``); a mixed grid such as
    # ``ranges=[0, 1000]`` keeps its NaN column and the masking warning.
    if (run_mode not in model._NO_RANGE_AXIS_MODES
            and receiver.range_max <= 0.0):
        raise ConfigurationError(
            f"{model.model_name}: the receiver's largest range is 0 m; "
            f"give at least one positive receiver range (a field at "
            f"r = 0 is undefined for a point source).",
            remediation="Receiver(depths=<your depths>, ranges=[1000.0])",
        )

    warn_receiver_below_resolvable(model.model_name, env, receiver,
                                   resolvable_depth)
    check_per_range_receiver_depth(
        model.model_name, env, receiver,
        paired=model._receiver_grid_is_paired(receiver),
        verbose=model.verbose)
    warn_on_range_coverage(model.model_name, env, receiver)
    warn_on_ssp_start(model.model_name, env)


def check_source_depths(model, env, source) -> float:
    """Refuse a source depth engine ``model`` cannot resolve; return the
    resolvable depth (``model._max_receiver_depth``), the default of
    :meth:`~uacpy.models.base.PropagationModel._check_source_depths`."""
    resolvable_depth = model._max_receiver_depth(env)
    if np.any(source.depths > resolvable_depth):
        exc = InvalidDepthError(
            float(source.depths.max()), resolvable_depth, "Source",
        )
        note = model._source_below_domain_note(env, resolvable_depth)
        if note:
            # Message enrichment only. ``InvalidDepthError.__reduce__``
            # rebuilds from the three constructor arguments, so a copy
            # that crosses a process boundary (``run_parallel``) carries
            # the base remediation.
            exc.remediation = f"{exc.remediation}.\n\n{note}"
        raise exc
    if np.any(source.depths < 0):
        raise ConfigurationError("Source depths must be non-negative.")
    return resolvable_depth


def reject_unsupported_run_kwargs(model_name: str, **kwargs) -> None:
    """Refuse the optional ``run()`` keywords engine ``model_name`` never
    consumes.

    Every model takes the full keyword set (``frequencies``,
    ``source_waveform``, ``sample_rate``, ``output_duration``,
    ``t_start``) so a polymorphic ``model.run(...)`` never raises
    ``TypeError``. A keyword that no mode of this model reads — the
    waveform keywords on a model without ``TIME_SERIES``, ``frequencies=``
    on one without a broadband path — would be ignored, and ignoring it
    *silently* would hide a caller mistake, so any of these passed a
    non-``None`` value raises here instead.
    ``PropagationModel._run_keywords_never_consumed`` names them; a keyword
    the model reads on another mode is warned about and dropped instead
    (:func:`warn_ignored_run_kwargs`)."""
    supplied = sorted(name for name, value in kwargs.items() if value is not None)
    if supplied:
        raise UnsupportedFeatureError(
            model_name,
            f"run parameter(s): {', '.join(supplied)}",
            alternatives_label='run parameters',
        )


def warn_ignored_run_kwargs(model_name: str, run_mode, reason=None,
                            **named_values) -> None:
    """Warn when the resolved ``run_mode`` of a call to engine
    ``model_name`` will not consume some of the optional ``run()`` keywords
    the caller supplied.

    Complements :func:`reject_unsupported_run_kwargs`: that check is for
    models that *never* consume a keyword (hard error); this one is for
    keywords the model does consume, just not on the resolved run mode's
    path. Only non-``None`` values are reported, in a single
    ``FallbackWarning``. Consuming paths (BROADBAND/TIME_SERIES, …) must not
    call it."""
    ignored = [
        f'{name}=' for name, value in named_values.items()
        if value is not None
    ]
    if not ignored:
        return
    if reason is None:
        reason = 'these apply to BROADBAND/TIME_SERIES only'
    warnings.warn(
        f"{model_name}.run(run_mode="
        f"{getattr(run_mode, 'name', run_mode)}): ignoring "
        f"{', '.join(ignored)} — {reason}.",
        FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )


def require_timeseries_signal(model_name: str, run_mode: 'RunMode',
                              source_waveform, sample_rate) -> None:
    """Refuse an unusable TIME_SERIES signal pair of a call to engine
    ``model_name``; returns nothing.

    Raises :class:`ConfigurationError` when the caller asked for a
    :attr:`RunMode.TIME_SERIES` result but did not supply both
    ``source_waveform`` and ``sample_rate``, when the waveform is not a
    real finite pressure pulse, or when ``sample_rate`` is not a
    positive finite number.

    The waveform rule is the one every waveform entry point applies
    (:func:`~uacpy.models._band._real_pulse`): a real, finite, 1-D array.
    A complex waveform is refused whatever its imaginary part, and a
    ``(time, signal)`` pair or a 2-D array with the pair hint.

    Used by every wrapper that synthesises p(t) from a broadband
    transfer function (Bellhop, RAM, Scooter, Kraken, OASP), and by
    SPARC for the series a ``pulse_type`` opening with ``'F'`` / ``'B'``
    stages as ``STSFIL``.
    """
    if run_mode == RunMode.TIME_SERIES and (
        source_waveform is None or sample_rate is None
    ):
        raise ConfigurationError(
            f"{model_name}.run(run_mode=TIME_SERIES) requires "
            f"source_waveform and sample_rate. For the broadband "
            f"transfer function H(f), use run_mode=RunMode.BROADBAND."
        )
    if run_mode == RunMode.TIME_SERIES and sample_rate is not None:
        # Refuse a sample rate that is not a positive finite number. The
        # test is NaN-closed (``not (sr > 0)`` rather than ``sr <= 0``)
        # because nan compares False both ways and would slip through:
        # nothing downstream stops it either — pad_waveform_to_duration
        # and time_series_band (models/_band.py) no-op on a
        # non-positive rate — so it surfaces as a ZeroDivisionError in
        # the delay-and-sum, a raw ValueError from the deep guard in
        # acoustic_signal/_synthesis.py (``waveform_synthesis_setup``),
        # or a trace on a descending time axis.
        try:
            sr = float(sample_rate)
        except (TypeError, ValueError):
            sr = float('nan')
        if not np.isfinite(sr) or not (sr > 0.0):
            raise ConfigurationError(
                f"{model_name}.run(run_mode=TIME_SERIES): "
                f"sample_rate must be a positive finite number of Hz; "
                f"got {sample_rate!r}."
            )
    if run_mode == RunMode.TIME_SERIES and source_waveform is not None:
        _, problem = _real_pulse(source_waveform)
        if problem is not None:
            raise ConfigurationError(
                f"{model_name}.run(run_mode=TIME_SERIES): {problem}")


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
