"""The refusals of a Bellhop run: constructor knobs no run could use, and
carriers whose deck the binary cannot run or would answer wrongly without a
diagnostic — a source on or outside a boundary, a ``'precalc'`` boundary, a
beam type whose influence routine lacks the run type or misindexes the
receiver grid, a ray box that cuts the receivers, a source beam pattern
narrower than the launch fan."""

import warnings
from pathlib import Path

import numpy as np

from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, UnsupportedFeatureError,
)
from uacpy.core.run_settings import RunMode
from uacpy.io.bellhop_writer import (
    BELLHOP_RAY_BOX_FACTOR, CERVENY_DEFAULTS, validate_beam_shape,
    validate_beam_type,
)
from uacpy.core._validate import equally_spaced
from uacpy.io.refl_io import read_source_beam_pattern
from uacpy.models.bellhop._plan import _STEP_PER_DEPTH
from uacpy.models.bellhop._tables import (
    _BEAM_TYPE_REFUSED_RUN_TYPES, _BEAM_TYPE_RUN_TYPES, _INFLUENCE_ROUTINE,
    _IRREGULAR_GRID_BEAM_TYPES, _RUN_MODE_TO_INFLUENCE_LETTER,
    _UNIFORM_RANGE_BEAM_TYPES, grid_is_paired,
)


def check_knobs(*, beam_type, beam_width_type, beam_curvature, component,
                n_beams, grid_type, launch_angles, ray_step, z_box, r_box, n_freqs,
                bandwidth_factor) -> None:
    """Refuse a constructor knob no run could use: an unknown beam type
    or beam shape letter, a beam count that is not a whole number >= 0,
    a grid type other than 'R'/'I', a launch fan that is not an
    ascending pair of numbers, a negative or non-finite step, a ray box
    that is not a positive number, a band of fewer than two bins or a
    non-positive width.

    Run at construction and again by every run (stage 2), since the
    attributes can be reassigned in between.
    """
    validate_beam_type(beam_type, 'Bellhop')
    validate_beam_shape(beam_width_type, beam_curvature,
                        component, 'Bellhop')
    if n_beams is not None and (
            isinstance(n_beams, bool)
            or not isinstance(n_beams, (int, np.integer))):
        raise ConfigurationError(
            f"Bellhop(n_beams={n_beams!r}) must be an integer beam "
            f"count. The deck writer emits int(n_beams), so a "
            f"fractional value would silently truncate toward zero."
        )
    if n_beams is not None and n_beams < 0:
        raise ConfigurationError(
            f"Bellhop(n_beams={n_beams}) must be >= 0. angleMod.f90:38 "
            f"auto-estimates the beam count only on an exact 0; a "
            f"negative value reaches ALLOCATE unchecked."
        )
    if grid_type not in ('R', 'I'):
        raise ConfigurationError(
            f"Bellhop(grid_type={grid_type!r}) is not valid. Use "
            f"'R' (rectilinear) or 'I' (irregular paired depth/range)."
        )
    if not isinstance(launch_angles, (tuple, list, np.ndarray)) or len(launch_angles) != 2:
        raise ConfigurationError(
            f"Bellhop(launch_angles={launch_angles!r}) must be a 2-element sequence "
            f"(min_deg, max_deg) of launch-angle limits."
        )
    try:
        alpha_lo, alpha_hi = float(launch_angles[0]), float(launch_angles[1])
    except (TypeError, ValueError):
        raise ConfigurationError(
            f"Bellhop(launch_angles={launch_angles!r}) entries must be numbers "
            f"(min_deg, max_deg) of launch-angle limits."
        ) from None
    if not (alpha_lo < alpha_hi):
        raise ConfigurationError(
            f"Bellhop(launch_angles={launch_angles!r}) limits must satisfy "
            f"min_deg < max_deg. SubTab (misc/subtabulate.f90:41-45) "
            f"fills the fan uniformly from alpha(1) to alpha(2); a "
            f"reversed pair traces a negative-width fan."
        )
    if not np.isfinite(ray_step) or ray_step < 0:
        raise ConfigurationError(
            f"Bellhop(ray_step={ray_step!r}) must be >= 0 and finite. 0 defers "
            f"to the automatic step (env.depth / "
            f"{_STEP_PER_DEPTH:g}); "
            f"the deck writes the value as the ray-marching step size "
            f"in meters."
        )
    for name, value in (('z_box', z_box), ('r_box', r_box)):
        if value is None:
            continue
        if (isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer,
                                          np.floating))
                or not np.isfinite(value) or value <= 0):
            raise ConfigurationError(
                f"Bellhop({name}={value!r}) must be a positive number of "
                f"metres, or None for 1.2 x the extent the rays must "
                f"cover. bellhop.f90:571-572 drops a ray once "
                f"ABS(x) exceeds the box, so a box at or below zero "
                f"drops every ray and the run returns an all-NaN field "
                f"at exit 0."
            )
    if n_freqs < 2:
        raise ConfigurationError(
            f"Bellhop(n_freqs={n_freqs}) cannot span a frequency "
            f"band: the grid a BROADBAND run expands a single centre "
            f"frequency to needs at least its two edges.",
            remediation=("Use n_freqs >= 2, or pass frequencies=[fc] "
                         "to run a single bin."),
        )
    if not (np.isfinite(bandwidth_factor)
            and bandwidth_factor > 0):
        raise ConfigurationError(
            f"Bellhop(bandwidth_factor={bandwidth_factor:g}) must "
            f"be a positive number: the band a BROADBAND run expands a "
            f"single centre frequency to is fc*(1 +/- "
            f"bandwidth_factor/2).",
            remediation="Use bandwidth_factor > 0 (default 0.5).",
        )


def check_component(*, component, beam_type) -> None:
    """``component`` is a Cerveny **ray-centred** knob only.

    ``Beam%Component`` has exactly one use site in the solver:
    ``influence.f90:120-130``, inside ``InfluenceCervenyRayCen`` —
    ``beam_type='R'``. ``InfluenceCervenyCart`` (``'C'``,
    ``influence.f90:157-289``) never reads it, and no geometric routine
    does either, while the writer still emits the letter and the ``.prt``
    echoes it back: the run looks configured and returns pressure.

    Where the letter *is* honoured the ``.shd`` holds particle velocity
    (m/s), which the :class:`~uacpy.core.results.Field` contract has no
    ``kind`` for — its pressure conventions (the point/line ``_shd_phase``
    correction, ``phase_reference='travelling_wave'``, ``unit='Pa'``,
    ``.dB`` as transmission loss) would all be applied to it and every
    one of them would be wrong. So that pairing is refused rather than
    mislabelled.
    """
    component = str(component).upper()
    if component == 'P':
        return
    if beam_type.upper() != 'R':
        warnings.warn(
            f"Bellhop(component={component!r}) is ignored for "
            f"beam_type={beam_type!r}: Beam%Component is read only by "
            f"InfluenceCervenyRayCen (influence.f90:120-130), the "
            f"beam_type='R' routine. The letter still reaches the deck and "
            f"the .prt echoes it, but the field returned is pressure.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
        return
    raise UnsupportedFeatureError(
        model_name='Bellhop',
        feature=(
            f"component={component!r} on beam_type='R' — the .shd "
            f"would then hold particle velocity (m/s), and uacpy's Field "
            f"carries no such kind: the result would report unit 'Pa', "
            f"apply the pressure point/line phase correction and read its "
            f".dB as transmission loss"
        ),
        alternatives=[
            "Bellhop(component='P') for the pressure field",
            "Derive the particle velocity from the pressure field "
            "(v = -grad(p) / (i·omega·rho))",
        ],
    )


def warn_on_ignored_cerveny_knobs(*, beam_type, beam_width_type,
                                  beam_curvature, eps_multiplier, r_loop,
                                  n_image, ib_win) -> None:
    """The Cerveny beam knobs are written only for ``beam_type`` in
    {'C', 'R'} (ReadEnvironmentBell.f90). Warn if any is set to a
    non-default value while a non-Cerveny beam is selected, since it
    would otherwise be silently ignored.

    ``component`` is narrower still — ray-centred Cerveny only — and is
    handled by :func:`check_component` on every beam type.
    """
    if beam_type.upper() in ('C', 'R'):
        return
    # ``component`` is handled by check_component on every beam type.
    knobs = {'beam_width_type': beam_width_type,
             'beam_curvature': beam_curvature,
             'eps_multiplier': eps_multiplier, 'r_loop': r_loop,
             'n_image': n_image, 'ib_win': ib_win}
    ignored = [name for name, default in CERVENY_DEFAULTS.items()
               if name != 'component' and knobs[name] != default]
    if ignored:
        warnings.warn(
            f"Bellhop: Cerveny beam knobs {', '.join(ignored)} are "
            f"ignored for beam_type={beam_type!r} (they apply only "
            f"to Cerveny beams, beam_type 'C' or 'R').",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )


def reject_source_outside_the_medium(env, source) -> None:
    """Reject a source on or below the seabed at its own range.

    ``bellhop.f90:488-492`` (``TraceRay2D``) tests ``DistBegTop <= 0 .OR.
    DistBegBot <= 0`` and, when either holds, sets ``Beam%Nsteps = 1``,
    prints *"Terminating the ray trace because the source is on or outside
    the boundaries"* and returns — *"source must be within the medium"*.
    Every beam is then one point long, so the run exits 0 with an all-NaN
    field, zero arrivals and 1-point rays, and warns about none of it.
    The test is ``<= 0``, so **on** the boundary already counts as outside.

    Measured against the seafloor at **r = 0**, not ``env.depth``:
    ``bellhop.f90:237`` launches from ``xs = [0.0, Pos%sz(is)]``, so on a
    sloping bottom a source can be buried at its own range while sitting
    well above ``env.depth`` — a check against ``env.depth`` misses it.

    Bellhop-only. Kraken, Scooter and RAM all return finite, continuous
    fields for a source at or below the seabed (measured -96.8, -58.6 and
    -101.2 dB on the same environment), so this must not go on the shared
    funnel.
    """
    seafloor = float(np.asarray(env.bathymetry.eval(range=0.0)).flat[0])
    zs = np.atleast_1d(np.asarray(source.depths, dtype=float))
    if np.any(zs >= seafloor):
        raise ConfigurationError(
            f"Bellhop: source depth {float(zs.max()):g} m is at or below "
            f"the seafloor at the source range (r = 0), which is "
            f"{seafloor:g} m. Bellhop terminates every ray at step 1 when "
            f"the source is on or outside a boundary "
            f"(Bellhop/bellhop.f90:488-492, 'source must be within the "
            f"medium'), so the run would return an all-NaN field at exit 0.",
            remediation="Move the source above the seafloor, or use "
                        "Kraken / Scooter / RAM, which resolve a source at "
                        "or below the seabed.")

    # Top half of the same Fortran test: bellhop.f90:488-492 rejects
    # DistBegTop <= 0 symmetrically with the bottom. The top boundary at
    # the launch range r = 0 sits at z = 0 for a flat surface and at
    # z = -height(0) with altimetry (``bellhop_writer`` writes the
    # .ati depth column as -heights: env.altimetry is positive-up,
    # the .ati positive-down), so a source at or above it has
    # every ray terminated at step 1 and the run returns an all-NaN
    # field at exit 0.
    top = 0.0
    if env.altimetry is not None:
        top = -float(np.asarray(env.altimetry.eval(range=0.0)).flat[0])
    if np.any(zs <= top):
        raise ConfigurationError(
            f"Bellhop: source depth {float(zs.min()):g} m is at or above "
            f"the top boundary at the source range (r = 0), which sits "
            f"at z = {top:g} m. Bellhop terminates every ray at step 1 "
            f"when the source is on or outside a boundary "
            f"(Bellhop/bellhop.f90:488-492, 'source must be within the "
            f"medium'), so the run would return an all-NaN field at "
            f"exit 0.",
            remediation="Move the source below the sea surface (below "
                        "the altimetry crest at r = 0, if any).")


def reject_precalc_boundary(env, *,
                            include_bottom: bool = True) -> None:
    """A ``'precalc'`` boundary has no reflection branch in BELLHOP.

    ``ReadEnvironmentBell.f90:459`` accepts the ``'P'`` option letter and
    prints "reading PRECALCULATED IRC", but ``bellhop.f90:681``'s
    ``SELECT CASE ( HS%BC )`` implements only ``'R'``, ``'V'``, ``'F'``
    and ``'A'``/``'G'`` — there is no ``'P'`` case, so the run fails with
    a bare exit code instead of naming the boundary. ``bounce/__init__.py``'s
    module docstring already records this; the guard belongs here,
    where the deck is built.
    KRAKENC and SCOOTER do read ``.irc``.

    ``include_bottom=False`` checks the surface alone, for a run whose
    seabed goes through BOUNCE (which refuses a ``'precalc'`` seabed
    itself) rather than onto the deck.
    """
    for boundary, where in ((env.bottom, 'bottom'), (env.surface, 'surface')):
        if boundary is None or (where == 'bottom' and not include_bottom):
            continue
        props = getattr(boundary, 'nodes', None) or getattr(
            boundary, 'columns', None) or []
        # A Surface entry is a BoundaryProperties; a Bottom entry is a
        # SeabedColumn whose acoustic_type lives on ``.halfspace``.
        types = {getattr(getattr(p, 'halfspace', p), 'acoustic_type', None)
                 for p in props}
        types.add(getattr(boundary, 'acoustic_type', None))
        if 'precalc' in types:
            raise UnsupportedFeatureError(
                'Bellhop',
                f"a {where} with acoustic_type='precalc' — "
                f"ReadEnvironmentBell.f90:459 accepts the 'P' option but "
                f"bellhop.f90:681 has no 'P' reflection branch, so the run "
                f"fails without naming the cause. Use acoustic_type='file' "
                f"with a .brc table instead",
                ['KrakenC', 'Scooter'])


def check_beam_type_supports_run_mode(run_mode, *, beam_type) -> None:
    """Reject ``beam_type`` × ``run_mode`` pairs the influence routine cannot
    run — see :data:`_BEAM_TYPE_RUN_TYPES` for the enumeration and its
    authority. Untrapped, the arrivals/eigenray pairs corrupt the heap
    (``bellhop.exe`` aborts with SIGABRT) and ``beam_type='S'`` with an
    incoherent run returns a field that is mostly NaN.

    Also reject the pairs :data:`_BEAM_TYPE_REFUSED_RUN_TYPES` lists: a
    Cerveny beam with INCOHERENT_TL / SEMICOHERENT_TL runs, but returns
    a level that falls with the beam count."""
    letter = _RUN_MODE_TO_INFLUENCE_LETTER.get(run_mode)
    if letter is None:
        return
    refused = _BEAM_TYPE_REFUSED_RUN_TYPES.get(beam_type, frozenset())
    if letter in refused:
        raise ConfigurationError(
            f"Bellhop(beam_type={beam_type!r}) refuses "
            f"{run_mode.name}: {_INFLUENCE_ROUTINE[beam_type]} "
            f"squares each beam's contribution (influence.f90:140 "
            f"ray-centred, :282 Cartesian), and ScalePressure takes the "
            f"root of the sum (:779) and scales it by "
            f"const = -Dalpha*SQRT(freq)/c (:772, :774), which is linear "
            f"in the beam spacing. The returned level therefore falls as "
            f"n_beams**-0.5, 4.8 dB per tripling of the beam count "
            f"(measured +10.0 / +14.6 / +19.4 dB against Kraken at "
            f"500 / 1500 / 4500 beams), so it is no transmission loss.",
            remediation=(
                f"Use a geometric beam, beam_type='G', 'B' or 'g', for "
                f"{run_mode.name}; keep beam_type={beam_type!r} for "
                f"COHERENT_TL."),
        )
    if letter in _BEAM_TYPE_RUN_TYPES[beam_type]:
        return
    usable = sorted(
        mode.name for mode, code in _RUN_MODE_TO_INFLUENCE_LETTER.items()
        if code in _BEAM_TYPE_RUN_TYPES[beam_type]
        and code not in refused)
    raise ConfigurationError(
        f"Bellhop(beam_type={beam_type!r}) cannot run "
        f"{run_mode.name}: {_INFLUENCE_ROUTINE[beam_type]} "
        f"(Bellhop/influence.f90) has no RunType(1:1)=='{letter}' branch, "
        f"so the run either corrupts the pressure matrix or returns NaN.",
        remediation=f"Use beam_type='G' or 'B' for {run_mode.name}, or "
                    f"keep beam_type={beam_type!r} and one of "
                    f"{usable}.",
    )


def check_beam_type_supports_receiver_grid(receiver, *, beam_type,
                                           grid_type) -> None:
    """Reject receiver grids the influence routine indexes incorrectly — see
    :data:`_IRREGULAR_GRID_BEAM_TYPES` and
    :data:`_UNIFORM_RANGE_BEAM_TYPES`. Both failures are silent and
    plausible-looking: up to 30 dB off with no NaN and no warning."""
    if (grid_is_paired(grid_type)
            and beam_type not in _IRREGULAR_GRID_BEAM_TYPES):
        raise ConfigurationError(
            f"Bellhop(grid_type='I', beam_type={beam_type!r}) would "
            f"evaluate every paired receiver at receiver.depths[0]: "
            f"bellhop.f90:202-204 pins NRz_per_range to 1 for an irregular "
            f"grid and {_INFLUENCE_ROUTINE[beam_type]} indexes the "
            f"depth by the depth-loop counter.",
            remediation="Use beam_type='G' or 'B' with grid_type='I', or "
                        "grid_type='R' for a rectilinear grid.",
        )
    ranges = np.atleast_1d(receiver.ranges)
    if beam_type in _UNIFORM_RANGE_BEAM_TYPES and ranges.size == 1:
        raise ConfigurationError(
            f"Bellhop(beam_type={beam_type!r}) cannot use a single "
            f"receiver range: {_INFLUENCE_ROUTINE[beam_type]} clamps "
            f"the receiver index to Pos%NRr (influence.f90:339,351), so "
            f"irA == irB at every step and influence.f90:354 skips the "
            f"whole ray — the run exits 0 with an all-NaN field, zero "
            f"eigenrays and zero arrivals.",
            remediation="Use beam_type='G' or 'B', which walk the range "
                        "index with a bracket test, or give "
                        "receiver.ranges at least two equally spaced "
                        "entries.",
        )
    if (beam_type in _UNIFORM_RANGE_BEAM_TYPES and ranges.size > 2
            and not equally_spaced(np.asarray(ranges, dtype=float))):
        raise ConfigurationError(
            f"Bellhop(beam_type={beam_type!r}) requires equally "
            f"spaced receiver.ranges: {_INFLUENCE_ROUTINE[beam_type]} "
            f"forms the range index by dividing by Pos%Delta_r, which "
            f"SourceReceiverPositions.f90:160 sets from the last gap "
            f"alone ({float(ranges[-1] - ranges[-2]):.6g} m here against a "
            f"first gap of {float(ranges[1] - ranges[0]):.6g} m).",
            remediation="Use np.linspace for receiver.ranges, or "
                        "beam_type='G', 'B' or 'S', which take an "
                        "arbitrary range vector.",
        )


def reject_unequal_paired_grid(receiver, *, grid_type) -> None:
    """Refuse a paired grid (``grid_type='I'``) whose depth and range
    lists differ in length."""
    # Irregular receiver grid ('I' in RunType position 5) requires the
    # receiver.depths and receiver.ranges arrays to have the same
    # length (they are paired point-by-point, after BELLHOP sorts each
    # list).  Rectilinear ('R') takes the Cartesian product.  Catch the
    # mismatch here so users see a clear error instead of a confusing
    # Bellhop .prt message.
    if (
        grid_is_paired(grid_type)
        and len(receiver.depths) != len(receiver.ranges)
    ):
        raise ConfigurationError(
            f"Bellhop grid_type='I' (irregular) requires "
            f"len(receiver.depths) == len(receiver.ranges); got "
            f"{len(receiver.depths)} depths and "
            f"{len(receiver.ranges)} ranges. BELLHOP sorts both lists "
            f"before pairing (SourceReceiverPositions.f90:224), so an 'I' "
            f"grid is always a monotone diagonal, shallow-near to "
            f"deep-far. Use grid_type='R' for a rectilinear "
            f"(Cartesian-product) grid — and Field.at for arbitrary "
            f"points — or rebuild the Receiver with matched arrays."
        )


def check_beam_pattern_spans_the_fan(pattern, *, launch_angles) -> None:
    """Require the beam pattern to cover every launch angle in ``launch_angles``.

    ``bellhop.f90:269-270`` clamps the table index but **not** the
    interpolation weight at ``:273``, so ``Amp0`` at ``:274`` extrapolates
    past both ends of the table. ``misc/beampattern.f90:59`` has already
    converted the levels to linear amplitude by then, so extrapolating a
    roll-off drives ``Amp0`` through zero and negative: the outermost beams
    are launched louder than the pattern declares and phase-inverted, and the
    field comes back partly NaN with no warning from any backend.
    ``third_party/MODIFICATIONS.md`` records that this site is deliberately
    left unclamped so the Fortran, C++ and CUDA backends stay identical,
    which makes this the only available guard.
    """
    if isinstance(pattern, (str, Path)):
        angles = read_source_beam_pattern(pattern)[:, 0]
    else:
        angles = np.asarray(pattern, dtype=float)[:, 0]
    lo, hi = float(np.min(angles)), float(np.max(angles))
    fan_lo, fan_hi = float(min(launch_angles)), float(max(launch_angles))
    if lo > fan_lo + 1e-9 or hi < fan_hi - 1e-9:
        raise ConfigurationError(
            f"Bellhop: the source beam pattern spans "
            f"[{lo:g}, {hi:g}]° but the launch fan launch_angles spans "
            f"[{fan_lo:g}, {fan_hi:g}]°. Bellhop extrapolates the pattern "
            f"past its ends on linear amplitude "
            f"(Bellhop/bellhop.f90:273), which inverts the amplitude of "
            f"the outer beams and returns a partly-NaN field with no "
            f"warning.",
            remediation=(
                f"Extend the pattern to cover [{fan_lo:g}, {fan_hi:g}]° — "
                f"repeat the edge level to hold it flat — or narrow "
                f"launch_angles= to the pattern's own span."
            ),
        )


def reject_ray_box_inside_the_domain(env, receiver, mode, z_box,
                                     r_box) -> None:
    """Refuse a ray box that cuts rays the receivers need.

    ``bellhop.f90:571-572`` stops a ray once ``ABS(x(2)) > Box%z`` or
    ``ABS(x(1)) > Box%r``. A ``z_box`` at or above the deepest seafloor
    drops every path that reaches it (measured on a 100 m Pekeris at
    500 Hz: ``z_box=50`` reads 16-21 dB low at a 60 m receiver, with no
    warning), and an ``r_box`` short of the farthest receiver returns
    NaN beyond it. A RAYS run has no receiver to reach, so its box, in
    depth and in range, is the trace extent the caller chose."""
    if mode == RunMode.RAYS:
        return
    seafloor = float(env.depth)
    if z_box <= seafloor:
        raise ConfigurationError(
            f"Bellhop(z_box={z_box:g}) does not reach below the seafloor, "
            f"which sits at {seafloor:g} m: bellhop.f90:571-572 drops a "
            f"ray once its depth exceeds z_box, so every path that "
            f"reflects off the bottom deeper than {z_box:g} m is lost "
            f"and the field comes back tens of dB low with no warning.",
            remediation=(f"Leave z_box=None "
                         f"({BELLHOP_RAY_BOX_FACTOR:g} x env.depth = "
                         f"{BELLHOP_RAY_BOX_FACTOR * seafloor:g} m), "
                         f"or pin it above "
                         f"{seafloor:g} m."),
        )
    if r_box < receiver.range_max:
        raise ConfigurationError(
            f"Bellhop(r_box={r_box:g}) stops short of the farthest "
            f"receiver at {receiver.range_max:g} m: bellhop.f90:571-572 "
            f"drops a ray once its range exceeds r_box, so every "
            f"receiver beyond it comes back NaN.",
            remediation=(f"Leave r_box=None "
                         f"({BELLHOP_RAY_BOX_FACTOR:g} x "
                         f"receiver.range_max), "
                         f"or pin it at or beyond "
                         f"{receiver.range_max:g} m."),
        )
