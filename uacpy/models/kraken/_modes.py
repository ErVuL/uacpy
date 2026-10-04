"""Reading a Kraken ``.mod`` and its ``.prt``: the mode set, the group
speeds krakenc prints, the checks that the modes are trapped, and the error
text of a run that found none."""

import dataclasses
import os
import re
import warnings
import numpy as np
from pathlib import Path
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.boundary import BoundaryType
from uacpy.core.results import MediaTable, Modes
from uacpy.core.exceptions import (
    FileFormatError, ModelExecutionError, ValidityWarning,
)
from uacpy.io.oalib_reader import read_prt
from uacpy.io.modes_reader import read_modes


#: Remedy shared by the two ways a below-cutoff run reaches the caller: the
#: binary's own empty-spectrum ERROUT (:func:`modes_error_message`) and
#: a mode set in which every mode is non-trapped
#: (:func:`check_non_trapped_modes`). Phrased as a sentence fragment so
#: both sites can splice it into their own opening clause.
_BELOW_CUTOFF_REMEDY = (
    "raise the frequency above the waveguide's modal cutoff — below it the "
    "field is carried by the continuous spectrum, which needs a "
    "wavenumber-integration model (Scooter)"
)

#: The ERROUT text both mode finders print when ``[cLow, cHigh]`` holds no mode
#: — ``Kraken/kraken.f90:962`` and ``Kraken/krakenc.f90:445``, identical in both
#: (only the 'KRAKEN' / 'KRAKENC' banner differs). Named once because two sites
#: match on it: :func:`modes_error_message`, which turns it into the
#: typed no-modes error, and ``spec.traits.benign_fortran_fatals``, which lets the
#: broadband floor search read it as "below cutoff" instead of a run failure.
_NO_MODES_ERROUT = 'No modes for given phase speed interval'


def read_group_speeds_from_prt(work_dir, base_name, n_modes, *, fills_vg):
    """Group speeds (m/s) from the modes run's ``.prt``, or ``None``.

    Both KRAKEN and KRAKENC print a five-column table — mode index,
    Re k, Im k, phase speed, group speed — but only KRAKENC fills the
    last column (``krakenc.f90:819``). KRAKEN's assignment is commented
    out at ``kraken.f90:815-819``, so it prints an ``ALLOCATABLE`` that was
    never written to.

    ``fills_vg`` therefore selects on the BINARY, not on the values.
    The zeros KRAKEN prints today are an uninitialised array reading
    back a freshly zeroed page — undefined behaviour, not a guarantee.
    Rebuilt with ``-finit-real=nan``, or with the
    ``DEALLOCATE``/``ALLOCATE`` at ``kraken.f90:939-940`` landing on
    recycled heap, that column becomes arbitrary, and an "all zeros
    means unfilled" rule would hand garbage back as a group speed.

    ``kraken.f90:101`` prints with stride ``MAX(1, M/30)``, so a run with
    more than 30 modes lists only about 30 of them. The returned array is
    full length with ``NaN`` in the gaps, read off the printed mode index
    rather than by position, so a strided table still lands correctly.
    """
    if not fills_vg:
        return None
    prt = Path(work_dir) / f'{base_name}.prt'
    text = read_prt(prt)
    if not text or n_modes < 1:
        return None
    # A .prt cut off mid-write ends in a partial row whose last field
    # is a truncated number; parsing it fabricated a plausible speed.
    lines = text.splitlines()
    if lines and not text.endswith('\n'):
        lines = lines[:-1]
    try:
        head = next(i for i, L in enumerate(lines) if 'Group Speed' in L)
    except StopIteration:
        return None

    speeds = {}
    for line in lines[head + 1:]:
        parts = line.split()
        if len(parts) != 5:
            if speeds:
                break          # table ended
            continue           # still in the units header
        try:
            mode = int(parts[0])
            vg = float(parts[4])
        except ValueError:
            if speeds:
                break
            continue
        if 1 <= mode <= n_modes:
            speeds[mode] = vg

    if not speeds:
        return None
    out = np.full(n_modes, np.nan, dtype=float)
    for mode, vg in speeds.items():
        out[mode - 1] = vg
    return out


def build_modes_field(modes, n_modes, source, *, backend_exe=None,
                       group_velocity=None,
                       water_density=None, leaky_modes, model_exe, model_name,
                       result_kwargs):
    """Restamp the :class:`Modes` :func:`~uacpy.io.read_modes` returned
    as this run's result.

    Returns the full mode set the reader produced; callers cap the
    count via :meth:`Modes.first_n` if they passed an ``n_modes``
    request. ``backend_exe`` records which modes binary ran
    (kraken.exe vs krakenc.exe); defaults to the resolved kraken.exe.
    ``water_density`` (g/cm³) is the density the modes
    were normalised against in the water column; it is recorded in the
    ``.mod``'s media table (:attr:`Modes.media`), so
    :meth:`Modes.modal_pressure_field` divides by the ``ρ(z_s)`` the
    run's own field divides by, for a source in the water or in a
    sediment medium.
    """
    exe = backend_exe or model_exe
    media = modes.media
    if water_density is not None:
        media = (MediaTable(water_density=float(water_density))
                 if media is None else dataclasses.replace(
                     media, water_density=float(water_density)))
    result = Modes(
        k=modes.k,
        phi=modes.phi,
        depths=modes.depths,
        group_velocity=group_velocity,
        media=media,
        **result_kwargs(
            source,
            backend=Path(exe).stem if exe else model_name.lower(),
            frequencies=float(source.frequencies[0]),
            n_modes_requested=n_modes,
            leaky_modes=leaky_modes,
        ),
    )
    if n_modes is not None:
        result = result.first_n(int(n_modes))
    return result


def read_modes_file(filepath: Path, *, model_name) -> Modes:
    """Read the Kraken ``.mod`` at ``<filepath>.mod`` as a :class:`Modes`;
    ``filepath`` is the deck's base path without a suffix, which
    :func:`~uacpy.io.read_modes` takes (it appends its own '.mod')."""
    basename = str(filepath)
    mod_file = basename + '.mod'

    # A .mod with no bytes at all means the binary died before it opened
    # the file — no header to read, so there is nothing to diagnose from.
    if os.path.exists(mod_file) and os.path.getsize(mod_file) == 0:
        raise ModelExecutionError(
            model_name, return_code=0, stdout=None,
            stderr=modes_error_message(basename),
        )

    # The MODES path writes a single-frequency ``.mod``; broadband goes
    # through ``field.exe`` instead. ``freq=0.0`` selects the only bin
    # present, closest-frequency matching.
    # A short or unreadable .mod arrives as FileFormatError:
    # ``read_modes`` wraps every parse failure (its ``except
    # PARSE_ERRORS`` arm, which already contains IndexError). The empty
    # spectrum is one of those on krakenc — see the M == 0 note below —
    # so the diagnosis has to happen on this branch too, not only on the
    # mode-count one.
    try:
        modes_data = read_modes(basename, frequency=0.0)
    except (FileFormatError, IndexError) as e:
        raise ModelExecutionError(
            model_name, return_code=0, stdout=None,
            stderr=modes_error_message(basename, original_error=e),
        ) from e

    # "No modes for given phase speed interval" is an ERROUT that still
    # leaves a file behind. kraken.f90:947-962 writes records 1 and 7, so
    # its .mod is a normal 4*LRecordLength*7 bytes and only the mode count
    # reports the state — this branch. krakenc.f90:432-446 writes records 1
    # and 5 of the same header, a 640-byte file the reader cannot finish,
    # so that path lands on the FileFormatError above instead.
    if modes_data.n_modes == 0:
        raise ModelExecutionError(
            model_name, return_code=0, stdout=None,
            stderr=modes_error_message(basename),
        )

    return modes_data


def non_trapped_phase_speeds(k, env, freq, *, leaky_modes):
    """``(cp, c_bottom)`` for the modes ``k`` at ``freq`` Hz, or ``None``.

    ``cp = omega / Re(k)`` is each mode's phase speed; ``c_bottom`` is the
    half-space compressional speed above which a mode radiates into the
    seabed instead of being trapped in the duct.

    ``None`` means the comparison says nothing and no caller should act on
    it: no modes to measure; a non-geoacoustic boundary (vacuum, rigid, or
    a reflection table), which has no half-space to leak into and resolves
    to an unbounded ``c_high`` anyway (:func:`resolve_phase_speed_bounds`);
    an elastic half-space, which traps on its *shear* speed instead and is
    the one case ``kraken.f90:209`` really does clamp ``cHigh`` for; or
    ``leaky_modes=True``, where the caller asked for exactly these modes.
    """
    if leaky_modes:
        return None
    k = np.atleast_1d(np.asarray(k, dtype=complex))
    if k.size == 0:
        return None
    halfspace = env.bottom.halfspace_at(range=0.0)
    if not BoundaryType.from_string(halfspace.acoustic_type).is_geoacoustic:
        return None
    if float(halfspace.shear_speed) > 0.0:
        return None
    kr = np.real(k)
    with np.errstate(divide='ignore', invalid='ignore'):
        cp = 2.0 * np.pi * float(freq) / kr
    cp = cp[np.isfinite(cp) & (cp > 0.0)]
    if cp.size == 0:
        return None
    return cp, float(halfspace.sound_speed)


def check_non_trapped_modes(k, env, freq, *, exe=None,
                             field_run=False, leaky_modes, log,
                             model_name) -> None:
    """Act on the modes whose phase speed sits above the seabed speed.

    Every mode non-trapped means the frequency is below the waveguide's
    modal cutoff and the field is carried by the continuous spectrum, so
    the modal sum answers a different problem than the one asked. The
    auto ``c_high`` sits 5 % past the bottom speed
    (:data:`~uacpy.models._window.C_HIGH_FACTOR`), so the mode search still
    finds something there and the binary's own empty-spectrum ERROUT —
    the existing diagnosis in :func:`modes_error_message` — never fires.
    A modes solve raises; a field run warns, because the broadband
    sub-cutoff recovery at :meth:`Kraken._band_with_sub_cutoff_bins` legitimately
    drives single below-cutoff bins through this path and NaN-fills them.

    On real-arithmetic ``kraken.exe`` those eigenvalues are not leaky modes
    at all. ``Kraken/BCImpedanceMod.f90:83-89`` (CASE 'A', ``cS <= 0``)
    forms ``gammaP = SQRT( x - omega2 / cP**2 )``; above the half-space
    speed the radicand is negative, ``gammaP`` is pure imaginary, and
    ``DBLE( f )`` keeps its real part = 0 — leaving ``f = 0, g = rho``,
    which is CASE 'R', the RIGID bottom (``:60-63``). What comes back is a
    rigid-bottom waveguide's spectrum plus the first-order radiation-loss
    perturbation of ``kraken.f90:766-773``. Measured: a 5 m / 150 Hz duct
    over 1700 m/s returns cp = 1730.70 m/s against the rigid-bottom
    prediction 1732.05.

    Some modes non-trapped is the ordinary default (14 modes at 200 Hz in
    100 m of water, 3 of them above the 1650 m/s seabed —
    ``docs/models/kraken.md §6.1 "Fourteen modes at 200 Hz"``), so that
    only logs.
    """
    probe = non_trapped_phase_speeds(k, env, freq, leaky_modes=leaky_modes)
    if probe is None:
        return
    cp, c_bottom = probe
    above = cp > c_bottom
    if not above.any():
        return
    if not above.all():
        log(
            f"{int(above.sum())} of {above.size} modes are non-trapped "
            f"(phase speed above the {c_bottom:.1f} m/s half-space speed, "
            f"up to {cp.max():.2f} m/s): the auto c_high sits 5 % past the "
            f"bottom speed and kraken.f90:212 leaves the acoustic cHigh "
            f"clamp commented out. On kraken.exe they are a rigid-bottom "
            f"solve plus a first-order radiation-loss perturbation, not "
            f"true leaky modes; leaky_modes=True computes those on "
            f"krakenc.exe."
        )
        return

    rigid = (
        " Real-arithmetic kraken.exe cannot represent a leaky mode: above "
        "the half-space speed BCImpedanceMod.f90:83-89 collapses to the "
        "rigid-bottom condition, so these eigenvalues are a rigid-bottom "
        "waveguide's plus the radiation-loss perturbation of "
        "kraken.f90:766-773."
        if exe is not None and Path(exe).stem == 'kraken' else ""
    )
    message = (
        f"Kraken returned {above.size} mode(s) and every one of them is "
        f"non-trapped: the slowest has a phase speed of {cp.min():.2f} m/s, "
        f"above the {c_bottom:.1f} m/s half-space speed, so it radiates "
        f"into the seabed rather than propagating in the duct.{rigid} The "
        f"modal sum is not a physical answer here; "
        f"{_BELOW_CUTOFF_REMEDY}."
    )
    if field_run:
        warnings.warn(
            f"{model_name}: {message}",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
        return
    raise ModelExecutionError(
        model_name, return_code=0, stdout=None, stderr=message,
    )


def refuse_a_profile_without_modes(mod_base, env, profile_ranges_m, exe, *,
                                   model_name) -> None:
    """Raise when the modes binary stopped on a profile of a multi-profile
    deck that holds no mode.

    Both binaries end the whole solve at the first such profile: the
    empty-spectrum ERROUT (``kraken.f90:962``, ``krakenc.f90:445``) stops the
    program, so every later profile is missing from the ``.mod``, and
    krakenc writes its empty header over the FIRST profile's records
    (``krakenc.f90:443-444``), after which field.exe returns an all-NaN
    field. Measured: ``leaky_modes=True`` on the ASA wedge (200 m to 0.1 m
    over 4 km, 25 Hz) stopped on a profile at 20 m or less, at every c_high
    from 3400 to 1e5 m/s; to 50 m it solved. The binary names neither the
    profile nor its depth, so they are read here: one ``cLow =`` line opens
    each profile's block of the ``.prt``."""
    prt = read_prt(Path(f"{mod_base}.prt"))
    if prt is None or _NO_MODES_ERROUT not in prt:
        return
    index = max(prt.count('cLow ='), 1) - 1
    r = float(profile_ranges_m[min(index, len(profile_ranges_m) - 1)])
    depth = float(np.asarray(env.bathymetry.eval(range=r)).flat[0])
    raise ModelExecutionError(
        model_name, return_code=0, stdout=None,
        stderr=(
            f"{Path(exe).name} found no mode in profile {index + 1} of "
            f"{len(profile_ranges_m)} (r = {r:g} m, seafloor {depth:g} m) and "
            f"stopped there, leaving every later profile out of the mode file "
            f"(and, on krakenc, the first profile's header overwritten), so "
            f"field.exe cannot sum a field. The solver found no mode of that "
            f"profile in the phase-speed window at this frequency, as on water "
            f"too shallow to trap one. End the "
            f"track (bathymetry and receivers) before r = {r:g} m, or use "
            f"RAM, which takes the shallow end."))


def check_field_modes_trapped(mod_base, env, source, exe, *, leaky_modes, log,
                              model_name) -> None:
    """Apply :func:`check_non_trapped_modes` to the ``.mod`` field.exe is
    about to sum — the only place a below-cutoff field run is visible,
    since the ``.shd`` it produces is a full, plausible-looking curve.

    Best effort: an unreadable mode file is not a diagnosis. The
    empty-spectrum ``.mod`` is precisely that — a zero-mode record breaks
    the reader's stride (see :func:`read_modes_file`) — and that case is
    already reported by the all-NaN guard further down
    :meth:`Kraken._field_of_launch`, so a read failure here stays quiet.
    """
    try:
        modes_data = read_modes(str(mod_base), frequency=0.0)
    except (FileFormatError, IndexError, OSError) as e:
        log(f"non-trapped-mode check skipped: unreadable mode file "
                  f"({type(e).__name__}: {e})", level="debug")
        return
    check_non_trapped_modes(
        modes_data.k, env,
        float(np.atleast_1d(source.frequencies)[0]),
        exe=exe, field_run=True, leaky_modes=leaky_modes, log=log,
        model_name=model_name)


def modes_error_message(basename, original_error=None):
    """Build error message for invalid mode files, checking .prt for clues.

    Two ``.prt`` strings carry a diagnosis specific enough to act on: the
    empty-spectrum ERROUT, which is a physical statement about
    ``[c_low, c_high]``, and the secant root-finder's convergence
    failure, which names interfacial modes. Anything else gets a pointer
    to the file rather than a guess.

    Deliberately NOT diagnosed here: the elastic seabed.
    ``misc/ReadEnvironmentMod.f90:260-266`` echoes
    ``'ACOUSTO-ELASTIC half-space'`` for every ``HS%BC == 'A'``, which is
    the ordinary FLUID half-space letter as well — a plain Pekeris deck
    prints it — so the marker says nothing about shear. And
    ``backend='krakenc'`` is no remedy here, by construction:
    :meth:`select_backend` sends any environment with shear
    (``env.bottom.is_elastic`` covers layers and half-space alike) to
    krakenc already, or raises when ``backend='kraken'`` is forced on
    one. A run that got here on an elastic seabed WAS krakenc, so being
    told to try it is worse than being told nothing.
    """
    prt_file = basename + '.prt'
    error_msg = "Kraken did not produce valid modes. "
    prt_content = read_prt(prt_file)
    if prt_content is not None:
        # 1. Slow/failed root-finding on interfacial (Scholte/Stoneley)
        #    modes. misc/RootFinderSecantMod.f90:80,136 sets the message;
        #    Kraken/kraken.f90:359,407 and Kraken/krakenc.f90:388 echo it
        #    into the .prt behind their own 'Warning in KRAKEN[C] -
        #    RootFinderSecant' banner. kraken.htm's remedy is to raise cLow
        #    to the minimum p-wave speed so those modes are skipped.
        secant_failure = bool(
            re.search(r'converge\s+in\s+RootFinderSecant',
                      prt_content, re.IGNORECASE)
        )

        # 2. The empty-spectrum ERROUT (Kraken/kraken.f90:962). This is a
        #    physical statement about [cLow, cHigh], not a solver failure,
        #    and it names its own remedy, so it is tested first.
        empty_spectrum = _NO_MODES_ERROUT in prt_content

        if empty_spectrum:
            error_msg += (
                "Kraken found no mode with a phase speed inside "
                "[c_low, c_high] at this frequency. Widen the window "
                "(lower c_low, raise c_high, or leaky_modes=True), or "
                f"{_BELOW_CUTOFF_REMEDY}."
            )
        elif secant_failure:
            error_msg += (
                "Kraken reported 'Failure to converge in "
                "RootFinderSecant': the root finder is converging slowly "
                "to interfacial (Scholte / Stoneley) modes. Set c_low to "
                "the minimum p-wave speed in the problem to exclude those "
                "modes (kraken.htm, Phase Speed Limits), or use "
                "Kraken(backend='krakenc')."
            )
        else:
            error_msg += f"Check the .prt file for details: {prt_file}"
            if original_error:
                error_msg += f". Original error: {original_error}"
    elif original_error:
        error_msg += f"Original error: {original_error}"
    return error_msg
