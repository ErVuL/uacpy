"""Reading a Kraken ``.mod`` and its ``.prt``: the mode set, the group
speeds krakenc prints, which modes are trapped, and the error text of a run
that found none."""

import dataclasses
import os
import re
import numpy as np
from pathlib import Path
from uacpy.core.boundary import BoundaryType
from uacpy.core.results import MediaTable, Modes
from uacpy.core.exceptions import (
    ConfigurationError, FileFormatError, ModelExecutionError,
)
from uacpy.io.oalib_reader import read_prt
from uacpy.io.modes_reader import read_modes
from uacpy.models._extract import result_kwargs


def below_cutoff_remedy(range_dependent) -> str:
    """The remedy of a frequency with no trapped mode, a sentence: below the
    first modal cut-off the field is carried by the continuous spectrum,
    which Scooter's wavenumber integration holds on a range-independent
    waveguide and RAM's parabolic equation on a ``range_dependent`` one."""
    model = ("a parabolic-equation model (RAM), which also takes the "
             "range dependence" if range_dependent
             else "a wavenumber-integration model (Scooter)")
    return (f"Raise the frequency above the waveguide's first modal cut-off "
            f"— below it the field is carried by the continuous spectrum, "
            f"which needs {model}.")


#: The ERROUT text both mode finders print when ``[cLow, cHigh]`` holds no mode
#: — ``Kraken/kraken.f90:962`` and ``Kraken/krakenc.f90:445``, identical in both
#: (only the 'KRAKEN' / 'KRAKENC' banner differs). Named once because two sites
#: match on it: :func:`zero_mode_stop`, which reads where the run stopped, and
#: ``spec.traits.benign_fortran_fatals``, which lets the run's launch through
#: so the run reads where it stopped rather than raising a bare error.
_NO_MODES_ERROUT = 'No modes for given phase speed interval'

#: What gfortran prints when a binary's zero-mode branch re-OPENs the mode
#: file it already holds open, on the deck's first solve only (``IF ( ifreq
#: == 1 .AND. iProf == 1 )``): kraken.exe with a new RECL
#: (``Kraken/kraken.f90:954-955``), krakenc.exe with STATUS='REPLACE'
#: (``Kraken/krakenc.f90:437-439``), after printing 'No modes OPEN statement'
#: on stdout. The run stops there, before the ERROUT is written.
_ZERO_MODE_REOPEN = ('Cannot change RECL parameter in OPEN statement',
                     'Cannot change STATUS parameter in OPEN statement',
                     'No modes OPEN statement')


def zero_mode_stop(prt_text):
    """Where a modes run stopped on a window holding no mode, from its
    ``.prt``: ``(profile, frequency)``, the 0-based indices of the solve
    that printed the empty-spectrum ERROUT (:data:`_NO_MODES_ERROUT`), or
    ``None`` when the ``.prt`` holds none.

    Both binaries loop over profiles, then frequencies, and stop at the
    first empty solve (``Kraken/kraken.f90:42-56``). ``cLow =`` opens each
    profile's block (``misc/ReadEnvironmentMod.f90:134``) and ``Frequency =``
    each frequency's within it when the deck carries more than one
    (``Kraken/kraken.f90:60-65``, ``Kraken/krakenc.f90:60-65``), so the
    stop is the last of each."""
    if not prt_text or _NO_MODES_ERROUT not in prt_text:
        return None
    blocks = prt_text.split('cLow =')
    profile = max(len(blocks) - 1, 1) - 1
    frequencies = len(re.findall(r'^\s*Frequency\s*=', blocks[-1], re.M))
    return profile, max(frequencies, 1) - 1


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
    if n_modes < 1:
        return None
    tables = [rows for rows in printed_tables(
        read_prt(Path(work_dir) / f'{base_name}.prt')) if rows]
    if not tables:
        return None
    speeds = {mode: vg for mode, _cp, vg in tables[0]
              if 1 <= mode <= n_modes}
    if not speeds:
        return None
    out = np.full(n_modes, np.nan, dtype=float)
    for mode, vg in speeds.items():
        out[mode - 1] = vg
    return out


def build_modes_field(modes, n_modes, source, *, backend_exe=None,
                       group_velocity=None,
                       water_density=None, leaky_modes, model_exe, model_name,
                       provenance):
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
            model_name, provenance, source,
            backend=Path(exe).stem if exe else model_name.lower(),
            frequencies=float(source.frequencies[0]),
            n_modes_requested=n_modes,
            leaky_modes=leaky_modes,
        ),
    )
    if n_modes is not None:
        result = result.first_n(int(n_modes))
    return result


def _prt_lists_no_mode(prt_text) -> bool:
    """Whether a modes run's ``.prt`` says it kept no mode: the empty-spectrum
    ERROUT, or a mode table with no row under it (:func:`printed_tables`),
    which krakenc leaves when it keeps no root at its last mesh (30 m over
    sand at 1 Hz) or inverse iteration failed on every root it found."""
    if not prt_text:
        return False
    return (_NO_MODES_ERROUT in prt_text
            or any(rows == [] for rows in printed_tables(prt_text)))


def printed_tables(prt_text):
    """The mode table a modes run's ``.prt`` prints for each frequency
    block, in order — every profile's blocks on a multi-profile deck: the
    rows ``(mode, phase speed,
    group speed)`` (m/s), ``[]`` for a table with no row, ``None`` for a
    block with no table. The one reading of the table: :func:`_prt_lists_no_mode`,
    :func:`printed_mode_tables` and :func:`read_group_speeds_from_prt` all
    take it from here.

    ``Frequency =`` opens each block when the deck carries more than one
    (``Kraken/kraken.f90:60-65``); else the one block follows the last
    ``cLow =`` (``misc/ReadEnvironmentMod.f90:134``). A table is its header,
    a units line, then the rows ``DO mode = 1, M, MAX(1, M/30)``
    (``kraken.f90:95-103``, ``krakenc.f90:97-105``): the index, Re and Im
    of k, the phase speed ``omega / Re(k)`` to ten digits and the group
    speed, which only krakenc fills (``kraken.f90:815-819``). A ``.prt`` cut
    off mid-write ends in a partial row, whose truncated last number would
    read as a plausible speed; it is dropped."""
    if not prt_text:
        return []
    lines = prt_text.splitlines()
    if not prt_text.endswith('\n'):
        lines = lines[:-1]
    starts = [i for i, line in enumerate(lines)
              if re.match(r'^\s*Frequency\s*=', line)]
    if not starts:
        starts = [max([i for i, line in enumerate(lines)
                       if 'cLow =' in line], default=0)]
    tables = []
    for a, b in zip(starts, starts[1:] + [len(lines)]):
        heads = [i for i in range(a, b)
                 if 'Phase Speed' in lines[i] and 'Group Speed' in lines[i]]
        if not heads:
            tables.append(None)
            continue
        rows = []
        for line in lines[heads[-1] + 2:b]:
            parts = line.split()
            if len(parts) != 5:
                break
            try:
                rows.append((int(parts[0]), float(parts[3]),
                             float(parts[4])))
            except ValueError:
                break
        tables.append(rows)
    return tables


def read_mode_record(mod_base, frequency, profile=1, *, model_name):
    """The :class:`Modes` of one record of the ``.mod`` at ``mod_base`` (the
    deck's base path without a suffix): ``profile`` (1-based) at
    ``frequency`` Hz, or ``None`` when the solver kept no mode there — a
    record of ``M = 0`` (``Kraken/kraken.f90:947-962`` writes one into a
    normal-sized file), or one the reader cannot finish after a ``.prt``
    that says no mode was kept (:func:`_prt_lists_no_mode`; krakenc's
    zero-mode header is 640 bytes, ``Kraken/krakenc.f90:432-446``). Any other
    unreadable file is a :class:`ModelExecutionError` naming its ``.prt``
    (:func:`modes_error_message`)."""
    basename = str(mod_base)
    mod_file = basename + '.mod'
    # A .mod with no bytes at all means the binary died before it opened
    # the file — no header to read, so there is nothing to diagnose from.
    if os.path.exists(mod_file) and os.path.getsize(mod_file) == 0:
        raise ModelExecutionError(
            model_name, return_code=0, stdout=None,
            stderr=modes_error_message(basename))
    try:
        modes = read_modes(basename, frequency=float(frequency),
                           profile=int(profile))
    except (FileFormatError, IndexError) as e:
        if _prt_lists_no_mode(read_prt(basename + '.prt')):
            return None
        raise ModelExecutionError(
            model_name, return_code=0, stdout=None,
            stderr=modes_error_message(basename, original_error=e),
        ) from e
    return modes if modes.n_modes > 0 else None


def halfspace_compressional_speed(env, range_m=0.0):
    """The compressional speed (m/s) of the fluid or elastic half-space
    under the column at ``range_m``, ``None`` for vacuum, rigid or a
    reflection table. Above it a root lies in the continuous spectrum
    (Computational Ocean Acoustics §2.4.5.1; over an elastic bottom "for
    wavenumbers up to k_r = 2πf/c_p", §4.10.1)."""
    halfspace = env.bottom.halfspace_at(range=float(range_m))
    if not BoundaryType.from_string(halfspace.acoustic_type).is_geoacoustic:
        return None
    return float(halfspace.sound_speed)


def fluid_halfspace_speed(env):
    """The compressional speed (m/s) of the fluid half-space under the
    r = 0 column, which a trapped mode's phase speed stays below; ``None``
    for a seabed with no such speed: vacuum, rigid, a reflection table, or
    an elastic half-space (:func:`halfspace_compressional_speed`). The
    ``.mod`` records the same speed for each profile
    (:attr:`~uacpy.core.results.MediaTable.halfspace_sound_speed`)."""
    halfspace = env.bottom.halfspace_at(range=0.0)
    if float(halfspace.shear_speed) > 0.0:
        return None
    return halfspace_compressional_speed(env)


def trapped_phase_speeds(k, freq, c_bottom) -> np.ndarray:
    """The phase speeds (m/s, ascending) of the modes ``k`` at ``freq`` Hz
    that are trapped.

    Computational Ocean Acoustics §2.4.5.1: the characteristic equation has
    real (lossless) roots only for ``k2 < kr < k1``; a root with
    ``kr < k2`` radiates into the bottom, a leaky mode. So a mode is trapped
    when its phase speed ``omega / Re(k)`` is below the half-space's
    compressional speed ``c_bottom`` (over an elastic half-space, too:
    above it lies the continuous spectrum, §4.10.1). With ``c_bottom=None``
    (a seabed with no such speed) every mode counts.

    Each binary's roots are its own: kraken.exe's attenuation is a
    first-order perturbation of a real root (``kraken.f90:766-773``) and
    krakenc's root is exact, so near the line one can sit just below
    ``c_bottom`` where the other sits just above. And on real-arithmetic
    kraken.exe a root above ``c_bottom`` is no leaky mode:
    ``Kraken/BCImpedanceMod.f90:83-89`` (CASE 'A', ``cS <= 0``) forms
    ``gammaP = SQRT( x - omega2 / cP**2 )``, pure imaginary there, and
    ``DBLE( f )`` keeps its real part = 0, leaving ``f = 0, g = rho`` —
    CASE 'R', the RIGID bottom (``:60-63``). Measured: a 5 m / 150 Hz duct
    over 1700 m/s returns cp = 1730.70 m/s against the rigid-bottom
    prediction 1732.05."""
    k = np.atleast_1d(np.asarray(k, dtype=complex))
    with np.errstate(divide='ignore', invalid='ignore'):
        cp = 2.0 * np.pi * float(freq) / np.real(k)
    return trapped_among(cp, c_bottom)


def trapped_among(cp, c_bottom) -> np.ndarray:
    """The phase speeds (m/s, ascending) of ``cp`` that are a trapped
    mode's: positive, finite and below ``c_bottom``
    (:func:`trapped_phase_speeds`)."""
    cp = np.atleast_1d(np.asarray(cp, dtype=float))
    keep = np.isfinite(cp) & (cp > 0.0)
    if c_bottom is not None:
        keep &= cp < float(c_bottom)
    return np.sort(cp[keep])


def record_seabed_speed(modes, env, range_m=0.0):
    """The compressional speed (m/s) of the bottom half-space the mode
    record ``modes`` was solved over: a fluid half-space's as its ``.mod``
    records it (:attr:`~uacpy.core.results.MediaTable.halfspace_sound_speed`,
    ``Kraken/kraken.f90:602-605``), else an elastic half-space's from the
    environment ``env`` the deck was written from, at the profile's
    ``range_m`` (m, :func:`halfspace_compressional_speed`); ``None`` for a
    vacuum, rigid or tabulated seabed."""
    media = modes.media
    speed = None if media is None else media.halfspace_sound_speed
    if speed is None:
        return halfspace_compressional_speed(env, range_m)
    return float(speed)


def printed_mode_tables(prt_text, n_frequencies):
    """The mode table a single-profile deck's ``.prt`` prints for each of
    its ``n_frequencies`` (:func:`printed_tables`): ``(complete, cp)``, the
    phase speeds ``omega / Re(k)`` (m/s) of its rows and whether they are
    every mode of that frequency — every mode when the rows number 1, 2,
    ... in turn, every ``M/30``-th one otherwise — or ``None`` where the
    table is not found, or every entry ``None`` when the blocks found are
    not ``n_frequencies``."""
    tables = printed_tables(prt_text)
    if len(tables) != n_frequencies:
        return [None] * n_frequencies
    return [None if rows is None else (
        [mode for mode, _cp, _vg in rows] == list(range(1, len(rows) + 1))
        and bool(rows), np.array([cp for _mode, cp, _vg in rows], dtype=float))
        for rows in tables]


def non_trapped_phase_speeds(k, env, freq, *, leaky_modes):
    """``(cp, c_bottom)`` for the modes ``k`` at ``freq`` Hz, or ``None``.

    ``cp = omega / Re(k)`` is each mode's phase speed; ``c_bottom`` is the
    half-space compressional speed above which a mode radiates into the
    seabed instead of being trapped in the duct.

    ``None`` means the comparison says nothing and no caller should act on
    it: no modes to measure; a non-geoacoustic boundary (vacuum, rigid, or
    a reflection table), which has no half-space to leak into and resolves
    to an unbounded ``c_high`` anyway (:func:`resolve_phase_speed_bounds`);
    an elastic half-space, whose water-borne roots between its shear and
    compressional speeds leak shear (Computational Ocean Acoustics
    §4.10.1), and where kraken.exe's own window stops at the shear speed
    (``kraken.f90:205-210``) and krakenc's does not; or
    ``leaky_modes=True``, where the caller asked for exactly these
    modes.
    """
    if leaky_modes:
        return None
    k = np.atleast_1d(np.asarray(k, dtype=complex))
    c_bottom = fluid_halfspace_speed(env)
    if k.size == 0 or c_bottom is None:
        return None
    kr = np.real(k)
    with np.errstate(divide='ignore', invalid='ignore'):
        cp = 2.0 * np.pi * float(freq) / kr
    cp = cp[np.isfinite(cp) & (cp > 0.0)]
    if cp.size == 0:
        return None
    return cp, c_bottom


def log_non_trapped_modes(k, env, freq, *, leaky_modes, log) -> None:
    """Log the modes of a modes solve whose phase speed sits above the
    seabed speed, when there are any.

    That is the ordinary default (14 modes at 200 Hz in 100 m of water, 3
    of them above the 1650 m/s seabed —
    ``docs/models/kraken.md §6.1 "Fourteen modes at 200 Hz"``): the auto
    ``c_high`` sits 5 % past the bottom speed
    (:data:`~uacpy.models._window.C_HIGH_FACTOR`); a set with every mode
    above it holds no trapped mode. ``leaky_modes=True`` asked for exactly
    these modes (krakenc's complex roots, the leaky modes of Computational
    Ocean Acoustics §2.4.5.1), so it logs nothing."""
    probe = non_trapped_phase_speeds(k, env, freq, leaky_modes=leaky_modes)
    if probe is None:
        return
    cp, c_bottom = probe
    above = cp > c_bottom
    if not above.any():
        return
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


def profile_location(env, profile_ranges_m, profile):
    """``(range, seafloor depth)`` (m) of the deck's ``profile`` (0-based)
    of a multi-profile deck whose profiles sit at ``profile_ranges_m``."""
    r = float(profile_ranges_m[min(profile, len(profile_ranges_m) - 1)])
    return r, float(np.asarray(env.bathymetry.eval(range=r)).flat[0])


def profile_without_modes_refusal(profile, env, profile_ranges_m, exe,
                                  frequency, reason, *, model_name
                                  ) -> ConfigurationError:
    """The typed refusal of a multi-profile deck the modes binary stopped
    on, finding no mode in its later ``profile`` (0-based) at ``frequency``
    Hz: the track crosses that profile's cut-off.

    Both binaries end the whole solve at the first such profile: the
    empty-spectrum ERROUT (``kraken.f90:962``, ``krakenc.f90:445``) stops the
    program, so every later profile is missing from the ``.mod``, and
    krakenc writes its empty header over the FIRST profile's records
    (``krakenc.f90:443-444``), after which field.exe returns an all-NaN
    field. Measured: ``leaky_modes=True`` on the ASA wedge (200 m to 0.1 m
    over 4 km, 25 Hz) stopped on a profile at 20 m or less, at every c_high
    from 3400 to 1e5 m/s; to 50 m it solved. The binary names neither the
    profile nor its depth, so they are read here
    (:func:`zero_mode_stop`, :func:`profile_location`); ``reason`` is the
    binary's own words for the stop."""
    r, depth = profile_location(env, profile_ranges_m, profile)
    return ConfigurationError(
        f"{model_name}: {Path(exe).name} found no mode at "
        f"{float(frequency):g} Hz in profile {profile + 1} of "
        f"{len(profile_ranges_m)} (r = {r:g} m, seafloor {depth:g} m) and "
        f"stopped there ({reason}), leaving every later profile out of the "
        f"mode file (and, on krakenc, the first profile's header "
        f"overwritten), so field.exe cannot sum a field.",
        remediation=end_track_or_ram(env, profile_ranges_m, profile))


def end_track_or_ram(env, profile_ranges_m, profile) -> str:
    """The remedy of a segmented solve that stopped on its later
    ``profile`` (0-based): below a profile's first modal cut-off its
    waveguide has no mode to carry the field on (Computational Ocean
    Acoustics §2.4.5), so ending the track — bathymetry and receivers — at
    the last profile that solved, whose depth the deck then ends on (every
    deck carries a profile at its track's end, so a track ended past it
    reaches the stopped one again), or RAM."""
    r, depth = profile_location(env, profile_ranges_m, max(profile - 1, 0))
    return (f"End the track (bathymetry and receivers) at or before "
            f"r = {r:g} m, the last profile that solved at this frequency "
            f"(seafloor {depth:g} m), or use RAM, which takes the track "
            f"whole.")


def modes_error_message(basename, original_error=None):
    """Build error message for invalid mode files, checking .prt for clues.

    One ``.prt`` string carries a diagnosis specific enough to act on: the
    secant root-finder's convergence failure, which names interfacial
    modes. Anything else gets a pointer to the file rather than a guess. (A
    ``.prt`` saying no mode was kept never gets here: :func:`read_mode_record`
    reads that first.)

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
        # Slow/failed root-finding on interfacial (Scholte/Stoneley)
        # modes. misc/RootFinderSecantMod.f90:80,136 sets the message;
        # Kraken/kraken.f90:359,407 and Kraken/krakenc.f90:388 echo it
        # into the .prt behind their own 'Warning in KRAKEN[C] -
        # RootFinderSecant' banner. kraken.htm's remedy is to raise cLow
        # to the minimum p-wave speed so those modes are skipped.
        secant_failure = bool(
            re.search(r'converge\s+in\s+RootFinderSecant',
                      prt_content, re.IGNORECASE)
        )
        if secant_failure:
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
