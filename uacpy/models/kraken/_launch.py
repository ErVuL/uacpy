"""Launching Kraken's decks and reading what field.exe reports: the decks
written per launch, field.exe's completion and fatal markers, and the
checks that a run left the output it was asked for."""

import warnings
import numpy as np
from pathlib import Path
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.receiver import Receiver
from uacpy.core.exceptions import IOWarning, ModelExecutionError
from uacpy.io.oalib_writer import (
    write_multi_profile_env, write_fieldflp, write_kraken_env_file,
)
from uacpy.io.oalib_reader import read_shd_bin, read_prt
from uacpy.io.refl_io import stage_source_beam_pattern
from uacpy.models._launch import attach_prt_tail
from uacpy.models.kraken._settings import KrakenLaunch
from uacpy.models.kraken._segments import segments_at_ranges
from uacpy.models.kraken._extract import deck_source, evaluated_receiver


# Cleared before each launch so a pinned work_dir cannot return an earlier
# run's output: the modes binary writes <base>.mod, field.exe writes <base>.shd.
_KRAKEN_MODES_OUTPUTS = ('.mod',)
_KRAKEN_FIELD_OUTPUTS = ('.shd',)

#: ``field.f90:44`` hard-codes its log name instead of deriving it from the
#: file root, so field.exe's diagnostics land here rather than in
#: ``<base_name>.prt`` and the shared post-run fatal scan never sees them.
_FIELD_PRT_ROOT = 'field'

#: field.exe's marker table, read off ``field.prt`` (lower-cased):
#: ``field.f90:240`` writes the completion line after ``FreqLoop`` closes
#: and before the teardown whose deallocation is the known benign failure;
#: the fatal line is ERROUT's banner (``misc/FatalError.f90:16-24``) or
#: ``EvaluateCM``'s bare ``'Fatal Error: …'`` (``EvaluateCMMod.f90:313-317``).
_FIELD_COMPLETION_MARKER = 'field completed successfully'
_FIELD_FATAL_MARKER = 'fatal error'


# Deck roots: the modes solve of a MODES run, the modes + field pair of a
# field run, and the single-frequency mode count of the broadband cutoff
# search.
_MODES_BASE = 'modes'
_FIELD_BASE = 'kfield'
_PROBE_BASE = 'mcut'
# Root of the finer solve a krakenc launch's mode count is held to.
_CHECK_BASE = 'mcheck'
# Root of the adiabatic pass a coupled launch's field is held to.
_ADIABATIC_BASE = 'kadiab'

# Far-field gap (dB) between field.exe's coupled and adiabatic sums of the
# same .mod above which a coupled run warns. The coupled projection
# (EvaluateCMMod.f90 NewProfile) is not unitary on modes above or just below a
# fluid half-space's speed: kraken.exe normalises their tails with the real
# part of the lossless impedance derivative (kraken.f90 Normalize), and
# field.exe projects them with the lossy complex gamma (CalculateTail). Over
# many short segments it can amplify or annihilate the field. Measured gap
# (coupled minus adiabatic intensity, every receiver beyond half the longest
# range): PE Workshop 4c at the automatic decomposition +27.5 dB (the shelf
# 22 dB above COUPLE07), 401 identical Pekeris profiles 25 m apart -62.1 and
# +74.0 dB; the runs that are right or merely approximate: 4c at 50 / 100 /
# 400 uniform profiles -0.7 / -1.0 / -0.7, the ASA wedge -1.5, the
# corrugated seafloor -2.7, the kraken.md 6.6 shelf +0.6, a third Pekeris
# identity case -3.3. 10 dB sits three times above the largest benign gap
# and well below the smallest pathological one.
_COUPLED_GAP_WARN_DB = 10.0


def far_field_gap_dB(shd_coupled: Path, shd_adiabatic: Path) -> float:
    """``10 log10`` of the coupled over the adiabatic intensity, summed over
    every frequency, depth and receiver beyond half the longest range."""
    totals = []
    for shd in (shd_coupled, shd_adiabatic):
        frequencies = np.atleast_1d(read_shd_bin(str(shd)).frequencies)
        total = 0.0
        for frequency in frequencies:
            field = read_shd_bin(str(shd), frequency=float(frequency))
            ranges = np.asarray(field.receiver_ranges, dtype=float)
            far = ranges >= 0.5 * np.max(ranges)
            total += float(np.nansum(np.abs(field.pressure[..., far]) ** 2))
        totals.append(total)
    return float(10.0 * np.log10(totals[0] / totals[1]))


def tolerate_field_teardown(exc, work_dir, *, model_name) -> bool:
    """Whether a failed field.exe launch is read anyway, saying so: only
    a non-zero teardown status after the field was written. A timeout
    means the run never finished, so there is no output to salvage.
    ``field.f90:240`` writes 'Field completed successfully' before the
    teardown block whose deallocation can fail, so that line
    (:func:`field_reached_completion`) is what separates a benign
    non-zero exit from a run that died mid-field; without it an abort
    would be downgraded to a warning and the truncated ``.shd`` would
    surface as a misleading "malformed file" error."""
    if exc.timed_out or not field_reached_completion(work_dir):
        return False
    warnings.warn(
        f"{model_name}: field.exe exited with non-zero "
        f"status ({exc}); attempting to read the .shd output "
        "anyway (known Fortran cleanup issue).",
        IOWarning,
        skip_file_prefixes=USER_FRAME_SKIP,
    )
    return True


def require_field_shd(work_dir, base_name, *, model_name) -> None:
    """Raise when field.exe left no usable ``.shd`` (missing or
    empty), quoting field.exe's own log — not ``<base_name>.prt``, which
    belongs to the modes binary and shows a successful mode calculation
    (``field.f90:44`` hard-codes 'field.prt')."""
    shd_file = work_dir / f'{base_name}.shd'
    if not shd_file.exists() or shd_file.stat().st_size == 0:
        exc = ModelExecutionError(
            model_name, return_code=0, stdout=None,
            stderr=(
                "field.exe produced no usable .shd file (missing or "
                "empty); check the .prt log at "
                f"{work_dir / (_FIELD_PRT_ROOT + '.prt')}"
            ),
        )
        attach_prt_tail(exc, work_dir, _FIELD_PRT_ROOT)
        raise exc


def field_reached_completion(work_dir) -> bool:
    """Whether field.exe got past its last field write.

    ``field.f90:240`` writes ``'Field completed successfully'`` after
    ``FreqLoop`` closes and before the clean-up block whose deallocation is
    the known benign failure, so the line is present for a teardown-only
    error and absent for a run that died while computing. Like
    :func:`raise_on_field_fatal` this reads ``field.prt``, whose name
    ``field.f90:44`` hard-codes.
    """
    prt = read_prt(Path(work_dir) / f'{_FIELD_PRT_ROOT}.prt')
    return bool(prt) and _FIELD_COMPLETION_MARKER in prt.lower()


def attach_field_prt_path(result, work_dir, *, cleanup) -> None:
    """Attach field.exe's diagnostic log as
    ``result.metadata['field_prt_file']``, iff the scratch survives
    (``cleanup=False``) and the file exists.

    ``field.f90:44`` hard-codes the log name ``field.prt``
    (``_FIELD_PRT_ROOT``), so :meth:`_attach_output_paths` — which keys
    ``prt_file`` on ``<base_name>.prt``, here the modes binary's
    ``kfield.prt`` — never sees it, and its ``primary_files`` tuple
    cannot carry a root that differs from ``base_name``.
    """
    if cleanup:
        return
    field_prt = work_dir / f'{_FIELD_PRT_ROOT}.prt'
    if field_prt.exists():
        result.metadata['field_prt_file'] = str(field_prt)


def raise_on_field_fatal(work_dir, *, model_name) -> None:
    """Raise when field.exe reported a fatal but left a readable ``.shd``.

    ``_raise_on_fortran_fatal`` cannot see either of field.exe's stop
    routes, because both write to ``field.prt`` — ``field.f90:44``
    hard-codes that name — rather than ``<base_name>.prt``:

    * every ERROUT site (11 in ``field.f90``, plus ``ReadModes.f90``,
      ``SourceReceiverPositions.f90`` and ``beampattern.f90``) writes the
      uppercase ``*** FATAL ERROR ***`` banner, the subroutine name and
      the message (``misc/FatalError.f90:16-24``), then stops with
      ``'Fatal Error: Check the print file for details'`` on stderr and an
      exit status of 0;
    * ``EvaluateCM``'s depth-grid check is a bare
      ``WRITE( *, * ) 'Fatal Error: …'`` followed by an argument-less
      ``STOP`` (``EvaluateCMMod.f90:313-317``), so it carries no banner at
      all.

    The scan is therefore case-insensitive on ``fatal error`` and covers
    both. field.exe has by then already created the ``.shd`` and written
    its header, so the file is present and non-empty but holds no field.
    Left alone it reaches the reader, whose header-count guard reports a
    size mismatch — a message that describes the stub and hides the
    diagnosis.
    """
    prt = read_prt(Path(work_dir) / f'{_FIELD_PRT_ROOT}.prt')
    if not prt:
        return
    marker = prt.lower().rfind(_FIELD_FATAL_MARKER)
    if marker < 0:
        return
    detail = ' '.join(
        line.strip() for line in prt[marker:].splitlines() if line.strip()
    )
    exc = ModelExecutionError(
        model_name, return_code=0, stdout=None,
        stderr=f"field.exe stopped — {detail}",
    )
    attach_prt_tail(exc, work_dir, _FIELD_PRT_ROOT)
    raise exc


def require_band_axis(work_dir: Path, base: str, band, *, model_name) -> None:
    """Refuse a band's ``.shd`` whose frequency axis does not match the
    request: a sub-cutoff (zero-mode) frequency corrupts the
    multi-frequency ``.mod``, and field.exe sometimes produces a ``.shd``
    anyway but with a garbage (e.g. all-zero) frequency axis. The error
    lets :meth:`_launch` recover the propagating sub-band instead of
    returning a zero field."""
    freqs_read = np.asarray(
        read_shd_bin(str(work_dir / f'{base}.shd')).frequencies, dtype=float)
    if (len(freqs_read) == len(band)
            and np.allclose(np.sort(freqs_read),
                            np.sort(np.asarray(band, float)),
                            rtol=1e-3, atol=1e-6)):
        return
    exc = ModelExecutionError(
        model_name, return_code=0, stdout=None,
        stderr=("field.exe returned a frequency axis that does "
                "not match the request — the modes file is "
                "corrupted by a sub-cutoff (zero-mode) frequency."))
    # The modes run's log, which names the frequency with no
    # modes; field.exe itself reports nothing wrong here.
    attach_prt_tail(exc, work_dir, base)
    raise exc


def write_modes_deck(deck: Path, env, source, receiver,
                      launch: KrakenLaunch, *, interp_ssp) -> None:
    """Write the modes ``.env`` of ``launch`` with the io writers: the
    single-profile deck (:func:`write_kraken_env_file`), or the
    multi-profile one (:func:`write_multi_profile_env`) over the
    segments at the recorded profile ranges. The mode tabulation grid
    is the deck's receiver-depth line; with no ``receiver`` (MODES) the
    depth array is written alone, RMax being the launch's."""
    tabulation = (np.array(launch.tabulation_depths) if receiver is None
                  else Receiver(depths=np.array(launch.tabulation_depths),
                                ranges=receiver.ranges))
    if launch.profile_ranges_m is None:
        write_kraken_env_file(
            deck, env, source, tabulation,
            interp_ssp=interp_ssp,
            frequencies=(None if launch.marched_frequencies is None
                         else np.array(launch.marched_frequencies)),
            n_mesh=launch.n_mesh,
            rmax_m=launch.rmax_m,
            c_low=launch.c_low, c_high=launch.c_high[0],
        )
        return
    write_multi_profile_env(
        filepath=deck,
        segments=segments_at_ranges(env, launch.profile_ranges_m),
        source=source,
        receiver=tabulation,
        interp_ssp=interp_ssp,
        n_mesh=launch.n_mesh,
        c_low=launch.c_low,
        c_high=launch.c_high,
        rmax_m=launch.rmax_m,
    )


def write_decks(inputs, launch: KrakenLaunch, *, interp_ssp, log) -> Path:
    """The modes ``.env`` of ``launch`` and, on a field run, the
    field.exe ``.flp`` and the staged ``.sbp`` beam pattern. Returns the
    path of the modes deck."""
    source = deck_source(inputs, launch)
    base = _MODES_BASE if launch.field_option is None else _FIELD_BASE
    deck = inputs.work_dir / f'{base}.env'
    log(f"Writing environment file: {deck}")
    write_modes_deck(deck, inputs.env, source, inputs.receiver,
                     launch, interp_ssp=interp_ssp)
    if launch.field_option is None:
        return deck
    receiver = evaluated_receiver(inputs)
    pos = {
        's': {'z': source.depths},
        'r': {'z': receiver.depths, 'r': receiver.ranges},
    }
    flp_kwargs = dict(
        title=getattr(inputs.env, 'name', ''),
        n_profiles=launch.n_profiles,
        profile_ranges=(None if launch.profile_ranges_m is None
                          else np.array(launch.profile_ranges_m)),
    )
    if inputs.settings.engine.n_modes is not None:
        flp_kwargs['n_modes'] = int(inputs.settings.engine.n_modes)
    write_fieldflp(inputs.work_dir / f'{base}.flp', launch.field_option,
                   pos, **flp_kwargs)
    # field.exe reads <base>.sbp when Opt(3:3)='*'.
    if source.beam_pattern is not None:
        stage_source_beam_pattern(
            source.beam_pattern, inputs.work_dir / f'{base}.sbp')
    return deck
