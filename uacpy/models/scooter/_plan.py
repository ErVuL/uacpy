"""What one Scooter deck is resolved to before it is written: the
phase-speed window, the spectral ``RMax`` and its multiplier, the wavenumber
count ``Nk`` the binary will derive, the memory the Green's-function cube and
the transform reach, and the notices of conditions the run warns about; and
the refusals of a deck the binary cannot run."""

from typing import Optional, Tuple

import numpy as np

from uacpy.core.exceptions import ConfigurationError, FallbackWarning
from uacpy.core.run_settings import RunMode
from uacpy.models._window import (
    check_pinned_window, steep_path_notice as _steep_path_notice,
)
from uacpy.models._budget import memory_budget
from uacpy.core.run_settings import Notice


#: Scooter's ``rmax_factor`` defaults: RMax in units of the
#: farthest receiver range, finer for the syntheses that sum many
#: frequencies (see :func:`resolve_rmax_factor`).
SCOOTER_RMAX_FACTOR_BROADBAND = 3.0
SCOOTER_RMAX_FACTOR_NARROWBAND = 2.0

# Bytes per (nk x nr) element the k->r transform holds at peak: the complex128
# phase array it exponentiates in place (16) plus the complex64 kernel that
# array is cast to (8). Measured against peak RSS at nk x nr = 120000 x 499,
# an estimate built on this bounds the real peak by 1.17x.
_TRANSFORM_BYTES_PER_ELEMENT = 24

# The modes whose deck carries a frequency vector and whose .grn is
# transformed at every frequency at once.
_BROADBAND_MODES = frozenset({RunMode.BROADBAND, RunMode.TIME_SERIES})

# The wavenumber branches the k->r transform integrates (``hankel_transform``'s
# ``spectrum``).
_WAVENUMBER_SPECTRA = ('positive', 'negative', 'both')


def _nk_consequence(nk: int) -> str:
    """What a wavenumber count below 2 costs the run.

    ``Nk <= 0`` allocates ``k( Nk )`` empty (``scooter.f90:73``) and the
    Green's function comes back with no wavenumber samples at all. ``Nk = 1``
    allocates one sample and then divides by ``Nk - 1 = 0`` at
    ``scooter.f90:77`` and again per frequency at ``:125``, so ``Deltak`` and
    the stabilising ``Atten`` (``:129``) are both infinite and every Green's
    function value is NaN. Neither case changes the exit status, so both reach
    the caller as a full-shape result unless something refuses them.
    """
    if nk == 1:
        return (
            "scooter.f90:77 spaces the grid as "
            "Deltak = (kMax - kMin) / (Nk - 1), so a single sample divides by "
            "zero: the binary writes an all-NaN Green's function at exit 0 "
            "and the transformed field is all-NaN."
        )
    return (
        "The wavenumber vector is empty and the Green's function comes back "
        "with no samples at exit 0."
    )


def _deck_nk(rmax_m: float, freq_max: float, c_low: float, c_high: float) -> int:
    """The ``Nk`` ``scooter.f90:69`` will derive from this deck.

    ``Nk = INT( 2000.0 * RMax * ( kMax - kMin ) / pi )`` with ``RMax`` in km
    and ``kMax - kMin = 2*pi*freqVec(Nfreq)*(1/cLow - 1/cHigh)`` from
    ``scooter.f90:67-68``, which reduces to
    ``4000 * RMax_km * freq_max * (1/cLow - 1/cHigh)``.

    The deck's own rounding is applied first — ``write_phase_speed_and_rmax``
    writes RMax as ``%.6f`` km and the phase-speed pair as ``%.1f`` — so this
    reproduces the binary's count rather than an unrounded neighbour of it.
    """
    rmax_km = round(float(rmax_m) / 1000.0, 6)
    cl = round(float(c_low), 1)
    ch = round(float(c_high), 1)
    return int(4000.0 * rmax_km * float(freq_max) * (1.0 / cl - 1.0 / ch))


def check_knobs(*, taper, c_low, c_high, wavenumber_spectrum) -> None:
    """Refuse a constructor knob no run could use: a taper outside
    ``[0, 0.5)``, two pinned phase-speed bounds with ``c_low >= c_high``,
    an unknown ``wavenumber_spectrum``.

    Run at construction and again by every run (the attributes can be
    reassigned in between). A single pinned bound is held to the other
    once a run derives it (:func:`~uacpy.models._window.resolve_window`).
    """
    if not (0.0 <= float(taper) < 0.5):
        raise ConfigurationError(
            f"Scooter: taper is the fraction of the wavenumber span "
            f"rolled off at EACH edge, so it must satisfy "
            f"0 <= taper < 0.5; got {taper!r}."
        )
    check_pinned_window('Scooter', c_low=c_low, c_high=c_high)
    if wavenumber_spectrum not in _WAVENUMBER_SPECTRA:
        raise ConfigurationError(
            f"Invalid wavenumber_spectrum '{wavenumber_spectrum}'. "
            f"Use 'positive', 'negative', or 'both'."
        )


def refuse_nonpositive_range(receiver) -> None:
    """Refuse a receiver with no positive range: the spectral ``RMax`` is
    ``receiver.range_max × rmax_factor``, and ``scooter.f90:69`` sizes
    the wavenumber grid from it."""
    if receiver.range_max <= 0.0:
        raise ConfigurationError(
            f"Scooter requires a positive receiver range: the spectral "
            f"RMax is receiver.range_max × rmax_factor, and "
            f"receiver.range_max = {receiver.range_max:.6g} m would "
            f"write RMax = 0, which scooter.exe rejects with an "
            f"unexplained STOP.",
            remediation="Pass a Receiver with at least one range > 0 m.",
        )


def resolve_rmax_factor(run_mode: RunMode, *,
                            rmax_factor) -> Tuple[float, str]:
    """``(rmax_factor, origin)``: the effective multiplier for this
    run and where it came from.

    ``scooter.exe`` writes only the wavenumber-domain ``.grn``; the k→r
    step is uacpy's :func:`~uacpy.core.acoustics.hankel_transform`, a
    direct trapezoidal-rule DFT (``fieldsco.m:5``), not an FFT. What
    ``RMax`` controls is the wavenumber grid the solver samples:
    ``scooter.f90:69`` sets ``Nk = INT(2000·RMax_km·(kMax−kMin)/π)``, so
    ``Δk ≈ π/(2·RMax_m)`` and both cost (``Nk`` samples of the
    finite-element solve) and resolution scale linearly with the
    multiplier. A uniform-``Δk`` DFT is periodic in range with period
    ``2π/Δk ≈ 4·RMax_m`` at the top frequency, so the wrap-around
    replica also moves out proportionally. ``BROADBAND`` /
    ``TIME_SERIES`` use the finer grid: their syntheses sum many
    frequencies, and an under-resolved ``G(k)`` shows up as trapezoidal
    error in every one of them. User-pinned values win.
    """
    if rmax_factor is not None:
        return float(rmax_factor), 'Scooter(rmax_factor=…)'
    if run_mode in _BROADBAND_MODES:
        return (SCOOTER_RMAX_FACTOR_BROADBAND,
                f'the {run_mode.name} default')
    return (SCOOTER_RMAX_FACTOR_NARROWBAND,
            f'the {run_mode.name} default')


def deck_max_frequency(source, frequencies):
    """``freqVec( Nfreq )`` for the deck ``Scooter._write_input`` writes.

    ``scooter.f90:67-68`` sizes the wavenumber grid from the *last* entry
    of the frequency vector, so this mirrors the writer's own branch:
    ``write_broadband_freqs`` emits the vector only when it holds more
    than one entry, and otherwise the deck carries the single header
    frequency ``write_header`` takes from ``source.frequencies[0]``.
    """
    if frequencies is not None:
        freqs = np.atleast_1d(frequencies)
        if len(freqs) > 1:
            return float(freqs[-1])
    return float(source.frequencies[0])


def refuse_too_few_wavenumbers(nk: int, rmax_m: float, f_deck: float,
                               c_low: float, c_high: float) -> None:
    """Refuse a deck whose wavenumber count ``scooter.f90`` cannot space:
    ``Nk < 2`` runs to completion at exit 0 (:func:`_nk_consequence`)."""
    if nk >= 2:
        return
    raise ConfigurationError(
        f"This deck asks Scooter for Nk = {nk} wavenumber sample(s): "
        f"scooter.f90:69 derives Nk = INT(2000 * RMax_km * "
        f"(kMax - kMin) / pi) from RMax = {rmax_m:g} m at "
        f"{f_deck:.6g} Hz with "
        f"c_low = {c_low:.1f} and c_high = {c_high:.1f} m/s. "
        f"{_nk_consequence(nk)}",
        remediation=(
            "Nk grows with RMax, frequency and the width of the "
            "phase-speed window: raise rmax_factor (RMax = "
            "receiver.ranges.max() x rmax_factor), raise the "
            "source frequency, or widen c_low/c_high. Lengthening "
            "the receiver ranges also raises RMax; shortening them "
            "lowers it."
        ),
    )


def green_cube_axes(source, receiver,
                    settings) -> Tuple[int, int, int, int]:
    """``(n_freqs, n_source_depths, n_receiver_depths, n_ranges)`` of
    the Green's-function cube and transform kernel ONE launch of this
    call holds.

    The deck writes the frequency vector only when it holds more than
    one entry (``write_broadband_freqs``); otherwise the ``.grn``
    carries the single header frequency — the same branch
    :func:`deck_max_frequency` mirrors. A per-depth loop launches one
    source depth at a time.
    """
    n_freqs = 1
    if settings.mode in _BROADBAND_MODES and settings.frequencies.size > 1:
        n_freqs = int(settings.frequencies.size)
    n_source_depths = (1 if settings.depth_loop == 'per_depth' else
                       int(np.atleast_1d(np.asarray(source.depths)).size))
    return (n_freqs, n_source_depths,
            int(np.atleast_1d(np.asarray(receiver.depths)).size),
            int(np.atleast_1d(np.asarray(receiver.ranges)).size))


def green_cube_detail(nk: int, n_freqs: int, n_source_depths: int,
                      n_receiver_depths: int, n_ranges: int, *,
                      rmax_m: float, f_deck: float, c_low: float,
                      c_high: float) -> Tuple[int, int, str]:
    """``(cube, peak, detail)``: the bytes of the complex64
    Green's-function cube and of the peak the read and transform reach,
    and the sentence that states them.

    ``scooter.exe`` holds one frequency's ``Green(NSz, NRz, Nk)`` at a
    time, but the ``.grn`` accumulates every frequency and
    ``read_grn_file`` allocates the whole ``(nfreq, nsd, nrd, nk)``
    complex64 cube in one ``np.zeros`` — the Python process, not the
    binary, takes the hit. ``Nk`` grows linearly with RMax
    (``receiver.ranges.max() × rmax_factor``), the top deck frequency
    and the phase-speed span, so a plausible broadband deck reaches tens
    of GB with no single knob looking unreasonable.

    The cube is not the whole bill, so counting it alone under-reads the
    peak: :func:`~uacpy.core.acoustics.hankel_transform` also builds
    ``outer(k, r)`` in double and exponentiates it in place, then casts the
    result down, so the kernel costs another ``nk × nr × 24`` bytes (16 for
    the complex128 phase, 8 for the complex64 copy it becomes) on top of a
    second copy of the cube. All of that is estimated here. Measured
    against peak RSS on an ``nk x nr`` of 120000 x 499, the estimate is
    1.17x the real peak — tight, and on the safe side.
    """
    cube = (8 * int(n_freqs) * int(n_source_depths)
            * int(n_receiver_depths) * int(nk))
    kernel = (_TRANSFORM_BYTES_PER_ELEMENT * int(nk)
              * max(int(n_ranges), 1))
    peak = 2 * cube + kernel
    detail = (
        f"Nk = {int(nk)} wavenumber samples: a "
        f"{cube / 1024 ** 3:.1f} GiB complex64 Green's-function cube "
        f"({int(n_freqs)} frequencies x {int(n_source_depths)} source "
        f"depth(s) x {int(n_receiver_depths)} receiver depth(s) x Nk x "
        f"8 B) and a {kernel / 1024 ** 3:.1f} GiB transform kernel "
        f"({int(nk)} x {int(n_ranges)} ranges), about "
        f"{peak / 1024 ** 3:.1f} GiB at peak. scooter.f90:69 derives "
        f"Nk = INT(2000 * RMax_km * (kMax - kMin) / pi) from "
        f"RMax = {rmax_m:g} m at {f_deck:.6g} Hz with "
        f"c_low = {c_low:.1f} and c_high = {c_high:.1f} m/s."
    )
    return cube, peak, detail


def reject_oversized_green_cube(
    nk: int, n_freqs: int, n_source_depths: int,
    n_receiver_depths: int, n_ranges: int, *, rmax_m: float,
    f_deck: float, c_low: float, c_high: float,
) -> Tuple[int, Optional[Notice]]:
    """``(peak, headroom)``: the deck's peak memory in bytes
    (:func:`green_cube_detail`) and the notice
    :func:`~uacpy.models._budget.memory_budget` gives for it (``None`` when
    it says nothing). Raises when the peak is more than the host reports
    free.
    """
    cube, peak, detail = green_cube_detail(
        nk, n_freqs, n_source_depths, n_receiver_depths, n_ranges,
        rmax_m=rmax_m, f_deck=f_deck, c_low=c_low, c_high=c_high)
    return peak, memory_budget(
        peak, model_name='Scooter', what='peak memory',
        detail=f"This deck asks for {detail}",
        remediation=(
            "Peak memory scales with Nk, which grows with RMax = "
            "receiver.ranges.max() x rmax_factor, with the top deck "
            "frequency and with the width of the c_low/c_high phase-speed "
            "window. Fewer receiver depths or ranges shrink it too."))


def mesh_floor_notice(freq0, freqs, *,
                      n_mesh) -> Optional[Tuple[str, str]]:
    """``(note, warning)`` when ``scooter.f90`` replaces a pinned
    ``n_mesh`` at some of ``freqs``, else ``None``.

    ``scooter.f90:106-111`` meshes each medium with
    ``N = MAX(INT(freq/freq0 * NG), 100)``, so wherever the scaled count
    falls below 100 the binary runs 100 points and the ``.prt`` still
    echoes the ``NG`` asked for. A convergence study over such values
    measures one mesh (scooter.md: ``n_mesh`` 34, 40, 70 and 100 give
    bit-identical TL at 50 Hz).
    """
    if not n_mesh or n_mesh <= 0:
        return None
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    scaled = (freqs / freq0 * n_mesh).astype(int)
    floored = freqs[scaled < 100]
    if floored.size == 0:
        return None
    where = (f"at {floored[0]:g} Hz" if floored.size == 1 else
             f"at {floored.size} of {freqs.size} frequencies "
             f"({floored.min():g}-{floored.max():g} Hz)")
    return Notice(
        f"n_mesh {n_mesh} is under the binary's 100-point floor "
        f"{where}",
        f"Scooter(n_mesh={n_mesh}) has no effect {where}: "
        f"scooter.f90:106-111 meshes each medium with "
        f"N = MAX(INT(freq/freq0 * n_mesh), 100), so the binary runs 100 "
        f"points there while the .prt echoes n_mesh. Step n_mesh above "
        f"100 (or leave it at 0) for a mesh that changes the answer.",
        FallbackWarning,
    )


def steep_path_notice(env, source, receiver,
                      c_high: float) -> Optional[Tuple[str, str]]:
    """``(note, warning)`` when a receiver's direct or surface-reflected
    path is steeper than the auto-derived ``c_high`` integrates, else
    ``None`` (:func:`~uacpy.models._window.steep_path_notice`).

    A steeper path is simply absent from the field, with no sign of it in
    the result: measured on a Lloyd geometry over a transparent bottom,
    23 dB of error at 0.2 km. Jensen et al. (*Computational Ocean
    Acoustics*, ch. 4 App. 1) recommend the unbounded ``k_min = 0``
    whenever in doubt."""
    return _steep_path_notice(
        env, source, receiver, c_high, model_name='Scooter',
        kept='paths', summed_in='integral',
        evidence='measured 23 dB at 0.2 km in a Lloyd geometry',
        remediation=("Pass c_high=1e9 to integrate from k = 0, the Acoustics "
                "Toolbox's unbounded value, and a small taper (e.g. "
                "taper=0.05) to remove the ripple the k_min edge leaves."))
