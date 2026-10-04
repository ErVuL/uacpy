"""The frequency and wavenumber sampling OASES settles on, computed before
a launch: the OASR/OASP frequency sweep and its resampling, OASP's FFT
grid and frequency ladder, and the wavenumber count of OASSP's
roughness branch."""

from typing import NamedTuple, Optional

import numpy as np

from uacpy.models._notices import give_notice
from uacpy.core._validate import steps_are_uniform
from uacpy.core.exceptions import ConfigurationError, FallbackWarning
from uacpy.io.oases_writer import _OASES_MAX_WAVENUMBERS


def _oases_resample_frequencies(
    freqs: np.ndarray, model_name: str, log_spaced: bool = False,
    argument: str = 'frequencies=',
) -> tuple[float, float, int]:
    """:func:`_oases_frequency_sweep`'s ``(freq_min, freq_max, N)``, its notice given
    at once as a warning (OASR's sweep, resolved outside the notices of a
    run's settings)."""
    freq_min, freq_max, n, notice = _oases_frequency_sweep(
        freqs, model_name, log_spaced=log_spaced, argument=argument)
    if notice is not None:
        give_notice(None, notice, FallbackWarning)
    return freq_min, freq_max, n


def _oases_frequency_sweep(
    freqs: np.ndarray, model_name: str, log_spaced: bool = False,
    argument: str = 'frequencies=',
) -> tuple[float, float, int, Optional[str]]:
    """Convert an arbitrary frequency vector to the ``(freq_min, freq_max, N)``
    triple OASR/OASP write into the input file, with the notice (text, or
    ``None``) of a vector the sweep does not reproduce.

    OASR's ``.dat`` and OASP's broadband kernel both express their
    frequency axis as a min/max/count, which implies equispaced
    sampling. A non-equispaced vector is resampled onto
    ``np.linspace(freq_min, freq_max, N)``, and the notice says so, so the result's
    ``Result.frequencies`` reflects what the model actually saw.
    ``argument`` names where the vector came from in the messages:
    ``'frequencies='`` (OASP's run keyword) or ``'source.frequencies'``
    (OASR).

    Returns
    -------
    freq_min, freq_max : float
    n : int
        Number of equispaced bins.
    notice : str or None
        What the sweep does to the vector it was given, or ``None``.

    Raises
    ------
    ConfigurationError
        When ``log_spaced`` is set and the vector's lowest frequency is not
        strictly positive: the OASR kernel takes ``LOG(FREQ1)``.
    """
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    freq_min = float(freqs.min())
    freq_max = float(freqs.max())
    n = int(freqs.size)
    if n > 1 and log_spaced:
        # OASR option 'C' makes the kernel sweep LOGARITHMIC
        # (unoasr21.f:123-125, :243) — it is not a plot option. The
        # equispacing test below is then exactly inverted: a linear request
        # is the one being silently regridded, and a log request is correct.
        # A log sweep has no grid to sit on unless every frequency is
        # positive: the kernel takes F1LOG = LOG(FREQ1) (unoasr21.f:123) and
        # the binary aborts a zero bound one line earlier
        # (:116-118, "FREQUENCIES MUST BE NON-ZERO") while a negative one
        # reaches LOG unchecked. Refused here so the deck is never written.
        # NaN-closed: nan fails `> 0` both ways.
        if not (freq_min > 0):
            raise ConfigurationError(
                f"{model_name}: option 'C' makes the frequency sweep "
                f"logarithmic (unoasr21.f:123-125), which needs every "
                f"frequency strictly positive; the {argument} vector "
                f"starts at {freq_min}. Raise the lower bound above 0 Hz, or "
                f"drop 'C' for a linear sweep."
            )
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = freqs[1:] / freqs[:-1]
        target = (freq_max / freq_min) ** (1.0 / (n - 1))
        notice = None
        if not steps_are_uniform(ratios, target):
            grid = np.geomspace(freq_min, freq_max, n)
            notice = (
                f"{model_name}: option 'C' makes the frequency sweep "
                f"logarithmic (unoasr21.f:123-125 computes the coefficients at "
                f"exp-spaced frequencies), but the {argument} vector is not "
                f"log-spaced. It will be evaluated at "
                f"np.geomspace({freq_min}, {freq_max}, {n}) — "
                f"{np.array2string(grid, precision=4, max_line_width=200)} — "
                f"not at the values given. Pass a log-spaced vector, or drop "
                f"'C' for a linear sweep.")
        return freq_min, freq_max, n, notice
    notice = None
    if n > 1:
        diffs = np.diff(freqs)
        # Equispaced if every diff matches the mean diff
        # (:func:`steps_are_uniform`).
        target = (freq_max - freq_min) / (n - 1)
        if not steps_are_uniform(diffs, target):
            notice = (
                f"{model_name}: {argument} vector is non-equispaced; "
                f"OASES expresses the frequency axis as (freq_min, freq_max, N) "
                f"so the vector has been resampled onto "
                f"np.linspace({freq_min}, {freq_max}, {n}). "
                f"Result.frequencies will be the resampled grid, not "
                f"your input. Pass an equispaced vector, e.g. "
                f"np.linspace(freq_min, freq_max, N), to suppress.")
    return freq_min, freq_max, n, notice


# OASES' wavenumber-array bound ``_OASES_MAX_WAVENUMBERS`` (imported from the
# deck writer, its one definition): ``NPEXP = 16, NP = 2**NPEXP``
# (third_party/oases/src/compar.f:37-38). OAST stops above it
# (unoast31.f:459 ``IF (NWVNO.GT.NP) STOP '>>> TOO MANY WAVENUMBERS <<<'``),
# which is what catches its automatic branch — a pinned NW never reaches
# that test, being clamped by ``NWVNO=MIN0(NWVNO,NP)`` at :435 first.
# OASP has no such test at all — AUTSAM (unoasp22.f:1130
# ``NW=(WNMAX-WNMIN)/DK+1``) applies no clamp and :174's MIN0 runs before
# it — so it runs past the arrays and writes a transfer function whose
# values are meaningless.

#: Largest NWVNO oassp2's roughness branch can take. It rounds NWVNO up to a
#: power of two ``nwvnor`` (unoassp30.f:542-544) and hands it to PV as NKR,
#: which works on NKR_I = 2*NKR wavenumbers (oasvun31.f:2273) and writes
#: ``cfk``/``sqrtp`` up to index NKR_I (:2532-2533) into arrays dimensioned
#: ``nnkr`` = 8192 (:2180-2181, comvol.f:3). PV's own test (:2217) bounds NKR,
#: not NKR_I, and the volume branch's ``nwvnor.gt.nkrm`` stop (unoassp30.f:560)
#: is not on the roughness path — so NWVNO > 4096 (nwvnor = 8192) overruns
#: them: SIGSEGV in ``vmov_``/``pv_``, or a corrupted ``.trf``.
_OASSP_MAX_WAVENUMBERS = 4096


def _oassp_wavenumber_count(*, layer_speeds, c_low, c_high, source_depth,
                            receiver_depths, r0_km, rspace_km, nplots,
                            n_time, freq_max, dt, options, n_wavenumbers):
    """NWVNO oassp2 integrates on, replicated from its own sampling logic.

    A pinned ``n_wavenumbers >= 1`` is taken as written (manual branch,
    unoassp30.f:344-349; the ``MIN0(NWVNO,NP)`` at :176 cannot bind below
    NP). Otherwise this is the automatic branch, unoassp30.f:281-340 with
    AUTSAM (:1069-1126), in the deck's own units — ``R0``/``RSPACE`` in km,
    and ``rmi = max(|RSPACE|, 0.1*cref*T)`` mixes that km step with metres
    exactly as :293 does:

    * ``FREQ = FREQM = DLFREQ*(MX-1)``, ``DLFREQ = 1/(DT*NX)``,
      ``MX = min(int(FR2/DLFREQ + 2), NX/2)`` (:223-229);
    * ``cref`` = the largest layer speed below 2e4 m/s, ``T = NX*DT``,
      ``RANREF = min(cref*T + 2*rm, 6*rm)`` with ``rm`` the farthest receiver
      range in metres (:287-306);
    * AUTSAM: ``DK = 2*pi/(RFAC*RANREF)``, ``RFAC = 1`` for Filon
      (``inttyp == 1``) else 1.5; ``rref = rmi + 10*zsep`` (one wavelength of
      the source layer when 0); ``taperk = 2*pi*10/rref``;
      ``WN1 = 2*pi*f/C2``, ``WN2 = 2*pi*f/C1`` (``2*pi*1.2*f/C1`` under
      option 'A'); ``wnmin = max(WN1 - taperk, 1e-3*DK)``,
      ``wnmax = WN2 + taperk``, ``DK = min(DK, (wnmax - wnmin)/199)``,
      ``NW = int((wnmax - wnmin)/DK + 1)``;
    * cylindrical geometry (no 'P') forces ``C2 = CMAXIN = 1e12`` and the
      full Hankel transform (``inttyp = 2``, :205-217); plane geometry ('P')
      keeps the deck's CMAX and doubles NWVNO (:333-340).

    ``layer_speeds`` is ``[(top_depth_m, cp_m_s), ...]`` in deck order;
    ``c_low``/``c_high`` are the deck's CMIN/CMAX.
    """
    if n_wavenumbers is not None and int(n_wavenumbers) >= 1:
        return min(int(n_wavenumbers), _OASES_MAX_WAVENUMBERS)

    opts = str(options or '')
    cylgeo = 'P' not in opts
    ibody = 0
    inttyp = 0
    for ch in opts:
        if ch == 'B' and ibody == 0:
            ibody = 1
        elif ch == 'A' and ibody == 0:
            ibody = 2
        elif ch == 'F':
            inttyp = 1
        elif ch == 'f':
            inttyp = 2
    if cylgeo:
        inttyp = 2

    nx = int(n_time)
    dlfreq = 1.0 / (dt * nx)
    mx = min(int(freq_max / dlfreq + 2), nx // 2)
    freq = dlfreq * (mx - 1)

    speeds = [c for _, c in layer_speeds]
    cref = max([c for c in speeds if c < 2e4] + [0.0])
    ranref = cref * nx * dt
    rm = max(abs(1e3 * (r0_km + i * rspace_km)) for i in range(int(nplots)))
    rmi = max(abs(rspace_km), 0.1 * ranref)
    ranref = min(ranref + 2.0 * rm, 6.0 * rm)

    rfac = 1.0 if inttyp == 1 else 1.5
    dk = 2.0 * np.pi / (rfac * ranref)
    zsep = min(abs(float(rd) - float(source_depth))
               for rd in np.atleast_1d(receiver_depths))
    rref = rmi + 10.0 * zsep
    if rref == 0.0:
        # v(lays(1),2)/freq: one wavelength in the source's layer.
        in_layer = [c for top, c in layer_speeds if top <= source_depth]
        rref = (in_layer[-1] if in_layer else speeds[0]) / freq
    taperk = 2.0 * np.pi * 10.0 / rref
    c2 = 1e12 if cylgeo else float(c_high)
    wn1 = 2.0 * np.pi * freq / c2
    wn2 = (2.0 * np.pi * (freq + 0.2 * freq) / c_low if ibody == 2
           else 2.0 * np.pi * freq / c_low)
    wnmax = wn2 + taperk
    wnmin = max(wn1 - taperk, dk * 1e-3)
    dk = min(dk, (wnmax - wnmin) / 199.0)
    nw = int((wnmax - wnmin) / dk + 1)
    return nw if cylgeo else 2 * nw


class _OaspFftGrid(NamedTuple):
    """Block VIII's ``NX FR1 FR2 DT`` as OASP runs on them
    (``unoasp22.f:194-246``), in Fortran REAL: the values it writes to a
    mean field's ``.rhs`` header (``unoasp22.f:254``,
    :func:`~uacpy.io.oases_reader.read_oases_rhs_header`), and the first
    and last bin (``LXP1``, ``MX``) of its ``.trf``."""
    n_time_samples: int
    freq_min: float
    freq_max: float
    time_step: float
    first_bin: int
    last_bin: int


def _oasp_fft_grid(n_time_samples: int, freq_min: float, freq_max: float,
                   time_step: float) -> _OaspFftGrid:
    """The FFT grid OASP runs on for the Block VIII ``NX FR1 FR2 DT`` it is
    given.

    Computed as ``unoasp22.f:194-246`` computes it, in Fortran REAL
    (float32) from the values the deck carries (FR1/FR2 written ``%.9f``, DT
    ``%.12g``): DT halved until ``DT <= 2.5/FR2`` with NX doubled alongside;
    NX rounded up to a power of two and clamped at ``2*NP``;
    ``DLFREQ = 1/(DT*NX)``, ``MX = FR2/DLFREQ + 2``, ``LX = FR1/DLFREQ + 1``,
    ``LX >= 1``, ``MX <= NX/2``, ``LXP1 = max(2, LX)``, and one bin
    (``MX = LXP1``) when ``FR2 - FR1 < DLFREQ``.
    """
    f32 = np.float32
    fr1 = f32(float(f"{freq_min:.9f}"))
    fr2 = f32(float(f"{freq_max:.9f}"))
    dt = f32(float(f"{time_step:.12g}"))
    nx = int(n_time_samples)
    if dt > f32(0.0):
        while dt > f32(2.5) / fr2:
            dt = dt * f32(0.5)
            nx *= 2
    else:
        dt = f32(2.5) / fr2
    ii = 2
    while ii < nx:
        ii *= 2
    nx = min(ii, 2 * _OASES_MAX_WAVENUMBERS)
    dlfreq = f32(1.0) / (dt * f32(nx))
    mx = int(fr2 / dlfreq + f32(2.0))
    lx = max(int(fr1 / dlfreq + f32(1.0)), 1)
    mx = min(mx, nx // 2)
    lxp1 = max(2, lx)
    if (fr2 - fr1) < dlfreq:
        mx = lxp1
    return _OaspFftGrid(n_time_samples=nx, freq_min=float(fr1),
                        freq_max=float(fr2), time_step=float(dt),
                        first_bin=lxp1, last_bin=mx)


def _oasp_marched_frequencies(n_time_samples: int, freq_min: float,
                           freq_max: float, time_step: float) -> np.ndarray:
    """The frequencies (Hz) of the bins OASP writes to its ``.trf`` for the
    Block VIII ``NX FR1 FR2 DT`` it is given, as
    :func:`~uacpy.io.oases_reader.read_oasp_trf` labels them: TRFHEAD writes
    ``NX LXP1 MX DT`` (``unoasp22.f:542-543``, ``oasiun23.f:881-882``) of
    :func:`_oasp_fft_grid` and bin ``k`` carries ``(k - 1)/(DT*NX)``.
    """
    grid = _oasp_fft_grid(n_time_samples, freq_min, freq_max, time_step)
    return np.array([(k - 1) / (grid.time_step * grid.n_time_samples)
                     for k in range(grid.first_bin, grid.last_bin + 1)],
                    dtype=np.float64)
