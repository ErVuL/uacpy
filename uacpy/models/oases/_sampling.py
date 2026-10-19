"""The frequency and wavenumber sampling OASES settles on, computed before
a launch: OASP's FFT grid and frequency ladder, and the wavenumber count of
OASSP's roughness branch. The frequency sweep a deck carries is
:func:`~uacpy.io.oases_writer.oases_frequency_sweep`."""

from typing import NamedTuple

import numpy as np

from uacpy.models._knobs import whole_count
from uacpy.core.exceptions import ConfigurationError
from uacpy.io.oases_writer import _OASES_MAX_WAVENUMBERS


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


class _OasspWavenumberSampling(NamedTuple):
    """What oassp2 integrates on (:func:`_oassp_wavenumber_sampling`):
    ``count`` wavenumbers (NWVNO) up to ``k_max`` (rad/m), ``WKMAX =
    2*pi*FREQM/CMIN`` (unoassp30.f:351-352). The window is fixed in
    wavenumber at the band's top frequency ``freq_top`` (Hz, FREQM), so at a
    frequency ``f`` it reaches down to the phase speed ``2*pi*f/k_max``."""
    count: int
    freq_top: float
    k_max: float


def _oassp_wavenumber_count(**sampling) -> int:
    """NWVNO oassp2 integrates on
    (:func:`_oassp_wavenumber_sampling`'s ``count``)."""
    return _oassp_wavenumber_sampling(**sampling).count


def _oassp_wavenumber_sampling(
        *, layer_speeds, c_low, c_high, source_depth, receiver_depths, r0_km,
        rspace_km, nplots, n_time, freq_max, dt, options,
        n_wavenumbers) -> _OasspWavenumberSampling:
    """NWVNO oassp2 integrates on and the window's top wavenumber,
    replicated from its own sampling logic.

    A pinned ``n_wavenumbers >= 1`` is written to the deck as given, and
    oassp2 takes it through its own ``NWVNO=MIN0(NWVNO,NP)`` (unoassp30.f:176)
    into the manual branch (:344-349). Otherwise this is the automatic
    branch, unoassp30.f:281-340 with
    AUTSAM (:1069-1134), in the deck's own units — ``R0``/``RSPACE`` in km,
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
      keeps the deck's CMAX and doubles NWVNO (:333-340);
    * ``WKMAX = 2*pi*FREQM/CMIN`` (:351-352): AUTSAM's ``CMIN =
      2*pi*FREQ/WNMAX`` (:1131) makes it ``WNMAX = wnmin + (NW - 1)*DK``
      (:1126); a pinned count keeps ``CMIN = CMININ`` (:346).

    ``layer_speeds`` is ``[(top_depth_m, cp_m_s), ...]`` in deck order;
    ``c_low``/``c_high`` are the deck's CMIN/CMAX.
    """
    nx = int(n_time)
    dlfreq = 1.0 / (dt * nx)
    mx = min(int(freq_max / dlfreq + 2), nx // 2)
    freq = dlfreq * (mx - 1)

    if n_wavenumbers is not None and int(n_wavenumbers) >= 1:
        # The binary's own cap on the count the deck carries
        # (unoassp30.f:176), mirrored to predict what oassp2 integrates on.
        return _OasspWavenumberSampling(
            count=min(int(n_wavenumbers), _OASES_MAX_WAVENUMBERS),
            freq_top=freq, k_max=2.0 * np.pi * freq / float(c_low))

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
    return _OasspWavenumberSampling(
        count=nw if cylgeo else 2 * nw, freq_top=freq,
        k_max=wnmin + (nw - 1) * dk)


class _OaspWavenumberTop(NamedTuple):
    """Where OASP's wavenumber window stops (:func:`_oasp_wavenumber_top`):
    ``k_max`` (rad/m) at the band's top frequency ``freq_top`` (Hz, FREQM),
    the phase speed ``edge`` = ``2π·freq_top/k_max`` (m/s) the run reaches
    there, and whether the window is ``fixed`` in wavenumber over the band
    (then at a frequency ``f`` it reaches ``2πf/k_max``)."""
    freq_top: float
    k_max: float
    edge: float
    fixed: bool


def _oasp_wavenumber_top(*, layer_speeds, c_low, c_high, source_depth,
                         receiver_depths, receiver_ranges, n_time, dt,
                         freq_top, options,
                         n_wavenumbers) -> _OaspWavenumberTop:
    """Where OASP's wavenumber integral stops, replicated from unoasp22.f.

    A pinned count samples ``WK0 + i·DLWVNO`` with ``DLWVNO`` set at the
    band top FREQM from ``WKMAX = 2π·FREQM/CMIN`` (:394-396, and the manual
    branch at :635-639): the window reaches ``2π·FREQM/c_low``, to within
    ``2π·FREQM/CMAX``, at every frequency. Under automatic sampling AUTSAM
    runs at each frequency ``f`` (:603-617) on ``CMININ·f/FREQM`` in
    wavenumber integration, so its ``WN2 = 2πf/C1`` is ``2π·FREQM/c_low``
    at every ``f`` (slowness integration, option 'B' or 'A', takes
    ``CMININ`` as given, and its window moves with ``f``), and
    (:1088-1136):

    * ``RFAC`` 1 under Filon (``'F'``), else 1.5; ``DK = 2π/(RFAC·RANREF)``;
    * ``rref = rmi + 10·zsep`` (``zsep`` the least source-receiver depth
      separation), one wavelength of the source's layer at ``f`` when 0;
      ``taperk = 2π·10/rref``;
    * ``wnmax = WN2 + taperk`` (:1125), ``wnmin = max(WN1 − taperk,
      1e-3·DK)``, ``DK = min(DK, (wnmax − wnmin)/199)``, ``NW =
      int((wnmax − wnmin)/DK + 1)``, ``WNMAX = wnmin + (NW − 1)·DK``,
      ``CMIN = 2πf/WNMAX`` and ``WKMAX = 2πf/CMIN = WNMAX`` (:617).

    The range axis is Block VIII's ``R0 RSPACE NPLOTS`` (the first receiver
    range, its step — 1 km for one range — and the count): under ``'J'``
    ``rmi = |RSPACE|``, ``RANREF = 3·r_max`` (:296-303); otherwise
    ``RANREF = max(cref·NX·DT, 6·r_max)`` and ``rmi = max(|RSPACE|,
    0.1·cref·NX·DT)``, ``cref`` the fastest layer speed under 2e4 m/s
    (:306-321). ``layer_speeds`` is ``[(top_depth_m, cp_m_s), ...]`` in
    deck order; ``n_time``, ``dt`` and ``freq_top`` are the deck's NX, DT
    and FREQM as OASP settles them (:func:`_oasp_fft_grid`).
    """
    opts = str(options or '')
    ibody, inttyp = 0, 0
    for ch in opts:
        if ch == 'B' and ibody == 0:
            ibody = 1
        elif ch == 'A' and ibody == 0:
            ibody = 2
        elif ch == 'F':
            inttyp = 1
        elif ch == 'f':
            inttyp = 2
    freq = float(freq_top)
    if n_wavenumbers is not None and int(n_wavenumbers) >= 1:
        k_max = 2.0 * np.pi * freq / float(c_low)
        return _OaspWavenumberTop(freq_top=freq, k_max=k_max,
                                  edge=float(c_low), fixed=ibody == 0)

    ranges = np.atleast_1d(np.asarray(receiver_ranges, dtype=float))
    rspace = float(np.diff(ranges)[0]) if ranges.size > 1 else 1e3
    rm = float(np.max(np.abs(ranges.min()
                             + np.arange(ranges.size) * rspace)))
    if 'J' in opts:
        rmi = abs(rspace)
        ranref = 3.0 * rm
    else:
        cref = max([c for _, c in layer_speeds if c < 2e4] + [0.0])
        ranref = cref * int(n_time) * float(dt)
        rmi = max(abs(rspace), 0.1 * ranref)
        ranref = max(ranref, 6.0 * rm)
    zsep = min(abs(float(rd) - float(source_depth))
               for rd in np.atleast_1d(receiver_depths))
    rref = rmi + 10.0 * zsep
    fixed = ibody == 0 and rref != 0.0
    if rref == 0.0:
        in_layer = [c for top, c in layer_speeds[1:] if top <= source_depth]
        rref = (in_layer[-1] if in_layer else layer_speeds[1][1]) / freq
    taperk = 2.0 * np.pi * 10.0 / rref
    dk = 2.0 * np.pi / ((1.0 if inttyp == 1 else 1.5) * ranref)
    wn1 = 2.0 * np.pi * freq / float(c_high)
    wn2 = (2.0 * np.pi * 1.2 * freq / float(c_low) if ibody == 2
           else 2.0 * np.pi * freq / float(c_low))
    wnmax = wn2 + taperk
    wnmin = max(wn1 - taperk, dk * 1e-3)
    dk = min(dk, (wnmax - wnmin) / 199.0)
    nw = int((wnmax - wnmin) / dk + 1)
    k_max = wnmin + (nw - 1) * dk
    return _OaspWavenumberTop(freq_top=freq, k_max=k_max,
                              edge=2.0 * np.pi * freq / k_max, fixed=fixed)


def _oassp_fitting_c_low(*, c_low, c_low_max, **sampling):
    """The lowest whole m/s above ``c_low``, at most ``c_low_max``, at which
    :func:`_oassp_wavenumber_count` on ``sampling`` (its keywords other than
    ``c_low``) fits :data:`_OASSP_MAX_WAVENUMBERS`; ``c_low_max`` itself
    when no whole m/s below it fits, and ``None`` when ``c_low_max`` does
    not fit either. ``c_low`` enters the count only through ``WN2 =
    2*pi*f/C1``, so the count never grows with it and the search bisects;
    ``c_low`` is a value whose count is past the bound."""
    def fits(c):
        return (_oassp_wavenumber_count(c_low=c, **sampling)
                <= _OASSP_MAX_WAVENUMBERS)

    if not fits(c_low_max):
        return None
    lo, hi = int(np.floor(c_low)), int(np.floor(c_low_max))
    if hi <= lo or not fits(hi):
        return float(c_low_max)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mid):
            hi = mid
        else:
            lo = mid
    return float(hi)


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


def check_power_of_two_time_samples(name: str, n_time_samples) -> None:
    """Refuse an ``n_time_samples`` the deck cannot carry as given: not a
    whole number of at least 2, or not a power of two. oasp.tex:129 states
    ``NT = 2^M``, and OASP changes any other count to the next power of
    two (unoasp22.f:222-232)."""
    whole_count(name, n_time_samples, 2)
    n = int(n_time_samples)
    if n & (n - 1):
        up = 1 << (n - 1).bit_length()
        raise ConfigurationError(
            f"{name}={n} is not a power of two: oasp.tex:129 states "
            f"NT = 2^M, and OASP changes any other count to the next one "
            f"(unoasp22.f:222-232).",
            remediation=f"Pass n_time_samples={up} or {up // 2}.")
