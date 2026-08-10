"""
Green's function reader for SCOOTER (and SPARC snapshot mode).

The wavenumber-domain Green's function file is the SHD-format binary written
by SCOOTER (``scooter.f90``) or SPARC in snapshot mode (``sparc.f90``).
The Hankel/Fourier transform here mirrors ``Acoustics-Toolbox/Matlab/Scooter/
fieldsco.m`` — the canonical reference implementation maintained by Porter.

Convention summary
------------------
* SCOOTER — phase speed vector ``cVec`` is stored at the highest frequency
  (``scooter.f90:79``); per-frequency wavenumbers are recovered as
  ``k(f) = 2π·f/cVec``.
* SPARC snapshot — the wavenumber grid is **frequency-independent** so
  ``cVec`` is mapped using the source frequency ``freq0`` from the GRN
  header (``fieldsco.m:100-102``); ``freqVec`` actually stores the output
  *time* vector (``sparc.f90:320``), not frequencies.

We detect SPARC by the ``'SPARC'`` prefix in the title (set at
``sparc.f90:84``).
"""

import warnings

import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Optional

from uacpy.acoustic_signal.waveforms import sparc_pulse
from uacpy.core.results import Field
from uacpy.io._fortran_helpers import detect_endian, typed_format_error
from uacpy.core.exceptions import ConfigurationError, FileFormatError


@typed_format_error
def read_grn_file(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Read a SCOOTER / SPARC Green's function file (``.grn``).

    The format is the same Fortran direct-access binary as ``.shd`` (record
    length stored in 4-byte words; see ``misc/RWSHDFile.f90``).

    Returns
    -------
    grn_data : dict
        ``freq``         : float — source frequency (Hz; from REC=3 ``freq0``).
        ``freqVec``      : ndarray — for SCOOTER, vector of frequencies; for
                           SPARC snapshot, vector of *output times* (s).
        ``nfreq``        : int — length of ``freqVec``.
        ``nsd``, ``nrd`` : int — number of source / receiver depths.
        ``nk``           : int — number of wavenumber samples.
        ``sd``, ``rd``   : ndarray — source / receiver depths (m).
        ``cVec``         : ndarray — phase-speed grid stored in REC=10
                           (``Pos%Rr`` slot), monotonically decreasing.
        ``atten``        : float — stabilising attenuation written by the
                           solver (REC=3). For SCOOTER this equals
                           ``Δk`` unless TopOpt(7)='0' (then 0). For SPARC
                           snapshot this is 0 (``sparc.f90:313``).
        ``G``            : complex64 ndarray, shape ``(nfreq, nsd, nrd, nk)``.
        ``title``        : str — title line (used to distinguish SCOOTER vs
                           SPARC).
        ``PlotType``     : str — ``'Green'`` for these files.
        ``is_sparc``     : bool — True iff ``title`` starts with ``'SPARC'``.
    """
    filepath = Path(filepath)

    with open(filepath, "rb") as f:
        head = f.read(4)
        f.seek(0)
        endian = detect_endian(head, source=f'read_grn_file:{filepath.name}')
        i4 = np.dtype(endian + 'i4')
        f4 = np.dtype(endian + 'f4')
        f8 = np.dtype(endian + 'f8')

        # Record 1: recl (int32, in 4-byte words) + title (80 chars)
        recl = int(np.fromfile(f, dtype=i4, count=1)[0])
        title = f.read(80).decode("ascii", errors="ignore").strip()

        f.seek(4 * recl, 0)

        # Record 2: PlotType (10 chars)
        PlotType = f.read(10).decode("ascii", errors="ignore").strip()

        f.seek(2 * 4 * recl, 0)

        # Record 3: 7 int32 + freq0 (float64) + atten (float64)
        nfreq = int(np.fromfile(f, dtype=i4, count=1)[0])
        np.fromfile(f, dtype=i4, count=3)       # Ntheta, NSx, NSy — unused
        nsd = int(np.fromfile(f, dtype=i4, count=1)[0])   # NSz
        nrd = int(np.fromfile(f, dtype=i4, count=1)[0])   # NRz
        nk = int(np.fromfile(f, dtype=i4, count=1)[0])    # NRr — number of k samples
        freq0 = float(np.fromfile(f, dtype=f8, count=1)[0])
        atten = float(np.fromfile(f, dtype=f8, count=1)[0])

        # File-size-aware sanity bound on the header counts before any
        # vector or the (nfreq, nsd, nrd, nk) cube is sized off them. A
        # corrupt/hostile header (e.g. nk=0x3ffffff0, or all counts = 1000)
        # would otherwise drive a multi-GB/TB allocation before a single
        # data record is validated. The G cube holds nfreq*nsd*nrd*nk
        # complex samples, each stored on disk as 2 float32 (8 bytes), so
        # the element count cannot exceed file_size // 8.
        f.seek(0, 2)
        file_size = f.tell()
        for _name, _val in (("nfreq", nfreq), ("nsd", nsd),
                            ("nrd", nrd), ("nk", nk)):
            if _val < 0:
                raise FileFormatError(
                    f"read_grn_file: negative header count {_name}={_val}."
                )
        if nfreq * nsd * nrd * nk > file_size // 8:
            raise FileFormatError(
                f"read_grn_file: header counts "
                f"(nfreq={nfreq}, nsd={nsd}, nrd={nrd}, nk={nk}) imply "
                f"{nfreq * nsd * nrd * nk} complex samples, implausible for "
                f"a {file_size}-byte file."
            )

        f.seek(3 * 4 * recl, 0)

        # Record 4: frequency vector (or time vector for SPARC snapshot)
        freqVec = np.fromfile(f, dtype=f8, count=nfreq)

        # Records 5-7: theta / sx / sy — skipped
        f.seek(7 * 4 * recl, 0)

        # Records 8-10 hold Pos%Sz, Pos%Rz and Pos%Rr (RWSHDFile.f90:111-114).
        # The depth vectors are REAL(KIND=4) and the range vector — whose slot
        # the phase speeds occupy — is REAL(KIND=8)
        # (SourceReceiverPositions.f90:23-25); the record-length formula counts
        # them the same way, ``Pos%NSz, Pos%NRz, 2 * Pos%NRr`` words
        # (RWSHDFile.f90:100).

        # Record 8: Source depths
        sd = np.fromfile(f, dtype=f4, count=nsd)

        f.seek(8 * 4 * recl, 0)

        # Record 9: Receiver depths
        rd = np.fromfile(f, dtype=f4, count=nrd)

        f.seek(9 * 4 * recl, 0)

        # Record 10: phase-speed vector, Nk entries
        cVec = np.fromfile(f, dtype=f8, count=nk)

        # Records 11+: complex Green's function, one record per
        # (freq, source_depth, receiver_depth) tuple, receiver depth fastest —
        # ``REC = 10 + (ifreq-1)*NSz*NRz + (iS-1)*NRz + iR`` (scooter.f90:588;
        # SPARC's snapshot writes the same slabs with output time in the
        # frequency slot, sparc.f90:286-287). ``irec`` is the 0-based record
        # index used by every seek above (``irec * 4 * recl`` is the head of
        # record ``irec + 1``), so starting at 9 and incrementing before the
        # first seek lands the first slab on record 11.
        G = np.zeros((nfreq, nsd, nrd, nk), dtype=np.complex64)
        irec = 9
        for ifreq in range(nfreq):
            for isd in range(nsd):
                for ird in range(nrd):
                    irec += 1
                    f.seek(irec * 4 * recl, 0)
                    data = np.fromfile(f, dtype=f4, count=2 * nk)
                    if data.size < 2 * nk:
                        raise FileFormatError(
                            f"read_grn_file: truncated Green's-function record "
                            f"at ifreq={ifreq}, isd={isd}, ird={ird} "
                            f"(expected {2 * nk} float32 values, got {data.size})"
                        )
                    G[ifreq, isd, ird, :] = data[0::2] + 1j * data[1::2]

    is_sparc = title.upper().startswith('SPARC')

    return {
        "freq": float(freq0),
        "freqVec": freqVec,
        "nfreq": nfreq,
        "nsd": nsd,
        "nrd": nrd,
        "nk": nk,
        "sd": sd,
        "rd": rd,
        "cVec": cVec,
        "atten": float(atten),
        "G": G,
        "title": title,
        "PlotType": PlotType,
        "is_sparc": is_sparc,
    }


def _wavenumbers_for_frequency(grn_data: Dict[str, Any], freq: float) -> np.ndarray:
    """Recover the wavenumber grid the solver used at frequency ``freq``.

    SCOOTER recomputes ``k`` per frequency from the same phase-speed grid
    (``scooter.f90:127``); SPARC's wavenumber grid is constant across the
    output-time axis so we always use the source frequency from the GRN
    header.
    """
    if grn_data["is_sparc"]:
        f_for_k = grn_data["freq"]   # source frequency
    else:
        f_for_k = float(freq)
    cVec = grn_data["cVec"]
    return 2.0 * np.pi * f_for_k / cVec


def _stab_attenuation(grn_data: Dict[str, Any], k: np.ndarray) -> float:
    """Stabilising attenuation to use in the integrand.

    ``fieldsco.m:113-115`` overrides ``atten = Δk`` for SCOOTER (the solver
    writes one ``atten`` to the header, but sets it per frequency as
    ``Atten = Deltak``, ``scooter.f90:129``). The override is unconditional
    there and here, so a run that zeroed the stabilising attenuation with
    ``TopOpt(7:7) = '0'`` (``scooter.f90:130``) still gets ``Δk`` back. For
    SPARC this is 0. For other titles we trust the header value.
    """
    if grn_data["is_sparc"]:
        return 0.0
    title = grn_data["title"].upper()
    if title.startswith('SCOOTER'):
        return float(k[1] - k[0]) if len(k) > 1 else float(grn_data["atten"])
    return float(grn_data["atten"])


def _hanning_taper(k: np.ndarray, freq: float,
                   cmin: Optional[float], cmax: Optional[float]) -> np.ndarray:
    """Build a window that tapers ``G(k)`` outside ``[ω/cmax, ω/cmin]``.

    Mirrors ``fieldsco.m:taper`` — symmetric Hanning roll-offs at the
    spectrum edges, ones in the middle. Returns ``ones`` when both bounds
    are inactive. Raises :class:`ConfigurationError` when the requested
    phase-speed band has no overlap with the file's wavenumber grid —
    the taper would zero the entire spectrum.
    """
    Nk = len(k)
    win = np.ones(Nk, dtype=float)
    if Nk == 0:
        return win

    omega = 2.0 * np.pi * freq
    k_left = omega / cmax if (cmax is not None and cmax > 0) else None
    k_right = omega / cmin if (cmin is not None and cmin > 0) else None

    # The pass band is [ω/cmax, ω/cmin]; the grid spans phase speeds
    # [ω/k[-1], ω/k[0]]. A band that misses the grid entirely would taper
    # every sample to zero (and the roll-off construction below would build
    # a window longer than the grid).
    c_grid_lo = omega / float(k[-1])
    c_grid_hi = omega / float(k[0])
    if k_left is not None and k_right is not None and k_left > k_right:
        raise ConfigurationError(
            f"phase-speed taper: cmin ({cmin:g} m/s) exceeds cmax "
            f"({cmax:g} m/s); the pass band [ω/cmax, ω/cmin] is empty."
        )
    if (k_left is not None and k_left > k[-1]) or \
            (k_right is not None and k_right < k[0]):
        raise ConfigurationError(
            f"phase-speed taper: the requested band "
            f"(cmin={cmin!r}, cmax={cmax!r} m/s) has no overlap with the "
            f"file's phase-speed grid [{c_grid_lo:.1f}, {c_grid_hi:.1f}] m/s "
            f"at {freq:g} Hz — the taper would zero the entire spectrum. "
            f"Widen or drop cmin/cmax."
        )
    if Nk < 4:
        return win

    if k_left is not None and k_left > k[0]:
        n = 2 * round((k_left - k[0]) / (k[-1] - k[0]) * Nk) + 1
        han = np.hanning(n)
        n_half = (n - 1) // 2
        win[:n_half] *= han[:n_half]

    if k_right is not None and k_right < k[-1]:
        n = 2 * round((k[-1] - k_right) / (k[-1] - k[0]) * Nk) + 1
        han = np.hanning(n)
        n_half = (n - 1) // 2
        win[-n_half:] *= han[-n_half:]

    return win


def _zero_range_mask(ranges: np.ndarray) -> np.ndarray:
    """Ranges at which the point-source ``1/√r`` spreading factor is singular.

    Mirrors the ``abs( Rr ) < realmin`` test of ``fieldsco.m:69``.
    """
    return np.abs(np.asarray(ranges, dtype=float)) < np.finfo(float).tiny


def _warn_zero_ranges(ranges: np.ndarray, source_type: str) -> None:
    """Warn once per transform that ``r = 0`` cells come back as no-data."""
    if source_type != 'R':
        return
    n_zero = int(np.count_nonzero(_zero_range_mask(ranges)))
    if n_zero:
        warnings.warn(
            f"{n_zero} receiver range(s) at r = 0: the point-source Hankel "
            "transform carries a 1/sqrt(r) cylindrical-spreading factor that "
            "is singular there, so those cells are returned as NaN (no data). "
            "Move the receiver off the source axis (e.g. r = 1 m) to get a "
            "field value.",
            UserWarning, stacklevel=3)


def _hankel_transform(
    G_src: np.ndarray,
    k: np.ndarray,
    ranges: np.ndarray,
    *,
    atten: float,
    source_type: str = 'R',
    spectrum: str = 'P',
) -> np.ndarray:
    """Wavenumber → range transform for one (source_depth, frequency) slab.

    A direct (trapezoidal-rule) DFT over the uniform ``k`` grid — the matrix
    product ``-G_scaled @ X`` below — not an FFT, matching ``fieldsco.m:5``
    ("This version uses the trapezoidal rule directly to do a DFT, rather
    than an FFT").

    Implements three of ``fieldsco.m``'s four source branches (its ``'H'``
    exact-Bessel branch, ``fieldsco.m:139-144``, is not exposed):

    ============  ==================================================
    source_type   Geometry
    ------------  --------------------------------------------------
    ``'R'``       cylindrical / point source (3-D), ``√(2πr)`` denom
    ``'X'``       Cartesian / line source (2-D), ``√(2π)`` denom
    ``'S'``       point source, cylindrical spreading removed, ``√(2π)`` denom
    ============  ==================================================

    ============  ==================================================
    spectrum      Half / full integration
    ------------  --------------------------------------------------
    ``'P'``       positive branch only (default; recommended)
    ``'N'``       negative branch only
    ``'B'``       both branches summed (full real-axis integral)
    ============  ==================================================

    Parameters
    ----------
    G_src : (nrd, nk) complex
    k     : (nk,) wavenumber grid
    ranges : (nr,) output ranges (m)
    atten : stabilising attenuation (added to k along the +i axis)
    source_type, spectrum : see table above
    """
    if source_type not in ('R', 'X', 'S'):
        raise ConfigurationError(
            f"source_type must be 'R', 'X', or 'S', got {source_type!r}")
    if spectrum not in ('P', 'N', 'B'):
        raise ConfigurationError(f"spectrum must be 'P', 'N', or 'B', got {spectrum!r}")

    dk = float(k[1] - k[0]) if len(k) > 1 else 1.0
    ck = k + 1j * atten
    abs_r = np.abs(ranges)
    x = np.outer(ck, abs_r)

    if source_type == 'X':
        # Line source: no √k weighting, no phase shift, 1/√(2π).
        factor1 = np.ones_like(ck)
        factor2 = dk / np.sqrt(2.0 * np.pi) * np.ones_like(abs_r)
        X_pos = np.exp(-1j * x)
        X_neg = np.exp(+1j * x)
    else:
        # Point source: phase factor exp(±i(kr - π/4)) and √k weighting.
        # 'R' adds 1/√(2πr) cylindrical spreading; 'S' omits it.
        factor1 = np.sqrt(ck)
        if source_type == 'R':
            # 1/√r diverges at r=0. ``fieldsco.m:69`` moves a zero range to
            # 1 m; uacpy reports no-data instead, so a cell is never labelled
            # with a range the field was not evaluated at. Kraken's modal sum
            # skips the same division (``EvaluateMod.f90:71-73``), leaving a
            # bare mode sum there — masking keeps the two models' grids
            # comparable cell by cell.
            with np.errstate(divide='ignore'):
                factor2 = dk / np.sqrt(2.0 * np.pi * abs_r)
            factor2 = np.where(_zero_range_mask(abs_r), np.nan, factor2)
        else:
            factor2 = dk / np.sqrt(2.0 * np.pi) * np.ones_like(abs_r)
        X_pos = np.exp(-1j * (x - np.pi / 4.0))
        X_neg = np.exp(+1j * (x - np.pi / 4.0))

    G_scaled = G_src * factor1[np.newaxis, :]

    if spectrum == 'P':
        Y = -G_scaled @ X_pos
    elif spectrum == 'N':
        Y = -G_scaled @ X_neg
    else:  # 'B'
        Y = -G_scaled @ (X_pos + X_neg)

    return Y * factor2[np.newaxis, :]


def _grn_pressure_slice(
    grn_data: Dict[str, Any],
    ranges: np.ndarray,
    ifreq: int,
    isd: int,
    *,
    source_type: str,
    spectrum: str,
    cmin: Optional[float],
    cmax: Optional[float],
) -> np.ndarray:
    """Transform one (frequency, source_depth) slab to the range domain."""
    G_src = grn_data["G"][ifreq, isd, :, :]                    # (nrd, nk)
    freq_i = float(grn_data["freqVec"][ifreq]) if grn_data["nfreq"] > 0 else grn_data["freq"]
    k = _wavenumbers_for_frequency(grn_data, freq_i)
    atten = _stab_attenuation(grn_data, k)

    if cmin is not None or cmax is not None:
        # Use the source frequency for SPARC (k grid is freq-independent),
        # otherwise the per-frequency value.
        f_for_taper = grn_data["freq"] if grn_data["is_sparc"] else freq_i
        win = _hanning_taper(k, f_for_taper, cmin, cmax)
        G_src = G_src * win[np.newaxis, :]

    return _hankel_transform(
        G_src, k, ranges,
        atten=atten,
        source_type=source_type,
        spectrum=spectrum,
    )


def grn_to_field(
    grn_data: Dict[str, Any],
    ranges: np.ndarray,
    *,
    method: str = "direct_dft",
    source_type: str = 'R',
    spectrum: str = 'P',
    source_depth_idx: int = 0,
    cmin: Optional[float] = None,
    cmax: Optional[float] = None,
) -> Field:
    """Transform a single-frequency Green's function to a complex narrowband Field.

    The reader returns a 4-D ``G`` regardless of ``nfreq``; this picks the
    first frequency slice (use :func:`grn_to_transfer_function` for the
    multi-frequency case).

    Parameters
    ----------
    method : str
        Only ``'direct_dft'`` — the trapezoidal-rule matrix-product DFT of
        :func:`_hankel_transform`.
    source_depth_idx : int
        Index into the source-depth axis when ``nsd > 1``. Defaults to the
        first source.
    source_type, spectrum : see :func:`_hankel_transform`.
    cmin, cmax : optional phase-speed taper bounds (m/s).
    """
    if method != "direct_dft":
        raise ConfigurationError(f"Unknown method: {method!r}. Use 'direct_dft'.")

    nsd = grn_data["nsd"]
    if not (0 <= source_depth_idx < nsd):
        raise ConfigurationError(
            f"source_depth_idx={source_depth_idx} out of range for nsd={nsd}"
        )
    _warn_zero_ranges(ranges, source_type)

    p_out = _grn_pressure_slice(
        grn_data, ranges, ifreq=0, isd=source_depth_idx,
        source_type=source_type, spectrum=spectrum, cmin=cmin, cmax=cmax,
    )

    return Field(
        data=p_out,
        coords={'depth': grn_data["rd"], 'range': ranges},
        model='', backend='',
        # The slab holds one source depth — the one ``source_depth_idx``
        # selected — so it carries that depth alone.
        source_depths=np.atleast_1d(
            np.asarray(grn_data['sd'], dtype=float)[source_depth_idx]),
        frequencies=float(grn_data["freq"]),
        phase_reference='travelling_wave',
        metadata={
            "transform_method": method,
            "source_type": source_type,
            "spectrum": spectrum,
        },
    )


def sparc_snapshot_to_field(
    grn_data: Dict[str, Any],
    ranges: np.ndarray,
    frequency: float,
    *,
    source_type: str = 'R',
    spectrum: str = 'P',
    source_depth_idx: int = 0,
    cmin: Optional[float] = None,
    cmax: Optional[float] = None,
    pulse_type: Optional[str] = None,
    normalize: str = 'source',
) -> Field:
    """Extract steady-state complex pressure at ``frequency`` from a SPARC snapshot.

    ``normalize`` (default ``'source'``): SPARC propagates the *actual* pulse, so
    the snapshot is ``S(omega)*g`` — the source spectrum times the transfer
    function — not the bare ``g`` that Kraken/Scooter report (Jensen,
    *Computational Ocean Acoustics*, Eq. 8.1). ``'source'`` deconvolves the known
    source spectrum: it runs the *same* steady-tone estimator on the pulse
    ``sparc_pulse(tout, 2*pi*frequency, pulse_type)`` to get ``S(omega0)`` and
    divides it out, recovering ``g`` — i.e. **absolute TL re 1 m**, directly
    comparable to Kraken/Scooter. Requires ``pulse_type`` (the SPARC pulse
    alphabet; the ``SPARC`` wrapper passes it). The window and ``2/Σwin`` factor
    cancel exactly; a residual remains only from the source high cut
    ``fHiCut`` (``sparc.f90:397-401``), which ``sparc_pulse`` does not
    replicate — ``rkT·cHigh/2π`` for ``Pulse(4:4)`` in ``'HB'`` and ``10·fMax``
    otherwise. ``fMin``/``fMax`` themselves are **not** a band-pass on the
    output: they set the wavenumber integration limits
    ``kMin = 2π·fMin/cHigh`` and ``kMax = 2π·fMax/cLow`` (``sparc.f90:111-112``).
    ``'none'`` returns the raw (uncalibrated) field and warns.

    SPARC's snapshot mode (``output_mode='S'``) writes the *time evolution*
    of the wavenumber-domain Green's function (``Green(itout, irz, ik)``,
    ``sparc.f90:283-289``). To recover the steady-state pressure at the
    source frequency we:

    1. FFT along the snapshot-time axis to obtain :math:`G(f, k, z)`.
    2. Pick the bin closest to the source ``frequency``.
    3. Hankel-transform :math:`G(k, z)` to range.

    Returns a complex narrowband :class:`Field` (``coords={'depth',
    'range'}``); use ``.tl`` or ``.to_tl()`` for transmission loss in dB.
    """
    if not grn_data["is_sparc"]:
        raise ConfigurationError(
            "sparc_snapshot_to_field expects a SPARC GRN; got title "
            f"{grn_data['title']!r} (no 'SPARC' prefix)."
        )

    if normalize not in ('source', 'none'):
        raise ConfigurationError(
            "sparc_snapshot_to_field: normalize must be 'source' or 'none'; "
            f"got {normalize!r}")
    if normalize == 'source' and not pulse_type:
        raise ConfigurationError(
            "sparc_snapshot_to_field: normalize='source' needs pulse_type (the "
            "SPARC pulse alphabet) to deconvolve the source spectrum. Pass it, "
            "or use normalize='none' for the raw (uncalibrated) field.")
    if normalize == 'none':
        warnings.warn(
            "sparc_snapshot_to_field: normalize='none' returns the RAW field "
            "(S(omega)*g), whose absolute level is uncalibrated — a "
            "pulse-dependent offset (tens of dB) above calibrated TL (Jensen, "
            "Computational Ocean Acoustics, Eq. 8.1). Use normalize='source' "
            "(with pulse_type) for calibrated absolute TL re 1 m, or treat only "
            "the field SHAPE as indicative.",
            UserWarning, stacklevel=2)

    nsd = grn_data["nsd"]
    if not (0 <= source_depth_idx < nsd):
        raise ConfigurationError(
            f"source_depth_idx={source_depth_idx} out of range for nsd={nsd}"
        )

    _warn_zero_ranges(ranges, source_type)
    G = grn_data["G"][:, source_depth_idx, :, :]   # (nt, nrd, nk)
    tout = grn_data["freqVec"]                      # actually the time vector
    nt = len(tout)
    if nt < 2:
        raise ConfigurationError(
            "SPARC snapshot has nt<2 — cannot extract a frequency component "
            "via time-FFT. Use a larger n_t_out."
        )
    dt = float(tout[1] - tout[0])

    # Steady-tone amplitude estimator 2·X_k/Σwin (mirrors rts_to_pressure for
    # the 'R'/'D' modes). This yields S(w0)·g; normalize='source' (default)
    # divides out the source spectrum S(w0) below to recover calibrated g.
    win = np.hanning(nt)
    G_freq = np.fft.fft(G * win[:, np.newaxis, np.newaxis], axis=0)
    fft_freqs = np.fft.fftfreq(nt, dt)
    nyquist = 0.5 / dt
    if frequency > nyquist:
        raise ConfigurationError(
            f"Source frequency {frequency:.3f} Hz exceeds the snapshot's "
            f"Nyquist {nyquist:.3f} Hz; reduce dt by raising n_t_out or "
            "shortening t_max."
        )
    f_idx = int(np.argmin(np.abs(fft_freqs - frequency)))

    if normalize == 'source':
        # Convolution theorem: the snapshot G(t) = s(t) ⊛ h(t) (source pulse
        # convolved with the medium response), so DFT(G)/DFT(s) = h(w0) — the
        # unit-source wavenumber Green's function Scooter computes, absolute-
        # calibrated (Jensen COA Eq. 8.1). Use the RECTANGULAR full DFT for
        # both: a taper would break the convolution theorem and would null the
        # transient source pulse (which lives in the first few samples, where a
        # Hann window is ~0). uacpy generated the pulse, so s(t) is known.
        s_t, _ = sparc_pulse(tout, 2.0 * np.pi * frequency, pulse_type[0])
        S_at_f0 = np.fft.fft(s_t)[f_idx]
        if S_at_f0 == 0:
            raise ConfigurationError(
                "sparc_snapshot_to_field: source spectrum is zero at "
                f"{frequency} Hz for pulse_type={pulse_type!r}; cannot "
                "deconvolve (check pulse / frequency).")
        G_at_f0 = np.fft.fft(G, axis=0)[f_idx, :, :] / S_at_f0
    else:
        G_at_f0 = 2.0 * G_freq[f_idx, :, :] / np.sum(win)   # (nrd, nk) = S·g

    # Wavenumber grid — SPARC's k vector is independent of frequency.
    k = _wavenumbers_for_frequency(grn_data, frequency)
    atten = _stab_attenuation(grn_data, k)          # 0 for SPARC

    if cmin is not None or cmax is not None:
        win = _hanning_taper(k, frequency, cmin, cmax)
        G_at_f0 = G_at_f0 * win[np.newaxis, :]

    p_out = _hankel_transform(
        G_at_f0, k, ranges,
        atten=atten, source_type=source_type, spectrum=spectrum,
    )
    if normalize == 'none':
        # Put the RAW field on the full inverse-Hankel weight
        # Δk·√(2k/(πr)) that sparc.f90's 'D' branch carries (:595 kernel with
        # the 1/√(π·Rr) write scale at :292) — the fieldsco-style Hankel above
        # carries 1/√(2πr) and the Scooter −1 prefactor instead, a constant −2
        # between the two. The calibrated path SKIPS this: after deconvolution
        # G_at_f0 is the Scooter unit-source Green's function, so the bare
        # Hankel (as in _grn_pressure_slice) already matches Scooter/Kraken.
        p_out = p_out * (-2.0)

    return Field(
        data=p_out,
        coords={'depth': grn_data["rd"], 'range': ranges},
        model='', backend='',
        # The slab holds one source depth — the one ``source_depth_idx``
        # selected — so it carries that depth alone.
        source_depths=np.atleast_1d(
            np.asarray(grn_data['sd'], dtype=float)[source_depth_idx]),
        frequencies=float(frequency),
        phase_reference='travelling_wave',
        metadata={
            "transform_method": "time_fft+hankel",
            "normalize": normalize,
            "absolute_tl_calibrated": normalize == 'source',
            "snapshot_freq_bin": float(fft_freqs[f_idx]),
            "snapshot_dt": dt,
            "snapshot_nt": nt,
            "source_type": source_type,
            "spectrum": spectrum,
        },
    )


def sparc_snapshot_to_time_field(
    grn_data: Dict[str, Any],
    ranges: np.ndarray,
    frequency: float,
    *,
    source_type: str = 'R',
    spectrum: str = 'P',
    source_depth_idx: int = 0,
) -> Field:
    """Range-domain time evolution of a SPARC snapshot (``output_mode='S'``).

    ``sparc.f90:580-591`` writes ``Green(Itout, irz, ik)`` — the
    *wavenumber-domain* field at each output time — and ``WriteHeaderSparc``
    (``:317-327``) stores the time vector in the ``.grn``'s frequency slot and
    the phase-speed vector ``sqrt(omega2)/k`` in its range slot.
    ``doc/sparc.htm`` prescribes running FIELDS afterwards "to convert the
    '.GRN' file to a '.SHD' file containing the pressure field"; this is that
    step done in-tree.

    Simpler than :func:`sparc_snapshot_to_field`, which recovers a *CW*
    component: the snapshot already is the propagated pulse, so one inverse
    Hankel transform per output time gives ``p(z, r, t)`` directly — no
    time-FFT, no frequency selection, and no source deconvolution (exactly as
    the ``'R'``/``'D'`` modes return their raw received time series).

    The wavenumber grid is taken at the source frequency and is **constant
    across the time axis** — SPARC's ``k`` does not scale with the ``.grn``'s
    nominal frequency axis, because that axis holds times.

    Returns a real :class:`Field` with ``coords={'depth', 'range', 'time'}``.
    """
    if not grn_data["is_sparc"]:
        raise ConfigurationError(
            "sparc_snapshot_to_time_field expects a SPARC GRN; got title "
            f"{grn_data['title']!r} (no 'SPARC' prefix)."
        )
    nsd = grn_data["nsd"]
    if not (0 <= source_depth_idx < nsd):
        raise ConfigurationError(
            f"source_depth_idx={source_depth_idx} out of range for nsd={nsd}"
        )

    _warn_zero_ranges(ranges, source_type)
    G = grn_data["G"][:, source_depth_idx, :, :]   # (nt, nrd, nk)
    tout = np.asarray(grn_data["freqVec"], dtype=float)   # times, not freqs
    k = _wavenumbers_for_frequency(grn_data, frequency)
    atten = _stab_attenuation(grn_data, k)                # 0 for SPARC
    ranges = np.atleast_1d(np.asarray(ranges, dtype=float))

    # (nrd, n_ranges) per output time -> (nrd, n_ranges, nt).
    p_t = np.stack(
        [_hankel_transform(G[it], k, ranges, atten=atten,
                           source_type=source_type, spectrum=spectrum)
         for it in range(G.shape[0])],
        axis=-1,
    )
    # Put the snapshot on the same inverse-Hankel weight the 'D' branch uses,
    # dk*sqrt(2k/(pi*r)) (sparc.f90:595 with the 1/sqrt(pi*Rr) write scale at
    # :292). The fieldsco-style Hankel above carries 1/sqrt(2*pi*r) and the
    # Scooter -1 prefactor instead; the constant between the two is -2.
    p_t = p_t * (-2.0)

    # The snapshot is a real transient field; the Hankel transform carries the
    # analytic-signal convention, so the physical pressure is its real part.
    dt = float(tout[1] - tout[0]) if tout.size > 1 else float('nan')

    return Field(
        data=np.real(p_t),
        coords={'depth': grn_data["rd"], 'range': ranges, 'time': tout},
        model='', backend='',
        # The slab holds one source depth — the one ``source_depth_idx``
        # selected — so it carries that depth alone.
        source_depths=np.atleast_1d(
            np.asarray(grn_data['sd'], dtype=float)[source_depth_idx]),
        frequencies=float(frequency),
        phase_reference='time_domain_native',
        metadata={
            "transform_method": "hankel_per_snapshot_time",
            "dt": dt,
            "fs": (1.0 / dt) if dt == dt and dt else float('nan'),
            "nt": int(tout.size),
            "t_start": float(tout[0]) if tout.size else 0.0,
            "source_type": source_type,
            "spectrum": spectrum,
        },
    )


def grn_to_transfer_function(
    grn_data: Dict[str, Any],
    ranges: np.ndarray,
    *,
    source_type: str = 'R',
    spectrum: str = 'P',
    source_depth_idx: int = 0,
    cmin: Optional[float] = None,
    cmax: Optional[float] = None,
) -> Field:
    """Transform a multi-frequency Green's function to a broadband Field.

    Output: complex ``Field`` with ``coords={'depth', 'range',
    'frequency'}``, shape ``(n_d, n_r, n_f)``.
    """
    nfreq = grn_data["nfreq"]
    nrd = grn_data["nrd"]
    nsd = grn_data["nsd"]
    if not (0 <= source_depth_idx < nsd):
        raise ConfigurationError(
            f"source_depth_idx={source_depth_idx} out of range for nsd={nsd}"
        )

    _warn_zero_ranges(ranges, source_type)
    pressure = np.zeros((nrd, len(ranges), nfreq), dtype=np.complex128)
    for ifreq in range(nfreq):
        pressure[:, :, ifreq] = _grn_pressure_slice(
            grn_data, ranges, ifreq=ifreq, isd=source_depth_idx,
            source_type=source_type, spectrum=spectrum, cmin=cmin, cmax=cmax,
        )

    freqVec = np.asarray(grn_data["freqVec"], dtype=float)
    return Field(
        data=pressure,
        coords={
            'depth': grn_data["rd"],
            'range': ranges,
            'frequency': freqVec,
        },
        phase_reference='travelling_wave',
        model='', backend='',
        # The slab holds one source depth — the one ``source_depth_idx``
        # selected — so it carries that depth alone.
        source_depths=np.atleast_1d(
            np.asarray(grn_data['sd'], dtype=float)[source_depth_idx]),
        frequencies=freqVec,
        metadata={
            'center_frequency': float(freqVec[len(freqVec) // 2]),
            'nfreq': nfreq,
            'source_type': source_type,
            'spectrum': spectrum,
        },
    )
