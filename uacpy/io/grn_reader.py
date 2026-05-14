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
  header (``fieldsco.m:97-100``); ``freqVec`` actually stores the output
  *time* vector (``sparc.f90:319``), not frequencies.

We detect SPARC by the ``'SPARC'`` prefix in the title (set at
``sparc.f90:84``).
"""

import numpy as np
import struct
from pathlib import Path
from typing import Union, Dict, Any, Optional

from uacpy.core.results import Field
from uacpy.io._fortran_helpers import detect_endian


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
        i4 = endian + 'i'
        d8 = endian + 'd'
        f4 = endian + 'f4'
        f8 = endian + 'f8'

        # Record 1: recl (int32, in 4-byte words) + title (80 chars)
        recl = struct.unpack(i4, f.read(4))[0]
        title = f.read(80).decode("utf-8", errors="ignore").strip()

        f.seek(4 * recl, 0)

        # Record 2: PlotType (10 chars)
        PlotType = f.read(10).decode("utf-8", errors="ignore").strip()

        f.seek(2 * 4 * recl, 0)

        # Record 3: 7 int32 + freq0 (float64) + atten (float64)
        nfreq = struct.unpack(i4, f.read(4))[0]
        struct.unpack(i4, f.read(4))[0]
        struct.unpack(i4, f.read(4))[0]
        struct.unpack(i4, f.read(4))[0]
        nsd = struct.unpack(i4, f.read(4))[0]   # NSz
        nrd = struct.unpack(i4, f.read(4))[0]   # NRz
        nk = struct.unpack(i4, f.read(4))[0]    # NRr — number of k samples
        freq0 = struct.unpack(d8, f.read(8))[0]
        atten = struct.unpack(d8, f.read(8))[0]

        f.seek(3 * 4 * recl, 0)

        # Record 4: frequency vector (or time vector for SPARC snapshot)
        freqVec = np.frombuffer(f.read(nfreq * 8), dtype=f8).copy()

        # Records 5-7: theta / sx / sy — skipped
        f.seek(7 * 4 * recl, 0)

        # Record 8: Source depths (float32)
        sd = np.frombuffer(f.read(nsd * 4), dtype=f4).copy()

        f.seek(8 * 4 * recl, 0)

        # Record 9: Receiver depths (float32)
        rd = np.frombuffer(f.read(nrd * 4), dtype=f4).copy()

        f.seek(9 * 4 * recl, 0)

        # Record 10: phase-speed vector (float64), Nk entries.
        cVec = np.frombuffer(f.read(nk * 8), dtype=f8).copy()

        # Records 11+: complex Green's function, one record per
        # (freq, source_depth, receiver_depth) tuple.
        G = np.zeros((nfreq, nsd, nrd, nk), dtype=np.complex64)
        irec = 9
        for ifreq in range(nfreq):
            for isd in range(nsd):
                for ird in range(nrd):
                    irec += 1
                    f.seek(irec * 4 * recl, 0)
                    raw = f.read(nk * 8)
                    if len(raw) < nk * 8:
                        break
                    data = np.frombuffer(raw, dtype=f4)
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

    ``fieldsco.m:110-112`` overrides ``atten = Δk`` for SCOOTER (the solver
    only writes one ``atten`` value to the header; for broadband each
    frequency has its own ``Δk`` so we recompute). For SPARC this is 0.
    For other titles we trust the value stored in the header.
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
    are inactive.
    """
    Nk = len(k)
    win = np.ones(Nk, dtype=float)
    if Nk < 4:
        return win

    omega = 2.0 * np.pi * freq
    k_left = omega / cmax if (cmax is not None and cmax > 0) else None
    k_right = omega / cmin if (cmin is not None and cmin > 0) else None

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

    Implements the four ``fieldsco.m`` branches we expose:

    ============  ==================================================
    source_type   Geometry
    ------------  --------------------------------------------------
    ``'R'``       cylindrical / point source (3-D), ``√(2πr)`` denom
    ``'X'``       Cartesian / line source (2-D), ``√(2π)`` denom
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
    if source_type not in ('R', 'X'):
        raise ValueError(f"source_type must be 'R' or 'X', got {source_type!r}")
    if spectrum not in ('P', 'N', 'B'):
        raise ValueError(f"spectrum must be 'P', 'N', or 'B', got {spectrum!r}")

    dk = float(k[1] - k[0]) if len(k) > 1 else 1.0
    ck = k + 1j * atten
    abs_r = np.abs(ranges)

    if source_type == 'R':
        # Point source: phase factor exp(±i(kr - π/4)), √k weighting,
        # 1/√(2πr) cylindrical spreading.
        x = np.outer(ck, abs_r)
        factor1 = np.sqrt(ck)
        factor2 = dk / np.sqrt(2.0 * np.pi * np.maximum(abs_r, np.finfo(float).tiny))
        X_pos = np.exp(-1j * (x - np.pi / 4.0))
        X_neg = np.exp(+1j * (x - np.pi / 4.0))
    else:  # 'X' — line source
        x = np.outer(ck, abs_r)
        factor1 = np.ones_like(ck)
        factor2 = dk / np.sqrt(2.0 * np.pi) * np.ones_like(abs_r)
        X_pos = np.exp(-1j * x)
        X_neg = np.exp(+1j * x)

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
    method: str = "fft_hankel",
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
    source_depth_idx : int
        Index into the source-depth axis when ``nsd > 1``. Defaults to the
        first source.
    source_type, spectrum : see :func:`_hankel_transform`.
    cmin, cmax : optional phase-speed taper bounds (m/s).
    """
    if method != "fft_hankel":
        raise ValueError(f"Unknown method: {method!r}. Use 'fft_hankel'.")

    nsd = grn_data["nsd"]
    if not (0 <= source_depth_idx < nsd):
        raise IndexError(
            f"source_depth_idx={source_depth_idx} out of range for nsd={nsd}"
        )

    p_out = _grn_pressure_slice(
        grn_data, ranges, ifreq=0, isd=source_depth_idx,
        source_type=source_type, spectrum=spectrum, cmin=cmin, cmax=cmax,
    )

    return Field(
        data=p_out,
        coords={'depth': grn_data["rd"], 'range': ranges},
        model='', backend='',
        source_depths=np.atleast_1d(np.asarray(grn_data['sd'], dtype=float)),
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
) -> Field:
    """Extract steady-state complex pressure at ``frequency`` from a SPARC snapshot.

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
        raise ValueError(
            "sparc_snapshot_to_field expects a SPARC GRN; got title "
            f"{grn_data['title']!r} (no 'SPARC' prefix)."
        )

    nsd = grn_data["nsd"]
    if not (0 <= source_depth_idx < nsd):
        raise IndexError(
            f"source_depth_idx={source_depth_idx} out of range for nsd={nsd}"
        )

    G = grn_data["G"][:, source_depth_idx, :, :]   # (nt, nrd, nk)
    tout = grn_data["freqVec"]                      # actually the time vector
    nt = len(tout)
    if nt < 2:
        raise ValueError(
            "SPARC snapshot has nt<2 — cannot extract a frequency component "
            "via time-FFT. Use a larger n_t_out."
        )
    dt = float(tout[1] - tout[0])

    G_freq = np.fft.fft(G, axis=0) * dt              # scale to spectral density
    fft_freqs = np.fft.fftfreq(nt, dt)
    nyquist = 0.5 / dt
    if frequency > nyquist:
        raise ValueError(
            f"Source frequency {frequency:.3f} Hz exceeds the snapshot's "
            f"Nyquist {nyquist:.3f} Hz; reduce dt by raising n_t_out or "
            "shortening t_max."
        )
    f_idx = int(np.argmin(np.abs(fft_freqs - frequency)))
    # One-sided spectrum from a real time series — multiply by 2 to
    # recover full amplitude (matches rts_to_pressure at
    # oalib_reader.py:1331). The DC bin is the only one that should
    # not be doubled, but f_idx > 0 for any physical source frequency.
    G_at_f0 = 2.0 * G_freq[f_idx, :, :]              # (nrd, nk)

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

    return Field(
        data=p_out,
        coords={'depth': grn_data["rd"], 'range': ranges},
        model='', backend='',
        source_depths=np.atleast_1d(np.asarray(grn_data['sd'], dtype=float)),
        frequencies=float(frequency),
        phase_reference='travelling_wave',
        metadata={
            "transform_method": "time_fft+hankel",
            "snapshot_freq_bin": float(fft_freqs[f_idx]),
            "snapshot_dt": dt,
            "snapshot_nt": nt,
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
        raise IndexError(
            f"source_depth_idx={source_depth_idx} out of range for nsd={nsd}"
        )

    pressure = np.zeros((nrd, len(ranges), nfreq), dtype=np.complex64)
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
        source_depths=np.atleast_1d(np.asarray(grn_data['sd'], dtype=float)),
        frequencies=freqVec,
        metadata={
            'center_frequency': float(freqVec[len(freqVec) // 2]),
            'nfreq': nfreq,
            'source_type': source_type,
            'spectrum': spectrum,
        },
    )
