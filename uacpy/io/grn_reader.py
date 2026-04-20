"""
Green's function reader for Scooter FFP model
"""

import numpy as np
import struct
from pathlib import Path
from typing import Union, Dict, Any

from uacpy.core.field import Field
from uacpy.core.constants import TL_FLOOR_PRESSURE


def read_grn_file(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Read Scooter Green's function file (.grn).

    Scooter computes Green's functions in the wavenumber domain, which
    must be transformed to range domain for field calculations.

    Parameters
    ----------
    filepath : str or Path
        Path to .grn file

    Returns
    -------
    grn_data : dict
        Dictionary containing:
        - 'freq': Frequency in Hz
        - 'nsd': Number of source depths
        - 'nrd': Number of receiver depths
        - 'nk': Number of wavenumbers
        - 'sd': Source depths (m)
        - 'rd': Receiver depths (m)
        - 'k': Wavenumber vector (1/m)
        - 'G': Green's function array, shape (nsd, nrd, nk)

    Notes
    -----
    The Green's function is in the wavenumber domain and needs to be
    transformed to range domain using Hankel transform or FFT.

    File format is Fortran direct access binary (same as SHD files).
    Uses fixed-length records with RECL in 4-byte words.
    """
    filepath = Path(filepath)

    with open(filepath, "rb") as f:
        # Record 1: recl (int32) + title (80 chars)
        # RECL is record length in 4-byte words, not bytes
        recl = struct.unpack("<i", f.read(4))[0]
        title = f.read(80).decode("utf-8", errors="ignore").strip()

        # Skip to end of record 1
        f.seek(4 * recl, 0)

        # Record 2: PlotType (10 chars)
        PlotType = f.read(10).decode("utf-8", errors="ignore").strip()

        # Skip to end of record 2
        f.seek(2 * 4 * recl, 0)

        # Record 3: Header parameters
        nfreq = struct.unpack("<i", f.read(4))[0]
        ntheta = struct.unpack("<i", f.read(4))[0]
        nsx = struct.unpack("<i", f.read(4))[0]
        nsy = struct.unpack("<i", f.read(4))[0]
        nsd = struct.unpack("<i", f.read(4))[0]  # NSz - number of source depths
        nrd = struct.unpack("<i", f.read(4))[0]  # NRz - number of receiver depths
        nk = struct.unpack("<i", f.read(4))[0]   # NRr - for GRN, this is number of wavenumbers
        freq0 = struct.unpack("<d", f.read(8))[0]
        atten = struct.unpack("<d", f.read(8))[0]

        # Skip to end of record 3
        f.seek(3 * 4 * recl, 0)

        # Record 4: frequency vector
        freqVec = np.frombuffer(f.read(nfreq * 8), dtype="<f8")
        freq = freqVec[0] if len(freqVec) > 0 else freq0

        # Skip to end of record 7 (skip theta, sx, sy records)
        f.seek(7 * 4 * recl, 0)

        # Record 8: Source depths (float32)
        sd = np.frombuffer(f.read(nsd * 4), dtype="<f4")

        # Skip to end of record 8
        f.seek(8 * 4 * recl, 0)

        # Record 9: Receiver depths (float32)
        rd = np.frombuffer(f.read(nrd * 4), dtype="<f4")

        # Skip to end of record 9
        f.seek(9 * 4 * recl, 0)

        # Record 10: For GRN files, this is the phase speed vector (float64)
        k_or_c = np.frombuffer(f.read(nk * 8), dtype="<f8")

        # Read Green's function data starting at record 10 (irec=10, which is the 11th record, 0-indexed)
        # Data is organized as: for each (frequency, source_x, source_y, source_z, receiver_z)
        # there's one record containing data for all wavenumbers
        G = np.zeros((nfreq, nsd, nrd, nk), dtype=np.complex64)

        irec = 9  # Start at 9, will increment to 10 (the 11th record, 0-indexed)
        for ifreq in range(nfreq):
            for isd in range(nsd):
                for ird in range(nrd):
                    irec += 1  # Now points to record 10, 11, 12, ...
                    f.seek(irec * 4 * recl, 0)
                    # Read complex data (2 * nk float32 values)
                    raw_data = f.read(nk * 8)
                    if len(raw_data) < nk * 8:
                        break
                    data = np.frombuffer(raw_data, dtype="<f4")
                    # Interleaved real/imaginary format
                    G[ifreq, isd, ird, :] = data[0::2] + 1j * data[1::2]

        # For single-frequency backward compatibility, also compute k from first freq
        # For multi-frequency, k is based on highest frequency (matches Scooter Fortran)
        if nfreq > 1:
            k_freq = freqVec[-1]  # Scooter uses highest freq for k grid
        else:
            k_freq = freq

        if k_freq > 0:
            k = 2 * np.pi * k_freq / k_or_c
        else:
            k = k_or_c

    return {
        "freq": freq,
        "freqVec": freqVec,
        "nfreq": nfreq,
        "nsd": nsd,
        "nrd": nrd,
        "nk": nk,
        "sd": sd,
        "rd": rd,
        "k": k,
        "k_or_c": k_or_c,
        "G": G[0] if nfreq == 1 else G,  # Backward compat: (nsd,nrd,nk) for single freq
        "G_all": G,  # Always (nfreq, nsd, nrd, nk)
        "title": title,
        "PlotType": PlotType,
    }


def _hankel_transform(G_src: np.ndarray, k: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """
    Hankel transform from wavenumber to range domain.

    Implements the fieldsco.m algorithm from the Acoustics Toolbox:
    matrix-based DFT with sqrt(k) weighting and cylindrical spreading.

    Parameters
    ----------
    G_src : ndarray, shape (nrd, nk)
        Green's function for one source depth at one frequency
    k : ndarray, shape (nk,)
        Wavenumber vector
    ranges : ndarray, shape (nr,)
        Output ranges in meters

    Returns
    -------
    p_out : ndarray, shape (nrd, nr)
        Complex pressure field
    """
    dk = k[1] - k[0] if len(k) > 1 else 1.0
    atten = dk  # Stabilization = delta_k

    ck = k + 1j * atten
    x = np.outer(ck, np.abs(ranges))
    X = np.exp(-1j * (x - np.pi / 4.0))

    factor1 = np.sqrt(ck)
    G_scaled = G_src * factor1[np.newaxis, :]
    Y = -G_scaled @ X

    factor2 = dk / np.sqrt(2.0 * np.pi * np.abs(ranges))
    return Y * factor2[np.newaxis, :]


def grn_to_field(grn_data: Dict[str, Any], ranges: np.ndarray, method: str = "fft_hankel") -> Field:
    """
    Transform Green's function to range-domain TL field (single frequency).

    Parameters
    ----------
    grn_data : dict
        Green's function data from read_grn_file()
    ranges : ndarray
        Desired output ranges in meters
    method : str, optional
        Transform method: 'fft_hankel' (default).

    Returns
    -------
    field : Field
        Transmission loss field
    """
    k = grn_data["k"]
    G = grn_data["G"]
    rd = grn_data["rd"]
    freq = grn_data["freq"]

    # Use first source depth; handle both 3D (nsd,nrd,nk) and 4D (nfreq,...) shapes
    if G.ndim == 4:
        G_src = G[0, 0, :, :]  # first freq, first source
    else:
        G_src = G[0, :, :]  # first source

    if method != "fft_hankel":
        raise ValueError(f"Unknown method: {method}. Use 'fft_hankel'.")

    p_out = _hankel_transform(G_src, k, ranges)

    tl = -20 * np.log10(np.abs(p_out) + TL_FLOOR_PRESSURE)

    return Field(
        field_type="tl",
        data=tl,
        ranges=ranges,
        depths=rd,
        metadata={
            "model": "Scooter",
            "frequency": freq,
            "transform_method": method,
        },
    )


def grn_to_transfer_function(grn_data: Dict[str, Any], ranges: np.ndarray) -> Field:
    """
    Transform multi-frequency Green's function to complex transfer function.

    For broadband Scooter runs, converts G(freq, source, receiver_depth, k)
    to pressure(receiver_depth, freq, range) via Hankel transform at each frequency.

    Parameters
    ----------
    grn_data : dict
        Green's function data from read_grn_file() with nfreq > 1
    ranges : ndarray
        Desired output ranges in meters

    Returns
    -------
    field : Field
        Transfer function field with data shape (nrd, nfreq, nr)
    """
    G_all = grn_data["G_all"]  # (nfreq, nsd, nrd, nk)
    rd = grn_data["rd"]
    freqVec = grn_data["freqVec"]
    k_or_c = grn_data["k_or_c"]  # phase speed vector (same grid for all freqs)
    nfreq = grn_data["nfreq"]
    nrd = grn_data["nrd"]
    nr = len(ranges)

    # Output: pressure(nrd, nfreq, nr)
    pressure = np.zeros((nrd, nfreq, nr), dtype=np.complex64)

    for ifreq in range(nfreq):
        freq_i = freqVec[ifreq]
        # Convert phase speeds to wavenumbers for this frequency
        k_i = 2 * np.pi * freq_i / k_or_c

        # Green's function for first source at this frequency
        G_src = G_all[ifreq, 0, :, :]  # (nrd, nk)

        # Hankel transform
        p_i = _hankel_transform(G_src, k_i, ranges)
        pressure[:, ifreq, :] = p_i

    return Field(
        field_type='transfer_function',
        data=pressure,
        ranges=ranges,
        depths=rd,
        frequencies=freqVec,
        metadata={
            'model': 'Scooter',
            'nfreq': nfreq,
            'center_frequency': freqVec[len(freqVec) // 2],
        },
    )
