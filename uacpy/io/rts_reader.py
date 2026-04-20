"""
Time series reader for SPARC model
"""

import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Tuple


def read_rts_file(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Read SPARC time series file (.rts).

    SPARC computes pressure time series at receiver locations.
    This data must be transformed to frequency domain for TL calculations.

    Parameters
    ----------
    filepath : str or Path
        Path to .rts file

    Returns
    -------
    rts_data : dict
        Dictionary containing:
        - 'title': Run title
        - 'freq': Reference frequency in Hz (placeholder)
        - 'dt': Time step in seconds
        - 'nt': Number of time samples
        - 'nr': Number of ranges/depths
        - 'ranges': Range/depth vector (m)
        - 'time': Time vector (s)
        - 'p': Pressure time series, shape (nt, nr)

    Notes
    -----
    SPARC outputs time-domain pressure fields which must be FFT'd
    to extract frequency-domain transmission loss.

    File format is Fortran ASCII (FORMATTED).
    Line 1: Title (in quotes)
    Line 2: NRr/NRz, then range/depth values
    Subsequent lines: time, then pressure values (12G15.6 format)
    """
    filepath = Path(filepath)

    with open(filepath, "r") as f:
        # Line 1: Title
        line1 = f.readline().strip()
        # Remove surrounding quotes
        if line1.startswith("'") and line1.endswith("'"):
            title = line1[1:-1]
        else:
            title = line1

        # Line 2: Nr and range/depth values
        line2 = f.readline().strip().split()
        nr = int(line2[0])
        ranges = np.array([float(x) for x in line2[1:nr+1]])

        # Read all time series data
        # Note: Fortran format 12G15.6 means 12 values per line,
        # so data may wrap across multiple lines
        time_list = []
        pressure_list = []

        all_values = []
        for line in f:
            values = [float(x) for x in line.strip().split()]
            all_values.extend(values)

        # Each time step has: 1 time value + nr pressure values
        values_per_timestep = 1 + nr
        nt = len(all_values) // values_per_timestep

        for i in range(nt):
            start_idx = i * values_per_timestep
            time_list.append(all_values[start_idx])
            pressure_list.append(all_values[start_idx+1:start_idx+1+nr])

    nt = len(time_list)
    time = np.array(time_list)
    p = np.array(pressure_list)  # shape (nt, nr)

    # Calculate dt
    if nt > 1:
        dt = time[1] - time[0]
    else:
        dt = 0.0

    # Frequency is not stored in RTS file - use placeholder
    freq = 0.0

    return {
        "title": title,
        "freq": freq,  # Not available in RTS file
        "dt": dt,
        "nt": nt,
        "nr": nr,
        "ranges": ranges,
        "time": time,
        "p": p,
    }


def rts_to_tl(rts_data: Dict[str, Any], freq: float, method: str = "fft") -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert time series to transmission loss at specified frequency.

    Parameters
    ----------
    rts_data : dict
        Time series data from read_rts_file()
    freq : float
        Frequency to extract in Hz
    method : str, optional
        Method: 'fft' or 'goertzel'. Default is 'fft'.

    Returns
    -------
    tl : ndarray
        Transmission loss in dB, shape (nr,)
    ranges : ndarray
        Range vector in meters

    Notes
    -----
    Uses FFT to transform time series to frequency domain, then extracts
    the amplitude at the specified frequency.

    The FFT approach:
    1. Apply window to time series (Hanning)
    2. FFT to frequency domain
    3. Find bin closest to target frequency
    4. Extract amplitude
    5. Convert to TL: TL = -20*log10(|p|)
    """
    p = rts_data["p"]
    dt = rts_data["dt"]
    time = rts_data["time"]
    ranges = rts_data["ranges"]

    nt, nr = p.shape

    if method == "fft":
        # Apply Hanning window to reduce spectral leakage
        window = np.hanning(nt)

        # FFT for each range
        p_freq = np.fft.rfft(p * window[:, np.newaxis], axis=0)

        # Frequency vector
        freqs = np.fft.rfftfreq(nt, dt)

        # Find frequency bin closest to target
        freq_idx = np.argmin(np.abs(freqs - freq))

        # Extract amplitude at target frequency
        p_at_freq = p_freq[freq_idx, :]

        # Apply correction for window energy
        window_correction = np.sum(window) / nt

        # Normalize amplitude
        p_at_freq = p_at_freq / (nt * window_correction)

    elif method == "goertzel":
        # Goertzel algorithm for single-frequency extraction
        # More efficient than FFT when only one frequency is needed
        omega = 2 * np.pi * freq
        coeff = 2 * np.cos(omega * dt)

        p_at_freq = np.zeros(nr, dtype=complex)

        for ir in range(nr):
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0

            for it in range(nt):
                s0 = p[it, ir] + coeff * s1 - s2
                s2 = s1
                s1 = s0

            # Final calculation
            p_at_freq[ir] = s0 - s1 * np.exp(-1j * omega * dt)

        # Normalize
        p_at_freq = p_at_freq / nt

    else:
        raise ValueError(f"Unknown method: {method}")

    # Convert to transmission loss
    tl = -20 * np.log10(np.abs(p_at_freq) + 1e-37)

    return tl, ranges
