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
        - 'dt': Time step in seconds
        - 'nt': Number of time samples
        - 'nr': Number of ranges/depths
        - 'ranges': Range/depth vector (m)
        - 'time': Time vector (s)
        - 'p': Pressure time series, shape (nt, nr)

    Notes
    -----
    SPARC outputs time-domain pressure fields which must be FFT'd
    to extract frequency-domain transmission loss. The RTS file does
    NOT store the analysis frequency; callers must pass it explicitly
    to :func:`rts_to_tl`.

    File format is Fortran ASCII (FORMATTED), written by SPARC's output
    routine (``Scooter/sparc.f90``):

    - Line 1: Title, enclosed in single quotes.
    - Subsequent whitespace-separated token stream:
        * token 0: NRr (or NRz in vertical-array mode), an integer.
        * tokens 1..NRr: range (or depth) values in metres.
        * then repeating blocks of ``1 + NRr`` tokens:
          ``t, p(r_1, t), ..., p(r_NRr, t)``.

    Fortran writes these with ``12G15.6`` formatting, so the tokens wrap
    to a new line every 12 values. The parser tokenises the whole stream
    and is therefore insensitive to line wrapping.
    """
    filepath = Path(filepath)

    # Tokenize the entire file. Fortran's 12G15.6 format wraps at 12
    # values per line, so NRr > 12 causes the range vector to span
    # multiple lines. Flattening the whole stream and walking token by
    # token makes parsing independent of line wrapping.
    with open(filepath, "r") as f:
        line1 = f.readline().strip()
        if line1.startswith("'") and line1.endswith("'"):
            title = line1[1:-1]
        else:
            title = line1

        # Read remaining tokens as a flat stream.
        raw_tokens = []
        for line in f:
            raw_tokens.extend(line.strip().split())

    if not raw_tokens:
        raise ValueError(f"RTS file {filepath} appears empty after the title line")

    # First token is NRr/NRz, then exactly NRr range/depth floats.
    nr = int(raw_tokens[0])
    if len(raw_tokens) < 1 + nr:
        raise ValueError(
            f"RTS file {filepath} truncated: expected {nr} range/depth values, "
            f"only {len(raw_tokens) - 1} tokens available after count."
        )
    ranges = np.array([float(x) for x in raw_tokens[1:1 + nr]])

    # Remaining tokens are time-series records: (1 time + nr pressures) per step.
    rest = raw_tokens[1 + nr:]
    values_per_timestep = 1 + nr
    nt = len(rest) // values_per_timestep

    time_list = []
    pressure_list = []
    for i in range(nt):
        start_idx = i * values_per_timestep
        time_list.append(float(rest[start_idx]))
        pressure_list.append([float(x) for x in rest[start_idx + 1:start_idx + 1 + nr]])

    time = np.array(time_list)
    p = np.array(pressure_list)  # shape (nt, nr)

    # Calculate dt
    if nt > 1:
        dt = time[1] - time[0]
    else:
        dt = 0.0

    return {
        "title": title,
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
