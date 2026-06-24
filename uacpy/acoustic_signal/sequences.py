"""Coded probe sequences: BPSK modulation, maximum-length (m-)sequences and the m-sequence channel probe."""


import numpy as np

from uacpy.core.exceptions import ConfigurationError


def bpsk_modulate(
    s_bipolar: np.ndarray, fc: float, sample_rate: float, chips_per_sec: float
) -> np.ndarray:
    """
    Encode binary sequence as Binary Phase Shift Keying (BPSK) signal.

    Parameters
    ----------
    s_bipolar : ndarray
        Binary source sequence (+1/-1 values)
    fc : float
        Carrier frequency in Hz
    sample_rate : float
        Sample frequency in Hz
    chips_per_sec : float
        Chip rate (symbols per second)

    Returns
    -------
    s : ndarray
        BPSK modulated signal

    Notes
    -----
    Each binary symbol (chip) is represented by a sinusoid of length
    samples_per_chip. The phase is 0 for +1, π for -1.

    Examples
    --------
    >>> # Binary sequence
    >>> bits = np.array([1, -1, 1, 1, -1, 1])
    >>>
    >>> # BPSK modulation
    >>> fc = 12000  # 12 kHz carrier
    >>> sample_rate = 48000  # 48 kHz sample rate
    >>> chips_per_sec = 3000  # 3k chips/sec
    >>> s = bpsk_modulate(bits, fc, sample_rate, chips_per_sec)

    References
    ----------
    Original MATLAB code by Michael B. Porter, April 2000
    """
    samples_per_chip = int(sample_rate / chips_per_sec)

    if sample_rate / chips_per_sec != samples_per_chip:
        raise ConfigurationError("samples_per_chip must be an integer")

    deltat = 1 / sample_rate
    t_chip = np.arange(samples_per_chip) * deltat
    sinwave = np.sin(2 * np.pi * fc * t_chip)

    # Outer product: each column is one chip
    s_matrix = np.outer(sinwave, s_bipolar)

    # Flatten to 1D signal
    s = s_matrix.flatten(order="F")

    return s


def mseq(m: int) -> np.ndarray:
    """
    Generate an m-sequence (maximum-length sequence).

    M-sequences are pseudorandom binary sequences with excellent
    autocorrelation properties, useful for coded waveforms in sonar.

    Parameters
    ----------
    m : int
        Sequence order (2 ≤ m ≤ 15).
        Generates sequence of length 2^m - 1.

    Returns
    -------
    s : ndarray
        M-sequence as +1/-1 values. Length = 2^m - 1

    Notes
    -----
    Uses shift register with feedback based on primitive polynomials.
    The resulting sequence has:
    - Length N = 2^m - 1
    - Nearly flat autocorrelation (ideal for matched filtering)
    - Balanced +1/-1 symbols

    Formulas from Proakis, Digital Communications

    Translated from OALIB mseq.m by Michael B. Porter

    Examples
    --------
    >>> # Generate m-sequence of order 5
    >>> s = mseq(5)
    >>> print(f"Length: {len(s)} (should be 2^5-1 = 31)")

    >>> # Check autocorrelation
    >>> shat = np.fft.fft(s)
    >>> scorr = np.real(np.fft.ifft(shat * np.conj(shat)))
    """
    if m < 2 or m > 15 or m != int(m):
        raise ConfigurationError("m must be an integer between 2 and 15")

    m = int(m)

    # Feedback coefficients for primitive polynomials
    coefficients = {
        2: [1, 1],
        3: [1, 0, 1],
        4: [1, 0, 0, 1],
        5: [1, 0, 0, 1, 0],
        6: [1, 0, 0, 0, 0, 1],
        7: [1, 0, 0, 0, 0, 0, 1],
        8: [1, 0, 0, 0, 1, 1, 1, 0],
        9: [1, 0, 0, 0, 0, 1, 0, 0, 0],
        10: [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        11: [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        12: [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        13: [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
        14: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        15: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    }

    c = np.array(coefficients[m])
    length = 2**m - 1

    # Successive shifts with feedback (Proakis p. 433)
    seed = np.zeros(m)
    seed[0] = 1  # All zero except first element
    s = np.zeros(length)

    for ii in range(length):
        out = np.zeros(m)
        out[: m - 1] = seed[1:m]
        out[m - 1] = np.mod(np.dot(c, seed), 2)  # Addition mod 2
        seed = out
        s[ii] = out[0]

    # Convert 0/1 to -1/+1
    s[s == 0] = -1

    return s


def make_mseq_probe(fmin: float, fmax: float, sample_rate: float, T_tot: float) -> np.ndarray:
    """
    Generate an m-sequence probe signal with BPSK modulation.

    Creates a repeated m-sequence probe with a leader, suitable for
    channel sounding experiments.

    Parameters
    ----------
    fmin : float
        Minimum frequency in Hz
    fmax : float
        Maximum frequency in Hz
    sample_rate : float
        Sampling rate in Hz
    T_tot : float
        Total duration in seconds

    Returns
    -------
    probe : ndarray
        BPSK-modulated m-sequence probe signal

    Notes
    -----
    The probe consists of:

    1. Leader (0.2 s of zeros)
    2. Repeated m-sequence (order 10, length 1023)
    3. BPSK modulation at center frequency fc = (fmin + fmax) / 2
    4. Zero-padding to T_tot

    Chip rate is (fmax - fmin) / 2. Output is normalized to 0.95 of
    full scale and is exactly ``round(T_tot * sample_rate)`` samples long.

    Raises :class:`~uacpy.core.exceptions.ConfigurationError` if ``T_tot`` is
    too short to hold the leader plus one full m-sequence period — a partial
    period would lose the two-valued autocorrelation the probe exists for, so
    increase ``T_tot`` or widen ``fmax - fmin`` to raise the chip rate.

    Translated from OALIB makemseq.m by mbp.

    Examples
    --------
    >>> # Generate 10-second probe, 1-2 kHz
    >>> probe = make_mseq_probe(1000, 2000, 10000, 10.0)
    >>> print(f"Probe length: {len(probe)} samples")
    """
    lead_time = 0.2  # seconds

    # M-sequence parameters
    fc = 0.5 * (fmin + fmax)  # center frequency
    chips_per_sec = 0.5 * (fmax - fmin)

    # Generate base m-sequence (order 10 → length 1023)
    s_m = mseq(10)
    s = bpsk_modulate(s_m, fc, sample_rate, chips_per_sec)

    # Whole m-sequence periods that fit after the leader, counted in samples so
    # the probe lands at exactly target_n. Counting the leader (the previous
    # rep-count ignored it) is what keeps the probe inside T_tot; a period is
    # never truncated, since a partial m-sequence loses the two-valued
    # autocorrelation the probe exists for.
    leader = np.zeros(int(lead_time * sample_rate))
    target_n = int(round(T_tot * sample_rate))
    Nreps = (target_n - leader.size) // len(s)
    if Nreps < 1:
        raise ConfigurationError(
            f"make_mseq_probe: T_tot={T_tot:g} s is too short for the "
            f"{lead_time:g} s leader plus one m-sequence period "
            f"({len(s) / sample_rate:.3f} s at chip rate {chips_per_sec:g} chips/s). "
            f"Increase T_tot, or widen (fmax - fmin) to raise the chip rate."
        )
    probe = np.tile(s, Nreps)
    probe_max = np.max(np.abs(probe))
    if probe_max > 0:
        probe = np.concatenate([leader, 0.95 * probe / probe_max])
    else:
        probe = np.concatenate([leader, probe])

    # Zero-fill to the exact total duration (leader + Nreps periods <= target_n).
    if probe.size < target_n:
        probe = np.concatenate([probe, np.zeros(target_n - probe.size)])

    return probe
