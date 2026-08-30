"""Direct-sequence spread spectrum (DSSS) for covert / low-SNR UW links.

Spreading each symbol by a pseudo-noise chip sequence trades bandwidth for
processing gain ``10*log10(N)`` (``N`` chips/symbol), pushing the signal below
the noise floor and rejecting narrowband interference — common in covert and
multi-user underwater systems.

References
----------
Proakis & Salehi. *Digital Communications* (spread spectrum, PN sequences,
    processing gain).
"""

from __future__ import annotations

import numpy as np

from uacpy.core.exceptions import ConfigurationError


def m_sequence(n_register, taps):
    """Maximal-length PN sequence (``+/-1``) of length ``2**n_register - 1``.

    ``taps`` are the 1-based feedback-tap positions of the LFSR (e.g. ``[5, 2]``
    for a length-31 sequence). Chips use the standard BPSK mapping
    ``s = 1 - 2*bit`` (bit 0 → +1, bit 1 → -1). Sibling of
    :func:`uacpy.acoustic_signal.sequences.mseq`, which generates m-sequences
    from preset polynomials keyed by register length with the same chip
    polarity — this variant takes the taps explicitly for spreading-code
    experiments.
    """
    n = int(n_register)
    tap_list = [int(t) for t in taps]
    if n < 2:
        raise ConfigurationError(
            f"m_sequence: n_register must be >= 2; got {n_register!r}")
    if not tap_list or any(t < 1 or t > n for t in tap_list):
        raise ConfigurationError(
            f"m_sequence: taps must be 1-based positions in 1..{n}; "
            f"got {taps!r}")
    if len(set(tap_list)) != len(tap_list):
        raise ConfigurationError(
            f"m_sequence: repeated tap position in {taps!r}; a tap XORed with "
            "itself cancels, leaving the register with no feedback")
    seed = (1,) * n
    reg = list(seed)
    length = (1 << n) - 1
    seq = np.empty(length, dtype=int)
    period = None
    for i in range(length):
        seq[i] = reg[-1]
        fb = 0
        for t in tap_list:
            fb ^= reg[t - 1]
        reg = [fb] + reg[:-1]
        if period is None and tuple(reg) == seed:
            period = i + 1
    # A non-primitive polynomial returns early to the seed, so the register
    # cycles through a subset of its states. The output still has the right
    # length, dtype and +/-1 alphabet, and processing_gain_db still reports
    # 10log10(N) — nothing looks wrong until a link budget is far out.
    if period != length:
        raise ConfigurationError(
            f"m_sequence: taps {taps!r} on a {n}-stage register give an LFSR "
            f"period of {period if period is not None else '>' + str(length)}, "
            f"not the maximal {length} — the feedback polynomial is not "
            f"primitive, so the output is not an m-sequence and its off-peak "
            f"autocorrelation is far above 1.",
            remediation=f"Use uacpy.acoustic_signal.sequences.mseq({n}) for a "
                        f"preset primitive polynomial, or a known primitive "
                        f"tap set (e.g. [5,2], [5,3], [6,1], [6,5], [7,6]).")
    return 1 - 2 * seq          # {0,1} -> {+1,-1}


def spread(symbols, code):
    """Spread each symbol by the chip ``code`` (Kronecker product). Returns chips."""
    s = np.asarray(symbols, dtype=complex).ravel()
    c = np.asarray(code, dtype=complex).ravel()
    if c.size < 1:
        raise ConfigurationError(
            "spread: empty code (0 chips); pass the spreading sequence the "
            "receiver will despread with, e.g. m_sequence(n_stages, taps).")
    return np.kron(s, c)


def despread(chips, code):
    """Correlate chips against ``code`` per symbol period -> symbol estimates.

    The correlation is normalised by the code energy ``sum(|c|^2)``, so
    ``despread(spread(s, c), c)`` returns ``s`` itself for any code amplitude
    (dividing by the chip count is equivalent only for unit-modulus codes).
    A trailing partial symbol (fewer than ``len(code)`` chips) is dropped.
    """
    c = np.asarray(code, dtype=complex).ravel()
    x = np.asarray(chips, dtype=complex).ravel()
    if c.size < 1:
        raise ConfigurationError(
            "despread: empty code (0 chips); pass the same spreading "
            "sequence spread() was called with.")
    energy = float(np.sum(np.abs(c) ** 2))
    if energy == 0.0:
        raise ConfigurationError(
            "despread: code energy sum(|c|^2) is zero (every chip is 0), so "
            "the per-symbol correlation carries no signal and the "
            "normalisation is undefined")
    n = c.size
    nsym = x.size // n
    blocks = x[: nsym * n].reshape(nsym, n)
    return (blocks @ np.conj(c)) / energy


def processing_gain_db(code):
    """Processing gain ``10*log10(N)`` in dB for an ``N``-chip code."""
    return float(10.0 * np.log10(np.asarray(code).size))
