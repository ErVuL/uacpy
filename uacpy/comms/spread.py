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
    for a length-31 sequence).
    """
    n = int(n_register)
    reg = [1] * n
    length = (1 << n) - 1
    seq = np.empty(length, dtype=int)
    for i in range(length):
        seq[i] = reg[-1]
        fb = 0
        for t in taps:
            fb ^= reg[t - 1]
        reg = [fb] + reg[:-1]
    return 1 - 2 * seq          # {0,1} -> {+1,-1}


def spread(symbols, code):
    """Spread each symbol by the chip ``code`` (Kronecker product). Returns chips."""
    s = np.asarray(symbols, dtype=complex).ravel()
    c = np.asarray(code, dtype=complex).ravel()
    if c.size < 1:
        raise ConfigurationError("spread: empty code")
    return np.kron(s, c)


def despread(chips, code):
    """Correlate chips against ``code`` per symbol period -> symbol estimates."""
    c = np.asarray(code, dtype=complex).ravel()
    x = np.asarray(chips, dtype=complex).ravel()
    n = c.size
    nsym = x.size // n
    blocks = x[: nsym * n].reshape(nsym, n)
    return (blocks @ np.conj(c)) / n


def processing_gain_db(code):
    """Processing gain ``10*log10(N)`` in dB for an ``N``-chip code."""
    return float(10.0 * np.log10(np.asarray(code).size))
