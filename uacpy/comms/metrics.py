"""Link-performance metrics: bit/symbol error rate, EVM, theoretical curves.

References
----------
Proakis & Salehi. *Digital Communications* (error-probability expressions).
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc

from uacpy.core.exceptions import ConfigurationError


def _q(x):
    """Gaussian Q-function ``Q(x) = 0.5*erfc(x/sqrt(2))``."""
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


def bit_error_rate(tx_bits, rx_bits):
    """Fraction of differing bits over the overlap of the two bit streams."""
    a = np.asarray(tx_bits, dtype=int).ravel()
    b = np.asarray(rx_bits, dtype=int).ravel()
    n = min(a.size, b.size)
    if n == 0:
        raise ConfigurationError("bit_error_rate: empty input")
    return float(np.mean(a[:n] != b[:n]))


def symbol_error_rate(tx_symbols_or_labels, rx_symbols_or_labels):
    """SER over the overlap; accepts integer labels or complex symbols (compared exactly)."""
    a = np.asarray(tx_symbols_or_labels).ravel()
    b = np.asarray(rx_symbols_or_labels).ravel()
    n = min(a.size, b.size)
    if n == 0:
        raise ConfigurationError("symbol_error_rate: empty input")
    return float(np.mean(a[:n] != b[:n]))


def evm(rx_symbols, ref_symbols):
    """RMS error-vector magnitude (fraction; multiply by 100 for percent).

    ``sqrt(mean|rx-ref|^2 / mean|ref|^2)`` over the overlap.
    """
    r = np.asarray(rx_symbols, dtype=complex).ravel()
    s = np.asarray(ref_symbols, dtype=complex).ravel()
    n = min(r.size, s.size)
    if n == 0:
        raise ConfigurationError("evm: empty input")
    err = np.mean(np.abs(r[:n] - s[:n]) ** 2)
    ref = np.mean(np.abs(s[:n]) ** 2)
    return float(np.sqrt(err / ref))


_BER_PSK_ORDERS = {"8psk": 8, "16psk": 16}
_BER_QAM_ORDERS = {"16qam": 16, "64qam": 64, "256qam": 256}


def ber_theory(scheme, ebn0_db):
    """Theoretical AWGN BER vs Eb/N0 (dB) for a Gray-mapped scheme.

    Exact for BPSK/QPSK; standard nearest-neighbour approximations for higher
    M-PSK and square M-QAM (Proakis).
    """
    ebn0 = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    s = scheme.lower()
    if s in ("bpsk", "qpsk"):
        return _q(np.sqrt(2.0 * ebn0))
    if s in _BER_PSK_ORDERS:
        M = _BER_PSK_ORDERS[s]
        k = np.log2(M)
        return (2.0 / k) * _q(np.sqrt(2.0 * k * ebn0) * np.sin(np.pi / M))
    if s in _BER_QAM_ORDERS:
        M = _BER_QAM_ORDERS[s]
        k = np.log2(M)
        c = 4.0 / k * (1.0 - 1.0 / np.sqrt(M))
        return c * _q(np.sqrt(3.0 * k / (M - 1.0) * ebn0))
    valid = ("bpsk", "qpsk", *_BER_PSK_ORDERS, *_BER_QAM_ORDERS)
    raise ConfigurationError(
        f"ber_theory: unsupported scheme {scheme!r}; valid: {', '.join(valid)}")
