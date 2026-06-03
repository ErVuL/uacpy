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
    err = np.mean(np.abs(r[:n] - s[:n]) ** 2)
    ref = np.mean(np.abs(s[:n]) ** 2)
    return float(np.sqrt(err / ref))


def ber_theory(scheme, ebn0_db):
    """Theoretical AWGN BER vs Eb/N0 (dB) for a Gray-mapped scheme.

    Exact for BPSK/QPSK; standard nearest-neighbour approximations for higher
    M-PSK and square M-QAM (Proakis).
    """
    ebn0 = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    s = scheme.lower()
    if s in ("bpsk", "qpsk"):
        return _q(np.sqrt(2.0 * ebn0))
    if s.endswith("psk"):
        M = {"8psk": 8, "16psk": 16}[s]
        k = np.log2(M)
        return (2.0 / k) * _q(np.sqrt(2.0 * k * ebn0) * np.sin(np.pi / M))
    if s.endswith("qam"):
        M = {"16qam": 16, "64qam": 64, "256qam": 256}[s]
        k = np.log2(M)
        c = 4.0 / k * (1.0 - 1.0 / np.sqrt(M))
        return c * _q(np.sqrt(3.0 * k / (M - 1.0) * ebn0))
    raise ConfigurationError(f"ber_theory: unsupported scheme {scheme!r}")


def scatter(symbols, ax=None, title="", **kwargs):
    """Constellation/scatter plot of received symbols. Returns ``(fig, ax)``."""
    import matplotlib.pyplot as plt
    s = np.asarray(symbols, dtype=complex).ravel()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    kwargs.setdefault("s", 6)
    kwargs.setdefault("alpha", 0.4)
    ax.scatter(s.real, s.imag, **kwargs)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("In-phase"); ax.set_ylabel("Quadrature")
    ax.set_title(f"[constellation] {title}", loc="left")
    return fig, ax


def eye_diagram(signal, samples_per_symbol, n_symbols=2, ax=None, title="",
                **kwargs):
    """Eye diagram: overlay ``n_symbols``-wide windows of the real signal.

    Returns ``(fig, ax)``. ``kwargs`` pass to ``ax.plot``.
    """
    import matplotlib.pyplot as plt
    x = np.real(np.asarray(signal)).ravel()
    sps = int(samples_per_symbol)
    span = sps * int(n_symbols)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    kwargs.setdefault("color", "#1f77b4")
    kwargs.setdefault("alpha", 0.15)
    kwargs.setdefault("lw", 0.7)
    t = np.arange(span) / sps
    n = (x.size - span) // sps
    for k in range(max(n, 0)):
        ax.plot(t, x[k * sps: k * sps + span], **kwargs)
    ax.set_xlabel("Symbol intervals"); ax.set_ylabel("Amplitude")
    ax.set_title(f"[eye] {title}", loc="left"); ax.grid(alpha=0.3)
    return fig, ax


def ber_curve(ebn0_db, ber_measured, scheme=None, ax=None, title="",
              label="measured", **kwargs):
    """Plot measured BER vs Eb/N0 (semilog-y) with the optional theory overlay.

    Returns ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt
    ebn0 = np.atleast_1d(np.asarray(ebn0_db, dtype=float))
    ber = np.atleast_1d(np.asarray(ber_measured, dtype=float))
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure
    kwargs.setdefault("marker", "o")
    ax.semilogy(ebn0, np.maximum(ber, 1e-12), label=label, **kwargs)
    if scheme is not None:
        fine = np.linspace(ebn0.min(), ebn0.max(), 100)
        ax.semilogy(fine, ber_theory(scheme, fine), "k--",
                    label=f"{scheme} theory")
    ax.set_xlabel("Eb/N0 [dB]"); ax.set_ylabel("BER")
    ax.set_title(f"[BER] {title}", loc="left")
    ax.grid(which="both", alpha=0.3); ax.legend()
    return fig, ax
