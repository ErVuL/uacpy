"""How good the link was: bit and symbol error rates, the error vector
magnitude, and the theoretical bit-error rate of each scheme."""

from __future__ import annotations

from scipy.special import erfc
import numpy as np

from uacpy.comms.constellations import SCHEMES
from uacpy.core.exceptions import ConfigurationError


def _q(x):
    """Gaussian Q-function ``Q(x) = 0.5*erfc(x/sqrt(2))``."""
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


def bit_error_rate(*, reference, received):
    """Fraction of ``received`` bits differing from the ``reference`` bits,
    over the overlap of the two streams.

    Parameters
    ----------
    reference : array_like
        The transmitted bits.
    received : array_like
        The received bits; compared over the overlap of the two.
    """
    a = np.asarray(reference, dtype=int).ravel()
    b = np.asarray(received, dtype=int).ravel()
    n = min(a.size, b.size)
    if n == 0:
        raise ConfigurationError(
            f"bit_error_rate: empty input — reference carries {a.size} bits, "
            f"received {b.size}. The BER is taken over the overlap of the two "
            f"streams, so both must be non-empty.")
    return float(np.mean(a[:n] != b[:n]))


def symbol_error_rate(*, reference, received):
    """SER of ``received`` against ``reference`` over the overlap; accepts
    integer labels or complex symbols (compared exactly).

    Parameters
    ----------
    reference : array_like
        The transmitted symbols or integer labels.
    received : array_like
        The received sequence; compared over the overlap of the two.
    """
    a = np.asarray(reference).ravel()
    b = np.asarray(received).ravel()
    n = min(a.size, b.size)
    if n == 0:
        raise ConfigurationError(
            f"symbol_error_rate: empty input — reference carries {a.size} "
            f"symbols, received {b.size}. The SER is taken over the overlap of the two "
            f"streams, so both must be non-empty.")
    return float(np.mean(a[:n] != b[:n]))


def evm(*, reference, received):
    """RMS error-vector magnitude of ``received`` against ``reference``
    (fraction; multiply by 100 for percent).

    ``sqrt(mean|received-reference|^2 / mean|reference|^2)`` over the
    overlap.

    Parameters
    ----------
    reference : array_like
        The transmitted symbols.
    received : array_like
        The received sequence; compared over the overlap of the two.
    """
    r = np.asarray(received, dtype=complex).ravel()
    s = np.asarray(reference, dtype=complex).ravel()
    n = min(r.size, s.size)
    if n == 0:
        raise ConfigurationError(
            f"evm: empty input — received carries {r.size} symbols, "
            f"reference {s.size}. The EVM is taken over the overlap of the "
            f"two streams, so both must be non-empty.")
    err = np.mean(np.abs(r[:n] - s[:n]) ** 2)
    ref = np.mean(np.abs(s[:n]) ** 2)
    if ref == 0.0:
        # EVM is a ratio to the reference power, so a zero-energy reference
        # leaves it undefined; returning inf behind a numpy divide warning
        # reads as "infinitely bad", not "not defined".
        raise ConfigurationError(
            "evm: reference symbols carry no energy, so the error vector has "
            "nothing to be relative to. Pass the transmitted constellation "
            "symbols as reference=.")
    return float(np.sqrt(err / ref))


def ber_theory(scheme, ebn0_dB):
    """Theoretical AWGN BER vs Eb/N0 (dB) for a Gray-mapped scheme.

    Exact for BPSK/QPSK; standard nearest-neighbour approximations for higher
    M-PSK and square M-QAM (Proakis & Salehi):

    * M-PSK symbol error ``P_M = 2 Q(sqrt(2 k Eb/N0) sin(pi/M))``, eq. (4.3-17).
    * square M-QAM ``P_M ~= 4 (1 - 1/sqrt(M)) Q(sqrt(3 k Eb/N0 / (M-1)))``,
      eqs. (4.3-29) into (4.3-27), dropping the second-order term.

    Both are **symbol** error rates; the ``1/k`` factor that converts them to a
    bit error rate is only valid under Gray mapping (eq. 4.3-20): adjacent
    constellation points then differ in a single bit, so the dominant
    nearest-neighbour symbol error costs exactly one of the ``k`` bits.
    :func:`uacpy.comms.constellation` is Gray-mapped throughout.

    The two non-coherent binary modems the package ships have exact closed
    forms (standard results, not checked against a source in this
    project's corpus; they match this package's own demodulators in
    simulation, pinned in the tests):

    * ``'dbpsk'`` — binary DPSK with differential detection
      (:func:`dpsk_modulate` / :func:`dpsk_demodulate`, ``M=2``):
      ``P_b = exp(-Eb/N0) / 2``.
    * ``'bfsk'`` — orthogonal binary FSK with non-coherent (tone-energy)
      detection (:func:`fsk_modulate` / :func:`fsk_demodulate` with two
      tones spaced by a multiple of ``1/symbol_duration_s``):
      ``P_b = exp(-Eb/(2 N0)) / 2``.

    Parameters
    ----------
    scheme : str
        A scheme of :data:`SCHEMES`, or ``'dbpsk'`` / ``'bfsk'``.
    ebn0_dB : float or array_like
        Energy per bit over the noise density, in dB.
    """
    ebn0 = 10.0 ** (np.asarray(ebn0_dB, dtype=float) / 10.0)
    s = scheme.lower()
    if s in ("bpsk", "qpsk"):
        return _q(np.sqrt(2.0 * ebn0))
    if s == "dbpsk":
        return 0.5 * np.exp(-ebn0)
    if s == "bfsk":
        return 0.5 * np.exp(-0.5 * ebn0)
    if s in SCHEMES:
        family, M = SCHEMES[s]
        k = np.log2(M)
        if family == 'psk':
            return (2.0 / k) * _q(np.sqrt(2.0 * k * ebn0) * np.sin(np.pi / M))
        c = 4.0 / k * (1.0 - 1.0 / np.sqrt(M))
        return c * _q(np.sqrt(3.0 * k / (M - 1.0) * ebn0))
    valid = (*SCHEMES, "dbpsk", "bfsk")
    raise ConfigurationError(
        f"ber_theory: unsupported scheme {scheme!r}; valid: {', '.join(valid)}.")
