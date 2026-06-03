"""Pilot-based channel estimation: least-squares and sparse OMP.

The UW channel impulse response is typically *sparse* — a few strong arrivals
separated by long quiet spans — so orthogonal matching pursuit (OMP) recovers it
from far fewer pilots than dense least squares (Berger, Zhou et al.).

References
----------
Proakis & Salehi. *Digital Communications* (LS channel estimation).
Berger, Zhou, Preisig & Willett (2010), *Sparse channel estimation for
    multicarrier underwater acoustic communication: from subspace methods to
    compressed sensing*, IEEE Trans. Signal Processing.
"""

from __future__ import annotations

import numpy as np

from uacpy.core.exceptions import ConfigurationError


def _conv_matrix(tx, n_taps, n_rows):
    """Tall convolution matrix ``A`` with ``A[k, l] = tx[k - l]`` (causal)."""
    p = np.asarray(tx, dtype=complex)
    A = np.zeros((n_rows, n_taps), dtype=complex)
    for l in range(n_taps):
        A[l:, l] = p[: n_rows - l]
    return A


def ls_estimate(rx, tx_pilots, n_taps):
    """Least-squares estimate of an ``n_taps`` channel from known pilots.

    Solves ``rx ≈ A h`` where ``A`` is the pilot convolution matrix. Returns the
    complex tap vector ``h``.
    """
    r = np.asarray(rx, dtype=complex).ravel()
    n = min(r.size, np.asarray(tx_pilots).size)
    A = _conv_matrix(tx_pilots, n_taps, n)
    h, *_ = np.linalg.lstsq(A, r[:n], rcond=None)
    return h


def omp_estimate(rx, tx_pilots, n_taps, sparsity):
    """Sparse channel estimate via orthogonal matching pursuit.

    Recovers at most ``sparsity`` non-zero taps out of ``n_taps`` candidate
    delays — the right model for the sparse UW multipath channel. Returns the
    full ``n_taps`` complex vector (zeros off the support).
    """
    if sparsity < 1 or sparsity > n_taps:
        raise ConfigurationError("omp_estimate: need 1 <= sparsity <= n_taps")
    r = np.asarray(rx, dtype=complex).ravel()
    n = min(r.size, np.asarray(tx_pilots).size)
    A = _conv_matrix(tx_pilots, n_taps, n)
    y = r[:n]
    residual = y.copy()
    support: list[int] = []
    coeffs = np.zeros(0, dtype=complex)
    norms = np.linalg.norm(A, axis=0) + 1e-12
    for _ in range(int(sparsity)):
        proj = np.abs(A.conj().T @ residual) / norms
        proj[support] = -1.0
        support.append(int(np.argmax(proj)))
        As = A[:, support]
        coeffs, *_ = np.linalg.lstsq(As, y, rcond=None)
        residual = y - As @ coeffs
    h = np.zeros(n_taps, dtype=complex)
    h[support] = coeffs
    return h
