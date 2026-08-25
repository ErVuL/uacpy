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

#: Column-norm floor for OMP atom normalisation, RELATIVE to the largest
#: column norm. Guards the division against an all-zero column without
#: putting an absolute scale into a dimensional quantity.
_COLUMN_NORM_REL_FLOOR = 1e-12


def _conv_matrix(tx, n_taps, n_rows):
    """Tall convolution matrix ``A`` with ``A[k, l] = tx[k - l]`` (causal)."""
    if n_taps > n_rows:
        raise ConfigurationError(
            f"channel estimate: n_taps ({n_taps}) exceeds the number of "
            f"available pilot/received samples ({n_rows}); reduce n_taps or "
            "provide more pilots."
        )
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
        raise ConfigurationError(
            f"omp_estimate: need 1 <= sparsity <= n_taps; got "
            f"sparsity={sparsity}, n_taps={n_taps}")
    r = np.asarray(rx, dtype=complex).ravel()
    n = min(r.size, np.asarray(tx_pilots).size)
    A = _conv_matrix(tx_pilots, n_taps, n)
    y = r[:n]
    residual = y.copy()
    support: list[int] = []
    coeffs = np.zeros(0, dtype=complex)
    # The selection rule |A^H r| / norms is scale-invariant on its own:
    # scaling the pilots and the record together scales every projection
    # identically, so the argmax does not move. A bare ``+ 1e-12`` breaks
    # that, because the offset is absolute while the norms are dimensional.
    # It bites whenever adjacent columns are near-degenerate, which is the
    # NORMAL condition here — neighbouring delay lags are highly correlated.
    # Measured on a 40-lag grid whose column norms span 49x, with the top
    # three normalised projections tied to six decimals: the epsilon is
    # 2.3e-11 of the smallest norm at unit scale (harmless, support [3, 30])
    # and 2.3e-5 at 1e-6 scale, where it reorders the tie and returns
    # [0, 30]. The same channel in Pa rather than uPa estimated a different
    # delay. A relative floor keeps the guard against an all-zero column —
    # the only thing the offset was needed for — without setting a scale.
    # Same idiom as ``_ZF_REL_FLOOR`` / ``_PILOT_REL_FLOOR`` in comms.ofdm.
    raw_norms = np.linalg.norm(A, axis=0)
    peak_norm = float(raw_norms.max()) if raw_norms.size else 0.0
    norms = np.maximum(raw_norms, _COLUMN_NORM_REL_FLOOR * peak_norm) \
        if peak_norm > 0.0 else np.ones_like(raw_norms)
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
