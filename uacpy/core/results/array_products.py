"""OASN array-product result types: Covariance and Replicas."""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from uacpy.core.exceptions import ConfigurationError

from uacpy.core.results._base import Result


class Covariance(Result):
    """OASN spatial covariance matrix ``C(f, i, j)``.

    Hydrophone × hydrophone correlation per frequency, written by OASN with
    option ``N`` to a ``.xsm`` file. The eigenvectors of ``C[ifreq]`` are
    matched-field-processing replica vectors used for signal-subspace
    detection and localization.

    Attributes
    ----------
    covariance : ndarray, shape ``(n_frequencies, n_receivers, n_receivers)``
        Complex covariance matrices.
    receiver_positions : ndarray, optional, shape ``(n_receivers, 3)``
        ``(x, y, z)`` positions in metres.

    Notes
    -----
    To extract MFP signal-subspace eigenvectors call
    ``np.linalg.eigh(cov.covariance[ifreq])`` directly.
    """
    field_type = "covariance"

    def __init__(
        self,
        *,
        covariance: np.ndarray,
        receiver_positions: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        cov = np.asarray(covariance)
        if cov.ndim != 3 or cov.shape[1] != cov.shape[2]:
            raise ConfigurationError(
                f"Covariance.covariance: must be 3-D (n_freq, n_rcv, n_rcv); "
                f"got shape {cov.shape}"
            )
        self.covariance = cov
        if receiver_positions is not None:
            rp = np.asarray(receiver_positions, dtype=float)
            if rp.ndim != 2 or rp.shape[1] != 3 or rp.shape[0] != cov.shape[1]:
                raise ConfigurationError(
                    f"Covariance.receiver_positions: must have shape "
                    f"(n_receivers={cov.shape[1]}, 3); got {rp.shape}"
                )
            self.receiver_positions = rp
        else:
            self.receiver_positions = None

    @property
    def n_frequencies(self) -> int:
        return int(self.covariance.shape[0])

    @property
    def n_receivers(self) -> int:
        return int(self.covariance.shape[1])

    def _repr_extra(self) -> str:
        return f"n_rcv={self.n_receivers}"

    def _replica_grid(self, replicas: "Replicas") -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Validate and reshape the replica field to ``(n_f, n_pts, n_rcv)``."""
        if replicas.replicas.shape[0] != self.n_frequencies:
            raise ConfigurationError(
                f"Covariance MFP: frequency mismatch — "
                f"covariance has {self.n_frequencies} freq, "
                f"replicas has {replicas.replicas.shape[0]}."
            )
        if replicas.replicas.shape[-1] != self.n_receivers:
            raise ConfigurationError(
                f"Covariance MFP: receiver-count mismatch — "
                f"covariance has {self.n_receivers}, "
                f"replicas has {replicas.replicas.shape[-1]}."
            )
        n_f = replicas.replicas.shape[0]
        nz, nx, ny = replicas.replicas.shape[1:4]
        flat = replicas.replicas.reshape(n_f, nz * nx * ny, self.n_receivers)
        return flat, (n_f, nz, nx, ny)

    @staticmethod
    def _normalise_weights(w: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(w, axis=-1, keepdims=True)
        norm = np.where(norm > 0, norm, 1.0)
        return w / norm

    def bartlett(self, replicas: "Replicas") -> np.ndarray:
        """Conventional Bartlett MFP ambiguity surface.

        ``B(z, x, y; f) = w(z, x, y; f)ᴴ · C(f) · w(z, x, y; f)``

        with ``w`` the replica vector at each candidate point, normalised
        to unit length.

        Returns
        -------
        ndarray, shape ``(n_freq, n_zr, n_xr, n_yr)``
            Real-valued ambiguity power. Argmax over the last three axes
            is the source-localisation peak.
        """
        flat, (n_f, nz, nx, ny) = self._replica_grid(replicas)
        out = np.empty((n_f, nz * nx * ny), dtype=float)
        for f in range(n_f):
            W = self._normalise_weights(flat[f])  # (n_pts, n_rcv)
            CW = self.covariance[f] @ W.T          # (n_rcv, n_pts)
            out[f] = np.real(np.einsum('pr,rp->p', W.conj(), CW))
        return out.reshape(n_f, nz, nx, ny)

    def mvdr(
        self,
        replicas: "Replicas",
        *,
        diagonal_loading: float = 1e-6,
    ) -> np.ndarray:
        """Minimum-Variance Distortionless-Response (Capon) MFP.

        ``M(z, x, y; f) = 1 / (wᴴ · (C(f) + δ·I)⁻¹ · w)`` with
        ``δ = diagonal_loading · trace(C(f))/N``. Small loading
        (~1e-6) stabilises rank-deficient covariance for sharp Capon
        peaks; larger loading (~0.1+) flattens the surface toward
        Bartlett for mismatch robustness. This is *not* the
        Cox/Zeskind/Owen white-noise-constrained processor (that
        requires per-replica Lagrange-multiplier bisection).
        """
        flat, (n_f, nz, nx, ny) = self._replica_grid(replicas)
        out = np.empty((n_f, nz * nx * ny), dtype=float)
        for f in range(n_f):
            C = self.covariance[f]
            tr = float(np.real(np.trace(C))) / max(C.shape[0], 1)
            Cload = C + diagonal_loading * tr * np.eye(C.shape[0])
            Cinv = np.linalg.inv(Cload)
            W = self._normalise_weights(flat[f])
            denom = np.einsum('pr,rs,ps->p', W.conj(), Cinv, W)
            out[f] = np.real(1.0 / np.where(np.abs(denom) > 0, denom, 1.0))
        return out.reshape(n_f, nz, nx, ny)


class Replicas(Result):
    """OASN matched-field-processing replicas.

    Frequency-domain Green's-function samples at every array element for
    every candidate source position. Written by OASN with option ``R`` to
    a ``.rpo`` file.

    Attributes
    ----------
    replicas : ndarray, shape ``(n_frequencies, n_zr, n_xr, n_yr, n_receivers)``
        Complex array responses per candidate source ``(z, x, y)``.
    replica_z, replica_x, replica_y : ndarray
        Coordinate axes of the candidate-source grid (m, m, m).
    receiver_positions : ndarray, optional, shape ``(n_receivers, 3)``
        ``(x, y, z)`` positions in metres.

    Notes
    -----
    To compute a Bartlett MFP ambiguity surface, contract a covariance
    estimate against the replica field across the array index.
    """
    field_type = "replicas"

    def __init__(
        self,
        *,
        replicas: np.ndarray,
        replica_z: np.ndarray,
        replica_x: np.ndarray,
        replica_y: np.ndarray,
        receiver_positions: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        rep = np.asarray(replicas)
        if rep.ndim != 5:
            raise ConfigurationError(
                f"Replicas.replicas: must be 5-D "
                f"(n_freq, n_zr, n_xr, n_yr, n_rcv); got shape {rep.shape}"
            )
        self.replicas = rep
        self.replica_z = np.atleast_1d(np.asarray(replica_z, dtype=float))
        self.replica_x = np.atleast_1d(np.asarray(replica_x, dtype=float))
        self.replica_y = np.atleast_1d(np.asarray(replica_y, dtype=float))
        expected = (
            len(self.replica_z), len(self.replica_x), len(self.replica_y),
        )
        if rep.shape[1:4] != expected:
            raise ConfigurationError(
                f"Replicas.replicas: axes 1-3 {rep.shape[1:4]} must match "
                f"(n_zr, n_xr, n_yr) = {expected}"
            )
        if receiver_positions is not None:
            rp = np.asarray(receiver_positions, dtype=float)
            if rp.ndim != 2 or rp.shape[1] != 3 or rp.shape[0] != rep.shape[4]:
                raise ConfigurationError(
                    f"Replicas.receiver_positions: must have shape "
                    f"(n_receivers={rep.shape[4]}, 3); got {rp.shape}"
                )
            self.receiver_positions = rp
        else:
            self.receiver_positions = None

    @property
    def n_frequencies(self) -> int:
        return int(self.replicas.shape[0])

    @property
    def n_receivers(self) -> int:
        return int(self.replicas.shape[4])

    @property
    def n_replica_points(self) -> int:
        return int(self.replicas.shape[1] * self.replicas.shape[2] * self.replicas.shape[3])

    def _repr_extra(self) -> str:
        return f"n_pts={self.n_replica_points}, n_rcv={self.n_receivers}"
