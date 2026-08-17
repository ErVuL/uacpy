"""OASN array-product result types: Covariance and Replicas."""

from __future__ import annotations

import warnings

import numpy as np
from typing import Optional, Tuple

from uacpy.core.exceptions import ConfigurationError

from uacpy.core._beamforming import loaded_inverse, quadratic_form

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
        # Copy on ingest so a caller mutating their source array can't silently
        # corrupt this result.
        cov = np.array(covariance)
        if cov.ndim != 3 or cov.shape[1] != cov.shape[2]:
            raise ConfigurationError(
                f"Covariance.covariance: must be 3-D (n_freq, n_rcv, n_rcv); "
                f"got shape {cov.shape}"
            )
        self.covariance = cov
        if receiver_positions is not None:
            rp = np.array(receiver_positions, dtype=float)
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
            # The shared Bartlett/MVDR core (core/_beamforming).
            out[f] = quadratic_form(self.covariance[f], W)
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

        The 1e-6 default matches :func:`uacpy.acoustic_signal.mvdr_spectrum`
        and suits the full-rank covariance OASN writes to its ``.xsm``.
        :func:`uacpy.sonar.mvdr` — the same processor over a *measured*
        CSDM — defaults to 1e-2 instead, because a few-snapshot ``csdm()``
        is routinely rank-deficient. The two agree numerically at equal
        loading, up to the max-scaling ``sonar.mvdr`` applies, and both
        return NaN for a degenerate candidate point.
        """
        flat, (n_f, nz, nx, ny) = self._replica_grid(replicas)
        out = np.empty((n_f, nz * nx * ny), dtype=float)
        for f in range(n_f):
            C = self.covariance[f]
            tr = float(np.real(np.trace(C))) / max(C.shape[0], 1)
            # The loading is a fraction of tr(C)/N and vanishes with it, so a
            # frequency bin carrying no power leaves C singular; that whole
            # bin's surface is undefined rather than zero.
            if tr <= 0.0:
                warnings.warn(
                    f"Covariance.mvdr: frequency bin {f} carries no power, so "
                    f"its ambiguity surface is undefined; returning NaN.",
                    UserWarning, stacklevel=2)
                out[f] = np.nan
                continue
            W = self._normalise_weights(flat[f])
            denom = quadratic_form(loaded_inverse(C, diagonal_loading), W)
            # For a positive-definite loaded covariance denom > 0. It reaches
            # 0 only for a replica carrying no energy (an unpopulated .rpo
            # cell) and goes negative only when C is not positive-definite,
            # i.e. not a covariance. Neither is a power: a finite value there
            # would sit in the surface as a genuine localisation peak. Same
            # rule as :func:`uacpy.sonar.mvdr`.
            with np.errstate(divide='ignore', invalid='ignore'):
                out[f] = np.where(denom > 0, 1.0 / denom, np.nan)
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
    Feed these to :meth:`Covariance.bartlett` or :meth:`Covariance.mvdr`
    for an ambiguity surface; both contract a covariance estimate against
    the replica field across the array index.
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
        # Copy on ingest so a caller mutating their source array can't silently
        # corrupt this result.
        rep = np.array(replicas)
        if rep.ndim != 5:
            raise ConfigurationError(
                f"Replicas.replicas: must be 5-D "
                f"(n_freq, n_zr, n_xr, n_yr, n_rcv); got shape {rep.shape}"
            )
        self.replicas = rep
        self.replica_z = np.atleast_1d(np.array(replica_z, dtype=float))
        self.replica_x = np.atleast_1d(np.array(replica_x, dtype=float))
        self.replica_y = np.atleast_1d(np.array(replica_y, dtype=float))
        expected = (
            len(self.replica_z), len(self.replica_x), len(self.replica_y),
        )
        if rep.shape[1:4] != expected:
            raise ConfigurationError(
                f"Replicas.replicas: axes 1-3 {rep.shape[1:4]} must match "
                f"(n_zr, n_xr, n_yr) = {expected}"
            )
        if receiver_positions is not None:
            rp = np.array(receiver_positions, dtype=float)
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
