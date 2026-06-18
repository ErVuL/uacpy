"""Reflection-coefficient result type."""

from __future__ import annotations

import numpy as np
from typing import Optional

from uacpy.core.exceptions import ConfigurationError

from uacpy.core.results._base import Result


class ReflectionCoefficient(Result):
    """Angle-dependent reflection coefficient ``R(theta[, f])``.

    Unifies what Bounce and OASR produce. Used for both bottom (BRC) and
    top (TRC) reflection coefficients. ``R`` and ``phi`` may be 1-D
    (single-frequency) or 2-D (frequency-resolved); ``theta`` is always 1-D.

    Attributes
    ----------
    theta : ndarray, shape ``(n_angles,)``  — grazing angles in degrees
    R     : ndarray, shape ``(n_angles,)`` or ``(n_angles, n_frequencies)``
            — magnitude in [0, 1]
    phi   : ndarray, same shape as ``R`` — phase in radians
    frequencies : ndarray, optional, shape ``(n_frequencies,)`` — Hz
        Required when ``R`` is 2-D.
    is_broadband : bool — True iff ``R.ndim == 2``.
    """
    field_type = "reflection_coefficients"

    def __init__(
        self,
        *,
        theta: np.ndarray,
        R: np.ndarray,
        phi: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.theta = np.atleast_1d(np.asarray(theta, dtype=float))
        self.R = np.asarray(R, dtype=float)
        self.phi = np.asarray(phi, dtype=float)
        if self.R.ndim == 1:
            self.R = self.R.reshape(-1)
            self.phi = self.phi.reshape(-1)
            if not (len(self.theta) == len(self.R) == len(self.phi)):
                raise ConfigurationError(
                    f"ReflectionCoefficient: theta/R/phi length mismatch "
                    f"({len(self.theta)}, {len(self.R)}, {len(self.phi)})"
                )
        elif self.R.ndim == 2:
            if self.R.shape != self.phi.shape:
                raise ConfigurationError(
                    f"ReflectionCoefficient: R.shape {self.R.shape} != "
                    f"phi.shape {self.phi.shape}"
                )
            if self.R.shape[0] != len(self.theta):
                raise ConfigurationError(
                    f"ReflectionCoefficient.R: axis 0 ({self.R.shape[0]}) "
                    f"must equal len(theta) ({len(self.theta)})"
                )
            if self.frequencies is None:
                raise ConfigurationError(
                    "ReflectionCoefficient: 2-D R requires frequencies="
                )
            if self.R.shape[1] != len(self.frequencies):
                raise ConfigurationError(
                    f"ReflectionCoefficient.R: axis 1 ({self.R.shape[1]}) "
                    f"must equal len(frequencies) ({len(self.frequencies)})"
                )
        else:
            raise ConfigurationError(
                f"ReflectionCoefficient.R: must be 1-D or 2-D; "
                f"got shape {self.R.shape}"
            )

    @property
    def n_angles(self) -> int:
        return len(self.theta)

    @property
    def is_broadband(self) -> bool:
        return self.R.ndim == 2

    def _repr_extra(self) -> str:
        kind = 'broadband' if self.is_broadband else 'narrowband'
        return f"n_θ={self.n_angles}, {kind}"

    def at(
        self,
        *,
        angle: Optional[float] = None,
        frequency: Optional[float] = None,
    ) -> "ReflectionCoefficient":
        """Label-based slice along the angle and/or frequency axis.

        Either kwarg picks the nearest grid sample. ``frequency=`` is
        valid only for broadband (2-D) reflection coefficients; passing
        it on a narrowband instance raises ``ConfigurationError``.
        """
        if frequency is not None and not self.is_broadband:
            raise ConfigurationError(
                "ReflectionCoefficient.at: frequency= requires a broadband "
                "(2-D) reflection coefficient"
            )
        R = self.R
        phi = self.phi
        theta = self.theta
        freqs = self.frequencies
        if angle is not None:
            ai = int(np.argmin(np.abs(self.theta - angle)))
            theta = self.theta[ai:ai + 1]
            R = R[ai:ai + 1, ...] if R.ndim == 2 else R[ai:ai + 1]
            phi = phi[ai:ai + 1, ...] if phi.ndim == 2 else phi[ai:ai + 1]
        if frequency is not None:
            fi = int(np.argmin(np.abs(self.frequencies - frequency)))
            R = R[:, fi]
            phi = phi[:, fi]
            freqs = float(self.frequencies[fi])
        return ReflectionCoefficient(
            theta=theta, R=R, phi=phi,
            model=self.model, backend=self.backend,
            source_depths=self.source_depths,
            frequencies=freqs,
            metadata=dict(self.metadata),
        )

    @property
    def data(self) -> np.ndarray:
        return self.R

    @property
    def ranges(self) -> np.ndarray:   # convenience alias — angles double as the abscissa for plot helpers
        return self.theta

    @property
    def depths(self) -> np.ndarray:
        return np.array([0.0])
