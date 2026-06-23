"""Reflection-coefficient result type."""

from __future__ import annotations

import numpy as np

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
        # Copy on ingest (small arrays) so caller-side mutation can't corrupt
        # this result.
        self.theta = np.atleast_1d(np.array(theta, dtype=float))
        self.R = np.array(R, dtype=float)
        self.phi = np.array(phi, dtype=float)
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

    def at(self, **kwargs) -> "ReflectionCoefficient":
        """Nearest-sample slice along the ``angle`` and/or ``frequency`` axis.

        The generic ``Field.at``-style form: each kwarg names an axis and gives
        a value; the nearest stored sample is selected (never fabricated) and
        that axis collapsed. Axes are ``angle`` (a.k.a. ``theta`` — the
        abscissa) and ``frequency``; ``frequency=`` is valid only for broadband
        (2-D) coefficients. See :meth:`eval` (interpolate) and :meth:`isel`
        (positional index).

        Collapse is deliberately asymmetric because ``theta`` is the permanent
        abscissa of this type (a 1-D ``R`` is *by definition* indexed by
        ``theta``): selecting one **frequency** removes the optional second
        dimension and returns a narrowband ``R(theta)`` of shape
        ``(n_angles,)``; selecting one **angle** keeps ``theta`` as a length-1
        axis, so ``R`` stays 2-D ``(1, n_frequencies)``. This differs from
        ``Field.at`` (which drops any sliced axis) precisely because a
        reflection coefficient is anchored on its angle abscissa."""
        angle, frequency = self._resolve_axes(kwargs, 'at')
        return self._select(angle, frequency, method='nearest')

    def isel(self, **kwargs) -> "ReflectionCoefficient":
        """Integer-index slice — the positional counterpart of :meth:`at`."""
        angle, frequency = self._resolve_axes(kwargs, 'isel')
        return self._index_select(angle, frequency)

    def eval(self, **kwargs) -> "ReflectionCoefficient":
        """Interpolated slice — the interpolating counterpart of :meth:`at`.

        ``method=`` picks the scheme (``'linear'`` default, ``'nearest'``,
        ``'cubic'``); constant extrapolation past the ends."""
        method = kwargs.pop('method', 'linear')
        angle, frequency = self._resolve_axes(kwargs, 'eval')
        return self._select(angle, frequency, method=method)

    def _resolve_axes(self, kwargs, who):
        valid = {'angle', 'theta', 'frequency'}
        unknown = sorted(set(kwargs) - valid)
        if unknown:
            raise ConfigurationError(
                f"ReflectionCoefficient.{who}: unknown axis {unknown}; "
                f"available: ['angle' (a.k.a. 'theta'), 'frequency']"
            )
        if 'angle' in kwargs and 'theta' in kwargs:
            raise ConfigurationError(
                f"ReflectionCoefficient.{who}: 'angle' and 'theta' are "
                f"synonyms; pass only one."
            )
        angle = kwargs.get('angle', kwargs.get('theta'))
        frequency = kwargs.get('frequency')
        if frequency is not None and not self.is_broadband:
            raise ConfigurationError(
                f"ReflectionCoefficient.{who}: frequency= requires a broadband "
                f"(2-D) reflection coefficient"
            )
        return angle, frequency

    def _build(self, theta, R, phi, freqs) -> "ReflectionCoefficient":
        return ReflectionCoefficient(
            theta=theta, R=R, phi=phi,
            model=self.model, backend=self.backend,
            source_depths=self.source_depths,
            frequencies=freqs, metadata=dict(self.metadata),
        )

    def _select(self, angle, frequency, *, method) -> "ReflectionCoefficient":
        from uacpy.core._grid import collapse_axis
        R, phi, theta = self.R, self.phi, self.theta
        freqs = self.frequencies
        if angle is not None:
            R, av = collapse_axis(R, self.theta, angle, method, axis=0)
            phi, _ = collapse_axis(phi, self.theta, angle, method, axis=0)
            R, phi, theta = R[None, ...], phi[None, ...], np.array([av])
        if frequency is not None:
            R, fv = collapse_axis(R, self.frequencies, frequency, method, axis=1)
            phi, _ = collapse_axis(phi, self.frequencies, frequency, method, axis=1)
            freqs = float(fv)
        return self._build(theta, R, phi, freqs)

    def _index_select(self, angle, frequency) -> "ReflectionCoefficient":
        R, phi, theta = self.R, self.phi, self.theta
        freqs = self.frequencies
        if angle is not None:
            ai = int(angle)
            theta = self.theta[[ai]]                 # raises IndexError if OOB
            R = R[[ai], ...] if R.ndim == 2 else R[[ai]]
            phi = phi[[ai], ...] if phi.ndim == 2 else phi[[ai]]
        if frequency is not None:
            fi = int(frequency)
            R, phi = R[:, fi], phi[:, fi]
            freqs = float(self.frequencies[fi])
        return self._build(theta, R, phi, freqs)

    @property
    def data(self) -> np.ndarray:
        return self.R

    @property
    def ranges(self) -> np.ndarray:   # convenience alias — angles double as the abscissa for plot helpers
        return self.theta

    @property
    def depths(self) -> np.ndarray:
        return np.array([0.0])
