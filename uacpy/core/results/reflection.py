"""Reflection-coefficient result type."""

from __future__ import annotations

import warnings

import numpy as np
from typing import Optional

from uacpy.core.acoustics.levels import transmission_loss_dB
from uacpy.core.exceptions import ConfigurationError, FallbackWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP

from uacpy.core.results._base import (PhaseReference, Result, _integer_index,
                                      coordinate_axis)
from uacpy.core.results.quantities import coordinate_unit


class ReflectionCoefficient(Result):
    """Angle-dependent reflection coefficient ``R(theta[, f])``.

    Unifies what Bounce and OASR produce. Used for both bottom (BRC) and
    top (TRC) reflection coefficients. ``magnitude`` and ``phase`` may be
    1-D (single-frequency) or 2-D (frequency-resolved); ``angles`` is always
    1-D.

    Attributes
    ----------
    angles : ndarray, shape ``(n_angles,)`` — grazing angles in degrees
    magnitude : ndarray, shape ``(n_angles,)`` or ``(n_angles, n_frequencies)``
            — ``|R|``. In ``[0, 1]`` for a reflection coefficient, and
            **not bounded by 1** when :attr:`reflection_type` is
            ``'transmission'``: a transmission coefficient is an amplitude
            ratio *across* an interface, so into a higher-impedance medium
            it exceeds 1 (Medwin & Clay, *Fundamentals of Acoustical
            Oceanography*, give ``T12 = 2ρ2c2cosθ1 / (ρ2c2cosθ1 +
            ρ1c1cosθ2)`` and note that air→water gives ``T12 ≈ 2``, "the
            pressure is doubled at the surface"; 1.15 measured here on a
            1700/400 m/s elastic half-space). Nothing in this class clamps
            or validates it, so ``-20·log10(R)`` is negative there.
    phase : ndarray, same shape as ``magnitude`` — phase in radians, assumed
            **unwrapped**; see :meth:`eval` for what a table carrying the
            ±π branch cut interpolates to.
    frequencies : ndarray, optional, shape ``(n_frequencies,)`` — Hz
        Required when ``magnitude`` is 2-D.
    is_broadband : bool — True iff ``magnitude.ndim == 2``.

    Sign convention of ``phase``: the package's travelling-wave form
    (:attr:`~uacpy.core.results.PhaseReference.TRAVELLING_WAVE`, time
    dependence ``exp(+iωt)``, propagator ``exp(-ikr)``), which is what
    Bounce writes and the Acoustics Toolbox engines read from a ``.brc`` /
    ``.trc`` table: a lossy fluid half-space below its critical angle has a
    **positive** ``phase``. A table derived under the physics ``exp(-iωt)``
    convention carries the opposite sign and must be conjugated before it
    is handed to this class. A result built without an explicit
    ``phase_reference`` is stamped ``'travelling_wave'``.

    :attr:`reflection_type` names what the magnitude column is (OASR's
    ``reflection_type``: ``'transmission'`` for a transmission
    coefficient), an attribute of the table (a ``metadata`` carrying
    ``reflection_type`` is refused); ``None`` is a reflection
    coefficient.
    """

    def __init__(
        self,
        *,
        angles: np.ndarray,
        magnitude: np.ndarray,
        phase: np.ndarray,
        reflection_type: Optional[str] = None,
        **kwargs,
    ):
        # What the column is belongs to the table; a metadata entry
        # naming it would be a second decider.
        if 'reflection_type' in (kwargs.get('metadata') or {}):
            raise ConfigurationError(
                "ReflectionCoefficient: metadata carries "
                "'reflection_type', which is an attribute of the table, "
                "not metadata.",
                remediation="Pass it as a keyword: ReflectionCoefficient("
                            "..., reflection_type=...).")
        kwargs.setdefault('phase_reference', PhaseReference.TRAVELLING_WAVE)
        super().__init__(**kwargs)
        self._reflection_type = (None if reflection_type is None
                                 else str(reflection_type))
        # A complex ``magnitude`` — the form reflection_coeff returns and
        # ``coefficient`` hands back — cast to float keeps its REAL part (not
        # |R|) and drops the phase with only numpy's ComplexWarning, naming
        # neither this class nor the remedy.
        for name, value in (('magnitude', magnitude), ('phase', phase)):
            if np.iscomplexobj(value):
                raise ConfigurationError(
                    f"ReflectionCoefficient: {name} is complex; magnitude is "
                    f"|R| and phase its phase (radians), both real.",
                    remediation="For a complex coefficient use "
                                "ReflectionCoefficient.from_complex(angles, "
                                "R_complex, frequencies=...).")
        # Copy on ingest (small arrays) so caller-side mutation can't corrupt
        # this result.
        self.angles = np.atleast_1d(np.array(angles, dtype=float))
        self.magnitude = np.array(magnitude, dtype=float)
        self.phase = np.array(phase, dtype=float)
        if not np.all(np.isfinite(self.angles)):
            raise ConfigurationError(
                "ReflectionCoefficient: angles holds a non-finite angle; every "
                "row of the table needs the grazing angle it is tabulated at.")
        if np.any(self.magnitude < 0.0):
            raise ConfigurationError(
                "ReflectionCoefficient: magnitude holds a negative value; "
                "a sign belongs in phase (a phase of pi).",
                remediation="For a signed or complex coefficient use "
                            "ReflectionCoefficient.from_complex(angles, "
                            "R_complex, frequencies=...).")
        if self.magnitude.ndim == 1:
            self.magnitude = self.magnitude.reshape(-1)
            self.phase = self.phase.reshape(-1)
            if not (len(self.angles) == len(self.magnitude) == len(self.phase)):
                raise ConfigurationError(
                    f"ReflectionCoefficient: angles/magnitude/phase length mismatch "
                    f"({len(self.angles)}, {len(self.magnitude)}, {len(self.phase)})"
                )
        elif self.magnitude.ndim == 2:
            if self.magnitude.shape != self.phase.shape:
                raise ConfigurationError(
                    f"ReflectionCoefficient: magnitude.shape "
                    f"{self.magnitude.shape} != phase.shape {self.phase.shape}."
                )
            if self.magnitude.shape[0] != len(self.angles):
                raise ConfigurationError(
                    f"ReflectionCoefficient.magnitude: axis 0 "
                    f"({self.magnitude.shape[0]}) must equal len(angles) "
                    f"({len(self.angles)})"
                )
            if self.frequencies is None:
                raise ConfigurationError(
                    "ReflectionCoefficient: a 2-D magnitude requires frequencies=."
                )
            if self.magnitude.shape[1] != len(self.frequencies):
                raise ConfigurationError(
                    f"ReflectionCoefficient.magnitude: axis 1 ({self.magnitude.shape[1]}) "
                    f"must equal len(frequencies) ({len(self.frequencies)})"
                )
        else:
            raise ConfigurationError(
                f"ReflectionCoefficient.magnitude: must be 1-D or 2-D; "
                f"got shape {self.magnitude.shape}."
            )

    @classmethod
    def from_complex(cls, angles, R_complex, frequencies=None
                     ) -> "ReflectionCoefficient":
        """A table from the complex coefficient ``R_complex = |R|·e^{iφ}``.

        ``magnitude`` is ``|R_complex|`` and ``phase`` its angle, unwrapped
        along ``angles`` (per frequency column for a 2-D table), which is the form
        :meth:`eval` interpolates and ``misc/RefCoef.f90:119`` assumes. The
        coefficient must be in the package's travelling-wave convention, as
        :func:`~uacpy.core.acoustics.reflection_coeff` returns it; a table
        derived under ``exp(-iωt)`` is conjugated first. :attr:`coefficient`
        returns it back.

        Parameters
        ----------
        angles : array_like, shape ``(n_angles,)``
            Grazing angles in degrees.
        R_complex : array_like, shape ``(n_angles,)`` or ``(n_angles, n_frequencies)``
            The complex reflection coefficient.
        frequencies : array_like, optional
            Hz; required for a 2-D table.
        """
        coefficient = np.asarray(R_complex)
        return cls(angles=angles, magnitude=np.abs(coefficient),
                   phase=np.unwrap(np.angle(coefficient), axis=0),
                   frequencies=frequencies)

    @property
    def reflection_type(self) -> Optional[str]:
        """What the magnitude column is, as OASR's ``reflection_type``
        names it (``'P-P'``, ``'transmission'``, ...), or ``None``: a
        reflection coefficient (Bounce, a table built by hand). A
        ``'transmission'`` table is not bounded by 1."""
        return self._reflection_type

    @property
    def n_angles(self) -> int:
        return len(self.angles)

    @property
    def is_broadband(self) -> bool:
        return self.magnitude.ndim == 2

    @property
    def coefficient(self) -> np.ndarray:
        """The complex coefficient ``|R|·exp(iφ)``, shape of :attr:`magnitude`.

        In the package's travelling-wave convention, like :attr:`phase`, so it
        compares directly with :func:`~uacpy.core.acoustics.reflection_coeff`
        and sums coherently with an engine's field.
        """
        return self.magnitude * np.exp(1j * self.phase)

    @property
    def dB(self) -> np.ndarray:
        """Reflection (bottom) loss ``-20·log10|R|`` in dB, shape of
        :attr:`magnitude`.

        The conversion is :func:`~uacpy.core.acoustics.transmission_loss_dB`'s,
        so a zero magnitude caps at the package's no-energy 600 dB rather than
        ``+inf``, as a Field's dB view does. A transmission table
        (:attr:`reflection_type` ``== 'transmission'``) can have
        ``|R| > 1``, and its dB view is then negative.
        """
        return transmission_loss_dB(self.magnitude)

    def _repr_bits(self) -> list:
        return [coordinate_axis('angle', self.angles), self.reflection_type]

    def to_dict(self) -> dict:
        """Serialise this table to plain arrays: ``angles`` (degrees),
        ``magnitude``, ``phase`` (radians), :attr:`reflection_type` and the
        identity, as :meth:`Field.to_dict` writes
        it. ``np.savez(f, **d)`` stores it; read it back with
        ``np.load(f, allow_pickle=True)`` into :meth:`from_dict`."""
        return {
            'angles': self.angles.copy(),
            'magnitude': self.magnitude.copy(),
            'phase': self.phase.copy(),
            'reflection_type': self._reflection_type,
            **self._identity_dict(),
        }

    def _axes(self):
        return ('angle', 'frequency') if self.magnitude.ndim == 2 else ('angle',)

    def _payload(self):
        return {'magnitude': (self.magnitude, self._axes(), ''),
                'phase': (self.phase, self._axes(), 'rad')}

    def _coords(self):
        coords = {'angle': (self.angles, coordinate_unit('angle'))}
        if self.magnitude.ndim == 2:
            coords['frequency'] = (self.frequencies,
                                   coordinate_unit('frequency'))
        return coords

    def _table(self):
        """Long form: one row per angle (per angle and frequency for a
        broadband table), columns ``angle`` (deg), ``frequency`` (Hz, a
        broadband table only), ``magnitude`` and ``phase`` (rad)."""
        if self.magnitude.ndim == 1:
            return {'angle': self.angles.copy(),
                    'magnitude': self.magnitude.copy(),
                    'phase': self.phase.copy()}
        n_f = len(self.frequencies)
        return {'angle': np.repeat(self.angles, n_f),
                'frequency': np.tile(np.asarray(self.frequencies), len(self.angles)),
                'magnitude': self.magnitude.ravel().copy(),
                'phase': self.phase.ravel().copy()}

    def _export_attrs(self):
        attrs = super()._export_attrs()
        if self._reflection_type is not None:
            attrs['reflection_type'] = self._reflection_type
        return attrs

    @classmethod
    def _from_export(cls, arrays, attrs):
        return cls(angles=arrays['angle'], magnitude=arrays['magnitude'],
                   phase=arrays['phase'],
                   reflection_type=attrs.get('reflection_type'),
                   **cls._identity_from_attrs(attrs, ('reflection_type',)))

    @classmethod
    def from_dict(cls, d: dict) -> "ReflectionCoefficient":
        """Reconstruct a :class:`ReflectionCoefficient` from :meth:`to_dict`
        output, or from the mapping ``np.load(f, allow_pickle=True)`` returns
        for a file written with ``np.savez(f, **table.to_dict())``.

        Parameters
        ----------
        d : mapping
            :meth:`to_dict` output, or the mapping ``np.load`` returns for it.
        """
        d = cls._unwrap_saved(d, payload=('angles', 'magnitude', 'phase'))
        identity = cls._identity_from_dict(d)
        # A file that keeps the type in its metadata loads it from there.
        metadata = dict(identity['metadata'] or {})
        kept = metadata.pop('reflection_type', None)
        identity['metadata'] = metadata or None
        stated = d.get('reflection_type')
        return cls(angles=d['angles'], magnitude=d['magnitude'], phase=d['phase'],
                   reflection_type=kept if stated is None else stated,
                   **identity)

    def at(self, **kwargs) -> "ReflectionCoefficient":
        """Nearest-sample slice along the ``angle`` and/or ``frequency`` axis.

        The generic ``Field.at``-style form: each kwarg names an axis and gives
        a value; the nearest stored sample is selected (never fabricated) and
        that axis collapsed. Axes are ``angle`` (the abscissa) and
        ``frequency``; ``frequency=`` is valid only for broadband
        (2-D) coefficients. See :meth:`eval` (interpolate) and :meth:`isel`
        (positional index).

        Collapse is deliberately asymmetric because ``angles`` is the
        permanent abscissa of this type (a 1-D ``magnitude`` is *by
        definition* indexed by ``angles``): selecting one **frequency**
        removes the optional second dimension and returns a narrowband
        ``R(theta)`` of shape ``(n_angles,)``; selecting one **angle** keeps
        ``angles`` as a length-1 axis, so ``magnitude`` stays 2-D
        ``(1, n_frequencies)``. This differs from
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
        ``'cubic'``); constant extrapolation past the ends.

        **:attr:`phase` must already be unwrapped.** The phase column is
        interpolated as a plain real column, exactly as Acoustics-Toolbox
        does (``misc/RefCoef.f90:167`` is the same
        ``(1-alpha)*phi_left + alpha*phi_right``), and AT states the
        precondition that goes with it: "Assumes phi has been unwrapped so
        that it varies smoothly" (``misc/RefCoef.f90:119``). uacpy holds the
        same assumption, so a table that steps across the ±π branch cut
        interpolates *through* the short way rather than around it. On
        ``angles = [10, 20, 30]`` with ``phase = [2.5, 3.0, -3.0]`` radians,
        ``eval(angle=25)`` returns 0.0 where the unwrapped table returns
        3.1416 — a phase reversal reported as no phase shift at all.

        Nothing here unwraps for you, and that is deliberate: silently
        unwrapping would make this method disagree with every AT solver
        reading the same file. Call ``np.unwrap`` on the phase column when
        you build the table from a source that wraps it.

        :meth:`at` and ``method='nearest'`` return a tabulated sample rather
        than a blend, so neither is affected."""
        method = kwargs.pop('method', 'linear')
        angle, frequency = self._resolve_axes(kwargs, 'eval')
        return self._select(angle, frequency, method=method)

    def _resolve_axes(self, kwargs, who):
        valid = {'angle', 'frequency'}
        unknown = sorted(set(kwargs) - valid)
        if unknown:
            raise ConfigurationError(
                f"ReflectionCoefficient.{who}: unknown axis {unknown}; "
                f"available: ['angle', 'frequency']."
            )
        angle = kwargs.get('angle')
        frequency = kwargs.get('frequency')
        if frequency is not None and not self.is_broadband:
            raise ConfigurationError(
                f"ReflectionCoefficient.{who}: frequency= requires a broadband "
                f"(2-D) reflection coefficient."
            )
        # Holding the end value matches every other carrier's eval, but it is
        # not what a solver reading the same table does: RefCoef.f90 sets
        # R = 0 and phi = 0 both below the table (``:139-140``) and above it
        # (``:146-147``), killing the ray.
        # Warn rather than return 0 — a lone carrier with a different
        # extrapolation rule would be a worse trap, and 0 is AT's kill
        # convention, not a claim about R at that angle.
        # ``isel``'s angle is a positional index, not degrees, so the
        # comparison below is meaningless there.
        if who != 'isel' and angle is not None and np.size(self.angles):
            lo, hi = float(np.min(self.angles)), float(np.max(self.angles))
            a = np.asarray(angle, dtype=float)
            if np.any(a < lo) or np.any(a > hi):
                warnings.warn(
                    f"ReflectionCoefficient.{who}: angle {angle} is outside the "
                    f"tabulated range {lo:g}-{hi:g} deg; the end value is held "
                    f"(the returned .angles records the angle actually used). "
                    f"Acoustics-Toolbox does not hold — "
                    f"misc/RefCoef.f90:139-140,146-147 sets R = 0 and phi = 0 "
                    f"outside the table, so a ray at this angle is killed by "
                    f"the solver rather than given the end value.",
                    FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
                )
        return angle, frequency

    def _build(self, theta, R, phi, freqs) -> "ReflectionCoefficient":
        return ReflectionCoefficient(
            angles=theta, magnitude=R, phase=phi,
            reflection_type=self._reflection_type,
            **dict(self.id_kwargs(), frequencies=freqs),
        )

    def _select(self, angle, frequency, *, method) -> "ReflectionCoefficient":
        from uacpy.core._grid import collapse_axis
        R, phi, theta = self.magnitude, self.phase, self.angles
        freqs = self.frequencies
        if angle is not None:
            R, av = collapse_axis(R, self.angles, angle, method, axis=0,
                                  name='angle')
            phi, _ = collapse_axis(phi, self.angles, angle, method, axis=0,
                                   name='angle')
            R, phi, theta = R[None, ...], phi[None, ...], np.array([av])
        if frequency is not None:
            R, fv = collapse_axis(R, self.frequencies, frequency, method,
                                  axis=1, name='frequency')
            phi, _ = collapse_axis(phi, self.frequencies, frequency, method,
                                   axis=1, name='frequency')
            freqs = float(fv)
        return self._build(theta, R, phi, freqs)

    def _index_select(self, angle, frequency) -> "ReflectionCoefficient":
        R, phi, theta = self.magnitude, self.phase, self.angles
        freqs = self.frequencies
        if angle is not None:
            ai = _integer_index(angle, "ReflectionCoefficient.isel: angle")
            theta = self.angles[[ai]]                 # raises IndexError if OOB
            R = R[[ai], ...] if R.ndim == 2 else R[[ai]]
            phi = phi[[ai], ...] if phi.ndim == 2 else phi[[ai]]
        if frequency is not None:
            fi = _integer_index(frequency,
                                "ReflectionCoefficient.isel: frequency")
            R, phi = R[:, fi], phi[:, fi]
            freqs = float(self.frequencies[fi])
        return self._build(theta, R, phi, freqs)
