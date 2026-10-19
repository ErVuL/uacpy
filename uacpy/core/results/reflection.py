"""Reflection-coefficient result type, and the plane-wave bottom loss of a
fluid seabed (:func:`bottom_loss_curve`) built as one."""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
from typing import Optional, Sequence

from uacpy.core.acoustics.boundaries import critical_angle, reflection_coeff
from uacpy.core.acoustics.levels import transmission_loss_dB
from uacpy.core.bottom import Bottom
from uacpy.core.constants import (DEFAULT_SOUND_SPEED,
                                  DEFAULT_WATER_DENSITY)
from uacpy.core.exceptions import ConfigurationError, FallbackWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP

from uacpy.core.results._base import (PhaseReference, Result, _integer_index,
                                      coordinate_axis)
from uacpy.core.results.quantities import coordinate_unit


class ReflectionCoefficient(Result):
    """Angle-dependent reflection coefficient ``R(theta[, f])``.

    Unifies what Bounce and OASR produce. Used for both bottom (BRC) and
    top (TRC) reflection coefficients. ``magnitude`` and ``phase`` may be
    1-D (single-frequency) or 2-D (frequency-resolved); ``angles_deg`` is always
    1-D.

    Attributes
    ----------
    angles_deg : ndarray, shape ``(n_angles,)`` — grazing angles in degrees
    magnitude : ndarray, shape ``(n_angles,)`` or ``(n_angles, n_frequencies)``
            — ``|R|``. In ``[0, 1]`` for a reflection coefficient, and
            **not bounded by 1** when :attr:`reflection_type` is
            ``'transmission'`` or ``'P-SV transmission'``: a transmission
            coefficient is an amplitude
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
    coefficient. :attr:`critical_angle` is the boundary's compressional
    critical grazing angle (degrees) when the table's producer knows it —
    :func:`bottom_loss_curve` does — NaN for a boundary slower than the
    water (it has none), ``None`` when not known.
    """

    #: The plotter that draws this result (``.plot()``).
    _plotter = '_plot_reflection_coefficient'

    def __init__(
        self,
        *,
        angles_deg: np.ndarray,
        magnitude: np.ndarray,
        phase: np.ndarray,
        reflection_type: Optional[str] = None,
        critical_angle_deg: Optional[float] = None,
        **kwargs,
    ):
        # What the column is belongs to the table; a metadata entry
        # naming it would be a second decider.
        carried = sorted({'reflection_type', 'critical_angle_deg'}
                         & set(kwargs.get('metadata') or {}))
        if carried:
            raise ConfigurationError(
                f"ReflectionCoefficient: metadata carries {carried}, which "
                f"are attributes of the table, not metadata.",
                remediation="Pass them as keywords: ReflectionCoefficient("
                            "..., " + ", ".join(f"{k}=..." for k in carried)
                            + ").")
        kwargs.setdefault('phase_reference', PhaseReference.TRAVELLING_WAVE)
        super().__init__(**kwargs)
        self._reflection_type = (None if reflection_type is None
                                 else str(reflection_type))
        self._critical_angle = (None if critical_angle_deg is None
                                else float(critical_angle_deg))
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
                                "ReflectionCoefficient.from_complex(angles_deg, "
                                "R_complex, frequencies=...).")
        # Copy on ingest (small arrays) so caller-side mutation can't corrupt
        # this result.
        self.angles_deg = np.atleast_1d(np.array(angles_deg, dtype=float))
        self.magnitude = np.array(magnitude, dtype=float)
        self.phase = np.array(phase, dtype=float)
        if not np.all(np.isfinite(self.angles_deg)):
            raise ConfigurationError(
                "ReflectionCoefficient: angles_deg holds a non-finite angle; every "
                "row of the table needs the grazing angle it is tabulated at.")
        if np.any(self.magnitude < 0.0):
            raise ConfigurationError(
                "ReflectionCoefficient: magnitude holds a negative value; "
                "a sign belongs in phase (a phase of pi).",
                remediation="For a signed or complex coefficient use "
                            "ReflectionCoefficient.from_complex(angles_deg, "
                            "R_complex, frequencies=...).")
        if self.magnitude.ndim == 1:
            self.magnitude = self.magnitude.reshape(-1)
            self.phase = self.phase.reshape(-1)
            if not (len(self.angles_deg) == len(self.magnitude) == len(self.phase)):
                raise ConfigurationError(
                    f"ReflectionCoefficient: angles_deg/magnitude/phase length mismatch "
                    f"({len(self.angles_deg)}, {len(self.magnitude)}, {len(self.phase)})"
                )
        elif self.magnitude.ndim == 2:
            if self.magnitude.shape != self.phase.shape:
                raise ConfigurationError(
                    f"ReflectionCoefficient: magnitude.shape "
                    f"{self.magnitude.shape} != phase.shape {self.phase.shape}."
                )
            if self.magnitude.shape[0] != len(self.angles_deg):
                raise ConfigurationError(
                    f"ReflectionCoefficient.magnitude: axis 0 "
                    f"({self.magnitude.shape[0]}) must equal len(angles_deg) "
                    f"({len(self.angles_deg)})"
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
    def from_complex(cls, angles_deg, R_complex, frequencies=None, *,
                     critical_angle_deg: Optional[float] = None
                     ) -> "ReflectionCoefficient":
        """A table from the complex coefficient ``R_complex = |R|·e^{iφ}``.

        ``magnitude`` is ``|R_complex|`` and ``phase`` its angle, unwrapped
        along ``angles_deg`` (per frequency column for a 2-D table), which is the form
        :meth:`eval` interpolates and ``misc/RefCoef.f90:119`` assumes. The
        coefficient must be in the package's travelling-wave convention, as
        :func:`~uacpy.core.acoustics.reflection_coeff` returns it; a table
        derived under ``exp(-iωt)`` is conjugated first. :attr:`coefficient`
        returns it back.

        Parameters
        ----------
        angles_deg : array_like, shape ``(n_angles,)``
            Grazing angles in degrees.
        R_complex : array_like, shape ``(n_angles,)`` or ``(n_angles, n_frequencies)``
            The complex reflection coefficient.
        frequencies : array_like, optional
            Hz; required for a 2-D table.
        critical_angle_deg : float, optional
            The boundary's critical grazing angle (deg), when known.
        """
        coefficient = np.asarray(R_complex)
        return cls(angles_deg=angles_deg, magnitude=np.abs(coefficient),
                   phase=np.unwrap(np.angle(coefficient), axis=0),
                   frequencies=frequencies, critical_angle_deg=critical_angle_deg)

    @classmethod
    def stack(cls, tables: Sequence["ReflectionCoefficient"], *,
              angles_deg=None) -> "ReflectionCoefficient":
        """One broadband table ``R(θ, f)`` from narrowband tables, one per
        frequency — several Bounce or OASR runs, say.

        Each table holds one frequency (its ``frequencies``); the columns are
        sorted by frequency. The tables share their ``angles_deg``, or, given
        ``angles_deg=``, each is read there by linear interpolation of its
        magnitude and its (unwrapped) phase — the rule :meth:`eval` and
        ``misc/RefCoef.f90:167`` apply — which puts tables a solver
        tabulated on grids of its own (BOUNCE chooses one per frequency) on
        one axis; an angle outside a table is refused. The table keeps the
        first one's identity and ``reflection_type`` (which must be one for
        all), and the ``critical_angle_deg`` they all carry, else none.

        Parameters
        ----------
        tables : sequence of ReflectionCoefficient
            Narrowband tables at distinct frequencies.
        angles_deg : array_like, optional
            Grazing angles (deg) to read every table at; ``None`` takes the
            tables' shared axis.

        Returns
        -------
        ReflectionCoefficient
            The broadband table; ``.plot()`` draws ``|R|(θ, f)``.
        """
        tables = list(tables)
        who = 'ReflectionCoefficient.stack'
        if len(tables) < 2:
            raise ConfigurationError(
                f"{who}: needs two or more narrowband tables; got "
                f"{len(tables)}.")
        for i, t in enumerate(tables):
            if not isinstance(t, ReflectionCoefficient) or t.is_broadband:
                raise ConfigurationError(
                    f"{who}: table {i} is not a narrowband "
                    f"ReflectionCoefficient.",
                    remediation="Pick one frequency of a broadband table "
                                "first: rc.at(frequency=f).")
            if t.frequencies is None or np.size(t.frequencies) != 1:
                raise ConfigurationError(
                    f"{who}: table {i} holds no single frequency "
                    f"(frequencies={t.frequencies}); a column of R(θ, f) "
                    f"needs its frequency.")
            if angles_deg is None and not (
                    t.angles_deg.shape == tables[0].angles_deg.shape
                    and np.array_equal(t.angles_deg, tables[0].angles_deg)):
                raise ConfigurationError(
                    f"{who}: table {i} is on other grazing angles than "
                    f"table 0; the columns of R(θ, f) share one angle axis.",
                    remediation="Pass angles_deg= to read every table on one "
                                "grid, or run every frequency on one.")
            if angles_deg is not None:
                grid = np.asarray(angles_deg, dtype=float)
                if (grid.min() < t.angles_deg.min()
                        or grid.max() > t.angles_deg.max()):
                    raise ConfigurationError(
                        f"{who}: angles_deg= spans {grid.min():g}-"
                        f"{grid.max():g}°, beyond table {i}'s "
                        f"{t.angles_deg.min():g}-{t.angles_deg.max():g}°; a table "
                        f"is read inside its own angles only.")
            if t.reflection_type != tables[0].reflection_type:
                raise ConfigurationError(
                    f"{who}: table {i} is a {t.reflection_type!r} table and "
                    f"table 0 a {tables[0].reflection_type!r} one; one "
                    f"quantity per table.")
        freqs = np.array([float(np.atleast_1d(t.frequencies)[0])
                          for t in tables])
        if np.unique(freqs).size != freqs.size:
            raise ConfigurationError(
                f"{who}: two tables share a frequency ({freqs.tolist()} "
                f"Hz); one column per frequency.")
        order = np.argsort(freqs)
        critical = {t.critical_angle_deg for t in tables}
        first = tables[0]
        if angles_deg is None:
            grid = first.angles_deg
            columns = [(tables[i].magnitude, tables[i].phase) for i in order]
        else:
            grid = np.asarray(angles_deg, dtype=float)
            columns = [(np.interp(grid, tables[i].angles_deg,
                                  tables[i].magnitude),
                        np.interp(grid, tables[i].angles_deg, tables[i].phase))
                       for i in order]
        return cls(angles_deg=grid,
                   magnitude=np.column_stack([m for m, _ in columns]),
                   phase=np.column_stack([p for _, p in columns]),
                   reflection_type=first.reflection_type,
                   critical_angle_deg=(critical.pop() if len(critical) == 1
                                       else None),
                   **dict(first.id_kwargs(), frequencies=freqs[order]))

    @property
    def reflection_type(self) -> Optional[str]:
        """What the magnitude column is, as OASR's ``reflection_type``
        names it (``'P-P'``, ``'transmission'``, ...), or ``None``: a
        reflection coefficient (Bounce, a table built by hand). A
        ``'transmission'`` table is not bounded by 1."""
        return self._reflection_type

    @property
    def critical_angle_deg(self) -> Optional[float]:
        """The boundary's compressional critical grazing angle (degrees),
        ``arccos(c_water / c_boundary)``, when the table's producer knows it
        (:func:`bottom_loss_curve`); NaN for a boundary slower than the
        water, which has none; ``None`` when not known. The loss view of
        :meth:`plot` marks it with ``mark_critical=True``."""
        return self._critical_angle

    @property
    def n_angles(self) -> int:
        return len(self.angles_deg)

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
        return [coordinate_axis('angle', self.angles_deg), self.reflection_type]

    def to_dict(self) -> dict:
        """Serialise this table to plain arrays: ``angles_deg`` (degrees),
        ``magnitude``, ``phase`` (radians), :attr:`reflection_type` and the
        identity, as :meth:`Field.to_dict` writes
        it. ``np.savez(f, **d)`` stores it; read it back with
        ``np.load(f, allow_pickle=True)`` into :meth:`from_dict`. That load
        unpickles, so use it only on files you wrote; :meth:`to_netcdf` /
        :meth:`from_netcdf` is the route for a file from elsewhere."""
        return {
            'angles_deg': self.angles_deg.copy(),
            'magnitude': self.magnitude.copy(),
            'phase': self.phase.copy(),
            'reflection_type': self._reflection_type,
            'critical_angle_deg': self._critical_angle,
            **self._identity_dict(),
        }

    def _axes(self):
        return ('angle', 'frequency') if self.magnitude.ndim == 2 else ('angle',)

    def _payload(self):
        return {'magnitude': (self.magnitude, self._axes(), ''),
                'phase': (self.phase, self._axes(), 'rad')}

    def _coords(self):
        coords = {'angle': (self.angles_deg, coordinate_unit('angle'))}
        if self.magnitude.ndim == 2:
            coords['frequency'] = (self.frequencies,
                                   coordinate_unit('frequency'))
        return coords

    def _table(self):
        """Long form: one row per angle (per angle and frequency for a
        broadband table), columns ``angle`` (deg), ``frequency`` (Hz, a
        broadband table only), ``magnitude`` and ``phase`` (rad)."""
        if self.magnitude.ndim == 1:
            return {'angle': self.angles_deg.copy(),
                    'magnitude': self.magnitude.copy(),
                    'phase': self.phase.copy()}
        n_f = len(self.frequencies)
        return {'angle': np.repeat(self.angles_deg, n_f),
                'frequency': np.tile(np.asarray(self.frequencies), len(self.angles_deg)),
                'magnitude': self.magnitude.ravel().copy(),
                'phase': self.phase.ravel().copy()}

    def _export_attrs(self):
        attrs = super()._export_attrs()
        if self._reflection_type is not None:
            attrs['reflection_type'] = self._reflection_type
        if self._critical_angle is not None:
            attrs['critical_angle_deg'] = self._critical_angle
        return attrs

    @classmethod
    def _from_export(cls, arrays, attrs):
        return cls(angles_deg=arrays['angle'], magnitude=arrays['magnitude'],
                   phase=arrays['phase'],
                   reflection_type=attrs.get('reflection_type'),
                   critical_angle_deg=attrs.get('critical_angle_deg'),
                   **cls._identity_from_attrs(
                       attrs, ('reflection_type', 'critical_angle_deg')))

    @classmethod
    def from_dict(cls, d: dict) -> "ReflectionCoefficient":
        """Reconstruct a :class:`ReflectionCoefficient` from :meth:`to_dict`
        output, or from the mapping ``np.load(f, allow_pickle=True)`` returns
        for a file written with ``np.savez(f, **table.to_dict())``. Load such
        a file only if you wrote it: ``allow_pickle=True`` unpickles.

        Parameters
        ----------
        d : mapping
            :meth:`to_dict` output, or the mapping ``np.load`` returns for it.
        """
        d = cls._unwrap_saved(d, payload=('angles_deg', 'magnitude', 'phase'),
                              required=('reflection_type', 'critical_angle_deg'))
        identity = cls._identity_from_dict(d)
        return cls(angles_deg=d['angles_deg'], magnitude=d['magnitude'], phase=d['phase'],
                   reflection_type=d['reflection_type'],
                   critical_angle_deg=d['critical_angle_deg'], **identity)

    def at(self, **kwargs) -> "ReflectionCoefficient":
        """Nearest-sample slice along the ``angle`` and/or ``frequency`` axis.

        The generic ``Field.at``-style form: each kwarg names an axis and gives
        a value; the nearest stored sample is selected (never fabricated) and
        that axis collapsed. Axes are ``angle`` (the abscissa) and
        ``frequency``; ``frequency=`` is valid only for broadband
        (2-D) coefficients. See :meth:`eval` (interpolate) and :meth:`isel`
        (positional index).

        Collapse is deliberately asymmetric because ``angles_deg`` is the
        permanent abscissa of this type (a 1-D ``magnitude`` is *by
        definition* indexed by ``angles_deg``): selecting one **frequency**
        removes the optional second dimension and returns a narrowband
        ``R(theta)`` of shape ``(n_angles,)``; selecting one **angle** keeps
        ``angles_deg`` as a length-1 axis, so ``magnitude`` stays 2-D
        ``(1, n_frequencies)``. This differs from
        ``Field.at`` (which drops any sliced axis) precisely because a
        reflection coefficient is anchored on its angle abscissa."""
        angle, frequency = self._resolve_axes(kwargs, 'at')
        return self._select(angle, frequency, method='nearest',
                            who='ReflectionCoefficient.at')

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
        ``angles_deg = [10, 20, 30]`` with ``phase = [2.5, 3.0, -3.0]`` radians,
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
        return self._select(angle, frequency, method=method,
                            who='ReflectionCoefficient.eval')

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
        # comparison below is meaningless there. A label is checked (one
        # real finite number, the rule every selection shares) before the
        # warning speaks of it.
        if who == 'isel':
            return angle, frequency
        from uacpy.core._grid import finite_scalar_label
        if frequency is not None:
            finite_scalar_label(frequency, 'frequency',
                                who=f"ReflectionCoefficient.{who}",
                                coords=self.frequencies)
        if angle is not None:
            a = finite_scalar_label(angle, 'angle',
                                    who=f"ReflectionCoefficient.{who}",
                                    coords=self.angles_deg)
        if angle is not None and np.size(self.angles_deg):
            lo, hi = float(np.min(self.angles_deg)), float(np.max(self.angles_deg))
            if a < lo or a > hi:
                warnings.warn(
                    f"ReflectionCoefficient.{who}: angle {angle} is outside the "
                    f"tabulated range {lo:g}-{hi:g} deg; the end value is held "
                    f"(the returned .angles_deg records the angle actually used). "
                    f"Acoustics-Toolbox does not hold — "
                    f"misc/RefCoef.f90:139-140,146-147 sets R = 0 and phi = 0 "
                    f"outside the table, so a ray at this angle is killed by "
                    f"the solver rather than given the end value.",
                    FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
                )
        return angle, frequency

    def _build(self, theta, R, phi, freqs) -> "ReflectionCoefficient":
        return ReflectionCoefficient(
            angles_deg=theta, magnitude=R, phase=phi,
            reflection_type=self._reflection_type,
            critical_angle_deg=self._critical_angle,
            **dict(self.id_kwargs(), frequencies=freqs),
        )

    def _select(self, angle, frequency, *, method,
                who) -> "ReflectionCoefficient":
        from uacpy.core._grid import collapse_axis
        R, phi, theta = self.magnitude, self.phase, self.angles_deg
        freqs = self.frequencies
        if angle is not None:
            R, av = collapse_axis(R, self.angles_deg, angle, method, axis=0,
                                  name='angle', who=who)
            phi, _ = collapse_axis(phi, self.angles_deg, angle, method, axis=0,
                                   name='angle', who=who)
            R, phi, theta = R[None, ...], phi[None, ...], np.array([av])
        if frequency is not None:
            R, fv = collapse_axis(R, self.frequencies, frequency, method,
                                  axis=1, name='frequency', who=who)
            phi, _ = collapse_axis(phi, self.frequencies, frequency, method,
                                   axis=1, name='frequency', who=who)
            freqs = float(fv)
        return self._build(theta, R, phi, freqs)

    def _index_select(self, angle, frequency) -> "ReflectionCoefficient":
        R, phi, theta = self.magnitude, self.phase, self.angles_deg
        freqs = self.frequencies
        if angle is not None:
            ai = _integer_index(angle, "ReflectionCoefficient.isel: angle")
            theta = self.angles_deg[[ai]]                 # raises IndexError if OOB
            R = R[[ai], ...] if R.ndim == 2 else R[[ai]]
            phi = phi[[ai], ...] if phi.ndim == 2 else phi[[ai]]
        if frequency is not None:
            fi = _integer_index(frequency,
                                "ReflectionCoefficient.isel: frequency")
            R, phi = R[:, fi], phi[:, fi]
            freqs = float(self.frequencies[fi])
        return self._build(theta, R, phi, freqs)


def bottom_loss_curve(
    material,
    *,
    grazing_angles_deg: Optional[np.ndarray] = None,
    water_sound_speed: float = DEFAULT_SOUND_SPEED,
    water_density: float = DEFAULT_WATER_DENSITY,
) -> ReflectionCoefficient:
    """Plane-wave fluid–fluid reflection of a seabed against grazing angle,
    as a :class:`ReflectionCoefficient` whose loss view is the bottom loss.

    :func:`~uacpy.core.acoustics.reflection_coeff` swept over grazing angle
    for a seabed given by name, properties or carrier, with its critical
    angle (:func:`~uacpy.core.acoustics.critical_angle`) carried as
    ``critical_angle_deg``. ``rc.plot(value='loss')`` draws the loss
    ``-20·log10|R|`` (``rc.dB``), ``mark_critical=True`` rules the critical
    angle, and ``label=``/``ax=`` overlay several seabeds. Shear is ignored
    (fluid–fluid only): a preset name is its fluid curve, and a carrier or
    dict whose ``shear_speed`` is positive
    (``BoundaryProperties.from_preset(name, elastic=True)``, the half-space
    of an elastic environment) is evaluated as that fluid with a
    :class:`~uacpy.core.exceptions.FallbackWarning`. An elastic seabed also
    loses energy to shear waves (Jensen et al., *Computational Ocean
    Acoustics*, §1.6.2, Eq. 1.61), most between the shear and compressional
    critical angles: lossless basalt (c_p 5250, c_s 2500 m/s) loses 2.7-3.4 dB
    per bounce at 55-70° grazing where this curve gives 0 dB. The elastic
    coefficient is :func:`~uacpy.core.acoustics.reflection_coeff` with
    ``shear_speed=`` (Brekhovskikh & Lysanov Eq. 3.4.22), as BOUNCE and OASR
    compute it.

    Parameters
    ----------
    material : str, dict, BoundaryProperties, SedimentLayer or Bottom
        Preset name (``'sand'``, ``'silt'``, …), a dict carrying
        ``sound_speed`` (m/s), ``density`` (g/cm³), ``attenuation``
        (dB/λ_p), a seabed carrier holding those three attributes — e.g.
        ``env.bottom.halfspace_at(range=0.0)`` — or an unlayered,
        range-independent ``Bottom`` (``env.bottom``), read at its
        half-space (a layered or range-dependent one is refused). A carrier's
        ``acoustic_type`` decides what those attributes mean: a
        ``'half-space'`` reads them; a ``'rigid'`` boundary reflects with
        ``R = +1`` and a ``'vacuum'`` (pressure-release) one with ``R = -1``
        at every angle (0 dB loss), their three numbers construction
        placeholders that are not read; a ``'file'`` or ``'precalc'``
        boundary carries its loss in a reflection table and is refused.
    grazing_angles_deg : array_like, optional
        Grazing-angle grid in degrees (``0`` = parallel to interface,
        ``90`` = normal incidence). Default ``np.linspace(0, 90, 181)``.
    water_sound_speed, water_density : float
        Water-column reference values (m/s, g/cm³). ``water_density``
        defaults to :data:`~uacpy.core.constants.DEFAULT_WATER_DENSITY`
        (1.027 g/cm³), the value every deck writes for the water column, so
        the curve matches what the propagation models see; pass ``1.0`` to
        reproduce a textbook benchmark that takes ρ_w = 1 (on 'sand' the
        two differ by at most 0.28 dB, at normal incidence).

    Returns
    -------
    ReflectionCoefficient
        Narrowband (no frequency: a fluid–fluid coefficient does not depend
        on it): ``angles_deg``, ``magnitude`` ``|R|``, ``phase`` (rad),
        ``critical_angle_deg`` (NaN for a seabed slower than the water or a
        rigid / pressure-release boundary). ``rc.dB`` is the bottom loss
        ``-20·log10|R|``.
    """
    who = 'bottom_loss_curve'
    if grazing_angles_deg is None:
        grazing_angles_deg = np.linspace(0.0, 90.0, 181)
    grazing_angles_deg = np.asarray(grazing_angles_deg, dtype=float)
    if isinstance(material, Bottom):
        # A Bottom has one half-space only when unlayered and
        # range-independent; dropping the layers or choosing a range would
        # draw a reflection the seabed does not have.
        if material.is_layered:
            raise ConfigurationError(
                f"{who}: a layered Bottom has no single plane-wave "
                f"half-space reflection; dropping the layers would give a "
                f"reflection the seabed does not have.",
                remediation="Compute the layered seabed's reflection with "
                            "Bounce().run(env, source, receiver) or "
                            "OASR(...).run(..., run_mode=RunMode.REFLECTION).")
        if material.is_range_dependent:
            raise ConfigurationError(
                f"{who}: a range-dependent Bottom has one half-space per "
                f"range and no single reflection.",
                remediation="Pick the half-space under one range: "
                            "bottom_loss_curve(bottom.halfspace_at(range=r)).")
        material = material.halfspace_at(range=0.0)
    if isinstance(material, str):
        from uacpy.core.materials import get_material
        m = get_material(material)
    elif isinstance(material, Mapping):
        m = dict(material)
    elif all(hasattr(material, k)
             for k in ('sound_speed', 'density', 'attenuation')):
        acoustic_type = str(getattr(material, 'acoustic_type',
                                    'half-space')).lower()
        if acoustic_type in ('file', 'precalc'):
            raise ConfigurationError(
                f"{who}: a {acoustic_type!r} seabed carries its "
                f"loss in a reflection-coefficient table; its sound_speed, "
                f"density and attenuation are placeholders, not a half-space.",
                remediation="Read the table itself (uacpy.io."
                            "read_reflection_coefficient), or pass a "
                            "half-space carrier or a preset name.")
        if acoustic_type in ('rigid', 'vacuum'):
            # A rigid (R = +1) and a pressure-release (R = -1) boundary
            # reflect totally at every angle.
            sign = 1.0 if acoustic_type == 'rigid' else -1.0
            return ReflectionCoefficient.from_complex(
                grazing_angles_deg,
                np.full(grazing_angles_deg.shape, sign, dtype=complex),
                critical_angle_deg=float('nan'))
        m = {k: getattr(material, k)
             for k in ('sound_speed', 'density', 'attenuation')}
        m['shear_speed'] = getattr(material, 'shear_speed', 0.0)
    else:
        raise ConfigurationError(
            f"{who}: material must be a preset name, a dict or a "
            f"seabed carrier (BoundaryProperties / SedimentLayer) with "
            f"sound_speed, density and attenuation; got "
            f"{type(material).__name__}.")
    missing = {'sound_speed', 'density', 'attenuation'} - set(m)
    if missing:
        raise ConfigurationError(
            f"{who}: the seabed properties are missing {sorted(missing)}; "
            f"got {sorted(m)}. Plane-wave bottom loss needs sound_speed, "
            f"density and attenuation together — none has a default.",
            remediation="Name a preset instead for its tabulated values.")
    shear_speed = (0.0 if isinstance(material, str)
                   else float(m.get('shear_speed') or 0.0))
    if shear_speed > 0.0:
        warnings.warn(
            f"{who}: the seabed's shear speed ({shear_speed:g} "
            f"m/s) is ignored. This is the fluid-fluid Rayleigh curve; an "
            f"elastic seabed also loses energy to shear waves (Jensen et al., "
            f"Computational Ocean Acoustics, §1.6.2, Eq. 1.61), most between "
            f"the shear and compressional critical angles, where this curve "
            f"can report total reflection. The elastic coefficient is "
            f"acoustics.reflection_coeff(..., shear_speed=, "
            f"shear_attenuation=), as BOUNCE and OASR compute it.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
    R = reflection_coeff(
        grazing_angles_deg,
        sound_speed=float(m['sound_speed']),
        density=float(m['density']),
        attenuation=float(m['attenuation']),
        water_sound_speed=float(water_sound_speed),
        water_density=float(water_density),
    )
    return ReflectionCoefficient.from_complex(
        grazing_angles_deg, R,
        critical_angle_deg=critical_angle(float(m['sound_speed']),
                                          float(water_sound_speed)))
