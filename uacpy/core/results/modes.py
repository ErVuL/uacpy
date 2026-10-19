"""Normal-mode result type."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from typing import Optional, Tuple, Union

from uacpy.core.exceptions import ConfigurationError, ValidityWarning
from uacpy.core._export import ExportRecord, read_only
from uacpy.core._plotting import plotter
from uacpy.core._records import FrozenRecord
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.acoustics.modal import (
    _mode_shapes_at, _modal_attenuation, medium_density_at, modal_excitation,
    modal_field, _modal_grazing_angles, modal_phase_speeds,
)
from uacpy.core.boundary import BoundaryProperties

from uacpy.core.results._base import (PhaseReference, Result, _count,
                                      coordinate_axis)
from uacpy.core.results.field import Field
from uacpy.core.results.quantities import coordinate_unit
from uacpy.core._repr import count
from uacpy.core.constants import DEFAULT_WATER_DENSITY


@dataclass(frozen=True, eq=False)
class MediaTable(FrozenRecord):
    """The densities a mode set was normalised in: the water column's and,
    from a ``.mod``, those of every medium below it (``kraken.f90:595-596``
    writes the density at the top of every medium the file tabulates, the
    water column first).

    Attributes
    ----------
    water_density : float
        g/cm³ of the water column.
    tops : tuple of float, optional
        Top depth (m) of every tabulated medium, the water column first.
    densities : tuple of float, optional
        g/cm³ of every tabulated medium, in the order of ``tops``. The
        first entry is the file's water density; :meth:`density_at` answers
        :attr:`water_density` there.
    bottom_depth : float, optional
        Depth (m) of the base of the last tabulated medium, where the bottom
        half-space starts.
    halfspace_density : float, optional
        g/cm³ of the bottom half-space.
    halfspace_sound_speed : float, optional
        Compressional speed (m/s) of a fluid bottom half-space, the speed a
        mode's phase speed must stay under to be trapped in the duct
        rather than radiate into the seabed; ``None`` when the seabed has no
        single such speed (an elastic, vacuum, rigid or tabulated one) or
        the table was built without it (a ``.moa``, a hand-built mode set).
    halfspace_attenuation : float, optional
        The seabed's own compressional attenuation (dB/wavelength) in that
        fluid half-space, the deck's volume law taken out of what the solver
        applied (:func:`~uacpy.io.read_modes`); ``None`` when the speed is,
        and when no deck beside the ``.mod`` says which volume law that
        was.
    """

    water_density: float
    tops: Optional[Tuple[float, ...]] = None
    densities: Optional[Tuple[float, ...]] = None
    bottom_depth: Optional[float] = None
    halfspace_density: Optional[float] = None
    halfspace_sound_speed: Optional[float] = None
    halfspace_attenuation: Optional[float] = None

    _REPR_UNITS = {'water_density': 'g/cm³', 'tops': 'm', 'densities': 'g/cm³',
                   'bottom_depth': 'm', 'halfspace_density': 'g/cm³',
                   'halfspace_sound_speed': 'm/s',
                   'halfspace_attenuation': 'dB/λ'}

    def __post_init__(self):
        object.__setattr__(self, 'water_density', float(self.water_density))
        for name in ('tops', 'densities'):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name,
                                   tuple(float(v) for v in value))
        if (self.tops is None) != (self.densities is None) or (
                self.tops is not None
                and len(self.tops) != len(self.densities)):
            raise ConfigurationError(
                f"MediaTable: tops {self.tops} and densities "
                f"{self.densities} must both be given, one density per "
                f"medium top.")
        for name in ('bottom_depth', 'halfspace_density',
                     'halfspace_sound_speed', 'halfspace_attenuation'):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))
        speed = self.halfspace_sound_speed
        if speed is not None and not (np.isfinite(speed) and speed > 0.0):
            raise ConfigurationError(
                f"MediaTable: halfspace_sound_speed must be a positive speed "
                f"(m/s); got {speed!r}.",
                remediation="Pass the fluid half-space's compressional speed, "
                            "or None when the seabed has none.")
        loss = self.halfspace_attenuation
        if loss is not None and not (np.isfinite(loss) and loss >= 0.0):
            raise ConfigurationError(
                f"MediaTable: halfspace_attenuation must be a non-negative "
                f"attenuation (dB/wavelength); got {loss!r}.",
                remediation="Pass the fluid half-space's compressional "
                            "attenuation, or None when it is not known.")

    @classmethod
    def from_dict(cls, d) -> 'MediaTable':
        """The table :meth:`to_dict` wrote.

        Parameters
        ----------
        d : mapping
            The table's fields by name.
        """
        return cls(**d)

    def density_at(self, depth: float) -> float:
        """``ρ(z)`` (g/cm³) of the medium holding ``depth`` (m)
        (:func:`~uacpy.acoustics.medium_density_at`): :attr:`water_density` down to the
        seafloor, each deeper medium's own density below its top, and
        :attr:`halfspace_density` below the last medium's base. A table with
        no tabulated media (a ``.moa``, or one built by hand) answers the
        water density for every depth.

        Parameters
        ----------
        depth : float
            Depth (m) below the surface.
        """
        if not self.tops or len(self.tops) < 2:
            return self.water_density
        return medium_density_at(
            depth, self.tops, (self.water_density,) + self.densities[1:],
            np.inf if self.bottom_depth is None else self.bottom_depth,
            (self.densities[-1] if self.halfspace_density is None
             else self.halfspace_density))


#: The half-spaces a mode set's :attr:`Modes.halfspace_tails` may describe,
#: and the terms each carries.
_HALFSPACE_SIDES = ('top', 'bottom')
_HALFSPACE_TERMS = ('k2', 'gamma', 'phi')


def _checked_halfspace_tails(tails, k_shape):
    """``tails`` as :attr:`Modes.halfspace_tails` holds it (fresh arrays),
    refused when a side or a term is unknown or missing, or when a
    per-mode term is not one value per mode."""
    if tails is None:
        return None
    unknown = sorted(set(tails) - set(_HALFSPACE_SIDES))
    if unknown:
        raise ConfigurationError(
            f"Modes.halfspace_tails: {unknown} are not half-spaces; the keys "
            f"are {list(_HALFSPACE_SIDES)}.")
    checked = {}
    for side, tail in tails.items():
        if set(tail) != set(_HALFSPACE_TERMS):
            raise ConfigurationError(
                f"Modes.halfspace_tails[{side!r}] holds {sorted(tail)}; it "
                f"must hold exactly {list(_HALFSPACE_TERMS)}.")
        gamma = np.array(tail['gamma'], dtype=complex)
        phi = np.array(tail['phi'])
        if gamma.shape != k_shape or phi.shape != k_shape:
            raise ConfigurationError(
                f"Modes.halfspace_tails[{side!r}]: 'gamma' {gamma.shape} and "
                f"'phi' {phi.shape} must each be one value per mode, "
                f"{k_shape}.")
        checked[side] = {'k2': complex(tail['k2']), 'gamma': gamma,
                         'phi': phi}
    return checked


def _copied_tails(tails):
    """A copy of :attr:`Modes.halfspace_tails` for :meth:`Modes.to_dict`."""
    if tails is None:
        return None
    return {side: {'k2': tail['k2'], 'gamma': tail['gamma'].copy(),
                   'phi': tail['phi'].copy()}
            for side, tail in tails.items()}


class Modes(Result):
    """Kraken normal modes — depth eigenfunctions of the Helmholtz operator.

    Attributes
    ----------
    k : ndarray, shape ``(n_modes,)`` complex
        Modal horizontal wavenumbers (rad/m). A nonzero imaginary part
        means an attenuating mode — leaky modes from ``backend='krakenc'``,
        or the perturbation :meth:`with_attenuation` applies. Decay is
        ``Im k < 0`` from every producer (the ``e^{+iωt}``, ``e^{-ikr}``
        convention Kraken writes); the modal sums decay each mode by
        ``|Im k|``.
    phi : ndarray, shape ``(n_depths, n_modes)``
        Mode shapes sampled at ``depths``.
    depths : ndarray, shape ``(n_depths,)``
        Depths (m below the surface) the mode shapes are tabulated at —
        KRAKEN's ``zTab``, the merged, sorted, duplicate-free union of the
        run's source and receiver depths (``Kraken/kraken.f90:573`` builds
        it, ``:598`` writes it to the ``.mod``). The modes are known
        nowhere else; past the span, a mode set read from a ``.mod``
        carries the evanescent tail into each acoustic half-space in
        :attr:`halfspace_tails`.
    media : MediaTable or None
        The densities the modes were normalised in, from which
        :meth:`modal_pressure_field` takes ``ρ(z_s)`` and
        :meth:`with_attenuation` its water density; ``None`` when no
        density is recorded (a ``.moa``, or a mode set built by hand).
        A ``metadata`` carrying ``water_density``, ``media_depths``,
        ``media_densities``, ``media_bottom_depth`` or ``halfspace_density``
        is refused.
    group_speeds : ndarray, shape ``(n_modes,)``, or None
        Group speed (m/s) each mode was reported at by the solver, from a
        SINGLE run — ``None`` when the binary did not supply one.

        Only ``backend='krakenc'`` supplies it. KRAKENC computes it in its
        perturbation pass (``krakenc.f90:819``, ``VG = 1/Slow``) and prints
        it. KRAKEN allocates and prints the same column but never fills it:
        the assignment is commented out at ``kraken.f90:815-819``, so the
        column reads 0.00000 for every mode, and this attribute is ``None``
        there rather than an array of zeros.

        Entries are ``NaN`` for any mode the print file skipped.
        ``kraken.f90:101`` prints ``MAX(1, M/30)``-stride, so a run with
        more than 30 modes reports only about 30 of them.

        Use :meth:`group_speeds_between` instead when this is ``None``, or
        when every mode is needed from a large set. The perturbation caveat
        the source states at ``kraken.f90:772`` applies either way: "group
        speeds will be wrong for leaky modes".
    halfspace_tails : dict or None
        The analytic continuation of each mode into an acoustic (``'A'``)
        half-space, as OALIB ``read_modes.m`` computes it for ``evalcm.m``:
        keys ``'top'`` / ``'bottom'`` for the half-spaces that are acoustic,
        each a dict of ``'k2'`` (complex, rad²/m², ``(2πf/c_p)²`` with the
        file's complex ``c_p``), ``'gamma'`` (complex, ``(n_modes,)``,
        1/m, ``sqrt(k_m² − k2)`` with ``Re ≥ 0``, from ``Re(k)`` for KRAKEN
        and the full ``k`` for KRAKENC, ``KrakenField/ReadModes.f90:79``,
        ``:254-272``) and ``'phi'`` (``(n_modes,)``, the mode at the
        interface: the first or last tabulated row). Below the deepest
        tabulated depth ``D`` a mode continues as
        ``phi * exp(-gamma * (z - D))``, and above the shallowest the same
        way. Set by :func:`~uacpy.io.read_modes` from a ``.mod``, and so on
        the mode sets :class:`~uacpy.Kraken` returns (an empty dict when no
        half-space is acoustic); ``None`` otherwise, and on the set
        :meth:`with_attenuation` returns, whose ``k`` it no longer matches.
    """

    #: The plotter that draws this result (``.plot()``).
    _plotter = '_plot_mode_shapes'

    #: The metadata spellings of the :class:`MediaTable` members: a
    #: ``metadata=`` carrying one is refused.
    _METADATA_MEDIA = {'water_density': 'water_density',
                       'media_depths': 'tops',
                       'media_densities': 'densities',
                       'media_bottom_depth': 'bottom_depth',
                       'halfspace_density': 'halfspace_density'}

    def __init__(
        self,
        *,
        k: np.ndarray,
        phi: np.ndarray,
        depths: np.ndarray,
        group_speeds: Optional[np.ndarray] = None,
        media: Optional[MediaTable] = None,
        halfspace_tails: Optional[dict] = None,
        **kwargs,
    ):
        carried = sorted(set(kwargs.get('metadata') or {})
                         & set(self._METADATA_MEDIA))
        if carried:
            raise ConfigurationError(
                f"Modes: metadata carries {carried}, which are members of "
                f"the mode set's media table, not metadata.",
                remediation="Pass media=MediaTable("
                            + ", ".join(f"{self._METADATA_MEDIA[t]}=..."
                                        for t in carried) + ").")
        if media is not None and not isinstance(media, MediaTable):
            raise ConfigurationError(
                f"Modes: media={media!r} is not a MediaTable.",
                remediation="Pass media=MediaTable(water_density=..., ...).")
        super().__init__(**kwargs)
        self.media = media
        # Copy on ingest so a caller mutating their source array can't silently
        # corrupt this result (mode arrays are small; the copy is cheap).
        self.k = np.array(k)
        self.phi = np.array(phi)
        self.depths = np.atleast_1d(np.array(depths, dtype=float))
        if self.phi.shape != (len(self.depths), len(self.k)):
            raise ConfigurationError(
                f"Modes.phi: shape {self.phi.shape} must equal "
                f"(len(depths), len(k)) = ({len(self.depths)}, {len(self.k)})"
            )
        if group_speeds is None:
            self.group_speeds = None
        else:
            gv = np.atleast_1d(np.array(group_speeds, dtype=float))
            if gv.shape != self.k.shape:
                raise ConfigurationError(
                    f"Modes.group_speeds: shape {gv.shape} must equal "
                    f"k's {self.k.shape} — one group speed per mode, NaN "
                    f"where the binary did not report one."
                )
            self.group_speeds = gv
        self.halfspace_tails = _checked_halfspace_tails(halfspace_tails,
                                                        self.k.shape)

    @property
    def n_modes(self) -> int:
        """Number of modes — always ``len(k)`` (single source of truth).

        Derived rather than stored so it can never desync from ``k``/``phi``;
        to use fewer modes, slice via :meth:`first_n` (which trims ``k`` and
        ``phi`` together).
        """
        return len(self.k)

    @property
    def trapped(self) -> Optional[np.ndarray]:
        """Which modes are trapped in the duct: a boolean mask, one per
        mode, True where the mode's phase speed is below the fluid seabed's
        compressional speed :attr:`media` ``.halfspace_sound_speed``
        (``Re k_r > ω/c_b``, the discrete spectrum, evanescent in the
        bottom) and False at or above it, a mode in the continuous spectrum
        that radiates into the seabed (Jensen et al., *Computational Ocean
        Acoustics*, sect. 2.4.5.1, Fig. 2.26). The default phase-speed
        window reaches 5 % past that speed (kraken.md §6.1), so a Kraken set
        can hold such modes. ``None`` when the media table records no such
        speed (an elastic, vacuum, rigid or tabulated seabed, a ``.moa``, a
        hand-built set) or the set has no frequency."""
        c_bottom = (None if self.media is None
                    else self.media.halfspace_sound_speed)
        if c_bottom is None or self.f0 is None:
            return None
        return read_only(self.phase_speeds < c_bottom)

    def _repr_bits(self) -> list:
        # The count splits into trapped and not when some mode is not
        # trapped (:attr:`trapped`), '8 modes (7 trapped, 1 above the
        # 1800 m/s seabed)'.
        modes = count(self.n_modes, 'mode')
        trapped = self.trapped
        if trapped is not None and not np.all(trapped):
            above = int(np.sum(~trapped))
            modes += (f" ({self.n_modes - above} trapped, {above} above "
                      f"the {self.media.halfspace_sound_speed:g} m/s "
                      f"seabed)")
        return [modes, coordinate_axis('depth', self.depths)]

    def to_dict(self) -> dict:
        """Serialise these modes to plain arrays: ``k``, ``phi`` (depth ×
        mode), ``depths``, ``group_speeds`` (``None`` when the solver
        reported none), ``media`` (the :class:`MediaTable` as its plain
        dict, or ``None``), ``halfspace_tails`` (``None`` or its dict of
        plain arrays) and the identity, as :meth:`Field.to_dict` writes it.
        ``np.savez(f, **d)`` stores it; read it back with
        ``np.load(f, allow_pickle=True)`` into :meth:`from_dict`. That load
        unpickles, so use it only on files you wrote; :meth:`to_netcdf` /
        :meth:`from_netcdf` is the route for a file from elsewhere."""
        return {
            'k': self.k.copy(),
            'phi': self.phi.copy(),
            'depths': self.depths.copy(),
            'group_speeds': (None if self.group_speeds is None
                             else self.group_speeds.copy()),
            'media': None if self.media is None else self.media.to_dict(),
            'halfspace_tails': _copied_tails(self.halfspace_tails),
            **self._identity_dict(),
        }

    #: ``attrs`` the xarray export writes for :attr:`media`, by member.
    _MEDIA_ATTRS = {member: f'media_{member}'
                    for member in ('water_density', 'tops', 'densities',
                                   'bottom_depth', 'halfspace_density',
                                   'halfspace_sound_speed',
                                   'halfspace_attenuation')}

    def _payload(self):
        payload = {'k': (self.k, ('mode',), 'rad/m'),
                   'phi': (self.phi, ('depth', 'mode'), '')}
        if self.group_speeds is not None:
            payload['group_speeds'] = (self.group_speeds, ('mode',),
                                       'm/s')
        for side, tail in (self.halfspace_tails or {}).items():
            payload[f'{side}_halfspace_gamma'] = (tail['gamma'], ('mode',),
                                                  '1/m')
            payload[f'{side}_halfspace_phi'] = (tail['phi'], ('mode',), '')
        return payload

    def _coords(self):
        return {'depth': (self.depths, coordinate_unit('depth')),
                'mode': (np.arange(1, self.n_modes + 1), '')}

    def _export_attrs(self):
        attrs = super()._export_attrs()
        if self.media is not None:
            for member, key in self._MEDIA_ATTRS.items():
                value = getattr(self.media, member)
                if value is not None:
                    attrs[key] = (np.asarray(value, dtype=float)
                                  if isinstance(value, tuple) else value)
        if self.halfspace_tails is not None:
            # NetCDF attributes hold no complex number, so k2 goes as its
            # two parts; the sides present are listed so an empty dict (no
            # acoustic half-space) reads back as one.
            attrs['halfspace_tails'] = ' '.join(sorted(self.halfspace_tails))
            for side, tail in self.halfspace_tails.items():
                k2 = complex(tail['k2'])
                attrs[f'{side}_halfspace_k2_real'] = k2.real
                attrs[f'{side}_halfspace_k2_imag'] = k2.imag
        return attrs

    @classmethod
    def _from_export(cls, arrays, attrs):
        stated = {member: attrs[key]
                  for member, key in cls._MEDIA_ATTRS.items() if key in attrs}
        media = (MediaTable(**{member: (tuple(np.atleast_1d(value))
                                        if member in ('tops', 'densities')
                                        else value)
                               for member, value in stated.items()})
                 if stated else None)
        tails = None
        tail_attrs = ()
        if 'halfspace_tails' in attrs:
            sides = str(attrs['halfspace_tails']).split()
            tail_attrs = ('halfspace_tails',) + tuple(
                f'{side}_halfspace_k2_{part}'
                for side in sides for part in ('real', 'imag'))
            tails = {side: {
                'k2': complex(attrs[f'{side}_halfspace_k2_real'],
                              attrs[f'{side}_halfspace_k2_imag']),
                'gamma': arrays[f'{side}_halfspace_gamma'],
                'phi': arrays[f'{side}_halfspace_phi']} for side in sides}
        return cls(k=arrays['k'], phi=arrays['phi'], depths=arrays['depth'],
                   group_speeds=arrays.get('group_speeds'), media=media,
                   halfspace_tails=tails,
                   **cls._identity_from_attrs(
                       attrs, tuple(cls._MEDIA_ATTRS.values()) + tail_attrs))

    def _table(self):
        """One row per mode: ``mode`` (1-based), ``k_real`` and ``k_imag``
        (rad/m), ``phase_speed`` (m/s; NaN without a frequency),
        ``group_speed`` (m/s; NaN where the solver reported none, as the
        default ``kraken`` backend never does — :meth:`group_speeds_between`
        gives it from a second solve) and
        ``attenuation_dB_per_km``, the modal decay ``20·log10(e)·|Im(k)|``
        per km: the decay rate :func:`~uacpy.core.acoustics.modal_field`
        applies, which takes ``|Im(k)|`` whichever sign convention the
        solver wrote."""
        k = np.asarray(self.k)
        phase_speed = (np.full(self.n_modes, np.nan) if self.f0 is None
                       else self.phase_speeds)
        group_speed = (np.full(self.n_modes, np.nan)
                       if self.group_speeds is None
                       else np.asarray(self.group_speeds, dtype=float))
        return {
            'mode': np.arange(1, self.n_modes + 1),
            'k_real': np.real(k).astype(float),
            'k_imag': np.imag(k).astype(float),
            'phase_speed': np.asarray(phase_speed, dtype=float),
            'group_speed': group_speed,
            'attenuation_dB_per_km': 20.0 * np.log10(np.e)
            * np.abs(np.imag(k)).astype(float) * 1000.0,
        }

    def _table_units(self):
        return {'mode': '', 'k_real': 'rad/m', 'k_imag': 'rad/m',
                'phase_speed': 'm/s', 'group_speed': 'm/s',
                'attenuation_dB_per_km': 'dB/km'}

    @classmethod
    def from_dict(cls, d: dict) -> "Modes":
        """Reconstruct :class:`Modes` from :meth:`to_dict` output, or from
        the mapping ``np.load(f, allow_pickle=True)`` returns for a file
        written with ``np.savez(f, **modes.to_dict())`` (load such a file
        only if you wrote it: ``allow_pickle=True`` unpickles).

        Parameters
        ----------
        d : mapping
            :meth:`to_dict` output, or the mapping ``np.load`` returns for it.
        """
        d = cls._unwrap_saved(d, payload=('k', 'phi', 'depths'),
                              required=('group_speeds', 'media',
                                        'halfspace_tails'))
        identity = cls._identity_from_dict(d)
        media = d.get('media')
        if media is not None:
            media = MediaTable.from_dict(media)
        return cls(k=d['k'], phi=d['phi'], depths=d['depths'],
                   group_speeds=d.get('group_speeds'), media=media,
                   halfspace_tails=d.get('halfspace_tails'),
                   **identity)

    def first_n(self, n: int) -> "Modes":
        """Return a new :class:`Modes` containing only the first ``n`` modes.

        No-op when ``n >= self.n_modes``. ``k`` is sliced as ``k[:n]`` and
        ``phi`` as ``phi[:, :n]``; depths and identification metadata are
        preserved.

        Parameters
        ----------
        n : int
            Modes to keep.
        """
        n = _count(n, 'Modes.first_n')
        if n >= len(self.k):
            return self
        new_k = self.k[:n]
        new_phi = self.phi[:, :n]
        return Modes(
            k=new_k,
            phi=new_phi,
            depths=self.depths,
            group_speeds=(None if self.group_speeds is None
                          else self.group_speeds[:n]),
            media=self.media,
            halfspace_tails=(None if self.halfspace_tails is None else {
                side: dict(tail, gamma=tail['gamma'][:n],
                           phi=tail['phi'][:n])
                for side, tail in self.halfspace_tails.items()}),
            **self.id_kwargs(),
        )

    @property
    def phase_speeds(self) -> np.ndarray:
        """Mode phase speeds ``v_p = ω / Re(k_r)`` in m/s.

        Raises
        ------
        ConfigurationError
            If this :class:`Modes` instance has no frequency context
            (``self.f0 is None``); without a frequency the phase speed
            is undefined. Pass ``frequencies=…`` to the wrapper that
            built this object, or set it on the instance, before reading it.
        """
        if self.f0 is None:
            raise ConfigurationError(
                "Modes.phase_speeds requires frequencies; got None."
            )
        return modal_phase_speeds(self.k, self.f0)

    def group_speeds_between(self, other: "Modes") -> np.ndarray:
        """Approximate group speed ``v_g = dω/dk`` using a second
        :class:`Modes` instance at a nearby frequency.

        Parameters
        ----------
        other : Modes
            Modes computed at a slightly different frequency.

        Returns
        -------
        v_g : ndarray, shape ``(min(self.n_modes, other.n_modes),)``
            Mode-by-mode group speed in m/s. Modes that exist in only
            one of the two results are dropped (the array is truncated to
            the shared count). The estimate is second-order accurate at the
            **midpoint** frequency ``(f0_self + f0_other)/2``, not at either
            endpoint, and it inherits the dtype of ``k`` — float32 for the
            complex64 wavenumbers a ``.mod`` file carries.

        Notes
        -----
        **There is an accuracy-optimal Δf, and it is not the smallest one.**
        The truncation error falls as Δf², but ``k`` arrives quantized —
        KRAKEN's ``.mod`` record is ``COMPLEX*8`` — so the difference carries
        about one float32 step of noise however small Δf is, and the storage
        contributes ``spacing(k_r)/|Δk_r|`` relative. The two balance at

        .. math:: \\Delta f \\approx \\left(
            \\frac{\\mathrm{spacing}(k_r)\\, v_g}
                 {2\\pi\\,|\\mathrm{d}^2 k_r/\\mathrm{d}\\omega^2|}
            \\right)^{1/3}

        The reasoning, the literature and the measurements behind this method
        are in ``docs/theory/broadband_products.md``, section "The
        finite-difference group velocity"."""
        f0_self, f0_other = self.f0, other.f0
        if f0_self is None or f0_other is None:
            raise ConfigurationError(
                "Modes.group_speeds_between: both Modes instances must "
                "have a frequency."
            )
        if f0_self == f0_other:
            raise ConfigurationError(
                "Modes.group_speeds_between: requires Modes at two distinct "
                "frequencies."
            )
        n = min(self.n_modes, other.n_modes)
        if n == 0:
            return np.array([])
        # The difference quotient, its storage-resolution notice and its
        # monotonicity refusal are modal_group_speeds's; what this method
        # adds is pairing two Modes and truncating to the shared set.
        #
        # That function REFUSES a k_r that does not rise strictly with
        # frequency, where the quotient would give nan for a flat step and a
        # NEGATIVE speed for a falling one. k_r rises because v_g is an
        # energy-transport speed, so a falling step is bad input and not a
        # cell with no answer; a negative group velocity handed back without
        # comment is the worse of the two.
        lo, hi = ((self, other) if f0_self < f0_other else (other, self))
        # Deferred: acoustic_signal pulls scipy, and uacpy's public
        # surface is imported without it (test_lazy_imports).
        from uacpy.acoustic_signal.dispersion import modal_group_speeds
        return modal_group_speeds(
            np.array([min(f0_self, f0_other), max(f0_self, f0_other)]),
            k_horizontal=np.stack([np.asarray(lo.k)[:n], np.asarray(hi.k)[:n]]))[0]

    def shapes_at(self, depths, *, outside: str = 'raise') -> np.ndarray:
        """Mode shapes interpolated onto ``depths``, ``(n_depths, n_modes)``.

        A depth outside this result's tabulation is refused
        (``outside='raise'``) or returned as NaN (``outside='nan'``), never
        extrapolated. On plain arrays this is
        :func:`~uacpy.core.acoustics.mode_shapes_at`.

        Parameters
        ----------
        depths : float or array_like
            Depths (m).
        outside : {'raise', 'nan'}, optional
            What a depth outside the tabulation gets. Default ``'raise'``.
        """
        return _mode_shapes_at(self.phi, self.depths, depths,
                               outside=outside, who='Modes.shapes_at')

    def excitation(self, source, *, sound_speed=None) -> np.ndarray:
        """Complex modal excitation of a ``Source``: ``Σₙ wₙ·φₘ(zₙ)``.

        What a source array actually does in a waveguide. Each element
        drives every mode in proportion to the mode's shape at that
        element's depth, and the array's weights set the sum — so choosing
        the weights chooses the modal content. That is the *mode filter* of
        Medwin & Clay §11.3.1 ("Arrays of sources and receivers in a
        waveguide: mode filters"), and the reason JKPS poses the vertical
        array as a "modal, rather than plane-wave beamformer": drive the
        elements with a mode's own shape and that mode dominates.

        This is exact in the channel, unlike the free-field
        :meth:`~uacpy.Source.array_factor`, which describes the same array
        as an angular pattern. The two agree only while that pattern is
        symmetric in ±θ — a trapped mode is a standing wave, equal up- and
        down-going halves, so a steered array is not a scaling of any single
        source. Mode shapes are interpolated onto the source depths from
        this result's tabulation.

        A directional element shades this too: each mode's excitation is
        multiplied by the element pattern at its grazing angle, ``S(θₘ)``,
        and every element of a uacpy ``Source`` shares one ``beam_pattern``
        and one mode angle, so ``S`` factors straight out of the array sum —
        the product theorem again, in the modal domain. This method reads
        the pattern at ``θₘ = arccos(c / vₚ,ₘ)`` against the
        ``sound_speed`` given: the physical angle at the source when that is
        the speed there. Kraken's ``field.exe`` applies the same shading
        (``C = C * REAL(S)``, ``KrakenField/field.f90:211``) but against a
        hard-coded ``c0 = 1500`` m/s (``:203``), and shades every mode
        slower than that at ``S(0°)`` (``:206``; see
        :class:`~uacpy.models.Kraken`, Note): ``sound_speed=1500.0``
        reproduces the engine's angles for the modes faster than 1500 m/s,
        while this method returns NaN for the slower ones. A source that
        carries a pattern and is given no speed raises rather than silently
        returning the omnidirectional answer.

        Parameters
        ----------
        source : Source
            Its ``depths`` and complex ``weights``. A single-depth source
            gives that depth's mode shapes scaled by its one weight.
        sound_speed : float, optional
            Reference speed (m/s) for the mode grazing angles
            ``θₘ = arccos(c / vₚ,ₘ)`` the element pattern is sampled at: the
            speed at the source depth for the physical angle, 1500.0 for
            the angles Kraken's ``field.exe`` uses (see above). Required
            when ``source.beam_pattern`` is set, unused otherwise.

        Returns
        -------
        ndarray, shape ``(n_modes,)``, complex
            One amplitude per mode, in the mode shapes' own normalisation.
        """
        at_source = _mode_shapes_at(self.phi, self.depths, source.depths,
                                    outside='raise', who='Modes.excitation')
        if getattr(source, 'beam_pattern', None) is None:
            return modal_excitation(at_source, source.weights)
        if sound_speed is None:
            raise ConfigurationError(
                "Modes.excitation: this Source carries a beam_pattern, "
                "which shades the modal excitation at each mode's grazing "
                "angle — so the answer depends on the speed those angles "
                "are measured against. Pass sound_speed= (the speed at the "
                "source depth for the physical angle; 1500.0 for the "
                "angles Kraken's field.exe shades at), or drop the pattern."
            )
        angles = self.grazing_angles(sound_speed)
        undefined = ~np.isfinite(angles)
        if np.any(undefined):
            warnings.warn(
                f"Modes.excitation: {int(np.sum(undefined))} of "
                f"{angles.size} modes have a phase speed below the "
                f"{float(sound_speed):g} m/s reference, so they are "
                f"evanescent there and have no real grazing angle. Their "
                f"excitation is returned as NaN: this method weights a mode "
                f"by the beam pattern at its real grazing angle only. "
                f"(Kraken's field.exe shades every mode slower than its "
                f"1500 m/s reference at S(0°) instead.)",
                ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)
        weighted = modal_excitation(
            at_source, source.weights,
            directivity=source.element_directivity(
                np.where(undefined, 0.0, angles)))
        return np.where(undefined, np.nan, weighted)

    def grazing_angles(self, sound_speed: float) -> np.ndarray:
        """Each mode's grazing angle in degrees against ``sound_speed``.

        ``theta_m = arccos(c / v_m)``, the dictionary between the modal and
        ray pictures, and the one home for that formula — every caller asks
        here rather than spelling it again. A mode whose phase speed is below
        ``c`` is evanescent there and has no real angle, so it comes back
        ``nan``; clipping the ratio to 1.0 instead would stack those modes on
        broadside, where a patterned source applies its notch to them. On
        plain arrays this is
        :func:`~uacpy.core.acoustics.modal_grazing_angles`.

        Parameters
        ----------
        sound_speed : float
            The speed (m/s) the angles are measured against.
        """
        if self.f0 is None:
            raise ConfigurationError(
                "Modes.grazing_angles: needs the mode set's frequency; this "
                "Modes has none."
            )
        return _modal_grazing_angles(self.k, self.f0, sound_speed,
                                    who='Modes.grazing_angles')

    @classmethod
    def dispersion(cls, modes_by_frequency) -> "ModeDispersion":
        """Phase and group speed of every mode against frequency — the
        dispersion diagram's numbers — from mode sets at several
        frequencies.

        Both speeds start at the seabed speed at a mode's cutoff and tend to
        the water speed at high frequency, the phase speed falling
        monotonically while the group speed passes through a minimum — the
        frequency of that minimum arrives last in a transient, the Airy
        phase (JKPS Sect. 2.4.5.2).

        The phase speed is each set's :attr:`phase_speeds`. The group speed
        is a set's own :attr:`group_speeds` where the solver supplied it
        (``backend='krakenc'``), placed at that set's frequency; elsewhere it
        is :meth:`group_speeds_between` the set and the next, a finite
        difference that belongs at the **midpoint** of the pair (that
        method's own docstring), placed there. Both carry the caveat the
        KRAKEN source states at ``kraken.f90:772``: group speeds are wrong
        for leaky modes.

        Parameters
        ----------
        modes_by_frequency : sequence of Modes
            At least two mode sets at distinct frequencies (``f0``), in any
            order.

        Returns
        -------
        ModeDispersion
            ``.plot()`` draws the diagram.
        """
        who = 'Modes.dispersion'
        sets = list(modes_by_frequency)
        if len(sets) < 2:
            raise ConfigurationError(
                f"{who}: needs Modes at two or more frequencies — got "
                f"{len(sets)}. Dispersion is a statement about how the modal "
                f"speeds vary with frequency, so one solve cannot show it.")
        for m in sets:
            if not isinstance(m, Modes):
                raise ConfigurationError(
                    f"{who}: expected a sequence of Modes, found "
                    f"{type(m).__name__}.")
            if m.f0 is None:
                raise ConfigurationError(
                    f"{who}: every Modes needs a frequency (f0); one carries "
                    f"none, so it cannot be placed on the frequency axis.")
        sets.sort(key=lambda m: float(m.f0))
        freqs = np.array([float(m.f0) for m in sets])
        if np.any(np.diff(freqs) <= 0):
            raise ConfigurationError(
                f"{who}: frequencies must be distinct, got {freqs}.")
        n_modes = max(m.n_modes for m in sets)
        phase = np.full((freqs.size, n_modes), np.nan)
        points = {}
        for j, m in enumerate(sets):
            phase[j, :m.n_modes] = m.phase_speeds
            reported = getattr(m, 'group_speeds', None)
            between = (m.group_speeds_between(sets[j + 1])
                       if j + 1 < len(sets) else np.array([]))
            for i in range(m.n_modes):
                if reported is not None and np.isfinite(reported[i]):
                    points[(freqs[j], i)] = float(reported[i])
                elif i < between.size:
                    points[(0.5 * (freqs[j] + freqs[j + 1]), i)] = float(
                        between[i])
        group_freqs = np.array(sorted({f for f, _ in points}))
        group = np.full((group_freqs.size, n_modes), np.nan)
        for (f, i), v in points.items():
            group[np.searchsorted(group_freqs, f), i] = v
        first = sets[0]
        source = getattr(first.model_source, 'id', None)
        return ModeDispersion(frequencies=freqs, phase_speeds=phase,
                              group_frequencies=group_freqs,
                              group_speeds=group, model=first.model,
                              backend=first.backend,
                              model_source_id=(None if source is None
                                               else str(source)))

    def excitation_pattern(self, source, *, sound_speed=None,
                           env=None) -> "ExcitationPattern":
        """What a source array drives, both ways, on one grazing-angle axis:
        each mode's excitation at its grazing angle, and the array's
        free-field pattern folded over ±θ.

        The excitation is :meth:`excitation` (``|Σₙ wₙ·φₘ(zₙ)|``, the *mode
        filter* of Medwin & Clay §11.3.1) at :meth:`grazing_angles`
        ``θₘ = arccos(c / vₚ,ₘ)``. The pattern is the same array described
        in free field: its **array beam pattern** ``P(θ) = f(θ)·A(θ)`` when
        the elements are directional, the bare array factor ``A(θ)`` when
        they are not (Butler & Sherman §7.1.1;
        :meth:`~uacpy.Source.array_beam_pattern`), folded as
        ``max(|P(+θ)|, |P(−θ)|)``: a trapped mode is a standing wave with
        equal up- and down-going halves, so the free-field quantity
        comparable to a mode is the larger of the two. The two agree while
        the pattern is symmetric in ±θ and part company once steering
        breaks that symmetry.

        Parameters
        ----------
        source : Source
            The array: its ``depths`` and complex ``weights``.
        sound_speed : float, optional
            Reference speed (m/s) the grazing angles are measured against.
            ``None`` reads it from ``env`` at the first source depth — the
            depth at which the array factor's ``θ`` is a launch angle. With
            neither, the call is refused: the angle axis is measured
            against this speed, so defaulting it would move every mode (up
            to 5.3° on a 1545 m/s environment). A mode whose phase speed is
            below it has no real grazing angle at that depth: its angle is
            NaN, with a warning naming how many.
        env : Environment, optional
            Supplies ``sound_speed`` at the source depth when it is not
            given.

        Returns
        -------
        ExcitationPattern
            ``.plot()`` draws both, each in dB re its own maximum.
        """
        who = 'Modes.excitation_pattern'
        if sound_speed is None:
            if env is None:
                raise ConfigurationError(
                    f"{who}: the angle axis is a grazing angle, "
                    f"arccos(c/v_m), so it needs the speed those angles are "
                    f"measured against. Pass sound_speed= (the speed at the "
                    f"source depth), or env= to read it off the profile.",
                    remediation="sound_speed=env.ssp.sound_speed_at("
                                "source.depths[0])")
            z_s = float(np.atleast_1d(source.depths)[0])
            sound_speed = float(np.atleast_1d(env.ssp.sound_speed_at(z_s))[0])
        excitation = np.abs(self.excitation(source, sound_speed=sound_speed))
        angles = self.grazing_angles(sound_speed)
        dropped = int(np.sum(~np.isfinite(angles)))
        # Silent when the source carries a pattern: excitation() has already
        # warned about these same modes against this same reference speed.
        if dropped and getattr(source, 'beam_pattern', None) is None:
            warnings.warn(
                f"{who}: {dropped} of {angles.size} modes have a phase speed "
                f"below the {float(sound_speed):g} m/s reference, so they "
                f"have no real grazing angle there; their angle is NaN. Pass "
                f"sound_speed= (or env=) for the speed at the source depth.",
                ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)
        finite = angles[np.isfinite(angles)]
        span = np.linspace(0.0, max(90.0, float(finite.max())
                                    if finite.size else 90.0), 721)
        directional = getattr(source, 'beam_pattern', None) is not None
        pattern = (source.array_beam_pattern if directional
                   else source.array_factor)
        folded = np.maximum(np.abs(pattern(span, sound_speed=sound_speed)),
                            np.abs(pattern(-span, sound_speed=sound_speed)))
        depths = np.atleast_1d(source.depths)
        model_source = getattr(self.model_source, 'id', None)
        return ExcitationPattern(
            grazing_angles_deg=angles, excitation=excitation,
            pattern_angles_deg=span, pattern=folded, directional=directional,
            sound_speed=float(sound_speed), n_sources=int(depths.size),
            aperture=float(np.ptp(depths)), model=self.model,
            backend=self.backend,
            model_source_id=(None if model_source is None
                             else str(model_source)))

    def with_attenuation(
        self,
        alpha_dB_per_m: Union[float, np.ndarray],
        *,
        sound_speed_z: Optional[Union[float, np.ndarray]] = None,
        density_z: Optional[Union[float, np.ndarray]] = None,
        bottom=None,
        seafloor_depth: Optional[float] = None,
    ) -> "Modes":
        """First-order modal attenuation perturbation.

        For mode ``m`` with horizontal wavenumber ``k_rm`` and depth
        eigenfunction ``ψ_m``, the imaginary part becomes ``−α_m``, with

        ``α_m = (ω / k_rm) · ∫ α(z)/(c(z) ρ(z)) · Re(ψ_m)² dz / ∫ Re(ψ_m)² / ρ(z) dz``

        replacing any prior ``k.imag``: decay written ``Im k < 0``, the sign
        Kraken writes (the ``e^{+iωt}``, ``e^{-ikr}`` convention). The ``1/ρ``-weighted denominator
        matches the Kraken-class normalisation ``∫|ψ|²/ρ dz = 1``.

        Both integrals run ``0 → ∞`` (JKPS Eq. 5.169, and the text after
        Eq. 5.176, which extends the interval into the bottom so the
        half-space evanescent tail is counted). The tabulated ``ψ`` stops
        at the seabed, so the tail is added in closed form —
        ``ψ(D)²/(2 γ_m ρ_b)`` for a trapped mode — which needs ``γ_m`` and
        ``ρ_b``, and the seabed-loss term beside it needs the seabed's
        attenuation. They come from ``bottom=`` or, without it, from the
        fluid half-space the mode set records (``media.halfspace_sound_speed``,
        ``halfspace_density`` and ``halfspace_attenuation``, from a Kraken
        ``.mod``) when it lies directly under the water column and the
        tabulation reaches it (``media.bottom_depth``): on Kraken's modes of
        a 100 m Pekeris guide over a 0, 0.1, 0.5 or 0.8 dB/λ seabed the
        recorded half-space returns the ``bottom=env.bottom`` result to
        rounding, on kraken and krakenc, with any volume law. **With neither, the normalisation
        stops at the seabed, no seabed-loss term is added, and every
        returned ``Im(k)`` is an upper bound on the water-column term**: the
        omitted tail is positive and only ever lowers ``α_m``. Measured in
        that case on an analytic Pekeris guide (D = 100 m, c 1500/1800 m/s,
        ρ 1.0/1.8 g/cm³, lossless seabed, 50 Hz, uniform α in the water) the
        water-only normalisation returns 0.7 %, 2.4 %, 5.1 % and 17 % high
        for modes 1-4 (0.7 %, 2.4 %, 5.2 % and 18 % on Kraken's modes of
        the same guide) — the near-cutoff mode, which dominates at long
        range, is over-attenuated the most because its shallow ``γ_m``
        pushes the most energy into the bottom.

        A trapped mode gets its tail term however weakly it is bound; a
        leaky mode (``k_r <= k_b``) gets none, its tail integral diverging,
        and the leaky warning names ``k_b``.

        Parameters
        ----------
        alpha_dB_per_m : float or ndarray
            Per-depth volume attenuation in dB/m, sampled on
            :attr:`depths`. Scalar broadcasts to every depth. Build one
            from an :class:`~uacpy.core.absorption.Absorption` via
            ``absorption.table(modes.f0, modes.depths, units='dB/m').data[:, 0]``.

            **It need not be the same number the solver used.** A law the
            SSP rows carry (Francois-Garrison, a table) is the law's
            value at each SSP node, interpolated by the solver between
            nodes, so a perturbation sampled between nodes is applied on top
            of a run that absorbed a slightly different alpha there. Both
            routes are correct for what they are; the divergence is worth
            knowing before the two are differenced.

            For :class:`~uacpy.core.absorption.ConstantAbsorption`,
            ``table`` converts dB/wavelength at its ``sound_speed``
            (:data:`~uacpy.core.constants.DEFAULT_SOUND_SPEED` unless
            given), while the deck converts at each SSP row's own ``c``
            (``AttenMod.f90:73``), a spread of ±3.3 % across sound speeds of
            1450-1550 m/s.

            To make the two agree, pass a scalar alpha you computed at the
            same argument the deck used: the SSP's own sound speed for a
            constant absorption.

            A **depth-varying** alpha needs a depth axis fine enough to
            carry ``psi²``: the two trapezoids below cancel exactly while
            ``alpha(z)/c(z)`` is constant, so a scalar alpha is right on any
            grid, and a structured one is only as good as the sampling.
            :attr:`depths` is the merged source/receiver vector, which is
            often 8-20 points, so this is checked against the shortest
            vertical wavelength in the set and a coarse axis raises a
            ``NumericsWarning`` naming the measured and wanted spacing. It
            *warns* rather than raises: the error is continuous in the
            spacing (1-3 % at 51 depths, up to 104 % at 5 on a step alpha),
            the threshold is a calibration rather than a physical boundary,
            and the integrals still return the best value the given samples
            support — unlike the ``seafloor_depth`` mismatch below, which
            reads ``psi(D)`` at a depth that is simply not the seabed.
        sound_speed_z : float or ndarray
            ``c(z)`` in m/s, sampled on :attr:`depths`: the profile the
            modes were computed on, a scalar for an isovelocity column or
            one value per depth (``env.ssp.sound_speed_at(modes.depths)``).
            Required: the perturbation term is ``2iα(z)ω/c(z)`` (Jensen et
            al., *Computational Ocean Acoustics*, sect. 5.9.1), so at a
            fixed ``α(z)`` every ``α_m`` scales as ``1/c`` — a 1480 m/s
            column read at 1500 m/s is 1.3 % under-attenuated — and a
            :class:`Modes` records no water sound speed to take it from, so
            an unset speed is refused rather than guessed.
        density_z : float or ndarray, optional
            ``ρ(z)`` in **g/cm³** (matches :class:`BoundaryProperties`).
            ``None`` reads the water density the producing model recorded
            (``media.water_density``), and the package's one water
            density ``DEFAULT_WATER_DENSITY`` when none is recorded.
            A uniform water density cancels from the water-column integrals;
            it matters against ``bottom.density`` in the seabed tail.
        bottom : BoundaryProperties or Bottom, optional
            Half-space below the water column; ``None`` takes the
            half-space the media table records, where it applies (above).
            ``env.bottom`` is accepted
            when it is range-independent and unlayered (its half-space is
            read); otherwise pass ``env.bottom.halfspace_at(range=0.0)``. When supplied, adds an
            evanescent-tail bottom-attenuation contribution proportional
            to ``ψ²(D)`` **and** completes the ``0 → ∞`` normalisation with
            the matching tail term, so numerator and denominator span the
            same domain; ``bottom.attenuation`` is read in dB/λ_p and
            ``bottom.density`` in g/cm³.

            ``ψ(D)`` is read at the **deepest tabulated depth**
            ``depths[-1]``, so the tabulation must reach the seafloor: a
            grid stopping 20 m short of a 100 m guide was measured to
            inflate the bottom term by 2–4×. Pass ``seafloor_depth`` so
            this is checked. The water-column integrals likewise want the
            full column — a truncated span perturbs the ``∫ψ²/ρ``
            normalisation by the missing tail.
        seafloor_depth : float, optional
            The seafloor depth D in metres. When given with ``bottom``,
            ``depths[-1]`` is validated against it (0.1 % tolerance) and a
            short tabulation raises instead of silently mis-evaluating
            ``ψ(D)``. When omitted, a ``FallbackWarning`` states which depth
            the bottom term was evaluated at; the recorded half-space takes
            ``media.bottom_depth``.

        Returns
        -------
        Modes
            New :class:`Modes` instance with updated complex ``k``.

        Notes
        -----
        The perturbed ``k`` is consumed by uacpy's Python-side modal
        synthesis (:meth:`modal_pressure_field`). It is **not**
        consumed by Acoustics-Toolbox ``field.exe`` / ``fieldS.exe``:
        uacpy has no ``.mod`` writer, and ``field.exe`` reads
        attenuation natively from the environment passed to its field
        run rather than from ``Im(k)`` of the ``.mod`` file. To get the
        perturbed TL from ``field.exe``, attach an :class:`Absorption`
        to the :class:`Environment` and run :class:`Kraken`.

        On plain ``k`` / ``phi`` arrays this is
        :func:`~uacpy.core.acoustics.modal_attenuation`.
        """
        # The whole perturbation — the integrals, the seabed tail, the
        # trapped/leaky split and its three warnings — is
        # acoustics.modal_attenuation's. What this method adds is reading
        # its own k/phi/depths/f0 and re-wrapping the result as a Modes.
        if density_z is None:
            density_z = (DEFAULT_WATER_DENSITY if self.media is None
                         else self.media.water_density)
        # Without bottom=, the fluid half-space the media table records
        # (speed, density, attenuation) gives both the normalisation tail
        # and the seabed-loss term when it lies directly under the water
        # column and the tabulation reaches it.
        media = self.media
        if (bottom is None and media is not None
                and media.halfspace_sound_speed is not None
                and media.halfspace_density is not None
                and media.halfspace_attenuation is not None
                and media.bottom_depth is not None
                and (media.tops is None or len(media.tops) == 1)
                and np.isclose(self.depths[-1], media.bottom_depth,
                               rtol=1e-3)):
            bottom = BoundaryProperties(
                acoustic_type='half-space',
                sound_speed=media.halfspace_sound_speed,
                density=media.halfspace_density,
                attenuation=media.halfspace_attenuation)
            if seafloor_depth is None:
                seafloor_depth = media.bottom_depth
        alpha_m = _modal_attenuation(
            self.k, self.phi, self.depths, alpha_dB_per_m,
            frequency=self.f0, sound_speed_z=sound_speed_z,
            density_z=density_z, bottom=bottom,
            seafloor_depth=seafloor_depth,
            who="Modes.with_attenuation")
        kr = np.real(self.k)
        new_k = kr - 1j * alpha_m
        # halfspace_tails is not carried: its gamma was formed from the k
        # this perturbation replaces.
        return Modes(
            k=new_k, phi=self.phi, depths=self.depths,
            # The solver's group speed survives: this perturbation moves the
            # IMAGINARY part of k, and first_n keeps it too, so the attribute
            # is there whichever method made the set.
            group_speeds=self.group_speeds,
            media=self.media,
            **self.id_kwargs(),
        )

    def modal_pressure_field(
        self,
        *,
        source_depth: float,
        receiver_depths: np.ndarray,
        ranges: np.ndarray,
        source_density: Optional[float] = None,
    ) -> "Field":
        """Coherent complex pressure field built from the modal sum.

        Asymptotic far-field form of the cylindrical-source modal
        expansion (large ``k_m·r``), written in the ``e^{i(ωt − k r)}``
        convention AT propagates under (``EvaluateMod.f90:34,42``):

        ``P(r, z_r) ≈ exp(−iπ/4)·√(2π/r) / ρ_s · Σ_m
        ψ_m(z_s)·ψ_m(z_r) · exp(−i k_m r) / √(k_m)``

        consistent with the ``∫|ψ|²/ρ dz = 1`` normalisation that Kraken
        and the analytic Pekeris helper use. Honors any imaginary
        ``k.imag`` set via :meth:`with_attenuation`.

        ``√(k_m)`` is the **complex** square root (principal branch), not
        ``√|k_m|`` — it carries a ``−arg(k_m)/2`` phase that
        phase-sensitive consumers (MFP, coherent integration) need.

        The reasoning, the literature and the measurements behind this method
        are in ``docs/theory/broadband_products.md``, section "The modal-sum
        prefactor".

        Parameters
        ----------
        source_depth, receiver_depths, ranges
            Source location and the target sample grid (m, m, m).
        source_density : float, optional
            Density at the source depth in **g/cm³** (matches
            :class:`BoundaryProperties`) — the ``ρ(z_s)`` of the modal sum,
            which must be the density the modes were normalised against.
            ``None`` reads it off the medium table the mode set records: the
            water density (``media.water_density``) for a source in
            the water column, the density of the sediment medium holding a
            deeper source (``media.tops`` / ``media.densities``, from the
            ``.mod``; :meth:`MediaTable.density_at`), and
            ``DEFAULT_WATER_DENSITY`` when nothing is recorded — the
            ``ρ(z_s)`` :class:`~uacpy.models.Kraken`'s own field divides by.
            Pass 1.0 to reproduce ``field.exe``'s own output, which omits
            the factor.

        Returns
        -------
        Field
            Complex narrowband ``Field`` with
            ``coords={'depth': receiver_depths, 'range': ranges}``, tagged
            ``phase_reference='travelling_wave'`` like the Kraken
            ``COHERENT_TL`` field the same modes produce.

        Notes
        -----
        **The bottom of an interference null is where a complex64 ``k``
        shows**, which is how a ``.mod`` file stores it. The float32 spacing of
        ``k`` is 3e-8 rad/m, so the *phase* error ``3e-8·r`` reaches one radian
        only near r = 3e4 km — but this sum is itself a cancellation, and where
        the modes cancel the relative error of the total is not bounded by the
        phase error of a term. Measured on a 7-mode 100 Hz Pekeris guide
        (D = 100 m, c 1500/1800 m/s, ρ 1.0/1.8) with exact longdouble roots
        stored both ways, over 1-50 km at ``z_s`` = 20 m, ``z_r`` = 50 m:

        * median 6.9e-04 dB, 0.050 dB at the 99.9th percentile, and 0.004 dB
          over the brightest half of the field. Each of these is stable to
          three digits across range steps from 0.49 m down to 0.024 m; at
          ``z_r`` = 35 m they are 8.5e-04, 0.065 and 0.004 dB.
        * the interference *structure* is unaffected: no null moved by more
          than 0.25 m, at any of those range steps.
        * **the depth of the deepest null does not converge.** It is of order
          1 dB and grows as the range grid is refined — 1.11 dB at a 0.49 m
          step, 1.25 dB at 0.245 m and at 0.122 m, 3.00 dB at 0.024 m —
          because a finer grid samples closer to the bottom of the
          cancellation. Read it as "the deepest nulls are worth about a dB and
          their floor is not determined", not as a number.

        On plain arrays the sum itself is
        :func:`~uacpy.core.acoustics.modal_field`, which takes the mode
        shapes already evaluated at the source and receiver depths.
        """
        if self.n_modes == 0:
            raise ConfigurationError(
                "Modes.modal_pressure_field: the mode set is empty (0 trapped "
                "modes). Below the waveguide's modal cutoff there is no "
                "propagating field to sum — raise the frequency, deepen the "
                "waveguide, or use a full-field model (Scooter/RAM).")
        z_s = float(source_depth)
        z_r = np.atleast_1d(np.asarray(receiver_depths, dtype=float))
        r = np.atleast_1d(np.asarray(ranges, dtype=float))

        # ``kraken.f90:573,598`` tabulates the modes on ``zTab`` — the merged
        # source/receiver depth vector the deck asked for — so ``self.depths``
        # is exactly where phi is known. Outside it there is no mode shape to
        # interpolate: ``np.interp`` would hold the end value flat, which is
        # neither the shape nor the evanescent tail the half-space carries, and
        # would report a plausible number for a depth this mode set never
        # covered. (AT is no better off the end — ``misc/calculateweights.f90:43-49``
        # stops its bracket search at ``L < Nx-1`` and extrapolates linearly —
        # and neither code carries the half-space wavenumber needed for the
        # true ``exp(-gamma_m (z-D))`` tail.) Follow uacpy's depth policy: a
        # receiver below the resolvable domain is accepted as no-data, a
        # source there is fatal because it defines the field.
        who = 'Modes.modal_pressure_field'
        # The source defines the excitation, so outside the tabulation it is
        # fatal; a receiver there is a no-data cell.
        phi_zs = _mode_shapes_at(self.phi, self.depths, z_s, outside='raise',
                                 who=f"{who} (source_depth={z_s:g} m)")[0]
        phi_zr = _mode_shapes_at(self.phi, self.depths, z_r,
                                 outside='nan', who=who)
        outside = np.isnan(phi_zr).all(axis=1)
        if np.any(outside):
            z_lo, z_hi = float(self.depths[0]), float(self.depths[-1])
            warnings.warn(
                f"Modes.modal_pressure_field: {int(outside.sum())} of "
                f"{z_r.size} receiver depths fall outside the tabulated mode "
                f"depths [{z_lo:g}, {z_hi:g}] m and are returned as NaN; the "
                f"mode shapes are not defined there.",
                ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
            # Zeroed for the sum and masked to NaN after it.
            phi_zr = np.where(outside[:, None], 0.0, phi_zr)
        # The sum itself — the Im(k) sign forcing, the complex sqrt, the
        # mode-axis contraction and the prefactor — is modal_field's. What
        # this method adds is the depth interpolation above, the
        # out-of-tabulation masking below and the Field re-wrap.
        if source_density is None:
            source_density = (DEFAULT_WATER_DENSITY if self.media is None
                              else self.media.density_at(z_s))
        P = modal_field(self.k, phi_zs, phi_zr, r,
                        source_density=source_density)
        P[outside, :] = np.nan
        id_kwargs = self.id_kwargs()
        id_kwargs['backend'] = 'modal_sum'
        id_kwargs['source_depths'] = np.array([z_s])
        # The sum is written in AT's e^{i(ωt − k r)} convention (above), the
        # travelling-wave form Kraken's own COHERENT_TL field is stamped
        # with, so the two carry one phase reference.
        id_kwargs['phase_reference'] = PhaseReference.TRAVELLING_WAVE
        return Field(
            data=P,
            coords={'depth': z_r, 'range': r},
            **id_kwargs,
        )


@dataclass(frozen=True, eq=False)
class ModeDispersion(ExportRecord):
    """Phase and group speed of every mode against frequency, what
    :meth:`Modes.dispersion` returns; ``.plot()`` draws the dispersion
    diagram with :func:`uacpy.plot.plot_dispersion`.

    Attributes
    ----------
    frequencies : ndarray, shape ``(n_f,)``
        The mode sets' frequencies (Hz), increasing.
    phase_speeds : ndarray, shape ``(n_f, n_modes)``
        Phase speed (m/s) of mode ``m`` (column ``m - 1``) at each
        frequency; NaN where the set holds no such mode (below its cutoff).
    group_frequencies : ndarray, shape ``(n_g,)``
        Where the group speeds sit (Hz): a set's frequency for a speed the
        solver reported, the midpoint of a pair for a finite difference.
    group_speeds : ndarray, shape ``(n_g, n_modes)``
        Group speed (m/s); NaN where none is known.
    model, backend, model_source_id : str or None
        The engine the mode sets came from, for the figure's credit.
    """

    frequencies: np.ndarray
    phase_speeds: np.ndarray
    group_frequencies: np.ndarray
    group_speeds: np.ndarray
    model: str = ''
    backend: Optional[str] = None
    model_source_id: Optional[str] = None

    _ARRAY_FIELDS = ('frequencies', 'phase_speeds', 'group_frequencies',
                     'group_speeds')
    _REPR_UNITS = {'phase_speeds': 'm/s', 'group_frequencies': 'Hz',
                   'group_speeds': 'm/s'}
    _XARRAY_FIELDS = {'frequency': 'frequencies',
                      'group_frequency': 'group_frequencies',
                      'phase_speed': 'phase_speeds',
                      'group_speed': 'group_speeds'}

    def _payload(self):
        return {'phase_speed': (self.phase_speeds, ('frequency', 'mode'),
                                'm/s'),
                'group_speed': (self.group_speeds, ('group_frequency',
                                                    'mode'), 'm/s')}

    def _coords(self):
        return {'frequency': (self.frequencies, 'Hz'),
                'group_frequency': (self.group_frequencies, 'Hz'),
                'mode': (np.arange(1, self.phase_speeds.shape[1] + 1), '')}

    @property
    def n_modes(self) -> int:
        return int(self.phase_speeds.shape[1])

    def plot(self, **kwargs):
        """Draw this diagram with :func:`uacpy.plot.plot_dispersion`;
        ``kwargs`` reach it. Returns ``(fig, ax)``."""
        return plotter('plot_dispersion')(self, **kwargs)


@dataclass(frozen=True, eq=False)
class ExcitationPattern(ExportRecord):
    """A source array's modal excitation at each mode's grazing angle, and
    its free-field pattern folded over ±θ, what
    :meth:`Modes.excitation_pattern` returns; ``.plot()`` draws both with
    :func:`uacpy.plot.plot_mode_excitation`.

    Attributes
    ----------
    grazing_angles_deg : ndarray, shape ``(n_modes,)``
        Each mode's grazing angle (deg); NaN for a mode evanescent at the
        reference speed.
    excitation : ndarray, shape ``(n_modes,)``
        ``|Σₙ wₙ·φₘ(zₙ)|`` per mode, in the mode shapes' normalisation.
    pattern_angles_deg : ndarray
        Grazing angles (deg) of the free-field pattern, from 0.
    pattern : ndarray
        ``max(|P(+θ)|, |P(−θ)|)``: the array beam pattern when
        ``directional``, the array factor otherwise.
    directional : bool
        Whether the elements carry a beam pattern.
    sound_speed : float
        The reference speed (m/s) the angles are measured against.
    n_sources : int
        Elements in the array.
    aperture : float
        The array's vertical extent (m).
    model, backend, model_source_id : str or None
        The engine the modes came from, for the figure's credit.
    """

    grazing_angles_deg: np.ndarray
    excitation: np.ndarray
    pattern_angles_deg: np.ndarray
    pattern: np.ndarray
    directional: bool
    sound_speed: float
    n_sources: int
    aperture: float
    model: str = ''
    backend: Optional[str] = None
    model_source_id: Optional[str] = None

    _ARRAY_FIELDS = ('grazing_angles_deg', 'excitation', 'pattern_angles_deg',
                     'pattern')
    _REPR_UNITS = {'grazing_angles_deg': 'deg', 'pattern_angles_deg': 'deg',
                   'sound_speed': 'm/s', 'aperture': 'm'}
    _XARRAY_FIELDS = {'grazing_angle': 'grazing_angles_deg',
                      'excitation': 'excitation',
                      'pattern_angle': 'pattern_angles_deg',
                      'pattern': 'pattern'}

    def _payload(self):
        return {'grazing_angle': (self.grazing_angles_deg, ('mode',), 'deg'),
                'excitation': (self.excitation, ('mode',), ''),
                'pattern': (self.pattern, ('pattern_angle',), '')}

    def _coords(self):
        return {'mode': (np.arange(1, self.excitation.size + 1), ''),
                'pattern_angle': (self.pattern_angles_deg, 'deg')}

    def plot(self, **kwargs):
        """Draw this with :func:`uacpy.plot.plot_mode_excitation`; ``kwargs``
        reach it. Returns ``(fig, ax)``."""
        return plotter('plot_mode_excitation')(self, **kwargs)
