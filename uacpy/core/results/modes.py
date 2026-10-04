"""Normal-mode result type."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from typing import Optional, Tuple, Union

from uacpy.core.exceptions import ConfigurationError, ValidityWarning
from uacpy.core._records import FrozenRecord
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.acoustics.modal import (
    _mode_shapes_at, _modal_attenuation, modal_excitation, modal_field,
    _modal_grazing_angles, modal_phase_speeds,
)
from uacpy.core.bottom import medium_density_at

from uacpy.core.results._base import (PhaseReference, Result, _count,
                                      coordinate_axis)
from uacpy.core.results.field import Field
from uacpy.core.results.quantities import coordinate_unit
from uacpy.core._repr import count
from uacpy.core.constants import DEFAULT_SOUND_SPEED, DEFAULT_WATER_DENSITY_G_CM3


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
    """

    water_density: float
    tops: Optional[Tuple[float, ...]] = None
    densities: Optional[Tuple[float, ...]] = None
    bottom_depth: Optional[float] = None
    halfspace_density: Optional[float] = None

    _REPR_UNITS = {'water_density': 'g/cm³', 'tops': 'm', 'densities': 'g/cm³',
                   'bottom_depth': 'm', 'halfspace_density': 'g/cm³'}

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
        for name in ('bottom_depth', 'halfspace_density'):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))

    @classmethod
    def from_dict(cls, d) -> 'MediaTable':
        return cls(**d)

    def density_at(self, depth: float) -> float:
        """``ρ(z)`` (g/cm³) of the medium holding ``depth`` (m)
        (:func:`~uacpy.core.bottom.medium_density_at`): :attr:`water_density` down to the
        seafloor, each deeper medium's own density below its top, and
        :attr:`halfspace_density` below the last medium's base. A table with
        no tabulated media (a ``.moa``, or one built by hand) answers the
        water density for every depth."""
        if not self.tops or len(self.tops) < 2:
            return self.water_density
        return medium_density_at(
            depth, self.tops, (self.water_density,) + self.densities[1:],
            np.inf if self.bottom_depth is None else self.bottom_depth,
            (self.densities[-1] if self.halfspace_density is None
             else self.halfspace_density))


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
        nowhere else, and this object carries no half-space wavenumber from
        which the evanescent tail below the span could be continued.
    media : MediaTable or None
        The densities the modes were normalised in, from which
        :meth:`modal_pressure_field` takes ``ρ(z_s)`` and
        :meth:`with_attenuation` its water density; ``None`` when no
        density is recorded (a ``.moa``, or a mode set built by hand).
        A ``metadata`` carrying ``water_density``, ``media_depths``,
        ``media_densities``, ``media_bottom_depth`` or ``halfspace_density``
        is refused.
    group_velocity : ndarray, shape ``(n_modes,)``, or None
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

        Use :meth:`group_velocity_between` instead when this is ``None``, or
        when every mode is needed from a large set. The perturbation caveat
        the source states at ``kraken.f90:772`` applies either way: "group
        speeds will be wrong for leaky modes".
    """

    #: The metadata spellings of the :class:`MediaTable` members: a file
    #: that keeps them in its metadata loads them into :attr:`media`, and a
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
        group_velocity: Optional[np.ndarray] = None,
        media: Optional[MediaTable] = None,
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
        if group_velocity is None:
            self.group_velocity = None
        else:
            gv = np.atleast_1d(np.array(group_velocity, dtype=float))
            if gv.shape != self.k.shape:
                raise ConfigurationError(
                    f"Modes.group_velocity: shape {gv.shape} must equal "
                    f"k's {self.k.shape} — one group speed per mode, NaN "
                    f"where the binary did not report one."
                )
            self.group_velocity = gv

    @property
    def n_modes(self) -> int:
        """Number of modes — always ``len(k)`` (single source of truth).

        Derived rather than stored so it can never desync from ``k``/``phi``;
        to use fewer modes, slice via :meth:`first_n` (which trims ``k`` and
        ``phi`` together).
        """
        return len(self.k)

    def _repr_bits(self) -> list:
        return [count(self.n_modes, 'mode'), coordinate_axis('depth', self.depths)]

    def to_dict(self) -> dict:
        """Serialise these modes to plain arrays: ``k``, ``phi`` (depth ×
        mode), ``depths``, ``group_velocity`` (``None`` when the solver
        reported none), ``media`` (the :class:`MediaTable` as its plain
        dict, or ``None``) and the identity, as :meth:`Field.to_dict` writes it.
        ``np.savez(f, **d)`` stores it; read it back with
        ``np.load(f, allow_pickle=True)`` into :meth:`from_dict`."""
        return {
            'k': self.k.copy(),
            'phi': self.phi.copy(),
            'depths': self.depths.copy(),
            'group_velocity': (None if self.group_velocity is None
                               else self.group_velocity.copy()),
            'media': None if self.media is None else self.media.to_dict(),
            **self._identity_dict(),
        }

    #: ``attrs`` the xarray export writes for :attr:`media`, by member.
    _MEDIA_ATTRS = {member: f'media_{member}'
                    for member in ('water_density', 'tops', 'densities',
                                   'bottom_depth', 'halfspace_density')}

    def _payload(self):
        payload = {'k': (self.k, ('mode',), 'rad/m'),
                   'phi': (self.phi, ('depth', 'mode'), '')}
        if self.group_velocity is not None:
            payload['group_velocity'] = (self.group_velocity, ('mode',),
                                         'm/s')
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
        return cls(k=arrays['k'], phi=arrays['phi'], depths=arrays['depth'],
                   group_velocity=arrays.get('group_velocity'), media=media,
                   **cls._identity_from_attrs(
                       attrs, tuple(cls._MEDIA_ATTRS.values())))

    def _table(self):
        """One row per mode: ``mode`` (1-based), ``k_real`` and ``k_imag``
        (rad/m), ``phase_speed`` (m/s; NaN without a frequency),
        ``group_speed`` (m/s; NaN where the solver reported none) and
        ``attenuation_dB_per_km``, the modal decay ``20·log10(e)·|Im(k)|``
        per km: the decay rate :func:`~uacpy.core.acoustics.modal_field`
        applies, which takes ``|Im(k)|`` whichever sign convention the
        solver wrote."""
        k = np.asarray(self.k)
        phase_speed = (np.full(self.n_modes, np.nan) if self.f0 is None
                       else self.phase_speeds)
        group_speed = (np.full(self.n_modes, np.nan)
                       if self.group_velocity is None
                       else np.asarray(self.group_velocity, dtype=float))
        return {
            'mode': np.arange(1, self.n_modes + 1),
            'k_real': np.real(k).astype(float),
            'k_imag': np.imag(k).astype(float),
            'phase_speed': np.asarray(phase_speed, dtype=float),
            'group_speed': group_speed,
            'attenuation_dB_per_km': 20.0 * np.log10(np.e)
            * np.abs(np.imag(k)).astype(float) * 1000.0,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Modes":
        """Reconstruct :class:`Modes` from :meth:`to_dict` output, or from
        the mapping ``np.load(f, allow_pickle=True)`` returns for a file
        written with ``np.savez(f, **modes.to_dict())``. A file that keeps
        the media table in its metadata loads it into :attr:`media`.

        Parameters
        ----------
        d : mapping
            :meth:`to_dict` output, or the mapping ``np.load`` returns for it.
        """
        d = cls._unwrap_saved(d, payload=('k', 'phi', 'depths'))
        identity = cls._identity_from_dict(d)
        metadata = dict(identity['metadata'] or {})
        stated = {member: metadata.pop(key)
                  for key, member in cls._METADATA_MEDIA.items()
                  if key in metadata}
        identity['metadata'] = metadata or None
        media = d.get('media')
        if media is not None:
            media = MediaTable.from_dict(media)
        elif stated:
            media = MediaTable(
                water_density=stated.pop('water_density',
                                         DEFAULT_WATER_DENSITY_G_CM3),
                **stated)
        return cls(k=d['k'], phi=d['phi'], depths=d['depths'],
                   group_velocity=d.get('group_velocity'), media=media,
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
            group_velocity=(None if self.group_velocity is None
                            else self.group_velocity[:n]),
            media=self.media,
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

    def group_velocity_between(self, other: "Modes") -> np.ndarray:
        """Approximate group velocity ``v_g = dω/dk`` using a second
        :class:`Modes` instance at a nearby frequency.

        Parameters
        ----------
        other : Modes
            Modes computed at a slightly different frequency.

        Returns
        -------
        v_g : ndarray, shape ``(min(self.n_modes, other.n_modes),)``
            Mode-by-mode group velocity in m/s. Modes that exist in only
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
                "Modes.group_velocity_between: both Modes instances must "
                "have a frequency."
            )
        if f0_self == f0_other:
            raise ConfigurationError(
                "Modes.group_velocity_between: requires Modes at two distinct "
                "frequencies."
            )
        n = min(self.n_modes, other.n_modes)
        if n == 0:
            return np.array([])
        # The difference quotient, its storage-resolution notice and its
        # monotonicity refusal are modal_group_velocity's; what this method
        # adds is pairing two Modes and truncating to the shared set.
        #
        # That function REFUSES a k_r that does not rise strictly with
        # frequency, where this returned nan for a flat step and a silently
        # NEGATIVE speed for a falling one. k_r rises because v_g is an
        # energy-transport speed, so a falling step is bad input and not a
        # cell with no answer; a negative group velocity handed back without
        # comment is the worse of the two.
        lo, hi = ((self, other) if f0_self < f0_other else (other, self))
        # Deferred: acoustic_signal pulls scipy, and uacpy's public
        # surface is imported without it (test_lazy_imports).
        from uacpy.acoustic_signal.dispersion import modal_group_velocity
        return modal_group_velocity(
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

        A directional element shades this too. ``field.f90`` multiplies the
        modal excitation by ``S(θₘ)`` before summing (``C = C * REAL(S)``),
        and every element of a uacpy ``Source`` shares one ``beam_pattern``
        and one mode angle, so ``S`` factors straight out of the array sum —
        the product theorem again, in the modal domain. Pass
        ``sound_speed`` to evaluate the mode angles it is read at; a source
        that carries a pattern and is given no speed raises rather than
        silently returning the omnidirectional answer.

        Parameters
        ----------
        source : Source
            Its ``depths`` and complex ``weights``. A single-depth source
            gives that depth's mode shapes scaled by its one weight.
        sound_speed : float, optional
            Reference speed (m/s) for the mode grazing angles
            ``θₘ = arccos(c / vₚ,ₘ)`` the element pattern is sampled at.
            Required when ``source.beam_pattern`` is set, unused otherwise.

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
                "which the engine applies to the modal excitation at each "
                "mode's grazing angle — so the answer depends on the speed "
                "those angles are measured against. Pass sound_speed= (the "
                "speed at the source depth), or drop the pattern."
            )
        angles = self.grazing_angles(sound_speed)
        undefined = ~np.isfinite(angles)
        if np.any(undefined):
            warnings.warn(
                f"Modes.excitation: {int(np.sum(undefined))} of "
                f"{angles.size} modes have a phase speed below the "
                f"{float(sound_speed):g} m/s reference, so they are "
                f"evanescent there and have no real grazing angle. Their "
                f"excitation is returned as NaN rather than weighted by the "
                f"beam pattern at a fabricated angle — clipping them to "
                f"broadside attenuated one by a factor of 100 through a "
                f"-40 dB notch, silently. Pass the speed at the source "
                f"depth if these modes matter.",
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
                "Modes.phase_speeds requires frequencies; got None."
            )
        return _modal_grazing_angles(self.k, self.f0, sound_speed,
                                    who='Modes.grazing_angles')

    def with_attenuation(
        self,
        alpha_dB_per_m: Union[float, np.ndarray],
        *,
        sound_speed_z: Union[float, np.ndarray] = DEFAULT_SOUND_SPEED,
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
        ``ρ_b`` and therefore needs ``bottom=``. **Without ``bottom=`` the
        normalisation stops at the seabed and every returned ``Im(k)`` is
        an upper bound**: the omitted term is positive and only ever lowers
        ``α_m``. Measured on an analytic Pekeris guide (D = 100 m, c
        1500/1800 m/s, ρ 1.0/1.8 g/cm³, 50 Hz, uniform α in the water) the
        water-only normalisation returns 0.7 %, 2.4 %, 5.1 % and 17 % high
        for modes 1-4 — the near-cutoff mode, which dominates at long
        range, is over-attenuated the most because its shallow ``γ_m``
        pushes the most energy into the bottom.

        **A barely trapped mode is warned about**, because the same ``γ_m``
        that makes its tail term large also makes that term rest on the part
        of ``kr`` a mode solver resolves least well. The trigger is a ratio
        of the two normalisation integrals — the seabed tail against the
        water column — and it fires where a converged KRAKEN mesh moved the
        total modal TL by 0.67-1.98 dB at 200-500 m on a 100 m Pekeris guide
        and by 7.07 dB at 10 km, against 0.41 dB at the first frequency above
        the trigger. Near a modal cutoff the mesh also decides *which side*
        of ``k_b`` the mode lands on, and a mode reported as leaky there gets
        no bottom term at all; the leaky warning names ``k_b`` so the margin
        can be read off.

        Parameters
        ----------
        alpha_dB_per_m : float or ndarray
            Per-depth volume attenuation in dB/m, sampled on
            :attr:`depths`. Scalar broadcasts to every depth. Build one
            from an :class:`~uacpy.core.absorption.Absorption` via
            ``absorption.alpha_dB_per_m(modes.f0, modes.depths)``.

            **It need not be the same number the solver used.** A law the
            SSP rows carry (Francois-Garrison, a table) is the accessor's
            value at each SSP node, interpolated by the solver between
            nodes, so a perturbation sampled between nodes is applied on top
            of a run that absorbed a slightly different alpha there. Both
            routes are correct for what they are; the divergence is worth
            knowing before the two are differenced.

            For :class:`~uacpy.core.absorption.ConstantAbsorption`, the
            accessor converts dB/wavelength at
            :data:`~uacpy.core.constants.DEFAULT_SOUND_SPEED`, while the
            deck converts at each SSP row's own ``c`` (``AttenMod.f90:73``),
            a spread of ±3.3 % across sound speeds of 1450-1550 m/s.

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
            ``c(z)`` in m/s. Defaults to 1500.
        density_z : float or ndarray, optional
            ``ρ(z)`` in **g/cm³** (matches :class:`BoundaryProperties`).
            ``None`` reads the water density the producing model recorded
            (``media.water_density``), and the package's one water
            density ``DEFAULT_WATER_DENSITY_G_CM3`` when none is recorded.
            A uniform water density cancels from the water-column integrals;
            it matters against ``bottom.density`` in the seabed tail.
        bottom : BoundaryProperties or Bottom, optional
            Half-space below the water column. ``env.bottom`` is accepted
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
            the bottom term was evaluated at.

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
            density_z = (DEFAULT_WATER_DENSITY_G_CM3 if self.media is None
                         else self.media.water_density)
        alpha_m = _modal_attenuation(
            self.k, self.phi, self.depths, alpha_dB_per_m,
            frequency=self.f0, sound_speed_z=sound_speed_z,
            density_z=density_z, bottom=bottom,
            seafloor_depth=seafloor_depth,
            who="Modes.with_attenuation")
        kr = np.real(self.k)
        new_k = kr - 1j * alpha_m
        return Modes(
            k=new_k, phi=self.phi, depths=self.depths,
            # The solver's group speed survives: this perturbation moves the
            # IMAGINARY part of k, and dropping it here while first_n keeps it
            # made the attribute disappear depending on which method was called.
            group_velocity=self.group_velocity,
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
            ``DEFAULT_WATER_DENSITY_G_CM3`` when nothing is recorded — the
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
            source_density = (DEFAULT_WATER_DENSITY_G_CM3 if self.media is None
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
