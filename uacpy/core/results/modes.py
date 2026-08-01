"""Normal-mode result type."""

from __future__ import annotations

import numpy as np
from typing import Union

from uacpy.core.exceptions import ConfigurationError

from uacpy.core.results._base import Result
from uacpy.core.results.field import Field


class Modes(Result):
    """Kraken normal modes — depth eigenfunctions of the Helmholtz operator.

    Attributes
    ----------
    k : ndarray, shape ``(n_modes,)`` complex
        Modal horizontal wavenumbers.
    phi : ndarray, shape ``(n_depths, n_modes)``
        Mode shapes sampled at ``depths``.
    depths : ndarray, shape ``(n_depths,)``
        Sampling depths.
    """
    field_type = "modes"

    def __init__(
        self,
        *,
        k: np.ndarray,
        phi: np.ndarray,
        depths: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
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

    @property
    def n_modes(self) -> int:
        """Number of modes — always ``len(k)`` (single source of truth).

        Derived rather than stored so it can never desync from ``k``/``phi``;
        to use fewer modes, slice via :meth:`first_n` (which trims ``k`` and
        ``phi`` together).
        """
        return len(self.k)

    def _repr_extra(self) -> str:
        return f"n_modes={self.n_modes}, n_z={self.depths.size}"

    def first_n(self, n: int) -> "Modes":
        """Return a new :class:`Modes` containing only the first ``n`` modes.

        No-op when ``n >= self.n_modes``. ``k`` is sliced as ``k[:n]`` and
        ``phi`` as ``phi[:, :n]``; depths and identification metadata are
        preserved.
        """
        if n >= len(self.k):
            return self
        new_k = self.k[:n]
        new_phi = self.phi[:, :n]
        return Modes(
            k=new_k,
            phi=new_phi,
            depths=self.depths,
            **self.id_kwargs(),
        )

    def compute_phase_speeds(self) -> np.ndarray:
        """Mode phase speeds ``v_p = ω / Re(k_r)`` in m/s.

        Raises
        ------
        ConfigurationError
            If this :class:`Modes` instance has no frequency context
            (``self.f0 is None``); without a frequency the phase speed
            is undefined. Pass ``frequencies=…`` to the wrapper that
            built this object, or set it on the instance, before calling.
        """
        if self.f0 is None:
            raise ConfigurationError(
                "Modes.compute_phase_speeds requires frequencies; got None"
            )
        omega = 2.0 * np.pi * self.f0
        return omega / np.real(self.k)

    def compute_group_velocity(self, other: "Modes") -> np.ndarray:
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
            the shared count).
        """
        f0_self, f0_other = self.f0, other.f0
        if f0_self is None or f0_other is None:
            raise ConfigurationError(
                "Modes.compute_group_velocity: both Modes instances must "
                "have a frequency"
            )
        if f0_self == f0_other:
            raise ConfigurationError(
                "Modes.compute_group_velocity: requires Modes at two distinct "
                "frequencies"
            )
        n = min(self.n_modes, other.n_modes)
        if n == 0:
            return np.array([])
        domega = 2.0 * np.pi * (f0_other - f0_self)
        dk = np.real(other.k[:n] - self.k[:n])
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(dk != 0, domega / dk, np.nan)

    def with_attenuation(
        self,
        alpha_db_per_m: Union[float, np.ndarray],
        *,
        sound_speed_z: Union[float, np.ndarray] = 1500.0,
        density_z: Union[float, np.ndarray] = 1.0,
        bottom=None,
    ) -> "Modes":
        """First-order modal attenuation perturbation.

        For mode ``m`` with horizontal wavenumber ``k_rm`` and depth
        eigenfunction ``ψ_m``, the imaginary part picks up

        ``α_m = (ω / k_rm) · ∫ α(z)/(c(z) ρ(z)) · |ψ_m|² dz / ∫ |ψ_m|² / ρ(z) dz``

        replacing any prior ``k.imag``. The ``1/ρ``-weighted denominator
        matches the Kraken-class normalisation ``∫|ψ|²/ρ dz = 1``.

        Parameters
        ----------
        alpha_db_per_m : float or ndarray
            Per-depth volume attenuation in dB/m, sampled on
            :attr:`depths`. Scalar broadcasts to every depth. Build one
            from an :class:`~uacpy.core.absorption.Absorption` via
            ``absorption.alpha_db_per_m(modes.f0, modes.depths)``.
        sound_speed_z : float or ndarray
            ``c(z)`` in m/s. Defaults to 1500.
        density_z : float or ndarray
            ``ρ(z)`` in **g/cm³** (matches :class:`BoundaryProperties`).
            Defaults to 1.0 (fresh-water reference; 1.025 for seawater).
        bottom : BoundaryProperties, optional
            Half-space below the water column. When supplied, adds an
            evanescent-tail bottom-attenuation contribution proportional
            to ``ψ²(D)``; ``bottom.attenuation`` is read in dB/λ_p and
            ``bottom.density`` in g/cm³.

        Returns
        -------
        Modes
            New :class:`Modes` instance with updated complex ``k``.

        Notes
        -----
        The perturbed ``k`` is consumed by uacpy's Python-side modal
        synthesis (:meth:`pressure_at`, :meth:`tl_at`). It is **not**
        consumed by Acoustics-Toolbox ``field.exe`` / ``fieldS.exe``:
        uacpy has no ``.mod`` writer, and ``field.exe`` reads
        attenuation natively from the environment passed to its field
        run rather than from ``Im(k)`` of the ``.mod`` file. To get the
        perturbed TL from ``field.exe``, attach an :class:`Absorption`
        to the :class:`Environment` and run :class:`Kraken`.
        """
        omega = 2.0 * np.pi * float(self.f0 or 0.0)
        if omega == 0.0:
            raise ConfigurationError(
                "Modes.with_attenuation: requires Modes.f0 to be set."
            )
        a = np.asarray(alpha_db_per_m, dtype=float).ravel()
        if a.size == 1:
            a = np.full_like(self.depths, float(a.item()))
        elif a.shape != self.depths.shape:
            raise ConfigurationError(
                f"Modes.with_attenuation: alpha shape {a.shape} "
                f"must match depths {self.depths.shape} (or scalar)."
            )
        c_arr = np.asarray(sound_speed_z, dtype=float).ravel()
        if c_arr.size == 1:
            c_arr = np.full_like(self.depths, float(c_arr.item()))
        elif c_arr.shape != self.depths.shape:
            raise ConfigurationError(
                f"Modes.with_attenuation: sound_speed_z shape "
                f"{c_arr.shape} must match depths {self.depths.shape}"
            )
        rho_g = np.asarray(density_z, dtype=float).ravel()
        if rho_g.size == 1:
            rho_g = np.full_like(self.depths, float(rho_g.item()))
        elif rho_g.shape != self.depths.shape:
            raise ConfigurationError(
                f"Modes.with_attenuation: density_z shape "
                f"{rho_g.shape} must match depths {self.depths.shape}"
            )
        rho_arr = rho_g * 1000.0  # g/cm³ → kg/m³
        a_neper = a * (np.log(10.0) / 20.0)
        phi_re = np.asarray(self.phi).real
        weight = phi_re ** 2
        norm = np.trapezoid(weight / rho_arr[:, None], self.depths, axis=0)
        norm = np.where(norm > 0, norm, 1.0)
        integrand = (a_neper / (c_arr * rho_arr))[:, None] * weight
        kr = np.real(self.k)
        kr_safe = np.where(kr > 0, kr, 1.0)
        alpha_m = (omega / kr_safe) * np.trapezoid(integrand, self.depths, axis=0) / norm
        if bottom is not None:
            from uacpy.core.environment import BoundaryProperties as _BP
            if not isinstance(bottom, _BP):
                raise ConfigurationError(
                    "Modes.with_attenuation: bottom must be a "
                    f"BoundaryProperties; got {type(bottom).__name__}"
                )
            cb = float(bottom.sound_speed)
            rho_b = float(bottom.density) * 1000.0
            ab_neper_per_m = (
                float(bottom.attenuation) * np.log(10.0) / 20.0
                * float(self.f0) / cb
            )
            psi_D = phi_re[-1, :]
            kb = omega / cb
            gamma_m = np.sqrt(np.maximum(kr ** 2 - kb ** 2, 0.0))
            gamma_safe = np.where(gamma_m > 0, gamma_m, 1.0)
            alpha_bottom = (
                psi_D ** 2 * ab_neper_per_m * omega
                / (2.0 * kr_safe * gamma_safe * cb * rho_b)
            )
            alpha_m = alpha_m + alpha_bottom / norm
        new_k = kr + 1j * alpha_m
        return Modes(
            k=new_k, phi=self.phi, depths=self.depths,
            **self.id_kwargs(),
        )

    def modal_propagation_loss(
        self,
        *,
        source_depth: float,
        receiver_depths: np.ndarray,
        ranges_m: np.ndarray,
        source_density: float = 1.0,
    ) -> "Field":
        """Coherent complex pressure field built from the modal sum.

        Asymptotic far-field form of the cylindrical-source modal
        expansion (large ``k_m·r``):

        ``P(r, z_r) ≈ i·exp(−iπ/4) / (ρ_s·√(8πr)) · Σ_m
        ψ_m(z_s)·ψ_m(z_r) · exp(i k_m r) / √|k_m|``

        consistent with the ``∫|ψ|²/ρ dz = 1`` normalisation that Kraken
        and the analytic Pekeris helper use. Honors any imaginary
        ``k.imag`` set via :meth:`with_attenuation`.

        Parameters
        ----------
        source_depth, receiver_depths, ranges_m
            Source location and the target sample grid (m, m, m).
        source_density : float
            Water density at the source depth in **g/cm³** (matches
            :class:`BoundaryProperties`). Defaults to 1.0.

        Returns
        -------
        Field
            Complex narrowband ``Field`` with
            ``coords={'depth': receiver_depths, 'range': ranges_m}``.
        """
        if self.n_modes == 0:
            raise ConfigurationError(
                "Modes.modal_propagation_loss: the mode set is empty (0 trapped "
                "modes). Below the waveguide's modal cutoff there is no "
                "propagating field to sum — raise the frequency, deepen the "
                "waveguide, or use a full-field model (Scooter/RAM).")
        z_s = float(source_depth)
        z_r = np.atleast_1d(np.asarray(receiver_depths, dtype=float))
        r = np.atleast_1d(np.asarray(ranges_m, dtype=float))
        phi = np.asarray(self.phi)
        is_complex = np.iscomplexobj(phi)
        if is_complex:
            phi_zs = np.array([
                np.interp(z_s, self.depths, phi[:, m].real)
                + 1j * np.interp(z_s, self.depths, phi[:, m].imag)
                for m in range(self.n_modes)
            ])
            phi_zr = np.column_stack([
                np.interp(z_r, self.depths, phi[:, m].real)
                + 1j * np.interp(z_r, self.depths, phi[:, m].imag)
                for m in range(self.n_modes)
            ])
        else:
            phi_zs = np.array([
                float(np.interp(z_s, self.depths, phi[:, m]))
                for m in range(self.n_modes)
            ])
            phi_zr = np.column_stack([
                np.interp(z_r, self.depths, phi[:, m])
                for m in range(self.n_modes)
            ])
        k = np.asarray(self.k)
        # Normalize the attenuation sign so modes always decay with range under
        # the exp(+i k r) convention used below. Raw Kraken eigenvalues
        # encode decay as k.imag < 0 (see modes_reader), whereas with_attenuation
        # builds k.imag > 0; a passive medium can only attenuate, so force
        # Im(k) >= 0 and the result is convention-agnostic either way.
        k = k.real + 1j * np.abs(k.imag)
        # Complex sqrt — preserves the -arg(k)/2 phase contribution that
        # matters for phase-sensitive consumers (MFP, coherent integration).
        # Numpy's sqrt picks the principal branch (positive real part).
        inv_sqrt_k = 1.0 / np.sqrt(k.astype(np.complex128))
        weights = phi_zs * inv_sqrt_k
        expikr = np.exp(1j * k[:, None] * r[None, :])
        contribution = (phi_zr * weights)[:, :, None] * expikr[None, :, :]
        P = contribution.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            sqrt_r = np.sqrt(r)
            sqrt_r = np.where(sqrt_r > 0, sqrt_r, 1.0)
        rho_s = float(source_density) * 1000.0  # g/cm³ → kg/m³
        pref = 1j * np.exp(-1j * np.pi / 4.0) / (rho_s * np.sqrt(8.0 * np.pi))
        P = pref * P / sqrt_r[None, :]
        id_kwargs = self.id_kwargs()
        id_kwargs['backend'] = 'modal_sum'
        id_kwargs['source_depths'] = np.array([z_s])
        return Field(
            data=P,
            coords={'depth': z_r, 'range': r},
            **id_kwargs,
        )
