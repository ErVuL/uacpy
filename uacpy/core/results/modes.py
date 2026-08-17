"""Normal-mode result type."""

from __future__ import annotations

import warnings

import numpy as np
from typing import Optional, Union

from uacpy.core.exceptions import ConfigurationError

from uacpy.core.results._base import Result
from uacpy.core.results.field import Field


class Modes(Result):
    """Kraken normal modes — depth eigenfunctions of the Helmholtz operator.

    Attributes
    ----------
    k : ndarray, shape ``(n_modes,)`` complex
        Modal horizontal wavenumbers (rad/m). A nonzero imaginary part
        means an attenuating mode — leaky modes from ``backend='krakenc'``,
        or the perturbation :meth:`with_attenuation` applies.
    phi : ndarray, shape ``(n_depths, n_modes)``
        Mode shapes sampled at ``depths``.
    depths : ndarray, shape ``(n_depths,)``
        Depths (m below the surface) the mode shapes are tabulated at —
        KRAKEN's ``zTab``, the merged, sorted, duplicate-free union of the
        run's source and receiver depths (``Kraken/kraken.f90:573`` builds
        it, ``:598`` writes it to the ``.mod``). The modes are known
        nowhere else, and this object carries no half-space wavenumber from
        which the evanescent tail below the span could be continued.
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
        if n < 1:
            raise ConfigurationError(
                f"Modes.first_n({n}): need n >= 1 — a negative n would slice "
                "from the end of the mode set and silently return a different "
                "subset than requested.")
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
        seafloor_depth: Optional[float] = None,
    ) -> "Modes":
        """First-order modal attenuation perturbation.

        For mode ``m`` with horizontal wavenumber ``k_rm`` and depth
        eigenfunction ``ψ_m``, the imaginary part picks up

        ``α_m = (ω / k_rm) · ∫ α(z)/(c(z) ρ(z)) · Re(ψ_m)² dz / ∫ Re(ψ_m)² / ρ(z) dz``

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
            ``ψ(D)``. When omitted, a ``UserWarning`` states which depth
            the bottom term was evaluated at.

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
        # Perturbation on Re(psi)**2: for krakenc's complex modes this is
        # an approximation (JKPS 5.176 wants the complex psi**2); the
        # imaginary part of a weakly-attenuated mode shape is O(alpha),
        # so the error is second-order in the loss being computed.
        phi_re = np.asarray(self.phi).real
        weight = phi_re ** 2
        norm = np.trapezoid(weight / rho_arr[:, None], self.depths, axis=0)
        if np.any(norm <= 0):
            warnings.warn(
                f"Modes.with_attenuation: {int(np.count_nonzero(norm <= 0))} "
                f"mode(s) have a non-positive shape normalisation "
                f"∫psi²/rho dz; their attenuation is computed with the "
                f"normalisation clamped to 1 and is not meaningful.",
                UserWarning, stacklevel=2)
        norm = np.where(norm > 0, norm, 1.0)
        integrand = (a_neper / (c_arr * rho_arr))[:, None] * weight
        kr = np.real(self.k)
        if np.any(kr <= 0):
            warnings.warn(
                f"Modes.with_attenuation: {int(np.count_nonzero(kr <= 0))} "
                f"mode(s) have Re(k) <= 0; their attenuation is computed "
                f"with k clamped to 1 m⁻¹ and is not meaningful.",
                UserWarning, stacklevel=2)
        kr_safe = np.where(kr > 0, kr, 1.0)
        alpha_m = (omega / kr_safe) * np.trapezoid(integrand, self.depths, axis=0) / norm
        # A truncated span shorts the water-column integrals too (measured
        # 0.5-0.7x on a half-column tabulation), so the check runs whenever
        # the caller names the seafloor, bottom term or not.
        if bottom is None and seafloor_depth is not None:
            D = float(seafloor_depth)
            z_last = float(self.depths[-1])
            if (not np.isfinite(D) or D <= 0.0
                    or not np.isclose(z_last, D, rtol=1e-3)):
                raise ConfigurationError(
                    f"Modes.with_attenuation: the mode tabulation ends at "
                    f"{z_last:g} m but seafloor_depth={seafloor_depth!r}; "
                    f"the water-column attenuation integral needs the full "
                    f"column (a half-column tabulation measured 0.5-0.7x "
                    f"the true value)."
                )
        if bottom is not None:
            from uacpy.core.environment import BoundaryProperties as _BP
            if not isinstance(bottom, _BP):
                raise ConfigurationError(
                    "Modes.with_attenuation: bottom must be a "
                    f"BoundaryProperties; got {type(bottom).__name__}"
                )
            # The closed-form tail below reads psi at depths[-1] as psi(D);
            # a tabulation stopping above the seafloor mis-evaluates it (a
            # 10-80 m grid in a 100 m guide measured 2-4x too much bottom
            # loss), so check the span when the caller can name D and say
            # which depth was used when they cannot.
            z_last = float(self.depths[-1])
            if seafloor_depth is not None:
                D = float(seafloor_depth)
                if not np.isfinite(D) or D <= 0.0:
                    raise ConfigurationError(
                        f"Modes.with_attenuation: seafloor_depth must be a "
                        f"positive finite depth in metres; got "
                        f"{seafloor_depth!r}.")
                # Symmetric check: a tabulation stopping SHORT reads psi(D)
                # in the water column (bottom term inflated up to 4x); one
                # running PAST D reads it down the evanescent tail (term
                # collapses to as little as 0.03x) — both silently wrong.
                if not np.isclose(z_last, D, rtol=1e-3):
                    raise ConfigurationError(
                        f"Modes.with_attenuation: the mode tabulation ends at "
                        f"{z_last:g} m but the seafloor is at {D:g} m, so "
                        f"psi(D) would be read {abs(D - z_last):g} m "
                        f"{'above' if z_last < D else 'below'} the seabed "
                        f"and the bottom term would be wrong by up to "
                        f"several ×.",
                        remediation="Tabulate the modes down to exactly the "
                                    "seafloor (e.g. include a receiver at D), "
                                    "or drop bottom= to skip the tail term.",
                    )
            else:
                warnings.warn(
                    f"Modes.with_attenuation: bottom term evaluated with "
                    f"psi(D) at the deepest tabulated depth {z_last:g} m; "
                    f"pass seafloor_depth= to check the tabulation reaches "
                    f"the seabed.",
                    UserWarning, stacklevel=2,
                )
            cb = float(bottom.sound_speed)
            rho_b = float(bottom.density) * 1000.0
            # dB/wavelength -> nepers/m: alpha[dB/lam] * (f/c) [lam/m] / (20/ln10).
            ab_neper_per_m = (
                float(bottom.attenuation) * np.log(10.0) / 20.0
                * float(self.f0) / cb
            )
            # Same form as the water integral above, but the half-space tail
            # has a closed form: the mode decays as psi(D)*exp(-gamma*(z-D)),
            # so int_D^inf psi^2 dz = psi(D)^2 / (2*gamma) — which is where the
            # 1/(2*gamma) comes from. gamma is real only for a trapped mode
            # (kr > kb); a leaky one clamps to 0 and contributes nothing.
            psi_D = phi_re[-1, :]
            kb = omega / cb
            gamma_m = np.sqrt(np.maximum(kr ** 2 - kb ** 2, 0.0))
            trapped = gamma_m > 0
            # A leaky mode gets no term, as the comment above already says.
            # Substituting gamma = 1 put a bare number carrying units of 1/m
            # into the denominator, so the invented loss scaled with the
            # library's length unit — the same physics expressed in km came
            # back 1000x different. There is no finite first-order bottom-
            # absorption perturbation for a radiating mode: the tail integral
            # diverges, so the closed form this line specialises does not exist.
            alpha_bottom = np.where(
                trapped,
                psi_D ** 2 * ab_neper_per_m * omega
                / (2.0 * kr_safe * np.where(trapped, gamma_m, 1.0) * cb * rho_b),
                0.0,
            )
            if not trapped.all():
                warnings.warn(
                    f"Modes.with_attenuation: {int((~trapped).sum())} of "
                    f"{gamma_m.size} mode(s) are leaky (kr <= omega/"
                    f"bottom.sound_speed = {kb:g} 1/m) and get no bottom term: "
                    "the evanescent-tail integral diverges for a radiating "
                    "mode, so first-order perturbation theory does not apply. "
                    "Their dominant loss is radiation into the half-space, "
                    "which backend='krakenc' already carries in Im(k) and this "
                    "method replaces.",
                    UserWarning, stacklevel=2,
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

        The textbook prefactor ``i·exp(−iπ/4)/(ρ_s·√(8πr))`` is written for
        ``e^{−iωt}`` with outgoing ``e^{+i k r}``; conjugating it into AT's
        ``e^{i(ωt − k r)}`` and folding in TL's free-field 1 m reference
        ``1/(4π)`` gives the ``exp(−iπ/4)·√(2π/r)/ρ_s`` used here
        (``4π/√(8π) = √(2π)``). That ``√(2π)`` is the magnitude of AT's own
        modal-evaluator prefactor ``i·√(2π)·exp(iπ/4)``
        (``KrakenField/EvaluateMod.f90:34``), so ``|P|`` — and therefore TL
        — lands on the same absolute scale as ``field.exe``; the two differ
        by an overall ``−1``, which neither ``−20·log10|P|`` nor a phase
        *difference* across the grid can see.

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

        # ``kraken.f90:573,598`` tabulates the modes on ``zTab`` — the merged
        # source/receiver depth vector the deck asked for — so ``self.depths``
        # is exactly where phi is known. Outside it there is no mode shape to
        # interpolate: ``np.interp`` would hold the end value flat, which is
        # neither the shape nor the evanescent tail the half-space carries, and
        # would report a plausible number for a depth this mode set never
        # covered. (AT is no better off the end — ``calculateweights.f90:43-49``
        # stops its bracket search at ``L < Nx-1`` and extrapolates linearly —
        # and neither code carries the half-space wavenumber needed for the
        # true ``exp(-gamma_m (z-D))`` tail.) Follow uacpy's depth policy: a
        # receiver below the resolvable domain is accepted as no-data, a
        # source there is fatal because it defines the field.
        z_lo, z_hi = float(self.depths[0]), float(self.depths[-1])
        if not z_lo - 1e-9 <= z_s <= z_hi + 1e-9:
            raise ConfigurationError(
                f"Modes.modal_propagation_loss: source_depth={z_s:g} m is "
                f"outside the tabulated mode depths [{z_lo:g}, {z_hi:g}] m, so "
                f"the excitation phi_m(z_s) is unknown.",
                remediation="Recompute the modes with this source depth in "
                            "the Source, or move the source into the "
                            "tabulated span.",
            )
        outside = (z_r < z_lo - 1e-9) | (z_r > z_hi + 1e-9)
        if np.any(outside):
            warnings.warn(
                f"Modes.modal_propagation_loss: {int(outside.sum())} of "
                f"{z_r.size} receiver depths fall outside the tabulated mode "
                f"depths [{z_lo:g}, {z_hi:g}] m and are returned as NaN; the "
                f"mode shapes are not defined there.",
                UserWarning, stacklevel=2,
            )
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
        # AT propagates as e^(i(omega t - k r)) (EvaluateMod.f90), so the
        # range factor here is e^(-i k r) and a decaying mode needs Im(k) <= 0.
        # Raw Kraken eigenvalues encode decay as k.imag < 0 while
        # with_attenuation builds k.imag > 0; a passive medium can only
        # attenuate, so force the sign and accept either input.
        k = k.real - 1j * np.abs(k.imag)
        # Complex sqrt — preserves the -arg(k)/2 phase contribution that
        # matters for phase-sensitive consumers (MFP, coherent integration).
        # Numpy's sqrt picks the principal branch (positive real part).
        inv_sqrt_k = 1.0 / np.sqrt(k.astype(np.complex128))
        weights = phi_zs * inv_sqrt_k
        expikr = np.exp(-1j * k[:, None] * r[None, :])
        contribution = (phi_zr * weights)[:, :, None] * expikr[None, :, :]
        P = contribution.sum(axis=1)
        # The asymptotic modal sum is a far-field form, singular at r = 0;
        # clamping sqrt(r) to 1 there returns a finite number that is NOT
        # the field at the source range - treat r = 0 samples as
        # placeholders, not physics.
        with np.errstate(divide='ignore', invalid='ignore'):
            sqrt_r = np.sqrt(r)
            sqrt_r = np.where(sqrt_r > 0, sqrt_r, 1.0)
        # KRAKEN normalises its modes with rho in g/cm³ (the .env unit), so the
        # density enters here in g/cm³ too.
        rho_s = float(source_density)
        # The textbook prefactor i*e^(-i*pi/4)/(rho*sqrt(8*pi*r)), conjugated
        # into AT's e^(i(omega t - k r)) convention and carrying TL's
        # free-field 1 m reference 1/(4*pi) (4*pi/sqrt(8*pi) == sqrt(2*pi)).
        pref = -1j * np.exp(1j * np.pi / 4.0) * np.sqrt(2.0 * np.pi) / rho_s
        P = pref * P / sqrt_r[None, :]
        P[outside, :] = np.nan
        id_kwargs = self.id_kwargs()
        id_kwargs['backend'] = 'modal_sum'
        id_kwargs['source_depths'] = np.array([z_s])
        return Field(
            data=P,
            coords={'depth': z_r, 'range': r},
            **id_kwargs,
        )
