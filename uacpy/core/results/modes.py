"""Normal-mode result type."""

from __future__ import annotations

import warnings

import numpy as np
from typing import Optional, Union

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._finite_difference import warn_if_storage_under_resolves

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

        which moves with frequency and mode order: on a 100 m Pekeris guide
        (c 1500/1800 m/s, ρ 1.0/1.8) at 100 Hz the optimum is Δf = 1 Hz, at
        7.3e-06 relative against exact roots; a decade below it the answer is
        6x worse, two decades below 85x worse, and at Δf = 1e-4 Hz the seven
        modes collapse onto four distinct speeds. Upcasting ``k`` recovers
        none of this — the bits were never written — so a step whose storage
        floor exceeds 1e-5 is warned about instead.

        The warning does **not** prescribe a step. ``|d²k_r/dω²|`` is not
        visible from two frequencies, and a step chosen from the floor alone
        made the answer worse in 14 of 30 firings on five ideal waveguides, by
        up to 16x (measured). It gives the test instead: recompute at twice
        the separation and keep the wider answer only if v_g stays inside the
        floor. :func:`uacpy.acoustic_signal.modal_group_velocity`, which is
        handed a whole sweep, runs that test itself and names the step.
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
            v_g = np.where(dk != 0, domega / dk, np.nan)
        warn_if_storage_under_resolves(
            np.real(self.k[:n]), dk, v_g, "Modes.compute_group_velocity")
        return v_g

    # Samples per vertical wavelength :meth:`with_attenuation` wants before it
    # trusts its trapezoids. psi² oscillates at twice the mode's vertical
    # wavenumber, so 8 per wavelength is 4 per period of the integrand.
    # Measured on a Pekeris set (D = 100 m, f = 50 Hz, 6 modes) against a
    # 8001-point reference, with alpha(z) stepping 0 → 2 dB/m at 50 m: 1-3 %
    # error at 51 depths (dz = 2 m, every mode over the bar), 3-7 % at 21,
    # 6-19 % at 9, and up to 104 % at 5.
    _SAMPLES_PER_VERTICAL_WAVELENGTH = 8.0

    def _warn_if_depth_axis_underresolves(
        self,
        a_neper: np.ndarray,
        kr: np.ndarray,
        omega: float,
        c_arr: np.ndarray,
    ) -> None:
        """Warn when :attr:`depths` is too coarse to carry
        :meth:`with_attenuation`'s integrals.

        The perturbation is a ratio of two trapezoid sums over the same axis,
        so their quadrature errors cancel *exactly* while ``alpha(z)/c(z)`` is
        constant — a scalar alpha comes out right on any grid, which is why a
        coarse axis is invisible there — and stop cancelling the moment it
        varies with depth. What is left over is then set by how well the axis
        resolves ``psi²``, so the test compares the coarsest sample spacing
        against the shortest vertical wavelength ``2π/sqrt(k_water² - k_rm²)``
        in the set.

        ``a_neper`` is the attenuation in nepers/m on :attr:`depths`, ``kr``
        the real horizontal wavenumbers, ``omega`` the angular frequency and
        ``c_arr`` the sound speed on the same axis.

        Structure in ``alpha(z)`` *below* the sample spacing (a step, or a
        layer thinner than one cell) is aliased in the values handed in and
        cannot be seen from them: a profile whose samples happen to land on
        one phase reads as uniform here however wrong the integral is.
        """
        if self.depths.size < 2:
            return
        c_min = float(np.min(c_arr))
        if not np.isfinite(c_min) or c_min <= 0.0:
            return
        ratio = a_neper / c_arr
        if np.allclose(ratio, ratio.flat[0], rtol=1e-12, atol=0.0):
            return
        kz_sq = (omega / c_min) ** 2 - np.real(kr) ** 2
        if not np.any(kz_sq > 0.0):
            return
        lambda_z = 2.0 * np.pi / float(np.sqrt(np.max(kz_sq)))
        wanted = lambda_z / self._SAMPLES_PER_VERTICAL_WAVELENGTH
        spacing = float(np.max(np.diff(self.depths)))
        if spacing <= wanted:
            return
        warnings.warn(
            f"Modes.with_attenuation: alpha(z)/c(z) varies with depth, but "
            f"the mode tabulation samples every {spacing:g} m — the shortest "
            f"vertical wavelength in this set is {lambda_z:g} m, so psi² is "
            f"carried by {lambda_z / spacing:.1f} samples per wavelength "
            f"where {self._SAMPLES_PER_VERTICAL_WAVELENGTH:g} are wanted "
            f"({wanted:g} m spacing). The numerator and denominator "
            f"trapezoids no longer cancel at this spacing: a step alpha(z) "
            f"measured 6-19 % off at 9 depths and up to 104 % at 5. Tabulate "
            f"the modes on a finer depth grid (Kraken takes it from the "
            f"Receiver depths) before reading Im(k).",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)

    def _on_depths(self, values, name: str) -> np.ndarray:
        """``values`` as one entry per tabulated depth: a scalar is spread
        over the tabulation, an array must already match it."""
        arr = np.asarray(values, dtype=float).ravel()
        if arr.size == 1:
            return np.full_like(self.depths, float(arr.item()))
        if arr.shape != self.depths.shape:
            raise ConfigurationError(
                f"Modes.with_attenuation: {name} shape {arr.shape} must match "
                f"depths {self.depths.shape} (or be a scalar).")
        return arr

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
        alpha_db_per_m : float or ndarray
            Per-depth volume attenuation in dB/m, sampled on
            :attr:`depths`. Scalar broadcasts to every depth. Build one
            from an :class:`~uacpy.core.absorption.Absorption` via
            ``absorption.alpha_db_per_m(modes.f0, modes.depths)``.

            A **depth-varying** alpha needs a depth axis fine enough to
            carry ``psi²``: the two trapezoids below cancel exactly while
            ``alpha(z)/c(z)`` is constant, so a scalar alpha is right on any
            grid, and a structured one is only as good as the sampling.
            :attr:`depths` is the merged source/receiver vector, which is
            often 8-20 points, so this is checked against the shortest
            vertical wavelength in the set and a coarse axis raises a
            ``UserWarning`` naming the measured and wanted spacing. It
            *warns* rather than raises: the error is continuous in the
            spacing (1-3 % at 51 depths, up to 104 % at 5 on a step alpha),
            the threshold is a calibration rather than a physical boundary,
            and the integrals still return the best value the given samples
            support — unlike the ``seafloor_depth`` mismatch below, which
            reads ``psi(D)`` at a depth that is simply not the seabed.
        sound_speed_z : float or ndarray
            ``c(z)`` in m/s. Defaults to 1500.
        density_z : float or ndarray
            ``ρ(z)`` in **g/cm³** (matches :class:`BoundaryProperties`).
            Defaults to 1.0 (fresh-water reference; 1.025 for seawater).
        bottom : BoundaryProperties, optional
            Half-space below the water column. When supplied, adds an
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
        # Each depth-tabulated input is a scalar (spread over the
        # tabulation) or one value per tabulated depth.
        a = self._on_depths(alpha_db_per_m, 'alpha')
        c_arr = self._on_depths(sound_speed_z, 'sound_speed_z')
        rho_g = self._on_depths(density_z, 'density_z')
        rho_arr = rho_g * 1000.0  # g/cm³ → kg/m³
        a_neper = a * (np.log(10.0) / 20.0)
        # Perturbation on Re(psi)**2: for krakenc's complex modes this is
        # an approximation (JKPS 5.176 wants the complex psi**2); the
        # imaginary part of a weakly-attenuated mode shape is O(alpha),
        # so the error is second-order in the loss being computed.
        phi_re = np.asarray(self.phi).real
        weight = phi_re ** 2
        # JKPS 5.169 normalises int_0^D psi²/rho dz = 1 for the ideal
        # waveguide; the text after 5.176 extends the interval into the
        # bottom for a penetrable seabed, giving 0->inf. This trapezoid
        # over the tabulated water column is only the first half of it; the
        # half-space evanescent tail is added below, inside the `bottom`
        # block, which is the only place gamma_m and rho_b exist.
        norm = np.trapezoid(weight / rho_arr[:, None], self.depths, axis=0)
        integrand = (a_neper / (c_arr * rho_arr))[:, None] * weight
        kr = np.real(self.k)
        unsolvable = kr <= 0
        if np.any(unsolvable):
            warnings.warn(
                f"Modes.with_attenuation: {int(np.count_nonzero(unsolvable))} "
                f"mode(s) have Re(k) <= 0, which the perturbation divides by; "
                f"their attenuation is returned as NaN. A number here would "
                f"sit among the valid modes' and propagate as physics.",
                UserWarning, stacklevel=2)
        # Clamped only to keep the array arithmetic finite; every entry it
        # covers is marked no-data before the result is returned.
        kr_safe = np.where(kr > 0, kr, 1.0)
        self._warn_if_depth_axis_underresolves(a_neper, kr, omega, c_arr)
        water_term = (omega / kr_safe) * np.trapezoid(integrand, self.depths, axis=0)
        bottom_term = np.zeros_like(water_term)
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
        if bottom is None:
            # The tail psi(D)²/(2 gamma_m rho_b) that carries the
            # normalisation from JKPS 5.169's 0->D out to the 0->inf the
            # text after 5.176 calls for needs a half-space gamma_m and rho_b, and
            # neither exists here, so `norm` stays a water-column integral
            # while the true denominator is larger. The omitted term is
            # positive, so alpha_m comes back high: an upper bound, measured
            # 0.7 % / 2.4 % / 5.1 % / 17 % over the exact perturbation
            # integral for modes 1-4 of an analytic 100 m Pekeris guide.
            # A pressure-release seabed zeroes psi(D): no tail, no error.
            # A rigid one zeroes psi'(D) with psi(D) at a maximum, and no
            # energy enters it either, so the water-column integral is
            # already exact there (the bottom= vacuum/rigid branch below
            # states the same rule). Modes carries no boundary metadata,
            # so psi(D)² is the only available trigger and a rigid-seabed
            # mode set fires this warning even though its value is exact.
            psi_end_sq = weight[-1, :]
            column_mean = np.mean(weight, axis=0)
            if np.any(psi_end_sq > 1e-6 * column_mean):
                warnings.warn(
                    f"Modes.with_attenuation: the mode shapes are non-zero "
                    f"at the deepest tabulated depth "
                    f"({float(self.depths[-1]):g} m). For a penetrable "
                    f"(lossy-boundary) mode set they continue as an "
                    f"evanescent tail into the seabed, but without bottom= "
                    f"there is no half-space gamma or density from which to "
                    f"form it: the ∫psi²/rho dz normalisation stops at the "
                    f"seabed while the text after JKPS Eq. 5.176 extends it "
                    f"into the bottom, and the returned Im(k) is an UPPER "
                    f"BOUND on the true attenuation for such sets (measured "
                    f"0.7-17 % high across the four modes of a 100 m "
                    f"Pekeris guide, worst for the near-cutoff mode that "
                    f"dominates at long range). A RIGID-seabed mode set "
                    f"(psi'(D)=0, psi(D) at a maximum) has no tail and its "
                    f"returned Im(k) is exact; pass "
                    f"bottom=BoundaryProperties(acoustic_type='rigid') to "
                    f"state that boundary and silence this warning.",
                    UserWarning, stacklevel=2)
        # A vacuum / rigid / file / precalc boundary carries no seabed
        # geoacoustics: its cp, rho and attenuation are the placeholders
        # __post_init__ resolved (1600 m/s, 1.5 g/cm3, 0.5 dB/lambda), and
        # reading them as a half-space fabricates an absorption the seabed does
        # not have. `Bottom.all_sound_speeds` states the same rule for the same
        # data. Measured on an ideal 100 m guide at 50 Hz with a RIGID seabed,
        # passing bottom=BoundaryProperties('rigid') added 0.40 / 0.51 dB of
        # loss at 1 km and 7.99 / 10.22 dB at 20 km on modes 1 / 2, and warned
        # that two modes were "leaky" against a 1600 m/s half-space that does
        # not exist. Runs BEFORE the bottom block so a dropped boundary takes
        # the water-only path rather than falling into it with bottom = None.
        if bottom is not None:
            from uacpy.core.environment import BoundaryProperties as _BP
            if not isinstance(bottom, _BP):
                raise ConfigurationError(
                    "Modes.with_attenuation: bottom must be a "
                    f"BoundaryProperties; got {type(bottom).__name__}"
                )
            from uacpy.core.bottom import _NON_GEOACOUSTIC_TYPES
            _btype = str(getattr(bottom, 'acoustic_type', '')).lower()
            if _btype in _NON_GEOACOUSTIC_TYPES:
                if _btype in ('vacuum', 'rigid'):
                    # No energy enters the bottom, so the first-order
                    # bottom-absorption term is exactly zero and the
                    # water-column integral alone is the right answer.
                    bottom = None
                else:
                    raise ConfigurationError(
                        f"Modes.with_attenuation: a {_btype!r} seabed carries "
                        f"its loss in a reflection-coefficient table, not in "
                        f"cp/rho/attenuation — those are placeholders, so "
                        f"no first-order bottom-absorption term can be formed "
                        f"from them.",
                        remediation="Drop bottom= to get the water-column "
                                    "term alone (reported as an upper bound), "
                                    "or pass a BoundaryProperties carrying "
                                    "real geoacoustics.",
                    )

        if bottom is not None:
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
            gamma_safe = np.where(trapped, gamma_m, 1.0)
            # The same int_D^inf psi(D)²exp(-2 gamma (z-D)) dz, without the
            # attenuation weight, is the piece of the 0->inf normalisation
            # (JKPS 5.169 extended per the text after 5.176) that lies
            # below the seabed, and adding it here is
            # what puts the denominator over the domain the numerator's
            # bottom term already spans. Stopping the denominator at the
            # seabed instead measured 0.7 % (mode 1) to 17 % (near-cutoff
            # mode 4) above the exact perturbation integral on an analytic
            # Pekeris guide. A leaky mode has no convergent tail on either
            # side, so its normalisation is the water column alone.
            tail_norm = np.where(
                trapped, psi_D ** 2 / (2.0 * gamma_safe * rho_b), 0.0)
            water_norm = norm
            norm = norm + tail_norm
            # A leaky mode gets no term, as the comment above already says.
            # Substituting gamma = 1 put a bare number carrying units of 1/m
            # into the denominator, so the invented loss scaled with the
            # library's length unit — the same physics expressed in km came
            # back 1000x different. There is no finite first-order bottom-
            # absorption perturbation for a radiating mode: the tail integral
            # diverges, so the closed form this line specialises does not exist.
            bottom_term = np.where(
                trapped,
                psi_D ** 2 * ab_neper_per_m * omega
                / (2.0 * kr_safe * gamma_safe * cb * rho_b),
                0.0,
            )
            # gamma at which the seabed tail carries exactly as much of the
            # normalisation as the whole water column — psi(D)²/(2 gamma rho_b)
            # == int psi²/rho dz — so the comparison below is a ratio of two
            # tabulated integrals and carries no absolute epsilon and no unit.
            # Below it the mode is bound so weakly that its bottom term rests
            # on the part of kr the mode solver resolves least well. Sweeping
            # a 100 m Pekeris guide across a modal cutoff with kraken.exe and
            # differencing the default mesh against a converged one: the
            # frequencies under this line moved the total modal TL by
            # 0.67-1.98 dB at 200-500 m (TL 43-52 dB there) and by 7.07 dB at
            # 10 km, and the smallest gamma of them changed side of the
            # branch point entirely — leaky on the default mesh, trapped on
            # the converged one. Just above the line the movement is 0.41 dB
            # and it falls under 0.04 dB within 0.7 Hz.
            gamma_equal = np.where(
                water_norm > 0,
                psi_D ** 2 / (2.0 * rho_b * np.where(water_norm > 0,
                                                     water_norm, 1.0)),
                0.0)
            barely_trapped = trapped & (gamma_m < gamma_equal)
            if barely_trapped.any():
                j = int(np.argmin(np.where(barely_trapped, gamma_m, np.inf)))
                warnings.warn(
                    f"Modes.with_attenuation: {int(barely_trapped.sum())} of "
                    f"{gamma_m.size} mode(s) are barely trapped — their "
                    f"evanescent tail into the seabed carries more of the "
                    f"∫psi²/rho dz normalisation than the whole water column "
                    f"does ({100.0 * tail_norm[j] / norm[j]:.0f} % of it for "
                    f"mode {j + 1}, whose gamma = sqrt(kr²-kb²) = "
                    f"{gamma_m[j]:.3g} 1/m spreads the tail over "
                    f"{0.5 / gamma_m[j]:.4g} m of seabed against a "
                    f"{float(self.depths[-1]):g} m water column). The bottom "
                    f"term is then set by the part of kr the mode solver "
                    f"resolves least well, and near a modal cutoff the mesh "
                    f"decides which side of kb = {kb:g} 1/m the mode lands "
                    f"on: on a 100 m Pekeris guide a converged mesh moved the "
                    f"total modal TL by 1.7 dB at 200 m and 7 dB at 10 km, "
                    f"and put the same mode on the leaky side, where it gets "
                    f"no bottom term at all. "
                    f"Re-run with a finer mesh (Kraken(n_mesh=...)) and "
                    f"compare before reading a level off this frequency.",
                    UserWarning, stacklevel=2,
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
        if np.any(norm <= 0):
            warnings.warn(
                f"Modes.with_attenuation: {int(np.count_nonzero(norm <= 0))} "
                f"mode(s) have a non-positive shape normalisation "
                f"∫psi²/rho dz, which the perturbation divides by; their "
                f"attenuation is returned as NaN. Clamping it produced a "
                f"LOSSLESS mode, which a range-marched sum propagates "
                f"undamped and which can dominate the far field.",
                UserWarning, stacklevel=2)
        unsolvable = unsolvable | (norm <= 0)
        norm = np.where(norm > 0, norm, 1.0)
        alpha_m = (water_term + bottom_term) / norm
        # The perturbation had no answer for these modes: report no data
        # rather than a level that participates in every field built from
        # this mode set.
        alpha_m = np.where(unsolvable, np.nan, alpha_m)
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
        # Contract the mode axis directly. Forming the (depth, mode, range)
        # product first and summing it afterwards asks for one complex128
        # temporary of n_depth·n_mode·n_range — 1.28 GB at 200 depths, 200
        # modes and 2000 ranges — for a result of n_depth·n_range.
        P = np.einsum('zm,mr->zr', phi_zr * weights, expikr, optimize=True)
        # The asymptotic modal sum is a far-field form, singular at r <= 0.
        # sqrt(r) is held at 1 there only to keep the division finite; those
        # columns are marked no-data below rather than returned, since the
        # number the substitution produces sits within a few dB of the 1 m
        # answer and would read as the field at the source.
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
        # r <= 0 is outside the form's domain, not a quiet spot in it — the
        # same no-data marking every other range-zero path in the package
        # applies.
        P[:, r <= 0.0] = np.nan
        id_kwargs = self.id_kwargs()
        id_kwargs['backend'] = 'modal_sum'
        id_kwargs['source_depths'] = np.array([z_s])
        return Field(
            data=P,
            coords={'depth': z_r, 'range': r},
            **id_kwargs,
        )
