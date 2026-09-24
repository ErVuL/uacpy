"""Normal-mode result type."""

from __future__ import annotations

import warnings

import numpy as np
from typing import Optional, Union

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.acoustics.modal import (
    SAMPLES_PER_VERTICAL_WAVELENGTH, modal_attenuation, modal_field)

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

        Use :meth:`compute_group_velocity` instead when this is ``None``, or
        when every mode is needed from a large set. The perturbation caveat
        the source states at ``kraken.f90:772`` applies either way: "group
        speeds will be wrong for leaky modes".
    """
    field_type = "modes"

    def __init__(
        self,
        *,
        k: np.ndarray,
        phi: np.ndarray,
        depths: np.ndarray,
        group_velocity: Optional[np.ndarray] = None,
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
            group_velocity=(None if self.group_velocity is None
                            else self.group_velocity[:n]),
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
        from uacpy.acoustic_signal.system import modal_group_velocity
        return modal_group_velocity(
            np.array([min(f0_self, f0_other), max(f0_self, f0_other)]),
            np.stack([np.asarray(lo.k)[:n], np.asarray(hi.k)[:n]]))[0]

    # Samples per vertical wavelength :meth:`with_attenuation` wants before it
    # trusts its trapezoids. psi² oscillates at twice the mode's vertical
    # wavenumber, so 8 per wavelength is 4 per period of the integrand.
    # Measured on a Pekeris set (D = 100 m, f = 50 Hz, 6 modes) against a
    # 8001-point reference, with alpha(z) stepping 0 → 2 dB/m at 50 m: 1-3 %
    # error at 51 depths (dz = 2 m, every mode over the bar), 3-7 % at 21,
    # 6-19 % at 9, and up to 104 % at 5.
    #: Alias of the module constant the perturbation reads.
    _SAMPLES_PER_VERTICAL_WAVELENGTH = SAMPLES_PER_VERTICAL_WAVELENGTH

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
        depths = np.atleast_1d(np.asarray(source.depths, dtype=float))
        weights = np.atleast_1d(np.asarray(source.weights,
                                           dtype=np.complex128))
        lo, hi = float(self.depths.min()), float(self.depths.max())
        outside = depths[(depths < lo) | (depths > hi)]
        if outside.size:
            raise ConfigurationError(
                f"Modes.excitation: source depth(s) "
                f"{np.array2string(outside, precision=2)} m lie outside the "
                f"tabulated mode depths ({lo:g}-{hi:g} m), so their mode "
                f"shapes would be extrapolated.",
                remediation="Compute the modes on a grid spanning the "
                            "source depths (Kraken.compute_modes uses a "
                            "dense grid over the whole waveguide).",
            )
        phi = np.asarray(self.phi)
        # Interpolate each mode onto the source depths; complex shapes
        # (krakenc) interpolate in real and imaginary parts.
        at_source = np.empty((depths.size, phi.shape[1]),
                             dtype=np.complex128)
        for m in range(phi.shape[1]):
            column = phi[:, m]
            at_source[:, m] = np.interp(depths, self.depths, column.real)
            if np.iscomplexobj(column):
                at_source[:, m] += 1j * np.interp(depths, self.depths,
                                                  column.imag)
        excited = weights @ at_source
        if getattr(source, 'beam_pattern', None) is None:
            return excited
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
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
        weighted = excited * source.element_directivity(
            np.where(undefined, 0.0, angles))
        return np.where(undefined, np.nan, weighted)

    def grazing_angles(self, sound_speed: float) -> np.ndarray:
        """Each mode's grazing angle in degrees against ``sound_speed``.

        ``theta_m = arccos(c / v_m)``, the dictionary between the modal and
        ray pictures, and the one home for that formula — every caller asks
        here rather than spelling it again. A mode whose phase speed is below
        ``c`` is evanescent there and has no real angle, so it comes back
        ``nan``; clipping the ratio to 1.0 instead would stack those modes on
        broadside, where a patterned source applies its notch to them.
        """
        c = float(sound_speed)
        if not c > 0.0:
            raise ConfigurationError(
                f"Modes.grazing_angles: sound_speed must be > 0 m/s; got {c}")
        speeds = np.asarray(self.compute_phase_speeds(), dtype=float)
        # No clip. ``c / v_m`` exceeds 1 exactly for the evanescent modes, and
        # ``arccos`` of that is already nan — which is the right answer. The
        # old spelling clipped the ratio to 1.0 first, turning "no angle" into
        # "broadside" and handing a patterned source its notch.
        with np.errstate(invalid='ignore'):
            return np.degrees(np.arccos(c / speeds))

    def with_attenuation(
        self,
        alpha_dB_per_m: Union[float, np.ndarray],
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
        alpha_dB_per_m : float or ndarray
            Per-depth volume attenuation in dB/m, sampled on
            :attr:`depths`. Scalar broadcasts to every depth. Build one
            from an :class:`~uacpy.core.absorption.Absorption` via
            ``absorption.alpha_dB_per_m(modes.f0, modes.depths)``.

            **This is not the same number the solver used.** The Python
            accessor and the Acoustics-Toolbox deck evaluate the same model
            at different arguments, so a perturbation built here is applied
            on top of a run that already absorbed a slightly different
            alpha. Both routes are correct for what they are; the divergence
            is worth knowing before the two are differenced.

            For :class:`~uacpy.core.absorption.FrancoisGarrison`, the
            accessor evaluates the formula at each depth you pass, while the
            deck carries one module-level ``z_bar`` and applies the single
            resulting alpha at every depth (``misc/AttenMod.f90:148-160``).
            On ``FrancoisGarrison(10, 35, 8, z_bar_m=1000)`` the accessor at
            the surface runs +3.0 % over the deck at 1 kHz, +13.9 % at
            10 kHz, +15.5 % at 30 kHz and +15.0 % at 100 kHz, crossing zero
            at ``z_bar_m`` and running as far the other way at the bottom of
            a 2 km column.

            For :class:`~uacpy.core.absorption.ConstantAbsorption`, the
            accessor converts dB/wavelength at
            :data:`~uacpy.core.constants.DEFAULT_SOUND_SPEED`, while the
            deck converts at each SSP row's own ``c`` (``AttenMod.f90:73``),
            a spread of ±3.3 % across sound speeds of 1450-1550 m/s.

            To make the two agree, pass a scalar alpha you computed at the
            same argument the deck used: ``z_bar_m`` for Francois-Garrison,
            the SSP's own sound speed for a constant absorption.

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
        synthesis (:meth:`modal_propagation_loss`). It is **not**
        consumed by Acoustics-Toolbox ``field.exe`` / ``fieldS.exe``:
        uacpy has no ``.mod`` writer, and ``field.exe`` reads
        attenuation natively from the environment passed to its field
        run rather than from ``Im(k)`` of the ``.mod`` file. To get the
        perturbed TL from ``field.exe``, attach an :class:`Absorption`
        to the :class:`Environment` and run :class:`Kraken`.

    On plain ``k`` / ``psi`` arrays this is
    :func:`~uacpy.core.acoustics.modal_attenuation`.
        """
        # The whole perturbation — the integrals, the seabed tail, the
        # trapped/leaky split and its three warnings — is
        # acoustics.modal_attenuation's. What this method adds is reading
        # its own k/psi/depths/f0 and re-wrapping the result as a Modes.
        alpha_m = modal_attenuation(
            self.k, self.phi, self.depths, alpha_dB_per_m,
            frequency=self.f0, sound_speed_z=sound_speed_z,
            density_z=density_z, bottom=bottom,
            seafloor_depth=seafloor_depth,
            who="Modes.with_attenuation")
        kr = np.real(self.k)
        new_k = kr + 1j * alpha_m
        return Modes(
            k=new_k, phi=self.phi, depths=self.depths,
            # The solver's group speed survives: this perturbation moves the
            # IMAGINARY part of k, and dropping it here while first_n keeps it
            # made the attribute disappear depending on which method was called.
            group_velocity=self.group_velocity,
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

    On plain arrays the sum itself is
    :func:`~uacpy.core.acoustics.modal_field`, which takes the mode
    shapes already evaluated at the source and receiver depths.
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
        # covered. (AT is no better off the end — ``misc/calculateweights.f90:43-49``
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
        # The sum itself — the Im(k) sign forcing, the complex sqrt, the
        # mode-axis contraction and the prefactor — is modal_field's. What
        # this method adds is the depth interpolation above, the
        # out-of-tabulation masking below and the Field re-wrap.
        P = modal_field(self.k, phi_zs, phi_zr, r,
                        source_density=source_density)
        P[outside, :] = np.nan
        id_kwargs = self.id_kwargs()
        id_kwargs['backend'] = 'modal_sum'
        id_kwargs['source_depths'] = np.array([z_s])
        return Field(
            data=P,
            coords={'depth': z_r, 'range': r},
            **id_kwargs,
        )
