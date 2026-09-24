"""Modal acoustics on plain arrays.

What a set of modes gives you once you have ``k`` and ``psi`` — from
KRAKEN, from another solver, from a file, from an analytic waveguide you
wrote down yourself. :class:`~uacpy.Modes` wraps each of these; none of
them needs a :class:`~uacpy.Modes` to be useful.
"""

import warnings
import numpy as np

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.bottom import (BoundaryProperties,
                               _NON_GEOACOUSTIC_TYPES)

#: Depth samples wanted per vertical wavelength for the perturbation
#: integrals below to be carried by the tabulation.
SAMPLES_PER_VERTICAL_WAVELENGTH = 8.0


def _on_depths(values, name, depths, who):
    """``values`` as one entry per tabulated depth: a scalar is spread
    over the tabulation, an array must already match it."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 1:
        return np.full_like(depths, float(arr.item()))
    if arr.shape != depths.shape:
        raise ConfigurationError(
            f"{who}: {name} shape {arr.shape} must match "
            f"depths {depths.shape} (or be a scalar).")
    return arr


def _warn_if_depth_axis_underresolves(
    a_neper: np.ndarray,
    kr: np.ndarray,
    omega: float,
    c_arr: np.ndarray,
    depths: np.ndarray,
    who: str = "modal_attenuation",
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
    if depths.size < 2:
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
    wanted = lambda_z / SAMPLES_PER_VERTICAL_WAVELENGTH
    spacing = float(np.max(np.diff(depths)))
    if spacing <= wanted:
        return
    warnings.warn(
        f"{who}: alpha(z)/c(z) varies with depth, but "
        f"the mode tabulation samples every {spacing:g} m — the shortest "
        f"vertical wavelength in this set is {lambda_z:g} m, so psi² is "
        f"carried by {lambda_z / spacing:.1f} samples per wavelength "
        f"where {SAMPLES_PER_VERTICAL_WAVELENGTH:g} are wanted "
        f"({wanted:g} m spacing). The numerator and denominator "
        f"trapezoids no longer cancel at this spacing: a step alpha(z) "
        f"measured 6-19 % off at 9 depths and up to 104 % at 5. Tabulate "
        f"the modes on a finer depth grid (Kraken takes it from the "
        f"Receiver depths) before reading Im(k).",
        UserWarning, skip_file_prefixes=USER_FRAME_SKIP)


def modal_attenuation(
    k,
    psi,
    depths,
    alpha_dB_per_m,
    *,
    frequency,
    sound_speed_z=1500.0,
    density_z=1.0,
    bottom=None,
    seafloor_depth=None,
    who: str = "modal_attenuation",
):
    """First-order modal attenuation ``alpha_m`` (Np/m), one per mode.

    For mode ``m`` with horizontal wavenumber ``k_rm`` and depth
    eigenfunction ``psi_m``, perturbation theory gives

    ``alpha_m = (omega / k_rm) * integral( alpha(z)/(c(z) rho(z)) *
    Re(psi_m)^2 dz ) / integral( Re(psi_m)^2 / rho(z) dz )``

    (Jensen, Kuperman, Porter & Schmidt, *Computational Ocean Acoustics*,
    Eq. 5.169), plus the closed-form evanescent-tail term
    ``psi(D)^2 / (2 gamma_m rho_b)`` when a penetrable ``bottom`` is given.

    Parameters
    ----------
    k : array_like
        Complex horizontal wavenumbers, one per mode (1/m).
    psi : array_like
        Mode shapes on ``depths``, shaped ``(n_depths, n_modes)``.
    depths : array_like
        Tabulation depths (m), ascending.
    alpha_dB_per_m : float or array_like
        Volume attenuation, a scalar or one value per depth.
    frequency : float
        Frequency (Hz). Required — the perturbation is ``omega``-scaled.
    sound_speed_z, density_z : float or array_like
        Sound speed (m/s) and density (g/cm3), scalar or per depth.
    bottom, seafloor_depth
        A penetrable half-space and where it starts; see the warnings the
        function raises when the tabulation and the seabed disagree.
    who : str, optional
        Name to put in the refusals and warnings.

    Returns
    -------
    ndarray
        ``alpha_m`` per mode (Np/m). ``nan`` for a mode the perturbation
        has no answer for — a non-positive ``Re k`` or a non-positive
        shape normalisation — rather than a level that would then
        propagate undamped through every field built from the set.
    """
    omega = 2.0 * np.pi * float(frequency or 0.0)
    if omega == 0.0:
        raise ConfigurationError(
            f"{who}: requires a frequency in Hz; got {frequency!r}."
        )
    # Each depth-tabulated input is a scalar (spread over the
    # tabulation) or one value per tabulated depth.
    a = _on_depths(alpha_dB_per_m, 'alpha', depths, who)
    c_arr = _on_depths(sound_speed_z, 'sound_speed_z', depths, who)
    rho_g = _on_depths(density_z, 'density_z', depths, who)
    rho_arr = rho_g * 1000.0  # g/cm³ → kg/m³
    a_neper = a * (np.log(10.0) / 20.0)
    # Perturbation on Re(psi)**2: for krakenc's complex modes this is
    # an approximation (JKPS 5.176 wants the complex psi**2); the
    # imaginary part of a weakly-attenuated mode shape is O(alpha),
    # so the error is second-order in the loss being computed.
    phi_re = np.asarray(psi).real
    weight = phi_re ** 2
    # JKPS 5.169 normalises int_0^D psi²/rho dz = 1 for the ideal
    # waveguide; the text after 5.176 extends the interval into the
    # bottom for a penetrable seabed, giving 0->inf. This trapezoid
    # over the tabulated water column is only the first half of it; the
    # half-space evanescent tail is added below, inside the `bottom`
    # block, which is the only place gamma_m and rho_b exist.
    norm = np.trapezoid(weight / rho_arr[:, None], depths, axis=0)
    integrand = (a_neper / (c_arr * rho_arr))[:, None] * weight
    kr = np.real(k)
    unsolvable = kr <= 0
    if np.any(unsolvable):
        warnings.warn(
            f"{who}: {int(np.count_nonzero(unsolvable))} "
            f"mode(s) have Re(k) <= 0, which the perturbation divides by; "
            f"their attenuation is returned as NaN. A number here would "
            f"sit among the valid modes' and propagate as physics.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
    # Clamped only to keep the array arithmetic finite; every entry it
    # covers is marked no-data before the result is returned.
    kr_safe = np.where(kr > 0, kr, 1.0)
    _warn_if_depth_axis_underresolves(
        a_neper, kr, omega, c_arr, depths, who)
    water_term = (omega / kr_safe) * np.trapezoid(integrand, depths, axis=0)
    bottom_term = np.zeros_like(water_term)
    # A truncated span shorts the water-column integrals too (measured
    # 0.5-0.7x on a half-column tabulation), so the check runs whenever
    # the caller names the seafloor, bottom term or not.
    if bottom is None and seafloor_depth is not None:
        D = float(seafloor_depth)
        z_last = float(depths[-1])
        if (not np.isfinite(D) or D <= 0.0
                or not np.isclose(z_last, D, rtol=1e-3)):
            raise ConfigurationError(
                f"{who}: the mode tabulation ends at "
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
                f"{who}: the mode shapes are non-zero "
                f"at the deepest tabulated depth "
                f"({float(depths[-1]):g} m). For a penetrable "
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
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
    # A vacuum / rigid / file / precalc boundary carries no seabed
    # geoacoustics: its cp, rho and attenuation are the placeholders
    # __post_init__ resolved (1600 m/s, 1.5 g/cm3, 0.5 dB/lambda), and
    # reading them as a half-space fabricates an attenuation the seabed
    # does
    # not have. `Bottom.all_sound_speeds` states the same rule for the same
    # data. Measured on an ideal 100 m guide at 50 Hz with a RIGID seabed,
    # passing bottom=BoundaryProperties('rigid') added 0.40 / 0.51 dB of
    # loss at 1 km and 7.99 / 10.22 dB at 20 km on modes 1 / 2, and warned
    # that two modes were "leaky" against a 1600 m/s half-space that does
    # not exist. Runs BEFORE the bottom block so a dropped boundary takes
    # the water-only path rather than falling into it with bottom = None.
    if bottom is not None:
        if not isinstance(bottom, BoundaryProperties):
            raise ConfigurationError(
                f"{who}: bottom must be a "
                f"BoundaryProperties; got {type(bottom).__name__}"
            )
        _btype = str(getattr(bottom, 'acoustic_type', '')).lower()
        if _btype in _NON_GEOACOUSTIC_TYPES:
            if _btype in ('vacuum', 'rigid'):
                # No energy enters the bottom, so the first-order
                # bottom-attenuation term is exactly zero and the
                # water-column integral alone is the right answer.
                bottom = None
            else:
                raise ConfigurationError(
                    f"{who}: a {_btype!r} seabed carries "
                    f"its loss in a reflection-coefficient table, not in "
                    f"cp/rho/attenuation — those are placeholders, so "
                    f"no first-order bottom-attenuation term can be "
                    f"formed "
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
        z_last = float(depths[-1])
        if seafloor_depth is not None:
            D = float(seafloor_depth)
            if not np.isfinite(D) or D <= 0.0:
                raise ConfigurationError(
                    f"{who}: seafloor_depth must be a "
                    f"positive finite depth in metres; got "
                    f"{seafloor_depth!r}.")
            # Symmetric check: a tabulation stopping SHORT reads psi(D)
            # in the water column (bottom term inflated up to 4x); one
            # running PAST D reads it down the evanescent tail (term
            # collapses to as little as 0.03x) — both silently wrong.
            if not np.isclose(z_last, D, rtol=1e-3):
                raise ConfigurationError(
                    f"{who}: the mode tabulation ends at "
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
                f"{who}: bottom term evaluated with "
                f"psi(D) at the deepest tabulated depth {z_last:g} m; "
                f"pass seafloor_depth= to check the tabulation reaches "
                f"the seabed.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
        cb = float(bottom.sound_speed)
        rho_b = float(bottom.density) * 1000.0
        # dB/wavelength -> nepers/m: alpha[dB/lam] * (f/c) [lam/m] / (20/ln10).
        ab_neper_per_m = (
            float(bottom.attenuation) * np.log(10.0) / 20.0
            * float(frequency) / cb
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
        # attenuation perturbation for a radiating mode: the tail integral
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
                f"{who}: {int(barely_trapped.sum())} of "
                f"{gamma_m.size} mode(s) are barely trapped — their "
                f"evanescent tail into the seabed carries more of the "
                f"∫psi²/rho dz normalisation than the whole water column "
                f"does ({100.0 * tail_norm[j] / norm[j]:.0f} % of it for "
                f"mode {j + 1}, whose gamma = sqrt(kr²-kb²) = "
                f"{gamma_m[j]:.3g} 1/m spreads the tail over "
                f"{0.5 / gamma_m[j]:.4g} m of seabed against a "
                f"{float(depths[-1]):g} m water column). The bottom "
                f"term is then set by the part of kr the mode solver "
                f"resolves least well, and near a modal cutoff the mesh "
                f"decides which side of kb = {kb:g} 1/m the mode lands "
                f"on: on a 100 m Pekeris guide a converged mesh moved the "
                f"total modal TL by 1.7 dB at 200 m and 7 dB at 10 km, "
                f"and put the same mode on the leaky side, where it gets "
                f"no bottom term at all. "
                f"Re-run with a finer mesh (Kraken(n_mesh=...)) and "
                f"compare before reading a level off this frequency.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
        if not trapped.all():
            warnings.warn(
                f"{who}: {int((~trapped).sum())} of "
                f"{gamma_m.size} mode(s) are leaky (kr <= omega/"
                f"bottom.sound_speed = {kb:g} 1/m) and get no bottom term: "
                "the evanescent-tail integral diverges for a radiating "
                "mode, so first-order perturbation theory does not apply. "
                "Their dominant loss is radiation into the half-space, "
                "which backend='krakenc' already carries in Im(k) and this "
                "method replaces.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
    if np.any(norm <= 0):
        warnings.warn(
            f"{who}: {int(np.count_nonzero(norm <= 0))} "
            f"mode(s) have a non-positive shape normalisation "
            f"∫psi²/rho dz, which the perturbation divides by; their "
            f"attenuation is returned as NaN. Clamping it produced a "
            f"LOSSLESS mode, which a range-marched sum propagates "
            f"undamped and which can dominate the far field.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
    unsolvable = unsolvable | (norm <= 0)
    norm = np.where(norm > 0, norm, 1.0)
    alpha_m = (water_term + bottom_term) / norm
    # The perturbation had no answer for these modes: report no data
    # rather than a level that participates in every field built from
    # this mode set.
    alpha_m = np.where(unsolvable, np.nan, alpha_m)
    return alpha_m


def modal_field(k_modes, psi_zs, psi_zr, ranges_m, *, source_density=1.0):
    """Coherent complex pressure from the asymptotic modal sum.

    .. math::
        p(z_r, r) \\approx \\frac{e^{-i\\pi/4}}{\\rho_s}
        \\sqrt{\\frac{2\\pi}{r}}\\sum_m
        \\psi_m(z_s)\\,\\psi_m(z_r)\\,
        \\frac{e^{-ik_m r}}{\\sqrt{k_m}}

    the far-field form of the normal-mode solution (Jensen, Kuperman,
    Porter & Schmidt, *Computational Ocean Acoustics*, sect. 5.3). Give it
    the mode shapes **already evaluated at the source and receiver
    depths**: choosing how to get there from a tabulation — interpolating,
    refusing to extrapolate, masking what lies outside — is the caller's
    question, and :meth:`~uacpy.Modes.modal_propagation_loss` answers it
    one way.

    Parameters
    ----------
    k_modes : array_like
        Complex horizontal wavenumbers, ``(n_modes,)``. Either sign of
        ``Im k`` is accepted: a passive medium can only attenuate, so the
        sign is forced.
    psi_zs : array_like
        Mode shapes at the source depth, ``(n_modes,)``.
    psi_zr : array_like
        Mode shapes at each receiver depth, ``(n_depths, n_modes)``.
    ranges_m : array_like
        Ranges (m). ``r <= 0`` is outside the form's domain and comes back
        ``nan``, not a number within a few dB of the 1 m answer.
    source_density : float, default 1.0
        Density at the source (**g/cm3** — the unit KRAKEN normalises its
        modes with, and the unit the ``.env`` file carries).

    Returns
    -------
    ndarray
        Complex pressure, ``(n_depths, n_ranges)``.
    """
    r = np.atleast_1d(np.asarray(ranges_m, dtype=float))
    psi_zs = np.asarray(psi_zs)
    psi_zr = np.asarray(psi_zr)
    k = np.asarray(k_modes)
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
    weights = psi_zs * inv_sqrt_k
    expikr = np.exp(-1j * k[:, None] * r[None, :])
    # Contract the mode axis directly. Forming the (depth, mode, range)
    # product first and summing it afterwards asks for one complex128
    # temporary of n_depth·n_mode·n_range — 1.28 GB at 200 depths, 200
    # modes and 2000 ranges — for a result of n_depth·n_range.
    P = np.einsum('zm,mr->zr', psi_zr * weights, expikr, optimize=True)
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
    # r <= 0 is outside the form's domain, not a quiet spot in it — the
    # same no-data marking every other range-zero path in the package
    # applies.
    P[:, r <= 0.0] = np.nan
    return P
