"""Modal acoustics on plain arrays.

What a set of modes gives you once you have ``k`` and ``phi`` — from
KRAKEN, from another solver, from a file, from an analytic waveguide you
wrote down yourself. :class:`~uacpy.Modes` wraps each of these; none of
them needs a :class:`~uacpy.Modes` to be useful.
"""

import warnings
import numpy as np

from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, NumericsWarning, ValidityWarning,
)
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.boundary import BoundaryProperties, BoundaryType
from uacpy.core.bottom import Bottom
from uacpy.core.constants import (
    DEFAULT_WATER_DENSITY, NEPER_TO_DB,
)

__all__ = [
    'SAMPLES_PER_VERTICAL_WAVELENGTH', 'modal_attenuation', 'mode_shapes_at',
    'modal_field', 'modal_phase_speeds', 'modal_grazing_angles',
    'modal_excitation', 'medium_density_at',
]


def modal_phase_speeds(k, frequency: float) -> np.ndarray:
    """Mode phase speeds ``v_p = ω / Re(k)`` (m/s) of horizontal
    wavenumbers ``k`` (rad/m) at ``frequency`` (Hz).

    Parameters
    ----------
    k : array_like
        Horizontal wavenumbers (rad/m).
    frequency : float
        Frequency (Hz).
    """
    frequency = float(frequency)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ConfigurationError(
            f"modal_phase_speeds: frequency={frequency!r} must be a finite "
            f"positive number of Hz.")
    omega = 2.0 * np.pi * frequency
    return omega / np.real(k)


def modal_grazing_angles(k, frequency: float,
                         sound_speed: float) -> np.ndarray:
    """Each mode's grazing angle in degrees against ``sound_speed``:
    ``θ_m = arccos(c / v_m)``, ``v_m`` the phase speed
    (:func:`modal_phase_speeds`).

    The dictionary between the modal and ray pictures. A mode whose phase
    speed is below ``c`` is evanescent there and has no real angle, so it
    comes back ``nan``; clipping the ratio to 1.0 instead would stack those
    modes on broadside, where a patterned source applies its notch to them.

    Parameters
    ----------
    k : array_like
        Horizontal wavenumbers (rad/m).
    frequency : float
        Frequency (Hz).
    sound_speed : float
        The speed (m/s) the angles are measured against.
    """
    return _modal_grazing_angles(k, frequency, sound_speed)


def _modal_grazing_angles(k, frequency: float, sound_speed: float, *,
                         who: str = 'modal_grazing_angles') -> np.ndarray:
    """:func:`modal_grazing_angles` reporting its refusals as ``who``."""
    c = float(sound_speed)
    if not c > 0.0:
        raise ConfigurationError(
            f"{who}: sound_speed must be > 0 m/s; got {c}.")
    speeds = np.asarray(modal_phase_speeds(k, frequency), dtype=float)
    # No clip: ``c / v_m`` exceeds 1 exactly for the evanescent modes, and
    # ``arccos`` of that is already nan, which is the right answer.
    with np.errstate(invalid='ignore'):
        return np.degrees(np.arccos(c / speeds))


def medium_density_at(depth: float, tops, densities, bottom_depth: float,
                      halfspace_density: float) -> float:
    """``ρ(z)`` (g/cm³) of the medium holding ``depth`` (m) in a stack of
    media.

    Medium ``i`` spans ``tops[i]`` to the next top (to ``bottom_depth`` for
    the last) at ``densities[i]``, and ``halfspace_density`` holds below the
    stack. A depth on an interface belongs to the upper medium (the
    convention of :meth:`~uacpy.core.bottom.SeabedColumn.at`). It is the
    ``ρ(z_s)`` of the point-source modal sum (Jensen et al., *Computational
    Ocean Acoustics*, eq. 5.14): a model run builds the stack from its
    environment, and a mode set from the medium table its ``.mod`` recorded
    (:meth:`~uacpy.core.results.MediaTable.density_at`).

    Parameters
    ----------
    depth : float
        Depth (m) whose medium is wanted.
    tops : sequence of float
        Top depth (m) of each medium, increasing.
    densities : sequence of float
        Density (g/cm³) of each medium, one per top.
    bottom_depth : float
        Depth (m) of the bottom of the last medium.
    halfspace_density : float
        Density (g/cm³) below ``bottom_depth``.

    Returns
    -------
    float
        The density (g/cm³) of the medium holding ``depth``.
    """
    z = float(depth)
    tops = [float(t) for t in tops]
    for i in range(len(tops)):
        bottom = (tops[i + 1] if i + 1 < len(tops)
                  else float(bottom_depth))
        if z <= bottom:
            return float(densities[i])
    return float(halfspace_density)


def modal_excitation(phi_at_sources, weights, directivity=None) -> np.ndarray:
    """Complex modal excitation of a source array: ``Σₙ wₙ·φₘ(zₙ)``, times
    the element response ``S(θₘ)`` when given.

    ``phi_at_sources`` is ``(n_sources, n_modes)``, the mode shapes at the
    element depths (:func:`mode_shapes_at`); ``weights`` the ``n_sources``
    complex drives. Each element drives every mode in proportion to the
    mode's shape at its depth and the weights set the sum, so choosing them
    chooses the modal content: the *mode filter* of Medwin & Clay §11.3.1.
    ``directivity`` is the ``n_modes`` element response at each mode's
    grazing angle: ``field.f90`` multiplies the modal excitation by it before
    summing (``C = C * REAL(S)``), and elements sharing one pattern and one
    mode angle factor it out of the array sum.

    Parameters
    ----------
    phi_at_sources : ndarray
        ``(n_sources, n_modes)`` mode shapes at the element depths.
    weights : array_like
        Complex drive of each element.
    directivity : array_like, optional
        Element response at each mode's grazing angle.
    """
    weights = np.atleast_1d(np.asarray(weights, dtype=np.complex128))
    excited = weights @ np.asarray(phi_at_sources)
    if directivity is None:
        return excited
    return excited * directivity


def _halfspace_of(bottom, who):
    """The half-space a modal perturbation reads from ``bottom``.

    A :class:`BoundaryProperties` passes through. A range-independent,
    unlayered :class:`~uacpy.core.bottom.Bottom` — what ``env.bottom`` holds —
    is unwrapped to its half-space, so the call the documentation shows,
    ``bottom=env.bottom``, works. A layered or range-dependent ``Bottom`` has
    no single half-space under the water column, so it is refused with the
    call that picks one."""
    if not isinstance(bottom, Bottom):
        return bottom
    if bottom.is_range_dependent or bottom.is_layered:
        kind = ("range-dependent" if bottom.is_range_dependent
                else "layered")
        raise ConfigurationError(
            f"{who}: bottom is a {kind} Bottom, which has no single "
            f"half-space under the water column; pass the one these modes "
            f"see, e.g. bottom=env.bottom.halfspace_at(range=0.0).")
    return bottom.halfspace_at(range=0.0)


# psi² oscillates at twice the mode's vertical wavenumber, so 8 samples per
# wavelength is 4 per period of the integrand. Measured on a Pekeris set
# (D = 100 m, f = 50 Hz, 6 modes) against an 8001-point reference, with
# alpha(z) stepping 0 → 2 dB/m at 50 m: 1-3 % error at 51 depths (dz = 2 m,
# every mode over the bar), 3-7 % at 21, 6-19 % at 9, and up to 104 % at 5.
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
        NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)


def _water_column_term(k, weight, depths, a_neper, c_arr, rho_arr, omega,
                       who):
    """The water column's first-order attenuation per mode, before the
    normalisation, ``(omega / k_rm) * integral(alpha / (c rho) psi**2 dz)``,
    and that normalisation's water-column part ``integral(psi**2 / rho dz)``.

    Returns ``(water_term, norm, kr, kr_safe, unsolvable)``: a mode whose
    ``Re k <= 0`` is flagged ``unsolvable`` and carries a placeholder
    ``kr_safe`` of 1 in the arithmetic.
    """
    # Perturbation on Re(psi)**2: for krakenc's complex modes this is
    # an approximation (JKPS 5.176 wants the complex psi**2); the
    # imaginary part of a weakly-attenuated mode shape is O(alpha),
    # so the error is second-order in the loss being computed.
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
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)
    # Clamped only to keep the array arithmetic finite; every entry it
    # covers is marked no-data before the result is returned.
    kr_safe = np.where(kr > 0, kr, 1.0)
    _warn_if_depth_axis_underresolves(
        a_neper, kr, omega, c_arr, depths, who)
    water_term = (omega / kr_safe) * np.trapezoid(integrand, depths, axis=0)
    return water_term, norm, kr, kr_safe, unsolvable


def _check_water_only_tabulation(weight, depths, seafloor_depth, who):
    """With no half-space: the tabulation must reach the seafloor when the
    caller names it, and a mode still non-zero at the deepest tabulated
    depth is reported, its normalisation stopping at the seabed so the
    water-column term returned is an upper bound on that term."""
    # A truncated span shorts the water-column integrals too (measured
    # 0.5-0.7x on a half-column tabulation), so the check runs whenever
    # the caller names the seafloor, bottom term or not.
    if seafloor_depth is not None:
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
    # The tail psi(D)²/(2 gamma_m rho_b) that carries the
    # normalisation from JKPS 5.169's 0->D out to the 0->inf the
    # text after 5.176 calls for needs a half-space gamma_m and rho_b, and
    # neither exists here, so `norm` stays a water-column integral
    # while the true denominator is larger. The omitted term is
    # positive, so the water-column term comes back high: an upper bound
    # on that term, measured 0.7 % / 2.4 % / 5.1 % / 17 % over the exact
    # perturbation integral for modes 1-4 of an analytic 100 m Pekeris
    # guide. The seabed-loss term is absent altogether.
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
            f"({float(depths[-1]):g} m), so for a penetrable seabed "
            f"they continue as an evanescent tail into it, but with "
            f"no half-space gamma, density or attenuation to form that "
            f"part from, the returned Im(k) is the water-column term "
            f"alone: an UPPER BOUND on that term (its ∫psi²/rho dz "
            f"normalisation stops at the seabed, where the text after "
            f"JKPS Eq. 5.176 extends it into the bottom; measured "
            f"0.7-17 % high across the four modes of a 100 m Pekeris "
            f"guide) with NO seabed-loss term, so over a lossy seabed "
            f"it is far below the true attenuation (23-487x low for "
            f"the trapped modes over a 0.5 dB/wavelength seabed at "
            f"50 Hz). Pass bottom= (env.bottom) for both. A "
            f"RIGID-seabed mode set (psi'(D)=0, psi(D) at a maximum) "
            f"has no tail and its returned Im(k) is exact; pass "
            f"bottom=BoundaryProperties(acoustic_type='rigid') to "
            f"state that boundary and silence this warning.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)


def _geoacoustic_halfspace(bottom, who):
    """``bottom`` when its type carries seabed geoacoustics; ``None`` for a
    vacuum or rigid seabed (no energy enters it, so the water-column term
    is exact); a refusal for a reflection-table seabed, whose cp, rho and
    attenuation are placeholders."""
    # A vacuum / rigid / file / precalc boundary carries no seabed
    # geoacoustics: its cp, rho and attenuation are the placeholders
    # __post_init__ resolved (1600 m/s, 1.5 g/cm3, 0.5 dB/lambda), and
    # reading them as a half-space would fabricate an attenuation the
    # seabed does not have. `Bottom.all_sound_speeds` states the same rule
    # for the same data. Measured on an ideal 100 m guide at 50 Hz with a
    # RIGID seabed, reading bottom=BoundaryProperties('rigid') as a
    # half-space would add 0.40 / 0.51 dB of loss at 1 km and 7.99 /
    # 10.22 dB at 20 km on modes 1 / 2, and name two modes "leaky" against a
    # 1600 m/s half-space that does not exist. Runs BEFORE the bottom block so a dropped boundary takes
    # the water-only path rather than falling into it with bottom = None.
    if not isinstance(bottom, BoundaryProperties):
        raise ConfigurationError(
            f"{who}: bottom must be a "
            f"BoundaryProperties; got {type(bottom).__name__}."
        )
    _btype = str(getattr(bottom, 'acoustic_type', '')).lower()
    if not BoundaryType.from_string(_btype).is_geoacoustic:
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
    return bottom


def _halfspace_tail_term(bottom, phi_re, depths, seafloor_depth, kr,
                         kr_safe, omega, frequency, norm, who):
    """The closed-form term of a penetrable half-space and its share of the
    normalisation: below the seafloor the mode decays as
    ``psi(D) exp(-gamma (z - D))``. Returns ``(bottom_term, norm)``, ``norm``
    extended over the tail."""
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
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    cb = float(bottom.sound_speed)
    rho_b = float(bottom.density) * 1000.0
    # dB/wavelength -> nepers/m: alpha[dB/lam] * (f/c) [lam/m] / (20/ln10).
    ab_neper_per_m = (
        float(bottom.attenuation) / NEPER_TO_DB
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
    norm = norm + tail_norm
    # A leaky mode gets no term, as the comment above already says:
    # there is no finite first-order bottom-attenuation perturbation for
    # a radiating mode — the tail integral diverges, so the closed form
    # this line specialises does not exist. gamma_safe = 1 only keeps the
    # masked lanes finite; used as a value, a bare 1 carrying units of 1/m
    # would make the loss scale with the length unit.
    bottom_term = np.where(
        trapped,
        psi_D ** 2 * ab_neper_per_m * omega
        / (2.0 * kr_safe * gamma_safe * cb * rho_b),
        0.0,
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
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    return bottom_term, norm


def modal_attenuation(k, phi, depths, alpha_dB_per_m, *, frequency,
                      sound_speed_z=None,
                      density_z=DEFAULT_WATER_DENSITY, bottom=None,
                      seafloor_depth=None):
    """First-order modal attenuation ``alpha_m`` (Np/m), one per mode.

    For mode ``m`` with horizontal wavenumber ``k_rm`` and depth
    eigenfunction ``psi_m``, perturbation theory gives

    ``alpha_m = (omega / k_rm) * integral( alpha(z)/(c(z) rho(z)) *
    Re(psi_m)^2 dz ) / integral( Re(psi_m)^2 / rho(z) dz )``

    (Jensen, Kuperman, Porter & Schmidt, *Computational Ocean Acoustics*,
    Eq. 5.176, with the normalisation of Eq. 5.169 written out as the
    denominator). When a penetrable ``bottom`` is given, the half-space
    below ``D`` adds the closed-form bottom term of Eq. 5.180,
    ``delta_alpha_m = psi(D)^2 * alpha_b * omega / (2 k_rm gamma_m c_b rho_b)``
    (``psi = psi(D) exp(-gamma_m (z - D))``, Eqs. 5.177-5.179), to the
    numerator, and its tail ``integral_D^inf psi^2/rho_b dz =
    psi(D)^2 / (2 gamma_m rho_b)`` to the normalisation.

    Parameters
    ----------
    k : array_like
        Complex horizontal wavenumbers, one per mode (1/m).
    phi : array_like
        Mode shapes on ``depths``, shaped ``(n_depths, n_modes)`` — the
        ``psi_m`` of the formula above.
    depths : array_like
        Tabulation depths (m), ascending.
    alpha_dB_per_m : float or array_like
        Volume attenuation, a scalar or one value per depth.
    frequency : float
        Frequency (Hz). Required — the perturbation is ``omega``-scaled.
    sound_speed_z : float or array_like
        Sound speed (m/s) the modes were computed in, a scalar or one value
        per depth. Required: the perturbation term is ``2iα(z)ω/c(z)``
        (Jensen et al., *Computational Ocean Acoustics*, sect. 5.9.1), so at
        a fixed ``α(z)`` every ``α_m`` scales as ``1/c`` (a 1480 m/s column
        read at 1500 m/s is 1.3 % under-attenuated), and an unset speed is
        refused rather than guessed. ``env.ssp.sound_speed_at(depths)`` gives
        the profile on the tabulation.
    density_z : float or array_like
        Density (g/cm3), scalar or per depth. Default: the package's one
        water density ``DEFAULT_WATER_DENSITY``.
    bottom, seafloor_depth
        A penetrable half-space (a ``BoundaryProperties``, or a
        range-independent unlayered ``Bottom`` such as ``env.bottom``) and
        where it starts; see the warnings the function raises when the
        tabulation and the seabed disagree.

    Returns
    -------
    ndarray
        ``alpha_m`` per mode (Np/m). ``nan`` for a mode the perturbation
        has no answer for — a non-positive ``Re k`` or a non-positive
        shape normalisation — rather than a level that would then
        propagate undamped through every field built from the set.
    """
    return _modal_attenuation(
        k,
        phi,
        depths,
        alpha_dB_per_m,
        frequency=frequency,
        sound_speed_z=sound_speed_z,
        density_z=density_z,
        bottom=bottom,
        seafloor_depth=seafloor_depth)


def _modal_attenuation(
    k,
    phi,
    depths,
    alpha_dB_per_m,
    *,
    frequency,
    sound_speed_z=None,
    density_z=DEFAULT_WATER_DENSITY,
    bottom=None,
    seafloor_depth=None,
    who: str = "modal_attenuation",
):
    """:func:`modal_attenuation` reporting its refusals as ``who``."""
    if sound_speed_z is None:
        raise ConfigurationError(
            f"{who}: sound_speed_z is required. Every alpha_m scales as "
            f"1/c(z) (the perturbation term 2i*alpha*omega/c, Jensen et al. "
            f"sect. 5.9.1), so the speed is not guessed.",
            remediation="Pass sound_speed_z= the water sound speed (m/s) "
                        "the modes were computed in, on the mode depths: a "
                        "scalar for an isovelocity column, or one value per "
                        "depth, e.g. env.ssp.sound_speed_at(depths).")
    bottom = _halfspace_of(bottom, who)
    omega = 2.0 * np.pi * float(frequency or 0.0)
    # A negative frequency returns a negative, amplifying attenuation, and a
    # NaN one a NaN per mode.
    if not (np.isfinite(omega) and omega > 0.0):
        raise ConfigurationError(
            f"{who}: requires a positive finite frequency in Hz; got "
            f"{frequency!r}."
        )
    # Each depth-tabulated input is a scalar (spread over the
    # tabulation) or one value per tabulated depth.
    a = _on_depths(alpha_dB_per_m, 'alpha', depths, who)
    c_arr = _on_depths(sound_speed_z, 'sound_speed_z', depths, who)
    rho_g = _on_depths(density_z, 'density_z', depths, who)
    rho_arr = rho_g * 1000.0  # g/cm³ → kg/m³
    a_neper = a / NEPER_TO_DB
    phi_re = np.asarray(phi).real
    weight = phi_re ** 2
    water_term, norm, kr, kr_safe, unsolvable = _water_column_term(
        k, weight, depths, a_neper, c_arr, rho_arr, omega, who)
    bottom_term = np.zeros_like(water_term)
    if bottom is None:
        _check_water_only_tabulation(weight, depths, seafloor_depth, who)
    else:
        bottom = _geoacoustic_halfspace(bottom, who)
    if bottom is not None:
        bottom_term, norm = _halfspace_tail_term(
            bottom, phi_re, depths, seafloor_depth, kr, kr_safe, omega,
            frequency, norm, who)
    if np.any(norm <= 0):
        warnings.warn(
            f"{who}: {int(np.count_nonzero(norm <= 0))} "
            f"mode(s) have a non-positive shape normalisation "
            f"∫psi²/rho dz, which the perturbation divides by; their "
            f"attenuation is returned as NaN: a non-positive "
            f"normalisation has no first-order perturbation.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)
    unsolvable = unsolvable | (norm <= 0)
    norm = np.where(norm > 0, norm, 1.0)
    alpha_m = (water_term + bottom_term) / norm
    # The perturbation had no answer for these modes: report no data
    # rather than a level that participates in every field built from
    # this mode set.
    alpha_m = np.where(unsolvable, np.nan, alpha_m)
    return alpha_m


def _mode_shapes_at(phi, depths, at_depths, *, outside, who):
    """:func:`mode_shapes_at` with the caller's name in its messages."""
    if outside not in ('raise', 'nan'):
        raise ConfigurationError(
            f"{who}: outside must be 'raise' or 'nan'; got {outside!r}.")
    phi = np.asarray(phi)
    zp = np.asarray(depths, dtype=float).ravel()
    z = np.atleast_1d(np.asarray(at_depths, dtype=float)).ravel()
    if phi.ndim != 2 or phi.shape[0] != zp.size:
        raise ConfigurationError(
            f"{who}: phi must be (n_tabulated_depths, n_modes) = "
            f"({zp.size}, n_modes); got {phi.shape}.")
    lo, hi = float(zp[0]), float(zp[-1])
    # A depth written to the tabulation's own precision sits on its end.
    tol = 1e-9 * max(1.0, abs(lo), abs(hi))
    off = (z < lo - tol) | (z > hi + tol)
    if outside == 'raise' and np.any(off):
        raise ConfigurationError(
            f"{who}: depth(s) {np.array2string(z[off], precision=2)} m lie "
            f"outside the tabulated mode depths [{lo:g}, {hi:g}] m, so their "
            f"mode shapes would be extrapolated.",
            remediation="Tabulate the modes over the full source/receiver "
                        "span (Kraken().compute_modes uses a dense grid over "
                        "the whole waveguide).")
    dtype = np.complex128 if np.iscomplexobj(phi) else float
    out = np.empty((z.size, phi.shape[1]), dtype=dtype)
    for m in range(phi.shape[1]):
        column = phi[:, m]
        out[:, m] = np.interp(z, zp, column.real)
        if np.iscomplexobj(column):
            out[:, m] += 1j * np.interp(z, zp, column.imag)
    out[off, :] = np.nan
    return out


def mode_shapes_at(phi, depths, at_depths, *, outside='raise'):
    """Mode shapes interpolated from their tabulation onto ``at_depths``.

    Linear in depth, real and imaginary parts separately for complex
    (krakenc) shapes. Outside the tabulation there is no mode shape to
    interpolate: ``np.interp`` would hold the end value flat, which is
    neither the shape nor the evanescent tail the half-space carries, so a
    depth there is refused (``outside='raise'``) or returned as NaN
    (``outside='nan'``) — never extrapolated. :meth:`~uacpy.Modes.shapes_at`,
    :meth:`~uacpy.Modes.excitation` and
    :meth:`~uacpy.Modes.modal_pressure_field` all read their shapes
    through this.

    Parameters
    ----------
    phi : array_like
        Mode shapes, ``(n_tabulated_depths, n_modes)`` (``Modes.phi``).
    depths : array_like
        Increasing depths (m) ``phi`` is tabulated on (``Modes.depths``).
    at_depths : float or array_like
        Depths (m) wanted.
    outside : {'raise', 'nan'}, default 'raise'
        What a depth outside the tabulation gets.

    Returns
    -------
    ndarray
        ``(n_at_depths, n_modes)``, real or complex as ``phi`` is.
    """
    return _mode_shapes_at(phi, depths, at_depths, outside=outside,
                           who='mode_shapes_at')


def modal_field(k, phi_zs, phi_zr, ranges, *,
                source_density=DEFAULT_WATER_DENSITY,
                form: str = 'asymptotic'):
    """Coherent complex pressure from the modal sum.

    .. math::
        p(z_r, r) \\approx \\frac{e^{-i\\pi/4}}{\\rho_s}
        \\sqrt{\\frac{2\\pi}{r}}\\sum_m
        \\psi_m(z_s)\\,\\psi_m(z_r)\\,
        \\frac{e^{-ik_m r}}{\\sqrt{k_m}}

    the far-field form of the normal-mode solution (Jensen, Kuperman,
    Porter & Schmidt, *Computational Ocean Acoustics*, sect. 5.3), which
    ``form='asymptotic'`` (the default, the engines' form) evaluates.
    ``form='hankel'`` evaluates the exact sum it approximates (JKPS
    sect. 5.2.1), written in the same ``e^(i(omega t - k r))`` convention:
    ``(-i pi / rho_s) sum psi_m(z_s) psi_m(z_r) H0^(2)(k_m r)``, whose
    large-``k r`` limit is the asymptotic form term by term, for real and
    complex (krakenc) mode shapes alike; it is what the closed-form reference
    fields of :mod:`uacpy.analytic` sum, and it holds near the source. Give it
    the mode shapes **already evaluated at the source and receiver
    depths**: choosing how to get there from a tabulation — interpolating,
    refusing to extrapolate, masking what lies outside — is the caller's
    question, and :meth:`~uacpy.Modes.modal_pressure_field` answers it
    one way.

    Parameters
    ----------
    k : array_like
        Complex horizontal wavenumbers, ``(n_modes,)``. Either sign of
        ``Im k`` is accepted: a passive medium can only attenuate, so the
        sign is forced.
    phi_zs : array_like
        Mode shapes at the source depth, ``(n_modes,)``.
    phi_zr : array_like
        Mode shapes at each receiver depth, ``(n_depths, n_modes)``.
    ranges : array_like
        Ranges (m). ``r <= 0`` is outside the form's domain and comes back
        ``nan``, not a number within a few dB of the 1 m answer.
    source_density : float, default ``DEFAULT_WATER_DENSITY``
        Density at the source (**g/cm3** — the unit KRAKEN normalises its
        modes with, and the unit the ``.env`` file carries). The default is
        the package's one water density (1.027), the water every deck
        writes; pass 1.0 to leave the factor out, as ``field.exe`` does.
        ``field.exe``'s prefactor ``i*sqrt(2*pi)*exp(i*pi/4)``
        (``EvaluateMod.f90:34``) is the negative of this sum's
        ``exp(-i*pi/4)*sqrt(2*pi)``, the conjugate of JKPS sect. 5.2.1's;
        :class:`~uacpy.models.Kraken`'s reader removes both the sign and the
        missing ``1/rho_s``, so a Kraken pressure Field equals this sum at
        ``source_density=rho(z_s)``.
    form : {'asymptotic', 'hankel'}
        The far-field form, or the exact Hankel-function sum.

    Returns
    -------
    ndarray
        Complex pressure, ``(n_depths, n_ranges)``.
    """
    if form not in ('asymptotic', 'hankel'):
        raise ConfigurationError(
            f"modal_field: form must be 'asymptotic' or 'hankel'; got "
            f"{form!r}.")
    r = np.atleast_1d(np.asarray(ranges, dtype=float))
    phi_zs = np.asarray(phi_zs)
    phi_zr = np.asarray(phi_zr)
    k = np.asarray(k)
    if form == 'hankel':
        return _hankel_mode_sum(k, phi_zs, phi_zr, float(source_density), r)
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
    # r <= 0 is outside the form's domain, not a quiet spot in it — the
    # same no-data marking every other range-zero path in the package
    # applies.
    P[:, r <= 0.0] = np.nan
    return P


def _hankel_mode_sum(k, phi_zs, phi_zr, rho_s, r):
    """``form='hankel'`` of :func:`modal_field`: ``(-i pi / rho_s) sum
    psi(z_s) psi(z) H0^(2)(k r)``, the travelling-wave form of JKPS
    sect. 5.2.1's ``(i / 4 rho_s) sum psi psi H0^(1)(k r)`` times the 1 m
    reference ``4 pi``; NaN at ``r <= 0``. ``k`` takes either sign of
    ``Im k``: the decaying branch is used."""
    # scipy loads on first use: ``import uacpy`` reaches this module and
    # pays for numpy only.
    from scipy.special import hankel1
    kr = np.asarray(k).real + 1j * np.abs(np.asarray(k).imag)
    p = np.full((phi_zr.shape[0], r.size), np.nan, dtype=complex)
    on = r > 0.0
    if kr.size and on.any():
        # H0^(2)(conj z) = conj(H0^(1)(z)): the Hankel factor alone changes
        # convention. The mode shapes enter as given, unconjugated, as they
        # do in the asymptotic branch and in EvaluateMod.f90:47.
        h2 = np.conj(hankel1(0, np.outer(kr, r[on])))   # (n_modes, n_range)
        p[:, on] = (-1j * np.pi / rho_s) * (phi_zr * phi_zs) @ h2
    elif on.any():
        p[:, on] = 0.0
    return p
