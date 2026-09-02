"""Geometric-regime target strength of simple rigid shapes.

Monostatic TS in dB re 1 m², ``TS = 10·log10(σ_diff)`` with the
differential cross-section conventions of Urick Table 9.1 (reproduced in
Abraham §3.4):

* Sphere (radius a):                 ``TS = 10·log10(a²/4)``
* Convex body (principal radii):     ``TS = 10·log10(a₁a₂/4)``
* Ellipsoid (along semi-axis a):     ``TS = 20·log10(bc/2a)``
* Cylinder (broadside):              ``TS = 10·log10(aL²/2λ)``
* Plate (normal incidence):          ``TS = 20·log10(A/λ)``

Every function warns below its geometric-regime bound: ``ka > 10`` for
the sphere family (Abraham §3.4: "reasonably accurate when ka > 10"),
``ka > 1`` on the cylinder radius (Urick's ``ka ≫ 1`` condition;
Abraham's reference cylinder runs at ka ≈ 4), and both plate dimensions
above one wavelength (``kw > 2π``, since the plate's check scale is a
full dimension rather than a radius). All formulas assume the far field
(``r > L²/λ`` for the cylinder, ``r > A/λ`` for the plate). Aspect
patterns for the cylinder and plate are valid near broadside / normal
incidence only: they null at end-on even though real end-caps and edges
still reflect (Abraham Fig. 3.24 keeps separate end-cap terms).

These feed the ``target_strength`` argument of
:func:`uacpy.sonar.active_signal_excess` /
:func:`uacpy.sonar.active_signal_excess_field`.
"""

from __future__ import annotations

import warnings

import numpy as np

from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.acoustic_signal._signal_validate import require_positive_finite_scalar
from uacpy.core._warn_frames import USER_FRAME_SKIP

# Geometric-scattering validity bounds. Sphere-family: the rigid-sphere
# approximation is "reasonably accurate when ka > 10" (Abraham §3.4,
# after eq. 3.218). Cylinder: Urick Table 9.1 requires only ka >> 1 on the
# radius, and Abraham's reference cylinder (Fig. 3.24) runs at ka ≈ 4.
#
# The plate needs its own, larger bound. Its check scale is a full dimension,
# not a radius, so ka > 1 would pass any plate wider than λ/2π ≈ 0.16λ — deep
# in the Rayleigh regime, where the physical-optics 20·log10(A/λ) has no
# validity at all (a 0.2 × 0.2 m plate at λ = 1 m returned TS = -28.0 dB
# unflagged). Urick Table 9.1 and Abraham §3.4 both require every dimension
# ≫ λ, so the bound is set at exactly one wavelength, k·w = 2π, and the
# formula only becomes accurate well above it.
_KA_MIN_SPHERE = 10.0
_KA_MIN_CYLINDER = 1.0
_KA_MIN_PLATE = 2.0 * np.pi


def _require_positive(value, label: str) -> float:
    """Validate a scalar dimension / frequency / sound speed and return it as
    ``float`` — the package's one positive-finite-scalar guard, under the
    ``"target strength"`` prefix."""
    return require_positive_finite_scalar(value, "target strength", label)


def _warn_below_geometric(scale_m: float, frequency_hz, sound_speed,
                          label: str, ka_min: float) -> None:
    """Warn when ``k·scale`` falls below the geometric-regime bound.

    Skipped when ``frequency_hz`` is ``None`` — the frequency-flat
    formulas (sphere / convex / ellipsoid) accept an optional frequency
    purely for this validity check.
    """
    if frequency_hz is None:
        return
    f = _require_positive(frequency_hz, 'frequency_hz')
    c = _require_positive(sound_speed, 'sound_speed')
    ka = 2.0 * np.pi * f / c * scale_m
    if ka < ka_min:
        warnings.warn(
            f"{label}: k·a = {ka:.2f} < {ka_min:g} — below the "
            f"geometric-scattering regime; the formula overestimates the "
            f"response of a Rayleigh/resonance-regime scatterer.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )


def ts_sphere(
    radius_m,
    *,
    frequency_hz=None,
    sound_speed=DEFAULT_SOUND_SPEED,
) -> float:
    """Rigid-sphere target strength ``TS = 10·log10(a²/4)`` (dB re 1 m²).

    Frequency-flat in the geometric regime — the classic anchor is the
    2-m-radius sphere at ``TS = 0`` dB. Pass ``frequency_hz`` to enable
    the ``ka > 10`` validity check (Urick Table 9.1; Abraham §3.4
    eq. 3.218).

    Parameters
    ----------
    radius_m : float
        Sphere radius (m).
    frequency_hz : float, optional
        Frequency (Hz), used only to warn when ``ka < 10``.
    sound_speed : float, optional
        Sound speed (m/s) for the validity check. Default 1500.
    """
    a = _require_positive(radius_m, 'radius_m')
    _warn_below_geometric(a, frequency_hz, sound_speed, 'ts_sphere',
                          _KA_MIN_SPHERE)
    return float(10.0 * np.log10(a ** 2 / 4.0))


def ts_convex(
    radius1_m,
    radius2_m,
    *,
    frequency_hz=None,
    sound_speed=DEFAULT_SOUND_SPEED,
) -> float:
    """Smooth convex body: ``TS = 10·log10(a₁a₂/4)`` (dB re 1 m²).

    ``a₁``, ``a₂`` are the principal radii of curvature at the point of
    the body closest to the sonar (Abraham §3.4 eq. 3.217). Reduces to
    :func:`ts_sphere` for ``a₁ = a₂``. Pass ``frequency_hz`` to enable
    the ``ka > 10`` check on the smaller radius.
    """
    a1 = _require_positive(radius1_m, 'radius1_m')
    a2 = _require_positive(radius2_m, 'radius2_m')
    _warn_below_geometric(min(a1, a2), frequency_hz, sound_speed,
                          'ts_convex', _KA_MIN_SPHERE)
    return float(10.0 * np.log10(a1 * a2 / 4.0))


def ts_ellipsoid(
    a_m,
    b_m,
    c_m,
    *,
    frequency_hz=None,
    sound_speed=DEFAULT_SOUND_SPEED,
) -> float:
    """Ellipsoid viewed along semi-axis ``a``: ``TS = 20·log10(bc/2a)``.

    Urick Table 9.1. Equivalent to :func:`ts_convex` with the principal
    radii at the tip of the ``a`` axis (``b²/a``, ``c²/a``); ``a = b =
    c`` recovers the sphere. The ensonified direction is along ``a`` —
    permute the semi-axes for other aspects.
    """
    a = _require_positive(a_m, 'a_m')
    b = _require_positive(b_m, 'b_m')
    c = _require_positive(c_m, 'c_m')
    return ts_convex(
        b ** 2 / a, c ** 2 / a,
        frequency_hz=frequency_hz, sound_speed=sound_speed,
    )


def ts_cylinder(
    radius_m,
    length_m,
    frequency_hz,
    *,
    angle_deg=0.0,
    sound_speed=DEFAULT_SOUND_SPEED,
):
    """Rigid finite cylinder: ``TS = 10·log10[(aL²/2λ)·sinc²β·cos²θ]``.

    ``β = kL·sinθ`` with ``θ`` the ensonification angle from broadside
    (perpendicular to the cylinder axis). Urick Table 9.1; Abraham §3.4
    eq. 3.221. Valid near broadside only — the pattern nulls at end-on
    even though real end-caps still reflect. The first null of
    ``[sinβ/β]²`` is at ``β = π``, i.e. ``sinθ = λ/(2L)``, so the main
    lobe spans ``≈ λ/L`` radians null to null.

    Parameters
    ----------
    radius_m, length_m : float
        Cylinder radius and length (m).
    frequency_hz : float
        Frequency (Hz).
    angle_deg : float or array_like, optional
        Aspect angle(s) from broadside (degrees). Default 0 (broadside).
    sound_speed : float, optional
        Sound speed (m/s). Default 1500.

    Returns
    -------
    float or ndarray
        TS (dB re 1 m²); ``-inf`` at exact pattern nulls.
    """
    a = _require_positive(radius_m, 'radius_m')
    L = _require_positive(length_m, 'length_m')
    f = _require_positive(frequency_hz, 'frequency_hz')
    c = _require_positive(sound_speed, 'sound_speed')
    _warn_below_geometric(a, f, c, 'ts_cylinder', _KA_MIN_CYLINDER)
    lam = c / f
    k = 2.0 * np.pi / lam
    theta = np.deg2rad(np.asarray(angle_deg, dtype=float))
    beta = k * L * np.sin(theta)
    # numpy's sinc is the normalized one, sinc(x) = sin(pi*x)/(pi*x), so the
    # argument is divided by pi to recover Abraham's unnormalized [sin b / b].
    # This also supplies the b -> 0 limit of 1 at broadside for free.
    pattern = np.sinc(beta / np.pi) ** 2 * np.cos(theta) ** 2
    sigma = a * L ** 2 / (2.0 * lam) * pattern
    with np.errstate(divide='ignore'):
        ts = 10.0 * np.log10(sigma)
    return float(ts) if np.ndim(angle_deg) == 0 else ts


def ts_plate(
    width_m,
    height_m,
    frequency_hz,
    *,
    angle_deg=0.0,
    sound_speed=DEFAULT_SOUND_SPEED,
):
    """Rigid rectangular plate: ``TS = 10·log10[(wh/λ)²·sinc²β·cos²θ]``.

    ``β = k·w·sinθ`` with ``θ`` the angle from normal incidence,
    rotating about the plate's height axis (``w`` = the dimension in
    the rotation plane). At normal incidence this is the physical-optics
    ``TS = 20·log10(A/λ)`` for a flat plate of area ``A`` (Urick
    Table 9.1; Abraham §3.4). Requires both dimensions ≫ λ; valid near
    normal incidence only. A dimension below one wavelength warns.

    Parameters
    ----------
    width_m, height_m : float
        Plate dimensions (m); ``width_m`` lies in the rotation plane.
    frequency_hz : float
        Frequency (Hz).
    angle_deg : float or array_like, optional
        Angle(s) from normal incidence (degrees). Default 0.
    sound_speed : float, optional
        Sound speed (m/s). Default 1500.

    Returns
    -------
    float or ndarray
        TS (dB re 1 m²); ``-inf`` at exact pattern nulls.
    """
    w = _require_positive(width_m, 'width_m')
    h = _require_positive(height_m, 'height_m')
    f = _require_positive(frequency_hz, 'frequency_hz')
    c = _require_positive(sound_speed, 'sound_speed')
    _warn_below_geometric(min(w, h), f, c, 'ts_plate',
                          _KA_MIN_PLATE)
    lam = c / f
    k = 2.0 * np.pi / lam
    theta = np.deg2rad(np.asarray(angle_deg, dtype=float))
    beta = k * w * np.sin(theta)
    # numpy's sinc is normalized, sinc(x) = sin(pi*x)/(pi*x); dividing the
    # argument by pi gives the unnormalized [sin b / b] of the physical-optics
    # pattern, and its b -> 0 limit of 1 at normal incidence.
    pattern = np.sinc(beta / np.pi) ** 2 * np.cos(theta) ** 2
    sigma = (w * h / lam) ** 2 * pattern
    with np.errstate(divide='ignore'):
        ts = 10.0 * np.log10(sigma)
    return float(ts) if np.ndim(angle_deg) == 0 else ts
