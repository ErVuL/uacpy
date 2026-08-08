"""Grain size (Wentworth ϕ) → bulk geoacoustic properties.

Pure, dependency-light conversion shared by the carrier layer
(:meth:`uacpy.core.bottom.BoundaryProperties.from_grain_size`) and the on-demand
data layer (:mod:`uacpy.data.sediment`). It lives in ``core`` so a bottom can be
built from a grain size without importing ``uacpy.data``.

Grain size is a **construction-time input only**: it is converted here to an
explicit half-space (``sound_speed`` / ``density`` / ``attenuation``) that every
propagation model consumes. There is no ``'grain-size'`` boundary *type* — no
model sees a grain size at run time.

Two models are provided:

- ``'hamilton'`` (default) — the **low-frequency** Hamilton & Bachman (1982)
  table + Hamilton (1980) ``k_p`` attenuation, reproduced from the open-access
  CC-BY ESAB supplement of Fonseca, Lurton, Fezzani & Roche (2025).
- ``'apl-uw'`` — APL-UW TR 9407 (1994) §IV.A.4 grain-size relations (the
  **high-frequency** ρ, ν polynomials + α₂/f). These are the same formulas the
  AT ``'G'`` bottom used internally.

Sediment sound speed and density are computed as **ratios to the overlying
seawater**, scaled by the in-situ water properties (so fine muds correctly come
out slower than seawater). Attenuation is returned in dB/wavelength, which is
frequency-independent (``α[dB/λ] = (α/f)·c/1000``).
"""

import warnings
from typing import Dict, Optional

import numpy as np

from uacpy.core.exceptions import ConfigurationError

__all__ = ['GRAIN_SIZE_MODELS', 'grain_size_to_geoacoustics']

# Hamilton & Bachman (1982) "Continental Terrace (Shelf and Slope)" granular
# sediments — bulk density (g/cm³) and sound-speed ratio (sediment/seawater) for
# the representative classes, referenced to seawater c_w = 1510 m/s, ρ_w = 1.030
# g/cm³. Tabulated in the open-access ESAB supplement (Fonseca et al. 2025); the
# Mz values are the Wentworth class centres of Hamilton's named classes.
_HB_REF_CW = 1510.0      # m/s   reference seawater sound speed
_HB_REF_RHOW = 1.030     # g/cm³ reference seawater density
_HB_TABLE = (
    # (Mz_phi, density_gcm3, velocity_ratio)   coarse → fine
    (0.5, 2.034, 1.201),    # coarse sand
    (2.5, 1.962, 1.152),    # fine sand
    (3.5, 1.878, 1.120),    # very fine sand
    (4.5, 1.783, 1.086),    # silty sand
    (5.0, 1.769, 1.076),    # sandy silt
    (6.0, 1.575, 1.036),    # sand-silt-clay
    (7.5, 1.489, 1.012),    # clayey silt
    (8.5, 1.480, 0.990),    # silty clay  (ratio < 1: slower than seawater)
)
_HB_PHI = np.array([r[0] for r in _HB_TABLE])
_HB_RHO = np.array([r[1] for r in _HB_TABLE])      # g/cm³ at the reference water
_HB_VRATIO = np.array([r[2] for r in _HB_TABLE])

GRAIN_SIZE_MODELS = ('hamilton', 'apl-uw')
# Seawater each model's ratios are referenced to, used when the caller gives no
# in-situ values. Hamilton & Bachman tabulate against 1510 m/s / 1.030 g/cm³;
# APL-UW's ratios are applied against 1500 m/s and a unit density ratio by the
# Acoustics-Toolbox 'G' bottom (ReadEnvironmentBell.f90:528 `alphaR = vr*1500`,
# and `HS%rho = rhoR` — the ratio used directly as g/cm³).
_MODEL_WATER_REFERENCE = {'hamilton': (_HB_REF_CW, _HB_REF_RHOW),
                          'apl-uw': (1500.0, 1.0)}
# Valid ϕ range per model; outside it ϕ is clamped, and a UserWarning fires only
# for genuine extrapolation (≳ 1 ϕ beyond).
_MODEL_RANGE = {'hamilton': (float(_HB_PHI[0]), float(_HB_PHI[-1])),
                'apl-uw': (-1.0, 9.0)}


def _hamilton_kp(impedance: float) -> float:
    """Hamilton (1980) attenuation factor ``k_p`` vs sediment impedance.

    ``α(dB/m) = k_p · f(kHz)``; ``impedance`` is ρ·c in 10³ kg m⁻² s⁻¹. The
    piecewise fit (ESAB supplement, after Fig. 18 of Hamilton 1980) peaks
    (~0.78) in medium sand and tails to ~0.46 for coarse / ~0.07 for fine.

    The first and last branches are unreachable through
    :func:`grain_size_to_geoacoustics` (ϕ clamped to ``_MODEL_RANGE`` maps to
    z ∈ ≈[2212, 3689]); they are kept for fidelity to the published curve.
    """
    z = impedance
    if z < 1784.0:
        return 0.07
    if z < 2478.0:
        return 0.07 + 7.2e-5 * (z - 1784.0)
    if z < 3034.0:
        return 0.12 + 1.19e-3 * (z - 2478.0)
    if z < 3270.0:
        return 0.78 - 1.10e-3 * (z - 3034.0)
    if z < 3869.0:
        return 0.52 - 1.00e-4 * (z - 3270.0)
    return 0.46


def _hamilton_geoacoustics(phi, water_sound_speed, water_density):
    """``(cp, density, attenuation)`` from the Hamilton & Bachman ϕ-table."""
    density_ratio = float(np.interp(phi, _HB_PHI, _HB_RHO)) / _HB_REF_RHOW
    velocity_ratio = float(np.interp(phi, _HB_PHI, _HB_VRATIO))
    cp = velocity_ratio * water_sound_speed
    # k_p is calibrated on the reference-water impedance (ρ·c at Hamilton's
    # c_w/ρ_w); evaluate it there, then express α in dB/λ for the in-situ c.
    z_ref = (density_ratio * _HB_REF_RHOW) * (velocity_ratio * _HB_REF_CW)
    attenuation = _hamilton_kp(z_ref) * cp / 1000.0
    return cp, density_ratio * water_density, attenuation


# APL-UW TR 9407 (1994), §IV.A.4 "Model Input Parameters Using Grain Size"
# (valid −1 ≤ Mz ≤ 9): density ratio ρ₂/ρ₁ and sound-speed ratio c₂/c₁ as
# piecewise polynomials in Mz, plus the attenuation α₂/f (dB m⁻¹ kHz⁻¹). These
# are the formulas the AT ``'G'`` bottom used internally (ReadEnvironmentBell).
def _apl_density_ratio(mz: float) -> float:
    if mz < 1.0:
        return 0.007797 * mz ** 2 - 0.17057 * mz + 2.3139
    if mz < 5.3:
        return (-0.0165406 * mz ** 3 + 0.2290201 * mz ** 2
                - 1.1069031 * mz + 3.0455)
    return -0.0012973 * mz + 1.1565


def _apl_velocity_ratio(mz: float) -> float:
    if mz < 1.0:
        return 0.002709 * mz ** 2 - 0.056452 * mz + 1.2778
    if mz < 5.3:
        return (-0.0014881 * mz ** 3 + 0.0213937 * mz ** 2
                - 0.1382798 * mz + 1.3425)
    return -0.0024324 * mz + 1.0019


def _apl_alpha_over_f(mz: float) -> float:
    """APL-UW attenuation ``α₂/f`` in dB m⁻¹ kHz⁻¹ (peaks in fine sand).

    The final branch is unreachable through :func:`grain_size_to_geoacoustics`
    (ϕ is clamped to ≤ 9.0 by ``_MODEL_RANGE``); it is kept for fidelity to
    TR 9407 §IV.A.4.
    """
    if mz < 0.0:
        return 0.4556
    if mz < 2.6:
        return 0.4556 + 0.0245 * mz
    if mz < 4.5:
        return 0.1978 + 0.1245 * mz
    if mz < 6.0:
        return 8.0399 - 2.5228 * mz + 0.20098 * mz ** 2
    if mz < 9.5:
        return 0.9431 - 0.2041 * mz + 0.0117 * mz ** 2
    return 0.0601


def _apl_uw_geoacoustics(phi, water_sound_speed, water_density):
    """``(cp, density, attenuation)`` from the APL-UW TR 9407 relations."""
    cp = _apl_velocity_ratio(phi) * water_sound_speed
    # α(dB/λ) = (α₂/f)[dB/m/kHz]·c[m/s]/1000 — frequency cancels (α ∝ f).
    attenuation = _apl_alpha_over_f(phi) * cp / 1000.0
    return cp, _apl_density_ratio(phi) * water_density, attenuation


_GEOACOUSTIC_MODELS = {'hamilton': _hamilton_geoacoustics,
                       'apl-uw': _apl_uw_geoacoustics}


def grain_size_to_geoacoustics(
    grain_size_phi: float, *, model: str = 'hamilton',
    water_sound_speed: Optional[float] = None,
    water_density: Optional[float] = None,
) -> Dict[str, float]:
    """Map a mean grain size (Wentworth ϕ) to bulk geoacoustic properties.

    Returns ``{'sound_speed', 'density', 'attenuation'}`` (m/s, g/cm³,
    dB/wavelength). Sediment sound speed and density are computed as **ratios to
    the overlying seawater**, scaled by the in-situ ``water_sound_speed`` /
    ``water_density`` (so fine muds correctly come out *slower* than seawater,
    ratio < 1). Attenuation is returned in dB/wavelength, which is
    frequency-independent (``α[dB/λ] = (α/f)·c/1000``) and peaks in sand.

    Parameters
    ----------
    grain_size_phi : float
        Mean grain size on the Wentworth ϕ scale.
    model : {'hamilton', 'apl-uw'}, optional
        ``'hamilton'`` (default) — the **low-frequency** Hamilton & Bachman
        (1982) / Hamilton (1980) relations. ``'apl-uw'`` — the **high-frequency**
        APL-UW TR 9407 (1994) grain-size relations (ρ, ν polynomials + α₂/f).
    water_sound_speed, water_density : float, optional
        In-situ seawater sound speed (m/s) and density (g/cm³) the ratios are
        scaled by. ``None`` (default) uses the reference the chosen ``model``
        was tabulated against — Hamilton's 1510 m/s / 1.030 g/cm³, or APL-UW's
        1500 m/s / 1.0 g/cm³, which reproduces the Acoustics-Toolbox ``'G'``
        bottom exactly.

    A ``UserWarning`` is emitted when ``grain_size_phi`` is well outside the
    model's valid range (ϕ is then clamped).
    """
    if model not in _GEOACOUSTIC_MODELS:
        raise ConfigurationError(
            f"grain_size_to_geoacoustics: unknown model {model!r}.",
            remediation=f"Use one of {GRAIN_SIZE_MODELS}.",
        )
    ref_cw, ref_rhow = _MODEL_WATER_REFERENCE[model]
    if water_sound_speed is None:
        water_sound_speed = ref_cw
    if water_density is None:
        water_density = ref_rhow
    lo, hi = _MODEL_RANGE[model]
    phi = float(np.clip(grain_size_phi, lo, hi))
    if grain_size_phi < lo - 1.0 or grain_size_phi > hi + 1.0:
        warnings.warn(
            f"grain_size_to_geoacoustics: ϕ={grain_size_phi:g} is well outside "
            f"the {model} valid range [{lo:g}, {hi:g}]; clamped.",
            UserWarning, stacklevel=2,
        )
    cp, density, attenuation = _GEOACOUSTIC_MODELS[model](
        phi, water_sound_speed, water_density)
    return {'sound_speed': cp, 'density': density, 'attenuation': attenuation}
