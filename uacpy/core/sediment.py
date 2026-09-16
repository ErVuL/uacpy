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
  table + Hamilton (1972) ``k_p`` grain-size regressions (Geophysics 37, Fig. 3
  CC-BY ESAB supplement of Fonseca, Lurton, Fezzani & Roche (2025).
- ``'apl-uw'`` — APL-UW TR 9407 (1994) §IV.A.4 grain-size relations (the
  **high-frequency** ρ, ν polynomials + α₂/f). These are the same formulas the
  AT ``'G'`` bottom uses internally.

Both predate the NRL/APL revision of the grain-size algorithm (Briggs, Jackson
& Moravan, *NRL-APL Grain Size Algorithm Upgrade*, NRL/MR/7430--02-8274, 2002),
whose regressions are fitted to shelf surficial sediment and whose stated
objective is "to supplant the tabulated results in APL-UW-TR-9407". uacpy
implements TR 9407: on the revision's own 18-site comparison TR 9407 gives the
"best" fit at seven sites against the new regressions' three, and it is what
the AT ``'G'`` bottom uses. The revision reports the better fit to geoacoustic
*properties*, so it is the one to reach for when the properties rather than
the backscatter are the product.

Sediment sound speed and density are computed as **ratios to the overlying
seawater**, scaled by the in-situ water properties (so fine muds correctly come
out slower than seawater). Attenuation is returned in dB/wavelength, which is
frequency-independent (``α[dB/λ] = (α/f)·c/1000``).
"""

import warnings
from typing import Dict, Optional

import numpy as np

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._warn_frames import USER_FRAME_SKIP

__all__ = ['GRAIN_SIZE_MODELS', 'GRAIN_SIZE_MODEL_RANGES',
           'grain_size_to_geoacoustics']

_HB_REF_CW = 1510.0      # m/s   reference seawater sound speed
_HB_REF_RHOW = 1.030     # g/cm³ reference seawater density
# Hamilton & Bachman (1982), JASA 72(6), "Continental Terrace (Shelf and Slope)"
# granular sediments: mean grain size from their Table I, bulk density (g/cm³)
# and sound-speed ratio (sediment/seawater) from their Table II. Their Table II
# footnote recommends the *median* rather than the mean when predicting clayey
# silt, so that row carries 1.484 / 1.006.
# The ϕ axis is irregularly spaced because each value is the measured mean over
# that class's samples (Table I lists n = 2, 28, 16, 40, 47, 19, 29, 105, 54),
# not a Wentworth class boundary or centre. ``np.interp`` handles the uneven
# spacing; do not "regularise" it.
_HB_TABLE = (
    # (Mz_phi, density_gcm3, velocity_ratio)   coarse → fine
    (0.92, 2.034, 1.201),   # coarse sand
    (2.61, 1.962, 1.152),   # fine sand
    (3.34, 1.878, 1.120),   # very fine sand
    (4.24, 1.783, 1.086),   # silty sand
    (4.88, 1.769, 1.076),   # sandy silt
    (5.40, 1.740, 1.057),   # silt
    (5.82, 1.575, 1.036),   # sand-silt-clay
    (7.13, 1.484, 1.006),   # clayey silt (median, per their Table II footnote)
    (8.80, 1.480, 0.990),   # silty clay  (ratio < 1: slower than seawater)
)
_HB_PHI = np.array([r[0] for r in _HB_TABLE])
_HB_RHO = np.array([r[1] for r in _HB_TABLE])      # g/cm³ at the reference water
_HB_VRATIO = np.array([r[2] for r in _HB_TABLE])

GRAIN_SIZE_MODELS = ('hamilton', 'apl-uw')
# Seawater each model's ratios are referenced to, used when the caller gives no
# in-situ values. These are uacpy's in-situ defaults, not the conditions the
# tables were measured at: Hamilton & Bachman's ratios are laboratory values at
# 23 degC / 1 atm (their Table II footnote a), whose implied reference water is
# ~1527 m/s — Table II velocity divided by velocity ratio spans 1524.9 to 1532.3
# m/s over the nine rows, with a median of 1527.0 and both extremes at the fine
# end (clayey silt, silty clay). A ratio is tabulated precisely so it can be
# re-applied at in-situ conditions, which is what happens here;
# APL-UW's ratios are applied against 1500 m/s and a unit density ratio by the
# Acoustics-Toolbox 'G' bottom (Bellhop/ReadEnvironmentBell.f90:526
# `alphaR = vr * 1500.0`, and :531 `HS%rho = rhoR` — the ratio used directly as
# g/cm³).
_MODEL_WATER_REFERENCE = {'hamilton': (_HB_REF_CW, _HB_REF_RHOW),
                          'apl-uw': (1500.0, 1.0)}
_HAMILTON_KP_PHI_RANGE = (0.0, 9.5)
#: ``{model: (lo, hi)}`` — the ϕ interval each model's relations are fitted
#: over. :func:`grain_size_to_geoacoustics` clamps to the entry for the model
#: it is given, so this is the domain outside which its answer is the fit at
#: the nearer end rather than at the grain size asked for. It is public because
#: that is the question a caller holding a measured grain size has *before*
#: converting it — a seabed coarser than the fit is a sediment class, not a
#: grain-size relation — and the answer has to be read off the relations
#: themselves rather than copied into every provider that asks.
#:
#: Outside it ϕ is clamped, and the clamp is silent wherever it changes
#: nothing — on 'hamilton' it never can: its velocity and density are an
#: np.interp lookup that already holds the table's end rows (0.92 and 8.8 ϕ)
#: flat, and its attenuation is the Hamilton (1972) k_p regression, which holds
#: its own limits, 0 and 9.5 ϕ. The model range is the wider of the two, so the
#: clamp lands exactly where each part already stops — and warns whenever it
#: moves the returned values, which is 'apl-uw' only.
GRAIN_SIZE_MODEL_RANGES = {'hamilton': _HAMILTON_KP_PHI_RANGE,
                           'apl-uw': (-1.0, 9.0)}


def _hamilton_kp(phi: float) -> float:
    """Hamilton (1972) attenuation constant ``k_p`` versus mean grain size.

    ``α(dB/m) = k_p · f(kHz)``. The four regressions are the Fig. 3 caption of
    Hamilton, "Compressional-wave attenuation in marine sediments",
    Geophysics 37 (1972), p. 636 (``external:hamilton1972.pdf`` page 17),
    "recommended only within the limiting values" 0 to 9.5 ϕ
    (``_HAMILTON_KP_PHI_RANGE``), so ϕ is held at those limits outside them. The branches meet within 0.003 at 2.6, 4.5 and
    6.0 ϕ; the peak is 0.758 at 4.5 ϕ — coarse silt in TR 9407 Table 2, and
    between ``_HB_TABLE``'s silty sand (4.24 ϕ) and sandy silt (4.88 ϕ) — and
    the fine end is 0.054 at 9 ϕ, 0.060 at the 9.5 ϕ limit.
    """
    m = min(max(float(phi), _HAMILTON_KP_PHI_RANGE[0]), _HAMILTON_KP_PHI_RANGE[1])
    if m <= 2.6:
        return 0.4556 + 0.0245 * m
    if m <= 4.5:
        return 0.1978 + 0.1245 * m
    if m <= 6.0:
        return 8.0399 - 2.5228 * m + 0.20098 * m * m
    return 0.9431 - 0.2041 * m + 0.0117 * m * m


def _hamilton_geoacoustics(phi, water_sound_speed, water_density):
    """``(cp, density, attenuation)`` from the Hamilton & Bachman ϕ-table."""
    density_ratio = float(np.interp(phi, _HB_PHI, _HB_RHO)) / _HB_REF_RHOW
    velocity_ratio = float(np.interp(phi, _HB_PHI, _HB_VRATIO))
    cp = velocity_ratio * water_sound_speed
    # α(dB/λ) = k_p · f(kHz) · λ = k_p · c / 1000: frequency drops out.
    attenuation = _hamilton_kp(phi) * cp / 1000.0
    return cp, density_ratio * water_density, attenuation


# APL-UW TR 9407 (1994), §IV.A.4 "Model Input Parameters Using Grain Size"
# (valid −1 ≤ Mz ≤ 9): density ratio ρ₂/ρ₁ and sound-speed ratio c₂/c₁ as
# piecewise polynomials in Mz, plus the attenuation α₂/f (dB m⁻¹ kHz⁻¹). These
# are the formulas the AT ``'G'`` bottom uses internally
# (Bellhop/ReadEnvironmentBell.f90:488 `CASE ( 'G' )`).
# Verified against that Fortran: the velocity-ratio, density-ratio and
# alpha2_f branches reproduce ``ReadEnvironmentBell.f90:497-520`` to 0.0e+00
# over Mz in [-1, 9], and the dB/wavelength attenuation below is AT's 'L' loss
# parameter times exactly 40*pi/ln(10) = 54.575054 (``AttenMod.f90:79-80``).
# That dB/λ scales by the sediment's own sound speed, as AT's 'W' unit does
# (``AttenMod.f90:72-73``, ``alphaT = alpha * freq / ( 8.6858896D0 * c )``), so
# the water speed enters only through c. TR 9407 Eq. 4 instead carries an
# explicit c₁ — the water sound speed, p. IV-8 "All subscripts 1 refer to
# water" — and the report states it used c₁ = 1.528 m/ms in determining its own
# loss parameter δ (p. IV-9); that value reproduces Table 2's printed δ column
# at all eight rows checked (Mz = -1, -0.5, 0, 0.5, 2.5, 5.5, 8, 9), where
# 1.500 reproduces none. ``water_sound_speed`` here is the in-situ speed AT's
# own source comment asks for at its hard-coded 1500
# (``ReadEnvironmentBell.f90:523-524``: "should be sound speed in the water,
# just above the sediment"), so at the 1500 m/s default this α runs
# 1 - 1500/1528 = 1.83 % below the report's tabulated δ, which is the same
# statement as the report's δ running 1528/1500 - 1 = 1.87 % above this one.
# :mod:`uacpy.sonar.bottom_scattering` keeps the report's constant and
# reproduces the table.
# One deliberate difference: AT guards its first branch with ``Mz >= -1``, so
# below -1 it falls through to the LINEAR branch, whereas the ratios here keep
# the quadratic. ``GRAIN_SIZE_MODEL_RANGES['apl-uw']`` clamps to [-1, 9]
# before either is called, so the public API never reaches that region.
# A second, equally deliberate difference, at the other end: AT does not clamp
# at all. ``ReadEnvironmentBell.f90``'s velocity ratio ends in a bare
# ``ELSE vr = -0.0024324*Mz + 1.0019`` that keeps running above Mz = 9 with no
# upper limit, and its ``alpha2_f`` takes 0.0601 above 9.5. uacpy holds the fit
# flat at the Wentworth bounds instead — ϕ = -1 is 2 mm gravel, ϕ = 9 is ~2 µm
# clay — because a cubic left to run past its last fitted point diverges
# (the -0.0165406 Mz³ density term turns over), whereas a flat endpoint stays
# physical. ``grain_size_to_geoacoustics`` warns when that clamp moves the
# answer, so the divergence is visible at the call site rather than only here.
# The range is the report's own: TR 9407 §IV.A.4 "Model Input Parameters
# Using Grain Size", which opens on p. IV-7, states on p. IV-8 that relations
# (2)-(5) "are defined only for -1 <= Mz <= 9", which is the interval the clamp
# holds the fits to. The polynomial coefficients themselves are verified
# against the AT transcription above; the report's equation blocks are not
# decoded in the parsed copy, and the fifth digit of the velocity ratio's third
# branch is indistinct in the scan even at 1200 dpi. -0.0024324 is the value
# that reproduces Table 2's printed nu column at all seven Mz >= 5.3 rows to
# four decimals (-0.0024224 reproduces two of the seven), and it is what
# ``ReadEnvironmentBell.f90:504`` uses.
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
    """APL-UW attenuation ``α₂/f`` in dB m⁻¹ kHz⁻¹ (peaks at 4.5 ϕ, coarse silt).

    The final branch is unreachable through :func:`grain_size_to_geoacoustics`
    (ϕ is clamped to ≤ 9.0 by ``GRAIN_SIZE_MODEL_RANGES``); it is kept for
    fidelity to TR 9407 §IV.A.4.
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
    frequency-independent (``α[dB/λ] = (α/f)·c/1000``) and peaks at 4.5 ϕ —
    coarse silt — for both models.

    Parameters
    ----------
    grain_size_phi : float
        Mean grain size on the Wentworth ϕ scale.
    model : {'hamilton', 'apl-uw'}, optional
        ``'hamilton'`` (default) — the **low-frequency** Hamilton & Bachman
        (1982) / Hamilton (1972) relations. ``'apl-uw'`` — the **high-frequency**
        APL-UW TR 9407 (1994) grain-size relations (ρ, ν polynomials + α₂/f).
    water_sound_speed, water_density : float, optional
        In-situ seawater sound speed (m/s) and density (g/cm³) the ratios are
        scaled by. ``None`` (default) uses the reference the chosen ``model``
        was tabulated against — Hamilton's 1510 m/s / 1.030 g/cm³, or APL-UW's
        1500 m/s / 1.0 g/cm³, which reproduces the Acoustics-Toolbox ``'G'``
        bottom exactly.

    ``grain_size_phi`` outside the model's ϕ range is clamped to it. A
    ``UserWarning`` is emitted exactly when that clamp changes the returned
    values, so the warning marks a real substitution rather than a boundary
    crossing. It fires for ``'apl-uw'`` at every ϕ outside ``[-1, 9]``,
    including ±inf, whose polynomials do keep extrapolating (ϕ = 9.5 differs by
    1.8 m/s, ϕ = -1.5 by 47 m/s). It never fires for ``'hamilton'``: its range
    is the 0 to 9.5 ϕ Hamilton (1972) recommends for ``k_p``, and inside it the
    velocity / density table already holds its end rows (0.92 and 8.8 ϕ) flat
    through ``np.interp``, so the clamp cannot move the result.

    What ``'hamilton'`` returns outside its range is therefore that end row
    itself, unannounced — any ϕ below 0 gives the 0.92 ϕ coarse-sand row of
    ``_HB_TABLE`` with ``k_p(0)``, and any ϕ above 9.5 the 8.8 ϕ silty-clay row
    with ``k_p(9.5)``. A grain size coarser than 0 ϕ — gravel, for one — wants
    ``'apl-uw'``, which is fitted to -1 ϕ and reports the substitution beyond
    it, or a sediment class from :mod:`uacpy.core.materials`. A NaN ϕ
    propagates to NaN outputs under both models and does not warn, there being
    no substituted value to report.
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
    lo, hi = GRAIN_SIZE_MODEL_RANGES[model]
    phi = float(np.clip(grain_size_phi, lo, hi))
    cp, density, attenuation = _GEOACOUSTIC_MODELS[model](
        phi, water_sound_speed, water_density)
    # Warn on the substitution, not on the boundary crossing: evaluate the fit
    # at the raw phi and compare. A deadband keyed on phi alone cannot tell the
    # two apart -- for 'hamilton' the clamp is a provable no-op at any phi
    # (np.interp and k_p both hold their own limits), so a deadband there cries
    # wolf, while for 'apl-uw' a deadband stays silent on a real change once
    # phi is past the band. The extra evaluation costs one call, and only when
    # the clamp engaged. NaN is excluded because it propagates to NaN outputs,
    # so there is nothing substituted to report, while comparing unequal to its
    # own clip; +-inf is included, being the largest substitution the clamp can
    # make.
    if phi != grain_size_phi and not np.isnan(grain_size_phi):
        raw = _GEOACOUSTIC_MODELS[model](
            float(grain_size_phi), water_sound_speed, water_density)
        if raw != (cp, density, attenuation):
            warnings.warn(
                f"grain_size_to_geoacoustics: ϕ={grain_size_phi:g} is outside "
                f"the {model} valid range [{lo:g}, {hi:g}] and was clamped to "
                f"ϕ={phi:g}; the unclamped fit gives "
                f"sound_speed={raw[0]:.4f} m/s, density={raw[1]:.4f} g/cm³, "
                f"attenuation={raw[2]:.4f} dB/λ.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
    return {'sound_speed': cp, 'density': density, 'attenuation': attenuation}
