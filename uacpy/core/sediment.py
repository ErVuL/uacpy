"""Grain size (Wentworth ϕ) → bulk geoacoustic properties.

Pure, dependency-light conversion shared by the carrier layer
(:meth:`uacpy.core.bottom.BoundaryProperties.from_grain_size`) and the on-demand
data layer (:mod:`uacpy.data.sediment`). It lives in ``core`` so a bottom can be
built from a grain size without importing ``uacpy.data``.

Grain size is a **construction-time input only**: it is converted here to an
explicit half-space (``sound_speed`` / ``density`` / ``attenuation``) that every
propagation model consumes. There is no ``'grain-size'`` boundary *type* — no
model sees a grain size at run time.

Two models are provided, and they are not independent:

- ``'hamilton'`` (default) — the Hamilton & Bachman (1982) continental-terrace
  (T) regressions for velocity and density (their Appendix, p. 1902) +
  Hamilton (1972) ``k_p`` for attenuation (Geophysics 37, Fig. 3). Their
  class-mean table is kept beside them as ``_HB_TABLE``: the paper prescribes
  the tables for a sediment *type* and the equations for a mean grain *size*
  (p. 1892), and a grain size is what this function is given.
- ``'apl-uw'`` — APL-UW TR 9407 (1994) §IV.A.4 grain-size relations (ρ, ν
  polynomials + α₂/f). These are the same formulas the AT ``'G'`` bottom uses
  internally.

**The attenuation is one function.** TR 9407 p. IV-8 says so — "Hamilton's
parameterization of α₂/f is reproduced below" — citing the same Hamilton (1972)
paper ``_hamilton_kp`` implements — so both models call that one function and
there is no second copy to drift from it. **Below 1 ϕ the ρ and ν are one function too**: TR 9407 p. IV-8,
"The density and sound speed ratios agree with those of Hamilton and Bachman for
-1 <= M_z < 1", its coarse branch being that regression over c₁ = 1528 m/s and
ρ₁ = 1.026 g/cm³ (residual 1.5e-05 in ν, 6.4e-05 in ρ across that band).

What differs, and only on 1 to 9 ϕ, is **the depth of sediment each describes**.
TR 9407 fits "the upper few centimeters of the sediment" (p. IV-7) with ratios
it calls "surficial values derived from model fitting" that "tend to be smaller
than the bulk values reported in the literature" (p. IV-5); Hamilton & Bachman
measured samples from "the upper 30 cm of the sea floor" (p. 1891).
Neither source states a frequency rule for choosing between the two: TR 9407
declares the *scattering* models its ratios feed "applicable over the frequency
interval 10-100 kHz", which is a statement about those models, not about the
grain-size relations; Hamilton's velocities were measured at ~200 kHz and his
``k_p`` at 14, 25 and 100 kHz. **A third source states one.** Ainslie,
*Principles of Sonar Performance Modelling* (2010) Sec. 4.4.1, tabulates both
sets and attaches a band and a depth to each: his Table 4.17 is "Default HF
geo-acoustic parameters (10-100 kHz). Near-surface sediment properties",
APL-UW's, "representative of the average properties of the top few
centimeters"; his Table 4.18 is "Default MF geo-acoustic parameters
(1-10 kHz). Bulk sediment properties", Hamilton's and Bachman's,
"representative of the uppermost few meters of sediment". That is this
module's ``model=`` selector with a frequency written against each arm:
``'apl-uw'`` for a high-frequency sonar reading the first few centimetres,
``'hamilton'`` for a
kilohertz problem whose field penetrates metres. uacpy does not apply the rule
itself, because a grain-size conversion is not given a frequency.

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
from functools import partial
from typing import Dict, Optional

import numpy as np

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._warn_frames import USER_FRAME_SKIP

__all__ = ['DEFAULT_GRAIN_SIZE_ENVIRONMENT', 'DEFAULT_GRAIN_SIZE_MODEL',
           'GRAIN_SIZE_ENVIRONMENTS',
           'GRAIN_SIZE_MODELS', 'GRAIN_SIZE_MODEL_RANGES',
           'GRAIN_SIZE_SOURCE_RANGES', 'check_grain_size_selection',
           'grain_size_from_density', 'grain_size_to_geoacoustics']

_HB_REF_CW = 1510.0      # m/s   reference seawater sound speed
_HB_REF_RHOW = 1.030     # g/cm³ reference seawater density
# Hamilton & Bachman (1982), JASA 72(6), "Continental Terrace (Shelf and Slope)"
# granular sediments: mean grain size from their Table I, bulk density (g/cm³)
# and sound-speed ratio (sediment/seawater) from their Table II. Their Table II
# footnote recommends the *median* rather than the mean when predicting clayey
# silt, so that row carries 1.484 / 1.006.
# The ϕ axis is irregularly spaced because each value is the measured mean over
# that class's samples (Table I lists n = 2, 28, 16, 40, 47, 19, 29, 105, 54),
# not a Wentworth class boundary or centre; do not "regularise" it.
#
# This is the **sediment-type** route, and the paper says which route to take,
# p. 1892: "If sediment type and environment are the only information available,
# then data from the tables can be used in predictions. ... When additional
# information such as mean grain size, density, or porosity are available, then
# one can enter the regression equations for the particular environment."
# ``grain_size_to_geoacoustics`` is handed a mean grain size, so it evaluates
# the regressions below. The table stays as the data it is: the class means the
# regressions were fitted to, the per-class sample counts, the clayey-silt
# median, and the check that the fit reproduces its own class means
# (``test_sediment.py``). ``uacpy.data.graw_local`` inverts it for ρ → ϕ.
_HB_TABLE = (
    # (Mz_phi, density_gcm3, velocity_ratio)   coarse → fine
    (0.92, 2.034, 1.201),   # coarse sand
    (2.61, 1.962, 1.152),   # fine sand
    (3.34, 1.878, 1.120),   # very fine sand
    (4.24, 1.783, 1.086),   # silty sand
    (4.88, 1.769, 1.076),   # sandy silt
    (5.40, 1.740, 1.057),   # silt
    (5.82, 1.575, 1.036),   # sand-silt-clay
    # Clayey silt carries the *median*, which their Table II footnote
    # recommends for this one class ("These mean and median values were
    # insignificantly different save in one case: clayey silt in the
    # continental terrace", p. 1892). That recommendation is attached to the
    # table, i.e. to the type route; the (T) regression is a least-squares fit
    # and tracks the mean, so it runs 10.6 m/s faster here than this row. Both
    # are Hamilton's, for the two different questions he distinguishes.
    (7.13, 1.484, 1.006),   # clayey silt (median, per their Table II footnote)
    (8.80, 1.480, 0.990),   # silty clay  (ratio < 1: slower than seawater)
)
_HB_PHI = np.array([r[0] for r in _HB_TABLE])
_HB_RHO = np.array([r[1] for r in _HB_TABLE])      # g/cm³ at the reference water
_HB_VRATIO = np.array([r[2] for r in _HB_TABLE])

# Hamilton & Bachman (1982) Appendix, p. 1902 — read from
# ``external:hamilton1982.pdf`` page 12 at 500 dpi, the equation blocks being
# undecoded in the parsed copy. The Appendix gives a separate fit for each of
# three environments, in its own words: "Separate equations are listed, where
# appropriate, for each of the three general environments as follows:
# continental terrace (shelf and slope) (T); abyssal hill (pelagic) (H);
# abyssal plain (turbidite) (P)."
#
#   "Sound velocity, V_p (m/s) versus mean grain size, M_z (ϕ), Figs. 10 and 11"
#       (T) V_p = 1952.5 - 86.26 M_z + 4.14 M_z²      σ = 29
#       (H) V_p = 1594.3 - 10.2 M_z                   σ = 12
#       (P) V_p = 1609.7 - 10.8 M_z                   σ = 11
#   "Density, ρ (g/cm³) versus mean grain size, M_z (ϕ), Fig. 5"
#       (T) ρ   = 2.374  - 0.175 M_z + 0.008 M_z²     σ = 0.11
#       (H) ρ   = 1.327  + 0.005 M_z                  σ = 0.09
#       (P) ρ   = 1.869  - 0.057 M_z                  σ = 0.11
#   "The limiting values of (x) in the equations below are (1) Mean grain
#    diameter, M_z, ϕ  (T) 1 to 9 ϕ  (H) and (P) 7 to 10 ϕ"
#
# The two abyssal families are straight lines over a deep-water fine-sediment
# band: they do not describe sand at all, which is why the selector clamps and
# reports rather than running them coarse of 7 ϕ.
_HB_T_VELOCITY = (1952.5, -86.26, 4.14)      # V_p, m/s, ascending powers of Mz
_HB_T_DENSITY = (2.374, -0.175, 0.008)       # ρ, g/cm³
# Both are absolute laboratory values at 23 °C / 1 atm (p. 1892), so uacpy needs
# a divisor to reach the ratio it stores. Hamilton publishes none — his tabulated
# ratios use each site's own pore-water speed (Table II footnote a), which is why
# the ratios Table II implies span 1524.85 to 1532.32 m/s. TR 9407 does publish
# one, for this exact conversion, and its own coarse branch is these two
# polynomials rescaled: p. IV-8, "The density and sound speed ratios agree with
# those of Hamilton and Bachman for -1 <= M_z < 1", and "c₁ = 1.528 m/ms is used
# in determining δ from Hamilton's expressions". Coefficient by coefficient
# against ``_apl_velocity_ratio`` and ``_apl_density_ratio``'s first branches:
#   1952.5/1.2778 = 1528.017   -86.26/-0.056452 = 1528.024   4.14/0.002709 = 1528.239
#   2.374/2.3139  = 1.02597    -0.175/-0.17057  = 1.02597    0.008/0.007797 = 1.02604
# Three ratios landing on one constant each is what makes the divisor a citation
# rather than a fitted choice. c₁ = 1528.0 m/s is TR 9407's printed value; the
# density divisor is not printed there and is recovered from those ratios to six
# significant figures, so it is written as 1.026 g/cm³.
_HB_T_REF_CW = 1528.0
_HB_T_REF_RHOW = 1.026
# The (T) regressions' own domain, p. 1902. The coarse half of the interval
# uacpy evaluates over rests on a different document: see
# ``GRAIN_SIZE_SOURCE_RANGES``.
_HB_T_PHI_RANGE = (1.0, 9.0)
#: Hamilton & Bachman's three environments, ``{name: {...}}``. Each carries its
#: own velocity and density coefficients (ascending powers of Mz), the standard
#: errors of estimate printed beside them, the ϕ interval that paper declares
#: for it, and the interval uacpy evaluates it over.
#:
#: The two differ only for the continental terrace, whose coarse end reaches
#: -1 ϕ on TR 9407's authority rather than Hamilton's (p. IV-8, its branch being
#: the (T) polynomials rescaled). Nothing extends the abyssal families: they are
#: straight lines fitted over 7 to 10 ϕ and describe no sand.
#:
#: The **attenuation is not split by environment** — ``_hamilton_kp`` is
#: Hamilton (1972), whose Fig. 3 regressions are "for data off San Diego and
#: selected literature values recommended for use in similar sediments" — so
#: all three return the same ``k_p``, over its own 0 to 9.5 ϕ.
GRAIN_SIZE_ENVIRONMENTS = {
    'continental-terrace': {
        'velocity': _HB_T_VELOCITY, 'sigma_velocity': 29.0,
        'density': _HB_T_DENSITY, 'sigma_density': 0.11,
        'published_range': _HB_T_PHI_RANGE, 'evaluated_range': (-1.0, 9.0),
    },
    'abyssal-hill': {
        'velocity': (1594.3, -10.2), 'sigma_velocity': 12.0,
        'density': (1.327, 0.005), 'sigma_density': 0.09,
        'published_range': (7.0, 10.0), 'evaluated_range': (7.0, 10.0),
    },
    'abyssal-plain': {
        'velocity': (1609.7, -10.8), 'sigma_velocity': 11.0,
        'density': (1.869, -0.057), 'sigma_density': 0.11,
        'published_range': (7.0, 10.0), 'evaluated_range': (7.0, 10.0),
    },
}
#: The environment ``'hamilton'`` uses when the caller names none. It is the
#: shelf-and-slope fit, which is what every uacpy seabed source has always got.
DEFAULT_GRAIN_SIZE_ENVIRONMENT = 'continental-terrace'

GRAIN_SIZE_MODELS = ('hamilton', 'apl-uw')
#: The model every uacpy entry point converts a grain size with when the caller
#: names none. It is ``'hamilton'``, and the reason is the band-and-depth rule
#: at the top of this module: the bulk (MF) relations describe "the uppermost
#: few meters of sediment" over 1-10 kHz, and a propagation model is what
#: uacpy's seabeds are built for -- at those frequencies the field penetrates
#: metres, so the metres are what it must see. ``'apl-uw'``'s near-surface (HF)
#: relations describe the top few centimetres over 10-100 kHz; pass
#: ``model='apl-uw'`` for a high-frequency problem, which is also what makes a
#: uacpy seabed identical to the Acoustics Toolbox's own ``'G'`` bottom.
#:
#: Two things this default does NOT decide. It does not reach
#: :mod:`uacpy.sonar.bottom_scattering`, whose APL-UW backscattering model is a
#: 10-100 kHz model and evaluates the APL-UW relations by construction; and it
#: is not applied from a frequency anywhere, because a grain-size conversion is
#: never given one. A caller working above 10 kHz has to say so.
DEFAULT_GRAIN_SIZE_MODEL = 'hamilton'
# Seawater each model's ratios are referenced to, used when the caller gives no
# in-situ values. These are uacpy's in-situ defaults, not the conditions the
# relations were measured at: Hamilton & Bachman's are laboratory values at
# 23 degC / 1 atm (their Table II footnote a), and the ratio uacpy forms from
# their regressions is against ``_HB_T_REF_CW`` = 1528 m/s. Their *tabulated*
# ratios are against no single speed at all — Table II velocity divided by
# velocity ratio spans 1524.9 to 1532.3 m/s over the nine rows, which is the
# per-site pore-water salinity the footnote describes, and 1528 sits 0.11 m/s
# from the mean of it. A ratio is formed precisely so it can be re-applied at
# in-situ conditions, which is what happens here;
# APL-UW's ratios are applied against 1500 m/s and a unit density ratio by the
# Acoustics-Toolbox 'G' bottom (Bellhop/ReadEnvironmentBell.f90:526
# `alphaR = vr * 1500.0`, and :531 `HS%rho = rhoR` — the ratio used directly as
# g/cm³).
_MODEL_WATER_REFERENCE = {'hamilton': (_HB_REF_CW, _HB_REF_RHOW),
                          'apl-uw': (1500.0, 1.0)}
_HAMILTON_KP_PHI_RANGE = (0.0, 9.5)
# TR 9407 §IV.A.4: relations (2)-(5) "are defined only for -1 <= Mz <= 9"
# (p. IV-8). One equation set, so all three quantities share one domain.
_APL_UW_PHI_RANGE = (-1.0, 9.0)
# uacpy evaluates the (T) regressions over -1 to 9 ϕ, and the two halves of that
# interval rest on two different documents. 1 to 9 ϕ is Hamilton & Bachman's own
# limit (p. 1902). Coarse of 1 ϕ the authority is TR 9407, which states these
# same polynomials as its own branch over -1 <= M_z < 1 (p. IV-8, and the
# coefficient arithmetic above) and declares -1 its floor. Past either end the
# quadratics are not evaluated: ϕ is held at the edge and the call says so.
_HB_T_EVALUATED_RANGE = (_APL_UW_PHI_RANGE[0], _HB_T_PHI_RANGE[1])
GRAIN_SIZE_ENVIRONMENTS['continental-terrace']['evaluated_range'] = (
    _HB_T_EVALUATED_RANGE)


def _source_ranges(model: str, environment: str) -> Dict[str, tuple]:
    """``{quantity: (lo, hi)}`` for one model and, for ``'hamilton'``, one
    environment. :data:`GRAIN_SIZE_SOURCE_RANGES` is this for the default."""
    if model != 'hamilton':
        return GRAIN_SIZE_SOURCE_RANGES[model]
    span = GRAIN_SIZE_ENVIRONMENTS[environment]['evaluated_range']
    return {'sound_speed': span, 'density': span,
            'attenuation': _HAMILTON_KP_PHI_RANGE}

#: ``{model: {quantity: (lo, hi)}}`` — the ϕ interval over which each returned
#: quantity's **own source** holds data. A validity range belongs to a source,
#: not to a model, and ``'hamilton'`` is two sources stitched together: its
#: ``sound_speed`` and ``density`` are the Hamilton & Bachman (1982) (T)
#: regressions over -1 to 9 ϕ (1 to 9 on their own authority, coarse of 1 on
#: TR 9407's — see ``_HB_T_EVALUATED_RANGE``), while its ``attenuation`` is the
#: Hamilton (1972) ``k_p`` regression over the 0 to 9.5 ϕ that paper recommends.
#: So at ϕ = 9.2 the honest answer is not "inside the model" or "outside" it but
#: "attenuation evaluated, velocity and density held at 9 ϕ", which is what
#: :func:`grain_size_to_geoacoustics` reports.
#:
#: ``'hamilton'``'s entry is the **default environment's**. The abyssal fits
#: carry their own 7 to 10 ϕ, which :data:`GRAIN_SIZE_ENVIRONMENTS` holds and
#: the conversion honours when it is given one.
GRAIN_SIZE_SOURCE_RANGES = {
    'hamilton': {'sound_speed': _HB_T_EVALUATED_RANGE,
                 'density': _HB_T_EVALUATED_RANGE,
                 'attenuation': _HAMILTON_KP_PHI_RANGE},
    'apl-uw': {'sound_speed': _APL_UW_PHI_RANGE,
               'density': _APL_UW_PHI_RANGE,
               'attenuation': _APL_UW_PHI_RANGE},
}

#: ``{model: (lo, hi)}`` — the ϕ interval :func:`grain_size_to_geoacoustics`
#: clamps to, which is the **union** of that model's source ranges above:
#: outside it every quantity is an endpoint, inside it at least one still
#: tracks ϕ. Derived from those ranges rather than restated, so the clamp and
#: the data cannot drift apart.
#:
#: It is public because that is the question a caller holding a measured grain
#: size has *before* converting it — a seabed coarser than the fits is a
#: sediment class, not a grain-size relation — and the answer has to be read off
#: the relations themselves rather than copied into every provider that asks.
#: Being a union, it is the *weaker* of the two published tests: a ϕ inside it
#: can still be outside a particular quantity's source, which is why the
#: per-quantity ranges above are published beside it.
GRAIN_SIZE_MODEL_RANGES = {
    model: (min(lo for lo, _ in quantities.values()),
            max(hi for _, hi in quantities.values()))
    for model, quantities in GRAIN_SIZE_SOURCE_RANGES.items()
}


def _hamilton_kp(phi: float) -> float:
    """Hamilton (1972) attenuation constant ``k_p`` versus mean grain size.

    ``α(dB/m) = k_p · f(kHz)``. The four regressions are the Fig. 3 caption of
    Hamilton, "Compressional-wave attenuation in marine sediments", Geophysics
    37 (1972), p. 636 (``external:hamilton1972.pdf`` pages 16-17), which names
    the sediments each was fitted to: coarse, medium and fine sand in part
    (0 to 2.6 ϕ); fine sand in part, very fine sand and mixed sizes (2.6 to
    4.5); mixed sizes (4.5 to 6.0); silt-clays (6.0 to 9.5). They are "strictly
    empirical and are recommended only within the limiting values indicated"
    (p. 635) — the 0 to 9.5 ϕ of ``_HAMILTON_KP_PHI_RANGE`` — so ϕ is held at
    those limits outside them.

    **Both models return this one function**, which is why it is written once:
    TR 9407 p. IV-8 says "Hamilton's parameterization of α₂/f is reproduced
    below" and cites this paper, and ``α₂/f`` is this ``k_p`` under the other
    report's name, in the same dB m⁻¹ kHz⁻¹. What used to be a second copy
    differed from this one in three places, all of them presentation:

    * its coarse constant, ``Mz < 0 → 0.4556``, which is ``k_p(0)`` exactly;
    * its tail, ``Mz >= 9.5 → 0.0601``, which is ``k_p(9.5) = 0.060075``
      printed to four decimals, on a branch neither model can reach (each
      clamps ϕ below 9.5 first);
    * the **tie at a branch join**, where it tested ``<`` and this tested
      ``<=``, so at exactly 2.6, 4.5 or 6.0 ϕ the two returned different
      values from identical coefficients — 0.5193 against 0.5215 at 2.6 ϕ.

    Hamilton's caption gives the ranges as "0 to 2.6ϕ", "2.6 to 4.5ϕ" and so
    on, sharing each endpoint between two branches and settling nothing. The
    Acoustics-Toolbox does settle it, and this follows it:
    ``ReadEnvironmentBell.f90:509-518`` reads
    ``ELSE IF( Mz >= 2.6 .AND. Mz < 4.5 )``, so a join belongs to the branch
    **above** it. ``model='apl-uw'`` has to match that binary, and now
    ``'hamilton'`` cannot drift from it because there is nothing to drift from.

    The four regressions are genuinely discontinuous at their joins — that is
    the published fit, not a seam in this code — but they meet within 0.003.
    The peak is 0.7571 at exactly 4.5 ϕ (0.75805 as the branch below it
    approaches 4.5) — coarse silt in TR 9407 Table 2, and between
    ``_HB_TABLE``'s silty sand (4.24 ϕ) and sandy silt (4.88 ϕ) — and the fine
    end is 0.054 at 9 ϕ, 0.060 at the 9.5 ϕ limit.

    Hamilton (1980) p. 1328 states what this grain-size route is worth against
    the porosity route he prefers — "the equations relating mean grain size and
    k_p ... will frequently yield lower values of k_p than will porosity for
    the same sediment. Thus, the expectable lower limit of k_p in
    higher-porosity sediments in the mean-grain-size versus k_p diagram
    (Fig. 19) is about 0.03." He expects "most future surficial data for
    high-porosity, deep-sea sediments will fall into the lower part of the
    range, between 0.03 and 0.10", and says predictions "have to include a
    probable range based on the dashed lines above and below the regression
    curves" — which is why his model tables carry three values under ``k_p``,
    probable minimum, recommended centre, probable maximum. This returns the
    centre; the envelopes are drawn in his figures and nowhere printed as a
    number. His Fig. 19 (p. 1329) is these four regressions replotted,
    "expressed as k_p, defined in Fig. 18".
    """
    m = min(max(float(phi), _HAMILTON_KP_PHI_RANGE[0]), _HAMILTON_KP_PHI_RANGE[1])
    if m < 2.6:
        return 0.4556 + 0.0245 * m
    if m < 4.5:
        return 0.1978 + 0.1245 * m
    if m < 6.0:
        return 8.0399 - 2.5228 * m + 0.20098 * m * m
    return 0.9431 - 0.2041 * m + 0.0117 * m * m


def check_grain_size_selection(model: str, environment: str, *,
                               caller: str = 'grain_size',
                               model_argument: str = 'model',
                               environment_argument: str = 'environment'
                               ) -> None:
    """Raise :class:`ConfigurationError` for a ``model`` / ``environment`` pair
    the sources do not publish.

    Every entry point that accepts the pair calls this, not only the innermost
    one: a caller who names an abyssal environment alongside
    ``model='apl-uw'`` several layers up would otherwise have it quietly
    dropped by a wrapper that never reaches the conversion — a seabed built
    from a class name or a literal ϕ, say. The mismatch is refused where it is
    typed.
    """
    if model not in GRAIN_SIZE_MODELS:
        raise ConfigurationError(
            f"{caller}: unknown {model_argument} {model!r}.",
            remediation=f"Use one of {GRAIN_SIZE_MODELS}.",
        )
    if environment not in GRAIN_SIZE_ENVIRONMENTS:
        raise ConfigurationError(
            f"{caller}: unknown {environment_argument} {environment!r}.",
            remediation=f"Use one of {tuple(GRAIN_SIZE_ENVIRONMENTS)}.",
        )
    if model != 'hamilton' and environment != DEFAULT_GRAIN_SIZE_ENVIRONMENT:
        raise ConfigurationError(
            f"{caller}: {model_argument}={model!r} has no {environment!r} form — "
            f"TR 9407 "
            f"publishes one set of relations, not one per environment.",
            remediation="Use model=DEFAULT_GRAIN_SIZE_MODEL for the abyssal fits, or drop "
                        "the environment.",
        )


def grain_size_from_density(
    density: float, water_density: Optional[float] = None,
    environment: str = DEFAULT_GRAIN_SIZE_ENVIRONMENT,
) -> float:
    """Mean grain size (ϕ) from a measured bulk density — the inverse of the
    ``'hamilton'`` density relation :func:`grain_size_to_geoacoustics` applies.

    ``density`` is in g/cm³ against ``water_density`` (default: the same
    in-situ reference the forward call uses), and ``environment`` selects the
    same fit the forward direction would use, so the two are inverses of each
    other whichever one is named.

    A density the chosen fit cannot represent is **announced and held at the
    nearer end**, like every other substitution here. That is not a rare edge:
    the continental-terrace quadratic bottoms out at 1.417 g/cm³ (1.423 against
    the default water) because its vertex sits at ϕ = 10.94, past the range it
    is evaluated over — and **45.5 % of the Graw grid's ocean cells are below
    it**, measured over its 6.2 million finite cells. Those cells are not
    artefacts: their bulk sits at 1.30-1.42 g/cm³, which is what Hamilton &
    Bachman's own Table IV measures for abyssal clay (1.352 and 1.414), and
    only 0.12 % of cells fall below 1.2. They are ordinary deep-ocean mud,
    which is to say **they are not continental terrace** — the abyssal-plain
    fit represents 65.3 % of the grid where the terrace fit reaches 40.8 %.
    So the warning names the environment whose range would cover the value.

    **What this inverse is, and what its own sources say about it.** Hamilton &
    Bachman's Appendix is directional by construction -- its equations
    "constitute the best indices (x) to obtain the desired properties (y)"
    (p. 1902), it
    lists no equation with mean grain size as the y, and it asks that each be
    "used only between the limiting values of the index property", which for
    (T) is 1 to 9 phi. Bachman's own later advice is stronger, as Ainslie
    relays it (*Principles of Sonar Performance Modelling* p. 177): the fit
    "is intended for computing c or rho, given the grain size M_z. Bachman
    (1985) advises against its use for any other purpose, including the reverse
    conversion from c or rho to M_z. Instead Bachman provides separate
    correlation equations for this". Those separate equations are not in hand
    -- Bachman, *JASA* 78, 616-621 (1985) is not in the corpus -- so this
    function inverts the forward fit, which is a different estimator: a fit of
    rho on phi minimises residuals in rho, so reading it backwards compresses
    the spread in phi and biases a phi estimated from a noisy density toward
    the middle of the range. Treat what comes back as an index, not as a
    measurement; if Bachman (1985) is ever ingested, this is the first thing to
    replace.
    """
    if environment not in GRAIN_SIZE_ENVIRONMENTS:
        raise ConfigurationError(
            f"grain_size_from_density: unknown environment {environment!r}.",
            remediation=f"Use one of {tuple(GRAIN_SIZE_ENVIRONMENTS)}.",
        )
    fit = GRAIN_SIZE_ENVIRONMENTS[environment]
    rho_w = _HB_REF_RHOW if water_density is None else float(water_density)
    lab = float(density) / rho_w * _HB_T_REF_RHOW
    lo, hi = fit['evaluated_range']
    coefficients = fit['density']
    if len(coefficients) == 2:                      # the abyssal fits are lines
        c0, c1 = coefficients
        phi = (lab - c0) / c1
    else:
        c0, c1, c2 = coefficients
        # (T)'s vertex is at -c1/(2 c2) = 10.94 ϕ, past the evaluated range, so
        # the relation is strictly decreasing there and the smaller root is the
        # one inside; the larger is always beyond 10.9 ϕ. Below the vertex's
        # value there is no root at all.
        discriminant = c1 * c1 - 4.0 * c2 * (c0 - lab)
        # No real root at all below the vertex's value: the density is finer
        # than the relation can represent, which is the *common* case over
        # deep ocean and must be announced rather than absorbed by a clamp.
        phi = (None if discriminant <= 0.0
               else (-c1 - np.sqrt(discriminant)) / (2.0 * c2))
    if phi is None or not lo <= phi <= hi:
        held = hi if phi is None else float(min(max(phi, lo), hi))
        span = sorted(np.polyval(coefficients[::-1], np.array([lo, hi]))
                      / _HB_T_REF_RHOW * rho_w)
        elsewhere = [name for name, other in GRAIN_SIZE_ENVIRONMENTS.items()
                     if name != environment
                     and min(_density_span(other, rho_w))
                     <= float(density) <= max(_density_span(other, rho_w))]
        warnings.warn(
            f"grain_size_from_density: ρ={float(density):g} g/cm³ is outside "
            f"the {span[0]:.3f} to {span[1]:.3f} g/cm³ the "
            f"{environment!r} relation represents, so the grain size is its "
            f"ϕ={held:g} end rather than one that reproduces ρ"
            + (f"; {_names_and(elsewhere)} covers this density"
               if elsewhere else "") + ".",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
        return held
    return float(phi)


def _density_span(fit: Dict, water_density: float):
    """The in-situ densities one environment's fit represents over its range."""
    return tuple(np.polyval(fit['density'][::-1],
                            np.array(fit['evaluated_range']))
                 / _HB_T_REF_RHOW * water_density)


def _hamilton_geoacoustics(phi, water_sound_speed, water_density,
                           environment=DEFAULT_GRAIN_SIZE_ENVIRONMENT):
    """``(cp, density, attenuation)`` from one environment's H&B regressions.

    The fits are evaluated at ϕ held to that environment's own range, so none
    is run past the interval its source declares; ``_hamilton_kp`` holds itself
    to its own range the same way, and is the same regression for all three
    environments. Both ratios are formed against the laboratory reference the
    (T) coefficients imply and then re-scaled to the in-situ water, which is
    what makes a laboratory number usable at depth.
    """
    fit = GRAIN_SIZE_ENVIRONMENTS[environment]
    lo, hi = fit['evaluated_range']
    m = min(max(float(phi), lo), hi)
    velocity_ratio = np.polyval(fit['velocity'][::-1], m) / _HB_T_REF_CW
    density_ratio = np.polyval(fit['density'][::-1], m) / _HB_T_REF_RHOW
    cp = float(velocity_ratio) * water_sound_speed
    # α(dB/λ) = k_p · f(kHz) · λ = k_p · c / 1000: frequency drops out.
    attenuation = _hamilton_kp(phi) * cp / 1000.0
    return cp, float(density_ratio) * water_density, attenuation


# APL-UW TR 9407 (1994), §IV.A.4 "Model Input Parameters Using Grain Size"
# (valid −1 ≤ Mz ≤ 9): density ratio ρ₂/ρ₁ and sound-speed ratio c₂/c₁ as
# piecewise polynomials in Mz, plus the attenuation α₂/f (dB m⁻¹ kHz⁻¹). These
# are the formulas the AT ``'G'`` bottom uses internally
# (Bellhop/ReadEnvironmentBell.f90:488 `CASE ( 'G' )`).
# Verified against that Fortran: the velocity-ratio, density-ratio and
# alpha2_f branches reproduce ``ReadEnvironmentBell.f90:497-520`` to 0.0e+00
# over Mz in [-1, 9] — which is *why* the printed coefficients are carried here
# rather than derived from the (T) polynomials they equal: AT implements the
# printed digits, uacpy's apl-uw exists to reproduce AT, and rescaling
# 1952.5/1528.0 in place of the printed 1.2778 would put uacpy 1.5e-05 away
# from the Fortran. The relationship is pinned by a test instead, so a
# correction to either set cannot leave the other stale, and the dB/wavelength attenuation below is AT's 'L' loss
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


def _apl_uw_geoacoustics(phi, water_sound_speed, water_density):
    """``(cp, density, attenuation)`` from the APL-UW TR 9407 relations."""
    cp = _apl_velocity_ratio(phi) * water_sound_speed
    # α(dB/λ) = (α₂/f)[dB/m/kHz]·c[m/s]/1000 — frequency cancels (α ∝ f).
    attenuation = _hamilton_kp(phi) * cp / 1000.0
    return cp, _apl_density_ratio(phi) * water_density, attenuation


_GEOACOUSTIC_MODELS = {'hamilton': _hamilton_geoacoustics,
                       'apl-uw': _apl_uw_geoacoustics}


def _names_and(names) -> str:
    """``'a'`` / ``'a and b'`` / ``'a, b and c'``."""
    return (names[0] if len(names) == 1
            else ' and '.join((', '.join(names[:-1]), names[-1])))


def _endpoint_quantities(model: str, environment: str, phi: float):
    """``[(names, (lo, hi)), ...]`` for the returned quantities whose own
    source does not cover ``phi``, grouped by the source they share.

    Empty when every quantity is interpolated from data that reaches ``phi``,
    which is the only case in which nothing has been substituted.
    """
    groups: Dict[tuple, list] = {}
    for name, span in _source_ranges(model, environment).items():
        if not span[0] <= phi <= span[1]:
            groups.setdefault(span, []).append(name)
    return [(names, span) for span, names in groups.items()]


def grain_size_to_geoacoustics(
    grain_size_phi: float, *, model: str = DEFAULT_GRAIN_SIZE_MODEL,
    environment: str = DEFAULT_GRAIN_SIZE_ENVIRONMENT,
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
        ``'hamilton'`` (default) — Hamilton & Bachman (1982) / Hamilton (1972).
        ``'apl-uw'`` — APL-UW TR 9407 (1994). What the choice does, and where
        it does nothing, is below.
    environment : str, optional
        Which of Hamilton & Bachman's three fits ``'hamilton'`` evaluates:
        ``'continental-terrace'`` (default, shelf and slope, 1 to 9 ϕ),
        ``'abyssal-hill'`` or ``'abyssal-plain'`` (both 7 to 10 ϕ, and
        straight lines — they describe no sand, so a coarser ϕ is held at 7
        and reported). :data:`GRAIN_SIZE_ENVIRONMENTS` carries each one's
        coefficients, published σ and range. **Hamilton states no rule for
        choosing**: the environments are named for where the samples came
        from, so the caller has to know their site. Passing a non-default
        ``environment`` with ``model='apl-uw'`` raises — TR 9407 publishes one
        set of relations, not one per environment.
    water_sound_speed, water_density : float, optional
        In-situ seawater sound speed (m/s) and density (g/cm³) the ratios are
        scaled by. ``None`` (default) uses the reference the chosen ``model``
        was tabulated against — Hamilton's 1510 m/s / 1.030 g/cm³, or APL-UW's
        1500 m/s / 1.0 g/cm³, which reproduces the Acoustics-Toolbox ``'G'``
        bottom exactly.

    **Each returned quantity is reported against its own source.** A
    ``UserWarning`` names the quantities whose source does not hold data at
    ``grain_size_phi``, together with the ϕ whose value they carry instead —
    because a held endpoint is indistinguishable from an interpolated value
    once returned, and because the answer differs *between quantities of one
    call*. Under ``'hamilton'`` at ϕ = -0.5 the sound speed and density are the
    (T) regressions evaluated there, while the attenuation is ``k_p`` held at
    its 0 ϕ start; ``'apl-uw'`` is one equation set over ``[-1, 9]``, so its
    three quantities always answer together.
    :data:`GRAIN_SIZE_SOURCE_RANGES` publishes the intervals this is read from.

    ``grain_size_phi`` outside the model's ϕ range (the union of those source
    ranges, :data:`GRAIN_SIZE_MODEL_RANGES`) is additionally clamped to it, and
    where the clamp moves the answer the warning also reports what the
    unclamped fit would have given — ``'apl-uw'`` only, whose polynomials keep
    extrapolating (ϕ = 9.5 differs by 1.8 m/s, ϕ = -1.5 by 47 m/s). For
    ``'hamilton'`` the clamp moves nothing, each of its sources already holding
    ϕ at its own edge, so there the report is the per-quantity one alone. ±inf is
    reported like any other ϕ past the end; a NaN ϕ propagates to NaN outputs
    and is not reported, there being no substituted value.

    **What ``model`` actually selects.** For attenuation at every ϕ, and for
    density and sound speed below 1 ϕ, it selects nothing: the two are one
    relation — the attenuation is literally one function both models call, and
    the speeds agree to 0.03 m/s, which is the rounding of TR 9407's printed
    coefficients. Above 1 ϕ it
    selects between **a measured bulk property and an effective scattering
    parameter**. TR 9407 says so itself, twice: its ratios "are surficial
    values derived from model fitting and tend to be smaller than the bulk
    values reported in the literature" (p. IV-5), and "the measured density
    ratios from these sites were **discarded** because of the large scatter and
    vertical variations in porosity ... Rather, the acoustic models were fit to
    data in order to determine the relationship between grain size and
    density" (p. IV-9). So at 4 ϕ its ρ = 1.224 is not a claim about silty
    sand's density — Hamilton measures 1.783 there on 340 laboratory samples —
    it is the density that made APL-UW's reflection-loss and backscatter model
    match. That is also why the attenuations agree: attenuation was taken from
    Hamilton, only the ratios were refitted.

    Which gives the rule: ``'hamilton'`` when the geoacoustics are the
    product — a propagation model's seabed — and ``'apl-uw'`` when driving the
    APL-UW scattering model those parameters were fitted for. Past -1 ϕ both stop, and the honest
    answer for gravel is a sediment class from :mod:`uacpy.core.materials`.

    Three places in uacpy answer an out-of-range grain size differently, and
    the difference is in what each one is *given* rather than in what each one
    holds to be right.

    - **here**, the ϕ is all there is. Clamping and reporting it is the only
      answer available, because nothing at this layer knows where the value
      came from or what else could stand in for it.
    - :func:`uacpy.data.sediment_db.fetch_bottom_local` is given a DECK41
      **lithology word**. ``'gravel'`` has no measurement behind it, so
      nothing is lost by answering it with the ``'gravel'`` material preset,
      whose geoacoustics are sourced — which is what it does, the same way
      ``'rock'`` takes ``'limestone'``.
    - :func:`uacpy.data.mars.fetch_bottom_mars` is given a **sample**: a
      measured grain size, measured gravel/sand/mud percentages, or a Folk
      class. Substituting a preset would throw that measurement away, and on
      the percentage route ϕ is continuous, so any rule for when to substitute
      would be a threshold drawn across a continuum. It converts the ϕ it was
      given and warns, naming the sample and the route that produced it.
    """
    check_grain_size_selection(model, environment,
                               caller='grain_size_to_geoacoustics')
    ref_cw, ref_rhow = _MODEL_WATER_REFERENCE[model]
    if water_sound_speed is None:
        water_sound_speed = ref_cw
    if water_density is None:
        water_density = ref_rhow
    spans = _source_ranges(model, environment).values()
    lo = min(span[0] for span in spans)
    hi = max(span[1] for span in spans)
    evaluate = _GEOACOUSTIC_MODELS[model]
    if model == 'hamilton':
        evaluate = partial(evaluate, environment=environment)
    phi = float(np.clip(grain_size_phi, lo, hi))
    cp, density, attenuation = evaluate(phi, water_sound_speed, water_density)
    # Report per quantity, against the source each one comes from. Comparing
    # the clamped fit with the fit at the raw phi cannot do it: on 'hamilton'
    # both hold their own limits flat, so the two are equal by construction at
    # every phi and the comparison is blind to the substitution it exists to
    # catch. "Did this quantity's own source cover this phi?" is answerable at
    # every phi, and it is the question whose answer differs between quantities
    # -- at phi = 0.5 the attenuation is interpolated while the velocity and
    # density are an end row. NaN is excluded because it propagates to NaN
    # outputs, so there is nothing substituted to report; +-inf is included,
    # being the largest substitution there is.
    if not np.isnan(grain_size_phi):
        notes = [
            f"{_names_and(names)} "
            f"{'come' if len(names) > 1 else 'comes'} from data covering "
            f"{src_lo:g} to {src_hi:g} ϕ, so "
            f"{'they are' if len(names) > 1 else 'it is'} its "
            f"ϕ={min(max(grain_size_phi, src_lo), src_hi):g} "
            f"{'values' if len(names) > 1 else 'value'}"
            for names, (src_lo, src_hi) in _endpoint_quantities(
                model, environment, grain_size_phi)
        ]
        if notes and phi != grain_size_phi:
            raw = evaluate(float(grain_size_phi), water_sound_speed,
                           water_density)
            if raw != (cp, density, attenuation):
                notes.append(
                    f"ϕ was clamped to [{lo:g}, {hi:g}], and "
                    f"the unclamped fit there gives sound_speed={raw[0]:.4f} "
                    f"m/s, density={raw[1]:.4f} g/cm³, "
                    f"attenuation={raw[2]:.4f} dB/λ")
        if notes:
            warnings.warn(
                f"grain_size_to_geoacoustics: at ϕ={grain_size_phi:g}, "
                + "; ".join(notes) + ".",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
    return {'sound_speed': cp, 'density': density, 'attenuation': attenuation}
