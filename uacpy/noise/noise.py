"""
Ambient-noise model — Tollefsen / Pecknold packaging.

`compute_windnoise` and the :class:`WenzNoise` class follow Tollefsen &
Pecknold "A simple yet practical ambient noise model"
(DRDC-RDDC-2022-D051, May 2022): wind / shipping / rain / thermal /
turbulence components summed in dB re 1 µPa²/Hz.

References (as listed in the DRDC report)
-----------------------------------------
Tollefsen, C. D. S. & Pecknold, S. (2022). *A simple yet practical
   ambient noise model.* DRDC-RDDC-2022-D051, DRDC-Atlantic.
Wenz, G. M. (1962). Acoustic ambient noise in the ocean: Spectra
   and sources. (cited via the DRDC report.)
Mellen, R. H. (1952). The thermal-noise limit. (cited via the report.)
Piggott, C. L. (1964). Ambient sea noise at low frequencies in shallow
   water. (cited via the report.)
Merklinger, H. M. (1979). Formulae for estimation of undersea noise
   spectra. (cited via the report.)
Torres, C. & Costa, C. (2019). Underwater ambient noise — an
   estimation. (cited via the report.)
Nichols, S. M. & Bradley, D. L. (2016). Global examination of the
   wind-dependence of low-frequency ambient noise. (cited via the
   report.)
"""
import warnings
from collections import namedtuple

import numpy as np

from uacpy.core.exceptions import ConfigurationError


NoiseComponents = namedtuple(
    "NoiseComponents", "total wind shipping rain thermal turbulence")


# IEC 61260-1 base-10 decidecade ("1/3-octave") bands run 10**(±1/20) about
# their centre — 0.1 decade wide, so consecutive centres sit 0.1 decade apart
# (:func:`uacpy.acoustic_signal.bands.decidecade_bands`). It is the narrowest
# analysis band uacpy defines, and so the yardstick for whether a frequency
# vector is a set of band centres or a plotting grid.
_DECIDECADE_DECADES = 0.1

# The canonical vector this threshold has to accept — ``decidecade_bands``'
# own centres, the call docs/guide/noise.md section 6 teaches — lands ON it,
# and ``log10`` of ``1000 * 10**(n/10)`` reproduces the 0.1 step only to a few
# ulp: over 100 Hz - 1 kHz the tightest gap computes as
# 0.09999999999999964, short by 3.6e-16 (relative 3.6e-15). A bare ``<`` then
# warns about the very usage the warning recommends. Relative to the spacing
# being tested, per the package's epsilon rule, not an absolute floor: the
# comparison lives in decades, so its tolerance has to scale with the band
# width. 1e-9 clears that float shortfall by six orders while staying far
# below any real grid difference — one extra point across the same decade
# (12 against 11) is 9% finer, seven orders above this tolerance.
_BAND_SPACING_RTOL = 1e-9


def _tightest_spacing_in_decades(f):
    """Smallest gap (decades) between consecutive positive frequencies.

    ``np.nan`` when fewer than two positive frequencies are present. Symmetric
    in the sign of the step, so a descending vector is measured the same way.
    Non-positive entries are dropped rather than treated as a gap of their own:
    that only widens the apparent spacing, so the test it feeds never fires on
    their account.
    """
    pos = f[f > 0]
    if pos.size < 2:
        return np.nan
    return float(np.min(np.abs(np.diff(np.log10(pos)))))


# ─────────────────────────────────────────────────────────────────────────────
# Wind noise (free function, used inside WenzNoise too)
# ─────────────────────────────────────────────────────────────────────────────


def compute_windnoise(frequencies, u, water_depth='deep', band_integrate=False):
    """
    Wind-driven ambient noise level (dB re 1 µPa²/Hz), with the
    Piggott (1964) shallow-water adjustment.

    Parameters
    ----------
    frequencies : ndarray or float
        Frequencies in Hz (1-Hz band assumed for a scalar).
    u : float
        Wind speed in **knots at the 10 m reference height** — the
        variable DRDC-RDDC-2022-D051 §2.3 fits (eq. 8 is stated for
        ``u`` in knots; the section defines it at 10 m). Must be
        non-negative; ``u == 0`` silences the wind component and the
        returned spectral level is ``-inf`` dB at every frequency (the
        surface-noise source has no power). **This deliberately differs
        from the DRDC report**, whose ``u = 0`` case returns 0 dB — a
        level that carries 1 µPa²/Hz of power and so adds a spurious
        floor to an incoherent dB sum. Expect a 0 dB / ``-inf`` dB
        mismatch at ``u = 0`` when diffing against the report; every
        ``u > 0`` level is unchanged.
    water_depth : {'deep', 'shallow'}
        Coefficient family. Default 'deep'. Matched exactly and
        case-sensitively; anything else raises rather than falling back on
        'deep', which would hand back a spectrum 3.0 dB from the one asked for
        (50 Hz, 'shallow') with nothing to say so.
    band_integrate : bool
        If True, return the band-integrated SPL (dB re 1 µPa²) where each
        band's bandwidth is set by the midpoints between consecutive
        input frequencies in ascending order — ``frequencies`` need not be
        pre-sorted, and each level comes back at its frequency's own position
        in the input. Default False — return the spectral level
        (dB re 1 µPa²/Hz). Use the band form to pair wind noise with a
        band-integrated source level; :class:`WenzNoise` is spectral-only.

        Because the widths come from the vector's own spacing,
        ``frequencies`` has to be a set of analysis-band **centres** for the
        result to be band levels. A vector finer than decidecade spacing is a
        plotting grid, and the levels then move with its density — 18.0 dB
        between 20 and 1200 points over one decade — so that case raises a
        :class:`UserWarning`.

    Returns
    -------
    NL : ndarray
        Wind noise spectral level (dB re 1 µPa²/Hz), or band-integrated
        SPL (dB re 1 µPa²) if ``band_integrate=True``. Always shaped
        ``(len(f),)`` (1-D, even for scalar ``f``).

    Notes
    -----
    Translated from the IDL implementation by Dan Hutt, rewritten by Vic
    Young, and packaged in Tollefsen & Pecknold (2018).
    """
    # Normalise ``frequencies`` up-front so scalars (the docstring promises
    # they work) don't crash at ``.size`` / ``.flatten()`` below.
    f = np.atleast_1d(np.asarray(frequencies, dtype=float)).flatten()

    u = float(u)
    # Written as the negation of the admissible condition so NaN is refused
    # too: ``nan < 0`` is False, and a NaN wind would then fall through to the
    # ``u == 0`` branch below and come back as the -inf switched-off sentinel,
    # i.e. a plausible finite spectrum tens of dB low.
    if not (u >= 0):
        raise ConfigurationError(
            f"compute_windnoise: wind speed u must be non-negative (knots) and "
            f"finite, got {u}"
        )

    # The deep/shallow offset c0 of DRDC §2.3 eq. 10, resolved before the
    # ``u == 0`` short circuit so an unusable ``water_depth`` is refused on
    # every path rather than only the ones that reach the fit.
    #
    # ``'deep'`` is the right value for an *omitted* argument — the signature
    # says so — which is exactly what makes it wrong for an unrecognised one:
    # defaulting a typo or a case variant ('SHALLOW') to it returns the deep
    # curve, 3.0 dB below the shallow one the caller asked for at 50 Hz. So
    # the match is exact and anything else raises here. ``WenzNoise.__init__``
    # raises on this same parameter with this same value set, which leaves the
    # direct ``compute_windnoise`` / ``WIND_MODELS['merklinger']`` calls — both
    # public exports — as the paths this guard covers.
    if water_depth == 'deep':
        cst = 42
    elif water_depth == 'shallow':
        cst = 45
    else:
        raise ConfigurationError(
            f"compute_windnoise: water_depth must be 'deep' or 'shallow', got "
            f"{water_depth!r} (the match is exact and case-sensitive). The two "
            f"select different Piggott (1964) offsets — c0 = 42 vs 45 dB, a "
            f"3.0 dB difference at 50 Hz."
        )

    # u == 0 silences the surface-noise source: return -inf dB at every
    # frequency so an incoherent dB sum with the other Wenz components
    # behaves correctly (10**(-inf/10) == 0, no contribution).
    if u == 0:
        NL = np.full_like(f, -np.inf)
    else:
        n_freq = f.size
        if band_integrate:
            if n_freq < 2:
                raise ConfigurationError(
                    "compute_windnoise(band_integrate=True) needs at least two "
                    "frequencies to define band edges; got one. Use "
                    "band_integrate=False for a scalar spectral level."
                )
            # The band form exists to pair wind noise with a band-integrated
            # source level, so ``frequencies`` has to be a set of analysis-band
            # CENTRES. Below decidecade spacing it is a plotting grid instead,
            # and each returned level is a PSD sample scaled by whatever sliver
            # of bandwidth the grid happens to carry: over one decade,
            # 100 Hz - 1 kHz at u = 10 kn, the loudest band reads 80.60 dB at
            # 20 points and 62.60 dB at 1200, an 18.0 dB swing on the same
            # physics. (The band SUM is grid-invariant — 90.47 dB either way —
            # it is the per-band level that moves.)
            tightest = _tightest_spacing_in_decades(f)
            floor = _DECIDECADE_DECADES * (1.0 - _BAND_SPACING_RTOL)
            if np.isfinite(tightest) and tightest < floor:
                # A repeated frequency gives a zero-width band, so the
                # points-per-decidecade count it would otherwise report is
                # unbounded; name the duplication instead.
                density = (
                    f"about {_DECIDECADE_DECADES / tightest:.0f} of these "
                    f"frequencies fall inside one decidecade"
                    if tightest > 0 else
                    "frequencies are repeated, so those bands have zero width"
                )
                # Reachable only from a user's own call: the one in-package
                # caller, ``_wind_merklinger``, passes three positional
                # arguments, so ``band_integrate`` is always False there and
                # this branch is dead on that path.
                warnings.warn(
                    f"compute_windnoise(band_integrate=True): consecutive "
                    f"frequencies come as close as {tightest:.4g} decades, "
                    f"finer than the {_DECIDECADE_DECADES:g}-decade decidecade "
                    f"band (uacpy.acoustic_signal.bands.decidecade_bands) — "
                    f"{density}. Each band's width is taken from this "
                    f"vector's own spacing, so at this density the returned "
                    f"levels are spectral levels scaled by an arbitrary "
                    f"bandwidth and move with the grid (measured 18.0 dB "
                    f"between 20 and 1200 points over one decade). Pass "
                    f"analysis-band centres for band levels, or "
                    f"band_integrate=False for the spectral level.",
                    UserWarning, stacklevel=2,
                )
            # Band edges at the midpoints between consecutive frequencies in
            # ascending order; the two outer bands span only the half-spacing
            # to their single neighbour (df[0]=(f[1]-f[0])/2,
            # df[-1]=(f[-1]-f[-2])/2). A leading 0 / symmetric extrapolation
            # would over-weight the end bands. The widths are computed on the
            # sorted grid and scattered back through the argsort, so each
            # frequency keeps its own bandwidth at the caller's ordering — a
            # descending vector returns the ascending band levels reversed,
            # where edges taken from the raw vector gave negative widths and an
            # all-NaN spectrum.
            order = np.argsort(f)
            f_sorted = f[order]
            mids = (f_sorted[1:] + f_sorted[:-1]) / 2
            edges = np.concatenate(([f_sorted[0]], mids, [f_sorted[-1]]))
            df = np.empty_like(f)
            df[order] = edges[1:] - edges[:-1]
        else:
            df = np.ones_like(f)

        # Symbols follow DRDC-RDDC-2022-D051 §2.3: f0w/L0w the peak frequency
        # and peak level of the wind curve (eqs. 8, 9), s1w/s2w its rising and
        # falling slopes in dB per **octave** (eqs. 14, 15 — the 1/log10(2)
        # factors below turn those into a coefficient on log10(f)), a the
        # exponent melding the two branches (eq. 16), cst the report's
        # deep/shallow offset c0 (eq. 10). The empirical fit is valid only up
        # to f_wind; above it the report continues as a power law of exponent
        # ``slope`` — s2w restated on linear power (m0, eq. 17), so
        # 10*log10(f**slope) falls s2w dB per doubling of f.
        f_wind = 2000
        s1w = 1.5
        s2w = -5.0
        a = -25
        slope = s2w * (0.1 / np.log10(2))
        NL = np.zeros_like(f)

        i_wind = f <= f_wind
        f0w = 770 - 100 * np.log10(u)
        L0w = cst + 20 * np.log10(u) - 17 * np.log10(f0w / 770)
        f_below_cutoff = f[i_wind]
        L1w = L0w + (s1w / np.log10(2)) * np.log10(f_below_cutoff / f0w)
        L2w = L0w + (s2w / np.log10(2)) * np.log10(f_below_cutoff / f0w)
        # The exponent is 1/a. DRDC's typeset eq. (13)/(20) prints -1/a, but
        # that is a sign typo: with -1/a the branch above f0w *rises* at
        # +9.1 dB/octave, contradicting the report's own eq. (12)/(15), which
        # define that branch by s2 = -5 dB/octave. With 1/a it falls at
        # -4.5 dB/octave, converging on s2. The Annex A.2 Matlab also has 1/a.
        #
        # With a < 0 the melding is a smooth minimum of the two branches: they
        # cross at f0w, so the rising L1w governs below it and the falling L2w
        # above, and the melding rounds off that corner (a few dB below the
        # crossing, which is why the melded curve peaks a little under f0w).
        # It raises ``1 + (L1w/L2w)**(-a)`` to a fractional power, which is real
        # only while that base is positive. Below about 0.01 kn the two
        # asymptotes fall through zero and the base goes negative — the same
        # physical statement as ``u == 0``: the surface source is silent, so
        # those bins contribute no power.
        # An empty `i_wind` carries the empty arrays straight through — the
        # sub-cutoff branch is only ever read at the same mask that selected
        # its frequencies. The above-cutoff branch below anchors on the level
        # at f_wind itself, not on anything computed here.
        with np.errstate(invalid='ignore', divide='ignore'):
            blend = 1 + (L1w / L2w) ** (-a)
            Lw = np.where(blend > 0, L1w * np.abs(blend) ** (1 / a), -np.inf)
        NL[i_wind] = 10 ** (Lw / 10) * df[i_wind]

        # Meld with a sensible line at freqs greater than 2000 Hz, per DRDC
        # eq. (18)-(19): K = Lw,2000 - m0*(10 log10 2000), Lw = K + m0*10 log10 f.
        # The anchor is the level at the f_wind cutoff *itself*, as those
        # equations specify — not whatever in-grid sample happens to be the last
        # one below it, which is the shortcut the Annex A.2 listing takes and
        # which makes every level above 2 kHz depend on the caller's frequency
        # spacing (measured: 17.6 dB
        # between a grid whose last sub-cutoff point is 1999 Hz and one where
        # it is 100 Hz). Same reasoning as the rain roll-off below.
        if np.any(~i_wind):
            L1c = L0w + (s1w / np.log10(2)) * np.log10(f_wind / f0w)
            L2c = L0w + (s2w / np.log10(2)) * np.log10(f_wind / f0w)
            with np.errstate(invalid='ignore', divide='ignore'):
                blend_c = 1 + (L1c / L2c) ** (-a)
                Lc = (L1c * abs(blend_c) ** (1 / a)
                      if blend_c > 0 else -np.inf)
            prop_const = 10 ** (Lc / 10) / f_wind ** slope
            NL[~i_wind] = prop_const * f[~i_wind] ** slope * df[~i_wind]

        with np.errstate(divide='ignore'):
            NL = 10 * np.log10(NL)

    return NL


# ─────────────────────────────────────────────────────────────────────────────
# Wenz composite — class API
# ─────────────────────────────────────────────────────────────────────────────


# Traffic-density coefficient c2 of the Wenz shipping fit. ``'no'`` is not a
# density in the fit — it is here only so ``WenzNoise`` accepts it as a
# shipping_level; the submodels short-circuit to -inf before reading its value.
_SHIPPING_C2 = {'low': 1, 'medium': 4, 'high': 7, 'no': 4}
_RAIN_INDEX = {'no': 0, 'light': 1, 'moderate': 2, 'heavy': 3, 'veryheavy': 4}
_RAIN_R0 = [0, 51.0769, 61.5358, 65.1107, 74.3464]
_RAIN_R1 = [0, 1.4687,  1.0147,  0.8226,  1.0131]
_RAIN_R2 = [0, -0.5232, -0.4255, -0.3825, -0.4258]
_RAIN_R3 = [0, 0.0335,  0.0277,  0.0251,  0.0277]


# ── Component submodels (registry of swappable formulas) ────────────────────
# Built-ins are the Canadian/DRDC composite (Tollefsen & Pecknold 2018;
# DRDC-RDDC-2022-D051 "WenzCurves"). Each submodel takes the full parameter
# bundle and ignores what it does not need via **_. Returns dB re 1 µPa²/Hz —
# the real (possibly sub-0-dB) spectral level at every band; -inf is reserved
# for a source that is switched *off* (shipping/rain 'no', wind speed 0), which
# the incoherent logaddexp sum then drops.

def _as_frequency_array(frequencies, caller: str) -> np.ndarray:
    """Coerce a submodel's ``frequencies`` argument to a 1-D float ndarray.

    ``WenzNoise``'s docstring advertises every registry entry as
    ``model(frequencies, *, wind_speed_kn, water_depth, shipping_level,
    rain_rate, **_)`` and the five registries are public, so a caller may hold
    one and call it directly with a list or a scalar. Doing that reached
    ``list / 1000.0`` as a ``TypeError`` in one submodel and a 0-d array as
    ``'float' object is not subscriptable`` in another, and returned a bare
    scalar from most and a ``(1,)`` array from one. Every submodel converts
    here first, so all seven answer the same shapes.
    """
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    if f.ndim != 1:
        raise ConfigurationError(
            f"{caller}: frequencies must be a scalar or a 1-D array of Hz; "
            f"got shape {f.shape}.")
    return f


def _thermal_mellen(frequencies, **_):
    """Thermal (Mellen 1952; DRDC §2.5 eq. 22): ``-75 + 20·log10(f_Hz)``."""
    f = _as_frequency_array(frequencies, "_thermal_mellen")
    return -75.0 + 20.0 * np.log10(f)


def _wind_merklinger(frequencies, *, wind_speed_kn, water_depth, **_):
    """Wind (Merklinger 1979 + Piggott 1964 shallow correction); DRDC §2.3."""
    f = _as_frequency_array(frequencies, "_wind_merklinger")
    return compute_windnoise(f, wind_speed_kn, water_depth)


def _wind_coates(frequencies, *, wind_speed_kn, **_):
    """Wind (Coates 1989 / Stojanović 2007, the standard UW-comms form):
    ``50 + 7.5·√w + 20·log10(f) − 40·log10(f + 0.4)`` with ``f`` in kHz and
    ``w`` the wind speed in m/s (converted here from the knots input).

    ``wind_speed_kn == 0`` silences the source (-inf dB) exactly as
    :func:`compute_windnoise` does: the formula has no wind term that vanishes
    with ``w``, so it would otherwise return ~44 dB re 1 µPa²/Hz at 1 kHz in a
    flat calm and the incoherent sum would inherit it."""
    fk = _as_frequency_array(frequencies, "_wind_coates") / 1000.0
    if float(wind_speed_kn) == 0:
        return np.full(fk.shape, -np.inf, dtype=float)
    w_ms = float(wind_speed_kn) / 1.9438445       # knots → m/s
    return (50.0 + 7.5 * np.sqrt(w_ms) + 20.0 * np.log10(fk)
            - 40.0 * np.log10(fk + 0.4))


def _shipping_wenz(frequencies, *, shipping_level, water_depth, **_):
    """Shipping (Wenz 1962; DRDC §2.2 eq. 5-7)."""
    f = _as_frequency_array(frequencies, "_shipping_wenz")
    # Explicitly two-way, not ``30 if deep else 65``: with a bare ``else`` an
    # unrecognised string took the *shallow* branch here while the same string
    # took the *deep* branch in ``compute_windnoise`` — opposite silent
    # defaults for one typo in one module. Checked ahead of the 'no' short
    # circuit so the parameter is refused on every path.
    if water_depth == 'deep':
        c1 = 30
    elif water_depth == 'shallow':
        c1 = 65
    else:
        raise ConfigurationError(
            f"_shipping_wenz: water_depth must be 'deep' or 'shallow', got "
            f"{water_depth!r} (the match is exact and case-sensitive). The "
            f"two select different Wenz spectrum peaks — c1 = 30 vs 65 Hz."
        )
    if shipping_level == 'no':
        # dtype=float, not full_like: an integer frequency vector casts -inf
        # to INT64_MIN, and these registries are public entry points.
        return np.full(np.shape(f), -np.inf, dtype=float)
    c2 = _SHIPPING_C2[shipping_level]
    return 76 - 20 * (np.log10(f) - np.log10(c1)) ** 2 + 5 * (c2 - 4)


_COATES_SHIP_ACTIVITY = {'no': None, 'low': 0.0, 'medium': 0.5, 'high': 1.0}


def _shipping_coates(frequencies, *, shipping_level, **_):
    """Shipping (Coates 1989 / Stojanović 2007 turbulent-shipping form):
    ``40 + 20(s − 0.5) + 26·log10(f) − 60·log10(f + 0.03)`` with ``f`` in kHz
    and ``s`` the shipping-activity factor in [0, 1] (low/medium/high →
    0/0.5/1; ``'no'`` is silent)."""
    s = _COATES_SHIP_ACTIVITY[shipping_level]
    fk = _as_frequency_array(frequencies, "_shipping_coates") / 1000.0
    if s is None:
        return np.full(fk.shape, -np.inf, dtype=float)
    return (40.0 + 20.0 * (s - 0.5) + 26.0 * np.log10(fk)
            - 60.0 * np.log10(fk + 0.03))


def _turbulence_wenz(frequencies, **_):
    """Turbulence (DRDC §2.1): ``NL_turb = NL_t + m_t·log10(f_Hz)`` with
    ``NL_t = 107 dB re µPa`` and ``m_t = −10 dB/octave``, i.e.
    ``−10/log10(2) = −33.2 dB/decade``. The slope is the primitive quantity —
    Wenz, Urick and Nichols & Bradley all cite −8 to −10 dB/octave — so it is
    written per octave and converted here.

    DRDC's Annex A listing does not implement these values — it has
    ``108.5 - 32.5*log10(f)`` (= -9.78 dB/octave), 1.5-2.9 dB higher over
    1-100 Hz. Both sit inside the -8 to -10 dB/octave range the report cites, but
    §2.1 is the specification and 107 dB is traceable to Nichols & Bradley Fig. 5
    in its own Table 1, so uacpy implements the specification. That is the same
    rule applied everywhere in this module: the numbered equations are
    normative, and the annex is an implementation that takes shortcuts."""
    f = _as_frequency_array(frequencies, "_turbulence_wenz")
    m_t = -10.0 / np.log10(2.0)
    return 107.0 + m_t * np.log10(f)


def _rain_torres_costa(frequencies, *, rain_rate, **_):
    """Rain (Torres & Costa 2019; DRDC §2.4 eq. 21), valid to ~7 kHz; melded above.

    The report's eq. (21) writes the cubic in a bare ``f``; its Table 2
    coefficients and the Annex A.1 listing (``fk = f/1000``) both take that
    argument in **kHz**, so ``cubic`` below is defined on kHz: the Hz grid is
    divided by 1000 and the 7 kHz anchor is passed as ``7.0``.
    """
    f = _as_frequency_array(frequencies, "_rain_torres_costa")
    if rain_rate == 'no':
        return np.full(np.shape(f), -np.inf, dtype=float)
    ir = _RAIN_INDEX[rain_rate]

    def cubic(f_khz):
        return (_RAIN_R0[ir] + _RAIN_R1[ir] * f_khz
                + _RAIN_R2[ir] * f_khz ** 2 + _RAIN_R3[ir] * f_khz ** 3)

    out = cubic(f / 1000.0)
    slope = -5.0 * (0.1 / np.log10(2))
    above = f > 7000.0
    if np.any(above):
        # §2.4 extends the fit "with constant slope ... in a similar manner to
        # that described for the wind noise (Section 2.3)", i.e. by eq. (18)-(19),
        # so the anchor is the cubic's value at the 7 kHz validity limit itself
        # — independent of whether the frequency grid contains a sub-7 kHz
        # sample (without an in-grid anchor the raw cubic would be extrapolated
        # to physically impossible levels).
        prop_const = 10 ** (cubic(7.0) / 10) / 7000.0 ** slope
        out[above] = 10 * np.log10(prop_const * f[above] ** slope)
    return out


WIND_MODELS = {'merklinger': _wind_merklinger, 'coates': _wind_coates}
SHIPPING_MODELS = {'wenz': _shipping_wenz, 'coates': _shipping_coates}
RAIN_MODELS = {'torres_costa': _rain_torres_costa}
THERMAL_MODELS = {'mellen': _thermal_mellen}
TURBULENCE_MODELS = {'wenz': _turbulence_wenz}


def _resolve_submodel(value, registry, default, label):
    """Resolve a submodel selector to ``(callable, name)``.

    ``None`` → registry ``default``; a ``str`` → registry lookup (validated);
    a callable → used directly (name ``'custom'``).
    """
    if value is None:
        return registry[default], default
    if callable(value):
        return value, 'custom'
    if isinstance(value, str):
        if value not in registry:
            raise ConfigurationError(
                f"{label}={value!r} is not a known model; choose from "
                f"{sorted(registry)} or pass a callable")
        return registry[value], value
    raise ConfigurationError(
        f"{label} must be None, a name {sorted(registry)}, or a callable; "
        f"got {type(value).__name__}")


def _eval_submodel(fn, label, f, params):
    """Call a (built-in or user) submodel and validate its output.

    Custom callables may omit ``**_`` or return a wrong-shaped array; turn both
    into a typed :class:`ConfigurationError` instead of a raw ``TypeError`` /
    broadcasting failure deep inside the incoherent sum.
    """
    try:
        out = np.asarray(fn(f, **params), dtype=float)
    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(
            f"{label} model failed: {type(e).__name__}: {e}") from e
    if out.shape != f.shape:
        raise ConfigurationError(
            f"{label} model returned shape {out.shape}, expected {f.shape} "
            f"(one spectral level per input frequency)")
    return out


class WenzNoise:
    """
    Composite Wenz ambient-noise spectrum (dB re 1 µPa²/Hz).

    Composes five swappable component submodels — shipping, wind, rain,
    thermal, and turbulence — into the total incoherent ambient spectrum.
    Each component is chosen via a ``*_model`` argument; ``None`` selects the
    default. The **defaults** are the Canadian/DRDC composite (Tollefsen &
    Pecknold 2018; DRDC "WenzCurves"): ``shipping='wenz'``,
    ``wind='merklinger'``, ``rain='torres_costa'``, ``thermal='mellen'``,
    ``turbulence='wenz'`` (DRDC §2.1 eq. 4, ``107 − 33.2·log10(f_Hz)``).
    Built-in alternatives include the Coates/Stojanović (2007) ``'coates'``
    wind and shipping models. Register your own by name
    (``WIND_MODELS['mine'] = fn``) or pass a callable directly. Each component
    is exposed as a typed attribute; plotting lives in
    :func:`uacpy.visualization.plot_wenz`.

    Every level here is a **spectral** level (dB re 1 µPa²/Hz). Differencing
    one against a band-integrated source level in the sonar equation is a
    ``10·log10(w)`` error — 20 dB over a 100 Hz band; see
    :func:`uacpy.sonar.noise_background` for the rule and
    :func:`compute_windnoise` (``band_integrate=True``) for a band level.

    Parameters
    ----------
    frequencies : array-like
        Frequencies in Hz.
    wind_speed_kn : float
        Wind speed in **knots** at the 10 m reference height — the variable
        DRDC-RDDC-2022-D051 §2.3 eq. 8 fits, and the same unit
        :func:`uacpy.sonar.chapman_harris_surface` names ``wind_speed_kn``.
        Keyword-only, so the unit is stated at every call site; a wind reading
        in m/s passed here understates the total ambient level (measured
        5.7 dB at 1 kHz for a 10 m/s reading). Convert with
        ``kn = m_per_s * 1.9438445``.
    rain_rate : {'no', 'light', 'moderate', 'heavy', 'veryheavy'}
        Default ``'no'``.
    water_depth : {'deep', 'shallow'}
        Default ``'deep'``.
    shipping_level : {'no', 'low', 'medium', 'high'}
        Default ``'medium'``.
    wind_model, shipping_model, rain_model, thermal_model, turbulence_model
        Submodel selector for each component: ``None`` → registry default, a
        ``str`` name from the corresponding ``*_MODELS`` registry, or a
        callable ``model(frequencies, *, wind_speed_kn, water_depth,
        shipping_level, rain_rate, **_) -> ndarray`` (dB re 1 µPa²/Hz).

    Attributes
    ----------
    frequencies : ndarray
        Input frequency vector (1-D, in Hz).
    wind_speed_kn : float
        The wind speed the spectrum was built from, in knots.
    total : ndarray
        Incoherent sum of all five components, dB re 1 µPa²/Hz.
    shipping, wind, rain, thermal, turbulence : ndarray
        Per-source noise spectral levels, dB re 1 µPa²/Hz.
    models : dict
        The submodel name selected for each component (``'custom'`` for a
        user callable), e.g. ``{'wind': 'merklinger', 'turbulence': 'wenz', …}``.
    components : NoiseComponents
        Named-tuple view ``(total, wind, shipping, rain, thermal, turbulence)``.

    Notes
    -----
    Beaufort vs wind-speed reference (Urick 1984):

    ============  =========  ==============  ================
    Beaufort      Sea state  Wind (knots)    Wind (m/s)
    ============  =========  ==============  ================
    0             0          <1              0 – 0.2
    1             1/2        1 – 3           0.3 – 1.5
    2             1          4 – 6           1.6 – 3.3
    3             2          7 – 10          3.4 – 5.4
    4             3          11 – 16         5.5 – 7.9
    5             4          17 – 21         8.0 – 10.7
    6             5          22 – 27         10.8 – 13.8
    7             6          28 – 33         13.9 – 17.1
    8             6          34 – 40         17.2 – 20.7
    ============  =========  ==============  ================
    """

    def __init__(
        self,
        frequencies,
        *,
        wind_speed_kn,
        rain_rate='no',
        water_depth='deep',
        shipping_level='medium',
        wind_model=None,
        shipping_model=None,
        rain_model=None,
        thermal_model=None,
        turbulence_model=None,
    ):
        if water_depth not in ('deep', 'shallow'):
            raise ConfigurationError(
                f"water_depth must be 'deep' or 'shallow', got {water_depth!r}"
            )
        if shipping_level not in _SHIPPING_C2:
            raise ConfigurationError(
                f"shipping_level must be one of {list(_SHIPPING_C2)}, "
                f"got {shipping_level!r}"
            )
        if rain_rate not in _RAIN_INDEX:
            raise ConfigurationError(
                f"rain_rate must be one of {list(_RAIN_INDEX)}, "
                f"got {rain_rate!r}"
            )

        self.frequencies = np.asarray(frequencies, dtype=float).flatten()
        # Every component is a log10(f) fit; a DC (0 Hz) or negative bin — common
        # when a user passes a raw rfft grid — would yield log10(0)=-inf/NaN
        # before the sentinel masks run. Reject it up front with a clear message.
        # ``~(f > 0)`` rather than ``f <= 0`` so a NaN bin is refused as well.
        if self.frequencies.size == 0 or np.any(~(self.frequencies > 0)):
            raise ConfigurationError(
                "WenzNoise: frequencies must be > 0 Hz and finite (the "
                "empirical fits are log10(f)); drop the DC bin, e.g. "
                "frequencies[frequencies > 0]."
            )
        self.wind_speed_kn = float(wind_speed_kn)
        # ``not (w >= 0)`` rather than ``w < 0``: a NaN wind speed passes the
        # latter and then fails the ``> 0`` blend test inside the wind model,
        # landing on the documented -inf switched-off sentinel — an all-finite
        # total spectrum tens of dB below the true level, with no warning.
        if not (self.wind_speed_kn >= 0):
            raise ConfigurationError(
                f"WenzNoise: wind_speed_kn must be non-negative (knots) and "
                f"finite, got {self.wind_speed_kn:g}; the Coates wind model "
                f"takes √(wind) and would otherwise return NaN."
            )
        self.rain_rate = rain_rate
        self.water_depth = water_depth
        self.shipping_level = shipping_level

        f = self.frequencies

        # Resolve each component to a submodel (None → registry default, str →
        # named registry model, callable → custom). Every component is a
        # log10(f) fit returning dB re 1 µPa²/Hz; -inf marks a switched-off
        # source, so the incoherent logaddexp sum drops only silent sources.
        params = dict(wind_speed_kn=self.wind_speed_kn,
                      water_depth=water_depth,
                      shipping_level=shipping_level, rain_rate=rain_rate)
        wfn, wname = _resolve_submodel(wind_model, WIND_MODELS,
                                       'merklinger', 'wind_model')
        sfn, sname = _resolve_submodel(shipping_model, SHIPPING_MODELS,
                                       'wenz', 'shipping_model')
        rfn, rname = _resolve_submodel(rain_model, RAIN_MODELS,
                                       'torres_costa', 'rain_model')
        tfn, tname = _resolve_submodel(thermal_model, THERMAL_MODELS,
                                       'mellen', 'thermal_model')
        ufn, uname = _resolve_submodel(turbulence_model, TURBULENCE_MODELS,
                                       'wenz', 'turbulence_model')
        self.wind = _eval_submodel(wfn, 'wind', f, params)
        self.shipping = _eval_submodel(sfn, 'shipping', f, params)
        self.rain = _eval_submodel(rfn, 'rain', f, params)
        self.thermal = _eval_submodel(tfn, 'thermal', f, params)
        self.turbulence = _eval_submodel(ufn, 'turbulence', f, params)
        self.models = {'wind': wname, 'shipping': sname, 'rain': rname,
                       'thermal': tname, 'turbulence': uname}
        # Sum incoherent dB sources via logsumexp to avoid 10**(x/10) overflow
        # on very loud components (e.g. heavy rain at high frequency).
        ln10 = np.log(10.0)
        stack = np.stack([self.thermal, self.wind, self.shipping,
                          self.turbulence, self.rain])
        self.total = (10.0 / ln10) * np.logaddexp.reduce(
            stack * (ln10 / 10.0), axis=0)

    # ── Convenience ────────────────────────────────────────────────────

    @property
    def components(self):
        """Named component spectra (dB re 1 µPa²/Hz) as a
        :class:`NoiseComponents` namedtuple
        ``(total, wind, shipping, rain, thermal, turbulence)``."""
        return NoiseComponents(self.total, self.wind, self.shipping,
                               self.rain, self.thermal, self.turbulence)

    def as_psd(self, ref=1.0):
        """Linear total PSD, by default in **µPa²/Hz** — the *same* 1 µPa
        reference as :attr:`total` (dB re 1 µPa²/Hz), so the linear and dB
        views stay harmonised::

            10 * np.log10(w.as_psd()) == w.total      # to ~1e-14 dB

        The round trip through the linear domain is float64 arithmetic, so
        most bins come back bit-exact and the rest within ~1e-14 dB.
        ``ref`` rescales the output to another pressure unit and is the value
        of the dB reference (1 µPa) expressed in that unit, so ``ref=1e-6``
        (1 µPa in Pa) returns SI **Pa²/Hz** — ready for
        :func:`uacpy.acoustic_signal.synthesize_noise_from_psd`, which expects
        a linear PSD in the signal's own pressure units. For any ``ref``,
        ``10 * np.log10(as_psd(ref) / ref**2)`` recovers ``total`` to the
        same ~1e-14 dB.
        """
        return 10 ** (self.total / 10) * ref ** 2

    def __repr__(self):
        return (
            f"WenzNoise(n_frequencies={self.frequencies.size}, "
            f"wind={self.wind_speed_kn:g} kn, "
            f"depth={self.water_depth!r}, "
            f"shipping={self.shipping_level!r}, "
            f"rain={self.rain_rate!r}, "
            f"models={self.models})"
        )
