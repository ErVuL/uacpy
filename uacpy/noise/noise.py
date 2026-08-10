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
        surface-noise source has no power).
    water_depth : {'deep', 'shallow'}
        Coefficient family. Default 'deep'.
    band_integrate : bool
        If True, return the band-integrated SPL (dB re 1 µPa²) where each
        band's bandwidth is set by the midpoints between consecutive
        input frequencies. Default False — return the spectral level
        (dB re 1 µPa²/Hz). Use the band form to pair wind noise with a
        band-integrated source level; :class:`WenzNoise` is spectral-only.

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
    if u < 0:
        raise ConfigurationError(
            f"compute_windnoise: wind speed u must be non-negative (knots), got {u}"
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
            # Band edges at the midpoints between consecutive frequencies; the
            # two outer bands span only the half-spacing to their single
            # neighbour (df[0]=(f[1]-f[0])/2, df[-1]=(f[-1]-f[-2])/2). A leading
            # 0 / symmetric extrapolation would over-weight the end bands.
            mids = (f[1:] + f[:-1]) / 2
            edges = np.concatenate(([f[0]], mids, [f[-1]]))
            df = edges[1:] - edges[:-1]
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

        if water_depth == 'shallow':
            cst = 45
        elif water_depth == 'deep':
            cst = 42
        else:
            warnings.warn(
                f"compute_windnoise: water_depth={water_depth!r} not "
                "recognised ('shallow' / 'deep'); falling back to deep "
                "(cst=42).",
                UserWarning, stacklevel=2,
            )
            cst = 42

        i_wind = f <= f_wind
        # With no sub-cutoff sample in the grid the block below computes a level
        # that the empty mask never assigns, so the placeholder value only has
        # to stay inside the fit's validity range.
        f_below_cutoff = f[i_wind] if np.any(i_wind) else np.array([2000])

        f0w = 770 - 100 * np.log10(u)
        L0w = cst + 20 * np.log10(u) - 17 * np.log10(f0w / 770)
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
        with np.errstate(invalid='ignore', divide='ignore'):
            blend = 1 + (L1w / L2w) ** (-a)
            Lw = np.where(blend > 0, L1w * np.abs(blend) ** (1 / a), -np.inf)
        psd_below_cutoff = 10 ** (Lw / 10)

        if np.any(i_wind):
            NL[i_wind] = psd_below_cutoff * df[i_wind]

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

def _thermal_mellen(frequencies, **_):
    """Thermal (Mellen 1952; DRDC §2.5 eq. 22): ``-75 + 20·log10(f_Hz)``."""
    return -75.0 + 20.0 * np.log10(frequencies)


def _wind_merklinger(frequencies, *, wind_speed, water_depth, **_):
    """Wind (Merklinger 1979 + Piggott 1964 shallow correction); DRDC §2.3."""
    return compute_windnoise(frequencies, wind_speed, water_depth)


def _wind_coates(frequencies, *, wind_speed, **_):
    """Wind (Coates 1989 / Stojanović 2007, the standard UW-comms form):
    ``50 + 7.5·√w + 20·log10(f) − 40·log10(f + 0.4)`` with ``f`` in kHz and
    ``w`` the wind speed in m/s (converted here from the knots input)."""
    fk = np.asarray(frequencies) / 1000.0
    w_ms = float(wind_speed) / 1.9438445          # knots → m/s
    return (50.0 + 7.5 * np.sqrt(w_ms) + 20.0 * np.log10(fk)
            - 40.0 * np.log10(fk + 0.4))


def _shipping_wenz(frequencies, *, shipping_level, water_depth, **_):
    """Shipping (Wenz 1962; DRDC §2.2 eq. 5-7)."""
    f = frequencies
    if shipping_level == 'no':
        return np.full_like(f, -np.inf)
    c1 = 30 if water_depth == 'deep' else 65
    c2 = _SHIPPING_C2[shipping_level]
    return 76 - 20 * (np.log10(f) - np.log10(c1)) ** 2 + 5 * (c2 - 4)


_COATES_SHIP_ACTIVITY = {'no': None, 'low': 0.0, 'medium': 0.5, 'high': 1.0}


def _shipping_coates(frequencies, *, shipping_level, **_):
    """Shipping (Coates 1989 / Stojanović 2007 turbulent-shipping form):
    ``40 + 20(s − 0.5) + 26·log10(f) − 60·log10(f + 0.03)`` with ``f`` in kHz
    and ``s`` the shipping-activity factor in [0, 1] (low/medium/high →
    0/0.5/1; ``'no'`` is silent)."""
    s = _COATES_SHIP_ACTIVITY[shipping_level]
    fk = np.asarray(frequencies) / 1000.0
    if s is None:
        return np.full_like(fk, -np.inf)
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
    m_t = -10.0 / np.log10(2.0)
    return 107.0 + m_t * np.log10(np.asarray(frequencies))


def _rain_torres_costa(frequencies, *, rain_rate, **_):
    """Rain (Torres & Costa 2019; DRDC §2.4 eq. 21), valid to ~7 kHz; melded above.

    The report's eq. (21) writes the cubic in a bare ``f``; its Table 2
    coefficients and the Annex A.1 listing (``fk = f/1000``) both take that
    argument in **kHz**, so ``cubic`` below is defined on kHz: the Hz grid is
    divided by 1000 and the 7 kHz anchor is passed as ``7.0``.
    """
    f = frequencies
    if rain_rate == 'no':
        return np.full_like(f, -np.inf)
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
    wind_speed : float
        Wind speed in **knots**.
    rain_rate : {'no', 'light', 'moderate', 'heavy', 'veryheavy'}
        Default ``'no'``.
    water_depth : {'deep', 'shallow'}
        Default ``'deep'``.
    shipping_level : {'no', 'low', 'medium', 'high'}
        Default ``'medium'``.
    wind_model, shipping_model, rain_model, thermal_model, turbulence_model
        Submodel selector for each component: ``None`` → registry default, a
        ``str`` name from the corresponding ``*_MODELS`` registry, or a
        callable ``model(frequencies, *, wind_speed, water_depth,
        shipping_level, rain_rate, **_) -> ndarray`` (dB re 1 µPa²/Hz).

    Attributes
    ----------
    frequencies : ndarray
        Input frequency vector (1-D, in Hz).
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
        wind_speed,
        *,
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
        if self.frequencies.size == 0 or np.any(self.frequencies <= 0):
            raise ConfigurationError(
                "WenzNoise: frequencies must be > 0 Hz (the empirical fits are "
                "log10(f)); drop the DC bin, e.g. frequencies[frequencies > 0]."
            )
        self.wind_speed = float(wind_speed)
        if self.wind_speed < 0:
            raise ConfigurationError(
                f"WenzNoise: wind_speed must be non-negative (knots), got "
                f"{self.wind_speed:g}; the Coates wind model takes √(wind) and "
                f"would otherwise return NaN."
            )
        self.rain_rate = rain_rate
        self.water_depth = water_depth
        self.shipping_level = shipping_level

        f = self.frequencies

        # Resolve each component to a submodel (None → registry default, str →
        # named registry model, callable → custom). Every component is a
        # log10(f) fit returning dB re 1 µPa²/Hz; -inf marks a switched-off
        # source, so the incoherent logaddexp sum drops only silent sources.
        params = dict(wind_speed=self.wind_speed, water_depth=water_depth,
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

            10 * np.log10(w.as_psd()) == w.total      # exactly

        ``ref`` rescales the output to another pressure unit and is the value
        of the dB reference (1 µPa) expressed in that unit, so ``ref=1e-6``
        (1 µPa in Pa) returns SI **Pa²/Hz** — ready for
        :func:`uacpy.acoustic_signal.synthesize_noise_from_psd`, which expects
        a linear PSD in the signal's own pressure units. For any ``ref``,
        ``10 * np.log10(as_psd(ref) / ref**2) == total``.
        """
        return 10 ** (self.total / 10) * ref ** 2

    def __repr__(self):
        return (
            f"WenzNoise(n_frequencies={self.frequencies.size}, "
            f"wind={self.wind_speed:g} kn, "
            f"depth={self.water_depth!r}, "
            f"shipping={self.shipping_level!r}, "
            f"rain={self.rain_rate!r}, "
            f"models={self.models})"
        )
