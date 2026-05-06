"""
Ambient-noise model — Tollefsen / Pecknold packaging.

`compute_windnoise` and the :class:`WenzNoise` class follow Tollefsen &
Pecknold "A simple yet practical ambient noise model" (DRDC-Atlantic
2018): wind / shipping / rain / thermal / turbulence components summed
in dB re 1 µPa²/Hz.

References
----------
Tollefsen, C. D. S. & Pecknold, S. (2018). *A simple yet practical
   ambient noise model.* Proc. UA 2018.
Wenz, G. M. (1962). *JASA* 34(12), 1936–1956.
Mellen, R. H. (1952). *JASA* 24(5), 478–480.
Piggott, C. L. (1964). *JASA* 36(11), 2152–2163.
Merklinger, H. M. (1979). *JASA* 65(4), 949–957.
Torres, F. & Costa, P. (2019). *J. Mar. Sci. Eng.* 7(11), 392.
Nichols, S. M. & Bradley, D. L. (2016). *JASA* 139(4), EL114.
"""
import numpy as _np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Wind noise (free function, used inside WenzNoise too)
# ─────────────────────────────────────────────────────────────────────────────


def compute_windnoise(f, u, water_depth='deep', band_integrate=False):
    """
    Wind-driven ambient noise level (dB re 1 µPa²/Hz), with the
    Piggott (1964) shallow-water adjustment.

    Parameters
    ----------
    f : ndarray or float
        Frequencies in Hz (1-Hz band assumed for a scalar).
    u : float
        Wind speed in knots.
    water_depth : {'deep', 'shallow'}
        Coefficient family. Default 'deep'.
    band_integrate : bool
        If True, return the band-integrated SPL (dB re 1 µPa²) where each
        band's bandwidth is set by the midpoints between consecutive
        input frequencies. Default False — return the spectral level
        (dB re 1 µPa²/Hz).

    Returns
    -------
    NL : ndarray
        Wind noise spectral level (dB re 1 µPa²/Hz), or band-integrated
        SPL (dB re 1 µPa²) if ``band_integrate=True``.

    Notes
    -----
    Translated from the IDL implementation by Dan Hutt, rewritten by Vic
    Young, and packaged in Tollefsen & Pecknold (2018).
    """

    # This all breaks down if u == 0 so account for that
    if u == 0:
        NL = _np.zeros_like(f)
    else:
        n2 = f.size
        f = _np.array(f).flatten()  # Make sure f is a 1D array
        if band_integrate:
            f2 = _np.concatenate(([0], f, [2 * f[-1] - f[-2]]))
            df = (f2[2:] - f2[:-2]) / 2
        else:
            df = _np.ones_like(f)

        # Bookkeeping:
        # Some constants
        f_wind = 2000  # Cutoff for wind noise section
        s1w    = 1.5   # Constant in wind calcs
        s2w    = -5.0  # Constant in wind calc
        a      = -25   # Curve melding exponent
        slope  = s2w * (0.1 / _np.log10(2))  # Slope at high freq
        NL     = _np.zeros_like(f)

        # Do the wind part for f <= 2000 Hz
        if water_depth == 'shallow':
            cst = 45
        elif water_depth == 'deep':
            cst = 42
        else:
            cst = 42  # default

        i_wind = f <= f_wind
        # so it doesn't crash if only f > 2000 are entered (arbitrary fallback):
        f_temp = f[i_wind] if _np.any(i_wind) else _np.array([2000])

        # These confusing letters were taken directly from the old code
        f0w             = 770 - 100 * _np.log10(u)
        L0w             = cst + 20 * _np.log10(u) - 17 * _np.log10(f0w / 770)
        L1w             = L0w + (s1w / _np.log10(2)) * _np.log10(f_temp / f0w)
        L2w             = L0w + (s2w / _np.log10(2)) * _np.log10(f_temp / f0w)
        Lw              = L1w * (1 + (L1w / L2w) ** (-a)) ** (1 / a)
        temp_noise_dist = 10 ** (Lw / 10)

        if _np.any(i_wind):
            NL[i_wind] = temp_noise_dist * df[i_wind]

        # Meld with a sensible line at freqs greater than 2000 Hz
        if _np.any(~i_wind):
            prop_const  = temp_noise_dist[-1] / f_temp[-1] ** slope
            NL[~i_wind] = prop_const * f[~i_wind] ** slope * df[~i_wind]

        NL = 10 * _np.log10(NL)

        if n2 != 1:
            NL = NL.reshape((n2,))

    return NL


# ─────────────────────────────────────────────────────────────────────────────
# Wenz composite — class API
# ─────────────────────────────────────────────────────────────────────────────


_SHIPPING_C2 = {'low': 1, 'medium': 4, 'high': 7, 'no': 4}
_RAIN_INDEX  = {'no': 0, 'light': 1, 'moderate': 2, 'heavy': 3, 'veryheavy': 4}
_RAIN_R0 = [0, 51.0769, 61.5358, 65.1107, 74.3464]
_RAIN_R1 = [0, 1.4687,  1.0147,  0.8226,  1.0131]
_RAIN_R2 = [0, -0.5232, -0.4255, -0.3825, -0.4258]
_RAIN_R3 = [0, 0.0335,  0.0277,  0.0251,  0.0277]


class WenzNoise:
    """
    Composite Wenz ambient-noise spectrum (dB re 1 µPa²/Hz).

    Computes shipping (Wenz 1962), wind (Merklinger 1979 + Piggott 1964
    shallow correction), rain (Torres & Costa 2019), thermal (Mellen
    1952), and turbulence (Nichols & Bradley 2016) components on
    construction. Each component is exposed as a typed attribute, so
    ``.plot()`` re-uses the same parameters without the caller having
    to pass them twice.

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

    Attributes
    ----------
    frequencies : ndarray
        Input frequency vector (1-D, in Hz).
    total : ndarray
        Incoherent sum of all five components, dB re 1 µPa²/Hz.
    shipping, wind, rain, thermal, turbulence : ndarray
        Per-source noise spectral levels, dB re 1 µPa²/Hz.

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
    ):
        if water_depth not in ('deep', 'shallow'):
            raise ValueError(
                f"water_depth must be 'deep' or 'shallow', got {water_depth!r}"
            )
        if shipping_level not in _SHIPPING_C2:
            raise ValueError(
                f"shipping_level must be one of {list(_SHIPPING_C2)}, "
                f"got {shipping_level!r}"
            )
        if rain_rate not in _RAIN_INDEX:
            raise ValueError(
                f"rain_rate must be one of {list(_RAIN_INDEX)}, "
                f"got {rain_rate!r}"
            )

        self.frequencies    = _np.asarray(frequencies, dtype=float).flatten()
        self.wind_speed     = float(wind_speed)
        self.rain_rate      = rain_rate
        self.water_depth    = water_depth
        self.shipping_level = shipping_level

        f = self.frequencies

        # Thermal (Mellen 1952) — deep-sea molecular contribution.
        thermal = -75.0 + 20.0 * _np.log10(f)
        thermal[thermal <= 0] = 1

        # Wind (Merklinger 1979 + Piggott 1964 shallow correction).
        wind = compute_windnoise(f, self.wind_speed, water_depth)

        # Shipping (Wenz 1962). c1 controls the spectral peak frequency
        # (deeper-water shipping band peaks lower); c2 sets the level via
        # the discrete shipping level.
        c1 = 30 if water_depth == 'deep' else 65
        c2 = _SHIPPING_C2[shipping_level]
        if shipping_level != 'no':
            shipping = 76 - 20 * (_np.log10(f) - _np.log10(c1)) ** 2 + 5 * (c2 - 4)
            shipping[shipping <= 0] = 1
        else:
            shipping = _np.zeros_like(f)

        # Turbulence (Nichols & Bradley 2016 — same coefficients as the
        # MATLAB ``calc_noise_level.m`` appendix in WenzCurves.pdf p.12).
        # NB: the prose in WenzCurves.pdf §2.1 quotes 107 − 33.2·log10(f)
        # instead; the appendix code uses the values below.
        turbulence = 108.5 - 32.5 * _np.log10(f)
        turbulence[turbulence <= 0] = 1

        # Rain (Torres & Costa 2019, valid up to ~7 kHz; melded above).
        ir = _RAIN_INDEX[rain_rate]
        fk = f / 1000.0
        rain = (
            _RAIN_R0[ir]
            + _RAIN_R1[ir] * fk
            + _RAIN_R2[ir] * fk ** 2
            + _RAIN_R3[ir] * fk ** 3
        )
        slope        = -5.0 * (0.1 / _np.log10(2))
        idxs_below_7k = _np.where(f < 7000)[0]
        if idxs_below_7k.size and (f > 7000).any():
            ind          = int(idxs_below_7k[-1])
            prop_const   = 10 ** (rain[ind] / 10) / f[ind] ** slope
            rain[f > 7000] = 10 * _np.log10(prop_const * f[f > 7000] ** slope)

        self.thermal    = thermal
        self.wind       = wind
        self.shipping   = shipping
        self.turbulence = turbulence
        self.rain       = rain
        self.total = 10 * _np.log10(
            10 ** (thermal / 10)
            + 10 ** (wind / 10)
            + 10 ** (shipping / 10)
            + 10 ** (turbulence / 10)
            + 10 ** (rain / 10)
        )

    # ── Convenience ────────────────────────────────────────────────────

    @property
    def components(self):
        """``(N, 6)`` ndarray with columns ``[total, shipping, wind, rain, thermal, turbulence]``."""
        return _np.column_stack(
            (self.total, self.shipping, self.wind,
             self.rain, self.thermal, self.turbulence)
        )

    def as_psd(self, ref=1e-6):
        """Linear total PSD in **Pa²/Hz**.

        ``self.total`` is in dB re ``ref²/Hz`` — by default
        ``ref = 1e-6 Pa = 1 µPa``, matching the underwater-acoustics
        convention used throughout :mod:`uacpy.noise`. The returned
        array is the linear power-spectral density in Pa²/Hz, ready for
        :func:`uacpy.signal.ssrp` (which expects linear PSD in the
        signal's own pressure units).
        """
        return 10 ** (self.total / 10) * ref ** 2

    def __repr__(self):
        return (
            f"WenzNoise(n_freq={self.frequencies.size}, "
            f"wind={self.wind_speed:g} kn, "
            f"depth={self.water_depth!r}, "
            f"shipping={self.shipping_level!r}, "
            f"rain={self.rain_rate!r})"
        )

    # ── Plot ───────────────────────────────────────────────────────────

    def plot(self, title='', show_components=True, ax=None, ymin=6, ymax=146):
        """Plot the noise spectrum.

        Parameters
        ----------
        title : str, optional
            Appended to the default ``'[ WENZ - Noise Level Estimate ]'``
            heading.
        show_components : bool, optional
            If True (default), overlay the per-source curves; otherwise
            plot only the total.
        ax : matplotlib Axes, optional
            Axes to draw on. When None, a new figure is created.
        ymin, ymax : float, optional
            Y-axis limits in dB. Default (6, 146) matches the reference
            implementation.

        Returns
        -------
        fig, ax : matplotlib Figure, Axes
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        f = self.frequencies
        if show_components:
            ax.semilogx(f, self.total, color='black', linewidth=2.0,
                        label=f'Total noise ({self.water_depth} water)')
            ax.semilogx(f, self.shipping, color='blue', linestyle='dashed',
                        label=f'Shipping noise ({self.shipping_level} traffic)')
            ax.semilogx(f, self.wind, color='green', linestyle='dashed',
                        label=f'Wind noise ({self.wind_speed:g} kn)')
            ax.semilogx(f, self.rain, color='orange', linestyle='dashed',
                        label=f'Rain noise ({self.rain_rate} rain)')
            ax.semilogx(f, self.thermal, color='red', linestyle='dashed',
                        label='Thermal noise')
            ax.semilogx(f, self.turbulence, color='purple', linestyle='dashed',
                        label='Turbulence noise')
        else:
            ax.semilogx(
                f, self.total, color='black', linewidth=2.0,
                label=(f'Total noise ({self.water_depth} water, '
                       f'{self.shipping_level} traffic, '
                       f'{self.wind_speed:g} kn, {self.rain_rate} rain)'),
            )

        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel(r'Noise Level [dB re 1$\mu$Pa$^2$/Hz]')
        ax.set_title(f'[ WENZ - Noise Level Estimate ] {title}')
        ax.set_xlim((f[0], f[-1]))
        ax.set_ylim((ymin, ymax))
        ax.legend()
        ax.grid(True)
        return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == '__main__':
    f = _np.linspace(1.0, 1e5, int(1e5 - 1))
    wenz = WenzNoise(
        f, wind_speed=24,
        water_depth='deep', shipping_level='high', rain_rate='heavy',
    )
    wenz.plot()
    plt.show()
