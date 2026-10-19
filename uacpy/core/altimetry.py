"""Altimetry shape carrier: sea-surface height as a function of range.

The top-surface analogue of :class:`uacpy.core.bathymetry.Bathymetry` — a 1-D
profile (``height`` vs ``range``). Heights are positive **up** (z = 0 at mean
sea level), so a crest is positive and a trough negative. Re-exported from
:mod:`uacpy.core.environment` for stable import paths.

Also the Pierson-Moskowitz sea-surface generator that realises one from a wind
speed (:func:`generate_sea_surface`, :meth:`Altimetry.from_sea_state`).
"""

import warnings
from typing import Optional, Tuple

import numpy as np
from uacpy.core._carrier import carrier

from uacpy.core.exceptions import ConfigurationError, NumericsWarning
from uacpy.core.units import KNOTS_PER_M_PER_S, knots_to_ms
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._grid import _RangeProfile
from uacpy.core._validate import (
    real_array_or_none, refuse_two_columns, require_finite,
    require_generator)
from uacpy.core.constants import STANDARD_GRAVITY
from uacpy.core._provenance import coerce_data_sources

__all__ = [
    'Altimetry', 'SEA_SURFACE_SAMPLES_PER_PEAK', 'SEA_SURFACE_MIN_POINTS',
    'SEA_SURFACE_MAX_POINTS', 'SEA_SURFACE_CALM_WIND',
    'DEFAULT_SEA_SURFACE_WIND_KN',
    'sea_surface_n_points', 'generate_sea_surface',
]


# eq=False: a dataclass __eq__ over ndarray fields raises; compare by identity.
@carrier(eq=False)
class Altimetry(_RangeProfile):
    """Sea-surface height (m, positive up) as a function of range (m).

    A 1-D grid library carrier mirroring :class:`Bathymetry`: select a height
    with :meth:`at` (nearest), :meth:`isel` (positional) or :meth:`eval`
    (interpolated). The single range axis is collapsed, so those return the
    **height value(s)** directly (a scalar for a scalar range, an array for an
    array of ranges).

    Attributes
    ----------
    ranges : ndarray, shape (N,)
        Range axis in metres, monotonically increasing.
    heights : ndarray, shape (N,)
        Surface height in metres at each range (positive up; any sign).
    data_sources : tuple
        Provenance of a fetched surface (tuple of ``DataProvenance``); empty
        for a literal/hand-built one. Aggregated by ``env.data_sources``.
    """

    ranges: np.ndarray
    heights: np.ndarray
    data_sources: tuple = ()

    _VALUE_FIELD = 'heights'
    _XARRAY_FIELDS = {'height': 'heights', 'range': 'ranges'}
    _VALUE_LABEL = 'sea-surface height'

    def __post_init__(self):
        self.data_sources = coerce_data_sources(self.data_sources, "Altimetry")
        self._init_range_profile()

    def _validate_values(self) -> None:
        require_finite(self.heights, "Altimetry heights",
                       hint="metres, positive up")

    # ── constructors ────────────────────────────────────────────────────────
    @classmethod
    def coerce(cls, value):
        """Coerce ``None`` / ``Altimetry`` / ``(N, 2)`` ``(range, height)``
        pairs into an :class:`Altimetry` (``None`` passes through — a flat
        z = 0 surface). Two columns ``(ranges, heights)`` are refused with
        the ``np.column_stack`` remedy: shape ``(2, N)`` with ``N != 2``, and
        two 1-D arrays or a tuple of two lists at any length. A list of
        tuples, a ``2 x 2`` nested list or a ``2 x 2`` array is read as two
        ``(range, height)`` pairs.

        Parameters
        ----------
        value : None, Altimetry or array_like
            The ``altimetry=`` value (see above).
        """
        if value is None or isinstance(value, Altimetry):
            return value
        arr = real_array_or_none(value)
        if arr is None:
            raise ConfigurationError(
                f"Altimetry: must be shape (N, 2) as [(range, height), ...]; "
                f"got non-numeric {value!r}.")
        refuse_two_columns(
            value, who='Altimetry', name='altimetry',
            columns='ranges, heights', pair='range, height',
            carrier='an Altimetry',
            remediation="Pass altimetry=np.column_stack([ranges, heights]) "
                        "or Altimetry(ranges=ranges, heights=heights).")
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ConfigurationError(
                f"Altimetry: must have shape (N, 2) as [(range, height), "
                f"...]; got shape {arr.shape}.")
        return cls(ranges=arr[:, 0], heights=arr[:, 1])

    @classmethod
    def from_sea_state(
        cls,
        rmax: float,
        wind_speed_19_5m_kn: Optional[float] = None,
        n_points: Optional[int] = None,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> 'Altimetry':
        """A Pierson-Moskowitz sea-surface realization as an :class:`Altimetry`.

        The carrier form of :func:`generate_sea_surface`, which draws the
        ``(range, height)`` samples; every argument is passed to it as given.

        Parameters
        ----------
        rmax : float
            Range extent (m).
        wind_speed_19_5m_kn : float, optional
            Wind speed (knots) at 19.5 m. ``None`` (default) is
            :data:`DEFAULT_SEA_SURFACE_WIND_KN` (10 m/s), as on
            :func:`generate_sea_surface`.
        n_points : int, optional
            Range samples; ``None`` sizes them from the sea state.
        rng : numpy.random.Generator, optional
            The generator the realization is drawn from.
        """
        return cls.coerce(generate_sea_surface(
            rmax, wind_speed_19_5m_kn, n_points, rng=rng))


#: Range samples per Pierson-Moskowitz nominal peak wavelength
#: ``lambda_p = 2*pi*U**2/g`` that a realization needs. Eight holds the
#: realized Hs within a few percent; a fixed 500 samples at U = 10 m/s
#: realizes 86 % of the requested Hs over 10 km, 2.7 % over 50 km and
#: numerically zero over 427 km, which is why the count scales with range.
SEA_SURFACE_SAMPLES_PER_PEAK = 8
#: Floor on a sea-state-sized realization, so a short range keeps a smooth
#: surface rather than the ~130 samples the peak wavelength alone asks for
#: over 1 km.
SEA_SURFACE_MIN_POINTS = 500
#: Ceiling on a sea-state-sized realization: 200 000 samples is a 3.2 MB
#: altimetry array and resolves the peak of a 5 m/s wind over 400 km. Past it
#: the count is capped and the under-resolution warned about, not allocated.
SEA_SURFACE_MAX_POINTS = 200_000
#: Wind (m/s) at or below which the sea is calm: its peak wavelength is
#: 0.16 m and its Hs 5 mm at the threshold, so the realization stands in for a flat
#: surface, keeps SEA_SURFACE_MIN_POINTS and is not warned about.
SEA_SURFACE_CALM_WIND = 0.5
#: The sea-surface generators' default wind, in knots: 10 m/s (Hs about
#: 2.1 m), uacpy guidance — the first example wind of the Pierson-Moskowitz
#: spectra in Brekhovskikh & Lysanov, *Fundamentals of Ocean Acoustics*,
#: Fig. 1.24 (wind at 19.5 m), written as that m/s value converted so the
#: default realisation is the 10 m/s one exactly.
DEFAULT_SEA_SURFACE_WIND_KN = 10.0 * KNOTS_PER_M_PER_S


def sea_surface_n_points(rmax: float, wind_speed_19_5m_kn: float) -> Tuple[int, int]:
    """``(n_points, needed)`` for a Pierson-Moskowitz realization.

    ``needed`` is the count that puts SEA_SURFACE_SAMPLES_PER_PEAK samples
    on the nominal peak wavelength ``2*pi*U**2/g`` over ``rmax``;
    ``n_points`` is that count clipped to [SEA_SURFACE_MIN_POINTS,
    SEA_SURFACE_MAX_POINTS], or SEA_SURFACE_MIN_POINTS for a calm sea. The
    one sizing rule behind :func:`generate_sea_surface` and
    :func:`uacpy.data.fetch_sea_surface`.
    """
    wind_ms = float(knots_to_ms(wind_speed_19_5m_kn))    # the PM spectrum's U, in m/s
    lambda_p = 2.0 * np.pi * wind_ms ** 2 / STANDARD_GRAVITY
    needed = int(np.ceil(SEA_SURFACE_SAMPLES_PER_PEAK * float(rmax)
                         / lambda_p)) + 1
    if wind_ms <= SEA_SURFACE_CALM_WIND:
        return SEA_SURFACE_MIN_POINTS, needed
    return (int(min(max(needed, SEA_SURFACE_MIN_POINTS),
                    SEA_SURFACE_MAX_POINTS)), needed)


def generate_sea_surface(
    rmax: float,
    wind_speed_19_5m_kn: Optional[float] = None,
    n_points: Optional[int] = None,
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a random sea surface realization from the Pierson-Moskowitz spectrum.

    Parameters
    ----------
    rmax : float
        Range extent of the surface in metres (the grid spans ``0..rmax``).
    wind_speed_19_5m_kn : float, optional
        Wind speed at 19.5 m height in knots, converted at entry to the
        m/s ``U`` of the Pierson-Moskowitz convention. The fully developed
        significant wave height is Hs = 4*sqrt(alpha/beta)*U^2/(2g) =
        0.021*U^2, U in m/s:
        - 10 kn (5.1 m/s): Hs ~ 0.6 m
        - 20 kn (10.3 m/s): Hs ~ 2.2 m
        - 30 kn (15.4 m/s): Hs ~ 5.0 m
        - 40 kn (20.6 m/s): Hs ~ 8.9 m
        ``None`` (default) is :data:`DEFAULT_SEA_SURFACE_WIND_KN` (10 m/s,
        Hs ~ 2.1 m).
    n_points : int, optional
        Number of range points in the output altimetry array. The surface is
        resolved with ``SEA_SURFACE_SAMPLES_PER_PEAK`` (8) samples per
        nominal peak wavelength ``lambda_p = 2*pi*U**2/g``, i.e.
        ``n_points >= 8 * rmax / lambda_p + 1`` (1250 points for 10 km at
        10 m/s, 19.44 kn). ``None`` (default) picks that count, at least 500 and at
        most ``SEA_SURFACE_MAX_POINTS`` (200 000) — past the cap it warns
        rather than allocating — and 500 for a calm sea (U <= 0.5 m/s, about
        1 kn), whose
        millimetre ripples stand in for a flat surface. An explicit coarser
        grid warns and returns a surface whose Hs falls short of the fully
        developed value.
    rng : numpy.random.Generator, optional
        The generator the realization is drawn from; pass one (e.g.
        ``np.random.default_rng(1)``) for a reproducible surface. ``None``
        (default) draws from a fresh, unseeded generator, so every call
        returns a different sea.

    Returns
    -------
    altimetry : ndarray, shape (n_points, 2)
        Column 0: range (m), Column 1: surface height (m, positive up).
        Suitable for passing directly to ``Environment(altimetry=...)``.

    References
    ----------
    Pierson, W. J. & Moskowitz, L. (1964). "A proposed spectral form for fully
    developed wind seas based on the similarity theory of S. A. Kitaigorodskii."
    JGR 69(24), 5181-5190. Spectrum and rms height as given by Medwin & Clay,
    *Fundamentals of Acoustical Oceanography*, eqs. (13.1.11) and (13.1.12).
    """
    if not np.isfinite(rmax) or rmax <= 0:
        raise ConfigurationError(
            f"generate_sea_surface: rmax must be a positive distance (m); "
            f"got {rmax}."
        )
    if wind_speed_19_5m_kn is None:
        wind_speed_19_5m_kn = DEFAULT_SEA_SURFACE_WIND_KN
    if not np.isfinite(wind_speed_19_5m_kn) or wind_speed_19_5m_kn <= 0:
        raise ConfigurationError(
            f"generate_sea_surface: wind_speed_19_5m_kn must be a positive speed in "
            f"knots; got {wind_speed_19_5m_kn}."
        )
    wind_ms = float(knots_to_ms(wind_speed_19_5m_kn))    # the PM spectrum's U, in m/s
    auto_n_points, needed = sea_surface_n_points(rmax, wind_speed_19_5m_kn)
    if n_points is not None:
        # A count is used as given: 600.9 is refused, not truncated to 600.
        if (isinstance(n_points, bool) or np.ndim(n_points) != 0
                or not float(n_points).is_integer()):
            raise ConfigurationError(
                f"generate_sea_surface: n_points must be a whole number of "
                f"points; got {n_points!r}.")
        n_points = int(n_points)
    else:
        n_points = auto_n_points
    if n_points < 2:
        raise ConfigurationError(
            f"generate_sea_surface: n_points must be >= 2; got {n_points}."
        )
    rng = require_generator(rng, "generate_sea_surface")
    g = STANDARD_GRAVITY

    ranges = np.linspace(0, rmax, n_points)
    dx = ranges[1] - ranges[0]

    # Spatial frequency grid (cycles/m)
    n_fft = n_points
    dk = 1.0 / (n_fft * dx)  # spatial freq resolution
    k = np.arange(1, n_fft // 2 + 1) * dk  # positive frequencies
    omega = np.sqrt(g * 2 * np.pi * k)  # deep-water dispersion: omega^2 = g*k_wave

    # Pierson-Moskowitz spectrum S(omega), M&C eq. (13.1.11):
    # S(omega) = (alpha * g^2 / omega^5) * exp(-beta * (omega_p / omega)^4)
    # with alpha = 8.1e-3, beta = 0.74 and the nominal spectral peak at
    # omega_p = g/W, W the wind speed 19.5 m above the surface.
    alpha_pm = 8.1e-3
    beta_pm = 0.74
    omega_p = g / wind_ms  # peak angular frequency
    S_omega = (alpha_pm * g**2 / omega**5) * np.exp(-beta_pm * (omega_p / omega)**4)

    # Convert to spatial spectrum S(k) via S(k) = S(omega) * domega/dk
    # with k in cycles/m: omega = sqrt(2*pi*g*k) so domega/dk = pi*g/omega
    domega_dk = np.pi * g / omega
    S_k = S_omega * domega_dk

    # The variance sits around the spectral peak (M&C 13.1.11's nominal
    # omega_p = g/W; the true maximum sits lower, at (4*beta/5)^(1/4)*omega_p,
    # so a grid sized on the nominal one errs on the fine side). A grid coarser
    # than SEA_SURFACE_SAMPLES_PER_PEAK samples per peak wavelength aliases
    # the spectrum away and returns a surface flatter than the wind's Hs.
    if wind_ms > SEA_SURFACE_CALM_WIND and n_points < needed:
        lambda_p = 2.0 * np.pi * wind_ms ** 2 / g
        remedy = (f"Shorten rmax: resolving it needs {needed} samples, "
                  f"past the {SEA_SURFACE_MAX_POINTS} cap"
                  if needed > SEA_SURFACE_MAX_POINTS
                  and n_points >= SEA_SURFACE_MAX_POINTS else
                  f"Pass n_points >= {needed}, or n_points=None to size the "
                  f"realization from the wind")
        warnings.warn(
            f"generate_sea_surface: {n_points} samples over {rmax:.0f} m "
            f"give dx = {dx:.1f} m, coarser than the "
            f"{lambda_p / SEA_SURFACE_SAMPLES_PER_PEAK:.1f} m that resolves "
            f"the {lambda_p:.1f} m wavelength of the *nominal* "
            f"Pierson-Moskowitz peak (omega_p = g/W, M&C 13.1.11) at "
            f"U = {wind_ms:g} m/s ({float(wind_speed_19_5m_kn):g} kn). "
            f"The realization aliases the wave spectrum away and its significant wave height falls short of "
            f"the fully developed {0.021 * wind_ms ** 2:.2f} m. "
            f"{remedy}.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

    # Random-phase realisation: each component carries variance S_k*dk, and a
    # cosine of amplitude a has variance a^2/2.
    amplitude = np.sqrt(2 * S_k * dk)
    phase = rng.uniform(0, 2 * np.pi, len(k))

    # Sum of a_j*cos(2*pi*k_j*x_m + phi_j) with k_j = j/(n_fft*dx) and
    # x_m = m*dx is an inverse DFT: the cosine arguments reduce to
    # 2*pi*j*m/n_fft, so placing a_j*exp(i*phi_j) at bin j and taking
    # Re(n_fft * ifft) evaluates the identical sum in O(n log n).
    spec = np.zeros(n_fft, dtype=complex)
    spec[1:n_fft // 2 + 1] = amplitude * np.exp(1j * phase)
    surface = n_fft * np.fft.ifft(spec).real

    return np.column_stack([ranges, surface])
