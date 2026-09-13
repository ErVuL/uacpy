"""APL-UW TR 9407 bottom forward-loss and backscattering models (10-100 kHz).

The high-frequency seabed models of the *APL-UW High-Frequency Ocean
Environmental Acoustic Models Handbook* (TR 9407, 1994), Section IV: a lossy
Rayleigh forward-loss model (IV.B, Eqs. 29-33) and the six-parameter
backscattering model (IV.C, Eqs. 34-66) that sums interface roughness
scattering — Kirchhoff near vertical, composite roughness at low angles, an
empirical large-roughness form for gravel and rock — and sediment volume
scattering. Both take the parameters of the report's Table 1, obtained from a
sediment name (Table 2), from grain size (Eqs. 2-10) or from geoacoustics.

Equation numbers below are the report's. The relief spectrum is
``W2(K) = (h0 K)^-gamma w2`` (Eq. 7) with ``h0 = 1 cm`` and ``w2`` in cm^4, so
every acoustic wavenumber in the roughness equations is used in cm^-1.

References
----------
APL-UW (1994). *High-Frequency Ocean Environmental Acoustic Models Handbook*,
Technical Report APL-UW TR 9407, Section IV (Bottom).
Mourad, P.D. & Jackson, D.R. (1989). Oceans '89 (the model's basis, TR 9407
ref. 9). Jackson, Winebrenner & Ishimaru (1986), JASA 79, 1410-1422.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from scipy.special import erf, gamma as _gamma

from uacpy.core.bottom import Bottom, BoundaryProperties, SeabedColumn
from uacpy.core.constants import DEFAULT_SOUND_SPEED, DEFAULT_WATER_DENSITY_G_CM3
from uacpy.core.exceptions import ConfigurationError
from uacpy.core._warn_frames import USER_FRAME_SKIP

__all__ = [
    'BottomParameters',
    'apl_uw_bottom_loss',
    'apl_uw_bottom_backscatter',
    'APL_UW_SEDIMENTS',
]

#: Water sound speed the report's Table 3 and Figure 2 were computed with
#: (TR 9407 p. IV-21), and the ``c1`` its loss-parameter relation Eq. 4 uses.
TABLE_WATER_SOUND_SPEED = 1528.0

#: Reference length of the relief spectrum, cm (Eq. 7).
_H0_CM = 1.0
#: Spectral exponent assigned in the absence of measurements (Eq. 8).
_DEFAULT_SPECTRAL_EXPONENT = 3.25
#: Grain-size range over which Eqs. 2-10 are defined (p. IV-8).
_GRAIN_SIZE_RANGE = (-1.0, 9.0)
#: Recommended limits on the inputs (Section IV.A.8, p. IV-17).
_LIMITS = {
    'density_ratio': (1.0, 3.0),
    'speed_ratio': (0.8, 3.0),
    'loss_parameter': (0.0, 0.1),
    'volume_parameter': (0.0, 1.0),
    'spectral_exponent': (2.4, 3.9),
    'spectral_strength': (0.0, 1.0),
}
_FREQUENCY_LIMITS_HZ = (10e3, 100e3)
_WATER_SPEED_LIMITS = (1400.0, 1600.0)
#: Below this grazing angle the report recommends using the value at it
#: (IV.C.4): the strength tends to -inf as the angle tends to zero.
_MIN_GRAZING_DEG = 0.001
#: Kirchhoff cross section is used at and above this angle (Eq. 37).
_KIRCHHOFF_MIN_DEG = 40.0
#: Level factor on the Kirchhoff cross section, ``2^(2(1-alpha)/alpha)``
#: (2.30 at gamma = 3.25). Eqs. 36-40 exactly as printed give a
#: near-vertical strength 3.56 +- 0.08 dB below the report's own Table 3
#: in all 24 cells where that branch governs — every sand, silt at both
#: sigma2, 10 to 100 kHz — while the composite-roughness, large-roughness
#: and volume terms match the table to 0.1 dB. The table was computed by the
#: authors' code and is what the model was validated as (IV.C.3), so it is
#: taken as the authority; this is the same as evaluating q_c with ``k/2``
#: in the level only. Whether the print or the code carries the slip cannot
#: be told from the report, and every table cell has gamma = 3.25, so the
#: gamma-dependence of the factor is the closed form's, not measured.
def _kirchhoff_level(alpha: float) -> float:
    return 2.0 ** (2.0 * (1.0 - alpha) / alpha)
#: Interpolation reference angles (Eq. 64), degrees.
_THETA_R_DEG, _DELTA_THETA_DEG = 7.0, 0.5
#: Critical angle assigned to bottoms slower than water (Eq. 54), degrees.
_SLOW_BOTTOM_CRITICAL_DEG = 2.5613
#: Three-point Gauss-Hermite slope-averaging weights and abscissae (Eqs. 47-48).
_GH_WEIGHTS = (0.295410, 1.181636, 0.295410)
_GH_NODES = (1.224745, 0.0, -1.224745)


def _grain_size_density_ratio(mz: float) -> float:
    """Eq. 2."""
    if mz < 1.0:
        return 0.007797 * mz ** 2 - 0.17057 * mz + 2.3139
    if mz < 5.3:
        return (-0.0165406 * mz ** 3 + 0.2290201 * mz ** 2
                - 1.1069031 * mz + 3.0455)
    return -0.0012973 * mz + 1.1565


def _grain_size_speed_ratio(mz: float) -> float:
    """Eq. 3."""
    if mz < 1.0:
        return 0.002709 * mz ** 2 - 0.056452 * mz + 1.2778
    if mz < 5.3:
        return (-0.0014881 * mz ** 3 + 0.0213937 * mz ** 2
                - 0.1382798 * mz + 1.3425)
    return -0.0024324 * mz + 1.0019


def _grain_size_alpha_over_f(mz: float) -> float:
    """Eq. 5, dB m^-1 kHz^-1 (Hamilton's parameterisation)."""
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


def _grain_size_loss_parameter(mz: float, speed_ratio: float) -> float:
    """Eq. 4 with Hamilton's alpha2/f: ``delta = (alpha2/f) nu c1 ln10 / 40 pi``
    with c1 in m/ms, the 1.528 m/ms the report states it used."""
    return (_grain_size_alpha_over_f(mz) * speed_ratio
            * (TABLE_WATER_SOUND_SPEED / 1000.0) * np.log(10.0) / (40.0 * np.pi))


def _grain_size_volume_parameter(mz: float) -> float:
    """Eq. 6."""
    return 0.002 if mz < 5.5 else 0.001


def _grain_size_spectral_strength(mz: float) -> float:
    """Eqs. 9-10: rms relief of a 100-cm track, then ``w2 = 0.00207 h^2 h0^2``
    (gamma = 3.25), cm^4."""
    h = ((2.03846 - 0.26923 * mz) / (1.0 + 0.076923 * mz) if mz < 5.0
         else 0.5)
    return 0.00207 * h ** 2 * _H0_CM ** 2


def _grain_size_from_speed_ratio(speed_ratio: float) -> float:
    """Invert Eq. 3 for Mz on [-1, 9] (monotone decreasing there): the
    report's preferred route from geoacoustics to grain size (p. IV-12)."""
    lo, hi = _GRAIN_SIZE_RANGE
    if speed_ratio >= _grain_size_speed_ratio(lo):
        return lo
    if speed_ratio <= _grain_size_speed_ratio(hi):
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _grain_size_speed_ratio(mid) > speed_ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class BottomParameters:
    """The six seabed inputs of TR 9407 Table 1.

    Parameters
    ----------
    density_ratio : float
        ``rho``, sediment to water mass density.
    speed_ratio : float
        ``nu``, sediment to water sound speed.
    loss_parameter : float
        ``delta``, imaginary to real sediment wavenumber (``2 delta = 1/Q``).
    volume_parameter : float
        ``sigma2``, sediment volume scattering cross section over the
        sediment attenuation coefficient — an empirical surface
        parameterisation of volume scattering (Eq. 66).
    spectral_strength : float
        ``w2`` (cm^4), the relief spectrum at ``2 pi / lambda = 1 cm^-1``.
    spectral_exponent : float
        ``gamma``, the relief-spectrum power-law exponent (3.25 by default,
        Eq. 8).

    Values outside the report's recommended limits (IV.A.8) warn: they are
    "extremes suggested by a combination of numerical and physical
    considerations" that "may yield suspect results". Build one with
    :meth:`from_sediment`, :meth:`from_grain_size`, :meth:`from_geoacoustics`,
    or from a uacpy seabed with :meth:`from_bottom` / :meth:`from_environment`,
    rather than by hand where you can.
    """
    density_ratio: float
    speed_ratio: float
    loss_parameter: float
    volume_parameter: float
    spectral_strength: float
    spectral_exponent: float = _DEFAULT_SPECTRAL_EXPONENT

    def __post_init__(self):
        for name, (lo, hi) in _LIMITS.items():
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ConfigurationError(
                    f"BottomParameters.{name} must be finite; got {value!r}.")
            strict_low = name in ('loss_parameter', 'spectral_strength')
            below = value <= lo if strict_low else value < lo
            if below or value > hi:
                warnings.warn(
                    f"BottomParameters.{name}={value:g} lies outside the "
                    f"limits TR 9407 recommends ({lo:g} {'<' if strict_low else '<='} "
                    f"{name} <= {hi:g}, Section IV.A.8): 'extreme values ... may "
                    f"yield suspect results'.",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
        # 0 < alpha < 1 (p. IV-27) is the spectral-exponent limit 2 < gamma < 4,
        # inside the recommended range; a value on or past it breaks Eqs. 36-41.
        if not 2.0 < float(self.spectral_exponent) < 4.0:
            raise ConfigurationError(
                f"BottomParameters.spectral_exponent must lie strictly between "
                f"2 and 4 (0 < alpha = gamma/2 - 1 < 1, TR 9407 p. IV-27); got "
                f"{self.spectral_exponent!r}.")

    @classmethod
    def from_grain_size(cls, grain_size_phi: float) -> 'BottomParameters':
        """The report's grain-size parameterisation (IV.A.4, Eqs. 2-10).

        ``grain_size_phi`` is the bulk mean grain size ``Mz`` in phi units,
        defined for ``-1 <= Mz <= 9``; values outside are clamped with a
        warning, the relations being fits over that interval only.
        """
        mz = float(grain_size_phi)
        if not np.isfinite(mz):
            raise ConfigurationError(
                f"BottomParameters.from_grain_size: grain_size_phi must be "
                f"finite; got {grain_size_phi!r}.")
        lo, hi = _GRAIN_SIZE_RANGE
        if mz < lo or mz > hi:
            warnings.warn(
                f"BottomParameters.from_grain_size: Mz={mz:g} is outside the "
                f"{lo:g} <= Mz <= {hi:g} interval TR 9407 Eqs. 2-10 are defined "
                f"on (p. IV-8); evaluating at the nearer end.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
            mz = min(max(mz, lo), hi)
        nu = _grain_size_speed_ratio(mz)
        return cls(density_ratio=_grain_size_density_ratio(mz),
                   speed_ratio=nu,
                   loss_parameter=_grain_size_loss_parameter(mz, nu),
                   volume_parameter=_grain_size_volume_parameter(mz),
                   spectral_strength=_grain_size_spectral_strength(mz),
                   spectral_exponent=_DEFAULT_SPECTRAL_EXPONENT)

    @classmethod
    def from_sediment(cls, name: str) -> 'BottomParameters':
        """Defaults by sediment name, TR 9407 Table 2 (p. IV-6).

        The rock, cobble and gravel rows are the table's constants; every
        other row is the table's ``Mz`` through :meth:`from_grain_size`,
        which reproduces its printed digits. The report cautions that the
        table "does not constitute the APL-UW bottom interaction model": for
        a given name the observed scattering strength spreads by an order of
        magnitude in ``sigma2`` and ``w2``.
        """
        key = ' '.join(str(name).lower().replace('-', ' ').replace('_', ' ')
                       .split())
        try:
            entry = APL_UW_SEDIMENTS[key]
        except KeyError:
            raise ConfigurationError(
                f"BottomParameters.from_sediment: unknown sediment {name!r}. "
                f"TR 9407 Table 2 names: {', '.join(sorted(APL_UW_SEDIMENTS))}."
            ) from None
        if isinstance(entry, BottomParameters):
            return entry
        return cls.from_grain_size(entry)

    @classmethod
    def from_geoacoustics(cls, *, sound_speed: float, density: float,
                          attenuation_dB_per_wavelength: float,
                          water_sound_speed: float,
                          water_density: float,
                          grain_size_phi: Optional[float] = None,
                          volume_parameter: Optional[float] = None,
                          spectral_strength: Optional[float] = None,
                          spectral_exponent: float = _DEFAULT_SPECTRAL_EXPONENT,
                          ) -> 'BottomParameters':
        """From measured surficial geoacoustics (IV.A.5).

        ``rho`` and ``nu`` are the ratios to the water; ``delta`` comes from
        the attenuation in dB per wavelength (``alpha = 40 pi log10(e) delta``,
        the definition behind Eq. 4). The two parameters geoacoustics cannot
        give, ``sigma2`` and ``w2``, are taken from grain size when supplied,
        otherwise from the grain size that inverts the sound-speed relation
        Eq. 3 — "the preferred means of determining Mz" (p. IV-12) — unless
        given explicitly.
        """
        nu = float(sound_speed) / float(water_sound_speed)
        rho = float(density) / float(water_density)
        delta = float(attenuation_dB_per_wavelength) * np.log(10.0) / (40.0 * np.pi)
        mz = (float(grain_size_phi) if grain_size_phi is not None
              else _grain_size_from_speed_ratio(nu))
        mz = min(max(mz, _GRAIN_SIZE_RANGE[0]), _GRAIN_SIZE_RANGE[1])
        return cls(
            density_ratio=rho, speed_ratio=nu, loss_parameter=delta,
            volume_parameter=(float(volume_parameter) if volume_parameter is not None
                              else _grain_size_volume_parameter(mz)),
            spectral_strength=(float(spectral_strength) if spectral_strength is not None
                               else _grain_size_spectral_strength(mz)),
            spectral_exponent=float(spectral_exponent))

    @classmethod
    def from_bottom(cls, bottom, *, water_sound_speed: float,
                    water_density: float = DEFAULT_WATER_DENSITY_G_CM3,
                    range: float = 0.0, method: str = 'auto',
                    ) -> 'BottomParameters':
        """From a uacpy seabed — the bridge from ``uacpy.data`` to this model.

        ``bottom`` is a :class:`~uacpy.core.bottom.BoundaryProperties`, a
        :class:`~uacpy.core.bottom.SeabedColumn` or a range-dependent
        :class:`~uacpy.core.bottom.Bottom` (read at ``range`` in m, nearest
        column). The **surficial** material is what scatters at these
        frequencies, so a layered column contributes its top layer and a
        pure half-space its half-space.

        ``method`` picks the route to the six parameters:

        - ``'grain-size'`` — the handbook's own relations (Eqs. 2-10) from the
          boundary's ``grain_size_phi``, which every seabed that
          ``fetch_environment`` builds from a grain size carries. Raises
          ``ConfigurationError`` when the boundary has none.
        - ``'geoacoustics'`` — :meth:`from_geoacoustics` on the boundary's
          ``sound_speed``, ``density`` and ``attenuation`` (dB/wavelength)
          against the given water values; ``grain_size_phi`` still supplies
          ``sigma2`` and ``w2`` when present. Shear is not part of the
          handbook's fluid model and is ignored.
        - ``'auto'`` (default) — ``'grain-size'`` when the boundary carries a
          grain size, else ``'geoacoustics'``. With the seabed fetched under
          ``bottom_model='apl-uw'`` the two routes give the same ``rho`` and
          ``nu``; under the default ``'hamilton'`` they differ by the two
          relations, and the grain-size route keeps the handbook's own
          seabed.

        ``water_sound_speed`` (m/s) and ``water_density`` (g/cm³) are the water
        at the seafloor the ratios are formed against;
        :meth:`from_environment` reads them off an ``Environment``.
        """
        if method not in ('auto', 'grain-size', 'geoacoustics'):
            raise ConfigurationError(
                f"BottomParameters.from_bottom: method must be 'auto', "
                f"'grain-size' or 'geoacoustics'; got {method!r}.")
        if isinstance(bottom, Bottom):
            column = bottom.at(range=range)
        elif isinstance(bottom, SeabedColumn):
            column = bottom
        elif isinstance(bottom, BoundaryProperties):
            column = SeabedColumn.from_halfspace(bottom)
        else:
            raise ConfigurationError(
                f"BottomParameters.from_bottom: expected a BoundaryProperties, "
                f"SeabedColumn or Bottom; got {type(bottom).__name__}.")
        top = column.at(depth=0.0)
        if top.acoustic_type != 'half-space':
            raise ConfigurationError(
                f"BottomParameters.from_bottom: the seabed is "
                f"{top.acoustic_type!r}, which carries no sediment to "
                f"parameterise.",
                remediation="Give the seabed a half-space (a grain size, a "
                            "sediment class or explicit cp / rho / "
                            "attenuation), or use from_sediment(name).")
        mz = top.grain_size_phi
        if method == 'grain-size' and mz is None:
            raise ConfigurationError(
                "BottomParameters.from_bottom: method='grain-size' needs a "
                "boundary with grain_size_phi, and this one has none.",
                remediation="Use method='geoacoustics' (or 'auto'), or build "
                            "the seabed from a grain size.")
        if method == 'grain-size' or (method == 'auto' and mz is not None):
            return cls.from_grain_size(mz)
        missing = [name for name in ('sound_speed', 'density', 'attenuation')
                   if getattr(top, name) is None]
        if missing:
            raise ConfigurationError(
                f"BottomParameters.from_bottom: the half-space has no "
                f"{', '.join(missing)}, which the geoacoustics route needs.")
        return cls.from_geoacoustics(
            sound_speed=top.sound_speed, density=top.density,
            attenuation_dB_per_wavelength=top.attenuation,
            water_sound_speed=water_sound_speed, water_density=water_density,
            grain_size_phi=mz)

    @classmethod
    def from_environment(cls, env, *, range: float = 0.0,
                         method: str = 'auto') -> 'BottomParameters':
        """From an :class:`~uacpy.core.environment.Environment`.

        :meth:`from_bottom` on ``env.bottom`` at ``range`` (m), with the water
        sound speed read from ``env`` at the seafloor under that range and
        ``env.water_density`` — so a seabed fetched by
        :func:`uacpy.data.fetch_environment` (``bottom_sources=`` and, for
        the handbook's own relations, ``bottom_model='apl-uw'``) drives the
        scattering model in one call.
        """
        depth = float(env.bathymetry.eval(range=range))
        water_c = float(np.asarray(env.get_sound_speed(depth, range=range)).ravel()[0])
        return cls.from_bottom(env.bottom, water_sound_speed=water_c,
                               water_density=float(env.water_density),
                               range=range, method=method)

    def with_(self, **changes) -> 'BottomParameters':
        """A copy with some fields replaced (e.g. a fitted ``sigma2``)."""
        return replace(self, **changes)


def _rock(speed_ratio: float, spectral_strength: float) -> BottomParameters:
    return BottomParameters(density_ratio=2.5, speed_ratio=speed_ratio,
                            loss_parameter=0.01374, volume_parameter=0.002,
                            spectral_strength=spectral_strength,
                            spectral_exponent=_DEFAULT_SPECTRAL_EXPONENT)


#: TR 9407 Table 2: a ready :class:`BottomParameters` for the rock and cobble
#: rows, the table's ``Mz`` for every other row (see ``from_sediment``).
APL_UW_SEDIMENTS = {
    'rough rock': _rock(2.5, 0.20693),
    'rock': _rock(2.5, 0.01862),
    # w2 read from the scan as 0.0186; 0.0156 reproduces the table's cobble
    # column to 0.3 dB where 0.0186 misses by 1.7 dB at low angles.
    'cobble': _rock(1.8, 0.0156),
    'gravel': _rock(1.8, 0.0156),
    'pebble': _rock(1.8, 0.0156),
    'sandy gravel': -1.0,
    'very coarse sand': -0.5,
    'muddy sandy gravel': 0.0,
    'coarse sand': 0.5,
    'gravelly sand': 0.5,
    'gravelly muddy sand': 1.0,
    'medium sand': 1.5,
    'muddy gravel': 2.0,
    'fine sand': 2.5,
    'silty sand': 2.5,
    'muddy sand': 3.0,
    'very fine sand': 3.5,
    'clayey sand': 4.0,
    'coarse silt': 4.5,
    'sandy silt': 5.0,
    'gravelly mud': 5.0,
    'medium silt': 5.5,
    'sand silt clay': 5.5,
    'sandy mud': 6.0,
    'fine silt': 6.5,
    'clayey silt': 6.5,
    'sandy clay': 7.0,
    'very fine silt': 7.5,
    'silty clay': 8.0,
    'clay': 9.0,
}


def _grazing(caller: str, grazing_deg) -> np.ndarray:
    theta = np.asarray(grazing_deg, dtype=float)
    if np.any(~np.isfinite(theta) | (theta < 0.0) | (theta > 90.0)):
        raise ConfigurationError(
            f"{caller}: grazing angles must be finite and within 0-90 deg "
            f"(TR 9407 IV.A.8); got {theta!r}.")
    return theta


def _rayleigh(theta_rad, p: BottomParameters):
    """Eqs. 30-33: ``R``, ``P`` for the lossy Rayleigh coefficient."""
    kappa = (1.0 + 1j * p.loss_parameter) / p.speed_ratio
    # The report defines the root as the one whose phase is half the
    # argument's (IV.B.3) — numpy's principal square root.
    P = np.sqrt(kappa ** 2 - np.cos(theta_rad) ** 2 + 0j)
    y = p.density_ratio * np.sin(theta_rad) / P
    return (y - 1.0) / (y + 1.0), P, kappa


def apl_uw_bottom_loss(grazing_deg, params: BottomParameters):
    """Bottom forward reflection loss, TR 9407 IV.B (Eqs. 29-33), dB.

    ``r0 = -20 log10 |R(theta)|`` with the lossy Rayleigh coefficient of a
    fluid half-space, ``R = (y-1)/(y+1)``, ``y = rho sin(theta) / P``,
    ``P = sqrt(kappa^2 - cos^2 theta)``, ``kappa = (1 + i delta)/nu``. Uses
    ``rho``, ``nu`` and ``delta`` only, and has no frequency dependence. The
    report's accuracy statement (IV.B.2): data at 20-30 kHz and 5-30 deg
    grazing, errors about equal to the loss itself on soft bottoms (a
    predicted 8 dB at 5 deg matched to within 6 dB) and 1-2 dB on hard sand.
    The model neglects roughness, so on rough bottoms it under-predicts the
    energy lost from the specular direction (IV.B.2, p. IV-20).

    Parameters
    ----------
    grazing_deg : float or array
        Grazing angle from the horizontal, 0-90 degrees.
    params : BottomParameters
    """
    theta = np.deg2rad(_grazing('apl_uw_bottom_loss', grazing_deg))
    R, _, _ = _rayleigh(theta, params)
    with np.errstate(divide='ignore'):
        return -20.0 * np.log10(np.abs(R))


def _shadowing(theta_rad, s):
    """Eqs. 43-45; unity for ``t > 2`` as IV.C.4 recommends."""
    t = np.tan(theta_rad) / s
    with np.errstate(divide='ignore', invalid='ignore'):
        Q = (np.exp(-t ** 2) / np.sqrt(np.pi) - t * (1.0 - erf(t))) / (4.0 * t)
        S = (1.0 - np.exp(-2.0 * Q)) / (2.0 * Q)
    return np.where(t > 2.0, 1.0, S)


def _slope_average(theta_deg, cross_section, theta_s_deg):
    """Eqs. 46-49: three-point Gauss-Hermite average of a cross section over
    bottom slope; an argument below 0 deg is set to 0 (where the cross
    sections vanish), above 90 deg to 90 deg (p. IV-30)."""
    total = np.zeros_like(theta_deg, dtype=float)
    for weight, node in zip(_GH_WEIGHTS, _GH_NODES):
        arg = np.clip(theta_deg - node * theta_s_deg, 0.0, 90.0)
        total = total + weight * cross_section(np.deg2rad(arg))
    return total / np.sqrt(np.pi)


def _interp(x):
    """Eq. 57 with the IV.C.4 clamps (``f = 1`` below -40, ``0`` above 40)."""
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(x))


def apl_uw_bottom_backscatter(grazing_deg, frequency: float,
                              params: BottomParameters, *,
                              water_sound_speed: float = DEFAULT_SOUND_SPEED):
    """Bottom backscattering strength, TR 9407 IV.C (Eqs. 34-66), dB.

    ``S_b = 10 log10(sigma_r + sigma_v)``: interface-roughness scattering
    ``sigma_r`` — the Kirchhoff approximation near vertical (Eqs. 35-40),
    the composite-roughness approximation with shadowing and slope
    averaging elsewhere (Eqs. 41-52), an empirical large-roughness form for
    gravel and rock (Eqs. 53-56), joined by the logistic interpolations of
    Eqs. 57-64 — plus sediment volume scattering ``sigma_v`` (Eqs. 65-66).

    Parameters
    ----------
    grazing_deg : float or array
        Grazing angle from the horizontal, 0-90 degrees. Below 0.001 deg the
        value at 0.001 deg is returned, as IV.C.4 recommends: the strength
        tends to -inf at zero grazing.
    frequency : float
        Acoustic frequency (Hz). The report's band is 10-100 kHz; outside it
        the call warns.
    params : BottomParameters
        The six Table 1 inputs.
    water_sound_speed : float
        Sound speed just above the bottom (m/s), which sets the acoustic
        wavenumber ``k = 2 pi f / c1`` (p. IV-5). The report's Table 3 and
        Figure 2 used 1528 m/s (``TABLE_WATER_SOUND_SPEED``).

    Returns
    -------
    ndarray
        Backscattering strength (dB), the shape of ``grazing_deg``.

    Notes
    -----
    Model uncertainty (IV.C.3): about 3 dB for well-characterised sand and
    silt above 5 deg grazing, 5 and 10 dB when poorly characterised, about
    10 dB for rock and gravel above 15 deg and more below. For soft bottoms
    (``Mz > 3``) the output is most sensitive to ``sigma2``, which is best
    fitted to backscatter data; for harder ones to ``gamma`` and ``w2``.
    """
    theta_deg = _grazing('apl_uw_bottom_backscatter', grazing_deg)
    f = float(frequency)
    c1 = float(water_sound_speed)
    if not (np.isfinite(f) and f > 0.0 and np.isfinite(c1) and c1 > 0.0):
        raise ConfigurationError(
            f"apl_uw_bottom_backscatter: frequency and water_sound_speed must "
            f"be > 0 and finite; got {frequency!r}, {water_sound_speed!r}.")
    lo, hi = _FREQUENCY_LIMITS_HZ
    if not lo <= f <= hi:
        warnings.warn(
            f"apl_uw_bottom_backscatter: frequency {f:g} Hz is outside the "
            f"{lo/1e3:g}-{hi/1e3:g} kHz band TR 9407 recommends (IV.A.8); the "
            f"model was fitted there and carries no accuracy statement outside.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
    lo, hi = _WATER_SPEED_LIMITS
    if not lo <= c1 <= hi:
        warnings.warn(
            f"apl_uw_bottom_backscatter: water_sound_speed {c1:g} m/s is "
            f"outside the {lo:g}-{hi:g} m/s range TR 9407 recommends (IV.A.8).",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)

    p = params
    theta_deg = np.maximum(theta_deg, _MIN_GRAZING_DEG)
    theta = np.deg2rad(theta_deg)
    rho, nu, delta = p.density_ratio, p.speed_ratio, p.loss_parameter
    gamma_, w2 = p.spectral_exponent, p.spectral_strength
    k = 2.0 * np.pi * f / (c1 * 100.0)          # cm^-1, see module docstring

    R90, _, _ = _rayleigh(np.pi / 2.0, p)
    R90_sq = float(np.abs(R90) ** 2)

    # --- Kirchhoff approximation, Eqs. 35-40 ---------------------------------
    alpha = gamma_ / 2.0 - 1.0
    ch2 = (2.0 * np.pi * w2 * _gamma(2.0 - alpha) * 2.0 ** (-2.0 * alpha)
           / (_H0_CM ** gamma_ * alpha * (1.0 - alpha) * _gamma(1.0 + alpha)))
    qc = ch2 * 2.0 ** (1.0 - 2.0 * alpha) * k ** (2.0 * (1.0 - alpha))
    a = (8.0 * alpha ** 2 * _gamma(1.0 / (2.0 * alpha) + 0.5)
         / (_gamma(0.5) * _gamma(1.0 / alpha) * _gamma(1.0 / (2.0 * alpha)))
         ) ** (2.0 * alpha)
    b = a ** (0.5 - 1.0 / (2.0 * alpha)) * _gamma(1.0 / alpha) / (2.0 * alpha)
    sigma_kr = np.where(
        theta_deg >= _KIRCHHOFF_MIN_DEG,
        _kirchhoff_level(alpha) * b * qc * R90_sq
        / (8.0 * np.pi * (np.cos(theta) ** (4.0 * alpha)
                                          + a * qc ** 2 * np.sin(theta) ** 4)
                           ** ((1.0 + alpha) / (2.0 * alpha))),
        0.0)

    # --- composite roughness, Eqs. 41-52 --------------------------------------
    s2 = ((2.0 * np.pi * w2 * _H0_CM ** (-gamma_)) ** (1.0 / alpha)
          / (2.0 * (1.0 - alpha)) * (k ** 2 / alpha) ** ((1.0 - alpha) / alpha))
    s = float(np.sqrt(s2))
    theta_s_deg = s * 180.0 / np.pi                       # Eq. 49
    kappa = (1.0 + 1j * delta) / nu

    def sigma_pr(th):                                     # Eqs. 50-52
        P = np.sqrt(kappa ** 2 - np.cos(th) ** 2 + 0j)
        Y = (((rho - 1.0) ** 2 * np.cos(th) ** 2 + rho ** 2 - kappa ** 2)
             / (rho * np.sin(th) + P) ** 2)
        K = np.sqrt(4.0 * k ** 2 * np.cos(th) ** 2 + (k / 10.0) ** 2)
        W2 = (_H0_CM * K) ** (-gamma_) * w2                # Eq. 7
        return 4.0 * k ** 4 * np.sin(th) ** 4 * np.abs(Y) ** 2 * W2

    shadow = _shadowing(theta, s)
    sigma_cr = shadow * _slope_average(theta_deg, sigma_pr, theta_s_deg)

    # --- large roughness, Eqs. 53-56 -----------------------------------------
    theta_c = (np.degrees(np.arccos(1.0 / nu)) if nu >= 1.001
               else _SLOW_BOTTOM_CRITICAL_DEG)
    m = 0.7263 * s ** (-1.0 / 3.0)
    sigma_1 = (0.04682 * s ** 1.25 * nu ** 3.25
               * ((1.0 - 2.0 / rho) * nu ** (-2.0) + 1.0) ** 2
               / (1.0 + 3.54 * theta_s_deg / theta_c))
    crit = 1.0 + 0.81 * theta_c ** 2 / theta_deg ** 2
    sigma_lr = (sigma_1 * np.sin(np.deg2rad(180.0 / crit)) ** m
                + 0.0260 * R90_sq
                / (s2 * (1.0 + (theta_deg - 90.0) ** 2 / (2.6 * theta_s_deg ** 2)) ** 1.9
                   * crit))

    # --- interpolations, Eqs. 57-64 -------------------------------------------
    c4 = 1000.0 ** (1.0 / (1.0 + alpha)) * (a * qc ** 2) ** (1.0 / alpha)
    cos_kdb = (1.0 / c4 + 4.0) ** (-0.25)
    fx = _interp((np.cos(theta) - cos_kdb) / 0.0125)
    sigma_mr = fx * sigma_kr + (1.0 - fx) * sigma_cr
    fy = _interp((np.degrees(np.arctan(s)) - _THETA_R_DEG) / _DELTA_THETA_DEG)
    sigma_r = fy * sigma_mr + (1.0 - fy) * sigma_lr

    # --- sediment volume scattering, Eqs. 65-66 -------------------------------
    sigma2 = p.volume_parameter

    def sigma_pv(th):
        R, P, _ = _rayleigh(th, p)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = (5.0 * delta * sigma2 * np.abs(1.0 - R ** 2) ** 2 * np.sin(th) ** 2
                   / (nu * np.log(10.0) * np.abs(P) ** 2 * np.imag(P)))
        return np.where(np.isfinite(out), out, 0.0)

    sigma_v = shadow * _slope_average(theta_deg, sigma_pv, theta_s_deg)

    with np.errstate(divide='ignore'):
        return 10.0 * np.log10(sigma_r + sigma_v)
