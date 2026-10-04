"""Closed-form reference fields for checking the numerical engines.

The four textbook solutions Jensen, Kuperman, Porter & Schmidt,
*Computational Ocean Acoustics* (2nd ed., 2011) present as benchmarks for
wave-equation codes, returned as the same complex-pressure :class:`Field`
an engine returns, so ``compare_models`` and :mod:`uacpy.metrics` take them
unchanged:

========================  ====================================================
:func:`free_field`        point source in an unbounded medium (JKPS §2.3.2,
                          Eq. 2.50)
:func:`lloyd_mirror`      point source below a pressure-release surface —
                          the image solution (JKPS §1.4.2.1, Eq. 1.19)
:func:`ideal_waveguide`   pressure-release surface over a rigid or
                          pressure-release bottom, modal sum (JKPS §2.4.4,
                          Eq. 2.150)
:func:`pekeris`           isovelocity water over a lossless fluid half-space,
                          trapped-mode sum (JKPS §2.4.5)
========================  ====================================================

Conventions are the engines': the travelling-wave phase ``exp(-i k R)``
(``phase_reference='travelling_wave'``), and a point source normalised to
unit pressure amplitude at 1 m, so ``Field.tl`` is transmission loss
re 1 m. A modal sum is ``p = (i pi / rho(z_s)) sum_m Psi_m(z_s) Psi_m(z)
H_0^(1)(k_rm r)`` (JKPS Eq. 5.13 times 4 pi for the 1 m normalisation),
conjugated to the travelling-wave form, with modes normalised to
``integral Psi_m^2 / rho dz = 1``.

A sample at zero distance from the source (free field, image), at range 0
(modal sums) or below the seabed of an ideal waveguide has no finite value
and is ``NaN``. So is every sample of an ideal waveguide at a mode's exact
cutoff frequency; see :func:`ideal_waveguide`.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import brentq

from uacpy.core.acoustics.boundaries import pekeris_root
from uacpy.core.acoustics.modal import modal_field
from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.exceptions import ConfigurationError, ValidityWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.results import Field, PhaseReference, ResultStack

__all__ = ["free_field", "lloyd_mirror", "ideal_waveguide", "pekeris"]

# Evanescent modes kept by the ideal-waveguide sum: every mode whose
# exp(-|k_rm| r) at the nearest nonzero receiver range is above this.
_EVANESCENT_FLOOR = 1e-12
# Hard cap on the ideal-waveguide mode count, reached only for receivers
# much closer to the source than a water depth.
_MAX_MODES = 20000


def _check_source(source, who):
    if getattr(source, 'source_type', 'point') != 'point':
        raise ConfigurationError(
            f"{who}: the closed form is a point source; got "
            f"source_type={source.source_type!r}.")
    if source.beam_pattern is not None:
        raise ConfigurationError(
            f"{who}: the closed form is an omnidirectional source; this "
            f"Source carries a beam_pattern.")
    if not source.has_unit_weights:
        raise ConfigurationError(
            f"{who}: the closed form returns one unit-amplitude field per "
            f"source depth; superpose weights with ResultStack.superpose.")


def _isovelocity_water(env, who):
    """Sound speed (m/s) and density (g/cm^3) of a range-independent
    isovelocity water column under a pressure-release surface, or raise."""
    if env.is_range_dependent:
        raise ConfigurationError(
            f"{who}: the closed form is range-independent; this "
            f"Environment varies with range.")
    c = np.asarray(env.ssp.sound_speed, dtype=float)
    if np.ptp(c) > 0.0:
        raise ConfigurationError(
            f"{who}: the closed form needs isovelocity water; the sound "
            f"speed spans {c.min():.6g}..{c.max():.6g} m/s.")
    surface = env.surface.nodes[0]
    if surface.acoustic_type != 'vacuum':
        raise ConfigurationError(
            f"{who}: the closed form has a pressure-release surface; got "
            f"surface acoustic_type={surface.acoustic_type!r}.")
    if env.absorption is not None:
        raise ConfigurationError(
            f"{who}: the closed form is lossless in the water; this "
            f"Environment carries volume absorption.")
    # ``is_range_dependent`` leaves altimetry out, so a rough surface is
    # refused here: the closed form's surface is the flat z = 0.
    if env.altimetry is not None and np.any(env.altimetry.heights != 0.0):
        raise ConfigurationError(
            f"{who}: the closed form has a flat surface at z = 0; this "
            f"Environment carries altimetry up to "
            f"{float(np.max(np.abs(env.altimetry.heights))):.3g} m.")
    return float(c.flat[0]), float(env.water_density)


def _assemble(pressure_of, source, receiver, *, backend, extra=None):
    """One Field per source depth — a ResultStack for several, like an
    engine's run — from ``pressure_of(z_s, f)`` -> (n_depth, n_range)."""
    freqs = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
    slabs = []
    for zs in np.atleast_1d(source.depths):
        planes = [pressure_of(float(zs), float(f)) for f in freqs]
        coords = {'depth': receiver.depths, 'range': receiver.ranges}
        if freqs.size == 1:
            data = planes[0]
        else:
            data = np.stack(planes, axis=-1)
            coords['frequency'] = freqs
        slabs.append(Field(
            data=data, coords=coords,
            model='analytic', backend=backend,
            source_depths=np.array([zs]), frequencies=freqs,
            phase_reference=PhaseReference.TRAVELLING_WAVE,
            source_level_dB=source.source_level_dB,
            metadata=dict(extra or {})))
    if len(slabs) == 1:
        return slabs[0]
    return ResultStack(slabs=slabs, coordinate=source.depths,
                       coordinate_name='source_depth')


def _point_source(k, dz, r):
    """exp(-i k R) / R on the (depth, range) grid; NaN at R = 0."""
    big_r = np.hypot(dz[:, None], r[None, :])
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.exp(-1j * k * big_r) / big_r
    p[big_r == 0.0] = np.nan
    return p


def free_field(source, receiver, *, sound_speed=DEFAULT_SOUND_SPEED):
    """Point source in an unbounded homogeneous medium (JKPS §2.3.2,
    Eq. 2.50).

    ``p = exp(-i k R) / R`` with ``R`` the distance from the source, so
    ``tl = 20 log10 R`` — spherical spreading.

    Parameters
    ----------
    source : Source
        Point source(s); one Field per depth.
    receiver : Receiver
        Depth x range grid (m).
    sound_speed : float
        Medium sound speed (m/s).
    """
    _check_source(source, 'free_field')
    z = np.asarray(receiver.depths, dtype=float)
    r = np.asarray(receiver.ranges, dtype=float)
    return _assemble(
        lambda zs, f: _point_source(2 * np.pi * f / sound_speed, z - zs, r),
        source, receiver, backend='free_field',
        extra={'sound_speed': float(sound_speed)})


def lloyd_mirror(source, receiver, *, sound_speed=DEFAULT_SOUND_SPEED):
    """Point source below a pressure-release surface in an otherwise
    unbounded medium — the image solution (JKPS §1.4.2.1, Eq. 1.19).

    ``p = exp(-i k R1)/R1 - exp(-i k R2)/R2``, ``R1`` from the source and
    ``R2`` from its image at ``-z_s``; the interference nulls of the
    Lloyd-mirror pattern.

    Parameters are as :func:`free_field`.

    Parameters
    ----------
    source : Source
        Point source(s); one Field per depth.
    receiver : Receiver
        Depth x range grid (m).
    sound_speed : float, optional
        Sound speed (m/s). Default :data:`~uacpy.core.constants.DEFAULT_SOUND_SPEED`.
    """
    _check_source(source, 'lloyd_mirror')
    z = np.asarray(receiver.depths, dtype=float)
    r = np.asarray(receiver.ranges, dtype=float)

    def pressure(zs, f):
        k = 2 * np.pi * f / sound_speed
        return _point_source(k, z - zs, r) - _point_source(k, z + zs, r)

    return _assemble(pressure, source, receiver, backend='lloyd_mirror',
                     extra={'sound_speed': float(sound_speed)})


def ideal_waveguide(env, source, receiver):
    """Isovelocity water between a pressure-release surface and a rigid or
    pressure-release flat bottom — the modal sum (JKPS §2.4.4.3,
    Eq. 2.150).

    ``env.bottom`` is ``BoundaryProperties(acoustic_type='rigid')`` or
    ``'vacuum'``, with no sediment layers above it. Modes are
    ``sin(k_zm z)`` with ``k_zm = (m - 1/2) pi / D`` (rigid) or
    ``m pi / D`` (pressure-release). The sum keeps every propagating mode
    and the evanescent ones still above ``1e-12`` at the nearest nonzero
    receiver range, so it is the full field there, not only its far-field
    part.

    At a mode's cutoff frequency, ``f = (m - 1/2) c / 2D`` (rigid) or
    ``m c / 2D`` (pressure-release), that mode has ``k_r = 0``: the lossless
    guide resonates, ``H_0(k_r r)`` is infinite and every sample is ``NaN``.
    Within a hair of it (``|k_r| r < 1`` out to the farthest receiver) the
    same logarithmic singularity dominates the sum, tens of dB above the
    level a step away. That is the physics of a lossless guide, not an
    error, so it is left in place; a ``ValidityWarning`` names the frequencies
    where it happens. Move a frequency grid off the cutoffs (a round-number
    grid lands on them) before synthesising a time series from it.

    Parameters
    ----------
    env : Environment
        Flat, isovelocity, lossless, rigid or vacuum bottom.
    source : Source
        Point source(s) in the water; one Field per depth.
    receiver : Receiver
        Depth x range grid (m); depths below the bottom are NaN.
    """
    who = 'ideal_waveguide'
    _check_source(source, who)
    c, rho = _isovelocity_water(env, who)
    kind = env.bottom.acoustic_type
    if kind not in ('rigid', 'vacuum'):
        raise ConfigurationError(
            f"{who}: needs a rigid or vacuum bottom; got acoustic_type="
            f"{kind!r}. A fluid half-space is pekeris().")
    # ``acoustic_type`` is the basement's; sediment layers above a rigid or
    # vacuum basement are a different waveguide.
    if env.bottom.is_layered:
        raise ConfigurationError(
            f"{who}: the closed form has the {kind} boundary at the "
            f"water's floor; this Environment has sediment layers above "
            f"it.")
    depth = float(env.depth)
    z = np.asarray(receiver.depths, dtype=float)
    r = np.asarray(receiver.ranges, dtype=float)
    offset = 0.5 if kind == 'rigid' else 0.0
    r_near = r[r > 0].min() if (r > 0).any() else np.inf

    def wavenumbers(f):
        """``(k_z, k_r)`` of the modes the sum keeps at ``f``."""
        k = 2 * np.pi * f / c
        n_prop = int(np.floor(k * depth / np.pi + offset))
        # Evanescent |k_r| ~ k_z beyond cutoff: keep k_z r_near <= -ln(floor).
        n_evan = (int(np.ceil(-np.log(_EVANESCENT_FLOOR) * depth
                              / (np.pi * r_near))) + 1
                  if np.isfinite(r_near) else 0)
        n = min(n_prop + n_evan, _MAX_MODES)
        kz = (np.arange(1, n + 1) - offset) * np.pi / depth
        kr = np.sqrt((k ** 2 - kz ** 2).astype(complex))
        kr = np.where(kr.imag < 0, -kr, kr)             # decaying branch
        return kz, kr

    def pressure(zs, f):
        kz, kr = wavenumbers(f)
        amp = np.sqrt(2 * rho / depth)
        psi_z = amp * np.sin(np.outer(z, kz))
        psi_s = amp * np.sin(kz * zs)
        p = modal_field(kr, psi_s, psi_z, r, source_density=rho,
                        form='hankel')
        p[z > depth, :] = np.nan
        return p

    for zs in np.atleast_1d(source.depths):
        if not 0.0 < zs < depth:
            raise ConfigurationError(
                f"{who}: source depth {zs} m is not inside the water "
                f"(0, {depth}) m.")
    r_max = float(r.max()) if r.size else 0.0
    resonant = []
    for f in np.atleast_1d(np.asarray(source.frequencies, dtype=float)):
        kr = wavenumbers(float(f))[1]
        if r_max > 0.0 and kr.size and np.min(np.abs(kr)) * r_max < 1.0:
            resonant.append(float(f))
    if resonant:
        warnings.warn(
            f"{who}: {len(resonant)} frequency(ies) sit on a mode's "
            f"cutoff ({', '.join(f'{f:.9g}' for f in resonant[:8])}"
            f"{', ...' if len(resonant) > 8 else ''} Hz), where k_r * r < 1 "
            f"out to {r_max:g} m. The lossless guide resonates there: the "
            f"field is NaN at the exact cutoff and tens of dB too loud next "
            f"to it. Cutoffs fall at (m - 1/2) c / 2D (rigid) or m c / 2D "
            f"(pressure-release); move the frequencies off them.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)
    return _assemble(pressure, source, receiver, backend='ideal_waveguide',
                     extra={'bottom': kind, 'sound_speed': c})


def _pekeris_kz(k, k1, depth, rho_w, rho_b):
    """Vertical wavenumbers of the trapped modes: roots of
    rho_w gamma sin(k_z D) + rho_b k_z cos(k_z D) = 0 on
    0 < k_z < sqrt(k^2 - k1^2), one per ((m - 1/2) pi / D, m pi / D)."""
    kz_max = np.sqrt(k ** 2 - k1 ** 2)

    def f(kz):
        gamma = np.sqrt(max(kz_max ** 2 - kz ** 2, 0.0))
        return rho_w * gamma * np.sin(kz * depth) + rho_b * kz * np.cos(kz * depth)

    roots = []
    m = 1
    while (m - 0.5) * np.pi / depth < kz_max:
        lo = (m - 0.5) * np.pi / depth
        hi = min(m * np.pi / depth, kz_max)
        roots.append(brentq(f, lo, hi, xtol=1e-14, rtol=1e-14))
        m += 1
    return np.array(roots), kz_max


def pekeris(env, source, receiver):
    """Isovelocity water over a lossless fluid half-space — the Pekeris
    waveguide's trapped-mode sum (JKPS §2.4.5).

    Water ``sin(k_z z)``, bottom ``sin(k_z D) exp(-gamma (z - D))``, the
    eigenvalues from continuity of pressure and of ``(1/rho) dp/dz`` at the
    seabed. Only the trapped modes — the residues of the wavenumber
    integral — are summed; the branch-line integral around ``k_r = k_1``
    (the continuous spectrum, JKPS Fig. 2.26) is omitted, so the field is
    exact only where that part has decayed — tens of water depths out when
    the bottom is lossless, because the near-cutoff leaky modes decay
    slowly. Against Scooter with ``c_high=1e9`` on 100 m of water over a
    lossless 1700 m/s, 1.5 g/cm³ bottom at 100 Hz, the median level
    difference over depth is 2.4 dB at 0.5 km, 0.95 dB at 1 km, 0.57 dB at
    2 km and at most 0.05 dB from 5 km out.

    Parameters
    ----------
    env : Environment
        Flat, isovelocity, lossless water over a half-space bottom with
        ``sound_speed`` above the water's and ``attenuation`` 0.
    source : Source
        Point source(s) in the water; one Field per depth.
    receiver : Receiver
        Depth x range grid (m); depths below the seabed take the bottom's
        evanescent form.
    """
    who = 'pekeris'
    _check_source(source, who)
    c, rho_w = _isovelocity_water(env, who)
    bottom = env.bottom
    if (bottom.acoustic_type != 'half-space' or bottom.is_layered
            or bottom.is_elastic):
        raise ConfigurationError(
            f"{who}: needs a fluid half-space bottom; got acoustic_type="
            f"{bottom.acoustic_type!r}"
            + (", layered" if bottom.is_layered else "")
            + (", elastic" if bottom.is_elastic else "") + ".")
    c_b = float(np.atleast_1d(bottom.halfspace_sound_speed)[0])
    rho_b = float(np.atleast_1d(bottom.halfspace_density)[0])
    alpha = float(np.atleast_1d(bottom.halfspace_attenuation)[0])
    if alpha != 0.0:
        raise ConfigurationError(
            f"{who}: the closed form is the lossless Pekeris waveguide; "
            f"the half-space attenuation is {alpha}. Set it to 0.")
    if c_b <= c:
        raise ConfigurationError(
            f"{who}: a trapped mode needs the bottom faster than the "
            f"water; got {c_b} m/s under {c} m/s.")
    depth = float(env.depth)
    z = np.asarray(receiver.depths, dtype=float)
    r = np.asarray(receiver.ranges, dtype=float)
    for zs in np.atleast_1d(source.depths):
        if not 0.0 < zs < depth:
            raise ConfigurationError(
                f"{who}: source depth {zs} m is not inside the water "
                f"(0, {depth}) m.")

    def psi(kz, gamma, zz):
        """Unnormalised mode shapes at depths ``zz`` -> (len(zz), n)."""
        zz = np.asarray(zz, dtype=float)[:, None]
        water = np.sin(kz * zz)
        below = np.sin(kz * depth) * np.exp(-gamma * np.clip(zz - depth, 0, None))
        return np.where(zz <= depth, water, below)

    def pressure(zs, f):
        k = 2 * np.pi * f / c
        k1 = 2 * np.pi * f / c_b
        kz, kz_max = _pekeris_kz(k, k1, depth, rho_w, rho_b)
        gamma = np.real(pekeris_root(kz_max ** 2 - kz ** 2))
        norm = ((depth / 2 - np.sin(2 * kz * depth) / (4 * kz)) / rho_w
                + np.sin(kz * depth) ** 2 / (2 * gamma * rho_b))
        amp = 1.0 / np.sqrt(norm)
        kr = np.sqrt(k ** 2 - kz ** 2)
        psi_z = amp * psi(kz, gamma, z)
        psi_s = amp * psi(kz, gamma, [zs])[0]
        return modal_field(kr, psi_s, psi_z, r, source_density=rho_w,
                           form='hankel')

    return _assemble(pressure, source, receiver, backend='pekeris',
                     extra={'sound_speed': c, 'bottom_sound_speed': c_b,
                            'bottom_density': rho_b})
