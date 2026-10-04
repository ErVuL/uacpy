"""The computational domain both RAM backends share: the reference speed
c0, the domain depth and its absorber, the flat-earth transform, the water
column and its volume attenuation as a deck carries them, and the shear
speeds of the seabed."""

import numpy as np
import warnings
from typing import Optional, List
from uacpy.models.base import StageInputs
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.models._notices import give_notice
from uacpy.models.pe_grid import optimal_c0
from uacpy.core.environment import (
    Environment,
)
from uacpy.core.surface import Surface
from uacpy.core.boundary import BoundaryType
from uacpy.core.absorption import ConstantAbsorption
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, NumericsWarning,
)
from uacpy.models.ram._settings import RamGrid


# ── Per-backend facts, each stated once ──────────────────────────────────────
# Each of these is asked for rather than re-derived at the call, so a fact
# about a backend cannot disagree with itself across the module. They sit
# beside ``_COLLINS_ARRAY_LIMITS``, which is the same idea as a table.

def deck_convention(kind: str) -> str:
    """The deck dialect a backend parses.

    ``ramgeo`` reads ramsurf's, so the writer must be told ``'ramsurf'`` for
    it; the other two read their own.
    """
    return 'ramsurf' if kind == 'ramgeo' else kind


def is_seafloor_relative(kind: str) -> bool:
    """Whether the backend's sediment depths are measured DOWN FROM THE
    SEAFLOOR rather than from the sea surface.

    ``rams`` takes absolute depths; ``ramgeo`` and ``ramsurf`` take
    seafloor-relative ones, which is why their layer blocks start at 0.
    """
    return kind in ('ramgeo', 'ramsurf')


def depth_index_base(kind: str) -> int:
    """First index of the backend's output depth grid.

    ``rams0.5`` writes from index ``1 + ndz``, ``ramsurf1.5`` and ``ramgeo``
    from ``ndz`` (``third_party/ramsurf/{rams0.5,ramsurf1.5}.f``, ``outpt``).
    """
    return 1 if kind == 'rams' else 0


#: Depth points over the (shallowest) water column past which the automatic
#: grid stops refining ``dz``: runtime safety, the same bound every backend
#: gets (the Collins binaries' ``mz`` is checked separately). Sized for kHz
#: work: a 100 m sand channel to 2 km needs 23 117 points at 5 kHz (3.2 s)
#: and 62 108 at 10 kHz (12.9 s) to meet the band's score, where 10 000
#: points left the field 3.5 / 7.4 dB rms from Kraken.
MAX_DEPTH_POINTS = 100000


#: Largest level drift (dB) the automatic rams grid lets its rotated
#: Crank-Nicolson step accumulate over the marched range. The step's
#: per-metre gain or loss on a propagating component (``pe_grid.
#: rotated_cn_growth``) is second order in ``dr`` and largest where ``ξ`` sits
#: far from 0 — the horizontal components, once Lytaev's ``c0`` is centred on
#: a fast seabed's critical-angle band. It predicts the measured drift: 100 m
#: of water over a 3000/1500 m/s limestone at 200 Hz, ``c0`` = 1897 m/s,
#: -0.359 dB/km at the λ/5 cap of 1.5 m against -0.36 dB/km measured (level
#: averaged over depth and 1 km bins, against Scooter, Kraken and OAST, which
#: agree within 0.04 dB), -0.090 at 0.75 m against -0.093, and -0.169 on a
#: 2400/1000 m/s chalk against -0.174 — those are the rates on the
#: horizontal component, which carries the level. The budget is scored on the
#: largest ``|rate|`` over the band, the steeper trapped components' gain
#: included (+0.45 dB/km on the limestone at 1.5 m), so it holds for every
#: component; there it puts ``dr`` at 0.50 m over 10 km.
RAMS_CN_DRIFT_BUDGET_DB = 0.5


#: Output samples per modal-beat period (``2π/Δk``) the Collins output stride
#: ``dr·ndr`` keeps — one rule bounding both the automatic ``dr``
#: (``grid.compute_grid_lytaev``) and ``ndr``
#: (``collins.collins_output_stride``). Six holds the receiver interpolation
#: between writes under ~1 % on the flat sand channel (an order under the λ/16
#: floor's own error) and costs 0.15–0.28 dB rms on a 200→100 m sand wedge,
#: where the stride's alignment with the bathymetry staircase moves the field
#: either way; the measured ladder (3.7 % at a third of the beat) is in
#: ``docs/models/ram.md`` §6.
COLLINS_SAMPLES_PER_BEAT = 6.0


# Every RAM backend marches the written SSP profile NEAREST the current
# range (mpiramS ``ram.f90:230-231, 307-309``; the Collins decks switch at
# the midpoint markers), while the carrier, Bellhop and Kraken interpolate
# linearly between declared profiles. So uacpy writes intermediate profiles
# from ``env.ssp.eval`` until no two adjacent written profiles differ by
# more than ``_SSP_STEP_MAX_MPS`` anywhere in the column: the nearest-profile
# march then follows the linear gradient. Measured on a two-profile front
# 10 km long, marching only the declared profiles was 3.3 dB (median) and up
# to 29 dB off the same front densified to 41 columns. Each profile change
# re-factorises the depth operator, so the count is capped.
_SSP_STEP_MAX_MPS = 0.5
_MAX_SSP_PROFILES = 128
# mpiramS/src/param.f90:10 — the radius its flat-earth transform uses.
_EARTH_RADIUS_M = 6378137.0


#: Bottom wavelengths of REAL (non-absorbing) seabed the automatic grid leaves
#: between the seafloor and the start of the artificial absorbing layer.
#: Calibrated, not derived — see :func:`adequate_zmax` for the sweep:
#: 1 wavelength leaves +0.54 dB of spurious loss at 25 Hz over a lossless
#: seabed, 2 reaches +0.12 dB against a +0.07 dB converged deep grid, and 4
#: is indistinguishable from converged.
_SEABED_WAVELENGTHS_BEFORE_ABSORBER = 2.0


#: Level (dB) under which a continuous-spectrum component no longer moves the
#: field: a component 20 dB under another changes their sum by at most
#: 20·log10(1.1) = 0.83 dB. :func:`leaky_field_depth` follows each component
#: until its leakage (or the seabed's own loss along its path) has taken it
#: this far down.
_LEAKY_COMPONENT_FLOOR_DB = 20.0

#: Loss (dB) at which a component has leaked ``10^(-floor/10)`` (1 %) of its
#: energy into the seabed: ``-10·log10(1 - 10^(-floor/10))`` = 0.0436 dB. A
#: component that leaks less over its reach has a bottom field
#: :data:`_LEAKY_COMPONENT_FLOOR_DB` under itself. A near-perfect reflector (a near-massless surrogate
#: for a pressure-release floor, ``|R|`` = 0.9998) leaks under it over the
#: whole track and needs no pad below its seafloor.
_LEAKED_ENERGY_FLOOR_DB = float(
    -10.0 * np.log10(1.0 - 10.0 ** (-_LEAKY_COMPONENT_FLOOR_DB / 10.0)))


#: Sub-bottom margin, in wavelengths, below which the absorbing layer has no
#: room to work. ``ram.pdf`` p.7 asks for the grid bottom "well below the ocean
#: bottom interface" with the attenuation raised "over the lower few
#: wavelengths of the grid"; three is that "few", calibrated against the
#: measured error curve rather than assumed.
_MIN_SUBBOTTOM_WAVELENGTHS = 3.0


def seabed_sound_speed(env: Environment, c0: float) -> float:
    """Sound speed just below the seafloor, for sizing the real-seabed pad.

    The evanescent tail this pad has to contain decays in the BOTTOM, so
    the bottom's own speed sets its scale. A boundary that carries no
    speed (vacuum / rigid, or a shape this wrapper cannot read) falls back
    to the PE reference speed, which keeps the pad finite and positive.
    """
    bottom = env.bottom
    # ``Bottom`` has no plain ``sound_speed``, so reading one silently
    # returns the water value and under-sizes the pad.
    # ``all_sound_speeds()`` is exactly the set wanted: every layer speed
    # plus the half-spaces that carry geoacoustics, with 'vacuum' /
    # 'rigid' / 'file' / 'precalc' skipped because their ``sound_speed``
    # is a placeholder rather than a seabed speed. Take the LARGEST: the
    # pad is a wavelength count, and the longest bottom wavelength is the
    # conservative one to size it by.
    speeds = [v for v in bottom.all_sound_speeds()
              if np.isfinite(v) and v > 0.0]
    if not speeds:
        # A rigid / vacuum boundary leaves ``all_sound_speeds`` empty but
        # still reports a nominal ``halfspace_sound_speed`` (1600 m/s), and
        # that is what sizes the pad there — harmlessly, since such a
        # boundary carries no transmitted field for the pad to hold. Only a
        # boundary that reports no usable speed at all falls back to the
        # reference speed.
        speeds = [v for v in
                  np.atleast_1d(np.asarray(bottom.halfspace_sound_speed,
                                           dtype=float)).ravel().tolist()
                  if np.isfinite(v) and v > 0.0]
    return max(speeds) if speeds else float(c0)


def resolve_c0(env: Environment, *, knobs, speed_bounds) -> float:
    """Resolve the PE reference speed ``c₀``.

    ``c₀`` is the algorithmic expansion point of the parabolic
    equation (the speed in ``exp(ik₀x)`` factored out of the
    Helmholtz solution), not a physical input.

    Resolution order:

    1. ``self.c0`` if the user pinned it explicitly.
    2. :func:`~uacpy.models.pe_grid.optimal_c0` — Eq. (15) of
       Lytaev (2023) on the accuracy band the grid chooser scores (the
       water column out to the wider of the aperture and the seabed's
       critical angle): 1591 m/s on 1500 m/s water over sand, 2047 m/s
       over hard rock.

    All four backends honour the resolved value: mpiramS reads it
    from the ``c0_user`` line in ``in.pe``; ramgeo, rams and ramsurf
    read it from the standard ``ram.in`` ``c0`` field
    (``ramgeo1.5.f:108``, ``rams0.5.f:109``, ``ramsurf1.5.f:80``).
    """
    if knobs.c0 is not None:
        return float(knobs.c0)
    c_min_w, c_max_w = water_speed_bounds(env)
    return float(optimal_c0(c_min_w, c_max_w, resolve_angle_max(env,
                                                                knobs=knobs),
                            c_max_all=speed_bounds(env)[1]))


def water_speed_bounds(env: Environment):
    """Slowest / fastest sound speeds (m/s) of the water column alone,
    over every profile; :meth:`_speed_bounds` adds the seabed."""
    c = np.asarray(env.ssp.sound_speed, dtype=float).ravel()
    c = c[np.isfinite(c)]
    return float(c.min()), float(c.max())


def resolve_angle_max(env: Environment, *, knobs) -> float:
    """Maximum propagation angle (degrees) bracketing the Padé spectrum.

    Lytaev (2023, https://doi.org/10.3390/jmse11030496) §5.5 estimates it
    as ``θ_max = max(θ_max^src, θ_max^bottom)``, where ``θ_max^bottom`` is
    the steepest slope between bottom and water, taken from the bathymetry
    relief. ``angle_max`` on the constructor supplies the source aperture;
    the seabed term is measured here, so a slope steeper than it widens
    ``[ξ_min, ξ_max]`` instead of leaving the auto grid coarser than the
    ``accuracy`` budget implies. Capped just under 90° — the bracket is
    undefined at grazing.
    """
    theta = float(knobs.angle_max)
    bathy = env.bathymetry
    if bathy is None or bathy.n_ranges < 2:
        return theta
    r = np.asarray(bathy.ranges, dtype=float)
    z = np.asarray(bathy.depths, dtype=float)
    dr = np.diff(r)
    valid = dr > 0.0
    if not np.any(valid):
        return theta
    slope = np.degrees(np.arctan(np.abs(np.diff(z))[valid] / dr[valid]))
    return float(min(max(theta, float(np.max(slope))), 89.0))


def compute_zmax(env: Environment, freq: float,
                  c0: Optional[float] = None, *, max_range: float,
                  notices=None, knobs, speed_bounds) -> float:
    """
    Compute PE domain depth (zmax) that extends below the seafloor.

    If self.zmax is set, uses that value directly. Otherwise adds:
    - The modelled sediment stack below the max seafloor depth
    - A real-seabed pad below that (:func:`adequate_zmax`)
    - An absorbing layer (``absorber_width_wavelengths`` wavelengths of
      ``c₀``, :func:`absorbing_layer_thickness`) to prevent spurious
      reflections from the domain boundary.

    Parameters
    ----------
    env : Environment
    freq : float
        Frequency in Hz (for wavelength calculation).
    c0 : float
        PE reference speed; only the pad's fallback speed reads it.

    One rule for all four backends: mpiramS snaps the result onto its
    depth grid (:func:`mpirams.mpirams_zmax`), the Collins family reads it as
    is (:func:`collins.resolve_collins_grid`).
    """
    if knobs.zmax is not None:
        # mpiramS reaches a pinned zmax only through here; the Collins path
        # resolves it in _resolve_collins_grid and guards it there. Both
        # backends clamp the seafloor index the same way
        # (mpiramS/src/ram.f90:101 iz=min(nz,iz), ramgeo1.5.f:135), so the
        # seabed-outside-the-grid pathology is not Collins-specific: measured
        # 29 dB silent error on mpiramS with zmax below depth. No ``kind`` for
        # the guard on purpose: its rams branch RAISES, and
        # ``collins.resolve_collins_grid`` already calls it with its own kind
        # and dz — passing one here would either raise early or warn twice.
        warn_if_seafloor_outside_grid(knobs.zmax, env, freq=freq,
                                      notices=notices, knobs=knobs,
                                      speed_bounds=speed_bounds)
        return knobs.zmax
    return adequate_zmax(env, freq, c0, max_range=max_range, knobs=knobs,
                         speed_bounds=speed_bounds)


def _seabed_top_media(env: Environment) -> List[tuple]:
    """``(sound_speed, density, attenuation)`` of the medium just below the
    seafloor in every column that carries geoacoustics: the top sediment
    layer, else the half-space."""
    bottom = env.bottom
    media = []
    for i in range(bottom.n_ranges):
        column = bottom.isel(range=i)
        if column.layers:
            top = column.layers[0]
        else:
            top = column.halfspace
            if (BoundaryType.from_string(top.acoustic_type).is_parameter_free
                    or top.acoustic_type in ('file', 'precalc')):
                continue
        media.append((float(top.sound_speed), float(top.density),
                      float(top.attenuation)))
    return media


def leaky_field_depth(env: Environment, freq: float, max_range: float, *,
                      knobs) -> float:
    """Depth (m) below the seafloor that the bottom field of the continuous
    spectrum reaches within ``max_range`` — the real seabed the grid must
    hold above its absorber.

    A water-borne component at grazing angle ``θ_w`` (up to the source
    aperture ``angle_max``) that the seabed transmits leaves
    into it at ``θ_b`` (Snell, ``cos θ_b = c_b cos θ_w / c_w``) and keeps
    leaking along the whole track, so at range ``r`` its bottom field
    reaches ``r·tan θ_b``; a grid that cuts it there changes the component
    in the water too. It is followed while it matters: until its leakage —
    ``-20·log10|R(θ_w)|`` per bounce (:func:`reflection_coeff`) over a
    bounce every ``2D/tan θ_w`` — or the seabed's own attenuation along
    its path (``α_b`` dB/λ over ``z / sin θ_b``) has taken it
    :data:`_LEAKY_COMPONENT_FLOOR_DB` down, and never past ``max_range``.
    A component that leaks less than :data:`_LEAKED_ENERGY_FLOOR_DB`, 1 % of
    its energy, over its reach puts a bottom field that far under itself and
    needs no pad.

    The aperture is the source's, not :func:`resolve_angle_max`'s: a relief
    steeper than ``angle_max`` widens the Padé bracket for the energy it
    turns, but the rule follows each component at full strength over the
    whole track. A 4 m step written as a 1 m ramp reads as a 76° slope and,
    over a near-massless seabed (``|R|`` = 0.9998) to 4 km, asked for a
    16 km pad; ramgeo reads the same field (0.081 / 0.256 dB median / p90
    against a coupled-mode reference) with ``zmax`` 1779 m and 3500 m.

    Bucker's waveguide (COA 4.10.2: 240 m of water over a lossless 1505 m/s,
    ρ 2.1 half-space, 100 Hz, 2-20 km) needs about 1.1 km of it: its first
    leaky mode leaves at 3.6° and loses only 1.24 dB/km (Kraken,
    ``leaky_modes=True``). A pad of 2 seabed wavelengths there read 2.41 dB
    median |dTL| from a wavenumber integral; ``zmax`` 1000 / 1600 / 2000 m
    read 0.92 / 0.14 / 0.09 dB. A lossy seabed bounds it by its own loss.
    """
    from uacpy.core.acoustics.boundaries import reflection_coeff
    media = _seabed_top_media(env)
    if not media or max_range <= 0.0:
        return 0.0
    depth = float(env.depth)
    c_w = float(np.max(np.asarray(env.ssp.sound_speed, dtype=float)[-1]))
    theta_w = np.radians(np.linspace(0.1, float(knobs.angle_max), 400))
    deepest = 0.0
    for c_b, rho_b, alpha_b in media:
        cos_b = c_b * np.cos(theta_w) / c_w
        leaks = cos_b < 1.0
        if not np.any(leaks):
            continue
        th_w = theta_w[leaks]
        th_b = np.arccos(cos_b[leaks])
        r_mag = np.abs(reflection_coeff(
            np.degrees(th_w), sound_speed=c_b, density=rho_b,
            attenuation=alpha_b, water_sound_speed=c_w,
            water_density=float(env.water_density)))
        loss_per_bounce = -20.0 * np.log10(np.maximum(r_mag, 1e-300))
        loss_per_m = loss_per_bounce * np.tan(th_w) / (2.0 * depth)
        loss_per_m = np.maximum(loss_per_m, 1e-30)
        reach = np.minimum(max_range,
                           _LEAKY_COMPONENT_FLOOR_DB / loss_per_m)
        # A component that leaks under the floor's share of its energy over
        # the whole reach puts no bottom field a cut could change.
        z = np.where(loss_per_m * reach >= _LEAKED_ENERGY_FLOOR_DB,
                     reach * np.tan(th_b), 0.0)
        if alpha_b > 0.0:
            z = np.minimum(z, _LEAKY_COMPONENT_FLOOR_DB * (c_b / max(freq, 1.0))
                           * np.sin(th_b) / alpha_b)
        deepest = max(deepest, float(np.max(z)))
    return deepest


def adequate_zmax(env: Environment, freq: float,
                   c0: Optional[float] = None, *, max_range: float, knobs,
                   speed_bounds) -> float:
    """The grid bottom ``ram.pdf`` p.7 asks for: the seafloor, the
    modelled sediment stack, a real-seabed pad and the absorbing layer.
    The pad is the deeper of two seabed wavelengths and the depth the
    continuous spectrum's bottom field reaches within ``max_range``
    (:func:`leaky_field_depth`).

    Split out from :func:`compute_zmax` so the seafloor guard can compare
    a pinned ``zmax`` against it without recursing back through the pinned
    branch.
    """
    if c0 is None:
        c0 = resolve_c0(env, knobs=knobs, speed_bounds=speed_bounds)
    absorbing_width = absorbing_layer_thickness(env, freq, knobs=knobs,
                                                speed_bounds=speed_bounds)
    dz_for_pad = float(knobs.dz) if knobs.dz is not None else 0.0
    # Leave REAL seabed between the seafloor and the absorber, not one
    # cell. ``absorber_span`` puts the ramp over the deepest
    # ``absorber_width_wavelengths`` wavelengths, so the non-absorbing
    # sub-bottom this function leaves is exactly
    # ``(zmax - depth) - absorbing_width`` — which, for a ``zmax`` of
    # ``depth + dz + absorbing_width``, is exactly ``dz``: 1.0 m at
    # 25 Hz, 0.31 m at 300 Hz. That is the very thing the
    # ``absorber_span`` docstring warns against, an artificial gradient
    # standing in for the seabed, and it costs energy the seabed should
    # have returned to modes whose evanescent tails reach past one cell.
    # Collins states the requirement directly (``RAM.md`` p.2): "To
    # prevent artificial reflections, the bottom of the computational grid
    # (the depth zmax) is placed WELL BELOW the ocean bottom interface and
    # the attenuation is increased over the LOWER FEW WAVELENGTHS of the
    # grid." One cell is not "well below", and an absorber that starts
    # there is not confined to the lower few wavelengths of anything.
    #
    # Measured against Kraken on a 200 m guide, source 30 m, receiver
    # 150 m, 1-20 km, comparing range-smoothed (incoherent) levels so
    # interference fringes cannot be mistaken for a level error. At 25 Hz
    # that one-cell grid is biased +2.24 dB with a lossless seabed; the bias
    # falls to +0.12 dB with two bottom wavelengths of real seabed and
    # +0.07 dB on a fully converged deep grid. One wavelength is not
    # enough (+0.54 dB) and four is indistinguishable from converged.
    # A lossy seabed absorbs the tails itself and hides most of it
    # (+1.03 dB at 0.1 dB/lambda, +0.14 dB at 0.5), and by 50 Hz the
    # whole effect is under 0.04 dB at every attenuation — so the pad
    # matters at low frequency and nowhere else.
    #
    # This pad sits BELOW the modelled sediment stack, never inside it.
    # A ``zmax`` of ``depth + dz + absorbing_width`` on a layered bottom
    # would keep ``absorber_span`` (what the mpiramS deck distributes
    # its sediment control points over) tight, but it leaves the stack
    # out of ``zmax`` entirely, a small absorber error traded for a large
    # modelling one: with 100 m of water over a 60 m layer at 800 Hz that
    # grid ends at 143.33 m against a layer base at 160 m, so the layer
    # runs past the grid floor, the absorbing ramp never starts inside
    # the domain (``collins.ramp_absorbing_attenuation`` returns the block
    # unchanged once ``z_abs >= z_bottom``) and the half-space never
    # enters the run at all. Against a converged ``zmax`` of 400 m, over
    # 200 m-2 km and 9 receiver depths: 2.75 dB rms / 15.77 dB max on
    # ramgeo (that grid itself converged — 400 m against 700 m is
    # 0.0001 dB rms); 3.37 / 16.15 on ramgeo, 3.65 / 11.89 on mpiramS
    # and 3.65 / 9.13 on rams at 200 Hz over a 200 m layer — silent on
    # all three. The control-point smearing a tight span avoids is
    # handled where it belongs, by :func:`mpirams.prepare_bottom_properties`
    # sizing ``nzs`` from ``sedlayer/dz`` so the interval never exceeds
    # one depth cell.
    stack = 0.0
    if env.bottom.is_layered:
        # Range-dependent safe: the deepest stack over every column, so a
        # bathymetry-following layer set stays inside the grid at the
        # range where it reaches deepest.
        stack = float(env.bottom.total_thickness_max())
    c_bottom = seabed_sound_speed(env, c0)
    seabed_pad = max(
        _SEABED_WAVELENGTHS_BEFORE_ABSORBER * c_bottom / max(freq, 1.0),
        leaky_field_depth(env, freq, max_range, knobs=knobs))
    # A bare half-space adds no stack: the pad alone separates the
    # seafloor from the absorber, on every backend. mpiramS starts its
    # ramp at control point ``nzs-1`` (``seafloor + sedlayer``,
    # ``ram.f90:334-342``) and :func:`mpirams.prepare_bottom_properties` sets
    # ``sedlayer`` from this same domain (:func:`absorber_span`), so the
    # two decks ramp from one depth.
    return env.depth + stack + max(dz_for_pad, seabed_pad) + absorbing_width


def flat_earth_depth(z):
    """``peramx.f90:272-274``'s depth map, ``eps = z/Re``,
    ``z' = z(1 + eps/2 + eps²/3)`` with ``Re`` from ``param.f90:10``
    (scalar or array, returned in kind)."""
    z = np.asarray(z, dtype=float)
    eps = z / _EARTH_RADIUS_M
    out = z * (1.0 + eps / 2.0 + eps * eps / 3.0)
    return float(out) if out.ndim == 0 else out


def flat_earth_depth_inverse(z_transformed):
    """Depth whose flat-earth image is ``z_transformed`` (scalar or
    array, returned in kind).

    The map is monotone and near-identity (``z/Re`` is ~1e-4 for ocean
    depths), so the fixed point converges in a couple of passes.
    """
    target = np.asarray(z_transformed, dtype=float)
    z = target
    for _ in range(4):
        eps = z / _EARTH_RADIUS_M
        z = target / (1.0 + eps / 2.0 + eps * eps / 3.0)
    return float(z) if z.ndim == 0 else z


def absorber_span(env: Environment, freq: float,
                   zmax: float, *, knobs, speed_bounds) -> float:
    """Depth below the deepest seafloor at which the absorbing layer starts.

    ``profl`` interpolates the sediment arrays linearly between control
    point ``nzs-1`` at ``seafloor + sedlayer`` and control point ``nzs`` at
    ``zmax`` (``mpiramS/src/ram.f90:334-342,345-350``), and uacpy raises
    only the last point to ``absorber_attenuation``. The absorbing layer is
    therefore exactly the span ``[seafloor + sedlayer, zmax]``, so
    ``sedlayer`` is what sets its width — it is not a free choice.

    Collins sizes that layer explicitly: "the bottom of the computational
    grid (the depth zmax) is placed well below the ocean bottom interface
    and the attenuation is increased over the lower **few wavelengths** of
    the grid" (RAM manual). Running the ramp from the seabed instead
    replaces the seabed's own attenuation with an artificial gradient over
    the whole sub-bottom, which absorbs energy the seabed should have
    returned.

    Below the deepest seafloor this is the depth
    ``collins.ramp_absorbing_attenuation`` starts the Collins ramp at,
    ``max(z_sediment_base, z_bottom - absorbing_width)``; elsewhere the
    two absorbers differ, since mpiramS starts its ramp at the LOCAL
    ``seafloor + sedlayer`` (``ram.f90:334-342``), which rises with the
    seabed, while the Collins ramp starts at the absolute
    ``zmax - absorbing_width`` at every range. The caller floors it with
    the modelled sediment thickness so the layer never eats into the
    seabed.
    """
    return (float(zmax) - float(env.depth)) - absorbing_layer_thickness(
        env, freq, knobs=knobs, speed_bounds=speed_bounds)


def absorbing_layer_thickness(env: Environment, freq: float, *, knobs,
                              speed_bounds) -> float:
    """Thickness (m) of the artificial absorbing layer:
    ``absorber_width_wavelengths`` wavelengths of the PE reference speed
    ``c₀``. One formula for the domain depth (:func:`adequate_zmax`),
    the mpiramS sediment span (:func:`absorber_span`) and the Collins
    attenuation ramp (:func:`collins.ramp_range_segments`).

    Collins sizes the layer as "the lower few wavelengths of the grid"
    (RAM guide) and names no medium; RAM reads attenuation in dB per
    LOCAL wavelength, so the ramp to ``absorber_attenuation`` absorbs
    ``absorber_attenuation/2 · width/λ_local`` dB one way — 100 dB on
    sand and 37 dB on 5500 m/s hard rock at the defaults, both far past
    what a grid-floor reflection needs. Counting the width in basement
    wavelengths instead was measured (lossless basements, λ/16 grid,
    9 × 19 receivers to 5 km): hard rock moved 0.000 dB and rock ≤ 0.3 dB
    between the width and twice it under EITHER count (rock 200 Hz:
    0.012), for ×2.1 depth nodes on hard rock at 100 Hz and ×2.3 at 25 Hz
    (zmax 2177 → 4940 m). What the width DOES set is the ramp's
    gradient, and that binds on a slow LOSSLESS seabed, where the
    near-cutoff modes' evanescent tails run into the ramp: at a
    converged ``dz`` on lossless sand at 200 Hz the field is 1.38 /
    1.23 / 1.04 dB rms from Kraken at 20 / 40 / 80 wavelengths (a 1/W
    convergence); with the seabed's own 0.5 dB/λ the same ladder is
    0.74 dB throughout (silt 0.3 dB/λ: 1.28 → 1.26; rock 0.2 dB/λ:
    2.27 → 2.20). Raise ``absorber_width_wavelengths`` on a lossless seabed.
    """
    return (knobs.absorber_width_wavelengths * resolve_c0(
        env, knobs=knobs, speed_bounds=speed_bounds)
            / max(float(freq), 1.0))


def warn_if_seafloor_outside_grid(zmax: float, env, *,
                                   dz: Optional[float] = None,
                                   kind: Optional[str] = None,
                                   freq: Optional[float] = None,
                                   notices=None, knobs, speed_bounds) -> None:
    """Warn when a **pinned** ``zmax`` puts the seafloor outside the PE
    domain.

    ``ram.pdf`` p.7 places the grid bottom "well below the ocean bottom
    interface" so the absorbing layer can kill downward energy before it
    reflects. Nothing enforces it: ``ramgeo1.5.f:133-135`` and
    ``ramsurf1.5.f:118-120`` clamp ``iz=min(nz,iz)`` and ``rams0.5.f:135``
    does not clamp at all, so the run simply proceeds with the bottom
    outside its own grid — measured at up to 68.8 dB against a sane
    ``zmax``, with nothing in the output to show for it.

    This warns rather than raises because ``zmax < depth`` is a legitimate
    deck: the vendored ``ramsurf/tests/deep_flat.test/ram.in`` pairs
    ``zb=20000`` with ``zmax=159.9`` deliberately, modelling a column with
    no bottom inside the domain. uacpy cannot tell the two intents apart,
    so it names the consequence and leaves the choice to the caller.

    Two callers reach here: a pinned ``zmax``, and the single ``zmax`` the
    Collins broadband loop fixes for the whole band. The latter is
    :func:`compute_zmax` at ``freq_min`` whenever the caller pinned nothing,
    which clears the seafloor by construction, so the guard is a no-op on
    that path rather than a second opinion about it.
    """
    depth = float(env.depth)

    # The binary's condition is on the seafloor INDEX, not the depth:
    # every backend sizes the grid as nz = floor(zmax/dz - 0.5)
    # (rams0.5.f:132, ramgeo1.5.f:130, ramsurf1.5.f:111, ram.f90:56), so
    # the grid holds the seafloor iff the seafloor index is <= nz.
    # Testing zmax > depth misses a band up to dz/2 wide where zmax
    # clears the seabed but the index does not — measured, zmax=200.3 m
    # over a 200 m seabed with dz=1 gives iz = nz+1.
    #
    # The index formula is NOT shared. rams0.5 alone writes iz = z/dz
    # (rams0.5.f:135); the fluid codes and mpiramS write iz = 1 + z/dz
    # and then clamp it (ramgeo1.5.f:133-135, ramsurf1.5.f:118-120,
    # ram.f90:101). Using rams' formula for all four understates the
    # index by one cell on three of them, so a seafloor exactly one cell
    # outside their grid was reported as inside.
    outside = float(zmax) <= depth
    if dz is not None and float(dz) > 0.0:
        iz = (int(float(depth) / float(dz)) if kind == 'rams'
              else int(1.0 + float(depth) / float(dz)))
        nz = int(float(zmax) / float(dz) - 0.5)
        outside = iz > nz

    if not outside:
        # A pinned zmax that clears the SEAFLOOR can still end inside the
        # sediment STACK. Nothing downstream notices: the Collins block builder
        # does not clip at zmax — ``_seabed.piecewise_breakpoints`` takes zmax
    # only
        # to give the half-space a non-zero extent, and emits every layer step
        # regardless — so ``zread`` simply interpolates the oversized block
        # onto the shorter grid, and on rams0.5 its fill loop replaces the
        # layer with a linear gradient to the half-space value. The absorbing
        # ramp is gone too, because ``collins.ramp_absorbing_attenuation``
        # returns the block unchanged once ``z_abs >= z_bottom``. Measured 2.75
        # dB rms / 15.77 dB max on ramgeo for a grid ending 17 m above a 60 m
        # layer's base, with no other sign. The automatic grid clears the stack
        # by construction (:func:`adequate_zmax`), so this can only be a pinned
        # value.
        stack = float(env.bottom.total_thickness_max())
        if stack > 0.0 and float(zmax) < depth + stack:
            give_notice(notices,
                f"RAM: zmax={float(zmax):.4g} m clears the seafloor "
                f"({depth:.4g} m) but ends inside the sediment stack, "
                f"whose base is at {depth + stack:.4g} m. The layers "
                f"below the grid floor are not modelled and the "
                f"absorbing layer has no room to start, so the domain "
                f"floor reflects — measured up to 16 dB, silently. Raise "
                f"zmax past {depth + stack:.4g} m, or leave it unset and "
                f"let uacpy size the domain around the stack.",
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
        # The seafloor is inside the grid, but ram.pdf p.7 asks for more
        # than that: "the bottom of the computational grid is placed WELL
        # BELOW the ocean bottom interface and the attenuation is
        # increased over the lower few wavelengths of the grid". A zmax
        # that merely clears the seabed leaves no absorbing layer, so the
        # bottom of the domain reflects. Measured on ramgeo, 220 m seabed:
        # zmax=222 m is 15.97 dB from an ample grid with no warning at
        # all, while zmax=220 — a threshold drawn at env.depth — is
        # 27.0 dB, which is where the error has already saturated.
        if freq is not None:
            # "the lower FEW wavelengths" (ram.pdf p.7), not uacpy's own
            # generous 20-lambda auto pad: comparing against
            # _adequate_zmax would warn at zmax=700 m here, which measures
            # 0.32 dB — noise. Three wavelengths tracks the measured error
            # curve instead: silent at 700 m (0.32 dB) and 1500 m, warning
            # at 300 m (3.1 dB) and 222 m (16.0 dB).
            lam = resolve_c0(env, knobs=knobs,
                             speed_bounds=speed_bounds) / max(float(freq), 1.0)
            adequate = depth + _MIN_SUBBOTTOM_WAVELENGTHS * lam
            if float(zmax) < adequate:
                give_notice(notices,
                    f"RAM: zmax={float(zmax):.4g} m clears the seafloor "
                    f"({depth:.4g} m) but leaves no room for the "
                    f"absorbing layer — ram.pdf p.7 places the grid "
                    f"bottom 'well below' the seabed, which here means "
                    f"about {adequate:.4g} m. The domain floor reflects "
                    f"instead of absorbing: measured 15.97 dB at 2 m of "
                    f"sub-bottom, silently.",
                    NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
                )
        return

    # rams0.5 does not clamp iz (rams0.5.f:135, against ramgeo1.5.f:135
    # and ramsurf1.5.f:120 which both do min(nz,iz)). Past nz the
    # fluid-solid stencil at rams0.5.f:590 reads lamw(iz+2) — one slot
    # beyond profl's 1..nz+2 (:200-205) — and divides by an unwritten
    # element, so the WHOLE field is NaN while the run still exits 0.
    # The fluid backends degrade to "seafloor at the grid bottom" and
    # return usable numbers, so only rams is fatal.
    if kind == 'rams':
        raise ConfigurationError(
            f"rams: zmax={float(zmax):.4g} m puts the seafloor "
            f"({depth:.4g} m) at grid index iz > nz, and rams0.5 does not "
            f"clamp iz (rams0.5.f:135, unlike ramgeo1.5.f:135 / "
            f"ramsurf1.5.f:120). The fluid-solid stencil at "
            f"rams0.5.f:590 then reads one slot past profl's initialised "
            f"1..nz+2 range and the entire field comes back NaN, from a "
            f"run that still exits 0.",
            remediation=("Raise zmax so floor(depth/dz) <= "
                         "floor(zmax/dz - 0.5) — half a cell is enough — "
                         "or leave zmax unset and let uacpy size the "
                         "domain."),
        )
    give_notice(notices,
        f"RAM: zmax={float(zmax):.4g} m is at or above the seafloor "
        f"({depth:.4g} m), so the seabed lies outside the PE grid. The "
        f"binaries do not reject this (ramgeo1.5.f:133-135 clamps "
        f"iz=min(nz,iz); rams0.5.f:135 does not clamp), and the run "
        f"returns plausible numbers with no other sign — measured up to "
        f"27.0 dB from an equivalent run with the seafloor inside the "
        f"grid. Intentional only if you mean 'no bottom in the domain'; "
        f"otherwise put zmax well below the seafloor (ram.pdf p.7).",
        NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )


def collapse_surface_to_pressure_release(
        env: Environment) -> Environment:
    """Replace any non-vacuum sea surface by a vacuum one, and warn.

    Every RAM binary hard-codes a pressure-release top row and reads no
    surface record: ``setup`` zeroes ``u(1..nz+2)`` and ``solve`` sweeps
    ``i=2..nz+1`` reading ``u(1)`` untouched (``ramgeo1.5.f:155-158``,
    ``:319-331``); mpiramS the same (``ram.f90:108``,
    ``solvetri.f90:28-52``); ramsurf forces ``u=0`` at and above the
    moving surface (``ramsurf1.5.f:279-289``). A rigid, fluid or elastic
    surface — an ice cover — would therefore run as vacuum in silence,
    so it is made vacuum here and the caller is told.
    """
    # ``Environment.__init__`` puts whatever it is given through
    # ``Surface.coerce``, which turns ``None`` into a vacuum surface, so
    # ``env.surface`` is always a ``Surface`` with at least one node.
    nodes = env.surface.nodes
    kinds = sorted({str(p.acoustic_type) for p in nodes})
    if kinds == ['vacuum']:
        return env
    e = env.copy()
    e.surface = Surface.coerce(None)
    shear = ("; surface shear is not supported by any backend either"
             if env.surface.is_elastic else "")
    warnings.warn(
        f"RAM: the sea surface has acoustic_type "
        f"{'/'.join(repr(k) for k in kinds)}, but every RAM backend "
        f"(mpiramS / rams0.5 / ramgeo / ramsurf1.5) hard-codes a "
        f"pressure-release surface — no deck carries a surface record "
        f"and the top row is held at zero pressure (ramgeo1.5.f:155-158, "
        f"mpiramS solvetri.f90:28-52) — so it is modelled as vacuum: the "
        f"whole layer is dropped, nothing of it survives{shear}. For a "
        f"rigid or ice-covered surface use Kraken, Scooter or Bellhop.",
        FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP
    )
    return e


def water_attenuation_active(env: Environment) -> bool:
    """Whether the deck carries a water-attenuation block: an absorption
    model is set and is not the zero constant, which the block would
    only spell out as a column of zeros."""
    absorption = env.absorption
    if absorption is None:
        return False
    return not (isinstance(absorption, ConstantAbsorption)
                and absorption.value_dB_per_wavelength == 0.0)


def water_attenuation_depths(env: Environment, zmax: float,
                              dz: Optional[float] = None) -> np.ndarray:
    """Depths at which the water block samples ``alpha(z)``: the SSP's
    breakpoints, 32 uniform intervals over the water column, the depths
    where the law itself bends (every biological layer edge, every row of
    a Francois-Garrison profile or a table), and the domain floor.

    ``zread`` pins a pair to the node ``1.5 + z/dz`` and pushes a second
    pair on the same node one node down (``ramgeo1.5.f:222-223``), so
    with ``dz`` given the list is thinned to one point per depth cell;
    the last kept value is held to the floor by ``zread`` itself.
    """
    zmax = float(zmax)
    pts = {0.0, zmax}
    pts.update(float(d) for d in np.atleast_1d(env.ssp.depths)
               if 0.0 <= float(d) <= zmax)
    pts.update(np.linspace(0.0, min(float(env.depth), zmax), 33).tolist())
    if env.absorption is not None:
        pts.update(float(v) for v in env.absorption._breakpoint_depths()
                   if 0.0 <= float(v) <= zmax)
    depths = np.array(sorted(pts), dtype=float)
    if dz is not None and float(dz) > 0.0:
        kept = [depths[0]]
        for z in depths[1:]:
            if z - kept[-1] >= float(dz):
                kept.append(z)
        depths = np.array(kept, dtype=float)
    return depths


def water_attenuation_block(env: Environment, freq: float,
                             ssp_pairs, zmax: float,
                             dz: Optional[float] = None,
                             band: Optional[dict] = None) -> list:
    """The water-attenuation block at one frequency: ``(depth, alpha)``
    pairs in dB per wavelength at absolute geometric depth.

    ``alpha[dB/λ] = alpha[dB/m](f, z) · c(z) / f`` — dB per *local*
    wavelength, the unit the binaries apply through ``k(1 + iηβ)``
    (Collins 1989; ``ramgeo1.5.f:199`` for the seabed, ``wattn`` for the
    water). ``c(z)`` is the section's own profile, so a range-dependent
    SSP gets its local wavelength. The Fortran never learns which law
    produced the profile — Thorp, Francois-Garrison, a biological layer
    stack or a constant all arrive as the same block.
    """
    depths = water_attenuation_depths(env, zmax, dz)
    pairs = np.asarray([(float(d), float(c)) for d, c in ssp_pairs])
    c = np.interp(depths, pairs[:, 0], pairs[:, 1])
    if band is not None:
        # The sweep's table (``water_alpha_band``) already holds alpha at
        # this bin on a superset of these depths: the thinned list keeps
        # a subset of the same sorted points.
        j = int(np.flatnonzero(band['frequencies'] == float(freq))[0])
        i = np.searchsorted(band['depths'], depths)
        attw = band['dB_per_m'][i, j] * c / float(freq)
    else:
        attw = water_alpha_dB_per_wavelength(env, float(freq),
                                             depths, c)
    return [(float(z), float(a)) for z, a in zip(depths, attw)]


def water_alpha_band(env: Environment, frequencies,
                      zmax: float) -> Optional[dict]:
    """alpha in dB/m over a whole broadband sweep, in ONE
    ``absorption.alpha`` call, on every depth
    :func:`water_attenuation_depths` can sample below ``zmax``.

    One call gives one out-of-range notice for the band (the absorption
    laws announce per call), where a per-bin evaluation printed one per
    frequency. ``None`` when no water block is written or the law is
    taken node by node at each node's sound speed (a
    :class:`ConstantAbsorption`, written as its value; a table).
    """
    if (not water_attenuation_active(env)
            or env.absorption._needs_node_sound_speed):
        return None
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    depths = water_attenuation_depths(env, zmax)
    table = env.absorption.table(f, depths, units='dB/m').data
    return {'frequencies': f, 'depths': depths,
            'dB_per_m': np.asarray(table, dtype=float).reshape(
                depths.size, f.size)}


def water_alpha_dB_per_wavelength(env: Environment, freq: float,
                                   depths: np.ndarray,
                                   c: np.ndarray) -> np.ndarray:
    """Water attenuation in dB per local wavelength at ``depths``
    (:meth:`~uacpy.core.absorption.Absorption.alpha_dB_per_wavelength`).

    A :class:`ConstantAbsorption` already is dB per local wavelength —
    the unit the AT writers put in the deck unchanged — so its value is
    written as given; routing it through ``alpha_dB_per_m`` (which
    converts at ``DEFAULT_SOUND_SPEED``) and back at ``c(z)`` would
    scale it by ``c(z)/DEFAULT_SOUND_SPEED``. Every other law is
    ``alpha[dB/m](f, z) · c(z) / f``.
    """
    return env.absorption.alpha_dB_per_wavelength(freq, depths, c)


def deck_depth(z, *, knobs):
    """A depth as the Collins deck carries it: flat-earth mapped when
    ``earth_curvature`` is set (:func:`flat_earth_depth`), else unchanged."""
    return flat_earth_depth(z) if knobs.earth_curvature else z


def deck_depth_inverse(z, *, knobs):
    """Geometric depth of a Collins-deck depth — the inverse of
    :func:`deck_depth`, applied to the binaries' output depth axis the
    way ``peramx.f90:444-449`` un-transforms mpiramS's."""
    return flat_earth_depth_inverse(z) if knobs.earth_curvature else z


def deck_block(pairs, seafloor: float, seafloor_relative: bool, *, knobs):
    """A ``(depth, value)`` block as the Collins deck carries it: every
    abscissa through :func:`deck_depth`. A seafloor-relative block
    (ramgeo / ramsurf) is mapped through its absolute depth and
    re-referenced to the mapped seafloor, so its points land under the
    seafloor the binary reads from the mapped bathymetry row and its last
    point stays on the mapped domain floor.
    """
    pairs = [(float(d), float(v)) for d, v in pairs]
    if not knobs.earth_curvature:
        return pairs
    if seafloor_relative:
        base = flat_earth_depth(seafloor)
        return [(flat_earth_depth(seafloor + d) - base, v)
                for d, v in pairs]
    return [(flat_earth_depth(d), v) for d, v in pairs]


def deck_water_column(pairs, *, knobs) -> list:
    """Water-SSP pairs as the Collins deck carries them.

    Under ``earth_curvature`` the pairs get the transform mpiramS applies to
    its own water column (``peramx.f90:268-281``): ``eps = z/Re``,
    ``z' = z(1 + eps/2 + eps²/3)``, ``c' = c(1 + eps + eps²)`` — so
    ``earth_curvature=True`` describes one physics on every backend.
    The bottom blocks go through :func:`deck_block`; the bathymetry,
    altimetry, source depth and ``zmax`` through :func:`deck_depth`
    where the deck is written — one frame for the whole deck, as
    mpiramS's ``zmax = maxval(zw)`` after its transform.
    """
    pairs = [(float(d), float(c)) for d, c in pairs]
    if not knobs.earth_curvature:
        return pairs
    out = []
    for d, c in pairs:
        eps = d / _EARTH_RADIUS_M
        out.append((flat_earth_depth(d),
                    c * (1.0 + eps + eps * eps)))
    return out


def ssp_range_axis(env: Environment) -> np.ndarray:
    """Ranges (m) at which RAM is handed a water profile: the SSP's own
    breaks plus intermediate ranges, so that no two adjacent written
    profiles differ by more than ``_SSP_STEP_MAX_MPS`` at any depth
    (see its note). A range-independent SSP returns ``[0]``."""
    ranges = (np.unique(np.asarray(env.ssp.ranges, dtype=float))
              if env.ssp.is_range_dependent else np.zeros(1))
    if ranges.size < 2:
        return ranges
    cols = [env.ssp.eval(range=float(r)).to_pairs() for r in ranges]
    z = np.unique(np.concatenate([p[:, 0] for p in cols]))
    c = np.column_stack([np.interp(z, p[:, 0], p[:, 1]) for p in cols])
    with np.errstate(invalid='ignore'):
        step = np.nanmax(np.abs(np.diff(c, axis=1)), axis=0)
    step = np.nan_to_num(step, nan=0.0)
    n_sub = np.maximum(1, np.ceil(step / _SSP_STEP_MAX_MPS)).astype(int)
    budget = _MAX_SSP_PROFILES - 1
    if n_sub.sum() > budget:
        n_sub = np.maximum(1, np.floor(n_sub * budget / n_sub.sum())
                           ).astype(int)
    out = [float(ranges[0])]
    for i, n in enumerate(n_sub):
        out.extend(np.linspace(ranges[i], ranges[i + 1], n + 1)[1:])
    return np.asarray(out, dtype=float)


def ssp_column(env: Environment, rng: float, depths) -> np.ndarray:
    """The water sound speed uacpy writes at ``depths`` for range ``rng``.

    One evaluator for both sides of the deck: ``mpirams.prepare_ssp`` writes
    ``ssp.dat`` from it and :func:`mpirams.sediment_profiles` subtracts it.
    That shared definition is what makes ``csg = cwg + cs``
    (``ram.f90:345-346``) reproduce the requested absolute bottom
    speed — the offset is only correct against the very column
    mpiramS will read back.

    Depths outside the tabulation hold the end values, matching how the
    profile is written.
    """
    pairs = (env.ssp.eval(range=rng).to_pairs()
             if env.ssp.is_range_dependent else env.ssp.to_pairs())
    return np.interp(np.asarray(depths, dtype=float),
                     pairs[:, 0], pairs[:, 1])


def cut_water_ssp_at_zmax(pairs, zmax: float) -> list:
    """A water-SSP block cut at the grid floor: samples below ``zmax``
    are dropped and, when the deepest kept sample lies above ``zmax``,
    the profile's value at ``zmax`` is appended. A profile already inside
    the grid comes back unchanged.

    ``zread`` (``ramgeo1.5.f:209-240``, the same routine in ramsurf1.5.f
    / rams0.5.f) writes every sample at node ``i = 1.5 + z/dz`` with no
    bound test: a node past ``mz`` overruns ``prof`` (SIGSEGV), and a
    node between ``nz+2`` and ``mz`` is what label 3 copies into
    ``prof(nz+2)``, so the fill loop ramps the whole column from the
    last in-grid sample to that DEEP value. Ending the block at ``zmax``
    with the interpolated value keeps the true gradient down to
    ``nz+2`` — the bottom blocks are cut the same way by
    ``_seabed.piecewise_breakpoints(zmax=...)``.
    """
    pairs = [(float(d), float(c)) for d, c in pairs]
    zmax = float(zmax)
    kept = [p for p in pairs if p[0] <= zmax]
    if len(kept) == len(pairs) or (kept and kept[-1][0] >= zmax):
        return kept
    c_at_zmax = float(np.interp(zmax, [d for d, _ in pairs],
                                [c for _, c in pairs]))
    return kept + [(zmax, c_at_zmax)]


def check_source_row_is_solved(zs: float, dz: float) -> None:
    """Reject a source shallower than one ``dz``.

    Every RAM binary plants the source with the same two statements —
    ``si=1.0+zs/dz`` / ``is=ifix(si)``, then splits the amplitude across
    ``u(is)`` and ``u(is+1)`` (``ramgeo1.5.f:389-393``,
    ``ramsurf1.5.f:396-400``, ``rams0.5.f:357-361``,
    ``mpiramS/src/ram.f90:110-114``). And every solver starts its sweep at
    row 2 (``ramgeo1.5.f:319``, ``rams0.5.f:838``,
    ``mpiramS/src/solvetri.f90:46``), so **row 1 is never written again**
    while being read into every step (``ramgeo1.5.f:320``).

    With ``zs < dz`` the index is 1, so a fraction of the source is frozen
    in ``u(1)`` for the whole march and acts as a permanent Dirichlet
    source sitting on the pressure-release surface. The field comes out
    far too loud — the opposite of the physics, which requires it to get
    *quieter* as the source approaches the surface. Measured against
    Kraken as an independent arbiter: ~46 dB mean, 72 dB peak, on
    uacpy's own default grid, with no warning on any backend.

    This is the flat-surface case of the same row-1 kill
    :func:`collins.check_source_below_depressed_surface` catches for a ramsurf
    keel; it needs no altimetry and applies to all four backends.
    """
    if float(dz) <= 0.0 or float(zs) >= float(dz):
        return
    raise ConfigurationError(
        f"RAM: source at {float(zs):.4g} m is shallower than one depth "
        f"cell (dz={float(dz):.4g} m), so selfs plants it at index 1 "
        f"(si=1.0+zs/dz, ifix). No solver writes row 1 — every sweep "
        f"starts at 2 — so the amplitude is frozen there for the whole "
        f"march and acts as a permanent source on the pressure-release "
        f"surface. Measured ~46 dB too loud against Kraken.",
        remediation=(f"Set dz <= {float(zs):.4g} m so the source lands at "
                     f"index 2 or deeper, or move the source below "
                     f"{float(dz):.4g} m."),
    )


def band_speeds(env, *, knobs, speed_bounds):
    """``(c0, c_min, c_max, c_min_all, c_max_all)`` the grid chooser scores
    with: the accuracy band is the water column widened to contain ``c0``
    (a one-sided interval would score fine, but the widening buys a finer
    dr whose near field, r ≲ 15·dr, is closer to Kraken — sand at 200 Hz
    on mpiramS, 9 × 19 receivers to 5 km, reads 1.20 dB rms at the widened
    25 m step against 1.27 at the water band's 57 m, the far field
    unchanged); the whole medium's hull is the stability band, held
    non-amplifying only (``pe_grid``)."""
    c0 = resolve_c0(env, knobs=knobs, speed_bounds=speed_bounds)
    c_min, c_max = water_speed_bounds(env)
    c_min_all, c_max_all = speed_bounds(env)
    return c0, min(c_min, c0), max(c_max, c0), c_min_all, c_max_all


def min_shear_speed(env: Environment) -> float:
    """Return the slowest non-zero shear speed in the env, or 0 if none.

    Used by the rams elastic path to floor ``dz`` so the rotated Padé
    operator stays stable.
    """
    # ``Bottom.halfspace_shear_speed`` is the SoA view, one entry per
    # column, so the layers are the only part that needs a walk. Every
    # value read here is a concrete float, so no None guard is needed:
    # ``SedimentLayer`` declares ``shear_speed`` as a plain float
    # defaulting to 0.0, and although ``BoundaryProperties`` *annotates*
    # its own ``shear_speed`` ``Optional[float]``, that class's
    # ``__post_init__`` fills every acoustic field from its defaults table
    # and then requires each to be non-negative, so the attribute is 0.0
    # rather than None on any constructed instance.
    speeds = nonzero_shear_speeds(env)
    return min(speeds) if speeds else 0.0


def max_shear_speed(env: Environment) -> float:
    """Return the fastest shear speed in the env, or 0 if none."""
    speeds = nonzero_shear_speeds(env)
    return max(speeds) if speeds else 0.0


def nonzero_shear_speeds(env: Environment) -> List[float]:
    """Every strictly positive shear speed the seabed carries, layers
    and half-spaces of every column together."""
    candidates: List[float] = [layer.shear_speed
                               for col in env.bottom.columns
                               for layer in col.layers]
    candidates += list(np.atleast_1d(env.bottom.halfspace_shear_speed))
    return [float(cs) for cs in candidates if float(cs) > 0.0]


def modal_beat_wavenumber(env: Environment, freq: float, *,
                          speed_bounds) -> float:
    """Widest horizontal-wavenumber spread of the trapped modes, rad/m:
    ``2πf (1/c_min − 1/c_max)`` over the environment's slowest and
    fastest compressional speeds. The modulus of the field beats at up to
    this rate, so its shortest period is ``2π/Δk`` and an output spacing
    past ``π/Δk`` aliases it. Zero when the environment is isovelocity.
    """
    c_min, c_max = speed_bounds(env)
    return max(0.0, 2.0 * np.pi * float(freq)
               * (1.0 / c_min - 1.0 / c_max))


def launch_grid(inputs: StageInputs) -> RamGrid:
    """The grid of launch ``inputs.launch``."""
    return inputs.settings.engine.grids[inputs.launch]
