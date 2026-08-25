"""
Range-dependent environment segmentation for Kraken.

A model-support helper (not a :class:`PropagationModel`): segments a
range-dependent environment into range slices, each with a range-independent
Environment, for use by AT's multi-profile .env format.
"""

import warnings

import numpy as np
from typing import List, Tuple, Optional

from uacpy.core.environment import Environment, SoundSpeedProfile
from uacpy.core.exceptions import ConfigurationError
from uacpy.io.oalib_writer import DECK_RANGE_QUANTUM_M
from uacpy.models.base import USER_FRAME_SKIP

# Ceiling (m) on the gap between consecutive automatic segment edges:
# adiabatic/coupled mode summation interpolates each mode between profiles,
# so a slowly-varying stretch still gets a profile at least every 2 km.
# The caller-facing knob is ``n_segments`` (an explicit uniform decomposition
# overrides the automatic change-point edges entirely).
_MAX_SEGMENT_LENGTH_M = 2000.0

# The metre ceiling alone does not bound the error, because what the mode
# interpolation has to follow is the change in the waveguide between profiles
# measured in WAVELENGTHS, not in metres. ``EvaluateADMod.f90:47-51,75``
# interpolates k and phi linearly between successive profiles and
# ``EvaluateCMMod.f90:262-305`` projects the coupling matrix at each boundary;
# neither the Fortran nor ``field.f90:122-134`` tests the spacing, so nothing
# downstream trips. Measured on a 200->100 m wedge over 10 km, adiabatic,
# against a converged 161-profile reference, the error collapses onto the
# depth change per segment over the wavelength:
#     dD/lambda   0.08   0.17   0.25   0.33   0.50   0.67   1.00   1.33   4.00
#     max dB      0.14   0.73   0.64   2.96   2.45   9.05   5.07  18.18  15.72
# 100 Hz and 300 Hz agree at matched dD/lambda, which is what identifies it as
# the controlling parameter. A quarter wavelength holds the error near 0.6 dB,
# so segments are subdivided until the depth change across each is under it.
_SEGMENT_DEPTH_STEP_PER_WAVELENGTH = 0.25

# Sound-speed change per segment, as a fraction of the mean column speed per
# Hz: dc <= c * _SSP_CHANGE_PER_HZ / f. Calibrated, not derived — see
# :func:`_ssp_change_ceiling` for the sweep it comes from.
_SSP_CHANGE_PER_HZ = 0.5

# Ceiling on the automatic profile count. AT's own coupled deck ships 51
# profiles (``tests/wedge/wedge.env``) and every profile costs a mode solve,
# so an unbounded wavelength criterion could ask for thousands on a steep
# slope at high frequency. When this binds, the caller is told what depth step
# per wavelength was actually achieved rather than left to assume convergence.
_MAX_AUTO_SEGMENTS = 200


def _highest_frequency(freq) -> Optional[float]:
    """The frequency the criteria bind at — the top of a sweep, where the
    wavelength is shortest. ``None`` leaves the metre ceiling standing alone.
    """
    if freq is None:
        return None
    f = float(np.max(np.atleast_1d(np.asarray(freq, dtype=float))))
    return f if np.isfinite(f) and f > 0.0 else None


def _depth_step_ceiling(env: Environment, freq) -> Optional[float]:
    """Largest depth change (m) an automatic segment may span, or ``None``.

    A quarter of the shortest wavelength in the water column.
    """
    f = _highest_frequency(freq)
    if f is None:
        return None
    speeds = np.asarray(env.ssp.data, dtype=float)
    c_min = float(np.min(speeds)) if speeds.size else 0.0
    if not np.isfinite(c_min) or c_min <= 0.0:
        return None
    return _SEGMENT_DEPTH_STEP_PER_WAVELENGTH * c_min / f


def _ssp_change_ceiling(env: Environment, freq) -> Optional[float]:
    """Largest sound-speed change (m/s) an automatic segment may span, or
    ``None``.

    The depth rule is blind to a waveguide whose PROFILE moves while its depth
    does not, and adiabatic mode interpolation is just as sensitive to that:
    on a flat 200 m bottom whose column relaxes from a thermocline to
    isovelocity over 20 km, the depth rule returned 11 profiles at EVERY
    frequency, and against a converged 201-profile run the 800 Hz field was
    2.32 dB rms / 6.41 dB max out while 100 Hz was 0.08 dB — the error scaling
    with frequency at a fixed decomposition, which is the signature of a
    criterion not being applied at all.

    ``_SSP_CHANGE_PER_HZ`` is CALIBRATED, not derived. Sweeping ``K`` in
    ``dc <= c * K / f`` on that case at 800 Hz, against the converged run:
    ``K = 1.0`` gives 23 profiles and 0.98 dB rms, ``K = 0.5`` gives 44 and
    0.14 dB (max 0.24), ``K = 0.25`` gives 87 and 0.04 dB. 0.5 is taken as the
    accuracy/cost balance, comparable to what the depth rule delivers. There is
    no published constant behind it — it is a judgement about how far a profile
    may move between two the solver interpolates across — and a convergence
    check in ``n_segments`` remains the only way to be sure.
    """
    f = _highest_frequency(freq)
    if f is None:
        return None
    speeds = np.asarray(env.ssp.data, dtype=float)
    c_ref = float(np.mean(speeds)) if speeds.size else 0.0
    if not np.isfinite(c_ref) or c_ref <= 0.0:
        return None
    return c_ref * _SSP_CHANGE_PER_HZ / f


def _max_profile_change(env: Environment, r_prev: float,
                        r_curr: float) -> float:
    """Largest sound-speed difference (m/s) between the columns at two ranges.

    Compared on the union of the two columns' depth axes, each column linearly
    interpolated onto it, so a profile sampled at different depths at the two
    ranges is still comparable.
    """
    a = np.asarray(env.ssp.eval(range=r_prev).to_pairs(), dtype=float)
    b = np.asarray(env.ssp.eval(range=r_curr).to_pairs(), dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    z = np.unique(np.concatenate([a[:, 0], b[:, 0]]))
    ca = np.interp(z, a[:, 0], a[:, 1])
    cb = np.interp(z, b[:, 0], b[:, 1])
    return float(np.max(np.abs(cb - ca)))


def _thin_to_budget(ranges, env: Environment,
                    depth_step: Optional[float]) -> List[float]:
    """Hold the automatic profile count to :data:`_MAX_AUTO_SEGMENTS`.

    Keeps the first and last edge and takes an even subset in between, then
    reports the depth step per segment the thinned set actually achieves so a
    caller is never left assuming the wavelength criterion was met.
    """
    if len(ranges) <= _MAX_AUTO_SEGMENTS:
        return list(ranges)
    keep_idx = np.unique(
        np.linspace(0, len(ranges) - 1, _MAX_AUTO_SEGMENTS).astype(int))
    thinned = [float(ranges[i]) for i in keep_idx]
    if depth_step is not None:
        depths = [float(np.asarray(env.bathymetry.eval(range=r)).flat[0])
                  for r in thinned]
        worst = max((abs(b - a) for a, b in zip(depths, depths[1:])),
                    default=0.0)
        warnings.warn(
            f"segment_environment_by_range: the quarter-wavelength criterion "
            f"asks for {len(ranges)} profiles, above the "
            f"{_MAX_AUTO_SEGMENTS}-profile ceiling, so the automatic "
            f"decomposition was thinned to {len(thinned)}. The worst segment "
            f"now spans {worst:.3g} m of depth against the "
            f"{depth_step:.3g} m target, so the modes are interpolated across "
            f"more than a quarter wavelength of change and the field may not be "
            f"converged.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
    return thinned


def segment_environment_by_range(
    env: Environment,
    n_segments: Optional[int] = None,
    freq=None,
) -> List[Tuple[float, Environment]]:
    """
    Segment a range-dependent environment into range slices

    Parameters
    ----------
    env : Environment
        Range-dependent environment to segment
    n_segments : int, optional
        Number of evenly spaced segments (``>= 2``). If None, automatically
        determined from the environment's change points, with intermediate
        edges wherever the gap exceeds :data:`_MAX_SEGMENT_LENGTH_M` or spans
        more than a quarter wavelength of depth change.
    freq : float or array-like, optional
        Frequency (Hz) the segmentation is for; the highest is used, since the
        criterion binds at the shortest wavelength. ``None`` falls back to the
        metre ceiling alone.

    Returns
    -------
    segments : list of (range, Environment)
        List of ``(range in metres, environment)`` tuples for each segment.

    Raises
    ------
    ConfigurationError
        If ``n_segments`` is given and smaller than 2.

    Notes
    -----
    Segments are created at bathymetry / SSP / RD-bottom change points and
    at regular intervals if those vary slowly. The first segment always sits
    at r=0 so no receiver range falls shoreward of every profile.
    """
    if not env.is_range_dependent:
        return [(0.0, env)]

    bathy_ranges_m = env.bathymetry.ranges
    max_range_m = float(bathy_ranges_m[-1])

    if env.ssp.is_range_dependent:
        max_range_m = max(max_range_m, float(env.ssp.ranges[-1]))
    if env.bottom.is_range_dependent:
        max_range_m = max(max_range_m, float(env.bottom.ranges[-1]))

    if max_range_m <= 0:
        return [(0.0, env)]

    if n_segments is not None:
        if n_segments < 2:
            raise ConfigurationError(
                f"n_segments must be >= 2 to describe a range-dependent "
                f"environment (got {n_segments}); 1 collapses the run to a "
                f"single profile at r=0. Pass n_segments=None for automatic "
                f"segmentation at the environment's change points."
            )
        segment_ranges_m = np.linspace(0, max_range_m, n_segments)
    else:
        # Automatic segmentation: union the change-point ranges from
        # bathymetry, 2-D SSP, and RD-bottom axes; insert intermediate
        # points where the gap between consecutive change points exceeds
        # ``_MAX_SEGMENT_LENGTH_M``.
        candidates = list(bathy_ranges_m.tolist())
        if env.ssp.is_range_dependent:
            candidates.extend(env.ssp.ranges.tolist())
        if env.bottom.is_range_dependent:
            candidates.extend(env.bottom.ranges.tolist())
        # Merge at the resolution the deck can express, not at exact float
        # equality: a bathymetry axis and an SSP axis that name the same physical
        # range through different arithmetic differ in the last bits, and both
        # would survive a set(). They then print as one token, and field.exe
        # divides by the zero gap (EvaluateADMod.f90:75) with no diagnostic.
        candidates.sort()
        key_ranges_m = []
        for r in candidates:
            if not key_ranges_m or r - key_ranges_m[-1] > DECK_RANGE_QUANTUM_M:
                key_ranges_m.append(r)
        # A profile axis that starts beyond r=0 would leave everything
        # shoreward of its first sample without a segment; anchor at 0 and let
        # the constant extrapolation every carrier already does fill it.
        if key_ranges_m[0] > 0:
            key_ranges_m.insert(0, 0.0)

        depth_step = _depth_step_ceiling(env, freq)
        ssp_step = _ssp_change_ceiling(env, freq)
        segment_ranges_m = [key_ranges_m[0]]
        for i in range(1, len(key_ranges_m)):
            prev = key_ranges_m[i - 1]
            curr = key_ranges_m[i]
            seg_length = curr - prev
            n_subseg = int(np.ceil(seg_length / _MAX_SEGMENT_LENGTH_M))
            if depth_step is not None:
                # How much the seafloor moves across THIS gap — not a global
                # slope, so a track flat for 20 km then dropping sharply gets
                # its profiles where it drops.
                d_prev = float(np.asarray(
                    env.bathymetry.eval(range=prev)).flat[0])
                d_curr = float(np.asarray(
                    env.bathymetry.eval(range=curr)).flat[0])
                delta_depth = abs(d_curr - d_prev)
                if delta_depth > depth_step:
                    n_subseg = max(n_subseg,
                                   int(np.ceil(delta_depth / depth_step)))
            if ssp_step is not None and env.ssp.is_range_dependent:
                # ...and how much the PROFILE moves, which a depth test cannot
                # see. Only for a range-dependent SSP: a single profile is
                # identical at both ends of every gap by construction.
                delta_c = _max_profile_change(env, prev, curr)
                if delta_c > ssp_step:
                    n_subseg = max(n_subseg,
                                   int(np.ceil(delta_c / ssp_step)))
            if n_subseg > 1:
                subseg_ranges = np.linspace(prev, curr, n_subseg + 1)[1:-1]
                segment_ranges_m.extend(subseg_ranges)
            segment_ranges_m.append(curr)
        segment_ranges_m = sorted(set(segment_ranges_m))
        segment_ranges_m = _thin_to_budget(segment_ranges_m, env,
                                           depth_step)

    # Annotated rather than inferred: the range labels come off a float64
    # array, so an inferred list[tuple[float64, Environment]] does not satisfy
    # the declared list[tuple[float, Environment]] — list is invariant, even
    # though float64 is a float.
    segments: List[Tuple[float, Environment]] = []
    for r in segment_ranges_m:
        depth_at_range = float(np.asarray(env.bathymetry.eval(range=r)).flat[0])

        bottom_segment = env.bottom.at(range=r)
        ssp_at_range = env.ssp.eval(range=r).to_pairs()

        # AT ends a medium only at the SSP sample matching the mesh-line depth
        # (Acoustics-Toolbox/misc/sspMod.f90:352-362). Quantise the segment
        # depth to the nearest 0.1 m — a deliberate stability quantum, not a
        # writer requirement (``deck_depth`` round-trips 6 decimals): it keeps
        # segment decks identical under mm-scale bathymetry noise. Truncate
        # the profile above it and append an interpolated sample exactly
        # there; that appended sample is the one ``extend_to`` lands on.
        depth_rounded = float(f"{depth_at_range:.1f}")
        ssp_for_segment = ssp_at_range[ssp_at_range[:, 0] < depth_rounded].copy()
        c_at_depth = float(np.interp(depth_rounded, ssp_at_range[:, 0], ssp_at_range[:, 1]))
        ssp_for_segment = np.vstack([ssp_for_segment, [depth_rounded, c_at_depth]])
        # Kraken needs ≥2 samples per medium; on shoaling segments where the
        # seafloor is shallower than every SSP sample, prepend the surface.
        if len(ssp_for_segment) < 2:
            c_at_surface = float(np.interp(0.0, ssp_at_range[:, 0], ssp_at_range[:, 1]))
            ssp_for_segment = np.vstack([[0.0, c_at_surface], ssp_for_segment])

        seg_ssp = SoundSpeedProfile.from_pairs(
            ssp_for_segment, shape=env.ssp.shape,
        )
        env_segment = Environment(
            name=f"{env.name} @ {r / 1000.0:.1f}km",
            ssp=seg_ssp,
            bathymetry=depth_at_range,
            bottom=bottom_segment,
            surface=env.surface,
            absorption=env.absorption,
        )

        segments.append((r, env_segment))

    return segments
