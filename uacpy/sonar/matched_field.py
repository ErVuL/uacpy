"""Matched-field processing (MFP) from a KRAKEN normal-mode set.

A *replica* is the modeled complex pressure at the array sensors for a
hypothesized source position. MFP scans candidate positions, correlating each
replica against a measured spatial covariance matrix (CSDM); the position
maximizing a processor output is the localization estimate.

This module is self-contained: it depends only on a KRAKEN
:class:`~uacpy.core.results.Modes` set and numpy — no OASES/OASN code path. The
replica is synthesized from the modes via the far-field modal sum

    p(r, z) = r^{-1/2} * sum_m  phi_m(z_s) phi_m(z) k_m^{-1/2} exp(-i k_m r)

which reproduces ``field.exe``'s complex pressure up to one global complex
scalar (verified in ``tests/test_matched_field.py``). Because the eigenpairs
``(k_m, phi_m)`` depend only on the environment, the modes are computed once and
every replica in the search grid is a cheap analytic re-sum.

Typical use::

    modes = Kraken().compute_modes(env, source)
    bank  = replica_bank(modes, array_depths, cand_depths, cand_ranges)
    K     = csdm(measured_snapshots)          # (n_rcv, n_snap) -> (n_rcv, n_rcv)
    surf  = bartlett(K, bank)                 # (n_cand_depths, n_cand_ranges)

Against ``acoustic_signal.arrays``
----------------------------------
:mod:`uacpy.acoustic_signal.arrays` runs the same two processors over a
plane-wave steering bank. They share the numerical core
(``core/_beamforming``) and differ in four stated ways, so a number carried
from one to the other needs converting:

=====================  ============================  ==========================
                       ``sonar`` (this module)       ``acoustic_signal.arrays``
=====================  ============================  ==========================
covariance             :func:`csdm`                  ``sample_covariance``
                                                     (identical estimate; this
                                                     one adds
                                                     ``diagonal_loading``)
weight bank            ``replicas`` ``(N, *grid)``   ``steering``
                       — column-major, one column    ``(n_angles, N)`` —
                       per candidate                 row-major, one row per
                                                     angle
Bartlett scaling       normalised to ``[0, 1]``:     unnormalised
                       divided by ``tr K``           ``e^H R e``
MVDR scaling           max-scaled to 1               unscaled ``1/(w^H R^-1 w)``
MVDR loading default   ``diagonal_loading=0.01``     ``diagonal_loading=1e-06``
=====================  ============================  ==========================

The loading defaults differ on purpose and are **not** aligned: a replica bank
over a dense candidate grid is routinely rank-deficient against a short
snapshot record, which is why ``1e-2`` is the default here (see
:func:`mvdr`); the array-processing surface assumes a full-rank sample
covariance and only needs a numerical floor. ``core/_beamforming``'s docstring
states that each surface keeps its own loading, normalisation and NaN policy.
"""

from __future__ import annotations

from typing import Tuple, Union

import warnings

import numpy as np

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._beamforming import (
    loaded_inverse, quadratic_form, snapshot_covariance)

__all__ = [
    "synthesize_replica",
    "replica_bank",
    "replica_bank_from_field",
    "csdm",
    "bartlett",
    "mvdr",
]

_ArrayLike = Union[float, np.ndarray]


def _interp_modes(modes, z: np.ndarray) -> np.ndarray:
    """Interpolate the mode shapes ``phi_m(z)`` to depths ``z``.

    Returns shape ``(len(z), n_modes)``. Complex mode shapes (krakenc) are
    interpolated component-wise.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    phi = modes.phi  # (n_depth, n_modes)
    zp = modes.depths
    # np.interp end-clamps outside [zp[0], zp[-1]], returning a flat
    # plateau that looks like physics; the modal sum in
    # Modes.modal_propagation_loss raises for the same condition, so this
    # path does too.
    if float(np.min(z)) < float(zp[0]) or float(np.max(z)) > float(zp[-1]):
        raise ConfigurationError(
            f"matched_field: depth(s) [{float(np.min(z)):g}, "
            f"{float(np.max(z)):g}] m fall outside the mode tabulation "
            f"[{float(zp[0]):g}, {float(zp[-1]):g}] m — the mode shapes are "
            f"unknown there (a clamped value would be a flat, wrong "
            f"replica). Tabulate the modes over the full source/receiver "
            f"span (Kraken().compute_modes does).")
    if np.iscomplexobj(phi):
        out = np.empty((z.size, phi.shape[1]), dtype=np.complex128)
        for m in range(phi.shape[1]):
            out[:, m] = np.interp(z, zp, phi[:, m].real) + 1j * np.interp(
                z, zp, phi[:, m].imag
            )
        return out
    out = np.empty((z.size, phi.shape[1]), dtype=float)
    for m in range(phi.shape[1]):
        out[:, m] = np.interp(z, zp, phi[:, m])
    return out


def synthesize_replica(
    modes,
    src_depth: float,
    ranges: _ArrayLike,
    array_depths: _ArrayLike,
) -> np.ndarray:
    """Complex pressure at ``array_depths`` for a source at ``(src_depth, ranges)``.

    Evaluates the KRAKEN far-field modal sum. The result is proportional to the
    physical replica vector; the omitted global complex scalar
    (``A_s exp(i pi/4) rho(z_s) / sqrt(2 pi)`` — source level, source-depth
    density and a constant phase, per Medwin & Clay / Computational Ocean
    Acoustics) is common to every sensor and divides out of the Bartlett/MVDR
    processors. Uses the asymptotic (far-field) Hankel form, valid for
    ``k_m r >> 1`` — the same approximation ``field.exe`` makes; it is not a
    near-source field.

    Parameters
    ----------
    modes : Modes
        KRAKEN eigenpairs (``k``, ``phi``, ``depths``).
    src_depth : float
        Hypothesized source depth (m).
    ranges : float or ndarray, shape (R,)
        Source-receiver range(s) (m). Must be > 0.
    array_depths : float or ndarray, shape (N,)
        Sensor depths (m).

    Returns
    -------
    ndarray
        Complex pressure. Shape ``(N, R)``; a scalar ``ranges`` drops the range
        axis, a scalar ``array_depths`` drops the sensor axis.
    """
    r = np.atleast_1d(np.asarray(ranges, dtype=float))
    if np.any(r <= 0):
        raise ConfigurationError(
            f"synthesize_replica: ranges must be > 0; got "
            f"{int(np.count_nonzero(r <= 0))} value(s) <= 0, first at index "
            f"{int(np.argmax(r <= 0))} ({r[r <= 0][0]:g} m)")
    z = np.atleast_1d(np.asarray(array_depths, dtype=float))

    k = np.asarray(modes.k, dtype=np.complex128)            # (M,)
    phi_s = _interp_modes(modes, src_depth)[0]              # (M,)
    phi_r = _interp_modes(modes, z)                          # (N, M)

    # (M, R): per-mode range term, far-field (asymptotic Hankel) convention.
    # Under exp(-i k r) a mode decays for Im(k) <= 0. Raw Kraken eigenvalues
    # already carry that sign while Modes.with_attenuation builds Im(k) > 0;
    # a passive medium can only attenuate, so the sign is forced and either
    # input synthesises a decaying replica.
    k = k.real - 1j * np.abs(k.imag)
    rng_term = np.exp(-1j * np.outer(k, r)) / np.sqrt(k)[:, None]
    p = (phi_r * phi_s[None, :]) @ rng_term                 # (N, R)
    p = p / np.sqrt(r)[None, :]

    if np.ndim(ranges) == 0:
        p = p[:, 0]
    if np.ndim(array_depths) == 0:
        p = p[0]
    return p


def replica_bank(
    modes,
    array_depths: np.ndarray,
    candidate_depths: np.ndarray,
    candidate_ranges: np.ndarray,
) -> np.ndarray:
    """Replica vectors over a candidate ``(depth, range)`` grid.

    Models a **vertical line array**: every sensor in ``array_depths`` shares
    the candidate range, so each replica column applies one range to all
    elements. A horizontal/tilted array (elements at differing ranges) would
    need per-element ranges and is out of scope here.

    Returns
    -------
    ndarray, shape ``(N, n_cand_depths, n_cand_ranges)``
        ``bank[:, i, j]`` is the replica vector for a source at
        ``(candidate_depths[i], candidate_ranges[j])`` on ``array_depths``.
    """
    z_arr = np.atleast_1d(np.asarray(array_depths, dtype=float))
    cz = np.atleast_1d(np.asarray(candidate_depths, dtype=float))
    cr = np.atleast_1d(np.asarray(candidate_ranges, dtype=float))
    bank = np.empty((z_arr.size, cz.size, cr.size), dtype=np.complex128)
    for i, zs in enumerate(cz):
        bank[:, i, :] = synthesize_replica(modes, zs, cr, z_arr)
    return bank


def _slab_pressure(slab, array_depths) -> np.ndarray:
    """Complex pressure of a single ``{depth, range}`` Field as ``(N, R)``.

    ``depth`` is the array (receiver) axis. Optionally verifies the depth axis
    against ``array_depths``.
    """
    # Matched-field processing correlates phase, so the requirement is on the
    # dtype axis, not the quantity: a real TL slab is the same ``kind`` and
    # would pass a kind check having already thrown its phase away.
    if getattr(slab, "kind", None) != "pressure" or not slab.is_complex:
        raise ConfigurationError(
            "replica_bank_from_field: slab must be complex narrowband pressure "
            f"(kind='pressure', complex dtype); got "
            f"kind={getattr(slab, 'kind', None)!r}, unit={slab.unit!r}, "
            f"dtype={slab.data.dtype}"
        )
    coords = list(slab.coords)
    if coords != ["depth", "range"]:
        raise ConfigurationError(
            "replica_bank_from_field: each slab needs canonical "
            f"['depth', 'range'] coords (depth = array elements); got {coords}"
        )
    if array_depths is not None:
        want = np.atleast_1d(np.asarray(array_depths, dtype=float))
        got = slab.coords["depth"]
        if got.shape != want.shape or not np.allclose(got, want):
            raise ConfigurationError(
                "replica_bank_from_field: slab 'depth' axis does not match "
                "array_depths — resample receivers to the element depths first "
                "(field.eval / field.resample_to). Got a slab 'depth' axis of "
                f"shape {got.shape} against array_depths of shape "
                f"{want.shape}."
            )
    return np.asarray(slab.p, dtype=np.complex128)


def replica_bank_from_field(field, *, array_depths=None) -> np.ndarray:
    """MFP replica bank from a coherent ``Field`` produced by *any* model.

    Model-agnostic counterpart of :func:`replica_bank` (which is KRAKEN-mode
    only). A replica is the ocean Green's function — the point-source pressure
    response — sampled at the array for a hypothesized source position (Baggeroer
    et al. 1993; Etter §11.5.7.1). Normal modes, a parabolic-equation march and a
    ray sum are just different numerical evaluations of that same Green's
    function, so any coherent forward run can supply replicas: run the model with
    the **receivers at the array element depths** and the **source swept over the
    candidate grid**, then feed the result here.

    Accepts either (duck-typed, like :func:`replica_bank`'s ``modes``):

    * a single coherent pressure :class:`~uacpy.core.results.Field` whose axes
      are a subset of ``{source_depth, depth, range}`` (``depth`` = array). A
      ``{depth, range}`` field gives a range-only bank at one source depth; a
      ``{source_depth, depth, range}`` field gives the full depth-range grid.
    * a :class:`~uacpy.core.results.ResultStack` of ``{depth, range}`` pressure
      fields over ``source_depth`` — what a multi-source model run returns; the
      stack axis becomes the candidate-depth axis.

    The ``depth`` (receiver) axis is moved to position 0; every remaining axis is
    a candidate-grid axis kept in canonical order, so the result drops straight
    into :func:`bartlett` / :func:`mvdr`. Those unit-normalise each replica, so
    the per-position amplitude and the global source scalar divide out — no
    normalisation is needed here, exactly as for the modal sum.

    Reciprocity (Pierce §4.9.3, *interchange of source and listener*:
    ``p(z_a; z_s, r) = p(z_s; z_a, r)``) gives a cheaper construction when the
    array has fewer elements than the candidate-depth grid: place the source at
    each element depth and the receivers over the whole candidate grid (one run
    per element instead of per candidate depth), then assemble the same bank.

    Caveat: opening MFP to range-dependent (PE/ray) replicas invites exactly the
    regime where Capon/MVDR mismatch sensitivity is worst — small environmental
    or model error collapses the MVDR peak (COA §10.6). Raise
    ``mvdr(diagonal_loading=…)``
    toward Bartlett for robustness; Bartlett is comparatively forgiving.

    Parameters
    ----------
    field : Field or ResultStack
        Coherent narrowband pressure (see above). Slice to one frequency first
        (``field.at(frequency=…)``) — MFP is single-frequency.
    array_depths : array_like, optional
        Expected element depths (m). When given, the ``depth`` axis is verified
        against them and a mismatch raises.

    Returns
    -------
    ndarray, shape ``(N, *candidate_grid)``
        ``N`` array elements first; the candidate-grid axes are the field's
        non-``depth`` axes in order. ``np.unravel_index(surf.argmax(),
        surf.shape)`` then indexes those coord vectors for the localization
        estimate.
    """
    # ResultStack over source_depth → stack slab pressures into a new
    # candidate-depth axis (axis 1), giving (N, n_cand_depths, n_ranges).
    if hasattr(field, "slabs") and hasattr(field, "coordinate_name"):
        if field.coordinate_name != "source_depth":
            raise ConfigurationError(
                "replica_bank_from_field: a ResultStack must stack over "
                "'source_depth' (the candidate-depth axis); got "
                f"{field.coordinate_name!r}"
            )
        ref = field.slabs[0]
        cols = []
        for slab in field.slabs:
            cols.append(_slab_pressure(slab, array_depths))
            # Same (array, range) shape is necessary but not sufficient: two
            # slabs sampled on *different* depth or range vectors of equal
            # length would stack into a bank with an undefined candidate axis
            # and silently mislocalize. Require identical axes across the stack.
            if (not np.array_equal(slab.coords["depth"], ref.coords["depth"])
                    or not np.array_equal(slab.coords["range"], ref.coords["range"])):
                raise ConfigurationError(
                    "replica_bank_from_field: every slab must share the same "
                    "depth (array) and range (candidate) axes — the stack mixes "
                    "different receiver geometries or range grids. Got slab "
                    f"{len(cols) - 1} on a "
                    f"{slab.coords['depth'].size}-depth x "
                    f"{slab.coords['range'].size}-range grid against slab 0 "
                    f"on {ref.coords['depth'].size} x "
                    f"{ref.coords['range'].size}."
                )
        return np.ascontiguousarray(np.stack(cols, axis=1), dtype=np.complex128)

    # Single Field.
    if getattr(field, "kind", None) != "pressure" or not field.is_complex:
        raise ConfigurationError(
            "replica_bank_from_field: needs complex narrowband pressure "
            f"(kind='pressure', complex dtype); got "
            f"kind={getattr(field, 'kind', None)!r}, unit={field.unit!r}, "
            f"dtype={field.data.dtype}. "
            "Slice one frequency with field.at(frequency=…) and ensure the "
            "field is coherent (complex), not TL."
        )
    coords = list(field.coords)
    extra = set(coords) - {"source_depth", "depth", "range"}
    if extra:
        raise ConfigurationError(
            "replica_bank_from_field: unsupported axes "
            f"{sorted(extra)}; supported: source_depth/depth/range. Slice a "
            "'frequency' axis to one frequency first (field.at(frequency=…))."
        )
    if "depth" not in coords:
        raise ConfigurationError(
            "replica_bank_from_field: field needs a 'depth' axis (the array "
            f"elements, as receivers); got axes {coords}"
        )
    if array_depths is not None:
        want = np.atleast_1d(np.asarray(array_depths, dtype=float))
        got = field.coords["depth"]
        if got.shape != want.shape or not np.allclose(got, want):
            raise ConfigurationError(
                "replica_bank_from_field: field 'depth' axis does not match "
                "array_depths — resample receivers to the element depths first "
                "(field.eval / field.resample_to). Got a field 'depth' axis "
                f"of shape {got.shape} against array_depths of shape "
                f"{want.shape}."
            )
    sensor_pos = coords.index("depth")
    # np.array (copy) — field.p is a read-only view; without copying, a
    # depth-first contiguous field would alias the Field's buffer and hand back
    # a read-only bank that aliases the source. The bank must be owned/writeable.
    bank = np.moveaxis(np.array(field.p, dtype=np.complex128), sensor_pos, 0)
    return np.ascontiguousarray(bank)


def csdm(snapshots: np.ndarray) -> np.ndarray:
    """Cross-spectral density matrix from complex single-frequency snapshots.

    Parameters
    ----------
    snapshots : ndarray, shape ``(N, L)``
        ``N`` sensors, ``L`` snapshots.

    Returns
    -------
    ndarray, shape ``(N, N)``
        ``K = (1/L) sum_l d_l d_l^H`` (Hermitian).

    Raises
    ------
    ConfigurationError
        If ``snapshots`` is not 2-D, carries no snapshot column, or holds a
        NaN/Inf. A single non-finite sample makes every entry of ``K`` NaN
        and :func:`bartlett` then returns an all-NaN ambiguity surface with
        no diagnostic — engines NaN their ``r <= 0`` columns and Bellhop NaNs
        shadow-zone cells, so snapshots assembled from modelled fields reach
        here non-finite.

    Notes
    -----
    :func:`uacpy.acoustic_signal.sample_covariance` is the same estimate
    under the array-processing name; both call
    ``core._beamforming.snapshot_covariance``.
    """
    return snapshot_covariance(snapshots, "csdm")


def _flatten_bank(replicas: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """(N, *grid) -> unit-norm (N, G) columns plus the grid shape."""
    N = replicas.shape[0]
    grid_shape = replicas.shape[1:]
    E = replicas.reshape(N, -1)
    norms = np.linalg.norm(E, axis=0)
    # An identically-zero replica column is a candidate position the forward
    # model put no energy at (a shadow-zone cell from a ray/PE bank, or a
    # source depth on a pressure-release boundary where every mode shape
    # vanishes). Dividing it by 1 leaves it zero, which :func:`bartlett` scores
    # as a genuine zero ("nothing matches here"); the unguarded 0/0 would put a
    # NaN in the surface and raise a RuntimeWarning instead.
    norms[norms == 0] = 1.0
    return E / norms[None, :], grid_shape


def bartlett(K: np.ndarray, replicas: np.ndarray) -> np.ndarray:
    """Bartlett (linear) matched-field ambiguity surface, normalized to [0, 1].

    ``P_B = e^H K e / (e^H e * tr K)`` with unit-norm replicas. Equals 1 where a
    replica matches a rank-one CSDM exactly; robust but broad-lobed.

    :meth:`uacpy.core.results.Covariance.bartlett` is the same processor over
    an OASN ``.xsm`` covariance: multi-frequency, replica-index-last, and left
    unnormalised, so its surface is this one times ``tr K``.

    Parameters
    ----------
    K : ndarray, shape ``(N, N)``
        CSDM.
    replicas : ndarray, shape ``(N, *grid)``
        Replica bank (e.g. from :func:`replica_bank`).

    Returns
    -------
    ndarray, shape ``grid``
        Real ambiguity surface; argmax is the localization estimate.
    """
    E, grid_shape = _flatten_bank(replicas)
    K = np.asarray(K, dtype=np.complex128)
    trK = np.real(np.trace(K))
    # The shared Bartlett/MVDR core (core/_beamforming); E is column-major
    # (N, G), the kernel takes row-major weights.
    num = quadratic_form(K, E.T)
    # A CSDM with zero trace is a positive-semidefinite matrix of zeros, so the
    # numerator is zero too: divide by 1 to return an all-zero surface rather
    # than 0/0. :func:`mvdr` warns on the same condition because there the
    # matrix has to be inverted.
    surf = num / (trK if trK != 0 else 1.0)
    return surf.reshape(grid_shape)


def mvdr(
    K: np.ndarray, replicas: np.ndarray, diagonal_loading: float = 1e-2
) -> np.ndarray:
    """Minimum-variance (Capon/MVDR) ambiguity surface, normalized to max 1.

    ``P_MV = 1 / (e^H Kinv e)`` for unit-norm replicas, with diagonal loading
    ``K + diagonal_loading * tr(K)/N * I``. Small loading gives sharp Capon
    peaks but is
    sensitive to environmental mismatch; larger loading flattens the surface
    toward Bartlett for robustness. Loading is required when ``K`` is
    rank-deficient (e.g. a single snapshot) — hence the 1e-2 default here,
    where ``K`` comes from :func:`csdm` over measured snapshots.
    :meth:`uacpy.core.results.Covariance.mvdr` is the same processor over
    OASN's full-rank ``.xsm`` covariance and defaults to 1e-6; at equal
    loading the two agree up to the max-scaling applied below, and both
    return NaN for a degenerate candidate point.

    Parameters
    ----------
    K : ndarray, shape ``(N, N)``
        CSDM.
    replicas : ndarray, shape ``(N, *grid)``
        Replica bank.
    diagonal_loading : float, optional
        Diagonal-loading fraction of the average eigenvalue. Default 1e-2.

    Returns
    -------
    ndarray, shape ``grid``
        Real ambiguity surface, scaled so its maximum is 1.
    """
    E, grid_shape = _flatten_bank(replicas)
    K = np.asarray(K, dtype=np.complex128)
    # Loading is a *fraction of* tr(K)/N, so it vanishes with the trace: it
    # rescues a rank-deficient CSDM that still carries power, but a CSDM with
    # no power at all (silent snapshots) leaves K singular and every candidate
    # point undefined.
    if np.real(np.trace(K)) <= 0.0:
        warnings.warn(
            "mvdr: the CSDM carries no power, so the ambiguity surface is "
            "undefined; returning NaN.", UserWarning, stacklevel=2)
        return np.full(grid_shape, np.nan)
    denom = quadratic_form(loaded_inverse(K, diagonal_loading), E.T)
    # e^H Kinv e is strictly positive for a positive-definite K and a non-zero
    # replica. A non-positive value therefore means the loaded CSDM inverted
    # without staying positive-definite, or the replica column was zero — mark
    # that candidate undefined instead of emitting a negative or infinite
    # "peak" that the max-scaling below would then normalise the surface to.
    denom[denom <= 0] = np.nan
    surf = 1.0 / denom
    m = np.nanmax(surf)
    if m and np.isfinite(m):
        surf = surf / m
    return surf.reshape(grid_shape)
