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
    bank  = replica_bank(modes, array_depths=zs, candidate_depths=cz,
                         candidate_ranges=cr)   # Replicas on (depth, range)
    K     = csdm(measured_snapshots)          # (n_rcv, n_snap) -> (n_rcv, n_rcv)
    amb   = bartlett(K, bank)                 # ambiguity Field, dB re peak
    best  = amb.max()                         # the estimate, with its coords

Against ``acoustic_signal.beamforming``
---------------------------------------
:func:`bartlett` and :func:`mvdr` here are
:func:`uacpy.acoustic_signal.bartlett` / :func:`~uacpy.acoustic_signal.mvdr`
over a :class:`~uacpy.core.results.Replicas` bank, returned as an ambiguity
Field. They differ from the array functions' defaults in two stated ways:

* normalisation: Bartlett here is ``normalize='trace'`` (a perfect match to
  a rank-one CSDM scores 1) and MVDR ``normalize='max'``; the array
  functions default to ``'none'``;
* MVDR loading: ``diagonal_loading=0.01`` here, ``1e-06`` there.

The loading defaults differ on purpose and are **not** aligned: a replica bank
over a dense candidate grid is routinely rank-deficient against a short
snapshot record, which is why ``1e-2`` is the default here (see
:func:`mvdr`); the array-processing surface assumes a full-rank sample
covariance and only needs a numerical floor. The covariance estimate is the
same under both names: :func:`csdm` and ``sample_covariance`` (which adds
``diagonal_loading``).
"""

from __future__ import annotations

from typing import Union


import numpy as np

from uacpy.core.exceptions import ConfigurationError
from uacpy.core.acoustics import modal_field, mode_shapes_at
from uacpy.core._beamforming import snapshot_covariance

__all__ = [
    "synthesize_replica",
    "replica_bank",
    "replica_bank_from_field",
    "csdm",
    "bartlett",
    "mvdr",
]

_ArrayLike = Union[float, np.ndarray]


def synthesize_replica(
    modes,
    *,
    source_depth: float,
    ranges: _ArrayLike,
    array_depths: _ArrayLike,
) -> np.ndarray:
    """Complex pressure at ``array_depths`` for a source at ``(source_depth, ranges)``.

    Evaluates the KRAKEN far-field modal sum through
    :func:`uacpy.core.acoustics.modal_field` with unit source density, on
    shapes read by :func:`uacpy.core.acoustics.mode_shapes_at`. The result is
    proportional to the physical replica vector; the omitted global scalar
    (source level and the source-depth density ``rho(z_s)``) is common to
    every sensor and divides out of the Bartlett/MVDR processors. Uses the asymptotic (far-field) Hankel form, valid for
    ``k_m r >> 1`` — the same approximation ``field.exe`` makes; it is not a
    near-source field.

    Everything after ``modes`` is keyword-only: the depths and ranges are
    all float arrays, so a positional call in another order would swap them
    silently rather than fail.

    Parameters
    ----------
    modes : Modes
        KRAKEN eigenpairs (``k``, ``phi``, ``depths``).
    source_depth : float
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

    # The shapes at the source and the sensors come from the tabulation the
    # way every modal sum in the package reads them (refused outside it, not
    # clamped), and the sum is the package's asymptotic one. Its
    # e^{-i pi/4} sqrt(2 pi) and the unit source density are the global
    # scalar this replica leaves out anyway.
    phi_s = mode_shapes_at(modes.phi, modes.depths,
                           np.atleast_1d(np.asarray(source_depth, dtype=float)),
                           outside='raise')[0]               # (M,)
    phi_r = mode_shapes_at(modes.phi, modes.depths, z,
                           outside='raise')                  # (N, M)
    p = modal_field(modes.k, phi_s, phi_r, r, source_density=1.0)  # (N, R)

    if np.ndim(ranges) == 0:
        p = p[:, 0]
    if np.ndim(array_depths) == 0:
        p = p[0]
    return p


def _vertical_array(depths) -> np.ndarray:
    """``(N, 3)`` ``(x, y, z)`` positions of a vertical line array at the
    origin with its elements at ``depths``: the geometry a replica bank over
    ``(depth, range)`` candidates is computed for."""
    z = np.atleast_1d(np.asarray(depths, dtype=float))
    return np.column_stack([np.zeros_like(z), np.zeros_like(z), z])


def replica_bank(
    modes,
    *,
    array_depths: np.ndarray,
    candidate_depths: np.ndarray,
    candidate_ranges: np.ndarray,
):
    """Replica vectors over a candidate ``(depth, range)`` grid.

    Everything after ``modes`` is keyword-only, as on
    :func:`synthesize_replica`: three float arrays in a positional call
    would swap silently.

    Models a **vertical line array**: every sensor in ``array_depths`` shares
    the candidate range, so each replica applies one range to all elements.
    A horizontal/tilted array (elements at differing ranges) would need
    per-element ranges and is out of scope here.

    Parameters
    ----------
    modes : Modes
        The modes the replicas are synthesised from.
    array_depths : ndarray
        Depths (m) of the vertical array's sensors.
    candidate_depths, candidate_ranges : ndarray
        The candidate source grid (m).

    Returns
    -------
    Replicas
        One frequency (``modes.frequencies``), ``candidates``
        ``{'depth': candidate_depths, 'range': candidate_ranges}``, and
        ``replicas[0, i, j]`` the replica vector over ``array_depths`` for a
        source at ``(candidate_depths[i], candidate_ranges[j])``;
        ``receiver_positions`` is the vertical array at the origin.
    """
    from uacpy.core.results import Replicas
    z_arr = np.atleast_1d(np.asarray(array_depths, dtype=float))
    cz = np.atleast_1d(np.asarray(candidate_depths, dtype=float))
    cr = np.atleast_1d(np.asarray(candidate_ranges, dtype=float))
    bank = np.empty((1, cz.size, cr.size, z_arr.size), dtype=np.complex128)
    for i, zs in enumerate(cz):
        bank[0, i] = synthesize_replica(modes, source_depth=zs, ranges=cr,
                                        array_depths=z_arr).T
    return Replicas(replicas=bank, candidates={'depth': cz, 'range': cr},
                    receiver_positions=_vertical_array(z_arr),
                    frequencies=getattr(modes, 'frequencies', None),
                    model=getattr(modes, 'model', ''))


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
            f"dtype={slab.data.dtype}."
        )
    coords = list(slab.coords)
    if coords != ["depth", "range"]:
        raise ConfigurationError(
            "replica_bank_from_field: each slab needs canonical "
            f"['depth', 'range'] coords (depth = array elements); got {coords}."
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


def replica_bank_from_field(field, *, array_depths=None):
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

    The ``depth`` (receiver) axis becomes the element axis, last; every
    remaining axis is a candidate-grid axis kept in canonical order, named
    as the ambiguity surface names it (the field's ``source_depth`` is the
    candidate ``'depth'``), so the result drops straight into
    :func:`bartlett` / :func:`mvdr`. Those unit-normalise each replica, so
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
    Replicas
        One frequency (the field's), ``candidates`` the field's
        non-``depth`` axes in order (``{'depth', 'range'}`` for a
        source-depth sweep, ``{'range'}`` for one source depth), the
        element axis last, and ``receiver_positions`` the vertical array
        at the field's ``depth`` axis.
    """
    from uacpy.core.results import Replicas
    # ResultStack over source_depth → stack slab pressures into a new
    # candidate-depth axis, giving (1, n_cand_depths, n_ranges, N).
    if hasattr(field, "slabs") and hasattr(field, "coordinate_name"):
        if field.coordinate_name != "source_depth":
            raise ConfigurationError(
                "replica_bank_from_field: a ResultStack must stack over "
                "'source_depth' (the candidate-depth axis); got "
                f"{field.coordinate_name!r}."
            )
        ref = field.slabs[0]
        cols = []
        for slab in field.slabs:
            cols.append(_slab_pressure(slab, array_depths).T)
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
        return Replicas(
            replicas=np.stack(cols, axis=0)[None],
            candidates={'depth': field.coordinate,
                        'range': ref.coords['range']},
            receiver_positions=_vertical_array(ref.coords['depth']),
            frequencies=ref.frequencies, model=ref.model)

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
            f"elements, as receivers); got axes {coords}."
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
    bank = np.moveaxis(np.asarray(field.p, dtype=np.complex128),
                       sensor_pos, -1)
    candidates = {('depth' if name == 'source_depth' else name):
                  field.coords[name] for name in coords if name != 'depth'}
    return Replicas(replicas=bank[None], candidates=candidates,
                    receiver_positions=_vertical_array(field.coords['depth']),
                    frequencies=field.frequencies, model=field.model)


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


def _single_frequency_bank(covariance, replicas, who):
    """``(K, rows)``: the CSDM as a complex ``(N, N)`` matrix and the one
    frequency's replica rows of ``replicas``. Refuses anything but a
    one-frequency :class:`~uacpy.core.results.Replicas` over the CSDM's
    ``N`` elements."""
    from uacpy.core.results import Replicas
    if not isinstance(replicas, Replicas):
        raise ConfigurationError(
            f"{who}: replicas must be a Replicas set (replica_bank or "
            f"replica_bank_from_field); got {type(replicas).__name__}. For a "
            f"bare weight array use uacpy.acoustic_signal.{who}, which "
            f"returns the linear surface.")
    if replicas.n_frequencies != 1:
        raise ConfigurationError(
            f"{who}: a CSDM is one frequency, but the replicas hold "
            f"{replicas.n_frequencies}; pick one, or use Covariance.{who} "
            f"for a covariance per frequency.")
    K = np.asarray(covariance, dtype=np.complex128)
    if K.shape != (replicas.n_receivers, replicas.n_receivers):
        raise ConfigurationError(
            f"{who}: the CSDM is {K.shape}, but the replicas are over "
            f"{replicas.n_receivers} array elements; build both on the same "
            f"array.")
    return K, replicas.replicas[0]


def bartlett(covariance: np.ndarray, replicas):
    """Bartlett (linear) matched-field ambiguity surface.

    :func:`uacpy.acoustic_signal.bartlett` with ``normalize='trace'``:
    ``P_B = e^H K e / tr K`` with unit-norm replicas ``e``, so a replica
    matching a rank-one CSDM exactly scores 1; robust but broad-lobed.
    :meth:`uacpy.core.results.Covariance.bartlett` is the same processor
    over an OASN ``.xsm`` covariance, multi-frequency and unnormalised.

    Parameters
    ----------
    covariance : ndarray, shape ``(N, N)``
        CSDM (:func:`csdm`), ``K`` in the formula above.
    replicas : Replicas
        One frequency's replica bank (:func:`replica_bank`,
        :func:`replica_bank_from_field`).

    Returns
    -------
    Field
        ``kind='ambiguity'`` in dB re the surface's peak, on the replicas'
        candidate axes; ``.max()`` is the localization estimate.
        ``reference`` is the peak ``P_B`` (``reference_unit='1'``), so
        ``reference * 10**(field.data / 10)`` is ``P_B`` itself.
    """
    from uacpy.acoustic_signal.beamforming import bartlett as power
    from uacpy.core.results import ambiguity_field
    K, rows = _single_frequency_bank(covariance, replicas, 'bartlett')
    return ambiguity_field(power(K, rows, normalize='trace'),
                           replicas.candidates, reference_unit='1',
                           frequencies=replicas.frequencies,
                           model=replicas.model)


def mvdr(
    covariance: np.ndarray, replicas, *,
    diagonal_loading: float = 1e-2
):
    """Minimum-variance (Capon/MVDR) matched-field ambiguity surface.

    :func:`uacpy.acoustic_signal.mvdr` with ``normalize='max'``:
    ``P_MV = 1 / (e^H Kinv e)`` for unit-norm replicas, with diagonal
    loading ``K + diagonal_loading * tr(K)/N * I``. Small loading gives sharp
    Capon peaks but is sensitive to environmental mismatch; larger loading
    flattens the surface toward Bartlett for robustness. Loading is required
    when ``K`` is rank-deficient (e.g. a single snapshot) — hence the 1e-2
    default here, where ``K`` comes from :func:`csdm` over measured
    snapshots. :meth:`uacpy.core.results.Covariance.mvdr` is the same
    processor over OASN's full-rank ``.xsm`` covariance and defaults to
    1e-6. A candidate the processor cannot evaluate is NaN.

    Parameters
    ----------
    covariance : ndarray, shape ``(N, N)``
        CSDM (:func:`csdm`), ``K`` in the formulas above.
    replicas : Replicas
        One frequency's replica bank.
    diagonal_loading : float, optional
        Diagonal-loading fraction of the average eigenvalue. Default 1e-2.

    Returns
    -------
    Field
        As :func:`bartlett`; ``reference`` is 1, the peak of the max-scaled
        surface.
    """
    from uacpy.acoustic_signal.beamforming import mvdr as power
    from uacpy.core.results import ambiguity_field
    K, rows = _single_frequency_bank(covariance, replicas, 'mvdr')
    return ambiguity_field(
        power(K, rows, diagonal_loading=diagonal_loading, normalize='max'),
        replicas.candidates, reference_unit='1',
        frequencies=replicas.frequencies, model=replicas.model)
