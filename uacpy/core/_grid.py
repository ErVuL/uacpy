"""Shared 1-D axis-collapse used by the grid-library ``eval``/``at`` selectors.

One implementation of "evaluate along an axis and drop it" so ``Field``,
``SoundSpeedProfile`` and ``ReflectionCoefficient`` share identical
interpolation semantics. ``at`` is ``method='nearest'``; ``eval`` defaults to
``'linear'`` and accepts ``'cubic'`` too.

:class:`_RangeProfile` is the companion base for the two single-axis
``value(range)`` carriers (``Bathymetry``, ``Altimetry``).
"""

from __future__ import annotations

import numpy as np

from uacpy.core._repr import axis, build, extent
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.constants import NO_ENERGY_DB
from uacpy.core.deck_limits import DECK_RANGE_RESOLUTION
from uacpy.core._validate import (
    normalize_axis, real_scalar_or_none, require_non_negative,
    require_real_array, require_strictly_increasing,
)
from uacpy.core._carrier import DeepCopyMixin, RevalidateOnAssignMixin
from uacpy.core._export import CarrierExport
from uacpy.core._plotting import plotter

INTERP_METHODS = ('linear', 'nearest', 'cubic')


def finite_scalar_label(value, name, *, who, several="", coords=None):
    """``value`` as one finite ``float`` label, or a ``ConfigurationError``
    naming ``who``, the call: the ``at``/``eval`` label rule every carrier
    and result shares. A label is one real number (the package's one rule
    for a scalar, :func:`~uacpy.core._validate.real_scalar_or_none`): a
    list, ``None``, a bool (bare or a 0-d array), a complex value or text
    is not one. A NaN or infinite label is not finite: every distance to
    the axis is NaN or inf, so a nearest lookup ranks nothing and would fall
    to index 0 — a real sample, which is why it would read as a successful
    slice — and an interpolated slice carries the NaN into the result.
    ``several`` ends the remedy with where several labels go; ``coords``,
    the axis's values, puts its span in the remedy for a non-finite label.
    """
    number = real_scalar_or_none(value)
    if number is None:
        raise ConfigurationError(
            f"{who}: {name}={value!r} is not one value; it selects at one "
            f"{name} per call, a real number.",
            remediation=f"Call {who} once per value{several}.")
    if not np.isfinite(number):
        values = (np.empty(0) if coords is None
                  else np.asarray(coords, dtype=float).ravel())
        span = (f" (the axis spans [{np.min(values):g}, "
                f"{np.max(values):g}])" if values.size else "")
        raise ConfigurationError(
            f"{who}: {name}={value!r} is not a finite label; every distance "
            f"to the {name!r} axis is NaN or inf, so no sample is nearest "
            f"and no value lies between two.",
            remediation=f"Pass a finite {name}{span}.")
    return number


def _reject_label_absorbing_axis(axis, label, name, *, who):
    """Refuse a finite label so far outside ``axis`` that argmin cannot rank.

    Outside a spread axis the distances ``|axis - label|`` are strictly
    ordered, so a dead tie means the subtraction lost the axis to the label's
    exponent (every ``|z - 1e300|`` rounds to 1e300) and argmin falls to
    index 0 — a real sample, which is why the slice would read as successful.
    Raises the same ``ConfigurationError`` ``Field.at`` raises on its own
    argmin path, naming ``who``; in-span labels and genuine midpoint ties
    pass through.
    """
    span_low, span_high = float(np.min(axis)), float(np.max(axis))
    if (span_high > span_low and not span_low <= label <= span_high
            and float(np.ptp(np.abs(axis - label))) == 0.0):
        raise ConfigurationError(
            f"{who}: {name}={label:g} is so far outside the {name!r} axis "
            f"([{span_low:g}, {span_high:g}]) that every sample rounds to "
            f"the same distance from it; no sample is nearer than another "
            f"and index 0 would be returned.",
            remediation=f"Pass a {name} near the axis span [{span_low:g}, "
                        f"{span_high:g}]; past either end the edge sample is "
                        f"the nearest.")


def coarse_axes(axes, limit: float, *, inclusive: bool = False):
    """``[(name, spacing), ...]`` of the ``(name, values)`` axes whose
    largest sample spacing ``|diff|`` exceeds ``limit`` (reaches it, with
    ``inclusive``). ``|diff|``: a descending axis is the same grid stored the
    other way round."""
    out = []
    for name, values in axes:
        spacing = float(np.max(np.abs(np.diff(values))))
        if spacing > limit or (inclusive and spacing == limit):
            out.append((name, spacing))
    return out


def nearest_index_on_axis(axis_values, value, name='range', *, who,
                          several=""):
    """Index of the sample in ``axis_values`` nearest to the label ``value``.

    The nearest rule the non-blendable carriers select with (``Surface``,
    ``Bottom``): they cannot interpolate between entries, so ``at`` and its
    read-only counterparts all land on a stored index. The label goes through
    :func:`finite_scalar_label` first, and *before* the single-entry
    shortcut — a carrier with no axis still refuses a NaN/inf or array-valued
    label rather than answering 0 to a query it never examined. A finite
    label large enough to absorb the whole axis is refused by
    :func:`_reject_label_absorbing_axis`. Refusals name ``who``, the call,
    and end with ``several`` (:func:`finite_scalar_label`).
    """
    label = finite_scalar_label(value, name, who=who, several=several,
                                    coords=axis_values)
    if axis_values is None:
        return 0
    axis = np.asarray(axis_values, dtype=float)
    _reject_label_absorbing_axis(axis, label, name, who=who)
    return int(np.argmin(np.abs(axis - label)))


def interpolate_dB(values, interpolate, *, method, who):
    """``interpolate``, a linear map from samples to interpolated values,
    applied to dB values (levels or losses) through the energy wherever a
    no-energy cell enters the stencil.

    A cell no energy reached holds the marker ``±NO_ENERGY_DB``
    (:func:`~uacpy.core.acoustics.levels.no_energy_mask`): its energy is
    zero, a dB value it does not have. An interpolated value whose stencil
    holds such a cell is therefore the interpolated energy, the marker
    counting as zero, converted back to dB; one whose every weighted sample
    is the marker is the marker. Every other value is the dB interpolation
    itself. The marker's sign says which view the values are: ``+600`` a
    loss, energy ``10^(-L/10)`` (TL = -10·log10(I/I0), Jensen et al.,
    *Computational Ocean Acoustics*, sect. 1.3.4, Eq. 1.15), ``-600`` a
    level, energy ``10^(L/10)``. ``method='cubic'`` is refused once a
    marker is present: the spline couples every sample of an axis, so the
    marker would reach every interpolated value. ``who``, the call, opens
    the refusals.
    """
    from uacpy.core.acoustics.levels import no_energy_mask
    v = np.asarray(values)
    if np.iscomplexobj(v):
        raise ConfigurationError(
            f"{who}: dB=True reads real dB values; got complex values.",
            remediation="Interpolate the complex pressure with dB=False and "
                        "take its level afterwards.")
    v = v.astype(float)
    marker = no_energy_mask(v)
    if not marker.any():
        return interpolate(v)
    if method == 'cubic':
        raise ConfigurationError(
            f"{who}: method='cubic' on dB values that hold "
            f"{int(marker.sum())} no-energy cell(s) (the ±{NO_ENERGY_DB:g} "
            f"dB marker): a cubic spline couples every sample of an axis, so "
            f"a cell with no level would enter every interpolated value.",
            remediation="Use method='linear' (the energy is interpolated "
                        "next to a no-energy cell) or 'nearest'.")
    signs = np.sign(v[marker])
    if np.any(signs != signs[0]):
        raise ConfigurationError(
            f"{who}: the dB values hold both the loss marker "
            f"+{NO_ENERGY_DB:g} and the level marker -{NO_ENERGY_DB:g}; one "
            f"grid is either losses or levels.",
            remediation="Pass losses (no energy at +600) or levels (no "
                        "energy at -600), not both.")
    louder = -1.0 if signs[0] > 0 else 1.0       # a loss falls as it gets louder
    oriented = louder * np.where(marker, np.nan, v)
    finite = oriented[np.isfinite(oriented)]
    # The energies relative to the loudest sample, so 10^(L/10) cannot
    # overflow whatever the dB reference.
    top = float(finite.max()) if finite.size else 0.0
    energy = np.where(marker, 0.0, 10.0 ** ((oriented - top) / 10.0))
    with np.errstate(divide='ignore', invalid='ignore'):
        through_energy = louder * (top + 10.0 * np.log10(interpolate(energy)))
    # Zero interpolated energy (every weighted sample silent) is the marker.
    through_energy = np.where(np.isinf(through_energy),
                              -louder * NO_ENERGY_DB, through_energy)
    touched = interpolate(marker.astype(float)) > 0.0
    return np.where(touched, through_energy,
                    interpolate(np.where(marker, 0.0, v)))


def collapse_axis(arr, axis_values, value, method='linear', *, axis=0,
                  name='axis', who, several="", dB=False):
    """Collapse ``arr`` along ``axis`` at coordinate ``value``.

    Returns ``(reduced_array, coord)`` where ``reduced_array`` has ``axis``
    removed and ``coord`` is the coordinate the slice sits at (the nearest
    stored value for ``'nearest'``, else the requested value clamped into
    range). ``value`` must be a finite scalar; a NaN/inf or array-valued
    label raises ``ConfigurationError``, and so does a ``'nearest'`` label
    large enough to absorb the whole axis
    (:func:`_reject_label_absorbing_axis`). ``axis_values`` is the 1-D
    coordinate vector, ascending or descending; a non-monotonic axis raises
    ``ConfigurationError`` for the interpolating methods, ``name`` labels
    the axis in these errors and ``who``, the call, opens them (``several``
    as on :func:`finite_scalar_label`). Values outside the axis ends use
    constant extrapolation. ``dB=True`` reads ``arr`` as dB values (levels
    or losses): next to a no-energy cell the energy is interpolated
    (:func:`interpolate_dB`).
    """
    if method not in INTERP_METHODS:
        raise ConfigurationError(
            f"{who}: interpolation method must be one of {INTERP_METHODS}; "
            f"got {method!r}."
        )
    label = finite_scalar_label(value, name, who=who, several=several,
                                    coords=axis_values)
    x = np.asarray(axis_values, dtype=float)
    n = x.size
    if n <= 1:
        return np.take(arr, 0, axis=axis), float(x[0])
    if method == 'nearest':
        _reject_label_absorbing_axis(x, label, name, who=who)
        i = int(np.argmin(np.abs(x - label)))
        return np.take(arr, i, axis=axis), float(x[i])

    # searchsorted and the clamp below both read the axis as ascending, so a
    # descending one is walked in reverse together with the data it indexes.
    if x[0] > x[-1]:
        x = x[::-1]
        arr = np.flip(arr, axis=axis)
    if not np.all(np.diff(x) >= 0):
        raise ConfigurationError(
            f"{who}: the {name!r} coordinate is not monotonic, so an "
            f"interpolated slice would bracket the query by binary search "
            f"over an unsorted axis. Sort the axis (with its data) first, or "
            f"use method='nearest'."
        )

    v = float(min(max(label, float(x[0])), float(x[-1])))   # clamp
    if method == 'linear':
        j = int(np.clip(np.searchsorted(x, v), 1, n - 1))
        x0, x1 = float(x[j - 1]), float(x[j])
        w = 0.0 if x1 == x0 else (v - x0) / (x1 - x0)

        def interpolate(values):
            return (np.take(values, j - 1, axis=axis) * (1.0 - w)
                    + np.take(values, j, axis=axis) * w)
    else:   # cubic
        if n < 4:
            raise ConfigurationError(
                f"{who}: method='cubic' needs at least 4 samples on the "
                f"axis; got {n}. "
                f"Use method='linear' (or 'nearest')."
            )
        from scipy.interpolate import interp1d

        def _interp1d(real):
            f = interp1d(x, real, kind='cubic', axis=-1, bounds_error=False,
                         fill_value=(real[..., 0], real[..., -1]))
            return f(v)

        def interpolate(values):
            moved = np.moveaxis(np.asarray(values), axis, -1)
            if np.iscomplexobj(moved):
                return _interp1d(moved.real) + 1j * _interp1d(moved.imag)
            return _interp1d(moved)
    if dB:
        return interpolate_dB(arr, interpolate, method=method, who=who), v
    return interpolate(arr), v


class _RangeProfile(RevalidateOnAssignMixin, DeepCopyMixin, CarrierExport):
    """Base for the 1-D ``value(range)`` shape carriers.

    :class:`~uacpy.core.bathymetry.Bathymetry` (seafloor depth, positive down)
    and :class:`~uacpy.core.altimetry.Altimetry` (sea-surface height, positive
    up) differ only in the name of their value vector and the guard it must
    satisfy; the range-axis validation, the ``at``/``eval``/``isel`` selectors
    and the derived properties all live here.

    Subclasses are ``@carrier(eq=False, repr=False)`` carrying ``ranges``
    plus the vector named by ``_VALUE_FIELD``. They call
    :meth:`_init_range_profile` from ``__post_init__`` and implement
    :meth:`_validate_values`.
    """

    # Declaration hooks: a subclass states what it holds, so ``__repr__`` and
    # ``_plot_range_profile`` tell the carriers apart by declaration rather
    # than by type-sniffing a concrete field name. ``_VALUE_FIELD`` names the
    # value vector; ``_VALUE_LABEL`` is the noun for it, capitalised into the
    # plot's y-label; ``_AXIS_DOWN`` is True when the value grows downward
    # (depth), which inverts that axis.
    _VALUE_FIELD: str = ''
    _VALUE_LABEL: str = ''
    _VALUE_UNIT: str = 'm'
    _AXIS_DOWN: bool = False

    @property
    def _values(self) -> np.ndarray:
        """The value vector, whatever ``_VALUE_FIELD`` names it."""
        return getattr(self, self._VALUE_FIELD)

    # The export protocol: the value vector on the range axis.
    def _payload(self):
        return {self._VALUE_FIELD[:-1]: (self._values, ('range',),
                                         self._VALUE_UNIT)}

    def _coords(self):
        return {'range': (self.ranges, 'm')}

    def _validate_values(self) -> None:
        """Subclass hook: guard the value vector (sign / finiteness)."""
        raise NotImplementedError

    def _init_range_profile(self) -> None:
        cls = type(self).__name__
        self.ranges = require_real_array(
            self.ranges, f"{cls} ranges", hint="metres").reshape(-1)
        setattr(self, self._VALUE_FIELD, require_real_array(
            self._values, f"{cls} {self._VALUE_FIELD}").reshape(-1))
        if self.ranges.size != self._values.size:
            raise ConfigurationError(
                f"{cls}: ranges ({self.ranges.size}) and {self._VALUE_FIELD} "
                f"({self._values.size}) must have the same length.")
        if self.ranges.size == 0:
            raise ConfigurationError(f"{cls}: needs at least one point.")
        require_non_negative(self.ranges, f"{cls} ranges", hint="metres")
        self._validate_values()
        if self.ranges.size > 1:
            require_strictly_increasing(
                self.ranges, f"{cls}.ranges", min_step=DECK_RANGE_RESOLUTION)

    def __repr__(self) -> str:
        return build(type(self).__name__, [
            axis(self.ranges, 'ranges', 'm'),
            f"{self._VALUE_LABEL}={extent(self._values, self._VALUE_UNIT)}"])

    def to_pairs(self) -> np.ndarray:
        """``(N, 2)`` ``(range, value)`` array."""
        return np.column_stack([self.ranges, self._values])

    @property
    def n_ranges(self) -> int:
        return int(self.ranges.size)

    @property
    def range_max(self) -> float:
        return float(np.max(self.ranges))

    @property
    def is_range_dependent(self) -> bool:
        """True when the profile carries more than one range node.

        A structural test (node count), like ``SoundSpeedProfile`` /
        ``Bottom`` / ``Surface``: nodes with equal values still count.
        :attr:`varies_with_range` asks whether the values differ."""
        return self.ranges.size > 1

    @property
    def varies_with_range(self) -> bool:
        """True when the value changes with range: more than one node,
        and not all at the same value."""
        return self.ranges.size > 1 and bool(np.ptp(self._values) > 0)

    def at(self, *, range):
        """Nearest value to ``range`` (m) — never fabricates. Float for a
        scalar range, array for an array of ranges; every range must be
        finite. See :meth:`eval` (interpolate) and :meth:`isel`
        (positional)."""
        return self._query(range, 'nearest', f"{type(self).__name__}.at")

    def eval(self, *, range, method: str = 'linear'):
        """Interpolated value at ``range`` (m). ``method`` is ``'linear'``
        (default), ``'nearest'`` or ``'cubic'``; constant extrapolation past
        the ends, and every range must be finite. The interpolating
        counterpart of :meth:`at`."""
        return self._query(range, method, f"{type(self).__name__}.eval")

    def isel(self, *, range):
        """Value at integer index ``range`` — the positional counterpart of
        :meth:`at`."""
        idx = np.asarray(range)
        v = self._values
        n = v.size
        if idx.ndim == 0:
            i = int(idx)
            if not -n <= i < n:
                raise IndexError(
                    f"{type(self).__name__}.isel: range index {i} out of "
                    f"range for {n} point(s)")
        out = v[idx.astype(int)]
        return float(out) if idx.ndim == 0 else out

    def _query(self, range, method, who):
        return _query_profile(self.ranges, self._values, range, method,
                              who=who)

    #: The plotter :func:`uacpy.plot.plot_carrier` draws this profile with.
    _plotter = 'plot_range_profile'

    def plot(self, ax=None, *, title=None, figsize=(10, 4), **kwargs):
        """Plot the profile against range.

        Depth-valued profiles (:class:`~uacpy.core.bathymetry.Bathymetry`)
        point their axis downward; height-valued ones
        (:class:`~uacpy.core.altimetry.Altimetry`) keep it upward. Draws
        through :func:`uacpy.plot.plot_range_profile`."""
        return plotter('plot_carrier')(self, ax=ax, title=title,
                                       figsize=figsize, **kwargs)


def resample_depth_range(values, depths, ranges, new_depths, new_ranges, *,
                         method: str = 'linear', axis=(-2, -1),
                         dB: bool = False) -> np.ndarray:
    """``values`` on a ``(depth, range)`` grid, resampled onto the grid
    ``(new_depths, new_ranges)``.

    Each new point is interpolated from the stored samples around it
    (``scipy.interpolate.RegularGridInterpolator``); complex values are
    interpolated as their real and imaginary parts. A new point outside
    the stored grid is NaN, and so is one whose stencil touches a NaN
    sample: a sample nothing computed is no data, not a value to average
    against its neighbours. Every other axis of ``values`` is carried
    through. :meth:`uacpy.Field.resample_to` is this on a Field.

    A level or loss grid marks a cell no energy reached with the marker
    ``±600`` dB (:func:`~uacpy.acoustics.no_energy_mask`), which has no dB
    value to interpolate. ``dB=True`` says ``values`` are such dB values: a
    new point whose stencil holds the marker is interpolated in energy (the
    marker counting as zero) and converted back, so halfway between an
    80 dB loss and a no-energy cell reads 83.01 dB; a point whose stencil is
    all marker is the marker, and every other point is the dB interpolation
    unchanged. ``method='cubic'`` is refused on dB values that hold the
    marker: the spline couples every sample of an axis, so the marker would
    reach every new point. ``dB=False`` (default) interpolates the numbers
    as they are, the marker included.

    Interpolating a coherent field is only faithful while its carrier is
    resolved: samples under a quarter wavelength apart on both axes (the
    check :meth:`uacpy.Field.resample_to` makes; this function, which may
    be handed a field with its carrier taken out, makes none). Take the
    level first if only the level is wanted.

    Parameters
    ----------
    values : array_like
        Real or complex samples; ``axis`` names their depth and range axes.
    depths, ranges : array_like
        The stored axes (m), strictly increasing.
    new_depths, new_ranges : array_like
        The axes (m) to resample onto.
    method : {'linear', 'nearest', 'cubic'}, optional
        Interpolation scheme. Default ``'linear'``.
    axis : (int, int), optional
        The depth and range axes of ``values``. Default ``(-2, -1)``.
    dB : bool, optional
        ``values`` are real dB levels or losses; interpolate the energy next
        to a no-energy cell (above). Default ``False``.

    Returns
    -------
    ndarray
        ``values`` with those two axes replaced by ``len(new_depths)`` and
        ``len(new_ranges)`` samples, in place.

    Raises
    ------
    ConfigurationError
        An unknown ``method``, ``'cubic'`` on an axis of fewer than 4
        samples, two axes that are not two distinct axes of ``values``,
        stored axes that do not match ``values``, or with ``dB=True``:
        complex values, ``'cubic'`` over the no-energy marker, or both the
        loss and the level marker in one grid.

    Examples
    --------
    >>> import numpy as np
    >>> loss = [[80.0, 600.0, 90.0]]
    >>> resample_depth_range(loss, [10], [100, 200, 300], [10], [150, 250],
    ...                      dB=True).round(2)
    array([[83.01, 93.01]])
    """
    return _resample_depth_range(values, depths, ranges, new_depths,
                                 new_ranges, method=method, axis=axis, dB=dB)


def _resample_depth_range(values, depths, ranges, new_depths, new_ranges, *,
                          method='linear', axis=(-2, -1), dB=False,
                          who='resample_depth_range'):
    """:func:`resample_depth_range` reporting its refusals as ``who``, the
    call the user made (:meth:`uacpy.Field.resample_to` passes its own)."""
    from scipy.interpolate import RegularGridInterpolator
    if method not in INTERP_METHODS:
        raise ConfigurationError(
            f"{who}: method={method!r}; choose from {list(INTERP_METHODS)}.")
    v = np.asarray(values)
    if np.ndim(axis) != 1 or np.size(axis) != 2:
        raise ConfigurationError(
            f"{who}: axis={axis!r} must be the pair (depth axis, range "
            f"axis).")
    d_ax, r_ax = (normalize_axis(v, a, who) for a in axis)
    if d_ax == r_ax:
        raise ConfigurationError(
            f"{who}: axis={axis!r} names one axis twice; the depth and range "
            f"axes are two.")
    grid = (np.asarray(depths, dtype=float), np.asarray(ranges, dtype=float))
    v = np.moveaxis(v, (d_ax, r_ax), (0, 1))
    if v.shape[:2] != (grid[0].size, grid[1].size):
        raise ConfigurationError(
            f"{who}: values hold {v.shape[0]} depth(s) by {v.shape[1]} "
            f"range(s) on the axes {axis}, but depths has {grid[0].size} and "
            f"ranges {grid[1].size}.")
    if method == 'cubic' and min(v.shape[:2]) < 4:
        raise ConfigurationError(
            f"{who}: method='cubic' needs at least 4 samples on each axis; "
            f"got {v.shape[0]} depth(s) by {v.shape[1]} range(s).",
            remediation="Use method='linear' (or 'nearest').")
    new_d = np.atleast_1d(np.asarray(new_depths, dtype=float))
    new_r = np.atleast_1d(np.asarray(new_ranges, dtype=float))
    DD, RR = np.meshgrid(new_d, new_r, indexing='ij')
    points = np.stack([DD.ravel(), RR.ravel()], axis=-1)

    def resampled(part):
        return RegularGridInterpolator(
            grid, np.asarray(part, dtype=float), method=method,
            bounds_error=False, fill_value=np.nan)(points)

    if dB:
        out = interpolate_dB(v, resampled, method=method, who=who)
    elif np.iscomplexobj(v):
        out = resampled(v.real) + 1j * resampled(v.imag)
    else:
        out = resampled(v)
    out = out.reshape((new_d.size, new_r.size) + v.shape[2:])
    return np.moveaxis(out, (0, 1), (d_ax, r_ax))


def _query_profile(ranges, values, query, method='linear', *, who):
    """Sample a 1-D ``(range -> value)`` profile at ``query`` range(s);
    ``who`` names the caller in a refusal.

    Shared by the :class:`Bathymetry` / :class:`Altimetry` carriers (seafloor
    depth and sea-surface height vs range). ``'nearest'`` / ``'linear'`` /
    ``'cubic'`` with constant extrapolation past the ends. Returns a float for
    a scalar ``query``, an ndarray for an array of queries. Every query range
    must be finite; a NaN/inf anywhere in ``query`` raises
    ``ConfigurationError``.
    """
    if method not in INTERP_METHODS:
        raise ConfigurationError(
            f"{who}: method must be one of {INTERP_METHODS}; got "
            f"{method!r}.")
    x = np.asarray(ranges, dtype=float)
    y = np.asarray(values, dtype=float)
    rq = np.asarray(query, dtype=float)
    # The array counterpart of ``finite_scalar_label``: this profile query
    # takes a whole vector of ranges (models/base.py hands it the receiver
    # range axis), so the finiteness check is element-wise rather than the
    # scalar helper the collapsing selectors use.
    if not np.all(np.isfinite(rq)):
        n_bad = int(np.count_nonzero(~np.isfinite(rq)))
        count = ('' if rq.ndim == 0
                 else f" ({n_bad} of {rq.size} query ranges)")
        raise ConfigurationError(
            f"{who}: range={query!r} is not a finite label{count}: NaN and inf "
            f"ranges have no nearest node. Every |range - query| is NaN, so "
            f"the nearest lookup ranks nothing and searchsorted lands on an "
            f"end node — a real sample, which is why it would read as a "
            f"successful lookup — and an interpolated query propagates the "
            f"NaN into the result."
        )
    scalar = rq.ndim == 0
    if x.size == 1:
        out = np.full(rq.shape, float(y[0]))
    elif method == 'nearest':
        # searchsorted + midpoint comparison: same nearest-node answer as the
        # abs-difference argmin, without materialising the n_nodes x
        # n_queries cross-product (measured 6 GB at 20k x 20k). Ties at an
        # exact midpoint go to the LEFT node, matching argmin(0)'s
        # first-minimum rule on an ascending axis.
        rq_flat = rq.ravel()
        hi = np.clip(np.searchsorted(x, rq_flat), 1, x.size - 1)
        lo = hi - 1
        idx = np.where(rq_flat - x[lo] <= x[hi] - rq_flat, lo, hi)
        out = y[idx].reshape(rq.shape)
    elif method == 'linear':
        out = np.interp(rq, x, y)                    # constant extrapolation
    else:  # cubic
        if x.size < 4:
            raise ConfigurationError(
                f"{who}: method='cubic' needs >= 4 ranges; got {x.size}.",
                remediation="Use method='linear'.")
        from scipy.interpolate import interp1d
        f = interp1d(x, y, kind='cubic', bounds_error=False,
                     fill_value=(float(y[0]), float(y[-1])))
        out = f(np.clip(rq, x[0], x[-1]))
    return float(out) if scalar else np.asarray(out, dtype=float)
