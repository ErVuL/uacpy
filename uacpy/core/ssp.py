"""Sound-speed-profile shape carrier.
Re-exported from :mod:`uacpy.core.environment` for stable import paths.
"""

import warnings

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import replace as _replace_fields

from uacpy.core._repr import axis, build, extent
from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.deck_limits import (
    AT_LAST_SSP_POINT_EPS_M, DECK_DEPTH_RESOLUTION_M, DECK_RANGE_RESOLUTION_M,
)
from uacpy.core.exceptions import ConfigurationError, FallbackWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._grid import (
    collapse_axis, INTERP_METHODS, _as_finite_scalar_label,
)
from uacpy.core._validate import (
    scalar_or_none, reject_complex, require_positive, require_non_negative,
    require_strictly_increasing, warn_speed_typed_in_km_per_s,
)
from uacpy.core.collapse import RANGE_COLLAPSE_METHODS, _method_list
from uacpy.core._provenance import coerce_data_sources
from uacpy.core._plotting import plotter
from uacpy.core._carrier import (
    DeepCopyMixin, RevalidateOnAssignMixin, carrier,
)
from uacpy.core._export import CarrierExport

__all__ = [
    'SoundSpeedProfile',
]


_VALID_SSP_KINDS = (
    'measured', 'isovelocity', 'munk', 'analytic', 'n2linear',
)


# A caller passing ``env.depth`` that has round-tripped through I/O can be a few
# ulps off ``depths[-1]``; below this the two are the same request and the profile
# is returned untouched. Anything larger is *moved* onto the target rather than
# appended beside it — see ``extend_to`` for why appending breaks the reader.
_ROUND_TRIP_NOISE_M = 1.0e-9


def _flat_numeric_sequence(value):
    """``value`` as a 1-D float array when it is a non-empty flat sequence
    of numbers, else ``None`` (pair tables, ragged input and anything the
    float cast refuses fall through to :meth:`SoundSpeedProfile.from_pairs`,
    which reports the shape)."""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return arr if arr.ndim == 1 and arr.size else None


# eq=False: a dataclass __eq__ over ndarray fields raises; compare by identity.
@carrier(eq=False)
class SoundSpeedProfile(RevalidateOnAssignMixin, DeepCopyMixin, CarrierExport):
    """
    Unified sound-speed profile (1-D or 2-D).

    Stores the full grid as a 2-D array ``sound_speed[n_depth, n_range]``.
    Range-independent profiles use ``n_range = 1`` and ``ranges = None``;
    range-dependent profiles set ``ranges`` to a monotonically-increasing
    metres vector of length ``n_range``.

    Attributes
    ----------
    depths : ndarray, shape (N,)
        Depth axis in metres, monotonically increasing.
    sound_speed : ndarray, shape (N, M)
        Sound speed in m/s. ``M = 1`` for 1-D profiles.
    ranges : ndarray, shape (M,), optional
        Range axis in **metres**, monotonically increasing. ``None`` for 1-D.
    kind : str
        Declaration of what the data represents:
        ``'measured'`` (default), ``'isovelocity'``, ``'munk'``,
        ``'analytic'`` or ``'n2linear'``. Only ``'isovelocity'``
        actually overrides ``TopOpt(1)`` (forces ``'C'`` — any connection
        scheme over constant data is constant). The other values are
        informational metadata; the model's ``interp_ssp`` kwarg drives
        the AT character.
    """
    depths: np.ndarray
    sound_speed: np.ndarray
    ranges: Optional[np.ndarray] = None
    kind: str = 'measured'
    data_sources: tuple = ()
    #: Sound-speed formula that built ``sound_speed`` from T/S ('teos10', 'unesco',
    #: 'delgrosso', or 'mackenzie' from :meth:`from_temperature_salinity`); ``None`` for a
    #: literal or hand-built profile. Read by
    #: :func:`uacpy.data.extend_ssp_below_data`, which continues the column
    #: under the same equation (a ``None`` column takes the package default,
    #: TEOS-10).
    formula: Optional[str] = None

    # The export protocol: sound speed on (depth, range).
    _XARRAY_FIELDS = {'sound_speed': 'sound_speed', 'depth': 'depths',
                      'range': 'ranges'}

    def _payload(self):
        return {'sound_speed': (self.sound_speed, ('depth', 'range'), 'm/s')}

    def _coords(self):
        coords = {'depth': (self.depths, 'm')}
        if self.ranges is not None:
            coords['range'] = (self.ranges, 'm')
        return coords

    def __post_init__(self):
        # Provenance of a fetched profile (tuple of DataProvenance); empty for a
        # literal/hand-built one. Physics-agnostic metadata — transforms that
        # return a new profile (extend_to/collapse/eval slices) carry it
        # forward; the fresh-construction classmethods do not.
        self.data_sources = coerce_data_sources(
            self.data_sources, "SoundSpeedProfile")
        if self.formula is not None:
            from uacpy.core.acoustics.seawater import canonical_formula
            self.formula = canonical_formula(self.formula, "SoundSpeedProfile")
        # Ahead of the float64 casts below, which discard an imaginary part —
        # see _reject_complex for the two ways they do it.
        reject_complex(self.depths, "SoundSpeedProfile.depths")
        reject_complex(self.sound_speed, "SoundSpeedProfile sound speeds")
        self.depths = np.array(self.depths, dtype=float).reshape(-1)
        self.sound_speed = np.array(self.sound_speed, dtype=float)
        if self.sound_speed.ndim == 1:
            self.sound_speed = self.sound_speed.reshape(-1, 1)
        if self.sound_speed.ndim != 2:
            raise ConfigurationError(
                f"SoundSpeedProfile: sound_speed must be 1-D or 2-D; got {self.sound_speed.ndim}-D."
            )
        if self.sound_speed.shape[0] != self.depths.size:
            raise ConfigurationError(
                f"SoundSpeedProfile: sound_speed rows ({self.sound_speed.shape[0]}) must match "
                f"depths length ({self.depths.size})"
            )
        if self.depths.size == 0:
            raise ConfigurationError(
                "SoundSpeedProfile: needs at least one depth/sound-speed sample."
            )
        require_positive(self.sound_speed, "SoundSpeedProfile sound speeds", hint="m/s")
        warn_speed_typed_in_km_per_s(self.sound_speed,
                                     "SoundSpeedProfile sound speeds")
        require_strictly_increasing(self.depths, "SoundSpeedProfile.depths",
                                    min_step=DECK_DEPTH_RESOLUTION_M)
        if self.ranges is not None:
            reject_complex(self.ranges, "SoundSpeedProfile.ranges")
            self.ranges = np.array(self.ranges, dtype=float).reshape(-1)
            if self.ranges.size != self.sound_speed.shape[1]:
                raise ConfigurationError(
                    f"SoundSpeedProfile: ranges length ({self.ranges.size}) must "
                    f"match sound_speed columns ({self.sound_speed.shape[1]})"
                )
            require_non_negative(
                self.ranges, "SoundSpeedProfile.ranges", hint="metres")
            require_strictly_increasing(
                self.ranges, "SoundSpeedProfile.ranges",
                min_step=DECK_RANGE_RESOLUTION_M)
        elif self.sound_speed.shape[1] != 1:
            raise ConfigurationError(
                f"SoundSpeedProfile: ranges=None requires single-column sound_speed; "
                f"got shape {self.sound_speed.shape}."
            )
        self.kind = str(self.kind).lower()
        if self.kind not in _VALID_SSP_KINDS:
            raise ConfigurationError(
                f"SoundSpeedProfile: kind={self.kind!r} not in "
                f"{_VALID_SSP_KINDS}."
            )
        # ``isovelocity`` is the one kind a writer acts on rather than merely
        # records: it lets the AT deck declare TopOpt(1)='C' on the grounds
        # that any connection scheme over constant data is constant
        # (``resolve_ssp_topopt``). That reasoning only holds if the data
        # really is constant, so the declaration is checked instead of
        # trusted — otherwise a gradient is silently flattened.
        if self.kind == 'isovelocity' and float(np.ptp(self.sound_speed)) > 0.0:
            raise ConfigurationError(
                f"SoundSpeedProfile: kind='isovelocity' but the data spans "
                f"{float(np.min(self.sound_speed)):g}-{float(np.max(self.sound_speed)):g} "
                f"m/s.",
                remediation="Drop kind='isovelocity' (the default 'measured' "
                            "keeps every sample), or supply a constant "
                            "profile.",
            )

    def __repr__(self) -> str:
        bits = [self.kind, axis(self.depths, 'depths', 'm')]
        if self.is_range_dependent:
            bits.append(axis(self.ranges, 'ranges', 'm'))
        bits.append(f"c={extent(self.sound_speed, 'm/s')}")
        if self.formula is not None:
            bits.append(f"formula={self.formula}")
        return build('SoundSpeedProfile', bits)

    def plot(self, ax=None, **kwargs):
        """Plot the sound-speed profile ``c(z)`` (depth increasing downward).

        The carrier counterpart of :meth:`Result.plot` — any uacpy object you
        plot on its own has ``.plot()``. A range-dependent profile draws one
        line per range column. ``ax`` draws into an existing Axes, spelled the
        way every other uacpy plot method spells it; the remaining ``kwargs``
        are forwarded to :func:`uacpy.plot.plot_ssp`.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing axes; a new figure is made when omitted.
        **kwargs
            Keywords of :func:`uacpy.plot.plot_ssp`.
        """
        return plotter('plot_carrier')(self, ax=ax, **kwargs)

    @property
    def is_range_dependent(self) -> bool:
        """True when the profile carries more than one range column.

        A structural test (node count on the ranged axis), like ``Bottom`` /
        ``Surface``: a 2-D profile whose columns are identical still counts
        as range-dependent. Contrast ``Bathymetry.varies_with_range`` /
        ``Altimetry.varies_with_range``, which test whether the *values*
        actually vary with range."""
        return self.ranges is not None and self.sound_speed.shape[1] > 1

    @property
    def n_depths(self) -> int:
        return int(self.depths.size)

    @property
    def n_ranges(self) -> int:
        return int(self.sound_speed.shape[1])

    def to_pairs(self) -> np.ndarray:
        """Return ``(N, 2)`` ``(depth, c)`` array of the 1-D form.

        For range-dependent profiles, returns the range-0 column. Use
        ``at(range=)`` / ``eval(range=)`` for an explicit slice or
        ``collapse_range`` for a chosen reduction.
        """
        return np.column_stack([self.depths, self.sound_speed[:, 0]])

    def at(
        self, *, depth: Optional[float] = None, range: Optional[float] = None,
    ) -> 'SoundSpeedProfile':
        """Nearest-sample slice at the requested depth and/or range.

        Returns the closest stored grid sample on each axis — **never
        fabricates** a value (the grid-library invariant shared with
        ``Field.at`` et al.). For interpolated evaluation use :meth:`eval`;
        for an integer-index slice use :meth:`isel`.

        On a range-dependent profile a depth-only slice is ambiguous and
        raises: pin the range too, or :meth:`collapse_range` the range axis first.

        Parameters
        ----------
        depth : float, optional
            Depth (m) to slice at.
        range : float, optional
            Range (m) to slice at; required on a range-dependent profile.
        """
        return self._slice(depth=depth, range=range, interp='nearest')

    def eval(
        self, *, depth: Optional[float] = None, range: Optional[float] = None,
        method: str = 'linear',
    ) -> 'SoundSpeedProfile':
        """Interpolated slice at the requested depth and/or range.

        ``method`` is the interpolation scheme — ``'linear'`` (default),
        ``'nearest'``, or ``'cubic'`` — with constant extrapolation outside
        ``[ranges[0], ranges[-1]]``. The interpolating counterpart of
        :meth:`at` (which is always nearest), and subject to the same
        pin-the-range rule on a range-dependent profile.

        Parameters
        ----------
        depth : float, optional
            Depth (m) to slice at.
        range : float, optional
            Range (m) to slice at; required on a range-dependent profile.
        method : {'linear', 'nearest', 'cubic'}, optional
            Interpolation scheme. Default ``'linear'``.
        """
        return self._slice(depth=depth, range=range, interp=method)

    def sound_speed_at(
        self, depths, *, range: float = 0.0, method: str = 'linear',
    ) -> np.ndarray:
        """Sound speed (m/s) at ``depths`` (m), on the profile's column at
        ``range`` (m) when it is range-dependent, as an array.

        ``method`` interpolates across range and depth: ``'linear'``
        (default), ``'nearest'`` or ``'cubic'``. A depth outside the profile
        takes the nearest end's value, with a
        :class:`~uacpy.core.exceptions.FallbackWarning` naming the span: the
        value is held flat, not fabricated.

        Parameters
        ----------
        depths : float or array_like
            Depths (m).
        range : float, optional
            Range (m) of the column read. Default 0.
        method : {'linear', 'nearest', 'cubic'}, optional
            Interpolation scheme. Default ``'linear'``.
        """
        column = (self.eval(range=range, method=method)
                  if self.is_range_dependent else self)
        d = np.atleast_1d(np.asarray(depths, dtype=float))
        z = column.depths
        c = column.sound_speed[:, 0]
        if d.size and (np.any(d < z[0]) or np.any(d > z[-1])):
            warnings.warn(
                f"SoundSpeedProfile.sound_speed_at: depth(s) outside the "
                f"profile [{float(z[0]):.1f}, {float(z[-1]):.1f}] m were "
                f"constant-extrapolated to the nearest endpoint.",
                FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
        if method == 'linear':
            return np.interp(d, z, c)
        held = np.clip(d, z[0], z[-1])
        return np.array([collapse_axis(c, z, float(x), method, axis=0,
                                       name='depth')[0] for x in held])

    def _replace(self, **changes) -> 'SoundSpeedProfile':
        """A copy through the constructor with ``changes`` applied.

        Every field not named in ``changes`` is carried over: ``kind``,
        ``data_sources`` and ``formula`` (read by the deep extension in
        ``uacpy.data.sound_speed`` to continue a column under the equation
        that built it). ``__post_init__`` re-validates and copies the arrays.
        """
        return _replace_fields(self, **changes)

    def _require_pinned_range(self, who: str) -> None:
        """Guard a depth-only slice of a range-dependent profile.

        Silently returning the r = 0 column would be wrong physics on exactly
        the profiles the 2-D carrier exists for, so the caller must pin the
        range or collapse the axis.
        """
        if self.is_range_dependent:
            raise ConfigurationError(
                f"SoundSpeedProfile.{who}: a depth-only slice of a "
                f"range-dependent profile is ambiguous ({self.n_ranges} range "
                f"columns). Pin the range too ({who}(depth=…, range=…)) or "
                f"collapse the range axis first (collapse_range('r0'|'mean'|…))."
            )

    def isel(
        self, *, depth: Optional[int] = None, range: Optional[int] = None,
    ) -> 'SoundSpeedProfile':
        """Integer-index slice on the depth and/or range axis — the positional
        counterpart of :meth:`at`, subject to the same pin-the-range rule.

        Parameters
        ----------
        depth : int, optional
            Depth index.
        range : int, optional
            Range index; required on a range-dependent profile.
        """
        if depth is not None and range is None:
            self._require_pinned_range('isel')
        sliced = self
        if range is not None:
            ridx = int(range)
            if not -self.sound_speed.shape[1] <= ridx < self.sound_speed.shape[1]:
                raise IndexError(
                    f"SoundSpeedProfile.isel: range index {ridx} out of range "
                    f"for {self.sound_speed.shape[1]} column(s)")
            # Picking the only column keeps a single-node ``ranges`` — a
            # coordinate at that range, read by ``env.max_range`` (the
            # ``Bottom.collapse_range`` rule); picking one of several columns
            # collapses the axis and drops it.
            sliced = self._replace(
                sound_speed=self.sound_speed[:, [ridx]],
                ranges=self.ranges if self.sound_speed.shape[1] == 1 else None)
        if depth is not None:
            didx = int(depth)
            if not -sliced.depths.size <= didx < sliced.depths.size:
                raise IndexError(
                    f"SoundSpeedProfile.isel: depth index {didx} out of range "
                    f"for {sliced.depths.size} depth(s)")
            sliced = sliced._replace(
                depths=sliced.depths[[didx]], sound_speed=sliced.sound_speed[[didx], :])
        return sliced

    def _slice(
        self, *, depth: Optional[float], range: Optional[float], interp: str,
    ) -> 'SoundSpeedProfile':
        if interp not in INTERP_METHODS:
            raise ConfigurationError(
                f"SoundSpeedProfile: interpolation method must be one of "
                f"{INTERP_METHODS}; got {interp!r}.")
        if depth is not None and range is None:
            self._require_pinned_range('at' if interp == 'nearest' else 'eval')
        data = self.sound_speed
        # A single-node ``ranges`` is a coordinate at that range
        # (``env.max_range`` reads it, environment.py), so it travels with
        # the column instead of being dropped here — the rule
        # ``Bottom.collapse_range`` states for the same case. Collapsing the
        # range axis of a range-dependent profile at a label drops it.
        out_ranges = self.ranges
        if range is not None:
            if not self.is_range_dependent:
                # Any range reads the single column, but the label contract
                # (finite scalar) is the same one collapse_axis applies on
                # the range-dependent path.
                _as_finite_scalar_label(range, 'range')
                col = data[:, 0]
            else:
                col, _ = collapse_axis(data, self.ranges, range, interp,
                                       axis=1, name='range')
                out_ranges = None
            data = col.reshape(-1, 1)
        if depth is None:
            if range is None:
                return self
            return self._replace(sound_speed=data, ranges=out_ranges)
        c, dv = collapse_axis(data[:, 0], self.depths, depth, interp,
                              axis=0, name='depth')
        return self._replace(
            depths=np.array([float(dv)]), sound_speed=np.array([[float(c)]]),
            ranges=out_ranges)

    @property
    def value(self) -> float:
        """The single sound speed this profile represents, when unambiguous.

        Valid for an isovelocity profile (every sample equal) or one
        collapsed to a single ``(depth, range)`` cell via
        ``at(depth=, range=)``. Raises if the profile actually varies."""
        if self.sound_speed.size > 1 and np.ptp(self.sound_speed) > 0:
            raise ConfigurationError(
                f"SoundSpeedProfile.value: profile varies (shape "
                f"{self.sound_speed.shape}); slice with at(depth=, range=) first."
            )
        return float(self.sound_speed.flat[0])

    def collapse_range(self, method: str = 'r0') -> 'SoundSpeedProfile':
        """Collapse a 2-D profile to 1-D using ``method``.

        Returns ``self`` (not a copy) when the profile is already 1-D.

        Parameters
        ----------
        method : {'r0', 'rmax', 'mean', 'median'}, optional
            The reduction (see above). Default ``'r0'``.

        Methods
        -------
        ``'r0'``     : keep the range-0 column.
        ``'mean'``   : depth-wise mean over the range columns, each column
                       weighing the same whatever its spacing (a node mean,
                       not a range average).
        ``'median'`` : depth-wise median over the range columns.
        ``'rmax'``   : keep the last (deepest range) column.
        """
        # Validated before the early return: a range-independent
        # carrier has nothing to reduce, and returning self first
        # made a typo silent until the user switched to a
        # range-dependent environment — the moment they are least
        # looking for one.
        if method not in RANGE_COLLAPSE_METHODS:
            raise ConfigurationError(
                f"SoundSpeedProfile.collapse_range: unknown method={method!r}; "
                f"valid: {_method_list(RANGE_COLLAPSE_METHODS)}."
            )
        if not self.is_range_dependent:
            return self
        if method == 'r0':
            col = self.sound_speed[:, 0]
        elif method == 'rmax':
            col = self.sound_speed[:, -1]
        elif method == 'mean':
            col = self.sound_speed.mean(axis=1)
        elif method == 'median':
            col = np.median(self.sound_speed, axis=1)
        else:                       # pragma: no cover - guarded at entry
            raise ConfigurationError(
                f"SoundSpeedProfile.collapse_range: unreachable method "
                f"{method!r} — the vocabulary is checked on entry."
            )
        return self._replace(sound_speed=col.reshape(-1, 1), ranges=None)

    def extend_to(self, depth_max: float) -> 'SoundSpeedProfile':
        """Return a copy with the deepest sample sitting exactly at
        ``depth_max``.

        Three cases:

        * ``depth_max == depths[-1]`` — return ``self`` unchanged.
        * ``depth_max > depths[-1]`` — append a new sample at
          ``depth_max`` carrying the deepest existing sound speed
          (constant extrapolation, the AT writer convention).
        * ``depth_max < depths[-1]`` — truncate samples below
          ``depth_max`` and interpolate a final sample exactly at
          ``depth_max`` so writers that require ``ssp[-1] == env.depth``
          (Bellhop / Kraken) round-trip without manual alignment. When the
          deepest surviving sample already sits within
          ``AT_LAST_SSP_POINT_EPS_M`` of ``depth_max`` it is *moved* onto it
          instead, for the reason the comment below gives.

        ``depth_max`` must be a finite depth below the profile's first
        sample (outside the epsilon windows above): the returned profile's
        deepest sample sits exactly at ``depth_max``, so a target at or
        above ``depths[0]`` would leave no sample to keep and raises
        ``ConfigurationError``.

        Parameters
        ----------
        depth_max : float
            Depth (m) the deepest sample is placed at.
        """
        try:
            depth_max = float(depth_max)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"SoundSpeedProfile.extend_to: depth_max={depth_max!r} is "
                f"not a single depth in metres."
            ) from exc
        if not np.isfinite(depth_max):
            raise ConfigurationError(
                f"SoundSpeedProfile.extend_to: depth_max={depth_max!r} is "
                f"not a finite depth, so no deepest sample can sit exactly "
                f"at it."
            )
        # ``misc/sspMod.f90:353`` ends a medium's SSP block at the first sample
        # within AT_LAST_SSP_POINT_EPS_M of the declared medium depth, so a second
        # sample inside that window is never read as an SSP row — the reader takes
        # it as the bottom-option record and the boundary condition comes out of a
        # sound speed. Anything the reader would already call the last point must
        # therefore *move* the existing sample, never add one beside it. The band
        # this closes is 1e-9 m to 1.19e-5 m, which about 1 in 8400 arbitrary
        # depths lands in; a fetched or interpolated bathymetry reaches it
        # naturally, and the abort the user saw named the boundary condition.
        last = float(self.depths[-1])
        gap = abs(float(depth_max) - last)
        if gap <= _ROUND_TRIP_NOISE_M:
            return self          # a float round-trip artefact, not a request
        if gap < AT_LAST_SSP_POINT_EPS_M:
            # Rebuild through the constructor so the moved sample is validated:
            # a downward snap larger than the deck resolution can land at or
            # below ``depths[-2]``, and that must raise as an invalid axis
            # rather than return a non-increasing profile.
            snapped_depths = self.depths.copy()
            snapped_depths[-1] = float(depth_max)
            return self._replace(depths=snapped_depths)
        first = float(self.depths[0])
        if depth_max <= first:
            raise ConfigurationError(
                f"SoundSpeedProfile.extend_to: depth_max={depth_max:g} m is "
                f"not below the profile's first sample ({first:g} m). The "
                f"returned profile's deepest sample sits exactly at "
                f"depth_max, so truncating to this target would discard "
                f"every sample the profile has.",
                remediation="Pass a depth below the first sample, or build "
                            "a new profile (from_pairs / from_isovelocity) "
                            "if the water column really is this shallow.",
            )
        if depth_max > last:
            new_depths = np.append(self.depths, depth_max)
            new_data = np.vstack([self.sound_speed, self.sound_speed[-1:, :]])
        else:
            keep = self.depths < depth_max
            kept_depths = self.depths[keep]
            kept_data = self.sound_speed[keep]
            if (kept_depths.size
                    and depth_max - kept_depths[-1] < AT_LAST_SSP_POINT_EPS_M):
                # The deepest surviving sample already falls inside the
                # reader's last-point window, so it is the row the reader will
                # treat as the end of the block: snap it onto depth_max rather
                # than interpolating a second row beside it, which the reader
                # would consume as the bottom-option record. The move is
                # upward, so the axis stays strictly increasing and needs no
                # revalidation.
                new_depths = kept_depths
                new_depths[-1] = float(depth_max)
                new_data = kept_data
            else:
                interp_row = np.array([
                    np.interp(depth_max, self.depths, self.sound_speed[:, j])
                    for j in range(self.sound_speed.shape[1])
                ])
                new_depths = np.append(kept_depths, depth_max)
                new_data = np.vstack([kept_data, interp_row[None, :]])
        return self._replace(depths=new_depths, sound_speed=new_data)

    @classmethod
    def coerce(
        cls, value, *, depth_max: float,
    ) -> 'SoundSpeedProfile':
        """Coerce the user-facing ``ssp=`` shorthand into a profile.

        Mirrors :meth:`Bathymetry.coerce` / :meth:`Altimetry.coerce` so
        :class:`~uacpy.core.environment.Environment` delegates instead of
        hand-rolling the dispatch:

        * ``None`` — isovelocity 1500 m/s spanning ``0..depth_max``.
        * scalar (m/s), in any spelling — a Python number, a numpy scalar or
          a 0-d array — isovelocity at that speed spanning ``0..depth_max``.
        * ``(depth, c)`` pairs — linear profile via :meth:`from_pairs`. A
          tuple of two equal-length 1-D columns ``(depths, speeds)`` is
          refused, naming ``from_pairs(np.column_stack([depths, speeds]))``.
          A 2x2 input is always two pairs ``((z0, c0), (z1, c1))``; two
          2-sample columns must go through ``column_stack`` themselves.
        * a :class:`SoundSpeedProfile` — returned as-is (by reference).

        ``None`` policy: an isovelocity 1500 m/s water column — a usable
        default profile, since every environment has *some* sound speed.

        ``depth_max`` (m) sets the column extent for the isovelocity cases.

        Parameters
        ----------
        value : None, float, array_like or SoundSpeedProfile
            The ``ssp=`` value (see above).
        depth_max : float
            Depth (m) an isovelocity profile spans.
        """
        if value is None:
            return cls.from_isovelocity(depth_max, DEFAULT_SOUND_SPEED)
        if isinstance(value, SoundSpeedProfile):
            return value
        # A scalar (a 0-d array included) is an isovelocity ocean; a bool is
        # refused as one — the shared guard says why.
        sound_speed = scalar_or_none(value, lambda v: (
            f"Environment: ssp={v!r} is a bool, not a sound speed — as a "
            f"scalar it would mean a {float(v):g} m/s ocean."))
        if sound_speed is not None:
            return cls.from_isovelocity(depth_max, sound_speed)
        if isinstance(value, np.ndarray) and value.ndim == 0:
            # A 0-d array holding a string or an object.
            raise ConfigurationError(
                f"Environment: ssp must be a scalar (m/s), a list of "
                f"(depth, sound_speed) pairs, or a SoundSpeedProfile; got "
                f"a 0-d array of dtype {value.dtype}.")
        if isinstance(value, (list, tuple, np.ndarray)):
            flat = _flat_numeric_sequence(value)
            if flat is not None:
                # A 1-D sequence of numbers is neither a scalar nor a pair
                # table; say which of the two the caller may have meant.
                raise ConfigurationError(
                    f"Environment: ssp must be a scalar (m/s), a list of "
                    f"(depth, sound_speed) pairs, or a SoundSpeedProfile; "
                    f"got ssp={value!r}, a flat sequence of {flat.size} "
                    f"number(s). For an isovelocity ocean pass "
                    f"ssp={flat[0]:g}; for a profile pass pairs of shape "
                    f"(N, 2): ssp=[(depth_m, sound_speed), ...].")
            if (isinstance(value, tuple) and len(value) == 2
                    and all(np.ndim(part) == 1 for part in value)
                    and len(value[0]) == len(value[1]) != 2):
                # ``(depths, speeds)``: two equal-length columns that cannot
                # be pairs, since a pair has two entries. A 2x2 input is read
                # as two pairs — the documented form — so a pair of
                # 2-sample columns is indistinguishable from it.
                raise ConfigurationError(
                    "Environment: ssp=(depths, speeds) is two columns; ssp= "
                    "takes (depth, sound_speed) PAIRS or a SoundSpeedProfile.",
                    remediation="Pass ssp=SoundSpeedProfile.from_pairs("
                                "np.column_stack([depths, speeds])).")
            return cls.from_pairs(value)
        raise ConfigurationError(
            f"Environment: ssp must be a scalar (m/s), a list of (depth, "
            f"sound_speed) pairs, or a SoundSpeedProfile; got "
            f"{type(value).__name__}."
        )

    @classmethod
    def from_isovelocity(
        cls, depth_max: float, sound_speed: float = DEFAULT_SOUND_SPEED
    ) -> 'SoundSpeedProfile':
        """Constant-``sound_speed`` (m/s) profile spanning 0 to ``depth_max`` (m).

        Parameters
        ----------
        depth_max : float
            Depth (m) the profile spans.
        sound_speed : float, optional
            Sound speed (m/s). Default :data:`~uacpy.core.constants.DEFAULT_SOUND_SPEED`.
        """
        return cls(
            depths=np.array([0.0, float(depth_max)]),
            sound_speed=np.full((2, 1), float(sound_speed)),
            ranges=None,
            kind='isovelocity',
        )

    @classmethod
    def from_pairs(
        cls,
        pairs: Union[List[Tuple[float, float]], np.ndarray],
        kind: str = 'measured',
    ) -> 'SoundSpeedProfile':
        """Build a 1-D profile from ``[(depth, c), …]`` pairs.

        ``kind`` is informational metadata (``'measured'`` default);
        see :class:`SoundSpeedProfile`. The model's ``interp_ssp`` kwarg
        drives the sample-connection scheme.

        Parameters
        ----------
        pairs : array_like
            ``(depth, sound_speed)`` rows, shape ``(N, 2)``.
        kind : str, optional
            Informational label. Default ``'measured'``.
        """
        arr = np.asarray(pairs, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            hint = (" — a (depths, speeds) pair of columns is "
                    "from_pairs(np.column_stack([depths, speeds]))"
                    if arr.ndim == 2 and arr.shape[0] == 2 else "")
            raise ConfigurationError(
                f"SoundSpeedProfile.from_pairs: pairs must have shape (N, 2) "
                f"as (depth, sound_speed); got shape {arr.shape}{hint}."
            )
        return cls(
            depths=arr[:, 0],
            sound_speed=arr[:, 1].reshape(-1, 1),
            ranges=None,
            kind=kind,
        )

    @classmethod
    def from_2d(
        cls,
        depths: np.ndarray,
        ranges: np.ndarray,
        matrix: np.ndarray,
        kind: str = 'measured',
    ) -> 'SoundSpeedProfile':
        """Build a 2-D profile from a depth axis, range axis (metres),
        and ``c(depth, range)`` matrix of shape ``(n_depth, n_range)``.

        For Bellhop, pair with ``Bellhop(interp_ssp='quad')`` to enable
        the external ``.ssp`` (quad) file format.

        Parameters
        ----------
        depths : ndarray
            Depth axis (m).
        ranges : ndarray
            Range axis (m).
        matrix : ndarray
            ``(n_depth, n_range)`` sound speeds (m/s).
        kind : str, optional
            Informational label. Default ``'measured'``.
        """
        return cls(
            depths=np.asarray(depths, dtype=float),
            sound_speed=np.asarray(matrix, dtype=float),
            ranges=np.asarray(ranges, dtype=float),
            kind=kind,
        )

    @classmethod
    def from_casts(
        cls,
        ranges,
        casts,
        *,
        depths=None,
        kind: str = 'measured',
    ) -> 'SoundSpeedProfile':
        """Build a range-dependent profile from sound-speed casts that each
        carry their own depth grid.

        Field data rarely shares one depth axis: a shelf cast stops at 60 m
        beside an 800 m slope cast, and a raw down/up cast repeats and
        unorders depths. Each cast is sorted by depth, samples taken at the
        same depth are averaged, and every cast is interpolated linearly onto
        one common axis — ``depths`` if given, otherwise the union of all the
        casts' depths, so no cast loses a node. A cast shorter than the axis
        holds its deepest value below its last sample, and one starting
        below the axis's top holds its shallowest value above it (constant
        extension, the rule :func:`uacpy.data.assemble_range_dependent`
        applies to fetched columns); extend a cast under a sound-speed
        formula first if the deep gradient matters.

        Parameters
        ----------
        ranges : array_like
            Range of each cast (m), strictly increasing, one per cast.
        casts : sequence of array_like
            One ``(n_i, 2)`` array — or list of ``(depth_m, sound_speed)``
            pairs — per cast: rows, as :meth:`from_pairs` takes them. A
            cast given as a tuple of two equal-length columns
            ``(depths, speeds)`` longer than 2 is refused naming
            ``np.column_stack``; a 2x2 cast is always two rows, as in
            ``Environment(ssp=...)``.
        depths : array_like, optional
            Common depth axis (m). Default the union of the casts' depths.
        kind : str, optional
            Informational metadata, as on :meth:`from_pairs`.

        Examples
        --------
        >>> shelf = [(0.0, 1500.0), (60.0, 1495.0)]
        >>> slope = [(0.0, 1502.0), (100.0, 1490.0), (800.0, 1485.0)]
        >>> ssp = SoundSpeedProfile.from_casts([0.0, 5000.0], [shelf, slope])
        >>> ssp.depths
        array([  0.,  60., 100., 800.])
        >>> ssp.sound_speed[:, 0]
        array([1500., 1495., 1495., 1495.])
        """
        who = "SoundSpeedProfile.from_casts"
        r = np.atleast_1d(np.asarray(ranges, dtype=float))
        casts = list(casts)
        if r.ndim != 1 or r.size != len(casts):
            raise ConfigurationError(
                f"{who}: one range per cast; got {r.size} range(s) for "
                f"{len(casts)} cast(s).")
        if r.size < 2:
            raise ConfigurationError(
                f"{who}: a range-dependent profile needs at least two casts; "
                f"for one, use from_pairs.")
        columns = []
        for i, cast in enumerate(casts):
            if (isinstance(cast, tuple) and len(cast) == 2
                    and all(np.ndim(part) == 1 for part in cast)
                    and len(cast[0]) == len(cast[1]) != 2):
                raise ConfigurationError(
                    f"{who}: cast {i} is a (depths, speeds) pair of columns; "
                    f"a cast is (depth, sound_speed) rows.",
                    remediation="Pass np.column_stack([depths, speeds]) for "
                                "that cast.")
            arr = np.asarray(cast, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 1:
                raise ConfigurationError(
                    f"{who}: cast {i} must be (depth, sound_speed) rows of "
                    f"shape (n, 2); got shape {arr.shape}.")
            if not np.all(np.isfinite(arr)):
                raise ConfigurationError(
                    f"{who}: cast {i} holds a non-finite depth or speed; "
                    f"drop those samples first.")
            z_u, inverse = np.unique(arr[:, 0], return_inverse=True)
            c_u = (np.bincount(inverse, weights=arr[:, 1])
                   / np.bincount(inverse))
            columns.append((z_u, c_u))
        if depths is None:
            axis = np.unique(np.concatenate([z for z, _ in columns]))
        else:
            axis = np.asarray(depths, dtype=float).ravel()
        matrix = np.column_stack([np.interp(axis, z, c) for z, c in columns])
        return cls(depths=axis, sound_speed=matrix, ranges=r, kind=kind)

    @classmethod
    def from_munk(
        cls, depth_max: float, n_points: int = 101
    ) -> 'SoundSpeedProfile':
        """Munk canonical profile with axis at 1300 m, c_min = 1500 m/s.

        ``c(z) = 1500·[1 + ε(z̃ − 1 + e^−z̃)]`` with ``ε = 0.00737`` and
        ``z̃ = 2(z − 1300)/1300``, i.e. Jensen, Kuperman, Porter & Schmidt,
        *Computational Ocean Acoustics*, §5.6 "A Deep Water Problem: The Munk
        Profile". ``z̃ − 1 + e^−z̃`` is zero at ``z̃ = 0`` and positive
        elsewhere, so 1500 m/s is the sound-channel minimum, on the axis.

        Parameters
        ----------
        depth_max : float
            Depth (m) the profile spans.
        n_points : int, optional
            Depth samples. Default 101.

        References
        ----------
        Munk, W. H. (1974). "Sound channel in an exponentially stratified ocean,
        with application to SOFAR." JASA 55(2), 220-226.
        """
        depths = np.linspace(0.0, float(depth_max), int(n_points))
        z_axis = 1300.0
        epsilon = 0.00737
        c_min = 1500.0
        eta = 2.0 * (depths - z_axis) / z_axis
        c = c_min * (1.0 + epsilon * (eta - 1.0 + np.exp(-eta)))
        return cls(
            depths=depths,
            sound_speed=c.reshape(-1, 1),
            ranges=None,
            kind='munk',
        )

    @classmethod
    def from_temperature_salinity(
        cls,
        depths: Optional[np.ndarray],
        temperature,
        salinity,
        *,
        formula: Optional[str] = None,
        latitude_deg: Optional[float] = None,
    ) -> 'SoundSpeedProfile':
        """Build a profile from in-situ ``T(z)`` and ``S(z)``.

        ``temperature`` and ``salinity`` each follow the water-property
        rule (:func:`uacpy.core._validate.water_property`): a single value,
        a 1-D array on ``depths``, or ``(depth, value)`` pairs on depths of
        their own, interpolated linearly onto the profile's depths (end
        values held). The profile's depths are ``depths``, or, given
        ``None``, the union of the pairs' depths.

        ``formula`` selects the equation and defaults to
        ``DEFAULT_SOUND_SPEED_FORMULA`` (TEOS-10), the same default every
        ``fetch_ssp*`` route carries, so an in-memory profile and a fetched
        one agree unless you ask otherwise. ``formula='mackenzie'`` selects
        Mackenzie's equation, which sits about 0.2 m/s from the default at
        4 km.

        Three of the four equations are stated in **pressure**; Mackenzie is
        stated in depth and is evaluated on ``depths`` directly. The
        conversion is Leroy & Parthiot's standard ocean and needs a latitude,
        which defaults to the equation's own reference 45 deg — see
        ``REFERENCE_LATITUDE_DEG`` for what that costs. The array-level
        function is :func:`uacpy.acoustics.sound_speed_at_depth`.

        The profile records ``formula``, so
        :func:`uacpy.data.extend_ssp_below_data` continues it under the same
        equation that built it.

        Parameters
        ----------
        depths : ndarray or None
            Depths (m) of the profile; ``None`` takes the union of the
            pairs' depths (then at least one property must be pairs).
        temperature : float, ndarray or (N, 2) pairs
            In-situ temperature (°C).
        salinity : float, ndarray or (N, 2) pairs
            Salinity (PSU).
        formula : str, optional
            Sound-speed equation; ``None`` is the package default (TEOS-10).
        latitude_deg : float, optional
            Latitude (deg) of the depth-to-pressure conversion; ``None`` is 45.

        Examples
        --------
        >>> ssp = SoundSpeedProfile.from_temperature_salinity(
        ...     None, [(0.0, 22.0), (60.0, 12.0), (200.0, 10.0)], 35.0,
        ...     formula='mackenzie')
        >>> ssp.depths.tolist()
        [0.0, 60.0, 200.0]
        """
        from uacpy.core._validate import water_property
        from uacpy.core.acoustics.seawater import (
            canonical_formula, sound_speed_at_depth,
        )
        who = 'SoundSpeedProfile.from_temperature_salinity'
        formula = canonical_formula(formula, who)
        given = {'temperature': temperature, 'salinity': salinity}
        pair_axes = [np.asarray(v, dtype=float)[:, 0]
                     for v in given.values()
                     if np.ndim(v) == 2 and np.shape(v)[1] == 2]
        if depths is None:
            if not pair_axes:
                raise ConfigurationError(
                    f"{who}: depths=None takes the union of the "
                    f"(depth, value) pairs' depths, and no property is "
                    f"given as pairs.",
                    remediation="Pass depths=, or give temperature / "
                                "salinity as (depth, value) pairs.")
            z = np.unique(np.concatenate(pair_axes))
        else:
            z = np.asarray(depths, dtype=float).ravel()
        values = {}
        for name, value in given.items():
            v = np.asarray(water_property(value, z, name=name, who=who),
                           dtype=float)
            if v.ndim == 0:
                v = np.full(z.shape, float(v))
            values[name] = v.ravel()
        T, S = values['temperature'], values['salinity']
        if not (T.shape == S.shape == z.shape):
            raise ConfigurationError(
                f"{who}: depths, temperature, salinity must share "
                f"shape; got {z.shape}, {T.shape}, {S.shape}."
            )
        c = sound_speed_at_depth(T, S, z, formula=formula,
                                 latitude_deg=latitude_deg)
        return cls(
            depths=z, sound_speed=np.asarray(c).reshape(-1, 1),
            ranges=None, formula=formula,
        )

