"""Sound-speed-profile shape carrier and the rough sea-surface generator.
Re-exported from :mod:`uacpy.core.environment` for stable import paths.
"""

import copy as _copy
import warnings

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

from uacpy.core.constants import (DEFAULT_SOUND_SPEED,
                                  AT_LAST_SSP_POINT_EPS_M,
                                  DECK_DEPTH_RESOLUTION_M,
                                  DECK_RANGE_RESOLUTION_M)
from uacpy.core.exceptions import ConfigurationError
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._carrier_validate import (
    _reject_complex,
    _require_positive, _require_non_negative, _require_strictly_increasing,
    _coerce_data_sources,
)


_VALID_SSP_SHAPES = (
    'measured', 'isovelocity', 'munk', 'analytic', 'n2linear',
)


# A caller passing ``env.depth`` that has round-tripped through I/O can be a few
# ulps off ``depths[-1]``; below this the two are the same request and the profile
# is returned untouched. Anything larger is *moved* onto the target rather than
# appended beside it — see ``extend_to`` for why appending breaks the reader.
_ROUND_TRIP_NOISE_M = 1.0e-9


# eq=False: a dataclass __eq__ over ndarray fields raises; compare by identity.
@dataclass(eq=False)
class SoundSpeedProfile:
    """
    Unified sound-speed profile (1-D or 2-D).

    Stores the full grid as a 2-D array ``data[n_depth, n_range]``.
    Range-independent profiles use ``n_range = 1`` and ``ranges = None``;
    range-dependent profiles set ``ranges`` to a monotonically-increasing
    metres vector of length ``n_range``.

    Attributes
    ----------
    depths : ndarray, shape (N,)
        Depth axis in metres, monotonically increasing.
    data : ndarray, shape (N, M)
        Sound speed in m/s. ``M = 1`` for 1-D profiles.
    ranges : ndarray, shape (M,), optional
        Range axis in **metres**, monotonically increasing. ``None`` for 1-D.
    shape : str
        Declaration of what the data represents:
        ``'measured'`` (default), ``'isovelocity'``, ``'munk'``,
        ``'analytic'`` or ``'n2linear'``. Only ``'isovelocity'``
        actually overrides ``TopOpt(1)`` (forces ``'C'`` — any connection
        scheme over constant data is constant). The other values are
        informational metadata; the model's ``interp_ssp`` kwarg drives
        the AT character.
    """
    depths: np.ndarray
    data: np.ndarray
    ranges: Optional[np.ndarray] = None
    shape: str = 'measured'
    data_sources: tuple = ()

    def __post_init__(self):
        # Provenance of a fetched profile (tuple of DataProvenance); empty for a
        # literal/hand-built one. Physics-agnostic metadata — transforms that
        # return a new profile (extend_to/collapse/eval slices) carry it
        # forward; the fresh-construction classmethods do not.
        self.data_sources = _coerce_data_sources(
            self.data_sources, "SoundSpeedProfile")
        # Ahead of the float64 casts below, which discard an imaginary part —
        # see _reject_complex for the two ways they do it.
        _reject_complex(self.depths, "SoundSpeedProfile.depths")
        _reject_complex(self.data, "SoundSpeedProfile sound speeds")
        self.depths = np.array(self.depths, dtype=float).reshape(-1)
        self.data = np.array(self.data, dtype=float)
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)
        if self.data.ndim != 2:
            raise ConfigurationError(
                f"SoundSpeedProfile: data must be 1-D or 2-D; got {self.data.ndim}-D"
            )
        if self.data.shape[0] != self.depths.size:
            raise ConfigurationError(
                f"SoundSpeedProfile: data rows ({self.data.shape[0]}) must match "
                f"depths length ({self.depths.size})"
            )
        if self.depths.size == 0:
            raise ConfigurationError(
                "SoundSpeedProfile: needs at least one depth/sound-speed sample"
            )
        _require_positive(self.data, "SoundSpeedProfile sound speeds", hint="m/s")
        _require_strictly_increasing(self.depths, "SoundSpeedProfile.depths",
                                     min_step=DECK_DEPTH_RESOLUTION_M)
        if self.ranges is not None:
            _reject_complex(self.ranges, "SoundSpeedProfile.ranges")
            self.ranges = np.array(self.ranges, dtype=float).reshape(-1)
            if self.ranges.size != self.data.shape[1]:
                raise ConfigurationError(
                    f"SoundSpeedProfile: ranges length ({self.ranges.size}) must "
                    f"match data columns ({self.data.shape[1]})"
                )
            _require_non_negative(
                self.ranges, "SoundSpeedProfile.ranges", hint="metres")
            _require_strictly_increasing(
                self.ranges, "SoundSpeedProfile.ranges",
                min_step=DECK_RANGE_RESOLUTION_M)
        elif self.data.shape[1] != 1:
            raise ConfigurationError(
                f"SoundSpeedProfile: ranges=None requires single-column data; "
                f"got shape {self.data.shape}"
            )
        self.shape = str(self.shape).lower()
        if self.shape not in _VALID_SSP_SHAPES:
            raise ConfigurationError(
                f"SoundSpeedProfile: shape={self.shape!r} not in "
                f"{_VALID_SSP_SHAPES}"
            )
        # ``isovelocity`` is the one shape a writer acts on rather than merely
        # records: it lets the AT deck declare TopOpt(1)='C' on the grounds
        # that any connection scheme over constant data is constant
        # (``resolve_ssp_topopt``). That reasoning only holds if the data
        # really is constant, so the declaration is checked instead of
        # trusted — otherwise a gradient is silently flattened.
        if self.shape == 'isovelocity' and float(np.ptp(self.data)) > 0.0:
            raise ConfigurationError(
                f"SoundSpeedProfile: shape='isovelocity' but the data spans "
                f"{float(np.min(self.data)):g}-{float(np.max(self.data)):g} "
                f"m/s.",
                remediation="Drop shape='isovelocity' (the default 'measured' "
                            "keeps every sample), or supply a constant "
                            "profile.",
            )

    def __repr__(self) -> str:
        c_lo = float(np.min(self.data))
        c_hi = float(np.max(self.data))
        bits = [
            f"shape={self.shape!r}",
            f"n_z={self.depths.size}",
            f"z=[{float(self.depths[0]):g}, {float(self.depths[-1]):g}] m",
        ]
        if self.is_range_dependent:
            r_lo = float(self.ranges[0]) / 1000
            r_hi = float(self.ranges[-1]) / 1000
            bits.append(f"n_r={self.data.shape[1]}")
            bits.append(f"range=[{r_lo:g}, {r_hi:g}] km")
        bits.append(f"c=[{c_lo:g}, {c_hi:g}] m/s")
        return f"SoundSpeedProfile({', '.join(bits)})"

    def plot(self, ax=None, **kwargs):
        """Plot the sound-speed profile ``c(z)`` (depth increasing downward).

        The carrier counterpart of :meth:`Result.plot` — any uacpy object you
        plot on its own has ``.plot()``. A range-dependent profile draws one
        line per range column. ``ax`` draws into an existing Axes, spelled the
        way every other uacpy plot method spells it; the remaining ``kwargs``
        are forwarded to the renderer."""
        # Deferred into the body: ``uacpy.visualization`` imports
        # ``uacpy.core`` at module scope, so this line at file scope makes
        # ``import uacpy`` raise ImportError. docs/DEV.md section 7 records
        # the inversion.
        from uacpy.visualization.plots.environment import _plot_ssp
        return _plot_ssp(self, ax=ax, **kwargs)

    @property
    def is_range_dependent(self) -> bool:
        """True when the profile carries more than one range column.

        A structural test (node count on the ranged axis), like ``Bottom`` /
        ``Surface``: a 2-D profile whose columns are identical still counts
        as range-dependent. Contrast ``Bathymetry.is_range_dependent`` /
        ``Altimetry.is_range_dependent``, which test whether the *values*
        actually vary with range."""
        return self.ranges is not None and self.data.shape[1] > 1

    @property
    def n_depths(self) -> int:
        return int(self.depths.size)

    @property
    def n_ranges(self) -> int:
        return int(self.data.shape[1])

    def to_pairs(self) -> np.ndarray:
        """Return ``(N, 2)`` ``(depth, c)`` array of the 1-D form.

        For range-dependent profiles, returns the range-0 column. Use
        ``at(range=)`` / ``eval(range=)`` for an explicit slice or
        ``collapse`` for a chosen reduction.
        """
        return np.column_stack([self.depths, self.data[:, 0]])

    def at(
        self, *, depth: Optional[float] = None, range: Optional[float] = None,
    ) -> 'SoundSpeedProfile':
        """Nearest-sample slice at the requested depth and/or range.

        Returns the closest stored grid sample on each axis — **never
        fabricates** a value (the grid-library invariant shared with
        ``Field.at`` et al.). For interpolated evaluation use :meth:`eval`;
        for an integer-index slice use :meth:`isel`.

        On a range-dependent profile a depth-only slice is ambiguous and
        raises: pin the range too, or :meth:`collapse` the range axis first.
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
        """
        return self._slice(depth=depth, range=range, interp=method)

    def _require_pinned_range(self, caller: str) -> None:
        """Guard a depth-only slice of a range-dependent profile.

        Silently returning the r = 0 column would be wrong physics on exactly
        the profiles the 2-D carrier exists for, so the caller must pin the
        range or collapse the axis.
        """
        if self.is_range_dependent:
            raise ConfigurationError(
                f"SoundSpeedProfile.{caller}: a depth-only slice of a "
                f"range-dependent profile is ambiguous ({self.n_ranges} range "
                f"columns). Pin the range too ({caller}(depth=…, range=…)) or "
                f"collapse the range axis first (collapse('r0'|'mean'|…))."
            )

    def isel(
        self, *, depth: Optional[int] = None, range: Optional[int] = None,
    ) -> 'SoundSpeedProfile':
        """Integer-index slice on the depth and/or range axis — the positional
        counterpart of :meth:`at`, subject to the same pin-the-range rule."""
        if depth is not None and range is None:
            self._require_pinned_range('isel')
        sliced = self
        if range is not None:
            ridx = int(range)
            if not -self.data.shape[1] <= ridx < self.data.shape[1]:
                raise IndexError(
                    f"SoundSpeedProfile.isel: range index {ridx} out of range "
                    f"for {self.data.shape[1]} column(s)")
            sliced = SoundSpeedProfile(
                depths=self.depths.copy(),
                data=self.data[:, [ridx]].copy(), ranges=None, shape=self.shape,
                data_sources=self.data_sources)
        if depth is not None:
            didx = int(depth)
            if not -sliced.depths.size <= didx < sliced.depths.size:
                raise IndexError(
                    f"SoundSpeedProfile.isel: depth index {didx} out of range "
                    f"for {sliced.depths.size} depth(s)")
            sliced = SoundSpeedProfile(
                depths=np.array([float(sliced.depths[didx])]),
                data=sliced.data[[didx], :].copy(), ranges=None,
                shape=sliced.shape, data_sources=self.data_sources)
        return sliced

    def _slice(
        self, *, depth: Optional[float], range: Optional[float], interp: str,
    ) -> 'SoundSpeedProfile':
        from uacpy.core._grid import (
            collapse_axis, INTERP_METHODS, _as_finite_scalar_label,
        )
        if interp not in INTERP_METHODS:
            raise ConfigurationError(
                f"SoundSpeedProfile: interpolation method must be one of "
                f"{INTERP_METHODS}; got {interp!r}")
        if depth is not None and range is None:
            self._require_pinned_range('at' if interp == 'nearest' else 'eval')
        data = self.data
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
            data = col.reshape(-1, 1)
        if depth is None:
            if range is None:
                return self
            return SoundSpeedProfile(
                depths=self.depths.copy(), data=data.copy(),
                ranges=None, shape=self.shape, data_sources=self.data_sources)
        c, dv = collapse_axis(data[:, 0], self.depths, depth, interp,
                              axis=0, name='depth')
        return SoundSpeedProfile(
            depths=np.array([float(dv)]), data=np.array([[float(c)]]),
            ranges=None, shape=self.shape, data_sources=self.data_sources)

    @property
    def value(self) -> float:
        """The single sound speed this profile represents, when unambiguous.

        Valid for an isovelocity profile (every sample equal) or one
        collapsed to a single ``(depth, range)`` cell via
        ``at(depth=, range=)``. Raises if the profile actually varies."""
        if self.data.size > 1 and np.ptp(self.data) > 0:
            raise ConfigurationError(
                f"SoundSpeedProfile.value: profile varies (shape "
                f"{self.data.shape}); slice with at(depth=, range=) first"
            )
        return float(self.data.flat[0])

    def copy(self) -> 'SoundSpeedProfile':
        """Deep copy (symmetric with the other carriers)."""
        return _copy.deepcopy(self)

    def collapse(self, method: str = 'r0') -> 'SoundSpeedProfile':
        """Collapse a 2-D profile to 1-D using ``method``.

        Returns ``self`` (not a copy) when the profile is already 1-D.

        Methods
        -------
        ``'r0'``     : keep the range-0 column.
        ``'mean'``   : depth-wise mean across all ranges.
        ``'median'`` : depth-wise median across all ranges.
        ``'rmax'``   : keep the last (deepest range) column.
        """
        if not self.is_range_dependent:
            return self
        if method == 'r0':
            col = self.data[:, 0]
        elif method == 'rmax':
            col = self.data[:, -1]
        elif method == 'mean':
            col = self.data.mean(axis=1)
        elif method == 'median':
            col = np.median(self.data, axis=1)
        else:
            raise ConfigurationError(
                f"SoundSpeedProfile.collapse: unknown method={method!r}; "
                "valid: 'r0', 'rmax', 'mean', 'median'"
            )
        return SoundSpeedProfile(
            depths=self.depths.copy(),
            data=col.reshape(-1, 1),
            ranges=None,
            shape=self.shape,
            data_sources=self.data_sources,
        )

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
            return SoundSpeedProfile(
                depths=snapped_depths,
                data=self.data.copy(),
                ranges=(self.ranges.copy() if self.ranges is not None
                        else None),
                shape=self.shape,
                data_sources=self.data_sources,
            )
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
            new_data = np.vstack([self.data, self.data[-1:, :]])
        else:
            keep = self.depths < depth_max
            kept_depths = self.depths[keep]
            kept_data = self.data[keep]
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
                    np.interp(depth_max, self.depths, self.data[:, j])
                    for j in range(self.data.shape[1])
                ])
                new_depths = np.append(kept_depths, depth_max)
                new_data = np.vstack([kept_data, interp_row[None, :]])
        return SoundSpeedProfile(
            depths=new_depths,
            data=new_data,
            ranges=(self.ranges.copy() if self.ranges is not None else None),
            shape=self.shape,
            data_sources=self.data_sources,
        )

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
        * ``(depth, c)`` pairs — linear profile via :meth:`from_pairs`.
        * a :class:`SoundSpeedProfile` — returned as-is (by reference).

        ``None`` policy: an isovelocity 1500 m/s water column — a usable
        default profile, since every environment has *some* sound speed.

        ``depth_max`` (m) sets the column extent for the isovelocity cases.
        """
        if value is None:
            return cls.from_isovelocity(depth_max, DEFAULT_SOUND_SPEED)
        if isinstance(value, SoundSpeedProfile):
            return value
        # A 0-d ndarray is a scalar in every respect except ``isinstance``,
        # so both the bool guard and the numeric branch have to see through
        # it — otherwise ``np.array(1500.0)`` reaches ``from_pairs`` and the
        # error names an ``(N, 2)`` shape the caller never asked for, and
        # ``np.array(True)`` walks past the bool guard entirely.
        # ``Bathymetry.coerce`` admits the same spelling.
        zero_d = isinstance(value, np.ndarray) and value.ndim == 0
        if (isinstance(value, (bool, np.bool_))
                or (zero_d and value.dtype == np.bool_)):
            raise ConfigurationError(
                f"Environment: ssp={value!r} is a bool, not a sound speed — "
                f"as a scalar it would mean a {float(value):g} m/s ocean."
            )
        if isinstance(value, (int, float, np.integer, np.floating)):
            return cls.from_isovelocity(depth_max, float(value))
        if zero_d:
            # Guarded because a 0-d array can hold a string or an object,
            # which ``float`` refuses with an untyped error.
            try:
                sound_speed = float(value)
            except (TypeError, ValueError):
                raise ConfigurationError(
                    f"Environment: ssp must be a scalar (m/s), a list of "
                    f"(depth, sound_speed) pairs, or a SoundSpeedProfile; got "
                    f"a 0-d array of dtype {value.dtype}."
                ) from None
            return cls.from_isovelocity(depth_max, sound_speed)
        if isinstance(value, (list, tuple, np.ndarray)):
            return cls.from_pairs(value)
        raise ConfigurationError(
            f"Environment: ssp must be a scalar (m/s), a list of (depth, "
            f"sound_speed) pairs, or a SoundSpeedProfile; got "
            f"{type(value).__name__}"
        )

    @classmethod
    def from_isovelocity(
        cls, depth_max: float, sound_speed: float = DEFAULT_SOUND_SPEED
    ) -> 'SoundSpeedProfile':
        """Constant-``sound_speed`` (m/s) profile spanning 0 to ``depth_max`` (m)."""
        return cls(
            depths=np.array([0.0, float(depth_max)]),
            data=np.full((2, 1), float(sound_speed)),
            ranges=None,
            shape='isovelocity',
        )

    @classmethod
    def from_pairs(
        cls,
        pairs: Union[List[Tuple[float, float]], np.ndarray],
        shape: str = 'measured',
    ) -> 'SoundSpeedProfile':
        """Build a 1-D profile from ``[(depth, c), …]`` pairs.

        ``shape`` is informational metadata (``'measured'`` default);
        see :class:`SoundSpeedProfile`. The model's ``interp_ssp`` kwarg
        drives the sample-connection scheme.
        """
        arr = np.asarray(pairs, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ConfigurationError(
                f"SoundSpeedProfile.from_pairs: pairs must have shape (N, 2) "
                f"as (depth, sound_speed); got shape {arr.shape}"
            )
        return cls(
            depths=arr[:, 0],
            data=arr[:, 1].reshape(-1, 1),
            ranges=None,
            shape=shape,
        )

    @classmethod
    def from_2d(
        cls,
        depths: np.ndarray,
        ranges: np.ndarray,
        matrix: np.ndarray,
        shape: str = 'measured',
    ) -> 'SoundSpeedProfile':
        """Build a 2-D profile from a depth axis, range axis (metres),
        and ``c(depth, range)`` matrix of shape ``(n_depth, n_range)``.

        For Bellhop, pair with ``Bellhop(interp_ssp='quad')`` to enable
        the external ``.ssp`` (quad) file format.
        """
        return cls(
            depths=np.asarray(depths, dtype=float),
            data=np.asarray(matrix, dtype=float),
            ranges=np.asarray(ranges, dtype=float),
            shape=shape,
        )

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
            data=c.reshape(-1, 1),
            ranges=None,
            shape='munk',
        )

    @classmethod
    def from_mackenzie(
        cls,
        depths: np.ndarray,
        temperature_c: np.ndarray,
        salinity_psu: np.ndarray,
    ) -> 'SoundSpeedProfile':
        """Build a profile from in-situ ``T(z)`` and ``S(z)`` via Mackenzie's
        nine-term seawater sound-speed equation.

        ``depths``, ``temperature_c``, ``salinity_psu`` must be 1-D arrays
        of equal length sampled at the same depth grid. Use
        ``np.full_like(depths, T_const)`` if the column is isothermal/
        isohaline. Valid range: ``T ∈ [−2, 30] °C``,
        ``S ∈ [25, 40] PSU``, ``z ∈ [0, 8000] m`` (Mackenzie 1981).
        """
        from uacpy.core.acoustics import soundspeed
        z = np.asarray(depths, dtype=float).ravel()
        T = np.asarray(temperature_c, dtype=float).ravel()
        S = np.asarray(salinity_psu, dtype=float).ravel()
        if not (T.shape == S.shape == z.shape):
            raise ConfigurationError(
                "from_mackenzie: depths, temperature_c, salinity_psu must "
                f"share shape; got {z.shape}, {T.shape}, {S.shape}"
            )
        c = soundspeed(temperature=T, salinity=S, depth=z)
        return cls(
            depths=z, data=np.asarray(c).reshape(-1, 1),
            ranges=None,
        )


def generate_sea_surface(
    max_range: float,
    wind_speed_mps: float = 10.0,
    n_points: int = 500,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a random sea surface realization from the Pierson-Moskowitz spectrum.

    Parameters
    ----------
    max_range : float
        Maximum range in meters.
    wind_speed_mps : float
        Wind speed at 19.5 m height in m/s (Pierson-Moskowitz
        convention). The fully developed significant wave height is
        Hs = 4*sqrt(alpha/beta)*U^2/(2g) = 0.021*U^2:
        - 5 m/s: Hs ~ 0.5 m
        - 10 m/s: Hs ~ 2.1 m
        - 15 m/s: Hs ~ 4.8 m
        - 20 m/s: Hs ~ 8.5 m
    n_points : int
        Number of range points in the output altimetry array.
    seed : int, optional
        Random seed for reproducibility.

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
    if not np.isfinite(max_range) or max_range <= 0:
        raise ConfigurationError(
            f"generate_sea_surface: max_range must be a positive distance (m); "
            f"got {max_range}."
        )
    if not np.isfinite(wind_speed_mps) or wind_speed_mps <= 0:
        raise ConfigurationError(
            f"generate_sea_surface: wind_speed_mps must be a positive m/s value; "
            f"got {wind_speed_mps}."
        )
    if n_points < 2:
        raise ConfigurationError(
            f"generate_sea_surface: n_points must be >= 2; got {n_points}."
        )
    g = 9.81
    rng = np.random.default_rng(seed)

    ranges = np.linspace(0, max_range, n_points)
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
    omega_p = g / wind_speed_mps  # peak angular frequency
    S_omega = (alpha_pm * g**2 / omega**5) * np.exp(-beta_pm * (omega_p / omega)**4)

    # Convert to spatial spectrum S(k) via S(k) = S(omega) * domega/dk
    # with k in cycles/m: omega = sqrt(2*pi*g*k) so domega/dk = pi*g/omega
    domega_dk = np.pi * g / omega
    S_k = S_omega * domega_dk

    # The variance sits around the spectral peak, so a grid whose Nyquist
    # wavenumber falls at or below it captures only the tail and returns a
    # surface far flatter than the Pierson-Moskowitz Hs for this wind.
    k_peak = omega_p ** 2 / (2.0 * np.pi * g)          # cycles/m
    k_nyquist = k[-1]
    if k_nyquist < 2.0 * k_peak:
        warnings.warn(
            f"generate_sea_surface: the range grid resolves wavenumbers only "
            f"to {k_nyquist:.4g} cycles/m, below 2x the Pierson-Moskowitz peak "
            f"at {k_peak:.4g} cycles/m for wind_speed_mps={wind_speed_mps:g}. The "
            f"realisation captures only the spectral tail and its significant "
            f"wave height will fall short of the fully developed "
            f"{0.021 * wind_speed_mps ** 2:.2f} m. Increase n_points (or "
            f"shorten max_range) to resolve the peak.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
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
