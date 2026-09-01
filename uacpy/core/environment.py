"""Top-level ocean :class:`Environment` carrier.

The seafloor/boundary classes and the sound-speed-profile carrier live in
:mod:`uacpy.core.bottom` and :mod:`uacpy.core.ssp`; they are re-exported here,
so ``from uacpy.core.environment import BoundaryProperties`` (etc.) is a valid
import path for every carrier an :class:`Environment` holds.
"""

import copy as _copy
import warnings
import numpy as np
from typing import TYPE_CHECKING, Union, List, Tuple, Optional

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._carrier_validate import _sanitize_title, _dedupe_provenance
from uacpy.core.bottom import (
    SedimentLayer, BoundaryProperties, SeabedColumn, Bottom,
)
from uacpy.core.ssp import SoundSpeedProfile, generate_sea_surface
from uacpy.core.bathymetry import Bathymetry
from uacpy.core.altimetry import Altimetry
from uacpy.core.surface import Surface

if TYPE_CHECKING:
    from uacpy.core.absorption import Absorption  # noqa: F401


def _coerce_coordinate(value, label):
    """Validate a ``(lat, lon)`` pair (decimal degrees, WGS84) → ``(float,
    float)``. Used for the optional geolocation an :class:`Environment`
    carries (e.g. stamped by ``uacpy.data.fetch_environment``).

    Longitude accepts either sign convention up to one full wrap
    (``|lon| <= 360``); a value already in ``[-180, 180)`` is stored exactly
    as given, anything else is wrapped into that interval, so e.g. 250°E
    stores as -110."""
    try:
        lat, lon = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError, KeyError):
        raise ConfigurationError(
            f"Environment: {label} must be a (lat, lon) pair; got {value!r}.")
    if not (np.isfinite(lat) and np.isfinite(lon)):
        raise ConfigurationError(
            f"Environment: {label} must be finite; got {value!r}.")
    if not -90.0 <= lat <= 90.0:
        raise ConfigurationError(
            f"Environment: {label} latitude must be in [-90, 90]; got {lat}.")
    if not -360.0 <= lon <= 360.0:
        raise ConfigurationError(
            f"Environment: {label} longitude must be in [-360, 360]; "
            f"got {lon}.")
    if not -180.0 <= lon < 180.0:
        # Wrap only out-of-interval values: the modulo arithmetic is not
        # bit-exact (-6.2 would come back -6.199999999999989), and a stored
        # coordinate must compare equal to the pair the caller passed.
        lon = ((lon + 180.0) % 360.0) - 180.0
    return (lat, lon)


def _transect_midpoint(start, end):
    """Midpoint ``(lat, lon)`` of a transect — simple mean, with a longitude
    wrap so it stays correct across the antimeridian. Good enough as a
    representative ``location`` label for the short transects acoustics uses."""
    (la0, lo0), (la1, lo1) = start, end
    if lo1 - lo0 > 180.0:
        lo1 -= 360.0
    elif lo1 - lo0 < -180.0:
        lo1 += 360.0
    lon = (0.5 * (lo0 + lo1) + 180.0) % 360.0 - 180.0
    return (0.5 * (la0 + la1), lon)


def _coerce_date(value):
    """Validate the optional time the env represents → a ``datetime.date``
    (ISO ``'YYYY-MM-DD'`` strings are parsed; ``None`` passes through)."""
    import datetime as _dt
    if value is None or isinstance(value, _dt.date):
        return value
    if isinstance(value, str):
        try:
            return _dt.date.fromisoformat(value)
        except ValueError:
            raise ConfigurationError(
                f"Environment: date must be an ISO 'YYYY-MM-DD' string or a "
                f"datetime.date; got {value!r}.")
    raise ConfigurationError(
        f"Environment: date must be a 'YYYY-MM-DD' string, a datetime.date, "
        f"or None; got {type(value).__name__}.")


class Environment:
    """
    Ocean environment definition.

    Combines a sound-speed profile, bathymetry, optional surface
    altimetry, and surface/bottom acoustic properties into the input
    object every propagation model consumes.

    Parameters
    ----------
    bathymetry : float or array-like
        Either a scalar water depth in metres (flat bottom), or a
        range-dependent bathymetry as ``[(range, depth), …]``.
        The maximum depth in this argument defines the water column
        extent; ``env.depth`` exposes it as a read-only property.
    ssp : scalar (m/s), list of (depth, c_m_s) pairs, or SoundSpeedProfile, optional
        Sound-speed profile.

        * Scalar — isovelocity at the given speed.
        * List/array of ``(depth, sound_speed)`` pairs — linear-interp
          ``SoundSpeedProfile`` built via :meth:`SoundSpeedProfile.from_pairs`.
        * ``SoundSpeedProfile`` instance — used as-is (1-D or 2-D).
        * ``None`` (default) — isovelocity at 1500 m/s.
    altimetry : Altimetry or array-like, optional
        Surface altimetry as ``[(range, height_m), …]`` (height
        positive up), or an :class:`Altimetry`. Default ``None`` (flat
        surface).
    bottom : Bottom, SeabedColumn, BoundaryProperties, float, or str, optional
        Seabed. Coerced to a :class:`Bottom`: a scalar is a half-space sound
        speed (``bottom=1800``), a string is a material preset
        (``bottom='sand'``), and a ``BoundaryProperties`` / ``SeabedColumn`` /
        ``Bottom`` is used directly. Default is a fluid sand-like half-space
        (``sound_speed=1600`` m/s, ``density=1.5`` g/cm³,
        ``attenuation=0.5`` dB/wavelength). For a perfectly reflecting bottom,
        pass ``BoundaryProperties(acoustic_type='rigid')``.
    surface : Surface, BoundaryProperties, or list, optional
        Top boundary. Coerced to a :class:`Surface`: a single
        ``BoundaryProperties`` is a uniform surface, a
        ``[(range_m, BoundaryProperties), …]`` list is a range-dependent one
        (e.g. a marginal ice zone), and a ``Surface`` is used directly.
        Default vacuum (pressure release).
    absorption : Absorption, optional
        Water-column volume-absorption model — one of
        :class:`uacpy.core.absorption.Thorp`,
        :class:`uacpy.core.absorption.FrancoisGarrison`,
        :class:`uacpy.core.absorption.Biological`, or
        :class:`uacpy.core.absorption.ConstantAbsorption`. Default ``None``
        (no volume absorption). Models inspect this field to set
        ``TopOpt`` position 4 and write the supporting per-formula lines.
    name : str, keyword-only
        Environment identifier. Default ``'unnamed'``.
    location : (float, float), keyword-only
        Representative site as ``(lat, lon)`` in WGS84 decimal degrees.
        Default ``None``; falls back to the ``transect`` midpoint when only a
        transect is given. Stamped by ``uacpy.data.fetch_environment``.
    transect : ((float, float), (float, float)), keyword-only
        Great-circle path as ``((lat, lon) start, (lat, lon) end)`` in WGS84
        decimal degrees, for a range-dependent environment fetched along a
        track. Default ``None``.
    date : str or datetime.date, keyword-only
        The time this environment represents — an ISO ``'YYYY-MM-DD'`` string
        or a :class:`datetime.date`. Default ``None``.

    Examples
    --------
    Isovelocity:

    >>> env = Environment(name='shallow', bathymetry=100, ssp=1500)

    Linear SSP:

    >>> env = Environment(
    ...     name='test', bathymetry=200,
    ...     ssp=SoundSpeedProfile.from_pairs(
    ...         [(0, 1520), (200, 1480)]),
    ... )

    Munk:

    >>> env = Environment(
    ...     name='deep', bathymetry=5000,
    ...     ssp=SoundSpeedProfile.from_munk(5000),
    ... )

    Range-dependent bathymetry:

    >>> env = Environment(
    ...     name='wedge', bathymetry=[(0, 100), (10000, 200)],
    ... )
    """

    def __init__(
        self,
        bathymetry: Union[float, List[Tuple[float, float]], np.ndarray],
        ssp: Optional[Union[
            float, int,
            List[Tuple[float, float]],
            np.ndarray,
            SoundSpeedProfile,
        ]] = None,
        altimetry: Optional[Union[
            Altimetry, List[Tuple[float, float]], np.ndarray,
        ]] = None,
        bottom: Optional[Union[
            Bottom, SeabedColumn, BoundaryProperties, float, str,
        ]] = None,
        surface: Optional[Union[
            Surface, BoundaryProperties,
            List[Tuple[float, BoundaryProperties]],
        ]] = None,
        absorption: Optional['Absorption'] = None,
        *,
        name: str = 'unnamed',
        location: Optional[Tuple[float, float]] = None,
        transect: Optional[Tuple[Tuple[float, float],
                                 Tuple[float, float]]] = None,
        date=None,
    ):
        from uacpy.core.absorption import Absorption
        if absorption is not None and not isinstance(absorption, Absorption):
            raise ConfigurationError(
                f"Environment: absorption must be an Absorption subclass "
                f"(Thorp / FrancoisGarrison / Biological / ConstantAbsorption); "
                f"got {type(absorption).__name__}"
            )
        self.absorption = absorption
        self.name = _sanitize_title(name)

        # Optional geolocation (WGS84 decimal degrees) and the time the env
        # represents. ``transect`` is the ((lat, lon) start, (lat, lon) end)
        # great-circle path for a range-dependent env; ``location`` is the
        # representative site point — an explicit value if given, else the
        # transect midpoint. Stamped by ``uacpy.data.fetch_environment``;
        # ``None`` for a hand-built env. All survive ``env.copy()`` (deepcopy).
        if transect is None:
            self.transect = None
        else:
            try:
                start, end = transect
            except (TypeError, ValueError):
                raise ConfigurationError(
                    "Environment: transect must be a ((lat, lon) start, "
                    f"(lat, lon) end) pair; got {transect!r}.")
            self.transect = (_coerce_coordinate(start, "transect start"),
                             _coerce_coordinate(end, "transect end"))
        if location is not None:
            self.location = _coerce_coordinate(location, "location")
        elif self.transect is not None:
            self.location = _transect_midpoint(*self.transect)
        else:
            self.location = None
        self.date = _coerce_date(date)

        # Provenance: the data sources used to build this env. Empty for a
        # hand-built env; ``uacpy.data.fetch_environment`` overwrites it with
        # the catalogue entries it fetched. Declared here so ``env.data_sources``
        # is always a valid (possibly empty) iterable — never an AttributeError.
        self.data_sources = ()

        # Bathymetry is a first-class carrier (seafloor depth vs range),
        # mirroring env.ssp; it validates in its own __post_init__.
        self.bathymetry = Bathymetry.coerce(bathymetry)

        max_bathy_depth = self.bathymetry.depth

        # Carrier instances (ssp / surface / bottom) are stored by reference,
        # not deep-copied: every model copies the whole env (``env.copy()``)
        # before mutating any of them, so the env never mutates a caller's
        # carrier. Do not mutate ``env.ssp`` / ``env.bottom`` / ``env.surface``
        # in place without an ``env.copy()`` first.
        self.ssp = SoundSpeedProfile.coerce(ssp, depth_max=max_bathy_depth)

        # Altimetry is a first-class carrier (surface height vs range), the
        # top-surface analogue of env.bathymetry; ``None`` = flat z = 0.
        self.altimetry = Altimetry.coerce(altimetry)

        # Extend only, never truncate: the profile has to span the water
        # column, but one that already reaches past the seabed is kept whole
        # (each writer calls ``extend_to`` again with its own deep-end depth).
        if max_bathy_depth > self.ssp.depths[-1]:
            self.ssp = self.ssp.extend_to(max_bathy_depth)

        # Surface is a first-class carrier (top boundary vs range), the
        # top-properties analogue of env.bottom; a single BoundaryProperties
        # is coerced to a uniform one-node Surface.
        self.surface = Surface.coerce(surface)

        self.bottom = self._coerce_bottom(bottom)

        # Harmonised provenance: the union of each carrier's own ``data_sources``
        # (every fetched carrier carries dated/located ``DataProvenance`` records;
        # literal carriers carry none), de-duplicated by source id in axis order
        # bathymetry → ssp → bottom → surface → altimetry. ``fetch_environment``
        # may refine this (e.g. fall back to a bare catalogue id for an
        # un-stamped layer).
        self.data_sources = self._aggregate_data_sources()

    def _aggregate_data_sources(self) -> tuple:
        """Union of the carriers' ``data_sources`` (dedup by source id, axis
        order). The single home for ``env.data_sources``, mirrored per-carrier
        by ``Bottom``/``Surface``/``SeabedColumn``."""
        return _dedupe_provenance((self.bathymetry, self.ssp, self.bottom,
                                   self.surface, self.altimetry))

    @staticmethod
    def _coerce_bottom(bottom) -> Bottom:
        """Coerce ``bottom=`` into a :class:`Bottom`, mirroring ``ssp=``:
        scalar cp, preset name, ``BoundaryProperties``, ``SeabedColumn`` or
        ``Bottom`` (``None`` → the default half-space)."""
        if bottom is None:
            return Bottom.from_halfspace(BoundaryProperties(
                acoustic_type='half-space', density=1.5,
                sound_speed=1600.0, attenuation=0.5))
        if isinstance(bottom, Bottom):
            return bottom
        if isinstance(bottom, SeabedColumn):
            return Bottom.from_column(bottom)
        if isinstance(bottom, BoundaryProperties):
            return Bottom.from_halfspace(bottom)
        # A 0-d ndarray is a scalar in every respect except ``isinstance``, so
        # both the bool guard and the numeric branch see through it, the way
        # ``ssp=`` and ``bathymetry=`` do — the docstring promises ``ssp=``'s
        # spellings, and ``np.array(1600.0)`` used to fall through to the
        # closing type error instead.
        zero_d = isinstance(bottom, np.ndarray) and bottom.ndim == 0
        if (isinstance(bottom, (bool, np.bool_))
                or (zero_d and bottom.dtype == np.bool_)):
            raise ConfigurationError(
                f"Environment: bottom={bottom!r} is a bool, not a scalar "
                f"sound speed — as a scalar it would mean a {float(bottom):g} "
                f"m/s half-space."
            )
        if zero_d and np.issubdtype(bottom.dtype, np.number):
            bottom = bottom.item()
        if isinstance(bottom, (int, float, np.integer, np.floating)):
            # Explicit type: a scalar always means "half-space at this cp" —
            # never let inference see it (a bare 1600.0 equals the resolved
            # default and would otherwise be indistinguishable from unset).
            return Bottom.from_halfspace(BoundaryProperties(
                acoustic_type='half-space', sound_speed=float(bottom)))
        if isinstance(bottom, str):
            return Bottom.from_halfspace(BoundaryProperties.from_preset(bottom))
        raise ConfigurationError(
            "Environment: bottom must be a Bottom, SeabedColumn, "
            "BoundaryProperties, a scalar sound speed (m/s), or a material "
            f"preset name; got {type(bottom).__name__}")

    def plot(self, ax=None, **kwargs):
        """Plot the water column + seafloor cross-section.

        Water column (SSP) + seafloor cross-section, with optional
        ``source=`` / ``receiver=`` markers. The carrier counterpart of
        :meth:`Result.plot` — any uacpy object you plot on its own has
        ``.plot()``. ``ax`` draws into an existing Axes, spelled the way every
        other uacpy plot method spells it; the remaining ``kwargs`` are
        forwarded to the renderer."""
        # Deferred into the body: ``uacpy.visualization`` imports
        # ``uacpy.core`` at module scope, so this line at file scope makes
        # ``import uacpy`` raise ImportError. docs/DEV.md section 7 records
        # the inversion.
        from uacpy.visualization.plots.environment import _plot_environment
        return _plot_environment(self, ax=ax, **kwargs)

    @property
    def depth(self) -> float:
        """Maximum water depth in metres (derived from bathymetry)."""
        return self.bathymetry.depth

    @property
    def max_range(self) -> float:
        """Range extent in metres across the environment's range-dependent axes.

        The largest range coordinate carried by the bathymetry, SSP, bottom,
        surface or altimetry; ``0.0`` for a range-independent environment.
        Derived (read-only), symmetric with :attr:`depth`. For an environment
        fetched along a transect this equals the transect's great-circle
        length, so it sizes a receiver range grid without recomputing the
        geodesic.
        """
        # Any carrier with a ranged axis contributes, including a single-node
        # one (ranges=[r] is a coordinate at range r even though
        # ``is_range_dependent`` — a >1-node test — is False for it).
        extent = self.bathymetry.range_max
        if self.ssp.ranges is not None:
            extent = max(extent, float(self.ssp.ranges[-1]))
        if self.bottom.ranges is not None:
            extent = max(extent, float(self.bottom.ranges[-1]))
        extent = max(extent, self.surface.range_max)
        if self.altimetry is not None:
            extent = max(extent, self.altimetry.range_max)
        return extent

    def get_sound_speed(
        self, depth: Union[float, np.ndarray], range: float = 0.0
    ) -> np.ndarray:
        """Sound speed at given depth(s), at ``range`` for 2-D profiles.

        Always **linear** in depth (``np.interp``); depths outside the profile
        are constant-extrapolated to the nearest endpoint and emit a
        ``UserWarning`` (the value is held flat, not fabricated).
        """
        slice_1d = (self.ssp.eval(range=range)
                    if self.ssp.is_range_dependent else self.ssp)
        d = np.atleast_1d(depth)
        z = slice_1d.depths
        if d.size and (np.any(d < z[0]) or np.any(d > z[-1])):
            warnings.warn(
                f"get_sound_speed: depth(s) outside the profile "
                f"[{float(z[0]):.1f}, {float(z[-1]):.1f}] m were "
                f"constant-extrapolated to the nearest endpoint.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
        return np.interp(d, z, slice_1d.data[:, 0])

    @property
    def has_range_dependent_bathymetry(self) -> bool:
        """``True`` iff the seafloor depth actually varies with range.

        A multi-point bathymetry whose depths are all equal (flat) counts as
        range-independent. (Seafloor depth at a range: ``env.bathymetry.at`` /
        ``eval`` / ``isel``.)
        """
        return self.bathymetry.is_range_dependent

    @property
    def has_range_dependent_ssp(self) -> bool:
        return self.ssp.is_range_dependent

    @property
    def has_range_dependent_bottom(self) -> bool:
        """``True`` for a range-dependent *half-space* bottom (no layers)."""
        return self.bottom.is_range_dependent and not self.bottom.is_layered

    @property
    def has_layered_bottom(self) -> bool:
        """``True`` for a range-*independent* layered bottom."""
        return self.bottom.is_layered and not self.bottom.is_range_dependent

    @property
    def has_range_dependent_layered_bottom(self) -> bool:
        """``True`` for a bottom that varies with range *and* has layers."""
        return self.bottom.is_range_dependent and self.bottom.is_layered

    @property
    def has_elastic_bottom(self) -> bool:
        """``True`` iff any layer or half-space of ``self.bottom`` has shear."""
        return self.bottom.is_elastic

    @property
    def has_elastic_surface(self) -> bool:
        """``True`` iff the surface carries non-zero shear at any range."""
        return self.surface.is_elastic

    @property
    def is_range_dependent(self) -> bool:
        """True when bathymetry, SSP, bottom or surface is range-dependent,
        under each carrier's own meaning of the term: for bathymetry the
        *values* must vary with range (a flat multi-point bathymetry does not
        count), while for ssp / bottom / surface the test is structural (more
        than one node on a ranged axis, identical or not).

        Altimetry is not consulted: a non-flat sea surface varies with range
        by nature, but it is boundary geometry that Bellhop — the only model
        reading altimetry — traces directly from its ``.ati`` file, so it
        never triggers the segmented-profile machinery this flag selects for
        the four carriers above."""
        return (
            self.has_range_dependent_bathymetry
            or self.ssp.is_range_dependent
            or self.bottom.is_range_dependent
            or self.surface.is_range_dependent
        )

    def __repr__(self) -> str:
        range_dep = "range-dep" if self.is_range_dependent else "range-indep"
        geo = ""
        if self.transect is not None:
            (la0, lo0), (la1, lo1) = self.transect
            geo = f", transect=({la0:.3f},{lo0:.3f})→({la1:.3f},{lo1:.3f})"
        elif self.location is not None:
            geo = f", location=({self.location[0]:.3f},{self.location[1]:.3f})"
        if self.date is not None:
            geo += f", date={self.date.isoformat()}"
        return (f"Environment(name='{self.name}', depth={self.depth:.1f}m, "
                f"ssp='{self.ssp.shape}', {range_dep}{geo})")

    def get_representative_depth(self, method: str = 'max') -> float:
        """
        Get representative depth from range-dependent bathymetry

        For models that don't support range-dependent environments,
        this provides a single representative depth value.

        Parameters
        ----------
        method : str, optional
            Method for computing representative value:
            - 'max': Maximum depth (deepest, default — matches the
              project-wide ``collapse={'bathymetry': 'max'}``)
            - 'median': Median depth
            - 'mean': Mean depth
            - 'min': Minimum depth (shallowest)
            - 'initial': Initial depth at range=0

        Returns
        -------
        depth : float
            Representative depth in meters

        Examples
        --------
        >>> env = Environment(name='slope',
        ...                   bathymetry=[(0, 100), (5000, 200), (10000, 300)])
        >>> env.get_representative_depth('median')
        200.0
        >>> env.get_representative_depth('mean')
        200.0
        >>> env.get_representative_depth('initial')
        100.0
        """
        depths = self.bathymetry.depths

        if method == 'median':
            return float(np.median(depths))
        elif method == 'mean':
            return float(np.mean(depths))
        elif method == 'min':
            return float(np.min(depths))
        elif method == 'max':
            return float(np.max(depths))
        elif method == 'initial':
            return float(depths[0])
        else:
            raise ConfigurationError(
                f"Environment.get_representative_depth: unknown method={method!r}; "
                "valid: 'max', 'median', 'mean', 'min', 'initial'"
            )

    def copy(self):
        """Deep copy of the environment.

        Uses ``copy.deepcopy`` so every field — including ``ssp``,
        ``altimetry``, and ``bottom`` — is duplicated without aliasing
        back to the original instance.
        """
        return _copy.deepcopy(self)


__all__ = [
    'Environment',
    'SedimentLayer', 'BoundaryProperties', 'SeabedColumn', 'Bottom',
    'SoundSpeedProfile', 'generate_sea_surface', 'Bathymetry', 'Altimetry',
    'Surface',
]
