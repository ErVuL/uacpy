"""Top-level ocean :class:`Environment` carrier.

The seafloor/boundary classes and the sound-speed-profile carrier live in
:mod:`uacpy.core.bottom` and :mod:`uacpy.core.ssp`; they are re-exported here,
so ``from uacpy.core.environment import BoundaryProperties`` (etc.) is a valid
import path for every carrier an :class:`Environment` holds.
"""

import copy as _copy
import datetime
from dataclasses import KW_ONLY
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np

import warnings

from uacpy.core.exceptions import (
    ConfigurationError, ProvenanceWarning, ValidityWarning)
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._plotting import plotter
from uacpy.core._validate import sanitize_title
from uacpy.core._provenance import (
    coerce_data_sources, dedupe_provenance, dedupe_records)
from uacpy.core._carrier import (
    DeepCopyMixin, _under_construction, carrier)
from uacpy.core._export import CarrierExport
from uacpy.core._repr import build, extent, qty
from uacpy.core.boundary import SedimentLayer, BoundaryProperties
from uacpy.core.bottom import SeabedColumn, Bottom
from uacpy.core.ssp import SoundSpeedProfile
from uacpy.core.bathymetry import Bathymetry
from uacpy.core.altimetry import Altimetry
from uacpy.core.surface import Surface
from uacpy.core.absorption import (
    Absorption, AbsorptionCoefficient, FrancoisGarrison, _TabulatedAbsorption,
)
from uacpy.core.constants import (
    DEFAULT_WATER_DENSITY_G_CM3, REFERENCE_SALINITY_PSU,
    REFERENCE_TEMPERATURE_C, WATER_DENSITY_MAX_G_CM3,
    WATER_DENSITY_MIN_G_CM3,
)
from uacpy.core.geo import as_coordinate, great_circle_midpoint, parse_date


def _check_absorption(value):
    """The law ``value`` stands for: an :class:`Absorption` law as it is, a
    measured :class:`AbsorptionCoefficient` (``model=None``) as a tabulated
    law, ``None`` as it is. A table a law computed is refused: the law is
    the one object, and the environment takes it."""
    if isinstance(value, AbsorptionCoefficient):
        if value.model is not None:
            raise ConfigurationError(
                f"Environment: absorption is the table a {value.model!r} law "
                f"computed (law.table(f)); a law is one object, and the "
                f"environment takes the law, not its samples.",
                remediation="Pass the law itself: Environment(absorption="
                            "<law>), e.g. Thorp() or FrancoisGarrison(...). "
                            "A measured alpha(f, z) is an "
                            "AbsorptionCoefficient with model=None.")
        return _TabulatedAbsorption(measured=value)
    if value is not None and not isinstance(value, Absorption):
        raise ConfigurationError(
            f"Environment: absorption must be an Absorption law "
            f"(Thorp / FrancoisGarrison / Biological / ConstantAbsorption) "
            f"or a measured AbsorptionCoefficient (model=None); got "
            f"{type(value).__name__}."
        )
    return value


def _warn_if_table_misses_water(absorption, depth: float) -> None:
    """Say so when a tabulated law's rows do not reach the whole water
    column ``0..depth``: the rows beyond them are held, not extrapolated."""
    if not isinstance(absorption, _TabulatedAbsorption):
        return
    gaps = absorption._water_past_rows(0.0, float(depth))
    if gaps:
        warnings.warn(
            f"Environment: the measured absorption table does not reach "
            f"the whole water column: {'; '.join(gaps)}, "
            f"held there rather than extrapolated.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)


#: How far (°C, psu) a water's temperature or salinity stands from the
#: reference water before an absorption on the reference water is called
#: clearly wrong for it. A threshold choice: 2 °C or 1 psu moves the
#: Francois-Garrison absorption by about 10 % at 10 kHz.
_REFERENCE_WATER_GAP_C = 2.0
_REFERENCE_WATER_GAP_PSU = 1.0


def _warn_if_reference_water_replaces(old, new) -> None:
    """Say so when a :class:`FrancoisGarrison` on the reference water's
    temperature and salinity replaces the environment's own water — the
    Francois-Garrison law it held, a measured profile or row — and the two
    clearly differ. Silent when the environment held no water to compare:
    the only T/S an environment carries is a Francois-Garrison law's, so the
    check cannot see the water behind a plain sound-speed profile (fetched
    or not), nor a reference law given to the constructor."""
    if not (isinstance(new, FrancoisGarrison) and not new.is_profile
            and isinstance(old, FrancoisGarrison)):
        return
    if not (new.temperature == REFERENCE_TEMPERATURE_C
            and new.salinity == REFERENCE_SALINITY_PSU):
        return
    t_gap = float(np.max(np.abs(np.asarray(old._values('temperature'),
                                           dtype=float)
                                - REFERENCE_TEMPERATURE_C)))
    s_gap = float(np.max(np.abs(np.asarray(old._values('salinity'),
                                           dtype=float)
                                - REFERENCE_SALINITY_PSU)))
    if t_gap <= _REFERENCE_WATER_GAP_C and s_gap <= _REFERENCE_WATER_GAP_PSU:
        return
    own = ', '.join(old._repr_bits())
    warnings.warn(
        f"Environment: absorption {new!r} is the reference water "
        f"({REFERENCE_TEMPERATURE_C:g} °C, {REFERENCE_SALINITY_PSU:g} psu), "
        f"and it replaces this environment's own water ({own}), up to "
        f"{t_gap:.3g} °C and {s_gap:.3g} psu away. Francois-Garrison follows "
        f"the temperature and salinity, so the run absorbs as in water this "
        f"environment does not hold. Keep the law it had, or build one from "
        f"the column: Environment(absorption=FrancoisGarrison("
        f"temperature=T, salinity=S, pH=8.0, depths=z)).",
        ProvenanceWarning, skip_file_prefixes=USER_FRAME_SKIP)


def _coerce_water_density(value) -> float:
    """The water density as a float in g/cm³ (``None`` is the package
    default); refused outside 0.9-1.1, where a kg/m³ slip lands."""
    if value is None:
        value = DEFAULT_WATER_DENSITY_G_CM3
    try:
        rho_w = float(value)
    except (TypeError, ValueError):
        raise ConfigurationError(
            f"Environment: water_density must be a number in g/cm³; "
            f"got {value!r}.")
    if not (WATER_DENSITY_MIN_G_CM3 <= rho_w <= WATER_DENSITY_MAX_G_CM3):
        raise ConfigurationError(
            f"Environment: water_density={rho_w:g} lies outside "
            f"{WATER_DENSITY_MIN_G_CM3:g}-{WATER_DENSITY_MAX_G_CM3:g} "
            f"g/cm³. The unit is g/cm³ (sea water is about 1.027); a "
            f"kg/m³ value has to be divided by 1000.")
    return rho_w


def _owned(coerced, given):
    """``coerced``, deep-copied when it is the caller's ``given`` object
    itself: an environment shares no carrier with its caller, so a later
    edit of the caller's object leaves the environment as it was built."""
    return _copy.deepcopy(coerced) if coerced is given else coerced


def _coerce_transect(value):
    """The transect as a pair of checked ``(lat, lon)`` coordinates, or
    ``None``."""
    if value is None:
        return None
    try:
        start, end = value
    except (TypeError, ValueError):
        raise ConfigurationError(
            "Environment: transect must be a ((lat, lon) start, "
            f"(lat, lon) end) pair; got {value!r}.")
    return (as_coordinate(start, label="Environment: transect start"),
            as_coordinate(end, label="Environment: transect end"))


def _coerce_location(value, transect):
    """The checked ``location``, else the ``transect``'s great-circle
    midpoint, else ``None``."""
    if value is not None:
        return as_coordinate(value, label="Environment: location")
    return None if transect is None else great_circle_midpoint(*transect)


def _spanning(ssp, depth):
    """``ssp`` extended down to ``depth`` when it ends above it. Extend only,
    never truncate: the profile has to span the water column, but one that
    already reaches past the seabed is kept whole (each writer calls
    ``extend_to`` again with its own deep-end depth)."""
    return ssp.extend_to(depth) if depth > ssp.depths[-1] else ssp


#: Each field an assignment checks alone: the value as the constructor
#: stores it. ``bathymetry``, ``ssp``, ``transect`` and ``location`` move
#: with another field and are handled in :meth:`Environment._assignment`.
_FIELD_COERCION = {
    'absorption': lambda v: _owned(_check_absorption(v), v),
    'water_density': _coerce_water_density,
    'name': sanitize_title,
    'date': lambda v: (None if v is None
                       else parse_date(v, label="Environment: date")),
    'extra_data_sources': lambda v: coerce_data_sources(
        v, "Environment extra_data_sources"),
    'altimetry': lambda v: _owned(Altimetry.coerce(v), v),
    'surface': lambda v: _owned(Surface.coerce(v), v),
    'bottom': lambda v: _owned(Bottom.coerce(v), v),
}


#: The unset value of a field whose default ``__post_init__`` resolves (the
#: isovelocity profile, the default seabed and surface, the package water
#: density). Typed ``Any`` so each field annotation states what the attribute
#: holds once the environment is built.
_RESOLVED_DEFAULT: Any = None


# eq=False: an environment compares by identity (a field-wise __eq__ over
# carriers holding ndarrays raises); repr=False keeps the one-line summary.
# The constructor keeps the input types the Parameters section documents; the
# field annotations state what each attribute holds once built.
@carrier(eq=False, repr=False, init_annotations=dict(
    bathymetry=Union[float, List[Tuple[float, float]], np.ndarray],
    ssp=Optional[Union[
        float, int,
        List[Tuple[float, float]],
        np.ndarray,
        SoundSpeedProfile,
    ]],
    altimetry=Optional[Union[
        Altimetry, List[Tuple[float, float]], np.ndarray,
    ]],
    bottom=Optional[Union[
        Bottom, SeabedColumn, BoundaryProperties, float, str,
    ]],
    surface=Optional[Union[
        Surface, BoundaryProperties,
        List[Tuple[float, BoundaryProperties]],
    ]],
    date=Optional[Union[str, datetime.date, np.datetime64]],
    water_density=Optional[float],
))
class Environment(DeepCopyMixin, CarrierExport):
    """
    Ocean environment definition.

    Combines a sound-speed profile, bathymetry, optional surface
    altimetry, and surface/bottom acoustic properties into the input
    object every propagation model consumes.

    The environment holds its own copy of every carrier it is given, so
    editing the caller's object afterwards leaves the environment as it was
    built. An assignment (``env.bathymetry = 500``) is checked and completed
    as the constructor checks and completes the same argument, copying only
    the assigned value: the profile is extended to span a deeper seafloor, a
    derived ``location`` follows a reassigned ``transect``, and a refused
    value leaves the environment as it was.

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
        * ``SoundSpeedProfile`` instance — copied (1-D or 2-D).
        * ``None`` (default) — isovelocity at 1500 m/s.

        A profile that ends above the deepest seafloor is extended to it
        (:meth:`SoundSpeedProfile.extend_to`); one that reaches past the
        seabed is kept whole.
    altimetry : Altimetry or array-like, optional
        Surface altimetry as ``[(range, height_m), …]`` (height
        positive up), or an :class:`Altimetry`. Default ``None`` (flat
        surface).
    bottom : Bottom, SeabedColumn, BoundaryProperties, float, or str, optional
        Seabed. Coerced to a :class:`Bottom`: a scalar is a half-space sound
        speed (``bottom=1800``), a string is a material preset
        (``bottom='sand'``), and a ``BoundaryProperties`` / ``SeabedColumn`` /
        ``Bottom`` is copied into one. Default is a generic fluid half-space
        (``sound_speed=1600`` m/s, ``density=1.5`` g/cm³,
        ``attenuation=0.5`` dB/wavelength) that matches no preset. For a perfectly reflecting bottom,
        pass ``BoundaryProperties(acoustic_type='rigid')``.
    surface : Surface, BoundaryProperties, or list, optional
        Top boundary. Coerced to a :class:`Surface`: a single
        ``BoundaryProperties`` is a uniform surface, a
        ``[(range_m, BoundaryProperties), …]`` list is a range-dependent one
        (e.g. a marginal ice zone), and a ``Surface`` is copied.
        Default vacuum (pressure release).
    absorption : Absorption or AbsorptionCoefficient, optional
        Water-column volume-absorption model — one of
        :class:`uacpy.core.absorption.Thorp`,
        :class:`uacpy.core.absorption.FrancoisGarrison` (one water row or a
        T/S profile), :class:`uacpy.core.absorption.Biological`, or
        :class:`uacpy.core.absorption.ConstantAbsorption` — or a measured
        α(f, z): an :class:`~uacpy.core.absorption.AbsorptionCoefficient`
        with ``model=None``, used as tabulated (that class's docstring has
        the rules). The table a law computes (``law.table(f)``) is refused:
        pass the law. Default ``None`` (no volume absorption). Models inspect this
        field to set ``TopOpt`` position 4 and write the supporting
        per-formula lines, or α per SSP row for a law with no letter.
        Replacing a Francois-Garrison law with one on the reference water
        that clearly differs from it gives a ``ProvenanceWarning``; the
        check sees only the water such a law carries, never the water
        behind a plain sound-speed profile.
    water_density : float, keyword-only
        Sea-water density in g/cm³. The decks that carry a water density
        write it (the Acoustics Toolbox and Bellhop SSP rows, the OASES
        water layers);
        the engines that fix the water at 1 and read seabed densities as
        ratios — the RAM codes and BOUNCE — receive each seabed density
        divided by it. Default ``None`` is
        :data:`~uacpy.core.constants.DEFAULT_WATER_DENSITY_G_CM3` (1.027);
        ``uacpy.core.acoustics.density(T, S) / 1000`` gives the value for a
        measured column. Seabed densities stay absolute g/cm³ (Hamilton's
        tables), so the impedance contrast the engines see is ρ_b/ρ_w
        rather than ρ_b/1 — 2.7 % less, a few tenths of a dB per bottom
        bounce. Pass ``water_density=1.0`` to reproduce a textbook
        benchmark that takes ρ_w = 1 by convention.
    name : str, keyword-only
        Environment identifier. Default ``'unnamed'``.
    location : (float, float), keyword-only
        Representative site as ``(lat, lon)`` in WGS84 decimal degrees.
        Default ``None``; falls back to the great-circle midpoint of the
        ``transect`` when only a transect is given, and follows the transect
        when one is assigned. Stamped by ``uacpy.data.fetch_environment``.
        Stored as given: the longitude in either sign convention within one
        full wrap (``|lon| <= 360``); a lookup wraps it
        (:func:`uacpy.core.geo.normalize_lon`).
    transect : ((float, float), (float, float)), keyword-only
        Great-circle path as ``((lat, lon) start, (lat, lon) end)`` in WGS84
        decimal degrees, for a range-dependent environment fetched along a
        track, stored as given. Default ``None``.
    date : str, datetime.date, datetime.datetime or numpy.datetime64, keyword-only
        The time this environment represents, stored as its UTC calendar date
        (:func:`uacpy.core.geo.parse_date`): no data source reads a finer
        time. Default ``None``.
    extra_data_sources : tuple of DataProvenance, keyword-only
        Provenance records that belong to no carrier:
        ``uacpy.data.fetch_environment`` puts the absorption's
        temperature/salinity row and its pH here. Default ``()``.
        :attr:`data_sources` merges them with the carriers' own records.

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

    bathymetry: Bathymetry
    ssp: SoundSpeedProfile = _RESOLVED_DEFAULT
    altimetry: Optional[Altimetry] = None
    bottom: Bottom = _RESOLVED_DEFAULT
    surface: Surface = _RESOLVED_DEFAULT
    absorption: Optional[Absorption] = None
    _: KW_ONLY
    name: str = 'unnamed'
    location: Optional[Tuple[float, float]] = None
    transect: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    date: Optional[datetime.date] = None
    water_density: float = _RESOLVED_DEFAULT
    extra_data_sources: tuple = ()

    if TYPE_CHECKING:
        # The attributes hold what ``__post_init__`` builds (the field
        # annotations above); the constructor takes the wide input unions
        # the Parameters section documents. Never executed; the runtime
        # ``__init__`` is ``@carrier``'s, with the same parameters.
        def __init__(
            self,
            bathymetry: Union[float, List[Tuple[float, float]], np.ndarray],
            ssp: Optional[Union[
                float, int, List[Tuple[float, float]], np.ndarray,
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
            absorption: Optional[Absorption] = None,
            *,
            name: str = 'unnamed',
            location: Optional[Tuple[float, float]] = None,
            transect: Optional[Tuple[Tuple[float, float],
                                     Tuple[float, float]]] = None,
            date: Optional[Union[str, datetime.date, np.datetime64]] = None,
            water_density: Optional[float] = None,
            extra_data_sources: tuple = (),
        ) -> None: ...

    def __post_init__(self):
        coerce = _FIELD_COERCION
        self.absorption = coerce['absorption'](self.absorption)
        self.water_density = coerce['water_density'](self.water_density)
        self.name = coerce['name'](self.name)

        # Optional geolocation (WGS84 decimal degrees) and the time the env
        # represents. ``transect`` is the ((lat, lon) start, (lat, lon) end)
        # great-circle path for a range-dependent env; ``location`` is the
        # representative site point — an explicit value if given, else the
        # transect midpoint. Stamped by ``uacpy.data.fetch_environment``;
        # ``None`` for a hand-built env. All survive ``env.copy()`` (deepcopy).
        self.transect = _coerce_transect(self.transect)
        self.location = _coerce_location(self.location, self.transect)
        self.date = coerce['date'](self.date)
        self.extra_data_sources = coerce['extra_data_sources'](
            self.extra_data_sources)

        # Bathymetry is a first-class carrier (seafloor depth vs range),
        # mirroring env.ssp; it validates in its own __post_init__.
        self.bathymetry = _owned(Bathymetry.coerce(self.bathymetry),
                                 self.bathymetry)
        max_bathy_depth = self.bathymetry.depth
        _warn_if_table_misses_water(self.absorption, max_bathy_depth)
        self.ssp = _owned(
            SoundSpeedProfile.coerce(self.ssp, depth_max=max_bathy_depth),
            self.ssp)

        # Altimetry is a first-class carrier (surface height vs range), the
        # top-surface analogue of env.bathymetry; ``None`` = flat z = 0.
        self.altimetry = coerce['altimetry'](self.altimetry)
        self.ssp = _spanning(self.ssp, max_bathy_depth)

        # Surface is a first-class carrier (top boundary vs range), the
        # top-properties analogue of env.bottom; a single BoundaryProperties
        # is coerced to a uniform one-node Surface.
        self.surface = coerce['surface'](self.surface)
        self.bottom = coerce['bottom'](self.bottom)

    # ── saving ─────────────────────────────────────────────────────────

    #: The carrier fields :meth:`to_netcdf` writes as one group each.
    _GROUPS = ('bathymetry', 'ssp', 'altimetry', 'bottom', 'surface',
               'absorption')

    def to_netcdf(self, path, **kwargs) -> None:
        """Save this environment to one NetCDF file, one group per carrier.

        Each of ``bathymetry``, ``ssp``, ``altimetry``, ``bottom``,
        ``surface`` and ``absorption`` that is set is written by its own
        ``to_xarray()`` as the group of that name — gridded values as
        variables with CF ``units``, everything else as JSON in the group's
        ``uacpy_fields`` attribute — and the environment's remaining fields
        (name, location, transect, date, water density, extra provenance) go
        in the root group's ``uacpy_fields``. :meth:`from_netcdf` reads it
        back. ``kwargs`` go to xarray's ``to_netcdf`` (a groups-capable
        engine is needed: netCDF4 or h5netcdf). ``np.savez(path,
        **env.to_dict())`` is the other save format.

        Parameters
        ----------
        path : str or Path
            Output file.
        **kwargs
            Keywords of xarray's ``to_netcdf``.
        """
        import json
        from uacpy.core._export import require_extra, _to_json
        xarray = require_extra('xarray', 'Environment.to_netcdf')
        rest = self.to_dict()
        for name in self._GROUPS:
            rest.pop(name)
        groups = [name for name in self._GROUPS
                  if getattr(self, name) is not None]
        rest['__groups__'] = groups
        xarray.Dataset(attrs={self._FIELDS_ATTR: json.dumps(_to_json(rest))}
                       ).to_netcdf(path, mode='w', **kwargs)
        for name in groups:
            getattr(self, name).to_xarray().to_netcdf(
                path, mode='a', group=name, **kwargs)

    @classmethod
    def from_netcdf(cls, path, **kwargs) -> 'Environment':
        """The environment :meth:`to_netcdf` wrote to ``path``, rebuilt
        through the constructor. ``kwargs`` go to ``xarray.open_dataset``.

        Parameters
        ----------
        path : str or Path
            A file :meth:`to_netcdf` wrote.
        **kwargs
            Keywords of ``xarray.open_dataset``.
        """
        import json
        from uacpy.core._export import (CarrierExport, _from_json,
                                        require_extra, _resolve_class)
        xarray = require_extra('xarray', 'Environment.from_netcdf')
        with xarray.open_dataset(path, **kwargs) as root:
            fields = _from_json(json.loads(root.attrs[cls._FIELDS_ATTR]))
        for name in fields.pop('__groups__'):
            with xarray.open_dataset(path, group=name, **kwargs) as group:
                group = group.load()
            saved = json.loads(group.attrs[cls._FIELDS_ATTR])
            klass = _resolve_class(saved['__class__'], CarrierExport)
            fields[name] = klass.from_xarray(group)
        return cls.from_dict(fields)

    def __setattr__(self, name, value):
        # A store while the constructor runs is construction. After it, a
        # field store is an assignment: checked and completed as the
        # constructor checks and completes that argument, with the fields it
        # moves with, all computed before any is stored, so a refused value
        # leaves the environment as it was. Only the assigned value is
        # copied; the other carriers are the environment's own already.
        if (name in self.__dataclass_fields__
                and id(self) not in _under_construction()):
            for field, new in self._assignment(name, value).items():
                object.__setattr__(self, field, new)
            return
        object.__setattr__(self, name, value)

    def _assignment(self, name, value) -> dict:
        """The fields assigning ``value`` to ``name`` stores: the value as the
        constructor stores it, plus the fields it moves with — the profile
        extended to span a deeper seafloor, and a ``location`` derived from
        the transect (its great-circle midpoint, or unset) derived again from
        a reassigned one; an explicit location stays."""
        if name == 'bathymetry':
            bathymetry = _owned(Bathymetry.coerce(value), value)
            _warn_if_table_misses_water(self.absorption, bathymetry.depth)
            return {'bathymetry': bathymetry,
                    'ssp': _spanning(self.ssp, bathymetry.depth)}
        if name == 'ssp':
            depth = self.bathymetry.depth
            ssp = _owned(SoundSpeedProfile.coerce(value, depth_max=depth),
                         value)
            return {'ssp': _spanning(ssp, depth)}
        if name == 'transect':
            transect = _coerce_transect(value)
            fields = {'transect': transect}
            derived = (self.location is None
                       or (self.transect is not None and self.location
                           == great_circle_midpoint(*self.transect)))
            if derived:
                fields['location'] = _coerce_location(None, transect)
            return fields
        if name == 'location':
            return {'location': _coerce_location(value, self.transect)}
        if name == 'absorption':
            absorption = _FIELD_COERCION[name](value)
            _warn_if_table_misses_water(absorption, self.bathymetry.depth)
            _warn_if_reference_water_replaces(self.absorption, absorption)
            return {name: absorption}
        return {name: _FIELD_COERCION[name](value)}

    @property
    def data_sources(self) -> tuple:
        """Provenance of this environment (read-only): each carrier's own
        ``data_sources`` in the order bathymetry → ssp → bottom → surface,
        then :attr:`extra_data_sources`, then the altimetry's, exact repeats
        removed: a transect keeps one record per column read, each with its
        ``range_m``. It is read from the carriers each time, so a reassigned
        carrier brings its own records."""
        return dedupe_records(
            dedupe_provenance((self.bathymetry, self.ssp, self.bottom,
                               self.surface))
            + self.extra_data_sources
            + dedupe_provenance((self.altimetry,)))

    def plot(self, ax=None, **kwargs):
        """Plot the water column + seafloor cross-section.

        Water column (SSP) + seafloor cross-section, with optional
        ``source=`` / ``receiver=`` markers. The carrier counterpart of
        :meth:`Result.plot` — any uacpy object you plot on its own has
        ``.plot()``. ``ax`` draws into an existing Axes, spelled the way every
        other uacpy plot method spells it; the remaining ``kwargs`` are
        forwarded to :func:`uacpy.plot.plot_environment`.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing axes; a new figure is made when omitted.
        **kwargs
            Keywords of :func:`uacpy.plot.plot_environment`.
        """
        return plotter('plot_carrier')(self, ax=ax, **kwargs)

    @property
    def depth(self) -> float:
        """Maximum water depth in metres (derived from bathymetry)."""
        return self.bathymetry.depth

    @property
    def range_max(self) -> float:
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

    @property
    def is_range_dependent(self) -> bool:
        """True when the bathymetry varies with range or the SSP, bottom or
        surface is range-dependent: the bathymetry counts when its depths
        change (:attr:`Bathymetry.varies_with_range`; a flat multi-point
        bathymetry does not), the ssp / bottom / surface when they carry
        more than one node on a ranged axis, identical or not.

        Altimetry is not consulted: a non-flat sea surface varies with range
        by nature, but it is boundary geometry that the models reading it
        take directly (Bellhop from its ``.ati`` file, RAM's ramsurf backend
        from its surface block), so it never triggers the segmented-profile
        machinery this flag selects for the four carriers above."""
        return (
            self.bathymetry.varies_with_range
            or self.ssp.is_range_dependent
            or self.bottom.is_range_dependent
            or self.surface.is_range_dependent
        )

    def __repr__(self) -> str:
        surface = self.surface
        bits = [None if self.name == 'unnamed' else repr(self.name),
                f"depth={extent(self.bathymetry.depths, 'm')}",
                'range-dependent' if self.is_range_dependent else None,
                f"c={extent(self.ssp.sound_speed, 'm/s')}",
                f"seabed {self.bottom._short()}"]
        if surface.is_range_dependent:
            bits.append(f"surface {len(surface.nodes)} nodes")
        elif surface.nodes[0].acoustic_type != 'vacuum':
            bits.append(f"surface {surface.nodes[0]._short()}")
        if self.altimetry is not None:
            bits.append(f"altimetry={extent(self.altimetry.heights, 'm')}")
        bits.append('no absorption' if self.absorption is None
                    else f"absorption={self.absorption._short()}")
        bits.append(f"ρw={qty(self.water_density, 'g/cm³')}")
        if self.transect is not None:
            (la0, lo0), (la1, lo1) = self.transect
            bits.append(f"transect ({la0:.3f}, {lo0:.3f})→({la1:.3f}, {lo1:.3f})")
        elif self.location is not None:
            bits.append(f"at ({self.location[0]:.3f}, {self.location[1]:.3f})")
        if self.date is not None:
            bits.append(f"date={self.date.isoformat()}")
        return build('Environment', bits)


__all__ = [
    'Environment',
    'SedimentLayer', 'BoundaryProperties', 'SeabedColumn', 'Bottom',
    'SoundSpeedProfile', 'Bathymetry', 'Altimetry', 'Surface',
]
