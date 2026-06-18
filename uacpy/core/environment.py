"""Top-level ocean :class:`Environment` carrier.

The seafloor/boundary classes and the sound-speed-profile carrier live in
:mod:`uacpy.core.bottom` and :mod:`uacpy.core.ssp`; they are re-exported here so
``from uacpy.core.environment import BoundaryProperties`` (etc.) keeps working.
"""

import copy as _copy
import numpy as np
from typing import TYPE_CHECKING, Union, List, Tuple, Optional

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._carrier_validate import _require_strictly_increasing, _sanitize_title
from uacpy.core.bottom import (
    SedimentLayer, BoundaryProperties, RangeDependentBottom,
    LayeredBottom, RangeDependentLayeredBottom, _boundary_has_shear,
)
from uacpy.core.ssp import SoundSpeedProfile, generate_sea_surface

if TYPE_CHECKING:
    from uacpy.core.absorption import Absorption  # noqa: F401


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
    altimetry : array-like, optional
        Surface altimetry as ``[(range, height_m), …]`` (height
        positive up). Default ``None`` (flat surface).
    bottom : BoundaryProperties, RangeDependentBottom, LayeredBottom, or RangeDependentLayeredBottom, optional
        Bottom acoustic properties. Default is a fluid sand-like half-space
        (``sound_speed=1600`` m/s, ``density=1.5`` g/cm³,
        ``attenuation=0.5`` dB/wavelength). For a perfectly reflecting bottom,
        pass ``BoundaryProperties(acoustic_type='rigid')``.
    surface : BoundaryProperties, optional
        Surface boundary properties. Default vacuum (pressure release).
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
        altimetry: Optional[Union[List[Tuple[float, float]], np.ndarray]] = None,
        bottom: Optional[Union[
            BoundaryProperties, RangeDependentBottom,
            'LayeredBottom', 'RangeDependentLayeredBottom',
        ]] = None,
        surface: Optional[BoundaryProperties] = None,
        absorption: Optional['Absorption'] = None,
        *,
        name: str = 'unnamed',
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

        if np.ndim(bathymetry) == 0:   # scalar or 0-D ndarray
            water_depth = float(bathymetry)
            if water_depth <= 0:
                raise ConfigurationError(
                    f"Environment: bathymetry depth must be positive (m); "
                    f"got {water_depth}"
                )
            self.bathymetry = np.array([[0.0, water_depth]], dtype=np.float64)
        else:
            self.bathymetry = np.array(bathymetry, dtype=np.float64)
            if self.bathymetry.ndim != 2 or self.bathymetry.shape[1] != 2:
                raise ConfigurationError(
                    f"Environment: bathymetry must be a positive scalar or shape "
                    f"(N, 2) as [(range, depth), ...]; got shape "
                    f"{self.bathymetry.shape} (example: [(0, 100), (5000, 200)])"
                )
            if np.any(self.bathymetry[:, 0] < 0):
                raise ConfigurationError(
                    f"Environment: bathymetry ranges must be non-negative (m); "
                    f"got {self.bathymetry[:, 0].tolist()}"
                )
            if np.any(self.bathymetry[:, 1] <= 0):
                raise ConfigurationError(
                    f"Environment: bathymetry depths must be positive (m); "
                    f"got {self.bathymetry[:, 1].tolist()}"
                )
            _require_strictly_increasing(
                self.bathymetry[:, 0], "Environment.bathymetry ranges",
            )

        max_bathy_depth = float(np.max(self.bathymetry[:, 1]))

        if ssp is None:
            # default isovelocity at 1500 m/s
            self.ssp = SoundSpeedProfile.from_isovelocity(max_bathy_depth, 1500.0)
        elif isinstance(ssp, SoundSpeedProfile):
            self.ssp = ssp
        elif isinstance(ssp, (int, float, np.integer, np.floating)):
            # scalar → isovelocity at the given speed
            self.ssp = SoundSpeedProfile.from_isovelocity(max_bathy_depth, float(ssp))
        elif isinstance(ssp, (list, tuple, np.ndarray)):
            # list of (z, c) pairs → from_pairs (linear interp)
            self.ssp = SoundSpeedProfile.from_pairs(ssp)
        else:
            raise ConfigurationError(
                f"Environment: ssp must be a scalar (m/s), a list of (depth, "
                f"sound_speed) pairs, or a SoundSpeedProfile; got "
                f"{type(ssp).__name__}"
            )

        if altimetry is not None:
            self.altimetry = np.array(altimetry, dtype=np.float64)
            if self.altimetry.ndim != 2 or self.altimetry.shape[1] != 2:
                raise ConfigurationError(
                    f"Environment: altimetry must have shape (N, 2) as "
                    f"(range, height_m); got shape {self.altimetry.shape}"
                )
            _require_strictly_increasing(
                self.altimetry[:, 0], "Environment.altimetry ranges",
            )
        else:
            self.altimetry = None

        if max_bathy_depth > self.ssp.depths[-1]:
            self.ssp = self.ssp.extend_to(max_bathy_depth)

        if surface is None:
            self.surface = BoundaryProperties(acoustic_type='vacuum')
        else:
            self.surface = surface

        if bottom is None:
            self.bottom = BoundaryProperties(
                acoustic_type='half-space',
                density=1.5,
                sound_speed=1600.0,
                attenuation=0.5,
            )
        elif isinstance(bottom, (BoundaryProperties, RangeDependentBottom,
                                 LayeredBottom, RangeDependentLayeredBottom)):
            self.bottom = bottom
        else:
            raise ConfigurationError(
                f"Environment: bottom must be BoundaryProperties, "
                f"RangeDependentBottom, LayeredBottom, or "
                f"RangeDependentLayeredBottom; got {type(bottom).__name__}"
            )

    @property
    def depth(self) -> float:
        """Maximum water depth in metres (derived from bathymetry)."""
        return float(np.max(self.bathymetry[:, 1]))

    @property
    def max_range(self) -> float:
        """Range extent in metres across the environment's range-dependent axes.

        The largest range coordinate carried by the bathymetry, SSP or bottom;
        ``0.0`` for a range-independent environment. Derived (read-only),
        symmetric with :attr:`depth`. For an environment fetched along a
        transect this equals the transect's great-circle length, so it sizes a
        receiver range grid without recomputing the geodesic.
        """
        extent = float(np.max(self.bathymetry[:, 0]))
        if self.ssp.is_range_dependent:
            extent = max(extent, float(self.ssp.ranges[-1]))
        if isinstance(self.bottom, (RangeDependentBottom,
                                    RangeDependentLayeredBottom)):
            extent = max(extent, float(self.bottom.ranges[-1]))
        return extent

    def get_sound_speed(
        self, depth: Union[float, np.ndarray], range: float = 0.0
    ) -> np.ndarray:
        """Sound speed at given depth(s), at ``range`` for 2-D profiles."""
        slice_1d = (self.ssp.at(range=range)
                    if self.ssp.is_range_dependent else self.ssp)
        return np.interp(np.atleast_1d(depth), slice_1d.depths,
                         slice_1d.data[:, 0])

    def bathymetry_at_range(self, range: Union[float, np.ndarray]) -> np.ndarray:
        """Bathymetry depth at the requested range(s). ``range`` can be
        a scalar or array; ``env.bathymetry`` is a plain ``(N, 2)``
        ndarray, so this helper carries the interpolation logic."""
        range = np.atleast_1d(range)
        if len(self.bathymetry) == 1:
            # dtype=float so an int range query doesn't truncate a
            # fractional seafloor depth (the interp branch returns float).
            return np.full_like(range, self.bathymetry[0, 1], dtype=float)
        return np.interp(range, self.bathymetry[:, 0], self.bathymetry[:, 1])

    def halfspace_at_range(self, range: float) -> 'BoundaryProperties':
        """Return the *halfspace* :class:`BoundaryProperties` at ``range`` (m).

        Always returns a flat :class:`BoundaryProperties` regardless of
        the bottom flavour: for :class:`LayeredBottom` and
        :class:`RangeDependentLayeredBottom` this is the underlying
        halfspace beneath all sediment layers; for
        :class:`RangeDependentBottom` it is the linearly-interpolated
        sample; for a plain :class:`BoundaryProperties` it is the
        bottom itself. Used by env-file writers that emit a single
        bottom row (acoustic_type / sound_speed / density / ...).
        """
        b = self.bottom
        if isinstance(b, RangeDependentLayeredBottom):
            return b.at(range=range).halfspace
        if isinstance(b, LayeredBottom):
            return b.halfspace
        if isinstance(b, RangeDependentBottom):
            return b.at(range=range)
        return b

    def bottom_at_range(self, range: float):
        """Bottom properties at the requested range. Returns
        :class:`LayeredBottom` for layered envs, otherwise
        :class:`BoundaryProperties`."""
        if isinstance(self.bottom, RangeDependentLayeredBottom):
            return self.bottom.at(range=range)
        if isinstance(self.bottom, RangeDependentBottom):
            return self.bottom.at(range=range)
        return self.bottom

    def has_range_dependent_bathymetry(self) -> bool:
        """``True`` iff the seafloor depth actually varies with range.

        A multi-point bathymetry whose depths are all equal (flat) counts as
        range-independent.
        """
        if len(self.bathymetry) <= 1:
            return False
        depths = self.bathymetry[:, 1]
        return not bool(np.allclose(depths, depths[0]))

    def has_range_dependent_ssp(self) -> bool:
        return self.ssp.is_range_dependent

    def has_range_dependent_bottom(self) -> bool:
        return isinstance(self.bottom, RangeDependentBottom)

    def has_layered_bottom(self) -> bool:
        return isinstance(self.bottom, LayeredBottom)

    def has_range_dependent_layered_bottom(self) -> bool:
        return isinstance(self.bottom, RangeDependentLayeredBottom)

    def has_elastic_bottom(self) -> bool:
        """``True`` iff any sample of ``self.bottom`` carries non-zero shear.

        Walks the layer/profile structure of :class:`LayeredBottom` and
        :class:`RangeDependentLayeredBottom` so a stratified env with a
        single elastic layer reports ``True``. For
        :class:`RangeDependentBottom`, ``True`` iff *any* range sample has
        ``shear_speed > 0``.
        """
        return _boundary_has_shear(self.bottom)

    def has_elastic_surface(self) -> bool:
        """``True`` iff ``self.surface`` carries non-zero shear."""
        return _boundary_has_shear(self.surface)

    @property
    def is_range_dependent(self) -> bool:
        return (
            self.has_range_dependent_bathymetry()
            or self.ssp.is_range_dependent
            or isinstance(self.bottom, (RangeDependentBottom,
                                        RangeDependentLayeredBottom))
        )

    def __repr__(self) -> str:
        range_dep = "range-dep" if self.is_range_dependent else "range-indep"
        return (f"Environment(name='{self.name}', depth={self.depth:.1f}m, "
                f"ssp='{self.ssp.shape}', {range_dep})")

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
        depths = self.bathymetry[:, 1]

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
    'SedimentLayer', 'BoundaryProperties', 'RangeDependentBottom',
    'LayeredBottom', 'RangeDependentLayeredBottom',
    'SoundSpeedProfile', 'generate_sea_surface',
]
