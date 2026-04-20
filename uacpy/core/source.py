"""
Source class for defining acoustic sources in underwater environments
"""

import numpy as np
from typing import Union, List, Tuple, Optional

from uacpy.core.constants import DEFAULT_SOUND_SPEED


class Source:
    """
    Acoustic source definition

    Represents one or more acoustic sources with specified positions,
    frequencies, and beam patterns.

    Parameters
    ----------
    depth : float or array-like
        Source depth(s) in meters. Positive down from surface.
    frequency : float or array-like
        Source frequency or frequencies in Hz
    position : tuple or list, optional
        Source position as (x, y) in meters for 3D problems.
        Default is (0, 0).
    angles : array-like, optional
        Launch angles in degrees. Used for ray-tracing models.
        Default is [-80, 80] with 361 angles.
    source_type : str, optional
        Type of source: 'point' (default), 'line', 'array'
    beam_pattern : str or callable, optional
        Beam pattern specification. Default is 'omni' (omnidirectional).
    power : float, optional
        Source power in dB re 1 μPa @ 1m. Default is 0.
    phase : float or array-like, optional
        Source phase in radians. Default is 0.

    Attributes
    ----------
    depth : ndarray
        Source depth(s)
    frequency : ndarray
        Source frequency/frequencies
    position : tuple
        Source position (x, y)
    angles : ndarray
        Launch angles for ray tracing
    source_type : str
        Source type
    beam_pattern : str or callable
        Beam pattern
    power : float
        Source power
    phase : ndarray
        Source phase(s)

    Examples
    --------
    Single source at 50m depth, 100 Hz:

    >>> source = Source(depth=50, frequency=100)

    Vertical source array:

    >>> source = Source(depth=[10, 20, 30], frequency=200)

    Custom launch angles:

    >>> source = Source(depth=50, frequency=100,
    ...                 angles=np.linspace(-45, 45, 91))
    """

    def __init__(
        self,
        depth: Union[float, List[float], np.ndarray],
        frequency: Union[float, List[float], np.ndarray],
        position: Optional[Tuple[float, float]] = None,
        angles: Optional[Union[List[float], np.ndarray]] = None,
        source_type: str = 'point',
        beam_pattern: Union[str, callable] = 'omni',
        power: float = 0.0,
        phase: Union[float, List[float], np.ndarray] = 0.0,
    ):
        # Convert to numpy arrays
        self.depth = np.atleast_1d(np.array(depth, dtype=np.float64))
        self.frequency = np.atleast_1d(np.array(frequency, dtype=np.float64))
        self.phase = np.atleast_1d(np.array(phase, dtype=np.float64))

        # Validate depths are positive
        if np.any(self.depth < 0):
            raise ValueError("Source depths must be positive (down from surface)")

        # Set position
        self.position = position if position is not None else (0.0, 0.0)

        # Set launch angles for ray tracing
        if angles is None:
            # Default: -80 to 80 degrees, 361 angles
            self.angles = np.linspace(-80, 80, 361)
        else:
            self.angles = np.array(angles, dtype=np.float64)

        # Set other parameters
        self.source_type = source_type
        self.beam_pattern = beam_pattern
        self.power = power

        # Validate source type
        valid_types = ['point', 'line', 'array']
        if source_type not in valid_types:
            raise ValueError(f"source_type must be one of {valid_types}")

    @property
    def n_sources(self) -> int:
        """Number of sources"""
        return len(self.depth)

    @property
    def n_frequencies(self) -> int:
        """Number of frequencies"""
        return len(self.frequency)

    @property
    def n_angles(self) -> int:
        """Number of launch angles"""
        return len(self.angles)

    @property
    def wavelength(self) -> np.ndarray:
        """
        Wavelength(s) in meters

        Computed assuming nominal sound speed of 1500 m/s.
        For accurate wavelengths, use environment.get_wavelength(source)
        """
        c = DEFAULT_SOUND_SPEED
        return c / self.frequency

    def get_beam_pattern_value(self, angle: Union[float, np.ndarray]) -> np.ndarray:
        """
        Get beam pattern amplitude at specified angle(s)

        Parameters
        ----------
        angle : float or array-like
            Angle(s) in degrees

        Returns
        -------
        amplitude : float or ndarray
            Beam pattern amplitude (0 to 1)
        """
        angle = np.atleast_1d(angle)

        if self.beam_pattern == 'omni':
            return np.ones_like(angle)
        elif callable(self.beam_pattern):
            return self.beam_pattern(angle)
        else:
            raise ValueError(f"Unknown beam pattern: {self.beam_pattern}")

    def __repr__(self) -> str:
        if self.n_sources == 1:
            depth_str = f"{self.depth[0]:.1f}m"
        else:
            depth_str = f"{self.n_sources} sources"

        if self.n_frequencies == 1:
            freq_str = f"{self.frequency[0]:.1f}Hz"
        else:
            freq_str = f"{self.n_frequencies} frequencies"

        return (f"Source({depth_str}, {freq_str}, "
                f"{self.n_angles} angles, type='{self.source_type}')")

    def copy(self):
        """Create a deep copy of the source"""
        return Source(
            depth=self.depth.copy(),
            frequency=self.frequency.copy(),
            position=self.position,
            angles=self.angles.copy(),
            source_type=self.source_type,
            beam_pattern=self.beam_pattern,
            power=self.power,
            phase=self.phase.copy(),
        )
