"""
Source class for defining acoustic sources in underwater environments
"""

import copy as _copy
import numpy as np
from pathlib import Path
from typing import Union, List, Optional
from dataclasses import dataclass

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._carrier_validate import (
    _require_positive, _require_non_negative, _require_strictly_increasing,
)


# eq=False: a dataclass __eq__ over ndarray fields raises; compare by identity.
@dataclass(eq=False)
class Source:
    """
    Acoustic source definition

    Represents one or more acoustic sources with specified depths and
    frequencies.

    Parameters
    ----------
    depths : float or array-like
        Source depth(s) in meters. Positive down from surface.
    frequencies : float or array-like
        Source frequency or frequencies in Hz
    source_type : str, optional
        Source *geometry*. 'point' (default) is a point source with cylindrical
        spreading; 'line' is an infinite coherent line source with Cartesian
        spreading; 'scaled' is a point source with the cylindrical spreading
        factor removed. Support is per-model — see each model's
        ``spec.source_types``. Note: this 'line' is a physical source shape and
        is unrelated to :class:`~uacpy.Receiver`'s ``receiver_type='line'``,
        which is a *sampling* rule (depths/ranges paired point-by-point rather
        than gridded). The shared word names two different concepts.
    beam_pattern : ndarray or Path, optional
        Source directivity: an ``(N, 2)`` array of ``[angle_deg, level_dB]``
        with strictly increasing angles, or a path to an existing ``.sbp``
        file. ``None`` (default) is omnidirectional. Read by Bellhop and
        Kraken. The engines convert levels to linear amplitude
        (``10**(dB/20)``, ``beampattern.f90:59``) *before* interpolating
        between samples, so a coarsely sampled pattern interpolates in
        amplitude rather than in dB — sample finely across steep roll-offs.
        Angles should span the full range the model queries: Bellhop uses
        launch angles (the reference ``shaded.sbp`` covers ±180°), Kraken
        uses mode angles in [0°, 90°].

    Attributes
    ----------
    depths : ndarray
        Source depth(s)
    frequencies : ndarray
        Source frequency/frequencies
    source_type : str
        Source geometry
    beam_pattern : ndarray or Path or None
        Source directivity

    Notes
    -----
    ``frequencies`` is the single source of truth for the frequency content.
    How it is consumed depends on the run mode:

    * Single-frequency modes (``COHERENT_TL`` / ``INCOHERENT_TL`` / rays /
      ``ARRIVALS`` / ``MODES``) require a length-1 ``frequencies``.
    * Broadband modes (``BROADBAND`` / ``TIME_SERIES``) use it as the grid:
      a multi-element array *is* the band (used as-is); a single value
      auto-expands to a default band ``fc·(1 ± bandwidth/2)``.

    ``run(..., frequencies=…)`` is an optional per-call **override** of this
    broadband grid (handy to reuse one ``Source`` while sweeping different
    grids); when omitted, ``source.frequencies`` is used.

    Examples
    --------
    Single source at 50m depth, 100 Hz:

    >>> source = Source(depths=50, frequencies=100)

    Vertical source array:

    >>> source = Source(depths=[10, 20, 30], frequencies=200)
    """

    depths: Union[float, List[float], np.ndarray]
    frequencies: Union[float, List[float], np.ndarray]
    source_type: str = 'point'
    beam_pattern: Optional[Union[np.ndarray, str, Path]] = None

    def __post_init__(self):
        self.depths = np.atleast_1d(np.array(self.depths, dtype=np.float64))
        self.frequencies = np.atleast_1d(
            np.array(self.frequencies, dtype=np.float64))

        if self.depths.size == 0:
            raise ConfigurationError(
                "Source requires at least one depth; got an empty array"
            )
        if self.frequencies.size == 0:
            raise ConfigurationError(
                "Source requires at least one frequency; got an empty array"
            )

        _require_non_negative(
            self.depths, "source depths", hint="metres, positive down from surface")
        # Strictly increasing, matching Receiver — outputs are indexed by source
        # depth, so a defined order keeps result rows unambiguous across models.
        _require_strictly_increasing(self.depths, "source depths")
        _require_positive(self.frequencies, "source frequencies", hint="Hz")

        valid_types = ['point', 'line', 'scaled']
        if self.source_type not in valid_types:
            raise ConfigurationError(
                f"source_type must be one of {valid_types}; "
                f"got {self.source_type!r}"
            )

        if self.beam_pattern is not None:
            if isinstance(self.beam_pattern, (str, Path)):
                self.beam_pattern = Path(self.beam_pattern)
            else:
                pattern = np.asarray(self.beam_pattern, dtype=np.float64)
                if pattern.ndim != 2 or pattern.shape[1] != 2:
                    raise ConfigurationError(
                        "Source beam_pattern must have shape (N, 2): "
                        f"[angle_deg, level_dB]; got shape {pattern.shape}"
                    )
                _require_strictly_increasing(
                    pattern[:, 0], "source beam-pattern angles")
                self.beam_pattern = pattern

    @property
    def n_sources(self) -> int:
        """Number of sources."""
        return len(self.depths)

    @property
    def n_frequencies(self) -> int:
        """Number of frequencies."""
        return len(self.frequencies)

    def __repr__(self) -> str:
        if self.n_sources == 1:
            depth_str = f"{self.depths[0]:.1f}m"
        else:
            depth_str = f"{self.n_sources} sources"

        if self.n_frequencies == 1:
            freq_str = f"{self.frequencies[0]:.1f}Hz"
        else:
            freq_str = f"{self.n_frequencies} frequencies"

        return (f"Source({depth_str}, {freq_str}, type='{self.source_type}')")

    def copy(self):
        """Deep copy (symmetric with the other carriers)."""
        return _copy.deepcopy(self)
