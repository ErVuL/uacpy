"""
Source class for defining acoustic sources in underwater environments
"""

import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Union, List, Optional
from dataclasses import dataclass

from uacpy.core.exceptions import ConfigurationError
from uacpy.core.constants import (
    DECK_DEPTH_RESOLUTION_M, SBP_ANGLE_RESOLUTION_DEG,
)
from uacpy.core._carrier_validate import (
    _DeepCopyMixin,
    _reject_complex,
    _require_positive, _require_non_negative, _require_strictly_increasing,
)


#: The source geometries a deck may ask for. This is the only declaration:
#: ``Source`` validates against it and ``uacpy.models.base`` re-exports it, so
#: a geometry added here reaches every layer that checks one.
VALID_SOURCE_TYPES: frozenset = frozenset({'point', 'line', 'scaled'})


# eq=False: a dataclass __eq__ over ndarray fields raises; compare by identity.
@dataclass(eq=False)
class Source(_DeepCopyMixin):
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
    weights : complex or array-like, optional
        Complex amplitude of each source: a scalar broadcasts to every
        depth, otherwise exactly one weight per depth (a length-1 array
        on a multi-depth source is a length mismatch); ``None`` (default)
        is unit weight everywhere. The engines never read it: every slab
        of a multi-depth run is the unit-amplitude field of one source,
        and the weights become the coefficients of
        :meth:`ResultStack.superpose`, which adds the slabs' complex
        pressure as ``Σ wᵢ·pᵢ``; a single-depth run scales its field by
        the one weight. ``[1, -1]`` drives two sources in antiphase;
        ``[1, 1j]`` puts them in quadrature. Every weight must be finite.
    source_level_dB : float, optional
        How hard a unit-weight element is driven, in dB re 1 uPa at 1 m.
        ``None`` (default) leaves it unstated, which is what transmission
        loss assumes -- TL is referenced to a *unit* source, so a run
        without this says how much quieter each cell is than the source
        rather than how loud it is. Giving it stamps
        ``metadata['source_level_dB']`` on every result, and
        :meth:`~uacpy.core.results.Field.at_source_level` then turns a loss
        into the absolute level a hydrophone there would read, with no
        argument to repeat at the call site. The engines never consume it:
        like ``weights`` it is applied to the field afterwards, so a run's
        TL is unchanged by setting it.

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
    weights : ndarray
        Complex amplitude per depth, shape ``(n_sources,)``

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

    Two sources in antiphase, summed after the run:

    >>> source = Source(depths=[40, 60], frequencies=200, weights=[1, -1])
    >>> field = model.run(env, source, receiver).superpose()
    """

    depths: np.ndarray
    frequencies: np.ndarray
    source_type: str = 'point'
    beam_pattern: Optional[Union[np.ndarray, str, Path]] = None
    weights: Optional[Union[complex, List[complex], np.ndarray]] = None
    source_level_dB: Optional[float] = None

    if TYPE_CHECKING:
        # The two roles of a dataclass field annotation, separated: the
        # attributes hold what ``__post_init__`` normalizes them to (float64
        # ndarrays, as the Attributes section above says), while the
        # constructor keeps taking the wide input union the Parameters
        # section documents. Declaring both through the field annotation
        # alone gives the union to every attribute read, so ``len(s.depths)``
        # and ``s.depths.shape`` are reported as errors in downstream code
        # that runs correctly. Never executed, so the decorator compiles the
        # runtime ``__init__`` from the fields exactly as before.
        def __init__(
            self,
            depths: Union[float, List[float], np.ndarray],
            frequencies: Union[float, List[float], np.ndarray],
            source_type: str = 'point',
            beam_pattern: Optional[Union[np.ndarray, str, Path]] = None,
            weights: Optional[Union[complex, List[complex],
                                    np.ndarray]] = None,
            source_level_dB: Optional[float] = None,
        ) -> None: ...

    def __post_init__(self):
        # Ahead of the float64 casts below, which discard an imaginary part —
        # see _reject_complex for the two ways they do it.
        _reject_complex(self.depths, "source depths")
        _reject_complex(self.frequencies, "source frequencies")
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
        _require_strictly_increasing(self.depths, "source depths",
                                     min_step=DECK_DEPTH_RESOLUTION_M)
        _require_positive(self.frequencies, "source frequencies", hint="Hz")

        if self.source_type not in VALID_SOURCE_TYPES:
            raise ConfigurationError(
                f"source_type must be one of {sorted(VALID_SOURCE_TYPES)}; "
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
                # The engines interpolate between adjacent rows, so a single row
                # leaves no pair to interpolate over: Bellhop's
                # ``bellhop.f90:273`` clamps its index to ``NSBPPts - 1 = 0``
                # and reads below the allocated bound, returning an all-NaN
                # field with exit code 0. Its own monotone guard
                # (``misc/monotonicMod.f90:20``) passes a one-element vector.
                if pattern.shape[0] < 2:
                    raise ConfigurationError(
                        "Source beam_pattern needs at least 2 (angle, level) "
                        f"rows; got {pattern.shape[0]}.",
                        remediation="Pass beam_pattern=None for an "
                                    "omnidirectional source, or give at least "
                                    "two angles spanning the launch fan.",
                    )
                # The ``.sbp`` angle column's resolution, so a pattern that
                # would collapse two angles into one token is refused here
                # rather than at write. A step of exactly the resolution is
                # refused here and accepted by ``write_source_beam_pattern``,
                # which writes the two distinct tokens it produces.
                _require_strictly_increasing(
                    pattern[:, 0], "source beam-pattern angles",
                    min_step=SBP_ANGLE_RESOLUTION_DEG, unit='deg')
                self.beam_pattern = pattern

        self.weights = self._normalise_weights(self.weights)

        if self.source_level_dB is not None:
            level = float(self.source_level_dB)
            if not np.isfinite(level):
                raise ConfigurationError(
                    f"Source source_level_dB must be finite (dB re 1 uPa at "
                    f"1 m); got {self.source_level_dB!r}"
                )
            self.source_level_dB = level

    def _normalise_weights(self, weights) -> np.ndarray:
        """One finite complex weight per depth: ``None`` is all ones, a
        scalar broadcasts, and a vector must match ``depths`` in length."""
        n = self.depths.size
        if weights is None:
            return np.ones(n, dtype=np.complex128)
        scalar = np.ndim(weights) == 0
        arr = np.atleast_1d(np.asarray(weights, dtype=np.complex128))
        if arr.ndim != 1:
            raise ConfigurationError(
                f"Source weights must be a scalar or a 1-D vector; got shape "
                f"{arr.shape}"
            )
        if scalar and n > 1:
            arr = np.repeat(arr, n)
        if arr.size != n:
            raise ConfigurationError(
                f"Source weights must give one weight per depth: "
                f"{n} depth(s) but {arr.size} weight(s)"
            )
        if not np.all(np.isfinite(arr)):
            bad = int(np.flatnonzero(~np.isfinite(arr))[0])
            raise ConfigurationError(
                f"Source weights must be finite (no NaN/inf); "
                f"weights[{bad}] = {arr[bad]}"
            )
        return arr

    @property
    def has_unit_weights(self) -> bool:
        """True when every weight is exactly 1, so a superposition is a
        plain sum of the slabs."""
        return bool(np.all(self.weights == 1.0))

    def at_depth(self, index: int) -> 'Source':
        """A single-depth copy of this source: ``depths[index]`` alone,
        unit weight, everything else shared.

        The per-depth loop in :meth:`PropagationModel.run` runs one of
        these per depth, so each slab of the returned stack is the field of
        one unit-amplitude source; the weight is applied at
        :meth:`ResultStack.superpose`."""
        return Source(
            depths=float(self.depths[index]),
            frequencies=self.frequencies,
            source_type=self.source_type,
            beam_pattern=self.beam_pattern,
            source_level_dB=self.source_level_dB,
        )

    def array_factor(self, angles_deg, *, frequency=None,
                     sound_speed: float = 1500.0) -> np.ndarray:
        """Complex free-field array factor ``AF(θ)`` of this source array.

        ``AF(θ) = Σₙ wₙ·exp(i·k·(zₙ - z̄)·sin θ)``, with ``θ`` in degrees
        from the horizontal (positive downward) and ``z̄`` the **mean** of
        the element depths — the array's phase centre, which for an
        unevenly spaced array is not its mid-depth. A caller composing
        this complex factor with a single element's field has to place
        that element at the same mean depth.
        By the product theorem the array's far field is the field of one
        element at that centre times this factor (Balanis, *Antenna Theory*,
        eq. 6-5). The theorem is stated for arrays of **identical**
        elements — which a uacpy ``Source`` satisfies, since one
        ``beam_pattern`` and one ``source_type`` describe every depth —
        and within that it holds for any magnitudes, phases and spacings.

        **This is a free-field quantity, and a waveguide is not free field.**
        A trapped mode is a standing wave — equal up- and down-going halves —
        so what the array actually does in a channel is set each mode's
        amplitude, ``Σₙ wₙ·φₘ(zₙ)``: the *mode filter* of Medwin & Clay
        §11.3.1, which :meth:`~uacpy.core.results.Modes.excitation` computes
        and which is exact there. Use this one for design intuition — where
        the lobes point, how steering moves them — and that one for what
        reaches the receiver. The two agree only while the pattern stays
        symmetric in ±θ, which steering is precisely what breaks.

        Parameters
        ----------
        angles_deg : array-like
            Angles from the horizontal, in degrees.
        frequency : float, optional
            Hz. Defaults to this source's single frequency, and is required
            when it carries a band.
        sound_speed : float
            Reference speed (m/s) setting the wavenumber. Default 1500.

        Returns
        -------
        ndarray
            Complex ``AF(θ)``, one entry per angle. A single-depth source
            returns its own weight at every angle.
        """
        angles = np.atleast_1d(np.asarray(angles_deg, dtype=float))
        if frequency is None:
            if self.frequencies.size != 1:
                raise ConfigurationError(
                    f"Source.array_factor: this source carries "
                    f"{self.frequencies.size} frequencies, so the one the "
                    f"factor is formed at has to be named: "
                    f"array_factor(angles, frequency=...)."
                )
            frequency = float(self.frequencies[0])
        frequency = float(frequency)
        if not (np.isfinite(frequency) and frequency > 0.0):
            raise ConfigurationError(
                f"Source.array_factor: frequency must be positive and "
                f"finite; got {frequency!r}.")
        if not (np.isfinite(sound_speed) and sound_speed > 0.0):
            raise ConfigurationError(
                f"Source.array_factor: sound_speed must be positive and "
                f"finite; got {sound_speed!r}.")
        k = 2.0 * np.pi * frequency / float(sound_speed)
        offsets = self.depths - self.depths.mean()
        phase = k * np.outer(np.sin(np.deg2rad(angles)), offsets)
        return np.exp(1j * phase) @ self.weights

    def element_directivity(self, angles_deg) -> np.ndarray:
        """This source's ``beam_pattern`` as a linear amplitude at
        ``angles_deg`` — ``f(θ)`` in the product theorem.

        Interpolated the way the engines read the ``.sbp``: levels are
        converted to amplitude (``10**(dB/20)``, ``beampattern.f90:59``)
        *before* interpolating between samples, so a coarsely sampled table
        gives the same numbers here as in the run. Ones everywhere when the
        source is omnidirectional; a table given as a path is not read, and
        raises.
        """
        angles = np.atleast_1d(np.asarray(angles_deg, dtype=float))
        if self.beam_pattern is None:
            return np.ones(angles.shape, dtype=float)
        if isinstance(self.beam_pattern, Path):
            raise ConfigurationError(
                f"Source.element_directivity: beam_pattern is a file path "
                f"({self.beam_pattern}); the table is read by the engine, "
                f"not here. Pass the (N, 2) [angle_deg, level_dB] array to "
                f"evaluate it in-process."
            )
        table = np.asarray(self.beam_pattern, dtype=float)
        return np.interp(angles, table[:, 0],
                         np.power(10.0, table[:, 1] / 20.0))

    def array_beam_pattern(self, angles_deg, *, frequency=None,
                           sound_speed: float = 1500.0) -> np.ndarray:
        """Complex **array beam pattern** ``P(θ) = f(θ)·A(θ)`` — what this
        source array actually radiates in the far field.

        The product theorem in the form Butler & Sherman state it for
        underwater arrays (*Transducers and Arrays for Underwater Sound*,
        §7.1.1): ``P(θ,φ) = f(θ,φ)·A(θ,φ)``, where ``f`` is the beam pattern
        of the identical elements — this source's :attr:`beam_pattern`, via
        :meth:`element_directivity` — and ``A`` is the beam pattern of the
        array of point sources at their centres, :meth:`array_factor`.

        The theorem needs the elements to be identical and *co-aligned*; it
        "does not apply to arrays on curved surfaces where the transducer
        axes do not all point in the same direction" (ibid.). A uacpy
        ``Source`` satisfies both by construction — one ``beam_pattern`` and
        one ``source_type`` describe every depth.

        This is the quantity to compare against a measurement or a
        specification. :meth:`array_factor` is the geometry alone, which is
        what the theorem isolates and what stays the same when the elements
        are swapped; it is **not** what a shaded array radiates. As with
        ``array_factor``, this is a free-field pattern: in a waveguide the
        array's effect is its modal excitation
        (:meth:`~uacpy.core.results.Modes.excitation`).
        """
        angles = np.atleast_1d(np.asarray(angles_deg, dtype=float))
        return (self.element_directivity(angles)
                * self.array_factor(angles, frequency=frequency,
                                    sound_speed=sound_speed))

    def plot_beam_pattern(self, ax=None, **kwargs):
        """Plot this source's directivity — the ``.sbp`` beam pattern.

        Dispatches to :func:`uacpy.visualization.plot_beam_pattern`. Named for
        the attribute it draws rather than spelled ``plot()`` like the other
        carriers, because a source's other rendering is the marker
        ``env.plot(source=...)`` / ``field.plot(source=...)`` draw, which needs
        an environment to sit in; ``.plot()`` would not say which of the two
        was meant. ``beam_pattern=None`` draws the flat 0 dB circle rather
        than raising, so the method answers "is this source directional?" for
        every source. ``ax`` draws into an existing Axes — a polar one unless
        ``polar=False`` — spelled the way every other uacpy plot method spells
        it; the remaining ``kwargs`` are forwarded."""
        # Deferred into the body: ``uacpy.visualization`` imports
        # ``uacpy.core`` at module scope, so this line at file scope makes
        # ``import uacpy`` raise ImportError. docs/DEV.md section 7 records
        # the inversion.
        from uacpy.visualization import plot_beam_pattern
        return plot_beam_pattern(self.beam_pattern, ax=ax, **kwargs)

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

        weight_str = ("" if self.has_unit_weights
                      else f", weights={self.weights.tolist()}")
        return (f"Source({depth_str}, {freq_str}, type='{self.source_type}'"
                f"{weight_str})")

# The dataclass compiles ``__init__`` from the *field* annotations, so
# ``inspect.signature`` / ``help()`` would advertise a default the annotation
# refuses (``depths: np.ndarray`` with no default is honest, but the two array
# fields still advertise the narrow type to a caller passing a float). Restate the input types on the
# generated ``__init__`` so the runtime signature says what the block above
# and the Parameters section say. Annotations only: no default, no field and
# no behaviour changes, and the class annotations — which are what an
# attribute read is checked against — are untouched.
Source.__init__.__annotations__.update(
    depths=Union[float, List[float], np.ndarray],
    frequencies=Union[float, List[float], np.ndarray],
    weights=Optional[Union[complex, List[complex], np.ndarray]],
    source_level_dB=Optional[float],
)
