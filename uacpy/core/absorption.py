"""Volume-attenuation models for the water column.

Volume attenuation is an environmental property — same water column, same
absorption regardless of which propagation solver runs over it. uacpy
stores it on :class:`~uacpy.core.environment.Environment` as
``env.absorption`` and each model writer reads it to emit the right
Acoustics-Toolbox ``TopOpt`` position-4 character and the supporting
per-formula parameters, or, for a law the deck has no formula for (a
Francois-Garrison T/S profile, a measured table), α at the deck frequency in
every water SSP row.

``Environment(absorption=...)`` takes a law, or a measured
:class:`AbsorptionCoefficient` with no law behind it (its docstring has the
rules): that table is the one public path to a tabulated α(f, z), and
``env.absorption`` is that table (the environment's copy).

Concrete subclasses
-------------------
:class:`Thorp`
    Frequency-only seawater absorption (Thorp 1967). No free parameters.
:class:`FrancoisGarrison`
    Francois–Garrison (1982) frequency / T / S / pH / depth model, for one
    water row or a T/S profile (``(depth, value)`` pairs).
:class:`Biological`
    Layered fish-bladder resonance model (multiple
    ``(Z_top, Z_bottom, f0, Q, a0)`` blocks).
:class:`ConstantAbsorption`
    Frequency-independent baseline written into every SSP-block ``alphaI``
    row (dB/wavelength). Useful for calibrated ad-hoc absorption.

One object per model: the law. ``law.table(frequencies, depths=)`` evaluates
it into an :class:`AbsorptionCoefficient` (α with its units, its axes, the
model and its parameters) to look at, plot or export; the law itself is what
an :class:`~uacpy.core.environment.Environment` takes. The formulas on plain arrays (Thorp, Francois–Garrison, the biological
resonance, the pH scale conversion) and :func:`convert_attenuation_units` live
in :mod:`uacpy.core.acoustics.attenuation`; the models here delegate to them.
"""

from __future__ import annotations

import warnings

import numpy as np
from dataclasses import dataclass, field
# ``Tuple`` is read by name: it is in Biological's ``init_annotations``
# string, which ``typing.get_type_hints`` evaluates in this namespace.
from typing import Any, Dict, List, Optional, Tuple, Union  # noqa: F401

from uacpy.core.constants import (
    DEFAULT_SOUND_SPEED, PH_MAX, PH_MIN, REFERENCE_PH,
    REFERENCE_SALINITY, REFERENCE_TEMPERATURE)
from uacpy.core.deck_limits import MAX_ATTENUATION
from uacpy.core.exceptions import (
    ConfigurationError, NumericsWarning, ValidityWarning,
)
from uacpy.core._water import WaterProfile
from uacpy.core.acoustics.attenuation import (
    PH_SCALES, absorption_biological, convert_attenuation_units,
    absorption_francois_garrison, ph_to_nbs, absorption_thorp,
)
from uacpy.core._validate import (
    require_finite, require_non_negative, warn_attenuation_past_at_ceiling,
    water_pairs, water_property, water_shape,
)
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._repr import axis, build, num, qty, unit_of
from uacpy.core._plotting import plotter
from uacpy.core._carrier import (
    DeepCopyMixin, RevalidateOnAssignMixin, carrier,
)
from uacpy.core._export import CarrierExport
from uacpy.core._provenance import coerce_data_sources

__all__ = [
    'AbsorptionCoefficient',
    'Absorption', 'Thorp', 'FrancoisGarrison', 'BiologicalLayer',
    'Biological', 'ConstantAbsorption', 'arrival_absorption_exponent',
    'minimax_anchor_frequency',
]


_ArrayLike = Union[float, np.ndarray]


# ─────────────────────────────────────────────────────────────────────────────
# Env-attachable Absorption hierarchy
# ─────────────────────────────────────────────────────────────────────────────


#: The repr unit of each ocean value an absorption formula takes, by its
#: keyword name (:attr:`AbsorptionCoefficient.parameters`).
_OCEAN_UNITS = {name: unit_of(name) for name in ('temperature', 'salinity')}
#: The letter a profile's span is shown under, by keyword name.
_PROFILE_LETTERS = {'temperature': 'T', 'salinity': 'S', 'pH': 'pH'}


def _span(values) -> str:
    """``'4–22'`` for a profile's values to three significant figures, one
    number when they are all equal."""
    lo, hi = float(np.min(values)), float(np.max(values))
    return f"{lo:.3g}" if f"{lo:.3g}" == f"{hi:.3g}" else f"{lo:.3g}–{hi:.3g}"


def _ocean_bits(parameters: Dict[str, Any]) -> List[str]:
    """The ocean values of an absorption formula in repr words:
    ``['10 °C', '35 psu', 'pH 8']`` for one water row; a property given as
    ``(depth, value)`` pairs (the stored
    :class:`~uacpy.core._water.WaterProfile`) by its span and
    its depth count, ``'T 4–22 °C (12 depths)'``. The pH scale is named when
    it is not the NBS scale the formula is written on."""
    bits = []
    for name, value in parameters.items():
        profile = isinstance(value, WaterProfile)
        shown = value.values if profile else value
        n = len(value.depths) if profile else 0
        count = f" ({n} depth{'s' if n > 1 else ''})" if profile else ''
        if name == 'pH':
            scale = parameters.get('ph_scale', 'nbs')
            text = _span(shown) if np.ndim(shown) else num(shown)
            bits.append(f"pH {text}"
                        + ('' if scale == 'nbs' else f" {scale}") + count)
        elif np.ndim(shown) and name in _PROFILE_LETTERS:
            bits.append(f"{_PROFILE_LETTERS[name]} {_span(shown)} "
                        f"{_OCEAN_UNITS[name]}{count}")
        elif name in _OCEAN_UNITS:
            bits.append(qty(value, _OCEAN_UNITS[name]))
        elif name == 'layers':
            top = min(layer[0] for layer in value)
            bottom = max(layer[1] for layer in value)
            n = len(value)
            bits.append(f"{n} layer{'s' if n > 1 else ''} "
                        f"{num(top)}–{num(bottom)} m")
        elif name == 'value':
            bits.append(qty(value, 'dB/λ'))
        elif name != 'ph_scale':
            bits.append(f"{name}={num(value)}")
    return bits


# eq=False: a dataclass __eq__ over ndarray fields raises; compare by identity.
@dataclass(frozen=True, eq=False)
class AbsorptionCoefficient(CarrierExport):
    """alpha over frequency, and optionally over depth, in stated units.

    The carrier a law's :meth:`Absorption.table` returns. It is a
    property of the **medium** that you supply, not something a propagation
    model computed, so it is not a :class:`~uacpy.core.results.Field`. Its
    shape follows
    :class:`~uacpy.core.results.reflection.ReflectionCoefficient`, the other
    coefficient-over-an-axis in the package.

    ``data`` is 1-D over frequency when no depth axis was asked for, and
    ``(n_depths, n_frequencies)`` when one was — depth first, the convention
    every other 2-D quantity here uses.

    The carrier exists for :meth:`to_units` rather than for :meth:`plot`.
    Three of the six conventions :func:`convert_attenuation_units` supports —
    ``dB/wavelength``, ``Q`` and ``L`` — are frequency dependent, so they
    cannot be evaluated once the frequency axis has been discarded. Returning
    a bare array would mean the unit had to be known at the call; keeping the
    axis means it can be changed after it. The unit is a value the result
    carries, never a suffix in a name.

    **A measured table is an environment's absorption.** A law is one
    object, and an environment takes the law itself; the table a law
    returns is for looking at, and ``Environment(absorption=law.table(f))``
    is refused (pass the law). A table with no law behind it — a measured
    α(f, z), built directly with ``model=None`` (the default) — is the one
    table an environment takes, used as tabulated: linear in depth between
    its rows, holding the first and last row beyond them with a
    ``ValidityWarning``, and linear in ``log f`` between its frequencies. A
    frequency outside :attr:`frequencies` is refused
    (``ConfigurationError``): a table carries no law to extrapolate with. A
    1-D table (no depth axis) applies at every depth. ``env.absorption`` is
    then this table (the environment's own copy), with its :meth:`plot`,
    :attr:`data` and exports.
    """

    frequencies: np.ndarray
    data: np.ndarray
    units: str
    #: The law that computed the table (``'thorp'``, ``'francois_garrison'``,
    #: ``'biological'``, ``'constant'``), or ``None`` for a measured table.
    model: Optional[str] = None
    depths: Optional[np.ndarray] = None
    #: The one depth a 1-D curve was evaluated at, in metres — the scalar
    #: ``table(depths=...)`` was given, or the surface (0 m) when it was
    #: given none. ``None`` exactly when :attr:`depths` is an axis. Separate
    #: from ``depths`` because ``depths`` is the *shape* flag and must stay
    #: ``None`` for a curve; without this the evaluation depth would be used
    #: and then dropped, and the empty-curve warning could not tell a
    #: Biological user which depth they had just evaluated at.
    depth: Optional[float] = None
    #: The values the law was evaluated with, under its field names
    #: (Francois-Garrison: ``temperature``, ``salinity``, ``pH`` — each a
    #: number or a profile, the
    #: :class:`~uacpy.core._water.WaterProfile` the law stores
    #: its ``(depth, value)`` pairs as — and
    #: ``ph_scale``; Biological: ``layers``, as ``(z_top, z_bottom, f0,
    #: Q, a0)`` tuples; constant: ``value``), so
    #: ``FrancoisGarrison(**a.parameters)`` is the law again. ``None`` for a
    #: model that takes no values (Thorp) and for a measured table.
    parameters: Optional[Dict[str, Any]] = None
    #: Where the water came from: the
    #: :class:`~uacpy.core.provenance.DataProvenance` records of a fetched
    #: law's temperature, salinity and pH, which :meth:`Absorption.table`
    #: copies from :attr:`Absorption.data_sources`; ``()`` for water typed
    #: by hand. The table's plot credits them.
    data_sources: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, 'data_sources', coerce_data_sources(
            self.data_sources, "AbsorptionCoefficient"))

    @property
    def is_depth_dependent(self) -> bool:
        return self.depths is not None

    # The export protocol: alpha on (depth, frequency) or (frequency).
    _XARRAY_FIELDS = {'alpha': 'data', 'frequency': 'frequencies',
                      'depth': 'depths'}

    def _payload(self):
        dims = (('depth', 'frequency') if self.is_depth_dependent
                else ('frequency',))
        return {'alpha': (np.asarray(self.data), dims, self.units)}

    def _coords(self):
        coords = {'frequency': (np.atleast_1d(self.frequencies), 'Hz')}
        if self.is_depth_dependent:
            coords['depth'] = (np.atleast_1d(self.depths), 'm')
        return coords

    @property
    def n_frequencies(self) -> int:
        return int(np.size(self.frequencies))

    @property
    def n_depths(self) -> int:
        return 0 if self.depths is None else int(np.size(self.depths))

    @classmethod
    def from_pairs(cls, pairs, *, units: str = 'dB/km'
                   ) -> "AbsorptionCoefficient":
        """A measured table from ``[(frequency, value), ...]`` pairs, the
        absorption counterpart of :meth:`SoundSpeedProfile.from_pairs`.

        ``pairs`` is either one spectrum, shape ``(N, 2)`` as ``(frequency
        in Hz, value)`` rows, which applies at every depth; or a mapping
        ``{depth: pairs}`` with one spectrum per depth, all on the same
        frequencies (a depth on other frequencies is refused rather than
        interpolated, which would invent data). The result is a measured
        table (``model=None``), the one table ``Environment(absorption=...)``
        takes, and is checked by the same rules: frequencies positive and
        strictly increasing, values finite and non-negative, a known unit.

        Parameters
        ----------
        pairs : array_like or mapping
            ``(frequency, value)`` rows, or ``{depth: rows}``.
        units : str, optional
            The values' unit: ``'dB/km'`` (default), ``'dB/m'``,
            ``'dB/wavelength'``, ``'Nepers/m'``, ``'Q'`` or ``'L'``.
        """
        who = "AbsorptionCoefficient.from_pairs"

        def rows(value, where):
            arr = np.asarray(value, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 1:
                raise ConfigurationError(
                    f"{who}: {where}pairs must have shape (N, 2) as "
                    f"(frequency, value); got shape {arr.shape}.",
                    remediation="Pass [(frequency, value), ...].")
            return arr[:, 0], arr[:, 1]

        if isinstance(pairs, dict):
            if not pairs:
                raise ConfigurationError(
                    f"{who}: the depth mapping is empty.",
                    remediation="Pass {depth: [(frequency, value), ...]}.")
            depths = np.array(sorted(float(z) for z in pairs), dtype=float)
            by_depth = {float(z): v for z, v in pairs.items()}
            freqs, data = None, []
            for z in depths:
                f, v = rows(by_depth[z], f"the depth {z:g} m ")
                if freqs is None:
                    freqs = f
                elif f.shape != freqs.shape or not np.array_equal(f, freqs):
                    raise ConfigurationError(
                        f"{who}: the depth {z:g} m uses other frequencies "
                        f"than the depth {depths[0]:g} m; every depth must "
                        f"list the same frequencies.",
                        remediation="Give every depth the same frequency "
                                    "column, or build one table per depth.")
                data.append(v)
            if np.any(depths < 0.0):
                raise ConfigurationError(
                    f"{who}: depths must be >= 0 m; got {depths.min():g}.")
            table = cls(frequencies=freqs, data=np.vstack(data),
                        units=units, depths=depths)
        else:
            f, v = rows(pairs, "")
            table = cls(frequencies=f, data=v, units=units)
        # The same rules the table meets when an environment takes it.
        _absorption_law(table)
        return table

    def to_units(self, units: str, *,
                 sound_speed: float = DEFAULT_SOUND_SPEED
                 ) -> "AbsorptionCoefficient":
        """The same alpha in another convention, converted per frequency.

        Per frequency, not once for the array: ``dB/wavelength``, ``Q`` and
        ``L`` all divide by the wavelength, so a single frequency applied to
        the whole axis would be right at one bin and wrong at every other.

        Parameters
        ----------
        units : str
            The target convention: ``'dB/km'``, ``'dB/m'``, ``'dB/wavelength'``,
            ``'Nepers/m'``, ``'Q'`` or ``'L'``.
        sound_speed : float, optional
            Sound speed (m/s). Default :data:`~uacpy.core.constants.DEFAULT_SOUND_SPEED`.
        """
        if units == self.units:
            return self
        # Frequency is the last axis of ``data`` whether or not a depth axis
        # precedes it, so the frequencies broadcast along it.
        out = np.asarray(convert_attenuation_units(
            np.asarray(self.data, dtype=float),
            np.asarray(self.frequencies, dtype=float), self.units, units,
            sound_speed=sound_speed), dtype=float)
        return AbsorptionCoefficient(
            frequencies=self.frequencies, data=out, units=units,
            model=self.model, depths=self.depths, depth=self.depth,
            parameters=self.parameters, data_sources=self.data_sources)

    #: The plotter :func:`uacpy.plot.plot_carrier` draws this table with.
    _plotter = 'plot_absorption'

    def plot(self, ax=None, **kwargs):
        """Draw alpha against frequency (log-log), or as a depth-frequency
        heatmap when a depth axis is present, through
        :func:`uacpy.plot.plot_absorption`.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing axes; a new figure is made when omitted.
        **kwargs
            Keywords of :func:`uacpy.plot.plot_absorption`.
        """
        return plotter('plot_carrier')(self, ax=ax, **kwargs)

    def __repr__(self) -> str:
        parameters = self.parameters or {}
        bits = [self.model or 'tabulated', *_ocean_bits(parameters)]
        if self.is_depth_dependent:
            bits.append(axis(self.depths, 'depths', 'm'))
        elif self.depth is not None:
            bits.append(f"at {qty(self.depth, 'm')}")
        bits += [axis(self.frequencies, 'frequencies', 'Hz'), self.units]
        return build('AbsorptionCoefficient', bits)


@carrier
class Absorption(RevalidateOnAssignMixin, DeepCopyMixin, CarrierExport):
    """Abstract base for water-column absorption models. Do not
    instantiate directly — pick one of :class:`Thorp`,
    :class:`FrancoisGarrison`, :class:`Biological`,
    :class:`ConstantAbsorption`.

    Subclasses implement :meth:`_alpha_dB_per_m`, which evaluates
    ``α(f, z)`` on a frequency axis and a depth axis at once, as an
    ``(n_depths, n_frequencies)`` grid (used by the Kraken-class modal
    perturbation kernel; the Acoustics-Toolbox writers read the model's
    letter and records from :mod:`uacpy.io.at_codes` instead, or
    :meth:`_alpha_dB_per_wavelength_at_frequency` per SSP row for a law with
    no letter). :meth:`table` evaluates its whole grid in one call and is
    the route to the law's values (``law.table(f, depths, units='dB/m')``,
    a carrier to plot or export); an engine's one-frequency evaluation,
    :meth:`_alpha_dB_per_m_at_frequency`, checks the frequency and delegates
    to it.
    """

    #: ``(low, high)`` Hz the model was fitted over, or ``None`` for a model
    #: with no stated band. A frequency outside it is evaluated as given and
    #: announced once per call by :meth:`_check_frequency_range`.
    _FREQUENCY_RANGE_HZ = None
    #: The source sentence the out-of-band notice quotes after the counts.
    _FREQUENCY_RANGE_NOTE = ""
    #: ``(low, high)`` m the equation is stated for, or ``None``; a depth
    #: outside it is evaluated as given and announced once per call by
    #: :meth:`_check_depth_range`.
    _DEPTH_RANGE_M = None
    #: The source sentence the out-of-range depth notice quotes.
    _DEPTH_RANGE_NOTE = ""

    #: Where the water came from: the
    #: :class:`~uacpy.core.provenance.DataProvenance` records of the
    #: temperature, salinity and pH a fetched law was built from
    #: (``uacpy.data.fetch_environment(..., with_absorption=True)`` sets
    #: them); ``()`` for water typed by hand. Keyword-only, after the law's
    #: own values. :meth:`table` copies them onto its table, and
    #: ``env.data_sources`` gathers them with the other carriers'.
    data_sources: tuple = field(default=(), kw_only=True)

    def __post_init__(self):
        if type(self) is Absorption:
            raise ConfigurationError(
                "Absorption is abstract; instantiate Thorp / "
                "FrancoisGarrison / Biological / ConstantAbsorption."
            )
        self.data_sources = coerce_data_sources(
            self.data_sources, type(self).__name__)

    def _check_frequency_range(self, frequencies) -> None:
        """Check ``frequencies`` against the law's fitted band: one notice
        per call for every frequency outside it, naming how many samples and
        which span fall below and above it. A table, which has no law to
        extrapolate with, refuses instead."""
        if self._FREQUENCY_RANGE_HZ is None:
            return
        low, high = self._FREQUENCY_RANGE_HZ
        f = np.atleast_1d(np.asarray(frequencies, dtype=float))
        parts = []
        for side, mask, edge in (('below', f < low, low),
                                 ('above', f > high, high)):
            if not np.any(mask):
                continue
            n, out = int(np.count_nonzero(mask)), f[mask]
            span = (f"{out.min():.10g} Hz" if n == 1 else
                    f"{out.min():.10g}-{out.max():.10g} Hz")
            parts.append(f"{n} of {f.size} "
                         f"{'frequency' if f.size == 1 else 'frequencies'} "
                         f"({span}) {'is' if n == 1 else 'are'} {side} "
                         f"{edge:g} Hz")
        if parts:
            warnings.warn(
                f"{type(self).__name__}: {'; '.join(parts)} — outside the "
                f"{low:g} Hz..{high:g} Hz the equation was fitted over. "
                f"{self._FREQUENCY_RANGE_NOTE}",
                ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)

    def _check_depth_range(self, depths) -> None:
        """One notice per call for the depths outside the law's stated
        depth range, naming how many and their span; nothing for a law with
        none."""
        if self._DEPTH_RANGE_M is None:
            return
        low, high = self._DEPTH_RANGE_M
        z = np.atleast_1d(np.asarray(depths, dtype=float))
        out = z[(z < low) | (z > high)]
        if out.size:
            span = (f"{out.min():.10g} m" if out.size == 1 else
                    f"{out.min():.10g}-{out.max():.10g} m")
            warnings.warn(
                f"{type(self).__name__}: {out.size} of {z.size} "
                f"{'depth' if z.size == 1 else 'depths'} ({span}) "
                f"{'is' if out.size == 1 else 'are'} outside the "
                f"{low:g}..{high:g} m the equation is stated for. "
                f"{self._DEPTH_RANGE_NOTE}",
                ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)

    def _alpha_dB_per_m_at_frequency(
        self,
        frequency: float,
        depths: _ArrayLike,
    ) -> np.ndarray:
        """Evaluate ``α(f, z)`` in dB/m at one frequency, on a depth grid.

        Parameters
        ----------
        frequency : float
            Frequency in Hz. Must be strictly positive — every model here is
            a formula in ``f`` (or in the wavelength ``c/f``) that has no
            value at or below zero. The check sits on this side of the
            dispatch so all four models share one rule: Thorp and
            Francois-Garrison are polynomials that evaluate happily at
            ``f <= 0`` and hand back a *positive* attenuation for a negative
            frequency. ``NaN`` is rejected here too.
        depths : float or 1-D array
            Depths (m).

        Returns
        -------
        ndarray, the shape of ``depths`` — a scalar depth comes back as a
        1-element array of shape ``(1,)``.
        """
        f = float(frequency)
        if not f > 0.0:
            raise ConfigurationError(
                f"{type(self).__name__}: frequency must be > 0 Hz; got "
                f"{frequency}."
            )
        self._check_frequency_range(f)
        z = np.atleast_1d(np.asarray(depths, dtype=float))
        self._check_depth_range(z)
        return self._alpha_dB_per_m(np.array([f]), z.ravel())[:, 0].reshape(
            z.shape)

    @property
    def _table_depths(self) -> _ArrayLike:
        """The depth argument :meth:`table` takes when given none: the
        surface (0 m), a curve. A Francois-Garrison profile overrides it with
        its own depth axis."""
        return 0.0

    def table(
        self,
        frequencies: _ArrayLike,
        depths: Optional[_ArrayLike] = None,
        *,
        units: str = 'dB/km',
        sound_speed: float = DEFAULT_SOUND_SPEED,
    ) -> AbsorptionCoefficient:
        """The law's alpha over frequency, and over depth if asked, as an
        :class:`AbsorptionCoefficient` that records this law (its
        :attr:`~AbsorptionCoefficient.model`,
        :attr:`~AbsorptionCoefficient.parameters` and
        :attr:`~AbsorptionCoefficient.data_sources`) — the table to look at,
        plot or export. The law, not its table, is what an environment takes.

        Vectorised on **both** axes, so one call answers alpha(f, z): an
        array of frequencies, an array of depths, or both together.

        The depth argument decides the shape, the way a scalar and a
        sequence decide it on :class:`~uacpy.core.receiver.Receiver`:

        - ``depths=None`` — the surface (0 m), 1-D over frequency; for a
          :class:`FrancoisGarrison` profile, the profile's own depths, an
          axis.
        - ``depths=50.0`` — a scalar: evaluate there, still 1-D. This is the
          single-depth curve, not a one-row grid.
        - ``depths=[0, 50, 100]`` — an axis: ``(n_depths, n_frequencies)``.

        Parameters
        ----------
        frequencies : float or array_like
            Frequencies (Hz), each > 0.
        depths : float or array_like, optional
            The depth argument (see above).
        units : str, optional
            Unit of the returned alpha. Default ``'dB/km'``.
        sound_speed : float, optional
            Sound speed (m/s). Default :data:`~uacpy.core.constants.DEFAULT_SOUND_SPEED`.
        """
        f = np.atleast_1d(np.asarray(frequencies, dtype=float))
        if f.size == 0:
            raise ConfigurationError(
                f"{type(self).__name__}.table: frequencies is empty; give at "
                f"least one frequency in Hz.")
        if not np.all(f > 0.0):
            raise ConfigurationError(
                f"{type(self).__name__}.table: every frequency must be > 0 "
                f"Hz; got {frequencies!r}.")
        self._check_frequency_range(f)
        if depths is None:
            depths = self._table_depths
        # A scalar depth is a place to evaluate, not an axis to span: it
        # returns a curve, so asking for one depth never yields a one-row
        # heatmap.
        scalar_depth = np.ndim(depths) == 0
        z_axis = (None if scalar_depth
                  else np.atleast_1d(np.asarray(depths, dtype=float)))
        if scalar_depth:
            z_eval = np.array([float(depths)], dtype=float)
        else:
            z_eval = z_axis
        self._check_depth_range(z_eval)
        # dB/m out of the per-model kernel, one column per frequency.
        grid = self._alpha_dB_per_m_at_sound_speed(f, z_eval,
                                                   float(sound_speed))
        carrier = AbsorptionCoefficient(
            frequencies=f,
            data=grid if z_axis is not None else grid[0, :],
            units='dB/m', model=self._model_name(), depths=z_axis,
            depth=None if z_axis is not None else float(z_eval[0]),
            parameters=self._carrier_parameters(),
            data_sources=self.data_sources)
        return carrier.to_units(units, sound_speed=sound_speed)

    def _model_name(self) -> str:
        """The name this model records on the carrier it produces."""
        return type(self).__name__.lower()

    def _carrier_parameters(self) -> Optional[Dict[str, Any]]:
        """The ocean values this model records on the carrier it produces
        (:attr:`AbsorptionCoefficient.parameters`); ``None`` when it takes
        none."""
        return None

    def _repr_bits(self) -> List[str]:
        """The model's values in repr words: its ocean values, by
        default."""
        return _ocean_bits(self._carrier_parameters() or {})

    def __repr__(self) -> str:
        return build(type(self).__name__, self._repr_bits())

    def _alpha_dB_per_m(
        self,
        frequencies: np.ndarray,
        depths: np.ndarray,
    ) -> np.ndarray:
        """Model-specific ``α(f, z)`` in dB/m on 1-D ``frequencies`` (Hz,
        already checked positive) and 1-D ``depths`` (m), as an
        ``(n_depths, n_frequencies)`` grid."""
        raise NotImplementedError

    def _alpha_dB_per_m_at_sound_speed(
        self,
        frequencies: np.ndarray,
        depths: np.ndarray,
        sound_speed: float,
    ) -> np.ndarray:
        """``α(f, z)`` in dB/m in water of ``sound_speed``, the
        ``(n_depths, n_frequencies)`` grid :meth:`table` evaluates, so its
        ``sound_speed`` reaches the kernel and not only the output conversion.
        The same as :meth:`_alpha_dB_per_m` for every model stated in dB/km;
        :class:`ConstantAbsorption` and a table, which may be stated per
        wavelength, override it."""
        return self._alpha_dB_per_m(frequencies, depths)

    def _alpha_dB_per_wavelength_at_frequency(
        self,
        frequency: float,
        depths: _ArrayLike,
        sound_speeds: _ArrayLike,
    ) -> np.ndarray:
        """``α(f, z)`` in dB per *local* wavelength at each node — the unit an
        Acoustics-Toolbox SSP row's ``alphaI`` and a RAM water block carry,
        which the solver turns back into a loss at that node's own sound
        speed (``misc/AttenMod.f90:73``).

        ``α[dB/m](f, z) · c(z) / f`` for a law stated per metre; a law stated
        per wavelength (:class:`ConstantAbsorption`, a table in
        dB/wavelength) returns its value, with no round trip through a
        reference sound speed.

        Parameters
        ----------
        frequency : float
            Frequency (Hz), > 0 (checked as
            :meth:`_alpha_dB_per_m_at_frequency` checks it).
        depths : float or 1-D array
            Node depths (m).
        sound_speeds : float or 1-D array
            Sound speed (m/s) at each node, or one for all.

        Returns
        -------
        ndarray, the shape of ``depths`` (a scalar depth as shape ``(1,)``).
        """
        f = float(frequency)
        c = np.asarray(sound_speeds, dtype=float)
        return np.atleast_1d(self._alpha_dB_per_m_at_frequency(f, depths)) * c / f

    @property
    def _scales_by_frequency_ratio(self) -> bool:
        """Whether a loss traced at ``f_t`` moves to ``f`` by the law's ratio
        ``α(f) / α(f_t)`` at the surface (:func:`arrival_absorption_exponent`)
        rather than linearly in ``f``: a law whose frequency dependence is
        one shape at every depth — exactly (Thorp) or up to the pressure terms
        (one Francois-Garrison water row). The ratio's error over the column
        is measured by :func:`_band_absorption_error_dB_per_km` with
        ``by_ratio=True``. False unless a law says so."""
        return False

    @property
    def _needs_node_sound_speed(self) -> bool:
        """Whether an engine takes the law through
        :meth:`_alpha_dB_per_wavelength_at_frequency` node by node, at each node's own
        sound speed, rather than through one dB/m table: a law stated per
        wavelength (its dB/m depends on the speed it is converted at) and a
        table (whose units may be)."""
        return False

    def _breakpoint_depths(self) -> np.ndarray:
        """Depths (m) where ``α`` changes slope or steps — a biological
        layer's edges, a profile's rows — which an engine sampling the law
        on its own grid adds to it. Empty for a law smooth in depth."""
        return np.empty(0)

    def _short(self) -> str:
        """The law in an :class:`~uacpy.core.environment.Environment` repr:
        its class name."""
        return type(self).__name__


@carrier
class Thorp(Absorption):
    """Thorp (1967) seawater volume attenuation. No parameters.

    Frequency-only — α(f, z) is constant in depth.
    """

    def _alpha_dB_per_m(
        self,
        frequencies: np.ndarray,
        depths: np.ndarray,
    ) -> np.ndarray:
        return absorption_thorp(frequencies[None, :],
                                depth=depths[:, None]) / 1000.0

    @property
    def _scales_by_frequency_ratio(self) -> bool:
        return True


#: The envelope Francois & Garrison (1982, Part II, §III) fitted and
#: tabulated: Table IV runs -1.8 to 30 °C at 30 and 35 ‰; the salinity range
#: is the union of the two salinity-dependent terms' data, the boric-acid
#: measurements at 34-41 ‰ and the MgSO4 field data at 30-35 ‰ (APL-UW TR 9407
#: §I.B); the seawater pH range is 7.7-8.3 (Mellen et al. 1987; TR 9407 puts
#: the "extreme" values outside 7.7-8.2). Inclusive on both ends.
_FG_TEMPERATURE_RANGE_C = (-1.8, 30.0)
_FG_SALINITY_RANGE_PSU = (30.0, 41.0)
_FG_PH_RANGE = (7.7, 8.3)
#: Hz. "The equation may not hold below 200 Hz, where the boric acid
#: contribution may be exceeded by a scattering loss" (Part II, §III); Table
#: IV stops at 1000 kHz, and the MgSO4 term is verified to 600 kHz (TR 9407
#: §I.B); above that the pure-water term carries over 90 % of the loss.
_FG_FREQUENCY_RANGE_HZ = (200.0, 1.0e6)


def _lowest(value):
    """A number as given; a profile by its lowest value, for a refusal."""
    if np.ndim(value) == 0:
        return value
    return (f"{float(np.min(value)):g} (the lowest of {np.size(value)} "
            f"profile values)")


def _g(value) -> str:
    """A number in ``:g`` form; a profile by its span."""
    return f"{float(value):g}" if np.ndim(value) == 0 else _span(value)


def _outside(values, low: float, high: float) -> str:
    """The value outside ``low..high`` in ``:g`` form: the number itself,
    or a profile's first value outside."""
    if np.ndim(values) == 0:
        return f"{float(values):g}"
    arr = np.asarray(values, dtype=float).ravel()
    return f"{arr[~((low <= arr) & (arr <= high))][0]:g}"


@carrier
class FrancoisGarrison(Absorption):
    """Francois–Garrison (1982) seawater absorption.

    ``temperature``, ``salinity`` and ``pH`` are the water, with the names
    and the reference-water defaults (10 °C, 35 PSU, pH 8) of
    :func:`absorption_francois_garrison`. The formula's depth (pressure)
    term varies down the column, so the
    law has no depth of its own: every evaluation takes the depth it is
    evaluated at — :meth:`table` at the depths asked or, given none, at the
    surface (0 m) for one water row and at the profile's own depths for a
    profile — and every engine sees the same α(z). ``data_sources``
    (keyword-only) records where fetched water came from
    (:attr:`Absorption.data_sources`); ``uacpy.data.fetch_environment``
    sets it.

    Notes
    -----
    The water is *refused* only where the formula itself has no
    value there (see :func:`absorption_francois_garrison`): the
    boric-acid relaxation takes ``sqrt(S/35)``, its temperature factor
    is ``10**(4 - 1245/(T + 273))``, and all three mechanisms divide by
    the sound speed ``c = 1412 + 3.21·T + 1.19·S + 0.0167·z``. Neither
    ``misc/AttenMod.f90`` (``Franc_Garr``) nor ``Matlab/Misc/franc_garr.m``
    — the two implementations this one follows — checks a validity
    envelope, so none is enforced here either.

    **Fitted envelope.** Outside the range the equation was fitted and
    tabulated over, a ``ValidityWarning`` names the field and the range, and
    the value is used as given. Francois & Garrison (1982, Part II, §III
    "Recommended absorption equation") determined the boric-acid term from
    measurements at 34–41 ‰ and 2–22 °C to 1500 m, tabulate the equation
    (Table IV) for −1.8 to 30 °C, 0.4–1000 kHz, at 30 and 35 ‰, and state
    that it "may not hold below 200 Hz, where the boric acid contribution
    may be exceeded by a scattering loss"; the MgSO4 term comes from field
    data at 30–35 ‰ and 2–22 °C (APL-UW TR 9407 §I.B). The constructor
    therefore warns for ``temperature`` outside −1.8..30 °C,
    ``salinity`` outside 30..41, the union of the two terms' data (a
    Mediterranean 38.5 ‰ lies inside the boric-acid data though past the
    MgSO4 term's 30–35 ‰; a Baltic 7 ‰ lies outside both, and the boric term
    then extrapolates in ``sqrt(S/35)``), and ``pH``
    outside 7.7..8.3 (the seawater range, Mellen et al. 1987; TR 9407 warns
    of discrepancies "as high as 40 % below 1 kHz" for pH under 7.7 or over
    8.2); :meth:`table` and every engine's evaluation warn for a frequency under 200 Hz or over
    1 MHz. The authors quote 5 % accuracy inside the measured range and
    about 10 % outside their frequency range; TR 9407 §I.B adds that the
    absorption in a particular area "may vary by ±10 % from the given
    equation".

    The ``__init__`` is written out (``init=False``) so the envelope warning
    names the caller's line whether the model is built by hand or by
    :meth:`from_temperature_salinity` from a fetched T/S/pH column: a
    generated ``__init__`` lives in the pseudo-file ``<string>``, which the
    attribution walk cannot step over. ``@dataclass`` still supplies
    ``__repr__`` / ``__eq__`` / ``fields()`` from the annotations; a test
    pins the signature against them.

    **Every engine sees the same α(z)**: the formula at each depth, with the
    water there. The Acoustics-Toolbox decks carry it in every water SSP
    row's ``alphaI`` at the deck frequency
    (:func:`uacpy.io.at_codes.writes_alpha_per_ssp_row`), RAM in its water
    block, OASES averaged over each water layer. One exception, where the
    rows would freeze the law at one frequency: a deck covering several
    frequencies (a Kraken or Scooter broadband deck, ``TopOpt(6)='B'``)
    carries one water row as AT's ``'F'``, exact in frequency, evaluated at
    mid-water column and applied at every depth, and in the sediment too
    (``misc/AttenMod.f90:84-110,148-160``); the writer warns when that one
    depth departs from the formula by the band rule
    (:func:`uacpy.io.oalib_writer.warn_if_francois_garrison_depth_frozen`).

    **A T/S profile.** The law has no evaluation depth of its own, so
    ``temperature``, ``salinity`` and ``pH`` each take, mixed freely, a
    single value (constant water) or ``(depth, value)`` pairs, an
    ``(M, 2)`` array — a profile on depths of its own (m, finite,
    non-negative, strictly increasing), interpolated at each depth
    evaluated, linear between its samples and held at its ends; the
    formula's own depth term takes the depth asked. Any other array is
    refused: a 1-D array of values has no depths to stand on.
    :meth:`from_temperature_salinity` builds the law from a column of arrays
    on one depth axis. Each profile is stored as a profile record, which the
    law hands back (``law.temperature``) and every water-property input
    takes again. The fitted-envelope warning is checked value by value,
    once per law; :meth:`table` and every engine's evaluation also announce,
    once per call, a frequency outside 200 Hz-1 MHz and a depth beyond the
    5000 m the equation is stated for (Part II, §IV). RAM samples every
    property's depths
    (:attr:`profile_depths`).

    **pH scale.** ``pH`` is taken on ``ph_scale`` — ``'nbs'`` (default, the
    scale the equation was fitted on, so the number is used as given),
    ``'total'`` or ``'seawater'`` — and converted once by :func:`ph_to_nbs`;
    :attr:`ph_nbs` is what the formula is evaluated on. GLODAP and the
    Copernicus BGC field are on the total scale, and the environment builder
    says so; a hand-typed ``pH=8.0`` stays on NBS.
    """
    _FREQUENCY_RANGE_HZ = _FG_FREQUENCY_RANGE_HZ
    _FREQUENCY_RANGE_NOTE = (
        "Francois & Garrison 1982 Part II, §III: it \"may not hold below "
        "200 Hz\", and Table IV stops at 1000 kHz. The polynomial is "
        "evaluated as given; below 200 Hz a scattering loss the equation "
        "omits can exceed the boric-acid term.")
    _DEPTH_RANGE_M = (0.0, 5000.0)
    _DEPTH_RANGE_NOTE = (
        "Francois & Garrison 1982 Part II, §IV: the equation predicts "
        "absorption \"to depths of 5000 m\" (its boric-acid term rests on "
        "measurements to 1500 m, §III). The pressure terms are evaluated as "
        "given.")

    # Each water property takes a single value (constant water) or
    # (depth, value) pairs, an (M, 2) array; a law has no evaluation depth,
    # so an array of values alone is refused (_profile_values).
    temperature: Union[float, np.ndarray] = REFERENCE_TEMPERATURE
    salinity: Union[float, np.ndarray] = REFERENCE_SALINITY
    pH: Union[float, np.ndarray] = REFERENCE_PH
    ph_scale: str = 'nbs'

    _WATER = ('temperature', 'salinity', 'pH')

    def _profile_values(self) -> None:
        """Store each water property as a number, or its ``(depth, value)``
        pairs as a :class:`~uacpy.core._water.WaterProfile`
        (:func:`~uacpy.core._validate.water_pairs`; a WaterProfile the law
        handed out is kept, its depths non-negative); refuse every other
        form. The law has no evaluation depth, so an array cannot be one
        value per point evaluated."""
        for name in self._WATER:
            value = getattr(self, name)
            if isinstance(value, WaterProfile):
                if np.any(value.depths < 0):
                    raise ConfigurationError(
                        f"FrancoisGarrison: the depths of {name} must be "
                        f"non-negative (m, positive down); got "
                        f"{value.depths.tolist()}.")
                continue
            shape = water_shape(value, name=name, who='FrancoisGarrison')
            if len(shape) == 0:
                setattr(self, name, float(value))
                continue
            if len(shape) == 2 and shape[1] == 2:
                setattr(self, name, water_pairs(value, name=name,
                                                who='FrancoisGarrison'))
                continue
            raise ConfigurationError(
                f"FrancoisGarrison: {name} has shape {shape}, and the law "
                f"has no evaluation depth of its own. A water property of "
                f"the law is a single value (constant water) or (depth, "
                f"value) pairs of shape (M, 2) (a profile on depths of its "
                f"own, interpolated onto each depth the law is evaluated "
                f"at); an array of values has no depths to stand on.",
                remediation="Pass np.column_stack([depths, values]) pairs "
                            "for a profile, or a single value for constant "
                            "water; FrancoisGarrison.from_temperature_"
                            "salinity(depths, T, S) builds the law from "
                            "arrays on one depth axis.")

    def _values(self, name: str) -> Union[float, np.ndarray]:
        """A water property's value: its number, or its profile's
        values."""
        value = getattr(self, name)
        return value.values if isinstance(value, WaterProfile) else value

    def _local(self, name: str, depths) -> np.ndarray:
        """A water property at ``depths`` by the shared water-property rule
        (:func:`~uacpy.core._validate.water_property`): its number, or its
        profile linear between its depths and held at the end values beyond
        them."""
        z = np.atleast_1d(np.asarray(depths, dtype=float))
        return np.broadcast_to(np.asarray(water_property(
            getattr(self, name), z, name=name, who='FrancoisGarrison'),
            dtype=float), z.shape).copy()

    def _nbs_at(self, depths) -> np.ndarray:
        """The NBS-scale pH at ``depths``, converted with the water there."""
        return np.asarray(ph_to_nbs(
            self._local('pH', depths), self.ph_scale,
            temperature=self._local('temperature', depths),
            salinity=self._local('salinity', depths)), dtype=float)

    def __post_init__(self):
        Absorption.__post_init__(self)
        self._profile_values()
        for name in self._WATER:
            require_finite(self._values(name), f"FrancoisGarrison: {name}")
        if self.ph_scale not in PH_SCALES:
            raise ConfigurationError(
                f"FrancoisGarrison: ph_scale must be one of {PH_SCALES}; "
                f"got {self.ph_scale!r}.",
                remediation="GLODAP and Copernicus BGC pH are 'total'; a "
                            "value typed for the formula itself is 'nbs'.",
            )
        salinity = self._values('salinity')
        temperature = self._values('temperature')
        if not np.all(np.asarray(salinity) >= 0):
            raise ConfigurationError(
                f"FrancoisGarrison: salinity must be non-negative (PSU); "
                f"got {_lowest(salinity)}. The boric-acid relaxation "
                f"frequency carries sqrt(S/35), which has no value below 0."
            )
        if not np.all(np.asarray(temperature) > -273.0):
            raise ConfigurationError(
                f"FrancoisGarrison: temperature must be above -273 (°C, "
                f"not kelvin); got {_lowest(temperature)}. The "
                f"relaxation frequencies carry 10**(4 - 1245/(T + 273)), "
                f"which is singular at -273 °C."
            )
        # With T > -273 °C, S >= 0 and a depth z >= 0 the formula's own sound speed
        # c = 1412 + 3.21·T + 1.19·S + 0.0167·z stays above 535 m/s, so every
        # absorption mechanism dividing by it is defined.
        ph = np.asarray(self._values('pH'))
        if not np.all((PH_MIN <= ph) & (ph <= PH_MAX)):
            raise ConfigurationError(
                f"FrancoisGarrison: pH={_outside(ph, PH_MIN, PH_MAX)} is "
                f"outside 0..14, which no water reaches (seawater runs about "
                f"7.5-8.5)."
            )
        # The value the equation is evaluated on. ph_to_nbs adds 0.06-0.18
        # to a 'total' / 'seawater' pH over -2..40 °C and 0..45 PSU, so an
        # input just under 14 can land past it, and a temperature where
        # the conversion's activity fit goes non-positive gives NaN.
        nbs = np.asarray(self.ph_nbs)
        if not np.all((PH_MIN <= nbs) & (nbs <= PH_MAX)):
            raise ConfigurationError(
                f"FrancoisGarrison: pH={_g(self._values('pH'))} on the "
                f"{self.ph_scale!r} scale is {_outside(nbs, PH_MIN, PH_MAX)} "
                f"on the NBS scale the formula is evaluated on (ph_to_nbs at "
                f"T={_g(temperature)} °C, "
                f"S={_g(salinity)} PSU), outside 0..14."
            )
        # The fitted envelope, checked after every rule that raises. One
        # warning naming every field outside it, so a fetched row that is
        # out on two axes is reported once; a profile is checked value by
        # value and names how many of its depths are out.
        outside = []
        ph_label = ('pH' if self.ph_scale == 'nbs'
                    else f"pH (NBS, from {_g(self._values('pH'))} "
                         f"{self.ph_scale!r})")
        for label, value, (low, high), unit, basis in (
                ('temperature', temperature,
                 _FG_TEMPERATURE_RANGE_C, '°C',
                 'the span Table IV tabulates'),
                ('salinity', salinity,
                 _FG_SALINITY_RANGE_PSU, 'PSU',
                 'the union of the data the two salinity terms were fitted '
                 'to, APL-UW TR 9407 §I.B: boric acid at 34-41, MgSO4 at '
                 '30-35'),
                (ph_label, self.ph_nbs, _FG_PH_RANGE, 'on the NBS scale',
                 'the seawater range')):
            values = np.atleast_1d(np.asarray(value, dtype=float))
            out = values[~((low <= values) & (values <= high))]
            if not out.size:
                continue
            if np.ndim(value) == 0:
                outside.append(
                    f"{label}={float(value):g} is outside {low:g}..{high:g} "
                    f"{unit} ({basis})")
            else:
                outside.append(
                    f"{label}={_span(out)} at {out.size} of {values.size} "
                    f"depths is outside {low:g}..{high:g} {unit} ({basis})")
        if outside:
            warnings.warn(
                f"FrancoisGarrison: {'; '.join(outside)}. Francois & "
                f"Garrison 1982 Part II, §III quote 5 % accuracy over the "
                f"parameters their measurements cover; the value is used as "
                f"given, extrapolating the fit.",
                ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)

    @classmethod
    def from_temperature_salinity(cls, depths, temperature, salinity, *,
                                  pH=REFERENCE_PH, ph_scale: str = 'nbs',
                                  collapse_to_depth: Optional[float] = None
                                  ) -> 'FrancoisGarrison':
        """The law from a temperature/salinity column, as the data fetchers
        return one: the whole column as a profile, or the one water row
        nearest ``collapse_to_depth``.

        Parameters
        ----------
        depths : array-like
            The column's depths (m), a non-empty 1-D array.
        temperature, salinity : float or array-like
            In-situ temperature (°C) and practical salinity (psu), each by
            the water-property rule on ``depths``
            (:func:`~uacpy.core._validate.water_property`): a single value,
            an array of exactly ``depths``' shape (one value per depth), or
            ``(depth, value)`` pairs of shape ``(M, 2)`` on depths of their
            own. Depths where an array of values holds a missing value are
            dropped, and repeated depths keep their first sample; each array
            of values becomes the profile the law stores.
        pH : float or array-like, optional
            Seawater pH by the same rule: one number (default 8.0, the
            package's reference sea water, on the formula's own NBS scale),
            an array on ``depths``, or ``(depth, pH)`` pairs on their own
            depths, as a GLODAP column gives them.
        ph_scale : {'nbs', 'total', 'seawater'}, optional
            The scale ``pH`` is on. Francois & Garrison (1982, Part II) took
            their pH from Lovett's (1980) charts of the Gorshkov (1978)
            atlas, whose scale is not reported; Brewer & Hester (Oceanography
            22(4), 2009, p. 91-92) judge it "probably" NBS and state that
            "the sound absorption equations are based on the old NBS scale".
            GLODAP and the Copernicus BGC field report the **total** scale,
            about 0.1 lower for the same water; pass ``'total'`` for them and
            the value is moved to NBS by :func:`ph_to_nbs` (Takahashi 1982
            ``fH``: +0.10 at 4 °C, +0.15 at 25 °C at S = 35). The shift
            raises the boric-acid term, and so the whole absorption below
            1 kHz, by about 20 %: 1 dB per 100 km at 1 kHz, 5-6 dB over a
            3000 km basin path at 300 Hz, under 2 dB per 100 km at any
            frequency.
        collapse_to_depth : float, optional
            Default ``None`` keeps the **whole column**: every engine
            evaluates the formula at each depth with the water there. A
            depth (m) instead keeps the T/S row nearest it, and the pH there:
            that row's water then stands for the whole column (the formula's
            depth term still takes each depth evaluated), which on a
            mid-latitude column (22 °C surface, 4 °C at 2 km) moves the
            absorption by tens of percent against the profile at the ends of
            the column.
        """
        who = 'FrancoisGarrison.from_temperature_salinity'
        z = np.asarray(depths, dtype=float)
        if z.ndim != 1 or z.size == 0:
            raise ConfigurationError(
                f"{who}: depths must be a non-empty 1-D array (m); got shape "
                f"{z.shape}.")
        given = {'temperature': temperature, 'salinity': salinity, 'pH': pH}
        on_z = {name: np.broadcast_to(np.asarray(water_property(
            value, z, name=name, who=who), dtype=float), z.shape)
            for name, value in given.items()}
        if collapse_to_depth is not None:
            i = int(np.argmin(np.abs(z - float(collapse_to_depth))))
            return cls(**{name: float(v[i]) for name, v in on_z.items()},
                       ph_scale=ph_scale)
        arrays = [name for name, value in given.items()
                  if not isinstance(value, WaterProfile)
                  and np.ndim(value) and np.shape(value) == z.shape]
        keep = np.isfinite(z)
        for name in arrays:
            keep &= np.isfinite(on_z[name])
        z_u, first = np.unique(z[keep], return_index=True)
        if arrays and z_u.size == 0:
            raise ConfigurationError(
                f"{who}: the column holds no depth with a finite value of "
                f"every array given ({', '.join(arrays)}).")

        def stored(name):
            value = given[name]
            if name in arrays:
                return WaterProfile(z_u, on_z[name][keep][first])
            if isinstance(value, WaterProfile) or np.ndim(value) == 0:
                return value
            return water_pairs(value, name=name, who=who)
        return cls(**{name: stored(name) for name in given},
                   ph_scale=ph_scale)

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return all(getattr(self, f) == getattr(other, f)
                   for f in self.__dataclass_fields__)

    __hash__ = None

    @property
    def is_profile(self) -> bool:
        """Whether any water property varies with depth rather than the
        water being one row."""
        return any(isinstance(getattr(self, n), WaterProfile)
                   for n in self._WATER)

    @property
    def profile_depths(self) -> Optional[np.ndarray]:
        """The union of the water properties' depths (m, increasing), or
        ``None`` for one water row."""
        axes = [getattr(self, n).depths for n in self._WATER
                if isinstance(getattr(self, n), WaterProfile)]
        return np.unique(np.concatenate(axes)) if axes else None

    @property
    def ph_nbs(self) -> Union[float, np.ndarray]:
        """``pH`` on the NBS scale — the value the equation is evaluated on;
        for a profile, at each of :attr:`profile_depths` with the water
        there."""
        if not self.is_profile:
            return float(ph_to_nbs(self.pH, self.ph_scale,
                                   temperature=self.temperature,
                                   salinity=self.salinity))
        return self._nbs_at(self.profile_depths)

    @property
    def _table_depths(self) -> _ArrayLike:
        if self.is_profile:
            return self.profile_depths
        return super()._table_depths

    @property
    def _scales_by_frequency_ratio(self) -> bool:
        return not self.is_profile

    def _breakpoint_depths(self) -> np.ndarray:
        return (self.profile_depths if self.is_profile else np.empty(0))

    def _model_name(self) -> str:
        return 'francois_garrison'

    def _carrier_parameters(self) -> Dict[str, Any]:
        # A WaterProfile is immutable, so the table holds the law's own.
        def value(v):
            return v if isinstance(v, WaterProfile) else float(v)
        return {'temperature': value(self.temperature),
                'salinity': value(self.salinity), 'pH': value(self.pH),
                'ph_scale': self.ph_scale}

    def _alpha_dB_per_m(
        self,
        frequencies: np.ndarray,
        depths: np.ndarray,
    ) -> np.ndarray:
        f = frequencies[None, :]
        z = depths[:, None]
        if not self.is_profile:
            a_km = absorption_francois_garrison(
                frequency=f,
                temperature=self.temperature,
                salinity=self.salinity,
                pH=self.ph_nbs,
                depth=z,
            )
            return a_km / 1000.0
        # Each property at each asked depth on its own axis, linear between
        # its samples and held at its ends; the formula's own depth term
        # takes the asked depth.
        a_km = absorption_francois_garrison(
            frequency=f,
            temperature=self._local('temperature', depths)[:, None],
            salinity=self._local('salinity', depths)[:, None],
            pH=self._nbs_at(depths)[:, None],
            depth=z,
        )
        return a_km / 1000.0

@carrier
class BiologicalLayer(RevalidateOnAssignMixin, DeepCopyMixin, CarrierExport):
    """Single fish-bladder resonance layer for :class:`Biological`.

    Parameters
    ----------
    z_top, z_bottom : float
        Depth bounds (m) of the layer in the water column.
    f0 : float
        Resonance frequency (Hz).
    Q : float
        Quality factor (dimensionless).
    a0 : float
        Acoustics-Toolbox resonance coefficient (dB/km): the absorption is
        ``a0 / ((1 - f0²/f²)² + 1/Q²)``, so the peak at ``f = f0`` is
        ``a0·Q²`` (see AttenMod.f90).

    Notes
    -----
    ``a0·Q²`` is checked against the AT solvers' ``CRCI`` ceiling and a
    ``ValidityWarning`` names the limit when it is over. ``AttenMod.f90``'s
    ``'B'`` branch (:105-106) adds ``a/8685.8896`` Nepers/m to ``alphaT``,
    :113 scales by ``c²/ω`` and :116 aborts the run when the result exceeds
    ``c`` — so a layer aborts once its dB/km absorption passes
    ``8685.8896·2πf/c``, which is 3638 dB/km at 100 Hz in 1500 m/s water
    (``a0 = 1, Q = 60`` gives 3600 and clears it; ``Q = 61`` gives 3721 and
    warns). This warns rather than raises because
    the peak is only reached when the run frequency sits on ``f0``, and the
    ceiling scales with the true ``c(z)`` over the layer, which a layer on
    its own does not carry — the check uses
    :data:`~uacpy.core.constants.DEFAULT_SOUND_SPEED`.

    The ceiling warning names the line the *user* wrote, whether the layer
    is built directly, from a tuple inside :class:`Biological`'s
    constructor one frame further down, or rebuilt by an assignment: every
    frame between the warning and the user is an ordinary package frame
    (``@carrier``'s ``__init__`` included) that :data:`USER_FRAME_SKIP`
    steps over.
    """
    _REPR_UNITS = {'z_top': 'm', 'z_bottom': 'm', 'f0': 'Hz'}

    z_top: float
    z_bottom: float
    f0: float
    Q: float
    a0: float

    def __repr__(self) -> str:
        return build('BiologicalLayer', [
            f"{num(self.z_top)}–{num(self.z_bottom)} m",
            f"f0={qty(self.f0, 'Hz')}", f"Q={num(self.Q)}",
            f"a0={qty(self.a0, 'dB/km')}"])

    def __post_init__(self):
        # Finiteness first, because every sign test below is a bare ``<=``
        # that NaN answers False to and that inf passes for ``a0``/``Q``: a
        # NaN layer reaches the AT deck through ``biological_records`` and turns
        # AttenMod.f90's band test ``z >= Z1 .AND. z <= Z2`` (:104) False at
        # every depth, so it is written to the file and then contributes
        # nothing, while an inf ``a0`` reaches the ceiling arithmetic below
        # and reports its own limit as ``nan dB/km``. A negative depth is
        # inert the same way — the band test compares against a water-column
        # depth measured down from the surface — so the depths are held to
        # ``>= 0`` here rather than only to their ordering. This is the same
        # pair of guards every other core carrier routes through.
        require_non_negative(self.z_top, "BiologicalLayer.z_top",
                             hint="metres below the surface")
        require_non_negative(self.z_bottom, "BiologicalLayer.z_bottom",
                             hint="metres below the surface")
        require_finite(self.f0, "BiologicalLayer.f0", hint="Hz")
        require_finite(self.Q, "BiologicalLayer.Q", hint="dimensionless")
        require_finite(self.a0, "BiologicalLayer.a0", hint="dB/km")
        if self.z_bottom <= self.z_top:
            raise ConfigurationError(
                "BiologicalLayer: z_bottom must be strictly greater than "
                f"z_top (got z_top={self.z_top}, "
                f"z_bottom={self.z_bottom})"
            )
        if self.f0 <= 0:
            raise ConfigurationError(
                f"BiologicalLayer: f0 must be positive (Hz); got {self.f0}."
            )
        if self.Q <= 0:
            raise ConfigurationError(
                f"BiologicalLayer: Q must be positive (dimensionless); got {self.Q}."
            )
        if self.a0 <= 0:
            raise ConfigurationError(
                f"BiologicalLayer: a0 must be positive (dB/km); got {self.a0}."
            )
        # The Lorentzian peaks at f = f0, where the denominator is 1/Q², so
        # a0*Q² is the most absorption this layer can present to CRCI. Taken
        # to dB/wavelength at f0 it meets the same package-wide ceiling the
        # seabed and surface carriers are held to.
        peak_dB_km = float(self.a0) * float(self.Q) ** 2
        peak_dB_lambda = float(convert_attenuation_units(
            peak_dB_km, float(self.f0), 'dB/km', 'dB/wavelength',
            sound_speed=DEFAULT_SOUND_SPEED))
        if peak_dB_lambda > MAX_ATTENUATION:
            ceiling_dB_km = peak_dB_km * (
                MAX_ATTENUATION / peak_dB_lambda)
            warnings.warn(
                f"BiologicalLayer: the on-resonance peak a0*Q² = "
                f"{peak_dB_km:g} dB/km is {peak_dB_lambda:g} dB/wavelength at "
                f"f0={self.f0:g} Hz in {DEFAULT_SOUND_SPEED:g} m/s water, "
                f"over the {MAX_ATTENUATION:.4f} above which "
                f"misc/AttenMod.f90's CRCI (:116) finds an imaginary sound "
                f"speed larger than the real part and aborts. A run at or near "
                f"f0 will fail in every AT solver; the ceiling here is "
                f"{ceiling_dB_km:g} dB/km, and scales with the water sound "
                f"speed over the layer.",
                # The walk, not a count: this constructor is reached from a
                # user's ``BiologicalLayer(...)`` and from the normalising
                # loop in ``Biological.__post_init__`` one frame further down, and
                # a hand count is right for only one of the two. Naming a
                # uacpy line is not merely untidy — ``warnings`` keys its
                # once-per-location registry on the attributed file and line,
                # so every ``Biological(...)`` in a program would collapse
                # onto the loop's line and only the first would be shown.
                ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)


#: Constructor sentinel for ``layers``: ``None`` means "none given", which
#: ``__post_init__`` refuses. Typed ``Any`` so the field can declare the list
#: every attribute read sees.
_LAYERS_NOT_GIVEN: Any = None


# The constructor takes tuples as well as layers, and ``None``.
@carrier(init_annotations=dict(
    layers='Optional[List[Union[BiologicalLayer, Tuple]]]'))
class Biological(Absorption):
    """Layered biological volume attenuation (fish-bladder resonance).

    Parameters
    ----------
    layers : list of BiologicalLayer or tuples
        Each entry can be a :class:`BiologicalLayer` or a 5-tuple
        ``(z_top, z_bottom, f0, Q, a0)``.
    data_sources : tuple of DataProvenance, keyword-only
        Where the layers' numbers came from (:attr:`Absorption.data_sources`).
        Default ``()``.

    """
    layers: List[BiologicalLayer] = _LAYERS_NOT_GIVEN

    def __post_init__(self):
        Absorption.__post_init__(self)
        normalized: List[BiologicalLayer] = []
        for index, entry in enumerate(self.layers or []):
            if isinstance(entry, BiologicalLayer):
                normalized.append(entry)
            else:
                if np.ndim(entry) != 1 or len(entry) != 5:
                    raise ConfigurationError(
                        f"Biological: layers[{index}] is {entry!r}; a layer "
                        f"is a BiologicalLayer or the 5 values (z_top, "
                        f"z_bottom, f0, Q, a0) — depths in m, f0 in Hz, a0 in "
                        f"dB/km.")
                z_top, z_bottom, f0, Q, a0 = entry
                normalized.append(BiologicalLayer(
                    z_top=float(z_top), z_bottom=float(z_bottom),
                    f0=float(f0), Q=float(Q), a0=float(a0),
                ))
        if not normalized:
            raise ConfigurationError(
                "Biological absorption requires at least one layer; got 0."
            )
        self.layers = normalized

    def _repr_bits(self) -> List[str]:
        n = len(self.layers)
        if not n:
            return ['no layers']
        top = min(layer.z_top for layer in self.layers)
        bottom = max(layer.z_bottom for layer in self.layers)
        return [f"{n} layer{'s' if n > 1 else ''} {num(top)}–{num(bottom)} m"]

    def _carrier_parameters(self) -> Dict[str, Any]:
        return {'layers': [(float(layer.z_top), float(layer.z_bottom),
                            float(layer.f0), float(layer.Q),
                            float(layer.a0)) for layer in self.layers]}

    def _breakpoint_depths(self) -> np.ndarray:
        return np.array(sorted({float(z) for layer in self.layers
                                for z in (layer.z_top, layer.z_bottom)}))

    def _alpha_dB_per_m(
        self,
        frequencies: np.ndarray,
        depths: np.ndarray,
    ) -> np.ndarray:
        # Sum each layer's resonance (:func:`absorption_biological`) over the
        # depths it spans.
        z = depths
        a_km = np.zeros((z.size, frequencies.size), dtype=float)
        # Each layer spans the closed interval [z_top, z_bottom] and is
        # tested independently, exactly as the AttenMod.f90:102-109 loop
        # tests ``z >= Z1 .AND. z <= Z2`` per layer and sums — so a depth
        # exactly on a boundary two stacked layers share receives both
        # layers' contributions.
        for layer in self.layers:
            in_layer = (z >= layer.z_top) & (z <= layer.z_bottom)
            a_km[in_layer, :] += absorption_biological(
                frequencies, layer.f0, layer.Q, layer.a0)
        return a_km / 1000.0


@carrier
class ConstantAbsorption(Absorption):
    """Frequency-independent baseline absorption written into the SSP
    block's ``alphaI`` column at every depth (dB/wavelength).

    Parameters
    ----------
    value : float
        Absorption coefficient (dB/wavelength). Non-negative.
    data_sources : tuple of DataProvenance, keyword-only
        Where the value came from (:attr:`Absorption.data_sources`). Default
        ``()``.

    Notes
    -----
    dB/wavelength is the unit the deck carries, so the value written to the
    ``alphaI`` column is exact and the divergence is only in the law's dB/m
    evaluation (:meth:`_alpha_dB_per_m_at_frequency`, what the modal
    perturbation reads). That evaluation holds no SSP, so it converts to
    dB/m at :data:`~uacpy.core.constants.DEFAULT_SOUND_SPEED`, while the
    solver converts at each SSP row's own ``c``
    (``misc/AttenMod.f90:73``, the ``'W'`` branch: ``alphaT = alpha * freq /
    (8.6858896 * c)``). Over sound speeds of 1450-1550 m/s that is a spread
    of ±3.3 % between the two answers. :meth:`table` takes a
    ``sound_speed`` and converts at it, so ``table(f,
    units='dB/wavelength', sound_speed=c)`` returns the value exactly at any
    ``c``; see
    :meth:`uacpy.core.results.modes.Modes.with_attenuation` for where the
    difference is felt.
    """
    value: float = 0.0

    def _repr_bits(self) -> List[str]:
        return [qty(self.value, 'dB/λ')]

    def __post_init__(self):
        Absorption.__post_init__(self)
        if not (self.value >= 0):
            raise ConfigurationError(
                f"ConstantAbsorption.value must be "
                f"non-negative; got {self.value}."
            )
        warn_attenuation_past_at_ceiling(
            self.value,
            "ConstantAbsorption.value")

    def _model_name(self) -> str:
        return 'constant'

    def _carrier_parameters(self) -> Dict[str, Any]:
        return {'value': float(self.value)}

    @property
    def _needs_node_sound_speed(self) -> bool:
        return True

    def _alpha_dB_per_wavelength_at_frequency(self, frequency, depths, sound_speeds):
        # The value is already per local wavelength: written as given at
        # every node, whatever its sound speed.
        return np.full(np.shape(np.atleast_1d(depths)),
                       float(self.value))

    def _alpha_dB_per_m(
        self,
        frequencies: np.ndarray,
        depths: np.ndarray,
    ) -> np.ndarray:
        # dB/wavelength → dB/m at each frequency (flat in depth). No SSP is
        # carried here, so the conversion uses the reference sound speed.
        return self._alpha_dB_per_m_at_sound_speed(frequencies, depths,
                                                   DEFAULT_SOUND_SPEED)

    def _alpha_dB_per_m_at_sound_speed(
        self,
        frequencies: np.ndarray,
        depths: np.ndarray,
        sound_speed: float,
    ) -> np.ndarray:
        # dB/wavelength → dB/m at the caller's sound speed, so ``table(...,
        # units='dB/wavelength', sound_speed=c)`` returns the value exactly
        # at any c.
        alpha = convert_attenuation_units(
            self.value, frequencies,
            'dB/wavelength', 'dB/m', sound_speed=sound_speed,
        )
        return np.broadcast_to(alpha[None, :],
                               (depths.size, frequencies.size)).copy()


#: Relative tolerance on a table's first and last frequency: a run frequency
#: that rounds onto one of them is inside the table.
_TABLE_FREQUENCY_RTOL = 1e-9

@carrier(eq=False)
class _TabulatedAbsorption(Absorption):
    """A measured α(f, z) as the water's law: an
    :class:`AbsorptionCoefficient` with no law behind it, evaluated the way
    every law is (the rules are in :class:`AbsorptionCoefficient`'s
    docstring). Not public: ``env.absorption`` holds the table itself, and
    :func:`_absorption_law` builds this law over it, held as
    :attr:`measured`, wherever the absorption is evaluated.

    Linear in depth between the rows and held at the first and last row
    beyond them (the Environment says so when its water reaches past them);
    linear in ``log f`` between the frequencies; a frequency outside the
    table is refused. The numbers stay in the table's own units, so a table
    in dB/wavelength reaches a deck exactly at each node's sound speed.
    """
    #: The measured α(f, z) this law interpolates.
    measured: AbsorptionCoefficient = None

    def __post_init__(self):
        Absorption.__post_init__(self)
        t = self.measured
        if not isinstance(t, AbsorptionCoefficient):
            raise ConfigurationError(
                f"tabulated absorption: measured must be an "
                f"AbsorptionCoefficient; got {type(t).__name__}.")
        who = "measured absorption table"
        # An unknown unit is refused by the conversion's own check.
        convert_attenuation_units(1.0, 1000.0, t.units, 'dB/m')
        f = np.atleast_1d(np.array(t.frequencies, dtype=float))
        data = np.array(t.data, dtype=float)
        require_finite(f, f"{who}: frequencies", hint="Hz")
        require_finite(data, f"{who}: data", hint=t.units)
        if f.ndim != 1 or np.any(f <= 0) or np.any(np.diff(f) <= 0):
            raise ConfigurationError(
                f"{who}: frequencies must be positive and strictly "
                f"increasing (Hz); got {f.tolist()}.")
        if np.any(data < 0):
            raise ConfigurationError(
                f"{who}: alpha must be non-negative; got {data.min():g} "
                f"{t.units}. A negative absorption is a gain.")
        depths = None
        if t.depths is not None:
            depths = np.atleast_1d(np.array(t.depths, dtype=float))
            require_finite(depths, f"{who}: depths", hint="m")
            if depths.ndim != 1 or np.any(np.diff(depths) <= 0):
                raise ConfigurationError(
                    f"{who}: depths must be strictly increasing (m); got "
                    f"{depths.tolist()}.")
        expected = (f.size,) if depths is None else (depths.size, f.size)
        if data.shape != expected:
            raise ConfigurationError(
                f"{who}: data has shape {data.shape}; the axes give "
                f"{expected} (depth first, then frequency).")
        self.measured = AbsorptionCoefficient(
            frequencies=f, data=data, units=t.units, model=t.model,
            depths=depths, depth=t.depth, parameters=t.parameters,
            data_sources=t.data_sources)

    def __repr__(self) -> str:
        return f"tabulated absorption (.measured = {self.measured!r})"

    def _short(self) -> str:
        return 'tabulated'

    def _model_name(self) -> str:
        return self.measured.model

    def _carrier_parameters(self) -> Optional[Dict[str, Any]]:
        return self.measured.parameters

    @property
    def _needs_node_sound_speed(self) -> bool:
        # A table in dB/wavelength, Q or L converts at each node's speed, and
        # the per-node route is exact in any unit.
        return True

    def _breakpoint_depths(self) -> np.ndarray:
        return (np.empty(0) if self.measured.depths is None
                else np.asarray(self.measured.depths, dtype=float))

    def _check_frequency_range(self, frequencies) -> None:
        # A table has no law to extrapolate with: outside it is refused.
        f = np.atleast_1d(np.asarray(frequencies, dtype=float))
        table_f = self.measured.frequencies
        low, high = float(table_f[0]), float(table_f[-1])
        out = f[(f < low * (1.0 - _TABLE_FREQUENCY_RTOL))
                | (f > high * (1.0 + _TABLE_FREQUENCY_RTOL))]
        if out.size:
            span = (f"{out.min():.10g} Hz" if out.size == 1
                    else f"{out.min():.10g}-{out.max():.10g} Hz")
            raise ConfigurationError(
                f"measured absorption table: {out.size} of "
                f"{f.size} frequencies ({span}) lie outside the table's "
                f"{low:g}..{high:g} Hz, and a table carries no law to "
                f"extrapolate with.",
                remediation="Tabulate alpha over the run's whole band, or "
                            "give the environment a law (Thorp, "
                            "FrancoisGarrison) that has a value at every "
                            "frequency.")

    def _native(self, frequencies: np.ndarray,
                depths: np.ndarray) -> np.ndarray:
        """The table at 1-D ``frequencies`` (linear in ``log f``) and 1-D
        ``depths`` (linear, held at the end rows), in the table's own units,
        as an ``(n_depths, n_frequencies)`` grid."""
        t = self.measured
        grid = np.atleast_2d(np.asarray(t.data, dtype=float))
        log_f = np.log(np.asarray(t.frequencies, dtype=float))
        if log_f.size == 1:
            columns = np.repeat(grid[:, :1], frequencies.size, axis=1)
        else:
            x = np.log(frequencies)
            j = np.clip(np.searchsorted(log_f, x), 1, log_f.size - 1)
            w = np.clip((x - log_f[j - 1]) / (log_f[j] - log_f[j - 1]),
                        0.0, 1.0)
            columns = (1.0 - w) * grid[:, j - 1] + w * grid[:, j]
        if t.depths is None:
            return np.repeat(columns[:1, :], depths.size, axis=0)
        rows = np.asarray(t.depths, dtype=float)
        if rows.size == 1:
            return np.repeat(columns[:1, :], depths.size, axis=0)
        # np.interp's rule, on every frequency at once: linear between the
        # rows, the end row held beyond them.
        i = np.clip(np.searchsorted(rows, depths), 1, rows.size - 1)
        wz = np.clip((depths - rows[i - 1]) / (rows[i] - rows[i - 1]),
                     0.0, 1.0)[:, None]
        return (1.0 - wz) * columns[i - 1, :] + wz * columns[i, :]

    def _alpha_dB_per_m(self, frequencies, depths) -> np.ndarray:
        return self._alpha_dB_per_m_at_sound_speed(frequencies, depths,
                                                   DEFAULT_SOUND_SPEED)

    def _alpha_dB_per_m_at_sound_speed(self, frequencies, depths,
                                       sound_speed) -> np.ndarray:
        return np.asarray(convert_attenuation_units(
            self._native(frequencies, depths), frequencies[None, :],
            self.measured.units, 'dB/m', sound_speed=float(sound_speed)),
            dtype=float)

    def _alpha_dB_per_wavelength_at_frequency(self, frequency, depths, sound_speeds):
        f = float(frequency)
        if not f > 0.0:
            raise ConfigurationError(
                f"tabulated absorption: frequency must be > 0 Hz; got "
                f"{frequency}.")
        self._check_frequency_range(f)
        z = np.atleast_1d(np.asarray(depths, dtype=float))
        native = self._native(np.array([f]), z.ravel())[:, 0].reshape(z.shape)
        if self.measured.units == 'dB/wavelength':
            return native
        c = np.broadcast_to(np.asarray(sound_speeds, dtype=float),
                            native.shape)
        return np.asarray(convert_attenuation_units(
            native, f, self.measured.units, 'dB/wavelength', sound_speed=c),
            dtype=float)

    def _water_past_rows(self, z_min: float, z_max: float) -> List[str]:
        """The sides of the water ``z_min..z_max`` (m) the table's rows do
        not reach, in words that name the rows only (empty when they cover
        it or the table has no depth axis)."""
        if self.measured.depths is None:
            return []
        rows = np.asarray(self.measured.depths, dtype=float)
        gaps = []
        if z_min < rows[0]:
            gaps.append(f"above {rows[0]:g} m it takes the {rows[0]:g} m row")
        if z_max > rows[-1]:
            gaps.append(f"below {rows[-1]:g} m it takes the {rows[-1]:g} m "
                        f"row")
        return gaps


def _absorption_law(absorption) -> Optional[Absorption]:
    """The law that evaluates ``absorption`` as an environment holds it: a
    law (or ``None``) as it is, and a measured :class:`AbsorptionCoefficient`
    as the tabulated law over it (:class:`_TabulatedAbsorption`, checked by
    the table rules). The one place a table becomes a law: every engine,
    deck writer and result that evaluates ``env.absorption`` reads it
    through this."""
    if isinstance(absorption, AbsorptionCoefficient):
        return _TabulatedAbsorption(measured=absorption,
                                    data_sources=absorption.data_sources)
    return absorption


def arrival_absorption_exponent(delays_imag, frequencies, *,
                                trace_frequency: Optional[float] = None,
                                absorption: Optional[Union[
                                    Absorption, AbsorptionCoefficient]] = None
                                ) -> np.ndarray:
    """The volume-absorption exponent of each ray arrival at each frequency:
    the received amplitude is ``A · exp(exponent)``.

    A ray code carries volume absorption in the imaginary travel time
    (Jensen et al., *Computational Ocean Acoustics*, §3.6.2: the real rays
    are traced and ``dτ₁/ds = −c₁/c₀²`` adds the loss along their path;
    §1.5.1: ``c_i ≃ (α/ω) c_r²``), so ``ω · Im τ = −∫ α(f, s) ds`` along the
    ray, in the water it crosses (``misc/AttenMod.f90:113``,
    ``Bellhop/Step.f90:73``). ``Im τ`` comes from one trace at
    ``trace_frequency``; the exponent at another frequency is

    - ``2π f_t Im τ · α(f)/α(f_t)``, the ratio at the surface, for a law
      whose frequency dependence is one shape down the column —
      :class:`Thorp` (exact: measured against per-frequency Bellhop traces,
      arrival by arrival, to 0.0011 dB on a multipath guide with refraction)
      and one :class:`FrancoisGarrison` water row, whose pressure terms bend
      the ratio with depth by an amount the callers measure
      (:func:`_band_absorption_error_dB_per_km`, ``by_ratio=True``).
    - ``2π f · Im τ``, linear in ``f``, otherwise: with no law, with no
      ``trace_frequency``, for :class:`ConstantAbsorption` (dB/wavelength,
      so exactly linear), and for a law that does not separate — a
      :class:`Biological` resonance confined in depth, a
      :class:`FrancoisGarrison` profile (its frequency dependence changes
      with the local water) or a tabulated α(f, z) — where the linear law
      is an approximation the callers warn about.

    Parameters
    ----------
    delays_imag : array_like, shape (n_arrivals,)
        Imaginary travel times (s), ≤ 0 for a lossy medium.
    frequencies : array_like, shape (n_frequencies,)
        Frequencies (Hz). Non-negative when a separable law is applied: the
        law is evaluated at each positive frequency, and a 0 Hz bin (the
        first of ``np.fft.rfftfreq``) gets exponent 0, as in the linear
        form.
    trace_frequency : float, optional
        Frequency (Hz) the arrivals were traced at.
    absorption : Absorption or AbsorptionCoefficient, optional
        The water-column absorption the trace carried in ``Im τ``, as the
        traced environment holds it: a law, or a measured table.

    Returns
    -------
    ndarray, shape (n_arrivals, n_frequencies)
    """
    absorption = _absorption_law(absorption)
    imag = np.atleast_1d(np.asarray(delays_imag, dtype=float)).ravel()
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=float)).ravel()
    omega = 2.0 * np.pi * freqs
    if (absorption is None or trace_frequency is None
            or not absorption._scales_by_frequency_ratio):
        return np.outer(imag, omega)
    f_t = float(trace_frequency)
    if np.all(freqs == f_t):
        # At the trace frequency the exponent is Im tau's own, and the law
        # (whose fitted band may not reach f_t) is not consulted.
        return np.outer(imag, omega)
    if np.any(freqs < 0.0):
        raise ConfigurationError(
            f"arrival_absorption_exponent: the {absorption._short()} law "
            f"scales the trace by its ratio alpha(f)/alpha(f_t), which has no "
            f"value at a negative frequency; got {freqs.min():g} Hz.",
            remediation="Pass the non-negative half of the frequency axis "
                        "(np.fft.rfftfreq), or no law for the linear form.")
    # At 0 Hz the exponent is 0, the value the linear form gives. The terms
    # JKPS assigns to absorption mechanisms (regions II-IV of Fig. 1.20:
    # boric acid, MgSO4 relaxation, viscosity) all vanish as f² at 0 Hz; the
    # law's constant term (Thorp's 3.3e-3 dB/km) is region I, which JKPS
    # §1.5.2 calls "conjectured" duct leakage rather than absorption (see
    # absorption_thorp), so no loss scales to the DC bin.
    positive = freqs > 0.0
    alpha = np.asarray(absorption.table(
        np.concatenate([[f_t], freqs[positive]]), units='dB/m').data,
        dtype=float)
    if not alpha[0] > 0.0:
        return np.outer(imag, omega)
    scale = np.zeros(freqs.shape)
    scale[positive] = 2.0 * np.pi * f_t * alpha[1:] / alpha[0]
    return np.outer(imag, scale)


def _band_absorption_error_dB_per_km(absorption: Union[
                                         Absorption, AbsorptionCoefficient],
                                     frequencies,
                                     anchor_frequency: float, *, depths,
                                     power: float = 1.0,
                                     by_ratio: bool = False) -> float:
    """Largest error (dB/km) of an absorption law frozen at one frequency.

    An engine that evaluates ``absorption`` once, at ``anchor_frequency``,
    and scales it as ``(f / f_a) ** power`` across the band applies
    ``α(f_a, z) · (f/f_a)^power`` where the law says ``α(f, z)``. This is
    the largest ``|difference|`` over ``frequencies`` × ``depths``, in dB/km:
    ``power=1`` for a dB-per-wavelength value re-applied at each frequency
    (an OASES water AC, ``oaseun31.f:1522``; Bellhop's ``Im τ`` under a law
    that does not separate), ``power=2`` for a loss that grows as ``f²``
    from its value at the anchor (a viscous damping term). The whole
    grid is scanned, not only the band edges, so a resonance inside the band
    (a :class:`Biological` layer) is caught; the depth axis catches a layer
    below the surface. ``0.0`` when the law is zero at the anchor.

    ``by_ratio=True`` measures the scaling :func:`arrival_absorption_exponent`
    applies to a law that :attr:`Absorption._scales_by_frequency_ratio`:
    ``α(f_a, z) · α(f, 0) / α(f_a, 0)``, the surface ratio at every depth
    (``power`` is then unused).

    ``absorption`` is a law, or a measured table as an environment holds
    it (evaluated as tabulated).
    """
    absorption = _absorption_law(absorption)
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=float)).ravel()
    z = np.atleast_1d(np.asarray(depths, dtype=float)).ravel()
    f_a = float(anchor_frequency)
    # One evaluation (one out-of-band notice), the surface row riding along
    # when the ratio is wanted.
    rows = np.concatenate([[0.0], z]) if by_ratio else z
    grid = np.asarray(absorption.table(
        np.concatenate([[f_a], freqs]), depths=rows, units='dB/km').data,
        dtype=float).reshape(rows.size, freqs.size + 1)
    if by_ratio:
        surface, grid = grid[0], grid[1:]
        scale = (surface[1:] / surface[0] if surface[0] > 0.0
                 else freqs / f_a)
    else:
        scale = (freqs / f_a) ** float(power)
    applied = grid[:, :1] * scale[None, :]
    error = np.abs(grid[:, 1:] - applied)
    return float(np.nanmax(error)) if error.size else 0.0


#: Frequencies (evenly spaced across the band) on which
#: :func:`minimax_anchor_frequency` measures each candidate's error, and the
#: number of candidates in each of its two passes.
_MINIMAX_ANCHOR_GRID = 257


def minimax_anchor_frequency(absorption: Optional[Union[
                                 Absorption, AbsorptionCoefficient]],
                             freq_min: float, freq_max: float, *, depths,
                             power: float = 1.0) -> float:
    """The frequency ``f_a`` in ``[freq_min, freq_max]`` at which to freeze
    ``absorption`` so that ``α(f_a, z) · (f/f_a) ** power`` departs least, at
    its worst over the band and ``depths``, from ``α(f, z)``: the ``f_a``
    minimising :func:`_band_absorption_error_dB_per_km`.

    A minimax choice: one frozen value serves every frequency, so the worst
    departure over the band is the error it sets. Thorp frozen at the band
    centre and scaled linearly misses by 0.0155 dB/km over 1-4 kHz and
    0.0078 dB/km over 100-600 Hz; frozen here, by 0.0063 and 0.0033 dB/km.
    Under ``power=1`` the line depends on ``f_a`` only through the slope
    ``α(f_a)/f_a``, so an anchor elsewhere in the band with the same slope
    gives the same line.

    The band is sampled at :data:`_MINIMAX_ANCHOR_GRID` evenly spaced
    positive frequencies. Each is tried as the anchor, then the best is
    refined on as many points spanning its two neighbours. The band centre
    is returned when no anchor beats it by more than ``1e-9`` of the law's
    largest value in the band — a law already linear in ``f`` (a constant
    dB/wavelength, under ``power=1``) misses by zero at every anchor — and
    when there is no law or no band with two positive frequencies.
    ``absorption`` is a law, or a measured table as an environment holds it
    (evaluated as tabulated).
    """
    absorption = _absorption_law(absorption)
    lo, hi = float(freq_min), float(freq_max)
    centre = 0.5 * (lo + hi)
    freqs = np.linspace(lo, hi, _MINIMAX_ANCHOR_GRID)
    freqs = freqs[freqs > 0.0]
    if absorption is None or not hi > lo or freqs.size < 2:
        return centre
    z = np.atleast_1d(np.asarray(depths, dtype=float)).ravel()
    p = float(power)

    def law(f):
        # table()'s own kernel in dB/km, without its out-of-band notice: the
        # trial grids are this search's, and the caller's evaluations at the
        # anchor and over the run's frequencies give the notice once.
        f = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        return 1000.0 * np.asarray(absorption._alpha_dB_per_m_at_sound_speed(
            f, z, DEFAULT_SOUND_SPEED), dtype=float).reshape(z.size, f.size)

    truth = law(freqs)
    scale = np.nanmax(np.abs(truth)) if np.any(np.isfinite(truth)) else 0.0
    if not scale > 0.0:
        return centre

    def worst(anchors):
        at = law(anchors)
        out = np.empty(anchors.size)
        for j, f_a in enumerate(anchors):
            err = np.abs(truth - at[:, j:j + 1] * (freqs / f_a) ** p)
            out[j] = np.inf if np.all(np.isnan(err)) else np.nanmax(err)
        return out

    coarse = freqs
    coarse_err = worst(coarse)
    j = int(np.argmin(coarse_err))
    fine = np.linspace(coarse[max(j - 1, 0)],
                       coarse[min(j + 1, coarse.size - 1)],
                       _MINIMAX_ANCHOR_GRID)
    fine_err = worst(fine)
    k = int(np.argmin(fine_err))
    best, best_err = ((fine[k], fine_err[k]) if fine_err[k] < coarse_err[j]
                      else (coarse[j], coarse_err[j]))
    centre_err = worst(np.array([centre]))[0] if centre > 0.0 else np.inf
    if not best_err < centre_err - 1e-9 * scale:
        return centre
    return float(best)


#: Absorption error (dB per km of path) above which a run that freezes its
#: water law at one frequency across a band says so. uacpy guidance: 0.05
#: dB/km is 0.5 dB over a 10 km path, so a run that stays silent is off the
#: law by less than half a decibel at that range.
_BAND_ABSORPTION_WARN_DB_PER_KM = 0.05

#: Depths (evenly spaced, surface to the water depth) the band check scans,
#: so a Biological layer below the surface is seen.
_BAND_ABSORPTION_CHECK_DEPTHS = 65


def _warn_if_band_absorption_frozen(who: str, absorption, frequencies,
                                    anchor_frequency: float, *,
                                    water_depth: float, mechanism: str,
                                    remediation: str,
                                    power: float = 1.0,
                                    by_ratio: bool = False) -> Optional[float]:
    """Warn when a law frozen at ``anchor_frequency`` and scaled as
    ``(f/f_a) ** power`` (or by the law's surface ratio, ``by_ratio``)
    misses ``absorption`` by at least
    :data:`_BAND_ABSORPTION_WARN_DB_PER_KM` somewhere in the band and the
    water column (:func:`_band_absorption_error_dB_per_km`).

    The one check every engine that freezes the law runs — Bellhop's single
    trace under a Biological layer, OASES's one water AC per deck — with the
    engine's own ``mechanism``
    (what froze the law, and where in the source) and ``remediation`` sentences.
    Returns the error in dB/km, or ``None`` when there is nothing to check
    (no law, fewer than two positive frequencies, a non-positive anchor).
    ``absorption`` is a law, or a measured table as an environment holds it
    (evaluated as tabulated).
    """
    absorption = _absorption_law(absorption)
    if absorption is None:
        return None
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=float)).ravel()
    freqs = freqs[freqs > 0.0]
    f_a = float(anchor_frequency)
    if freqs.size < 2 or not f_a > 0.0:
        return None
    depths = np.linspace(0.0, float(water_depth), _BAND_ABSORPTION_CHECK_DEPTHS)
    err = _band_absorption_error_dB_per_km(absorption, freqs, f_a,
                                           depths=depths, power=power,
                                           by_ratio=by_ratio)
    if err >= _BAND_ABSORPTION_WARN_DB_PER_KM:
        warnings.warn(
            f"{who}: {mechanism} Across {freqs.min():.4g}-{freqs.max():.4g} "
            f"Hz and the water column the {absorption._short()} law "
            f"departs from that by up to {err:.3g} dB/km of path — about "
            f"{err * 10.0:.3g} dB over 10 km. {remediation}",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    return err
