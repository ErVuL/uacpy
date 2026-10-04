"""The settings records of a Kraken run: :class:`KrakenLaunch` (one
launch of the modes binary and, on a field run, field.exe) and
:class:`KrakenSettings` (every launch of a run and how their outputs make
the result)."""

from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple
from uacpy.core._records import FrozenRecord
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.run_settings import EngineSettings


#: How a Kraken run turns its launches into a result
#: (:attr:`KrakenSettings.route`): ``'modes'`` a mode set; ``'field'`` one
#: narrowband field (a stack over a multi-depth source); ``'band'`` one
#: ``TopOpt(6)='B'`` deck carrying the whole band; ``'band_bin'`` a band of
#: one bin, solved narrowband; ``'band_by_frequency'`` one narrowband launch
#: per bin (a range-dependent band: the multi-profile deck carries one
#: frequency).
_ROUTES = ('modes', 'field', 'band', 'band_bin', 'band_by_frequency')
_BAND_ROUTES = ('band', 'band_bin', 'band_by_frequency')


@dataclass(frozen=True, eq=False)
class KrakenLaunch(FrozenRecord):
    """One solve of the modes binary in a :class:`Kraken` run, and the
    ``field.exe`` run that sums its modes (none on a MODES run).

    Attributes
    ----------
    deck_frequency : float
        The modes deck's header frequency ``freq0`` (Hz): the frequency of a
        narrowband solve, and the one AT checks the mesh floor at.
    marched_frequencies : ndarray or None
        The frequency vector of a ``TopOpt(6)='B'`` deck (one ``.mod`` block
        per frequency), ``None`` on a narrowband deck.
    tabulation_depths : ndarray
        The receiver-depth line of the modes deck (m): where the ``.mod``
        tabulates the mode shapes, merged with the source depths by
        ``kraken.f90:573``.
    profile_ranges_m : ndarray or None
        Start range (m) of every profile of a multi-profile deck; ``None``
        for a single profile.
    c_low : float
        The deck's ``cLow`` (m/s), the same in every profile.
    c_high : tuple of float
        The deck's ``cHigh`` (m/s), one per profile.
    rmax_m : float
        The deck's ``RMax`` (m), the mesh-convergence scale of the solve.
    n_mesh : int
        The ``NG`` mesh count written on every medium line (0: the binary
        sizes each medium).
    field_option : str or None
        ``field.exe``'s four-letter option (source geometry, coupling, beam
        pattern, coherence); ``None`` when no field is summed.
    check_n_mesh : int or None
        The mesh of the second, coarser solve the launch's mode count is
        compared with (a krakenc solve of an elastic problem); ``None``: no
        second solve.
    """

    _REPR_UNITS = {'deck_frequency': 'Hz', 'marched_frequencies': 'Hz',
                   'tabulation_depths': 'm', 'profile_ranges_m': 'm',
                   'c_low': 'm/s', 'c_high': 'm/s'}

    deck_frequency: float
    marched_frequencies: Optional[np.ndarray]
    tabulation_depths: np.ndarray
    profile_ranges_m: Optional[np.ndarray]
    c_low: float
    c_high: Tuple[float, ...]
    rmax_m: float
    n_mesh: int
    field_option: Optional[str] = None
    check_n_mesh: Optional[int] = None

    _ARRAY_FIELDS = ('marched_frequencies', 'tabulation_depths', 'profile_ranges_m')

    def __post_init__(self):
        # A list (the to_dict form) is stored as the tuple a frozen record
        # holds.
        object.__setattr__(self, 'c_high',
                           tuple(float(c) for c in self.c_high))
        self._freeze_arrays()

    @classmethod
    def from_dict(cls, d) -> 'KrakenLaunch':
        return cls(**d)

    @property
    def n_profiles(self) -> int:
        """Profiles of the deck (1 for a single-profile deck)."""
        return len(self.c_high)

    def summary(self) -> str:
        freq = (f"{self.marched_frequencies.size} frequencies, {self.marched_frequencies[0]:g}-"
                f"{self.marched_frequencies[-1]:g} Hz" if self.marched_frequencies is not None
                else f"{self.deck_frequency:g} Hz")
        window = (f"{self.c_high[0]:.6g}" if len(set(self.c_high)) == 1
                  else f"{min(self.c_high):.6g}-{max(self.c_high):.6g}")
        text = (f"{freq}: c {self.c_low:.6g}..{window} m/s, RMax "
                f"{self.rmax_m:.6g} m, {self.tabulation_depths.size} "
                f"tabulation depths, n_mesh {self.n_mesh}")
        if self.n_profiles > 1:
            text += f", {self.n_profiles} profiles"
        if self.field_option is not None:
            text += f", field option {self.field_option!r}"
        if self.check_n_mesh is not None:
            text += f", mode count checked at n_mesh {self.check_n_mesh}"
        return text


@dataclass(frozen=True, eq=False)
class KrakenSettings(EngineSettings):
    """The settings one :class:`Kraken` run resolved, before launching:
    ``Kraken().run_settings(env, source, receiver).engine``, and
    ``result.run_settings.engine`` on the result it produced.

    Attributes
    ----------
    backend : str
        The modes binary, ``'kraken'`` or ``'krakenc'``.
    backend_origin : str
        ``'Kraken(backend=…)'`` when pinned, else the condition of the
        environment that chose it (:meth:`Kraken.select_backend`).
    route : str
        How the launches become the result (``'modes'``, ``'field'``,
        ``'band'``, ``'band_bin'``, ``'band_by_frequency'``). Every route
        but ``'modes'`` sums the modes with ``field.exe``.
    c_low_origin, c_high_origin : str
        Which rule of the phase-speed window
        (:func:`_window.phase_speed_window`) set each bound.
    rmax_origin : str
        Where every launch's ``RMax`` came from.
    mode_points_per_meter : float or None
        The density (points/m) of the mode tabulation grid of a field run;
        ``None`` on a MODES run, whose grid is the receiver's depths.
    mode_coupling : str
        ``'adiabatic'`` or ``'coupled'`` on a multi-profile deck, ``'none'``
        on a single profile.
    n_modes : int or None
        The cap on the modes ``field.exe`` sums (FLP ``MLimit``) and a MODES
        result keeps.
    evaluated_depths : ndarray or None
        The receiver depths ``field.exe`` evaluates, when some requested
        depths lie in an elastic sub-bottom medium it cannot evaluate;
        ``None`` when it evaluates every requested depth.
    receiver_keep : tuple of bool or None
        Which requested receiver depths are evaluated (the others come back
        NaN); ``None`` when all are.
    launches : tuple of KrakenLaunch
        Every modes solve of the run, in launch order.
    notices : tuple of Notice
        One per condition of these settings the run warns about: its
        short ``note`` and the ``message`` ``run`` and ``run_settings``
        announce (``validate_inputs`` announces nothing).
    """

    backend: str
    backend_origin: str
    route: str
    c_low_origin: str
    c_high_origin: str
    rmax_origin: str
    mode_points_per_meter: Optional[float]
    mode_coupling: str
    n_modes: Optional[int]
    evaluated_depths: Optional[np.ndarray]
    receiver_keep: Optional[Tuple[bool, ...]]
    launches: Tuple[KrakenLaunch, ...]

    _ARRAY_FIELDS = ('evaluated_depths',)
    _TABLE_FIELD = 'launches'

    def __post_init__(self):
        if self.route not in _ROUTES:
            raise ConfigurationError(
                f"KrakenSettings.route must be one of {_ROUTES}; "
                f"got {self.route!r}.")
        # The to_dict forms (lists, dicts) are stored as the tuples and
        # records a frozen record holds.
        object.__setattr__(self, 'launches', tuple(
            launch if isinstance(launch, KrakenLaunch)
            else KrakenLaunch.from_dict(launch) for launch in self.launches))
        if self.receiver_keep is not None:
            object.__setattr__(self, 'receiver_keep',
                               tuple(bool(k) for k in self.receiver_keep))
        super().__post_init__()

    def summary_lines(self):
        """One ``(name, text)`` pair per field, one line per launch (a band
        solved bin by bin is summarised), the notices shortened."""
        out = [(name, text) for name, text in super().summary_lines()
               if name not in ('launches', 'note', 'notices',
                               'receiver_keep')]
        if self.receiver_keep is not None:
            out.append(('receiver_keep', f"{sum(self.receiver_keep)} of "
                                         f"{len(self.receiver_keep)} depths"))
        if len(self.launches) <= 3:
            out.extend((f"launch {i}", launch.summary())
                       for i, launch in enumerate(self.launches))
        else:
            freqs = [launch.deck_frequency for launch in self.launches]
            out.append(('launches', (
                f"{len(self.launches)}, one per frequency, "
                f"{min(freqs):g}-{max(freqs):g} Hz; first: "
                f"{self.launches[0].summary()}")))
        out.extend((name, text) for name, text in super().summary_lines()
                   if name == 'note')
        return out
