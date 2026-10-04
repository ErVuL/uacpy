"""Power-delay-profile statistics of a channel: :func:`rms_delay_spread`,
:func:`energy_support`, the coherence bandwidth
(:func:`coherence_bandwidth`, :func:`coherence_factor`) and the
:func:`channel_regime` verdict they add up to; and the frequency grid a
synthesis of the profile needs (:func:`synthesis_band`), with what a record
too short for it folds (:func:`_fold_notice`, the notice the synthesis
methods give).
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Optional

import numpy as np
from uacpy.core._export import CarrierExport
from uacpy.core._repr import FieldsRepr
from uacpy.core.exceptions import ConfigurationError, NumericsWarning
from uacpy.core._validate import require_positive, require_positive_finite_scalar
from uacpy.core._warn_frames import USER_FRAME_SKIP


# ──────────────────────────────────────────────────────────────────────
# Power delay profile
#
# A power delay profile is a list of delays and the power arriving at each.
# It comes from a model's arrival list, from a chirp sounding, from a
# measured channel in a file — these functions ask nothing about which.
# `Arrivals` supplies its own delays and absorption-corrected powers and
# wraps each of them.
# ──────────────────────────────────────────────────────────────────────

#: ``k`` of ``1 / (k tau_rms)``. ``inverse_spread`` is the corpus's own
#: statement (APL-UW TR 9407 sect. II.7.b p. II-32; Abraham sect. 8.8.1);
#: the Rappaport correlation rules (2nd ed. sect. 5.4.3, eqs 5.39-5.40)
#: come from outside the corpus, so they are options and not the default.
COHERENCE_BANDWIDTH_FACTORS = {'inverse_spread': 1.0,
                               'rappaport_0.5': 5.0,
                               'rappaport_0.9': 50.0}


@dataclass(frozen=True)
class ChannelRegime(FieldsRepr, CarrierExport):
    """Verdict of :func:`channel_regime` for one symbol rate.

    ``frequency_selective`` is
    ``signal_bandwidth_hz > coherence_bandwidth_hz``:
    the symbol band spans more than one fade of the channel, so the symbols
    overlap their neighbours (``isi_symbols`` of them, the rms delay spread
    in symbol periods) and a flat gain cannot describe the link.

    On the export protocol: ``to_dict`` / ``from_dict`` through the
    constructor, ``to_xarray`` / ``to_netcdf`` with the fields as JSON.
    """
    coherence_bandwidth_hz: float
    signal_bandwidth_hz: float
    rms_delay_spread_s: float
    symbol_duration_s: float
    frequency_selective: bool
    isi_symbols: float
    convention: str

    def __str__(self) -> str:
        verdict = ("frequency-selective" if self.frequency_selective
                   else "frequency-flat")
        sign = ">" if self.frequency_selective else "<="
        return (f"{verdict}: signal {self.signal_bandwidth_hz:g} Hz {sign} "
                f"coherence {self.coherence_bandwidth_hz:g} Hz "
                f"[{self.convention}] (rms delay spread "
                f"{self.rms_delay_spread_s:g} s = {self.isi_symbols:g} "
                f"symbols of {self.symbol_duration_s:g} s)")


def _profile(delays_s, powers, who):
    """The two arrays every function below takes, checked once."""
    delays = np.asarray(delays_s, dtype=float).ravel()
    power = np.asarray(powers, dtype=float).ravel()
    if delays.size != power.size:
        raise ConfigurationError(
            f"{who}: delays_s and powers must have the same length; got "
            f"{delays.size} and {power.size}.")
    if np.any(power < 0.0):
        raise ConfigurationError(
            f"{who}: powers must be non-negative — this is a POWER delay "
            f"profile, so pass |a|**2, not the complex amplitudes.")
    return delays, power


def rms_delay_spread(delays_s, powers) -> float:
    """Energy-weighted spread of a power delay profile, in seconds.

    The second central moment of the profile: delays weighted by ``powers``,
    about their weighted mean. It measures how much the arrival pattern
    smears a pulse in time, so it bounds the time resolution any processing
    of the channel can have — the smearing of a transmitted pulse, the
    length a replica or matched filter has to cover, the interval a symbol
    would have to exceed to avoid overlapping its neighbour.

    Prefer it to the peak-to-peak spread ``ptp(delays)``, which is set by
    whichever path arrives last no matter how faint: on a 1 km
    bottom-to-bottom path in 1000 m of water at 40 kHz the two differ by
    more than two orders of magnitude, because a path tens of dB down lands
    seconds late while almost all the energy arrives within a millisecond
    of the first.

    Parameters
    ----------
    delays_s : array_like
        Arrival delays (s). Order does not matter.
    powers : array_like
        Power at each delay, same length. Non-negative: pass ``|a|**2``.
        Any consistent scaling works — the result is scale-invariant.

    Returns
    -------
    float
        Seconds. ``0.0`` for a single arrival and for a profile carrying no
        energy at all; ``nan`` if a delay or power is non-finite, rather
        than a spread computed from whatever else was finite.

    Notes
    -----
    Its reciprocal is the frequency scale over which the transfer function
    decorrelates — see :func:`coherence_bandwidth`, which states the
    constant relating the two.
    """
    return _rms_delay_spread(delays_s, powers)


def _rms_delay_spread(delays_s, powers, *,
                     who: str = "rms_delay_spread") -> float:
    """:func:`rms_delay_spread` reporting its refusals as ``who``."""
    delays, power = _profile(delays_s, powers, who)
    total = float(power.sum())
    if delays.size < 2 or total <= 0.0:
        return 0.0
    weights = power / total
    mean = float((weights * delays).sum())
    return float(np.sqrt((weights * (delays - mean) ** 2).sum()))


def energy_support(delays_s, powers, fraction: float = 0.999) -> float:
    """Delay span holding ``fraction`` of a profile's energy, in seconds.

    Measured from the first arrival to the one by which ``fraction`` of the
    energy has arrived. It answers the question a synthesis window asks —
    how long does the response have to be? — which neither of the other two
    measures does: ``ptp(delays)`` is an extremum, moved by one faint
    straggler however little it carries, and :func:`rms_delay_spread` is a
    second moment, a width rather than a span the energy fits inside.

    Parameters
    ----------
    delays_s, powers : array_like
        The profile, as for :func:`rms_delay_spread`.
    fraction : float, default 0.999
        Share of the total energy the span must hold, in ``(0, 1]``. ``1.0``
        is the peak-to-peak span. The default leaves a thousandth of the
        energy — 30 dB down — outside.

    Returns
    -------
    float
        Seconds. ``0.0`` for a single arrival and for a profile carrying no
        energy at all; ``nan`` if a delay or power is non-finite.
    """
    return _energy_support(delays_s, powers, fraction)


def _energy_support(delays_s, powers, fraction: float = 0.999, *,
                   who: str = "energy_support") -> float:
    """:func:`energy_support` reporting its refusals as ``who``."""
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ConfigurationError(
            f"{who}: fraction={fraction:g} is not a share of the energy. "
            f"Pass 0 < fraction <= 1 (1.0 spans every arrival, i.e. the "
            f"peak-to-peak delay).")
    delays, power = _profile(delays_s, powers, who)
    if delays.size < 2:
        return 0.0
    if not (np.all(np.isfinite(delays)) and np.all(np.isfinite(power))):
        return float('nan')
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    order = np.argsort(delays)
    delays = delays[order]
    # The cumulative share is monotone, so the first entry at or above the
    # target is the last arrival that has to fit. Rounding can leave the
    # final entry a hair under 1.0, which would put the index one past the
    # end, so clamp it.
    cumulative = np.cumsum(power[order]) / total
    cut = min(int(np.searchsorted(cumulative, fraction, side='left')),
              delays.size - 1)
    return float(delays[cut] - delays[0])


def _fold_notice(delays_s, powers, record: float, *, who: str, remediation: str,
                first=None) -> Optional[str]:
    """What a ``record``-second synthesis record folds, as text, or ``None``.

    ``delays_s`` and ``powers`` are per arrival. ``first`` is where each
    arrival's record starts — one value, or one per arrival when several
    receiver cells share a record and each starts at its own earliest
    arrival — and defaults to the earliest delay. An inverse FFT is
    circular, so an arrival later than ``first + record`` is not dropped:
    it lands back on the early part of the trace, where it reads as an
    extra early path. The level it returns at is the one number that says
    whether that matters, so it is stated rather than left to be discovered
    in the trace; ``who`` names the caller and ``remediation`` its way out.
    ``None`` when nothing folds, and for fewer than two arrivals or a
    non-finite delay.

    Parameters
    ----------
    delays_s : array_like
        Arrival delays (s).
    powers : array_like
        Arrival powers.
    record : float
        Record length (s) of the synthesis.
    who : str
        Name of the caller, opening the text.
    remediation : str
        The caller's way out, closing the text.
    first : float or array_like, optional
        Where each record starts (s); ``None`` is the earliest delay.
    """
    delays = np.asarray(delays_s, dtype=float).ravel()
    power = np.asarray(powers, dtype=float).ravel()
    if delays.size < 2 or not np.all(np.isfinite(delays)):
        return None
    start = (float(delays.min()) if first is None
             else np.asarray(first, dtype=float))
    folded = delays > start + record
    if not folded.any():
        return None
    total = float(power.sum())
    share = float(power[folded].sum()) / total if total > 0.0 else 0.0
    level = (f"{10.0 * np.log10(share):.0f} dB" if share > 0.0
             else "no measurable level")
    return (f"{who}: a {record:g} s record does not reach the last arrival "
            f"at {float(delays.max()):g} s, so the {int(folded.sum())} "
            f"arrival(s) past its end fold back onto the early trace at "
            f"{level} relative to the whole. {remediation}")


def synthesis_band(delays_s, powers, *, bandwidth: float, centre: float,
                   record: Optional[float] = None,
                   energy_fraction: Optional[float] = None,
                   margin: float = 1.2) -> np.ndarray:
    """Frequency grid wide enough to synthesise a power delay profile
    un-aliased.

    A record built by an inverse FFT is ``1/df`` long, so the frequency
    spacing — not the bandwidth — decides how much multipath the trace can
    hold. The window is the primitive and the spacing follows from it
    (Jensen, Kuperman, Porter and Schmidt, *Computational Ocean Acoustics*,
    sect. 8.2). Pass ``record`` to state that window; leave it out and it is
    the span holding ``energy_fraction`` of the profile's energy
    (:func:`energy_support`), times ``margin``, and never shorter than
    ``1/bandwidth``.

    Parameters
    ----------
    delays_s, powers : array_like
        The profile, as for :func:`energy_support`, pooled over every cell
        the synthesis shares one record across.
    bandwidth : float, keyword-only
        Width of the band (Hz), centred on ``centre``.
    centre : float, keyword-only
        Band centre (Hz). The band must start above 0 Hz.
    record : float, optional
        Length of the record (s), stated rather than derived. Refused with
        ``energy_fraction``: they are two answers to one question.
    energy_fraction : float, optional
        Share of the energy the derived window must hold (default 0.999).
    margin : float, optional
        Headroom on the derived span (default 1.2, at least 1). It is also
        the number of frequency samples per interference fringe of two
        paths the span apart. Not used when ``record`` is given.

    Returns
    -------
    ndarray
        Ascending frequencies, ``max(ceil(bandwidth * record) + 1, 2)`` of
        them over ``centre ± bandwidth / 2``.

    Warns
    -----
    NumericsWarning
        When the record does not reach the last arrival, giving how many
        arrivals fold back and the level they fold in at
        (:func:`_fold_notice`).
    """
    return _synthesis_band(
        delays_s,
        powers,
        bandwidth=bandwidth,
        centre=centre,
        record=record,
        energy_fraction=energy_fraction,
        margin=margin)


def _synthesis_band(delays_s, powers, *, bandwidth: float, centre: float,
                   record: Optional[float] = None,
                   energy_fraction: Optional[float] = None,
                   margin: float = 1.2,
                   who: str = "synthesis_band") -> np.ndarray:
    """:func:`synthesis_band` reporting its refusals as ``who``."""
    bandwidth = float(bandwidth)
    require_positive(bandwidth, f"{who} bandwidth", hint="Hz")
    centre = float(centre)
    require_positive(centre, f"{who} centre", hint="Hz")
    # A positive centre and a positive width still describe a band that
    # runs through 0 Hz into negative frequency when the width exceeds
    # twice the centre, and a model run on it returns an H that is not
    # conjugate-symmetric, which an IFFT turns into a complex trace.
    if centre - bandwidth / 2.0 <= 0.0:
        raise ConfigurationError(
            f"{who}: a {bandwidth:g} Hz band centred on "
            f"{centre:g} Hz starts at "
            f"{centre - bandwidth / 2.0:g} Hz, at or below 0 Hz — there "
            f"is no field to synthesise there. Narrow bandwidth= below "
            f"{2.0 * centre:g} Hz, or pass centre= high enough to carry "
            f"the band.")
    if record is not None and energy_fraction is not None:
        raise ConfigurationError(
            f"{who}: record= and energy_fraction= are "
            "two answers to one question — how long the record has to "
            "be. Pass record= (s) to state the window, or "
            "energy_fraction= to derive it from these arrivals.")
    if record is not None:
        record = float(record)
        require_positive(record, f"{who} record", hint="s")
    else:
        margin = float(margin)
        if margin < 1.0:
            raise ConfigurationError(
                f"{who}: margin={margin:g} would size "
                f"the record SHORTER than the span it has to hold, which "
                f"is the aliasing this function exists to prevent. Pass "
                f"margin >= 1 (1.2 leaves the last arrival inside the "
                f"window off the final sample).")
        support = _energy_support(
            delays_s, powers,
            0.999 if energy_fraction is None else energy_fraction, who=who)
        if not np.isfinite(support):
            raise ConfigurationError(
                f"{who}: these arrivals carry a "
                "non-finite delay or amplitude, so the span they occupy "
                "is undefined and no window can be derived from them. "
                "Pass record= (s) to state one.")
        # A single arrival spans no time, and a grid still needs two
        # points to define a spacing: fall back to the shortest record
        # that holds it.
        record = max(margin * support, 1.0 / bandwidth)
    delays = np.asarray(delays_s, dtype=float).ravel()
    if delays.size >= 2 and np.all(np.isfinite(delays)):
        span = float(delays.max() - delays.min())
        notice = _fold_notice(
            delays, powers, record, who=who,
            remediation=(f"Pass energy_fraction=1.0 (or record={span:g}) to hold "
                    f"every arrival, or drop the tail arrivals before "
                    f"sizing the band rather than folding them."))
        if notice is not None:
            warnings.warn(notice, NumericsWarning,
                          skip_file_prefixes=USER_FRAME_SKIP)
    n_freq = max(int(np.ceil(bandwidth * record)) + 1, 2)
    return np.linspace(centre - bandwidth / 2.0,
                       centre + bandwidth / 2.0, n_freq)


def coherence_factor(convention: str = 'inverse_spread', factor=None) -> float:
    """The ``k`` of ``1 / (k tau_rms)``: ``factor`` when given (any finite
    ``k > 0``), else the named convention's — see
    :data:`COHERENCE_BANDWIDTH_FACTORS`. ``who`` (keyword-only) is the name
    the refusals carry, for a method that delegates here.

    Parameters
    ----------
    convention : str, optional
        A named ``k`` of :data:`COHERENCE_BANDWIDTH_FACTORS`. Default
        ``'inverse_spread'`` (``k = 1``).
    factor : float, optional
        An explicit ``k > 0``, in place of ``convention``.
    """
    return _coherence_factor(convention, factor)


def _coherence_factor(convention: str = 'inverse_spread', factor=None, *,
                     who: str = "coherence_factor") -> float:
    """:func:`coherence_factor` reporting its refusals as ``who``."""
    if factor is not None:
        return require_positive_finite_scalar(
            factor, who, "factor",
            why=(f" It is the k of 1 / (k * tau_rms); leave it out to use "
                 f"convention={convention!r}."))
    try:
        return COHERENCE_BANDWIDTH_FACTORS[convention]
    except KeyError:
        raise ConfigurationError(
            f"{who}: convention must be one of "
            f"{sorted(COHERENCE_BANDWIDTH_FACTORS)} (1/tau_rms, "
            f"1/(5 tau_rms), 1/(50 tau_rms)), or pass factor=k for "
            f"1/(k tau_rms); got {convention!r}.") from None


def coherence_bandwidth(delays_s, powers, *,
                        convention: str = 'inverse_spread',
                        factor=None) -> float:
    """Bandwidth over which a channel's transfer function stays correlated,
    in Hz, as ``1 / (k * tau_rms)`` with ``tau_rms`` the
    :func:`rms_delay_spread` of the profile.

    The default ``k = 1`` is the convention the corpus states: "the inverse
    [of the elongation time] in hertz is a measure of the coherence
    bandwidth of the channel" (APL-UW TR 9407, sect. II.7.b, p. II-32) and
    ``W < 1 / sigma_t = W_c`` (Abraham, *Underwater Acoustic Signal
    Processing*, sect. 8.8.1, Fig. 8.34; 1/(33 ms) = 30 Hz). The
    named options ``'rappaport_0.5'`` (``k = 5``) and ``'rappaport_0.9'``
    (``k = 50``) are the 0.5- and 0.9-correlation rules of Rappaport,
    *Wireless Communications*, 2nd ed., sect. 5.4.3, eqs 5.39-5.40 — a
    source outside the corpus, so they are options and not the default.

    Returns ``inf`` for a single arrival (no spread, a flat channel), and
    ``nan`` when the spread is. ``who`` (keyword-only) is the name the
    refusals carry, for a method that delegates here.

    Parameters
    ----------
    delays_s : array_like
        Arrival delays (s).
    powers : array_like
        Arrival powers.
    convention : str, optional
        A named ``k`` of :data:`COHERENCE_BANDWIDTH_FACTORS`. Default
        ``'inverse_spread'`` (``k = 1``).
    factor : float, optional
        An explicit ``k > 0``, in place of ``convention``.
    """
    return _coherence_bandwidth(
        delays_s,
        powers,
        convention=convention,
        factor=factor)


def _coherence_bandwidth(delays_s, powers, *,
                        convention: str = 'inverse_spread',
                        factor=None, who: str = "coherence_bandwidth"
                        ) -> float:
    """:func:`coherence_bandwidth` reporting its refusals as ``who``."""
    k = _coherence_factor(convention, factor, who=who)
    spread = _rms_delay_spread(delays_s, powers, who=who)
    if not np.isfinite(spread):
        return float('nan')
    if spread <= 0.0:
        return float('inf')
    return 1.0 / (k * spread)


def channel_regime(delays_s, powers, symbol_rate: float, *,
                   convention: str = 'inverse_spread', factor=None,
                   rolloff: float = 0.0) -> ChannelRegime:
    """Whether a modem at ``symbol_rate`` sees this profile as flat or
    frequency-selective.

    Compares the signal bandwidth ``(1 + rolloff) * symbol_rate`` with
    :func:`coherence_bandwidth` under ``convention`` / ``factor``:
    selective when the signal is the wider (Proakis, *Digital
    Communications*, 4th ed., sect. 14.1.2; Stojanovic and Preisig 2009,
    sect. II). ``isi_symbols`` is the rms delay spread in symbol periods,
    the number of neighbours each symbol overlaps. The result records the
    convention it was judged under (``factor=k`` is recorded as
    ``'factor=k'``).

    Parameters
    ----------
    delays_s, powers : array_like
        The profile, as for :func:`rms_delay_spread`.
    symbol_rate : float
        Symbol rate (Bd).
    convention, factor
        As on :func:`coherence_bandwidth`.
    rolloff : float, default 0.0
        Excess bandwidth of the pulse; ``0`` takes the Nyquist bandwidth
        equal to the symbol rate.
    """
    return _channel_regime(
        delays_s,
        powers,
        symbol_rate,
        convention=convention,
        factor=factor,
        rolloff=rolloff)


def _channel_regime(delays_s, powers, symbol_rate: float, *,
                   convention: str = 'inverse_spread', factor=None,
                   rolloff: float = 0.0,
                   who: str = "channel_regime") -> ChannelRegime:
    """:func:`channel_regime` reporting its refusals as ``who``."""
    symbol_rate = require_positive_finite_scalar(
        symbol_rate, who, "symbol_rate", " Bd")
    rolloff = float(rolloff)
    if not 0.0 <= rolloff <= 1.0:
        raise ConfigurationError(
            f"{who}: rolloff must be in [0, 1]; got {rolloff!r}.")
    k = _coherence_factor(convention, factor, who=who)
    coherence = _coherence_bandwidth(delays_s, powers, factor=k, who=who)
    spread = _rms_delay_spread(delays_s, powers, who=who)
    signal = (1.0 + rolloff) * symbol_rate
    return ChannelRegime(
        coherence_bandwidth_hz=coherence,
        signal_bandwidth_hz=signal,
        rms_delay_spread_s=spread,
        symbol_duration_s=1.0 / symbol_rate,
        frequency_selective=bool(signal > coherence),
        isi_symbols=spread * symbol_rate,
        convention=(str(convention) if factor is None
                    else f"factor={float(factor):g}"),
    )
