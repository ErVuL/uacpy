"""Ray/eigenray and arrival result types."""

from __future__ import annotations

import warnings
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._carrier_validate import _require_positive

from uacpy.core.results._base import Result


def _arrival_kind(n_top: int, n_bot: int) -> str:
    if n_top >= 1 and n_bot >= 1:
        return 'both'
    if n_bot >= 1:
        return 'bottom'
    if n_top >= 1:
        return 'surface'
    return 'direct'


_BOUNCE_KINDS = ('direct', 'surface', 'bottom', 'both')


def _bounce_in_bounds(value: int, spec) -> bool:
    """Match ``value`` against an int (exact) or ``(lo, hi)`` tuple
    (closed range, ``None`` = unbounded). Shared with :class:`Rays`."""
    if spec is None:
        return True
    if isinstance(spec, int):
        return value == spec
    lo, hi = spec
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True


def _bounce_predicate(kind, top, bot):
    """Build a predicate ``(n_top, n_bot) -> bool`` for bounce filtering.

    Shared by :class:`Arrivals` and :class:`Rays`. ``kind`` is one of
    :data:`_BOUNCE_KINDS` (or ``None``); ``top`` / ``bot`` are
    int / (lo, hi) / None specs matching :func:`_bounce_in_bounds`.
    """
    if kind is not None and kind not in _BOUNCE_KINDS:
        raise ConfigurationError(
            f"bounce filter: kind={kind!r} not in {_BOUNCE_KINDS}"
        )

    def predicate(n_top: int, n_bot: int) -> bool:
        if kind is not None and _arrival_kind(n_top, n_bot) != kind:
            return False
        return (_bounce_in_bounds(n_top, top)
                and _bounce_in_bounds(n_bot, bot))

    return predicate


def _fold_notice(delays, power, record: float, *, who: str, remedy: str,
                 first=None) -> Optional[str]:
    """What a ``record``-second record folds, as text, or ``None``.

    ``delays`` and ``power`` are per arrival. ``first`` is where each
    arrival's record starts — one value, or one per arrival when several
    receiver cells share a record and each starts at its own earliest
    arrival — and defaults to the earliest delay. An inverse FFT is
    circular, so an arrival later than ``first + record`` is not dropped:
    it lands back on the early part of the trace, where it reads as an
    extra early path. The level it returns at is the one number that says
    whether that matters, so it is stated rather than left to be discovered
    in the trace; ``who`` names the caller and ``remedy`` its way out.
    """
    delays = np.asarray(delays, dtype=float).ravel()
    power = np.asarray(power, dtype=float).ravel()
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
            f"{level} relative to the whole. {remedy}")


class Arrivals(Result):
    """Ray arrivals from Bellhop — a flat list of arrival events.

    Each arrival is a dict with: ``delay`` (s), ``amplitude``, ``phase``
    (**degrees** — the unit the ``.arr`` reader stores; the :attr:`phases`
    accessor converts to radians), ``n_top_bounces``, ``n_bot_bounces``,
    ``src_angle``, ``rcv_angle``, ``kind`` ('direct' / 'surface' /
    'bottom' / 'both'), plus the cell of origin (``src_idx``,
    ``depth_idx``, ``range_idx``) so multi-cell runs can be filtered back
    to one cell if needed.

    Mirrors the :class:`Rays` API surface: filter / chain / sort.
    """
    field_type = "arrivals"

    def __init__(
        self,
        *,
        arrivals: Optional[List[Dict[str, Any]]] = None,
        by_receiver: Any = None,
        receiver_depths: np.ndarray,
        receiver_ranges: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.receiver_depths = np.atleast_1d(np.asarray(receiver_depths, dtype=float))
        self.receiver_ranges = np.atleast_1d(np.asarray(receiver_ranges, dtype=float))
        # Nested ``[src][depth][range] -> dict`` form that Bellhop's IO
        # produces and the broadband delay-and-sum path needs.
        self.by_receiver = by_receiver
        if arrivals is not None:
            self.arrivals = list(arrivals)
        else:
            self.arrivals = self._flatten_by_receiver(by_receiver)

    @staticmethod
    def _flatten_by_receiver(by_receiver: Any) -> List[Dict[str, Any]]:
        """Flatten Bellhop's ``arrivals_data[src][depth][range] -> dict``
        nesting into a single per-arrival list. Each emitted record
        carries its source/cell indices so callers can filter back."""
        if by_receiver is None:
            return []
        out: List[Dict[str, Any]] = []
        for s_idx, by_src in enumerate(by_receiver if isinstance(by_receiver, list) else []):
            for d_idx, by_depth in enumerate(by_src if isinstance(by_src, list) else []):
                for r_idx, cell in enumerate(by_depth if isinstance(by_depth, list) else []):
                    if not isinstance(cell, dict):
                        continue
                    delays = np.asarray(cell.get('delays', []))
                    if len(delays) == 0:
                        continue
                    amps = np.asarray(cell.get('amplitudes', np.zeros_like(delays)))
                    phs = np.asarray(cell.get('phases', np.zeros_like(delays)))
                    nt = np.asarray(cell.get('n_top_bounces', np.zeros(len(delays), int)))
                    nb = np.asarray(cell.get('n_bot_bounces', np.zeros(len(delays), int)))
                    sa = np.asarray(cell.get('src_angles', np.zeros_like(delays)))
                    ra = np.asarray(cell.get('rcv_angles', np.zeros_like(delays)))
                    # Im(delay) carries Bellhop's volume-attenuation loss as a
                    # separate multiplicative term exp(omega * Im(delay))
                    # (ArrMod.f90:118-125 writes it as its own field), so it
                    # travels with the flat records too.
                    di = np.asarray(cell.get('delays_imag', np.zeros_like(delays)))
                    for i in range(len(delays)):
                        n_top, n_bot = int(nt[i]), int(nb[i])
                        out.append({
                            'delay': float(delays[i]),
                            'delay_imag': float(di[i]),
                            'amplitude': float(amps[i]),
                            'phase': float(phs[i]),
                            'n_top_bounces': n_top,
                            'n_bot_bounces': n_bot,
                            'src_angle': float(sa[i]),
                            'rcv_angle': float(ra[i]),
                            'kind': _arrival_kind(n_top, n_bot),
                            'src_idx': s_idx,
                            'depth_idx': d_idx,
                            'range_idx': r_idx,
                        })
        return out

    def __len__(self) -> int:
        return len(self.arrivals)

    def __iter__(self):
        return iter(self.arrivals)

    def _repr_extra(self) -> str:
        return f"n_arrivals={len(self.arrivals)}"

    # Per-field bulk views ---------------------------------------------------

    @property
    def delays(self) -> np.ndarray:
        """Travel times (s) of every arrival in the list."""
        return np.asarray([a['delay'] for a in self.arrivals], dtype=float)

    @property
    def amplitudes(self) -> np.ndarray:
        """Amplitudes (linear) of every arrival in the list."""
        return np.asarray([a['amplitude'] for a in self.arrivals], dtype=float)

    @property
    def phases(self) -> np.ndarray:
        """Phases (rad) of every arrival in the list.

        The Bellhop ``.arr`` file stores phase in **degrees** (``ArrMod.f90``
        writes ``RadDeg * Phase``); this accessor converts to **radians** so the
        values drop straight into ``exp(1j * phase)`` for phase-coherent
        synthesis."""
        return np.deg2rad(
            np.asarray([a['phase'] for a in self.arrivals], dtype=float))

    # Filter / chain / sort --------------------------------------------------

    def _spawn(self, arrivals: List[Dict[str, Any]]) -> 'Arrivals':
        """Build a filtered/sorted ``Arrivals`` from a subset of the flat list.

        ``by_receiver`` is rebuilt from the surviving records so the nested
        and flat views never disagree — Bellhop's broadband delay-and-sum
        reads the nested form, and a subset that kept the parent's full
        nesting would re-introduce the arrivals the caller filtered out.
        """
        return Arrivals(
            arrivals=arrivals,
            by_receiver=self._rebuild_by_receiver(arrivals),
            receiver_depths=self.receiver_depths,
            receiver_ranges=self.receiver_ranges,
            **self.id_kwargs(),
        )

    def _rebuild_by_receiver(self, arrivals: List[Dict[str, Any]]):
        """Regroup flat arrival records into ``[src][depth][range] -> dict``.

        ``None`` when the parent carried no nested view; otherwise the parent's
        cell grid with only ``arrivals`` present in each cell.
        """
        if self.by_receiver is None:
            return None
        shape = (len(self.by_receiver),
                 len(self.by_receiver[0]) if self.by_receiver else 0,
                 len(self.by_receiver[0][0]) if self.by_receiver
                 and self.by_receiver[0] else 0)
        # The per-cell record ``io/oalib_reader.py`` builds when it parses a
        # ``.arr``, key for key and dtype for dtype. A rebuilt cell that
        # differs in either is one a consumer of ``by_receiver`` — e.g.
        # ``models.bellhop.delayandsum``, which reads ``cell['n_arrivals']``
        # and indexes the columns — handles on the freshly-read path and not
        # on the filtered/sorted one.
        keys = {'delays': 'float64', 'delays_imag': 'float64',
                'amplitudes': 'float64', 'phases': 'float64',
                'n_top_bounces': 'int32', 'n_bot_bounces': 'int32',
                'src_angles': 'float64', 'rcv_angles': 'float64'}
        cells = [[[{k: [] for k in keys} for _ in range(shape[2])]
                  for _ in range(shape[1])] for _ in range(shape[0])]
        for a in arrivals:
            s, d, r = a['src_idx'], a['depth_idx'], a['range_idx']
            if not (s < shape[0] and d < shape[1] and r < shape[2]):
                continue
            cell = cells[s][d][r]
            cell['delays'].append(a['delay'])
            cell['delays_imag'].append(a['delay_imag'])
            cell['amplitudes'].append(a['amplitude'])
            cell['phases'].append(a['phase'])
            cell['n_top_bounces'].append(a['n_top_bounces'])
            cell['n_bot_bounces'].append(a['n_bot_bounces'])
            cell['src_angles'].append(a['src_angle'])
            cell['rcv_angles'].append(a['rcv_angle'])
        def _finish(cell):
            # ``n_arrivals`` is a Python ``int`` on the reader's cell, so it
            # is written after the array pass: inside it, ``np.asarray`` would
            # make it a 0-d array and the key sets would agree while the value
            # types did not.
            out = {k: np.asarray(cell[k], dtype=dtype)
                   for k, dtype in keys.items()}
            out['n_arrivals'] = int(len(out['delays']))
            return out

        return [[[_finish(cell) for cell in by_depth] for by_depth in by_src]
                for by_src in cells]

    def filter(self, predicate) -> 'Arrivals':
        """Return a new ``Arrivals`` keeping arrivals for which
        ``predicate(arrival_dict)`` returns true."""
        return self._spawn([a for a in self.arrivals if predicate(a)])

    def filter_by_bounces(
        self,
        kind: Optional[str] = None,
        top: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
        bot: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
    ) -> 'Arrivals':
        """Subset by multipath component — same semantics as
        :meth:`Rays.filter_by_bounces`. ``kind`` ∈
        ``{'direct', 'surface', 'bottom', 'both'}``; ``top`` / ``bot`` are
        an int (exact) or ``(lo, hi)`` tuple (closed range, ``None`` =
        unbounded)."""
        pred = _bounce_predicate(kind, top, bot)
        return self.filter(
            lambda a: pred(int(a['n_top_bounces']), int(a['n_bot_bounces']))
        )

    def in_delay_window(
        self,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
    ) -> 'Arrivals':
        """Keep arrivals whose ``delay`` falls inside ``[t_min, t_max]``
        (each bound optional)."""
        def pred(a):
            d = a['delay']
            if t_min is not None and d < t_min:
                return False
            if t_max is not None and d > t_max:
                return False
            return True
        return self.filter(pred)

    def sorted_by_amplitude(self, descending: bool = True) -> 'Arrivals':
        """Return a copy sorted by received amplitude (descending by default).

        Ranked on what reaches the receiver, not on the ``amplitude`` column
        alone: Bellhop keeps volume absorption in the imaginary travel time
        (see :meth:`_arrival_power`), so a long path can carry the larger
        geometric amplitude and still arrive far quieter. At 40 kHz a 6 km
        bounce path with three times the direct path's amplitude lands 55 dB
        below it, and ranking on the column alone puts it first.

        With no frequency on the result the absorption factor is 1 and this
        is the column order, unchanged.
        """
        # argsort on power: monotone in received amplitude, so it orders the
        # same way without the square root, and it is the same quantity
        # ``rms_delay_spread`` and ``energy_support`` weigh by.
        order = np.argsort(self._arrival_power(), kind='stable')
        if descending:
            order = order[::-1]
        return self._spawn([self.arrivals[int(i)] for i in order])

    def top_n_by_amplitude(self, n: int) -> 'Arrivals':
        """Keep the ``n`` arrivals that reach the receiver loudest.

        "Loudest" is the received level, absorption included — see
        :meth:`sorted_by_amplitude` for why the amplitude column alone
        answers a different question.
        """
        return self._spawn(self.sorted_by_amplitude(descending=True)
                           .arrivals[:int(n)])

    def _arrival_power(self) -> np.ndarray:
        """Power each arrival delivers to the receiver, absorption included.

        Volume absorption does not live in the amplitude column: Bellhop
        carries it in the IMAGINARY travel time, so the received amplitude is
        ``A * exp(w * Im tau)`` — the convention ``read_arr_file`` documents
        for ``delays_imag`` and ``Bellhop._arrivals_to_tf`` applies. Scoring
        on ``A`` alone treats a late, heavily absorbed path as though the
        water were lossless, and the late paths are the ones every caller of
        this is weighing.
        """
        amplitudes = np.abs(np.asarray(self.amplitudes, dtype=float).ravel())
        delays_imag = np.asarray(
            [a.get('delay_imag', 0.0) for a in self.arrivals], dtype=float)
        omega = 2.0 * np.pi * self.f0 if self.f0 else 0.0
        with np.errstate(over='ignore'):
            return (amplitudes * np.exp(omega * delays_imag)) ** 2

    def rms_delay_spread(self) -> float:
        """Energy-weighted spread of the arrival delays, in seconds.

        The second central moment of the power delay profile: delays weighted
        by ``amplitude**2``, about their weighted mean. It measures how much
        the arrival pattern smears a pulse in time — the width the multipath
        gives an impulse — so it bounds the time resolution any processing of
        this channel can have, whatever the processing is for: the smearing
        of a transmitted pulse, the length a replica or matched filter has to
        cover, the interval a symbol would have to exceed to avoid
        overlapping its neighbour.

        Prefer it to the peak-to-peak spread ``ptp(delays)``, which is set by
        whichever ray arrives last no matter how faint: on a 1 km
        bottom-to-bottom path in 1000 m of water at 40 kHz the two differ by
        more than two orders of magnitude, because a path tens of dB down
        lands seconds late while almost all the energy arrives within a
        millisecond of the first.

        Returns ``0.0`` for a single arrival, and for arrivals carrying no
        energy at all. A non-finite delay or amplitude propagates: the result
        is ``nan``, not a spread computed from whatever else was finite.

        Notes
        -----
        Its reciprocal is the frequency scale over which the transfer
        function decorrelates: a widely dispersed arrival pattern fades over
        a narrow band, which is why a broadband view of such a channel shows
        structure far finer than the band. The constant relating the two is a
        correlation-threshold convention rather than a law, so it is left to
        the caller to state.
        """
        delays = np.asarray(self.delays, dtype=float).ravel()
        power = self._arrival_power()
        total = float(power.sum())
        if delays.size < 2 or total <= 0.0:
            return 0.0
        weights = power / total
        mean = float((weights * delays).sum())
        return float(np.sqrt((weights * (delays - mean) ** 2).sum()))

    def energy_support(self, fraction: float = 0.999) -> float:
        """Delay span holding ``fraction`` of the arrival energy, in seconds.

        Measured from the first arrival to the one by which ``fraction`` of
        the received energy has arrived. It answers the question a synthesis
        window asks — how long does the response have to be? — which neither
        of the other two measures does: ``ptp(delays)`` is an extremum, moved
        by one faint straggler however little it carries, and
        :meth:`rms_delay_spread` is a second moment, a width rather than a
        span the energy fits inside.

        Parameters
        ----------
        fraction : float, default 0.999
            Share of the total energy the span must hold, in ``(0, 1]``.
            ``1.0`` is the peak-to-peak span. The default leaves a thousandth
            of the energy — 30 dB down — outside.

        Returns
        -------
        float
            Seconds. ``0.0`` for a single arrival and for arrivals carrying
            no energy at all; ``nan`` if a delay or amplitude is non-finite,
            rather than a span computed from whatever else was finite.
        """
        fraction = float(fraction)
        if not 0.0 < fraction <= 1.0:
            raise ConfigurationError(
                f"Arrivals.energy_support: fraction={fraction:g} is not a "
                f"share of the energy. Pass 0 < fraction <= 1 (1.0 spans "
                f"every arrival, i.e. the peak-to-peak delay).")
        delays = np.asarray(self.delays, dtype=float).ravel()
        power = self._arrival_power()
        if delays.size < 2:
            return 0.0
        if not (np.all(np.isfinite(delays)) and np.all(np.isfinite(power))):
            return float('nan')
        total = float(power.sum())
        if total <= 0.0:
            return 0.0
        order = np.argsort(delays)
        delays = delays[order]
        # The cumulative share is monotone, so the first entry at or above
        # the target is the last arrival that has to fit. Rounding can leave
        # the final entry a hair under 1.0, which would put the index one
        # past the end, so clamp it.
        cumulative = np.cumsum(power[order]) / total
        cut = min(int(np.searchsorted(cumulative, fraction, side='left')),
                  delays.size - 1)
        return float(delays[cut] - delays[0])

    def _record_fold_notice(self, record: float) -> Optional[str]:
        """What a record shorter than the arrival span costs, or ``None``.

        Arrivals past the end of the record are not dropped — an inverse FFT
        is circular, so they land back on the early part of the trace, where
        they read as extra early paths. The level they return at is the one
        number that says whether that matters, so it is stated rather than
        left to be discovered in the trace.

        Returns the text rather than raising it: a hand-counted
        ``stacklevel`` has to be 2, which means the warning belongs in the
        method the caller actually called, not in a helper below it.
        """
        delays = np.asarray(self.delays, dtype=float).ravel()
        if delays.size < 2 or not np.all(np.isfinite(delays)):
            return None
        span = float(delays.max() - delays.min())
        # The fold itself is measured by the shared helper, which Bellhop's
        # BROADBAND run also uses on its default grid.
        return _fold_notice(
            delays, self._arrival_power(), record,
            who="Arrivals.synthesis_band",
            remedy=(f"Pass energy_fraction=1.0 (or record={span:g}) to hold "
                    f"every arrival, or drop the tail outright with "
                    f"in_delay_window / top_n_by_amplitude rather than "
                    f"folding it."))

    def synthesis_band(
        self,
        *,
        bandwidth: float,
        centre: Optional[float] = None,
        record: Optional[float] = None,
        energy_fraction: Optional[float] = None,
        margin: float = 1.2,
    ) -> np.ndarray:
        """Frequency grid wide enough to synthesise these arrivals un-aliased.

        A record built by an inverse FFT is ``1/df`` long, so the frequency
        spacing — not the bandwidth — decides how much multipath the trace
        can hold. Choose it without looking at the arrivals and the late
        paths wrap onto the early ones, which reads as extra arrivals rather
        than as a mistake.

        The window is the primitive here and the spacing follows from it, the
        way the textbook formulation puts it: "it is convenient to properly
        select the time windowing T and sampling dt needed to represent the
        response at all the receivers. This, in turn, constrains the
        frequency sampling" (Jensen, Kuperman, Porter and Schmidt,
        *Computational Ocean Acoustics*, sect. 8.2). Pass ``record`` to state
        that window. Leave it out and it is derived from the arrivals: the
        span holding ``energy_fraction`` of their energy
        (:meth:`energy_support`), times ``margin``.

        Parameters
        ----------
        bandwidth : float
            Width of the band to synthesise (Hz), centred on ``centre``.
        centre : float, optional
            Band centre (Hz); defaults to this result's own frequency.
        record : float, optional
            Length of the record to synthesise (s), stated rather than
            derived. Passing it with ``energy_fraction`` is refused — they
            are two answers to one question.
        energy_fraction : float, optional
            Share of the arrival energy the derived window must hold
            (default 0.999). Lower it for a shorter record and a coarser
            grid; whatever falls outside wraps, at a level this reports.
        margin : float, optional
            Headroom on the derived span (default 1.2), so the last arrival
            inside it lands within the record rather than on its final
            sample. It is also the number of frequency samples per
            interference fringe: two paths ``dtau`` apart beat in ``|H(f)|``
            with period ``1/dtau``, and a record ``margin * dtau`` long
            samples that at ``df = 1/(margin * dtau)`` — ``margin`` points
            per fringe. 1.2 is enough for the inverse FFT, which needs only
            that the record hold the arrivals; it is not enough to LOOK at
            the transfer function, whose curve between samples is then the
            plotter's, not the model's. Raise it to 8 or so for a drawn
            ``|H(f)|``. Not used when ``record`` is given.

        Returns
        -------
        ndarray
            Ascending frequencies to hand to a broadband run.

        Warns
        -----
        UserWarning
            When the record does not reach the last arrival, giving how many
            arrivals fold back and the level they fold in at.

        Notes
        -----
        Sizing the window to the last arrival however faint is what makes
        this expensive, because the peak-to-peak span is an extremum: on a
        1 km bottom-mounted link at 40 kHz it asks for some 440 000 bins to
        hold one ray around 200 dB down, where 0.999 of the energy is spanned
        by a few milliseconds. Trading that tail for a shorter record is a
        choice rather than an approximation, so it is made explicitly and its
        cost is reported instead of absorbed. Filtering the arrivals first
        (:meth:`in_delay_window`, :meth:`top_n_by_amplitude`) drops the tail
        outright rather than folding it.

        Folding is a property of the FREQUENCY route, not of the model.
        Sampling ``H(f)`` every ``df`` and inverse-transforming reproduces the
        true response repeated every ``1/df``, so whatever does not fit lands
        back at the wrong time. A ray model does not have to take that route:
        "the ray/beam process calculates the amplitudes and travel-times of
        all the echoes and can therefore calculate the received timeseries by
        simply summing up the echoes" (Bellhop User Guide sect. 9). That is
        ``RunMode.TIME_SERIES`` (:func:`uacpy.models.bellhop.delayandsum`),
        where an echo past the window is omitted rather than folded — the
        honest truncation, at the cost of giving up the transfer function.

        There is a third way, which keeps the whole arrival set on the
        frequency route and makes the wrap-around harmless instead: displace the
        frequency contour to ``w + i*delta``, which damps the synthesised
        trace by ``exp(-delta*t)``, so energy that wraps a full record length
        returns ``exp(-delta*T)`` down and the damping is undone on the trace
        afterwards. Mallick and Frazer put ``delta = log(50)/T`` — a factor of
        50 — and warn against more, which invents arrivals; the vendored
        OASES does exactly that (``third_party/oases/src/unoasp22.f``,
        ``OMEGIM``). It is not what this does, because it needs the transfer
        function evaluated at COMPLEX frequency and Bellhop takes a real one.
        Note also that the contour MAGNIFIES aliasing from earlier windows,
        so it additionally requires the record to start before the first
        arrival.
        """
        bandwidth = float(bandwidth)
        _require_positive(bandwidth, "Arrivals.synthesis_band bandwidth",
                          hint="Hz")
        if centre is None:
            centre = self.f0
            if centre is None:
                raise ConfigurationError(
                    "Arrivals.synthesis_band: this result carries no "
                    "frequency, so the band has no centre. Pass centre= (Hz).")
        centre = float(centre)
        _require_positive(centre, "Arrivals.synthesis_band centre", hint="Hz")
        # A positive centre and a positive width still describe a band that
        # runs through 0 Hz into negative frequency when the width exceeds
        # twice the centre. ``Source`` refuses those, but a model run accepts
        # them and returns an H that is not conjugate-symmetric, which an IFFT
        # then turns into a complex trace — so refuse the band where it is
        # built rather than leave it to be noticed downstream.
        if centre - bandwidth / 2.0 <= 0.0:
            raise ConfigurationError(
                f"Arrivals.synthesis_band: a {bandwidth:g} Hz band centred on "
                f"{centre:g} Hz starts at "
                f"{centre - bandwidth / 2.0:g} Hz, at or below 0 Hz — there "
                f"is no field to synthesise there. Narrow bandwidth= below "
                f"{2.0 * centre:g} Hz, or pass centre= high enough to carry "
                f"the band.")
        if record is not None and energy_fraction is not None:
            raise ConfigurationError(
                "Arrivals.synthesis_band: record= and energy_fraction= are "
                "two answers to one question — how long the record has to "
                "be. Pass record= (s) to state the window, or "
                "energy_fraction= to derive it from these arrivals.")
        if record is not None:
            record = float(record)
            _require_positive(record, "Arrivals.synthesis_band record",
                              hint="s")
        else:
            margin = float(margin)
            if margin < 1.0:
                raise ConfigurationError(
                    f"Arrivals.synthesis_band: margin={margin:g} would size "
                    f"the record SHORTER than the span it has to hold, which "
                    f"is the aliasing this method exists to prevent. Pass "
                    f"margin >= 1 (1.2 leaves the last arrival inside the "
                    f"window off the final sample).")
            support = self.energy_support(
                0.999 if energy_fraction is None else energy_fraction)
            if not np.isfinite(support):
                raise ConfigurationError(
                    "Arrivals.synthesis_band: these arrivals carry a "
                    "non-finite delay or amplitude, so the span they occupy "
                    "is undefined and no window can be derived from them. "
                    "Pass record= (s) to state one.")
            # A single arrival spans no time, and a grid still needs two
            # points to define a spacing: fall back to the shortest record
            # that holds it.
            record = max(margin * support, 1.0 / bandwidth)
        notice = self._record_fold_notice(record)
        if notice is not None:
            warnings.warn(notice, UserWarning, stacklevel=2)
        n_freq = max(int(np.ceil(bandwidth * record)) + 1, 2)
        return np.linspace(centre - bandwidth / 2.0,
                           centre + bandwidth / 2.0, n_freq)


class Rays(Result):
    """Ray paths from Bellhop (any backend).

    Pure data container: a list of ray polylines plus the geometric
    context of the run. Filtering helpers return new ``Rays`` objects;
    none of them call back into a solver. To compute "rays at a
    receiver" use :meth:`uacpy.models.PropagationModel.compute_eigenrays`, which
    runs Bellhop's eigenray solver (``RunType='E'``).

    Attributes
    ----------
    rays : list
        Ray dicts with ``r``, ``z``, ``alpha``, ``n_top_bounces``,
        ``n_bot_bounces``. **Polyline coordinates ``r`` (range) and
        ``z`` (depth) are in metres**; ``alpha`` is the launch angle
        in degrees. Bellhop writes the polyline as ``ray2D%x`` in metres
        (``Bellhop/WriteRay.f90:45``) behind a take-off angle already
        converted to degrees (``Bellhop/bellhop.f90:263``), and the reader
        (:func:`uacpy.io.oalib_reader.read_ray_file`) passes both through
        unconverted — so downstream helpers such as
        :meth:`filter_by_miss_distance` work in metres without any
        unit detection.
    is_eigen : bool
        ``True`` for output of Bellhop's eigenray solver (``RunType='E'``),
        ``False`` for a regular ray fan (``RunType='R'``). Set by the
        wrapper from the run type, not by post-processing.
    receiver_depths, receiver_ranges : ndarray or None
        Receiver geometry the run targeted, when available. ``None``
        when the ``Rays`` came from a standalone reader call without
        receiver context.
    """
    field_type = "rays"

    def __init__(
        self,
        *,
        rays: List[Any],
        is_eigen: bool = False,
        receiver_depths: Optional[np.ndarray] = None,
        receiver_ranges: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rays = list(rays)
        self.is_eigen = bool(is_eigen)
        self.receiver_depths = (
            np.atleast_1d(np.asarray(receiver_depths, dtype=float))
            if receiver_depths is not None else None
        )
        self.receiver_ranges = (
            np.atleast_1d(np.asarray(receiver_ranges, dtype=float))
            if receiver_ranges is not None else None
        )

    def _repr_extra(self) -> str:
        kind = 'eigenrays' if self.is_eigen else 'rays'
        return f"n_{kind}={len(self.rays)}"

    # ------------------------------------------------------------------
    # Filtering helpers — pure data subsets. ``is_eigen`` is preserved
    # (a subset of a fan stays a fan; a subset of eigenrays stays
    # eigenrays). None of these accept receiver coordinates: geometric
    # "rays at a receiver" is what ``PropagationModel.compute_eigenrays`` is for.
    # ------------------------------------------------------------------

    def filter(self, predicate) -> 'Rays':
        """Return a new ``Rays`` keeping rays for which ``predicate(ray)`` is true."""
        kept = [r for r in self.rays if predicate(r)]
        return self._spawn(kept)

    def filter_by_bounces(
        self,
        kind: Optional[str] = None,
        top: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
        bot: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
    ) -> 'Rays':
        """Subset by multipath component.

        ``kind`` ∈ ``{'direct', 'surface', 'bottom', 'both'}`` keeps a
        qualitative bounce class.

        ``top`` / ``bot`` further constrain the exact bounce count on
        each boundary:

        * ``None``       — any count
        * ``int``        — exact match (e.g. ``top=2``)
        * ``(lo, hi)``   — closed range; ``None`` on either end is
                           unbounded. ``bot=(1, None)`` keeps rays with
                           at least one bottom bounce; ``top=(0, 1)``
                           keeps 0–1 surface bounces.
        """
        if self.rays and not any(
            'n_top_bounces' in r or 'n_bot_bounces' in r for r in self.rays
        ):
            warnings.warn(
                "Rays.filter_by_bounces: rays carry no bounce counts, so "
                "every ray classifies as 'direct'. A .ray file read through "
                "uacpy.io.read_ray_file always supplies them; a hand-built "
                "Rays must set 'n_top_bounces' / 'n_bot_bounces' per ray.",
                UserWarning, stacklevel=2,
            )
        pred = _bounce_predicate(kind, top, bot)
        return self.filter(
            lambda r: pred(
                int(r.get('n_top_bounces', 0) or 0),
                int(r.get('n_bot_bounces', 0) or 0),
            )
        )

    def filter_by_launch_angle(
        self,
        min_deg: Optional[float] = None,
        max_deg: Optional[float] = None,
    ) -> 'Rays':
        """Keep rays whose launch angle ``alpha`` is within ``[min_deg, max_deg]``."""
        if self.rays and not any('alpha' in r for r in self.rays):
            warnings.warn(
                "Rays.filter_by_launch_angle: rays carry no launch angles, "
                "so the filter drops every ray. A .ray file read through "
                "uacpy.io.read_ray_file always supplies 'alpha'; a "
                "hand-built Rays must set it per ray.",
                UserWarning, stacklevel=2,
            )
        def pred(ray):
            a = ray.get('alpha')
            if a is None:
                return False
            if min_deg is not None and a < min_deg:
                return False
            if max_deg is not None and a > max_deg:
                return False
            return True
        return self.filter(pred)

    def filter_nfirst(
        self,
        n: int = 10
    ) -> 'Rays':
        """Keep only the first ``n`` rays."""
        return self._spawn(self.rays[:n])

    def _miss_distance_to(
        self, ray, target_range_m: float, target_depth_m: float,
    ) -> Tuple[float, int]:
        """Closest-approach miss distance and its index along the polyline.

        Ray polylines are required to carry ``r`` / ``z`` in **metres**
        (see :class:`Rays` docstring). The Bellhop reader in
        :mod:`uacpy.io.oalib_reader` already preserves Bellhop's native
        metres, so no unit-detection heuristic is needed here.
        """
        r = np.asarray(ray.get('r', []))
        z = np.asarray(ray.get('z', []))
        if len(r) == 0:
            return float('inf'), 0
        d2 = (r - target_range_m) ** 2 + (z - target_depth_m) ** 2
        k = int(np.argmin(d2))
        return float(np.sqrt(d2[k])), k

    def _resolve_target(
        self,
        target_range_m: Optional[float],
        target_depth_m: Optional[float],
    ) -> Tuple[float, float]:
        """Default target to the receiver context when this Rays was built
        from a single-point eigenray query."""
        if target_range_m is None:
            if self.receiver_ranges is None or len(self.receiver_ranges) != 1:
                raise ConfigurationError(
                    "Rays.miss-distance helpers: target_range_m must be "
                    "supplied unless this Rays carries a single-point "
                    "receiver context."
                )
            target_range_m = float(self.receiver_ranges[0])
        if target_depth_m is None:
            if self.receiver_depths is None or len(self.receiver_depths) != 1:
                raise ConfigurationError(
                    "Rays.miss-distance helpers: target_depth_m must be "
                    "supplied unless this Rays carries a single-point "
                    "receiver context."
                )
            target_depth_m = float(self.receiver_depths[0])
        return target_range_m, target_depth_m

    def filter_by_miss_distance(
        self,
        max_miss: float,
        target_range_m: Optional[float] = None,
        target_depth_m: Optional[float] = None,
    ) -> 'Rays':
        """Keep rays whose closest approach to the target is ``≤ max_miss``.

        Each kept ray gets a ``miss_distance_m`` entry attached. Target
        defaults to the single-point receiver this ``Rays`` was built for.
        """
        tr, td = self._resolve_target(target_range_m, target_depth_m)
        kept = []
        for ray in self.rays:
            miss, _ = self._miss_distance_to(ray, tr, td)
            if miss <= max_miss:
                ray = dict(ray)
                ray['miss_distance_m'] = miss
                kept.append(ray)
        return self._spawn(kept)

    def sorted_by_miss(
        self,
        target_range_m: Optional[float] = None,
        target_depth_m: Optional[float] = None,
    ) -> 'Rays':
        """Return rays sorted by ascending miss-distance to the target.

        Each ray gets ``miss_distance_m`` attached. Target defaults to
        the single-point receiver this ``Rays`` was built for. Compose
        with ``filter_nfirst`` to cap, or ``truncate_at_receiver`` to
        clip polylines.
        """
        tr, td = self._resolve_target(target_range_m, target_depth_m)
        scored = []
        for ray in self.rays:
            miss, _ = self._miss_distance_to(ray, tr, td)
            ray = dict(ray)
            ray['miss_distance_m'] = miss
            scored.append((miss, ray))
        scored.sort(key=lambda t: t[0])
        return self._spawn([r for _, r in scored])

    def top_n_by_miss(
        self,
        n: int,
        target_range_m: Optional[float] = None,
        target_depth_m: Optional[float] = None,
    ) -> 'Rays':
        """Return the ``n`` rays with smallest miss-distance to the target.

        Equivalent to ``self.sorted_by_miss(...).filter_nfirst(n)``.
        Target defaults to the single-point receiver this ``Rays`` was
        built for.
        """
        return self.sorted_by_miss(target_range_m, target_depth_m).filter_nfirst(n)

    def truncate_at_receiver(
        self,
        target_range_m: Optional[float] = None,
        target_depth_m: Optional[float] = None,
    ) -> 'Rays':
        """Clip each ray polyline at its closest-approach index.

        Target defaults to the single-point receiver this ``Rays`` was
        built for. Useful before plotting eigenrays so each path stops
        at the receiver instead of running off to its full extent.
        """
        tr, td = self._resolve_target(target_range_m, target_depth_m)
        clipped = []
        for ray in self.rays:
            miss, k = self._miss_distance_to(ray, tr, td)
            ray = dict(ray)
            ray['miss_distance_m'] = miss
            r = np.asarray(ray.get('r', []))
            z = np.asarray(ray.get('z', []))
            if k + 1 < len(r):
                ray['r'] = r[:k + 1]
                ray['z'] = z[:k + 1]
            clipped.append(ray)
        return self._spawn(clipped)

    def _spawn(self, rays: List[Any]) -> 'Rays':
        """Build a new ``Rays`` from a subset, preserving identification."""
        return Rays(
            rays=rays,
            is_eigen=self.is_eigen,
            receiver_depths=self.receiver_depths,
            receiver_ranges=self.receiver_ranges,
            **self.id_kwargs(),
        )
