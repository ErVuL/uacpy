"""Ray/eigenray and arrival result types."""

from __future__ import annotations

import warnings
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union

from uacpy.core.exceptions import ConfigurationError

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


class Arrivals(Result):
    """Ray arrivals from Bellhop — a flat list of arrival events.

    Each arrival is a dict with: ``delay``, ``amplitude``, ``phase``,
    ``n_top_bounces``, ``n_bot_bounces``, ``src_angle``, ``rcv_angle``,
    ``kind`` ('direct' / 'surface' / 'bottom' / 'both'), plus the cell
    of origin (``src_idx``, ``depth_idx``, ``range_idx``) so multi-cell
    runs can be filtered back to one cell if needed.

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
        """Return a copy sorted by ``amplitude`` (descending by default)."""
        return self._spawn(sorted(self.arrivals,
                                  key=lambda a: a['amplitude'],
                                  reverse=bool(descending)))

    def top_n_by_amplitude(self, n: int) -> 'Arrivals':
        """Keep the ``n`` loudest arrivals."""
        return self._spawn(self.sorted_by_amplitude(descending=True)
                           .arrivals[:int(n)])


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
