"""The Field side of the broadband-to-time-series IFFT synthesis behind
:meth:`Field.to_time_trace` and :meth:`Field.synthesize_time_series`: what
it reads off the Field (the frequency axis, the phase reference, the stamped
speeds and sample count, the cell coordinates) and the Field it builds. The
arrays are synthesised by :mod:`uacpy.acoustic_signal._synthesis`."""

from __future__ import annotations

import warnings

import numpy as np
from typing import Optional

from uacpy.acoustic_signal._synthesis import (
    SynthesisPlan, record_edge_finding, record_edges_are_judgeable,
    record_lead, record_start, reduced_record_starts,
    refuse_record_shorter_than_waveform, synthesis_grid, synthesize_cells,
    synthesize_trace, warn_band_edge_cuts_spectrum, warn_record_edge_energy,
    warn_unsolved_bins, waveform_synthesis_setup, zero_no_mode_bins)
from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.exceptions import ConfigurationError, NumericsWarning
from uacpy.core._validate import require_positive_finite_scalar
from uacpy.core._grid import nearest_index_on_axis
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.results._base import PhaseReference
from uacpy.core.results.field import Field


def _synthesis_plan(
    tf: "Field",
    *,
    window: str,
    nfft: Optional[int],
    sample_rate: Optional[float],
    who: str,
) -> SynthesisPlan:
    """Validate ``tf``'s frequency axis and size the synthesis grid that every
    cell of the Field shares (:class:`~uacpy.acoustic_signal._synthesis.
    SynthesisPlan`).

    The Field half of the IFFT synthesis: it reads the frequency axis, the
    phase reference and the model's reported sample count
    (:attr:`Field.synthesis_floor`) off ``tf`` and
    hands the grid itself to
    :func:`~uacpy.acoustic_signal._synthesis.synthesis_grid`, which the
    array-level :func:`uacpy.acoustic_signal.synthesize_time_series` shares.
    ``_ifft_to_trace`` (one cell) and ``_synthesize_time_series`` (every cell)
    both build on it, so the paths cannot drift apart. ``who`` is the
    public entry point's name and prefixes every diagnostic.
    """
    _refuse_time_domain_native(tf, who)
    return synthesis_grid(
        np.asarray(tf.coords['frequency'], dtype=float), window=window,
        nfft=nfft, sample_rate=sample_rate,
        n_samples_floor=int(tf.synthesis_floor or 0), who=who)


def _refuse_time_domain_native(tf: "Field", who: str) -> None:
    """Refuse a Field whose payload is ``p(t)`` or the transform of one."""
    if tf.phase_reference == 'time_domain_native':
        raise ConfigurationError(
            f"{who}: this field is tagged phase_reference="
            f"'time_domain_native' — it is p(t), or the spectrum of a p(t) "
            f"(SPARC's H(f), or Field.to_transfer_function of a synthesised "
            f"trace), which still carries the source spectrum or band window "
            f"the trace was made with. Synthesising it again would apply "
            f"that a second time. Read the time-domain Field itself (a "
            f"RunMode.TIME_SERIES run, or the trace it was transformed "
            f"from), or synthesise from the model's BROADBAND H(f)."
        )


def _stamped_speeds(tf: "Field"):
    """``(c_max, c0)`` a synthesis record is placed by
    (:func:`~uacpy.acoustic_signal._synthesis.record_start`), 0 where not
    stated: the fastest PHYSICAL speed of the paths the producing model
    traces (``speeds.water_max``), else the fastest speed of the waveguide
    its run resolved (``speeds.waveguide_max``); and the sea-surface water
    speed ``speeds.surface`` (Bellhop)."""
    speeds = tf.speeds
    if speeds is None:
        return 0.0, 0.0
    c_max = float(speeds.water_max or speeds.waveguide_max or 0.0)
    c0 = float(speeds.surface or 0.0)
    return c_max, c0


def _stamped_slow_speed(tf: "Field") -> float:
    """The slowest water speed a synthesis record must reach past
    (:func:`~uacpy.acoustic_signal._synthesis.record_start`), 0 where not
    stated: ``speeds.water_min``, else the surface speed, else the slowest
    compressional speed of the waveguide the run resolved."""
    speeds = tf.speeds
    if speeds is None:
        return 0.0
    return float(speeds.water_min or speeds.surface
                 or speeds.waveguide_min or 0.0)


def _ifft_to_trace(
    tf: "Field",
    *,
    depth: Optional[float],
    range: Optional[float],
    source_spectrum: Optional[np.ndarray],
    window: str,
    nfft: Optional[int],
    t_start: Optional[float],
    sample_rate: Optional[float] = None,
    who: str = 'Field.to_time_trace',
    pulse_s: Optional[float] = None,
    waveform: Optional[np.ndarray] = None,
) -> "Field":
    """One (depth, range) cell of a broadband Field synthesised into a
    time-domain trace Field, at exactly ``sample_rate`` when one is given.
    ``waveform``, the transmitted signal behind ``source_spectrum`` when
    there is one, is refused when the ``1/Δf`` record cannot hold it.

    Evaluates the Fourier synthesis ``p(t) = 2·Re Σ H(f_k)·S(f_k)·
    e^{2πi f_k t}·df`` — a Riemann sum of the continuous inverse
    transform, so the amplitude is independent of ``nfft`` and of the
    bin grid. ``source_spectrum`` must therefore be the *continuous*
    source spectrum sampled at the Field frequencies (a raw DFT times
    the source sampling interval); ``None`` synthesizes the
    band-limited impulse response. A 0 Hz bin, which has no
    negative-frequency twin, is counted once rather than doubled.

    The record is exactly ``1/Δf`` long (``Δf = f[1] - f[0]``) — a longer
    record requires a finer frequency grid, not a larger ``nfft`` — so the
    frequency axis must be uniformly spaced and ascending. The sum is
    evaluated at the record's samples exactly
    (:func:`~uacpy.acoustic_signal._synthesis.band_sum`), whatever the
    band's offset from a multiple of ``Δf``. Without ``sample_rate`` an
    auto-sized ``nfft`` keeps the largest data bin below Nyquist; an
    explicit ``nfft`` that would not is rejected.
    """
    data = tf.data                                # (n_d, n_r, n_f)
    depths = tf.coords['depth']
    ranges = tf.coords['range']
    n_d, n_r, _ = data.shape

    plan = _synthesis_plan(tf, window=window, nfft=nfft,
                           sample_rate=sample_rate, who=who)
    if waveform is not None:
        refuse_record_shorter_than_waveform(plan.df, waveform, sample_rate,
                                            who)
    warn_band_edge_cuts_spectrum(
        source_spectrum, window, who,
        freqs=np.asarray(tf.coords['frequency'], dtype=float),
        waveform=waveform, sample_rate=sample_rate)

    d_idx = (nearest_index_on_axis(depths, depth, 'depth', who=who)
             if depth is not None else n_d // 2)
    r_idx = (nearest_index_on_axis(ranges, range, 'range', who=who)
             if range is not None else 0)
    actual_depth = float(depths[d_idx])
    actual_range = float(ranges[r_idx])

    c_max, c0 = _stamped_speeds(tf)
    # ``pulse_s`` is the source waveform's duration (0 for the bare impulse
    # response, None for a raw source spectrum of unknown extent).
    time, trace = synthesize_trace(
        data[d_idx, r_idx, :], source_spectrum, plan, range=actual_range,
        t_start=t_start, c_max=c_max, c0=c0, c_slow=_stamped_slow_speed(tf),
        cell_label=f"depth {actual_depth:g} m, range {actual_range:g} m",
        who=who, pulse_s=pulse_s,
        no_mode_bins=tf.no_mode_bins)

    # The source Field's identity and metadata carry forward (output paths
    # attached under a pinned work_dir, the run settings), as on
    # every other derived Field.
    id_kwargs = tf.id_kwargs()
    # The payload is p(t) from here on, whatever convention H(f) carried.
    id_kwargs['phase_reference'] = PhaseReference.TIME_DOMAIN_NATIVE
    # A trace is another quantity than H(f): its coherence is decided anew.
    id_kwargs['coherent'] = None
    id_kwargs['metadata'].update({'source_model': tf.model})
    id_kwargs['synthesis_window'] = window
    if source_spectrum is None:
        # With no source spectrum the trace is sum H e^{2πift} Δf, the
        # band-limited impulse response, per second rather than a
        # pressure; with one it is the received p(t) in the parent's unit.
        id_kwargs.update(kind='impulse_response', unit='1/s')
    return tf.replace(
        data=trace,
        coords={'time': time},
        # The parent's pinned axes carry through (the accumulation contract
        # in the class doc), with this cell's coordinates added on top.
        pinned={**dict(tf.pinned),
                'depth': actual_depth, 'range': actual_range},
        **id_kwargs,
    )


def _cell_labels(tf: "Field"):
    """``label(i)`` naming row ``i`` of ``tf``'s cells flattened in axis
    order (any ``source_depth``, then depth, then range) by its
    coordinates."""
    names = [n for n in ('source_depth', 'depth', 'range') if n in tf.coords]
    axes = [np.asarray(tf.coords[n]) for n in names]
    shape = [a.size for a in axes]
    words = {'source_depth': 'source depth', 'depth': 'depth',
             'range': 'range'}

    def label(i):
        index = np.unravel_index(int(i), shape)
        return ", ".join(f"{words[n]} {float(a[k]):g} m"
                         for n, a, k in zip(names, axes, index))
    return label


def _record_placement(tf: "Field", ranges: np.ndarray, record_s: float,
                      bandwidth: float, t_start, who: str,
                      fold_report: Optional[dict] = None):
    """Where a synthesis's records sit on the range axis ``ranges``:
    ``(starts, origin, reduction_speed)``, ``starts`` one per range (s),
    ``origin`` what the result's time axis adds to ``n/fs`` and
    ``reduction_speed`` the speed it is reduced by (``None`` for an absolute
    axis).

    ``t_start`` a number: that shared start. ``None``: one shared start
    anchored on the range nearest the source (:func:`~uacpy.acoustic_signal.
    _synthesis.record_start`). ``'reduced'``: each range its own record
    (:func:`~uacpy.acoustic_signal._synthesis.reduced_record_starts`), on
    the reduced time ``t − |r|/c``. ``fold_report`` collects the shared
    start's late-arrival finding instead of warning it
    (:func:`~uacpy.acoustic_signal._synthesis.record_start`)."""
    c_max, c0 = _stamped_speeds(tf)
    if isinstance(t_start, str):
        if t_start != 'reduced':
            raise ConfigurationError(
                f"{who}: t_start must be a time in seconds, None (one shared "
                f"record placed from the nearest range) or 'reduced' (each "
                f"range its own record, on the reduced time t − r/c); got "
                f"{t_start!r}.")
        starts, speed, lead = reduced_record_starts(
            ranges, record_s, bandwidth=bandwidth, c_max=c_max, c0=c0,
            c_slow=_stamped_slow_speed(tf), who=who)
        return starts, -lead, speed
    if t_start is None:
        # One window for every cell, anchored on the range nearest the
        # source — the smallest |range|, which is ranges[0] only on an
        # ascending axis; anchored on a far cell, the near arrivals fall
        # before the window and wrap to the record's end. The record is 1/Δf
        # long whatever the sample rate.
        t_start = record_start(float(np.min(np.abs(ranges))), record_s,
                               bandwidth=bandwidth, c_max=c_max, c0=c0,
                               c_slow=_stamped_slow_speed(tf), who=who,
                               fold_report=fold_report)
    t_start = float(t_start)
    if not np.isfinite(t_start):
        raise ConfigurationError(
            f"{who}: t_start must be finite (s); got {t_start!r}.")
    return np.full(ranges.size, t_start), t_start, None


def _synthesize_time_series(
    tf: "Field",
    *,
    source_waveform: np.ndarray,
    sample_rate: float,
    t_start,
    placement_checks: bool = True,
    window: str,
    who: str = 'Field.synthesize_time_series',
    exposure: bool = False,
    fold_report: Optional[dict] = None,
) -> "Field":
    """Convolve every grid cell of a broadband Field with a source waveform.

    Output: a time-domain Field with ``coords={'depth', 'range', 'time'}``,
    sampled at exactly ``sample_rate``: ``sample_rate/Δf`` samples per
    ``1/Δf`` record when that is a whole number, else the next whole number
    above it. Amplitude is grid-independent: with ``window=None`` a flat
    ``H ≡ 1`` reproduces the (band-limited) source waveform; any other
    window filters it. ``t_start='reduced'`` gives each range its own
    record on the reduced time axis (:func:`_record_placement`). ``who`` is
    the public method the caller called, and names every diagnostic;
    ``exposure`` words the band-edge warning for an exposure
    (:func:`~uacpy.acoustic_signal._synthesis.warn_band_edge_cuts_spectrum`);
    ``fold_report`` collects the shared record's fold findings for the
    caller to say — the late arrival (:func:`_record_placement`), a
    range span longer than the record (``'range_span'``: the span in m and
    s, the record in s) and energy at an auto-placed record's edge
    (``'record_edge'``: its level in dB re the trace maximum, ``'end'`` or
    ``'start'``, the record's first and last time) — instead of warning
    them.
    """
    wf = np.asarray(source_waveform, dtype=float).ravel()
    n_src = len(wf)
    lead, n_f = tf.data.shape[:-1], tf.data.shape[-1]
    n_r = lead[-1]
    ranges = np.asarray(tf.coords['range'])
    _refuse_time_domain_native(tf, who)
    source_spectrum, plan = waveform_synthesis_setup(
        np.asarray(tf.coords['frequency'], dtype=float), wf, sample_rate,
        window=window, n_samples_floor=int(tf.synthesis_floor or 0),
        who=who, exposure=exposure)
    df = plan.df
    bandwidth = float(np.ptp(plan.freqs)) if len(plan.freqs) > 1 else 0.0
    t_start_estimated = t_start is None or isinstance(t_start, str)
    starts, origin, reduction_speed = _record_placement(
        tf, ranges, 1.0 / df, bandwidth, t_start, who,
        fold_report=fold_report)

    n_cells = int(np.prod(lead))
    spectra = zero_no_mode_bins(
        tf.data.reshape(n_cells, n_f), tf.no_mode_bins,
        who=who, frequencies=plan.freqs)
    warn_unsolved_bins(spectra, who=who, cell_label=_cell_labels(tf),
                       frequencies=plan.freqs)
    out, elapsed, head_dB, tail_dB = synthesize_cells(
        spectra, source_spectrum, plan, np.tile(starts, n_cells // n_r),
        who=who)
    time_vec = origin + elapsed
    out = out.reshape(lead + (plan.n_samples,))
    # The two placement checks ask whether each arrival lands inside its
    # window. A caller that integrates each cell over its whole periodic
    # record (an exposure, a peak) reads the same number wherever the
    # window starts, and skips them.
    if (placement_checks and t_start_estimated
            and record_edges_are_judgeable(
                1.0 / df, n_src / float(sample_rate),
                record_lead(1.0 / df, bandwidth))):
        if fold_report is None:
            warn_record_edge_energy(head_dB, tail_dB, time=time_vec, df=df,
                                    who=who)
        else:
            edge = record_edge_finding(head_dB, tail_dB)
            if edge is not None:
                fold_report['record_edge'] = (
                    edge[0], edge[1], float(time_vec[0]),
                    float(time_vec[-1]))

    # All cells share one time window anchored at the nearest range;
    # arrivals for ranges further out than the window can hold wrap back
    # into early bins (DFT periodicity) — flag it rather than alias silently.
    if (placement_checks and reduction_speed is None and n_r > 1
            and time_vec.size > 1):
        # The water speed the producer stated (Bellhop's surface speed),
        # else the slowest speed of the waveguide the run resolved, which
        # spreads the arrivals over the span the most; the default without
        # either.
        speeds = tf.speeds
        c0 = float((speeds is not None
                    and (speeds.surface or speeds.waveguide_min))
                   or DEFAULT_SOUND_SPEED)
        span_s = float(ranges.max() - ranges.min()) / c0
        window_s = float(time_vec[-1] - time_vec[0])
        if span_s > window_s and fold_report is not None:
            # A run says this with what it measures on the trace, as one
            # fold (warn_if_circular_record_rings).
            fold_report['range_span'] = (
                float(ranges.max() - ranges.min()), span_s, window_s)
        elif span_s > window_s:
            # The remedies hold for both callers: this method, and a model's
            # TIME_SERIES run, which places one shared record and takes a
            # numeric t_start only.
            warnings.warn(
                f"{who}: the receiver range span "
                f"({ranges.max() - ranges.min():.0f} m ≈ {span_s:.2f}s of "
                f"travel time) exceeds the {window_s:.2f}s synthesis window "
                f"— far-range arrivals wrap back into early bins. Widen the "
                f"one shared window with a frequency grid of Δf ≤ "
                f"{1.0/span_s:.3g} Hz: on a TIME_SERIES run, output_duration "
                f"≥ {span_s:.2f}s sets that grid for you; a BROADBAND run "
                f"takes it from frequencies=. Or give each range its own "
                f"record on the reduced time t − r/c, which need hold only "
                f"the dispersion: Field.synthesize_time_series(..., "
                f"t_start='reduced') on a BROADBAND result.",
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

    # The source Field's identity, metadata and run settings carry forward
    # (see _ifft_to_trace).
    id_kwargs = tf.id_kwargs()
    # The payload is p(t) from here on, whatever convention H(f) carried.
    id_kwargs['phase_reference'] = PhaseReference.TIME_DOMAIN_NATIVE
    # A trace is another quantity than H(f): its coherence is decided anew.
    id_kwargs['coherent'] = None
    id_kwargs['metadata'].update({'source_waveform_sample_rate': sample_rate,
                                  'source_model': tf.model})
    id_kwargs['synthesis_window'] = window
    id_kwargs['reduction_speed'] = reduction_speed
    return tf.replace(
        data=out,
        coords={**{n: v for n, v in tf.coords.items() if n != 'frequency'},
                'time': time_vec},
        # The parent's pinned axes carry through (the accumulation contract
        # in the class doc); no axis collapses here, so nothing is added.
        pinned=dict(tf.pinned),
        **id_kwargs,
    )


def _impulse_response(
    tf: "Field",
    *,
    sample_rate: float,
    t_start,
    window: Optional[str],
) -> "Field":
    """Every cell's impulse response ``h(t)`` at exactly ``sample_rate``:
    the taps of :func:`~uacpy.acoustic_signal.
    impulse_response_from_transfer_function` over one ``1/Δf`` record per
    cell, placed by :func:`_record_placement`, times ``sample_rate`` — the
    density ``Σ H e^{2πift} Δf`` in 1/s, ``kind='impulse_response'``."""
    who = 'Field.impulse_response'
    _refuse_time_domain_native(tf, who)
    lead, n_f = tf.data.shape[:-1], tf.data.shape[-1]
    ranges = np.asarray(tf.coords['range'])
    freqs = np.asarray(tf.coords['frequency'], dtype=float)
    fs = require_positive_finite_scalar(sample_rate, who, "sample_rate",
                                        " Hz")
    # The grid's checks and its record length are the synthesis's.
    plan = synthesis_grid(freqs, window=window, nfft=None, sample_rate=fs,
                          n_samples_floor=0, who=who)
    bandwidth = float(np.ptp(freqs))
    starts, origin, reduction_speed = _record_placement(
        tf, ranges, 1.0 / plan.df, bandwidth, t_start, who)
    spectra = zero_no_mode_bins(
        tf.data.reshape(-1, n_f), tf.no_mode_bins, who=who,
        frequencies=freqs)
    warn_unsolved_bins(spectra, who=who, cell_label=_cell_labels(tf),
                       frequencies=freqs)
    # Deferred: acoustic_signal pulls scipy, and uacpy's public surface is
    # imported without it (test_lazy_imports).
    from uacpy.acoustic_signal.channel import (
        impulse_response_from_transfer_function,
    )
    _, taps = impulse_response_from_transfer_function(
        spectra.reshape(lead + (n_f,)) * plan.win, frequencies=freqs,
        sample_rate=fs,
        t_start=np.broadcast_to(starts, lead),
        n_samples=plan.n_samples)
    time_vec = origin + np.arange(plan.n_samples) / fs
    id_kwargs = tf.id_kwargs()
    id_kwargs['phase_reference'] = PhaseReference.TIME_DOMAIN_NATIVE
    id_kwargs['metadata'].update({'source_model': tf.model})
    id_kwargs['synthesis_window'] = window
    id_kwargs['reduction_speed'] = reduction_speed
    return tf.replace(
        data=taps * fs,
        coords={**{n: v for n, v in tf.coords.items() if n != 'frequency'},
                'time': time_vec},
        pinned=dict(tf.pinned), kind='impulse_response', unit='1/s',
        coherent=None, **id_kwargs)
