"""The Field side of the broadband-to-time-series IFFT synthesis behind
:meth:`Field.to_time_trace` and :meth:`Field.synthesize_time_series`: what
it reads off the Field (the frequency axis, the phase reference, the stamped
speeds and sample count, the cell coordinates) and the Field it builds. The
arrays are synthesised by :mod:`uacpy.acoustic_signal._synthesis`."""

from __future__ import annotations

import warnings

import numpy as np
from typing import Optional, Tuple

from uacpy.acoustic_signal._synthesis import (
    record_edges_are_judgeable, record_lead, record_start, synthesis_grid,
    synthesize_cells,
    synthesize_trace, warn_band_edge_cuts_spectrum, warn_record_edge_energy,
    warn_unsolved_bins, waveform_synthesis_setup)
from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.exceptions import ConfigurationError, NumericsWarning
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
) -> Tuple[np.ndarray, float, np.ndarray, float, int, np.ndarray]:
    """Validate ``tf``'s frequency axis and size the synthesis grid that every
    cell of the Field shares: returns ``(freqs, df, bin_indices,
    bin_offset_hz, nfft, win)``.

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
    who: str = 'to_time_trace',
    pulse_s: Optional[float] = None,
) -> "Field":
    """IFFT one (depth, range) cell of a broadband Field → time-domain trace Field.

    Evaluates the Fourier synthesis ``p(t) = 2·Re Σ H(f_k)·S(f_k)·
    e^{2πi f_k t}·df`` — a Riemann sum of the continuous inverse
    transform, so the amplitude is independent of ``nfft`` and of the
    bin grid. ``source_spectrum`` must therefore be the *continuous*
    source spectrum sampled at the Field frequencies (a raw DFT times
    the source sampling interval); ``None`` synthesizes the
    band-limited impulse response. A 0 Hz bin, which has no
    negative-frequency twin, is counted once rather than doubled.

    Places each model frequency at bin ``round(f / Δf)`` with
    ``Δf = f[1] - f[0]``, so the record length is exactly ``1/Δf`` — a longer
    record requires a finer frequency grid, not a larger ``nfft``. The
    frequency axis must therefore be uniformly spaced and ascending. A grid
    whose first bin is not itself a multiple of ``Δf`` lands offset by a
    common ``|δ| <= Δf/2``; the synthesis de-rotates the complex sum by
    ``exp(-2πiδt)``, which recovers the requested band exactly rather than a
    frequency-shifted copy of it. An auto-sized ``nfft`` always keeps the
    largest data bin below Nyquist; an explicit ``nfft`` that would not is
    rejected.
    """
    data = tf.data                                # (n_d, n_r, n_f)
    depths = tf.coords['depth']
    ranges = tf.coords['range']
    n_d, n_r, _ = data.shape

    plan = _synthesis_plan(tf, window=window, nfft=nfft,
                           sample_rate=sample_rate, who=who)
    warn_band_edge_cuts_spectrum(source_spectrum, window, who)

    d_idx = (nearest_index_on_axis(depths, depth, 'depth')
             if depth is not None else n_d // 2)
    r_idx = (nearest_index_on_axis(ranges, range, 'range')
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
        sub_cutoff_bins=int(tf.sub_cutoff_bins or 0))

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


def _synthesize_time_series(
    tf: "Field",
    *,
    source_waveform: np.ndarray,
    sample_rate: float,
    t_start: Optional[float],
    window: str,
    nfft: Optional[int],
) -> "Field":
    """Convolve every grid cell of a broadband Field with a source waveform.

    Output: a time-domain Field with ``coords={'depth', 'range', 'time'}``.
    ``nfft`` is sized so the output sample rate ``1/dt = nfft·df`` is at
    least ``sample_rate`` (rounded up to a power of two, so up to 2×
    finer); read the actual grid from ``coords['time']``. Amplitude is
    grid-independent: with ``window=None`` a flat ``H ≡ 1`` reproduces
    the (band-limited) source waveform; any other window filters it.
    """
    wf = np.asarray(source_waveform, dtype=float).ravel()
    n_src = len(wf)
    n_d, n_r, n_f = tf.data.shape
    depths = np.asarray(tf.coords['depth'])
    ranges = np.asarray(tf.coords['range'])
    _refuse_time_domain_native(tf, 'synthesize_time_series')
    source_spectrum, plan = waveform_synthesis_setup(
        np.asarray(tf.coords['frequency'], dtype=float), wf, sample_rate,
        window=window, nfft=nfft,
        n_samples_floor=int(tf.synthesis_floor or 0),
        who='synthesize_time_series')
    df, nfft = plan[1], plan[4]
    bandwidth = float(np.ptp(plan[0])) if len(plan[0]) > 1 else 0.0

    t_start_estimated = t_start is None
    if t_start_estimated:
        # One window for every cell, anchored on the range nearest the
        # source — the smallest |range|, which is ranges[0] only on an
        # ascending axis; anchored on a far cell, the near arrivals fall
        # before the window and wrap to the record's end. The record is 1/Δf
        # long whatever nfft is.
        c_max, c0 = _stamped_speeds(tf)
        t_start = record_start(float(np.min(np.abs(ranges))), 1.0 / df,
                               bandwidth=bandwidth,
                               c_max=c_max, c0=c0,
                               c_slow=_stamped_slow_speed(tf),
                               who='synthesize_time_series')

    n_cells = n_d * n_r
    spectra = tf.data.reshape(n_cells, n_f)
    warn_unsolved_bins(
        spectra, who='synthesize_time_series',
        cell_label=lambda i: (f"depth {float(depths[i // n_r]):g} m, "
                              f"range {float(ranges[i % n_r]):g} m"),
        sub_cutoff_bins=int(tf.sub_cutoff_bins or 0))
    out, time_vec, head_dB, tail_dB = synthesize_cells(
        spectra, source_spectrum, plan, t_start)
    out = out.reshape(n_d, n_r, nfft)
    if (t_start_estimated and time_vec is not None
            and record_edges_are_judgeable(
                1.0 / df, n_src / float(sample_rate),
                record_lead(1.0 / df, bandwidth))):
        warn_record_edge_energy(head_dB, tail_dB, time=time_vec, df=df,
                                 who='synthesize_time_series')

    # All cells share one time window anchored at the nearest range;
    # arrivals for ranges further out than the window can hold wrap back
    # into early bins (DFT periodicity) — flag it rather than alias silently.
    if n_r > 1 and time_vec is not None and time_vec.size > 1:
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
        if span_s > window_s:
            warnings.warn(
                f"synthesize_time_series: the receiver range span "
                f"({ranges.max() - ranges.min():.0f} m ≈ {span_s:.2f}s of "
                f"travel time) exceeds the {window_s:.2f}s synthesis window "
                f"— far-range arrivals wrap back into early bins. The window "
                f"is 1/Δf, so widen it with a frequency grid of "
                f"Δf ≤ {1.0/span_s:.3g} Hz; on a TIME_SERIES run "
                f"output_duration ≥ {span_s:.2f}s sets that grid for you "
                f"(BROADBAND takes the grid from frequencies= and ignores "
                f"output_duration).",
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
    return tf.replace(
        data=out,
        coords={'depth': depths, 'range': ranges, 'time': time_vec},
        # The parent's pinned axes carry through (the accumulation contract
        # in the class doc); no axis collapses here, so nothing is added.
        pinned=dict(tf.pinned),
        **id_kwargs,
    )
