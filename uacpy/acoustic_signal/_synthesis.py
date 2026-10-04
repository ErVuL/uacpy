"""The array half of the broadband-to-time-series IFFT synthesis: the
grid, the per-cell synthesis and its diagnostics, shared by
:func:`uacpy.acoustic_signal.synthesize_time_series`,
:meth:`uacpy.Field.to_time_trace`, :meth:`uacpy.Field.synthesize_time_series`
and :meth:`BeamformedField.to_time_trace`.

A package-internal module: its names are public to the package, not to
users.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Tuple

import numpy as np

from uacpy.acoustic_signal.windows import _taper
from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, NumericsWarning,
)
from uacpy.core._validate import UNIFORM_STEP_RTOL
from uacpy.core._warn_frames import USER_FRAME_SKIP


#: Elements of the largest scratch block a batched transform builds at
#: once: 4e6 complex128 ~ 64 MB.
_SCRATCH_BLOCK_ELEMS = 4_000_000


# Auto-sized IFFT length is ~sample_rate/df rounded up to a power of two, so a
# too-high sample_rate (or a too-fine frequency grid) can silently demand a
# multi-GB buffer and OOM the process. Cap the *auto* size at 2**26 ≈ 67 M
# samples (~1 GB complex) and raise instead; an explicit ``nfft=`` bypasses it.
MAX_SYNTHESIS_NFFT = 1 << 26


def synthesis_grid(
    freqs: np.ndarray,
    *,
    window: str,
    nfft: Optional[int],
    sample_rate: Optional[float],
    n_samples_floor: int,
    who: str,
) -> Tuple[np.ndarray, float, np.ndarray, float, int, np.ndarray]:
    """Validate a frequency axis and size the synthesis grid on it: returns
    ``(freqs, df, bin_indices, bin_offset_hz, nfft, win)``.

    Array-level: ``n_samples_floor`` is the least ``nfft`` the caller wants
    (a model's reported time-sample count, or 0). Shared by the Field
    synthesis and :func:`uacpy.acoustic_signal.synthesize_time_series`.
    """
    freqs = np.asarray(freqs, dtype=float)
    n_freq = freqs.size

    if n_freq < 2:
        raise ConfigurationError(
            f"{who}: need at least 2 frequencies for IFFT; got {n_freq}."
        )
    if float(np.min(freqs)) < 0.0:
        # The synthesis is one-sided: a negative bin has no place on it and
        # would be dropped from the sum without a word.
        raise ConfigurationError(
            f"{who}: the frequency axis reaches {float(np.min(freqs)):g} Hz; "
            f"the synthesis takes the one-sided band, 0 Hz and up.")

    # The transfer function is sampled at df_data, so the trace it can
    # represent without aliasing is exactly 1/df_data long. Refining df below
    # that (to force a longer window) has to invent the samples in between,
    # and linear interpolation of the spectrum is a convolution with a
    # triangular kernel — i.e. a sinc^2(pi df_data t) taper in time, which
    # progressively attenuates arrivals away from the anchor it is centred on.
    # Return the honest extent instead; a longer record needs a finer
    # frequency grid, which means more model runs.
    from uacpy.acoustic_signal.channel import _uniform_frequency_step
    df_data = _uniform_frequency_step(freqs, who=who)
    df = df_data

    # A DFT of spacing df can only carry frequencies at integer multiples of
    # df, so each model frequency lands at bin round(f/df). When f[0] is not
    # itself a multiple of df the whole band is placed offset by a common
    # ``bin_offset_hz`` (|offset| <= df/2); ``synthesize_traces`` removes it
    # exactly by de-rotating the complex sum, so the trace is the band the
    # caller asked for rather than a frequency-shifted copy of it.
    bin_indices = np.floor(freqs / df + 0.5).astype(int)
    bin_offset_hz = float(bin_indices[0] * df - freqs[0])
    # That the offset is COMMON is an assumption, and it fails on a knife
    # edge: when freqs[0]/df sits on the .5 boundary that np.floor(x + 0.5)
    # breaks, the first sample rounds one way and the rest the other, and
    # part of the band is placed a whole bin from where it belongs. The
    # result is silently wrong, not obviously broken — on a 25 Hz-4 kHz band
    # with df = 2/3 (freqs[0]/df = 37.5) a two-path SEL read 52.34 dB
    # against a true 54.01, on a 1500 ms record where nothing folds. Nudging
    # df by 0.005 Hz either way is exact, so it cannot be left to the
    # caller to notice. Checked rather than assumed.
    residual = np.abs(bin_indices * df - freqs - bin_offset_hz)
    if residual.size and float(np.max(residual)) > UNIFORM_STEP_RTOL * df:
        raise ConfigurationError(
            f"{who}: this frequency grid cannot be placed on a DFT of "
            f"spacing {df:g} Hz — freqs[0]/df = {freqs[0] / df:g} lands on "
            f"the rounding boundary, so the band would be split across two "
            f"bin offsets and the result would be wrong by decibels without "
            f"looking wrong. Shift the band start or df by a fraction of a "
            f"bin (a few parts in 1e3 of {df:g} Hz is enough), or build the "
            f"grid so freqs[0] is a multiple of df.")
    max_bin = int(bin_indices[-1])
    explicit_nfft = nfft is not None

    if nfft is None:
        # Floor the auto length at 4 bins per model frequency, and never below
        # a time-sample count the model already reported. On a baseband grid
        # the anti-aliasing minimum below is 2*max_bin + 2 = 2*n_freq, so the
        # floor leaves the trace time-oversampled ~2x rather than critically
        # sampled — the extra bins are zero-padding, which interpolates the
        # trace without changing its band.
        nfft_min = max(int(n_samples_floor), 4 * n_freq)
        nfft_target = max(nfft_min, 2 * max_bin + 2)
        if sample_rate is not None:
            nfft_target = max(nfft_target, int(np.ceil(sample_rate / df)))
        nfft = 1
        while nfft < nfft_target:
            nfft *= 2
        if nfft > MAX_SYNTHESIS_NFFT:
            raise ConfigurationError(
                f"{who}: the requested grid implies an "
                f"{nfft:,}-sample output (~{nfft * 16 / 1e9:.1f} GB), above the "
                f"{MAX_SYNTHESIS_NFFT:,}-sample safety cap. This is driven by "
                f"sample_rate={sample_rate!r} Hz against a frequency resolution "
                f"df={df:.4g} Hz (length ~ sample_rate/df). Lower sample_rate, "
                f"widen df (coarser frequency grid / shorter window), or pass an "
                f"explicit nfft= if you really need an output this large.",
                remediation="A typical fix is a smaller sample_rate.",
            )

    if explicit_nfft and max_bin >= nfft // 2:
        raise ConfigurationError(
            f"{who}: nfft={nfft} puts the highest data bin "
            f"({max_bin}, {freqs[-1]:.6g} Hz at Δf = {df:.6g} Hz) at or above "
            f"Nyquist (bin {nfft // 2}); those bins fold into the "
            f"negative-frequency half and alias onto the wrong frequencies. "
            f"Use nfft >= {2 * max_bin + 2}, or drop nfft= to size it "
            f"automatically.",
            remediation=f"Pass nfft={2 * max_bin + 2} or larger.",
        )

    win = _taper(window, n_freq, who=who)

    return freqs, df, bin_indices, bin_offset_hz, int(nfft), win


def warn_unsolved_bins(
    spectra: np.ndarray,
    *,
    who: str,
    cell_label: Optional[Callable[[int], str]] = None,
    sub_cutoff_bins: int = 0,
) -> None:
    """Warn about NaN bins in a batch of cell spectra ``(M, n_f)``.

    A NaN bin is a frequency the model did not solve, not one carrying no
    energy, so it is never zeroed: filling it would put a spectral notch the
    model never produced into a trace that then looks finite and ordinary.
    The NaNs are kept and propagate through the IFFT, which makes the whole
    trace no-data — a trace cannot be synthesised from a spectrum with holes
    in it.

    With ``cell_label`` (row index -> the cell's coordinates as text, e.g.
    ``'depth 10 m, range 1500 m'``) each affected cell is named in its own
    warning, and an all-NaN cell (one masked below the seafloor, say) gets
    its own wording, since nothing about it was solved. Without it — plain
    arrays, whose rows have no coordinates to name — one warning counts the
    affected cells. ``who`` is the public entry point's name and prefixes
    the diagnostic.

    ``sub_cutoff_bins`` is the count a normal-mode model records
    (:attr:`~uacpy.core.results.Field.sub_cutoff_bins`, Kraken) for bins
    below the lowest
    mode's cutoff. Re-running them cannot help — no mode propagates there —
    so when it is non-zero the warning names that cause and the remedies
    that do.
    """
    nan_bins = np.isnan(spectra)
    if cell_label is None:
        unsolved = np.any(nan_bins, axis=1)
        if np.any(unsolved):
            warnings.warn(
                f"{who}: {int(unsolved.sum())} of {unsolved.size} cell(s) of H "
                f"hold NaN bins, so their traces are NaN rather than carrying a "
                f"notch the model never produced. Re-run those frequencies, or "
                f"narrow the band to the bins that solved.",
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
        return
    all_nan = np.all(nan_bins, axis=1)
    n_f = spectra.shape[1]
    for i in np.flatnonzero(np.any(nan_bins, axis=1)):
        where = f"H(f) at {cell_label(int(i))}"
        if sub_cutoff_bins > 0 and not all_nan[i]:
            detail = (f"has {int(np.count_nonzero(nan_bins[i]))} of {n_f} "
                      f"NaN bins; {int(sub_cutoff_bins)} of the band's bins "
                      f"lie below the lowest mode's cutoff, where a "
                      f"normal-mode model has no field, so the synthesised "
                      f"trace is NaN. Re-running them cannot help: start the "
                      f"band above the cutoff, use a wavenumber-integration "
                      f"model (Scooter), or run RunMode.TIME_SERIES, which "
                      f"zeroes the sub-cutoff bins inside its synthesis.")
        elif all_nan[i]:
            detail = ("is entirely NaN (no valid model output at this "
                      "cell); the synthesised trace is NaN, not silence.")
        else:
            detail = (f"has {int(np.count_nonzero(nan_bins[i]))} of {n_f} "
                      f"bins the model did not solve; the synthesised trace "
                      f"is NaN rather than carrying a notch at those "
                      f"frequencies. Re-run them, or narrow the band to the "
                      f"bins that solved.")
        warnings.warn(f"{who}: {where} {detail}",
                      NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)


# Level (dB re the spectrum's peak) above which a hard band edge cuts
# through the source spectrum and an untapered synthesis rings. The model
# TIME_SERIES route cuts its band at the waveform's -40 dB support, so it
# never trips this; a user band narrower than the pulse does.
_BAND_EDGE_RING_DB = -40.0


def warn_band_edge_cuts_spectrum(source_spectrum, window: str,
                                  who: str) -> None:
    """Warn when an untapered synthesis's band edge cuts the source spectrum.

    With ``window=None`` the band edge is a rectangular filter; it is
    harmless where ``S(f)`` has already fallen away, and rings (Gibbs
    overshoot, energy outside the pulse) where it has not."""
    if window is not None or source_spectrum is None:
        return
    mag = np.abs(np.asarray(source_spectrum))
    if mag.size < 2 or not np.any(mag > 0):
        return
    edge = max(mag[0], mag[-1])
    if edge == 0.0:
        # A spectrum already zero at both edges is not cut: the rectangular
        # band edge multiplies zero there, so there is no step to ring.
        return
    edge_dB = 20.0 * np.log10(edge / mag.max())
    if edge_dB > _BAND_EDGE_RING_DB:
        warnings.warn(
            f"{who}: the band's edge cuts the source spectrum at "
            f"{edge_dB:.1f} dB re its peak (above {_BAND_EDGE_RING_DB:.0f} "
            f"dB), so the untapered synthesis rings at the band edges. "
            f"Widen the frequency band to the waveform's support, or pass "
            f"window='hann' to trade the ringing for a level bias.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)


# A Fourier-synthesised record is periodic with period 1/Δf, so an arrival
# train still ringing at the record's end continues onto its start: the
# trace is truncated and the tail wraps. Level (dB re the trace's own
# envelope maximum) of the envelope over the record's first / last
# ``_RECORD_EDGE_FRACTION`` above which that is reported. Measured on a
# 100 m Pekeris at 4 km, 25-75 Hz Gaussian pulse: a 1 s record (Δf = 1 Hz)
# ending before the 2.89 s main arrival reads 0 dB at its head and -1.8 dB
# at its tail (Kraken, Scooter) and -13.6 dB at its tail (Bellhop), while
# the 2 s and 4 s records of the same run read below -54 dB at both edges.
# Example 24 (Bellhop, 5 km Pekeris, 1/Δf = 0.73 s) reads -17.7 dB at its
# tail, where a 2.9 s record shows steep bottom-bounce arrivals continuing
# to 3.76 s, past the 3.70 s record end.
_RECORD_EDGE_LEVEL_DB = -20.0


_RECORD_EDGE_FRACTION = 0.05

#: The fast/slow path-speed spread the window-wrap warning assumes
#: (see :func:`record_start`): 5 % of the surface speed.
_ASSUMED_PATH_SPEED_SPREAD = 0.05

#: The least share of a synthesis record placed before the estimated first
#: arrival (see :func:`record_lead`): clear of the head region the edge
#: check reads (``_RECORD_EDGE_FRACTION``).
RECORD_LEAD_FRACTION = 0.1

#: The band-limited onset a record's lead must hold, in units of
#: ``1/bandwidth``: the sidelobes of an untapered band's impulse,
#: ``~1/(π·B·t)``, fall under the edge check's -20 dB
#: (``_RECORD_EDGE_LEVEL_DB``) about ``3.2/B`` before the peak.
RECORD_ONSET_BANDWIDTHS = 4.0


def record_lead(record_s: float, bandwidth: float) -> float:
    """The lead (s) of a synthesis record ``record_s`` long over a band
    ``bandwidth`` Hz wide: the larger of :data:`RECORD_LEAD_FRACTION` of
    the record and :data:`RECORD_ONSET_BANDWIDTHS` ``/ bandwidth``, never
    more than half the record."""
    onset = (RECORD_ONSET_BANDWIDTHS / bandwidth if bandwidth > 0.0
             else 0.5 * record_s)
    return min(0.5 * record_s, max(RECORD_LEAD_FRACTION * record_s, onset))


def record_edge_levels(analytic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row envelope maxima over the record's head and tail, in dB re
    each row's own envelope maximum (``-inf`` for a row with no energy).

    ``analytic`` is the complex synthesis before ``2·Re``: it holds only
    positive frequencies, so its modulus is the trace's envelope."""
    env = np.abs(analytic)
    n = max(1, int(round(_RECORD_EDGE_FRACTION * env.shape[-1])))
    with np.errstate(invalid='ignore', divide='ignore'):
        peak = np.max(env, axis=-1)
        ok = np.isfinite(peak) & (peak > 0.0)
        head = np.full(env.shape[0], -np.inf)
        tail = np.full(env.shape[0], -np.inf)
        head[ok] = 20.0 * np.log10(np.max(env[ok, :n], axis=-1) / peak[ok])
        tail[ok] = 20.0 * np.log10(np.max(env[ok, -n:], axis=-1) / peak[ok])
    return head, tail


def record_edges_are_judgeable(record_s: float, pulse_s: Optional[float],
                                 lead_s: float) -> bool:
    """Whether an auto-placed record's edges can be quiet at all.

    :func:`record_start` puts the estimated first arrival ``lead_s`` in
    (:func:`record_lead`), so a lone arrival there occupies
    ``[lead_s, lead_s + pulse_s]`` and reaches the tail region
    ``[T(1 - f), T]`` (``f`` = ``_RECORD_EDGE_FRACTION``) unless
    ``pulse_s < T(1 - f) - lead_s``. Below that
    the edge energy measures the grid against the pulse — which the
    DFT-period check and the model TIME_SERIES route's derived-grid notice
    name — not an arrival train running past the record. ``pulse_s=None``
    (a raw source spectrum of unknown duration) is not judged."""
    if pulse_s is None:
        return False
    return record_s * (1.0 - _RECORD_EDGE_FRACTION) - lead_s > pulse_s


def warn_record_edge_energy(head_dB: np.ndarray, tail_dB: np.ndarray, *,
                             time: np.ndarray, df: float, who: str) -> None:
    """Warn when an auto-placed synthesised record ends (or starts) inside
    the arrivals.

    Callers invoke this only for a record whose start was auto-estimated
    (:func:`record_start`), which puts :func:`record_lead` before the
    estimated first arrival, clear of the head region read here: both
    edges of such a record should be quiet.
    The tail is the primary signal — energy not yet decayed at the record's
    end continues, by the 1/Δf periodicity, onto its start — and the head
    catches the same train already wrapped. A ``t_start`` the caller pinned
    is a placement, and may open on an arrival (whose band-limited precursor
    then sits at the record end) by choice, so it is not judged."""
    worst_tail = float(np.max(tail_dB)) if tail_dB.size else -np.inf
    worst_head = float(np.max(head_dB)) if head_dB.size else -np.inf
    worst = max(worst_tail, worst_head)
    if not worst > _RECORD_EDGE_LEVEL_DB:
        return
    edge = 'end' if worst_tail >= worst_head else 'start'
    warnings.warn(
        f"{who}: the synthesised record [{time[0]:.4g}, {time[-1]:.4g}] s "
        f"holds energy at its {edge} ({worst:.1f} dB re the trace maximum, "
        f"above {_RECORD_EDGE_LEVEL_DB:.0f} dB), so it cuts through the "
        f"arrival train: the record is periodic with 1/Δf = {1.0 / df:.4g} s "
        f"and the energy past its end wraps onto its start. Refine Δf (more "
        f"frequencies, or a narrower spacing) so 1/Δf spans the arrivals, "
        f"or pass t_start= to place the record.",
        NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)


def synthesize_traces(
    spectra: np.ndarray,
    *,
    freqs: np.ndarray,
    win: np.ndarray,
    source_spectrum: Optional[np.ndarray],
    bin_indices: np.ndarray,
    bin_offset_hz: float,
    nfft: int,
    df: float,
    t_start: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Fourier-synthesize a batch of cell spectra ``(M, n_f)`` into time
    traces ``(M, nfft)`` sharing one window anchored at ``t_start``; returns
    ``(traces, time, (head_dB, tail_dB))``, the last pair from
    :func:`record_edge_levels`. One ``np.fft.ifft`` over the batch computes every
    cell's transform in a single call. See :func:`synthesize_trace` for the
    synthesis contract and :func:`synthesis_grid` for the grid inputs."""
    dt = 1.0 / (nfft * df)
    spectra = spectra * win
    if source_spectrum is not None:
        spectra = spectra * np.asarray(source_spectrum)
    # The one-sided sum is doubled below (2·Re) to stand for the negative
    # frequencies, but a 0 Hz bin has no negative twin: it enters the
    # two-sided integral once, so it is halved here to be counted once.
    if freqs.size and freqs[0] == 0.0:
        spectra = spectra.astype(complex, copy=True)
        spectra[:, 0] *= 0.5

    # Advance the record to t_start. The synthesis below evaluates
    # sum H(f) e^{+2*pi*i*f*t}, so pre-rotating by e^{+2*pi*i*f*t_start} puts
    # ifft sample n at t = t_start + n*dt instead of at n*dt.
    spectra = spectra * np.exp(1j * 2.0 * np.pi * freqs * t_start)

    # Only the positive-frequency half is physical here: 2·Re(ifft) folds
    # anything at or above Nyquist onto the wrong frequency.
    padded = np.zeros((spectra.shape[0], nfft), dtype=complex)
    valid = (bin_indices >= 0) & (bin_indices < nfft // 2)
    padded[:, bin_indices[valid]] = spectra[:, valid]

    # ifft carries 1/nfft; ×(nfft·df) turns the bin sum into ∫…df
    analytic = np.fft.ifft(padded, axis=-1) * (nfft * df)
    elapsed = np.arange(nfft) * dt
    if bin_offset_hz != 0.0:
        analytic = analytic * np.exp(-2j * np.pi * bin_offset_hz * elapsed)
    return (2.0 * np.real(analytic), t_start + elapsed,
            record_edge_levels(analytic))


def waveform_synthesis_setup(freqs, wf, sample_rate, *, window, nfft,
                              n_samples_floor, who):
    """``(source_spectrum, plan)`` for synthesising a waveform through
    ``H(f)`` on ``freqs``: the checks, the exact source spectrum on the axis
    and the shared grid (:func:`synthesis_grid`). The one setup behind
    :meth:`Field.synthesize_time_series` and
    :func:`uacpy.acoustic_signal.synthesize_time_series`."""
    n_src = wf.size
    if n_src < 2:
        raise ConfigurationError(
            f"{who}: source_waveform must have at least 2 samples; got "
            f"{n_src}.")
    # NaN-closed (``not (sr > 0)``, not ``sr <= 0``): nan compares False
    # against both bounds, so the plain inequality passes it through to
    # int(nfft), which raises a raw ValueError instead of this typed one.
    if not np.isfinite(sample_rate) or not (sample_rate > 0):
        raise ConfigurationError(
            f"{who}: sample_rate must be positive and finite; got "
            f"{sample_rate}.")
    freqs = np.asarray(freqs, dtype=float)
    if freqs.size > 1:
        df_tf = float(np.diff(freqs).mean())
        t_dft = 1.0 / df_tf if df_tf > 0 else float('inf')
        t_dur = n_src / float(sample_rate)
        # Catches a caller-supplied grid coarser than the pulse itself: the
        # record 1/Δf then cannot even hold the source waveform. It cannot
        # fire on a grid derived from the waveform (Δf = fs/n makes 1/Δf the
        # pulse length exactly), and it says nothing about the CHANNEL: how
        # long the arrivals ring is not visible in H(f) sampled every Δf,
        # and a late arrival folds to a fixed place mid-record, leaving the
        # record's end silent — so that check lives where the arrivals are
        # (``Arrivals.synthesis_band``; Bellhop's BROADBAND run on its
        # default grid), by count and level. One-sample tolerance: float
        # roundoff in Δf can make t_dft and t_dur evaluate as < when they
        # should be ==.
        if t_dft < t_dur - 1.0 / float(sample_rate):
            warnings.warn(
                f"{who}: DFT period 1/Δf = {t_dft:.4f}s "
                f"is shorter than the source-waveform duration "
                f"{t_dur:.4f}s — the late-time response wraps back into "
                f"early bins. Refine the frequency grid to Δf ≤ "
                f"{1.0/t_dur:.4g} Hz, or shorten the waveform.",
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
        # Refuse a non-uniform axis HERE, not where the grid below reaches
        # it: the source spectrum is evaluated on this axis first, and its
        # chirp-z contour assumes exactly the grid the grid's guard
        # describes. (With fewer than 2 frequencies there is no spacing to
        # check; the grid raises on the count itself.)
        from uacpy.acoustic_signal.channel import _uniform_frequency_step
        _uniform_frequency_step(freqs, who=who)
    from uacpy.acoustic_signal.spectrum_at import waveform_spectrum_at
    source_spectrum = waveform_spectrum_at(wf, sample_rate, freqs)
    plan = synthesis_grid(freqs, window=window, nfft=nfft,
                           sample_rate=sample_rate,
                           n_samples_floor=n_samples_floor, who=who)
    warn_band_edge_cuts_spectrum(source_spectrum, window, who)
    return source_spectrum, plan


def synthesize_cells(spectra, source_spectrum, plan, t_start):
    """Synthesise every row of ``spectra`` ``(M, n_f)`` on one shared grid:
    returns ``(traces (M, nfft), time, head_dB, tail_dB)``.

    Every cell shares one synthesis grid, so one batched ifft per chunk of
    cells replaces a per-cell transform. Chunked over the cell axis so the
    ``(cells × nfft)`` complex scratch stays bounded (~4M elements ≈ 64 MB)
    however many cells there are.
    """
    plan_freqs, df, bin_indices, bin_offset_hz, nfft, win = plan
    n_cells = spectra.shape[0]
    out = np.empty((n_cells, nfft), dtype=np.float64)
    time_vec = None
    head_dB = np.full(n_cells, -np.inf)
    tail_dB = np.full(n_cells, -np.inf)
    chunk = max(1, _SCRATCH_BLOCK_ELEMS // nfft)
    for a in range(0, n_cells, chunk):
        idx = np.arange(a, min(a + chunk, n_cells))
        traces, time_vec, (head_dB[idx], tail_dB[idx]) = synthesize_traces(
            spectra[idx], freqs=plan_freqs, win=win,
            source_spectrum=source_spectrum, bin_indices=bin_indices,
            bin_offset_hz=bin_offset_hz, nfft=nfft, df=df, t_start=t_start)
        out[idx] = traces
    return out, time_vec, head_dB, tail_dB


def _band_width(freqs) -> float:
    """The width (Hz) of the band a synthesis grid ``freqs`` spans."""
    freqs = np.asarray(freqs, dtype=float)
    return float(freqs.max() - freqs.min()) if freqs.size > 1 else 0.0


def record_start(range: float, record_s: float, *, bandwidth: float,
                 c_max: float, c0: float, c_slow: float, who: str) -> float:
    """Start of a synthesis record ``record_s`` seconds long over a band
    ``bandwidth`` Hz wide for a cell at ``range``: :func:`record_lead`
    before the estimated first arrival, the rest of it after.

    The estimate anchors on the faster of the stamped speeds ``c_max``
    (the fastest speed of the paths the producer models, or of the
    waveguide its run resolved) and ``c0`` (a surface water speed), 0
    meaning not stamped; it warns when there is neither, or only ``c0``
    while the record is short against the travel time. It warns too when
    the record ends before ``range / c_slow``, the arrival at the slowest
    stated water speed (0: none stated): the later paths then fold onto
    the record's start. ``t_start=`` on the caller replaces this
    placement.
    """
    # The earliest arrival travels at the FASTEST speed in the waveguide,
    # so r/c_fast bounds it from below. Candidates are physical speeds:
    # 'c_max', the fastest speed of the paths the producer models, which
    # a producer states as ``Field.speeds.water_max`` when it differs from
    # the waveguide's (Bellhop traces the water alone), else the fastest
    # speed of the waveguide the run resolved
    # (``Field.speeds.waveguide_max``); and 'c0' (Bellhop, the sea-surface
    # water speed ``Field.speeds.surface``).
    # Anchoring on a speed above c_fast opens the window too early: the
    # error adds to the lead, the late multipath
    # tail falls past the end of the record and wraps to the beginning
    # — so no algorithmic speed (e.g. a PE expansion point) may enter
    # this max, and c_min never binds it.
    # The 1500 m/s default is a fallback for when nothing physical was
    # stamped, never a candidate beside a stamped speed: a stamped speed
    # BELOW it (cold or fresh water) must win, or the window opens early by
    # r·(1/c − 1/1500) and the arrival wraps a whole record with the time
    # axis mislabelled and nothing to show for it.
    stamped = [speed for speed in (c_max, c0) if speed > 0.0]
    anchor_speed = max(stamped) if stamped else DEFAULT_SOUND_SPEED
    travel = range / anchor_speed
    # The anchor is a lower bound on the travel time, so the lead only has
    # to hold the band-limited onset of the first path; the rest of the
    # record holds the multipath behind it.
    lead = record_lead(record_s, bandwidth)
    t_start = max(0.0, travel - lead)
    if c_slow > 0.0 and t_start + record_s < range / c_slow:
        warnings.warn(
            f"{who}: the {record_s:.3g}s record [{t_start:.4g}, "
            f"{t_start + record_s:.4g}] s ends before r/c = "
            f"{range / c_slow:.4g} s, the arrival at the slowest water "
            f"speed ({c_slow:g} m/s) at {range:.0f} m, so every later path "
            f"folds onto the record's start. Lengthen the record (a finer "
            f"frequency grid; output_duration= on a TIME_SERIES run) or "
            f"place it with t_start=.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    if not c_max and not c0 and t_start > 0.0:
        # With no stamped speed at all the anchor is the 1500 m/s
        # default, which bounds nothing: a fast seabed (head waves at
        # 2-6 km/s) puts the earliest arrival well before r/1500, past
        # any lead the window can offer.
        warnings.warn(
            f"{who}: the response stamped no sound speed (a Field's "
            f"speeds state no water_max, waveguide_max or surface; a "
            f"BeamformedField carries none), so the window is anchored on "
            f"the {DEFAULT_SOUND_SPEED:g} m/s default at t_start="
            f"{t_start:.3g}s. Any path faster than that (e.g. a head "
            f"wave in a fast seabed) arrives before the window and "
            f"wraps to the end of the record. Pass t_start= to pin the "
            f"window start.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    # With only a c0 the anchor carries the fast/slow path spread as
    # error. A 5 % spread is representative of an ocean waveguide; when
    # it exceeds the lead the first arrival can fall before the window
    # and wrap to the end of the record.
    #
    # Where the 5 % lives, exactly: c0 is a SURFACE sample, so it is the
    # fastest water speed only on a downward-refracting profile. The
    # window wraps when r(1/c0 - 1/c_fast) > T/2, and this test fires when
    # 0.05·r/c0 > T/2, so it covers the wrap iff c0 >= 0.95·c_fast — i.e.
    # while the surface sample is within 5 % of the profile maximum. An
    # upward-refracting column with a wider spread (a cold surface over a
    # deep sound channel) leaves a band of window lengths uncovered.
    elif not c_max and t_start > 0.0 and \
            _ASSUMED_PATH_SPEED_SPREAD * travel > lead:
        warnings.warn(
            f"{who}: the {record_s:.3g}s synthesis window is "
            f"short against the {travel:.3g}s travel time at "
            f"{range:.0f} m, and the model reported no maximum "
            f"sound speed, so the window start is an estimate — the "
            f"earliest arrival may fall before it and wrap to the end of "
            f"the record. Pass t_start= to pin it, or refine the "
            f"frequency grid (the window is 1/Δf).",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    return t_start


def synthesize_trace(spectrum, source_spectrum, plan, *, range: float,
                     t_start: Optional[float], c_max: float, c0: float,
                     c_slow: float, cell_label: str, who: str,
                     sub_cutoff_bins: int = 0,
                     pulse_s: Optional[float] = None):
    """One cell's spectrum ``H(f)`` on the grid ``plan``
    (:func:`synthesis_grid`) synthesised into a time trace: returns
    ``(time, trace)``.

    Evaluates the Fourier synthesis ``p(t) = 2·Re Σ H(f_k)·S(f_k)·
    e^{2πi f_k t}·df`` — a Riemann sum of the continuous inverse transform,
    so the amplitude is independent of ``nfft`` and of the bin grid.
    ``source_spectrum`` is the *continuous* source spectrum sampled at the
    grid frequencies (a raw DFT times the source sampling interval);
    ``None`` synthesises the band-limited impulse response.

    ``t_start=None`` places the record by :func:`record_start` from the
    cell's ``range`` and the stamped speeds ``c_max`` / ``c0`` / ``c_slow``,
    and then
    judges the record's edges (:func:`warn_record_edge_energy`) when
    ``pulse_s`` — the transmitted signal's duration, 0 for the impulse
    response, ``None`` when unknown — lets them be quiet. ``cell_label``
    names the cell in the unsolved-bin warning (:func:`warn_unsolved_bins`).
    """
    freqs, df, bin_indices, bin_offset_hz, nfft, win = plan
    spectra = np.asarray(spectrum)[None, :]
    warn_unsolved_bins(spectra, who=who, cell_label=lambda i: cell_label,
                       sub_cutoff_bins=sub_cutoff_bins)

    dt = 1.0 / (nfft * df)

    t_start_estimated = t_start is None
    if t_start_estimated:
        t_start = record_start(range, nfft * dt, bandwidth=_band_width(freqs),
                               c_max=c_max, c0=c0, c_slow=c_slow, who=who)

    traces, time, (head_dB, tail_dB) = synthesize_traces(
        spectra, freqs=freqs, win=win, source_spectrum=source_spectrum,
        bin_indices=bin_indices, bin_offset_hz=bin_offset_hz, nfft=nfft,
        df=df, t_start=t_start)
    if t_start_estimated and record_edges_are_judgeable(
            1.0 / df, pulse_s, record_lead(1.0 / df, _band_width(freqs))):
        warn_record_edge_energy(head_dB, tail_dB, time=time, df=df, who=who)
    return time, traces[0]


def source_waveform_problem(source_waveform) -> Optional[str]:
    """Why ``source_waveform`` is not a usable transmitted pulse, or
    ``None`` when it is.

    The one rule every entry point that takes a source waveform applies —
    the Field synthesis methods, :func:`~uacpy.acoustic_signal.synthesize_time_series`
    and a model's TIME_SERIES run: a real, finite, 1-D array of pressure
    samples. The synthesis is ``2·Re`` of a one-sided sum over the waveform's
    spectrum, so a complex waveform has no meaning there, and a
    ``(time, signal)`` pair or a stacked 2-D array would be read as one
    flattened signal."""
    if isinstance(source_waveform, tuple):
        return ("source_waveform must be the 1-D signal, not a (time, "
                "signal) pair — pass lfm_chirp(...)[1].")
    wf = np.asarray(source_waveform)
    if np.iscomplexobj(wf):
        return ("source_waveform must be a real pressure pulse; got a "
                "complex array. Pass np.real(w) if its real part is the "
                "pulse you mean.")
    if wf.ndim != 1:
        return (f"source_waveform must be a 1-D signal; got an array of "
                f"shape {wf.shape}. The generators return a (time, signal) "
                f"pair, so pass lfm_chirp(...)[1].")
    if wf.dtype.kind not in 'biuf':
        return (f"source_waveform must hold numbers; got an array of dtype "
                f"{wf.dtype}.")
    if not np.all(np.isfinite(wf)):
        return "source_waveform contains non-finite values (NaN/inf)."
    return None


def require_source_waveform(source_waveform, who: str) -> np.ndarray:
    """``source_waveform`` as a float64 1-D array, refused with
    :func:`source_waveform_problem`'s reason when it is not a usable
    pulse."""
    problem = source_waveform_problem(source_waveform)
    if problem is not None:
        raise ConfigurationError(f"{who}: {problem}")
    return np.asarray(source_waveform, dtype=float)


def waveform_spectrum_on(frequencies, source_waveform, sample_rate,
                         source_spectrum, who: str):
    """``source_spectrum``, or with ``source_waveform`` given in its place,
    the waveform's exact spectrum on ``frequencies``
    (:func:`~uacpy.acoustic_signal.waveform_spectrum_at`). Refuses both at
    once, a waveform without ``sample_rate``, and a waveform that breaks the
    one rule (:func:`source_waveform_problem`)."""
    if source_waveform is None:
        return source_spectrum
    if source_spectrum is not None:
        raise ConfigurationError(
            f"{who}: pass either source_spectrum= (already on this "
            f"field's frequency axis) or source_waveform= (sampled in "
            f"time), not both.")
    if sample_rate is None:
        raise ConfigurationError(
            f"{who}: source_waveform= needs sample_rate= to have a "
            f"spectrum at all.")
    wf = require_source_waveform(source_waveform, who)
    from uacpy.acoustic_signal.spectrum_at import waveform_spectrum_at
    return waveform_spectrum_at(wf, sample_rate,
                                np.asarray(frequencies, dtype=float))


def check_source_spectrum(source_spectrum, n_frequencies: int,
                          who: str) -> None:
    """Refuse a ``source_spectrum`` that is not one value per frequency of
    an ``n_frequencies`` axis."""
    if (np.ndim(source_spectrum) != 1
            or np.size(source_spectrum) != n_frequencies):
        raise ConfigurationError(
            f"{who}: source_spectrum has shape "
            f"{np.shape(source_spectrum)} but the frequency axis has "
            f"{n_frequencies} samples; it is the source spectrum ON this "
            f"field's grid, one value per frequency.")
