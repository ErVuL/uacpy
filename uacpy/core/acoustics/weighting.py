"""Frequency weighting of pressure: how much each frequency counts toward a
level.

A weighting is a gain ``W(f)`` in dB applied to the pressure before a level
is taken. The level of the weighted pressure is then the part of the sound a
listener, an instrument or an exposure criterion counts. Southall et al.
(2019) define their exposure metric this way, as the time integral of "the
square of the instantaneous frequency-weighted sound pressure".

Everywhere uacpy takes ``weighting=`` it accepts one of three forms:

* a hearing-group name of :data:`HEARING_GROUPS` (``'LF'``, ``'VHF'``, …):
  the Southall et al. (2019) auditory weighting, :func:`auditory_weighting`;
* a callable ``W(frequency_Hz) -> dB``: any weighting of your own, evaluated
  at positive frequencies only;
* a ``(frequencies_Hz, W_dB)`` table: interpolated linearly in
  log-frequency; frequencies outside the table carry no weight, and a table
  that leaves more than half of a record's energy outside its span warns
  (:func:`~uacpy.core._validate.warn_table_leaves_energy_out`: a table typed
  in kHz).

The in-air groups ``'PCA'`` and ``'OCA'`` weight a level at the underwater
reference (1 µPa) with a warning (:func:`~uacpy.core._validate.warn_in_air_group_in_water`).

DC carries no weight in any form.

A weighting is a magnitude, with no phase. :func:`weighted_pressure`
applies it as a zero-phase filter, multiplying the record's spectrum by
``10**(W/20)`` itself. It is not approximated by a truncated FIR, whose
leakage floor would sit above the 70-110 dB of attenuation the hearing
groups reach at 10-50 Hz. A record too long for one FFT is filtered in
overlapping blocks (overlap-save).
:func:`frequency_weighting` evaluates any weighting on any grid.

The Southall auditory weighting (Eq. 2) is::

    W(f) = C + 10·log10[ (f/f1)^(2a) / ( (1+(f/f1)^2)^a · (1+(f/f2)^2)^b ) ]

with ``f, f1, f2`` in kHz; the low- and high-frequency roll-off slopes are
``+20a`` and ``-20b`` dB/decade, and ``C`` sets the function peak to 0 dB.

References
----------
Southall, B. L., Finneran, J. J., Reichmuth, C., et al. (2019). "Marine Mammal
    Noise Exposure Criteria: Updated Scientific Recommendations for Residual
    Hearing Effects." *Aquatic Mammals* 45(2), 125-232 — Table 5. The
    parameters agree with NMFS (2018) Technical Guidance (NMFS-OPR-59) where
    the two overlap, but Southall renamed the cetacean groups, so the labels
    move one step: NMFS LF / MF / HF / PW / OW are Southall LF / HF / VHF /
    PCW / OCW here. NMFS "HF" (harbour porpoise) is this module's ``"VHF"``;
    this module's ``"HF"`` is NMFS "MF" (most dolphins): Southall et al.
    (2019) draw "the distinction between HF and VHF cetacean groups (as
    opposed to mid- and high-frequency)", and their HF group "represents
    most of the same species identified as MF cetaceans by ... NMFS".
"""

import numpy as np

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._validate import (
    refuse_non_finite_samples, require_positive_finite_scalar,
    warn_table_leaves_energy_out)

__all__ = ['EXPOSURE_THRESHOLDS', 'HEARING_GROUPS', 'WEIGHTING_PARAMS',
           'auditory_weighting', 'frequency_weighting', 'weighted_pressure']

# Southall et al. (2019) Table 5: a, b, f1 [kHz], f2 [kHz], C [dB] (weighting)
# and K [dB]. Only a/b/f1/f2/C enter :func:`auditory_weighting`.
#
# K positions the **non-impulsive TTS** exposure function only — Table 5 is
# titled "auditory weighting function and TTS exposure function parameters".
# Since C sets the weighting's peak to 0 dB and K sets the exposure
# function's minimum to the weighted TTS threshold Tw, Southall shows
# Tw = C + K (the discussion of their Eqs. 2-3); C + K rounds to Table 6's
# TTS column in all eight groups (HF: 1.20 + 177 = 178.2, Table 6: 178).
# The thresholds themselves are :data:`EXPOSURE_THRESHOLDS`.
WEIGHTING_PARAMS = {
    "LF":  {"a": 1.0, "b": 2, "f1": 0.20, "f2": 19.0,  "C": 0.13, "K": 179},
    "HF":  {"a": 1.6, "b": 2, "f1": 8.8,  "f2": 110.0, "C": 1.20, "K": 177},
    "VHF": {"a": 1.8, "b": 2, "f1": 12.0, "f2": 140.0, "C": 1.36, "K": 152},
    "SI":  {"a": 1.8, "b": 2, "f1": 4.3,  "f2": 25.0,  "C": 2.62, "K": 183},
    "PCW": {"a": 1.0, "b": 2, "f1": 1.9,  "f2": 30.0,  "C": 0.75, "K": 180},
    "OCW": {"a": 2.0, "b": 2, "f1": 0.94, "f2": 25.0,  "C": 0.64, "K": 198},
    "PCA": {"a": 2.0, "b": 2, "f1": 0.75, "f2": 8.3,   "C": 1.50, "K": 132},
    "OCA": {"a": 1.4, "b": 2, "f1": 2.0,  "f2": 20.0,  "C": 1.39, "K": 156},
}

#: Southall et al. (2019) hearing groups. The cetacean labels are NOT NMFS
#: (2018)'s: NMFS MF is ``"HF"`` here and NMFS HF is ``"VHF"`` (see the module
#: docstring); NMFS PW / OW are ``"PCW"`` / ``"OCW"``.
HEARING_GROUPS = {
    "LF": "Low-frequency cetaceans",
    "HF": "High-frequency cetaceans",
    "VHF": "Very-high-frequency cetaceans",
    "SI": "Sirenians",
    "PCW": "Phocid carnivores in water",
    "OCW": "Other marine carnivores in water",
    "PCA": "Phocid carnivores in air",
    "OCA": "Other marine carnivores in air",
}

#: Southall et al. (2019) TTS- and PTS-onset exposure criteria per hearing
#: group: Table 6 (``'non_impulsive'`` noise, weighted SEL) and Table 7
#: (``'impulsive'`` noise, weighted SEL and unweighted ``'peak'`` SPL), as
#: ``EXPOSURE_THRESHOLDS[group][kind][onset][metric]`` in dB. SEL is re
#: 1 µPa²·s and peak re 1 µPa under water; the in-air groups ``'PCA'`` and
#: ``'OCA'`` are re (20 µPa)²·s and re 20 µPa. A SEL is compared after the
#: group's own weighting (``weighting=group``); a peak with none. Their
#: rules: non-impulsive PTS = TTS + 20 dB; impulsive PTS = TTS + 15 dB
#: (SEL) and + 6 dB (peak).
EXPOSURE_THRESHOLDS = {
    group: {
        'non_impulsive': {'TTS': {'SEL': tts}, 'PTS': {'SEL': pts}},
        'impulsive': {'TTS': {'SEL': i_tts, 'peak': pk_tts},
                      'PTS': {'SEL': i_pts, 'peak': pk_pts}},
    }
    for group, (tts, pts, i_tts, pk_tts, i_pts, pk_pts) in {
        # Table 6 TTS, PTS; Table 7 TTS SEL, TTS peak, PTS SEL, PTS peak
        "LF":  (179, 199, 168, 213, 183, 219),
        "HF":  (178, 198, 170, 224, 185, 230),
        "VHF": (153, 173, 140, 196, 155, 202),
        "SI":  (186, 206, 175, 220, 190, 226),
        "PCW": (181, 201, 170, 212, 185, 218),
        "OCW": (199, 219, 188, 226, 203, 232),
        "PCA": (134, 154, 123, 138, 138, 144),
        "OCA": (157, 177, 146, 161, 161, 167),
    }.items()
}


#: NMFS (2018) group names with no Southall (2019) group of the same name,
#: and the Southall group each one is.
_NMFS_ONLY_GROUPS = {"MF": "HF", "PW": "PCW", "OW": "OCW"}

#: Seconds of record a block reads past each of its edges when a record is
#: too long to weight in one FFT. Past 1 s every Southall weighting's impulse
#: response holds less than -66 dB (LF) of amplitude, at most -85 dB for the
#: other in-water groups. That residue is the response below about 1 Hz, so a
#: join is exact above a few hertz.
_MARGIN_SECONDS = 1.0

#: Complex FFT bins held at once across the channels of one block: 2**23
#: bins of complex128, 128 MB.
_FFT_BINS = 2 ** 23

#: Shortest FFT span, in seconds: ``W`` is sampled at 1 Hz or finer. A
#: weighting with sharp edges (a table of your own) has a long impulse
#: response, and a short record's own grid would blur them: a 1 ms click
#: in a 2 ms record read 0.12 dB high through a 2-4 kHz brick wall.
_MIN_FFT_SECONDS = 1.0


def _hearing_group(group, who):
    """``group`` as an upper-case key of :data:`WEIGHTING_PARAMS`, or a
    refusal naming ``who``."""
    g = str(group).upper()
    if g in _NMFS_ONLY_GROUPS:
        raise ConfigurationError(
            f"{who}: {group!r} is an NMFS (2018) group name; "
            f"this module uses Southall et al. (2019)'s, where NMFS "
            f"{g} is {_NMFS_ONLY_GROUPS[g]!r}. Note that NMFS 'HF' is "
            f"Southall 'VHF', and Southall 'HF' is NMFS 'MF'.")
    if g not in WEIGHTING_PARAMS:
        raise ConfigurationError(
            f"{who}: unknown group {group!r}; choose from "
            f"{sorted(WEIGHTING_PARAMS)}.")
    return g


def auditory_weighting(frequency, group):
    """Auditory weighting ``W(f)`` [dB] at ``frequency`` [Hz] for a hearing group.

    ``group`` is one of :data:`HEARING_GROUPS` (e.g. ``"LF"``, ``"VHF"``,
    ``"PCW"``), in Southall et al. (2019)'s naming: a harbour porpoise, NMFS
    (2018) "HF", is ``"VHF"`` here. The peak of the function is 0 dB to
    within 0.006 dB, and the rest of the curve is below it: Table 5 rounds
    ``C`` to two decimals, which leaves the peaks between -0.003 dB (PCW)
    and +0.006 dB (OCA).

    Parameters
    ----------
    frequency : float or array_like
        Frequency (Hz).
    group : str
        A hearing group of :data:`HEARING_GROUPS`.
    """
    p = WEIGHTING_PARAMS[_hearing_group(group, "auditory_weighting")]
    f = np.asarray(frequency, dtype=float) / 1000.0        # Hz -> kHz
    # Eq. (2) is even in f, so a negative frequency squares away and comes back
    # as a plausible finite weighting; f = 0 gives -inf behind a bare numpy
    # divide-by-zero warning. Both are input errors, rejected here the way
    # WenzNoise rejects a non-positive frequency grid.
    # ``~(f > 0)`` rather than ``f <= 0``: NaN passes every comparison and would
    # return a NaN weighting silently.
    if f.size == 0 or np.any(~(f > 0.0)):
        raise ConfigurationError(
            "auditory_weighting: frequency must be > 0 Hz and finite (the "
            "weighting is a log10 of (f/f1)^2a); drop the DC bin, e.g. "
            "f[f > 0].")
    r1 = (f / p["f1"]) ** 2
    r2 = (f / p["f2"]) ** 2
    return p["C"] + 10.0 * np.log10(
        r1 ** p["a"] / ((1 + r1) ** p["a"] * (1 + r2) ** p["b"]))


def frequency_weighting(frequency, weighting):
    """``W(f)`` in dB of any weighting uacpy accepts, at ``frequency`` (Hz).

    One door for the three forms ``weighting=`` takes everywhere: a hearing
    group of :data:`HEARING_GROUPS` (:func:`auditory_weighting`), a function
    ``W(frequency_Hz) -> dB``, or a ``(frequencies_Hz, W_dB)`` table,
    interpolated linearly in log-frequency. The result is ``-inf`` wherever
    the weighting carries no weight: at DC and below, and outside a table.

    Parameters
    ----------
    frequency : float or array_like
        Frequency (Hz).
    weighting : str, callable or (array_like, array_like)
        The weighting.

    Returns
    -------
    ndarray
        ``W(f)`` in dB, shaped like ``frequency``.

    Examples
    --------
    >>> import numpy as np
    >>> table = ([100.0, 1000.0], [-20.0, 0.0])
    >>> frequency_weighting([50.0, 100.0, 316.2, 1000.0, 2000.0], table)
    array([-inf, -20., -10.,   0., -inf])
    """
    f = np.asarray(frequency, dtype=float)
    w = np.full(f.shape, -np.inf)
    positive = f > 0.0
    if np.any(positive):
        w[positive] = _weighting_dB(weighting, f[positive],
                                    "frequency_weighting")
    return w


def _weighting_dB(weighting, frequency, who):
    """``W(f)`` in dB of any accepted ``weighting`` at positive
    ``frequency`` (Hz); ``-inf`` where it carries no weight."""
    if isinstance(weighting, str):
        _hearing_group(weighting, who)
        return auditory_weighting(frequency, weighting)
    if callable(weighting):
        try:
            w = np.broadcast_to(np.asarray(weighting(frequency), dtype=float),
                                frequency.shape)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"{who}: a weighting function must take an array of "
                f"frequencies (Hz) and return one W(f) in dB per frequency; "
                f"calling it on {frequency.size} frequencies failed: {exc}."
            ) from exc
        return w
    if isinstance(weighting, (tuple, list)) and len(weighting) == 2:
        table_f, table_w = (np.asarray(v, dtype=float) for v in weighting)
        if (table_f.ndim != 1 or table_f.shape != table_w.shape
                or table_f.size < 2):
            raise ConfigurationError(
                f"{who}: a weighting table is (frequencies_Hz, W_dB), two "
                f"1-D arrays of equal length >= 2; got shapes "
                f"{table_f.shape} and {table_w.shape}.")
        if not (np.all(table_f > 0) and np.all(np.diff(table_f) > 0)):
            raise ConfigurationError(
                f"{who}: the weighting table's frequencies must be positive "
                f"and strictly increasing (it is interpolated in "
                f"log-frequency).")
        w = np.full(frequency.shape, -np.inf)
        inside = (frequency >= table_f[0]) & (frequency <= table_f[-1])
        w[inside] = np.interp(np.log10(frequency[inside]), np.log10(table_f),
                              table_w)
        return w
    raise ConfigurationError(
        f"{who}: weighting must be a hearing group name "
        f"({sorted(WEIGHTING_PARAMS)}), a function W(f_Hz) -> dB, or a "
        f"(frequencies_Hz, W_dB) table; got {type(weighting).__name__}.")


def _weighting_gain(weighting, frequency, who):
    """Power gain ``10**(W/10)`` of ``weighting`` at ``frequency`` (Hz):
    0 at DC and wherever the weighting carries no weight."""
    f = np.asarray(frequency, dtype=float)
    gain = np.zeros(f.shape)
    positive = f > 0.0
    if not np.any(positive):
        return gain
    w = _weighting_dB(weighting, f[positive], who)
    # -inf is "no weight"; NaN and +inf are not a weighting at all.
    if np.any(np.isnan(w) | (w == np.inf)):
        raise ConfigurationError(
            f"{who}: the weighting returned NaN or +inf dB at "
            f"{int(np.count_nonzero(np.isnan(w) | (w == np.inf)))} "
            f"frequencies; a weighting is a finite gain in dB, or -inf "
            f"where a frequency does not count.")
    gain[positive] = 10.0 ** (w / 10.0)
    return gain


def _weighted_blocks(x, sample_rate, weighting, *, who):
    """``(channels, start, y)`` triples: ``x``, shaped ``(channels, time)``,
    weighted by ``weighting`` as a zero-phase filter, a block at a time.
    ``y`` runs from sample ``start``. The first block starts before sample 0,
    and the last ends past the record, so the weighting's spread past both
    ends is included.

    Each block is multiplied by the ideal response ``10**(W/20)`` on its own
    FFT grid. That grid is at least twice the block, so the circular wrap
    only reaches lags past the block's length. It also spans at least
    :data:`_MIN_FFT_SECONDS`, so ``W`` is sampled at 1 Hz or finer however
    short the record. A record that fits is one block per group of
    channels, filtered exactly. A longer one is cut into blocks that read
    :data:`_MARGIN_SECONDS` past each edge (overlap-save), with zeros
    beyond the record. ``x`` is read one slice at a time, so a ``np.memmap``
    is never loaded whole. A non-finite sample is refused in the block that
    holds it (:func:`~uacpy.core._validate.refuse_non_finite_samples`)."""
    # Deferred: importing scipy here would load it with every core.acoustics
    # import, which uacpy.io and the data layer keep lean.
    from scipy.fft import irfft, next_fast_len, rfft, rfftfreq
    n_channels, n = x.shape
    shortest = int(np.ceil(_MIN_FFT_SECONDS * sample_rate))
    nfft = next_fast_len(max(2 * n, shortest))
    # The record's energy per bin, read for a table only: what the table's
    # span leaves out (warn_table_leaves_energy_out).
    table = isinstance(weighting, (tuple, list))
    if nfft // 2 + 1 <= _FFT_BINS:
        freqs = rfftfreq(nfft, 1.0 / sample_rate)
        amplitude = np.sqrt(_weighting_gain(weighting, freqs, who))
        spectrum = np.zeros(freqs.size)
        group = max(1, _FFT_BINS // (nfft // 2 + 1))
        # The padding past the record holds the spread past its end, then
        # (wrapped) the spread before its start; half each.
        after = (nfft - n) // 2
        for c0 in range(0, n_channels, group):
            channels = slice(c0, min(c0 + group, n_channels))
            segment = _finite_block(x[channels, :], 0, n, who,
                                    first_channel=c0,
                                    n_channels=n_channels)
            X = rfft(segment, nfft, axis=-1)
            if table:
                spectrum += np.sum(np.abs(X) ** 2, axis=0)
            y = irfft(X * amplitude, nfft, axis=-1)
            yield (channels, n + after - nfft,
                   np.concatenate((y[:, n + after:], y[:, :n + after]),
                                  axis=-1))
        warn_table_leaves_energy_out(weighting, freqs, spectrum, who)
        return
    margin = int(round(_MARGIN_SECONDS * sample_rate))
    block = max(_FFT_BINS - 2 * margin, margin)
    nfft = next_fast_len(2 * (block + 2 * margin))
    freqs = rfftfreq(nfft, 1.0 / sample_rate)
    amplitude = np.sqrt(_weighting_gain(weighting, freqs, who))
    spectrum = np.zeros(freqs.size)
    for c in range(n_channels):
        channels = slice(c, c + 1)
        for start in range(0, n, block):
            stop = min(start + block, n)
            lo, hi = start - margin, stop + margin
            a, b = max(lo, 0), min(hi, n)
            segment = np.zeros((1, hi - lo))
            segment[:, a - lo: b - lo] = _finite_block(
                x[channels, a:b], a, b, who, first_channel=c,
                n_channels=n_channels)
            X = rfft(segment, nfft, axis=-1)
            if table:
                spectrum += np.abs(X[0]) ** 2
            y = irfft(X * amplitude, nfft, axis=-1)
            keep_lo = 0 if start == 0 else margin
            keep_hi = (stop - lo) + (margin if stop == n else 0)
            yield channels, lo + keep_lo, y[:, keep_lo:keep_hi]
    warn_table_leaves_energy_out(weighting, freqs, spectrum, who)


def _finite_block(values, a, b, who, *, first_channel=0, n_channels=1):
    """``values`` (samples ``a`` to ``b`` of channels ``first_channel`` on,
    of a record of ``n_channels``) as a float array, refused at its first
    NaN or inf (:func:`~uacpy.core._validate.refuse_non_finite_samples`)."""
    values = np.asarray(values, dtype=float)
    refuse_non_finite_samples(values, who, start=a,
                              first_channel=first_channel,
                              n_channels=n_channels)
    return values


def _real_pressure(x, complex_record, who):
    """The real pressure a weighting filters: ``x`` itself, or the real part
    of an analytic record. A baseband record is refused, since its
    frequencies are offsets from a carrier it does not carry."""
    if not np.iscomplexobj(x):
        return x
    if complex_record == 'analytic':
        return x.real
    raise ConfigurationError(
        f"{who}: a weighting is a function of the true acoustic frequency, "
        f"which a complex record states only as an analytic signal; got "
        f"complex_record={complex_record!r}.",
        remediation="Pass the real pressure, or an analytic record with "
                    "complex_record='analytic'. Remodulate a baseband "
                    "record onto its carrier first.")


def weighted_pressure(pressure, sample_rate, weighting, *, axis=-1,
                      complex_record=None):
    """The frequency-weighted pressure ``p_w(t)`` of a record.

    The record's spectrum is multiplied by ``10**(W/20)``, a zero-phase
    filter, so ``p_w`` stays time-aligned with ``pressure``. Its level is
    the weighted level: ``sound_exposure_level(weighted_pressure(p, fs,
    'LF'), fs)`` is the LF-weighted sound exposure level of Southall et al.
    (2019), which ``sound_exposure_level(p, fs, weighting='LF')`` computes
    without holding ``p_w``. Any weighting of your own, A-weighting
    included, works the same way.

    A record whose FFT fits in about 128 MB is filtered in one piece,
    exactly, on an FFT grid of 1 Hz or finer whatever the record's length.
    A longer one is filtered in blocks that read 1 s past each edge, so the
    joins are exact above a few hertz. A ``np.memmap`` is read one block at
    a time; the output is a full array. The weighting spreads a little past
    the first and last samples. That spread is not returned, but the
    weighted SEL counts its energy.

    Parameters
    ----------
    pressure : array_like
        Pressure record (Pa), time along ``axis``. Real, or analytic with
        ``complex_record='analytic'`` (the pressure is the real part).
    sample_rate : float
        Sample rate (Hz).
    weighting : str, callable or (array_like, array_like)
        A hearing group of :data:`HEARING_GROUPS`; a function
        ``W(frequency_Hz) -> dB``; or a ``(frequencies_Hz, W_dB)`` table,
        interpolated linearly in log-frequency, outside which a frequency
        carries no weight. DC carries no weight in any form.
    axis : int, default -1
        The time axis; every other axis is filtered alongside.
    complex_record : {'analytic'}, optional
        Required for a complex record; see
        :func:`~uacpy.core.acoustics.sound_exposure_level`. A baseband
        record is refused: the weighting needs the true frequency.

    Returns
    -------
    ndarray
        ``p_w``, real, shaped like ``pressure``.

    Examples
    --------
    A 2 kHz tone loses ``W(2 kHz)`` = 26.9 dB of level under the
    very-high-frequency-cetacean weighting:

    >>> import numpy as np
    >>> fs = 48000.0
    >>> t = np.arange(int(fs)) / fs
    >>> p = np.sin(2 * np.pi * 2000.0 * t) * np.hanning(t.size)
    >>> pw = weighted_pressure(p, fs, 'VHF')
    >>> round(float(10 * np.log10(np.sum(pw ** 2) / np.sum(p ** 2))), 1)
    -26.9
    """
    who = "weighted_pressure"
    fs = require_positive_finite_scalar(sample_rate, who, "sample_rate",
                                        " Hz")
    x = np.moveaxis(_real_pressure(np.asarray(pressure), complex_record, who),
                    axis, -1)
    if x.size == 0:
        raise ConfigurationError(f"{who}: the record is empty.")
    rows = x.reshape(-1, x.shape[-1])
    n = rows.shape[-1]
    out = np.empty(rows.shape)
    for channels, start, y in _weighted_blocks(rows, fs, weighting, who=who):
        a, b = max(start, 0), min(start + y.shape[-1], n)
        out[channels, a:b] = y[:, a - start: b - start]
    return np.moveaxis(out.reshape(x.shape), -1, axis)
