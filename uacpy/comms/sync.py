"""Frame synchronization: preamble detection and timing recovery.

A matched filter against a known preamble (e.g. a chirp or an m-sequence) gives
the frame start; normalizing by the running energy yields a detection metric
robust to level changes. This is the front end every coherent receiver runs
before equalization.

References
----------
Proakis & Salehi. *Digital Communications* (synchronization, matched filtering).
"""

from __future__ import annotations

import numpy as np

from uacpy.core.exceptions import ConfigurationError


def matched_filter_metric(rx, preamble):
    """Energy-normalized cross-correlation magnitude of ``rx`` with ``preamble``.

    Returns a real metric in ``[0, 1]`` (1 = perfect match) aligned so that
    ``metric[k]`` scores the window ``rx[k:k+len(preamble)]``.
    """
    r = np.asarray(rx, dtype=complex)
    p = np.asarray(preamble, dtype=complex)
    if p.size > r.size:
        raise ConfigurationError("matched_filter_metric: preamble longer than signal")
    corr = np.correlate(r, p, mode="valid")
    pe = np.sum(np.abs(p) ** 2)
    win = np.convolve(np.abs(r) ** 2, np.ones(p.size), mode="valid")
    return np.abs(corr) / np.sqrt(pe * win + 1e-12)


def detect_preamble(rx, preamble, threshold=0.5):
    """Index of the best preamble alignment if its metric exceeds ``threshold``.

    Returns ``(start_index, metric_array)``; ``start_index`` is ``None`` when no
    peak clears the threshold.
    """
    metric = matched_filter_metric(rx, preamble)
    k = int(np.argmax(metric))
    return (k if metric[k] >= threshold else None), metric


def detect_frames(rx, preamble, threshold=0.5, min_gap=None):
    """All frame-start indices whose metric clears ``threshold`` (peak-picked).

    ``min_gap`` (samples) suppresses detections closer than that to a stronger
    one; defaults to the preamble length. Returns ``(starts, metric_array)``.
    """
    metric = matched_filter_metric(rx, preamble)
    gap = int(min_gap) if min_gap is not None else np.asarray(preamble).size
    cand = np.flatnonzero(metric >= threshold)
    starts = []
    for k in cand[np.argsort(metric[cand])[::-1]]:
        if all(abs(k - s) >= gap for s in starts):
            starts.append(int(k))
    return sorted(starts), metric
