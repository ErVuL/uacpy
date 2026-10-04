"""The spectral windows the package's estimators and syntheses share."""

from __future__ import annotations

import warnings

import numpy as np

from uacpy.core.exceptions import ConfigurationError, NumericsWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP


# Fewest samples a window can act on and still leave an
# interior. numpy's hann/hamming/blackman are symmetric with (near-)zero
# endpoints, so at 2 bins the window is [0, 0] and at 3 it is [0, 1, 0] — a
# taper there does not soften the edges, it deletes the band.
_MIN_TAPERABLE_BINS = 4

_WINDOWS = ('hann', 'hamming', 'blackman', 'tukey', 'boxcar')

#: The Welch segment length (samples) every Welch-based estimator defaults
#: to: about 1 Hz bins at an 8 kHz rate, 5.9 Hz at 48 kHz.
_DEFAULT_NPERSEG = 8192

#: Least overlap (fraction of a segment) that keeps every independent sample
#: a flat-top window yields. Segments may be spaced at most ``N / eta_w``
#: apart (Abraham, *Underwater Acoustic Signal Processing*, 2019, §9.2.10.1,
#: Eq. 9.162), ``eta_w`` being the window's independent sample rate (§4.10,
#: Eq. 4.122). That equation gives eta_w = 4.60 for scipy's ``flattop`` at
#: any N from 128 up (hann 2.08, boxcar 1.50 as in Table 4.2), so the overlap
#: is 1 - 1/4.60 = 78.3 %.
_FLATTOP_OVERLAP = 1.0 - 1.0 / 4.60


def _default_noverlap(window, nperseg: int) -> int:
    """The default overlap (samples) of a Welch segmentation of ``nperseg``
    samples under ``window``, the one rule every Welch-based estimator uses.

    Enough overlap to keep every independent sample the window yields
    (Abraham §9.2.10.1): half a segment for Hann — Abraham's own figure,
    "approximately a 50% overlap" — and :data:`_FLATTOP_OVERLAP`, 78.3 %
    rounded up, for flat-top, whose stronger taper leaves fewer independent
    samples per segment. Less overlap than this raises the variance of the
    estimate above what the record allows; more costs only computation. Any
    other window (a name or an array) takes half a segment.
    """
    nperseg = int(nperseg)
    if isinstance(window, str) and window.lower() == 'flattop':
        return min(nperseg - 1, int(np.ceil(_FLATTOP_OVERLAP * nperseg)))
    return nperseg // 2


def _taper(name, n: int, *, who: str) -> np.ndarray:
    """Window of ``n`` samples, shared by the tone extractor and the IFFT.

    The window spans all ``n`` samples — it is not an edge taper. On a time
    record (the tone extractor) the caller normalises by its sum, so a tone's
    amplitude is unchanged; on a frequency band (the IFFT synthesis) it is a
    filter that reshapes a pulse, which is why the synthesis defaults to
    ``None`` (rectangular) whenever a waveform is given.

    Returns a flat window for ``None`` and ``'boxcar'``, and for a span too
    short to keep an interior (warning in the latter case).

    The two callers sit at different depths — the tone extractor is one frame
    below its public method, the synthesis planner three — so the warning
    below carries no frame count of its own: a count passed in by the caller
    can only be right for one of the two depths, and points at this helper's
    own frame from the other."""
    if name is not None and name not in _WINDOWS:
        raise ConfigurationError(
            f"{who}: unknown window={name!r}; "
            f"valid: None (rectangular), "
            f"{', '.join(repr(w) for w in _WINDOWS)}."
        )
    if name is None or name == 'boxcar':
        return np.ones(n)
    if n < _MIN_TAPERABLE_BINS:
        warnings.warn(
            f"{who}: a {n}-sample span is too narrow to taper (a {name!r} "
            f"window needs at least {_MIN_TAPERABLE_BINS} samples to leave an "
            f"interior); continuing untapered. Widen the span for a resolved "
            f"result.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
        return np.ones(n)
    if name == 'hann':
        return np.hanning(n)
    if name == 'hamming':
        return np.hamming(n)
    if name == 'blackman':
        return np.blackman(n)
    from scipy.signal import windows
    return windows.tukey(n, alpha=0.5)


def _spectral_window(spec, n):
    """Length-``n`` taper for a ``scipy.signal.get_window`` spec.

    ``spec`` is ``None`` (rectangular ``ones``) or any get_window argument —
    a name (``'hann'``) or a ``(name, *params)`` tuple (``('kaiser', 8)``).
    Periodic (``fftbins=True``) form, the correct convention for spectra.
    Distinct from :func:`uacpy.acoustic_signal.windows._taper`, which takes
    five named symmetric windows and spells rectangular ``None``.
    """
    if spec is None:
        return np.ones(int(n))
    # Deferred: the synthesis imports this module, and uacpy.io loads the
    # synthesis without scipy (test_lazy_imports).
    from scipy.signal import get_window
    return get_window(spec, int(n), fftbins=True).astype(float)


def _fk_tapers(window, nt, nx):
    """Separable (time, space) tapers for the 2-D f-k window.

    ``window`` applies one spec to both axes; a 2-element ``list``
    ``[time_spec, space_spec]`` tapers the axes independently.
    """
    if isinstance(window, list):
        if len(window) != 2:
            raise ConfigurationError(
                "fk_transform: window list must be [time_window, space_window]"
                f"; got {len(window)} entries.")
        t_spec, x_spec = window
    else:
        t_spec = x_spec = window
    return _spectral_window(t_spec, nt), _spectral_window(x_spec, nx)
