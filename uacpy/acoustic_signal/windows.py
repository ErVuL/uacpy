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
#: to: about 1 Hz bins at an 8 kHz rate, 5.9 Hz at 48 kHz (uacpy guidance).
#: Being in samples, it is 8.2 s at 1 kHz and 43 ms at 192 kHz.
_DEFAULT_NPERSEG = 8192

#: Least overlap (fraction of a segment) that keeps every independent sample
#: a flat-top window yields. Segments may be spaced at most ``N / eta_w``
#: apart (Abraham, *Underwater Acoustic Signal Processing*, 2019, §9.2.10.1,
#: Eq. 9.162), ``eta_w`` being the window's independent sample rate (§4.10,
#: Eq. 4.122). That equation gives eta_w = 4.60 for scipy's periodic
#: ``flattop`` at any N from 128 up (hann 2.08, boxcar 1.50; Table 4.2's
#: 2.10 and 1.5 are the symmetric windows at N = 128), so the overlap is
#: 1 - 1/4.60 = 78.3 %.
_FLATTOP_OVERLAP = 1.0 - 1.0 / 4.60

#: Phase-averaged power bias (dB) above which a real tone's negative-frequency
#: image is reported. The image leaks through the analysis window with weight
#: ``ratio = |Σ w e^{-4πi f t}| / Σ w`` (``|W(2f)|/Σw``) and inflates the tone's
#: power by ``1 + ratio**2`` averaged over its phase; 0.01 dB is
#: ``ratio ≈ 0.048``. The constant-Q estimators and the tone extractor share
#: it. uacpy guidance: the literature states the effect — the mirror-image term
#: "will be small as long as it does not 'interfere' with the positive
#: frequency term … if f0 is not near 0 or 1/2" (Kay, *Fundamentals of
#: Statistical Signal Processing* Vol. III, App. 1A, Fig. 1A.1) — but no level.
_IMAGE_BIAS_WARN_DB = 0.01


def _default_noverlap(window, nperseg: int) -> int:
    """The default overlap (samples) of a Welch segmentation of ``nperseg``
    samples under ``window``, the one rule every Welch-based estimator uses.

    Abraham §9.2.10.1 spaces segments at most ``N / eta_w`` apart so that no
    independent sample the window yields goes unused, ``eta_w`` being the
    window's independent sample rate (Eq. 4.122). This function applies that
    rule to the two windows the estimators default to: half a segment for
    Hann — Abraham's own figure, "approximately a 50% overlap" — and
    :data:`_FLATTOP_OVERLAP`, 78.3 % rounded up, for flat-top. Abraham adds
    that "overlapping the windows more … improves performance".

    Any other window (a name or an array) takes half a segment (uacpy
    guidance: one overlap for every window but flat-top), which meets the
    rule only for ``eta_w <= 2``: rectangular (1.50) needs 33.3 % and
    Hamming (1.90) 47.3 %, but Kaiser ``beta = 2.5π`` (2.31) needs 56.7 %
    and Blackman (2.41) 58.5 % (Eq. 4.122 on scipy's periodic windows at
    N = 1024; Table 4.2's 1.91, 2.33 and 2.43 are the symmetric windows at
    N = 128), so half a segment leaves those estimates with more variance
    than the record allows. Pass ``noverlap`` for them.
    """
    nperseg = int(nperseg)
    if isinstance(window, str) and window.lower() == 'flattop':
        return min(nperseg - 1, int(np.ceil(_FLATTOP_OVERLAP * nperseg)))
    return nperseg // 2


def _welch_nperseg(nperseg, window, n, who, *, span="the record"):
    """The ``nperseg`` a Welch segmentation hands scipy on ``span``, ``n``
    samples long, the one rule every Welch-based estimator uses.

    Unset (``None``), it is an array window's own length, else
    :data:`_DEFAULT_NPERSEG` clamped to the span: the segment scipy would
    clamp it to, without scipy's UserWarning for a segment longer than the
    input. A given one goes as given, and is refused when longer than the
    span, which holds no segment of it. An array window IS the segment: a
    given ``nperseg`` that differs from its length is refused, as is a window
    longer than the span (scipy's own refusals of both name neither the
    function called nor the fix)."""
    if not isinstance(window, (str, tuple)):
        if (nperseg is not None and np.ndim(window) == 1
                and len(window) != int(nperseg)):
            raise ConfigurationError(
                f"{who}: an array window of {len(window)} samples sets the "
                f"segment length, and nperseg={nperseg} differs from it.",
                remediation=(f"Pass nperseg={len(window)}, or nperseg=None to "
                             f"take the window's length, or a named window "
                             f"such as 'hann'."))
        size = len(window) if nperseg is None else int(nperseg)
        if size > n:
            raise ConfigurationError(
                f"{who}: the array window of {size} samples is longer than "
                f"{span} ({n} samples), which holds no segment of it.",
                remediation=(f"Pass a window of {n} samples or fewer, or a "
                             f"named window such as 'hann'."))
        return size
    if nperseg is None:
        return min(_DEFAULT_NPERSEG, n)
    if int(nperseg) > n:
        raise ConfigurationError(
            f"{who}: nperseg={nperseg} is longer than {span} ({n} "
            f"samples), which holds no segment of that length.",
            remediation=(f"Pass nperseg={n} or fewer, or leave nperseg "
                         f"unset: it then takes {_DEFAULT_NPERSEG} samples, "
                         f"or all {n} when that is shorter."))
    return int(nperseg)


def _taper(name, n: int, *, who: str) -> np.ndarray:
    """Window of ``n`` samples, shared by the tone extractor, the tone burst
    and the IFFT synthesis.

    The window spans all ``n`` samples — it is not an edge taper. On a time
    record (the tone extractor) the caller normalises by its sum, so a tone's
    amplitude is unchanged; on a frequency band (the IFFT synthesis) it is a
    filter that reshapes a pulse, which is why the synthesis defaults to no
    taper.

    ``name`` is one of :data:`_WINDOWS`; each caller resolves its own
    ``window=None`` to its documented default first, so ``None`` is refused
    here. Returns a flat window for ``'boxcar'``, and for a span too short to
    keep an interior (warning in that case). ``'tukey'`` is scipy's tapered
    cosine with ``alpha = 0.5`` (half the span tapered, uacpy guidance).

    The two callers sit at different depths — the tone extractor is one frame
    below its public method, the synthesis planner three — so the warning
    below carries no frame count of its own: a count passed in by the caller
    can only be right for one of the two depths, and points at this helper's
    own frame from the other."""
    if name not in _WINDOWS:
        raise ConfigurationError(
            f"{who}: unknown window={name!r}; valid: "
            f"{', '.join(repr(w) for w in _WINDOWS)} ('boxcar' is no "
            f"taper)."
        )
    if name == 'boxcar':
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
    """Length-``n`` taper for a ``scipy.signal.get_window`` spec: a name
    (``'hann'``; ``'boxcar'`` is no taper) or a ``(name, *params)`` tuple
    (``('kaiser', 8)``). Periodic (``fftbins=True``) form, the correct
    convention for spectra. The caller resolves ``window=None`` to its own
    documented default first. Distinct from
    :func:`uacpy.acoustic_signal.windows._taper`, which takes five named
    symmetric windows.
    """
    # Deferred: the synthesis imports this module, and uacpy.io loads the
    # synthesis without scipy (test_lazy_imports).
    from scipy.signal import get_window
    return get_window(spec, int(n), fftbins=True).astype(float)


def _fk_tapers(window, nt, nx):
    """Separable (time, space) tapers for the 2-D f-k window.

    ``window`` applies one spec to both axes; a 2-element ``list``
    ``[time_spec, space_spec]`` tapers the axes independently. ``None``, on
    the whole or on either axis, is the f-k transforms' documented default,
    ``'boxcar'`` (no taper).
    """
    if isinstance(window, list):
        if len(window) != 2:
            raise ConfigurationError(
                "fk_transform: window list must be [time_window, space_window]"
                f"; got {len(window)} entries.")
        t_spec, x_spec = window
    else:
        t_spec = x_spec = window
    t_spec = "boxcar" if t_spec is None else t_spec
    x_spec = "boxcar" if x_spec is None else x_spec
    return _spectral_window(t_spec, nt), _spectral_window(x_spec, nx)
