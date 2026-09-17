"""What a recorded signal contains: spectra, levels, bands and
time-frequency views.

One question — *measure this signal* — over three axes. Time-averaged
estimates (:func:`welch`, :func:`constant_q`, :func:`sound_exposure` and a
``probabilistic_`` twin of each) say what is there on average or as a
distribution; time-resolved views (:func:`spectrogram`,
:func:`constant_q_spectrogram`, :func:`cwt`, :func:`wigner_ville`,
:func:`cepstrum`) say when; and the band ladders
(:func:`decidecade_bands`, :func:`decidecade_band_levels`) say on which
standard bands a level is reported.

They live together because they share one subject and one vocabulary: the
constant-Q transform is how :func:`constant_q` resolves frequency, and
:func:`sound_exposure` is written on the same ladder
:func:`decidecade_bands` defines. Keeping them apart meant each side
importing the other.
"""

from __future__ import annotations

import math
import warnings
from collections import namedtuple
import numpy as np
import scipy.signal as _sig
from scipy.signal import get_window
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.constants import REFERENCE_PRESSURE_WATER
from uacpy.core.acoustics import power_to_dB
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.acoustic_signal._signal_validate import (
    require_at_most_nyquist,
    require_finite_signal,
    require_positive_finite_scalar,
)
from uacpy.acoustic_signal._signal_validate import require_increasing_axis
from functools import lru_cache
from scipy.integrate import cumulative_trapezoid
from scipy.signal import hilbert
from scipy.special import gamma


# ──────────────────────────────────────────────────────────────────────
# Spectral estimators
#
# The six — welch, constant_q and
# sound_exposure, each with a probabilistic twin — their two result types, and
# the constant-Q machinery they are built on.
# ──────────────────────────────────────────────────────────────────────

def _warn_two_sided(caller: str, data):
    """Warn that complex input yields a two-sided, unsorted frequency axis.

    scipy's Welch/spectrogram switch to ``return_onesided=False`` for complex
    input: the axis then runs ``0 .. fs/2`` and continues at ``-fs/2 .. 0``,
    which is neither one-sided nor monotonic, so anything that assumes an
    ascending axis (interpolation, band selection, a plot) reads it wrong.
    The estimate itself is correct — only the docstrings' one-sided promise
    does not hold — so this is a warning, not the rejection the real-pressure
    estimators (``sound_exposure``, ``cwt``, ``analytic_signal``) issue.
    """
    if np.iscomplexobj(data):
        warnings.warn(
            f"{caller}: complex input gives a TWO-SIDED spectrum — "
            f"'frequencies' runs 0..fs/2 then -fs/2..0 and is not sorted, and "
            f"the density is not the one-sided Pa^2/Hz this function "
            f"documents. Sort both arrays together (i = np.argsort(f)) or "
            f"np.fft.fftshift them, or pass a real pressure series.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)


#: The three normalisations every spectral estimator here offers.
#:
#: ``'density'`` divides by the window's noise-equivalent bandwidth, giving
#: power per hertz (Pa²/Hz). The result is then independent of ``window`` and
#: ``nperseg``, which is what makes it the statistic for NOISE.
#:
#: ``'spectrum'`` does not, giving the power in each band (Pa²). An on-bin tone
#: reads its full A²/2, which is what makes it the statistic for TONES and for
#: band levels.
#:
#: ``'exposure'`` multiplies the band power by the record's duration, giving
#: the energy in each band (Pa²·s) — the quantity ISO 18405 calls sound
#: exposure, and with a ``band_type`` the sound exposure level per standard
#: band. It is the one scaling that depends on how long the record
#: is, which is the point: an event is measured by the energy it delivers,
#: not by the power it averages.
#: It also pins the window (:data:`_ENERGY_WINDOWS`), because only an
#: energy-preserving window keeps the sum over bins equal to the signal's own
#: energy.
#:
#: Density and spectrum differ by that bandwidth — 70.3 Hz, or 18.5 dB, for a
#: 1024-point Hann at 48 kHz — and spectrum and exposure by the duration, so
#: which one a number is cannot be read off its value.
_SPECTRAL_SCALINGS = ("density", "spectrum", "exposure")
#: What a BIN estimator offers: the two normalisations that are one
#: divide apart. An energy is :func:`sound_exposure`'s, for the reason
#: :func:`_require_bin_scaling` gives.
_BIN_SCALINGS = ("density", "spectrum")
#: How the frequency axis is resolved, orthogonal to :data:`_SPECTRAL_SCALINGS`.
#:
#: ``'welch'`` averages the periodograms of uniform, equal-length segments:
#: linear frequency bins of equal width, set by ``nperseg``.
#:
#: ``'constant_q'`` correlates against one kernel per geometric bin, each Q
#: cycles long: logarithmic bins whose width grows with frequency, set by
#: ``bins_per_octave``.
#:
#: The two take different options because the option IS the method --
#: ``nperseg``/``noverlap``/``nfft`` against
#: ``bins_per_octave``/``hop`` -- and passing one method's to the other raises
#: rather than being ignored.
#:
#: Reporting on standard frequency bands is NOT a third method: it is those
#: same Welch bins summed into the bands of ``band_type``, so it is an
#: argument (see :data:`BAND_TYPES`) rather than a peer of the two transforms
#: that actually resolve frequency differently.
_SPECTRAL_METHODS = ("welch", "constant_q")
#: Standard ladders :func:`sound_exposure` can integrate its bins into.
#:
#: ``'decidecade'`` is IEC 61260-1 / ISO 18405 base-10 (centres
#: ``1000*10**(n/10)``), ``'third_octave'`` and ``'octave'`` the base-2
#: ladders, and ``'linear'`` is ``num_bands`` equal-width bands. ``None``, the
#: default, reports the method's own bins.
#:
#: Only the Welch bins can be integrated: they are orthogonal, so each bin
#: belongs to exactly one band and the sum is the band's power. Constant-Q
#: kernels overlap by construction, so summing them would count the same
#: energy more than once — asking for both raises.
BAND_TYPES = ("decidecade", "third_octave", "octave", "linear")
#: Options that belong to the banding rather than to either method, so they
#: are refused when no ``band_type`` was asked for: there is nothing for them
#: to describe.
_BAND_OPTIONS = ("num_bands", "batch_size")
#: Spectral lines a band wants before an FFT-synthesised band level behaves
#: like a filter-bank one: "at least ten spectral lines in each band" (Fahy,
#: *Sound Intensity*, on synthesising constant-percentage-bandwidth spectra
#: from an FFT). At the 1 Hz bins ``nperseg`` defaults to, the seven lowest
#: decidecade bands (10-40 Hz) hold 2 to 7 lines and fall short of it.
#:
#: That is a resolution limit, not an error in the total: bins are
#: orthogonal, so each one's energy lands in exactly one band and the sum
#: over bands stays exact. What coarse bins cost is WHERE the energy lands —
#: a band edge falls between bins, so energy either side of it is assigned in
#: whole-bin quanta. Raising ``nperseg`` to about ``10 * sample_rate / 2.3``
#: (4.3 s of record, for the 10 Hz band's 2.3 Hz width) resolves them, at the
#: cost of time resolution; it is a choice rather than a default because the
#: 1 Hz grid is what soundscape reporting is written on.
_MIN_LINES_PER_BAND = 10
#: How much of a record the band method reads at a time when ``batch_size`` is
#: left unset, rounded to whole segments. Large enough that the per-batch
#: overhead is negligible, small enough that a multi-hour record never
#: materialises as one segment matrix.
_TARGET_BATCH_SAMPLES = 262144
#: Window each scaling uses when the caller names none, because the two are
#: measuring different things and the best window differs with the question.
#:
#: A density averages noise, where a narrow main lobe matters and scalloping
#: does not: ``hann``, 1.50-bin noise-equivalent bandwidth.
#:
#: A spectrum reads a tone's level, where scalloping IS the error: a tone half
#: a bin off centre reads 1.42 dB low under ``hann`` and about 0.01 dB low
#: under ``flattop``. The cost is resolution — flat-top's main lobe spans
#: 3.77 bins against hann's 1.50 — so two tones closer than that merge. Pass
#: ``window='hann'`` to a spectrum when resolving neighbours matters more than
#: reading their exact level.
#: Constant-Q keeps hann under both scalings: a kernel's length sets its
#: bin's bandwidth, so swapping in flattop would widen every bin by 2.5x and
#: change the Q the method is named for. Welch has no such coupling.
_DEFAULT_WINDOW = {
    "welch": {"density": "hann", "spectrum": "flattop", "exposure": "boxcar"},
    "constant_q": {"density": "hann", "spectrum": "hann",
                   "exposure": "hann"},
}
#: Integrating bins into bands needs every bin counted once and whole, which
#: only a rectangular window with no overlap gives, so banding pins the window
#: whatever the scaling is.
_BAND_WINDOW = "boxcar"


def _trim_record(data, sample_rate, integration_time, caller):
    """The first ``integration_time`` seconds of ``data``, or all of it.

    ISO 18405's integration time, and the same idea for the other statistics:
    the stretch of record the estimate is taken over. Guarded before it
    reaches the slice, because a negative one is a Python end-slice — -1.0 s
    of a 5 s record would return bit-identically what +4.0 s returns — and
    NaN / Inf raise an untyped ValueError out of ``int()``.
    """
    if integration_time is None:
        return data
    # Before the arithmetic below divides by it: a zero or non-finite rate
    # reaches this function ahead of the estimator's own check (the trim
    # happens first), and 1/0 is a ZeroDivisionError with no name on it.
    sample_rate = require_positive_finite_scalar(
        sample_rate, caller, "sample_rate", " Hz")
    integration_time = require_positive_finite_scalar(
        integration_time, caller, "integration_time", " s")
    n = min(int(integration_time * sample_rate), np.shape(data)[-1])
    if n < 1:
        raise ConfigurationError(
            f"{caller}: no samples to integrate — integration_time="
            f"{integration_time:g} s is shorter than one sample at "
            f"{sample_rate:g} Hz.",
            remediation=f"Pass integration_time >= "
                        f"{1.0 / float(sample_rate):g} s, or leave it unset "
                        f"to use the whole record.")
    return np.asarray(data)[..., :n]


def _crop_to_range(frequencies, values, fmin, fmax):
    """The bins inside ``[fmin, fmax]``, for a method that computes them all.

    Welch resolves the whole spectrum whatever range is asked for, so the
    range is a crop of what came back rather than a saving — the estimate in
    the bins that remain is the one it would have been either way.
    """
    if fmin is None and fmax is None:
        return frequencies, values
    keep = np.ones(np.shape(frequencies), dtype=bool)
    if fmin is not None:
        keep &= frequencies >= fmin
    if fmax is not None:
        keep &= frequencies <= fmax
    return frequencies[keep], values[..., keep]


class SpectralEstimate(namedtuple("SpectralEstimate", "frequencies power")):
    """An averaged power estimate: ``frequencies`` and linear ``power``.

    The tuple is the measurement, so ``frequencies, power = welch(x, fs)``
    keeps working; everything that says what the measurement MEANS
    is an attribute, survives pickling and copying, and takes no part in
    equality:

    ``scaling``
        ``'density'`` (Pa²/Hz), ``'spectrum'`` (Pa²) or ``'exposure'``
        (Pa²·s) — which statistic ``power`` holds.
    ``method``
        ``'welch'`` or ``'constant_q'``: whether the bins are equal width or
        geometric, which is what lets a plot choose its frequency axis
        without being told.
    ``bands`` / ``band_type``
        The ``(low, centre, high)`` edges and the ladder they came from when
        the values sit on standard bands (``sound_exposure``), else ``None``.

    Carried at all because a plot axis reading "/Hz" over band power is wrong
    by the window's noise-equivalent bandwidth and says nothing about being
    wrong. A consumer that has to be told which it holds is a consumer that
    can be told the wrong one.
    """

    def __new__(cls, frequencies, power, scaling="density", method="welch",
                bands=None, band_type=None):
        self = super().__new__(cls, frequencies, power)
        self.scaling = str(scaling)
        self.method = str(method)
        self.bands = bands
        self.band_type = band_type
        return self

    def __getnewargs__(self):
        return (self.frequencies, self.power, self.scaling, self.method,
                self.bands, self.band_type)

    def plot(self, **kwargs):
        """Draw this estimate through the plotter it calls for.

        A bin estimate goes to :func:`uacpy.visualization.plot_psd` as a
        line; a banded one goes to :func:`uacpy.visualization.plot_sel` as
        bars, since its values belong to whole standard bands rather than to
        points on a frequency axis. The unit and the title come from the
        estimate's own ``scaling`` and ``method``, so the call cannot
        mislabel it. ``kwargs`` reach the plotter — ``ref=``,
        ``freq_scale='linear'``, ``ymin`` / ``ymax``, any matplotlib keyword.
        Returns ``(fig, ax)``, as every plotter in the package does.
        """
        # Deferred into the body: ``uacpy.visualization`` imports
        # ``uacpy.core`` at module scope, so this at file scope would make
        # ``import uacpy`` raise (docs/DEV.md section 7).
        from uacpy import visualization
        if self.bands is not None:
            return visualization.plot_sel(self, **kwargs)
        return visualization.plot_psd(self, **kwargs)


class ProbabilisticSpectralEstimate(
        namedtuple("ProbabilisticSpectralEstimate",
                   "frequencies level_edges pdf")):
    """A histogram of spectral levels: one probability column per frequency.

    What the ``probabilistic_*`` estimators return, whichever statistic they
    name. The tuple is the measurement — ``frequencies``, ``level_edges`` and
    the ``pdf`` over them — and everything that says what it MEANS is an
    attribute, the same names :class:`SpectralEstimate` uses where they
    overlap:

    ``mean_dB`` / ``std_dB``
        Per-frequency summaries over *all* segments, so they are not clipped
        by ``lvlmin`` / ``lvlmax`` the way the histogram is.
    ``binwidth_dB``
        Height of one histogram bin, the ``ddB`` it was asked for.
    ``scaling`` / ``method`` / ``bands`` / ``band_type``
        As on :class:`SpectralEstimate`.
    ``ref``
        The reference the dB levels are stated against.
    ``seg_duration``
        The interval each sample describes, in seconds; ``None`` for a
        constant-Q histogram, which samples single frames whose length is per
        bin rather than one duration for the whole estimate.
    """

    def __new__(cls, frequencies, level_edges, pdf, *, mean_dB=None,
                std_dB=None, binwidth_dB=None, seg_duration=None,
                ref=REFERENCE_PRESSURE_WATER, scaling="density",
                method="welch", bands=None, band_type=None):
        self = super().__new__(cls, frequencies, level_edges, pdf)
        self.mean_dB = mean_dB
        self.std_dB = std_dB
        self.binwidth_dB = binwidth_dB
        self.seg_duration = seg_duration
        self.ref = float(ref)
        self.scaling = str(scaling)
        self.method = str(method)
        self.bands = bands
        self.band_type = band_type
        return self

    def __getnewargs_ex__(self):
        # Pickle and copy go back through __new__, which takes the
        # descriptors by keyword; the plain __getnewargs__ a namedtuple
        # supplies would drop every one of them.
        return ((self.frequencies, self.level_edges, self.pdf),
                {"mean_dB": self.mean_dB, "std_dB": self.std_dB,
                 "binwidth_dB": self.binwidth_dB,
                 "seg_duration": self.seg_duration, "ref": self.ref,
                 "scaling": self.scaling, "method": self.method,
                 "bands": self.bands, "band_type": self.band_type})

    def plot(self, **kwargs):
        """Draw this histogram through the plotter its ``method`` calls for.

        A constant-Q histogram goes to
        :func:`uacpy.visualization.plot_constant_q_ppsd` and the rest to
        :func:`uacpy.visualization.plot_ppsd`; both read ``ref`` and
        ``scaling`` off the result, so the level axis names the reference the
        estimate was computed against and claims "/Hz" only over a density.
        Returns ``(fig, ax)``.
        """
        from uacpy import visualization
        if self.method == "constant_q":
            return visualization.plot_constant_q_ppsd(self, **kwargs)
        return visualization.plot_ppsd(self, **kwargs)


def _require_bin_scaling(scaling, caller):
    """A bin estimator returns a power, never an energy.

    ``'exposure'`` is the third member of :data:`_SPECTRAL_SCALINGS`, but only
    :func:`sound_exposure` reaches it: an energy needs a rectangular window,
    no overlap and no detrending, and those are not arguments there — which
    is the whole point of it being its own function. Reaching it through a bin
    estimator would honour that estimator's own window and overlap instead,
    and return a number 0.9 % out at the default settings and 54 % out under
    ``window='hann'``, with nothing in it to say so.
    """
    if scaling not in _BIN_SCALINGS:
        raise ConfigurationError(
            f"{caller}: unknown scaling {scaling!r}.",
            remediation=f"Use one of {_BIN_SCALINGS}: 'density' for power per "
                        f"hertz (noise), 'spectrum' for power per band "
                        f"(tones). For the energy a record delivered, call "
                        f"sound_exposure(), which pins the window and overlap "
                        f"that identity needs.")


def _check_band_type(band_type, caller):
    """The ladder a caller named has to be one the package builds.

    The other ways a banding request could go wrong — a ladder over
    constant-Q kernels, a Welch knob that breaks the bin sum, a banding
    option with no banding — are not reachable any more: no function's
    signature offers that combination, which is the point of a function per
    statistic.
    """
    if band_type not in BAND_TYPES:
        raise ConfigurationError(
            f"{caller}: unknown band_type {band_type!r}.",
            remediation=f"Use one of {BAND_TYPES}.")


def _estimate(data, sample_rate, *, method="welch", scaling="density",
                      window=None, fmin=None, fmax=None, band_type=None,
                      integration_time=None, _caller="welch",
                      **options):
    """Power estimate over frequency, under any method and any scaling.

    Returns a :class:`SpectralEstimate` carrying both choices, so a consumer
    never has to be told what it holds.

    ``method`` (:data:`_SPECTRAL_METHODS`) resolves the frequency axis:
    ``'welch'`` averages uniform segments into equal-width bins and takes
    ``nperseg`` / ``noverlap`` / ``nfft``; ``'constant_q'`` correlates one
    Q-cycle kernel per geometric bin and takes ``bins_per_octave`` / ``hop``;
    An option belonging to the other method raises rather than being
    ignored.

    ``band_type`` (:data:`BAND_TYPES`) reports on standard frequency bands
    instead of on the method's own bins: the Welch bins summed into each
    band, which is exact because they are orthogonal. It takes ``num_bands``
    (for ``'linear'``) and ``batch_size``, pins the window to ``boxcar`` so
    every bin is counted once and whole, and is refused over constant-Q,
    whose kernels overlap. Unset, the estimate is per bin.

    ``scaling`` (:data:`_SPECTRAL_SCALINGS`) sets the normalisation, and it is
    orthogonal to the method: ``'density'`` divides each bin's power by that
    bin's own noise-equivalent bandwidth — one number for Welch, a different
    one per bin for constant-Q, since its bins widen with frequency —
    ``'spectrum'`` leaves the power in the band, and ``'exposure'`` multiplies
    it by the duration of the record, giving the energy per band (Pa²·s).

    ``fmin`` / ``fmax`` are the frequency range of the estimate under every
    method, and ``integration_time`` the stretch of record it is taken over
    (seconds from the start). What an unset one means is the method's own
    answer: Welch resolves the whole spectrum and the range crops what comes
    back, constant-Q starts at 20 Hz and runs to Nyquist, and the band ladder
    spans the 10 Hz to 20 kHz reporting range.

    ``window=None`` takes the default that scaling measures best under that
    method (:data:`_DEFAULT_WINDOW`), which is ``flattop`` for a Welch
    spectrum, ``boxcar`` wherever the sum has to equal the record's energy,
    and ``hann`` otherwise; see :func:`welch` for why.

    :func:`welch`, :func:`constant_q` and :func:`sound_exposure` are the public
    doors onto this; each names one statistic and takes only the arguments
    that statistic can honour.

    Reference-free (linear). Convert to dB at plot time with
    :func:`uacpy.visualization.plot_psd` (which takes ``ref=``).
    """
    if scaling not in _SPECTRAL_SCALINGS:
        raise ConfigurationError(
            f"{_caller}: unknown scaling {scaling!r}.",
            remediation=f"Use one of {_BIN_SCALINGS}, or sound_exposure() "
                        f"for the energy a record delivered.",
        )
    band_options = {k: options.pop(k) for k in _BAND_OPTIONS if k in options}
    window = (_BAND_WINDOW if band_type is not None else
              _DEFAULT_WINDOW[method][scaling]) if window is None else window
    data = _trim_record(data, sample_rate, integration_time, _caller)
    if band_type is not None:
        # Not a third way of resolving frequency: the Welch bins this same
        # function returns, summed into the ladder. ``_band_estimate`` calls
        # back into it, so there is one estimator underneath.
        band_range = {k: v for k, v in (("fmin", fmin), ("fmax", fmax))
                      if v is not None}
        centres, values, bands = _band_estimate(
            data, sample_rate, scaling=scaling, band_type=band_type,
            window=window, caller=_caller, **band_range, **band_options,
            **options)
        return SpectralEstimate(centres, values, scaling, method, bands,
                                band_type)
    if method == "constant_q":
        kernel_range = {k: v for k, v in (("fmin", fmin), ("fmax", fmax))
                        if v is not None}
        freqs, power = _constant_q_estimate(
            data, sample_rate, window=window,
            scaling=scaling, caller=_caller, **kernel_range, **options)
        return SpectralEstimate(freqs, power, scaling, method)
    # Guards name the function the CALLER used, not this one: a message
    # reading "_estimate: ..." for a bad argument to ``welch`` sends the
    # reader to a function they did not call.
    data = require_finite_signal(data, _caller)
    sample_rate = require_positive_finite_scalar(
        sample_rate, _caller, "sample_rate", " Hz")
    _warn_two_sided(_caller, data)
    duration = np.shape(data)[-1] / float(sample_rate)
    if scaling == "exposure":
        # Welch averages whole segments and DROPS the samples that do not
        # fill one, so a plain average times the record length would quietly
        # lose the tail (0.19 % of a 4 s record at nperseg=1024). Padding to a
        # whole number of segments adds no energy, and the padded duration is
        # then what turns the average back into the energy that passed the
        # sensor — exactly, for any nperseg. It is the same trick the band
        # method uses on its chunks.
        nps = int(options.get("nperseg", 8192))
        n = np.shape(data)[-1]
        n_seg = max(1, -(-n // nps))
        pad = n_seg * nps - n
        if pad:
            data = np.pad(np.asarray(data),
                          [(0, 0)] * (np.ndim(data) - 1) + [(0, pad)])
        duration = n_seg * nps / float(sample_rate)
    freqs, Pxx = _sig.welch(data, sample_rate, window=window,
                            nperseg=options.get("nperseg", 8192),
                            noverlap=options.get("noverlap", 0
                                                 if scaling == "exposure"
                                                 else None),
                            nfft=options.get("nfft"),
                            detrend=options.get("detrend", False
                                                if scaling == "exposure"
                                                else "constant"),
                            average=options.get("average", "mean"),
                            scaling="spectrum" if scaling == "exposure"
                            else scaling)
    if scaling == "exposure":
        # Parseval: with a boxcar window, no overlap and no detrending the
        # bin powers sum to the signal's mean square, so multiplying by the
        # duration gives the energy that passed the sensor, split over the
        # bins. Overlap and detrending are defaulted off here for the same
        # reason the window is boxcar, and an explicit one is honoured — that
        # is the caller saying they want the windowed estimate, and it is no
        # longer the record's energy.
        Pxx = Pxx * duration
    # Welch resolves every bin whatever range was asked for, so the range is
    # a crop of the answer rather than a cheaper answer.
    freqs, Pxx = _crop_to_range(freqs, Pxx, fmin, fmax)
    return SpectralEstimate(freqs, Pxx, scaling, method)


def welch(data, sample_rate, *, scaling="density", nperseg=8192,
          noverlap=None, nfft=None, detrend="constant", average="mean",
          window=None, fmin=None, fmax=None, integration_time=None):
    """Welch estimate on equal-width bins: Pa²/Hz or Pa² per bin.

    Averaged periodograms of overlapping segments (Welch 1967), which is the
    workhorse spectral estimate: ``nperseg`` sets the resolution, and the
    averaging trades variance for it.

    Parameters
    ----------
    scaling : {'density', 'spectrum'}, optional
        ``'density'`` divides each bin's power by the window's
        noise-equivalent bandwidth, giving Pa²/Hz — independent of ``window``
        and ``nperseg``, which is what makes it the statistic for NOISE.
        ``'spectrum'`` leaves the power in the bin, giving Pa², where an
        on-bin tone reads its full ``A²/2`` — the statistic for TONES. They
        differ by that bandwidth (18.5 dB for a 1024-point hann at 48 kHz),
        so which one a number is cannot be read off its value; the estimate
        carries ``.scaling`` for exactly that reason.
    nperseg, noverlap, nfft, detrend, average
        The segmentation, forwarded to :func:`scipy.signal.welch`.
        ``average='median'`` is the robust choice for a record with
        transients in it — a passing ship moves the mean of the periodograms
        and leaves the median on the background.
    window : str, optional
        ``None`` takes the default the scaling measures best: ``hann`` for a
        density, where a narrow main lobe matters and scalloping does not;
        ``flattop`` for a spectrum, where a tone half a bin off centre reads
        about 0.01 dB low instead of hann's **1.42 dB**. Flat-top pays for it
        in resolution: its noise-equivalent bandwidth is 3.77 bins against
        hann's 1.50, and its main lobe 10 bins against hann's 4, so two tones
        closer than that merge. Pass ``window='hann'`` where separating
        neighbours matters more than reading their level.
    fmin, fmax : float, optional
        Frequency range to report. Welch resolves the whole spectrum either
        way, so this crops the answer rather than making it cheaper.
    integration_time : float, optional
        Seconds of record to use, from the start; the whole record by
        default.

    Returns
    -------
    SpectralEstimate
        ``(frequencies, power)``, linear and reference-free, carrying
        ``.scaling``. ``.plot()`` draws it in dB.
    """
    _require_bin_scaling(scaling, "welch")
    return _estimate(data, sample_rate, scaling=scaling, window=window,
                     fmin=fmin, fmax=fmax, integration_time=integration_time,
                     nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                     detrend=detrend, average=average, _caller="welch")


def constant_q(data, sample_rate, *, scaling="density", fmin=20.0, fmax=None,
               bins_per_octave=24, hop=None, window="hann",
               integration_time=None):
    """Constant-Q estimate on geometric bins: Pa²/Hz or Pa² per bin.

    One kernel per bin, each Q cycles long, so bin width grows with frequency
    — the resolution hearing has, and the one a soundscape spanning decades
    wants. ``scaling`` means what it means on :func:`welch`, except that the
    bandwidth a density divides by is a different number in every bin.

    Parameters
    ----------
    fmin, fmax : float, optional
        First and last bin centre; 20 Hz to Nyquist by default. Lower
        ``fmax`` to read tone levels near Nyquist, where the negative
        frequency image leaks through the kernel.
    bins_per_octave : int, optional
        Bins per octave, which fixes Q.
    hop : int, optional
        Samples between frame centres; an eighth of the longest kernel by
        default. Frames whose window falls outside the record are dropped per
        bin, so a low bin averages fewer frames than a high one.
    window : str, optional
        The kernel's own taper — its length IS the bin's bandwidth, so
        changing it changes the Q the estimator is named for. Unlike
        :func:`welch` there is no per-scaling default for that reason.
    integration_time : float, optional
        Seconds of record to use, from the start.

    Returns
    -------
    SpectralEstimate
        ``(frequencies, power)`` carrying ``.method='constant_q'``.
    """
    _require_bin_scaling(scaling, "constant_q")
    return _estimate(data, sample_rate, method="constant_q", scaling=scaling,
                     window=window, fmin=fmin, fmax=fmax,
                     integration_time=integration_time,
                     bins_per_octave=bins_per_octave, hop=hop,
                     _caller="constant_q")


def sound_exposure(data, sample_rate, *, band_type="decidecade",
                   num_bands=30, nperseg=None, batch_size=None, fmin=8.9125,
                   fmax=22387, integration_time=None):
    """Sound exposure per standard band, in Pa²·s — ISO 18405's SEL, linear.

    The energy the record delivered in each band: the statistic for an
    *event* rather than for a state. A pile-driving strike or a passage is
    measured by the energy it puts in the water, not by the power it averages
    over however long you happened to record.

    Parameters
    ----------
    band_type : str, optional
        The ladder, one of :data:`BAND_TYPES`. ``'decidecade'`` (IEC 61260-1
        / ISO 18405 base-10) by default, also ``'third_octave'`` and
        ``'octave'`` (base-2) and ``'linear'``.
    num_bands : int, optional
        How many equal-width bands ``band_type='linear'`` cuts; ignored by
        the geometric ladders, which are fixed by the standard.
    nperseg : int, optional
        FFT segment length; the sample rate by default, i.e. 1 Hz bins.
    batch_size : int, optional
        How many samples are read at a time, so a multi-hour record never
        materialises as one segment matrix. A whole number of segments by
        default. Memory, not method.
    fmin, fmax : float, optional
        The band span. The defaults are the base-10 edges of the nominal
        10 Hz – 20 kHz reporting range.
    integration_time : float, optional
        Seconds of record to integrate, from the start.

    Notes
    -----
    At the default 1 Hz bins the lowest decidecade bands hold only a few FFT
    lines each (:data:`_MIN_LINES_PER_BAND` says what that costs and what
    fixes it); the totals stay exact either way.

    This takes no ``window``, ``noverlap``, ``detrend`` or ``average``: a
    band's value is the sum of the bins inside it, which is the band's energy
    only when every bin is counted once and whole. The boxcar window, zero
    overlap and absent detrending that make that true are not choices here, so
    they are not arguments at all — the enforcement is the signature, not a
    runtime refusal. The record is padded to whole segments rather than
    truncated, so no sample is dropped either.

    Returns
    -------
    SpectralEstimate
        ``(centres, exposure)`` carrying ``.bands``, the ``(low, centre,
        high)`` edges. ``.plot()`` draws it as bars in dB re 1 µPa²·s.
    """
    # An exposure is reported per band, so the ladder is not optional here;
    # for the energy in one FFT bin, multiply a welch spectrum by the
    # record's duration.
    _check_band_type(band_type, "sound_exposure")
    return _estimate(data, sample_rate, scaling="exposure",
                     band_type=band_type, num_bands=num_bands,
                     nperseg=nperseg, batch_size=batch_size, fmin=fmin,
                     fmax=fmax, integration_time=integration_time,
                     _caller="sound_exposure")


def _probabilistic_estimate(
        data, sample_rate, *, method="welch", scaling="density",
        seg_duration=1.0, overlap_pct=50, ddB=1.0, lvlmin=0, lvlmax=150,
        window=None, fmin=None, fmax=None, band_type=None,
        integration_time=None, ref=REFERENCE_PRESSURE_WATER,
        _caller="probabilistic_welch", **options):
    """Probability density of spectral levels over time segments.

    Segments the signal(s), estimates a spectrum per segment and histograms
    the dB levels per frequency. Returns a
    :class:`ProbabilisticSpectralEstimate`. 2-D input uses the longer axis as
    time; a **square** input gives that rule nothing to choose on, so the
    first axis is taken as time (one signal per column) and a ``UserWarning``
    is issued — pass a list of 1-D arrays to be explicit.

    ``method='constant_q'`` puts the levels on a geometric frequency axis and
    histograms a different population: each sample here is a *Welch average*
    over a ``seg_duration`` chunk (variance shrinks as the chunk holds more
    Welch segments), where constant-Q histograms single unaveraged frames, so
    its level spread is wider for the same signal. It takes the kernel
    options (``fmin``, ``fmax``, ``bins_per_octave``, ``hop``) in place of the
    segment ones.

    ``fmin`` / ``fmax`` are the frequency range and ``integration_time`` the
    stretch of record the histogram is built from, as on
    :func:`welch`.

    ``overlap_pct`` steps the *time segments*; ``nperseg``/``noverlap`` are the
    Welch parameters *within* a segment. ``nperseg`` is clamped to the segment
    length when ``seg_duration`` is shorter. ``noverlap=None`` (default)
    derives half the (clamped) ``nperseg``; an explicit ``noverlap`` is used
    as given, unless the clamp leaves no room for it (``noverlap >= nperseg``),
    in which case it falls back to half the clamped ``nperseg`` with a
    warning.

    ``pdf`` is a probability *density* (each frequency column integrates to 1
    over the level axis, i.e. ``nansum(col) * binwidth_dB == 1``) and **empty
    bins are NaN, not 0**, so a level never observed stays blank instead of
    being drawn as the lowest colour. Reduce it with the ``nan``-aware
    functions: plain ``pdf.sum()`` returns NaN.

    ``mean_dB`` and ``std_dB`` are taken over *all* segments, so they are not
    clipped by ``lvlmin`` / ``lvlmax`` the way the histogram is; if levels fall
    outside that window the two stop describing the same population.

    Every level in the result is dB re ``ref**2`` (per Hz, since the default
    ``scaling='density'``), and the result carries ``ref`` back so a consumer
    does not have to assume it. A plot axis or a report that hardcodes the
    package default instead is 120 dB out whenever the caller works in µPa.

    **Complex input** is accepted but returns a two-sided spectrum on an
    unsorted frequency axis, as in :func:`welch`; a
    ``UserWarning`` says so.
    """
    # No ``method`` guard: every caller of this private core passes a literal
    # ('constant_q', or nothing). ``scaling`` is guarded because the band
    # route passes 'exposure' through it, and the doors have already refused
    # that spelling for a bin estimate.
    if scaling not in _SPECTRAL_SCALINGS:
        raise ConfigurationError(
            f"{_caller}: unknown scaling {scaling!r}.",
            remediation=f"Use one of {_BIN_SCALINGS}, or sound_exposure() "
                        f"for the energy a record delivered.")
    band_options = {k: options.pop(k) for k in _BAND_OPTIONS if k in options}
    data = _trim_record(data, sample_rate, integration_time, _caller)
    frequency_range = {k: v for k, v in (("fmin", fmin), ("fmax", fmax))
                       if v is not None}
    if method == "constant_q":
        # The constant-Q histogram populates from single unaveraged frames
        # rather than from Welch averages over a seg_duration chunk, so it
        # takes the kernel options and not the segment ones. See its own
        # docstring for what that does to the level spread.
        window = _DEFAULT_WINDOW[method][scaling] if window is None else window
        return _probabilistic_constant_q(
            data, sample_rate, window=window, scaling=scaling, caller=_caller,
            ddB=ddB, lvlmin=lvlmin, lvlmax=lvlmax, ref=ref,
            **frequency_range, **options)
    nperseg = options.get("nperseg", 8192)
    noverlap = options.get("noverlap")
    if isinstance(data, list):
        signals = [np.asarray(s) for s in data]
        for i, s in enumerate(signals):
            if s.ndim != 1:
                raise ConfigurationError(
                    f"{_caller}: data must be 1-D, 2-D, or a list of 1-D arrays; "
                    f"list element {i} has ndim={s.ndim}")
    else:
        data = np.asarray(data)
        if data.ndim == 1:
            signals = [data]
        elif data.ndim == 2:
            if data.shape[0] == data.shape[1]:
                warnings.warn(
                    f"{_caller}: square {data.shape} input — 'the longer axis is "
                    "time' cannot choose, so the first axis is taken as time "
                    "(one signal per column). Pass a list of 1-D arrays, or "
                    "transpose, to be explicit.",
                    UserWarning, stacklevel=2)
            if data.shape[0] < data.shape[1]:
                signals = [data[i, :] for i in range(data.shape[0])]
            else:
                signals = [data[:, i] for i in range(data.shape[1])]
        else:
            raise ConfigurationError(
                f"{_caller}: data must be 1-D, 2-D, or a list of 1-D arrays; "
                f"got ndim={data.ndim}")
    signals = [require_finite_signal(s, _caller) for s in signals]
    sample_rate = require_positive_finite_scalar(
        sample_rate, _caller, "sample_rate", " Hz")
    complex_signal = next((s for s in signals if np.iscomplexobj(s)), None)
    if complex_signal is not None:
        _warn_two_sided(_caller, complex_signal)

    # Samples in one time segment — the interval each histogram sample
    # describes. Not the band method's ``batch_size``, which is how much
    # of the record is read at a time.
    segment_samples = int(seg_duration * sample_rate)
    if segment_samples < 1:
        raise ConfigurationError(
            f"{_caller}: seg_duration ({seg_duration} s) x sample_rate "
            f"({sample_rate} Hz) is {seg_duration * sample_rate:g} samples, "
            "which truncates to an empty time segment; require seg_duration "
            f">= 1/sample_rate ({1.0 / sample_rate:g} s).")
    overlap_samples = int(segment_samples * overlap_pct / 100)
    step = segment_samples - overlap_samples
    if step <= 0:
        raise ConfigurationError(
            f"{_caller}: overlap_pct ({overlap_pct}) too high — chunks never "
            "advance; require overlap_pct < 100.")

    window = (_BAND_WINDOW if band_type is not None else
              _DEFAULT_WINDOW[method][scaling]) if window is None else window
    level_edges = np.arange(lvlmin, lvlmax + ddB, ddB)
    # The Welch segmentation inside each time segment. A banded histogram
    # resolves its own (the band route takes nperseg through, defaulting to
    # one-hertz bins), so this clamp is the bin path's alone.
    nps = min(int(nperseg if nperseg is not None else 8192), segment_samples)
    if noverlap is None:
        nov = nps // 2
    else:
        nov = int(noverlap)
        if nov >= nps:
            warnings.warn(
                f"{_caller}: noverlap={noverlap} does not fit the Welch segment "
                f"length nperseg={nps} (clamped to the {seg_duration}s "
                f"chunk); using {nps // 2} instead.",
                UserWarning, stacklevel=2)
            nov = nps // 2
    psd_list = []
    bands = None                 # set by the band method, whose values sit
    for sig in signals:          # on standard bands rather than on bins
        for i in range(0, len(sig) - segment_samples + 1, step):
            chunk = sig[i: i + segment_samples]
            if band_type is not None:
                # One value per standard band per time segment, so the
                # histogram is over band levels rather than bin levels. The
                # segment is the record the exposure integrates over, which
                # is what makes ``scaling='exposure'`` here a per-segment
                # exposure rather than the whole record's.
                freqs, p, bands = _band_estimate(
                    chunk, sample_rate, scaling=scaling, band_type=band_type,
                    window=window, caller=_caller, **frequency_range,
                    **band_options, **options)
            elif scaling == "exposure":
                # Each sample is the energy of ONE segment: the segment is
                # the record it integrates over, so the duration that turns
                # its band power into an exposure is the segment's, not the
                # whole signal's. The chunk is padded to a whole number of
                # Welch segments first (padding adds no energy), which is
                # what makes the bins sum to that segment's energy.
                n_seg = max(1, -(-len(chunk) // nps))
                padded = (np.pad(chunk, (0, n_seg * nps - len(chunk)))
                          if n_seg * nps != len(chunk) else chunk)
                freqs, p = _sig.welch(padded, sample_rate, window=window,
                                      nperseg=nps, noverlap=0, detrend=False,
                                      scaling="spectrum")
                p = p * (n_seg * nps / float(sample_rate))
            else:
                freqs, p = _sig.welch(chunk, sample_rate, window=window,
                                      nperseg=nps, noverlap=nov,
                                      scaling=scaling)
            psd_list.append(p)

    if len(psd_list) == 0:
        raise ConfigurationError(
            f"{_caller}: no PSD segments computed; seg_duration="
            f"{seg_duration}s vs signal length="
            f"{len(signals[-1])/sample_rate:.2f}s")

    psd_array = np.array(psd_list)
    if method == "welch" and band_type is None:
        # Welch resolves every bin whatever range was asked for, so the range
        # crops the answer; the band and kernel methods were built inside it.
        freqs, psd_array = _crop_to_range(freqs, psd_array, fmin, fmax)
    psd_segments_dB = power_to_dB(psd_array, ref)
    mean_psd = np.mean(psd_segments_dB, axis=0)
    std_psd = np.std(psd_segments_dB, axis=0)

    pdf_matrix = np.zeros((len(level_edges) - 1, len(freqs)))
    for i in range(len(freqs)):
        # density=True over a frequency column with no level inside
        # [lvlmin, lvlmax] normalises an all-zero count by its zero sum
        # (0/0): the NaN column is the intended "nothing observed" answer,
        # so numpy's RuntimeWarning is suppressed here the way
        # probabilistic_constant_q suppresses it, and the all-NaN case gets
        # the one named diagnostic below instead.
        with np.errstate(invalid="ignore", divide="ignore"):
            hist, _ = np.histogram(psd_segments_dB[:, i], bins=level_edges,
                                   density=True)
        pdf_matrix[:, i] = hist
    pdf_matrix[pdf_matrix == 0] = np.nan
    if np.all(np.isnan(pdf_matrix)):
        warnings.warn(
            f"{_caller}: no PSD level falls inside the histogram window "
            f"[lvlmin={lvlmin:g}, lvlmax={lvlmax:g}] dB — the segments span "
            f"{float(np.min(psd_segments_dB)):.1f} to "
            f"{float(np.max(psd_segments_dB)):.1f} dB re ref² — so pdf is "
            f"all-NaN (mean_dB/std_dB still cover every segment). Widen "
            f"lvlmin/lvlmax to cover that span, or pass ref in the data's "
            f"own pressure unit (µPa-scaled samples against the default "
            f"Pa-based ref read 120 dB high).",
            UserWarning, stacklevel=2)

    # ``ref`` and ``scaling`` ride along because the levels mean nothing
    # without them: the same signal read against a Pascal reference sits
    # 120 dB from these numbers, and 'spectrum' levels are per band where
    # 'density' levels are per hertz. A consumer that has to be told them
    # separately is a consumer that can be told the wrong ones.
    return ProbabilisticSpectralEstimate(
        freqs, level_edges, pdf_matrix, mean_dB=mean_psd, std_dB=std_psd,
        binwidth_dB=ddB, seg_duration=seg_duration, ref=float(ref),
        scaling=str(scaling), method=str(method), bands=bands,
        band_type=band_type)


def probabilistic_welch(
        data, sample_rate, *, scaling="density", seg_duration=1.0,
        overlap_pct=50, ddB=1.0, lvlmin=0, lvlmax=150, nperseg=8192,
        noverlap=None, window=None, fmin=None, fmax=None,
        integration_time=None, ref=REFERENCE_PRESSURE_WATER):
    """Probability density of Welch levels, in dB re Pa²/Hz or Pa².

    The McNamara & Buland (2004) soundscape statistic: a Welch estimate per
    time segment, histogrammed per frequency, so a month of recording reads as
    a distribution rather than as one averaged line.

    Parameters
    ----------
    scaling, nperseg, noverlap, window, fmin, fmax, integration_time
        As on :func:`welch`; ``nperseg`` is clamped to the time segment when
        ``seg_duration`` is shorter.
    seg_duration : float, optional
        Seconds per time segment — the interval each sample of the histogram
        describes.
    overlap_pct : float, optional
        How far consecutive time segments overlap, in percent.
    ddB : float, optional
        Height of one level bin, in dB.
    lvlmin, lvlmax : float, optional
        The level window the histogram covers; levels outside it are absent
        from ``pdf`` but still counted in ``mean_dB`` / ``std_dB``.
    ref : float, optional
        Reference pressure the dB levels are stated against (1 µPa in Pa by
        default), carried on the result so a plot cannot assume another.

    Returns
    -------
    ProbabilisticSpectralEstimate
        ``(frequencies, level_edges, pdf)`` carrying ``.mean_dB``,
        ``.std_dB`` and the descriptors; ``.plot()`` draws the histogram.
    """
    _require_bin_scaling(scaling, "probabilistic_welch")
    return _probabilistic_estimate(
        data, sample_rate, scaling=scaling, seg_duration=seg_duration,
        overlap_pct=overlap_pct, ddB=ddB, lvlmin=lvlmin, lvlmax=lvlmax,
        window=window, fmin=fmin, fmax=fmax,
        integration_time=integration_time, ref=ref, nperseg=nperseg,
        noverlap=noverlap, _caller="probabilistic_welch")


def probabilistic_constant_q(
        data, sample_rate, *, scaling="density", fmin=20.0, fmax=None,
        bins_per_octave=24, hop=None, window="hann", ddB=1.0, lvlmin=0,
        lvlmax=150, integration_time=None, ref=REFERENCE_PRESSURE_WATER):
    """Probability density of constant-Q levels, in dB re Pa²/Hz or Pa².

    The constant-Q analogue of a PPSD, and a different population from it:
    each sample here is a *single unaveraged frame*, where the Welch histogram
    samples an average over a whole time segment. Its spread is therefore
    wider, and ``mean_dB`` sits ``10·γ/ln10`` = 2.51 dB below the power mean
    :func:`constant_q` returns from the same record.

    Parameters
    ----------
    scaling, fmin, fmax, bins_per_octave, hop, window, integration_time
        As on :func:`constant_q`. There is no ``seg_duration`` or
        ``overlap_pct`` here: a constant-Q histogram samples single kernel
        frames, whose length is set per bin by Q, not time segments a caller
        cuts.
    ddB : float, optional
        Height of one level bin, in dB.
    lvlmin, lvlmax : float, optional
        The level window the histogram covers; levels outside it are absent
        from ``pdf`` but still counted in ``mean_dB`` / ``std_dB``.
    ref : float, optional
        Reference pressure the dB levels are stated against (1 µPa in Pa by
        default), carried on the result so a plot cannot assume another.

    Returns
    -------
    ProbabilisticSpectralEstimate
        ``(frequencies, level_edges, pdf)`` with ``.seg_duration = None`` —
        each sample is one kernel frame, whose length is per bin rather than
        one duration for the whole estimate.
    """
    _require_bin_scaling(scaling, "probabilistic_constant_q")
    return _probabilistic_estimate(
        data, sample_rate, method="constant_q", scaling=scaling,
        window=window, fmin=fmin, fmax=fmax, bins_per_octave=bins_per_octave,
        hop=hop, ddB=ddB, lvlmin=lvlmin, lvlmax=lvlmax,
        integration_time=integration_time, ref=ref,
        _caller="probabilistic_constant_q")


def probabilistic_sound_exposure(
        data, sample_rate, *, seg_duration=1.0, overlap_pct=50, ddB=1.0,
        lvlmin=0, lvlmax=150, band_type="decidecade", num_bands=30,
        nperseg=None, batch_size=None, fmin=8.9125, fmax=22387,
        integration_time=None, ref=REFERENCE_PRESSURE_WATER):
    """Probability density of sound exposure levels, in dB re Pa²·s per band.

    How a monitoring record reports what a day of piling or a week of
    passages delivered, rather than what its loudest minute did: each sample
    is ONE ``seg_duration`` segment's energy, so the level axis moves with
    that duration: doubling it adds 3.01 dB to the energy each sample
    integrates. ``mean_dB`` moves slightly more than that — measured +3.07 to
    +3.11 dB — because it is a mean of logs, and a longer segment averages
    more Welch segments and so carries less of that bias.

    Parameters
    ----------
    band_type, num_bands, nperseg, batch_size, fmin, fmax, integration_time
        As on :func:`sound_exposure`, which also explains why there is no
        window, overlap, detrending or averaging to set.
    seg_duration : float, optional
        Seconds per time segment — here, the record each exposure integrates,
        which is why the level axis moves with it.
    overlap_pct : float, optional
        How far consecutive time segments overlap, in percent.
    ddB : float, optional
        Height of one level bin, in dB.
    lvlmin, lvlmax : float, optional
        The level window the histogram covers; levels outside it are absent
        from ``pdf`` but still counted in ``mean_dB`` / ``std_dB``.
    ref : float, optional
        Reference pressure the dB levels are stated against (1 µPa in Pa by
        default), carried on the result so a plot cannot assume another.

    Returns
    -------
    ProbabilisticSpectralEstimate
        ``(frequencies, level_edges, pdf)`` carrying ``.bands`` and the
        descriptors; ``.plot()`` draws the histogram.
    """
    _check_band_type(band_type, "probabilistic_sound_exposure")
    return _probabilistic_estimate(
        data, sample_rate, scaling="exposure", band_type=band_type,
        num_bands=num_bands, nperseg=nperseg, batch_size=batch_size,
        seg_duration=seg_duration, overlap_pct=overlap_pct, ddB=ddB,
        lvlmin=lvlmin, lvlmax=lvlmax, fmin=fmin, fmax=fmax,
        integration_time=integration_time, ref=ref,
        _caller="probabilistic_sound_exposure")


CQTResult = namedtuple("CQTResult", "frequencies coefficients")
CQSpectrogramResult = namedtuple("CQSpectrogramResult", "frequencies times power")
#: The constant-Q helpers speak the same two-valued vocabulary the bin
#: estimators do; an energy is not one of them (see
#: :func:`_require_bin_scaling`).
_SCALINGS = _BIN_SCALINGS


# ── shared kernel construction ──────────────────────────────────────────────
def _cq_frequencies(fmin, fmax, bins_per_octave):
    if fmin <= 0 or fmax <= fmin:
        raise ConfigurationError(
            "constant-Q: require 0 < fmin < fmax; got "
            f"fmin={fmin}, fmax={fmax}")
    n = int(np.floor(bins_per_octave * np.log2(fmax / fmin))) + 1
    return fmin * 2.0 ** (np.arange(n) / float(bins_per_octave))


def _cq_quality(bins_per_octave):
    """Constant quality factor Q = 1 / (2**(1/B) - 1)."""
    return 1.0 / (2.0 ** (1.0 / float(bins_per_octave)) - 1.0)


def _cq_kernels(frequencies, Q, fs, window):
    """List of ``(N_k, kernel, density_factor)`` per bin.

    ``kernel = w·exp(-2j·pi·f_k·n/fs) / Σw`` (so ``|X|**2`` is band power, the
    'spectrum' scaling). ``density_factor = (Σw)**2 / (fs·Σw**2)`` converts that
    band power to a one-sided PSD (per Hz), matching ``scipy.signal.welch``
    density scaling; the one-sided factor 2 is applied in
    :func:`_cq_power_frames`.
    """
    kernels = []
    for fk in frequencies:
        Nk = max(1, int(np.ceil(Q * fs / fk)))
        w = get_window(window, Nk, fftbins=True)
        sw = float(np.sum(w))
        sw2 = float(np.sum(w * w))
        n = np.arange(Nk)
        ker = (w * np.exp(-2j * np.pi * fk * n / fs)) / sw
        # |X|**2 captures only the positive-frequency sideband of a real signal
        # (a tone A·cos gives |X|=A/2); _cq_power_frames doubles it to one-sided
        # band power A**2/2. density_factor then divides by the noise-equivalent
        # bandwidth fs·Σw²/(Σw)² to give a one-sided PSD (welch-density).
        density_factor = sw * sw / (fs * sw2)
        kernels.append((Nk, ker.astype(np.complex128), density_factor))
    return kernels


# A bin whose one-sided band power is inflated by more than this much warns.
# The bias is 10*log10(1 + ratio**2) and rises smoothly with f_k/fs: measured
# on a Hann window at B=24 it is under 0.005 dB for every f_k/fs <= 0.4865,
# first crosses 0.01 dB at f_k/fs ~ 0.4875, and reaches 3.01 dB at f_k = fs/2.
# 3.01 dB is the phase average, not a bound: at exactly fs/2 the image lands on
# DC and the reading follows the tone's own phase as 10*log10(4 cos**2 phi),
# so a cosine reads +6.02 dB and a sine is identically zero on the grid.
_CQ_IMAGE_BIAS_WARN_DB = 0.01


def _cq_image_ratio(frequencies, fs, kernels):
    """``|W(2 f_k)| / sum(w)`` per bin — the negative-frequency image leak.

    A real tone at ``f_k`` carries a ``+f_k`` and a ``-f_k`` component. The
    kernel demodulates the first to DC and the second to ``-2 f_k``, where the
    window's own transfer function ``W`` attenuates it; this ratio is what
    survives. It inflates the one-sided band power ``2|X|**2`` by
    ``1 + ratio**2`` averaged over frame phase, so the estimators read a
    coherent tone high by that factor. The ratio is negligible while
    ``2 f_k`` sits well outside the window's main lobe and rises to 1 at
    ``f_k = fs/2``, where the image lands on DC. That last point is the one
    place the frame average does not apply: an image on DC keeps the same
    phase in every frame, so the reading there is ``10*log10(4 cos**2 phi)``
    in the tone's own phase (+6.02 dB for a cosine) rather than the 3.01 dB
    the ``1 + ratio**2`` average gives.

    Broadband noise is untouched: its band power is ``sigma**2 * sum(w**2) /
    sum(w)**2`` at every bin, independent of ``f_k``, so the correct remedy is
    to report the affected bins rather than divide the bias out.
    """
    ratios = np.empty(len(kernels))
    for i, (Nk, ker, _) in enumerate(kernels):
        n = np.arange(Nk)
        image = np.sum(ker * np.exp(-2j * np.pi * frequencies[i] * n / fs))
        ratios[i] = abs(image)
    return ratios


def _cq_frame(x, center, kernels):
    """``(coeffs, valid)``: complex constant-Q band amplitudes with each bin's
    window centred on sample ``center`` (zero-padded at the signal edges), and a
    per-bin bool — ``True`` where the window lay fully inside the signal."""
    nx = x.size
    coeffs = np.empty(len(kernels), dtype=np.complex128)
    valid = np.empty(len(kernels), dtype=bool)
    for i, (Nk, ker, _) in enumerate(kernels):
        start = center - Nk // 2
        stop = start + Nk
        inside = start >= 0 and stop <= nx
        valid[i] = inside
        if inside:
            coeffs[i] = np.sum(x[start:stop] * ker)
        else:
            a, b = max(start, 0), min(stop, nx)
            seg = np.zeros(Nk, dtype=float)
            seg[a - start: b - start] = x[a:b]
            coeffs[i] = np.sum(seg * ker)
    return coeffs, valid


def _cq_power_frames(x, fs, kernels, hop, scaling):
    """Per-frame constant-Q power ``(n_bins, n_frames)`` and a matching per-bin
    validity mask (``True`` where the window lay fully inside the signal).

    There is no exposure here: an energy is :func:`sound_exposure`'s, whose
    bins are orthogonal so that a band sum IS an energy. Constant-Q kernels
    overlap by construction, so no factor turns these into one.
    """
    centers = np.arange(0, x.size, hop)
    dfac = np.array([k[2] for k in kernels])
    power = np.empty((len(kernels), centers.size), dtype=float)
    valid = np.empty((len(kernels), centers.size), dtype=bool)
    for j, c in enumerate(centers):
        coeffs, ok = _cq_frame(x, int(c), kernels)
        p = 2.0 * np.abs(coeffs) ** 2          # one-sided band power (tone -> A²/2)
        power[:, j] = p * dfac if scaling == "density" else p
        valid[:, j] = ok
    return centers / fs, power, valid


def _check_scaling(scaling, caller):
    if scaling not in _SCALINGS:
        raise ConfigurationError(
            f"{caller}: scaling must be one of {_SCALINGS}; got {scaling!r}")


def _resolve_hop(hop, kernels, n_samples, caller):
    n_lowest = kernels[0][0]
    if hop is None:
        # 1/8 of the longest window (8x overlap of the lowest bin), but capped
        # so a short signal still yields enough frame centres to average.
        hop = max(1, min(n_lowest // 8, max(1, n_samples // 8)))
    hop = int(hop)
    if hop < 1:
        raise ConfigurationError(f"{caller}: hop must be >= 1; got {hop}")
    return hop


def _cq_setup(data, sample_rate, fmin, fmax, bins_per_octave, window, caller,
              drops_short_bins):
    """Validate, build the kernel bank and warn about bins that never fit;
    ``drops_short_bins`` says whether ``caller`` drops such a bin (the
    averaging estimators) or evaluates it on a zero-padded window."""
    data = np.asarray(data)
    if np.iscomplexobj(data):
        raise ConfigurationError(
            f"{caller}: data must be a real signal; got complex input "
            "(a complex array would be silently real-cast).")
    x = np.asarray(data, dtype=float)
    if x.ndim != 1:
        raise ConfigurationError(
            f"{caller}: data must be 1-D; got shape {x.shape}")
    require_finite_signal(x, caller)
    fs = require_positive_finite_scalar(sample_rate, caller,
                                        "sample_rate", " Hz")
    if fmax is None:
        fmax = fs / 2.0
    # The analyser side of the Nyquist split: fmax == fs/2 is admitted
    # because the default fmax=None resolves to exactly that.
    require_at_most_nyquist(fmax, fs, caller, "fmax",
                            "the bin has no signal to analyse")
    freqs = _cq_frequencies(fmin, fmax, bins_per_octave)
    Q = _cq_quality(bins_per_octave)
    kernels = _cq_kernels(freqs, Q, fs, window)
    n_lowest = kernels[0][0]
    if n_lowest > x.size:
        fate = ("dropped from the average" if drops_short_bins else
                "evaluated on a zero-padded window and read low there")
        # Every public constant-Q estimator funnels through this setup helper,
        # so the frame the user wrote is two deep here, not one: a
        # hand-counted ``stacklevel=2`` named this module's own call line for
        # all four of them (measured). ``skip_file_prefixes`` walks out to the
        # first frame outside the package instead, which is the caller's own
        # line whichever estimator they chose.
        warnings.warn(
            f"{caller}: lowest bin needs {n_lowest} samples (Q·fs/fmin) but the "
            f"signal has {x.size}; low-frequency bins never fit a full window "
            f"and are {fate}. Raise fmin or lengthen the signal.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
    bias_dB = 10.0 * np.log10(1.0 + _cq_image_ratio(freqs, fs, kernels) ** 2)
    hot = np.flatnonzero(bias_dB > _CQ_IMAGE_BIAS_WARN_DB)
    if hot.size:
        # ``fmax=None`` resolves to fs/2, which puts the top bin in this
        # region on every default call, so this is a statement about the
        # caller's ``fmax`` and has to name the caller's line. Same frame walk
        # as the short-signal warning above and for the same reason: measured,
        # ``stacklevel=2`` here names four *different* lines of this module,
        # one per public estimator, and never the caller's — which also
        # collapses every call site onto one dedup key.
        warnings.warn(
            f"{caller}: {hot.size} bin(s) above {freqs[hot[0]]:.4g} Hz sit "
            f"close enough to Nyquist that the tone's negative-frequency "
            f"image leaks through the analysis window: a coherent tone in "
            f"those bins reads high by up to {bias_dB[hot].max():.2f} dB "
            f"averaged over frame phase (a bin at exactly fs/2 follows the "
            f"tone's own phase instead: +6.02 dB for a cosine) "
            f"(highest bin {freqs[-1]:.4g} Hz, f/fs = {freqs[-1] / fs:.4f}). "
            f"Broadband noise in the same bins is unaffected. Lower fmax to "
            f"read tone levels there.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
    return x, fs, freqs, kernels


# ── public API ──────────────────────────────────────────────────────────────
def constant_q_transform(data, sample_rate, *, fmin=20.0, fmax=None,
                         bins_per_octave=24, window="hann"):
    """Complex constant-Q spectrum of one frame centred on the signal.

    Returns a :class:`CQTResult` ``(frequencies, coefficients)`` — the centre
    frequencies (Hz) and complex band amplitudes (``abs`` is the magnitude
    spectrum). Best for short transients; for longer signals use
    :func:`constant_q_spectrogram`, or :func:`constant_q`.
    """
    x, fs, freqs, kernels = _cq_setup(
        data, sample_rate, fmin, fmax, bins_per_octave, window,
        "constant_q_transform", drops_short_bins=False)
    coeffs, _ = _cq_frame(x, x.size // 2, kernels)
    return CQTResult(freqs, coeffs)


def constant_q_spectrogram(data, sample_rate, *, fmin=20.0, fmax=None,
                           bins_per_octave=24, hop=None, window="hann",
                           scaling="spectrum"):
    """Constant-Q spectrogram: constant-Q power over time.

    Slides the constant-Q analysis at ``hop`` samples (default: 1/8 of the
    longest window, i.e. of the lowest-frequency bin). ``scaling='spectrum'``
    (default) returns band power ``|X_cq|**2``; ``scaling='density'`` returns a
    one-sided PSD per Hz. Returns a :class:`CQSpectrogramResult`
    ``(frequencies, times, power)`` with ``power`` shaped ``(n_freqs,
    n_frames)`` — every frame is kept, so edge frames carry visible edge
    effects.
    """
    _check_scaling(scaling, "constant_q_spectrogram")
    x, fs, freqs, kernels = _cq_setup(
        data, sample_rate, fmin, fmax, bins_per_octave, window,
        "constant_q_spectrogram", drops_short_bins=False)
    hop = _resolve_hop(hop, kernels, x.size, "constant_q_spectrogram")
    times, power, _valid = _cq_power_frames(x, fs, kernels, hop, scaling)
    return CQSpectrogramResult(freqs, times, power)


def _constant_q_estimate(data, sample_rate, *, window, scaling, caller,
                         fmin=20.0, fmax=None, bins_per_octave=24, hop=None):
    """Time-averaged constant-Q power per geometric bin, as ``(freqs, power)``.

    What :func:`constant_q` returns once it has wrapped this in a
    :class:`SpectralEstimate`. The average is taken, *per bin*, only over
    frames whose window lay fully inside the signal (zero-padded edge frames
    are excluded); a bin with no such frame is ``NaN``.
    """
    _check_scaling(scaling, caller)
    x, fs, freqs, kernels = _cq_setup(
        data, sample_rate, fmin, fmax, bins_per_octave, window,
        caller, drops_short_bins=True)
    hop = _resolve_hop(hop, kernels, x.size, caller)
    _, power, valid = _cq_power_frames(x, fs, kernels, hop, scaling)
    avg = np.full(freqs.size, np.nan)
    counts = valid.sum(axis=1)
    good = counts > 0
    avg[good] = np.array([power[i, valid[i]].mean() for i in np.flatnonzero(good)])
    return freqs, avg


def _probabilistic_constant_q(data, sample_rate, *, window, scaling, caller,
                              fmin=20.0, fmax=None, bins_per_octave=24,
                              hop=None, ddB=1.0, lvlmin=0, lvlmax=150,
                              ref=REFERENCE_PRESSURE_WATER):
    """Probability density of constant-Q power levels over time.

    Histograms the per-bin dB levels of the constant-Q frames, the constant-Q
    analogue of :func:`probabilistic_welch` — note the two histogram
    different populations: each sample here is a *single unaveraged frame*,
    whereas ``probabilistic_welch`` histograms averages over ``seg_duration``
    chunks,
    so the level spread here is wider for the same signal, and ``mean_dB``
    — the mean of those dB levels — sits ``10*gamma/ln(10)`` = 2.51 dB below
    the power mean :func:`constant_q` returns from the same record
    (measured 2.507 +/- 0.004 dB over four seeds, 60 s of white noise at
    ``bins_per_octave=24``). That figure is the two-degrees-of-freedom case:
    a bin essentially *at* Nyquist loses its quadrature component, so
    ``|X|**2`` tends toward one dof and the offset climbs toward
    ``10*(gamma+ln2)/ln(10)`` = 5.52 dB — measured 2.91 dB at
    ``f_k/fs = 0.4995`` against 2.48-2.53 dB across the rest of the band. This
    does not contradict "broadband noise is not affected" above: the band
    *power* there is unchanged, and it is the shape of its distribution, hence
    the mean of the logs, that moves. ``probabilistic_welch`` carries the same bias wherever
    its own ``nperseg`` is clamped to the ``seg_duration`` chunk, which is one
    look too. Compare a target curve against ``constant_q``; ``mean_dB``
    is the centre of the histogram. Only frames whose
    window lay fully inside the signal contribute (per bin). Returns a
    :class:`ProbabilisticSpectralEstimate` — ``(frequencies, level_edges,
    pdf)`` carrying ``.mean_dB``, ``.std_dB``, ``.ref``, ``.scaling`` and
    ``.method='constant_q'``;
    ``pdf`` is shaped ``(n_levels, n_freqs)`` and density-normalised per
    frequency column (empty bins are ``NaN``). With ``scaling='density'`` the
    levels are PSD levels (dB re ref²/Hz) rather than band-power levels
    (dB re ref²). The ``ref`` those levels are stated against comes back in the
    result rather than having to be assumed downstream — a consumer that
    hardcodes the package default is 120 dB out for a caller working in µPa.
    """
    _check_scaling(scaling, caller)
    x, fs, freqs, kernels = _cq_setup(
        data, sample_rate, fmin, fmax, bins_per_octave, window,
        caller, drops_short_bins=True)
    hop = _resolve_hop(hop, kernels, x.size, caller)
    _, power, valid = _cq_power_frames(x, fs, kernels, hop, scaling)
    level_edges = np.arange(lvlmin, lvlmax + ddB, ddB)
    pdf = np.zeros((level_edges.size - 1, freqs.size))
    mean_dB = np.full(freqs.size, np.nan)
    std_dB = np.full(freqs.size, np.nan)
    with np.errstate(divide="ignore"):
        levels_dB = power_to_dB(power, ref)
    for i in range(freqs.size):
        vals = levels_dB[i, valid[i]]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            hist, _ = np.histogram(vals, bins=level_edges, density=True)
        pdf[:, i] = hist
        mean_dB[i] = vals.mean()
        std_dB[i] = vals.std()
    pdf[pdf == 0] = np.nan
    # The reference, the scaling and the method are part of what the levels
    # MEAN, so they travel with them rather than being restated by every
    # consumer. ``seg_duration`` is None: each sample is one unaveraged frame,
    # and a frame here is per bin rather than one duration for the estimate.
    return ProbabilisticSpectralEstimate(
        freqs, level_edges, pdf, mean_dB=mean_dB, std_dB=std_dB,
        binwidth_dB=ddB, seg_duration=None, ref=float(ref),
        scaling=str(scaling), method="constant_q")


# IEC 61260-1 anchors both band systems at 1 kHz; the step is the band width
# in octaves.
_SEL_REF_FREQ = 1000.0
_SEL_OCTAVE_STEP = {"octave": 1.0, "third_octave": 1.0 / 3.0}


def _sel_adjust_fmin_fmax(fmin, fmax, band_type, sample_rate):
    """Snap configured band edges onto the IEC 61260-1 ladder anchored at 1 kHz.

    The anchor is what makes a band table comparable with anyone else's:
    1 kHz is a standard centre in both the base-2 and the base-10 system
    (Pierce, *Acoustics* — "1, 10, 100, 1000, 10,000 Hz … are also standard
    1/3-octave-band f_o's"). Snapping to a ladder built from ``fmin`` instead
    made the grid move with the caller's request — ``fmin=8.9125`` and
    ``fmin=10.0`` gave disjoint, interleaved centres, and the nearest centre
    to 1 kHz was 1024 Hz (+2.4 %) or 912.3 Hz (-8.8 %) depending on it.

    the base-2 band types are the names, so it keeps the base-2 ladder;
    :func:`~uacpy.acoustic_signal.estimate.decidecade_bands` owns base-10 and
    already anchors at 1 kHz (``bands.py`` ``_REF_FREQ``).
    """
    step = _SEL_OCTAVE_STEP.get(band_type)
    if step is None:
        return fmin, fmax

    def centre(k):
        return _SEL_REF_FREQ * 2.0 ** (k * step)

    # fmin -> the centre of the band containing it. Rounding on the centre
    # grid, rather than flooring onto the half-band grid, is what stops the
    # ladder shifting by half a step with the parity of the floor.
    fmin = centre(round(math.log2(fmin / _SEL_REF_FREQ) / step))
    # fmax -> the upper edge of the highest band needed, clamped to the
    # highest band whose whole support is sampled. Clamping the *edge* rather
    # than the centre keeps a band that is fully below Nyquist.
    k_hi = math.ceil(math.log2(fmax / _SEL_REF_FREQ) / step - 0.5)
    k_nyq = math.floor(math.log2(sample_rate / 2.0 / _SEL_REF_FREQ) / step - 0.5)
    fmax = centre(min(k_hi, k_nyq)) * 2.0 ** (step / 2.0)
    return fmin, fmax


def _sel_bands(fmin, fmax, band_type, num_bands, sample_rate,
               caller="sound_exposure"):
    """Generate ``(low, center, high)`` frequency bands."""
    if fmin <= 0 or fmax <= fmin:
        raise ConfigurationError(
            f"{caller}: require fmin > 0 and fmax > fmin; got fmin={fmin}, "
            f"fmax={fmax}")
    if band_type == "decidecade":
        # The base-10 ladder of IEC 61260-1 / ISO 18405, shared with
        # :func:`uacpy.acoustic_signal.decidecade_bands` so a band level and a
        # band exposure are stated over the very same edges.
        #
        # A band is kept when its CENTRE is in the requested range and its
        # whole width is sampled. Selecting on the centre rather than on the
        # edges is what makes the range robust to how it was written: the
        # nominal edges are irrational (10**4.35 = 22387.21...), so the
        # rounded 22387 anyone types drops the 20 kHz band by a hundredth of
        # a hertz, while its centre is 387 Hz clear of the limit. A band cut
        # by Nyquist is excluded outright: it would report the part that was
        # sampled as if it were the band.
        from uacpy.acoustic_signal.estimate import decidecade_bands
        lower, centres, upper = decidecade_bands(fmin, fmax)
        keep = ((centres >= fmin) & (centres <= fmax)
                & (upper <= sample_rate / 2.0))
        return [(float(lo), float(c), float(hi))
                for lo, c, hi in zip(lower[keep], centres[keep], upper[keep])]
    if band_type in ("octave", "third_octave"):
        fmin, fmax = _sel_adjust_fmin_fmax(fmin, fmax, band_type, sample_rate)
    bands = []
    if band_type in ("octave", "third_octave"):
        # One loop for both: the centres advance by a whole octave or a third
        # of one, and each band's edges sit half that step either side of its
        # centre. ``math.pow``, not ``math.sqrt``: the merged form reproduces
        # the two former branches BIT-FOR-BIT only because ``(1/3)/2 == 1/6``
        # exactly in binary floating point (halving is exact), so
        # ``math.pow(2, step/2)`` returns the identical double the
        # third-octave branch's ``math.pow(2, 1/6)`` did, and the identical
        # ``math.sqrt(2)`` for step 1.
        step = _SEL_OCTAVE_STEP[band_type]
        base = math.pow(2, step / 2)
        factor = math.pow(2, step)
        f_center = fmin
        while f_center < fmax:
            bands.append((f_center / base, f_center, f_center * base))
            f_center *= factor
        if bands and bands[-1][2] > fmax:
            bands[-1] = (bands[-1][0], bands[-1][1], fmax)
    elif band_type == "linear":
        if num_bands <= 0:
            raise ConfigurationError(
                f"{caller}: num_bands must be positive for linear bands; got "
            f"{num_bands}")
        bw = (fmax - fmin) / num_bands
        f_low = fmin
        for _ in range(num_bands):
            f_high = f_low + bw
            bands.append((f_low, (f_low + f_high) / 2, f_high))
            f_low = f_high
        # Accumulating bw num_bands times drifts the final edge a few ULPs off
        # fmax; pinning it keeps a bin sitting exactly on fmax (the Nyquist bin
        # of a full-span request) inside the top band.
        bands[-1] = (bands[-1][0], bands[-1][1], fmax)
    else:
        raise ConfigurationError(
            f"{caller}: unknown band_type={band_type!r}; valid: "
            "'decidecade' (IEC 61260-1 base-10, the default), 'third_octave' "
            "and 'octave' (base-2 ladders), 'linear'")
    return bands


def _band_estimate(data, sample_rate, *, scaling="exposure",
                   fmin=8.9125, fmax=22387, band_type="decidecade",
                   num_bands=30, nperseg=None, window=_BAND_WINDOW,
                   batch_size=None, caller="sound_exposure", **welch_kwargs):
    """Standard-band estimate: one value per IEC 61260-1 / ISO 18405 band.

    Returns ``(centres, values, bands)``, reference-free and linear.
    ``scaling='exposure'`` gives the sound exposure in each band (Pa²·s),
    ``'spectrum'`` that divided by the record duration (Pa², the band power)
    and ``'density'`` the band power per hertz of the band's own width
    (Pa²/Hz). ``bands`` are the ``(low, centre, high)`` edges those values sit
    on.

    Uses a rectangular (boxcar) window with ``noverlap=0`` and no detrending so
    the summed PSD equals the band exposure exactly (Parseval). A tapering
    window would corrupt that identity, which is why the public door offers
    none: :func:`sound_exposure` has no ``window``, ``noverlap``, ``detrend``
    or ``average`` argument, and the bin estimators refuse
    ``scaling='exposure'`` outright.

    The default ``fmin``/``fmax`` are the base-10 band edges of the nominal
    10 Hz — 20 kHz reporting range: ``10^0.95 = 8.9125`` Hz is the lower edge of
    the 10 Hz band and ``10^4.35 = 22387`` Hz the upper edge of the 10^4.3 =
    19953 Hz ("20 kHz") band. For ``band_type='octave'``/``'third_octave'``
    they are then snapped to that base-2 grid and to Nyquist
    (:func:`_sel_adjust_fmin_fmax`); ``'linear'`` uses them as given.
    ``nperseg`` is the FFT segment length, the same quantity Welch calls
    ``nperseg``, and defaults to ``sample_rate`` — 1 Hz wide bins.

    **Band selectivity:** the *total* exposure (sum over all bands) is
    Parseval-exact **over the covered band** — it equals ``sum(data**2) /
    sample_rate`` restricted to the FFT bins inside ``[bands[0][0],
    bands[-1][2]]``, each bin counted in exactly one band. Bins outside that
    span are dropped, which always includes DC (band edges must be > 0) and
    includes Nyquist unless a bin falls exactly on the top edge. So the total
    is *not* the whole signal's exposure whenever the bands do not span the
    full spectrum: for white noise sampled at 2 kHz the default third-octave
    request snaps to 8.8-891 Hz (the highest whole band under Nyquist) and
    returns about 88 % of it, and even a DC-to-Nyquist ``'linear'`` request
    falls short by the DC bin alone. Compare a total against the band span it
    covers, not against ``sum(data**2) / sample_rate``.

    Each band is a plain sum of rectangular FFT bins, not an IEC 61260
    fractional-octave filter. A tone that does not fall on a bin
    centre leaks into each adjacent band at a floor of roughly -33 dB relative
    to its own band (IEC 61260 class-1 filters provide 60-75 dB of stopband
    rejection). Band levels of broadband signals are accurate; strong tonals
    bleed into neighbouring bands at about that level.

    **Batch alignment:** ``batch_size`` is how many samples are read at a
    time, so a multi-hour record never materialises as one segment matrix.
    Unset, it is a whole number of segments near
    :data:`_TARGET_BATCH_SAMPLES`, so the default never pays the penalty
    below. It is a memory knob, not an estimator one — but only while each
    batch holds whole segments. Each batch is zero-padded up to a whole number of
    ``nperseg`` segments; the padding adds no energy — the total stays
    exact — but a tone truncated by the padded partial segment leaks part of
    its power into neighbouring bands. With ``batch_size`` not a multiple of
    ``nperseg`` *every* batch ends in such a partial segment and
    neighbouring-band levels wobble by a few tenths of a dB (a warning is
    issued when the signal is actually batched); with a multiple, only the
    signal's final partial batch does.
    """
    data = require_finite_signal(data, caller)
    if np.iscomplexobj(data):
        raise ConfigurationError(
            f"{caller}: data must be real (got complex input); band "
            "exposure is "
            "defined for a real pressure time series. Demodulate to a real "
            "signal first.")
    sample_rate = require_positive_finite_scalar(
        sample_rate, caller, "sample_rate", " Hz")
    bands = _sel_bands(fmin, fmax, band_type, num_bands, sample_rate,
                       caller)
    if len(data) <= 0:
        raise ConfigurationError(
            f"{caller}: no samples to integrate. Provide a non-empty "
            f"signal; got {len(data)} sample(s) at "
            f"sample_rate={sample_rate:g} Hz.")
    if nperseg is None:
        # One-hertz bins: the resolution a band ladder is read at, and the
        # one that makes even the narrowest low band hold whole bins.
        nperseg = sample_rate
    nperseg = int(nperseg)
    if batch_size is None:
        # Whole segments, about a quarter-million samples of them: a batch
        # that ends mid-segment pays the zero-padding penalty on EVERY batch,
        # and a fixed sample count would never divide the default nperseg
        # (the sample rate). Written in segments, the default never does.
        batch_size = max(1, round(_TARGET_BATCH_SAMPLES / nperseg)) * nperseg
    batch_size = min(int(batch_size), max(1, len(data)))

    if not bands:
        raise ConfigurationError(
            f"{caller}: no {band_type} band fits below Nyquist "
            f"({sample_rate / 2:g} Hz) with the requested fmin/fmax — the "
            f"snapped lower edge sits above the Nyquist-clamped upper edge; "
            f"got fmin={fmin!r}, fmax={fmax!r}.",
            remediation="Raise sample_rate, or pass a lower fmin explicitly.")
    if len(data) > batch_size and batch_size % nperseg:
        warnings.warn(
            f"{caller}: batch_size ({batch_size}) is not a multiple of "
            f"nperseg ({nperseg}), so every batch ends in a zero-padded "
            "partial segment; tonal energy truncated there leaks into "
            "neighbouring bands (band wobble of a few tenths of a dB). Pass "
            "a batch_size that is a multiple of nperseg.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
    f = np.fft.rfftfreq(nperseg, d=1 / sample_rate)
    edges = np.array([b[0] for b in bands] + [bands[-1][2]])
    # digitize is 1-based, so -1 gives the 0-based band index; bins below the
    # first edge become -1 and bins above the last become len(bands), and
    # neither matches any k in range(len(bands)) — they drop out of the sum.
    # Interior edges are half-open [lo, hi), but the top edge is closed: a
    # bin exactly on edges[-1] belongs to the last band, not outside it.
    bin_band = np.digitize(f, edges) - 1
    bin_band[f == edges[-1]] = len(bands) - 1
    band_bins = [np.where(bin_band == k)[0] for k in range(len(bands))]
    # A band holding no bin sums to exactly 0 Pa^2*s, which reads as a measured
    # silence rather than as "not measured" — 'linear' bands are used as given,
    # so a fmax above Nyquist produces whole empty bands (the octave ladders
    # clamp to Nyquist instead, but a band narrower than the bin spacing is
    # empty on any ladder).
    empty = [k for k, idx in enumerate(band_bins) if idx.size == 0]
    if empty:
        named = ', '.join(f"{bands[k][0]:.4g}-{bands[k][2]:.4g}" for k in empty[:4])
        if len(empty) > 4:
            named += f", ... ({len(empty)} in all)"
        warnings.warn(
            f"{caller}: {len(empty)} of {len(bands)} bands contain no FFT "
            f"bin and "
            f"are returned as exactly 0 Pa^2*s, which is not a measurement: "
            f"{named} Hz. A band above Nyquist ({sample_rate / 2:g} Hz) has no "
            f"data at all; a band narrower than the "
            f"{sample_rate / nperseg:g} Hz bin spacing falls between bins. "
            f"Lower fmax, or raise nperseg.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
    out = np.zeros(len(bands))

    for i in range(0, len(data), batch_size):
        chunk = data[i: min(i + batch_size, len(data))]
        # One estimator, not two: the per-bin exposure of this batch is what
        # the Welch route returns under ``scaling='exposure'`` — boxcar, no
        # overlap, no detrending, and the batch padded to whole segments so
        # no sample is dropped — and a band's exposure is the bins it covers.
        # Summing (rather than averaging) over segments is what makes it an
        # energy, and that sum is already inside that route: it multiplies
        # the mean by the padded duration.
        #
        # The batching is memory, not method: a multi-hour record never
        # materialises as one segment matrix. Each batch's own exposure adds,
        # because energy does.
        per_bin = _estimate(
            chunk, sample_rate, method="welch", scaling="exposure",
            window=window, nperseg=nperseg, _caller=caller, **welch_kwargs).power
        for k, idx in enumerate(band_bins):
            out[k] += per_bin[idx].sum()

    centres = np.array([b[1] for b in bands], dtype=float)
    edges = np.asarray(bands, dtype=float)
    if scaling == "exposure":
        values = out
    else:
        # Band power is the exposure spread back over the record it was
        # integrated from; a density then divides by the band's own width,
        # which is what makes decidecade levels comparable across bands that
        # get wider with frequency.
        values = out / (len(data) / sample_rate)
        if scaling == "density":
            values = values / (edges[:, 2] - edges[:, 0])
    return centres, values, edges


# ──────────────────────────────────────────────────────────────────────
# Standard band ladders
#
# The IEC 61260-1 / ISO 18405 base-10 bands,
# and a PSD integrated onto them. ``sound_exposure`` is written on these edges.
# ──────────────────────────────────────────────────────────────────────

_REF_FREQ = 1000.0      # reference frequency [Hz]


def decidecade_bands(f_low, f_high):
    """Decidecade band ``(lower, center, upper)`` edges spanning ``[f_low, f_high]``.

    Centre frequencies are ``1000 * 10^(n/10)`` (IEC 61260-1 base-10); band edges
    are ``center * 10^(±1/20)``. Returns three arrays of equal length covering
    every band that overlaps the requested range.
    """
    if f_low <= 0 or f_high <= f_low:
        raise ConfigurationError(
            "decidecade_bands: need 0 < f_low < f_high; "
            f"got f_low={f_low!r}, f_high={f_high!r}")
    n_lo = int(np.floor(10.0 * np.log10(f_low / _REF_FREQ)))
    n_hi = int(np.ceil(10.0 * np.log10(f_high / _REF_FREQ)))
    centers = _REF_FREQ * 10.0 ** (np.arange(n_lo, n_hi + 1) / 10.0)
    lower = centers * 10.0 ** (-1.0 / 20.0)
    upper = centers * 10.0 ** (1.0 / 20.0)
    keep = (upper >= f_low) & (lower <= f_high)
    return lower[keep], centers[keep], upper[keep]


def decidecade_band_levels(psd, frequencies, ref=REFERENCE_PRESSURE_WATER):
    """Integrate a one-sided PSD into decidecade band levels.

    Parameters
    ----------
    psd : array_like
        One-sided power spectral density [pressure²/Hz, e.g. Pa²/Hz].
    frequencies : array_like
        Frequencies [Hz] matching ``psd`` (monotonic, > 0).
    ref : float
        Reference pressure (default ``1e-6`` Pa = 1 µPa, the water standard).

    Returns
    -------
    centers, levels : numpy.ndarray
        Band centre frequencies [Hz] and band levels [dB re ``ref²``]; bands with
        no spectral support are ``nan``.

    Notes
    -----
    Each band is integrated over its full support ``[lo, hi]``: the band edges
    are spliced into the in-band grid points and the PSD is interpolated onto
    them, so the edge intervals carry their true width. A band reaching past
    the ends of ``frequencies`` is returned as ``nan`` — a partial integral is
    not a band level.

    **The first and last band are normally ``nan``, and that is structural.**
    The band set comes from :func:`decidecade_bands`, which keeps every band
    *overlapping* ``[min(frequencies), max(frequencies)]``, so the band holding
    the first frequency starts below it and the band holding the last ends
    above it unless both land exactly on decidecade band edges — which no
    ``rfftfreq`` grid does. Those two ``nan`` levels are the diagnostic; they
    are not warned about, because a warning that fires on every well-formed
    call cannot distinguish "the grid is too short" from "the function was
    called". The returned arrays stay parallel to :func:`decidecade_bands` on
    the same span, so a caller masks with ``np.isfinite(levels)``.

    A band holding fewer than two interior grid points rests almost entirely on
    its interpolated edges; a :class:`UserWarning` names how many such bands
    the grid produced. That one *is* a warning: it qualifies levels that came
    back finite.
    """
    psd = np.asarray(psd, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    ref = require_positive_finite_scalar(
        ref, "decidecade_band_levels", "ref", " Pa")
    if np.any(psd < 0):
        raise ConfigurationError(
            "decidecade_band_levels: psd contains negative values; a power "
            "spectral density is non-negative, so this input is a dB level "
            "or a signed spectrum, whose band integral is not a band level. "
            f"Got {int(np.count_nonzero(psd < 0))} negative value(s), "
            f"minimum {psd.min():g}.")
    if frequencies.shape != psd.shape:
        raise ConfigurationError(
            f"decidecade_band_levels: psd shape {psd.shape} and frequencies "
            f"shape {frequencies.shape} differ")
    if frequencies.size > 1 and np.any(np.diff(frequencies) <= 0):
        raise ConfigurationError(
            "decidecade_band_levels: frequencies must be strictly increasing. "
            "A two-sided np.fft.fftfreq grid is not — take the one-sided "
            "np.fft.rfftfreq half (and the matching half of the PSD). Got "
            f"{int(np.count_nonzero(np.diff(frequencies) <= 0))} "
            f"non-increasing step(s), first at index "
            f"{int(np.argmax(np.diff(frequencies) <= 0))}.")
    require_increasing_axis(frequencies, "decidecade_band_levels: frequencies")
    if frequencies.size < 2:
        # A one-point axis has f_min == f_max, which reached decidecade_bands
        # as "need 0 < f_low < f_high" — an error naming two arguments this
        # caller never passed.
        raise ConfigurationError(
            f"decidecade_band_levels: frequencies needs at least 2 samples to "
            f"span a band; got {frequencies.size}. A single point has no "
            f"width, so no decidecade band covers it.")
    pos = frequencies > 0
    # Tested after the DC bin is dropped: a two-sample rfftfreq grid passes
    # the size guard above yet leaves a single positive frequency, which has
    # no width for a band and would reach decidecade_bands as f_low == f_high
    # — an error naming two arguments this caller never passed.
    if int(np.count_nonzero(pos)) < 2:
        raise ConfigurationError(
            f"decidecade_band_levels: frequencies "
            f"[{frequencies[0]:g}, {frequencies[-1]:g}] Hz holds "
            f"{int(np.count_nonzero(pos))} positive sample(s) once the DC "
            f"bin is dropped, so the grid spans no decidecade band. Use a "
            f"longer FFT so the one-sided grid holds at least two positive "
            f"frequencies.")
    f_min = frequencies[pos].min()
    f_max = frequencies[pos].max()
    lower, centers, upper = decidecade_bands(f_min, f_max)
    levels = np.full(centers.size, np.nan)
    n_coarse = 0
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        # Integrate over [lo, hi] itself: splice the edges into the in-band
        # grid points and interpolate the PSD onto them.
        #
        # A band the supplied grid does not fully cover is left ``nan``.
        # Clipping the nodes to the support instead returned the integral over
        # the covered part, which is not that band's level — measured 3.8 dB
        # (first band) and 3.2 dB (last) off their own trend on a flat PSD,
        # and 5.5 dB low on the realistic psd -> band_levels path.
        if lo < f_min * (1.0 - 1e-12) or hi > f_max * (1.0 + 1e-12):
            continue
        interior = frequencies[(frequencies > lo) & (frequencies < hi)]
        # `nodes` always holds at least the two spliced edges: hi/lo is the
        # constant 10**0.1 for every band, so lo < hi strictly at every
        # positive centre and np.unique keeps both.
        nodes = np.unique(np.concatenate(([lo], interior, [hi])))
        if interior.size < 2:
            n_coarse += 1
        power = np.trapezoid(np.interp(nodes, frequencies, psd), nodes)
        if power > 0:
            levels[i] = 10.0 * np.log10(power / ref ** 2)
    if n_coarse:
        warnings.warn(
            f"decidecade_band_levels: {n_coarse} band(s) hold fewer than two "
            "interior PSD grid points and rest almost entirely on interpolated "
            "band edges; the PSD grid is too coarse to resolve them. Use a "
            "finer-resolution PSD for a fully integrated level.",
            UserWarning, stacklevel=2,
        )
    return centers, levels


# ──────────────────────────────────────────────────────────────────────
# Time-resolved views
#
# What the record contains as a function of
# time as well as frequency: Hilbert, spectrogram, wavelet, Wigner-Ville,
# cepstrum.
# ──────────────────────────────────────────────────────────────────────

WignerVilleResult = namedtuple("WignerVilleResult",
                               "frequencies times distribution")
CWTResult = namedtuple("CWTResult", "frequencies coefficients")
SpectrogramResult = namedtuple("SpectrogramResult", "frequencies times power")


def analytic_signal(data):
    """Analytic signal ``data + j*Hilbert(data)`` of a real signal."""
    xa = np.asarray(data)
    if np.iscomplexobj(xa):
        raise ConfigurationError(
            "analytic_signal: data must be real; the analytic/Hilbert "
            "representation is only defined for a real signal (got complex input)."
        )
    xr = xa.astype(float)
    if xr.ndim != 1:
        raise ConfigurationError(
            f"analytic_signal: data must be 1-D; got shape {xr.shape}")
    require_finite_signal(xr, "analytic_signal")
    return hilbert(xr)


def envelope(data):
    """Instantaneous amplitude envelope ``|analytic_signal(data)|``."""
    return np.abs(analytic_signal(data))


def instantaneous_frequency(data, sample_rate: float):
    """Instantaneous frequency (Hz) from the analytic-signal phase derivative.

    Returns an array of length ``len(data)`` (centred differences of the
    unwrapped phase via :func:`numpy.gradient`, time-aligned with ``data``).
    """
    fs = require_positive_finite_scalar(
        sample_rate, "instantaneous_frequency", "sample_rate", " Hz")
    phase = np.unwrap(np.angle(analytic_signal(data)))
    return np.gradient(phase) / (2.0 * np.pi) * fs


def _smoothing_window(spec, name):
    """Centered, odd-length smoothing window from a ``None`` / int / array spec.

    ``None`` -> no smoothing. An int ``L`` -> a symmetric Hann window of odd length (``L-1`` for even ``L``). A 1-D
    array is used verbatim when its length is odd; an even-length array has
    its centre sample deleted, giving an odd length. Returns ``(w, half)`` with ``w`` of length
    ``2*half+1`` centered at index ``half`` (so ``w[half+k]`` weights offset
    ``k``); ``(None, 0)`` for no smoothing.
    """
    if spec is None:
        return None, 0
    if np.isscalar(spec):
        L = int(spec)
        if L < 1:
            raise ConfigurationError(
                f"wigner_ville: {name} length must be >= 1; got {L}")
        # Generate an odd length directly: trimming an even symmetric
        # window (hann(6)[:-1]) puts the peak off-centre, which breaks the
        # acc(-tau) = conj(acc(tau)) symmetry the transform's .real relies
        # on (measured up to 59 % error at L = 4).
        w = _sig.get_window("hann", L - 1 if L % 2 == 0 else L,
                            fftbins=False).astype(float)
    else:
        w = np.asarray(spec, dtype=float)
        if w.ndim != 1 or w.size < 1:
            raise ConfigurationError(
                f"wigner_ville: {name} must be a 1-D window; "
                f"got shape {w.shape}")
        if w.size % 2 == 0:
            # Delete the centre sample, not the last one: an even symmetric
            # window has two equal middle samples, so dropping one of them
            # keeps the peak on-centre, while trimming the tail puts the peak
            # off-centre and breaks the acc(-tau) = conj(acc(tau)) symmetry
            # the transform's .real relies on (same reason the int path
            # generates an odd length directly).
            w = np.delete(w, w.size // 2)
    return w, w.size // 2


# Cap on the (NF, n) float64 distribution, checked before it is allocated:
# 2**27 cells is 1 GiB, the ceiling ``active._MAX_AMBIGUITY_CELLS`` sets.
_MAX_WIGNER_CELLS = 1 << 27


def wigner_ville(data, sample_rate: float, *, analytic: bool = True,
                 freq_window=None, time_window=None, nfft=None):
    """Discrete (smoothed-pseudo-) Wigner-Ville distribution of a 1-D signal.

    Returns a :class:`WignerVilleResult` ``(frequencies, times, distribution)``
    with the distribution real, shape ``(NF, n)``; ``f`` spans
    ``[0, fs/2)``. The kernel ``z(t+tau)z*(t-tau)`` doubles the apparent
    frequency, so the physical frequency axis is ``k*fs/(2*NF)``.

    A quadratic energy distribution — there is no routine inverse (like a
    spectrogram, it maps a signal to a 2-D density, not reversibly).

    Parameters
    ----------
    data : 1-D array
        Real or complex signal.
    sample_rate : float
        Sample rate (Hz).
    analytic : bool
        Use the analytic signal for real input (default), suppressing
        cross-terms with the negative spectrum. ``False`` runs the raw signal.
        Ignored when ``data`` is already complex.
    freq_window : None, int, or 1-D array
        Lag-domain smoothing window ``h(tau)`` — the *pseudo*-WVD. Smooths
        along frequency and limits the lag extent (shorter window -> more
        cross-term suppression, coarser frequency resolution). ``None`` is the
        full-lag WVD. An int gives a symmetric Hann window of odd length (L-1 for even L).
    time_window : None, int, or 1-D array
        Time-domain smoothing window ``g`` — the *smoothed*-pseudo-WVD. Averages
        the instantaneous autocorrelation over neighbouring times (more
        cross-term suppression, coarser time resolution). ``None`` disables it.
    nfft : int, optional
        Zero-pad the lag FFT to ``nfft >= n`` bins (finer frequency spacing).
        ``None`` uses ``n``.

    Returns
    -------
    WignerVilleResult
        ``(frequencies, times, distribution)``: frequency axis (Hz), time axis
        (s), and the distribution ``(NF, n)`` — the same axis order as
        :class:`SpectrogramResult`.
    """
    xc = np.asarray(data)
    if xc.ndim != 1:
        raise ConfigurationError(
            f"wigner_ville: data must be 1-D; got shape {xc.shape}")
    require_finite_signal(xc, "wigner_ville")
    if np.iscomplexobj(xc):
        z = xc
    elif analytic:
        z = analytic_signal(xc)
    else:
        z = xc.astype(complex)
    n = z.size
    fs = require_positive_finite_scalar(sample_rate, "wigner_ville",
                                        "sample_rate", " Hz")
    NF = n if nfft is None else int(nfft)
    if NF < n:
        raise ConfigurationError(
            f"wigner_ville: nfft ({NF}) must be >= n ({n}) (zero-pad only)")
    hv, Lh = _smoothing_window(freq_window, "freq_window")
    gv, Lg = _smoothing_window(time_window, "time_window")
    lag_cap = n - 1 if hv is None else Lh
    cells = NF * n
    if cells > _MAX_WIGNER_CELLS:
        raise ConfigurationError(
            f"wigner_ville: the distribution would be {NF} x {n} = {cells} "
            f"float64 cells ({cells * 8 / 2 ** 30:.2f} GiB), past the "
            f"{_MAX_WIGNER_CELLS} cell cap "
            f"({_MAX_WIGNER_CELLS * 8 / 2 ** 30:.2f} GiB). Shorten or "
            f"decimate the record, or pass a smaller nfft; freq_window "
            f"bounds the lags, not the surface.")
    W = np.zeros((NF, n))
    for ti in range(n):
        taumax = min(ti, n - 1 - ti, lag_cap)
        taus = np.arange(-taumax, taumax + 1)
        if gv is None:
            acc = z[ti + taus] * np.conj(z[ti - taus])
        else:
            # All lags at once over the full time-smoothing support
            # m in [-Lg, Lg]. Per lag tau the valid support is |m| <= mmax
            # with mmax = min(Lg, ti - |tau|, n - 1 - ti - |tau|) (both
            # z[ti+tau+m] and z[ti-tau+m] in range); out-of-support terms
            # are masked to zero, indices clipped so the masked positions
            # never index out of bounds.
            ms = np.arange(-Lg, Lg + 1)
            mmax = np.minimum(Lg, np.minimum(ti - np.abs(taus),
                                             n - 1 - ti - np.abs(taus)))
            valid = np.abs(ms)[None, :] <= mmax[:, None]
            ip = np.clip(ti + taus[:, None] + ms[None, :], 0, n - 1)
            im = np.clip(ti - taus[:, None] + ms[None, :], 0, n - 1)
            gw = gv[Lg + ms]
            prod = np.where(valid, gw[None, :] * z[ip] * np.conj(z[im]), 0.0)
            wsum = np.where(valid, gw[None, :], 0.0).sum(axis=1)
            acc = prod.sum(axis=1) / wsum
        if hv is not None:
            acc = acc * hv[Lh + taus]
        kernel = np.zeros(NF, dtype=complex)
        # Lag axis in FFT order: lag 0 at index 0, negative lags wrapped to the
        # top of the buffer, zeros in the unfilled middle (the zero-padding).
        kernel[(taus + NF) % NF] = acc
        # acc(-tau) = conj(acc(tau)) — a symmetric smoothing window preserves
        # that — so the transform is real and `.real` drops only rounding
        # error, not signal. An asymmetric `freq_window` array breaks the
        # symmetry and the discarded imaginary part is then meaningful.
        W[:, ti] = np.real(np.fft.fft(kernel))
    f = np.arange(NF) * fs / (2.0 * NF)
    t = np.arange(n) / fs
    return WignerVilleResult(f, t, W)


def _wavelet_fourier(wavelet, s, omega, w0, order):
    """Fourier-domain daughter wavelet ``psi_hat(s*omega)`` and the Fourier
    frequency<->scale factor ``f = factor*fs/s`` (Torrence & Compo 1998, Table 1).
    ``omega`` is in rad/sample."""
    so = s * omega
    pos = omega > 0.0
    if wavelet == "morlet":
        psi = (np.pi ** -0.25) * np.exp(-0.5 * (so - w0) ** 2) * pos
        factor = (w0 + np.sqrt(2.0 + w0 ** 2)) / (4.0 * np.pi)
    elif wavelet == "paul":
        m = order
        norm = 2.0 ** m / np.sqrt(m * math.factorial(2 * m - 1))
        psi = np.zeros(omega.shape, dtype=complex)
        psi[pos] = norm * (so[pos] ** m) * np.exp(-so[pos])
        factor = (2.0 * m + 1.0) / (4.0 * np.pi)
    elif wavelet == "dog":
        m = order
        norm = -(1j ** m) / np.sqrt(gamma(m + 0.5))
        psi = norm * (so ** m) * np.exp(-0.5 * so ** 2)  # real wavelet: all omega
        factor = np.sqrt(m + 0.5) / (2.0 * np.pi)
    else:
        raise ConfigurationError(
            f"cwt: unknown wavelet {wavelet!r}; choose 'morlet', 'paul', or 'dog'"
        )
    return psi, factor


@lru_cache(maxsize=None)
def _reconstruction_constants(wavelet, w0, order):
    """``(C_delta, psi0(0))`` for the Torrence & Compo (1998) eq.-11 inverse.

    ``psi0(0)``, the wavelet at zero lag, is ``(2*pi)**-0.5 * int psihat0(w) dw``.
    ``C_delta`` is the delta-function calibration of T&C eq. 13-14 taken in the
    continuous-scale limit: for a unit impulse, ``Re{W(s)}/sqrt(s)`` collapses to
    ``psi0(0)*R(pi*s)/s``, where ``R(a)`` is the fraction of the wavelet's
    spectral mass inside the discrete band ``|w| <= pi`` rad/sample. The
    calibration is therefore a one-dimensional quadrature over scale, and it is
    a property of the wavelet alone — not of the scale set the caller analysed
    with, so a band-limited scale set still reconstructs only its own band.

    Against T&C's Table 2 (whose caption calls those factors "empirically
    derived"): ``psi0(0)`` agrees to every tabulated digit for Morlet ``w0=6``,
    Paul ``m=4`` and DOG ``m=2``/``m=6``; ``C_delta`` agrees to 0.3 % for
    Morlet and Paul but differs by ~2 % for the DOG pair (0.776/1.132/3.541/
    1.966 tabulated against 0.778/1.133/3.616/1.929 here). Extends the table to
    any admissible ``w0`` / ``order``.
    """
    u_max = 60.0 + w0 + 6.0 * order
    u = np.linspace(-u_max, u_max, 100001)
    psi_hat, factor = _wavelet_fourier(wavelet, 1.0, u, w0, order)
    mass = cumulative_trapezoid(np.real(psi_hat), u, initial=0.0)
    total = mass[-1] - mass[0]
    psi0_zero = total / np.sqrt(2.0 * np.pi)
    if abs(psi0_zero) < 1e-9:
        raise ConfigurationError(
            f"inverse_cwt: the {wavelet!r} wavelet of order {order} is an odd "
            "function, so psi0(0) = 0 and the Torrence & Compo eq.-11 "
            "reconstruction is undefined for it. Use an even order.")
    dj = 0.02
    s_min = 2.0 * factor / 256.0            # well below the Nyquist-cut scale
    n_scale = int(np.log2(2.0e6 / s_min) / dj) + 1
    scales = s_min * 2.0 ** (np.arange(n_scale) * dj)
    a = np.minimum(np.pi * scales, u_max)
    in_band = (np.interp(a, u, mass) - np.interp(-a, u, mass)) / total
    return float(np.sum(in_band / scales) * dj), float(psi0_zero)


def cwt(data, sample_rate, frequencies=None, wavelet="morlet", *, w0=6.0,
        order=None, n_freqs=64):
    """Continuous wavelet transform with a selectable wavelet (FFT-based).

    At each scale the signal is filtered by the chosen analysing wavelet
    (Mallat, *A Wavelet Tour of Signal Processing*, Ch. 4; scale<->frequency
    factors from Torrence & Compo 1998). Linear in time, logarithmic in
    frequency — well suited to dispersive / transient ocean-acoustic arrivals.

    Parameters
    ----------
    data : 1-D array
        Real signal.
    sample_rate : float
        Sample rate (Hz).
    frequencies : array, optional
        Frequencies (Hz) to analyse. Default: ``n_freqs`` log-spaced points
        from ``4*fs/N`` to ``fs/2``.
    wavelet : {'morlet', 'paul', 'dog'}
        Analysing wavelet. ``'morlet'`` (complex, best frequency resolution),
        ``'paul'`` (complex, best time resolution), ``'dog'`` (real Derivative
        Of Gaussian; ``order=2`` is the Mexican-hat / Ricker wavelet).
    w0 : float
        Morlet central (non-dimensional) frequency; ``>= 5`` keeps it
        admissible. Default 6.
    order : int, optional
        Wavelet order ``m``. Default 4 for ``'paul'``, 2 for ``'dog'``; ignored
        for ``'morlet'``.
    n_freqs : int
        Number of log-spaced frequencies when ``frequencies`` is None.

    Returns
    -------
    CWTResult
        ``(frequencies, coefficients)``: the analysis frequencies (Hz) and the
        complex CWT coefficients, shape ``(n_freqs, len(data))``;
        ``abs(coefficients)`` is the scalogram.
    """
    xa = np.asarray(data)
    if np.iscomplexobj(xa):
        raise ConfigurationError(
            "cwt: data must be real (got complex input); the transform "
            "analyses a real signal.")
    xr = xa.astype(float)
    if xr.ndim != 1:
        raise ConfigurationError(
            f"cwt: data must be 1-D; got shape {xr.shape}")
    require_finite_signal(xr, "cwt")
    fs = require_positive_finite_scalar(sample_rate, "cwt", "sample_rate",
                                        " Hz")
    if order is None:
        order = 4 if wavelet == "paul" else 2
    n = xr.size
    if frequencies is None:
        f_lo = 4.0 * fs / n  # lowest default frequency = 4 cycles per record
        if f_lo >= fs / 2.0:
            raise ConfigurationError(
                f"cwt: signal too short (n={n}) for the default frequency "
                f"range — the lowest analysis frequency {f_lo:.1f} Hz already "
                f"exceeds Nyquist {fs / 2.0:.1f} Hz. Pass an explicit "
                "`frequencies=` below Nyquist, or use a longer signal.")
        frequencies = np.geomspace(f_lo, fs / 2.0, int(n_freqs))
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
    if np.any(frequencies <= 0):
        bad = frequencies <= 0
        raise ConfigurationError(
            f"cwt: frequencies must be > 0; got {int(bad.sum())} value(s) "
            f"<= 0, first at index {int(np.argmax(bad))} "
            f"({frequencies[bad][0]:g} Hz)")
    require_at_most_nyquist(frequencies, fs, "cwt", "analysis frequencies",
                            "the sampled wavelet aliases and the coefficients "
                            "are numerical residue, not band content")
    omega = 2.0 * np.pi * np.fft.fftfreq(n)  # rad/sample
    Xf = np.fft.fft(xr)
    # One scale gives the factor; scales follow from f = factor*fs/s.
    _, factor = _wavelet_fourier(wavelet, 1.0, omega, w0, order)
    scales = factor * fs / frequencies  # scale in samples
    W = np.empty((frequencies.size, n), dtype=complex)
    for i, s in enumerate(scales):
        psi_hat, _ = _wavelet_fourier(wavelet, s, omega, w0, order)
        # Torrence & Compo eq. 6 normalisation sqrt(2*pi*s/dt), with dt = 1
        # because `omega` is rad/sample: equal energy at every scale.
        psi_hat = np.sqrt(2.0 * np.pi * s) * psi_hat
        W[i] = np.fft.ifft(Xf * np.conj(psi_hat))
    return CWTResult(frequencies, W)


def inverse_cwt(W, frequencies, sample_rate, wavelet="morlet", *, w0=6.0,
                order=None):
    """Inverse CWT (Torrence & Compo 1998, eq. 11).

    ``x_n = dj/(C_delta*psi0(0)) * sum_j Re(W_n(s_j))/sqrt(s_j)`` with ``dj``
    the log2 scale spacing. The reconstruction constants ``C_delta`` and
    ``psi0(0)`` are derived for the wavelet order actually in use
    (:func:`_reconstruction_constants`), so non-default ``w0`` / ``order``
    reconstruct at the right amplitude. Pass the same ``frequencies`` /
    ``wavelet`` / ``w0`` / ``order`` used in :func:`cwt`; amplitude is
    recovered to the accuracy of the scale coverage (a band-limited scale set
    reconstructs only the band it spans). The quadrature assumes the scale
    grid is **uniform in log2** — true of :func:`cwt`'s default log-spaced
    frequencies; a non-uniform grid (e.g. linearly spaced frequencies, or a
    single frequency) biases the amplitude and raises a ``UserWarning``.

    Parameters
    ----------
    W : ndarray
        CWT coefficients ``(n_freqs, n_time)`` from :func:`cwt`.
    frequencies : array
        The analysis frequencies returned by :func:`cwt`.
    sample_rate : float
        Sample rate (Hz).

    Returns
    -------
    ndarray
        Reconstructed 1-D signal.
    """
    Wc = np.asarray(W)
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
    if Wc.ndim != 2 or Wc.shape[0] != frequencies.size:
        raise ConfigurationError(
            "inverse_cwt: W must be (n_freqs, n_time) matching frequencies"
            f"; got W shape {Wc.shape} and {frequencies.size} frequencies")
    if order is None:
        order = 4 if wavelet == "paul" else 2
    fs = require_positive_finite_scalar(sample_rate, "inverse_cwt",
                                        "sample_rate", " Hz")
    _, factor = _wavelet_fourier(wavelet, 1.0, np.array([1.0]), w0, order)
    scales = factor * fs / frequencies
    if scales.size > 1:
        dlog = np.abs(np.diff(np.log2(scales)))
        dj = float(np.mean(dlog))
        # Eq. 11 is a quadrature over log2-uniform scales with spacing dj; on
        # a non-uniform grid a single mean dj misweights every scale and the
        # amplitude comes out wrong.
        if float(np.std(dlog)) > 0.01 * dj:
            warnings.warn(
                "inverse_cwt: the scale grid implied by `frequencies` is not "
                "uniform in log2 (std(diff(log2(scales))) = "
                f"{float(np.std(dlog)):.3g} vs mean {dj:.3g}), but the "
                "Torrence & Compo eq.-11 sum assumes log2-uniform scale "
                "spacing dj — the reconstruction amplitude will be biased. "
                "Use log-spaced frequencies (cwt's default grid).",
                UserWarning, stacklevel=2)
    else:
        dj = 1.0
        warnings.warn(
            "inverse_cwt: a single scale cannot calibrate the log2 scale "
            "spacing dj the Torrence & Compo eq.-11 sum assumes (dj is taken "
            "as 1.0), so the reconstruction amplitude is arbitrary. Analyse "
            "with several log-spaced frequencies.",
            UserWarning, stacklevel=2)
    c_delta, psi0_zero = _reconstruction_constants(wavelet, float(w0), int(order))
    return (np.sum(np.real(Wc) / np.sqrt(scales)[:, None], axis=0)
            * dj / (c_delta * psi0_zero))


def _apply_lifter(c, lifter):
    """Quefrency-domain liftering of a real cepstrum ``c`` (length ``NF``).

    ``lifter`` is an int cutoff or a 1-D weight array. A positive int ``L``
    short-passes (keeps ``|quefrency| <= L`` -> spectral envelope); a negative
    int long-passes (zeros ``|quefrency| <= |L|`` -> excitation / echo). An
    array multiplies the cepstrum element-wise (symmetric weighting is on you).
    """
    nf = c.size
    if np.isscalar(lifter):
        L = int(lifter)
        w = np.zeros(nf)
        keep = abs(L)
        w[:keep + 1] = 1.0
        if keep:
            # Negative quefrencies live at the top of the buffer. max(1, ...)
            # is what makes an over-long cutoff (keep >= nf) keep everything:
            # a bare nf - keep would go negative and w[-2:] would set two
            # elements instead of the whole tail.
            w[max(1, nf - keep):] = 1.0
        if L < 0:
            w = 1.0 - w
        return c * w
    w = np.asarray(lifter, dtype=float)
    if w.shape != c.shape:
        raise ConfigurationError(
            f"cepstrum: lifter array {w.shape} must match cepstrum {c.shape}")
    return c * w


def cepstrum(data, *, window=None, nfft=None, lifter=None):
    """Real cepstrum ``irfft(log|rfft(data)|)``.

    Not invertible: discards phase. Use :func:`complex_cepstrum` /
    :func:`inverse_complex_cepstrum` for a reversible homomorphic transform.

    Parameters
    ----------
    data : 1-D array
        Real signal.
    window : None, str, or tuple
        :func:`scipy.signal.get_window` spec applied before the FFT to curb
        spectral leakage. ``None`` is rectangular.
    nfft : int, optional
        Zero-pad the FFT to ``nfft >= len(data)`` bins (finer quefrency
        spacing). ``None`` uses ``len(data)``.
    lifter : None, int, or 1-D array
        Quefrency liftering — see :func:`_apply_lifter`. ``None`` returns the
        raw cepstrum; a positive int keeps low quefrencies (spectral envelope),
        a negative int keeps high quefrencies (pitch / echo structure).
    """
    xa = np.asarray(data)
    if np.iscomplexobj(xa):
        raise ConfigurationError(
            "cepstrum: data must be real (got complex input); the real cepstrum "
            "irfft(log|rfft(x)|) is defined for a real signal. For a complex "
            "spectrum use complex_cepstrum.")
    xr = xa.astype(float)
    if xr.ndim != 1:
        raise ConfigurationError(
            f"cepstrum: data must be 1-D; got shape {xr.shape}")
    require_finite_signal(xr, "cepstrum")
    n = xr.size
    NF = n if nfft is None else int(nfft)
    if NF < n:
        raise ConfigurationError(
            f"cepstrum: nfft ({NF}) must be >= len(x) ({n}) (zero-pad only)")
    if window is not None:
        xr = xr * _sig.get_window(window, n, fftbins=True).astype(float)
    spectrum = np.abs(np.fft.rfft(xr, n=NF))
    spectrum = np.maximum(spectrum, np.finfo(float).tiny)
    c = np.fft.irfft(np.log(spectrum), n=NF)
    if lifter is not None:
        c = _apply_lifter(c, lifter)
    return c


ComplexCepstrum = namedtuple("ComplexCepstrum", "cepstrum delay")


def complex_cepstrum(data):
    """Complex cepstrum with the linear-phase (rotation) term removed.

    Returns ``ComplexCepstrum(cepstrum, delay)``. ``delay`` is the integer
    number of samples of linear phase taken out — the signal's delay, positive
    for a late signal;
    :func:`inverse_complex_cepstrum` needs it to reconstruct ``data``.

    Without the removal the unwrapped phase carries a ramp whose inverse
    transform is a ``1/q`` tail that swamps the echo structure the cepstrum
    exists to show, and makes the result depend on the signal's absolute
    arrival time rather than on its echo delays.

    The cepstrum stays **complex**: unwrapping breaks the Hermitian symmetry
    of ``log(fft(x))``, and the imaginary part is what makes the homomorphic
    transform reversible.
    """
    xa = np.asarray(data)
    if np.iscomplexobj(xa):
        raise ConfigurationError(
            "complex_cepstrum: data must be real (got complex input); the "
            "homomorphic cepstrum is defined for a real signal.")
    xr = xa.astype(float)
    if xr.ndim != 1:
        raise ConfigurationError(
            f"complex_cepstrum: data must be 1-D; got shape {xr.shape}")
    require_finite_signal(xr, "complex_cepstrum")
    spectrum = np.fft.fft(xr)
    n = xr.size
    mag = np.maximum(np.abs(spectrum), np.finfo(float).tiny)
    phase = np.unwrap(np.angle(spectrum))
    # Remove the linear-phase term: round the end-to-end ramp to a whole
    # number of samples and subtract it, so the residual phase carries only
    # the echo structure.
    # A delay of d samples is the phase ramp -2*pi*d*k/n, so the ramp's
    # coefficient is -d: negate it, and ``delay`` is the delay in samples
    # (positive for a late signal), as the docstring says.
    delay = -int(np.round(phase[n // 2] * n / (2.0 * np.pi * (n // 2)))) if n > 1 else 0
    phase = phase + 2.0 * np.pi * delay * np.arange(n) / n
    log_spectrum = np.log(mag) + 1j * phase
    return ComplexCepstrum(np.fft.ifft(log_spectrum), delay)


def inverse_complex_cepstrum(c):
    """Invert :func:`complex_cepstrum`: ``x = real(ifft(exp(fft(c))))``.

    Takes the :class:`ComplexCepstrum` namedtuple :func:`complex_cepstrum`
    returns (the imaginary part of its ``cepstrum`` field is significant —
    see there) and reconstructs the real signal ``x``. Both fields are
    required: the ``delay`` restores the linear-phase term the forward
    transform removed, without which the signal comes back at the wrong
    arrival time."""
    if not isinstance(c, ComplexCepstrum):
        raise ConfigurationError(
            "inverse_complex_cepstrum: c must be the ComplexCepstrum "
            "namedtuple returned by complex_cepstrum — its delay field "
            "restores the linear-phase term the forward transform removed, "
            "and a bare cepstrum array carries no delay. Pass the "
            "complex_cepstrum result unchanged, or "
            "ComplexCepstrum(cepstrum=your_array, delay=your_delay). "
            f"Got {type(c).__name__}.")
    c, delay = c.cepstrum, int(c.delay)
    cr = np.asarray(c, dtype=complex)
    if cr.ndim != 1:
        raise ConfigurationError(
            f"inverse_complex_cepstrum: c must be 1-D; got shape {cr.shape}")
    n = cr.size
    log_spectrum = np.fft.fft(cr)
    # Restore the linear-phase term complex_cepstrum took out.
    log_spectrum = log_spectrum - 1j * 2.0 * np.pi * delay * np.arange(n) / n
    return np.real(np.fft.ifft(np.exp(log_spectrum)))


def spectrogram(data, sample_rate, *, window="hann", nperseg=8192,
                noverlap=None, nfft=None, scaling="density", mode="psd"):
    """Short-time spectrogram. Returns a :class:`SpectrogramResult`
    ``(frequencies, times, power)`` (Pa²/Hz with the default
    ``scaling='density'``/``mode='psd'``).

    ``noverlap=None`` (default) lets scipy derive the overlap (``nperseg // 8``)
    and clamp ``nperseg`` to the input length, so short signals don't raise; pass
    an int to override. ``nfft`` (zero-pad length) mirrors
    :func:`uacpy.acoustic_signal.welch`. ``mode`` is passed through to
    :func:`scipy.signal.spectrogram`: ``'psd'`` (default) is the power
    spectral density the stated Pa²/Hz units apply to; ``'complex'`` /
    ``'magnitude'`` return the (windowed) STFT itself in Pa /
    ``'angle'``/``'phase'`` its phase in radians — for those the ``power``
    field is not a power and the Pa²/Hz units do not apply. For logarithmic /
    constant-Q frequency resolution, see
    :func:`uacpy.acoustic_signal.constant_q_spectrogram`.

    **Complex input** is accepted, unlike :func:`cwt` or
    :func:`analytic_signal`, but returns a two-sided spectrum on an unsorted
    frequency axis (``0 .. fs/2`` then ``-fs/2 .. 0``) rather than the
    one-sided density above; a ``UserWarning`` says so."""
    data = require_finite_signal(data, "spectrogram")
    _warn_two_sided("spectrogram", data)
    sample_rate = require_positive_finite_scalar(
        sample_rate, "spectrogram", "sample_rate", " Hz")
    f, t, Sxx = _sig.spectrogram(data, sample_rate, window=window,
                                 nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                                 scaling=scaling, mode=mode)
    return SpectrogramResult(f, t, Sxx)
