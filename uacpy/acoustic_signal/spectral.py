"""Time-averaged spectral and level estimates of a recorded signal.

:func:`welch`, :func:`constant_q` and :func:`sound_exposure` say what a
record holds on average, each as a :class:`SpectralEstimate`; their
``probabilistic_`` twins give the distribution of the level over the
record's segments as a :class:`ProbabilisticSpectralEstimate`. The
constant-Q machinery is in :mod:`~uacpy.acoustic_signal.cqt`, the
standard band ladders in :mod:`~uacpy.acoustic_signal.bands`.
"""

from __future__ import annotations

import warnings
from collections import namedtuple
import numpy as np
import scipy.signal as _sig
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, NumericsWarning, ValidityWarning,
)
from uacpy.core.constants import REFERENCE_PRESSURE_WATER
from uacpy.core.acoustics import power_to_dB
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._validate import (
    require_finite_signal, require_positive_finite_scalar,
)
from uacpy.acoustic_signal.windows import _DEFAULT_NPERSEG, _default_noverlap
from uacpy.acoustic_signal._results import (
    POWER_UNITS, PlottedResult, level_unit,
)


# ──────────────────────────────────────────────────────────────────────
# Spectral estimators
#
# The six — welch, constant_q and
# sound_exposure, each with a probabilistic twin — and their two result types.
# ──────────────────────────────────────────────────────────────────────

def _warn_two_sided(who: str, data):
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
            f"{who}: complex input gives a TWO-SIDED spectrum — "
            f"'frequencies' runs 0..fs/2 then -fs/2..0 and is not sorted, and "
            f"the density is not the one-sided Pa²/Hz this function "
            f"documents. Sort both arrays together (i = np.argsort(f)) or "
            f"np.fft.fftshift them, or pass a real pressure series.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)


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
#: It also pins the window (the ``'exposure'`` entry of
#: :data:`_DEFAULT_WINDOW`), because only an
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
#: ``bins_per_octave`` -- and passing one method's to the other raises
#: rather than being ignored.
#:
#: Reporting on standard frequency bands is NOT a third method: it is those
#: same Welch bins summed into the bands of ``band_type``, so it is an
#: argument (see :data:`BAND_TYPES`) rather than a peer of the two transforms
#: that actually resolve frequency differently.
_SPECTRAL_METHODS = ("welch", "constant_q")


#: Standard ladders :func:`sound_exposure` can integrate its bins into.
#:
#: ``'decidecade'`` (centres ``1000*10**(n/10)``, ISO 18405) and ``'octave'``
#: (every third decidecade) are the IEC 61260-1 base-10 ladders, and
#: ``'linear'`` is ``n_bands`` equal-width bands. ``None``, the
#: default, reports the method's own bins.
#:
#: Only the Welch bins can be integrated: they are orthogonal, so each bin's
#: energy is shared out once over the bands its interval overlaps and the
#: weighted sum is the band's power. Constant-Q
#: kernels overlap by construction, so summing them would count the same
#: energy more than once — asking for both raises.
BAND_TYPES = ("decidecade", "octave", "linear")


#: Options that belong to the banding rather than to either method, so they
#: are refused when no ``band_type`` was asked for: there is nothing for them
#: to describe.
_BAND_OPTIONS = ("n_bands", "batch_size")


#: Spectral lines a band wants before an FFT-synthesised band level behaves
#: like a filter-bank one: "at least ten spectral lines in each band" (Fahy,
#: *Sound Intensity*, on synthesising constant-percentage-bandwidth spectra
#: from an FFT). At the 1 Hz bins ``nperseg`` defaults to, the seven lowest
#: decidecade bands (10-40 Hz) hold 2 to 7 lines and fall short of it.
#:
#: That is a resolution limit, not an error in the total or the width: each
#: bin is split between the bands its interval overlaps in proportion to the
#: overlap (``_fractional_bin_weights``), so a band integrates exactly its own
#: width and the sum over bands stays exact — on a flat spectrum the 1 Hz-bin
#: band levels carry no width bias, as :func:`decidecade_band_levels` (a PSD
#: integrated over the exact band edges) carries none. What few lines cost is
#: spectral detail inside a band and the variance of the band estimate.
#: Raising ``nperseg`` to about ``10 * sample_rate / 2.3`` (4.3 s of record,
#: for the 10 Hz band's 2.3 Hz width) resolves them, at the cost of time
#: resolution; it is a choice rather than a default because the 1 Hz grid is
#: what soundscape reporting is written on.
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
#: under ``flattop``. The cost is resolution — flat-top's noise-equivalent
#: bandwidth is 3.77 bins against hann's 1.50, and its main lobe 10 bins
#: against hann's 4 — so two tones closer than that merge. Pass
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


# Longest trailing axis the time-axis guard reads as a channel count rather
# than a record: ``read_wav`` (and soundfile) return ``(n_samples,
# n_channels)``, and a transform along that last axis is 2-3 "samples" long.
_MAX_CHANNELS_GUESSED = 64


def _time_axis_last(data, axis, who):
    """``data`` with its time axis moved last, ready for an estimator.

    ``axis=None`` means the last axis, as in scipy, but an input shaped like a
    multichannel record — ``(n_samples, n_channels)``, the layout
    :func:`uacpy.io.read_wav` returns — is refused rather than transformed
    along its channels: a Welch estimate of a (48000, 3) array along its last
    axis is 48 000 two-bin "spectra", with no error. Pass ``axis=0`` for that
    layout, or ``axis=-1`` to confirm the last axis is time.
    """
    try:
        arr = np.asarray(data)
    except ValueError as exc:
        raise ConfigurationError(
            f"{who}: data is not a rectangular array ({exc}).",
            remediation="Pass one record as a 1-D array, or channels of "
                        "equal length as a 2-D array.") from exc
    if arr.dtype == object:
        # Not numeric (a Field, say): the signal validator downstream names
        # what was passed and the bridge to its data.
        return data
    if axis is None:
        if (arr.ndim == 2 and arr.shape[-1] <= _MAX_CHANNELS_GUESSED
                and arr.shape[0] > arr.shape[-1]):
            raise ConfigurationError(
                f"{who}: data is shaped {arr.shape}, which reads as "
                f"(n_samples, n_channels) — read_wav's layout — so the last "
                f"axis would be transformed across the channels.",
                remediation="Pass axis=0 for (n_samples, n_channels), or "
                            "axis=-1 if the last axis really is time.")
        return arr
    try:
        return np.moveaxis(arr, int(axis), -1)
    except (np.exceptions.AxisError, TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{who}: axis={axis!r} is not an axis of data shaped "
            f"{arr.shape}.") from exc


def _is_record_list(data):
    """Whether ``data`` is a list of records (the probabilistic estimators'
    form for records of different lengths) rather than one record's samples.

    A list whose first element is itself a sequence is a list of records; a
    list of numbers is a plain signal.
    """
    return isinstance(data, list) and len(data) > 0 and np.ndim(data[0]) >= 1


def _split_records(data, who):
    """The 1-D records a probabilistic estimator pools into one histogram.

    A list of records is taken as it is; an array has its time axis last
    already, and every other axis is a channel contributing its own record.
    """
    if isinstance(data, list):
        records = [np.asarray(s) for s in data]
        for i, s in enumerate(records):
            if s.ndim != 1:
                raise ConfigurationError(
                    f"{who}: a list of records must hold 1-D arrays; "
                    f"list element {i} has ndim={s.ndim}.")
        return records
    data = np.asarray(data)
    return list(data.reshape(-1, data.shape[-1])) if data.ndim > 1 else [data]


def _trim_record(data, sample_rate, integration_time, who):
    """The first ``integration_time`` seconds of ``data``, or all of it.

    ISO 18405's integration time, and the same idea for the other statistics:
    the stretch of record the estimate is taken over. Guarded before it
    reaches the slice, because a negative one is a Python end-slice — -1.0 s
    of a 5 s record would return bit-identically what +4.0 s returns — and
    NaN / Inf raise an untyped ValueError out of ``int()``.
    """
    if integration_time is None:
        return data
    # Records of different lengths share no last axis to slice.
    if _is_record_list(data):
        shapes = [np.shape(s) for s in data]
        if len(set(shapes)) > 1:
            raise ConfigurationError(
                f"{who}: integration_time= takes the first seconds of one "
                f"record, and the {len(shapes)} records given have different "
                f"shapes {sorted(set(shapes))}.",
                remediation="Trim each record to the stretch wanted before "
                            "passing the list, or leave integration_time "
                            "unset to use every record whole.")
    # Before the arithmetic below divides by it: a zero or non-finite rate
    # reaches this function ahead of the estimator's own check (the trim
    # happens first), and 1/0 is a ZeroDivisionError with no name on it.
    sample_rate = require_positive_finite_scalar(
        sample_rate, who, "sample_rate", " Hz")
    integration_time = require_positive_finite_scalar(
        integration_time, who, "integration_time", " s")
    n = min(int(integration_time * sample_rate), np.shape(data)[-1])
    if n < 1:
        raise ConfigurationError(
            f"{who}: no samples to integrate — integration_time="
            f"{integration_time:g} s is shorter than one sample at "
            f"{sample_rate:g} Hz.",
            remediation=f"Pass integration_time >= "
                        f"{1.0 / float(sample_rate):g} s, or leave it unset "
                        f"to use the whole record.")
    return np.asarray(data)[..., :n]


def _crop_to_range(frequencies, values, freq_min, freq_max):
    """The bins inside ``[freq_min, freq_max]``, for a method that computes them all.

    Welch resolves the whole spectrum whatever range is asked for, so the
    range is a crop of what came back rather than a saving — the estimate in
    the bins that remain is the one it would have been either way.
    """
    if freq_min is None and freq_max is None:
        return frequencies, values
    keep = np.ones(np.shape(frequencies), dtype=bool)
    if freq_min is not None:
        keep &= frequencies >= freq_min
    if freq_max is not None:
        keep &= frequencies <= freq_max
    return frequencies[keep], values[..., keep]


class SpectralEstimate(PlottedResult,
                       namedtuple("SpectralEstimate", "frequencies power")):
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

    _attrs = ("scaling", "method", "bands", "band_type")

    def __new__(cls, frequencies, power, scaling="density", method="welch",
                bands=None, band_type=None):
        self = super().__new__(cls, frequencies, power)
        self.scaling = str(scaling)
        self.method = str(method)
        self.bands = bands
        self.band_type = band_type
        return self

    def _plotter_name(self):
        # A banded estimate is drawn as bars over its standard bands
        # (plot_sel), since its values belong to whole bands rather than to
        # points on a frequency axis; a bin estimate as a line (plot_psd).
        # The unit and the title come from the estimate's own scaling and
        # method, so the call cannot mislabel it.
        return "plot_sel" if self.bands is not None else "plot_psd"

    def _field_units(self):
        return {"frequencies": "Hz", "power": POWER_UNITS[self.scaling]}


class ProbabilisticSpectralEstimate(
        PlottedResult,
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
        by ``level_min_dB`` / ``level_max_dB`` the way the histogram is.
    ``level_step_dB``
        Height of one histogram bin, the ``level_step_dB`` it was asked for.
    ``scaling`` / ``method`` / ``bands`` / ``band_type``
        As on :class:`SpectralEstimate`.
    ``ref``
        The reference the dB levels are stated against.
    ``segment_duration``
        The interval each sample describes, in seconds; ``None`` for a
        constant-Q histogram, which samples single frames whose length is per
        bin rather than one duration for the whole estimate.
    ``segment_levels_dB`` / ``segment_times``
        The level (dB) of every segment at every frequency,
        ``(n_segments, n_frequencies)``, the population the histogram,
        ``mean_dB`` and ``std_dB`` are taken over, and the centre time (s)
        of each segment within its record. With several records the
        segments are pooled record by record, so the times restart with
        each. ``None`` for a constant-Q histogram, whose frames per bin
        differ in number.

    :meth:`percentiles` reads level percentiles off ``segment_levels_dB``,
    over every segment, where a quantile read off ``pdf`` is clipped by the
    histogram window.
    """

    _attrs = ("mean_dB", "std_dB", "level_step_dB", "segment_duration", "ref",
              "scaling", "method", "bands", "band_type",
              "segment_levels_dB", "segment_times")

    def __new__(cls, frequencies, level_edges, pdf, *, mean_dB=None,
                std_dB=None, level_step_dB=None, segment_duration=None,
                ref=REFERENCE_PRESSURE_WATER, scaling="density",
                method="welch", bands=None, band_type=None,
                segment_levels_dB=None, segment_times=None):
        self = super().__new__(cls, frequencies, level_edges, pdf)
        self.segment_levels_dB = segment_levels_dB
        self.segment_times = segment_times
        self.mean_dB = mean_dB
        self.std_dB = std_dB
        self.level_step_dB = level_step_dB
        self.segment_duration = segment_duration
        self.ref = float(ref)
        self.scaling = str(scaling)
        self.method = str(method)
        self.bands = bands
        self.band_type = band_type
        return self

    def percentiles(self, q):
        """Level percentiles (dB) per frequency over every segment:
        :func:`level_percentiles` of ``segment_levels_dB``, shape
        ``(len(q), n_frequencies)``, or ``(n_frequencies,)`` for a scalar
        ``q``."""
        if self.segment_levels_dB is None:
            raise ConfigurationError(
                "ProbabilisticSpectralEstimate.percentiles: this "
                f"{self.method!r} estimate keeps no per-segment levels (its "
                f"frames per bin differ in number), so there is no "
                f"population to take percentiles of.",
                remediation="Use probabilistic_welch or "
                            "probabilistic_sound_exposure, whose estimates "
                            "keep segment_levels_dB.")
        return level_percentiles(self.segment_levels_dB, q)

    def _plotter_name(self):
        # Both plotters read ref and scaling off the result, so the level
        # axis names the reference the estimate was computed against and
        # claims "/Hz" only over a density.
        return ("plot_constant_q_ppsd" if self.method == "constant_q"
                else "plot_ppsd")

    def _field_units(self):
        return {"frequencies": "Hz",
                "level_edges": level_unit(self.ref, self.scaling),
                "pdf": "1/dB"}


def _require_bin_scaling(scaling, who):
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
            f"{who}: unknown scaling {scaling!r}.",
            remediation=f"Use one of {_BIN_SCALINGS}: 'density' for power per "
                        f"hertz (noise), 'spectrum' for power per band "
                        f"(tones). For the energy a record delivered, call "
                        f"sound_exposure(), which pins the window and overlap "
                        f"that identity needs.")


def _check_band_type(band_type, who):
    """The ladder a caller named has to be one the package builds.

    The other ways a banding request could go wrong — a ladder over
    constant-Q kernels, a Welch knob that breaks the bin sum, a banding
    option with no banding — are not reachable any more: no function's
    signature offers that combination, which is the point of a function per
    statistic.
    """
    if band_type not in BAND_TYPES:
        raise ConfigurationError(
            f"{who}: unknown band_type {band_type!r}.",
            remediation=f"Use one of {BAND_TYPES}.")


def _estimate(data, sample_rate, *, method="welch", scaling="density",
                      window=None, freq_min=None, freq_max=None, band_type=None,
                      integration_time=None, who="welch", axis=None,
                      **options):
    """Power estimate over frequency, under any method and any scaling.

    Returns a :class:`SpectralEstimate` carrying both choices, so a consumer
    never has to be told what it holds.

    ``method`` (:data:`_SPECTRAL_METHODS`) resolves the frequency axis:
    ``'welch'`` averages uniform segments into equal-width bins and takes
    ``nperseg`` / ``noverlap`` / ``nfft``; ``'constant_q'`` correlates one
    Q-cycle kernel per geometric bin and takes ``bins_per_octave``;
    An option belonging to the other method raises rather than being
    ignored.

    ``band_type`` (:data:`BAND_TYPES`) reports on standard frequency bands
    instead of on the method's own bins: the Welch bins summed into each
    band, which is exact because they are orthogonal. It takes ``n_bands``
    (for ``'linear'``) and ``batch_size``, pins the window to ``boxcar`` so
    every bin is counted once and whole, and is refused over constant-Q,
    whose kernels overlap. Unset, the estimate is per bin.

    ``scaling`` (:data:`_SPECTRAL_SCALINGS`) sets the normalisation, and it is
    orthogonal to the method: ``'density'`` divides each bin's power by that
    bin's own noise-equivalent bandwidth — one number for Welch, a different
    one per bin for constant-Q, since its bins widen with frequency —
    ``'spectrum'`` leaves the power in the band, and ``'exposure'`` multiplies
    it by the duration of the record, giving the energy per band (Pa²·s).

    ``freq_min`` / ``freq_max`` are the frequency range of the estimate under every
    method, and ``integration_time`` the stretch of record it is taken over
    (seconds from the start). What an unset one means is the method's own
    answer: Welch resolves the whole spectrum and the range crops what comes
    back, constant-Q starts at 20 Hz and stops below the near-Nyquist bins
    whose image leaks into a tone's level, and the band ladder
    spans the 10 Hz to 20 kHz reporting range.

    ``window=None`` takes the default that scaling measures best under that
    method (:data:`_DEFAULT_WINDOW`), which is ``flattop`` for a Welch
    spectrum, ``boxcar`` wherever the sum has to equal the record's energy,
    and ``hann`` otherwise; see :func:`welch` for why.

    :func:`welch`, :func:`constant_q` and :func:`sound_exposure` are the public
    doors onto this; each names one statistic and takes only the arguments
    that statistic can honour.

    Reference-free (linear). Convert to dB at plot time with
    :func:`uacpy.plot.plot_psd` (which takes ``ref=``).
    """
    # Deferred: bands and cqt import this module at load
    # time, so the two cannot be imported back at the top.
    from uacpy.acoustic_signal.bands import _band_estimate
    from uacpy.acoustic_signal.cqt import _constant_q_estimate
    if scaling not in _SPECTRAL_SCALINGS:
        raise ConfigurationError(
            f"{who}: unknown scaling {scaling!r}.",
            remediation=f"Use one of {_BIN_SCALINGS}, or sound_exposure() "
                        f"for the energy a record delivered.",
        )
    band_options = {k: options.pop(k) for k in _BAND_OPTIONS if k in options}
    data = _time_axis_last(data, axis, who)
    window = (_BAND_WINDOW if band_type is not None else
              _DEFAULT_WINDOW[method][scaling]) if window is None else window
    data = _trim_record(data, sample_rate, integration_time, who)
    if band_type is not None:
        # Not a third way of resolving frequency: the Welch bins this same
        # function returns, summed into the ladder. ``_band_estimate`` calls
        # back into it, so there is one estimator underneath.
        band_range = {k: v for k, v in (("freq_min", freq_min), ("freq_max", freq_max))
                      if v is not None}
        centres, values, bands = _band_estimate(
            data, sample_rate, scaling=scaling, band_type=band_type,
            window=window, who=who, **band_range, **band_options,
            **options)
        return SpectralEstimate(centres, values, scaling, method, bands,
                                band_type)
    if method == "constant_q":
        kernel_range = {k: v for k, v in (("freq_min", freq_min), ("freq_max", freq_max))
                        if v is not None}
        arr = np.asarray(data)
        if arr.ndim > 1 and arr.dtype != object:
            # One kernel bank, run per channel: time is the last axis here
            # (moved there above) and power carries frequency last, one
            # spectrum per channel, as welch returns it.
            rows = [_constant_q_estimate(
                row, sample_rate, window=window, scaling=scaling,
                who=who, **kernel_range, **options)
                for row in arr.reshape(-1, arr.shape[-1])]
            freqs = rows[0][0]
            power = np.stack([p for _, p in rows]).reshape(
                arr.shape[:-1] + (freqs.size,))
            return SpectralEstimate(freqs, power, scaling, method)
        freqs, power = _constant_q_estimate(
            data, sample_rate, window=window,
            scaling=scaling, who=who, **kernel_range, **options)
        return SpectralEstimate(freqs, power, scaling, method)
    # Guards name the function the CALLER used, not this one: a message
    # reading "_estimate: ..." for a bad argument to ``welch`` sends the
    # reader to a function they did not call.
    data = require_finite_signal(data, who)
    sample_rate = require_positive_finite_scalar(
        sample_rate, who, "sample_rate", " Hz")
    _warn_two_sided(who, data)
    duration = np.shape(data)[-1] / float(sample_rate)
    if scaling == "exposure":
        # Welch averages whole segments and DROPS the samples that do not
        # fill one, so a plain average times the record length would quietly
        # lose the tail (0.19 % of a 4 s record at nperseg=1024). Padding to a
        # whole number of segments adds no energy, and the padded duration is
        # then what turns the average back into the energy that passed the
        # sensor — exactly, for any nperseg. It is the same trick the band
        # method uses on its chunks.
        nps = int(options.get("nperseg", _DEFAULT_NPERSEG))
        n = np.shape(data)[-1]
        n_seg = max(1, -(-n // nps))
        pad = n_seg * nps - n
        if pad:
            data = np.pad(np.asarray(data),
                          [(0, 0)] * (np.ndim(data) - 1) + [(0, pad)])
        duration = n_seg * nps / float(sample_rate)
    nperseg = options.get("nperseg", _DEFAULT_NPERSEG)
    noverlap = options.get("noverlap")
    if noverlap is None:
        # scipy clamps nperseg to the record (and takes 256 for None), so the
        # overlap is sized on the segment scipy will use.
        segment = min(256 if nperseg is None else int(nperseg),
                      np.shape(data)[-1])
        noverlap = (0 if scaling == "exposure"
                    else _default_noverlap(window, segment))
    freqs, Pxx = _sig.welch(data, sample_rate, window=window,
                            nperseg=nperseg, noverlap=noverlap,
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
    freqs, Pxx = _crop_to_range(freqs, Pxx, freq_min, freq_max)
    return SpectralEstimate(freqs, Pxx, scaling, method)


def welch(data, sample_rate, *, scaling="density", nperseg=_DEFAULT_NPERSEG,
          noverlap=None, nfft=None, detrend="constant", average="mean",
          window=None, freq_min=None, freq_max=None, integration_time=None,
          axis=None):
    """Welch estimate on equal-width bins: Pa²/Hz or Pa² per bin.

    Averaged periodograms of overlapping segments (Welch 1967), which is the
    workhorse spectral estimate: ``nperseg`` sets the resolution, and the
    averaging trades variance for it.

    Parameters
    ----------
    data : array_like
        Pressure record (Pa) with time along ``axis``.
    sample_rate : float
        Sample rate (Hz).
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
        ``noverlap=None`` keeps every independent sample the window yields:
        half a segment under hann, 78.3 % under flattop (Abraham §9.2.10.1,
        :func:`~uacpy.acoustic_signal.windows._default_noverlap`).
        ``average='median'`` is the robust choice for a record with
        transients in it — a passing ship moves the mean of the periodograms
        and leaves the median on the background. ``detrend`` is
        ``'constant'`` (subtract each segment's mean, scipy's default),
        ``'linear'``, a callable, or ``False``. Subtracting the mean of a
        segment that holds part of a transient adds a step the transient does
        not have, which puts energy into the lowest bins: pass
        ``detrend=False`` for pulses and model time series.
    window : str, optional
        ``None`` takes the default the scaling measures best: ``hann`` for a
        density, where a narrow main lobe matters and scalloping does not;
        ``flattop`` for a spectrum, where a tone half a bin off centre reads
        about 0.01 dB low instead of hann's **1.42 dB**. Flat-top pays for it
        in resolution: its noise-equivalent bandwidth is 3.77 bins against
        hann's 1.50, and its main lobe 10 bins against hann's 4, so two tones
        closer than that merge. Pass ``window='hann'`` where separating
        neighbours matters more than reading their level.
    freq_min, freq_max : float, optional
        Frequency range to report. Welch resolves the whole spectrum either
        way, so this crops the answer rather than making it cheaper.
    integration_time : float, optional
        Seconds of record to use, from the start; the whole record by
        default.
    axis : int, optional
        The time axis of a multichannel ``data``; the estimate runs along
        it and ``power`` carries frequency LAST (one spectrum per channel).
        ``None`` is the last axis, except that a ``(n_samples, n_channels)``
        array — :func:`uacpy.io.read_wav`'s layout — is refused rather than
        transformed across its channels; pass ``axis=0`` for it.

    Returns
    -------
    SpectralEstimate
        ``(frequencies, power)``, linear and reference-free, carrying
        ``.scaling``. ``.plot()`` draws it in dB.

    Examples
    --------
    A unit sine reads its full ``A²/2`` on its bin under ``'spectrum'``:

    >>> import numpy as np
    >>> fs = 1000.0
    >>> t = np.arange(10_000) / fs
    >>> est = welch(np.sin(2 * np.pi * 100.0 * t), fs, scaling='spectrum',
    ...             nperseg=1000)
    >>> peak = int(np.argmax(est.power))
    >>> float(est.frequencies[peak]), round(float(est.power[peak]), 6)
    (100.0, 0.5)
    """
    _require_bin_scaling(scaling, "welch")
    return _estimate(data, sample_rate, scaling=scaling, window=window,
                     freq_min=freq_min, freq_max=freq_max, integration_time=integration_time,
                     nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                     detrend=detrend, average=average, who="welch",
                     axis=axis)


def constant_q(data, sample_rate, *, scaling="density", freq_min=20.0, freq_max=None,
               bins_per_octave=24, window="hann",
               integration_time=None, axis=None):
    """Constant-Q estimate on geometric bins: Pa²/Hz or Pa² per bin.

    One kernel per bin, each Q cycles long, so bin width grows with frequency
    — the resolution hearing has, and the one a soundscape spanning decades
    wants. ``scaling`` means what it means on :func:`welch`, except that the
    bandwidth a density divides by is a different number in every bin.

    There is no ``hop``: every bin steps by a quarter of its own kernel, so
    every sample carries equal weight in every bin and a transient reads its
    time-averaged power wherever it falls. Frames whose window reaches past
    the record are dropped, so a low bin averages fewer frames than a high
    one.

    Parameters
    ----------
    data : array_like
        Pressure record (Pa) with time along ``axis``.
    sample_rate : float
        Sample rate (Hz).
    scaling : {'density', 'spectrum'}, optional
        As on :func:`welch`; the bandwidth a density divides by is a
        different number in every bin.
    freq_min, freq_max : float, optional
        First and last bin centre. ``freq_min`` is 20 Hz by default; an unset
        ``freq_max`` ends the ladder at the last bin below the first one whose
        negative-frequency image leaks more than 0.01 dB into a tone's
        level — measured at 0.4525·fs for 12 bins per octave, 0.4795·fs for
        24 and 0.4935·fs for 48 (longer kernels reject the image closer to
        Nyquist). An explicit ``freq_max`` above that is honoured, with a
        warning naming how high those bins read a tone.
    bins_per_octave : int, optional
        Bins per octave, which fixes Q.
    window : str, optional
        The kernel's own taper — its length IS the bin's bandwidth, so
        changing it changes the Q the estimator is named for. Unlike
        :func:`welch` there is no per-scaling default for that reason.
    integration_time : float, optional
        Seconds of record to use, from the start.
    axis : int, optional
        The time axis of a multichannel ``data``, as on :func:`welch`: the
        estimate runs along it and ``power`` carries frequency LAST (one
        spectrum per channel).

    Returns
    -------
    SpectralEstimate
        ``(frequencies, power)`` carrying ``.method='constant_q'``.
    """
    _require_bin_scaling(scaling, "constant_q")
    return _estimate(data, sample_rate, method="constant_q", scaling=scaling,
                     window=window, freq_min=freq_min, freq_max=freq_max,
                     integration_time=integration_time,
                     bins_per_octave=bins_per_octave,
                     who="constant_q", axis=axis)


def sound_exposure(data, sample_rate, *, band_type="decidecade",
                   n_bands=30, nperseg=None, batch_size=None, freq_min=8.9125,
                   freq_max=22387, integration_time=None):
    """Sound exposure per standard band, in Pa²·s — ISO 18405's SEL, linear.

    The energy the record delivered in each band: the statistic for an
    *event* rather than for a state. A pile-driving strike or a passage is
    measured by the energy it puts in the water, not by the power it averages
    over however long you happened to record.

    Parameters
    ----------
    data : array_like
        One channel's pressure record (Pa), 1-D; a 2-D array is refused.
    sample_rate : float
        Sample rate (Hz).
    band_type : str, optional
        The ladder, one of :data:`BAND_TYPES`: ``'decidecade'`` (ISO 18405)
        by default, ``'octave'`` (both IEC 61260-1 base-10) or
        ``'linear'``.
    n_bands : int, optional
        How many equal-width bands ``band_type='linear'`` cuts; ignored by
        the geometric ladders, which are fixed by the standard.
    nperseg : int, optional
        FFT segment length; the sample rate by default, i.e. 1 Hz bins.
    batch_size : int, optional
        How many samples are read at a time, so a multi-hour record never
        materialises as one segment matrix. A whole number of segments by
        default. Memory, not method.
    freq_min, freq_max : float, optional
        The band span. The defaults are the base-10 edges of the nominal
        10 Hz – 20 kHz reporting range.
    integration_time : float, optional
        Seconds of record to integrate, from the start.

    Notes
    -----
    At the default 1 Hz bins the lowest decidecade bands hold only a few FFT
    lines each (:data:`_MIN_LINES_PER_BAND` says what that costs and what
    fixes it); the totals stay exact either way. Each band takes the FFT
    bins it overlaps in proportion to the overlap, so it integrates exactly
    its own width, as :func:`decidecade_band_levels` does.

    This takes no ``window``, ``noverlap``, ``detrend`` or ``average``: a
    band's value is the overlap-weighted sum of the bins it covers, which is
    the band's energy only when every bin's energy is counted once and whole. The boxcar window, zero
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
    if np.ndim(data) != 1:
        # The band estimator sums one record's bins; a 2-D array reached it
        # as rows of "samples" (an IndexError on (channels, n), a plausible
        # but wrong total on (n, channels)).
        raise ConfigurationError(
            f"sound_exposure: data must be one channel (1-D); got shape "
            f"{np.shape(data)}.",
            remediation="Pass one channel at a time, e.g. x[:, ch] for "
                        "read_wav's (n_samples, n_channels) layout.")
    return _estimate(data, sample_rate, scaling="exposure",
                     band_type=band_type, n_bands=n_bands,
                     nperseg=nperseg, batch_size=batch_size, freq_min=freq_min,
                     freq_max=freq_max, integration_time=integration_time,
                     who="sound_exposure")


def _probabilistic_estimate(
        data, sample_rate, *, method="welch", scaling="density",
        segment_duration=1.0, segment_overlap_percent=50, level_step_dB=1.0, level_min_dB=0, level_max_dB=150,
        window=None, freq_min=None, freq_max=None, band_type=None,
        integration_time=None, ref=REFERENCE_PRESSURE_WATER,
        who="probabilistic_welch", axis=None, **options):
    """Probability density of spectral levels over time segments.

    Segments the signal(s), estimates a spectrum per segment and histograms
    the dB levels per frequency. Returns a
    :class:`ProbabilisticSpectralEstimate`. Multichannel input names its
    time axis with ``axis``, under :func:`welch`'s rule: ``None`` is the last
    axis, and a ``(n_samples, n_channels)`` array — ``read_wav``'s layout —
    is refused rather than guessed at. Every channel's segments go into the
    one histogram. A list of 1-D arrays (records of different lengths) is
    taken as it is.

    ``method='constant_q'`` puts the levels on a geometric frequency axis and
    histograms a different population: each sample here is a *Welch average*
    over a ``segment_duration`` chunk (variance shrinks as the chunk holds more
    Welch segments), where constant-Q histograms single unaveraged frames, so
    its level spread is wider for the same signal. It takes the kernel
    options (``freq_min``, ``freq_max``, ``bins_per_octave``) in place of the
    segment ones.

    ``freq_min`` / ``freq_max`` are the frequency range and ``integration_time`` the
    stretch of record the histogram is built from, as on
    :func:`welch`.

    ``segment_overlap_percent`` steps the *time segments*; ``nperseg``/``noverlap`` are the
    Welch parameters *within* a segment. ``nperseg`` is clamped to the segment
    length when ``segment_duration`` is shorter. ``noverlap=None`` (default)
    derives the window's overlap on the (clamped) ``nperseg``, as on
    :func:`welch`; an explicit ``noverlap`` is used as given, unless the
    clamp leaves no room for it (``noverlap >= nperseg``), in which case it
    falls back to that default with a warning.

    ``pdf`` is a probability *density*: each frequency column integrates to 1
    over the level axis (``col.sum() * level_step_dB == 1``), and a level
    never observed is 0. A column whose frequency has no level inside the
    window is ``NaN`` throughout. The plotter draws a 0 bin blank rather
    than as the lowest colour.

    ``mean_dB`` and ``std_dB`` are taken over *all* segments, so they are not
    clipped by ``level_min_dB`` / ``level_max_dB`` the way the histogram is; if levels fall
    outside that window the two stop describing the same population.

    Every level in the result is dB re ``ref**2`` (per Hz, since the default
    ``scaling='density'``), and the result carries ``ref`` back so a consumer
    does not have to assume it. A plot axis or a report that hardcodes the
    package default instead is 120 dB out whenever the caller works in µPa.

    **Complex input** is accepted but returns a two-sided spectrum on an
    unsorted frequency axis, as in :func:`welch`; a
    ``ValidityWarning`` says so.
    """
    # Deferred: bands and cqt import this module at load
    # time, so the two cannot be imported back at the top.
    from uacpy.acoustic_signal.bands import _band_estimate
    from uacpy.acoustic_signal.cqt import _probabilistic_constant_q
    # No ``method`` guard: every caller of this private core passes a literal
    # ('constant_q', or nothing). ``scaling`` is guarded because the band
    # route passes 'exposure' through it, and the doors have already refused
    # that spelling for a bin estimate.
    if scaling not in _SPECTRAL_SCALINGS:
        raise ConfigurationError(
            f"{who}: unknown scaling {scaling!r}.",
            remediation=f"Use one of {_BIN_SCALINGS}, or sound_exposure() "
                        f"for the energy a record delivered.")
    band_options = {k: options.pop(k) for k in _BAND_OPTIONS if k in options}
    level_step_dB = require_positive_finite_scalar(level_step_dB, who, "level_step_dB", " dB")
    if not (np.isfinite(level_min_dB) and np.isfinite(level_max_dB) and level_max_dB > level_min_dB):
        raise ConfigurationError(
            f"{who}: the level window needs finite level_min_dB < level_max_dB (dB); "
            f"got level_min_dB={level_min_dB!r}, level_max_dB={level_max_dB!r}.")
    if not _is_record_list(data):
        data = _time_axis_last(data, axis, who)
    data = _trim_record(data, sample_rate, integration_time, who)
    frequency_range = {k: v for k, v in (("freq_min", freq_min), ("freq_max", freq_max))
                       if v is not None}
    if method == "constant_q":
        # The constant-Q histogram populates from single unaveraged frames
        # rather than from Welch averages over a segment_duration chunk, so it
        # takes the kernel options and not the segment ones. See its own
        # docstring for what that does to the level spread.
        window = _DEFAULT_WINDOW[method][scaling] if window is None else window
        return _probabilistic_constant_q(
            data, sample_rate, window=window, scaling=scaling, who=who,
            level_step_dB=level_step_dB, level_min_dB=level_min_dB, level_max_dB=level_max_dB, ref=ref,
            **frequency_range, **options)
    nperseg = options.get("nperseg", _DEFAULT_NPERSEG)
    noverlap = options.get("noverlap")
    detrend = options.pop("detrend", "constant")
    signals = [require_finite_signal(s, who)
               for s in _split_records(data, who)]
    sample_rate = require_positive_finite_scalar(
        sample_rate, who, "sample_rate", " Hz")
    complex_signal = next((s for s in signals if np.iscomplexobj(s)), None)
    if complex_signal is not None:
        _warn_two_sided(who, complex_signal)

    # Samples in one time segment — the interval each histogram sample
    # describes. Not the band method's ``batch_size``, which is how much
    # of the record is read at a time.
    segment_duration = require_positive_finite_scalar(
        segment_duration, who, "segment_duration", " s")
    segment_samples = int(segment_duration * sample_rate)
    if segment_samples < 1:
        raise ConfigurationError(
            f"{who}: segment_duration ({segment_duration} s) x sample_rate "
            f"({sample_rate} Hz) is {segment_duration * sample_rate:g} samples, "
            "which truncates to an empty time segment; require segment_duration "
            f">= 1/sample_rate ({1.0 / sample_rate:g} s).")
    overlap_samples = int(segment_samples * segment_overlap_percent / 100)
    step = segment_samples - overlap_samples
    if step <= 0:
        raise ConfigurationError(
            f"{who}: segment_overlap_percent ({segment_overlap_percent}) too high — chunks never "
            "advance; require segment_overlap_percent < 100.")

    window = (_BAND_WINDOW if band_type is not None else
              _DEFAULT_WINDOW[method][scaling]) if window is None else window
    level_edges = np.arange(level_min_dB, level_max_dB + level_step_dB, level_step_dB)
    # The Welch segmentation inside each time segment. A banded histogram
    # resolves its own (the band route takes nperseg through, defaulting to
    # one-hertz bins), so this clamp is the bin path's alone.
    nps = min(int(nperseg if nperseg is not None else _DEFAULT_NPERSEG),
              segment_samples)
    if noverlap is None:
        nov = _default_noverlap(window, nps)
    else:
        nov = int(noverlap)
        if nov >= nps:
            warnings.warn(
                f"{who}: noverlap={noverlap} does not fit the Welch segment "
                f"length nperseg={nps} (clamped to the {segment_duration}s "
                f"chunk); using {_default_noverlap(window, nps)} instead.",
                FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
            nov = _default_noverlap(window, nps)
    psd_list = []
    segment_times = []
    bands = None                 # set by the band method, whose values sit
    for sig in signals:          # on standard bands rather than on bins
        for i in range(0, len(sig) - segment_samples + 1, step):
            chunk = sig[i: i + segment_samples]
            segment_times.append((i + segment_samples / 2.0) / sample_rate)
            if band_type is not None:
                # One value per standard band per time segment, so the
                # histogram is over band levels rather than bin levels. The
                # segment is the record the exposure integrates over, which
                # is what makes ``scaling='exposure'`` here a per-segment
                # exposure rather than the whole record's.
                freqs, p, bands = _band_estimate(
                    chunk, sample_rate, scaling=scaling, band_type=band_type,
                    window=window, who=who, **frequency_range,
                    **band_options, **options)
            else:
                freqs, p = _sig.welch(chunk, sample_rate, window=window,
                                      nperseg=nps, noverlap=nov,
                                      detrend=detrend, scaling=scaling)
            psd_list.append(p)

    if len(psd_list) == 0:
        raise ConfigurationError(
            f"{who}: no PSD segments computed; segment_duration="
            f"{segment_duration}s vs signal length="
            f"{len(signals[-1])/sample_rate:.2f}s.")

    psd_array = np.array(psd_list)
    if method == "welch" and band_type is None:
        # Welch resolves every bin whatever range was asked for, so the range
        # crops the answer; the band and kernel methods were built inside it.
        freqs, psd_array = _crop_to_range(freqs, psd_array, freq_min, freq_max)
    psd_segments_dB = power_to_dB(psd_array, ref)
    pdf_matrix, mean_psd, std_psd = level_histogram(psd_segments_dB,
                                                    level_edges)
    if np.all(np.isnan(pdf_matrix)):
        warnings.warn(
            f"{who}: no PSD level falls inside the histogram window "
            f"[level_min_dB={level_min_dB:g}, level_max_dB={level_max_dB:g}] dB — the segments span "
            f"{float(np.min(psd_segments_dB)):.1f} to "
            f"{float(np.max(psd_segments_dB)):.1f} dB re ref² — so pdf is "
            f"all-NaN (mean_dB/std_dB still cover every segment). Widen "
            f"level_min_dB/level_max_dB to cover that span, or pass ref in the data's "
            f"own pressure unit (µPa-scaled samples against the default "
            f"Pa-based ref read 120 dB high).",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    else:
        _warn_levels_outside_window(
            who, list(psd_segments_dB.T), freqs, level_edges)

    # ``ref`` and ``scaling`` ride along because the levels mean nothing
    # without them: the same signal read against a Pascal reference sits
    # 120 dB from these numbers, and 'spectrum' levels are per band where
    # 'density' levels are per hertz. A consumer that has to be told them
    # separately is a consumer that can be told the wrong ones.
    return ProbabilisticSpectralEstimate(
        freqs, level_edges, pdf_matrix, mean_dB=mean_psd, std_dB=std_psd,
        level_step_dB=level_step_dB, segment_duration=segment_duration, ref=float(ref),
        scaling=str(scaling), method=str(method), bands=bands,
        band_type=band_type, segment_levels_dB=psd_segments_dB,
        segment_times=np.asarray(segment_times))


def spectral_centroid(frequencies, weights) -> float:
    """The ``weights``-weighted mean of ``frequencies`` (Hz):
    ``Σ wᵢ fᵢ / Σ wᵢ``, or the plain mean of ``frequencies`` when the
    weights sum to nothing.

    The frequency a band-collapsing reducer pins: a 500 Hz burst weighted by
    its energy spectrum ``|S(f)|²`` onto a 25 Hz-4 kHz axis is a map of
    500 Hz, not of the axis midpoint.

    Parameters
    ----------
    frequencies : array_like
        Frequencies (Hz).
    weights : array_like
        A non-negative weight per frequency (e.g. an energy spectrum).
    """
    band = np.asarray(frequencies, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if total > 0.0:
        return float(np.sum(weights * band) / total)
    return float(np.mean(band))


def level_histogram(levels_dB, level_edges, *, axis=0):
    """The probability density of dB levels over ``level_edges``, with
    their mean and standard deviation: ``(pdf, mean_dB, std_dB)``.

    Each 1-D slice of ``levels_dB`` along ``axis`` is one population (a
    frequency's levels over the segments, as the ``probabilistic_*``
    estimators pass it) and gets its own density column,
    ``np.histogram(..., density=True)``: it integrates to 1 over the levels
    inside ``[level_edges[0], level_edges[-1]]``, and a bin holding no level
    is 0. A population with no level inside the window has no density at
    all: its whole column is ``NaN``.
    ``mean_dB`` and ``std_dB`` are taken over every level of the slice,
    inside the window or not.

    Parameters
    ----------
    levels_dB : array_like
        Levels (dB); the populations run along ``axis``.
    level_edges : array_like, 1-D
        Increasing bin edges (dB).
    axis : int
        The axis each population runs along.

    Returns
    -------
    pdf : ndarray
        ``(len(level_edges) - 1, *other axes)``: one density column per
        population.
    mean_dB, std_dB : ndarray
        The shape of ``levels_dB`` without ``axis``.

    Examples
    --------
    >>> import numpy as np
    >>> pdf, mean, std = level_histogram(np.array([60.0, 61.0, 61.5, 70.0]),
    ...                                  np.arange(60.0, 64.0, 1.0))
    >>> pdf.tolist()
    [0.3333333333333333, 0.6666666666666666, 0.0]
    >>> float(mean), round(float(std), 3)
    (63.125, 4.006)
    """
    levels = np.asarray(levels_dB, dtype=float)
    edges = np.asarray(level_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ConfigurationError(
            f"level_histogram: level_edges must be 1-D, at least 2 long and "
            f"increasing; got shape {edges.shape}.")
    if levels.ndim == 0:
        raise ConfigurationError(
            "level_histogram: levels_dB must have an axis to histogram "
            "along; got a scalar.")
    moved = np.moveaxis(levels, axis, 0)
    columns = moved.reshape(moved.shape[0], -1)
    pdf = np.zeros((edges.size - 1, columns.shape[1]))
    for i in range(columns.shape[1]):
        # density=True over a population with no level inside the window
        # normalises an all-zero count by its zero sum (0/0): the NaN column
        # is the intended "nothing observed" answer, so numpy's
        # RuntimeWarning is suppressed and the estimators name the case.
        with np.errstate(invalid="ignore", divide="ignore"):
            hist, _ = np.histogram(columns[:, i], bins=edges, density=True)
        pdf[:, i] = hist
    return (pdf.reshape((edges.size - 1,) + moved.shape[1:]),
            np.mean(levels, axis=axis), np.std(levels, axis=axis))


def level_percentiles(levels_dB, q, *, axis=0):
    """Percentiles ``q`` (0-100) of dB levels along ``axis``:
    :func:`numpy.percentile`, with ``q`` checked by name. What
    :meth:`ProbabilisticSpectralEstimate.percentiles` returns from a
    ``probabilistic_*`` estimate's ``segment_levels_dB``.

    Parameters
    ----------
    levels_dB : array_like
        Levels (dB).
    q : float or array_like
        Percentiles, 0-100.
    axis : int, optional
        The axis the percentiles run along. Default 0.

    Examples
    --------
    >>> import numpy as np
    >>> level_percentiles(np.array([[60.0], [62.0], [70.0]]), [50, 100]).tolist()
    [[62.0], [70.0]]
    """
    qs = np.asarray(q, dtype=float)
    if not (np.all(np.isfinite(qs)) and np.all((qs >= 0) & (qs <= 100))):
        raise ConfigurationError(
            f"level_percentiles: q must be percentiles in [0, 100]; got "
            f"{q!r}.")
    return np.percentile(np.asarray(levels_dB, dtype=float), qs, axis=axis)


#: Share of a frequency column's levels that may fall outside the histogram
#: window before the estimators warn: each ``pdf`` column is normalised over
#: the levels inside the window, so past this its quantiles describe a
#: visibly truncated population.
_LEVELS_OUTSIDE_WARN_FRACTION = 0.05


def _warn_levels_outside_window(who, columns, freqs, level_edges):
    """Warn when a column loses more than
    :data:`_LEVELS_OUTSIDE_WARN_FRACTION` of its levels to the window.

    ``columns`` holds each frequency's dB levels (one 1-D array per
    frequency). ``np.histogram(density=True)`` normalises a column over the
    levels inside ``[level_edges[0], level_edges[-1]]`` alone, so a column
    that lost 90 % of them still integrates to 1 and every quantile read off
    it describes the surviving 10 %.
    """
    lo, hi = float(level_edges[0]), float(level_edges[-1])
    outside = np.full(len(columns), np.nan)
    for i, col in enumerate(columns):
        col = np.asarray(col)
        col = col[np.isfinite(col)]
        if col.size:
            outside[i] = np.mean((col < lo) | (col > hi))
    lossy = np.flatnonzero(outside > _LEVELS_OUTSIDE_WARN_FRACTION)
    if lossy.size == 0:
        return
    worst = int(lossy[np.argmax(outside[lossy])])
    warnings.warn(
        f"{who}: {lossy.size} of {len(columns)} frequency column(s) have "
        f"more than {100 * _LEVELS_OUTSIDE_WARN_FRACTION:g} % of their levels "
        f"outside the histogram window [{lo:g}, {hi:g}] dB (worst: "
        f"{100 * outside[worst]:.0f} % at {float(freqs[worst]):.4g} Hz). Each "
        f"pdf column is normalised over the levels inside the window only, "
        f"so its quantiles describe that part alone; mean_dB and std_dB "
        f"still cover every level. Widen level_min_dB/level_max_dB to keep them all.",
        NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)


def probabilistic_welch(
        data, sample_rate, *, scaling="density", segment_duration=1.0,
        segment_overlap_percent=50, level_step_dB=1.0, level_min_dB=0, level_max_dB=150,
        nperseg=_DEFAULT_NPERSEG,
        noverlap=None, detrend="constant", window=None, freq_min=None, freq_max=None,
        integration_time=None, ref=REFERENCE_PRESSURE_WATER, axis=None):
    """Probability density of Welch levels, in dB re Pa²/Hz or Pa².

    The McNamara & Buland (2004) soundscape statistic: a Welch estimate per
    time segment, histogrammed per frequency, so a month of recording reads as
    a distribution rather than as one averaged line.

    Parameters
    ----------
    data : array_like or list of array_like
        Pressure record (Pa) with time along ``axis``, or a list of 1-D
        records of different lengths; every channel and record goes into the
        one histogram.
    sample_rate : float
        Sample rate (Hz).
    scaling, nperseg, noverlap, detrend, window, freq_min, freq_max, integration_time
        As on :func:`welch`; ``nperseg`` is clamped to the time segment when
        ``segment_duration`` is shorter.
    segment_duration : float, optional
        Seconds per time segment — the interval each sample of the histogram
        describes.
    segment_overlap_percent : float, optional
        How far consecutive time segments overlap, in percent.
    level_step_dB : float, optional
        Height of one level bin, in dB.
    level_min_dB, level_max_dB : float, optional
        The level window the histogram covers; levels outside it are absent
        from ``pdf`` but still counted in ``mean_dB`` / ``std_dB``. Each
        ``pdf`` column is normalised over the levels inside the window, so
        a warning names any frequency losing more than 5 % of its levels.
    ref : float, optional
        Reference pressure the dB levels are stated against (1 µPa in Pa by
        default), carried on the result so a plot cannot assume another.
    axis : int, optional
        The time axis of a multichannel ``data``, as on :func:`welch`;
        every channel's segments go into the one histogram. A list of 1-D
        records is also accepted.

    Returns
    -------
    ProbabilisticSpectralEstimate
        ``(frequencies, level_edges, pdf)`` carrying ``.mean_dB``,
        ``.std_dB`` and the descriptors; ``.plot()`` draws the histogram.
        A ``pdf`` bin no segment fell in is 0 (``pdf[:, i].sum() *
        level_step_dB == 1``); a frequency with no level inside the window
        is a ``NaN`` column. ``.plot()`` draws empty bins blank.
    """
    _require_bin_scaling(scaling, "probabilistic_welch")
    return _probabilistic_estimate(
        data, sample_rate, scaling=scaling, segment_duration=segment_duration,
        segment_overlap_percent=segment_overlap_percent, level_step_dB=level_step_dB, level_min_dB=level_min_dB, level_max_dB=level_max_dB,
        window=window, freq_min=freq_min, freq_max=freq_max,
        integration_time=integration_time, ref=ref, nperseg=nperseg,
        noverlap=noverlap, detrend=detrend, who="probabilistic_welch",
        axis=axis)


def probabilistic_constant_q(
        data, sample_rate, *, scaling="density", freq_min=20.0, freq_max=None,
        bins_per_octave=24, window="hann", level_step_dB=1.0, level_min_dB=0,
        level_max_dB=150, integration_time=None, ref=REFERENCE_PRESSURE_WATER,
        axis=None):
    """Probability density of constant-Q levels, in dB re Pa²/Hz or Pa².

    The constant-Q analogue of a PPSD, and a different population from it:
    each sample here is a *single unaveraged frame*, where the Welch histogram
    samples an average over a whole time segment. Its spread is therefore
    wider, and ``mean_dB`` sits ``10·γ/ln10`` = 2.51 dB below the power mean
    :func:`constant_q` returns from the same record.

    Parameters
    ----------
    data : array_like or list of array_like
        Pressure record (Pa) with time along ``axis``, or a list of 1-D
        records of different lengths; every channel and record goes into the
        one histogram.
    sample_rate : float
        Sample rate (Hz).
    scaling, freq_min, freq_max, bins_per_octave, window, integration_time
        As on :func:`constant_q`. There is no ``segment_duration`` or
        ``segment_overlap_percent`` here: a constant-Q histogram samples single kernel
        frames, whose length is set per bin by Q, not time segments a caller
        cuts. There is no ``hop`` either: every bin steps by a quarter of its
        own kernel, so every sample carries equal weight in the histogram.
    level_step_dB : float, optional
        Height of one level bin, in dB.
    level_min_dB, level_max_dB : float, optional
        The level window the histogram covers; levels outside it are absent
        from ``pdf`` but still counted in ``mean_dB`` / ``std_dB``. Each
        ``pdf`` column is normalised over the levels inside the window, so
        a warning names any frequency losing more than 5 % of its levels.
    ref : float, optional
        Reference pressure the dB levels are stated against (1 µPa in Pa by
        default), carried on the result so a plot cannot assume another.

    axis : int, optional
        The time axis, as on :func:`welch`.

    Returns
    -------
    ProbabilisticSpectralEstimate
        ``(frequencies, level_edges, pdf)`` with ``.segment_duration = None`` —
        each sample is one kernel frame, whose length is per bin rather than
        one duration for the whole estimate.
    """
    _require_bin_scaling(scaling, "probabilistic_constant_q")
    return _probabilistic_estimate(
        data, sample_rate, method="constant_q", scaling=scaling,
        window=window, freq_min=freq_min, freq_max=freq_max, bins_per_octave=bins_per_octave,
        level_step_dB=level_step_dB, level_min_dB=level_min_dB, level_max_dB=level_max_dB,
        integration_time=integration_time, ref=ref,
        who="probabilistic_constant_q", axis=axis)


def probabilistic_sound_exposure(
        data, sample_rate, *, segment_duration=1.0, segment_overlap_percent=50, level_step_dB=1.0,
        level_min_dB=0, level_max_dB=150, band_type="decidecade", n_bands=30,
        nperseg=None, batch_size=None, freq_min=8.9125, freq_max=22387,
        integration_time=None, ref=REFERENCE_PRESSURE_WATER, axis=None):
    """Probability density of sound exposure levels, in dB re Pa²·s per band.

    How a monitoring record reports what a day of piling or a week of
    passages delivered, rather than what its loudest minute did: each sample
    is ONE ``segment_duration`` segment's energy, so the level axis moves with
    that duration: doubling it adds 3.01 dB to the energy each sample
    integrates. ``mean_dB`` moves slightly more than that — measured +3.07 to
    +3.11 dB — because it is a mean of logs, and a longer segment averages
    more Welch segments and so carries less of that bias.

    Parameters
    ----------
    data : array_like or list of array_like
        Pressure record (Pa) with time along ``axis``, or a list of 1-D
        records of different lengths; every channel and record goes into the
        one histogram.
    sample_rate : float
        Sample rate (Hz).
    band_type, n_bands, nperseg, batch_size, freq_min, freq_max, integration_time
        As on :func:`sound_exposure`, which also explains why there is no
        window, overlap, detrending or averaging to set.
    segment_duration : float, optional
        Seconds per time segment — here, the record each exposure integrates,
        which is why the level axis moves with it.
    segment_overlap_percent : float, optional
        How far consecutive time segments overlap, in percent.
    level_step_dB : float, optional
        Height of one level bin, in dB.
    level_min_dB, level_max_dB : float, optional
        The level window the histogram covers; levels outside it are absent
        from ``pdf`` but still counted in ``mean_dB`` / ``std_dB``. Each
        ``pdf`` column is normalised over the levels inside the window, so
        a warning names any frequency losing more than 5 % of its levels.
    ref : float, optional
        Reference pressure the dB levels are stated against (1 µPa in Pa by
        default), carried on the result so a plot cannot assume another.

    axis : int, optional
        The time axis of a multichannel ``data``, as on :func:`welch`;
        every channel's segments go into the one histogram.

    Returns
    -------
    ProbabilisticSpectralEstimate
        ``(frequencies, level_edges, pdf)`` carrying ``.bands`` and the
        descriptors; ``.plot()`` draws the histogram.
    """
    _check_band_type(band_type, "probabilistic_sound_exposure")
    return _probabilistic_estimate(
        data, sample_rate, scaling="exposure", band_type=band_type,
        n_bands=n_bands, nperseg=nperseg, batch_size=batch_size,
        segment_duration=segment_duration, segment_overlap_percent=segment_overlap_percent, level_step_dB=level_step_dB,
        level_min_dB=level_min_dB, level_max_dB=level_max_dB, freq_min=freq_min, freq_max=freq_max,
        integration_time=integration_time, ref=ref,
        who="probabilistic_sound_exposure", axis=axis)
