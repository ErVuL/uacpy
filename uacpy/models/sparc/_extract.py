"""What one SPARC launch returns, made into the result: the receiver of
each launch, the ``.rts`` time-grid check, the looped traces stacked onto one
``p(z, r, t)``, the package's unit-source level, and the check that the output
window was long enough."""

import warnings
from typing import Optional, Tuple

import numpy as np

from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.exceptions import (
    ConfigurationError, ModelExecutionError, NumericsWarning,
)
from uacpy.core.receiver import Receiver
from uacpy.core.results import Field
from uacpy.io.oalib_reader import RtsFile
from uacpy.models._conventions import _source_density
from uacpy.models._knobs import positive_finite
from uacpy.models._extract import (
    RECORD_TAIL_FRACTION, RECORD_TAIL_LEVEL, mask_zero_range_columns,
    record_tail_ratio, result_kwargs,
)
from uacpy.models._launch import attach_prt_tail
from uacpy.models.sparc._plan import _LOOPED_TIME_SERIES_MODES

# The factor, with the division by rho(z_s), that brings SPARC's field onto
# the package's point source of unit amplitude at 1 m. Scooter forces the
# depth equation with 2/rhoSz (``Scooter/scooter.f90:662-663``), whose Hankel
# inverse is e^{ikR}/R. SPARC adds the bare nodal load ``deltat2 * ST`` split
# by (1 - ws) and ws (``Scooter/sparc.f90:529-535``): the source
# -S(t) delta(z - z_s) of the Hankel-transformed time-marched FFP, JKPS
# section 8.3.2 eq. (8.36), whose field is s(t - R/c) / (2R) for rho = 1.
_UNIT_SOURCE_GAIN = 2.0


def _output_times(time_max: float, n_time_samples: int) -> np.ndarray:
    """The output times the binary samples: the deck's ``0.0 time_max /`` pair
    (written with six decimals) expanded to ``n_time_samples`` uniform points by
    SubTab (``misc/subtabulate.f90:27-28``), each field sample interpolated
    onto its time exactly (``sparc.f90:572-577``)."""
    return np.linspace(0.0, float(f"{time_max:.6f}"), int(n_time_samples))


def launch_receiver(inputs):
    """``(receiver, value)`` of launch ``inputs.launch``: the looped axis
    passed one value at a time and the held one whole
    (:data:`_LOOPED_TIME_SERIES_MODES`), or the whole receiver for a
    snapshot (``value`` ``None``)."""
    engine = inputs.settings.engine
    if engine.output_mode == 'S':
        return inputs.receiver, None
    spec = _LOOPED_TIME_SERIES_MODES[engine.output_mode]
    value = float(np.atleast_1d(np.asarray(
        getattr(inputs.receiver, spec['loop_attr']),
        dtype=float))[inputs.launch])
    return Receiver(**{
        spec['loop_attr']: np.array([value]),
        spec['held_attr']: getattr(inputs.receiver, spec['held_attr']),
    }), value


def rts_time_matches(parsed, deck_time) -> bool:
    """Whether an ``.rts`` time column is ``deck_time`` as written.

    ``sparc.f90:294,299`` writes each time with ``G15.6`` — six
    significant digits — so a label may sit up to half a unit in its
    sixth digit off the true sample time, plus the REAL*4 rounding of
    ``tout``. Anything further off is a different grid (a stale or
    foreign file, or a run that sized its own window).
    """
    parsed = np.asarray(parsed, dtype=float)
    if parsed.shape != deck_time.shape:
        return False
    mag = np.abs(deck_time)
    exponent = np.floor(np.log10(np.where(mag > 0.0, mag, 1.0)))
    half_digit = np.where(mag > 0.0, 0.5 * 10.0 ** (exponent - 5), 0.0)
    tol = half_digit + 4.0 * np.finfo(np.float32).eps * mag + 1e-12
    return bool(np.all(np.abs(parsed - deck_time) <= tol))


def require_deck_time_grid(rts_data, deck_time, work_dir, run_base, *,
                           model_name):
    """Every launch of a loop must land on the deck's output time grid:
    the per-run traces are stacked onto that one ``time`` coordinate, so
    a divergence would silently mislabel every trace."""
    if rts_time_matches(rts_data.times, deck_time):
        return
    got = np.asarray(rts_data.times, dtype=float)
    span = (float(got[-1] - got[0]) if got.size else 0.0)
    ref_span = (float(deck_time[-1] - deck_time[0]) if deck_time.size
                else 0.0)
    exc = ModelExecutionError(
        model_name, return_code=0, stdout=None,
        stderr=(
            f"SPARC run {run_base} returned an output time grid "
            f"(n={got.size}, span={span:.6g} s) that differs from the "
            f"deck's (n={deck_time.size}, span={ref_span:.6g} s); the "
            f"per-run traces share one time axis and would be mislabelled."
        ),
    )
    attach_prt_tail(exc, work_dir, run_base)
    raise exc


def stack_traces(inputs, runs, *, band, model_name, provenance):
    """The received time series for ``output_mode='R'`` and ``'D'``, on
    the marched pulse band ``band``.

    ``sparc.f90`` writes one slice per run — one receiver depth in
    horizontal mode, one receiver range in vertical mode
    (``sparc.f90:593-606`` accumulates ``RTSrz(ir, Itout)``, a time series
    at each *depth* for a fixed range) — so the wrapper looped over that
    axis and stacks the traces here. The two modes are the same procedure
    on transposed axes; :data:`_LOOPED_TIME_SERIES_MODES` carries every
    difference between them.
    """
    settings = inputs.settings
    engine = settings.engine
    mode = engine.output_mode
    spec = _LOOPED_TIME_SERIES_MODES[mode]
    loop_values = np.atleast_1d(np.asarray(
        getattr(inputs.receiver, spec['loop_attr']), dtype=float))
    # ``runs[0].positions`` is the file's own second axis: the output
    # ranges in horizontal mode, the output DEPTHS in vertical mode (see
    # ``other_coord`` in the table).
    other_axis = np.asarray(runs[0].positions, dtype=float)
    # The time coordinate is the deck's uniform grid, not the file's
    # six-digit labels, which are uneven by up to half a unit in their
    # last digit and would make every SPARC trace fail Field's
    # uniform-spacing contract.
    time = _output_times(engine.time_max, engine.n_time_samples)
    # rts_data.pressure is (nt, n_other) here; want (n_other, nt). The
    # ``scale`` divides out the sqrt(pi) the 'R' branch is hot by.
    traces = [np.asarray(rts_data.pressure).T * spec['scale']
              for rts_data in runs]

    # (n_depth, n_range, n_time) — the shared Field contract, in that
    # order whichever axis was looped. The range axis is SPARC's actual
    # output grid; Field validates its length against the data shape.
    axes = {spec['loop_coord']: loop_values,
            spec['other_coord']: other_axis}
    return Field(
        data=mask_zero_range_columns(
            model_name,
            np.stack(traces, axis=spec['stack_axis']), axes['range'],
            spec['mask_reason']),
        coords={'depth': axes['depth'], 'range': axes['range'],
                'time': time},
        band=band,
        **result_kwargs(
            model_name, provenance, inputs.source,
            backend='sparc',
            frequencies=settings.frequencies,
            phase_reference=settings.output.phase_reference,
        ),
    )


def scale_to_unit_source_level(result, env, source) -> None:
    """Bring ``result`` onto the package's unit-source level: multiply
    by :data:`_UNIT_SOURCE_GAIN` / ``rho(z_s)``.

    Every element mass and stiffness matrix of SPARC's finite-element
    march carries ``1/rho`` (``sparc.f90:434-456``) while the source term
    added to ``U2`` carries neither ``1/rho`` nor the factor 2
    (``sparc.f90:529-535``), so the binary returns ``rho(z_s)/2`` times
    the field of the package's unit source. Scooter forces with
    ``2/rhoSz`` (``scooter.f90:662-663``), and Kraken's field.exe output
    is divided by the same ``_source_density``, so one environment
    gives one level on every engine. Measured on a 100 m rigid guide at
    50 Hz, moving ``water_density`` from 1.0 to 1.027 raised the raw
    SPARC peak by 20*log10(1.027) = 0.230 dB while Scooter and Kraken
    moved by less than 1e-6 dB; against an exact image sum on a 100 m
    rigid guide the raw field sits at half the unit-source pressure
    (-6.05 dB) on every output mode.

    The model runs one source depth per launch (the base class splits a
    multi-depth ``Source``), so one ``rho(z_s)`` covers the whole field.
    """
    depth = float(np.atleast_1d(np.asarray(source.depths, dtype=float))[0])
    result.data = (_UNIT_SOURCE_GAIN * np.asarray(result.data)
                   / _source_density(env, depth))


def warn_on_truncated_window(result, *, pinned_by: Optional[str] = None
                             ) -> None:
    """Warn when p(t) is still ringing at the end of the output window.

    ``pinned_by`` names where a caller-set window came from
    (``'SPARC(time_max=…)'``, ``'run(output_duration=…)'``), so the advice
    is about that window; ``None`` is the automatic one.

    ``time_max`` is sized from the direct travel time
    (``SPARC._resolve_engine_settings``), which does not bound the last
    arrival: in a waveguide that is set by the slowest modal *group*
    velocity, which falls to zero at cutoff, and SPARC's vacuum / rigid
    boundaries leave the near-cutoff tail undamped. An arrival past
    ``time_max`` is absent from p(t) with nothing in the output marking it —
    so measure the tail that did come back
    (:func:`~uacpy.models._extract.record_tail_ratio`: the peak over the
    final tenth of the window against the peak over the whole of it) and
    say so while it is within ``RECORD_TAIL_LEVEL`` of the peak. An all-NaN
    window (every receiver masked) is not a finding.
    """
    time = np.asarray(result.coords.get('time', ()), dtype=float)
    ratio = record_tail_ratio(time, result.data)
    if ratio is None or ratio < RECORD_TAIL_LEVEL:
        return
    if pinned_by is None:
        cause = ("The auto window is 2.5 direct travel times, which does not "
                 "bound the slow modal tail of a waveguide — raise it with "
                 "SPARC(time_max=...)")
    else:
        cause = (f"The window is the one {pinned_by} set — lengthen it "
                 f"there")
    warnings.warn(
        f"SPARC TIME_SERIES: p(t) is still at {100.0 * ratio:.0f}% of "
        f"its peak over the last {100.0 * RECORD_TAIL_FRACTION:.0f}%"
        f" of the [0, {float(time[-1]):.4g}] s output window, so arrivals "
        f"past it are missing from the returned trace. {cause} (and keep "
        f"n_time_samples / rmax_factor in step with the longer window).",
        NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )


def rts_to_pressure(
    rts: RtsFile, frequency: float,
    *, pulse_type: Optional[str] = None,
    pulse_frequency: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project SPARC time-series data onto complex pressure at one frequency.

    Without ``pulse_type``, it evaluates the Hann-windowed transform **at**
    ``frequency`` — :func:`~uacpy.acoustic_signal.tone_phasor`,
    ``A = 2/Σw · Σ x_n w_n e^{-2πi f t_n}``, the same estimator
    :meth:`~uacpy.Field.extract_tone` uses — and returns
    ``(p_at_freq, ranges)``: the complex amplitude of a steady tone at
    ``frequency`` in the record. That is the field's complex pressure only
    when the record holds a steady tone. A SPARC ``.rts`` is the response to
    the transient pulse SPARC marches, so this value is the pulse response
    weighted by the window where the arrival sits and divided by the
    window's sum: it changes with the record length and the arrival time,
    and the source spectrum is not divided out (measured: one Ricker-shaped
    arrival at 0.2 s in records of 0.5 / 1.0 / 2.0 s, fs 2 kHz, gives
    -24.4 / -38.8 / -56.0 dB at 50 Hz). Pass ``pulse_type`` to divide the
    source pulse's spectrum out.

    A ``.rts`` picks its own ``nt`` and ``dt``, so ``frequency`` is
    essentially never on an rfft bin; the nearest bin is off by -0.056 dB
    and 18 deg at a tenth of a bin and -1.418 dB and 89.8 deg at half of
    one, against this estimator. With ``pulse_type=`` the rectangular
    transform is evaluated at ``frequency`` on both the record and the
    known source pulse, and their ratio is returned (see the note there).
    Both evaluate on the file's own ``time`` axis, so a record whose output
    times start after 0 keeps its phase.

    The canned pulse is shaped by its own characteristic frequency ``F``,
    the deck frequency of the SPARC run (``sparc.f90:525``; a SPARC
    result's ``run_settings.engine.deck_frequency``), which the ``.rts``
    does not record: pass it as ``pulse_frequency``. ``None`` takes
    ``frequency``, right only when projecting at ``F`` — elsewhere the
    record is divided by another pulse's spectrum (measured on a 50 Hz
    'PN+B' run: 0.65 dB and 36° off at 40 Hz). The deconvolution also needs
    the record to hold the whole response: a record still ringing at its
    end breaks the convolution theorem it rests on.

    A post-processing utility for a ``.rts`` read by
    :func:`uacpy.io.read_rts_file`;
    :class:`uacpy.models.SPARC` returns ``p(t)`` and does not call it. Even
    deconvolved, the projection is not a calibrated substitute for
    Kraken/Scooter TL: ``sparc.f90:313`` sets ``Atten = 0`` (no stabilizing
    attenuation; Scooter uses ``Atten = Deltak``, ``scooter.f90:129``),
    ``Nk`` is sized across the whole pulse band, and the default
    ``'PN+B'`` band-pass is applied per wavenumber while the deconvolution
    divides by one scalar ``S(ω0)``. Measured on a guide with exactly one
    propagating mode, the CW TL taken from SPARC this way is 2.4 dB off in
    median with 13 dB excursions (Scooter: 0.07 dB) and converges under no
    grid setting.
    """
    p = rts.pressure
    times = rts.times
    ranges = rts.positions

    nt = p.shape[0]
    # A run that wrote a single output time has no second sample to
    # difference against: its step is 0.
    dt = times[1] - times[0] if nt > 1 else 0.0

    # Both branches below estimate the tone at ``frequency`` from the
    # record on its own output times (``tone_phasor``), which takes at least
    # two of them: one sample holds no frequency information.
    if nt < 2 or not float(dt) > 0.0:
        raise ConfigurationError(
            f"rts_to_pressure: the .rts holds {nt} time step(s) at dt={dt!r}, "
            f"so there is no time series to estimate the {frequency} Hz "
            f"tone from — the estimate needs at least two output times, "
            f"in increasing order.",
            remediation="Re-run SPARC with more than one output time (a "
                        "larger n_time_samples / shorter output interval).",
        )

    positive_finite('rts_to_pressure: pulse_frequency (Hz)', pulse_frequency,
                    optional=True)
    if pulse_type is not None:
        # Deconvolve the known source spectrum (convolution theorem): the range
        # time-series r(t) = s(t) ⊛ h(t), so rfft(r)/rfft(s) = h(w0) — the CW
        # transfer function ≈ absolute TL re 1 m (Jensen COA Eq. 8.1). uacpy
        # generated the pulse, so s(t) is known. The estimate is physical and
        # window/grid-independent once the output Nyquist clears the pulse band
        # (the SPARC model sizes n_time_samples for this), but SPARC's discretised
        # pulse and band-pass leave a frequency-dependent bias of a few dB vs
        # Kraken/Scooter — it is not a calibrated replacement for them.
        # Use the RECTANGULAR DFT (no taper): a window breaks the convolution
        # theorem and would null the transient source pulse (first few samples).
        # Imported here rather than at module level: sparc_pulse pulls scipy
        # in, and only this deconvolution path needs it.
        from uacpy.acoustic_signal.generate import sparc_pulse
        from uacpy.acoustic_signal.spectrum_at import _tone_phasor
        t = np.asarray(rts.times, dtype=float)
        s_t, _ = sparc_pulse(
            t, frequency if pulse_frequency is None else pulse_frequency,
            pulse_type[0])
        # Both sides evaluated AT ``frequency``, with NO taper: the
        # rectangular transform is what the convolution theorem needs (a
        # window would null the transient source pulse in the first few
        # samples), and evaluating at the frequency rather than at the
        # nearest bin makes the ratio exact off-bin.
        #
        # Taking the same bin on both sides does NOT cancel the leakage,
        # although it nearly does and a constant H cannot show the
        # difference — on a constant H the ratio is exact at every offset
        # by construction. Against a three-path 60 ms channel the bin
        # ratio drifts to +0.06 dB and +6.0 deg at half a bin, while the
        # pair below stays at 0.0000 dB and 0.000 deg.
        # The DTFT of a transient at ``frequency`` is its spectrum there:
        # no steady tone, so no negative-frequency image to solve for.
        S_at_f0 = _tone_phasor(s_t, t, frequency, window='boxcar',
                              who="rts_to_pressure", steady_tone=False)
        if S_at_f0 == 0:
            raise ConfigurationError(
                "rts_to_pressure: source spectrum is zero at "
                f"{frequency} Hz for pulse_type={pulse_type!r}; cannot "
                "deconvolve (check pulse / frequency).")
        return (_tone_phasor(p, t, frequency, window='boxcar', axis=0,
                            who="rts_to_pressure", steady_tone=False)
                / S_at_f0), ranges

    # The Hann-weighted tone fit AT `frequency` on the file's own output
    # times (sparc.f90:159 reads them with ReadVector, so they may start
    # anywhere); tone_phasor solves for the tone's negative-frequency image,
    # so a steady tone returns its own amplitude and phase. Deferred:
    # acoustic_signal pulls scipy, and uacpy's public surface is imported
    # without it (test_lazy_imports).
    from uacpy.acoustic_signal.spectrum_at import _tone_phasor
    p_at_freq = _tone_phasor(p, np.asarray(rts.times, dtype=float),
                             frequency, window='hann', axis=0,
                             who="rts_to_pressure")
    return p_at_freq, ranges
