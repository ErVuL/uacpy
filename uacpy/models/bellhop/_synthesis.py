"""BROADBAND and TIME_SERIES from the arrivals of one ray trace at the
carrier: the delay-and-sum of a source pulse (:meth:`Arrivals.to_time_series
<uacpy.core.results.Arrivals.to_time_series>`, TIME_SERIES)
and the transfer function the arrivals sum to (BROADBAND), with the notices
of what one trace at ``fc`` cannot carry across the band."""

import warnings
import numpy as np

from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.absorption import (
    ConstantAbsorption, warn_if_band_absorption_frozen)
from uacpy.core.results import Field, PhaseReference, SoundSpeeds
from uacpy.core.run_settings import RunMode
from uacpy.models._extract import result_kwargs
from uacpy.models.base import StageInputs
from uacpy.models.bellhop._output import (
    mask_paired_receivers_below_seafloor, restore_broadband_depth_axis,
)
from uacpy.models.bellhop._settings import _BAND_FROM_CALL, _BAND_FROM_SOURCE
from uacpy.models.bellhop._tables import grid_is_paired
from uacpy.core.exceptions import NumericsWarning


def arrivals_run_settings(settings, source, *, output):
    """The settings the arrivals run behind a BROADBAND / TIME_SERIES call
    ran with: ``settings`` as an ARRIVALS run at the one carrier
    ``settings.engine.center_frequency``, every depth of ``source`` in one
    deck, unweighted, with ``output`` (the model's ARRIVALS output)."""
    return settings._replace(
        mode=RunMode.ARRIVALS,
        frequencies=np.array([settings.engine.center_frequency]),
        source_depths=np.atleast_1d(np.asarray(source.depths,
                                               dtype=float)),
        depth_loop='single', source_weights=None, weights_applied=False,
        time=None, output=output)


def synthesise(inputs: StageInputs, arr_field, *, model_name, provenance,
               backend, grid_type, log):
    """Stage 5 of BROADBAND / TIME_SERIES: the synthesis built from the
    arrivals one ray trace at ``fc`` returned (``arr_field``), as the
    Bellhop User Guide (Sec 9) describes:

    1. **BROADBAND**: the frequency-domain transfer function H(f) on
       ``settings.frequencies`` from the arrivals → ``Field``. Use
       ``Field.to_time_trace()`` (raw IFFT) or
       ``Field.synthesize_time_series(source_waveform, sample_rate)``
       (windowed convolution) downstream.

    2. **TIME_SERIES**: a delay-and-sum convolution of the source pulse
       with each arrival (amplitude, phase, delay) → ``Field`` with data
       shape (n_depths, n_ranges, n_samples) and metadata carrying
       'time', 'dt', 'fs'. Its window is ``output_duration`` long and
       opens at ``t_start``, else just before the earliest arrival — the
       placement the wave-equation solvers use — and, when
       ``output_duration`` is unset, it closes after the latest arrival
       plus twice the pulse.

    This is a key advantage of ray tracing: the ray geometry (paths,
    travel times) is frequency-independent, so a single arrivals
    calculation at fc provides the impulse response. ``arr_field`` is
    kept on ``components['arrivals']``, stamped by the caller as the
    ARRIVALS result it is (:func:`arrivals_run_settings`).
    """
    settings = inputs.settings
    engine = settings.engine
    run_mode = settings.mode
    env, source, receiver = inputs.env, inputs.source, inputs.receiver
    fc = engine.center_frequency
    time = settings.time
    if run_mode == RunMode.TIME_SERIES:
        source_waveform = np.array(
            time.source_waveform[:engine.pulse_samples], dtype=float)
        sample_rate = time.sample_rate
        output_duration = time.output_duration
        # ``output_duration`` is the record's LENGTH and says nothing
        # about where it starts; ``t_start`` places it. Unset, the start
        # is just before the earliest arrival (below), as the IFFT
        # engines place theirs — an ``output_duration`` anchored at
        # emission would be all zeros for any receiver beyond
        # c·output_duration.
        effective_time_window = (float(output_duration)
                                 if output_duration is not None else None)
        effective_t_start = time.t_start

    # The synthesised result is built entirely from these arrivals, so it
    # inherits their file paths — the only handle on the scratch dir that
    # survives when ``cleanup=False`` — and their components (the
    # in-memory Bounce table a routed seabed attached, which the
    # constructor docstring promises on ``result.components['bounce']``),
    # beside the arrivals themselves.
    arr_paths = {
        key: value for key, value in arr_field.metadata.items()
        if key.endswith('_file')
    }
    components = {**arr_field.components, 'arrivals': arr_field}

    arrivals_by_rcv = arr_field.by_receiver
    rz = arr_field.receiver_depths
    rr = arr_field.receiver_ranges  # in meters

    # ``ArrMod.f90:101-102`` writes ``NRz_per_range`` depth blocks, which
    # ``bellhop.f90:202-206`` sets to 1 for an irregular grid: the entries
    # of that single block are the paired receivers (Rz(i), Rr(i)). The
    # depth axis therefore collapses onto the range axis, and the paired
    # depths ride on ``aux_coords['receiver_depth']``, which
    # ``mask_paired_receivers_below_seafloor`` states — the same shape
    # ``read_shd_file`` gives the TL path.
    irregular = grid_is_paired(grid_type)
    nrd = 1 if irregular else len(rz)
    nrr = len(rr)

    def _emit(data, axis_name, axis_values):
        coords = {'range': np.asarray(rr, dtype=float),
                  axis_name: axis_values}
        if not irregular:
            coords = {'depth': np.asarray(rz, dtype=float), **coords}
        return (data[0] if irregular else data), coords

    # ── Path A: time-domain delay-and-sum with source waveform ──
    if run_mode == RunMode.TIME_SERIES:
        log(f"Delay-and-sum over {nrd}×{nrr} receiver grid")
        if settings.frequencies is not None:
            warn_if_attenuation_extrapolates(
                env, settings.frequencies, fc, model_name=model_name)

        # One clock for the whole grid, spanning every cell's arrivals.
        t_vec, data = arr_field.to_time_series(
            source_waveform, sample_rate,
            time_window=effective_time_window,
            t_start=effective_t_start,
            who=f"{model_name}.run(run_mode=TIME_SERIES)")
        t_start_locked = float(t_vec[0])
        n_t = len(t_vec)

        data, coords = _emit(data, 'time', t_vec)
        # The stamped frequency axis is the band the synthesised p(t)
        # represents, derived from the (padded) source waveform the same
        # way the IFFT-based engines derive their broadband grid — so a
        # TIME_SERIES result names its band identically across engines.
        # The ray-trace carrier fc stays on metadata['center_frequency'].
        stamp_freqs = settings.frequencies
        field = Field(
            data=data,
            coords=coords,
            phase_reference=PhaseReference.TIME_DOMAIN_NATIVE,
            **result_kwargs(
                model_name, provenance, source, backend=backend,
                frequencies=stamp_freqs if stamp_freqs is not None else fc,
                dt=1.0 / sample_rate, fs=sample_rate, nt=n_t,
                t_start=t_start_locked, center_frequency=fc,
                components=components, **arr_paths,
            ),
        )
        # The same no-data rule as Path B and the TL modes: a paired
        # receiver under the seabed is clamped onto it by BELLHOP, so its
        # trace is another receiver's, not a buried one's.
        if irregular:
            return mask_paired_receivers_below_seafloor(
                field, receiver, env, model_name=model_name)
        return restore_broadband_depth_axis(field, receiver, env,
                                            model_name=model_name)

    # ── Path B: frequency-domain transfer function ──
    # A grid the package expands from a lone carrier is the package's
    # choice; one passed in, or a multi-frequency Source that IS the
    # band, is the caller's.
    grid_was_derived = engine.band_origin not in (_BAND_FROM_CALL,
                                                  _BAND_FROM_SOURCE)
    frequencies = settings.frequencies
    n_freq = len(frequencies)
    warn_if_attenuation_extrapolates(env, frequencies, fc,
                                     model_name=model_name)
    if grid_was_derived:
        warn_if_default_grid_folds(
            arrivals_by_rcv, nrd, nrr, frequencies, fc,
            model_name=model_name)

    # H(d, r, f) over every receiver cell (NaN where no arrival reached),
    # trailing-axis convention; a paired grid's one block carries its
    # receivers on the range axis.
    H = np.asarray(arr_field.transfer_function(frequencies).data).reshape(
        nrd, nrr, n_freq)

    log(f"Built transfer function "
        f"({nrd} depths x {nrr} ranges x {n_freq} freqs)")

    # Sea-surface sound speed of the first profile: ``ssp.sound_speed`` is
    # (n_depths, n_ranges) of speeds alone, the depths living on
    # ``ssp.depths``. Carried on the result for ``Field.to_time_trace`` /
    # ``Field.synthesize_time_series``, which use it as the reference speed
    # that anchors the synthesis window to r/c.
    c0 = float(env.ssp.sound_speed[0, 0])
    # The fastest water speed, which those two readers anchor on when
    # it is faster than c0 (max(c_max, c0)), so an upward-refracting
    # column or a range-dependent one whose fastest water is not at the
    # surface opens the window before its first arrival. The water
    # alone: Bellhop traces rays in the water and has no head wave
    # through the seabed, so the seabed speeds the run's
    # ``run_settings.waveguide.c_max`` counts for the wave-equation
    # engines would open the window early for nothing. The stamp takes
    # precedence over that waveguide in the readers.
    c_max = float(np.max(env.ssp.sound_speed))

    H, coords = _emit(H, 'frequency', frequencies)
    field = Field(
        data=H,
        coords=coords,
        phase_reference=PhaseReference.TRAVELLING_WAVE,
        speeds=SoundSpeeds(surface=c0, water_max=c_max),
        **result_kwargs(
            model_name,
            provenance,
            source,
            backend=backend,
            frequencies=frequencies,
            center_frequency=fc,
            components=components,
            **arr_paths,
        ),
    )
    if irregular:
        return mask_paired_receivers_below_seafloor(
            field, receiver, env, model_name=model_name)
    return restore_broadband_depth_axis(field, receiver, env,
                                        model_name=model_name)


def warn_if_attenuation_extrapolates(env, frequencies, fc: float, *,
                                     model_name) -> None:
    """Report the volume-attenuation error the single-trace band incurs.

    The arrival set comes from one trace at ``fc``: ``Step.f90:73``
    accumulates ``tau += hw/CMPLX(c, cimag)`` with
    ``cimag = alphaT*c**2/omega`` (``misc/AttenMod.f90:113``), so
    ``2*pi*fc*Im(tau) = -integral of alpha(fc, s) ds``. A constant
    dB/wavelength is linear in ``f`` and scales exactly. Thorp and one
    Francois-Garrison water row are rescaled by their surface ratio
    ``alpha(f)/alpha(fc)``
    (:func:`~uacpy.core.absorption.arrival_absorption_exponent`): exact for
    Thorp, and off by the pressure terms' bend of the ratio with depth for
    Francois-Garrison — 0.058 dB/km at worst over 5-15 kHz in a 5 km column,
    where the linear line misses by 0.54. The ratio is anchored at the
    surface, the depth the law's own curve (``table`` with no depths) is
    drawn at; the trace carries no water depth to anchor it elsewhere. Every other law — a :class:`Biological` resonance
    confined in a depth band, a Francois-Garrison profile whose frequency
    dependence follows the local water, a tabulated α(f, z) — is scaled
    linearly in ``f`` from the ``Im(tau)`` the rows' ``alphaI`` gave at
    ``fc``. The shared band check
    (:func:`~uacpy.core.absorption.warn_if_band_absorption_frozen`)
    measures the scaling applied against the law over the whole band and
    water column, so a resonance inside the band, a layer below the surface
    and the depth bend of a ratio are all seen; Thorp measures zero.

    ``fc`` is the carrier the single trace was run at, taken from the
    source. Re-deriving it as the band's midpoint holds only for a band
    `_band.broadband_band` built around that carrier: an
    explicit off-centre ``frequencies=`` moves the midpoint away from it
    (8-16 kHz around a 10 kHz source reads 12 kHz), which names a
    frequency no ray was traced at.
    """
    absorption = env.absorption
    if absorption is None or isinstance(absorption, ConstantAbsorption):
        return
    by_ratio = absorption._scales_by_frequency_ratio
    applied = (f"the {absorption._short()} absorption is scaled from it by "
               f"the law's ratio alpha(f)/alpha(fc) at the surface, which "
               f"the pressure terms bend with depth"
               if by_ratio else
               f"a {absorption._short()} absorption that varies with depth "
               f"does not scale from one frequency by one ratio, so it is "
               f"applied linearly in frequency")
    warn_if_band_absorption_frozen(
        model_name, absorption, frequencies, float(fc),
        water_depth=float(env.depth), by_ratio=by_ratio,
        mechanism=(
            f"the arrival set is traced once at {float(fc):.4g} Hz, and "
            f"{applied} (Step.f90:73 with cimag = alphaT*c^2/omega, "
            f"misc/AttenMod.f90:113)."),
        remediation=("Run each frequency separately (ARRIVALS per f), or "
                "narrow the band, if the band edges matter."))


def warn_if_default_grid_folds(arrivals_by_rcv, nrd: int, nrr: int,
                               frequencies, fc: float, *,
                               model_name) -> None:
    """Say what the package-chosen broadband grid folds, per these arrivals.

    A transfer function sampled every Δf synthesises to a record 1/Δf
    long, and the default grid — ``DEFAULT_BROADBAND_N_FREQS`` bins over
    ``fc·(1 ± bw/2)`` — sets Δf from the carrier alone: ``fc/254`` for the
    default band, a record ``254/fc`` s long with no reference to how long
    this channel rings. The arrivals are in hand here, so the fold is
    measured rather than presumed, with the yardstick
    ``Arrivals.synthesis_band`` uses: an arrival later than its own cell's
    first by more than the record lands back on the early trace. Only for
    a grid the package chose — a caller who passed ``frequencies=`` made
    that decision, and ``Arrivals.synthesis_band`` reports on the grid it
    builds already.
    """
    from uacpy.acoustic_signal.delay_profile import _fold_notice
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.size < 2:
        return
    record = 1.0 / float(np.mean(np.diff(frequencies)))
    omega_c = 2.0 * np.pi * float(fc)
    delays, power, first = [], [], []
    for ird in range(nrd):
        for irr in range(nrr):
            cell = arrivals_by_rcv[0][ird][irr]
            if int(cell.get('n_arrivals', 0)) == 0:
                continue
            d = np.asarray(cell['delays'], dtype=float)
            # Received amplitude: the column times the volume-attenuation
            # factor Bellhop keeps in the imaginary travel time.
            a = (np.abs(np.asarray(cell['amplitudes'], dtype=float))
                 * np.exp(omega_c * np.asarray(cell['delays_imag'],
                                               dtype=float)))
            delays.append(d)
            power.append(a ** 2)
            first.append(np.full(d.size, float(d.min())))
    if not delays:
        return
    notice = _fold_notice(
        np.concatenate(delays), np.concatenate(power), record,
        first=np.concatenate(first),
        who=f"{model_name}.run(run_mode=BROADBAND)",
        remediation=(f"The grid was the default — {frequencies.size} bins over "
                f"the band, Δf = {1.0 / record:.4g} Hz — chosen from the "
                f"carrier alone. Pass frequencies= to set it from the "
                f"channel: Arrivals.synthesis_band(bandwidth=..., "
                f"centre=...) on an ARRIVALS run of this geometry sizes "
                f"the record to hold the energy, and Arrivals.window / "
                f"top_n_by_amplitude drop the tail outright instead of "
                f"folding it."))
    if notice is not None:
        warnings.warn(notice, NumericsWarning,
                      skip_file_prefixes=USER_FRAME_SKIP)
