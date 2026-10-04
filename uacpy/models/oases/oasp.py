"""OASP: the OASES broadband (pulse) transfer-function program."""

import warnings
import dataclasses
from dataclasses import dataclass
from types import MappingProxyType
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple, Union

import numpy as np

from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.models._extract import _restore_requested_axis
from uacpy.models._band import FREQUENCY_ARRAY_FIELDS, BandResolution
from uacpy.models._spec import ModelSpec
from uacpy.core.run_settings import EngineSettings, OutputSpec, RunMode
from uacpy.core.results import Field, PhaseReference
from uacpy.core.exceptions import ConfigurationError, FallbackWarning
from uacpy.io.oases_writer import (
    OASP_FREQ_MAX_PER_CENTER, write_oasp_input,
    oases_wavenumber_bounds, water_ac_anchor_frequency,
)
from uacpy.io._parsers import parse_oasp_trf
from uacpy.models.base import DEFAULT_RUN_TIMEOUT_S
from uacpy.models.oases._common import (
    _OASES_TRAITS, _SCTOUT_BARE_FORT46, _geometry_options,
    _oases_find_executable, _reject_plane_geometry_letter_on_a_point_source,
    _reject_wavenumber_overrun, _warn_if_water_ac_extrapolates,
    _check_n_wavenumbers_knob, _warn_offset_ignored_under_auto_sampling,
)
from uacpy.models.oases._sampling import (
    _oases_frequency_sweep, _oasp_marched_frequencies,
)
from uacpy.models.oases._base import OASES
from uacpy.models._notices import message_notice
from uacpy.core.engine_defaults import OASP_FREQ_MIN, OASP_INTEGRATION_OFFSET, OASP_N_TIME_SAMPLES


def _warn_if_trf_grid_replaced_request(requested, produced) -> None:
    """Warn when OASP's ``.trf`` grid is not the frequency the caller asked.

    OASP is a *pulse* model: its frequency axis is the FFT ladder implied by
    the time window and sample rate, not the bins the caller listed. The
    ``(freq_min, freq_max, N)`` triple in the deck bounds the computation, but the
    ``.trf`` comes back on the ladder — measured, ``frequencies=[150, 200,
    250]`` returns 820 bins spanning 149.902-249.878 Hz. ``Kraken`` on the
    identical call returns exactly the three requested, so a caller who has
    used one has no reason to expect the other.

    Not an error: the ladder is what OASP computes and it is usable. But the
    caller asked for specific bins and silently got a different axis, so the
    substitution has to be visible. Both OASP branches call this: BROADBAND
    with the whole ladder, COHERENT_TL with the single nearest bin its Field
    is stamped with, hence the two phrasings below. ``Field.at(frequency=…)``
    picks the nearest ladder bin.
    """
    if requested is None:
        return
    req = np.atleast_1d(np.asarray(requested, dtype=float))
    got = np.atleast_1d(np.asarray(produced, dtype=float))
    if req.size == got.size and np.allclose(req, got, rtol=1e-6, atol=1e-9):
        return
    if req.size == 1 and got.size == 1:
        warnings.warn(
            f"OASP: the run asked for {req[0]:g} Hz, but OASP's internal FFT "
            f"ladder has no bin there; the Field carries the nearest bin, "
            f"{got[0]:.5f} Hz. OASP is a pulse model — its frequency axis "
            f"follows the time window — and the phase error of the "
            f"substituted bin grows with range. Use Kraken/Scooter for "
            f"exactly the frequency named, or pick n_time_samples/freq_max "
            f"so a ladder bin lands on it.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
        return
    warnings.warn(
        f"OASP: frequencies= asked for {req.size} bin(s) spanning "
        f"{req.min():g}-{req.max():g} Hz; the .trf returns OASP's own FFT "
        f"ladder of {got.size} bin(s) spanning {got.min():g}-{got.max():g} Hz, "
        f"which is the axis on the Field. OASP is a pulse model — its "
        f"frequency axis follows the time window, not the requested list. Use "
        f"Field.at(frequency=…) to take the nearest bin, or Kraken/Scooter if "
        f"you need exactly the frequencies you name.",
        FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )


#: File root of every deck and output one OASP run writes. OASSP's mean-field
#: producer leaves its ``.045``/``.046`` under this stem.
_OASP_BASE_NAME = 'oasp_run'


#: Relative precision of the range axis a ``.trf`` gives back: TRFHEAD
#: stores ``R0`` and ``RSPACE`` (km) as REAL*4 (``oasiun23.f:836``,
#: ``:878-880``), so
#: ``R0 + i*RSPACE`` carries both roundings, each at most 2**-24 of a range
#: no larger than the axis's largest.
_TRF_RANGE_RTOL = 2.0 ** -23


def _read_trf_on_requested_ranges(output_file: Path, receiver) -> dict:
    """The ``.trf`` of an OASP or OASSP run as
    :func:`~uacpy.io._parsers.parse_oasp_trf` reads it, on the receiver's
    depths and — when the file's REAL*4 axis agrees with them to
    :data:`_TRF_RANGE_RTOL` — on the receiver's ranges (read as they are, a
    200 m range comes back 200.000003 m)."""
    trf = parse_oasp_trf(output_file, receiver_depths=receiver.depths)
    trf['ranges'] = _restore_requested_axis(
        trf['ranges'], np.atleast_1d(np.asarray(receiver.ranges, dtype=float)),
        _TRF_RANGE_RTOL)
    return trf


class _OaspSweep(NamedTuple):
    """Block III/VIII of one OASP deck as :func:`_oasp_sweep`
    resolves it (see :class:`OASPSettings` for the fields)."""
    requested: Optional[np.ndarray]
    n_time_samples: int
    n_time_samples_origin: str
    freq_min: float
    freq_max: float
    freq_max_origin: str
    center_frequency: float
    center_frequency_origin: str
    ladder: np.ndarray
    notices: Tuple[str, ...]


@dataclass(frozen=True, eq=False)
class OASPSettings(EngineSettings):
    """The settings one :class:`OASP` run resolved, before launching:
    ``OASP().run_settings(env, source, receiver).engine``, and
    ``result.run_settings.engine`` on the field it produced.

    Attributes
    ----------
    options : str
        The option line the deck carries (Block II), ``'P'`` included for a
        line Source.
    requested_frequencies : ndarray or None
        The frequencies (Hz) the call asked for: ``run(frequencies=)``, the
        grid a TIME_SERIES pulse implies, a multi-element
        ``source.frequencies``, or the source frequency of a COHERENT_TL
        run; ``None`` for the default sweep of a one-carrier BROADBAND run.
    n_time_samples : int
        Block VIII's ``NX``.
    n_time_samples_origin : str
        Where ``NX`` came from.
    freq_min, freq_max : float
        Block VIII's ``FR1 FR2`` (Hz).
    freq_max_origin : str
        Where ``FR2`` came from.
    time_step : float
        Block VIII's ``DT`` (s), the Nyquist step of ``FR2``.
    center_frequency : float
        Block III's ``FREQS`` (Hz), the pulse carrier.
    center_frequency_origin : str
        Where ``FREQS`` came from.
    marched_frequencies : ndarray
        The bins (Hz) the ``.trf`` carries (:func:`_oasp_marched_frequencies`).
    c_low, c_high : float
        Block VII's phase-speed window ``CMIN CMAX`` (m/s).
    c_low_origin, c_high_origin : str
        Where ``c_low`` and ``c_high`` came from: the constructor, or
        :func:`~uacpy.io.oases_writer.oases_wavenumber_bounds` on the water
        column.
    n_wavenumbers : int or None
        Block VII's ``NW``; ``None`` is OASES' automatic sampling.
    integration_offset : float
        The contour offset (dB/wavelength) on the frequency line.
    integrand_plot_step : int or None
        Block VII's ``INTF``; ``None`` writes the writer's 40.
    dip_angle : float or None
        The dip-slip source's dip (degrees), read only under ``'4'``.
    notices : tuple of Notice
        What the run says about the sweep it resolved (a non-equispaced
        request resampled, a constructor edge overridden).
    """

    options: str
    requested_frequencies: Optional[np.ndarray]
    n_time_samples: int
    n_time_samples_origin: str
    freq_min: float
    freq_max: float
    freq_max_origin: str
    time_step: float
    center_frequency: float
    center_frequency_origin: str
    marched_frequencies: np.ndarray
    c_low: float
    c_low_origin: str
    c_high: float
    c_high_origin: str
    n_wavenumbers: Optional[int]
    integration_offset: float
    integrand_plot_step: Optional[int]
    dip_angle: Optional[float]

    _ARRAY_FIELDS = FREQUENCY_ARRAY_FIELDS


def _check_oasp_knobs(freq_min, freq_max) -> None:
    """Refuse an OASP ``freq_min`` not below ``freq_max``.

    An inverted pair reaches the deck unvalidated (the writer formats
    whatever it is given) and OASP then computes LX > MX, so the
    frequency loop never executes and the .trf comes back empty rather
    than wrong. The band check of a run compares freq_max against the
    band top only, never against freq_min, so this is the one place the
    ordering is tested."""
    if freq_max is not None and freq_min >= freq_max:
        raise ConfigurationError(
            f"OASP: freq_min={freq_min:g} Hz is not below "
            f"freq_max={freq_max:g} Hz. Block VIII writes the two as "
            f"FR1 FR2 (unoasp22.f:176) and the kernel indexes them as "
            f"LX = FR1/DLFREQ + 1 .. MX = FR2/DLFREQ + 2 "
            f"(unoasp22.f:238-239), so an inverted pair gives LX > MX and "
            f"the frequency loop runs zero times.",
            remediation="Pass the low edge as freq_min and the high edge "
                        "as freq_max.",
        )


def _nearest_bin(ladder, frequency: float) -> int:
    """Index of the ladder bin nearest ``frequency`` (the first of a
    tie)."""
    if len(ladder) <= 1:
        return 0
    return int(np.argmin(np.abs(np.asarray(ladder) - frequency)))


def _oasp_sweep(mode, source, requested, *, knob_center_frequency,
                knob_freq_min, knob_freq_max,
                knob_n_time_samples) -> _OaspSweep:
    """Block III/VIII of the deck — the carrier, ``NX``, ``FR1``, ``FR2``
    — and the ladder of bins they give, resolved once for a call.

    A requested band (:meth:`OASP._requested_frequencies`) sets the
    sweep:
    ``FR1``/``FR2`` its edges and ``NX`` the smallest power of two whose
    bins ``DLFREQ = 2·FR2/NX`` are as fine as the requested spacing
    (floored at a pinned ``n_time_samples``), with the Nyquist lifted by
    ``NX/(NX - 2)`` so the top bin lands on the band's upper edge. With
    no request (the default sweep of a one-carrier BROADBAND run) the
    sweep is ``freq_min .. freq_max`` (``2.5·fc`` when unpinned) on
    ``n_time_samples`` (4096 when unpinned). Pure: what the run says
    about it is returned in ``notices``. The ``knob_*`` arguments are
    the model's constructor values (``None`` when unpinned).
    """
    notices = []
    if requested is not None:
        # COHERENT_TL is a single-frequency mode (the base refuses a
        # multi-frequency Source for it). An explicit multi-element
        # ``frequencies=`` would otherwise run a broadband sweep and
        # return only the bin nearest the source frequency.
        if mode == RunMode.COHERENT_TL and requested.size > 1:
            raise ConfigurationError(
                "OASP.run(run_mode=COHERENT_TL) takes a single frequency; "
                "got a multi-element frequencies= vector. For broadband "
                "H(f) use RunMode.BROADBAND."
            )
        if requested.size == 0:
            raise ConfigurationError(
                "OASP.run(frequencies=…) requires at least one positive "
                "frequency."
            )
        # The (freq_min, freq_max, N) triple is OASP's internal language; a
        # non-equispaced request is named as such.
        fmin_user, freq_max, n_user, sweep_notice = \
            _oases_frequency_sweep(requested, 'OASP')
        if sweep_notice is not None:
            notices.append(message_notice(sweep_notice, FallbackWarning))
        # Deck fc: the centre of the requested band — the same centre
        # convention the sibling broadband models use (RAM's (fc, Q, T)
        # sweep is parameterised around the band midpoint). A pinned
        # ``center_frequency`` wins.
        if knob_center_frequency is not None:
            fc_run = knob_center_frequency
            fc_origin = 'OASP(center_frequency=…)'
        else:
            fc_run = float(0.5 * (requested.min() + requested.max()))
            fc_origin = 'centre of the requested band'
        if knob_freq_min and abs(knob_freq_min - fmin_user) > 1e-9:
            notices.append(
                message_notice(f"OASP.run(frequencies=…) sets the sweep's lower edge to "
                            f"{fmin_user:.3f} Hz, overriding the constructor's "
                            f"freq_min={knob_freq_min:.3f} Hz.",
                            FallbackWarning))
        if (knob_freq_max is not None
                and abs(knob_freq_max - freq_max) > 1e-9):
            notices.append(
                message_notice(f"OASP.run(frequencies=…) sets the sweep's upper edge to "
                            f"{freq_max:.3f} Hz, overriding the constructor's "
                            f"freq_max={knob_freq_max:.3f} Hz.",
                            FallbackWarning))
        freq_min = fmin_user
        freq_max_origin = 'top of the requested band'
        n_time_samples = knob_n_time_samples
        n_time_origin = ('OASP(n_time_samples=…)'
                         if n_time_samples is not None else
                         f'default {OASP_N_TIME_SAMPLES}')
        if n_user > 1:
            # df implied by the equispaced (freq_min, freq_max, N) grid that
            # OASES will actually run.
            df_user = (freq_max - fmin_user) / (n_user - 1)
        else:
            df_user = float(requested[0])
        if df_user > 0:
            # OASP's bin spacing is DLFREQ = 1/(DT*NX) (unoasp22.f:237)
            # and the deck sets DT at Nyquist, 1/(2*freq_max), so
            # NX = 2*freq_max/DLFREQ samples are needed to land bins as
            # fine as the requested df. OASP requires NT = 2^M
            # (oasp.tex:129), so round that up to a power of two — never
            # down past a pinned n_time_samples.
            # Two samples more than that, so the Nyquist lift below keeps
            # DLFREQ = 2*freq_max/(NX - 2) no coarser than df.
            target = max(int(n_time_samples or 0),
                         int(np.ceil(2.0 * freq_max / df_user)) + 2)
            if target > 1:
                n_time_samples = 1 << (target - 1).bit_length()
            else:
                n_time_samples = 2
            n_time_origin = (
                f"the power of two reaching the requested spacing "
                f"{df_user:g} Hz"
                + (" (floored at OASP(n_time_samples=…))"
                   if knob_n_time_samples is not None else ""))
            # OASP's top bin is DLFREQ*(NX/2 - 1): MX = FR2/DLFREQ + 2 is
            # clamped to NX/2 (unoasp22.f:237-246), and with the Nyquist
            # at the band's top the last bin fell one DLFREQ short of it
            # (measured: a 90-110 Hz band topped out at 109.95 Hz, a
            # 100-300 Hz Source band at 298.8 Hz). Lifting the Nyquist by
            # NX/(NX - 2) puts that bin on the band's upper edge; for a
            # band, the 1e-6 keeps the .trf's REAL*4 axis (measured
            # 299.9999916 for a 300 Hz edge at 1e-9) from landing it just
            # below. One requested frequency is the ladder's one bin,
            # DLFREQ itself (MX is clamped to NX/2 = 2), so it takes no
            # guard: the bin then lands on the request to REAL*4 (<= 1.2e-7
            # relative from 20 Hz to 2 kHz), where the guard put it 1e-6
            # above — 5 deg of phase at 2 kHz and 10 km, and past the
            # substituted-bin notice's rtol=1e-6 at 214 of 397 frequencies.
            guard = 1.0 + 1e-6 if n_user > 1 else 1.0
            if n_time_samples > 2:
                freq_max = (freq_max * n_time_samples
                            / (n_time_samples - 2) * guard)
                freq_max_origin = ('top of the requested band, the '
                                   'Nyquist lifted by NX/(NX - 2)')
        elif n_time_samples is None:
            n_time_samples = OASP_N_TIME_SAMPLES
    else:
        # Centre convention shared with the sibling broadband models
        # (RAM's (fc, Q, T) sweep centres on the band midpoint).
        src_freqs = np.atleast_1d(
            np.asarray(source.frequencies, dtype=float))
        if knob_center_frequency is not None:
            fc_run = knob_center_frequency
            fc_origin = 'OASP(center_frequency=…)'
        else:
            fc_run = float(0.5 * (src_freqs.min() + src_freqs.max()))
            fc_origin = 'source.frequencies'
        f_top = max(fc_run, float(src_freqs.max()))
        freq_max = knob_freq_max
        derived_freq_max = freq_max is None
        if derived_freq_max:
            # Band headroom above the carrier (the writer's own rule).
            freq_max = OASP_FREQ_MAX_PER_CENTER * fc_run
            freq_max_origin = (f'{OASP_FREQ_MAX_PER_CENTER:g} x centre '
                               f'frequency')
        else:
            freq_max_origin = 'OASP(freq_max=…)'
        # The derived edge needs the same test as a pinned one. It is
        # 2.5x the band centre, so it clears the band by construction —
        # unless ``center_frequency`` pins fc below it, and then a
        # 1000 Hz source with center_frequency=10 writes FR2=25 Hz and
        # comes back labelled with a 25 Hz bin.
        if f_top > freq_max:
            origin = (f"the derived sweep edge freq_max="
                      f"{OASP_FREQ_MAX_PER_CENTER:g}×fc="
                      f"{freq_max:.1f} Hz" if derived_freq_max
                      else f"the pinned sweep edge "
                           f"freq_max={freq_max:.1f} Hz")
            fix = ("Raise center_frequency to the band's own centre (or "
                   "leave it None), pin freq_max above the band, or pass "
                   "frequencies= explicitly."
                   if derived_freq_max else
                   f"Raise freq_max (or leave it None to derive "
                   f"{OASP_FREQ_MAX_PER_CENTER:g}×fc), or pass frequencies= "
                   f"explicitly.")
            raise ConfigurationError(
                f"OASP: the requested band reaches {f_top:.1f} Hz "
                f"(centre {fc_run:.1f} Hz), above {origin} — the sweep "
                f"would never compute the top of the band. {fix}"
            )
        freq_min = knob_freq_min
        if knob_n_time_samples is not None:
            n_time_samples = knob_n_time_samples
            n_time_origin = 'OASP(n_time_samples=…)'
        else:
            n_time_samples = OASP_N_TIME_SAMPLES
            n_time_origin = f'default {OASP_N_TIME_SAMPLES}'
    # DT at Nyquist against FR2 (see write_oasp_input).
    ladder = _oasp_marched_frequencies(n_time_samples, freq_min, freq_max,
                                    1.0 / (2.0 * freq_max))
    return _OaspSweep(
        requested=requested, n_time_samples=int(n_time_samples),
        n_time_samples_origin=n_time_origin, freq_min=float(freq_min),
        freq_max=float(freq_max), freq_max_origin=freq_max_origin,
        center_frequency=float(fc_run),
        center_frequency_origin=fc_origin, ladder=ladder,
        notices=tuple(notices))


def _oasp_settings(mode, requested, *, options, c_low_pinned,
                   c_high_pinned, n_wavenumbers, integration_offset,
                   integrand_plot_step, dip_angle,
                   knob_center_frequency, knob_freq_min,
                   knob_freq_max, knob_n_time_samples, env,
                   source) -> OASPSettings:
    """Every value the OASP deck is written from, resolved once on the
    projected ``env`` for a call in ``mode`` asking for ``requested``:
    the option line (with ``'P'`` for a line Source), the sweep
    (:func:`_oasp_sweep`), Block VII's phase-speed window and
    wavenumber count."""
    sweep = _oasp_sweep(mode, source, requested,
                        knob_center_frequency=knob_center_frequency,
                        knob_freq_min=knob_freq_min,
                        knob_freq_max=knob_freq_max,
                        knob_n_time_samples=knob_n_time_samples)
    options = _geometry_options(
        options if options is not None else 'N J', source)
    ssp_data = env.ssp.extend_to(env.depth).to_pairs()
    c_low, c_high = oases_wavenumber_bounds(ssp_data, cmax=1e9)
    derived = "oases_wavenumber_bounds(water column)"
    if c_low_pinned is not None:
        c_low = c_low_pinned
    if c_high_pinned is not None:
        c_high = c_high_pinned
    c_low_origin = 'OASP(c_low=…)' if c_low_pinned is not None else derived
    c_high_origin = ('OASP(c_high=…)' if c_high_pinned is not None
                     else derived)
    return OASPSettings(
        options=options,
        requested_frequencies=sweep.requested,
        n_time_samples=sweep.n_time_samples,
        n_time_samples_origin=sweep.n_time_samples_origin,
        freq_min=sweep.freq_min,
        freq_max=sweep.freq_max,
        freq_max_origin=sweep.freq_max_origin,
        time_step=1.0 / (2.0 * sweep.freq_max),
        center_frequency=sweep.center_frequency,
        center_frequency_origin=sweep.center_frequency_origin,
        marched_frequencies=sweep.ladder,
        c_low=float(c_low),
        c_low_origin=c_low_origin,
        c_high=float(c_high),
        c_high_origin=c_high_origin,
        n_wavenumbers=n_wavenumbers,
        integration_offset=integration_offset,
        integrand_plot_step=integrand_plot_step,
        dip_angle=dip_angle,
        notices=sweep.notices,
    )


def _write_oasp_deck(deck: Path, inputs, name: str) -> None:
    """The OASP deck at ``deck``, written by
    :func:`~uacpy.io.oases_writer.write_oasp_input` from the settings
    ``inputs`` carries; ``name`` is the model the water-absorption
    notice names."""
    engine = inputs.settings.engine
    writer_kwargs: dict = {
        'integration_offset': engine.integration_offset,
        'n_wavenumbers': engine.n_wavenumbers,
        'freq_min': engine.freq_min,
        'center_frequency': engine.center_frequency,
        'time_step': engine.time_step,
        'c_low': engine.c_low,
        'c_high': engine.c_high,
    }
    if engine.integrand_plot_step is not None:
        writer_kwargs['integrand_plot_step'] = \
            engine.integrand_plot_step
    if engine.dip_angle is not None:
        writer_kwargs['dip_angle'] = engine.dip_angle
    _warn_if_water_ac_extrapolates(
        name, inputs.env, engine.marched_frequencies,
        water_ac_anchor_frequency(inputs.env, engine.freq_min,
                                  engine.freq_max))
    write_oasp_input(
        filepath=deck,
        env=inputs.env,
        source=inputs.source,
        receiver=inputs.receiver,
        options=engine.options,
        n_time_samples=engine.n_time_samples,
        freq_max=engine.freq_max,
        **writer_kwargs,
    )


def _reject_unreadable_oasp_options(options, n_wavenumbers) -> None:
    """Reject raw ``options`` letters whose ``.trf`` uacpy cannot read back.

    Every case below writes a well-formed ``.trf`` that the reader parses
    without complaint but that no longer means what the axis labels say —
    an extra output component the reader flattens away, a range axis that
    is really slowness, or a spectrum carrying a complex-contour offset
    the time-series synthesis does not undo. Left unchecked they surface
    as a plausible wrong answer, so each raises here instead. Leaving
    ``options`` unset (the writer's default ``'N J'``) hits none of them
    and skips the whole check.

    OASES GETOPT (``unoasp22.f:871-872`` ``READ(1,200) OPT`` /
    ``200 FORMAT(40A1)``, scanned character by character at ``:873``)
    ignores whitespace, so ``'NJO'`` enables ``'O'`` exactly like
    ``'N J O'`` — hence the tests below are on the character set, not on
    whitespace-split tokens.
    """
    if not options:
        return
    opt_chars = set(str(options)) - set(' \t\n')

    # The .trf reader collapses the MSUFT / ISROW / NOUT axes onto the
    # first slot, so extra output components are dropped, not reported.
    # Letters that add an IOUT slot, i.e. increment NOUT and put a second
    # component in every .trf record (unoasp22.f:887-918: N→1 V→2 H→3 R→5
    # K→6 S→7), plus 'U' (DECOMP), which splits the output over NCOMPO=5
    # separate files (unoasp22.f:455-456, opened one unit apart at
    # :539-540).
    multi_axis = {'V', 'H', 'R', 'K', 'S', 'U'} & opt_chars
    if multi_axis:
        raise ConfigurationError(
            f"OASP.run: options {sorted(multi_axis)} request "
            "multi-component / decomposed output, which the .trf "
            "reader currently flattens. Pass options without "
            "these letters (default 'N J' returns scalar pressure) "
            "or read the .trf directly."
        )
    if 'O' in opt_chars:
        # 'O' moves the frequency integration onto a complex
        # contour (Im(omega) = -ln(50)·Δf, unoasp22.f:372-373). The
        # .trf reader discards that offset and synthesize_time_series
        # does not re-apply exp(-Im(omega)·t), so the time series
        # would be silently wrong. Reject rather than mis-synthesise.
        raise ConfigurationError(
            "OASP.run: option 'O' (complex frequency integration) "
            "bakes an exp(-ln(50)·Δf·t) contour into the spectrum "
            "that uacpy's time-series synthesis does not undo. "
            "Drop 'O' (the default 'N J' uses a real frequency axis)."
        )
    if 't' in opt_chars:
        # Lowercase 't' sets INTTYP=-1 (unoasp22.f:1028-1029), and
        # unoasp22.f:178-189 then overwrites the deck's R0 / RSPACE /
        # NPLOTS with a slowness axis derived from CMIN/CMAX. The units are
        # s/km, not s/m: ``r0 = 1e3/cmaxin`` and
        # ``rspace = (1e3/cminin - r0)/(nwvno-1)`` at :186-188 carry the
        # 1e3. The amplitudes differ too — the tau-p branch writes
        # ``1.0E6*real(CFFX(...))`` (oasiun23.f:452-453) where the two
        # normal paths write the bare value (:318, :770), so a .trf read
        # without dividing it out is 120 dB high. uacpy would label the
        # abscissa 'range' in metres and the amplitude unscaled.
        raise ConfigurationError(
            "OASP.run: option 't' (tau-p seismograms) replaces the "
            "receiver range axis with slowness "
            "(unoasp22.f:178-189), which uacpy's Field has no "
            "coordinate for and would mislabel as range. Drop 't'."
        )
    if 'J' not in opt_chars and n_wavenumbers is None:
        # Under automatic wavenumber sampling (n_wavenumbers=None →
        # AUSAMP), OASES forces the complex frequency contour
        # OMEGIM = -ln(50)·Δf unless 'J' keeps ICNTIN > 0
        # (unoasp22.f:288-296, :304-306 then :372-373). Without 'J'
        # the .trf then carries the same offset 'O' would, which the
        # time-series synthesis cannot undo.
        raise ConfigurationError(
            "OASP.run: a custom options string without 'J' enables the "
            "complex frequency contour (OMEGIM≠0) under automatic "
            "wavenumber sampling, which uacpy's time-series synthesis "
            "cannot undo. Add 'J' (the default 'N J' keeps a real "
            "frequency axis) or pin n_wavenumbers≥1."
        )


class OASP(OASES):
    """
    OASP - OASES Pulse / Broadband Transfer-Function Model

    Computes broadband acoustic transfer functions via wavenumber integration
    followed by FFT to produce time-series / pulse responses. OASP is the
    "Pulse" variant of SAFARI, using the same wavenumber-integration kernel
    as OAST but evaluated across a frequency sweep.

    Notes
    -----
    ``RunMode.COHERENT_TL`` returns the complex pressure at the source
    frequency: the sweep is narrowed to the one bin OASP's ladder puts on it
    (``FR1 = f``, ``unoasp22.f:237-248``), so the run integrates one or two
    bins rather than the whole default sweep. ``RunMode.BROADBAND`` returns
    the transfer function on OASP's own frequency ladder — for a single
    source frequency ``fc`` and no ``frequencies=``, the bins of
    ``0 .. 2.5·fc``, a wider band than the ``fc·(1 ± 0.25)`` the other
    broadband engines expand one carrier to — and ``RunMode.TIME_SERIES``
    its synthesis (``synthesize_time_series``). The bins are
    ``run_settings(...).frequencies``. For a single-cell trace use
    ``tf.to_time_trace(depth=…, range=…)``. A line Source runs in OASP's plane
    geometry (option ``'P'``). For range-dependent problems, RAM is
    recommended.

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Per-model: ``'ssp': 'mean'``, ``'bottom_range': 'median'`` (the
    layer stack is kept).

    Examples
    --------
    >>> from uacpy.models import OASP
    >>> oasp = OASP(n_time_samples=256, freq_max=120)
    >>> result = oasp.run(env, source, receiver)
    """

    # Declarative metadata (see PropagationModel / ModelSpec). OASP:
    # range-independent broadband wavenumber integration / pulse synthesis;
    # multi-layer fluid + elastic bottom honoured. Single spectral solve per
    # frequency → mean SSP / median bottom column represent the path.
    # The band edges follow run(frequencies=…) when the call passes one.
    _UNPINNED_FIELDS = frozenset({'options', 'freq_min', 'freq_max'})

    # A line Source runs in OASP's plane geometry, option 'P' (ICDR = 1,
    # unoasp22.f:946-948), which _resolve_engine_settings writes for it.
    spec = ModelSpec(
        modes=(RunMode.COHERENT_TL, RunMode.BROADBAND, RunMode.TIME_SERIES),
        supports={'layered_bottom', 'elastic_media',
                  'rough_surface', 'rough_bottom'},
        source_types=frozenset({'point', 'line'}),
        collapse={'ssp': 'mean', 'bottom_range': 'median'},
        traits=dataclasses.replace(
            _OASES_TRAITS,
            consumes_run_t_start=True,
            consumes_single_mode_frequencies=True,
            # BROADBAND runs OASP's own FFT ladder, not the
            # ``fc·(1 ± 0.25)`` band whose 1 Hz floor the base announces.
            announced_band_modes=frozenset({RunMode.TIME_SERIES}),
        ),
    )
    provenance_id = 'oases'
    outputs = MappingProxyType({
        RunMode.COHERENT_TL: OutputSpec(
            'Field', kind='pressure', unit='Pa',
            phase_reference=PhaseReference.TRAVELLING_WAVE.value,
            coherent=True),
        RunMode.BROADBAND: OutputSpec(
            'Field', kind='pressure', unit='Pa',
            phase_reference=PhaseReference.TRAVELLING_WAVE.value,
            coherent=True),
        RunMode.TIME_SERIES: OutputSpec(
            'Field', kind='pressure', unit='Pa',
            phase_reference=PhaseReference.TIME_DOMAIN_NATIVE.value),
    })

    def __init__(
        self,
        *,
        executable: Optional[Path] = None,
        n_time_samples: Optional[int] = None,
        freq_min: float = OASP_FREQ_MIN,
        freq_max: Optional[float] = None,
        center_frequency: Optional[float] = None,
        integrand_plot_step: Optional[int] = None,
        options: Optional[str] = None,
        integration_offset: float = OASP_INTEGRATION_OFFSET,
        n_wavenumbers: Optional[int] = None,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        dip_angle: Optional[float] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = DEFAULT_RUN_TIMEOUT_S,
        collapse: Optional[Dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to OASP binary. Auto-detected if ``None``.
        n_time_samples : int, optional
            FFT length NX (samples per receiver trace; OASP rounds it up to
            a power of two). ``None`` (default) takes 4096 for the default
            sweep of a one-carrier BROADBAND run, and for a requested band —
            ``run(frequencies=)``, a TIME_SERIES pulse, a multi-element
            ``source.frequencies``, the source frequency of a COHERENT_TL
            run — the smallest power of two whose bins are as fine as the
            requested spacing, so the TIME_SERIES record is the
            ``output_duration`` asked for, rounded up to that power of two.
            A value given is a floor on NX in every case.
        freq_min : float, optional
            Lower edge of the OASP broadband sweep (Hz). Default 0.0.
            Must be below ``freq_max`` when that is pinned.
            Default ``0.0``.
        freq_max : float, optional
            Upper edge of the OASP broadband sweep (Hz). ``None``
            (default) derives ``2.5 ×`` the centre frequency at
            ``run()`` time; a pinned value below the centre frequency
            or the top of the requested band raises at ``run()``.
        center_frequency : float, optional
            Carrier frequency for the pulse (Hz). ``None`` defaults to
            the centre (midpoint) of the run's frequency band at
            ``run()`` time — a single ``source.frequencies`` entry is
            its own centre.
        integrand_plot_step : int, optional
            OASP's ``INTF`` (Block VII, ``unoasp22.f:160``). It gates how
            often the wavenumber *integrand* is plotted
            (``unoasp22.f:591``: ``IF (MOD(JJ-LXP1,INTF).EQ.0) KPLOT=1``,
            consumed at ``unoasp22.f:678``) and does **not** decimate the
            ``.trf`` frequency axis, which always carries every bin
            ``LXP1..MX``. ``None`` → 40.
        options : str, optional
            Raw OASES option string. ``None`` takes ``'N J'``; the ``'J'``
            forces a real frequency axis (OMEGIM=0) so the broadband
            ``.trf`` synthesis stays valid. A line Source adds ``'P'``;
            ``'P'`` given here with a point Source is refused.
        integration_offset : float, optional
            Wavenumber-contour offset (dB/wavelength). Default 0, which under
            the 'J' option of the default option line is not "no offset": OASES
            takes any value below 1e-10 as a request for its own default,
            60*c*(1/c_min - 1/c_max)/N_k dB/wavelength (unoasp22.f:354-361; 40
            in place of 60 for Bessel integration). A value above 1e-10 is used
            as given.
        n_wavenumbers : int, optional
            Wavenumber sample count. ``None`` lets OASP auto-pick.
        c_low, c_high : float, optional
            ``CMIN``/``CMAX`` (m/s), Block VII's phase-speed window.
            ``None`` derives the pair from the water column alone, which
            leaves an elastic seabed's shear and interface branches — phase
            speeds below the slowest water speed — outside ``k_max`` and out
            of the integral. Same knob and same reason as :class:`OAST`.
        dip_angle : float, optional
            Fault dip angle (degrees) for the dip-slip moment source that
            option ``'4'`` selects (``unoasp22.f:993-995``). INSRC reads it
            from the source record — as the second token without ``'L'``
            (``oaseun31.f:1114``), as the seventh with it
            (``oaseun31.f:1089``). ``None`` writes 0 when ``'4'`` is
            present and raises when it is not.
        use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
            Standard plumbing (see :class:`PropagationModel`).
        The receiver ranges are ``receiver.ranges``: Block VIII's
        ``R0 RSPACE NPLOTS`` are read from them.
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )
        self.n_time_samples = (
            int(n_time_samples) if n_time_samples is not None else None
        )
        self.freq_min = float(freq_min)
        self.freq_max = float(freq_max) if freq_max is not None else None
        self.center_frequency = (
            float(center_frequency) if center_frequency is not None else None
        )
        self.integrand_plot_step = (
            int(integrand_plot_step)
            if integrand_plot_step is not None else None
        )
        self.options = options
        self.integration_offset = float(integration_offset)
        self.n_wavenumbers = n_wavenumbers
        # Block VII's CMIN/CMAX; ``None`` keeps the water-column
        # derivation, as on OAST.
        self.c_low = float(c_low) if c_low is not None else None
        self.c_high = float(c_high) if c_high is not None else None
        # OASP reads the offset token under 'J' or 'd' (unoasp22.f:126-133)
        # and otherwise sets OFFDB = OFFDBIN = 0 at :365. ``options=None``
        # takes the writer's default line.
        _warn_offset_ignored_under_auto_sampling(
            'OASP', self.integration_offset, self.n_wavenumbers,
            options=self.options if self.options is not None else 'N J',
            offset_letters='Jd')
        # DA, the dip angle INSRC reads only under the '4' (dip-slip) option.
        self.dip_angle = float(dip_angle) if dip_angle is not None else None
        self._check_knobs()

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        self._exe = self._resolve_executable(
            executable, lambda: _oases_find_executable(self, 'oasp'),
        )

    def _check_knobs(self) -> None:
        """Refuse a ``freq_min`` not below ``freq_max``
        (:func:`_check_oasp_knobs`). Run at construction and again by every
        run (:meth:`_validate_engine`), since the attributes can be
        reassigned in between."""
        _check_n_wavenumbers_knob(self.n_wavenumbers)
        _check_oasp_knobs(self.freq_min, self.freq_max)

    def _validate_engine(self, env, source, receiver, run_mode, *,
                         request=None) -> None:
        """Stage 2: the refusals of an option line the ``.trf`` reader cannot
        read back (:func:`_reject_unreadable_oasp_options`) and of a raw
        ``'P'`` with a point Source."""
        _reject_unreadable_oasp_options(self.options, self.n_wavenumbers)
        _reject_plane_geometry_letter_on_a_point_source(
            'OASP', self.options if self.options is not None else 'N J',
            self.options, source, 'unoasp22.f:946-948')

    def _requested_frequencies(self, mode, source, frequencies, time):
        """The frequencies (Hz) a call asks OASP for, which its sweep is
        resolved from (:func:`_oasp_sweep`), or ``None`` for the default
        sweep: ``run(frequencies=)``; for TIME_SERIES, the grid the pulse
        implies; for BROADBAND, a multi-element ``source.frequencies`` (it
        names a band as explicitly as ``frequencies=`` does); for
        COHERENT_TL, the source frequency, so its sweep narrows onto the bin
        at that frequency. :meth:`_marched_frequencies` then records the
        bins the sweep gives."""
        if frequencies is not None:
            return BandResolution(
                np.atleast_1d(np.asarray(frequencies, dtype=float)))
        if mode == RunMode.TIME_SERIES:
            return super()._requested_frequencies(mode, source, None, time)
        source_freqs = np.atleast_1d(np.asarray(source.frequencies,
                                                dtype=float))
        if mode == RunMode.BROADBAND:
            return BandResolution(source_freqs if source_freqs.size > 1
                                  else None)
        return BandResolution(source_freqs[:1])

    def _marched_frequencies(self, settings):
        """The bins OASP propagates: the ladder of its sweep
        (``settings.engine.marched_frequencies``), for COHERENT_TL the one bin
        nearest the source frequency, which the Field carries. A one-carrier
        BROADBAND call gets no 1 Hz floor notice of the ``fc·(1 ± 0.25)``
        band, which OASP does not run (``spec.traits.announced_band_modes``).
        """
        ladder = settings.engine.marched_frequencies
        if settings.mode == RunMode.COHERENT_TL:
            ladder = ladder[[_nearest_bin(
                ladder, settings.engine.requested_frequencies[0])]]
        return ladder

    def _resolve_engine_settings(self, env, source, receiver, settings, *,
                                 given_env, request=None) -> 'OASPSettings':
        """Stage 3: :func:`_oasp_settings` for this model's knobs."""
        return _oasp_settings(
            settings.mode, settings.frequencies, options=self.options,
            c_low_pinned=self.c_low, c_high_pinned=self.c_high,
            n_wavenumbers=self.n_wavenumbers,
            integration_offset=self.integration_offset,
            integrand_plot_step=self.integrand_plot_step,
            dip_angle=self.dip_angle,
            knob_center_frequency=self.center_frequency,
            knob_freq_min=self.freq_min, knob_freq_max=self.freq_max,
            knob_n_time_samples=self.n_time_samples,
            env=env, source=source)

    def _write_input(self, inputs) -> Path:
        """Stage 4: the OASP deck (:func:`_write_oasp_deck`)."""
        deck = inputs.work_dir / f'{_OASP_BASE_NAME}.dat'
        self._log(f"Writing OASP input file: {deck}")
        _write_oasp_deck(deck, inputs, self.model_name)
        return deck

    #: The transfer-function files an OASP run may write, in the order they
    #: are looked for: option '8' renames it to <base>.dtrf
    #: (oasiun23.f:837-845 `bufch='d'//trfext`) with the record layout
    #: unchanged, CFFX being declared plain COMPLEX (oasiun23.f:18).
    _TRF_SUFFIXES = ('.trf', '.dtrf')

    def _launch(self, inputs, deck: Path) -> None:
        """Stage 4: run ``oasp`` on the deck; refuse a run that sampled
        past OASES' wavenumber bound or wrote no transfer function, quoting
        the binary's streams (OASES writes no print file)."""
        proc = self._execute(deck.stem, inputs.work_dir)
        _reject_wavenumber_overrun(self.model_name, proc, 1)
        self._require_output(
            [inputs.work_dir / f'{deck.stem}{suffix}'
             for suffix in self._TRF_SUFFIXES],
            what='a transfer-function file', process=proc,
        )

    def _read_output(self, inputs, deck: Path) -> dict:
        """Stage 4: the transfer function on the requested axes
        (:func:`_read_trf_on_requested_ranges`)."""
        output_file = self._require_output(
            [inputs.work_dir / f'{deck.stem}{suffix}'
             for suffix in self._TRF_SUFFIXES],
            what='a transfer-function file',
        )
        self._log(f"Reading OASP output: {output_file}")
        return _read_trf_on_requested_ranges(output_file, inputs.receiver)

    def _to_result(self, inputs, deck: Path, raw: dict) -> Field:
        """Stage 5: the Field the run mode asks for — the whole complex
        H(f) cube for BROADBAND, the one bin at the source frequency for
        COHERENT_TL, or the synthesised trace for TIME_SERIES.

        The raw ``.trf`` payload is the normal stress, not the pressure:
        "N  Normal stress sigma_zz (= -p in fluids)" (oases/doc/oasp.tex
        :185; rdoast.tex:183-187). OASES runs on the same e^{+i omega t}
        convention as the other engines — a source delay is
        exp(-ai*dsq*sdelay), dsq = 2*pi*freq (oaseun31.f:1681,
        oasiun22.f:1056) — so -1 is the whole conversion, and measured
        against Scooter's Hankel path on the same Pekeris case,
        arg(raw/Scooter) = 176.8-179.9 deg with the magnitude ratio
        0.94-1.02. The negation makes the 'travelling_wave' tag mean the
        convention every coherent engine shares (field.exe gets a sign of
        its own in kraken/_extract.py's assemble_field_from_shd). The
        payload is upcast (the .trf holds COMPLEX*8) so every uacpy engine
        returns one dtype.
        """
        settings = inputs.settings
        engine = settings.engine
        source = inputs.source
        transfer_func = raw['transfer_function']  # (n_frequencies, n_range, n_depth)
        if settings.mode in (RunMode.BROADBAND, RunMode.TIME_SERIES):
            _warn_if_trf_grid_replaced_request(engine.requested_frequencies,
                                               raw['freq'])
            # Convention: (n_depth, n_range, n_frequencies) — trailing
            # axis is the variable dim. Source axes: (freq, range, depth).
            tf_reordered = -np.transpose(transfer_func, (2, 1, 0)).astype(
                np.complex128)
            result = Field(
                data=tf_reordered,
                coords={
                    'depth': raw['depths'],
                    'range': raw['ranges'],
                    'frequency': raw['freq'],
                },
                # OASP's FFT length NX: the synthesis floors its own at it.
                synthesis_floor=engine.n_time_samples,
                **self._result_kwargs(
                    source,
                    phase_reference=PhaseReference.TRAVELLING_WAVE.value,
                    backend='oasp',
                    frequencies=raw['freq'],
                    source_depth=raw['source_depth'],
                    center_frequency=raw['center_frequency'],
                ),
            )
            result = self._mask_source_axis(result, source)
            if settings.time is not None:
                result = self._finish_broadband(result, settings)
        else:
            # COHERENT_TL: the bin at the source frequency, as complex
            # narrowband pressure on (n_depth, n_range). Users get TL via
            # ``field.dB`` or ``.to_dB()``.
            f_req = float(engine.requested_frequencies[0])
            freq_idx = _nearest_bin(raw['freq'], f_req)
            # A ladder bin off the request is announced, as the BROADBAND
            # branch announces its axis.
            _warn_if_trf_grid_replaced_request(f_req, raw['freq'][freq_idx])
            p_at_freq = -transfer_func[freq_idx, :, :].T.astype(
                np.complex128)  # (n_d, n_r)
            result = Field(
                data=p_at_freq,
                coords={
                    'depth': raw['depths'],
                    'range': raw['ranges'],
                },
                synthesis_floor=engine.n_time_samples,
                **self._result_kwargs(
                    source,
                    phase_reference=settings.output.phase_reference,
                    backend='oasp',
                    frequencies=float(raw['freq'][freq_idx]),
                    frequencies_available=raw['freq'],
                    source_depth=raw['source_depth'],
                    center_frequency=raw['center_frequency'],
                ),
            )
            result = self._mask_source_axis(result, source)

        self._attach_output_paths(
            result, inputs.work_dir, deck.stem,
            primary_files=(('trf_file', '.trf'),),
        )

        self._log("OASP simulation complete")
        return result

    _FOR_FILES = {
        'FOR002': 'src',
    }
    # '.045'/'.046' are the option-'s' kernel files (SCTOUT, unoasp22.f:1039;
    # unit 46 opened at :253). They are OASSP *inputs* — _run_mean_field
    # requires both from this run's stem — so a pinned work_dir holding a
    # previous sweep's pair would hand oassp2 the wrong kernels, and they are
    # megabytes each.
    _OUTPUT_SUFFIXES = ('.trf', '.dtrf', '.plt', '.plp', '.045', '.046')
    _OUTPUT_FORT_FILES = _SCTOUT_BARE_FORT46
