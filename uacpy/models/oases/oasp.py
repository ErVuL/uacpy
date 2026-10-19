"""OASP: the OASES broadband (pulse) transfer-function program."""

import warnings
import dataclasses
from dataclasses import dataclass
from types import MappingProxyType
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple, Union

import numpy as np

from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.models._extract import (
    _restore_requested_axis, attach_output_paths, finish_broadband,
    mask_source_axis, result_kwargs,
)
from uacpy.models._band import FREQUENCY_ARRAY_FIELDS, BandResolution
from uacpy.models._spec import ModelSpec
from uacpy.models._knobs import is_real_number, positive_finite, whole_count
from uacpy.models._window import (
    check_pinned_window, refuse_inverted_window, scholte_wave_notice,
)
from uacpy.core.run_settings import EngineSettings, OutputSpec, RunMode
from uacpy.core.results import Field, PhaseReference
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, NumericsWarning,
)
from uacpy.io.oases_writer import (
    OASP_FREQ_MAX_PER_CENTER, oasp_centre_frequency, write_oasp_input,
    oases_frequency_sweep, oases_wavenumber_bounds,
    water_ac_anchor_frequency, refuse_unusable_integration_window,
    oases_layer_records,
)
from uacpy.io._parsers import oasp_trf_pressure, parse_oasp_trf
from uacpy.models.base import DEFAULT_RUN_TIMEOUT, _host_options
from uacpy.models.oases._common import (
    _OASES_TRAITS, _SCTOUT_BARE_FORT46, _geometry_options,
    _oases_binary, _reject_plane_geometry_letter_on_a_point_source,
    _reject_wavenumber_overrun, _warn_if_water_ac_extrapolates,
    _check_n_wavenumbers_knob, _offset_ignored_notice, _pinned_count_note,
    _reject_unreadable_trf_options, resolve_integration_window,
)
from uacpy.models.oases._sampling import (
    _oasp_fft_grid, _oasp_marched_frequencies, _oasp_wavenumber_top,
    check_power_of_two_time_samples,
)
from uacpy.models.oases._base import OASES
from uacpy.models._notices import message_notice
from uacpy.core.engine_defaults import (
    OASES_PRESSURE_LETTER, OASP_FREQ_MIN, OASP_INTEGRATION_OFFSET,
    OASP_N_TIME_SAMPLES, default_c_low,
)
from uacpy.models._launch import require_output


#: OASP's option line when no raw ``options`` is given: the bulk pressure
#: (:data:`~uacpy.core.engine_defaults.OASES_PRESSURE_LETTER`) and ``'J'``,
#: the manual's OASP TL deck ``N J`` (oasp.tex:764) with the bulk pressure in
#: place of the normal stress; ``'J'`` keeps a real frequency axis under
#: automatic sampling (oasp.tex:45-46).
_OASP_DEFAULT_OPTIONS = f"{OASES_PRESSURE_LETTER} J"


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
    fc: float
    fc_origin: str
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
    fc : float
        Block III's ``FREQS`` (Hz), the pulse carrier.
    fc_origin : str
        Where ``FREQS`` came from.
    marched_frequencies : ndarray
        The bins (Hz) the ``.trf`` carries (:func:`_oasp_marched_frequencies`).
    c_low, c_high : float
        Block VII's phase-speed window ``CMIN CMAX`` (m/s).
    c_low_origin, c_high_origin : str
        Where ``c_low`` and ``c_high`` came from: the constructor, or
        :func:`~uacpy.io.oases_writer.oases_wavenumber_bounds` — the
        sentence :func:`~uacpy.core.engine_defaults.default_c_low` states for
        ``c_low``, OASES' "no upper limit" 1e8 for ``c_high``.
    n_wavenumbers : int or None
        Block VII's ``NW``; ``None`` is OASES' automatic sampling.
    integration_window : tuple of (int, int) or None
        Block VII's ``IC1 IC2``: the given window, the whole ``(1, NW)``
        under a pinned ``n_wavenumbers``, ``None`` under automatic sampling
        (where OASES sets them).
    integration_offset : float
        The contour offset (dB/wavelength) on the frequency line.
    integrand_plot_step : int or None
        Block VII's ``INTF``; ``None`` writes the writer's 0, the manual's
        worked deck (oasp.tex:582).
    dip_angle_deg : float or None
        The dip-slip source's dip (degrees), read only under ``'4'``.
    notices : tuple of Notice
        What the run says about the sweep it resolved (a constructor edge
        overridden, a pinned ``NX`` coarser than the requested spacing).
    """

    options: str
    requested_frequencies: Optional[np.ndarray]
    n_time_samples: int
    n_time_samples_origin: str
    freq_min: float
    freq_max: float
    freq_max_origin: str
    time_step: float
    fc: float
    fc_origin: str
    marched_frequencies: np.ndarray
    c_low: float
    c_low_origin: str
    c_high: float
    c_high_origin: str
    n_wavenumbers: Optional[int]
    integration_window: Optional[Tuple[int, int]]
    integration_offset: float
    integrand_plot_step: Optional[int]
    dip_angle_deg: Optional[float]

    _ARRAY_FIELDS = FREQUENCY_ARRAY_FIELDS

    def __post_init__(self):
        # A list (the to_dict form) is stored as the tuple a frozen record
        # holds.
        if self.integration_window is not None:
            object.__setattr__(self, 'integration_window',
                               tuple(int(v) for v in self.integration_window))
        super().__post_init__()


def _band_named(mode, source, request) -> bool:
    """Whether the call names the band OASP sweeps — ``run(frequencies=)``,
    or a multi-frequency Source on BROADBAND — rather than the run deriving
    it (:meth:`OASP._requested_frequencies`)."""
    if request is not None and request.kwargs.get('frequencies') is not None:
        return True
    return (mode == RunMode.BROADBAND
            and np.atleast_1d(np.asarray(source.frequencies)).size > 1)


def _check_oasp_knobs(freq_min, freq_max, n_time_samples) -> None:
    """Refuse an OASP ``freq_min`` that is negative or not below
    ``freq_max``, a non-positive ``freq_max``, and an ``n_time_samples``
    that is not a power of two of at least 2
    (:func:`~uacpy.models.oases._sampling.check_power_of_two_time_samples`).

    An inverted pair reaches the deck unvalidated (the writer formats
    whatever it is given) and OASP then computes LX > MX, so the
    frequency loop never executes and the .trf comes back empty rather
    than wrong. The band check of a run compares freq_max against the
    band top only, never against freq_min, so this is the one place the
    ordering is tested. ``n_time_samples`` is the FFT length NX, which a
    fraction would reach truncated."""
    if freq_min is not None and not (
            is_real_number(freq_min) and np.isfinite(freq_min)
            and freq_min >= 0.0):
        raise ConfigurationError(
            f"OASP: freq_min={freq_min!r} must be a finite frequency of at "
            f"least 0 Hz (FR1 of Block VIII).")
    positive_finite('OASP: freq_max (Hz)', freq_max, optional=True)
    if n_time_samples is not None:
        check_power_of_two_time_samples('OASP: n_time_samples',
                                        n_time_samples)
    if freq_min is None:
        freq_min = OASP_FREQ_MIN
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


def _oasp_sweep(mode, source, requested, *, knob_fc,
                knob_freq_min, knob_freq_max,
                knob_n_time_samples, band_named=False) -> _OaspSweep:
    """Block III/VIII of the deck — the carrier, ``NX``, ``FR1``, ``FR2``
    — and the ladder of bins they give, resolved once for a call.

    A requested band (:meth:`OASP._requested_frequencies`) sets the
    sweep:
    ``FR1``/``FR2`` its edges and ``NX`` the smallest power of two whose
    bins ``DLFREQ = 2·FR2/NX`` are as fine as the requested spacing (a
    pinned ``n_time_samples`` as given, with a notice when its bins are
    coarser), with the Nyquist lifted by
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
        # non-equispaced request is refused.
        fmin_user, freq_max, n_user = oases_frequency_sweep(
            requested, 'OASP', argument='frequencies=')
        # Deck fc: the centre of the requested band — the same centre
        # convention the sibling broadband models use (RAM's (fc, Q, T)
        # sweep is parameterised around the band midpoint). A pinned
        # ``fc`` wins.
        if knob_fc is not None:
            fc_run = knob_fc
            fc_origin = 'OASP(fc=…)'
            notices.append(message_notice(
                f"OASP(fc={knob_fc:g}) changes nothing computed: OASP only "
                f"writes FRC to the transfer-function file (oasp.tex:276-278), "
                f"and this run sweeps {fmin_user:g}-{freq_max:g} Hz, the band "
                f"it was asked for.", FallbackWarning))
        else:
            fc_run = oasp_centre_frequency(requested)
            fc_origin = 'centre of the requested band'
        # A pinned edge the call's band does not share: the deck carries
        # one sweep (FR1 FR2, unoasp22.f:176). Beside a band the caller
        # named (``band_named``: run(frequencies=), a multi-frequency
        # Source) the two pins contradict each other; beside a band the run
        # derives (a TIME_SERIES pulse, COHERENT_TL's source frequency) the
        # pinned edge is not used.
        unused = [f"{name}={value:g}" for name, value, edge in
                  (('freq_min', knob_freq_min, fmin_user),
                   ('freq_max', knob_freq_max, freq_max))
                  if value is not None and abs(value - edge) > 1e-9]
        if unused and band_named:
            raise ConfigurationError(
                f"OASP({', '.join(unused)}) pins the sweep and the call "
                f"names the band too ({fmin_user:g}-{freq_max:g} Hz, from "
                f"run(frequencies=) or the Source's frequencies); the deck "
                f"carries one sweep (FR1 FR2, unoasp22.f:176), so the two "
                f"contradict each other.",
                remediation=(
                    "Leave OASP(freq_min=, freq_max=) unset to sweep the "
                    "call's band, or drop the band from the call (a "
                    "one-frequency Source, no frequencies=) to sweep the "
                    "pinned edges."))
        if unused:
            notices.append(message_notice(
                f"OASP({', '.join(unused)}) is not used: this "
                f"{getattr(mode, 'name', mode)} run sweeps "
                f"{fmin_user:g}-{freq_max:g} Hz, the band the run derives "
                f"(the source waveform's, or the bin at the source "
                f"frequency).",
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
            # (oasp.tex:129), so an unpinned NX is that count rounded up to
            # a power of two, two samples more than it so the Nyquist lift
            # below keeps DLFREQ = 2*freq_max/(NX - 2) no coarser than df.
            # A pinned NX is written as given.
            target = int(np.ceil(2.0 * freq_max / df_user)) + 2
            spacing_nx = 1 << (target - 1).bit_length()
            if knob_n_time_samples is None:
                n_time_samples = spacing_nx
                n_time_origin = (
                    f"the power of two reaching the requested spacing "
                    f"{df_user:g} Hz")
            # OASP's top bin is DLFREQ*(NX/2 - 1): MX = FR2/DLFREQ + 2 is
            # clamped to NX/2 (unoasp22.f:237-246), so with the Nyquist at
            # the band's top the last bin falls one DLFREQ short of it
            # (measured: a 90-110 Hz band tops out at 109.95 Hz, a
            # 100-300 Hz Source band at 298.8 Hz). Lifting the Nyquist by
            # NX/(NX - 2) puts that bin on the band's upper edge; for a
            # band, the 1e-6 keeps the .trf's REAL*4 axis (measured
            # 299.9999916 for a 300 Hz edge at 1e-9) from landing it just
            # below. One requested frequency is the ladder's one bin,
            # DLFREQ itself (MX is clamped to NX/2 = 2), so it takes no
            # guard: the bin then lands on the request to REAL*4 (<= 1.2e-7
            # relative from 20 Hz to 2 kHz), where the guard would put it
            # 1e-6 above — 5 deg of phase at 2 kHz and 10 km, and past the
            # substituted-bin notice's rtol=1e-6 at 214 of 397 frequencies.
            guard = 1.0 + 1e-6 if n_user > 1 else 1.0
            if n_time_samples > 2:
                freq_max = (freq_max * n_time_samples
                            / (n_time_samples - 2) * guard)
                freq_max_origin = ('top of the requested band, the '
                                   'Nyquist lifted by NX/(NX - 2)')
            dlfreq = 2.0 * freq_max / n_time_samples
            if (knob_n_time_samples is not None
                    and dlfreq > df_user * (1.0 + 1e-6)):
                notices.append(message_notice(
                    f"OASP(n_time_samples={n_time_samples}) gives bins "
                    f"DLFREQ = 1/(DT*NX) = {dlfreq:.6g} Hz apart "
                    f"(unoasp22.f:237), coarser than the requested spacing "
                    f"{df_user:.6g} Hz; the result carries that ladder. "
                    f"n_time_samples={spacing_nx} or more reaches the "
                    f"requested spacing.",
                    NumericsWarning))
        elif n_time_samples is None:
            n_time_samples = OASP_N_TIME_SAMPLES
    else:
        # Centre convention shared with the sibling broadband models
        # (RAM's (fc, Q, T) sweep centres on the band midpoint).
        src_freqs = np.atleast_1d(
            np.asarray(source.frequencies, dtype=float))
        if knob_fc is not None:
            fc_run = knob_fc
            fc_origin = 'OASP(fc=…)'
        else:
            fc_run = oasp_centre_frequency(src_freqs)
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
        # unless a pinned ``fc`` sits below it, and then a
        # 1000 Hz source with fc=10 writes FR2=25 Hz and
        # comes back labelled with a 25 Hz bin.
        if f_top > freq_max:
            origin = (f"the derived sweep edge freq_max="
                      f"{OASP_FREQ_MAX_PER_CENTER:g}×fc="
                      f"{freq_max:.1f} Hz" if derived_freq_max
                      else f"the pinned sweep edge "
                           f"freq_max={freq_max:.1f} Hz")
            fix = ("Raise fc to the band's own centre (or "
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
        freq_min = (OASP_FREQ_MIN if knob_freq_min is None
                    else knob_freq_min)
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
        fc=float(fc_run),
        fc_origin=fc_origin, ladder=ladder,
        notices=tuple(notices))


def _oasp_settings(mode, requested, *, options, c_low_pinned,
                   c_high_pinned, n_wavenumbers, integration_offset,
                   integration_window,
                   integrand_plot_step, dip_angle_deg,
                   knob_fc, knob_freq_min,
                   knob_freq_max, knob_n_time_samples, env,
                   source, receiver, band_named=False,
                   offset_notice=True, scholte_notice=True) -> OASPSettings:
    """Every value the OASP deck is written from, resolved once on the
    projected ``env`` for a call in ``mode`` asking for ``requested``:
    the option line (with ``'P'`` for a line Source), the sweep
    (:func:`_oasp_sweep`), Block VII's phase-speed window and
    wavenumber count. A window whose edge
    (:func:`_oasp_scholte_notice`) leaves out a Scholte wave the geometry
    can see is noticed, unless the run is the mean field OASSP builds
    (``scholte_notice`` False), whose own fitted-``c_low`` rule speaks."""
    sweep = _oasp_sweep(mode, source, requested,
                        knob_fc=knob_fc,
                        knob_freq_min=knob_freq_min,
                        knob_freq_max=knob_freq_max,
                        knob_n_time_samples=knob_n_time_samples,
                        band_named=band_named)
    options = _geometry_options(
        options if options is not None else _OASP_DEFAULT_OPTIONS, source)
    c_low, c_high = oases_wavenumber_bounds(env)
    if c_low_pinned is not None:
        c_low = c_low_pinned
    if c_high_pinned is not None:
        c_high = c_high_pinned
    refuse_inverted_window('OASP', c_low, c_high, c_low=c_low_pinned,
                           c_high=c_high_pinned)
    c_low_origin = ('OASP(c_low=…)' if c_low_pinned is not None
                    else default_c_low(env)[1])
    c_high_origin = ('OASP(c_high=…)' if c_high_pinned is not None
                     else "1e8, OASES' no upper limit (oast.tex:487, :568)")
    notices = list(sweep.notices)
    # OASP reads the offset token under 'J' or 'd' (unoasp22.f:126-133) and
    # otherwise sets OFFDB = OFFDBIN = 0 at :365. The producer OASSP builds
    # takes OASSP's offset, whose own notice says it (offset_notice False).
    if offset_notice:
        notices.append(_offset_ignored_notice(
            'OASP', integration_offset, n_wavenumbers, options,
            offset_letters='Jd'))
    window, window_notice = resolve_integration_window(
        'OASP', integration_window, n_wavenumbers, 'oasp.tex:677-678')
    notices.append(window_notice)
    if scholte_notice:
        notices.append(_oasp_scholte_notice(
            env, source, receiver, options=options, c_low=float(c_low),
            c_high=float(c_high), n_wavenumbers=n_wavenumbers,
            n_time_samples=sweep.n_time_samples, freq_min=sweep.freq_min,
            freq_max=sweep.freq_max))
    return OASPSettings(
        options=options,
        requested_frequencies=sweep.requested,
        n_time_samples=sweep.n_time_samples,
        n_time_samples_origin=sweep.n_time_samples_origin,
        freq_min=sweep.freq_min,
        freq_max=sweep.freq_max,
        freq_max_origin=sweep.freq_max_origin,
        time_step=1.0 / (2.0 * sweep.freq_max),
        fc=sweep.fc,
        fc_origin=sweep.fc_origin,
        marched_frequencies=sweep.ladder,
        c_low=float(c_low),
        c_low_origin=c_low_origin,
        c_high=float(c_high),
        c_high_origin=c_high_origin,
        n_wavenumbers=n_wavenumbers,
        integration_window=window,
        integration_offset=integration_offset,
        integrand_plot_step=integrand_plot_step,
        dip_angle_deg=dip_angle_deg,
        notices=tuple(n for n in notices if n is not None),
    )


def _oasp_scholte_notice(env, source, receiver, *, options, c_low, c_high,
                         n_wavenumbers, n_time_samples, freq_min,
                         freq_max):
    """The Scholte-wave notice of one OASP run
    (:func:`~uacpy.models._window.scholte_wave_notice`), on the edge its
    deck reaches at the band's top frequency FREQM
    (:func:`~uacpy.models.oases._sampling._oasp_wavenumber_top`), one deck
    per source depth; ``None`` when no deck leaves a wave out."""
    grid = _oasp_fft_grid(n_time_samples, freq_min, freq_max,
                          1.0 / (2.0 * freq_max))
    freq_top = (grid.last_bin - 1) / (grid.time_step * grid.n_time_samples)
    layers = oases_layer_records(env)
    for i in range(np.atleast_1d(source.depths).size):
        one = source.at_depth(i)
        top = _oasp_wavenumber_top(
            layer_speeds=layers, c_low=c_low, c_high=c_high,
            source_depth=float(np.atleast_1d(one.depths)[0]),
            receiver_depths=receiver.depths,
            receiver_ranges=receiver.ranges,
            n_time=grid.n_time_samples, dt=grid.time_step,
            freq_top=freq_top, options=options,
            n_wavenumbers=n_wavenumbers)
        how = (f"c_low = {c_low:.1f} m/s with a pinned n_wavenumbers "
               f"(unoasp22.f:394)" if n_wavenumbers is not None else
               f"AUTSAM's CMIN for c_low = {c_low:.1f} m/s, the window "
               f"2π·FREQM/c_low plus its taper (unoasp22.f:1125, :1136)")
        notice = scholte_wave_notice(
            env, one, receiver, top.edge, top.freq_top, model_name='OASP',
            edge_text=(f"{top.edge:.1f} m/s at the {top.freq_top:.2f} Hz "
                       f"band top: {how}"),
            remediation="Leave c_low unset (its derived value sits under "
                        "every Scholte wave) or pass c_low below the wave "
                        "speed to include it.",
            wavenumber_edge=top.k_max if top.fixed else None)
        if notice is not None:
            return notice
    return None


def _write_oasp_deck(deck: Path, inputs, name: str) -> None:
    """The OASP deck at ``deck``, written by
    :func:`~uacpy.io.oases_writer.write_oasp_input` from the settings
    ``inputs`` carries; ``name`` is the model the water-absorption
    notice names."""
    engine = inputs.settings.engine
    writer_kwargs: dict = {
        'integration_offset': engine.integration_offset,
        'n_wavenumbers': engine.n_wavenumbers,
        'integration_window': engine.integration_window,
        'freq_min': engine.freq_min,
        'fc': engine.fc,
        'time_step': engine.time_step,
        'c_low': engine.c_low,
        'c_high': engine.c_high,
    }
    if engine.integrand_plot_step is not None:
        writer_kwargs['integrand_plot_step'] = \
            engine.integrand_plot_step
    if engine.dip_angle_deg is not None:
        writer_kwargs['dip_angle_deg'] = engine.dip_angle_deg
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
    ``tf.to_time_trace(depth=…, range=…, sample_rate=…)``. A line Source runs in OASP's plane
    geometry (option ``'P'``). For range-dependent problems, RAM is
    recommended.

    The returned field is OASES' output parameter ``'K'``, the bulk
    pressure: the acoustic pressure in a fluid (``oasaun31.f:1448-1449``,
    ``:1471-1472``) and, for a receiver inside an elastic layer, the bulk
    pressure of the compressional potential alone (``oasaun31.f:1509-1513``,
    oasp.tex:178-183). A raw ``options`` line with the normal stress ``'N'``
    instead returns ``-σzz`` (oasp.tex:185), the pressure in a fluid but,
    inside an elastic layer, a stress built from both the compressional and
    the shear potentials (``oasaun31.f:1489-1491``).

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Per-model: ``'ssp': 'mean'`` (the layer stack is kept).

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
    # The band edges follow run(frequencies=…) when the call passes one,
    # and fc is the centre of that band or of the Source's frequencies.
    _UNPINNED_FIELDS = frozenset({'options', 'freq_min', 'freq_max', 'fc'})
    #: True on the mean-field producer OASSP builds itself: it carries
    #: OASSP's ``integration_offset``, whose notice OASSP's own settings
    #: give, so this run's settings do not repeat it.
    _offset_said_by_owner = False
    #: True on that producer too: OASSP's fitted-``c_low`` rule speaks for
    #: the window its mean field carries, so this run gives no Scholte-wave
    #: notice of its own.
    _scholte_said_by_owner = False

    # A line Source runs in OASP's plane geometry, option 'P' (ICDR = 1,
    # unoasp22.f:946-948), which _resolve_engine_settings writes for it.
    spec = ModelSpec(
        modes=(RunMode.COHERENT_TL, RunMode.BROADBAND, RunMode.TIME_SERIES),
        # A rough interface is column 7 (RG) of the layer record
        # (oaseun31.f:54).
        supports={'layered_bottom', 'elastic_media',
                  'rough_surface', 'rough_bottom'},
        # A line Source runs in the plane geometry, option 'P'.
        source_types=frozenset({'point', 'line'}),
        collapse={'ssp': 'mean'},
        traits=dataclasses.replace(
            _OASES_TRAITS,
            consumes_run_t_start=True,
            consumes_single_mode_frequencies=True,
            # BROADBAND runs OASP's own FFT ladder, not the base's
            # ``fc·(1 ± 0.25)`` band.
            announced_band_modes=frozenset({RunMode.TIME_SERIES}),
        ),
    )
    provenance_id = 'oases'
    _BINARIES = MappingProxyType({'oasp': _oases_binary('oasp')})
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
        freq_min: Optional[float] = None,
        freq_max: Optional[float] = None,
        fc: Optional[float] = None,
        integrand_plot_step: Optional[int] = None,
        options: Optional[str] = None,
        integration_offset: float = OASP_INTEGRATION_OFFSET,
        n_wavenumbers: Optional[int] = None,
        integration_window: Optional[Tuple[int, int]] = None,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        dip_angle_deg: Optional[float] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: Optional[float] = DEFAULT_RUN_TIMEOUT,
        collapse: Optional[Dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to OASP binary. Auto-detected if ``None``.
        n_time_samples : int, optional
            FFT length NX (samples per receiver trace, a power of two).
            ``None`` (default) takes 4096 for the default
            sweep of a one-carrier BROADBAND run (uacpy guidance: OASES
            requires only ``NT = 2^M``, oasp.tex:129; its worked decks use
            2048, oasp.tex:583, and 1024, oasp.tex:773), and for a requested
            band —
            ``run(frequencies=)``, a TIME_SERIES pulse, a multi-element
            ``source.frequencies``, the source frequency of a COHERENT_TL
            run — the smallest power of two whose bins are as fine as the
            requested spacing, so the TIME_SERIES record is the
            ``output_duration`` asked for, rounded up to that power of two.
            A value given is written as given in every case, and must be a
            power of two (oasp.tex:129 ``NT = 2^M``); a notice says when
            its bins are coarser than a requested spacing.
        freq_min : float, optional
            Lower edge of the OASP broadband sweep (Hz), at least 0 and
            below ``freq_max`` when that is pinned. ``None`` (default) is
            0, the manual's worked deck (oasp.tex:583). A pinned edge beside
            a band the call names (``run(frequencies=)``, a
            multi-frequency Source) is refused, naming both; beside a band
            the run derives (a TIME_SERIES pulse, COHERENT_TL's source
            frequency) the run says it is not used. ``freq_max`` follows the
            same rule.
        freq_max : float, optional
            Upper edge of the OASP broadband sweep (Hz). ``None``
            (default) derives ``2.5 ×`` the centre frequency at
            ``run()`` time; a pinned value below the centre frequency
            or the top of the requested band raises at ``run()``.
        fc : float, optional
            Carrier frequency for the pulse (Hz). ``None`` defaults to
            the centre (midpoint) of the run's frequency band at
            ``run()`` time — a single ``source.frequencies`` entry is
            its own centre: the manual's FRC, "Center frequency of
            source" (oasp.tex:84). OASP itself only records it
            (oasp.tex:278-280); uacpy derives the default ``freq_max``
            from it.
        integrand_plot_step : int, optional
            OASP's ``INTF`` (Block VII, ``unoasp22.f:160``). It gates how
            often the wavenumber *integrand* is plotted
            (``unoasp22.f:591``: ``IF (MOD(JJ-LXP1,INTF).EQ.0) KPLOT=1``,
            consumed at ``unoasp22.f:678``) and does **not** decimate the
            ``.trf`` frequency axis, which always carries every bin
            ``LXP1..MX``. A whole number >= 0 (``IF >= 0``, oasp.tex:125).
            ``None`` → 0, the manual's worked deck (oasp.tex:582;
            :data:`~uacpy.core.engine_defaults.OASP_INTEGRAND_PLOT_STEP`),
            which plots no integrand and writes none of the plot files a
            value > 0 turns on (``unoasp22.f:173``).
        options : str, optional
            Raw OASES option string. ``None`` takes ``'K J'`` (the bulk
            pressure, :data:`~uacpy.core.engine_defaults.
            OASES_PRESSURE_LETTER`); the ``'J'``
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
            Wavenumber sample count. ``None`` lets OASP auto-pick (NW = -1,
            the automatic sampling oasp.tex:671-674 offers "without the
            usual convergence testing"). A pinned count is written as given,
            with an info-level note (``verbose='info'``) that the manual
            leaves it to the convergence test.
        integration_window : (int, int), optional
            ``(IC1, IC2)``, the first and last wavenumber samples
            integrated: the transform is Hanning-windowed outside them
            (oasp.tex:652-660). Two whole numbers with ``1 <= IC1 <= IC2 <=
            n_wavenumbers`` (oasp.tex:123-124); it applies with a pinned
            ``n_wavenumbers``, and under automatic sampling, where they
            "have no effect" (oasp.tex:677-678), the run says so. ``None``
            integrates the whole window.
        c_low, c_high : float, optional
            ``CMIN``/``CMAX`` (m/s), Block VII's phase-speed window.
            ``None`` derives the pair as :class:`OAST` does — ``c_low``
            0.90 × the slowest wave of the problem, shear speeds and
            Scholte waves included; ``c_high`` 1e8. A window whose edge at
            the band top (:func:`_oasp_scholte_notice`) lies above a
            Scholte wave the source or a receiver is within one
            wavelength of is noticed.
        dip_angle_deg : float, optional
            Fault dip angle (degrees) for the dip-slip moment source that
            option ``'4'`` selects (``unoasp22.f:993-995``). INSRC reads it
            from the source record — as the second token without ``'L'``
            (``oaseun31.f:1114``), as the seventh with it
            (``oaseun31.f:1089``). ``None`` writes 0 when ``'4'`` is
            present and nothing when it is not; a value set without
            ``'4'`` is refused.
        use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
            Standard plumbing (see :class:`PropagationModel`).
        The receiver ranges are ``receiver.ranges``: Block VIII's
        ``R0 RSPACE NPLOTS`` are read from them.
        """
        super().__init__(**_host_options(locals()))
        self.n_time_samples = n_time_samples
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.fc = float(fc) if fc is not None else None
        # INTF is an INTEGER plot decimation (oasp.tex:125, IF >= 0).
        if integrand_plot_step is not None:
            whole_count('OASP: integrand_plot_step', integrand_plot_step, 0)
        self.integrand_plot_step = (
            int(integrand_plot_step)
            if integrand_plot_step is not None else None
        )
        self.options = options
        self.integration_offset = float(integration_offset)
        self.n_wavenumbers = n_wavenumbers
        # Block VII's IC1/IC2, stored as given (checked by _check_knobs).
        self.integration_window = integration_window
        # Block VII's CMIN/CMAX; ``None`` keeps the derivation, as on
        # OAST.
        self.c_low = float(c_low) if c_low is not None else None
        self.c_high = float(c_high) if c_high is not None else None
        # DA, the dip angle INSRC reads only under the '4' (dip-slip) option.
        self.dip_angle_deg = float(dip_angle_deg) if dip_angle_deg is not None else None
        self._check_knobs()
        if self.n_wavenumbers is not None:
            self._log(_pinned_count_note('OASP', self.n_wavenumbers))
        if n_time_samples is not None:
            self.n_time_samples = int(n_time_samples)
        if freq_min is not None:
            self.freq_min = float(freq_min)
        if freq_max is not None:
            self.freq_max = float(freq_max)

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        self._exe = self._resolve_executable(
            executable, lambda: self._find_binary('oasp'),
        )

    def _check_knobs(self) -> None:
        """Refuse a band or FFT length no run could use
        (:func:`_check_oasp_knobs`), and a pinned phase-speed window that is
        not positive or not ordered
        (:func:`~uacpy.models._window.check_pinned_window`). Run at construction and again by every
        run (:meth:`_validate_engine`), since the attributes can be
        reassigned in between."""
        _check_n_wavenumbers_knob(self.n_wavenumbers)
        check_pinned_window('OASP', c_low=self.c_low, c_high=self.c_high)
        refuse_unusable_integration_window('OASP', self.integration_window,
                                           self.n_wavenumbers)
        _check_oasp_knobs(self.freq_min, self.freq_max, self.n_time_samples)

    def _validate_engine(self, env, source, receiver, run_mode, *,
                         request=None) -> None:
        """Stage 2: the family's boundary refusals
        (:meth:`OASES._validate_engine`), then the refusals of an option
        line the ``.trf`` reader cannot
        read back (:func:`~uacpy.models.oases._common._reject_unreadable_trf_options`)
        and of a raw ``'P'`` with a point Source."""
        super()._validate_engine(env, source, receiver, run_mode,
                                 request=request)
        _reject_unreadable_trf_options('OASP', self.options,
                                       self.n_wavenumbers)
        _reject_plane_geometry_letter_on_a_point_source(
            'OASP', (self.options if self.options is not None
                     else _OASP_DEFAULT_OPTIONS),
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
        BROADBAND call runs that ladder, not the base's ``fc·(1 ± 0.25)``
        band (``spec.traits.announced_band_modes``).
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
            integration_window=self.integration_window,
            integrand_plot_step=self.integrand_plot_step,
            dip_angle_deg=self.dip_angle_deg,
            knob_fc=self.fc,
            knob_freq_min=self.freq_min, knob_freq_max=self.freq_max,
            knob_n_time_samples=self.n_time_samples,
            env=env, source=source, receiver=receiver,
            band_named=_band_named(settings.mode, source, request),
            offset_notice=not self._offset_said_by_owner,
            scholte_notice=not self._scholte_said_by_owner)

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
        require_output(
            self.model_name,
            [inputs.work_dir / f'{deck.stem}{suffix}'
             for suffix in self._TRF_SUFFIXES],
            what='a transfer-function file', process=proc,
        )

    def _read_output(self, inputs, deck: Path) -> dict:
        """Stage 4: the transfer function on the requested axes
        (:func:`_read_trf_on_requested_ranges`)."""
        output_file = require_output(
            self.model_name,
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

        The raw ``.trf`` payload is read by
        :func:`~uacpy.io._parsers.oasp_trf_pressure`: the bulk pressure
        ``'K'`` as written, the normal stress ``'N'`` ("N  Normal stress
        sigma_zz (= -p in fluids)", oases/doc/oasp.tex:185) negated. OASES
        runs on the same e^{+i omega t} convention as the other engines, so
        that is the whole conversion: measured against Scooter's Hankel path
        on the same Pekeris case, the negated 'N' gives arg(-N/Scooter)
        within 3.2 deg of 0 with the magnitude ratio 0.94-1.02, so the
        'travelling_wave' tag means the convention every coherent engine
        shares (field.exe gets a sign of its own in kraken/_extract.py's
        assemble_field_from_shd).
        """
        settings = inputs.settings
        engine = settings.engine
        source = inputs.source
        # (n_depth, n_range, n_frequencies): the trailing axis is the
        # variable dim.
        pressure = oasp_trf_pressure(raw, 'OASP', deck.with_suffix('.trf'))
        if settings.mode in (RunMode.BROADBAND, RunMode.TIME_SERIES):
            _warn_if_trf_grid_replaced_request(engine.requested_frequencies,
                                               raw['freq'])
            result = Field(
                data=pressure,
                coords={
                    'depth': raw['depths'],
                    'range': raw['ranges'],
                    'frequency': raw['freq'],
                },
                # OASP's FFT length NX: the synthesis floors its own at it.
                synthesis_floor=engine.n_time_samples,
                **result_kwargs(
                    self.model_name, self.provenance, source,
                    phase_reference=PhaseReference.TRAVELLING_WAVE.value,
                    backend='oasp',
                    frequencies=raw['freq'],
                    source_depth=raw['source_depth'],
                    fc=raw['fc'],
                ),
            )
            result = mask_source_axis(self.model_name, result, source)
            if settings.time is not None:
                result = finish_broadband(self.model_name, result, settings)
        else:
            # COHERENT_TL: the bin at the source frequency, as complex
            # narrowband pressure on (n_depth, n_range). Users get TL via
            # ``field.dB`` or ``.to_dB()``.
            f_req = float(engine.requested_frequencies[0])
            freq_idx = _nearest_bin(raw['freq'], f_req)
            # A ladder bin off the request is announced, as the BROADBAND
            # branch announces its axis.
            _warn_if_trf_grid_replaced_request(f_req, raw['freq'][freq_idx])
            result = Field(
                data=pressure[:, :, freq_idx],
                coords={
                    'depth': raw['depths'],
                    'range': raw['ranges'],
                },
                synthesis_floor=engine.n_time_samples,
                **result_kwargs(
                    self.model_name, self.provenance, source,
                    phase_reference=settings.output.phase_reference,
                    backend='oasp',
                    frequencies=float(raw['freq'][freq_idx]),
                    frequencies_available=raw['freq'],
                    source_depth=raw['source_depth'],
                    fc=raw['fc'],
                ),
            )
            result = mask_source_axis(self.model_name, result, source)

        attach_output_paths(
            result, inputs.work_dir, deck.stem,
            primary_files=(('trf_file', '.trf'),),
            cleanup=self.cleanup,
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
