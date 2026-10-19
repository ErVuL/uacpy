"""OASSP: the OASES rough-interface scattered-field pulse program."""

import io
import dataclasses
from dataclasses import dataclass
from types import MappingProxyType
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from uacpy.models._band import BandResolution
from uacpy.models._knobs import whole_count
from uacpy.models._spec import ModelSpec
from uacpy.models._window import check_pinned_window
from uacpy.core.run_settings import (
    EngineSettings, Notice, OutputSpec, RunMode, RunSettings,
)
from uacpy.core.source import Source
from uacpy.core.results import Field, PhaseReference
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, ModelExecutionError,
    NumericsWarning, UnsupportedFeatureError, ValidityWarning,
)
from uacpy.io.oases_writer import (
    write_oassp_input, oases_wavenumber_bounds, bottom_interface_roughness,
    oassp_bottom_interfaces, refuse_unintegrable_spectral_exponent,
    refuse_unusable_correlation_length, refuse_unusable_integration_window,
)
from uacpy.io.oases_reader import OasesRhsHeader
from uacpy.models.base import DEFAULT_RUN_TIMEOUT, _host_options
from uacpy.models.oases._common import (
    C_LOW_MATCH_RTOL,
    _check_n_wavenumbers_knob, _OASES_TRAITS, _OASS_MIN_MEAN_FIELD_ROUGHNESS, _OASS_SPECTRA,
    _pinned_count_note,
    _geometry_options, _oases_binary, _option_letters,
    _reject_plane_geometry_letter_on_a_point_source,
    _reject_unreadable_trf_options, _require_mean_field_rhs,
    _reject_wavenumber_overrun,
    SCATTERING_UNPINNED_FIELDS, resolve_spectral_exponent,
    _roughness_notice, _offset_ignored_notice, resolve_integration_window,
)
from uacpy.models.oases._sampling import (
    _OASSP_MAX_WAVENUMBERS, _oasp_fft_grid, _oassp_fitting_c_low,
    _oassp_wavenumber_count, _oassp_wavenumber_sampling,
    check_power_of_two_time_samples,
)
from uacpy.io._parsers import oasp_trf_pressure
from uacpy.models.oases.oasp import (
    OASP, _OASP_DEFAULT_OPTIONS, _read_trf_on_requested_ranges,
)
from uacpy.models.oases._mean_field import _ScatteringPostProcessor
from uacpy.models._notices import message_notice
from uacpy.core.engine_defaults import (
    OASES_PRESSURE_LETTER, OASSP_INTEGRATION_OFFSET, OASSP_REALIZATION,
    default_c_low, scholte_waves,
)
from uacpy.models._projection import collapse_elastic_boundary, has_shear
from uacpy.models._extract import (
    attach_output_paths, finish_broadband, mask_source_axis, result_kwargs,
)
from uacpy.models._launch import require_output


#: File root of the deck and outputs of the scattering half of an OASSP run.
_OASSP_BASE_NAME = 'oassp_run'


#: Largest mean-field wavenumber count oassp2 reads: CALSRP/CALSVC read one
#: ``.rhs`` record per mean-field wavenumber (NKMEAN) into arrays of
#: ``nnkr`` = 8192 (``comvol.f:3``) and stop past it with ``'>>> NKMEAN too
#: large'`` (``oasvun31.f:61-65``, ``:319-323``).
_OASSP_MAX_MEAN_FIELD_WAVENUMBERS = 8192


@dataclass(frozen=True, eq=False)
class OASSPSettings(EngineSettings):
    """The settings one :class:`OASSP` run resolved, before launching:
    ``OASSP(...).run_settings(env, source, receiver).engine``, and
    ``result.run_settings.engine`` on the field it produced.

    Block VIII (NT, FR1, FR2, DT) and the scattering interface are the
    values the mean field's ``.rhs`` will carry, which the binary itself
    reads (``unoassp30.f:181-188``, ``:548-549``): resolved here from the
    mean field's settings and the environment, and checked against the
    ``.rhs`` once the mean field has run.

    Attributes
    ----------
    options : str
        The OASSP deck's option line (Block II), ``'P'`` included for a
        line Source.
    interface : int
        The scattering interface, in the OASP/OASSP deck's layer numbering:
        the environment's one rough bottom interface, which the mean field's
        ``.rhs`` names (``oaseun31.f:2306-2310``).
    mean_field_grid : tuple of (int, float, float, float)
        ``(NX, FR1, FR2, DT)`` the mean field runs on and writes to its
        ``.rhs`` header (``unoasp22.f:194-254``).
    n_integrated_wavenumbers : tuple of int
        NWVNO oassp2 integrates on (:func:`_oassp_wavenumber_count`), one
        per source depth (each depth is its own deck), each at most
        :data:`_OASSP_MAX_WAVENUMBERS`.
    correlation_length, spectral_exponent : float
        ``CL`` (m) and ``M`` of the roughness power spectrum.
    rms_roughness : float or None
        ``|RG|`` (m) on the OASSP deck; ``None`` takes the environment's.
    realization : int
        Block VII's fourth token, the realization seed ``k``.
    c_low : float
        Block VII's ``CMIN`` (m/s): the pinned value, else the documented
        rule's (:func:`~uacpy.core.engine_defaults.default_c_low`) or, when
        that one's wavenumber count passes :data:`_OASSP_MAX_WAVENUMBERS`
        over a shear seabed, the lowest whole m/s whose count fits
        (:func:`_fitted_c_low`).
    c_low_origin : str
        Where ``c_low`` came from.
    c_high : float
        Block VII's ``CMAX`` (m/s), which cylindrical geometry (no ``'P'``)
        replaces with 1e12 (``unoassp30.f:205-217``).
    c_high_origin : str
        Where ``c_high`` came from: the constructor, or OASES' "no upper
        limit" 1e8 (:func:`~uacpy.io.oases_writer.oases_wavenumber_bounds`).
    n_wavenumbers : int or None
        Block VII's ``NW``; ``None`` is OASES' automatic sampling.
    integration_window : tuple of (int, int) or None
        Block VII's ``IC1 IC2`` on the OASSP deck: the given window, the
        whole ``(1, NW)`` under a pinned ``n_wavenumbers``, ``None`` under
        automatic sampling (where OASES sets them).
    integration_offset : float
        The contour offset (dB/wavelength) on the frequency line.
    mean_field : RunSettings
        The settings of the mean-field OASP run whose ``.rhs`` the deck
        reads.
    notices : tuple of Notice
        What the run says about the chain it resolved (a CMIN that differs
        from the mean field's, a fitted ``c_low``).
    """

    options: str
    interface: int
    mean_field_grid: Tuple[int, float, float, float]
    n_integrated_wavenumbers: Tuple[int, ...]
    correlation_length: float
    spectral_exponent: float
    rms_roughness: Optional[float]
    realization: int
    c_low: float
    c_low_origin: str
    c_high: float
    c_high_origin: str
    n_wavenumbers: Optional[int]
    integration_window: Optional[Tuple[int, int]]
    integration_offset: float
    mean_field: RunSettings

    def __post_init__(self):
        # The to_dict form holds lists; the frozen record tuples.
        if self.integration_window is not None:
            object.__setattr__(self, 'integration_window',
                               tuple(int(v) for v in self.integration_window))
        object.__setattr__(self, 'mean_field_grid', tuple(
            t(v) for t, v in zip((int, float, float, float),
                                 self.mean_field_grid)))
        object.__setattr__(self, 'n_integrated_wavenumbers',
                           tuple(int(n) for n in self.n_integrated_wavenumbers))
        super().__post_init__()

    @classmethod
    def from_dict(cls, d) -> 'OASSPSettings':
        d = dict(d)
        d['mean_field'] = RunSettings.from_dict(d['mean_field'])
        return super().from_dict(d)


#: The option line of the OASP mean field OASSP builds when none is supplied:
#: OASP's own default letters (:data:`~uacpy.models.oases.oasp.
#: _OASP_DEFAULT_OPTIONS`) and ``'s'``, which writes the ``.rhs`` OASSP reads
#: (unoasp22.f:250-254).
_OASSP_MEAN_FIELD_OPTIONS = f"{_OASP_DEFAULT_OPTIONS} s"


def _check_oassp_knobs(*, correlation_length,
                       spectral_exponent, roughness_spectrum,
                       mean_field, mean_field_knobs, options,
                       scattered_only) -> None:
    """Refuse an OASSP knob no run could use: a missing, negative (volume),
    zero or non-finite correlation length, an unintegrable spectral exponent, an unknown
    roughness spectrum, a supplied ``mean_field`` whose option line has no
    ``'s'`` (:func:`~uacpy.models.oases._common._require_mean_field_rhs`),
    FFT knobs beside the ``mean_field`` that owns
    them (``mean_field_knobs``, name to value), and a raw ``options``
    string beside the typed flags it would discard."""
    if correlation_length is None:
        raise ConfigurationError(
            "OASSP requires correlation_length: the roughness power "
            "spectrum is what it integrates, and OASES reads CL from this "
            "deck rather than adopting it from the mean-field run "
            "(oass.tex:182-183).",
            remediation="Pass correlation_length in metres, e.g. "
                        "OASSP(correlation_length=5.0).",
        )
    # A negative length is OASES' volume-scattering switch; zero and a
    # non-finite value are no length. The deck writer refuses the same.
    refuse_unusable_correlation_length('OASSP', correlation_length)
    if spectral_exponent is not None:
        refuse_unintegrable_spectral_exponent('OASSP', spectral_exponent)
    if str(roughness_spectrum).lower() not in _OASS_SPECTRA:
        raise ConfigurationError(
            f"OASSP(roughness_spectrum={roughness_spectrum!r}) is "
            f"not a roughness "
            f"spectrum OASES offers; GETOPT has one letter, 'g' for "
            f"Goff-Jordan (unoassp30.f:1034), and the default is "
            f"Gaussian.",
            remediation=f"Pass one of {sorted(_OASS_SPECTRA)}.",
        )
    if mean_field is not None:
        _require_mean_field_rhs(
            'OASSP', mean_field,
            (mean_field.options if mean_field.options is not None
             else _OASP_DEFAULT_OPTIONS),
            'unoasp22.f:250-254', f"OASP(options={_OASSP_MEAN_FIELD_OPTIONS!r})")
        pinned = [n for n, value in mean_field_knobs.items()
                  if value is not None]
        if pinned:
            raise ConfigurationError(
                f"OASSP: mean_field= owns the FFT grid, so "
                f"{', '.join(pinned)} would be discarded silently — "
                f"OASSP reads NT/FR1/FR2/DT back out of the .rhs the "
                f"mean field wrote (unoassp30.f:181-188). Set them on "
                f"the mean-field model instead.",
                remediation=(f"OASSP(mean_field=OASP("
                             f"{'=…, '.join(pinned)}=…))."),
            )
    if options is not None:
        pinned = [n for n, off in (
            ('roughness_spectrum',
             str(roughness_spectrum).lower() != 'gaussian'),
            ('scattered_only', not scattered_only)) if off]
        if pinned:
            raise ConfigurationError(
                f"OASSP: options={options!r} replaces the whole "
                f"option line, so {', '.join(pinned)} would be discarded "
                f"silently. Pass either the raw string or the typed "
                f"flags, not both."
            )


def _oassp_wavenumber_echo_scale(options: str) -> int:
    """2 in plane geometry, where the echo under-reports the real grid.

    ``unoassp30.f:327`` prints AUTSAM's count, and the plane-geometry
    branch immediately below (``:337-341``) doubles it — "scattering
    kernels not symmetric", so the integration also runs over negative
    wavenumbers: ``nwvno = 2*nwvno`` with ``ICUT2 = NWVNO``. An echoed
    40000 is 80000 integrated, already past ``NP``, so a guard reading
    the echo verbatim returns silently on exactly the runs it exists to
    catch. Cylindrical geometry keeps ``icut2 = nwvno`` and needs no
    correction.
    """
    return 2 if 'P' in _option_letters(options) else 1


def _oassp_options(options, scattered_only, roughness_spectrum) -> str:
    """OASSP option letters from the typed flags, or the raw string. The
    geometry letter ``'P'`` is the Source's
    (:meth:`OASSP._resolve_engine_settings`)."""
    if options is not None:
        return options
    letters = [OASES_PRESSURE_LETTER, 'J']
    if scattered_only:
        letters.append('s')
    if str(roughness_spectrum).lower() == 'goff-jordan':
        letters.append('g')
    return ' '.join(letters)


def _mean_field_source(source: Source) -> Source:
    """The Source the producer runs: one unit-weight depth of the call's
    (the weight belongs to the run the user called), its geometry
    included, so a line Source runs the mean field in plane geometry
    too."""
    return source.at_depth(0)


def _deck_wavenumber_count(deck_text: str, receiver) -> int:
    """NWVNO an OASSP deck makes oassp2 integrate on
    (:func:`_oassp_wavenumber_count` on :func:`_deck_sampling`)."""
    return _oassp_wavenumber_count(**_deck_sampling(deck_text, receiver))


def _deck_sampling(deck_text: str, receiver) -> dict:
    """The keywords of :func:`_oassp_wavenumber_count` an OASSP deck
    carries.

    Read back from the deck's text itself — title, option line, frequency line,
    ``NL``, the ``NL`` layer records (depth, cp first), the source record
    (``SD`` first), the receiver block, then Blocks VI-VIII as its last
    three records — so the prediction sees exactly the CMIN, CMAX,
    sampling and range axis the binary reads, rather than re-resolving
    them. Receiver depths come from ``receiver`` (the receiver block may
    be an explicit list over several records).
    """
    lines = [ln.split() for ln in deck_text.splitlines() if ln.strip()]
    options = ''.join(lines[1])
    n_layers = int(lines[3][0])
    layers = [(float(rec[0]), float(rec[1]))
              for rec in lines[4:4 + n_layers]]
    source_depth = float(lines[4 + n_layers][0])
    cmin, cmax = (float(t) for t in lines[-3][:2])
    nw = int(lines[-2][0])
    nx, _fr1, fr2, dt, r0_km, rspace_km, nplots = lines[-1][:7]
    return dict(
        layer_speeds=layers, c_low=cmin, c_high=cmax,
        source_depth=source_depth,
        receiver_depths=np.asarray(receiver.depths, dtype=float),
        r0_km=float(r0_km), rspace_km=float(rspace_km),
        nplots=int(nplots), n_time=int(nx), freq_max=float(fr2),
        dt=float(dt), options=options, n_wavenumbers=nw)


def _reject_roughness_wavenumber_overflow(deck_text: str,
                                          receiver) -> int:
    """NWVNO of the deck, refused when it overruns oassp2's roughness
    arrays (stage 3, from the deck text the settings write).

    See :data:`_OASSP_MAX_WAVENUMBERS`: past 4096 wavenumbers the binary
    dies with SIGSEGV in ``pv_`` or writes a ``.trf`` that cannot be read.
    Measured on a 200 m guide (1500-1520 m/s over a rough 1600 m/s
    half-space), 256 samples at 1 ms, receivers at 30/60 m: ranges
    500-1500 m give 2243 wavenumbers and run; 500-3000 m give 4216 and
    crash. Pinning ``n_wavenumbers=4096`` on the crashing grid runs and
    returns a finite field (rms |H| 1.84e-4, against 1.95e-4 for the
    500-1500 m automatic run); 2048 under-resolves the integral (rms 4.8
    dB below 4096 on the same grid), so the remedy names the bound
    itself.
    """
    predicted = _deck_wavenumber_count(deck_text, receiver)
    if predicted <= _OASSP_MAX_WAVENUMBERS:
        return predicted
    raise ConfigurationError(
        f"OASSP: this deck makes oassp2 integrate on {predicted} "
        f"wavenumbers, past the {_OASSP_MAX_WAVENUMBERS} its roughness "
        f"branch can hold — it rounds the count up to a power of two "
        f"(unoassp30.f:542-544) and PV then indexes arrays of "
        f"nnkr = 8192 up to twice that (oasvun31.f:2273, :2532-2533; "
        f"comvol.f:3), so the binary crashes with SIGSEGV or writes an "
        f"unreadable .trf. The automatic count grows with the reference "
        f"range min(cref*NX*DT + 2*r_max, 6*r_max) (unoassp30.f:287-306).",
        remediation=(
            f"Pin n_wavenumbers={_OASSP_MAX_WAVENUMBERS} (fewer samples "
            f"under-resolve the wavenumber integral), or shorten the "
            f"receiver range extent, or shorten the mean field's time "
            f"window NX*DT (fewer n_time_samples; the cref*NX*DT "
            f"term)."),
    )


def _compressional_c_low(env) -> float:
    """The ``c_low`` (m/s) :func:`~uacpy.core.engine_defaults.default_c_low`
    gives on ``env`` with every shear speed left out (its media collapsed
    to fluids, as an engine without shear collapses them): 0.90 × the
    slowest compressional speed ("phase speed must be some 10% below
    slowest acoustic speed", ``scooter.f90:258``). Equal to the derived
    value itself when no medium carries shear."""
    fluid = env.copy()
    if fluid.surface is not None and has_shear(fluid.surface):
        fluid.surface = collapse_elastic_boundary(fluid.surface, 'fluid')
    if fluid.bottom.is_elastic:
        fluid.bottom = collapse_elastic_boundary(fluid.bottom, 'fluid')
    return default_c_low(fluid)[0]


def _fitted_c_low(name, env, source, receiver, engine):
    """``(c_low, origin, notice)`` an unpinned OASSP ``c_low`` takes when
    the documented rule's value (``engine.c_low``,
    :func:`~uacpy.core.engine_defaults.default_c_low`) makes a source
    depth's deck pass :data:`_OASSP_MAX_WAVENUMBERS`; ``None`` when every
    deck fits, or when no value fits.

    The value is the lowest whole m/s whose count fits on every source
    depth's deck (:func:`_oassp_fitting_c_low`), at most the rule's value
    with the shear speeds left out (:func:`_compressional_c_low`), so it gives
    up only phase speeds the seabed's shear brings into the integral. Over
    a seabed without shear the two values are equal and nothing fits; a
    pinned ``n_wavenumbers`` fixes the count whatever ``c_low`` is. Either
    way :func:`_reject_roughness_wavenumber_overflow` then refuses the run.
    The notice names oassp2's bound and where the fitted deck's window
    stops: oassp2 fixes it in wavenumber at the band's top frequency,
    ``WKMAX = 2π·FREQM/CMIN`` (``unoassp30.f:351-352``,
    :class:`~uacpy.models.oases._sampling._OasspWavenumberSampling`), so
    the slowest phase speed it reaches is ``2π·FREQM/WKMAX`` at FREQM and
    falls in proportion to the frequency below it. A Scholte wave of
    ``env`` (:func:`~uacpy.core.engine_defaults.scholte_waves`,
    Brekhovskikh & Lysanov §3.4.2, eq. 3.4.29) slower than that edge at
    FREQM is inside the window below ``v·WKMAX/2π`` and left out above it.
    """
    samplings = []
    for i in range(int(np.atleast_1d(source.depths).size)):
        deck = io.StringIO()
        _write_oassp_deck(deck, env, source.at_depth(i), receiver, engine,
                          warn=False)
        samplings.append(_deck_sampling(deck.getvalue(), receiver))
    needed = max(_oassp_wavenumber_count(**s) for s in samplings)
    if needed <= _OASSP_MAX_WAVENUMBERS:
        return None
    c_low_max = _compressional_c_low(env)
    fitted = [_oassp_fitting_c_low(c_low_max=c_low_max, **s)
              for s in samplings]
    if None in fitted:
        return None
    c_low = max(fitted)
    windows = [_oassp_wavenumber_sampling(**{**s, 'c_low': c_low})
               for s in samplings]
    count = max(w.count for w in windows)
    k_max = min(w.k_max for w in windows)
    freq_top = windows[0].freq_top
    edge = 2.0 * np.pi * freq_top / k_max
    documented = engine.c_low
    below = sorted(w.speed for w in scholte_waves(env) if w.speed < edge)
    waves = ("; ".join(
        f"the Scholte wave at {v:.1f} m/s is inside the window below "
        f"{v * k_max / (2.0 * np.pi):.1f} Hz and left out above it"
        for v in below) + " (Brekhovskikh & Lysanov §3.4.2, eq. 3.4.29)"
        if below else
        f"no Scholte wave of the environment is slower than that edge "
        f"(Brekhovskikh & Lysanov §3.4.2, eq. 3.4.29)")
    origin = (f"{c_low:g} m/s, the lowest value whose oassp2 wavenumber "
              f"count fits its {_OASSP_MAX_WAVENUMBERS}; the documented "
              f"{documented:g} m/s ({engine.c_low_origin}) needs {needed}")
    notice = Notice(
        f"c_low {c_low:g} m/s: the documented {documented:g} m/s needs "
        f"{needed} wavenumbers, past oassp2's {_OASSP_MAX_WAVENUMBERS}",
        f"{name}: the documented c_low = {documented:g} m/s makes oassp2 "
        f"integrate on {needed} wavenumbers, past the "
        f"{_OASSP_MAX_WAVENUMBERS} its roughness branch holds (it rounds the "
        f"count up to a power of two, unoassp30.f:542-544, and PV indexes "
        f"arrays of nnkr = 8192 up to twice that, oasvun31.f:2273, "
        f":2532-2533; comvol.f:3). The run takes c_low = {c_low:g} m/s, the "
        f"lowest value whose count ({count}) fits. oassp2 fixes the "
        f"window in wavenumber at the band's top frequency, WKMAX = "
        f"2π·FREQM/CMIN (unoassp30.f:351-352): at FREQM = {freq_top:.2f} Hz "
        f"it reaches down to {edge:.1f} m/s and leaves slower phase speeds "
        f"out, and at a frequency f below FREQM down to "
        f"{edge:.1f}·f/{freq_top:.2f} m/s; {waves}. "
        f"Pin n_wavenumbers={_OASSP_MAX_WAVENUMBERS} to keep "
        f"c_low = {documented:g} m/s on a coarser wavenumber step, or pin "
        f"c_low.", NumericsWarning)
    return c_low, origin, notice


def _require_single_rough_interface(env) -> None:
    """Refuse environments whose ``.rhs`` OASSP cannot consume, before a
    mean-field run is spent producing it.

    ``SCTRHS`` writes one ``.rhs`` record per rough interface per
    wavenumber, the sea surface's first (``oaseun31.f:2306-2310``,
    ``:2395``), and ``oassp2`` reads back exactly one record per
    wavenumber with **no interface filter**, taking its scattering
    interface from the first record (``unoassp30.f:549``,
    ``oasvun31.f:66-70``) — unlike OASS, which filters records by its
    deck's ``INTFC`` (``oassun26.f:360-362``). Measured on a rough
    surface + rough bottom: the binary announces ``Roughness scattering
    Layer: 2`` (the sea surface, whose water record carries no roughness
    spectrum), derives a zero wavenumber step from the interleaved
    records (``dkr_i= (0,0)``) and stops on ``>>> ERROR: Frequency
    mismatch in rhs file <<<``, leaving a truncated ``.trf``; two rough
    *bottom* interfaces fail identically. A mean field with no rough
    interface writes a ``.rhs`` with no SCTRHS records at all (skipped
    below ``ROUGH2 < 1e-10``, ``oaseun31.f:2310``), equally unusable.
    So exactly one interface may be rough, and it must be a bottom one.
    """
    # ``env.surface`` is always a ``Surface``: ``Environment.__init__``
    # puts whatever it is given through ``Surface.coerce``, which turns
    # ``None`` into a vacuum surface rather than passing it along. And
    # ``roughness`` delegates to the r=0 ``BoundaryProperties``, whose
    # ``__post_init__`` leaves it a concrete float on every boundary type
    # — so no default and no ``or 0.0``.
    surface_rg = abs(float(env.surface.roughness))
    rough_bottoms = [rg for rg in bottom_interface_roughness(env)
                     if abs(rg) > _OASS_MIN_MEAN_FIELD_ROUGHNESS]
    if surface_rg > _OASS_MIN_MEAN_FIELD_ROUGHNESS:
        raise UnsupportedFeatureError(
            'OASSP',
            f"a rough sea surface (env.surface.roughness = "
            f"{surface_rg:g} m) — OASSP scatters from the seabed: the "
            f"mean field writes the surface's boundary operators into "
            f"the .rhs first, and oassp2 takes its scattering interface "
            f"from that first record and reads one record per wavenumber "
            f"with no interface filter, so the run scatters from the "
            f"water record's empty roughness spectrum and dies on "
            f"interleaved records",
            ['OASS (its binary filters .rhs records by the deck '
             'interface, so it handles a rough surface alongside a '
             'rough bottom)',
             'a smooth surface (env.surface.roughness <= 1e-5 m) with '
             'the roughness on the seabed'],
            alternatives_label='configurations',
        )
    if not rough_bottoms:
        raise ConfigurationError(
            "OASSP: every bottom interface is smooth in the environment, "
            "so the mean-field run would write a .rhs with no boundary-"
            "operator records — SCTRHS skips any interface with "
            "ROUGH2 < 1e-10 (oaseun31.f:2310, :379) — and OASSP would "
            "have nothing to scatter from.",
            remediation=("Set the roughness on the environment, e.g. "
                         "BoundaryProperties(..., roughness=0.5). "
                         "OASSP(rms_roughness=…) overrides only the "
                         "OASSP deck, not the mean field's."),
        )
    if len(rough_bottoms) > 1:
        raise UnsupportedFeatureError(
            'OASSP',
            f"{len(rough_bottoms)} rough bottom interfaces — SCTRHS "
            f"writes one .rhs record per rough interface per wavenumber "
            f"and oassp2 reads back exactly one per wavenumber, so the "
            f"interleaved records give it a zero wavenumber step and "
            f"the run dies with a truncated .trf",
            ['OASS (its binary filters .rhs records by the deck '
             'interface)',
             'an environment with exactly one rough bottom interface'],
            alternatives_label='configurations',
        )


def _oassp_scattering_interface(env, interface) -> int:
    """The OASES layer index (OASP/OASSP numbering) whose roughness
    OASSP will scatter from, decided from the environment in stage 3.

    ``SCTRHS`` loops over interfaces innermost (``oaseun31.f:2306-2310``)
    and OASSP reads the *first* record's index (``unoassp30.f:548-549``);
    :func:`_require_single_rough_interface` has already guaranteed the mean
    field writes exactly one rough interface, a bottom one, so that first
    record names that interface (:func:`oassp_bottom_interfaces`). A
    pinned ``interface`` that is not it is refused, before the mean
    field runs.
    """
    first, roughness = oassp_bottom_interfaces(env)
    rough = [i for i, rg in enumerate(roughness)
             if abs(rg) > _OASS_MIN_MEAN_FIELD_ROUGHNESS]
    from_env = first + rough[0]
    if interface is not None and interface != from_env:
        raise ConfigurationError(
            f"OASSP(interface={interface}) disagrees with the "
            f"interface the mean-field .rhs names, {from_env}: the "
            f"environment's one rough bottom interface "
            f"(unoassp30.f:548-549 reads it from the file, not the deck). "
            f"Writing the roughness spectrum onto layer "
            f"{interface} would leave layer {from_env} with "
            f"CLEN = 0 (oaseun31.f:102) and the run would exit 0 with no "
            f"back-scatter.",
            remediation=(f"Drop interface=, or set it to {from_env}."),
        )
    return from_env


def _check_rhs_matches_settings(rhs_header: OasesRhsHeader,
                                engine) -> None:
    """Hold the mean field's ``.rhs`` header to the interface and Block
    VIII the settings resolved (and the deck carries): a difference is
    a resolution this wrapper got wrong, not a user's configuration."""
    got = (int(rhs_header.interface),
           (int(rhs_header.n_time_samples),
            float(rhs_header.freq_min), float(rhs_header.freq_max),
            float(rhs_header.time_step)))
    resolved = (engine.interface, tuple(engine.mean_field_grid))
    if got != resolved:
        raise ModelExecutionError(
            'OASSP', return_code=0, stdout=None, stderr=
            f"OASSP: the mean field's .rhs names interface {got[0]} and "
            f"(NX, FR1, FR2, DT) = {got[1]}, but the run resolved "
            f"interface {resolved[0]} and {resolved[1]} and wrote the "
            f"deck from them; oassp2 reads the .rhs "
            f"(unoassp30.f:181-188, :548-549).")


def _write_oassp_deck(target, env, source, receiver, engine, *,
                      c_high_given: bool = True, warn: bool = True) -> None:
    """Write the OASSP deck to ``target`` (a path or a text stream) with
    :func:`~uacpy.io.oases_writer.write_oassp_input`, from the settings
    ``engine`` alone: the one call behind the stage-3 wavenumber count
    and the deck the launch runs. ``warn=False`` writes it without the
    writer's warnings (the stage-3 count, whose deck the launch writes
    again). ``c_high_given=False`` (a ``c_high`` the caller did not set)
    hands the writer ``None``, which writes the same 1e8 the record
    carries, so the writer's notice of an unused ``c_high`` without
    option ``'P'`` is for a value the caller set."""
    n_time, freq_min, freq_max, time_step = engine.mean_field_grid
    write_oassp_input(
        filepath=target,
        env=env,
        source=source,
        receiver=receiver,
        options=engine.options,
        interface=engine.interface,
        correlation_length=engine.correlation_length,
        spectral_exponent=engine.spectral_exponent,
        rms_roughness=engine.rms_roughness,
        realization=engine.realization,
        n_time_samples=n_time,
        freq_min=freq_min,
        freq_max=freq_max,
        time_step=time_step,
        fc=engine.mean_field.engine.fc,
        integration_offset=engine.integration_offset,
        n_wavenumbers=engine.n_wavenumbers,
        integration_window=engine.integration_window,
        c_low=engine.c_low,
        c_high=engine.c_high if c_high_given else None,
        warn=warn,
    )


def _oassp_settings(name: str, mean_settings, projection_notices, *,
                    options, correlation_length, spectral_exponent,
                    rms_roughness, realization, interface, c_low_pinned,
                    c_high, n_wavenumbers, integration_window,
                    integration_offset, env, source,
                    receiver) -> OASSPSettings:
    """The OASSP deck's option line (``options`` resolved) and Block
    IV/VII values, over ``mean_settings``, the mean field's own
    settings (its BROADBAND run on the same environment and receiver,
    :meth:`OASSP._build_mean_field`), whose pinned wavenumber count
    oassp2 must be able to hold
    (:data:`_OASSP_MAX_MEAN_FIELD_WAVENUMBERS`). An unpinned ``c_low`` is
    the documented rule's, or the value :func:`_fitted_c_low` gives when
    that one's wavenumber count passes oassp2's bound. ``name`` is the
    model the roughness and ``c_low`` notices name."""
    mean_nw = mean_settings.engine.n_wavenumbers
    if mean_nw is not None and mean_nw > _OASSP_MAX_MEAN_FIELD_WAVENUMBERS:
        raise ConfigurationError(
            f"OASSP: the mean field pins n_wavenumbers={mean_nw}, and oassp2 "
            f"reads one .rhs record per mean-field wavenumber into arrays "
            f"of nnkr = {_OASSP_MAX_MEAN_FIELD_WAVENUMBERS} "
            f"(comvol.f:3): past that CALSRP/CALSVC stop with '>>> "
            f"NKMEAN too large' (oasvun31.f:61-65, :319-323).",
            remediation=(f"Pin the mean field's n_wavenumbers <= "
                         f"{_OASSP_MAX_MEAN_FIELD_WAVENUMBERS}."),
        )
    if c_low_pinned is not None:
        c_low, origin = c_low_pinned, 'OASSP(c_low=…)'
    else:
        c_low = oases_wavenumber_bounds(env)[0]
        origin = default_c_low(env)[1]
    options = _geometry_options(options, source)
    if c_high is None:
        c_high_origin = "1e8, OASES' no upper limit (oast.tex:487, :568)"
        c_high = oases_wavenumber_bounds(env)[1]
    else:
        c_high_origin = f'{name}(c_high=…)'
    if 'P' not in _option_letters(options):
        c_high_origin += ("; without 'P' OASSP replaces it with 1e12 "
                          "(unoassp30.f:205-217)")
    # 'g' is the Goff-Jordan spectrum; OASSP's 'G' is another option
    # (unoassp30.f:933, :1034).
    spectral_exponent, unused = resolve_spectral_exponent(
        name, spectral_exponent, goff='g' in options)
    interface = _oassp_scattering_interface(env, interface)
    mean = mean_settings.engine
    grid = _oasp_fft_grid(mean.n_time_samples, mean.freq_min,
                          mean.freq_max, mean.time_step)
    mean_field_grid = (grid.n_time_samples, grid.freq_min,
                       grid.freq_max, grid.time_step)
    window, window_notice = resolve_integration_window(
        name, integration_window, n_wavenumbers, 'oassp.tex:494-500')
    engine = OASSPSettings(
        options=options,
        interface=interface,
        mean_field_grid=mean_field_grid,
        n_integrated_wavenumbers=(),
        correlation_length=correlation_length,
        spectral_exponent=spectral_exponent,
        rms_roughness=rms_roughness,
        realization=realization,
        c_low=float(c_low),
        c_low_origin=origin,
        c_high=float(c_high),
        c_high_origin=c_high_origin,
        n_wavenumbers=n_wavenumbers,
        integration_window=window,
        integration_offset=integration_offset,
        mean_field=mean_settings,
        notices=(),
    )
    fitted = (None if c_low_pinned is not None
              else _fitted_c_low(name, env, source, receiver, engine))
    if fitted is not None:
        c_low, origin, _ = fitted
    notices = list(projection_notices)
    mean_c_low = mean_settings.engine.c_low
    if not np.isclose(float(c_low), mean_c_low, rtol=C_LOW_MATCH_RTOL):
        notices.append(
            message_notice(f"OASSP's CMIN {float(c_low):g} m/s differs from the mean "
                        f"field's {mean_c_low:g} m/s. oassp2 takes its incident "
                        f"spectrum from the mean field's .rhs records alone "
                        f"(oasvun31.f:313-332, :55-74), so the scattering integral "
                        f"holds no incident energy below the mean field's CMIN. Set "
                        f"the same c_low on both, or leave the mean field to OASSP.",
                        ValidityWarning))
    notices.append(message_notice(unused, FallbackWarning))
    # P grows with k, so the top of the band is the worst case. The band
    # OASSP computes is the mean field's FR1..FR2, which on the OASP
    # producer reaches past the source frequency.
    notices.append(_roughness_notice(
        name, env, interface,
        max(grid.freq_max, float(np.max(np.asarray(
            source.frequencies, dtype=float)))),
        rms_roughness, oass_index_space=False))
    if fitted is not None:
        notices.append(fitted[2])
    # Same read gate as OASP (unoassp30.f:135-142, OFFDB=OFFDBIN at :375).
    notices.append(_offset_ignored_notice(
        name, integration_offset, n_wavenumbers, options,
        offset_letters='Jd'))
    notices.append(window_notice)
    engine = dataclasses.replace(
        engine, c_low=float(c_low), c_low_origin=origin,
        notices=tuple(n for n in notices if n is not None))
    counts = []
    for i in range(int(np.atleast_1d(source.depths).size)):
        # One deck per source depth (the base loop launches each).
        deck = io.StringIO()
        # Written for its wavenumber count only; its warnings are the
        # launch's, said when the deck is written for real.
        _write_oassp_deck(deck, env, source.at_depth(i), receiver,
                          engine, warn=False)
        counts.append(_reject_roughness_wavenumber_overflow(
            deck.getvalue(), receiver))
    return dataclasses.replace(engine, n_integrated_wavenumbers=tuple(counts))


class OASSP(_ScatteringPostProcessor):
    """
    OASSP — OASES Scattered-Field Realization Model

    Generates one **realization** of the broadband field scattered from a
    rough interface, from the mean-field boundary operators an OASP run writes
    with option ``'s'``. OASSP is a post-processor, so ``run()`` drives the
    mean-field binary first and then ``oassp2``; the mean-field
    :class:`~uacpy.core.results.Field` is kept as
    ``components['mean_field']`` so the coherent field is not thrown away.

    Where :class:`OASP` returns the coherent field, OASSP returns the
    scattered one — same ``.trf`` format, same reader, same grid.

    Parameters
    ----------
    executable : Path, optional
        Path to the ``oassp2`` binary. Auto-detected if ``None``.
    correlation_length : float
        ``CL`` (m) of the interface roughness power spectrum. Required:
        ``oass.tex:182-183`` — the roughness statistics "must be specified.
        These are not adopted from OASR or OAST". A positive finite length,
        as on :class:`OASS`: zero, NaN and infinity are refused before the
        mean field runs, a negative one is volume scattering.
    spectral_exponent : float, optional
        ``M`` of the same spectrum. Must exceed 1.5. ``None`` (default) is
        2.5, OASES' own Goff-Jordan exponent, as on :class:`OASS`; the
        default Gaussian spectrum does not read it, and set there the run
        says so.
    roughness_spectrum : {'gaussian', 'goff-jordan'}, optional
        ``'goff-jordan'`` adds option ``'g'``. Default ``'gaussian'``,
        OASES' own (oass.tex:136-137).
        The ROUGHNESS power spectrum the scattering integral runs over.
    rms_roughness : float, optional
        ``|RG|`` (m) at the scattering interface. ``None`` takes the
        environment's own value for that interface.
    interface : int, optional
        Cross-check on the OASES layer index that scatters. OASSP reads it
        from the ``.rhs`` (``unoassp30.f:548-549``); a value that disagrees
        with the file raises rather than attaching the roughness spectrum to
        a layer the binary will not look at.
    realization : int, optional
        Realization index ``k``. The OASES seed is ``-123 - k``
        (``unoassp30.f:170``, ``:535``), so a given ``k`` is reproducible and
        different ``k`` are different draws. A whole number >= 0. Default 0,
        the value of both worked OASSP decks (oassp.tex:604, :664). In
        cylindrical geometry (a point Source) the same token also sets the
        azimuthal Fourier order the binary computes, ``mbf_0 = k/2``
        (``unoassp30.f:171``; its log prints ">>> Fourier order"), whose
        Bessel functions ``J_{m+k/2}`` it tabulates (``oascun31.f:550-552``):
        ``k`` = 0 and 1 are two draws of order 0, ``k >= 2`` a higher order.
        Measured on a 100 m Pekeris guide (rough seabed, 80-120 Hz, 200 m to
        1.8 km): ``k`` = 0 and 1 lie within 1 dB of their mean at every
        range, ``k`` = 2-7 sit 2-8 dB below it. A plane-geometry run (a line
        Source, option ``'P'``) has no Fourier order.
    scattered_only : bool, optional
        Option ``'s'`` — zero the source arrays so the ``.trf`` holds the
        scattered field alone (``unoassp30.f:628-635``). Default True, as
        both of the manual's worked OASSP decks (oassp.tex:592, :653); the
        program's own default is the total field ("By default the total
        field will be calculated", oassp.tex:231-233).
    mean_field : OASP, optional
        The producer run. ``None`` builds one from ``n_time_samples`` /
        ``freq_min`` / ``freq_max`` / ``fc`` and from
        ``c_low`` / ``c_high`` / ``n_wavenumbers`` / ``integration_offset``;
        passing one makes the first four exclusive with it, since the mean
        field owns the FFT grid. A supplied producer's option line must
        carry ``'s'`` (e.g. ``OASP(options='K J s')``), or it writes no
        ``.rhs`` and is refused. The mean field runs on the call's Source,
        so it has the scatterer's geometry (a raw ``'P'`` on a point Source
        is refused, as on :class:`OASP`); a pinned ``n_wavenumbers`` above 8192
        (oassp2's ``nnkr``) is refused, and a ``CMIN`` that differs from
        OASSP's is warned about. A supplied model keeps its own ``verbose``
        and ``timeout``; its ``cleanup`` is OASSP's, whose directory holds
        the ``.rhs`` oassp2 reads.
    n_time_samples, freq_min, freq_max, fc : optional
        Passed to the mean-field :class:`OASP`. OASSP's own Block VIII is then
        read back out of the ``.rhs``, never guessed — see Notes. A pinned
        ``freq_min`` / ``freq_max`` beside ``run(frequencies=)`` is refused,
        naming both; beside the band a TIME_SERIES waveform derives, the run
        says it is not used.
    options : str, optional
        Raw OASSP option string, exclusive with the typed flags above.
        ``None`` derives the letters.
    integration_window : (int, int), optional
        As for :class:`OASP`, on the OASSP deck alone ("insignificant for
        automatic sampling", oassp.tex:494-500); a mean field takes its own
        through ``mean_field=OASP(integration_window=…)``.
    integration_offset, n_wavenumbers, c_low, c_high : optional
        As for :class:`OASP`, on both decks of the chain: oassp2 takes its
        incident spectrum from the mean field's ``.rhs`` alone
        (``oasvun31.f:313-332``, ``:55-74``), so a ``c_low`` that admits
        slow seabed branches must reach the mean field too. ``c_high`` is
        inert in cylindrical geometry (a point Source). The receiver ranges
        are ``receiver.ranges``. An unpinned ``c_low`` whose derived value
        makes oassp2 integrate on more than its 4096 wavenumbers (a shear
        seabed's, set by its shear speed or its Scholte wave) is the
        lowest whole m/s whose count fits, at most 0.90 × the slowest
        compressional speed, on both decks,
        with a ``NumericsWarning`` (:func:`_fitted_c_low`); a pinned
        ``c_low`` or ``n_wavenumbers`` is written as given, and refused past
        the bound. A pinned ``n_wavenumbers`` gets an info-level note
        (``verbose='info'``) that the manual leaves it to the convergence
        test (oassp.tex:479-488, :511-515).
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Supports ``RunMode.BROADBAND`` (complex ``H(f)``, the default) and
    ``RunMode.TIME_SERIES``.

    **The Source decides the geometry.** ``Source(source_type='line')`` runs
    both decks of the chain in plane geometry (option ``'P'``, a line
    source, ``oassp.tex:166-168``), as the manual's example does
    (``:559-600``); a point Source runs both cylindrical, where OASSP forces
    ``CMAX = 1e12`` and a full Hankel transform (``unoassp30.f:205-217``).

    **Exactly one interface may be rough, and it must be a bottom one.**
    ``oassp2`` reads the mean field's ``.rhs`` back one record per
    wavenumber with no interface filter and takes its scattering interface
    from the first record (``oasvun31.f:66-70``, ``unoassp30.f:549``), so a
    rough sea surface — whose records SCTRHS writes first — or a second
    rough bottom interface hands it interleaved records it cannot use.
    ``run()`` raises up front for either; :class:`OASS`, whose binary
    filters records by the deck interface, handles the multi-interface
    cases.

    **Block VIII is not the user's to set.** OASSP replaces its deck's
    ``NT``/``FR1``/``FR2``/``DT`` with the ``.rhs``'s own values and warns on
    only two of the four (``unoassp30.f:181-188``), so a band that differs
    from the mean field's is substituted with no message at all. This wrapper
    reads the four straight out of the ``.rhs``
    (:func:`~uacpy.io.oases_reader.read_oases_rhs_header`) and writes those,
    which makes the substitution a no-op. Ask for a different band by
    configuring the *mean field*, or by passing ``run(frequencies=…)``, which
    is forwarded to it.

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Per-model: ``'ssp': 'mean'``.

    Examples
    --------
    >>> from uacpy.models import OASSP
    >>> oassp = OASSP(correlation_length=5.0, n_time_samples=512,
    ...               freq_min=400, freq_max=600)
    >>> h_scattered = oassp.run(env, source, receiver)
    """

    # Declarative metadata (see PropagationModel / ModelSpec). OASSP:
    # range-independent scattered-field realizations over the same layered
    # stack OASP solves the mean field on — INENVI is shared verbatim between
    # the binaries, so the supported axes are OASP's. A line Source runs
    # both decks in plane geometry, option 'P' (unoassp30.f:959-961).
    spec = ModelSpec(
        modes=(RunMode.BROADBAND, RunMode.TIME_SERIES),
        # The layered stack OASP solves the mean field on (INENVI is shared
        # verbatim between the binaries), so the capabilities are OASP's;
        # a rough interface is column 7 (RG) of its layer record
        # (oaseun31.f:54).
        supports={'layered_bottom', 'elastic_media',
                  'rough_surface', 'rough_bottom'},
        # Both decks run in plane geometry for a line Source, option 'P'
        # (unoassp30.f:959-961).
        source_types=frozenset({'point', 'line'}),
        collapse={'ssp': 'mean'},
        traits=dataclasses.replace(
            _OASES_TRAITS,
            consumes_run_t_start=True,
            # BROADBAND runs the mean field's FFT ladder (see :class:`OASP`).
            announced_band_modes=frozenset({RunMode.TIME_SERIES}),
        ),
    )
    provenance_id = 'oases'
    _BINARIES = MappingProxyType({'oassp': _oases_binary('oassp2')})
    outputs = MappingProxyType({
        RunMode.BROADBAND: OutputSpec(
            'Field', kind='pressure', unit='Pa',
            phase_reference=PhaseReference.TRAVELLING_WAVE.value,
            coherent=True),
        RunMode.TIME_SERIES: OutputSpec(
            'Field', kind='pressure', unit='Pa',
            phase_reference=PhaseReference.TIME_DOMAIN_NATIVE.value),
    })

    #: The mean field's FFT-grid knobs, exclusive with ``mean_field=``.
    #: Named so the mutual-exclusion message can list them.
    _MEAN_FIELD_KNOBS = ('n_time_samples', 'freq_min', 'freq_max', 'fc')

    _UNPINNED_FIELDS = SCATTERING_UNPINNED_FIELDS

    def __init__(
        self,
        *,
        executable: Optional[Path] = None,
        correlation_length: Optional[float] = None,
        spectral_exponent: Optional[float] = None,
        roughness_spectrum: str = 'gaussian',
        rms_roughness: Optional[float] = None,
        interface: Optional[int] = None,
        realization: int = OASSP_REALIZATION,
        scattered_only: bool = True,
        mean_field: Optional['OASP'] = None,
        n_time_samples: Optional[int] = None,
        freq_min: Optional[float] = None,
        freq_max: Optional[float] = None,
        fc: Optional[float] = None,
        options: Optional[str] = None,
        integration_offset: float = OASSP_INTEGRATION_OFFSET,
        n_wavenumbers: Optional[int] = None,
        integration_window: Optional[Tuple[int, int]] = None,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: Optional[float] = DEFAULT_RUN_TIMEOUT,
        collapse: Optional[Dict[str, str]] = None,
    ):
        super().__init__(**_host_options(locals()))
        self.correlation_length = (
            float(correlation_length) if correlation_length is not None
            else None
        )
        self.spectral_exponent = (float(spectral_exponent)
                                  if spectral_exponent is not None else None)
        # Kept as passed until the check, whose refusal quotes it; then
        # normalised, so roughness_spectrum='Gaussian' behaves as the
        # default rather than as a pinned non-default. Every comparison
        # (the check, the option letter, the options-exclusivity check)
        # folds the case. OASS does the same.
        self.roughness_spectrum = roughness_spectrum
        self.rms_roughness = (
            float(rms_roughness) if rms_roughness is not None else None
        )
        # The interface is a layer index and the realization the INTEGER
        # fourth token of Block VII (unoassp30.f:169, ISEED = -123 - k at
        # :535); n_time_samples is the mean field's NX (unoasp22.f:176),
        # a power of two (oasp.tex:129).
        if interface is not None:
            whole_count('OASSP: interface', interface, 1)
        whole_count('OASSP: realization', realization, 0)
        if n_time_samples is not None:
            check_power_of_two_time_samples('OASSP: n_time_samples',
                                            n_time_samples)
        self.interface = int(interface) if interface is not None else None
        self.realization = int(realization)
        self.scattered_only = bool(scattered_only)
        # OASSP takes no multiple_scattering (OASS does): the vendor disabled
        # re-scattering in OASSP's integration path on 980402 — every routine
        # unoassp30.f:677-688 dispatches to (oasvun31.f:76-80, :336-340,
        # :962-966, :1367-1371) has the `if (.not.rescat)` guard commented
        # out with ROUGH2(II)=0 left live, so option 'p' would return a
        # single-scatter answer under a multiple-scattering label.
        self.mean_field = mean_field
        self.n_time_samples = (
            int(n_time_samples) if n_time_samples is not None else None
        )
        self.freq_min = float(freq_min) if freq_min is not None else None
        self.freq_max = float(freq_max) if freq_max is not None else None
        self.fc = float(fc) if fc is not None else None
        self.options = options
        self.integration_offset = float(integration_offset)
        self.n_wavenumbers = n_wavenumbers
        self.integration_window = integration_window
        self.c_low = float(c_low) if c_low is not None else None
        self.c_high = float(c_high) if c_high is not None else None
        self._check_knobs()
        if self.n_wavenumbers is not None:
            self._log(_pinned_count_note('OASSP', self.n_wavenumbers))
        self.roughness_spectrum = str(roughness_spectrum).lower()

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        # install.sh's ``POSSIBLE_OASES_BINS`` loop copies the raw executables;
        # its compatibility symlinks cover oasn/oasp/oasr/oast but not this
        # one, so the name is 'oassp2', not 'oassp'.
        self._exe = self._resolve_executable(
            executable, lambda: self._find_binary('oassp'),
        )

    def _build_mean_field(self, band_from_call: bool = False,
                          c_low: Optional[float] = None) -> 'OASP':
        """The OASP producer: the supplied one (whose ``'s'`` the
        constructor checks, :func:`_check_oassp_knobs`), or one built with
        :data:`_OASSP_MEAN_FIELD_OPTIONS` — without ``'s'`` OASP never calls
        ``opfilb(45)`` (``unoasp22.f:251-254``) and no ``.rhs`` is written.

        The default producer describes the problem the OASSP deck does, as
        the manual requires ("The input files for OASSP are virtually
        identical to the ones used for computing the mean field using OASP",
        ``oassp.tex:36-37``; its worked example ``:559-600`` carries ``'P'``
        and the same ``CMIN``/``CMAX`` on both decks): the FFT grid knobs,
        ``c_low``/``c_high``/``n_wavenumbers``/``integration_offset``, and the
        geometry — the producer runs on the call's Source, so a line Source
        puts ``'P'`` on both decks (:func:`_mean_field_source`). oassp2
        takes its incident spectrum solely from the producer's ``.rhs``
        records (``oasvun31.f:313-332``, ``:55-74``), so a knob left off the
        producer never reaches the scattering integral. A supplied producer
        keeps its own ``verbose`` and ``timeout``; the default one takes
        OASSP's. Its ``cleanup`` is OASSP's: its files share OASSP's
        directory, which OASSP's ``cleanup`` keeps or wipes, so the mean
        field's result records their paths exactly when they survive.
        ``c_low`` replaces OASSP's own on the default producer: the value
        :func:`_fitted_c_low` gives an unpinned one.
        """
        mean = self.mean_field
        if mean is None:
            # A band the call carries (``band_from_call``: run(frequencies=)
            # or the source waveform's) sets the sweep, so the pinned edges
            # do not reach the producer (stage 3 says so).
            skipped = {'freq_min', 'freq_max'} if band_from_call else set()
            kwargs = {k: getattr(self, k) for k in self._MEAN_FIELD_KNOBS
                      if getattr(self, k) is not None and k not in skipped}
            mean = OASP(options=_OASSP_MEAN_FIELD_OPTIONS,
                        n_wavenumbers=self.n_wavenumbers,
                        c_low=self.c_low if c_low is None else c_low,
                        c_high=self.c_high,
                        verbose=self.verbose, timeout=self.timeout,
                        cleanup=self.cleanup, **kwargs)
            # OASSP's own settings say whether the offset reaches the
            # binaries (the same read gate), so the producer does not.
            mean.integration_offset = self.integration_offset
            mean._offset_said_by_owner = True
            mean._scholte_said_by_owner = True
            return mean
        return mean.copy(cleanup=self.cleanup)

    def _check_knobs(self) -> None:
        """Refuse a constructor knob no run could use
        (:func:`_check_oassp_knobs`, and a pinned phase-speed window that is not
        positive or not ordered,
        :func:`~uacpy.models._window.check_pinned_window`). Run at
        construction and again by every
        run (:meth:`_validate_engine`), since the attributes can be
        reassigned in between."""
        _check_n_wavenumbers_knob(self.n_wavenumbers)
        check_pinned_window('OASSP', c_low=self.c_low, c_high=self.c_high)
        refuse_unusable_integration_window('OASSP', self.integration_window,
                                           self.n_wavenumbers)
        _check_oassp_knobs(
            correlation_length=self.correlation_length,
            spectral_exponent=self.spectral_exponent,
            roughness_spectrum=self.roughness_spectrum,
            mean_field=self.mean_field,
            mean_field_knobs={n: getattr(self, n)
                              for n in self._MEAN_FIELD_KNOBS},
            options=self.options, scattered_only=self.scattered_only)

    def _validate_engine(self, env, source, receiver, run_mode, *,
                         request=None) -> None:
        """Stage 2: the family's boundary refusals
        (:meth:`OASES._validate_engine`), then an option line whose ``.trf``
        uacpy cannot read back
        (:func:`~uacpy.models.oases._common._reject_unreadable_trf_options`),
        a raw ``'P'`` with a point
        Source (the Source decides the geometry), and an environment whose
        ``.rhs`` oassp2 cannot consume
        (:func:`_require_single_rough_interface`) — all before a mean-field
        run is spent (stage 3 refuses a pinned ``interface=`` that is not its
        rough one, :func:`_oassp_scattering_interface`). A supplied mean field's own raw ``'P'`` on a point Source is
        refused by its own checks, in stage 3."""
        super()._validate_engine(env, source, receiver, run_mode,
                                 request=request)
        _reject_unreadable_trf_options('OASSP', self.options,
                                       self.n_wavenumbers)
        _reject_plane_geometry_letter_on_a_point_source(
            'OASSP', _oassp_options(self.options, self.scattered_only,
                                    self.roughness_spectrum),
            self.options, source,
            'unoassp30.f:959-961')
        _require_single_rough_interface(env)

    def _requested_frequencies(self, mode, source, frequencies, time):
        """The frequencies (Hz) the call asks the mean field for — which
        owns the FFT grid — or ``None`` for its default sweep:
        ``run(frequencies=)``, or for TIME_SERIES the grid the pulse
        implies. :meth:`_marched_frequencies` then records the bins the mean
        field's sweep gives, which are the scattered field's."""
        if frequencies is not None:
            return BandResolution(
                np.atleast_1d(np.asarray(frequencies, dtype=float)))
        if mode == RunMode.TIME_SERIES:
            return super()._requested_frequencies(mode, source, None, time)
        return BandResolution(None)

    def _marched_frequencies(self, settings):
        """The bins the mean field propagates
        (``settings.engine.mean_field.frequencies``), which oassp2 reads back
        from its ``.rhs``, not the base's BROADBAND band
        (``spec.traits.announced_band_modes``)."""
        return settings.engine.mean_field.frequencies

    def _resolve_engine_settings(self, env, source, receiver, settings, *,
                                 given_env, request=None) -> 'OASSPSettings':
        """Stage 3: the mean field's settings, from its producer, and
        :func:`_oassp_settings` over them for this model's knobs. A
        ``c_low`` fitted to oassp2's wavenumber bound (:func:`_fitted_c_low`)
        is handed to the producer OASSP builds, whose settings are resolved
        again on it, its origin stating the fit."""
        band_from_call = settings.frequencies is not None
        own_notices = []
        pinned = [f"{name}={getattr(self, name):g}"
                  for name in ('freq_min', 'freq_max')
                  if getattr(self, name) is not None]
        if band_from_call and pinned:
            if (request is not None
                    and request.kwargs.get('frequencies') is not None):
                raise ConfigurationError(
                    f"OASSP({', '.join(pinned)}) pins the mean field's "
                    f"sweep and run(frequencies=) names the band too; the "
                    f"mean field runs one sweep (FR1 FR2, unoasp22.f:176), "
                    f"so the two contradict each other.",
                    remediation=(
                        "Leave OASSP(freq_min=, freq_max=) unset to sweep "
                        "the call's band, or drop frequencies= to sweep the "
                        "pinned edges."))
            own_notices.append(message_notice(
                f"OASSP({', '.join(pinned)}) is not used: the mean field "
                f"sweeps the band this {settings.mode.name} run derives "
                f"from its source waveform.", FallbackWarning))
        def resolve(c_low=None):
            mean_settings, projection_notices = (
                self._build_mean_field(band_from_call, c_low)
                ._producer_settings(
                    env, _mean_field_source(source), receiver,
                    RunMode.BROADBAND, frequencies=settings.frequencies))
            return mean_settings, _oassp_settings(
                type(self).__name__, mean_settings,
                list(projection_notices) + own_notices,
                options=_oassp_options(self.options, self.scattered_only,
                                       self.roughness_spectrum),
                correlation_length=self.correlation_length,
                spectral_exponent=self.spectral_exponent,
                rms_roughness=self.rms_roughness,
                realization=self.realization, interface=self.interface,
                c_low_pinned=self.c_low, c_high=self.c_high,
                n_wavenumbers=self.n_wavenumbers,
                integration_window=self.integration_window,
                integration_offset=self.integration_offset,
                env=env, source=source, receiver=receiver)

        mean_settings, engine = resolve()
        if (self.mean_field is None and self.c_low is None
                and engine.c_low != mean_settings.engine.c_low):
            # A fitted c_low (_fitted_c_low) reaches the mean field OASSP
            # builds, as the manual's worked example carries one CMIN on
            # both decks (oassp.tex:559-600); the mean field's FFT grid,
            # which the count reads, does not depend on it.
            _, engine = resolve(engine.c_low)
            mean = engine.mean_field
            engine = dataclasses.replace(engine, mean_field=dataclasses.replace(
                mean, engine=dataclasses.replace(
                    mean.engine, c_low_origin=engine.c_low_origin)))
        return engine

    #: Why a missing .rhs means what it means, for the refusal.
    _RHS_HINT = ("OPFILB opens unit 45 with STATUS='UNKNOWN' "
                 "(oashun21.f:634), so an empty file here means the "
                 "producer wrote no scattering right-hand sides.")

    def _producer_for(self, settings) -> 'OASP':
        """The mean-field producer for a call with ``settings``: its band
        set by the call when the call carries one
        (:meth:`_build_mean_field`)."""
        return self._build_mean_field(settings.frequencies is not None)

    def _mean_field_source(self, source):
        """The Source the producer runs on (:func:`_mean_field_source`)."""
        return _mean_field_source(source)

    # ── stage 4: oassp2 on the mean field's .rhs/.vol ──────────────────

    def _write_scattering_deck(self, inputs) -> Path:
        """Stage 4, launch 1: the OASSP deck, written from the settings
        alone (:func:`_write_oassp_deck`), once the mean field's ``.rhs`` is
        held to them (:func:`_check_rhs_matches_settings`): Block VIII (NT,
        FR1, FR2, DT) and the scattering interface are the file's, which
        ``unoassp30.f:181-188`` substitutes anyway, and Block III's FRC is
        the carrier the mean field ran with — TRFHEAD writes it into the
        ``.trf`` header (``oasiun23.f:815``, ``:864-865``, ``:919``)."""
        engine = inputs.settings.engine
        _check_rhs_matches_settings(inputs.earlier[0].rhs_header, engine)
        deck = inputs.work_dir / f'{_OASSP_BASE_NAME}.dat'
        self._log(f"Writing OASSP input file: {deck} "
                  f"(options={engine.options}, "
                  f"realization={engine.realization}, "
                  f"interface={engine.interface})")
        _write_oassp_deck(deck, inputs.env, inputs.source, inputs.receiver,
                          engine, c_high_given=self._caller_set('c_high'))
        return deck

    def _launch_scattering(self, inputs, deck: Path) -> None:
        """Stage 4, launch 1: ``oassp2`` on the deck, its FOR045/FOR046
        pointed at the mean field's kernel files (named after that run's
        stem, see :meth:`OASES._execute`); refuse a run past OASES'
        wavenumber bound or with no transfer function."""
        stem = inputs.earlier[0].stem
        proc = self._execute(deck.stem, inputs.work_dir, extra_env={
            'FOR045': f'{stem}.045', 'FOR046': f'{stem}.046',
        })
        # unoassp30.f has no NWVNO.GT.NP test either (see
        # _reject_wavenumber_overrun), and its reference range
        # min(cref·NX·DT + 2·RM, 6·RM) (:295-306: RANREF + RM at :304, then
        # min(RANREF + RM, 6·RM) at :305) can push NW ~2× past the mean-field
        # OASP's, so the
        # producer passing is no proxy for this run being in bounds.
        _reject_wavenumber_overrun(
            self.model_name, proc,
            _oassp_wavenumber_echo_scale(inputs.settings.engine.options))
        require_output(
            self.model_name, [inputs.work_dir / f'{deck.stem}.trf'],
            what='a transfer-function file', process=proc,
        )

    def _read_scattering_output(self, inputs, deck: Path):
        """Stage 4, launch 1: the scattered transfer function on the
        requested axes (:func:`_read_trf_on_requested_ranges`)."""
        output_file = inputs.work_dir / f'{deck.stem}.trf'
        self._log(f"Reading OASSP output: {output_file}")
        return _read_trf_on_requested_ranges(output_file, inputs.receiver)

    def _require_mean_field_files(self, inputs, deck: Path) -> None:
        """The ``.vol`` (FOR046) oassp2 opens beside the ``.rhs``."""
        require_output(
            self.model_name, [inputs.work_dir / f'{deck.stem}.046'],
            what="a mean-field .vol (FOR046)",
            hint=("OASSP opens unit 46 unconditionally (unoassp30.f:128) "
                  "and its IOER test only covers the last OPFILb (:130)."),
        )

    def _to_result(self, inputs, deck, raw) -> Field:
        """Stage 5: the scattered complex ``H(f)`` on ``(depth, range,
        frequency)`` for BROADBAND, its synthesis for TIME_SERIES, with the
        mean field's result as ``components['mean_field']``.

        Convention: (n_depth, n_range, n_frequencies) — trailing axis is the
        variable dim. Converted like OASP's
        (:func:`~uacpy.io._parsers.oasp_trf_pressure`: the default payload
        is the bulk pressure 'K', read as written, oasp.tex:178-183; a raw
        'N' line's normal stress is negated, oasp.tex:185): the raw payload
        comes from the same TRFHEAD writer with the same sign (measured with
        scattered_only=False on a near-smooth seabed, the raw total field
        over OASP's raw mean field is +1: |ratio| 0.995, arg 0.35 deg), so
        the same conversion is what makes 'travelling_wave' mean the
        family's convention here too.
        """
        mean, trf_data = raw
        settings = inputs.settings
        source = inputs.source
        rhs = mean.rhs_header
        rhs_path = inputs.work_dir / f'{mean.stem}.045'
        vol_path = inputs.work_dir / f'{mean.stem}.046'
        result = Field(
            data=oasp_trf_pressure(trf_data, 'OASSP',
                                   inputs.work_dir / f'{deck[1].stem}.trf'),
            coords={
                'depth': trf_data['depths'],
                'range': trf_data['ranges'],
                'frequency': trf_data['freq'],
            },
            # The same complex H(f) OASP returns, from the same TRFHEAD
            # writer (oasiun23.f:842-845).
            # NT from the mean-field .rhs, which OASSP runs with in place of
            # the deck's (unoassp30.f:181-188): the synthesis floors its FFT
            # length at it.
            synthesis_floor=rhs.n_time_samples,
            **result_kwargs(
                self.model_name, self.provenance, source,
                phase_reference=PhaseReference.TRAVELLING_WAVE.value,
                backend='oassp',
                frequencies=trf_data['freq'],
                source_depth=trf_data['source_depth'],
                fc=trf_data['fc'],
                freq_max=rhs.freq_max,
                components={'mean_field': mean.result},
            ),
        )
        result = mask_source_axis(self.model_name, result, source)
        if settings.time is not None:
            result = finish_broadband(self.model_name, result, settings)
        attach_output_paths(
            result, inputs.work_dir, deck[1].stem,
            primary_files=(('trf_file', '.trf'),),
            cleanup=self.cleanup,
        )
        if not self.cleanup:
            result.metadata['rhs_file'] = str(rhs_path)
            result.metadata['vol_file'] = str(vol_path)

        self._log("OASSP simulation complete")
        return result

    # Mirrors third_party/oases/bin/oassp (oassp.tex:524-536). The .trf is
    # named from the deck stem by TRFHEAD (oasiun23.f:842-845), not by an
    # environment variable, so it needs no unit here; FOR045/FOR046 are set
    # per-run in _execute because they name the mean-field deck's stem.
    # Units 68-71 and 85 are unconditional ASCII dumps of the realized
    # perturbation field (oasvun31.f:421-448, :724), and dum.dum is the
    # scratch file itran() shells out for under option 'r' (unoassp30.f:1190-1209) —
    # all listed so a pinned work_dir starts each run clean. The bare
    # fort.46 is absent because oassp2 neither writes nor reads it: units
    # 45/46 are assigned to the mean-field stem's .045/.046 above, which are
    # different files, and nothing here opens the default name.
    _FOR_FILES: dict = {}
    _OUTPUT_SUFFIXES = ('.trf', '.plt', '.plp')
    _OUTPUT_FORT_FILES = ('fort.68', 'fort.69', 'fort.70', 'fort.71',
                          'fort.85', 'dum.dum')
