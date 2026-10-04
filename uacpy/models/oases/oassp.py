"""OASSP: the OASES rough-interface scattered-field pulse program."""

import io
import dataclasses
from dataclasses import dataclass
from types import MappingProxyType
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from uacpy.models._band import BandResolution
from uacpy.models._spec import ModelSpec
from uacpy.core.run_settings import (
    EngineSettings, OutputSpec, RunMode, RunSettings,
)
from uacpy.core.source import Source
from uacpy.core.results import Field, PhaseReference
from uacpy.core.exceptions import (
    ConfigurationError, ModelExecutionError, UnsupportedFeatureError,
    ValidityWarning,
)
from uacpy.io.oases_writer import (
    write_oassp_input, oases_wavenumber_bounds, bottom_interface_roughness,
    oassp_bottom_interfaces,
)
from uacpy.io.oases_reader import OasesRhsHeader, read_oases_rhs_header
from uacpy.models.base import DEFAULT_RUN_TIMEOUT_S
from uacpy.models.oases._common import (
    C_LOW_MATCH_RTOL,
    _check_n_wavenumbers_knob, _OASES_TRAITS, _OASS_MIN_MEAN_FIELD_ROUGHNESS, _OASS_SPECTRA,
    _geometry_options, _oases_find_executable, _option_letters,
    _reject_plane_geometry_letter_on_a_point_source,
    _reject_unintegrable_spectral_exponent, _reject_wavenumber_overrun,
    _roughness_notice, _warn_offset_ignored_under_auto_sampling,
)
from uacpy.models.oases._sampling import (
    _OASSP_MAX_WAVENUMBERS, _oasp_fft_grid, _oassp_wavenumber_count,
)
from uacpy.models.oases._base import OASES
from uacpy.models.oases.oasp import OASP, _read_trf_on_requested_ranges
from uacpy.models.oases._mean_field import _MeanFieldRun, _mean_field_inputs
from uacpy.models._notices import message_notice
from uacpy.core.engine_defaults import OASSP_INTEGRATION_OFFSET, OASSP_REALIZATION, OASSP_SPECTRAL_EXPONENT


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
        Block VII's ``CMIN`` (m/s).
    c_low_origin : str
        Where ``c_low`` came from.
    c_high : float or None
        Block VII's ``CMAX`` (m/s); ``None`` writes 1e9, which cylindrical
        geometry replaces with 1e12 (``unoassp30.f:205-217``).
    n_wavenumbers : int or None
        Block VII's ``NW``; ``None`` is OASES' automatic sampling.
    integration_offset : float
        The contour offset (dB/wavelength) on the frequency line.
    mean_field : RunSettings
        The settings of the mean-field OASP run whose ``.rhs`` the deck
        reads.
    notices : tuple of Notice
        What the run says about the chain it resolved (a CMIN that differs
        from the mean field's).
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
    c_high: Optional[float]
    n_wavenumbers: Optional[int]
    integration_offset: float
    mean_field: RunSettings

    def __post_init__(self):
        # The to_dict form holds a list; the frozen record a tuple.
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
        return cls(**d)


def _check_oassp_knobs(*, correlation_length,
                       spectral_exponent, roughness_spectrum,
                       mean_field, mean_field_knobs, options,
                       scattered_only) -> None:
    """Refuse an OASSP knob no run could use: a missing, negative (volume) or zero
    correlation length, an unintegrable spectral exponent, an unknown
    roughness spectrum, FFT knobs beside the ``mean_field`` that owns
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
    if correlation_length < 0.0:
        raise UnsupportedFeatureError(
            'OASSP',
            "volume scattering (correlation_length < 0 is OASES' switch "
            "to the twelve-token layer record whose extra fields — SKW, "
            "M, RMS, GAM — have no home on any uacpy carrier; "
            "oaseun31.f:76-90)",
            alternatives=["interface roughness scattering "
                          "(correlation_length > 0)"],
            alternatives_label='configurations',
        )
    if correlation_length == 0.0:
        raise ConfigurationError(
            "OASSP(correlation_length=0) gives a degenerate roughness "
            "power spectrum: OASSP forms r_l = |CLEN(INTFCE)| and hands "
            "it to PV (unoassp30.f:606-614).",
            remediation="Pass correlation_length > 0 (metres).",
        )
    _reject_unintegrable_spectral_exponent('OASSP', spectral_exponent)
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
    letters = ['N', 'J']
    if scattered_only:
        letters.append('s')
    if str(roughness_spectrum).lower() == 'goff-jordan':
        letters.append('g')
    return ' '.join(letters)


def _reject_unreadable_oassp_options(options, n_wavenumbers) -> None:
    """Reject raw option letters whose ``.trf`` uacpy cannot read back.

    unoassp30.f carries the same complex-frequency-contour trap as OASP:
    ``:285-290`` sets CFRFL under automatic wavenumber sampling unless
    ``'J'`` keeps ICNTIN = 1, ``:983`` sets it for ``'O'``, and
    ``:382-385`` then bakes OMEGIM = -ln(50)·Δf into the spectrum, which
    the time-series synthesis does not undo — the last output sample
    comes back 50× (34 dB) too small. Only a caller-supplied
    ``options`` string can reach this: :func:`_oassp_options`' typed
    path always carries ``'J'`` and never ``'O'``.
    """
    if options is None:
        return
    opt_chars = set(str(options)) - set(' \t\n')
    if 'O' in opt_chars:
        raise ConfigurationError(
            "OASSP.run: option 'O' (complex frequency integration) bakes "
            "an exp(-ln(50)·Δf·t) contour into the spectrum "
            "(unoassp30.f:983, :382-385) that uacpy's time-series "
            "synthesis does not undo. Drop 'O'."
        )
    if 'J' not in opt_chars and n_wavenumbers is None:
        raise ConfigurationError(
            "OASSP.run: a custom options string without 'J' enables the "
            "complex frequency contour (OMEGIM≠0) under automatic "
            "wavenumber sampling (unoassp30.f:285-290, :382-385), which "
            "uacpy's time-series synthesis cannot undo. Add 'J' or pin "
            "n_wavenumbers >= 1."
        )


def _mean_field_source(source: Source) -> Source:
    """The Source the producer runs: one unit-weight depth of the call's
    (the weight belongs to the run the user called), its geometry
    included, so a line Source runs the mean field in plane geometry
    too."""
    return source.at_depth(0)


def _deck_wavenumber_count(deck_text: str, receiver) -> int:
    """NWVNO an OASSP deck makes oassp2 integrate on.

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
    return _oassp_wavenumber_count(
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
                      warn: bool = True) -> None:
    """Write the OASSP deck to ``target`` (a path or a text stream) with
    :func:`~uacpy.io.oases_writer.write_oassp_input`, from the settings
    ``engine`` alone: the one call behind the stage-3 wavenumber count
    and the deck the launch runs. ``warn=False`` writes it without the
    writer's warnings (the stage-3 count, whose deck the launch writes
    again)."""
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
        center_frequency=engine.mean_field.engine.center_frequency,
        integration_offset=engine.integration_offset,
        n_wavenumbers=engine.n_wavenumbers,
        c_low=engine.c_low,
        c_high=engine.c_high,
        warn=warn,
    )


def _oassp_settings(name: str, mean_settings, projection_notices, *,
                    options, correlation_length, spectral_exponent,
                    rms_roughness, realization, interface, c_low_pinned,
                    c_high, n_wavenumbers, integration_offset, env,
                    source, receiver) -> OASSPSettings:
    """The OASSP deck's option line (``options`` resolved) and Block
    IV/VII values, over ``mean_settings``, the mean field's own
    settings (its BROADBAND run on the same environment and receiver,
    :meth:`OASSP._build_mean_field`), whose pinned wavenumber count
    oassp2 must be able to hold
    (:data:`_OASSP_MAX_MEAN_FIELD_WAVENUMBERS`). ``name`` is the model
    the roughness notice names."""
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
        ssp_data = env.ssp.extend_to(env.depth).to_pairs()
        c_low = oases_wavenumber_bounds(ssp_data)[0]
        origin = "oases_wavenumber_bounds(water column)"
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
    options = _geometry_options(options, source)
    interface = _oassp_scattering_interface(env, interface)
    mean = mean_settings.engine
    grid = _oasp_fft_grid(mean.n_time_samples, mean.freq_min,
                          mean.freq_max, mean.time_step)
    mean_field_grid = (grid.n_time_samples, grid.freq_min,
                       grid.freq_max, grid.time_step)
    # P grows with k, so the top of the band is the worst case. The band
    # OASSP computes is the mean field's FR1..FR2, which on the OASP
    # producer reaches past the source frequency.
    notices.append(_roughness_notice(
        name, env, interface,
        max(grid.freq_max, float(np.max(np.asarray(
            source.frequencies, dtype=float)))),
        rms_roughness, oass_index_space=False))
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
        c_high=c_high,
        n_wavenumbers=n_wavenumbers,
        integration_offset=integration_offset,
        mean_field=mean_settings,
        notices=tuple(n for n in notices if n is not None),
    )
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


class OASSP(OASES):
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
        These are not adopted from OASR or OAST".
    spectral_exponent : float, optional
        ``M`` of the same spectrum. Must exceed 1.5. Default 2.0.
    roughness_spectrum : {'gaussian', 'goff-jordan'}, optional
        ``'goff-jordan'`` adds option ``'g'``. Default ``'gaussian'``.
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
        different ``k`` are different draws. Default 0.
    scattered_only : bool, optional
        Option ``'s'`` — zero the source arrays so the ``.trf`` holds the
        scattered field alone (``unoassp30.f:628-635``). Default True.
    mean_field : OASP, optional
        The producer run. ``None`` builds one from ``n_time_samples`` /
        ``freq_min`` / ``freq_max`` / ``center_frequency`` and from
        ``c_low`` / ``c_high`` / ``n_wavenumbers`` / ``integration_offset``;
        passing one makes the first four exclusive with it, since the mean
        field owns the FFT grid. The mean field runs on the call's Source,
        so it has the scatterer's geometry (a raw ``'P'`` on a point Source
        is refused, as on :class:`OASP`); a pinned ``n_wavenumbers`` above 8192
        (oassp2's ``nnkr``) is refused, and a ``CMIN`` that differs from
        OASSP's is warned about.
    n_time_samples, freq_min, freq_max, center_frequency : optional
        Passed to the mean-field :class:`OASP`. OASSP's own Block VIII is then
        read back out of the ``.rhs``, never guessed — see Notes.
    options : str, optional
        Raw OASSP option string, exclusive with the typed flags above.
        ``None`` derives the letters.
    integration_offset, n_wavenumbers, c_low, c_high : optional
        As for :class:`OASP`, on both decks of the chain: oassp2 takes its
        incident spectrum from the mean field's ``.rhs`` alone
        (``oasvun31.f:313-332``, ``:55-74``), so a ``c_low`` that admits
        slow seabed branches must reach the mean field too. ``c_high`` is
        inert in cylindrical geometry (a point Source). The receiver ranges
        are ``receiver.ranges``.
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
    Per-model: ``'ssp': 'mean'``, ``'bottom_range': 'median'``.

    Examples
    --------
    >>> from uacpy.models import OASSP
    >>> oassp = OASSP(correlation_length=5.0, spectral_exponent=2.5,
    ...               n_time_samples=512, freq_min=400, freq_max=600)
    >>> h_scattered = oassp.run(env, source, receiver)
    """

    # Declarative metadata (see PropagationModel / ModelSpec). OASSP:
    # range-independent scattered-field realizations over the same layered
    # stack OASP solves the mean field on — INENVI is shared verbatim between
    # the binaries, so the supported axes are OASP's. A line Source runs
    # both decks in plane geometry, option 'P' (unoassp30.f:959-961).
    spec = ModelSpec(
        modes=(RunMode.BROADBAND, RunMode.TIME_SERIES),
        supports={'layered_bottom', 'elastic_media',
                  'rough_surface', 'rough_bottom'},
        source_types=frozenset({'point', 'line'}),
        collapse={'ssp': 'mean', 'bottom_range': 'median'},
        traits=dataclasses.replace(
            _OASES_TRAITS,
            consumes_run_t_start=True,
            # BROADBAND runs the mean field's FFT ladder (see :class:`OASP`).
            announced_band_modes=frozenset({RunMode.TIME_SERIES}),
        ),
    )
    provenance_id = 'oases'
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
    _MEAN_FIELD_KNOBS = ('n_time_samples', 'freq_min', 'freq_max',
                         'center_frequency')

    def __init__(
        self,
        *,
        executable: Optional[Path] = None,
        correlation_length: Optional[float] = None,
        spectral_exponent: float = OASSP_SPECTRAL_EXPONENT,
        roughness_spectrum: str = 'gaussian',
        rms_roughness: Optional[float] = None,
        interface: Optional[int] = None,
        realization: int = OASSP_REALIZATION,
        scattered_only: bool = True,
        mean_field: Optional['OASP'] = None,
        n_time_samples: Optional[int] = None,
        freq_min: Optional[float] = None,
        freq_max: Optional[float] = None,
        center_frequency: Optional[float] = None,
        options: Optional[str] = None,
        integration_offset: float = OASSP_INTEGRATION_OFFSET,
        n_wavenumbers: Optional[int] = None,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = DEFAULT_RUN_TIMEOUT_S,
        collapse: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )
        self.correlation_length = (
            float(correlation_length) if correlation_length is not None
            else None
        )
        self.spectral_exponent = float(spectral_exponent)
        # Kept as passed until the check, whose refusal quotes it; then
        # normalised, so roughness_spectrum='Gaussian' behaves as the
        # default rather than as a pinned non-default. Every comparison
        # (the check, the option letter, the options-exclusivity check)
        # folds the case. OASS does the same.
        self.roughness_spectrum = roughness_spectrum
        self.rms_roughness = (
            float(rms_roughness) if rms_roughness is not None else None
        )
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
        self.center_frequency = (
            float(center_frequency) if center_frequency is not None else None
        )
        self.options = options
        self.integration_offset = float(integration_offset)
        self.n_wavenumbers = n_wavenumbers
        self.c_low = float(c_low) if c_low is not None else None
        self.c_high = float(c_high) if c_high is not None else None
        self._check_knobs()
        self.roughness_spectrum = str(roughness_spectrum).lower()
        # Same read gate as OASP (unoassp30.f:135-142, OFFDB=OFFDBIN at
        # :375). :func:`_oassp_options` always emits 'J' unless a raw string
        # replaces the line.
        _warn_offset_ignored_under_auto_sampling(
            'OASSP', self.integration_offset, self.n_wavenumbers,
            options=_oassp_options(self.options, self.scattered_only,
                                   self.roughness_spectrum),
            offset_letters='Jd')

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        # install.sh:1157 copies the raw executables; bin/oases carries
        # symlinks for oasn/oasp/oasr/oast but not for this one, so the
        # name is 'oassp2', not 'oassp'.
        self._exe = self._resolve_executable(
            executable, lambda: _oases_find_executable(self, 'oassp2'),
        )

    def _build_mean_field(self) -> 'OASP':
        """The OASP producer, with option ``'s'`` guaranteed.

        Without ``'s'`` OASP never calls ``opfilb(45)`` (``unoasp22.f:251-254``)
        and no ``.rhs`` is written at all, so the letter is plumbing rather
        than a physics choice and is added rather than demanded of the caller.

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
        producer never reaches the scattering integral. Its ``cleanup`` is
        OASSP's: its files share OASSP's directory, which OASSP's
        ``cleanup`` keeps or wipes, so the mean field's result records their
        paths exactly when they survive.
        """
        mean = self.mean_field
        if mean is None:
            kwargs = {k: getattr(self, k) for k in self._MEAN_FIELD_KNOBS
                      if getattr(self, k) is not None}
            mean = OASP(options='N J s', n_wavenumbers=self.n_wavenumbers,
                        c_low=self.c_low, c_high=self.c_high,
                        verbose=self.verbose, timeout=self.timeout,
                        cleanup=self.cleanup, **kwargs)
            # Set after construction: OASSP's own constructor has already
            # said whether the offset reaches the binary.
            mean.integration_offset = self.integration_offset
            return mean
        if 's' not in set(str(mean.options or 'N J')) - set(' \t\n'):
            mean = mean.copy(options=f"{mean.options or 'N J'} s")
            self._log("Added option 's' to the mean-field model: without it "
                      "OASP writes no .rhs (unoasp22.f:251-254)")
        return mean.copy(verbose=self.verbose, timeout=self.timeout,
                         cleanup=self.cleanup)

    def _check_knobs(self) -> None:
        """Refuse a constructor knob no run could use
        (:func:`_check_oassp_knobs`). Run at construction and again by every
        run (:meth:`_validate_engine`), since the attributes can be
        reassigned in between."""
        _check_n_wavenumbers_knob(self.n_wavenumbers)
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
        """Stage 2: an option line whose ``.trf`` uacpy cannot read back
        (:func:`_reject_unreadable_oassp_options`), a raw ``'P'`` with a point
        Source (the Source decides the geometry), and an environment whose
        ``.rhs`` oassp2 cannot consume
        (:func:`_require_single_rough_interface`) — all before a mean-field
        run is spent (stage 3 refuses a pinned ``interface=`` that is not its
        rough one, :func:`_oassp_scattering_interface`). A supplied mean field's own raw ``'P'`` on a point Source is
        refused by its own checks, in stage 3."""
        _reject_unreadable_oassp_options(self.options, self.n_wavenumbers)
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
        from its ``.rhs``. A BROADBAND call gets no 1 Hz floor notice of a
        band OASP does not run (``spec.traits.announced_band_modes``)."""
        return settings.engine.mean_field.frequencies

    def _resolve_engine_settings(self, env, source, receiver, settings, *,
                                 given_env, request=None) -> 'OASSPSettings':
        """Stage 3: the mean field's settings, from its producer, and
        :func:`_oassp_settings` over them for this model's knobs."""
        mean_settings, projection_notices = (
            self._build_mean_field()._producer_settings(
                env, _mean_field_source(source), receiver,
                RunMode.BROADBAND, frequencies=settings.frequencies))
        return _oassp_settings(
            type(self).__name__, mean_settings, projection_notices,
            options=_oassp_options(self.options, self.scattered_only,
                                   self.roughness_spectrum),
            correlation_length=self.correlation_length,
            spectral_exponent=self.spectral_exponent,
            rms_roughness=self.rms_roughness,
            realization=self.realization, interface=self.interface,
            c_low_pinned=self.c_low, c_high=self.c_high,
            n_wavenumbers=self.n_wavenumbers,
            integration_offset=self.integration_offset,
            env=env, source=source, receiver=receiver)

    def _announce_engine_settings(self, env, source, receiver,
                                  settings) -> None:
        """Stage 3 (run and run_settings): the mean field's notices, then
        OASSP's own."""
        mean = settings.engine.mean_field
        self._build_mean_field()._announce_engine_settings(
            env, _mean_field_source(source), receiver, mean)
        super()._announce_engine_settings(env, source, receiver, settings)

    # ── stage 4: the mean field, then oassp2 ───────────────────────────

    def _n_launches(self, settings) -> int:
        """Two: the mean-field producer, then oassp2 on its ``.rhs``."""
        return 2

    def _prepare_launches(self, env, settings):
        """The mean-field producer both launches' hooks call."""
        return self._build_mean_field()

    def _mean_field_stage_inputs(self, inputs):
        """The producer's stage inputs for this launch's source depth."""
        return _mean_field_inputs(inputs, inputs.settings.engine.mean_field,
                                  _mean_field_source(inputs.source))

    def _write_input(self, inputs) -> Path:
        """Stage 4. Launch 0: the mean field's deck, written by the
        producer's own :meth:`OASP._write_input`. Launch 1: the OASSP deck,
        written from the settings alone (:func:`_write_oassp_deck`), once the
        mean field's ``.rhs`` is held to them
        (:func:`_check_rhs_matches_settings`): Block VIII (NT, FR1, FR2, DT)
        and the scattering interface are the file's, which
        ``unoassp30.f:181-188`` substitutes anyway, and Block III's FRC is
        the carrier the mean field ran with — TRFHEAD writes it into the
        ``.trf`` header (``oasiun23.f:815``, ``:864-865``, ``:919``)."""
        if inputs.launch == 0:
            producer = inputs.prepared
            self._log(f"Running {producer.model_name} for the mean field "
                      f"(option 's')")
            return producer._write_producer_deck(
                self._mean_field_stage_inputs(inputs))
        engine = inputs.settings.engine
        _check_rhs_matches_settings(inputs.earlier[0].rhs_header, engine)
        deck = inputs.work_dir / f'{_OASSP_BASE_NAME}.dat'
        self._log(f"Writing OASSP input file: {deck} "
                  f"(options={engine.options}, "
                  f"realization={engine.realization}, "
                  f"interface={engine.interface})")
        _write_oassp_deck(deck, inputs.env, inputs.source, inputs.receiver,
                          engine)
        return deck

    def _launch(self, inputs, deck: Path) -> None:
        """Stage 4. Launch 0: the producer's own :meth:`OASP._launch`.
        Launch 1: ``oassp2`` on the deck, its FOR045/FOR046 pointed at the
        mean field's kernel files (named after that run's stem, see
        :meth:`OASES._execute`); refuse a run past OASES' wavenumber bound
        or with no transfer function."""
        if inputs.launch == 0:
            inputs.prepared._launch_producer(
                self._mean_field_stage_inputs(inputs), deck)
            return
        stem = inputs.earlier[0].stem
        proc = self._execute(deck.stem, inputs.work_dir, extra_env={
            'FOR045': f'{stem}.045', 'FOR046': f'{stem}.046',
        })
        # unoassp30.f has no NWVNO.GT.NP test either (see
        # _reject_wavenumber_overrun), and its reference range min(cref·NX·DT + RM, 6·RM)
        # (:295-306) can push NW ~2× past the mean-field OASP's, so the
        # producer passing is no proxy for this run being in bounds.
        _reject_wavenumber_overrun(
            self.model_name, proc,
            _oassp_wavenumber_echo_scale(inputs.settings.engine.options))
        self._require_output(
            [inputs.work_dir / f'{deck.stem}.trf'],
            what='a transfer-function file', process=proc,
        )

    def _read_output(self, inputs, deck: Path):
        """Stage 4. Launch 0: the mean field's own result (the producer's
        stages 4-6, :meth:`~PropagationModel._run_producer_launch`) and the
        header of the ``.rhs`` it wrote, with the ``.045``/``.046`` pair oassp2
        opens required. Launch 1: the scattered transfer function on the
        requested axes (:func:`_read_trf_on_requested_ranges`)."""
        if inputs.launch == 0:
            return self._read_mean_field(inputs, deck)
        output_file = inputs.work_dir / f'{deck.stem}.trf'
        self._log(f"Reading OASSP output: {output_file}")
        return _read_trf_on_requested_ranges(output_file, inputs.receiver)

    def _read_mean_field(self, inputs, deck: Path) -> _MeanFieldRun:
        """The mean-field launch's output: the producer's own result, and
        the header of the ``.rhs`` beside the ``.046`` oassp2 opens."""
        result = inputs.prepared._run_producer_launch(
            self._mean_field_stage_inputs(inputs), deck)
        rhs_path = self._require_output(
            [inputs.work_dir / f'{deck.stem}.045'],
            what="a mean-field .rhs (option 's')",
            hint=("OPFILB opens unit 45 with STATUS='UNKNOWN' "
                  "(oashun21.f:634), so an empty file here means the "
                  "producer wrote no scattering right-hand sides."),
        )
        self._require_output(
            [inputs.work_dir / f'{deck.stem}.046'],
            what="a mean-field .vol (FOR046)",
            hint=("OASSP opens unit 46 unconditionally (unoassp30.f:128) "
                  "and its IOER test only covers the last OPFILb (:130)."),
        )
        return _MeanFieldRun(result=result,
                             rhs_header=read_oases_rhs_header(rhs_path),
                             stem=deck.stem)

    def _to_result(self, inputs, deck, raw) -> Field:
        """Stage 5: the scattered complex ``H(f)`` on ``(depth, range,
        frequency)`` for BROADBAND, its synthesis for TIME_SERIES, with the
        mean field's result as ``components['mean_field']``.

        Convention: (n_depth, n_range, n_frequencies) — trailing axis is the
        variable dim. Source axes: (freq, range, depth). Negated like
        OASP's (the payload is the normal stress, -p in the water:
        oasp.tex:185): the raw payload comes from the same TRFHEAD writer
        with the same sign (measured with scattered_only=False on a
        near-smooth seabed, the raw total field over OASP's raw mean field
        is +1: |ratio| 0.995, arg 0.35 deg), so the same negation is what
        makes 'travelling_wave' mean the family's convention here too.
        """
        mean, trf_data = raw
        settings = inputs.settings
        source = inputs.source
        rhs = mean.rhs_header
        rhs_path = inputs.work_dir / f'{mean.stem}.045'
        vol_path = inputs.work_dir / f'{mean.stem}.046'
        tf_reordered = -np.transpose(
            trf_data['transfer_function'], (2, 1, 0)).astype(np.complex128)
        result = Field(
            data=tf_reordered,
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
            **self._result_kwargs(
                source,
                phase_reference=PhaseReference.TRAVELLING_WAVE.value,
                backend='oassp',
                frequencies=trf_data['freq'],
                source_depth=trf_data['source_depth'],
                center_frequency=trf_data['center_frequency'],
                freq_max=rhs.freq_max,
                components={'mean_field': mean.result},
            ),
        )
        result = self._mask_source_axis(result, source)
        if settings.time is not None:
            result = self._finish_broadband(result, settings)
        self._attach_output_paths(
            result, inputs.work_dir, deck[1].stem,
            primary_files=(('trf_file', '.trf'),),
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
