"""OASS: the OASES rough-interface reverberation program."""

import re
from dataclasses import dataclass
from types import MappingProxyType
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from uacpy.models.base import DEFAULT_RUN_TIMEOUT, _host_options
from uacpy.models._knobs import whole_count
from uacpy.models._spec import ModelSpec
from uacpy.models._window import (
    check_pinned_window, refuse_inverted_window,
)
from uacpy.core.run_settings import (
    EngineSettings, OutputSpec, RunMode, RunSettings,
)
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.results import Result, Field, Covariance
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, UnsupportedFeatureError,
    ValidityWarning,
)
from uacpy.io.oases_writer import (
    write_oass_input, oass_bottom_interfaces, oases_wavenumber_bounds,
    refuse_non_equispaced_oass_ranges, refuse_unintegrable_spectral_exponent,
    refuse_unusable_correlation_length, _OASES_MAX_WAVENUMBERS, _check_oass_range_count, _OASS_REVERB_OPTIONS,
)
from uacpy.io._parsers import parse_oast_tl
from uacpy.io.oases_reader import read_oasn_covariance
from uacpy.models.oases._common import (
    C_LOW_MATCH_RTOL,
    _OASES_TRAITS, _OASS_MIN_MEAN_FIELD_ROUGHNESS, _OASS_SPECTRA,
    _OASS_SURFACE_INTERFACE, _geometry_options, _oases_binary,
    _reject_plane_geometry_letter_on_a_point_source,
    _require_mean_field_rhs,
    _roughness_notice, SCATTERING_UNPINNED_FIELDS, resolve_spectral_exponent,
)
from uacpy.core.engine_defaults import OASES_PRESSURE_LETTER
from uacpy.models.oases._base import OASES
from uacpy.models.oases.oast import OAST, _oast_options
from uacpy.models.oases.oasr import OASR
from uacpy.models.oases._mean_field import _ScatteringPostProcessor
from uacpy.models._notices import message_notice
from uacpy.models._extract import (
    attach_output_paths, mask_source_axis, result_kwargs,
)
from uacpy.models._launch import require_output


#: The option line of the OAST mean field OASS builds when none is supplied:
#: OAST's own default letters (the bulk pressure,
#: :data:`~uacpy.core.engine_defaults.OASES_PRESSURE_LETTER`, ``'J'`` and the
#: TL table ``'T'`` its wrapper reads) and ``'s'``, which writes the ``.rhs``
#: OASS reads (oaseun31.f:1899-1900).
_OASS_MEAN_FIELD_OPTIONS = f"{OASES_PRESSURE_LETTER} J T s"


#: OASS echoes the wavenumber count it settled on after re-deriving it from
#: the .rhs (unoass21.f:231, FORMAT 550 at :50).
_OASS_NW_ECHO = re.compile(r'NW\s+IC1\s+IC2\s*\n\s*(\d+)')


#: FORMAT 550 (unoass21.f:50) prints NWVNO under 1X,I5: any count >= 100000
#: renders as asterisks, which is itself proof of a count far past NP.
_OASS_NW_I5_OVERFLOW = re.compile(r'NW\s+IC1\s+IC2\s*\n\s*\*+')


def _oass_wavenumber_echo(process) -> Optional[int]:
    """``NWVNO`` OASS ran with, from its own stdout.

    Under the reverberation options the deck's Block VII is discarded and the
    count comes from the mean field's wavenumber step (``unoass21.f:209-215``),
    so the binary's echo is the only place the value it used appears.
    """
    stdout = getattr(process, 'stdout', '') or ''
    match = _OASS_NW_ECHO.search(stdout)
    if match:
        return int(match.group(1))
    if _OASS_NW_I5_OVERFLOW.search(stdout):
        # The I5 field overflowed: the count is >= 100000, far past
        # NP = 65536 — return a value the overrun check trips on rather
        # than degrading to silence exactly when the overrun is worst.
        return 100000
    return None


#: File root of the deck and outputs of the scattering half of an OASS run.
_OASS_BASE_NAME = 'oass_run'


#: The one product letter each OASS run mode emits (see the Notes of
#: :class:`OASS`).
_OASS_PRODUCT_LETTER = {
    RunMode.REVERBERATION: 'r',
    RunMode.COVARIANCE: 'a',
}


@dataclass(frozen=True, eq=False)
class OASSSettings(EngineSettings):
    """The settings one :class:`OASS` run resolved, before launching:
    ``OASS(...).run_settings(env, source, receiver).engine``, and
    ``result.run_settings.engine`` on the result it produced.

    Attributes
    ----------
    options : str
        The OASS deck's option line, with the run mode's product letter
        and ``'P'`` for a line Source.
    interface : int
        ``INTFC``, the deck layer whose roughness scatters.
    correlation_length, spectral_exponent : float
        ``CL`` (m) and ``M`` of the roughness power spectrum.
    rms_roughness : float or None
        ``|RG|`` (m) on the OASS deck; ``None`` takes the environment's.
    frequency : float
        The one frequency (Hz) of the chain.
    c_low : float
        Block VII's ``CMIN`` (m/s).
    c_low_origin : str
        Where ``c_low`` came from: the constructor, or the mean field's own
        bound.
    c_high : float
        Block VII's ``CMAX`` (m/s).
    c_high_origin : str
        Where ``c_high`` came from: the constructor, or OASES' "no upper
        limit" 1e8 (:func:`~uacpy.io.oases_writer.oases_wavenumber_bounds`).
    receiver_gains_dB : ndarray or None
        Block VI column 5, per element (dB).
    mean_field : RunSettings
        The settings of the mean-field OAST run whose ``.rhs`` OASS reads.
    notices : tuple of Notice
        What the run says about the chain it resolved (a CMIN that differs
        from the mean field's, a roughness past the small-roughness theory
        at the run's frequency).
    """

    options: str
    interface: int
    correlation_length: float
    spectral_exponent: float
    rms_roughness: Optional[float]
    frequency: float
    c_low: float
    c_low_origin: str
    c_high: float
    c_high_origin: str
    receiver_gains_dB: Optional[np.ndarray]
    mean_field: RunSettings

    _ARRAY_FIELDS = ('receiver_gains_dB',)

    @classmethod
    def from_dict(cls, d) -> 'OASSSettings':
        d = dict(d)
        d['mean_field'] = RunSettings.from_dict(d['mean_field'])
        return cls(**d)


def _check_oass_knobs(*, correlation_length, spectral_exponent,
                      roughness_spectrum, mean_field, options,
                      multiple_scattering) -> None:
    """Refuse an OASS knob no run could use: a missing or non-positive
    correlation length, an unintegrable spectral exponent, an unknown roughness
    spectrum, a mean field that cannot feed OASS, a raw ``options``
    string beside the typed flags it would discard."""
    if correlation_length is None:
        raise ConfigurationError(
            "OASS(correlation_length=…) is required: the roughness power "
            "spectrum P(k) is what the reverberation integral integrates "
            "over, and oass.tex:182-183 states the rms roughness, "
            "correlation length and spectral exponent are not adopted "
            "from the OAST/OASR mean-field run.",
            remediation=("Pass the interface correlation length in "
                         "metres, e.g. OASS(correlation_length=10.0)."),
        )
    # A negative length is OASES' volume-scattering switch; zero and a
    # non-finite value are no length. The deck writer refuses the same.
    refuse_unusable_correlation_length('OASS', correlation_length)
    if spectral_exponent is not None:
        refuse_unintegrable_spectral_exponent('OASS', spectral_exponent)
    if str(roughness_spectrum).lower() not in _OASS_SPECTRA:
        raise ConfigurationError(
            f"OASS(roughness_spectrum={roughness_spectrum!r}) is not "
            f"a roughness spectrum "
            f"OASS implements; valid are "
            f"{sorted(_OASS_SPECTRA)} — GETOPT's only spectrum letter is "
            f"'g'/'G' (goff, unoass21.f:674) and its absence means "
            f"Gaussian."
        )
    if isinstance(mean_field, OASR):
        raise ConfigurationError(
            "OASS(mean_field=OASR(...)) divides by zero inside oass2. "
            "OASS keeps only the .rhs records whose interface index "
            "matches INTFC (unoass21.f:200), and takes DLWVNO from the "
            "gap between the first two it keeps (:206) before "
            "NWVNO=(WKMAX-WK0)/DLWVNO+1 at :209. Two things stop that "
            "from working on an OASR .rhs. Its layer 1 is the water "
            "half-space, so SCTRHS stamps IN1=2 (oaseun31.f:2308-2309, "
            "IN1=IN+1) on every record, while INTFC here is a bottom "
            "deck layer, 3 or deeper — no record matches, KCNT stays 0 "
            "and DLWVNO is never assigned. And OASR calls SCTRHS from "
            "inside its angle loop with WVNO = Re(AK(1,1))*cos(ANG) "
            "(oasjun21.f:70,108), so even a matching record set is "
            "cosine-spaced in wavenumber, not the uniform grid the "
            "DLWVNO-from-two-records inference assumes. oass.tex:18-23 "
            "names OASR as a producer, but only for a deck whose "
            "scattering interface is its layer 2.",
            remediation="Pass an OAST whose option line has 's' (the "
                        "default mean field), or None.",
        )
    if (mean_field is not None
            and not isinstance(mean_field, OAST)):
        raise ConfigurationError(
            f"OASS(mean_field={type(mean_field).__name__}) cannot "
            f"produce the boundary operators OASS consumes: oass.tex:18-23 "
            f"names "
            f"OAST and OASR as the producers, and OASS reads exactly one "
            f"per-frequency header record from the .rhs "
            f"(unoass21.f:123).",
            remediation="Pass an OAST whose option line has 's'.",
        )
    if options is not None:
        pinned = [n for n, on in (
            ('roughness_spectrum',
             str(roughness_spectrum).lower() != 'gaussian'),
            ('multiple_scattering', bool(multiple_scattering)))
            if on]
        if pinned:
            raise ConfigurationError(
                f"OASS: options={options!r} replaces the whole "
                f"option line, so {', '.join(pinned)} would be discarded "
                f"silently. Pass either the raw string or the typed "
                f"flags, not both — note the derived string carries the "
                f"product letter for the run mode ('r' or 'a'), which a "
                f"raw string must repeat."
            )


def _oass_options(options, run_mode: RunMode, roughness_spectrum,
                  multiple_scattering) -> str:
    """The OASS option line for this run.

    A raw ``options`` string is written verbatim; otherwise the letters
    are the run mode's product letter plus the typed flags. The
    constructor rejects the two being combined. The geometry letter
    ``'P'`` is the Source's (:meth:`OASS._resolve_engine_settings`).
    """
    if options is not None:
        return options
    opt = [_OASS_PRODUCT_LETTER[run_mode]]
    if str(roughness_spectrum).lower() == 'goff-jordan':
        opt.append('g')
    if multiple_scattering:
        opt.append('p')
    return ' '.join(opt)


def _reject_unreadable_oass_options(options: str,
                                    run_mode: RunMode) -> None:
    """Refuse raw option letters whose output uacpy cannot read back.

    OASES scans the option line character by character
    (``unoass21.f:599`` ``READ(1,200) OPT`` / ``FORMAT(40A1)``), so the
    tests are on the character set rather than on whitespace-split
    tokens. The deck-level letter validation (unknown letters, two
    products in one deck) belongs to ``write_oass_input`` and is not
    repeated here.
    """
    chars = set(str(options)) - set(' \t\n')

    contours = sorted({'C', 'D'} & chars)
    if contours:
        raise UnsupportedFeatureError(
            'OASS',
            f"option(s) {contours} (horizontal-correlation / "
            f"reverberation-intensity contours) — the output goes to the "
            f"CONDRW/CONDRB pair on units 28/29 (unoass21.f:277-278) and "
            f"uacpy has no reader for that format. 'D' computes the same "
            f"REVINT quantity RunMode.REVERBERATION already returns",
        )
    kernels = sorted({'I', 'S', 'c'} & chars)
    if kernels:
        raise UnsupportedFeatureError(
            'OASS',
            f"option(s) {kernels} (scattering kernels for one incident "
            f"plane wave) — their curves carry the INTGR / SPECT plot "
            f"tags, and read_oast_tl reads only the TL-vs-range curves "
            f"(tag TLRAN), so nothing would be found in the .plt",
        )
    if 'Z' in chars:
        raise UnsupportedFeatureError(
            'OASS',
            "option 'Z' (SVP plot) — it makes the binary read two further "
            "records after Block IX (unoass21.f:319-320) that "
            "write_oass_input does not emit, so the deck ends early and "
            "OASS reads past it",
        )
    letter = _OASS_PRODUCT_LETTER[run_mode]
    if letter not in chars:
        raise ConfigurationError(
            f"OASS(options={options!r}) does not carry {letter!r}, the "
            f"product letter for run_mode={run_mode.name}: the deck would "
            f"compute something other than what run() then tries to read.",
            remediation=(f"Add {letter!r} to the option string, or run the "
                         f"mode whose product it does carry."),
        )


def _require_single_frequency(source: Source) -> float:
    """The one frequency this chain runs at.

    ``unoass21.f:123`` pins ``nfreq = 1`` before reading the ``.rhs``, so
    OASS reads exactly one per-frequency header record and treats every
    record after it as a 19-word ``SCTRHS`` record. A multi-frequency
    mean-field run therefore ends in a Fortran I/O error rather than a
    clean stop.
    """
    freqs = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
    if freqs.size > 1:
        raise ConfigurationError(
            f"OASS: the mean-field run must be single-frequency, but "
            f"source.frequencies has {freqs.size} entries "
            f"({freqs.min():g}-{freqs.max():g} Hz). OASS pins nfreq=1 "
            f"before reading the .rhs (unoass21.f:123) and reads the "
            f"second frequency's 5-word header as a 19-word boundary-"
            f"operator record.",
            remediation=("Run one frequency at a time: "
                         "Source(depths=…, frequencies=f)."),
        )
    return float(freqs[0])


def _oass_interface(env: Environment, interface) -> int:
    """``INTFC`` for this environment: ``interface`` when pinned, else
    the seafloor.

    The sea surface, deck layer 2, is a legal target as well as the bottom
    records: ``ROUGH(M)`` is the roughness of the interface at the top of
    layer M (``oaseun31.f:377-383``), ``SCTRHS`` writes its operators under
    ``IN1 = 2`` like any other, and ``write_oass_input`` routes the
    nine-token roughness tail to the water record for it. An ice keel or a
    wind-roughened surface is the usual case, and ``spec.supports``
    advertises ``'rough_surface'``.

    Also checks that the interface is rough in the **mean-field** deck:
    ``SCTRHS`` skips any interface with ``ROUGH2 < 1e-10``
    (``oaseun31.f:2310``, ``ROUGH2 = ROUGH**2`` at ``:379``), so a smooth
    environment produces a ``.rhs`` with no boundary-operator records at
    all — ``KCNT`` stays 0, ``DLWVNO`` is never set and OASS divides by
    zero. The environment's roughness reaches the mean-field deck
    whatever ``rms_roughness`` overrides in the OASS deck.
    """
    first_bottom, roughness = oass_bottom_interfaces(env)
    interface = (first_bottom if interface is None
                 else int(interface))
    if interface == _OASS_SURFACE_INTERFACE:
        rg = abs(float(env.surface.roughness))
        named = "the sea surface (deck layer 2)"
        fix = ("Set the roughness on the surface, e.g. "
               "BoundaryProperties(..., roughness=0.5) on env.surface. ")
    else:
        index = interface - first_bottom
        if not 0 <= index < len(roughness):
            raise ConfigurationError(
                f"OASS(interface={interface}) is not a scattering "
                f"interface of this deck: it carries the sea surface at "
                f"deck layer {_OASS_SURFACE_INTERFACE} and the seabed "
                f"stack at layers {first_bottom}.."
                f"{first_bottom + len(roughness) - 1}.",
                remediation=("The roughness spectrum can only be attached "
                             "to the first water record or a bottom layer "
                             "record; an interior water interface has no "
                             "RG column INENVI would re-read as CL and M."),
            )
        rg = abs(float(roughness[index]))
        named = (f"the seafloor (deck layer {interface})" if index == 0
                 else f"the top of seabed layer {index + 1} "
                      f"(deck layer {interface})")
        fix = ("Set the roughness on the environment, e.g. "
               "BoundaryProperties(..., roughness=0.5). ")
    if rg <= _OASS_MIN_MEAN_FIELD_ROUGHNESS:
        raise ConfigurationError(
            f"OASS: {named} is smooth in the environment "
            f"(roughness = {rg:g} m), so the mean-field run "
            f"writes an empty .rhs — SCTRHS skips any interface with "
            f"ROUGH2 < 1e-10 (oaseun31.f:2310, :379) and OASS then has no "
            f"wavenumber step to integrate on.",
            remediation=(fix + "OASS(rms_roughness=…) overrides only the "
                         "OASS deck, not the mean field's."),
        )
    return interface


def _reject_unusable_mean_field_options(mean_field) -> None:
    """Refuse a producer whose option line cannot feed OASS, before it
    is run: without ``'s'`` it writes no ``.rhs``, and without ``'T'``
    its own wrapper has no TL table to read (OAST refuses such a line at
    construction; this catches one assigned afterwards)."""
    mean = mean_field if mean_field is not None else OAST(
        options=_OASS_MEAN_FIELD_OPTIONS)
    # The option line the producer (an OAST: the constructor refuses
    # every other) will write, derived from its typed flags or its raw
    # string by _oast_options.
    mean_options = _oast_options(mean.options, mean.complex_contour,
                                 mean.compute_depth_average)
    chars = set(mean_options) - set(' \t\n')
    # SCTOUT gates the whole boundary-operator dump.
    _require_mean_field_rhs(
        'OASS', mean, mean_options, 'oaseun31.f:1899-1900',
        f"OAST(options={_OASS_MEAN_FIELD_OPTIONS!r})")
    # The producer runs through its own stage hooks, which read its own
    # primary table: OAST's .plt (FOR020), written under 'T'. Its own
    # knob check refuses a line without 'T' only when the producer
    # runs, so OASS names its requirement before anything runs.
    if 'T' not in chars:
        raise ConfigurationError(
            f"OASS(mean_field={type(mean).__name__}(options="
            f"{mean_options!r})): the producer's option line must also "
            f"contain 'T'. OASS consumes only the .rhs, but the mean field "
            f"is run through its own wrapper, which reads the table 'T' "
            f"writes (OAST's FOR020 .plt) and refuses a line without it.",
            remediation=(f"e.g. OASS(mean_field=OAST(options="
                         f"{_OASS_MEAN_FIELD_OPTIONS!r}))."),
        )


def _c_low_mismatch_notice(c_low: float,
                           mean_c_low: float) -> Optional[str]:
    """The notice for an OASS ``c_low`` that is not the mean field's
    ``CMIN``, or ``None``.

    Measured on one ``.rhs``, option ``'r'``, only ``CMIN`` changed:
    1350 → 2000 m/s moved the field by 30.15 dB; 1350 → 900 m/s moved it
    by 1e-4 dB. Raising ``c_low`` truncates the scattering integral,
    because ``REVINT``/``REVCOV`` bound their reads of the mean field's
    buffer themselves (``oassun26.f:744-748``, ``:958-962``) — wavenumbers
    past the mean field's grid simply contribute nothing.
    """
    if np.isclose(c_low, mean_c_low, rtol=C_LOW_MATCH_RTOL):
        return None
    direction = ('truncates' if c_low > mean_c_low
                 else 'does not extend')
    return (
        f"OASS(c_low={c_low:g}) differs from the mean field's "
        f"{mean_c_low:g} m/s. c_low is physically significant here, not a "
        f"tuning knob: it {direction} the scattering integral over the "
        f".rhs wavenumber grid (measured 30.15 dB for 1350 → 2000 m/s, "
        f"1e-4 dB for 1350 → 900). Leave c_low unset to inherit the mean "
        f"field's bound.")


def _oass_settings(name: str, mean_settings, projection_notices, *,
                   options, interface, c_low_pinned, c_high,
                   correlation_length, spectral_exponent,
                   rms_roughness, receiver_gains_dB, env,
                   source) -> OASSSettings:
    """The OASS deck's option line (``options`` resolved for the run
    mode), interface and Block VII window, over ``mean_settings``, the
    mean field's own settings (its COHERENT_TL run on the same
    environment and receiver). ``c_low_pinned`` unset takes the mean
    field's ``CMIN``; one that differs from it is noted (see
    :func:`_c_low_mismatch_notice`). The resolved window is refused when
    ``c_low`` is not below the ``CMAX`` the deck carries (``c_high``, else
    :func:`oases_wavenumber_bounds`'s "no upper limit"): ``WK0 = ω/CMAX``
    then lies above ``WKMAX = ω/CMIN`` and the scattered field is zero
    (unoass21.f:190-191). ``name`` is the model the refusals and the
    roughness notice name."""
    mean_c_low = mean_settings.engine.c_low
    notices = list(projection_notices)
    if c_low_pinned is None:
        c_low, origin = mean_c_low, "the mean field's CMIN"
    else:
        c_low, origin = c_low_pinned, 'OASS(c_low=…)'
        notice = _c_low_mismatch_notice(c_low_pinned, mean_c_low)
        if notice is not None:
            notices.append(message_notice(notice, ValidityWarning))
    if c_high is None:
        cmax = oases_wavenumber_bounds(env)[1]
        c_high_origin = "1e8, OASES' no upper limit (oast.tex:487, :568)"
    else:
        cmax, c_high_origin = c_high, f'{name}(c_high=…)'
    refuse_inverted_window(name, float(c_low), float(cmax),
                           c_low=c_low_pinned, c_high=c_high)
    options = _geometry_options(options, source)
    interface = _oass_interface(env, interface)
    frequency = _require_single_frequency(source)
    notices.append(_roughness_notice(
        name, env, interface, frequency,
        rms_roughness, oass_index_space=True))
    # GETOPT reads 'g' or 'G' as the Goff-Jordan spectrum (unoass21.f:674).
    spectral_exponent, unused = resolve_spectral_exponent(
        name, spectral_exponent, goff=bool(set(options) & {'g', 'G'}))
    notices.append(message_notice(unused, FallbackWarning))
    return OASSSettings(
        options=options,
        interface=interface,
        correlation_length=correlation_length,
        spectral_exponent=spectral_exponent,
        rms_roughness=rms_roughness,
        frequency=frequency,
        c_low=float(c_low),
        c_low_origin=origin,
        c_high=float(cmax),
        c_high_origin=c_high_origin,
        receiver_gains_dB=receiver_gains_dB,
        mean_field=mean_settings,
        notices=tuple(n for n in notices if n is not None),
    )


class OASS(_ScatteringPostProcessor):
    """
    OASS — OASES Scattering and Reverberation Module.

    Computes the spatial statistics of the field a rough interface scatters:
    the reverberation LOSS against range and receiver depth
    (``RunMode.REVERBERATION``, the default) or the reverberation covariance
    across the receiver array (``RunMode.COVARIANCE``).

    A loss, not a level: OASES writes ``-10·log10 E[|p_scat|²]``, so a larger
    number is a *weaker* scattered field — the same direction as transmission
    loss, and ``RL = SL -`` this. See :meth:`_reverberation_field` for the
    derivation from the vendored source.

    **Covariance unit.** The ``COVARIANCE`` matrix is ``E[p_i p_j*]`` of the
    scattered pressure for a source of unit level — Pa² for a 1 Pa source
    (a hydrophone records the normal stress ``σ_zz``, "negative of
    pressure in water", "in Pa for source level 1 Pa", ``oass.tex:216-218``;
    the sign cancels in every product) — times each
    receiver's gain (Block VI, dB). It is the variance of a deterministic
    source's scattered field, not a power spectral density: OASN's noise
    covariance is a power spectral density in Pa²/Hz. It is
    computed at zero range offset (``oass.tex:255-258``) for the array the
    deck carries on the z axis (``X = Y = 0``), i.e. a vertical array at the
    source's range: ``receiver.ranges`` place no receiver. For a receiver
    inside an elastic layer ``σ_zz`` is built from both the compressional
    and the shear potentials (``oasaun31.f:1489-1491``) and is not ``-p``:
    there the covariance is that of the normal stress. They size the
    mean field's run, whose wavenumber sampling the integral reuses (2 %
    between a 0.2-2 km and a 0.5-0.9 km axis, 100 m Pekeris, 100 Hz).

    OASS is a **post-processor**. It consumes the mean-field boundary
    operators an OAST run writes with option ``'s'`` into a ``.rhs``
    file, and under the reverberation options it re-derives its whole
    wavenumber axis from them (``unoass21.f:197-216``). ``run()`` therefore
    drives two binaries in one work dir: the mean-field model first, then
    ``oass2``. The mean-field ``Result`` is kept as
    ``components['mean_field']`` so the coherent field is not thrown
    away.

    Parameters
    ----------
    executable : Path, optional
        Path to the ``oass2`` binary. Auto-detected if ``None``.
    correlation_length : float
        ``CL`` (m), the roughness correlation length of the scattering
        interface. **Required**: ``oass.tex:182-183`` — the rms roughness,
        correlation length and spectral exponent "must be specified. These
        are not adopted from OASR or OAST." A positive finite length:
        zero, NaN and infinity are refused (``ConfigurationError``), a
        negative one is OASES' volume-scattering switch
        (``UnsupportedFeatureError``; :func:`~uacpy.io.oases_writer.
        refuse_unusable_correlation_length`).
    spectral_exponent : float
        ``M``, the roughness power-spectrum exponent. Must exceed 1.5 or the
        spectrum is not integrable. ``None`` (default) is 2.5, OASES' own
        Goff-Jordan exponent (oaseun31.f:103-104;
        :data:`~uacpy.core.engine_defaults.OASSP_SPECTRAL_EXPONENT`). The
        default Gaussian spectrum does not read it (oaseun31.f:3689-3692): set
        there, the run says it is not used.
    roughness_spectrum : str
        ``'gaussian'`` (default, OASES' own: "Uses a Goff-Jordan roughness
        spectrum as opposed to the default gaussian spectrum",
        oass.tex:136-137) or ``'goff-jordan'`` (option ``'g'``,
        ``unoass21.f:674``).
    rms_roughness : float, optional
        ``|RG|`` (m) at the scattering interface in the **OASS** deck.
        ``None`` takes the environment's own roughness there. The mean-field
        deck always uses the environment's value — ``oass.tex:184-186``
        allows the two to differ.
    interface : int, optional
        ``INTFC``, the OASES deck-layer index of the scattering interface
        (``unoass21.f:151``). ``None`` selects the seafloor, i.e. the first
        bottom record of the deck.
    multiple_scattering : bool
        Option ``'p'`` (``rescat``): perturbed boundary operator, i.e. the
        loss through re-scattering is included. ``oass.tex:163-166`` states
        the bound in LEVEL terms — this option "yields lower bound for reverb
        levels", the default single-scattering kernel an upper bound. The
        Field stores the negated quantity, so on the numbers you get back the
        direction reverses: ``'p'`` returns the larger (weaker-field) losses
        and the default the smaller ones.
    mean_field : OAST, optional
        The producer run. ``None`` builds ``OAST(options='K J T s')`` (the
        bulk pressure, as every OAST run's default, and ``'s'``, which writes
        the ``.rhs``); a supplied producer whose option line has no ``'s'``
        is refused. A
        supplied model keeps its own configuration, ``verbose`` and
        ``timeout`` included; only ``work_dir``, ``cleanup`` and
        ``use_tmpfs`` are redirected onto this run, whose directory holds
        the ``.rhs`` oass2 reads. An ``OASR`` is refused: its ``.rhs``
        stamps every record with interface index 2 and samples wavenumber
        as ``cos(angle)``, neither of which the ``INTFC``-match /
        uniform-``DLWVNO`` inference at ``unoass21.f:200-211`` can consume.
    c_low, c_high : float, optional
        ``CMIN``/``CMAX`` (m/s). ``c_low`` is **physically significant, not a
        tuning knob**: it truncates the scattering integral. Measured on one
        ``.rhs``, raising it from 1350 to 2000 m/s moved the answer by
        30.15 dB while lowering it to 900 m/s changed nothing (1e-4 dB),
        because ``REVINT``/``REVCOV`` bound their own reads of the mean
        field's wavenumber buffer (``oassun26.f:744-748``, ``:958-962``).
        ``None`` takes the mean field's own bound, and a value that differs
        from it warns.
    receiver_gains_dB : ndarray, optional
        Per-element gain (dB), Block VI column 5. Scalar or one per element.
    options : str, optional
        Raw OASS option letters, written verbatim; ``None`` derives them from
        ``run_mode`` plus ``roughness_spectrum`` / ``multiple_scattering``,
        and ``'P'`` for a line Source. Combining a raw string with either
        flag raises ``ConfigurationError`` — the string replaces the whole option
        line, so a flag passed alongside it would be discarded.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    **The Source decides the geometry.** ``Source(source_type='line')`` runs
    the mean-field OAST and the OASS deck in plane geometry (option ``'P'``,
    ``ICDR=1``, ``unoass21.f:654-656``), a point Source both cylindrical, so
    the two decks always describe the same problem, as OASSP's do.

    **There is no field-parameter option letter in OASS** and ``'R'`` means
    *reverberant field*, not radial stress — see ``write_oass_input``'s
    docstring for the manual/source disagreements the deck rides on.

    One product per run: ``'a'`` on the same option line as ``'r'`` returns a
    **silently zero** covariance, because ``REVINT`` zeroes ``ROUGH2`` for
    every layer before ``REVCOV`` reads it (``oassun26.f:683-688`` vs
    ``:899-904``). The run mode picks exactly one letter, and the writer
    refuses any deck that asks for two.

    ``RunMode.REVERBERATION`` returns a real dB :class:`Field` tagged
    ``kind='reverberation'`` — ``-10·log10 E[|p_scat|²]``, **not**
    transmission loss, so it does not compare against a TL field.

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).** Per-model:
    ``'ssp': 'mean'`` (the layer stack is kept).

    Examples
    --------
    >>> from uacpy.models import OASS
    >>> oass = OASS(correlation_length=10.0, rms_roughness=0.5)
    >>> reverb = oass.run(env, source, receiver)
    >>> cov = oass.compute_covariance(env, source, receiver)
    """

    # Declarative metadata (see PropagationModel / ModelSpec). OASS:
    # range-independent reverberation from one rough interface; multi-layer
    # fluid + elastic bottom honoured. Single spectral solve → mean SSP /
    # median bottom column.
    spec = ModelSpec(
        modes=(RunMode.REVERBERATION, RunMode.COVARIANCE),
        # The layered stack OASP solves the mean field on (INENVI is shared
        # verbatim between the binaries), so the capabilities are OASP's;
        # a rough interface is column 7 (RG) of its layer record
        # (oaseun31.f:54).
        supports={'layered_bottom', 'elastic_media',
                  'rough_surface', 'rough_bottom'},
        # Both decks run in plane geometry for a line Source, option 'P'
        # (unoass21.f:654-656).
        source_types=frozenset({'point', 'line'}),
        collapse={'ssp': 'mean'},
        traits=_OASES_TRAITS,
    )
    provenance_id = 'oases'

    _UNPINNED_FIELDS = SCATTERING_UNPINNED_FIELDS
    _BINARIES = MappingProxyType({'oass': _oases_binary('oass2')})
    # REVERBERATION: -10·log10 E[|p_scat|²] in dB, a loss (see
    # _reverberation_field); COVARIANCE: the .xsm matrix.
    outputs = MappingProxyType({
        RunMode.REVERBERATION: OutputSpec(
            'Field', kind='reverberation', unit='dB'),
        RunMode.COVARIANCE: OutputSpec('Covariance'),
    })

    def __init__(
        self,
        *,
        executable: Optional[Path] = None,
        # Roughness statistics of the scattering interface (Block IV).
        correlation_length: Optional[float] = None,
        spectral_exponent: Optional[float] = None,
        roughness_spectrum: str = 'gaussian',
        rms_roughness: Optional[float] = None,
        interface: Optional[int] = None,
        multiple_scattering: bool = False,
        # Mean-field producer. ``None`` builds
        # OAST(options=_OASS_MEAN_FIELD_OPTIONS).
        mean_field: Optional['OASES'] = None,
        # Wavenumber bounds (Block VII line 1).
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        # Array element gains (Block VI column 5), dB.
        receiver_gains_dB: Optional[np.ndarray] = None,
        options: Optional[str] = None,
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
        # normalised, so the option letter and the options-exclusivity check
        # compare case-folded values and OASS and OASSP accept the same
        # spellings.
        self.roughness_spectrum = roughness_spectrum
        self.rms_roughness = (
            float(rms_roughness) if rms_roughness is not None else None
        )
        # OASS reads INTFCE as an INTEGER layer index (unoass21.f:151).
        if interface is not None:
            whole_count('OASS: interface', interface, 1)
        self.interface = int(interface) if interface is not None else None
        self.multiple_scattering = bool(multiple_scattering)
        self.mean_field = mean_field
        self.c_low = float(c_low) if c_low is not None else None
        self.c_high = float(c_high) if c_high is not None else None
        self.receiver_gains_dB = (
            np.asarray(receiver_gains_dB, dtype=float)
            if receiver_gains_dB is not None else None
        )
        # Raw OASS option string. ``None`` lets the wrapper derive it from the
        # run mode and the typed flags; a raw string replaces the deck's
        # option line outright, so the two ways of specifying it are
        # exclusive.
        self.options = options
        # OASS takes no integration_offset, n_wavenumbers or
        # integration_window (OAST/OASN/OASP do): Block III's COFF is read
        # only when ICNTIN > 0, which OASS's GETOPT never sets
        # (unoass21.f:596, :557-698), and under every option OASS supports
        # the binary recomputes NWVNO from the .rhs file's own wavenumber
        # step (unoass21.f:209-215; a deck asking for 2048 ran 2778); "NW,
        # IC1, IC2 are only used for kernel computations" (oass.tex:238-243).
        # They are set on the mean-field model instead.
        self._check_knobs()
        self.roughness_spectrum = str(roughness_spectrum).lower()

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        # install.sh's ``POSSIBLE_OASES_BINS`` loop copies the raw executable;
        # its compatibility symlinks cover oasn/oasp/oasr/oast but none for oass.
        self._exe = self._resolve_executable(
            executable, lambda: self._find_binary('oass'),
        )

    def _build_mean_field(self) -> 'OAST':
        """The mean-field producer: the supplied OAST with its own
        ``verbose`` and ``timeout``, or ``OAST(options='K J T s')`` with
        this run's; either with OASS's ``cleanup``: its files share OASS's
        directory, which OASS's ``cleanup`` keeps or wipes, so the mean
        field's result records their paths exactly when they survive."""
        if self.mean_field is None:
            return OAST(options=_OASS_MEAN_FIELD_OPTIONS, verbose=self.verbose,
                        timeout=self.timeout, cleanup=self.cleanup)
        return self.mean_field.copy(cleanup=self.cleanup)

    def _check_knobs(self) -> None:
        """Refuse a constructor knob no run could use
        (:func:`_check_oass_knobs`, and a pinned phase-speed window that is not
        positive or not ordered,
        :func:`~uacpy.models._window.check_pinned_window`). Run at
        construction and again by every
        run (:meth:`_validate_engine`), since the attributes can be
        reassigned in between."""
        check_pinned_window('OASS', c_low=self.c_low, c_high=self.c_high)
        _check_oass_knobs(
            correlation_length=self.correlation_length,
            spectral_exponent=self.spectral_exponent,
            roughness_spectrum=self.roughness_spectrum,
            mean_field=self.mean_field, options=self.options,
            multiple_scattering=self.multiple_scattering)

    def _validate_engine(self, env, source, receiver, run_mode, *,
                         request=None) -> None:
        """Stage 2: the family's boundary refusals
        (:meth:`OASES._validate_engine`), then an option line whose output
        uacpy cannot read back, a
        frequency sweep (:func:`_require_single_frequency`), a smooth
        scattering interface (:func:`_oass_interface`), a producer
        that cannot feed OASS and a receiver with fewer than the two ranges
        Block VIII divides by or ranges that are not the equispaced axis it
        evaluates (:func:`~uacpy.io.oases_writer.
        refuse_non_equispaced_oass_ranges`) — all before a mean-field run is
        spent."""
        super()._validate_engine(env, source, receiver, run_mode,
                                 request=request)
        options = _oass_options(self.options, run_mode,
                                self.roughness_spectrum,
                                self.multiple_scattering)
        _reject_unreadable_oass_options(options, run_mode)
        _reject_plane_geometry_letter_on_a_point_source(
            'OASS', options, self.options, source,
            'unoass21.f:654-656')
        _require_single_frequency(source)
        _oass_interface(env, self.interface)
        _reject_unusable_mean_field_options(self.mean_field)
        if set(options) & _OASS_REVERB_OPTIONS:
            _check_oass_range_count(
                np.atleast_1d(np.asarray(receiver.ranges)).size)
            refuse_non_equispaced_oass_ranges('OASS', receiver.ranges)

    def _resolve_engine_settings(self, env, source, receiver, settings, *,
                                 given_env, request=None) -> 'OASSSettings':
        """Stage 3: the mean field's settings, from its producer, and
        :func:`_oass_settings` over them for this model's knobs."""
        mean_settings, projection_notices = (
            self._build_mean_field()._producer_settings(
                env, source.at_depth(0), receiver, RunMode.COHERENT_TL))
        return _oass_settings(
            type(self).__name__, mean_settings, projection_notices,
            options=_oass_options(self.options, settings.mode,
                                  self.roughness_spectrum,
                                  self.multiple_scattering),
            interface=self.interface, c_low_pinned=self.c_low,
            c_high=self.c_high,
            correlation_length=self.correlation_length,
            spectral_exponent=self.spectral_exponent,
            rms_roughness=self.rms_roughness,
            receiver_gains_dB=self.receiver_gains_dB, env=env, source=source)

    #: Why a missing .rhs means what it means, for the refusal.
    _RHS_HINT = ("The producer ran but left no boundary operators; check "
                 "that the scattering interface is rough.")

    def _producer_for(self, settings) -> 'OAST':
        """The mean-field producer (:meth:`_build_mean_field`)."""
        return self._build_mean_field()

    def _mean_field_source(self, source):
        """The Source the producer runs on: the call's at its first
        depth."""
        return source.at_depth(0)

    # ── stage 4: oass2 on the mean field's .rhs ────────────────────────

    def _write_scattering_deck(self, inputs) -> Path:
        """Stage 4, launch 1: the OASS deck, written by
        :func:`~uacpy.io.oases_writer.write_oass_input` from the
        settings."""
        engine = inputs.settings.engine
        deck = inputs.work_dir / f'{_OASS_BASE_NAME}.dat'
        self._log(f"Writing OASS input file: {deck} "
                  f"(options={engine.options})")
        write_oass_input(
            filepath=deck,
            env=inputs.env,
            source=inputs.source,
            receiver=inputs.receiver,
            options=engine.options,
            interface=engine.interface,
            correlation_length=engine.correlation_length,
            spectral_exponent=engine.spectral_exponent,
            rms_roughness=engine.rms_roughness,
            c_low=engine.c_low,
            c_high=engine.c_high,
            receiver_gains_dB=engine.receiver_gains_dB,
        )
        return deck

    def _launch_scattering(self, inputs, deck: Path) -> None:
        """Stage 4, launch 1: ``oass2`` on the deck, its FOR045 pointed at
        the mean field's ``.rhs`` (named after that run's stem, see
        :meth:`OASES._execute`), and a run past OASES' wavenumber bound
        refused."""
        proc = self._execute(deck.stem, inputs.work_dir, extra_env={
            'FOR045': f'{inputs.earlier[0].stem}.045'})
        inputs.prepared.process = proc
        nwvno = _oass_wavenumber_echo(proc)
        # unoass21.f:209 re-derives NWVNO inside IF(REVERB) and the
        # MIN0(NWVNO,NP) clamp at :225 sits in the ELSE branch only, so a
        # reverberation run can integrate past the arrays and write a
        # full-size garbage table at exit 0. The echo is the count the
        # binary actually used — bound it here.
        if nwvno is not None and nwvno > _OASES_MAX_WAVENUMBERS:
            raise ConfigurationError(
                f"OASS sampled {nwvno} wavenumbers, past OASES' "
                f"array bound NP = {_OASES_MAX_WAVENUMBERS} "
                f"(oases/src/compar.f:37-38); the reverberation "
                f"branch has no clamp (unoass21.f:209 vs :225), so "
                f"the table it wrote is not the scattered field.",
                remediation="Raise c_low (WKMAX = 2π·f/c_low sets "
                            "the count), lower the frequency, or "
                            "shorten receiver.ranges.",
            )

    def _read_scattering_output(self, inputs, deck: Path):
        """Stage 4, launch 1: the product the run mode asks for, as the io
        reader returns it — the reverberation table (FOR020 ``.plt``) or the
        covariance (``.xsm``) — with the wavenumber count oass2 echoed."""
        proc = inputs.prepared.process
        if inputs.settings.mode == RunMode.COVARIANCE:
            cov_path = require_output(
                self.model_name, [inputs.work_dir / f'{deck.stem}.xsm'],
                what='a covariance file', process=proc,
            )
            self._log(f"Reading OASS covariance file: {cov_path}")
            product = read_oasn_covariance(
                cov_path, receiver_depths=inputs.receiver.depths)
        else:
            plt_path = require_output(
                self.model_name, [inputs.work_dir / f'{deck.stem}.plt'],
                what='a reverberation table (FOR020 .plt)', process=proc,
            )
            self._log(f"Reading OASS output: {plt_path}")
            product = parse_oast_tl(filepath=plt_path,
                                   receiver_depths=inputs.receiver.depths)
        return _oass_wavenumber_echo(proc), product

    def _check_mean_field_header(self, inputs, header) -> None:
        """Refuse a ``.rhs`` written at another frequency than this run's,
        read off the bytes the consumer will see.

        A mismatch is not an abort in the binary: unoass21.f:128-135 prints
        '>>> WARNING: Frequency sampling mismatch' to stdout and then
        executes ``freq=fr1_in_file``, so the run silently moves to the
        .rhs's frequency while uacpy would label the result with the deck's.
        (The '>>> FREQUENCY MISMATCH. ABORTING <<<' test at :143 compares the
        already-replaced freq against record 2's, which the same producer
        wrote, so it never fires.)

        The first record is ``NX, FR1, FR2, DT`` for every producer, but a
        single-frequency OAST writes ``nfreq`` into the NX slot and
        ``1/(nfreq*dlfreq)`` into DT (unoast31.f:164-165) — the same four
        fields OASS reads back as nx_in_file/fr1_in_file (unoass21.f:126).
        """
        frequency = inputs.settings.engine.frequency
        nfreq = header.n_time_samples
        rhs_frequency = header.freq_min
        if nfreq != 1 or abs(rhs_frequency - frequency) > 1e-3 * frequency:
            raise ConfigurationError(
                f"OASS: the mean field wrote {nfreq} frequency block(s) at "
                f"{rhs_frequency:g} Hz but OASS runs at {frequency:g} Hz. "
                f"OASS pins nfreq=1 (unoass21.f:123) and replaces its own "
                f"frequency with the .rhs's (:135) after a stdout-only "
                f"warning, so the result would carry the wrong frequency.",
                remediation=("Give the mean-field model the same single-"
                             "frequency Source this run uses."),
            )

    def _to_result(self, inputs, deck, raw) -> Result:
        """Stage 5: the reverberation :class:`Field` or the
        :class:`Covariance`, with the mean field's result as
        ``components['mean_field']``."""
        mean, (nwvno, product) = raw
        engine = inputs.settings.engine
        # Result keywords shared by both products; each builder adds its
        # own and routes the lot through _result_kwargs.
        extras = dict(
            frequencies=engine.frequency,
            components={'mean_field': mean.result},
        )
        if nwvno is not None:
            extras['n_integrated_wavenumbers'] = nwvno
        if inputs.settings.mode == RunMode.COVARIANCE:
            result = self._covariance_result(inputs, deck[1], product,
                                             extras)
        else:
            result = self._reverberation_field(inputs, deck[1], product,
                                               extras)
        # The .rhs is named after the producer's stem, so it cannot go
        # through _attach_output_paths — but it follows the same rule:
        # paths only when the work dir survives (DOCUMENTATION.md §8).
        if not self.cleanup:
            rhs = inputs.work_dir / f'{mean.stem}.045'
            if rhs.exists():
                result.metadata['rhs_file'] = str(rhs)

        self._log("OASS simulation complete")
        return result

    def _reverberation_field(self, inputs, deck: Path, oass_out: dict,
                             extras: dict) -> Field:
        """Build the reverberation :class:`Field` from the ``.plt`` curves.

        Option ``'r'`` sets ``PLTL`` (``unoass21.f:607-609``), whose branch
        calls ``REVINT`` and then ``PLTLOS`` (``unoass21.f:381,402``) — the
        routine OAST uses for TL — and ``oasfun22.f:330`` tags the curves
        ``…TLRAN``, so ``read_oast_tl`` reads them unchanged. What they carry
        is ``-10·log10 E[|p_scat|²]``: ``REVINT`` accumulates
        ``cff(index,1) += facin·cff·conjg(cff)`` (``oassun26.f:843-844``), an
        intensity, and its "CONVERT TO dB" block (``oassun26.f:853-858``)
        applies ``CVMAGS → VCLIP → VALG10 → VSMUL(-5E0)`` into ``CFFs``, which
        ``unoass21.f:38`` equivalences to the ``XS`` that ``PLTLOS`` plots.
        The extra square from ``CVMAGS`` is what makes the ``-5`` a ``-10`` on
        the intensity, and the leading minus makes it a **loss, not a level**:
        a larger value is a WEAKER scattered field, the same direction as
        transmission loss, with ``RL = SL -`` this. That direction is not a
        convention uacpy chose — it is what ``VSMUL(…, -5E0, …)`` at
        ``oassun26.f:858`` writes. It is why the Field is tagged
        ``kind='reverberation'`` rather than left as pressure in dB: it shares
        TL's representation but is a different quantity, so the two do not
        compare.

        ``REVRAN``'s own dB block (``oassun26.f:633-638``) is the same
        arithmetic on a *cross-range covariance*
        (``cff(inr+iof,2)·conjg(cff(index+iof,2))``, ``:624-625``) and feeds
        the ``CCONTU`` contour branch (``unoass21.f:355-379``), which is
        enabled by the **capital** option ``'C'`` (``unoass21.f:626-628``) —
        lowercase ``'c'`` sets ``ICONTU``, the depth-integrand contours, which
        is a third thing again. Neither feeds the ``.plt`` curves read here.
        """
        source = inputs.source
        data = oass_out['tl']
        native_depths = oass_out['depths']
        native_ranges = oass_out['ranges']
        kw = result_kwargs(
            self.model_name, self.provenance, source,
            backend='oass',
            oass_quantity='reverberation_loss_dB',
            **extras,
        )
        native = Field(
            data=data,
            coords={'depth': native_depths, 'range': native_ranges},
            kind='reverberation',
            **kw,
        )

        # Block VIII is RMIN RMAX NR and the binary rebuilds the axis as
        # R0 + (i-1)*RSTEP (unoass21.f:267-271); stage 2 refused a receiver
        # whose ranges are not that axis, so the level is returned on the
        # ranges OASS evaluated.
        result = mask_source_axis(self.model_name, native, source)
        attach_output_paths(
            result, inputs.work_dir, deck.stem,
            primary_files=(('plt_file', '.plt'),),
            cleanup=self.cleanup,
        )
        return result

    def _covariance_result(self, inputs, deck: Path, covariance,
                           extras: dict) -> Covariance:
        """Stamp the :class:`Covariance` read from the ``.xsm``.

        ``PUTXSM`` (``oasmun21_bin.f:335-346``) is the writer OASN uses, so
        ``read_oasn_covariance`` reads it unchanged — verified against the
        binary's own normalised ASCII dump on unit 24 to 6 decimals.
        FOR016 is always set, so the matrix can only appear under ``.xsm``.
        """
        extras = dict(extras)
        # E[p_i p_j*] of the scattered pressure for a source of level 1 Pa
        # ("in Pa for source level 1 Pa", oass.tex:216-218), each pair
        # scaled by the receivers' linear gains (oassun26.f:1054-1058): the
        # variance of a CW field, Pa², not a spectral density.
        covariance.unit = 'Pa²'
        result = self._stamp_file_result(
            covariance, inputs.source, backend='oass',
            frequencies=extras.pop('frequencies', None), **extras)
        attach_output_paths(
            result, inputs.work_dir, deck.stem,
            primary_files=(('xsm_file', '.xsm'), ('cor_file', '.cor')),
            cleanup=self.cleanup,
        )
        return result

    # Mirrors third_party/oases/bin/oass (oass.tex:438-448): covariance on
    # unit 16, the normalised-correlation ASCII dump on 24 (oassun26.f:1068),
    # and the unit-26 echo of the array INPRCV opens at oasnun22.f:37.
    # FOR045 is set per-run in _execute — it names the mean-field deck's stem,
    # not this one's.
    _FOR_FILES = {
        'FOR016': 'xsm',
        'FOR024': 'cor',
        'FOR026': 'chk',
    }
    # Every unit above is env-assigned, so the outputs always land on the
    # base_name suffixes and no bare fort.NN is written under this stem
    # (the mean-field OAST's fort.46 belongs to that producer's own list).
    _OUTPUT_SUFFIXES = ('.xsm', '.cor', '.plt', '.plp', '.chk')
