"""
OASES Input File Writers

This module provides functions for writing input files for OASES models:
- OAST: Transmission loss module (wavenumber integration)
- OASN: Noise/covariance module (also used for normal mode computation)
- OASR: Plane-wave reflection-coefficient module
- OASP: Pulse / broadband transfer-function module

OASES (Ocean Acoustics and Seismic Exploration Synthesis) was developed by
Henrik Schmidt at MIT.

References:
    Schmidt, H. OASES Version 2.1 User Guide and Reference Manual (bundled
    under ``third_party/oases``). The public OASES release is 3.1 but this
    distribution ships with the 2.1 source tree — per the bundled README.
"""

import warnings
from pathlib import Path
from typing import Callable, Optional, TextIO, Tuple, Union
import numpy as np

from uacpy.core.environment import BoundaryProperties, Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.exceptions import ConfigurationError
from uacpy.io.units import m_to_km
from uacpy.io.oalib_writer import writable_layers


def _oases_option_chars(options: str) -> set:
    """Significant option characters of an OASES option string.

    OASES's GETOPT reads the option line character by character
    (``CHARACTER*1 OPT(40)``) — whitespace is insignificant and
    ``'NJd'`` enables ``d`` exactly like ``'N J d'``. Every gate in
    this writer must therefore test character membership, never
    whitespace-split tokens or raw substrings.
    """
    return set(str(options)) - set(' \t\n')


def _write_oases_header(
    f: TextIO, env: Environment, options: str, fallback_title: str,
) -> None:
    """Write OASES Block I (title) + Block II (options)."""
    title = env.name if env.name else fallback_title
    f.write(f"{title}\n")
    f.write(f"{options}\n")


def _warn_volume_attenuation_ignored(
    env: 'Environment', *, lossless_water: bool = False,
) -> None:
    """OASES does not consume ``env.absorption``. Unlike the AT family /
    RAM (which emit the chosen Thorp / Francois-Garrison / Biological /
    Constant water attenuation), OAST/OASN/OASP fall through to PINIT2's
    empirical Skretting-Leroy substitution for any AC=0 fluid layer
    (oaseun31.f:1516-1521), so the water column is still attenuated — but
    the user's chosen formula is not propagated. Warn once per run so the
    choice isn't silently dropped.

    ``lossless_water`` is OASR's case: unoasr21.f:95-97 overwrites a zero
    AC on layer 1 with 1e-8 before PINIT runs, which defeats the
    ``V(I,4).LE.0`` test and leaves the water halfspace lossless.

    (The OASES Block II option letters ``T`` / ``F`` / ``B`` already carry
    sub-model-specific meanings, so the Acoustics-Toolbox ``TopOpt``
    absorption codes cannot be injected into the options string.)
    """
    if getattr(env, 'absorption', None) is None:
        return
    kind = type(env.absorption).__name__
    if lossless_water:
        warnings.warn(
            f"OASR ignores env.absorption ({kind}): its water halfspace is "
            f"lossless by construction — unoasr21.f:95-97 rewrites a zero AC "
            f"as 1e-8, which suppresses the empirical substitution at "
            f"oaseun31.f:1516-1521. Volume absorption does not enter a "
            f"plane-wave interface reflection coefficient; apply it along "
            f"the path in the propagation model instead.",
            UserWarning, stacklevel=3,
        )
        return
    warnings.warn(
        f"OASES ignores env.absorption "
        f"({kind}): it applies its own internal "
        f"Skretting-Leroy water attenuation to AC=0 water layers "
        f"(oaseun31.f:1516-1521), so the "
        f"chosen seawater-absorption formula is not propagated. Use "
        f"Scooter for a wavenumber-integration / FFP model that honours "
        f"env.absorption, or accept OASES's built-in attenuation.",
        UserWarning, stacklevel=3,
    )


def _extract_bottom_props(bottom: BoundaryProperties) -> dict:
    """Pull geoacoustic values from a ``BoundaryProperties`` for OAS[TNPR] writers."""
    return dict(
        rho=bottom.density,
        c_p=bottom.sound_speed,
        c_s=bottom.shear_speed,
        alpha_p=bottom.attenuation,
        alpha_s=bottom.shear_attenuation,
    )


def _emit_bottom_layers(
    f: TextIO,
    env: Environment,
    water_depth: float,
    fallback_c_p: float,
    fallback_c_s: float,
    fallback_alpha_p: float,
    fallback_alpha_s: float,
    fallback_rho: float,
    *,
    extra_columns: int = 0,
    suffix_fn: Optional[Callable[[int], str]] = None,
    iface_start: int = 0,
) -> None:
    """Emit sediment layers + bottom halfspace to an OASES input file.

    Writes one interface line per SedimentLayer when ``env.bottom`` is
    layered, followed by the halfspace at the correct depth. Falls back to a
    single halfspace line for a pure half-space column.

    OASES interface format: D CC CS AC AS RO RG [IG]  (extra ``IG`` appended
    when ``extra_columns`` > 0 — required by some OASP/OASN writers).

    Parameters
    ----------
    suffix_fn : callable, optional
        Maps interface index → trailing string (e.g. OASR roughness suffix
        ``" -0.5000 100 3.5"``). When given, overrides ``extra_columns``.
        Index numbering starts at ``iface_start`` and increments per emitted
        line (sediment layer + halfspace).
    iface_start : int, optional
        Index of the first emitted interface (default 0). OASR uses
        ``iface_start=1`` because its layer-1 (water) is interface 0.
    """
    if suffix_fn is None:
        static_trail = (' 0' * (1 + extra_columns))  # RG [IG [extra...]]
        def suffix_fn(_i): return static_trail  # noqa: E731  -- static-suffix shortcut

    iface = iface_start
    if env.has_layered_bottom:
        lb = env.bottom.columns[0]
        current_depth = water_depth
        for layer in writable_layers(lb):
            layer_as = getattr(layer, 'shear_attenuation', 0.0)
            f.write(f"{current_depth:.2f} {layer.sound_speed:.2f} "
                    f"{layer.shear_speed:.2f} {layer.attenuation:.3f} "
                    f"{layer_as:.3f} {layer.density:.2f}{suffix_fn(iface)}\n")
            current_depth += layer.thickness
            iface += 1
        # Deepest halfspace below all sediment layers.
        hs = getattr(lb, 'halfspace', None)
        if hs is not None:
            # Plain getattr defaults only — ``or fallback`` would turn a
            # legitimate 0.0 (e.g. fluid halfspace shear) into the fallback.
            c_p = getattr(hs, 'sound_speed', fallback_c_p)
            c_s = getattr(hs, 'shear_speed', fallback_c_s)
            alpha_p = getattr(hs, 'attenuation', fallback_alpha_p)
            alpha_s = getattr(hs, 'shear_attenuation', fallback_alpha_s)
            rho = getattr(hs, 'density', fallback_rho)
        else:
            c_p, c_s, alpha_p, alpha_s, rho = (
                fallback_c_p, fallback_c_s,
                fallback_alpha_p, fallback_alpha_s, fallback_rho,
            )
        f.write(f"{current_depth:.2f} {c_p:.2f} {c_s:.2f} "
                f"{alpha_p:.3f} {alpha_s:.3f} {rho:.2f}{suffix_fn(iface)}\n")
    else:
        f.write(f"{water_depth:.2f} {fallback_c_p:.2f} {fallback_c_s:.2f} "
                f"{fallback_alpha_p:.3f} {fallback_alpha_s:.3f} "
                f"{fallback_rho:.2f}{suffix_fn(iface)}\n")


def _resolve_freq_sweep(
    source: Source, freq_fallback: float
) -> Tuple[float, float, int]:
    """Resolve the (freq_min, freq_max, nfreq) sweep tuple from a Source.

    Multi-frequency sources advertise the actual sweep bounds; single-freq
    sources collapse to ``(freq, freq, 1)``. The fallback covers the rare
    case where ``source.frequencies`` is empty (callers pass the model's
    canonical centre frequency).
    """
    freqs_arr = np.atleast_1d(source.frequencies)
    if len(freqs_arr) > 1:
        return float(freqs_arr.min()), float(freqs_arr.max()), int(len(freqs_arr))
    return freq_fallback, freq_fallback, 1


def _emit_oases_freq_line(
    f: TextIO,
    freq_min: float,
    freq_max: float,
    nfreq: int,
    *,
    integration_offset: float,
    doppler: bool = False,
    vrec: Optional[float] = None,
) -> None:
    """Write the OASES Block-III frequency-sweep line.

    Layout: ``FREQ1 FREQ2 NFREQ COFF [VREC]``.  The 5-token Doppler form
    fires only when the lowercase ``'d'`` option is enabled (OAST). OASN
    and OASR always emit the 4-token form.
    """
    if doppler:
        vrec_val = vrec if vrec is not None else 0.0
        f.write(
            f"{freq_min:.1f} {freq_max:.1f} {nfreq} "
            f"{integration_offset} {vrec_val}\n"
        )
    else:
        f.write(
            f"{freq_min:.1f} {freq_max:.1f} {nfreq} {integration_offset}\n"
        )


def _oases_wavenumber_bounds(
    ssp_data: np.ndarray,
    *,
    ssp_factor: float = 0.9,
    cmax: float = 1e8,
) -> Tuple[float, float]:
    """Return the (cmin, cmax) wavenumber-integration bounds.

    ``cmin = min(SSP) * ssp_factor`` (0.9 for OAST/OASP, 0.95 for the OASN
    noise blocks). ``cmax`` is the "no upper limit" sentinel: OAST uses 1e8,
    OASP 1e9.
    """
    cmin = float(ssp_data[:, 1].min()) * ssp_factor
    return cmin, float(cmax)


def _count_bottom_layers(env: Environment) -> int:
    """Number of sediment layers (not counting halfspace) when env.bottom is layered.

    Sub-resolution layers are excluded — see ``writable_layers`` — so the count
    matches what the writer actually emits."""
    if env.has_layered_bottom:
        return len(writable_layers(env.bottom))
    return 0


# OASES' own layer limit, `parameter (NLA = 1001)` in
# third_party/oases/src/compar.f:23, enforced against NL (the deck's total
# layer count, upper halfspace + water + sediments + bottom halfspace) at
# oaseun31.f:44 with '*** TOO MANY LAYERS ***'.
_OASES_MAX_LAYERS = 1001

# OASES' wavenumber-array bound, `NPEXP = 16, NP = 2**NPEXP` in
# third_party/oases/src/compar.f:37-38. OAST stops above it
# (unoast31.f:459 '>>> TOO MANY WAVENUMBERS <<<'); OASP has no such test, so
# an overrun there corrupts the transfer function instead of aborting.
_OASES_MAX_WAVENUMBERS = 65536

# OASES' receiver-depth limit, `parameter (NRD = 501)` in
# third_party/oases/src/compar.f:35 (and the twin `NRMAX = 501` at :52).
# INREC stops on `iabs(ir).gt.nrd` (oaseun31.f:1172); OASN checks the same
# bound twice, on NRCV in INPRCV (oasnun22.f:32-35) and on IR afterwards
# (unoasn22.f:163-165). All three are hard STOPs with an empty output file.
_OASES_MAX_RECEIVER_DEPTHS = 501


def _check_receiver_depth_count(n_depths: int) -> None:
    """Raise when a receiver array overruns OASES' NRD arrays."""
    if n_depths <= _OASES_MAX_RECEIVER_DEPTHS:
        return
    raise ConfigurationError(
        f"receiver.depths has {n_depths} entries but OASES resolves at most "
        f"{_OASES_MAX_RECEIVER_DEPTHS} receiver depths "
        f"(oases/src/compar.f:35 `parameter (NRD = "
        f"{_OASES_MAX_RECEIVER_DEPTHS})`); INREC stops with "
        f"'>>> Too many receiver depths <<<' (oaseun31.f:1172) and writes no "
        f"output.",
        remediation=(f"Decimate receiver.depths to at most "
                     f"{_OASES_MAX_RECEIVER_DEPTHS} entries, or split the "
                     f"array across several runs."),
    )


_OASN_NOISE_SAMPLES = (400, 400, 100)   # NW*C NW*D NW*E manual defaults


def _noise_cmin(kwargs, ssp_data) -> float:
    """Slow phase-speed edge for an OASN noise block: the model's ``c_low``
    when pinned, else the manual's 0.95*c_min."""
    c_low = kwargs.get('c_low')
    if c_low:
        return float(c_low)
    return float(ssp_data[:, 1].min()) * 0.95


def _noise_cmax(kwargs) -> float:
    """Fast phase-speed edge: the model's ``c_high`` when pinned, else the
    manual's 1E8 ("no upper limit")."""
    c_high = kwargs.get('c_high')
    return float(c_high) if c_high else 1.0e8


def _noise_nw(kwargs) -> str:
    """``NW*C NW*D NW*E`` wavenumber-sample counts for a noise block.

    Unlike the replica and discrete-source blocks, the surface- and deep-noise
    blocks have no automatic-sampling branch: NOIPAR reads three explicit counts
    and sums them into ``NWVNON``/``NWVNOP`` (oasnun22.f:312, :358). A negative
    count would make that total negative and the block would contribute nothing,
    so ``nw_samples <= 0`` falls back to the manual's counts.
    """
    nw = kwargs.get('nw_samples')
    if nw and int(nw) > 0:
        n = int(nw)
        return f"{n} {n} {max(1, n // 4)}"
    return " ".join(str(v) for v in _OASN_NOISE_SAMPLES)


def _oases_nw_line(nw_samples, icut2_auto: int) -> str:
    """``NW ICUT1 ICUT2`` wavenumber-sampling line.

    ``NW < 0`` selects OASES's automatic sampling — AUTSMN then recomputes
    ICUT1/ICUT2 (unoasn22.f:240-241, oasnun22.f:432) so the trailing values
    are inert. A pinned ``NW`` is clamped by ``ICUT2 = MIN0(NW, ICUT2)``
    (oast.tex:73-75), so emit ``ICUT2 = NW`` to keep the whole spectrum.
    """
    nw = int(nw_samples) if nw_samples else -1
    if nw <= 0:
        return f"-1 1 {icut2_auto}"
    return f"{nw} 1 {nw}"


#: Keys each OASES writer's deck actually reads. Every writer takes
#: ``**kwargs``, so an unread key would otherwise be dropped without a trace and
#: the run would quietly use the default.
_OAST_KWARGS = frozenset({
    'integration_offset', 'nw_samples', 'plot_rmin', 'plot_rmax', 'vrec',
    'dip_angle',
})
_OASN_KWARGS = frozenset({
    'surface_noise_level', 'white_noise_level', 'deep_noise_level',
    'deep_source_depth', 'discrete_sources',
    'integration_offset', 'offdb', 'nw_samples', 'c_low', 'c_high',
    'cmins_discrete', 'cmaxs_discrete', 'cmins_replica', 'cmaxs_replica',
    'replica_zmin', 'replica_zmax', 'replica_nz',
    'replica_xmin', 'replica_xmax', 'replica_nx',
    'replica_ymin', 'replica_ymax', 'replica_ny',
})
#: Keys a ``discrete_sources`` entry may carry — the four fields NOIPAR reads
#: (oasnun22.f:380 ``READ(1,*) ZDN(I),XDN(I),YDN(I),DNLEVDB(I)``).
_OASN_DISCRETE_SOURCE_KEYS = frozenset({'depth', 'x', 'y', 'level'})


def _check_oasn_noise_level(name: str, level, *, dead_band: float = 0.0) -> None:
    """Raise on a source level OASN would read as a spectrum-file unit number.

    NOIPAR overloads the sign: ``>= 0`` is a level in dB, a negative value is
    minus the Fortran unit number of a source-spectrum file it opens with
    OPFILR and stops on if absent (oasnun22.f:383-390 for a discrete source,
    oasnun22.f:191-193 for the surface). This writer emits no such file, so
    such a level can only abort the run.

    ``dead_band`` is the surface level's extra step: ``abs(SNLEVDB).LT.0.01``
    switches the source off before the sign test runs (oasnun22.f:183), so a
    tiny negative value is simply "off". The discrete-source level has no
    such band (oasnun22.f:381).
    """
    value = float(level)
    if value >= 0.0 or abs(value) < dead_band:
        return
    raise ConfigurationError(
        f"write_oasn_input: {name}={value:g} is negative, which OASN "
        f"reads as minus the unit number of a source-spectrum file "
        f"(oasnun22.f:383-385) rather than a level in dB; it stops with "
        f"'>>>> ERROR: NO FILE NO. …' because this writer emits no such file.",
        remediation="Pass a non-negative level in dB (0 disables the source).",
    )


_OASP_KWARGS = frozenset({
    'center_frequency', 'freq_min', 'freq_max', 'freq_output_increment',
    'n_time_samples', 'time_step', 'range_start', 'range_step',
    'integration_offset', 'nw_samples', 'dip_angle',
})
_OASR_KWARGS = frozenset({
    'freq_min', 'freq_max', 'n_frequencies', 'freq_output_increment',
    'angle_min', 'angle_max', 'n_angles', 'angle_output_increment',
})


def _reject_unknown_kwargs(writer: str, kwargs: dict, known: frozenset) -> None:
    """Raise on a writer knob no block of this deck reads."""
    unknown = sorted(set(kwargs) - known)
    if unknown:
        raise ConfigurationError(
            f"{writer}: parameter(s) {unknown} are not read by this deck.",
            remediation=f"Drop them, or check they belong to this OASES "
                        f"program; it reads {sorted(known)}.",
        )


#: Option letters whose GETOPT flag makes the program read input none of these
#: writers produces — a whole deck block, extra tokens on a record the writer
#: does emit, or a separate auxiliary file. The read then runs off the end of
#: the deck, eats the block after it, or aborts on a missing unit, so the
#: failure surfaces as a bare Fortran error with no indication of which letter
#: caused it. Keyed by writer; each entry maps the letter to what the program
#: demands and the Fortran line that demands it. An empty entry records that
#: the program's option-gated reads are all covered.
_UNWRITTEN_OPTION_BLOCKS = {
    'write_oast_input': {
        'E': ('patch-scattering parameters (PCENTER INPATCH NFFT_X NFFT_Y '
              'SPLEN_X SPLEN_Y)', 'unoast31.f:299'),
        'l': ('an external source-array file on unit 2 — LS, then LS rows of '
              'depth/delay/strength', 'oaseun31.f:1074-1087'),
        'v': ('a source transfer-function file through RDSTRF',
              'unoast31.f:182'),
        't': ('a tabulated surface reflection coefficient on unit 23 (.trc)',
              'oaseun31.f:3727-3728'),
        'b': ('a tabulated bottom reflection coefficient on unit 23 (.trc)',
              'oaseun31.f:3757-3758'),
    },
    'write_oasp_input': {
        'd': ("the Doppler frequency line's ISTYP VSOU VREC tokens",
              'unoasp22.f:127'),
        'G': ('a dispersion-curve block (NMODES, then two axis rows)',
              'unoasp22.f:387-390'),
        'T': ('two extra receiver-line tokens ZREF and ANGLE for the tilted '
              'array', 'oaseun31.f:1157-1158'),
        'Z': ('two velocity-profile plot-axis rows', 'unoasp22.f:466-467'),
        'E': ('patch-scattering parameters (PCENTER INPATCH NFFT_X NFFT_Y '
              'SPLEN_X SPLEN_Y)', 'unoasp22.f:479'),
        'l': ('an external source-array file on unit 2 — LS, then LS rows of '
              'depth/delay/strength', 'oaseun31.f:1074-1087'),
        'v': ('a source transfer-function file through RDSTRF',
              'unoasp22.f:261'),
    },
    'write_oasn_input': {
        'Z': ('two velocity-profile plot-axis rows', 'unoasn22.f:322-323'),
        'z': ('two velocity-profile plot-axis rows', 'unoasn22.f:322-323'),
        't': ('a tabulated surface reflection coefficient on unit 23 (.trc)',
              'oaseun31.f:3727-3728'),
        'b': ('a tabulated bottom reflection coefficient on unit 23 (.trc)',
              'oaseun31.f:3757-3758'),
    },
    'write_oasr_input': {},
}


def _reject_unwritten_option_blocks(writer: str, options: str) -> None:
    """Raise on an option letter demanding input this writer never produces."""
    demanded = _UNWRITTEN_OPTION_BLOCKS[writer]
    used = sorted(_oases_option_chars(options) & set(demanded))
    if not used:
        return
    detail = '; '.join(f"{c!r} makes it read {demanded[c][0]} "
                       f"({demanded[c][1]})" for c in used)
    raise ConfigurationError(
        f"{writer}: option letter(s) {used} are not supported — {detail}, "
        f"and this writer produces no such input. The deck would fail inside "
        f"Fortran instead of here.",
        remediation="Drop the letter(s) from options=.",
    )


def _check_frequency_contours(writer: str, options: str, letter: str,
                              nfreq: int) -> None:
    """Raise on a frequency-contour option with a single-frequency sweep.

    ``unoast31.f:138-140`` (``FRCONT``, OAST option ``'o'``) and
    ``unoasr21.f:119-121`` (``CONTUR``, OASR option ``'C'``) both end at
    ``STOP '*** CONTOURS REQUIRE NRFR>1 … ***'`` when ``NFREQ <= 1`` — a
    character stop, so the binary exits 0 with no ``.prt`` to consult.
    """
    if letter not in _oases_option_chars(options) or int(nfreq) > 1:
        return
    raise ConfigurationError(
        f"{writer}: option {letter!r} plots contours against frequency, which "
        f"needs more than one frequency; the deck carries {int(nfreq)}. "
        f"The binary would stop with '*** CONTOURS REQUIRE NRFR>1 … ***'.",
        remediation=f"Give the Source a frequency array, or drop {letter!r} "
                    f"from options=.",
    )


def _reject_tau_p(writer: str, options: str) -> None:
    """Raise on ``'t'``, which silently redefines the deck's range axis.

    Lowercase ``'t'`` sets ``INTTYP=-1`` (``oases/src/unoasp22.f:1028-1029``)
    and ``:178-189`` then overwrites the deck's ``R0`` / ``RSPACE`` /
    ``NPLOTS`` with ``r0 = 1e3/cmaxin``, ``rspace = (1e3/cminin - r0)/(nwvno-1)``,
    ``nplots = nwvno`` — a slowness axis in s/m. The written range triple is
    discarded, so the ``.trf``'s second coordinate is no longer range and
    ``read_oasp_trf`` would label slowness as metres.
    """
    if 't' not in _oases_option_chars(options):
        return
    raise ConfigurationError(
        f"{writer}: option 't' (tau-p seismograms) replaces the receiver "
        f"range axis with slowness (oases/src/unoasp22.f:178-189), which "
        f"uacpy's Field has no coordinate for and would mislabel as range.",
        remediation="Drop 't' from options=.",
    )


# OAST Block VIII trailing pair. XAXIS is the plot axis length in cm, consumed
# only by OASES's own plotters. XINC is the TL-vs-depth range increment
# (unoast31.f:255, :705) — expressed as a curve count so it tracks the run's
# range span instead of a fixed 1 km.
_OAST_PLOT_AXIS_CM = 20
_OAST_TLDEP_CURVES = 20

#: OAST option letters that select an output parameter, i.e. set an ``IOUT``
#: entry in GETOPT (unoast31.f:981-1015). Their count drives how many times
#: OAST re-reads Blocks IX and XI.
_OAST_OUTPUT_PARAM_CHARS = frozenset('NVHRKS')


def _check_ssp_layer_count(
    ssp_data: np.ndarray, n_other_layers: int,
) -> np.ndarray:
    """Pass an SSP through, decimating only if it would overrun OASES' arrays.

    OASES bounds NL — the deck's *total* layer count — so the SSP budget is
    ``_OASES_MAX_LAYERS - n_other_layers``, where ``n_other_layers`` counts
    every non-SSP layer the deck emits (upper halfspace + sediment layers +
    bottom halfspace). It is required: a caller that omitted it would bound
    the SSP alone and hand OASES a deck one halfspace too deep.

    Below the budget nothing is touched — thinning a measured profile changes
    the ocean being modelled (a 201-point duct profile cut to 15 rows moved TL
    by 4.1 dB median against Kraken). Above it, decimate evenly and say so:
    the alternative is a deck OASES rejects outright.
    """
    max_rows = _OASES_MAX_LAYERS - int(n_other_layers)
    if max_rows < 1:
        raise ConfigurationError(
            f"The seabed stack alone needs {n_other_layers} of OASES' "
            f"{_OASES_MAX_LAYERS} layers (oases/src/compar.f:23 "
            f"`parameter (NLA = {_OASES_MAX_LAYERS})`), leaving no room for "
            f"the water column.",
            remediation="Merge sediment layers so the seabed stack fits.",
        )

    n_rows = len(ssp_data)
    if n_rows <= max_rows:
        return ssp_data

    # Keep the first and last rows so the layer interfaces still pin to the
    # surface and env.depth; thin the interior evenly.
    keep = np.unique(np.linspace(0, n_rows - 1, max_rows).astype(int))
    warnings.warn(
        f"env.ssp has {n_rows} samples but this deck can spend at most "
        f"{max_rows} of OASES' {_OASES_MAX_LAYERS} layers on the water column "
        f"(oases/src/compar.f:23 `parameter (NLA = {_OASES_MAX_LAYERS})`, "
        f"{n_other_layers} taken by the halfspaces and sediment stack); "
        f"decimated to {keep.size} evenly-spaced samples. Decimate env.ssp "
        f"yourself if you want to choose which features survive.",
        UserWarning, stacklevel=3,
    )
    return ssp_data[keep, :]


#: Compressional speed the ``'rigid'`` surface substitute emits (OASES has no
#: rigid upper halfspace; a high-impedance fluid is the closest stand-in).
_OASES_RIGID_SURFACE_CP = 4000.0


def _upper_halfspace_sound_speed(env: Environment) -> float:
    """Compressional speed OASES reads as ``V(1,2)`` for the upper halfspace.

    Mirrors :func:`_format_upper_halfspace` without emitting its warning, so
    gates that depend on the layer-1 speed can be evaluated up front.
    """
    surface = env.surface
    atype = str(getattr(surface, 'acoustic_type', 'vacuum')).lower()
    if 'vacuum' in atype:
        return 0.0
    if 'rigid' in atype:
        return _OASES_RIGID_SURFACE_CP
    return float(getattr(surface, 'sound_speed', 0.0) or 0.0)


def _format_upper_halfspace(env: Environment) -> str:
    """Format the OASES upper halfspace line from env.surface properties.

    OASES layer format: D CC CS AC AS RO RG [IG]
    For vacuum: all zeros.
    For elastic (ice): sound_speed, shear_speed, attenuation, density.
    """
    surface = env.surface
    acoustic_type = getattr(surface, 'acoustic_type', 'vacuum')
    if isinstance(acoustic_type, str):
        atype = acoustic_type.lower()
    else:
        atype = str(acoustic_type).lower()

    # OASES layer format: D CC CS AC AS RO RG IG (8 columns).
    if 'vacuum' in atype:
        return "0 0 0 0 0 0 0 0"

    c_p = getattr(surface, 'sound_speed', 0.0) or 0.0
    c_s = getattr(surface, 'shear_speed', 0.0) or 0.0
    alpha_p = getattr(surface, 'attenuation', 0.0) or 0.0
    alpha_s = getattr(surface, 'shear_attenuation', 0.0) or 0.0
    rho = getattr(surface, 'density', 0.0) or 0.0

    if 'rigid' in atype:
        warnings.warn(
            f"OASES does not natively support a rigid upper halfspace; "
            f"emitting a high-impedance fluid halfspace "
            f"(cp={_OASES_RIGID_SURFACE_CP:g}, rho=2.5) as a substitute. "
            f"Acoustic match is partial — pressure-release (vacuum) and "
            f"acoustic-half-space surfaces are the only physically exact "
            f"OASES top BCs.",
            UserWarning, stacklevel=3,
        )
        return f"0 {_OASES_RIGID_SURFACE_CP:.1f} 0.0 0.000 0.000 2.50 0 0"

    return (f"0 {c_p:.2f} {c_s:.2f} {alpha_p:.3f} "
            f"{alpha_s:.3f} {rho:.2f} 0 0")


def _receiver_block_lines(
    receiver: Receiver,
    *,
    trailing: str = '',
) -> list:
    """Emit OASES receiver-block text lines.

    OASES supports non-equidistant receiver depths via NR<0: a negative
    receiver count on the first line followed by the explicit depth list on
    the next line(s). See oast.tex:464-493 and oasp.tex:559-585.

    Returns
    -------
    list[str]
        Lines to write, each WITHOUT a trailing newline. Caller adds '\\n'.

    Parameters
    ----------
    receiver : Receiver
        Receiver object providing `.depths`.
    trailing : str
        Extra whitespace-separated tokens to append to the header line
        (e.g. ``' 1'`` for OAST's IR column). An empty string matches OASP.
    """
    depths = np.asarray(receiver.depths, dtype=float)
    n = len(depths)
    _check_receiver_depth_count(n)
    z_min = float(depths.min())
    z_max = float(depths.max())

    if n <= 1:
        return [f"{z_min:.2f} {z_max:.2f} {n}{trailing}"]

    diffs = np.diff(depths)
    # A grid is considered equidistant when all spacings match within 1 cm.
    uniform = np.allclose(diffs, diffs[0], atol=1e-2)
    if uniform:
        return [f"{z_min:.2f} {z_max:.2f} {n}{trailing}"]

    # Non-uniform: emit NR = -n, then individual depths on the next line.
    header = f"{z_min:.2f} {z_max:.2f} {-n}{trailing}"
    depth_line = ' '.join(f"{d:.2f}" for d in depths)
    return [header, depth_line]


def _source_block_line(
    src_depth: float,
    *,
    dip_angle: Optional[float] = None,
    linear_array: bool = False,
) -> str:
    """OASES Block-V source record for one source at ``src_depth``.

    INSRC picks the record shape from LINA and dip_sou, not from the manual's
    column table: ``SD NS DS AN IA FD DA`` with LINA=1 (oaseun31.f:1089),
    ``SD DA`` with dip_sou alone (oaseun31.f:1114), plain ``SD`` otherwise
    (oaseun31.f:1119). The dip angle therefore sits in a different column
    depending on the option letters, and the 7-token form fed to the 2-item
    read would hand OASES the NS column as the dip angle.

    Shared by the OAST/OASP writers so the source-block format lives in one
    place, like ``_receiver_block_lines``.
    """
    if linear_array:
        return f"{src_depth:.2f} 1 0 0 1 0 {dip_angle or 0.0:g}"
    if dip_angle is not None:
        return f"{src_depth:.2f} {dip_angle:g}"
    return f"{src_depth:.2f} 1 0 0 1 0 0"


#: Option letters that put INSRC on a source-record shape other than the
#: plain ``SD`` form: '4' selects the dip-slip moment source (dip_sou,
#: unoast31.f:1117-1122 / unoasp22.f:993-995) and 'L' the internal vertical
#: array (LINA=1, unoast31.f:1051-1054 / unoasp22.f:929-932). 'l' and 'v'
#: also set LINA but are rejected outright — they need a source file this
#: writer does not produce.
_OASES_DIP_SOURCE_CHAR = '4'
_OASES_LINEAR_ARRAY_CHAR = 'L'


def _resolve_source_record(
    writer: str, options: str, dip_angle: Optional[float],
) -> dict:
    """Keyword arguments for :func:`_source_block_line` given the options."""
    opt_chars = _oases_option_chars(options)
    dip_on = _OASES_DIP_SOURCE_CHAR in opt_chars
    if dip_angle is not None and not dip_on:
        raise ConfigurationError(
            f"{writer}: dip_angle={dip_angle} is only read when the option "
            f"string selects the dip-slip source; OASES takes it from the "
            f"source record solely under dip_sou (oaseun31.f:1089, :1114).",
            remediation=f"Add {_OASES_DIP_SOURCE_CHAR!r} to options=, or drop "
                        f"dip_angle.",
        )
    if not dip_on:
        return {}
    return {
        'dip_angle': 0.0 if dip_angle is None else float(dip_angle),
        'linear_array': _OASES_LINEAR_ARRAY_CHAR in opt_chars,
    }


def _check_nw_samples(writer: str, nw_samples, *, power_of_two: bool) -> None:
    """Validate a pinned wavenumber-sample count against OASES' NP array bound.

    A pinned ``NW`` above NP is silently clamped by ``NWVNO=MIN0(NWVNO,NP)``
    (unoast31.f:435, unoasp22.f:174), and OAST additionally rounds it up to
    the next power of two (unoast31.f:447-458). Both change the wavenumber
    step — and hence the range at which the FFT wraps — from what was asked
    for, so say so here rather than let the run answer a different question.
    """
    if nw_samples is None:
        return
    nw = int(nw_samples)
    if nw <= 0:                      # automatic sampling
        return
    if nw > _OASES_MAX_WAVENUMBERS:
        raise ConfigurationError(
            f"{writer}: nw_samples={nw} exceeds OASES' wavenumber array bound "
            f"NP = {_OASES_MAX_WAVENUMBERS} (oases/src/compar.f:37-38 "
            f"`NPEXP = 16, NP = 2**NPEXP`); OASES would clamp it to "
            f"{_OASES_MAX_WAVENUMBERS} and integrate on a coarser wavenumber "
            f"grid than requested.",
            remediation=(f"Pass nw_samples <= {_OASES_MAX_WAVENUMBERS}, or "
                         f"-1 for OASES' automatic sampling."),
        )
    if power_of_two and nw & (nw - 1):
        rounded = 1 << (nw - 1).bit_length()
        warnings.warn(
            f"{writer}: nw_samples={nw} is not a power of two; OAST rounds it "
            f"up to {rounded} (unoast31.f:447-458) and integrates on that "
            f"grid, so the wavenumber step — and the range at which the FFT "
            f"wraps — is not the one implied by {nw}.",
            UserWarning, stacklevel=3,
        )


def write_oast_input(
    filepath: Union[str, Path],
    env: Environment,
    source: Source,
    receiver: Receiver,
    options: Optional[str] = None,
    **kwargs
) -> None:
    """
    Write OAST (OASES Transmission Loss) input file

    OAST uses wavenumber integration with Direct Global Matrix solution.
    Supports elastic layers and seismo-acoustic propagation.

    Parameters
    ----------
    filepath : str or Path
        Output file path (typically .dat extension)
    env : Environment
        Ocean environment (must be range-independent)
    source : Source
        Acoustic source specification
    receiver : Receiver
        Receiver array specification
    options : str, optional
        OAST option string. If None, uses default 'N J T'
        Common options:
        - N: Normal stress (pressure)
        - J: Complex integration contour
        - T: Transmission loss vs range plot
        - C: Range-depth contour plot
        - I: Integrands plot (for debugging)
        - Z: Sound speed profile plot
    **kwargs : dict
        Additional parameters:
        - integration_offset : float
            Integration contour offset in dB/wavelength (default: 0)
        - nw_samples : int
            Number of wavenumber samples (default: -1 for automatic)
        - plot_rmin : float
            Minimum range for plots in **metres** (default: 0)
        - plot_rmax : float
            Maximum range for plots in **metres** (default: max receiver range)
        - dip_angle : float
            Fault dip angle in degrees for the ``'4'`` dip-slip moment
            source (default 0). Only read when ``'4'`` is in ``options``.

    Notes
    -----
    OAST input file has 12 blocks:
    I.   Title
    II.  Options (output control)
    III. Frequencies
    IV.  Environment (layers)
    V.   Sources
    VI.  Receivers
    VII. Wavenumber sampling
    VIII. Range axes (for plots)
    IX.  Transmission loss axes
    X.   Depth axes
    XI.  Contour levels
    XII. SVP axes (sound velocity profile)

    Examples
    --------
    >>> env = Environment(bathymetry=100, ssp=1500)
    >>> source = Source(depths=50, frequencies=100)
    >>> receiver = Receiver(depths=np.linspace(10,90,40),
    ...                     ranges=np.linspace(100,10000,100))
    >>> write_oast_input('test.dat', env, source, receiver)
    """
    _reject_unknown_kwargs('write_oast_input', kwargs, _OAST_KWARGS)
    filepath = Path(filepath)

    # Extract parameters
    freq = float(source.frequencies[0])
    depth = env.depth

    # Bottom properties
    bottom = env.bottom.halfspace_at(range=0.0)
    _bp = _extract_bottom_props(bottom)
    rho, c_p, c_s = _bp['rho'], _bp['c_p'], _bp['c_s']
    alpha_p, alpha_s = _bp['alpha_p'], _bp['alpha_s']

    # Sound speed profile — align to env.depth so the deepest sample sits
    # exactly at the seabed interface (OASES expects monotone depth + the
    # last entry to terminate the water column).
    ssp_data = env.ssp.extend_to(depth).to_pairs()

    # Source and receiver parameters (receiver depth bookkeeping now lives in
    # ``_receiver_block_lines`` which handles equidistant/explicit cases).
    src_depth = float(source.depths[0])
    r_max = float(receiver.ranges.max())

    # Optional parameters
    integration_offset = kwargs.get('integration_offset', 0)
    nw_samples = kwargs.get('nw_samples', -1)  # -1 = automatic
    plot_rmin = kwargs.get('plot_rmin', 0.0)            # metres (public API)
    plot_rmax = kwargs.get('plot_rmax', r_max)          # metres (public API)

    # Get reference sound speed for plot axes
    c_ref = float(ssp_data[0, 1])  # Sound speed at surface

    # Options string
    if options is None:
        options = 'N J T'  # Normal stress, complex contour, TL vs range
    _reject_unwritten_option_blocks('write_oast_input', options)
    _check_nw_samples('write_oast_input', nw_samples, power_of_two=True)
    source_record = _resolve_source_record(
        'write_oast_input', options, kwargs.get('dip_angle'))
    _warn_volume_attenuation_ignored(env)

    freq_min, freq_max, nfreq = _resolve_freq_sweep(source, freq)
    _check_frequency_contours('write_oast_input', options, 'o', nfreq)

    with open(filepath, 'w') as f:
        _write_oases_header(f, env, options, "OAST Simulation via UACPY")

        # Block III: Frequencies — FREQ1 FREQ2 NFREQ COFF [VREC].
        # unoast31.f:125-133: lowercase 'd' enables Doppler and demands the
        # 5th VREC token. Uppercase 'D' is TL-vs-depth (TLDEP,
        # unoast31.f:1086-1090) and stays in the 4-token form.
        doppler_on = 'd' in _oases_option_chars(options)
        _emit_oases_freq_line(
            f, freq_min, freq_max, nfreq,
            integration_offset=integration_offset,
            doppler=doppler_on,
            vrec=kwargs.get('vrec', 0.0),
        )

        # Block IV: Environment. OASES uses CS = -|c_next| in the
        # Airy-layer convention to encode a continuous SSP gradient
        # (oaseun31.f:160-192).
        # Count sediment layers from the seabed column
        n_sed_layers = _count_bottom_layers(env)
        # NL = upper halfspace + water + sediments + bottom halfspace.
        ssp_subset = _check_ssp_layer_count(ssp_data, 2 + n_sed_layers)

        # Special case: if isovelocity (all sound speeds are the same), write only 1 layer
        # Check if all sound speeds are equal within tolerance
        c_values = ssp_subset[:, 1]
        is_isovelocity = np.allclose(c_values, c_values[0], rtol=1e-6)

        if is_isovelocity:
            # Isovelocity: write only 1 water layer at surface
            n_layers = 3 + n_sed_layers  # vacuum + water + sed_layers + bottom
            f.write(f"{n_layers}\n")

            # Upper halfspace (from env.surface)
            f.write(f"{_format_upper_halfspace(env)}\n")

            # Single water layer
            f.write(f"0.00 {c_values[0]:.2f} 0 0.0 0 1.0 0 0\n")

            # Sediment layers + bottom halfspace (shared helper)
            _emit_bottom_layers(
                f, env, depth,
                c_p, c_s, alpha_p, alpha_s, rho,
                extra_columns=1,
            )
        else:
            # Non-isovelocity: write layers with negative CS field for gradients
            # OASES format: D CC CS means:
            #   - If CS >= 0: layer is isovelocity with shear speed CS
            #   - If CS < 0: layer has linear gradient from CC to -CS (abs value)
            n_water_layers = len(ssp_subset)
            n_layers = 1 + n_water_layers + n_sed_layers + 1
            f.write(f"{n_layers}\n")

            # Upper halfspace (from env.surface)
            f.write(f"{_format_upper_halfspace(env)}\n")

            # Water layers with gradient using negative CS field
            for i in range(len(ssp_subset)):
                d, c = ssp_subset[i]
                if i < len(ssp_subset) - 1:
                    # Intermediate layer: CS = -sound_speed_at_bottom_of_layer
                    c_bottom = ssp_subset[i + 1, 1]
                    cs = -abs(c_bottom)
                else:
                    # Last water layer: no gradient, connects to bottom
                    cs = 0
                f.write(f"{d:.2f} {c:.2f} {cs:.2f} 0.0 0 1.0 0 0\n")

            # Sediment layers + bottom halfspace (shared helper)
            _emit_bottom_layers(
                f, env, depth,
                c_p, c_s, alpha_p, alpha_s, rho,
                extra_columns=1,
            )

        # Block V: Sources — one source at src_depth, in the record shape
        # INSRC's LINA / dip_sou branch expects.
        f.write(_source_block_line(src_depth, **source_record) + '\n')

        # Block VI: Receivers
        # RD1 RD2 NR IR  (NR<0 signals explicit depth list — oast.tex:464-493).
        for line in _receiver_block_lines(receiver, trailing=' 1'):
            f.write(line + '\n')

        # Block VII: Wavenumber sampling — CMIN CMAX.
        cmin, cmax = _oases_wavenumber_bounds(ssp_data)
        f.write(f"{cmin:.1f} {cmax:.1e}\n")

        # NW IC1 IC2 — automatic sampling (NW=-1) ignores IC1/IC2 per
        # oast.tex:541-549. When NW>0 the constraint IC2 ≤ NW (oast.tex:73-75)
        # must hold, so clamp.
        f.write(f"{_oases_nw_line(nw_samples, 1)}\n")

        # Block VIII (unoast31.f:234): XLEFT XRIGHT XAXIS XINC, all km except
        # XAXIS. XLEFT/XRIGHT set the FFT output grid, not merely a plot window
        # (the .plp echoes XOFF/DX/N derived from them), so %.1f km = 100 m
        # resolution would snap the native grid — and any run shorter than
        # ~50 m would write XRIGHT = 0.0 and return an all-NaN field with no
        # exception. %.9f is sub-micron; the read is list-directed.
        #
        # XAXIS is the plot axis length in cm, used only by OASES's own
        # plotters (PLTLOS/PLDAV/CONDRW) — inert here, uacpy reads the numeric
        # output. XINC is *not* inert: unoast31.f:255 sets
        # NTLDEP = INT(|XRIGHT-XLEFT|/XINC)+1 and :705 places the TL-vs-depth
        # curves at XLEFT + (L-1)*XINC, so a hardcoded 1 km collapses to a
        # single curve on any run shorter than that. Scale it to the span.
        rmin_km = float(m_to_km(plot_rmin))
        rmax_km = float(m_to_km(plot_rmax))
        span_km = abs(rmax_km - rmin_km)
        xinc_km = (span_km / _OAST_TLDEP_CURVES) if span_km > 0 else 1.0
        f.write(f"{rmin_km:.9f} {rmax_km:.9f} "
                f"{_OAST_PLOT_AXIS_CM} {xinc_km:.9f}\n")

        # Block gating. GETOPT (unoast31.f:1016-1090) sets the flags the
        # READs are guarded by:
        #   DEPTAV='A'  DRCONT='C'  FRCONT='o'  ICONTU='c'
        #   PLTL='T'    PLKERN='I'  ANSPEC='a'  TLDEP='D'  IPROF='Z'
        # Blocks IX and XI are read once per selected output parameter
        # (`DO 980` :240-244 / `DO 990` :260-263, both skipping IOUT(i)==0);
        # blocks X and XII are read once.
        opt_chars = _oases_option_chars(options)

        def has(*codes):
            return any(code in opt_chars for code in codes)

        # IOUT entries GETOPT sets: N→1 V→2 H→3 R→5 K→6 S→7. With none of
        # them OASES falls back to IOUT(1)=1, NOUT=1 (unoast31.f:1175-1177).
        n_out = max(1, len(opt_chars & _OAST_OUTPUT_PARAM_CHARS))

        if (has('A') and not has('o')) or has('T', 'I', 'a', 'D'):
            # Block IX — YUP YDOWN YAXIS YINC (unoast31.f:242)
            f.write("20 100 12 10\n" * n_out)

        if has('C', 'D', 'c'):
            # Block X — RDUP RDDOWN CYAXIS RDINC (unoast31.f:250)
            f.write(f"0 {depth:.1f} 12 {depth/10:.1f}\n")

        if has('C', 'o'):
            # Block XI — ZMIN ZMAX ZSTEP contour levels (unoast31.f:262)
            f.write("40 100 10\n" * n_out)

        if has('Z'):
            # Block XII — VLEF VRIG VLEN VINC + DVUP DVLO DVLN DVIN
            c_min = float(ssp_data[:, 1].min())
            c_max = float(ssp_data[:, 1].max())
            c_range = c_max - c_min
            if c_range < 1.0:
                c_range = c_ref * 0.1
            c_plot_min = c_min - c_range * 0.05
            c_plot_max = c_max + c_range * 0.05
            c_inc = max(10, c_range / 10)
            f.write(f"{c_plot_min:.1f} {c_plot_max:.1f} 12 {c_inc:.1f}\n")
            f.write(f"0 {depth:.1f} 12 {depth/10:.1f}\n")


def write_oasn_input(
    filepath: Union[str, Path],
    env: Environment,
    source: Source,
    receiver: Receiver,
    options: Optional[str] = None,
    **kwargs
) -> None:
    """
    Write OASN (OASES Noise, Covariance Matrices and Signal Replicas) input file.

    Per oasn.tex:1, OASN produces:
    - Noise-field covariance matrices for ambient noise characterisation.
    - Array-response covariance matrices from discrete or continuous sources.
    - Signal replicas on a depth/range grid for matched-field processing.

    Parameters
    ----------
    filepath : str or Path
        Output file path (typically .dat extension)
    env : Environment
        Ocean environment (must be range-independent)
    source : Source
        Source specification (frequency used for mode computation)
    receiver : Receiver
        Receiver array specification (for covariance matrices)
    options : str, optional
        OASN option string. If None, uses 'N J' for normal mode computation
        Common options:
        - N: Output covariance matrices to .xsm file
        - R: Output replicas to .rpo file
        - J: Complex integration contour
        - F: Noise level vs frequency plot
        - P: Noise intensity vs receiver plot
    **kwargs : dict
        Additional parameters:
        - surface_noise_level : float
            Surface noise source strength in dB (default: 0, disabled).
            OASN disables the source when ``abs(SNLEVDB) < 0.01``
            (oasnun22.f:183) and gates Block VII on the complementary test
            ``abs(SNLEVDB) >= 0.01`` (oasnun22.f:276). A negative level is
            *not* "off": it names the Fortran unit number of a
            source-spectrum file (oasnun22.f:191), which this writer does
            not produce, so negative values are rejected.
        - white_noise_level : float
            White noise level in dB (default: 0, disabled)
        - deep_noise_level : float
            Deep broad-area source strength in dB (default: 0, disabled).
            The two OASN tests are *not* complementary: the source is
            computed only for ``DPLEVDB > 0.01`` (oasnun22.f:233 skips it on
            ``.LE.0.01``) while Block VIII is read for ``DPLEVDB >= 0.01``
            (oasnun22.f:324), so exactly 0.01 reads the block and radiates
            nothing. Unlike the surface level the deep source has no
            spectrum-file form, so a negative level simply disables it.
        - deep_source_depth : float
            Depth (m) of the deep source sheet (default: half the water depth)
        - discrete_sources : list of dict
            List of discrete sources, each carrying 'depth' (m), 'x', 'y'
            (km) and 'level' (dB) — the four fields NOIPAR reads
            (oasnun22.f:380). No other key is accepted; OASES has no
            per-source phase. As for the surface level, a negative ``'level'``
            names a spectrum-file unit number (oasnun22.f:383-385) rather
            than a level in dB, and is rejected.
        - integration_offset, offdb : float
            Wavenumber-integration contour offset in dB/wavelength; ``offdb``
            wins when both are given (default: 0)
        - nw_samples : int
            Number of wavenumber samples for every integration block;
            ``<= 0`` selects OASES's automatic sampling (default: -1)
        - c_low, c_high : float
            Phase-speed bounds (m/s) for the surface/deep noise blocks
        - cmins_discrete, cmaxs_discrete, cmins_replica, cmaxs_replica : float
            Per-block phase-speed bounds for the discrete-source and replica
            integrations
        - replica_zmin, replica_zmax, replica_nz : float, float, int
            Replica depth grid (m)
        - replica_xmin, replica_xmax, replica_nx : float, float, int
            Replica x grid (km)
        - replica_ymin, replica_ymax, replica_ny : float, float, int
            Replica y grid (km)

    Notes
    -----
    OASN input file has up to 10 blocks:
    I.   Title
    II.  Options
    III. Frequencies
    IV.  Environment
    V.   Receiver Array
    VI.  Sources (noise and discrete)
    VII. Surface noise parameters (if abs(SSLEV) >= 0.01)
    VIII. Deep noise parameters (if DSLEV >= 0.01)
    IX.  Discrete source parameters (if NDNS > 0)
    X.   Replica parameters (if option R)

    For normal mode computation, set options='N J' and ensure
    receiver array is specified properly.

    Examples
    --------
    >>> env = Environment(bathymetry=100, ssp=1500)
    >>> source = Source(depths=50, frequencies=100)
    >>> receiver = Receiver(depths=[30, 50, 70], ranges=[0])
    >>> write_oasn_input('test.dat', env, source, receiver,
    ...                  options='N J', surface_noise_level=70)
    """
    _reject_unknown_kwargs('write_oasn_input', kwargs, _OASN_KWARGS)
    filepath = Path(filepath)

    # Extract parameters
    freq = float(source.frequencies[0])
    depth = env.depth

    # Bottom properties
    bottom = env.bottom.halfspace_at(range=0.0)
    _bp = _extract_bottom_props(bottom)
    rho, c_p, c_s = _bp['rho'], _bp['c_p'], _bp['c_s']
    alpha_p, alpha_s = _bp['alpha_p'], _bp['alpha_s']

    # Sound speed profile — align to env.depth (see OAST writer for rationale).
    ssp_data = env.ssp.extend_to(depth).to_pairs()

    # Noise/source parameters. The levels are gated on the value OASN reads
    # back from the deck, so round to the '%.1f' the deck carries before
    # testing: a level of 0.03 is written as 0.0 and OASN then skips the
    # sub-block the unrounded value would have demanded.
    surface_noise_level = round(float(kwargs.get('surface_noise_level', 0)), 1)
    white_noise_level = float(kwargs.get('white_noise_level', 0))
    deep_noise_level = round(float(kwargs.get('deep_noise_level', 0)), 1)
    _check_oasn_noise_level('surface_noise_level', surface_noise_level,
                            dead_band=0.01)
    discrete_sources = kwargs.get('discrete_sources', [])
    n_discrete = len(discrete_sources)
    for i, ds in enumerate(discrete_sources):
        unknown = sorted(set(ds) - _OASN_DISCRETE_SOURCE_KEYS)
        if unknown:
            raise ConfigurationError(
                f"write_oasn_input: discrete_sources[{i}] carries key(s) "
                f"{unknown} that OASES never reads — NOIPAR reads exactly "
                f"ZDN, XDN, YDN, DNLEVDB (oasnun22.f:380).",
                remediation=f"Drop them; the accepted keys are "
                            f"{sorted(_OASN_DISCRETE_SOURCE_KEYS)}.",
            )
        _check_oasn_noise_level(f"discrete_sources[{i}]['level']",
                                ds.get('level', 180.0))

    # OASN aborts inside NOIPAR when surface noise is switched on over a
    # non-air upper halfspace: oasnun22.f:187 `IF (V(1,2).GT.500) STOP
    # '*** UPPER HALFSPACE MUST BE VACUUM OR AIR ***'`, reached whenever
    # SNLEVDB >= 0.01. Name the surface here rather than let the binary die.
    surface_c_p = _upper_halfspace_sound_speed(env)
    if surface_noise_level >= 0.01 and surface_c_p > 500.0:
        surface_kind = getattr(env.surface, 'acoustic_type', 'vacuum')
        raise ConfigurationError(
            f"OASN surface noise requires a vacuum or air upper halfspace, "
            f"but env.surface (acoustic_type={surface_kind!r}) writes "
            f"c_p = {surface_c_p:g} m/s; OASES stops with "
            f"'*** UPPER HALFSPACE MUST BE VACUUM OR AIR ***' "
            f"(oasnun22.f:187).",
            remediation=("Use a vacuum or air surface (c_p <= 500 m/s) for "
                         "surface-generated noise, or drop "
                         "surface_noise_level and drive the array with "
                         "discrete_sources / deep_noise_level instead."),
        )

    # Options string
    if options is None:
        options = 'N J'  # Covariance output, complex contour
    _reject_unwritten_option_blocks('write_oasn_input', options)

    # OASN reads the noise-source block only under `IF (CALNSE.or.trfout)`
    # (unoasn22.f:173-174); CALNSE comes from 'N'/'n' and TRFOUT from
    # uppercase 'T'. Written unconditionally, the block would be consumed by
    # whatever READ comes next — the replica grid under 'R' — and every
    # record below it shifts by one.
    noise_block = bool(_oases_option_chars(options) & {'N', 'n', 'T'})
    if not noise_block:
        supplied = sorted(
            k for k in ('surface_noise_level', 'white_noise_level',
                        'deep_noise_level', 'discrete_sources')
            if kwargs.get(k)
        )
        if supplied:
            raise ConfigurationError(
                f"write_oasn_input: {supplied} describe the noise field, but "
                f"OASN reads that block only when the option string carries "
                f"'N', 'n' or 'T' (unoasn22.f:173-174); options={options!r} "
                f"carries none of them.",
                remediation="Add 'N' to options=, or drop the noise arguments.",
            )

    _warn_volume_attenuation_ignored(env)

    # Integration parameters
    integration_offset = kwargs.get('integration_offset', 0)
    nw_samples = kwargs.get('nw_samples', -1)  # -1 = automatic

    freq_min_b, freq_max_b, nfreq = _resolve_freq_sweep(source, freq)

    with open(filepath, 'w') as f:
        _write_oases_header(f, env, options, "OASN Simulation via UACPY")

        # Block III: Frequencies — FREQ1 FREQ2 NFREQ COFF
        # (unoasn22.f:142 READ(1,*) FREQ1,FREQ2,NFREQ,OFFDBIN). ``offdb`` and
        # ``integration_offset`` name the same OASES field, so an explicit
        # ``offdb`` wins rather than being silently dropped.
        _emit_oases_freq_line(
            f, freq_min_b, freq_max_b, nfreq,
            integration_offset=kwargs.get('offdb', integration_offset),
        )

        # Block IV: Environment. NL = total layers (upper halfspace +
        # water + sediments + bottom halfspace).
        n_sed_layers = _count_bottom_layers(env)
        ssp_subset = _check_ssp_layer_count(ssp_data, 2 + n_sed_layers)
        n_water_layers = len(ssp_subset)
        n_layers = 1 + n_water_layers + n_sed_layers + 1
        f.write(f"{n_layers}\n")

        # Upper halfspace (from env.surface)
        f.write(f"{_format_upper_halfspace(env)}\n")

        # Water layers from SSP using OASES Airy-layer convention
        # (oaseun31.f:160-192): CC > 0 and CS < 0 declare an n^2-linear
        # layer whose speed varies from CC at the top of THIS layer to
        # -CS at the top of the NEXT layer.
        for i in range(len(ssp_subset)):
            d, c = ssp_subset[i]
            if i < len(ssp_subset) - 1:
                c_next = float(ssp_subset[i + 1, 1])
                cs = -abs(c_next)
            else:
                cs = 0
            f.write(f"{d:.2f} {c:.2f} {cs:.2f} 0.0 0 1.0 0 0\n")

        # Sediment layers + bottom halfspace (shared helper, B15)
        _emit_bottom_layers(
            f, env, depth,
            c_p, c_s, alpha_p, alpha_s, rho,
            extra_columns=1,
        )

        # Block V: Receiver Array
        # NRCV
        n_receivers = len(receiver.depths)
        _check_receiver_depth_count(n_receivers)
        f.write(f"{n_receivers}\n")

        # For each receiver: Z X Y ITYP GAIN
        # ITYP=1 for hydrophone, GAIN in dB
        for z in receiver.depths:
            f.write(f"{z:.2f} 0 0 1 0\n")

        # Blocks VI-IX are NOIPAR's, called only under
        # `IF (CALNSE.or.trfout)` (unoasn22.f:173-174).
        if noise_block:
            # Block VI: Sources — SSLEV WNLEV DSLEV NDNS
            f.write(f"{surface_noise_level:.1f} {white_noise_level:.1f} "
                    f"{deep_noise_level:.1f} {n_discrete}\n")

            # Block VII: surface-noise wavenumber parameters. NOIPAR reads
            # them only when `abs(SNLEVDB).GE.0.01` (oasnun22.f:276).
            if abs(surface_noise_level) >= 0.01:
                # CMINS CMAXS — the model's c_low/c_high when given, else the
                # OASN-manual defaults (0.95*c_min for the slow edge, 1E8 for
                # "no upper limit"). Documented as reaching this block, so it
                # must.
                f.write(f"{_noise_cmin(kwargs, ssp_data):.1f} "
                        f"{_noise_cmax(kwargs):.6g}\n")
                # NWSC NWSD NWSE (samples in continuous, discrete, evanescent)
                f.write(f"{_noise_nw(kwargs)}\n")

            # Block VIII: deep-noise parameters. NOIPAR reads them only when
            # `DPLEVDB.GE.0.01` (oasnun22.f:324) — deliberately asymmetric
            # with the surface gate above: a negative deep level just disables
            # the source, so emitting these lines would desynchronise every
            # later READ.
            if deep_noise_level >= 0.01:
                # DPSD (depth of deep source sheet)
                deep_source_depth = kwargs.get('deep_source_depth',
                                               depth * 0.5)
                f.write(f"{deep_source_depth:.2f}\n")
                # CMIND CMAXD
                f.write(f"{_noise_cmin(kwargs, ssp_data):.1f} "
                        f"{_noise_cmax(kwargs):.6g}\n")
                # NWDC NWDD NWDE
                f.write(f"{_noise_nw(kwargs)}\n")

        # Block IX: Discrete sources (if n_discrete > 0)
        if noise_block and n_discrete > 0:
            for ds in discrete_sources:
                # ZDN XDN YDN DNLEV (depth in m, x/y in km, level in dB)
                z_ds = ds.get('depth', 50.0)
                x_ds = ds.get('x', 1.0)  # km
                y_ds = ds.get('y', 0.0)  # km
                level_ds = ds.get('level', 180.0)
                f.write(f"{z_ds:.2f} {x_ds:.3f} {y_ds:.3f} {level_ds:.1f}\n")

            # Wavenumber sampling for discrete sources.
            # CMIN CMAX. ``cmaxs`` defaults to OASN-manual 1E8 for
            # surface-style noise integrations; let the caller pin it
            # via kwargs for fast-bottom critical-angle work.
            c_water_min = float(ssp_data[:, 1].min())
            cmins = kwargs.get('cmins_discrete', c_water_min * 0.95)
            cmaxs = kwargs.get('cmaxs_discrete', 1.0e8)
            f.write(f"{cmins:.1f} {cmaxs:.1f}\n")
            # NWDIN ICUT1D ICUT2D (oasnun22.f:420-421).
            f.write(f"{_oases_nw_line(nw_samples, 2000)}\n")

        # Block X: Replica parameters (if option 'R' is present)
        if _oases_option_chars(options) & {'R', 'r'}:
            # Replica grid: depths, x-ranges, y-ranges
            replica_zmin = kwargs.get('replica_zmin', 10.0)
            replica_zmax = kwargs.get('replica_zmax', depth - 10.0)
            replica_nz = kwargs.get('replica_nz', 20)
            replica_xmin = kwargs.get('replica_xmin', 0.1)  # km
            replica_xmax = kwargs.get('replica_xmax', 10.0)  # km
            replica_nx = kwargs.get('replica_nx', 50)
            replica_ymin = kwargs.get('replica_ymin', 0.0)  # km
            replica_ymax = kwargs.get('replica_ymax', 0.0)  # km
            replica_ny = kwargs.get('replica_ny', 1)

            f.write(f"{replica_zmin:.2f} {replica_zmax:.2f} {replica_nz}\n")
            f.write(f"{replica_xmin:.3f} {replica_xmax:.3f} {replica_nx}\n")
            f.write(f"{replica_ymin:.3f} {replica_ymax:.3f} {replica_ny}\n")

            # Wavenumber sampling for replicas (CMIN CMAX). Same
            # OASN-manual 1E8 default for the upper bound as the
            # discrete-source block — overridable via kwargs.
            c_water_min = float(ssp_data[:, 1].min())
            cmins = kwargs.get('cmins_replica', c_water_min * 0.95)
            cmaxs = kwargs.get('cmaxs_replica', 1.0e8)
            f.write(f"{cmins:.1f} {cmaxs:.1f}\n")
            # NWSIN ICUT1S ICUT2S (unoasn22.f:226-227).
            f.write(f"{_oases_nw_line(nw_samples, 2000)}\n")


def write_oasp_input(
    filepath: Union[str, Path],
    env: Environment,
    source: Source,
    receiver: Receiver,
    options: Optional[str] = None,
    **kwargs
) -> None:
    """
    Write OASP (OASES pulse) input file.

    OASP computes broadband transfer functions via wavenumber
    integration (range-independent), for postprocessing with the PP
    module or uacpy's time-series synthesis.

    Parameters
    ----------
    filepath : str or Path
        Output file path (typically .dat extension)
    env : Environment
        Ocean environment (can be range-dependent)
    source : Source
        Acoustic source specification
    receiver : Receiver
        Receiver array specification
    options : str, optional
        OASP option string. If None, uses default 'N V J'
        Common options:
        - N: Normal stress (pressure)
        - V: Vertical velocity
        - H: Horizontal velocity
        - J: Complex wavenumber contour
        - O: Complex frequency contour (for time-domain damping)
        - C: Omega-k contour plot
        - Z: Sound speed profile plot
        - f: Full Hankel transform for near field
    **kwargs : dict
        Additional parameters:
        - center_frequency : float
            Center frequency in Hz (default: source frequency)
        - integration_offset : float
            Integration contour offset in dB/wavelength (default: 0)
        - n_time_samples : int
            Number of time samples, must be power of 2 (default: 4096)
        - freq_min : float
            Lower frequency limit in Hz (default: 0)
        - freq_max : float
            Upper frequency limit in Hz (default: center_freq*2.5)
        - time_step : float
            Time sampling increment in seconds (default: auto)
        - range_start : float
            First range in **metres** (default: min receiver range)
        - range_step : float
            Range increment in **metres** (default: auto)
        - nw_samples : int
            Number of wavenumber samples (default: -1 for automatic)
        - dip_angle : float
            Fault dip angle in degrees for the ``'4'`` dip-slip moment
            source (default 0). Only read when ``'4'`` is in ``options``.

    Notes
    -----
    OASP input file has 8 blocks:
    I.   Title
    II.  Options
    III. Source frequency and integration offset
    IV.  Environment (layers)
    V.   Sources
    VI.  Receiver depths
    VII. Wavenumber sampling
    VIII. Frequency and range sampling

    OASP outputs .trf files (transfer functions) for postprocessing.

    Examples
    --------
    >>> env = Environment(bathymetry=100, ssp=1500)
    >>> source = Source(depths=80, frequencies=30)
    >>> receiver = Receiver(depths=np.linspace(20,100,5),
    ...                     ranges=np.linspace(1000,5000,5))
    >>> write_oasp_input('pulse.dat', env, source, receiver)
    """
    _reject_unknown_kwargs('write_oasp_input', kwargs, _OASP_KWARGS)
    filepath = Path(filepath)

    # Extract parameters
    center_freq = kwargs.get('center_frequency', float(source.frequencies[0]))
    depth = env.depth

    # Bottom properties
    bottom = env.bottom.halfspace_at(range=0.0)
    _bp = _extract_bottom_props(bottom)
    rho, c_p, c_s = _bp['rho'], _bp['c_p'], _bp['c_s']
    alpha_p, alpha_s = _bp['alpha_p'], _bp['alpha_s']

    # Sound speed profile — align to env.depth (see OAST writer).
    ssp_data = env.ssp.extend_to(depth).to_pairs()

    # Source parameter (receiver depth bookkeeping handled by
    # ``_receiver_block_lines`` which emits equidistant/explicit as needed).
    src_depth = float(source.depths[0])

    # Frequency and time parameters
    n_time = kwargs.get('n_time_samples', 4096)
    freq_min = kwargs.get('freq_min', 0.0)
    freq_max = kwargs.get('freq_max', center_freq * 2.5)

    # Auto-calculate time step if not provided
    if 'time_step' in kwargs:
        dt = kwargs['time_step']
    else:
        # Nyquist: dt = 1/(2*freq_max)
        dt = 1.0 / (2.0 * freq_max)

    # Range parameters — public API in metres, OASES expects km on disk.
    r_min_m = float(receiver.ranges.min())
    r_max_m = float(receiver.ranges.max())
    r1_km = float(m_to_km(kwargs.get('range_start', r_min_m)))

    if 'range_step' in kwargs:
        dr_km = float(m_to_km(kwargs['range_step']))
    else:
        n_ranges = len(receiver.ranges)
        if n_ranges > 1:
            # OASP evaluates r0 + i*dr (unoasp22.f:176 reads R0, RSPACE, NPLOTS
            # and nothing else), so a non-uniform request cannot be honoured —
            # it would silently return a different set of ranges than asked
            # for. OAST warns and interpolates; OASP has no such option.
            steps = np.diff(np.asarray(receiver.ranges, dtype=float))
            if not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-9):
                raise ConfigurationError(
                    f"OASP evaluates a uniform range axis (r0 + i*dr) but "
                    f"receiver.ranges is not uniformly spaced (steps "
                    f"{steps.min():.4g}-{steps.max():.4g} m).",
                    remediation=(
                        "Pass uniformly spaced ranges (np.linspace), or run "
                        "on a uniform grid and interpolate the Field "
                        "afterwards."),
                )
            dr_km = float(m_to_km((r_max_m - r_min_m) / (n_ranges - 1)))
        else:
            dr_km = 1.0

    nr = len(receiver.ranges)

    # Wavenumber sampling
    integration_offset = kwargs.get('integration_offset', 0)
    nw_samples = kwargs.get('nw_samples', -1)  # -1 = automatic
    cmin, cmax = _oases_wavenumber_bounds(ssp_data, cmax=1e9)

    # Options string. Default is single-component (normal stress / pressure)
    # plus the complex-contour integration flag. A second output letter such
    # as ``V`` raises NOUT, and the .trf reader keeps only the first
    # component, so the default asks for one.
    if options is None:
        options = 'N J'
    _reject_unwritten_option_blocks('write_oasp_input', options)
    _reject_tau_p('write_oasp_input', options)
    _check_nw_samples('write_oasp_input', nw_samples, power_of_two=False)
    source_record = _resolve_source_record(
        'write_oasp_input', options, kwargs.get('dip_angle'))

    _warn_volume_attenuation_ignored(env)

    with open(filepath, 'w') as f:
        _write_oases_header(f, env, options, "OASP Simulation via UACPY")

        # Block III: Source frequency and integration offset
        # FRC COFF [IT VS VR for Doppler]
        f.write(f"{center_freq:.1f} {integration_offset}\n")

        # Block IV: Environment. NL = total layers (upper halfspace +
        # water + sediments + bottom halfspace).
        n_sed_layers = _count_bottom_layers(env)
        ssp_array = np.asarray(
            _check_ssp_layer_count(ssp_data, 2 + n_sed_layers), dtype=float,
        ).reshape(-1, 2)
        n_water_layers = len(ssp_array)
        n_layers = 1 + n_water_layers + n_sed_layers + 1
        f.write(f"{n_layers}\n")

        # Upper halfspace (from env.surface)
        f.write(f"{_format_upper_halfspace(env)}\n")

        # Water layers from SSP. Per oast.tex §IV (Airy-layer convention):
        # CS < 0 carries the bottom-of-layer compressional speed so OASES
        # treats the layer as fluid with 1/c^2 linear. Last layer leaves
        # CS = 0 since the next interface is the sediment block.
        for i in range(len(ssp_array)):
            d, c = ssp_array[i]
            if i < len(ssp_array) - 1:
                cs = -abs(float(ssp_array[i + 1, 1]))
            else:
                cs = 0.0
            f.write(f"{d:.2f} {c:.2f} {cs:.2f} 0.0 0 1.0 0 0 0\n")

        # Sediment layers + bottom halfspace (shared helper, B15)
        _emit_bottom_layers(
            f, env, depth,
            c_p, c_s, alpha_p, alpha_s, rho,
            extra_columns=2,
        )

        # Block V: Sources — one source at src_depth, in the record shape
        # INSRC's LINA / dip_sou branch expects.
        f.write(_source_block_line(src_depth, **source_record) + '\n')

        # Block VI: Receiver depths (NRD<0 signals explicit depth list —
        # oasp.tex:559-585).
        for line in _receiver_block_lines(receiver):
            f.write(line + '\n')

        # Block VII: Wavenumber sampling
        # CMIN CMAX
        f.write(f"{cmin:.1f} {cmax:.1e}\n")

        # NW IC1 IC2 IF
        # NW=-1 for automatic sampling — IC1/IC2 have no effect (oasp.tex:677).
        # When NW > 0, IC2 must be set to NW so the Hankel transform is *not*
        # prematurely zeroed.
        # IF = frequency sample increment for kernels (0 disables plotting).
        if nw_samples is None or nw_samples <= 0:
            ic1, ic2 = 1, 1
        else:
            ic1, ic2 = 1, int(nw_samples)
        # NWVNO ICW1 ICW2 INTF (unoasp22.f:160). INTF gates integrand
        # *plots* (:541 `IF (MOD(JJ-LXP1,INTF).EQ.0) KPLOT=1`); the
        # .trf always carries every bin LXP1..MX regardless.
        intf = int(kwargs.get('freq_output_increment') or 40)
        f.write(f"{nw_samples} {ic1} {ic2} {intf}\n")

        # Block VIII: Frequency and range sampling
        # NT FR1 FR2 DT R1 DR NR
        # R0/RSPACE are km, so %.3f would be 1 m resolution: sub-metre receiver
        # spacing rounds to zero (every receiver onto one range) and metre-scale
        # spacing accumulates drift over the axis. %.9f is sub-micron.
        # unoasp22.f:176 is a list-directed read, so the extra digits are free.
        f.write(f"{n_time} {freq_min:.1f} {freq_max:.1f} {dt:.6f} "
                f"{r1_km:.9f} {dr_km:.9f} {nr}\n")


_REFL_TYPE_TO_OPTION = {
    'P-P': 'N',          # default — P-wave to P-wave reflection
    'P-SV': 'S',         # P-wave to vertical shear
    'P-Slow': 'B',       # P-wave to Biot slow wave
    'transmission': 't',  # transmission instead of reflection
}


def write_oasr_input(
    filepath: Union[str, Path],
    env: Environment,
    source: Source,
    receiver: Receiver,
    options: Optional[str] = None,
    interface_roughness: Optional[list] = None,
    angles: Optional[np.ndarray] = None,
    angle_type: str = 'grazing',
    reflection_type: str = 'P-P',
    **kwargs
) -> None:
    """
    Write OASR (OASES Reflection coefficient) input file

    OASR computes reflection/transmission coefficients as a function of
    frequency and grazing angle (or horizontal slowness). These coefficients
    can be used as input for other OASES modules or for analysis of
    seismo-acoustic interface properties.

    Parameters
    ----------
    filepath : str or Path
        Output file path (typically .dat extension)
    env : Environment
        Ocean environment specification
    source : Source
        Source specification (frequency used for computation)
    receiver : Receiver
        Receiver specification (not used directly in OASR)
    options : str, optional
        OASR option string. If None, uses default 'N T'
        Common options:
        - N: Default P-P reflection coefficient
        - T: Generate table of reflection coefficients (.rco/.trc files)
        - L: Loss in dB in addition to linear magnitude
        - P: Phase angle in addition to magnitude
        - S: P-SV reflection coefficient (replaces P-P)
        - C: Loss contours in frequency and angle
        - Z: Plot velocity profiles
        - p: Use slowness sampling instead of angle
        - t: Transmission coefficients instead of reflection
    angles : ndarray, optional
        Angles (degrees), which must be uniformly spaced — OASR's deck holds
        only ``(ANGLE1, ANGLE2, NANG)`` and generates the grid itself
        (unoasr21.f:173-177). If provided, overrides
        angle_min/angle_max/n_angles in ``kwargs``. Interpreted per
        ``angle_type`` (see below).
    angle_type : str, optional
        'grazing' (default) or 'incidence'. OASES expects grazing angles;
        when ``angle_type='incidence'``, angles are converted via
        ``grazing = 90 - incidence`` before being written.
    **kwargs : dict
        Additional parameters:
        - angle_min : float
            Minimum grazing angle in degrees (default: 0)
        - angle_max : float
            Maximum grazing angle in degrees (default: 90)
        - n_angles : int
            Number of angles (default: 181)
        - freq_min : float
            Minimum frequency in Hz (default: source.frequencies)
        - freq_max : float
            Maximum frequency in Hz (default: source.frequencies)
        - n_frequencies : int
            Number of frequencies (default: 1)

    Notes
    -----
    OASR input file has 9 blocks (some conditional on options):
    I.   Title
    II.  Options
    III. Environment (layers)
    IV.  Frequency sampling
    V.   Angle/Slowness sampling
    VI.  Angle/Slowness axes (for plots, if output requested)
    VII. Loss/Frequency axes (for plots)
    VIII. Loss contour plots (option C)
    IX.  SVP axes (option Z)

    OASR outputs:
    - .rco: Reflection coefficient vs slowness
    - .trc: Reflection coefficient vs angle
    - .plp, .plt: Plot files
    - .cdr, .bdr: Contour plot files

    Examples
    --------
    >>> env = Environment(bathymetry=100, ssp=1500)
    >>> env.bottom.sound_speed = 1600
    >>> env.bottom.shear_speed = 400
    >>> source = Source(depths=50, frequencies=100)
    >>> receiver = Receiver(depths=[50], ranges=[1000])
    >>> write_oasr_input('test.dat', env, source, receiver,
    ...                  angle_min=0, angle_max=90, n_angles=91)
    """
    _reject_unknown_kwargs('write_oasr_input', kwargs, _OASR_KWARGS)
    filepath = Path(filepath)

    # Extract parameters
    freq = float(source.frequencies[0])
    depth = env.depth

    # Bottom properties
    bottom = env.bottom.halfspace_at(range=0.0)
    _bp = _extract_bottom_props(bottom)
    rho, c_p, c_s = _bp['rho'], _bp['c_p'], _bp['c_s']
    alpha_p, alpha_s = _bp['alpha_p'], _bp['alpha_s']

    # Sound speed profile - for OASR we only need a single representative
    # water sound speed: OASR is a *local* interface reflection solver, the
    # source is placed 1 mm above the top of layer 2 (see unoasr21.f), and
    # layer 1 is treated as the upper halfspace carrying the incident wave.
    # A full stratified water column has no meaning here — OASR only sees the
    # (homogeneous) medium immediately above the reflecting interface.
    # Sound speed right above the seabed interface.
    c_water = float(env.ssp.extend_to(depth).to_pairs()[-1, 1])

    # Multi-frequency support — OASR sweep parameters. An explicit
    # freq_min/freq_max/n_frequencies (passed by ``OASR.run`` when the caller
    # supplied ``frequencies=``) takes precedence over ``source.frequencies``;
    # otherwise the source's own frequency vector drives the sweep.
    if 'freq_min' in kwargs:
        freq_min = float(kwargs['freq_min'])
        freq_max = float(kwargs.get('freq_max', freq_min))
        n_frequencies = int(kwargs.get('n_frequencies', 1))
    else:
        freqs_arr = np.atleast_1d(source.frequencies)
        if len(freqs_arr) > 1:
            freq_min = float(freqs_arr.min())
            freq_max = float(freqs_arr.max())
            n_frequencies = int(len(freqs_arr))
        else:
            freq_min = freq
            freq_max = freq
            n_frequencies = 1
    freq_out_inc = kwargs.get('freq_output_increment', max(1, n_frequencies // 10))
    _check_frequency_contours('write_oasr_input', options, 'C', n_frequencies)

    # Angle parameters. OASES natively uses grazing angles; if the caller
    # requested 'incidence', convert to grazing via 90 - incidence.
    if angle_type not in ('grazing', 'incidence'):
        raise ConfigurationError(
            f"OASR: angle_type must be 'grazing' or 'incidence', got {angle_type!r}"
        )
    if angles is not None:
        angles_arr = np.atleast_1d(np.asarray(angles, dtype=float))
        # OASR evaluates ANGLE1 + (i-1)*DLANGLE (unoasr21.f:173-177) and the
        # deck carries only (ANGLE1, ANGLE2, NANG), so a non-uniform request
        # cannot be honoured — it would return a different angle grid than
        # asked for. Same rule as OASP's uniform range axis.
        if angles_arr.size > 1:
            steps = np.diff(angles_arr)
            if not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-9):
                raise ConfigurationError(
                    f"OASR evaluates a uniform angle axis "
                    f"(angle_min + i*d_angle) but ``angles`` is not uniformly "
                    f"spaced (steps {steps.min():.4g}-{steps.max():.4g} deg).",
                    remediation=(
                        "Pass uniformly spaced angles (np.linspace), or run "
                        "on a uniform grid and interpolate the "
                        "ReflectionCoefficient afterwards."),
                )
        if angle_type == 'incidence':
            angles_arr = 90.0 - angles_arr
        angle_min = float(angles_arr.min())
        angle_max = float(angles_arr.max())
        n_angles = int(len(angles_arr))
    else:
        angle_min = kwargs.get('angle_min', 0.0)
        angle_max = kwargs.get('angle_max', 90.0)
        if angle_type == 'incidence':
            # Convert scalar bounds as well so the user-facing axis is honored.
            angle_min, angle_max = 90.0 - angle_max, 90.0 - angle_min
        n_angles = kwargs.get('n_angles', 181)
    angle_out_inc = kwargs.get('angle_output_increment', max(1, n_angles // 10))

    # Options string. The wrapper translates ``reflection_type`` to the
    # corresponding OASR letter; the ASCII ``T`` table is always added so
    # the Python reader has something to parse.
    if reflection_type not in _REFL_TYPE_TO_OPTION:
        raise ConfigurationError(
            f"OASR: reflection_type must be one of {list(_REFL_TYPE_TO_OPTION)}, "
            f"got {reflection_type!r}"
        )
    if options is None:
        opt_letter = _REFL_TYPE_TO_OPTION[reflection_type]
        options = f"{opt_letter} T"
    _reject_unwritten_option_blocks('write_oasr_input', options)

    _warn_volume_attenuation_ignored(env, lossless_water=True)

    # Interface roughness (RG / CL / M) per interface (B13 #6).
    # Indexed starting at 0 = upper-halfspace/surface interface; None -> no roughness.
    if interface_roughness is None:
        interface_roughness = []

    def _roughness_tail(i):
        """Return roughness-suffix string for interface index ``i``.

        OASES convention (oases_gen.tex): RG > 0 → RMS roughness only;
        RG < 0 → |RG| plus CL and M on same line (Goff-Jordan power spectrum).
        """
        if i < 0 or i >= len(interface_roughness):
            return " 0"
        spec = interface_roughness[i]
        if spec is None:
            return " 0"
        if isinstance(spec, (int, float)):
            return f" {float(spec):.4f}"
        # dict or tuple
        if isinstance(spec, dict):
            rg = spec.get('RG', spec.get('roughness', 0.0))
            cl = spec.get('CL', spec.get('correlation_length', None))
            m = spec.get('M', spec.get('spectral_exponent', None))
        else:  # assume tuple/list
            rg = spec[0] if len(spec) > 0 else 0.0
            cl = spec[1] if len(spec) > 1 else None
            m = spec[2] if len(spec) > 2 else None
        if cl is None or m is None:
            return f" {float(rg):.4f}"
        # Flag negative RG to signal CL + M follow.
        return f" {-abs(float(rg)):.4f} {float(cl):.4f} {float(m):.4f}"

    with open(filepath, 'w') as f:
        _write_oases_header(f, env, options, "OASR Simulation via UACPY")

        # Block III: Environment (B13)
        # OASR convention: layer 1 IS the upper halfspace in which the source
        # sits (placed 1 mm above layer 2's top). Reflection is computed at the
        # interface between layer 1 and layer 2. We therefore emit the water
        # column as the upper halfspace — NOT a separate vacuum layer above a
        # water layer, which would place the source in the vacuum and make the
        # solver compute a vacuum/water reflection (empty .rco output).
        #
        # Reference: oasr.tex section "Output Files" (saffipr1.dat example) —
        # NL = 3 with `0 1500 0 0 0 1 0` as layer 1 (water), followed by
        # sediment + halfspace.
        n_sed_layers = _count_bottom_layers(env)
        n_layers = 1 + n_sed_layers + 1  # water-halfspace + sediment + bottom
        f.write(f"{n_layers}\n")

        # Layer 1: water as upper halfspace (D is dummy for layer 1).
        # AC=0 with CS=0 makes OASR overwrite V(1,4) with 1e-8
        # (unoasr21.f:95-97), which defeats the `V(I,4).LE.0` test in PINIT2
        # (oaseun31.f:1516-1521) and so suppresses the empirical
        # Skretting-Leroy substitution — a lossless upper halfspace, which is
        # what a plane-wave reflection coefficient wants.
        f.write(f"0.00 {c_water:.2f} 0 0.0 0 1.0{_roughness_tail(0)}\n")

        # Sediment stack + bottom halfspace via the shared helper, with
        # OASR's per-interface roughness suffix; iface_start=1 because
        # interface 0 was the water/sediment-top line above.
        _emit_bottom_layers(
            f, env, depth,
            c_p, c_s, alpha_p, alpha_s, rho,
            suffix_fn=_roughness_tail,
            iface_start=1,
        )

        # Block IV: Frequency sampling
        # FMIN FMAX NFREQ NFOU
        f.write(f"{freq_min:.1f} {freq_max:.1f} {n_frequencies} {freq_out_inc}\n")

        # Block V: Angle/Slowness sampling
        # AMIN AMAX NRAN NAOU
        f.write(f"{angle_min:.1f} {angle_max:.1f} {n_angles} {angle_out_inc}\n")

        # Block VI: Angle/Slowness axes (if plot output)
        if freq_out_inc > 0:
            # ALEF ARIG ALEN AINC
            # RALO RAUP RALN RAIN
            f.write(f"{angle_min:.1f} {angle_max:.1f} 12 {max(10, (angle_max-angle_min)/10):.1f}\n")
            f.write("0 1 12 0.2\n")  # Reflection coefficient magnitude 0-1

        # Block VII: Loss/Frequency axes (if angle output)
        if angle_out_inc > 0:
            # FLEF FRIG FLEN FINC
            # RFLO RFUP RFLN RFIN
            f_range = max(freq_max - freq_min, freq_min * 0.1)
            f.write(f"{freq_min:.1f} {freq_max:.1f} 12 {f_range/10:.1f}\n")
            f.write("0 30 12 5\n")  # Reflection loss 0-30 dB

        # Block VIII: Loss contour plots (if option C). OASR's GETOPT
        # (unoasr21.f:386) sets CONTUR from uppercase 'C' only — it defines no
        # lowercase 'c', so matching one would emit a block OASR never reads
        # and shift the 'Z' block below onto it.
        if 'C' in _oases_option_chars(options):
            # ALEF ARIG ALEN AINC
            # FRLO FRUP OCLN NTKM
            # ZMIN ZMAX ZINC
            f.write(f"{angle_min:.1f} {angle_max:.1f} 12 {(angle_max-angle_min)/10:.1f}\n")
            octave_range = np.log2(freq_max / freq_min) if freq_max > freq_min else 1.0
            f.write(f"{freq_min:.1f} {freq_max:.1f} {octave_range*2:.1f} 5\n")
            f.write("0 20 2\n")  # Contour levels 0-20 dB in 2 dB increments

        # Block IX: SVP axes (if option Z). Uppercase only — unoasr21.f:392.
        if 'Z' in _oases_option_chars(options):
            # VLEF VRIG VLEN VINC
            # DVUP DVLO DVLN DVIN
            c_min = min(c_water, c_p) * 0.95
            c_max = max(c_water, c_p) * 1.05
            f.write(f"{c_min:.1f} {c_max:.1f} 12 100\n")
            f.write(f"0 {depth:.1f} 12 {depth/10:.1f}\n")
