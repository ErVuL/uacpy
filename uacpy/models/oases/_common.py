"""What the six OASES programs share: the executable lookup, the FORnnn
environment of a launch, the echo patterns of the binaries' stdout, the
option-letter helpers, the auto-sampling offset notice and the rough-
interface checks OASS and OASSP both make."""

import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from uacpy.models.base import PropagationModel
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.models._spec import EngineTraits
from uacpy.core.source import Source
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, ValidityWarning,
)
from uacpy.io.oases_writer import (
    MIN_SPECTRAL_EXPONENT, oass_bottom_interfaces, _OASES_MAX_WAVENUMBERS,
)
from uacpy.core.absorption import warn_if_band_absorption_frozen
from uacpy.models._knobs import whole_count
from uacpy.models._notices import message_notice


#: A scattering run's ``c_low`` agrees with its mean field's within
#: this relative tolerance.
C_LOW_MATCH_RTOL = 1e-6


def _rough_interface_sigma(env, interface: int, oass_index_space: bool) -> float:
    """RMS roughness (m) of the interface being scattered from.

    ``INTFC`` lives in TWO different deck index spaces and
    :func:`~uacpy.io.oases_writer.oass_bottom_interfaces` says outright in its
    own docstring that they "must not be mixed": OAST/OASS collapse an isovelocity water column to one record, the
    OASP/OASSP decks never do, so the same seafloor is deck layer 3 in one and
    ``2 + n_ssp_rows`` in the other. Looking an OASSP interface up in the OASS
    table therefore indexes past the end of the roughness list and reads 0.0 —
    silently disarming the caller's check.

    So only OASS indexes. OASSP is guaranteed by
    ``_require_single_rough_interface`` to have exactly one rough interface,
    which can be found without any index at all.
    """
    first_bottom, roughness = oass_bottom_interfaces(env)
    surface = abs(float(env.surface.roughness)) if env.surface is not None else 0.0
    if not oass_index_space:
        candidates = [surface] + [abs(float(r)) for r in roughness]
        return max(candidates) if candidates else 0.0
    if interface == _OASS_SURFACE_INTERFACE:
        return surface
    index = interface - first_bottom
    return (abs(float(roughness[index]))
            if 0 <= index < len(roughness) else 0.0)


def _roughness_notice(model_name: str, env, interface: int,
                      frequency_hz: float, rms_override,
                      *, oass_index_space: bool) -> Optional[str]:
    """The notice for a rough-interface run outside the small-roughness
    theory, or ``None`` (a stage-3 notice of OASS and OASSP).

    OASS and OASSP both treat the rough interface as a perturbation of the
    flat one, which is a small-roughness expansion: valid while the RMS height
    is small against the wavelength normal to the surface. Nothing in the
    returned field marks where that stops being true — the scattered
    intensity stays smooth and plausible — so the Rayleigh parameter is the
    only warning available.

    The deck's own ``rms_roughness`` wins when set, since that is what the
    scattering run uses; otherwise the environment's roughness is the one in
    force. For a broadband run pass the HIGHEST frequency: ``P`` grows with
    ``k``, so that is the worst case.
    """
    from uacpy.sonar.scattering import non_perturbative_roughness_notice

    if rms_override is not None:
        sigma = abs(float(rms_override))
    else:
        sigma = _rough_interface_sigma(env, interface, oass_index_space)

    depth = (0.0 if (oass_index_space and interface == _OASS_SURFACE_INTERFACE)
             else float(env.depth))
    # ssp.sound_speed_at returns an array even for a scalar depth.
    c_interface = float(np.ravel(env.ssp.sound_speed_at(depth))[0])
    # The one Rayleigh-parameter rule is the sonar check's.
    return message_notice(non_perturbative_roughness_notice(
                    model_name, float(frequency_hz), sigma, sound_speed=c_interface),
                       ValidityWarning)


def _check_n_wavenumbers_knob(n_wavenumbers) -> None:
    """Refuse a pinned wavenumber count that is not a whole number >= 1;
    ``None`` (OASES' automatic sampling) passes."""
    if n_wavenumbers is not None:
        whole_count('n_wavenumbers', n_wavenumbers, 1)


def _warn_offset_ignored_under_auto_sampling(model_name, integration_offset,
                                             n_wavenumbers, options=None,
                                             offset_letters='J',
                                             auto_sampling_zeroes_offset=True):
    """Warn that a caller's contour offset never reaches the integration.

    Two independent ways the binary discards it, either of which is enough:

    * **automatic wavenumber sampling** — ``unoast31.f:429`` sets
      ``OFFDB=0E0`` inside the ``NWVNOin < 0`` branch (``unoasp22.f:323`` for
      OASP), so ``:499-507`` then applies the kernel's own default and prints
      ``THE DEFAULT CONTOUR OFFSET IS APPLIED``. The offset still reaches the
      deck; the binary throws it away. The manual (``oast.tex:547-550``)
      documents only that ``IC1``/``IC2`` have no effect here, so following it
      is not enough to know this.
    * **an option line that never reads the token** — the frequency line is
      read with the offset field only under the letters in ``offset_letters``
      (``unoast31.f:126-133``, ``unoasp22.f:126-133``,
      ``unoassp30.f:135-142``). Without one of them the binary zeroes the
      offset *before* the read and consumes a shorter record, so the value
      uacpy wrote is not even parsed. 'J' also gates the application itself
      (``unoast31.f:499``, ``unoasp22.f:354-366``).

    OASN is unaffected by the first — ``unoasn22.f:283`` tests ``OFFDBIN``,
    which its automatic branch never touches — and passes
    ``auto_sampling_zeroes_offset=False`` to say so. It is *not* exempt from
    the second: ``:141-145`` reads the ``OFFDBIN`` token only under
    ``ICNTIN > 0``, which only ``'J'``/``'j'`` sets (``:673-675``).
    """
    if not integration_offset:
        return
    reasons = []
    if auto_sampling_zeroes_offset and n_wavenumbers is None:
        reasons.append(
            f"automatic wavenumber sampling (n_wavenumbers=None) zeroes "
            f"the contour offset and applies the kernel's own default — pin "
            f"n_wavenumbers >= 1 to use the value")
    if options is not None and not (set(str(options)) & set(offset_letters)):
        reasons.append(
            f"the option line for this run ({str(options)!r}) carries none of "
            f"{sorted(offset_letters)}, so the binary zeroes the offset and "
            f"reads a frequency line without it — add 'J' (complex "
            f"integration contour) to use the value")
    if not reasons:
        return
    warnings.warn(
        f"{model_name}(integration_offset={integration_offset:g}) has no "
        f"effect: " + "; ".join(reasons) + ". Drop integration_offset to "
        f"silence this.",
        FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)


def _reject_unintegrable_spectral_exponent(model_name, spectral_exponent):
    """Refuse a roughness exponent whose power spectrum does not converge.

    ``M <= 1.5`` leaves the spectrum unintegrable (``oassp.tex:356-362``; the
    exponent reaches ``amod(m) = fac(3+…)`` at ``oaseun31.f:99``). The deck
    writer refuses it too, but by then the scattering models have already
    spent the mean-field binary — and this depends on a constructor argument
    alone, so it belongs here.
    """
    if float(spectral_exponent) > MIN_SPECTRAL_EXPONENT:
        return
    raise ConfigurationError(
        f"{model_name}(spectral_exponent={float(spectral_exponent)}) must "
        f"exceed {MIN_SPECTRAL_EXPONENT:g}, or the roughness power spectrum "
        f"is not integrable "
        f"(oassp.tex:356-362; the exponent reaches amod(m)=fac(3+…) at "
        f"oaseun31.f:99).",
        remediation=f"Pass spectral_exponent > {MIN_SPECTRAL_EXPONENT:g} "
                    f"(2.0 is the default).",
    )


def _oases_find_executable(model: PropagationModel, name: str) -> Path:
    """Locate an OASES binary, preferring the ``<name>_bash`` wrapper.

    Searches ``uacpy/bin/oases``, ``uacpy/bin/oalib``, and
    ``uacpy/third_party/oases/bin``.
    """
    return model._find_executable_in_paths(
        [f'{name}_bash', name],
        bin_subdirs=['oases', 'oalib'],
        dev_subdir='oases',
    )


#: OASP echoes the count it settled on before the frequency loop
#: (unoasp22.f:340, :349).
_OASES_NWVNO_ECHO = re.compile(r'NO\. OF WAVENUMBERS:\s*(\d+)')


def _oases_subprocess_env(base_name: str, **extras: str) -> dict:
    """Build the FORnnn env-var dict the OASES csh wrappers set.

    OPFILW/OPFILR/OPFILB resolve every Fortran unit through
    ``GETENV('FORnnn')`` (oases/src/oashun21.f:555-556, :590, :629), falling
    back to ``<deck stem>.<nnn>`` when the variable is unset (:560, :633) —
    so a unit with no entry here still lands on the base_name (e.g. the
    ``.021`` OASN and OASR leave behind), and only a WRITE that bypasses
    OPFIL entirely produces a compiler-named ``fort.NN``. The binary is
    therefore run directly and this dict stands in for
    ``third_party/oases/bin/{oast,oasn,oasp,oasr}``. The keys every
    sub-model needs are FOR001 (the ``.dat`` deck), FOR019/FOR020 (the
    ``.plp`` header and ``.plt`` data OASES' plotters write), FOR028/FOR029
    (contour output, ``.cdr``/``.bdr`` in the wrappers, which uacpy never
    reads) and FOR045 (scattering right-hand sides, ``.rhs``). FOR045's
    ``.045`` suffix is load-bearing: it is the file the whole OASS/OASSP
    chain consumes — their mean-field launch requires ``<stem>.045`` from
    the producer run and ``_execute`` re-points the consumer's FOR045 at
    that same name. ``extras``
    supplies the per-binary units the wrappers add (e.g. FOR002='src' for
    OAST, FOR016='xsm' for OASN) — pass the suffix without the
    ``base_name + '.'`` prefix. ``base_name`` is the stem of the input file
    (no extension).
    """
    import os
    # ``base_name`` is interpolated straight into the FORnnn filenames the
    # OASES csh wrappers open; a value with a path separator or ``..`` would
    # become a traversal. It is a hard-coded literal in every caller today —
    # keep it that way by rejecting anything that isn't a plain stem.
    if not re.fullmatch(r'[A-Za-z0-9_]+', base_name):
        raise ConfigurationError(
            f"OASES base_name must match [A-Za-z0-9_]+; got {base_name!r}."
        )
    env = os.environ.copy()
    env['FOR001'] = f'{base_name}.dat'
    env['FOR019'] = f'{base_name}.plp'
    env['FOR020'] = f'{base_name}.plt'
    env['FOR028'] = f'{base_name}.028'
    env['FOR029'] = f'{base_name}.029'
    env['FOR045'] = f'{base_name}.045'
    for key, suffix in extras.items():
        env[key] = f'{base_name}.{suffix}'
    return env


#: Roughness power spectra OASS and OASSP implement. GETOPT's only spectrum
#: letter is 'g'/'G' (goff, unoass21.f:674, unoassp30.f:1034); its absence
#: means Gaussian.
_OASS_SPECTRA = ('gaussian', 'goff-jordan')


#: RMS roughness (m) below which SCTRHS writes no boundary operators at all:
#: it skips any interface with ROUGH2 < 1e-10 (oaseun31.f:2310) and
#: ROUGH2 = ROUGH**2 (:379).
_OASS_MIN_MEAN_FIELD_ROUGHNESS = 1e-5


#: Deck layer 2 is the first water record, and ``ROUGH(M)`` is the roughness of
#: the interface at the TOP of layer M (``oaseun31.f:381-383``), so layer 2's
#: RG is the sea surface. ``SCTRHS`` stamps its records ``IN1 = 2``
#: (``oaseun31.f:2308-2309``), which is exactly what ``INTFC = 2`` selects.
_OASS_SURFACE_INTERFACE = 2


#: The bare ``fort.46`` an option-``'s'`` run leaves in the work dir.
#: ``oaseun31.f:1900`` and ``:2007`` WRITE unit 46 with no ``OPFILW`` (the
#: call is commented out at ``:2012``), so the write lands on the default
#: name instead of a ``base_name`` suffix. Declared by the classes whose
#: binary can reach those writes, so a pinned ``work_dir`` starts each run
#: clean rather than carrying a previous run's file forward.
#:
#: Not hoisted onto :class:`OASES`, because the list is a census of what each
#: binary writes and two of the six do not write this one. ``OASS`` cannot
#: reach the writes at all: its consumer deck carries no ``'s'``, so
#: ``SCTOUT`` stays at the ``.FALSE.`` ``unoass21.f:721-722`` initialises it
#: to, and the ``fort.46`` seen after an OASS run is the OAST producer's,
#: cleared under that class. ``OASSP`` stages units 45/46 on the mean-field
#: stem (``<mean_stem>.045`` / ``.046``), which are different files from the
#: bare name, so it neither writes nor reads this one. A base-class entry
#: would be inert for both — it would just stop saying which binary writes
#: what.
_SCTOUT_BARE_FORT46 = ('fort.46',)


#: What every OASES program does with the parts of a call the base decides
#: (each program's ``spec.traits`` starts here).
_OASES_TRAITS = EngineTraits(
    # The water layers' AC column carries env.absorption in dB/wavelength
    # (oases_writer._water_ac), and a lossless-water AC when it is None,
    # never the 0 that OASES would replace with its own Skretting-Leroy law.
    # OASR, whose water is a lossless half-space, overrides this.
    consumes_volume_absorption=True,
    # OASES meshes the sediment layers as media in their own right, so a
    # source or receiver inside the seabed is a supported geometry — buried
    # and sub-bottom sources are much of what the seismo-acoustic family is
    # for.
    receivers_reach_sediment=True,
)


def _warn_if_water_ac_extrapolates(name: str, env, frequencies,
                                   anchor: float) -> None:
    """Say what the one water AC per layer costs a multi-frequency deck
    of the OASES program ``name``.

    ``oases_writer._water_ac`` evaluates ``env.absorption`` once, at
    ``anchor``, in dB per wavelength, and OASES re-applies it at every
    frequency as ``VI4 = FREQ*VV/(8.68588964*V(I,2))``
    (``oaseun31.f:1522``): the water absorption is linear in ``f``
    across the band. Exact for a constant dB/wavelength law; for Thorp,
    Francois-Garrison or a Biological resonance the shared band check
    (:func:`~uacpy.core.absorption.warn_if_band_absorption_frozen`)
    reports the gap over the band and the water column."""
    warn_if_band_absorption_frozen(
        name, env.absorption, frequencies, float(anchor),
        water_depth=float(env.depth),
        mechanism=(
            f"the deck carries one water AC per layer, evaluated at "
            f"{float(anchor):.4g} Hz, and OASES re-applies it at every "
            f"frequency as VI4 = FREQ*VV/(8.68588964*V(I,2)) "
            f"(oaseun31.f:1522), so the water absorption is linear in "
            f"frequency."),
        remediation=("Run narrower sub-bands (one deck each), or a "
                "per-frequency engine (Kraken, Scooter, RAM), if the "
                "band edges matter."))


def _reject_wavenumber_overrun(name: str, process, scale: int) -> None:
    """Raise when the run of the OASES program ``name`` integrated on
    more wavenumbers than NP holds.

    Under automatic sampling AUTSAM sizes the wavenumber axis from the
    frequency-range product with no bound (unoasp22.f:1130 and its OASSP
    copy unoassp30.f:1125, ``NW=(WNMAX-WNMIN)/DK+1``), and — unlike OAST,
    which stops at unoast31.f:459 — neither OASP nor OASSP has an
    ``NWVNO.GT.NP`` test (their ``MIN0(NWVNO,NP)`` runs *before* AUTSAM,
    so it cannot bound the automatic branch). Past NP the run completes,
    exits 0 and writes a full-size ``.trf`` holding numbers that are not
    the field (measured 3.2e27 against 1.3e-4 for the same geometry with
    NW pinned under NP). The count the binary settled on is in its own
    stdout, so read it from there rather than re-deriving AUTSAM.
    ``scale`` is the factor between the echoed count and the count
    integrated (1 but for OASSP's roughness branch,
    :func:`_oassp_wavenumber_echo_scale`).
    """
    counts = [scale * int(m) for m in
              _OASES_NWVNO_ECHO.findall(getattr(process, 'stdout', '') or '')]
    if not counts or max(counts) <= _OASES_MAX_WAVENUMBERS:
        return
    raise ConfigurationError(
        f"{name} sampled {max(counts)} wavenumbers, past "
        f"OASES' array bound NP = {_OASES_MAX_WAVENUMBERS} "
        f"(oases/src/compar.f:37-38); "
        f"this binary has no bound check on automatic sampling, so the "
        f"transfer function it wrote is not the acoustic field. The "
        f"count grows with the highest frequency times the largest "
        f"receiver range.",
        remediation=(
            "Shorten receiver.ranges, lower freq_max / the source "
            "frequency, or pin n_wavenumbers <= "
            f"{_OASES_MAX_WAVENUMBERS} to integrate on a bounded grid."),
    )


def _option_letters(options) -> set:
    """The letters of an OASES option line (GETOPT reads it character by
    character and skips blanks)."""
    return set(str(options or '')) - set(' \t\n')


def _reject_plane_geometry_letter_on_a_point_source(
        model_name: str, options: str, raw_options, source: Source,
        where: str) -> None:
    """Refuse an option line carrying ``'P'`` with a point Source.

    ``'P'`` switches OAST, OASP, OASSP and OASS to plane geometry (at
    ``where``), i.e. a line source, and the field they return is that of a
    line source (measured ~35 dB above the point-source TL of the same
    geometry). The Source decides the geometry: pass
    ``Source(source_type='line')`` and the option line gets its ``'P'`` from
    it. ``options`` is the line the deck carries, ``raw_options`` the
    constructor's string the message names.
    """
    if source.source_type == 'line' or 'P' not in _option_letters(options):
        return
    raise ConfigurationError(
        f"{model_name}(options={raw_options!r}) carries 'P', which makes "
        f"{model_name} compute a line source (plane geometry, {where}), "
        f"but the Source is a {source.source_type!r} source; the result "
        f"would be a line-source field recorded as a point-source run.",
        remediation=("Pass Source(source_type='line') for a line source "
                     "(its 'P' is then written for you), or drop 'P' from "
                     "options= for a point source."),
    )


def _geometry_options(options: str, source: Source) -> str:
    """The option line with the geometry the Source decides: ``'P'``
    (plane geometry, a line source) appended for a line Source that does not
    already carry it; any other line as it is. The one rule OAST, OASP,
    OASSP and OASS write their option line by; a raw ``'P'`` with a point
    Source is refused by
    :func:`_reject_plane_geometry_letter_on_a_point_source`."""
    if source.source_type == 'line' and 'P' not in _option_letters(options):
        return f"{options} P"
    return options
