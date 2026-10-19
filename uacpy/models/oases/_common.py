"""What the six OASES programs share: the executable lookup, the FORnnn
environment of a launch, the echo patterns of the binaries' stdout, the
option-letter helpers, the auto-sampling offset notice and the rough-
interface checks OASS and OASSP both make."""

import re
from typing import Optional

import numpy as np

from uacpy.models._launch import BinarySearch
from uacpy.models._spec import EngineTraits
from uacpy.core.source import Source
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, ValidityWarning,
)
from uacpy.io.oalib_writer import writable_layers
from uacpy.io.oases_writer import (
    oass_bottom_interfaces, _OASES_MAX_WAVENUMBERS,
    refuse_unusable_integration_window,
)
from uacpy.core.absorption import _warn_if_band_absorption_frozen
from uacpy.core.engine_defaults import OASSP_SPECTRAL_EXPONENT
from uacpy.models._knobs import whole_count
from uacpy.models._notices import message_notice


#: A scattering run's ``c_low`` agrees with its mean field's within
#: this relative tolerance.
C_LOW_MATCH_RTOL = 1e-6


def _rough_interface_sigma(env, interface: int, oass_index_space: bool) -> float:
    """RMS roughness (m) of the interface being scattered from.

    ``INTFC`` lives in TWO different deck index spaces and
    :func:`~uacpy.io.oases_writer.oass_bottom_interfaces` says outright in its
    own docstring that they "must not be mixed": OAST/OASS collapse an
    isovelocity water column to one record, the OASP/OASSP decks never do,
    so the same seafloor is deck layer 3 in one and
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


def _speed_above_interface(env, interface: int,
                           oass_index_space: bool) -> float:
    """Compressional speed (m/s) of the medium the incident wave crosses to
    reach the scattering interface: the water's at the surface and at the
    seafloor, the sediment layer's above a sub-bottom interface.

    The Rayleigh parameter's wavenumber is the one along the reflected
    path, which lengthens by ``2 h sin(theta)`` in that medium
    (:func:`~uacpy.sonar.scattering.rayleigh_parameter`, JKPS §1.7). The
    bottom interfaces are numbered as :func:`oass_bottom_interfaces` lists
    them: the seafloor, then the top of each deeper layer and of the
    half-space. In the OASP/OASSP index space the one rough interface is
    the roughest of the surface and those (``_rough_interface_sigma``).
    """
    first_bottom, roughness = oass_bottom_interfaces(env)
    if oass_index_space:
        k = (None if interface == _OASS_SURFACE_INTERFACE
             else interface - first_bottom)
    else:
        rough = [abs(float(r)) for r in roughness]
        surface = abs(float(env.surface.roughness))
        k = (None if not rough or surface >= max(rough)
             else int(np.argmax(rough)))
    layers = (writable_layers(env.bottom.columns[0])
              if env.bottom.is_layered else [])
    if k is not None and 1 <= k <= len(layers):
        return float(layers[k - 1].sound_speed)
    depth = 0.0 if k is None else float(env.depth)
    # ssp.sound_speed_at returns an array even for a scalar depth.
    return float(np.ravel(env.ssp.sound_speed_at(depth))[0])


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
    from uacpy.sonar.scattering import _non_perturbative_roughness_notice

    if rms_override is not None:
        sigma = abs(float(rms_override))
    else:
        sigma = _rough_interface_sigma(env, interface, oass_index_space)

    c_interface = _speed_above_interface(env, interface, oass_index_space)
    # The one Rayleigh-parameter rule is the sonar check's.
    return message_notice(_non_perturbative_roughness_notice(
                    model_name, float(frequency_hz), sigma, sound_speed=c_interface),
                       ValidityWarning)


def _check_n_wavenumbers_knob(n_wavenumbers) -> None:
    """Refuse a pinned wavenumber count that is not a whole number >= 1 or
    that exceeds OASES' wavenumber array bound NP (``oases/src/compar.f:37-38``,
    :data:`~uacpy.io.oases_writer._OASES_MAX_WAVENUMBERS`); ``None``
    (OASES' automatic sampling) passes."""
    if n_wavenumbers is None:
        return
    whole_count('n_wavenumbers', n_wavenumbers, 1)
    if int(n_wavenumbers) > _OASES_MAX_WAVENUMBERS:
        raise ConfigurationError(
            f"n_wavenumbers={int(n_wavenumbers)} exceeds OASES' wavenumber "
            f"array bound NP = {_OASES_MAX_WAVENUMBERS} (oases/src/compar.f:37-38 "
            f"`NPEXP = 16, NP = 2**NPEXP`).",
            remediation=(f"Pass n_wavenumbers <= {_OASES_MAX_WAVENUMBERS}, "
                         f"or None for OASES' automatic sampling."))


#: Each program's manual on a pinned wavenumber count: where ``NW`` places
#: its samples (equidistantly over the window CMIN/CMAX set), and where the
#: manual offers automatic sampling instead (OAST and OASP: "correct answers
#: in the first attempt without the usual convergence testing"; OASSP: "as
#: for the other OASES modules").
_PINNED_COUNT_MANUAL = {
    'OAST': ('oast.tex:522-526', 'oast.tex:543-546'),
    'OASP': ('oasp.tex:642-650', 'oasp.tex:671-674'),
    'OASSP': ('oassp.tex:479-488', 'oassp.tex:511-515, which refers to the '
                                    'other modules\' oasp.tex:671-674'),
}


def _pinned_count_note(model_name, n_wavenumbers) -> Optional[str]:
    """The info-level note a pinned ``n_wavenumbers`` gets on OAST, OASP
    and OASSP (:data:`_PINNED_COUNT_MANUAL`), or ``None`` when the count is
    OASES' automatic sampling; the value reaches the deck as given. OASN
    gives its own note at the same level (:class:`~uacpy.models.OASN`: its
    pinned count also sets the integration window)."""
    if n_wavenumbers is None:
        return None
    placed, automatic = _PINNED_COUNT_MANUAL[model_name]
    return (f"{model_name}(n_wavenumbers={n_wavenumbers}): a pinned count "
            f"places its samples equidistantly over the wavenumber window "
            f"c_low and c_high set ({placed}), where automatic sampling "
            f"(n_wavenumbers=None) is what the manual offers to obtain "
            f"correct answers \"without the usual convergence testing\" "
            f"({automatic}): double n_wavenumbers until the field stops "
            f"changing.")


#: OASES output letters that are not a pressure: vertical and horizontal
#: particle velocity, radial stress and the stress equivalent of the shear
#: potential (GETOPT: V→2 H→3 R→5 S→7; unoast31.f:981-1016,
#: unoasp22.f:887-918). uacpy reads the bulk pressure ``'K'``
#: (:data:`~uacpy.core.engine_defaults.OASES_PRESSURE_LETTER`, IOUT 6) or the
#: normal stress ``'N'`` (IOUT 1), which is ``-p`` in a fluid.
_NON_PRESSURE_OUTPUT_LETTERS = frozenset('VHRS')

#: OASP/OASSP letters whose ``.trf`` is not the one pressure uacpy reads: the
#: non-pressure output parameters, each adding an ``IOUT`` slot to every
#: ``.trf`` record, and
#: ``'U'`` (DECOMP), which splits the output over five files opened one unit apart
#: (unoassp30.f:459-465, :505-508). Chased through KERDEC
#: (oasaun31.f:1626-1648) and the buffer→unit pairing at oasvun31.f:779,
#: :862-869 and unoassp30.f:725-726, the DATA order at unoassp30.f:48-52 is
#: the correct one: trf = total, trfdc = down-going compressional, trfds =
#: down-going shear, trfuc = up-going compressional, trfus = up-going shear;
#: oassp.tex:185-189 has 'ds' and 'uc' swapped.
_TRF_OTHER_OUTPUT_LETTERS = _NON_PRESSURE_OUTPUT_LETTERS | {'U'}

#: Per program, the source lines each refusal of
#: :func:`_reject_unreadable_trf_options` cites.
_TRF_OPTION_SOURCES = {
    'OASP': {'output': 'unoasp22.f:887-918, :455-456',
             'O': 'unoasp22.f:372-373',
             't': 'unoasp22.f:178-189, :1028-1029',
             'J': 'unoasp22.f:288-296, :372-373'},
    'OASSP': {'output': 'unoassp30.f:897-932, :1004, :505-508',
              'O': 'unoassp30.f:983, :382-385',
              't': 'unoassp30.f:233-245, :1023-1024',
              'J': 'unoassp30.f:285-290, :382-385'},
}


def _reject_two_pressure_letters(model_name: str, chars, cite: str,
                                 what: str) -> None:
    """Refuse an option line carrying both ``'N'`` and ``'K'``: two output
    parameters, so ``what`` (the TL curves or the ``.trf`` records) holds
    two quantities where uacpy reads one."""
    if {'N', 'K'} <= set(chars):
        raise ConfigurationError(
            f"{model_name}: the option line carries both 'N' (normal "
            f"stress) and 'K' (bulk pressure), two output parameters, so "
            f"{what} ({cite}); uacpy reads one of them as the pressure.",
            remediation="Keep 'K' (the default, the bulk pressure) or 'N' "
                        "(the normal stress, -p in a fluid).")


def _reject_unreadable_oast_options(options) -> None:
    """Refuse an OAST option line whose TL table is not the pressure the
    Field labels: an output parameter of
    :data:`_NON_PRESSURE_OUTPUT_LETTERS` (velocity or stress), or ``'N'``
    and ``'K'`` together. ``options`` is the line the deck carries; the
    derived one hits neither."""
    chars = set(str(options)) - set(' \t\n')
    other = sorted(_NON_PRESSURE_OUTPUT_LETTERS & chars)
    if other:
        raise ConfigurationError(
            f"OAST: options={options!r} carries {other}, output "
            f"parameter(s) that are not a pressure (GETOPT, "
            f"unoast31.f:981-1016); their TL would be returned labelled "
            f"kind='pressure'.",
            remediation="Drop these letters: the default option line "
                        "returns the bulk pressure 'K'.")
    _reject_two_pressure_letters(
        'OAST', chars, 'unoast31.f:631-638',
        "the .plp holds two TL curves per receiver")


def _require_mean_field_rhs(model_name: str, mean, mean_options: str,
                            cite: str, example: str) -> None:
    """Refuse a supplied mean-field producer whose option line has no
    ``'s'``: without it the producer opens and writes no ``.rhs`` (``cite``),
    the only input the scattering run reads. ``mean_options`` is the line
    the producer will write; ``example`` the remedy's producer call. The
    option line is the caller's, so it is refused, not amended."""
    if 's' not in set(str(mean_options)) - set(' \t\n'):
        raise ConfigurationError(
            f"{model_name}(mean_field={type(mean).__name__}(options="
            f"{mean_options!r})): the producer's option line must contain "
            f"'s' or it writes no .rhs ({cite}), which is what "
            f"{model_name} reads.",
            remediation=f"e.g. {model_name}(mean_field={example}).",
        )


def _reject_unreadable_trf_options(model_name: str, options,
                                   n_wavenumbers) -> None:
    """Refuse raw ``options`` letters of an OASP or OASSP run whose ``.trf``
    uacpy cannot read back as the field it labels.

    Each letter writes a valid OASES deck — the public writers write it —
    but its ``.trf`` no longer means what uacpy's Field says:

    * an output parameter other than ``'K'`` or ``'N'``
      (:data:`_TRF_OTHER_OUTPUT_LETTERS`) is not the pressure
      (:func:`~uacpy.io._parsers.oasp_trf_pressure` reads those two), and
      ``'K'`` with ``'N'`` writes two parameters per record;
    * ``'O'`` moves the frequency integration onto a complex contour,
      ``Im(omega) = -ln(50)·Δf``, which uacpy's time-series synthesis does
      not undo;
    * ``'t'`` (tau-p) replaces the receiver range axis with slowness in
      s/km (``r0 = 1e3/cmaxin``), which uacpy would label range in metres,
      and OASP writes ``1.0E6`` times the field there (oasiun23.f:452-453);
    * no ``'J'`` under automatic wavenumber sampling (``n_wavenumbers``
      unset) engages the same complex frequency contour as ``'O'``.

    ``options`` unset (the models' own option line) hits none of them.
    GETOPT scans the line character by character (``unoasp22.f:871-873``),
    ignoring blanks, so the tests are on the character set.
    """
    if not options:
        return
    cites = _TRF_OPTION_SOURCES[model_name]
    chars = set(str(options)) - set(' \t\n')
    other = sorted(_TRF_OTHER_OUTPUT_LETTERS & chars)
    if other:
        raise ConfigurationError(
            f"{model_name}.run: options {other} ask for an output parameter "
            f"other than the pressure ({cites['output']}); uacpy reads the "
            f"bulk pressure 'K' or the normal stress 'N' (-p in a fluid) "
            f"from the .trf, as the pressure.",
            remediation="Pass options without these letters (the default "
                        "option line returns the bulk pressure 'K').")
    _reject_two_pressure_letters(
        f"{model_name}.run", chars, cites['output'],
        "every .trf record holds two components")
    if 'O' in chars:
        raise ConfigurationError(
            f"{model_name}.run: option 'O' (complex frequency integration) "
            f"bakes an exp(-ln(50)·Δf·t) contour into the spectrum "
            f"({cites['O']}) that uacpy's time-series synthesis does not "
            f"undo.",
            remediation="Drop 'O' (the default option line uses a real "
                        "frequency axis).")
    if 't' in chars:
        raise ConfigurationError(
            f"{model_name}.run: option 't' (tau-p seismograms) replaces the "
            f"receiver range axis with slowness ({cites['t']}), which "
            f"uacpy's Field has no coordinate for and would mislabel as "
            f"range.",
            remediation="Drop 't'.")
    if 'J' not in chars and n_wavenumbers is None:
        raise ConfigurationError(
            f"{model_name}.run: a custom options string without 'J' enables "
            f"the complex frequency contour (OMEGIM≠0) under automatic "
            f"wavenumber sampling ({cites['J']}), which uacpy's time-series "
            f"synthesis cannot undo.",
            remediation="Add 'J', or pin n_wavenumbers >= 1.")


def resolve_integration_window(model_name, window, n_wavenumbers, manual):
    """``(pair, notice)`` for the deck's wavenumber line: ``pair`` is the
    ``(IC1, IC2)`` it carries — the given ``window``
    (:func:`~uacpy.io.oases_writer.refuse_unusable_integration_window`), else
    the whole window ``(1, NW)`` under a pinned ``n_wavenumbers``, else
    ``None`` (automatic sampling, where OASES sets the pair) — and
    ``notice``, which says a given window has no effect under automatic
    sampling (``manual``: the program's own line), else ``None``."""
    pair = refuse_unusable_integration_window(model_name, window,
                                              n_wavenumbers)
    pinned = n_wavenumbers is not None and n_wavenumbers > 0
    notice = None
    if pair is not None and not pinned:
        notice = message_notice(
            f"{model_name}(integration_window={pair}) has no effect: with "
            f"n_wavenumbers unset OASES samples automatically, where IC1 "
            f"and IC2 have no effect ({manual}). Pin n_wavenumbers to "
            f"integrate over the window.", FallbackWarning)
    if pair is None and pinned:
        pair = (1, int(n_wavenumbers))
    return pair, notice


def _offset_ignored_notice(model_name, integration_offset, n_wavenumbers,
                           options, offset_letters='J',
                           auto_sampling_zeroes_offset=True):
    """The stage-3 notice that a caller's contour offset never reaches the
    integration, or ``None``. ``options`` is the option line the run's deck
    carries; every run's settings say it, since the knobs can be reassigned
    between runs.

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
        return None
    reasons = []
    if auto_sampling_zeroes_offset and n_wavenumbers is None:
        reasons.append(
            f"automatic wavenumber sampling (n_wavenumbers=None) zeroes "
            f"the contour offset and applies the kernel's own default — pin "
            f"n_wavenumbers >= 1 to use the value")
    if not set(str(options)) & set(offset_letters):
        reasons.append(
            f"the option line for this run ({str(options)!r}) carries none of "
            f"{sorted(offset_letters)}, so the binary zeroes the offset and "
            f"reads a frequency line without it — add 'J' (complex "
            f"integration contour) to use the value")
    if not reasons:
        return None
    return message_notice(
        f"{model_name}(integration_offset={integration_offset:g}) has no "
        f"effect: " + "; ".join(reasons) + ". Drop integration_offset to "
        f"silence this.", FallbackWarning)


#: The settings fields of OASS and OASSP that ``from_run_settings`` does not
#: pin: the option line (composed from the other knobs) and the deck's
#: ``spectral_exponent``, OASES' own 2.5 when unset, which as a pin under the
#: Gaussian spectrum would draw the notice that the run does not use it.
SCATTERING_UNPINNED_FIELDS = frozenset({'options', 'spectral_exponent'})


def resolve_spectral_exponent(model_name, spectral_exponent, *, goff):
    """``(M, notice)``: the roughness power-spectrum exponent the deck
    carries — the pinned value, else
    :data:`~uacpy.core.engine_defaults.OASSP_SPECTRAL_EXPONENT` — and, when
    one is pinned under the Gaussian spectrum (``goff`` False), the notice
    that the run does not use it: the Gaussian ``P`` reads only the
    correlation length (``oaseun31.f:3689-3692``; oassp.tex:364-366
    "Insignificant for Gaussian spectrum (option g not specified) but a value
    must be given")."""
    if spectral_exponent is None:
        return OASSP_SPECTRAL_EXPONENT, None
    notice = None if goff else (
        f"{model_name}(spectral_exponent={float(spectral_exponent):g}) is "
        f"not used: the Gaussian roughness spectrum reads only the "
        f"correlation length (oaseun31.f:3689-3692). "
        f"roughness_spectrum='goff-jordan' is the spectrum that takes it.")
    return float(spectral_exponent), notice


def _oases_binary(name: str) -> BinarySearch:
    """Where an OASES binary is looked for: the ``<name>_bash`` wrapper
    first, then ``name``, under ``uacpy/bin/oases``, ``uacpy/bin/oalib``
    and ``uacpy/third_party/oases``."""
    return BinarySearch((f'{name}_bash', name), ('oases', 'oalib'), 'oases')


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
    (:func:`~uacpy.core.absorption._warn_if_band_absorption_frozen`)
    reports the gap over the band and the water column."""
    _warn_if_band_absorption_frozen(
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
