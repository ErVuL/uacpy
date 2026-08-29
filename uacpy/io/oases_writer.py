"""
OASES Input File Writers

This module provides functions for writing input files for OASES models:
- OAST: Transmission loss module (wavenumber integration)
- OASN: Noise/covariance module (also used for normal mode computation)
- OASR: Plane-wave reflection-coefficient module
- OASP: Pulse / broadband transfer-function module
- OASS: Reverberation / scattered-field statistics from rough interfaces
- OASSP: Broadband realizations of the scattered field

OASES (Ocean Acoustics and Seismic Exploration Synthesis) was developed by
Henrik Schmidt at MIT.

References:
    Schmidt, H. OASES Version 2.1 User Guide and Reference Manual (bundled
    under ``third_party/oases``). The public OASES release is 3.1 but this
    distribution ships with the 2.1 source tree — per the bundled README.
"""

import warnings
from pathlib import Path
from typing import (Callable, List, Mapping, Optional, TextIO,
                    Tuple, Union)
import numpy as np

from uacpy.core.environment import BoundaryProperties, Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.exceptions import ConfigurationError, UnsupportedFeatureError
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.io.utils import (
    _collapsed_pair_index,
    equally_spaced,
    # Shared with the Bellhop and multi-profile writers; the module-private
    # alias keeps the six OASES writers' call sites unchanged.
    reject_unknown_kwargs as _reject_unknown_kwargs,
)
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


#: GETOPT reads the option record as FORMAT(40A1) in all four programs, so a
#: letter past column 40 is discarded without a diagnostic.
_OASES_OPTION_LINE_CHARS = 40


#: Title characters each binary can take. The manuals all say 80
#: (``oast.tex:22``, ``oasn.tex:23``, ``oasp.tex:76``, ``oasr.tex:27``,
#: ``oass.tex:48``) and every program reads the record as ``20A4``, so 80 is
#: right for all but one. OAST is the exception: ``unoast31.f:119`` re-emits
#: the title as ``write(ctitle,'(a,a)') atitle(1:ltit),' - '`` into a
#: ``character*80`` buffer (``:37``), so it needs ``ltit + 3 <= 80``. Measured
#: against the binary: a 77-character title runs, 78 dies with a Fortran
#: "Error termination" and exit 2, exactly where that arithmetic predicts.
#: OASP was checked at 78 and 140 characters and is unaffected.
_OASES_TITLE_CHARS = 80
_OAST_TITLE_CHARS = 77


def _write_oases_header(
    f: TextIO, env: Environment, options: str, fallback_title: str,
    *, title_chars: int = _OASES_TITLE_CHARS,
) -> None:
    """Write OASES Block I (title) + Block II (options).

    Every program reads these two first and in this order: the title as a
    whole record (unoast31.f:117, unoasn22.f:126, unoasp22.f:121,
    unoasr21.f:77), then GETOPT's ``READ(1,200) OPT`` /
    ``200 FORMAT(40A1)`` (unoast31.f:978-979, unoasn22.f:635-636,
    unoasp22.f:871-872, unoasr21.f:347-348). The option record is 40 single
    characters, so whitespace in it is insignificant — see
    :func:`_oases_option_chars`.
    """
    title = env.name if env.name else fallback_title
    if len(str(options)) > _OASES_OPTION_LINE_CHARS:
        raise ConfigurationError(
            f"OASES option line is {len(str(options))} characters, over the "
            f"{_OASES_OPTION_LINE_CHARS} GETOPT reads: its "
            f"'READ(1,200) OPT / 200 FORMAT(40A1)' takes exactly 40 single "
            f"characters and discards the rest of the record, so "
            f"{str(options)[_OASES_OPTION_LINE_CHARS:]!r} would never reach "
            f"the option scan — and unlike an unknown letter it produces no "
            f"'UNKNOWN OPTION' diagnostic.",
            remediation="Drop options, or remove the spaces between letters "
                        "(the record is 40 single characters, so whitespace "
                        "in it is insignificant).")
    title = str(title).replace("\n", " ").replace("\r", " ")
    if len(title) > title_chars:
        warnings.warn(
            f"OASES deck title is {len(title)} characters; truncated to "
            f"{title_chars}. Every program reads the record as '20A4' (80 "
            f"characters), and OAST needs three more for the ' - ' it appends "
            f"into its own character*80 buffer (unoast31.f:37,119) — a "
            f"78-character title kills it with a Fortran 'Error termination' "
            f"and exit 2, which surfaces only as a failed run.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
        title = title[:title_chars]
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
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
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
        UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )


def _extract_bottom_props(bottom: BoundaryProperties) -> dict:
    """Geoacoustic values of the lower half-space row for the OAS* writers.

    Branches on ``acoustic_type`` — the placeholders a ``'vacuum'`` /
    ``'rigid'`` / ``'file'`` boundary carries must never be written as if
    they were a real sediment. A vacuum maps to OASES' own pressure-release
    half-space: INENVI flags any layer with ``|V(M,2)| < 1e-10`` as
    ``LAYTYP = -1`` (oaseun31.f:125-126), so an all-zero row is the deck's
    native spelling of it. A rigid or tabulated boundary has no OASES deck
    spelling, so those raise instead of silently substituting a 1600 m/s
    fluid.
    """
    kind = bottom.acoustic_type
    if kind == 'vacuum':
        return dict(rho=0.0, c_p=0.0, c_s=0.0, alpha_p=0.0, alpha_s=0.0)
    if kind != 'half-space':
        raise UnsupportedFeatureError(
            'OASES', f"a bottom with acoustic_type={kind!r} — the OAS* decks "
            f"express the seabed as geoacoustic layers, and this type has no "
            f"layer spelling (a vacuum does: an all-zero row)",
            alternatives=['a geoacoustic half-space bottom',
                          'Scooter / Kraken / Bellhop, which take '
                          "'rigid'/'file' natively"],
        )
    return dict(
        rho=bottom.density,
        c_p=bottom.sound_speed,
        c_s=bottom.shear_speed,
        alpha_p=bottom.attenuation,
        alpha_s=bottom.shear_attenuation,
    )


#: Margin within which INENVI folds an n²-linear layer back to isovelocity:
#: ``abs(v(m,2)+v(m,3)).lt.1e-2`` at oaseun31.f:181. The test runs on the
#: numbers the deck carries, so it is applied here to the FORMATTED speed
#: tokens rather than to the caller's floats.
_OASES_ISOVELOCITY_FOLD_MS = 1e-2


def _check_water_layer_thickness(ssp_rows) -> None:
    """Reject two SSP samples that write the same depth token.

    The depth column is ``%.2f`` while ``SoundSpeedProfile`` requires
    adjacent depths to differ by only 1e-6 m
    (``DECK_DEPTH_RESOLUTION_M``, core/ssp.py), so a sharp thermocline
    written as 30.0 -> 30.001 m reaches the deck as two records at the same
    depth. INENVI's only sequence test is ``v(m,1).lt.v(m-1,1)``
    (oaseun31.f:58-61), which EQUAL depths pass; PINIT2 then computes
    ``THICK(I)=V(I+1,1)-V(I,1)`` = 0 (:1508) and divides by it at
    ``GRAD=(AKL2-AKU2)/THICK(I)`` (:1537). Measured on a four-sample profile
    stepping 30.0 -> 30.004 m: oast prints '**** EXECUTION TERMINATED ***
    / **** ERROR NUMBER : 1', signals IEEE_DIVIDE_BY_ZERO and leaves a
    0-byte .plt.

    Only the n²-linear records reach that division, so only they are
    rejected:

    * A pair whose FORMATTED speeds agree within
      ``_OASES_ISOVELOCITY_FOLD_MS`` is folded to LAYTYP 1 at
      oaseun31.f:181-185 before PINIT2 runs — measured, the same deck then
      finishes normally — and the fold costs nothing, the two samples being
      one sample at the deck's resolution.
    * The deepest record carries CS = 0, which fails the
      ``V(M,3).LT.0`` gate of the gradient branch (:160) and takes the
      isovelocity branch instead. Its depth is shared with the seafloor
      record below it by construction — that is the water/sediment
      interface — and that pairing is why the check stops one record short.

    The sediment column's counterpart is the sub-5 mm warning in
    :func:`_emit_bottom_layers`: those records are isovelocity, so a
    collapsed one is dropped rather than divided by.
    """
    n_rows = len(ssp_rows)
    for i in range(n_rows - 1):
        d, c = float(ssp_rows[i][0]), float(ssp_rows[i][1])
        d_next, c_next = float(ssp_rows[i + 1][0]), float(ssp_rows[i + 1][1])
        if f"{d:.2f}" != f"{d_next:.2f}":
            continue
        cc = float(f"{c:.2f}")
        cs = float(f"{-abs(c_next):.2f}")
        if abs(cc + cs) < _OASES_ISOVELOCITY_FOLD_MS:
            continue
        raise ConfigurationError(
            f"SSP samples at {d:.6g} m ({c:g} m/s) and {d_next:.6g} m "
            f"({c_next:g} m/s) both write as {d:.2f} m in the OASES layer "
            f"record, whose depth column resolves 0.01 m. OASES accepts the "
            f"equal depths (oaseun31.f:58-61 rejects only a DECREASING one) "
            f"and then computes a zero layer thickness for the sound-speed "
            f"gradient between them, ``GRAD=(AKL2-AKU2)/THICK(I)`` with "
            f"THICK = 0 (oaseun31.f:1508, :1537), which stops the run with "
            f"'ERROR NUMBER : 1' and no output.",
            remediation="Separate the two SSP depths by more than 0.01 m, or "
                        "drop one of them — at 0.01 m apart they are the same "
                        "sample to every OASES deck.",
        )


def _emit_water_layers(
    f: TextIO, ssp_rows, *, surface_roughness: float, extra_columns: int,
    surface_suffix: Optional[str] = None,
) -> None:
    """Emit one OASES layer record per SSP sample, top down.

    CC > 0 with CS < 0 declares an n²-linear (Airy) layer whose speed runs
    from CC at its own top to |CS| at the top of the next layer
    (oaseun31.f:160, labelled ``n^2 linear layer (Airy)`` at :157). It is
    genuinely n² and not c: PINIT2 takes k² at the two interfaces and
    linearises between them, ``GRAD=(AKL2-AKU2)/THICK(I)``
    (oaseun31.f:1527-1537). Carrying the NEXT sample's speed in the shear
    column is therefore how a sampled profile is handed over as a
    continuous one. The deepest water layer leaves CS = 0 so the seabed
    record below it terminates the column, and a layer whose |CS| equals
    its CC is folded back to isovelocity by INENVI (oaseun31.f:181-182).

    The fixed ``0.0 0 1.0`` are AC, AS and RO. AC = 0 is not "lossless": it
    is what hands the water column to OASES' own Skretting-Leroy attenuation
    (oaseun31.f:1516-1521) — see :func:`_warn_volume_attenuation_ignored`.

    ``surface_roughness`` rides on the FIRST record's RG. These records are
    layers 2..N of the deck, so the first one's RG is ROUGH(2) — the
    sea surface — under the top-of-layer rule of
    :func:`bottom_interface_roughness`. The remaining records' interfaces
    are SSP sample boundaries inside the water and carry no roughness.

    ``extra_columns`` is the inert padding past column 7 that this program's
    deck carries — see :func:`_emit_bottom_layers`.

    The depth column is validated before the first record goes out, so a
    profile OASES cannot integrate never reaches the file as a half-written
    layer block — see :func:`_check_water_layer_thickness`.
    """
    _check_water_layer_thickness(ssp_rows)
    # Anchor the column at z = 0. Each record carries its own layer's TOP
    # depth (oaseun31.f:54), and INENVI then places the vacuum upper
    # half-space at the FIRST water record's depth — `if (m.le.2) then /
    # v(1,1)=v(2,1)` (oaseun31.f:56-57). A profile whose shallowest sample is
    # at 10 m therefore moved the pressure-release surface down to 10 m and
    # modelled a 90 m waveguide instead of the 100 m `env.depth` describes:
    # measured on a 100 m Pekeris guide at 100 Hz, median |dTL| 4.23 dB and
    # max 39.97 dB over 2064 .plt values against the same profile anchored at
    # 0, both at exit 0 with no warning. A source shallower than the first
    # sample lands INSIDE the vacuum ("LAYER 1", ERROR NUMBER 1, a 0-byte
    # .plt at exit 0).
    #
    # `oalib_writer` documents and avoids exactly this for the AT decks
    # (:930-938), and OAST's and OASS's isovelocity branches already hardcode
    # "0.00" — so z = 0 is the intended anchor and only the sampled-profile
    # path missed it, which also meant one Environment produced two different
    # waveguides depending on which OASES program was written.
    ssp_rows = [tuple(r) for r in ssp_rows]
    if ssp_rows and float(ssp_rows[0][0]) > 0.0:
        ssp_rows.insert(0, (0.0, float(ssp_rows[0][1])))
    trail = ' 0' * extra_columns
    n_rows = len(ssp_rows)
    for i in range(n_rows):
        d, c = ssp_rows[i]
        cs = -abs(float(ssp_rows[i + 1][1])) if i < n_rows - 1 else 0.0
        rg = float(surface_roughness) if i == 0 else 0.0
        if i == 0:
            _warn_rough_gradient_surface(rg, float(c), cs)
            if surface_suffix is not None:
                # The sea surface is the scattering interface: its record
                # carries the nine-token -|RG| CL M form instead of a bare
                # RMS, and the trailing padding column is dropped because
                # INENVI re-reads the record as nine (oaseun31.f:91-93).
                f.write(f"{d:.2f} {c:.2f} {cs:.2f} 0.0 0 1.0"
                        f"{surface_suffix}\n")
                continue
        f.write(f"{d:.2f} {c:.2f} {cs:.2f} 0.0 0 1.0 {rg:.4f}{trail}\n")


def _warn_rough_gradient_surface(rg: float, c_top: float, cs: float) -> None:
    """Warn when a rough sea surface caps an n²-linear first water layer.

    INENVI rejects the pairing at oaseun31.f:382-388: any interface with
    ``ROUGH2 > 1e-10`` whose layer above or below is LAYTYP 2 (the gradient
    layer set at :189) prints '*** SURFACE ROUGHNESS NOT ALLOWED BETWEEN
    LAYERS WITH SOUND SPEED GRADIENT ... THIS RUN MUST BE FOR VOLUME
    SCATTERING ONLY ***'. Its ``STOP`` is commented out at :388, so OASES
    goes on to apply the Kirchhoff perturbation anyway and the message only
    reaches the ``.prt``. Surface the caveat here instead.

    A first layer whose ``|CS|`` matches its ``CC`` within 1 cm/s is folded
    back to isovelocity at :181-182 and never reaches the test.
    """
    if abs(rg) <= 1e-5 or abs(c_top + cs) < 1e-2:
        return
    warnings.warn(
        f"env.surface.roughness = {rg:g} m caps a water column whose first "
        f"layer has a sound-speed gradient ({c_top:g} -> {abs(cs):g} m/s). "
        f"OASES writes '*** SURFACE ROUGHNESS NOT ALLOWED BETWEEN LAYERS "
        f"WITH SOUND SPEED GRADIENT ***' to the .prt for that pairing "
        f"(oases/src/oaseun31.f:382-388) and applies its Kirchhoff "
        f"perturbation regardless, so the scattered field is approximate.",
        UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )


def _check_roughness_sign(label: str, rg: float) -> float:
    """Raise on a negative RG, which changes the layer record's arity.

    INENVI branches on the sign: ``rough(m).lt.-1e-10`` backspaces and re-reads
    the record as eight items with CLEN (oaseun31.f:72-75), then reads it a
    third time as ``V(1..6)`` plus three roughness-spectrum values (:91-93).
    The records this writer emits carry at most eight tokens, so that third
    read runs off the end and takes its ninth value from the record below —
    every subsequent READ in the deck is then shifted. The carriers reject a
    negative roughness at construction; this catches one assigned afterwards.
    """
    value = float(rg)
    if value >= 0.0:
        return value
    raise ConfigurationError(
        f"{label} = {value:g} m is negative, but OASES reads a negative RG as "
        f"the flag for a Goff-Jordan roughness spectrum and re-reads the layer "
        f"record as nine tokens (oases/src/oaseun31.f:72-93); this writer emits "
        f"eight, so the read would run into the record below and shift the rest "
        f"of the deck.",
        remediation="Pass an RMS roughness >= 0 m (0 is smooth).",
    )


def _surface_roughness(env: Environment) -> float:
    """RMS roughness (m) of the sea surface, i.e. OASES' ROUGH(2)."""
    return _check_roughness_sign(
        "env.surface.roughness",
        float(getattr(env.surface, 'roughness', 0.0) or 0.0))


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

    OASES interface format: D CC CS AC AS RO RG (oast.tex:42-48). INENVI
    consumes exactly those seven —
    ``READ(1,*) (V(M,N),N=1,6),ROUGH(M)`` at oaseun31.f:54 — and a
    list-directed read drops whatever else the record holds, so the
    ``extra_columns`` trailing zeros are inert padding that keeps the record
    shaped like the manual's column table.

    RG < 0 is the exception, and it changes the record's arity: INENVI
    backspaces and re-reads the record as ``V(1..6), ROUGH, CLEN``, then
    backspaces again and reads it as ``V(1..6)`` plus three roughness-spectrum
    values (oaseun31.f:72-93). So a negative RG obliges the record to carry
    nine tokens — RG, the correlation length and the spectral exponent —
    which only the OASR ``suffix_fn`` produces. Every RG reaching the
    ``extra_columns`` path must be >= 0 or the third read runs past the end of
    the record and takes its ninth value from the block below.

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
        # Inert padding after RG; the manual labels the 8th column CL
        # (oast.tex:42), read only when RG < 0.
        trail = ' 0' * extra_columns
        rgs = bottom_interface_roughness(env)

        def suffix_fn(i):
            k = i - iface_start
            rg = rgs[k] if 0 <= k < len(rgs) else 0.0
            return f" {rg:.4f}{trail}"

    iface = iface_start
    if env.has_layered_bottom:
        lb = env.bottom.columns[0]
        current_depth = water_depth
        for layer in writable_layers(lb):
            layer_as = getattr(layer, 'shear_attenuation', 0.0)
            f.write(f"{current_depth:.2f} {layer.sound_speed:.2f} "
                    f"{layer.shear_speed:.2f} {layer.attenuation:.3f} "
                    f"{layer_as:.3f} {layer.density:.2f}{suffix_fn(iface)}\n")
            # The depth column is "%.2f", so a layer thinner than 5 mm puts
            # the next record at the SAME formatted depth. INENVI rejects only
            # a decreasing depth (oaseun31.f:58-61), so equal depths pass and
            # THICK(I) comes out 0 (:1508) — the layer is silently dropped.
            # Nothing measurable is lost at that thickness (it is far below
            # lambda/100 in any usable band); the warning exists because the
            # request vanishes without trace.
            if f"{current_depth:.2f}" == f"{current_depth + layer.thickness:.2f}":
                warnings.warn(
                    f"OASES layer record: a sediment layer {layer.thickness:g} "
                    f"m thick is below the 0.01 m resolution of the deck's "
                    f"depth column, so it is written at the same depth as the "
                    f"layer below and OASES computes THICK=0 "
                    f"(oaseun31.f:1508) — the layer is dropped.",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
                )
            current_depth += layer.thickness
            iface += 1
        # Deepest halfspace below all sediment layers.
        hs = getattr(lb, 'halfspace', None)
        if hs is not None and getattr(hs, 'acoustic_type', None) is not None:
            # Same acoustic_type branch as _extract_bottom_props: a vacuum
            # termination writes the all-zero LAYTYP=-1 row, and rigid/file
            # raise — never the construction placeholders (measured ~10 dB
            # of bottom loss per bounce when the placeholder was written).
            props = _extract_bottom_props(hs)
            c_p, c_s = props['c_p'], props['c_s']
            alpha_p, alpha_s = props['alpha_p'], props['alpha_s']
            rho = props['rho']
        elif hs is not None:
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


def bottom_interface_roughness(env: Environment) -> List[float]:
    """RMS roughness of each interface :func:`_emit_bottom_layers` writes, in order.

    An OASES layer record opens with the depth of its own upper interface
    (``doc/oast.tex:42``) and carries that interface's roughness in column 7
    (``doc/oast.tex:48``, read at ``src/oaseun31.f:54``). So a record's RG is
    the roughness at the TOP of its layer: the first sediment layer's is the
    seafloor, and the half-space's is the base of the stack. A column with no
    layers has one record, and its RG is the seafloor.
    """
    if not env.has_layered_bottom:
        hs = env.bottom.halfspace_at(range=0.0)
        return [_check_roughness_sign("env.bottom halfspace roughness",
                                      getattr(hs, 'roughness', 0.0) or 0.0)]
    column = env.bottom.columns[0]
    rgs = [_check_roughness_sign(f"env.bottom layer {i} roughness",
                                 layer.roughness)
           for i, layer in enumerate(writable_layers(column))]
    hs = getattr(column, 'halfspace', None)
    rgs.append(_check_roughness_sign("env.bottom halfspace roughness",
                                     getattr(hs, 'roughness', 0.0) or 0.0))
    return rgs


#: Smallest |RG| the OASES layer record can carry. The record formats RG with
#: "%.4f", so anything below this renders as "-0.0000" and fails INENVI's
#: ``rough(m).lt.-1e-10`` test (oaseun31.f:72) — the negative-RG flag is lost
#: and the interface silently becomes smooth.
_MIN_REPRESENTABLE_RG = 5e-5


def _make_roughness_tail(
    env: Environment,
    interface_roughness=None,
) -> Callable[[int], str]:
    """Build the ``suffix_fn`` :func:`_emit_bottom_layers` appends to each
    layer record — the interface-roughness tail.

    ``_check_roughness_sign`` rejects a negative RG for the writers that can
    only express an RMS value, which is right for them and wrong for the
    scattering modules: OASES flags "a correlation length and spectral
    exponent follow" precisely *by* making RG negative
    (``oaseun31.f:72-75``, ``:91-93``), so a rough interface with a finite
    correlation length is the nine-token form
    ``D CC CS AC AS RO -|RG| CL M``. Every writer that needs that arity gets
    it from here rather than reimplementing the split.

    ``interface_roughness`` is the per-interface override, indexed the way
    ``_emit_bottom_layers`` indexes interfaces; ``None`` entries and indices
    past its end fall back to the environment's own roughness so the deck
    describes the same interfaces the other writers would emit. Only the
    override can reach the CL / M form.

    Each entry may be a scalar (RMS only), a ``(RG, CL, M)`` tuple, or a dict
    keyed ``RG``/``CL``/``M`` (or the spelled-out
    ``roughness``/``correlation_length``/``spectral_exponent``).
    """
    if interface_roughness is None:
        interface_roughness = []
    env_rg = [0.0] + bottom_interface_roughness(env)

    is_mapping = isinstance(interface_roughness, Mapping)

    def _roughness_tail(i: int) -> str:
        def _from_env():
            return f" {env_rg[i]:.4f}" if 0 <= i < len(env_rg) else " 0"

        # A mapping is sparse — only the interfaces it names are overridden —
        # so membership decides, not length. Testing length on a mapping
        # silently drops every entry whose key is >= the number of entries,
        # which is the usual case when one deep interface is overridden.
        if is_mapping:
            if i not in interface_roughness:
                return _from_env()
        elif i < 0 or i >= len(interface_roughness):
            return _from_env()
        spec = interface_roughness[i]
        if spec is None:
            return _from_env()
        if isinstance(spec, (int, float)):
            return f" {float(spec):.4f}"
        if isinstance(spec, dict):
            rg = spec.get('RG', spec.get('roughness', 0.0))
            cl = spec.get('CL', spec.get('correlation_length', None))
            m = spec.get('M', spec.get('spectral_exponent', None))
        else:  # tuple / list
            rg = spec[0] if len(spec) > 0 else 0.0
            cl = spec[1] if len(spec) > 1 else None
            m = spec[2] if len(spec) > 2 else None
        if cl is None or m is None:
            return f" {float(rg):.4f}"
        # Negative RG is the flag that CL and M follow on the same record —
        # but only if it SURVIVES formatting. INENVI gates the extended
        # re-read on ``rough(m).lt.-1e-10`` (oaseun31.f:72), and "%.4f"
        # renders every |RG| < 5e-5 m as "-0.0000", which fails that test.
        # The deck still carries nine tokens so nothing shifts; INENVI simply
        # takes the seven-token read, sets clen(m)=0 (:102) and promotes it to
        # CLEN=1E6 (:111-114) — an infinite correlation length. Downstream
        # RG2 is 0 (oaseun31.f:379), the integration constant FF is
        # identically zero (oassun26.f:697), and the run exits 0 having
        # produced a flat clipped level. Refuse the value the format cannot
        # carry rather than emit a deck that means the opposite of the
        # request.
        if abs(float(rg)) < _MIN_REPRESENTABLE_RG:
            raise ConfigurationError(
                f"interface roughness |RG| = {abs(float(rg)):.3g} m is below "
                f"{_MIN_REPRESENTABLE_RG:g} m, the smallest magnitude the "
                f"OASES layer record's %.4f field can carry. It would be "
                f"written as -0.0000, which INENVI reads as a SMOOTH "
                f"interface with an infinite correlation length "
                f"(oaseun31.f:72, :111-114) — the scattering integral is then "
                f"identically zero and the run returns a flat level with no "
                f"error.",
                remediation=("Use a roughness of at least 1e-4 m, or model a "
                             "smooth interface by not requesting a roughness "
                             "spectrum at all."),
            )
        # CL rides the SAME "%.4f" as RG, and its failure is worse than a
        # lost flag: INENVI reads 0.0 and then promotes it to an INFINITE
        # correlation length (oaseun31.f:111-113 `IF (CLEN(M).EQ.0.0) …
        # CLEN(M)=1E6`). Measured, a CL=1e-5 deck produces a .rco
        # bit-identical to a hand-written CL=1e6 deck — 11 orders of
        # magnitude, exit 0, no warning; > 6.2 dB of reflection loss by
        # extrapolation from the 0.01 m case.
        if abs(float(cl)) < _MIN_REPRESENTABLE_RG:
            raise ConfigurationError(
                f"correlation_length = {float(cl):.3g} m is below "
                f"{_MIN_REPRESENTABLE_RG:g} m, the smallest magnitude the "
                f"OASES layer record's %.4f field can carry. It would be "
                f"written as 0.0000, which INENVI promotes to an INFINITE "
                f"correlation length of 1e6 m (oaseun31.f:111-113) — the "
                f"opposite of the request, with no error.",
                remediation="Use a correlation length of at least 1e-4 m.",
            )
        # A NEGATIVE CL selects INENVI's twelve-token VOLUME-scattering read
        # (oaseun31.f:77-79) from a record that carries nine. The
        # list-directed read runs on into the following record and the whole
        # deck shifts: measured, the source line is consumed and the binary
        # dies naming a frequency the user never set. This is the same class
        # _check_roughness_sign refuses for RG on the eight-token writers.
        if float(cl) < 0.0:
            raise ConfigurationError(
                f"correlation_length = {float(cl):.3g} m is negative. OASES "
                f"reads a negative CL as the flag for a twelve-token "
                f"volume-scattering record (oaseun31.f:77-79), so the "
                f"nine-token record written here would be over-read and the "
                f"rest of the deck consumed with it.",
                remediation="Pass a positive correlation length.",
            )
        return f" {-abs(float(rg)):.4f} {float(cl):.4f} {float(m):.4f}"

    return _roughness_tail


def _resolve_freq_sweep(
    writer: str, source: Source, freq_fallback: float
) -> Tuple[float, float, int]:
    """Resolve the (freq_min, freq_max, nfreq) sweep tuple from a Source.

    Multi-frequency sources advertise the actual sweep bounds; single-freq
    sources collapse to ``(freq, freq, 1)``. The fallback covers the rare
    case where ``source.frequencies`` is empty (callers pass the model's
    canonical centre frequency).

    Block III carries only ``(FREQ1, FREQ2, NFREQ)`` and the binaries walk it
    as ``FREQ = FREQ1 + (JJ-1)*DLFREQ`` (unoast31.f:395), so a non-uniform
    request cannot be honoured — the run would happen at a different set of
    frequencies than the one that was asked for, and nothing on disk would
    say so. Same rule as the uniform range axis (:func:`_oasp_range_axis`)
    and OASR's uniform angle axis.
    """
    freqs_arr = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
    if len(freqs_arr) > 1:
        steps = np.diff(freqs_arr)
        if not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-9):
            realised = freqs_arr[0] + np.arange(len(freqs_arr)) * (
                (freqs_arr[-1] - freqs_arr[0]) / (len(freqs_arr) - 1))
            raise ConfigurationError(
                f"{writer}: the deck evaluates a uniform frequency sweep "
                f"(FREQ1 + i*DLFREQ, unoast31.f:395) but source.frequencies "
                f"is not uniformly spaced (steps {steps.min():.4g}-"
                f"{steps.max():.4g} Hz). The run would happen at "
                f"{np.array2string(realised, precision=4)} Hz.",
                remediation=(
                    "Pass uniformly spaced frequencies (np.linspace), or run "
                    "one deck per frequency."),
            )
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
    """Write the OASES Block-III frequency-sweep line for OAST and OASN.

    Layout: ``FREQ1 FREQ2 NFREQ COFF [VREC]``. The 5-token Doppler form
    fires only when the lowercase ``'d'`` option is enabled, which is OAST's
    alone (unoast31.f:125-126); OASN has no Doppler read.

    COFF is read only when the complex contour is on — ``ICNTIN > 0`` from
    'J', or CFRFL from 'O' in OAST (unoast31.f:128-133, unoasn22.f:141-146).
    Writing it unconditionally is safe: the read is list-directed, so a
    program that wants three values takes three and discards the rest of the
    record. It does mean ``integration_offset`` reaches nothing without 'J',
    both programs zeroing OFFDB themselves in that branch.
    """
    if doppler:
        vrec_val = vrec if vrec is not None else 0.0
        f.write(
            f"{freq_min:.9f} {freq_max:.9f} {nfreq} "
            f"{integration_offset} {vrec_val}\n"
        )
    else:
        f.write(
            f"{freq_min:.9f} {freq_max:.9f} {nfreq} {integration_offset}\n"
        )


def oases_wavenumber_bounds(
    ssp_data: np.ndarray,
    *,
    ssp_factor: float = 0.9,
    cmax: float = 1e8,
) -> Tuple[float, float]:
    """Return the (cmin, cmax) wavenumber-integration bounds.

    The pair truncates the horizontal-wavenumber axis at
    ``k_max = 2*pi*f/cmin`` and ``k_min = 2*pi*f/cmax`` (oast.tex:505-515),
    so ``cmin`` must sit below the slowest phase speed in the problem for the
    steep and evanescent part of the spectrum to be sampled at all. The
    ``ssp_factor`` margin below ``min(SSP)`` (0.9 here, 0.95 in the OASN
    noise blocks) is uacpy's own choice — OASES states no rule for it.

    ``cmax`` is the "no upper limit" value; 1e8 is the one OASES substitutes
    itself when the deck's CMIN/CMAX signs conflict (unoast31.f:226,
    unoasp22.f:169).
    """
    cmin = float(ssp_data[:, 1].min()) * ssp_factor
    return cmin, float(cmax)


#: CMAX's format in the OAST and OASP/OASSP Block-VII records. Six
#: significant digits, not a fixed decimal count, because the same token has
#: to carry both a user's ``c_high`` (1550 m/s) and the 1e8 "no upper limit"
#: sentinel of :func:`oases_wavenumber_bounds`; ``%.6g`` writes those as
#: ``1550`` and ``1e+08``. The reads are list-directed (unoast31.f:213,
#: unoasp22.f:159), so no width is imposed. CMAX sets the LOWER edge of the
#: wavenumber integration, ``WK0 = 2*pi*f/CMAX``, and with a pinned NW it also
#: moves DLWVNO and hence the FFT range step, so rounding it is not cosmetic.
_OASES_CMAX_FMT = '{:.6g}'


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

# OASES' transform-length bound for the time axis: `NX = MIN0(NX, 2*NP)`
# (unoasp22.f:234, unoassp30.f:202) over the same NP, applied after NT has
# been rounded up to the next power of two. Silent — neither the clamp nor
# the resulting DLFREQ appears anywhere but stdout.
_OASES_MAX_TIME_SAMPLES = 2 * _OASES_MAX_WAVENUMBERS

# Replica-axis bound: OASN STOPs '>>> TOO MANY REPLICA POINTS <<<' when any
# of NSRCZ/NSRCX/NSRCY exceeds NSMAX = 201 (oases/src/compar.f:51,
# unoasn22.f:187-188) — a character STOP, so the binary exits 0 with no
# ``.rpo`` written.
_OASES_MAX_REPLICA_POINTS = 201

# Discrete-noise-source bound: NOIPAR dimensions ZDN/XDN/YDN/DNLEVDB to the
# same NSMAX = 201 (oases/src/noiprm.f:13-14, compar.f:51) and OASN STOPs
# '*** TOO MANY DISCRETE NOISE SOURCES ***' above it (oasnun22.f:372-373),
# again a character STOP that exits 0.
_OASES_MAX_DISCRETE_NOISE_SOURCES = 201

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


#: Decimals in every receiver-depth column these decks write, and the
#: resolution that implies. OASES reads them list-directed, so no width is
#: imposed; two decimals is what the manuals' worked decks carry.
_OASES_RECEIVER_DEPTH_DECIMALS = 2
_OASES_RECEIVER_DEPTH_RESOLUTION_M = 10.0 ** -_OASES_RECEIVER_DEPTH_DECIMALS


def _check_receiver_depth_tokens(depths, tokens, *, what: str) -> None:
    """Raise when two receiver depths write the same 0.01 m token.

    ``Receiver.depths`` need only be ``DECK_DEPTH_RESOLUTION_M`` = 1e-6 m
    apart (core/receiver.py), so an array spaced under a centimetre reaches
    OASES as one repeated depth: INREC reads the explicit list with a single
    ``READ(1,*) (RDC(JJ),jj=1,ir)`` and applies no duplicate test
    (oaseun31.f:1186), and INPRCV's array records are read the same way
    (oasnun22.f:36). OASES then evaluates the field at one depth several
    times over while ``read_oasp_trf`` / ``read_oasn_*`` hand the caller back
    the distinct depths it asked for, so the axis and the field stop
    describing the same array.

    The test is on the WRITTEN tokens rather than on the carrier's floats,
    for the reason ``oalib_writer.write_receiver_ranges`` gives for the AT
    decks: OASES reads the file, not the array.
    """
    bad = _collapsed_pair_index(tokens)
    if bad is None:
        return
    raise ConfigurationError(
        f"{what} {float(depths[bad]):g} m and {float(depths[bad + 1]):g} m "
        f"both write as {tokens[bad]} m at the deck's "
        f"{_OASES_RECEIVER_DEPTH_RESOLUTION_M:g} m resolution. OASES reads "
        f"the depth list with no duplicate test (oaseun31.f:1186), so it "
        f"would compute one depth twice while uacpy's readers return the two "
        f"distinct depths that were asked for.",
        remediation=(f"Separate the receiver depths by more than "
                     f"{_OASES_RECEIVER_DEPTH_RESOLUTION_M:g} m."),
    )


#: NW*C NW*D NW*E — the counts the OASN manual's worked deck uses for the
#: continuous / discrete / evanescent bands (oasn.tex:480).
_OASN_NOISE_SAMPLES = (400, 400, 100)


def _noise_cmin(kwargs, ssp_data) -> float:
    """Slow phase-speed edge for an OASN noise block: the model's ``c_low``
    when pinned, else 0.95*c_min. The 0.95 margin is uacpy's own — OASES
    states no rule for it; the manual's worked deck just pins CMINS = 1400
    m/s under a 1431 m/s water column (oasn.tex:438, :479)."""
    c_low = kwargs.get('c_low')
    if c_low:
        return float(c_low)
    return float(ssp_data[:, 1].min()) * 0.95


def _noise_cmax(kwargs) -> float:
    """Fast phase-speed edge: the model's ``c_high`` when pinned, else the
    manual's 1E8 ("no upper limit")."""
    c_high = kwargs.get('c_high')
    return float(c_high) if c_high else 1.0e8


#: Documented minimum for the discrete-band count, ``oasn.tex:313``
#: ("NWSD $\\ge 10$"). OASES never checks it; see :func:`_noise_nw`.
_OASN_MIN_DISCRETE_SAMPLES = 10


def _noise_nw(kwargs) -> str:
    """``NW*C NW*D NW*E`` wavenumber-sample counts for a noise block.

    Unlike the replica and discrete-source blocks, the surface- and deep-noise
    blocks have no automatic-sampling branch: NOIPAR reads three explicit counts
    and sums them into ``NWVNON``/``NWVNOP`` (oasnun22.f:312, :358). A negative
    count would make that total negative and the block would contribute nothing,
    so ``nw_samples <= 0`` falls back to the manual's counts.

    The three-band total is bounded by OASES' wavenumber array: NOIPAR
    STOPs with '*** TOO MANY SAMPLING POINTS ***' when it exceeds NP
    (oasnun22.f:319, :366) — no clamp, no output — so the 2.25x total this
    writer derives from a pinned ``nw_samples`` is checked here first.

    The DISCRETE band has a documented floor the binary does not enforce.
    ``oasn.tex:313`` writes the parameter as ``NWSD >= 10``, and nothing in
    ``unoasn22.f`` checks it — the only use of ``NWS(2)`` anywhere is as a
    DIVISOR, ``OFFDB = 60*V*(SLS(3)-SLS(2))/NWS(2)`` at ``:289``, the default
    complex-contour offset when the deck asks for one without a value. That
    numerator is the discrete band's slowness span, so the offset is
    proportional to the band's sampling INTERVAL: at NWSD = 1 the "interval"
    is the whole band, and the contour is pushed ten times further off the
    real axis than the manual's minimum intends — damping the modal poles the
    discrete band exists to resolve. Floored here rather than left to produce
    a quietly over-damped run.
    """
    nw = kwargs.get('nw_samples')
    if nw and int(nw) > 0:
        n = int(nw)
        discrete = max(n, _OASN_MIN_DISCRETE_SAMPLES)
        if discrete != n:
            warnings.warn(
                f"write_oasn_input: nw_samples={n} would put "
                f"{n} sample(s) in the discrete spectral band, below the "
                f"NWSD >= {_OASN_MIN_DISCRETE_SAMPLES} the OASES manual "
                f"specifies (oasn.tex:313). OASES does not check it, and the "
                f"band count divides the default contour offset "
                f"(unoasn22.f:289), so a smaller count silently over-damps "
                f"the modal poles. Raised to "
                f"{_OASN_MIN_DISCRETE_SAMPLES}.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
        counts = (n, discrete, max(1, n // 4))
        total = sum(counts)
        if total > _OASES_MAX_WAVENUMBERS:
            raise ConfigurationError(
                f"write_oasn_input: nw_samples={n} makes the noise block's "
                f"three-band total {counts[0]}+{counts[1]}+{counts[2]} = "
                f"{total} wavenumber samples, above OASES' array bound "
                f"NP = {_OASES_MAX_WAVENUMBERS} (oases/src/compar.f:37-38); "
                f"NOIPAR stops with '*** TOO MANY SAMPLING POINTS ***' "
                f"(oasnun22.f:319) and writes no output.",
                remediation=(f"Pass nw_samples <= "
                             f"{int(_OASES_MAX_WAVENUMBERS / 2.25)} so the "
                             f"2.25x band total stays within NP, or <= 0 for "
                             f"the manual's 400/400/100 counts."),
            )
        return " ".join(str(v) for v in counts)
    return " ".join(str(v) for v in _OASN_NOISE_SAMPLES)


def _emit_receiver_array(
    f: TextIO,
    receiver: Receiver,
    *,
    receiver_types=None,
    receiver_gains=None,
    writer: str = 'write_oases_input',
) -> None:
    """Emit the ``NRCV`` count and the ``Z X Y ITYP GAIN`` array records.

    ``INPRCV`` (``oasnun22.f:30``, ``:36``) is shared by OASN and OASS, so
    both decks carry this block verbatim: a count, then one record per
    element consumed by a **single** list-directed read of ``5·NRCV`` values.
    Line breaks inside it are therefore free, and one element per line is the
    readable choice.

    Coordinates are metres (``oasn.tex:47-49``); uacpy places its array on
    the z axis, so ``X = Y = 0``. ``ITYP`` indexes ``CRTYP`` = HYDROP /
    X-GEOP / Y-GEOP / Z-GEOP (``oasnun22.f:18``), so 1 is a hydrophone.

    In OASS this column does double duty: there is no field-parameter option
    letter, and ``oasnun22.f:97-118`` derives ``IOUT``/``NOUT`` from these
    types instead. Mixing them then trips ``unoass21.f:165-169``
    ``STOP '*** INTEGRATION ONLY ALLOWED FOR ONE FIELD PARAMETER ***'`` — a
    hard stop with no output and no diagnostic uacpy could attribute — so a
    mixed array is refused here, before the deck is written.
    """
    depths = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
    n = len(depths)
    _check_receiver_depth_count(n)
    depth_tokens = [f"{z:.2f}" for z in depths]
    _check_receiver_depth_tokens(depths, depth_tokens,
                                 what=f"{writer}: receiver depths")

    if receiver_types is None:
        types = [1] * n
    else:
        types = [int(t) for t in np.atleast_1d(receiver_types)]
        if len(types) == 1:
            types = types * n
        if len(types) != n:
            raise ConfigurationError(
                f"{writer}: receiver_types has {len(types)} entries but the "
                f"array has {n} elements; pass one per element or a single "
                f"value for all."
            )
    unknown = sorted({t for t in types if t not in (1, 2, 3, 4)})
    if unknown:
        raise ConfigurationError(
            f"{writer}: receiver_types {unknown} are not OASES receiver "
            f"types; valid are 1 (hydrophone), 2 (x-geophone), "
            f"3 (y-geophone), 4 (z-geophone) — oasnun22.f:18."
        )
    if len(set(types)) > 1:
        raise ConfigurationError(
            f"{writer}: receiver_types mixes {sorted(set(types))}, but the "
            f"array must be all one type — the receiver type selects the "
            f"field parameter (oasnun22.f:97-118) and a mixed array trips "
            f"'INTEGRATION ONLY ALLOWED FOR ONE FIELD PARAMETER' "
            f"(unoass21.f:165-169), which stops the binary with no output."
        )

    if receiver_gains is None:
        gains = [0.0] * n
    else:
        gains = [float(g) for g in np.atleast_1d(receiver_gains)]
        if len(gains) == 1:
            gains = gains * n
        if len(gains) != n:
            raise ConfigurationError(
                f"{writer}: receiver_gains has {len(gains)} entries but the "
                f"array has {n} elements; pass one per element or a single "
                f"value for all."
            )

    f.write(f"{n}\n")
    for z_tok, t, g in zip(depth_tokens, types, gains):
        f.write(f"{z_tok} 0 0 {t} {g:g}\n")


def _oases_nw_line(nw_samples, icut2_auto: int) -> str:
    """``NW ICUT1 ICUT2`` wavenumber-sampling line.

    ``NW < 0`` selects OASES's automatic sampling — AUTSMN then recomputes
    ICUT1/ICUT2 (unoasn22.f:240-241, oasnun22.f:432) so the trailing values
    are inert. Exactly ``-1`` is emitted, never a more negative value: OAST
    reads ``NW < -1`` as a request for the Padé approximation with
    ``NPADE = -NW`` and IC1/IC2 as its parameters (unoast31.f:419-423).
    A pinned ``NW`` is clamped by ``ICUT2 = MIN0(NW, ICUT2)``
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
    'dip_angle', 'c_low', 'c_high',
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
    'integration_offset', 'nw_samples', 'dip_angle', 'c_low', 'c_high',
})
_OASR_KWARGS = frozenset({
    'freq_min', 'freq_max', 'n_frequencies', 'freq_output_increment',
    'angle_min', 'angle_max', 'n_angles', 'angle_output_increment',
})




#: Option letters whose GETOPT flag makes the program read input none of these
#: writers produces — a whole deck block, extra tokens on a record the writer
#: does emit, or a separate auxiliary file. The read then runs off the end of
#: the deck, eats the block after it, or aborts on a missing unit, so the
#: failure surfaces as a bare Fortran error with no indication of which letter
#: caused it. Keyed by writer; each entry maps the letter to what the program
#: demands and the Fortran line that demands it. An empty entry records that
#: the program's option-gated reads are all covered.
_UNWRITTEN_OPTION_BLOCKS = {
    'write_oass_input': {
        'Z': ('two profile-axis records — VPLEFT VPRIGHT VPLEN VPINC then '
              'DPUP DPLO DPLEN DPINC (read when IPROF > 0)',
              'unoass21.f:318-320, :668-671'),
    },
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
    'write_oassp_input': {
        'd': ("the Doppler frequency line's ISTYP VSOU VREC tokens",
              'unoassp30.f:136'),
        'G': ('a dispersion-curve block (NMODES, then two axis rows)',
              'unoassp30.f:397-400'),
        'T': ('two extra receiver-line tokens ZREF and ANGLE for the tilted '
              'array', 'oaseun31.f:1157-1158'),
        'Z': ('two velocity-profile plot-axis rows', 'unoassp30.f:469-470'),
        'l': ('an external source-array file on unit 2 — LS, then LS rows of '
              'depth/delay/strength', 'oaseun31.f:1074-1087'),
        'v': ('a source transfer-function file through RDSTRF',
              'unoassp30.f:253'),
        'b': ('a third input file of bistatic data on unit 47',
              'unoassp30.f:129'),
    },
}


def _reject_unwritten_option_blocks(writer: str, options: str) -> None:
    """Raise on an option letter demanding input this writer never produces."""
    demanded = _UNWRITTEN_OPTION_BLOCKS[writer]
    used = sorted(_oases_option_chars(options) & set(demanded))
    if not used:
        return
    detail = '; '.join(f"{c!r} makes it read {demanded[c][0]} "
                       f"({demanded[c][1]})" for c in used)
    # The letters are valid OASES; what is missing is a block this writer
    # emits, so the refusal is a capability of uacpy's deck writer rather
    # than an illegal argument. The alternative is the same option string
    # with the unsupported letters dropped — GETOPT reads the record
    # character by character, so removing the characters is the whole edit.
    raise UnsupportedFeatureError(
        writer,
        f"option letter(s) {used} — {detail}, and this writer produces no "
        f"such input. The deck would fail inside Fortran instead of here.",
        alternatives=[repr(''.join(c for c in str(options)
                                   if c not in used).strip())],
        alternatives_label='option strings',
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


def _reject_log_frequency_ladder(options: str, nfreq: int) -> None:
    """Raise on OAST option ``'o'`` with a multi-frequency sweep.

    ``unoast31.f:393`` runs the NFREQ-point sweep LOG-spaced under
    ``FRCONT`` (option ``'o'``). :func:`~uacpy.io.oases_reader.read_oast_tl`
    reads each curve's frequency from its ``Freq:`` label, so a log ladder
    is recoverable in principle — but the label is written ``F7.1``, and a
    log sweep crowds its low end, so two rungs can print the same value.
    Colliding labels drop the reader onto its positional walk, which
    assumes the deck's linear grid and would mislabel every curve. The
    deck is refused rather than left to depend on how close the rungs
    round. (OASR's ``'C'`` twin is fine: the OASR model relabels its own
    log sweep.)
    """
    if 'o' not in _oases_option_chars(options) or int(nfreq) <= 1:
        return
    raise ConfigurationError(
        f"write_oast_input: option 'o' makes the binary run its "
        f"{int(nfreq)}-frequency sweep log-spaced (unoast31.f:393), and the "
        f".plp records each frequency only to one decimal, so two rungs of "
        f"a log ladder can print the same label and the reader falls back "
        f"to the deck's linear grid — a mislabeled frequency axis.",
        remediation="Drop 'o' from options=, or run one frequency per deck.",
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
    ssp_data: np.ndarray, n_other_layers: int, *, warn: bool = True,
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

    ``warn=False`` decimates silently, for callers that only need the row
    count the deck will end up with and whose caller hears about the
    decimation from the writing call itself. It suppresses this one warning
    where a ``warnings.catch_warnings()`` window around the call would mute
    the whole process — filter state is global, so such a window also
    swallows warnings raised on other threads while it is open.
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
    if warn:
        warnings.warn(
            f"env.ssp has {n_rows} samples but this deck can spend at most "
            f"{max_rows} of OASES' {_OASES_MAX_LAYERS} layers on the water "
            f"column (oases/src/compar.f:23 `parameter (NLA = "
            f"{_OASES_MAX_LAYERS})`, {n_other_layers} taken by the halfspaces "
            f"and sediment stack); decimated to {keep.size} evenly-spaced "
            f"samples. Decimate env.ssp yourself if you want to choose which "
            f"features survive.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    return ssp_data[keep, :]


#: Compressional speed the ``'rigid'`` surface substitute emits (OASES has no
#: rigid upper halfspace; a high-impedance fluid is the closest stand-in).
_OASES_RIGID_SURFACE_CP = 4000.0

#: Column 7 of the layer-1 record. INENVI overwrites ROUGH(1) with ROUGH(2)
#: (oaseun31.f:377), so nothing written here reaches OASES; the field is kept
#: in the ``%.4f`` shape of every other RG so the record's column table holds.
_OASES_DUMMY_RG = '0.0000'


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

    OASES layer format: D CC CS AC AS RO RG [CL] (oast.tex:42; CL is read
    only when RG < 0)
    For vacuum: all zeros.
    For elastic (ice): sound_speed, shear_speed, attenuation, density.

    Column 7 (RG) is the dummy ``_OASES_DUMMY_RG``: the manual calls RG(1)
    dummy (oast.tex:344) and INENVI overwrites it with ROUGH(2) at
    oaseun31.f:377, immediately after the layer-read loop closes. ROUGH(M) is
    the roughness of the interface at the TOP of layer M — ``DO 1111
    M=2,NUML`` compares ``LAYTYP(M-1)`` against ``LAYTYP(M)``
    (oaseun31.f:381-383) — so the sea surface is carried by the first water
    layer's RG (:func:`_emit_water_layers`).
    """
    surface = env.surface
    acoustic_type = getattr(surface, 'acoustic_type', 'vacuum')
    if isinstance(acoustic_type, str):
        atype = acoustic_type.lower()
    else:
        atype = str(acoustic_type).lower()

    if 'vacuum' in atype:
        return f"0 0 0 0 0 0 {_OASES_DUMMY_RG} 0"

    # A tabulated or precomputed surface has no OASES deck spelling, so it must
    # not fall through to the fluid row below: the placeholder a
    # BoundaryProperties carries for those types is cp = 1600 m/s, rho = 1.5,
    # alpha = 0.5, which INENVI reads as an ordinary fluid half-space
    # (oaseun31.f:129-131) — only |V(M,2)| < 1e-10 spells pressure release
    # (:125-126). Measured against a vacuum surface on a 100 m isovelocity
    # guide at 100 Hz with a 1700 m/s bottom: median 4.36 dB, max 31.01 dB
    # over 1121 ranges, at exit 0 with no warning. _extract_bottom_props
    # refuses the same substitution for the seabed; the surface gets the same
    # treatment.
    if 'file' in atype or 'precalc' in atype:
        raise UnsupportedFeatureError(
            'OASES',
            f"a {acoustic_type!r} sea surface — OASES spells a top boundary "
            f"only as vacuum (pressure release) or as an explicit "
            f"fluid/elastic half-space, and the placeholder properties such a "
            f"surface carries would be written as a 1600 m/s fluid",
            ['surface=None or a vacuum surface (pressure release)',
             "a BoundaryProperties(acoustic_type='half-space', ...) surface "
             'carrying the properties you want written',
             'Bellhop or Kraken, which read a tabulated top reflection '
             'coefficient'],
            alternatives_label='surfaces',
        )

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
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
        return (f"0 {_OASES_RIGID_SURFACE_CP:.1f} 0.0 0.000 0.000 2.50 "
                f"{_OASES_DUMMY_RG} 0")

    return (f"0 {c_p:.2f} {c_s:.2f} {alpha_p:.3f} "
            f"{alpha_s:.3f} {rho:.2f} {_OASES_DUMMY_RG} 0")


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
        Extra whitespace-separated tokens to append to the header line.
        INREC picks the record shape from the program name, not the option
        line: only OAST reads a 4th token, ``IDINC`` (oaseun31.f:1156, taken
        under ``PROGNM(1:5).EQ.'OASTL'``); every other program reads the
        3-token form at :1165. IDINC arrives in OAST as ``INTF`` and gates
        integrand plots per receiver, ``MOD(NREC-1,INTF)`` at
        unoast31.f:591 — the manual names it IR, "plot output increment"
        (oast.tex:67). An empty string matches OASP/OASR.
    """
    depths = np.asarray(receiver.depths, dtype=float)
    n = len(depths)
    _check_receiver_depth_count(n)
    z_min = float(depths.min())
    z_max = float(depths.max())

    if n <= 1:
        return [f"{z_min:.2f} {z_max:.2f} {n}{trailing}"]

    # The compact (z_min, z_max, n) form makes OASES build its own equidistant
    # grid, discarding the requested depths, so it may only be used for an axis
    # that really is equidistant. Use the Acoustics-Toolbox's own test — a
    # verbatim port of ``Matlab/ReadWrite/equally_spaced.m``, which the .flp
    # writer uses for the same decision — rather than a local tolerance:
    # it measures against the ideal uniform grid, so a deviation cannot
    # accumulate with the receiver count the way a per-interval comparison does.
    if equally_spaced(depths):
        # Only the endpoints reach OASES here — it rebuilds the interior from
        # RDSTEP=(RDLOW-RD)/(IR-1) (oaseun31.f:1167-1168) — so this form
        # survives interior spacings finer than the depth column resolves,
        # and it is the SPAN that has to be distinguishable: two endpoints on
        # one token give RDSTEP = 0 and put all n receivers at one depth.
        if f"{z_min:.2f}" == f"{z_max:.2f}":
            raise ConfigurationError(
                f"the {n} receiver depths span {z_min:g} m to {z_max:g} m, "
                f"which both write as {z_min:.2f} m at the deck's "
                f"{_OASES_RECEIVER_DEPTH_RESOLUTION_M:g} m resolution. OASES "
                f"builds the array from the two endpoints, "
                f"RDSTEP=(RDLOW-RD)/(IR-1) (oaseun31.f:1167-1168), so it "
                f"would place every receiver at that one depth while uacpy's "
                f"readers return the {n} distinct depths that were asked for.",
                remediation=(f"Span the receiver depths by more than "
                             f"{_OASES_RECEIVER_DEPTH_RESOLUTION_M:g} m."),
            )
        return [f"{z_min:.2f} {z_max:.2f} {n}{trailing}"]

    # Non-uniform: emit NR = -n, then individual depths on the next line.
    tokens = [f"{d:.2f}" for d in depths]
    _check_receiver_depth_tokens(depths, tokens, what="receiver depths")
    header = f"{z_min:.2f} {z_max:.2f} {-n}{trailing}"
    return [header, ' '.join(tokens)]


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


def _check_nw_samples(writer: str, nw_samples, *, power_of_two: bool):
    """Validate a pinned wavenumber-sample count against OASES' NP array bound.

    A pinned ``NW`` above NP is silently clamped by ``NWVNO=MIN0(NWVNO,NP)``
    (unoast31.f:435, unoasp22.f:174), and OAST additionally rounds it up to
    the next power of two (unoast31.f:447-458). Both change the wavenumber
    step — and hence the range at which the FFT wraps — from what was asked
    for, so say so here rather than let the run answer a different question.
    """
    if nw_samples is None:
        return nw_samples
    nw = int(nw_samples)
    if nw <= 0:                      # automatic sampling
        return nw_samples
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
        # Round up HERE rather than letting the binary do it. OAST sets its
        # integration cut from the pre-rounding count — `icut1=1, icut2=nwvno`
        # at unoast31.f:441-442 — and only then rounds NWVNO up to a power of
        # two (:447-458), computing DLWVNO from the rounded count (:474). The
        # kernel is zeroed above ICUT2 (oasfun22.f:264-265), so the deck's own
        # NW became a cut partway up a wider grid and the integral stopped
        # short of CMIN, deleting the trapped modes: measured on a 100 m
        # Pekeris guide at 100 Hz, nw_samples=3000 came back 78.8 dB and
        # nw_samples=5000 91.8 dB from the nw_samples=4096 answer, both at
        # exit 0. Writing the rounded count makes the binary's rounding a
        # no-op, so ICUT2 spans the grid it is cut on.
        warnings.warn(
            f"{writer}: nw_samples={nw} is not a power of two, so it is "
            f"written as {rounded}. OAST takes its integration cut from the "
            f"count in the deck (unoast31.f:441-442) and only then rounds the "
            f"grid up (:447-458), so passing {nw} would have integrated only "
            f"the first {nw} of {rounded} wavenumbers and truncated the "
            f"spectrum short of c_low.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
        return rounded
    return nw_samples


def _single_source_depth(writer: str, source: Source) -> float:
    """The one source depth an OASES Block-V record can carry.

    INSRC reads a single ``SD`` (oaseun31.f:1089, :1114, :1119) and the
    programs place exactly one source at it, so a Source carrying several
    depths cannot be written — taking ``depths[0]`` would answer for a
    different array than the one that was asked for.
    """
    depths = np.atleast_1d(np.asarray(source.depths, dtype=float))
    if depths.size != 1:
        raise ConfigurationError(
            f"{writer}: source.depths carries {depths.size} depths "
            f"({np.array2string(depths, precision=2)} m), but the OASES "
            f"Block-V source record holds one SD (oaseun31.f:1089-1119).",
            remediation="Pass a single-depth Source, or write one deck per "
                        "source depth.",
        )
    return float(depths[0])


def _check_n_time_samples(writer: str, n_time) -> None:
    """Validate a pinned FFT length against OASP/OASSP's NX handling.

    ``NT`` is not taken as given: both binaries round it up to the next power
    of two and then clamp it with ``NX = MIN0( NX, 2*NP )`` — 131072
    (unoasp22.f:222-234, unoassp30.f:190-202, oases/src/compar.f:37-38). The
    round-up is announced on stdout only, and the clamp not at all, while
    ``DLFREQ = 1/(DT*NX)`` (unoasp22.f:237) moves with it — so a rejected
    ``n_time_samples`` silently relabels every bin of the ``.trf``.
    """
    if n_time is None:
        return
    nt = int(n_time)
    if nt < 1:
        raise ConfigurationError(
            f"{writer}: n_time_samples={nt} is not a positive FFT length; "
            f"OASP walks 2**IPOW up from 2 until it reaches NT "
            f"(unoasp22.f:222-227), so a non-positive count leaves NX = 2.",
            remediation="Pass a positive power of two.",
        )
    if nt > _OASES_MAX_TIME_SAMPLES:
        raise ConfigurationError(
            f"{writer}: n_time_samples={nt} exceeds OASES' transform bound "
            f"2*NP = {_OASES_MAX_TIME_SAMPLES} (oases/src/compar.f:37-38); "
            f"the binary clamps NX to {_OASES_MAX_TIME_SAMPLES} silently "
            f"(unoasp22.f:234) and every .trf bin then sits at "
            f"DLFREQ = 1/(DT*{_OASES_MAX_TIME_SAMPLES}), not the spacing "
            f"{nt} implies.",
            remediation=(f"Pass n_time_samples <= {_OASES_MAX_TIME_SAMPLES}, "
                         f"or lengthen time_step to cover the same duration."),
        )
    if nt & (nt - 1):
        rounded = 1 << (nt - 1).bit_length()
        warnings.warn(
            f"{writer}: n_time_samples={nt} is not a power of two; OASES "
            f"rounds it up to {rounded} (unoasp22.f:222-231) and the .trf "
            f"bin spacing becomes DLFREQ = 1/(DT*{rounded}), so the "
            f"transform length is not the one implied by {nt}.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
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
    A source with a frequency sweep writes ``NFREQ > 1`` on Block III; the
    run then emits one TL curve per receiver per frequency, which
    :func:`~uacpy.io.oases_reader.read_oast_tl` returns as an
    ``(n_freq, n_depths, n_ranges)`` stack.

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
    >>> import tempfile, os
    >>> env = Environment(bathymetry=100, ssp=1500)
    >>> source = Source(depths=50, frequencies=100)
    >>> receiver = Receiver(depths=np.linspace(10,90,40),
    ...                     ranges=np.linspace(100,10000,100))
    >>> with tempfile.TemporaryDirectory() as d:
    ...     write_oast_input(os.path.join(d, 'test.dat'),
    ...                      env, source, receiver)
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
    src_depth = _single_source_depth('write_oast_input', source)
    r_max = float(receiver.ranges.max())

    # Optional parameters
    integration_offset = kwargs.get('integration_offset', 0)
    nw_samples = kwargs.get('nw_samples', -1)  # -1 = automatic
    plot_rmin = kwargs.get('plot_rmin', 0.0)            # metres (public API)
    plot_rmax = kwargs.get('plot_rmax', r_max)          # metres (public API)
    c_low = kwargs.get('c_low')
    c_high = kwargs.get('c_high')

    # Get reference sound speed for plot axes
    c_ref = float(ssp_data[0, 1])  # Sound speed at surface

    # Options string
    if options is None:
        options = 'N J T'  # Normal stress, complex contour, TL vs range
    _reject_unwritten_option_blocks('write_oast_input', options)
    _unknown = sorted(_oases_option_chars(options) - _OAST_OPTIONS)
    if _unknown:
        raise ConfigurationError(
            f"write_oast_input: {_unknown} are not "
            f"option letters this binary tests — GETOPT "
            f"(unoast31.f:935-1162) recognises only "
            f"{''.join(sorted(_OAST_OPTIONS))}. Its ladder prints '>>>> UNKNOWN OPTION' to unit 6 (:1166-1168), which uacpy captures rather than surfaces, so the letter is dropped where the caller cannot see it.",
            remediation="Drop the letter(s) from options=.",
        )
    nw_samples = _check_nw_samples(
        'write_oast_input', nw_samples, power_of_two=True)
    source_record = _resolve_source_record(
        'write_oast_input', options, kwargs.get('dip_angle'))
    _warn_volume_attenuation_ignored(env)

    freq_min, freq_max, nfreq = _resolve_freq_sweep(
        'write_oast_input', source, freq)
    _check_frequency_contours('write_oast_input', options, 'o', nfreq)
    _reject_log_frequency_ladder(options, nfreq)

    with open(filepath, 'w') as f:
        _write_oases_header(f, env, options, "OAST Simulation via UACPY",
                            title_chars=_OAST_TITLE_CHARS)

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

        # Block IV: Environment — NL (oaseun31.f:43), then NL layer records
        # (:54), read by INENVI from unoast31.f:171.
        n_sed_layers = _count_bottom_layers(env)
        # NL = upper halfspace + water + sediments + bottom halfspace.
        ssp_subset = _check_ssp_layer_count(ssp_data, 2 + n_sed_layers)

        c_values = ssp_subset[:, 1]
        is_isovelocity = np.allclose(c_values, c_values[0], rtol=1e-6)

        if is_isovelocity:
            # One water layer covers the whole column. Emitting the general
            # form would work — INENVI folds every CC = |CS| layer back to
            # isovelocity (oaseun31.f:181-182) — but it would spend one of
            # OASES' NLA layer slots per SSP sample to say the same thing.
            n_layers = 3 + n_sed_layers  # vacuum + water + sed_layers + bottom
            f.write(f"{n_layers}\n")
            f.write(f"{_format_upper_halfspace(env)}\n")
            f.write(f"0.00 {c_values[0]:.2f} 0 0.0 0 1.0 "
                    f"{_surface_roughness(env):.4f} 0\n")
            _emit_bottom_layers(
                f, env, depth,
                c_p, c_s, alpha_p, alpha_s, rho,
                extra_columns=1,
            )
        else:
            n_water_layers = len(ssp_subset)
            n_layers = 1 + n_water_layers + n_sed_layers + 1
            f.write(f"{n_layers}\n")
            f.write(f"{_format_upper_halfspace(env)}\n")
            _emit_water_layers(f, ssp_subset,
                               surface_roughness=_surface_roughness(env),
                               extra_columns=1)
            _emit_bottom_layers(
                f, env, depth,
                c_p, c_s, alpha_p, alpha_s, rho,
                extra_columns=1,
            )

        # Block V: Sources — one source at src_depth, in the record shape
        # INSRC's LINA / dip_sou branch expects (called from
        # unoast31.f:175).
        f.write(_source_block_line(src_depth, **source_record) + '\n')

        # Block VI: Receivers — RD1 RD2 NR IR, read by INREC
        # (unoast31.f:208 → oaseun31.f:1156). NR<0 signals an explicit depth
        # list on the following record (oast.tex:464-493).
        for line in _receiver_block_lines(receiver, trailing=' 1'):
            f.write(line + '\n')

        # Block VII: Wavenumber sampling — CMIN CMAX (unoast31.f:213).
        # The SSP-derived pair bounds k on the water column alone, so an
        # elastic seabed's shear and interface branches — phase speeds below
        # the slowest water speed — fall outside k_max and never enter the
        # integral. c_low= admits them; c_high= caps the evanescent tail.
        cmin, cmax = oases_wavenumber_bounds(ssp_data)
        if c_low is not None:
            cmin = float(c_low)
        if c_high is not None:
            cmax = float(c_high)
        f.write(f"{cmin:.1f} {_OASES_CMAX_FMT.format(cmax)}\n")

        # NW IC1 IC2 (unoast31.f:230) — automatic sampling (NW=-1) ignores
        # IC1/IC2 per oast.tex:541-550 and forces the complex contour even
        # without 'J' (unoast31.f:426). When NW>0 the constraint IC2 ≤ NW
        # (oast.tex:73-75) must hold, so clamp.
        f.write(f"{_oases_nw_line(nw_samples, 1)}\n")

        # Block VIII (unoast31.f:234): XLEFT XRIGHT XAXIS XINC, all km except
        # XAXIS. XLEFT/XRIGHT set the FFT output grid, not merely a plot window
        # — unoast31.f:482 takes R0 = XLEFT and :490 the point count
        # LF = INT((XRIGHT-XLEFT)/RSTEP+1.5), and PLTWRI echoes that
        # N/XOFF/DX triple into the .plp (oasgun21.f:653-655) as the axis
        # uacpy reads the .plt against. So %.1f km = 100 m resolution would
        # snap the native grid — and any run shorter than ~50 m would write
        # XRIGHT = 0.0 and return an all-NaN field with no exception. %.9f is
        # sub-micron; the read is list-directed. In cylindrical geometry the
        # first range is floored at one step, R0 = MAX(R0, RSTEP)
        # (unoast31.f:483-485), so plot_rmin = 0 comes back as RSTEP.
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

        # Blocks IX-XII are plot axes: OAST passes them straight to its own
        # plotters, which put them in the .plp header via PLPWRI and write
        # the curve unclipped to the .plt (PLTLOS, oasfun22.f:361-365). They
        # cannot alter a number uacpy reads back, but each one OASES asks for
        # must be present or the next READ eats the wrong record — hence the
        # literal axis windows below, in the units of whatever they bound.
        if (has('A') and not has('o')) or has('T', 'I', 'a', 'D'):
            # Block IX — YUP YDOWN YAXIS YINC (unoast31.f:242): 20-100 dB
            # loss axis, 12 cm long, 10 dB ticks.
            f.write("20 100 12 10\n" * n_out)

        if has('C', 'D', 'c'):
            # Block X — RDUP RDDOWN CYAXIS RDINC (unoast31.f:250): the whole
            # water column over a 12 cm axis, ten ticks.
            f.write(f"0 {depth:.1f} 12 {depth/10:.1f}\n")

        if has('C', 'o'):
            # Block XI — ZMIN ZMAX ZSTEP contour levels (unoast31.f:262):
            # 40-100 dB loss contoured every 10 dB.
            f.write("40 100 10\n" * n_out)

        if has('Z'):
            # Block XII — VLEF VRIG VLEN VINC + DVUP DVLO DVLN DVIN
            # (unoast31.f:286-287), the speed and depth axes of the profile
            # plot. c_range falls back to a tenth of the surface speed on a
            # near-isovelocity profile so the axis does not collapse.
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
        - white_noise_level : float or None
            White noise level in dB, added to every covariance-matrix
            diagonal element. OASES has **no off switch for this field**:
            NOIPAR forms ``WNLEV = 10.0**(WNLEVDB/10.0)`` (oasnun22.f:228)
            and WNOISE adds it to the diagonal unconditionally
            (oasnun22.f:670, :1154-1157), so ``0.0`` means a literal 0 dB —
            unit linear power — on each sensor, not "disabled". The manual
            (oasn.tex:269) describes it as a plain sensor level. Default
            ``None`` writes -200 dB, whose linear power of 1e-20 is
            numerically nil against any noise field.
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
    >>> import tempfile, os
    >>> env = Environment(bathymetry=100, ssp=1500)
    >>> source = Source(depths=50, frequencies=100)
    >>> receiver = Receiver(depths=[30, 50, 70], ranges=[0])
    >>> with tempfile.TemporaryDirectory() as d:
    ...     write_oasn_input(os.path.join(d, 'test.dat'), env, source,
    ...                      receiver, options='N J', surface_noise_level=70)
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
    # WNLEV has no dead band: WNOISE adds 10**(WNLEVDB/10) to every
    # covariance diagonal unconditionally (oasnun22.f:228, :670, :1157), so
    # "disabled" must be written as a level whose linear power is nil.
    # None (the default) -> -200 dB = 1e-20 linear; an explicit 0.0 is a
    # literal 0 dB = unit linear power per sensor.
    _wn = kwargs.get('white_noise_level')
    white_noise_level = -200.0 if _wn is None else float(_wn)
    deep_noise_level = round(float(kwargs.get('deep_noise_level', 0)), 1)
    _check_oasn_noise_level('surface_noise_level', surface_noise_level,
                            dead_band=0.01)
    discrete_sources = kwargs.get('discrete_sources', [])
    n_discrete = len(discrete_sources)
    if n_discrete > _OASES_MAX_DISCRETE_NOISE_SOURCES:
        raise ConfigurationError(
            f"write_oasn_input: discrete_sources carries {n_discrete} "
            f"sources, above OASN's compiled bound NSMAX = "
            f"{_OASES_MAX_DISCRETE_NOISE_SOURCES} (oases/src/compar.f:51, "
            f"noiprm.f:13-14); the binary would STOP with '*** TOO MANY "
            f"DISCRETE NOISE SOURCES ***' (oasnun22.f:372-373) and EXIT "
            f"CODE 0, leaving no .xsm.",
            remediation=f"Pass at most "
                        f"{_OASES_MAX_DISCRETE_NOISE_SOURCES} discrete "
                        f"sources, or run the field in batches.",
        )
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
    _unknown = sorted(_oases_option_chars(options) - _OASN_OPTIONS)
    if _unknown:
        raise ConfigurationError(
            f"write_oasn_input: {_unknown} are not "
            f"option letters this binary tests — GETOPT "
            f"(unoasn22.f:571-727) recognises only "
            f"{''.join(sorted(_OASN_OPTIONS))}. Its ladder ends with a bare END IF (:727), so the binary discards them in silence and runs a different configuration than asked for.",
            remediation="Drop the letter(s) from options=.",
        )

    # OASN reads the noise-source block only under `IF (CALNSE.or.trfout)`
    # (unoasn22.f:173-174); CALNSE comes from 'N'/'n' and TRFOUT from
    # uppercase 'T'. Written unconditionally, the block would be consumed by
    # whatever READ comes next — the replica grid under 'R' — and every
    # record below it shifts by one.
    noise_block = bool(_oases_option_chars(options) & {'N', 'n', 'T'})
    if not noise_block:
        # Truthiness deliberately lets white_noise_level=0.0 pass unflagged:
        # a deck without the noise block writes no Block VI at all, so a
        # 0 dB request loses nothing there, and the OASN model class always
        # forwards its own default 0.0.
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

    freq_min_b, freq_max_b, nfreq = _resolve_freq_sweep(
        'write_oasn_input', source, freq)

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

        # Block IV: Environment — NL then NL layer records, read by INENVI
        # from unoasn22.f:156. NL = upper halfspace + water + sediments +
        # bottom halfspace.
        n_sed_layers = _count_bottom_layers(env)
        ssp_subset = _check_ssp_layer_count(ssp_data, 2 + n_sed_layers)
        n_water_layers = len(ssp_subset)
        n_layers = 1 + n_water_layers + n_sed_layers + 1
        f.write(f"{n_layers}\n")

        f.write(f"{_format_upper_halfspace(env)}\n")
        _emit_water_layers(f, ssp_subset,
                           surface_roughness=_surface_roughness(env),
                           extra_columns=1)
        _emit_bottom_layers(
            f, env, depth,
            c_p, c_s, alpha_p, alpha_s, rho,
            extra_columns=1,
        )

        # Block V: Receiver Array — NRCV (oasnun22.f:30), then one
        # Z X Y ITYP GAIN record per element, all consumed by INPRCV's single
        # list-directed read at oasnun22.f:36. Unlike the discrete sources of
        # Block IX, the array coordinates are in metres (oasn.tex:47-49);
        # OASN places uacpy's array on the z axis, so X = Y = 0. ITYP indexes
        # CRTYP = HYDROP / X-GEOP / Y-GEOP / Z-GEOP (oasnun22.f:18), so 1 is
        # a hydrophone, and GAIN is per-element gain in dB.
        _emit_receiver_array(f, receiver, writer='write_oasn_input')

        # Blocks VI-IX are NOIPAR's, called only under
        # `IF (CALNSE.or.trfout)` (unoasn22.f:173-174).
        if noise_block:
            # Block VI: Sources — SSLEV WNLEV DSLEV NDNS (oasnun22.f:179).
            # These four values are what gates Blocks VII, VIII and IX below.
            f.write(f"{surface_noise_level:.1f} {white_noise_level:.1f} "
                    f"{deep_noise_level:.1f} {n_discrete}\n")

            # Block VII: surface-noise wavenumber parameters. NOIPAR reads
            # them only when `abs(SNLEVDB).GE.0.01` (oasnun22.f:276).
            if abs(surface_noise_level) >= 0.01:
                # CMINS CMAXS (oasnun22.f:277) — the model's c_low/c_high
                # when given, else the slow edge below the water column and
                # 1E8 for "no upper limit", which oasn.tex:284-287 requires
                # here because the continuous spectrum always matters for
                # surface noise. Documented as reaching this block, so it
                # must.
                f.write(f"{_noise_cmin(kwargs, ssp_data):.1f} "
                        f"{_noise_cmax(kwargs):.6g}\n")
                # NWSC NWSD NWSE, samples in the continuous / discrete /
                # evanescent bands (oasnun22.f:278). NWSD > 1 splits the
                # slowness axis into those three bands (:282); NWSD <= 1
                # collapses them into one equidistant band (:305-310).
                f.write(f"{_noise_nw(kwargs)}\n")

            # Block VIII: deep-noise parameters. NOIPAR reads them only when
            # `DPLEVDB.GE.0.01` (oasnun22.f:324) — deliberately asymmetric
            # with the surface gate above: a negative deep level just disables
            # the source, so emitting these lines would desynchronise every
            # later READ.
            if deep_noise_level >= 0.01:
                # DPSD, depth of the deep source sheet (oasnun22.f:325).
                deep_source_depth = kwargs.get('deep_source_depth',
                                               depth * 0.5)
                f.write(f"{deep_source_depth:.2f}\n")
                # CMIND CMAXD (oasnun22.f:326), then NWDC NWDD NWDE (:327) —
                # the same three-band split as Block VII.
                f.write(f"{_noise_cmin(kwargs, ssp_data):.1f} "
                        f"{_noise_cmax(kwargs):.6g}\n")
                f.write(f"{_noise_nw(kwargs)}\n")

        # Block IX: Discrete sources, gated on NDNS > 0 (oasnun22.f:371).
        if noise_block and n_discrete > 0:
            for ds in discrete_sources:
                # ZDN XDN YDN DNLEV (oasnun22.f:380): depth in m, x/y in km
                # — unlike the Block V array coordinates, which are metres
                # (oasn.tex:341-344) — and level in dB.
                z_ds = ds.get('depth', 50.0)
                x_ds = ds.get('x', 1.0)  # km
                y_ds = ds.get('y', 0.0)  # km
                level_ds = ds.get('level', 180.0)
                # x/y are KILOMETRES here (oasnun22.f:418-419 multiplies by
                # 1e3) while the public API takes metres, so %.3f quantised the
                # source position onto a 1 m lattice. At 1 kHz that is 0.67
                # wavelengths: x = 1.0000 km and x = 1.0004 km wrote the same
                # token and produced byte-identical .xsm files, while the 1 m
                # step that IS representable moves covariance element 22 by 85%
                # (7.99e11 -> 1.48e12, about 2.7 dB). The replica grid below
                # already carries %.9f for this reason. INSRC reads the record
                # list-directed (oasnun22.f:380), so the width is free.
                #
                # This write MUST stay inside the loop: oasnun22.f:371,:380
                # reads exactly NDNS records (`DO 105 I=1,NDNS`), and Block VI
                # above declares NDNS, so writing one record leaves the reader
                # consuming the CMIN/CMAX and NW records as source 2 and dying
                # at :380 with "End of file", exit 2. N = 1 is indistinguishable
                # either way, which is why every existing test missed it.
                f.write(f"{z_ds:.2f} {x_ds:.9f} {y_ds:.9f} {level_ds:.1f}\n")

            # CMINDIN CMAXDIN (oasnun22.f:420). 1E8 keeps the whole
            # spectrum, as in Block VII; the caller can pin it via kwargs for
            # fast-bottom critical-angle work.
            c_water_min = float(ssp_data[:, 1].min())
            cmins = kwargs.get('cmins_discrete', c_water_min * 0.95)
            cmaxs = kwargs.get('cmaxs_discrete', 1.0e8)
            f.write(f"{cmins:.1f} {cmaxs:.1f}\n")
            # NWDIN ICUT1D ICUT2D (oasnun22.f:421).
            f.write(f"{_oases_nw_line(nw_samples, 2000)}\n")

        # Block X: Replica parameters, gated on IPARES (unoasn22.f:180),
        # which GETOPT sets from 'R' or 'r' alike (unoasn22.f:679-681).
        if _oases_option_chars(options) & {'R', 'r'}:
            # Replica grid: depths, x-ranges, y-ranges.
            #
            # The default axis brackets the water column with a 10 m stand-off
            # off each boundary, which is an open-ocean number: it needs 20 m
            # of water to describe an axis at all. At exactly 20 m the two ends
            # meet; below that they cross, and by 5 m the axis runs 10.00 →
            # -5.00, putting 4 of its 5 points above the sea surface. Nothing
            # downstream refuses it — OASN runs the deck and returns a
            # covariance built on a search grid that is partly not in the
            # ocean — so the stand-off falls back to a tenth of the column on
            # exactly the depths where a flat 10 m leaves no axis. Anything
            # deeper keeps the flat number it already had: at 50 m, 10-40 m is
            # a perfectly good search grid and rescaling it would move existing
            # replica sets for no reason.
            stand_off = 10.0 if depth > 20.0 else 0.1 * depth
            replica_zmin = kwargs.get('replica_zmin', stand_off)
            replica_zmax = kwargs.get('replica_zmax', depth - stand_off)
            if depth <= 20.0 and ('replica_zmin' not in kwargs
                                  or 'replica_zmax' not in kwargs):
                warnings.warn(
                    f"write_oasn_input: {depth:.3g} m of water is too thin for "
                    f"the default 10 m replica stand-off, which needs more "
                    f"than 20 m to leave an axis at all; the derived end(s) "
                    f"were scaled to the column instead, giving a replica "
                    f"depth axis of {replica_zmin:.2f}-{replica_zmax:.2f} m. "
                    f"Pass zmin=/zmax= (OASN) or replica_zmin=/replica_zmax= "
                    f"to choose the search depths yourself.",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
                )
            replica_nz = kwargs.get('replica_nz', 20)
            replica_xmin = kwargs.get('replica_xmin', 0.1)  # km
            replica_xmax = kwargs.get('replica_xmax', 10.0)  # km
            replica_nx = kwargs.get('replica_nx', 50)
            replica_ymin = kwargs.get('replica_ymin', 0.0)  # km
            replica_ymax = kwargs.get('replica_ymax', 0.0)  # km
            replica_ny = kwargs.get('replica_ny', 1)

            over = {name: int(n) for name, n in
                    (('replica_nz', replica_nz), ('replica_nx', replica_nx),
                     ('replica_ny', replica_ny))
                    if int(n) > _OASES_MAX_REPLICA_POINTS}
            if over:
                raise ConfigurationError(
                    f"write_oasn_input: replica grid "
                    f"{', '.join(f'{k}={v}' for k, v in over.items())} "
                    f"exceeds OASN's compiled bound NSMAX = "
                    f"{_OASES_MAX_REPLICA_POINTS} points per axis "
                    f"(oases/src/compar.f:51); the binary would STOP with "
                    f"'>>> TOO MANY REPLICA POINTS <<<' and EXIT CODE 0, "
                    f"leaving no .rpo.",
                    remediation="Coarsen the replica grid to at most "
                                f"{_OASES_MAX_REPLICA_POINTS} points per "
                                f"axis.",
                )

            # ZSMIN ZSMAX NSRCZ / XSMIN XSMAX NSRCX / YSMIN YSMAX NSRCY
            # (unoasn22.f:184-186) — the candidate source positions the
            # replicas are generated for. OASN builds each axis itself by
            # linear interpolation over the count (:197-222), and stops with
            # '>>> TOO MANY REPLICA POINTS <<<' if any count exceeds
            # NSMAX = 201 (unoasn22.f:187-188, oases/src/compar.f:51).
            #
            # The x/y endpoints are in km and OASN interpolates the whole axis
            # from them (:210-222), so the written precision is the grid's
            # own resolution: %.3f km quantises every replica position onto a
            # 1 m lattice, which is coarser than half a wavelength above
            # ~750 Hz and so moves the replica's phase. The reads are
            # list-directed, so match the range axis's %.9f. The z endpoints
            # are already in metres, where %.2f is a centimetre.
            f.write(f"{replica_zmin:.2f} {replica_zmax:.2f} {int(replica_nz)}\n")
            f.write(f"{replica_xmin:.9f} {replica_xmax:.9f} {int(replica_nx)}\n")
            f.write(f"{replica_ymin:.9f} {replica_ymax:.9f} {int(replica_ny)}\n")

            # CMINSIN CMAXSIN (unoasn22.f:226). Same 1E8 upper bound as the
            # discrete-source block — overridable via kwargs.
            c_water_min = float(ssp_data[:, 1].min())
            cmins = kwargs.get('cmins_replica', c_water_min * 0.95)
            cmaxs = kwargs.get('cmaxs_replica', 1.0e8)
            f.write(f"{cmins:.1f} {cmaxs:.1f}\n")
            # NWSIN ICUT1S ICUT2S (unoasn22.f:227).
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
    >>> import tempfile, os
    >>> env = Environment(bathymetry=100, ssp=1500)
    >>> source = Source(depths=80, frequencies=30)
    >>> receiver = Receiver(depths=np.linspace(20,100,5),
    ...                     ranges=np.linspace(1000,5000,5))
    >>> with tempfile.TemporaryDirectory() as d:
    ...     write_oasp_input(os.path.join(d, 'pulse.dat'),
    ...                      env, source, receiver)
    """
    _reject_unknown_kwargs('write_oasp_input', kwargs, _OASP_KWARGS)

    # Extract parameters
    center_freq = kwargs.get('center_frequency', float(source.frequencies[0]))

    # Frequency and time parameters
    n_time = kwargs.get('n_time_samples', 4096)
    freq_min = kwargs.get('freq_min', 0.0)
    freq_max = kwargs.get('freq_max', center_freq * 2.5)

    # DT fixes both the bin spacing DLFREQ = 1/(DT*NX) and the highest bin
    # index MX = FR2/DLFREQ + 2 (unoasp22.f:237-238), so Nyquist against FR2
    # is the natural default. OASP halves whatever it is given until
    # ``DT <= 2.5/FR2`` (unoasp22.f:194-199), doubling NX with it, so a
    # coarser ``time_step`` does not stay coarse.
    if 'time_step' in kwargs:
        dt = kwargs['time_step']
    else:
        dt = 1.0 / (2.0 * freq_max)

    nw_samples = kwargs.get('nw_samples', -1)  # -1 = automatic

    # Options string. Default is single-component (normal stress / pressure)
    # plus the complex-contour integration flag. A second output letter such
    # as ``V`` raises NOUT, and the .trf reader keeps only the first
    # component, so the default asks for one.
    if options is None:
        options = 'N J'
    _reject_unwritten_option_blocks('write_oasp_input', options)
    _unknown = sorted(_oases_option_chars(options) - _OASP_OPTIONS)
    if _unknown:
        raise ConfigurationError(
            f"write_oasp_input: {_unknown} are not "
            f"option letters this binary tests — GETOPT "
            f"(unoasp22.f:827-1053) recognises only "
            f"{''.join(sorted(_OASP_OPTIONS))}. Its ladder ends with a bare ELSE (:1052-1053), so the binary discards them in silence and runs a different configuration than asked for.",
            remediation="Drop the letter(s) from options=.",
        )
    _reject_tau_p('write_oasp_input', options)
    _check_nw_samples('write_oasp_input', nw_samples, power_of_two=False)
    _check_n_time_samples('write_oasp_input', n_time)

    # INTF gates integrand *plots* (unoasp22.f:591
    # `IF (MOD(JJ-LXP1,INTF).EQ.0) KPLOT=1`, consumed at :678) and, being > 0,
    # also turns on the .plp/.plt plot files (:173); the .trf carries every bin
    # LXP1..MX either way. The default 40 is uacpy's own — the manual only
    # requires INTF >= 0, with 0 disabling the plots (oasp.tex:125, :663-664),
    # so the default may only apply when the caller supplied nothing at all.
    intf_arg = kwargs.get('freq_output_increment')

    _write_oasp_family_deck(
        filepath, env, source, receiver, options,
        writer='write_oasp_input',
        title="OASP Simulation via UACPY",
        center_freq=center_freq,
        integration_offset=kwargs.get('integration_offset', 0),
        dip_angle=kwargs.get('dip_angle'),
        cmax=1e9,
        nw_samples=nw_samples,
        block_vii_token4=40 if intf_arg is None else int(intf_arg),
        n_time=n_time, freq_min=freq_min, freq_max=freq_max, dt=dt,
        range_start=kwargs.get('range_start'),
        range_step=kwargs.get('range_step'),
        c_low=kwargs.get('c_low'),
        c_high=kwargs.get('c_high'),
    )


def _oasp_range_axis(writer: str, receiver: Receiver,
                     range_start, range_step) -> Tuple[float, float, int]:
    """``R0``, ``RSPACE`` (km) and ``NPLOTS`` of Block VIII.

    OASP and OASSP both evaluate ``r = R0 + i*RSPACE`` and read nothing else
    about the range axis (``unoasp22.f:176``, ``unoassp30.f:179``), so a
    non-uniform request cannot be honoured — it would silently return a
    different set of ranges than asked for. OAST warns and interpolates;
    neither of these has that option.
    """
    ranges = np.asarray(receiver.ranges, dtype=float)
    r_min_m, r_max_m = float(ranges.min()), float(ranges.max())
    r1_km = float(m_to_km(r_min_m if range_start is None else range_start))

    if range_step is not None:
        return r1_km, float(m_to_km(range_step)), len(ranges)

    n_ranges = len(ranges)
    if n_ranges <= 1:
        return r1_km, 1.0, n_ranges
    steps = np.diff(ranges)
    if not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-9):
        raise ConfigurationError(
            f"{writer}: the deck evaluates a uniform range axis (r0 + i*dr) "
            f"but receiver.ranges is not uniformly spaced (steps "
            f"{steps.min():.4g}-{steps.max():.4g} m).",
            remediation=(
                "Pass uniformly spaced ranges (np.linspace), or run "
                "on a uniform grid and interpolate the Field "
                "afterwards."),
        )
    return r1_km, float(m_to_km((r_max_m - r_min_m) / (n_ranges - 1))), n_ranges


def _write_oasp_family_deck(
    filepath: Union[str, Path],
    env: Environment,
    source: Source,
    receiver: Receiver,
    options: str,
    *,
    writer: str,
    title: str,
    center_freq: float,
    integration_offset,
    dip_angle: Optional[float],
    cmax: float,
    nw_samples,
    block_vii_token4: int,
    n_time: int,
    freq_min: float,
    freq_max: float,
    dt: float,
    range_start=None,
    range_step=None,
    c_low: Optional[float] = None,
    c_high: Optional[float] = None,
    roughness_tail: Optional[Callable[[int], str]] = None,
) -> None:
    """Write the eight-block deck OASP and OASSP share.

    ``grep -n 'READ(1,\\*)' unoasp22.f unoassp30.f`` gives the same record
    sequence with the same arities — OASSP is OASP's deck token for token. The
    two differences are parameters here rather than a second copy of the body:

    * **Block VII token 4.** OASP's ``INTF`` gates integrand plots
      (``unoasp22.f:160``, ``:591``). OASSP reads the same column into
      ``msft_0`` and derives the realization seed ``ISEED = -123 - msft_0``
      from it (``unoassp30.f:170``, ``:535``), having disabled kernel plotting
      unconditionally at ``:172``.
    * **Block IV's roughness column.** OASSP's scattering interface must carry
      the nine-token ``… -|RG| CL M`` form, which only a ``suffix_fn`` can
      emit; OASP's records stop at the seven INENVI reads plus inert padding.

    Everything else — Block III's two tokens, the layer stack, the single
    ``SD`` source record, the three-token receiver record OASSP shares with
    OASP by virtue of ``PROGNM`` (``oaseun31.f:1165``), and Block VIII's seven
    fields — is identical, so it lives here once.
    """
    filepath = Path(filepath)
    depth = env.depth

    bottom = env.bottom.halfspace_at(range=0.0)
    _bp = _extract_bottom_props(bottom)
    rho, c_p, c_s = _bp['rho'], _bp['c_p'], _bp['c_s']
    alpha_p, alpha_s = _bp['alpha_p'], _bp['alpha_s']

    # Sound speed profile — align to env.depth (see OAST writer).
    ssp_data = env.ssp.extend_to(depth).to_pairs()

    # Source parameter (receiver depth bookkeeping handled by
    # ``_receiver_block_lines`` which emits equidistant/explicit as needed).
    src_depth = _single_source_depth(writer, source)

    r1_km, dr_km, nr = _oasp_range_axis(writer, receiver,
                                        range_start, range_step)

    cmin, cmax = oases_wavenumber_bounds(ssp_data, cmax=cmax)
    if c_low is not None:
        cmin = float(c_low)
    if c_high is not None:
        cmax = float(c_high)

    source_record = _resolve_source_record(writer, options, dip_angle)

    _warn_volume_attenuation_ignored(env)

    with open(filepath, 'w') as f:
        _write_oases_header(f, env, options, title)

        # Block III: FREQS OFFDBIN — the pulse centre frequency and the
        # contour offset (unoasp22.f:129, unoassp30.f:138). OFFDBIN is only
        # read under ICNTIN > 0, i.e. option 'J'; without it the binary
        # zeroes the offset itself and the second token is discarded by the
        # list-directed read (unoasp22.f:131-132, unoassp30.f:140-142).
        # Option 'd' would demand a 5-token Doppler form and is rejected by
        # the callers.
        f.write(f"{center_freq:.9f} {integration_offset}\n")

        # Block IV: Environment — NL then NL layer records, read by INENVI
        # from unoasp22.f:144 / unoassp30.f:155. NL = upper halfspace + water
        # + sediments + bottom halfspace.
        geom = _oasp_layer_geometry(env)
        f.write(f"{geom['n_layers']}\n")

        f.write(f"{_format_upper_halfspace(env)}\n")
        _emit_water_layers(f, geom['ssp_array'],
                           surface_roughness=_surface_roughness(env),
                           extra_columns=2)
        if roughness_tail is None:
            _emit_bottom_layers(
                f, env, depth,
                c_p, c_s, alpha_p, alpha_s, rho,
                extra_columns=2,
            )
        else:
            _emit_bottom_layers(
                f, env, depth,
                c_p, c_s, alpha_p, alpha_s, rho,
                suffix_fn=roughness_tail,
                iface_start=1,
            )

        # Block V: Sources — one source at src_depth, in the record shape
        # INSRC's LINA / dip_sou branch expects (called from unoasp22.f:148,
        # unoassp30.f:157).
        f.write(_source_block_line(src_depth, **source_record) + '\n')

        # Block VI: Receiver depths — RD RDLOW IR, read by INREC
        # (unoasp22.f:149 / unoassp30.f:158 → oaseun31.f:1165; three tokens,
        # not OAST's four, because neither PROGNM is 'OASTL'). NRD<0 signals
        # an explicit depth list on the following record (oasp.tex:559-585).
        for line in _receiver_block_lines(receiver):
            f.write(line + '\n')

        # Block VII: Wavenumber sampling — CMIN CMAX (unoasp22.f:159,
        # unoassp30.f:168).
        f.write(f"{cmin:.1f} {_OASES_CMAX_FMT.format(cmax)}\n")

        # NWVNO ICW1 ICW2 <token4> (unoasp22.f:160, unoassp30.f:169). Anything
        # below 1 selects automatic sampling (``AUSAMP=(NWVNO.LT.1)``), which
        # recomputes ICW1/ICW2 from AUTSAM and overwrites ICUT1/ICUT2
        # (unoasp22.f:341-342), so the pair is inert there — oasp.tex:677 says
        # as much. A pinned NW needs ICW2 = NW or the Hankel transform is
        # Hanning-windowed away over ICW2+1..NW before integration
        # (oasp.tex:657-660).
        if nw_samples is None or nw_samples <= 0:
            ic1, ic2 = 1, 1
        else:
            ic1, ic2 = 1, int(nw_samples)
        # nw_samples=None means automatic sampling, the -1 the binaries read.
        f.write(f"{int(nw_samples if nw_samples is not None else -1)} "
                f"{int(ic1)} {int(ic2)} {block_vii_token4}\n")

        # Block VIII: NX FR1 FR2 DT R0 RSPACE NPLOTS (unoasp22.f:176,
        # unoassp30.f:179) — the FFT length, band edges, time step and the
        # uniform range axis r = R0 + i*RSPACE the .trf is written on.
        # R0/RSPACE are km, so %.3f would be 1 m resolution: sub-metre receiver
        # spacing rounds to zero (every receiver onto one range) and metre-scale
        # spacing accumulates drift over the axis. %.9f is sub-micron.
        # The read is list-directed, so the extra digits are free.
        # DT at full precision: %.6f is 1 us resolution, which writes 0.0 at
        # freq_max = 1 MHz (the binary then substitutes dt = 2.5/fr2,
        # unoasp22.f:194-201, silently shrinking the band by 5x).
        f.write(f"{int(n_time)} {freq_min:.9f} {freq_max:.9f} {dt:.12g} "
                f"{r1_km:.9f} {dr_km:.9f} {nr}\n")


def _oasp_layer_geometry(env: Environment) -> dict:
    """Block IV's layer counts, resolved once so callers can key on them.

    ``write_oassp_input`` needs ``n_water_layers`` *before* the deck is
    written, because the OASES interface index its roughness override is keyed
    on counts deck layers from 1 while ``_emit_bottom_layers``' ``suffix_fn``
    counts from its own first record.

    The OASP-family deck emits one water layer per SSP row and never
    collapses an isovelocity column, unlike OAST/OASS (whose index space
    :func:`oass_bottom_interfaces` mirrors); for the same isovelocity
    environment the first bottom record is deck layer
    ``2 + n_water_layers`` here versus 3 there. Interface indices are
    per-deck and must not be carried between the two families.
    """
    n_sed_layers = _count_bottom_layers(env)
    ssp_data = env.ssp.extend_to(env.depth).to_pairs()
    ssp_array = np.asarray(
        _check_ssp_layer_count(ssp_data, 2 + n_sed_layers), dtype=float,
    ).reshape(-1, 2)
    n_water_layers = len(ssp_array)
    return {
        'ssp_array': ssp_array,
        'n_water_layers': n_water_layers,
        'n_sed_layers': n_sed_layers,
        'n_layers': 1 + n_water_layers + n_sed_layers + 1,
    }


#: Option letters ``GETOPT`` tests at ``unoassp30.f:887-1046``. Its closing
#: ``ELSE`` is EMPTY (``:1049-1050``) — unlike OASS (``unoass21.f:688-690``)
#: and OAST (``unoast31.f:1102-1104``), which print
#: ``>>>> UNKNOWN OPTION <<<<`` — so a typo'd or wrong-case letter is discarded
#: with no diagnostic anywhere. The wrapper therefore validates the string
#: itself (spec guard G9).
_OASSP_OPTIONS = frozenset('BANVHRKSGLlvPZJFfOX2m3CUTQtdsgprb')

#: Option letters each ``GETOPT`` actually tests. Extracted from the
#: ``opt(i).eq.'x'`` comparisons, which are written in BOTH cases in these
#: files — a case-sensitive read of the ladder misses ``'4'`` (dip-slip
#: source, ``unoast31.f:1117``) and every other lowercase-tested letter.
#:
#: What an unrecognised letter costs differs per binary, so the three checks
#: below say different things:
#:   * OASP's ladder ends with a bare ``ELSE`` (``unoasp22.f:1051-1052``) and
#:     OASN's with a bare ``END IF`` (``unoasn22.f:727``) — the letter is
#:     dropped in silence and the run proceeds mis-configured.
#:   * OAST is NOT silent: ``:1166-1168`` prints '>>>> UNKNOWN OPTION: x <<<<'
#:     to unit 6. uacpy captures stdout rather than surfacing it, so the
#:     notice never reaches the caller and the letter is dropped just the same.
#: OASN additionally reads digits 1-9 as the source-directionality order MFAC
#: through an ``ICHAR`` range test (``:719-725``), not as named letters, so
#: they belong in its set even though no ``opt(i).eq.'1'`` appears.
_OAST_OPTIONS = frozenset('2345aAbcCdDEfFghHiIJKlLmNoOpPQrRsStTvVXZ')
_OASN_OPTIONS = frozenset('123456789bcCdDfFiIjJkKnNpPqQrRtTzZ')
_OASP_OPTIONS = frozenset('23458ABCdEfFgGhHJKlLmNOPQRsStTUvVxXZ')


#: Letters whose output the ``.trf`` reader cannot represent. ``U`` splits the
#: field over five files (see :func:`write_oassp_input`); the rest add an
#: ``IOUT`` slot, i.e. a second component in every ``.trf`` record that
#: ``read_oasp_trf`` flattens onto the first (``unoassp30.f:897-931``:
#: N→1 V→2 H→3 R→5 K→6 S→7).
_OASSP_MULTI_COMPONENT = frozenset('VHRKSU')


def write_oassp_input(
    filepath: Union[str, Path],
    env: Environment,
    source: Source,
    receiver: Receiver,
    options: Optional[str] = None,
    *,
    interface: int,
    correlation_length: float,
    n_time_samples: int,
    freq_min: float,
    freq_max: float,
    time_step: float,
    spectral_exponent: float = 2.0,
    rms_roughness: Optional[float] = None,
    realization: int = 0,
    center_frequency: Optional[float] = None,
    integration_offset: float = 0.0,
    nw_samples: int = -1,
    c_low: Optional[float] = None,
    c_high: Optional[float] = None,
    range_start: Optional[float] = None,
    range_step: Optional[float] = None,
    **kwargs,
) -> None:
    """Write an OASSP (OASES scattered-field realization) input deck.

    OASSP is the *consumer* half of a two-binary chain: it reads the mean field
    an OASP run with option ``'s'`` left in the ``.rhs`` (and its ``.vol``
    sibling) and generates one **realization** of the field scattered from a
    rough interface. Its output is an OASP ``.trf``, written by the same
    ``TRFHEAD`` path (``oasiun23.f:842-845``), so the deck below is
    OASP's token for token — :func:`_write_oasp_family_deck` writes both.

    Parameters
    ----------
    filepath, env, source, receiver
        As for the sibling writers.
    options : str, optional
        OASSP option letters, validated against ``GETOPT``. ``None`` writes
        ``'N J s'``: normal stress, real wavenumber contour, scattered field
        only. See Notes for why every letter of that default is load-bearing.
    interface : int
        ``INTFCE``, the 1-based OASES layer index whose roughness scatters.
        This is **not** a deck token — OASSP reads it out of the ``.rhs``
        (``unoassp30.f:546-547``) — but the deck's own layer record for that
        index must carry the nine-token ``… -|RG| CL M`` form, because
        ``:601-607`` takes ``RG2 = ROUGH(INTFCE)**2`` and
        ``r_l = |CLEN(INTFCE)|`` from INENVI's arrays. Pass the index the
        ``.rhs`` names (:func:`~uacpy.io.oases_reader.read_oases_rhs_header`).
    correlation_length, spectral_exponent
        ``CL`` and ``M`` of the roughness power spectrum, in metres and
        dimensionless. A **negative** ``CL`` is OASES' switch to the 12-token
        volume-scattering record and is refused — see Notes.
    rms_roughness : float, optional
        ``|RG|`` at ``interface`` (m). Defaults to the environment's own value
        for that interface.
    realization : int
        Block VII's fourth token. See Notes.
    n_time_samples, freq_min, freq_max, time_step
        Block VIII's ``NT``, ``FR1``, ``FR2``, ``DT``. **Required, and they
        must be the mean field's own** — OASSP overwrites all four from the
        ``.rhs`` at ``unoassp30.f:181-188`` but its mismatch test covers only
        ``NT`` and ``DT``, so a band that differs from the mean field's is
        replaced with no message at all.
    center_frequency : float, optional
        Block III's ``FRC``. Defaults to the source's first frequency.
    integration_offset, nw_samples, c_low, c_high, range_start, range_step
        As for :func:`write_oasp_input`.

    Notes
    -----
    **Block VII's fourth token is not a plot increment.** ``unoassp30.f:170``
    assigns ``msft_0 = intf`` and ``:172`` then zeroes ``intf``, so kernel
    plotting is unconditionally off and the manual's description
    (``oassp.tex:503-504``) does not describe this binary. What the token does
    reach is the random-number seed, ``ISEED = -123 - msft_0`` (``:535``) —
    which hands uacpy a reproducible ensemble — and, through
    ``mbf_0 = msft_0/2`` (``:171``), the tapered full-Bessel tabulation window
    ``brk_0``/``bfrmax`` in cylindrical geometry (``:639-640``). OASP pins
    ``mbf_0 = 0`` (``unoasp22.f:568``), so that second effect is OASSP's alone:
    successive realizations are not a *pure* statistical ensemble under
    cylindrical geometry, they also widen the Bessel table by ``k/2`` in ``kr``.

    **The default option string.** ``'N'`` is the single output component the
    ``.trf`` reader keeps; ``'J'`` holds ``ICNTIN = 1`` so automatic wavenumber
    sampling takes the complex *wavenumber* contour rather than the complex
    *frequency* contour ``OMEGIM = -ln(50)·Δf`` (``unoassp30.f:281-289``,
    ``:382-385``), which uacpy's time-series synthesis does not undo; ``'s'``
    sets ``SCTOUT``, which zeroes the source arrays (``:628-635``) so the
    ``.trf`` holds the scattered field alone rather than scattered + direct.
    ``'s'`` cannot overwrite the input ``.rhs`` here: the write at
    ``oaseun31.f:1899`` is inside ``CALINT``, and OASSP integrates through
    ``CALSRC``/``CALSRP`` in ``oasvun31.f`` instead (``unoassp30.f:684-687``).

    **Volume scattering is not supported.** OASES selects it with a negative
    ``CL`` on the layer record, which makes INENVI read twelve tokens
    ``D CC CS AC AS RO RG CL SKW M RMS GAM`` (``oaseun31.f:76-90``). The last
    four — skew, spectral exponent, Δc/c and the density-to-speed ratio γ —
    have no home on any uacpy carrier, so a negative ``correlation_length``
    raises rather than writing three of the twelve columns as zeros.
    """
    _reject_unknown_kwargs('write_oassp_input', kwargs, frozenset())

    if options is None:
        options = 'N J s'
    chars = _oases_option_chars(options)

    unknown = sorted(chars - _OASSP_OPTIONS)
    if unknown:
        raise ConfigurationError(
            f"write_oassp_input: {unknown} are not OASSP option letters — "
            f"GETOPT (unoassp30.f:887-1046) tests only "
            f"{''.join(sorted(_OASSP_OPTIONS))}. Its closing ELSE is empty "
            f"(:1049-1050), so the binary would discard them in silence and "
            f"run a different configuration than asked for.",
            remediation="Drop the letter(s) from options=.",
        )
    _reject_unwritten_option_blocks('write_oassp_input', options)
    _reject_tau_p('write_oassp_input', options)

    multi = sorted(chars & _OASSP_MULTI_COMPONENT)
    if multi:
        # 'U' is the only one that is not merely a second IOUT slot: DECOMP
        # splits the output over NCOMPO=5 files opened one unit apart
        # (unoassp30.f:459-465, :505-508). Chased through KERDEC
        # (oasaun31.f:1626-1648) and the buffer→unit pairing at
        # oasvun31.f:779, :862-869 and unoassp30.f:725-726, the DATA order at
        # unoassp30.f:48-52 is the correct one: trf=total, trfdc=down-going
        # compressional, trfds=down-going shear, trfuc=up-going compressional,
        # trfus=up-going shear. oassp.tex:185-189 has 'ds' and 'uc' swapped.
        # Either way uacpy reads one .trf, so the letter is refused.
        raise UnsupportedFeatureError(
            'OASSP',
            f"option letter(s) {multi} — multi-component / decomposed output. "
            f"'U' writes five separate transfer-function files (trf, trfdc, "
            f"trfds, trfuc, trfus; unoassp30.f:48-52, :505-508) and the "
            f"others add an IOUT component that read_oasp_trf flattens onto "
            f"the first",
            alternatives=["options without these letters (default 'N J s' "
                          "returns scalar pressure)"],
            alternatives_label='option strings',
        )

    if float(correlation_length) < 0.0:
        raise UnsupportedFeatureError(
            'OASSP',
            "volume scattering (correlation_length < 0 is OASES' switch to "
            "the twelve-token layer record D CC CS AC AS RO RG CL SKW M RMS "
            "GAM, oaseun31.f:76-90). SKW (correlation-ellipse skew, deg), M "
            "(spectral exponent), RMS (dc/c) and GAM (density-to-speed "
            "ratio) have no field on any uacpy carrier",
            alternatives=["interface roughness scattering "
                          "(correlation_length > 0)"],
            alternatives_label='configurations',
        )
    if float(correlation_length) == 0.0:
        raise ConfigurationError(
            "write_oassp_input: correlation_length must be positive — OASSP "
            "forms r_l = |CLEN(INTFCE)| and hands it to PV "
            "(unoassp30.f:606-614), so a zero correlation length gives a "
            "degenerate roughness power spectrum."
        )
    if float(spectral_exponent) <= 1.5:
        raise ConfigurationError(
            f"write_oassp_input: spectral_exponent must exceed 1.5 or the "
            f"roughness power spectrum is not integrable (oassp.tex:356-362; "
            f"the exponent reaches amod(m)=fac(3+…) at oaseun31.f:99); got "
            f"{spectral_exponent}.")
    if int(realization) < 0:
        raise ConfigurationError(
            f"write_oassp_input: realization must be >= 0; it is written as "
            f"Block VII's fourth token and OASSP forms ISEED = -123 - "
            f"realization from it (unoassp30.f:170, :535), so a negative "
            f"value walks the seed towards 0. Got {realization}.")
    _check_nw_samples('write_oassp_input', nw_samples, power_of_two=False)
    _check_n_time_samples('write_oassp_input', n_time_samples)

    if c_high is not None and 'P' not in chars:
        # unoassp30.f:205-217: without 'P' the run is cylindrical, and CMAXIN
        # is then forced to 1e12 with inttyp driven to full Bessel. The deck
        # still carries the user's value; the binary throws it away.
        warnings.warn(
            f"write_oassp_input: c_high={c_high:g} m/s has no effect without "
            f"option 'P'. In cylindrical geometry OASSP forces CMAX = 1e12 "
            f"and a full Hankel transform (unoassp30.f:205-217), printing "
            f"'>>> Full Hankel transform forced <<<'. Add 'P' for plane "
            f"geometry, or drop c_high.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

    # INTFC counts deck layers from 1 (the upper halfspace), but the suffix_fn
    # _emit_bottom_layers calls is indexed from its own first bottom record.
    # Bottom records begin at deck layer ``1 + n_water_layers + 1``, so that
    # record is suffix index 1 — hence the offset below. Keying the override
    # off INTFC directly lands on the wrong record, and the failure is SILENT:
    # the named interface then carries a non-negative RG, INENVI leaves
    # CLEN = 0 (oaseun31.f:102), and unoassp30.f:606-615 calls PV with a zero
    # correlation length, from a run that exits 0.
    geom = _oasp_layer_geometry(env)
    first_bottom_layer = 1 + geom['n_water_layers'] + 1
    suffix_index = int(interface) - (first_bottom_layer - 1)
    if not 1 <= suffix_index <= geom['n_sed_layers'] + 1:
        raise ConfigurationError(
            f"write_oassp_input: interface={interface} is not a bottom "
            f"interface of this deck. It has {geom['n_water_layers']} water "
            f"layer(s), so the scattering interfaces are deck layers "
            f"{first_bottom_layer}.."
            f"{first_bottom_layer + geom['n_sed_layers']}.",
            remediation=("The roughness spectrum can only be attached to a "
                         "bottom layer record; a water interface has no RG "
                         "column INENVI would re-read as CL and M."),
        )

    env_rgs = bottom_interface_roughness(env)
    if rms_roughness is None:
        rg = env_rgs[min(suffix_index - 1, len(env_rgs) - 1)]
    else:
        rg = abs(float(rms_roughness))
    if abs(rg) <= 1e-5:
        raise ConfigurationError(
            f"write_oassp_input: interface {interface} has rms roughness "
            f"{rg:g} m, so there is nothing to scatter from. OASSP takes "
            f"pow = |ROUGH(INTFCE)| straight from this deck "
            f"(unoassp30.f:601, :613) and the scattered field would be "
            f"identically zero; the mean-field run is worse off still, since "
            f"SCTRHS skips every interface with ROUGH2 < 1e-10 "
            f"(oaseun31.f:2310, :379) and would write an empty .rhs.",
            remediation=("Set the roughness on the environment "
                         "(BoundaryProperties(roughness=…) or the sediment "
                         "layer), or pass rms_roughness=."),
        )

    _write_oasp_family_deck(
        filepath, env, source, receiver, options,
        writer='write_oassp_input',
        title="OASSP Simulation via UACPY",
        center_freq=(float(np.atleast_1d(source.frequencies)[0])
                     if center_frequency is None else float(center_frequency)),
        integration_offset=integration_offset,
        dip_angle=None,
        # Cylindrical geometry forces CMAX = 1e12 anyway (:205-217); the OASP
        # writer's own 1e9 default is kept so the two decks agree wherever the
        # binary does not override.
        cmax=1e9,
        nw_samples=nw_samples,
        block_vii_token4=int(realization),
        n_time=int(n_time_samples), freq_min=float(freq_min),
        freq_max=float(freq_max), dt=float(time_step),
        range_start=range_start, range_step=range_step,
        c_low=c_low, c_high=c_high,
        roughness_tail=_make_roughness_tail(
            env,
            {suffix_index: (rg, float(correlation_length),
                            float(spectral_exponent))},
        ),
    )


REFL_TYPE_TO_OPTION = {
    'P-P': 'N',          # default — P-wave to P-wave reflection
    'P-SV': 'S',         # P-wave to vertical shear
    'P-Slow': 'B',       # P-wave to Biot slow wave
    'transmission': 't',  # transmission instead of reflection
}

#: The OASR option letters that raise ``NOUT`` — ``N`` sets ``IOUT(1)``,
#: ``S`` ``IOUT(2)``, ``B`` ``IOUT(3)`` (unoasr21.f:350-372). ``'t'`` is not
#: among them: it flips ``transmit`` and leaves ``NOUT`` alone (:373), which
#: is why ``'N t'`` is a legal pair.
_OASR_FIELD_PARAMETER_OPTIONS = frozenset('NSB')


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
    interface_roughness : list, optional
        Per-interface RMS roughness in metres, one entry per layer record,
        ordered top → bottom. Entry 0 is the layer-1 (water half-space)
        record, whose RG INENVI discards (``oaseun31.f:377``); entry 1 is the
        reflecting water/seabed interface. A float is an RMS roughness; a
        ``(RG, CL, M)`` tuple or ``{'RG', 'CL', 'M'}`` dict selects the
        Goff-Jordan spectrum and is the only route to OASES' negative-RG form.
        Unset entries fall back to the environment's own roughness.
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
    >>> import tempfile, os
    >>> env = Environment(bathymetry=100, ssp=1500)
    >>> env.bottom.sound_speed = 1600
    >>> env.bottom.shear_speed = 400
    >>> source = Source(depths=50, frequencies=100)
    >>> receiver = Receiver(depths=[50], ranges=[1000])
    >>> with tempfile.TemporaryDirectory() as d:
    ...     write_oasr_input(os.path.join(d, 'test.dat'), env, source,
    ...                      receiver, angle_min=0, angle_max=90, n_angles=91)
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
    # source is placed 1 mm above the top of layer 2 (unoasr21.f:89), and
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
    if reflection_type not in REFL_TYPE_TO_OPTION:
        raise ConfigurationError(
            f"OASR: reflection_type must be one of {list(REFL_TYPE_TO_OPTION)}, "
            f"got {reflection_type!r}"
        )
    if options is None:
        opt_letter = REFL_TYPE_TO_OPTION[reflection_type]
        options = f"{opt_letter} T"
    _reject_unwritten_option_blocks('write_oasr_input', options)
    field_params = sorted(
        _oases_option_chars(options) & _OASR_FIELD_PARAMETER_OPTIONS)
    if len(field_params) > 1:
        raise ConfigurationError(
            f"write_oasr_input: options={options!r} names {field_params} — "
            f"{len(field_params)} field parameters. OASR computes one "
            f"coefficient per run and STOPs with '*** ONLY ONE FIELD "
            f"PARAMETER ALLOWED***' (unoasr21.f:429), a character STOP that "
            f"exits 0 with no .rco written.",
            remediation="Keep one of 'N' (P-P), 'S' (P-SV) or 'B' (P-Slow) "
                        "and run a second deck for the other, or pass "
                        "reflection_type= and let the writer choose the "
                        "letter.",
        )
    _check_frequency_contours('write_oasr_input', options, 'C', n_frequencies)

    _warn_volume_attenuation_ignored(env, lossless_water=True)

    # Interface roughness (RG / CL / M) per interface. Index 0 is OASR's
    # layer-1 record, whose RG is the dummy INENVI overwrites with ROUGH(2)
    # (oaseun31.f:377); 1.. are the bottom interfaces _emit_bottom_layers
    # writes, index 1 being the reflecting water/seabed interface. OASR has
    # no sea surface — layer 1 IS the water halfspace — so env.surface's
    # roughness has nothing to attach to here and the default is 0.
    _roughness_tail = _make_roughness_tail(env, interface_roughness)

    with open(filepath, 'w') as f:
        _write_oases_header(f, env, options, "OASR Simulation via UACPY")

        # Block III: Environment — NL then NL layer records, read by INENVI
        # from unoasr21.f:85.
        # OASR convention: layer 1 IS the upper halfspace in which the source
        # sits — ``SD=V(2,1)-1.0E-3``, 1 mm above layer 2's top
        # (unoasr21.f:89). Reflection is computed at the
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

        # Layer 1: water as upper halfspace (D is dummy for layer 1, and so
        # is its RG — "RG(1) is dummy", oast.tex:344-345, because INENVI
        # replaces ROUGH(1) with ROUGH(2) at oaseun31.f:377).
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

        # Block IV: FREQ1 FREQ2 NFREQ IOUTF (unoasr21.f:116). IOUTF is the
        # manual's NFOU, the output/plot increment along frequency.
        f.write(f"{freq_min:.9f} {freq_max:.9f} {n_frequencies} {freq_out_inc}\n")

        # Block V: ANGLE1 ANGLE2 NANG IOUTA (unoasr21.f:128). OASR builds the
        # grid itself as ANGLE1 + (i-1)*DLANGLE (:173-177, evaluated at :284),
        # which is why ``angles`` must be uniformly spaced. IOUTA is NAOU.
        f.write(f"{angle_min:.6f} {angle_max:.6f} {int(n_angles)} "
                f"{angle_out_inc}\n")

        # Blocks VI and VII are plot axes, and their gates are crossed over:
        # the angle axes here ride on NIPLOT, which is non-zero iff IOUTF > 0
        # (unoasr21.f:160-161), and the frequency axes below ride on NAPLOT,
        # non-zero iff IOUTA > 0 (:167-168). Both counts are formed by integer
        # division and cannot be zero once their increment is positive, so
        # testing the increment is the same test.

        # Block VI: ALEF ARIG ALEN AINC / RALO RAUP RALN RAIN
        # (unoasr21.f:162-163) — the angle axis over a 12 cm plot, then the
        # reflection-coefficient magnitude over its full 0-1 range.
        if freq_out_inc > 0:
            f.write(f"{angle_min:.6f} {angle_max:.6f} 12 {max(10, (angle_max-angle_min)/10):.1f}\n")
            f.write("0 1 12 0.2\n")

        # Block VII: FLEF FRIG FLEN FINC / RFLO RFUP RFLN RFIN
        # (unoasr21.f:169-170) — the frequency axis, then a 0-30 dB
        # reflection-loss axis. f_range floors the tick spacing at a tenth of
        # FREQ1 so a single-frequency deck does not ask for a zero increment.
        if angle_out_inc > 0:
            f_range = max(freq_max - freq_min, freq_min * 0.1)
            f.write(f"{freq_min:.9f} {freq_max:.9f} 12 {f_range/10:.9f}\n")
            f.write("0 30 12 5\n")

        # Block VIII: Loss contours in angle and frequency — three rows read
        # under CONTUR (unoasr21.f:178-181). OASR's GETOPT (:386-388) sets
        # CONTUR from uppercase 'C' only — it defines no lowercase 'c', so
        # matching one would emit a block OASR never reads and shift the 'Z'
        # block below onto it.
        if 'C' in _oases_option_chars(options):
            # ALEF ARIG ALEN AINC / FRLO FRUP OCLN NTKM / ZMIN ZMAX ZINC.
            # The contour plot's frequency axis is logarithmic — OASR steps
            # it in LOG(FREQ) (unoasr21.f:123-125) — so OCLN is the length of
            # ONE octave in cm and NTKM the tickmarks per octave
            # (oasr.tex:101-102), not a whole-axis length.
            f.write(f"{angle_min:.6f} {angle_max:.6f} 12 {(angle_max-angle_min)/10:.1f}\n")
            octave_range = np.log2(freq_max / freq_min) if freq_max > freq_min else 1.0
            f.write(f"{freq_min:.9f} {freq_max:.9f} {octave_range*2:.9f} 5\n")
            f.write("0 20 2\n")  # Contour levels 0-20 dB in 2 dB increments

        # Block IX: velocity-profile plot axes — two rows read under IPROF
        # (unoasr21.f:214-216), which GETOPT sets from uppercase 'Z' only
        # (:392-394).
        if 'Z' in _oases_option_chars(options):
            # VLEF VRIG VLEN VINC / DVUP DVLO DVLN DVIN: a speed axis spanning
            # the water and sediment speeds with 5% headroom, then the water
            # column over a 12 cm axis.
            c_min = min(c_water, c_p) * 0.95
            c_max = max(c_water, c_p) * 1.05
            f.write(f"{c_min:.1f} {c_max:.1f} 12 100\n")
            f.write(f"0 {depth:.1f} 12 {depth/10:.1f}\n")


#: Option letters ``GETOPT`` tests in ``unoass21.f:557-698``. Anything else
#: prints ``>>>> UNKNOWN OPTION: <c> <<<<`` (``:688-690``) and is ignored, so
#: the wrapper validates rather than let a typo vanish.
_OASS_OPTIONS = frozenset('CDGgIPRSacprdkZQ')

#: Letters that set ``REVERB`` on the way in — ``C`` (:626), ``D`` (:635),
#: ``R`` (:616), ``a`` (:646), ``r`` (:607). Their presence is what makes
#: Block VIII a required record.
_OASS_REVERB_OPTIONS = frozenset('CDRar')

#: Integrand/spectrum plot letters. ``unoass21.f`` computes them in the
#: ``IF (.not.REVERB)`` branch only (``:342-344``), so pairing one with a
#: reverberation letter silently produces no plot.
_OASS_KERNEL_OPTIONS = frozenset('ISc')

#: The scattered-field *products*, one per run. ``oassun26.f:683-688`` versus
#: ``:897-902``: asking for the covariance alongside a reverberation curve
#: returns a silently ZERO covariance.
_OASS_PRODUCT_OPTIONS = frozenset('arCD')


def oass_bottom_interfaces(env: Environment):
    """``(deck index of the first bottom interface, RMS roughness per bottom
    interface)``, in the order the deck emits them.

    Mirrors the layer block :func:`write_oass_input` writes — one record for
    an isovelocity water column, one per SSP row otherwise, on top of the
    upper half-space — because ``INTFC`` indexes *deck layers*, and uacpy has
    to name one without making the caller count records. Kept in this module
    so a change to the layer emission sits next to the numbering it implies.

    This is the **OASS/OAST** index space. The OASP/OASSP decks never
    collapse an isovelocity column (:func:`_oasp_layer_geometry` counts one
    water layer per SSP row), so for the same isovelocity environment the
    seafloor is deck layer 3 here but layer ``2 + n_ssp_rows`` in an OASSP
    deck — each helper is verified against its own written deck, and the
    two must not be mixed.
    """
    ssp_data = env.ssp.extend_to(float(env.depth)).to_pairs()
    # The writer emits the SSP-decimation warning where it matters, so this
    # count-only call asks that one warning for silence. A
    # ``warnings.catch_warnings()`` window here would mute UserWarnings
    # process-wide for its duration, other threads' included.
    ssp_subset = _check_ssp_layer_count(
        ssp_data, 2 + _count_bottom_layers(env), warn=False)
    c_values = ssp_subset[:, 1]
    n_water = (1 if np.allclose(c_values, c_values[0], rtol=1e-6)
               else len(ssp_subset))
    return 2 + n_water, bottom_interface_roughness(env)


def write_oass_input(
    filepath: Union[str, Path],
    env: Environment,
    source: Source,
    receiver: Receiver,
    options: str,
    *,
    interface: int,
    correlation_length: float,
    spectral_exponent: float,
    rms_roughness: Optional[float] = None,
    phase_speed: float = 0.0,
    c_low: Optional[float] = None,
    c_high: Optional[float] = None,
    range_min: Optional[float] = None,
    range_max: Optional[float] = None,
    n_ranges: Optional[int] = None,
    receiver_gains=None,
    receiver_types=None,
    **kwargs,
) -> None:
    """Write an OASS (OASES reverberation / scattered-field) input deck.

    OASS is the *consumer* half of a two-binary chain: it reads the mean
    field a producer run left in the ``.rhs`` file and integrates the
    scattered field over it. The deck below is only the consumer's; the
    producer deck is the caller's business (see :class:`~uacpy.OASS`).

    Record order follows the reads in ``unoass21.f``; block numbers are the
    manual's (``oass.tex:40-106``).

    Parameters
    ----------
    filepath, env, source, receiver
        As for the sibling writers.
    options : str
        OASS option letters. **Required and validated** — there is no safe
        default, because the letter chooses the product (see Notes).
    interface : int
        ``INTFC``, the 1-based interface index the scattering is computed at
        (``unoass21.f:151``); must be >= 2.
    correlation_length, spectral_exponent
        ``CL`` and ``M`` of the roughness power spectrum. OASS *requires*
        them — ``oass.tex:182-183``: "the interface rms roughness,
        correlation and spectral exponent **must** be specified. These are
        not adopted from OASR or OAST."
    rms_roughness : float, optional
        ``|RG|`` at ``interface``. Defaults to the environment's own value
        for that interface.
    phase_speed : float
        ``CPH`` (m/s), a dummy unless one of ``I``/``S``/``c`` is requested.
    c_low, c_high : float, optional
        ``CMIN``/``CMAX``. ``c_low`` is physically significant, not a tuning
        knob — see Notes.
    range_min, range_max : float, optional
        Block VIII range bounds in **metres** (converted to km on the way
        out). Default to the receiver's own range span.
    n_ranges : int, optional
        ``NR``; must be >= 2.
    receiver_gains, receiver_types
        Passed to :func:`_emit_receiver_array`. ``receiver_types`` selects
        the field parameter and may not be mixed.

    Notes
    -----
    **There is no field-parameter option letter in OASS.** ``GETOPT`` never
    assigns ``IOUT``/``NOUT``; it ends ``IF (NOUT.NE.0) RETURN / IOUT(1)=1 /
    NOUT=1`` (``unoass21.f:694-696``), and the parameter comes from the
    receiver ``ITYP`` column instead (``oasnun22.f:97-118``). So the ``'N'``
    prefix the OAST/OASP writers use is *not* an OASS option — the manual's
    own worked example ``oass.tex:305`` (``P N I S``) makes the binary print
    ``>>>> UNKNOWN OPTION: N <<<<``.

    **``R`` means "reverberant field", not "radial stress".** In
    OAST/OASP/OASSP it selects σ_rr; here it is the master reverberation
    switch (``unoass21.f:616``). A shared option-translation table across the
    OASES family silently turns one into the other.

    **``c_low`` moves the answer.** Measured on one ``.rhs``, option ``r``,
    only ``CMIN`` changed: lowering it below the mean field's is inert
    (1e-4 dB), raising it truncates the scattering integral by **30 dB**.
    ``REVINT``/``REVCOV`` bound their own buffer reads
    (``oassun26.f:744-748``, ``:958-962``), so it should default to the mean
    field's ``c_low`` rather than to a constant.
    """
    _reject_unknown_kwargs('write_oass_input', kwargs,
                           frozenset({'n_wavenumbers'}))
    _reject_unwritten_option_blocks('write_oass_input', options)

    chars = _oases_option_chars(options)
    # 'k' is a real GETOPT letter (:664), so it is in _OASS_OPTIONS and would
    # pass the unknown-letter test below; it is refused for its own reason.
    if 'k' in chars:
        raise ConfigurationError(
            "write_oass_input: option 'k' (PLPOWER) is passed into GETOPT "
            "but never initialised there (unoass21.f:577-590), leaving an "
            "uninitialised LOGICAL in the main program (:35).",
            remediation="Drop 'k'; the roughness power spectrum is not "
                        "readable through uacpy.",
        )
    unknown = sorted(chars - _OASS_OPTIONS)
    if unknown:
        raise ConfigurationError(
            f"write_oass_input: {unknown} are not OASS option letters — "
            f"GETOPT (unoass21.f:557-698) tests only "
            f"{''.join(sorted(_OASS_OPTIONS))}, and prints "
            f"'>>>> UNKNOWN OPTION <<<<' for anything else.",
            remediation=(
                "'N' and 'J' in particular are OAST/OASP habits that OASS "
                "does not share: the field parameter comes from the receiver "
                "type (oasnun22.f:97-118) and the complex contour is "
                "inherited from the .rhs (unoass21.f:213)."),
        )
    # oass.tex:141, :151, :160 each state that I, S and c "cannot be applied
    # together with the reverb options C,D,R,a,r", and :146 that R "cancels
    # options I, S, c". The mechanism is unoass21.f:342-344 gating CALSIN on
    # IF (.not.REVERB): with any reverb letter present the kernel is never
    # computed, so the requested plot simply never appears.
    kernels = sorted(chars & _OASS_KERNEL_OPTIONS)
    reverb = sorted(chars & _OASS_REVERB_OPTIONS)
    if kernels and reverb:
        raise ConfigurationError(
            f"write_oass_input: options {kernels} (Hankel/angular/depth "
            f"integrand plots) cannot be combined with {reverb} — CALSIN is "
            f"gated on IF (.not.REVERB) (unoass21.f:342-344), so with a "
            f"reverberation letter present the kernel is never computed and "
            f"the plot never appears.",
            remediation=("Run the kernel plots and the reverberation product "
                         "as separate runs (oass.tex:141, :146, :151, :160)."),
        )

    products = sorted(chars & _OASS_PRODUCT_OPTIONS)
    if len(products) > 1:
        raise ConfigurationError(
            f"write_oass_input: options {products} request more than one "
            f"scattered-field product. OASS computes one per run — asking "
            f"for the covariance ('a') alongside a reverberation curve "
            f"returns a SILENTLY ZERO covariance (oassun26.f:683-688 vs "
            f":897-902).",
            remediation="Run once per product.",
        )

    if int(interface) < 2:
        raise ConfigurationError(
            f"write_oass_input: interface must be >= 2 (INTFC indexes the "
            f"scattering interface, unoass21.f:151); got {interface}.")
    if float(spectral_exponent) <= 1.5:
        raise ConfigurationError(
            f"write_oass_input: spectral_exponent must exceed 1.5 or the "
            f"roughness power spectrum is not integrable "
            f"(oassp.tex:356-362; the exponent reaches "
            f"amod(m)=fac(3+…) at oaseun31.f:99); got {spectral_exponent}.")
    if float(correlation_length) <= 0.0:
        raise ConfigurationError(
            f"write_oass_input: correlation_length must be positive; "
            f"got {correlation_length}.")

    depth = float(env.depth)
    bottom = env.bottom.halfspace_at(range=0.0)
    _bp = _extract_bottom_props(bottom)
    rho, c_p, c_s = _bp['rho'], _bp['c_p'], _bp['c_s']
    alpha_p, alpha_s = _bp['alpha_p'], _bp['alpha_s']
    ssp_data = env.ssp.extend_to(depth).to_pairs()

    frequency = float(np.atleast_1d(source.frequencies)[0])

    cmin, cmax = oases_wavenumber_bounds(ssp_data)
    n_wavenumbers = int(kwargs.pop('n_wavenumbers', 2048))
    if c_low is not None:
        cmin = float(c_low)
    if c_high is not None:
        cmax = float(c_high)

    # Block VIII is read whenever any REVERB letter is present — including
    # 'a', whose GETOPT branch sets REVERB on the way in (unoass21.f:646-653)
    # even though oass.tex:254-257 says the range offset "does not apply to
    # option a". Omitting it for a covariance run makes the binary consume
    # the *next* record as RMIN RMAX NR and shifts the whole deck.
    needs_range_block = bool(chars & _OASS_REVERB_OPTIONS)
    if not needs_range_block and float(phase_speed) <= 0.0:
        raise ConfigurationError(
            f"write_oass_input: phase_speed must be positive on a deck with "
            f"no reverberation letter. unoass21.f:342-344 takes the "
            f"`IF (.not.REVERB)` branch and forms ETA=DSQ/CPHASE, so "
            f"CPH={float(phase_speed):g} divides by zero — gfortran does not "
            f"trap it, and CALSIN then runs on Inf.",
            remediation=("Set phase_speed to the trace speed the kernel is "
                         "wanted at (oass.tex:194-197), or request a "
                         "reverberation product instead."),
        )
    if needs_range_block:
        r = np.atleast_1d(np.asarray(receiver.ranges, dtype=float))
        r_min = float(r.min()) if range_min is None else float(range_min)
        r_max = float(r.max()) if range_max is None else float(range_max)
        nr = int(len(r)) if n_ranges is None else int(n_ranges)
        if nr < 2:
            raise ConfigurationError(
                f"write_oass_input: n_ranges must be >= 2 — OASS forms "
                f"RSTEP=(RMAX-RMIN)/(LF-1) with LF=NR (unoass21.f:269), so "
                f"NR=1 divides by zero; got {nr}.")

    # The environment block, resolved here because the roughness override has
    # to be keyed against it.
    n_sed_layers = _count_bottom_layers(env)
    ssp_subset = _check_ssp_layer_count(ssp_data, 2 + n_sed_layers)
    c_values = ssp_subset[:, 1]
    isovelocity = bool(np.allclose(c_values, c_values[0], rtol=1e-6))
    n_water_layers = 1 if isovelocity else len(ssp_subset)

    # INTFC counts deck layers from 1 (upper halfspace), but the suffix_fn
    # _emit_bottom_layers calls is indexed from its own first bottom record.
    # Bottom records begin at deck layer ``1 + n_water_layers + 1``, so that
    # record is suffix index 1 — hence the offset below. Keying the override
    # off INTFC directly lands on the wrong record, and the failure is
    # SILENT: the interface then carries a positive RG, which OASES reads as
    # an infinite correlation length (oassp.tex:344-348), making P(k) a delta
    # at k=0 — no back-scatter at all, from a run that exits 0.
    first_bottom_layer = 1 + n_water_layers + 1
    # Deck layer 2 is the first water record, whose RG is the SEA SURFACE:
    # ROUGH(M) is the roughness of the interface at the top of layer M
    # (oaseun31.f:381-383) and ROUGH(1)=ROUGH(2) at :377. OASS scatters from
    # it as readily as from the seabed — an ice keel or a wind-roughened
    # surface is the usual case — so it is a legal INTFC, not an error.
    surface_is_target = int(interface) == 2
    suffix_index = int(interface) - (first_bottom_layer - 1)
    if not surface_is_target and not 1 <= suffix_index <= n_sed_layers + 1:
        raise ConfigurationError(
            f"write_oass_input: interface={interface} names no interface this "
            f"deck carries. It has {n_water_layers} water layer(s), so the "
            f"scattering interfaces are deck layer 2 (the sea surface) and "
            f"{first_bottom_layer}..{first_bottom_layer + n_sed_layers} "
            f"(the seabed and any sediment interfaces).",
            remediation=("Interior water records carry no RG column INENVI "
                         "would re-read as CL and M; only the first water "
                         "record and the bottom records do."),
        )

    _roughness_tail = _make_roughness_tail(
        env,
        # The scattering interface must carry the nine-token form so INENVI
        # re-reads CL and M (oaseun31.f:72-75, :91-93). When the target is the
        # sea surface the tail goes on the water record instead, so the bottom
        # records keep their plain RMS.
        {} if surface_is_target else
        {suffix_index: (
            bottom_interface_roughness(env)[min(
                suffix_index - 1,
                len(bottom_interface_roughness(env)) - 1)]
            if rms_roughness is None else float(rms_roughness),
            float(correlation_length),
            float(spectral_exponent),
        )},
    )

    # A surface target gets the same nine-token tail the bottom override
    # would have produced, routed to the water record instead. Built through
    # _make_roughness_tail so the |RG| representability guard applies equally.
    surface_suffix = None
    if surface_is_target:
        surface_rg = (_surface_roughness(env) if rms_roughness is None
                      else float(rms_roughness))
        surface_suffix = _make_roughness_tail(
            env, {0: (surface_rg, float(correlation_length),
                      float(spectral_exponent))})(0)

    with open(filepath, 'w') as f:
        _write_oases_header(f, env, options, "OASS Simulation via UACPY")

        # Block III: FREQ. ICNTIN is initialised to 0 (unoass21.f:596) and
        # GETOPT never sets it, so the two-token READ at :105 is unreachable
        # and a trailing COFF would be inert. One token.
        f.write(f"{frequency:.9f}\n")

        # Block IV: NL (oaseun31.f:43) then NL layer records (:54), the same
        # environment block the OAST writer emits — mirrored rather than
        # re-derived so the two decks describe one environment identically.
        # extra_columns=0: the roughness tail here is the suffix_fn's job,
        # and it is the nine-token form at the scattering interface.
        if isovelocity:
            # One record for an isovelocity column; INENVI folds every
            # CC = |CS| layer back to isovelocity anyway (oaseun31.f:181-182),
            # so the general form would only burn NLA slots.
            f.write(f"{3 + n_sed_layers}\n")
            f.write(f"{_format_upper_halfspace(env)}\n")
            # The sea surface's roughness rides on the FIRST water record:
            # ROUGH(M) is the roughness of the interface at the TOP of layer M
            # (oaseun31.f:381-383), and ROUGH(1)=ROUGH(2) at :377. Taking it
            # from _roughness_tail(0) would read the hardcoded 0.0 anchor
            # instead, which the gradient branch below and OAST both avoid.
            # It is not inert: under option 'p' OASS keeps every interface's
            # ROUGH2 (oassun26.f:685-688 zeroes the array only if .not.rescat)
            # and oaskun21.f:54-66 builds the perturbed boundary operator from
            # it, so dropping it silently omits surface re-scattering.
            if surface_suffix is not None:
                f.write(f"0.00 {c_values[0]:.2f} 0 0.0 0 1.0"
                        f"{surface_suffix}\n")
            else:
                f.write(f"0.00 {c_values[0]:.2f} 0 0.0 0 1.0 "
                        f"{_surface_roughness(env):.4f}\n")
        else:
            f.write(f"{1 + len(ssp_subset) + n_sed_layers + 1}\n")
            f.write(f"{_format_upper_halfspace(env)}\n")
            _emit_water_layers(f, ssp_subset,
                               surface_roughness=_surface_roughness(env),
                               extra_columns=0,
                               surface_suffix=surface_suffix)
        _emit_bottom_layers(
            f, env, depth,
            c_p, c_s, alpha_p, alpha_s, rho,
            suffix_fn=_roughness_tail,
            iface_start=1,
        )

        # Block V: CPH INTFC (unoass21.f:151).
        f.write(f"{float(phase_speed):.2f} {int(interface)}\n")

        # Block VI: the 3-D receiver array, which also picks the field
        # parameter (§2.2a).
        _emit_receiver_array(
            f, receiver,
            receiver_types=receiver_types,
            receiver_gains=receiver_gains,
            writer='write_oass_input',
        )

        # Block VII: CMIN CMAX, then NW IC1 IC2 — three tokens, not the four
        # OASP/OASSP carry (unoass21.f:189).
        #
        # CMIN/CMAX are live: WK0 and WKMAX are formed from them at :190-191
        # and survive. The NW record is not — under REVERB the binary
        # recomputes NWVNO from the .rhs file's own DLWVNO and forces
        # ICUT1=1, ICUT2=NWVNO (:211-215), discarding whatever the deck said.
        # That asymmetry is exactly why c_low moves the answer by tens of dB
        # while this record cannot.
        #
        # It is written explicitly rather than as the siblings' ``-1`` auto
        # sentinel, because the override only happens under REVERB: on a
        # plots-only deck (say 'I' alone) a -1 would survive into NWVNO as a
        # negative wavenumber count.
        f.write(f"{cmin:.2f} {cmax:.2f}\n")
        f.write(f"{n_wavenumbers} 1 {n_wavenumbers}\n")

        # Block VIII: RMIN RMAX NR, in km (unoass21.f:263).
        if needs_range_block:
            f.write(f"{m_to_km(r_min):.6f} {m_to_km(r_max):.6f} {nr}\n")

        # Block IX: DR NDR (unoass21.f:265), read only under 'C'.
        if 'C' in chars:
            f.write("1.0 1\n")
