"""The phase-speed window ``[c_low, c_high]`` an Acoustics-Toolbox deck
searches or integrates over: the default rule the engines derive an
unpinned bound from, the window a wavenumber integral resolves with the
origin of each bound, the check that a pinned pair is ordered, and what a
window leaves out — the steep paths its ``c_high`` cuts, a Scholte wave
below its lower edge — with the notices that state it.

The rules an engine derives on its own (Kraken's ``c_low`` of 0 and its
reflection-table ``c_high``, BOUNCE's ``min(SSP)``) stay in that engine; what
is here is what more than one engine reads.
"""

from typing import NamedTuple, Optional, Tuple

import numpy as np

from uacpy.core.boundary import BoundaryType
from uacpy.core.engine_defaults import default_c_low, scholte_waves
from uacpy.core.environment import Environment
from uacpy.core.exceptions import ConfigurationError, NumericsWarning
from uacpy.core.run_settings import Notice
from uacpy.models._knobs import positive_finite

# The unpinned ``c_low`` is the wavenumber-integration default every engine
# shares (:func:`uacpy.core.engine_defaults.default_c_low`, with its sources):
# Scooter / SPARC pin ``k_max = omega/c_low``, so a positive floor is required
# (c_low=0 blows up the integral).
#
# ``C_HIGH_FACTOR`` pads the upper bound of the mode search over a half-space:
# :func:`resolve_phase_speed_bounds` takes ``c_high = 1.05 · max(c_max, bottom
# cp)`` so the fastest speed in the problem sits strictly inside the interval
# rather than on its edge. The 1.05 is uacpy guidance, measured on Kraken
# (models/kraken/_window.py); doc/kraken.htm leaves CHIGH to "a matter of
# experience". The wavenumber integrals (Scooter, SPARC) do not use it: their
# unpinned c_high is unbounded, k_min = 0, as Jensen et al. (*Computational
# Ocean Acoustics*, ch. 4 App. 1) recommend (:func:`resolve_window`).
# Kraken only searches [c_low, c_high] for roots, so
# there the pad means a default run returns a few modes with a phase speed
# above the bottom speed: on kraken.exe roots of a rigid seabed
# (``Kraken/BCImpedanceMod.f90:83-88``), which a run that chose kraken.exe or
# a c_high above the seabed speed says (``models/kraken/_checks.py``,
# ``kraken_exe_above_seabed_notice``; a default run says nothing), on krakenc
# leaky modes. ``models/kraken/_modes.py`` logs the count, and a field run whose
# every mode is one of them warns (``models/kraken/_model.py``, ``_warn_leaky``).
# (KRAKEN's modal-solver c_low default is the literal 0.0 written at its use
# site in ``models/kraken/_window.py`` — 0 makes KRAKEN compute the bound
# itself, per
# kraken.htm, Phase Speed Limits.)
C_HIGH_FACTOR = 1.05

# "No upper phase-speed limit" for the AT family. A vacuum / rigid boundary
# traps every mode, so the mode search must not be capped on a half-space
# speed that does not exist. Acoustics-Toolbox/doc/bounce.htm prescribes the
# value: "For a full 90 degree calculation set CMin to the lowest speed in the
# problem (say 1400.0) CMax to 1.0E9." bounce.f90:47 takes kMin = 0 above 1e6.
# Kraken's ``leaky_modes`` uses 10 x the fastest speed instead
# (``models/kraken/_window.py``, ``_LEAKY_C_HIGH_FACTOR``).
DEFAULT_C_MAX_UNBOUNDED = 1.0e9


def resolve_phase_speed_bounds(
    env: Environment,
    c_low: Optional[float] = None,
    c_high: Optional[float] = None,
) -> Tuple[float, float]:
    """Resolve effective ``(c_low, c_high)`` for an AT-family run.

    Precedence:
      1. Explicit caller values win.
      2. Otherwise: ``c_low`` from
         :func:`~uacpy.core.engine_defaults.default_c_low` and
         ``c_high = max(c_max, env.bottom.halfspace_at(range=0).sound_speed) · C_HIGH_FACTOR``.

    A **non-geoacoustic** bottom (vacuum, rigid, or a reflection table —
    'file'/'precalc') carries no physical sound speed — modes above the
    half-space speed are leaky only when there *is* a half-space to leak into,
    and a parameter-free ``BoundaryProperties`` still carries the placeholder
    ``sound_speed`` its constructor defaults to. Capping on that placeholder
    silently truncates the mode spectrum (a 100 m rigid-bottom guide at 50 Hz
    keeps 3 of its 7 modes, a 10.6 dB error; a 'file' bottom's 1600 m/s
    placeholder capped cHigh at 1680 m/s), so those boundaries resolve to
    :data:`DEFAULT_C_MAX_UNBOUNDED` instead — the AT idiom for "no upper
    limit".

    The derived ``c_low`` is the documented rule: 0.90 × the slowest wave
    of the problem — every medium's speed, shear speeds and half-spaces
    included, and every Scholte wave solved from Brekhovskikh
    & Lysanov eq. 3.4.29 (``scooter.f90:258``; Jensen et al.,
    *Computational Ocean Acoustics*, ch. 4 App. 1 step 1, "the lowest wave
    speed in the problem"; :func:`~uacpy.core.engine_defaults.default_c_low`).
    Over an elastic seabed the Scholte wave is then in the window, at
    the cost of wavenumbers: ``Nk`` grows as ``1/c_low``
    (docs/models/scooter.md, "Layered and elastic seabeds", measures the
    run on elastic sand against a water-only window).
    """
    if c_low is not None and c_high is not None:
        return float(c_low), float(c_high)
    ssp_pairs = env.ssp.to_pairs()
    halfspace = env.bottom.halfspace_at(range=0.0)
    if not BoundaryType.from_string(halfspace.acoustic_type).is_geoacoustic:
        c_high_auto = DEFAULT_C_MAX_UNBOUNDED
    else:
        c_max = max(float(ssp_pairs[:, 1].max()),
                    float(halfspace.sound_speed))
        c_high_auto = c_max * C_HIGH_FACTOR
    return (
        float(c_low) if c_low is not None else default_c_low(env)[0],
        float(c_high) if c_high is not None else c_high_auto,
    )


class PhaseSpeedWindow(NamedTuple):
    """A resolved phase-speed window (m/s) and where each bound came
    from, as the engine settings record it."""
    c_low: float
    c_high: float
    c_low_origin: str
    c_high_origin: str


def check_pinned_window(model_name: str, *, c_low, c_high) -> None:
    """Refuse a pinned bound that is not a positive finite speed (the
    integral's ``k_max = omega/c_low`` and ``k_min = omega/c_high`` need
    one), and two pinned bounds with ``c_low >= c_high``. A single pinned
    bound is held to the other once a run derives it
    (:func:`refuse_inverted_window`)."""
    positive_finite(f"{model_name}: c_low (m/s)", c_low, optional=True)
    positive_finite(f"{model_name}: c_high (m/s)", c_high, optional=True)
    if (c_low is not None and c_high is not None
            and c_low >= c_high):
        raise ConfigurationError(
            f"{model_name} spectral phase-velocity band requires "
            f"c_low < c_high; got c_low={c_low} m/s, "
            f"c_high={c_high} m/s."
        )


def refuse_inverted_window(model_name: str, cl: float, ch: float, *,
                           c_low, c_high) -> None:
    """Refuse a resolved window ``[cl, ch]`` with ``cl >= ch``, naming the
    bound the caller pinned (``c_low`` / ``c_high``, ``None`` when derived),
    since that is the one to move."""
    if cl < ch:
        return
    if c_low is not None and c_high is None:
        pinned = (f"the pinned c_low={c_low} m/s is at or above "
                  f"the env-derived c_high = {ch:.1f} m/s")
    elif c_high is not None and c_low is None:
        pinned = (f"the pinned c_high={c_high} m/s is at or "
                  f"below the env-derived c_low = {cl:.1f} m/s")
    else:
        pinned = (f"the env-derived band collapsed to c_low = "
                  f"{cl:.1f} m/s, c_high = {ch:.1f} m/s")
    raise ConfigurationError(
        f"{model_name} spectral phase-velocity band requires "
        f"c_low < c_high: {pinned}.",
        remediation="Widen the pinned bound, or leave both unset to "
                    "derive the band from the SSP and bottom.",
    )


def resolve_window(env, *, c_low, c_high, c_high_unbounded: float,
                   model_name: str) -> PhaseSpeedWindow:
    """The window of a wavenumber integral (Scooter, SPARC): the pinned
    bounds; an unpinned ``c_low`` from :func:`resolve_phase_speed_bounds`
    on the projected ``env`` (0.90 × the slowest wave of the problem,
    Scholte waves included,
    :func:`~uacpy.core.engine_defaults.default_c_low`); an unpinned ``c_high``
    of ``c_high_unbounded`` over every seabed — the engine's own "no upper
    limit", which puts ``k_min = omega/c_high`` at 0 to every grazing angle
    short of 90° — as Jensen et al. (*Computational Ocean Acoustics*, ch. 4
    App. 1 step 1) recommend: "it is recommended to initially include these
    components by choosing k min = 0"; with the origin of each.

    The constructor can only compare two pinned bounds
    (:func:`check_pinned_window`). A single pinned bound is only comparable
    once the other has been derived from this env, and an inverted pair
    reaches ReadEnvironmentMod.f90:135, which stops the binary after the
    deck has been written and the process spawned. It is refused here
    instead, naming the bound the user pinned, since that is the one to
    move.
    """
    cl, ch = resolve_phase_speed_bounds(
        env, c_low, c_high_unbounded if c_high is None else c_high)
    refuse_inverted_window(model_name, cl, ch, c_low=c_low, c_high=c_high)
    if c_high is not None:
        high_origin = f"{model_name}(c_high=…)"
    else:
        high_origin = (f"unbounded ({c_high_unbounded:g} m/s): k_min = 0, "
                       f"every phase speed (Jensen et al., Computational "
                       f"Ocean Acoustics, ch. 4 App. 1)")
    return PhaseSpeedWindow(
        c_low=cl, c_high=ch,
        c_low_origin=(f"{model_name}(c_low=…)" if c_low is not None
                      else default_c_low(env)[1]),
        c_high_origin=high_origin)


def steep_path_cut(c_water: float, c_high: float, source_depths,
                   receiver_depths, receiver_ranges
                   ) -> Optional[Tuple[float, float]]:
    """``(cut, steepest)`` in degrees when a receiver's direct or
    surface-reflected path is steeper than ``c_high`` keeps, else ``None``.

    A path at grazing angle ``θ`` in water of speed ``c`` has horizontal
    phase speed ``c / cos θ``; a spectrum cut at ``c_high`` keeps it only
    while that is below ``c_high``, i.e. ``θ <= arccos(c / c_high)`` (the
    fastest water speed makes this the tightest cut). A steeper path is
    simply absent from the field, with no sign of it in the result. One
    rule for every engine, so the geometry cannot drift between them.

    ``cut`` is ``arccos(c_water / c_high)``; ``steepest`` is the steepest
    direct or surface-reflected path from any source depth to any receiver
    at the closest positive range. ``None`` also when ``c_high`` is not
    above ``c_water`` or no receiver range is positive.
    """
    if not (c_high > c_water):
        return None
    cut = float(np.degrees(np.arccos(c_water / c_high)))
    zs = np.atleast_1d(np.asarray(source_depths, dtype=float))
    zr = np.atleast_1d(np.asarray(receiver_depths, dtype=float))
    rr = np.atleast_1d(np.asarray(receiver_ranges, dtype=float))
    rr = rr[rr > 0.0]
    if not rr.size:
        return None
    steepest = 0.0
    for s in zs:
        direct = np.abs(zr[:, None] - s)
        surface = zr[:, None] + s
        depth_span = np.maximum(direct, surface)
        steepest = max(steepest, float(np.degrees(np.arctan(
            depth_span.max() / rr.min()))))
    if not steepest > cut:
        return None
    return cut, steepest


def steep_path_clear_range(source_depths, receiver_depths,
                           cut: float) -> int:
    """The range (m, rounded up to a whole metre) from which no direct or
    surface-reflected path between ``source_depths`` and
    ``receiver_depths`` is steeper than ``cut`` degrees: the deepest
    surface-reflected span over ``tan(cut)``."""
    zs = np.atleast_1d(np.asarray(source_depths, dtype=float))
    zr = np.atleast_1d(np.asarray(receiver_depths, dtype=float))
    span = float(np.max(zr[:, None] + zs[None, :]))
    return int(np.ceil(span / np.tan(np.radians(cut))))


def steep_path_notice(env, source, receiver, c_high: float, *,
                      model_name: str, kept: str, summed_in: str,
                      evidence: str, remediation: str,
                      pinned: bool = False) -> Optional[Notice]:
    """A :class:`~uacpy.core.run_settings.Notice` when a receiver's direct
    or surface-reflected path is steeper than ``c_high`` keeps (:func:`steep_path_cut`, at the
    fastest water speed), else ``None``.

    The engine names what its window keeps (``kept``: Scooter's paths,
    Kraken's modes carrying them), what the energy is missing from
    (``summed_in``), where the claim is shown (``evidence``: a measurement,
    or the docs that hold one) and its remediation, and whether the user
    pinned ``c_high`` (``pinned``: Scooter's notice concerns a pinned bound
    only, Kraken's its derived one); the message adds the range from
    which no direct or surface-reflected path is steeper than the cut
    (RAM's notice states the same for its band)."""
    c_water = float(np.nanmax(np.asarray(env.ssp.sound_speed, dtype=float)))
    reach = steep_path_cut(c_water, c_high, source.depths,
                           receiver.depths, receiver.ranges)
    if reach is None:
        return None
    cut, steepest = reach
    clear = steep_path_clear_range(source.depths, receiver.depths, cut)
    return Notice(
        f"c_high {c_high:.1f} m/s drops paths steeper than {cut:.1f}° "
        f"that the receiver geometry needs (up to {steepest:.1f}°)",
        f"{model_name}: the {'pinned' if pinned else 'auto-derived'} "
        f"c_high = {c_high:.1f} m/s keeps "
        f"{kept} up to {cut:.1f}° grazing, but the closest receiver sees a "
        f"direct or surface-reflected path at {steepest:.1f}°; that energy "
        f"is not in the {summed_in}, so the near-range field is wrong "
        f"({evidence}); from {clear} m on no direct or surface-reflected "
        f"path is steeper than {cut:.1f}°. {remediation}", NumericsWarning,
    )


#: OAST's automatic wavenumber sampling, SUBROUTINE AUTSAM
#: (``unoast31.f:1181-1227``): ``NR`` points over the deck's range axis set
#: the least wavenumber span, ``NKMIN`` the least count, and the period
#: factor ``RFAC`` is 6 under the full Hankel transform (option ``'f'``,
#: ``INTTYP = 2``, ``:1139``) and 3 otherwise (``:1192``, ``:1197-1201``).
#: OASN's discrete-source sampling, SUBROUTINE AUTSMN
#: (``oasnun22.f:1677-1718``), is the same scheme with ``NR = 10``,
#: ``RFAC = 3`` and a fixed ``CREFA = 1500`` m/s (``:1687-1694``).
_AUTSAM_NR = 300
_AUTSAM_NKMIN = 256
_AUTSMN_NR = 10
_AUTSMN_RFAC = 3.0
_AUTSMN_CREFA = 1500.0


def _autsam_edge(c_low: float, c_high: float, *, frequency: float,
                 r_min: float, r_max: float, reference_speed: float,
                 rfac: float, n_ranges: int) -> Tuple[float, int]:
    """``(CMIN, NW)``: the slowest phase speed (m/s) and the wavenumber
    count of OAST's AUTSAM / OASN's AUTSMN, which compute it alike
    (``unoast31.f:1202-1224``, ``oasnun22.f:1693-1715``), from the deck's
    range span ``r_min``..``r_max`` (m):

    * ``DK = min(2π/(RFAC·r_max), 2πf/(CREFA·NKMIN))``;
    * ``WM = 2π/DR`` with ``DR = (r_max − r_min)/(NR − 1)``;
    * ``WN1 = 2πf/c_high``, ``WN2 = 2πf/c_low``, ``WNMAX = max(WM,
      1.1·(WN2 − WN1) + WN1)``, ``WNMIN = max(min(WNMAX − WM, WN1 −
      0.1·(WN2 − WN1)), DK/2)``;
    * ``NW`` the smallest power of two from NKMIN up holding
      ``(WNMAX − WNMIN)/DK + 1`` wavenumbers, ``WNMAX = WNMIN + (NW −
      1)·DK`` and ``CMIN = 2πf/WNMAX``.

    A span of 0 makes ``WM = 2π/DR`` infinite: no edge, ``(0, 0)``."""
    if not r_max > r_min:
        return 0.0, 0
    omega = 2.0 * np.pi * float(frequency)
    dk = min(2.0 * np.pi / (rfac * r_max),
             omega / (float(reference_speed) * _AUTSAM_NKMIN))
    wm = 2.0 * np.pi * (n_ranges - 1) / (r_max - r_min)
    wn1 = omega / float(c_high)
    wn2 = omega / float(c_low)
    wnmax = max(wm, 1.1 * (wn2 - wn1) + wn1)
    wnmin = max(min(wnmax - wm, wn1 - 0.1 * (wn2 - wn1)), 0.5 * dk)
    needed = int((wnmax - wnmin) / dk + 1)
    nw = _AUTSAM_NKMIN
    while nw < needed:
        nw *= 2
    return omega / (wnmin + (nw - 1) * dk), nw


def oases_integration_edge(c_low: float, c_high: float, *,
                           automatic: bool, frequency: float,
                           range_min: float, range_max: float,
                           source_speed: float,
                           full_hankel: bool) -> Tuple[float, str]:
    """``(edge, text)``: the slowest phase speed (m/s) OAST's wavenumber
    integral reaches on its deck, and the sentence stating its source.

    A pinned count integrates from ``c_low`` as given
    (``unoast31.f:432-433``). Under automatic sampling (no
    ``n_wavenumbers``) the edge is AUTSAM's ``CMIN`` (:func:`_autsam_edge`,
    ``unoast31.f:1224``) on the deck's values: ``r_max = max(|XLEFT|,
    |XRIGHT|)`` and ``r_min = r_max − |XRIGHT − XLEFT|`` (``:427-428``),
    here ``range_min`` / ``range_max`` in metres; CREFA the compressional
    speed of the source's layer (``source_speed``, ``:1203``); RFAC 6
    under option ``'f'`` (``full_hankel``), else 3.

    The kernel past ``2πf/c_low`` is integrated under a cosine taper that
    falls to zero at WNMAX (CALIN3, ``oaseun31.f:2157-2159``): a wave
    between ``c_low`` and the edge is in the field, attenuated by the
    taper, not left out; a wave slower than the edge has no sample.
    """
    if not automatic:
        return float(c_low), (f"c_low = {c_low:.1f} m/s with a pinned "
                              f"n_wavenumbers (unoast31.f:432-433)")
    r_max = max(abs(float(range_min)), abs(float(range_max)))
    r_min = r_max - abs(float(range_max) - float(range_min))
    edge, nw = _autsam_edge(c_low, c_high, frequency=frequency,
                            r_min=r_min, r_max=r_max,
                            reference_speed=source_speed,
                            rfac=6.0 if full_hankel else 3.0,
                            n_ranges=_AUTSAM_NR)
    if nw == 0:
        return 0.0, "0 m/s: the deck's range axis has no span"
    return edge, (f"{edge:.1f} m/s at {float(frequency):g} Hz: OAST's "
                  f"automatic sampling (AUTSAM, unoast31.f:1202-1224) "
                  f"integrates {nw} wavenumbers up to it on this deck, "
                  f"c_low = {c_low:.1f} m/s over a {r_min:g}-{r_max:g} m "
                  f"range axis")


def oasn_discrete_integration_edge(c_low: float, c_high: float, *,
                                   automatic: bool, frequency: float,
                                   first_range: float) -> Tuple[float, str]:
    """``(edge, text)``: the slowest phase speed (m/s) OASN's
    discrete-source wavenumber integral reaches. A pinned count integrates
    from ``c_low`` as given (``oasnun22.f:436-437``). Under automatic
    sampling AUTSMN runs once, at the band centre ``frequency`` (``FREQ =
    0.5·(FREQ1 + FREQ2)``, ``unoasn22.f:147``), on the FIRST record's range
    ``first_range`` (m) as ``RMAX`` and a tenth of it as ``RMIN``
    (``oasnun22.f:424-433``), with ``NR = 10``, ``RFAC = 3`` and ``CREFA =
    1500`` m/s (:func:`_autsam_edge`). Its ``CMIND`` bounds the integral in
    slowness, ``SLOW2D = 2π/CMIND`` (``oasnun22.f:447``), so the edge is
    that phase speed at every frequency."""
    if not automatic:
        return float(c_low), (f"c_low = {c_low:.1f} m/s with a pinned "
                              f"n_wavenumbers (oasnun22.f:436-437)")
    edge, nw = _autsam_edge(c_low, c_high, frequency=frequency,
                            r_min=0.1 * float(first_range),
                            r_max=float(first_range),
                            reference_speed=_AUTSMN_CREFA,
                            rfac=_AUTSMN_RFAC, n_ranges=_AUTSMN_NR)
    if nw == 0:
        return 0.0, "0 m/s: the first discrete source sits at range 0"
    return edge, (f"{edge:.1f} m/s: OASN's automatic discrete-source "
                  f"sampling (AUTSMN, oasnun22.f:422-447, :1677-1718) "
                  f"integrates {nw} wavenumbers to it at the "
                  f"{float(frequency):g} Hz band centre, c_low = "
                  f"{c_low:.1f} m/s, the first source "
                  f"{float(first_range):g} m out")


def scholte_wave_notice(env, source, receiver, edge: float,
                        frequency: float, *, model_name: str,
                        edge_text: str, remediation: str,
                        wavenumber_edge: Optional[float] = None,
                        ) -> Optional[Notice]:
    """A notice when the slowest phase speed the engine reaches, ``edge``
    (m/s), lies above a Scholte wave of ``env``
    (:func:`~uacpy.core.engine_defaults.scholte_waves`, Brekhovskikh &
    Lysanov §3.4.2, eq. 3.4.29) that the source or a receiver lies close
    to, else ``None``.

    ``edge`` is the engine's own — the lower edge of its wavenumber
    integral or the floor of its mode search — and the caller states it
    with its source in ``edge_text``, and the engine's remedy in
    ``remediation``. A derived
    :func:`~uacpy.core.engine_defaults.default_c_low` sits under every
    solved wave, so the notice concerns a pinned ``c_low`` and Kraken's
    window over shear. The wave's "amplitude decreases exponentially in
    both media with distance from the interface" (Brekhovskikh & Lysanov
    §3.4.2), and "interface waves can only be excited in a stratified
    waveguide by sources close in terms of wavelength to the interface"
    (Jensen et al., *Computational Ocean Acoustics*, §9.2.6.2): a wave
    counts when the source or a receiver lies within one of its
    wavelengths, ``v / f`` at the lowest frequency ``frequency``, of its
    interface. The one-wavelength reach is uacpy guidance, set by the
    measurement it rests on: far from the bed leaving the wave out costs
    nothing measurable (docs/models/scooter.md, "Layered and elastic
    seabeds": 0.004 dB median for a source 75 m above an elastic sand
    seabed); 1 m above it, up to 2.4 dB.

    OAST and OASP integrate the kernel past ``2πf/c_low`` under a cosine
    taper down to their edge (CALIN3, ``oaseun31.f:2157-2159``): a wave
    inside that taper zone is attenuated, not left out, and is not told.
    An engine that fixes its window in wavenumber for the whole band (OASP,
    ``unoasp22.f:394``, :605-617) passes its top wavenumber
    ``wavenumber_edge`` (rad/m), ``edge`` being its phase speed at
    ``frequency``: the notice then says below which frequency, ``v ·
    wavenumber_edge / 2π``, the wave is inside the window.
    """
    depths = np.concatenate([
        np.atleast_1d(np.asarray(source.depths, dtype=float)),
        np.atleast_1d(np.asarray(receiver.depths, dtype=float))])
    for wave in sorted(scholte_waves(env), key=lambda w: w.speed):
        if not wave.speed < edge:
            return None
        wavelength = wave.speed / float(frequency)
        nearest = float(np.abs(depths - wave.depth).min())
        if nearest > wavelength:
            continue
        inside = ('' if wavenumber_edge is None else
                  f" at {float(frequency):g} Hz; the window is fixed in "
                  f"wavenumber at {float(wavenumber_edge):.4g} rad/m, so "
                  f"the wave is inside it below "
                  f"{wave.speed * float(wavenumber_edge) / (2 * np.pi):.1f}"
                  f" Hz")
        return Notice(
            f"the window's edge {edge:.1f} m/s leaves out the Scholte "
            f"wave ({wave.speed:.0f} m/s) within {nearest:.3g} m of its "
            f"interface",
            f"{model_name}: the slowest phase speed the run reaches is "
            f"{edge_text}, above the Scholte wave "
            f"{wave.where}, {wave.speed:.1f} m/s (Brekhovskikh & Lysanov "
            f"§3.4.2, eq. 3.4.29, for the two media as half-spaces), so "
            f"the field leaves it out{inside}; the geometry reaches within "
            f"{nearest:.3g} m of that interface, inside the wave's "
            f"{wavelength:.3g} m wavelength at {float(frequency):g} Hz, "
            f"where it carries energy (up to 2.4 dB measured 1 m above an "
            f"elastic sand seabed; 33 dB at 1 km on the limestone preset "
            f"in 30 m of water, docs/models/scooter.md). {remediation}",
            NumericsWarning)
    return None
