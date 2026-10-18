"""The (fc, Q, T) band of a RAM broadband run: its frequency grid, the
requested bins and the cost of the band's grid."""

import numpy as np
from typing import Tuple
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.models._notices import give_notice
from uacpy.core.environment import (
    Environment,
)
from uacpy.core.source import Source
from uacpy.models._defaults import (
    DEFAULT_BROADBAND_BANDWIDTH_FACTOR, DEFAULT_BROADBAND_N_FREQUENCIES,
)
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, NumericsWarning,
)
from uacpy.models.ram._settings import RamGrid
from uacpy.models.ram._domain import absorbing_layer_thickness


def resolve_broadband_grid(source: Source, *,
                            require_uniform: bool = True, notices=None, knobs,
                            log):
    """Resolve ``(fc, Q, T)`` for the native broadband sweep.

    mpiramS's internal loop is parameterised as

        band = fc · [1 - 1/Q, 1 + 1/Q]     (width = 2·fc/Q)
        Δf   = 1/T

    so ``1/Q`` is the HALF fractional width — ``Q = 2`` spans
    ``fc·(1 ± 1/2)``, whereas the base broadband default's
    ``DEFAULT_BROADBAND_BANDWIDTH_FACTOR`` is the FULL fractional width
    (``fc·(1 ± bw/2)``).

    A multi-element ``frequencies`` array defines the band itself, in any
    order (as on the :class:`~uacpy.core.source.Source`, which refuses a
    repeated bin): ``fc`` is a bin near the centre of the sorted array and
    ``Q`` / ``T`` are derived from its half-width and spacing
    (:func:`array_sweep`), so the marched sweep contains every requested
    bin and the result is trimmed back to exactly them, in the order given
    (:func:`requested_broadband_bins`; the derivation is logged, not
    warned). A pinned ``Q``
    and/or ``T`` beside the array is a second description of the same
    band: when the sweep it gives around the array's ``fc`` marches the
    array's own bins the pinned values are used, otherwise the two
    contradict each other and ``ConfigurationError`` names both. (A
    TIME_SERIES grid derived from the pulse is a default, not a request:
    :meth:`RAM._requested_frequencies` hands it over already rebuilt on
    the pins, :func:`sweep_on_the_pins`.)

    ``require_uniform=False`` (the Collins backends on ``BROADBAND``,
    which march the requested bins one subprocess each) lets a
    non-uniform array through; ``Q`` / ``T`` then come back ``None``,
    since no ``(fc, Q, T)`` sweep describes the grid — and a pinned ``Q``
    or ``T`` beside it is refused for the same reason.

    Single-element arrays use ``frequencies[0]`` as fc: one bin when
    neither ``Q`` nor ``T`` is pinned (``run()`` has already expanded a
    Source's lone fc to the default band, so this is an explicit
    ``frequencies=[fc]``), an ``(fc, Q, T)`` band around it otherwise.
    A knob left unpinned beside the other takes the value of the default
    band every engine shares (:func:`default_band_knobs`), so pinning
    one knob changes only what it names.
    """
    freqs = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
    if len(freqs) == 1:
        fc = float(freqs[0])
        if knobs.q_factor is None and knobs.record_duration is None:
            # ``run()`` expands a Source's lone fc to the default band,
            # so a single frequency with no band spec arrives here only
            # as an explicit ``run(frequencies=[fc])`` — the single-bin
            # request: collapse the sweep the same way COHERENT_TL does
            # (Q→∞, T=1) and let ``requested_broadband_bins`` trim the
            # result to exactly that bin.
            return fc, 1e6, 1.0
        Q_band, T_band = default_band_knobs(fc)
        Q = Q_band if knobs.q_factor is None else float(knobs.q_factor)
        T = T_band if knobs.record_duration is None else float(knobs.record_duration)
        frq = broadband_frequencies(fc, Q, T)
        if knobs.q_factor is None or knobs.record_duration is None:
            give_notice(notices,
                f"RAM BROADBAND: a single source frequency does not define "
                f"a band, and mpiramS / Collins march an internal "
                f"(fc, Q, T) sweep. fc={fc:g} Hz was given q_factor={Q:g} "
                f"({'pinned' if knobs.q_factor is not None else 'the default band'}"
                f") and record_duration={T:g} s "
                f"({'pinned' if knobs.record_duration is not None else 'the default band'}"
                f"), i.e. {np.size(frq)} bins over "
                f"{float(np.min(frq)):.4g}-{float(np.max(frq)):.4g} Hz "
                f"(df = 1/T = {1.0 / T:.4g} Hz). The default band is "
                f"{DEFAULT_BROADBAND_N_FREQUENCIES} bins over "
                f"fc·(1 ± {DEFAULT_BROADBAND_BANDWIDTH_FACTOR / 2:g}).",
                FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
        return fc, Q, T

    freqs = np.sort(freqs)
    freq_min, freq_max = float(freqs[0]), float(freqs[-1])
    pinned = [f"{name}={getattr(knobs, name):g}"
              for name in ('q_factor', 'record_duration')
              if getattr(knobs, name) is not None]
    pins = ' and '.join(pinned)
    sweep = array_sweep(freqs)
    if sweep is None:
        spacings = np.diff(freqs)
        if pinned:
            raise ConfigurationError(
                f"RAM BROADBAND: {pins} describe{'s' if len(pinned) == 1 else ''} "
                f"a uniform (fc, Q, T) sweep, and the {len(freqs)}-bin "
                f"frequency array ({freq_min:.4g}-{freq_max:.4g} Hz) is "
                f"non-uniform (Δf from {spacings.min():.4g} to "
                f"{spacings.max():.4g} Hz), which no sweep describes: the "
                f"two settings contradict each other.",
                remediation=("Leave q_factor and record_duration unset to march the "
                             "frequency array, or pass a single-frequency "
                             "Source without frequencies= to march the (fc, "
                             "q_factor, record_duration) sweep around it."),
            )
        if not require_uniform:
            return float(freqs[len(freqs) // 2]), None, None
        raise ConfigurationError(
            f"RAM BROADBAND: non-uniform frequency spacing "
            f"(min Δf={spacings.min():.4g}, max Δf={spacings.max():.4g} Hz). "
            f"The mpiramS sweep and the TIME_SERIES synthesis need a "
            f"uniform grid — pass uniformly spaced frequencies. The "
            f"Collins backends (backend='ramgeo' / 'rams' / 'ramsurf') "
            f"march any increasing list on BROADBAND."
        )
    fc, Q, T = sweep
    marched = broadband_frequencies(fc, Q, T)
    if pinned:
        Q_pin = Q if knobs.q_factor is None else float(knobs.q_factor)
        T_pin = (T if knobs.record_duration is None
                 else float(knobs.record_duration))
        on_pins = broadband_frequencies(fc, Q_pin, T_pin)
        if not (on_pins.size == marched.size
                and np.allclose(on_pins, marched, rtol=0.0, atol=1e-3 / T)):
            raise ConfigurationError(
                f"RAM BROADBAND: {pins} around the frequency array's centre "
                f"bin fc={fc:.4g} Hz march{'es' if len(pinned) == 1 else ''} "
                f"{on_pins.size} bins over {float(on_pins[0]):.4g}-"
                f"{float(on_pins[-1]):.4g} Hz at Δf = {1.0 / T_pin:.4g} Hz, "
                f"and the {len(freqs)}-bin frequency array asks for "
                f"{freq_min:.4g}-{freq_max:.4g} Hz at Δf = {1.0 / T:.4g} Hz: "
                f"the two settings contradict each other.",
                remediation=("Leave q_factor and record_duration unset to march the "
                             "frequency array, or pass a single-frequency "
                             "Source without frequencies= to march the (fc, "
                             "q_factor, record_duration) sweep around it."),
            )
        Q, T = Q_pin, T_pin
    log(
        f"RAM BROADBAND: the {len(freqs)}-bin frequency array "
        f"({freq_min:.2f}-{freq_max:.2f} Hz, Δf={1.0 / T:.4g} Hz) resolves to "
        f"fc={fc:.2f} Hz, Q={Q:.4f}, T={T:.4f} s"
        + (f" ({pins} pinned)" if pinned else "") +
        f"; the sweep spans {np.size(marched)} bins and the result carries "
        f"exactly the {len(freqs)} requested bins."
    )
    return fc, Q, T


def array_sweep(freqs):
    """``(fc, Q, T)`` of the ``(fc, Q, T)`` sweep that marches every bin of
    the increasing array ``freqs`` (uniform to ``rtol=1e-4``), or ``None``
    for a non-uniform array, which no sweep describes.

    ``(fc, Q, T)`` can only march bins at ``fc + m·Δf``, so ``fc`` is an
    actual array bin (the upper-middle one) rather than the band midpoint —
    an even-length array's midpoint sits half a bin off-grid, which would
    drop one bin and shift every other by ``Δf/2``. The half-band
    ``(n//2 + 1/2)·Δf`` puts ``peramx.f90:361``'s
    ``nf1 = int((bw-df)/df)+1`` mid-interval (robust to float noise): an
    odd count round-trips exactly, an even count marches one extra bin at
    ``freq_max + Δf`` — a superset of the request that
    :func:`requested_broadband_bins` trims back off the result."""
    freqs = np.asarray(freqs, dtype=float)
    spacings = np.diff(freqs)
    if not np.allclose(spacings, spacings[0], rtol=1e-4):
        return None
    df = float(spacings[0])
    fc = float(freqs[len(freqs) // 2])
    return fc, fc / ((len(freqs) // 2 + 0.5) * df), 1.0 / df


def sweep_on_the_pins(freqs, *, knobs) -> np.ndarray:
    """The bins of the sweep a derived (uniform) grid ``freqs`` becomes
    under a pinned ``q_factor`` / ``record_duration``: its
    :func:`array_sweep` with each pinned knob in place of the derived one.
    A TIME_SERIES grid derived from the pulse is uacpy's default, so a pin
    replaces what it names and the grid supplies the rest."""
    fc, Q, T = array_sweep(freqs)
    if knobs.q_factor is not None:
        Q = float(knobs.q_factor)
    if knobs.record_duration is not None:
        T = float(knobs.record_duration)
    return broadband_frequencies(fc, Q, T)


def warn_if_band_pins_unused_on_tl(*, notices=None, knobs) -> None:
    """One notice naming a pinned ``q_factor`` / ``record_duration`` on a
    COHERENT_TL run, which marches the source frequency alone: mpiramS's
    deck carries the one-bin sweep ``(fc, 1e6, 1)`` (``peramx.f90:362-370``
    marches ``fc`` alone for a band narrower than one bin) and a Collins deck
    carries one ``freq`` (``ramgeo1.5.f:105``), so the pins describe a
    sweep the run does not march."""
    pinned = [f"{name}={value:g}" for name, value in (
        ('q_factor', knobs.q_factor),
        ('record_duration', knobs.record_duration)) if value is not None]
    if not pinned:
        return
    give_notice(notices,
        f"RAM: {' and '.join(pinned)} set the (fc, Q, T) sweep of a "
        f"BROADBAND / TIME_SERIES run; a COHERENT_TL run marches the source "
        f"frequency alone, so {'they are' if len(pinned) > 1 else 'it is'} "
        f"not used here.",
        FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )


def default_band_knobs(fc: float) -> Tuple[float, float]:
    """``(Q, T)`` of the default band every engine expands a lone ``fc``
    to — ``DEFAULT_BROADBAND_N_FREQUENCIES`` bins over
    ``fc·(1 ± DEFAULT_BROADBAND_BANDWIDTH_FACTOR/2)``: the half-width
    ``fc/Q`` is ``fc·bandwidth_factor/2`` and the spacing ``1/T`` that of
    the default grid, ``fc·bandwidth_factor/(n_frequencies - 1)``."""
    Q = 2.0 / DEFAULT_BROADBAND_BANDWIDTH_FACTOR
    T = ((DEFAULT_BROADBAND_N_FREQUENCIES - 1)
         / (float(fc) * DEFAULT_BROADBAND_BANDWIDTH_FACTOR))
    return Q, T


def broadband_frequencies(fc: float, q_factor: float, record_duration: float) -> np.ndarray:
    """The symmetric frequency vector a ``(fc, Q, T)`` sweep marches.

    Reproduces ``peramx.f90:353-379``: half-bandwidth ``bw = fc/Q``,
    ``df = fs/Nsam = 1/T``, ``nf1 = int((bw - df)/df) + 1`` — ``0`` for
    a band narrower than one bin (``bw < df``, the UACPY patch at
    ``:362-370``), so that band marches ``fc`` alone — and
    ``frq(ii) = -(nf1 - (ii-1))·df + fc`` for ``ii = 1..2·nf1+1``. Every
    backend sweeps this same vector — mpiramS inside the Fortran loop, the
    Collins backends one subprocess per element.

    ``frq(1) = fc - nf1·df`` goes non-positive whenever
    ``fc <= nf1·df`` — a small ``q_factor`` (half-bandwidth at or above fc) —
    which no PE can march. The serial driver uacpy builds has no guard
    and writes NaN bins at zero and negative frequency; its MPI sibling
    stops on exactly this test and names ``q_factor`` (``peramx_mpi.f90:417-423``).
    """
    bw = float(fc) / float(q_factor)
    df = 1.0 / float(record_duration)
    nf1 = 0 if bw < df else int((bw - df) / df) + 1
    frq = (np.arange(2 * nf1 + 1, dtype=float) - nf1) * df + float(fc)
    if frq[0] <= 0.0:
        # The advice names the knob that moves this edge: the
        # half-bandwidth fc/Q (nf1·Δf never exceeds it).
        if float(fc) <= 0.0:
            advice = ("fc itself is not positive; no (Q, T) admits a "
                      "non-positive centre frequency.")
        else:
            advice = (f"Raise q_factor (= {q_factor:g}) so the half-bandwidth "
                      f"fc/q_factor = {bw:.4g} Hz falls below fc.")
        raise ConfigurationError(
            f"RAM broadband: the (fc, Q, T) sweep marches "
            f"{2 * nf1 + 1} bins at fc ± nf1·Δf with nf1 = {nf1} and "
            f"Δf = 1/record_duration = {df:.4g} Hz, so fc = {fc:g} Hz must exceed "
            f"nf1·Δf = {nf1 * df:.4g} Hz; its lower band edge sits at "
            f"{frq[0]:.4g} Hz and a PE cannot march zero or negative "
            f"frequencies. {advice}"
        )
    return frq


def requested_broadband_bins(source: Source, *, knobs):
    """The exact frequency bins a BROADBAND result must carry, or
    ``None`` when the ``(fc, Q, T)`` sweep itself is the requested grid.

    The symmetric ``(fc, Q, T)`` sweep of :func:`broadband_frequencies`
    can only march ``2·nf1 + 1`` bins, so it is a superset of a
    caller-supplied frequency array whenever the two differ (one extra
    bin past ``freq_max`` for an even count). The broadband runners trim
    their output onto this grid, in the caller's order, so ``H(f)``
    round-trips the request bin for bin — a pinned ``Q``
    / ``T`` beside an array describes the array's own sweep
    (:func:`resolve_broadband_grid` refuses one that does not). ``None``
    — keep the full sweep — for a single frequency together with a
    pinned ``Q`` or ``T`` (a band around fc).
    """
    freqs = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
    if len(freqs) == 1 and (knobs.q_factor is not None or knobs.record_duration is not None):
        return None
    return freqs


#: Depth points past which a band grid whose absorbing layer holds most of
#: them is worth a warning rather than a log line: a wide band meshes metres
#: of sponge at the finest ``dz`` (:func:`report_band_grid_cost`).
ABSORBER_COST_NOTICE_POINTS = 10000


def report_band_grid_cost(env: Environment, kind: str,
                           grid: RamGrid, freq_min: float,
                           freq_max: float, *, notices=None, knobs,
                          log) -> None:
    """Say what one band grid costs: its depth points and the share the
    absorbing layer takes. A band marches one grid, ``dz`` sized at its
    highest frequency and the absorber (``absorber_width_wavelengths``
    wavelengths) at its lowest, so a wide band — a pulse's -40 dB extent
    is ~40:1 — meshes metres of sponge at the finest ``dz``. Logged; a
    warning past :data:`ABSORBER_COST_NOTICE_POINTS` points when the
    absorber holds most of them, naming what narrows it."""
    absorber = absorbing_layer_thickness(env, freq_min, knobs=knobs)
    share = min(1.0, absorber / float(grid.zmax))
    text = (
        f"RAM:{kind}: the band {freq_min:.4g}-{freq_max:.4g} Hz marches "
        f"{grid.n_depth_points} depth points (dz={grid.dz:.4g} m sized at "
        f"{freq_max:.4g} Hz, zmax={grid.zmax:.5g} m), {100.0 * share:.0f} % "
        f"of them the {absorber:.5g} m absorbing layer sized at "
        f"{freq_min:.4g} Hz."
    )
    if grid.n_depth_points > ABSORBER_COST_NOTICE_POINTS and share > 0.5:
        give_notice(notices,
            text + " Pass a narrower frequencies= band, or lower "
            "absorber_width_wavelengths, to shrink it.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    else:
        log(text)
