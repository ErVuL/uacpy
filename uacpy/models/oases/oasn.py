"""OASN: the OASES noise, covariance and replica program."""

import dataclasses
from dataclasses import dataclass
from types import SimpleNamespace
from types import MappingProxyType
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from uacpy.core.constants import REFERENCE_PRESSURE_WATER
from uacpy.models._knobs import whole_count
from uacpy.models._spec import ModelSpec
from uacpy.models._window import (
    check_pinned_window, oasn_discrete_integration_edge,
    refuse_inverted_window, scholte_wave_notice,
)
from uacpy.core.run_settings import EngineSettings, OutputSpec, RunMode
from uacpy.core.results import PhaseReference
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, ValidityWarning,
)
from uacpy.io.oases_writer import (
    OASN_LEVEL_DEAD_BAND_DB, OasnNoise, OasnReplicaGrid,
    oases_wavenumber_bounds, oasn_deep_source_depth, oasn_replica_ends,
    refuse_unusable_integration_window, write_oasn_input,
    _check_oasn_discrete_sources,
    _check_oasn_noise_level, _check_oasn_replica_counts, oases_frequency_sweep,
    water_ac_anchor_frequency,
)
from uacpy.io.oases_reader import read_oasn_covariance, read_oasn_replicas
from uacpy.models.base import DEFAULT_RUN_TIMEOUT, _host_options
from uacpy.models.oases._common import (
    _OASES_TRAITS, _oases_binary, _option_letters,
    _check_n_wavenumbers_knob, _warn_if_water_ac_extrapolates, _offset_ignored_notice,
    resolve_integration_window,
)
from uacpy.models.oases._base import OASES
from uacpy.models._notices import message_notice
from uacpy.core.engine_defaults import OASN_INTEGRATION_OFFSET, default_c_low
from uacpy.models._extract import attach_output_paths
from uacpy.models._launch import require_output


#: File root of every deck and output one OASN run writes.
_OASN_BASE_NAME = 'oasn_run'



@dataclass(frozen=True, eq=False)
class OASNSettings(EngineSettings):
    """The settings one :class:`OASN` run resolved, before launching:
    ``OASN().run_settings(env, source, receiver, mode).engine``, and
    ``result.run_settings.engine`` on the covariance or replicas it
    produced.

    Attributes
    ----------
    options : str
        The option line the deck carries: the constructor's line as
        written (``'J'`` when none), with ``'N'`` appended for a covariance
        and ``'R'`` for replicas when the line lacks that character.
    frequency_sweep : tuple of (float, float, int)
        Block III's ``FREQ1 FREQ2 NFREQ``.
    integration_offset : float
        The contour offset (dB/wavelength) on the frequency line.
    n_wavenumbers : int or None
        The wavenumber count; ``None`` is OASES' automatic sampling for the
        replica and discrete-source blocks and each noise block's default
        counts (see :class:`OASN`).
    integration_window : tuple of (int, int) or None
        ``IC1 IC2`` of the discrete-source and replica blocks: the given
        window, the whole ``(1, NW)`` under a pinned ``n_wavenumbers``,
        ``None`` under automatic sampling (where OASES sets them).
    surface_noise_level_dB, deep_noise_level_dB : float
        Block VI source levels (dB re 1 µPa²/Hz).
    white_noise_level_dB : float or None
        Block VI white-noise level (dB re 1 µPa²/Hz); ``None`` writes the
        writer's -200 dB.
    deep_source_depth : float
        Depth (m) of the deep noise sheet, which the deck carries when the
        deep level is on: the constructor's, else half the water depth
        (:func:`~uacpy.io.oases_writer.oasn_deep_source_depth`).
    discrete_sources : tuple
        One ``((key, value), ...)`` tuple per discrete noise source.
    receiver_gains_dB : ndarray or None
        The array records' ``GAIN`` column, per element (dB); ``None``
        writes 0 dB.
    c_low, c_high : float
        Phase-speed bounds (m/s) of every integration (noise, discrete
        sources, replicas).
    c_low_origin, c_high_origin : str
        Where ``c_low`` and ``c_high`` came from: the constructor, or
        :func:`~uacpy.io.oases_writer.oases_wavenumber_bounds` — the
        sentence :func:`~uacpy.core.engine_defaults.default_c_low` states for
        ``c_low``, OASES' "no upper limit" 1e8 for ``c_high``.
    replica_xmin, replica_xmax, replica_ymin, replica_ymax, replica_zmin, \
replica_zmax : float or None
        Replica grid bounds (m): on a REPLICA run the bounds the deck
        carries (an unset one resolved by
        :func:`~uacpy.io.oases_writer.oasn_replica_ends`), on a COVARIANCE
        run, which reads no replica grid, as given.
    replica_nx, replica_ny, replica_nz : int or None
        Replica grid counts: on a REPLICA run the counts the deck carries
        (an unset count resolved to :class:`~uacpy.io.oases_writer.
        OasnReplicaGrid`'s 50 / 1 / 20), on a COVARIANCE run, which reads no
        replica grid, as given.
    """

    options: str
    frequency_sweep: Tuple[float, float, int]
    integration_offset: float
    n_wavenumbers: Optional[int]
    integration_window: Optional[Tuple[int, int]]
    surface_noise_level_dB: float
    white_noise_level_dB: Optional[float]
    deep_noise_level_dB: float
    deep_source_depth: float
    discrete_sources: Tuple[Tuple[Tuple[str, object], ...], ...]
    receiver_gains_dB: Optional[np.ndarray]
    c_low: float
    c_low_origin: str
    c_high: float
    c_high_origin: str
    replica_xmin: Optional[float]
    replica_xmax: Optional[float]
    replica_nx: Optional[int]
    replica_ymin: Optional[float]
    replica_ymax: Optional[float]
    replica_ny: Optional[int]
    replica_zmin: Optional[float]
    replica_zmax: Optional[float]
    replica_nz: Optional[int]

    _ARRAY_FIELDS = ('receiver_gains_dB',)

    def __post_init__(self):
        # Lists (the to_dict form) are stored as the tuples a frozen record
        # holds.
        freq_min, freq_max, n = self.frequency_sweep
        object.__setattr__(self, 'frequency_sweep',
                           (float(freq_min), float(freq_max), int(n)))
        object.__setattr__(self, 'discrete_sources', tuple(
            tuple(tuple(pair) for pair in ds)
            for ds in self.discrete_sources))
        if self.integration_window is not None:
            object.__setattr__(self, 'integration_window',
                               tuple(int(v) for v in self.integration_window))
        super().__post_init__()


def _has_noise_field(*, surface_noise_level_dB, white_noise_level_dB,
                     deep_noise_level_dB, discrete_sources) -> bool:
    """True when these noise knobs configure any Block VI source.

    White noise counts whenever it was set at all: an explicit 0.0 is
    a literal 0 dB diagonal term, not "off", so the test is against
    ``None`` rather than truthiness.
    """
    return bool(surface_noise_level_dB
                or white_noise_level_dB is not None
                or deep_noise_level_dB or discrete_sources)


def _reject_doppler_letter(options) -> None:
    """Refuse a raw option line carrying ``'d'`` or ``'D'``.

    OASN's GETOPT sets ``doppler=.true.`` on either (``unoasn22.f:702-
    704``) while ``vrec`` keeps its 15.0 m/s DATA default (``:762``): no
    deck field sets it. CALIN3, which OASN integrates through
    (``oasnun22.f:604``), then shifts every wavenumber sample's frequency
    by ``real(wvno)*vrec/(2*pi)`` (``oaseun31.f:2030-2041``), so the run
    is a moving-array problem at a speed nobody chose — measured, the
    replicas differ from the static ones by more than their own norm.
    """
    used = sorted(_option_letters(options) & {'d', 'D'})
    if not used:
        return
    raise ConfigurationError(
        f"OASN(options={options!r}): {used} set OASN's Doppler flag "
        f"(unoasn22.f:702-704), which shifts every wavenumber sample's "
        f"frequency at the fixed 15 m/s source/receiver speed of the "
        f"binary's DATA statement (unoasn22.f:762, oaseun31.f:2030-2041); "
        f"no deck field sets that speed, so the run would model a moving "
        f"array nobody configured.",
        remediation=("Drop 'd'/'D' from options=. For a moving source "
                     "and receiver use OAST with options containing 'd' "
                     "and vrec=."),
    )


def _oasn_options(options, run_mode: RunMode) -> str:
    """The OASN option line for a run in ``run_mode``.

    The user-supplied line (or the default ``'J'``) is kept as written,
    with ``N`` (covariance) or ``R`` (replica) appended for the run mode
    when the line does not carry that character already. The order is
    kept because GETOPT takes the source directionality order from the
    first digit it meets (``unoasn22.f:719-725``), and the test is on
    characters because GETOPT reads the line character by character.
    A replica run gets no ``N``: NOIPAR — the noise
    covariance integration — runs under ``IF (CALNSE.or.trfout)``
    (``unoasn22.f:172-174``) and CALNSE comes from 'N', so the letter
    would integrate a covariance the run never returns.
    """
    user_options = options or 'J'
    letter = 'N' if run_mode == RunMode.COVARIANCE else 'R'
    if letter in set(user_options):
        return user_options
    return f'{user_options} {letter}'


def _oasn_settings(mode, *, options, integration_offset, n_wavenumbers,
                   integration_window,
                   surface_noise_level_dB, white_noise_level_dB,
                   deep_noise_level_dB, deep_source_depth,
                   discrete_sources, receiver_gains_dB, c_low_pinned, c_high_pinned, replica_xmin, replica_xmax, replica_nx,
                   replica_ymin, replica_ymax, replica_ny, replica_zmin, replica_zmax, replica_nz, source,
                   receiver, env) -> OASNSettings:
    """The option line for the run ``mode``, Block III's frequency
    sweep, and every value the deck is written from, resolved once on the
    projected ``env``: the phase-speed window
    (:func:`~uacpy.io.oases_writer.oases_wavenumber_bounds` for an unset
    side, refused when inverted), the deep sheet's depth
    (:func:`~uacpy.io.oases_writer.oasn_deep_source_depth`) and, on a
    REPLICA run, the replica grid
    (:func:`~uacpy.io.oases_writer.oasn_replica_ends`); with the notices of
    how the run will go: the receiver ranges OASN does not read, a replica
    depth axis scaled to a thin column and a covariance with no noise
    field. A non-uniform ``source.frequencies`` is refused here: the deck
    carries only ``FREQ1 FREQ2 NFREQ``."""
    c_low, c_high = oases_wavenumber_bounds(env)
    if c_low_pinned is not None:
        c_low = c_low_pinned
    if c_high_pinned is not None:
        c_high = c_high_pinned
    refuse_inverted_window('OASN', float(c_low), float(c_high),
                           c_low=c_low_pinned, c_high=c_high_pinned)
    c_low_origin = ('OASN(c_low=…)' if c_low_pinned is not None
                    else default_c_low(env)[1])
    c_high_origin = ('OASN(c_high=…)' if c_high_pinned is not None
                     else "1e8, OASES' no upper limit (oasn.tex:284-285, "
                          ":479)")
    noise_field = _has_noise_field(
        surface_noise_level_dB=surface_noise_level_dB,
        white_noise_level_dB=white_noise_level_dB,
        deep_noise_level_dB=deep_noise_level_dB,
        discrete_sources=discrete_sources)
    notices = []
    # The OASN writer places a vertical array at x = y = 0;
    # receiver.ranges never reaches the deck.
    rcv_ranges = np.atleast_1d(np.asarray(receiver.ranges, dtype=float))
    if rcv_ranges.size > 1 or (rcv_ranges.size == 1
                               and rcv_ranges[0] > 0.0):
        notices.append(
            message_notice("OASN: receiver.ranges is ignored — OASN models a vertical "
                        "array at x = y = 0 (depths only). Use the replica grid "
                        "(replica_xmin/replica_xmax/...) for horizontal apertures.",
                        FallbackWarning))
    replica_knobs = [
        f"{name}={value:g}" for name, value in (
            ('replica_xmin', replica_xmin), ('replica_xmax', replica_xmax),
            ('replica_nx', replica_nx),
            ('replica_ymin', replica_ymin), ('replica_ymax', replica_ymax),
            ('replica_ny', replica_ny),
            ('replica_zmin', replica_zmin), ('replica_zmax', replica_zmax),
            ('replica_nz', replica_nz))
        if value is not None]
    if mode == RunMode.COVARIANCE and replica_knobs:
        notices.append(message_notice(
            f"OASN({', '.join(replica_knobs)}) is not used: the replica "
            f"grid is read by REPLICA runs only, and this is a COVARIANCE "
            f"run.", FallbackWarning))
    option_line = _oasn_options(options, mode)
    noise_letters = bool(set(option_line) & {'N', 'n', 'T'})
    if mode == RunMode.REPLICA and noise_field and not noise_letters:
        notices.append(message_notice(
            "OASN: the noise field given is not used: a REPLICA run returns "
            "the replicas, and its deck carries no 'N', under which alone "
            "NOIPAR reads the noise block (CALNSE, unoasn22.f:172-174); run "
            "COVARIANCE for the noise covariance.", FallbackWarning))
    if mode == RunMode.REPLICA:
        # The counts the replica block carries: an unset one takes
        # OasnReplicaGrid's (uacpy guidance).
        default_z, default_x, default_y = (OasnReplicaGrid.z[2],
                                           OasnReplicaGrid.x[2],
                                           OasnReplicaGrid.y[2])
        replica_nz = default_z if replica_nz is None else replica_nz
        replica_nx = default_x if replica_nx is None else replica_nx
        replica_ny = default_y if replica_ny is None else replica_ny
        # The bounds it carries, an unset one derived as the writer
        # derives it for a direct caller.
        ((replica_zmin, replica_zmax), (replica_xmin, replica_xmax),
         (replica_ymin, replica_ymax), thin) = oasn_replica_ends(
            env.depth, (replica_zmin, replica_zmax),
            (replica_xmin, replica_xmax), (replica_ymin, replica_ymax),
            who='OASN')
        notices.append(message_notice(thin, FallbackWarning))
    if (mode == RunMode.COVARIANCE
            and not noise_field):
        # With no Block VI source the covariance is just the default
        # white-noise diagonal — 10**(-200/10) = 1e-20 per sensor with
        # zero cross terms (measured) — which reads like a computed
        # noise field but carries none.
        notices.append(
            message_notice("OASN: no noise source is configured, so the covariance "
                        "matrix is only the white-noise floor — a diagonal of 1e-20 "
                        "(the -200 dB default) with zero cross terms. Configure a "
                        "noise field on the constructor: surface_noise_level_dB=, "
                        "deep_noise_level_dB=, discrete_sources=, or an explicit "
                        "white_noise_level_dB=.", ValidityWarning))
    # The line is ``options or 'J'`` plus the run-mode letter, so only a
    # custom string can drop 'J' — and without it (or the 'j' OASN's GETOPT
    # also takes, unoasn22.f:673-675) the binary zeroes the offset and reads
    # a three-token frequency line, the same silent discard every sibling
    # model reports.
    notices.append(_offset_ignored_notice(
        'OASN', integration_offset, n_wavenumbers, option_line,
        offset_letters='Jj', auto_sampling_zeroes_offset=False))
    frequency_sweep = oases_frequency_sweep(source.frequencies, 'OASN')
    window, window_notice = resolve_integration_window(
        'OASN', integration_window, n_wavenumbers,
        'oasn.tex:357-358, :374-375')
    notices.append(window_notice)
    if (integration_window is not None and window_notice is None
            and not ('R' in option_line or 'r' in option_line
                     or (noise_letters and discrete_sources))):
        # The noise blocks read band counts, not a window (oasnun22.f:278,
        # :327); IC1/IC2 sit on the discrete-source and replica lines only
        # (oasnun22.f:421, unoasn22.f:227).
        notices.append(message_notice(
            f"OASN(integration_window={window}) has no effect: this run "
            f"writes no discrete-source or replica integration, the only "
            f"OASN blocks that carry IC1/IC2 (oasn.tex:103-105).",
            FallbackWarning))
    settings = OASNSettings(
        options=option_line,
        frequency_sweep=frequency_sweep,
        integration_offset=integration_offset,
        n_wavenumbers=n_wavenumbers,
        integration_window=window,
        surface_noise_level_dB=surface_noise_level_dB,
        white_noise_level_dB=white_noise_level_dB,
        deep_noise_level_dB=deep_noise_level_dB,
        deep_source_depth=oasn_deep_source_depth(env.depth,
                                                 deep_source_depth),
        discrete_sources=tuple(tuple(ds.items())
                               for ds in (discrete_sources or ())),
        receiver_gains_dB=receiver_gains_dB,
        c_low=float(c_low),
        c_low_origin=c_low_origin,
        c_high=float(c_high),
        c_high_origin=c_high_origin,
        replica_xmin=replica_xmin,
        replica_xmax=replica_xmax,
        replica_nx=replica_nx,
        replica_ymin=replica_ymin,
        replica_ymax=replica_ymax,
        replica_ny=replica_ny,
        replica_zmin=replica_zmin,
        replica_zmax=replica_zmax,
        replica_nz=replica_nz,
        notices=tuple(n for n in notices if n is not None),
    )
    scholte = _oasn_scholte_notice(settings, env, receiver)
    if scholte is None:
        return settings
    return dataclasses.replace(settings,
                               notices=settings.notices + (scholte,))


def _oasn_scholte_notice(engine, env, receiver):
    """The Scholte-wave notice of one OASN run
    (:func:`~uacpy.models._window.scholte_wave_notice`), on the edge of each
    wavenumber integration its deck carries, in deck order, the first that
    leaves a wave out; ``None`` when none does.

    * The surface- and deep-noise blocks integrate in slowness up to
      ``1/CMINN`` (``oasnun22.f:284``, ``:333``; a single band, ``:310``,
      ``:355``): their edge is ``c_low`` at every frequency. Their sources
      are the receivers' reach and, for the deep block, its sheet.
    * The discrete-source block's edge is AUTSMN's or a pinned ``c_low``
      (:func:`~uacpy.models._window.oasn_discrete_integration_edge`), on
      the first record, the farthest source
      (``_oasn_discrete_sources_farthest_first``).
    * The replica block's edge is ``c_low`` under a pinned count
      (``unoasn22.f:243-244``). Under automatic sampling AUTSMN reads a
      range minimum, ``rmins``, the program never sets before its ``MIN``
      (``unoasn22.f:233-234``), so its edge is not known and nothing is
      said.

    The frequency is the band's lowest, FREQ1, the longest wavelength."""
    freq1 = engine.frequency_sweep[0]
    remediation = ("Leave c_low unset (its derived value sits under every "
                   "Scholte wave) or pass c_low below the wave speed to "
                   "include it.")
    letters = set(engine.options)
    blocks = []
    if letters & {'N', 'n', 'T'}:
        dead_band = float(np.float32(OASN_LEVEL_DEAD_BAND_DB))
        surface, _white, deep = _oasn_writer_kwargs(engine)['noise'] \
            .deck_levels()
        pinned = (f"c_low = {engine.c_low:.1f} m/s, the slowness edge of "
                  f"the {{}}-noise integration (oasnun22.f:{{}})")
        if abs(surface) >= dead_band:
            blocks.append((engine.c_low, pinned.format('surface', 284), []))
        if deep >= dead_band:
            blocks.append((engine.c_low, pinned.format('deep', 333),
                           [engine.deep_source_depth]))
        if engine.discrete_sources:
            sources = [dict(ds) for ds in engine.discrete_sources]
            first = max(sources, key=lambda ds: float(np.hypot(
                float(ds['x']), float(ds.get('y', 0.0)))))
            edge, text = oasn_discrete_integration_edge(
                engine.c_low, engine.c_high,
                automatic=engine.n_wavenumbers is None,
                frequency=0.5 * (engine.frequency_sweep[0]
                                 + engine.frequency_sweep[1]),
                first_range=float(np.hypot(float(first['x']),
                                           float(first.get('y', 0.0)))))
            blocks.append((edge, text,
                           [float(ds['depth']) for ds in sources]))
    if letters & {'R', 'r'} and engine.n_wavenumbers is not None:
        (zmin, zmax), _x, _y, _thin = oasn_replica_ends(
            env.depth, (engine.replica_zmin, engine.replica_zmax),
            (engine.replica_xmin, engine.replica_xmax),
            (engine.replica_ymin, engine.replica_ymax), who='OASN')
        nz = (OasnReplicaGrid.z[2] if engine.replica_nz is None
              else engine.replica_nz)
        blocks.append((engine.c_low,
                       f"c_low = {engine.c_low:.1f} m/s with a pinned "
                       f"n_wavenumbers, the replica integration "
                       f"(unoasn22.f:243-244)",
                       list(np.linspace(zmin, zmax, int(nz)))))
    for edge, text, depths in blocks:
        notice = scholte_wave_notice(
            env, SimpleNamespace(depths=np.asarray(depths, dtype=float)),
            receiver, edge, freq1, model_name='OASN', edge_text=text,
            remediation=remediation)
        if notice is not None:
            return notice
    return None


def _oasn_writer_kwargs(engine: OASNSettings) -> dict:
    """The writer's keywords from the resolved settings: the noise field
    (when the option line carries a letter under which OASN reads it,
    ``unoasn22.f:172-174``) and the replica grid as the writer's records
    (metres throughout; the writer converts x/y to the km the deck holds;
    the counts of a COVARIANCE run, which no record carries, take the
    grid's defaults). The phase-speed bounds
    apply identically to the noise, discrete-source and replica
    integrations."""
    def floats(*values):
        return tuple(None if v is None else float(v) for v in values)

    def count(value, default):
        return default if value is None else value

    noise_block = bool(set(engine.options) & {'N', 'n', 'T'})
    return {
        'integration_offset': engine.integration_offset,
        'n_wavenumbers': engine.n_wavenumbers,
        'integration_window': engine.integration_window,
        'receiver_gains_dB': engine.receiver_gains_dB,
        'noise': None if not noise_block else OasnNoise(
            surface_noise_level_dB=engine.surface_noise_level_dB,
            white_noise_level_dB=engine.white_noise_level_dB,
            deep_noise_level_dB=engine.deep_noise_level_dB,
            deep_source_depth=engine.deep_source_depth,
            discrete_sources=tuple(dict(ds)
                                   for ds in engine.discrete_sources),
            c_low=engine.c_low, c_high=engine.c_high,
            c_low_discrete=engine.c_low, c_high_discrete=engine.c_high,
        ),
        'replica': OasnReplicaGrid(
            z=(engine.replica_zmin, engine.replica_zmax,
               count(engine.replica_nz, OasnReplicaGrid.z[2])),
            x=floats(engine.replica_xmin, engine.replica_xmax)
            + (count(engine.replica_nx, OasnReplicaGrid.x[2]),),
            y=floats(engine.replica_ymin, engine.replica_ymax)
            + (count(engine.replica_ny, OasnReplicaGrid.y[2]),),
            c_low=engine.c_low, c_high=engine.c_high,
        ),
    }


def _write_oasn_deck(deck: Path, inputs, name: str) -> None:
    """The OASN deck at ``deck``, written by
    :func:`~uacpy.io.oases_writer.write_oasn_input` from the settings
    ``inputs`` carries; ``name`` is the model the water-absorption
    notice names."""
    engine = inputs.settings.engine
    freq_min, freq_max, n_freq = engine.frequency_sweep
    _warn_if_water_ac_extrapolates(
        name, inputs.env, np.linspace(freq_min, freq_max, int(n_freq)),
        water_ac_anchor_frequency(inputs.env, freq_min, freq_max))
    write_oasn_input(
        filepath=deck,
        env=inputs.env,
        source=inputs.source,
        receiver=inputs.receiver,
        options=engine.options,
        **_oasn_writer_kwargs(engine),
    )


class OASN(OASES):
    """
    OASN — OASES Noise, Covariance Matrices and Signal Replicas.

    Per the OASES manual (``third_party/oases/doc/oasn.tex``) OASN produces
    two frequency-domain array products:

    - **Covariance matrices** (option ``N`` → ``.xsm``): hydrophone ×
      hydrophone correlation per frequency. Used for ambient-noise
      characterisation or as an MFP measurement covariance.
    - **Replica fields** (option ``R`` → ``.rpo``): array response per
      candidate source position per frequency. Used as MFP templates.

    For depth-eigenfunction normal modes use :class:`Kraken`.

    Parameters
    ----------
    executable : Path, optional
        Path to OASN binary. Auto-detected if ``None``.
    options : str, optional
        Custom OASES options string, written as given — its order kept,
        since GETOPT takes the source directionality order from the first
        digit it meets (``unoasn22.f:719-725``) — with the run mode's letter
        (``N`` for COVARIANCE, ``R`` for REPLICA) appended when the line
        lacks it; ``None`` writes ``'J'`` (the complex integration contour
        of the manual's worked OASN deck, oasn.tex:434) and that letter.
    surface_noise_level_dB : float
        Surface-generated noise spectral level (dB re 1 µPa²/Hz),
        Block VI. OASES disables the source when ``abs(level) < 0.01``
        (``oasnun22.f:183``); a level at or below ``-0.01`` is not "off" —
        it names the Fortran unit number of a source-spectrum file uacpy
        does not write, so it is rejected. Requires a vacuum or air
        ``env.surface`` (``oasnun22.f:187``).
    white_noise_level_dB : float, optional
        Uncorrelated (white) noise spectral level per hydrophone
        (dB re 1 µPa²/Hz), added to every covariance-matrix diagonal
        element. OASES has no off switch for this field: it adds
        ``10**(dB/10)`` to the diagonal unconditionally
        (``oasnun22.f:228``, ``:1157``), so an explicit ``0.0`` is a
        literal 0 dB re 1 µPa²/Hz — 1e-12 Pa²/Hz on the returned
        diagonal — per sensor. ``None`` (the
        default) writes -200 dB, whose 1e-20 linear power is
        numerically nil against any noise field (uacpy guidance: the manual
        gives no level; its worked deck writes 50 dB, oasn.tex:475).
    deep_noise_level_dB : float
        Deep broad-area source spectral level (dB re 1 µPa²/Hz). OASES
        uses two thresholds that disagree at exactly 0.01: the source
        radiates only for ``level > 0.01`` (``oasnun22.f:233`` skips it on
        ``.LE.0.01``) while Block VIII is read for ``level >= 0.01``
        (``oasnun22.f:324``). Unlike the surface level, a negative value
        simply disables it — the deep source has no spectrum-file form.

        OASES computes the sheet as written, with two defects in its deep
        loop that uacpy keeps. The plane-geometry flag ``ICDR = 1`` is set
        only for surface noise (``oasnun22.f:509-511``), yet the deep loop
        divides by ``SQRT(FREQ/V)`` as for plane geometry (``:705``;
        PINIT2 multiplies by it only under ``ICDR = 1``,
        ``oaseun31.f:1701-1709``): with ``surface_noise_level_dB`` off, the
        level reads ``10·log10(c/f)`` dB high, c the sound speed at the
        sheet (11.8 dB at 100 Hz in 1500 m/s water). And the deep loop sets
        ``ICUT1``/``ICUT2`` but not the taper window ``ICW1``/``ICW2``
        (``:713-714``; the surface loop sets both, ``:591-594``), so
        CALIN3 tapers each deep band's kernel from a stale index
        (``oaseun31.f:2157-2158``): 3.5-5.3 dB at a receiver 30 m above a
        mid-depth sheet, moving with ``c_low``. Against the same sheet
        computed without both (a 0.02 dB surface block, run first with an
        evanescent count above the deep counts, sets the flag and the
        window), whose depth shape then matches a normal-mode
        sum to 0.04 dB, a deep-only run reads -12 to +10 dB on a 100 m and
        a 2000 m column at 25-400 Hz. A third difference costs only
        samples: the deep block's discrete band spans ``1/CMX`` to
        ``1/CMX`` (``oasnun22.f:349``; the surface block's ends at
        ``1/CMN``, ``:299``), so its samples are weighted by zero.
    deep_source_depth : float, optional
        Depth (m) of the deep broad-area noise source sheet; ``None``
        → half the water depth, uacpy guidance (the manual names DPSD only
        as "the depth of the deep noise source sheet", oasn.tex:334-335).
        Only written when ``deep_noise_level_dB`` is non-zero.
    discrete_sources : list of dict, optional
        Point sources; each dict may carry ``'depth'`` (m), ``'x'``
        (m), ``'y'`` (m) and ``'level_dB'`` (dB, non-negative) — the four
        fields OASES reads (``oasnun22.f:380``). Any other key raises
        ``ConfigurationError``; OASES has no per-source phase. ``'depth'``,
        ``'x'`` and ``'level_dB'`` are required — a source position and level
        have no default — and a missing ``'y'`` is 0, the source on the
        array's x axis (uacpy guidance). The deck lists them farthest from
        the array origin first, whatever their order here: under automatic
        sampling OASN sizes the discrete-source wavenumber step from the
        first record only (``oasnun22.f:426-428``, AUTSMN ``:1693``), and a
        source beyond the resulting range period ``2π/Δk`` would wrap back
        into the window (JKPS §4.5.3); the covariance is a sum over the
        sources (``oasnun22.f:1291``), so their order changes nothing else
        (:func:`~uacpy.io.oases_writer._oasn_discrete_sources_farthest_first`).
    receiver_gains_dB : float or array_like, optional
        Per-element gain (dB) of the array records' ``GAIN`` column
        (``oasnun22.f:36``), one value for all or one per element; ``None``
        writes 0 dB. OASN multiplies the signal and the ambient noise by it
        (``10**(GAIN/20)``, ``oasnun22.f:99``, ``:1084``, ``:1108``,
        ``:1247``, ``:1259``) but not the white noise ("Note that the gain
        is not applied to the white noise!", ``oasn.tex:242-244``); on the
        replicas only the direct (Filon) integration applies it
        (``oasmun21_bin.f:136``; the FFT-interpolation branch, ``:127``,
        does not). Every element is a hydrophone (receiver type 1).
    replica_xmin, replica_xmax : float, optional
        Replica candidate-grid x bounds (m); ``None`` → 100 / 10000,
        uacpy guidance: OASN reads the whole replica grid from the deck
        and has no default of its own (unoasn22.f:184-186).
    replica_nx : int, optional
        Number of replica grid points in x, a whole number >= 1. ``None``
        → 50, uacpy guidance (OASN reads the grid from the deck,
        unoasn22.f:184-186, and has no default). A count pinned on a
        COVARIANCE run, which reads no replica grid, is noticed.
    replica_ymin, replica_ymax : float, optional
        Replica candidate-grid y bounds (m); ``None`` → 0 / 0.
    replica_ny : int, optional
        Number of replica grid points in y, a whole number >= 1. ``None``
        → 1, one y-range, as the manual's worked deck (oasn.tex:491).
    replica_zmin, replica_zmax : float, optional
        Replica candidate-grid depth bounds (m); ``None`` → 10 /
        ``env.depth - 10``, uacpy guidance as for x.
    replica_nz : int, optional
        Number of replica grid points in depth, a whole number >= 1.
        ``None`` → 20, uacpy guidance as for x.
    c_low, c_high : float, optional
        Phase-speed bounds (m/s) for the wavenumber integrations,
        applied to both the noise and replica blocks; ``None`` →
        0.90 × the slowest wave of the problem, shear speeds and Scholte waves
        included
        (:func:`~uacpy.core.engine_defaults.default_c_low`), and ``1e8``.
        An integration whose edge (:func:`_oasn_scholte_notice`) lies above
        a Scholte wave its sources or a receiver are within one wavelength
        of is noticed.
    integration_offset : float
        Wavenumber-integration contour offset (dB/wavelength). Default 0, which
        under the 'J' option of the default option line is not "no offset":
        OASES takes any value below 1e-10 as a request for its own default,
        60*c*(1/c_min - 1/c_max)/N_k dB/wavelength (unoasn22.f:283-293). A
        value above 1e-10 is used as given.
    n_wavenumbers : int, optional
        Number of wavenumber samples; ``None`` lets OASES choose for the
        replica and discrete-source blocks (AUTSMN, which samples the
        propagating band at a step set by the replica grid's far edge and
        tapers the kernel at the phase-speed bounds,
        ``doc/oasn.tex:356-362``). A pinned count is placed equidistantly over
        the whole window between the wavenumbers of ``c_high`` and ``c_low``
        and integrated in full (``doc/oast.tex:522-526``); the manual leaves
        such a sampling to the usual convergence test
        (``doc/oast.tex:543-546``): increase it until the replica and
        covariance levels stop changing. A pinned count is written as given,
        with an info-level note saying so (``verbose='info'``).

        The surface- and deep-noise blocks have no automatic sampling: each
        reads three counts, for its continuous, discrete and evanescent
        bands. ``None`` writes 1600 / 63836 / 100 for surface noise — uacpy
        guidance, measured: OASES's artificial attenuation on the discrete
        band lowers the level as 1/count, and these counts fill its
        65536-sample limit. The level then reads 0.05-0.47 dB below the
        integral without that attenuation on a 100 m column (400 / 400 /
        100: 2.9-5.4 dB) and 1.2-3.5 dB below on a 2000 m sound channel
        (8.4-10.2 dB), whose lossless channel modes keep that integral from
        converging within the limit; 0.1-1.5 s per frequency. Deep noise
        keeps the manual's worked-deck 400 / 400 / 100 (``oasn.tex:480``).
        A pinned ``n`` writes ``n``, ``n`` and ``max(1, n // 4)`` to those
        three bands of either block, as given, with a ``NumericsWarning``
        below 10 discrete samples (uacpy guidance: under ``'J'`` with no
        offset OASES' default contour offset is the discrete band's span
        over its count, unoasn22.f:283-291, and JKPS §4.5.5 warns against an
        offset chosen too large; a count of 1 runs the block as one band,
        oasnun22.f:305-310); in the deep block the middle
        band has zero width (see ``deep_noise_level_dB``), so its trapped modes
        take the ``n // 4``.
    integration_window : (int, int), optional
        ``(IC1, IC2)`` of the discrete-source and replica integrations, the
        first and last wavenumber samples integrated (oasn.tex:103-105; the
        noise blocks read band counts instead). Two whole numbers with ``1
        <= IC1 <= IC2 <= n_wavenumbers``; it applies with a pinned
        ``n_wavenumbers``, and under automatic sampling, where they are
        "insignificant" (oasn.tex:357-358, :374-375), or on a run with
        neither integration, the run says so. ``None`` integrates the whole
        window.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Supported run modes: ``RunMode.COVARIANCE`` (``.xsm`` →
    :class:`Covariance`) and ``RunMode.REPLICA`` (``.rpo`` →
    :class:`Replicas`). The convenience methods
    ``compute_covariance(...)`` / ``compute_replicas(...)`` route to
    these.

    **Units.** The covariance is linear power spectral density in SI
    **Pa²/Hz** (``Covariance.unit``), the unit of a ``csdm()`` of Field
    pressures. OASES has no pressure reference of its own — a hydrophone
    reads "Pa for source level 1 Pa (or µPa for source level 1 µPa)"
    (``oasn.tex:229-230``) and each dB level becomes ``10**(dB/10)``
    (``oasnun22.f:228``) — so the ``.xsm`` holds the covariance in the
    reference of the noise levels, µPa²/Hz, and the model multiplies it by
    ``REFERENCE_PRESSURE_WATER**2`` (1e-12); ``white_noise_level_dB=0.0``
    puts 1e-12 Pa²/Hz on the diagonal. Its ``metadata`` keeps the ``.xsm``
    header's ``surface_noise_level_dB`` and ``white_noise_level_dB`` (dB re
    1 µPa²/Hz). The replicas are complex
    pressure for a unit source, in the travelling-wave phase convention
    (``phase_reference``), as a Field of :class:`OASP` or :class:`Scooter`
    carries it: OASN writes the normal stress ``σ_zz`` (a hydrophone
    receiver selects it, ``oasnun22.f:100-105``), which the wrapper negates.
    In the water ``σ_zz = -p``, so the result is the pressure there; for a
    receiver inside an elastic layer OASES builds ``σ_zz`` from both the
    compressional and the shear potentials (``oasaun31.f:1489-1491``), so
    the covariance and replicas there are of the negated normal stress, not
    the pressure (the model page's §4 measures the gap for OAST).

    A raw ``options`` string carrying ``'d'`` or ``'D'`` is refused: in
    OASN those letters apply a fixed 15 m/s source/receiver Doppler that no
    deck field can set (``unoasn22.f:702-704``, ``:762``). OASN has no
    line-source geometry (its ``'P'`` selects noise-intensity plots), so a
    line Source is refused.

    ``OASN().run_settings(env, source, receiver, mode).engine`` is the
    :class:`OASNSettings` a run would write its deck from, without
    launching anything; every result carries its own as
    ``result.run_settings.engine``.

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Per-model: ``'ssp': 'mean'`` (the layer stack is kept).

    Examples
    --------
    >>> from uacpy.models import OASN
    >>> # A covariance needs a noise field: with no source configured the
    >>> # matrix is only the -200 dB white-noise floor (diagonal 1e-20).
    >>> oasn = OASN(surface_noise_level_dB=40.0)
    >>> cov = oasn.compute_covariance(env, source, receiver)
    >>> # cov.covariance has shape (n_frequencies, n_rcv, n_rcv)
    """

    # Declarative metadata (see PropagationModel / ModelSpec). OASN:
    # range-independent covariance / replica field; multi-layer bottom
    # honoured. Single spectral solve per frequency → mean SSP / median column.
    spec = ModelSpec(
        modes=(RunMode.COVARIANCE, RunMode.REPLICA),
        # A rough interface is column 7 (RG) of the layer record
        # (oaseun31.f:54).
        supports={'layered_bottom', 'elastic_media',
                  'rough_surface', 'rough_bottom'},
        # Point sources only: OASN has no line-source geometry, its 'P'
        # option selects noise-intensity plots (unoasn22.f:655).
        source_types=frozenset({'point'}),
        collapse={'ssp': 'mean'},
        traits=_OASES_TRAITS,
    )
    provenance_id = 'oases'
    _BINARIES = MappingProxyType({'oasn': _oases_binary('oasn2_bin')})
    # The replicas are pressure in the travelling-wave convention once
    # _to_result negates OASN's normal stress; a covariance has no phase.
    outputs = MappingProxyType({
        RunMode.COVARIANCE: OutputSpec('Covariance'),
        RunMode.REPLICA: OutputSpec(
            'Replicas',
            phase_reference=PhaseReference.TRAVELLING_WAVE.value),
    })

    def __init__(
        self,
        *,
        executable: Optional[Path] = None,
        # Output / option control
        options: Optional[str] = None,
        # Noise field (Block VI): broad-area sources expressed as
        # spectral levels (dB re 1 µPa²/Hz) at three depths.
        surface_noise_level_dB: float = 0.0,
        white_noise_level_dB: Optional[float] = None,
        deep_noise_level_dB: float = 0.0,
        deep_source_depth: Optional[float] = None,
        # Discrete (point) sources — list of dicts; each may carry
        # 'depth' (m), 'x' (m), 'y' (m), 'level_dB' (dB).
        discrete_sources: Optional[list] = None,
        # Array element gains (the receiver records' GAIN column), dB.
        receiver_gains_dB: Optional[np.ndarray] = None,
        # Replica candidate-position grid (Block X). x/y/z in metres on
        # the public API, converted to km at write time. ``None`` takes
        # uacpy's defaults, resolved in stage 3 (10 / depth-10 in z, 100 /
        # 10000 in x, 0 / 0 in y; counts 20 / 50 / 1); OASN itself has
        # none.
        replica_xmin: Optional[float] = None,
        replica_xmax: Optional[float] = None,
        replica_nx: Optional[int] = None,
        replica_ymin: Optional[float] = None,
        replica_ymax: Optional[float] = None,
        replica_ny: Optional[int] = None,
        replica_zmin: Optional[float] = None,
        replica_zmax: Optional[float] = None,
        replica_nz: Optional[int] = None,
        # Phase-speed bounds for the wavenumber integrations (m/s).
        # Applied identically to OASN Block VIII (noise / discrete
        # sources) and Block X (replica generator). ``None`` → stage 3
        # derives default_c_low(env) (c_low) and 1e8 (c_high).
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        # Wavenumber-axis sampling & TL plot axes.
        integration_offset: float = OASN_INTEGRATION_OFFSET,
        n_wavenumbers: Optional[int] = None,
        integration_window: Optional[Tuple[int, int]] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: Optional[float] = DEFAULT_RUN_TIMEOUT,
        collapse: Optional[Dict[str, str]] = None,
    ):
        super().__init__(**_host_options(locals()))
        self.options = options
        self.surface_noise_level_dB = float(surface_noise_level_dB)
        # None means "no white noise": the writer maps it to the -200 dB
        # level whose linear power is numerically nil (OASES itself has no
        # off switch for the diagonal term).
        self.white_noise_level_dB = (
            float(white_noise_level_dB) if white_noise_level_dB is not None else None
        )
        self.deep_noise_level_dB = float(deep_noise_level_dB)
        self.deep_source_depth = (
            float(deep_source_depth) if deep_source_depth is not None else None
        )
        self.discrete_sources = list(discrete_sources) if discrete_sources else None
        self.receiver_gains_dB = (
            np.asarray(receiver_gains_dB, dtype=float)
            if receiver_gains_dB is not None else None
        )
        # OASN reads the replica grid's counts NSRCZ / NSRCX / NSRCY as
        # INTEGER point counts (unoasn22.f:184-186), so a pinned count is a
        # whole number >= 1; ``None`` takes the default on a REPLICA run.
        for axis, count in (('x', replica_nx), ('y', replica_ny),
                            ('z', replica_nz)):
            if count is not None:
                whole_count(f'OASN: replica_n{axis}', count, 1)
        self.replica_xmin = replica_xmin
        self.replica_xmax = replica_xmax
        self.replica_nx = None if replica_nx is None else int(replica_nx)
        self.replica_ymin = replica_ymin
        self.replica_ymax = replica_ymax
        self.replica_ny = None if replica_ny is None else int(replica_ny)
        self.replica_zmin = replica_zmin
        self.replica_zmax = replica_zmax
        self.replica_nz = None if replica_nz is None else int(replica_nz)
        self.c_low = c_low
        self.c_high = c_high
        self.integration_offset = float(integration_offset)
        self.n_wavenumbers = n_wavenumbers
        # The discrete-source and replica lines' IC1/IC2, stored as given
        # (checked by _check_knobs).
        self.integration_window = integration_window
        if self.n_wavenumbers is not None:
            # On OASN a pinned NW is not just a density knob: it also sets the
            # INTEGRATION WINDOW. Automatic sampling calls AUTSMN
            # (oasnun22.f:432), which takes DK = 2*pi/(3*RMAX) from the replica
            # grid's far edge (:1693) and returns IC1/IC2 bracketing the
            # propagating band [2*pi*f/C2, 2*pi*f/C1] with 10% margins
            # (:1701-1712); the deck writer emits ``NW 1 NW`` for a pinned
            # count, and :438-440 takes ICUT1=1, ICUT2=NW verbatim, so the
            # pinned deck integrates the whole window, its NW samples placed
            # equidistantly between the wavenumbers of CMAX and CMIN
            # (doc/oast.tex:522-526). The manual offers automatic sampling as
            # the way to a correct answer "without the usual convergence
            # testing" (doc/oast.tex:543-546), so a pinned count needs that
            # test. Said at info level, as OAST, OASP and OASSP say it
            # (``_common._pinned_count_note``): a routine note about the
            # user's choice, which the value reaches the deck as given.
            self._log(
                f"OASN(n_wavenumbers={self.n_wavenumbers}): a pinned wavenumber "
                f"count spreads its samples equidistantly over the whole "
                f"wavenumber window set by c_low and c_high and integrates "
                f"all of it (ICUT1=1, ICUT2=NW), where automatic sampling "
                f"(n_wavenumbers=None) sets the step from the replica grid's "
                f"far edge and tapers at the phase-speed bounds. The OASES "
                f"manual leaves a manual sampling to the usual convergence "
                f"test: double n_wavenumbers until the replica and covariance "
                f"levels stop changing.")
        # OASN's deck has no range-axis or receiver-velocity field: it emits
        # covariance matrices and replicas, not TL-vs-range, so OAST's
        # range_min / range_max / vrec are not knobs here. ``unoasn22.f``
        # recognises a 'D'/'d' option and carries a ``vrec`` variable
        # (:702-704, defaulted to 15.0 m/s at :762) that nothing reads from
        # the input file. The option letter sets ``doppler=.true.``, and
        # CALIN3 — which OASN integrates through (oasnun22.f:604) — then
        # shifts every wavenumber sample's frequency by real(wvno)*vrec/(2*pi)
        # at that fixed 15 m/s (oaseun31.f:2030-2041), a moving-array problem
        # uacpy cannot configure; _reject_doppler_letter refuses it.
        self._check_knobs()

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        self._exe = self._resolve_executable(
            executable, lambda: self._find_binary('oasn'),
        )

    def _check_knobs(self) -> None:
        """Refuse a raw option line carrying the Doppler letter
        (:func:`_reject_doppler_letter`), a pinned phase-speed window that is
        not positive or not ordered
        (:func:`~uacpy.models._window.check_pinned_window`), and a noise or
        replica-grid knob the
        deck's records cannot carry, by the knob's own name: a negative
        ``surface_noise_level_dB`` (read as a spectrum-file unit number, past
        the dead band OASN tests at single precision,
        :meth:`OasnNoise.deck_levels`), a ``discrete_sources`` list OASN
        cannot read, and a ``replica_nz``/``replica_nx``/``replica_ny``
        past NSMAX — the writer's
        own rules (:func:`~uacpy.io.oases_writer._check_oasn_noise_level`,
        ``_check_oasn_discrete_sources``, ``_check_oasn_replica_counts``).
        Run at construction and again by every run
        (:meth:`_validate_engine`), since the attributes can be reassigned
        in between."""
        _reject_doppler_letter(self.options)
        _check_n_wavenumbers_knob(self.n_wavenumbers)
        check_pinned_window('OASN', c_low=self.c_low, c_high=self.c_high)
        refuse_unusable_integration_window('OASN', self.integration_window,
                                           self.n_wavenumbers)
        _check_oasn_noise_level(
            'surface_noise_level_dB',
            OasnNoise(surface_noise_level_dB=self.surface_noise_level_dB).deck_levels()[0],
            dead_band=OASN_LEVEL_DEAD_BAND_DB, who='OASN')
        _check_oasn_discrete_sources(self.discrete_sources or (), who='OASN')
        _check_oasn_replica_counts(
            {'z': self.replica_nz, 'x': self.replica_nx, 'y': self.replica_ny},
            who='OASN', label='replica_n{axis}={n}')

    def _resolve_engine_settings(self, env, source, receiver, settings, *,
                                 given_env, request=None) -> 'OASNSettings':
        """Stage 3: :func:`_oasn_settings` for this model's knobs."""
        return _oasn_settings(
            settings.mode, options=self.options,
            integration_offset=self.integration_offset,
            n_wavenumbers=self.n_wavenumbers,
            integration_window=self.integration_window,
            surface_noise_level_dB=self.surface_noise_level_dB,
            white_noise_level_dB=self.white_noise_level_dB,
            deep_noise_level_dB=self.deep_noise_level_dB,
            deep_source_depth=self.deep_source_depth,
            discrete_sources=self.discrete_sources,
            receiver_gains_dB=self.receiver_gains_dB,
            c_low_pinned=self.c_low, c_high_pinned=self.c_high,
            replica_xmin=self.replica_xmin, replica_xmax=self.replica_xmax, replica_nx=self.replica_nx,
            replica_ymin=self.replica_ymin, replica_ymax=self.replica_ymax, replica_ny=self.replica_ny,
            replica_zmin=self.replica_zmin, replica_zmax=self.replica_zmax, replica_nz=self.replica_nz,
            source=source, receiver=receiver, env=env)

    def _write_input(self, inputs) -> Path:
        """Stage 4: the OASN deck (:func:`_write_oasn_deck`)."""
        deck = inputs.work_dir / f'{_OASN_BASE_NAME}.dat'
        self._log(f"Writing OASN input file: {deck} "
                  f"(options={inputs.settings.engine.options})")
        _write_oasn_deck(deck, inputs, self.model_name)
        return deck

    def _launch(self, inputs, deck: Path) -> None:
        """Stage 4: run ``oasn`` on the deck, and refuse a run that wrote
        no covariance (``.xsm``, COVARIANCE) or no replicas (``.rpo``,
        REPLICA), quoting the binary's streams. FOR016 and FOR014 are always
        set (:func:`_oases_subprocess_env`), so each output can only appear
        under its own name."""
        proc = self._execute(deck.stem, inputs.work_dir)
        if inputs.settings.mode == RunMode.COVARIANCE:
            require_output(
                self.model_name, [inputs.work_dir / f'{deck.stem}.xsm'],
                what='a covariance file', process=proc,
            )
            return
        require_output(
            self.model_name, [inputs.work_dir / f'{deck.stem}.rpo'],
            what='a replica file',
            process=proc,
            hint=('Set the replica grid via OASN(replica_xmin=…, replica_xmax=…, replica_nx=…, '
                  'replica_zmin=…, replica_zmax=…, replica_nz=…) on the constructor.'),
        )

    def _read_output(self, inputs, deck: Path):
        """Stage 4: the ``.xsm`` covariance
        (:func:`~uacpy.io.oases_reader.read_oasn_covariance`, the array
        elements placed at the receiver depths) or the ``.rpo`` replicas
        (:func:`~uacpy.io.oases_reader.read_oasn_replicas`, every grid axis
        in metres)."""
        if inputs.settings.mode == RunMode.COVARIANCE:
            cov_path = inputs.work_dir / f'{deck.stem}.xsm'
            self._log(f"Reading OASN covariance file: {cov_path}")
            return read_oasn_covariance(
                cov_path, receiver_depths=inputs.receiver.depths)
        rep_path = inputs.work_dir / f'{deck.stem}.rpo'
        self._log(f"Reading OASN replica file: {rep_path}")
        return read_oasn_replicas(rep_path)

    def _to_result(self, inputs, deck: Path, raw):
        """Stage 5: the :class:`Covariance` or :class:`Replicas` as the
        run's own.

        The covariance is the ``.xsm``'s linear power, µPa²/Hz in the
        reference of the deck's noise levels, times
        ``REFERENCE_PRESSURE_WATER**2``: Pa²/Hz, computed in float64 and
        stored in the file's own dtype, with ``Covariance.unit`` stating
        it. The header's noise levels (``surface_noise_level_dB``,
        ``white_noise_level_dB``, dB re 1 µPa²/Hz) stay in its ``metadata``.

        The replicas are OASN's field parameter 'N', the normal stress
        ``σ_zz``, which in a fluid is ``-p``; they are negated here to
        pressure — the sign rule OASP applies to its ``.trf``
        (:func:`~uacpy.io._parsers.oasp_trf_pressure`) — and
        tagged with the travelling-wave phase convention of
        ``inputs.settings.output``. Measured against Scooter at the same
        geometry, the raw replicas sit at arg -180° ± 2° from its pressure.
        """
        settings = inputs.settings
        if settings.mode == RunMode.COVARIANCE:
            header = raw.metadata or {}
            levels = {key: header[key]
                      for key in ('surface_noise_level_dB', 'white_noise_level_dB')
                      if key in header}
            raw.covariance = np.multiply(
                raw.covariance, REFERENCE_PRESSURE_WATER ** 2,
                dtype=np.complex128).astype(raw.covariance.dtype)
            raw.unit = 'Pa²/Hz'
            result = self._stamp_file_result(raw, inputs.source,
                                             backend='oasn', **levels)
            primary = (('xsm_file', '.xsm'),)
        else:
            raw.replicas = -raw.replicas
            result = self._stamp_file_result(raw, inputs.source,
                                             backend='oasn')
            result.phase_reference = PhaseReference(
                settings.output.phase_reference)
            primary = (('rpo_file', '.rpo'),)
        attach_output_paths(result, inputs.work_dir, deck.stem,
                            primary_files=primary, cleanup=self.cleanup)
        return result

    # ``compute_covariance`` / ``compute_replicas`` come from the base class
    # (RunMode.COVARIANCE / RunMode.REPLICA dispatch).

    # Mirrors third_party/oases/bin/oasn: replica vectors on unit 14,
    # covariance on unit 16 (oasmun21_bin.f:335 reads FOR016 back), and the
    # unit-26 echo of the array and noise levels INPRCV opens at
    # oasnun22.f:37.
    _FOR_FILES = {
        'FOR014': 'rpo',
        'FOR016': 'xsm',
        'FOR026': 'chk',
    }
    # Every unit above is env-assigned, so the outputs always land on the
    # base_name suffixes (a replica run additionally leaves .021 behind).
    _OUTPUT_SUFFIXES = ('.xsm', '.rpo', '.plt', '.plp', '.chk', '.021')
    # A replica run dumps every replica vector as per-receiver ASCII to
    # unit 92 with a plain WRITE and no OPFILW — a debug block whose guard
    # is commented out (oasmun21_bin.f:157-162) — so a bare fort.92 stays
    # behind in the work dir.
    _OUTPUT_FORT_FILES = ('fort.92',)
