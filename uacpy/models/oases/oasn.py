"""OASN: the OASES noise, covariance and replica program."""

import warnings
from dataclasses import dataclass
from types import MappingProxyType
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.constants import REFERENCE_PRESSURE_WATER
from uacpy.models._spec import ModelSpec
from uacpy.core.run_settings import EngineSettings, OutputSpec, RunMode
from uacpy.core.results import PhaseReference
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, NumericsWarning, ValidityWarning,
)
from uacpy.io.oases_writer import (
    OASN_LEVEL_DEAD_BAND_DB, OasnNoise, OasnReplicaGrid, write_oasn_input,
    _check_oasn_discrete_sources,
    _check_oasn_noise_level, _check_oasn_replica_counts, _resolve_freq_sweep,
    water_ac_anchor_frequency,
)
from uacpy.io.oases_reader import read_oasn_covariance, read_oasn_replicas
from uacpy.models.base import DEFAULT_RUN_TIMEOUT_S
from uacpy.models.oases._common import (
    _OASES_TRAITS, _oases_find_executable, _option_letters,
    _check_n_wavenumbers_knob, _warn_if_water_ac_extrapolates, _warn_offset_ignored_under_auto_sampling,
)
from uacpy.models.oases._base import OASES
from uacpy.models._notices import message_notice
from uacpy.core.engine_defaults import OASN_INTEGRATION_OFFSET


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
        The option line the deck carries: the constructor's letters (``'J'``
        when none) plus ``'N'`` for a covariance or a noise field and
        ``'R'`` for replicas.
    frequency_sweep : tuple of (float, float, int)
        Block III's ``FREQ1 FREQ2 NFREQ``.
    integration_offset : float
        The contour offset (dB/wavelength) on the frequency line.
    n_wavenumbers : int or None
        The wavenumber count; ``None`` is OASES' automatic sampling.
    surface_noise_level, deep_noise_level : float
        Block VI source levels (dB re 1 µPa²/Hz).
    white_noise_level : float or None
        Block VI white-noise level (dB re 1 µPa²/Hz); ``None`` writes the
        writer's -200 dB.
    deep_source_depth : float or None
        Depth (m) of the deep noise sheet; ``None`` is half the water depth.
    discrete_sources : tuple
        One ``((key, value), ...)`` tuple per discrete noise source.
    c_low, c_high : float or None
        Phase-speed bounds (m/s) of every integration; ``None`` leaves the
        writer's derivation.
    replica_xmin, replica_xmax, replica_ymin, replica_ymax, replica_zmin, \
replica_zmax : float or None
        Replica grid bounds (m); ``None`` leaves the writer's default.
    replica_nx, replica_ny, replica_nz : int
        Replica grid counts.
    """

    options: str
    frequency_sweep: Tuple[float, float, int]
    integration_offset: float
    n_wavenumbers: Optional[int]
    surface_noise_level: float
    white_noise_level: Optional[float]
    deep_noise_level: float
    deep_source_depth: Optional[float]
    discrete_sources: Tuple[Tuple[Tuple[str, object], ...], ...]
    c_low: Optional[float]
    c_high: Optional[float]
    replica_xmin: Optional[float]
    replica_xmax: Optional[float]
    replica_nx: int
    replica_ymin: Optional[float]
    replica_ymax: Optional[float]
    replica_ny: int
    replica_zmin: Optional[float]
    replica_zmax: Optional[float]
    replica_nz: int

    def __post_init__(self):
        # Lists (the to_dict form) are stored as the tuples a frozen record
        # holds.
        freq_min, freq_max, n = self.frequency_sweep
        object.__setattr__(self, 'frequency_sweep',
                           (float(freq_min), float(freq_max), int(n)))
        object.__setattr__(self, 'discrete_sources', tuple(
            tuple(tuple(pair) for pair in ds)
            for ds in self.discrete_sources))
        super().__post_init__()


def _has_noise_field(*, surface_noise_level, white_noise_level,
                     deep_noise_level, discrete_sources) -> bool:
    """True when these noise knobs configure any Block VI source.

    White noise counts whenever it was set at all: an explicit 0.0 is
    a literal 0 dB diagonal term, not "off", so the test is against
    ``None`` rather than truthiness.
    """
    return bool(surface_noise_level
                or white_noise_level is not None
                or deep_noise_level or discrete_sources)


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


def _oasn_options(options, run_mode: RunMode, noise_field: bool) -> str:
    """The OASN option line for a run in ``run_mode``.

    The user-supplied options (or default 'J') win for additional
    letters (F, …); ``N`` (covariance) or ``R`` (replica) is added for
    the run mode. NOIPAR — and with it every noise argument — runs only
    under ``IF (CALNSE.or.trfout)`` (``unoasn22.f:173-174``), and CALNSE
    comes from 'N', so a replica run that was given a noise field gets
    'N' too, or the levels it was handed would never reach the deck.
    """
    user_options = options or 'J'
    opt_tokens = set(user_options.split())
    if run_mode == RunMode.COVARIANCE:
        opt_tokens.add('N')
    else:
        opt_tokens.add('R')
    if noise_field:
        opt_tokens.add('N')
    return ' '.join(sorted(opt_tokens))


def _oasn_settings(mode, *, options, integration_offset, n_wavenumbers,
                   surface_noise_level, white_noise_level,
                   deep_noise_level, deep_source_depth,
                   discrete_sources, c_low, c_high, replica_xmin, replica_xmax, replica_nx,
                   replica_ymin, replica_ymax, replica_ny, replica_zmin, replica_zmax, replica_nz, source,
                   receiver) -> OASNSettings:
    """The option line for the run ``mode``, Block III's frequency
    sweep, and every constructor value the deck is written from, with
    the two notices of how the run will go: the receiver ranges OASN
    does not read, and a covariance with no noise field. A non-uniform
    ``source.frequencies`` is refused here: the deck carries only
    ``FREQ1 FREQ2 NFREQ``."""
    noise_field = _has_noise_field(
        surface_noise_level=surface_noise_level,
        white_noise_level=white_noise_level,
        deep_noise_level=deep_noise_level,
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
                        "noise field on the constructor: surface_noise_level=, "
                        "deep_noise_level=, discrete_sources=, or an explicit "
                        "white_noise_level=.", ValidityWarning))
    return OASNSettings(
        options=_oasn_options(options, mode, noise_field),
        frequency_sweep=_resolve_freq_sweep(
            'write_oasn_input', source, float(source.frequencies[0])),
        integration_offset=integration_offset,
        n_wavenumbers=n_wavenumbers,
        surface_noise_level=surface_noise_level,
        white_noise_level=white_noise_level,
        deep_noise_level=deep_noise_level,
        deep_source_depth=deep_source_depth,
        discrete_sources=tuple(tuple(ds.items())
                               for ds in (discrete_sources or ())),
        c_low=c_low,
        c_high=c_high,
        replica_xmin=replica_xmin,
        replica_xmax=replica_xmax,
        replica_nx=replica_nx,
        replica_ymin=replica_ymin,
        replica_ymax=replica_ymax,
        replica_ny=replica_ny,
        replica_zmin=replica_zmin,
        replica_zmax=replica_zmax,
        replica_nz=replica_nz,
        notices=tuple(notices),
    )


def _oasn_writer_kwargs(engine: OASNSettings) -> dict:
    """The writer's keywords from the resolved settings: the noise field
    and the replica grid as the writer's records (metres throughout; the
    writer converts x/y to the km the deck holds). The phase-speed bounds
    apply identically to the noise, discrete-source and replica
    integrations; an unset value is left to the writer's own rule."""
    def floats(*values):
        return tuple(None if v is None else float(v) for v in values)

    return {
        'integration_offset': engine.integration_offset,
        'n_wavenumbers': engine.n_wavenumbers,
        'noise': OasnNoise(
            surface_level=engine.surface_noise_level,
            white_level=engine.white_noise_level,
            deep_level=engine.deep_noise_level,
            deep_source_depth=engine.deep_source_depth,
            discrete_sources=tuple(dict(ds)
                                   for ds in engine.discrete_sources),
            c_low=engine.c_low, c_high=engine.c_high,
            c_low_discrete=engine.c_low, c_high_discrete=engine.c_high,
        ),
        'replica': OasnReplicaGrid(
            z=(engine.replica_zmin, engine.replica_zmax,
               engine.replica_nz),
            x=floats(engine.replica_xmin, engine.replica_xmax)
            + (engine.replica_nx,),
            y=floats(engine.replica_ymin, engine.replica_ymax)
            + (engine.replica_ny,),
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
        Custom OASES options string; ``None`` derives it from the run
        mode (``N`` for COVARIANCE, ``R`` for REPLICA).
    surface_noise_level : float
        Surface-generated noise spectral level (dB re 1 µPa²/Hz),
        Block VI. OASES disables the source when ``abs(level) < 0.01``
        (``oasnun22.f:183``); a level at or below ``-0.01`` is not "off" —
        it names the Fortran unit number of a source-spectrum file uacpy
        does not write, so it is rejected. Requires a vacuum or air
        ``env.surface`` (``oasnun22.f:187``).
    white_noise_level : float, optional
        Uncorrelated (white) noise spectral level per hydrophone
        (dB re 1 µPa²/Hz), added to every covariance-matrix diagonal
        element. OASES has no off switch for this field: it adds
        ``10**(dB/10)`` to the diagonal unconditionally
        (``oasnun22.f:228``, ``:1157``), so an explicit ``0.0`` is a
        literal 0 dB re 1 µPa²/Hz — 1e-12 Pa²/Hz on the returned
        diagonal — per sensor. ``None`` (the
        default) writes -200 dB, whose 1e-20 linear power is
        numerically nil against any noise field.
    deep_noise_level : float
        Deep broad-area source spectral level (dB re 1 µPa²/Hz). OASES
        uses two thresholds that disagree at exactly 0.01: the source
        radiates only for ``level > 0.01`` (``oasnun22.f:233`` skips it on
        ``.LE.0.01``) while Block VIII is read for ``level >= 0.01``
        (``oasnun22.f:324``). Unlike the surface level, a negative value
        simply disables it — the deep source has no spectrum-file form.
    deep_source_depth : float, optional
        Depth (m) of the deep broad-area noise source sheet; ``None``
        → half the water depth. Only written when ``deep_noise_level``
        is non-zero.
    discrete_sources : list of dict, optional
        Point sources; each dict may carry ``'depth'`` (m), ``'x'``
        (m), ``'y'`` (m) and ``'level'`` (dB, non-negative) — the four
        fields OASES reads (``oasnun22.f:380``). Any other key raises
        ``ConfigurationError``; OASES has no per-source phase.
    replica_xmin, replica_xmax : float, optional
        Replica candidate-grid x bounds (m); ``None`` → OASES defaults
        (100 / 10000).
    replica_nx : int
        Number of replica grid points in x. Default 50.
    replica_ymin, replica_ymax : float, optional
        Replica candidate-grid y bounds (m); ``None`` → 0 / 0.
    replica_ny : int
        Number of replica grid points in y. Default 1.
    replica_zmin, replica_zmax : float, optional
        Replica candidate-grid depth bounds (m); ``None`` → 10 /
        ``env.depth - 10``.
    replica_nz : int
        Number of replica grid points in depth. Default 20.
    c_low, c_high : float, optional
        Phase-speed bounds (m/s) for the wavenumber integrations,
        applied to both the noise and replica blocks; ``None`` →
        ``0.95 · min(c_water)`` and ``1e8``.
    integration_offset : float
        Wavenumber-integration contour offset (dB/wavelength). Default 0, which
        under the 'J' option of the default option line is not "no offset":
        OASES takes any value below 1e-10 as a request for its own default,
        60*c*(1/c_min - 1/c_max)/N_k dB/wavelength (unoasn22.f:283-293). A
        value above 1e-10 is used as given.
    n_wavenumbers : int, optional
        Number of wavenumber samples; ``None`` lets OASES choose (AUTSMN, which
        samples the propagating band at a step set by the replica grid's far
        edge). The automatic grid reads the replica levels low toward that
        edge — measured against Scooter on a 100 m Pekeris guide at 150 Hz,
        -0.57 dB at 4.1 km and -3.32 dB at the 10 km edge of the default
        grid. A pinned count integrates the whole wavenumber window and
        converges as it grows (-8.45, -1.95, -0.49, -0.14 dB at that edge
        for NW = 2048, 4096, 8192, 16384), so pin a count large enough for
        the grid when replica or covariance *levels* matter; Bartlett and
        MVDR normalise each replica and are unaffected.
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
    ``REFERENCE_PRESSURE_WATER**2`` (1e-12); ``white_noise_level=0.0``
    puts 1e-12 Pa²/Hz on the diagonal. Its ``metadata`` keeps the ``.xsm``
    header's ``surface_noise_level`` and ``white_noise_level`` (dB re
    1 µPa²/Hz). The replicas are complex
    pressure for a unit source, in the travelling-wave phase convention
    (``phase_reference``), as a Field of :class:`OASP` or :class:`Scooter`
    carries it: OASN writes the normal stress ``σ_zz = -p``, which the
    wrapper negates.

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
    Per-model: ``'ssp': 'mean'``, ``'bottom_range': 'median'`` (the
    layer stack is kept).

    Examples
    --------
    >>> from uacpy.models import OASN
    >>> # A covariance needs a noise field: with no source configured the
    >>> # matrix is only the -200 dB white-noise floor (diagonal 1e-20).
    >>> oasn = OASN(surface_noise_level=40.0)
    >>> cov = oasn.compute_covariance(env, source, receiver)
    >>> # cov.covariance has shape (n_frequencies, n_rcv, n_rcv)
    """

    # Declarative metadata (see PropagationModel / ModelSpec). OASN:
    # range-independent covariance / replica field; multi-layer bottom
    # honoured. Single spectral solve per frequency → mean SSP / median column.
    spec = ModelSpec(
        modes=(RunMode.COVARIANCE, RunMode.REPLICA),
        supports={'layered_bottom', 'elastic_media',
                  'rough_surface', 'rough_bottom'},
        collapse={'ssp': 'mean', 'bottom_range': 'median'},
        traits=_OASES_TRAITS,
    )
    provenance_id = 'oases'
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
        surface_noise_level: float = 0.0,
        white_noise_level: Optional[float] = None,
        deep_noise_level: float = 0.0,
        deep_source_depth: Optional[float] = None,
        # Discrete (point) sources — list of dicts; each may carry
        # 'depth' (m), 'x' (m), 'y' (m), 'level' (dB).
        discrete_sources: Optional[list] = None,
        # Replica candidate-position grid (Block X). x/y/z in metres on
        # the public API, converted to km at write time. ``None`` lets
        # the writer apply OASES defaults (10 / depth-10 in z, 100 /
        # 10000 in x, 0 / 0 in y).
        replica_xmin: Optional[float] = None,
        replica_xmax: Optional[float] = None,
        replica_nx: int = OasnReplicaGrid.x[2],
        replica_ymin: Optional[float] = None,
        replica_ymax: Optional[float] = None,
        replica_ny: int = OasnReplicaGrid.y[2],
        replica_zmin: Optional[float] = None,
        replica_zmax: Optional[float] = None,
        replica_nz: int = OasnReplicaGrid.z[2],
        # Phase-speed bounds for the wavenumber integrations (m/s).
        # Applied identically to OASN Block VIII (noise / discrete
        # sources) and Block X (replica generator). ``None`` → writer
        # derives c_water_min * 0.95 (c_low) and 1e8 (c_high).
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        # Wavenumber-axis sampling & TL plot axes.
        integration_offset: float = OASN_INTEGRATION_OFFSET,
        n_wavenumbers: Optional[int] = None,
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
        self.options = options
        self.surface_noise_level = float(surface_noise_level)
        # None means "no white noise": the writer maps it to the -200 dB
        # level whose linear power is numerically nil (OASES itself has no
        # off switch for the diagonal term).
        self.white_noise_level = (
            float(white_noise_level) if white_noise_level is not None else None
        )
        self.deep_noise_level = float(deep_noise_level)
        self.deep_source_depth = (
            float(deep_source_depth) if deep_source_depth is not None else None
        )
        self.discrete_sources = list(discrete_sources) if discrete_sources else None
        self.replica_xmin = replica_xmin
        self.replica_xmax = replica_xmax
        self.replica_nx = int(replica_nx)
        self.replica_ymin = replica_ymin
        self.replica_ymax = replica_ymax
        self.replica_ny = int(replica_ny)
        self.replica_zmin = replica_zmin
        self.replica_zmax = replica_zmax
        self.replica_nz = int(replica_nz)
        self.c_low = c_low
        self.c_high = c_high
        self.integration_offset = float(integration_offset)
        self.n_wavenumbers = n_wavenumbers
        if self.n_wavenumbers is not None:
            # On OASN a pinned NW is not just a density knob: it also sets the
            # INTEGRATION WINDOW. Automatic sampling calls AUTSMN
            # (oasnun22.f:432), which takes DK = 2*pi/(3*RMAX) from the replica
            # grid's far edge (:1693) and returns IC1/IC2 bracketing the
            # propagating band [2*pi*f/C2, 2*pi*f/C1] with 10% margins
            # (:1701-1712); the deck writer emits ``NW 1 NW`` for a pinned
            # count, and :438-440 takes ICUT1=1, ICUT2=NW verbatim, so the
            # pinned deck integrates the whole window. Measured against
            # Scooter (an independent engine) on a 100 m Pekeris guide at
            # 150 Hz, array at 20-80 m, replica grid 0.1-10 km, the replicas'
            # array energy at the grid's far edge is -8.45, -1.95, -0.49 and
            # -0.14 dB at NW = 2048, 4096, 8192, 16384: a pinned count
            # converges onto Scooter as it grows, and one too small for the
            # grid reads low. The automatic default reads -3.32 dB there
            # (-0.57 dB at 4.1 km), low by an amount that follows the grid's
            # own far edge. Bartlett and MVDR normalise each replica, so
            # localisation is unaffected; the levels are not.
            warnings.warn(
                f"OASN(n_wavenumbers={self.n_wavenumbers}): a pinned wavenumber "
                f"count integrates the whole wavenumber window (ICUT1=1, "
                f"ICUT2=NW) rather than the propagating band automatic "
                f"sampling selects, and replica and covariance levels "
                f"converge as NW grows: against Scooter on a 10 km replica "
                f"grid at 150 Hz, the far edge reads -8.45 dB at NW=2048, "
                f"-0.49 dB at 8192 and -0.14 dB at 16384. Double n_wavenumbers "
                f"until the levels stop changing.",
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
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
        # run() writes ``self.options or 'J'`` plus the run-mode letter, so
        # only a custom string can drop 'J' — and without it the binary
        # zeroes the offset and reads a three-token frequency line, the same
        # silent discard every sibling model warns about.
        _warn_offset_ignored_under_auto_sampling(
            'OASN', self.integration_offset, self.n_wavenumbers,
            options=(self.options if self.options is not None else 'J'),
            auto_sampling_zeroes_offset=False)

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        self._exe = self._resolve_executable(
            executable, lambda: _oases_find_executable(self, 'oasn2_bin'),
        )

    def _check_knobs(self) -> None:
        """Refuse a raw option line carrying the Doppler letter
        (:func:`_reject_doppler_letter`), and a noise or replica-grid knob the
        deck's records cannot carry, by the knob's own name: a negative
        ``surface_noise_level`` (read as a spectrum-file unit number, after
        the deck's one-decimal rounding and dead band,
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
        _check_oasn_noise_level(
            'surface_noise_level',
            OasnNoise(surface_level=self.surface_noise_level).deck_levels()[0],
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
            surface_noise_level=self.surface_noise_level,
            white_noise_level=self.white_noise_level,
            deep_noise_level=self.deep_noise_level,
            deep_source_depth=self.deep_source_depth,
            discrete_sources=self.discrete_sources,
            c_low=self.c_low, c_high=self.c_high,
            replica_xmin=self.replica_xmin, replica_xmax=self.replica_xmax, replica_nx=self.replica_nx,
            replica_ymin=self.replica_ymin, replica_ymax=self.replica_ymax, replica_ny=self.replica_ny,
            replica_zmin=self.replica_zmin, replica_zmax=self.replica_zmax, replica_nz=self.replica_nz,
            source=source, receiver=receiver)

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
            self._require_output(
                [inputs.work_dir / f'{deck.stem}.xsm'],
                what='a covariance file', process=proc,
            )
            return
        self._require_output(
            [inputs.work_dir / f'{deck.stem}.rpo'], what='a replica file',
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
        it. The header's noise levels (``surface_noise_level``,
        ``white_noise_level``, dB re 1 µPa²/Hz) stay in its ``metadata``.

        The replicas are OASN's field parameter 'N', the normal stress
        ``σ_zz``, which in a fluid is ``-p``; they are negated here to
        pressure — the same conversion OASP applies to its ``.trf`` — and
        tagged with the travelling-wave phase convention of
        ``inputs.settings.output``. Measured against Scooter at the same
        geometry, the raw replicas sit at arg -180° ± 2° from its pressure.
        """
        settings = inputs.settings
        if settings.mode == RunMode.COVARIANCE:
            header = raw.metadata or {}
            levels = {key: header[key]
                      for key in ('surface_noise_level', 'white_noise_level')
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
        self._attach_output_paths(result, inputs.work_dir, deck.stem,
                                  primary_files=primary)
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
