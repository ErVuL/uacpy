"""
BOUNCE - Reflection Coefficient Computation Module

BOUNCE computes reflection coefficients for a stack of acoustic/elastic layers.
Part of the Acoustics Toolbox (OALIB).

Outputs:
- .BRC file: Bottom Reflection Coefficient (BotOpt 'F')
  -> Read by BELLHOP (``Bellhop/bellhop.f90:136`` loads the table, ``:688-693``
     applies it in the ``Reflect2D`` contained in that same file;
     ``Bellhop/ReflectMod.f90`` holds a near-identical ``Reflect2D`` that only
     bellhop3D links, per ``Bellhop/Makefile:4,8``), SCOOTER, KRAKENC
- .IRC file: Internal Reflection Coefficient (BotOpt 'P')
  -> Read by KRAKENC (``Kraken/BCImpedancecMod.f90:105``), SCOOTER
     (``Scooter/scooter.f90:357``) and KRAKEL; BELLHOP has no 'P' branch at
     all. Real KRAKEN parses it (``Kraken/BCImpedanceMod.f90:118``) but its
     mode search runs with ``ComplexFlag = .FALSE.``, which discards the
     table for a rigid boundary (:121-125) — uacpy therefore routes an
     ``.irc``/``.brc`` environment to krakenc.exe.

Note: SPARC does not support reflection coefficient files.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union

from uacpy.models.base import (
    PropagationModel, RunMode, ModelSpec, VALID_SOURCE_TYPES,
)
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import Result
from uacpy.core.constants import (
    DEFAULT_C_MIN, DEFAULT_C_MAX_UNBOUNDED,
    parse_boundary_type,
)
from uacpy.core.exceptions import (
    ConfigurationError,
    ModelExecutionError,
    UnsupportedFeatureError,
)
from uacpy.io.refl_io import read_reflection_coefficient, dedupe_reflection_file
from uacpy.io.oalib_writer import write_bounce_input_file, writable_layers
from uacpy.io.units import m_to_km

# bounce.f90 zeroes kMin (drops the 1/cHigh term in NkTab) once cHigh > 1e6.
_KMIN_CUTOFF_CHIGH = 1.0e6

# BOUNCE always writes both tables, so both are cleared before a launch — but a
# tabulated seabed makes one of them an *input* too: write_bottom_section stages
# the user's table there and misc/RefCoef.f90:39 ('F' -> .brc) / :92 ('P' ->
# .irc) open it with STATUS='OLD' before ComputeReflectionCoefficient rewrites
# it. That one must survive the pre-launch sweep.
_STAGED_TABLE_SUFFIX = {'file': '.brc', 'precalc': '.irc'}
_BOUNCE_OUTPUTS = ('.brc', '.irc')

# Mesh density of each medium of the sediment stack: 20 points per wavelength,
# the same density misc/ReadEnvironmentMod.f90:103 uses for its own automatic
# mesh, floored so a thin layer still gets a usable mesh. The cap bounds the
# difference-equation grid bounce.f90:78 allocates (B1..B4/rho/cP/cS sized from
# the sum over media); a stack that needs more than that is refused rather than
# clipped, since a clipped count below Nneeded/2 is a deck the binary rejects
# (ReadEnvironmentMod.f90:110-112).
#: BOUNCE's own manual asks for this, and it is NOT the generic AT auto-mesh
#: density. ``doc/bounce.htm``: "BOUNCE is very fast, there's no reason to
#: skimp. The finer the grid, the more accurate the result. I'll suggest
#: perhaps 100 points/wavelength as a good balance between run time and
#: accuracy. I have seen cases where 10 points/wavelength gave very poor
#: accuracy in R( theta )."
#:
#: 10/wavelength is exactly the binary's own acceptance floor — AT sizes
#: ``Nneeded`` at 20/wavelength (``ReadEnvironmentMod.f90:103``) and rejects a
#: deck below ``Nneeded/2`` — so the density BOUNCE will just barely accept is
#: the one its manual calls very poor. Measured on a 20 m sediment layer over
#: a half-space at 200 Hz, against a converged 400/wavelength reference:
#: max |dR| is 0.0049 at 20/wavelength, 0.0031 at 50, 0.00074 at 100 and
#: 0.00015 at 200. The cost is what the manual says it is — 0.02 s against
#: 0.03 s for the same run at 2 kHz.
_MESH_POINTS_PER_WAVELENGTH = 100

#: The density AT's own automatic mesh uses, kept because the binary's
#: acceptance floor is derived from it (``Nneeded/2``), not because it is the
#: right density to ask for.
_AT_AUTO_MESH_POINTS_PER_WAVELENGTH = 20
_MIN_MESH_POINTS = 100
_MAX_MESH_POINTS = 20000


class Bounce(PropagationModel):
    """
    BOUNCE - Reflection Coefficient Model (Acoustics Toolbox)

    Computes plane wave reflection coefficients for a stack of acoustic/elastic
    layers. The reflection coefficient is written to both .BRC (Bottom Reflection
    Coefficient) and .IRC (Internal Reflection Coefficient) files.

    Model Support:
    - .BRC files: BELLHOP, SCOOTER, KRAKENC
    - .IRC files: KRAKENC, SCOOTER (BELLHOP has no 'P' branch)
    - SPARC: does not support reflection files

    Parameters
    ----------
    executable : Path, optional
        Path to ``bounce``. Auto-detected if ``None``.
    c_low, c_high : float, optional
        Phase-velocity bounds for tabulation (m/s). ``c_low`` must be
        strictly positive (BOUNCE rejects ``c_low <= 0``); ``None``
        (default) derives it from the environment at ``run()`` time as
        ``min(DEFAULT_C_MIN, min(env.ssp))``. ``c_high`` defaults to
        ``DEFAULT_C_MAX_UNBOUNDED``.
    rmax : float, optional
        Max range (m) for angular sampling. ``None`` (default) auto-
        derives from ``receiver.range_max`` at ``run()`` time, falling
        back to ``10000`` m when no receiver range is available.
        Ignored when ``n_angles`` is provided.
    n_angles : int, optional
        Explicit number of angular samples (``NkTab`` in
        ``bounce.f90``). When provided, uacpy back-derives ``rmax`` to
        hit ``~n_angles``.
    interp_ssp : str, optional
        Sample-connection scheme written into ``TopOpt(1)``. A BOUNCE deck
        carries only the seabed stack — its media are 2-point linear slabs —
        so this rarely changes the answer, and ``'quad'`` is rejected
        outright (no water column means no ``.ssp`` for it to read).
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Only emits ``RunMode.REFLECTION``. The result always carries the
    in-memory reflection coefficient as typed attributes
    (``.theta``, ``.R``, ``.phi``); the standalone Python user does not
    need the on-disk files.

    To **chain to another model** (Bellhop / Scooter / Kraken /
    Kraken reading ``acoustic_type='file'``), pin ``work_dir=`` so the
    ``.brc`` / ``.irc`` files outlive the call. The same uniform
    ``(work_dir, cleanup)`` rule every other model uses applies here:

    - ``Bounce(work_dir='./bounce_out')`` ⇒ files persist there
      (``cleanup=False`` because the user owns the dir);
      ``result.metadata['brc_file']`` is a valid path.
    - ``Bounce()`` (no ``work_dir``) ⇒ uacpy uses a temp dir,
      ``cleanup=True`` ⇒ files are removed when ``run()`` returns;
      ``result.metadata`` does not carry the (now stale) file paths.

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    BOUNCE produces ONE BRC consumed across the whole receiver-range
    axis; the median sample is the most representative single profile.
    Per-model: ``'bottom_range': 'median'`` (the layer stack is kept since
    BOUNCE consumes layered seabed columns natively).

    Model characteristics:

    - BOUNCE uses the same environmental file format as KRAKEN
    - The reflection coefficient depends on impedance contrast
    - Supports acoustic, elastic, and poro-elastic layers
    - Tabulated reflection coefficients cover angles from phase velocities [c_low, c_high]
    - **Recommended workflow**: BOUNCE -> .brc -> SCOOTER (most reliable)
    - Both tables go through the standard AT reflection-coefficient path:
      ``.brc`` via ``acoustic_type='file'``, ``.irc`` via
      ``acoustic_type='precalc'``. Kraken routes either to krakenc.exe.

    Defaults auto-derived at ``run()`` time:

    - ``rmax=None`` → ``receiver.range_max`` (or 10 km if 0).
    - ``c_low=None`` → ``min(DEFAULT_C_MIN, min(env.ssp))``, AT's
      ``bounce.htm`` rule "the lowest speed in the problem"; with
      ``c_high=None`` → ``DEFAULT_C_MAX_UNBOUNDED`` this tabulates the full
      0–90° grazing span in cold and brackish water as well as ordinary sea
      water. Lower ``c_high`` to concentrate the samples on a narrower
      angular band. ``c_low`` cannot be raised past the water sound speed —
      that truncates the grazing wedge instead (``run()`` refuses it).
    - TopOpt position 4 reads ``env.absorption``.

    With ``verbose='info'`` the resolved ``rmax`` is logged.

    Examples
    --------
    Compute reflection coefficients for use in other models:

    >>> from uacpy.models import Bounce
    >>> from uacpy.core import Environment, Source, Receiver, BoundaryProperties
    >>> import numpy as np
    >>>
    >>> # Define environment with elastic bottom
    >>> bottom = BoundaryProperties(
    ...     acoustic_type='half-space',
    ...     sound_speed=1600,
    ...     shear_speed=400,
    ...     density=1.8,
    ...     attenuation=0.2,
    ...     shear_attenuation=0.5
    ... )
    >>> env = Environment(name="test", bathymetry=100, bottom=bottom)
    >>> source = Source(depths=50, frequencies=50)
    >>> receiver = Receiver(depths=np.array([50]))
    >>>
    >>> # Pin work_dir so the .brc/.irc files persist for the consumer; a
    >>> # temporary directory keeps them out of the caller's own tree. The
    >>> # files are removed when the ``with`` block exits, so the consumer
    >>> # runs inside it.
    >>> import tempfile
    >>> from uacpy.models import Scooter
    >>> with tempfile.TemporaryDirectory() as d:
    ...     bounce = Bounce(c_low=1400, c_high=10000, rmax=10000, work_dir=d)
    ...     result = bounce.run(env, source, receiver)
    ...     # Output files can be used by different models:
    ...     # - .brc file → BELLHOP, SCOOTER, KRAKENC (experimental)
    ...     # - .irc file → KRAKEN
    ...     bottom_with_rc = BoundaryProperties(
    ...         acoustic_type='file',
    ...         reflection_file=result.metadata['brc_file'],
    ...     )
    ...     env_with_rc = Environment(name="test", bathymetry=100,
    ...                               bottom=bottom_with_rc)
    ...     tl = Scooter().compute_tl(env_with_rc, source, receiver)

    References
    ----------
    - Porter, M.B., "The KRAKEN Normal Mode Program", SACLANT Undersea Research
      Centre Memorandum SM-245, 1991
    - Acoustics Toolbox: http://oalib.hlsresearch.com/
    """

    # Declarative metadata read and validated by PropagationModel. BOUNCE
    # emits only plane-wave reflection coefficients; it consumes layered and
    # elastic seabed columns natively (so those env shapes are *not*
    # collapsed) but handles no range dependence. It produces ONE BRC used
    # across the whole receiver-range axis, so a range-dependent bottom is
    # reduced to its most-representative single column (median).
    spec = ModelSpec(
        modes=(RunMode.REFLECTION,),
        supports={'layered_bottom', 'elastic_media'},
        # Reflection coefficients are independent of source geometry, so
        # rejecting one would break reusing a Source across models.
        source_types=VALID_SOURCE_TYPES,
        collapse={'bottom_range': 'median'},
    )
    source = 'acoustics_toolbox'

    def __init__(
        self,
        executable: Optional[Path] = None,
        c_low: Optional[float] = None,
        c_high: float = DEFAULT_C_MAX_UNBOUNDED,
        rmax: Optional[float] = None,
        n_angles: Optional[int] = None,
        interp_ssp: Optional[str] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = 600.0,
        collapse: Optional[Dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to bounce executable. Auto-detected if None.
        c_low : float, optional
            Minimum phase velocity (m/s) for tabulation. ``None`` (default)
            derives it from the environment at ``run()`` time as
            ``min(DEFAULT_C_MIN, min(env.ssp))`` — AT's ``bounce.htm`` asks
            for "the lowest speed in the problem (say 1400.0)", and 1400 is
            that sentence's example, not its rule, so a column with any water
            slower than 1400 m/s needs the lower value.
            Must be strictly positive (BOUNCE rejects ``c_low <= 0`` — the
            angular grid is derived from ``kx = omega/c``) and must not
            exceed the water sound speed at the seafloor, which BOUNCE takes
            as its reference speed (``bounce.f90:186-195``): above it the
            table starts at ``atan2(sqrt(k0**2 - kMax**2), kMax)`` rather
            than 0 deg and the grazing wedge is unrecoverable. ``run()``
            raises rather than emit such a table.
        c_high : float, optional
            Maximum phase velocity (m/s) for tabulation. Default: 1e9, which
            trips ``bounce.f90:47``'s ``IF ( cHigh > 1.0E6 ) kMin = 0.0`` and
            so tabulates the full 0–90° grazing span. A finite ``c_high``
            stops the table at ``acos(c0 / c_high)`` and every consumer
            silently returns ``R = 0, phi = 0`` above the last tabulated
            angle (``misc/RefCoef.f90:144-149``, both of whose warning WRITEs
            are commented out). Must be strictly greater than c_low.
        rmax : float, optional
            Maximum range (m) for angular sampling. ``None`` (default)
            auto-derives from ``receiver.range_max`` at ``run()`` time,
            falling back to 10000 m when no receiver range is available.
            Ignored when ``n_angles`` is provided. (Internally converted to
            km because BOUNCE's input format is in km.)
        n_angles : int, optional
            Explicit override for the number of angular samples (``NkTab``
            in AT's bounce). If None (default), bounce computes NkTab
            internally from ``rmax``. When provided, uacpy sets ``rmax``
            such that bounce's internal formula yields approximately
            ``n_angles`` samples.
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )
        # BOUNCE meshes only the seabed stack — write_bounce_input_file emits
        # no water column and no .ssp — so the range-dependent 'quad' scheme
        # has no file to read and ERROUTs at NMedia.
        if interp_ssp is not None and str(interp_ssp).lower() == 'quad':
            raise ConfigurationError(
                "Bounce(interp_ssp='quad') is not usable: BOUNCE reflection "
                "decks carry no water column, so no .ssp file is written for "
                "the quad scheme to read and the binary stops at NMedia.",
                remediation="Use 'linear' (default), 'n2linear', 'pchip' or "
                            "'cubic'.",
            )
        self.interp_ssp = interp_ssp

        self.c_low = c_low
        self.c_high = c_high
        self.rmax = rmax
        self.n_angles = n_angles

        # Validate phase velocity bounds up front
        self._validate_phase_speed_bounds()

        # Run modes, capability flags and collapse defaults now come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        self._exe = self._resolve_executable(
            executable,
            lambda: self._find_executable_in_paths(
                'bounce', bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Kraken',
            ),
        )

    def _validate_geometry(self, env, source, receiver, run_mode=None) -> None:
        """No-op: BOUNCE reads no source or receiver geometry.

        Its Fortran driver stops after ``TopOpt``, the SSP, ``BotOpt``,
        ``cLow``/``cHigh`` and ``RMax`` (``bounce.f90``) — it never calls
        ``ReadSzRz``. The plane-wave reflection coefficient is independent of
        source position, so the field models' depth and range-coverage checks
        would reject and warn about geometry that never reaches the binary.
        ``receiver`` is read only to auto-derive ``rmax``.
        """

    def _resolve_c_low(self, env: Environment) -> float:
        """Effective ``c_low`` for this environment (m/s).

        AT's ``doc/bounce.htm``: "The angles used for calculating the
        reflection coefficient are calculated based on the phase-velocity
        interval [ CMin, CMax ]. For a full 90 degree calculation set CMin to
        the lowest speed in the problem (say 1400.0) CMax to 1.0E9." The rule
        is *the lowest speed in the problem*; the 1400 is that sentence's
        illustrative example. ``min(SSP)`` reads the rule directly, and capping
        it at ``DEFAULT_C_MIN`` keeps the tabulation grid identical to the
        historical fixed default for every column whose water never drops below
        1400 m/s — ``NkTab = rmax * f / c_low`` moves only once ``min(SSP)``
        undercuts it.

        ``env.ssp.data``, not :meth:`~uacpy.core.ssp.SoundSpeedProfile.to_pairs`:
        that method returns the **range-0 column** of a range-dependent profile
        by contract (its own docstring says so), so it would read one column and
        miss a slower one further out. ``data`` is the full
        ``(n_depth, n_range)`` block, and every ``collapse['ssp']`` method is a
        per-depth reduction of those same columns, so ``min(data)`` is at or
        below the projected profile's minimum whichever method runs. The two
        agree for a 1-D profile and for the ``'r0'`` collapse default.

        Below the seafloor water speed this buys **head rows, not coverage**:
        ``bounce.f90:198-210`` computes ``theta`` only ``WHERE( k0 > kx )``
        with ``k0 = omega/HSTop%cP``, so every sample slower than the seafloor
        reference takes the ``ELSEWHERE`` branch and comes out as
        ``theta = 0, R = 1, phase = 180`` — byte-identical duplicate rows,
        which ``dedupe_reflection_file`` collapses (and ``stage_reflection_file``
        runs over every staged table). The manual's rule is followed because
        the manual is ground truth for the deck, and the cost of following it
        is rows that are already removed downstream.
        """
        if self.c_low is not None:
            return float(self.c_low)
        return min(DEFAULT_C_MIN, float(env.ssp.data.min()))

    def _reject_c_low_above_the_water(self, env: Environment,
                                      c_low: float) -> None:
        """``c_low`` above the water sound speed drops the grazing wedge.

        ``bounce.f90:46`` sets ``kMax = omega/cLow`` and ``:195`` takes
        ``k0 = omega/c0`` with ``c0 = HSTop%cP`` — which
        ``write_bounce_input_file`` fills with the water sound speed at the
        seafloor. ``:198-210`` computes ``theta = ATAN2( kz, kx )`` only
        ``WHERE( k0 > kx )``, so when ``cLow > c0`` the table simply starts at
        ``ATAN2( sqrt(k0**2 - kMax**2), kMax ) > 0`` instead of 0 deg. Every
        consumer then substitutes ``R = 0, phi = 0`` below that first angle
        (``misc/RefCoef.f90:137-141``, whose warning goes to the ``.prt`` only),
        and ``Bellhop/bellhop.f90:688-693`` applies it as
        ``ray2D%Amp = Amp * RInt%R``, annihilating the ray on its first bounce.
        ``c_low`` 1.3 % above the water speed already costs a mean 5.1 dB /
        max 25 dB against the same environment run as a direct half-space.
        """
        c_ref = float(np.atleast_1d(env.get_sound_speed(env.depth))[0])
        if c_low <= c_ref:
            return
        theta_min = np.degrees(np.arctan2(
            np.sqrt(max(1.0 / c_ref ** 2 - 1.0 / c_low ** 2, 0.0)),
            1.0 / c_low))
        raise ConfigurationError(
            f"Bounce(c_low={c_low}) exceeds the water sound speed at the "
            f"seafloor ({c_ref:.1f} m/s), which BOUNCE uses as its reference "
            f"speed: the table would start at {theta_min:.2f} deg grazing "
            f"instead of 0, and every consumer silently reads R = 0 below "
            f"that.",
            remediation=f"Set c_low <= {c_ref:.1f} m/s, or leave it None so "
                        f"uacpy derives min({DEFAULT_C_MIN:.0f}, min(env.ssp)) "
                        f"— AT bounce.htm's 'lowest speed in the problem', "
                        f"which covers the full grazing span for any column; "
                        f"to concentrate the samples on a narrower band, "
                        f"lower c_high instead.",
        )

    def _validate_phase_speed_bounds(self, c_low: Optional[float] = None
                                     ) -> None:
        """Phase-velocity bounds invariant, enforced at construction AND at
        ``run()`` (a single source of truth for both call sites).

        ``c_low`` is the resolved value. It is ``None`` at construction time
        whenever the caller left ``c_low`` auto, so the two c_low-dependent
        checks run once ``run()`` has resolved it against the environment;
        ``c_high > 0`` is checked either way, since no environment can make a
        non-positive phase velocity admissible.
        """
        if c_low is None:
            c_low = self.c_low
        if self.c_high <= 0:
            raise ConfigurationError(
                f"Bounce requires c_high > 0 strictly (got {self.c_high}). "
                "c_high is the largest phase velocity on the tabulated grid."
            )
        if c_low is not None:
            if c_low <= 0:
                raise ConfigurationError(
                    f"Bounce requires c_low > 0 strictly (got {c_low}). "
                    "c_low is the smallest phase velocity on the tabulated "
                    "grid; 0 would give an infinite wavenumber."
                )
            if self.c_high <= c_low:
                raise ConfigurationError(
                    f"c_high ({self.c_high}) must be strictly greater than "
                    f"c_low ({c_low})."
                )
        if self.rmax is not None and float(self.rmax) <= 0:
            raise ConfigurationError(
                f"Bounce requires rmax > 0 (got {self.rmax}). RMax sets the "
                f"angular sampling density — bounce.f90:49 makes the number of "
                f"tabulated angles proportional to it, and "
                f"misc/ReadEnvironmentMod.f90:140 stops outright on a negative "
                f"value."
            )

    def _n_ktab(self, rmax_m: float, frequency: float, c_low: float) -> int:
        """Angles BOUNCE will tabulate for this deck.

        Reproduces ``bounce.f90:45-49`` on the RMax string the writer is about
        to emit: ``kMin = omega / cHigh`` (zeroed at :47 once ``cHigh > 1e6``),
        ``kMax = omega / cLow`` and
        ``NkTab = INT( 1000 * RMax_km * ( kMax - kMin ) / 2 pi )``.

        The ``.6f`` round-trip is the deck's own precision: ``RMax`` is written
        in km at six decimals (``oalib_writer.write_phase_speed_and_rmax``), so
        the count has to come off the rounded value the binary will read back,
        not off the exact metres.
        """
        rmax_km = float(f"{m_to_km(rmax_m):.6f}")
        omega = 2.0 * np.pi * float(frequency)
        k_min = 0.0 if self.c_high > _KMIN_CUTOFF_CHIGH else omega / self.c_high
        k_max = omega / c_low
        return int(1000.0 * rmax_km * (k_max - k_min) / (2.0 * np.pi))

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        output_duration: Optional[float] = None,
    ) -> Result:
        """
        Run BOUNCE reflection coefficient computation.

        ``.brc`` / ``.irc`` files are written into the model's
        ``work_dir`` (constructor kwarg). Pin the location with
        ``Bounce(work_dir='./bounce_out')`` — a pinned work dir defaults
        ``cleanup=False`` so the files outlive the call and can be
        consumed by Bellhop / Scooter / Kraken; an unpinned
        temp work dir is wiped after ``run()``.

        Parameters
        ----------
        env : Environment
            Ocean environment (bottom properties define the stack).
        source : Source
            Source definition. Only ``frequencies[0]`` is read; BOUNCE
            consumes no source geometry.
        receiver : Receiver
            Read only for ``range_max``, which auto-derives ``rmax`` when the
            constructor left it ``None``.
        run_mode : RunMode, optional
            Must be ``RunMode.REFLECTION`` (the only mode BOUNCE emits)
            or ``None`` (defaults to REFLECTION). Other values raise
            :class:`UnsupportedFeatureError`.

        Returns
        -------
        ReflectionCoefficient
            Typed result with ``theta`` (grazing angles in degrees),
            ``R`` (reflection coefficient magnitude), and ``phi`` (phase
            in radians). The persisted ``.brc`` / ``.irc`` paths live on
            ``result.metadata['brc_file']`` and
            ``result.metadata['irc_file']``.

        Notes
        -----
        - The defaults give full 90-degree coverage: ``c_low=None`` resolves
          to ``min(DEFAULT_C_MIN, min(env.ssp))``, AT ``bounce.htm``'s "lowest
          speed in the problem", and ``c_high=1e9`` triggers ``kmin=0`` in the
          Fortran (see ``bounce.f90``).
        - Larger ``rmax`` gives finer angular resolution.
        - ``.brc`` is consumed by Bellhop / Scooter / Kraken via
          ``BoundaryProperties(acoustic_type='file', reflection_file=…)``.
          ``.irc`` is consumed by Kraken (true normal modes).
        """
        self._require_run_triple(env, source, receiver,
                                 allow_none_receiver=True)
        self._reject_unsupported_run_kwargs(
            frequencies=frequencies, source_waveform=source_waveform,
            sample_rate=sample_rate, output_duration=output_duration)
        run_mode = self._resolve_run_mode(run_mode)
        # Resolved off the raw environment, before the SSP is projected onto a
        # single column. ``_resolve_c_low`` reads ``env.ssp.data`` — every
        # column, not the range-0 one — and each ``collapse['ssp']`` method is
        # a per-depth reduction of those columns, so the value taken here is at
        # or below the projected profile's minimum for any collapse method and
        # can only widen the tabulated wavenumber band, never truncate it.
        c_low = self._resolve_c_low(env)
        self._validate_phase_speed_bounds(c_low)

        # BOUNCE writes one frequency into the .env and returns one table, so a
        # multi-frequency Source would be silently truncated. RunMode.REFLECTION
        # itself stays out of ``_SINGLE_FREQUENCY_MODES`` — OASR does sweep.
        src_freqs = np.atleast_1d(source.frequencies)
        if len(src_freqs) > 1:
            raise ConfigurationError(
                f"Bounce tabulates the reflection coefficient at one "
                f"frequency; got {len(src_freqs)}: {list(src_freqs)}. Loop "
                f"over single-frequency Sources, or use OASR for a "
                f"multi-frequency reflection sweep."
            )

        # Per-call rmax. ``n_angles`` (below) overrides via the inverse of
        # bounce.f90:49  NkTab = INT(1000*RMax_km*(kMax-kMin)/(2π)).
        rmax_origin = 'Bounce(rmax=…)'
        if self.rmax is not None:
            rmax = float(self.rmax)
        else:
            if receiver is None:
                raise TypeError(
                    "Bounce.run(receiver=None): a Receiver is required to "
                    "auto-derive rmax (the range the reflection table is "
                    "propagated to). Pass a Receiver, or pin the range with "
                    "Bounce(rmax=...)."
                )
            recv_rmax = float(receiver.range_max)
            rmax = recv_rmax if recv_rmax > 0 else 10000.0
            rmax_origin = ('receiver.range_max' if recv_rmax > 0
                           else 'the 10 km fallback (no receiver range)')
        if self.n_angles is not None:
            if self.n_angles < 2:
                raise ConfigurationError(
                    f"n_angles must be >= 2 (got {self.n_angles}). "
                    f"bounce.f90:172 spaces the wavenumber grid as "
                    f"Deltak = (kMax - kMin) / (NkTab - 1), so a single "
                    f"tabulated angle divides by zero and the binary spins "
                    f"until the model timeout."
                )
            f_hz = float(np.atleast_1d(source.frequencies)[0])
            omega = 2.0 * np.pi * f_hz
            inv_c_diff = 1.0 / c_low
            if self.c_high is not None and self.c_high <= _KMIN_CUTOFF_CHIGH:
                inv_c_diff -= 1.0 / float(self.c_high)
            if omega * inv_c_diff <= 0:
                raise ConfigurationError(
                    f"Cannot derive rmax from n_angles={self.n_angles}: "
                    f"omega·(1/cLow - 1/cHigh) is non-positive "
                    f"(omega={omega:.3g}, 1/cLow-1/cHigh={inv_c_diff:.3g})."
                )
            # NkTab = INT(1000 * RMax_km * (kMax - kMin) / 2π) and
            # 1000 * RMax_km IS RMax in metres, so the km conversion cancels
            # out of the inversion and this lands directly in metres.
            rmax = (
                float(self.n_angles) * 2.0 * np.pi / (omega * inv_c_diff)
            )
            rmax_origin = f'n_angles={self.n_angles}'

        # Logged after the n_angles branch, which overwrites rmax.
        self._log(f"rmax = {rmax:.1f} m (from {rmax_origin})")

        frequency = float(src_freqs[0])
        n_ktab = self._n_ktab(rmax, frequency, c_low)
        # NkTab is bounded below only. No upper cap: the binary holds four
        # NkTab-length tables (xTab/fTab/gTab/ITab, bounce.f90:52) — tens of
        # bytes per tabulated angle, ~5 MB at a count whose Green's-function
        # cube costs Scooter gigabytes (the sibling Scooter guards with
        # ``_reject_oversized_green_cube``) — and refuses a failed
        # allocation itself through its IAllocStat test.
        if n_ktab < 2:
            consequence = (
                "BOUNCE would write an empty reflection-coefficient table"
                if n_ktab == 0 else
                "bounce.f90:172 spaces the grid as "
                "Deltak = (kMax - kMin) / (NkTab - 1), so a single angle "
                "divides by zero and the binary spins until the model timeout"
            )
            raise ConfigurationError(
                f"This deck asks for {n_ktab} tabulated angle(s): "
                f"bounce.f90:49 derives NkTab = INT(1000 * RMax_km * "
                f"(kMax - kMin) / 2 pi) from RMax = {rmax:g} m at "
                f"{frequency:.4g} Hz with c_low={c_low:g} and "
                f"c_high={self.c_high:g} m/s. {consequence}.",
                remediation=(
                    f"rmax came from {rmax_origin} — raise it (or raise "
                    f"n_angles), or widen the phase-speed window so more "
                    f"wavenumbers fall inside it."
                ),
            )
        self._log(f"NkTab = {n_ktab} tabulated angles", level='debug')

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)
        self._reject_c_low_above_the_water(env, c_low)

        seabed_type = env.bottom.halfspace_at(range=0.0).acoustic_type
        if seabed_type == 'precalc':
            raise UnsupportedFeatureError(
                self.model_name,
                "a 'precalc' (.irc) seabed — misc/RefCoef.f90:103-104 leaves "
                "xTab/fTab/gTab/iTab allocated for the table it just read, so "
                "bounce.f90:52 cannot allocate them for the table it is about "
                "to write and stops with 'Too many points in reflection "
                "coefficient'",
                alternatives=[
                    "acoustic_type='file' with the equivalent .brc table, "
                    "which BOUNCE reads through its own RBot array",
                    "feed the .irc straight to Kraken or Scooter instead of "
                    "re-running BOUNCE on it",
                ],
            )

        fm = self._setup_file_manager()

        try:
            base_name = 'bounce_run'
            input_file = fm.get_path(f'{base_name}.env')

            self._log(f"Writing input file: {input_file}")
            self._write_bounce_input(
                filepath=input_file,
                env=env,
                source=source,
                c_low=c_low,
                c_high=self.c_high,
                rmax=rmax,
            )

            self._log("Running...")
            self._execute(input_file, fm.work_dir,
                          staged_input=_STAGED_TABLE_SUFFIX.get(seabed_type))

            # A missing or empty .brc means the binary died silently; the
            # raised error carries the .prt tail with the actual cause.
            brc_file = self._require_output(
                [fm.get_path(f'{base_name}.brc')],
                what='a reflection-coefficient table (.brc)',
                prt_base=base_name, work_dir=fm.work_dir,
            )

            # Normalise the raw BOUNCE table before it is read back into the
            # ReflectionCoefficient result: bellhopcuda's strict monotonicity
            # check rejects the duplicate near-zero angles bounce.f90 emits when
            # many high-c samples round to the same kx, and the phase column
            # needs unwrapping because the incrementing branch of the Fortran's
            # own unwrap (Kraken/bounce.f90:219) cannot fire. Staging repeats
            # this for the consumer's copy; the rewrite is idempotent. The .irc
            # is left byte-for-byte as BOUNCE wrote it: a different,
            # fixed-format layout that only the Fortran BotOpt='P' path reads.
            dedupe_reflection_file(brc_file)

            self._log(f"Reading output: {brc_file}")
            result = read_reflection_coefficient(str(brc_file))

            theta_out = np.atleast_1d(np.asarray(result.get('theta', []), dtype=float))
            if theta_out.size == 0:
                # The deck asked for n_ktab rows and the binary wrote none:
                # that is an outcome of the run, not a bad configuration, so
                # it is a ModelExecutionError carrying the .prt tail the
                # message points at.
                exc = ModelExecutionError(
                    self.model_name, return_code=0, stdout=None,
                    stderr=(
                        f"Bounce produced an empty reflection-coefficient "
                        f"table — {brc_file.name} has no angle rows, although "
                        f"the deck asked for {n_ktab} (RMax = {rmax:g} m, "
                        f"from {rmax_origin})."
                    ),
                )
                self._attach_prt_tail(exc, fm.work_dir, base_name)
                raise exc

            from uacpy.core.results import ReflectionCoefficient

            field = ReflectionCoefficient(
                theta=result.get('theta', np.array([])),
                R=result.get('R', np.array([])),
                phi=result.get('phi', np.array([])),
                **self._result_kwargs(
                    source,
                    frequencies=frequency,
                    n_points=result.get('n_pts', 0),
                    c_low=c_low,
                    c_high=self.c_high,
                    rmax=rmax,
                    full_result=result,
                ),
            )
            self._attach_output_paths(
                field, fm.work_dir, base_name,
                primary_files=(
                    ('brc_file', '.brc'),
                    ('irc_file', '.irc'),
                ),
            )

            self._log("Simulation complete")
            return field

        finally:
            fm.finish()

    def _resolve_n_mesh(self, env: Environment, frequency: float) -> list:
        """Mesh-point count for each medium of the sediment stack.

        ``write_bounce_input_file`` omits the water column, so the media are
        exactly the writable sediment layers (a bare half-space seabed carries
        none and returns an empty list).

        ``misc/ReadEnvironmentMod.f90:101-112`` sizes every medium
        independently: ``c = alphaR``, then ``IF ( betaR > 0.0 ) c = betaR``,
        ``deltaz = c / freq / 20``, ``Nneeded = INT( thickness / deltaz )``,
        and it aborts with *Mesh is too coarse* when the deck asks for fewer
        than ``Nneeded / 2``. The meshing speed is therefore the medium's
        **shear** speed wherever it has one — an ordinary sand (cs ~ 200 m/s
        against cp ~ 1700 m/s) needs an order of magnitude more points than its
        compressional wavelength suggests.

        ``doc/bounce.htm`` does NOT state the same rule: it asks for 100
        points/wavelength and calls 10 "very poor", where the Fortran merely
        sets the floor at that 10. This wrapper follows the manual and uses the
        Fortran only for the floor — see
        :data:`_MESH_POINTS_PER_WAVELENGTH`.
        """
        counts = []
        for layer in writable_layers(env.bottom.at(range=0.0)):
            thickness = float(layer.thickness)
            shear = float(getattr(layer, 'shear_speed', 0.0) or 0.0)
            speed = shear if shear > 0.0 else float(layer.sound_speed)
            needed = _MESH_POINTS_PER_WAVELENGTH * thickness * frequency / speed
            # The binary's own requirement, and the floor it rejects below.
            at_needed = (_AT_AUTO_MESH_POINTS_PER_WAVELENGTH
                         * thickness * frequency / speed)
            if needed > _MAX_MESH_POINTS and _MAX_MESH_POINTS >= at_needed / 2:
                # Asking for the manual's density would exceed the ceiling, but
                # the ceiling itself still clears what the binary demands, so
                # clip instead of refusing a deck that BOUNCE accepts.
                counts.append(_MAX_MESH_POINTS)
                continue
            if needed > _MAX_MESH_POINTS:
                raise ConfigurationError(
                    f"Bounce: the {thickness:g} m sediment layer needs "
                    f"{int(np.ceil(needed))} mesh points at {frequency:.4g} Hz "
                    f"({_MESH_POINTS_PER_WAVELENGTH} per wavelength of its "
                    f"{speed:.1f} m/s "
                    f"{'shear' if shear > 0.0 else 'compressional'} speed), "
                    f"above the {_MAX_MESH_POINTS}-point ceiling on a single "
                    f"BOUNCE medium.",
                    remediation="Lower the frequency, split the layer into "
                                "thinner ones, or drop the shear speed if the "
                                "layer is meant to be fluid.",
                )
            counts.append(max(_MIN_MESH_POINTS, int(np.ceil(needed))))
        self._log(f"mesh points per medium = {counts}", level='debug')
        return counts

    def _write_bounce_input(
        self,
        filepath: Path,
        env: Environment,
        source: Source,
        c_low: float,
        c_high: float,
        rmax: float,
    ):
        """
        Write BOUNCE input file using ATEnvWriter

        BOUNCE uses ENV format similar to KRAKEN with additional sections:
        - c_low, c_high (phase velocity bounds)
        - RMax in km (for angular sampling) — converted from ``rmax``

        BOUNCE does NOT call ``ReadSzRz``; its Fortran driver reads only
        TopOpt, SSP, BotOpt, cLow/cHigh, RMax. We therefore omit the
        source/receiver depth blocks.
        """
        from uacpy.io.oalib_writer import resolve_ssp_topopt
        ssp_topopt = resolve_ssp_topopt(env, self.interp_ssp)
        bottom_type = parse_boundary_type(env.bottom.halfspace_at(range=0.0).acoustic_type)

        frequency = float(source.frequencies[0])
        n_mesh = self._resolve_n_mesh(env, frequency)

        write_bounce_input_file(
            filepath, env, source,
            ssp_topopt=ssp_topopt,
            bottom_type=bottom_type,
            n_mesh=n_mesh,
            c_low=c_low,
            c_high=c_high,
            rmax=rmax,
            verbose=self.verbose,
        )

    def _execute(self, input_file: Path, work_dir: Path,
                 staged_input: Optional[str] = None):
        """Execute BOUNCE binary via the shared binary-launch helper.

        ``staged_input`` is the suffix ``write_bottom_section`` copied next to
        the deck as *input* for this run (see :data:`_STAGED_TABLE_SUFFIX`); it
        is kept out of the stale-output sweep so the binary can still read it.
        """
        base_name = input_file.stem
        self._run_and_attach_prt(
            [str(self._exe), base_name], work_dir, base_name,
            stale_outputs=tuple(s for s in _BOUNCE_OUTPUTS
                                if s != staged_input))
