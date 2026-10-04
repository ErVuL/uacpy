"""The :class:`Bellhop` engine class: its knobs, its declarations and the
protocol hooks that validate, resolve, write, launch and read one Bellhop
deck (and, for a seabed routed through BOUNCE, the BOUNCE launch before
it)."""

import copy
import dataclasses
import warnings
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Optional, Union

import numpy as np

from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.environment import Bottom, BoundaryProperties, Environment
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, ModelExecutionError, NumericsWarning,
    UnsupportedFeatureError, ValidityWarning,
)
from uacpy.core.receiver import Receiver
from uacpy.core.results import PhaseReference, Result, ResultStack
from uacpy.core.run_settings import OutputSpec, RunMode
from uacpy.core.source import Source
from uacpy.io.bellhop_writer import CERVENY_DEFAULTS, write_bellhop_env_file
from uacpy.io.oalib_writer import (
    SOURCE_TYPE_CODE as _SOURCE_TYPE_CODE,
    reject_biological_edges_under_neighbour_interp,
)
from uacpy.io.refl_io import stage_source_beam_pattern
from uacpy.models._defaults import (
    DEFAULT_BROADBAND_BANDWIDTH_FACTOR, DEFAULT_BROADBAND_N_FREQS,
)
from uacpy.models._spec import EngineTraits, ModelSpec
from uacpy.models._stacking import _slabs_of
from uacpy.models.base import (
    DEFAULT_RUN_TIMEOUT_S, PropagationModel, StageInputs,
)
from uacpy.models.provenance import model_provenance
from uacpy.models.bellhop._backend import (
    _BELLHOP_OUTPUT_SUFFIXES, _bellhop_variant, arrivals_memory_bytes,
    build_command, find_bellhop_executable, has_no_device_signature,
    record_cuda_without_a_device, warn_on_engine_stdout_warnings,
)
from uacpy.models.bellhop._bounce_route import (
    _BounceKnobs, bounce_inputs, bounce_producer, bounce_route,
    bounce_route_notices, is_bounce_launch, prepare_bounce_launch,
    read_bounce_output, resolve_bounce_settings, tabulated_seabed,
)
from uacpy.models.bellhop._checks import (
    check_beam_pattern_spans_the_fan, check_beam_type_supports_receiver_grid,
    check_beam_type_supports_run_mode, check_component, check_knobs,
    reject_precalc_boundary, reject_ray_box_inside_the_domain,
    reject_source_outside_the_medium, reject_unequal_paired_grid,
    warn_on_ignored_cerveny_knobs,
)
from uacpy.models.bellhop._output import assemble, read_bellhop_output
from uacpy.models._budget import work_dir_is_memory_backed
from uacpy.models.bellhop._plan import (
    band_origin, beam_type_run_mode_notice, carrier_frequency,
    carrier_source, deck_receiver, geometry_notices, launch_source,
    line_source_sgb_notice, pad_receiver_ranges, pulse_samples,
    resolve_interp_ssp, resolve_ray_box, resolve_ray_step, ray_step_origin,
)
from uacpy.models.bellhop._settings import BellhopSettings
from uacpy.models.bellhop._synthesis import arrivals_run_settings, synthesise
from uacpy.models.bellhop._tables import (
    _RUN_MODE_TO_INFLUENCE_LETTER, grid_is_paired,
)
from uacpy.models._notices import message_notice
from uacpy.core.engine_defaults import BELLHOP_BEAM_SHIFT, BELLHOP_BEAM_TYPE, BELLHOP_GRID_TYPE, BELLHOP_INTERP_ALTIMETRY, BELLHOP_INTERP_BATHYMETRY, BELLHOP_LAUNCH_ANGLES, BELLHOP_N_BEAMS, BELLHOP_RAY_STEP

#: File root of every Bellhop deck and output.
_BASE_NAME = 'model'


class Bellhop(PropagationModel):
    """
    Bellhop Gaussian beam/ray tracing model

    High-fidelity underwater acoustic ray tracing model developed by
    Michael B. Porter. Automatically detects and uses the fastest available
    version (bellhopcuda > bellhopcxx > Fortran).

    Performance comparison:
    - Fortran: Baseline single-threaded
    - bellhopcxx (C++): 10-30x faster (CPU multithreaded)
    - bellhopcuda (CUDA): 20-100x+ faster (GPU accelerated)

    The backends are not the same code. ``'fortran'`` is Porter's
    ``Acoustics-Toolbox`` BELLHOP; ``'cxx'`` and ``'cuda'`` are ports of
    ``A-New-BellHope``, a fork that deliberately fixes bugs and edge cases in
    it (``bellhopcuda/doc/accuracy.md:38-39``: results are compared "to our
    modified Fortran version, not to the original BELLHOP"). Measured over
    two 2-D scenarios, the ports agree with the Fortran to ~0.3 dB at p99
    with excursions of a few dB at interference nulls, and **no bias**
    (|signed mean| <= 0.0007 dB); ``'cxx'`` and ``'cuda'`` are identical to
    each other. uacpy builds them from its own fork of that port (upstream
    v1.5 plus a Francois-Garrison fix, ``third_party/MODIFICATIONS.md``).
    :attr:`Result.backend` records which binary ran.

    Notes
    -----
    Defaults auto-derived from inputs (no need to override unless tuning):

    - ``n_beams=0`` → Bellhop auto-picks the beam count.
    - ``step=0.0`` → ``env.depth / 50``. Bellhop's own default for a zero
      step is ``depth/10`` (``bellhop.f90:170-174``), which under-resolves a
      near-horizontal refracted ray: measured 26.56 dB max against a
      converged step on Munk 5000 m at 100 Hz.
    - ``z_box=None`` → ``1.2 × env.depth``.
    - ``r_box=None`` → ``1.2 × receiver.range_max`` (or 10 km if 0).
    - ``TopOpt`` position 4 reads from ``env.absorption``
      (``Thorp`` → ``'T'``, ``Biological`` → ``'B'`` + layers,
      ``FrancoisGarrison`` / ``ConstantAbsorption`` / ``None`` → ``' '``,
      the first two carried in the SSP rows' ``alphaI``).
    - Bottom reflection: when ``env.bottom`` is layered and
      ``auto_bounce=True``, BOUNCE is invoked transparently to derive
      the ``.brc`` reflection coefficient table. An elastic halfspace
      needs no table — ``bellhop.f90:694-712`` evaluates the exact
      acousto-elastic reflection coefficient at each boundary hit.

    **Auto-route through BOUNCE.** ``Bellhop.run(...)`` detects a layered
    ``Bottom`` (sediment layers over the halfspace, anywhere along range),
    runs BOUNCE upstream to derive a ``.brc``
    reflection-coefficient table, and re-runs Bellhop against
    ``acoustic_type='file'`` (one ``FallbackWarning``). The user's
    ``collapse={…}`` dict is forwarded to the spawned Bounce. Use
    :meth:`run_with_bounce` for explicit control over BOUNCE parameters.
    A non-layered elastic bottom runs natively: the halfspace's cp/cs pair
    is written to the deck (per range node on a long-format ``.bty``) and
    the ray tracer applies the exact acousto-elastic reflection
    coefficient, so no BOUNCE pass — which would also collapse a
    range-dependent bottom to one column — is interposed.

    Bellhop uses the global :data:`DEFAULT_COLLAPSE` policy without
    overrides — RD bathymetry / RD bottom / RD-SSP (when the model's
    ``interp_ssp='quad'``) are honoured natively.

    **Broadband amplitude approximation (BROADBAND / TIME_SERIES).** The
    arrival set is computed by a *single* ray trace at the band centre
    ``fc`` and reused across the synthesised band
    ``[fc(1-bw/2), fc(1+bw/2)]`` (``bw = bandwidth_factor``). On the
    **BROADBAND** route the travel time ``Re(τ)`` is applied exactly per
    frequency (``exp(-i2πf·τ)``,
    :func:`~uacpy.acoustic_signal.arrival_grid_transfer_function`), so the *timing*
    and spreading of every arrival are correct at all frequencies.
    ``Im(τ)``, however, is frozen at ``fc`` — ``Step.f90:73`` accumulates
    ``tau += hw/CMPLX(c, cimag)`` with ``cimag = alphaT·c²/ω``
    (``misc/AttenMod.f90:113``) — so ``exp(2πf·Im τ)`` applies the volume
    attenuation **linearly in f**. Real absorption laws are not linear
    (Thorp goes as ≈ ``f^1.8``), so the band edges are over- and
    under-attenuated: measured with ``Thorp()`` at ``fc = 10`` kHz and the
    default ``bandwidth_factor=0.5``, +11.51 / −6.75 dB at 40 km against a
    trace run at the edge frequency itself. A run whose band incurs more
    than 0.05 dB/km of this says so. **TIME_SERIES** synthesises in
    the time domain instead (:meth:`Arrivals.to_time_series <uacpy.core.results.Arrivals.to_time_series>`), which delays each
    arrival exactly but takes its volume attenuation as the single factor
    ``exp(2π·fc·Im(τ))`` — frequency-flat like the amplitude below. What is
    held frequency-flat on *both* routes is the geometric beam
    **amplitude and caustic phase**: a Gaussian beam's
    half-width scales as ``√(c/f)``, so the amplitude is only first-order
    correct near ``fc`` and the error grows toward the band edges, largest
    at caustics and in tight ducts. Keep ``bandwidth_factor`` ≲ 1 (±50 %
    of ``fc``) for amplitude-faithful results; for wider bands run several
    sub-bands at different ``fc`` and stitch them (Bellhop User Guide §9).
    A ``ValidityWarning`` fires when ``bandwidth_factor > 1``. The seabed
    **reflection coefficient** is likewise taken at ``fc`` alone: a native
    half-space is evaluated at ``fc``, and a layered bottom's BOUNCE table
    (``auto_bounce``) is tabulated at ``fc`` and reused for every bin. A
    half-space with attenuation in dB/λ has a frequency-independent R(θ),
    but a layer stack's nulls move with ``f·sinθ``, so a BROADBAND /
    TIME_SERIES run that routes through BOUNCE warns.

    Examples
    --------
    >>> bellhop = Bellhop()
    >>> result = bellhop.run(env, source, receiver, run_mode=RunMode.COHERENT_TL)

    Force a specific backend:

    >>> bellhop = Bellhop(backend='fortran')
    >>> bellhop_gpu = Bellhop(backend='cuda')
    """

    # Declarative metadata (see PropagationModel / ModelSpec). Bellhop is the
    # ray engine: honours altimetry, range-dependent bathymetry/bottom,
    # elastic media and a native multi-source-depth grid. No collapse
    # override — uses the base DEFAULT_COLLAPSE unchanged. ``layered_bottom``
    # is False (ray model takes a single half-space per column).
    # ``range_dependent_ssp`` is *instance-dependent* (honoured only on the
    # 'quad' interp), so it is set in __init__ below, not here.
    spec = ModelSpec(
        modes=(
            RunMode.COHERENT_TL, RunMode.INCOHERENT_TL, RunMode.SEMICOHERENT_TL,
            RunMode.RAYS, RunMode.EIGENRAYS, RunMode.ARRIVALS,
            RunMode.BROADBAND, RunMode.TIME_SERIES,
        ),
        supports={
            'altimetry',
            'range_dependent_bathymetry',
            'range_dependent_bottom',
            'elastic_media',
            'source_beam_pattern',
        },
        source_types=frozenset({'point', 'line'}),
        traits=EngineTraits(
            consumes_run_t_start=True,
            # TopOpt position 4 carries env.absorption to the engine.
            consumes_volume_absorption=True,
            # One deck carries every source depth in these modes and the
            # .shd / .ray / .arr readers split the output into a
            # ResultStack, so the base's per-depth loop stands aside.
            # EIGENRAYS (the .ray file cannot be split) and BROADBAND /
            # TIME_SERIES loop through the base.
            native_multi_depth_modes=frozenset({
                RunMode.COHERENT_TL, RunMode.INCOHERENT_TL,
                RunMode.SEMICOHERENT_TL, RunMode.RAYS, RunMode.ARRIVALS,
            }),
            # Stacked by the base's per-depth loop rather than by one deck:
            # the ``.ray`` file carries no per-source boundary to split on.
            python_stacked_modes=frozenset({RunMode.EIGENRAYS}),
            # TIME_SERIES synthesises p(t) by delay-and-sum from one arrivals
            # run, so the grid derived from the pulse only labels the result:
            # it is refused when the pulse has no usable band and never
            # announced, and a ``frequencies=`` is ignored with a warning.
            announced_band_modes=frozenset({RunMode.BROADBAND}),
            time_series_ignores_frequencies=(
                'synthesises p(t) by delay-and-sum from a single arrivals '
                'run at fc'),
        ),
    )
    provenance_id = 'acoustics_toolbox'

    # The TL modes return what influence.f90 writes into the .shd: complex
    # pressure for 'C', the incoherent / semicoherent magnitude sum stored as
    # real dB TL for 'I' / 'S' (as Kraken stores its INCOHERENT_TL). The
    # BROADBAND and TIME_SERIES syntheses are built from the arrivals run.
    outputs = MappingProxyType({
        RunMode.COHERENT_TL: OutputSpec(
            'Field', kind='pressure', unit='Pa',
            phase_reference=PhaseReference.TRAVELLING_WAVE.value,
            coherent=True),
        RunMode.INCOHERENT_TL: OutputSpec(
            'Field', kind='pressure', unit='dB', coherent=False),
        RunMode.SEMICOHERENT_TL: OutputSpec(
            'Field', kind='pressure', unit='dB', coherent=False),
        RunMode.RAYS: OutputSpec('Rays'),
        RunMode.EIGENRAYS: OutputSpec('Rays'),
        RunMode.ARRIVALS: OutputSpec('Arrivals'),
        RunMode.BROADBAND: OutputSpec(
            'Field', kind='pressure', unit='Pa',
            phase_reference=PhaseReference.TRAVELLING_WAVE.value,
            coherent=True),
        RunMode.TIME_SERIES: OutputSpec(
            'Field', kind='pressure', unit='Pa',
            phase_reference=PhaseReference.TIME_DOMAIN_NATIVE.value),
    })

    #: Catalogue entry per resolved engine. ``bellhopcxx`` and ``bellhopcuda``
    #: are one codebase under one copyright holder, so both credit one entry.
    _PROVENANCE_BY_BACKEND = {
        'fortran': 'acoustics_toolbox',
        'cxx': 'bellhopcxx',
        'cuda': 'bellhopcxx',
    }

    @property
    def provenance(self):
        """The catalogue entry for the binary this instance resolved.

        ``__init__`` auto-selects CUDA > C++ > Fortran among whatever
        ``install.sh`` built (:meth:`_find_bellhop_executable`), and the
        C++/CUDA ports are a separate codebase under a separate copyright
        holder — The Regents of the University of California / Scripps MPL,
        ``third_party/bellhopcuda/README.md`` — so the credit follows
        :meth:`select_backend`, the engine that runs, rather than the
        class-level :attr:`provenance_id`. Results therefore agree with themselves:
        ``result.backend`` and ``result.model_source`` name the same binary.

        A pinned ``executable=`` whose basename matches no engine resolves to
        the ``'custom'`` backend; uacpy cannot say what that binary is, so it
        carries no provenance and its results render no model credit rather
        than an invented one.
        """
        backend = getattr(self, '_resolved_backend', None)
        if backend == 'custom':
            return None
        return model_provenance(self._PROVENANCE_BY_BACKEND.get(
            backend, self.provenance_id))

    def select_backend(self, env=None, run_mode=None) -> str:
        """The Bellhop build that runs: ``'fortran'``, ``'cxx'``,
        ``'cuda'`` or ``'custom'`` (a pinned ``executable=`` uacpy cannot
        name), resolved at construction from ``backend=`` and the installed
        binaries (CUDA > C++ > Fortran). Round-trips with ``backend=`` and
        mirrors ``Kraken.select_backend`` / ``RAM.select_backend``; ``env``
        and ``run_mode`` are accepted for signature parity and do not
        change the choice.

        Parameters
        ----------
        env, run_mode : optional
            Accepted for signature parity; they do not change the choice.
        """
        return self._resolved_backend

    def __init__(
        self,
        *,
        # Binary selection
        executable: Optional[Path] = None,
        backend: Optional[str] = None,
        dimensionality: str = '2D',
        # Ray fan and trace bounding box
        beam_type: str = BELLHOP_BEAM_TYPE,
        n_beams: int = BELLHOP_N_BEAMS,
        launch_angles: tuple = BELLHOP_LAUNCH_ANGLES,
        ray_step: float = BELLHOP_RAY_STEP,
        z_box: Optional[float] = None,
        r_box: Optional[float] = None,
        # Receiver grid and environment interpolation
        grid_type: str = BELLHOP_GRID_TYPE,
        interp_ssp: Optional[str] = None,
        interp_bathymetry: str = BELLHOP_INTERP_BATHYMETRY,
        interp_altimetry: str = BELLHOP_INTERP_ALTIMETRY,
        # Cerveny beam shape (written only for beam_type 'C' / 'R')
        beam_width_type: str = CERVENY_DEFAULTS['beam_width_type'],
        beam_curvature: str = CERVENY_DEFAULTS['beam_curvature'],
        eps_multiplier: float = CERVENY_DEFAULTS['eps_multiplier'],
        r_loop: float = CERVENY_DEFAULTS['r_loop'],
        n_image: int = CERVENY_DEFAULTS['n_image'],
        ib_win: int = CERVENY_DEFAULTS['ib_win'],
        component: str = CERVENY_DEFAULTS['component'],
        beam_shift: bool = BELLHOP_BEAM_SHIFT,
        # Broadband synthesis knobs (BROADBAND / TIME_SERIES paths)
        n_freqs: int = DEFAULT_BROADBAND_N_FREQS,
        bandwidth_factor: float = DEFAULT_BROADBAND_BANDWIDTH_FACTOR,
        auto_bounce: bool = True,
        # Standard plumbing
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = DEFAULT_RUN_TIMEOUT_S,
        collapse: Optional[Dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        executable : str or Path, optional
            Path to bellhop executable. Auto-detected if None.
        backend : str, optional
            Force a binary variant: ``'fortran'`` (bellhop), ``'cxx'``
            (bellhopcxx), or ``'cuda'`` (bellhopcuda). ``None`` (default)
            auto-selects, preferring CUDA > C++ > Fortran among whatever
            ``install.sh`` built. An explicitly requested variant that
            isn't installed raises :class:`ExecutableNotFoundError` naming
            the ``install.sh`` flag that builds it. Mirrors
            ``RAM(backend=...)``.
        dimensionality : str, optional
            Only ``'2D'`` (default) is supported — it is passed to the
            bellhopcxx / bellhopcuda CLIs as ``--2D`` (accepted, not required:
            without a flag they assume 2D, ``cmdline.cpp:186-191``; the Fortran
            binary takes no flag).
            ``'3D'`` is rejected because 3-D running is not yet available: the
            env writer cannot emit a 3D-format input file, so a 3D flag would
            mis-drive the binary. The BELLHOP3D / FIELD3D *file* readers and
            writers are already in :mod:`uacpy.io`, retained for it —
            ``write_bty_3d`` / ``read_boundary_3d``, ``read_ssp_3d``,
            ``write_field3dflp`` / ``read_flp3d``.
        beam_type : str, optional
            ``G`` geometric hat, Cartesian (default) | ``g`` geometric hat,
            ray-centered | ``B`` geometric Gaussian, Cartesian | ``R`` Cerveny
            ray-centered | ``C`` Cerveny Cartesian | ``S`` Bucker's simple
            Gaussian, Cartesian
            (``influence.f90:658``). Each letter selects one influence routine at
            ``bellhop.f90:296-311``; ``G`` is the ``CASE DEFAULT`` there, so an
            unrecognised letter would also run geometric-hat Cartesian —
            :func:`validate_beam_type` refuses one instead.

            **These letters are RunType(2:2) and share their alphabet with
            RunType(1:1), where the same characters mean something else:** ``C``
            is coherent TL in position 1 and Cerveny-Cartesian in position 2,
            ``S`` semi-coherent TL against Bucker's Gaussian, ``R`` a ray trace
            against Cerveny ray-centred. ``run_mode`` sets position 1 and
            ``beam_type`` position 2; they are never interchangeable.

            ``B`` and ``G`` are both GEOMETRIC beams (JKPS Sect. 3.3.5.5: the
            Gaussian replaces the hat shape function at the same fan-derived
            width and "serves simply to interpolate the field"). Neither is right
            everywhere — measured against Kraken, ``G`` wins on a
            boundary-dominated Pekeris guide (1.4 vs 2.3 dB rms) and ``B`` on the
            Munk profile's shadow zones and caustics (2.7 vs 3.2 dB). ``G`` is
            the default because ``B``'s width floor of ``πλ`` biases TL in a
            duct only a few ``πλ`` deep: on a 100 m channel at 200 Hz ``B``
            reads +2.55 dB above a wavenumber-integration reference where
            ``G`` reads +0.32 dB (+3.35 / −0.12 dB on Pekeris at 100 Hz). Pass
            ``beam_type='B'`` for finite caustics. Only ``G`` is exactly
            reciprocal (``B``
            differs by ~1 dB under source/receiver exchange, at every ray step and
            beam count tried); use ``G`` when a reciprocal field matters, e.g. to
            fill a source-receiver matrix from one half. The measurements are
            pinned in ``test_bellhop.py``; ``C``, ``R`` and ``g`` never write the
            first receiver-range column, which the wrapper pads for (see
            :func:`~uacpy.models.bellhop._plan.pad_receiver_ranges`).

            The Cerveny beams ``C`` and ``R`` run COHERENT_TL (and RAYS) only:
            their INCOHERENT_TL / SEMICOHERENT_TL level falls as
            ``n_beams**-0.5`` (4.8 dB per tripling of the beam count) and is
            refused, with the geometric beams as the remedy. ``S`` on a line
            source differs between the backends: the Fortran ``InfluenceSGB``
            applies the point-source launch weight ``sqrt(cos(alpha))``
            unconditionally (``influence.f90:665``, where every other routine
            tests ``RunType(4:4) == 'R'``), while bellhopcxx / bellhopcuda weight
            a line source by 1. Measured on a 100 m Pekeris guide at 300 Hz, the
            mean TL against Scooter is 0.45 dB on ``backend='fortran'`` and
            0.28 dB on the ports (0.27 dB on both for a point source); a
            COHERENT_TL run on the Fortran backend says so.
            ``'b'`` (geometric Gaussian, ray-centered) is rejected — the
            Fortran solver aborts on it and the C++/CUDA ports silently
            substitute the Cartesian beam.
        n_beams : int, optional
            Number of beams, ``>= 0``. Passing 0 defers to Bellhop's own
            estimate — ``angleMod.f90:38`` tests ``Nalpha == 0`` exactly,
            then takes ``MAX(INT(0.3 * Rmax * f / c0), 300)`` raised further
            by a beam-width-versus-depth rule (``angleMod.f90:44-50``); a
            ray-trace run gets 50. Default: 0.
            Default ``0``.
        launch_angles : tuple, optional
            Launch angle limits (min, max) in degrees, min < max.
            Default: (-80, 80).
            Default ``(-80, 80)``.
        ray_step : float, optional
            Ray step size in meters, ``>= 0``. 0 resolves to
            ``env.depth / 50`` (see
            :data:`~uacpy.models.bellhop._plan._STEP_PER_DEPTH`).
            Default: 0.0.
            Default ``0.0``.
        z_box : float, optional
            Maximum depth for ray box. None = 1.2 * max depth. Default: None.
            A box at or above the seafloor is refused on every mode but
            RAYS, where it sets the extent of the trace.
        z_box, r_box : float, optional
            Ray-trace bounding box (m); rays are dropped once they leave it.
            ``None`` ⇒ ``1.2 ×`` the receiver extent (``z_box = 1.2 × env.depth``,
            ``r_box = 1.2 × range_max``, or 10 km when the receiver range is 0).
            Box%r is a horizontal-range cut-off, so the 1.2× pad already
            captures arrivals at the outer receivers; do not enlarge it past a
            range-dependent SSP's defined extent.
        r_box : float, optional
            Maximum range for ray box. None = 1.2 * max range. Default: None.
            A box short of the farthest receiver is refused on every mode
            but RAYS, where it sets the extent of the trace.
        grid_type : str, optional
            ``'R'`` rectilinear (default) | ``'I'`` irregular: the i-th depth
            pairs with the i-th range after BELLHOP sorts both lists
            (``SourceReceiverPositions.f90:224``), so the pairs always run
            shallow-near to deep-far. Arbitrary (depth, range) points: run
            ``'R'`` and sample with ``Field.at``.
        interp_ssp : str, optional
            SSP connection scheme. ``None`` (default) auto-picks
            ``'quad'`` for a range-dependent ``env.ssp`` and ``'linear'``
            otherwise. Explicit values: ``'linear'``, ``'pchip'``,
            ``'spline'``, ``'quad'``, ``'n2linear'``.
            ``env.ssp.kind='isovelocity'`` always forces ``'C'`` regardless.
            ``'analytic'`` is **not** accepted: AT's ``'A'`` profile is a
            hard-coded Munk curve on a fixed 5000 m grid (``misc/munk.f90``)
            that ignores ``env.ssp``, so it is refused with that explanation
            (:func:`~uacpy.io.oalib_writer.resolve_ssp_topopt`).
        interp_bathymetry : str, optional
            ``.bty`` interpolation. ``'linear'`` (default) or
            ``'curvilinear'``.
        interp_bathymetry, interp_altimetry : str, optional
            ``.bty`` / ``.ati`` interpolation, ``'linear'`` (default) or
            ``'curvilinear'``.
        interp_altimetry : str, optional
            ``.ati`` interpolation. ``'linear'`` (default) or
            ``'curvilinear'``.
        beam_width_type : {'F', 'M', 'W'}, optional
            Cerveny beam width type (``ReadEnvironmentBell.f90:178-181``):
            'F' = space-filling (default), 'M' = minimum width, 'W' = WKB
            beams. Only used when ``beam_type`` ∈ ('C', 'R').
            Cerveny only (``bellhop.f90:373-390``).
        beam_curvature : {'D', 'S', 'Z'}, optional
            Curvature condition applied on a boundary reflection
            (``ReadEnvironmentBell.f90:202-210``, ``bellhop.f90:667-672``).
            ``'D'`` curvature doubling | ``'S'`` standard curvature |
            ``'Z'`` curvature zeroing.
            Default ``'D'``.
        eps_multiplier : float, optional
            Beam-width epsilon multiplier. Default: 1.0.
        eps_multiplier, r_loop, n_image, ib_win : optional
            Cerveny advanced beam knobs (used when ``beam_type ∈ {C, R}``).
            ``r_loop`` is in metres.
        r_loop : float, optional
            Range (m) at which to choose the beam width. Default: 1000.0.
        n_image : int, optional
            Number of images. Default: 1.
        ib_win : int, optional
            Beam-windowing parameter. Default: 4.
        component : {'P', 'V', 'H'}, optional
            Reserved: only 'P' (pressure, the default) returns a field.
            The letter selects the component the Cerveny **ray-centred**
            influence routine computes (influence.f90:120-130): 'P'
            pressure, 'V' vertical particle velocity, 'H' horizontal
            particle velocity. That routine is ``beam_type='R'`` alone —
            every other beam type ignores the letter (with a warning), and
            'V'/'H' on 'R' is refused, since the resulting .shd holds
            particle velocity in m/s and a :class:`Field` can only report it
            as pressure.
        beam_shift : bool, optional
            When True, sets RunType position 7 to 'S' enabling beam-shift
            on boundary reflections. Default: False.
        n_freqs : int, optional
            Number of frequency bins for BROADBAND / TIME_SERIES
            synthesis when the band is expanded from a single centre
            frequency. Default: :data:`DEFAULT_BROADBAND_N_FREQS`.
        n_freqs, bandwidth_factor : optional
            The band a BROADBAND / TIME_SERIES run synthesises around a single
            centre frequency: its bin count and its fractional bandwidth (see
            the constructor's entries).
        bandwidth_factor : float, optional
            Fractional bandwidth of the synthesised band
            ``[fc·(1-bw/2), fc·(1+bw/2)]`` around a single centre
            frequency. Default:
            :data:`DEFAULT_BROADBAND_BANDWIDTH_FACTOR`.
        auto_bounce : bool, optional
            Default ``True``. When ``env`` carries a *layered* ``Bottom``
            (sediment layers Bellhop's ray tracer cannot mesh), ``run(...)``
            auto-routes through BOUNCE to derive a ``.brc`` reflection-
            coefficient table and re-runs Bellhop against
            ``acoustic_type='file'``, attaching the in-memory
            :class:`ReflectionCoefficient` as
            ``result.components['bounce']``. Elastic half-spaces
            (shear, range-dependent included) never route: Bellhop applies
            the exact acousto-elastic reflection coefficient natively
            (``Bellhop/bellhop.f90:694-712``). Set ``False`` to skip the
            auto-route — Bellhop then collapses the layered bottom via its
            own ``collapse={…}`` policy, with one ``FallbackWarning``.
            ``run_with_bounce(...)`` always uses BOUNCE regardless.
        use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
            Standard plumbing (see :class:`PropagationModel`).
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        #
        # Instance-dependent override: RD-SSP is honoured natively only via the
        # external 2-D .ssp file, which Bellhop reaches on the 'quad' interp.
        # ``interp_ssp=None`` auto-picks 'quad' for a range-dependent SSP
        # (oalib_writer.resolve_ssp_interp), so the default path honours it. A
        # user who *pins* a non-quad interp ('linear', 'spline', …) gets a 1-D
        # collapse — so the flag must be False for that instance, and
        # ``_project_environment`` does the collapse with the standard
        # one-warning-per-feature. Keeping the flag honest means the advertised
        # capability matches the run-time behaviour.
        self._supports_range_dependent_ssp = (
            interp_ssp is None or str(interp_ssp).lower() == 'quad'
        )

        self.beam_type = beam_type
        self.n_beams = n_beams
        self.launch_angles = launch_angles
        self.ray_step = ray_step
        self.z_box = z_box
        self.r_box = r_box
        self.grid_type = grid_type
        self.interp_ssp = interp_ssp
        self.interp_bathymetry = interp_bathymetry
        self.interp_altimetry = interp_altimetry
        self.beam_width_type = beam_width_type
        self.beam_curvature = beam_curvature
        self.eps_multiplier = float(eps_multiplier)
        self.r_loop = float(r_loop)
        self.n_image = int(n_image)
        self.ib_win = int(ib_win)
        self.component = component
        self.beam_shift = bool(beam_shift)
        self.n_freqs = int(n_freqs)
        self.bandwidth_factor = float(bandwidth_factor)
        self._check_knobs()
        check_component(component=self.component, beam_type=self.beam_type)
        warn_on_ignored_cerveny_knobs(
            beam_type=self.beam_type, beam_width_type=self.beam_width_type,
            beam_curvature=self.beam_curvature,
            eps_multiplier=self.eps_multiplier, r_loop=self.r_loop,
            n_image=self.n_image, ib_win=self.ib_win)
        if self.bandwidth_factor > 1.0:
            warnings.warn(
                f"Bellhop: bandwidth_factor={self.bandwidth_factor} spans "
                f">±50% of fc; arrival amplitudes are computed at fc and held "
                f"frequency-flat, so broadband amplitudes degrade toward the "
                f"band edges (worst at caustics/ducts). Run sub-bands at "
                f"different fc for wide bandwidths (Bellhop User Guide §9).",
                ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
        self.auto_bounce = bool(auto_bounce)
        if backend is not None and backend not in ('fortran', 'cxx', 'cuda'):
            raise ConfigurationError(
                f"Bellhop(backend={backend!r}) is not a known backend. "
                f"Choose 'fortran', 'cxx', 'cuda', or None for automatic "
                f"selection (CUDA > C++ > Fortran)."
            )
        self.backend = backend
        if dimensionality != '2D':
            raise UnsupportedFeatureError(
                'Bellhop',
                f"dimensionality={dimensionality!r} — 3-D running is not yet "
                f"available: the env writer produces 2D-format input files, "
                f"so a '3D' (or any other) flag would mis-drive the binary "
                f"(silent 2D run on Fortran, abort on cxx/cuda). The file "
                f"layer for it is already in the package and kept for when "
                f"3-D is wired up — uacpy.io.write_bty_3d / read_boundary_3d "
                f"(BELLHOP3D boundary grids), read_ssp_3d (hexahedral SSP) "
                f"and write_field3dflp / read_flp3d (FIELD3D decks)",
                alternatives=["'2D'"],
                alternatives_label='dimensionality values',
            )
        self.dimensionality = dimensionality
        self._resolved_backend = "unknown"

        # A copy re-resolves the binary honoring ``backend=`` instead of
        # re-pinning the already-resolved path. A pinned path is read the
        # way an auto-picked one is — its basename says which engine it is,
        # and with it whether the .arr needs the pair-merge and whether the
        # CLI takes ``--<dim>``; a name matching no engine is 'custom'.
        self._exe = self._resolve_executable(
            executable, self._find_bellhop_executable,
        )
        if self.executable is not None:
            self._resolved_backend = _bellhop_variant(self._exe) or "custom"

        self._log(f"Using Bellhop {self._resolved_backend}: {self._exe}")

    def _validate_geometry(self, env, source, receiver, run_mode=None):
        """The shared geometry checks, then the refusal only Bellhop needs:
        :func:`~uacpy.models.bellhop._checks.reject_source_outside_the_medium`
        of a source on or outside a boundary at its own range."""
        super()._validate_geometry(env, source, receiver, run_mode)
        reject_source_outside_the_medium(env, source)

    def _receiver_grid_is_paired(self, receiver) -> bool:
        """:func:`~uacpy.models.bellhop._tables.grid_is_paired` of this model's
        ``grid_type``."""
        return grid_is_paired(self.grid_type)

    def _find_bellhop_executable(self) -> Path:
        """:func:`~uacpy.models.bellhop._backend.find_bellhop_executable` for
        this model's ``backend`` and path search; :meth:`select_backend` is
        inferred
        from the name of the returned path."""
        path = find_bellhop_executable(self.backend,
                                       find=self._find_executable_in_paths)
        self._resolved_backend = _bellhop_variant(path) or 'fortran'
        return path

    # ── stage 1: the environment the engine runs on ────────────────────

    def _project_environment(self, env, *, request=None):
        """Stage 1: the environment Bellhop runs on.

        Without the BOUNCE route, the shared projection
        (:meth:`PropagationModel._project_environment`); a layered bottom
        with ``auto_bounce=False`` is flattened to a halfspace there, which
        this says first. With the route, the water column and the surface
        are projected as Bellhop reads them and the seabed is kept as given:
        BOUNCE tabulates it (see
        :func:`~uacpy.models.bellhop._bounce_route.resolve_bounce_settings`)
        and the deck carries the table in its place."""
        if bounce_route(env, request, auto_bounce=self.auto_bounce) is None:
            if env.bottom.is_layered:
                kind = ('layered bottom (elastic)' if env.bottom.is_elastic
                        else 'layered bottom')
                warnings.warn(
                    f"{self.model_name}: env.bottom is a {kind}; "
                    f"auto_bounce=False → collapsing the layer stack to a "
                    f"halfspace via the model's collapse policy. The stack's "
                    f"interference structure is lost from the bottom "
                    f"reflection. Set auto_bounce=True (default) or call "
                    f"run_with_bounce() to keep the layers via a BOUNCE "
                    f"table.",
                    FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
                )
            return super()._project_environment(env, request=request)
        stand_in = env.copy()
        stand_in.bottom = Bottom.from_halfspace(
            BoundaryProperties(acoustic_type='vacuum'))
        projected = super()._project_environment(stand_in, request=request)
        projected.bottom = copy.deepcopy(env.bottom)
        return projected

    # ── stage 2: the carriers this engine cannot run ───────────────────

    def _check_knobs(self) -> None:
        """Refuse a knob no run could use
        (:func:`~uacpy.models.bellhop._checks.check_knobs`)."""
        check_knobs(
            beam_type=self.beam_type, beam_width_type=self.beam_width_type,
            beam_curvature=self.beam_curvature, component=self.component,
            n_beams=self.n_beams, grid_type=self.grid_type, launch_angles=self.launch_angles,
            ray_step=self.ray_step, z_box=self.z_box, r_box=self.r_box,
            n_freqs=self.n_freqs, bandwidth_factor=self.bandwidth_factor)

    def _validate_engine(self, env, source, receiver, run_mode, *,
                         request=None) -> None:
        """Stage 2, on the projected environment: a ``'precalc'`` boundary
        the deck would carry, a beam type whose
        influence routine cannot run the mode or index the receiver grid,
        an irregular grid of unequal depth and range lists, a source beam
        pattern narrower than the launch fan.
        Checked before the backend is chosen, so all three backends refuse
        alike: the ports implemented branches the Fortran lacks, which
        otherwise makes the answer depend on which binaries install.sh
        built."""
        reject_biological_edges_under_neighbour_interp(
            'Bellhop', env, self.interp_ssp)
        reject_precalc_boundary(env, include_bottom=bounce_route(
            env, request, auto_bounce=self.auto_bounce) is None)
        check_beam_type_supports_run_mode(run_mode, beam_type=self.beam_type)
        if run_mode != RunMode.RAYS:      # bellhop.f90:288 skips influence
            check_beam_type_supports_receiver_grid(
                receiver, beam_type=self.beam_type, grid_type=self.grid_type)
        reject_unequal_paired_grid(receiver, grid_type=self.grid_type)
        if source.beam_pattern is not None:
            check_beam_pattern_spans_the_fan(source.beam_pattern,
                                             launch_angles=self.launch_angles)

    # ── stage 3: the settings of the deck ──────────────────────────────

    def _resolve_engine_settings(self, env, source, receiver, settings, *,
                                 given_env, request=None) -> BellhopSettings:
        """Stage 3: every value the deck is written from, resolved once, the
        refusals of a deck Bellhop cannot run (a ray box inside the domain),
        the BOUNCE run of a routed seabed, and the notices of how the run
        will go; on BROADBAND where the band came from, on TIME_SERIES the
        pulse length the delay-and-sum convolves (both read off the call,
        ``request``)."""
        mode = settings.mode
        kw = request.kwargs if request is not None else {}
        band = band_origin(mode, source, kw, n_freqs=self.n_freqs,
                           bandwidth_factor=self.bandwidth_factor)
        n_pulse = pulse_samples(mode, kw)
        run_type = _RUN_MODE_TO_INFLUENCE_LETTER[mode]
        fc = carrier_frequency(source, mode)
        step = resolve_ray_step(env, ray_step=self.ray_step)
        z_box, z_box_origin, r_box, r_box_origin = resolve_ray_box(
            env, receiver, z_box=self.z_box, r_box=self.r_box)
        reject_ray_box_inside_the_domain(
            env, receiver, mode, z_box, r_box)

        notices = geometry_notices(
            env, source, receiver, mode, launch_angles=self.launch_angles,
            grid_type=self.grid_type, n_beams=self.n_beams,
            model_name=self.model_name, ray_step=step,
            memory_backed=work_dir_is_memory_backed(self.scratch_policy()))
        for notice in (message_notice(beam_type_run_mode_notice(mode,
                                                             beam_type=self.beam_type),
                                   ValidityWarning),
                       message_notice(line_source_sgb_notice(
                                       source, mode, beam_type=self.beam_type,
                                       backend=self._resolved_backend, model_name=self.model_name),
                                   ValidityWarning)):
            if notice is not None:
                notices.append(notice)
        bounce = bounce_origin = None
        knobs = _BounceKnobs()
        route = bounce_route(env, request, auto_bounce=self.auto_bounce)
        if route is not None:
            bounce_origin, knobs = route
            notices.extend(bounce_route_notices(
                env, mode, fc, bounce_origin, model_name=self.model_name))
            bounce, bounce_notices = resolve_bounce_settings(
                env, carrier_source(source, mode, fc), receiver,
                producer=self._bounce_producer(knobs),
                model_name=self.model_name)
            notices.extend(bounce_notices)
        interp_ssp, interp_notice = resolve_interp_ssp(
            env, interp_ssp=self.interp_ssp, log=self._log)
        if interp_notice is not None:
            notices.append(message_notice(interp_notice, FallbackWarning))
        deck_ranges, range_trim, pad_notice = pad_receiver_ranges(
            receiver, run_type, beam_type=self.beam_type,
            grid_type=self.grid_type, model_name=self.model_name)
        if pad_notice is not None:
            notices.append(message_notice(pad_notice, NumericsWarning))
        return BellhopSettings(
            backend=str(self._resolved_backend),
            executable=str(self._exe),
            run_type=run_type,
            center_frequency=fc,
            beam_type=self.beam_type,
            n_beams=self.n_beams,
            launch_angles=tuple(self.launch_angles),
            ray_step=float(step),
            ray_step_origin=ray_step_origin(step, ray_step=self.ray_step),
            z_box=float(z_box),
            z_box_origin=z_box_origin,
            r_box=float(r_box),
            r_box_origin=r_box_origin,
            interp_ssp=interp_ssp,
            deck_ranges=deck_ranges,
            range_trim=range_trim,
            band_origin=band,
            pulse_samples=n_pulse,
            bounce=bounce,
            bounce_origin=bounce_origin,
            bounce_c_low=knobs.c_low,
            bounce_c_high=knobs.c_high,
            bounce_rmax_m=knobs.rmax_m,
            notices=tuple(notices),
        )

    def _bounce_producer(self, knobs: _BounceKnobs):
        """:func:`~uacpy.models.bellhop._bounce_route.bounce_producer` with
        this model's verbosity, collapse policy, timeout and ``cleanup``."""
        return bounce_producer(knobs, verbose=self.verbose,
                               user_collapse=self._user_collapse,
                               timeout=self.timeout, cleanup=self.cleanup)

    # ── stage 4: BOUNCE (routed seabed), then the Bellhop deck ─────────

    def _n_launches(self, settings) -> int:
        """Two for a seabed routed through BOUNCE (the table, then the
        Bellhop run that reads it), one otherwise."""
        return 1 if settings.engine.bounce is None else 2

    def _prepare_launches(self, env, settings):
        """For a routed seabed, the BOUNCE producer and the environment it
        tabulates (:func:`~uacpy.models.bellhop._bounce_route.bounce_view`
        of ``env``, projected as Bounce projects it; its notices were said
        in stage 3)."""
        engine = settings.engine
        if engine.bounce is None:
            return None
        return prepare_bounce_launch(env, self._bounce_producer(_BounceKnobs(
            engine.bounce_c_low, engine.bounce_c_high, engine.bounce_rmax_m)))

    def _write_input(self, inputs: StageInputs) -> Path:
        """Stage 4. For a routed seabed, launch 0 is BOUNCE's own deck,
        written by :meth:`Bounce._write_input`. The Bellhop deck
        (``model.env``, plus ``model.sbp`` for a source beam pattern) is
        written from the settings, its seabed replaced by BOUNCE's ``.brc``
        table when routed."""
        engine = inputs.settings.engine
        if is_bounce_launch(inputs):
            self._log("Running BOUNCE to compute reflection coefficients...")
            return inputs.prepared.producer._write_producer_deck(
                bounce_inputs(inputs))
        env = inputs.env
        if engine.bounce is not None:
            self._log("Running Bellhop with BOUNCE reflection coefficients...")
            env = tabulated_seabed(env, inputs.earlier[0].brc_file)
        source = launch_source(inputs)
        env_file = inputs.work_dir / f'{_BASE_NAME}.env'
        self._log(f"Writing environment file: {env_file}")
        # Bellhop reads <base>.sbp when RunType position 3 is '*'.
        use_sbp = source.beam_pattern is not None
        if use_sbp:
            sbp_dest = env_file.with_suffix('.sbp')
            stage_source_beam_pattern(source.beam_pattern, sbp_dest)
            self._log(f"Wrote source beam pattern: {sbp_dest}")
        write_bellhop_env_file(
            filepath=env_file,
            env=env,
            source=source,
            receiver=deck_receiver(inputs.receiver, engine),
            run_type=engine.run_type,
            beam_type=engine.beam_type,
            # Letter lands in RunType(4:4), ReadEnvironmentBell.f90:398-
            # 406; spec.source_types keeps 'scaled' out of this deck.
            source_type=_SOURCE_TYPE_CODE[source.source_type],
            grid_type=self.grid_type,
            verbose=self.verbose,
            n_beams=engine.n_beams,
            launch_angles=engine.launch_angles,
            ray_step=engine.ray_step,
            z_box=engine.z_box,
            r_box=engine.r_box,
            source_beam_pattern=use_sbp,
            beam_width_type=self.beam_width_type,
            beam_curvature=self.beam_curvature,
            eps_multiplier=self.eps_multiplier,
            r_loop=self.r_loop,
            n_image=self.n_image,
            ib_win=self.ib_win,
            component=self.component,
            beam_shift=self.beam_shift,
            interp_ssp=engine.interp_ssp,
            interp_bathymetry=self.interp_bathymetry,
            interp_altimetry=self.interp_altimetry,
        )
        return env_file

    def _launch(self, inputs: StageInputs, deck: Path) -> None:
        """Stage 4: BOUNCE's own launch for launch 0 of a routed seabed;
        else the Bellhop binary on ``deck``, an ARRIVALS deck with the
        ports' ``-mem=`` budget sized from its grid.

        Bellhop reports most fatal errors in ``<base>.prt`` rather than on
        stderr. If the child exits non-zero, the tail of the .prt file (up to
        2000 chars) is appended to the raised ``ModelExecutionError`` so the
        diagnostic reaches the user instead of a blank stderr.

        The three mutually-exclusive outputs are cleared first: a pinned
        work_dir carries the previous run's ``.shd`` / ``.arr`` / ``.ray``,
        and this run writes only one of them.

        An auto-selected bellhopcuda that finds no CUDA device is launched
        again on the next installed binary
        (:meth:`_leave_cuda_without_a_device`).
        """
        if is_bounce_launch(inputs):
            inputs.prepared.producer._launch_producer(
                bounce_inputs(inputs), deck)
            return
        engine = inputs.settings.engine
        self._log("Running Bellhop...")
        memory_bytes = None
        if engine.run_type == 'A':
            memory_bytes = arrivals_memory_bytes(
                inputs.source, deck_receiver(inputs.receiver, engine),
                grid_type=self.grid_type)
        base_name, work_dir = deck.stem, inputs.work_dir

        def command():
            # Read at each launch: a move off bellhopcuda changes both.
            return build_command(self._exe, base_name, backend=self._resolved_backend,
                                 dimensionality=self.dimensionality,
                                 memory_bytes=memory_bytes)

        try:
            result = self._run_and_attach_prt(
                command(), work_dir, base_name,
                stale_outputs=_BELLHOP_OUTPUT_SUFFIXES,
            )
        except ModelExecutionError as exc:
            if not self._leave_cuda_without_a_device(exc):
                raise
            result = self._run_and_attach_prt(
                command(), work_dir, base_name,
                stale_outputs=_BELLHOP_OUTPUT_SUFFIXES,
            )
        if self._resolved_backend in ('cuda', 'cxx'):
            warn_on_engine_stdout_warnings(result.stdout or '',
                                           model_name=self.model_name,
                                           backend=self._resolved_backend)

    def _leave_cuda_without_a_device(self, exc) -> bool:
        """Move an auto-selected bellhopcuda that found no CUDA device onto
        the next installed binary, and say whether it did.

        Only ``backend=None`` moves, and only on the no-device signature
        (:data:`~uacpy.models.bellhop._backend._CUDA_NO_DEVICE`): a CUDA
        build installed on a host, job or container with no visible GPU is
        still a working install, and bellhopcxx beside it runs the same
        deck. An explicit ``backend='cuda'`` and every other failure
        raise as they are. The finding is kept for the process, warned
        about once, and the result carries the backend that ran.
        """
        if (self.backend is not None or self._resolved_backend != 'cuda'
                or not has_no_device_signature(exc)):
            return False
        first = record_cuda_without_a_device()
        self._exe = self._resolve_executable(
            self.executable, self._find_bellhop_executable)
        if first:
            warnings.warn(
                f"Bellhop: bellhopcuda found no CUDA device on this host "
                f"(cudaErrorNoDevice), so this and every later "
                f"auto-selected Bellhop in this process runs "
                f"{self._resolved_backend} ({self._exe}). Pass backend='cxx' to choose "
                f"it directly.",
                FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
        return True

    def _read_output(self, inputs: StageInputs, deck: Path):
        """Stage 4: BOUNCE's reflection table (its own stage 5, stamped with
        its settings, as its ``run`` returns it) for launch 0 of a routed
        seabed; else the one output the Bellhop deck's run type writes, as
        its io reader returns it
        (:func:`~uacpy.models.bellhop._output.read_bellhop_output`)."""
        if is_bounce_launch(inputs):
            return read_bounce_output(inputs, deck)
        return read_bellhop_output(
            inputs.work_dir, deck.stem, inputs.settings.engine.run_type,
            model_name=self.model_name, grid_type=self.grid_type,
            backend=self._resolved_backend, exe=self._exe)

    # ── stage 5 ─────────────────────────────────────────────────────────

    def _to_result(self, inputs: StageInputs, deck, raw):
        """Stage 5: the Bellhop output assembled in the package's
        conventions (:func:`~uacpy.models.bellhop._output.assemble`),
        every slab carrying the BOUNCE table of a routed seabed as
        ``components['bounce']``; on BROADBAND / TIME_SERIES, the
        synthesis built from those arrivals
        (:func:`~uacpy.models.bellhop._synthesis.synthesise`)."""
        engine = inputs.settings.engine
        decks = deck if isinstance(deck, list) else [deck]
        raws = raw if isinstance(raw, list) else [raw]
        result = assemble(
            inputs.env, launch_source(inputs),
            inputs.receiver, raws[-1], inputs.work_dir, decks[-1].stem,
            engine.run_type, trim=engine.range_trim,
            model_name=self.model_name, provenance=self.provenance,
            backend=self._resolved_backend, cleanup=self.cleanup,
            grid_type=self.grid_type, log=self._log)
        if engine.bounce is not None:
            # A multi-depth Source stacks the slabs and
            # ``ResultStack.components`` reads slab 0's, so the one table
            # (BOUNCE reads no source depth) goes on every slab.
            for slab in _slabs_of(result):
                slab._components = {**slab.components,
                                    'bounce': raws[0].result}
        if inputs.settings.mode in (RunMode.BROADBAND, RunMode.TIME_SERIES):
            self._stamp_run_settings(result, arrivals_run_settings(
                inputs.settings, inputs.source,
                output=self.outputs[RunMode.ARRIVALS]))
            return synthesise(
                inputs, result, model_name=self.model_name,
                provenance=self.provenance, backend=self._resolved_backend,
                grid_type=self.grid_type, log=self._log)
        return result

    def _broadband_band_knobs(self):
        """The band a single BROADBAND carrier expands with: this model's
        ``n_freqs`` / ``bandwidth_factor`` knobs."""
        return self.n_freqs, self.bandwidth_factor

    def run_with_bounce(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        *,
        run_mode: Optional[RunMode] = None,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        rmax_m: Optional[float] = None,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        output_duration: Optional[float] = None,
        t_start: Optional[float] = None,
    ) -> Union[Result, ResultStack]:
        """
        Run Bellhop using BOUNCE-generated reflection coefficients.

        Runs BOUNCE first to compute the reflection coefficient of the
        environment's seabed, then Bellhop on the resulting ``.brc`` table.
        This provides accurate handling of elastic/layered bottoms that
        Bellhop cannot model directly.

        It is :meth:`run` with the seabed routed through BOUNCE: the same
        argument checks, source-depth loop, source weights and
        ``result.run_settings``, whose ``engine.bounce`` holds the settings
        BOUNCE ran with. Both launches share the run's work directory.

        Parameters
        ----------
        env : Environment
            Ocean environment (bottom properties define the layer stack)
        source : Source
            Acoustic source
        receiver : Receiver
            Receiver array
        run_mode : RunMode, optional
            Bellhop mode for the second (field) pass; same values ``run()``
            takes. Keyword-only, like every argument after ``receiver``, so
            it cannot bind to a BOUNCE knob by position.
        c_low : float, optional
            Minimum phase velocity for the reflection table (m/s). ``None``
            (default) is resolved by :class:`~uacpy.Bounce` itself, as
            ``min(DEFAULT_C_MIN, min(env.ssp))`` — bounce.htm's "lowest speed
            in the problem" — which keeps the table's grazing wedge intact in
            cold or fresh water too; a larger value is refused.
        c_high : float, optional
            Maximum phase velocity for the reflection table (m/s). ``None``
            (default) is Bounce's ``DEFAULT_C_MAX_UNBOUNDED``, which zeroes
            BOUNCE's ``kMin`` and so covers grazing angles down to 0. AT's
            bounce.htm: CMax must be ~1e9 "for a full 90 degree
            calculation"; a finite value truncates the table at
            asin(c_water/CMax), and RefCoef.f90:144-149 then returns R = 0
            for every steeper ray.
        rmax_m : float, optional
            Maximum range for angular resolution (m). ``None`` (default) is
            resolved by Bounce: ``receiver.range_max``, the range the table
            is propagated to (bounce.htm: "RMax should be the maximum range
            to which you are propagating"), or its 10 km fallback when every
            receiver sits at r = 0 (the same fallback Bellhop's own ``r_box``
            takes).
        frequencies, source_waveform, sample_rate, output_duration, t_start : optional
            Keyword-only; passed to the field pass as :meth:`run` takes them.

        Returns
        -------
        result : Result or ResultStack
            What :meth:`run` returns for the same call: a ``ResultStack``
            (one slab per source depth) for a multi-depth ``Source``, one of
            the typed :mod:`uacpy.core.results` subclasses otherwise. Every
            slab carries BOUNCE's :class:`ReflectionCoefficient` as
            ``components['bounce']``.
        """
        call = self._check_call(
            env, source, receiver, run_mode, frequencies=frequencies,
            source_waveform=source_waveform, sample_rate=sample_rate,
            output_duration=output_duration, t_start=t_start)
        return self._run_call(env, source, receiver, dataclasses.replace(
            call, engine_request=_BounceKnobs(
                c_low=c_low, c_high=c_high, rmax_m=rmax_m)))
