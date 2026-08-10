"""
Kraken Normal Mode Suite - one model, backend dispatcher

A single :class:`Kraken` wraps the AT Kraken pipeline, mirroring
``RAM``'s ``backend=`` convention:

- ``backend=`` selects the modes binary — ``'kraken'`` (real arithmetic)
  or ``'krakenc'`` (complex: elastic media / attenuation / leaky modes);
  ``None`` (default) auto-picks krakenc when the env carries shear/leaky.
- ``field.exe`` runs only when the requested run mode produces a field:
  ``compute_modes`` stops after the modes binary; ``compute_tl`` /
  ``compute_transfer_function`` / ``compute_time_series`` chain field.exe.
- Range-dependence (bathy / SSP) is handled natively for field modes via
  ``field.exe`` adiabatic / coupled modes (``mode_coupling=``); the
  range-independent MODES path samples the r=0 profile.

Note
----
The Acoustics Toolbox also ships ``krakel.exe`` (true elastic normal
modes with shear support using an FEM discretisation). It is bundled in
``uacpy/uacpy/bin/oalib/`` but NOT wrapped by uacpy at this time. Users
who need elastic modes can either:

* drive ``Kraken(backend='krakenc')`` (which handles elastic half-spaces
  via complex wavenumbers), or
* invoke ``krakel.exe`` manually with a Kraken-format .env file.

Usage
-----
```python
from uacpy.models import Kraken, RunMode

kraken = Kraken()
modes = kraken.compute_modes(env, source, receiver)   # Modes (k, phi)
tl    = kraken.compute_tl(env, source, receiver)       # Field (TL)

# Complex modes for elastic bottom (auto, or force backend)
modes = Kraken(backend='krakenc').compute_modes(env_elastic, source, receiver)

# Range-dependent field via coupled modes
kraken = Kraken(mode_coupling='coupled', n_segments=10)
tl = kraken.compute_tl(env_rd, source, receiver)
```
"""

import os
import re
import warnings
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union

from uacpy.models.base import (
    PropagationModel, RunMode, ModelSpec, USER_FRAME_SKIP,
)
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import Result, Modes, Field
from uacpy.core.constants import (
    parse_boundary_type,
    DEFAULT_SOUND_SPEED,
    C_LOW_FACTOR_KRAKEN,
    PRESSURE_FLOOR,
)
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, ModelExecutionError,
    UnsupportedFeatureError,
)
from uacpy.io.oalib_writer import (
    write_multi_profile_env, write_fieldflp, write_kraken_env_file,
    resolve_phase_speed_bounds, plan_multi_profile_media,
    at_mesh_floor, reject_coarse_at_mesh,
)
from uacpy.io.oalib_reader import read_shd_file, read_shd_bin, read_prt
from uacpy.io.refl_io import stage_source_beam_pattern
from uacpy.models._segmentation import segment_environment_by_range


# Source geometry -> field.exe Opt(1:1), per field.f90:70-79.
_SOURCE_TYPE_CODE = {'point': 'R', 'line': 'X', 'scaled': 'S'}

# Cleared before each launch so a pinned work_dir cannot return an earlier
# run's output: the modes binary writes <base>.mod, field.exe writes <base>.shd.
_KRAKEN_MODES_OUTPUTS = ('.mod',)
_KRAKEN_FIELD_OUTPUTS = ('.shd',)

#: ``field.f90:44`` hard-codes its log name instead of deriving it from the
#: file root, so field.exe's diagnostics land here rather than in
#: ``<base_name>.prt`` and the shared post-run fatal scan never sees them.
_FIELD_PRT_ROOT = 'field'

#: ``KrakenField/field.f90:24`` declares ``MaxNfreq = 1000``, allocates
#: ``freqVec( MaxNfreq )`` (:164) and runs ``FreqLoop`` to that bound (:168);
#: ``KrakenField/ReadModes.f90:8,35`` repeats the constant and :187 reads
#: ``freqVec( 1 : Nfreq )`` with the ``Nfreq`` taken from the mode-file header.
#: The solver has no such cap (``misc/SourceReceiverPositions.f90:58``
#: allocates ``freqVec`` dynamically), so a longer grid is solved and written
#: happily and only overruns when field.exe reads the header back.
_FIELD_MAX_NFREQ = 1000

#: ``acoustic_type`` values that put a tabulated reflection coefficient on a
#: boundary — AT letters ``'F'`` (``.brc`` / ``.trc``) and ``'P'`` (``.irc``).
#: Neither is usable on real ``kraken.exe``: ``Kraken/kraken.f90:47-48`` stops
#: outright on a bottom ``'F'`` or a top ``'P'``, and the two mirror cases pass
#: that guard only to be thrown away — every mode-finding call passes
#: ``ComplexFlag = .FALSE.`` (``Kraken/kraken.f90:447,451,613,654,785-786,
#: 793-794``) and ``Kraken/BCImpedanceMod.f90:113-116,121-125`` then replaces
#: the tabulated impedance with a rigid boundary (``f = 0, g = 1``).
#: ``krakenc.exe`` honours both (``Kraken/BCImpedancecMod.f90:88-106``).
_REFLECTION_TABLE_TYPES = ('file', 'precalc')


class Kraken(PropagationModel):
    """
    Kraken - Normal modes + field computation (multi-backend).

    One model over the AT Kraken pipeline, mirroring ``RAM``'s
    ``backend=`` dispatcher. The modes binary is selected by ``backend=``
    (``'kraken'`` real / ``'krakenc'`` complex, or ``None`` to auto-pick
    krakenc for elastic / leaky media). ``field.exe`` runs only when the
    requested run mode produces a field:

    * ``RunMode.MODES`` (``compute_modes``) → runs the modes binary only,
      returns a :class:`Modes` result (``k``, ``phi``).
    * ``COHERENT_TL`` / ``BROADBAND`` / ``TIME_SERIES``
      (``compute_tl`` / ``compute_transfer_function`` / ``compute_time_series``)
      → modes binary → ``field.exe`` → ``.shd``, returns a :class:`Field`.
    * ``INCOHERENT_TL`` → same pipeline with ``field.exe`` ``Opt(4:4)='I'``,
      which drops the range phase (``ik = REAL(ik)``) and returns
      ``SQRT(SUM(z**2))`` over the per-mode contributions ``z``
      (``EvaluateMod.f90:43,66``). That is the energy sum
      ``SQRT(SUM(|z|**2))`` only while the mode functions are real, i.e.
      on the ``kraken.exe`` path; with ``backend='krakenc'`` the complex
      ``phi``/``k`` leave cross-mode phase in the square and the run
      warns. The result is a real dB :class:`Field` (``kind='tl'``) — it
      carries no phase, so it cannot feed a time-series synthesis.
      ``mode_coupling='coupled'`` has no incoherent path in ``field.exe``
      and is rejected.

    Note that ``run()`` defaults to a field mode (``COHERENT_TL``, or
    ``BROADBAND`` when a multi-frequency ``frequencies=`` is given) — use
    ``compute_modes(...)`` (or ``run(run_mode=RunMode.MODES)``) for modes.

    Supports range-independent and range-dependent environments via
    adiabatic or coupled mode theory (delegated to AT's field.exe); the
    MODES path is range-independent and samples the r=0 profile of any RD
    environment (with a warning).

    Note
    ----
    Range-dependent bathymetry is supported via environment segmentation
    (multi-profile .env + field.exe coupled/adiabatic modes). Sea-surface
    altimetry (non-flat sea surface) is NOT supported; Bellhop is the
    only uacpy model that supports altimetry.

    Parameters
    ----------
    mode_coupling : str, optional
        ``'adiabatic'`` (default) or ``'coupled'``. Controls how
        ``field.exe`` handles range-dependent mode transitions.
    n_segments : int, optional
        Range segments for RD scenarios. Default ``None`` lets
        :func:`_segmentation.segment_environment_by_range` pick segment
        edges from the union of bathymetry / RD-SSP / RD-bottom
        change-points, inserting intermediates wherever the gap
        exceeds ``max_segment_length`` (2 km). Pass an explicit int
        to override with a uniform linspace decomposition.
    mode_points_per_meter : float, optional
        Mode-depth grid density. Default ``1.5`` pts/m.
    executable, field_executable : Path, optional
        ``kraken.exe`` and ``field.exe`` paths. Auto-detected if ``None``.
    backend : str, optional
        Force the modes binary: ``'kraken'`` or ``'krakenc'``. ``None``
        (default) auto-selects — ``krakenc.exe`` for elastic media / leaky
        modes (complex eigenvalues) and for a tabulated reflection
        coefficient (``acoustic_type='file'`` / ``'precalc'``, which
        ``kraken.exe`` either refuses or silently replaces with a rigid
        boundary), ``kraken.exe`` otherwise. Forcing ``'kraken'`` on either
        raises ``ConfigurationError``.
    c_low : float, optional
        Lower phase speed limit (m/s). None ⇒ 0.0 (``C_LOW_FACTOR_KRAKEN``),
        which makes KRAKEN compute cLow automatically — the modal-solver
        default. A positive c_low skips slower modes and excludes interfacial
        (Scholte / Stoneley) modes; set it to the minimum p-wave speed if KRAKEN
        fails to converge on those. (The 0.95·min-SSP rule is the Scooter/SPARC
        wavenumber-integration default, not Kraken's.) Must be non-negative and
        strictly less than ``c_high``.
    c_high : float, optional
        Upper phase speed limit (m/s). None = auto (1.05 * max of SSP and bottom speed).
        Must be strictly greater than ``c_low``.
    n_mesh : int, optional
        Total number of mesh points PER MEDIUM used by the finite-difference
        mode solver (AT's ``NMESH`` column on the SSP mesh line). 0 = let
        Kraken pick automatically from frequency / wavelength. Default: 0.
        Note: this is NOT a "points per wavelength" density — it is a total
        point count per medium.
    interp_ssp : str, optional
        SSP connection scheme written into ``TopOpt(1)``. ``None``
        (default) resolves to ``'linear'`` (C-linear). Explicit values:
        ``'linear'``, ``'n2linear'``, ``'pchip'``, ``'cubic'`` /
        ``'spline'``. ``'quad'`` is Bellhop-only and is
        rejected with a :class:`ConfigurationError`.
    n_modes : int, optional
        Cap on the number of modes field.exe propagates (FLP ``MLimit``).
        ``None`` (default) uses every mode the solver found. Also truncates a
        :class:`Modes` result to the first ``n_modes``, matching what the
        field evaluation used; :meth:`Modes.first_n` slices an existing result
        after the fact.
    leaky_modes : bool, optional
        If True, override ``c_high`` to 1e9 so the modes binary attempts leaky
        modes — modes whose phase speed exceeds the half-space S- or P-wave
        speed, so they radiate into the half-space instead of being trapped.
        This forces ``backend='krakenc'``: ``doc/kraken.htm:650-657`` has KRAKEN
        "reduce CHIGH so that only trapped (non-leaky) modes are computed" while
        "KRAKENC will attempt to compute leaky modes if CHIGH exceeds the phase
        velocity of either the S-wave or P-wave speed in the half-space".
        Default: False.
    top_reflection_file : Path, optional
        A ``.trc`` top-reflection-coefficient table. Overrides the surface
        boundary condition to ``'F'`` and is staged next to the ``.env``.
    rmax_m : float, optional
        ``RMax`` written into the ``.env`` (m here; the writer converts to
        the km the deck expects). It is the mode solver's
        mesh-convergence tolerance, scaled to range, and nothing else:
        ``kraken.f90:80`` / ``krakenc.f90:82`` refine the mesh until
        ``Error·1000·RMax < 1``, where ``Error`` is the change in the
        Richardson-extrapolated eigenvalue between two successive meshes
        and ``1000·RMax`` is RMax in metres. A **larger** RMax is a
        **tighter** tolerance (finer mesh, slower, more accurate).
        ``None`` (default) derives it from the outermost receiver range:
        ×1.05 for narrowband, ×3 for a broadband sweep.
    mode_depth_grid : ndarray, optional
        Explicit depth grid (m) the ``.mod`` eigenfunctions are sampled
        on by ``compute_modes``. ``None`` (default) builds
        ``max(100, total_media_depth × mode_points_per_meter)`` points
        spanning water + sediment.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    ``field.exe`` ``Opt(3)`` only accepts ``'*'``, ``'O'``, or ``' '``
    (``field.f90:83-90``); anything else raises FATAL ERROR. Purely
    elastic component selection (H/V/T/N) is not reachable through
    ``field.exe`` — an upstream Fortran limitation.

    **Auto-route to krakenc.exe** when ``env`` carries shear (delegates the
    modes step to the complex-arithmetic binary).

    Defaults auto-derived at ``run()`` time (override only when tuning):

    - ``c_low=None`` → ``0.0`` (``C_LOW_FACTOR_KRAKEN``; KRAKEN computes cLow automatically)
    - ``c_high=None`` → ``max(max(env.ssp), env.bottom.sound_speed) × 1.05``
    - ``n_mesh=0`` → Kraken picks mesh from frequency / wavelength.
    - TopOpt position 4 reads ``env.absorption`` (``Thorp`` / ``FrancoisGarrison``
      / ``Biological`` / ``ConstantAbsorption`` / ``None``).

    With ``verbose='info'`` the resolved ``c_low`` / ``c_high`` are logged.

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    RD bathymetry and RD-SSP are honoured natively (segments). Per-model
    default: ``'bottom_range': 'median'`` (median over RD halfspace
    samples); the layer stack is kept since Kraken consumes layered
    seabed columns natively, so a range-dependent layered bottom collapses
    to the median range with its layers intact.

    Examples
    --------
    >>> kraken = Kraken()
    >>> tl = kraken.compute_tl(env, source, receiver)        # field via field.exe

    >>> modes = kraken.compute_modes(env, source)            # modes only (no receiver)

    >>> # Elastic bottom → complex modes (auto, or force backend='krakenc')
    >>> modes = Kraken(backend='krakenc').compute_modes(env_elastic, source)

    >>> # Range-dependent with coupled modes
    >>> kraken = Kraken(mode_coupling='coupled', n_segments=20)
    >>> tl = kraken.compute_tl(env_rd, source, receiver)
    """

    def _max_receiver_depth(self, env) -> float:
        # The Kraken family meshes through fluid sediment layers, so it
        # resolves receivers down to the deepest interface, not just the
        # seafloor. Equals env.depth when there are no sediment layers.
        return self._total_media_depth(env)

    def _c_low_for(self, env) -> float:
        """Resolved KRAKEN cLow for ``env``.

        An explicit ``c_low`` always wins. Otherwise ``0.0`` hands the choice
        to KRAKEN, which is right for a fluid environment.

        It is *not* right once any medium carries shear. ``krakenc.f90:189``
        folds the shear speeds into ``cMin``, then ``:228-230`` applies
        ``IF ( ElasticFlag ) cMin = 0.85 * cMin`` and
        ``cLow = MAX( cLow, 0.99 * cMin )`` — so with cLow=0 the search floor
        lands near 0.84x the slowest *shear* speed and the solver returns
        interfacial (Scholte / Stoneley) modes instead of the waterborne
        field, for a TL of several hundred dB. KRAKEN's own documentation
        prescribes the minimum compressional speed here, which is what
        ``_modes_error_message`` already tells the user on the failure path.

        Only the bottom is inspected: ``_reject_elastic_surface`` runs ahead
        of every path that reaches this, so an elastic surface never arrives.
        """
        if self.c_low is not None:
            return float(self.c_low)
        if not env.has_elastic_bottom:
            return C_LOW_FACTOR_KRAKEN * DEFAULT_SOUND_SPEED
        speeds = [float(np.min(env.ssp.to_pairs()[:, 1]))]
        speeds.extend(env.bottom.all_sound_speeds())
        return min(speeds)

    def _validate_phase_speed_limits(self):
        """Check 0 <= c_low < c_high when either is explicitly set."""
        cl = self.c_low
        ch = self.c_high
        if cl is not None and cl < 0:
            raise ConfigurationError(
                f"c_low must be >= 0, got {cl}"
            )
        if cl is not None and ch is not None and ch <= cl:
            raise ConfigurationError(
                f"c_high ({ch}) must be strictly greater than c_low ({cl})"
            )

    def _reject_coarse_mesh(self, env, frequency: float) -> None:
        """Reject a pinned ``n_mesh`` the AT reader will call too coarse."""
        reject_coarse_at_mesh('Kraken', self.n_mesh, env, frequency)

    def _check_kraken_ssp_type(self):
        """Reject SSP interpolation choices kraken does not implement.

        Per AT ``misc/sspMod.f90:61-89`` kraken accepts codes A (analytic),
        N (N^2-linear), C (C-linear), P (PCHIP), S (spline). The 'Q'
        quadrilateral code is Bellhop-only (see RangeDepSSPFile.htm).

        Default ``self.interp_ssp=None`` resolves to 'linear' for
        Kraken's env (auto-quad only applies to Bellhop), so the
        rejection only fires on explicit 'quad'.
        """
        if self.interp_ssp is None:
            return
        if str(self.interp_ssp).lower() in ('q', 'quad', 'quadratic'):
            raise ConfigurationError(
                "Kraken does not support the 'quad' "
                "SSP interpolation — it is Bellhop-only. Pick one of "
                "'linear' (C-linear), 'n2linear', 'pchip', 'cubic' / "
                "'spline'."
            )

    def _build_modes_field(self, modes, n_modes, source, *, backend_exe=None,
                           bounds=None):
        """Wrap a modes-reader payload as a :class:`Modes` Result.

        Returns the full mode set the reader produced; callers cap the
        count via :meth:`Modes.first_n` if they passed an ``n_modes``
        request. ``backend_exe`` records which modes binary ran
        (kraken.exe vs krakenc.exe); defaults to the resolved kraken.exe.
        ``bounds`` is the resolved ``{'c_low', 'c_high', 'rmax'}`` the deck
        was written with.
        """
        k_arr = modes.get('k', np.array([]))
        phi_arr = modes.get('phi', np.array([]))
        z_arr = modes.get('z', np.array([]))

        exe = backend_exe or self._exe
        result = Modes(
            k=k_arr,
            phi=phi_arr,
            depths=z_arr,
            **self._result_kwargs(
                source,
                backend=Path(exe).stem if exe else self.model_name.lower(),
                frequencies=float(source.frequencies[0]),
                n_modes_requested=n_modes,
                leaky_modes=self.leaky_modes,
                **(bounds or {}),
            ),
        )
        if n_modes is not None:
            result = result.first_n(int(n_modes))
        return result

    @staticmethod
    def _compute_rmax_m(
        receiver, fallback_m: float = 100_000.0, *, multiplier: float = 1.05,
    ) -> float:
        """Derive the mode solver's RMax (m) from the receiver ranges.

        RMax scales the Richardson mesh-convergence test
        (``kraken.f90:80``: ``Error·1000·RMax < 1``), so it should be at
        least the longest range the modes will be propagated to —
        the outermost receiver. ``multiplier`` adds margin on top;
        raising it only tightens the tolerance (finer mesh, longer
        solve), never loosens it. Falls back to ``fallback_m`` if the
        receiver has no range vector.
        """
        if receiver is None:
            return float(fallback_m)
        ranges = getattr(receiver, 'ranges', None)
        if ranges is None or len(np.atleast_1d(ranges)) == 0:
            return float(fallback_m)
        rmax_m = float(np.max(np.asarray(ranges, dtype=float)))
        if rmax_m <= 0:
            return float(fallback_m)
        return rmax_m * float(multiplier)

    def _write_kraken_env(
        self,
        filepath,
        env,
        source,
        *,
        receiver_obj: Optional[Receiver] = None,
        receiver_depths=(100.0,),
        frequencies: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Write Kraken environment file using shared ATEnvWriter

        Kraken has additional sections beyond the standard ENV format:
        - Phase speed limits (cLow, cHigh)
        - Maximum range (RMax)
        - Optional broadband frequency vector (TopOpt(6)='B')

        Returns the resolved ``{'c_low', 'c_high', 'rmax'}`` (m/s, m) the
        deck was written with, for the caller to stamp onto its result.
        """
        # Reject 'quad' SSP interp (Bellhop-only)
        self._check_kraken_ssp_type()
        # Re-validate in case caller mutated attributes after __init__
        self._validate_phase_speed_limits()
        self._reject_coarse_mesh(env, float(source.frequencies[0]))

        from uacpy.io.oalib_writer import resolve_ssp_topopt
        ssp_topopt = resolve_ssp_topopt(env, self.interp_ssp)
        surface_type = parse_boundary_type(env.surface.acoustic_type)
        bottom_acoustic_type = env.bottom.halfspace_at(range=0.0).acoustic_type
        bottom_type = parse_boundary_type(bottom_acoustic_type)

        # RMax sets the mesh-convergence tolerance (see _compute_rmax_m):
        # 5 % past the outermost receiver for narrowband, ×3 for a broadband
        # sweep, which solves every frequency off one mesh sequence and so
        # gets the tighter tolerance as accuracy margin.
        is_broadband = frequencies is not None and len(np.atleast_1d(frequencies)) > 1
        rmax_m = (
            self.rmax_m
            if self.rmax_m is not None
            else self._compute_rmax_m(
                receiver_obj, fallback_m=100_000.0,
                multiplier=3.0 if is_broadband else 1.05,
            )
        )

        cl, ch = resolve_phase_speed_bounds(env, self._c_low_for(env), self.c_high)
        if self.c_high is None:
            self._log(
                f"c_high auto-derived from env.ssp + bottom = {ch:.1f} m/s "
                f"(c_low = {cl:.1f} m/s)"
            )

        write_kraken_env_file(
            filepath, env, source,
            receiver_obj if receiver_obj is not None else receiver_depths,
            ssp_topopt=ssp_topopt,
            surface_type=surface_type,
            bottom_type=bottom_type,
            frequencies=frequencies,
            n_mesh=self.n_mesh,
            rmax_m=rmax_m,
            c_low=cl, c_high=ch,
        )
        return {'c_low': cl, 'c_high': ch, 'rmax': float(rmax_m)}

    def _reject_elastic_over_fluid_halfspace(self, env: Environment) -> None:
        """krakenc.exe spins forever in setup on a solid-over-liquid bottom
        interface: a layered column with an elastic layer (shear_speed > 0)
        terminated by a fluid halfspace (shear_speed == 0). Reject it up front
        so the caller gets a fast typed error instead of a ``timeout``-long
        hang. Elastic half-spaces and elastic-over-elastic stacks are fine."""
        if not env.bottom.is_layered:
            return
        col = env.bottom.at(range=0)
        has_elastic_layer = any(
            (getattr(layer, 'shear_speed', 0) or 0) > 0 for layer in col.layers
        )
        halfspace_fluid = (getattr(col.halfspace, 'shear_speed', 0) or 0) == 0
        if has_elastic_layer and halfspace_fluid:
            raise UnsupportedFeatureError(
                self.model_name,
                "an elastic sediment layer over a fluid halfspace "
                "(solid-over-liquid bottom interface) — krakenc.exe does not "
                "converge on it",
                alternatives=['Scooter', 'OAST'],
            )

    def _reject_elastic_surface(self, env: Environment) -> None:
        """krakenc.exe heap-corrupts (``free(): invalid pointer``) computing the
        interfacial (Scholte) modes of an elastic *top* half-space — a
        solid-over-liquid surface such as an ice canopy. (An elastic *bottom*
        half-space is fine.) Reject it up front with a clear typed error rather
        than the opaque SIGABRT. Checks the surface as the writer will see it:
        a range-dependent surface is first reduced by the configured
        ``collapse['surface']`` method (Kraken carries a single global top)."""
        surface = env.surface
        if surface.is_range_dependent:
            surface = surface.collapse(self._collapse['surface'])
        if surface.is_elastic:
            raise UnsupportedFeatureError(
                self.model_name,
                "an elastic (shear) sea-surface half-space such as an ice "
                "canopy — krakenc.exe aborts on the interfacial modes of a "
                "solid-over-liquid top interface",
                alternatives=['Bellhop'],
            )

    def _run_kraken_executable(self, base_name: str, work_dir: Path, exe=None):
        """Execute the modes binary (``exe`` selects kraken.exe vs krakenc.exe;
        defaults to the resolved kraken.exe) via the shared binary-launch helper."""
        self._run_and_attach_prt(
            [str(exe or self._exe), base_name], work_dir, base_name,
            stale_outputs=_KRAKEN_MODES_OUTPUTS)

    def _compute_modes_impl(self, env, source, n_modes):
        """Solve the modes on a depth grid dense enough to carry mode shapes.

        The ``.mod`` eigenfunctions are sampled only where the ``.env`` asks
        for receivers, so the grid handed to the binary decides how much of
        each mode shape survives — a sparse one leaves too few samples per
        mode to plot or to reconstruct a field from. The grid spans the water
        column plus any configured sediment depth: ``self.mode_depth_grid``
        verbatim when pinned, else ``max(100, total_depth *
        mode_points_per_meter)`` points linearly spaced from 0 to the total
        media depth.

        Parameters
        ----------
        env, source, n_modes : see PropagationModel.compute_modes
        """
        from uacpy.core.receiver import Receiver as _Receiver

        if self.mode_depth_grid is not None:
            mode_depths = self.mode_depth_grid
        else:
            total_depth = self._total_media_depth(env)
            ppm = float(getattr(self, 'mode_points_per_meter', 0.0) or 0.0)
            n_pts = max(100, int(round(float(total_depth) * ppm)))
            mode_depths = np.linspace(0.0, float(total_depth), n_pts)

        dense_receiver = _Receiver(depths=mode_depths, ranges=[0.0])
        # ``n_modes`` is constructor state; a per-call cap is applied by
        # running a copy so ``run()`` keeps the fixed model-wide signature.
        model = self if n_modes is None else self.copy(n_modes=int(n_modes))
        return model.run(env, source, dense_receiver, run_mode=RunMode.MODES)

    def _read_modes_file(self, filepath: Path) -> Dict:
        """Read a Kraken ``.mod`` file using the binary reader."""
        from uacpy.io.modes_reader import read_modes_bin

        # read_modes_bin expects the filename without extension and appends
        # its own ('.mod'); strip '.mod' before handing it over.
        filepath_str = str(filepath)
        if filepath_str.endswith('.mod'):
            basename = filepath_str[:-4]
        else:
            basename = filepath_str

        mod_file = basename + '.mod'

        # A .mod with no bytes at all means the binary died before it opened
        # the file — no header to read, so there is nothing to diagnose from.
        if os.path.exists(mod_file) and os.path.getsize(mod_file) == 0:
            raise ModelExecutionError(
                self.model_name, return_code=0, stdout=None,
                stderr=self._modes_error_message(basename),
            )

        # The MODES path writes a single-frequency ``.mod``; broadband goes
        # through ``field.exe`` instead. ``freq=0.0`` selects the only bin
        # present; ``read_modes_bin`` would otherwise fall back to
        # closest-frequency matching.
        try:
            modes_data = read_modes_bin(basename, frequency=0.0)
        except IndexError as e:
            raise ModelExecutionError(
                self.model_name, return_code=0, stdout=None,
                stderr=self._modes_error_message(basename, original_error=e),
            ) from e

        # "No modes for given phase speed interval" is an ERROUT that still
        # leaves a well-formed file behind: Kraken/kraken.f90:947-961 writes the
        # header and an M = 0 record before stopping, so the .mod is a normal
        # 4*LRecordLength*7 bytes and only the mode count reports the state.
        if int(modes_data.get('M', 0)) == 0:
            raise ModelExecutionError(
                self.model_name, return_code=0, stdout=None,
                stderr=self._modes_error_message(basename),
            )

        return modes_data

    @staticmethod
    def _modes_error_message(basename, original_error=None):
        """Build error message for invalid mode files, checking .prt for clues.

        Only suggest Kraken when the PRT evidences an actual elastic
        configuration (acousto-elastic boundary or a non-zero shear speed
        in the halfspace summary) or when kraken reports it couldn't find
        modes at cLow — not on any stray 'elastic' token in the PRT.
        """
        prt_file = basename + '.prt'
        error_msg = "Kraken did not produce valid modes. "
        prt_content = read_prt(prt_file)
        if prt_content is not None:
            # 1. True "acousto-elastic" mention (used in AT PRT for elastic HS)
            has_acousto_elastic = bool(
                re.search(r'acousto[-\s]*elastic', prt_content, re.IGNORECASE)
            )

            # 2. Non-zero shear speed anywhere in the halfspace summary.
            #    AT prints lines like "Shear speed = <value>" and
            #    "Bot. Shear speed  = <value>". A non-zero value means
            #    the elastic code path is engaged.
            has_nonzero_shear = False
            for m in re.finditer(
                r'[Ss]hear\s*speed\s*=?\s*([0-9.+\-eE]+)',
                prt_content,
            ):
                try:
                    if abs(float(m.group(1))) > 0.0:
                        has_nonzero_shear = True
                        break
                except ValueError:
                    pass

            # 3. Slow/failed root-finding on interfacial (Scholte/Stoneley)
            #    modes. misc/RootFinderSecantMod.f90:80,136 sets the message;
            #    Kraken/kraken.f90:359,407 and Kraken/krakenc.f90:388 echo it
            #    into the .prt behind their own 'Warning in KRAKEN[C] -
            #    RootFinderSecant' banner. kraken.htm's remedy is to raise cLow
            #    to the minimum p-wave speed so those modes are skipped.
            secant_failure = bool(
                re.search(r'converge\s+in\s+RootFinderSecant',
                          prt_content, re.IGNORECASE)
            )

            # 4. The empty-spectrum ERROUT (Kraken/kraken.f90:962). This is a
            #    physical statement about [cLow, cHigh], not a solver failure,
            #    and it names its own remedy — so it is tested before the
            #    elastic markers, which the bottom's own 'ACOUSTO-ELASTIC
            #    half-space' echo sets on any half-space seabed.
            empty_spectrum = (
                'No modes for given phase speed interval' in prt_content
            )

            if empty_spectrum:
                error_msg += (
                    "Kraken found no mode with a phase speed inside "
                    "[c_low, c_high] at this frequency. Widen the window "
                    "(lower c_low, raise c_high, or leaky_modes=True), or "
                    "raise the frequency above the waveguide's modal cutoff — "
                    "below it the field is carried by the continuous spectrum, "
                    "which needs a wavenumber-integration model (Scooter)."
                )
            elif secant_failure:
                error_msg += (
                    "Kraken reported 'Failure to converge in "
                    "RootFinderSecant': the root finder is converging slowly "
                    "to interfacial (Scholte / Stoneley) modes. Set c_low to "
                    "the minimum p-wave speed in the problem to exclude those "
                    "modes (kraken.htm, Phase Speed Limits), or use "
                    "Kraken(backend='krakenc')."
                )
            elif has_acousto_elastic or has_nonzero_shear:
                error_msg += (
                    "Kraken (real arithmetic) failed on an acousto-elastic "
                    "bottom (non-zero shear speed). Try "
                    "Kraken(backend='krakenc') (complex arithmetic), which "
                    "handles shear and leaky modes. Alternatives: Bellhop, "
                    "RAM, Scooter, OAST."
                )
            else:
                error_msg += f"Check the .prt file for details: {prt_file}"
                if original_error:
                    error_msg += f". Original error: {original_error}"
        elif original_error:
            error_msg += f"Original error: {original_error}"
        return error_msg

    # Declarative metadata (see PropagationModel / ModelSpec). Kraken: normal
    # modes. Segments RD-bathymetry / RD-SSP natively; the bottom (and the RDLB
    # axis) and a range-dependent *surface* still collapse — the RD .env carries
    # one global top/bottom boundary, only the SSP varies per range. Honours
    # layered + elastic bottom. Median across range / mean SSP represent the
    # per-segment profile.
    spec = ModelSpec(
        modes=(
            RunMode.MODES, RunMode.COHERENT_TL, RunMode.INCOHERENT_TL,
            RunMode.BROADBAND, RunMode.TIME_SERIES,
        ),
        supports={
            'range_dependent_bathymetry',
            'range_dependent_ssp',
            'layered_bottom',
            'elastic_media',
            'source_beam_pattern',
            'rough_surface',
            'rough_bottom',
        },
        source_types=frozenset({'point', 'line', 'scaled'}),
        collapse={'ssp': 'mean', 'bottom_range': 'median'},
    )
    source = 'acoustics_toolbox'

    # Below the waveguide's modal cutoff KRAKEN funnels through ERROUT, but
    # "no trapped modes in this phase-speed interval" is a physical answer
    # this model surfaces itself — as an all-NaN field plus a warning from
    # ``_compute_field_via_exe``, a zero-fill from the broadband floor search,
    # or a typed error from ``compute_modes`` — not a run failure.
    _BENIGN_FORTRAN_FATALS: tuple = (
        'No modes for given phase speed interval',
    )

    def __init__(
        self,
        mode_points_per_meter: float = 1.5,
        mode_coupling: str = 'adiabatic',
        n_segments: Optional[int] = None,
        executable: Optional[Path] = None,
        field_executable: Optional[Path] = None,
        backend: Optional[str] = None,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        n_mesh: int = 0,
        n_modes: Optional[int] = None,
        interp_ssp: Optional[str] = None,
        leaky_modes: bool = False,
        top_reflection_file: Optional[Path] = None,
        rmax_m: Optional[float] = None,
        mode_depth_grid: Optional[np.ndarray] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = 600.0,
        collapse: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )
        # --- modal-solver knobs ---
        self.interp_ssp = interp_ssp
        # c_low default 0.0 (C_LOW_FACTOR_KRAKEN) → KRAKEN computes cLow
        # automatically; a positive c_low skips slower/interfacial (Scholte /
        # Stoneley) modes. Stored raw (None preserved) so repr/copy stay clean;
        # resolved via ``_c_low_for`` at write time.
        self.c_low = None if c_low is None else float(c_low)
        self.c_high = c_high
        self.n_mesh = n_mesh
        if n_modes is not None and int(n_modes) < 1:
            raise ConfigurationError(
                f"Kraken(n_modes={n_modes}): the mode cap must be >= 1. "
                "field.exe treats MLimit <= 0 as zero propagating modes "
                "(field.f90:68,185), which returns an empty field.")
        self.n_modes = None if n_modes is None else int(n_modes)
        self.leaky_modes = leaky_modes
        self.top_reflection_file = (
            Path(top_reflection_file) if top_reflection_file is not None else None
        )
        # rmax_m scales the mode solver's mesh-convergence tolerance;
        # None → derive at run() from receiver.range_max.
        self.rmax_m = float(rmax_m) if rmax_m is not None else None
        # mode_depth_grid overrides compute_modes's dense grid; None → density.
        self.mode_depth_grid = (
            np.asarray(mode_depth_grid, dtype=float)
            if mode_depth_grid is not None else None
        )
        if leaky_modes:
            # CHIGH→∞ so kraken/krakenc attempts leaky modes (Kraken doc).
            self.c_high = 1e9
        self._validate_phase_speed_limits()
        if backend is not None and backend not in ('kraken', 'krakenc'):
            raise ConfigurationError(
                f"Kraken(backend={backend!r}) is not a known backend. "
                f"Choose 'kraken', 'krakenc', or None for automatic dispatch."
            )
        self.backend = backend
        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        if mode_coupling not in ('adiabatic', 'coupled'):
            raise ConfigurationError(
                f"mode_coupling must be 'adiabatic' or 'coupled', "
                f"got {mode_coupling!r}"
            )

        # Keep the user's ``executable`` arg verbatim (``None`` when
        # auto-detected) so ``model.copy()`` re-resolves the binary instead of
        # re-pinning the already-resolved absolute path. The resolved kraken.exe
        # path lives in ``self._exe``; ``_select_kraken_exe`` may swap in
        # krakenc.exe per-env.
        self.executable = Path(executable) if executable is not None else None
        if self.executable is None:
            self._exe = self._find_executable_in_paths(
                'kraken.exe',
                bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Kraken',
            )
        else:
            self._exe = self.executable

        if not self._exe.exists():
            raise ExecutableNotFoundError(self.model_name, str(self._exe))

        # field.exe is only needed for field-producing run modes (TL /
        # broadband / time-series), not for MODES. Store the user arg for
        # copy() round-tripping and resolve lazily on first field run so a
        # MODES-only Kraken doesn't require field.exe to be installed.
        self.field_executable = (
            Path(field_executable) if field_executable is not None else None
        )
        self._field_exe: Optional[Path] = None

        self.mode_points_per_meter = mode_points_per_meter
        self.mode_coupling = mode_coupling
        self.n_segments = n_segments

    def _build_field_option(self, is_range_dependent: bool,
                            source: Source, run_mode: RunMode) -> str:
        """Build the 4-character option string for field.exe.

        Columns follow AT ``field.f90`` / ``ReadModes.f90``:

        * pos 1: source geometry from ``source.source_type`` — 'R' point
          source (cylindrical), 'X' line source (Cartesian), 'S' scaled
          point source.
        * pos 2: coupling — 'C' coupled modes, 'A' adiabatic.
          For NProf > 1 we honour ``mode_coupling``; for range-independent
          runs we default to 'C' (coupled) so the option string is fully
          populated rather than containing a padding blank. AT's
          field.f90 treats NProf == 1 identically for 'A' and 'C'.
        * pos 3: source beam pattern — '*' when ``source.beam_pattern`` is
          set, else ' ' (omnidirectional). Field.exe rejects any other character
          (``field.f90:83-90``), so the elastic Comp selector (H/V/T/N)
          is not exposed here; it is only reachable if a user invokes
          ReadModes directly.
        * pos 4: 'C' coherent TL, 'I' incoherent — from ``run_mode``
          (``RunMode.INCOHERENT_TL`` is the only 'I' case; BROADBAND /
          TIME_SERIES are coherent by construction).
        """
        pos1 = _SOURCE_TYPE_CODE[source.source_type]
        if is_range_dependent:
            pos2 = 'C' if self.mode_coupling.lower() == 'coupled' else 'A'
        else:
            # Range-independent: AT doesn't require 'A'/'C', but setting
            # 'C' keeps the option string fully specified and matches what
            # AT's own field.f90 does internally when NProf == 1.
            pos2 = 'C'
        # pos3: '*' => field.exe reads <base>.sbp, else omnidirectional.
        pos3 = '*' if source.beam_pattern is not None else ' '
        pos4 = 'I' if run_mode == RunMode.INCOHERENT_TL else 'C'
        return f"{pos1}{pos2}{pos3}{pos4}"

    def _resolve_field_executable(self) -> Path:
        """Resolve ``field.exe`` lazily (only field-producing run modes need
        it). Cached in ``self._field_exe`` after the first call."""
        if self._field_exe is not None:
            return self._field_exe
        if self.field_executable is not None:
            fx = self.field_executable
        else:
            fx = self._find_executable_in_paths(
                'field.exe',
                bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Kraken',
            )
        if not fx.exists():
            raise ExecutableNotFoundError(
                f"{self.model_name} (field.exe)", str(fx),
            )
        self._field_exe = fx
        return fx

    def select_backend(self, env, run_mode=None) -> str:
        """Logical name of the modes backend that would run for ``env`` —
        ``'kraken'`` or ``'krakenc'``. Round-trips with ``backend=`` and
        mirrors ``RAM.select_backend`` (``run_mode`` accepted for signature
        parity; it does not affect the modes-binary choice). Introspects the
        dispatch without running anything."""
        return 'krakenc' if self._select_kraken_exe(env).name.startswith(
            'krakenc') else 'kraken'

    def _project_environment(self, env):
        """Collapse unsupported features, then express ``top_reflection_file``
        as the surface carrier it is shorthand for.

        Staging goes through the one path that owns it — ``write_header``
        copies ``env.surface.reflection_file`` to ``<root>.trc``, which
        ``misc/RefCoef.f90:64-76`` opens for ``TopOpt(2:2)='F'``.

        This rewrite belongs here, not in a deck writer: ``_write_field_env``
        branches to ``write_multi_profile_env`` for a range-dependent run and
        never calls ``_write_kraken_env``, so a rewrite living there left the
        multi-profile deck with a vacuum ``TopOpt`` and no staged ``.trc`` on
        exactly the runs ``_reflection_table_boundaries`` had already routed to
        ``krakenc``. Every entry path projects first, so doing it here reaches
        both writers.
        """
        env = super()._project_environment(env)
        if self.top_reflection_file is None:
            return env
        from uacpy.core.bottom import BoundaryProperties
        from uacpy.core.surface import Surface
        if not self.top_reflection_file.exists():
            raise ConfigurationError(
                f"top_reflection_file not found: {self.top_reflection_file}"
            )
        env.surface = Surface(properties=[BoundaryProperties(
            acoustic_type='file',
            reflection_file=str(self.top_reflection_file))])
        return env

    def _reflection_table_boundaries(self, env) -> list:
        """Boundaries of ``env`` that carry a tabulated reflection coefficient.

        Returns human-readable labels (empty when there are none). Every range
        node is inspected, matching the ``any``-semantics of
        ``env.has_elastic_*``: the dispatch runs before the environment is
        projected, so a table anywhere on the axis still ends up in the deck.
        """
        labels = []
        if self.top_reflection_file is not None:
            labels.append("surface (Kraken(top_reflection_file=...) → '.trc')")
        for p in env.surface.properties:
            if p.acoustic_type in _REFLECTION_TABLE_TYPES:
                labels.append(f"surface (acoustic_type={p.acoustic_type!r})")
        for column in env.bottom.columns:
            if column.halfspace.acoustic_type in _REFLECTION_TABLE_TYPES:
                labels.append(
                    f"bottom (acoustic_type={column.halfspace.acoustic_type!r})")
        return sorted(set(labels))

    def _reject_precalc_surface(self, env) -> None:
        """A ``'precalc'`` surface has no reader in the Kraken family.

        ``misc/RefCoef.f90:92`` reads an ``.irc`` only for ``BotRC == 'P'`` —
        there is no top branch — so ``TopOpt(2)='P'`` leaves ``xTab``/``fTab``
        unpopulated. ``kraken.exe`` stops on it at ``Kraken/kraken.f90:47-48``
        and ``krakenc.exe`` runs ``InterpolateIRC`` over the empty table and
        dies with SIGSEGV.
        """
        if any(p.acoustic_type == 'precalc' for p in env.surface.properties):
            raise UnsupportedFeatureError(
                self.model_name,
                "a 'precalc' (.irc) sea surface — the Acoustics Toolbox reads "
                "an internal reflection coefficient for the bottom only "
                "(misc/RefCoef.f90:92), so the top table is never loaded",
                alternatives=[
                    "acoustic_type='file' with the .brc/.trc table BOUNCE also "
                    "writes",
                    'Bellhop',
                ],
            )

    def _select_kraken_exe(self, env):
        """The modes binary for this env: ``krakenc.exe`` for elastic media,
        leaky modes (complex eigenvalues) or a tabulated reflection
        coefficient, ``kraken.exe`` otherwise.

        A constructor ``backend=`` override forces the choice; forcing
        ``'kraken'`` on an environment kraken.exe cannot answer raises
        ``ConfigurationError`` rather than producing a wrong field.
        """
        self._reject_precalc_surface(env)
        complex_modes = (
            env.has_elastic_bottom
            or env.has_elastic_surface
            or getattr(self, 'leaky_modes', False)
        )
        reflection_tables = self._reflection_table_boundaries(env)
        forced = getattr(self, 'backend', None)
        if forced == 'kraken' and complex_modes:
            raise ConfigurationError(
                "Kraken(backend='kraken') answers incorrectly on elastic media "
                "/ leaky modes: kraken.exe clamps c_high to the half-space "
                "shear speed (dropping every faster mode) and its absorption "
                "perturbation skips elastic media (returning Im(k)=0). Use "
                "backend='krakenc', or backend=None for automatic dispatch."
            )
        if forced == 'kraken' and reflection_tables:
            raise ConfigurationError(
                f"Kraken(backend='kraken') cannot honour the tabulated "
                f"reflection coefficient on the "
                f"{', '.join(reflection_tables)}: kraken.exe either stops "
                f"('The option to read a file for the reflection loss is not "
                f"implemented in KRAKEN') or silently substitutes a rigid "
                f"boundary. Use backend='krakenc', or backend=None for "
                f"automatic dispatch."
            )
        if forced == 'krakenc' or (
                forced is None and (complex_modes or reflection_tables)):
            return self._find_executable_in_paths(
                'krakenc.exe',
                bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Kraken',
            )
        return self._exe  # kraken.exe

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
        Compute TL field using normal modes.

        Uses AT pipeline: kraken.exe → field.exe → .shd.

        Parameters
        ----------
        env : Environment
            Can be range-dependent or range-independent.
        source : Source
            Source configuration.
        receiver : Receiver
            Receiver grid.
        run_mode : RunMode, optional
            ``COHERENT_TL`` (default), ``INCOHERENT_TL``, ``BROADBAND``,
            or ``TIME_SERIES``. ``TIME_SERIES`` requires
            ``source_waveform`` + ``sample_rate``. ``INCOHERENT_TL`` sums
            mode magnitudes and returns real dB TL (``kind='tl'``).
        frequencies : ndarray, optional
            Frequency vector (Hz) for native broadband computation. A
            multi-element grid uses TopOpt(6)='B' so kraken writes one
            multi-frequency .mod file and field.exe handles all
            frequencies in a single pass; a single-element grid is solved
            through the narrowband pipeline and returned on a length-1
            frequency axis.
        source_waveform : ndarray, optional
            Source pulse for ``TIME_SERIES`` mode.
        sample_rate : float, optional
            Sampling rate of ``source_waveform`` in Hz.
        output_duration : float, optional
            Desired output duration (seconds) for ``TIME_SERIES``. When
            given, the source waveform is zero-padded internally so the
            auto-derived broadband grid is tight enough (``Δf =
            1/output_duration``). Defaults to
            ``len(source_waveform)/sample_rate``.
        """
        self._reject_elastic_over_fluid_halfspace(env)
        self._reject_elastic_surface(env)

        # Default run mode: BROADBAND if a freq vector is provided,
        # else single-frequency coherent TL.
        smart_default = (
            RunMode.BROADBAND
            if frequencies is not None and len(np.atleast_1d(frequencies)) > 1
            else RunMode.COHERENT_TL
        )
        run_mode = self._resolve_run_mode(run_mode, default=smart_default)

        # Early gate: coupled-mode field calculations cannot be
        # combined with incoherent mode addition. ``KrakenField/field.f90:
        # 125-129`` calls ERROUT on Opt(2:2)='C' + Opt(4:4)='I', which surfaces
        # in Python as an opaque "no .shd file" error. Fail loudly up front.
        if (
            run_mode == RunMode.INCOHERENT_TL
            and env.is_range_dependent
            and self.mode_coupling == 'coupled'
        ):
            raise ConfigurationError(
                "Kraken: coupled mode calculations do not support "
                "incoherent addition of modes. Use mode_coupling="
                "'adiabatic' with run_mode=RunMode.INCOHERENT_TL, or keep "
                "mode_coupling='coupled' with run_mode=RunMode.COHERENT_TL."
            )

        if run_mode not in (RunMode.BROADBAND, RunMode.TIME_SERIES):
            self._warn_ignored_run_kwargs(
                run_mode,
                frequencies=frequencies,
                source_waveform=source_waveform,
                sample_rate=sample_rate,
                output_duration=output_duration,
            )

        # Resolve the modes binary once, as a visible dispatch step (parity
        # with RAM.select_backend): kraken.exe vs krakenc.exe is a
        # deterministic function of env elasticity / leaky_modes — preserved
        # through _project_environment — so the sub-paths receive it rather
        # than each re-resolving. A forced backend incompatible with the env
        # raises here.
        kraken_exe = self._select_kraken_exe(env)
        self._log(f"Dispatching to {kraken_exe.name} (modes binary)")

        if (run_mode == RunMode.INCOHERENT_TL
                and kraken_exe.name.startswith('krakenc')):
            warnings.warn(
                f"{self.model_name}: INCOHERENT_TL on the krakenc backend is "
                "not a strict incoherent sum. field.exe's Opt(4:4)='I' branch "
                "computes SQRT(SUM(z**2)) over the per-mode contributions "
                "(EvaluateMod.f90:66), which equals the energy sum "
                "SQRT(SUM(|z|**2)) only for real mode functions; krakenc's "
                "complex phi and k leave cross-mode phase in the square. Use "
                "backend='kraken' where the environment allows it, or "
                "RunMode.COHERENT_TL.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

        if run_mode == RunMode.MODES:
            # Modes only need the modes binary, never field.exe. The elastic
            # sub-bottom receiver partition below is a field-evaluation concern
            # (and would emit a spurious NaN-receivers warning here), so it is
            # skipped entirely for the modes solve.
            return self._run_modes(
                env, source, receiver, n_modes=self.n_modes, exe=kraken_exe)

        # field.exe cannot evaluate the field inside an elastic medium: its
        # Comp selector (field.f90:171 -> ReadModes.f90:315-324) has no
        # elastic component. With an elastic layer present, receivers below
        # the water column are computed in the water column only and returned
        # as NaN below it. (The modes solve itself is fine here — the
        # elastic-over-fluid-halfspace case that hangs krakenc.exe was already
        # rejected by _reject_elastic_over_fluid_halfspace above.)
        elastic_subbottom = (
            env.has_layered_bottom
            and any(
                (getattr(layer, 'shear_speed', 0) or 0) > 0
                for layer in env.bottom.columns[0].layers
            )
        )
        rcv, keep = self._partition_elastic_subbottom(env, receiver, elastic_subbottom)

        if run_mode in (RunMode.BROADBAND, RunMode.TIME_SERIES):
            source_waveform, frequencies = self._prepare_timeseries(
                run_mode, source, frequencies, source_waveform, sample_rate,
                output_duration,
            )
            # _project_environment + validate_inputs run inside
            # _compute_broadband_field (per-frequency), not here.
            tf = self._compute_broadband_field(
                env, source, rcv,
                frequencies=frequencies, n_modes=self.n_modes, exe=kraken_exe,
            )
            tf = self._reinsert_nan_depths(tf, receiver, keep)
            if run_mode == RunMode.TIME_SERIES:
                return tf.synthesize_time_series(
                    source_waveform=source_waveform,
                    sample_rate=sample_rate,
                )
            return tf

        env = self._project_environment(env)
        self.validate_inputs(env, source, rcv, run_mode=run_mode)
        field = self._compute_field_via_exe(
            env, source, rcv, n_modes=self.n_modes, exe=kraken_exe,
            run_mode=run_mode,
        )
        return self._reinsert_nan_depths(field, receiver, keep)

    def _modes_single_profile(self, env: Environment) -> Environment:
        """Normal modes are range-independent; reduce any range-dependent env
        to its r=0 profile (a run()-time numerical requirement — the field
        path segments RD natively, the modes solve cannot)."""
        if not env.is_range_dependent:
            return env
        warnings.warn(
            f"{self.model_name}: normal modes are range-independent; sampling "
            f"the r=0 profile of the range-dependent environment.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
        ssp = env.ssp.collapse('r0') if env.has_range_dependent_ssp else env.ssp
        bottom = env.bottom
        if bottom is not None and bottom.is_range_dependent:
            bottom = bottom.select_range('r0')
        return Environment(
            name=env.name,
            bathymetry=float(env.bathymetry.eval(range=0.0)),
            ssp=ssp,
            bottom=bottom,
            surface=env.surface.collapse('r0'),
            absorption=env.absorption,
        )

    def _run_modes(self, env, source, receiver, *, n_modes, exe=None):
        """Run the modes binary only (kraken.exe / krakenc.exe by ``backend=``
        / elasticity) and return a :class:`Modes` result — no field.exe.

        Reached only through :meth:`run` (``compute_modes`` dispatches there
        too), which resolves ``exe`` and has already run the elastic-media
        guards."""
        kraken_exe = exe if exe is not None else self._select_kraken_exe(env)
        env = self._project_environment(self._modes_single_profile(env))
        self.validate_inputs(env, source, receiver, run_mode=RunMode.MODES)

        fm = self._setup_file_manager()
        base_name = 'modes'
        try:
            env_file = fm.get_path(f'{base_name}.env')
            self._log(f"Writing environment file: {env_file}")
            bounds = self._write_kraken_env(
                env_file, env, source,
                receiver_obj=receiver,
                receiver_depths=receiver.depths,
            )
            self._log(f"Running {kraken_exe.name} (modes)...")
            self._run_kraken_executable(base_name, fm.work_dir, exe=kraken_exe)

            modes_file = fm.get_path(base_name)
            self._log(f"Reading mode file: {modes_file}.mod")
            modes = self._read_modes_file(modes_file)

            field = self._build_modes_field(
                modes, n_modes, source, backend_exe=kraken_exe, bounds=bounds,
            )
            self._attach_output_paths(
                field, fm.work_dir, base_name,
                primary_files=(('mod_file', '.mod'),),
            )
            return field
        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _partition_elastic_subbottom(self, env, receiver, elastic_subbottom):
        """Split receivers into the water column (field.exe can evaluate) and
        the elastic sub-bottom (it cannot). Returns ``(compute_receiver,
        keep_mask)`` with ``keep_mask`` marking water-column depths in the
        original ordering, or ``(receiver, None)`` when no split is needed."""
        if not elastic_subbottom:
            return receiver, None
        depths = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
        keep = depths <= env.depth
        if keep.all():
            return receiver, None
        warnings.warn(
            f"{self.model_name}: {int((~keep).sum())} receiver depth(s) lie in "
            "the elastic sub-bottom, where field.exe cannot evaluate the "
            "field; returning NaN there.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
        # ``misc/SourceReceiverPositions.f90:212`` ERROUTs on an empty receiver
        # vector, so a deck whose every requested depth sits in the sub-bottom
        # still needs one legal depth. Mid-water is arbitrary: with ``keep``
        # all-False ``_reinsert_nan_depths`` discards the computed column and
        # returns NaN at every original depth.
        compute_depths = depths[keep] if keep.any() else np.array([0.5 * env.depth])
        compute_receiver = Receiver(
            depths=compute_depths, ranges=receiver.ranges,
        )
        return compute_receiver, keep

    def _mask_zero_range(self, field, receiver, source):
        """NaN the ``r = 0`` column of a point-source field.

        ``EvaluateMod.f90:71-73`` guards the ``1/√(r + ro)`` cylindrical
        spreading with a ``TINY`` test, so at ``r = 0`` field.exe returns the
        bare modal sum — a number that belongs to no range. Scooter's
        wavenumber transform masks the same cells
        (``grn_reader._hankel_transform``); masking here keeps the two models'
        grids comparable cell by cell. ``'line'`` / ``'scaled'`` sources carry
        no ``1/√r`` and are left alone, as they are in both codes.
        """
        if source.source_type != 'point':
            return field
        ranges = np.atleast_1d(np.asarray(receiver.ranges, dtype=float))
        zero = np.abs(ranges) < np.finfo(float).tiny
        if not zero.any():
            return field
        warnings.warn(
            f"{self.model_name}: {int(zero.sum())} receiver range(s) at r = 0, "
            "where the point-source cylindrical-spreading factor 1/sqrt(r) is "
            "singular; those cells are returned as NaN (no data). Move the "
            "receiver off the source axis (e.g. r = 1 m) to get a field value.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
        data = np.array(field.data)
        data[:, zero, ...] = np.nan
        field.data = data
        return field

    def _reinsert_nan_depths(self, field, receiver, keep):
        """Map ``field`` (computed on the water-column receivers) back onto the
        original receiver depth axis, NaN at the dropped sub-bottom depths.
        No-op when ``keep`` is None."""
        if keep is None:
            return field
        full_depths = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
        d = field.to_dict()
        data = np.asarray(d['data'])
        full = np.full((full_depths.size,) + data.shape[1:], np.nan, dtype=data.dtype)
        if keep.any():
            full[keep, ...] = data
        d['data'] = full
        d['coords'] = {**d['coords'], 'depth': full_depths}
        return Field.from_dict(d)

    def _count_modes_at_freq(self, env, source, receiver, freq, exe) -> int:
        """Number of trapped modes Kraken finds at a single ``freq`` (Hz).

        ``0`` means the frequency is below the waveguide's modal cutoff — no
        propagating modes (Computational Ocean Acoustics; Brekhovskikh &
        Lysanov). Uses a single-frequency modes run: a multi-frequency ``.mod``
        that contains a zero-mode record is itself unreadable (the record stride
        breaks at ``M=0``), so the cutoff must be probed one frequency at a
        time."""
        from uacpy.core.source import Source as _Source
        from uacpy.io.modes_reader import read_modes_bin
        fm = self._setup_file_manager()
        base = 'mcut'
        try:
            self._write_kraken_env(
                fm.get_path(f'{base}.env'), env,
                _Source(depths=source.depths, frequencies=float(freq)),
                receiver_obj=receiver, receiver_depths=receiver.depths,
            )
            self._run_kraken_executable(base, fm.work_dir, exe=exe)
            return int(read_modes_bin(str(fm.get_path(base)),
                                      frequency=float(freq)).get('M', 0))
        except TypeError:
            # A signature mismatch here is a programming error, not a
            # below-cutoff condition; the broad handler below would report it
            # as "zero modes" at every frequency and silently disable the
            # sub-cutoff recovery entirely.
            raise
        except Exception as e:   # noqa: BLE001
            # Below cutoff the single-freq .mod is unreadable (zero-mode record),
            # which legitimately means "no modes". Infrastructure failures
            # (missing exe, subprocess crash, disk) also land here — log so they
            # are not fully silent, then treat as below cutoff.
            self._log(f"_count_modes_at_freq({float(freq):g} Hz) failed "
                      f"({type(e).__name__}: {e}); treating as below cutoff.",
                      level="debug")
            return 0
        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _propagating_frequency_floor(self, env, source, receiver, freqs, exe):
        """Index of the first frequency in (sorted, ascending) ``freqs`` that
        has propagating modes — i.e. the end of the contiguous sub-cutoff band.

        Mode count rises monotonically with frequency, so the zero-mode
        frequencies are a prefix ``[0, floor)`` and ``floor`` can be found by
        binary search in O(log N) single-frequency probes (bounded — not one
        probe per frequency). Returns ``len(freqs)`` if none propagate."""
        freqs = np.atleast_1d(freqs)
        lo, hi = 0, len(freqs)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._count_modes_at_freq(
                    env, source, receiver, float(freqs[mid]), exe) > 0:
                hi = mid          # modes here → cutoff is at or below mid
            else:
                lo = mid + 1      # no modes here → cutoff is above mid
        return lo

# ── field.exe pipeline ──────────────────────────────────────────────

    def _segment_env_for_field(self, env):
        """Segment a range-dependent env into per-range profiles for the
        multi-profile kraken field run.

        Returns ``(segments, n_profiles, profile_ranges_m, max_total_depth)``.
        ``max_total_depth`` is the shared bottom ``write_multi_profile_env``
        will declare, taken from the same planner the writer uses so the two
        cannot drift apart: the mode-tabulation grid is built to span
        ``[0, max_total_depth]``, and ``EvaluateCMMod.f90:313`` stops a coupled
        run outright unless that grid ends *exactly* on the declared bottom.
        """
        segments = segment_environment_by_range(env, n_segments=self.n_segments)
        n_profiles = len(segments)
        _, max_total_depth, _ = plan_multi_profile_media(segments)

        profile_ranges_m = np.array([s[0] for s in segments])
        self._log(f"Range-dependent: {n_profiles} profiles, "
                  f"mode_coupling={self.mode_coupling}")
        return segments, n_profiles, profile_ranges_m, max_total_depth

    def _fixed_mesh_points(self, segments, max_total_depth, freq) -> int:
        """Mesh points per medium shared by every profile of a multi-profile
        ``.env``.

        One value is used throughout so the ``.mod`` record length cannot grow
        between profiles (``krakenc.f90:629``). AT sizes each medium at 20
        points per wavelength and aborts with *Mesh is too coarse* below half
        of that (``misc/ReadEnvironmentMod.f90:99-112``); crucially it takes
        the wavelength from the medium's **shear** speed wherever one is set,
        and an elastic sediment's shear wavelength can be an order of magnitude
        shorter than its compressional one.

        The shared value must clear every medium's own requirement, so each
        medium class is bounded by the widest span it can occupy and the
        slowest speed it can mesh on: the water column reaches at most the
        deepest seafloor, while the seabed media run from the shallowest
        seafloor down to ``max_total_depth`` (``write_multi_profile_env``
        stretches its deepest pad there). Bounding the two separately keeps a
        thin elastic layer under deep water from sizing the whole mesh.

        A pinned ``n_mesh`` is honoured, and rejected when it falls under the
        ``Nneeded / 2`` floor the same reader enforces. That floor is measured
        per medium off the media the deck actually carries
        (:meth:`_multi_profile_media`), not off these bounding spans: the spans
        deliberately overshoot to keep the auto mesh generous, and reusing them
        as the rejection threshold blocks configurations KRAKEN accepts.
        """
        water_speeds, seabed_speeds = [], []
        for _, seg in segments:
            water_speeds.append(float(np.min(seg.ssp.data)))
            if seg.bottom is None:
                continue
            for column in seg.bottom.columns:
                media = list(column.layers)
                # Same skip set as ``Bottom.all_sound_speeds``: a vacuum /
                # rigid / file half-space carries the placeholder speed
                # ``BoundaryProperties.__post_init__`` filled in, not a
                # physical one, so it must not size the mesh.
                if column.halfspace.acoustic_type not in ('vacuum', 'rigid',
                                                          'file'):
                    media.append(column.halfspace)
                for medium in media:
                    shear = float(getattr(medium, 'shear_speed', 0.0) or 0.0)
                    seabed_speeds.append(shear if shear > 0.0
                                         else float(medium.sound_speed))

        seafloors = [float(seg.depth) for _, seg in segments]
        spans = [(max(seafloors), water_speeds)]
        if seabed_speeds:
            spans.append((float(max_total_depth) - min(seafloors),
                          seabed_speeds))

        needed = [
            span * freq / (min(speeds) if speeds else DEFAULT_SOUND_SPEED) * 20
            for span, speeds in spans
        ]
        computed = max(500, int(max(needed)))
        if self.n_mesh <= 0:
            return computed
        floor = at_mesh_floor(self._multi_profile_media(segments), freq)
        if self.n_mesh < floor:
            raise ConfigurationError(
                f"Kraken(n_mesh={self.n_mesh}) is below the {floor} mesh "
                f"points misc/ReadEnvironmentMod.f90:110-112 requires for the "
                f"coarsest medium of this range-dependent environment at "
                f"{freq:.4g} Hz; the run would stop with 'Mesh is too coarse'.",
                remediation=f"Pass n_mesh >= {floor}, or leave n_mesh=0 to let "
                            f"uacpy size the shared mesh ({computed} here).",
            )
        return int(self.n_mesh)

    @staticmethod
    def _multi_profile_media(segments):
        """``(thickness, speed)`` per medium across every profile of the deck.

        ``plan_multi_profile_media`` is the deck's geometry of record — it
        returns each profile's media already quantised to the written ``.1f``
        resolution and with the last one stretched onto the common bottom — so
        the mesh bound is read off it rather than re-derived. The water column
        of each profile is medium 1.
        """
        from uacpy.io.oalib_writer import deck_depth

        media = []
        for _range_km, seg in segments:
            seafloor = deck_depth(seg.depth)
            water = seg.ssp.extend_to(seafloor).to_pairs()
            media.append((seafloor, float(water[-1, 1])))

        _n_media, _bottom_depth, plans = plan_multi_profile_media(segments)
        for plan in plans:
            for top, bot, cp, cs, *_rest in plan:
                shear = float(cs or 0.0)
                media.append((float(bot) - float(top),
                              shear if shear > 0.0 else float(cp)))
        return media

    def _write_field_env(self, env, source, receiver, fm, base_name,
                         segments, max_total_depth, broadband, freq_vec
                         ) -> Dict[str, float]:
        """Write the kraken ``.env`` for the field run — a multi-profile env for
        a range-dependent (segmented) environment, else a single-profile env.
        Mode depths span the full ocean + sediment; range-dependent broadband
        is unsupported and raises.

        Returns the resolved ``{'c_low', 'c_high', 'rmax'}`` (m/s, m) the deck
        was written with."""
        # Both deck paths share one gate: the range-dependent branch below
        # writes the multi-profile deck itself instead of going through
        # _write_kraken_env, so the SSP-type and phase-speed checks are applied
        # here rather than inside the single-profile writer alone.
        self._check_kraken_ssp_type()
        self._validate_phase_speed_limits()

        env_file = fm.get_path(f'{base_name}.env')

        # Mode depths must cover the full ocean + sediment for all
        # profiles. Use max total media depth across all segments.
        n_mode_depths = max(100, int(max_total_depth * self.mode_points_per_meter))
        mode_depths = np.linspace(0, max_total_depth, n_mode_depths)
        receiver_for_modes = Receiver(depths=mode_depths, ranges=receiver.ranges)

        if env.is_range_dependent and segments is not None:
            # 5 % past the outermost receiver — the same mesh-convergence
            # tolerance the range-independent path uses — or ``self.rmax_m``
            # when pinned.
            rmax_m = (
                self.rmax_m
                if self.rmax_m is not None
                else self._compute_rmax_m(receiver, multiplier=1.05)
            )

            n_mesh_fixed = self._fixed_mesh_points(
                segments, max_total_depth, float(source.frequencies[0]))

            if broadband:
                # ``write_multi_profile_env`` has no broadband form; refuse
                # rather than drop the frequency vector.
                raise ConfigurationError(
                    "Kraken does not support range-dependent "
                    "broadband runs. Pass a single frequency or make "
                    "the environment range-independent."
                )

            c_low = self._c_low_for(env)
            write_multi_profile_env(
                filepath=env_file,
                segments=segments,
                source=source,
                receiver=receiver_for_modes,
                interp_ssp=self.interp_ssp,
                n_mesh=n_mesh_fixed,
                c_low=c_low,
                c_high=self.c_high,
                rmax_m=rmax_m,
            )
            # cLow and RMax are written identically into every profile block;
            # an unpinned cHigh resolves per profile from that profile's SSP
            # and half-space, so the widest of them is what bounds the run.
            c_high = max(
                resolve_phase_speed_bounds(seg_env, c_low, self.c_high)[1]
                for _, seg_env in segments
            )
            return {'c_low': float(c_low), 'c_high': float(c_high),
                    'rmax': float(rmax_m)}

        return self._write_kraken_env(
            env_file, env, source,
            receiver_obj=receiver_for_modes,
            receiver_depths=mode_depths,
            frequencies=freq_vec if broadband else None,
        )

    def _run_field_exe(self, fm, base_name, option):
        """Run field.exe → ``.shd`` and return the output path. field.exe may
        exit non-zero on a successful run (known Fortran teardown bug) — warn
        and read the ``.shd`` anyway; a *missing* ``.shd`` is a real failure
        and raises ``ModelExecutionError``.

        Tolerating a non-zero exit is exactly why the missing-``.shd`` test
        below has to be trustworthy, so any earlier ``.shd`` in a pinned
        ``work_dir`` is cleared first (``_run_and_attach_prt``'s
        ``stale_outputs``, which this launch path cannot use). ``field.prt``
        goes with it — it is read back as a failure signal, so an earlier
        run's copy must not be mistaken for this one's."""
        for suffix in _KRAKEN_FIELD_OUTPUTS:
            fm.get_path(f'{base_name}{suffix}').unlink(missing_ok=True)
        fm.get_path(f'{_FIELD_PRT_ROOT}.prt').unlink(missing_ok=True)
        self._log(f"Running field.exe (option='{option}')...")
        try:
            self._run_subprocess(
                [str(self._resolve_field_executable()), base_name],
                cwd=fm.work_dir,
            )
        except ModelExecutionError as exc:
            # A timeout means the run never finished, so there is no output to
            # salvage; only a non-zero teardown status is worth continuing past.
            if exc.timed_out:
                self._attach_prt_tail(exc, fm.work_dir, base_name)
                raise
            warnings.warn(
                f"{self.model_name}: field.exe exited with non-zero "
                f"status ({exc}); attempting to read the .shd output "
                "anyway (known Fortran cleanup issue).",
                UserWarning,
                skip_file_prefixes=USER_FRAME_SKIP,
            )

        self._raise_on_field_fatal(fm.work_dir)

        shd_file = fm.get_path(f'{base_name}.shd')
        if not shd_file.exists() or shd_file.stat().st_size == 0:
            # field.exe's own log, not <base_name>.prt — that one belongs to
            # the modes binary and shows a successful mode calculation
            # (field.f90:44 hard-codes 'field.prt').
            exc = ModelExecutionError(
                self.model_name, return_code=0, stdout=None,
                stderr=(
                    "field.exe produced no usable .shd file (missing or "
                    "empty); check the .prt log at "
                    f"{fm.get_path(_FIELD_PRT_ROOT + '.prt')}"
                ),
            )
            self._attach_prt_tail(exc, fm.work_dir, _FIELD_PRT_ROOT)
            raise exc
        return shd_file

    def _raise_on_field_fatal(self, work_dir) -> None:
        """Raise when field.exe reported a fatal but left a readable ``.shd``.

        ``_raise_on_fortran_fatal`` cannot see either of field.exe's stop
        routes, because both write to ``field.prt`` — ``field.f90:44``
        hard-codes that name — rather than ``<base_name>.prt``:

        * every ERROUT site (11 in ``field.f90``, plus ``ReadModes.f90``,
          ``SourceReceiverPositions.f90`` and ``beampattern.f90``) writes the
          uppercase ``*** FATAL ERROR ***`` banner, the subroutine name and
          the message (``misc/FatalError.f90:16-24``), then stops with
          ``'Fatal Error: Check the print file for details'`` on stderr and an
          exit status of 0;
        * ``EvaluateCM``'s depth-grid check is a bare
          ``WRITE( *, * ) 'Fatal Error: …'`` followed by an argument-less
          ``STOP`` (``EvaluateCMMod.f90:313-317``), so it carries no banner at
          all.

        The scan is therefore case-insensitive on ``fatal error`` and covers
        both. field.exe has by then already created the ``.shd`` and written
        its header, so the file is present and non-empty but holds no field.
        Left alone it reaches the reader, whose header-count guard reports a
        size mismatch — a message that describes the stub and hides the
        diagnosis.
        """
        prt = read_prt(Path(work_dir) / f'{_FIELD_PRT_ROOT}.prt')
        if not prt:
            return
        marker = prt.lower().rfind('fatal error')
        if marker < 0:
            return
        detail = ' '.join(
            line.strip() for line in prt[marker:].splitlines() if line.strip()
        )
        exc = ModelExecutionError(
            self.model_name, return_code=0, stdout=None,
            stderr=f"field.exe stopped — {detail}",
        )
        self._attach_prt_tail(exc, work_dir, _FIELD_PRT_ROOT)
        raise exc

    def _assemble_field_from_shd(self, shd_file, source, receiver, is_rd,
                                 n_profiles, broadband, freq_vec, return_pressure,
                                 fm, base_name, run_mode, bounds):
        """Build the result Field from field.exe's ``.shd``: native-broadband
        ``(n_d, n_r, n_f)``, single-frequency complex pressure
        (``return_pressure``), or narrowband TL. field.exe's modal sum differs
        from Scooter's Hankel path by an overall -1, negated here so every
        coherent branch carries the ``'travelling_wave'`` phase reference.
        ``RunMode.INCOHERENT_TL`` instead yields real dB TL with no phase
        reference — a magnitude sum has no phase to reference."""
        if broadband:
            shd0 = read_shd_bin(str(shd_file))
            freqs_read = np.asarray(shd0['freqVec'], dtype=float)
            # A sub-cutoff (zero-mode) frequency corrupts the multi-frequency
            # .mod: field.exe sometimes produces a .shd anyway but with a
            # garbage (e.g. all-zero) frequency axis. Detect the mismatch and
            # raise so the broadband caller recovers (drops the sub-cutoff
            # band and zero-fills) instead of silently returning a zero field.
            if (len(freqs_read) != len(freq_vec)
                    or not np.allclose(np.sort(freqs_read),
                                       np.sort(np.asarray(freq_vec, float)),
                                       rtol=1e-3, atol=1e-6)):
                exc = ModelExecutionError(
                    self.model_name, return_code=0, stdout=None,
                    stderr=("field.exe returned a frequency axis that does "
                            "not match the request — the modes file is "
                            "corrupted by a sub-cutoff (zero-mode) frequency."))
                self._attach_prt_tail(exc, fm.work_dir, base_name)
                raise exc
            # New layout: (n_d, n_r, n_f).
            p_stack = np.zeros(
                (len(receiver.depths), len(receiver.ranges), len(freqs_read)),
                dtype=np.complex128,
            )
            for i_freq, fr in enumerate(freqs_read):
                shd_i = read_shd_bin(str(shd_file), frequency=float(fr))
                # read_shd_bin returns pressure as (Ntheta, Nsz, Nrz, Nrr);
                # [0, 0] selects the single bearing and single source depth the
                # deck was written with.
                # field.exe (EvaluateMod.f90:34,42) emits the modal sum
                # under the engineering carrier exp(-ikr) but with a
                # leading factor i·√(2π)·exp(iπ/4); Scooter's Hankel
                # path produces -exp(iπ/4)/√(2πr). The two prefactors
                # differ by an overall -1 (NOT a conjugation), so a
                # plain negation aligns the two travelling-wave fields.
                p_stack[:, :, i_freq] = -shd_i['pressure'][0, 0, :, :]
            field = Field(
                data=p_stack,
                coords={
                    'depth': receiver.depths,
                    'range': receiver.ranges,
                    'frequency': freqs_read,
                },
                phase_reference='travelling_wave',
                **self._result_kwargs(
                    source,
                    backend='field',
                    frequencies=freqs_read,
                    mode_coupling=self.mode_coupling if is_rd else 'none',
                    n_profiles=n_profiles,
                    native_broadband=True,
                    **bounds,
                ),
            )
        elif return_pressure:
            shd_data = read_shd_bin(str(shd_file))
            # (Ntheta, Nsz, Nrz, Nrr) -> (nrz, nrr) at the deck's single
            # bearing and source depth.
            p = -shd_data['pressure'][0, 0, :, :]
            field = Field(
                data=p,
                coords={'depth': receiver.depths, 'range': receiver.ranges},
                **self._result_kwargs(
                    source,
                    backend='field',
                    frequencies=float(np.atleast_1d(source.frequencies)[0]),
                    mode_coupling=self.mode_coupling if is_rd else 'none',
                    n_profiles=n_profiles,
                    **bounds,
                ),
            )
            # Same negated-Hankel convention as the COHERENT_TL / broadband
            # branches, so the complex pressure carries one phase reference.
            field.phase_reference = 'travelling_wave'
        else:
            field = read_shd_file(shd_file)
            if run_mode == RunMode.INCOHERENT_TL:
                # Opt(4:4)='I' returns SQRT(SUM(z**2)) over the per-mode
                # contributions with the range phase dropped
                # (EvaluateMod.f90:43,66); AT parks that in the complex .shd
                # slot, where its phase is an artefact. Store real dB TL so
                # the result claims only what it has.
                field.data = np.asarray(field.tl, dtype=float)
                phase_reference = None
            else:
                # field.exe emits the modal sum with a prefactor that differs
                # from Scooter's Hankel path by an overall -1 (see the broadband
                # branch above). Negate here too and tag travelling_wave so the
                # COHERENT_TL complex pressure carries the SAME phase convention
                # as the broadband / return_pressure branches and as Scooter
                # (|TL| is unchanged; this only fixes the complex phase).
                field.data = -field.data
                phase_reference = 'travelling_wave'
            self._stamp_result(
                field, source, backend='field',
                frequencies=float(np.atleast_1d(source.frequencies)[0]),
                phase_reference=phase_reference,
            )
            field.metadata['mode_coupling'] = self.mode_coupling if is_rd else 'none'
            field.metadata['n_profiles'] = n_profiles
            field.metadata.update(bounds)
        return field

    def _compute_field_via_exe(
        self, env, source, receiver,
        return_pressure=False, n_modes=None, frequencies=None, exe=None,
        run_mode=RunMode.COHERENT_TL,
    ):
        """Compute field using kraken.exe → field.exe AT pipeline.

        Parameters
        ----------
        return_pressure : bool
            If True, return complex pressure Field instead of TL.
        n_modes : int, optional
            Maximum number of modes to use during field reconstruction
            (mapped to ``MLimit`` in the FLP file).
        frequencies : ndarray, optional
            Broadband frequency vector. If given and multi-entry, kraken
            runs once with TopOpt(6)='B' producing a multi-freq .mod file
            that field.exe handles natively.
        run_mode : RunMode
            Sets field.exe ``Opt(4:4)`` and the result's phase reference —
            ``INCOHERENT_TL`` sums mode magnitudes, everything else sums
            complex modes.
        """
        fm = self._setup_file_manager()
        base_name = 'kfield'

        broadband = (
            frequencies is not None
            and len(np.atleast_1d(frequencies)) > 1
        )
        freq_vec = np.asarray(frequencies, dtype=float) if broadband else None
        if broadband and freq_vec.size > _FIELD_MAX_NFREQ:
            raise ConfigurationError(
                f"{self.model_name}: {freq_vec.size} frequencies exceed "
                f"field.exe's MaxNfreq = {_FIELD_MAX_NFREQ} "
                f"(KrakenField/field.f90:24). kraken.exe writes the longer "
                f"grid into the .mod header, then field.exe overruns its fixed "
                f"freqVec buffer reading it back.",
                remediation=(
                    f"Pass at most {_FIELD_MAX_NFREQ} frequencies in "
                    f"frequencies=, or for RunMode.TIME_SERIES shorten "
                    f"output_duration / lower sample_rate so the auto-derived "
                    f"grid is coarser. Longer bands can be run as successive "
                    f"<= {_FIELD_MAX_NFREQ}-frequency passes and concatenated."
                ),
            )

        try:
            is_rd = env.is_range_dependent
            segments = None
            profile_ranges_m = None
            n_profiles = 1

            if is_rd:
                segments, n_profiles, profile_ranges_m, max_total_depth = \
                    self._segment_env_for_field(env)
            else:
                max_total_depth = self._total_media_depth(env)

            # 1. Write the .env (multi-profile when segmented)
            bounds = self._write_field_env(
                env, source, receiver, fm, base_name,
                segments, max_total_depth, broadband, freq_vec)

            # 2. Run kraken.exe → .mod (using base-class subprocess helper)
            kraken_exe = exe if exe is not None else self._select_kraken_exe(env)
            self._log(f"Running {kraken_exe.name}...")
            self._run_and_attach_prt(
                [str(kraken_exe), base_name], fm.work_dir, base_name,
                stale_outputs=_KRAKEN_MODES_OUTPUTS)

            # 3. Write .flp file
            flp_file = fm.get_path(f'{base_name}.flp')
            option = self._build_field_option(is_rd, source, run_mode)
            pos = {
                's': {'z': source.depths},
                'r': {'z': receiver.depths, 'r': receiver.ranges},
            }
            flp_kwargs = dict(
                title=getattr(env, 'name', ''),
                n_profiles=n_profiles,
                profile_ranges_m=profile_ranges_m,
            )
            if n_modes is not None:
                flp_kwargs['M_limit'] = int(n_modes)
            write_fieldflp(flp_file, option, pos, **flp_kwargs)

            # field.exe reads <base>.sbp when Opt(3:3)='*'.
            if source.beam_pattern is not None:
                stage_source_beam_pattern(
                    source.beam_pattern, fm.get_path(f'{base_name}.sbp'))

            # 4. Run field.exe → .shd
            shd_file = self._run_field_exe(fm, base_name, option)

            field = self._assemble_field_from_shd(
                shd_file, source, receiver, is_rd, n_profiles,
                broadband, freq_vec, return_pressure, fm, base_name, run_mode,
                bounds)

            # No-propagation guard: an empty modal sum — Kraken found 0
            # trapped modes (frequency below the waveguide's modal cutoff, or
            # c_high too low) — leaves field.exe's grid untouched, which the
            # SHD reader surfaces as all-NaN (no-data). Flag it rather than
            # return a silent empty field (compute_modes raises on the same
            # case).
            # Tested on TL so it covers both the complex branches and the
            # real dB INCOHERENT_TL one: an empty sum saturates at the
            # PRESSURE_FLOOR clamp.
            tl = np.asarray(field.tl, dtype=float)
            finite = np.isfinite(tl)
            if not np.any(tl[finite] < -20.0 * np.log10(PRESSURE_FLOOR)):
                warnings.warn(
                    f"{self.model_name}: no propagating field — the modal sum "
                    f"is empty (0 trapped modes: the frequency is below the "
                    f"waveguide's modal cutoff, or c_high is too low). The "
                    f"returned field is all-NaN (no data), not a physical "
                    f"result; raise the frequency or c_high.",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
                )

            # After the guard above: the guard reads the whole TL grid, and a
            # single-range r=0 request would otherwise look like an empty
            # modal sum.
            field = self._mask_zero_range(field, receiver, source)

            self._attach_output_paths(
                field, fm.work_dir, base_name,
                primary_files=(
                    ('shd_file', '.shd'),
                    ('mod_file', '.mod'),
                ),
            )

            self._log("Kraken simulation complete")
            return field

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    # ── Broadband ───────────────────────────────────────────────────────

    def _transfer_function_for(self, env, source, receiver, frequencies,
                               n_modes, exe):
        """Complex ``H(f)`` — ``(n_depths, n_ranges, n_frequencies)`` — on the
        given grid.

        A grid of two or more frequencies goes through kraken's native
        ``TopOpt(6)='B'`` multi-frequency ``.mod``. A single bin cannot: both
        ``write_header`` and ``write_kraken_env_file`` only emit ``'B'`` for a
        vector, and AT's ``ReadfreqVec`` is not exercised at ``NFreq=1``. That
        bin is therefore solved through the narrowband pipeline at the
        requested frequency and lifted onto a length-1 frequency axis, so the
        BROADBAND contract holds for every grid size.
        """
        frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
        if frequencies.size > 1:
            return self._compute_field_via_exe(
                env, source, receiver,
                frequencies=frequencies, n_modes=n_modes, exe=exe,
            )

        f0 = float(frequencies[0])
        pinned = Source(
            depths=source.depths,
            frequencies=f0,
            source_type=source.source_type,
            beam_pattern=source.beam_pattern,
        )
        field = self._compute_field_via_exe(
            env, pinned, receiver,
            return_pressure=True, n_modes=n_modes, exe=exe,
        )
        d = field.to_dict()
        d['data'] = np.asarray(d['data'])[:, :, None]
        d['coords'] = {**d['coords'], 'frequency': frequencies}
        d['frequencies'] = frequencies
        d['metadata'] = {**d['metadata'], 'native_broadband': False}
        return Field.from_dict(d)

    def _compute_broadband_field(
        self, env, source, receiver,
        frequencies=None, n_modes=None, exe=None,
    ):
        """
        Compute broadband transfer function.

        Issues ONE kraken.exe run with ``TopOpt(6)='B'`` and the broadband
        frequency vector. kraken writes a single multi-frequency .mod
        file, and field.exe handles every frequency natively (O(N) work
        instead of O(N) subprocess startups).

        Returns
        -------
        Field
            Transfer function field with shape
            ``(n_depths, n_ranges, n_frequency)`` containing complex
            pressure. Trailing-frequency convention matches
            Bellhop/RAM/Scooter broadband outputs.
        """
        frequencies = np.atleast_1d(
            self._resolve_broadband_frequencies(source, frequencies))

        self._log(f"Broadband: {len(frequencies)} frequencies, "
                  f"{frequencies[0]:.1f}-{frequencies[-1]:.1f} Hz")

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=RunMode.BROADBAND)
        kraken_exe = exe if exe is not None else self._select_kraken_exe(env)

        # Optimistic single run — no overhead in the common all-propagating case.
        try:
            return self._transfer_function_for(
                env, source, receiver, frequencies, n_modes, kraken_exe,
            )
        except ModelExecutionError as exc:
            # field.exe can crash (ReadModes.f90) when a frequency is below the
            # waveguide's modal cutoff: kraken writes an empty mode record there
            # which also corrupts the multi-frequency .mod. The field below
            # cutoff is zero for a normal-mode model (Computational Ocean
            # Acoustics §2; below-cutoff/continuous-spectrum propagation needs a
            # wavenumber-integration model like Scooter), so compute the
            # propagating subset and zero-fill the dropped bins — preserving the
            # uniform frequency grid TIME_SERIES synthesis needs. The cutoff is
            # probed (cheap single-frequency runs) only on this failure path.
            floor = self._propagating_frequency_floor(
                env, source, receiver, frequencies, kraken_exe)
            if floor == 0:
                raise   # every frequency propagates — failure was something else
            if floor >= len(frequencies):
                raise ConfigurationError(
                    f"{self.model_name}: no propagating modes at any requested "
                    f"frequency ({frequencies[0]:.1f}-{frequencies[-1]:.1f} Hz) — "
                    f"every frequency is below the waveguide's modal cutoff.",
                    remediation="Raise the frequency band above the modal "
                    "cutoff, or use a wavenumber-integration model (Scooter) "
                    "for below-cutoff fields.",
                ) from exc
            warnings.warn(
                f"{self.model_name}: {floor} broadband frequency(ies) <= "
                f"{frequencies[floor - 1]:.2f} Hz are below the modal cutoff (no "
                f"propagating modes); their field is zero for a normal-mode "
                f"model. Computing the {len(frequencies) - floor} propagating "
                f"frequencies and zero-filling the rest.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
            tf = self._transfer_function_for(
                env, source, receiver, frequencies[floor:], n_modes, kraken_exe,
            )
            data_full = np.zeros(
                tf.data.shape[:2] + (len(frequencies),), dtype=tf.data.dtype)
            data_full[:, :, floor:] = tf.data        # zero-fill the sub-cutoff bins
            id_kwargs = tf.id_kwargs()
            id_kwargs['frequencies'] = frequencies
            return Field(
                data=data_full,
                coords={'depth': receiver.depths, 'range': receiver.ranges,
                        'frequency': frequencies},
                **id_kwargs,
            )
