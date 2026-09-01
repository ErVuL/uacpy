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
modes = kraken.compute_modes(env, source)   # Modes (k, phi)
tl    = kraken.compute_tl(env, source, receiver)       # Field (TL)

# Complex modes for elastic bottom (auto, or force backend)
modes = Kraken(backend='krakenc').compute_modes(env_elastic, source)

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
    _max_roughness, _smooth_surface,
)
from uacpy.core.bottom import _NON_GEOACOUSTIC_TYPES
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import Result, Modes, Field, PhaseReference
from uacpy.core.constants import (
    parse_boundary_type,
    DEFAULT_SOUND_SPEED,
    PRESSURE_FLOOR,
)
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, FileFormatError,
    ModelExecutionError, UnsupportedFeatureError,
)
from uacpy.io.oalib_writer import (
    write_multi_profile_env, write_fieldflp, write_kraken_env_file,
    resolve_phase_speed_bounds, plan_multi_profile_media,
    at_mesh_floor, reject_coarse_at_mesh, deck_depth, writable_layers,
    SOURCE_TYPE_CODE as _SOURCE_TYPE_CODE,
)
from uacpy.io.oalib_reader import read_shd_file, read_shd_bin, read_prt
from uacpy.io.refl_io import stage_source_beam_pattern
from uacpy.models._segmentation import segment_environment_by_range

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

#: Remedy shared by the two ways a below-cutoff run reaches the caller: the
#: binary's own empty-spectrum ERROUT (:meth:`Kraken._modes_error_message`) and
#: a mode set in which every mode is non-trapped
#: (:meth:`Kraken._check_non_trapped_modes`). Phrased as a sentence fragment so
#: both sites can splice it into their own opening clause.
_BELOW_CUTOFF_REMEDY = (
    "raise the frequency above the waveguide's modal cutoff — below it the "
    "field is carried by the continuous spectrum, which needs a "
    "wavenumber-integration model (Scooter)"
)

#: The ERROUT text both mode finders print when ``[cLow, cHigh]`` holds no mode
#: — ``Kraken/kraken.f90:962`` and ``Kraken/krakenc.f90:445``, identical in both
#: (only the 'KRAKEN' / 'KRAKENC' banner differs). Named once because two sites
#: match on it: :meth:`Kraken._modes_error_message`, which turns it into the
#: typed no-modes error, and ``_BENIGN_FORTRAN_FATALS``, which lets the
#: broadband floor search read it as "below cutoff" instead of a run failure.
_NO_MODES_ERROUT = 'No modes for given phase speed interval'

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

#: Top ``TopOpt(2:2)`` letters whose branch of ``Kraken/kraken.f90:850-867``
#: (and ``krakenc.f90:848-865``) leaves ``rho1``/``eta1Sq`` non-zero, so the
#: Kuperman-Ingenito determinant ``Del = rho1*eta2 + rho2*eta1``
#: (``Kraken/Scattering.f90:21``) is non-zero and the sea-surface roughness in
#: ``SSP%sigma(1)`` reaches the eigenvalue perturbation. Every other letter
#: falls to ``CASE DEFAULT``, which zeroes both, so ``KupIng`` returns its
#: initialised ``0.0D0`` (``Scattering.f90:17,23``) — see
#: :meth:`Kraken._drop_roughness_on_tabulated_top`.
_ROUGHNESS_BEARING_TOP_CODES = ('A', 'V', 'R')


# Sampling density for the mode TABULATION grid (the receiver-depth vector
# modes are written on), not for the internal medium mesh. Both manuals ask
# for this figure, and the vendored tree ships them as HTML:
#   ``doc/kraken.htm``: "Fine sampling (about 10 points/wavelength) is needed
#   to calculate the coupling integrals accurately."
#   ``doc/field.htm``: "a large number of receiver depths (NRD) when you do
#   the KRAKEN run. This number should be set to give about 10
#   points/wavelength."
#
# The INTERNAL mesh is a separate quantity with a different number, so the two
# must not be conflated: ``doc/EnvironmentalFile.htm`` says NMESH "should be
# about 10 per vertical wavelength in acoustic media. In elastic media ... 20
# per wavelength is a reasonable starting point", and
# ``misc/ReadEnvironmentMod.f90:103`` applies the conservative figure to every
# medium — ``deltaz = c / freq0 / 20``, "default sampling: 20 points per
# wavelength".
#
# The floor keeps a low-frequency run from getting a coarser grid than the
# historical fixed density.
MODE_POINTS_PER_WAVELENGTH = 10.0
MODE_POINTS_PER_METER_FLOOR = 1.5


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
      warns. The result is a real :class:`Field` in dB (``kind='pressure'``,
      ``unit='dB'`` — transmission loss is the dB unit of the pressure
      kind, not a kind of its own; see :attr:`Field.kind`). It
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
        exceeds :data:`_segmentation._MAX_SEGMENT_LENGTH_M` (2 km). Pass
        an explicit int to override with a uniform linspace decomposition.
    mode_points_per_meter : float, optional
        Mode-depth grid density in pts/m. ``None`` (default) derives it from
        the run as ``10 * f_max / c_min`` — the ~10 points/wavelength the
        KRAKEN and FIELD manuals require of the mode tabulation grid — with a
        1.5 pts/m floor. An explicit value is used verbatim and warns if it
        falls under that.
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
        Lower phase speed limit (m/s). None ⇒ 0.0,
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
        This forces ``backend='krakenc'``, because only complex arithmetic can
        represent a leaky eigenvalue: ``doc/kraken.htm:654-657`` has "KRAKENC
        will attempt to compute leaky modes if CHIGH exceeds the phase velocity
        of either the S-wave or P-wave speed in the half-space".

        ``False`` (the default) does **not** mean "no leaky modes". The same
        page claims KRAKEN "will (if necessary) reduce CHIGH so that only
        trapped (non-leaky) modes are computed" (``doc/kraken.htm:648-650``),
        but the vendored source only does that for an **elastic** half-space:
        ``Kraken/kraken.f90:209`` clamps cHigh to the shear speed, while the
        acoustic clamp one branch below is commented out —
        ``Kraken/kraken.f90:212``, ``! cHigh = MIN( cHigh, DBLE( HSBot%cP ) )``.
        Over a fluid seabed cHigh therefore stands as written, and the auto
        value sits 5 % past the bottom speed
        (:data:`~uacpy.core.constants.C_HIGH_FACTOR`), so an ordinary run keeps
        a few modes with ``c_p`` above it. :meth:`compute_modes` logs how many
        at ``verbose='info'``, and refuses outright when *every* returned mode
        is one of them. Default: False.
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
        ×1.05 for narrowband, ×3 for a broadband sweep. A receiver with no
        positive range gives it nothing to scale, and falls back to
        100 km — the tightest of these tolerances. ``compute_modes`` is
        always that case: it takes no receiver at all, because the
        eigenfunctions it returns are not evaluated at one, so its modes are
        converged to the 100 km target regardless of where the caller later
        propagates them.
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

    - ``c_low=None`` → ``0.0`` (KRAKEN computes cLow automatically)
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

        An elastic SURFACE counts for exactly the same reason as an elastic
        bottom: ``krakenc.f90:220-222`` folds ``HSTop%cS`` into ``cMin``
        symmetrically with ``HSBot%cS`` at ``:210-212``, so an ice canopy sets
        ``ElasticFlag`` and drags the search floor down to 0.84x its shear
        speed just as a seabed would. Measured on a 300 m water column under a
        3500/1800 m/s ice canopy: with the floor left to KRAKEN the run fails
        at 1 kHz and 2 kHz ("No modes for given phase speed interval", a
        640-byte .mod that then crashes field.exe), while the minimum
        compressional speed — 1500 m/s here — returns 58.17 and 60.24 dB.

        This floor is what makes an elastic sea surface usable at all, and it
        is why ``Kraken`` refuses one nowhere — a refuted claim, kept here so
        it is not re-derived. The claim was that krakenc.exe heap-corrupts
        (``free(): invalid pointer``) on the interfacial (Scholte) modes of a
        solid-over-liquid interface. That abort does not reproduce on the
        binaries this repository ships: the same ice canopy returns finite,
        physical TL at 50, 120, 200 and 400 Hz (64.82 / 65.03 / 82.09 /
        56.70 dB), sensible against a vacuum top. Refusing on it costs every
        Arctic environment the package's own data layer builds
        (``core/constants.py``'s ``SEA_ICE_*`` presets and
        ``data/seaice_local.py``), for a mechanism that is not the cause.

        The seabed term sweeps every compressional speed the deck carries
        (``Bottom.all_sound_speeds``) whichever medium set ``ElasticFlag``:
        a fluid sediment layer slower than the water — mud under an ice
        canopy — ducts modes of its own, and since ``krakenc.f90:230`` only
        ever raises the written floor, a floor at the water minimum would
        silently delete them.

        The water term reads ``env.ssp.data``, not
        :meth:`~uacpy.core.ssp.SoundSpeedProfile.to_pairs`: that method returns
        the **range-0 column** of a range-dependent profile by contract, while
        the value returned here is stamped into *every* profile block of the
        multi-profile deck (``write_multi_profile_env``). ``kraken.f90:230``
        and ``krakenc.f90:230`` only ever raise the written floor —
        ``cLow = MAX( cLow, cMin )`` against that profile's own ``cMin`` — so a
        floor above a profile's slowest water is never corrected downward and
        simply deletes every mode below it. Measured on a 200 m guide whose
        water runs 1500 m/s at r=0 to 1450 m/s at 10 km over a 1600/400 m/s
        elastic half-space at 200 Hz, four profiles: the range-0 column gives
        cLow = 1500 and 24/14/12/3 modes per profile, the block minimum gives
        cLow = 1450 and 24/26/28/31, a mean 8.1 dB / max 28.4 dB TL difference
        over the receiver grid (the counts include leaky modes, so they move by
        one or two with the receiver grid and the auto RMax; the *ratio* is the
        finding). Left to the default automatic segmentation, which puts 16
        profiles on that track instead of 4, the range-0 floor does not degrade
        the answer but destroys it: a profile whose whole water column sits
        below the written floor aborts the run with ``RootFinderSecant:
        Failure to converge`` followed by ``*** FATAL ERROR *** KRAKENC: No
        modes for given phase speed interval``.

        Too *low* a floor is the harmless direction, but not because of that
        ``MAX``: a derived floor is written only when something carries shear,
        and there ``krakenc.f90:229-230`` applies ``cMin = 0.85 * cMin`` first,
        so for a 400 m/s shear speed the clamp sits at 0.99 x 0.85 x 400 =
        336.6 m/s and never binds — a written 1300 passes through untouched.
        What makes it harmless is the physics: no eigenvalue exists below the
        minimum water sound speed, so ``min(SSP)`` *is* the mode floor and
        widening the interval below it finds nothing. Measured as a convergence
        ladder on the environment above, 1450 / 1400 / 1350 / 1300 agree to
        ~1e-7 dB (float32 deck round-off) at a fixed 109 modes with the slowest
        mode phase speed identical at 1450.19 m/s on every rung.

        The block minimum is a lower bound on every profile the deck carries
        because ``segment_environment_by_range`` builds each column with linear
        ``ssp.eval(range=)``, which cannot undershoot its own samples (measured
        over 4005 slices across six segmentations, including both
        extrapolation flanks: zero undershoot). One residual it cannot see,
        pre-existing and second-order: ``interp_ssp='cubic'`` / ``'pchip'``
        writes AT's ``'S'`` / ``'P'`` code and KRAKEN's own ``EvaluateSSP``
        can then dip below the tabulated samples *inside* a layer, which no
        Python-side bound over stored samples can bound. The auto default for
        a range-dependent SSP is C-linear, which cannot dip.

        The two readings agree for a 1-D profile, which is what the
        single-profile and modes paths already carry.
        """
        if self.c_low is not None:
            return float(self.c_low)
        if not env.has_elastic_bottom and not self._has_elastic_surface(env):
            # 0.0 hands the search floor to KRAKEN, which computes cLow
            # automatically (kraken.htm, Phase Speed Limits) — the right
            # choice for a fluid environment.
            return 0.0
        speeds = [float(env.ssp.data.min())]
        speeds.extend(env.bottom.all_sound_speeds())
        return min(speeds)

    def _has_elastic_surface(self, env) -> bool:
        """Whether the surface the writer will emit carries shear.

        Reduced the way the deck sees it: Kraken carries a single global top,
        so a range-dependent surface is first collapsed by the configured
        ``collapse['surface']`` method.
        """
        surface = env.surface
        if surface.is_range_dependent:
            surface = surface.collapse(self._collapse['surface'])
        return bool(surface.is_elastic)

    def _resolve_mode_points_per_meter(self, env, frequencies) -> float:
        """Mode-depth grid density (pts/m) for this run.

        ``kraken.htm`` block (9) and ``field.htm`` §(2) both require *"about
        10 points/wavelength"* on the receiver grid the modes are tabulated
        on, since that grid carries the mode shapes and the coupling
        integrals. A density fixed in pts/**metre** meets it only up to one
        frequency, so ``None`` derives it from the run instead, using the
        highest frequency and the slowest medium.

        Nothing downstream catches a grid that is too coarse:
        ``KrakenField/ReadModes.f90:78`` sets ``Tolerance = 1500/freq`` — a
        whole wavelength — so its "Modes not tabulated near requested pt."
        warning stays silent and the ``.prt`` is clean.

        The wavelength that matters in an elastic sediment is the **shear**
        one: AT meshes such a medium at ``c_s/(20·f)``
        (``misc/ReadEnvironmentMod.f90:101-103``), and ``c_p/c_s`` reaches ~8
        for a soft seabed, so sizing the tabulation grid on compressional
        speeds alone under-samples the mode shape by that factor exactly
        where it varies fastest. ``_multi_profile_n_mesh`` and
        ``Bounce._resolve_n_mesh`` already read the shear speed the same way.

        "Slowest medium" means slowest *anywhere in the deck*: one grid is
        tabulated for the whole multi-profile ``.env``, so the water term reads
        ``env.ssp.data`` — the full ``(n_depth, n_range)`` block — rather than
        :meth:`~uacpy.core.ssp.SoundSpeedProfile.to_pairs`, which returns the
        range-0 column by contract. The seabed term below already sweeps every
        range column of ``env.bottom``, so reading one SSP column would leave
        the two halves of the same reduction disagreeing about which ranges
        count.

        That correction is a rule-compliance fix, and the honest measurement
        is that its effect here is small. On a four-profile deck whose water
        runs 1500 m/s at r=0 to 1000 m/s at 10 km, at 2 kHz, the range-0
        reading gives 13.33 pts/m (6.7 points per wavelength in the slowest
        water, 2934 grid depths at dz = 0.0750 m) against 20 pts/m from the
        block minimum (4402 depths, dz = 0.0500 m) — and the two TL fields
        differ by a mean 0.004 dB / max 0.029 dB, with no receiver past
        0.1 dB. A grid sized on the range-0 column is genuinely under-sampled
        against the manuals' ~10 points/wavelength and one sized on the block
        minimum is not, but the error that buys is far from the several-dB
        regime a fixed 1.5 pts/m density produces.
        """
        f = np.atleast_1d(np.asarray(frequencies, dtype=float))
        f_max = float(np.max(f)) if f.size else 0.0
        speeds = [float(env.ssp.data.min())]
        if env.bottom is not None:
            speeds.extend(s for s in env.bottom.all_sound_speeds() if s > 0)
            speeds.extend(
                s for s in (
                    float(getattr(layer, 'shear_speed', 0.0) or 0.0)
                    for column in env.bottom.columns
                    for layer in column.layers
                ) if s > 0
            )
        c_min = min(speeds) if speeds else DEFAULT_SOUND_SPEED
        needed = (MODE_POINTS_PER_WAVELENGTH * f_max / c_min) if f_max else 0.0
        if self.mode_points_per_meter is not None:
            ppm = float(self.mode_points_per_meter)
            if needed and ppm < needed:
                warnings.warn(
                    f"Kraken(mode_points_per_meter={ppm:g}) gives "
                    f"{ppm * c_min / f_max:.2g} points per wavelength at "
                    f"{f_max:g} Hz, under the ~{MODE_POINTS_PER_WAVELENGTH:g} "
                    f"the KRAKEN and FIELD manuals require for the mode "
                    f"tabulation grid. Mode shapes and coupling integrals are "
                    f"interpolated from this grid, so the TL error is silent — "
                    f"measured 8.2 dB against Scooter at 1600 Hz. Pass "
                    f"{needed:.3g} or leave mode_points_per_meter=None to "
                    f"derive it.",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
            return ppm
        return max(MODE_POINTS_PER_METER_FLOOR, needed)

    def _effective_c_high(self):
        """The c_high the deck gets: the user's value, or the leaky-mode
        'unbounded' sentinel (CHIGH → ∞ per the Kraken doc), or ``None``
        for the writer's SSP/bottom auto-derivation."""
        if self.c_high is not None:
            return self.c_high
        return 1e9 if self.leaky_modes else None

    def _validate_phase_speed_limits(self):
        """Check ``0 <= c_low < c_high`` for whichever bound is set.

        ``c_high`` is checked on its own as well as against ``c_low``: with
        ``c_low=None`` the writer derives the lower bound from the SSP and
        the bottom, so a non-positive ``c_high`` would otherwise sail past
        every check here and die inside the Fortran with an empty spectrum.
        """
        cl = self.c_low
        ch = self._effective_c_high()
        if cl is not None and cl < 0:
            raise ConfigurationError(
                f"c_low must be >= 0, got {cl}"
            )
        if ch is not None and ch <= 0:
            raise ConfigurationError(
                f"c_high must be > 0, got {ch}: it is the top of the phase-"
                f"speed window kraken searches for modes, and no mode has a "
                f"non-positive phase speed."
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
            raise UnsupportedFeatureError(
                'Kraken',
                "the 'quad' SSP interpolation — it is Bellhop-only, the "
                "external 2-D .ssp scheme the shared EvaluateSSP has no "
                "case for",
                alternatives=["'linear' (C-linear)", "'n2linear'", "'pchip'",
                              "'cubic' / 'spline'"],
                alternatives_label='SSP interpolations',
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
        # A pinned n_mesh is checked at the deck's freq0, which is where AT
        # applies its own floor: misc/ReadEnvironmentMod.f90:103-112 sizes
        # Nneeded from freq0 alone, during the environment read, and
        # kraken.f90:75 then scales N with freq/freq0 for every swept
        # frequency — so a mesh that clears the floor at freq0 stays
        # proportionally as fine across the whole sweep. Testing max(freq)
        # instead rejected meshes the binary would have run.
        self._reject_coarse_mesh(
            env, float(np.atleast_1d(
                np.asarray(source.frequencies, dtype=float))[0]))

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

        c_high_eff = self._effective_c_high()
        cl, ch = resolve_phase_speed_bounds(env, self._c_low_for(env),
                                            c_high_eff)
        if c_high_eff is None:
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

    def _bottom_collapse_for(self, run_mode) -> str:
        """Range-reduction method the bottom will actually get for ``run_mode``.

        The two paths do not agree, and both are right for what they build.
        :meth:`_modes_single_profile` samples the r = 0 profile of every
        range-dependent quantity — SSP, bathymetry, surface — so its bottom is
        ``select_range('r0')`` too; taking the median column there would pair a
        seabed from one range with a water column from another. Every field
        path instead goes through :meth:`_project_environment`, which applies
        the declared ``collapse['bottom_range']`` policy ('median' by default).

        Guards that predict what the deck will carry have to ask here rather
        than assume one of the two.
        """
        if run_mode == RunMode.MODES:
            return 'r0'
        return self._collapse['bottom_range']

    def _reject_acoustic_below_elastic(self, env: Environment,
                                       run_mode=None) -> None:
        """Reject any acoustic medium sitting below an elastic one.

        Two failure modes, one shape — a solid-over-liquid interface inside
        the bottom stack:

        * An elastic layer (shear_speed > 0) terminated by a **fluid
          halfspace**: krakenc.exe spins forever in setup, so the caller
          would wait out the whole ``timeout`` for nothing.
        * An elastic layer with a **fluid layer under it**: krakenc.exe
          aborts (SIGABRT, "double free or corruption"). ``kraken.f90:170``
          sets ``LastAcoustic`` to the *deepest* acoustic medium, so
          ``FirstAcoustic .. LastAcoustic`` spans the elastic medium in
          between and ``Vector``'s loops (``:560-568``, ``:632-651``) walk it
          as if it were acoustic.

        Elastic half-spaces and elastic-over-elastic stacks are fine, as is a
        fluid layer *above* an elastic one.

        ``run_mode`` selects the range reduction (see
        :meth:`_bottom_collapse_for`) so the column tested is the column the
        deck will carry. Testing one fixed policy for both let a MODES run
        over a range-dependent bottom past the guard whenever the median
        column was fluid and r = 0 was not — straight into the krakenc hang —
        and refused the mirror case, which would have run.
        """
        # A bottom with no layered column anywhere cannot carry the
        # solid-over-liquid stack this guard rejects — return before any
        # reduction, so a mixed-type non-layered RD bottom (which the MODES
        # path reduces with 'r0') never hits select_range('median')'s
        # single-type check.
        if not env.bottom.is_layered:
            return
        # Inspect the column the deck will carry: projection reduces a
        # range-dependent bottom before the writer runs, so an elastic layer
        # at r > 0 can reach the deck even when the r = 0 column is fluid.
        bottom = env.bottom
        if bottom.is_range_dependent:
            bottom = bottom.select_range(self._bottom_collapse_for(run_mode))
        col = bottom.at(range=0)
        shear = [(getattr(layer, 'shear_speed', 0) or 0) > 0
                 for layer in col.layers]
        if not any(shear):
            return
        first_elastic = shear.index(True)
        fluid_below = not all(shear[first_elastic:])
        halfspace_fluid = (getattr(col.halfspace, 'shear_speed', 0) or 0) == 0
        if fluid_below:
            raise UnsupportedFeatureError(
                self.model_name,
                "a fluid sediment layer below an elastic one "
                "(solid-over-liquid interface inside the bottom stack) — "
                "kraken.f90:170 spans the elastic medium with "
                "FirstAcoustic..LastAcoustic and krakenc.exe aborts on it",
                alternatives=['Scooter', 'OAST'],
            )
        if halfspace_fluid:
            raise UnsupportedFeatureError(
                self.model_name,
                "an elastic sediment layer over a fluid halfspace "
                "(solid-over-liquid bottom interface) — krakenc.exe does not "
                "converge on it",
                alternatives=['Scooter', 'OAST'],
            )

    def _reject_rough_elastic_layer(self, env: Environment,
                                    run_mode=None) -> None:
        """Reject roughness on the interface above an elastic sediment layer.

        ``kraken.f90:169`` / ``krakenc.f90:182`` stop with *"Rough elastic
        interfaces are not allowed"* whenever an ELASTIC medium carries a
        non-zero ``SSP%sigma``, and ``sigma`` belongs to the interface at the
        **top** of its own medium — which the writer takes from that layer's
        own ``roughness`` (``oalib_writer:1072``). The model's spec advertises
        ``rough_bottom`` and ``elastic_media`` separately and both are true;
        it is only this pairing that the binaries refuse.

        A rough elastic *half-space* is fine: its ``sigma(NMedia+1)`` sits on
        the ``BotOpt`` line and feeds the Kirchhoff (``KupIng``) correction
        instead of the medium loop.
        """
        if not env.bottom.is_layered:
            return
        bottom = env.bottom
        if bottom.is_range_dependent:
            bottom = bottom.select_range(self._bottom_collapse_for(run_mode))
        for i, layer in enumerate(bottom.at(range=0).layers):
            shear = float(getattr(layer, 'shear_speed', 0.0) or 0.0)
            sigma = float(getattr(layer, 'roughness', 0.0) or 0.0)
            if shear > 0.0 and sigma != 0.0:
                raise UnsupportedFeatureError(
                    self.model_name,
                    f"roughness ({sigma:g} m) on the interface above elastic "
                    f"sediment layer {i + 1} (shear_speed={shear:g} m/s) — "
                    f"kraken.f90:169 / krakenc.f90:182 stop with 'Rough "
                    f"elastic interfaces are not allowed'",
                    alternatives=[
                        'Set that layer roughness to 0 (a rough elastic '
                        'half-space is accepted)',
                        'Scooter', 'OAST',
                    ],
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
            ppm = self._resolve_mode_points_per_meter(env, source.frequencies)
            n_pts = max(100, int(round(float(total_depth) * ppm)))
            mode_depths = np.linspace(0.0, float(total_depth), n_pts)

        # ``ranges=[0.0]`` carries no geometry, and none is available: modes
        # are depth eigenfunctions, so ``compute_modes`` takes no receiver.
        # ``_compute_rmax_m`` reads that as "nothing to scale" and writes the
        # 100 km RMax, i.e. the tightest mesh-convergence tolerance
        # (``kraken.f90:80``, ``Error·1000·RMax < 1``) rather than one keyed
        # to a range the caller never gave. ``rmax_m=`` overrides it.
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
        # A short or unreadable .mod arrives as FileFormatError:
        # ``read_modes_bin`` wraps every parse failure (its ``except
        # PARSE_ERRORS`` arm, which already contains IndexError). The empty
        # spectrum is one of those on krakenc — see the M == 0 note below —
        # so the diagnosis has to happen on this branch too, not only on the
        # mode-count one.
        try:
            modes_data = read_modes_bin(basename, frequency=0.0)
        except (FileFormatError, IndexError) as e:
            raise ModelExecutionError(
                self.model_name, return_code=0, stdout=None,
                stderr=self._modes_error_message(basename, original_error=e),
            ) from e

        # "No modes for given phase speed interval" is an ERROUT that still
        # leaves a file behind. kraken.f90:947-962 writes records 1 and 7, so
        # its .mod is a normal 4*LRecordLength*7 bytes and only the mode count
        # reports the state — this branch. krakenc.f90:432-446 writes records 1
        # and 5 of the same header, a 640-byte file the reader cannot finish,
        # so that path lands on the FileFormatError above instead.
        if int(modes_data.get('M', 0)) == 0:
            raise ModelExecutionError(
                self.model_name, return_code=0, stdout=None,
                stderr=self._modes_error_message(basename),
            )

        return modes_data

    def _non_trapped_phase_speeds(self, k, env, freq):
        """``(cp, c_bottom)`` for the modes ``k`` at ``freq`` Hz, or ``None``.

        ``cp = omega / Re(k)`` is each mode's phase speed; ``c_bottom`` is the
        half-space compressional speed above which a mode radiates into the
        seabed instead of being trapped in the duct.

        ``None`` means the comparison says nothing and no caller should act on
        it: no modes to measure; a non-geoacoustic boundary (vacuum, rigid, or
        a reflection table), which has no half-space to leak into and resolves
        to an unbounded ``c_high`` anyway (:func:`resolve_phase_speed_bounds`);
        an elastic half-space, which traps on its *shear* speed instead and is
        the one case ``kraken.f90:209`` really does clamp ``cHigh`` for; or
        ``leaky_modes=True``, where the caller asked for exactly these modes.
        """
        if self.leaky_modes:
            return None
        k = np.atleast_1d(np.asarray(k, dtype=complex))
        if k.size == 0:
            return None
        halfspace = env.bottom.halfspace_at(range=0.0)
        if halfspace.acoustic_type in _NON_GEOACOUSTIC_TYPES:
            return None
        if float(getattr(halfspace, 'shear_speed', 0.0) or 0.0) > 0.0:
            return None
        kr = np.real(k)
        with np.errstate(divide='ignore', invalid='ignore'):
            cp = 2.0 * np.pi * float(freq) / kr
        cp = cp[np.isfinite(cp) & (cp > 0.0)]
        if cp.size == 0:
            return None
        return cp, float(halfspace.sound_speed)

    def _check_non_trapped_modes(self, k, env, freq, *, exe=None,
                                 field_run=False) -> None:
        """Act on the modes whose phase speed sits above the seabed speed.

        Every mode non-trapped means the frequency is below the waveguide's
        modal cutoff and the field is carried by the continuous spectrum, so
        the modal sum answers a different problem than the one asked. The
        auto ``c_high`` sits 5 % past the bottom speed
        (:data:`~uacpy.core.constants.C_HIGH_FACTOR`), so the mode search still
        finds something there and the binary's own empty-spectrum ERROUT —
        the existing diagnosis in :meth:`_modes_error_message` — never fires.
        A modes solve raises; a field run warns, because the broadband
        sub-cutoff recovery at :meth:`_compute_broadband_field` legitimately
        drives single below-cutoff bins through this path and zero-fills them.

        On real-arithmetic ``kraken.exe`` those eigenvalues are not leaky modes
        at all. ``Kraken/BCImpedanceMod.f90:83-89`` (CASE 'A', ``cS <= 0``)
        forms ``gammaP = SQRT( x - omega2 / cP**2 )``; above the half-space
        speed the radicand is negative, ``gammaP`` is pure imaginary, and
        ``DBLE( f )`` keeps its real part = 0 — leaving ``f = 0, g = rho``,
        which is CASE 'R', the RIGID bottom (``:60-63``). What comes back is a
        rigid-bottom waveguide's spectrum plus the first-order radiation-loss
        perturbation of ``kraken.f90:766-773``. Measured: a 5 m / 150 Hz duct
        over 1700 m/s returns cp = 1730.70 m/s against the rigid-bottom
        prediction 1732.05.

        Some modes non-trapped is the ordinary default (14 modes at 200 Hz in
        100 m of water, 3 of them above the 1650 m/s seabed —
        ``docs/models/kraken.md §6.1 "Fourteen modes at 200 Hz"``), so that
        only logs.
        """
        probe = self._non_trapped_phase_speeds(k, env, freq)
        if probe is None:
            return
        cp, c_bottom = probe
        above = cp > c_bottom
        if not above.any():
            return
        if not above.all():
            self._log(
                f"{int(above.sum())} of {above.size} modes are non-trapped "
                f"(phase speed above the {c_bottom:.1f} m/s half-space speed, "
                f"up to {cp.max():.2f} m/s): the auto c_high sits 5 % past the "
                f"bottom speed and kraken.f90:212 leaves the acoustic cHigh "
                f"clamp commented out. On kraken.exe they are a rigid-bottom "
                f"solve plus a first-order radiation-loss perturbation, not "
                f"true leaky modes; leaky_modes=True computes those on "
                f"krakenc.exe."
            )
            return

        rigid = (
            " Real-arithmetic kraken.exe cannot represent a leaky mode: above "
            "the half-space speed BCImpedanceMod.f90:83-89 collapses to the "
            "rigid-bottom condition, so these eigenvalues are a rigid-bottom "
            "waveguide's plus the radiation-loss perturbation of "
            "kraken.f90:766-773."
            if exe is not None and Path(exe).stem == 'kraken' else ""
        )
        message = (
            f"Kraken returned {above.size} mode(s) and every one of them is "
            f"non-trapped: the slowest has a phase speed of {cp.min():.2f} m/s, "
            f"above the {c_bottom:.1f} m/s half-space speed, so it radiates "
            f"into the seabed rather than propagating in the duct.{rigid} The "
            f"modal sum is not a physical answer here; "
            f"{_BELOW_CUTOFF_REMEDY}."
        )
        if field_run:
            warnings.warn(
                f"{self.model_name}: {message}",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
            return
        raise ModelExecutionError(
            self.model_name, return_code=0, stdout=None, stderr=message,
        )

    def _check_field_modes_trapped(self, mod_base, env, source, exe) -> None:
        """Apply :meth:`_check_non_trapped_modes` to the ``.mod`` field.exe is
        about to sum — the only place a below-cutoff field run is visible,
        since the ``.shd`` it produces is a full, plausible-looking curve.

        Best effort: an unreadable mode file is not a diagnosis. The
        empty-spectrum ``.mod`` is precisely that — a zero-mode record breaks
        the reader's stride (see :meth:`_read_modes_file`) — and that case is
        already reported by the all-NaN guard further down
        :meth:`_compute_field_via_exe`, so a read failure here stays quiet.
        """
        from uacpy.io.modes_reader import read_modes_bin

        try:
            modes_data = read_modes_bin(str(mod_base), frequency=0.0)
        except (FileFormatError, IndexError, OSError) as e:
            self._log(f"non-trapped-mode check skipped: unreadable mode file "
                      f"({type(e).__name__}: {e})", level="debug")
            return
        self._check_non_trapped_modes(
            modes_data.get('k'), env,
            float(np.atleast_1d(source.frequencies)[0]),
            exe=exe, field_run=True)

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
            empty_spectrum = _NO_MODES_ERROUT in prt_content

            if empty_spectrum:
                error_msg += (
                    "Kraken found no mode with a phase speed inside "
                    "[c_low, c_high] at this frequency. Widen the window "
                    "(lower c_low, raise c_high, or leaky_modes=True), or "
                    f"{_BELOW_CUTOFF_REMEDY}."
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
    # layered + elastic bottom. ``bottom_range: 'median'`` picks the column
    # that represents the whole transect.
    #
    # No ``'ssp'`` entry, unlike Scooter / SPARC / OASES. The one reader of
    # ``collapse['ssp']`` is ``_project_environment``'s range-dependent-SSP
    # branch, guarded by ``not _supports_range_dependent_ssp`` — which cannot
    # be true for a model declaring ``'range_dependent_ssp'`` just above. A
    # Kraken-specific ``'mean'`` here advertised a collapse that never runs on
    # either path; inheriting ``DEFAULT_COLLAPSE``'s ``'r0'`` instead states
    # what the modes path actually does (``_modes_single_profile``) and what
    # the field path's first segment carries. OASR and Bounce omit the key on
    # the same grounds — it does not reach their answer either.
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
        collapse={'bottom_range': 'median'},
    )
    source = 'acoustics_toolbox'

    # Below the waveguide's modal cutoff KRAKEN funnels through ERROUT, but
    # "no trapped modes in this phase-speed interval" is a physical answer
    # this model surfaces itself — as an all-NaN field plus a warning from
    # ``_compute_field_via_exe``, a zero-fill from the broadband floor search,
    # or a typed error from ``compute_modes`` — not a run failure.
    _BENIGN_FORTRAN_FATALS: tuple = (_NO_MODES_ERROUT,)

    def __init__(
        self,
        mode_points_per_meter: Optional[float] = None,
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
        # c_low default 0.0 → KRAKEN computes cLow
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
        # None → derive at run() from receiver.range_max. Zero or negative
        # short-circuits that test: ``kraken.f90:80`` leaves the refinement
        # loop as soon as ``Error*1000*RMax < 1`` and ``Error`` starts at
        # 1e10, so RMax <= 0 makes it true on the first, coarsest mesh — no
        # Richardson extrapolation, measured 2.37 dB max |ΔTL| against the
        # default, at exit 0. Bounce rejects the same value.
        if rmax_m is not None and float(rmax_m) <= 0.0:
            raise ConfigurationError(
                f"Kraken(rmax_m={rmax_m}) must be > 0: RMax scales the mode "
                f"solver's mesh-convergence test (kraken.f90:80, "
                f"Error*1000*RMax < 1), and a non-positive value satisfies it "
                f"on the coarsest mesh, so the modes come back unrefined and "
                f"unextrapolated with nothing in the .prt to say so. Pass the "
                f"longest range the modes will be propagated to, or None to "
                f"derive it from receiver.ranges."
            )
        self.rmax_m = float(rmax_m) if rmax_m is not None else None
        # mode_depth_grid overrides compute_modes's dense grid; None → density.
        self.mode_depth_grid = (
            np.asarray(mode_depth_grid, dtype=float)
            if mode_depth_grid is not None else None
        )
        # leaky_modes drives CHIGH to the 'unbounded' sentinel at deck time
        # (_effective_c_high); pinning c_high alongside it is a contradiction
        # refused here rather than resolved by discarding the pinned value
        # silently — and c_high stays
        # stored verbatim so copy()/repr() round-trip the constructor args.
        if leaky_modes and c_high is not None:
            raise ConfigurationError(
                f"Kraken(leaky_modes=True) needs CHIGH at the 'unbounded' "
                f"sentinel (1e9 m/s) for kraken/krakenc to search leaky "
                f"modes; an explicit c_high={c_high} contradicts it. Pass "
                f"one or the other."
            )
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

        # The resolved kraken.exe path lives in ``self._exe``;
        # ``_select_kraken_exe`` may swap in krakenc.exe per-env.
        self._exe = self._resolve_executable(
            executable,
            lambda: self._find_executable_in_paths(
                'kraken.exe', bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Kraken',
            ),
        )

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
        # Source geometry letter lands in field.exe Opt(1:1), field.f90:70-79.
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
        ``'krakenc'`` for elastic media, leaky modes (complex eigenvalues)
        or a tabulated reflection coefficient, ``'kraken'`` otherwise.
        Round-trips with ``backend=`` and mirrors ``RAM.select_backend``
        (``run_mode`` accepted for signature parity; it does not affect the
        modes-binary choice). Pure introspection of the dispatch — no disk
        access; :meth:`_select_kraken_exe` resolves the named binary (and
        checks a ``'precalc'`` bottom's ``.irc`` header) on the run path.

        Raises
        ------
        UnsupportedFeatureError
            For a ``'precalc'`` sea surface — no Kraken-family binary reads
            a top ``.irc`` table.
        ConfigurationError
            When ``backend='kraken'`` is forced on an environment
            kraken.exe cannot answer (elastic media / leaky modes, or a
            tabulated reflection coefficient).
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
            return 'krakenc'
        return 'kraken'

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

        The roughness drop runs last, on the resolved surface, so it covers
        both ways a tabulated top reaches the deck — this rewrite and a
        user-built ``Surface(acoustic_type='file')``.
        """
        env = super()._project_environment(env)
        if self.top_reflection_file is not None:
            from uacpy.core.bottom import BoundaryProperties
            from uacpy.core.surface import Surface
            if not self.top_reflection_file.exists():
                raise ConfigurationError(
                    f"top_reflection_file not found: {self.top_reflection_file}"
                )
            # Replace only the boundary condition; the roughness carries over
            # so the drop below sees the value the user set, and reports it.
            env.surface = Surface(properties=[BoundaryProperties(
                acoustic_type='file',
                reflection_file=str(self.top_reflection_file),
                roughness=env.surface.roughness)])
        return self._drop_roughness_on_tabulated_top(env)

    def _drop_roughness_on_tabulated_top(self, env):
        """``env`` with the sea-surface roughness zeroed, and a warning, when
        the top boundary condition cannot carry it.

        ``write_ssp_section`` in ``oalib_writer.py`` writes
        ``env.surface.roughness`` as ``SSP%sigma(1)`` on the water mesh line
        for every ``TopOpt`` letter, and ``Kraken/kraken.f90:902`` feeds that
        slot into ``KupIng``. Whether it reaches the answer is decided one
        branch earlier: ``kraken.f90:850-867`` selects on ``HSTop%BC``, and the
        ``CASE DEFAULT`` a tabulated top ``'F'`` lands in sets
        ``rho1 = eta1Sq = 0``. ``Kraken/Scattering.f90:21`` then forms
        ``Del = rho1*eta2 + rho2*eta1``, which is exactly zero
        (``ScatterRoot(0) = 0``), the ``IF ( Del /= 0.0D0 )`` at
        ``Scattering.f90:23`` is false, and ``KupIng`` returns the ``0.0D0`` it
        was initialised to at ``Scattering.f90:17``. ``krakenc.f90:848-865,899``
        is the same shape. Measured on a 100 m Pekeris guide at 100 Hz with a
        19-angle ``.trc`` top: ``roughness=0.5`` against ``0.0`` moves the field
        by exactly 0.0 under a tabulated top and by 3.3 % under a vacuum one.

        Dropping it here rather than declaring ``rough_surface`` conditionally
        keys the decision on the resolved environment, which is what decides
        the ``TopOpt`` letter — one ``Kraken`` instance can be run against a
        tabulated-top env and a vacuum-top one, and only the first loses the
        roughness.
        """
        sigma = _max_roughness(env.surface.properties)
        if not sigma:
            return env
        acoustic_type = env.surface.acoustic_type
        code = parse_boundary_type(acoustic_type).to_acoustics_toolbox_code()
        if code in _ROUGHNESS_BEARING_TOP_CODES:
            return env
        env.surface = _smooth_surface(env.surface)
        warnings.warn(
            f"{self.model_name} cannot apply sea-surface roughness to a "
            f"{acoustic_type!r} top boundary: kraken.f90:864-866 zeroes the "
            f"density and vertical wavenumber a tabulated top has none of, so "
            f"the Kuperman-Ingenito scatter term (Scattering.f90:23) is "
            f"identically zero. env.surface.roughness={sigma:g} m was dropped "
            f"rather than written into a deck that discards it. Use a vacuum, "
            f"rigid or half-space surface to keep it.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
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
        """The modes binary for the backend :meth:`select_backend` names —
        the run-path half of the dispatch, which touches the disk (the
        executable lookup, and the ``.irc`` header check) where the pure
        name decision never does.
        """
        backend = self.select_backend(env)
        # A 'precalc' bottom is staged verbatim as <root>.irc; a table in the
        # wrong layout (typically a theta/|R|/phase angle table) aborts the
        # binary with a bare Fortran backtrace, so the header is checked here,
        # ahead of any launch.
        self._reject_malformed_irc_bottom(env)
        if backend == 'krakenc':
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
            mode magnitudes and returns real dB TL (``kind='pressure'``,
            ``unit='dB'``) in ``Field.data`` with no phase reference — a
            magnitude sum has no phase, so the complex slot AT parks it in
            is not preserved.
            Bellhop's ``INCOHERENT_TL`` instead keeps complex Pa in
            ``.data``; the two engines agree on ``.db``, the uniform
            cross-engine surface for magnitude-sum results.
        frequencies : ndarray, optional
            Frequency vector (Hz) for native broadband computation. Only a
            multi-element grid selects a broadband run on its own: it uses
            TopOpt(6)='B' so kraken writes one multi-frequency .mod file and
            field.exe handles all frequencies in a single pass. A
            single-element grid leaves the default run mode at
            ``COHERENT_TL``, which reads nothing from ``frequencies`` and
            warns that it was ignored (``_warn_ignored_run_kwargs``); pass
            ``run_mode=RunMode.BROADBAND`` explicitly to solve that one
            frequency through the narrowband pipeline and get it back on a
            length-1 frequency axis.
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
        self._require_run_triple(env, source, receiver)
        # Default run mode: BROADBAND only for a frequencies= vector of more
        # than one element, else single-frequency coherent TL — so a length-1
        # vector defaults to COHERENT_TL and is reported as ignored below.
        smart_default = (
            RunMode.BROADBAND
            if frequencies is not None and len(np.atleast_1d(frequencies)) > 1
            else RunMode.COHERENT_TL
        )
        run_mode = self._resolve_run_mode(run_mode, default=smart_default)

        # Resolved first because the elastic-media guards test the reduced
        # bottom column, and MODES reduces it differently from the field
        # paths (_bottom_collapse_for).
        self._reject_acoustic_below_elastic(env, run_mode)
        self._reject_rough_elastic_layer(env, run_mode)

        # Early gate: coupled-mode field calculations cannot be
        # combined with incoherent mode addition. ``KrakenField/field.f90:
        # 125-129`` calls ERROUT on Opt(2:2)='C' + Opt(4:4)='I', which surfaces
        # in Python as an opaque "no .shd file" error. Fail loudly up front.
        # Gate on the range dependence that survives projection: bathymetry
        # and SSP drive the multi-profile (coupled) deck, while a
        # range-dependent bottom or surface alone is collapsed to one column
        # and yields a single-profile run field.exe accepts.
        if (
            run_mode == RunMode.INCOHERENT_TL
            and (env.has_range_dependent_bathymetry
                 or env.has_range_dependent_ssp)
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

        if run_mode == RunMode.MODES:
            # Modes only need the modes binary, never field.exe. The elastic
            # sub-bottom receiver partition below is a field-evaluation concern
            # (and would emit a spurious NaN-receivers warning here), so it is
            # skipped entirely for the modes solve.
            return self._run_modes(
                env, source, receiver, n_modes=self.n_modes, exe=kraken_exe)

        # field.exe cannot evaluate the field inside an elastic medium: the
        # binary tabulates the eigenvector over the acoustic media only, so
        # depths below the last acoustic node come back as a straight-line
        # extrapolation of the column above (:meth:`_elastic_depth_intervals`).
        # A *fluid* sediment layer above the elastic one evaluates perfectly
        # well, so the exclusion is per medium rather than "everything under
        # the water column"; a fluid layer below one was already refused by
        # _reject_acoustic_below_elastic above, along with the
        # elastic-over-fluid-halfspace stack that hangs krakenc.exe.
        # The column measured is the one the deck will carry: projection
        # reduces a range-dependent bottom, so columns[0] of the raw env can be
        # fluid while the written deck is elastic (or vice versa).
        deck_bottom = env.bottom
        if deck_bottom.is_range_dependent:
            deck_bottom = deck_bottom.select_range(
                self._bottom_collapse_for(run_mode))
        elastic_spans = (
            self._elastic_depth_intervals(env, deck_bottom.at(range=0.0))
            if deck_bottom.is_layered else []
        )
        rcv, keep = self._partition_elastic_subbottom(env, receiver, elastic_spans)

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
        path segments RD natively, the modes solve cannot).

        r = 0 is applied to every quantity here, overriding the configured
        ``collapse`` methods rather than consulting them (see also
        :meth:`_bottom_collapse_for`): the source sits in one real column, and
        taking its SSP from one reduction while the bathymetry, bottom and
        surface came from another would assemble a waveguide that exists at no
        range at all. The warning names whichever configured method the
        sampling dropped, so a setting that was asked for and not applied says
        so instead of passing silently.
        """
        if not env.is_range_dependent:
            return env
        overridden = [
            f"collapse[{key!r}]={self._collapse[key]!r}"
            for key, is_rd in (
                ('ssp', env.has_range_dependent_ssp),
                ('bathymetry', env.bathymetry.is_range_dependent),
                ('bottom_range',
                 env.bottom is not None and env.bottom.is_range_dependent),
                ('surface', env.surface.is_range_dependent),
            )
            if is_rd and self._collapse[key] != 'r0'
        ]
        dropped = (f" This drops {', '.join(overridden)}: the modes solve "
                   f"applies no collapse method, taking every quantity from "
                   f"the source's own column." if overridden else '')
        warnings.warn(
            f"{self.model_name}: normal modes are range-independent; sampling "
            f"the r=0 profile of the range-dependent environment.{dropped}",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
        ssp = env.ssp.collapse('r0') if env.has_range_dependent_ssp else env.ssp
        bottom = env.bottom
        if bottom is not None and bottom.is_range_dependent:
            bottom = bottom.select_range('r0')
        # Carry the full context of the original env: altimetry passes
        # through untouched so ``_project_environment`` still sees it and
        # emits its collapse disclosure, and the geolocation / date /
        # provenance fields survive onto the reduced env.
        reduced = Environment(
            name=env.name,
            bathymetry=float(env.bathymetry.eval(range=0.0)),
            ssp=ssp,
            altimetry=env.altimetry,
            bottom=bottom,
            surface=env.surface.collapse('r0'),
            absorption=env.absorption,
            location=env.location,
            transect=env.transect,
            date=env.date,
        )
        reduced.data_sources = env.data_sources
        return reduced

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
            # Before any n_modes cap: the reader returns the modes in order of
            # decreasing wavenumber, i.e. increasing phase speed, so if the
            # full set is entirely non-trapped every prefix of it is too.
            self._check_non_trapped_modes(
                modes.get('k'), env, float(source.frequencies[0]),
                exe=kraken_exe)
            self._mask_elastic_mode_depths(modes, env)

            field = self._build_modes_field(
                modes, n_modes, source, backend_exe=kraken_exe, bounds=bounds,
            )
            self._attach_output_paths(
                field, fm.work_dir, base_name,
                primary_files=(('mod_file', '.mod'),),
            )
            return field
        finally:
            fm.finish()

    def _mask_elastic_mode_depths(self, modes, env) -> None:
        """NaN the eigenfunction samples that fall inside an elastic medium.

        The modes path sizes its depth grid on the whole media stack
        (``_compute_modes_impl`` → ``_total_media_depth``), but the binary
        tabulates the eigenvector over the ACOUSTIC media only, and the
        samples below the last acoustic node are a straight-line
        extrapolation of the two above it — see
        :meth:`_elastic_depth_intervals`. The half-space record still names
        the sediment base as the domain bottom, so nothing downstream marks
        them. Mark them here, with the same no-data policy the field paths
        apply through :meth:`_partition_elastic_subbottom`.
        """
        phi = modes.get('phi')
        z = modes.get('z')
        if phi is None or z is None:
            return
        z = np.atleast_1d(np.asarray(z, dtype=float))
        if z.size == 0 or not env.bottom.is_layered:
            return
        bottom = env.bottom
        if bottom.is_range_dependent:
            bottom = bottom.select_range(
                self._bottom_collapse_for(RunMode.MODES))
        spans = self._elastic_depth_intervals(env, bottom.at(range=0.0))
        if not spans:
            return
        mask = np.zeros(z.shape, dtype=bool)
        for top, base in spans:
            mask |= (z > top) & (z <= base)
        if not mask.any():
            return
        phi = np.array(phi, dtype=complex)
        phi[mask, ...] = np.nan
        modes['phi'] = phi
        warnings.warn(
            f"{self.model_name}: {int(mask.sum())} mode-shape depth(s) lie in "
            "an elastic sub-bottom medium, which kraken/krakenc does not "
            "tabulate — the samples the binary returns there are a linear "
            "extrapolation of the acoustic column, not a mode shape. "
            "Returning NaN at those depths. Pass "
            "Kraken(mode_depth_grid=...) to keep the grid in the acoustic "
            "media, or use Scooter / OAST for the elastic sub-bottom.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

    @staticmethod
    def _elastic_depth_intervals(env, column):
        """Depth intervals ``(top, bottom]`` whose mode samples are not written.

        The deck's AT media are the water column, then each layer of ``column``
        at its written thickness (:func:`deck_depth`); the half-space below
        them is a boundary condition, not a medium.

        An elastic medium is not *mis*-tabulated in a kraken/krakenc ``.mod``;
        it is absent. ``kraken.f90:592`` builds the mode's depth vector over
        ``FirstAcoustic .. LastAcoustic`` only, and the file's ``Material``
        record (``:594``) is written over the same span — so
        ``ReadModes.f90:246-250`` never sees an ``'ELASTIC'`` entry, ``TufLuk``
        stays false and the stress-displacement compaction at ``:296-331``
        (a KRAKEL output format, which a kraken/krakenc ``.mod`` can never
        carry) is never reached. What the caller gets instead is
        ``calculateweights.f90``'s documented out-of-domain extrapolation:
        past the last acoustic node the index sticks at ``Nx-1`` and the
        weight runs above 1, so ``PhiTab`` (``kraken.f90:674``) continues the
        final two acoustic samples in a straight line. The acoustic media
        above the elastic one are unaffected, which is why the exclusion is
        per medium rather than "everything under the water column".

        Open at the top and closed at the bottom because an interface depth is
        tabulated twice, once per adjoining medium, and
        ``calculateweights.f90:43-49`` brackets it with ``w = 1`` on the upper
        copy: a receiver at an elastic medium's top interface reads the
        acoustic sample above it, one at its bottom interface reads the
        elastic sample.
        """
        spans = []
        top = deck_depth(float(env.depth))
        layers = writable_layers(column) if column is not None else []
        for i, layer in enumerate(layers):
            bottom = deck_depth(top + float(layer.thickness))
            if (getattr(layer, 'shear_speed', 0.0) or 0.0) > 0.0:
                # A receiver below the deepest medium is extrapolated off
                # Phi(NTot) (``ReadModes.f90:279``), the last tabulated
                # sample — so an elastic bottom-most medium takes the
                # half-space under it with it.
                last = (i == len(layers) - 1)
                spans.append((top, float('inf') if last else bottom))
            top = bottom
        return spans

    def _partition_elastic_subbottom(self, env, receiver, elastic_spans):
        """Split receivers into depths field.exe can evaluate and depths inside
        an elastic medium, which it cannot. Returns ``(compute_receiver,
        keep_mask)`` with ``keep_mask`` marking evaluable depths in the
        original ordering, or ``(receiver, None)`` when no split is needed.

        ``elastic_spans`` is the ``(top, bottom]`` interval list from
        :meth:`_elastic_depth_intervals`. ``True`` stands for the whole
        sub-bottom, for a caller that knows an elastic medium is present but
        not where it sits.
        """
        if elastic_spans is True:
            elastic_spans = [(float(env.depth), float('inf'))]
        if not elastic_spans:
            return receiver, None
        depths = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
        keep = np.ones(depths.shape, dtype=bool)
        for top, bottom in elastic_spans:
            keep &= ~((depths > top) & (depths <= bottom))
        if keep.all():
            return receiver, None
        warnings.warn(
            f"{self.model_name}: {int((~keep).sum())} receiver depth(s) lie in "
            "an elastic sub-bottom medium, where field.exe cannot evaluate "
            "the field; returning NaN there. Depths in a fluid sediment layer "
            "*above* the elastic one are computed normally (a fluid layer "
            "below it is refused up front — see "
            "``_reject_acoustic_below_elastic``).",
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
        no ``1/√r`` and are left alone, as they are in both codes. Column
        masking and the warning come from the shared
        :meth:`PropagationModel._mask_zero_range_columns`.
        """
        if source.source_type != 'point':
            return field
        masked = self._mask_zero_range_columns(
            field.data, receiver.ranges,
            'the point-source cylindrical-spreading factor 1/sqrt(r)')
        if masked is not field.data:
            field.data = masked
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
        time.

        Only the *below-cutoff* signatures are read as zero: an unreadable or
        absent ``.mod`` after a clean binary exit, and the binary's own empty-
        spectrum ERROUT. Anything else — a missing binary, a crash, a full
        disk — propagates, because the caller turns a zero here into "every
        frequency is below the waveguide's modal cutoff" with the remediation
        "raise the frequency band", which would bury the real cause.
        """
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
        except (FileFormatError, IndexError, FileNotFoundError) as e:
            # The binary ran; the .mod it left is a zero-mode record the
            # reader cannot stride over (or was never written) — the physical
            # answer is "no modes here".
            self._log(f"_count_modes_at_freq({float(freq):g} Hz): unreadable "
                      f"mode file ({type(e).__name__}: {e}); reading it as "
                      f"below cutoff.", level="debug")
            return 0
        except ModelExecutionError as e:
            # ``_raise_on_fortran_fatal`` already lets the empty-spectrum
            # ERROUT through as a physical outcome; this covers the paths that
            # surface the same banner as an error instead.
            if not any(m in str(e) for m in self._BENIGN_FORTRAN_FATALS):
                raise
            self._log(f"_count_modes_at_freq({float(freq):g} Hz): empty "
                      f"spectrum reported; reading it as below cutoff.",
                      level="debug")
            return 0
        finally:
            fm.finish()

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

    def _segment_env_for_field(self, env, freq=None):
        """Segment a range-dependent env into per-range profiles for the
        multi-profile kraken field run.

        Returns ``(segments, n_profiles, profile_ranges_m, max_total_depth)``.
        ``max_total_depth`` is the shared bottom ``write_multi_profile_env``
        will declare, taken from the same planner the writer uses so the two
        cannot drift apart: the mode-tabulation grid is built to span
        ``[0, max_total_depth]``, and ``EvaluateCMMod.f90:313`` stops a coupled
        run outright unless that grid ends *exactly* on the declared bottom.
        """
        segments = segment_environment_by_range(
            env, n_segments=self.n_segments, freq=freq)
        n_profiles = len(segments)
        _, max_total_depth, _ = plan_multi_profile_media(segments)

        profile_ranges_m = np.array([s[0] for s in segments])
        self._log(f"Range-dependent: {n_profiles} profiles, "
                  f"mode_coupling={self.mode_coupling}")
        return segments, n_profiles, profile_ranges_m, max_total_depth

    def _multi_profile_n_mesh(self, segments, freq) -> int:
        """``NG`` mesh count written on every medium line of a multi-profile
        ``.env``.

        0 (the default ``n_mesh``) asks KRAKEN to size each medium of each
        profile itself, at 20 points per wavelength with a 10-point floor
        (``misc/ReadEnvironmentMod.f90:99-110``). Per-profile meshes are
        legal: the ``.mod`` record length has no mesh term
        (``kraken.f90:587`` / ``krakenc.f90:629`` — it is set by the
        frequency count, the source/receiver tabulation and the padded
        media count, all identical across profiles).

        A pinned ``n_mesh`` is honoured on every medium of every profile,
        after the ``Nneeded / 2`` floor check the same reader enforces.
        That floor is measured per medium off the media the deck actually
        carries (:meth:`_multi_profile_media`); crucially AT takes the
        wavelength from a medium's **shear** speed wherever one is set,
        and an elastic sediment's shear wavelength can be an order of
        magnitude shorter than its compressional one.
        """
        if self.n_mesh <= 0:
            return 0
        floor = at_mesh_floor(self._multi_profile_media(segments), freq)
        if self.n_mesh < floor:
            raise ConfigurationError(
                f"Kraken(n_mesh={self.n_mesh}) is below the {floor} mesh "
                f"points misc/ReadEnvironmentMod.f90:110-112 requires for the "
                f"coarsest medium of this range-dependent environment at "
                f"{freq:.4g} Hz; the run would stop with 'Mesh is too coarse'.",
                remediation=f"Pass n_mesh >= {floor}, or leave n_mesh=0 to "
                            f"let KRAKEN size each medium of each profile "
                            f"itself.",
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
        ppm = self._resolve_mode_points_per_meter(
            env, freq_vec if freq_vec is not None else source.frequencies)
        n_mode_depths = max(100, int(max_total_depth * ppm))
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

            n_mesh_deck = self._multi_profile_n_mesh(
                segments, float(source.frequencies[0]))

            if broadband:
                # ``write_multi_profile_env`` has no broadband form; refuse
                # rather than drop the frequency vector. Both the env and the
                # run mode are legal on their own — it is Kraken's deck that
                # cannot carry the pair.
                raise UnsupportedFeatureError(
                    'Kraken',
                    "range-dependent broadband runs — the multi-profile "
                    "deck has no broadband form",
                    alternatives=["a single source frequency",
                                  "a range-independent environment"],
                    alternatives_label='inputs',
                )

            c_low = self._c_low_for(env)
            write_multi_profile_env(
                filepath=env_file,
                segments=segments,
                source=source,
                receiver=receiver_for_modes,
                interp_ssp=self.interp_ssp,
                n_mesh=n_mesh_deck,
                c_low=c_low,
                c_high=self._effective_c_high(),
                rmax_m=rmax_m,
            )
            # cLow and RMax are written identically into every profile block;
            # an unpinned cHigh resolves per profile from that profile's SSP
            # and half-space, so the widest of them is what bounds the run.
            c_high = max(
                resolve_phase_speed_bounds(
                    seg_env, c_low, self._effective_c_high())[1]
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
            # field.f90:228 writes 'Field completed successfully' before the
            # teardown block whose deallocation can fail, so that line is what
            # separates a benign non-zero exit from a run that died mid-field.
            # Without this check an abort is downgraded to a warning and the
            # truncated .shd surfaces as a misleading "malformed file" error.
            if not self._field_reached_completion(fm.work_dir):
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

    def _field_reached_completion(self, work_dir) -> bool:
        """Whether field.exe got past its last field write.

        ``field.f90:228`` writes ``'Field completed successfully'`` after
        ``FreqLoop`` closes and before the clean-up block whose deallocation is
        the known benign failure, so the line is present for a teardown-only
        error and absent for a run that died while computing. Like
        :meth:`_raise_on_field_fatal` this reads ``field.prt``, whose name
        ``field.f90:44`` hard-codes.
        """
        prt = read_prt(Path(work_dir) / f'{_FIELD_PRT_ROOT}.prt')
        return bool(prt) and 'field completed successfully' in prt.lower()

    def _attach_field_prt_path(self, result, fm) -> None:
        """Attach field.exe's diagnostic log as
        ``result.metadata['field_prt_file']``, iff the scratch survives
        (``cleanup=False``) and the file exists.

        ``field.f90:44`` hard-codes the log name ``field.prt``
        (``_FIELD_PRT_ROOT``), so :meth:`_attach_output_paths` — which keys
        ``prt_file`` on ``<base_name>.prt``, here the modes binary's
        ``kfield.prt`` — never sees it, and its ``primary_files`` tuple
        cannot carry a root that differs from ``base_name``.
        """
        if self.cleanup:
            return
        field_prt = fm.get_path(f'{_FIELD_PRT_ROOT}.prt')
        if field_prt.exists():
            result.metadata['field_prt_file'] = str(field_prt)

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
        from Scooter's Hankel path by an overall -1, negated here — times
        ``exp(-i*pi/4)`` for a line source — so every coherent branch carries
        the ``'travelling_wave'`` phase reference, upcast to complex128 (the
        ``.shd`` payload is complex64) so every uacpy engine returns one
        dtype. ``RunMode.INCOHERENT_TL`` instead yields real dB TL with no
        phase reference — a magnitude sum has no phase to reference."""
        # EvaluateMod.f90:34 applies one prefactor i*SQRT(2*pi)*EXP(i*pi/4)
        # to every source geometry; its pi/4 is the stationary-phase term of
        # the POINT-source Hankel asymptote. The 2-D line-source field has no
        # such term — its exact kernel already integrates to the modal sum —
        # so field.exe's 'X' branch comes out exp(+i*pi/4) ahead of Scooter's
        # line-source transform and needs the same exp(-i*pi/4) Bellhop
        # applies via _LINE_SOURCE_PHASE. 'point' / 'scaled' need only the
        # overall -1 shared by every branch.
        phase_corr = np.complex128(-1.0)
        if source.source_type == 'line':
            phase_corr = -np.exp(-1j * np.pi / 4.0)
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
                # field.exe already marches the engineering carrier
                # exp(-ikr) (EvaluateMod.f90:42); only its constant is off.
                # Its prefactor (EvaluateMod.f90:34) is i·√(2π)·e^{iπ/4}
                # = -√(2π)·e^{-iπ/4}, whereas the point-source modal sum
                # normalized to 1 m under this carrier carries
                # +√(2π)·e^{-iπ/4} in that slot (the e^{-iπ/4}/√(8π) of
                # the standard sum times the 4π the 1 m free-field
                # reference strips) — the form Scooter's Hankel path
                # returns. field.exe is therefore the wanted field times
                # -1 (times e^{iπ/4} for a line source), so ``phase_corr``
                # aligns the two; conjugating instead would flip the
                # already-correct carrier sign.
                p_stack[:, :, i_freq] = phase_corr * shd_i['pressure'][0, 0, :, :]
            field = Field(
                data=p_stack,
                coords={
                    'depth': receiver.depths,
                    'range': receiver.ranges,
                    'frequency': freqs_read,
                },
                phase_reference=PhaseReference.TRAVELLING_WAVE,
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
            # bearing and source depth; complex128 like every other engine.
            p = phase_corr * np.asarray(
                shd_data['pressure'][0, 0, :, :], dtype=np.complex128)
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
            field.phase_reference = PhaseReference.TRAVELLING_WAVE
        else:
            field = read_shd_file(shd_file)
            if run_mode == RunMode.INCOHERENT_TL:
                # Opt(4:4)='I' returns SQRT(SUM(z**2)) over the per-mode
                # contributions with the range phase dropped
                # (EvaluateMod.f90:43,66); AT parks that in the complex .shd
                # slot, where its phase is an artefact. Store real dB TL so
                # the result claims only what it has.
                field.data = np.asarray(field.db, dtype=float)
                phase_reference = None
            else:
                # field.exe emits the modal sum with a prefactor that differs
                # from Scooter's Hankel path by an overall -1 — times
                # e^{+iπ/4} for a line source (see the broadband branch
                # above). Apply ``phase_corr`` here too, upcast to
                # complex128, and tag travelling_wave so the COHERENT_TL
                # complex pressure carries the SAME phase convention and
                # dtype as the broadband / return_pressure branches and as
                # Scooter (|TL| is unchanged; this only fixes the complex
                # phase).
                field.data = phase_corr * np.asarray(
                    field.data, dtype=np.complex128)
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

    def _warn_krakenc_incoherent_sum(self, run_mode, kraken_exe,
                                     n_profiles: int) -> None:
        """Warn when field.exe's incoherent branch will square complex mode
        contributions without taking their magnitude first.

        ``field.f90:202-203`` picks the evaluator by profile count. The
        single-profile one, ``EvaluateMod.f90:66``, computes
        ``SQRT(SUM(z**2))`` — no ``ABS`` — which equals the energy sum
        ``SQRT(SUM(|z|**2))`` only for real mode functions, so krakenc's
        complex ``phi`` and ``k`` leave cross-mode phase inside the square.
        The multi-profile adiabatic evaluator, ``EvaluateADMod.f90:110``,
        uses ``SQRT(SUM(ABS(...)**2))`` and is sound on either backend; the
        multi-profile coupled one is unreachable here because
        ``field.f90:125-129`` refuses 'C' with 'I' and :meth:`run` rejects
        that pairing up front.
        """
        if run_mode != RunMode.INCOHERENT_TL:
            return
        if not Path(kraken_exe).name.startswith('krakenc'):
            return
        if int(n_profiles) > 1:
            return
        warnings.warn(
            f"{self.model_name}: INCOHERENT_TL on the krakenc backend is "
            "not a strict incoherent sum for a range-independent run. "
            "field.exe sends the single-profile case to EvaluateMod.f90:66, "
            "whose Opt(4:4)='I' branch computes SQRT(SUM(z**2)) over the "
            "per-mode contributions — the energy sum SQRT(SUM(|z|**2)) only "
            "for real mode functions, and krakenc's phi and k are complex. "
            "Use backend='kraken' where the environment allows it, or "
            "RunMode.COHERENT_TL.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

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
        broadband = (
            frequencies is not None
            and len(np.atleast_1d(frequencies)) > 1
        )
        freq_vec = np.asarray(frequencies, dtype=float) if broadband else None
        if broadband and source.beam_pattern is not None:
            # field.f90:191 allocates the beam-pattern work arrays inside
            # FreqLoop under `SBPFlag == '*' .AND. iS == 1` but deallocates them
            # only after the loop closes (:226), so the second frequency
            # re-allocates an already-allocated array and gfortran terminates.
            # The gate is the frequency count reaching field.exe, not the
            # caller's run_mode: the multi-frequency path reaches here with the
            # default COHERENT_TL. Dropping the pattern would silently change
            # the source's directivity, so this raises.
            raise UnsupportedFeatureError(
                self.model_name,
                f"a source beam pattern on a multi-frequency run "
                f"({freq_vec.size} frequencies) — field.exe re-allocates its "
                f"beam-pattern arrays once per frequency "
                f"(KrakenField/field.f90:191) and aborts on the second one",
                ["one frequency per call, which accepts the pattern",
                 "Source(beam_pattern=None) for a multi-frequency run"],
                alternatives_label='options',
            )
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

        # Created only after the guards above, so a rejected call does not
        # leave an orphaned scratch directory behind.
        fm = self._setup_file_manager()
        base_name = 'kfield'

        try:
            is_rd = env.is_range_dependent
            segments = None
            profile_ranges_m = None
            n_profiles = 1

            if is_rd:
                segments, n_profiles, profile_ranges_m, max_total_depth = \
                    self._segment_env_for_field(
                        env,
                        freq=(freq_vec if freq_vec is not None
                              else source.frequencies))
            else:
                max_total_depth = self._total_media_depth(env)

            # 1. Write the .env (multi-profile when segmented)
            bounds = self._write_field_env(
                env, source, receiver, fm, base_name,
                segments, max_total_depth, broadband, freq_vec)

            # 2. Run kraken.exe → .mod (using base-class subprocess helper)
            kraken_exe = exe if exe is not None else self._select_kraken_exe(env)
            self._warn_krakenc_incoherent_sum(run_mode, kraken_exe, n_profiles)
            self._log(f"Running {kraken_exe.name}...")
            self._run_and_attach_prt(
                [str(kraken_exe), base_name], fm.work_dir, base_name,
                stale_outputs=_KRAKEN_MODES_OUTPUTS)

            # Read the eigenvalues back before field.exe sums them: a run
            # below the waveguide's modal cutoff produces a complete .shd that
            # looks like any other. Narrowband and range-independent only — a
            # broadband .mod holds one block per frequency (only some of which
            # may be sub-cutoff) and a segmented one holds one profile per
            # range, and neither is a single statement about the run.
            if not broadband and not is_rd:
                self._check_field_modes_trapped(
                    fm.get_path(base_name), env, source, kraken_exe)

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

            # Physical fastest compressional speed in the waveguide (water
            # column + sediment + half-space) on the complex-spectrum
            # results: the time-series synthesis helpers anchor their
            # output window at r / c_max, ahead of the earliest
            # bottom-refracted arrival.
            if broadband or return_pressure:
                c_max = self._resolve_c_max(env)
                if c_max is not None:
                    field.metadata['c_max'] = c_max

            # No-propagation guard: an empty modal sum — Kraken found 0
            # trapped modes (frequency below the waveguide's modal cutoff, or
            # c_high too low) — leaves field.exe's grid untouched, which the
            # SHD reader surfaces as all-NaN (no-data). Flag it rather than
            # return a silent empty field (compute_modes raises on the same
            # case).
            # Tested on TL so it covers both the complex branches and the
            # real dB INCOHERENT_TL one: an empty sum saturates at the
            # PRESSURE_FLOOR clamp.
            tl = np.asarray(field.db, dtype=float)
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
            self._attach_field_prt_path(field, fm)

            self._log("Kraken simulation complete")
            return field

        finally:
            fm.finish()

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
