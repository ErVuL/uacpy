"""
RAM - Range-dependent Acoustic Model wrapper (multi-backend dispatcher)

The :class:`RAM` class auto-selects one of three vendored Collins-family PE
binaries based on the environment:

- **mpiramS** (default — fluid bottom + flat surface): Dushaw's Fortran 90/95
  rewrite of Collins' original RAM. Native broadband Q/T loop, MPI-ready
  upstream (uacpy builds the serial variant). Custom `inpe`/SSP/BTH multi-file
  input format via :mod:`uacpy.io.mpirams_writer`.
- **rams0.5** (any ``shear_speed > 0`` anywhere): Collins' RAMS elastic PE
  for sediments with shear waves. Single-frequency (``COHERENT_TL`` only).
  Collins-style ``rams.in`` input via :mod:`uacpy.io.ramsurf_writer`.
- **ramsurf1.5** (``env.altimetry is not None``): Collins' rough-surface /
  beach-geometry PE. Single-frequency. Same writer as rams.

Elastic bottom + altimetry raises ``UnsupportedFeatureError`` — no published
Collins PE handles that combination; use OASES for range-independent elastic.

Run modes by backend:
- mpiramS: ``COHERENT_TL``, ``BROADBAND``, ``TIME_SERIES`` (with
  ``source_waveform`` + ``sample_rate``).
- rams0.5 / ramsurf1.5: ``COHERENT_TL`` only — these binaries write real TL
  (no complex pressure), so broadband / time-series isn't available without
  an upstream patch.

The lower boundary at zmax is an absorbing layer in all three backends, not
a rigid Neumann floor.
"""

import numpy as np
import os
import time
import warnings
from pathlib import Path
from typing import Optional, List, Union
from scipy.interpolate import RegularGridInterpolator, interp1d

from uacpy.models.base import PropagationModel, RunMode
from uacpy.models._pe_phase import psi_to_travelling_wave
from uacpy.core.environment import (
    Environment, LayeredBottom, RangeDependentLayeredBottom,
)
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import Result, Field
from uacpy.core.constants import DEFAULT_SOUND_SPEED, TL_MAX_DB
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, ModelExecutionError,
    UnsupportedFeatureError,
)
from uacpy.io.mpirams_writer import write_inpe, write_ssp_file, write_bth_file, write_ranges_file
from uacpy.io.mpirams_reader import read_psif
from uacpy.io.ramsurf_writer import write_ramin
from uacpy.io.ramsurf_reader import read_tl_grid, read_pcomplex_grid


# Collins-family PE numerics constants.
#
# LAMBDA_PER_DZ_FLOOR — minimum acoustic wavelengths per dz step for the
#   Collins finite-difference march. Below this the seafloor interface is
#   smeared between adjacent grid points and accuracy collapses (Collins
#   1993 / Lytaev 2023 §4).
# RAMS_DR_LAMBDA_CAP — empirical upper stability bound on dr for rams0.5's
#   rotated Padé elastic march, expressed as a divisor of c_min/freq.
#   ``dr ≤ c_min / (RAMS_DR_LAMBDA_CAP·f)`` ≈ 0.2 λ per step.
LAMBDA_PER_DZ_FLOOR = 16.0
RAMS_DR_LAMBDA_CAP = 5.0


class RAM(PropagationModel):
    """
    RAM - Range-dependent Acoustic Model (Parabolic Equation), multi-backend.

    A unified façade that picks one of three vendored Collins-family PE
    binaries at run-time based on the environment:

    ============================  =====================================================
    Environment                   Backend selected
    ============================  =====================================================
    fluid bottom + flat surface   ``mpiramS`` — Dushaw's broadband PE (Q/T loop)
    elastic bottom (any shear>0)  ``rams0.5`` — Collins' elastic PE (single-frequency)
    fluid bottom + altimetry      ``ramsurf1.5`` — Collins' rough-surface PE (single-freq)
    elastic + altimetry           ``UnsupportedFeatureError`` (no published Collins PE)
    ============================  =====================================================

    Use ``RAM(...).select_backend(env)`` to inspect the choice without
    actually running. Range-dependent SSP and bathymetry are supported by
    every backend; range-dependent layered bottoms are supported by mpiramS
    and the Collins backends via the same ``LayeredBottom`` /
    ``RangeDependentLayeredBottom`` plumbing (Collins backends emit the
    layered bottom as a Collins-style depth/value piecewise profile).

    Limitations
    -----------
    - Water-column volume attenuation (Thorp / Francois-Garrison / biological)
      is not exposed by any RAM backend. Use Bellhop or Kraken instead.
    - The lower boundary at ``zmax`` is an absorbing layer, not a rigid
      Neumann floor — true rigid bottoms are not supported.
    - Collins backends (rams0.5, ramsurf1.5) are single-frequency at the
      Fortran level. uacpy's local patch dumps the complex envelope (see
      ``third_party/MODIFICATIONS.md``); the wrapper drives the binary
      in a Python-side frequency loop to produce ``BROADBAND`` /
      ``TIME_SERIES`` outputs. mpiramS is still faster for fluid+flat
      broadband (in-process Fortran loop with shared setup).

    Run modes
    ---------
    COHERENT_TL:
        Narrowband TL over a range-depth grid. Available on every backend.
        Returns ``Field``.

    BROADBAND — *mpiramS only*:
        Broadband complex pressure field. Returns
        ``Field`` with ψ(depth, frequency,
        range) for downstream IFFT to time domain.

    TIME_SERIES — *mpiramS only*:
        Real pressure p(t) at each receiver. Internally runs BROADBAND
        and convolves with ``source_waveform`` (sampled at ``sample_rate``).
        Returns ``Field`` / ``Field`` with shape (n_d, n_t, n_r).

    Some constructor kwargs are backend-specific. The list below tags each
    one with the backends that consume it; settings tagged ``[mpiramS]``
    are silently ignored by the Collins backends (rams0.5, ramsurf1.5),
    and uacpy emits a ``UserWarning`` when any such setting is overridden
    from its default and the dispatcher then picks a Collins backend.

    Parameters
    ----------
    executable : Path, optional
        Path to s_mpiram binary. Auto-detected if None. **[mpiramS]**
    dr : float, optional
        Range step in meters. Default: None (auto-select based on
        frequency: dr = c0/freq, i.e. one wavelength, capped at 500m).
        **[all backends]**
    dz : float, optional
        Depth step in meters. Default: None (auto-select based on
        frequency: dz = c_min/(16·freq) clipped to [0.05, 1.0] m and
        snapped so env.depth/dz is an integer — puts the seafloor on a
        depth grid point). **[all backends]**
    np_pade : int, optional
        Number of Pade coefficients (2-8). Default: 6. **[all backends]**
    ns_stability : int, optional
        Number of stability terms. Default: 1 (use 0 for short ranges).
        **[mpiramS, ramsurf1.5]** — rams0.5 uses (rams_irot, rams_theta) instead.
    rs_stability : float, optional
        Stability range in meters. Default: max output range.
        **[mpiramS, ramsurf1.5]**
    Q : float, optional
        Q value for broadband mode (bandwidth = fc/Q). Default: ``None``,
        which resolves to ``2.0`` for broadband paths and to ``1e6`` for
        COHERENT_TL (effectively single-frequency — wide Q collapses
        bandwidth so mpiramS doesn't sweep ~500 frequencies per call).
        Used by every backend's broadband mode to derive the frequency
        vector — mpiramS internally, Collins backends as the Python-side
        frequency-loop grid.
    T : float, optional
        Time window width in seconds (broadband resolution df = 1/T).
        Default: ``None``, which resolves to ``10.0`` for broadband paths
        and to ``1.0`` for COHERENT_TL.
    depth_decimation : int, optional
        Output depth decimation factor. Default: 1 (no decimation).
        **[all backends]**
    flat_earth : bool, optional
        Apply flat-earth transformation (Earth-curvature correction
        applied to the SSP and bathymetry before the PE marches in
        range). Default: True. **[mpiramS]** — the Collins binaries
        (rams0.5, ramsurf1.5) have no equivalent flag and don't apply
        this correction; long-range elastic / rough-surface runs over
        curved Earth will need to be pre-transformed by the caller.
    absorbing_layer_width : float, optional
        Width of the absorbing layer below the seafloor, in wavelengths
        at the centre frequency. Default: 20.0. **[mpiramS]**
    absorbing_layer_attn : float, optional
        Attenuation at the bottom of the absorbing layer, in dB per
        wavelength. Default: 10.0. **[mpiramS]**
    n_sed_points : int, optional
        Number of sediment-profile sampling points. Default: 50.
        **[mpiramS]** — Collins backends consume the layered bottom as
        Collins-style ``(depth, value)`` breakpoints (see
        ``LayeredBottom.to_piecewise_breakpoints``).
    rams_theta : float or callable, optional
        Padé rotation angle in degrees for elastic stability (0 < theta
        < 90). Default: 45.0 (tuned against KrakenC on the Pekeris-elastic
        scenario in tests/test_cross_model_agreement.py). May also be a
        callable ``theta_fn(freq_hz) -> float`` to vary the angle across
        a broadband run — useful when stability degrades with frequency.
        **[rams0.5]**
    rams_irot : int, optional
        Padé rotation flag (1 = on). Default: 1. **[rams0.5]**
    use_tmpfs : bool, optional
        Use RAM-based filesystem for I/O. Default: False.
    verbose : bool, optional
        Print detailed output. Default: False.
    work_dir : Path, optional
        Working directory. If None, creates temporary.

    Notes
    -----
    Defaults auto-derived at ``run()`` time:

    - ``dr=None`` / ``dz=None`` → Lytaev (2023) Padé-error optimizer
      picks the coarsest grid that meets ``accuracy``.
    - ``zmax=None`` → ``_compute_zmax`` (water + absorbing layer).
    - ``c0=None`` → Lytaev Eq. (15) from speed spectrum.
    - ``Q`` / ``T`` → narrowband ``(1e6, 1.0)`` for ``COHERENT_TL``,
      broadband ``(2.0, 10.0)`` for ``BROADBAND`` / ``TIME_SERIES``.
    - Backend (mpiramS / rams0.5 / ramsurf1.5) picked by :meth:`select_backend`
      from ``env`` shape.

    With ``verbose='info'`` the resolved Padé grid is logged per frequency.
    """

    def __init__(
        self,
        executable: Optional[Path] = None,
        dr: Optional[float] = None,
        dz: Optional[float] = None,
        zmax: Optional[float] = None,
        np_pade: int = 6,
        ns_stability: int = 1,
        rs_stability: Optional[float] = None,
        Q: Optional[float] = None,
        T: Optional[float] = None,
        depth_decimation: int = 1,
        flat_earth: bool = True,
        absorbing_layer_width: float = 20.0,
        absorbing_layer_attn: float = 10.0,
        n_sed_points: int = 50,
        c0: Optional[float] = None,
        timeout: float = 600.0,
        # ``dr`` / ``dz`` are picked by the Lytaev (2023) Padé-error
        # optimizer when not set explicitly. ``accuracy`` is the per-run
        # error budget; ``theta_max`` (degrees) bounds the PE spectrum.
        accuracy: float = 1e-3,
        theta_max: float = 30.0,
        # Collins backends only — ignored when the dispatcher picks mpiramS.
        # `theta` is the Padé rotation angle (degrees, 0–90) used by RAMS
        # for elastic stability; defaults are tuned against KrakenC on the
        # Pekeris-elastic problem. ``irot`` is the rotation flag (1 = on).
        rams_theta: float = 45.0,
        rams_irot: int = 1,
        # Multiplicative tightening of the Lytaev-optimised ``dr`` for
        # the ``rams`` backend. Independent of the ``c_min/(5·f)`` λ cap
        # also applied by ``_compute_grid_lytaev``; the tighter of the
        # two wins. Default 5.0 is empirically validated; raise for very
        # long-range or unusually noisy runs.
        rams_dr_safety_factor: float = 5.0,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        **kwargs
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to s_mpiram binary. Auto-detected if None.
        dr : float, optional
            Range step (m). None ⇒ Lytaev optimizer (see ``accuracy``,
            ``theta_max``). Default: None.
        dz : float, optional
            Depth step (m). None ⇒ Lytaev optimizer, then snapped so
            ``env.depth / dz`` is an integer. Default: None.
        zmax : float, optional
            PE domain depth (m). None = auto (seafloor + absorbing layer).
            Default: None.
        np_pade : int, optional
            Number of Pade coefficients (2-8). Default: 6.
        ns_stability : int, optional
            Number of stability terms. Default: 1.
        rs_stability : float, optional
            Stability range (m). None = max output range. Default: None.
        Q : float, optional
            Q value for broadband bandwidth (fc/Q). Default: ``None``,
            which resolves to ``2.0`` for broadband paths and to
            ``1e6`` for COHERENT_TL (effectively single-frequency).
        T : float, optional
            Time window width (s). Default: ``None``, which resolves
            to ``10.0`` for broadband paths and to ``1.0`` for
            COHERENT_TL.
        depth_decimation : int, optional
            Output depth decimation factor. Default: 1.
        flat_earth : bool, optional
            Apply flat-earth transformation. Default: True.
        absorbing_layer_width : float, optional
            Width of the absorbing layer below the seafloor, in
            wavelengths.  Prevents spurious reflections from the bottom
            of the PE domain.  Default: 20.0.
        absorbing_layer_attn : float, optional
            Attenuation at the floor of the absorbing layer
            (dB/wavelength).  Linearly ramped from the environment's
            sediment attenuation at the seafloor to this value at the
            domain bottom.  Default: 10.0.
        n_sed_points : int, optional
            Number of sediment depth control points for the mpiramS
            sediment profile.  More points give finer resolution of
            layered bottoms.  Default: 50.
        c0 : float, optional
            PE reference sound speed (m/s). ``c0`` is the *algorithmic*
            expansion point of the parabolic equation (the speed factored
            out as ``exp(ik₀x)``), **not** a physical input. ``None``
            (default) → uacpy resolves it via Lytaev Eq. (15), the c₀
            that centres the spectrum ``[ξ_min, ξ_max]`` around 0 to
            minimise the Padé approximation error. All three backends
            (mpiramS, rams, ramsurf) honour the resolved value.
            Pass an explicit float to override.
        timeout : float, optional
            Subprocess timeout (s) for each mpiramS run. Default: 600.0.
        accuracy : float, optional
            Lytaev optimiser's per-run accuracy budget (max
            ``|τ · n_steps|``). Default 1e-3.
        theta_max : float, optional
            Maximum propagation angle (degrees) used by the Lytaev
            optimiser to bound the PE spectrum. 30° is the standard
            wide-angle PE assumption. Default 30.
        rams_dr_safety_factor : float, optional
            Tightening factor on the Lytaev-optimised ``dr`` for the
            rams backend (rotated Padé, Milinazzo-Zala-Brooke 1997).
            Applied alongside an independent ``dr ≤ c_min/(5·f)`` λ
            cap; the tighter of the two wins. Default 5.0 — set to 1.0
            to disable, raise for unusually noisy long-range runs.
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            timeout=timeout, **kwargs
        )

        self._supported_modes = [
            RunMode.COHERENT_TL,
            RunMode.BROADBAND,
            RunMode.TIME_SERIES,
        ]
        # The dispatcher routes env.altimetry to the ramsurf1.5 backend,
        # so altimetry IS honoured (just not by mpiramS itself).
        self._supports_altimetry = True
        # RAM dispatches to mpiramS (full range-dep + layered) for fluid+flat,
        # or to rams0.5 / ramsurf1.5 (range-0 only) for elastic / rough surface.
        # Flag everything True at the dispatcher level; the Collins backends
        # already emit their own warnings when they drop range-dep data.
        self._supports_range_dependent_bathymetry = True
        self._supports_range_dependent_ssp = True
        self._supports_range_dependent_bottom = True
        self._supports_layered_bottom = True
        self._supports_range_dependent_layered_bottom = True
        # Elastic bottom auto-routes to rams0.5; surface elastic isn't
        # supported by any backend. _drop_unsupported_surface_shear() in
        # run() warns + zeros surface shear when present.
        self._supports_elastic_media = True
        self._supports_multi_source_depth = False

        if executable is not None:
            self.executable = Path(executable)
        else:
            self.executable = self._find_executable_in_paths(
                's_mpiram', bin_subdirs=['mpirams'], dev_subdir='mpiramS'
            )

        if not self.executable.exists():
            raise ExecutableNotFoundError('RAM:mpiramS', str(self.executable))

        if not isinstance(np_pade, int) or not (2 <= np_pade <= 8):
            raise ConfigurationError(
                f"np_pade must be an integer in [2, 8] (mpiramS limit); "
                f"got {np_pade!r}."
            )
        # Catch garbage scalar inputs at construction time so they fail
        # with a clear Python error instead of a Fortran array-bound or
        # divide-by-zero crash 30 seconds into a binary call.
        for name, val in (('dr', dr), ('dz', dz), ('zmax', zmax),
                          ('rs_stability', rs_stability)):
            if val is not None and (not np.isfinite(val) or val <= 0):
                raise ConfigurationError(f"{name} must be positive and finite if "
                                         f"set; got {val!r}.")
        if c0 is not None and (not np.isfinite(c0) or c0 <= 0):
            raise ConfigurationError(f"c0 must be positive and finite if set; "
                                     f"got {c0!r}.")
        for name, val in (('Q', Q), ('T', T)):
            if val is not None and (not np.isfinite(val) or val <= 0):
                raise ConfigurationError(f"{name} must be positive and finite if "
                                         f"set; got {val!r}.")
        for name, val in (('timeout', timeout),
                          ('absorbing_layer_width', absorbing_layer_width),
                          ('absorbing_layer_attn', absorbing_layer_attn)):
            if not np.isfinite(val) or val <= 0:
                raise ConfigurationError(f"{name} must be positive and finite; "
                                         f"got {val!r}.")
        if not isinstance(n_sed_points, int) or n_sed_points < 2:
            raise ConfigurationError(f"n_sed_points must be an integer >= 2; "
                                     f"got {n_sed_points!r}.")
        if not isinstance(depth_decimation, int) or depth_decimation < 1:
            raise ConfigurationError(f"depth_decimation must be an integer >= 1; "
                                     f"got {depth_decimation!r}.")
        if not isinstance(ns_stability, int) or ns_stability < 0:
            raise ConfigurationError(f"ns_stability must be a non-negative integer; "
                                     f"got {ns_stability!r}.")
        if not callable(rams_theta):
            theta_val = float(rams_theta)
            if not (0.0 <= theta_val <= 90.0):
                raise ConfigurationError(f"rams_theta must be in [0, 90] degrees; "
                                         f"got {theta_val!r}.")

        self.dr = dr
        self.dz = dz
        self.zmax = zmax
        self.np_pade = np_pade
        self.ns_stability = ns_stability
        self.rs_stability = rs_stability
        self.Q = Q
        self.T = T
        self.depth_decimation = depth_decimation
        self.flat_earth = flat_earth
        self.absorbing_layer_width = absorbing_layer_width
        self.absorbing_layer_attn = absorbing_layer_attn
        self.n_sed_points = n_sed_points
        self.c0 = c0

        self.accuracy = float(accuracy)
        self.theta_max = float(theta_max)
        # ``rams_theta`` is either a float (used for every frequency) or
        # a callable ``theta_fn(freq_hz) -> float`` resolved per
        # frequency by ``_theta_for_freq``.
        if not callable(rams_theta):
            rams_theta = float(rams_theta)
        self.rams_theta = rams_theta
        if rams_irot not in (0, 1):
            raise ConfigurationError(f"rams_irot must be 0 or 1; got {rams_irot!r}.")
        self.rams_irot = int(rams_irot)
        if not np.isfinite(rams_dr_safety_factor) or rams_dr_safety_factor < 1.0:
            raise ConfigurationError(
                f"rams_dr_safety_factor must be ≥ 1.0; got "
                f"{rams_dr_safety_factor!r}. Use 1.0 to disable the "
                f"noise-accumulation tightening."
            )
        self.rams_dr_safety_factor = float(rams_dr_safety_factor)

        # Warn on low absorbing-layer attenuation: values < 1 dB/wavelength
        # let bottom reflections leak back into the PE domain and contaminate
        # the field (see Collins, JASA 1996 and mpiramS doc).
        if self.absorbing_layer_attn < 1.0:
            warnings.warn(
                f"RAM absorbing_layer_attn={self.absorbing_layer_attn} "
                "dB/wavelength is low; spurious reflections from the PE "
                "domain bottom may contaminate the field. Typical values "
                "are 5-10 dB/wavelength.",
                UserWarning,
                stacklevel=2
            )

    def _resolve_c0(self, env: Environment) -> float:
        """Resolve the PE reference speed ``c₀``.

        ``c₀`` is the algorithmic expansion point of the parabolic
        equation (the speed in ``exp(ik₀x)`` factored out of the
        Helmholtz solution), not a physical input.

        Resolution order:

        1. ``self.c0`` if the user pinned it explicitly.
        2. Eq. (15) of Lytaev (2023) — the c₀ that centres the spectrum
           ``[ξ_min, ξ_max]`` around 0 and minimises the Padé
           approximation error.

        All three backends honour the resolved value: mpiramS reads it
        from the ``c0_user`` line in ``in.pe``; rams / ramsurf read it
        from the standard ``ram.in`` ``c0`` field.
        """
        if self.c0 is not None:
            return float(self.c0)
        from uacpy.models._pade_optimizer import optimal_c0
        speeds = [float(c) for c in env.ssp.data[:, 0]]
        b = env.bottom
        if isinstance(b, RangeDependentLayeredBottom):
            for prof in b.profiles:
                for layer in prof.layers:
                    if getattr(layer, 'sound_speed', None):
                        speeds.append(float(layer.sound_speed))
                if getattr(prof.halfspace, 'sound_speed', None):
                    speeds.append(float(prof.halfspace.sound_speed))
        elif isinstance(b, LayeredBottom):
            for layer in b.layers:
                if getattr(layer, 'sound_speed', None):
                    speeds.append(float(layer.sound_speed))
            if getattr(b.halfspace, 'sound_speed', None):
                speeds.append(float(b.halfspace.sound_speed))
        elif b is not None and getattr(b, 'sound_speed', None) is not None:
            cs = b.sound_speed
            if hasattr(cs, '__len__'):
                speeds.extend(float(c) for c in cs)
            else:
                speeds.append(float(cs))
        if not speeds:
            return DEFAULT_SOUND_SPEED
        c_min = float(min(speeds))
        c_max = float(max(speeds))
        return float(optimal_c0(c_min, c_max, float(self.theta_max)))

    def _resolve_broadband_grid(self, source: Source):
        """Resolve ``(fc, Q, T)`` for the native broadband sweep.

        mpiramS and the Collins binaries don't accept an arbitrary
        frequency list — their internal loop is parameterised as

            band = fc · [1 - 1/Q, 1 + 1/Q]     (width = 2·fc/Q)
            Δf   = 1/T

        For a multi-element ``frequencies`` array, ``fc`` is always taken
        from the array's centre (the band midpoint) — a band naturally
        identifies its centre frequency, not its lower edge. ``Q`` and
        ``T`` come from the array's half-width and spacing when not
        pinned on the constructor; pinned values take precedence. A
        warning fires whenever either ``Q`` or ``T`` was auto-derived.
        Single-element arrays trivially use ``frequencies[0]`` as fc.
        """
        freqs = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
        if len(freqs) == 1:
            Q = 2.0 if self.Q is None else float(self.Q)
            T = 10.0 if self.T is None else float(self.T)
            return float(freqs[0]), Q, T

        f_min, f_max = float(freqs[0]), float(freqs[-1])
        if f_max <= f_min:
            raise ConfigurationError(
                f"RAM BROADBAND: degenerate frequency range "
                f"[{f_min}, {f_max}] Hz."
            )
        spacings = np.diff(freqs)
        if not np.allclose(spacings, spacings[0], rtol=1e-4):
            raise ConfigurationError(
                f"RAM BROADBAND: non-uniform frequency spacing "
                f"(min Δf={spacings.min():.4g}, max Δf={spacings.max():.4g} Hz). "
                f"mpiramS / Collins broadband sweep is uniform — either pass "
                f"uniformly spaced frequencies, or set `Q` and `T` on the "
                f"constructor and pass a single fc."
            )
        df = float(spacings[0])
        fc = 0.5 * (f_min + f_max)
        half_width = 0.5 * (f_max - f_min)
        Q_auto = fc / half_width
        T_auto = 1.0 / df
        Q = Q_auto if self.Q is None else float(self.Q)
        T = T_auto if self.T is None else float(self.T)
        if self.Q is None or self.T is None:
            warnings.warn(
                f"RAM BROADBAND: mpiramS / Collins use an internal "
                f"(fc, Q, T) sweep. From the {len(freqs)}-element "
                f"frequency array ({f_min:.2f}-{f_max:.2f} Hz, "
                f"Δf={df:.4g} Hz), picked fc={fc:.2f} Hz "
                f"(band centre), Q={Q:.4f} "
                f"({'pinned' if self.Q is not None else 'auto'}), "
                f"T={T:.4f} s "
                f"({'pinned' if self.T is not None else 'auto'}). "
                f"To silence, pin both `Q=` and `T=` on the constructor.",
                UserWarning, stacklevel=3,
            )
        return fc, Q, T

    def _compute_zmax(self, env: Environment, freq: float, c0: Optional[float] = None) -> float:
        """
        Compute PE domain depth (zmax) that extends below the seafloor.

        If self.zmax is set, uses that value directly. Otherwise adds:
        - A thin sediment layer (dz) below the max seafloor depth
        - An absorbing layer (``absorbing_layer_width`` wavelengths) to
          prevent spurious reflections from the domain boundary.

        Parameters
        ----------
        env : Environment
        freq : float
            Frequency in Hz (for wavelength calculation).
        c0 : float
            Reference sound speed for wavelength estimate.
        """
        if self.zmax is not None:
            return self.zmax
        if c0 is None:
            c0 = self._resolve_c0(env)
        max_depth = env.depth
        wavelength = c0 / max(freq, 1.0)
        absorbing_width = self.absorbing_layer_width * wavelength
        dz_for_pad = (float(self.dz) if self.dz is not None
                      else self._compute_dz(env, freq, c0))
        zmax = max_depth + dz_for_pad + absorbing_width
        return zmax

    def _prepare_ssp(self, env: Environment, work_dir: Path,
                     freq: float = 100.0) -> str:
        """
        Write SSP file from environment. Returns filename.

        The SSP is extended below the seafloor to define the PE computation
        domain (zmax). Below the deepest SSP point, sound speed is held
        constant at its last value; sediment properties are handled
        separately by the profl() routine in mpiramS.
        """
        depths_orig = env.ssp.depths.copy()
        speeds_orig = env.ssp.data[:, 0].copy()

        zmax_pe = self._compute_zmax(env, freq)

        # Build depth grid from surface to zmax_pe
        interp_func = interp1d(
            depths_orig, speeds_orig,
            kind='linear', bounds_error=False,
            fill_value=(speeds_orig[0], speeds_orig[-1])
        )
        dz_for_grid = (float(self.dz) if self.dz is not None
                       else self._compute_dz(env, freq))
        n_points = max(50, int(zmax_pe / dz_for_grid / 2))
        depths = np.linspace(0, zmax_pe, n_points)
        base_speeds = interp_func(depths)

        ssp_filename = 'ssp.dat'

        if env.ssp.is_range_dependent:
            self._log("Using 2D SSP matrix")
            ranges_m = env.ssp.ranges.copy()
            ssp_depths = env.ssp.depths

            speeds_2d = np.zeros((len(depths), len(ranges_m)))
            for i in range(len(ranges_m)):
                profile = env.ssp.data[:, i]
                interp_func = interp1d(
                    ssp_depths, profile, kind='linear',
                    bounds_error=False,
                    fill_value=(profile[0], profile[-1])
                )
                speeds_2d[:, i] = interp_func(depths)

            # write_ssp_file (mpiramS .ssp format) expects ranges in km.
            write_ssp_file(work_dir / ssp_filename, depths, speeds_2d, ranges_m / 1000.0)
        else:
            write_ssp_file(work_dir / ssp_filename, depths, base_speeds)

        return ssp_filename

    def _prepare_bathymetry(self, env: Environment, rmax: float, work_dir: Path) -> tuple:
        """
        Write bathymetry file. Returns (bth_filename, ibot).
        """
        bth_filename = 'bathy.dat'

        bathy = env.bathymetry.copy()
        if bathy[0, 0] > 0.0:
            bathy = np.vstack([[0.0, bathy[0, 1]], bathy])
        if bathy[-1, 0] < rmax:
            bathy = np.vstack([bathy, [rmax, bathy[-1, 1]]])
        write_bth_file(work_dir / bth_filename, bathy[:, 0], bathy[:, 1])
        return bth_filename, 1

    def _get_water_speed_at_bottom(
        self, env: Environment, range: Optional[float] = None
    ) -> float:
        """
        Get the water-column sound speed at the nominal seafloor depth.

        This is what mpiramS's ``cwg`` is at the water-sediment interface.
        Used to convert an absolute bottom sound speed into the perturbation
        that mpiramS expects (``cs = cb - cwg``).

        Range-aware: when ``env`` has a range-dependent SSP and ``range``
        is provided, the profile at that range is used. With no ``range``
        but RD SSP, the profile at range 0 is used (first column). This
        keeps the range-independent callers consistent with the RD-aware
        branches in :meth:`_prepare_bottom_properties`.
        """
        depth = env.depth
        if env.has_range_dependent_ssp():
            if range is None:
                range = 0.0
            ssp = env.ssp.eval(range=range).to_pairs()
        else:
            ssp = env.ssp.to_pairs()
        return float(np.interp(depth, ssp[:, 0], ssp[:, 1]))

    def _prepare_bottom_properties(self, env: Environment, work_dir=None):
        """
        Extract bottom properties from environment and convert to mpiramS format.

        mpiramS's sediment model (profl in ram.f90) uses an N-point profile
        (``n_sed_points``) interpolated over depth points
        [0, seafloor, ..interior.., seafloor+sedlayer, zmax]:

        - cs: sediment sound speed *perturbation* relative to water column.
        - rho: sediment density (g/cm^3).
        - attn: sediment attenuation (dB/wavelength).
              The last point is set to absorbing-layer attenuation.

        Returns (sedlayer, nzs, cs, rho, attn, isedrd, sed_filename)
        """
        nzs = self.n_sed_points
        cwg_bottom = self._get_water_speed_at_bottom(env)

        # Use thin sediment layer for sharp interface (≈ half-space)
        sedlayer = self._effective_dz(env)

        if env.has_range_dependent_layered_bottom() and work_dir is not None:
            rdl = env.bottom
            n_ranges = len(rdl.ranges)
            sedlayer_rdl = max(rdl.max_total_thickness(), self._effective_dz(env))

            cs_profiles = np.zeros((nzs, n_ranges))
            rho_profiles = np.zeros((nzs, n_ranges))
            attn_profiles = np.zeros((nzs, n_ranges))

            for i in range(n_ranges):
                cs_samp, rho_samp, attn_samp = rdl.sample_at_depths(i, n_points=nzs)
                lb = rdl.profiles[i]

                # Last point maps to domain bottom (beyond sediment) — use halfspace
                cs_samp[-1] = lb.halfspace.sound_speed
                rho_samp[-1] = lb.halfspace.density
                attn_samp[-1] = lb.halfspace.attenuation

                seafloor_i = float(np.asarray(
                    env.bathymetry_at_range(rdl.ranges[i])
                ).flat[0])
                if env.has_range_dependent_ssp():
                    ssp_at_range = env.ssp.eval(range=rdl.ranges[i]).to_pairs()
                    cwg_local = float(np.interp(seafloor_i,
                                                ssp_at_range[:, 0], ssp_at_range[:, 1]))
                else:
                    cwg_local = float(np.interp(seafloor_i,
                                                env.ssp.depths, env.ssp.data[:, 0]))

                # Points 0,1 = water (zero perturbation), rest = sediment
                cs_profiles[0, i] = 0.0
                cs_profiles[1, i] = 0.0
                cs_profiles[2:, i] = cs_samp[2:] - cwg_local

                rho_profiles[:, i] = rho_samp
                attn_profiles[:, i] = attn_samp
                attn_profiles[-1, i] = self.absorbing_layer_attn

            from uacpy.io.mpirams_writer import write_sediment_file
            sed_filename = 'sediment.sed'
            # write_sediment_file (mpiramS .sed format) expects ranges in km.
            write_sediment_file(
                work_dir / sed_filename,
                rdl.ranges / 1000.0,
                cs_profiles, rho_profiles, attn_profiles
            )

            self._log(f"Range-dependent layered sediment: {n_ranges} profiles, "
                      f"nzs={nzs}, sedlayer={sedlayer_rdl:.1f} m")

            cs = cs_profiles[:, 0].copy()
            rho_arr = rho_profiles[:, 0].copy()
            attn_arr = attn_profiles[:, 0].copy()
            return sedlayer_rdl, nzs, cs, rho_arr, attn_arr, 1, sed_filename

        if env.has_layered_bottom():
            layered = env.bottom
            total_thick = layered.total_thickness()
            sedlayer_lay = max(total_thick, self._effective_dz(env))

            cs_samp, rho_samp, attn_samp = self._sample_layered_bottom(
                layered, nzs)

            # Last point maps to domain bottom (beyond sediment) — use halfspace
            cs_samp[-1] = layered.halfspace.sound_speed
            rho_samp[-1] = layered.halfspace.density
            attn_samp[-1] = layered.halfspace.attenuation

            # Convert absolute sound speeds to perturbations
            cs = cs_samp - cwg_bottom
            cs[0] = 0.0
            cs[1] = 0.0

            rho_arr = rho_samp.copy()
            attn_arr = attn_samp.copy()
            attn_arr[-1] = self.absorbing_layer_attn

            self._log(f"Layered bottom: {len(layered.layers)} layers, "
                      f"nzs={nzs}, sedlayer={sedlayer_lay:.1f} m")

            return sedlayer_lay, nzs, cs, rho_arr, attn_arr, 0, ''

        if env.has_range_dependent_bottom() and work_dir is not None:
            bottom_rd = env.bottom
            n_ranges = len(bottom_rd.ranges)

            cs_profiles = np.zeros((nzs, n_ranges))
            rho_profiles = np.zeros((nzs, n_ranges))
            attn_profiles = np.zeros((nzs, n_ranges))

            for i in range(n_ranges):
                cb = bottom_rd.sound_speed[i]
                seafloor_i = float(env.bathymetry_at_range(bottom_rd.ranges[i])[0])
                if env.has_range_dependent_ssp():
                    ssp_at_range = env.ssp.eval(range=bottom_rd.ranges[i]).to_pairs()
                    cwg_local = float(np.interp(seafloor_i,
                                                ssp_at_range[:, 0], ssp_at_range[:, 1]))
                else:
                    cwg_local = float(np.interp(seafloor_i,
                                                env.ssp.depths, env.ssp.data[:, 0]))
                cs_offset = cb - cwg_local
                cs_profiles[:2, i] = 0.0
                cs_profiles[2:, i] = cs_offset

                rho_val = bottom_rd.density[i] if bottom_rd.density is not None else 1.2
                rho_profiles[:, i] = rho_val

                attn_val = bottom_rd.attenuation[i] if bottom_rd.attenuation is not None else 0.5
                attn_profiles[:, i] = attn_val
                attn_profiles[-1, i] = self.absorbing_layer_attn

            from uacpy.io.mpirams_writer import write_sediment_file
            sed_filename = 'sediment.sed'
            # write_sediment_file (mpiramS .sed format) expects ranges in km.
            write_sediment_file(
                work_dir / sed_filename,
                bottom_rd.ranges / 1000.0,
                cs_profiles, rho_profiles, attn_profiles
            )

            self._log(f"Range-dependent sediment: {n_ranges} profiles, nzs={nzs}")

            cs = cs_profiles[:, 0].copy()
            rho_arr = rho_profiles[:, 0].copy()
            attn_arr = attn_profiles[:, 0].copy()
            return sedlayer, nzs, cs, rho_arr, attn_arr, 1, sed_filename

        # Range-independent halfspace bottom
        cs_offset = 200.0
        rho_val = 1.2
        attn_val = 0.5

        if env.bottom is not None:
            hs = env.halfspace_at_range(0.0)
            cb_val = float(getattr(hs, 'sound_speed', 1600.0) or 1600.0)
            rho_val = float(getattr(hs, 'density', 1.2) or 1.2)
            attn_val = float(getattr(hs, 'attenuation', 0.5) or 0.5)
            cs_offset = cb_val - cwg_bottom

        cs = np.zeros(nzs)
        cs[2:] = cs_offset
        rho_arr = np.full(nzs, rho_val)
        attn_arr = np.full(nzs, attn_val)
        attn_arr[-1] = self.absorbing_layer_attn

        return sedlayer, nzs, cs, rho_arr, attn_arr, 0, ''

    @staticmethod
    def _sample_layered_bottom(layered, n_points):
        """Sample a LayeredBottom at n_points evenly-spaced depths."""
        total_thick = layered.total_thickness()
        if total_thick <= 0:
            total_thick = 1.0
        sample_depths = np.linspace(0, total_thick, n_points)

        cs = np.empty(n_points)
        rho = np.empty(n_points)
        attn = np.empty(n_points)

        for i, d in enumerate(sample_depths):
            cumulative = 0.0
            found = False
            for layer in layered.layers:
                if d <= cumulative + layer.thickness:
                    cs[i] = layer.sound_speed
                    rho[i] = layer.density
                    attn[i] = layer.attenuation
                    found = True
                    break
                cumulative += layer.thickness
            if not found:
                cs[i] = layered.halfspace.sound_speed
                rho[i] = layered.halfspace.density
                attn[i] = layered.halfspace.attenuation

        return cs, rho, attn

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        frequencies=None,
        source_waveform=None,
        sample_rate=None,
        output_duration: Optional[float] = None,
    ) -> Result:
        """
        Run RAM (mpiramS) simulation.

        Parameters
        ----------
        env : Environment
            Ocean environment (supports range-dependent SSP and bathymetry)
        source : Source
            Acoustic source
        receiver : Receiver
            Receiver array
        run_mode : RunMode, optional
            ``COHERENT_TL`` (default) — narrowband TL grid.
            ``BROADBAND`` — complex H(f) over (depth, range, frequency).
            The wrapper converts the PE envelope ψ to engineering
            travelling-wave pressure ``p ∝ conj(ψ)·exp(-i k0 r)/√r``
            before tagging (``metadata['phase_reference']
            ='travelling_wave'``).
            ``TIME_SERIES`` — real pressure p(t); requires
            ``source_waveform`` and ``sample_rate``.
        frequencies : ndarray, optional
            Frequency vector (Hz) for ``BROADBAND`` / ``TIME_SERIES``.
            When provided, overrides ``source.frequencies`` for the duration
            of this call (mirrors Bellhop / Kraken / Scooter). When
            ``None``, RAM uses ``source.frequencies`` as the frequency grid.
        source_waveform : ndarray, optional
            1-D source pulse (required for ``TIME_SERIES``).
        sample_rate : float, optional
            Source-waveform sampling rate in Hz (required for ``TIME_SERIES``).
        output_duration : float, optional
            Desired output duration (seconds) for ``TIME_SERIES``. When
            given, the source waveform is zero-padded internally so the
            auto-derived broadband grid is tight enough (``Δf =
            1/output_duration``) — for mpiramS this also tightens the
            ``(fc, Q, T)`` sweep parameters via
            ``_resolve_broadband_grid``. Defaults to
            ``len(source_waveform)/sample_rate``.

        Returns
        -------
        result : Result
            :class:`Field` for COHERENT_TL, :class:`Field`
            for BROADBAND, :class:`Field` for TIME_SERIES.
        """
        run_mode = self._resolve_run_mode(run_mode)
        source_waveform = self._pad_waveform_to_duration(
            source_waveform, sample_rate, output_duration,
        )
        frequencies = self._resolve_time_series_frequencies(
            run_mode, source, frequencies, source_waveform, sample_rate,
        )

        if frequencies is not None:
            freqs_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))
            if freqs_arr.size == 0:
                raise ConfigurationError(
                    "RAM.run(frequencies=…) requires at least one positive "
                    "frequency"
                )
            source = Source(depths=source.depths, frequencies=freqs_arr)

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)

        backend = self.select_backend(env)
        elastic = self._env_has_elastic_bottom(env)
        rough = getattr(env, 'altimetry', None) is not None
        self._log(
            f"Dispatching to {backend} backend "
            f"(elastic_bottom={elastic}, altimetry={rough})"
        )
        self._warn_on_mpirams_only_overrides(backend)
        env = self._drop_unsupported_surface_shear(env)

        if backend == 'mpiramS':
            if run_mode == RunMode.BROADBAND:
                return self._run_broadband(env, source, receiver)
            if run_mode == RunMode.TIME_SERIES:
                self._require_timeseries_signal(run_mode, source_waveform, sample_rate)
                tf = self._run_broadband(env, source, receiver)
                return tf.synthesize_time_series(
                    source_waveform=source_waveform,
                    sample_rate=sample_rate
                )
            return self._run_tl(env, source, receiver)

        # Collins backends: rams0.5 / ramsurf1.5 are single-frequency
        # PE solvers but uacpy's local patch dumps the complex
        # envelope (see third_party/MODIFICATIONS.md), so BROADBAND is
        # implemented as a Python-side frequency loop and TIME_SERIES
        # builds on top of that via Field.synthesize_time_series.
        if run_mode == RunMode.COHERENT_TL:
            return self._run_collins(env, source, receiver, kind=backend)
        if run_mode == RunMode.BROADBAND:
            return self._run_collins_broadband(
                env, source, receiver, kind=backend
            )
        if run_mode == RunMode.TIME_SERIES:
            self._require_timeseries_signal(run_mode, source_waveform, sample_rate)
            tf = self._run_collins_broadband(
                env, source, receiver, kind=backend
            )
            return tf.synthesize_time_series(
                source_waveform=source_waveform,
                sample_rate=sample_rate
            )
        raise UnsupportedFeatureError(
            f"RAM:{backend}", str(run_mode),
            alternatives=[str(m) for m in self._supported_modes],
            alternatives_label='run modes',
        )

    @staticmethod
    def _min_shear_speed(env: Environment) -> float:
        """Return the slowest non-zero shear speed in the env, or 0 if none.

        Used by the rams elastic path to floor ``dz`` so the rotated Padé
        operator stays stable.
        """
        speeds: List[float] = []

        def _maybe_add(cs):
            if cs is None:
                return
            try:
                arr = np.atleast_1d(cs).astype(float)
            except Exception:
                return
            for v in arr:
                if v > 0:
                    speeds.append(float(v))

        b = env.bottom
        if isinstance(b, LayeredBottom):
            for layer in b.layers:
                _maybe_add(getattr(layer, 'shear_speed', None))
            _maybe_add(getattr(b.halfspace, 'shear_speed', None))
        elif b is not None:
            _maybe_add(getattr(b, 'shear_speed', None))
        return min(speeds) if speeds else 0.0

    @staticmethod
    def _env_has_elastic_bottom(env: Environment) -> bool:
        """Return True if any bottom container carries shear_speed > 0."""
        return env.has_elastic_bottom()

    def select_backend(self, env: Environment) -> str:
        """Inspect which RAM-family binary will run for a given environment.

        Useful for diagnostics and tests — call this before ``run()`` to
        confirm dispatch without executing the binary.

        Returns
        -------
        str
            ``'mpiramS'`` (fluid + flat surface, native broadband),
            ``'rams'`` (elastic bottom, flat surface),
            ``'ramsurf'`` (fluid bottom, variable surface).

        Raises
        ------
        UnsupportedFeatureError
            For elastic + variable-surface environments — no published
            Collins PE handles that combination. Use OASES for
            range-independent elastic propagation, or approximate by
            either flattening the surface (``rams``) or fluidising the
            bottom (``ramsurf``).
        """
        elastic = self._env_has_elastic_bottom(env)
        rough = getattr(env, 'altimetry', None) is not None
        if elastic and rough:
            raise UnsupportedFeatureError(
                'RAM',
                'elastic bottom + sea-surface altimetry',
                alternatives=[
                    "OASES (range-independent elastic + rough)",
                    "drop env.altimetry to use rams0.5 (Collins elastic PE)",
                    "fluidise the bottom (set shear_speed=0) to use ramsurf1.5",
                ]
            )
        if elastic:
            return 'rams'
        if rough:
            return 'ramsurf'
        return 'mpiramS'

    def _collins_binary(self, kind: str) -> Path:
        """Resolve the path to a Collins-family binary on disk."""
        # ram1.5 (Collins fluid PE) is intentionally not built — uacpy
        # uses mpiramS for fluid+flat (broadband + RD bottom support).
        names = {'rams': 'rams0.5', 'ramsurf': 'ramsurf1.5'}
        if kind not in names:
            raise ConfigurationError(f"Unknown Collins kind {kind!r}")
        return self._find_executable_in_paths(
            names[kind],
            bin_subdirs=['ramsurf'],
            dev_subdir='ramsurf'
        )

    def _theta_for_freq(self, freq: float) -> float:
        """Resolve ``rams_theta`` for a single frequency.

        ``rams_theta`` may be:
          - a float: same theta for every frequency (default 45.0).
          - a callable taking a float frequency in Hz and returning a
            float theta in degrees. Use this when the elastic PE needs
            different stability angles across the band.
        """
        t = self.rams_theta
        if callable(t):
            return float(t(float(freq)))
        return float(t)

    def _run_collins(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        kind: str
    ) -> Result:
        """Run a Collins-family binary (rams0.5 / ramsurf1.5) at the
        source's centre frequency and return a TL Field.

        Wraps :meth:`_run_collins_one_freq` and converts the binary's
        ``tl.grid`` to a :class:`Field` interpolated onto the
        requested receiver grid.
        """
        fc = float(np.atleast_1d(source.frequencies)[0])
        raw = self._run_collins_one_freq(
            env, source, receiver, kind=kind, freq=fc,
            theta=self._theta_for_freq(fc)
        )

        n_nonfinite = int(np.count_nonzero(~np.isfinite(raw['tl'])))
        if n_nonfinite > 0:
            # expected; not in filterwarnings — emerges to user
            warnings.warn(
                f"RAM:{kind}: {n_nonfinite}/{raw['tl'].size} TL samples "
                f"are NaN/inf (Padé instability or PE divergence) and "
                f"have been clamped to {TL_MAX_DB} dB. Try a smaller dr "
                f"or larger np_pade.",
                UserWarning, stacklevel=3
            )
        tl_clamped = np.where(np.isfinite(raw['tl']), raw['tl'], TL_MAX_DB)
        # Receivers outside the PE output grid get NaN so pcolormesh and
        # downstream consumers render them transparent rather than as a
        # saturated edge band. Use ``fill_value=TL_MAX_DB`` only inside
        # the grid via the np.where above for non-finite samples.
        interp = RegularGridInterpolator(
            (raw['depths'], raw['ranges']), tl_clamped,
            bounds_error=False, fill_value=np.nan
        )
        rcv_d = np.atleast_1d(receiver.depths).astype(float)
        rcv_r = np.atleast_1d(receiver.ranges).astype(float)
        DD, RR = np.meshgrid(rcv_d, rcv_r, indexing='ij')
        tl_out = interp(np.stack([DD.ravel(), RR.ravel()], axis=-1)).reshape(DD.shape)

        field = Field(
            data=tl_out,
            coords={'depth': rcv_d, 'range': rcv_r},
            **self._result_kwargs(
                source,
                backend=kind,
                frequencies=fc,
                dr=raw['dr'], dz=raw['dz'], zmax=raw['zmax'],
                c0=self._resolve_c0(env)
            )
        )
        self._attach_output_paths(
            field, raw['work_dir'], '',
            primary_files=(
                ('tl_grid_file', 'tl.grid'),
                ('pcomplex_file', 'pcomplex.bin'),
                ('in_file', raw['in_name'])
            )
        )
        return field

    def _run_collins_one_freq(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        *,
        kind: str,
        freq: float,
        theta: float,
        dr_override: Optional[float] = None,
        dz_override: Optional[float] = None,
        zmax_override: Optional[float] = None
    ) -> dict:
        """Execute a Collins-family binary once at ``freq`` and read both
        outputs. Returns a dict with raw arrays (no Field wrapping).

        Keys: ``tl`` (depth × range, dB), ``pcomplex`` (depth × range,
        complex envelope ``u·f3 / sqrt(r)``), ``ranges`` / ``depths``
        (binary output grid), ``dr`` / ``dz`` / ``zmax`` (the values
        actually used).
        """
        binary = self._collins_binary(kind)

        # The Collins binaries handle one (zs, fc) per call; mpiramS does
        # the same. Anything beyond source.depths[0] / source.frequencies[0]
        # is silently dropped without these warnings.

        # Range-dependent inputs that the Collins path doesn't (yet) thread
        # through. Loud warning rather than silent drop — proper RD support
        # would need one segment per range break in the writer call.
        if env.has_range_dependent_ssp():
            # expected; not in filterwarnings — emerges to user
            warnings.warn(
                f"RAM:{kind} backend uses the range-0 SSP only — "
                f"range-dependent SSP from env is dropped. For range-dependent "
                f"fluid PE, use the mpiramS backend (env.bottom without "
                f"shear, no altimetry).",
                UserWarning, stacklevel=3
            )
        if env.has_range_dependent_layered_bottom():
            warnings.warn(
                f"RAM:{kind} backend uses the range-0 layered bottom only — "
                f"env.bottom range-dependence is dropped.",
                UserWarning, stacklevel=3
            )
        if env.has_range_dependent_bottom():
            warnings.warn(
                f"RAM:{kind} backend uses the range-0 bottom geoacoustics only — "
                f"env.bottom range-dependence is dropped. For range-dependent "
                f"fluid PE, use the mpiramS backend (env.bottom without "
                f"shear, no altimetry).",
                UserWarning, stacklevel=3
            )

        fc = float(freq)
        zs = float(np.atleast_1d(source.depths)[0])

        max_range = float(np.max(np.atleast_1d(receiver.ranges)))
        # Numerics resolution: explicit overrides (from broadband loop, which
        # picks one set for the whole band — matching mpiramS) take priority,
        # then user-set self.*, then the Lytaev Padé-error optimizer
        # for whatever is still ``None``. The rams shear-stability dz
        # floor is fed into the optimizer as a hard lower bound; the
        # 5× rams ``dr`` safety factor is applied to its output.
        dr = float(dr_override) if dr_override is not None else (
            float(self.dr) if self.dr is not None else None
        )
        dz = float(dz_override) if dz_override is not None else (
            float(self.dz) if self.dz is not None else None
        )

        if dr is None or dz is None:
            dr_auto, dz_auto = self._compute_grid_lytaev(
                env, fc, max_range=max_range, kind=kind
            )
            if dr is None:
                dr = dr_auto
            if dz is None:
                dz = dz_auto
        if zmax_override is not None:
            zmax = float(zmax_override)
        elif self.zmax is not None:
            zmax = float(self.zmax)
        else:
            zmax = self._compute_zmax(env, fc)

        # Catch receiver depths that exceed the PE computational domain —
        # without this warning the RegularGridInterpolator below silently
        # extrapolates to TL_MAX_DB.
        rcv_d = np.atleast_1d(receiver.depths).astype(float)
        if float(np.max(rcv_d)) > zmax:
            # expected; not in filterwarnings — emerges to user
            warnings.warn(
                f"RAM:{kind}: receiver depths up to {float(np.max(rcv_d)):.1f} m "
                f"exceed the PE domain (zmax={zmax:.1f} m); TL is extrapolated "
                f"and clamped to {TL_MAX_DB} dB outside the domain. Increase "
                f"zmax to cover all receiver depths.",
                UserWarning, stacklevel=3
            )

        target_depth = float(np.atleast_1d(receiver.depths)[0])
        zmplt = max(target_depth + dz, env.depth + dz)
        zmplt = min(zmplt, zmax)

        ndr = max(1, int(np.floor((max_range / dr) / 1000.0)) or 1)
        ndz = max(1, int(self.depth_decimation))

        bathymetry = [(float(r), float(d)) for r, d in env.bathymetry.tolist()]
        if bathymetry[-1][0] < max_range:
            bathymetry.append((float(max_range), bathymetry[-1][1]))

        if kind == 'ramsurf':
            if env.altimetry is None:
                raise ConfigurationError(
                    "ramsurf backend requires env.altimetry to be set; "
                    "got env.altimetry=None. Use the mpiramS backend "
                    "(no altimetry) or supply an altimetry profile."
                )
            # Sign convention: env.altimetry is (range, height) with
            # height positive UP from sea level (Bellhop / .ati convention).
            # ramsurf1.5 expects (range, zsrf) with zsrf >= 0 = depth BELOW
            # z=0 (the pressure-release surface drops by zsrf at that range).
            # So negate, then clamp wave crests (height > 0 → would imply
            # zsrf < 0) to 0 with a warning — ramsurf only models surface
            # depressions / ice keels, not crests above z=0.
            zsrf = [(float(r), -float(z)) for r, z in env.altimetry]
            crests = [(r, h) for r, h in env.altimetry if float(h) > 0]
            if crests:
                warnings.warn(
                    f"ramsurf1.5 only models pressure-release surfaces at or "
                    f"below z=0 (zsrf >= 0). {len(crests)} altimetry sample(s) "
                    f"with height > 0 (wave crests above mean sea level) "
                    f"clamped to z=0. For two-sided wave fields use Bellhop.",
                    UserWarning, stacklevel=3
                )
                zsrf = [(r, max(0.0, z)) for r, z in zsrf]
            surface = zsrf
            if surface[-1][0] < max_range:
                surface.append((float(max_range), surface[-1][1]))
        else:
            surface = None

        properties_fluid = ('sound_speed', 'density', 'attenuation')
        properties_elastic = (
            'sound_speed', 'shear_speed',
            'density', 'attenuation', 'shear_attenuation'
        )

        layered = (
            env.bottom if isinstance(env.bottom, LayeredBottom)
            else self._fallback_layered_from_bottom(env)
        )

        if kind == 'rams':
            bp = layered.to_piecewise_breakpoints(
                seafloor_depth=env.depth,
                zmax=zmax,
                properties=properties_elastic
            )
            seg = dict(
                range=0.0,
                water_ssp=[(float(d), float(c)) for d, c in env.ssp.to_pairs()],
                bottom_c=bp['sound_speed'],
                bottom_cs=bp['shear_speed'],
                bottom_rho=bp['density'],
                bottom_attn=bp['attenuation'],
                bottom_attns=bp['shear_attenuation']
            )
        else:
            bp = layered.to_piecewise_breakpoints(
                seafloor_depth=env.depth,
                zmax=zmax,
                properties=properties_fluid
            )
            seg = dict(
                range=0.0,
                water_ssp=[(float(d), float(c)) for d, c in env.ssp.to_pairs()],
                bottom_c=bp['sound_speed'],
                bottom_rho=bp['density'],
                bottom_attn=bp['attenuation']
            )

        # Bottom depth in Collins format is referenced from z=0 (top of
        # the water), the same convention as everywhere else in uacpy.
        # The to_piecewise_breakpoints helper already emits that.

        fm = self._setup_file_manager()
        self.file_manager = fm
        try:
            # rams0.5 hardcodes 'rams.in'; ramsurf1.5 reads 'ram.in'.
            in_name = 'rams.in' if kind == 'rams' else 'ram.in'
            ram_in = fm.get_path(in_name)
            c0_pe = self._resolve_c0(env)
            write_ramin(
                str(ram_in),
                kind=kind,
                fc=fc, zs=zs, zr_line=target_depth,
                rmax=max_range, dr=dr, ndr=ndr,
                zmax=zmax, dz=dz, ndz=ndz, zmplt=zmplt,
                c0=c0_pe, np_pade=int(self.np_pade),
                ns_stab=int(self.ns_stability),
                rs_stab=float(self.rs_stability or 0.0),
                irot=int(self.rams_irot),
                theta=float(theta),
                bathymetry=bathymetry,
                surface=surface,
                range_segments=[seg],
                title=f"uacpy {kind} run @ {fc:.1f} Hz"
            )

            self._log(f"Executing: {binary} (cwd={fm.work_dir})")
            proc_result = self._run_subprocess(
                [str(binary)],
                cwd=fm.work_dir,
                timeout=self.timeout
            )

            tlgrid = fm.work_dir / 'tl.grid'
            if not tlgrid.exists():
                raise ModelExecutionError(
                    self.model_name,
                    return_code=0,
                    stdout=proc_result.stdout,
                    stderr=(
                        f"{kind}: tl.grid not produced (cwd={fm.work_dir})\n"
                        + (proc_result.stderr or "")
                    )
                )
            ranges, depths, tl = read_tl_grid(
                tlgrid, dr=dr, ndr=ndr, dz=dz, ndz=ndz
            )

            # The patched outpt (third_party/ramsurf/{rams0.5,ramsurf1.5}.f
            # + MODIFICATIONS.md) writes pcomplex.bin alongside tl.grid.
            pcgrid = fm.work_dir / 'pcomplex.bin'
            if not pcgrid.exists():
                raise ModelExecutionError(
                    self.model_name,
                    return_code=0,
                    stdout=proc_result.stdout,
                    stderr=(
                        f"{kind}: pcomplex.bin not produced. Rebuild the "
                        f"binaries via install.sh so the patched outpt "
                        f"routine emits the complex envelope.\n"
                        + (proc_result.stderr or "")
                    )
                )
            _, _, pcomplex = read_pcomplex_grid(
                pcgrid, dr=dr, ndr=ndr, dz=dz, ndz=ndz
            )

            return {
                'tl': tl,
                'pcomplex': pcomplex,
                'depths': depths,
                'ranges': ranges,
                'dr': dr, 'dz': dz, 'zmax': zmax,
                'frequency': fc, 'source_depth': zs,
                'work_dir': fm.work_dir,
                'in_name': in_name,
            }
        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _run_collins_broadband(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        kind: str
    ) -> Result:
        """Loop a Collins binary over the broadband frequency vector and
        assemble a transfer-function Field.

        The frequency vector matches mpiramS's convention: centred on
        ``source.frequencies[0]`` with bandwidth ``fc/Q`` and frequency
        resolution ``df = 1/T``. Each frequency runs the Collins binary
        once; the patched binary writes a complex envelope (see
        ``third_party/MODIFICATIONS.md``) which the loop stacks into
        ``(n_d, n_r, n_f)``. The carrier ``exp(-i k0 r)`` is baked in
        below before tagging, so the result is the same engineering
        travelling-wave H(f) as every other broadband-capable model.

        ``rams_theta`` may be a callable; when it is, ``theta`` is
        resolved per frequency by ``_theta_for_freq`` — useful when the
        elastic stability angle has to vary across the band.
        """

        fc, Q_used, T_used = self._resolve_broadband_grid(source)
        # Match mpiramS bandwidth / df conventions: bw = fc/Q, df = 1/T.
        bw = fc / Q_used
        df = 1.0 / T_used
        nf1 = max(1, int(np.floor((bw - df) / df)))
        # Symmetric grid centred on fc, like mpiramS' peramx.f90.
        frequencies = np.array(
            [(ii - nf1) * df + fc for ii in range(2 * nf1 + 1)],
            dtype=float
        )
        # Drop non-positive (fc - bw can dip below 0 for small Q).
        frequencies = frequencies[frequencies > 0.0]

        # Pick numerics ONCE for the whole broadband loop. dr is sized to
        # the lowest freq (largest λ → coarsest acceptable step); dz to
        # the highest freq (smallest λ_min → finest required step). Both
        # auto-paths honour explicit self.dr / self.dz / self.zmax. This
        # mirrors mpiramS' approach (single dr,dz across the whole band)
        # and avoids per-frequency grid changes that would force the
        # binary to re-allocate depth grids of varying sizes.
        f_min = float(frequencies[0])
        f_max = float(frequencies[-1])
        rmax_band = float(np.max(np.atleast_1d(receiver.ranges)))

        dr_band = float(self.dr) if self.dr is not None else None
        dz_band = float(self.dz) if self.dz is not None else None
        # dr is sized at f_min (largest λ → coarsest acceptable step),
        # dz at f_max (smallest λ → finest required step) — two
        # independent Lytaev calls.
        if dr_band is None:
            dr_band, _ = self._compute_grid_lytaev(
                env, f_min, max_range=rmax_band, kind=kind
            )
        if dz_band is None:
            _, dz_band = self._compute_grid_lytaev(
                env, f_max, max_range=rmax_band, kind=kind
            )
        zmax_band = (float(self.zmax) if self.zmax is not None
                     else self._compute_zmax(env, f_min))

        self._log(
            f"{kind} broadband: {len(frequencies)} frequencies, "
            f"{frequencies[0]:.2f}-{frequencies[-1]:.2f} Hz, "
            f"df={df:.2f} Hz, bw={bw:.2f} Hz, "
            f"dr={dr_band:.2f}m, dz={dz_band:.3f}m, zmax={zmax_band:.0f}m"
        )

        rcv_d = np.atleast_1d(receiver.depths).astype(float)
        rcv_r = np.atleast_1d(receiver.ranges).astype(float)
        DD, RR = np.meshgrid(rcv_d, rcv_r, indexing='ij')

        # Convention: trailing axis is the variable dim (frequency).
        H = np.zeros((rcv_d.size, rcv_r.size, frequencies.size), dtype=complex)

        zmax_used = None
        dr_first = None
        dz_first = None
        # Print progress every ~10% of frequencies; on a 500-freq elastic
        # run each iteration takes ~1 s of subprocess overhead, so without
        # this the verbose log goes silent for many minutes.
        log_every = max(1, len(frequencies) // 10)
        for k, freq in enumerate(frequencies):
            if self.verbose and (k % log_every == 0 or k == len(frequencies) - 1):
                self._log(
                    f"{kind} broadband: freq {k + 1}/{len(frequencies)} "
                    f"({float(freq):.2f} Hz)"
                )
            theta_k = self._theta_for_freq(float(freq))
            raw = self._run_collins_one_freq(
                env, source, receiver,
                kind=kind, freq=float(freq), theta=theta_k,
                dr_override=dr_band, dz_override=dz_band,
                zmax_override=zmax_band
            )
            zmax_used = raw['zmax']
            if dr_first is None:
                dr_first = raw['dr']
            if dz_first is None:
                dz_first = raw['dz']

            # Out-of-grid receivers → NaN so the resulting H(f) cell is
            # NaN (transparent in plots) instead of 0 (which clips TL to
            # TL_MAX_DB and saturates the heatmap edges).
            interp_re = RegularGridInterpolator(
                (raw['depths'], raw['ranges']),
                np.real(raw['pcomplex']),
                bounds_error=False, fill_value=np.nan
            )
            interp_im = RegularGridInterpolator(
                (raw['depths'], raw['ranges']),
                np.imag(raw['pcomplex']),
                bounds_error=False, fill_value=np.nan
            )
            pts = np.stack([DD.ravel(), RR.ravel()], axis=-1)
            re = interp_re(pts).reshape(DD.shape)
            im = interp_im(pts).reshape(DD.shape)
            H[:, :, k] = re + 1j * im

        # Convert each backend's raw output to the engineering travelling-
        # wave form. See ``models/_pe_phase.py`` for the per-convention
        # math. H is shaped (n_d, n_r, n_f) here; the Collins binaries
        # already include the 1/√r radial scaling in the file they write,
        # so ``apply_radial=False``.
        c0 = self._resolve_c0(env)
        omega = 2.0 * np.pi * np.asarray(frequencies, dtype=np.float64)
        H = psi_to_travelling_wave(
            H,
            convention=kind,
            ranges_m=rcv_r,
            range_axis=1,
            k0=omega / c0,
            freq_axis=2,
            apply_radial=False,
        )

        field = Field(
            data=H,
            coords={'depth': rcv_d, 'range': rcv_r, 'frequency': frequencies},
            phase_reference='travelling_wave',
            **self._result_kwargs(
                source,
                backend=kind,
                frequencies=frequencies,
                Q=Q_used, T=T_used,
                bandwidth_hz=bw, df_hz=df,
                dr=dr_first, dz=dz_first, zmax=zmax_used,
                c0=c0,
                c_min=c0,
            )
        )
        self._attach_output_paths(
            field, raw['work_dir'], '',
            primary_files=(
                ('tl_grid_file', 'tl.grid'),
                ('pcomplex_file', 'pcomplex.bin'),
                ('in_file', raw['in_name'])
            )
        )
        return field

    def _fallback_layered_from_bottom(self, env: Environment) -> 'LayeredBottom':
        """Synthetic single-layer LayeredBottom for the Collins backends.

        Delegates to :meth:`LayeredBottom.from_halfspace` so the wrapping
        rule (10% of water depth, 5 m floor) is shared with any other
        wrapper that needs a synthetic sediment layer.
        """
        if env.bottom is None:
            raise ConfigurationError(
                "RAM backend requires env.bottom"
            )
        if isinstance(env.bottom, LayeredBottom):
            return env.bottom
        return LayeredBottom.from_halfspace(env.halfspace_at_range(0.0), env.depth)

    # Settings that only the mpiramS backend consumes. When the dispatcher
    # picks rams0.5 / ramsurf1.5 and one of these has been overridden from
    # its default, ``_warn_on_mpirams_only_overrides`` warns rather than
    # silently dropping the override. Each entry is (attribute, default).
    # Q and T are honoured by every backend's broadband mode: the Collins
    # path uses them as the Python-side frequency-loop grid, mpiramS uses
    # them inside the Fortran loop.
    _MPIRAMS_ONLY_SETTINGS = (
        ('flat_earth', True),
        ('absorbing_layer_width', 20.0),
        ('absorbing_layer_attn', 10.0),
        ('n_sed_points', 50)
    )

    def _drop_unsupported_surface_shear(self, env: Environment) -> Environment:
        """No RAM backend reads surface shear properties; warn and zero them."""
        s = getattr(env, 'surface', None)
        cs = getattr(s, 'shear_speed', None) if s is not None else None
        if cs is None or float(cs) <= 0.0:
            return env
        import warnings as _w
        e = env.copy()
        e.surface = self._collapse_elastic_boundary(
            e.surface, self._collapse["elastic"]
        )
        _w.warn(
            "RAM: surface shear is not supported by any backend "
            "(mpiramS / rams0.5 / ramsurf1.5 all model the surface as "
            "pressure-release); collapsed surface shear "
            f"(collapse['elastic']={self._collapse['elastic']!r}). "
            "For an elastic surface use Bellhop or KrakenC.",
            UserWarning, stacklevel=3
        )
        return e

    def _warn_on_mpirams_only_overrides(self, backend: str) -> None:
        if backend == 'mpiramS':
            return
        nondefault = [
            name for name, default in self._MPIRAMS_ONLY_SETTINGS
            if getattr(self, name) != default
        ]
        if nondefault:
            warnings.warn(
                f"RAM:{backend} ignores these mpiramS-only settings "
                f"(left at their effective default in the Collins binary): "
                f"{', '.join(nondefault)}. See the RAM constructor docstring "
                f"for the per-backend applicability.",
                UserWarning, stacklevel=4
            )

    def _compute_grid_lytaev(
        self, env: 'Environment', freq: float,
        *, max_range: float, kind: str
    ) -> 'tuple[float, float]':
        """Padé-error-based ``(dr, dz)`` selection following Lytaev
        (2023, https://doi.org/10.3390/jmse11030496).

        Picks the coarsest ``(dr, dz)`` whose accumulated single-step
        Padé error stays under ``self.accuracy`` over the marched range.
        The PE reference speed ``c₀`` comes from ``_resolve_c0`` (Lytaev
        Eq. (15) by default, the user's value when pinned). The rams
        shear-stability floor is enforced as a hard lower bound on
        ``dz``. Raises ``ValueError`` if no candidate ``(dr, dz)`` pair
        meets the accuracy budget.
        """
        from uacpy.models._pade_optimizer import optimize_grid, rams_dz_floor

        c0_pe = self._resolve_c0(env)

        # Spectrum bounds: slowest / fastest acoustic speeds in the env.
        speeds = [c0_pe]
        for c in env.ssp.data[:, 0]:
            speeds.append(float(c))
        b = env.bottom
        if isinstance(b, RangeDependentLayeredBottom):
            for prof in b.profiles:
                for layer in prof.layers:
                    if getattr(layer, 'sound_speed', None):
                        speeds.append(float(layer.sound_speed))
                if getattr(prof.halfspace, 'sound_speed', None):
                    speeds.append(float(prof.halfspace.sound_speed))
        elif isinstance(b, LayeredBottom):
            for layer in b.layers:
                if getattr(layer, 'sound_speed', None):
                    speeds.append(float(layer.sound_speed))
            if getattr(b.halfspace, 'sound_speed', None):
                speeds.append(float(b.halfspace.sound_speed))
        elif b is not None and getattr(b, 'sound_speed', None) is not None:
            cs = b.sound_speed
            if hasattr(cs, '__len__'):
                speeds.extend(float(c) for c in cs)
            else:
                speeds.append(float(cs))
        c_min = float(min(speeds))
        c_max = float(max(speeds))

        # Per-backend dz floor: λ_p/16 acoustic stability for Collins
        # backends + cost cap for mpiramS; 0.55·λ_s shear-mode aliasing
        # for rams. Override via ``dr=…``/``dz=…``.
        if kind in ('mpiramS', 'rams', 'ramsurf'):
            dz_floor_acoustic = c_min / (LAMBDA_PER_DZ_FLOOR * max(freq, 1.0))
            cs_min = self._min_shear_speed(env) if kind == 'rams' else 0.0
            dz_floor_shear = rams_dz_floor(cs_min, freq, factor=0.55)
            dz_floor = max(dz_floor_shear, dz_floor_acoustic)
        else:
            cs_min = 0.0
            dz_floor = 0.0

        # Auto-loosen on infeasibility: hard environments (deep ocean,
        # high c_max, wide θ_max) can't satisfy the user's ε with the
        # Collins 2nd-order Numerov. Relax ε progressively up to 0.5,
        # then drop θ_max stepwise (30°→20°→15°→10°). The optimizer
        # still picks a *Lytaev-derived* grid; we surface every relax
        # via a warning so the user knows their target wasn't met.
        eps0 = float(self.accuracy)
        theta0 = float(self.theta_max)
        # Floor at 15° — Collins (1993) treats 30° as the standard
        # wide-angle PE; below 15° the operator is essentially paraxial
        # and the term "wide-angle" stops being meaningful. Users who
        # genuinely want narrow-angle physics should pass an explicit
        # ``theta_max`` to the constructor.
        theta_floor = 15.0
        eps_used, theta_used, res, last_exc = eps0, theta0, None, None
        for theta_trial in (theta0, 20.0, theta_floor):
            if theta_trial > theta0:
                continue
            theta_used = theta_trial
            eps_used = eps0
            for _ in range(8):
                try:
                    res = optimize_grid(
                        freq=float(freq),
                        c_min=c_min, c_max=c_max,
                        x_max=float(max_range),
                        c0=c0_pe,
                        theta_max=float(theta_used),  # degrees
                        eps=eps_used,
                        p=int(self.np_pade),
                        alpha=0.0
                    )
                    break
                except RuntimeError as exc:
                    last_exc = exc
                    eps_used *= 3.0
                    if eps_used > 0.5:
                        break
            if res is not None:
                break
        if res is None:
            raise ConfigurationError(
                f"RAM:{kind}: no Lytaev grid feasible even at ε=0.5, "
                f"θ_max={theta_floor:.0f}° for f={freq:.1f} Hz, "
                f"x_max={max_range:.0f} m. Set ``dr``/``dz`` explicitly. "
                f"Optimiser said: {last_exc}"
            ) from last_exc
        if eps_used > eps0 or theta_used < theta0:
            warnings.warn(
                f"RAM:{kind}: Lytaev relaxed ε={eps0:.0e}→{eps_used:.0e}, "
                f"θ_max={theta0:.0f}°→{theta_used:.0f}° to find a feasible "
                f"grid at f={freq:.1f} Hz, x_max={max_range:.0f} m. "
                f"Expect TL errors larger than your original target.",
                UserWarning, stacklevel=3
            )

        dr_opt, dz_opt = float(res['dr']), float(res['dz'])

        # rams0.5's rotated Padé (Milinazzo, Zala & Brooke 1997) is
        # L-stable by construction, but ``|G|`` sits close to 1 for
        # spectrum eigenvalues near the marginally-stable boundary, so
        # floating-point noise compounds over thousands of range steps.
        # Two independent constraints:
        #   1. ``rams_dr_safety_factor`` shrinks Lytaev's accuracy-
        #      optimal ``dr`` by a constant factor (noise margin).
        #   2. A wavelength cap ``dr ≤ c_min / (5·f)`` ≈ 0.2 λ per step,
        #      empirically validated as the upper stability bound for
        #      the rotated elastic march across a wide env range.
        # We apply BOTH and take the tighter (smaller) ``dr``.
        if kind == 'rams':
            dr_pre = dr_opt
            dr_safety = dr_opt / self.rams_dr_safety_factor
            dr_cap = c_min / (RAMS_DR_LAMBDA_CAP * freq)
            dr_opt = min(dr_safety, dr_cap)
            limit = 'safety factor' if dr_safety <= dr_cap else 'λ cap'
            self._log(
                f"rams: tightened dr from {dr_pre:.2f} m to "
                f"{dr_opt:.2f} m (safety={dr_safety:.2f}, "
                f"λ-cap={dr_cap:.2f}; {limit} active)."
            )

        # Snap dz to a depth-grid-aligned value so the seafloor lands on
        # a node (PE accuracy degrades sharply otherwise).
        bathy = getattr(env, 'bathymetry', None)
        if bathy is not None and len(bathy) > 0:
            h = float(np.min(np.asarray(bathy)[:, 1]))
        else:
            h = float(getattr(env, 'depth', None) or 0.0)
        if h > 0:
            n_layers = max(1, int(round(h / dz_opt)))
            dz_opt = float(h / n_layers)

        # Practical depth-grid cap. Pure runtime safety — Lytaev's
        # optimizer at very low freq / deep ocean / wide θ_max can
        # demand dz ≈ λ/300 (5 cm at 25 Hz) → 100k+ depth points and
        # very slow per-step compute. Stability is handled separately
        # by ``dz_floor`` for Collins backends. Raise via dr/dz
        # override for accuracy-sensitive runs.
        MAX_DEPTH_POINTS = 10000
        if h > 0 and h / dz_opt > MAX_DEPTH_POINTS:
            dz_pre = dz_opt
            n_layers = MAX_DEPTH_POINTS
            dz_opt = float(h / n_layers)
            warnings.warn(
                f"RAM:{kind}: raised dz from {dz_pre:.4f} m to {dz_opt:.3f} m "
                f"to keep the depth grid under {MAX_DEPTH_POINTS} points "
                f"(seafloor depth {h:.0f} m). Lytaev accuracy budget "
                f"ε={self.accuracy:.0e} is no longer met. Reduce "
                f"``theta_max`` or set dr/dz explicitly to override.",
                UserWarning, stacklevel=3
            )

        if dz_floor > 0 and dz_opt < dz_floor:
            dz_pre = dz_opt
            if h > 0:
                n_layers = max(1, int(np.floor(h / dz_floor)))
                dz_opt = float(h / n_layers)
            else:
                dz_opt = dz_floor
            if cs_min > 0:
                reason = 'shear-mode stability (λ_s × 0.55)'
            elif kind == 'mpiramS':
                reason = 'mpiramS runtime cap (λ_p / 16)'
            else:
                reason = 'acoustic stability (λ_p / 16)'
            warnings.warn(
                f"RAM:{kind}: raised dz from {dz_pre:.3f} m to "
                f"{dz_opt:.3f} m for {reason} "
                f"(floor={dz_floor:.3f} m at cs_min={cs_min:.0f} m/s, "
                f"f={freq:.0f} Hz). The Lytaev accuracy budget "
                f"ε={self.accuracy:.0e} is no longer met — expect TL "
                f"errors larger than the target. Set dr/dz explicitly "
                f"to override. For broadband sweeps the cap is computed "
                f"at the *lowest* frequency in the sweep, so it may be "
                f"sub-Nyquist for the upper band — pin dz≈λ(f_max)/8 "
                f"to resolve the full pulse spectrum.",
                UserWarning, stacklevel=3
            )

        c0_origin = 'user' if self.c0 is not None else 'Lytaev Eq.15'
        self._log(
            f"{kind}: Lytaev grid → dr={dr_opt:.2f} m, "
            f"dz={dz_opt:.3f} m (predicted error "
            f"{res['predicted_error']:.2e}, c₀={c0_pe:.1f} m/s "
            f"[{c0_origin}], θ_max={self.theta_max:.0f}°, "
            f"ε={self.accuracy:.0e})."
        )
        return dr_opt, dz_opt

    def _effective_dz(self, env: 'Environment',
                      freq: Optional[float] = None) -> float:
        """Resolve `dz` outside the per-frequency hot paths (sediment layer
        thickness, layered-bottom padding, metadata). Honours an explicit
        `self.dz` if set, otherwise falls back to the adaptive value at
        ``freq`` (or 0.5 m if no freq is in scope).
        """
        if self.dz is not None:
            return float(self.dz)
        if freq is not None:
            return self._compute_dz(env, freq)
        return 0.5

    def _compute_dz(self, env: 'Environment', freq: float,
                    c0: Optional[float] = None) -> float:
        """Quick ``λ_min/16`` depth-step estimate, clipped to [0.05, 1.0] m.

        Used only for auxiliary sizing (absorbing-layer thickness in
        ``_compute_zmax``, sediment-layer placeholders in
        ``_effective_dz``). The main PE grid is always picked by the
        Lytaev optimizer or by an explicit user value.

        ``λ_min`` uses the slowest **acoustic** (compressional) wave
        speed in the env — water ``c0`` and sediment ``cp``. Shear
        speeds are deliberately excluded:

        * The shear wave lives in the sediment, not in the water
          column where the PE march computes the acoustic field. The
          elastic seafloor is handled by the interface impedance
          condition, which only requires the right boundary terms in
          the wave equation, not depth-grid resolution of λ_s/16.
        * Including shear when it is "fast enough to propagate"
          (cs ≳ 200 m/s) actually drives the Padé march unstable on
          ``rams0.5``: the resulting dz ≈ λ_s/16 ≈ 0.15 m gives a
          dr/dz ratio of 100+ that the rotated Padé operator cannot
          handle (validated empirically on Pekeris-with-elastic
          regressions — bad-sample fraction goes from 0% at shear=150
          to >90% at shear=200 with the same auto-numerics).
        """
        if c0 is None:
            c0 = self._resolve_c0(env)
        speeds = [float(c0)]

        def _add(cp_attr):
            if cp_attr:
                speeds.append(float(cp_attr))

        b = env.bottom
        if isinstance(b, LayeredBottom):
            for layer in b.layers:
                _add(getattr(layer, 'sound_speed', None))
            _add(getattr(b.halfspace, 'sound_speed', None))
        elif b is not None:
            _add(getattr(b, 'sound_speed', None))
        c_min = min(speeds)
        wavelength = c_min / max(freq, 1.0)
        target = float(np.clip(wavelength / LAMBDA_PER_DZ_FLOOR, 0.05, 1.0))
        # Snap dz so the seafloor lands on a depth grid point. PE
        # accuracy degrades sharply when dz does NOT divide the water
        # depth (the seafloor interface gets smeared between adjacent
        # samples) — depth-FD discretisation artifact, not a physics
        # limit. For range-dependent bathymetry we cannot align the
        # grid to every range simultaneously: snap to the SHALLOWEST
        # bathymetry point, which is the most numerically demanding
        # (fewest grid points in the water column → largest relative
        # interface displacement when off-grid). The deeper ranges
        # then have the seafloor between grid points by at most one
        # dz, which is small relative to their thicker water columns.
        bathy = getattr(env, 'bathymetry', None)
        if bathy is not None and len(bathy) > 0:
            h = float(np.min(np.asarray(bathy)[:, 1]))
        else:
            h = float(getattr(env, 'depth', None) or 0.0)
        if h > 0:
            n_layers = max(1, int(round(h / target)))
            return float(h / n_layers)
        return target

    def _run_tl(self, env, source, receiver):
        """
        Run in narrowband TL mode.

        Honours ``self.Q`` / ``self.T``. For a single-frequency TL grid the
        conventional choice is a very large Q so the bandwidth collapses,
        but the user is free to widen it.
        """
        start_time = time.time()

        freq = float(source.frequencies[0])
        zsrc = float(source.depths[0])
        ranges = receiver.ranges
        rmax = float(np.max(ranges))

        # ``dr`` / ``dz`` come from the user, else from the Lytaev
        # Padé-error optimizer.
        dr = float(self.dr) if self.dr is not None else None
        dz = float(self.dz) if self.dz is not None else None
        if dr is None or dz is None:
            dr_auto, dz_auto = self._compute_grid_lytaev(
                env, freq, max_range=rmax, kind='mpiramS'
            )
            if dr is None:
                dr = dr_auto
            if dz is None:
                dz = dz_auto

        # COHERENT_TL collapses the mpiramS broadband window to ~one bin
        # (Q→∞, T=1) unless the user widened it via Q=/T=.
        Q_tl = 1e6 if self.Q is None else float(self.Q)
        T_tl = 1.0 if self.T is None else float(self.T)
        self._log(
            f"mpiramS (TL mode): freq={freq:.1f} Hz, zs={zsrc:.1f} m, "
            f"dr={dr:.1f} m, dz={dz:.3f} m, Q={Q_tl:g}, T={T_tl:g}s"
        )
        self._log(f"Output grid: {len(ranges)} ranges x {len(receiver.depths)} depths")

        fm = self._setup_file_manager()
        work_dir = fm.work_dir

        try:
            # Prepare input files
            ssp_filename = self._prepare_ssp(env, work_dir, freq)
            bth_filename, ibot = self._prepare_bathymetry(env, rmax, work_dir)
            sedlayer, nzs, cs, rho_arr, attn_arr, isedrd, sed_filename = \
                self._prepare_bottom_properties(env, work_dir)

            write_ranges_file(work_dir / 'ranges.dat', ranges)

            rs = self.rs_stability if self.rs_stability is not None else rmax

            # Only use horizontal interpolation for range-dependent SSPs
            ihorz = 1 if env.has_range_dependent_ssp() else 0

            write_inpe(
                filepath=work_dir / 'in.pe',
                fc=freq,
                Q=Q_tl,
                T=T_tl,
                zsrc=zsrc,
                deltaz=dz,
                deltar=dr,
                np_pade=self.np_pade,
                nss=self.ns_stability,
                rs=rs,
                dzm=self.depth_decimation,
                ssp_filename=ssp_filename,
                iflat=1 if self.flat_earth else 0,
                ihorz=ihorz,
                ibot=ibot,
                bth_filename=bth_filename,
                sedlayer=sedlayer,
                nzs=nzs,
                cs=cs,
                rho=rho_arr,
                attn=attn_arr,
                isedrd=isedrd,
                sed_filename=sed_filename,
                c0_user=self._resolve_c0(env)
            )

            # Run mpiramS
            self._run_binary(work_dir)
            result = read_psif(work_dir)

            psif = result['psif']  # (nzo, nf, nr)
            zg = result['zg']
            rout = result['rout']

            # Extract center frequency (middle of frequency vector)
            center_idx = result['nf'] // 2
            # pressure at center freq for all depths and ranges: (nzo, nr)
            pressure = psif[:, center_idx, :]

            # Interpolate COMPLEX PRESSURE from PE grid to receiver grid
            # BEFORE computing TL. Interpolating in dB destroys interference
            # nulls because linear interpolation of log-scale values smooths
            # out the sharp zeros in the field.

            rcv_depths = receiver.depths
            # Warn only when receiver ranges genuinely exceed the PE
            # marched range; tolerate float-edge drift (rout[-1] often
            # lands ~µm below the user's requested rmax due to dr×nstep
            # accumulation).
            dr_eff = rout[-1] - rout[-2] if rout.size >= 2 else 1.0
            tol = max(1e-6, 0.5 * float(dr_eff))
            if np.any(receiver.ranges > rout[-1] + tol):
                beyond = receiver.ranges[receiver.ranges > rout[-1] + tol]
                warnings.warn(
                    f"{self.model_name}: receiver ranges {beyond} exceed "
                    f"PE computed range; clamped to {rout[-1]}",
                    UserWarning,
                    stacklevel=2
                )
            rcv_ranges = np.clip(receiver.ranges, rout[0], rout[-1])

            # Interpolate real and imaginary parts separately. NaN samples
            # in the spectrum (e.g. above the seafloor for some backends,
            # or PE divergence) are zeroed before interpolation; warn if
            # any are present so the user knows their broadband output is
            # not fully converged.
            n_nan_p = int(np.count_nonzero(~np.isfinite(pressure)))
            if n_nan_p > 0:
                # expected; not in filterwarnings — emerges to user
                warnings.warn(
                    f"RAM:mpiramS broadband: {n_nan_p}/{pressure.size} "
                    f"complex samples are NaN/inf and have been zeroed "
                    f"for interpolation. Inspect the result before use.",
                    UserWarning, stacklevel=2
                )
            # Receivers outside the PE output domain return NaN pressure
            # so the resulting TL row is NaN (transparent in pcolormesh)
            # instead of saturating to ``TL_MAX_DB`` via PRESSURE_FLOOR.
            interp_re = RegularGridInterpolator(
                (zg.astype(np.float64), rout.astype(np.float64)),
                np.nan_to_num(pressure.real).astype(np.float64),
                method='linear', bounds_error=False, fill_value=np.nan
            )
            interp_im = RegularGridInterpolator(
                (zg.astype(np.float64), rout.astype(np.float64)),
                np.nan_to_num(pressure.imag).astype(np.float64),
                method='linear', bounds_error=False, fill_value=np.nan
            )

            range_mesh, depth_mesh = np.meshgrid(rcv_ranges, rcv_depths)
            points = np.column_stack([depth_mesh.ravel(), range_mesh.ravel()])
            p_re = interp_re(points).reshape(len(rcv_depths), len(rcv_ranges))
            p_im = interp_im(points).reshape(len(rcv_depths), len(rcv_ranges))
            pressure_rcv = p_re + 1j * p_im

            # Compute TL from interpolated pressure.
            #
            # Collins' RAM convention (ram1.5.f, User Guide eq 4):
            #   psi = uu * f3 has r^{-1/2} removed from the actual pressure.
            #   TL = -20*log10(|psi|) + 10*log10(r)
            #
            # In mpiramS, psif = psi * exp(i*(k0*r + pi/4)) / (4*pi),
            # so |psi| = |psif| * 4*pi.
            #
            # Protect 10*log10(r) from r=0; warn if we had to clip.
            log_ranges = rcv_ranges.astype(np.float64).copy()
            if log_ranges.size > 0 and log_ranges[0] <= 0.0:
                # expected; not in filterwarnings — emerges to user
                warnings.warn(
                    f"{self.model_name}: receiver range at index 0 is "
                    f"{log_ranges[0]}; clipping to dr={dr} for TL "
                    f"conversion to avoid log(0). The receiver.ranges "
                    f"array is not modified.",
                    UserWarning,
                    stacklevel=2
                )
                log_ranges[log_ranges <= 0.0] = dr

            # Convert the mpiramS .psif output to engineering travelling-
            # wave pressure (see ``models/_pe_phase.py``). ``Field.tl``
            # only needs |p|, but downstream consumers that do coherent
            # integration get a meaningful phase.
            with np.errstate(divide='ignore', invalid='ignore'):
                pressure_field = psi_to_travelling_wave(
                    pressure_rcv,
                    convention='mpiramS',
                    ranges_m=log_ranges,
                    range_axis=1,
                ).astype(np.complex128)

            elapsed = time.time() - start_time
            self._log(f"TL completed in {elapsed:.2f}s")

            field = Field(
                data=pressure_field,
                coords={'depth': receiver.depths, 'range': receiver.ranges},
                phase_reference='travelling_wave',
                **self._result_kwargs(
                    source,
                    backend='mpiramS',
                    frequencies=float(freq),
                    dr=float(dr), dz=float(dz),
                    c0=self._resolve_c0(env)
                )
            )
            field = field.mask_below_seafloor(env.bathymetry)
            self._attach_output_paths(
                field, fm.work_dir, '',
                primary_files=(('psif_file', 'psif.dat'),)
            )
            return field

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _run_broadband(self, env, source, receiver):
        """
        Run in broadband mode (native mpiramS use case).

        Returns complex transfer function psi(depth, frequency, range).
        """
        start_time = time.time()

        freq, Q_bb, T_bb = self._resolve_broadband_grid(source)
        zsrc = float(source.depths[0])
        ranges = receiver.ranges
        rmax = float(np.max(ranges))

        dr = float(self.dr) if self.dr is not None else None
        dz = float(self.dz) if self.dz is not None else None
        if dr is None or dz is None:
            dr_auto, dz_auto = self._compute_grid_lytaev(
                env, freq, max_range=rmax, kind='mpiramS'
            )
            if dr is None:
                dr = dr_auto
            if dz is None:
                dz = dz_auto
        self._log(
            f"mpiramS (broadband): fc={freq:.1f} Hz, Q={Q_bb}, T={T_bb}s, "
            f"dr={dr:.1f} m, dz={dz:.3f} m"
        )
        self._log(f"Bandwidth: {freq/Q_bb:.2f} Hz")

        fm = self._setup_file_manager()
        work_dir = fm.work_dir

        try:
            ssp_filename = self._prepare_ssp(env, work_dir, freq)
            bth_filename, ibot = self._prepare_bathymetry(env, rmax, work_dir)
            sedlayer, nzs, cs, rho_arr, attn_arr, isedrd, sed_filename = \
                self._prepare_bottom_properties(env, work_dir)

            write_ranges_file(work_dir / 'ranges.dat', ranges)

            rs = self.rs_stability if self.rs_stability is not None else rmax

            ihorz = 1 if env.has_range_dependent_ssp() else 0

            write_inpe(
                filepath=work_dir / 'in.pe',
                fc=freq,
                Q=Q_bb,
                T=T_bb,
                zsrc=zsrc,
                deltaz=dz,
                deltar=dr,
                np_pade=self.np_pade,
                nss=self.ns_stability,
                rs=rs,
                dzm=self.depth_decimation,
                ssp_filename=ssp_filename,
                iflat=1 if self.flat_earth else 0,
                ihorz=ihorz,
                ibot=ibot,
                bth_filename=bth_filename,
                sedlayer=sedlayer,
                nzs=nzs,
                cs=cs,
                rho=rho_arr,
                attn=attn_arr,
                isedrd=isedrd,
                sed_filename=sed_filename,
                c0_user=self._resolve_c0(env)
            )

            self._run_binary(work_dir)

            result = read_psif(work_dir)

            # mpiramS stores psif = ψ·exp(+i(k0 r + π/4)) / (4π) under the
            # exp(+iωt) (engineering) carrier sign opposite to the
            # outgoing-wave convention every other uacpy model uses.
            # Conjugating flips the carrier sign and the constant scale
            # 4π·exp(-iπ/4)/√r recovers Collins' p(f,r,z) = ψ·exp(+ik0 r)/√r
            # in the engineering travelling-wave form p ∝ ψ̄·exp(-ik0 r)/√r.
            psif = result['psif']  # (nzo, nf, nr)
            rout = result['rout']  # (nr,)
            zg = result['zg']
            # ``scale`` divides by sqrt(rout); rout=0 (mpiramS sometimes
            # emits a zero-range bin) would NaN the entire column. Mirror
            # the _run_tl clip+warn pattern.
            rout_safe = np.asarray(rout, dtype=np.float64).copy()
            if rout_safe.size > 0 and np.any(rout_safe <= 0.0):
                clip_to = float(dr) if dr and dr > 0 else 1.0
                # expected; not in filterwarnings — emerges to user
                warnings.warn(
                    f"RAM broadband: rout contained non-positive values "
                    f"(min={float(rout_safe.min())}); clipping to "
                    f"{clip_to} for the 1/sqrt(r) scaling. The returned "
                    f"`ranges` array is not modified.",
                    UserWarning, stacklevel=2
                )
                rout_safe[rout_safe <= 0.0] = clip_to
            # psif shape: (nzo, nf, nr) — convert to engineering
            # travelling-wave pressure via models/_pe_phase.py.
            pressure = psi_to_travelling_wave(
                psif,
                convention='mpiramS',
                ranges_m=rout_safe,
                range_axis=2,
            )

            # Map to receiver depth grid. PE domain extends below the
            # seafloor; output only the requested receiver depths.
            out_depths = receiver.depths
            if not np.array_equal(zg, out_depths):
                # Find interpolation indices and weights (zg is regular)
                dz_grid = zg[1] - zg[0] if len(zg) > 1 else 1.0
                idx_float = (out_depths - zg[0]) / dz_grid
                idx_lo = np.clip(np.floor(idx_float).astype(int), 0, len(zg) - 2)
                w = np.clip(idx_float - idx_lo, 0.0, 1.0)
                # Vectorized interpolation: (n_out, nf, nr)
                pressure = (pressure[idx_lo, :, :] * (1.0 - w[:, None, None]) +
                            pressure[idx_lo + 1, :, :] * w[:, None, None])
            else:
                out_depths = zg

            elapsed = time.time() - start_time
            self._log(f"Broadband completed in {elapsed:.2f}s")
            self._log(f"Output: {len(out_depths)} depths x {result['nf']} freqs x {result['nr']} ranges")

            # (n_d, n_r, n_f).
            pressure = np.moveaxis(pressure, 1, 2)

            tf = Field(
                data=pressure,
                coords={
                    'depth': out_depths,
                    'range': rout,
                    'frequency': result['frq'],
                },
                phase_reference='travelling_wave',
                **self._result_kwargs(
                    source,
                    backend='mpiramS',
                    frequencies=result['frq'],
                    dr=float(dr), dz=float(dz),
                    n_samples=result['Nsam'],
                    fs=result['fs'],
                    Q=result['Q'],
                    c0=result['c0'],
                    c_min=result['cmin'],
                )
            )
            # Mask sub-bottom samples with NaN for consistency with _run_tl.
            # RAM computes valid fields in the sediment, but other uacpy
            # models return NaN below the seafloor.
            bathy = np.asarray(env.bathymetry, dtype=float)
            seafloor = np.interp(rout, bathy[:, 0], bathy[:, 1])
            for j, bd in enumerate(seafloor):
                mask = out_depths > bd
                tf.data[mask, j, :] = np.nan
            self._attach_output_paths(
                tf, fm.work_dir, '',
                primary_files=(('psif_file', 'psif.dat'),)
            )
            return tf

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _run_binary(self, work_dir: Path):
        """Execute s_mpiram in the given working directory."""
        env = os.environ.copy()
        # Limit OpenMP threads to avoid oversubscription
        omp_source = 'inherited'
        if 'OMP_NUM_THREADS' not in env:
            env['OMP_NUM_THREADS'] = str(os.cpu_count() or 1)
            omp_source = 'auto = os.cpu_count()'
        self._log(
            f"Executing mpiramS: {self.executable} "
            f"(cwd={work_dir}, OMP_NUM_THREADS={env['OMP_NUM_THREADS']} "
            f"{omp_source})"
        )

        result = self._run_subprocess(
            [str(self.executable)],
            cwd=work_dir,
            timeout=self.timeout,
            env=env
        )

        if self.verbose and result.stdout:
            self._log(f"mpiramS output:\n{result.stdout}", level='debug')

        # Verify output exists
        if not (work_dir / 'psif.dat').exists():
            raise ModelExecutionError(
                self.model_name,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=(
                    "mpiramS produced no output file (psif.dat). "
                    "Check input parameters.\n" + (result.stderr or "")
                )
            )
