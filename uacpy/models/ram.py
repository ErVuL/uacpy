"""
RAM - Range-dependent Acoustic Model wrapper (multi-backend dispatcher)

The :class:`RAM` class auto-selects one of four vendored Collins-family PE
binaries based on the environment:

- **mpiramS** (default — fluid bottom + flat surface): Dushaw's Fortran 90/95
  rewrite of Collins' original RAM. Native broadband Q/T loop, MPI-ready
  upstream (uacpy builds the serial variant). Custom `inpe`/SSP/BTH multi-file
  input format via :mod:`uacpy.io.mpirams_writer`.
- **rams0.5** (any ``shear_speed > 0`` anywhere): Collins' RAMS elastic PE
  for sediments with shear waves. Collins-style ``rams.in`` input via
  :mod:`uacpy.io.ramsurf_writer`.
- **ramsurf1.5** (``env.altimetry is not None``): Collins' rough-surface /
  beach-geometry PE. Same writer as rams.
- **ramgeo** (fluid + flat surface with a layered bottom, narrowband):
  Collins' RAMGeo range-dependent-geoacoustics PE — its sediment layers
  parallel the bathymetry rather than lying flat (``ramgeo1.5.f:3-4``).

Elastic bottom + altimetry raises ``UnsupportedFeatureError`` — no published
Collins PE handles that combination; use OASES for range-independent elastic.

Run modes by backend:
- mpiramS: ``COHERENT_TL``, ``BROADBAND``, ``TIME_SERIES`` (native Q/T loop).
- rams0.5 / ramsurf1.5 / ramgeo: ``COHERENT_TL`` natively, plus ``BROADBAND``
  and ``TIME_SERIES`` via the patched complex-envelope output (``pcomplex.bin``,
  driven by ``_run_collins_broadband``). See ``third_party/MODIFICATIONS.md``
  for the upstream patch.

The lower boundary at zmax is an absorbing layer in all four backends, not
a rigid Neumann floor: mpiramS ramps the sediment attenuation to
``absorbing_layer_attn`` over the deepest ``absorbing_layer_width``
wavelengths of the PE domain, and ``_collins_range_segments`` writes the
same ramp into each Collins profile section.
"""

import numpy as np
import os
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, List, Union
from scipy.interpolate import RegularGridInterpolator

from uacpy.models.base import (
    PropagationModel, RunMode, ModelSpec, USER_FRAME_SKIP,
)
from uacpy.models._pe_phase import psi_to_travelling_wave
from uacpy.core.environment import (
    Environment, SeabedColumn,
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
# LAMBDA_PER_DZ_FLOOR — depth samples per acoustic wavelength that an
#   auto-picked dz is floored to (``dz >= c_min/(16·f)``); it bounds the cost
#   of the depth grid, which Lytaev's error model on its own does not. The
#   value is uacpy's — no samples-per-wavelength floor is prescribed by the
#   RAM sources, their readme, Collins 1993 or Lytaev 2023 §4.
# RAMS_DR_LAMBDA_CAP — empirical upper stability bound on dr for rams0.5's
#   rotated Padé elastic march, expressed as a divisor of c_min/freq.
#   ``dr ≤ c_min / (RAMS_DR_LAMBDA_CAP·f)`` ≈ 0.2 λ per step.
# Fortran array dimensions of the Collins binaries (``parameter (mr=…,mz=…)``
# at the top of each source); all three are enlarged locally over upstream, see
# third_party/MODIFICATIONS.md. ``mr`` bounds the bathymetry arrays. ``mz``
# bounds the depth arrays, but the codes do not consume it at the same rate:
# rams0.5 interleaves the elastic field vector and indexes 2*nz+4
# (rams0.5.f:154 ``do 3 i=1,2*nz+4``, and the 2*nz solve loops at :807, :811,
# :813), while the fluid codes index nz+2 (ramgeo1.5.f:138). The matching
# bounds check at rams0.5.f:141 is a uacpy addition — see
# third_party/MODIFICATIONS.md. ``mz`` is set so all three reach nz=20000.
# Each binary stops with a diagnostic on an overrun but writes no output, so it
# would otherwise surface as a truncated-file read — guard before launching.
_COLLINS_ARRAY_LIMITS = {
    'rams':    {'mr': 505, 'mz': 40004, 'nz_factor': 2, 'nz_pad': 4},
    'ramsurf': {'mr': 505, 'mz': 20002, 'nz_factor': 1, 'nz_pad': 2},
    'ramgeo':  {'mr': 505, 'mz': 20002, 'nz_factor': 1, 'nz_pad': 2},
}


# Lytaev grid-accuracy target used when the caller does not pin one.
DEFAULT_RAM_ACCURACY = 1e-3

LAMBDA_PER_DZ_FLOOR = 16.0
# ``zread`` pins each sediment-block point to the node ``i = 1.5 + z/dz`` and
# remembers only the *immediately* preceding index, so its collision push-down
# (``ramsurf1.5.f:208``, identical in ramgeo1.5.f:229 and rams0.5.f:232) protects
# one duplicate depth and no more. Two consecutive distinct depths landing in the
# same cell make the later point overwrite the earlier, and the fill loop at
# :218-219 then ramps linearly across the whole gap it left. A gap of at least
# ``BLOCK_GAP_PER_DZ * dz`` keeps the nodes distinct, since
# ``floor(x + g/dz) - floor(x) >= floor(g/dz)``.
BLOCK_GAP_PER_DZ = 2.0
RAMS_DR_LAMBDA_CAP = 5.0

# Ceiling on the number of range records a Collins binary writes. The stride
# ``ndr`` is otherwise lowered until the first written range reaches the
# nearest receiver, which for a near-field receiver on a long run would write
# one record per ``dr``.
_COLLINS_MAX_OUTPUT_RANGES = 20000

# Ceiling on the number of mpiramS sediment profiles written for a
# range-independent bottom whose seafloor water speed varies with range, and
# the speed spread (m/s) below which that range dependence is ignored.
_MAX_SED_PROFILES = 128
_CWG_RANGE_TOL = 0.01
# mpiramS/src/param.f90:10 — the radius its flat-earth transform uses.
_EARTH_RADIUS_M = 6378137.0

# Output files each family writes into the work dir. Cleared before launch so
# a pinned ``work_dir`` cannot hand an earlier run's output back as this run's
# answer — the binaries write fixed names, with no run-specific stem.
_COLLINS_OUTPUTS = ('tl.grid', 'pcomplex.bin')
_MPIRAMS_OUTPUTS = ('psif.dat',)


def _mask_below_seafloor(data, depths, ranges, bathymetry):
    """Set ``data`` samples below the local seafloor to NaN, in place.

    ``data`` has depth on axis 0 and range on axis 1 (any trailing freq/time
    axis broadcasts). RAM computes valid fields inside the sediment, but uacpy
    returns NaN below the seafloor for consistency with the other models.

    ``Field.mask_below_seafloor`` does the same thing but only accepts the
    canonical 2-D ``['depth', 'range']`` coords, so the broadband
    ``(depth, range, frequency)`` arrays here go through this helper instead.
    """
    if hasattr(bathymetry, 'to_pairs'):        # Bathymetry carrier → (N, 2)
        bathymetry = bathymetry.to_pairs()
    bathy = np.asarray(bathymetry, dtype=float)
    depths = np.asarray(depths, dtype=float)
    seafloor = np.interp(np.asarray(ranges, dtype=float), bathy[:, 0], bathy[:, 1])
    for j, bd in enumerate(seafloor):
        data[depths > bd, j] = np.nan
    return data


def _interp_envelope_to_receiver_grid(src_depths, src_ranges, psi,
                                      rcv_depths, rcv_ranges, *,
                                      carrier_rate=0.0):
    """Resample a complex PE envelope, interpolating modulus and phase apart.

    The Collins output grid is spaced ``dr·ndr`` — sized for Padé march
    accuracy, with nothing tying it to the envelope's range Nyquist. When
    ``c0`` sits far from the water-column speed, ψ still rotates fast enough
    that adjacent output samples are >90° apart, and interpolating the
    complex field averages across opposite-phase lobes: on the Pekeris
    reference (100 m, c=1500, half-space 1700/1.7/0.5, 250 Hz) that raises
    the median TL by 1.5-2.3 dB.

    The modulus has no carrier — it varies on the modal-beat scale — so it
    is well sampled and is interpolated on its own, tracking the binary's
    own ``tl.grid`` resampling to ~0.05 dB median. The unit phasor is
    interpolated separately and renormalised, so it carries phase only and
    contributes nothing to the level.

    ``carrier_rate`` (rad/m, see :meth:`RAM._collins_carrier_rate`) is the
    phase rate the backend baked into ψ. Linear interpolation of the unit
    phasor is only faithful while the rotation between two source samples
    stays under π, so the carrier is divided out on the source grid,
    the slowly varying residual is interpolated, and the carrier is
    restored on the receiver grid — the two are exact inverses at the
    source samples, so the grid values are untouched."""
    psi = np.asarray(psi)
    src_r = np.asarray(src_ranges, dtype=float)
    rcv_r = np.atleast_1d(np.asarray(rcv_ranges, dtype=float))
    if carrier_rate:
        psi = psi * np.exp(-1j * float(carrier_rate) * src_r)[None, :]
    mag = np.abs(psi)
    with np.errstate(invalid='ignore', divide='ignore'):
        unit = np.where(mag > 0.0, psi / mag, 0.0)
    mag_out = _interp_to_receiver_grid(
        src_depths, src_r, mag, rcv_depths, rcv_r)
    unit_out = _interp_to_receiver_grid(
        src_depths, src_r, unit, rcv_depths, rcv_r)
    with np.errstate(invalid='ignore', divide='ignore'):
        mod = np.abs(unit_out)
        unit_out = np.where(mod > 0.0, unit_out / mod, 1.0 + 0.0j)
    out = mag_out * unit_out
    if carrier_rate:
        out = out * np.exp(1j * float(carrier_rate) * rcv_r)[None, :]
    return out


def _interp_to_receiver_grid(src_depths, src_ranges, values,
                             rcv_depths, rcv_ranges, *, sanitize=False):
    """Bilinear-interpolate a PE-grid ``(depth, range)`` field onto the
    receiver grid. Real or complex ``values`` (complex via independent re/im);
    out-of-grid samples become NaN. ``sanitize`` zeroes NaN/inf first (the
    mpiramS pressure path). Returns ``(len(rcv_depths), len(rcv_ranges))``."""
    rd = np.atleast_1d(np.asarray(rcv_depths, dtype=float))
    rr = np.atleast_1d(np.asarray(rcv_ranges, dtype=float))
    grid = (np.asarray(src_depths, dtype=float), np.asarray(src_ranges, dtype=float))
    DD, RR = np.meshgrid(rd, rr, indexing='ij')
    pts = np.stack([DD.ravel(), RR.ravel()], axis=-1)

    def _one(v):
        v = np.asarray(v)
        if sanitize:
            v = np.nan_to_num(v)
        rgi = RegularGridInterpolator(grid, v.astype(np.float64),
                                      bounds_error=False, fill_value=np.nan)
        return rgi(pts).reshape(DD.shape)

    if np.iscomplexobj(values):
        return _one(np.real(values)) + 1j * _one(np.imag(values))
    return _one(values)


# Cap on profile sections added so a layered elastic seabed follows sloping
# bathymetry under rams0.5. Each section costs six blocks in the deck.
MAX_BATHY_SECTIONS = 64


#: Sub-bottom margin, in wavelengths, below which the absorbing layer has no
#: room to work. ``ram.pdf`` p.7 asks for the grid bottom "well below the ocean
#: bottom interface" with the attenuation raised "over the lower few
#: wavelengths of the grid"; three is that "few", calibrated against the
#: measured error curve rather than assumed.
_MIN_SUBBOTTOM_WAVELENGTHS = 3.0


class RAM(PropagationModel):
    """
    RAM - Range-dependent Acoustic Model (Parabolic Equation), multi-backend.

    A unified façade that picks one of three vendored Collins-family PE
    binaries at run-time based on the environment:

    ================================  =================================================
    Environment                       Backend selected
    ================================  =================================================
    fluid + flat, layered, narrowband ``ramgeo`` — Collins' RD layered fluid PE
    fluid + flat (simple / broadband) ``mpiramS`` — Dushaw's broadband PE (Q/T loop)
    elastic bottom (any shear>0)      ``rams0.5`` — Collins' elastic PE
    fluid bottom + altimetry          ``ramsurf1.5`` — Collins' rough-surface PE
    elastic + altimetry               ``UnsupportedFeatureError`` (no published PE)
    ================================  =================================================

    ``ramgeo`` tracks sediment layers *parallel to the bathymetry* — the most
    faithful Collins treatment of a sloping layered fluid seabed — and is
    auto-selected for narrowband (COHERENT_TL) layered cases. For a simple
    half-space mpiramS is preferred (native, vs ramgeo's synthetic-layer
    wrapping), but ramgeo *accepts* a simple bottom when forced. Like the
    other Collins backends it supports every run mode via uacpy's complex-
    envelope patch; auto-dispatch hands broadband / time-series to mpiramS's
    faster native sweep. Pass ``RAM(backend=...)`` to force a backend
    (``'mpiramS'`` / ``'ramgeo'`` / ``'rams'`` / ``'ramsurf'``).

    Use ``RAM(...).select_backend(env)`` to inspect the choice without
    actually running. Range-dependent SSP, bathymetry and (layered) bottom
    are supported by every backend: mpiramS threads them through its native
    range-dependent setup, and the Collins backends emit one ``ram.in``
    profile section per range break (each carrying its range-local SSP and
    Collins-style depth/value bottom profile).

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

    BROADBAND:
        Broadband complex pressure field. Returns ``Field`` with ψ(depth,
        frequency, range) for downstream IFFT to time domain. Available on
        every backend: mpiramS sweeps the (fc, Q, T) band inside the Fortran
        loop, the Collins backends run one subprocess per band frequency and
        read the patched ``pcomplex.bin``.

    TIME_SERIES:
        Real pressure p(t) at each receiver. Internally runs BROADBAND
        and convolves with ``source_waveform`` (sampled at ``sample_rate``).
        Returns ``Field`` / ``Field`` with shape (n_d, n_t, n_r). Available
        on every backend, same split as BROADBAND.

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
        Range step in meters. Default: None (selected by the Lytaev
        (2023) Padé-error optimizer; see ``accuracy`` / ``theta_max``).
        **[all backends]**
    dz : float, optional
        Depth step in meters. Default: None (selected by the Lytaev
        optimizer, then snapped so the shallowest bathymetry point sits on
        a depth grid node, capped at ``MAX_DEPTH_POINTS`` grid points, and
        floored at c_min/(16·freq) — 0.55·λ_s on rams). **[all backends]**
    np_pade : int, optional
        Number of Pade coefficients (2-10). Default: 6. **[all backends]**
    ns_stability : int, optional
        Number of stability terms. Default: 1 (use 0 for short ranges).
        **[mpiramS, ramgeo1.5, ramsurf1.5]** — rams0.5's row 5 carries
        (rams_irot, rams_theta) in these two slots instead.
    rs_stability : float, optional
        Stability range in meters. Default: max output range on mpiramS;
        on the Collins fluid codes ``None`` writes 0, which the binary
        then expands to ``2 × rmax`` (``if(rs.lt.dr) rs = 2.0*rmax``).
        **[mpiramS, ramgeo1.5, ramsurf1.5]**
    Q : float, optional
        Q value for broadband mode (half-bandwidth = fc/Q, so the band
        spans 2·fc/Q). Default: ``None``,
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
        at the centre frequency. Default: 20.0. **[all backends]**
    absorbing_layer_attn : float, optional
        Attenuation at the bottom of the absorbing layer, in dB per
        wavelength. Default: 10.0. **[all backends]**
    n_sed_points : int, optional
        Number of sediment-profile control points. Default: 1000, minimum 4.
        **[mpiramS]** — ``profl`` lays them out as [sea surface, seafloor,
        ``n_sed_points-3`` interior points spanning the seabed down to the top
        of the absorbing layer, domain floor] and interpolates linearly between
        them (``mpiramS/src/ram.f90:321-337``), so a material step is resolved
        to ``sedlayer/(n_sed_points-3)`` where ``sedlayer`` is that span
        (:meth:`_absorber_span`). The default keeps that interval far below an
        acoustic wavelength for ordinary geometries: at 50 points a 200 m span
        resolves its steps only to 4.3 m. Collins backends ignore this — they
        consume the layered bottom as Collins-style ``(depth, value)``
        breakpoints (see ``SeabedColumn.to_piecewise_breakpoints``).
    rams_theta : float or callable, optional
        Padé rotation angle in degrees for elastic stability (0 < theta
        < 90). Default: 45.0 (tuned against Kraken on the Pekeris-elastic
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

    # Declarative metadata (see PropagationModel / ModelSpec). RAM is the
    # range-dependent PE engine: every range-dependence axis is honoured by
    # some backend (mpiramS / rams0.5 / ramsurf1.5), so all flags except
    # multi_source_depth are True. _validate_forced_backend rejects real
    # per-backend mismatches at run() time. No collapse override — RAM uses
    # the base DEFAULT_COLLAPSE.
    spec = ModelSpec(
        modes=(RunMode.COHERENT_TL, RunMode.BROADBAND, RunMode.TIME_SERIES),
        supports={
            'altimetry',
            'range_dependent_bathymetry',
            'range_dependent_ssp',
            'range_dependent_bottom',
            'layered_bottom',
            'range_dependent_layered_bottom',
            'elastic_media',
        },
    )
    source = 'collins_ram'

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
        n_sed_points: int = 1000,
        c0: Optional[float] = None,
        timeout: float = 600.0,
        # ``dr`` / ``dz`` are picked by the Lytaev (2023) Padé-error
        # optimizer when not set explicitly. ``accuracy`` is the per-run
        # error budget; ``theta_max`` (degrees) bounds the PE spectrum.
        accuracy: Optional[float] = None,
        theta_max: float = 30.0,
        # Collins backends only — ignored when the dispatcher picks mpiramS.
        # `theta` is the Padé rotation angle (degrees, 0–90) used by RAMS
        # for elastic stability; defaults are tuned against Kraken on the
        # Pekeris-elastic problem. ``irot`` is the rotation flag (1 = on).
        rams_theta: float = 45.0,
        rams_irot: int = 1,
        # Multiplicative tightening of the Lytaev-optimised ``dr`` for
        # the ``rams`` backend. Independent of the ``c_min/(5·f)`` λ cap
        # also applied by ``_compute_grid_lytaev``; the tighter of the
        # two wins. Default 5.0 is empirically validated; raise for very
        # long-range or unusually noisy runs.
        rams_dr_safety_factor: float = 5.0,
        backend: Optional[str] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        collapse: Optional[Dict[str, str]] = None,
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
            Number of Pade coefficients (2-10). Default: 6.
        ns_stability : int, optional
            Number of stability terms. Default: 1.
        rs_stability : float, optional
            Stability range (m). None = max output range on mpiramS, 0 on
            the Collins fluid codes (which expand it to 2 × rmax).
            Default: None.
        Q : float, optional
            Q value for the broadband half-bandwidth (fc/Q; the band
            spans 2·fc/Q). Default: ``None``,
            which resolves to ``2.0`` for broadband paths and to
            ``1e6`` for COHERENT_TL (effectively single-frequency).
        T : float, optional
            Time window width (s). Default: ``None``, which resolves
            to ``10.0`` for broadband paths and to ``1.0`` for
            COHERENT_TL.
        backend : str, optional
            Force a specific RAM-family backend instead of automatic
            dispatch: ``'mpiramS'``, ``'ramgeo'``, ``'rams'`` or
            ``'ramsurf'``. ``None`` (default) auto-selects from the
            environment (see :meth:`select_backend`). A forced backend
            that cannot represent the environment (e.g. a fluid backend
            for an elastic bottom) raises ``ConfigurationError`` at run
            time.
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
            layered bottoms.  Default: 1000.
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
            Source-side maximum propagation angle (degrees) bounding the
            PE spectrum for the Lytaev optimiser and for Eq. (15)'s ``c₀``.
            30° is the standard wide-angle PE assumption. Default 30. The
            seabed term of Lytaev's ``θ_max = max(θ_max^src, θ_max^bottom)``
            is measured from ``env.bathymetry``, so a slope steeper than this
            widens the spectrum on its own (:meth:`_resolve_theta_max`).
        rams_dr_safety_factor : float, optional
            Tightening factor on the Lytaev-optimised ``dr`` for the
            rams backend (rotated Padé, Milinazzo-Zala-Brooke 1997).
            Applied alongside an independent ``dr ≤ c_min/(5·f)`` λ
            cap; the tighter of the two wins. Default 5.0 — set to 1.0
            to disable, raise for unusually noisy long-range runs.
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            timeout=timeout, cleanup=cleanup, collapse=collapse
        )

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        #
        # Keep the user's ``executable`` arg verbatim (``None`` when
        # auto-detected) so ``model.copy()`` re-resolves the binary instead of
        # re-pinning the already-resolved absolute path. The resolved mpiramS
        # path lives in ``self._exe``; the Collins binaries resolve per-run via
        # ``_collins_binary``.
        self.executable = Path(executable) if executable is not None else None
        if self.executable is None:
            self._exe = self._find_executable_in_paths(
                's_mpiram', bin_subdirs=['mpirams'], dev_subdir='mpiramS'
            )
        else:
            self._exe = self.executable

        if not self._exe.exists():
            raise ExecutableNotFoundError('RAM:mpiramS', str(self._exe))

        # mpiramS allocates its Padé arrays dynamically (epade.f90:30); the
        # Collins binaries carry ``parameter (mp=10)`` and stop above it
        # (``ramgeo1.5.f:142-145``). The bound below is the binaries' own, so
        # np_pade=9 and 10 — legal, and the widest-angle operators available —
        # are no longer refused by a margin nothing justified.
        if not isinstance(np_pade, int) or not (2 <= np_pade <= 10):
            raise ConfigurationError(
                f"np_pade must be an integer in [2, 10] (Collins mp=10, "
                f"ramgeo1.5.f:142-145); got {np_pade!r}."
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
        # ``profl`` lays out the sub-bottom as [surface, seafloor, nzs-3
        # interior points, domain floor] and only builds interior points when
        # nzs > 3 (mpiramS/src/ram.f90:321-329).
        if not isinstance(n_sed_points, int) or n_sed_points < 4:
            raise ConfigurationError(f"n_sed_points must be an integer >= 4; "
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

        # Kept raw so ``copy()`` round-trips: materialising the default
        # here would make every copy look caller-pinned. Resolved on use
        # via ``_accuracy`` / ``_accuracy_explicit``.
        self.accuracy = None if accuracy is None else float(accuracy)
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
        if backend is not None and backend not in self._BACKENDS:
            raise ConfigurationError(
                f"RAM(backend={backend!r}) is not a known backend. "
                f"Choose one of {sorted(self._BACKENDS)}, or None for "
                f"automatic dispatch."
            )
        self.backend = backend

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
                skip_file_prefixes=USER_FRAME_SKIP
            )

    @property
    def _accuracy(self) -> float:
        """The Lytaev accuracy target actually used."""
        return (DEFAULT_RAM_ACCURACY if self.accuracy is None
                else float(self.accuracy))

    @property
    def _accuracy_explicit(self) -> bool:
        """True when the caller pinned ``accuracy``.

        A grid cap that misses uacpy's own default target is a status
        fact; missing a target the caller asked for is a warning.
        """
        return self.accuracy is not None

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
        bounds = self._speed_bounds(env)
        if bounds is None:
            return DEFAULT_SOUND_SPEED
        c_min, c_max = bounds
        return float(optimal_c0(c_min, c_max, self._resolve_theta_max(env)))

    def _resolve_theta_max(self, env: Environment) -> float:
        """Maximum propagation angle (degrees) bracketing the Padé spectrum.

        Lytaev (2023, https://doi.org/10.3390/jmse11030496) §5.5 estimates it
        as ``θ_max = max(θ_max^src, θ_max^bottom)``, where ``θ_max^bottom`` is
        the steepest slope between bottom and water, taken from the bathymetry
        relief. ``theta_max`` on the constructor supplies the source aperture;
        the seabed term is measured here, so a slope steeper than it widens
        ``[ξ_min, ξ_max]`` instead of leaving the auto grid coarser than the
        ``accuracy`` budget implies. Capped just under 90° — the bracket is
        undefined at grazing.
        """
        theta = float(self.theta_max)
        bathy = getattr(env, 'bathymetry', None)
        if bathy is None or bathy.n_ranges < 2:
            return theta
        r = np.asarray(bathy.ranges, dtype=float)
        z = np.asarray(bathy.depths, dtype=float)
        dr = np.diff(r)
        valid = dr > 0.0
        if not np.any(valid):
            return theta
        slope = np.degrees(np.arctan(np.abs(np.diff(z))[valid] / dr[valid]))
        return float(min(max(theta, float(np.max(slope))), 89.0))

    def _resolve_c_max(self, env: Environment) -> float:
        """Fastest compressional speed (m/s) the PE domain meshes through.

        The RAM domain extends well below the seafloor, so at long range the
        earliest arrival is the bottom-refracted path travelling at the
        fastest *seabed* speed, not the fastest water speed. Tagged on every
        result as ``c_max``; the time-series synthesis helpers read it to
        anchor the output window ahead of that arrival.
        """
        bounds = self._speed_bounds(env)
        return float(bounds[1]) if bounds else self._resolve_c0(env)

    @staticmethod
    def _speed_bounds(env: Environment):
        """Slowest / fastest compressional speeds (m/s) anywhere in ``env``.

        Water column plus every bottom layer and half-space. Returns
        ``None`` when the environment declares no speeds at all.
        """
        # Every profile, not just the one at r=0: a range-dependent SSP can
        # hold its extremes in any column, and they set c0 and the Pade
        # spectrum width for the whole run.
        speeds = [float(c) for c in np.asarray(env.ssp.data).ravel()
                  if np.isfinite(c)]
        speeds.extend(c for c in env.bottom.all_sound_speeds() if c)
        if not speeds:
            return None
        return float(min(speeds)), float(max(speeds))

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
            fc = float(freqs[0])
            frq = self._broadband_frequencies(fc, Q, T)
            if self.Q is None or self.T is None:
                warnings.warn(
                    f"RAM BROADBAND: a single source frequency does not define "
                    f"a band, and mpiramS / Collins march an internal "
                    f"(fc, Q, T) sweep. fc={fc:g} Hz was given Q={Q:g} "
                    f"({'pinned' if self.Q is not None else 'default'}) and "
                    f"T={T:g} s "
                    f"({'pinned' if self.T is not None else 'default'}), i.e. "
                    f"{np.size(frq)} bins over "
                    f"{float(np.min(frq)):.4g}-{float(np.max(frq)):.4g} Hz "
                    f"(df = 1/T = {1.0 / T:.4g} Hz).",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
                )
            return fc, Q, T

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
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
        self._broadband_frequencies(fc, Q, T)
        return fc, Q, T

    @staticmethod
    def _broadband_frequencies(fc: float, Q: float, T: float) -> np.ndarray:
        """The symmetric frequency vector a ``(fc, Q, T)`` sweep marches.

        Reproduces ``peramx.f90:345-362``: half-bandwidth ``bw = fc/Q``,
        ``df = fs/Nsam = 1/T``, ``nf1 = int((bw - df)/df) + 1`` and
        ``frq(ii) = -(nf1 - (ii-1))·df + fc`` for ``ii = 1..2·nf1+1``. Every
        backend sweeps this same vector — mpiramS inside the Fortran loop, the
        Collins backends one subprocess per element.

        ``frq(1) = fc - nf1·df`` goes non-positive for small ``Q``, which no
        PE can march. The serial driver uacpy builds has no guard and writes
        NaN bins at zero and negative frequency; its MPI sibling stops on
        exactly this test and names ``Q`` (``peramx_mpi.f90:417-423``).
        """
        bw = float(fc) / float(Q)
        df = 1.0 / float(T)
        nf1 = max(1, int((bw - df) / df) + 1)
        frq = (np.arange(2 * nf1 + 1, dtype=float) - nf1) * df + float(fc)
        if frq[0] <= 0.0:
            raise ConfigurationError(
                f"RAM broadband: Q={Q:g} and T={T:g} s put the lower band edge "
                f"at {frq[0]:.4g} Hz (fc={fc:g} Hz, half-bandwidth fc/Q="
                f"{bw:.4g} Hz, Δf=1/T={df:.4g} Hz, {2 * nf1 + 1} bins). A PE "
                f"cannot march zero or negative frequencies. Use Q > 1 so the "
                f"half-bandwidth stays below fc, or shorten T so Δf is coarser."
            )
        return frq

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
            # mpiramS reaches a pinned zmax only through here; the Collins
            # path resolves it in _resolve_collins_grid and guards it there.
            # Both backends clamp the seafloor index the same way
            # (mpiramS/src/ram.f90:101 iz=min(nz,iz), ramgeo1.5.f:135), so the
            # seabed-outside-the-grid pathology is not Collins-specific:
            # measured 29 dB silent error on mpiramS with zmax below depth.
            self._warn_if_seafloor_outside_grid(self.zmax, env, freq=freq)
            return self.zmax
        return self._adequate_zmax(env, freq, c0)

    def _adequate_zmax(self, env: Environment, freq: float,
                       c0: Optional[float] = None) -> float:
        """The grid bottom ``ram.pdf`` p.7 asks for: the seafloor plus one
        cell plus the absorbing layer.

        Split out from :meth:`_compute_zmax` so the seafloor guard can compare
        a pinned ``zmax`` against it without recursing back through the pinned
        branch.
        """
        if c0 is None:
            c0 = self._resolve_c0(env)
        wavelength = c0 / max(freq, 1.0)
        absorbing_width = self.absorbing_layer_width * wavelength
        dz_for_pad = (float(self.dz) if self.dz is not None
                      else self._compute_dz(env, freq, c0))
        return env.depth + dz_for_pad + absorbing_width

    @staticmethod
    def _flat_earth_depth(z: float) -> float:
        """``peramx.f90:264-266``'s depth map, ``eps = z/Re``,
        ``z' = z(1 + eps/2 + eps²/3)`` with ``Re`` from ``param.f90:10``."""
        eps = float(z) / _EARTH_RADIUS_M
        return float(z) * (1.0 + eps / 2.0 + eps * eps / 3.0)

    @classmethod
    def _flat_earth_depth_inverse(cls, z_transformed: float) -> float:
        """Depth whose flat-earth image is ``z_transformed``.

        The map is monotone and near-identity (``z/Re`` is ~1e-4 for ocean
        depths), so the fixed point converges in a couple of passes.
        """
        z = float(z_transformed)
        for _ in range(4):
            eps = z / _EARTH_RADIUS_M
            z = float(z_transformed) / (1.0 + eps / 2.0 + eps * eps / 3.0)
        return z

    def _mpirams_zmax(self, env: Environment, freq: float, dz: float) -> float:
        """PE domain depth for an mpiramS march, snapped onto the ``deltaz``
        grid.

        ``peramx.f90:374,387`` sizes the depth grid as
        ``icount = floor(zmax/deltaz - 0.5) + 2`` and then fills it with
        ``linspace(zg, 0, zmax, icount)``, so its actual spacing is
        ``zmax/(icount-1)`` while the depth operator (``ram.f90:51``
        ``cfact = 0.5/deltaz**2``, consumed at ``matrc.f90:60-62``) and the
        seafloor index (``ram.f90:101`` ``iz = floor(1 + zbc/deltaz)``) both
        assume ``deltaz``. The two agree only when ``zmax`` is an exact
        multiple of ``deltaz``.

        The value snapped has to be the one ``:371`` actually reads. Under
        ``flat_earth`` (the default) ``:260-266`` rescales the whole depth axis
        *before* ``zmax = maxval(zw)``, so snapping the geometric depth leaves
        the transformed grid off the multiple again. Snapping in the
        transformed frame and mapping back is what makes the spacing exact on
        both paths.
        """
        zmax = self._compute_zmax(env, freq)
        transformed = (self._flat_earth_depth(zmax) if self.flat_earth
                       else zmax)
        snapped = (int(np.floor(transformed / float(dz) - 0.5)) + 1) * float(dz)
        result = (self._flat_earth_depth_inverse(snapped) if self.flat_earth
                  else snapped)
        if self.zmax is not None and abs(result - float(self.zmax)) > 1e-9:
            self._log(
                f"zmax {float(self.zmax):.6g} m snapped to {result:.6g} m so "
                f"mpiramS's depth grid reproduces dz={float(dz):g} m exactly"
            )
        return result

    def _prepare_ssp(self, env: Environment, work_dir: Path,
                     freq: float, dz: float) -> str:
        """
        Write SSP file from environment. Returns filename.

        The SSP is extended to ``_mpirams_zmax``, which is what mpiramS reads
        back as its domain depth (``zmax = maxval(zw)``, ``peramx.f90:371``).

        The column carries the user's true profile at every depth, sub-bottom
        included. ``matrc.f90:43-55`` reads ``cwg`` as a water property only
        above the seafloor index, and below it ``cwg`` is merely the reference
        the sediment speed is rebuilt against (``csg = cwg + cs``,
        ``mpiramS/src/ram.f90:332-333``) — which
        :meth:`_sediment_offsets` cancels per control point. Holding the column
        flat below the seabed instead would corrupt real water: mpiramS picks
        this column by nearest neighbour (``ram.f90:295-296``) while
        interpolating the seafloor continuously (``ram.f90:315-317``), so on a
        slope every range between two written columns takes a flat-start depth
        that is not its own seafloor.

        The range axis is the SSP's own breaks and nothing else: the column
        is not keyed to the seabed, so it depends on the bathymetry only
        through the profiles the caller declared.
        """
        zmax_pe = self._mpirams_zmax(env, freq, dz)
        n_points = max(50, int(zmax_pe / float(dz) / 2))
        depths = np.linspace(0, zmax_pe, n_points)

        def _column(rng: float) -> np.ndarray:
            return self._ssp_column(env, rng, depths)

        ssp_filename = 'ssp.dat'
        ranges_m = (np.unique(np.asarray(env.ssp.ranges, dtype=float))
                    if env.ssp.is_range_dependent else np.zeros(1))
        if len(ranges_m) > 1:
            self._log(f"Writing {len(ranges_m)} SSP profiles")
            speeds_2d = np.column_stack([_column(r) for r in ranges_m])
            write_ssp_file(work_dir / ssp_filename, depths, speeds_2d, ranges_m)
        else:
            write_ssp_file(work_dir / ssp_filename, depths, _column(0.0))

        return ssp_filename

    def _prepare_bathymetry(self, env: Environment, rmax: float, work_dir: Path) -> tuple:
        """
        Write bathymetry file. Returns (bth_filename, ibot).
        """
        bth_filename = 'bathy.dat'

        bathy = env.bathymetry.to_pairs()
        if bathy[0, 0] > 0.0:
            bathy = np.vstack([[0.0, bathy[0, 1]], bathy])
        if bathy[-1, 0] < rmax:
            bathy = np.vstack([bathy, [rmax, bathy[-1, 1]]])
        write_bth_file(work_dir / bth_filename, bathy[:, 0], bathy[:, 1])
        return bth_filename, 1

    @staticmethod
    def _ssp_column(env: Environment, rng: float, depths) -> np.ndarray:
        """The water sound speed uacpy writes at ``depths`` for range ``rng``.

        One evaluator for both sides of the deck: ``_prepare_ssp`` writes
        ``ssp.dat`` from it and the sediment builders subtract it. That shared
        definition is what makes ``csg = cwg + cs`` (``ram.f90:332-333``)
        reproduce the requested absolute bottom speed — the offset is only
        correct against the very column mpiramS will read back.

        Depths outside the tabulation hold the end values, matching how the
        profile is written.
        """
        pairs = (env.ssp.eval(range=rng).to_pairs()
                 if env.ssp.is_range_dependent else env.ssp.to_pairs())
        return np.interp(np.asarray(depths, dtype=float),
                         pairs[:, 0], pairs[:, 1])

    @staticmethod
    def _control_point_depths(seafloor: float, sedlayer: float, nzs: int,
                              zmax: float) -> np.ndarray:
        """Absolute depths of ``profl``'s ``nzs`` sediment control points.

        ``zwork = [0, d, d + k·sedlayer/(nzs-3) (k = 1..nzs-3),
        max(zg(n), zwork(nzs-1)+1e-6)]`` (``mpiramS/src/ram.f90:321-329``), so
        points ``2..nzs-1`` are ``linspace(0, sedlayer, nzs-2)`` below the
        local seafloor ``d``, point 1 sits at the sea surface and point ``nzs``
        at the domain floor.
        """
        z = np.empty(int(nzs), dtype=float)
        z[0] = 0.0
        z[1:nzs - 1] = float(seafloor) + np.linspace(0.0, float(sedlayer),
                                                     int(nzs) - 2)
        z[nzs - 1] = max(float(zmax), z[nzs - 2] + 1e-6)
        return z

    def _sediment_offsets(self, env, rng, cp_abs, nzs, sedlayer, zmax):
        """``cs`` for one range: the offset that rebuilds ``cp_abs`` from the
        water column mpiramS reads at each control point.

        ``profl`` reconstructs the sediment speed as ``csg = cwg + cs``
        (``ram.f90:332-333``) with ``cwg`` the water profile splined onto the
        *whole* depth grid, sub-bottom included — so a single scalar offset
        only reproduces ``cp_abs`` where ``cwg`` happens to be flat. Taking the
        offset per control point makes the sub-bottom speed exact for any
        water column, which is what lets ``_prepare_ssp`` write the true SSP
        instead of holding it flat below the seabed.
        """
        seafloor = float(np.asarray(env.bathymetry.eval(range=rng)).flat[0])
        z_ctrl = self._control_point_depths(seafloor, sedlayer, nzs, zmax)
        return np.asarray(cp_abs, dtype=float) - self._ssp_column(env, rng,
                                                                  z_ctrl)

    def _water_speed_at_seafloor(self, env: Environment,
                                 range: float = 0.0) -> float:
        """
        Water-column sound speed at the *local* seafloor for range ``range``.

        This is what mpiramS's ``cwg`` is at the water-sediment interface.
        ``profl`` builds the sediment speed as ``csg = cwg + cs``
        (``ram.f90:332-333``) on a depth grid anchored at the seafloor depth
        *at that range*, so the offset that reproduces an absolute bottom
        speed ``cb`` is ``cb - cwg(z_seafloor(range))``. Referencing it to the
        deepest bathymetry point instead leaves the sediment fast wherever the
        seabed rises above it.
        """
        seafloor = float(np.asarray(env.bathymetry.eval(range=range)).flat[0])
        if env.has_range_dependent_ssp:
            ssp = env.ssp.eval(range=range).to_pairs()
        else:
            ssp = env.ssp.to_pairs()
        return float(np.interp(seafloor, ssp[:, 0], ssp[:, 1]))

    def _seafloor_speed_ranges(self, env: Environment) -> List[float]:
        """Ranges at which the sediment offsets change.

        ``cs`` is referenced to the water column at control points that sit
        below the *local* seafloor (:meth:`_sediment_offsets`), so it moves
        with the bathymetry — and it moves **continuously**, because
        ``ram.f90:315-317`` interpolates the seafloor between breakpoints while
        ``:295-296`` selects the nearest written profile. Sampling only the
        declared breaks would therefore leave every range between them
        referenced to a seafloor that is not its own.

        The bathymetry span is sampled uniformly and unioned with the declared
        SSP breaks, then decimated to ``_MAX_SED_PROFILES``.
        """
        breaks = {0.0}
        bathy = getattr(env, 'bathymetry', None)
        if bathy is not None and bathy.n_ranges > 1:
            breaks.update(float(r) for r in bathy.ranges)
            r = np.asarray(bathy.ranges, dtype=float)
            breaks.update(float(v) for v in np.linspace(
                r.min(), r.max(), _MAX_SED_PROFILES))
        if env.has_range_dependent_ssp:
            breaks.update(float(r) for r in env.ssp.ranges)
        ranges = sorted(breaks)
        if len(ranges) > _MAX_SED_PROFILES:
            keep = np.unique(
                np.linspace(0, len(ranges) - 1, _MAX_SED_PROFILES).astype(int))
            ranges = [ranges[i] for i in keep]
        return ranges

    def _absorber_span(self, env: Environment, freq: float,
                       zmax: float) -> float:
        """Depth below the deepest seafloor at which the absorbing layer starts.

        ``profl`` interpolates the sediment arrays linearly between control
        point ``nzs-1`` at ``seafloor + sedlayer`` and control point ``nzs`` at
        ``zmax`` (``mpiramS/src/ram.f90:321-329,332-337``), and uacpy raises
        only the last point to ``absorbing_layer_attn``. The absorbing layer is
        therefore exactly the span ``[seafloor + sedlayer, zmax]``, so
        ``sedlayer`` is what sets its width — it is not a free choice.

        Collins sizes that layer explicitly: "the bottom of the computational
        grid (the depth zmax) is placed well below the ocean bottom interface
        and the attenuation is increased over the lower **few wavelengths** of
        the grid" (RAM manual). Running the ramp from the seabed instead
        replaces the seabed's own attenuation with an artificial gradient over
        the whole sub-bottom, which absorbs energy the seabed should have
        returned.

        This is the same quantity ``_ramp_absorbing_attenuation`` gives the
        Collins backends as ``max(z_sediment_base, z_bottom - absorbing_width)``
        — computing it once here is what keeps the two decks describing one
        absorber. The caller floors it with the modelled sediment thickness so
        the layer never eats into the seabed.
        """
        absorbing_width = (self.absorbing_layer_width * self._resolve_c0(env)
                           / max(float(freq), 1.0))
        return (float(zmax) - float(env.depth)) - absorbing_width

    def _prepare_bottom_properties(self, env: Environment, work_dir: Path,
                                   absorber_span: float, zmax: float):
        """
        Extract bottom properties from environment and convert to mpiramS format.

        mpiramS's sediment model (profl in ram.f90) uses an N-point profile
        (``n_sed_points``) interpolated over depth points
        [0, seafloor, ..interior.., seafloor+sedlayer, zmax]:

        - cs: sediment sound speed *perturbation* relative to water column,
              taken per control point (:meth:`_sediment_offsets`).
        - rho: sediment density (g/cm^3).
        - attn: sediment attenuation (dB/wavelength).
              The last point is set to absorbing-layer attenuation.

        ``absorber_span`` (:meth:`_absorber_span`) is the depth below the
        seafloor where the absorbing layer starts; every builder stretches
        ``sedlayer`` to it, floored by the modelled sediment thickness.
        ``zmax`` locates the final control point.

        Returns (sedlayer, nzs, cs, rho, attn, isedrd, sed_filename). Dispatches
        to a per-bottom-shape builder; each returns that same 7-tuple.
        """
        nzs = self.n_sed_points
        sedlayer = max(self._effective_dz(), float(absorber_span))

        if env.has_range_dependent_layered_bottom:
            return self._bottom_rd_layered(env, work_dir, nzs, sedlayer, zmax)
        if env.has_layered_bottom:
            return self._bottom_layered(env, work_dir, nzs, sedlayer, zmax)
        if env.has_range_dependent_bottom:
            return self._bottom_rd_halfspace(env, work_dir, nzs, sedlayer, zmax)
        return self._bottom_halfspace(env, work_dir, nzs, sedlayer, zmax)

    @staticmethod
    def _sample_layered_column(col, nzs: int, sedlayer: float):
        """Sample ``col`` onto mpiramS's ``nzs`` sediment control points.

        ``profl`` places them at ``zwork = [0, d, d + k·sedlayer/(nzs-3)
        (k = 1..nzs-3), max(zg(n), zwork(nzs-1)+1e-6)]``
        (``mpiramS/src/ram.f90:321-329``) and interpolates the supplied arrays
        between them linearly (``gorp``, ``ram.f90:332-337`` and ``:360-390``).
        Points ``2..nzs-1`` are therefore ``linspace(0, sedlayer, nzs-2)`` below
        the local seafloor, point 1 sits at the sea surface and point ``nzs`` at
        the domain floor — the same depths :meth:`_control_point_depths`
        returns, which is what lets :meth:`_sediment_offsets` subtract the
        water column control point by control point.

        Point ``nzs-1`` carries the half-space, so the layer stack ends in a
        step resolved to ``sedlayer/(nzs-3)`` and every depth from there to the
        domain floor is constant.

        Constant is the physically required profile there, not merely the
        convenient one: everything below the half-space top is the absorbing
        layer, whose purpose is to swallow downgoing energy so it cannot
        reflect off the truncated grid at ``zmax``. Collins states the design
        directly — "the bottom of the computational grid (the depth zmax) is
        placed well below the ocean bottom interface and the attenuation is
        increased over the lower few wavelengths of the grid" (RAM manual;
        Collins & Siegmann, *Parabolic Wave Equations with Applications*,
        §on absorbing layers) — and Jensen, Kuperman, Porter & Schmidt,
        *Computational Ocean Acoustics* §7.5 adds the constraint that "the
        sponge layer must be designed such that the internal reflections are
        insignificant". A sound-speed gradient inside the sponge is itself a
        refracting, partially reflecting feature, which is exactly what the
        layer exists to prevent; only the attenuation may vary through it.
        Spreading the layer/half-space contrast linearly to ``zmax`` puts that
        gradient across the whole absorber.

        Point 1 repeats the top-of-sediment value: ``matrc`` reads the bottom
        arrays only below the seafloor index
        (``mpiramS/src/matrc.f90:43-55``), so it never enters the water column
        and the water/sediment interface stays a step.
        """
        cp, rho, attn = col.sample_at_depths(nzs - 2, max_thickness=sedlayer)
        cp[-1] = col.halfspace.sound_speed
        rho[-1] = col.halfspace.density
        attn[-1] = col.halfspace.attenuation
        return (np.concatenate(([cp[0]], cp, [cp[-1]])),
                np.concatenate(([rho[0]], rho, [rho[-1]])),
                np.concatenate(([attn[0]], attn, [attn[-1]])))

    def _bottom_rd_layered(self, env, work_dir, nzs, sedlayer, zmax):
        """Range-dependent *layered* seabed → per-range sediment .sed profiles."""
        rdl = env.bottom
        n_ranges = len(rdl.ranges)
        sedlayer_rdl = max(rdl.max_total_thickness(), sedlayer)

        cs_profiles = np.zeros((nzs, n_ranges))
        rho_profiles = np.zeros((nzs, n_ranges))
        attn_profiles = np.zeros((nzs, n_ranges))

        for i in range(n_ranges):
            lb = rdl.columns[i]
            cs_samp, rho_samp, attn_samp = self._sample_layered_column(
                lb, nzs, sedlayer_rdl)

            cs_profiles[:, i] = self._sediment_offsets(
                env, rdl.ranges[i], cs_samp, nzs, sedlayer_rdl, zmax)

            rho_profiles[:, i] = rho_samp
            attn_profiles[:, i] = attn_samp
            attn_profiles[-1, i] = self.absorbing_layer_attn

        sed_filename = self._write_sediment_profiles(
            work_dir, rdl.ranges, cs_profiles, rho_profiles, attn_profiles)

        self._log(f"Range-dependent layered sediment: {n_ranges} profiles, "
                  f"nzs={nzs}, sedlayer={sedlayer_rdl:.1f} m")

        cs = cs_profiles[:, 0].copy()
        rho_arr = rho_profiles[:, 0].copy()
        attn_arr = attn_profiles[:, 0].copy()
        return sedlayer_rdl, nzs, cs, rho_arr, attn_arr, 1, sed_filename

    def _bottom_layered(self, env, work_dir, nzs, sedlayer, zmax):
        """Range-independent *layered* seabed → single sediment profile,
        or one profile per range break when the water speed at the seafloor
        varies with range (see :meth:`_water_speed_at_seafloor`)."""
        col = env.bottom.columns[0]
        sedlayer_lay = max(col.total_thickness(), sedlayer)

        cs_samp, rho_arr, attn_arr = self._sample_layered_column(
            col, nzs, sedlayer_lay)
        attn_arr[-1] = self.absorbing_layer_attn

        self._log(f"Layered bottom: {len(col.layers)} layers, "
                  f"nzs={nzs}, sedlayer={sedlayer_lay:.1f} m")

        return self._offsets_over_seafloor_speeds(
            env, work_dir, cs_samp, nzs, sedlayer_lay, zmax, rho_arr, attn_arr)

    def _bottom_rd_halfspace(self, env, work_dir, nzs, sedlayer, zmax):
        """Range-dependent *halfspace* seabed → per-range sediment .sed profiles."""
        bottom_rd = env.bottom
        n_ranges = len(bottom_rd.ranges)

        cs_profiles = np.zeros((nzs, n_ranges))
        rho_profiles = np.zeros((nzs, n_ranges))
        attn_profiles = np.zeros((nzs, n_ranges))

        cp_arr = bottom_rd.halfspace_sound_speed
        rho_view = bottom_rd.halfspace_density
        attn_view = bottom_rd.halfspace_attenuation
        for i in range(n_ranges):
            cs_profiles[:, i] = self._sediment_offsets(
                env, bottom_rd.ranges[i], np.full(nzs, float(cp_arr[i])),
                nzs, sedlayer, zmax)

            rho_profiles[:, i] = rho_view[i]
            attn_profiles[:, i] = attn_view[i]
            attn_profiles[-1, i] = self.absorbing_layer_attn

        sed_filename = self._write_sediment_profiles(
            work_dir, bottom_rd.ranges, cs_profiles, rho_profiles,
            attn_profiles)

        self._log(f"Range-dependent sediment: {n_ranges} profiles, nzs={nzs}")

        cs = cs_profiles[:, 0].copy()
        rho_arr = rho_profiles[:, 0].copy()
        attn_arr = attn_profiles[:, 0].copy()
        return sedlayer, nzs, cs, rho_arr, attn_arr, 1, sed_filename

    def _bottom_halfspace(self, env, work_dir, nzs, sedlayer, zmax):
        """Range-independent *halfspace* seabed (the Environment default).

        Emits one profile per range break when the water speed at the seafloor
        varies with range (see :meth:`_water_speed_at_seafloor`).
        """
        # ``env.bottom`` is always a coerced ``Bottom`` carrier (Environment
        # defaults None → a half-space), so ``halfspace_at`` returns real
        # geoacoustics — no fabricated fallback.
        hs = env.bottom.halfspace_at(range=0.0)
        cb_val = float(hs.sound_speed)
        rho_val = float(hs.density)
        attn_val = float(hs.attenuation)

        rho_arr = np.full(nzs, rho_val)
        attn_arr = np.full(nzs, attn_val)
        attn_arr[-1] = self.absorbing_layer_attn
        cp_abs = np.full(nzs, cb_val)

        return self._offsets_over_seafloor_speeds(
            env, work_dir, cp_abs, nzs, sedlayer, zmax, rho_arr, attn_arr)

    def _offsets_over_seafloor_speeds(self, env, work_dir, cp_abs, nzs,
                                      sedlayer, zmax, rho_arr, attn_arr):
        """Turn one range-independent sediment column ``cp_abs`` (absolute
        speeds at the ``nzs`` control points) into the 7-tuple
        :meth:`_prepare_bottom_properties` returns.

        The column is range-independent but its *offsets* need not be: one
        profile per range break whenever the water speed at the seafloor moves
        (:meth:`_varying_seafloor_speeds`), a single profile otherwise. Density
        and attenuation are the same column at every range — only ``cs`` is
        referenced to the local water column.
        """
        ranges, _cwg = self._varying_seafloor_speeds(env)
        if ranges is None:
            cs = self._sediment_offsets(env, 0.0, cp_abs, nzs, sedlayer, zmax)
            return sedlayer, nzs, cs, rho_arr, attn_arr, 0, ''

        cs_profiles = np.column_stack([
            self._sediment_offsets(env, r, cp_abs, nzs, sedlayer, zmax)
            for r in ranges
        ])
        sed_filename = self._write_sediment_profiles(
            work_dir, ranges, cs_profiles,
            np.repeat(rho_arr[:, None], len(ranges), axis=1),
            np.repeat(attn_arr[:, None], len(ranges), axis=1),
        )
        return (sedlayer, nzs, cs_profiles[:, 0].copy(), rho_arr, attn_arr,
                1, sed_filename)

    def _varying_seafloor_speeds(self, env: Environment):
        """``(ranges, cwg)`` when the water speed at the seafloor varies with
        range, else ``(None, None)``.

        A range-independent bottom still needs one sediment profile per range
        whenever ``cwg`` moves, because mpiramS reconstructs the sediment
        speed as ``cwg + cs`` against the *local* water column.
        """
        ranges = self._seafloor_speed_ranges(env)
        if len(ranges) < 2:
            return None, None
        cwg = np.array([self._water_speed_at_seafloor(env, r) for r in ranges])
        if float(np.ptp(cwg)) <= _CWG_RANGE_TOL:
            return None, None
        self._log(
            f"Seafloor water speed varies {cwg.min():.1f}-{cwg.max():.1f} m/s "
            f"over range; writing {len(ranges)} sediment profiles so the "
            f"sediment speed stays referenced to the local seafloor."
        )
        return ranges, cwg

    @staticmethod
    def _write_sediment_profiles(work_dir, ranges, cs_profiles,
                                 rho_profiles, attn_profiles) -> str:
        """Write the mpiramS ``.sed`` deck and return its filename."""
        from uacpy.io.mpirams_writer import write_sediment_file
        sed_filename = 'sediment.sed'
        write_sediment_file(work_dir / sed_filename, np.asarray(ranges, float),
                            cs_profiles, rho_profiles, attn_profiles)
        return sed_filename

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
        if run_mode not in (RunMode.BROADBAND, RunMode.TIME_SERIES):
            # BROADBAND is covered inside ``_prepare_timeseries``; warning
            # here too would double up.
            self._warn_ignored_run_kwargs(
                run_mode,
                source_waveform=source_waveform,
                sample_rate=sample_rate,
                output_duration=output_duration,
            )
        source_waveform, frequencies = self._prepare_timeseries(
            run_mode, source, frequencies, source_waveform, sample_rate,
            output_duration,
        )

        if frequencies is not None:
            freqs_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))
            if freqs_arr.size == 0:
                raise ConfigurationError(
                    "RAM.run(frequencies=…) requires at least one positive "
                    "frequency"
                )
            source = Source(
                depths=source.depths, frequencies=freqs_arr,
                source_type=source.source_type,
                beam_pattern=source.beam_pattern,
            )

        env = self._project_environment(env)
        # Finish all env-shaping before validation so validate_inputs sees the
        # final env. Surface shear is no backend's concern, and select_backend
        # keys only on bottom elasticity / altimetry, so this reordering does
        # not change dispatch.
        env = self._drop_unsupported_surface_shear(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)
        self._warn_on_dropped_absorption(env)

        backend = self.select_backend(env, run_mode)
        elastic = self._env_has_elastic_bottom(env)
        rough = getattr(env, 'altimetry', None) is not None
        self._log(
            f"Dispatching to {backend} backend "
            f"(elastic_bottom={elastic}, altimetry={rough})"
        )
        self._warn_on_mpirams_only_overrides(backend)

        if backend == 'mpiramS':
            if run_mode == RunMode.BROADBAND:
                return self._run_broadband(env, source, receiver)
            if run_mode == RunMode.TIME_SERIES:
                tf = self._run_broadband(env, source, receiver)
                return tf.synthesize_time_series(
                    source_waveform=source_waveform,
                    sample_rate=sample_rate
                )
            return self._run_tl(env, source, receiver)

        # Collins backends: rams0.5 / ramsurf1.5 / ramgeo are single-frequency
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

        for col in env.bottom.columns:
            for layer in col.layers:
                _maybe_add(getattr(layer, 'shear_speed', None))
            _maybe_add(getattr(col.halfspace, 'shear_speed', None))
        return min(speeds) if speeds else 0.0

    @staticmethod
    def _env_has_elastic_bottom(env: Environment) -> bool:
        """Return True if any bottom container carries shear_speed > 0."""
        return env.has_elastic_bottom

    _BACKENDS = ('mpiramS', 'ramgeo', 'rams', 'ramsurf')

    def select_backend(self, env: Environment, run_mode=None) -> str:
        """Inspect which RAM-family binary will run for a given environment.

        Useful for diagnostics and tests — call this before ``run()`` to
        confirm dispatch without executing the binary. When the model was
        constructed with an explicit ``backend=``, that choice is returned
        (after a compatibility check) regardless of the environment.

        Parameters
        ----------
        run_mode : RunMode, optional
            The run mode the dispatch is for. Only ``COHERENT_TL`` (the
            default) routes a fluid+flat *layered* bottom to ``ramgeo``;
            broadband / time-series stay on ``mpiramS`` (native
            multi-frequency sweep). ``None`` assumes ``COHERENT_TL``.

        Returns
        -------
        str
            ``'ramgeo'`` (fluid + flat, narrowband layered; sediment layers
            parallel to bathymetry — a simple bottom is accepted when forced),
            ``'mpiramS'`` (fluid + flat: simple bottom, or broadband),
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
        ConfigurationError
            When a forced ``backend=`` cannot represent the environment.
        """
        if self.backend is not None:
            self._validate_forced_backend(self.backend, env)
            return self.backend
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
        # Fluid + flat: RAMGEO for narrowband TL through a *layered* bottom —
        # its sediment layers track (parallel) the bathymetry, the most
        # faithful Collins treatment of a sloping layered seabed. mpiramS keeps
        # the broadband path and the simple half-space cases, where it models
        # the seabed natively (ramgeo would wrap it in a synthetic layer, a
        # small accuracy cost). ramgeo still *accepts* a simple bottom when
        # forced via backend='ramgeo'.
        if self._prefer_ramgeo(env, run_mode):
            return 'ramgeo'
        return 'mpiramS'

    def _prefer_ramgeo(self, env: Environment, run_mode) -> bool:
        """RAMGEO is auto-selected for a narrowband (COHERENT_TL) fluid,
        flat-surface environment whose bottom is layered. A simple half-space
        is still accepted when ``backend='ramgeo'`` is forced."""
        if run_mode is not None and run_mode != RunMode.COHERENT_TL:
            return False
        return env.bottom.is_layered

    def _validate_forced_backend(self, backend: str, env: Environment) -> None:
        """Reject a forced ``backend=`` that cannot represent ``env``."""
        elastic = self._env_has_elastic_bottom(env)
        rough = getattr(env, 'altimetry', None) is not None
        if backend in ('mpiramS', 'ramgeo', 'ramsurf') and elastic:
            raise ConfigurationError(
                f"RAM(backend={backend!r}) is a fluid PE and cannot model the "
                f"elastic bottom (shear>0) in this environment. Use "
                f"backend='rams', or backend=None for automatic dispatch."
            )
        if backend in ('mpiramS', 'ramgeo', 'rams') and rough:
            raise ConfigurationError(
                f"RAM(backend={backend!r}) models a flat pressure-release "
                f"surface and cannot honour env.altimetry. Use "
                f"backend='ramsurf', or backend=None for automatic dispatch."
            )
        if backend == 'ramsurf' and not rough:
            raise ConfigurationError(
                "RAM(backend='ramsurf') needs a variable surface "
                "(env.altimetry). For a flat surface use backend='mpiramS' / "
                "'ramgeo', or backend=None for automatic dispatch."
            )
        if backend == 'rams' and not elastic:
            # rams0.5 is the elastic (RAMS) PE. Run on a fluid bottom its
            # shear machinery degenerates and it returns a null field —
            # TL saturated at 200 dB at every range — rather than failing.
            # Mirrors the ramsurf rule above: a backend whose defining
            # feature is absent from the env is a configuration error.
            raise ConfigurationError(
                "RAM(backend='rams') is the elastic PE and needs a bottom "
                "with shear (shear_speed > 0); on a fluid bottom it returns "
                "a null field. Use backend='mpiramS' / 'ramgeo', or "
                "backend=None for automatic dispatch."
            )

    def _collins_binary(self, kind: str) -> Path:
        """Resolve the path to a Collins-family binary on disk."""
        # ram1.5 (Collins fluid PE) is intentionally not built — uacpy
        # uses mpiramS for fluid+flat (broadband + RD bottom support).
        # ramgeo (range-dependent layered fluid) lives in its own vendor
        # dir; rams0.5 / ramsurf1.5 share the ramsurf/ tree.
        if kind == 'ramgeo':
            return self._find_executable_in_paths(
                'ramgeo', bin_subdirs=['ramgeo'], dev_subdir='ramgeo'
            )
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

    def _rams_rot0(self, theta: float) -> complex:
        """``rot0`` of the rams0.5 rotated-Padé scalar ``g0``.

        Mirrors ``rpade`` (``third_party/ramsurf/rams0.5.f:859-892``);
        ``epade`` uses ``rot0 = 1`` when ``rams_irot = 0``.
        """
        if self.rams_irot != 1:
            return 1.0 + 0.0j
        tfact = np.exp(-1j * float(theta) * np.pi / 360.0)
        den = float(2 * int(self.np_pade) + 1)
        rot0 = 1.0 + 0.0j
        for j in range(1, int(self.np_pade) + 1):
            pade1 = (2.0 / den) * np.sin(j * np.pi / den) ** 2
            pade2 = np.cos(j * np.pi / den) ** 2
            rot0 += pade1 * (tfact ** 2 - 1.0) / (1.0 + pade2 * (tfact ** 2 - 1.0))
        return complex(rot0 / tfact)

    def _collins_carrier_rate(self, env: Environment, kind: str,
                              freq: float, theta: float) -> float:
        """Phase rate (rad/m) carried by a Collins backend's stored envelope.

        Two terms, both fast enough to alias across a ``dr·ndr`` output step:

        * Every backend writes ψ with only ``exp(i k0 r)`` factored out, so
          what remains still rotates at ``k_r - k0``. ``c0`` is the Lytaev
          expansion point rather than a medium speed, so that difference is
          not small — 0.09 rad/m against a 21 m Lytaev ``dr`` on the 250 Hz
          Pekeris reference, i.e. 1.9 rad per output step. ``k_r`` is taken
          at the mean water sound speed, which sits inside the propagating
          modal band.
        * ``rams0.5`` additionally multiplies u by ``g0 = exp(i k0 dr rot0)``
          on every range step (``rams0.5.f:848-851``), i.e. the whole
          carrier is baked in — ~0.89 rad/m at 250 Hz. The rotation makes
          ``rot0`` complex (``rams0.5.f:865-888``); only ``Re(rot0)`` is a
          phase rate — ``Im(rot0)`` is the rotation's amplitude decay per
          step, which does not alias and must not be divided out.

        :func:`_interp_envelope_to_receiver_grid` divides this out before
        interpolating and restores it afterwards.
        """
        c0 = self._resolve_c0(env)
        k0 = 2.0 * np.pi * float(freq) / c0
        speeds = np.asarray(env.ssp.data, dtype=float)
        speeds = speeds[np.isfinite(speeds)]
        c_water = float(np.mean(speeds)) if speeds.size else c0
        rate = 2.0 * np.pi * float(freq) / c_water - k0
        if kind == 'rams':
            rate += k0 * self._rams_rot0(theta).real
        return float(rate)

    @staticmethod
    def _collins_output_stride(dr: float, max_range: float, rcv_ranges):
        """``(ndr, rmax_march)`` — the Collins range-output stride and the
        ``rmax`` to write into the input deck.

        The binaries write only at ``r = k·dr·ndr`` and test ``r < rmax``
        *after* writing (``rams0.5.f:79-80``, ``ramsurf1.5.f:52-53``,
        ``ramgeo1.5.f:83-84``), so a decimated march handed ``rmax =
        max_range`` verbatim stops up to ``(ndr-1)·dr`` short of the farthest
        receiver and the outermost receiver column comes back NaN.
        ``rmax_march`` extends the march to the first written range at or
        beyond ``max_range``, and is set half a range step short of it so the
        stop test fires exactly on that write — single-precision drift in the
        binary's ``r = r + dr`` sum cannot then cost the final record.

        ``ndr`` is capped so the *first* written range ``dr·ndr`` is not past
        the nearest receiver either, subject to a ceiling on the number of
        output ranges.
        """
        ndr = max(1, int((max_range / dr) / 1000.0))
        rr = np.atleast_1d(np.asarray(rcv_ranges, dtype=float))
        near = rr[rr > 0.0]
        if near.size:
            ndr = max(1, min(ndr, int(np.floor(float(near.min()) / dr))))
        ndr = max(ndr, int(np.ceil(max_range / dr / _COLLINS_MAX_OUTPUT_RANGES)))
        block = dr * ndr
        # The epsilon absorbs the rounding of ``max_range / block`` when the
        # two divide exactly: ceil() on a value a few ulps above the integer
        # would buy a whole extra output block.
        n_blocks = max(1, int(np.ceil(max_range / block - 1e-9)))
        return ndr, float((n_blocks * ndr - 0.5) * dr)

    @staticmethod
    def _collins_mz_budget(kind: str, zmax: float):
        """``(needed, mz, min_dz)`` for one Collins depth grid.

        ``needed`` is a callable ``needed(dz) -> int`` giving how many ``mz``
        slots the binary's own ``nz = zmax/dz - 0.5`` consumes at that ``dz``,
        per the backend's indexing; ``min_dz`` is the coarsest-resolution
        bound that fits, with half a grid point of headroom against the
        Fortran truncation.
        """
        limits = _COLLINS_ARRAY_LIMITS.get(kind)
        if limits is None:
            return None
        nz_max = (limits['mz'] - limits['nz_pad']) // limits['nz_factor']

        def needed(dz):
            nz = int(zmax / dz - 0.5)
            return limits['nz_factor'] * nz + limits['nz_pad']

        return needed, limits['mz'], zmax / (nz_max - 0.5)

    def _check_source_row_is_solved(self, zs: float, dz: float) -> None:
        """Reject a source shallower than one ``dz``.

        Every RAM binary plants the source with the same two statements —
        ``si=1.0+zs/dz`` / ``is=ifix(si)``, then splits the amplitude across
        ``u(is)`` and ``u(is+1)`` (``ramgeo1.5.f:389-393``,
        ``ramsurf1.5.f:396-400``, ``rams0.5.f:357-361``,
        ``mpiramS/src/ram.f90:110-114``). And every solver starts its sweep at
        row 2 (``ramgeo1.5.f:312``, ``rams0.5.f:836``,
        ``mpiramS/src/solvetri.f90:47``), so **row 1 is never written again**
        while being read into every step (``ramgeo1.5.f:320``).

        With ``zs < dz`` the index is 1, so a fraction of the source is frozen
        in ``u(1)`` for the whole march and acts as a permanent Dirichlet
        source sitting on the pressure-release surface. The field comes out
        far too loud — the opposite of the physics, which requires it to get
        *quieter* as the source approaches the surface. Measured against
        Kraken as an independent arbiter: ~46 dB mean, 72 dB peak, on
        uacpy's own default grid, with no warning on any backend.

        This is the flat-surface case of the same row-1 kill
        :meth:`_check_source_below_depressed_surface` catches for a ramsurf
        keel; it needs no altimetry and applies to all four backends.
        """
        if float(dz) <= 0.0 or float(zs) >= float(dz):
            return
        raise ConfigurationError(
            f"RAM: source at {float(zs):.4g} m is shallower than one depth "
            f"cell (dz={float(dz):.4g} m), so selfs plants it at index 1 "
            f"(si=1.0+zs/dz, ifix). No solver writes row 1 — every sweep "
            f"starts at 2 — so the amplitude is frozen there for the whole "
            f"march and acts as a permanent source on the pressure-release "
            f"surface. Measured ~46 dB too loud against Kraken.",
            remediation=(f"Set dz <= {float(zs):.4g} m so the source lands at "
                         f"index 2 or deeper, or move the source below "
                         f"{float(dz):.4g} m."),
        )

    def _check_rams_seafloor_index_floor(self, kind: str, dz: float,
                                         bathymetry) -> None:
        """Reject a track that drives ``rams0.5``'s seafloor index below 2.

        ``rams0.5.f:135`` and ``:305`` both assign ``iz=z/dz`` with no clamp,
        where ``ramgeo1.5.f:134`` and ``ramsurf1.5.f:119`` apply
        ``max(2,iz)``. Below 2 the binary indexes before the start of its own
        arrays: ``matrc:418`` runs ``do 2 i=ia-1,iz`` with ``ia=min(iz,jz)=1``
        and reads ``lamw(0)``, and at ``iz=1`` ``matrc:717`` forms
        ``i0=2*iz-3=-1`` and reads ``r5(0)``.

        The reason this is worth refusing rather than warning: at ``iz=1`` the
        run exits 0 and returns a **fully finite, entirely plausible** field.
        A bounds-checked build of the same deck aborts. Only ``iz=0`` is loud
        (all-NaN).

        ``updat`` recomputes the index from the *interpolated* bathymetry at
        every range step (``:305``), so the shallowest point anywhere on the
        track decides this — not ``env.depth``, which is the deepest. Every
        other depth guard in this class is written against the maximum, which
        is why this one is separate.
        """
        if kind != 'rams' or not bathymetry:
            return
        shallowest = min(float(d) for _, d in bathymetry)
        if shallowest >= 2.0 * float(dz):
            return
        raise ConfigurationError(
            f"rams: the track shoals to {shallowest:.4g} m, which is less "
            f"than 2*dz ({2.0 * float(dz):.4g} m), so the seafloor index "
            f"iz=z/dz falls below 2. rams0.5 does not clamp it "
            f"(rams0.5.f:135, :305 — unlike ramgeo1.5.f:134 / "
            f"ramsurf1.5.f:119), and matrc then reads lamw(0) at :418 and "
            f"r5(0) at :717. At iz=1 the run still exits 0 and returns a "
            f"plausible finite field, so nothing downstream can catch it.",
            remediation=(f"Set dz <= {shallowest / 2.0:.4g} m, or use "
                         f"backend='ramgeo'/'ramsurf' (which clamp) if the "
                         f"seabed needs no shear."),
        )

    def _check_collins_array_limits(self, kind: str, dz: float, zmax: float,
                                    bathymetry, surface=None) -> None:
        """Reject a run that would overrun a Collins binary's fixed arrays.

        ``mr`` bounds the range arrays ``rb``/``zb`` (and ``rsrf``/``zsrf`` on
        ramsurf); ``mz`` bounds the depth arrays, at the per-backend rate
        recorded in :data:`_COLLINS_ARRAY_LIMITS`.

        The profiles actually written are passed in rather than re-derived from
        ``env``: the writer appends a point to reach the last receiver, and the
        Fortran read loop stores the ``-1 -1`` terminator at index ``N+1``
        before testing ``i.gt.mr`` (``ramgeo1.5.f:146``, ``ramsurf1.5.f:131``),
        so the true capacity is ``mr - 1`` written points.

        Each binary does test its own depth bound and ``stop`` with a
        diagnostic, but exits before writing ``tl.grid``, so the failure would
        otherwise reach the caller as a ``FileFormatError`` about a truncated
        file with the real message lost. ``ramsurf`` does not even check its
        surface arrays — ``ramsurf1.5.f:131`` is fed by the *bathymetry*
        counter — so an over-long altimetry corrupts memory instead. Only a
        ``dz`` the caller pinned can reach the depth check; an auto-picked grid
        is coarsened to fit in ``_resolve_collins_grid``.
        """
        limits = _COLLINS_ARRAY_LIMITS.get(kind)
        if limits is None:
            return
        mr = limits['mr']
        for label, profile in (('bathymetry', bathymetry),
                               ('altimetry', surface)):
            if profile is None:
                continue
            # +1 for the terminator the Fortran stores past the last point.
            if len(profile) + 1 > mr:
                raise ConfigurationError(
                    f"RAM(backend={kind!r}): the run writes {len(profile)} "
                    f"{label} points but the binary's arrays hold {mr} "
                    f"(mr={mr}, one slot taken by the list terminator), so it "
                    f"would overrun them. Decimate env.{label} to at most "
                    f"{mr - 1} points, or use backend='mpiramS' (no fixed "
                    f"limit)."
                )
        needed, mz, dz_min = self._collins_mz_budget(kind, zmax)
        if needed(dz) > mz:
            raise ConfigurationError(
                f"RAM(backend={kind!r}): dz={dz:g} m over a zmax={zmax:.1f} m "
                f"domain needs {needed(dz)} depth slots but the binary's "
                f"arrays hold {mz} (mz={mz}). Coarsen to dz>={dz_min:.4g} m, "
                f"lower zmax, or use backend='mpiramS' (no fixed limit)."
            )

    def _run_collins(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        kind: str
    ) -> Result:
        """Run a Collins-family binary (rams0.5 / ramsurf1.5 / ramgeo) at
        ``source.frequencies[0]`` and return a TL Field.

        Wraps :meth:`_run_collins_one_freq` and converts the binary's
        ``tl.grid`` to a :class:`Field` interpolated onto the
        requested receiver grid.
        """
        fc = float(np.atleast_1d(source.frequencies)[0])
        theta = self._theta_for_freq(fc)
        raw = self._run_collins_one_freq(
            env, source, receiver, kind=kind, freq=fc, theta=theta
        )

        psi_clamped = self._clamp_collins_envelope(raw, env, kind)

        # Receivers outside the PE output grid get NaN so pcolormesh and
        # downstream consumers render them transparent rather than as a
        # saturated edge band.
        rcv_d = np.atleast_1d(receiver.depths).astype(float)
        rcv_r = np.atleast_1d(receiver.ranges).astype(float)
        psi_out = _interp_envelope_to_receiver_grid(
            raw['depths'], raw['ranges'], psi_clamped, rcv_d, rcv_r,
            carrier_rate=self._collins_carrier_rate(env, kind, fc, theta))

        # Same per-backend phase bookkeeping as the broadband loop; the
        # Collins files already carry the 1/√r scaling, so apply_radial=False.
        pressure = psi_to_travelling_wave(
            psi_out,
            convention='ramsurf' if kind == 'ramgeo' else kind,
            ranges_m=rcv_r,
            range_axis=1,
            k0=2.0 * np.pi * fc / self._resolve_c0(env),
            apply_radial=False,
        ).astype(np.complex128)

        # Mask sub-seafloor samples with NaN (same semantics as every backend).
        _mask_below_seafloor(pressure, rcv_d, rcv_r, env.bathymetry)

        field = Field(
            data=pressure,
            coords={'depth': rcv_d, 'range': rcv_r},
            phase_reference='travelling_wave',
            **self._result_kwargs(
                source,
                backend=kind,
                frequencies=fc,
                dr=raw['dr'], dz=raw['dz'], zmax=raw['zmax'],
                c0=self._resolve_c0(env),
                c_max=self._resolve_c_max(env),
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

    def _clamp_collins_envelope(self, raw: dict, env: Environment,
                                kind: str) -> np.ndarray:
        """Sanitise one Collins run's complex envelope, warning on divergence.

        Invalid samples are NaN/inf or an unphysically negative TL. A
        negative TL implies ``|p/p0| > 1`` (field gain), impossible for a
        passive medium, so it is rotated-Padé elastic divergence (rams0.5
        on a fast-shear seabed) rather than a real value — folded into the
        same clamp-and-warn path so the failure is visible instead of a
        plausible-but-wrong number. A tiny negative near the source
        (``|TL| < NEG_TL_TOL``) is numerical noise around 0 and is clamped
        up to 0, not flagged.

        The clamp acts on the envelope so phase is preserved: an invalid
        sample's level is pinned to the ``TL_MAX_DB`` floor, its direction
        kept.
        """
        NEG_TL_TOL = 1.0  # dB
        tl_raw = np.asarray(raw['tl'], dtype=float)
        invalid = ~np.isfinite(tl_raw) | (tl_raw < -NEG_TL_TOL)
        n_invalid = int(np.count_nonzero(invalid))
        if n_invalid > 0:
            note = ""
            if env.bottom.is_elastic and kind == 'rams':
                note = (" The Collins rams0.5 elastic PE is numerically "
                        "unstable for fast shear speeds; use OAST / Scooter "
                        "for an elastic seabed.")
            warnings.warn(
                f"RAM:{kind}: {n_invalid}/{tl_raw.size} TL samples at "
                f"f={float(raw['frequency']):.2f} Hz are NaN/inf or "
                f"unphysically negative (Padé instability or PE divergence) "
                f"and have been clamped to {TL_MAX_DB} dB. Try a smaller dr "
                f"or larger np_pade.{note}",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP
            )
        psi_raw = np.asarray(raw['pcomplex'], dtype=np.complex128)
        floor_mag = 10.0 ** (-TL_MAX_DB / 20.0)
        mag = np.abs(psi_raw)
        with np.errstate(invalid='ignore', divide='ignore'):
            unit = np.where(mag > 0.0, psi_raw / mag, 1.0 + 0.0j)
        # TL < 0 means |p/p0| > 1; cap the magnitude at unity to match the
        # ``np.maximum(tl_raw, 0.0)`` the dB path applies.
        mag = np.where(invalid, floor_mag, np.minimum(mag, 1.0))
        return mag * unit

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
        zmax_override: Optional[float] = None,
        fm=None,
    ) -> dict:
        """Execute a Collins-family binary once at ``freq`` and read both
        outputs. Returns a dict with raw arrays (no Field wrapping).

        Keys: ``tl`` (depth × range, dB), ``pcomplex`` (depth × range,
        complex envelope ``u·f3 / sqrt(r)``), ``ranges`` / ``depths``
        (binary output grid), ``dr`` / ``dz`` / ``zmax`` (the values
        actually used).

        ``fm`` lets a caller sweeping many frequencies own one work
        directory for the whole sweep; each call clears the stale outputs
        before running, so the directory is reused rather than multiplied.
        When ``fm`` is None this call creates and finalises its own.
        """
        binary = self._collins_binary(kind)

        # The Collins binaries handle one (zs, fc) per call; mpiramS does
        # the same. ``base.validate_inputs`` guards the extra source depths
        # and frequencies. Range-dependent SSP and (layered) bottom ARE
        # threaded through — one ``ram.in`` profile section per range break,
        # built in ``_collins_range_segments``.

        fc = float(freq)
        zs = float(np.atleast_1d(source.depths)[0])

        max_range = float(np.max(np.atleast_1d(receiver.ranges)))
        dr, dz, zmax = self._resolve_collins_grid(
            env, fc, kind, max_range,
            dr_override, dz_override, zmax_override,
        )
        # Built before the stride because the section spacing bounds ``dr``:
        # the binary consumes at most one profile section per range step.
        range_segments = self._collins_range_segments(env, kind, zmax, fc)
        bathy_r = [float(r) for r, _ in env.bathymetry.to_pairs().tolist()]
        alti_r = None
        if kind == 'ramsurf' and env.altimetry is not None:
            alti_r = [float(r) for r, _ in env.altimetry.to_pairs().tolist()]
        dr = self._constrain_dr_to_sections(
            dr, range_segments,
            # A dr pinned on the CONSTRUCTOR is just as user-set as one passed
            # to run(); testing only the override rewrites it in silence.
            pinned=(dr_override is not None or self.dr is not None),
            bathymetry_ranges=bathy_r, altimetry_ranges=alti_r)
        ndr, rmax_march = self._collins_output_stride(
            dr, max_range, receiver.ranges)
        ndz = max(1, int(self.depth_decimation))

        rcv_d = np.atleast_1d(receiver.depths).astype(float)
        target_depth = float(np.max(rcv_d))
        zmplt = self._collins_zmplt(max(target_depth, float(env.depth)),
                                    dz, zmax, ndz, kind)
        z_deepest = self._collins_deepest_output(zmplt, dz, ndz, kind)
        # Receivers below the deepest stored output sample come back NaN from
        # ``_interp_to_receiver_grid`` (fill_value=nan); warn so the empty
        # rows are attributable. Reachable only when ``zmax`` clamps ``zmplt``.
        if target_depth > z_deepest:
            # expected; not in filterwarnings — emerges to user
            warnings.warn(
                f"RAM:{kind}: receiver depths up to {target_depth:.1f} m "
                f"exceed the PE domain (zmax={zmax:.1f} m, deepest stored "
                f"output sample {z_deepest:.1f} m); samples below it "
                f"are returned as NaN. Increase zmax to cover all receiver "
                f"depths.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP
            )

        r_first = dr * ndr
        near = np.atleast_1d(np.asarray(receiver.ranges, dtype=float))
        near = near[near > 0.0]
        if near.size and float(near.min()) < r_first:
            # expected; not in filterwarnings — emerges to user
            warnings.warn(
                f"RAM:{kind}: the binary writes its first output range at "
                f"{r_first:.3f} m (dr={dr:.3f} m × ndr={ndr}); receiver ranges "
                f"below that are returned as NaN. Pin a smaller dr to move the "
                f"first output range in.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP
            )
        if ndz > 1:
            # rams0.5 writes from grid index 1+ndz, the fluid codes from ndz.
            z_first = (ndz if kind == 'rams' else ndz - 1) * dz
            in_gap = rcv_d[(rcv_d > 0.0) & (rcv_d < z_first)]
            if in_gap.size:
                warnings.warn(
                    f"RAM:{kind}: depth_decimation={ndz} makes the shallowest "
                    f"computed output depth {z_first:.3f} m; {in_gap.size} "
                    f"receiver depth(s) below it are interpolated between the "
                    f"pressure-release surface and that sample. Set "
                    f"depth_decimation=1 to resolve them.",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP
                )

        bathymetry = [(float(r), float(d)) for r, d in env.bathymetry.to_pairs().tolist()]
        if bathymetry[-1][0] < max_range:
            bathymetry.append((float(max_range), bathymetry[-1][1]))

        surface = (self._build_ramsurf_surface(env, max_range)
                   if kind == 'ramsurf' else None)
        if surface is not None:
            self._check_source_below_depressed_surface(surface, zs, dz)

        # Checked here rather than in ``_run_collins`` so the broadband sweep,
        # which calls this method directly, is covered too; and after the
        # profiles are built so the bound sees exactly what gets written.
        self._check_collins_array_limits(kind, dz, zmax, bathymetry, surface)
        self._check_rams_seafloor_index_floor(kind, dz, bathymetry)
        self._check_source_row_is_solved(zs, dz)

        owns_fm = fm is None
        fm = self._setup_file_manager() if owns_fm else fm
        try:
            # rams0.5 hardcodes 'rams.in'; ramsurf1.5 reads 'ram.in'.
            in_name = {'rams': 'rams.in', 'ramgeo': 'ramgeo.in'}.get(
                kind, 'ram.in')
            ram_in = fm.get_path(in_name)
            c0_pe = self._resolve_c0(env)
            # ``tl.line``'s receiver depth indexes u(ir)/f3(ir+1) with no bounds
            # check (ramsurf1.5.f:427, rams0.5.f:251), so keep it inside the
            # binary's own nz = zmax/dz - 0.5 depth arrays.
            nz_march = int(zmax / dz - 0.5)
            zr_line = min(target_depth, max(0.0, (nz_march - 1) * dz))
            write_ramin(
                str(ram_in),
                kind=kind,
                fc=fc, zs=zs, zr_line=zr_line,
                rmax=rmax_march, dr=dr, ndr=ndr,
                zmax=zmax, dz=dz, ndz=ndz, zmplt=zmplt,
                c0=c0_pe, np_pade=int(self.np_pade),
                ns_stab=int(self.ns_stability),
                rs_stab=float(self.rs_stability or 0.0),
                irot=int(self.rams_irot),
                theta=float(theta),
                bathymetry=bathymetry,
                surface=surface,
                range_segments=range_segments,
                title=f"uacpy {kind} run @ {fc:.1f} Hz"
            )

            self._log(f"Executing: {binary} (cwd={fm.work_dir})")
            self._clear_stale_outputs(fm.work_dir, _COLLINS_OUTPUTS)
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
            # rams0.5 writes its output grid from index 1+ndz, ramsurf1.5
            # from ndz (third_party/ramsurf/{rams0.5,ramsurf1.5}.f outpt).
            depth_index_offset = 1 if kind == 'rams' else 0
            ranges, depths, tl = read_tl_grid(
                tlgrid, dr=dr, ndr=ndr, dz=dz, ndz=ndz,
                depth_index_offset=depth_index_offset
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
                pcgrid, dr=dr, ndr=ndr, dz=dz, ndz=ndz,
                depth_index_offset=depth_index_offset
            )
            depths, tl, pcomplex = self._prepend_surface_node(
                depths, tl, pcomplex)

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
            if owns_fm and fm.cleanup:
                fm.cleanup_work_dir()

    @staticmethod
    def _collins_zmplt(target_depth: float, dz: float, zmax: float,
                       ndz: int = 1, kind: str = 'ramgeo') -> float:
        """Row-4 ``zmplt`` that makes the binary store an output sample at or
        below ``target_depth``.

        The Collins codes assign the real expression ``zmplt/dz - 0.5`` to the
        integer ``nzplt`` — a truncation (``ramgeo1.5.f:131``,
        ``ramsurf1.5.f:112``, ``rams0.5.f:133``) — and run their output loop up
        to grid index ``nzplt``, stepping by ``ndz`` from ``ndz`` on the fluid
        codes (``ramgeo1.5.f:429``, ``ramsurf1.5.f:437``) and from ``1+ndz`` on
        rams0.5 (``:262``). Grid index ``i`` sits at depth ``(i-1)·dz``
        (``ri = 1 + zr/dz``, ``ramgeo1.5.f:126-127``), so reaching
        ``target_depth`` needs the loop to visit an index of at least
        ``ceil(target_depth/dz) + 1``. The extra ``0.75·dz`` above the exact
        ``(nzplt + 0.5)·dz`` threshold keeps the binaries' single-precision
        evaluation of ``zmplt/dz - 0.5`` from truncating one grid point short.
        Clamped at ``zmax``: ``nzplt`` beyond ``nz = zmax/dz - 0.5`` would
        index past the marched arrays.
        """
        dz = float(dz)
        ndz = max(1, int(ndz))
        n = int(np.ceil(max(float(target_depth), 0.0) / dz)) + 1
        # Round up onto an index the output loop actually visits; it starts at
        # ``base + ndz``, so at least one stride is always needed.
        base = 1 if kind == 'rams' else 0
        n = base + max(1, int(np.ceil((n - base) / ndz))) * ndz
        return min((n + 0.75) * dz, float(zmax))

    @staticmethod
    def _collins_deepest_output(zmplt: float, dz: float,
                                ndz: int = 1, kind: str = 'ramgeo') -> float:
        """Deepest depth (m) a Collins binary stores for ``zmplt`` — the
        inverse of :meth:`_collins_zmplt`. ``-1.0`` when the output loop
        visits no index at all."""
        dz = float(dz)
        ndz = max(1, int(ndz))
        nzplt = int(float(zmplt) / dz - 0.5)
        base = 1 if kind == 'rams' else 0
        # Loop indices are base + k·ndz for k >= 1, up to nzplt.
        k = (nzplt - base) // ndz
        return (base + k * ndz - 1) * dz if k >= 1 else -1.0

    @staticmethod
    def _prepend_surface_node(depths, tl, pcomplex):
        """Prepend the ``z = 0`` node to a Collins output grid.

        ``rams0.5`` starts its output loop at grid index ``1+ndz`` and the
        fluid codes at ``ndz`` (``outpt`` in each source), so the shallowest
        stored sample sits at ``ndz·dz`` / ``(ndz-1)·dz`` and a receiver at
        the sea surface falls outside the interpolator's grid. The surface is
        pressure-release in every Collins backend, so the node carries no
        energy — an exact boundary value, not an extrapolation. It is written
        at the ``TL_MAX_DB`` deep-shadow floor rather than a literal zero so
        the row reads like every other no-energy sample uacpy returns.
        """
        depths = np.asarray(depths, dtype=float)
        if depths.size == 0 or depths[0] <= 0.0:
            return depths, tl, pcomplex
        n_r = np.asarray(tl).shape[1]
        floor_mag = 10.0 ** (-TL_MAX_DB / 20.0)
        return (
            np.concatenate([[0.0], depths]),
            np.vstack([np.full((1, n_r), float(TL_MAX_DB)), np.asarray(tl)]),
            np.vstack([np.full((1, n_r), floor_mag, dtype=np.complex128),
                       np.asarray(pcomplex)]),
        )

    def _resolve_collins_grid(self, env, fc, kind, max_range,
                              dr_override, dz_override, zmax_override):
        """Resolve the PE numerics grid ``(dr, dz, zmax)`` for one Collins run.

        Explicit overrides (from the broadband loop, which picks one set for the
        whole band — matching mpiramS) take priority, then user-set ``self.*``,
        then the Lytaev Padé-error optimizer for whatever is still ``None``. The
        rams shear-stability dz floor and the 5× rams ``dr`` safety factor are
        applied to the optimizer's output (see ``_compute_grid_lytaev``).

        ``self.dz`` alone decides whether ``dz`` counts as caller-pinned: the
        broadband override is the band's Lytaev grid whenever the caller left
        ``dz`` unset, and equals ``self.dz`` when they did set it.
        """
        dr = float(dr_override) if dr_override is not None else (
            float(self.dr) if self.dr is not None else None
        )
        dz = float(dz_override) if dz_override is not None else (
            float(self.dz) if self.dz is not None else None
        )
        dz_pinned = self.dz is not None

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
        if (zmax_override is not None or self.zmax is not None):
            self._warn_if_seafloor_outside_grid(zmax, env, dz=dz,
                                                kind=kind, freq=fc)

        if not dz_pinned:
            dz = self._fit_dz_to_mz(kind, dz, zmax)

        # Resolving the sediment block outranks every coarsening above,
        # including the mz budget: a block zread cannot represent is not a
        # coarser answer, it is a different environment. Only the three Collins
        # backends pin block points to grid nodes — mpiramS carries no such
        # arithmetic in any of its sources and interpolates the profile onto the
        # grid with ``interpolators.f90``'s ``interp1``, so it is exempt.
        if (kind in _COLLINS_ARRAY_LIMITS
                and self._block_loses_a_point(env, dz, zmax, kind, fc)):
            block_cap = self._block_dz_cap(env, zmax, kind, fc)
            if dz_pinned:
                raise ConfigurationError(
                    f"RAM(dz={dz:.4f}) cannot represent this sediment block: "
                    f"zread ({kind}) pins block points to nodes 1.5 + z/dz and "
                    f"two of them collide at this dz, so the deeper value "
                    f"overwrites the shallower one and the fill loop replaces "
                    f"the layer with a linear ramp across the whole sub-bottom. "
                    f"The thinnest step is "
                    f"{block_cap * BLOCK_GAP_PER_DZ:.4f} m.",
                    remediation=f"Use dz <= {block_cap:.4f} m, leave dz=None to "
                                f"have it derived, or merge the step into its "
                                f"neighbour if it is not physically meant to be "
                                f"resolved.",
                )
            budget = self._collins_mz_budget(kind, zmax)
            if budget is not None and budget[0](block_cap) > budget[1]:
                raise ConfigurationError(
                    f"RAM:{kind}: resolving this sediment block needs dz <= "
                    f"{block_cap:.4f} m, i.e. {budget[0](block_cap)} depth "
                    f"slots against the binary's mz={budget[1]} over a "
                    f"zmax={zmax:.1f} m domain. A coarser grid would silently "
                    f"replace the block with a linear ramp, so this cannot be "
                    f"met by coarsening.",
                    remediation="Lower zmax, use backend='mpiramS' (no fixed "
                                "depth-array bound), or thicken/merge the "
                                "thinnest sediment step.",
                )
            self._log(
                f"RAM:{kind}: tightened dz from {dz:.4f} m to {block_cap:.4f} m "
                f"so zread resolves the sediment block (thinnest step "
                f"{block_cap * BLOCK_GAP_PER_DZ:.4f} m)."
            )
            dz = block_cap
        return dr, dz, zmax

    def _sediment_blocks(self, env: 'Environment', kind: str, zmax: float,
                         freq: float):
        """Exactly the ``(depth, value)`` blocks the deck will carry.

        Taken from :meth:`_collins_range_segments` rather than rebuilt, because
        three of that method's behaviours change the node arithmetic and a
        hand-rolled copy got all three wrong: the depths are written **relative
        to the seafloor** for ramgeo/ramsurf (:2660) so an absolute-depth copy
        runs the arithmetic in the wrong frame; a pure half-space column is
        wrapped as one **synthetic layer** (:2657) so it has interior block
        points after all; and :meth:`_ramp_absorbing_attenuation` **adds** points
        to the attenuation block. One section per range break, each with its own
        seafloor.
        """
        blocks = []
        for segment in self._collins_range_segments(env, kind, zmax, freq):
            for key in ('bottom_c', 'bottom_rho', 'bottom_attn',
                        'bottom_cs', 'bottom_attns'):
                block = segment.get(key)
                if block:
                    blocks.append([(float(z), float(v)) for z, v in block])
        return blocks

    def _block_loses_a_point(self, env: 'Environment', dz: float,
                             zmax: float, kind: str, freq: float) -> bool:
        """Whether ``zread`` would lose a sediment-block point at this ``dz``.

        This runs the vendored node assignment (``ramsurf1.5.f:200-211``) rather
        than a bound on it, because the natural sufficient bound
        (``gap >= BLOCK_GAP_PER_DZ * dz``) is far from necessary and would reject
        grids that are in fact clean: a 3 m step on ``dz = 2 m`` assigns nodes 1,
        3 and 4, and the node it skips is filled between two *equal* values.

        What is not clean is an **overwrite**. A 0.6 m layer over an 1800 m/s
        basement on the auto grid ``dz = 1.887 m`` puts the half-space value on
        node 1 — the seafloor — and the fill loop at :218-219 then ramps across
        the whole sub-bottom: the layer was marched as a 692 m gradient
        1500 → 1800 m/s, 22.6 dB from Scooter, on the default dispatch for any
        layered fluid bottom.
        """
        if dz <= 0:
            return False
        for block in self._sediment_blocks(env, kind, zmax, freq):
            assigned, previous = {}, None
            for depth, value in block:
                node = int(1.5 + depth / dz)
                if previous is not None and node == previous:
                    node += 1                       # :208 collision push-down
                if node in assigned and assigned[node] != value:
                    return True
                assigned[node] = value
                previous = node
        return False

    def _block_dz_cap(self, env: 'Environment', zmax: float, kind: str,
                      freq: float) -> float:
        """A ``dz`` that ``zread`` is *guaranteed* to represent — the smallest
        positive block gap over :data:`BLOCK_GAP_PER_DZ`. Used only to pick a
        replacement once :meth:`_block_loses_a_point` has said the current grid
        fails, never to judge a grid: see that method for why the bound is
        sufficient but not necessary. ``0.0`` when there is no block to resolve.
        """
        gaps = set()
        for block in self._sediment_blocks(env, kind, zmax, freq):
            depths = [z for z, _ in block]
            gaps |= {b - a for a, b in zip(depths, depths[1:]) if b > a}
        if not gaps:
            return 0.0
        # The smallest gap always clears the collision, but it is often far finer
        # than needed: a gap whose two points carry the *same* value loses nothing
        # when they collide, and the absorbing-attenuation ramp routinely places
        # such a point within a decimetre of the sediment base. Take the coarsest
        # gap-derived candidate the exact predicate accepts — for a 0.9 m layer
        # that is 0.45 m rather than the 0.05 m the ramp's gap would have forced.
        candidates = sorted((g / BLOCK_GAP_PER_DZ for g in gaps), reverse=True)
        for candidate in candidates:
            if not self._block_loses_a_point(env, candidate, zmax, kind, freq):
                return candidate
        return candidates[-1]

    def _fit_dz_to_mz(self, kind: str, dz: float, zmax: float) -> float:
        """Coarsen an auto-picked ``dz`` so the depth grid fits ``mz``.

        Mirrors the ``MAX_DEPTH_POINTS`` clamp in ``_compute_grid_lytaev``: an
        auto grid is uacpy's own choice, so a hard array bound it cannot meet
        is coarsened here rather than raised at the caller. A ``dz`` the caller
        pinned is left alone and rejected by ``_check_collins_array_limits``.
        """
        budget = self._collins_mz_budget(kind, zmax)
        if budget is None or dz <= 0 or zmax <= 0:
            return dz
        needed, mz, dz_min = budget
        if needed(dz) <= mz:
            return dz
        warnings.warn(
            f"RAM:{kind}: raised dz from {dz:.4f} m to {dz_min:.3f} m to fit "
            f"the binary's depth arrays ({needed(dz)} slots needed, mz={mz}) "
            f"over a zmax={zmax:.1f} m domain. Lytaev accuracy budget "
            f"eps={self._accuracy:.0e} is no longer met — lower zmax, or use "
            f"backend='mpiramS' (no fixed limit) to keep the finer grid.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
        return dz_min

    def _warn_if_seafloor_outside_grid(self, zmax: float, env, *,
                                       dz: Optional[float] = None,
                                       kind: Optional[str] = None,
                                       freq: Optional[float] = None) -> None:
        """Warn when a **pinned** ``zmax`` puts the seafloor outside the PE
        domain.

        ``ram.pdf`` p.7 places the grid bottom "well below the ocean bottom
        interface" so the absorbing layer can kill downward energy before it
        reflects. Nothing enforces it: ``ramgeo1.5.f:133-135`` and
        ``ramsurf1.5.f:118-120`` clamp ``iz=min(nz,iz)`` and ``rams0.5.f:135``
        does not clamp at all, so the run simply proceeds with the bottom
        outside its own grid — measured at up to 68.8 dB against a sane
        ``zmax``, with nothing in the output to show for it.

        This warns rather than raises because ``zmax < depth`` is a legitimate
        deck: the vendored ``ramsurf/tests/deep_flat.test/ram.in`` pairs
        ``zb=20000`` with ``zmax=159.9`` deliberately, modelling a column with
        no bottom inside the domain. uacpy cannot tell the two intents apart,
        so it names the consequence and leaves the choice to the caller.

        Only a pinned ``zmax`` reaches here — :meth:`_compute_zmax` already
        clears the seafloor by construction.
        """
        depth = float(env.depth)

        # The binary's condition is on the seafloor INDEX, not the depth:
        # nz = zmax/dz - 0.5 and iz = z/dz (rams0.5.f:132, :135), so the
        # grid holds the seafloor iff floor(depth/dz) <= floor(zmax/dz - 0.5).
        # Testing zmax > depth misses a band up to dz/2 wide where zmax
        # clears the seabed but the index does not — measured, zmax=200.3 m
        # over a 200 m seabed with dz=1 gives iz = nz+1.
        outside = float(zmax) <= depth
        if dz is not None and float(dz) > 0.0:
            iz = int(float(depth) / float(dz))
            nz = int(float(zmax) / float(dz) - 0.5)
            outside = iz > nz

        if not outside:
            # The seafloor is inside the grid, but ram.pdf p.7 asks for more
            # than that: "the bottom of the computational grid is placed WELL
            # BELOW the ocean bottom interface and the attenuation is
            # increased over the lower few wavelengths of the grid". A zmax
            # that merely clears the seabed leaves no absorbing layer, so the
            # bottom of the domain reflects. Measured on ramgeo, 220 m seabed:
            # zmax=222 m is 15.97 dB from an ample grid with no warning at
            # all, while zmax=220 (the old threshold) is 27.0 dB. Drawing the
            # line at env.depth put it where the error had already saturated.
            if freq is not None:
                # "the lower FEW wavelengths" (ram.pdf p.7), not uacpy's own
                # generous 20-lambda auto pad: comparing against
                # _adequate_zmax would warn at zmax=700 m here, which measures
                # 0.32 dB — noise. Three wavelengths tracks the measured error
                # curve instead: silent at 700 m (0.32 dB) and 1500 m, warning
                # at 300 m (3.1 dB) and 222 m (16.0 dB).
                lam = self._resolve_c0(env) / max(float(freq), 1.0)
                adequate = depth + _MIN_SUBBOTTOM_WAVELENGTHS * lam
                if float(zmax) < adequate:
                    warnings.warn(
                        f"RAM: zmax={float(zmax):.4g} m clears the seafloor "
                        f"({depth:.4g} m) but leaves no room for the "
                        f"absorbing layer — ram.pdf p.7 places the grid "
                        f"bottom 'well below' the seabed, which here means "
                        f"about {adequate:.4g} m. The domain floor reflects "
                        f"instead of absorbing: measured 15.97 dB at 2 m of "
                        f"sub-bottom, silently.",
                        UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
                    )
            return

        # rams0.5 does not clamp iz (rams0.5.f:135, against ramgeo1.5.f:135
        # and ramsurf1.5.f:120 which both do min(nz,iz)). Past nz the
        # fluid-solid stencil at rams0.5.f:590 reads lamw(iz+2) — one slot
        # beyond profl's 1..nz+2 (:200-205) — and divides by an unwritten
        # element, so the WHOLE field is NaN while the run still exits 0.
        # The fluid backends degrade to "seafloor at the grid bottom" and
        # return usable numbers, so only rams is fatal.
        if kind == 'rams':
            raise ConfigurationError(
                f"rams: zmax={float(zmax):.4g} m puts the seafloor "
                f"({depth:.4g} m) at grid index iz > nz, and rams0.5 does not "
                f"clamp iz (rams0.5.f:135, unlike ramgeo1.5.f:135 / "
                f"ramsurf1.5.f:120). The fluid-solid stencil at "
                f"rams0.5.f:590 then reads one slot past profl's initialised "
                f"1..nz+2 range and the entire field comes back NaN, from a "
                f"run that still exits 0.",
                remediation=("Raise zmax so floor(depth/dz) <= "
                             "floor(zmax/dz - 0.5) — half a cell is enough — "
                             "or leave zmax unset and let uacpy size the "
                             "domain."),
            )
        warnings.warn(
            f"RAM: zmax={float(zmax):.4g} m is at or above the seafloor "
            f"({depth:.4g} m), so the seabed lies outside the PE grid. The "
            f"binaries do not reject this (ramgeo1.5.f:133-135 clamps "
            f"iz=min(nz,iz); rams0.5.f:135 does not clamp), and the run "
            f"returns plausible numbers with no other sign — measured up to "
            f"27.0 dB from an equivalent run with the seafloor inside the "
            f"grid. Intentional only if you mean 'no bottom in the domain'; "
            f"otherwise put zmax well below the seafloor (ram.pdf p.7).",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

    def _check_source_below_depressed_surface(self, surface, zs: float,
                                              dz: float) -> None:
        """Refuse (or warn about) a source at or above the depressed surface.

        ``matrc`` forces ``r2=1, r1=r3=s1=s2=s3=0`` on rows ``1..izsrf``
        (``ramsurf1.5.f:281-290``) with ``izsrf = 1.0 + zsrf/dz`` — a hard
        Dirichlet zero at and above the surface index. ``selfs`` plants the
        source delta at ``is = ifix(1 + zs/dz)`` and splits it across
        ``u(is)``, ``u(is+1)`` (``:396-400``) **without consulting ``izsrf``**,
        so a source inside the depressed region is zeroed the moment the
        march starts.

        The result is a field that is identically zero — TL ``inf``
        everywhere — with nothing to flag it:
        :meth:`_clamp_collins_envelope` cannot fire because ``|u| = 0`` gives
        a large POSITIVE TL, not a negative or NaN one.

        Two cases, deliberately handled differently:

        * the surface is at or below the source **at the source's own range**
          — the field is dead from the first step, so this is a bad
          configuration and raises;
        * a keel deeper than ``zs`` only further along the track — the field
          dies partway and reads as a plausible shadow zone, which is the more
          dangerous variant precisely because it looks like physics. That
          warns, since the near field before the keel is still meaningful.
        """
        if not surface:
            return

        def zeroed_to(zsrf: float) -> float:
            """Depth of the deepest row matrc actually zeroes for ``zsrf``.

            ``izsrf = 1.0 + zsrf/dz`` (``ramsurf1.5.f:115``) truncates on
            assignment to an integer, and ``matrc`` zeroes rows ``1..izsrf``
            (``:282``). Row ``i`` sits at ``(i-1)*dz``, so the zeroed region
            ends up to one ``dz`` **above** ``zsrf``. Comparing against
            ``zsrf`` itself refuses a source in that band, where the field is
            measurably alive — 84-91 dB with ``dz=0.7``, ``zsrf=30``,
            ``zs=29.7``.
            """
            return (int(1.0 + float(zsrf) / float(dz)) - 1) * float(dz)

        zsrf_at_source = zeroed_to(surface[0][1])
        deepest = max(zeroed_to(z) for _, z in surface)
        if zs <= zsrf_at_source:
            raise ConfigurationError(
                f"ramsurf: source at {zs:.4g} m is at or above the depressed "
                f"surface at r=0 ({zsrf_at_source:.4g} m). matrc zeroes every "
                f"row down to izsrf (ramsurf1.5.f:281-290) while selfs plants "
                f"the source without checking it (:396-400), so the field is "
                f"identically zero. outpt adds eps=1e-20 before the log "
                f"(ramsurf1.5.f:101), so this reports as ~414-437 dB rather "
                f"than inf — there is no NaN or inf to test for.",
                remediation=("Put the source below the deepest surface "
                             "depression, or reduce env.altimetry's depth."),
            )
        if zs <= deepest:
            warnings.warn(
                f"ramsurf: source at {zs:.4g} m is shallower than the deepest "
                f"surface depression ({deepest:.4g} m). Where the keel reaches "
                f"below the source the field is forced to zero "
                f"(ramsurf1.5.f:281-290), which reads as a shadow zone rather "
                f"than as a configuration error. The dead cells report "
                f"~414-437 dB, not inf: outpt adds eps=1e-20 before the log "
                f"(ramsurf1.5.f:101), so no isnan/isinf check will find them.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

    def _build_ramsurf_surface(self, env, max_range):
        """Build the ramsurf1.5 surface profile from ``env.altimetry``.

        Sign convention: env.altimetry is (range, height) with height positive
        UP from sea level (Bellhop / .ati convention). ramsurf1.5 expects
        (range, zsrf) with zsrf >= 0 = depth BELOW z=0 (the pressure-release
        surface drops by zsrf at that range). So negate, then clamp wave crests
        (height > 0 → would imply zsrf < 0) to 0 with a warning — ramsurf only
        models surface depressions / ice keels, not crests above z=0.
        """
        if env.altimetry is None:
            raise ConfigurationError(
                "ramsurf backend requires env.altimetry to be set; "
                "got env.altimetry=None. Use the mpiramS backend "
                "(no altimetry) or supply an altimetry profile."
            )
        zsrf = [(float(r), -float(z)) for r, z in env.altimetry.to_pairs()]
        crests = [(r, h) for r, h in env.altimetry.to_pairs() if float(h) > 0]
        if crests:
            warnings.warn(
                f"ramsurf1.5 only models pressure-release surfaces at or "
                f"below z=0 (zsrf >= 0). {len(crests)} altimetry sample(s) "
                f"with height > 0 (wave crests above mean sea level) "
                f"clamped to z=0. For two-sided wave fields use Bellhop.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP
            )
            zsrf = [(r, max(0.0, z)) for r, z in zsrf]
        surface = zsrf
        if surface[-1][0] < max_range:
            surface.append((float(max_range), surface[-1][1]))
        return surface

    def _run_collins_broadband(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        kind: str
    ) -> Result:
        """Loop a Collins binary over the broadband frequency vector and
        assemble a transfer-function Field.

        The frequency vector matches mpiramS's convention: ``fc`` from
        :meth:`_resolve_broadband_grid`, half-bandwidth ``fc/Q`` (the band
        spans ``2·fc/Q``) and frequency resolution ``df = 1/T``.
        Each frequency runs the Collins binary
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
        bw = fc / Q_used
        df = 1.0 / T_used
        frequencies = self._broadband_frequencies(fc, Q_used, T_used)

        # Pick numerics ONCE for the whole broadband loop. dr is sized at
        # f_min (largest λ → coarsest acceptable step); dz at f_max (smallest
        # λ → finest required step). Two independent Lytaev calls.
        #
        # dr is deliberately sized at f_min, NOT f_max: a finer dr means more
        # range steps, and rams0.5's rotated-Padé elastic march is only
        # marginally stable (|G|≈1 near the evanescent boundary), so per-step
        # floating-point noise compounds. Sizing dr at f_max (~3x more steps
        # here) injects a spurious acausal precursor into the rams0.5 broadband
        # synthesis — verified against the fluid baseline and confirmed by a
        # rotation-angle sweep (Milinazzo 1997). The coarser f_min dr keeps the
        # step count low and was accuracy-validated by the cross-model tests.
        f_min = float(frequencies[0])
        f_max = float(frequencies[-1])
        rmax_band = float(np.max(np.atleast_1d(receiver.ranges)))

        dr_band = float(self.dr) if self.dr is not None else None
        dz_band = float(self.dz) if self.dz is not None else None
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
        if self.dz is None:
            # Coarsen once for the whole band so every frequency marches the
            # same grid and the warning is emitted once, not per frequency.
            dz_band = self._fit_dz_to_mz(kind, dz_band, zmax_band)

        self._log(
            f"{kind} broadband: {len(frequencies)} frequencies, "
            f"{frequencies[0]:.2f}-{frequencies[-1]:.2f} Hz, "
            f"df={df:.2f} Hz, bw={bw:.2f} Hz, "
            f"dr={dr_band:.2f}m, dz={dz_band:.3f}m, zmax={zmax_band:.0f}m"
        )

        rcv_d = np.atleast_1d(receiver.depths).astype(float)
        rcv_r = np.atleast_1d(receiver.ranges).astype(float)

        # Convention: trailing axis is the variable dim (frequency).
        H = np.zeros((rcv_d.size, rcv_r.size, frequencies.size), dtype=complex)

        # One work directory for the whole sweep: each frequency clears the
        # stale binary outputs before it runs, so the band reuses a single
        # directory and the path attached to the result describes it.
        band_fm = self._setup_file_manager()
        try:

            zmax_used = None
            dr_first = None
            dz_first = None
            # Print progress every ~10% of frequencies; on a 500-freq elastic
            # run each iteration takes ~1 s of subprocess overhead, so without
            # this the verbose log goes silent for many minutes.
            log_every = max(1, len(frequencies) // 10)
            for k, freq in enumerate(frequencies):
                if k % log_every == 0 or k == len(frequencies) - 1:
                    self._log(
                        f"{kind} broadband: freq {k + 1}/{len(frequencies)} "
                        f"({float(freq):.2f} Hz)"
                    )
                theta_k = self._theta_for_freq(float(freq))
                raw = self._run_collins_one_freq(
                    env, source, receiver,
                    kind=kind, freq=float(freq), theta=theta_k,
                    dr_override=dr_band, dz_override=dz_band,
                    zmax_override=zmax_band, fm=band_fm,
                )
                zmax_used = raw['zmax']
                if dr_first is None:
                    dr_first = raw['dr']
                if dz_first is None:
                    dz_first = raw['dz']

                # Out-of-grid receivers → NaN so the resulting H(f) cell is
                # NaN (transparent in plots) instead of 0 (which clips TL to
                # TL_MAX_DB and saturates the heatmap edges).
                H[:, :, k] = _interp_envelope_to_receiver_grid(
                    raw['depths'], raw['ranges'],
                    self._clamp_collins_envelope(raw, env, kind),
                    rcv_d, rcv_r,
                    carrier_rate=self._collins_carrier_rate(
                        env, kind, float(freq), theta_k))

            # Convert each backend's raw output to the engineering travelling-
            # wave form. See ``models/_pe_phase.py`` for the per-convention
            # math. H is shaped (n_d, n_r, n_f) here; the Collins binaries
            # already include the 1/√r radial scaling in the file they write,
            # so ``apply_radial=False``.
            c0 = self._resolve_c0(env)
            omega = 2.0 * np.pi * np.asarray(frequencies, dtype=np.float64)
            # ramgeo's UACPY envelope dump (u·f3/√r, carrier factored out) is
            # identical to ramsurf1.5's, so it uses the same phase convention.
            H = psi_to_travelling_wave(
                H,
                convention='ramsurf' if kind == 'ramgeo' else kind,
                ranges_m=rcv_r,
                range_axis=1,
                k0=omega / c0,
                freq_axis=2,
                apply_radial=False,
            )

            # Mask sub-seafloor samples with NaN (same semantics as every backend).
            _mask_below_seafloor(H, rcv_d, rcv_r, env.bathymetry)

            field = Field(
                data=H,
                coords={'depth': rcv_d, 'range': rcv_r, 'frequency': frequencies},
                phase_reference='travelling_wave',
                **self._result_kwargs(
                    source,
                    backend=kind,
                    frequencies=frequencies,
                    Q=Q_used, T=T_used,
                    bandwidth_hz=2.0 * bw, df_hz=df,
                    dr=dr_first, dz=dz_first, zmax=zmax_used,
                    c0=c0,
                    c_min=(self._speed_bounds(env) or (c0, c0))[0],
                    c_max=self._resolve_c_max(env),
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
        finally:
            # Released on the failure path too: a binary that fails partway
            # through the band would otherwise strand the directory.
            if band_fm.cleanup:
                band_fm.cleanup_work_dir()

    def _constrain_dr_to_sections(self, dr, range_segments, *, pinned,
                                  bathymetry_ranges=None,
                                  altimetry_ranges=None):
        """Bound ``dr`` by the closest pair of profile-section markers.

        ``profl`` reads exactly ONE section marker per call
        (``ramgeo1.5.f:195`` ``read(1,*,end=1)rp``, defaulted to ``2.0*rmax`` at
        ``:194`` so an exhausted deck parks the marker past the end of the
        march), ``updat`` re-enters it
        only ``if(r.ge.rp)`` (``:359``), and the march calls ``updat`` once
        per range step (``:78-84``). So the binary can consume at most one
        section per ``dr`` — and because ``profl`` reads *sequentially* from
        the deck, sections written closer together than ``dr`` are not merely
        ignored, they are never reached. The march falls permanently behind
        and every later section is lost.

        Nothing downstream can detect this: the run exits 0 and writes a full
        grid, computed from a truncated environment. Measured on two decks
        differing only in range sampling of the same physics, the finer one
        was wrong by up to 8.75 dB — the deck carried 101 sections and the
        march could reach 19.

        The manual states the rule directly (``ram.pdf`` p.8): "The size of
        the smallest region is an upper bound on Delta-r."
        """
        markers = sorted({float(seg['range']) for seg in range_segments})

        # `updat` advances THREE indices the same way, one step at a time:
        # the profile marker (`if(r.ge.rp)`), the bathymetry index
        # (`ramgeo1.5.f:348` `if(r.ge.rb(ib+1))ib=ib+1`) and, on ramsurf, the
        # altimetry index (`ramsurf1.5.f:346`). Bounding only the first leaves
        # the other two to fall behind, after which the seafloor is linearly
        # extrapolated from a pair of points far astern for the rest of the
        # march — range dependence silently lost. mpiramS is immune to all
        # three: it interpolates (`ram.f90:198`) rather than consuming.
        for stream in (bathymetry_ranges, altimetry_ranges):
            if stream:
                markers = sorted(set(markers) | {float(r) for r in stream})

        if len(markers) < 2:
            return dr
        min_gap = min(b - a for a, b in zip(markers, markers[1:]))
        if min_gap <= 0.0 or dr <= min_gap:
            return dr
        if pinned:
            warnings.warn(
                f"RAM: dr={dr:.4g} m exceeds the closest profile-section "
                f"spacing ({min_gap:.4g} m), so the binary could consume only "
                f"part of the {len(markers)}-section environment and the rest "
                f"would be silently dropped (one section per range step; "
                f"ramgeo1.5.f:194-195, :359, :78-84). dr has been reduced to "
                f"{min_gap:.4g} m. Coarsen the environment's range axis to "
                f"keep the dr you asked for.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
        else:
            self._log(
                f"dr reduced {dr:.4g} -> {min_gap:.4g} m so every one of the "
                f"{len(markers)} profile sections is reachable "
                f"(ram.pdf p.8: the smallest region bounds dr)."
            )
        return min_gap

    def _collins_range_segments(
        self, env: Environment, kind: str, zmax: float, freq: float
    ) -> list:
        """Build the Collins ``range_segments`` list — one ``ram.in`` profile
        section per range break — from the environment's range-dependent SSP
        and (layered) bottom.

        rams0.5 / ramsurf1.5 read a piecewise range-dependent ``ram.in``: an
        initial profile plus one extra ``(range, profile blocks)`` section per
        break. Sections carry the environment at the union of the bottom's and
        the SSP's range breakpoints; a range-independent axis contributes its
        single column at every break.

        A section takes effect from its marker range on — ``if(r.ge.rp)``
        (``ramgeo1.5.f:359``, ``ramsurf1.5.f:364``, ``rams0.5.f:332``) — so the
        marker is written at the midpoint between consecutive breakpoints. That
        makes the switch happen where ``Bottom.at`` / mpiramS put it: mpiramS
        marches with the profile nearest the current range
        (``minloc(abs(rp-rint))``, ``mpiramS/src/ram.f90:218`` for the SSP and
        ``:228`` for the sediment), and ``uacpy.core.bottom.Bottom.at`` is
        documented nearest, so all four backends transition at the same range
        for the same ``Environment``.

        Bottom-profile depth reference differs per binary: ramgeo/ramsurf
        ``matrc`` restarts the profile index at the grid point below the local
        seafloor, so their ``z cb/rhob/attn`` blocks are **depth below the
        seafloor** (layers track the bathymetry — RAMGEO's defining feature);
        rams0.5 indexes absolutely from z=0. Water-SSP blocks are absolute for
        all three.

        Each section's compressional-attenuation block is ramped to
        ``absorbing_layer_attn`` over the deepest ``absorbing_layer_width``
        wavelengths of the PE domain (:meth:`_ramp_absorbing_attenuation`).
        """
        properties = (
            ('sound_speed', 'shear_speed', 'density',
             'attenuation', 'shear_attenuation')
            if kind == 'rams'
            else ('sound_speed', 'density', 'attenuation')
        )
        b = env.bottom
        seafloor_relative = kind in ('ramgeo', 'ramsurf')

        breaks = {0.0}
        if b.is_range_dependent:
            breaks.update(float(r) for r in b.ranges)
        if env.has_range_dependent_ssp:
            breaks.update(float(r) for r in env.ssp.ranges)
        # rams0.5 reads the bottom arrays at absolute depth indices
        # (``rams0.5.f:490-516``, ``do 6 i=iz+1,ib+2`` reading ``lamb(i)``),
        # so a layered elastic column stays where the first section put it
        # and stops following the seafloor. ramgeo/ramsurf re-anchor at the
        # local seafloor in ``matrc`` (``ramgeo1.5.f:262-268``, ``ii=1 …
        # ii=ii+1``) and need no extra sections. A half-space column is
        # immune either way — ``from_halfspace(synthesize=True)`` gives every
        # profile point the same value.
        if (not seafloor_relative and b.is_layered
                and env.bathymetry.is_range_dependent):
            breaks.update(self._bathy_anchor_ranges(env, b))
        ranges = sorted(breaks)

        absorbing_width = (
            self.absorbing_layer_width * self._resolve_c0(env)
            / max(float(freq), 1.0)
        )

        segments = []
        for i, rng in enumerate(ranges):
            # Section 0 is the initial profile; write_ramin ignores its range.
            marker = rng if i == 0 else 0.5 * (ranges[i - 1] + rng)
            seafloor = float(np.asarray(env.bathymetry.eval(range=rng)).flat[0])
            col = b.at(range=rng)
            # The Collins PE update needs a sediment layer above the half-space,
            # so a pure half-space column is wrapped as one synthetic layer.
            if not col.is_layered:
                col = SeabedColumn.from_halfspace(
                    col.halfspace, water_depth=seafloor, synthesize=True)

            z_top = 0.0 if seafloor_relative else seafloor
            z_bottom = (zmax - seafloor) if seafloor_relative else zmax
            bp = col.to_piecewise_breakpoints(
                seafloor_depth=z_top,
                zmax=z_bottom,
                properties=properties,
            )
            ssp_pairs = (
                env.ssp.eval(range=rng).to_pairs()
                if env.has_range_dependent_ssp else env.ssp.to_pairs()
            )
            seg = dict(
                range=float(marker),
                water_ssp=[(float(d), float(c)) for d, c in ssp_pairs],
                bottom_c=bp['sound_speed'],
                bottom_rho=bp['density'],
                bottom_attn=self._ramp_absorbing_attenuation(
                    bp['attenuation'], z_top + col.total_thickness(),
                    z_bottom, absorbing_width),
            )
            if kind == 'rams':
                seg['bottom_cs'] = bp['shear_speed']
                seg['bottom_attns'] = bp['shear_attenuation']
            segments.append(seg)
        return segments

    @staticmethod
    def _bathy_anchor_ranges(env, bottom) -> list:
        """Ranges at which to re-anchor an absolute-depth bottom profile.

        Sections are emitted so the seafloor never moves by more than half the
        thinnest sediment layer between two of them — the bound at which the
        layer would start to slide off its own interval. The bathymetry's own
        control points are not enough: a two-point linear slope has none in
        between, and that is exactly where the layer detaches.
        """
        r_axis = np.atleast_1d(np.asarray(env.bathymetry.ranges, dtype=float))
        r_end = float(np.max(r_axis))
        if not r_end > 0.0:
            return []
        thicknesses = [
            float(layer.thickness)
            for r in r_axis
            for layer in bottom.at(range=float(r)).layers
            if float(layer.thickness) > 0.0
        ]
        if not thicknesses:
            return []
        tol = 0.5 * min(thicknesses)
        probe = np.linspace(0.0, r_end, 1024)
        floor = np.array([float(np.asarray(env.bathymetry.eval(range=float(r))).flat[0])
                          for r in probe])
        out, anchor = [], floor[0]
        for r, z in zip(probe[1:], floor[1:]):
            if abs(z - anchor) >= tol:
                out.append(float(r))
                anchor = z
        if len(out) > MAX_BATHY_SECTIONS:
            idx = np.linspace(0, len(out) - 1, MAX_BATHY_SECTIONS)
            out = [out[int(round(i))] for i in idx]
        return out

    def _ramp_absorbing_attenuation(self, pairs, z_sediment_base, z_bottom,
                                    absorbing_width):
        """Ramp a Collins attenuation block into the artificial absorbing layer.

        ``zread`` (``rams0.5.f:212``, and the same routine in ramgeo1.5.f /
        ramsurf1.5.f) linearly interpolates a ``(depth, value)`` block onto the
        depth grid, so replacing the block's tail with two points
        ``(z_abs, attn_local)`` and ``(z_bottom, absorbing_layer_attn)`` gives
        the linear ramp Collins' own readme prescribes
        (``third_party/ramsurf/readme.orig:127-134``) — without it the flat
        half-space attenuation runs to the domain floor and energy reaching
        it reflects back into the field. This mirrors what mpiramS gets from
        ``attn[-1] = absorbing_layer_attn`` at ``zmax`` over its own linearly
        interpolated sediment profile.

        The ramp starts at ``max(z_sediment_base, z_bottom - absorbing_width)``
        so it never eats into the modelled sediment column. Depths are in
        whichever frame the backend reads (seafloor-relative for
        ramgeo/ramsurf, absolute for rams0.5).
        """
        pairs = [(float(d), float(v)) for d, v in pairs]
        if not pairs or absorbing_width <= 0.0:
            return pairs
        z_abs = max(float(z_sediment_base),
                    float(z_bottom) - float(absorbing_width))
        if not z_abs < float(z_bottom):
            return pairs
        # Keep every control point down to and including z_abs. A strict `<`
        # drops the point that pins the deepest layer's value at its own base,
        # which is exactly the point present when the ramp is clamped to the
        # sediment base. The block carries duplicated abscissae at each layer
        # interface, so the value entering the ramp is the last one *at or
        # above* z_abs — the layer's, not the half-space's that np.interp
        # would return from the right branch.
        head = [p for p in pairs if p[0] <= z_abs]
        attn_local = (head[-1][1] if head
                      else float(np.interp(z_abs, [d for d, _ in pairs],
                                           [v for _, v in pairs])))
        if not head or head[-1][0] < z_abs:
            head = head + [(z_abs, attn_local)]      # pin the ramp's start
        attn_floor = max(attn_local, float(self.absorbing_layer_attn))
        return head + [(float(z_bottom), attn_floor)]

    # Settings that only the mpiramS backend consumes. When the dispatcher
    # picks rams0.5 / ramsurf1.5 and one of these has been overridden from
    # its default, ``_warn_on_mpirams_only_overrides`` warns rather than
    # silently dropping the override. Each entry is (attribute, default).
    # Q and T are honoured by every backend's broadband mode: the Collins
    # path uses them as the Python-side frequency-loop grid, mpiramS uses
    # them inside the Fortran loop. absorbing_layer_width / _attn are honoured
    # too — the first sizes zmax in ``_compute_zmax``, both drive the
    # attenuation ramp in ``_ramp_absorbing_attenuation``.
    _MPIRAMS_ONLY_SETTINGS = (
        ('flat_earth', True),
        ('n_sed_points', 1000)
    )

    # rams0.5's row 5 is ``c0 np irot theta`` (rams0.5.f:109) where the fluid
    # codes read ``c0 np ns rs`` (ramgeo1.5.f:108, ramsurf1.5.f:80), so the
    # two stability pairs are mutually exclusive at the Fortran level and
    # ``write_ramin`` switches the row per kind. Overriding the pair the
    # selected backend does not read discards the value silently.
    _RAMS_ONLY_SETTINGS = (
        ('rams_theta', 45.0),
        ('rams_irot', 1),
    )
    _NOT_RAMS_SETTINGS = (
        ('ns_stability', 1),
        ('rs_stability', None),
    )

    def _drop_unsupported_surface_shear(self, env: Environment) -> Environment:
        """No RAM backend reads surface shear properties; warn and zero them."""
        s = getattr(env, 'surface', None)
        cs = getattr(s, 'shear_speed', None) if s is not None else None
        if cs is None or float(cs) <= 0.0:
            return env
        e = env.copy()
        e.surface = self._collapse_elastic_boundary(
            e.surface, self._collapse["elastic"]
        )
        warnings.warn(
            "RAM: surface shear is not supported by any backend "
            "(mpiramS / rams0.5 / ramsurf1.5 all model the surface as "
            "pressure-release); collapsed surface shear "
            f"(collapse['elastic']={self._collapse['elastic']!r}). "
            "For an elastic surface use Bellhop or Kraken.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP
        )
        return e

    def _warn_on_mpirams_only_overrides(self, backend: str) -> None:
        """Warn about knobs the selected backend cannot read.

        Three disjoint groups: mpiramS-only options the Collins writers
        have no field for, and the two mutually exclusive row-5 stability
        pairs (``ns/rs`` on the fluid codes, ``irot/theta`` on rams0.5).
        """
        groups = []
        if backend != 'mpiramS':
            groups.append(('mpiramS-only', self._MPIRAMS_ONLY_SETTINGS))
        if backend == 'rams':
            groups.append(("mpiramS/ramgeo/ramsurf-only", self._NOT_RAMS_SETTINGS))
        else:
            groups.append(('rams0.5-only', self._RAMS_ONLY_SETTINGS))

        for label, settings in groups:
            nondefault = [
                name for name, default in settings
                if getattr(self, name) != default
            ]
            if nondefault:
                warnings.warn(
                    f"RAM:{backend} ignores these {label} settings "
                    f"(left at their effective default in the binary): "
                    f"{', '.join(nondefault)}. See the RAM constructor "
                    f"docstring for the per-backend applicability.",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP
                )

    def _warn_on_dropped_absorption(self, env: Environment) -> None:
        """No RAM backend consumes water-column volume attenuation; warn
        loudly rather than silently producing a lossless water column."""
        absorption = getattr(env, 'absorption', None)
        if absorption is None:
            return
        from uacpy.core.absorption import ConstantAbsorption
        if (isinstance(absorption, ConstantAbsorption)
                and absorption.value_db_per_wavelength == 0.0):
            return
        warnings.warn(
            f"RAM ignores env.absorption ({type(absorption).__name__}): no "
            f"RAM backend models water-column volume attenuation. Use Bellhop "
            f"or Kraken for volume-attenuation-sensitive runs.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

    # Narrowest source aperture the auto-loosening below will fall back to.
    # Collins (1993) treats 30° as the standard wide-angle PE; under 15° the
    # operator is essentially paraxial and "wide-angle" stops meaning
    # anything, so a caller who genuinely wants narrow-angle physics has to
    # say so via ``theta_max=`` rather than reach it by relaxation.
    _THETA_MAX_FLOOR = 15.0

    def _optimize_grid_relaxing(self, *, freq, c_min, c_max, max_range,
                                c0_pe, eps0, theta0, kind):
        """Run the Lytaev optimizer, loosening its inputs until one converges.

        A hard environment (deep ocean, high ``c_max``, wide ``θ_max``) admits
        no grid meeting ``eps0`` under the Collins second-order Numerov, and
        :func:`~uacpy.models._pade_optimizer.optimize_grid` signals that with a
        ``RuntimeError``. Two nested fallbacks, tried in this order because
        a looser error budget still describes the requested physics whereas a
        narrower aperture no longer does:

        1. ``ε`` is tripled, up to 8 times or until it passes 0.5;
        2. only if the whole ε ladder failed, ``θ_max`` steps down through
           20° and :data:`_THETA_MAX_FLOOR`, restarting the ε ladder each
           time. Steps at or above ``theta0`` are skipped, so a caller who
           already asked for a narrow aperture never has it widened.

        Returns ``(result, eps_used, theta_used)`` — the optimizer payload
        plus the inputs that produced it, so the caller can warn when they
        differ from what was asked for. Raises :class:`ConfigurationError`
        when even the loosest combination is infeasible.
        """
        from uacpy.models._pade_optimizer import optimize_grid

        eps_used, theta_used, res, last_exc = eps0, theta0, None, None
        for theta_trial in (theta0, 20.0, self._THETA_MAX_FLOOR):
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
                f"θ_max={self._THETA_MAX_FLOOR:.0f}° for f={freq:.1f} Hz, "
                f"x_max={max_range:.0f} m. Set ``dr``/``dz`` explicitly. "
                f"Optimiser said: {last_exc}"
            ) from last_exc
        return res, eps_used, theta_used

    def _compute_grid_lytaev(
        self, env: 'Environment', freq: float,
        *, max_range: float, kind: str
    ) -> 'tuple[float, float]':
        """Padé-error-based ``(dr, dz)`` selection following Lytaev
        (2023, https://doi.org/10.3390/jmse11030496).

        Picks the coarsest ``(dr, dz)`` whose accumulated single-step
        Padé error stays under ``accuracy`` over the marched range.
        The PE reference speed ``c₀`` comes from ``_resolve_c0`` (Lytaev
        Eq. (15) by default, the user's value when pinned).

        The optimizer minimises Lytaev's error model alone. Four
        constraints it does not represent are applied to its output
        afterwards: the rams ``dr`` stability tightening, seafloor-node
        snapping, the ``MAX_DEPTH_POINTS`` runtime cap, and the
        shear/acoustic ``dz`` floor. The accuracy that gets logged is
        therefore recomputed on the grid that is actually marched, which
        can be orders of magnitude above ``accuracy`` once a floor
        has bound.

        Raises ``ConfigurationError`` if no candidate ``(dr, dz)`` pair
        meets the accuracy budget even after auto-loosening.
        """
        from uacpy.models._pade_optimizer import grid_error, rams_dz_shear_cap

        c0_pe = self._resolve_c0(env)

        # Spectrum bounds: slowest / fastest acoustic speeds in the env,
        # widened to contain c₀ so [ξ_min, ξ_max] brackets the expansion
        # point even when the caller pinned an out-of-range c0.
        bounds = self._speed_bounds(env) or (c0_pe, c0_pe)
        c_min = min(bounds[0], c0_pe)
        c_max = max(bounds[1], c0_pe)

        # Per-backend dz floor: λ_p/16 for the Collins backends, a cost bound
        # so the optimizer cannot demand an absurdly fine depth grid.
        # Override via ``dr=…``/``dz=…``.
        if kind in ('mpiramS', 'rams', 'ramsurf', 'ramgeo'):
            dz_floor = c_min / (LAMBDA_PER_DZ_FLOOR * max(freq, 1.0))
            cs_min = self._min_shear_speed(env) if kind == 'rams' else 0.0
        else:
            cs_min = 0.0
            dz_floor = 0.0
        # The shear wavelength is the binding physical scale for the elastic
        # march, and resolving it is a correctness requirement rather than a
        # cost preference — so it caps dz, and it also bounds how far the cost
        # floor above may coarsen it.
        dz_shear_cap = rams_dz_shear_cap(cs_min, freq) if kind == 'rams' else 0.0
        if dz_shear_cap > 0:
            dz_floor = min(dz_floor, dz_shear_cap)

        eps0 = self._accuracy
        theta0 = self._resolve_theta_max(env)
        res, eps_used, theta_used = self._optimize_grid_relaxing(
            freq=freq, c_min=c_min, c_max=c_max, max_range=max_range,
            c0_pe=c0_pe, eps0=eps0, theta0=theta0, kind=kind,
        )
        if eps_used > eps0 or theta_used < theta0:
            warnings.warn(
                f"RAM:{kind}: Lytaev relaxed ε={eps0:.0e}→{eps_used:.0e}, "
                f"θ_max={theta0:.0f}°→{theta_used:.0f}° to find a feasible "
                f"grid at f={freq:.1f} Hz, x_max={max_range:.0f} m. "
                f"Expect TL errors larger than your original target.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP
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
        if bathy is not None and bathy.n_ranges > 0:
            h = float(np.min(bathy.depths))
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
                f"ε={self._accuracy:.0e} is no longer met. Reduce "
                f"``theta_max`` or set dr/dz explicitly to override.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP
            )

        dz_pre_floor = dz_opt
        if dz_floor > 0 and dz_opt < dz_floor:
            if h > 0:
                n_layers = max(1, int(np.floor(h / dz_floor)))
                dz_opt = float(h / n_layers)
            else:
                dz_opt = dz_floor

        # Resolving the shear wavelength outranks every coarsening above: a
        # rams0.5 march on a grid coarser than λ_s/14 does not merely lose
        # accuracy, it diverges (measured 134 dB against OASES on Collins
        # 1991's own example D at 0.55 λ_s, versus 0.83 dB at λ_s/14).
        if dz_shear_cap > 0 and dz_opt > dz_shear_cap:
            dz_pre_cap = dz_opt
            dz_opt = dz_shear_cap
            self._log(
                f"rams: tightened dz from {dz_pre_cap:.3f} m to "
                f"{dz_opt:.3f} m to resolve the shear wavelength "
                f"(λ_s/14 = {dz_shear_cap:.3f} m, c_s = {cs_min:.0f} m/s)."
            )
            # This cap outranks the MAX_DEPTH_POINTS budget applied above, since
            # a coarser grid diverges rather than merely costing accuracy. Say so
            # when it bites, so a slow run has a stated cause.
            if h > 0 and h / dz_opt > MAX_DEPTH_POINTS:
                warnings.warn(
                    f"RAM:{kind}: resolving the shear wavelength needs dz="
                    f"{dz_opt:.4f} m, i.e. {h / dz_opt:.0f} depth points — past "
                    f"the {MAX_DEPTH_POINTS}-point runtime budget. A coarser "
                    f"grid makes the elastic march diverge, so accuracy wins "
                    f"here; expect a slow run.",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP
                )

        # The optimizer knows nothing about the adjustments above, so its
        # own ``predicted_error`` describes a grid that may never be
        # marched. Recompute on the grid being returned.
        err = grid_error(
            dr=dr_opt, dz=dz_opt, freq=float(freq),
            c_min=c_min, c_max=c_max, x_max=float(max_range), c0=c0_pe,
            theta_max=float(theta_used), p=int(self.np_pade), alpha=0.0,
        )

        if dz_opt > dz_pre_floor:
            if cs_min > 0:
                reason = 'shear-mode stability (λ_s × 0.55)'
            elif kind == 'mpiramS':
                reason = 'mpiramS runtime cap (λ_p / 16)'
            else:
                reason = 'acoustic stability (λ_p / 16)'
            msg = (
                f"RAM:{kind}: raised dz from {dz_pre_floor:.3f} m to "
                f"{dz_opt:.3f} m for {reason} "
                f"(floor={dz_floor:.3f} m at cs_min={cs_min:.0f} m/s, "
                f"f={freq:.0f} Hz). The Lytaev accuracy budget "
                f"ε={self._accuracy:.0e} is not met on this grid — its "
                f"predicted error is {err:.2e}. Set dr/dz explicitly "
                f"to override. For broadband sweeps the cap is computed "
                f"at the *lowest* frequency in the sweep, so it may be "
                f"sub-Nyquist for the upper band — pin dz≈λ(f_max)/8 "
                f"to resolve the full pulse spectrum."
            )
            # The stability floor sits above the default Lytaev dz for every
            # ordinary frequency, so warning on the default target fires on
            # essentially every run and trains callers to ignore uacpy
            # warnings. Warn only when the caller pinned an accuracy that is
            # then not delivered; otherwise report it as status.
            if self._accuracy_explicit:
                warnings.warn(msg, UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
            else:
                self._log(msg, level="info")

        c0_origin = 'user' if self.c0 is not None else 'Lytaev Eq.15'
        self._log(
            f"{kind}: Lytaev grid → dr={dr_opt:.2f} m, "
            f"dz={dz_opt:.3f} m (predicted error "
            f"{err:.2e}, c₀={c0_pe:.1f} m/s "
            f"[{c0_origin}], θ_max={theta_used:.0f}°, "
            f"ε={self._accuracy:.0e})."
        )
        return dr_opt, dz_opt

    def _effective_dz(self) -> float:
        """Resolve `dz` outside the per-frequency hot paths (sediment layer
        thickness, layered-bottom padding, metadata). Honours an explicit
        `self.dz` if set; no frequency is in scope at these call sites, so
        the fallback is a fixed 0.5 m.
        """
        if self.dz is not None:
            return float(self.dz)
        return 0.5

    def _compute_dz(self, env: 'Environment', freq: float,
                    c0: Optional[float] = None) -> float:
        """Quick ``λ_min/16`` depth-step estimate, clipped to [0.05, 1.0] m.

        Used only for auxiliary sizing (absorbing-layer thickness in
        ``_compute_zmax``). The main PE grid is always picked by the
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

        def _add(cp):
            # Smallest positive speed drives the finest λ/16 dz requirement.
            if cp and cp > 0:
                speeds.append(float(cp))

        for cp in env.bottom.all_sound_speeds():
            _add(cp)
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
        if bathy is not None and bathy.n_ranges > 0:
            h = float(np.min(bathy.depths))
        else:
            h = float(getattr(env, 'depth', None) or 0.0)
        if h > 0:
            n_layers = max(1, int(round(h / target)))
            return float(h / n_layers)
        return target

    def _resolve_mpirams_grid(self, env, freq: float, rmax: float):
        """``(dr, dz)`` for an mpiramS march: user values where pinned, the
        Lytaev Padé-error optimizer for whatever is still ``None``."""
        dr = float(self.dr) if self.dr is not None else None
        dz = float(self.dz) if self.dz is not None else None
        if dr is None or dz is None:
            dr_auto, dz_auto = self._compute_grid_lytaev(
                env, freq, max_range=rmax, kind='mpiramS'
            )
            dr = dr_auto if dr is None else dr
            dz = dz_auto if dz is None else dz
        return dr, dz

    def _write_mpirams_deck(self, env, source, receiver, work_dir,
                            freq: float, Q: float, T: float,
                            dr: float, dz: float) -> None:
        """Write every input file one mpiramS march reads: ``ssp.dat``,
        the bathymetry and sediment files, ``ranges.dat`` and ``in.pe``.

        Shared by the narrowband and broadband paths, which differ only in
        ``(freq, Q, T)`` — keeping one writer is what stops the two decks
        from drifting apart.
        """
        rmax = float(np.max(receiver.ranges))
        ssp_filename = self._prepare_ssp(env, work_dir, freq, dz)
        bth_filename, ibot = self._prepare_bathymetry(env, rmax, work_dir)
        zmax_pe = self._mpirams_zmax(env, freq, dz)
        sedlayer, nzs, cs, rho_arr, attn_arr, isedrd, sed_filename = \
            self._prepare_bottom_properties(
                env, work_dir, self._absorber_span(env, freq, zmax_pe),
                zmax_pe)

        write_ranges_file(work_dir / 'ranges.dat', receiver.ranges)

        rs = self.rs_stability if self.rs_stability is not None else rmax

        # mpiramS's horizontal-interpolation branch (ihorz=1) resamples the
        # SSP onto a uniform grid of nrp=nint(rmax/10000) points
        # (peramx.f90:245). For rmax < 5 km that rounds to nrp=0 -> a
        # zero-length allocate, an all-NaN sound-speed field, IEEE
        # divide-by-zero and a SIGABRT (exit -6); for rmax < 15 km it
        # collapses range dependence to 1-2 coarse 10-km samples. Always use
        # ihorz=0 so mpiramS steps directly between the per-range profiles
        # uacpy already builds in _prepare_ssp (at env.ssp.ranges) — both
        # crash-free and more faithful than the buggy 10-km resample.
        # mpiramS plants the source identically (ram.f90:110-114) and its
        # solver also starts at row 2 (solvetri.f90:47), so the row-1 kill is
        # not a Collins peculiarity.
        self._check_source_row_is_solved(float(source.depths[0]), dz)

        write_inpe(
            filepath=work_dir / 'in.pe',
            fc=freq,
            Q=Q,
            T=T,
            zsrc=float(source.depths[0]),
            deltaz=dz,
            deltar=dr,
            np_pade=self.np_pade,
            nss=self.ns_stability,
            rs=rs,
            dzm=self.depth_decimation,
            ssp_filename=ssp_filename,
            iflat=1 if self.flat_earth else 0,
            ihorz=0,
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

    def _run_tl(self, env, source, receiver):
        """
        Run in narrowband TL mode at ``source.frequencies[0]``.

        Honours ``self.Q`` / ``self.T``. For a single-frequency TL grid the
        conventional choice is a very large Q so the bandwidth collapses,
        but the user is free to widen it.
        """
        start_time = time.time()

        freq = float(source.frequencies[0])
        zsrc = float(source.depths[0])
        ranges = receiver.ranges
        rmax = float(np.max(ranges))

        dr, dz = self._resolve_mpirams_grid(env, freq, rmax)

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
            self._write_mpirams_deck(
                env, source, receiver, work_dir, freq, Q_tl, T_tl, dr, dz)

            self._run_binary(work_dir)
            result = read_psif(work_dir)

            return self._assemble_tl_field(
                result, env, source, receiver, fm, freq, dr, dz, start_time)

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _assemble_tl_field(self, result, env, source, receiver, fm,
                           freq, dr, dz, start_time):
        """Assemble the TL :class:`Field` from a finished mpiramS ``psif``
        result: pick the centre-frequency bin, interpolate complex pressure to
        the receiver grid (NaN-safe), convert to travelling-wave pressure, mask
        below the seafloor, and tag."""
        psif = result['psif']  # (nzo, nf, nr)
        zg = result['zg']
        rout = result['rout']

        # Center-frequency bin: nearest to fc. mpiramS sweeps a band
        # symmetric about fc, so odd nf has an exact middle; for an even
        # custom (Q,T) band this picks the closest bin (nf//2 would bias to
        # fc+Δf/2).
        center_idx = int(np.argmin(np.abs(result['frq'] - freq)))
        # pressure at center freq for all depths and ranges: (nzo, nr)
        pressure = psif[:, center_idx, :]

        # Interpolate COMPLEX PRESSURE from PE grid to receiver grid
        # BEFORE computing TL. Interpolating in dB destroys interference
        # nulls because linear interpolation of log-scale values smooths
        # out the sharp zeros in the field. Valid here — mpiramS writes its
        # output on the receiver range grid (write_ranges_file), so psif is
        # not undersampled in range. The Collins backends need the
        # modulus/phasor split instead (_interp_envelope_to_receiver_grid).

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
                skip_file_prefixes=USER_FRAME_SKIP
            )
        rcv_ranges = np.clip(receiver.ranges, rout[0], rout[-1])

        # Interpolate real and imaginary parts separately. NaN samples
        # in the centre-frequency slice (PE divergence, or a depth the
        # march did not resolve) are zeroed before interpolation; warn if
        # any are present so the user knows the field is not fully
        # converged.
        n_nan_p = int(np.count_nonzero(~np.isfinite(pressure)))
        if n_nan_p > 0:
            # expected; not in filterwarnings — emerges to user
            warnings.warn(
                f"RAM:mpiramS: {n_nan_p}/{pressure.size} "
                f"complex samples are NaN/inf and have been zeroed "
                f"for interpolation. Inspect the result before use.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP
            )
        # Receivers outside the PE output domain return NaN pressure
        # so the resulting TL row is NaN (transparent in pcolormesh)
        # instead of saturating to ``TL_MAX_DB`` via PRESSURE_FLOOR.
        pressure_rcv = _interp_to_receiver_grid(
            zg, rout, pressure, rcv_depths, rcv_ranges, sanitize=True)

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
                skip_file_prefixes=USER_FRAME_SKIP
            )
            log_ranges[log_ranges <= 0.0] = dr

        # Convert the mpiramS .psif output to engineering travelling-
        # wave pressure (see ``models/_pe_phase.py``). ``Field.db``
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
                c0=self._resolve_c0(env),
                c_max=self._resolve_c_max(env),
            )
        )
        field = field.mask_below_seafloor(env.bathymetry)
        self._attach_output_paths(
            field, fm.work_dir, '',
            primary_files=(('psif_file', 'psif.dat'),)
        )
        return field

    def _run_broadband(self, env, source, receiver):
        """
        Run in broadband mode (native mpiramS use case).

        Returns complex transfer function psi(depth, frequency, range).
        """
        start_time = time.time()

        freq, Q_bb, T_bb = self._resolve_broadband_grid(source)
        ranges = receiver.ranges
        rmax = float(np.max(ranges))

        dr, dz = self._resolve_mpirams_grid(env, freq, rmax)
        self._log(
            f"mpiramS (broadband): fc={freq:.1f} Hz, Q={Q_bb}, T={T_bb}s, "
            f"dr={dr:.1f} m, dz={dz:.3f} m"
        )
        self._log(f"Bandwidth: {2.0 * freq / Q_bb:.2f} Hz "
                  f"(fc ± {freq / Q_bb:.2f} Hz)")

        fm = self._setup_file_manager()
        work_dir = fm.work_dir

        try:
            self._write_mpirams_deck(
                env, source, receiver, work_dir, freq, Q_bb, T_bb, dr, dz)

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
                    f"`ranges` reflect the clipped value so coordinates match "
                    f"the scaled field.",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP
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
                # zg is monotone but NOT uniform: with flat_earth=1 peramx
                # un-transforms it by zg/(1 + eps/2 + eps²/3), eps = zg/Re
                # (peramx.f90:427-432), a quadratic map that stretches by
                # metres over a deep column. Bracket against the real axis.
                idx_lo = np.clip(
                    np.searchsorted(zg, out_depths, side='right') - 1,
                    0, len(zg) - 2)
                span = zg[idx_lo + 1] - zg[idx_lo]
                with np.errstate(divide='ignore', invalid='ignore'):
                    w = np.where(span > 0.0,
                                 (out_depths - zg[idx_lo]) / span, 0.0)
                w = np.clip(w, 0.0, 1.0)
                # Vectorized interpolation: (n_out, nf, nr)
                pressure = (pressure[idx_lo, :, :] * (1.0 - w[:, None, None]) +
                            pressure[idx_lo + 1, :, :] * w[:, None, None])
                # Depths outside the PE grid are NaN, matching the
                # COHERENT_TL below-domain convention — never a
                # plausible-looking edge extrapolation.
                outside = (out_depths < zg[0]) | (out_depths > zg[-1])
                pressure[outside, :, :] = np.nan
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
                    'range': rout_safe,
                    'frequency': result['frq'],
                },
                phase_reference='travelling_wave',
                **self._result_kwargs(
                    source,
                    backend='mpiramS',
                    frequencies=result['frq'],
                    dr=float(dr), dz=float(dz),
                    n_samples=result['n_samples'],
                    fs=result['fs'],
                    Q=result['Q'],
                    c0=result['c0'],
                    c_min=result['c_min'],
                    c_max=self._resolve_c_max(env),
                )
            )
            # Mask sub-seafloor samples with NaN (same semantics as every
            # backend), against the ranges the Field advertises.
            _mask_below_seafloor(tf.data, out_depths, rout_safe, env.bathymetry)
            self._attach_output_paths(
                tf, fm.work_dir, '',
                primary_files=(('psif_file', 'psif.dat'),)
            )
            return tf

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    @staticmethod
    def _clear_stale_outputs(work_dir: Path, names) -> None:
        """Delete ``names`` under ``work_dir`` before launching a binary."""
        for name in names:
            Path(work_dir).joinpath(name).unlink(missing_ok=True)

    def _run_binary(self, work_dir: Path):
        """Execute s_mpiram in the given working directory."""
        env = os.environ.copy()
        # Avoid oversubscription: under ``run_parallel`` (a process pool) every
        # worker would otherwise spawn ``cpu_count()`` OpenMP threads → N×N
        # threads. Respect an explicit ``OMP_NUM_THREADS`` if the user set one;
        # otherwise pin to a single thread and let parallelism come from the
        # process pool. Opt into intra-run threading by exporting the var.
        if 'OMP_NUM_THREADS' in env:
            omp_source = 'inherited'
        else:
            env['OMP_NUM_THREADS'] = '1'
            omp_source = 'default = 1 (set OMP_NUM_THREADS to opt in)'
        self._log(
            f"Executing mpiramS: {self._exe} "
            f"(cwd={work_dir}, OMP_NUM_THREADS={env['OMP_NUM_THREADS']} "
            f"{omp_source})"
        )

        self._clear_stale_outputs(work_dir, _MPIRAMS_OUTPUTS)
        result = self._run_subprocess(
            [str(self._exe)],
            cwd=work_dir,
            timeout=self.timeout,
            env=env
        )

        if result.stdout:
            self._log(f"mpiramS output:\n{result.stdout}", level='debug')

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
