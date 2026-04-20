"""
RAM - Range-dependent Acoustic Model wrapper

Uses mpiramS (Fortran 90/95 PE binary) as the backend.
mpiramS is a broadband parabolic equation model for deep-water,
long-range, low-frequency acoustic propagation. Based on Mike Collins'
RAM, adapted to Fortran 95 by B. Dushaw.

Supports two run modes:
- COHERENT_TL: Narrowband transmission loss over range-depth grid
- TIME_SERIES: Broadband complex field at output range(s) for IFFT
"""

import numpy as np
import subprocess
import os
from pathlib import Path
from typing import Optional
import scipy.interpolate

from uacpy.models.base import PropagationModel, RunMode, _UNSET, _resolve_overrides
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.field import Field
from uacpy.core.constants import DEFAULT_SOUND_SPEED, TL_FLOOR_PRESSURE, TL_MAX_DB
from uacpy.io.mpirams_writer import write_inpe, write_ssp_file, write_bth_file, write_ranges_file
from uacpy.io.mpirams_reader import read_psif


class RAM(PropagationModel):
    """
    RAM - Range-dependent Acoustic Model (Parabolic Equation)

    Wrapper for mpiramS, a Fortran 90/95 broadband PE model for
    deep-water, long-range, low-frequency acoustic propagation.

    Supports range-dependent sound speed profiles and bathymetry.
    Bottom properties are configurable (sediment speed, density, attenuation).

    Run modes
    ---------
    COHERENT_TL:
        Narrowband TL over a range-depth grid. Uses large Q internally
        so mpiramS computes a single frequency. Returns Field(field_type='tl').

    TIME_SERIES:
        Broadband complex pressure field. Returns Field(field_type='transfer_function')
        with psi(depth, frequency, range) for downstream IFFT to time domain.

    Parameters
    ----------
    executable : Path, optional
        Path to s_mpiram binary. Auto-detected if None.
    dr : float, optional
        Range step in meters. Default: None (auto-select based on
        frequency: dr = c0/freq, i.e. one wavelength, capped at 500m).
    dz : float, optional
        Depth step in meters. Default: 0.5
    np_pade : int, optional
        Number of Pade coefficients (2-8). Default: 6
    ns_stability : int, optional
        Number of stability terms. Default: 1 (use 0 for short ranges)
    rs_stability : float, optional
        Stability range in meters. Default: max output range.
    Q : float, optional
        Q value for broadband mode (bandwidth = fc/Q). Default: 2.0.
        Ignored in COHERENT_TL mode (set internally to large value).
    T : float, optional
        Time window width in seconds. Default: 10.0.
        Ignored in COHERENT_TL mode (set internally).
    depth_decimation : int, optional
        Output depth decimation factor. Default: 1 (no decimation).
    flat_earth : bool, optional
        Apply flat-earth transformation. Default: True.
    use_tmpfs : bool, optional
        Use RAM-based filesystem for I/O. Default: False.
    verbose : bool, optional
        Print detailed output. Default: False.
    work_dir : Path, optional
        Working directory. If None, creates temporary.
    """

    def __init__(
        self,
        executable: Optional[Path] = None,
        dr: Optional[float] = None,
        dz: float = 0.5,
        zmax: Optional[float] = None,
        np_pade: int = 6,
        ns_stability: int = 1,
        rs_stability: Optional[float] = None,
        Q: float = 2.0,
        T: float = 10.0,
        depth_decimation: int = 1,
        flat_earth: bool = True,
        absorbing_layer_width: float = 20.0,
        absorbing_layer_attn: float = 10.0,
        n_sed_points: int = 50,
        use_tmpfs: bool = False,
        verbose: bool = False,
        work_dir: Optional[Path] = None,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to s_mpiram binary. Auto-detected if None.
        dr : float, optional
            Range step (m). None = auto (one wavelength, capped at 500m). Default: None.
        dz : float, optional
            Depth step (m). Default: 0.5.
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
            Q value for broadband bandwidth (fc/Q). Default: 2.0.
        T : float, optional
            Time window width (s). Default: 10.0.
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
        """
        super().__init__(use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir)

        self._supported_modes = [RunMode.COHERENT_TL, RunMode.TIME_SERIES]

        if executable is not None:
            self.executable = Path(executable)
        else:
            self.executable = self._find_executable('s_mpiram')

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

    def _find_executable(self, name: str) -> Path:
        """Find mpiramS executable."""
        return self._find_executable_in_paths(name, 'mpirams')

    def _compute_zmax(self, env: Environment, freq: float, c0: float = DEFAULT_SOUND_SPEED) -> float:
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
        # Maximum seafloor depth across all ranges
        if hasattr(env, 'bathymetry') and env.bathymetry is not None and len(env.bathymetry) > 0:
            max_depth = float(np.max(env.bathymetry[:, 1]))
        else:
            max_depth = env.depth
        wavelength = c0 / max(freq, 1.0)
        absorbing_width = self.absorbing_layer_width * wavelength
        zmax = max_depth + self.dz + absorbing_width
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
        ssp_data = env.ssp_data
        depths_orig = ssp_data[:, 0].copy()
        speeds_orig = ssp_data[:, 1] if ssp_data.ndim == 2 else ssp_data[:, 1].copy()

        zmax_pe = self._compute_zmax(env, freq)

        # Build depth grid from surface to zmax_pe
        interp_func = scipy.interpolate.interp1d(
            depths_orig, speeds_orig,
            kind='linear', bounds_error=False,
            fill_value=(speeds_orig[0], speeds_orig[-1])
        )
        n_points = max(50, int(zmax_pe / self.dz / 2))
        depths = np.linspace(0, zmax_pe, n_points)
        base_speeds = interp_func(depths)

        ssp_filename = 'ssp.dat'

        if env.has_range_dependent_ssp():
            if hasattr(env, 'ssp_2d_matrix') and env.ssp_2d_matrix is not None:
                self._log("Using 2D SSP matrix", level='info')
                ranges_km = env.ssp_2d_ranges.copy()
                ssp_depths = env.ssp_data[:, 0]

                speeds_2d = np.zeros((len(depths), len(ranges_km)))
                for i in range(len(ranges_km)):
                    profile = env.ssp_2d_matrix[:, i]
                    interp_func = scipy.interpolate.interp1d(
                        ssp_depths, profile, kind='linear',
                        bounds_error=False,
                        fill_value=(profile[0], profile[-1])
                    )
                    speeds_2d[:, i] = interp_func(depths)

                write_ssp_file(work_dir / ssp_filename, depths, speeds_2d, ranges_km)

            elif hasattr(env, 'ssp_profiles') and env.ssp_profiles is not None:
                self._log(f"Using {len(env.ssp_ranges)} SSP profiles", level='info')
                ranges_km = np.array(env.ssp_ranges)
                speeds_2d = np.zeros((len(depths), len(ranges_km)))
                for i, profile in enumerate(env.ssp_profiles):
                    interp_func = scipy.interpolate.interp1d(
                        profile[:, 0], profile[:, 1], kind='linear',
                        bounds_error=False,
                        fill_value=(profile[0, 1], profile[-1, 1])
                    )
                    speeds_2d[:, i] = interp_func(depths)
                write_ssp_file(work_dir / ssp_filename, depths, speeds_2d, ranges_km)
            else:
                write_ssp_file(work_dir / ssp_filename, depths, base_speeds)
        else:
            write_ssp_file(work_dir / ssp_filename, depths, base_speeds)

        return ssp_filename

    def _prepare_bathymetry(self, env: Environment, rmax: float, work_dir: Path) -> tuple:
        """
        Write bathymetry file. Returns (bth_filename, ibot).
        """
        bth_filename = 'bathy.dat'

        if hasattr(env, 'bathymetry') and env.bathymetry is not None and len(env.bathymetry) > 0:
            bathy = env.bathymetry.copy()
            # Extend to rmax if needed
            if bathy[-1, 0] < rmax:
                bathy = np.vstack([bathy, [rmax, bathy[-1, 1]]])
            write_bth_file(work_dir / bth_filename, bathy[:, 0], bathy[:, 1])
            return bth_filename, 1
        else:
            # No bathymetry file — use default flat bottom
            depth = env.depth if env.depth > 0 else 100.0
            write_bth_file(work_dir / bth_filename, np.array([0.0, rmax]), np.array([depth, depth]))
            return bth_filename, 1

    def _get_water_speed_at_bottom(self, env: Environment) -> float:
        """
        Get the water column sound speed at the nominal seafloor depth.

        This is what mpiramS's ``cwg`` will be at the water-sediment
        interface.  Used to convert an absolute bottom sound speed into the
        perturbation that mpiramS expects (``cs = cb - cwg``).
        """
        ssp = env.ssp_data  # (depth, speed)
        depth = env.depth
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
        sedlayer = self.dz

        if hasattr(env, 'bottom_rd_layered') and env.bottom_rd_layered is not None and work_dir is not None:
            # Range-dependent layered bottom
            rdl = env.bottom_rd_layered
            n_ranges = len(rdl.ranges_km)
            sedlayer_rdl = max(rdl.max_total_thickness(), self.dz)

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

                if env.has_range_dependent_ssp():
                    ssp_at_range = env.get_ssp_at_range(rdl.ranges_km[i] * 1000.0)
                    cwg_local = float(np.interp(rdl.depths[i],
                                                ssp_at_range[:, 0], ssp_at_range[:, 1]))
                else:
                    cwg_local = float(np.interp(rdl.depths[i],
                                                env.ssp_data[:, 0], env.ssp_data[:, 1]))

                # Points 0,1 = water (zero perturbation), rest = sediment
                cs_profiles[0, i] = 0.0
                cs_profiles[1, i] = 0.0
                cs_profiles[2:, i] = cs_samp[2:] - cwg_local

                rho_profiles[:, i] = rho_samp
                attn_profiles[:, i] = attn_samp
                attn_profiles[-1, i] = self.absorbing_layer_attn

            from uacpy.io.mpirams_writer import write_sediment_file
            sed_filename = 'sediment.sed'
            write_sediment_file(
                work_dir / sed_filename,
                rdl.ranges_km,
                cs_profiles, rho_profiles, attn_profiles
            )

            self._log(f"Range-dependent layered sediment: {n_ranges} profiles, "
                      f"nzs={nzs}, sedlayer={sedlayer_rdl:.1f} m", level='info')

            cs = cs_profiles[:, 0].copy()
            rho_arr = rho_profiles[:, 0].copy()
            attn_arr = attn_profiles[:, 0].copy()
            return sedlayer_rdl, nzs, cs, rho_arr, attn_arr, 1, sed_filename

        if hasattr(env, 'bottom_layered') and env.bottom_layered is not None:
            # Range-independent layered bottom: sample at nzs depths
            layered = env.bottom_layered
            total_thick = layered.total_thickness()
            sedlayer_lay = max(total_thick, self.dz)

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
                      f"nzs={nzs}, sedlayer={sedlayer_lay:.1f} m", level='info')

            return sedlayer_lay, nzs, cs, rho_arr, attn_arr, 0, ''

        if hasattr(env, 'bottom_rd') and env.bottom_rd is not None and work_dir is not None:
            # Range-dependent halfspace bottom
            bottom_rd = env.bottom_rd
            n_ranges = len(bottom_rd.ranges_km)

            cs_profiles = np.zeros((nzs, n_ranges))
            rho_profiles = np.zeros((nzs, n_ranges))
            attn_profiles = np.zeros((nzs, n_ranges))

            for i in range(n_ranges):
                cb = bottom_rd.sound_speed[i]
                if env.has_range_dependent_ssp():
                    ssp_at_range = env.get_ssp_at_range(bottom_rd.ranges_km[i] * 1000.0)
                    cwg_local = float(np.interp(bottom_rd.depths[i],
                                                ssp_at_range[:, 0], ssp_at_range[:, 1]))
                else:
                    cwg_local = float(np.interp(bottom_rd.depths[i],
                                                env.ssp_data[:, 0], env.ssp_data[:, 1]))
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
            write_sediment_file(
                work_dir / sed_filename,
                bottom_rd.ranges_km,
                cs_profiles, rho_profiles, attn_profiles
            )

            self._log(f"Range-dependent sediment: {n_ranges} profiles, nzs={nzs}", level='info')

            cs = cs_profiles[:, 0].copy()
            rho_arr = rho_profiles[:, 0].copy()
            attn_arr = attn_profiles[:, 0].copy()
            return sedlayer, nzs, cs, rho_arr, attn_arr, 1, sed_filename

        # Range-independent halfspace bottom
        cs_offset = 200.0
        rho_val = 1.2
        attn_val = 0.5

        if hasattr(env, 'bottom') and env.bottom is not None:
            bottom = env.bottom
            cb_val = getattr(bottom, 'sound_speed', 1600.0) or 1600.0
            rho_val = getattr(bottom, 'density', 1.2) or 1.2
            attn_val = getattr(bottom, 'attenuation', 0.5) or 0.5
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
        zmax=_UNSET,
        **kwargs
    ) -> Field:
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
            COHERENT_TL (default) for narrowband TL grid,
            TIME_SERIES for broadband complex field.
        zmax : float, optional
            Per-call override for constructor default.
        **kwargs
            Additional parameters (currently unused)

        Returns
        -------
        field : Field
            For COHERENT_TL: transmission loss field (depth x range)
            For TIME_SERIES: complex transfer function (depth x frequency x range)
        """
        if run_mode is None:
            run_mode = RunMode.COHERENT_TL


        with _resolve_overrides(self, zmax=zmax):
            import time
            start_time = time.time()

            self.validate_inputs(env, source, receiver)

            if run_mode == RunMode.TIME_SERIES:
                return self._run_broadband(env, source, receiver, **kwargs)
            else:
                return self._run_tl(env, source, receiver, **kwargs)

    def _compute_dr(self, freq: float, c0: float = DEFAULT_SOUND_SPEED) -> float:
        """
        Compute adaptive range step based on frequency.

        Uses dr = c0/freq (one wavelength) capped between 1m and 500m.
        For shallow water at high frequency this gives fine stepping;
        for deep water at low frequency it gives coarser stepping.
        """
        dr = c0 / freq
        return float(np.clip(dr, 1.0, 500.0))

    def _run_tl(self, env, source, receiver, **kwargs):
        """
        Run in narrowband TL mode.

        Sets Q to a very large value so bandwidth -> 0 and nf~3.
        All receiver ranges are passed to mpiramS at once.
        """
        import time
        start_time = time.time()

        freq = float(source.frequency[0])
        zsrc = float(source.depth[0])
        ranges = receiver.ranges
        rmax = float(np.max(ranges))

        # Adaptive range step if not explicitly set
        dr = self.dr if self.dr is not None else self._compute_dr(freq)

        self._log(f"Running RAM (TL mode): freq={freq:.1f} Hz, zs={zsrc:.1f} m, dr={dr:.1f} m", level='info')
        self._log(f"  Output grid: {len(ranges)} ranges x {len(receiver.depths)} depths", level='info')

        # Use very large Q for narrowband (single frequency)
        Q_tl = 1e6
        T_tl = 1.0  # Short time window

        fm = self._setup_file_manager()
        work_dir = fm.work_dir

        try:
            # Prepare input files
            ssp_filename = self._prepare_ssp(env, work_dir, freq)
            bth_filename, ibot = self._prepare_bathymetry(env, rmax, work_dir)
            sedlayer, nzs, cs, rho_arr, attn_arr, isedrd, sed_filename = \
                self._prepare_bottom_properties(env, work_dir)

            ranges_filename = 'ranges.dat'
            write_ranges_file(work_dir / ranges_filename, ranges)

            rs = self.rs_stability if self.rs_stability is not None else rmax

            # Only use horizontal interpolation for range-dependent SSPs
            ihorz = 1 if env.has_range_dependent_ssp() else 0

            write_inpe(
                filepath=work_dir / 'in.pe',
                fc=freq,
                Q=Q_tl,
                T=T_tl,
                zsrc=zsrc,
                deltaz=self.dz,
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
                ranges_filename=ranges_filename,
                sedlayer=sedlayer,
                nzs=nzs,
                cs=cs,
                rho=rho_arr,
                attn=attn_arr,
                isedrd=isedrd,
                sed_filename=sed_filename,
            )

            # Run mpiramS
            self._run_binary(work_dir)

            # Read output
            result = read_psif(work_dir)

            psif = result['psif']  # (nzo, nf, nr)
            zg = result['zg']
            rout = result['rout']
            c0 = result['c0']

            # Extract center frequency (middle of frequency vector)
            center_idx = result['nf'] // 2
            # pressure at center freq for all depths and ranges: (nzo, nr)
            pressure = psif[:, center_idx, :]

            # Interpolate COMPLEX PRESSURE from PE grid to receiver grid
            # BEFORE computing TL. Interpolating in dB destroys interference
            # nulls because linear interpolation of log-scale values smooths
            # out the sharp zeros in the field.
            from scipy.interpolate import RegularGridInterpolator

            rcv_depths = receiver.depths
            rcv_ranges = np.clip(receiver.ranges, rout[0], rout[-1])

            # Interpolate real and imaginary parts separately
            interp_re = RegularGridInterpolator(
                (zg.astype(np.float64), rout.astype(np.float64)),
                np.nan_to_num(pressure.real).astype(np.float64),
                method='linear', bounds_error=False, fill_value=0.0
            )
            interp_im = RegularGridInterpolator(
                (zg.astype(np.float64), rout.astype(np.float64)),
                np.nan_to_num(pressure.imag).astype(np.float64),
                method='linear', bounds_error=False, fill_value=0.0
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
            with np.errstate(divide='ignore', invalid='ignore'):
                psi_mag = np.abs(pressure_rcv) * 4.0 * np.pi
                tl_output = -20.0 * np.log10(psi_mag + TL_FLOOR_PRESSURE) + 10.0 * np.log10(rcv_ranges)[np.newaxis, :]

            tl_output = np.clip(tl_output, 0.0, TL_MAX_DB)

            # Mask TL values below the seafloor at each range.
            # RAM computes valid field in the sediment, but for plotting
            # consistency with other models, set sub-bottom values to NaN.
            if hasattr(env, 'bathymetry') and env.bathymetry is not None and len(env.bathymetry) > 0:
                bathy = env.bathymetry
                bathy_depths = np.interp(receiver.ranges, bathy[:, 0], bathy[:, 1])
            else:
                bathy_depths = np.full(len(receiver.ranges), env.depth)
            for j, bd in enumerate(bathy_depths):
                mask = receiver.depths > bd
                tl_output[mask, j] = np.nan

            elapsed = time.time() - start_time
            self._log(f"RAM TL completed in {elapsed:.2f}s", level='info')
            self._log(f"  TL range: {np.nanmin(tl_output):.1f} to {np.nanmax(tl_output):.1f} dB", level='info')

            return Field(
                field_type='tl',
                data=tl_output,
                ranges=receiver.ranges,
                depths=receiver.depths,
                frequencies=np.array([freq]),
            )

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _run_broadband(self, env, source, receiver, **kwargs):
        """
        Run in broadband mode (native mpiramS use case).

        Returns complex transfer function psi(depth, frequency, range).
        """
        import time
        start_time = time.time()

        freq = float(source.frequency[0])
        zsrc = float(source.depth[0])
        ranges = receiver.ranges
        rmax = float(np.max(ranges))

        # Adaptive range step if not explicitly set
        dr = self.dr if self.dr is not None else self._compute_dr(freq)

        self._log(f"Running RAM (broadband): fc={freq:.1f} Hz, Q={self.Q}, T={self.T}s, dr={dr:.1f} m", level='info')
        self._log(f"  Bandwidth: {freq/self.Q:.2f} Hz", level='info')

        fm = self._setup_file_manager()
        work_dir = fm.work_dir

        try:
            ssp_filename = self._prepare_ssp(env, work_dir, freq)
            bth_filename, ibot = self._prepare_bathymetry(env, rmax, work_dir)
            sedlayer, nzs, cs, rho_arr, attn_arr, isedrd, sed_filename = \
                self._prepare_bottom_properties(env, work_dir)

            ranges_filename = 'ranges.dat'
            write_ranges_file(work_dir / ranges_filename, ranges)

            rs = self.rs_stability if self.rs_stability is not None else rmax

            ihorz = 1 if env.has_range_dependent_ssp() else 0

            write_inpe(
                filepath=work_dir / 'in.pe',
                fc=freq,
                Q=self.Q,
                T=self.T,
                zsrc=zsrc,
                deltaz=self.dz,
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
                ranges_filename=ranges_filename,
                sedlayer=sedlayer,
                nzs=nzs,
                cs=cs,
                rho=rho_arr,
                attn=attn_arr,
                isedrd=isedrd,
                sed_filename=sed_filename,
            )

            self._run_binary(work_dir)

            result = read_psif(work_dir)

            # Convert mpiramS psif to actual pressure spectrum.
            #
            # mpiramS stores: psif = psi * exp(i*(k0*r + pi/4)) / (4*pi)
            # Collins' PE: p(f,r,z) = psi(f,r,z) * exp(i*k0*r) / sqrt(r)
            # Therefore:  p = psif * 4*pi * exp(-i*pi/4) / sqrt(r)
            #
            # The k0*r phase terms cancel: exp(-i*k0*r) * exp(+i*k0*r) = 1.
            # The conversion factor is constant (not frequency-dependent).
            #
            # mpiramS uses exp(+i*omega*t) convention (opposite standard).
            # Conjugation converts to exp(-i*omega*t) consistent with other
            # models (Bellhop, Scooter, KrakenField).
            psif = result['psif']  # (nzo, nf, nr)
            rout = result['rout']  # (nr,)
            zg = result['zg']
            scale = 4.0 * np.pi * np.exp(-1j * np.pi / 4.0) / np.sqrt(rout)
            # Broadcast: psif is (nzo, nf, nr), scale is (nr,)
            pressure = np.conj(psif) * scale[np.newaxis, np.newaxis, :]

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
            self._log(f"RAM broadband completed in {elapsed:.2f}s", level='info')
            self._log(f"  Output: {len(out_depths)} depths x {result['nf']} freqs x {result['nr']} ranges", level='info')

            return Field(
                field_type='transfer_function',
                data=pressure,  # (n_depths, nf, nr) complex pressure
                ranges=rout,
                depths=out_depths,
                frequencies=result['frq'],
                metadata={
                    'Nsam': result['Nsam'],
                    'fs': result['fs'],
                    'Q': result['Q'],
                    'c0': result['c0'],
                    'cmin': result['cmin'],
                },
            )

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _run_binary(self, work_dir: Path):
        """Execute s_mpiram in the given working directory."""
        self._log(f"Executing: {self.executable}", level='info')

        env = os.environ.copy()
        # Limit OpenMP threads to avoid oversubscription
        if 'OMP_NUM_THREADS' not in env:
            env['OMP_NUM_THREADS'] = str(os.cpu_count() or 1)

        try:
            result = subprocess.run(
                [str(self.executable)],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("mpiramS execution timed out (600s limit)")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"mpiramS binary not found at {self.executable}. "
                f"Run install.sh to compile."
            )

        if self.verbose and result.stdout:
            self._log(f"mpiramS output:\n{result.stdout}", level='debug')

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            raise RuntimeError(f"mpiramS failed (code {result.returncode}): {error_msg}")

        # Verify output exists
        if not (work_dir / 'psif.dat').exists():
            raise RuntimeError(
                f"mpiramS produced no output. Check input parameters.\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
