"""
OASES Input File Writers

This module provides functions for writing input files for OASES models:
- OAST: Transmission loss module (wavenumber integration)
- OASN: Noise/covariance module (also used for normal mode computation)
- OASR: Range-dependent model (stub for future implementation)

OASES (Ocean Acoustics and Seismic Exploration Synthesis) was developed by
Henrik Schmidt at MIT.

References:
    Schmidt, H. (2004). OASES Version 3.1 User Guide and Reference Manual.
"""

from pathlib import Path
from typing import Optional, Union, List, TYPE_CHECKING
import numpy as np

from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.constants import DEFAULT_SOUND_SPEED

# Avoid circular import
if TYPE_CHECKING:
    from uacpy.models.base import RunMode


def _format_upper_halfspace(env: Environment) -> str:
    """Format the OASES upper halfspace line from env.surface properties.

    OASES layer format: D CC CS AC AS RO RG [IG]
    For vacuum: all zeros.
    For elastic (ice): sound_speed, shear_speed, attenuation, density.
    """
    surface = env.surface
    acoustic_type = getattr(surface, 'acoustic_type', 'vacuum')
    if isinstance(acoustic_type, str):
        atype = acoustic_type.lower()
    else:
        atype = str(acoustic_type).lower()

    if 'vacuum' in atype:
        return "0 0 0 0 0 0 0"

    # Elastic / half-space / rigid surface (e.g. ice)
    c_p = getattr(surface, 'sound_speed', 0.0) or 0.0
    c_s = getattr(surface, 'shear_speed', 0.0) or 0.0
    alpha_p = getattr(surface, 'attenuation', 0.0) or 0.0
    alpha_s = getattr(surface, 'shear_attenuation', 0.0) or 0.0
    rho = getattr(surface, 'density', 0.0) or 0.0

    if 'rigid' in atype:
        # Rigid: very high impedance — use large density, zero speed
        return "0 0 0 0 0 0 0"

    return (f"0 {c_p:.2f} {c_s:.2f} {alpha_p:.3f} "
            f"{alpha_s:.3f} {rho:.2f} 0")


def write_oast_input(
    filepath: Union[str, Path],
    env: Environment,
    source: Source,
    receiver: Receiver,
    run_mode: Optional[Union[str, 'RunMode']] = None,
    options: Optional[str] = None,
    **kwargs
) -> None:
    """
    Write OAST (OASES Transmission Loss) input file

    OAST uses wavenumber integration with Direct Global Matrix solution.
    Supports elastic layers and seismo-acoustic propagation.

    Parameters
    ----------
    filepath : str or Path
        Output file path (typically .dat extension)
    env : Environment
        Ocean environment (must be range-independent)
    source : Source
        Acoustic source specification
    receiver : Receiver
        Receiver array specification
    run_mode : RunMode, optional
        Computation mode. Default is COHERENT_TL.
    options : str, optional
        OAST option string. If None, uses default 'N J T'
        Common options:
        - N: Normal stress (pressure)
        - J: Complex integration contour
        - T: Transmission loss vs range plot
        - C: Range-depth contour plot
        - I: Integrands plot (for debugging)
        - Z: Sound speed profile plot
    **kwargs : dict
        Additional parameters:
        - integration_offset : float
            Integration contour offset in dB/wavelength (default: 0)
        - nw_samples : int
            Number of wavenumber samples (default: -1 for automatic)
        - plot_rmin : float
            Minimum range for plots in km (default: 0)
        - plot_rmax : float
            Maximum range for plots in km (default: max receiver range)

    Notes
    -----
    OAST input file has 12 blocks:
    I.   Title
    II.  Options (output control)
    III. Frequencies
    IV.  Environment (layers)
    V.   Sources
    VI.  Receivers
    VII. Wavenumber sampling
    VIII. Range axes (for plots)
    IX.  Transmission loss axes
    X.   Depth axes
    XI.  Contour levels
    XII. SVP axes (sound velocity profile)

    Examples
    --------
    >>> env = Environment(depth=100, sound_speed=1500)
    >>> source = Source(depth=50, frequency=100)
    >>> receiver = Receiver(depths=np.linspace(10,90,40),
    ...                     ranges=np.linspace(100,10000,100))
    >>> write_oast_input('test.dat', env, source, receiver)
    """
    filepath = Path(filepath)

    # Extract parameters
    freq = float(source.frequency[0])
    depth = env.depth

    # Bottom properties
    bottom = env.bottom
    rho = bottom.density if hasattr(bottom, 'density') else 1.5
    c_p = bottom.sound_speed if hasattr(bottom, 'sound_speed') else 1600.0
    c_s = bottom.shear_speed if hasattr(bottom, 'shear_speed') else 0.0
    alpha_p = bottom.attenuation if hasattr(bottom, 'attenuation') else 0.5
    alpha_s = bottom.shear_attenuation if hasattr(bottom, 'shear_attenuation') else 0.0

    # Sound speed profile
    ssp_data = env.ssp_data
    if ssp_data is None or len(ssp_data) == 0:
        # Fallback to simple isovelocity
        c = env.sound_speed if env.sound_speed else DEFAULT_SOUND_SPEED
        ssp_data = np.array([[0.0, c], [depth, c]])

    # Source and receiver parameters
    src_depth = float(source.depth[0])
    z_min = float(receiver.depths.min())
    z_max = float(receiver.depths.max())
    n_depths = len(receiver.depths)
    r_min = float(receiver.ranges.min())
    r_max = float(receiver.ranges.max())

    # Optional parameters
    integration_offset = kwargs.get('integration_offset', 0)
    nw_samples = kwargs.get('nw_samples', -1)  # -1 = automatic
    plot_rmin = kwargs.get('plot_rmin', 0)
    plot_rmax = kwargs.get('plot_rmax', r_max / 1000.0)  # Convert m to km

    # Get reference sound speed for plot axes
    c_ref = float(ssp_data[0, 1])  # Sound speed at surface

    # Options string
    if options is None:
        options = 'N J T'  # Normal stress, complex contour, TL vs range

    with open(filepath, 'w') as f:
        # Block I: Title
        title = env.name if env.name else "OAST Simulation via UACPY"
        f.write(f"{title}\n")

        # Block II: OPTIONS
        f.write(f"{options}\n")

        # Block III: Frequencies
        # FREQ1 FREQ2 NFREQ COFF
        f.write(f"{freq:.1f} {freq:.1f} 1 {integration_offset}\n")

        # Block IV: Environment
        # OASES uses CS = -999.999 to indicate continuous SVP gradient to next layer
        # Reference: oases_gen.tex documentation

        # Subsample SSP - OASES works best with ~10-15 layers for complex SSPs
        # Too many layers with -999.999 gradient flag causes numerical issues
        max_ssp_layers = 15
        if len(ssp_data) > max_ssp_layers:
            # Keep first, last, and evenly spaced points
            indices = list(range(0, len(ssp_data), max(1, len(ssp_data) // (max_ssp_layers - 1))))
            if indices[-1] != len(ssp_data) - 1:
                indices.append(len(ssp_data) - 1)
            # Remove duplicates and sort
            indices = sorted(set(indices))
            ssp_subset = ssp_data[indices, :]
        else:
            ssp_subset = ssp_data

        # Special case: if isovelocity (all sound speeds are the same), write only 1 layer
        # Check if all sound speeds are equal within tolerance
        c_values = ssp_subset[:, 1]
        is_isovelocity = np.allclose(c_values, c_values[0], rtol=1e-6)

        # Count sediment layers from LayeredBottom
        n_sed_layers = 0
        if hasattr(env, 'bottom_layered') and env.bottom_layered is not None:
            n_sed_layers = len(env.bottom_layered.layers)

        if is_isovelocity:
            # Isovelocity: write only 1 water layer at surface
            n_layers = 3 + n_sed_layers  # vacuum + water + sed_layers + bottom
            f.write(f"{n_layers}\n")

            # Upper halfspace (from env.surface)
            f.write(f"{_format_upper_halfspace(env)}\n")

            # Single water layer
            f.write(f"0.00 {c_values[0]:.2f} 0 0.0 0 1.0 0 0\n")

            # Sediment layers (if layered bottom)
            if n_sed_layers > 0:
                current_depth = depth
                for layer in env.bottom_layered.layers:
                    layer_as = getattr(layer, 'shear_attenuation', 0.0)
                    f.write(f"{current_depth:.2f} {layer.sound_speed:.2f} "
                            f"{layer.shear_speed:.2f} {layer.attenuation:.3f} "
                            f"{layer_as:.3f} {layer.density:.2f} 0 0\n")
                    current_depth += layer.thickness

                # Bottom halfspace at depth below all layers
                f.write(f"{current_depth:.2f} {c_p:.2f} {c_s:.2f} "
                        f"{alpha_p:.3f} {alpha_s:.3f} {rho:.2f} 0 0\n")
            else:
                # Bottom halfspace
                f.write(f"{depth:.2f} {c_p:.2f} {c_s:.2f} {alpha_p:.3f} {alpha_s:.3f} {rho:.2f} 0 0\n")
        else:
            # Non-isovelocity: write layers with negative CS field for gradients
            # OASES format: D CC CS means:
            #   - If CS >= 0: layer is isovelocity with shear speed CS
            #   - If CS < 0: layer has linear gradient from CC to -CS (abs value)
            n_water_layers = len(ssp_subset)
            n_layers = 1 + n_water_layers + n_sed_layers + 1
            f.write(f"{n_layers}\n")

            # Upper halfspace (from env.surface)
            f.write(f"{_format_upper_halfspace(env)}\n")

            # Water layers with gradient using negative CS field
            for i in range(len(ssp_subset)):
                d, c = ssp_subset[i]
                if i < len(ssp_subset) - 1:
                    # Intermediate layer: CS = -sound_speed_at_bottom_of_layer
                    c_bottom = ssp_subset[i + 1, 1]
                    cs = -abs(c_bottom)
                else:
                    # Last water layer: no gradient, connects to bottom
                    cs = 0
                f.write(f"{d:.2f} {c:.2f} {cs:.2f} 0.0 0 1.0 0 0\n")

            # Sediment layers (if layered bottom)
            if n_sed_layers > 0:
                current_depth = depth
                for layer in env.bottom_layered.layers:
                    layer_as = getattr(layer, 'shear_attenuation', 0.0)
                    f.write(f"{current_depth:.2f} {layer.sound_speed:.2f} "
                            f"{layer.shear_speed:.2f} {layer.attenuation:.3f} "
                            f"{layer_as:.3f} {layer.density:.2f} 0 0\n")
                    current_depth += layer.thickness

                # Bottom halfspace at depth below all layers
                f.write(f"{current_depth:.2f} {c_p:.2f} {c_s:.2f} "
                        f"{alpha_p:.3f} {alpha_s:.3f} {rho:.2f} 0 0\n")
            else:
                # Bottom halfspace
                f.write(f"{depth:.2f} {c_p:.2f} {c_s:.2f} {alpha_p:.3f} {alpha_s:.3f} {rho:.2f} 0 0\n")

        # Block V: Sources
        # SD NS DS AN IA FD DA
        # SD = source depth, NS = number of sources (1 for single source)
        f.write(f"{src_depth:.2f} 1 0 0 1 0 0\n")

        # Block VI: Receivers
        # RD1 RD2 NR IR
        f.write(f"{z_min:.2f} {z_max:.2f} {n_depths} 1\n")

        # Block VII: Wavenumber sampling
        # CMIN CMAX
        c_water_min = float(ssp_data[:, 1].min())
        cmin = c_water_min * 0.9
        cmax = max(c_p * 1.1, 1e8)
        f.write(f"{cmin:.1f} {cmax:.1e}\n")

        # NW IC1 IC2 (use -1 for automatic sampling)
        f.write(f"{nw_samples} 1 2000\n")

        # Block VIII: Range axes (for plots)
        # RMIN RMAX RLEN RINC
        f.write(f"{plot_rmin:.1f} {plot_rmax:.1f} 20 1\n")

        # Block IX: Transmission loss axes
        # TMIN TMAX TLEN TINC
        f.write("20 100 12 10\n")

        # Block X: Depth axes (for contour plots)
        # DUP DLO DLN DIN
        f.write(f"0 {depth:.1f} 12 {depth/10:.1f}\n")

        # Block XI: Contour levels
        # ZMIN ZMAX ZINC
        f.write("40 100 10\n")

        # Block XII: SVP axes (if option Z used)
        # VLEF VRIG VLEN VINC
        c_min = float(ssp_data[:, 1].min())
        c_max = float(ssp_data[:, 1].max())
        c_range = c_max - c_min
        if c_range < 1.0:  # For isovelocity, add some margin
            c_range = c_ref * 0.1
        c_plot_min = c_min - c_range * 0.05
        c_plot_max = c_max + c_range * 0.05
        c_inc = max(10, c_range / 10)
        f.write(f"{c_plot_min:.1f} {c_plot_max:.1f} 12 {c_inc:.1f}\n")
        # DVUP DVLO DVLN DVIN
        f.write(f"0 {depth:.1f} 12 {depth/10:.1f}\n")


def write_oasn_input(
    filepath: Union[str, Path],
    env: Environment,
    source: Source,
    receiver: Receiver,
    run_mode: Optional[Union[str, 'RunMode']] = None,
    options: Optional[str] = None,
    **kwargs
) -> None:
    """
    Write OASN (OASES Noise/Normal Mode) input file

    OASN can compute:
    - Normal modes and mode shapes
    - Covariance matrices for ambient noise
    - Signal replicas for matched field processing

    Parameters
    ----------
    filepath : str or Path
        Output file path (typically .dat extension)
    env : Environment
        Ocean environment (must be range-independent)
    source : Source
        Source specification (frequency used for mode computation)
    receiver : Receiver
        Receiver array specification (for covariance matrices)
    run_mode : RunMode, optional
        Computation mode. Default is MODES.
    options : str, optional
        OASN option string. If None, uses 'N J' for normal mode computation
        Common options:
        - N: Output covariance matrices to .xsm file
        - R: Output replicas to .rpo file
        - J: Complex integration contour
        - F: Noise level vs frequency plot
        - P: Noise intensity vs receiver plot
    **kwargs : dict
        Additional parameters:
        - surface_noise_level : float
            Surface noise source strength in dB (default: 0, disabled)
        - white_noise_level : float
            White noise level in dB (default: 0, disabled)
        - discrete_sources : list of dict
            List of discrete sources with 'depth', 'x', 'y', 'level'
        - n_modes : int
            Number of modes to compute (optional)

    Notes
    -----
    OASN input file has up to 10 blocks:
    I.   Title
    II.  Options
    III. Frequencies
    IV.  Environment
    V.   Receiver Array
    VI.  Sources (noise and discrete)
    VII. Surface noise parameters (if SSLEV != 0)
    VIII. Deep noise parameters (if DSLEV != 0)
    IX.  Discrete source parameters (if NDNS > 0)
    X.   Replica parameters (if option R)

    For normal mode computation, set options='N J' and ensure
    receiver array is specified properly.

    Examples
    --------
    >>> env = Environment(depth=100, sound_speed=1500)
    >>> source = Source(depth=50, frequency=100)
    >>> receiver = Receiver(depths=[30, 50, 70], ranges=[0])
    >>> write_oasn_input('test.dat', env, source, receiver,
    ...                  options='N J', surface_noise_level=70)
    """
    filepath = Path(filepath)

    # Extract parameters
    freq = float(source.frequency[0])
    depth = env.depth

    # Bottom properties
    bottom = env.bottom
    rho = bottom.density if hasattr(bottom, 'density') else 1.5
    c_p = bottom.sound_speed if hasattr(bottom, 'sound_speed') else 1600.0
    c_s = bottom.shear_speed if hasattr(bottom, 'shear_speed') else 0.0
    alpha_p = bottom.attenuation if hasattr(bottom, 'attenuation') else 0.5
    alpha_s = bottom.shear_attenuation if hasattr(bottom, 'shear_attenuation') else 0.0

    # Sound speed profile
    ssp_data = env.ssp_data
    if ssp_data is None or len(ssp_data) == 0:
        # Fallback to simple isovelocity
        c = env.sound_speed if env.sound_speed else DEFAULT_SOUND_SPEED
        ssp_data = np.array([[0.0, c], [depth, c]])
    c_ref = float(ssp_data[0, 1])

    # Noise/source parameters
    surface_noise_level = kwargs.get('surface_noise_level', 0)
    white_noise_level = kwargs.get('white_noise_level', 0)
    deep_noise_level = kwargs.get('deep_noise_level', 0)
    discrete_sources = kwargs.get('discrete_sources', [])
    n_discrete = len(discrete_sources)

    # Options string
    if options is None:
        options = 'N J'  # Covariance output, complex contour

    # Integration parameters
    integration_offset = kwargs.get('integration_offset', 0)

    with open(filepath, 'w') as f:
        # Block I: Title
        title = env.name if env.name else "OASN Simulation via UACPY"
        f.write(f"{title}\n")

        # Block II: OPTIONS
        f.write(f"{options}\n")

        # Block III: Frequencies
        # FREQ1 FREQ2 NFREQ COFF
        f.write(f"{freq:.1f} {freq:.1f} 1 {integration_offset}\n")

        # Block IV: Environment
        # NL = number of layers (including halfspaces and all water layers)
        n_water_layers = len(ssp_data)
        n_layers = 1 + n_water_layers + 1
        f.write(f"{n_layers}\n")

        # Upper halfspace (from env.surface)
        f.write(f"{_format_upper_halfspace(env)} 0\n")

        # Water layers from SSP using OASES negative sound speed convention
        # In OASES: negative CC means linear gradient from current layer to next
        # Example: layer at 100m with c=1500, next at 200m with c=-1520
        # means sound speed varies linearly from 1500 at 100m to 1520 at 200m

        # Write first SSP point (top of water column)
        d0, c0 = ssp_data[0]
        f.write(f"{d0:.2f} {c0:.2f} 0 0.0 0 1.0 0 0\n")

        # Write remaining SSP points with negative sound speeds to indicate gradients
        for i in range(1, len(ssp_data)):
            d, c = ssp_data[i]
            # Use negative sound speed to tell OASES to interpolate linearly
            f.write(f"{d:.2f} {-abs(c):.2f} 0 0.0 0 1.0 0 0\n")

        # Bottom halfspace
        f.write(f"{depth:.2f} {c_p:.2f} {c_s:.2f} {alpha_p:.3f} {alpha_s:.3f} {rho:.2f} 0 0\n")

        # Block V: Receiver Array
        # NRCV
        n_receivers = len(receiver.depths)
        f.write(f"{n_receivers}\n")

        # For each receiver: Z X Y ITYP GAIN
        # ITYP=1 for hydrophone, GAIN in dB
        for z in receiver.depths:
            f.write(f"{z:.2f} 0 0 1 0\n")

        # Block VI: Sources
        # SSLEV WNLEV DSLEV NDNS
        f.write(f"{surface_noise_level:.1f} {white_noise_level:.1f} {deep_noise_level:.1f} {n_discrete}\n")

        # Block VII: Surface noise parameters (if surface_noise_level != 0)
        if surface_noise_level != 0:
            # CMINS CMAXS
            c_water_min = float(ssp_data[:, 1].min())
            f.write(f"{c_water_min*0.95:.1f} 1E8\n")
            # NWSC NWSD NWSE (samples in continuous, discrete, evanescent)
            f.write("400 400 100\n")

        # Block VIII: Deep noise parameters (if deep_noise_level != 0)
        if deep_noise_level != 0:
            # DPSD (depth of deep source sheet)
            deep_source_depth = kwargs.get('deep_source_depth', depth * 0.5)
            f.write(f"{deep_source_depth:.2f}\n")
            # CMIND CMAXD
            c_water_min = float(ssp_data[:, 1].min())
            f.write(f"{c_water_min*0.95:.1f} 1E8\n")
            # NWDC NWDD NWDE
            f.write("400 400 100\n")

        # Block IX: Discrete sources (if n_discrete > 0)
        if n_discrete > 0:
            for ds in discrete_sources:
                # ZDN XDN YDN DNLEV (depth in m, x/y in km, level in dB)
                z_ds = ds.get('depth', 50.0)
                x_ds = ds.get('x', 1.0)  # km
                y_ds = ds.get('y', 0.0)  # km
                level_ds = ds.get('level', 180.0)
                f.write(f"{z_ds:.2f} {x_ds:.3f} {y_ds:.3f} {level_ds:.1f}\n")

            # Wavenumber sampling for discrete sources
            # CMIN CMAX
            c_water_min = float(ssp_data[:, 1].min())
            f.write(f"{c_water_min*0.95:.1f} {c_p*1.05:.1f}\n")
            # NW IC1 IC2 (-1 for automatic)
            f.write("-1 1 2000\n")

        # Block X: Replica parameters (if option 'R' is present)
        if 'R' in options or 'r' in options:
            # Replica grid: depths, x-ranges, y-ranges
            replica_zmin = kwargs.get('replica_zmin', 10.0)
            replica_zmax = kwargs.get('replica_zmax', depth - 10.0)
            replica_nz = kwargs.get('replica_nz', 20)
            replica_xmin = kwargs.get('replica_xmin', 0.1)  # km
            replica_xmax = kwargs.get('replica_xmax', 10.0)  # km
            replica_nx = kwargs.get('replica_nx', 50)

            f.write(f"{replica_zmin:.2f} {replica_zmax:.2f} {replica_nz}\n")
            f.write(f"{replica_xmin:.3f} {replica_xmax:.3f} {replica_nx}\n")
            f.write("0 0 1\n")  # y-range (omnidirectional)

            # Wavenumber sampling for replicas
            c_water_min = float(ssp_data[:, 1].min())
            f.write(f"{c_water_min*0.95:.1f} {c_p*1.05:.1f}\n")
            f.write("-1 1 2000\n")


def write_oasp_input(
    filepath: Union[str, Path],
    env: Environment,
    source: Source,
    receiver: Receiver,
    run_mode: Optional[Union[str, 'RunMode']] = None,
    options: Optional[str] = None,
    **kwargs
) -> None:
    """
    Write OASP (OASES Parabolic Equation) input file

    OASP uses split-step Padé PE for broadband/time-domain propagation.
    Generates transfer functions for postprocessing with PP module.

    Parameters
    ----------
    filepath : str or Path
        Output file path (typically .dat extension)
    env : Environment
        Ocean environment (can be range-dependent)
    source : Source
        Acoustic source specification
    receiver : Receiver
        Receiver array specification
    run_mode : RunMode, optional
        Computation mode (default: None)
    options : str, optional
        OASP option string. If None, uses default 'N V J'
        Common options:
        - N: Normal stress (pressure)
        - V: Vertical velocity
        - H: Horizontal velocity
        - J: Complex wavenumber contour
        - O: Complex frequency contour (for time-domain damping)
        - C: Omega-k contour plot
        - Z: Sound speed profile plot
        - f: Full Hankel transform for near field
    **kwargs : dict
        Additional parameters:
        - center_frequency : float
            Center frequency in Hz (default: source frequency)
        - integration_offset : float
            Integration contour offset in dB/wavelength (default: 0)
        - n_time_samples : int
            Number of time samples, must be power of 2 (default: 4096)
        - freq_min : float
            Lower frequency limit in Hz (default: 0)
        - freq_max : float
            Upper frequency limit in Hz (default: center_freq*2.5)
        - time_step : float
            Time sampling increment in seconds (default: auto)
        - range_start : float
            First range in km (default: min receiver range)
        - range_step : float
            Range increment in km (default: auto)
        - nw_samples : int
            Number of wavenumber samples (default: -1 for automatic)

    Notes
    -----
    OASP input file has 8 blocks:
    I.   Title
    II.  Options
    III. Source frequency and integration offset
    IV.  Environment (layers)
    V.   Sources
    VI.  Receiver depths
    VII. Wavenumber sampling
    VIII. Frequency and range sampling

    OASP outputs .trf files (transfer functions) for postprocessing.

    Examples
    --------
    >>> env = Environment(depth=100, sound_speed=1500)
    >>> source = Source(depth=80, frequency=30)
    >>> receiver = Receiver(depths=np.linspace(20,100,5),
    ...                     ranges=np.linspace(1000,5000,5))
    >>> write_oasp_input('pulse.dat', env, source, receiver)
    """
    filepath = Path(filepath)

    # Extract parameters
    center_freq = kwargs.get('center_frequency', float(source.frequency[0]))
    depth = env.depth

    # Bottom properties
    bottom = env.bottom
    rho = bottom.density if hasattr(bottom, 'density') else 1.5
    c_p = bottom.sound_speed if hasattr(bottom, 'sound_speed') else 1600.0
    c_s = bottom.shear_speed if hasattr(bottom, 'shear_speed') else 0.0
    alpha_p = bottom.attenuation if hasattr(bottom, 'attenuation') else 0.5
    alpha_s = bottom.shear_attenuation if hasattr(bottom, 'shear_attenuation') else 0.0

    # Sound speed profile
    ssp_data = env.ssp_data
    if ssp_data is None or len(ssp_data) == 0:
        # Fallback to simple isovelocity
        c = env.sound_speed if env.sound_speed else DEFAULT_SOUND_SPEED
        ssp_data = np.array([[0.0, c], [depth, c]])

    # Source and receiver parameters
    src_depth = float(source.depth[0])
    rd1 = float(receiver.depths.min())
    rd2 = float(receiver.depths.max())
    nrd = len(receiver.depths)

    # Frequency and time parameters
    n_time = kwargs.get('n_time_samples', 4096)
    freq_min = kwargs.get('freq_min', 0.0)
    freq_max = kwargs.get('freq_max', center_freq * 2.5)

    # Auto-calculate time step if not provided
    if 'time_step' in kwargs:
        dt = kwargs['time_step']
    else:
        # Nyquist: dt = 1/(2*freq_max)
        dt = 1.0 / (2.0 * freq_max)

    # Range parameters
    r_min_m = float(receiver.ranges.min())
    r_max_m = float(receiver.ranges.max())
    r1_km = kwargs.get('range_start', r_min_m / 1000.0)

    if 'range_step' in kwargs:
        dr_km = kwargs['range_step']
    else:
        # Auto-calculate based on receiver spacing
        n_ranges = len(receiver.ranges)
        if n_ranges > 1:
            dr_km = (r_max_m - r_min_m) / (n_ranges - 1) / 1000.0
        else:
            dr_km = 1.0  # Default 1 km

    nr = len(receiver.ranges)

    # Wavenumber sampling
    integration_offset = kwargs.get('integration_offset', 0)
    nw_samples = kwargs.get('nw_samples', -1)  # -1 = automatic
    c_water_min = float(ssp_data[:, 1].min())
    cmin = c_water_min * 0.9
    cmax = max(c_p * 1.1, 1e9)

    # Options string
    if options is None:
        options = 'N V J'  # Normal stress, vertical velocity, complex contour

    with open(filepath, 'w') as f:
        # Block I: Title
        title = env.name if env.name else "OASP Simulation via UACPY"
        f.write(f"{title}\n")

        # Block II: OPTIONS
        f.write(f"{options}\n")

        # Block III: Source frequency and integration offset
        # FRC COFF [IT VS VR for Doppler]
        f.write(f"{center_freq:.1f} {integration_offset}\n")

        # Block IV: Environment
        # NL = number of layers (including halfspaces and all water layers)
        n_water_layers = len(ssp_data)
        n_layers = 1 + n_water_layers + 1
        f.write(f"{n_layers}\n")

        # Upper halfspace (from env.surface)
        f.write(f"{_format_upper_halfspace(env)} 0\n")

        # Water layers from SSP
        for i in range(len(ssp_data)):
            d, c = ssp_data[i]
            f.write(f"{d:.2f} {c:.2f} 0 0.0 0 1.0 0 0 0\n")

        # Bottom halfspace
        f.write(f"{depth:.2f} {c_p:.2f} {c_s:.2f} {alpha_p:.3f} {alpha_s:.3f} {rho:.2f} 0 0 0\n")

        # Block V: Sources
        # SD NS DS AN IA FD DA
        # For single source: SD 1 0 0 1 0 0
        f.write(f"{src_depth:.2f} 1 0 0 1 0 0\n")

        # Block VI: Receiver depths
        # RD1 RD2 NRD
        f.write(f"{rd1:.2f} {rd2:.2f} {nrd}\n")

        # Block VII: Wavenumber sampling
        # CMIN CMAX
        f.write(f"{cmin:.1f} {cmax:.1e}\n")

        # NW IC1 IC2 IF
        # NW=-1 for automatic sampling
        # IF = frequency sample increment for kernels (0 for all)
        f.write(f"{nw_samples} 1 1 40\n")

        # Block VIII: Frequency and range sampling
        # NT FR1 FR2 DT R1 DR NR
        f.write(f"{n_time} {freq_min:.1f} {freq_max:.1f} {dt:.6f} {r1_km:.3f} {dr_km:.3f} {nr}\n")


def write_oasr_input(
    filepath: Union[str, Path],
    env: Environment,
    source: Source,
    receiver: Receiver,
    options: Optional[str] = None,
    **kwargs
) -> None:
    """
    Write OASR (OASES Reflection coefficient) input file

    OASR computes reflection/transmission coefficients as a function of
    frequency and grazing angle (or horizontal slowness). These coefficients
    can be used as input for other OASES modules or for analysis of
    seismo-acoustic interface properties.

    Parameters
    ----------
    filepath : str or Path
        Output file path (typically .dat extension)
    env : Environment
        Ocean environment specification
    source : Source
        Source specification (frequency used for computation)
    receiver : Receiver
        Receiver specification (not used directly in OASR)
    options : str, optional
        OASR option string. If None, uses default 'N T'
        Common options:
        - N: Default P-P reflection coefficient
        - T: Generate table of reflection coefficients (.rco/.trc files)
        - L: Loss in dB in addition to linear magnitude
        - P: Phase angle in addition to magnitude
        - S: P-SV reflection coefficient (replaces P-P)
        - C: Loss contours in frequency and angle
        - Z: Plot velocity profiles
        - p: Use slowness sampling instead of angle
        - t: Transmission coefficients instead of reflection
    **kwargs : dict
        Additional parameters:
        - angle_min : float
            Minimum grazing angle in degrees (default: 0)
        - angle_max : float
            Maximum grazing angle in degrees (default: 90)
        - n_angles : int
            Number of angles (default: 181)
        - freq_min : float
            Minimum frequency in Hz (default: source.frequency)
        - freq_max : float
            Maximum frequency in Hz (default: source.frequency)
        - n_frequencies : int
            Number of frequencies (default: 1)

    Notes
    -----
    OASR input file has 9 blocks (some conditional on options):
    I.   Title
    II.  Options
    III. Environment (layers)
    IV.  Frequency sampling
    V.   Angle/Slowness sampling
    VI.  Angle/Slowness axes (for plots, if output requested)
    VII. Loss/Frequency axes (for plots)
    VIII. Loss contour plots (option C)
    IX.  SVP axes (option Z)

    OASR outputs:
    - .rco: Reflection coefficient vs slowness
    - .trc: Reflection coefficient vs angle
    - .plp, .plt: Plot files
    - .cdr, .bdr: Contour plot files

    Examples
    --------
    >>> env = Environment(depth=100, sound_speed=1500)
    >>> env.bottom.sound_speed = 1600
    >>> env.bottom.shear_speed = 400
    >>> source = Source(depth=50, frequency=100)
    >>> receiver = Receiver(depths=[50], ranges=[1000])
    >>> write_oasr_input('test.dat', env, source, receiver,
    ...                  angle_min=0, angle_max=90, n_angles=91)
    """
    filepath = Path(filepath)

    # Extract parameters
    freq = float(source.frequency[0])
    depth = env.depth

    # Bottom properties
    bottom = env.bottom
    rho = bottom.density if hasattr(bottom, 'density') else 1.5
    c_p = bottom.sound_speed if hasattr(bottom, 'sound_speed') else 1600.0
    c_s = bottom.shear_speed if hasattr(bottom, 'shear_speed') else 0.0
    alpha_p = bottom.attenuation if hasattr(bottom, 'attenuation') else 0.5
    alpha_s = bottom.shear_attenuation if hasattr(bottom, 'shear_attenuation') else 0.0

    # Sound speed profile - for OASR, use water sound speed at bottom
    ssp_data = env.ssp_data
    if ssp_data is None or len(ssp_data) == 0:
        c = env.sound_speed if env.sound_speed else DEFAULT_SOUND_SPEED
    else:
        # Use sound speed at the bottom depth for reflection coefficient
        c = float(ssp_data[-1, 1])

    # Frequency parameters
    freq_min = kwargs.get('freq_min', freq)
    freq_max = kwargs.get('freq_max', freq)
    n_frequencies = kwargs.get('n_frequencies', 1)
    freq_out_inc = kwargs.get('freq_output_increment', max(1, n_frequencies // 10))

    # Angle parameters
    angle_min = kwargs.get('angle_min', 0.0)
    angle_max = kwargs.get('angle_max', 90.0)
    n_angles = kwargs.get('n_angles', 181)
    angle_out_inc = kwargs.get('angle_output_increment', max(1, n_angles // 10))

    # Options string
    if options is None:
        options = 'N T'  # Normal (P-P), Table output

    with open(filepath, 'w') as f:
        # Block I: Title
        title = env.name if env.name else "OASR Simulation via UACPY"
        f.write(f"{title}\n")

        # Block II: OPTIONS
        f.write(f"{options}\n")

        # Block III: Environment
        # NL = number of layers (including halfspaces)
        # OASR uses 2 layers: water + bottom halfspace (no vacuum layer)
        f.write("2\n")

        # Water layer (upper halfspace, starts at depth 0)
        # D CC CS AC AS RO RG
        f.write(f"0 {c:.2f} 0 0.0 0 1.0 0\n")

        # Bottom halfspace (starts at water depth)
        # For reflection coefficients, bottom is at depth=0 (interface is at water surface)
        f.write(f"0 {c_p:.2f} {c_s:.2f} {alpha_p:.3f} {alpha_s:.3f} {rho:.2f} 0\n")

        # Block IV: Frequency sampling
        # FMIN FMAX NFREQ NFOU
        f.write(f"{freq_min:.1f} {freq_max:.1f} {n_frequencies} {freq_out_inc}\n")

        # Block V: Angle/Slowness sampling
        # AMIN AMAX NRAN NAOU
        f.write(f"{angle_min:.1f} {angle_max:.1f} {n_angles} {angle_out_inc}\n")

        # Block VI: Angle/Slowness axes (if plot output)
        if freq_out_inc > 0:
            # ALEF ARIG ALEN AINC
            # RALO RAUP RALN RAIN
            f.write(f"{angle_min:.1f} {angle_max:.1f} 12 {max(10, (angle_max-angle_min)/10):.1f}\n")
            f.write("0 1 12 0.2\n")  # Reflection coefficient magnitude 0-1

        # Block VII: Loss/Frequency axes (if angle output)
        if angle_out_inc > 0:
            # FLEF FRIG FLEN FINC
            # RFLO RFUP RFLN RFIN
            f_range = max(freq_max - freq_min, freq_min * 0.1)
            f.write(f"{freq_min:.1f} {freq_max:.1f} 12 {f_range/10:.1f}\n")
            f.write("0 30 12 5\n")  # Reflection loss 0-30 dB

        # Block VIII: Loss contour plots (if option C)
        if 'C' in options or 'c' in options:
            # ALEF ARIG ALEN AINC
            # FRLO FRUP OCLN NTKM
            # ZMIN ZMAX ZINC
            f.write(f"{angle_min:.1f} {angle_max:.1f} 12 {(angle_max-angle_min)/10:.1f}\n")
            octave_range = np.log2(freq_max / freq_min) if freq_max > freq_min else 1.0
            f.write(f"{freq_min:.1f} {freq_max:.1f} {octave_range*2:.1f} 5\n")
            f.write("0 20 2\n")  # Contour levels 0-20 dB in 2 dB increments

        # Block IX: SVP axes (if option Z)
        if 'Z' in options or 'z' in options:
            # VLEF VRIG VLEN VINC
            # DVUP DVLO DVLN DVIN
            c_min = min(c, c_p) * 0.95
            c_max = max(c, c_p) * 1.05
            f.write(f"{c_min:.1f} {c_max:.1f} 12 100\n")
            f.write(f"0 {depth:.1f} 12 {depth/10:.1f}\n")
