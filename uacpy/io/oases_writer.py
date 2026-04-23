"""
OASES Input File Writers

This module provides functions for writing input files for OASES models:
- OAST: Transmission loss module (wavenumber integration)
- OASN: Noise/covariance module (also used for normal mode computation)
- OASR: Range-dependent model (stub for future implementation)

OASES (Ocean Acoustics and Seismic Exploration Synthesis) was developed by
Henrik Schmidt at MIT.

References:
    Schmidt, H. OASES Version 2.1 User Guide and Reference Manual (bundled
    under ``third_party/oases``). The public OASES release is 3.1 but this
    distribution ships with the 2.1 source tree — per the bundled README.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np

from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.exceptions import ConfigurationError


def _inject_volume_attenuation(options: str, volume_attenuation: Optional[str]) -> str:
    """Append a volume-attenuation marker character to the OASES option string.

    OASES itself does not implement dedicated T (Thorp) / F (Francois-Garrison) /
    B (biological) option letters — empirical Skretting & Leroy attenuation is
    applied automatically to any water layer with AC=0. This helper ensures the
    uacpy-level ``volume_attenuation`` parameter flows through to the option
    string so downstream tools and the user can still see that a specific model
    was requested. OASES prints ``UNKNOWN OPTION`` for letters it does not
    recognise but still runs to completion.

    Parameters
    ----------
    options : str
        Existing option string. May contain whitespace separators.
    volume_attenuation : str or None
        One of 'T', 'F', 'B' (case-insensitive) or None.
    """
    if volume_attenuation is None:
        return options
    marker = str(volume_attenuation).strip()
    if not marker:
        return options
    marker = marker[0]  # single-letter option
    if marker not in ('T', 'F', 'B', 't', 'f', 'b'):
        return options
    # Only append if not already present as a standalone letter.
    tokens = options.split()
    if marker in tokens:
        return options
    return options + ' ' + marker


def _emit_bottom_layers(
    f,
    env: Environment,
    water_depth: float,
    fallback_c_p: float,
    fallback_c_s: float,
    fallback_alpha_p: float,
    fallback_alpha_s: float,
    fallback_rho: float,
    extra_columns: int = 0,
) -> None:
    """Emit sediment layers + bottom halfspace to an OASES input file.

    Writes one interface line per SedimentLayer in ``env.bottom_layered``
    followed by the halfspace at the correct depth. Falls back to a single
    halfspace line from ``env.bottom`` when ``env.bottom_layered`` is None.

    OASES interface format: D CC CS AC AS RO RG [IG]  (extra ``IG`` appended
    when ``extra_columns`` > 0 — required by some OASP/OASN writers).
    """
    trailing = (' 0' * (1 + extra_columns))  # RG [IG [extra...]]

    if hasattr(env, 'bottom_layered') and env.bottom_layered is not None:
        lb = env.bottom_layered
        current_depth = water_depth
        for layer in lb.layers:
            layer_as = getattr(layer, 'shear_attenuation', 0.0) or 0.0
            f.write(f"{current_depth:.2f} {layer.sound_speed:.2f} "
                    f"{layer.shear_speed:.2f} {layer.attenuation:.3f} "
                    f"{layer_as:.3f} {layer.density:.2f}{trailing}\n")
            current_depth += layer.thickness
        # Deepest halfspace below all sediment layers.
        hs = getattr(lb, 'halfspace', None)
        if hs is not None:
            c_p = getattr(hs, 'sound_speed', fallback_c_p) or fallback_c_p
            c_s = getattr(hs, 'shear_speed', fallback_c_s) or fallback_c_s
            alpha_p = getattr(hs, 'attenuation', fallback_alpha_p) or fallback_alpha_p
            alpha_s = getattr(hs, 'shear_attenuation', fallback_alpha_s) or fallback_alpha_s
            rho = getattr(hs, 'density', fallback_rho) or fallback_rho
        else:
            c_p, c_s, alpha_p, alpha_s, rho = (
                fallback_c_p, fallback_c_s,
                fallback_alpha_p, fallback_alpha_s, fallback_rho,
            )
        f.write(f"{current_depth:.2f} {c_p:.2f} {c_s:.2f} "
                f"{alpha_p:.3f} {alpha_s:.3f} {rho:.2f}{trailing}\n")
    else:
        f.write(f"{water_depth:.2f} {fallback_c_p:.2f} {fallback_c_s:.2f} "
                f"{fallback_alpha_p:.3f} {fallback_alpha_s:.3f} "
                f"{fallback_rho:.2f}{trailing}\n")


def _count_bottom_layers(env: Environment) -> int:
    """Number of sediment layers (not counting halfspace) in env.bottom_layered."""
    if hasattr(env, 'bottom_layered') and env.bottom_layered is not None:
        return len(env.bottom_layered.layers)
    return 0


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


def _receiver_block_lines(
    receiver: Receiver,
    *,
    trailing: str = '',
) -> list:
    """Emit OASES receiver-block text lines.

    OASES supports non-equidistant receiver depths via NR<0: a negative
    receiver count on the first line followed by the explicit depth list on
    the next line(s). See oast.tex:464-493 and oasp.tex:559-585.

    Returns
    -------
    list[str]
        Lines to write, each WITHOUT a trailing newline. Caller adds '\\n'.

    Parameters
    ----------
    receiver : Receiver
        Receiver object providing `.depths`.
    trailing : str
        Extra whitespace-separated tokens to append to the header line
        (e.g. ``' 1'`` for OAST's IR column). An empty string matches OASP.
    """
    depths = np.asarray(receiver.depths, dtype=float)
    n = len(depths)
    z_min = float(depths.min())
    z_max = float(depths.max())

    if n <= 1:
        return [f"{z_min:.2f} {z_max:.2f} {n}{trailing}"]

    diffs = np.diff(depths)
    # A grid is considered equidistant when all spacings match within 1 cm.
    uniform = np.allclose(diffs, diffs[0], atol=1e-2)
    if uniform:
        return [f"{z_min:.2f} {z_max:.2f} {n}{trailing}"]

    # Non-uniform: emit NR = -n, then individual depths on the next line.
    header = f"{z_min:.2f} {z_max:.2f} {-n}{trailing}"
    depth_line = ' '.join(f"{d:.2f}" for d in depths)
    return [header, depth_line]


def write_oast_input(
    filepath: Union[str, Path],
    env: Environment,
    source: Source,
    receiver: Receiver,
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

    # Source and receiver parameters (receiver depth bookkeeping now lives in
    # ``_receiver_block_lines`` which handles equidistant/explicit cases).
    src_depth = float(source.depth[0])
    r_max = float(receiver.ranges.max())

    # Optional parameters
    integration_offset = kwargs.get('integration_offset', 0)
    nw_samples = kwargs.get('nw_samples', -1)  # -1 = automatic
    plot_rmin = kwargs.get('plot_rmin', 0)
    plot_rmax = kwargs.get('plot_rmax', r_max / 1000.0)  # Convert m to km
    volume_attenuation = kwargs.get('volume_attenuation', None)

    # Get reference sound speed for plot axes
    c_ref = float(ssp_data[0, 1])  # Sound speed at surface

    # Options string
    if options is None:
        options = 'N J T'  # Normal stress, complex contour, TL vs range
    options = _inject_volume_attenuation(options, volume_attenuation)

    # Multi-frequency sweep support (B17).
    freqs_arr = np.atleast_1d(source.frequency)
    if len(freqs_arr) > 1:
        freq_min = float(freqs_arr.min())
        freq_max = float(freqs_arr.max())
        nfreq = int(len(freqs_arr))
    else:
        freq_min = freq_max = freq
        nfreq = 1

    with open(filepath, 'w') as f:
        # Block I: Title
        title = env.name if env.name else "OAST Simulation via UACPY"
        f.write(f"{title}\n")

        # Block II: OPTIONS
        f.write(f"{options}\n")

        # Block III: Frequencies
        # FREQ1 FREQ2 NFREQ COFF [VREC]
        # Per unoast31.f:125-133, when the 'd' (lowercase Doppler) option is
        # enabled the block must include a 5th value VREC (source/receiver
        # velocity). Capital 'D' is the depth-averaged-TL flag and does NOT
        # trigger Doppler, so keep the check case-specific.
        opt_tokens_freq = options.split()
        doppler_on = 'd' in opt_tokens_freq
        if doppler_on:
            vrec = kwargs.get('vrec', 0.0)
            offdb = kwargs.get('offdb', integration_offset)
            f.write(f"{freq_min:.1f} {freq_max:.1f} {nfreq} {offdb} {vrec}\n")
        else:
            f.write(f"{freq_min:.1f} {freq_max:.1f} {nfreq} {integration_offset}\n")

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
        n_sed_layers = _count_bottom_layers(env)

        if is_isovelocity:
            # Isovelocity: write only 1 water layer at surface
            n_layers = 3 + n_sed_layers  # vacuum + water + sed_layers + bottom
            f.write(f"{n_layers}\n")

            # Upper halfspace (from env.surface)
            f.write(f"{_format_upper_halfspace(env)}\n")

            # Single water layer
            f.write(f"0.00 {c_values[0]:.2f} 0 0.0 0 1.0 0 0\n")

            # Sediment layers + bottom halfspace (shared helper)
            _emit_bottom_layers(
                f, env, depth,
                c_p, c_s, alpha_p, alpha_s, rho,
                extra_columns=1,
            )
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

            # Sediment layers + bottom halfspace (shared helper)
            _emit_bottom_layers(
                f, env, depth,
                c_p, c_s, alpha_p, alpha_s, rho,
                extra_columns=1,
            )

        # Block V: Sources
        # SD NS DS AN IA FD DA
        # SD = source depth, NS = number of sources (1 for single source)
        f.write(f"{src_depth:.2f} 1 0 0 1 0 0\n")

        # Block VI: Receivers
        # RD1 RD2 NR IR  (NR<0 signals explicit depth list — oast.tex:464-493).
        for line in _receiver_block_lines(receiver, trailing=' 1'):
            f.write(line + '\n')

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

        # Block IX: Transmission loss axes (only for options A, D, T, I, or
        # the ANSPEC / TLDEP plot flags — see oast.tex Table "BLOCK IX").
        # Emitting this block unconditionally would be consumed as noise by
        # the next block when no TL plot is requested.
        opt_tokens = options.split()
        needs_tl_axes = (
            any(tok in opt_tokens for tok in ('A', 'D', 'T', 'I'))
            or 'ANSPEC' in options
            or 'TLDEP' in options
        )
        if needs_tl_axes:
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
    options: Optional[str] = None,
    **kwargs
) -> None:
    """
    Write OASN (OASES Noise, Covariance Matrices and Signal Replicas) input file.

    OASN is NOT a normal-modes solver — per oasn.tex:1 it produces:
    - Noise-field covariance matrices for ambient noise characterisation.
    - Array-response covariance matrices from discrete or continuous sources.
    - Signal replicas on a depth/range grid for matched-field processing.

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

    # Noise/source parameters
    surface_noise_level = kwargs.get('surface_noise_level', 0)
    white_noise_level = kwargs.get('white_noise_level', 0)
    deep_noise_level = kwargs.get('deep_noise_level', 0)
    discrete_sources = kwargs.get('discrete_sources', [])
    n_discrete = len(discrete_sources)

    # Options string
    if options is None:
        options = 'N J'  # Covariance output, complex contour
    volume_attenuation = kwargs.get('volume_attenuation', None)
    options = _inject_volume_attenuation(options, volume_attenuation)

    # Integration parameters
    integration_offset = kwargs.get('integration_offset', 0)

    # Multi-frequency sweep support (B17).
    freqs_arr = np.atleast_1d(source.frequency)
    if len(freqs_arr) > 1:
        freq_min_b = float(freqs_arr.min())
        freq_max_b = float(freqs_arr.max())
        nfreq = int(len(freqs_arr))
    else:
        freq_min_b = freq_max_b = freq
        nfreq = 1

    with open(filepath, 'w') as f:
        # Block I: Title
        title = env.name if env.name else "OASN Simulation via UACPY"
        f.write(f"{title}\n")

        # Block II: OPTIONS
        f.write(f"{options}\n")

        # Block III: Frequencies
        # FREQ1 FREQ2 NFREQ COFF
        f.write(f"{freq_min_b:.1f} {freq_max_b:.1f} {nfreq} {integration_offset}\n")

        # Block IV: Environment
        # NL = number of layers (including halfspaces and all water layers)
        n_water_layers = len(ssp_data)
        n_sed_layers = _count_bottom_layers(env)
        n_layers = 1 + n_water_layers + n_sed_layers + 1
        f.write(f"{n_layers}\n")

        # Upper halfspace (from env.surface)
        f.write(f"{_format_upper_halfspace(env)} 0\n")

        # Water layers from SSP using OASES Airy-layer convention.
        # Per oaseun31.f:160-192, the SSP-gradient signal is:
        #   CC > 0 and CS < 0  →  n^2-linear layer with speed varying from
        #   CC at the top of THIS layer to -CS at the top of the NEXT layer.
        # Negative CC (the previous implementation) instead matches the
        # Cp<0, Cs>=0 branch at oaseun31.f:302, which unconditionally flags
        # the layer TRANSVERSELY ISOTROPIC. OASN then tries to read 5 complex
        # tensor constants + density and dies with "End of file" in inenvi_.
        # This mirrors OAST writer's gradient encoding at lines 340-349.
        for i in range(len(ssp_data)):
            d, c = ssp_data[i]
            if i < len(ssp_data) - 1:
                c_next = float(ssp_data[i + 1, 1])
                cs = -abs(c_next)
            else:
                cs = 0
            f.write(f"{d:.2f} {c:.2f} {cs:.2f} 0.0 0 1.0 0 0\n")

        # Sediment layers + bottom halfspace (shared helper, B15)
        _emit_bottom_layers(
            f, env, depth,
            c_p, c_s, alpha_p, alpha_s, rho,
            extra_columns=1,
        )

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

    # Source parameter (receiver depth bookkeeping handled by
    # ``_receiver_block_lines`` which emits equidistant/explicit as needed).
    src_depth = float(source.depth[0])

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
    volume_attenuation = kwargs.get('volume_attenuation', None)
    options = _inject_volume_attenuation(options, volume_attenuation)

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
        n_sed_layers = _count_bottom_layers(env)
        n_layers = 1 + n_water_layers + n_sed_layers + 1
        f.write(f"{n_layers}\n")

        # Upper halfspace (from env.surface)
        f.write(f"{_format_upper_halfspace(env)} 0\n")

        # Water layers from SSP
        for i in range(len(ssp_data)):
            d, c = ssp_data[i]
            f.write(f"{d:.2f} {c:.2f} 0 0.0 0 1.0 0 0 0\n")

        # Sediment layers + bottom halfspace (shared helper, B15)
        _emit_bottom_layers(
            f, env, depth,
            c_p, c_s, alpha_p, alpha_s, rho,
            extra_columns=2,
        )

        # Block V: Sources
        # SD NS DS AN IA FD DA
        # For single source: SD 1 0 0 1 0 0
        f.write(f"{src_depth:.2f} 1 0 0 1 0 0\n")

        # Block VI: Receiver depths (NRD<0 signals explicit depth list —
        # oasp.tex:559-585).
        for line in _receiver_block_lines(receiver):
            f.write(line + '\n')

        # Block VII: Wavenumber sampling
        # CMIN CMAX
        f.write(f"{cmin:.1f} {cmax:.1e}\n")

        # NW IC1 IC2 IF
        # NW=-1 for automatic sampling — IC1/IC2 have no effect (oasp.tex:677).
        # When NW > 0, IC2 must be set to NW so the Hankel transform is *not*
        # prematurely zeroed. The old hard-coded '1 1' truncated everything
        # past sample #1 whenever the user specified NW explicitly.
        # IF = frequency sample increment for kernels (0 disables plotting).
        if nw_samples is None or nw_samples <= 0:
            ic1, ic2 = 1, 1
        else:
            ic1, ic2 = 1, int(nw_samples)
        f.write(f"{nw_samples} {ic1} {ic2} 40\n")

        # Block VIII: Frequency and range sampling
        # NT FR1 FR2 DT R1 DR NR
        f.write(f"{n_time} {freq_min:.1f} {freq_max:.1f} {dt:.6f} {r1_km:.3f} {dr_km:.3f} {nr}\n")


def write_oasr_input(
    filepath: Union[str, Path],
    env: Environment,
    source: Source,
    receiver: Receiver,
    options: Optional[str] = None,
    interface_roughness: Optional[list] = None,
    angles: Optional[np.ndarray] = None,
    angle_type: str = 'grazing',
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
    angles : ndarray, optional
        Angles (degrees). If provided, overrides angle_min/angle_max/n_angles
        in ``kwargs``. Interpreted per ``angle_type`` (see below).
    angle_type : str, optional
        'grazing' (default) or 'incidence'. OASES expects grazing angles;
        when ``angle_type='incidence'``, angles are converted via
        ``grazing = 90 - incidence`` before being written.
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

    # Sound speed profile - for OASR we only need a single representative
    # water sound speed: OASR is a *local* interface reflection solver, the
    # source is placed 1 mm above the top of layer 2 (see unoasr21.f), and
    # layer 1 is treated as the upper halfspace carrying the incident wave.
    # A full stratified water column has no meaning here — OASR only sees the
    # (homogeneous) medium immediately above the reflecting interface.
    ssp_data = env.ssp_data
    if ssp_data is None or len(ssp_data) == 0:
        c_water = env.sound_speed if env.sound_speed else DEFAULT_SOUND_SPEED
    else:
        # Sound speed right above the seabed interface.
        c_water = float(ssp_data[-1, 1])

    # Multi-frequency support (B17) — OASR sweep parameters.
    freqs_arr = np.atleast_1d(source.frequency)
    if len(freqs_arr) > 1:
        freq_min = float(freqs_arr.min())
        freq_max = float(freqs_arr.max())
        n_frequencies = int(len(freqs_arr))
    else:
        freq_min = kwargs.get('freq_min', freq)
        freq_max = kwargs.get('freq_max', freq)
        n_frequencies = kwargs.get('n_frequencies', 1)
    freq_out_inc = kwargs.get('freq_output_increment', max(1, n_frequencies // 10))

    # Angle parameters. OASES natively uses grazing angles; if the caller
    # requested 'incidence', convert to grazing via 90 - incidence.
    if angle_type not in ('grazing', 'incidence'):
        raise ConfigurationError(
            f"OASR: angle_type must be 'grazing' or 'incidence', got {angle_type!r}"
        )
    if angles is not None:
        angles_arr = np.atleast_1d(np.asarray(angles, dtype=float))
        if angle_type == 'incidence':
            angles_arr = 90.0 - angles_arr
        angle_min = float(angles_arr.min())
        angle_max = float(angles_arr.max())
        n_angles = int(len(angles_arr))
    else:
        angle_min = kwargs.get('angle_min', 0.0)
        angle_max = kwargs.get('angle_max', 90.0)
        if angle_type == 'incidence':
            # Convert scalar bounds as well so the user-facing axis is honored.
            angle_min, angle_max = 90.0 - angle_max, 90.0 - angle_min
        n_angles = kwargs.get('n_angles', 181)
    angle_out_inc = kwargs.get('angle_output_increment', max(1, n_angles // 10))

    # Options string
    if options is None:
        options = 'N T'  # Normal (P-P), Table output
    volume_attenuation = kwargs.get('volume_attenuation', None)
    options = _inject_volume_attenuation(options, volume_attenuation)

    # Interface roughness (RG / CL / M) per interface (B13 #6).
    # Indexed starting at 0 = upper-halfspace/surface interface; None -> no roughness.
    if interface_roughness is None:
        interface_roughness = []

    def _roughness_tail(i):
        """Return roughness-suffix string for interface index ``i``.

        OASES convention (oases_gen.tex): RG > 0 → RMS roughness only;
        RG < 0 → |RG| plus CL and M on same line (Goff-Jordan power spectrum).
        """
        if i < 0 or i >= len(interface_roughness):
            return " 0"
        spec = interface_roughness[i]
        if spec is None:
            return " 0"
        if isinstance(spec, (int, float)):
            return f" {float(spec):.4f}"
        # dict or tuple
        if isinstance(spec, dict):
            rg = spec.get('RG', spec.get('roughness', 0.0))
            cl = spec.get('CL', spec.get('correlation_length', None))
            m = spec.get('M', spec.get('spectral_exponent', None))
        else:  # assume tuple/list
            rg = spec[0] if len(spec) > 0 else 0.0
            cl = spec[1] if len(spec) > 1 else None
            m = spec[2] if len(spec) > 2 else None
        if cl is None or m is None:
            return f" {float(rg):.4f}"
        # Flag negative RG to signal CL + M follow.
        return f" {-abs(float(rg)):.4f} {float(cl):.4f} {float(m):.4f}"

    with open(filepath, 'w') as f:
        # Block I: Title
        title = env.name if env.name else "OASR Simulation via UACPY"
        f.write(f"{title}\n")

        # Block II: OPTIONS
        f.write(f"{options}\n")

        # Block III: Environment (B13)
        # OASR convention: layer 1 IS the upper halfspace in which the source
        # sits (placed 1 mm above layer 2's top). Reflection is computed at the
        # interface between layer 1 and layer 2. We therefore emit the water
        # column as the upper halfspace — NOT a separate vacuum layer above a
        # water layer, which would place the source in the vacuum and make the
        # solver compute a vacuum/water reflection (empty .rco output).
        #
        # Reference: oasr.tex section "Output Files" (saffipr1.dat example) —
        # NL = 3 with `0 1500 0 0 0 1 0` as layer 1 (water), followed by
        # sediment + halfspace.
        n_sed_layers = _count_bottom_layers(env)
        n_layers = 1 + n_sed_layers + 1  # water-halfspace + sediment + bottom
        f.write(f"{n_layers}\n")

        # Layer 1: water as upper halfspace (D is dummy for layer 1).
        # AC=0 triggers Skretting-Leroy empirical water attenuation inside OASR.
        f.write(f"0.00 {c_water:.2f} 0 0.0 0 1.0{_roughness_tail(0)}\n")

        # Sediment stack + bottom halfspace (first interface at z = env.depth).
        iface_idx = 1
        if hasattr(env, 'bottom_layered') and env.bottom_layered is not None:
            lb = env.bottom_layered
            current_depth = depth
            for layer in lb.layers:
                layer_as = getattr(layer, 'shear_attenuation', 0.0) or 0.0
                f.write(
                    f"{current_depth:.2f} {layer.sound_speed:.2f} "
                    f"{layer.shear_speed:.2f} {layer.attenuation:.3f} "
                    f"{layer_as:.3f} {layer.density:.2f}"
                    f"{_roughness_tail(iface_idx)}\n"
                )
                current_depth += layer.thickness
                iface_idx += 1
            hs = getattr(lb, 'halfspace', None)
            if hs is not None:
                hs_cp = getattr(hs, 'sound_speed', c_p) or c_p
                hs_cs = getattr(hs, 'shear_speed', c_s) or c_s
                hs_ap = getattr(hs, 'attenuation', alpha_p) or alpha_p
                hs_as = getattr(hs, 'shear_attenuation', alpha_s) or alpha_s
                hs_ro = getattr(hs, 'density', rho) or rho
            else:
                hs_cp, hs_cs, hs_ap, hs_as, hs_ro = c_p, c_s, alpha_p, alpha_s, rho
            f.write(
                f"{current_depth:.2f} {hs_cp:.2f} {hs_cs:.2f} "
                f"{hs_ap:.3f} {hs_as:.3f} {hs_ro:.2f}"
                f"{_roughness_tail(iface_idx)}\n"
            )
        else:
            f.write(
                f"{depth:.2f} {c_p:.2f} {c_s:.2f} "
                f"{alpha_p:.3f} {alpha_s:.3f} {rho:.2f}"
                f"{_roughness_tail(iface_idx)}\n"
            )

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
            c_min = min(c_water, c_p) * 0.95
            c_max = max(c_water, c_p) * 1.05
            f.write(f"{c_min:.1f} {c_max:.1f} 12 100\n")
            f.write(f"0 {depth:.1f} 12 {depth/10:.1f}\n")
