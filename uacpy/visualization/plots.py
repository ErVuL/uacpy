"""
Plotting functions for underwater acoustics visualization
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional, Tuple, Union, Dict

from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.field import Field
from uacpy.core.constants import PRESSURE_FLOOR
from uacpy.visualization.style import (
    get_cmap_for_field,
    format_axes_professional,
    create_professional_colorbar,
    get_model_color,
    COLORMAPS,
)

# Z-order constants for consistent layering (Acoustic Toolbox standard)
ZORDER_GRID = 0
ZORDER_FIELD = 1
ZORDER_SEDIMENT = 2
ZORDER_RAYS = 2.5  # Between field and bathymetry
ZORDER_BATHYMETRY = 3
ZORDER_SURFACE = 4
ZORDER_RECEIVERS = 5
ZORDER_SOURCE = 6
ZORDER_ANNOTATIONS = 7


def _select_2d_slice(field: Field, frequency: Optional[float] = None) -> np.ndarray:
    """Return a 2-D (n_depths, n_ranges) view of a Field.

    For 2-D data (``ndim == 2``) the array is returned unchanged.
    For 3-D broadband data with shape ``(n_depths, n_freqs, n_ranges)``
    (per Field class contract), a frequency slice is picked:

    * if ``frequency`` is None, the nearest-to-mid-band frequency is used;
    * otherwise the nearest frequency in ``field.frequencies``.

    Parameters
    ----------
    field : Field
        Field whose ``data`` attribute is inspected.
    frequency : float, optional
        Target frequency in Hz.

    Returns
    -------
    np.ndarray
        A 2-D view of the field's data at the chosen frequency.
    """
    if field.data.ndim == 2:
        return field.data
    if field.data.ndim == 3:
        freqs = np.asarray(field.frequencies)
        if frequency is None:
            # Default: middle of the frequency axis
            f_idx = len(freqs) // 2 if len(freqs) else 0
        else:
            f_idx = int(np.argmin(np.abs(freqs - frequency))) if len(freqs) else 0
        return field.data[:, f_idx, :]
    raise ValueError(
        f"Unsupported Field.data ndim={field.data.ndim}; expected 2 or 3."
    )


def plot_transmission_loss(
    field: Field,
    env: Optional[Environment] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 7),
    show_bathymetry: bool = True,
    show_colorbar: bool = True,
    contours: Optional[list] = None,
    ax: Optional[Axes] = None,
    tl_span: float = 50.0,
    tl_round: int = 10,
    seafloor_alpha: float = 1.0,
    frequency: Optional[float] = None,
):
    """
    Plot transmission loss field

    Parameters
    ----------
    field : Field
        Transmission loss field
    env : Environment, optional
        Environment for bathymetry overlay
    vmin, vmax : float, optional
        Color scale limits in dB. If None, auto-computed using Acoustic Toolbox
        method: vmax = median + 0.75*std, rounded to nearest tl_round dB;
        vmin = vmax - tl_span.
    cmap : str, optional
        Colormap name. Default is 'jet_r' (Acoustic Toolbox standard).
    figsize : tuple, optional
        Figure size (width, height). Default is (12, 7).
    show_bathymetry : bool, optional
        Show bathymetry line. Default is True.
    show_colorbar : bool, optional
        Show colorbar. Set to False for subplot layouts. Default is True.
    contours : list of float, optional
        TL values (in dB) to overlay as contour lines. For example,
        [70, 80, 90] will draw contours at 70, 80, and 90 dB.
        Contour lines are labeled automatically. Default is None (no contours).
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.
    tl_span : float, optional
        TL span in dB when auto-computing limits. Default is 50 dB (standard).
    tl_round : int, optional
        Round vmax to nearest multiple of this value. Default is 10 dB.
    seafloor_alpha : float, optional
        Transparency of seafloor/ground fill (0.0 = fully transparent, 1.0 = opaque).
        Default is 1.0 (no transparency).
    frequency : float, optional
        For broadband fields (``field.data.ndim == 3`` with shape
        ``(n_depths, n_freqs, n_ranges)``), the frequency at which to
        slice. Defaults to the nearest-to-mid-band frequency.

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes
    cbar : Colorbar or None
        Colorbar object if show_colorbar=True, None otherwise

    Examples
    --------
    >>> fig, ax, cbar = plot_transmission_loss(result, env)
    >>> plt.show()

    >>> # With contour lines at 70, 80, 90 dB
    >>> fig, ax, cbar = plot_transmission_loss(result, env, contours=[70, 80, 90])
    >>> plt.show()

    >>> # For subplots, disable colorbar
    >>> fig, axes = plt.subplots(1, 2)
    >>> fig1, ax1, _ = plot_transmission_loss(result1, env, ax=axes[0], show_colorbar=False)
    >>> fig2, ax2, _ = plot_transmission_loss(result2, env, ax=axes[1], show_colorbar=False)
    """
    # Handle None (failed model)
    if field is None:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        ax.text(0.5, 0.5, 'Model Failed', ha='center', va='center',
               transform=ax.transAxes, fontsize=14, color='red')
        ax.axis('off')
        return fig, ax, None

    if field.field_type != 'tl':
        field = field.to_db()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Use professional colormap
    if cmap is None:
        cmap = get_cmap_for_field('tl')

    # Convert ranges to km for display
    ranges_km = field.ranges / 1000.0
    depths = field.depths

    # Select 2-D slice (handles broadband 3-D arrays)
    data2d = _select_2d_slice(field, frequency=frequency)

    # Create meshgrid
    R, Z = np.meshgrid(ranges_km, depths)

    # Auto-scale using Acoustic Toolbox method (median + 0.75*std, rounded)
    if vmin is None or vmax is None:
        valid_data = data2d[np.isfinite(data2d)]

        if len(valid_data) > 0:
            # Acoustic Toolbox standard: median + 0.75 * std
            tl_median = np.median(valid_data)
            tl_std = np.std(valid_data)

            if vmax is None:
                # Max = median + 0.75 * std, rounded to nearest tl_round dB
                vmax = tl_median + 0.75 * tl_std
                vmax = tl_round * np.round(vmax / tl_round)

            if vmin is None:
                # Min = max - span (standard 50 dB)
                vmin = vmax - tl_span
        else:
            # Fallback if no valid data
            vmin, vmax = 30.0, 80.0

    # Ensure vmin < vmax
    if vmin >= vmax:
        vmin = vmax - tl_span

    im = ax.pcolormesh(
        R, Z, data2d,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading='auto',
        zorder=1  # Ensure TL is below bathymetry
    )

    # Invert y-axis (depth increases downward)
    ax.invert_yaxis()

    # Set proper y-axis limits to show full depth range
    max_depth = depths.max()
    if env is not None:
        if len(env.bathymetry) > 1:
            max_depth = max(max_depth, env.bathymetry[:, 1].max())
        else:
            max_depth = max(max_depth, env.depth)
    ax.set_ylim([max_depth * 1.05, 0])  # Add 5% padding below bottom

    # Professional formatting
    format_axes_professional(
        ax,
        title='Transmission Loss',
        xlabel='Range (km)',
        ylabel='Depth (m)'
    )

    # Conditionally create colorbar
    cbar = None
    if show_colorbar:
        # Check if data exceeds colorbar limits (for extend parameter)
        data_min = np.nanmin(data2d)
        data_max = np.nanmax(data2d)
        extend = 'neither'
        if data_min < vmin and data_max > vmax:
            extend = 'both'
        elif data_min < vmin:
            extend = 'min'
        elif data_max > vmax:
            extend = 'max'

        # Professional colorbar with extensions
        cbar = fig.colorbar(im, ax=ax, label='TL (dB)', fraction=0.046, pad=0.02, extend=extend)
        cbar.ax.tick_params(labelsize=10)
        cbar.outline.set_linewidth(1.0)

    # Bathymetry overlay - MUST BE BEFORE CONTOURS to mask data below seafloor
    if show_bathymetry and env is not None:
        if len(env.bathymetry) > 1:
            bathy_ranges = env.bathymetry[:, 0] / 1000.0
            bathy_depths = env.bathymetry[:, 1]
            # Seafloor fill MUST have high zorder to mask everything below it
            ax.fill_between(
                bathy_ranges, bathy_depths, max_depth * 1.05,
                color='sienna', alpha=seafloor_alpha, zorder=ZORDER_SEDIMENT + 5, edgecolor='none'
            )
            # Bathymetry line on top of fill
            ax.plot(bathy_ranges, bathy_depths, 'k-', linewidth=2.5, label='Bathymetry',
                   zorder=ZORDER_SEDIMENT + 6)
        else:
            # Flat bottom - fill MUST mask everything below
            ax.fill_between(
                ranges_km, env.depth, max_depth * 1.05,
                color='sienna', alpha=seafloor_alpha, zorder=ZORDER_SEDIMENT + 5, edgecolor='none'
            )
            # Bathymetry line on top
            ax.axhline(env.depth, color='k', linewidth=2.5, linestyle='--',
                      zorder=ZORDER_SEDIMENT + 6)

    # Contour overlay (AT standard: black lines with labels)
    # Contours drawn BEFORE bathymetry so seafloor masks them
    if contours is not None and len(contours) > 0:
        # Create contour lines
        CS = ax.contour(
            R, Z, data2d,
            levels=contours,
            colors='black',
            linewidths=1.5,
            linestyles='solid',
            alpha=0.7,
            zorder=ZORDER_FIELD + 0.5  # Just above TL field, below seafloor
        )

        # Add contour labels (inline, with background)
        ax.clabel(CS, inline=True, fontsize=9, fmt='%g dB',
                 inline_spacing=10, use_clabeltext=True)

    ax.grid(True, alpha=0.3, zorder=0)

    return fig, ax, cbar


def plot_rays(
    field: Field,
    env: Optional[Environment] = None,
    max_rays: Optional[int] = None,
    figsize: Tuple[float, float] = (12, 6),
    ax: Optional[Axes] = None,
    color_by_bounces: bool = True,
    ray_colors: Optional[Dict[str, str]] = None,
    seafloor_alpha: float = 1.0,
) -> Tuple[Figure, Axes]:
    """
    Plot ray paths

    Parameters
    ----------
    field : Field
        Ray field from ray tracing model
    env : Environment, optional
        Environment for bathymetry
    max_rays : int, optional
        Maximum number of rays to plot. If None, plots all.
    figsize : tuple, optional
        Figure size. Default is (12, 6).
    ax : Axes, optional
        Matplotlib axes
    color_by_bounces : bool, optional
        Color rays by bounce type (Acoustic Toolbox standard). Default is True.
        Red=direct, Green=surface, Blue=bottom, Black=both boundaries.
    ray_colors : dict, optional
        Custom ray colors. Keys: 'direct', 'surface', 'bottom', 'both'.
    seafloor_alpha : float, optional
        Transparency of seafloor/ground fill (0.0 = fully transparent, 1.0 = opaque).
        Default is 1.0 (no transparency).

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes

    Examples
    --------
    >>> fig, ax = plot_rays(result, env, max_rays=50)
    >>> plt.show()

    Notes
    -----
    Ray coloring follows Acoustic Toolbox convention:
    - Red: Direct path (no bounces)
    - Green: Surface reflections only
    - Blue: Bottom reflections only
    - Black: Both boundary reflections
    """
    if field.field_type not in ['rays', 'eigenrays']:
        raise ValueError("plot_rays requires ray-type field")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    rays = field.extract_rays()

    if not rays:
        warnings.warn("No ray data found in field", UserWarning, stacklevel=2)
        return fig, ax

    # Plot rays
    n_rays = len(rays)
    if max_rays is not None and n_rays > max_rays:
        # Sample rays
        indices = np.linspace(0, n_rays - 1, max_rays, dtype=int)
        rays_to_plot = [rays[i] for i in indices]
    else:
        rays_to_plot = rays

    # Default ray colors (Acoustic Toolbox standard)
    if ray_colors is None:
        ray_colors = {
            'direct': 'r',    # Red: no bounces
            'surface': 'g',   # Green: surface only
            'bottom': 'b',    # Blue: bottom only
            'both': 'k'       # Black: both boundaries
        }

    # Plot rays with color-coding
    for ray in rays_to_plot:
        if 'r' in ray and 'z' in ray:
            r_km = ray['r'] / 1000.0
            z = ray['z']

            # Determine color based on bounce type
            if color_by_bounces:
                n_top = ray.get('num_top_bounces', 0)
                n_bot = ray.get('num_bottom_bounces', 0)

                if n_top >= 1 and n_bot >= 1:
                    color = ray_colors['both']
                elif n_bot >= 1:
                    color = ray_colors['bottom']
                elif n_top >= 1:
                    color = ray_colors['surface']
                else:
                    color = ray_colors['direct']
            else:
                color = 'b'  # fallback to blue

            ax.plot(r_km, z, color=color, linewidth=0.5, alpha=0.7, zorder=ZORDER_RAYS)

    # Invert y-axis first
    ax.invert_yaxis()

    # Determine depth range for proper axis limits
    max_depth = 0
    if env is not None:
        if len(env.bathymetry) > 1:
            max_depth = env.bathymetry[:, 1].max()
        else:
            max_depth = env.depth

    # Find maximum ray depth
    for ray in rays_to_plot:
        if 'z' in ray:
            max_depth = max(max_depth, np.max(ray['z']))

    # Set ylim to show full depth
    if max_depth > 0:
        ax.set_ylim([max_depth * 1.1, -max_depth * 0.05])  # Add padding

    # Environment
    if env is not None:
        # Surface
        ax.axhline(0, color='cyan', linewidth=2, linestyle='-', label='Surface', zorder=3)

        # Bathymetry
        if len(env.bathymetry) > 1:
            bathy_r = env.bathymetry[:, 0] / 1000.0
            bathy_z = env.bathymetry[:, 1]
            ax.plot(bathy_r, bathy_z, 'k-', linewidth=2.5, label='Bottom', zorder=3)
            ylims = ax.get_ylim()
            ax.fill_between(
                bathy_r, bathy_z, ylims[0],  # Fill to bottom of plot
                color='sienna', alpha=seafloor_alpha, zorder=1, edgecolor='none'
            )
        else:
            ax.axhline(env.depth, color='k', linewidth=2.5, linestyle='-', label='Bottom', zorder=3)
            xlims = ax.get_xlim()
            ylims = ax.get_ylim()
            ax.fill_between(
                xlims, env.depth, ylims[0],  # Fill to bottom of plot
                color='sienna', alpha=seafloor_alpha, zorder=1, edgecolor='none'
            )

    # Labels
    ax.set_xlabel('Range (km)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title(f'Ray Diagram ({len(rays_to_plot)} rays)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=ZORDER_GRID)

    # Add legend with ray color coding if enabled
    if color_by_bounces and env is not None:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=ray_colors['direct'], label='Direct (no bounces)'),
            Patch(facecolor=ray_colors['surface'], label='Surface reflected'),
            Patch(facecolor=ray_colors['bottom'], label='Bottom reflected'),
            Patch(facecolor=ray_colors['both'], label='Both boundaries'),
        ]
        # Add environment elements
        if hasattr(ax, 'lines'):  # Check if surface/bottom lines were plotted
            ax.legend(handles=legend_elements, loc='best', fontsize=9)
    else:
        # Only call ax.legend() if at least one artist has a label,
        # otherwise matplotlib emits a "No artists with labels found"
        # UserWarning.
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()

    return fig, ax


def plot_ssp(
    env: Environment,
    figsize: Tuple[float, float] = (6, 8),
    ax: Optional[Axes] = None,
    show_shear: bool = True,
    show_data_points: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Plot sound speed profile

    Parameters
    ----------
    env : Environment
        Environment with SSP data
    figsize : tuple, optional
        Figure size. Default is (6, 8).
    ax : Axes, optional
        Matplotlib axes
    show_shear : bool, optional
        Plot shear wave speed if available (red line). Default is True.
    show_data_points : bool, optional
        Show data points as markers (black circles 'ko'). Default is True.

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes

    Examples
    --------
    >>> fig, ax = plot_ssp(env)
    >>> plt.show()

    Notes
    -----
    Follows Acoustic Toolbox plotting convention:
    - Compression wave speed: blue line with data points
    - Shear wave speed (if present): red line with data points
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot SSP
    depths = env.ssp_data[:, 0]
    sound_speeds = env.ssp_data[:, 1]

    # Plot compression wave speed
    if show_data_points:
        ax.plot(sound_speeds, depths, 'ko', markersize=6, label='Data points', zorder=5)

    # Smooth interpolated line
    depths_interp = np.linspace(depths[0], depths[-1], 200)
    c_interp = np.interp(depths_interp, depths, sound_speeds)
    ax.plot(c_interp, depths_interp, 'b-', linewidth=2, label='Compression', zorder=3)

    # Plot shear wave speed if available (AT standard: red line)
    if show_shear and hasattr(env, 'shear_speed') and env.shear_speed is not None:
        if isinstance(env.shear_speed, np.ndarray) and np.any(env.shear_speed > 0):
            # Filter out zero values
            nonzero_mask = env.shear_speed > 0
            if show_data_points:
                ax.plot(env.shear_speed[nonzero_mask], depths[nonzero_mask],
                       'ko', markersize=6, zorder=5)

            # Interpolate shear wave speed
            s_interp = np.interp(depths_interp, depths, env.shear_speed)
            ax.plot(s_interp, depths_interp, 'r-', linewidth=2, label='Shear', zorder=3)

    # Invert y-axis
    ax.invert_yaxis()

    # Labels
    ax.set_xlabel('Sound Speed (m/s)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title(f'Sound Speed Profile\n{env.name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    return fig, ax


def plot_bathymetry(
    env: Environment,
    figsize: Tuple[float, float] = (12, 4),
    ax: Optional[Axes] = None,
    seafloor_alpha: float = 1.0,
) -> Tuple[Figure, Axes]:
    """
    Plot bathymetry profile

    Parameters
    ----------
    env : Environment
        Environment with bathymetry data
    figsize : tuple, optional
        Figure size. Default is (12, 4).
    ax : Axes, optional
        Matplotlib axes
    seafloor_alpha : float, optional
        Transparency of seafloor/ground fill (0.0 = fully transparent, 1.0 = opaque).
        Default is 1.0 (no transparency).

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot bathymetry
    ranges_km = env.bathymetry[:, 0] / 1000.0
    depths = env.bathymetry[:, 1]

    ax.plot(ranges_km, depths, 'k-', linewidth=2.5)
    ax.fill_between(ranges_km, depths, depths.max() * 1.2, color='sienna', alpha=seafloor_alpha, edgecolor='none')

    # Invert y-axis
    ax.invert_yaxis()

    # Labels
    ax.set_xlabel('Range (km)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title(f'Bathymetry\n{env.name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_arrivals(
    field: Field,
    figsize: Tuple[float, float] = (10, 6),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot arrival pattern (time vs amplitude)

    Parameters
    ----------
    field : Field
        Arrivals field
    figsize : tuple, optional
        Figure size. Default is (10, 6).
    ax : Axes, optional
        Matplotlib axes

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes
    """
    if field.field_type != 'arrivals':
        raise ValueError("plot_arrivals requires arrivals-type field")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    arrivals = field.extract_arrivals()

    if 'delays' in arrivals and 'amplitudes' in arrivals:
        delays = arrivals['delays']
        amplitudes = arrivals['amplitudes']

        # Plot as stem plot (use_line_collection removed from mpl>=3.9)
        ax.stem(delays, amplitudes, basefmt=' ')

        ax.set_xlabel('Travel Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title('Arrival Pattern', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        warnings.warn("No arrival data found", UserWarning, stacklevel=2)

    return fig, ax


def plot_environment(
    env: Environment,
    source: Optional[Source] = None,
    receiver: Optional[Receiver] = None,
    figsize: Tuple[float, float] = (14, 8),
    seafloor_alpha: float = 1.0,
) -> Tuple[Figure, Axes]:
    """
    Plot complete environment overview

    Parameters
    ----------
    env : Environment
        Environment to plot
    source : Source, optional
        Source to overlay
    receiver : Receiver, optional
        Receivers to overlay
    figsize : tuple, optional
        Figure size. Default is (14, 8).
    seafloor_alpha : float, optional
        Transparency of seafloor/ground fill (0.0 = fully transparent, 1.0 = opaque).
        Default is 1.0 (no transparency).

    Returns
    -------
    fig : Figure
        Matplotlib figure
    axes : array of Axes
        Matplotlib axes array

    Examples
    --------
    >>> fig, axes = plot_environment(env, source, receiver)
    >>> plt.show()
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # SSP
    ax_ssp = fig.add_subplot(gs[:, 0])
    plot_ssp(env, ax=ax_ssp)

    # Bathymetry
    ax_bathy = fig.add_subplot(gs[0, 1])
    plot_bathymetry(env, ax=ax_bathy)

    # Environment overview
    ax_env = fig.add_subplot(gs[1, 1])

    # Determine max depth for proper plotting
    if len(env.bathymetry) > 1:
        max_depth = env.bathymetry[:, 1].max()
    else:
        max_depth = env.depth

    # Plot bathymetry
    if len(env.bathymetry) > 1:
        ranges_km = env.bathymetry[:, 0] / 1000.0
        depths = env.bathymetry[:, 1]
        ax_env.plot(ranges_km, depths, 'k-', linewidth=2.5, zorder=3)
        ax_env.fill_between(
            ranges_km, depths, max_depth * 1.15,
            color='sienna', alpha=seafloor_alpha, zorder=1, edgecolor='none'
        )
    else:
        # Flat bottom: draw a horizontal line across the full axes
        ax_env.axhline(env.depth, color='k', linewidth=2.5, zorder=3, label='Bottom')
        # Fill is applied below, after xlim is established by source/receiver.

    # Plot source
    if source is not None:
        for sd in source.depth:
            ax_env.plot(0, sd, 'r*', markersize=18, label='Source', zorder=5, markeredgecolor='darkred', markeredgewidth=1.5)

    # Plot receivers
    if receiver is not None:
        r_km = receiver.ranges / 1000.0
        if receiver.receiver_type == 'grid':
            # For grid receivers, plot subset for visualization
            R_km, D = np.meshgrid(r_km, receiver.depths)
            # Plot subset (every 5th point to avoid clutter)
            ax_env.scatter(R_km[::5, ::5].flatten(), D[::5, ::5].flatten(),
                          c='b', marker='v', s=20, alpha=0.5, label='Receivers', zorder=4)
        else:
            # For line receivers, plot all points
            ax_env.scatter(r_km, receiver.depths, c='b', marker='v',
                          s=30, alpha=0.6, label='Receivers', zorder=4)

    # Now add flat bottom fill if needed (after xlim is established)
    if len(env.bathymetry) == 1:
        xlims = ax_env.get_xlim()
        ax_env.fill_between(
            xlims, env.depth, max_depth * 1.15,
            color='sienna', alpha=seafloor_alpha, zorder=1, edgecolor='none'
        )

    ax_env.invert_yaxis()
    # Set proper y-limits
    ax_env.set_ylim([max_depth * 1.15, -max_depth * 0.05])
    ax_env.set_xlabel('Range (km)', fontsize=12)
    ax_env.set_ylabel('Depth (m)', fontsize=12)
    ax_env.set_title('Environment Overview', fontsize=12, fontweight='bold')
    ax_env.grid(True, alpha=0.3, zorder=0)
    ax_env.legend()

    fig.suptitle(f'Environment: {env.name}', fontsize=16, fontweight='bold')

    return fig, fig.axes


def plot_modes(
    modes: Field,
    n_modes: Optional[int] = None,
    figsize: Tuple[float, float] = (14, 6),
    show_imaginary: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Plot normal mode shapes and wavenumber spectrum

    Parameters
    ----------
    modes : Field
        Mode Field object from compute_modes()
    n_modes : int, optional
        Number of modes to plot. If None, plots first 6 modes.
    figsize : tuple, optional
        Figure size. Default is (14, 6).

    Returns
    -------
    fig : Figure
        Matplotlib figure
    axes : tuple of Axes
        Matplotlib axes (mode shapes, wavenumber spectrum)

    Examples
    --------
    >>> modes = kraken.compute_modes(env, source)
    >>> fig, axes = plot_modes(modes)
    >>> plt.show()

    >>> # Or even simpler with Field.plot()
    >>> modes = kraken.compute_modes(env, source)
    >>> fig, axes = modes.plot(n_modes=10)
    >>> plt.show()
    """
    if modes.field_type != 'modes':
        raise ValueError(f"Expected modes field, got {modes.field_type}")

    # Extract mode data from Field
    phi = modes.metadata.get('phi', modes.data)
    z = modes.metadata.get('z', modes.depths)
    k = modes.metadata['k']
    M = len(k)
    freq = modes.metadata.get('frequency', modes.frequencies[0] if len(modes.frequencies) > 0 else 0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    if n_modes is None:
        n_modes = min(6, M)

    # Plot mode shapes
    for i in range(n_modes):
        # Normalize mode shape for visualization
        phi_real = phi[:, i].real
        peak = np.max(np.abs(phi_real))
        if peak == 0:
            peak = 1.0
        phi_norm = phi_real / peak
        ax1.plot(phi_norm, z, linewidth=2, label=f'Mode {i+1}')

        # Overlay imaginary part when requested and non-trivial
        if show_imaginary:
            phi_imag = phi[:, i].imag
            if np.any(np.abs(phi_imag) > 1e-10 * peak):
                phi_imag_norm = phi_imag / peak
                ax1.plot(phi_imag_norm, z, linewidth=1.2, linestyle='--',
                         alpha=0.7, label=f'Mode {i+1} (imag)')

    ax1.set_xlabel('Normalized Amplitude', fontsize=12)
    ax1.set_ylabel('Depth (m)', fontsize=12)
    ax1.set_title(f'First {n_modes} Mode Shapes\nf = {freq:.1f} Hz',
                  fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    ax1.axvline(0, color='k', linewidth=0.5, linestyle='--')

    # Plot wavenumber spectrum
    k_real = np.real(k)
    k_imag = np.abs(np.imag(k))
    phase_speed = 2 * np.pi * freq / k_real

    # Use different colors for different attenuation levels
    colors = plt.cm.viridis(np.linspace(0, 1, M))

    for i in range(M):
        ax2.scatter(phase_speed[i], k_real[i], s=150, c=[colors[i]],
                   edgecolors='black', linewidths=1.5, alpha=0.8,
                   label=f'Mode {i+1}' if i < 6 else None, zorder=3)

    ax2.set_xlabel('Phase Speed (m/s)', fontsize=12)
    ax2.set_ylabel('Wavenumber k (1/m)', fontsize=12)
    ax2.set_title(f'Mode Wavenumber Spectrum\n{M} modes',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    if M <= 6:
        ax2.legend(loc='best', fontsize=9)

    plt.tight_layout()

    return fig, (ax1, ax2)


def plot_mode_functions(
    modes: Field,
    mode_indices: Optional[list] = None,
    figsize: Tuple[float, float] = (10, 8),
) -> Tuple[Figure, Axes]:
    """
    Plot individual mode functions with detailed information

    Parameters
    ----------
    modes : Field
        Mode Field object from compute_modes()
    mode_indices : list of int, optional
        Specific mode indices to plot. If None, plots first 4.
    figsize : tuple, optional
        Figure size. Default is (10, 8).

    Returns
    -------
    fig : Figure
        Matplotlib figure
    axes : array of Axes
        Matplotlib axes array

    Examples
    --------
    >>> modes = kraken.compute_modes(env, source)
    >>> fig, axes = plot_mode_functions(modes, mode_indices=[0, 1, 2])
    >>> plt.show()
    """
    if modes.field_type != 'modes':
        raise ValueError(f"Expected modes field, got {modes.field_type}")

    # Extract mode data from Field
    phi = modes.metadata.get('phi', modes.data)
    z = modes.metadata.get('z', modes.depths)
    k = modes.metadata['k']
    M = len(k)
    freq = modes.metadata.get('frequency', modes.frequencies[0] if len(modes.frequencies) > 0 else 0.0)

    if mode_indices is None:
        mode_indices = list(range(min(4, M)))

    n_modes = len(mode_indices)
    fig, axes = plt.subplots(1, n_modes, figsize=figsize, sharey=True)

    if n_modes == 1:
        axes = [axes]

    for idx, mode_idx in enumerate(mode_indices):
        ax = axes[idx]

        # Plot mode shape (real and imaginary parts)
        phi_real = phi[:, mode_idx].real
        phi_imag = phi[:, mode_idx].imag

        ax.plot(phi_real, z, 'b-', linewidth=2, label='Real')
        if np.any(np.abs(phi_imag) > 1e-6):
            ax.plot(phi_imag, z, 'r--', linewidth=2, label='Imag')

        # Mode info
        k_val = k[mode_idx]
        phase_speed = 2 * np.pi * freq / k_val.real

        title = f'Mode {mode_idx+1}\n'
        title += f'k = {k_val.real:.4f}\n'
        title += f'c_p = {phase_speed:.1f} m/s'

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Amplitude', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
        ax.legend(loc='best', fontsize=8)

        if idx == 0:
            ax.set_ylabel('Depth (m)', fontsize=12)

        ax.invert_yaxis()

    fig.suptitle(f'Mode Functions at f = {freq:.1f} Hz',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig, axes


def plot_mode_wavenumbers(
    modes: Field,
    figsize: Tuple[float, float] = (10, 8),
    annotate_modes: bool = True,
    max_annotations: int = 20,
) -> Tuple[Figure, Axes]:
    """
    Plot mode wavenumbers in the complex k-plane

    Creates a scatter plot of modal wavenumbers showing real(k) vs imag(k),
    following the Acoustic Toolbox standard (plotmode.m). This visualization
    helps identify propagating modes (small imag(k)) vs evanescent modes
    (large imag(k)).

    Parameters
    ----------
    modes : Field
        Mode Field object from compute_modes()
    figsize : tuple, optional
        Figure size. Default is (10, 8).
    annotate_modes : bool, optional
        If True, annotate each point with its mode number. Default is True.
    max_annotations : int, optional
        Maximum number of mode annotations to show. Default is 20.

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes

    Examples
    --------
    >>> modes = kraken.compute_modes(env, source)
    >>> fig, ax = plot_mode_wavenumbers(modes)
    >>> plt.show()

    Notes
    -----
    - Propagating modes have small imaginary parts (low attenuation)
    - Evanescent modes have large imaginary parts (high attenuation)
    - The pattern reveals the modal structure of the waveguide
    - Follows Acoustic Toolbox plotmode.m standard
    """
    if modes.field_type != 'modes':
        raise ValueError(f"Expected modes field, got {modes.field_type}")

    # Extract wavenumbers from Field
    k = modes.metadata['k']
    freq = modes.metadata.get('frequency', modes.frequencies[0] if len(modes.frequencies) > 0 else 0.0)
    M = len(k)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Extract real and imaginary parts
    k_real = k.real
    k_imag = k.imag

    # Scatter plot of wavenumbers (AT standard: blue circles)
    ax.scatter(k_real, k_imag, c='b', s=50, alpha=0.7, edgecolors='k', linewidth=0.5, zorder=3)

    # Annotate mode numbers (first max_annotations modes)
    if annotate_modes:
        n_annotate = min(M, max_annotations)
        for i in range(n_annotate):
            # Offset annotation slightly to avoid overlap with marker
            ax.annotate(f'{i+1}',
                       xy=(k_real[i], k_imag[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, color='darkblue',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='gray', alpha=0.7),
                       zorder=4)

    # Add reference lines
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.3, zorder=1)  # Real axis
    ax.axvline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.3, zorder=1)  # Imaginary axis

    # Labels and title
    ax.set_xlabel('Real(k) (rad/m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Imag(k) (rad/m)', fontsize=12, fontweight='bold')
    ax.set_title(f'Mode Wavenumbers in Complex k-Plane\nf = {freq:.1f} Hz, {M} modes',
                fontsize=14, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, zorder=ZORDER_GRID)

    # Add text box with information
    textstr = f'Modes: {M}\n'
    textstr += f'Frequency: {freq:.1f} Hz\n'
    textstr += f'k range: [{k_real.min():.4f}, {k_real.max():.4f}] rad/m'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, zorder=ZORDER_ANNOTATIONS)

    plt.tight_layout()

    return fig, ax


def compare_models(
    results: dict,
    env: Environment,
    figsize: Tuple[float, float] = (16, 6),
    vmin: float = 30.0,
    vmax: float = 90.0,
) -> Tuple[Figure, Axes]:
    """
    Compare transmission loss from multiple models

    Parameters
    ----------
    results : dict
        Dictionary with model names as keys and Field objects as values.
        Example: {'RAM': field1, 'Bellhop': field2}
    env : Environment
        Environment for reference
    figsize : tuple, optional
        Figure size. Default is (16, 6).

    Returns
    -------
    fig : Figure
        Matplotlib figure
    axes : array of Axes
        Matplotlib axes array

    Examples
    --------
    >>> results = {
    ...     'RAM': ram_result,
    ...     'Bellhop': bellhop_result,
    ... }
    >>> fig, axes = compare_models(results, env)
    >>> plt.show()
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=True)

    if n_models == 1:
        axes = [axes]

    for idx, (model_name, field) in enumerate(results.items()):
        ax = axes[idx]

        # Skip if model failed
        if field is None:
            ax.text(0.5, 0.5, f'{model_name}\n(Failed)', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='red')
            ax.set_title(model_name, fontweight='bold')
            ax.axis('off')
            continue

        # Convert to TL if needed
        if field.field_type != 'tl':
            field = field.to_db()

        # Plot
        ranges_km = field.ranges / 1000.0
        depths = field.depths
        R, Z = np.meshgrid(ranges_km, depths)

        # Centralized TL colormap (style.COLORMAPS['tl'])
        data2d = _select_2d_slice(field)
        im = ax.pcolormesh(
            R, Z, data2d,
            cmap=get_cmap_for_field('tl'),
            vmin=vmin,
            vmax=vmax,
            shading='auto',
            zorder=1
        )

        ax.invert_yaxis()

        # Set proper y-axis limits
        max_depth = depths.max()
        if env is not None:
            max_depth = max(max_depth, env.depth)
        ax.set_ylim([max_depth * 1.05, 0])

        ax.set_xlabel('Range (km)', fontsize=12)
        ax.set_title(f'{model_name}\nTL (dB)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, zorder=0)

        # Bottom
        ax.axhline(env.depth, color='k', linewidth=2.5, linestyle='--', alpha=0.8, zorder=2)
        ax.fill_between(ranges_km, env.depth, max_depth * 1.05,
                       color='sienna', alpha=0.5, zorder=2, edgecolor='none')

        if idx == 0:
            ax.set_ylabel('Depth (m)', fontsize=12)

    # Shared colorbar
    fig.colorbar(im, ax=axes, label='TL (dB)',
                 fraction=0.046, pad=0.04)

    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    try:
        plt.tight_layout()
    except (RuntimeError, ValueError):
        pass  # Ignore tight_layout warnings for complex figures

    return fig, axes


def plot_dispersion_curves(
    modes_dict: dict,
    figsize: Tuple[float, float] = (12, 8),
) -> Tuple[Figure, Axes]:
    """
    Plot dispersion curves (phase speed vs frequency) for multiple frequencies

    Parameters
    ----------
    modes_dict : dict
        Dictionary with frequencies as keys and mode data as values
        Example: {50: modes_50hz, 100: modes_100hz, ...}
    figsize : tuple, optional
        Figure size. Default is (12, 8).

    Returns
    -------
    fig : Figure
        Matplotlib figure
    axes : tuple of Axes
        Matplotlib axes (dispersion, attenuation, mode count)

    Examples
    --------
    >>> modes_dict = {50: kraken.compute_modes(env, source_50),
    ...               100: kraken.compute_modes(env, source_100)}
    >>> fig, axes = plot_dispersion_curves(modes_dict)
    >>> plt.show()
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)

    frequencies = sorted(modes_dict.keys())

    # Collect data for each mode across frequencies
    max_modes = max(modes['M'] for modes in modes_dict.values())

    # Plot dispersion curves (phase speed vs frequency)
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, max_modes)))

    for mode_idx in range(max_modes):
        freqs = []
        phase_speeds = []
        attenuations = []

        for freq in frequencies:
            modes = modes_dict[freq]
            if mode_idx < modes['M']:
                k = modes['k'][mode_idx]
                cp = 2 * np.pi * freq / k.real
                alpha = -8.686 * k.imag  # Convert to dB/m

                freqs.append(freq)
                phase_speeds.append(cp)
                attenuations.append(alpha)

        if freqs:
            color = colors[mode_idx % len(colors)]
            ax1.plot(freqs, phase_speeds, 'o-', linewidth=2, markersize=6,
                    color=color, label=f'Mode {mode_idx+1}')
            ax2.semilogy(freqs, attenuations, 'o-', linewidth=2, markersize=6,
                        color=color)

    # Plot number of modes vs frequency
    mode_counts = [modes_dict[f]['M'] for f in frequencies]
    ax3.plot(frequencies, mode_counts, 'ko-', linewidth=2, markersize=8)

    # Labels
    ax1.set_ylabel('Phase Speed (m/s)', fontsize=12)
    ax1.set_title('Dispersion Curves', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', ncol=2, fontsize=9)

    ax2.set_ylabel('Attenuation (dB/m)', fontsize=12)
    ax2.set_title('Modal Attenuation', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    ax3.set_xlabel('Frequency (Hz)', fontsize=12)
    ax3.set_ylabel('Number of Modes', fontsize=12)
    ax3.set_title('Mode Count vs Frequency', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig, (ax1, ax2, ax3)


def plot_range_cut(
    field: Field,
    depth: float,
    figsize: Tuple[float, float] = (10, 5),
    ax: Optional[Axes] = None,
    frequency: Optional[float] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot transmission loss vs range at a specific depth

    Parameters
    ----------
    field : Field
        Transmission loss field
    depth : float
        Depth at which to extract range cut (m)
    figsize : tuple, optional
        Figure size. Default is (10, 5).
    ax : Axes, optional
        Matplotlib axes
    frequency : float, optional
        For broadband 3-D fields, frequency (Hz) to slice. Defaults to
        the middle frequency.

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes

    Examples
    --------
    >>> fig, ax = plot_range_cut(result, depth=50.0)
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Find closest depth index
    depth_idx = np.argmin(np.abs(field.depths - depth))
    actual_depth = field.depths[depth_idx]

    # Extract TL at this depth (handle 3-D broadband data)
    data2d = _select_2d_slice(field, frequency=frequency)
    tl_vs_range = data2d[depth_idx, :]
    ranges_km = field.ranges / 1000.0

    ax.plot(ranges_km, tl_vs_range, 'b-', linewidth=2)

    ax.set_xlabel('Range (km)', fontsize=12)
    ax.set_ylabel('Transmission Loss (dB)', fontsize=12)
    ax.set_title(f'TL vs Range at {actual_depth:.1f} m depth',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    return fig, ax


def plot_depth_cut(
    field: Field,
    range_m: float,
    figsize: Tuple[float, float] = (6, 8),
    ax: Optional[Axes] = None,
    frequency: Optional[float] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot transmission loss vs depth at a specific range

    Parameters
    ----------
    field : Field
        Transmission loss field
    range_m : float
        Range at which to extract depth cut (m)
    figsize : tuple, optional
        Figure size. Default is (6, 8).
    ax : Axes, optional
        Matplotlib axes
    frequency : float, optional
        For broadband 3-D fields, frequency (Hz) to slice. Defaults to
        the middle frequency.

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes

    Examples
    --------
    >>> fig, ax = plot_depth_cut(result, range_m=2000.0)
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Find closest range index
    range_idx = np.argmin(np.abs(field.ranges - range_m))
    actual_range = field.ranges[range_idx]

    # Extract TL at this range (handle 3-D broadband data)
    data2d = _select_2d_slice(field, frequency=frequency)
    tl_vs_depth = data2d[:, range_idx]

    ax.plot(tl_vs_depth, field.depths, 'b-', linewidth=2)

    ax.set_xlabel('Transmission Loss (dB)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title(f'TL vs Depth at {actual_range/1000:.2f} km range',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    return fig, ax


def compare_range_cuts(
    results: dict,
    depth: float,
    figsize: Tuple[float, float] = (12, 6),
    frequency: Optional[float] = None,
) -> Tuple[Figure, Axes]:
    """
    Compare transmission loss vs range from multiple models at specific depth

    Parameters
    ----------
    results : dict
        Dictionary with model names as keys and Field objects as values
    depth : float
        Depth at which to extract range cuts (m)
    figsize : tuple, optional
        Figure size. Default is (12, 6).
    frequency : float, optional
        For broadband 3-D fields, frequency (Hz) to slice. Defaults to
        the middle frequency of each field.

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes

    Examples
    --------
    >>> results = {'RAM': field1, 'Bellhop': field2}
    >>> fig, ax = compare_range_cuts(results, depth=50.0)
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    for model_name, field in results.items():
        # Use professional model colors (centralized in style.MODEL_COLORS)
        color = get_model_color(model_name)

        # Find closest depth index
        depth_idx = np.argmin(np.abs(field.depths - depth))

        # Extract TL at this depth (handle 3-D broadband)
        data2d = _select_2d_slice(field, frequency=frequency)
        tl = data2d[depth_idx, :]
        ranges_km = field.ranges / 1000.0

        ax.plot(ranges_km, tl, linewidth=2.5, label=model_name,
               color=color, alpha=0.9)

    actual_depth = field.depths[depth_idx]

    # Professional formatting
    format_axes_professional(
        ax,
        title=f'Model Comparison: TL vs Range at {actual_depth:.1f} m depth',
        xlabel='Range (km)',
        ylabel='Transmission Loss (dB)'
    )

    ax.legend(loc='best', fontsize=11, framealpha=0.95, edgecolor='gray', fancybox=False)
    ax.invert_yaxis()

    return fig, ax


def plot_model_statistics(
    results: dict,
    compute_times: dict,
    figsize: Tuple[float, float] = (14, 10),
) -> Tuple[Figure, Axes]:
    """
    Plot comprehensive statistics for multiple model results

    Parameters
    ----------
    results : dict
        Dictionary with model names as keys and Field objects as values
    compute_times : dict
        Dictionary with model names as keys and computation times (s) as values
    figsize : tuple, optional
        Figure size. Default is (14, 10).

    Returns
    -------
    fig : Figure
        Matplotlib figure
    axes : array of Axes
        Matplotlib axes array

    Examples
    --------
    >>> results = {'RAM': field1, 'Bellhop': field2}
    >>> times = {'RAM': 1.5, 'Bellhop': 0.3}
    >>> fig, axes = plot_model_statistics(results, times)
    >>> plt.show()
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes.flatten()

    model_names = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    # 1. Computation time comparison
    times = [compute_times[name] for name in model_names]
    bars = ax1.barh(model_names, times, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Computation Time (s)', fontsize=12)
    ax1.set_title('Model Performance', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # Add time labels on bars
    for bar, time in zip(bars, times):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2,
                f'{time:.3f}s', ha='left', va='center', fontsize=10,
                fontweight='bold')

    # 2. TL range comparison (collapse broadband 3-D to a 2-D slice first)
    slices_2d = [_select_2d_slice(field) for field in results.values()]
    tl_mins = [np.min(s[np.isfinite(s)]) for s in slices_2d]
    tl_maxs = [np.max(s[np.isfinite(s)]) for s in slices_2d]

    x = np.arange(len(model_names))
    width = 0.35
    ax2.bar(x - width/2, tl_mins, width, label='Min TL', color='lightblue',
           edgecolor='black', alpha=0.7)
    ax2.bar(x + width/2, tl_maxs, width, label='Max TL', color='coral',
           edgecolor='black', alpha=0.7)
    ax2.set_ylabel('Transmission Loss (dB)', fontsize=12)
    ax2.set_title('TL Range by Model', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Grid size comparison (use 2-D slice so labels stay readable)
    grid_sizes = [f"{s.shape[0]}×{s.shape[1]}" for s in slices_2d]
    n_points = [s.size for s in slices_2d]

    bars = ax3.bar(model_names, n_points, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Number of Grid Points', fontsize=12)
    ax3.set_title('Grid Resolution', fontsize=12, fontweight='bold')
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add grid size labels on bars
    for bar, grid_size in zip(bars, grid_sizes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height,
                grid_size, ha='center', va='bottom', fontsize=9,
                fontweight='bold')

    # 4. Range cut comparison at mid-depth
    for idx, (model_name, field) in enumerate(results.items()):
        mid_depth_idx = len(field.depths) // 2
        data2d = slices_2d[idx]
        tl = data2d[mid_depth_idx, :]
        ranges_km = field.ranges / 1000.0
        ax4.plot(ranges_km, tl, linewidth=2, label=model_name,
                color=colors[idx], alpha=0.8)

    ax4.set_xlabel('Range (km)', fontsize=12)
    ax4.set_ylabel('Transmission Loss (dB)', fontsize=12)
    ax4.set_title('TL Comparison at Mid-Depth', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best', fontsize=9)
    ax4.invert_yaxis()

    plt.tight_layout()

    return fig, axes


def plot_model_comparison_matrix(
    results: dict,
    comparison_metric: str = 'rms',
    source_depth: Optional[float] = None,
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = 'RdYlGn_r',
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot colored comparison matrix showing agreement between models

    Visualizes pairwise differences between models using RMS error or correlation.
    This professional-grade visualization helps assess model agreement and identify
    outliers in multi-model comparisons.

    Parameters
    ----------
    results : dict
        Dictionary with model names as keys and Field objects as values
    comparison_metric : str, optional
        Metric for comparison: 'rms' (RMS error), 'correlation', 'max_diff'.
        Default is 'rms'.
    source_depth : float, optional
        Depth at which to compare models. If None, uses full field comparison.
    figsize : tuple, optional
        Figure size. Default is (10, 8).
    cmap : str, optional
        Colormap for matrix. Default is 'RdYlGn_r' (red=bad, green=good).
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes

    Examples
    --------
    >>> results = {'Bellhop': field1, 'RAM': field2, 'Kraken': field3}
    >>> fig, ax = plot_model_comparison_matrix(results, comparison_metric='rms')
    >>> plt.show()

    Notes
    -----
    - RMS error: Root Mean Square difference (lower is better, shown in dB)
    - Correlation: Pearson correlation coefficient (higher is better, 0-1)
    - Max difference: Maximum absolute difference (lower is better, dB)

    The matrix is symmetric with zeros on the diagonal (model vs itself).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    model_names = list(results.keys())
    n_models = len(model_names)

    if n_models < 2:
        ax.text(0.5, 0.5, 'Need at least 2 models\nfor comparison matrix',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, fontweight='bold')
        ax.axis('off')
        return fig, ax

    # Initialize comparison matrix
    comparison_matrix = np.zeros((n_models, n_models))

    # Compute pairwise comparisons
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                comparison_matrix[i, j] = 0.0  # Perfect agreement with self
            else:
                field_i = results[model_names[i]]
                field_j = results[model_names[j]]

                # Skip if either field is None (model couldn't run)
                if field_i is None or field_j is None:
                    comparison_matrix[i, j] = np.nan
                    continue

                # Extract data for comparison (collapse 3-D to 2-D first)
                slice_i = _select_2d_slice(field_i)
                slice_j = _select_2d_slice(field_j)

                if source_depth is not None:
                    # Compare at specific depth
                    depth_idx_i = np.argmin(np.abs(field_i.depths - source_depth))
                    depth_idx_j = np.argmin(np.abs(field_j.depths - source_depth))
                    data_i = slice_i[depth_idx_i, :]
                    data_j = slice_j[depth_idx_j, :]

                    # Interpolate to common range grid if needed
                    if not np.array_equal(field_i.ranges, field_j.ranges):
                        ranges_common = np.union1d(field_i.ranges, field_j.ranges)
                        data_i = np.interp(ranges_common, field_i.ranges, data_i)
                        data_j = np.interp(ranges_common, field_j.ranges, data_j)
                else:
                    # Full field comparison - flatten to 1D
                    data_i = slice_i.flatten()
                    data_j = slice_j.flatten()

                    # Handle different grid sizes by interpolation
                    if data_i.size != data_j.size:
                        # Use smaller grid size for fair comparison
                        target_size = min(data_i.size, data_j.size)
                        data_i = np.interp(np.linspace(0, 1, target_size),
                                          np.linspace(0, 1, data_i.size), data_i)
                        data_j = np.interp(np.linspace(0, 1, target_size),
                                          np.linspace(0, 1, data_j.size), data_j)

                # Remove NaN and Inf values
                valid_mask = np.isfinite(data_i) & np.isfinite(data_j)
                data_i = data_i[valid_mask]
                data_j = data_j[valid_mask]

                if len(data_i) == 0:
                    comparison_matrix[i, j] = np.nan
                    continue

                # Compute comparison metric
                if comparison_metric == 'rms':
                    # RMS error in dB
                    comparison_matrix[i, j] = np.sqrt(np.mean((data_i - data_j)**2))
                elif comparison_metric == 'correlation':
                    # Pearson correlation (convert to distance: 1-r)
                    if np.std(data_i) > 0 and np.std(data_j) > 0:
                        corr = np.corrcoef(data_i, data_j)[0, 1]
                        comparison_matrix[i, j] = 1.0 - corr  # Distance metric
                    else:
                        comparison_matrix[i, j] = 1.0  # No correlation
                elif comparison_metric == 'max_diff':
                    # Maximum absolute difference
                    comparison_matrix[i, j] = np.max(np.abs(data_i - data_j))
                else:
                    raise ValueError(f"Unknown comparison metric: {comparison_metric}")

    # Determine colorbar limits based on metric
    if comparison_metric == 'rms':
        vmax = max(10, np.nanmax(comparison_matrix[comparison_matrix < np.inf]))
        vmin = 0
        cbar_label = 'RMS Error (dB)'
    elif comparison_metric == 'correlation':
        vmax = 1.0  # Maximum distance
        vmin = 0.0
        cbar_label = 'Correlation Distance (1-r)'
    elif comparison_metric == 'max_diff':
        vmax = max(20, np.nanmax(comparison_matrix[comparison_matrix < np.inf]))
        vmin = 0
        cbar_label = 'Max Difference (dB)'

    # Plot matrix with no interpolation (sharp color boundaries)
    im = ax.imshow(comparison_matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect='equal', interpolation='none')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    # Set ticks and labels
    ax.set_xticks(range(n_models))
    ax.set_yticks(range(n_models))
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(model_names, fontsize=11)

    # Add value annotations
    for i in range(n_models):
        for j in range(n_models):
            if i != j and np.isfinite(comparison_matrix[i, j]):
                value = comparison_matrix[i, j]
                # Choose text color based on background
                if comparison_metric == 'rms' or comparison_metric == 'max_diff':
                    text_color = 'white' if value > vmax/2 else 'black'
                else:
                    text_color = 'white' if value > 0.5 else 'black'

                ax.text(j, i, f'{value:.2f}',
                       ha='center', va='center',
                       color=text_color, fontsize=10, fontweight='bold')

    # Title
    metric_titles = {
        'rms': 'Model Agreement (RMS Error)',
        'correlation': 'Model Correlation Distance',
        'max_diff': 'Model Agreement (Max Difference)'
    }
    title = metric_titles.get(comparison_metric, 'Model Comparison Matrix')
    if source_depth is not None:
        title += f' at {source_depth:.0f}m depth'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    # Grid for readability
    ax.set_xticks(np.arange(n_models) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_models) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

    plt.tight_layout()

    return fig, ax


def plot_comparison_curves(
    results: dict,
    source_depth: float,
    mid_range_km: Optional[float] = None,
    figsize: Tuple[float, float] = (14, 6),
    frequency: Optional[float] = None,
) -> Tuple[Figure, np.ndarray]:
    """
    Plot TL comparison curves: TL vs range and TL vs depth

    Creates side-by-side plots showing transmission loss profiles for all models.
    Essential for quantitative comparison of model behavior and identifying
    differences in propagation characteristics.

    Parameters
    ----------
    results : dict
        Dictionary with model names as keys and Field objects as values
    source_depth : float
        Depth in meters for TL vs range plot
    mid_range_km : float, optional
        Range in km for TL vs depth plot. If None, uses median range.
    figsize : tuple, optional
        Figure size. Default is (14, 6).

    Returns
    -------
    fig : Figure
        Matplotlib figure
    axes : ndarray
        Array of matplotlib axes [ax_range, ax_depth]

    Examples
    --------
    >>> results = {'Bellhop': field1, 'RAM': field2}
    >>> fig, axes = plot_comparison_curves(results, source_depth=50.0)
    >>> plt.show()

    Notes
    -----
    - Left plot: TL vs range at specified source depth
    - Right plot: TL vs depth at mid-range (or specified range)
    - Uses consistent color scheme for models across both plots
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax_range, ax_depth = axes

    # Plot 1: TL vs Range at source depth
    for name, field in results.items():
        # Skip None fields (models that couldn't run)
        if field is None:
            continue
        # Find closest depth to source_depth
        depth_idx = np.argmin(np.abs(field.depths - source_depth))
        actual_depth = field.depths[depth_idx]

        data2d = _select_2d_slice(field, frequency=frequency)
        tl_vs_range = data2d[depth_idx, :]
        ranges_km = field.ranges / 1000.0

        # Centralized model palette (style.MODEL_COLORS via get_model_color).
        color = get_model_color(name)

        ax_range.plot(ranges_km, tl_vs_range, linewidth=2.5,
                     label=f'{name} ({actual_depth:.1f}m)',
                     color=color, alpha=0.85)

    ax_range.set_xlabel('Range (km)', fontsize=12, fontweight='bold')
    ax_range.set_ylabel('Transmission Loss (dB)', fontsize=12, fontweight='bold')
    ax_range.set_title(f'TL vs Range at {source_depth:.0f}m Depth',
                      fontsize=13, fontweight='bold')
    ax_range.legend(fontsize=10, loc='best', framealpha=0.9)
    ax_range.grid(True, alpha=0.3, linestyle='--')

    # Plot 2: TL vs Depth at mid-range
    # Determine mid-range if not specified
    if mid_range_km is None:
        all_ranges = np.concatenate([field.ranges for field in results.values() if field is not None])
        mid_range_km = np.median(all_ranges) / 1000.0

    for name, field in results.items():
        # Skip None fields (models that couldn't run)
        if field is None:
            continue
        # Find closest range to mid_range_km
        range_idx = np.argmin(np.abs(field.ranges/1000.0 - mid_range_km))
        actual_range_km = field.ranges[range_idx] / 1000.0

        data2d = _select_2d_slice(field, frequency=frequency)
        tl_vs_depth = data2d[:, range_idx]
        depths = field.depths

        color = get_model_color(name)

        ax_depth.plot(tl_vs_depth, depths, linewidth=2.5,
                     label=f'{name} ({actual_range_km:.1f}km)',
                     color=color, alpha=0.85)

    ax_depth.invert_yaxis()
    ax_depth.set_xlabel('Transmission Loss (dB)', fontsize=12, fontweight='bold')
    ax_depth.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
    ax_depth.set_title(f'TL vs Depth at {mid_range_km:.1f}km Range',
                      fontsize=13, fontweight='bold')
    ax_depth.legend(fontsize=10, loc='best', framealpha=0.9)
    ax_depth.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    return fig, axes


def plot_ssp_2d(
    env: Environment,
    figsize: Tuple[float, float] = (12, 8),
    cmap: Optional[str] = None,
    ax: Optional[Axes] = None,
    seafloor_alpha: float = 1.0,
) -> Tuple[Figure, Axes]:
    """
    Plot 2D range-dependent sound speed profile

    Creates heatmap showing how sound speed varies with depth and range.

    Parameters
    ----------
    env : Environment
        Environment with 2D SSP (ssp_2d_matrix must be provided)
    figsize : tuple, optional
        Figure size. Default is (12, 8).
    cmap : str, optional
        Colormap. Default is 'RdYlBu_r' (red=warm, blue=cold).
    ax : Axes, optional
        Matplotlib axes
    seafloor_alpha : float, optional
        Transparency of seafloor/ground fill (0.0 = fully transparent, 1.0 = opaque).
        Default is 1.0 (no transparency).

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes

    Examples
    --------
    >>> env = Environment(..., ssp_2d_ranges=ranges, ssp_2d_matrix=ssp_matrix)
    >>> fig, ax = plot_ssp_2d(env)
    >>> plt.show()
    """
    if not env.has_range_dependent_ssp():
        raise ValueError("Environment must have 2D SSP (ssp_2d_matrix)")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Centralized SSP colormap
    if cmap is None:
        cmap = get_cmap_for_field('ssp')

    # Get data
    ranges_km = env.ssp_2d_ranges
    ssp_matrix = env.ssp_2d_matrix

    # Use depths that match ssp_2d_matrix dimensions
    # env.ssp_data may have extra points added by Environment processing
    if env.ssp_2d_matrix.shape[0] != env.ssp_data.shape[0]:
        # Create depth array matching ssp_2d_matrix first dimension
        depths = np.linspace(0, env.depth, env.ssp_2d_matrix.shape[0])
    else:
        depths = env.ssp_data[:, 0]

    # Create meshgrid
    R, Z = np.meshgrid(ranges_km, depths)

    # Plot heatmap
    im = ax.pcolormesh(R, Z, ssp_matrix, cmap=cmap, shading='auto')

    # Overlay bathymetry if available
    if len(env.bathymetry) > 1:
        bathy_ranges = env.bathymetry[:, 0] / 1000.0
        bathy_depths = env.bathymetry[:, 1]
        ax.plot(bathy_ranges, bathy_depths, 'k-', linewidth=3, label='Bathymetry', zorder=5)
        ax.fill_between(
            bathy_ranges, bathy_depths, depths.max() * 1.1,
            color='sienna', alpha=seafloor_alpha, edgecolor='black', linewidth=1.5, zorder=4
        )

    # Invert y-axis
    ax.invert_yaxis()

    # Labels
    ax.set_xlabel('Range (km)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title(f'Range-Dependent Sound Speed Profile\n{env.name}',
                 fontsize=14, fontweight='bold')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, label='Sound Speed (m/s)')

    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    if len(env.bathymetry) > 1:
        ax.legend(loc='upper right')

    return fig, ax


def plot_bottom_properties(
    env: Environment,
    figsize: Tuple[float, float] = (14, 10),
    seafloor_alpha: float = 1.0,
) -> Tuple[Figure, Axes]:
    """
    Plot range-dependent bottom properties

    Shows how bottom acoustic properties vary with range.

    Parameters
    ----------
    env : Environment
        Environment with range-dependent bottom (bottom_rd)
    figsize : tuple, optional
        Figure size. Default is (14, 10).
    seafloor_alpha : float, optional
        Transparency of seafloor/ground fill (0.0 = fully transparent, 1.0 = opaque).
        Default is 1.0 (no transparency).

    Returns
    -------
    fig : Figure
        Matplotlib figure
    axes : array of Axes
        Matplotlib axes array (4 subplots)

    Examples
    --------
    >>> env = Environment(..., bottom=RangeDependentBottom(...))
    >>> fig, axes = plot_bottom_properties(env)
    >>> plt.show()
    """
    if not env.has_range_dependent_bottom():
        raise ValueError("Environment must have range-dependent bottom (bottom_rd)")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    bottom_rd = env.bottom_rd
    ranges_km = bottom_rd.ranges_km

    # Plot 1: Depth (Bathymetry)
    axes[0].plot(ranges_km, bottom_rd.depths, 'o-', linewidth=2, markersize=8, color='brown')
    axes[0].fill_between(ranges_km, bottom_rd.depths, bottom_rd.depths.max() * 1.1,
                         color='saddlebrown', alpha=seafloor_alpha)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Range (km)', fontsize=11)
    axes[0].set_ylabel('Depth (m)', fontsize=11)
    axes[0].set_title('Bottom Depth', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Sound Speed
    axes[1].plot(ranges_km, bottom_rd.sound_speed, 'o-', linewidth=2,
                 markersize=8, color='darkblue')
    axes[1].set_xlabel('Range (km)', fontsize=11)
    axes[1].set_ylabel('Sound Speed (m/s)', fontsize=11)
    axes[1].set_title('Bottom Sound Speed (Compressional)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Density
    axes[2].plot(ranges_km, bottom_rd.density, 'o-', linewidth=2,
                 markersize=8, color='darkgreen')
    axes[2].set_xlabel('Range (km)', fontsize=11)
    axes[2].set_ylabel('Density (g/cm³)', fontsize=11)
    axes[2].set_title('Bottom Density', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Attenuation
    axes[3].plot(ranges_km, bottom_rd.attenuation, 'o-', linewidth=2,
                 markersize=8, color='darkred')
    axes[3].set_xlabel('Range (km)', fontsize=11)
    axes[3].set_ylabel('Attenuation (dB/λ)', fontsize=11)
    axes[3].set_title('Bottom Attenuation', fontsize=12, fontweight='bold')
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(f'Range-Dependent Bottom Properties\n{env.name}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig, axes


def plot_layered_bottom(
    env: Environment,
    figsize: Tuple[float, float] = (10, 8),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot layered bottom structure showing sediment layers and properties.

    Draws a vertical cross-section of the sub-bottom with each layer as a
    colored rectangle annotated with its acoustic properties.

    Parameters
    ----------
    env : Environment
        Environment with a LayeredBottom (env.bottom_layered).
    figsize : tuple, optional
        Figure size. Default is (10, 8).
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes
    """
    from uacpy.core.environment import LayeredBottom

    if not env.has_layered_bottom():
        raise ValueError("Environment must have a LayeredBottom (env.bottom_layered)")

    lb = env.bottom_layered
    seafloor = env.depth

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Color layers by sound speed
    cs_values = [layer.sound_speed for layer in lb.layers]
    cs_min, cs_max = min(cs_values), max(cs_values)
    cs_range = cs_max - cs_min if cs_max > cs_min else 1.0
    cmap = plt.cm.YlOrBr

    # Draw water column
    ax.axhspan(0, seafloor, color='lightblue', alpha=0.3, label='Water')
    ax.axhline(seafloor, color='saddlebrown', linewidth=2)

    # Draw each layer
    depth_top = seafloor
    for i, layer in enumerate(lb.layers):
        depth_bot = depth_top + layer.thickness
        norm_cs = (layer.sound_speed - cs_min) / cs_range if cs_range > 0 else 0.5
        color = cmap(0.2 + 0.6 * norm_cs)  # avoid extremes

        ax.axhspan(depth_top, depth_bot, color=color, alpha=0.7)
        ax.axhline(depth_bot, color='saddlebrown', linewidth=1, linestyle='--')

        # Annotate
        mid = (depth_top + depth_bot) / 2
        props = (f"Layer {i+1}: {layer.thickness:.1f} m\n"
                 f"cp={layer.sound_speed:.0f} m/s, "
                 f"\u03c1={layer.density:.2f} g/cm\u00b3, "
                 f"\u03b1={layer.attenuation:.2f} dB/\u03bb")
        if layer.shear_speed > 0:
            props += f"\ncs={layer.shear_speed:.0f} m/s"

        ax.text(0.5, mid, props, transform=ax.get_yaxis_transform(),
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        depth_top = depth_bot

    # Draw halfspace below
    hs = lb.halfspace
    hs_display = depth_top + max(10.0, lb.total_thickness() * 0.3)
    ax.axhspan(depth_top, hs_display, color='saddlebrown', alpha=0.4)
    hs_mid = (depth_top + hs_display) / 2
    hs_text = (f"Halfspace\n"
               f"cp={hs.sound_speed:.0f} m/s, "
               f"\u03c1={hs.density:.2f} g/cm\u00b3")
    ax.text(0.5, hs_mid, hs_text, transform=ax.get_yaxis_transform(),
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_ylim(hs_display, max(0, seafloor - 20))
    ax.set_ylabel('Depth (m)', fontsize=11)
    ax.set_title(f'Layered Bottom Structure — {env.name}', fontsize=13, fontweight='bold')
    ax.set_xticks([])
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_rd_layered_bottom(
    env: Environment,
    figsize: Tuple[float, float] = (14, 8),
) -> Tuple[Figure, np.ndarray]:
    """
    Plot range-dependent layered bottom as a 2-D cross-section.

    Each range point's layer stack is drawn as coloured rectangles below the
    seafloor, with properties annotated.  The water column is shown above.

    Parameters
    ----------
    env : Environment
        Environment with ``bottom_rd_layered`` set.
    figsize : tuple, optional
        Figure size. Default (14, 8).

    Returns
    -------
    fig : Figure
    axes : ndarray of Axes
        Two axes: [cross-section, property legend].
    """
    from uacpy.core.environment import RangeDependentLayeredBottom

    if not env.has_range_dependent_layered_bottom():
        raise ValueError("Environment must have a RangeDependentLayeredBottom")

    rdl = env.bottom_rd_layered
    n_ranges = len(rdl.ranges_km)

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             gridspec_kw={'width_ratios': [3, 1]})
    ax_main = axes[0]
    ax_legend = axes[1]

    # Collect all sound speeds for colour normalisation
    all_cs = []
    for lb in rdl.profiles:
        all_cs.extend([layer.sound_speed for layer in lb.layers])
        all_cs.append(lb.halfspace.sound_speed)
    cs_min, cs_max = min(all_cs), max(all_cs)
    cs_range = cs_max - cs_min if cs_max > cs_min else 1.0
    cmap = plt.cm.YlOrBr

    # Draw water column
    max_depth = float(rdl.depths.max())
    ax_main.fill_between(rdl.ranges_km, 0, rdl.depths, color='lightblue', alpha=0.3)
    ax_main.plot(rdl.ranges_km, rdl.depths, color='saddlebrown', linewidth=2,
                 label='Seafloor')

    # Draw layer stacks at each range as vertical columns
    # Column width = gap between range points (or fraction of total range)
    total_range = rdl.ranges_km[-1] - rdl.ranges_km[0]
    col_width = total_range / (n_ranges * 1.5) if n_ranges > 1 else total_range * 0.3

    max_bottom = max_depth
    for i_r in range(n_ranges):
        r_km = rdl.ranges_km[i_r]
        seafloor = rdl.depths[i_r]
        lb = rdl.profiles[i_r]
        depth_top = seafloor

        for layer in lb.layers:
            depth_bot = depth_top + layer.thickness
            norm_cs = (layer.sound_speed - cs_min) / cs_range
            color = cmap(0.2 + 0.6 * norm_cs)

            rect = plt.Rectangle(
                (r_km - col_width / 2, depth_top), col_width,
                layer.thickness, facecolor=color, edgecolor='saddlebrown',
                linewidth=0.8, alpha=0.85,
            )
            ax_main.add_patch(rect)
            depth_top = depth_bot

        # Halfspace hint
        hs_h = max(5, lb.total_thickness() * 0.3)
        rect_hs = plt.Rectangle(
            (r_km - col_width / 2, depth_top), col_width,
            hs_h, facecolor='saddlebrown', edgecolor='saddlebrown',
            linewidth=0.8, alpha=0.4,
        )
        ax_main.add_patch(rect_hs)
        max_bottom = max(max_bottom, depth_top + hs_h)

    ax_main.set_ylim(max_bottom * 1.05, 0)
    ax_main.set_xlim(rdl.ranges_km[0] - col_width, rdl.ranges_km[-1] + col_width)
    ax_main.set_xlabel('Range (km)', fontsize=11)
    ax_main.set_ylabel('Depth (m)', fontsize=11)
    ax_main.set_title(f'Range-Dependent Layered Bottom — {env.name}',
                      fontsize=13, fontweight='bold')
    ax_main.grid(True, alpha=0.3)

    # Right panel: property summary table per range
    ax_legend.axis('off')
    text_lines = []
    for i_r in range(n_ranges):
        lb = rdl.profiles[i_r]
        text_lines.append(f'r = {rdl.ranges_km[i_r]:.1f} km '
                          f'(z = {rdl.depths[i_r]:.0f} m)')
        for j, layer in enumerate(lb.layers):
            text_lines.append(
                f'  L{j+1}: {layer.thickness:.0f}m, '
                f'cp={layer.sound_speed:.0f}, '
                f'\u03c1={layer.density:.2f}, '
                f'\u03b1={layer.attenuation:.2f}'
            )
        text_lines.append(
            f'  HS: cp={lb.halfspace.sound_speed:.0f}, '
            f'\u03c1={lb.halfspace.density:.2f}'
        )
        text_lines.append('')

    ax_legend.text(0.05, 0.95, '\n'.join(text_lines),
                   transform=ax_legend.transAxes, fontsize=8,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig, axes


def plot_environment_advanced(
    env: Environment,
    source: Optional[Source] = None,
    receiver: Optional[Receiver] = None,
    figsize: Tuple[float, float] = (16, 12),
    seafloor_alpha: float = 1.0,
) -> Tuple[Figure, Axes]:
    """
    Advanced environment visualization with all features

    Automatically adapts to show range-dependent SSP, bottom properties,
    bathymetry, and source/receiver configuration.

    Parameters
    ----------
    env : Environment
        Environment to plot (can have 2D SSP and/or range-dependent bottom)
    source : Source, optional
        Source to overlay
    receiver : Receiver, optional
        Receivers to overlay
    figsize : tuple, optional
        Figure size. Default is (16, 12).

    Returns
    -------
    fig : Figure
        Matplotlib figure
    axes : list of Axes
        Matplotlib axes list

    Examples
    --------
    >>> fig, axes = plot_environment_advanced(env, source, receiver)
    >>> plt.show()
    """
    has_2d_ssp = env.has_range_dependent_ssp()
    has_rd_bottom = env.has_range_dependent_bottom()

    if has_2d_ssp and has_rd_bottom:
        # Full advanced layout
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        # Top row: 2D SSP (full width)
        ax_ssp_2d = fig.add_subplot(gs[0, :])
        plot_ssp_2d(env, ax=ax_ssp_2d)

        # Middle row: Bottom properties
        ax_cs = fig.add_subplot(gs[1, 0])
        ax_rho = fig.add_subplot(gs[1, 1])
        ax_atten = fig.add_subplot(gs[1, 2])

        bottom_rd = env.bottom_rd
        ranges_km = bottom_rd.ranges_km

        ax_cs.plot(ranges_km, bottom_rd.sound_speed, 'o-', linewidth=2, color='darkblue')
        ax_cs.set_xlabel('Range (km)', fontsize=10)
        ax_cs.set_ylabel('c (m/s)', fontsize=10)
        ax_cs.set_title('Bottom Sound Speed', fontweight='bold')
        ax_cs.grid(True, alpha=0.3)

        ax_rho.plot(ranges_km, bottom_rd.density, 'o-', linewidth=2, color='darkgreen')
        ax_rho.set_xlabel('Range (km)', fontsize=10)
        ax_rho.set_ylabel('ρ (g/cm³)', fontsize=10)
        ax_rho.set_title('Bottom Density', fontweight='bold')
        ax_rho.grid(True, alpha=0.3)

        ax_atten.plot(ranges_km, bottom_rd.attenuation, 'o-', linewidth=2, color='darkred')
        ax_atten.set_xlabel('Range (km)', fontsize=10)
        ax_atten.set_ylabel('α (dB/λ)', fontsize=10)
        ax_atten.set_title('Bottom Attenuation', fontweight='bold')
        ax_atten.grid(True, alpha=0.3)

        # Bottom row: Setup overview
        ax_setup = fig.add_subplot(gs[2, :])

        # Plot bathymetry
        if len(env.bathymetry) > 1:
            bathy_ranges = env.bathymetry[:, 0] / 1000.0
            bathy_depths = env.bathymetry[:, 1]
            ax_setup.plot(bathy_ranges, bathy_depths, 'k-', linewidth=2.5, label='Bathymetry')
            ax_setup.fill_between(
                bathy_ranges, bathy_depths, bathy_depths.max() * 1.2,
                color='saddlebrown', alpha=0.3, edgecolor='black'
            )

        # Plot source and receivers
        if source is not None:
            for sd in source.depth:
                ax_setup.plot(0, sd, 'r*', markersize=18, label='Source', zorder=10)

        if receiver is not None:
            r_km = receiver.ranges / 1000.0
            if receiver.receiver_type == 'grid':
                R_km, D = np.meshgrid(r_km, receiver.depths)
                ax_setup.scatter(R_km[::3, ::3].flatten(), D[::3, ::3].flatten(),
                                c='blue', marker='v', s=25, alpha=0.6, label='Receivers')
            else:
                ax_setup.scatter(r_km, receiver.depths, c='blue', marker='v',
                                s=40, alpha=0.7, label='Receivers')

        ax_setup.invert_yaxis()
        ax_setup.set_xlabel('Range (km)', fontsize=12)
        ax_setup.set_ylabel('Depth (m)', fontsize=12)
        ax_setup.set_title('Source/Receiver Configuration', fontsize=12, fontweight='bold')
        ax_setup.grid(True, alpha=0.3)
        ax_setup.legend(loc='upper right')

        axes = [ax_ssp_2d, ax_cs, ax_rho, ax_atten, ax_setup]

    elif has_2d_ssp:
        # Simpler layout with just 2D SSP
        fig = plt.figure(figsize=(figsize[0], figsize[1] * 0.7))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        ax_ssp_2d = fig.add_subplot(gs[0, :])
        plot_ssp_2d(env, ax=ax_ssp_2d)

        ax_ssp = fig.add_subplot(gs[1, 0])
        plot_ssp(env, ax=ax_ssp)

        ax_bathy = fig.add_subplot(gs[1, 1])
        plot_bathymetry(env, ax=ax_bathy)

        axes = [ax_ssp_2d, ax_ssp, ax_bathy]

    elif has_rd_bottom:
        # Just bottom properties
        return plot_bottom_properties(env, figsize=figsize)

    else:
        # Fall back to standard environment plot
        return plot_environment(env, source, receiver, figsize=figsize)

    fig.suptitle(f'Advanced Environment: {env.name}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig, axes


def plot_transmission_loss_polar(
    field: Field,
    receiver_depth: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 10),
    show_colorbar: bool = True,
    ax: Optional[Axes] = None,
):
    """
    Plot transmission loss in polar coordinates (bearing/azimuth vs range)

    For omnidirectional sources or 3D propagation with azimuthal variation.
    Requires shade file with theta/bearing dimension.

    Parameters
    ----------
    field : Field
        TL field with theta data in metadata['theta']
    receiver_depth : float, optional
        Receiver depth to extract (m). If None, uses first depth.
    vmin, vmax : float, optional
        TL color scale limits in dB. If None, auto-computed.
    cmap : str, optional
        Colormap name. Default is 'jet_r' (Acoustic Toolbox standard).
    figsize : tuple, optional
        Figure size. Default is (10, 10) for square polar plot.
    show_colorbar : bool, optional
        Show colorbar. Default is True.
    ax : Axes, optional
        Matplotlib polar axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib polar axes
    cbar : Colorbar or None
        Colorbar object if show_colorbar=True

    Examples
    --------
    >>> # Requires Bellhop3D or model with bearing output
    >>> result = bellhop3d.run(env, source, receiver)
    >>> fig, ax, cbar = plot_transmission_loss_polar(result)
    >>> plt.show()

    >>> # Specify receiver depth
    >>> fig, ax, cbar = plot_transmission_loss_polar(result, receiver_depth=50.0)

    Notes
    -----
    This function is equivalent to MATLAB's plotshdpol.m
    The field must have theta/bearing data stored in metadata.

    See Also
    --------
    plot_transmission_loss : Standard 2D rectangular TL plot
    """
    if field.field_type not in ['tl', 'pressure']:
        raise ValueError(f"Polar TL plot requires TL or pressure field, got {field.field_type}")

    # Extract theta from metadata
    theta = field.metadata.get('theta', None)
    if theta is None:
        raise ValueError("Field must have 'theta' in metadata for polar plots. "
                        "This requires models with bearing/azimuth output (e.g., Bellhop3D).")

    # Extract pressure data (shape depends on model)
    # Typically: pressure[theta, source_depth, receiver_depth, range]
    pressure = field.metadata.get('pressure', None)
    if pressure is None:
        raise ValueError("Field must have complex 'pressure' in metadata for polar plots.")

    # Get receiver depths
    receiver_depths = field.depths
    if receiver_depth is None:
        receiver_depth = receiver_depths[0]
        depth_idx = 0
    else:
        depth_idx = np.argmin(np.abs(receiver_depths - receiver_depth))
        receiver_depth = receiver_depths[depth_idx]

    # Extract TL at specified depth
    # Assume pressure shape: (ntheta, nsz, nrz, nrr)
    # Take first source depth [0], specified receiver depth [depth_idx], all theta, all ranges
    if len(pressure.shape) == 4:
        p_slice = pressure[:, 0, depth_idx, :]  # shape: (ntheta, nrr)
    elif len(pressure.shape) == 3:
        # Could be (ntheta, nrz, nrr) without source depth dimension
        if pressure.shape[1] == len(receiver_depths):
            p_slice = pressure[:, depth_idx, :]
        else:
            # Or (nsz, nrz, nrr) - take first source
            p_slice = pressure[0, depth_idx, :]
    else:
        raise ValueError(f"Unexpected pressure shape: {pressure.shape}")

    # Convert to TL
    tl = np.abs(p_slice)
    tl[tl < PRESSURE_FLOOR] = PRESSURE_FLOOR
    tl = -20.0 * np.log10(tl)

    # Handle full circle by duplicating first bearing
    ntheta = len(theta)
    if ntheta > 1:
        d_theta = (theta[-1] - theta[0]) / (ntheta - 1)
        # Check if full circle (360 degrees)
        if np.abs((theta[-1] + d_theta - theta[0]) % 360.0) < 0.01:
            theta = np.append(theta, theta[-1] + d_theta)
            tl = np.vstack([tl, tl[0:1, :]])

    # Convert theta to radians
    theta_rad = np.deg2rad(theta)

    # Get range grid (convert to km to match other TL plots)
    ranges = field.ranges / 1000.0

    # Create meshgrid for polar plot
    Theta, R = np.meshgrid(theta_rad, ranges, indexing='ij')

    # Auto-compute color limits if not provided
    if vmin is None or vmax is None:
        tl_valid = tl[np.isfinite(tl)]
        if len(tl_valid) > 0:
            tlmed = np.median(tl_valid)
            tlstd = np.std(tl_valid)
            if vmax is None:
                vmax = tlmed + 0.75 * tlstd
                vmax = 10 * np.round(vmax / 10)
            if vmin is None:
                vmin = vmax - 50

    # Create figure with polar projection
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')
    else:
        fig = ax.figure
        if ax.name != 'polar':
            raise ValueError("Provided ax must be a polar projection")

    # Select colormap (match standard TL plots)
    if cmap is None:
        cmap = 'jet_r'

    # Plot
    im = ax.pcolormesh(Theta, R, tl, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    # Configure polar axes
    ax.set_theta_zero_location('N')  # 0 degrees at top
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_xlabel('Range (km)', fontsize=11)
    ax.set_title(f'Transmission Loss (Polar)\nDepth = {receiver_depth:.1f} m',
                fontsize=12, fontweight='bold', pad=20)

    # Colorbar
    cbar = None
    if show_colorbar:
        cbar = create_professional_colorbar(fig, im, ax, label='TL (dB)')

    return fig, ax, cbar


def plot_transfer_function(
    field: Field,
    depth_idx: Optional[int] = None,
    range_idx: int = 0,
    figsize: Tuple[float, float] = (12, 6),
    ax: Optional[Axes] = None,
):
    """
    Plot broadband transfer function magnitude vs frequency.

    For transfer_function fields from broadband Scooter, RAM, or KrakenField.
    Shows |H(f)| at a specific depth and range.

    Parameters
    ----------
    field : Field
        Transfer function field with data shape (n_depths, n_freqs, n_ranges)
    depth_idx : int, optional
        Depth index to plot. If None, uses middle depth.
    range_idx : int, optional
        Range index to plot. Default 0.
    figsize : tuple, optional
        Figure size. Default (12, 6).
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    data = field.data  # (n_depths, n_freqs, n_ranges)
    freqs = field.frequencies

    if depth_idx is None:
        depth_idx = data.shape[0] // 2

    # Extract spectrum at chosen depth and range
    spectrum = data[depth_idx, :, range_idx]
    mag_db = 20 * np.log10(np.abs(spectrum) + PRESSURE_FLOOR)

    depth = field.depths[depth_idx] if len(field.depths) > depth_idx else depth_idx
    rng = field.ranges[range_idx] if len(field.ranges) > range_idx else range_idx

    ax.plot(freqs, mag_db, 'b-', linewidth=1.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('|H(f)| (dB)')
    ax.set_title(f'Transfer Function — depth {depth:.0f} m, range {rng:.0f} m')
    ax.grid(True, alpha=0.3)

    model = field.metadata.get('model', '')
    if model:
        ax.set_title(f'{model} Transfer Function — depth {depth:.0f} m, range {rng:.0f} m')

    return fig, ax


def plot_time_series(
    field: Optional[Field] = None,
    time_series_data: Optional[Dict] = None,
    receiver_depths: Optional[np.ndarray] = None,
    stacked: bool = True,
    scale: float = 2.5,
    color: Optional[str] = 'black',
    figsize: Tuple[float, float] = (12, 8),
    ax: Optional[Axes] = None,
):
    """
    Plot pressure time series at receiver locations

    For time-domain model output (e.g., SPARC). Shows waveform evolution
    at one or multiple receiver depths.

    Parameters
    ----------
    field : Field, optional
        Field with time series data in metadata['time_series'].
        Either field or time_series_data must be provided.
    time_series_data : dict, optional
        Direct time series data dict with keys 'time', 'pressure', 'receiver_depth'.
        Alternative to passing field.
    receiver_depths : ndarray, optional
        Subset of receiver depths to plot. If None, plots all available.
    stacked : bool, optional
        If True, plots time series stacked with offset (default).
        If False, plots overlaid.
    scale : float, optional
        Waveform amplitude scale factor (as multiple of trace spacing).
        Larger values = larger waveforms. Default is 2.5.
        Use smaller values (0.5-1.5) if waveforms overlap.
        Use larger values (3.0-5.0) if waveforms are too small.
    color : str or None, optional
        Color for all traces. If None, uses matplotlib color cycle.
        Default is 'black'. Examples: 'red', 'blue', '#FF5733', (0.5, 0.5, 0.5).
    figsize : tuple, optional
        Figure size. Default is (12, 8).
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes

    Examples
    --------
    >>> # From SPARC model output
    >>> result = sparc.compute_tl(env, source, receiver)
    >>> fig, ax = plot_time_series(result)
    >>> plt.show()

    >>> # Plot specific depths only
    >>> fig, ax = plot_time_series(result, receiver_depths=np.array([10, 50, 90]))

    >>> # Overlaid instead of stacked
    >>> fig, ax = plot_time_series(result, stacked=False)

    >>> # Adjust waveform scale for better visibility
    >>> fig, ax = plot_time_series(result, scale=2.5)  # Larger waveforms
    >>> fig, ax = plot_time_series(result, scale=0.8)  # Smaller waveforms

    >>> # Set custom color for all traces (default is black)
    >>> fig, ax = plot_time_series(result, color='blue')
    >>> fig, ax = plot_time_series(result, color='#FF5733')  # Hex color
    >>> fig, ax = plot_time_series(result, color=None)  # Use color cycle

    >>> # From direct time series data
    >>> from uacpy.io.ts_reader import read_ts
    >>> ts_data = read_ts('timeseries.txt')
    >>> fig, ax = plot_time_series(time_series_data=ts_data)

    Notes
    -----
    This function is equivalent to MATLAB's plotts.m
    Time series data is produced by SPARC and some other time-domain models.

    See Also
    --------
    plot_arrivals : Plot discrete arrival times and amplitudes
    """
    # Extract time series data
    if field is not None:
        if field.field_type == 'time_series' and 'time' in field.metadata:
            # Direct time_series field (e.g., SPARC TIME_SERIES mode)
            ts_data = {
                'time': field.metadata['time'],
                'pressure': field.data,
                'dt': field.metadata.get('dt', None),
                'receiver_depth': field.metadata.get('receiver_depth', None),
            }
        else:
            ts_data = field.metadata.get('time_series', None)
            if ts_data is None:
                raise ValueError("Field must have 'time_series' in metadata or be field_type='time_series'.")
    elif time_series_data is not None:
        ts_data = time_series_data
    else:
        raise ValueError("Must provide either field or time_series_data")

    # Extract components from time series data
    if 'time' in ts_data and 'pressure' in ts_data:
        # SPARC format
        time = ts_data['time']
        pressure = ts_data['pressure']  # shape: (nt, nr) where nr = num_ranges

        # Check if we have receiver_depth (single) or receiver_range info
        if 'receiver_depth' in ts_data:
            receiver_depth = ts_data['receiver_depth']
        elif 'pos' in ts_data and 'r' in ts_data['pos'] and 'z' in ts_data['pos']['r']:
            rd = ts_data['pos']['r']['z']
        else:
            rd = np.array([30.0])  # Default depth

        # For SPARC single-depth runs, pressure columns are for different RANGES
        # We need to treat this properly
        if 'receiver_ranges' in ts_data:
            ranges = ts_data['receiver_ranges']
            rd = np.full(len(ranges), receiver_depth)  # Same depth, different ranges
        else:
            # Assume columns are for different depths (standard case)
            if 'receiver_depth' in ts_data:
                rd = np.array([receiver_depth])
            else:
                rd = np.arange(pressure.shape[1])  # Default indices

    elif 'tout' in ts_data and 'RTS' in ts_data:
        # ASCII TS format
        time = ts_data['tout']
        pressure = ts_data['RTS']  # shape: (nt, nrd)
        rd = ts_data.get('pos', {}).get('r', {}).get('z', np.arange(pressure.shape[1]))
    else:
        raise ValueError("Unrecognized time series data format")

    nt, nrd = pressure.shape

    # Ensure rd has correct length
    if len(rd) != nrd:
        if len(rd) == 1:
            # Single depth - extend to all traces (e.g., different ranges at same depth)
            rd = np.full(nrd, rd[0])
        else:
            # Mismatch - use indices as fallback
            rd = np.arange(nrd)

    # Select subset of depths if requested
    if receiver_depths is not None:
        indices = []
        for depth in receiver_depths:
            idx = np.argmin(np.abs(rd - depth))
            indices.append(idx)
        pressure = pressure[:, indices]
        rd = rd[indices]
        nrd = len(indices)

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Check if we should stack by range instead of depth
    stack_by_range = False
    ranges = None

    if stacked:
        # Check if all depths are the same (single depth, multiple ranges case)
        unique_depths = np.unique(rd)
        if len(unique_depths) == 1 and field is not None and hasattr(field, 'ranges'):
            # Single depth, multiple ranges - stack by range (SPARC time series case)
            if len(field.ranges) == nrd:
                stack_by_range = True
                ranges = field.ranges
        elif len(unique_depths) == 1:
            # Single depth but no range info - use indices as stack values
            stack_by_range = False
            # Create artificial spacing based on trace index
            rd = np.arange(nrd, dtype=float)

    # Always stack (never use overlaid mode when stacked=True)
    if stacked:
        # Determine what to stack by: range or depth
        if stack_by_range:
            # Stack by range (SPARC single-depth case)
            stack_values = ranges / 1000  # Convert to km
            ylabel = 'Range (km)'
            title = 'Pressure Time Series (Stacked by Range)'
            invert_axis = False
        else:
            # Stack by depth (standard case)
            stack_values = rd
            ylabel = 'Receiver Depth (m)'
            title = 'Pressure Time Series (Stacked by Depth)'
            invert_axis = True

        # Scale waveforms to be clearly visible
        # Use user-specified scale factor (as multiple of trace spacing)
        if nrd > 1:
            spacing = (stack_values[-1] - stack_values[0]) / (nrd - 1)
            # Scale so waveforms are prominent and visible
            waveform_scale = scale * spacing
        else:
            waveform_scale = stack_values[0] * scale * 0.5 if len(stack_values) > 0 else 1.0

        # Normalize by max amplitude and apply waveform scale
        amp_scale = np.max(np.abs(pressure))
        if amp_scale == 0:
            amp_scale = 1.0
        pressure_scaled = (pressure / amp_scale) * waveform_scale

        # Each trace centered at its stack value (range or depth)
        for ird in range(nrd):
            p_offset = pressure_scaled[:, ird] + stack_values[ird]

            if stack_by_range:
                trace_label = f'{stack_values[ird]:.1f} km'
            else:
                trace_label = f'{stack_values[ird]:.1f} m'

            # Determine trace color
            trace_color = color if color is not None else f'C{ird % 10}'

            # Plot the waveform
            ax.plot(time, p_offset, linewidth=0.8, color=trace_color, label=trace_label if ird < 10 else '')

            # Add a reference line
            ax.axhline(stack_values[ird], color='gray', linewidth=0.3, alpha=0.3, linestyle='--', zorder=0)

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        if invert_axis:
            ax.invert_yaxis()  # Depth increases downward

    else:
        # Overlaid plot
        for ird in range(nrd):
            depth_label = f'{rd[ird]:.1f} m' if ird < len(rd) else f'Depth {ird+1}'

            # Determine trace color
            trace_color = color if color is not None else f'C{ird % 10}'

            ax.plot(time, pressure[:, ird], linewidth=1.5, color=trace_color, label=depth_label, alpha=0.7)

        ax.set_ylabel('Pressure', fontsize=12)
        ax.set_title('Pressure Time Series', fontsize=13, fontweight='bold')
        ax.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.3)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.grid(True, alpha=0.3)

    if nrd <= 10:  # Only show legend if not too many traces
        legend_title = 'Range' if stack_by_range else 'Depth'
        ax.legend(loc='best', fontsize=9, title=legend_title)

    format_axes_professional(ax)
    plt.tight_layout()

    return fig, ax


def plot_modes_heatmap(
    modes: Field,
    mode_range: Optional[Tuple[int, int]] = None,
    figsize: Tuple[float, float] = (12, 8),
    cmap: Optional[str] = None,
    normalize: bool = True,
    ax: Optional[Axes] = None,
):
    """
    Plot all mode shapes as a heatmap (depth vs mode number)

    Shows overview of all propagating modes simultaneously, useful for
    identifying mode structure, cutoff depths, and mode coupling regions.

    Parameters
    ----------
    modes : Field
        Mode Field object from compute_modes()
    mode_range : tuple of int, optional
        (start, end) mode indices to plot. If None, plots all modes.
    figsize : tuple, optional
        Figure size. Default is (12, 8).
    cmap : str, optional
        Colormap name. Default is 'RdBu_r' (red-blue diverging).
    normalize : bool, optional
        If True, normalize each mode independently. Default is True.
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes
    cbar : Colorbar
        Colorbar object

    Examples
    --------
    >>> # Plot all modes as heatmap
    >>> modes = kraken.compute_modes(env, source, n_modes=50)
    >>> fig, ax, cbar = plot_modes_heatmap(modes)
    >>> plt.show()

    >>> # Plot subset of modes
    >>> fig, ax, cbar = plot_modes_heatmap(modes, mode_range=(0, 20))

    >>> # Without normalization (shows relative amplitude)
    >>> fig, ax, cbar = plot_modes_heatmap(modes, normalize=False)

    Notes
    -----
    This function is equivalent to the pcolor panel in MATLAB's plotmode.m
    Each mode is shown as a vertical slice, with color indicating amplitude.

    See Also
    --------
    plot_modes : Standard mode plotting with line plots
    plot_mode_functions : Detailed individual mode function plots
    """
    if modes.field_type != 'modes':
        raise ValueError(f"Expected modes field, got {modes.field_type}")

    # Centralized mode colormap via style.COLORMAPS['modes'].  We keep
    # 'RdBu_r' as a fallback only if user didn't opt into the style map.
    if cmap is None:
        cmap = COLORMAPS.get('modes', 'RdBu_r')

    # Extract mode data from Field
    phi = modes.metadata.get('phi', modes.data)
    z = modes.metadata.get('z', modes.depths)
    k = modes.metadata['k']
    M = len(k)
    freq = modes.metadata.get('frequency', modes.frequencies[0] if len(modes.frequencies) > 0 else 0.0)

    # Select mode range
    if mode_range is None:
        mode_start, mode_end = 0, M
    else:
        mode_start, mode_end = mode_range
        mode_end = min(mode_end, M)

    n_modes_plot = mode_end - mode_start

    # Extract mode subset
    phi_plot = phi[:, mode_start:mode_end].real

    # Normalize each mode if requested
    if normalize:
        for i in range(n_modes_plot):
            max_val = np.max(np.abs(phi_plot[:, i]))
            if max_val > 0:
                phi_plot[:, i] = phi_plot[:, i] / max_val

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Create mode number array for x-axis (centers)
    mode_numbers = np.arange(mode_start + 1, mode_end + 1)  # 1-indexed for display

    # Plot heatmap
    # MATLAB: pcolor(x, Modes.z, real(phi)) where phi is (n_depths, n_modes)
    if normalize:
        vmin, vmax = -1, 1
    else:
        abs_max = np.max(np.abs(phi_plot))
        vmin, vmax = -abs_max, abs_max

    # Use pcolormesh with shading='nearest' for center-aligned data
    # This avoids edge calculation issues and matches MATLAB pcolor behavior
    # phi_plot is (n_depths, n_modes) which matches (len(z), len(mode_numbers))
    im = ax.pcolormesh(mode_numbers, z, phi_plot, shading='nearest', cmap=cmap,
                      vmin=vmin, vmax=vmax)

    # Configure axes
    ax.set_xlabel('Mode Number', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_title(f'Mode Shapes Heatmap\nf = {freq:.1f} Hz, {n_modes_plot} modes',
                fontsize=13, fontweight='bold')
    # Ensure depth increases downward.  Matplotlib's default ordering for
    # pcolormesh is already "y increases upward"; we invert exactly once.
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_aspect('auto')

    # Colorbar
    cbar_label = 'Normalized Amplitude' if normalize else 'Amplitude'
    cbar = create_professional_colorbar(fig, im, ax, label=cbar_label)

    # Add grid for mode boundaries
    ax.set_xticks(mode_numbers[::max(1, n_modes_plot // 10)])
    ax.grid(True, alpha=0.2, axis='x')

    format_axes_professional(ax)
    plt.tight_layout()

    return fig, ax, cbar

def plot_reflection_coefficient(
    field: 'Field',
    figsize: Tuple[float, float] = (10, 6),
    ax: Optional[Axes] = None,
    show_magnitude: bool = True,
    show_phase: bool = False,
    show_critical_angle: bool = True,
    title: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot bottom reflection coefficient from BOUNCE

    Parameters
    ----------
    field : Field
        Field object from BOUNCE with reflection coefficient data
    figsize : tuple, optional
        Figure size. Default is (10, 6).
    ax : Axes, optional
        Matplotlib axes. If None, creates new figure.
    show_magnitude : bool, optional
        Plot magnitude |R|. Default is True.
    show_phase : bool, optional
        Plot phase angle. Default is False.
    show_critical_angle : bool, optional
        Mark critical angle with vertical line. Default is True.
    title : str, optional
        Custom title. If None, uses default title.

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes (or tuple of axes if show_phase=True)

    Examples
    --------
    >>> from uacpy.models import Bounce
    >>> bounce = Bounce()
    >>> result = bounce.run(env, source, receiver)
    >>> fig, ax = plot_reflection_coefficient(result)
    >>> plt.show()

    >>> # Plot both magnitude and phase
    >>> fig, (ax1, ax2) = plot_reflection_coefficient(result, show_phase=True)

    Notes
    -----
    - Reflection coefficient data comes from BOUNCE model
    - Angles are grazing angles (degrees from horizontal)
    - Critical angle is estimated from large changes in |R|
    - Follows Acoustic Toolbox conventions
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract reflection coefficient data from field metadata
    if field.field_type != 'reflection_coefficients':
        raise ValueError(f"Expected field_type='reflection_coefficients', got '{field.field_type}'")

    if 'theta' not in field.metadata or 'R' not in field.metadata:
        raise ValueError("Field metadata must contain 'theta' (angles) and 'R' (magnitude)")

    angles = field.metadata['theta']  # Grazing angles (degrees)
    R_mag = field.metadata['R']       # Magnitude
    R_phase = field.metadata.get('phi', None)  # Phase (radians)

    # Create figure
    if show_phase and show_magnitude:
        if ax is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        else:
            # If ax is provided but we need two plots, use subplots from ax
            fig = ax.get_figure()
            ax1 = ax
            ax2 = ax.get_figure().add_subplot(2, 1, 2, sharex=ax1)
    else:
        if ax is None:
            fig, ax1 = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
            ax1 = ax
        ax2 = None

    # Plot magnitude
    if show_magnitude:
        ax1.plot(angles, R_mag, 'b-', linewidth=2.5, label='|R|')
        ax1.set_ylabel('Reflection Coefficient Magnitude |R|', fontsize=11, fontweight='bold')
        ax1.set_ylim([0, 1.1])
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10, loc='best')

        # Mark critical angle if requested
        if show_critical_angle and len(R_mag) > 10:
            # Find critical angle: large change in |R|
            dR = np.diff(R_mag)
            critical_idx = np.where(np.abs(dR) > 0.05)[0]
            if len(critical_idx) > 0:
                crit_angle = angles[critical_idx[0]]
                ax1.axvline(crit_angle, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
                ax1.text(crit_angle + 2, 0.5, f'Critical angle\n≈{crit_angle:.1f}°',
                        fontsize=9, color='red', fontweight='bold')

        if not show_phase:
            ax1.set_xlabel('Grazing Angle (degrees)', fontsize=11, fontweight='bold')

        # Title
        if title is None:
            if 'brc_file' in field.metadata:
                import os
                filename = os.path.basename(field.metadata['brc_file'])
                title = f'Bottom Reflection Coefficient\n{filename}'
            else:
                title = 'Bottom Reflection Coefficient (BOUNCE)'
        ax1.set_title(title, fontsize=12, fontweight='bold')

    # Plot phase
    if show_phase and R_phase is not None:
        # Convert phase from radians to degrees
        phase_deg = np.degrees(R_phase)
        ax2.plot(angles, phase_deg, 'g-', linewidth=2.5, label='Phase')
        ax2.set_xlabel('Grazing Angle (degrees)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Phase (degrees)', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10, loc='best')

        # Mark critical angle on phase plot too
        if show_critical_angle and len(R_mag) > 10:
            critical_idx = np.where(np.abs(np.diff(R_mag)) > 0.05)[0]
            if len(critical_idx) > 0:
                crit_angle = angles[critical_idx[0]]
                ax2.axvline(crit_angle, color='r', linestyle='--', linewidth=1.5, alpha=0.7)

    plt.tight_layout()

    if show_phase and show_magnitude:
        return fig, (ax1, ax2)
    else:
        return fig, ax1
