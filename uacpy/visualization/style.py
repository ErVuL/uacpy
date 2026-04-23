"""
Professional matplotlib styling for UACPY visualizations.

Styles are applied automatically when ``uacpy.visualization`` is imported
(see ``visualization/__init__.py``). Callers can re-apply them with
``apply_professional_style()`` after tweaking their own rcParams, or
revert to matplotlib defaults with ``matplotlib.rcdefaults()``.
"""
import matplotlib as mpl
from typing import Optional


# Professional color schemes
COLORS = {
    'primary': '#2E86AB',      # Professional blue
    'secondary': '#A23B72',    # Accent purple
    'success': '#06A77D',      # Success green
    'warning': '#F18F01',      # Warning orange
    'danger': '#C73E1D',       # Danger red
    'dark': '#2D3142',         # Dark gray
    'light': '#F5F5F5',        # Light gray
    'grid': '#E0E0E0',         # Grid color
}

# Professional color palettes for model comparisons
MODEL_COLORS = {
    'Bellhop': '#2E86AB',      # Blue
    'RAM': '#06A77D',        # Green
    'Kraken': '#A23B72',       # Purple
    'KrakenField': '#8B2F97',  # Dark purple
    'Scooter': '#F18F01',      # Orange
    'SPARC': '#C73E1D',        # Red
    'OASN': '#1B998B',         # Teal
    'OASR': '#2D6A4F',         # Dark green
    'OASP': '#E76F51',         # Coral
    'OAST': '#E9C46A',         # Yellow
}

# Professional colormaps
COLORMAPS = {
    'tl': 'jet_r',             # Transmission loss: blue (low TL/good) → red (high TL/poor)
                               # Matches Acoustic Toolbox standard: flipud(jet)
    'ssp': 'RdYlBu_r',         # Sound speed profile (temperature-like)
    'pressure': 'seismic',     # Pressure field
    'phase': 'twilight',       # Phase data
    'modes': 'cividis',        # Mode shapes
    'bathymetry': 'terrain',   # Bathymetry
}


def apply_professional_style(dpi: int = 150):
    """
    Apply professional matplotlib styling globally

    Parameters
    ----------
    dpi : int, optional
        Dots per inch for figure resolution. Default is 150 (high quality).
        Use 300 for publication-quality figures.
    """
    # Reset to defaults first
    mpl.rcdefaults()

    # Professional style configuration
    style_dict = {
        # Figure settings
        'figure.dpi': dpi,
        'figure.figsize': (10, 6),
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'figure.autolayout': False,  # Use tight_layout instead

        # Axes settings
        'axes.facecolor': 'white',
        'axes.edgecolor': COLORS['dark'],
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'axes.axisbelow': True,  # Grid behind data
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'normal',
        'axes.spines.top': False,     # Remove top spine
        'axes.spines.right': False,   # Remove right spine
        'axes.prop_cycle': mpl.cycler(color=[
            COLORS['primary'], COLORS['success'], COLORS['secondary'],
            COLORS['warning'], COLORS['danger']
        ]),

        # Grid settings
        'grid.color': COLORS['grid'],
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.6,

        # Line settings
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'lines.markeredgewidth': 0.0,

        # Font settings
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
        'font.size': 11,
        'font.weight': 'normal',

        # Text settings
        'text.color': COLORS['dark'],
        'text.antialiased': True,

        # Legend settings
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.facecolor': 'white',
        'legend.edgecolor': COLORS['grid'],
        'legend.shadow': False,
        'legend.fancybox': False,

        # Tick settings
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.color': COLORS['dark'],
        'ytick.color': COLORS['dark'],

        # Saving settings
        'savefig.dpi': dpi,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.transparent': False,

        # Image settings
        'image.cmap': 'RdYlBu_r',      # Default colormap
        'image.interpolation': 'bilinear',
        'image.origin': 'upper',

        # PDF settings (for vector output)
        'pdf.fonttype': 42,            # TrueType fonts in PDF
        'ps.fonttype': 42,

        # SVG settings
        'svg.fonttype': 'none',        # Embed fonts in SVG
    }

    # Apply settings
    mpl.rcParams.update(style_dict)


def get_model_color(model_name: str) -> str:
    """
    Get professional color for a model

    Parameters
    ----------
    model_name : str
        Model name

    Returns
    -------
    color : str
        Hex color code
    """
    return MODEL_COLORS.get(model_name, COLORS['primary'])


def get_cmap_for_field(field_type: str) -> str:
    """
    Get appropriate colormap for field type

    Parameters
    ----------
    field_type : str
        Field type ('tl', 'pressure', 'ssp', etc.)

    Returns
    -------
    cmap : str
        Colormap name
    """
    return COLORMAPS.get(field_type, 'viridis')


def format_axes_professional(ax, title: Optional[str] = None,
                            xlabel: Optional[str] = None,
                            ylabel: Optional[str] = None):
    """
    Apply professional formatting to axes

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to format
    title : str, optional
        Axes title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    """
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, fontweight='normal')

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight='normal')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Thicken remaining spines
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)

    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10,
                   length=5, width=1.0, direction='out')


def create_professional_colorbar(fig, im, ax, label: str = ''):
    """
    Create a professional-looking colorbar

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object
    im : matplotlib.image.AxesImage
        Image to create colorbar for
    ax : matplotlib.axes.Axes
        Axes the image is on
    label : str, optional
        Colorbar label

    Returns
    -------
    cbar : matplotlib.colorbar.Colorbar
        Colorbar object
    """
    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(label, fontsize=11, fontweight='normal')
    cbar.ax.tick_params(labelsize=10)
    cbar.outline.set_linewidth(1.0)

    return cbar


