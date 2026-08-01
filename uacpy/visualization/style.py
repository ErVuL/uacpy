"""Shared colour scheme for uacpy plots: field colormaps, marker styles and
the sediment palette. Importing this module does not touch ``rcParams``."""


# Professional color schemes
COLORMAPS = {
    'tl': 'jet_r',             # Transmission loss: blue (low TL/good) → red (high TL/poor)
                               # Matches Acoustic Toolbox standard: flipud(jet)
    'pressure': 'seismic',     # Pressure field
}


# ── Sediment colour — single source of truth ─────────────────────────────
# Change ``BOTTOM_HALFSPACE_COLOR`` and the fill + hatch colours below
# follow automatically (they're alpha-blended derivations of it). The
# rendered fill is opaque so it cleanly covers the water heatmap
# underneath without translucent bleed-through.
BOTTOM_HALFSPACE_COLOR = 'saddlebrown'
BOTTOM_FILL_HATCH = '///'


def _blend(a, b, t):
    """Alpha-blend ``a`` over ``b`` (RGB tuples or named colours), where
    ``t`` is the weight of ``a``."""
    import matplotlib.colors as mc
    ra, ga, ba = mc.to_rgb(a)
    rb, gb, bb = mc.to_rgb(b)
    return (
        ra * t + rb * (1 - t),
        ga * t + gb * (1 - t),
        ba * t + bb * (1 - t),
    )


# Sandy-tan fill ≈ saddlebrown × 0.35 + white × 0.65
BOTTOM_FILL_COLOR = _blend(BOTTOM_HALFSPACE_COLOR, 'white', 0.35)
# Brownish-grey hatch ≈ black × 0.35 + fill × 0.65
BOTTOM_HATCH_COLOR = _blend('black', BOTTOM_FILL_COLOR, 0.35)
# facecolor (NOT color): a ``color=`` entry makes mpl draw the hatch in the
# fill colour, hiding the '///'; ``facecolor`` leaves the hatch in ``edgecolor``.
BOTTOM_FILL_STYLE = {
    'facecolor': BOTTOM_FILL_COLOR,
    'hatch': BOTTOM_FILL_HATCH,
    'edgecolor': BOTTOM_HATCH_COLOR,
    'linewidth': 0.4,
}
# Plain (un-hatched) seabed fill for TL / ray data overlays — the '///'
# half-space hatch is reserved for the environment cross-section, where it
# reads as the semi-infinite substrate; over a TL heatmap it just clutters.
BOTTOM_FILL_STYLE_SOLID = {
    'facecolor': BOTTOM_FILL_COLOR,
    'edgecolor': 'none',
}

# "Ground" colormap used to shade the seabed by sound speed: light sandy tan
# (soft sediment) → a rich medium-brown (hard rock). Terrain-toned like
# ``copper`` but deliberately capped well short of black, so the hard/fast end
# reads as warm brown ground rather than a near-black band. Callers sample its
# 0.25–0.85 band, which spans light-tan → medium-brown.
def _bottom_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('uacpy_bottom', [
        (0.00, '#efe2c8'),   # very light tan
        (0.25, '#e0c49a'),   # light sand-tan  (soft sediment)
        (0.55, '#b9824c'),   # sienna / copper
        (0.85, '#835331'),   # medium-dark brown (hard rock) — not black
        (1.00, '#6a4226'),
    ])


BOTTOM_CMAP = _bottom_cmap()

# Seafloor edge styles — applied above ``BOTTOM_FILL_STYLE`` at the
# water-sediment interface. RD bathymetry traces the actual seafloor and
# is drawn solid; a flat bottom is an idealization and is drawn dashed.
BOTTOM_LINE_STYLE = {
    'color': 'k',
    'linewidth': 2.0,
    'linestyle': '-',
}
BOTTOM_LINE_STYLE_FLAT = {
    'color': 'k',
    'linewidth': 2.0,
    'linestyle': '--',
}

# Source/receiver marker styles — applied via ``ax.plot(..., **STYLE)``.
SOURCE_MARKER_STYLE = {
    'marker': '*',
    'color': 'red',
    'markersize': 15,
    'markeredgecolor': 'black',
    'markeredgewidth': 0.5,
    'linestyle': 'none',
}

RECEIVER_MARKER_STYLE = {
    'marker': 'o',
    'color': 'limegreen',
    'markersize': 8,
    'markeredgecolor': 'black',
    'markeredgewidth': 0.5,
    'linestyle': 'none',
}


def get_cmap_for_field(field_type: str) -> str:
    """
    Return the colormap name associated with a field type.

    Parameters
    ----------
    field_type : str
        Field type ('tl', 'pressure', 'ssp', ...).

    Returns
    -------
    cmap : str
        Colormap name (falls back to ``'viridis'`` for unknown types).
    """
    return COLORMAPS.get(field_type, 'viridis')
