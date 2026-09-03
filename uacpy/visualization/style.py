"""Shared colour scheme for uacpy plots: field colormaps, marker styles and
the sediment palette. Importing this module does not touch ``rcParams``."""


# Professional color schemes
# Heatmap colormap for a Field's dB view, keyed by ``Field.kind``. A rendering
# choice only — what each quantity *is*, and how it is labelled, lives in
# uacpy/core/results/quantities.py, which must not depend on this module.
DB_VIEW_COLORMAPS = {
    # Transmission loss, Acoustics-Toolbox convention: jet_r is flipud(jet),
    # so LOW TL (loud, near) is red and HIGH TL (quiet, far) is blue —
    # measured, jet_r(0.0) = (0.5, 0, 0) and jet_r(1.0) = (0, 0, 0.5).
    'pressure': 'jet_r',
                               # Matches Acoustic Toolbox standard: flipud(jet)
    'reverberation': 'jet_r',
    'signal_excess': 'RdBu_r',  # diverging: the SE = 0 dB detection boundary is the midpoint
    'difference': 'RdBu_r',     # diverging: zero difference is the midpoint
}

# Every linear view (magnitude, real, imaginary part) of a SIGNED quantity.
LINEAR_VIEW_COLORMAP = 'seismic'

# A probability is bounded [0, 1] and unsigned, so the signed diverging map
# above cannot describe it: half of that map is unreachable and its neutral
# midpoint lands on P_D = 0. Red = lost, green = detected, on a fixed [0, 1]
# window. Shared with ``plot_detection_probability`` so a P_D field renders
# the same through the dedicated plotter and through ``Field.plot``.
PROBABILITY_COLORMAP = 'RdYlGn'
PROBABILITY_LIMITS = (0.0, 1.0)


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


def reversed_cmap(name: str) -> str:
    """The mirror of a named colormap — ``'jet_r'`` <-> ``'jet'``.

    A dB view carrying a LEVEL runs the opposite way to one carrying a LOSS:
    ``mag_db`` is ``-field.db``, the same water with the sign flipped. It
    therefore needs the same colours in the opposite order, or the loud end
    of one view is painted the colour the other reserves for silence.
    """
    return name[:-2] if name.endswith('_r') else name + '_r'


def cmap_for_field(kind: str, *, db: bool) -> str:
    """Colormap for a :class:`~uacpy.core.results.Field` heatmap.

    Parameters
    ----------
    kind : str
        The field's ``kind`` — what quantity it carries.
    db : bool
        Whether the dB view is being rendered. Every linear view shares one
        signed colormap regardless of quantity, so only the dB view varies:
        transmission loss wants the Acoustic-Toolbox reversed jet, signal
        excess a diverging map centred on its detection boundary.

    Returns
    -------
    cmap : str
        Colormap name (``'viridis'`` for a quantity with no registered map).
    """
    if not db:
        return LINEAR_VIEW_COLORMAP
    return DB_VIEW_COLORMAPS.get(kind, 'viridis')
