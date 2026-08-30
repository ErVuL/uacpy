"""Shared plumbing for the documentation figure generators.

Every page's figures come from one module in this package. A module exposes
``FIGURES``: a mapping of ``figure stem -> zero-argument callable`` returning a
matplotlib ``Figure``. The driver (``docs/generate_model_figures.py``) imports
each module, calls the callables and writes the PNGs, so a page's figures are
always reproducible from committed code.

Scenarios live here rather than in the per-page modules so that pages compare
like with like: the same shallow-water channel drawn by Bellhop, Kraken and RAM
is genuinely the same environment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FIG_DIR = Path(__file__).resolve().parent.parent / 'models' / 'figures'
GUIDE_FIG_DIR = Path(__file__).resolve().parent.parent / 'guide' / 'figures'

# Figures are read inline on GitHub at ~800 px wide; 110 dpi on a 9-inch
# canvas lands close to that without bloating the repo.
DPI = 110
WIDE = (9.0, 4.2)
SQUARE = (6.4, 5.2)
TALL = (7.0, 6.4)


def save(fig, stem: str, *, guide: bool = False) -> Path:
    """Write ``fig`` as ``<stem>.png`` and close it. Returns the path."""
    import matplotlib.pyplot as plt

    out_dir = GUIDE_FIG_DIR if guide else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{stem}.png'
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


# ── Canonical scenarios ──────────────────────────────────────────────────────
# Kept small enough that regenerating every figure stays a couple of minutes,
# and physically ordinary so the plots show real structure rather than
# artefacts of a contrived setup.


def shallow_water():
    """100 m isovelocity-ish channel over a sand half-space, 200 Hz.

    The workhorse case: shallow enough for strong multipath, slow enough that
    a modal solver still resolves a handful of modes.
    """
    import uacpy

    env = uacpy.Environment(
        name='Shallow-water channel',
        bathymetry=100.0,
        ssp=[(0.0, 1500.0), (30.0, 1495.0), (100.0, 1490.0)],
        bottom=uacpy.BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1650.0, density=1.8, attenuation=0.6,
        ),
    )
    source = uacpy.Source(depths=25.0, frequencies=200.0)
    receiver = uacpy.Receiver(
        depths=np.linspace(1.0, 99.0, 100),
        ranges=np.linspace(50.0, 5000.0, 250),
    )
    return env, source, receiver


def deep_water():
    """5000 m Munk profile, 50 Hz — the deep-sound-channel case."""
    import uacpy
    from uacpy.core.ssp import SoundSpeedProfile

    env = uacpy.Environment(
        name='Deep water (Munk)',
        bathymetry=5000.0,
        ssp=SoundSpeedProfile.from_munk(5000.0),
        bottom=uacpy.BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0, density=1.8, attenuation=0.2,
        ),
    )
    source = uacpy.Source(depths=1000.0, frequencies=50.0)
    receiver = uacpy.Receiver(
        depths=np.linspace(0.0, 5000.0, 120),
        ranges=np.linspace(1000.0, 100_000.0, 300),
    )
    return env, source, receiver


def sloping_shelf():
    """Range-dependent: 100 m shelf falling to 400 m over 20 km, 100 Hz."""
    import uacpy

    env = uacpy.Environment(
        name='Continental shelf',
        bathymetry=[(0.0, 100.0), (8000.0, 120.0), (12000.0, 220.0),
                    (20000.0, 400.0)],
        ssp=[(0.0, 1520.0), (50.0, 1505.0), (200.0, 1490.0), (400.0, 1485.0)],
        bottom=uacpy.BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1700.0, density=1.9, attenuation=0.5,
        ),
    )
    source = uacpy.Source(depths=50.0, frequencies=100.0)
    receiver = uacpy.Receiver(
        depths=np.linspace(1.0, 395.0, 110),
        ranges=np.linspace(100.0, 20_000.0, 260),
    )
    return env, source, receiver


def layered_elastic():
    """Sand over a granite basement — the elastic case Bounce/OASES need."""
    import uacpy

    env = uacpy.Environment(
        name='Layered elastic seabed',
        bathymetry=100.0,
        ssp=[(0.0, 1500.0), (100.0, 1490.0)],
        bottom=uacpy.SeabedColumn.from_presets(
            layers=[('sand', 8.0)], halfspace='granite',
        ),
    )
    source = uacpy.Source(depths=25.0, frequencies=200.0)
    receiver = uacpy.Receiver(
        depths=np.linspace(1.0, 99.0, 90),
        ranges=np.linspace(50.0, 5000.0, 220),
    )
    return env, source, receiver
