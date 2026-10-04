"""The plotting surface: every public plotter, under one importable name.

``uacpy.plot.plot_field(...)`` after ``import uacpy`` and
``from uacpy.plot import plot_field`` both resolve here. The plotters are
defined in :mod:`uacpy.visualization.plots` and re-exported unchanged, so each
name is the same object in both places. ``import uacpy`` does not import this
module (it would load matplotlib); the first ``uacpy.plot`` attribute access
or an explicit import does.
"""

from uacpy.visualization.plots import *  # noqa: F401,F403
from uacpy.visualization.plots import __all__  # noqa: F401
