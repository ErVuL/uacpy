"""
Underwater acoustics utilities for UACPY

One module per subject, each answering a different question:

* ``seawater``   — *what does the water do*: sound speed (Mackenzie, UNESCO,
                   Del Grosso, TEOS-10), density, and the Doppler shift
                   they set
* ``boundaries`` — *what does the seabed do*: plane-wave reflection, the
                   bottom loss it makes over grazing angle, and the Pekeris
                   branch of the complex square root
* ``bubbles``    — *what does a bubble do*: Minnaert resonance, bubbly-water
                   sound speed, surface bubble loss
* ``levels``     — *what does the recording read*: volts to pascals, pressure
                   to SPL, power to dB

Every public name is re-exported here, so a caller writes
``uacpy.acoustics.soundspeed`` and never names a sub-module: those boundaries
are for whoever maintains the package.

Note
----
Physics-only helpers. Nothing here imports a model, a reader or a plotter —
only :mod:`uacpy.core.constants`, :mod:`uacpy.core.exceptions` and the warning
frames, plus :mod:`uacpy.core.materials` inside
:func:`bottom_loss_curve` for the named-sediment lookup — so any layer may
import them: the data layer
uses the seawater equations (:mod:`uacpy.data.sound_speed`,
:mod:`uacpy.data.argo`), :mod:`uacpy.core.ssp` builds profiles from
Mackenzie, :mod:`uacpy.io.modes_reader` takes :func:`pekeris_root`, and the
spectral estimators and their plotters share :func:`power_to_dB`. They are
also public API for notebooks and examples (e.g.
example_12_attenuation_models.py).

Portions are adapted from arlpy; the per-module headers carry the attribution
and uacpy/third_party/arlpy/NOTICE lists which function came from where.
"""

from uacpy.core.acoustics.seawater import (
    soundspeed,
    soundspeed_unesco,
    soundspeed_delgrosso,
    soundspeed_teos10,
    density,
    doppler,
)
from uacpy.core.acoustics.boundaries import (
    reflection_coeff,
    bottom_loss_curve,
    pekeris_root,
)
from uacpy.core.acoustics.bubbles import (
    bubble_resonance,
    bubble_surface_loss,
    bubble_soundspeed,
)
from uacpy.core.acoustics.levels import (
    pressure,
    spl,
    power_to_dB,
)

__all__ = [
    'soundspeed',
    'soundspeed_unesco',
    'soundspeed_delgrosso',
    'soundspeed_teos10',
    'density',
    'doppler',
    'reflection_coeff',
    'bottom_loss_curve',
    'bubble_resonance',
    'bubble_surface_loss',
    'bubble_soundspeed',
    'pressure',
    'spl',
    'power_to_dB',
    'pekeris_root',
]
