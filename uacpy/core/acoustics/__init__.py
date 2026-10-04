"""
Underwater acoustics utilities for UACPY

One module per subject, each answering a different question:

* ``seawater``   — *what does the water do*: sound speed (Mackenzie, UNESCO,
                   Del Grosso, TEOS-10) from pressure or depth, the
                   depth <-> pressure conversion, density, and the Doppler
                   shift they set
* ``boundaries`` — *what does the seabed do*: plane-wave reflection, the
                   bottom loss it makes over grazing angle, and the Pekeris
                   branch of the complex square root
* ``bubbles``    — *what does a bubble do*: Minnaert resonance, bubbly-water
                   sound speed, surface bubble loss
* ``levels``     — *what does the recording read*: volts to pascals, pressure
                   to SPL, peak level, sound exposure level, power to dB,
                   pressure to transmission loss, the incoherent sum of levels
* ``wavenumber`` — *how does a wavenumber integral become a field*: the
                   Hankel transform, its taper and alias period
* ``modal``      — *how do modes become a field*: modal attenuation, mode
                   shapes at given depths and the normal-mode sum on arrays
                   (asymptotic, or the exact Hankel sum)
* ``arrays``     — *what does a line array radiate*: the free-field array
                   factor and a tabulated element's directivity
* ``ray_geometry`` — *how close does a ray pass*: the miss distance of a ray
                   polyline to a receiver
* ``attenuation`` — *what does the water absorb*: Thorp, Francois–Garrison and
                   the biological resonance in dB/km, the pH scale they need,
                   and the conversion between attenuation units

Every public name is re-exported here (and again by the root module
:mod:`uacpy.acoustics`), so a caller writes
``uacpy.acoustics.sound_speed_mackenzie`` and never names a sub-module: those boundaries
are for whoever maintains the package.

Note
----
Physics-only helpers. Nothing here imports a model, a reader or a plotter —
only core modules below them: the constants, exceptions, validators and
warning frames; the boundary and seabed carriers that ``modal`` reads a
half-space from; :mod:`uacpy.core.materials` inside
:func:`bottom_loss_curve` for the named-sediment lookup — so any layer may
import them: the data layer
uses the seawater equations (:mod:`uacpy.data.sound_speed`,
:mod:`uacpy.data.argo`), :mod:`uacpy.core.ssp` builds profiles through
:func:`sound_speed_at_depth` (TEOS-10 by default),
:mod:`uacpy.io.modes_reader` takes :func:`pekeris_root`, and the
spectral estimators and their plotters share :func:`power_to_dB`. They are
also public API for notebooks and examples (e.g.
example_12_attenuation_models.py).

Portions are adapted from arlpy; the per-module headers carry the attribution
and uacpy/third_party/arlpy/NOTICE lists which function came from where.
"""

from uacpy.core.acoustics.seawater import (
    sound_speed_mackenzie,
    sound_speed_unesco,
    sound_speed_delgrosso,
    sound_speed_teos10,
    sound_speed_at_depth,
    depth_to_pressure_dbar,
    pressure_dbar_to_depth,
    insitu_from_potential,
    density,
    doppler,
)
from uacpy.core.acoustics.boundaries import (
    reflection_coeff,
    bottom_loss_curve,
    critical_angle,
    pekeris_root,
)
from uacpy.core.acoustics.bubbles import (
    bubble_resonance,
    bubble_surface_loss,
    bubble_sound_speed,
)
from uacpy.core.acoustics.wavenumber import (
    alias_period,
    hankel_transform,
    ranges_fit_alias_period,
    snapshot_frequency_component,
    wavenumber_taper,
    wavenumbers_from_phase_speeds,
)
from uacpy.core.acoustics.attenuation import (
    absorption_thorp,
    absorption_francois_garrison,
    absorption_biological,
    ph_to_nbs,
    convert_attenuation_units,
)
from uacpy.core.acoustics.arrays import (
    array_factor,
    element_directivity,
)
from uacpy.core.acoustics.ray_geometry import polyline_miss_distance
from uacpy.core.acoustics.modal import (
    modal_attenuation,
    modal_excitation,
    modal_field,
    modal_grazing_angles,
    modal_phase_speeds,
    mode_shapes_at,
)
from uacpy.core.acoustics.levels import (
    pressure,
    spl,
    power_to_dB,
    peak_level,
    transmission_loss_dB,
    received_level_dB,
    sound_exposure_level,
    no_energy_mask,
    sum_levels_dB,
    integrate_psd,
    band_level,
)

__all__ = [
    'sound_speed_mackenzie',
    'sound_speed_unesco',
    'sound_speed_delgrosso',
    'sound_speed_teos10',
    'sound_speed_at_depth',
    'depth_to_pressure_dbar',
    'pressure_dbar_to_depth',
    'insitu_from_potential',
    'density',
    'doppler',
    'reflection_coeff',
    'bottom_loss_curve',
    'critical_angle',
    'bubble_resonance',
    'bubble_surface_loss',
    'bubble_sound_speed',
    'pressure',
    'spl',
    'power_to_dB',
    'peak_level',
    'transmission_loss_dB',
    'received_level_dB',
    'no_energy_mask',
    'sum_levels_dB',
    'integrate_psd',
    'band_level',
    'alias_period',
    'hankel_transform',
    'ranges_fit_alias_period',
    'snapshot_frequency_component',
    'wavenumber_taper',
    'wavenumbers_from_phase_speeds',
    'polyline_miss_distance',
    'modal_attenuation',
    'modal_excitation',
    'modal_field',
    'modal_grazing_angles',
    'modal_phase_speeds',
    'mode_shapes_at',
    'sound_exposure_level',
    'pekeris_root',
    'absorption_thorp',
    'absorption_francois_garrison',
    'absorption_biological',
    'ph_to_nbs',
    'convert_attenuation_units',
    'array_factor',
    'element_directivity',
]
