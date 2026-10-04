"""The cross-model level conventions a model applies to its engine's field:
the line-source reference at 1 m, and the sound speed and density at the
source."""

import numpy as np

from uacpy.core.bottom import medium_density_at


def _line_source_unit_at_1m(c_source: float, frequencies) -> np.ndarray:
    """Factor ``√k0``, ``k0 = 2πf / c(z_s)``, per frequency, that brings an
    engine's 2-D line-source field normalised as ``Σ ψψ e^{ikx} / k_x``
    (Kraken ``KrakenField/EvaluateMod.f90:36``, Scooter's ``'X'`` branch of
    ``TransformG.f90``) to UNIT AMPLITUDE AT 1 m in free space — JKPS
    §5.2.2's ``p/p0(1)`` reference, the same convention the package uses for
    a point source (``TL(1 m) = 0``). JKPS states that no line-source
    normalisation is established; this one is chosen so every engine reports
    one level. Bellhop's raw line field is ``4√π/√R`` (``influence.f90:784``,
    ``ArrMod.f90:104``) and takes ``1/(4√π)`` instead (:data:`Bellhop
    ._LINE_SOURCE_LEVEL`)."""
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    return np.sqrt(2.0 * np.pi * f / float(c_source))



def _source_sound_speed(env, source) -> float:
    """Sound speed at the (first) source depth — the ``c(z_s)`` in the
    line-source reference wavenumber ``k0 = 2πf / c(z_s)``."""
    depth = float(np.atleast_1d(np.asarray(source.depths, dtype=float))[0])
    return float(np.atleast_1d(env.ssp.sound_speed_at(depth))[0])


def _source_density(env, depth: float) -> float:
    """Density (g/cm³) of the medium at source ``depth`` in the range-0
    profile — the ``ρ(z_s)`` of the point-source modal sum
    ``p = i e^{-iπ/4} / (ρ(z_s) √(8πr)) Σ ψ(z_s) ψ(z) e^{ikr}/√k``
    (Jensen et al., *Computational Ocean Acoustics*, eq. 5.14), in the unit
    the Acoustics-Toolbox decks normalise the modes in.

    The water density in the water column (a source on the seafloor counts
    as water); below it, the seabed column's material at that sub-bottom
    depth. The rule is :func:`uacpy.core.bottom.medium_density_at`
    on the range-0 profile's stack of media, the one
    :meth:`~uacpy.core.results.Modes.modal_pressure_field` applies to the
    medium table of a mode set."""
    seafloor = float(env.bathymetry.eval(range=0.0))
    column = env.bottom.at(range=0.0)
    tops, densities = [0.0], [float(env.water_density)]
    stack_base = seafloor
    for layer in column.layers:
        tops.append(stack_base)
        densities.append(float(layer.density))
        stack_base += float(layer.thickness)
    return medium_density_at(depth, tops, densities, stack_base,
                             float(column.halfspace.density))
