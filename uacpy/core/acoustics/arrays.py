"""Free-field line-array physics on plain arrays.

The plane-wave replicas a beamformer correlates against
(:func:`steering_vectors`), the complex array factor a line of point sources
radiates (:func:`array_factor`), and one element's tabulated beam pattern as
a linear amplitude (:func:`element_directivity`). By the product theorem the
far field of an array of identical elements is their product;
:class:`~uacpy.core.source.Source` evaluates its own array through these, and
:mod:`uacpy.acoustic_signal` beamforms with the first. All are free-field
quantities: in a waveguide the array's effect is its modal excitation
(:meth:`~uacpy.core.results.Modes.excitation`).
"""

import numpy as np

from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.exceptions import ConfigurationError
from uacpy.core._validate import require_positive_finite_scalar

__all__ = [
    'steering_vectors', 'array_factor', 'element_directivity',
]


def steering_vectors(positions_m, angles_deg, frequency: float,
                     sound_speed: float = DEFAULT_SOUND_SPEED):
    """Unit plane-wave steering vectors for a line array.

    ``e_n(theta) = exp(-j*k*z_n*sin(theta)) / sqrt(N)`` with
    ``k = 2*pi*f/sound_speed``,
    ``theta`` measured from broadside and positive downward (the declination
    sign of ``Bellhop/bellhop.f90:453``). ``positions_m`` are element
    coordinates along the array axis (m).

    The ``-j`` is the conjugate of Acoustics-Toolbox ``planewave_rep.m:32``
    because every consumer here applies the vector in Hermitian form —
    ``e**H p`` for :func:`beamform`, ``e**H R e`` for the spectra — whereas
    ``Matlab/beamform.m:20`` applies its ``+j`` vector as ``e p``. Under AT's
    ``exp(+i*omega*t)`` convention (``KrakenField/EvaluateMod.f90:42``) an
    arrival at declination ``+theta`` carries depth phase
    ``exp(-i*k*z*sin(theta))``, so ``e**H p`` peaks at ``+theta`` only with
    this sign.

    Parameters
    ----------
    positions_m : array_like
        Element coordinates along the array axis (m).
    angles_deg : array_like
        Angles from broadside (deg), positive downward.
    frequency : float
        Frequency (Hz).
    sound_speed : float, optional
        Sound speed (m/s). Default
        :data:`~uacpy.core.constants.DEFAULT_SOUND_SPEED`.

    Returns
    -------
    ndarray
        Shape ``(n_angles, n_elements)``, unit-norm per row.
    """
    z = np.atleast_1d(np.asarray(positions_m, dtype=float))
    if z.ndim != 1:
        # np.outer flattens, so an (N, 2) coordinate array would come back as
        # a unit-norm (n_angles, 2N) manifold — the right shape for a 2N-element
        # array that does not exist. sample_covariance checks ndim for the same
        # reason.
        raise ConfigurationError(
            f"steering_vectors: positions_m must be 1-D element coordinates "
            f"along the array axis (m); got shape {z.shape}. For a planar or "
            f"volumetric array, project the coordinates onto the array axis.")
    angles = np.atleast_1d(np.asarray(angles_deg, dtype=float))
    frequency = require_positive_finite_scalar(
        frequency, "steering_vectors", "frequency", " Hz")
    sound_speed = require_positive_finite_scalar(
        sound_speed, "steering_vectors", "sound_speed", " m/s")
    k = 2.0 * np.pi * frequency / sound_speed
    phase = np.outer(np.sin(np.deg2rad(angles)), z)
    # The MINUS sign is the replica convention: this is what you correlate
    # a snapshot against, so it is the conjugate of the field a source
    # array radiates. array_factor is
    # sqrt(N) * conj(this, about the phase centre) @ w, and says so.
    e = np.exp(-1j * k * phase)
    return e / np.sqrt(z.size)


def array_factor(positions_m, weights, angles_deg, frequency: float,
                 sound_speed: float = DEFAULT_SOUND_SPEED) -> np.ndarray:
    """Complex free-field array factor of a line of point sources.

    ``AF(θ) = Σₙ wₙ·exp(i·k·(zₙ - z̄)·sin θ)``, with ``θ`` in degrees from
    the horizontal (positive downward) and ``z̄`` the **mean** of the element
    positions, the array's phase centre. It is :func:`steering_vectors` seen
    from the source::

        AF(θ) = sqrt(N) * conj(steering_vectors(z - z.mean(), θ, f, c)) @ w

    and this function is that line: the replica is the conjugate convention
    measured from zero, the factor the radiated field measured from the phase
    centre, and ``sqrt(N)`` undoes the replica's unit norm.

    Parameters
    ----------
    positions_m : array_like
        Element positions along the array axis (m), 1-D.
    weights : array_like
        Complex element weights, one per position.
    angles_deg : array_like
        Angles from the horizontal, in degrees.
    frequency : float
        Hz.
    sound_speed : float
        Reference speed (m/s) setting the wavenumber. Default 1500.

    Returns
    -------
    ndarray
        Complex ``AF(θ)``, one entry per angle.
    """
    positions = np.atleast_1d(np.asarray(positions_m, dtype=float))
    offsets = positions - positions.mean()
    replicas = steering_vectors(offsets, angles_deg, frequency, sound_speed)
    return np.sqrt(offsets.size) * np.conj(replicas) @ np.asarray(weights)


def element_directivity(beam_pattern, angles_deg) -> np.ndarray:
    """A tabulated element beam pattern as a linear amplitude at
    ``angles_deg`` — ``f(θ)`` in the product theorem.

    ``beam_pattern`` is the ``(N, 2)`` ``[angle_deg, level_dB]`` table of an
    ``.sbp`` file, or ``None`` for an omnidirectional element (ones
    everywhere). Levels are converted to amplitude (``10**(dB/20)``,
    ``beampattern.f90:59``) *before* interpolating between samples, as the
    engines read the table, so a coarsely sampled table gives the same
    numbers here as in the run.

    Parameters
    ----------
    beam_pattern : ndarray or None
        ``(N, 2)`` ``[angle_deg, level_dB]`` table, or ``None`` for an
        omnidirectional element.
    angles_deg : array_like
        Angles (deg) to evaluate at.
    """
    angles = np.atleast_1d(np.asarray(angles_deg, dtype=float))
    if beam_pattern is None:
        return np.ones(angles.shape, dtype=float)
    table = np.asarray(beam_pattern, dtype=float)
    if table.ndim != 2 or table.shape[1] != 2:
        raise ConfigurationError(
            f"element_directivity: beam_pattern must be an (N, 2) "
            f"[angle_deg, level_dB] table; got shape {table.shape}.")
    return np.interp(angles, table[:, 0],
                     np.power(10.0, table[:, 1] / 20.0))
