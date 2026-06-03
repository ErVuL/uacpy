"""System-level sonar performance: sonar equation, reverberation, detection.

Builds on uacpy's propagation outputs (TL fields) and noise spectra to assemble
active/passive sonar performance: scattering-strength laws, cell-scattering
reverberation, the sonar equation (signal excess, figure of merit, detection
range), and detection-theory thresholds.
"""

from .scattering import (
    LAMBERT_MU_DB,
    chapman_harris_surface,
    column_scattering_strength,
    lambert_bottom,
)
from .reverberation import (
    boundary_reverberation,
    total_reverberation,
    volume_reverberation,
)
from .sonar_equation import (
    active_signal_excess,
    detection_range,
    echo_level,
    figure_of_merit,
    noise_background,
    passive_signal_excess,
)
from .detection import (
    albersheim_snr,
    deflection_coefficient,
    detection_index,
    detection_threshold_energy,
    probability_of_detection,
    roc_curve,
)

from . import scattering, reverberation, sonar_equation, detection

__all__ = [
    "LAMBERT_MU_DB",
    "lambert_bottom",
    "chapman_harris_surface",
    "column_scattering_strength",
    "boundary_reverberation",
    "volume_reverberation",
    "total_reverberation",
    "echo_level",
    "noise_background",
    "passive_signal_excess",
    "active_signal_excess",
    "figure_of_merit",
    "detection_range",
    "deflection_coefficient",
    "detection_index",
    "probability_of_detection",
    "roc_curve",
    "albersheim_snr",
    "detection_threshold_energy",
    "scattering",
    "reverberation",
    "sonar_equation",
    "detection",
]
