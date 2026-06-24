"""System-level sonar performance: sonar equation, reverberation, detection,
and matched-field localization.

Builds on uacpy's propagation outputs (TL fields, normal modes) and noise
spectra to assemble active/passive sonar performance: scattering-strength laws,
cell-scattering reverberation, the sonar equation (signal excess, figure of
merit, detection range), and detection-theory thresholds. The ``*_field``
helpers map the sonar equation over a model TL
:class:`~uacpy.core.results.Field` — signal-excess and detection-probability
maps over ``(depth, range)``, plus the per-depth detection-range profile.

``matched_field`` adds source localization by matched-field processing:
replica vectors synthesized directly from a KRAKEN
:class:`~uacpy.core.results.Modes` set (:func:`synthesize_replica`,
:func:`replica_bank`) or assembled from the coherent pressure of *any* model
run (:func:`replica_bank_from_field`), a cross-spectral density matrix from
array snapshots (:func:`csdm`), and the Bartlett / MVDR ambiguity-surface
processors (:func:`bartlett`, :func:`mvdr`). Self-contained — KRAKEN modes or a
``Field``/``ResultStack`` plus numpy, no OASES dependency.
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
    active_signal_excess_field,
    detection_range,
    detection_range_by_depth,
    echo_level,
    figure_of_merit,
    noise_background,
    passive_signal_excess,
    passive_signal_excess_field,
    probability_of_detection_field,
)
from .detection import (
    albersheim_snr,
    deflection_coefficient,
    detection_index,
    detection_threshold_energy,
    probability_of_detection,
    roc_curve,
)
from .target_strength import (
    ts_convex,
    ts_cylinder,
    ts_ellipsoid,
    ts_plate,
    ts_sphere,
)
from .matched_field import (
    synthesize_replica,
    replica_bank,
    replica_bank_from_field,
    csdm,
    bartlett,
    mvdr,
)

from . import (
    scattering, reverberation, sonar_equation, detection, target_strength,
    matched_field,
)

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
    "passive_signal_excess_field",
    "active_signal_excess_field",
    "figure_of_merit",
    "detection_range",
    "detection_range_by_depth",
    "probability_of_detection_field",
    "deflection_coefficient",
    "detection_index",
    "probability_of_detection",
    "roc_curve",
    "albersheim_snr",
    "detection_threshold_energy",
    "ts_sphere",
    "ts_convex",
    "ts_ellipsoid",
    "ts_cylinder",
    "ts_plate",
    "synthesize_replica",
    "replica_bank",
    "replica_bank_from_field",
    "csdm",
    "bartlett",
    "mvdr",
    "scattering",
    "reverberation",
    "sonar_equation",
    "detection",
    "target_strength",
    "matched_field",
]
