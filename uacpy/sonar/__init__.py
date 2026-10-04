"""System-level sonar performance: sonar equation, reverberation, detection,
and matched-field localization.

Builds on uacpy's propagation outputs (TL fields, normal modes) and noise
spectra to assemble active/passive sonar performance: scattering-strength laws,
cell-scattering reverberation, the sonar equation (signal excess, figure of
merit, detection range, with :class:`SonarBudget` holding one budget's
terms), and detection-theory thresholds. The ``*_field``
helpers map the sonar equation over a model TL
:class:`~uacpy.core.results.Field` — signal-excess and detection-probability
maps over ``(depth, range)``, plus the per-depth detection-range profile.

``matched_field`` adds source localization by matched-field processing:
replica vectors synthesized directly from a KRAKEN
:class:`~uacpy.core.results.Modes` set (:func:`synthesize_replica`,
:func:`replica_bank`) or assembled from the coherent pressure of *any* model
run (:func:`replica_bank_from_field`), a cross-spectral density matrix from
array snapshots (:func:`csdm`), and the Bartlett / MVDR ambiguity-surface
processors (:func:`bartlett`, :func:`mvdr`), which return the surface as an
ambiguity :class:`~uacpy.core.results.Field`. Self-contained — KRAKEN modes
or a ``Field``/``ResultStack`` plus numpy, no OASES dependency.
"""

from .bottom_scattering import (
    BottomParameters,
    APL_UW_SEDIMENTS,
    apl_uw_bottom_backscatter,
    apl_uw_bottom_loss,
)
from .scattering import (
    LAMBERT_MU_DB,
    apl_uw_surface_backscatter,
    chapman_harris_surface,
    coherent_reflection_factor,
    column_scattering_strength,
    lambert_bottom,
    perturbative_grazing_limit,
    rayleigh_parameter,
)
from .reverberation import (
    boundary_reverberation,
    total_reverberation,
    volume_reverberation,
)
from .sonar_equation import (
    SonarBudget,
    active_signal_excess,
    active_signal_excess_field,
    detection_annuli,
    detection_range,
    detection_ranges_by_depth,
    detection_range_from_field,
    detection_ranges,
    echo_level,
    figure_of_merit,
    noise_background,
    passive_signal_excess,
    passive_signal_excess_field,
    transition_probability_field,
    transition_probability,
)
from .detection import (
    albersheim_snr,
    deflection_coefficient,
    detection_index,
    detection_threshold_energy,
    per_look_false_alarm,
    scan_false_alarm,
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

# Imported so each submodule is reachable as an attribute; not in __all__.
from . import (  # noqa: F401
    scattering, bottom_scattering, reverberation, sonar_equation, detection,
    target_strength, matched_field,
)

__all__ = [
    "LAMBERT_MU_DB",
    "lambert_bottom",
    "chapman_harris_surface",
    "column_scattering_strength",
    "rayleigh_parameter",
    "coherent_reflection_factor",
    "perturbative_grazing_limit",
    "BottomParameters",
    "APL_UW_SEDIMENTS",
    "apl_uw_bottom_backscatter",
    "apl_uw_bottom_loss",
    "apl_uw_surface_backscatter",
    "boundary_reverberation",
    "volume_reverberation",
    "total_reverberation",
    "SonarBudget",
    "echo_level",
    "noise_background",
    "passive_signal_excess",
    "active_signal_excess",
    "passive_signal_excess_field",
    "active_signal_excess_field",
    "figure_of_merit",
    "detection_annuli",
    "detection_range",
    "detection_ranges_by_depth",
    "detection_range_from_field",
    "detection_ranges",
    "transition_probability_field",
    "transition_probability",
    "deflection_coefficient",
    "detection_index",
    "probability_of_detection",
    "roc_curve",
    "albersheim_snr",
    "detection_threshold_energy",
    "per_look_false_alarm",
    "scan_false_alarm",
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
]
