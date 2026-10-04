"""The mean-field producer run the scattering programs chain onto: OAST
for OASS, OASP for OASSP."""

from typing import NamedTuple

import numpy as np

from uacpy.models.base import StageInputs
from uacpy.core.results import Result
from uacpy.io.oases_reader import OasesRhsHeader


class _MeanFieldRun(NamedTuple):
    """What the mean-field launch of an OASSP/OASS call returns: the
    producer's own result (kept as ``components['mean_field']``), the
    header of the ``.rhs`` it wrote, and the stem its files carry."""
    result: Result
    rhs_header: OasesRhsHeader
    stem: str


def _mean_field_inputs(inputs, producer_settings, source):
    """The :class:`~uacpy.models.base.StageInputs` of the mean-field launch:
    the scatterer's work directory and environment, the producer's settings
    for this launch's source depth."""
    return StageInputs(
        work_dir=inputs.work_dir, env=inputs.env, source=source,
        receiver=inputs.receiver,
        settings=producer_settings._replace(
            source_depths=np.atleast_1d(np.asarray(source.depths,
                                                   dtype=float))))
