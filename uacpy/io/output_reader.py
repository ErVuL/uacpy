"""
Aggregated re-exports for acoustic model output readers.

Kept as a thin shim so older imports of the form
``from uacpy.io.output_reader import …`` continue to work; new code should
prefer the topic modules: ``oalib_reader``, ``modes_reader``, ``boundary_io``.
"""

from uacpy.io.oalib_reader import (
    read_shd_file, read_shd_bin, read_shd_asc,
    read_arr_file, read_ray_file,
    read_ssp_2d, read_ssp_3d,
)
from uacpy.io.modes_reader import (
    read_modes, read_modes_bin, read_modes_asc, get_component,
)
from uacpy.io.boundary_io import (
    read_bathymetry, read_altimetry, read_boundary_3d,
    read_reflection_coefficient, read_source_beam_pattern,
)

__all__ = [
    "read_shd_file", "read_shd_bin", "read_shd_asc",
    "read_arr_file", "read_ray_file",
    "read_ssp_2d", "read_ssp_3d",
    "read_modes", "read_modes_bin", "read_modes_asc", "get_component",
    "read_bathymetry", "read_altimetry", "read_boundary_3d",
    "read_reflection_coefficient", "read_source_beam_pattern",
]
