"""I/O utilities for acoustic model file handling.

Layout:

* ``oalib_writer`` / ``oalib_reader`` — shared Acoustics Toolbox / OALIB
  formats (``.env``, ``.shd``, ``.arr``, ``.ray``, ``.ssp``, ``.flp``,
  ``.rts``, ``.ts``).
* ``modes_reader`` — Kraken normal-mode files (kept separate from
  ``oalib_reader`` because of size and self-containment).
* ``bellhop_writer`` — Bellhop-specific env writer (kept separate; Bellhop's
  run-type and beam-parameter knobs diverge from the AT family).
* ``boundary_io`` — auxiliary boundary-file I/O (``.bty``, ``.ati``,
  ``.brc``/``.irc``/``.trc``, ``.sbp``).
* ``oases_writer`` / ``oases_reader`` — OASES sub-models (OAST/OASN/OASR/OASP).
* ``mpirams_writer`` / ``mpirams_reader`` — RAM mpiramS backend.
* ``ramsurf_writer`` / ``ramsurf_reader`` — Collins rams0.5 / ramsurf1.5.
* ``grn_reader`` — Scooter / SPARC Green's-function with post-processing.
* ``shd_utils`` — broadband ``.shd`` merging.
* ``utils`` — shared helpers (``crci``, ``complex_ssp``, ``equally_spaced``).
* ``file_manager`` — temp-dir / tmpfs management.
* ``_fortran_helpers`` — private low-level Fortran-record helpers.
"""

from uacpy.io.file_manager import FileManager
from uacpy.io.oalib_reader import (
    read_shd_file, read_shd_bin, read_shd_asc,
    read_arr_file, read_ray_file,
    read_ssp_2d, read_ssp_3d,
    read_flp, read_flp3d,
    read_rts_file, rts_to_tl, read_ts,
)
from uacpy.io.oalib_writer import (
    get_top_bc_code, write_surface_halfspace, write_ssp,
    write_header, write_fg_params, write_bio_layers, write_broadband_freqs,
    write_ssp_section, write_layer_sections, write_bottom_section,
    write_source_depths, write_receiver_depths, write_receiver_ranges,
    write_multi_profile_env,
    write_fieldflp, write_field3dflp,
)
from uacpy.io.modes_reader import (
    read_modes, read_modes_bin, read_modes_asc, get_component,
)
from uacpy.io.boundary_io import (
    read_bathymetry, read_altimetry, read_boundary_3d,
    read_reflection_coefficient, read_source_beam_pattern,
    write_bty_file, write_bty_3d, write_ati_file,
    write_reflection_coefficient, write_source_beam_pattern,
)
from uacpy.io.bellhop_writer import write_bellhop_env_file
from uacpy.io.grn_reader import (
    read_grn_file, grn_to_field, grn_to_transfer_function,
    sparc_snapshot_to_field,
)
from uacpy.io.utils import equally_spaced, crci, complex_ssp
from uacpy.io.shd_utils import merge_shd_files
from uacpy.io.oases_writer import (
    write_oast_input, write_oasn_input, write_oasp_input, write_oasr_input,
)
from uacpy.io.oases_reader import (
    read_oast_tl, read_oasn_covariance, read_oasn_replicas, read_oasp_trf,
    read_oasr_reflection_coefficients,
)
from uacpy.io.mpirams_writer import (
    write_inpe, write_ssp_file, write_bth_file, write_ranges_file,
    write_sediment_file,
)
from uacpy.io.mpirams_reader import read_psif

__all__ = [
    # File management
    "FileManager",
    # OALIB writers
    "write_ssp",
    "write_header", "write_fg_params", "write_bio_layers",
    "write_broadband_freqs", "write_ssp_section", "write_layer_sections",
    "write_bottom_section", "write_source_depths", "write_receiver_depths",
    "write_receiver_ranges", "write_multi_profile_env",
    "write_fieldflp", "write_field3dflp",
    # Bellhop writer
    "write_bellhop_env_file",
    # Boundary auxiliary I/O
    "read_bathymetry", "read_altimetry", "read_boundary_3d",
    "read_reflection_coefficient", "read_source_beam_pattern",
    "write_bty_file", "write_bty_3d", "write_ati_file",
    "write_reflection_coefficient", "write_source_beam_pattern",
    # OALIB readers
    "read_shd_file", "read_shd_bin", "read_shd_asc",
    "read_arr_file", "read_ray_file",
    "read_ssp_2d", "read_ssp_3d",
    "read_flp", "read_flp3d",
    # Mode readers (Kraken)
    "read_modes", "read_modes_bin", "read_modes_asc", "get_component",
    # Scooter / SPARC outputs
    "read_grn_file",
    "grn_to_field",
    "grn_to_transfer_function",
    "sparc_snapshot_to_field",
    "read_rts_file", "rts_to_tl",
    "read_ts",
    # Utilities
    "equally_spaced", "crci", "complex_ssp",
    "merge_shd_files",
    # OASES I/O
    "write_oast_input", "write_oasn_input", "write_oasp_input", "write_oasr_input",
    "read_oast_tl",
    "read_oasn_covariance", "read_oasn_replicas",
    "read_oasp_trf",
    "read_oasr_reflection_coefficients",
    # mpiramS I/O
    "write_inpe", "write_ssp_file", "write_bth_file", "write_ranges_file",
    "write_sediment_file",
    "read_psif",
]
