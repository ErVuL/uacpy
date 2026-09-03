"""I/O utilities for acoustic model file handling.

Layout:

* ``oalib_writer`` / ``oalib_reader`` — shared Acoustics Toolbox / OALIB
  formats (``.env``, ``.shd``, ``.arr``, ``.ray``, ``.ssp``, ``.flp``,
  ``.rts``, ``.ts``).
* ``modes_reader`` — Kraken normal-mode files, binary ``.mod`` and ASCII
  ``.moa`` (kept separate from ``oalib_reader`` because of size and
  self-containment).
* ``bellhop_writer`` — Bellhop-specific env writer (kept separate; Bellhop's
  run-type and beam-parameter knobs diverge from the AT family).
* ``bathy_io`` — bathymetry / altimetry / 3-D boundary blocks
  (``.bty``, ``.ati``).
* ``refl_io`` — precomputed reflection coefficients (``.brc``/``.irc``/
  ``.trc``) and source beam patterns (``.sbp``).
* ``oases_writer`` / ``oases_reader`` — OASES sub-models
  (OAST/OASN/OASR/OASP/OASS/OASSP).
* ``mpirams_writer`` / ``mpirams_reader`` — RAM mpiramS backend.
* ``ramsurf_writer`` / ``ramsurf_reader`` — Collins rams0.5 / ramsurf1.5.
* ``grn_reader`` — Scooter / SPARC Green's-function with post-processing.
* ``utils`` — shared helpers (``equally_spaced``).
* ``file_manager`` — temp-dir / tmpfs management.
* ``_fortran_helpers`` — private low-level Fortran-record helpers.

Planned 3-D support
-------------------

Five names here read and write the BELLHOP3D / FIELD3D formats:
``read_boundary_3d`` and ``write_bty_3d`` (``bathy_io``), ``read_ssp_3d``
and ``read_flp3d`` (``oalib_reader``), ``write_field3dflp``
(``oalib_writer``). **They are deliberately retained and are not dead
code.** No uacpy model runs ``bellhop3d`` or ``field3d`` yet — Bellhop's
RunType position 6 is hardwired to the 2-D blank and
``Bellhop(dimensionality='3D')`` raises — so nothing in the 2-D public API calls them; the 2-D readers refuse
3-D input and name them as what a future implementer builds on.
``uacpy/tests/test_io_restored_capabilities.py`` pins all five, so a
dead-code sweep meets that test before proposing their removal a second
time.
"""

from uacpy.io.file_manager import FileManager
from uacpy.io.oalib_reader import (
    read_shd_file, read_shd_bin, read_shd_asc,
    read_arr_file, read_ray_file,
    read_ssp_2d, read_ssp_3d,
    read_flp, read_flp3d,
    read_rts_file, rts_to_pressure, read_ts,
    read_prt,
)
from uacpy.io.oalib_writer import (
    write_ssp,
    write_header, write_absorption_block,
    write_fg_params, write_bio_layers, write_broadband_freqs,
    write_ssp_section, write_layer_sections, write_bottom_section,
    writable_layers,
    write_source_depths, write_receiver_depths, write_receiver_ranges,
    write_multi_profile_env,
    write_kraken_env_file, write_scooter_env_file, write_sparc_env_file,
    write_bounce_input_file,
    write_fieldflp, write_field3dflp,
    write_phase_speed_and_rmax,
    resolve_ssp_interp, resolve_ssp_topopt, resolve_phase_speed_bounds,
)
from uacpy.io.modes_reader import (
    read_modes, read_modes_bin, read_modes_asc, get_component,
)
from uacpy.io.bathy_io import (
    read_bathymetry, read_altimetry, read_boundary_3d,
    write_bty_file, write_bty_long_format, write_bty_3d, write_ati_file,
)
from uacpy.io.refl_io import (
    read_reflection_coefficient, read_source_beam_pattern,
    write_reflection_coefficient, write_source_beam_pattern,
    stage_reflection_file, stage_source_beam_pattern,
    dedupe_reflection_file,
)
from uacpy.io.bellhop_writer import write_bellhop_env_file
from uacpy.io.grn_reader import (
    read_grn_file, grn_to_field, grn_to_transfer_function,
    sparc_snapshot_to_field, sparc_snapshot_to_time_field,
)
from uacpy.io.utils import equally_spaced
from uacpy.io.oases_writer import (
    write_oast_input, write_oasn_input, write_oasp_input, write_oasr_input,
    write_oass_input, write_oassp_input,
)
from uacpy.io.oases_reader import (
    read_oast_tl, read_oasn_covariance, read_oasn_replicas, read_oasp_trf,
    read_oasr_reflection_coefficients,
    read_oases_rhs_header,
)
from uacpy.io.mpirams_writer import (
    write_inpe, write_ssp_file, write_bth_file, write_ranges_file,
    write_sediment_file,
)
from uacpy.io.mpirams_reader import read_psif
from uacpy.io.ramsurf_writer import write_ramin
from uacpy.io.ramsurf_reader import (
    read_tl_line, read_tl_grid, read_pcomplex_grid,
)

__all__ = [
    # File management
    "FileManager",
    # OALIB readers
    "read_shd_file", "read_shd_bin", "read_shd_asc",
    "read_arr_file", "read_ray_file",
    "read_ssp_2d", "read_ssp_3d",
    "read_flp", "read_flp3d",
    "read_rts_file", "rts_to_pressure", "read_ts",
    "read_prt",
    # Boundary auxiliary I/O
    "read_bathymetry", "read_altimetry", "read_boundary_3d",
    "read_reflection_coefficient",
    "read_source_beam_pattern",
    "dedupe_reflection_file",
    # Mode readers (Kraken)
    "read_modes", "read_modes_bin", "read_modes_asc", "get_component",
    # Scooter / SPARC outputs
    "read_grn_file",
    "grn_to_field", "grn_to_transfer_function",
    "sparc_snapshot_to_field", "sparc_snapshot_to_time_field",
    # OASES outputs
    "read_oast_tl", "read_oasn_covariance", "read_oasn_replicas",
    "read_oasp_trf", "read_oasr_reflection_coefficients",
    "read_oases_rhs_header",
    # mpiramS outputs
    "read_psif",
    # ramsurf / rams (Collins) outputs
    "read_tl_line", "read_tl_grid", "read_pcomplex_grid",
    # OALIB writers
    "write_ssp",
    "write_header", "write_absorption_block",
    "write_fg_params", "write_bio_layers", "write_broadband_freqs",
    "write_ssp_section", "write_layer_sections", "write_bottom_section",
    "writable_layers",
    "write_source_depths", "write_receiver_depths", "write_receiver_ranges",
    "write_multi_profile_env",
    "write_kraken_env_file", "write_scooter_env_file", "write_sparc_env_file",
    "write_bounce_input_file",
    "write_fieldflp", "write_field3dflp",
    "write_phase_speed_and_rmax",
    "resolve_ssp_interp", "resolve_ssp_topopt", "resolve_phase_speed_bounds",
    # Boundary auxiliary writers
    "write_bty_file", "write_bty_long_format", "write_bty_3d",
    "write_ati_file",
    "write_reflection_coefficient", "write_source_beam_pattern",
    "stage_reflection_file", "stage_source_beam_pattern",
    # Bellhop writer
    "write_bellhop_env_file",
    # OASES writers
    "write_oast_input", "write_oasn_input", "write_oasp_input",
    "write_oasr_input", "write_oass_input", "write_oassp_input",
    # mpiramS writers
    "write_inpe", "write_ssp_file", "write_bth_file", "write_ranges_file",
    "write_sediment_file",
    # ramsurf writer
    "write_ramin",
    # Utilities
    "equally_spaced",
    # Submodules (importing uacpy.io makes each reachable as an attribute)
    "bathy_io", "bellhop_writer", "file_manager", "grn_reader",
    "modes_reader", "mpirams_reader", "mpirams_writer",
    "oalib_reader", "oalib_writer", "oases_reader", "oases_writer",
    "ramsurf_reader", "ramsurf_writer", "refl_io", "utils",
]
