"""I/O utilities for acoustic model file handling.

Layout:

* ``oalib_writer`` / ``oalib_reader`` — shared Acoustics Toolbox / OALIB
  formats (``.env``, ``.shd``, ``.arr``, ``.ray``, ``.ssp``, ``.flp``,
  ``.rts``, ``.ts``).
* ``modes_reader`` — Kraken normal-mode files, binary ``.mod`` and ASCII
  ``.moa``, read into the ``Modes`` carrier (kept separate from
  ``oalib_reader`` because of size and self-containment).
* ``env_reader`` — an Acoustics-Toolbox ``.env`` deck back into
  ``Environment``, ``Source``, ``Receiver`` and solver options.
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
* ``grn_reader`` — Scooter / SPARC Green's-function files, read into the
  ``GreensFunction`` result.
* ``audio_io`` — ``.wav`` output for a computed or measured signal.
* ``input_checks`` — what a reader or writer checks before it trusts its
  input (``reject_unknown_kwargs``, ``_collapsed_pair_index``).
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
``uacpy/tests/test_io_public_names_without_callers.py`` pins all five, so a
dead-code sweep meets that test before proposing their removal a second
time.
"""

from uacpy.io.oalib_reader import (
    RtsFile,
    ShdFile,
    Ssp3dFile,
    SspTable,
    read_shd_file,
    read_shd_bin,
    read_shd_asc,
    read_arr_file,
    read_ray_file,
    read_ssp_2d,
    read_ssp_3d,
    FlpFile,
    Flp3dFile,
    read_flp,
    read_flp3d,
    read_rts_file,
    read_ts,
    read_prt,
)
from uacpy.io.oalib_writer import (
    write_ssp,
    write_multi_profile_env,
    write_kraken_env_file,
    write_scooter_env_file,
    write_sparc_env_file,
    write_sparc_source_time_series,
    write_bounce_input_file,
    write_fieldflp,
    write_field3dflp,
)
from uacpy.io.modes_reader import (
    read_modes,
)
from uacpy.io.bathy_io import (
    BoundaryTable, read_bathymetry, read_altimetry, read_boundary_3d,
    write_bty_file, write_bty_long_format, write_bty_3d, write_ati_file,
)
from uacpy.io.refl_io import (
    read_reflection_coefficient,
    read_source_beam_pattern,
    write_reflection_coefficient,
    write_source_beam_pattern,
)
from uacpy.io.bellhop_writer import write_bellhop_env_file
from uacpy.io.grn_reader import read_grn_file
from uacpy.io.oases_writer import (
    OasnNoise, OasnReplicaGrid,
    write_oast_input, write_oasn_input, write_oasp_input, write_oasr_input,
    write_oass_input, write_oassp_input,
)
from uacpy.io.oases_reader import (
    read_oast_tl, read_oasn_covariance, read_oasn_replicas, read_oasp_trf,
    read_oasr_reflection_coefficients,
    read_oases_rhs_header,
    OasesRhsHeader,
)
from uacpy.io.mpirams_writer import (
    write_inpe, write_ssp_file, write_bth_file, write_ranges_file,
    write_sediment_file,
    write_water_attenuation_file,
)
from uacpy.io.mpirams_reader import PsifFile, read_psif
from uacpy.io.ramsurf_writer import write_ramin
from uacpy.io.ramsurf_reader import (
    PeGrid, read_tl_line, read_tl_grid, read_pcomplex_grid,
)
from uacpy.io.audio_io import read_wav, read_wav_metadata, write_wav
from uacpy.io.env_reader import read_env

__all__ = [
    # OALIB readers
    "read_shd_file", "read_shd_bin", "read_shd_asc", "ShdFile",
    "read_arr_file", "read_ray_file",
    "read_ssp_2d", "SspTable", "read_ssp_3d", "Ssp3dFile",
    "read_flp", "read_flp3d", "FlpFile", "Flp3dFile",
    "read_rts_file", "read_ts", "RtsFile",
    "read_prt",
    # Boundary auxiliary I/O
    "read_bathymetry", "read_altimetry", "read_boundary_3d", "BoundaryTable",
    "read_reflection_coefficient",
    "read_source_beam_pattern",
    # Mode readers (Kraken)
    "read_modes",
    # AT .env decks back into carriers
    "read_env",
    # Scooter / SPARC outputs
    "read_grn_file",
    # OASES outputs
    "read_oast_tl", "read_oasn_covariance", "read_oasn_replicas",
    "read_oasp_trf", "read_oasr_reflection_coefficients",
    "read_oases_rhs_header", "OasesRhsHeader",
    # mpiramS outputs
    "read_psif", "PsifFile",
    # ramsurf / rams (Collins) outputs
    "read_tl_line", "read_tl_grid", "read_pcomplex_grid", "PeGrid",
    # OALIB writers
    "write_ssp",
    "write_multi_profile_env",
    "write_kraken_env_file", "write_scooter_env_file", "write_sparc_env_file",
    "write_sparc_source_time_series",
    "write_bounce_input_file",
    "write_fieldflp", "write_field3dflp",
    # Boundary auxiliary writers
    "write_bty_file", "write_bty_long_format", "write_bty_3d",
    "write_ati_file",
    "write_reflection_coefficient", "write_source_beam_pattern",
    # Bellhop writer
    "write_bellhop_env_file",
    # OASES writers
    "write_oast_input", "write_oasn_input", "write_oasp_input",
    "write_oasr_input", "write_oass_input", "write_oassp_input",
    "OasnNoise", "OasnReplicaGrid",
    # mpiramS writers
    "write_inpe", "write_ssp_file", "write_bth_file", "write_ranges_file",
    "write_sediment_file",
    "write_water_attenuation_file",
    # ramsurf writer
    "write_ramin",
    # Audio
    "write_wav", "read_wav", "read_wav_metadata",
]
