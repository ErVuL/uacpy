"""I/O utilities for acoustic model file handling."""

from uacpy.io.file_manager import FileManager
from uacpy.io.env_writer import (
    write_env_file, write_ssp,
    write_reflection_coefficient, write_source_beam_pattern
)
from uacpy.io.env_reader import read_env_file, read_env_core
from uacpy.io.output_reader import read_shd_file, read_arr_file, read_ray_file, read_modes, read_modes_bin, read_modes_asc
from uacpy.io.bty_writer import write_bty_file, write_bty_3d, write_ati_file
from uacpy.io.grn_reader import read_grn_file, grn_to_field
from uacpy.io.rts_reader import read_rts_file, rts_to_tl
from uacpy.io.flp_reader import read_flp, read_flp3d
from uacpy.io.flp_writer import write_fieldflp, write_field3dflp
from uacpy.io.ts_reader import read_ts
from uacpy.io.utils import equally_spaced, crci, complex_ssp
from uacpy.io.shd_utils import merge_shd_files
from uacpy.io.oases_writer import write_oast_input, write_oasn_input, write_oasp_input, write_oasr_input
from uacpy.io.oases_reader import (
    read_oast_tl, read_oasn_covariance, read_oasn_replicas, read_oases_modes, read_oasp_trf
)
from uacpy.io.mpirams_writer import write_inpe, write_ssp_file, write_bth_file, write_ranges_file
from uacpy.io.mpirams_reader import read_psif

__all__ = [
    # File management
    "FileManager",
    # Writers
    "write_env_file",
    "write_ssp",  # Sound speed profile writer
    "write_bty_file",
    "write_bty_3d",
    "write_ati_file",  # Altimetry writer
    "write_reflection_coefficient",  # Reflection coefficient (.brc/.trc)
    "write_source_beam_pattern",  # Source beam pattern (.sbp)
    "write_fieldflp",  # Field parameters (.flp) writer
    "write_field3dflp",  # 3D field parameters writer
    # Primary readers (Field objects)
    "read_shd_file",
    "read_arr_file",
    "read_ray_file",
    # Mode readers (Kraken)
    "read_modes",  # Wrapper for binary/ASCII mode files
    "read_modes_bin",  # Binary .mod format
    "read_modes_asc",  # ASCII .moa format
    # Model-specific readers
    "read_grn_file",  # Scooter Green's functions
    "grn_to_field",  # GRN → Field transform
    "read_rts_file",  # SPARC time series (binary)
    "rts_to_tl",  # RTS → TL transform
    "read_ts",  # Generic time series (ASCII)
    # Field parameters
    "read_flp",  # 2D field parameters
    "read_flp3d",  # 3D field parameters
    "read_env_file",  # 2D env
    "read_env_core",
    # Utilities
    "equally_spaced",  # Test if vector is equally spaced
    "crci",  # Complex sound speed computation
    "complex_ssp",  # Complex SSP from real/imaginary parts
    "merge_shd_files",  # Merge multiple broadband .shd files
    # OASES I/O
    "write_oast_input",  # OAST input file writer
    "write_oasn_input",  # OASN input file writer
    "write_oasp_input",  # OASP input file writer
    "write_oasr_input",  # OASR input file writer (stub)
    "read_oast_tl",  # OAST transmission loss reader
    "read_oasn_covariance",  # OASN covariance matrix reader (.xsm)
    "read_oasn_replicas",  # OASN replica field reader (.rpo)
    "read_oases_modes",  # OASES mode file reader
    "read_oasp_trf",  # OASP transfer function reader (.trf)
    # mpiramS I/O
    "write_inpe",  # mpiramS input file writer
    "write_ssp_file",  # mpiramS SSP file writer
    "write_bth_file",  # mpiramS bathymetry file writer
    "write_ranges_file",  # mpiramS output ranges writer
    "read_psif",  # mpiramS psif.dat reader
]
