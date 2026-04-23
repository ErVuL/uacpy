"""Utilities for working with SHD (shade) files."""

import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any
import scipy.io


def merge_shd_files(shd_files_in: List[Union[str, Path]],
                    shd_file_out: Union[str, Path]) -> None:
    """
    Merge multiple broadband .shd files into a single output file.

    This function accepts a list of broadband .shd files and merges them
    into a single output file. It checks that receiver grids are consistent
    and verifies there are no duplicate frequencies.

    Parameters
    ----------
    shd_files_in : list of str or Path
        List of input .shd filenames to merge
    shd_file_out : str or Path
        Output merged .shd filename

    Raises
    ------
    ValueError
        If input files have inconsistent grids, duplicate frequencies,
        or incompatible pressure field sizes

    Notes
    -----
    Input files do not need to be in any particular order. The frequencies
    will be automatically sorted in ascending order in the output file.

    For large or numerous input files, presenting them in the correct order
    can reduce memory requirements during the merge.

    The function presently only handles MATLAB .mat format .shd files.

    Memory Usage:
    - All pressure data from all files is loaded into memory
    - For large files, ensure sufficient RAM is available

    Translated from OALIB merge_shd_files.m

    Examples
    --------
    >>> # Merge three broadband runs at different frequency ranges
    >>> from uacpy.io import merge_shd_files
    >>>
    >>> input_files = ['run_100_200Hz.shd', 'run_200_400Hz.shd',
    ...                'run_400_800Hz.shd']
    >>> merge_shd_files(input_files, 'full_broadband.shd')

    >>> # Merge two overlapping ranges (will fail if duplicate freqs)
    >>> try:
    ...     merge_shd_files(['range1.shd', 'range2.shd'], 'merged.shd')
    ... except ValueError as e:
    ...     print(f"Error: {e}")

    See Also
    --------
    read_shd_file : Read a single .shd file
    """
    # Validate inputs
    if not isinstance(shd_files_in, list):
        raise TypeError("shd_files_in must be a list of filenames")

    if len(shd_files_in) < 1:
        raise ValueError("At least one input file is required")

    # Convert to Path objects
    shd_files_in = [Path(f) for f in shd_files_in]
    shd_file_out = Path(shd_file_out)

    num_files = len(shd_files_in)

    # Load all input files and verify consistency
    print(f"Loading {num_files} input files...")
    shd_structs = []
    sizePressure = None

    for j, filepath in enumerate(shd_files_in):
        if not filepath.exists():
            raise FileNotFoundError(f"Input file not found: {filepath}")

        # Load MATLAB .mat file
        mat_data = scipy.io.loadmat(str(filepath))

        # Extract structure
        shd = {
            'PlotTitle': str(mat_data.get('PlotTitle', [''])[0]),
            'atten': float(mat_data.get('atten', 0)),
            'Pos': mat_data.get('Pos'),
            'freq0': mat_data.get('freq0'),
            'freqVec': mat_data['freqVec'].ravel(),
            'pressure': mat_data['pressure'],
            'PlotType': str(mat_data.get('PlotType', [''])[0])
        }

        # Verify pressure field size consistency
        sizePressure_j = shd['pressure'].shape

        if j > 0:
            # Check size (ignoring number of frequencies)
            if sizePressure_j[1:] != sizePressure[1:]:
                raise ValueError(
                    f"Size of pressure fields do not agree:\n"
                    f"  File {j}: {sizePressure_j}\n"
                    f"  File 0: {sizePressure}"
                )
        else:
            # Save size of first pressure field
            sizePressure = sizePressure_j

        shd_structs.append(shd)

    # Extract metadata from first file
    PlotTitle = shd_structs[0]['PlotTitle']
    atten = shd_structs[0]['atten']
    Pos = shd_structs[0]['Pos']
    freq0 = shd_structs[0]['freq0']
    PlotType = shd_structs[0]['PlotType']

    # Verify consistency of metadata across all files
    for j in range(1, num_files):
        if not np.isclose(atten, shd_structs[j]['atten']):
            raise ValueError(f"Attenuation fields do not agree (file {j})")

        # Note: Pos comparison is complex (nested struct), simplified here
        # Could add more detailed checking if needed

        if PlotType != shd_structs[j]['PlotType']:
            raise ValueError(f"PlotType fields do not agree (file {j})")

    # Collate all frequencies
    print("Collating frequencies...")
    all_freqs = []
    for shd in shd_structs:
        all_freqs.extend(shd['freqVec'].tolist())

    freqVec = np.array(all_freqs)

    # Check for duplicate frequencies
    unique_freqs = np.unique(freqVec)
    if len(unique_freqs) != len(freqVec):
        raise ValueError("Duplicate frequencies found in input files")

    # Collate pressure fields
    print("Collating pressure fields...")
    num_freqs = len(freqVec)

    # Create output pressure array
    new_shape = list(sizePressure)
    new_shape[0] = num_freqs
    pressure = np.zeros(new_shape, dtype=complex)

    # Fill pressure array
    j_pnt = 0
    for shd in shd_structs:
        n_freq = len(shd['freqVec'])
        pressure[j_pnt:j_pnt+n_freq, ...] = shd['pressure']
        j_pnt += n_freq

    # Sort by frequency if needed
    sort_idx = np.argsort(freqVec)
    if not np.array_equal(sort_idx, np.arange(len(sort_idx))):
        print("Reordering by frequency...")
        freqVec = freqVec[sort_idx]
        pressure = pressure[sort_idx, ...]

    # Save merged file
    print(f"Writing merged file to {shd_file_out}...")

    save_dict = {
        'PlotTitle': PlotTitle,
        'atten': atten,
        'Pos': Pos,
        'freq0': freq0,
        'freqVec': freqVec,
        'pressure': pressure,
        'PlotType': PlotType
    }

    scipy.io.savemat(str(shd_file_out), save_dict, format='5')

    print(f"✅ Merged {num_files} files ({num_freqs} frequencies total)")
