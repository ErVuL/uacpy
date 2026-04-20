"""
Time series file reader for acoustic models
"""

import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Tuple


def read_ts(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Read time-series file from acoustic models.

    This is a simple ASCII time series format, different from the binary
    RTS format used by SPARC. Used by some AT models for time-domain output.

    Parameters
    ----------
    filepath : str or Path
        Path to time series file

    Returns
    -------
    ts_data : dict
        Dictionary containing:
        - 'PlotTitle': str - Plot title
        - 'pos': dict with 'r': {'z': receiver depths (m)}
        - 'tout': ndarray - Time vector (s), shape (nt,)
        - 'RTS': ndarray - Time series data, shape (nt, nrd)
          RTS[it, ird] is pressure at time tout[it], depth pos['r']['z'][ird]

    Notes
    -----
    File format:
    - Line 1: Plot title
    - Line 2: nrd (number of receiver depths)
    - Line 3: rd values (receiver depths in m)
    - Following lines: time RTS[0,:] RTS[1,:] ... (nt rows)
      First column is time, remaining columns are RTS values at each depth

    This format is simpler than the binary .rts format used by SPARC.

    Translated from OALIB read_ts.m

    Examples
    --------
    >>> ts = read_ts('timeseries.txt')
    >>> print(f"Time range: {ts['tout'][0]:.3f} to {ts['tout'][-1]:.3f} s")
    >>> print(f"Receiver depths: {ts['pos']['r']['z']}")
    >>> print(f"Time series shape: {ts['RTS'].shape}")

    >>> # Plot time series at first depth
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(ts['tout'], ts['RTS'][:, 0])
    >>> plt.xlabel('Time (s)')
    >>> plt.ylabel('Pressure')
    >>> plt.title(f"Depth = {ts['pos']['r']['z'][0]} m")

    See Also
    --------
    read_rts_file : Read binary RTS format from SPARC
    """
    filepath = Path(filepath)

    # Check for .mat file (MATLAB format)
    if filepath.suffix == '.mat':
        import scipy.io
        mat_data = scipy.io.loadmat(str(filepath))
        return {
            'PlotTitle': str(mat_data.get('PlotTitle', [''])[0]),
            'pos': {'r': {'z': mat_data['Pos'][0, 0]['r'][0, 0]['z'].ravel()}},
            'tout': mat_data['tout'].ravel(),
            'RTS': mat_data['RTS'].T  # MATLAB stores transposed
        }

    # Read ASCII format
    with open(filepath, 'r') as f:
        # Read title
        plot_title = f.readline().strip()

        # Read number of receiver depths
        nrd = int(f.readline().strip())

        # Read receiver depths
        rd = np.array([float(x) for x in f.readline().strip().split()])

        if len(rd) != nrd:
            raise ValueError(f"Expected {nrd} receiver depths, got {len(rd)}")

        # Read time series data
        data = []
        for line in f:
            line = line.strip()
            if line:
                values = [float(x) for x in line.split()]
                data.append(values)

    data = np.array(data)

    # Extract time and RTS
    # Column 0 is time, columns 1:nrd+1 are RTS values
    tout = data[:, 0]
    RTS = data[:, 1:nrd+1]

    return {
        'PlotTitle': plot_title,
        'pos': {'r': {'z': rd}},
        'tout': tout,
        'RTS': RTS
    }
