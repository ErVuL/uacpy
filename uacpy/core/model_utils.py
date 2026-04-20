"""
Shared utilities for propagation models

Eliminates code duplication across model implementations.
"""

import subprocess
import shutil
import numpy as np
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

from uacpy.core.exceptions import ExecutableNotFoundError
from uacpy.models.base import RunMode


class ExecutableFinder:
    """
    Unified executable finding logic for all models

    Eliminates duplicate _find_executable() methods across models.
    """

    @staticmethod
    def find(
        name: str,
        model_name: str,
        search_subdirs: List[str] = None,
        required: bool = True
    ) -> Optional[Path]:
        """
        Find executable in standard locations

        Parameters
        ----------
        name : str
            Executable name (e.g., 'bellhop.exe', 'oasn', 'kraken.exe')
        model_name : str
            Model name for error messages
        search_subdirs : list of str, optional
            Subdirectories to search in uacpy/bin/
            Default: ['oalib', 'bellhopcuda', 'oases']
        required : bool
            If True, raises error if not found. If False, returns None.

        Returns
        -------
        path : Path or None
            Path to executable, or None if not found and not required

        Raises
        ------
        ExecutableNotFoundError
            If executable not found and required=True
        """
        if search_subdirs is None:
            search_subdirs = ['oalib', 'bellhopcuda', 'oases']

        base_dir = Path(__file__).parent.parent
        searched_paths = []

        # Search in uacpy/bin subdirectories
        for subdir in search_subdirs:
            search_path = base_dir / 'bin' / subdir / name
            searched_paths.append(str(search_path))
            if search_path.exists():
                return search_path

        # Search in development locations (third_party/)
        dev_locations = [
            base_dir / 'third_party' / 'oases' / 'bin' / name,
            base_dir / 'third_party' / 'Acoustics-Toolbox' / 'bin' / name,
        ]
        for dev_path in dev_locations:
            searched_paths.append(str(dev_path))
            if dev_path.exists():
                return dev_path

        # Search in system PATH using shutil.which (cross-platform)
        exe_in_path = shutil.which(name)
        if exe_in_path:
            return Path(exe_in_path)

        # Fallback: try subprocess which (Unix-like systems)
        try:
            result = subprocess.run(
                ['which', name],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                path_str = result.stdout.strip()
                if path_str:
                    return Path(path_str)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Not found
        if required:
            raise ExecutableNotFoundError(model_name, name, searched_paths)
        return None

    @staticmethod
    def find_oalib(name: str, model_name: str, required: bool = True) -> Optional[Path]:
        """Find an Acoustics Toolbox (OALIB) executable.

        Searches the ``bin/oalib`` subdirectory.

        Parameters
        ----------
        name : str
            Executable name (e.g., 'bellhop.exe', 'kraken.exe').
        model_name : str
            Human-readable model name for error messages.
        required : bool
            If True, raise ExecutableNotFoundError when not found.

        Returns
        -------
        Path or None
            Path to executable, or None if not found and not required.
        """
        return ExecutableFinder.find(
            name, model_name,
            search_subdirs=['oalib'],
            required=required
        )

    @staticmethod
    def find_bellhopcuda(name: str, model_name: str, required: bool = True) -> Optional[Path]:
        """Find a BellhopCUDA executable.

        Searches the ``bin/bellhopcuda`` subdirectory.

        Parameters
        ----------
        name : str
            Executable name (e.g., 'bellhopcuda').
        model_name : str
            Human-readable model name for error messages.
        required : bool
            If True, raise ExecutableNotFoundError when not found.

        Returns
        -------
        Path or None
            Path to executable, or None if not found and not required.
        """
        return ExecutableFinder.find(
            name, model_name,
            search_subdirs=['bellhopcuda'],
            required=required
        )

    @staticmethod
    def find_oases(name: str, model_name: str, required: bool = True) -> Optional[Path]:
        """Find an OASES executable.

        Searches the ``bin/oases`` and ``bin/oalib`` subdirectories.

        Parameters
        ----------
        name : str
            Executable name (e.g., 'oast', 'oasn').
        model_name : str
            Human-readable model name for error messages.
        required : bool
            If True, raise ExecutableNotFoundError when not found.

        Returns
        -------
        Path or None
            Path to executable, or None if not found and not required.
        """
        return ExecutableFinder.find(
            name, model_name,
            search_subdirs=['oases', 'oalib'],
            required=required
        )


class ParameterMapper:
    """
    Maps user-friendly parameters to model-specific formats

    Provides consistent parameter naming across models.
    """

    # Volume attenuation mappings
    VOLUME_ATTEN_MAP = {
        'thorp': 'T',
        't': 'T',
        'francois': 'F',
        'francois-garrison': 'F',
        'f': 'F',
        'biological': 'B',
        'bio': 'B',
        'b': 'B',
        'none': None,
        '': None,
    }

    @classmethod
    def map_volume_attenuation(cls, value: Optional[str]) -> Optional[str]:
        """
        Standardize volume attenuation parameter

        Parameters
        ----------
        value : str or None
            User-friendly name ('thorp', 'francois', 'biological', etc.)

        Returns
        -------
        mapped : str or None
            Acoustics Toolbox format ('T', 'F', 'B', or None)

        Examples
        --------
        >>> ParameterMapper.map_volume_attenuation('thorp')
        'T'
        >>> ParameterMapper.map_volume_attenuation('francois-garrison')
        'F'
        """
        if value is None:
            return None

        normalized = value.lower().strip()
        return cls.VOLUME_ATTEN_MAP.get(normalized, value)

    @classmethod
    def map_run_mode_to_bellhop(cls, run_mode: Union[RunMode, str]) -> str:
        """
        Map RunMode enum to Bellhop run_type

        Parameters
        ----------
        run_mode : RunMode or str
            Run mode (enum or legacy string)

        Returns
        -------
        run_type : str
            Bellhop run_type ('C', 'I', 'S', 'R', 'E', 'A')
        """
        if isinstance(run_mode, str):
            # Legacy support - return as-is
            return run_mode.upper()

        mapping = {
            RunMode.COHERENT_TL: 'C',
            RunMode.INCOHERENT_TL: 'I',
            RunMode.SEMICOHERENT_TL: 'S',
            RunMode.RAYS: 'R',
            RunMode.EIGENRAYS: 'E',
            RunMode.ARRIVALS: 'A',
            RunMode.TIME_SERIES: 'A',  # arrivals are the input for time-series synthesis
        }
        return mapping.get(run_mode, 'C')


class ReceiverGridBuilder:
    """
    Builds standard receiver grids for common scenarios

    Eliminates need for users to manually create receiver grids.
    """

    @staticmethod
    def build_tl_grid(
        env_depth: float,
        max_range: float,
        n_depths: int = 50,
        n_ranges: int = 100,
        depth_margin: float = 5.0
    ):
        """
        Build standard TL computation grid

        Parameters
        ----------
        env_depth : float
            Environment depth in meters
        max_range : float
            Maximum range in meters
        n_depths : int
            Number of depth points
        n_ranges : int
            Number of range points
        depth_margin : float
            Margin from surface/bottom in meters

        Returns
        -------
        depths : ndarray
            Depth grid
        ranges : ndarray
            Range grid
        """
        depths = np.linspace(
            depth_margin,
            env_depth - depth_margin,
            n_depths
        )

        # Start at 1% of max range (avoid zero range issues)
        ranges = np.linspace(
            max(max_range * 0.01, 10.0),  # At least 10m
            max_range,
            n_ranges
        )

        return depths, ranges

    @staticmethod
    def build_ray_grid(env_depth: float, max_range: float):
        """
        Build grid suitable for ray visualization.

        Parameters
        ----------
        env_depth : float
            Maximum environment depth in meters.
        max_range : float
            Maximum range in meters.

        Returns
        -------
        depths : ndarray
            Depth grid (200 points, high resolution).
        ranges : ndarray
            Range grid (200 points, high resolution).
        """
        depths = np.linspace(0, env_depth, 200)
        ranges = np.linspace(0, max_range, 200)
        return depths, ranges


class SSPInterpolationMapper:
    """
    Maps SSP types to Acoustics Toolbox interpolation codes

    Provides consistent SSP interpolation specification.
    """

    # Maps user-friendly SSP types to AT interpolation codes
    # AT codes: N=N2-Linear, C=C-Linear, P=PCHIP, S=Spline, Q=Quad, A=Analytic
    SSP_MAP = {
        # Profile types → default to C-Linear
        'isovelocity': 'C',
        'munk': 'C',
        'linear': 'C',
        'bilinear': 'C',
        # Explicit interpolation types
        'n2linear': 'N',
        'n2-linear': 'N',
        'c-linear': 'C',
        'clinear': 'C',
        'clin': 'C',
        'pchip': 'P',
        'spline': 'S',
        'cubic': 'S',
        'quad': 'Q',
        'analytic': 'A',
    }

    @classmethod
    def get_interpolation_code(cls, ssp_type: str) -> str:
        """
        Get Acoustics Toolbox interpolation code

        Parameters
        ----------
        ssp_type : str
            SSP type from Environment

        Returns
        -------
        code : str
            AT interpolation code ('C', 'N', 'P', 'S', 'Q', 'A')
        """
        normalized = ssp_type.lower().strip()
        return cls.SSP_MAP.get(normalized, 'C')  # Default to C-Linear


class BoundaryTypeMapper:
    """
    Maps boundary types to model-specific codes

    Different models use different boundary specification conventions.
    """

    # Acoustics Toolbox boundary types
    AT_SURFACE_MAP = {
        'vacuum': 'V',
        'v': 'V',
        'rigid': 'R',
        'r': 'R',
        'acousto-elastic': 'A',
        'half-space': 'A',
        'halfspace': 'A',
        'a': 'A',
    }

    AT_BOTTOM_MAP = {
        'vacuum': 'V',
        'v': 'V',
        'rigid': 'R',
        'r': 'R',
        'acousto-elastic': 'A',
        'half-space': 'A',
        'halfspace': 'A',
        'a': 'A',
    }

    @classmethod
    def get_surface_code(cls, boundary_type: str, model: str = 'AT') -> str:
        """Get Acoustics Toolbox surface boundary code.

        Parameters
        ----------
        boundary_type : str
            Boundary type string (e.g., 'vacuum', 'rigid', 'halfspace').
        model : str
            Model family identifier (currently only 'AT' supported).

        Returns
        -------
        str
            Single-character AT boundary code ('V', 'R', or 'A').
            Defaults to 'V' (vacuum) if unrecognized.
        """
        normalized = boundary_type.lower().strip()
        return cls.AT_SURFACE_MAP.get(normalized, 'V')

    @classmethod
    def get_bottom_code(cls, boundary_type: str, model: str = 'AT') -> str:
        """Get Acoustics Toolbox bottom boundary code.

        Parameters
        ----------
        boundary_type : str
            Boundary type string (e.g., 'vacuum', 'rigid', 'halfspace').
        model : str
            Model family identifier (currently only 'AT' supported).

        Returns
        -------
        str
            Single-character AT boundary code ('V', 'R', or 'A').
            Defaults to 'A' (acousto-elastic halfspace) if unrecognized.
        """
        normalized = boundary_type.lower().strip()
        return cls.AT_BOTTOM_MAP.get(normalized, 'A')


def validate_source_depth(source_depth: float, env_depth: float, margin: float = 1.0):
    """
    Validate source depth against environment

    Parameters
    ----------
    source_depth : float
        Source depth in meters
    env_depth : float
        Environment depth in meters
    margin : float
        Safety margin in meters

    Raises
    ------
    InvalidDepthError
        If source depth is invalid
    """
    from uacpy.core.exceptions import InvalidDepthError

    if source_depth < 0:
        raise InvalidDepthError(source_depth, env_depth, "Source")

    if source_depth > env_depth - margin:
        raise InvalidDepthError(source_depth, env_depth, "Source")


def validate_receiver_depths(receiver_depths: np.ndarray, env_depth: float, margin: float = 1.0):
    """
    Validate receiver depths against environment

    Parameters
    ----------
    receiver_depths : ndarray
        Receiver depths in meters
    env_depth : float
        Environment depth in meters
    margin : float
        Safety margin in meters

    Raises
    ------
    InvalidDepthError
        If any receiver depth is invalid
    """
    from uacpy.core.exceptions import InvalidDepthError

    if np.any(receiver_depths < 0):
        bad_depth = receiver_depths[receiver_depths < 0][0]
        raise InvalidDepthError(bad_depth, env_depth, "Receiver")

    if np.any(receiver_depths > env_depth - margin):
        bad_depth = receiver_depths[receiver_depths > env_depth - margin][0]
        raise InvalidDepthError(bad_depth, env_depth, "Receiver")
