"""
Pytest configuration and fixtures for UACPY tests
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uacpy
from uacpy.models import Bellhop, Kraken, KrakenField


@pytest.fixture
def simple_env():
    """Simple isovelocity environment"""
    return uacpy.Environment(
        name="Test Environment",
        depth=100.0,
        sound_speed=1500.0,
        ssp_type='isovelocity'
    )


@pytest.fixture
def munk_env():
    """Munk profile environment"""
    depths = np.linspace(0, 100, 21)
    # Simple Munk-like profile
    axis_depth = 50
    c_axis = 1485
    sound_speeds = c_axis * (1 + 0.00737 * ((depths - axis_depth) / axis_depth) ** 2)

    return uacpy.Environment(
        name="Munk Profile",
        depth=100.0,
        ssp_data=np.column_stack([depths, sound_speeds]),
        ssp_type='pchip'
    )


@pytest.fixture
def range_dependent_env():
    """Range-dependent environment with bathymetry"""
    # Create bathymetry: slope from 80m to 120m over 10km
    ranges = np.linspace(0, 10000, 11)
    depths = np.linspace(80, 120, 11)
    bathymetry = np.column_stack([ranges, depths])

    return uacpy.Environment(
        name="Range Dependent",
        depth=100.0,
        sound_speed=1500.0,
        ssp_type='isovelocity',
        bathymetry=bathymetry
    )


@pytest.fixture
def source():
    """Standard acoustic source"""
    return uacpy.Source(depth=50.0, frequency=100.0)


@pytest.fixture
def receiver_grid():
    """Standard receiver grid"""
    return uacpy.Receiver(
        depths=np.linspace(10, 90, 9),
        ranges=np.linspace(100, 5000, 11)
    )


@pytest.fixture
def receiver_small():
    """Small receiver grid for fast tests"""
    return uacpy.Receiver(
        depths=np.array([25.0, 50.0, 75.0]),
        ranges=np.array([1000.0, 3000.0, 5000.0])
    )


@pytest.fixture
def bellhop_model():
    """Bellhop model instance"""
    return Bellhop(verbose=False)


@pytest.fixture
def kraken_model():
    """Kraken model instance"""
    return Kraken(verbose=False)


@pytest.fixture
def krakenfield_model():
    """KrakenField model instance"""
    return KrakenField(verbose=False)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_binary: marks tests requiring compiled native binaries (Fortran/C)"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_oases: marks tests requiring compiled OASES binaries"
    )
