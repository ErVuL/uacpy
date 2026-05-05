"""Pytest configuration and fixtures for UACPY tests."""

# Lock matplotlib to a non-interactive backend before any test imports it.
# Must run before any other matplotlib import in the test session.
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import uacpy


@pytest.fixture(autouse=True)
def _seed_numpy():
    """Seed numpy.random before every test to keep fixture data reproducible."""
    np.random.seed(0xACED)


@pytest.fixture
def simple_env():
    """Simple isovelocity environment."""
    return uacpy.Environment(
        name="Test Environment",
        depth=100.0,
        sound_speed=1500.0,
    )


@pytest.fixture
def munk_env():
    """Munk profile environment."""
    from uacpy.core.environment import SoundSpeedProfile
    depths = np.linspace(0, 100, 21)
    axis_depth = 50
    c_axis = 1485
    sound_speeds = c_axis * (1 + 0.00737 * ((depths - axis_depth) / axis_depth) ** 2)

    return uacpy.Environment(
        name="Munk Profile",
        depth=100.0,
        ssp=SoundSpeedProfile.from_pairs(
            np.column_stack([depths, sound_speeds]), interp='pchip',
        ),
    )


@pytest.fixture
def range_dependent_env():
    """Range-dependent environment with bathymetry."""
    ranges = np.linspace(0, 10000, 11)
    depths = np.linspace(80, 120, 11)
    bathymetry = np.column_stack([ranges, depths])

    return uacpy.Environment(
        name="Range Dependent",
        depth=100.0,
        sound_speed=1500.0,
        bathymetry=bathymetry,
    )


@pytest.fixture
def source():
    """Standard acoustic source."""
    return uacpy.Source(depth=50.0, frequency=100.0)


@pytest.fixture
def receiver_grid():
    """Standard receiver grid."""
    return uacpy.Receiver(
        depths=np.linspace(10, 90, 9),
        ranges=np.linspace(100, 5000, 11)
    )


@pytest.fixture
def receiver_small():
    """Small receiver grid for fast tests."""
    return uacpy.Receiver(
        depths=np.array([25.0, 50.0, 75.0]),
        ranges=np.array([1000.0, 3000.0, 5000.0])
    )


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_binary: marks tests requiring compiled native binaries (Fortran/C)"
    )
    config.addinivalue_line(
        "markers", "requires_oases: marks tests requiring compiled OASES binaries"
    )
