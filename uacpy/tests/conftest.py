"""Pytest configuration and fixtures for UACPY tests."""

# Lock matplotlib to a non-interactive backend before any test imports it.
# Must run before any other matplotlib import in the test session.
import matplotlib

matplotlib.use("Agg")

import tempfile  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import uacpy  # noqa: E402


@pytest.fixture(autouse=True)
def _seed_numpy():
    """Seed numpy.random before every test to keep fixture data reproducible."""
    np.random.seed(0xACED)


@pytest.fixture(autouse=True)
def _release_matplotlib_figures():
    """Close every matplotlib figure after each test."""
    yield
    import matplotlib.pyplot as plt
    plt.close('all')


@pytest.fixture(autouse=True)
def _redirect_tempdir(tmp_path, monkeypatch):
    """Route ``tempfile.gettempdir()`` to per-test ``tmp_path`` so any
    ``FileManager`` scratch dir is reaped by pytest (xdist-safe, no
    /dev/shm leakage)."""
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))


@pytest.fixture
def simple_env():
    """Simple isovelocity environment."""
    return uacpy.Environment(
        name="Test Environment",
        bathymetry=100.0,
        ssp=1500.0,
    )


@pytest.fixture
def parabolic_ssp_env():
    """100-m shallow-water env with a parabolic SSP centred at 50 m.

    Despite its previous name, this is *not* the canonical Munk profile
    (which carries an exponential ``η - 1 + exp(-η)`` term and channels
    at ~1300 m in deep water). Used for SSP-shape smoke checks where
    the only requirement is "a non-flat profile with a minimum
    somewhere".
    """
    from uacpy.core.environment import SoundSpeedProfile
    depths = np.linspace(0, 100, 21)
    axis_depth = 50
    c_axis = 1485
    sound_speeds = c_axis * (1 + 0.00737 * ((depths - axis_depth) / axis_depth) ** 2)

    return uacpy.Environment(
        name="Parabolic SSP",
        bathymetry=100.0,
        ssp=SoundSpeedProfile.from_pairs(
            np.column_stack([depths, sound_speeds])
        ),
    )


@pytest.fixture
def munk_env():
    """Deep-water Munk profile (canonical, axis at 1300 m).

    Built via :meth:`SoundSpeedProfile.from_munk`, which implements
    ``c(z) = c_min * (1 + ε * (η - 1 + exp(-η)))`` with
    ``η = 2(z - z_axis)/z_axis``, ``c_min = 1500 m/s``, ``ε = 7.37e-3``.
    Bathymetry is 5 km with a fluid half-space bottom.
    """
    from uacpy.core.environment import SoundSpeedProfile
    return uacpy.Environment(
        name="Munk Profile",
        bathymetry=5000.0,
        ssp=SoundSpeedProfile.from_munk(5000.0),
        bottom=uacpy.BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0,
            density=1.8,
            attenuation=0.3,
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
        ssp=1500.0,
        bathymetry=bathymetry,
    )


@pytest.fixture
def source():
    """Standard acoustic source."""
    return uacpy.Source(depths=50.0, frequencies=100.0)


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


@pytest.fixture
def receiver():
    """Default receiver grid (alias for ``receiver_grid``)."""
    return uacpy.Receiver(
        depths=np.linspace(10, 90, 9),
        ranges=np.linspace(100, 5000, 11),
    )


@pytest.fixture
def halfspace_bottom():
    """Standard fluid half-space sediment used by Pekeris-style tests."""
    from uacpy.core.environment import BoundaryProperties
    return BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1700.0,
        density=1.8,
        attenuation=0.5,
    )


@pytest.fixture
def elastic_bottom():
    """Half-space with shear (used by elastic-bottom Pekeris cases)."""
    from uacpy.core.environment import BoundaryProperties
    return BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1700.0,
        shear_speed=400.0,
        density=1.8,
        attenuation=0.5,
        shear_attenuation=0.8,
    )


@pytest.fixture
def pekeris_env(halfspace_bottom):
    """Classic 100-m Pekeris waveguide with a fluid half-space bottom."""
    return uacpy.Environment(
        name="Pekeris (fluid bottom)",
        bathymetry=100.0,
        ssp=1500.0,
        bottom=halfspace_bottom,
    )


@pytest.fixture
def elastic_env(elastic_bottom):
    """Pekeris waveguide with an elastic half-space bottom (shear=400 m/s)."""
    return uacpy.Environment(
        name="Pekeris (elastic bottom)",
        bathymetry=100.0,
        ssp=1500.0,
        bottom=elastic_bottom,
    )
