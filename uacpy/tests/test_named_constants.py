"""Each named constant holds its value, and every site that reads the
quantity reads the name."""
import inspect

import numpy as np
import pytest

import uacpy
from uacpy.core import constants
from uacpy.core.constants import (
    NO_ENERGY_DB, PH_MAX, PH_MIN, PRESSURE_FLOOR, WATER_DENSITY_MAX_G_CM3,
    WATER_DENSITY_MIN_G_CM3,
)
from uacpy.core.boundary import BoundaryProperties
from uacpy.core.deck_limits import (
    AT_DB_PER_NEPER, MAX_ATTENUATION_DB_PER_WAVELENGTH,
)
from uacpy.io import env_reader
from uacpy.core.exceptions import ConfigurationError
from uacpy.models.base import DEFAULT_RUN_TIMEOUT_S, PropagationModel


def test_the_constants_hold_their_values():
    assert DEFAULT_RUN_TIMEOUT_S == 600.0
    assert (WATER_DENSITY_MIN_G_CM3, WATER_DENSITY_MAX_G_CM3) == (0.9, 1.1)
    assert (PH_MIN, PH_MAX) == (0.0, 14.0)
    assert NO_ENERGY_DB == -20.0 * np.log10(PRESSURE_FLOOR) == 600.0
    assert AT_DB_PER_NEPER == 8.6858896
    assert AT_DB_PER_NEPER != constants.NEPER_TO_DB
    assert MAX_ATTENUATION_DB_PER_WAVELENGTH == 8.6858896 * 2.0 * np.pi
    assert env_reader.AT_DB_PER_NEPER is AT_DB_PER_NEPER
    for name in ('WATER_DENSITY_MIN_G_CM3', 'WATER_DENSITY_MAX_G_CM3',
                 'PH_MIN', 'PH_MAX'):
        assert name in constants.__all__


def test_the_standard_values_are_the_exact_ones():
    assert constants.STANDARD_GRAVITY_M_S2 == 9.80665
    assert constants.STANDARD_ATMOSPHERE_PA == 101325.0
    from uacpy.core.acoustics import bubbles, seawater
    p0 = inspect.signature(bubbles.bubble_resonance).parameters['p0']
    assert p0.default == constants.STANDARD_ATMOSPHERE_PA
    assert seawater._DBAR_PER_KGF_CM2 == constants.STANDARD_GRAVITY_M_S2


def test_the_peak_wavelength_reads_standard_gravity():
    from uacpy.core.altimetry import sea_surface_n_points
    # 1000.8 peak wavelengths of 8 samples: 1001 + 1 at standard gravity,
    # 1002 + 1 at 9.81 (a 0.034 % shorter wavelength)
    from uacpy.core.units import ms_to_knots
    lam = 2.0 * np.pi * 10.0 ** 2 / constants.STANDARD_GRAVITY_M_S2
    # The wind argument is in knots: 10 m/s converted with the package's own factor.
    _, needed = sea_surface_n_points(1000.8 * lam / 8.0, ms_to_knots(10.0))
    assert needed == 1002


@pytest.mark.parametrize('engine', [
    'Bellhop', 'Kraken', 'Scooter', 'SPARC', 'RAM', 'Bounce', 'OAST', 'OASN',
    'OASP', 'OASR', 'OASS', 'OASSP'])
def test_every_engine_times_out_at_the_one_default(engine):
    default = inspect.signature(getattr(uacpy, engine)).parameters[
        'timeout'].default
    assert default == DEFAULT_RUN_TIMEOUT_S
    base = inspect.signature(PropagationModel.__init__).parameters['timeout']
    assert base.default == DEFAULT_RUN_TIMEOUT_S


@pytest.mark.parametrize('rho, refused', [
    (WATER_DENSITY_MIN_G_CM3, False), (WATER_DENSITY_MAX_G_CM3, False),
    (np.nextafter(WATER_DENSITY_MIN_G_CM3, 0.0), True),
    (np.nextafter(WATER_DENSITY_MAX_G_CM3, 2.0), True)])
def test_the_water_density_band_is_the_named_one(rho, refused):
    def build():
        return uacpy.Environment(
            name='rho', bathymetry=100.0, ssp=1500.0, water_density=rho,
            bottom=BoundaryProperties(sound_speed=1700.0, density=1.5,
                                      attenuation=0.5))
    if refused:
        with pytest.raises(ConfigurationError, match='0.9-1.1'):
            build()
    else:
        assert build().water_density == rho


@pytest.mark.parametrize('ph, refused', [
    (PH_MIN, False), (PH_MAX - 0.5, False),
    (np.nextafter(PH_MIN, -1.0), True), (np.nextafter(PH_MAX, 15.0), True)])
def test_the_ph_scale_is_the_named_one(ph, refused):
    if refused:
        with pytest.raises(ConfigurationError, match='0..14'):
            _francois_garrison(ph)
    else:
        _francois_garrison(ph)


def _francois_garrison(ph):
    return uacpy.FrancoisGarrison(temperature=10.0, salinity=35.0, pH=ph)
