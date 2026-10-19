"""Each named constant holds its value, and every site that reads the
quantity reads the name."""
import inspect

import numpy as np
import pytest

import uacpy
from uacpy.core import constants
from uacpy.core.constants import (
    NO_ENERGY_DB, PH_MAX, PH_MIN, PRESSURE_FLOOR,
)
from uacpy.core.boundary import BoundaryProperties
from uacpy.core._validate import DENSITY_UNITS_SUSPECT
from uacpy.core.deck_limits import (
    AT_DB_PER_NEPER, MAX_ATTENUATION,
)
from uacpy.io import env_reader
from uacpy.core.exceptions import ConfigurationError
from uacpy.models.base import DEFAULT_RUN_TIMEOUT, PropagationModel


def test_the_constants_hold_their_values():
    assert DEFAULT_RUN_TIMEOUT == 600.0
    assert (PH_MIN, PH_MAX) == (0.0, 14.0)
    assert NO_ENERGY_DB == -20.0 * np.log10(PRESSURE_FLOOR) == 600.0
    assert AT_DB_PER_NEPER == 8.6858896
    assert AT_DB_PER_NEPER != constants.NEPER_TO_DB
    assert MAX_ATTENUATION == 8.6858896 * 2.0 * np.pi
    assert env_reader.AT_DB_PER_NEPER is AT_DB_PER_NEPER
    for name in ('PH_MIN', 'PH_MAX'):
        assert name in constants.__all__


def test_the_standard_values_are_the_exact_ones():
    assert constants.STANDARD_GRAVITY == 9.80665
    assert constants.STANDARD_ATMOSPHERE == 101325.0
    from uacpy.core.acoustics import bubbles, seawater
    p0 = inspect.signature(bubbles.bubble_resonance).parameters['p0']
    assert p0.default == constants.STANDARD_ATMOSPHERE
    assert seawater._DBAR_PER_KGF_CM2 == constants.STANDARD_GRAVITY


def test_the_peak_wavelength_reads_standard_gravity():
    from uacpy.core.altimetry import sea_surface_n_points
    # 1000.8 peak wavelengths of 8 samples: 1001 + 1 at standard gravity,
    # 1002 + 1 at 9.81 (a 0.034 % shorter wavelength)
    from uacpy.core.units import ms_to_knots
    lam = 2.0 * np.pi * 10.0 ** 2 / constants.STANDARD_GRAVITY
    # The wind argument is in knots: 10 m/s converted with the package's own factor.
    _, needed = sea_surface_n_points(1000.8 * lam / 8.0, ms_to_knots(10.0))
    assert needed == 1002


@pytest.mark.parametrize('engine', [
    'Bellhop', 'Kraken', 'Scooter', 'SPARC', 'RAM', 'Bounce', 'OAST', 'OASN',
    'OASP', 'OASR', 'OASS', 'OASSP'])
def test_every_engine_times_out_at_the_one_default(engine):
    default = inspect.signature(getattr(uacpy, engine)).parameters[
        'timeout'].default
    assert default == DEFAULT_RUN_TIMEOUT
    base = inspect.signature(PropagationModel.__init__).parameters['timeout']
    assert base.default == DEFAULT_RUN_TIMEOUT


@pytest.mark.parametrize('rho, refused', [
    (1.2, None), (0.85, None), (DENSITY_UNITS_SUSPECT, None),
    (np.nextafter(DENSITY_UNITS_SUSPECT, 21.0), 'reads as kg/m³'),
    (1027.0, 'reads as kg/m³'), (0.0, 'positive and finite'),
    (-1.0, 'positive and finite'), (np.nan, 'positive and finite')])
def test_the_water_density_reads_the_one_kg_per_m3_threshold(rho, refused):
    """Any positive g/cm³ water density is accepted (a brine of 1.2, a
    laboratory liquid of 0.85: no engine refuses one); a value above the
    threshold every density input shares reads as kg/m³."""
    def build():
        return uacpy.Environment(
            name='rho', bathymetry=100.0, ssp=1500.0, water_density=rho,
            bottom=BoundaryProperties(sound_speed=1700.0, density=1.5,
                                      attenuation=0.5))
    if refused:
        with pytest.raises(ConfigurationError, match=refused):
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
