"""Capability-flag harmonization tests.

Locks in the per-model `_supports_*` matrix and the per-feature
``_unsupported_env_alternatives`` dict shape. If a model gains or loses
support for an Environment feature, the change must come with an update
to this test so the public capability surface stays explicit.
"""

import pytest

from uacpy.models.bellhop import Bellhop, BellhopCUDA
from uacpy.models.kraken import Kraken, KrakenC, KrakenField
from uacpy.models.scooter import Scooter
from uacpy.models.sparc import SPARC
from uacpy.models.bounce import Bounce
from uacpy.models.oases import OAST, OASN, OASR, OASP
from uacpy.models.ram import RAM


_FEATURES = (
    'altimetry',
    'range_dependent_bathymetry',
    'range_dependent_ssp',
    'range_dependent_bottom',
    'layered_bottom',
    'range_dependent_layered_bottom',
    'elastic_media',
)


# (model factory, expected flags by feature). Use lambdas because some
# constructors hit binary lookups eagerly.
_EXPECTED = {
    'Bellhop': (
        lambda: Bellhop(),
        {'altimetry': True, 'range_dependent_bathymetry': True,
         'range_dependent_ssp': True,
         'range_dependent_bottom': True, 'layered_bottom': False,
         'range_dependent_layered_bottom': False,
         'elastic_media': True},
    ),
    'BellhopCUDA': (
        lambda: BellhopCUDA(),
        {'altimetry': True, 'range_dependent_bathymetry': True,
         'range_dependent_ssp': True,
         'range_dependent_bottom': True, 'layered_bottom': False,
         'range_dependent_layered_bottom': False,
         'elastic_media': True},
    ),
    'Kraken': (
        lambda: Kraken(),
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'range_dependent_layered_bottom': False,
         'elastic_media': True},
    ),
    'KrakenC': (
        lambda: KrakenC(),
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'range_dependent_layered_bottom': False,
         'elastic_media': True},
    ),
    'KrakenField': (
        lambda: KrakenField(),
        {'altimetry': False, 'range_dependent_bathymetry': True,
         'range_dependent_ssp': True,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'range_dependent_layered_bottom': False,
         'elastic_media': True},
    ),
    'Scooter': (
        lambda: Scooter(),
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'range_dependent_layered_bottom': False,
         'elastic_media': True},
    ),
    'SPARC': (
        lambda: SPARC(),
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'range_dependent_layered_bottom': False,
         'elastic_media': False},
    ),
    'Bounce': (
        lambda: Bounce(),
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'range_dependent_layered_bottom': False,
         'elastic_media': True},
    ),
    'OAST': (
        lambda: OAST(),
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'range_dependent_layered_bottom': False,
         'elastic_media': True},
    ),
    'OASN': (
        lambda: OASN(),
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'range_dependent_layered_bottom': False,
         'elastic_media': True},
    ),
    'OASR': (
        lambda: OASR(),
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'range_dependent_layered_bottom': False,
         'elastic_media': True},
    ),
    'OASP': (
        lambda: OASP(),
        {'altimetry': False, 'range_dependent_bathymetry': False,
         'range_dependent_ssp': False,
         'range_dependent_bottom': False, 'layered_bottom': True,
         'range_dependent_layered_bottom': False,
         'elastic_media': True},
    ),
    'RAM': (
        lambda: RAM(),
        {'altimetry': True, 'range_dependent_bathymetry': True,
         'range_dependent_ssp': True,
         'range_dependent_bottom': True, 'layered_bottom': True,
         'range_dependent_layered_bottom': True,
         'elastic_media': True},
    ),
}


_OASES_MODELS = {'OAST', 'OASN', 'OASR', 'OASP'}


def _model_param(name):
    """Wrap parametrize values so OASES models carry the requires_oases marker.

    This lets ``pytest -m 'not requires_oases'`` deselect at collection
    time when the OASES binaries are absent, instead of relying on the
    in-test ``FileNotFoundError`` fallback.
    """
    marks = [pytest.mark.requires_oases] if name in _OASES_MODELS else []
    return pytest.param(name, marks=marks, id=name)


_MODEL_PARAMS = [_model_param(n) for n in _EXPECTED.keys()]


@pytest.mark.parametrize('model_name', _MODEL_PARAMS)
@pytest.mark.parametrize('feature', _FEATURES)
def test_capability_flag(model_name, feature):
    factory, expected = _EXPECTED[model_name]
    try:
        m = factory()
    except FileNotFoundError:
        pytest.skip(f"{model_name} binary not available")
    flag = getattr(m, f'_supports_{feature}')
    assert flag is expected[feature], (
        f"{model_name}._supports_{feature} = {flag}, "
        f"expected {expected[feature]}"
    )


@pytest.mark.parametrize('model_name', _MODEL_PARAMS)
def test_alternatives_dict_complete(model_name):
    """Every model exposes a dict with one entry per feature axis."""
    factory, _ = _EXPECTED[model_name]
    try:
        m = factory()
    except FileNotFoundError:
        pytest.skip(f"{model_name} binary not available")
    alts = m._unsupported_env_alternatives
    assert isinstance(alts, dict), f"{model_name} alts is {type(alts)}"
    assert set(alts.keys()) == set(_FEATURES), (
        f"{model_name} alts keys {set(alts.keys())} != {set(_FEATURES)}"
    )
    for feature, hint in alts.items():
        assert isinstance(hint, str) and hint, (
            f"{model_name}.{feature} hint is empty/non-string"
        )


def test_bellhop_layered_alts_mention_run_with_bounce():
    """Bellhop's layered-bottom suggestion includes the run_with_bounce hint."""
    try:
        b = Bellhop()
    except FileNotFoundError:
        pytest.skip("Bellhop binary not available")
    assert 'run_with_bounce' in b._unsupported_env_alternatives['layered_bottom']
    assert 'run_with_bounce' in (
        b._unsupported_env_alternatives['range_dependent_layered_bottom']
    )
