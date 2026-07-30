"""Capability-flag harmonization tests.

Locks in the per-model `_supports_*` matrix. If a model gains or loses
support for an Environment feature, the change must come with an update
to this test so the public capability surface stays explicit.
"""

import numpy as np
import pytest

import uacpy

from uacpy.models.bellhop import Bellhop
from uacpy.models.kraken import Kraken
from uacpy.models.scooter import Scooter
from uacpy.models.sparc import SPARC
from uacpy.models.bounce import Bounce
from uacpy.models.oases import OAST, OASN, OASR, OASP
from uacpy.models.ram import RAM
from uacpy.core.exceptions import ExecutableNotFoundError, ConfigurationError
from uacpy.models.base import VALID_SOURCE_TYPES


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
    'Kraken': (
        lambda: Kraken(),
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
    """Wrap parametrize values with the binary markers each model needs.

    Every model resolves (and existence-checks) its binary in ``__init__``,
    so this test — which constructs the model to read its capability flags —
    needs ``requires_binary`` for all, plus ``requires_oases`` for the
    separately-licensed OASES family. Lets ``pytest -m 'not requires_binary'``
    / ``'not requires_oases'`` deselect at collection time.
    """
    marks = [pytest.mark.requires_binary]
    if name in _OASES_MODELS:
        marks.append(pytest.mark.requires_oases)
    return pytest.param(name, marks=marks, id=name)


_MODEL_PARAMS = [_model_param(n) for n in _EXPECTED.keys()]


@pytest.mark.parametrize('model_name', _MODEL_PARAMS)
@pytest.mark.parametrize('feature', _FEATURES)
def test_capability_flag(model_name, feature):
    factory, expected = _EXPECTED[model_name]
    try:
        m = factory()
    except ExecutableNotFoundError:
        pytest.skip(f"{model_name} binary not available")
    flag = getattr(m, f'_supports_{feature}')
    assert flag is expected[feature], (
        f"{model_name}._supports_{feature} = {flag}, "
        f"expected {expected[feature]}"
    )


# (source geometries the model honours, whether it reads a .sbp beam pattern).
_EXPECTED_SOURCE_TYPES = {
    'Bellhop': ({'point', 'line'}, True),
    'Kraken':  ({'point', 'line', 'scaled'}, True),
    'Scooter': ({'point', 'line', 'scaled'}, False),
    'SPARC':   ({'point'}, False),
    'Bounce':  ({'point', 'line', 'scaled'}, False),
    'OAST':    ({'point'}, False),
    'OASN':    ({'point'}, False),
    'OASR':    ({'point'}, False),
    'OASP':    ({'point'}, False),
    'RAM':     ({'point'}, False),
}


def _reference_environment():
    return uacpy.Environment(
        bathymetry=200.0,
        ssp=uacpy.SoundSpeedProfile(depths=[0, 200], data=[1500, 1500]),
    )


def _reference_receiver():
    return uacpy.Receiver(depths=100.0, ranges=np.linspace(100, 2000, 20))


@pytest.mark.parametrize('model_name', _MODEL_PARAMS)
def test_source_capability_matrix(model_name):
    """Locks the per-model source-geometry / beam-pattern surface."""
    factory = _EXPECTED[model_name][0]
    try:
        m = factory()
    except ExecutableNotFoundError:
        pytest.skip(f"{model_name} binary not available")
    types, pattern = _EXPECTED_SOURCE_TYPES[model_name]
    assert set(m._supported_source_types) == types
    assert m._supports_source_beam_pattern is pattern


def test_every_declared_source_type_is_valid():
    for types, _ in _EXPECTED_SOURCE_TYPES.values():
        assert types <= VALID_SOURCE_TYPES


@pytest.mark.requires_binary
def test_sparc_gains_geometry_only_in_snapshot_mode():
    # sparc.py:812 is reached only from _run_snapshot, which output_mode='S'
    # selects; 'R' and 'D' are range-/depth-native and never run a Hankel
    # transform, so they honour no geometry.
    try:
        assert set(SPARC()._supported_source_types) == {'point'}
        assert set(SPARC(output_mode='S')._supported_source_types) == {
            'point', 'line', 'scaled'}
    except ExecutableNotFoundError:
        pytest.skip("SPARC binary not available")


@pytest.mark.requires_binary
def test_unsupported_source_type_is_rejected():
    try:
        m = RAM()
    except ExecutableNotFoundError:
        pytest.skip("RAM binary not available")
    with pytest.raises(ConfigurationError, match="source_type"):
        m.validate_inputs(
            _reference_environment(),
            uacpy.Source(depths=50, frequencies=100, source_type='line'),
            _reference_receiver(),
        )


@pytest.mark.requires_binary
def test_unsupported_beam_pattern_is_rejected():
    try:
        m = Scooter()
    except ExecutableNotFoundError:
        pytest.skip("Scooter binary not available")
    pat = np.array([[-90.0, -20.0], [90.0, 0.0]])
    with pytest.raises(ConfigurationError, match="beam pattern"):
        m.validate_inputs(
            _reference_environment(),
            uacpy.Source(depths=50, frequencies=100, beam_pattern=pat),
            _reference_receiver(),
        )


_EXPECTED_ROUGH_SURFACE = {
    'Bellhop': False, 'Kraken': True, 'Scooter': True, 'SPARC': False,
    'Bounce': False, 'OAST': False, 'OASN': False, 'OASR': False,
    'OASP': False, 'RAM': False,
}


@pytest.mark.parametrize('model_name', _MODEL_PARAMS)
def test_rough_surface_capability_matrix(model_name):
    """Only solvers that accept a non-zero SSP%sigma may declare it.

    ``sparc.f90:177`` ERROUTs on any non-zero ``SSP%sigma(1:NMedia)`` and
    ``bounce.f90:104`` on a rough elastic interface, so those must not receive
    ``env.surface.roughness``; Kraken and Scooter consume it (Kraken via the
    Kuperman-Ingenito perturbation, Scooter at ``scooter.f90:309``).
    """
    try:
        m = _EXPECTED[model_name][0]()
    except ExecutableNotFoundError:
        pytest.skip(f"{model_name} binary not available")
    assert m._supports_rough_surface is _EXPECTED_ROUGH_SURFACE[model_name]


@pytest.mark.requires_binary
def test_rough_surface_is_dropped_for_solvers_that_reject_it():
    """A rough surface must be collapsed with a warning, not passed through.

    Surface roughness reaches the water column's mesh line (sigma(1)); handing
    it to SPARC would trip its 'Rough interfaces not allowed' ERROUT.
    """
    import warnings
    env = _reference_environment()
    env.surface.roughness = 2.0
    try:
        m = _EXPECTED['SPARC'][0]()
    except ExecutableNotFoundError:
        pytest.skip("SPARC binary not available")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        projected = m._project_environment(env)
    assert projected.surface.roughness == 0.0
    assert [x for x in w if 'rough sea surface' in str(x.message)]
    # The caller's environment must not be mutated in place.
    assert env.surface.roughness == 2.0
