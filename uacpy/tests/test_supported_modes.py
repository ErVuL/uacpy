"""Supported-RunMode harmonization tests.

Locks in the per-model ``_supported_modes`` set — the mode-axis analogue of
``test_capability_flags``. If a model gains or loses a RunMode, the change must
come with an update here so the public mode surface stays explicit, and so
every *unsupported* mode keeps being refused.
"""

import pytest

from uacpy.models.base import RunMode
from uacpy.models.bellhop import Bellhop
from uacpy.models.kraken import Kraken
from uacpy.models.scooter import Scooter
from uacpy.models.sparc import SPARC
from uacpy.models.bounce import Bounce
from uacpy.models.oases import OAST, OASN, OASR, OASP, OASS, OASSP
from uacpy.models.ram import RAM
from uacpy.core.exceptions import ExecutableNotFoundError


# (model factory, expected supported RunModes). Lambdas because constructors
# resolve their binary eagerly.
_EXPECTED = {
    'Bellhop': (
        lambda: Bellhop(),
        {RunMode.COHERENT_TL, RunMode.INCOHERENT_TL, RunMode.SEMICOHERENT_TL,
         RunMode.RAYS, RunMode.EIGENRAYS, RunMode.ARRIVALS,
         RunMode.BROADBAND, RunMode.TIME_SERIES},
    ),
    'Kraken': (
        lambda: Kraken(),
        {RunMode.MODES, RunMode.COHERENT_TL, RunMode.INCOHERENT_TL,
         RunMode.BROADBAND, RunMode.TIME_SERIES},
    ),
    'Scooter': (
        lambda: Scooter(),
        {RunMode.COHERENT_TL, RunMode.BROADBAND, RunMode.TIME_SERIES},
    ),
    'SPARC': (
        lambda: SPARC(),
        # CW transmission loss withdrawn: the pulse-to-CW extraction is not
        # quantitative. SPARC's product is its native time series.
        {RunMode.TIME_SERIES},
    ),
    'Bounce': (
        lambda: Bounce(),
        {RunMode.REFLECTION},
    ),
    'OAST': (
        lambda: OAST(),
        {RunMode.COHERENT_TL},
    ),
    'OASN': (
        lambda: OASN(),
        {RunMode.COVARIANCE, RunMode.REPLICA},
    ),
    'OASR': (
        lambda: OASR(),
        {RunMode.REFLECTION},
    ),
    'OASP': (
        lambda: OASP(),
        {RunMode.COHERENT_TL, RunMode.BROADBAND, RunMode.TIME_SERIES},
    ),
    'OASS': (
        lambda: OASS(correlation_length=10.0),
        {RunMode.REVERBERATION, RunMode.COVARIANCE},
    ),
    'OASSP': (
        lambda: OASSP(correlation_length=10.0),
        {RunMode.BROADBAND, RunMode.TIME_SERIES},
    ),
    'RAM': (
        lambda: RAM(),
        {RunMode.COHERENT_TL, RunMode.BROADBAND, RunMode.TIME_SERIES},
    ),
}


_OASES_MODELS = {'OAST', 'OASN', 'OASR', 'OASP', 'OASS', 'OASSP'}


def _model_param(name):
    """Wrap parametrize values with the binary markers each model needs
    (every constructor existence-checks its binary), plus ``requires_oases``
    for the separately-licensed OASES family."""
    marks = [pytest.mark.requires_binary]
    if name in _OASES_MODELS:
        marks.append(pytest.mark.requires_oases)
    return pytest.param(name, marks=marks, id=name)


_MODEL_PARAMS = [_model_param(n) for n in _EXPECTED.keys()]


@pytest.mark.parametrize('model_name', _MODEL_PARAMS)
def test_supported_modes(model_name):
    """``_supported_modes`` matches exactly; ``supports_mode`` agrees on the
    full RunMode enum (every unsupported mode refused)."""
    factory, expected = _EXPECTED[model_name]
    try:
        m = factory()
    except ExecutableNotFoundError:
        pytest.skip(f"{model_name} binary not available")

    assert set(m._supported_modes) == expected, (
        f"{model_name}._supported_modes = {set(m._supported_modes)}, "
        f"expected {expected}"
    )
    for mode in RunMode:
        assert m.supports_mode(mode) is (mode in expected), (
            f"{model_name}.supports_mode({mode}) disagrees with "
            f"_supported_modes"
        )
