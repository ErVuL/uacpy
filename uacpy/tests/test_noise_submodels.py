import numpy as np
import pytest
from uacpy.noise import noise as N
from uacpy.core.exceptions import ConfigurationError

F = np.array([10.0, 100.0, 1000.0, 10000.0])


def test_registries_have_defaults():
    assert 'merklinger' in N.WIND_MODELS
    assert 'wenz' in N.SHIPPING_MODELS
    assert 'torres_costa' in N.RAIN_MODELS
    assert 'mellen' in N.THERMAL_MODELS
    assert 'wenz' in N.TURBULENCE_MODELS


def test_thermal_matches_formula():
    out = N.THERMAL_MODELS['mellen'](F)
    ref = -75.0 + 20.0 * np.log10(F)
    ref[ref <= 0] = -np.inf
    assert np.array_equal(out, ref)


def test_turbulence_is_canonical_wenz():
    out = N.TURBULENCE_MODELS['wenz'](F)
    ref = 17.0 - 30.0 * np.log10(F / 1000.0)   # = 107 - 30*log10(f_Hz)
    ref[ref <= 0] = -np.inf
    assert np.allclose(out, ref, equal_nan=True)
    # 1 Hz → 107 dB, the canonical Wenz turbulence intercept
    assert np.isclose(N.TURBULENCE_MODELS['wenz'](np.array([1.0]))[0], 107.0)


def test_shipping_no_is_silent():
    out = N.SHIPPING_MODELS['wenz'](F, shipping_level='no', water_depth='deep')
    assert np.all(out == -np.inf)


def test_resolve_submodel():
    fn, name = N._resolve_submodel(None, N.WIND_MODELS, 'merklinger', 'wind_model')
    assert name == 'merklinger' and callable(fn)
    fn, name = N._resolve_submodel('merklinger', N.WIND_MODELS, 'merklinger', 'wind_model')
    assert name == 'merklinger'

    def custom(f, **k):
        return np.zeros_like(f)
    fn, name = N._resolve_submodel(custom, N.WIND_MODELS, 'merklinger', 'wind_model')
    assert name == 'custom' and fn is custom
    with pytest.raises(ConfigurationError):
        N._resolve_submodel('nope', N.WIND_MODELS, 'merklinger', 'wind_model')
    with pytest.raises(ConfigurationError):
        N._resolve_submodel(123, N.WIND_MODELS, 'merklinger', 'wind_model')


def test_default_matches_registry_defaults():
    w = N.WenzNoise(F, wind_speed=15.0, rain_rate='heavy',
                    water_depth='deep', shipping_level='high')
    th = N.THERMAL_MODELS['mellen'](F)
    wi = N.WIND_MODELS['merklinger'](F, wind_speed=15.0, water_depth='deep')
    sh = N.SHIPPING_MODELS['wenz'](F, shipping_level='high', water_depth='deep')
    tu = N.TURBULENCE_MODELS['wenz'](F)
    ra = N.RAIN_MODELS['torres_costa'](F, rain_rate='heavy')
    ln10 = np.log(10.0)
    total = (10.0 / ln10) * np.logaddexp.reduce(
        np.stack([th, wi, sh, tu, ra]) * (ln10 / 10.0), axis=0)
    assert np.array_equal(w.thermal, th) and np.array_equal(w.wind, wi)
    assert np.array_equal(w.shipping, sh) and np.array_equal(w.turbulence, tu)
    assert np.array_equal(w.rain, ra)
    assert np.allclose(w.total, total, rtol=0, atol=0, equal_nan=True)


def test_models_recorded():
    w = N.WenzNoise(F, 15.0)
    assert w.models == {'wind': 'merklinger', 'shipping': 'wenz',
                        'rain': 'torres_costa', 'thermal': 'mellen',
                        'turbulence': 'wenz'}


def test_string_selector_equals_default():
    a = N.WenzNoise(F, 15.0)
    b = N.WenzNoise(F, 15.0, wind_model='merklinger')
    assert np.array_equal(a.wind, b.wind)


def test_bad_selector_raises():
    with pytest.raises(ConfigurationError):
        N.WenzNoise(F, 15.0, wind_model='nope')


def test_custom_callable_changes_only_that_component():
    base = N.WenzNoise(F, 15.0)
    flat = N.WenzNoise(F, 15.0, wind_model=lambda f, **k: np.full_like(f, 50.0))
    assert np.allclose(flat.wind, 50.0)
    assert np.array_equal(flat.shipping, base.shipping)
    assert flat.models['wind'] == 'custom'


def test_components_namedtuple():
    w = N.WenzNoise(F, 15.0, rain_rate='light')
    c = w.components
    assert isinstance(c, N.NoiseComponents)
    assert c._fields == ('total', 'wind', 'shipping', 'rain',
                         'thermal', 'turbulence')
    assert np.array_equal(c.wind, w.wind)
    assert np.array_equal(c.total, w.total)


def test_registry_extensible_and_exported():
    import uacpy.noise as pkg
    pkg.WIND_MODELS['flat50'] = lambda f, **k: np.full_like(f, 50.0)
    try:
        w = N.WenzNoise(F, 15.0, wind_model='flat50')
        assert np.allclose(w.wind, 50.0) and w.models['wind'] == 'flat50'
    finally:
        del pkg.WIND_MODELS['flat50']
    assert hasattr(pkg, 'NoiseComponents')


def test_coates_alternatives_registered():
    assert 'coates' in N.WIND_MODELS
    assert 'coates' in N.SHIPPING_MODELS


def test_coates_wind_plausible_and_differs():
    base = N.WenzNoise(F, 19.4)                       # ~10 m/s
    coates = N.WenzNoise(F, 19.4, wind_model='coates')
    assert coates.models['wind'] == 'coates'
    assert np.all(np.isfinite(coates.wind[np.isfinite(coates.wind)]))
    # Coates wind @1 kHz, 10 m/s ≈ 50 + 7.5*sqrt(10) - 40*log10(1.4) ≈ 68 dB
    w1k = N.WIND_MODELS['coates'](np.array([1000.0]), wind_speed=19.4)[0]
    assert 60.0 < w1k < 75.0
    assert not np.array_equal(coates.wind, base.wind)   # genuinely different model


def test_coates_shipping_silent_and_activity_order():
    assert np.all(N.SHIPPING_MODELS['coates'](
        F, shipping_level='no') == -np.inf)
    lo = N.SHIPPING_MODELS['coates'](np.array([100.0]), shipping_level='low')[0]
    hi = N.SHIPPING_MODELS['coates'](np.array([100.0]), shipping_level='high')[0]
    assert hi > lo                                     # more traffic → louder
