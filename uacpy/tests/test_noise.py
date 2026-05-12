"""
Smoke tests for the simplified uacpy.noise module.

Covers ``compute_windnoise`` and the ``WenzNoise`` class
(wind / shipping / rain / thermal / turbulence).
"""

import numpy as np
import pytest

from uacpy.noise import compute_windnoise, WenzNoise


@pytest.fixture
def freqs():
    # Log-spaced 1 Hz → 100 kHz so every Wenz band has plenty of points.
    return np.logspace(0.0, 5.0, 200)


# ─── compute_windnoise ───────────────────────────────────────────────────────


def test_compute_windnoise_zero_wind(freqs):
    """``u == 0`` silences the surface-noise source: spectral level is
    ``-inf`` dB at every frequency so the incoherent dB sum with the
    other Wenz components drops wind cleanly."""
    NL = compute_windnoise(freqs, u=0)
    assert NL.shape == freqs.shape
    assert np.all(np.isneginf(NL))


def test_compute_windnoise_scalar_frequency():
    """``compute_windnoise(scalar_f, u)`` returns a 1-element 1-D array."""
    NL = compute_windnoise(100.0, u=10, water_depth='deep')
    assert NL.shape == (1,)
    assert np.isfinite(NL[0])


def test_compute_windnoise_negative_wind_raises():
    """Negative wind speed raises :class:`ValueError`."""
    with pytest.raises(ValueError, match="non-negative"):
        compute_windnoise(np.array([100.0]), u=-5, water_depth='deep')


def test_compute_windnoise_increases_with_wind(freqs):
    low = compute_windnoise(freqs, u=5,  water_depth='deep')
    high = compute_windnoise(freqs, u=25, water_depth='deep')
    assert np.all(np.isfinite(low))
    assert np.all(np.isfinite(high))
    band = (freqs >= 100) & (freqs <= 1500)
    assert np.mean(high[band]) > np.mean(low[band])


def test_compute_windnoise_shallow_louder_than_deep(freqs):
    deep = compute_windnoise(freqs, u=10, water_depth='deep')
    shallow = compute_windnoise(freqs, u=10, water_depth='shallow')
    band = (freqs >= 100) & (freqs <= 1500)
    assert np.mean(shallow[band]) > np.mean(deep[band])


def test_compute_windnoise_band_integrate(freqs):
    """``band_integrate=True`` integrates each spectral band — must not crash."""
    pointwise = compute_windnoise(freqs, u=10, water_depth='deep')
    integrated = compute_windnoise(freqs, u=10, water_depth='deep', band_integrate=True)
    assert pointwise.shape == freqs.shape
    assert integrated.shape == freqs.shape
    assert np.all(np.isfinite(integrated))


def test_wenznoise_high_freq_only_no_rain_meld_crash():
    """Rain melding only fires when both <7kHz and >7kHz frequencies exist."""
    f_high = np.logspace(4.0, 5.0, 30)  # all > 7 kHz
    wenz = WenzNoise(f_high, wind_speed=10, rain_rate='moderate')
    assert np.all(np.isfinite(wenz.rain))


# ─── WenzNoise — invariants ─────────────────────────────────────────────────


def test_wenznoise_default_attributes(freqs):
    """Inactive bands are carried as ``-inf`` in each component array;
    ``.total`` stays finite as long as at least one source is active
    at every frequency."""
    wenz = WenzNoise(freqs, wind_speed=10)
    for attr in ('total', 'shipping', 'wind', 'rain', 'thermal', 'turbulence'):
        v = getattr(wenz, attr)
        assert v.shape == freqs.shape
    # The total is the incoherent sum — must be finite when any of the
    # five components is active at each frequency (it is, for these inputs).
    assert np.all(np.isfinite(wenz.total))
    # Components are allowed to be -inf when inactive but never NaN.
    for attr in ('shipping', 'wind', 'rain', 'thermal', 'turbulence'):
        v = getattr(wenz, attr)
        assert not np.any(np.isnan(v)), f"{attr} contains NaN"


def test_wenznoise_components_matrix_layout(freqs):
    wenz = WenzNoise(freqs, wind_speed=10, shipping_level='medium',
                     rain_rate='moderate')
    M = wenz.components
    assert M.shape == (freqs.size, 6)
    np.testing.assert_array_equal(M[:, 0], wenz.total)
    np.testing.assert_array_equal(M[:, 1], wenz.shipping)
    np.testing.assert_array_equal(M[:, 2], wenz.wind)
    np.testing.assert_array_equal(M[:, 3], wenz.rain)
    np.testing.assert_array_equal(M[:, 4], wenz.thermal)
    np.testing.assert_array_equal(M[:, 5], wenz.turbulence)


def test_wenznoise_total_geq_components(freqs):
    wenz = WenzNoise(freqs, wind_speed=10, shipping_level='medium',
                     rain_rate='moderate')
    components = np.column_stack([wenz.shipping, wenz.wind, wenz.rain,
                                  wenz.thermal, wenz.turbulence])
    # Total in dB must be ≥ each individual component everywhere.
    assert np.all(wenz.total + 1e-6 >= components.max(axis=1))


def test_wenznoise_shipping_levels_ordered(freqs):
    band = (freqs >= 30) & (freqs <= 200)
    low = WenzNoise(freqs, wind_speed=5, shipping_level='low').total[band]
    med = WenzNoise(freqs, wind_speed=5, shipping_level='medium').total[band]
    high = WenzNoise(freqs, wind_speed=5, shipping_level='high').total[band]
    assert np.mean(low) < np.mean(med) < np.mean(high)


def test_wenznoise_as_psd_round_trip(freqs):
    wenz = WenzNoise(freqs, wind_speed=10)
    # Default ref=1e-6 Pa (= 1 µPa) returns Pa²/Hz; converting back to
    # dB re 1 µPa²/Hz must reproduce the .total attribute exactly.
    pa2 = wenz.as_psd()
    db_back = 10 * np.log10(pa2 / 1e-12)
    np.testing.assert_allclose(db_back, wenz.total, rtol=0, atol=1e-9)


def test_wenznoise_repr_contains_params(freqs):
    wenz = WenzNoise(freqs, wind_speed=15, water_depth='shallow',
                     shipping_level='high', rain_rate='heavy')
    s = repr(wenz)
    assert 'wind=15' in s
    assert "'shallow'" in s
    assert "'high'" in s
    assert "'heavy'" in s


def test_wenznoise_rejects_invalid_kwargs(freqs):
    with pytest.raises(ValueError, match='water_depth'):
        WenzNoise(freqs, wind_speed=10, water_depth='abyssal')
    with pytest.raises(ValueError, match='shipping_level'):
        WenzNoise(freqs, wind_speed=10, shipping_level='extreme')
    with pytest.raises(ValueError, match='rain_rate'):
        WenzNoise(freqs, wind_speed=10, rain_rate='monsoon')


def test_wenznoise_plot_returns_fig_ax(freqs):
    wenz = WenzNoise(freqs, wind_speed=15)
    fig, ax = wenz.plot()
    assert fig is not None and ax is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_wenznoise_plot_total_only(freqs):
    wenz = WenzNoise(freqs, wind_speed=15)
    fig, ax = wenz.plot(show_components=False)
    assert fig is not None and ax is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_wenznoise_total_matches_linear_sum_of_components(freqs):
    """``total`` equals the incoherent linear sum of the five
    components within numerical precision; ``-inf`` components
    contribute zero linear power."""
    wenz = WenzNoise(freqs, wind_speed=10, shipping_level='medium',
                     rain_rate='moderate', water_depth='deep')
    # Treat -inf rigorously: 10**(-inf/10) = 0.
    lin_sum = (
        10.0 ** (wenz.shipping / 10.0)
        + 10.0 ** (wenz.wind / 10.0)
        + 10.0 ** (wenz.rain / 10.0)
        + 10.0 ** (wenz.thermal / 10.0)
        + 10.0 ** (wenz.turbulence / 10.0)
    )
    expected = 10.0 * np.log10(lin_sum)
    np.testing.assert_allclose(wenz.total, expected, rtol=1e-10, atol=1e-10)


def test_wenznoise_zero_wind_drops_wind_from_total(freqs):
    """``wind_speed=0`` makes the wind component ``-inf`` so it drops out
    of ``.total``; the total equals the sum of the four remaining
    components."""
    wenz = WenzNoise(freqs, wind_speed=0, shipping_level='medium',
                     rain_rate='moderate', water_depth='deep')
    assert np.all(np.isneginf(wenz.wind))
    lin_no_wind = (
        10.0 ** (wenz.shipping / 10.0)
        + 10.0 ** (wenz.rain / 10.0)
        + 10.0 ** (wenz.thermal / 10.0)
        + 10.0 ** (wenz.turbulence / 10.0)
    )
    expected = 10.0 * np.log10(lin_no_wind)
    np.testing.assert_allclose(wenz.total, expected, rtol=1e-10, atol=1e-10)
