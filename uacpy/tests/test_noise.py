"""
Smoke tests for uacpy.noise (Wenz / Knudsen / wind / shipping / thermal).

These tests check qualitative invariants of the empirical models — the
spectral level is finite, monotonic where the physics says it should be,
and the AmbientNoiseSimulator builder composes multiple sources without
crashing. They are NOT regression tests against published curves; that
work belongs in a separate benchmark file with stored reference data.
"""

import numpy as np
import pytest

from uacpy.noise import AmbientNoiseSimulator
from uacpy.noise.noise import (
    shipping_noise_wenz,
    thermal_noise_mellen,
    wind_noise_knudsen,
)


class TestEmpiricalCurves:
    def test_thermal_noise_mellen_increases_with_frequency(self):
        """Mellen's thermal noise grows as 20·log10(f)."""
        f = np.logspace(3, 5, 25)  # 1 kHz .. 100 kHz
        nl = thermal_noise_mellen(f)
        assert np.all(np.isfinite(nl))
        assert np.all(np.diff(nl) > 0), "thermal NL must be monotone increasing in f"
        # 20·log10 → 20 dB per decade, check to ±1 dB
        per_decade = nl[-1] - nl[0]
        assert 38 <= per_decade <= 42, (
            f"thermal NL should rise 40 dB over 2 decades, got {per_decade:.1f}"
        )

    def test_wind_knudsen_higher_wind_louder(self):
        """Knudsen wind noise: higher wind → louder spectrum (in band)."""
        f = np.array([200.0, 500.0, 1000.0, 5000.0])
        nl_low = wind_noise_knudsen(f, wind_speed=2.0)
        nl_high = wind_noise_knudsen(f, wind_speed=15.0)
        # Inside the validity band [100, 25000], higher wind must not be quieter.
        # (Knudsen is a step table — equal-or-louder, not strictly louder.)
        assert np.all(nl_high >= nl_low - 1e-9), (
            f"high-wind NL {nl_high} should be ≥ low-wind {nl_low}"
        )
        assert np.any(nl_high > nl_low), "expected at least one band where high>low"

    def test_shipping_wenz_levels_ordered(self):
        """Wenz shipping levels: low < medium < high at low frequencies."""
        f = np.array([10.0, 50.0, 100.0])
        nl_low = shipping_noise_wenz(f, shipping_level="low")
        nl_med = shipping_noise_wenz(f, shipping_level="medium")
        nl_high = shipping_noise_wenz(f, shipping_level="high")
        # Levels are offset by +5 dB per category in Wenz's parameterization
        assert np.all(nl_low < nl_med)
        assert np.all(nl_med < nl_high)


class TestAmbientNoiseSimulator:
    def test_default_frequency_grid_is_log_spaced_decade(self):
        sim = AmbientNoiseSimulator()
        assert sim.freq.shape == (1000,)
        # Default is 1 Hz .. 100 kHz log-spaced
        assert sim.freq[0] == pytest.approx(1.0)
        assert sim.freq[-1] == pytest.approx(1.0e5)
        # Log-spaced → ratios between consecutive entries are constant
        ratios = sim.freq[1:] / sim.freq[:-1]
        assert np.allclose(ratios, ratios[0], rtol=1e-6)

    def test_unknown_model_raises(self):
        sim = AmbientNoiseSimulator()
        with pytest.raises(ValueError, match="Unknown model"):
            sim.add_wind("not_a_model")

    def test_compute_combines_sources(self):
        """Composite spectrum must dominate any single component (incoherent sum)."""
        f = np.logspace(2, 4, 100)
        sim = AmbientNoiseSimulator(freq=f)
        sim.add_wind("knudsen", wind_speed=10.0)
        sim.add_shipping("wenz", shipping_level="medium")
        sim.add_thermal("mellen")
        components, total = sim.compute()

        assert set(components.keys())  # at least one labeled component
        assert total.shape == f.shape
        finite = np.isfinite(total)
        assert finite.any(), "no finite values in composite spectrum"

        # Composite must be ≥ each individual component at every frequency
        # where both are finite (incoherent sum can only add power).
        for label, comp in components.items():
            mask = np.isfinite(comp) & finite & (comp > 0)
            if not mask.any():
                continue
            assert np.all(total[mask] >= comp[mask] - 1e-6), (
                f"composite below component '{label}'"
            )

    def test_plot_returns_figure(self):
        """`plot()` must return a matplotlib (Figure, Axes) pair."""
        import matplotlib.pyplot as plt

        sim = AmbientNoiseSimulator(freq=np.logspace(2, 4, 50))
        sim.add_wind("knudsen", wind_speed=5.0)
        sim.add_thermal("mellen")
        sim.compute()
        fig, ax = sim.plot(title="test")
        try:
            assert fig is not None
            assert ax is not None
        finally:
            plt.close(fig)
