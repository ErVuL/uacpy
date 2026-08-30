"""Volume-attenuation-focused tests."""

import pytest
import numpy as np

from uacpy.models import Bellhop, Kraken, Scooter
from uacpy import Field
from uacpy.core.results import Modes
from uacpy.models.base import RunMode
from uacpy.core import Environment, Source, Receiver
from uacpy.core.absorption import Thorp

pytestmark = pytest.mark.requires_binary


class TestVolumeAttenuation:
    """``env.absorption`` reaches the solvers and adds loss.

    Only :func:`test_bellhop_thorp_attenuation` checks a magnitude; the
    Kraken / Scooter cases are reachability smoke tests (the model accepts an
    absorbing environment and returns a finite result).
    """

    @pytest.fixture
    def shallow_env(self):
        """Shallow water environment without volume absorption."""
        return Environment(
            name="atten_test",
            bathymetry=100.0,
            ssp=1500.0,
        )

    @pytest.fixture
    def shallow_env_thorp(self):
        """Shallow water environment with Thorp volume absorption."""
        return Environment(
            name="atten_test_thorp",
            bathymetry=100.0,
            ssp=1500.0,
            absorption=Thorp(),
        )

    @pytest.fixture
    def high_freq_source(self):
        """High frequency source where attenuation is significant."""
        return Source(depths=50.0, frequencies=10000.0)  # 10 kHz

    @pytest.fixture
    def low_freq_source(self):
        """Low frequency source where attenuation is minimal."""
        return Source(depths=50.0, frequencies=100.0)  # 100 Hz

    @pytest.fixture
    def receiver(self):
        return Receiver(depths=[50.0], ranges=[1000.0, 3000.0, 5000.0])

    @pytest.mark.requires_binary
    def test_bellhop_thorp_attenuation(self, shallow_env, shallow_env_thorp,
                                       high_freq_source, receiver):
        """Test Bellhop with Thorp attenuation formula.

        At 10 kHz, Thorp absorption is 1.19 dB/km, i.e. 5.9 dB of extra
        one-way loss over the 5 km longest range. We assert the depth-mean
        difference there is within the predicted-times-[0.1, 10] band — a
        sign-error or unit confusion would not satisfy that band.
        """
        bellhop = Bellhop(verbose=False)

        result_no_atten = bellhop.run(
            env=shallow_env, source=high_freq_source, receiver=receiver,
            run_mode=RunMode.COHERENT_TL,
        )
        result_thorp = bellhop.run(
            env=shallow_env_thorp, source=high_freq_source, receiver=receiver,
            run_mode=RunMode.COHERENT_TL,
        )

        # Thorp formula at 10 kHz (f in kHz):
        #   alpha = 0.11 f^2/(1+f^2) + 44 f^2/(4100+f^2)
        #         + 2.75e-4 f^2 + 0.003   [dB/km]
        f_khz = high_freq_source.frequencies[0] / 1000.0
        alpha_db_per_km = (
            0.11 * f_khz**2 / (1 + f_khz**2)
            + 44.0 * f_khz**2 / (4100.0 + f_khz**2)
            + 2.75e-4 * f_khz**2
            + 0.003
        )
        range_km_max = float(receiver.ranges[-1]) / 1000.0
        expected_extra_db = alpha_db_per_km * range_km_max

        assert isinstance(result_thorp, Field)
        observed_extra = (
            np.mean(result_thorp.db[:, -1]) - np.mean(result_no_atten.db[:, -1])
        )
        # Sign must be right (Thorp adds loss, never reduces it).
        assert observed_extra > 0, (
            f"Thorp gave less loss than no-attenuation case: {observed_extra:.2f} dB"
        )
        # Magnitude must be the right order — within 10× of the predicted dB.
        # This is loose enough to absorb implementation differences (per-arrival
        # vs per-range application, alpha-formula variants) while still
        # catching unit confusion (which would be off by ~1000×).
        assert 0.1 * expected_extra_db < observed_extra < 10 * expected_extra_db, (
            f"Thorp absorption magnitude wrong: observed {observed_extra:.2f} dB "
            f"vs predicted {expected_extra_db:.2f} dB at {range_km_max:.1f} km"
        )

    @pytest.mark.requires_binary
    def test_kraken_thorp_attenuation(self, shallow_env, shallow_env_thorp,
                                      high_freq_source, receiver):
        """An absorbing environment reaches the modes path and adds modal
        loss: at 10 kHz Thorp is ~1.19 dB/km ≈ 1.4e-4 nepers/m of extra
        Im(k) on every mode, so the Thorp run's mean Im(k) must sit above
        the no-absorption run's (the bottom's own loss is present in both)."""
        kraken = Kraken(verbose=False)
        result = kraken.compute_modes(
            env=shallow_env_thorp,
            source=high_freq_source,
        )
        assert isinstance(result, Modes)
        assert result.k is not None
        plain = kraken.compute_modes(env=shallow_env,
                                     source=high_freq_source)
        im_thorp = np.imag(np.asarray(result.k))
        im_plain = np.imag(np.asarray(plain.k))
        assert im_thorp.size and im_plain.size
        # Im(k) <= 0 in the decaying convention: more absorption pushes it
        # further NEGATIVE, so the comparison is on magnitudes.
        assert float(np.abs(im_thorp).mean()) > float(np.abs(im_plain).mean()), (
            "Thorp volume absorption did not increase the modal Im(k)")

    def test_ram_warns_that_absorption_is_ignored(self, shallow_env,
                                                  shallow_env_thorp):
        """No RAM backend consumes water-column volume attenuation
        (ram.md §7); an absorbing env warns instead of silently running a
        lossless water column. Unit-tests the helper — no binary runs."""
        import warnings as _w
        from uacpy.models import RAM
        from uacpy.core.absorption import ConstantAbsorption
        m = RAM(verbose=False)
        with pytest.warns(UserWarning, match='env.absorption'):
            m._warn_on_dropped_absorption(shallow_env_thorp)
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter('always')
            m._warn_on_dropped_absorption(shallow_env)
            m._warn_on_dropped_absorption(Environment(
                name='zero', bathymetry=100.0, ssp=1500.0,
                absorption=ConstantAbsorption(0.0)))
        assert not [w for w in caught
                    if 'env.absorption' in str(w.message)]

    @pytest.mark.requires_binary
    def test_frequency_dependent_attenuation(self, shallow_env_thorp,
                                             low_freq_source, high_freq_source,
                                             receiver):
        """The same absorbing environment runs at 100 Hz and 10 kHz and both
        fields are finite. Thorp spans four orders of magnitude in alpha across
        that pair, so an alpha that overflowed or went NaN at one end shows up
        here; the ordering of the two losses is not asserted."""
        bellhop = Bellhop(verbose=False)

        result_low = bellhop.run(
            env=shallow_env_thorp,
            source=low_freq_source,
            receiver=receiver,
            run_mode=RunMode.COHERENT_TL,
        )

        result_high = bellhop.run(
            env=shallow_env_thorp,
            source=high_freq_source,
            receiver=receiver,
            run_mode=RunMode.COHERENT_TL,
        )

        assert isinstance(result_low, Field)
        assert isinstance(result_high, Field)
        assert np.all(np.isfinite(result_low.data))
        assert np.all(np.isfinite(result_high.data))

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_attenuation_with_scooter(self, shallow_env_thorp,
                                      high_freq_source, receiver):
        """The spectral-integral path also accepts an absorbing environment and
        returns a finite field. The size of the added loss is not asserted."""
        scooter = Scooter(verbose=False)
        result = scooter.run(
            env=shallow_env_thorp,
            source=high_freq_source,
            receiver=receiver,
        )
        assert isinstance(result, Field)
        assert np.all(np.isfinite(result.data))
