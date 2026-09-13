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
        alpha_dB_per_km = (
            0.11 * f_khz**2 / (1 + f_khz**2)
            + 44.0 * f_khz**2 / (4100.0 + f_khz**2)
            + 2.75e-4 * f_khz**2
            + 0.003
        )
        range_km_max = float(receiver.ranges[-1]) / 1000.0
        expected_extra_dB = alpha_dB_per_km * range_km_max

        assert isinstance(result_thorp, Field)
        observed_extra = (
            np.mean(result_thorp.dB[:, -1]) - np.mean(result_no_atten.dB[:, -1])
        )
        # Sign must be right (Thorp adds loss, never reduces it).
        assert observed_extra > 0, (
            f"Thorp gave less loss than no-attenuation case: {observed_extra:.2f} dB"
        )
        # Magnitude must be the right order — within 10× of the predicted dB.
        # This is loose enough to absorb implementation differences (per-arrival
        # vs per-range application, alpha-formula variants) while still
        # catching unit confusion (which would be off by ~1000×).
        assert 0.1 * expected_extra_dB < observed_extra < 10 * expected_extra_dB, (
            f"Thorp absorption magnitude wrong: observed {observed_extra:.2f} dB "
            f"vs predicted {expected_extra_dB:.2f} dB at {range_km_max:.1f} km"
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

    def test_ram_deck_carries_thorp_as_dB_per_local_wavelength(
            self, shallow_env, shallow_env_thorp):
        """Every RAM backend applies ``env.absorption`` through a water
        block in dB per wavelength (ram.md §3). At 10 kHz Thorp is
        ``thorp_dB_per_km(f)/1000 · c/f`` per wavelength, the block sits at
        absolute depths inside the domain, and a lossless env or a zero
        constant writes no block. Deck-level — no binary runs."""
        from uacpy.models import RAM
        from uacpy.core.absorption import ConstantAbsorption, thorp_dB_per_km
        m = RAM(verbose=False, flat_earth=False)
        seg = m._collins_range_segments(shallow_env_thorp, 'ramgeo', 200.0,
                                        10000.0, dz=0.05)[0]
        block = seg['water_attn']
        expected = float(thorp_dB_per_km(10000.0)) / 1000.0 * 1500.0 / 10000.0
        assert block[0][0] == 0.0 and block[-1][0] == 200.0
        assert all(abs(v - expected) < 1e-12 * expected for _, v in block)
        assert 'water_attn' not in m._collins_range_segments(
            shallow_env, 'ramgeo', 200.0, 10000.0)[0]
        zero = Environment(name='zero', bathymetry=100.0, ssp=1500.0,
                           absorption=ConstantAbsorption(0.0))
        assert 'water_attn' not in m._collins_range_segments(
            zero, 'ramgeo', 200.0, 10000.0)[0]

    @pytest.mark.requires_binary
    @pytest.mark.parametrize('law', ['thorp', 'fg'])
    def test_every_bellhop_port_applies_the_same_law_as_the_fortran(
            self, shallow_env, law):
        """The C++ / CUDA port carried Francois-Garrison's boric-acid
        relaxation frequency with base 1 instead of 10 (a constant 2.8 kHz),
        1.34 x the Fortran's loss at 5 kHz while its Thorp agreed
        (third_party/MODIFICATIONS.md, bellhopcuda). The increment
        TL(law) - TL(lossless) of every available port must match the
        Fortran binary's to 3 %."""
        from uacpy.core.absorption import FrancoisGarrison
        from uacpy.models.bellhop import Bellhop as _B
        absorption = (Thorp() if law == 'thorp' else FrancoisGarrison(
            temperature_c=10.0, salinity_psu=35.0, pH=8.0, z_bar_m=50.0))
        lossy = Environment(name='lossy', bathymetry=100.0, ssp=1500.0,
                            bottom=shallow_env.bottom, absorption=absorption)
        src = Source(depths=30.0, frequencies=5000.0)
        rcv = Receiver(depths=[20.0, 30.0, 50.0, 70.0],
                       ranges=np.arange(1000.0, 5001.0, 500.0))

        def increment(backend):
            m = _B(backend=backend, verbose=False)
            tl = [np.asarray(m.run(e, src, rcv,
                                   run_mode=RunMode.COHERENT_TL).tl, float)
                  for e in (shallow_env, lossy)]
            return float(np.nanmedian(tl[1] - tl[0]))

        ref = increment('fortran')
        # medians over 1-5 km: Thorp 1.2 dB, Francois-Garrison 1.0 dB
        assert ref > 0.5, f"Fortran {law} increment {ref:.2f} dB is not measurable"
        for backend in ('cxx', 'cuda'):
            try:
                inc = increment(backend)
            except Exception as exc:          # port not built on this host
                pytest.skip(f"{backend}: {type(exc).__name__}")
            assert abs(inc / ref - 1.0) < 0.03, (
                f"Bellhop {backend} {law} increment {inc:.3f} dB vs Fortran "
                f"{ref:.3f} dB")

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
