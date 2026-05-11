"""Tests for :meth:`Modes.with_attenuation` perturbation +
:meth:`Modes.modal_propagation_loss` synthesis."""

import numpy as np
import pytest

from uacpy.core.results import Modes, PressureField


def _pekeris_modes(n_modes=3, water_depth=100.0, c0=1500.0, freq=50.0):
    """Synthetic Pekeris-fluid modes — sinusoidal eigenfunctions."""
    depths = np.linspace(0.0, water_depth, 51)
    phi = np.zeros((depths.size, n_modes))
    k = np.empty(n_modes, dtype=complex)
    omega = 2.0 * np.pi * freq
    for m in range(n_modes):
        kz = (m + 0.5) * np.pi / water_depth
        phi[:, m] = np.sin(kz * depths)
        k[m] = np.sqrt((omega / c0) ** 2 - kz ** 2 + 0j)
    return Modes(
        k=k, phi=phi, depths=depths,
        model='Test', frequencies=freq,
    )


class TestWithAttenuation:
    def test_zero_attenuation_keeps_real_k(self):
        modes = _pekeris_modes()
        out = modes.with_attenuation(0.0)
        assert np.allclose(out.k.imag, 0.0)

    def test_uniform_attenuation_recovers_kratio_scaling(self):
        # For uniform c, ρ, the perturbation reduces to
        # α_m = (ω/(c·k_rm)) · α  =  (k₀/k_rm) · α    (in Np/m)
        modes = _pekeris_modes()
        alpha_db_m = 0.01
        alpha_np_m = alpha_db_m * np.log(10.0) / 20.0
        out = modes.with_attenuation(
            alpha_db_m, sound_speed_z=1500.0, density_z=1.0,
        )
        omega = 2.0 * np.pi * float(modes.f0)
        k0 = omega / 1500.0
        expected = (k0 / modes.k.real) * alpha_np_m
        assert np.allclose(out.k.imag, expected, rtol=1e-6)

    def test_thorp_absorption(self):
        from uacpy.core.absorption import Thorp
        modes = _pekeris_modes(freq=1000.0)
        out = modes.with_attenuation(absorption=Thorp())
        assert np.all(out.k.imag > 0)

    def test_francois_garrison_absorption(self):
        from uacpy.core.absorption import FrancoisGarrison
        modes = _pekeris_modes(freq=1000.0)
        out = modes.with_attenuation(
            absorption=FrancoisGarrison(
                temperature_c=15.0, salinity_psu=35.0, pH=8.1, z_bar_m=50.0,
            ),
        )
        assert np.all(out.k.imag > 0)

    def test_one_of_attenuation_args_required(self):
        from uacpy.core.absorption import Thorp
        modes = _pekeris_modes()
        with pytest.raises(ValueError, match="exactly one of"):
            modes.with_attenuation()
        with pytest.raises(ValueError, match="exactly one of"):
            modes.with_attenuation(0.005, absorption=Thorp())

    def test_depth_dependent_alpha_weighted_by_phi_square(self):
        modes = _pekeris_modes()
        a = np.where(modes.depths < 50.0, 0.0, 2.0)
        out = modes.with_attenuation(a)
        max_np_m = 2.0 * np.log(10.0) / 20.0
        # Allow some slack — the (k₀/k_rm) factor pushes the result a
        # bit; bound it loosely below the deepest uniform value.
        assert 0.0 < out.k[0].imag < 2.0 * max_np_m

    def test_shape_mismatch_raises(self):
        modes = _pekeris_modes()
        with pytest.raises(ValueError, match="must match depths"):
            modes.with_attenuation(np.array([0.001, 0.002]))


class TestModalPropagationLoss:
    def test_returns_complex_pressure_field(self):
        modes = _pekeris_modes()
        ranges = np.linspace(100.0, 5000.0, 10)
        depths = np.linspace(10.0, 90.0, 5)
        pf = modes.modal_propagation_loss(
            source_depth=20.0, receiver_depths=depths, ranges_m=ranges,
        )
        assert isinstance(pf, PressureField)
        assert pf.units == 'complex'
        assert pf.data.shape == (5, 10)
        assert np.all(np.isfinite(pf.data))

    def test_cylindrical_spreading_envelope(self):
        # With zero damping the envelope should fall like 1/sqrt(r);
        # check |P|·sqrt(r) is roughly constant after a few wavelengths.
        modes = _pekeris_modes()
        ranges = np.linspace(2000.0, 8000.0, 25)
        pf = modes.modal_propagation_loss(
            source_depth=50.0, receiver_depths=np.array([50.0]), ranges_m=ranges,
        )
        envelope = np.abs(pf.data[0]) * np.sqrt(ranges)
        # Tolerate modal interference — fluctuations of ~a factor of 5
        # are expected, but the trend should be flat (no monotonic decay).
        slope = np.polyfit(ranges, envelope, 1)[0]
        rel_drift = abs(slope) * (ranges[-1] - ranges[0]) / np.mean(envelope)
        assert rel_drift < 0.5

    def test_attenuation_decays_field(self):
        # Attenuated modes give smaller |P| than lossless at the same range.
        m_loss = _pekeris_modes()
        m_at = m_loss.with_attenuation(0.005)  # 0.005 dB/m
        pf_loss = m_loss.modal_propagation_loss(
            source_depth=50.0,
            receiver_depths=np.array([50.0]),
            ranges_m=np.array([10000.0]),
        )
        pf_at = m_at.modal_propagation_loss(
            source_depth=50.0,
            receiver_depths=np.array([50.0]),
            ranges_m=np.array([10000.0]),
        )
        assert abs(pf_at.data[0, 0]) < abs(pf_loss.data[0, 0])
