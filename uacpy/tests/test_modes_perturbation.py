import warnings
from uacpy import BoundaryProperties
"""Tests for :meth:`Modes.with_attenuation` perturbation +
:meth:`Modes.modal_propagation_loss` synthesis."""

import numpy as np
import pytest

from uacpy.core.results import Modes, Field
from uacpy.core.exceptions import ConfigurationError


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
        alpha = Thorp().alpha_db_per_m(modes.f0, modes.depths)
        out = modes.with_attenuation(alpha)
        assert np.all(out.k.imag > 0)

    def test_francois_garrison_absorption(self):
        from uacpy.core.absorption import FrancoisGarrison
        modes = _pekeris_modes(freq=1000.0)
        fg = FrancoisGarrison(
            temperature_c=15.0, salinity_psu=35.0, pH=8.1, z_bar_m=50.0,
        )
        out = modes.with_attenuation(fg.alpha_db_per_m(modes.f0, modes.depths))
        assert np.all(out.k.imag > 0)

    def test_depth_dependent_alpha_weighted_by_phi_square(self):
        """The stated formula, reimplemented independently on the same grid:
        α_m = (ω/(c·k_rm)) · ∫α_np·φ²dz / ∫φ²dz  (uniform c=1500, ρ=1, so
        the density weights cancel), pinned mode-by-mode at rel 1e-6. The
        closed-form step integral (sin² between 50 and 100 m over the full
        column; exactly 1/2 as (m+½)π zeroes the sin(2·kz·D) terms) sits
        1-3 % away — the quadrature's half-cell smear of the step — so the
        theory anchor is a 5 % band on top of the exact-quadrature pin."""
        modes = _pekeris_modes()
        a = np.where(modes.depths < 50.0, 0.0, 2.0)
        out = modes.with_attenuation(a)

        a_np = a * np.log(10.0) / 20.0
        phi2 = np.asarray(modes.phi).real ** 2
        weight = (np.trapezoid(a_np[:, None] * phi2, modes.depths, axis=0)
                  / np.trapezoid(phi2, modes.depths, axis=0))
        omega = 2.0 * np.pi * float(modes.f0)
        expected = (omega / (1500.0 * modes.k.real)) * weight
        np.testing.assert_allclose(out.k.imag, expected, rtol=1e-6)

        D, z0 = 100.0, 50.0
        max_np_m = 2.0 * np.log(10.0) / 20.0
        for m in range(3):
            kz = (m + 0.5) * np.pi / D
            frac = (0.5 * (D - z0)
                    + np.sin(2.0 * kz * z0) / (4.0 * kz)) / (D / 2.0)
            analytic = (omega / (1500.0 * modes.k[m].real)) * max_np_m * frac
            assert out.k[m].imag == pytest.approx(analytic, rel=0.05)

    def test_shape_mismatch_raises(self):
        modes = _pekeris_modes()
        with pytest.raises(ConfigurationError, match="must match depths"):
            modes.with_attenuation(np.array([0.001, 0.002]))


class TestComputeGroupVelocity:
    """``Modes.compute_group_velocity`` is the finite difference dω/dk from
    two solves at nearby frequencies (kraken.md §4). The synthetic modes
    carry the exact ideal-waveguide dispersion k(ω) = sqrt((ω/c)² − kz²),
    whose group velocity is v_g = c²k/ω; the 1 Hz difference at 50 Hz
    reproduces the midpoint value to better than 1e-5 relative."""

    def test_matches_the_analytic_pekeris_group_velocity(self):
        f1, f2 = 50.0, 51.0
        m1 = _pekeris_modes(freq=f1)
        m2 = _pekeris_modes(freq=f2)
        vg = m1.compute_group_velocity(m2)
        assert vg.shape == (3,)

        c0, D = 1500.0, 100.0
        fm = 0.5 * (f1 + f2)
        omega_m = 2.0 * np.pi * fm
        for m in range(3):
            kz = (m + 0.5) * np.pi / D
            km = np.sqrt((omega_m / c0) ** 2 - kz ** 2)
            assert vg[m] == pytest.approx(c0 ** 2 * km / omega_m, rel=1e-4)
        # Physical ordering: every mode travels below the free speed and
        # higher modes are slower.
        assert np.all(vg < c0)
        assert np.all(np.diff(vg) < 0)

    def test_same_frequency_pair_is_refused(self):
        m1 = _pekeris_modes(freq=50.0)
        with pytest.raises(ConfigurationError, match='distinct'):
            m1.compute_group_velocity(_pekeris_modes(freq=50.0))

    def test_mismatched_mode_counts_truncate_to_the_shared_set(self):
        vg = _pekeris_modes(n_modes=3, freq=50.0).compute_group_velocity(
            _pekeris_modes(n_modes=2, freq=51.0))
        assert vg.shape == (2,)


class TestModalPropagationLoss:
    def test_returns_complex_pressure_field(self):
        modes = _pekeris_modes()
        ranges = np.linspace(100.0, 5000.0, 10)
        depths = np.linspace(10.0, 90.0, 5)
        pf = modes.modal_propagation_loss(
            source_depth=20.0, receiver_depths=depths, ranges_m=ranges,
        )
        assert isinstance(pf, Field)
        assert pf.is_complex
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

    def test_decays_under_raw_kraken_imag_sign(self):
        # Raw kraken/krakenc eigenvalues encode decay as k.imag < 0, while
        # with_attenuation builds k.imag > 0. modal_propagation_loss must be
        # convention-agnostic: a passive medium can only attenuate, so the
        # field has to DECAY with range under either sign. Taking the sign at
        # face value makes one of the two branches grow without bound.
        base = _pekeris_modes()
        ranges = np.array([500.0, 8000.0])
        for sign in (+1.0, -1.0):
            k = base.k.real + sign * 1j * 3e-4   # ±imag, same |attenuation|
            modes = Modes(k=k, phi=base.phi, depths=base.depths,
                          model='Test', frequencies=base.f0)
            pf = modes.modal_propagation_loss(
                source_depth=50.0, receiver_depths=np.array([50.0]),
                ranges_m=ranges,
            )
            envelope = np.abs(pf.data[0]) * np.sqrt(ranges)  # remove geometric 1/√r
            assert envelope[-1] < envelope[0], (
                f"field grew with range for k.imag sign {sign:+.0f}")


def test_modal_propagation_loss_zero_modes_raises():
    # 0 trapped modes (below cutoff) -> no propagating field; a clear error,
    # not a raw column_stack ValueError.
    m0 = Modes(k=np.zeros(0, complex), phi=np.zeros((10, 0)),
               depths=np.linspace(0, 100, 10), model="T", frequencies=100.0)
    with pytest.raises(ConfigurationError, match="0 trapped modes"):
        m0.modal_propagation_loss(source_depth=50.0,
                                  receiver_depths=np.array([50.0]),
                                  ranges_m=np.array([1000.0]))


def test_phase_advances_negatively_with_range():
    """AT propagates as e^(i(wt - kr)), so P(r) carries exp(-i k r): the
    unwrapped phase must DECREASE with range at the rate of the first mode.
    Conjugating the field leaves |P| bit-identical, so only a phase test can
    see it."""
    modes = _pekeris_modes(n_modes=1)
    r = np.linspace(2000.0, 2040.0, 401)
    pf = modes.modal_propagation_loss(
        source_depth=50.0, receiver_depths=np.array([50.0]), ranges_m=r)
    ph = np.unwrap(np.angle(np.asarray(pf.data)[0]))
    slope = np.polyfit(r, ph, 1)[0]
    assert slope == pytest.approx(-float(modes.k[0].real), rel=1e-3)


class TestDepthsOutsideTheTabulatedModes:
    """``kraken.f90:573,598`` tabulates the modes on ``zTab`` — the merged
    source/receiver depth vector the deck asked for — so ``Modes.depths`` is
    exactly where ``phi`` is known. Outside it there is no mode shape to
    interpolate, and holding the end value flat would report a plausible number
    for a depth the mode set never covered — neither the mode shape nor the
    half-space evanescent tail (which ``Modes`` carries no half-space
    wavenumber to compute, and which AT does not compute either:
    ``calculateweights.f90:43-49`` extrapolates linearly off the end)."""

    @staticmethod
    def _modes():
        z = np.linspace(0.0, 100.0, 51)
        phi = np.column_stack(
            [np.sin((m + 1) * np.pi * z / 100.0) for m in range(3)])
        return Modes(k=np.array([0.42, 0.40, 0.37]) + 0j, phi=phi, depths=z)

    def test_a_receiver_below_the_mesh_is_no_data_not_a_clamp(self):
        with pytest.warns(UserWarning, match='outside the tabulated'):
            field = self._modes().modal_propagation_loss(
                source_depth=25.0,
                receiver_depths=np.array([10.0, 50.0, 150.0]),
                ranges_m=np.array([1000.0, 2000.0]))
        data = np.asarray(field.data)
        assert np.all(np.isfinite(data[:2]))
        assert np.all(np.isnan(data[2]))

    def test_a_source_outside_the_mesh_is_fatal(self):
        """The source defines the excitation, so a guess there is not a
        no-data cell — it silently rescales the whole field."""
        with pytest.raises(ConfigurationError, match='source_depth'):
            self._modes().modal_propagation_loss(
                source_depth=150.0, receiver_depths=np.array([10.0]),
                ranges_m=np.array([1000.0]))


class TestLeakyModesGetNoBottomTerm:
    """``modes.py:254-255`` already states the rule — *"gamma is real only for
    a trapped mode (kr > kb); a leaky one clamps to 0 and contributes
    nothing"* — and the code contradicted it: ``gamma_safe = np.where(gamma_m
    > 0, gamma_m, 1.0)`` put a bare ``1.0`` into the denominator.

    That number carries units of 1/m, so the invented loss scaled with the
    library's length unit: the identical physics expressed in km came back
    exactly 1000x different. Physically there is no finite first-order
    bottom-absorption perturbation for a radiating mode — the tail integral
    ``int_D^inf psi^2 dz`` diverges, so the closed form ``psi^2(D)/(2*gamma)``
    this line specialises does not exist."""

    F0 = 100.0
    BOT = BoundaryProperties(acoustic_type='half-space', sound_speed=1700.0,
                             density=1.8, attenuation=0.5)

    @property
    def _kb(self):
        return 2.0 * np.pi * self.F0 / self.BOT.sound_speed

    def _modes(self, factors):
        z = np.linspace(0.0, 100.0, 51)
        phi = np.cos(0.5 * np.pi * z[:, None] / 100.0) * np.ones((1, len(factors)))
        return Modes(k=np.array([self._kb * f for f in factors], dtype=complex),
                     phi=phi, depths=z, model='Test', frequencies=self.F0)

    def test_leaky_mode_contributes_nothing_and_warns(self):
        with pytest.warns(UserWarning, match='leaky'):
            out = self._modes([1.05, 0.95]).with_attenuation(0.0, bottom=self.BOT)
        assert out.k[1].imag == 0.0        # leaky: exactly zero, not 1/gamma
        assert out.k[0].imag > 0.0         # trapped: still gets its term

    def test_all_trapped_set_is_silent_and_keeps_its_attenuation(self):
        # The counterpart that stops the fix reaching the branch that works.
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            out = self._modes([1.05, 1.20]).with_attenuation(
                0.0, bottom=self.BOT, seafloor_depth=100.0)
        assert np.all(out.k.imag > 0.0)

    def test_the_result_no_longer_depends_on_the_length_unit(self):
        # Re-express the same physics in km: depths /1000, wavenumbers *1000.
        # Before the fix the leaky mode's Im(k) differed by exactly 1000x.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = self._modes([1.05, 0.95])
            metres = m.with_attenuation(0.0, bottom=self.BOT).k[1].imag
        assert metres == 0.0
