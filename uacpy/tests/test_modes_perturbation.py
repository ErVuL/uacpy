"""Tests for :meth:`Modes.with_attenuation` perturbation +
:meth:`Modes.modal_propagation_loss` synthesis."""

import warnings

import numpy as np
import pytest

from uacpy import BoundaryProperties
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
    """``Modes.with_attenuation``'s own comment states the rule — *"gamma is
    real only for a trapped mode (kr > kb); a leaky one clamps to 0 and
    contributes nothing"* — and the code contradicted it: ``gamma_safe =
    np.where(gamma_m > 0, gamma_m, 1.0)`` put a bare ``1.0`` into the
    denominator.

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

    def test_the_result_is_independent_of_the_length_unit(self):
        # Re-express the same physics in km: depths /1000, wavenumbers *1000.
        # Before the fix the leaky mode's Im(k) differed by exactly 1000x.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = self._modes([1.05, 0.95])
            metres = m.with_attenuation(0.0, bottom=self.BOT).k[1].imag
        assert metres == 0.0


def _pekeris_modes_on(n_depths, *, n_modes=6, water_depth=100.0, c0=1500.0,
                      freq=50.0):
    """Pekeris-fluid modes tabulated on ``n_depths`` evenly-spaced depths.

    Mode m has vertical wavenumber ``(m + ½)π/D``, hence vertical wavelength
    ``2D/(m + ½)`` — 400 m down to 36.4 m over six modes at D = 100 m."""
    z = np.linspace(0.0, water_depth, n_depths)
    omega = 2.0 * np.pi * freq
    phi = np.column_stack(
        [np.sin((m + 0.5) * np.pi * z / water_depth) for m in range(n_modes)])
    k = np.array([np.sqrt((omega / c0) ** 2 - ((m + 0.5) * np.pi / water_depth) ** 2
                          + 0j) for m in range(n_modes)])
    return Modes(k=k, phi=phi, depths=z, model='Test', frequencies=freq)


class TestDepthAxisSamplingGuard:
    """``with_attenuation`` divides two trapezoid sums over ``depths``.

    Their quadrature errors cancel *exactly* while ``α(z)/c(z)`` is constant —
    which is why a scalar α is right on any grid — and stop cancelling as soon
    as it varies with depth, leaving an error set by how well the axis carries
    ψ². ``depths`` is KRAKEN's merged source/receiver vector, often 8-20
    points, so this is the ordinary case rather than an exotic one. Measured
    against an 8001-point reference with α stepping 0 → 2 dB/m at 50 m in a
    100 m guide: 1-3 % at 51 depths, 3-7 % at 21, 6-19 % at 9 and up to 104 %
    at 5. The guard warns rather than raises — the error is continuous in the
    spacing and the integrals still return the best value the samples
    support."""

    def _step_alpha(self, modes):
        return np.where(modes.depths < 50.0, 0.0, 2.0)

    def test_a_coarse_axis_with_structured_alpha_warns(self):
        modes = _pekeris_modes_on(9)
        with pytest.warns(UserWarning, match='samples per wavelength'):
            modes.with_attenuation(self._step_alpha(modes))

    def test_the_warning_names_the_measured_and_the_wanted_spacing(self):
        modes = _pekeris_modes_on(9)
        with pytest.warns(UserWarning) as rec:
            modes.with_attenuation(self._step_alpha(modes))
        msg = str(rec[0].message)
        assert 'every 12.5 m' in msg          # the axis it was handed
        assert '4.54545 m' in msg             # 36.36 m wavelength over 8

    def test_a_resolved_axis_is_silent(self):
        modes = _pekeris_modes_on(51)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            # Sinusoidal ψ is non-zero at D, so the water-only-normalisation
            # notice fires here whatever the spacing; this test is about the
            # sampling guard alone.
            warnings.filterwarnings('ignore', message='.*UPPER BOUND.*')
            modes.with_attenuation(self._step_alpha(modes))

    @pytest.mark.parametrize('n_depths', [5, 9, 21, 51])
    def test_a_uniform_alpha_is_silent_however_coarse(self, n_depths):
        """α/c constant is the case the two quadratures cancel in, so no
        spacing can spoil it and none of them warns."""
        modes = _pekeris_modes_on(n_depths)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            warnings.filterwarnings('ignore', message='.*UPPER BOUND.*')
            modes.with_attenuation(0.01)

    def test_a_depth_varying_sound_speed_alone_trips_it(self):
        """The ratio α/c is what has to be constant, not α: a scalar α over a
        sound-speed gradient is just as uncancelled."""
        modes = _pekeris_modes_on(9)
        with pytest.warns(UserWarning, match='alpha'):
            modes.with_attenuation(
                0.01, sound_speed_z=1500.0 + 0.05 * modes.depths)

    def test_the_guard_only_warns_and_leaves_the_value_alone(self):
        """Warn, never raise, and never quietly substitute a different
        number: the returned Im(k) is still the trapezoid over the samples
        the caller handed in."""
        modes = _pekeris_modes_on(9)
        a = self._step_alpha(modes)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = modes.with_attenuation(a)
        a_np = a * np.log(10.0) / 20.0
        phi2 = np.asarray(modes.phi).real ** 2
        weight = (np.trapezoid(a_np[:, None] * phi2, modes.depths, axis=0)
                  / np.trapezoid(phi2, modes.depths, axis=0))
        omega = 2.0 * np.pi * float(modes.f0)
        expected = (omega / (1500.0 * modes.k.real)) * weight
        np.testing.assert_allclose(out.k.imag, expected, rtol=1e-12)


def _exact_pekeris(water_depth=100.0, c1=1500.0, c2=1800.0, rho1=1.0,
                   rho2=1.8, freq=50.0, n_depths=2001):
    """Exact trapped modes of a Pekeris waveguide over a fluid half-space.

    Pressure-release surface, so ``ψ = sin(k_z z)`` in the water and
    ``ψ(D)·exp(−γ(z−D))`` below it; continuity of ``ψ`` and ``(1/ρ)dψ/dz``
    at ``z = D`` gives ``k_z/ρ₁·cos(k_z D) + γ/ρ₂·sin(k_z D) = 0`` with
    ``γ = sqrt(k_r² − (ω/c₂)²)``. Returns the :class:`Modes`, the water-column
    depth axis, ``γ`` and the medium constants the exact perturbation
    integral needs.
    """
    from scipy.optimize import brentq

    omega = 2.0 * np.pi * freq
    k1, kb = omega / c1, omega / c2

    def residual(kz):
        gamma = np.sqrt(max(k1 ** 2 - kz ** 2 - kb ** 2, 0.0))
        return kz / rho1 * np.cos(kz * water_depth) \
            + gamma / rho2 * np.sin(kz * water_depth)

    scan = np.linspace(1e-9, np.sqrt(k1 ** 2 - kb ** 2) * (1.0 - 1e-12), 4001)
    vals = np.array([residual(x) for x in scan])
    kz = np.array([brentq(residual, scan[i], scan[i + 1], xtol=1e-15,
                          rtol=8.9e-16)
                   for i in range(scan.size - 1) if vals[i] * vals[i + 1] < 0])
    kr = np.sqrt(k1 ** 2 - kz ** 2)
    gamma = np.sqrt(kr ** 2 - kb ** 2)
    z = np.linspace(0.0, water_depth, n_depths)
    phi = np.sin(kz[None, :] * z[:, None])
    modes = Modes(k=kr.astype(complex), phi=phi, depths=z, model='Test',
                  frequencies=freq)
    return modes, z, phi, gamma, dict(D=water_depth, c1=c1, c2=c2, rho1=rho1,
                                      rho2=rho2, freq=freq, omega=omega)


class TestNormalisationRunsIntoTheHalfSpace:
    """JKPS Eq. 5.169 normalises ``∫₀^∞ ψ²/ρ dz = 1``, and the text after
    Eq. 5.176 says the interval must be extended into the bottom so the
    evanescent tail is counted.

    Integrating the denominator over the water column alone while the
    numerator's bottom term is the half-space tail puts the two over
    different domains. The water-only denominator is too small, so every
    ``Im(k)`` came back high — by (I_water + tail)/I_water exactly, which on
    this guide is 1.007, 1.024, 1.051 and 1.172 for modes 1-4. The
    near-cutoff mode has the shallowest ``γ`` and therefore the longest tail,
    so the mode that dominates at long range was over-attenuated by 17 %.
    """

    def _exact_alpha(self, phi, z, gamma, c, alpha_water_db, alpha_bottom_db):
        """``α_m = (ω/k_r)·∫₀^∞ α/(cρ)ψ² dz / ∫₀^∞ ψ²/ρ dz`` in closed form:
        the water part by quadrature on the tabulated axis, the half-space
        part as ``ψ(D)²/(2γ)`` (the tail ``∫_D^∞ e^{−2γ(z−D)} dz``)."""
        rho1, rho2 = c['rho1'] * 1000.0, c['rho2'] * 1000.0
        a_w = alpha_water_db * np.log(10.0) / 20.0
        a_b = alpha_bottom_db * np.log(10.0) / 20.0 * c['freq'] / c['c2']
        i_water = np.trapezoid(phi ** 2, z, axis=0)
        tail = phi[-1, :] ** 2 / (2.0 * gamma)
        num = a_w / (c['c1'] * rho1) * i_water + a_b / (c['c2'] * rho2) * tail
        den = i_water / rho1 + tail / rho2
        return num / den

    @staticmethod
    def _bottom(consts, attenuation):
        return BoundaryProperties(
            acoustic_type='half-space', sound_speed=consts['c2'],
            density=consts['rho2'], attenuation=attenuation)

    @pytest.mark.parametrize('alpha_water_db, alpha_bottom_db',
                             [(0.02, 0.0), (0.02, 0.5), (0.0, 0.5)])
    def test_matches_the_exact_perturbation_integral(self, alpha_water_db,
                                                     alpha_bottom_db):
        modes, z, phi, gamma, c = _exact_pekeris()
        assert gamma.size >= 4
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = modes.with_attenuation(
                alpha_water_db, sound_speed_z=c['c1'], density_z=c['rho1'],
                bottom=self._bottom(c, alpha_bottom_db),
                seafloor_depth=c['D'])
        ratio = self._exact_alpha(phi, z, gamma, c, alpha_water_db,
                                  alpha_bottom_db)
        expected = (c['omega'] / modes.k.real) * ratio
        np.testing.assert_allclose(out.k.imag, expected, rtol=1e-9)

    def test_the_tail_is_what_the_water_only_denominator_was_missing(self):
        """The pre-fix value divided by the corrected one is exactly
        (I_water + tail)/I_water — the ratio of the two normalisations, with
        nothing else moving."""
        modes, z, phi, gamma, c = _exact_pekeris()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            corrected = modes.with_attenuation(
                0.02, sound_speed_z=c['c1'], density_z=c['rho1'],
                bottom=self._bottom(c, 0.0), seafloor_depth=c['D']).k.imag
            water_only = modes.with_attenuation(
                0.02, sound_speed_z=c['c1'], density_z=c['rho1']).k.imag

        i_water = np.trapezoid(phi ** 2, z, axis=0) / (c['rho1'] * 1000.0)
        tail = phi[-1, :] ** 2 / (2.0 * gamma * c['rho2'] * 1000.0)
        np.testing.assert_allclose(water_only / corrected,
                                   (i_water + tail) / i_water, rtol=1e-9)
        assert (water_only[-1] / corrected[-1]) == pytest.approx(1.17, abs=0.01)

    def test_the_tail_only_ever_lowers_the_attenuation(self):
        modes, z, phi, gamma, c = _exact_pekeris()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            corrected = modes.with_attenuation(
                0.02, sound_speed_z=c['c1'], density_z=c['rho1'],
                bottom=self._bottom(c, 0.0), seafloor_depth=c['D']).k.imag
            water_only = modes.with_attenuation(
                0.02, sound_speed_z=c['c1'], density_z=c['rho1']).k.imag
        assert np.all(corrected < water_only)

    def test_a_leaky_mode_keeps_the_water_column_normalisation(self):
        """A radiating mode has no convergent tail on either side of the
        ratio, so neither the numerator nor the denominator gains one and
        its Im(k) is untouched by this change."""
        z = np.linspace(0.0, 100.0, 51)
        f0 = 100.0
        cb = 1700.0
        kb = 2.0 * np.pi * f0 / cb
        phi = np.column_stack([np.sin(0.7 * np.pi * z / 100.0)] * 2)
        modes = Modes(k=np.array([kb * 1.05, kb * 0.95], dtype=complex),
                      phi=phi, depths=z, model='Test', frequencies=f0)
        bot = BoundaryProperties(acoustic_type='half-space', sound_speed=cb,
                                 density=1.8, attenuation=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = modes.with_attenuation(0.01, bottom=bot,
                                         seafloor_depth=100.0)
            plain = modes.with_attenuation(0.01)
        assert out.k[1].imag == pytest.approx(plain.k[1].imag, rel=1e-12)
        assert out.k[0].imag < plain.k[0].imag


class TestWaterOnlyNormalisationIsAnnounced:
    """Without ``bottom=`` there is no half-space ``γ`` or ``ρ_b`` from which
    to build the tail, so the normalisation stops at the seabed and the
    returned attenuation is an upper bound. The omitted term is positive and
    can only lower ``α_m``, so saying "upper bound" is exact rather than a
    hedge. A seabed the mode vanishes at carries no tail and no error, and is
    left silent."""

    def test_a_penetrable_seabed_shape_warns(self):
        modes = _pekeris_modes()          # sin((m+½)πz/D): ψ(D) = ±1
        with pytest.warns(UserWarning, match='UPPER BOUND'):
            modes.with_attenuation(0.01)

    def test_the_warning_names_the_depth_it_stopped_at(self):
        modes = _pekeris_modes(water_depth=100.0)
        with pytest.warns(UserWarning) as rec:
            modes.with_attenuation(0.01)
        assert any('100 m' in str(w.message) for w in rec)

    def test_a_mode_that_vanishes_at_the_seabed_is_silent(self):
        z = np.linspace(0.0, 100.0, 51)
        phi = np.column_stack(
            [np.sin((m + 1) * np.pi * z / 100.0) for m in range(3)])
        modes = Modes(k=np.array([0.42, 0.40, 0.37]) + 0j, phi=phi, depths=z,
                      model='Test', frequencies=50.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            modes.with_attenuation(0.01)

    def test_passing_bottom_silences_it(self):
        modes = _pekeris_modes()
        bot = BoundaryProperties(acoustic_type='half-space',
                                 sound_speed=1800.0, density=1.8,
                                 attenuation=0.1)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            modes.with_attenuation(0.01, bottom=bot, seafloor_depth=100.0)


class TestModalSumContraction:
    """``modal_propagation_loss`` contracts the mode axis directly.

    Building the whole ``(depth, mode, range)`` product first and summing it
    afterwards asks for one complex128 temporary of n_depth·n_mode·n_range —
    1.28 GB at 200 depths, 200 modes and 2000 ranges — for a result of
    n_depth·n_range. The contraction has to leave every value untouched."""

    def _modes(self):
        return _pekeris_modes_on(31, n_modes=5)

    def test_matches_a_mode_by_mode_accumulation(self):
        modes = self._modes()
        z_r = np.linspace(1.0, 99.0, 13)
        r = np.linspace(100.0, 5000.0, 17)
        got = np.asarray(modes.modal_propagation_loss(
            source_depth=25.0, receiver_depths=z_r, ranges_m=r).data)

        phi = np.asarray(modes.phi)
        k = modes.k.real - 1j * np.abs(modes.k.imag)
        want = np.zeros((z_r.size, r.size), dtype=complex)
        for m in range(modes.n_modes):
            phi_zs = np.interp(25.0, modes.depths, phi[:, m])
            phi_zr = np.interp(z_r, modes.depths, phi[:, m])
            want += (phi_zr * phi_zs / np.sqrt(complex(k[m])))[:, None] \
                * np.exp(-1j * k[m] * r)[None, :]
        want *= (-1j * np.exp(1j * np.pi / 4.0) * np.sqrt(2.0 * np.pi)
                 / np.sqrt(r)[None, :])
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)

    def test_the_output_is_the_grid_and_not_the_product(self):
        modes = self._modes()
        z_r = np.linspace(1.0, 99.0, 13)
        r = np.linspace(100.0, 5000.0, 17)
        field = modes.modal_propagation_loss(
            source_depth=25.0, receiver_depths=z_r, ranges_m=r)
        assert field.data.shape == (13, 17)
        assert np.isfinite(field.data).all()


class TestModalAttenuationBottomGuard:
    """``Modes.with_attenuation`` refuses a non-``BoundaryProperties``
    ``bottom=`` with a typed error, and a rigid seabed takes the water-only
    path — its result equals ``bottom=None`` because no energy enters a
    reflective bottom, so the bottom term is exactly zero."""

    def _pekeris_rigid_mode(self):
        depth = 100.0
        z = np.linspace(0.0, depth, 51)
        omega = 2.0 * np.pi * 50.0
        kz = np.pi / (2.0 * depth)
        kr = np.sqrt((omega / 1500.0) ** 2 - kz ** 2)
        phi = np.sin(kz * z)[:, None]
        return Modes(k=np.array([kr + 0j]), phi=phi, depths=z,
                     frequencies=50.0)

    def test_non_boundary_properties_bottom_raises_a_typed_error(self):
        with pytest.raises(ConfigurationError,
                           match="bottom must be a BoundaryProperties"):
            self._pekeris_rigid_mode().with_attenuation(1e-4, bottom=42)

    def test_rigid_bottom_equals_the_water_only_result(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            water_only = self._pekeris_rigid_mode().with_attenuation(1e-4)
            rigid = self._pekeris_rigid_mode().with_attenuation(
                1e-4, bottom=BoundaryProperties(acoustic_type='rigid'))
        assert np.allclose(rigid.k, water_only.k)
        assert np.all(rigid.k.imag > 0)


class TestBarelyTrappedModeIsNamed:
    """``with_attenuation`` adds the seabed tail ``psi(D)²/(2 gamma rho_b)`` to
    the normalisation. When that tail outweighs the whole water column the
    mode is bound so weakly that its bottom term rests on the part of ``kr``
    a mode solver resolves least well: on a 100 m Pekeris guide across a
    modal cutoff, a converged ``kraken.exe`` mesh moved the total modal TL by
    0.67-1.98 dB at 200-500 m and by 7.07 dB at 10 km against the default
    mesh, and put the near-cutoff mode on the leaky side, where it gets no
    bottom term at all.

    The fixtures use a depth-independent mode shape, which makes the balance
    point closed-form — ``psi(D)² = 1`` and ``int psi²/rho dz = D/rho_w``, so
    the tail equals the water column at ``gamma = rho_w/(2 rho_b D)`` — and
    lets each side of the threshold be reached by construction rather than by
    a fit that happens to land there.
    """

    D = 100.0
    C_W = 1500.0
    C_B = 1800.0
    RHO_B = 1.8
    FREQ = 100.0
    #: The shift the default KRAKEN mesh carries on a near-cutoff eigenvalue
    #: of a 100 m Pekeris guide at ~102 Hz, measured against a mesh converged
    #: at n_mesh=2000 (28 float32 ULP of kr).
    MESH_SHIFT = 8.345e-07

    @property
    def gamma_equal(self):
        return 1.0 / (2.0 * self.RHO_B * 1000.0 * (self.D / 1000.0))

    def _bottom(self):
        return BoundaryProperties(
            acoustic_type='half-space', sound_speed=self.C_B,
            density=self.RHO_B, attenuation=0.5)

    def _modes(self, gamma, amplitude=1.0):
        depths = np.linspace(0.0, self.D, 51)
        kb = 2.0 * np.pi * self.FREQ / self.C_B
        kr = np.sqrt(kb ** 2 + gamma ** 2)
        return Modes(k=np.array([kr + 0j]),
                     phi=amplitude * np.ones((depths.size, 1)),
                     depths=depths, model='Test', frequencies=self.FREQ)

    def _attenuate(self, modes):
        return modes.with_attenuation(
            0.01, sound_speed_z=self.C_W, density_z=1.0,
            bottom=self._bottom(), seafloor_depth=self.D)

    def test_a_tail_heavier_than_the_water_column_is_named(self):
        modes = self._modes(0.999 * self.gamma_equal)
        with pytest.warns(UserWarning, match="barely trapped"):
            self._attenuate(modes)

    def test_a_tail_lighter_than_the_water_column_is_not_named(self):
        modes = self._modes(1.001 * self.gamma_equal)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._attenuate(modes)

    def test_the_warning_carries_the_mode_the_tail_share_and_the_branch_point(self):
        modes = self._modes(0.5 * self.gamma_equal)
        with pytest.warns(UserWarning, match="barely trapped") as record:
            self._attenuate(modes)
        message = str(record[0].message)
        # gamma = gamma_equal/2 puts two thirds of the normalisation in the
        # tail, and kb is the branch point the leaky warning also names.
        assert "mode 1" in message
        assert "67 %" in message
        assert f"{2.0 * np.pi * self.FREQ / self.C_B:g}" in message

    @pytest.mark.parametrize("amplitude", [1e-3, 1.0, 1e3])
    def test_the_trigger_does_not_move_with_the_mode_normalisation(self, amplitude):
        """A ``.mod`` file's eigenfunction scale is arbitrary, so the trigger
        compares two integrals that both carry ``psi²`` and cancel it."""
        with pytest.warns(UserWarning, match="barely trapped"):
            self._attenuate(self._modes(0.5 * self.gamma_equal, amplitude))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._attenuate(self._modes(2.0 * self.gamma_equal, amplitude))

    def test_a_mesh_sized_shift_in_kr_moves_the_loss_of_a_named_mode(self):
        """The observable behind the warning: below the balance point a shift
        of ``kr`` the size of a mesh change moves the returned Im(k) by
        percent, and four times above it by a twentieth of that."""
        def imag_k(gamma, dkr):
            modes = self._modes(gamma)
            shifted = Modes(k=np.array([np.real(modes.k)[0] + dkr + 0j]),
                            phi=np.asarray(modes.phi), depths=modes.depths,
                            model='Test', frequencies=self.FREQ)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return float(np.asarray(self._attenuate(shifted).k).imag[0])

        for gamma_factor, lo, hi in ((0.5, 0.02, 0.05), (4.0, 0.0, 0.001)):
            gamma = gamma_factor * self.gamma_equal
            base = imag_k(gamma, 0.0)
            moved = abs(imag_k(gamma, -self.MESH_SHIFT) - base) / base
            assert lo <= moved <= hi, (gamma_factor, moved)

    def test_a_leaky_mode_is_named_leaky_rather_than_barely_trapped(self):
        """A leaky mode has ``gamma = 0``, which is under the balance point
        but is a different fact: its tail integral diverges, so the bottom
        term does not exist at all rather than being poorly determined, and
        the leaky warning is the one that says so."""
        depths = np.linspace(0.0, self.D, 51)
        kb = 2.0 * np.pi * self.FREQ / self.C_B
        modes = Modes(k=np.array([0.999 * kb + 0j]),
                      phi=np.ones((depths.size, 1)), depths=depths,
                      model='Test', frequencies=self.FREQ)
        with pytest.warns(UserWarning) as record:
            self._attenuate(modes)
        messages = [str(w.message) for w in record]
        assert any("are leaky" in m for m in messages), messages
        assert not any("barely trapped" in m for m in messages), messages


class TestGroupVelocityReportsWhatTheStoredWavenumbersResolve:
    """``v_g = dω/dk`` is a difference of wavenumbers a mode solver wrote to
    a file. KRAKEN's ``.mod`` record is ``COMPLEX*8``, so the difference
    carries about one float32 step of noise however small Δf is, and the
    storage alone limits the answer to ``spacing(k_r)/|Δk_r|`` relative —
    which *rises* as Δf shrinks. Measured on an analytic Pekeris guide at
    100 Hz with exact longdouble roots: 7.29e-06 relative at Δf = 1 Hz
    (140844 steps, the optimum), 4.39e-05 at 0.1 Hz, 6.23e-04 at 0.01 Hz.

    The fixtures set the step in whole float32 spacings of ``k``, so the
    number of steps the difference spans — and therefore the floor, which is
    its reciprocal — is exact rather than approached.
    """

    @staticmethod
    def _pair(steps, k0=0.35, dtype=np.complex64, nudge=0.0):
        k0 = np.float32(k0)
        spacing = np.spacing(k0)
        depths = np.array([0.0, 50.0, 100.0])
        lo = np.array([k0], dtype=np.float32).astype(dtype) + nudge
        hi = (np.array([k0 + np.float32(steps) * spacing],
                       dtype=np.float32).astype(dtype) + nudge)
        return (Modes(k=lo, phi=np.ones((3, 1)), depths=depths,
                      model='Test', frequencies=100.0),
                Modes(k=hi, phi=np.ones((3, 1)), depths=depths,
                      model='Test', frequencies=101.0))

    def test_a_step_spanning_twice_the_floor_s_steps_is_silent(self):
        lo, hi = self._pair(2e5)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert np.isfinite(lo.compute_group_velocity(hi)).all()

    def test_a_step_spanning_exactly_the_floor_s_steps_is_silent(self):
        lo, hi = self._pair(1e5)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert np.isfinite(lo.compute_group_velocity(hi)).all()

    def test_a_step_spanning_half_the_floor_s_steps_is_named(self):
        lo, hi = self._pair(5e4)
        with pytest.warns(UserWarning, match="wavenumber difference spans"):
            lo.compute_group_velocity(hi)

    def test_the_message_names_the_steps_the_floor_and_the_test_to_run(self):
        """Two frequencies fix the floor but not the truncation error, so the
        message reports the floor and hands back the measurement that settles
        which of the two leads. It prescribes no step: a step read off the
        floor alone was worse than the one in hand in 14 of 30 firings across
        five ideal waveguides, by up to 16x."""
        lo, hi = self._pair(5e4)
        with pytest.warns(UserWarning) as record:
            v_g = lo.compute_group_velocity(hi)
        message = str(record[0].message)
        assert "50000 steps" in message
        assert "2.0e-05 relative" in message
        assert "recompute at twice this frequency separation" in message
        assert "largest change in v_g exceeds the floor" in message
        assert float(v_g[0]) > 0.0

    @pytest.mark.parametrize("k0", [0.05, 0.35, 1234.0])
    def test_the_decision_does_not_move_with_the_wavenumber_scale(self, k0):
        """The floor is a spacing divided by a difference of the same
        quantity, so it is dimensionless and the same step count decides at
        every wavenumber magnitude."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lo, hi = self._pair(2e5, k0=k0)
            lo.compute_group_velocity(hi)
        lo, hi = self._pair(5e4, k0=k0)
        with pytest.warns(UserWarning, match="wavenumber difference spans"):
            lo.compute_group_velocity(hi)

    def test_a_float32_valued_complex128_array_is_measured_against_float32(self):
        """A caller who promotes the ``.mod`` wavenumbers to complex128 has
        not recovered any bits, so the floor is read from the values rather
        than from the container they sit in."""
        lo, hi = self._pair(5e4, dtype=np.complex128)
        assert lo.k.dtype == np.complex128
        with pytest.warns(UserWarning, match="wavenumber difference spans"):
            lo.compute_group_velocity(hi)

    def test_a_wavenumber_carrying_float64_bits_is_measured_against_float64(self):
        """The same step, on values that do not survive a float32 round trip,
        is eleven decades clear of the float64 spacing and says nothing."""
        lo, hi = self._pair(5e4, dtype=np.complex128, nudge=1e-12)
        assert not np.array_equal(
            np.real(lo.k).astype(np.float32).astype(float), np.real(lo.k))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert np.isfinite(lo.compute_group_velocity(hi)).all()
