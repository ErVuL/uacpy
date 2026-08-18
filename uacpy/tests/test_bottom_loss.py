"""Tests for the plane-wave bottom-loss helper
:func:`uacpy.core.acoustics.bottom_loss_curve` and the matching plot
helper."""

import numpy as np
import pytest

from uacpy.core.acoustics import bottom_loss_curve, reflection_coeff


class TestBottomLossCurve:
    def test_returns_matched_arrays(self):
        ang, loss = bottom_loss_curve('sand')
        assert ang.shape == loss.shape
        assert ang[0] == 0.0
        assert ang[-1] == 90.0

    def test_grazing_perfect_reflection(self):
        # At grazing=0° all energy reflects → zero loss; at normal
        # incidence some transmits → finite loss for a fluid bottom
        # denser & faster than water.
        _, loss = bottom_loss_curve('sand')
        assert loss[0] == pytest.approx(0.0, abs=1e-6)
        assert loss[-1] > 0.5

    def test_subcritical_loss_smaller_than_supercritical(self):
        # Below the critical angle the ray bends fully back into the water →
        # loss is small; above it (up to normal incidence) some energy
        # transmits → larger loss. Angles here are grazing (0 = along the
        # interface), so for fluid 'sand' c_2=1650 over c_1=1500 the critical
        # grazing angle is arccos(c1/c2) ≈ 24.6°; the 20°/30° windows below
        # straddle it without touching it.
        ang, loss = bottom_loss_curve('sand')
        below = loss[ang < 20.0].mean()
        above = loss[(ang > 30.0) & (ang < 60.0)].mean()
        assert below < above

    def test_dict_material_produces_the_same_curve_as_equivalent_preset(self):
        # A dict carrying the 'sand' preset's three fluid properties must
        # take exactly the code path the preset name takes — identical
        # arrays, not merely finite ones.
        m = dict(sound_speed=1650.0, density=1.9, attenuation=0.8)
        ang_d, loss_d = bottom_loss_curve(m)
        ang_p, loss_p = bottom_loss_curve('sand')
        np.testing.assert_array_equal(ang_d, ang_p)
        np.testing.assert_array_equal(loss_d, loss_p)
        # A dict with different values still yields a finite curve.
        other = dict(sound_speed=1700.0, density=1.8, attenuation=0.3)
        _, loss_o = bottom_loss_curve(other)
        assert np.all(np.isfinite(loss_o))
        assert not np.allclose(loss_o, loss_p)

    def test_custom_angle_grid(self):
        custom = np.linspace(5.0, 85.0, 41)
        ang, loss = bottom_loss_curve('limestone', grazing_angles_deg=custom)
        assert np.array_equal(ang, custom)


class TestPresetBottomLossAnchors:
    """docs/models/bounce.md §"Reading the catalogue" anchors, reproduced by
    the Rayleigh closed form (no binary): sand 0.7 dB per bounce at 10°
    grazing and 2.0 dB by 24° (just under its critical angle
    arccos(1500/1650) = 24.6°); clay — c_p equal to the water speed — has no
    critical angle and |R| = (ρ₂−ρ₁)/(ρ₂+ρ₁) = 0.2 from the density
    contrast alone (13.98 dB), 13.9 dB at 10°."""

    def test_sand_per_bounce_losses_at_10_and_24_degrees(self):
        _, loss = bottom_loss_curve('sand',
                                    grazing_angles_deg=np.array([10.0, 24.0]))
        # Computed 0.719 / 2.042 dB; the doc rounds to 0.7 / 2.0.
        assert loss[0] == pytest.approx(0.72, abs=0.02)
        assert loss[1] == pytest.approx(2.04, abs=0.02)

    def test_sand_critical_grazing_angle_is_arccos_c1_over_c2(self):
        crit = np.degrees(np.arccos(1500.0 / 1650.0))
        assert crit == pytest.approx(24.62, abs=0.01)
        # The loss curve jumps across it: mean loss in the 5° above the
        # critical angle (3.90 dB measured) is 2.5x the mean in the 5°
        # below (1.54 dB) — sand's absorption keeps the sub-critical floor
        # well above zero, so the factual ratio is ~2.5, not "several".
        g = np.linspace(crit - 5.0, crit + 5.0, 101)
        _, loss = bottom_loss_curve('sand', grazing_angles_deg=g)
        assert loss[g > crit].mean() > 2.0 * loss[g < crit].mean()

    def test_clay_reflects_the_bare_density_contrast(self):
        # clay c_p = 1500 m/s = water speed exactly (COA Table 1.3 ratio
        # 1.00), so the angle dependence drops out and
        # |R| = (1.5-1.0)/(1.5+1.0) = 0.2 -> 13.98 dB — flat at every angle
        # steeper than a few degrees (attenuation perturbs the 4th digit).
        ang, loss = bottom_loss_curve('clay')
        steep = ang >= 10.0
        R = 10.0 ** (-loss[steep] / 20.0)
        np.testing.assert_allclose(R, 0.2, atol=3e-3)
        assert loss[np.argmin(np.abs(ang - 10.0))] == pytest.approx(13.9,
                                                                    abs=0.1)


class TestSlowBottomIntromission:
    """A seabed slower than the water (docs/guide/utilities.md,
    docs/models/bounce.md): no critical angle, but an intromission angle
    where the impedances match and |R| → 0. Soft mud at 1450 m/s and
    1.4 g/cm³ puts it at grazing 15.7° — sin²θ_inc = (m²−n²)/(m²−1)
    (COA eq. 1.60; their c₂=1300/ρ₂=1.8 example gives 22.6°, reproduced by
    the same closed form)."""

    MUD = dict(sound_speed=1450.0, density=1.4, attenuation=0.0)

    def test_loss_peaks_at_the_closed_form_intromission_angle(self):
        m2 = 1.4 ** 2
        n2 = (1500.0 / 1450.0) ** 2
        theta_inc = np.degrees(np.arcsin(np.sqrt((m2 - n2) / (m2 - 1.0))))
        grazing_intro = 90.0 - theta_inc
        assert grazing_intro == pytest.approx(15.68, abs=0.01)  # doc's 15.7°
        g = np.linspace(10.0, 20.0, 2001)
        _, loss = bottom_loss_curve(self.MUD, grazing_angles_deg=g)
        assert g[np.argmax(loss)] == pytest.approx(grazing_intro, abs=0.02)

    def test_reflection_vanishes_at_intromission(self):
        # |R| at the sampled dip is < 1e-3 (the doc quotes ≈ 0.0007; the
        # lossless closed form goes to 0 at the exact angle, and 2.5e-4 at
        # the rounded 15.7°, so the sub-1e-3 bound is the robust pin).
        g = np.linspace(15.0, 16.5, 2001)
        _, loss = bottom_loss_curve(self.MUD, grazing_angles_deg=g)
        assert 10.0 ** (-loss.max() / 20.0) < 1e-3
        # The curve PEAKS there rather than saturating: both window edges
        # are far below the dip's loss.
        assert loss.max() > loss[0] + 20.0 and loss.max() > loss[-1] + 20.0

    def test_phase_steps_through_180_degrees_across_intromission(self):
        # For the lossless slow bottom R is real and changes sign at the
        # intromission angle (COA Fig. 2.12 shows the same 180° step).
        R_below = reflection_coeff(np.deg2rad(90.0 - 12.0), rho1=1400.0,
                                   c1=1450.0, rho=1000.0, c=1500.0)
        R_above = reflection_coeff(np.deg2rad(90.0 - 20.0), rho1=1400.0,
                                   c1=1450.0, rho=1000.0, c=1500.0)
        assert np.real(R_below) * np.real(R_above) < 0.0


class TestReflectionCoeffCrossCheck:
    """DOCUMENTATION.md §15: ``reflection_coeff`` is the SI/radians helper
    (ρ in kg/m³, incidence angle from the normal) while
    ``bottom_loss_curve`` sits on the acoustic-input side (g/cm³, grazing
    degrees, dB/λ attenuation). With the units converted explicitly the two
    must be the same Rayleigh coefficient."""

    def test_same_curve_after_explicit_unit_conversion(self):
        g = np.linspace(1.0, 89.0, 89)
        _, loss = bottom_loss_curve('sand', grazing_angles_deg=g)
        # grazing deg -> incidence rad; g/cm³ -> kg/m³; dB/λ -> loss tangent.
        alpha = 0.8 * np.log(10.0) / (40.0 * np.pi)
        R = reflection_coeff(np.pi / 2.0 - np.deg2rad(g),
                             rho1=1900.0, c1=1650.0, alpha=alpha,
                             rho=1000.0, c=1500.0)
        np.testing.assert_allclose(loss, -20.0 * np.log10(np.abs(R)),
                                   atol=1e-10)


def _fluid_solid_R(graz_deg, cp1, rho1, cp2, cs2, rho2,
                   ap_dbl=0.0, as_dbl=0.0):
    """Plane-wave reflection off a solid half-space (COA §1.6, eq. 1.61):
    R = (Z_tot − Z₁)/(Z_tot + Z₁) with the effective solid impedance
    Z_tot = Z_p·cos²(2θ_s) + Z_s·sin²(2θ_s), Z_i = ρ·c_i/sin θ_i on the
    grazing angles Snell couples to θ₁. Lossy media enter as complex
    speeds c/(1 + i·α·ln10/(40π)). Used as the independent elastic
    reference for the fluid-only ``bottom_loss_curve``; with c_s = 0 it
    reduces to the Rayleigh fluid–fluid coefficient exactly (asserted
    below)."""
    th1 = np.deg2rad(np.asarray(graz_deg, dtype=float))
    conv = np.log(10.0) / (40.0 * np.pi)
    cp2c = cp2 / (1.0 + 1j * ap_dbl * conv)
    cos1 = np.cos(th1)
    sin_p = np.lib.scimath.sqrt(1.0 - (cos1 * cp2c / cp1) ** 2)
    Zp = rho2 * cp2c / sin_p
    Z1 = rho1 * cp1 / np.sin(th1)
    if cs2 > 0.0:
        cs2c = cs2 / (1.0 + 1j * as_dbl * conv)
        sin_s = np.lib.scimath.sqrt(1.0 - (cos1 * cs2c / cp1) ** 2)
        cos_s = cos1 * cs2c / cp1
        Zs = rho2 * cs2c / sin_s
        s2 = 2.0 * sin_s * cos_s
        c2 = 1.0 - 2.0 * sin_s ** 2
        Ztot = Zp * c2 ** 2 + Zs * s2 ** 2
    else:
        Ztot = Zp
    return (Ztot - Z1) / (Ztot + Z1)


class TestShearLossMagnitude:
    """docs/guide/environment.md §"What shear does": treating an elastic
    seabed as fluid under-predicts bottom loss. Quantified against the
    fluid–solid closed form above: clay–gravel (c_s 80–180 m/s) cost under
    0.25 dB; chalk and limestone lose an extra 11–16 dB near 20–30°
    grazing; basalt/granite (c_s > 1500 m/s) lose nothing below their shear
    critical angle arccos(1500/c_s)."""

    G = np.linspace(1.0, 89.0, 89)

    def _extra_loss(self, name):
        from uacpy.core.materials import get_material
        m = get_material(name)
        R = _fluid_solid_R(self.G, 1500.0, 1.0, m['sound_speed'],
                           m['shear_speed'], m['density'],
                           ap_dbl=m['attenuation'],
                           as_dbl=m['shear_attenuation'])
        _, fluid = bottom_loss_curve(name, grazing_angles_deg=self.G)
        return -20.0 * np.log10(np.abs(R)) - fluid

    def test_reference_reduces_to_the_fluid_curve_without_shear(self):
        # The elastic reference with c_s = 0 IS the package's Rayleigh
        # curve — validates the test-local closed form against the code.
        R = _fluid_solid_R(self.G, 1500.0, 1.0, 1650.0, 0.0, 1.9,
                           ap_dbl=0.8)
        _, fluid = bottom_loss_curve('sand', grazing_angles_deg=self.G)
        np.testing.assert_allclose(-20.0 * np.log10(np.abs(R)), fluid,
                                   atol=1e-10)

    @pytest.mark.parametrize('name', ['clay', 'silt', 'sand', 'gravel'])
    def test_soft_sediment_shear_costs_under_a_quarter_db(self, name):
        assert np.max(np.abs(self._extra_loss(name))) < 0.25

    def test_chalk_and_limestone_lose_an_extra_11_to_16_db(self):
        band = (self.G >= 20.0) & (self.G <= 30.0)
        # Computed peaks: chalk 16.2 dB, limestone 11.4 dB in the band —
        # the doc's "extra 11–16 dB near 20–30°". abs=0.5 covers the 1°
        # grid.
        assert np.max(self._extra_loss('chalk')[band]) == pytest.approx(
            16.2, abs=0.5)
        assert np.max(self._extra_loss('limestone')[band]) == pytest.approx(
            11.4, abs=0.5)

    @pytest.mark.parametrize('name,cs', [('basalt', 2500.0),
                                         ('granite', 3000.0)])
    def test_fast_rock_shear_is_evanescent_below_its_critical_angle(self,
                                                                    name, cs):
        # Below arccos(1500/c_s) the shear wave is evanescent and the rock
        # loses (nearly) nothing to conversion; computed < 0.1 dB through
        # the 20–30° band against critical angles of 53.1° / 60°.
        band = (self.G >= 20.0) & (self.G <= 30.0)
        assert np.degrees(np.arccos(1500.0 / cs)) > 50.0
        assert np.max(np.abs(self._extra_loss(name)[band])) < 0.1
