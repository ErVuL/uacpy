"""Tests for the plane-wave bottom-loss helper
:func:`uacpy.core.acoustics.bottom_loss_curve` and the matching plot
helper :func:`uacpy.plot.plot_bottom_loss`."""

import warnings

import numpy as np
import pytest

from uacpy.core.acoustics import (
    bottom_loss_curve, reflection_coeff)
from uacpy.core.constants import DEFAULT_WATER_DENSITY_G_CM3
from uacpy.core.exceptions import ConfigurationError

# The water density every deck writes (1.027 g/cm³), which is also the
# wrapper's default; the closed forms below are evaluated against it.
_RHO_W = DEFAULT_WATER_DENSITY_G_CM3



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
        m = dict(sound_speed=1650.0, density=1.9 * _RHO_W, attenuation=0.8)
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
    the Rayleigh closed form (no binary) against the 1.027 g/cm³ water every
    deck writes. The presets carry Jensen et al. Table 1.3's RATIOS against
    that water, so the anchors are the table's: sand 0.7 dB per bounce at 10°
    grazing and 2.0 dB by 24° (just under its critical angle
    arccos(1500/1650) = 24.6°); clay — c_p equal to the water speed — has no
    critical angle and |R| = (1.5−1)/(1.5+1) = 0.2 from the density ratio
    alone (13.98 dB), 13.9 dB at 10°."""

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
        # 1.00), so the angle dependence drops out and, the preset carrying
        # the table's density ratio 1.5 against the deck water,
        # |R| = (1.5-1)/(1.5+1) = 0.2 -> 13.98 dB — flat at every angle
        # steeper than a few degrees (attenuation perturbs the 3rd digit:
        # 13.88 dB at 10°).
        ang, loss = bottom_loss_curve('clay')
        steep = ang >= 10.0
        R = 10.0 ** (-loss[steep] / 20.0)
        np.testing.assert_allclose(R, 0.2, atol=3e-3)
        assert loss[np.argmin(np.abs(ang - 10.0))] == pytest.approx(13.9,
                                                                    abs=0.1)

    def test_a_textbook_ratio_in_unit_water_is_the_same_seabed(self):
        # The remedy for a rho_w = 1 benchmark: the ratio itself as density,
        # in water of density 1.0, is the same seabed the preset is in the
        # deck water — and an absolute 1.5 read against 1.027 water is not.
        ang, textbook = bottom_loss_curve(
            dict(sound_speed=1500.0, density=1.5, attenuation=0.2),
            water_density=1.0)
        _, preset = bottom_loss_curve('clay')
        np.testing.assert_allclose(textbook, preset, atol=1e-10)
        _, absolute = bottom_loss_curve(
            dict(sound_speed=1500.0, density=1.5, attenuation=0.2))
        R = 10.0 ** (-absolute[ang >= 10.0] / 20.0)
        np.testing.assert_allclose(R, (1.5 - _RHO_W) / (1.5 + _RHO_W),
                                   atol=3e-3)


class TestSlowBottomIntromission:
    """A seabed slower than the water (docs/guide/utilities.md,
    docs/models/bounce.md): no critical angle, but an intromission angle
    where the impedances match and |R| → 0. Soft mud at 1450 m/s and
    1.4 g/cm³ over 1.027 g/cm³ water puts it at grazing 16.6° —
    sin²θ_inc = (m²−n²)/(m²−1) with m = ρ₂/ρ_w (COA eq. 1.60; their
    c₂=1300/ρ₂=1.8 example over ρ_w = 1 gives 22.6°, reproduced by the same
    closed form with ``water_density=1.0``)."""

    MUD = dict(sound_speed=1450.0, density=1.4, attenuation=0.0)

    @staticmethod
    def _closed_form_intromission_grazing(rho_w):
        m2 = (1.4 / rho_w) ** 2
        n2 = (1500.0 / 1450.0) ** 2
        theta_inc = np.degrees(np.arcsin(np.sqrt((m2 - n2) / (m2 - 1.0))))
        return 90.0 - theta_inc

    def test_loss_peaks_at_the_closed_form_intromission_angle(self):
        grazing_intro = self._closed_form_intromission_grazing(_RHO_W)
        assert grazing_intro == pytest.approx(16.61, abs=0.01)  # doc's 16.6°
        g = np.linspace(10.0, 20.0, 2001)
        _, loss = bottom_loss_curve(self.MUD, grazing_angles_deg=g)
        assert g[np.argmax(loss)] == pytest.approx(grazing_intro, abs=0.02)

    def test_textbook_water_density_moves_the_dip_to_the_coa_angle(self):
        # The rho_w = 1 remedy typed back: COA's own convention puts the same
        # mud's intromission at 15.7°, and the wrapper reproduces it.
        grazing_intro = self._closed_form_intromission_grazing(1.0)
        assert grazing_intro == pytest.approx(15.68, abs=0.01)
        g = np.linspace(10.0, 20.0, 2001)
        _, loss = bottom_loss_curve(self.MUD, grazing_angles_deg=g,
                                    water_density=1.0)
        assert g[np.argmax(loss)] == pytest.approx(grazing_intro, abs=0.02)

    def test_reflection_vanishes_at_intromission(self):
        # |R| at the sampled dip is < 1e-3 (the lossless closed form goes to
        # 0 at the exact angle, 16.61°, so the sub-1e-3 bound is the robust
        # pin on a grid that need not land on it).
        g = np.linspace(16.0, 17.5, 2001)
        _, loss = bottom_loss_curve(self.MUD, grazing_angles_deg=g)
        assert 10.0 ** (-loss.max() / 20.0) < 1e-3
        # The curve PEAKS there rather than saturating: both window edges
        # are far below the dip's loss.
        assert loss.max() > loss[0] + 20.0 and loss.max() > loss[-1] + 20.0

    def test_phase_steps_through_180_degrees_across_intromission(self):
        # For the lossless slow bottom R is real and changes sign at the
        # intromission angle (COA Fig. 2.12 shows the same 180° step).
        R_below = reflection_coeff(12.0, sound_speed=1450.0, density=1.4,
                                   water_density=_RHO_W, water_sound_speed=1500.0)
        R_above = reflection_coeff(20.0, sound_speed=1450.0, density=1.4,
                                   water_density=_RHO_W, water_sound_speed=1500.0)
        assert np.real(R_below) * np.real(R_above) < 0.0


class TestReflectionCoeffCrossCheck:
    """``reflection_coeff`` and ``bottom_loss_curve`` speak the same units
    (grazing degrees, g/cm³, dB/λ), so the preset's own numbers passed
    straight through give the same Rayleigh coefficient; the closed form is
    re-derived here in B&L's incidence-from-normal variables."""

    def test_same_curve_from_the_presets_numbers(self):
        from uacpy.core.materials import get_material
        g = np.linspace(1.0, 89.0, 89)
        _, loss = bottom_loss_curve('sand', grazing_angles_deg=g)
        sand = get_material('sand')
        R = reflection_coeff(g, sound_speed=sand['sound_speed'],
                             density=sand['density'],
                             attenuation=sand['attenuation'])
        np.testing.assert_allclose(loss, -20.0 * np.log10(np.abs(R)),
                                   atol=1e-10)

    def test_the_closed_form_in_incidence_variables(self):
        g = np.array([10.0, 30.0, 60.0])
        theta = np.pi / 2.0 - np.deg2rad(g)
        alpha = 0.8 * np.log(10.0) / (40.0 * np.pi)
        n = 1500.0 / 1650.0 * (1 + 1j * alpha)
        m = 1.9 / _RHO_W
        V = np.conj((m * np.cos(theta) - np.sqrt(n ** 2 - np.sin(theta) ** 2))
                    / (m * np.cos(theta) + np.sqrt(n ** 2 - np.sin(theta) ** 2)))
        np.testing.assert_allclose(
            reflection_coeff(g, sound_speed=1650.0, density=1.9,
                             attenuation=0.8), V, rtol=1e-12)


class TestTheTwoEntryPointsShareOneWater:
    """``reflection_coeff`` without ``water_*`` and ``bottom_loss_curve``
    evaluate the same reference water — the nominal 1500 m/s and the
    package's one water density, 1027 kg/m³.

    Before, the direct call fell back to Mackenzie and EOS-80 at 27 °C
    (1539.087 m/s, 1022.72 kg/m³) and warned, and the same seabed's loss
    differed by about 4 dB near the critical angle between the two entry
    points."""

    def test_the_default_call_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            reflection_coeff(45.0, sound_speed=1700.0, density=1.8)

    def test_the_default_water_is_the_package_reference(self):
        R_default = reflection_coeff(45.0, sound_speed=1700.0, density=1.8)
        R_explicit = reflection_coeff(45.0, sound_speed=1700.0, density=1.8,
                                      water_density=1.027,
                                      water_sound_speed=1500.0)
        assert R_default == R_explicit

    def test_the_two_entry_points_give_one_curve(self):
        g, loss = bottom_loss_curve(
            dict(sound_speed=1700.0, density=1.8, attenuation=0.0))
        R = reflection_coeff(g, sound_speed=1700.0, density=1.8)
        np.testing.assert_allclose(
            loss, -20.0 * np.log10(np.abs(R) + 1e-300), atol=1e-9)

    def test_the_docstring_example_reads_its_printed_values(self):
        R_doc = reflection_coeff(45.0, sound_speed=1600.0, density=1.2)
        assert f"{R_doc:.4f}" == '0.1461'
        assert f"{20.0 * np.log10(abs(R_doc)):.2f}" == '-16.71'


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
        R = _fluid_solid_R(self.G, 1500.0, _RHO_W, m['sound_speed'],
                           m['shear_speed'], m['density'],
                           ap_dbl=m['attenuation'],
                           as_dbl=m['shear_attenuation'])
        _, fluid = bottom_loss_curve(name, grazing_angles_deg=self.G)
        return -20.0 * np.log10(np.abs(R)) - fluid

    def test_reference_reduces_to_the_fluid_curve_without_shear(self):
        # The elastic reference with c_s = 0 IS the package's Rayleigh
        # curve — validates the test-local closed form against the code.
        R = _fluid_solid_R(self.G, 1500.0, _RHO_W, 1650.0, 0.0,
                           1.9 * _RHO_W, ap_dbl=0.8)
        _, fluid = bottom_loss_curve('sand', grazing_angles_deg=self.G)
        np.testing.assert_allclose(-20.0 * np.log10(np.abs(R)), fluid,
                                   atol=1e-10)

    @pytest.mark.parametrize('name', ['clay', 'silt', 'sand', 'gravel'])
    def test_soft_sediment_shear_costs_under_a_quarter_dB(self, name):
        assert np.max(np.abs(self._extra_loss(name))) < 0.25

    def test_chalk_and_limestone_lose_an_extra_11_to_16_dB(self):
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


class TestReflectionCoeffSpeaksThePackagesUnits:
    """Grazing degrees in [0, 90], g/cm³, dB/λ, the medium keyword-only. The
    old arlpy call ``reflection_coeff(angle_rad, rho1_kg_m3, c1)`` fails
    loudly — positional medium arguments are a TypeError, and a kg/m³
    density is refused naming the g/cm³ value."""

    def test_the_whole_grazing_range_is_accepted_and_beyond_it_refused(self):
        for g in (0.0, 45.0, 90.0):
            assert np.isfinite(np.abs(reflection_coeff(
                g, sound_speed=1700.0, density=1.8)))
        for g in (-0.1, 90.1):
            with pytest.raises(ConfigurationError, match='grazing_deg'):
                reflection_coeff(g, sound_speed=1700.0, density=1.8)

    def test_the_old_positional_call_is_a_type_error(self):
        with pytest.raises(TypeError, match='takes 1 positional argument but'):
            reflection_coeff(0.5, 1800.0, 1700.0)

    def test_a_kg_per_m3_density_is_refused_naming_g_per_cm3(self):
        with pytest.raises(ConfigurationError, match='g/cm'):
            reflection_coeff(30.0, sound_speed=1700.0, density=1800.0)
        assert np.isfinite(np.abs(reflection_coeff(
            30.0, sound_speed=1700.0, density=20.0)))
        # The threshold itself: just above 20 g/cm³ is a kg/m³ number.
        with pytest.raises(ConfigurationError, match='g/cm'):
            reflection_coeff(30.0, sound_speed=1700.0, density=20.001)
        with pytest.raises(ConfigurationError, match='water_density'):
            reflection_coeff(30.0, sound_speed=1700.0, density=1.8,
                             water_density=20.001)
        assert np.isfinite(np.abs(reflection_coeff(
            30.0, sound_speed=1700.0, density=1.8, water_density=20.0)))

    def test_bottom_loss_curve_names_the_water_speed_water_sound_speed(self):
        with pytest.raises(TypeError,
                           match="unexpected keyword argument 'water_speed'"):
            bottom_loss_curve('sand', water_speed=1490.0)
        g, loss = bottom_loss_curve('sand', water_sound_speed=1490.0)
        assert np.all(np.isfinite(loss))


class TestPlotBottomLoss:
    """The plotter that draws what ``bottom_loss_curve`` computes. Before it
    existed every caller -- example 25 included -- looped and called
    ``ax.plot`` itself."""

    @staticmethod
    def _close(fig):
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_one_curve_per_material_named_after_it(self):
        from uacpy.plot import plot_bottom_loss
        fig, ax = plot_bottom_loss(['sand', 'silt', 'basalt'])
        try:
            assert len(ax.lines) == 3
            assert [l.get_label() for l in ax.lines] == ['sand', 'silt', 'basalt']
            assert 'Grazing angle' in ax.get_xlabel()
            assert 'dB' in ax.get_ylabel()
        finally:
            self._close(fig)

    def test_a_property_dict_is_drawn_beside_the_presets(self):
        from uacpy.plot import plot_bottom_loss
        fetched = dict(sound_speed=1521.0, density=1.52, attenuation=0.11)
        fig, ax = plot_bottom_loss({'sand': 'sand', 'fetched': fetched})
        try:
            assert [l.get_label() for l in ax.lines] == ['sand', 'fetched']
            # the dict curve is a real curve, not an empty one
            assert np.nanmax(ax.lines[1].get_ydata()) > 1.0
        finally:
            self._close(fig)

    def test_a_label_mapping_is_not_read_as_one_property_dict(self):
        """Both are dicts. Only the property keys tell them apart, and
        getting it wrong sends the whole mapping to ``bottom_loss_curve``
        as though it described one seabed."""
        from uacpy.plot import plot_bottom_loss
        fig, ax = plot_bottom_loss({'a': 'sand', 'b': 'silt', 'c': 'basalt'})
        try:
            assert len(ax.lines) == 3
            assert [l.get_label() for l in ax.lines] == ['a', 'b', 'c']
        finally:
            self._close(fig)

    def test_the_curve_is_bottom_loss_curve_verbatim(self):
        from uacpy.plot import plot_bottom_loss
        angles, loss = bottom_loss_curve('sand', water_sound_speed=1490.0)
        fig, ax = plot_bottom_loss('sand', water_sound_speed=1490.0)
        try:
            np.testing.assert_allclose(ax.lines[0].get_xdata(), angles)
            np.testing.assert_allclose(ax.lines[0].get_ydata(), loss)
        finally:
            self._close(fig)

    def test_the_water_speed_is_named_water_sound_speed(self):
        """One name for the water's speed across the package; the retired
        spelling is not quietly forwarded to matplotlib."""
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.plot import plot_bottom_loss
        with pytest.raises(ConfigurationError, match='water_speed'):
            plot_bottom_loss('sand', water_speed=1490.0)

    def test_mark_critical_rules_the_fast_seabeds_only(self):
        """A seabed slower than the water has no critical angle, so it gets
        no rule rather than one at an angle it does not have."""
        from uacpy.plot import plot_bottom_loss
        # clay is 1500 m/s: slower than 1510 water, faster than 1480
        fig, ax = plot_bottom_loss(['clay', 'basalt'], water_sound_speed=1510.0,
                                   mark_critical=True)
        try:
            rules = [l for l in ax.lines if l.get_linestyle() == ':']
            assert len(rules) == 1          # basalt only
            x = rules[0].get_xdata()[0]
            assert x == pytest.approx(
                np.degrees(np.arccos(1510.0 / 5250.0)), abs=1e-6)
        finally:
            self._close(fig)

    def test_the_packages_own_seabed_objects_are_drawn(self):
        """``env.bottom`` and a ``BoundaryProperties`` failed with a raw
        ``TypeError: ... is not iterable``. A carrier, alone or in a list or
        mapping, draws the curve its three properties give; a layered or
        range-dependent Bottom, which has no single half-space reflection, is
        refused naming ``halfspace_at``."""
        import uacpy
        from uacpy.core.boundary import BoundaryProperties
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.plot import plot_bottom_loss
        props = dict(sound_speed=1650.0, density=1.8, attenuation=0.8)
        _, expected = bottom_loss_curve(props)
        bp = BoundaryProperties(acoustic_type='half-space', **props)
        env = uacpy.Environment(bathymetry=100.0, ssp=1500.0,
                                bottom=bp)
        for materials in (bp, [bp], {'site': bp, 'sand': 'sand'},
                          env.bottom):
            fig, ax = plot_bottom_loss(materials, mark_critical=True)
            try:
                np.testing.assert_allclose(ax.lines[0].get_ydata(), expected)
                rule = [l for l in ax.lines if l.get_linestyle() == ':'][0]
                assert rule.get_xdata()[0] == pytest.approx(
                    np.degrees(np.arccos(1500.0 / 1650.0)))
            finally:
                self._close(fig)
        from uacpy.core.bottom import Bottom
        rd = Bottom.from_halfspaces([0.0, 5000.0], sound_speed=[1600.0, 1700.0],
                                    density=[1.8, 1.8], attenuation=[0.8, 0.8])
        assert rd.is_range_dependent
        with pytest.raises(ConfigurationError, match='range-dependent') as rd_err:
            plot_bottom_loss(rd)
        assert 'halfspace_at(range=r)' in str(rd_err.value.remediation)
        # A layered seabed is refused on its own: its half-space alone is a
        # different reflection, so the remedy names the routes that model
        # the layers, not halfspace_at.
        from uacpy.core.boundary import SedimentLayer
        from uacpy.core.bottom import SeabedColumn
        layered = Bottom.from_column(SeabedColumn(
            layers=[SedimentLayer(thickness=15.0, sound_speed=1650.0,
                                  density=1.6, attenuation=0.4)],
            halfspace=bp))
        assert layered.is_layered and not layered.is_range_dependent
        with pytest.raises(ConfigurationError, match='layered Bottom') as err:
            plot_bottom_loss(layered)
        assert 'Bounce' in str(err.value.remediation)
        assert 'halfspace_at' not in str(err.value.remediation)
        with pytest.raises(ConfigurationError, match='cannot draw a int'):
            plot_bottom_loss(3)

    def test_without_mark_critical_there_are_no_rules(self):
        from uacpy.plot import plot_bottom_loss
        fig, ax = plot_bottom_loss(['sand', 'basalt'], water_sound_speed=1500.0)
        try:
            assert not [l for l in ax.lines if l.get_linestyle() == ':']
        finally:
            self._close(fig)

    def test_an_empty_material_list_is_refused(self):
        from uacpy.plot import plot_bottom_loss
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match='no materials'):
            plot_bottom_loss([])


class TestTheCriticalAngleHasAnEntryPointOfItsOwn:
    """`arccos(c_water / c_bottom)` used to be spelled only inside
    `plot_bottom_loss`, which ruled it on the axes. The number could not be
    obtained without drawing the figure — `critical` appeared in no public
    name in `uacpy`, `uacpy.core.acoustics` or `uacpy.sonar`.
    """

    def test_a_faster_seabed_has_one(self):
        import uacpy
        assert uacpy.acoustics.critical_angle(1650.0, 1500.0) == pytest.approx(
            np.degrees(np.arccos(1500.0 / 1650.0)), rel=1e-12)

    def test_a_seabed_no_faster_than_the_water_has_none(self):
        """`clay` is 1500 m/s, under sea water: it reflects weakly at every
        angle and shows an angle of intromission instead. Returning 0 there
        would read as "totally reflecting everywhere", the opposite."""
        import uacpy
        from uacpy.core.materials import get_material
        assert np.isnan(uacpy.acoustics.critical_angle(
            float(get_material('clay')['sound_speed']), 1500.0))
        assert np.isnan(uacpy.acoustics.critical_angle(1400.0, 1500.0))

    def test_a_non_positive_speed_is_refused(self):
        import uacpy
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match='must be > 0'):
            uacpy.acoustics.critical_angle(0.0, 1500.0)

    def test_the_figure_rules_exactly_what_the_function_returns(self):
        """One formula: the plotter asks, it does not re-derive."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import uacpy
        from uacpy.core.materials import get_material
        from uacpy.plot import plot_bottom_loss

        names = ['sand', 'granite', 'clay']
        fig, ax = plot_bottom_loss(names, mark_critical=True,
                                   water_sound_speed=1500.0)
        try:
            ruled = sorted(float(l.get_xdata()[0]) for l in ax.lines
                           if l.get_linestyle() == ':')
            want = sorted(
                a for a in (uacpy.acoustics.critical_angle(
                    float(get_material(n)['sound_speed']), 1500.0)
                    for n in names) if np.isfinite(a))
            assert ruled == pytest.approx(want, rel=1e-12)
            assert len(ruled) == 2, 'clay must not be ruled'
        finally:
            plt.close(fig)


class TestBottomLossCurveReadsTheSeabedCarriers:
    """A user holds a ``BoundaryProperties`` (``env.bottom.halfspace_at``) or
    a ``SedimentLayer``; ``bottom_loss_curve`` reads their sound speed,
    density and attenuation exactly as it reads the same numbers in a dict,
    and refuses an object that carries none of them by name."""

    props = dict(sound_speed=1700.0, density=1.8, attenuation=0.4)

    def test_a_boundary_properties_gives_the_dict_curve(self):
        from uacpy.core.boundary import BoundaryProperties
        hs = BoundaryProperties(acoustic_type='half-space', **self.props)
        np.testing.assert_array_equal(bottom_loss_curve(hs)[1],
                                      bottom_loss_curve(self.props)[1])

    def test_a_sediment_layer_gives_the_dict_curve(self):
        from uacpy.core.boundary import SedimentLayer
        layer = SedimentLayer(thickness=5.0, **self.props)
        np.testing.assert_array_equal(bottom_loss_curve(layer)[1],
                                      bottom_loss_curve(self.props)[1])

    def test_an_object_without_the_properties_is_refused(self):
        with pytest.raises(ConfigurationError, match='bottom_loss_curve'):
            bottom_loss_curve(object())

    def test_the_module_docstring_describes_one_water(self):
        import uacpy.core.acoustics.boundaries as B
        assert '27 °C' not in B.__doc__ and '4 dB' not in B.__doc__


class TestANonHalfSpaceCarrierIsNotReadAsOne:
    """A rigid, vacuum, file or precalc boundary carries placeholder
    sound_speed / density / attenuation (1600 m/s, 1.5, 0.5); read as a
    half-space they drew a 12-13 dB loss curve for a boundary that
    reflects totally."""

    @pytest.mark.parametrize('kind', ['rigid', 'vacuum'])
    def test_a_totally_reflecting_boundary_loses_nothing(self, kind):
        from uacpy.core.boundary import BoundaryProperties
        angles, loss = bottom_loss_curve(
            BoundaryProperties(acoustic_type=kind),
            grazing_angles_deg=[10.0, 45.0, 90.0])
        np.testing.assert_array_equal(angles, [10.0, 45.0, 90.0])
        np.testing.assert_array_equal(loss, [0.0, 0.0, 0.0])

    def test_the_environment_route_of_the_docstring_agrees(self):
        import uacpy
        from uacpy.core.boundary import BoundaryProperties
        env = uacpy.Environment(
            bathymetry=100.0, bottom=BoundaryProperties(acoustic_type='rigid'))
        _, loss = bottom_loss_curve(env.bottom.halfspace_at(range=0.0))
        assert np.all(loss == 0.0) and loss.size == 181

    def test_a_reflection_table_boundary_is_refused(self):
        from uacpy.core.boundary import BoundaryProperties
        with pytest.raises(ConfigurationError, match='reflection-coefficient'):
            bottom_loss_curve(BoundaryProperties(acoustic_type='file'))

    def test_a_half_space_carrier_reads_its_own_numbers(self):
        from uacpy.core.boundary import BoundaryProperties
        hs = BoundaryProperties(acoustic_type='half-space', sound_speed=1650.0,
                                density=1.9, attenuation=0.8)
        np.testing.assert_array_equal(
            bottom_loss_curve(hs)[1],
            bottom_loss_curve({'sound_speed': 1650.0, 'density': 1.9,
                               'attenuation': 0.8})[1])


def test_the_quoted_water_density_difference_on_sand_is_the_computed_one():
    """``bottom_loss_curve``'s docstring quotes the largest gap between the
    1.027 g/cm³ default water and the textbook 1.0 on 'sand'. Recomputed
    here so the figure cannot drift from the curve: Z ratio 3219.6/1540.5
    against 3219.6/1500 at normal incidence is 0.28 dB."""
    from uacpy.core.acoustics.boundaries import bottom_loss_curve
    angles, default = bottom_loss_curve('sand')
    _, textbook = bottom_loss_curve('sand', water_density=1.0)
    gap = np.abs(np.asarray(default) - np.asarray(textbook))
    assert angles[int(np.argmax(gap))] == pytest.approx(90.0)
    assert f"at most {gap.max():.2f} dB, at normal incidence" in \
        bottom_loss_curve.__doc__
