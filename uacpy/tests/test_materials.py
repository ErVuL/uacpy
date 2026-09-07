"""Tests for the geoacoustic material preset catalog
(:mod:`uacpy.core.materials`) and the ``from_preset`` factories on
:class:`BoundaryProperties` / :class:`SedimentLayer`."""

import pytest

import uacpy
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.environment import (
    BoundaryProperties, SedimentLayer, SeabedColumn,
)
from uacpy.core.materials import MATERIALS, list_materials, get_material


class TestMaterialsCatalog:
    def test_canonical_values_pinned(self):
        # Spot-check a few entries against the canonical class-typical values.
        sand = get_material('sand')
        assert sand['sound_speed'] == 1650.0
        assert sand['density'] == 1.9
        assert sand['attenuation'] == 0.8
        assert sand['shear_speed'] == 110.0
        assert sand['porosity'] == 45.0

        clay = get_material('clay')
        assert clay['sound_speed'] == 1500.0
        assert clay['attenuation'] == 0.2

        basalt = get_material('basalt')
        assert basalt['sound_speed'] == 5250.0
        assert basalt['shear_speed'] == 2500.0

    # The full nine-row catalog (docs/guide/environment.md table plus the
    # porosity/roughness columns), keyed (c_p, ρ, α_p, c_s, α_s, porosity,
    # ϕ, roughness). Rows clay..basalt are JKPS *Computational Ocean
    # Acoustics* Table 1.3 (c_p = ratio × 1500 m/s; clay c_s "<100" and the
    # depth-dependent silt/sand/gravel c_s(z̄) take the catalog's 1 m
    # values); granite has no Table 1.3 row (see core/materials.py). ϕ is
    # Hamilton & Bachman (1982) per the module comment. Roughness is 0.0
    # by construction unless overridden.
    _FULL_TABLE = {
        'clay':      (1500.0, 1.5, 0.2, 80.0, 1.0, 70.0, 8.80, 0.0),
        'silt':      (1575.0, 1.7, 1.0, 80.0, 1.5, 55.0, 5.40, 0.0),
        'sand':      (1650.0, 1.9, 0.8, 110.0, 2.5, 45.0, 3.34, 0.0),
        'gravel':    (1800.0, 2.0, 0.6, 180.0, 1.5, 35.0, -1.5, 0.0),
        'moraine':   (1950.0, 2.1, 0.4, 600.0, 1.0, 25.0, None, 0.0),
        'chalk':     (2400.0, 2.2, 0.2, 1000.0, 0.5, None, None, 0.0),
        'limestone': (3000.0, 2.4, 0.1, 1500.0, 0.2, None, None, 0.0),
        'basalt':    (5250.0, 2.7, 0.1, 2500.0, 0.2, None, None, 0.0),
        'granite':   (5500.0, 2.7, 0.1, 3000.0, 0.2, None, None, 0.0),
    }

    @pytest.mark.parametrize('name', sorted(_FULL_TABLE))
    def test_full_catalog_row_pinned(self, name):
        cp, rho, ap, cs, a_s, poro, phi, rough = self._FULL_TABLE[name]
        m = get_material(name)
        assert m['sound_speed'] == cp
        assert m['density'] == rho
        assert m['attenuation'] == ap
        assert m['shear_speed'] == cs
        assert m['shear_attenuation'] == a_s
        assert m['porosity'] == poro
        assert m['grain_size_phi'] == phi
        assert m['roughness'] == rough

    def test_catalog_holds_exactly_the_nine_documented_rows(self):
        assert set(MATERIALS) == set(self._FULL_TABLE)

    def test_get_material_is_case_insensitive(self):
        assert get_material('Sand')['sound_speed'] == get_material('sand')['sound_speed']
        assert get_material('  GRAVEL  ')['sound_speed'] == 1800.0

    def test_get_material_unknown_lists_options(self):
        with pytest.raises(ConfigurationError, match="Available"):
            get_material('not_a_real_material')

    def test_list_materials_sorted(self):
        names = list_materials()
        assert names == sorted(names)
        assert 'sand' in names and 'granite' in names

    def test_every_preset_has_required_keys(self):
        required = {
            'sound_speed', 'density', 'attenuation',
            'shear_speed', 'shear_attenuation',
            'porosity', 'grain_size_phi', 'roughness',
        }
        for name, m in MATERIALS.items():
            assert set(m).issuperset(required), f"{name} missing keys"
            assert m['sound_speed'] > 0
            assert m['density'] > 0
            assert m['attenuation'] >= 0


class TestBoundaryPropertiesFromPreset:
    def test_sand_halfspace_fluid_by_default(self):
        bp = BoundaryProperties.from_preset('sand')
        assert bp.acoustic_type == 'half-space'
        assert bp.sound_speed == 1650.0
        assert bp.density == 1.9
        assert bp.attenuation == 0.8
        assert bp.shear_speed == 0.0
        assert bp.shear_attenuation == 0.0
        assert bp.grain_size_phi == 3.34

    def test_elastic_keeps_shear(self):
        bp = BoundaryProperties.from_preset('sand', elastic=True)
        assert bp.shear_speed == 110.0
        assert bp.shear_attenuation == 2.5

    def test_shear_override_wins_over_fluid_default(self):
        bp = BoundaryProperties.from_preset('sand', shear_speed=300.0)
        assert bp.shear_speed == 300.0

    def test_overrides_apply_last(self):
        bp = BoundaryProperties.from_preset('sand', sound_speed=1700.0, roughness=0.05)
        assert bp.sound_speed == 1700.0
        assert bp.roughness == 0.05
        assert bp.density == 1.9


class TestSedimentLayerFromPreset:
    def test_thickness_required(self):
        layer = SedimentLayer.from_preset('silt', thickness=15.0)
        assert layer.thickness == 15.0
        assert layer.sound_speed == 1575.0
        assert layer.density == 1.7

    def test_thickness_kwarg_only(self):
        with pytest.raises(TypeError):
            SedimentLayer.from_preset('silt')

    def test_overrides(self):
        layer = SedimentLayer.from_preset(
            'sand', thickness=5.0, attenuation=0.5,
        )
        assert layer.attenuation == 0.5
        assert layer.sound_speed == 1650.0


class TestPublicReexports:
    def test_top_level(self):
        assert uacpy.materials is uacpy.core.materials
        assert 'sand' in uacpy.materials.list_materials()


class TestLayeredBottomFromPresets:
    def test_from_presets_stacks_layers_in_order_with_catalog_values(self):
        bot = SeabedColumn.from_presets(
            layers=[('clay', 5.0), ('silt', 15.0), ('sand', 30.0)],
            halfspace='limestone',
        )
        assert isinstance(bot, SeabedColumn)
        assert len(bot.layers) == 3
        assert [layer.thickness for layer in bot.layers] == [5.0, 15.0, 30.0]
        assert bot.layers[0].sound_speed == 1500.0  # clay c_p
        assert bot.layers[2].sound_speed == 1650.0  # sand c_p
        assert bot.halfspace.sound_speed == 3000.0  # limestone c_p

    def test_layer_overrides(self):
        bot = SeabedColumn.from_presets(
            layers=[('silt', 10.0, {'attenuation': 1.5})],
            halfspace='sand',
        )
        assert bot.layers[0].attenuation == 1.5
        assert bot.layers[0].sound_speed == 1575.0  # silt baseline kept

    def test_halfspace_overrides(self):
        bot = SeabedColumn.from_presets(
            layers=[('sand', 5.0)],
            halfspace='limestone',
            halfspace_overrides={'attenuation': 0.05},
        )
        assert bot.halfspace.attenuation == 0.05
        assert bot.halfspace.sound_speed == 3000.0

    def test_bad_entry_shape_raises(self):
        with pytest.raises(ConfigurationError, match="(name, thickness)"):
            SeabedColumn.from_presets(
                layers=[('sand',)],
                halfspace='limestone',
            )


class TestHamiltonAttenuationFollowsThe1972GrainSizeRegressions:
    """``k_p`` is the Fig. 3 caption of Hamilton (1972), Geophysics 37, p. 636."""

    @pytest.mark.parametrize('phi, expected', [
        (0.92, 0.4556 + 0.0245 * 0.92),
        (4.24, 0.1978 + 0.1245 * 4.24),
        (5.40, 8.0399 - 2.5228 * 5.40 + 0.20098 * 5.40 ** 2),
        (8.80, 0.9431 - 0.2041 * 8.80 + 0.0117 * 8.80 ** 2),
    ])
    def test_each_branch_reproduces_the_printed_regression(self, phi, expected):
        from uacpy.core.sediment import _hamilton_kp
        assert _hamilton_kp(phi) == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize('join', [2.6, 4.5, 6.0])
    def test_the_branches_meet_at_their_joins(self, join):
        from uacpy.core.sediment import _hamilton_kp
        assert abs(_hamilton_kp(join - 1e-9) - _hamilton_kp(join + 1e-9)) < 0.005

    def test_grain_sizes_outside_the_recommended_limits_hold_the_end_values(self):
        from uacpy.core.sediment import _hamilton_kp
        assert _hamilton_kp(-1.0) == _hamilton_kp(0.0)
        assert _hamilton_kp(12.0) == _hamilton_kp(9.5)

    def test_the_peak_sits_in_very_fine_sand(self):
        import numpy as np
        from uacpy.core.sediment import _hamilton_kp
        phi = np.linspace(0.0, 9.5, 951)
        k = np.array([_hamilton_kp(p) for p in phi])
        assert phi[int(np.argmax(k))] == pytest.approx(4.5, abs=0.02)
        assert k.max() == pytest.approx(0.758, abs=0.002)

    def test_attenuation_in_db_per_wavelength_is_k_p_times_c_over_1000(self):
        from uacpy.core.sediment import _hamilton_kp, grain_size_to_geoacoustics
        props = grain_size_to_geoacoustics(5.4, model='hamilton')
        assert props['attenuation'] == pytest.approx(_hamilton_kp(5.4) * props['sound_speed'] / 1000.0, rel=1e-9)
