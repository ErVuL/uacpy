"""Tests for the geoacoustic material preset catalog
(:mod:`uacpy.core.materials`) and the ``from_preset`` factories on
:class:`BoundaryProperties` / :class:`SedimentLayer`."""

import pytest

import uacpy
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.environment import (
    BoundaryProperties, SedimentLayer, LayeredBottom,
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

    def test_get_material_is_case_insensitive(self):
        assert get_material('Sand')['sound_speed'] == get_material('sand')['sound_speed']
        assert get_material('  GRAVEL  ')['sound_speed'] == 1800.0

    def test_get_material_unknown_lists_options(self):
        with pytest.raises(KeyError, match="Available"):
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
    def test_sand_halfspace(self):
        bp = BoundaryProperties.from_preset('sand')
        assert bp.acoustic_type == 'half-space'
        assert bp.sound_speed == 1650.0
        assert bp.density == 1.9
        assert bp.attenuation == 0.8
        assert bp.shear_speed == 110.0
        assert bp.grain_size_phi == 2.0

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
    def test_simple_stack(self):
        bot = LayeredBottom.from_presets(
            layers=[('clay', 5.0), ('silt', 15.0), ('sand', 30.0)],
            halfspace='limestone',
        )
        assert isinstance(bot, LayeredBottom)
        assert len(bot.layers) == 3
        assert [layer.thickness for layer in bot.layers] == [5.0, 15.0, 30.0]
        assert bot.layers[0].sound_speed == 1500.0  # clay c_p
        assert bot.layers[2].sound_speed == 1650.0  # sand c_p
        assert bot.halfspace.sound_speed == 3000.0  # limestone c_p

    def test_layer_overrides(self):
        bot = LayeredBottom.from_presets(
            layers=[('silt', 10.0, {'attenuation': 1.5})],
            halfspace='sand',
        )
        assert bot.layers[0].attenuation == 1.5
        assert bot.layers[0].sound_speed == 1575.0  # silt baseline kept

    def test_halfspace_overrides(self):
        bot = LayeredBottom.from_presets(
            layers=[('sand', 5.0)],
            halfspace='limestone',
            halfspace_overrides={'attenuation': 0.05},
        )
        assert bot.halfspace.attenuation == 0.05
        assert bot.halfspace.sound_speed == 3000.0

    def test_bad_entry_shape_raises(self):
        with pytest.raises(ConfigurationError, match="(name, thickness)"):
            LayeredBottom.from_presets(
                layers=[('sand',)],
                halfspace='limestone',
            )
