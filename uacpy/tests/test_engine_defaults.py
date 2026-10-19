"""The engine knob defaults: each constructor states its default once, as
the named constant of ``uacpy.core.engine_defaults`` or as ``None`` where the
model derives the value (ruling 26), and every writer defaults to ``None``
and resolves it to the constant, the value the engine's deck takes."""
import inspect

import pytest

import uacpy
import uacpy.io as io

#: (engine, knob, value) of each constructor default.
DEFAULTS = [
    ('Bellhop', 'beam_type', 'G'), ('Bellhop', 'n_beams', 0),
    ('Bellhop', 'launch_angles_deg', (-80, 80)), ('Bellhop', 'ray_step', None),
    ('Bellhop', 'grid_type', 'R'), ('Bellhop', 'interp_bathymetry', 'linear'),
    ('Bellhop', 'interp_altimetry', 'linear'), ('Bellhop', 'beam_shift', False),
    ('Kraken', 'n_mesh', None), ('Kraken', 'mode_coupling', None),
    ('Scooter', 'n_mesh', None), ('SPARC', 'n_mesh', None),
    ('SPARC', 'output_mode', 'R'), ('OAST', 'integration_offset', 0.0),
    ('OAST', 'vrec', 0.0), ('OASN', 'integration_offset', 0.0),
    ('OASP', 'integration_offset', 0.0), ('OASP', 'freq_min', None),
    ('OASR', 'angle_type', 'grazing'), ('OASSP', 'spectral_exponent', None),
    ('OASS', 'spectral_exponent', None),
    ('OASSP', 'realization', 0), ('OASSP', 'integration_offset', 0.0),
    ('Bellhop', 'beam_width_type', None), ('Bellhop', 'beam_curvature', None),
    ('Bellhop', 'eps_multiplier', None), ('Bellhop', 'r_loop', None),
    ('Bellhop', 'n_image', None), ('Bellhop', 'ib_win', None),
]


@pytest.mark.parametrize('engine, knob, value', DEFAULTS)
def test_the_constructor_default_keeps_its_value(engine, knob, value):
    default = inspect.signature(getattr(uacpy, engine)).parameters[knob].default
    assert default == value and type(default) is type(value)


@pytest.mark.parametrize('writer, engine', [
    ('write_bellhop_env_file', 'BELLHOP'), ('write_kraken_env_file', 'KRAKEN'),
    ('write_scooter_env_file', 'SCOOTER'), ('write_sparc_env_file', 'SPARC'),
    ('write_oast_input', 'OAST'), ('write_oasn_input', 'OASN'),
    ('write_oasp_input', 'OASP'), ('write_oasr_input', 'OASR'),
    ('write_oassp_input', 'OASSP')])
def test_a_writer_knob_defaults_to_none(writer, engine):
    from uacpy.core import engine_defaults
    knobs = {name.split('_', 1)[1].lower() for name in engine_defaults.__all__
             if name.split('_', 1)[0] == engine}
    params = inspect.signature(getattr(io, writer)).parameters
    assert knobs and knobs <= set(params)
    assert all(params[k].default is None for k in knobs)


def _halfspace(cp, cs=0.0, **kw):
    from uacpy.core import BoundaryProperties
    return BoundaryProperties(acoustic_type='half-space', sound_speed=cp,
                              shear_speed=cs, density=kw.pop('rho', 1.8),
                              attenuation=0.5, **kw)


def _over(bottom, surface=None):
    """1490-1500 m/s water over ``bottom``, under ``surface``."""
    return uacpy.Environment(name='c_low', bathymetry=100.0,
                             ssp=[(0.0, 1500.0), (100.0, 1490.0)],
                             bottom=bottom, surface=surface)


def _sand_over(halfspace):
    from uacpy.core import SedimentLayer
    return uacpy.SeabedColumn(layers=[SedimentLayer(
        8.0, sound_speed=1650.0, density=1.9, shear_speed=110.0)],
        halfspace=halfspace)


def _parameter_free(kind):
    from uacpy.core import BoundaryProperties
    return BoundaryProperties(acoustic_type=kind)


def _cases():
    """``(label, env, c_low, origin fragment)``: where the slowest medium
    sets the edge, the fragment names it; where a Scholte wave sits
    under it, the edge is 0.90 × that wave (its speed pinned to 0.1 m/s,
    the root ``test_acoustics_check_values.py::
    TestTheScholteWaveSpeedIsBrekhovskikhsRoot`` checks against
    eq. 3.4.29)."""
    from uacpy.core import Bottom, SeabedColumn
    rule = ' — the slowest speed of every medium'
    wave = 'the Scholte wave between '
    return [
        ('vacuum bottom', _over(_parameter_free('vacuum')), 0.90 * 1490.0,
         '0.9 × 1490 m/s, the water' + rule),
        ('rigid bottom, no medium', _over(_parameter_free('rigid')),
         0.90 * 1490.0, '0.9 × 1490 m/s, the water' + rule),
        ('fluid half-space slower than the water', _over(_halfspace(1450.0)),
         0.90 * 1450.0, '0.9 × 1450 m/s, the bottom half-space' + rule),
        ('elastic half-space', _over(_halfspace(1800.0, 600.0)),
         0.90 * 523.162, '0.9 × 523.2 m/s, ' + wave
         + 'the water and the bottom half-space'),
        ('elastic layer on a fluid half-space',
         _over(_sand_over(_halfspace(1800.0))), 0.90 * 92.926,
         '0.9 × 92.9 m/s, ' + wave + 'a sediment layer and the bottom '
         'half-space'),
        ('elastic layer on an elastic half-space',
         _over(_sand_over(_halfspace(5250.0, 3000.0, rho=2.65))),
         0.90 * 98.378, '0.9 × 98.4 m/s, ' + wave
         + 'the water and a sediment layer'),
        ('elastic top half-space over fluid water',
         _over(_halfspace(1800.0), surface=_halfspace(3500.0, 1800.0,
                                                       rho=0.9)),
         0.90 * 1217.225, '0.9 × 1217.2 m/s, ' + wave
         + 'the top half-space and the water'),
        ('a shear speed above the water\'s (granite)',
         _over(_halfspace(5750.0, 3000.0, rho=2.65)), 0.90 * 1485.110,
         '0.9 × 1485.1 m/s, ' + wave + 'the water and the bottom '
         'half-space'),
        ('the slowest column of a range-dependent seabed',
         _over(Bottom.from_columns(
             [SeabedColumn(layers=[], halfspace=_halfspace(1800.0)),
              SeabedColumn(layers=[], halfspace=_halfspace(1460.0))],
             ranges=[0.0, 5000.0])), 0.90 * 1460.0,
         '0.9 × 1460 m/s, the bottom half-space' + rule),
    ]


@pytest.mark.parametrize('case', range(9))
def test_the_derived_c_low_is_the_slowest_speed_or_scholte_wave(case):
    """0.90 (``scooter.f90:258``) × the slowest wave of the problem: the
    water, each sediment layer and each acoustic half-space (compressional
    and shear speeds), and every Scholte wave a liquid carries against a
    solid (Brekhovskikh & Lysanov eq. 3.4.29), which is the slowest in
    every elastic case here — over granite too, where the shear speed is
    above the water's and the wave sits just under the water speed (not
    scooter.f90:256's 0.85 × the water); a rigid or vacuum boundary
    carries no medium. The origin names the medium or the interface that
    set it."""
    from uacpy.core.engine_defaults import default_c_low
    label, env, expected, origin_start = _cases()[case]
    c_low, origin = default_c_low(env)
    assert c_low == pytest.approx(expected, rel=1e-5), label
    assert origin.startswith(origin_start), origin


def test_a_shear_speed_near_the_waters_puts_the_wave_under_085_cs():
    """The limestone preset (cp 3000, cs 1500 m/s) under 1500 m/s water:
    the Scholte wave is 1240.4 m/s, 0.827 × the shear speed and slower
    than the water, so the derived edge is 0.90 × 1240.4 m/s, under
    scooter.f90:256's 0.85 × 1500 = 1275 m/s, and the wave is inside the
    window."""
    from uacpy.core import BoundaryProperties
    from uacpy.core.engine_defaults import default_c_low, scholte_waves
    env = uacpy.Environment(
        name='limestone', bathymetry=30.0, ssp=1500.0,
        bottom=BoundaryProperties.from_preset('limestone', elastic=True))
    (wave,) = scholte_waves(env)
    assert wave.speed == pytest.approx(1240.44, abs=0.01)
    assert (wave.depth, wave.where) == (
        30.0, 'between the water and the bottom half-space')
    c_low, origin = default_c_low(env)
    assert c_low == pytest.approx(0.90 * 1240.44, abs=0.01)
    assert c_low < 0.85 * 1500.0
    assert origin.startswith('0.9 × 1240.4 m/s, the Scholte wave between '
                             'the water and the bottom half-space')


def test_two_elastic_media_in_contact_carry_no_liquid_solid_wave():
    """Eq. 3.4.29 is the liquid-solid interface: sand on granite gives the
    water-sand wave only, and a fluid seabed none."""
    from uacpy.core.engine_defaults import scholte_waves
    waves = scholte_waves(_over(_sand_over(_halfspace(5250.0, 3000.0,
                                                      rho=2.65))))
    assert [(w.depth, w.where) for w in waves] == [
        (100.0, 'between the water and a sediment layer')]
    assert scholte_waves(_over(_halfspace(1800.0))) == []


def test_the_c_low_factor_is_scooters():
    """``scooter.f90:258`` (0.90), one factor for fluid and elastic
    problems: the Scholte wave scooter.f90:256's 0.85 estimates is
    solved instead."""
    from uacpy.core import engine_defaults
    assert engine_defaults.C_LOW_FACTOR == 0.90
    assert not hasattr(engine_defaults, 'C_LOW_FACTOR_ELASTIC')
