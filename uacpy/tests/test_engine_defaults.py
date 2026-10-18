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
    ('Bellhop', 'n_frequencies', None), ('Bellhop', 'bandwidth_factor', None),
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
    from uacpy.core import Bottom, SeabedColumn
    return [
        ('vacuum bottom', _over(_parameter_free('vacuum')), 0.90 * 1490.0,
         'the water'),
        ('rigid bottom, no medium', _over(_parameter_free('rigid')),
         0.90 * 1490.0, 'the water'),
        ('fluid half-space slower than the water', _over(_halfspace(1450.0)),
         0.90 * 1450.0, 'the bottom half-space'),
        ('elastic half-space', _over(_halfspace(1800.0, 600.0)),
         0.85 * 600.0, "the bottom half-space's shear speed"),
        ('elastic layer on a fluid half-space',
         _over(_sand_over(_halfspace(1800.0))), 0.90 * 110.0,
         "a sediment layer's shear speed"),
        ('elastic layer on an elastic half-space',
         _over(_sand_over(_halfspace(5250.0, 3000.0, rho=2.65))),
         0.85 * 110.0, "a sediment layer's shear speed"),
        ('elastic top half-space over fluid water',
         _over(_halfspace(1800.0), surface=_halfspace(3500.0, 1800.0,
                                                       rho=0.9)),
         0.85 * 1490.0, 'the water'),
        ('the slowest column of a range-dependent seabed',
         _over(Bottom.from_columns(
             [SeabedColumn(layers=[], halfspace=_halfspace(1800.0)),
              SeabedColumn(layers=[], halfspace=_halfspace(1460.0))],
             ranges=[0.0, 5000.0])), 0.90 * 1460.0, 'the bottom half-space'),
    ]


@pytest.mark.parametrize('case', range(8))
def test_the_derived_c_low_is_the_slowest_speed_of_every_medium(case):
    """``scooter.f90:193-259``: cMin over the water, each sediment layer
    (its shear speed when elastic) and each acoustic half-space (shear when
    elastic), × 0.85 when a half-space — bottom or top — is elastic and
    × 0.90 otherwise; a rigid or vacuum boundary carries no medium. The
    origin names the medium that set it."""
    from uacpy.core.engine_defaults import default_c_low
    label, env, expected, where = _cases()[case]
    c_low, origin = default_c_low(env)
    assert c_low == pytest.approx(expected, rel=1e-12), label
    assert f", {where} — the slowest speed of every medium" in origin
    assert origin.startswith(
        '0.85 × ' if expected in (0.85 * 600.0, 0.85 * 110.0, 0.85 * 1490.0)
        else '0.9 × ')


def test_the_c_low_factors_are_scooters():
    """``scooter.f90:256`` (0.85, an elastic half-space) and :258 (0.90)."""
    from uacpy.core.engine_defaults import C_LOW_FACTOR, C_LOW_FACTOR_ELASTIC
    assert (C_LOW_FACTOR, C_LOW_FACTOR_ELASTIC) == (0.90, 0.85)
