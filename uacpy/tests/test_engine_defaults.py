"""The engine knob defaults keep their values now that one constant
states each, and the writers resolve None to the same value."""
import inspect

import pytest

import uacpy
import uacpy.io as io

#: (engine, knob, value) as the constructors stated them before the move.
DEFAULTS = [
    ('Bellhop', 'beam_type', 'G'), ('Bellhop', 'n_beams', 0),
    ('Bellhop', 'launch_angles', (-80, 80)), ('Bellhop', 'ray_step', 0.0),
    ('Bellhop', 'grid_type', 'R'), ('Bellhop', 'interp_bathymetry', 'linear'),
    ('Bellhop', 'interp_altimetry', 'linear'), ('Bellhop', 'beam_shift', False),
    ('Kraken', 'n_mesh', 0), ('Scooter', 'n_mesh', 0), ('SPARC', 'n_mesh', 0),
    ('SPARC', 'output_mode', 'R'), ('OAST', 'integration_offset', 0.0),
    ('OAST', 'vrec', 0.0), ('OASN', 'integration_offset', 0.0),
    ('OASP', 'integration_offset', 0.0), ('OASP', 'freq_min', 0.0),
    ('OASR', 'angle_type', 'grazing'), ('OASSP', 'spectral_exponent', 2.0),
    ('OASSP', 'realization', 0), ('OASSP', 'integration_offset', 0.0),
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
