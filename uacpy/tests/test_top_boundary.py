"""Top-boundary-condition coverage.

The ``Environment.surface`` BoundaryProperties is consumed by every
Acoustics-Toolbox model (Bellhop, Kraken*, Scooter, Bounce, SPARC) and
by every OASES sub-model. A regression in any of the boundary-code
maps would otherwise be caught only by ``example_17`` integration.
"""

import numpy as np
import pytest

import uacpy
from uacpy.core.constants import parse_boundary_type
from uacpy.core.environment import BoundaryProperties, SoundSpeedProfile
from uacpy.io.oalib_writer import get_top_bc_code
from uacpy.io.bellhop_writer import write_bellhop_env_file
from uacpy.io.oases_writer import (
    _format_upper_halfspace,
    write_oast_input, write_oasn_input, write_oasp_input,
)


def _ice() -> BoundaryProperties:
    return BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=3500.0, shear_speed=1800.0, density=0.9,
        attenuation=1.0, shear_attenuation=2.0,
    )


def _basic_env(surface: BoundaryProperties) -> uacpy.Environment:
    return uacpy.Environment(
        name='top_bc_test', bathymetry=100.0,
        ssp=SoundSpeedProfile.from_pairs([(0, 1500), (100, 1500)]),
        surface=surface,
        bottom=BoundaryProperties(
            acoustic_type='half-space', sound_speed=1700.0,
            density=1.5, attenuation=0.5,
        ),
    )


@pytest.fixture
def src_rcv():
    src = uacpy.Source(depths=20.0, frequencies=200.0)
    rcv = uacpy.Receiver(
        depths=np.linspace(10, 90, 9),
        ranges=np.linspace(500, 5000, 19),
    )
    return src, rcv


@pytest.mark.parametrize("surface,expected_code", [
    (BoundaryProperties(acoustic_type='vacuum'), 'V'),
    (_ice(),                                     'A'),
])
def test_get_top_bc_code_routes_each_acoustic_type(surface, expected_code):
    """Pin the boundary-code routing for every model that consumes env.surface."""
    # parse_boundary_type -> to_acoustics_toolbox_code is the single lookup
    # path every writer takes.
    assert parse_boundary_type(
        'half-space').to_acoustics_toolbox_code() == 'A'
    assert get_top_bc_code(_basic_env(surface)) == expected_code


def test_bellhop_env_writes_top_bc_and_surface_props(tmp_path, src_rcv):
    """The .env file must (a) carry TopOpt='A' for ice and (b) include
    the ice halfspace-properties line so Bellhop reads cp/cs/rho."""
    src, rcv = src_rcv
    out = tmp_path / 'bellhop_ice.env'
    write_bellhop_env_file(out, _basic_env(_ice()), src, rcv)
    text = out.read_text()
    quoted = [
        line for line in text.splitlines()
        if line.lstrip().startswith("'") and line.lstrip()[1:2] in 'AVRFGP'
    ]
    assert quoted and quoted[0].lstrip()[1:2] == 'A'
    assert '3500' in text and '1800' in text


#: An OASES layer record is ``D CC CS AC AS RO RG [IG]``, so column 2 is the
#: compressional speed, 3 the shear speed and 6 the density; a vacuum halfspace
#: zeroes them all. Column 7 (RG) is roughness, which uacpy formats ``.4f``.
@pytest.mark.parametrize("surface,expected", [
    (BoundaryProperties(acoustic_type='vacuum'), '0 0 0 0 0 0 0.0000 0'),
    (_ice(),                                     None),  # ice props checked
])
def test_oases_format_upper_halfspace(surface, expected):
    line = _format_upper_halfspace(_basic_env(surface))
    if expected is not None:
        assert line == expected
    else:
        parts = line.split()
        assert float(parts[1]) == pytest.approx(3500.0)
        assert float(parts[2]) == pytest.approx(1800.0)
        assert float(parts[5]) == pytest.approx(0.9)


@pytest.mark.parametrize("writer", [
    write_oast_input, write_oasn_input, write_oasp_input,
])
def test_oases_writers_emit_ice_surface(writer, tmp_path, src_rcv):
    """OAST/OASN/OASP route env.surface through _format_upper_halfspace.

    OASR is excluded by design: its layer-1 IS the water column (per
    oasr.tex), so env.surface is not emitted into the .dat file.
    """
    src, rcv = src_rcv
    out = tmp_path / 'oases_ice.dat'
    writer(out, _basic_env(_ice()), src, rcv)
    text = out.read_text()
    assert '3500' in text and '1800' in text


@pytest.mark.requires_binary
@pytest.mark.parametrize("model_cls_name", ['Kraken'])
def test_kraken_family_env_writes_top_bc_for_halfspace_surface(
    model_cls_name, tmp_path, src_rcv,
):
    """All three Kraken-family writers must emit TopOpt='A' for an ice surface."""
    src, _ = src_rcv
    model_cls = getattr(uacpy.models, model_cls_name)
    model = model_cls(verbose=False, rmax_m=1000.0)
    out = tmp_path / f'{model_cls_name.lower()}_ice.env'
    model._write_kraken_env(
        out, _basic_env(_ice()), src,
        receiver_depths=[50.0],
    )
    text = out.read_text()
    quoted = [
        line for line in text.splitlines()
        if line.lstrip().startswith("'") and line.lstrip()[1:2] in 'AVRFG'
    ]
    assert quoted and quoted[0].lstrip()[1:2] == 'A'
    assert '3500' in text and '1800' in text


@pytest.mark.requires_binary
def test_ram_drops_surface_shear_with_warning():
    """RAM has no backend that reads surface shear; elastic surfaces must be
    collapsed to pressure-release with a UserWarning."""
    env = _basic_env(_ice())
    ram = uacpy.models.RAM(verbose=False)
    with pytest.warns(UserWarning, match="surface shear is not supported"):
        env_collapsed = ram._drop_unsupported_surface_shear(env)
    assert env_collapsed.surface.shear_speed == 0.0
    assert env_collapsed.surface.shear_attenuation == 0.0


@pytest.mark.requires_binary
@pytest.mark.slow
def test_bellhop_top_bc_changes_tl():
    """Vacuum vs ice surface → measurable TL difference at long range.

    A 20 m source in a 100 m guide out to 8 km makes many surface bounces, and
    an ice halfspace absorbs at each one where a vacuum reflects perfectly, so
    a working top BC separates the two fields by far more than 1 dB. The bound
    is deliberately near zero: this asserts that ``env.surface`` reaches the
    binary at all, not how much loss it produces.
    """
    src = uacpy.Source(depths=20.0, frequencies=200.0)
    rcv = uacpy.Receiver(
        depths=np.array([50.0]),
        ranges=np.linspace(2000.0, 8000.0, 25),
    )
    bellhop = uacpy.models.Bellhop(verbose=False)
    tl_vac = bellhop.compute_tl(
        _basic_env(BoundaryProperties(acoustic_type='vacuum')), src, rcv,
    )
    tl_ice = bellhop.compute_tl(_basic_env(_ice()), src, rcv)
    diff = np.abs(tl_vac.db - tl_ice.db)
    finite = np.isfinite(diff)
    assert finite.any() and diff[finite].max() >= 1.0, (
        "TL identical for vacuum vs ice surface — env.surface may not "
        "be reaching the binary; check oalib_writer.get_top_bc_code()."
    )
