"""Input-validation guards, grouped by the G1-G8 labels used in the section
headers below: monotonicity rejection (G1), range-coverage warnings (G3),
per-range receiver-depth checks (G4), the RAM Collins RD-bottom path (G2),
the range=0 SSP fallback in Bellhop's .env block (G6) and the acoustic_type
guard (G8).

Most cases reach only ``validate_inputs`` / ``_project_environment`` / a
writer, so they need no binary; the handful marked ``requires_binary``
construct a model (which resolves its executable) and a few of those do run
one. Whole file: a few seconds.
"""

import re
import warnings

import functools

import numpy as np
import pytest

import uacpy
from uacpy.core.exceptions import (
    ConfigurationError, ProvenanceWarning, UnsupportedFeatureError,
)
from uacpy.models.bellhop import Bellhop
from uacpy.models.ram import RAM
from uacpy.models.scooter import Scooter
from uacpy.core.environment import (
    BoundaryProperties,
    Environment,
    SeabedColumn,
    Bottom,
    SedimentLayer,
    SoundSpeedProfile,
)
from uacpy.tests.conftest import build_engine, engine_entry, engine_params
from uacpy.models.ram import _band as ram_band
from uacpy.tests.conftest import recorded_warnings
from uacpy.tests.conftest import make_pekeris


# --- G1 monotonicity -------------------------------------------------------

# ── assignment is checked as construction is (D5, one carrier style) ─────────
#
# Each row: a valid carrier, a field and a value its constructor refuses. The
# constructor must refuse the value, and so must an assignment to a built
# carrier, leaving it as it was. A row whose constructor accepted the value
# would test nothing, so that half is asserted first.
def _ssp():
    return uacpy.SoundSpeedProfile(depths=[0.0, 100.0],
                                   sound_speed=[1500.0, 1490.0])


def _layer():
    return SedimentLayer(thickness=10.0, sound_speed=1650.0, density=1.9)


def _halfspace():
    return BoundaryProperties(sound_speed=1700.0, density=1.8)


def _fg():
    return uacpy.FrancoisGarrison(10.0, 35.0, 8.0)


_REFUSED_ASSIGNMENTS = [
    (_ssp, 'depths', [5.0, 1.0]),
    (_ssp, 'sound_speed', [-1.0, -1.0]),
    (lambda: uacpy.Bathymetry(ranges=[0.0, 1000.0], depths=[100.0, 120.0]),
     'depths', [-1.0, -1.0]),
    (lambda: uacpy.Altimetry(ranges=[0.0, 1000.0], heights=[0.0, 1.0]),
     'heights', [np.nan, 1.0]),
    (_layer, 'thickness', -1.0),
    (_layer, 'attenuation', -0.5),
    (_halfspace, 'density', -1.0),
    (_halfspace, 'acoustic_type', 'halfspace'),
    (lambda: BoundaryProperties(), 'density', 2.0),
    (_fg, 'temperature', float('nan')),
    (_fg, 'pH', 14.1),
    (_fg, 'pH', -0.1),
    (lambda: uacpy.BiologicalLayer(10.0, 50.0, 1000.0, 5.0, 0.01),
     'z_bottom_m', 5.0),
    (lambda: uacpy.ConstantAbsorption(0.1), 'value_dB_per_wavelength', -1.0),
    (lambda: uacpy.Source(depths=50.0, frequencies=100.0), 'depths', -5.0),
    (lambda: uacpy.Receiver(depths=[10.0], ranges=[100.0, 200.0]),
     'ranges', [3.0, 1.0]),
    (_layer, 'roughness', -1.0),
    (_halfspace, 'roughness', -1.0),
    (lambda: BoundaryProperties(acoustic_type='vacuum'), 'roughness', -1.0),
    (lambda: Environment(bathymetry=100.0), 'location', (999.0, 0.0)),
    (lambda: Environment(bathymetry=100.0), 'transect', ((43.0, 5.0),)),
    (lambda: Environment(bathymetry=100.0), 'bathymetry', -10.0),
    (lambda: Environment(bathymetry=100.0), 'water_density', 1027.0),
    (lambda: Environment(bathymetry=100.0), 'absorption', 'thorp'),
    (lambda: Environment(bathymetry=100.0), 'extra_data_sources',
     ('GEBCO',)),
]


@pytest.mark.parametrize('build, field, bad', _REFUSED_ASSIGNMENTS,
                         ids=[f'{i}-{row[1]}' for i, row
                              in enumerate(_REFUSED_ASSIGNMENTS)])
def test_every_carrier_refuses_an_assignment_its_constructor_refuses(
        build, field, bad):
    import dataclasses
    import re
    carrier = build()
    kwargs = {f.name: carrier.__dict__[f.name]
              for f in dataclasses.fields(carrier)}
    if isinstance(carrier, BoundaryProperties):
        kwargs = carrier._fields_for_assignment(field, bad)
    kwargs[field] = bad
    try:
        type(carrier)(**kwargs)
    except ConfigurationError as exc:
        refusal = str(exc)
    else:
        pytest.fail(f"{type(carrier).__name__}({field}={bad!r}) was accepted "
                    f"by the constructor; the row tests nothing")
    before = repr(carrier)
    # The assignment is refused with the constructor's own message.
    with pytest.raises(ConfigurationError, match=re.escape(refusal)):
        setattr(carrier, field, bad)
    assert repr(carrier) == before


def test_a_carrier_is_under_construction_only_while_its_init_runs():
    """``RevalidateOnAssignMixin`` treats a store as construction while the
    carrier's id is in ``_carrier``'s thread-local set. The id is added on
    entry to ``__init__`` and removed in a ``finally``, so the set is empty
    between constructions — also after a refused one — and a carrier born at
    an address a freed one used is checked on assignment like any other.
    Thousands of short-lived carriers make the address reuse happen."""
    from uacpy.core._carrier import _under_construction
    assert _under_construction() == set()
    with pytest.raises(ConfigurationError, match='thickness'):
        SedimentLayer(thickness=-1.0, sound_speed=1650.0, density=1.9)
    assert _under_construction() == set()
    seen = set()
    for i in range(5000):
        layer = SedimentLayer(thickness=10.0 + i % 7, sound_speed=1650.0,
                              density=1.9)
        seen.add(id(layer))
        assert _under_construction() == set()
        if i % 250 == 0:
            with pytest.raises(ConfigurationError, match='thickness'):
                layer.thickness = -1.0
            assert layer.thickness == 10.0 + i % 7
        del layer
    # The loop reused addresses: fewer distinct ids than carriers built.
    assert len(seen) < 5000


def test_an_assignment_is_normalised_as_the_constructor_normalises():
    """An accepted assignment is stored as the constructor would store it:
    a list of speeds becomes the ``(n_depth, 1)`` float array."""
    ssp = _ssp()
    ssp.sound_speed = [1510.0, 1500.0]
    assert isinstance(ssp.sound_speed, np.ndarray)
    assert ssp.sound_speed.shape == (2, 1)


@pytest.mark.parametrize('kind', ['vacuum', 'rigid'])
def test_a_parameter_free_boundary_takes_a_roughness_and_refuses_geoacoustics(
        kind):
    """A vacuum or rigid node holds the resolved defaults of the half-space
    parameters it ignores; rebuilding it with them passed explicitly would
    read as the conflict its constructor refuses, so an interface write and a
    provenance write go through, and a geoacoustic one is refused as
    ``BoundaryProperties(kind, density=2)`` is."""
    node = BoundaryProperties(acoustic_type=kind)
    node.roughness = 1.0
    node.data_sources = ()
    assert (node.acoustic_type, node.roughness) == (kind, 1.0)
    for field in ('density', 'sound_speed', 'attenuation', 'shear_speed',
                  'shear_attenuation'):
        with pytest.raises(ConfigurationError, match='ignores half-space'):
            setattr(node, field, 2.0)
    assert (node.acoustic_type, node.roughness) == (kind, 1.0)
    halfspace = _halfspace()
    halfspace.acoustic_type = kind
    assert repr(halfspace) == f'BoundaryProperties({kind})'


def test_ssp_depth_must_be_strictly_increasing():
    with pytest.raises(ConfigurationError, match="strictly increasing"):
        SoundSpeedProfile.from_pairs([(0, 1500), (10, 1490), (5, 1495)])


def test_ssp_ranges_must_be_strictly_increasing():
    depths = np.array([0.0, 100.0])
    data = np.array([[1500.0, 1490.0, 1500.0], [1480.0, 1470.0, 1480.0]])
    with pytest.raises(ConfigurationError, match="strictly increasing"):
        SoundSpeedProfile(
            depths=depths, sound_speed=data,
            ranges=np.array([0.0, 5000.0, 3000.0]),
        )


def test_ssp_duplicate_depths_rejected():
    with pytest.raises(ConfigurationError, match="strictly increasing"):
        SoundSpeedProfile.from_pairs([(0, 1500), (10, 1490), (10, 1480)])


def test_rd_bottom_ranges_must_be_strictly_increasing():
    with pytest.raises(ConfigurationError, match="strictly increasing"):
        Bottom.from_halfspaces(
            np.array([0.0, 5000.0, 3000.0]),
            sound_speed=np.array([1600.0, 1700.0, 1800.0]),
            density=np.array([1.5, 1.6, 1.7]),
            attenuation=np.array([0.3, 0.4, 0.5]),
            acoustic_type='half-space',
        )


def test_rd_bottom_shear_array_length_validated():
    # A mismatched explicit shear array must raise ConfigurationError at
    # construction (not a bare numpy ValueError later inside at()).
    with pytest.raises(ConfigurationError, match="shear_speed length"):
        Bottom.from_halfspaces(
            np.array([0.0, 1000.0, 2000.0]),
            sound_speed=np.array([1600.0, 1700.0, 1800.0]),
            density=np.array([1.5, 1.6, 1.7]),
            attenuation=np.array([0.3, 0.4, 0.5]),
            shear_speed=np.array([200.0, 300.0]),     # length 2 != 3
        )


def test_sediment_layer_rejects_negative_attenuation():
    with pytest.raises(ConfigurationError, match="attenuation must be .*non-negative"):
        SedimentLayer(thickness=5, sound_speed=1600, density=1.6, attenuation=-0.1)


def test_rd_layered_bottom_ranges_must_be_strictly_increasing():
    layer = SedimentLayer(thickness=5, sound_speed=1600, density=1.6, attenuation=0.4)
    hs = BoundaryProperties(acoustic_type='half-space',
                            sound_speed=1800, density=2.0, attenuation=0.1)
    lb = SeabedColumn(layers=[layer], halfspace=hs)
    with pytest.raises(ConfigurationError, match="strictly increasing"):
        Bottom.from_columns([lb, lb, lb], ranges=np.array([0.0, 1000.0, 500.0]))


def test_bathymetry_must_be_strictly_increasing():
    with pytest.raises(ConfigurationError, match="strictly increasing"):
        Environment(bathymetry=[(0.0, 100.0), (5000.0, 200.0), (3000.0, 150.0)])


def test_altimetry_must_be_strictly_increasing():
    with pytest.raises(ConfigurationError, match="strictly increasing"):
        Environment(
            bathymetry=100.0,
            altimetry=[(0.0, 0.0), (2000.0, 1.0), (1000.0, 0.5)],
        )


def test_receiver_grid_ranges_must_be_increasing():
    with pytest.raises(ConfigurationError, match="strictly increasing"):
        uacpy.Receiver(depths=np.array([10.0, 20.0]),
                       ranges=np.array([1000.0, 500.0]))


# --- G8 acoustic_type ------------------------------------------------------

def test_acoustic_type_takes_each_value_in_any_case():
    assert BoundaryProperties(acoustic_type='HALF-SPACE').acoustic_type \
        == 'half-space'
    assert BoundaryProperties(acoustic_type='Rigid').acoustic_type == 'rigid'


@pytest.mark.parametrize('alias', ['halfspace', 'elastic', 'half_space',
                                   'HALF_SPACE', 'A', 'v', 'R', 'f', 'p'])
def test_acoustic_type_refuses_every_other_spelling(alias):
    with pytest.raises(ConfigurationError, match="not recognized"):
        BoundaryProperties(acoustic_type=alias)


def test_acoustic_type_typo_rejected():
    with pytest.raises(ConfigurationError, match="not recognized"):
        BoundaryProperties(acoustic_type='vaccum')


def test_rd_bottom_acoustic_type_validated():
    with pytest.raises(ConfigurationError, match="not recognized"):
        Bottom.from_halfspaces(
            np.array([0.0, 1000.0]),
            sound_speed=np.array([1600.0, 1700.0]),
            density=np.array([1.5, 1.6]),
            attenuation=np.array([0.3, 0.4]),
            acoustic_type='spam-eggs',
        )


# --- G4 per-range receiver depth check -------------------------------------

@pytest.mark.requires_binary  # constructs a model (resolves its binary)
def test_per_range_receiver_below_shoaling_seafloor():
    bellhop = Bellhop()
    env = Environment(
        bathymetry=[(0.0, 200.0), (10_000.0, 50.0)],
        ssp=1500.0,
    )
    src = uacpy.Source(depths=10.0, frequencies=100.0)
    rcv = uacpy.Receiver(
        depths=np.array([100.0]),
        ranges=np.array([1000.0, 5000.0, 9000.0]),
    )
    with recorded_warnings() as caught:
        bellhop.validate_inputs(env, src, rcv, run_mode=uacpy.RunMode.COHERENT_TL)
    msgs = [str(w.message) for w in caught]
    assert any("below the local seafloor" in m for m in msgs)


@pytest.mark.requires_binary  # constructs a model (resolves its binary)
def test_per_range_receiver_check_passes_when_under_seafloor():
    bellhop = Bellhop()
    env = Environment(
        bathymetry=[(0.0, 200.0), (10_000.0, 50.0)],
        ssp=1500.0,
    )
    src = uacpy.Source(depths=10.0, frequencies=100.0)
    rcv = uacpy.Receiver(
        depths=np.array([30.0]),
        ranges=np.array([1000.0, 5000.0, 9000.0]),
    )
    bellhop.validate_inputs(env, src, rcv, run_mode=uacpy.RunMode.COHERENT_TL)


@pytest.mark.requires_binary  # constructs a model (resolves its binary)
@pytest.mark.parametrize('run_mode', [None, 'coherent_tl'])
def test_validate_inputs_resolves_the_default_and_string_run_mode(run_mode):
    """``validate_inputs`` resolves ``run_mode`` as ``run()`` does: the
    default ``None`` and the value string both check as ``COHERENT_TL`` and
    refuse a two-frequency Source; one frequency passes."""
    bellhop = Bellhop()
    env = Environment(bathymetry=100.0, ssp=1500.0)
    rcv = uacpy.Receiver(depths=[30.0], ranges=[1000.0])
    with pytest.raises(ConfigurationError, match='single source frequency'):
        bellhop.validate_inputs(
            env, uacpy.Source(depths=10.0, frequencies=[100.0, 200.0]), rcv,
            run_mode=run_mode)
    bellhop.validate_inputs(
        env, uacpy.Source(depths=10.0, frequencies=100.0), rcv,
        run_mode=run_mode)


# --- G3 range coverage warning ---------------------------------------------

@pytest.mark.requires_binary  # constructs a model (resolves its binary)
def test_warn_when_receiver_overruns_bathymetry():
    bellhop = Bellhop()
    env = Environment(
        bathymetry=[(0.0, 100.0), (5_000.0, 200.0)],
        ssp=1500.0,
    )
    src = uacpy.Source(depths=10.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]),
                         ranges=np.array([8_000.0]))
    with recorded_warnings() as caught:
        bellhop.validate_inputs(env, src, rcv, run_mode=uacpy.RunMode.COHERENT_TL)
    msgs = [str(w.message) for w in caught]
    assert any("env.bathymetry" in m and "constant-extrapolated" in m for m in msgs)


@pytest.mark.requires_binary  # constructs a model (resolves its binary)
def test_warn_when_receiver_overruns_ssp_ranges():
    bellhop = Bellhop()
    ssp = SoundSpeedProfile.from_2d(
        depths=np.array([0.0, 100.0]),
        ranges=np.array([0.0, 2_000.0]),
        matrix=np.array([[1500.0, 1495.0], [1480.0, 1475.0]])
    )
    env = Environment(bathymetry=100.0, ssp=ssp)
    src = uacpy.Source(depths=10.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]),
                         ranges=np.array([5_000.0]))
    with recorded_warnings() as caught:
        bellhop.validate_inputs(env, src, rcv, run_mode=uacpy.RunMode.COHERENT_TL)
    msgs = [str(w.message) for w in caught]
    assert any("env.ssp.ranges" in m for m in msgs)


# --- G2 RAM Collins RD-bottom warning --------------------------------------

def _elastic_rd_bottom(sound_speed, density, attenuation):
    """Two-column elastic bottom breaking at 2 km. Shear is held constant across
    the break: rams0.5 goes unstable on a 500 m/s shear column at this grid
    whether or not the bottom is range-dependent, which would mask the property
    under test."""
    return Bottom.from_halfspaces(
        np.array([0.0, 2_000.0]),
        sound_speed=np.array(sound_speed),
        density=np.array(density),
        attenuation=np.array(attenuation),
        shear_speed=np.array([400.0, 400.0]),
        acoustic_type='half-space',
    )


@pytest.mark.requires_binary  # constructs RAM (resolves its binary)
def test_ram_collins_threads_rd_bottom(tmp_path):
    """The Collins backends emit one ``ram.in`` profile section per range
    break, so a range-dependent bottom is modelled rather than reduced to its
    r=0 column. No 'range-0' warning may therefore be raised on this env."""
    env = Environment(bathymetry=100.0, ssp=1500.0,
                      bottom=_elastic_rd_bottom([1700.0, 1800.0], [1.7, 1.9],
                                                [0.5, 0.4]))
    assert env.bottom.is_elastic
    ram = RAM(work_dir=str(tmp_path), cleanup=False)
    assert ram.select_backend(env) == 'rams'
    src = uacpy.Source(depths=10.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]),
                         ranges=np.array([2_000.0]))
    with recorded_warnings() as caught:
        field = ram.run(env, src, rcv, run_mode=uacpy.RunMode.COHERENT_TL)
    p_rd = complex(np.asarray(field.data).ravel()[0])
    assert np.isfinite(p_rd)
    assert not any("range-0" in str(w.message) for w in caught)

    # rams writes six profile blocks per section, so bathymetry plus two
    # sections is 13 terminators. The section marker is the midpoint of the two
    # breakpoints, inside the 2 km march, and each section carries its own
    # compressional column. Values are parsed from the tokens rather than
    # matched as strings so the assertion is independent of the writer's
    # float format (%.12g renders the marker as a bare '1000').
    deck = (tmp_path / 'rams.in').read_text().splitlines()
    assert deck.count('-1 -1') == 1 + 6 * 2

    def _lone_token(line):
        tokens = line.split()
        if len(tokens) != 1:
            return None
        try:
            return float(tokens[0])
        except ValueError:
            return None

    # The section marker is the deck's only single-token line.
    markers = [i for i, ln in enumerate(deck) if _lone_token(ln) == 1000.0]
    assert len(markers) == 1
    cut = markers[0]

    def _column_values(lines):
        values = set()
        for ln in lines:
            tokens = ln.split()
            if len(tokens) == 2:
                try:
                    values.add(float(tokens[1]))
                except ValueError:
                    pass
        return values

    head, tail = _column_values(deck[:cut]), _column_values(deck[cut:])
    assert 1700.0 in head
    assert 1800.0 not in head
    assert 1800.0 in tail
    assert 1700.0 not in tail

    # Reducing the bottom to its r=0 column would reproduce this field exactly.
    p_r0 = complex(np.asarray(RAM().run(
        Environment(bathymetry=100.0, ssp=1500.0,
                    bottom=_elastic_rd_bottom([1700.0, 1700.0], [1.7, 1.7],
                                              [0.5, 0.5])),
        src, rcv, run_mode=uacpy.RunMode.COHERENT_TL).data).ravel()[0])
    assert abs(p_rd - p_r0) > 0.05 * abs(p_r0)


def test_oalib_writer_keeps_thin_layers():
    """Every layer of positive thickness reaches the deck.

    ``deck_depth`` rounds each interface *up*, so a sub-quantum layer is modelled
    at one quantum rather than collapsing to a degenerate (top == bottom)
    medium — measured on a 0.08 m layer at 5 kHz against an exact plane-wave
    impedance recursion, writing it costs 5e-4 in |R| while omitting it costs
    0.212, i.e. 4.7 dB per bottom bounce. Both AT and OASES read the depth
    column list-directed (``misc/ReadEnvironmentMod.f90:88``,
    ``misc/sspMod.f90:334``, ``oases/src/oaseun31.f:54``), so no format forbids a
    thin layer, and the depth column is written at a resolution fine enough to
    carry the layer at its own thickness. ``SedimentLayer`` already refuses a
    non-positive thickness at construction, so the writer has no degenerate case
    left to guard.
    """
    from uacpy.io.oalib_writer import writable_layers
    hs = BoundaryProperties(acoustic_type='half-space', sound_speed=1800,
                            density=2.0, attenuation=0.1)
    thin = SeabedColumn(
        layers=[SedimentLayer(thickness=0.08, sound_speed=1600, density=1.6,
                              attenuation=0.5),
                SedimentLayer(thickness=20.0, sound_speed=1700, density=1.8,
                              attenuation=0.4)],
        halfspace=hs)
    kept = writable_layers(thin)
    assert [round(lyr.thickness, 2) for lyr in kept] == [0.08, 20.0]

    # The carrier is where a degenerate layer is stopped, so the writer needs no
    # guard of its own.
    with pytest.raises(ConfigurationError, match='thickness must be positive'):
        SedimentLayer(thickness=0.0, sound_speed=1600, density=1.6,
                      attenuation=0.5)


# --- G6 Bellhop .env range=0 SSP fallback ----------------------------------

def test_bellhop_env_ssp_block_uses_range_zero_profile(tmp_path):
    """The .env SSP block is the profile at r=0 even when the 2-D SSP carries
    no r=0 column: the grid here starts at 1000 m, so r=0 is the constant
    back-extrapolation of the first profile."""
    from uacpy.io.bellhop_writer import write_bellhop_env_file

    ssp = SoundSpeedProfile.from_2d(
        depths=np.array([0.0, 100.0]),
        ranges=np.array([1_000.0, 5_000.0]),
        matrix=np.array([[1500.0, 1480.0], [1490.0, 1470.0]])
    )
    env = Environment(bathymetry=100.0, ssp=ssp)
    src = uacpy.Source(depths=10.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]),
                         ranges=np.array([2_500.0]))
    env_path = tmp_path / 'test.env'
    write_bellhop_env_file(env_path, env, src, rcv)

    # AT's SSP block opens with the medium mesh line ``NMESH SIGMA ZMAX,``
    # (trailing comma, e.g. ``2  0.0  100.0,``) and then runs one
    # ``z c ... /`` record per depth, so the comma opens the block and the
    # first line without a ``/`` closes it.
    block = []
    in_ssp = False
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not in_ssp and stripped.endswith(",") and "0.0" in stripped:
            in_ssp = True
            continue
        if in_ssp:
            parts = stripped.split()
            if not parts or "/" not in line:
                break
            block.append((float(parts[0]), float(parts[1])))

    assert len(block) >= 2
    # Surface sound speed of the first profile (r = 1000 m), which constant
    # back-extrapolation carries to r=0 unchanged.
    assert block[0][1] == pytest.approx(1500.0, abs=1e-3)


# --- Positive-path tests: independent grids should reconcile cleanly -------

def test_ssp_eval_interpolates_off_grid_range():
    ssp = SoundSpeedProfile.from_2d(
        depths=np.array([0.0, 100.0]),
        ranges=np.array([0.0, 4_000.0, 10_000.0]),
        matrix=np.array([[1500.0, 1490.0, 1480.0],
                         [1480.0, 1470.0, 1460.0]])
    )
    sliced = ssp.eval(range=2_000.0)
    assert sliced.sound_speed[0, 0] == pytest.approx(1495.0)
    assert sliced.sound_speed[1, 0] == pytest.approx(1475.0)


def test_ssp_eval_clamps_beyond_last_range():
    ssp = SoundSpeedProfile.from_2d(
        depths=np.array([0.0, 100.0]),
        ranges=np.array([0.0, 4_000.0]),
        matrix=np.array([[1500.0, 1490.0], [1480.0, 1470.0]])
    )
    sliced = ssp.eval(range=10_000.0)
    assert sliced.sound_speed[0, 0] == pytest.approx(1490.0)
    assert sliced.sound_speed[1, 0] == pytest.approx(1470.0)


def test_rd_bottom_halfspace_at_steps_to_the_nearest_column_at_the_midpoint():
    rd = Bottom.from_halfspaces(
        np.array([0.0, 5_000.0]),
        sound_speed=np.array([1600.0, 1800.0]),
        density=np.array([1.5, 1.9]),
        attenuation=np.array([0.3, 0.5]),
        acoustic_type='half-space',
    )
    below = rd.halfspace_at(range=2_500.0 - 1e-6)
    above = rd.halfspace_at(range=2_500.0 + 1e-6)
    assert (below.sound_speed, below.density) == (1600.0, 1.5)
    assert (above.sound_speed, above.density) == (1800.0, 1.9)


def test_bathymetry_eval_interpolates_off_grid():
    env = Environment(bathymetry=[(0.0, 100.0), (10_000.0, 200.0)])
    assert env.bathymetry.eval(range=5_000.0) == pytest.approx(150.0)
    # Constant extrapolation past the last range.
    assert env.bathymetry.eval(range=20_000.0) == pytest.approx(200.0)


def test_independent_bathy_ssp_bottom_ranges_compose_ok():
    """Bathymetry, RD-SSP, and RD-bottom each have their own range axis
    of different lengths; the env constructs without complaint and
    everything is reachable via the lookup helpers."""
    bathy = [(0.0, 100.0), (2_000.0, 120.0), (8_000.0, 180.0)]
    ssp = SoundSpeedProfile.from_2d(
        depths=np.array([0.0, 200.0]),
        ranges=np.array([0.0, 5_000.0, 12_000.0]),
        matrix=np.array([[1500.0, 1495.0, 1490.0],
                         [1480.0, 1475.0, 1470.0]])
    )
    rd_bot = Bottom.from_halfspaces(
        np.array([0.0, 3_000.0, 6_000.0, 9_000.0]),
        sound_speed=np.array([1600.0, 1650.0, 1700.0, 1750.0]),
        density=np.array([1.5, 1.6, 1.7, 1.8]),
        attenuation=np.array([0.3, 0.35, 0.4, 0.45]),
        acoustic_type='half-space',
    )
    env = Environment(bathymetry=bathy, ssp=ssp, bottom=rd_bot)
    assert env.is_range_dependent
    assert env.bathymetry.eval(range=4_000.0) == pytest.approx(140.0)
    assert env.ssp.eval(range=4_000.0).sound_speed[0, 0] == pytest.approx(1496.0)
    # The seabed steps midway between its 3 and 6 km columns.
    assert env.bottom.halfspace_at(range=4_500.0 - 1e-6).sound_speed == 1650.0
    assert env.bottom.halfspace_at(range=4_500.0 + 1e-6).sound_speed == 1700.0


def test_bty_long_format_rows_are_bathymetry_nodes_plus_column_switches(tmp_path):
    """The written rows are the union of the bathymetry nodes and the switch
    midway between the bottom's columns; the switch row carries the next
    column (Bellhop holds a row's geoacoustics to its right), depth is
    interpolated onto every row, and there is no other row."""
    from uacpy.io.bathy_io import write_bty_long_format

    bathy = np.array([[0.0, 100.0],
                      [3_000.0, 130.0],
                      [9_000.0, 200.0]])
    rd_bot = Bottom.from_halfspaces(
        np.array([0.0, 6_000.0]),
        sound_speed=np.array([1600.0, 1800.0]),
        density=np.array([1.5, 1.9]),
        attenuation=np.array([0.3, 0.5]),
        acoustic_type='half-space',
    )
    out = tmp_path / 'test.bty'
    write_bty_long_format(out, bathy, rd_bot)
    lines = [ln.split() for ln in out.read_text().splitlines() if ln.strip()
             and not ln.strip().startswith("'")]
    n_rows = int(lines[0][0])
    rows = [list(map(float, row)) for row in lines[1:1 + n_rows]]
    # Bathymetry nodes 0, 3, 9 km; the one switch (midway 0 -> 6 km) is the
    # 3 km node itself; the 6 km bottom node is not a row.
    assert [r[0] for r in rows] == [0.0, 3.0, 9.0]
    assert n_rows == 3
    by_range = {r[0]: r for r in rows}
    assert by_range[0.0][2] == pytest.approx(1600.0)    # first column
    assert by_range[3.0][2] == pytest.approx(1800.0)    # switch row: next column
    assert by_range[3.0][1] == pytest.approx(130.0)     # depth at its node
    assert by_range[9.0][2] == pytest.approx(1800.0)    # still the second column


@pytest.mark.requires_binary  # constructs Scooter/Kraken/Bellhop (resolves their binaries)
def test_receiver_depth_accepted_across_models_harmonized():
    """A below-seafloor receiver never raises — it is accepted on every
    model, returning that model's below-domain value. Within its resolvable
    depth a model is silent; below it, one harmonized warning fires. Only
    the source (an input that must sit in the medium) is a hard error."""
    from uacpy.models.scooter import Scooter
    from uacpy.core.exceptions import InvalidDepthError

    ssp = SoundSpeedProfile.from_isovelocity(100.0, 1500.0)
    bottom = SeabedColumn(
        layers=[SedimentLayer(thickness=50.0, sound_speed=1600.0,
                              density=1.5, attenuation=0.5)],
        halfspace=BoundaryProperties(acoustic_type='half-space',
                                     sound_speed=1800.0, density=2.0,
                                     attenuation=0.8),
    )
    env = Environment(bathymetry=100.0, ssp=ssp, bottom=bottom)
    src = uacpy.Source(depths=20.0, frequencies=50.0)

    # Solvers that mesh through the sediment (Scooter, the Kraken family)
    # resolve a 130 m receiver in the sediment (env.depth=100, media=150)
    # → accepted, no warning.
    from uacpy.models.kraken import Kraken
    for model in (Scooter(), Kraken()):
        with recorded_warnings() as caught:
            model.validate_inputs(
                env, src,
                uacpy.Receiver(depths=np.array([50.0, 130.0]),
                               ranges=np.array([500.0, 1000.0])),
                run_mode=uacpy.RunMode.COHERENT_TL,
            )
        assert not any("resolvable depth" in str(w.message) for w in caught)

    # A ray model stops at the seafloor: the same 130 m receiver is still
    # accepted, but warns that the result reflects below-domain behaviour.
    with recorded_warnings() as caught:
        Bellhop().validate_inputs(
            env, src,
            uacpy.Receiver(depths=np.array([130.0]), ranges=np.array([500.0])),
        )
    assert any("below the model's resolvable depth" in str(w.message)
               for w in caught)

    # The source, by contrast, must lie within the resolvable medium.
    with pytest.raises(InvalidDepthError, match='exceeds resolvable depth'):
        Bellhop().validate_inputs(
            env, uacpy.Source(depths=130.0, frequencies=50.0),
            uacpy.Receiver(depths=np.array([50.0]), ranges=np.array([500.0])),
        )


@pytest.mark.requires_binary  # constructs Scooter (resolves its binary)
def test_scooter_collapses_rd_env_with_warning():
    """A model without RD support collapses the env and emits one warning per
    dropped axis; the env returned by ``_project_environment`` is
    range-independent regardless of input shape.

    Scooter is the vehicle because it is a range-independent FFP. Kraken
    segments RD natively via field.exe, so it would not collapse here."""
    scooter = Scooter()
    ssp = SoundSpeedProfile.from_2d(
        depths=np.array([0.0, 100.0]),
        ranges=np.array([0.0, 5_000.0]),
        matrix=np.array([[1500.0, 1480.0], [1490.0, 1470.0]])
    )
    env = Environment(
        bathymetry=[(0.0, 100.0), (5_000.0, 200.0)],
        ssp=ssp,
    )
    with recorded_warnings() as caught:
        projected = scooter._project_environment(env)
    assert not projected.is_range_dependent
    text = " ".join(str(w.message) for w in caught)
    assert "range-dependent bathymetry" in text
    assert "range-dependent SSP" in text


@pytest.mark.requires_binary  # constructs RAM (resolves its binary)
def test_per_range_receiver_below_seafloor_emits_warning_not_error():
    """RAM accepts receivers below the local seafloor; G4 should warn,
    not raise."""
    env = Environment(
        bathymetry=[(0.0, 200.0), (10_000.0, 50.0)],
        ssp=1500.0,
    )
    src = uacpy.Source(depths=10.0, frequencies=100.0)
    rcv = uacpy.Receiver(
        depths=np.array([80.0]),
        ranges=np.array([2_000.0, 9_000.0]),
    )
    ram = RAM()
    with recorded_warnings() as caught:
        ram.validate_inputs(env, src, rcv, run_mode=uacpy.RunMode.COHERENT_TL)
    assert any("below the local seafloor" in str(w.message) for w in caught)


def test_kraken_segmentation_unions_distinct_axes():
    """Kraken builds its segment list from the union of bathy / SSP
    / bottom change-points, so a bathy with 3 ranges and an SSP with 5
    ranges should yield at least 5 segments."""
    from uacpy.models.kraken._segments import segment_environment_by_range

    ssp = SoundSpeedProfile.from_2d(
        depths=np.array([0.0, 200.0]),
        ranges=np.array([0.0, 2_000.0, 4_000.0, 6_000.0, 10_000.0]),
        matrix=np.tile(np.array([[1500.0], [1480.0]]), (1, 5))
    )
    env = Environment(
        bathymetry=[(0.0, 100.0), (5_000.0, 150.0), (10_000.0, 200.0)],
        ssp=ssp,
    )
    segments = segment_environment_by_range(env)
    seg_ranges = [r for r, _ in segments]
    for rk in (0.0, 2_000.0, 4_000.0, 5_000.0, 6_000.0, 10_000.0):
        assert any(abs(r - rk) < 1.0 for r in seg_ranges), (
            f"missing union point {rk} in {seg_ranges}"
        )


def test_bellhop_quad_ssp_emits_unchanged_ssp_file(tmp_path):
    """Bellhop should pass ssp.ranges/.sound_speed through verbatim to .ssp,
    independent of bathymetry / receiver grids — plus one prepended
    negative-range guard column so back-scattered rays do not trip
    bellhopcuda's BHC_ERR_OUTSIDE_SSP (x < Seg.r[0]) range-box check."""
    from uacpy.io.bellhop_writer import write_bellhop_env_file

    ssp = SoundSpeedProfile.from_2d(
        depths=np.array([0.0, 100.0]),
        ranges=np.array([0.0, 1_000.0, 5_000.0]),
        matrix=np.array([[1500.0, 1495.0, 1485.0],
                         [1480.0, 1475.0, 1465.0]])
    )
    env = Environment(bathymetry=100.0, ssp=ssp)
    src = uacpy.Source(depths=10.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]),
                         ranges=np.array([2_000.0]))
    env_path = tmp_path / 'rdssp.env'
    write_bellhop_env_file(env_path, env, src, rcv, interp_ssp='quad')
    ssp_path = env_path.with_suffix('.ssp')
    assert ssp_path.exists()

    lines = ssp_path.read_text().splitlines()
    # AT/bellhopcuda LDIFile expects Npts and the range vector on
    # separate records (one line each), then one line per depth row.
    # Npts == 3 real columns + 1 prepended guard column.
    assert lines[0].strip() == '4', (
        f"line 1 must contain only Npts; got {lines[0]!r}"
    )
    ranges_km = list(map(float, lines[1].split()))
    # r_box = 1.2 * 2000 m = 2400 m -> guard at -1.1 * r_box = -2.64 km.
    assert ranges_km[0] == pytest.approx(-2.64)
    assert ranges_km[1:] == [0.0, 1.0, 5.0]
    # Two depths -> two SSP rows; the guard column duplicates the first
    # real profile, the rest pass through verbatim.
    assert len(lines) >= 4
    row0 = list(map(float, lines[2].split()))
    row1 = list(map(float, lines[3].split()))
    assert row0 == [1500.0, 1500.0, 1495.0, 1485.0]
    assert row1 == [1480.0, 1480.0, 1475.0, 1465.0]


# ──────────────────────────────────────────────────────────────────────
# Unknown run() kwargs → TypeError (no concrete ``run()`` takes
# ``**kwargs``, so Python rejects them at the call site).
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model_cls", engine_params())
def test_run_rejects_unknown_kwarg(model_cls):
    """Every concrete ``run()`` declares its full keyword set; unknown
    names raise :class:`TypeError` at the call site. Constructing the model
    resolves its binary, so this carries ``requires_binary`` (and
    ``requires_oases`` for the OASES models).
    """
    env = uacpy.Environment(name='unknown-kwarg', bathymetry=100.0, ssp=1500.0)
    src = uacpy.Source(depths=50.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=[25.0, 50.0], ranges=[1000.0, 2000.0])

    model = build_engine(model_cls.__name__)
    with pytest.raises(TypeError, match=r"unexpected keyword argument"):
        model.run(env, src, rcv, totally_bogus_kwarg=1)


def test_generate_sea_surface_rejects_nonpositive_range():
    from uacpy import generate_sea_surface
    for bad in (0.0, -100.0):
        with pytest.raises(ConfigurationError, match="rmax_m"):
            generate_sea_surface(bad)


@pytest.mark.parametrize("bad", ['deep', object(), {'a': 1}])
def test_bathymetry_nonnumeric_is_typed(bad):
    with pytest.raises(ConfigurationError, match='Bathymetry: .*non-numeric'):
        uacpy.Environment(bathymetry=bad, ssp=1500.0)


def test_altimetry_nonnumeric_is_typed():
    with pytest.raises(ConfigurationError, match='Altimetry: .*non-numeric'):
        uacpy.Environment(bathymetry=100.0, ssp=1500.0, altimetry='wavy')


@pytest.mark.requires_binary  # constructs a model (resolves its binary)
def test_surface_source_warns_for_field_runs():
    """A source at z=0 sits ON Bellhop's top boundary: bellhop.f90:488-492
    terminates every ray (DistBegTop <= 0), so the run is refused rather
    than returning an all-NaN field; a positive depth runs clean."""
    env = uacpy.Environment(
        bathymetry=100.0, ssp=1500.0,
        bottom=uacpy.BoundaryProperties(sound_speed=1600.0, density=1.5,
                                        attenuation=0.5))
    rcv = uacpy.Receiver(depths=[50.0], ranges=[2000.0])
    with pytest.raises(uacpy.ConfigurationError, match="top boundary"):
        Bellhop().compute_tl(env, uacpy.Source(depths=0.0, frequencies=200.0), rcv)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        Bellhop().compute_tl(env, uacpy.Source(depths=10.0, frequencies=200.0), rcv)


# --- Silent-coercion guards: extrapolation and container types -------------
def test_sound_speed_at_warns_on_extrapolation():
    """Depths outside the (bathymetry-extended) SSP are constant-extrapolated
    with a UserWarning, not silently."""
    ssp = SoundSpeedProfile(depths=[0, 50, 100], sound_speed=[1500, 1490, 1480])
    env = Environment(ssp=ssp, bathymetry=200.0)  # SSP extended to 200 m
    with pytest.warns(UserWarning, match="constant-extrapolated"):
        assert float(env.ssp.sound_speed_at(250)[0]) == 1480.0
    with pytest.warns(UserWarning, match="constant-extrapolated"):
        assert float(env.ssp.sound_speed_at(-10)[0]) == 1500.0
    # in-range query must not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        env.ssp.sound_speed_at(75)


def test_francois_garrison_accepts_list_pH():
    """pH as a Python list must not raise a bare TypeError (it is coerced)."""
    from uacpy.core.acoustics.attenuation import absorption_francois_garrison
    out = absorption_francois_garrison(10000, 10, 35, [8.0, 8.1], 100)
    out = np.atleast_1d(np.asarray(out, dtype=float))
    assert out.shape == (2,) and np.all(out > 0)


@pytest.mark.requires_binary  # constructs the named model (resolves its binary)
@pytest.mark.parametrize('model_name', ['Bellhop', 'Kraken', 'Scooter', 'RAM'])
def test_paired_samples_come_from_a_grid_run(model_name):
    """The Receiver docstring's recipe for paired (depth, range) samples
    runs on every model: a grid run, indexed by the pairs.
    """
    import uacpy

    env = uacpy.Environment(
        name='p', bathymetry=200.0, ssp=1500.0,
        bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                        sound_speed=1800.0, density=1.8,
                                        attenuation=0.5))
    src = uacpy.Source(depths=50.0, frequencies=100.0)
    d = np.array([60.0, 90.0, 120.0])
    r = np.array([1000.0, 2000.0, 3000.0])

    model = getattr(uacpy, model_name)(verbose=False)
    tl = np.asarray(model.run(env, src,
                              uacpy.Receiver(depths=d, ranges=r)).dB)
    i = np.arange(len(d))
    assert tl[i, i].shape == (3,)


# ── The AT solvers' own limits, enforced on the carriers ────────────────────
#
# Each guard below mirrors a specific fatal test (or a specific silent collapse)
# in the vendored Fortran. They live on the carriers rather than in a writer
# because the carrier owns the value: the same object feeds every model, and the
# ad-hoc writer guards that existed before covered one axis on one model each.


class TestHalfSpaceSoundSpeedMustBePositive:
    """``misc/ReadEnvironmentMod.f90:292`` aborts a ``'A'`` half-space whose
    compressional speed *or* density vanishes. uacpy mirrored only the density
    half, so ``sound_speed=0`` gave a Kraken ``ModelExecutionError`` (``TopBot:
    Sound speed or density vanishes in halfspace``) and Bellhop an all-NaN field
    at exit 0 — the same carrier, two different wrong outcomes."""

    def test_zero_sound_speed_on_a_half_space_is_refused(self):
        with pytest.raises(uacpy.core.exceptions.ConfigurationError,
                           match='sound_speed'):
            uacpy.core.BoundaryProperties(sound_speed=0.0, density=1.7)

    def test_a_real_half_space_is_untouched(self):
        bp = uacpy.core.BoundaryProperties(sound_speed=1600.0, density=1.7)
        assert bp.acoustic_type == 'half-space'

    @pytest.mark.parametrize('kind', ['vacuum', 'rigid'])
    def test_types_that_never_use_the_speed_are_untouched(self, kind):
        """vacuum/rigid carry placeholder speeds the deck never emits, so the
        guard must be keyed on the resolved acoustic_type, not on the field."""
        assert uacpy.core.BoundaryProperties(
            acoustic_type=kind).acoustic_type == kind


class TestAttenuationCeiling:
    """``misc/AttenMod.f90`` converts attenuation to an imaginary sound speed
    (:73 for ``AttenUnit 'W'``, which uacpy always writes, then :113) and aborts
    at :116 once that exceeds the real part. Substituting one into the other
    leaves a bound on alpha alone, independent of frequency and sound speed:
    ``8.6858896 * 2*pi``. Kraken/Scooter/OAST/Bellhop-Fortran all abort above it;
    ``bellhopcxx`` instead returns *less* loss at 1e6 dB/lambda than at 0.5."""

    def test_the_constant_is_the_one_the_fortran_implies(self):
        from uacpy.core.deck_limits import MAX_ATTENUATION_DB_PER_WAVELENGTH
        assert MAX_ATTENUATION_DB_PER_WAVELENGTH == pytest.approx(54.57505391,
                                                                 abs=1e-7)

    @pytest.mark.parametrize('alpha', [54.6, 60.0, 1e6])
    def test_above_the_ceiling_is_refused(self, alpha):
        with pytest.raises(uacpy.core.exceptions.ConfigurationError,
                           match='dB/wavelength'):
            uacpy.core.BoundaryProperties(sound_speed=1800.0, density=2.0,
                                          attenuation=alpha)

    @pytest.mark.parametrize('alpha', [0.0, 0.5, 2.0, 54.0])
    def test_below_the_ceiling_is_accepted(self, alpha):
        """54.0 must still pass: the measured Kraken/Scooter transition sits
        between 54.0 and 54.6, so a rounder bound would reject legal input."""
        assert uacpy.core.BoundaryProperties(
            sound_speed=1800.0, density=2.0, attenuation=alpha) is not None

    def test_the_shear_channel_is_bounded_too(self):
        from uacpy.core.boundary import SedimentLayer
        with pytest.raises(uacpy.core.exceptions.ConfigurationError,
                           match='shear_attenuation'):
            SedimentLayer(thickness=1.0, sound_speed=1600.0, density=1.7,
                          attenuation=0.5, shear_attenuation=100.0)

    def test_sediment_layers_are_bounded_too(self):
        from uacpy.core.boundary import SedimentLayer
        with pytest.raises(uacpy.core.exceptions.ConfigurationError,
                           match='dB/wavelength'):
            SedimentLayer(thickness=1.0, sound_speed=1600.0, density=1.7,
                          attenuation=1e6)


class TestAxesMustSurviveTheDeckPrintResolution:
    """Every carrier axis was validated at full float precision while every deck
    prints depths in metres and ranges in kilometres at ``%.6f``. Two samples
    closer than that become one token: on a range axis the readers reject it
    (``misc/sspMod.f90:342``, ``Bellhop/bdryMod.f90:132``/``:231``,
    ``misc/SourceReceiverPositions.f90:163``, with ``monotonicMod`` strict), and
    on a source/receiver depth axis the equivalent ERROUTs are commented out
    (``SourceReceiverPositions.f90:142``/``:146``) so the deck silently carries
    the same depth twice. Two of the seven axes produced silent all-NaN from
    Kraken before this."""

    def test_ssp_depths(self):
        with pytest.raises(uacpy.core.exceptions.ConfigurationError,
                           match='must increase by more than'):
            uacpy.core.SoundSpeedProfile.from_pairs(
                np.array([[0.0, 1500.0], [1e-7, 1500.0], [100.0, 1500.0]]))

    def test_ssp_ranges(self):
        with pytest.raises(uacpy.core.exceptions.ConfigurationError,
                           match='must increase by more than'):
            uacpy.core.SoundSpeedProfile(
                depths=[0.0, 100.0], sound_speed=np.full((2, 3), 1500.0),
                ranges=[0.0, 1e-7, 5000.0])

    def test_bathymetry_ranges(self):
        with pytest.raises(uacpy.core.exceptions.ConfigurationError,
                           match='must increase by more than'):
            uacpy.core.bathymetry.Bathymetry(
                depths=[100.0, 100.0, 120.0], ranges=[0.0, 1e-7, 5000.0])

    def test_altimetry_ranges(self):
        from uacpy.core.altimetry import Altimetry
        with pytest.raises(uacpy.core.exceptions.ConfigurationError,
                           match='must increase by more than'):
            Altimetry(heights=[0.0, 0.0, 0.0], ranges=[0.0, 1e-7, 5000.0])

    def test_source_depths(self):
        with pytest.raises(uacpy.core.exceptions.ConfigurationError,
                           match='must increase by more than'):
            uacpy.core.Source(depths=[25.0, 25.0 + 1e-7], frequencies=200.0)

    def test_receiver_ranges(self):
        with pytest.raises(uacpy.core.exceptions.ConfigurationError,
                           match='must increase by more than'):
            uacpy.core.Receiver(depths=50.0, ranges=[1000.0, 1000.0 + 1e-7])

    def test_receiver_depths(self):
        with pytest.raises(uacpy.core.exceptions.ConfigurationError,
                           match='must increase by more than'):
            uacpy.core.Receiver(depths=[50.0, 50.0 + 1e-7], ranges=1000.0)

    def test_bottom_ranges(self):
        from uacpy.core.bottom import Bottom, SeabedColumn
        hs = uacpy.core.BoundaryProperties(sound_speed=1600.0, density=1.7)
        col = SeabedColumn(layers=[], halfspace=hs)
        with pytest.raises(uacpy.core.exceptions.ConfigurationError,
                           match='must increase by more than'):
            Bottom(columns=[col, col, col], ranges=[0.0, 1e-7, 5000.0])

    def test_the_resolutions_are_the_ones_the_decks_print(self):
        """1 µm on a depth axis (metres at %.6f) and 1 mm on a range axis
        (kilometres at %.6f)."""
        from uacpy.core.deck_limits import (
            DECK_DEPTH_RESOLUTION_M, DECK_RANGE_RESOLUTION_M,
        )
        assert DECK_DEPTH_RESOLUTION_M == 1e-6
        assert DECK_RANGE_RESOLUTION_M == 1e-3

    def test_the_writer_prints_at_the_resolution_the_carriers_admit(self):
        """The link the two numbers rest on, driven rather than restated.

        The carriers admit a step of ``DECK_DEPTH_RESOLUTION_M`` because the
        deck can print it; the writer's column format is what decides whether
        it can. Both sides of that threshold: one resolution apart must give
        two tokens, and anything below it must give one."""
        from uacpy.core.deck_limits import (
            DECK_AXIS_DECIMALS, DECK_DEPTH_RESOLUTION_M,
            DECK_RANGE_RESOLUTION_M,
        )
        from uacpy.core.deck_limits import DECK_DEPTH_FMT
        from uacpy.core.units import m_to_km

        def token(value, fmt):
            return format(value, fmt)

        depth = 100.0
        assert (token(depth, DECK_DEPTH_FMT)
                != token(depth + DECK_DEPTH_RESOLUTION_M, DECK_DEPTH_FMT))
        assert (token(depth, DECK_DEPTH_FMT)
                == token(depth + DECK_DEPTH_RESOLUTION_M / 10.0,
                         DECK_DEPTH_FMT))

        range_fmt = f'.{DECK_AXIS_DECIMALS}f'
        r = 4000.0
        assert (token(float(m_to_km(r)), range_fmt)
                != token(float(m_to_km(r + DECK_RANGE_RESOLUTION_M)),
                         range_fmt))
        assert (token(float(m_to_km(r)), range_fmt)
                == token(float(m_to_km(r + DECK_RANGE_RESOLUTION_M / 10.0)),
                         range_fmt))


    @pytest.mark.parametrize('axis', ['depths', 'ranges'])
    def test_ordinary_axes_pass(self, axis):
        kw = {'depths': 50.0, 'ranges': np.linspace(500.0, 5000.0, 10)}
        if axis == 'depths':
            kw['depths'] = np.linspace(10.0, 90.0, 9)
        assert uacpy.core.Receiver(**kw) is not None


class TestExtendToUsesTheReadersOwnEpsilon:
    """``misc/sspMod.f90:353`` ends a medium's SSP block at the first sample
    within ``100*EPSILON(1.0e0)`` = 1.1920929e-05 m of the declared medium depth.
    ``extend_to``'s own tolerance was ``rtol=atol=1e-9`` — 277× tighter at 42 m —
    so for any target in between it *appended* a second terminal row that the
    reader never reads as SSP: the next READ takes it as the bottom-option
    record, so ``BotOpt(1:1)`` became ``'4'`` and sigma became 1500 m. Kraken,
    Bellhop and Scooter all aborted, naming the boundary condition."""

    @staticmethod
    def _profile(last):
        return uacpy.core.SoundSpeedProfile.from_pairs(
            np.array([[0.0, 1500.0], [last, 1500.0]]))

    @pytest.mark.parametrize('delta', [5.1e-7, 1e-6, 5e-6, 1.1e-5])
    def test_a_target_inside_the_epsilon_never_appends(self, delta):
        """The defect was the *appended* row, so this is the assertion that
        matters at every delta inside the reader's window."""
        base = 42.299996
        out = self._profile(base).extend_to(base + delta)
        assert np.asarray(out.depths).size == 2, (
            f"appended a row {delta:g} m from the last sample — inside the "
            f"reader's 1.1920929e-05 m window, so it is never read as SSP and "
            f"the next READ takes it as the bottom-option record")

    @pytest.mark.parametrize('delta', [5.1e-7, 1e-6, 5e-6, 1.1e-5])
    def test_a_real_target_inside_the_window_moves_the_sample(self, delta):
        """Anything past float round-trip noise and inside the reader's window is
        a real request, so the last sample is moved onto it rather than left
        beside it — leaving it beside it is the defect."""
        base = 42.299996
        out = self._profile(base).extend_to(base + delta)
        assert float(np.asarray(out.depths)[-1]) == pytest.approx(
            base + delta, abs=1e-12)

    @pytest.mark.parametrize('delta', [0.0, 1e-12, 1e-9])
    def test_float_round_trip_noise_is_the_same_object(self, delta):
        """``env.depth`` round-tripped through I/O can be a few ulps off; that is
        the same request, not a re-alignment."""
        base = 42.299996
        ssp = self._profile(base)
        assert ssp.extend_to(base + delta) is ssp

    @pytest.mark.parametrize('delta', [2e-5, 1e-3, 6e-2])
    def test_a_target_outside_the_epsilon_appends(self, delta):
        base = 42.299996
        out = self._profile(base).extend_to(base + delta)
        assert np.asarray(out.depths).size == 3
        assert float(np.asarray(out.depths)[-1]) == pytest.approx(base + delta)

    def test_the_epsilon_is_the_fortran_one(self):
        from uacpy.core.deck_limits import AT_LAST_SSP_POINT_EPS_M
        assert AT_LAST_SSP_POINT_EPS_M == pytest.approx(1.1920929e-05, rel=1e-9)

    def test_a_snap_that_would_cross_the_previous_sample_raises(self):
        """The reader's window (1.19e-5 m) is wider than the legal minimum
        depth step (1e-6 m), so a downward snap can land the moved sample at
        or below ``depths[-2]``. The rebuilt profile goes through axis
        validation, so that case is a typed error — never a silently
        non-increasing depth axis."""
        ssp = uacpy.core.SoundSpeedProfile.from_pairs(
            np.array([[0.0, 1500.0], [10.0, 1500.0], [10.000002, 1500.0]]))
        with pytest.raises(ConfigurationError, match='strictly increasing'):
            ssp.extend_to(9.999999)

    def test_a_snap_inside_the_window_returns_a_valid_profile(self):
        ssp = uacpy.core.SoundSpeedProfile.from_pairs(
            np.array([[0.0, 1500.0], [10.0, 1500.0], [10.000002, 1500.0]]))
        out = ssp.extend_to(10.00001)
        assert np.all(np.diff(out.depths) > 0)
        assert float(out.depths[-1]) == pytest.approx(10.00001, abs=1e-12)
        assert out.n_depths == 3

    def test_an_exact_match_is_a_no_op(self):
        base = 42.299996
        out = self._profile(base).extend_to(base)
        assert np.asarray(out.depths).size == 2
        assert float(np.asarray(out.depths)[-1]) == pytest.approx(base)

    @pytest.mark.requires_binary
    @pytest.mark.parametrize('depth', [87.499995, 1234.599995, 42.299996])
    def test_depths_inside_the_readers_window_run_end_to_end(self, depth):
        """These arrive naturally from a fetched bathymetry — about 1 in 8400
        arbitrary depths lands in the window."""
        from uacpy.models import Kraken, Scooter
        env = uacpy.core.Environment(
            name='eps', bathymetry=depth, ssp=1500.0,
            bottom=uacpy.core.BoundaryProperties(sound_speed=1800.0,
                                                 density=2.0, attenuation=0.5))
        src = uacpy.core.Source(depths=0.25 * depth, frequencies=200.0)
        rcv = uacpy.core.Receiver(depths=0.5 * depth, ranges=[1000.0, 2000.0])
        for model in (Kraken(verbose=False), Scooter(verbose=False)):
            tl = np.asarray(model.run(env, src, rcv).dB)
            assert np.isfinite(tl).all(), f"{model.model_name} returned {tl}"


# --- Silent-all-zero guards added by the 2026-08 audit ----------------------
@pytest.mark.requires_binary  # constructs SPARC (resolves its binary)
def test_sparc_refuses_empty_wavenumber_loop():
    """A near-CW band at tank-scale range gives sparc.f90:116 Nk <= 0: the
    march loop runs empty and every output would come back all-zero at
    exit 0 — the wrapper must refuse the deck instead."""
    from uacpy.models.sparc import SPARC
    env = uacpy.Environment(
        bathymetry=100.0, ssp=1500.0,
        bottom=uacpy.BoundaryProperties(acoustic_type='rigid'))
    src = uacpy.Source(depths=25.0, frequencies=50.0)
    rcv = uacpy.Receiver(depths=[50.0], ranges=[2.0, 4.0])
    model = SPARC(freq_min=49.0, freq_max=50.0, c_low=1400.0, c_high=1600.0,
                  rmax_factor=1.0)
    with pytest.raises(ConfigurationError, match="wavenumber"):
        model.run(env, src, rcv)


@pytest.mark.requires_binary  # constructs SPARC (resolves its binary)
def test_sparc_refuses_inverted_pulse_band():
    """freq_min >= freq_max makes Nk negative through the sparc.f90:116 formula
    ``Nk = INT(1000·RMax·(kMax − kMin)/2π)``; the constructor names the cause
    instead. A negative freq_min is refused for the same reason.

    ``freq_min = 0`` is *not* refused: ``sparc.f90:114`` clamps the resulting
    ``kMin`` to 1e-20 ("avoid a zero that would produce a divide check when
    the phase speed is written"), so the binary runs that deck.
    """
    from uacpy.models.sparc import SPARC
    with pytest.raises(ConfigurationError, match="freq_min < freq_max"):
        SPARC(freq_min=200.0, freq_max=50.0)
    with pytest.raises(ConfigurationError, match="freq_min >= 0"):
        SPARC(freq_min=-1.0)
    assert SPARC(freq_min=0.0).freq_min == 0.0


@pytest.mark.requires_binary
@pytest.mark.requires_oases
def test_oassp_rejects_complex_contour_options():
    """A custom options string that leaves the complex frequency contour on
    ('O', or no 'J' under automatic sampling — unoassp30.f:285-290, :983,
    :382-385) writes a spectrum the time-series synthesis cannot undo."""
    from uacpy.models.oases import OASSP
    from uacpy.models.oases.oassp import _reject_unreadable_oassp_options

    def check(model):
        _reject_unreadable_oassp_options(model.options, model.n_wavenumbers)

    with pytest.raises(ConfigurationError, match="'O'"):
        check(OASSP(correlation_length=5.0, options='N J s O'))
    with pytest.raises(ConfigurationError, match="'J'"):
        check(OASSP(correlation_length=5.0, options='N s'))
    # The typed-flag default path always carries 'J' and never 'O'.
    check(OASSP(correlation_length=5.0))


@pytest.mark.requires_binary  # constructs RAM (resolves its binary)
def test_ram_broadband_grid_round_trips_the_request():
    """The (fc, Q, T) inversion must reproduce every requested bin: odd
    counts exactly, even counts as a superset (one extra bin at freq_max+Δf) —
    never the old lose-a-bin-and-shift-by-Δf/2 behaviour. Pinning both Q and
    T alongside a frequency array is a contradiction and raises."""
    from uacpy.models.ram import RAM as _RAM
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in (8, 9, 2, 16):
            req = np.linspace(100.0, 100.0 + 5.0 * (n - 1), n)
            model = _RAM()
            fc, q, t = ram_band.resolve_broadband_grid(
                uacpy.Source(depths=25.0, frequencies=req),
                knobs=model._knob_record(), log=model._log)
            marched = ram_band.broadband_frequencies(fc, q, t)
            for f in req:
                assert np.isclose(marched, f).any(), (n, f, marched)
            assert marched.size <= req.size + 1
    with pytest.warns(UserWarning, match="pinned"):
        ram = RAM(q_factor=2.0, record_duration=10.0)
        ram_band.resolve_broadband_grid(
            uacpy.Source(depths=25.0, frequencies=np.linspace(100, 110, 11)),
            knobs=ram._knob_record(), log=ram._log)


@pytest.mark.requires_binary  # constructs Kraken (resolves its binary)
def test_kraken_leaky_modes_keeps_c_high_verbatim():
    """leaky_modes keeps self.c_high as given (the copy()/repr() verbatim-
    storage invariant): the leaky window, 10 x the fastest speed in the
    profile, is resolved into the run's settings, and a pinned c_high beside
    it raises."""
    from uacpy.models.kraken import Kraken
    with pytest.raises(ConfigurationError, match="leaky_modes"):
        Kraken(c_high=1700.0, leaky_modes=True)
    env = uacpy.Environment(name='lk', bathymetry=100.0, ssp=1500.0,
                            bottom=uacpy.BoundaryProperties(
                                acoustic_type='half-space',
                                sound_speed=1600.0, density=1.5,
                                attenuation=0.5))
    src = uacpy.Source(depths=50.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=[50.0], ranges=[1000.0])
    m = Kraken(leaky_modes=True)
    assert m.c_high is None
    assert m.run_settings(env, src, rcv).engine.launches[0].c_high == (
        pytest.approx(16000.0),)
    clone = m.copy(leaky_modes=False)
    assert clone.c_high is None
    assert clone.run_settings(env, src, rcv).engine.launches[0].c_high == (
        pytest.approx(1680.0),)


@pytest.mark.requires_binary  # constructs Kraken (resolves its binary)
def test_kraken_top_reflection_file_carries_roughness_into_the_drop(tmp_path):
    """The tabulated-top rewrite replaces only the boundary condition, and
    carries the caller's roughness forward far enough for the tabulated-top
    drop to name the value that was set.

    ``Kraken/kraken.f90:864-866`` zeroes ``rho1``/``eta1Sq`` for a tabulated
    top, so ``Kraken/Scattering.f90:23`` is false and the roughness in
    ``SSP%sigma(1)`` never reaches the eigenvalue perturbation; the value is
    reported and dropped instead of being written into a deck that discards
    it. A rewrite that clobbered the roughness before the drop ran would warn
    about 0 m, or not warn at all."""
    from uacpy.models.kraken import Kraken
    from uacpy.core.surface import Surface
    trc = tmp_path / 'top.trc'
    trc.write_text('3\n0.0 1.0 180.0\n45.0 0.9 170.0\n90.0 0.8 160.0\n')
    env = uacpy.Environment(
        bathymetry=100.0, ssp=1500.0,
        bottom=uacpy.BoundaryProperties(sound_speed=1600.0, density=1.5,
                                        attenuation=0.5),
        surface=Surface(nodes=[uacpy.BoundaryProperties(
            acoustic_type='vacuum', roughness=1.5)]))
    with pytest.warns(UserWarning, match=r'roughness=1\.5 m was dropped'):
        projected = Kraken(top_reflection_file=trc)._project_environment(env)
    assert projected.surface.nodes[0].acoustic_type == 'file'
    assert float(projected.surface.roughness) == 0.0


# --- shared carrier validators (uacpy/core/_validate.py) -------------------

class TestValidatorMessagesStayBounded:
    """The shared validators report the first offending element (value, flat
    index) and the axis length, never the whole array — so the exception for
    a large axis stays a single readable line."""

    def test_finite_reports_first_offender_not_the_array(self):
        from uacpy.core._validate import require_finite
        big = np.arange(50000.0)
        big[123] = np.nan
        with pytest.raises(ConfigurationError, match="must be finite") as exc:
            require_finite(big, "X")
        msg = str(exc.value)
        assert len(msg) < 500
        assert "index 123" in msg
        assert "50000" in msg

    def test_positive_reports_first_offender(self):
        from uacpy.core._validate import require_positive
        vals = np.ones(10000)
        vals[7] = -2.0
        with pytest.raises(ConfigurationError, match="must be positive") as exc:
            require_positive(vals, "X")
        msg = str(exc.value)
        assert len(msg) < 500
        assert "-2" in msg and "index 7" in msg

    def test_non_negative_reports_first_offender(self):
        from uacpy.core._validate import require_non_negative
        vals = np.zeros(10000)
        vals[42] = -1.5
        with pytest.raises(ConfigurationError,
                           match="must be non-negative") as exc:
            require_non_negative(vals, "X")
        msg = str(exc.value)
        assert len(msg) < 500
        assert "-1.5" in msg and "index 42" in msg

    def test_strictly_increasing_reports_pair_and_length(self):
        from uacpy.core._validate import require_strictly_increasing
        axis = np.arange(10000.0)
        axis[500] = 0.0
        with pytest.raises(ConfigurationError,
                           match="strictly increasing") as exc:
            require_strictly_increasing(axis, "X")
        msg = str(exc.value)
        assert len(msg) < 500
        assert "axis length 10000" in msg


class TestTheRealSignalGuard:
    """``require_real_signal`` refuses complex input before the float cast
    that would drop its imaginary part, then a wrong dimension, then an empty
    or non-finite signal; a real signal of the asked dimension comes back as
    float."""

    def test_complex_input_is_refused_with_the_callers_reason(self):
        from uacpy.core._validate import require_real_signal
        with pytest.raises(ConfigurationError,
                           match=r"f: data must be real \(got complex "
                                 r"input\); because") as exc:
            require_real_signal(np.array([1.0, 1j]), "f", why="; because",
                                remediation="Do this.")
        assert 'Do this.' in str(exc.value)

    @pytest.mark.parametrize('ndim, shape', [(1, (2, 3)), (2, (6,))])
    def test_a_wrong_dimension_is_refused(self, ndim, shape):
        from uacpy.core._validate import require_real_signal
        with pytest.raises(ConfigurationError,
                           match=rf"must be {ndim}-D \(nt\); got shape"):
            require_real_signal(np.ones(shape), "f", ndim=ndim,
                                shape_hint=" (nt)")

    def test_a_nan_is_refused_and_a_real_signal_returns_as_float(self):
        from uacpy.core._validate import require_real_signal
        with pytest.raises(ConfigurationError, match="NaN or Inf"):
            require_real_signal(np.array([1.0, np.nan]), "f")
        out = require_real_signal(np.array([1, 2, 3]), "f")
        assert out.dtype == float and out.tolist() == [1.0, 2.0, 3.0]


class TestTheAxisGuard:
    """``normalize_axis`` returns the non-negative axis index, on both sides of
    each end of the array's axes, and refuses a non-integer."""

    @pytest.mark.parametrize('axis, index', [(-3, 0), (-1, 2), (0, 0), (2, 2)])
    def test_every_axis_of_the_array_is_admitted(self, axis, index):
        from uacpy.core._validate import normalize_axis
        assert normalize_axis(np.ones((2, 3, 4)), axis, "f") == index

    @pytest.mark.parametrize('axis', [-4, 3])
    def test_an_axis_past_either_end_is_refused(self, axis):
        from uacpy.core._validate import normalize_axis
        with pytest.raises(ConfigurationError,
                           match=rf"f: axis={axis} is not an axis of an array "
                                 r"with shape \(2, 3, 4\)"):
            normalize_axis(np.ones((2, 3, 4)), axis, "f")

    def test_a_non_integer_axis_is_refused(self):
        from uacpy.core._validate import normalize_axis
        with pytest.raises(ConfigurationError,
                           match="f: axis must be an integer; got 'x'"):
            normalize_axis(np.ones(3), 'x', "f")


class TestAnEmptyAxisIsRefused:
    """``_require_strictly_increasing`` guards every range / depth axis that
    a deck writer indexes and an interpolator samples, and both read the axis
    positionally: a zero-length one reaches them as a bare ``IndexError`` or
    zero-size reduction. One sample is a legitimate axis and stays accepted.
    """

    def test_zero_samples_are_refused_and_one_sample_is_accepted(self):
        from uacpy.core._validate import require_strictly_increasing
        with pytest.raises(ConfigurationError,
                           match="at least one value") as exc:
            require_strictly_increasing(np.array([]), "X.ranges")
        assert "X.ranges" in exc.value.remediation
        require_strictly_increasing(np.array([7.0]), "X.ranges")

    def test_an_ssp_range_axis_of_zero_columns_is_refused(self):
        """Measured entry point: this profile carries no sound speeds at all.
        It reaches ``write_bellhop_env_file``'s ``ssp_matrix[:, 0]`` as an
        ``IndexError`` naming no input the caller passed."""
        with pytest.raises(ConfigurationError, match="at least one value"):
            SoundSpeedProfile(depths=np.array([0.0, 200.0]),
                              sound_speed=np.zeros((2, 0)), ranges=np.array([]))

    @pytest.mark.parametrize('build', [
        lambda: uacpy.Receiver(depths=[], ranges=[1000.0]),
        lambda: uacpy.Receiver(depths=[10.0], ranges=[]),
        lambda: uacpy.Source(depths=[], frequencies=[100.0]),
        lambda: SoundSpeedProfile(depths=np.array([]), sound_speed=np.array([])),
        lambda: SoundSpeedProfile(depths=np.array([0.0, 200.0]),
                                  sound_speed=np.zeros((2, 0)), ranges=np.array([])),
        lambda: Bottom(columns=[SeabedColumn(
            layers=[], halfspace=BoundaryProperties(
                acoustic_type='half-space', sound_speed=1600.0,
                density=1.8, attenuation=0.5))], ranges=np.array([])),
        lambda: uacpy.Bathymetry(ranges=np.array([]), depths=np.array([])),
        lambda: uacpy.Altimetry(ranges=np.array([]), heights=np.array([])),
    ])
    def test_every_axis_carrier_refuses_an_empty_axis(self, build):
        with pytest.raises(ConfigurationError, match='at least one'):
            build()

    def test_the_value_predicates_accept_an_empty_array(self):
        """The asymmetry is deliberate: ``_require_finite`` also guards
        ``Field.coords``, where an axis sliced to nothing is a supported
        state that ``Field.max`` reports in those terms."""
        from uacpy.core._validate import (
            require_finite, require_positive, require_non_negative,
        )
        empty = np.array([])
        require_finite(empty, "X")
        require_positive(empty, "X")
        require_non_negative(empty, "X")
        field = uacpy.Field(data=np.zeros((0,), dtype=complex),
                            coords={'range': empty})
        assert field.coords['range'].size == 0
        with pytest.raises(ConfigurationError, match="sliced to nothing"):
            field.max()


def test_coerce_data_sources_none_is_empty_provenance():
    """``data_sources=None`` means no provenance and coerces to ``()``,
    like an empty sequence."""
    from uacpy.core._provenance import coerce_data_sources
    assert coerce_data_sources(None, "X") == ()
    assert coerce_data_sources((), "X") == ()


# --- base.py validate_inputs funnel: frequency / depth / surface guards ----
#
# Every concrete ``run()`` calls ``validate_inputs`` (Bounce overrides
# ``_validate_geometry`` to a no-op — reflection decks read no geometry — so
# it appears below only where its own guards fire). The tests call
# ``validate_inputs`` directly where the guard lives there, and ``run()``
# where the guard fires later on the run path but still before any deck is
# written or binary launched.


def _guard_env():
    """Scalar Pekeris env with geoacoustic half-space (RAM's own
    validate_inputs refuses vacuum/rigid/tabulated bottoms)."""
    return make_pekeris(name='guards', density=1.7)


def _guard_rcv():
    return uacpy.Receiver(depths=[50.0], ranges=[1000.0])


def _guard_env_for(model_cls):
    """:func:`_guard_env`, with a rough seabed for OASS and OASSP: both
    refuse a smooth one in their checking stage, since the mean field would
    write an empty ``.rhs``; and a rigid floor for SPARC, whose checking
    stage refuses a half-space (its deck carries only vacuum and rigid
    bottoms)."""
    # OASS/OASSP arrive as partials carrying their correlation_length.
    name = getattr(model_cls, 'func', model_cls).__name__
    if name == 'SPARC':
        return uacpy.Environment(
            name='guards-rigid', bathymetry=100.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(acoustic_type='rigid'))
    if name not in ('OASS', 'OASSP'):
        return _guard_env()
    return uacpy.Environment(
        name='guards-rough', bathymetry=100.0, ssp=1500.0,
        bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                        sound_speed=1700.0, density=1.7,
                                        attenuation=0.5, roughness=0.5))


def _model_mode_params(entries):
    """(model class, run mode) pytest params carrying the binary marks.

    Constructing any model resolves (and existence-checks) its binary, so
    every param carries ``requires_binary``; the OASES family adds
    ``requires_oases``.
    """
    from uacpy.models.bellhop import Bellhop
    from uacpy.models.bounce import Bounce
    from uacpy.models.kraken import Kraken
    from uacpy.models.oases import OAST, OASN, OASP, OASR, OASS, OASSP
    from uacpy.models.ram import RAM
    from uacpy.models.scooter import Scooter
    from uacpy.models.sparc import SPARC
    classes = {cls.__name__: cls for cls in (
        Bellhop, Bounce, Kraken, OAST, OASN, OASP, OASR, OASS, OASSP,
        RAM, Scooter, SPARC)}
    params = []
    for name, mode in entries:
        marks = [pytest.mark.requires_binary]
        if name.startswith('OAS'):
            marks.append(pytest.mark.requires_oases)
        ctor = classes[name]
        if name in ('OASS', 'OASSP'):
            # Both require the roughness correlation length at construction;
            # without it the ctor guard fires before the guards under test.
            ctor = functools.partial(ctor, correlation_length=10.0)
        mode_tag = '' if mode is None else f'-{mode.name}'
        params.append(pytest.param(ctor, mode,
                                   id=f'{name}{mode_tag}', marks=marks))
    return params


@pytest.mark.parametrize('model_cls,mode', _model_mode_params([
    ('Bellhop', uacpy.RunMode.COHERENT_TL),
    ('Kraken', uacpy.RunMode.COHERENT_TL),
    ('Kraken', uacpy.RunMode.MODES),
    ('Scooter', uacpy.RunMode.COHERENT_TL),
    ('RAM', uacpy.RunMode.COHERENT_TL),
    ('OASP', uacpy.RunMode.COHERENT_TL),
]))
def test_single_frequency_mode_refuses_multi_frequency_source(model_cls,
                                                              mode):
    """A multi-frequency Source passed to a mode in
    ``spec.traits.single_frequency_modes`` raises, pointing at BROADBAND/TIME_SERIES.

    Not covered here because the base guard genuinely does not apply:
    SPARC (TIME_SERIES only), Bounce (REFLECTION stays out of the set;
    its own run()-level guard is pinned below) and OASR/OASN/OASS/OASSP,
    whose modes (REFLECTION/COVARIANCE/REPLICA/REVERBERATION/BROADBAND)
    sweep multiple frequencies by design. OAST is refused by the same rule
    but phrases it itself (``_multi_frequency_refusal`` names OASP, since
    it has no BROADBAND mode): ``test_oases.py::TestOastRefusesAFrequencySweep``.
    """
    src = uacpy.Source(depths=10.0, frequencies=[100.0, 200.0])
    with pytest.raises(ConfigurationError, match='single source frequency'):
        model_cls().validate_inputs(_guard_env(), src, _guard_rcv(),
                                    run_mode=mode)


@pytest.mark.requires_binary  # constructs Bounce (resolves its binary)
def test_bounce_run_refuses_multi_frequency_source():
    """``RunMode.REFLECTION`` stays out of the single-frequency modes (OASR
    does sweep), so Bounce guards multi-frequency itself in ``run()`` — with
    its own message, before any deck is written."""
    from uacpy.models.bounce import Bounce
    src = uacpy.Source(depths=10.0, frequencies=[100.0, 200.0])
    with pytest.raises(ConfigurationError,
                       match='reflection coefficient at one'):
        Bounce().run(_guard_env(), src, _guard_rcv())


@pytest.mark.parametrize('model_cls,mode', _model_mode_params([
    ('OASN', uacpy.RunMode.COVARIANCE),
    ('OASN', uacpy.RunMode.REPLICA), ('OASR', uacpy.RunMode.REFLECTION),
    ('OASS', uacpy.RunMode.REVERBERATION), ('OASS', uacpy.RunMode.COVARIANCE),
]))
def test_multi_depth_source_refused_in_a_non_field_mode(model_cls, mode):
    """Reflection tables and array products have no per-source linear sum
    (Kraken's mode set needs none), so a multi-depth Source in one of these
    modes raises
    'single source depth' from ``_validate_geometry`` and names the field
    modes that do stack. Bellhop declares ``multi_source_depth`` and is
    excluded; Bounce's geometry validation is a no-op, so the guard
    genuinely does not exist for it."""
    src = uacpy.Source(depths=[10.0, 20.0], frequencies=100.0)
    with pytest.raises(ConfigurationError,
                       match=f'single source depth per {mode.name} run'):
        model_cls().validate_inputs(_guard_env(), src, _guard_rcv(),
                                    run_mode=mode)


def test_a_rectangular_grid_over_a_slope_logs_its_buried_cells(capsys):
    """Seafloor 50 / 100 / 150 m: 80 and 120 m sit below it at 0 m, 120 m
    at 1 km — three points, but every range keeps a receiver in the water,
    so the grid is drawn as asked and the count is an info line."""
    from uacpy.models._checks import check_per_range_receiver_depth
    env = uacpy.Environment(bathymetry=[(0.0, 50.0), (2000.0, 150.0)],
                            bottom='sand')
    receiver = uacpy.Receiver(depths=[40.0, 80.0, 120.0],
                              ranges=[0.0, 1000.0, 2000.0])
    with recorded_warnings() as record:
        check_per_range_receiver_depth('Kraken', env, receiver,
                                       paired=False, verbose='info')
    assert record == []
    assert ('3 receiver point(s) sit below the local seafloor'
            in capsys.readouterr().out)


@pytest.mark.parametrize('shallowest, warns', [(50.0, False), (50.1, True)])
def test_a_range_with_every_receiver_under_the_seafloor_warns(shallowest,
                                                              warns):
    """The 0 m column is buried once its shallowest receiver passes the 50 m
    seafloor there; the deeper columns keep receivers in the water."""
    from uacpy.models._checks import check_per_range_receiver_depth
    env = uacpy.Environment(bathymetry=[(0.0, 50.0), (2000.0, 150.0)],
                            bottom='sand')
    receiver = uacpy.Receiver(depths=[shallowest, 120.0],
                              ranges=[0.0, 1000.0, 2000.0])
    with recorded_warnings() as record:
        check_per_range_receiver_depth('Kraken', env, receiver,
                                       paired=False)
    messages = [str(w.message) for w in record]
    if warns:
        (message,) = messages
        assert ('1 receiver range(s) lie entirely below the local seafloor, '
                'the first at range=0.0 m') in message
    else:
        assert messages == []


@pytest.mark.parametrize('first_depth, warns', [(0.0, False), (0.5, True)])
def test_a_profile_starting_under_the_surface_is_announced(first_depth,
                                                           warns):
    from uacpy.models._checks import warn_on_ssp_start
    env = uacpy.Environment(bathymetry=100.0, bottom='sand',
                            ssp=[(first_depth, 1500.0), (100.0, 1490.0)])
    with recorded_warnings() as record:
        warn_on_ssp_start('Kraken', env)
    messages = [str(w.message) for w in record]
    if warns:
        (message,) = messages
        assert 'starts at 0.5 m, not at the sea surface' in message
        assert '(1500 m/s) is held up to z = 0' in message
    else:
        assert messages == []


@pytest.mark.parametrize('first_range, warns', [(0.0, False), (0.5, True)])
def test_a_range_axis_starting_past_the_source_is_announced(first_range,
                                                            warns):
    from uacpy.models._checks import warn_on_range_coverage
    env = uacpy.Environment(
        bathymetry=[(first_range, 100.0), (5000.0, 200.0)], bottom='sand')
    receiver = uacpy.Receiver(depths=50.0, ranges=[1000.0, 5000.0])
    with recorded_warnings() as record:
        warn_on_range_coverage('Kraken', env, receiver)
    messages = [str(w.message) for w in record if 'past the source' in
                str(w.message)]
    if warns:
        (message,) = messages
        assert 'env.bathymetry starts at 0.5 m' in message
    else:
        assert messages == []


@pytest.mark.requires_binary  # constructs Kraken (resolves its binary)
@pytest.mark.parametrize('run_mode, refusal', [
    ('coherent_tl', None),
    (uacpy.RunMode.COHERENT_TL, None),
    ('COHERENT_TL', "'COHERENT_TL' is the name of RunMode.COHERENT_TL"),
    ('cohernt_tl', "run_mode='cohernt_tl' must be a RunMode member or its "
                   "string value"),
])
def test_a_run_mode_string_is_a_value_or_a_configuration_error(run_mode,
                                                               refusal):
    from uacpy.models import Kraken
    args = (_guard_env(), uacpy.Source(depths=25.0, frequencies=100.0),
            _guard_rcv())
    if refusal is None:
        Kraken().validate_inputs(*args, run_mode=run_mode)
    else:
        with pytest.raises(ConfigurationError, match=re.escape(refusal)):
            Kraken().validate_inputs(*args, run_mode=run_mode)


@pytest.mark.requires_binary  # runs Kraken
def test_kraken_modes_returns_one_mode_set_for_a_multi_depth_source():
    """The mode set does not depend on the source depth: one ``Modes``,
    not a stack, whose ``source_depths`` lists every depth."""
    from uacpy.core.results import Modes
    from uacpy.models import Kraken
    src = uacpy.Source(depths=[20.0, 60.0], frequencies=200.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        modes = Kraken().run(_guard_env(), src, _guard_rcv(),
                             run_mode=uacpy.RunMode.MODES)
    assert type(modes) is Modes
    np.testing.assert_array_equal(modes.source_depths, [20.0, 60.0])


@pytest.mark.parametrize('model_cls,mode', _model_mode_params([
    ('Kraken', None), ('Scooter', None), ('SPARC', None), ('RAM', None),
    ('OAST', None), ('OASP', None), ('OASSP', None),
    ('Kraken', uacpy.RunMode.BROADBAND), ('RAM', uacpy.RunMode.TIME_SERIES),
]))
def test_multi_depth_source_accepted_in_a_field_mode(model_cls, mode):
    """In a field mode ``run()`` splits the depths into one run each
    (``_stacking.run_per_source_depth``), so validation of the
    whole Source passes; the refusal above is the other side of the same
    ``_FIELD_MODES`` test."""
    src = uacpy.Source(depths=[10.0, 20.0], frequencies=100.0)
    kw = {}
    if mode == uacpy.RunMode.TIME_SERIES:
        # validate_inputs refuses a TIME_SERIES call without its pulse, as
        # run() does.
        kw = dict(source_waveform=np.hanning(40), sample_rate=400.0)
    model_cls().validate_inputs(_guard_env_for(model_cls), src, _guard_rcv(),
                                run_mode=mode, **kw)


def test_bounce_rejects_quad_interp_at_construction():
    """BOUNCE decks carry no water column, so there is no .ssp file for the
    'quad' scheme to read; the constructor refuses it with the same exception
    type the other AT wrappers raise for 'quad'. The guard precedes binary
    resolution in ``__init__``, so no binary is needed."""
    from uacpy.models.bounce import Bounce
    with pytest.raises(UnsupportedFeatureError, match="'quad'"):
        Bounce(interp_ssp='quad')


@pytest.mark.parametrize('model_cls,mode', _model_mode_params([
    ('Scooter', None), ('SPARC', None),
]))
def test_quad_interp_refused_before_launch(model_cls, mode):
    """``reject_unsupported_ssp_interp`` fires on the run() path right after
    ``validate_inputs`` — before any deck is written — because the shared AT
    reader (``misc/sspMod.f90:61-89``) has no 'Q' code; it is Bellhop-only.
    Kraken's equivalent guard is pinned in test_kraken.py."""
    src = uacpy.Source(depths=10.0, frequencies=100.0)
    # SPARC refuses a half-space bottom first, so it gets a rigid one.
    env = (_guard_env() if model_cls.__name__ != 'SPARC'
           else uacpy.Environment(
               name='guards-rigid', bathymetry=100.0, ssp=1500.0,
               bottom=uacpy.BoundaryProperties(acoustic_type='rigid')))
    with pytest.raises(UnsupportedFeatureError, match='Bellhop-only'):
        model_cls(interp_ssp='quad').run(env, src, _guard_rcv())


@pytest.mark.parametrize('model_cls,mode', _model_mode_params([
    ('Kraken', uacpy.RunMode.COHERENT_TL),
    ('Scooter', uacpy.RunMode.COHERENT_TL),
    ('SPARC', uacpy.RunMode.TIME_SERIES),
    ('RAM', uacpy.RunMode.COHERENT_TL),
    ('OAST', uacpy.RunMode.COHERENT_TL),
    ('OASP', uacpy.RunMode.COHERENT_TL),
    ('OASN', uacpy.RunMode.COVARIANCE),
    ('OASS', uacpy.RunMode.REVERBERATION),
    ('OASSP', uacpy.RunMode.BROADBAND),
]))
def test_surface_source_warns_on_non_bellhop_field_models(model_cls, mode):
    """A source at exactly z=0 sits on the pressure-release surface, where
    the field is ~0: field-producing modes warn (base.py
    ``_validate_geometry``) rather than silently returning a degenerate
    result. Bellhop instead *raises* (its rays terminate on the boundary) —
    pinned in test_surface_source_warns_for_field_runs above."""
    src = uacpy.Source(depths=0.0, frequencies=100.0)
    with pytest.warns(UserWarning, match='pressure-release sea surface'):
        if model_cls is RAM:
            # RAM's validate_inputs resolves its grid too, and a source
            # above the first depth cell is refused there, as run() does.
            with pytest.raises(ConfigurationError,
                               match='shallower than one depth cell'):
                model_cls().validate_inputs(_guard_env(), src, _guard_rcv(),
                                            run_mode=mode)
            return
        # OASS's Block VIII range step divides by NR - 1, so its receiver
        # needs two ranges (refused in stage 2 otherwise).
        model = model_cls()
        rcv = (uacpy.Receiver(depths=[50.0], ranges=[1000.0, 2000.0])
               if type(model).__name__ == 'OASS' else _guard_rcv())
        model.validate_inputs(_guard_env_for(model_cls), src, rcv,
                              run_mode=mode)


@pytest.mark.parametrize('model_cls,mode', _model_mode_params([
    ('Kraken', uacpy.RunMode.MODES),
    ('OASR', uacpy.RunMode.REFLECTION),
    ('Bounce', uacpy.RunMode.REFLECTION),
]))
def test_surface_source_is_silent_for_modes_and_reflection(model_cls, mode):
    """Mode shapes and reflection coefficients propagate no source field, so
    the z=0 surface-source warning is suppressed for
    ``RunMode.MODES``/``RunMode.REFLECTION`` (and Bounce validates no
    geometry at all).

    Scoped to the surface-source phrase rather than erroring on every
    ``UserWarning``. The blanket form also caught the OASES licence notice,
    which ``PropagationModel`` deduplicates per PROCESS
    (``_WARNED_MODEL_PROVENANCE``, ``models/_notices.py``), so the OASR case passed
    only when some earlier test in the same worker had already consumed it —
    green under the full suite, red whenever the selection changed.
    """
    src = uacpy.Source(depths=0.0, frequencies=100.0)
    with recorded_warnings() as rec:
        model_cls().validate_inputs(_guard_env(), src, _guard_rcv(),
                                    run_mode=mode)
    assert [w for w in rec
            if 'pressure-release sea surface' in str(w.message)] == []


@pytest.mark.requires_oases
def test_the_oases_licence_notice_warns_once_per_process(monkeypatch):
    """The licence notice is a ProvenanceWarning the first time an OASES
    engine is built in a process, and silent for every later one, OAST or
    another OASES program alike (one source, ``oases``)."""
    from uacpy.models import base as models_base
    from uacpy.models.oases import OASR, OAST
    monkeypatch.setattr(models_base, '_WARNED_MODEL_PROVENANCE', set())

    def licence_notices(cls):
        with recorded_warnings() as rec:
            cls()
        return [w for w in rec if issubclass(w.category, ProvenanceWarning)
                and 'OASES' in str(w.message)]

    assert len(licence_notices(OAST)) == 1
    assert licence_notices(OAST) == []
    assert licence_notices(OASR) == []


# --- receiver grids whose largest range is 0 m ------------------------------

def _field_model_params():
    """Field-computing wrappers, one param each with the markers of the
    installs its registry entry declares (their constructors resolve the
    executable)."""
    from uacpy.models.kraken import Kraken
    from uacpy.models.oases import OAST
    params = []
    for cls in (Bellhop, Kraken, Scooter, RAM, OAST):
        marks = [getattr(pytest.mark, f'requires_{install}')
                 for install in engine_entry(cls.__name__).requires]
        params.append(pytest.param(cls, id=cls.__name__, marks=marks))
    return params


class TestAReceiverWhoseLargestRangeIsZeroIsRefusedBeforeTheDeck:
    """``PropagationModel._validate_geometry`` refuses a receiver whose
    ``range_max`` is 0 m on every field-computing wrapper with the same
    ``ConfigurationError``: a point source's field is singular on its own
    axis, so no engine can fill such a grid (RAM's grid chooser divided by
    the zero range step). The check sits in ``validate_inputs``, which every
    ``run`` calls before its work directory exists, so no deck is written.
    A mixed grid keeps one positive range and passes; the modes that read
    no receiver range (``MODES``, ``RAYS``, ``REFLECTION``, the OASN array
    products) accept an all-zero grid."""

    @staticmethod
    def _triple():
        env = uacpy.Environment(name='r0', bathymetry=100.0, ssp=1500.0)
        src = uacpy.Source(depths=50.0, frequencies=100.0)
        return env, src

    @pytest.mark.parametrize('model_cls', _field_model_params())
    def test_a_single_zero_range_raises_before_any_work_directory(
            self, model_cls, monkeypatch):
        env, src = self._triple()
        model = model_cls(verbose=False)

        def _no_deck(*a, **k):
            raise AssertionError("a work directory was created before the "
                                 "receiver-range check")
        monkeypatch.setattr(model, '_setup_file_manager', _no_deck)
        with recorded_warnings() as record:
            with pytest.raises(ConfigurationError,
                               match=r"largest range is 0 m"):
                model.run(env, src,
                          uacpy.Receiver(depths=[50.0], ranges=[0.0]))
        # The refusal is the whole answer: no NaN-column warning ahead of it.
        assert [str(w.message) for w in record] == []

    @pytest.mark.parametrize('model_cls', _field_model_params())
    def test_a_mixed_grid_with_one_positive_range_passes_validation(
            self, model_cls):
        env, src = self._triple()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model_cls(verbose=False).validate_inputs(
                env, src, uacpy.Receiver(depths=[50.0], ranges=[0.0, 1000.0]))

    @pytest.mark.requires_binary  # constructs Kraken (resolves its binary)
    def test_modes_accepts_an_all_zero_range_grid(self):
        from uacpy.models.kraken import Kraken
        from uacpy.core.run_settings import RunMode
        env, src = self._triple()
        Kraken(verbose=False).validate_inputs(
            env, src, uacpy.Receiver(depths=[50.0], ranges=[0.0]),
            run_mode=RunMode.MODES)


# --- Assignment after construction runs the constructor's checks -----------

class TestAssignmentIsValidatedLikeConstruction:
    """``Environment``, ``Source`` and ``Receiver`` refuse on assignment what
    they refuse at construction, and normalise what they normalise there; a
    refused assignment leaves the carrier as it was."""

    def test_environment_refuses_a_kg_per_m3_water_density(self):
        env = Environment(bathymetry=100.0)
        with pytest.raises(ConfigurationError, match="kg/m³"):
            env.water_density = 1027.0
        assert env.water_density == pytest.approx(1.027)

    def test_environment_stores_an_assigned_density_as_a_float(self):
        env = Environment(bathymetry=100.0)
        env.water_density = np.float32(1.03)
        assert type(env.water_density) is float

    def test_environment_refuses_an_absorption_that_is_not_a_model(self):
        env = Environment(bathymetry=100.0)
        with pytest.raises(ConfigurationError, match="must be an Absorption law"):
            env.absorption = 'thorp'
        assert env.absorption is None

    def test_source_normalises_assigned_depths_and_frequencies(self):
        src = uacpy.Source(depths=50.0, frequencies=100.0)
        src.depths = [40.0, 60.0]
        src.frequencies = 200.0
        assert src.depths.dtype == np.float64
        np.testing.assert_array_equal(src.frequencies, [200.0])
        np.testing.assert_array_equal(src.weights, [1.0, 1.0])

    def test_source_refuses_decreasing_depths_and_keeps_its_own(self):
        src = uacpy.Source(depths=[40.0, 60.0], frequencies=100.0)
        with pytest.raises(ConfigurationError, match="strictly increasing"):
            src.depths = [60.0, 40.0]
        np.testing.assert_array_equal(src.depths, [40.0, 60.0])

    def test_a_uniform_source_weight_follows_new_depths(self):
        src = uacpy.Source(depths=[40.0, 60.0], frequencies=100.0,
                           weights=2.0)
        src.depths = [10.0, 20.0, 30.0]
        np.testing.assert_array_equal(src.weights, [2.0, 2.0, 2.0])

    def test_distinct_source_weights_refuse_a_new_depth_count(self):
        src = uacpy.Source(depths=[40.0, 60.0], frequencies=100.0,
                           weights=[1.0, -1.0])
        with pytest.raises(ConfigurationError, match="different weights"):
            src.depths = [10.0, 20.0, 30.0]
        src.depths = [10.0, 20.0]
        np.testing.assert_array_equal(src.weights, [1.0, -1.0])

    def test_source_refuses_an_assigned_weight_count_mismatch(self):
        src = uacpy.Source(depths=[40.0, 60.0], frequencies=100.0)
        with pytest.raises(ConfigurationError, match="one weight per depth"):
            src.weights = [1.0, 2.0, 3.0]

    def test_receiver_refuses_a_negative_non_increasing_axis(self):
        rcv = uacpy.Receiver(depths=[10.0, 20.0], ranges=[100.0, 200.0])
        with pytest.raises(ConfigurationError, match="non-negative"):
            rcv.depths = np.array([20.0, 10.0, -5.0])
        np.testing.assert_array_equal(rcv.depths, [10.0, 20.0])
        rcv.ranges = 500.0
        np.testing.assert_array_equal(rcv.ranges, [500.0])

    def test_a_copy_and_a_pickle_keep_validating(self):
        import pickle
        src = uacpy.Source(depths=[40.0, 60.0], frequencies=100.0)
        for other in (src.copy(), pickle.loads(pickle.dumps(src))):
            with pytest.raises(ConfigurationError,
                               match='depths must be strictly increasing'):
                other.depths = [60.0, 40.0]
            assert sorted(vars(other)) == sorted(vars(src))


@pytest.mark.requires_binary
def test_compute_modes_checks_every_source_depth():
    """RA-CONTRACT-22: the modes are solved on the first source depth only
    (they do not depend on it), but every depth is checked as ``run``
    checks it, so a depth below the domain is refused instead of being
    dropped silently. Refused before anything is launched."""
    from uacpy.core.exceptions import InvalidDepthError
    from uacpy.models.kraken import Kraken
    env = uacpy.Environment(
        bathymetry=100.0, ssp=1500.0,
        bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                        sound_speed=1700.0, density=1.8,
                                        attenuation=0.5))
    model = Kraken()

    def _no_launch(*a, **k):
        raise AssertionError('launched')
    model._run_subprocess = _no_launch
    with pytest.raises(InvalidDepthError, match='exceeds resolvable depth'):
        model.compute_modes(env, uacpy.Source(depths=[50.0, 5000.0],
                                              frequencies=100.0))


class TestOneUniformStepDecider:
    """``steps_are_uniform`` decides "are these axis steps equal" for both
    transforms that ask it (``Field.to_transfer_function``'s time axis and
    ``uniform_frequency_step``'s frequency axis), at ``UNIFORM_STEP_RTOL``.
    The axes the package writes pass; an axis with a real step error is
    refused."""

    #: float64 ``arange`` time axis, 2**23 samples at 96 kHz: step jitter
    #: 1.16e-9 relative, which a 1e-9 tolerance refused.
    LONG_TIME = np.arange(2 ** 23) / 96000.0
    #: 3000 bins of 1/3 Hz from 10 Hz, printed ``%.12g`` as an AT deck writes
    #: them and read back: step jitter 2.0e-8 relative.
    DECK_FREQS = np.array([float('%.12g' % (10.0 + i / 3.0))
                           for i in range(3000)])

    @staticmethod
    def _jitter(axis):
        steps = np.diff(axis)
        return float(np.max(np.abs(steps - steps.mean())) / steps.mean())

    def test_the_constant(self):
        from uacpy.core._validate import UNIFORM_STEP_RTOL
        assert UNIFORM_STEP_RTOL == 1e-6

    def test_the_measured_axes_carry_the_jitter_they_were_measured_at(self):
        assert self._jitter(self.LONG_TIME) > 1e-9
        assert self._jitter(self.DECK_FREQS) > 1e-8

    def test_a_long_float64_time_axis_transforms(self):
        from uacpy.core.results import Field
        trace = Field(data=np.zeros(self.LONG_TIME.size),
                      coords={'time': self.LONG_TIME},
                      frequencies=1000.0, kind='pressure')
        H = trace.to_transfer_function()
        assert 'frequency' in H.coords

    def test_a_deck_printed_frequency_axis_has_one_step(self):
        from uacpy.acoustic_signal.channel import uniform_frequency_step
        assert uniform_frequency_step(self.DECK_FREQS) == pytest.approx(
            1.0 / 3.0, rel=1e-6)

    def test_a_time_axis_with_a_real_step_error_is_refused(self):
        t = np.arange(64) / 1000.0
        t[40:] += 1e-4 / 1000.0          # one step 1e-4 too long
        from uacpy.core.results import Field
        trace = Field(data=np.zeros(t.size), coords={'time': t},
                      frequencies=100.0, kind='pressure')
        with pytest.raises(ConfigurationError, match='not uniformly spaced'):
            trace.to_transfer_function()

    def test_a_frequency_axis_with_a_real_step_error_is_refused(self):
        from uacpy.acoustic_signal.channel import uniform_frequency_step
        f = np.arange(64) * 0.5 + 10.0
        f[40:] += 1e-4 * 0.5
        with pytest.raises(ConfigurationError, match='uniformly spaced'):
            uniform_frequency_step(f)
