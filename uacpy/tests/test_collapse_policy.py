"""Tests for the collapse policy: per-key validation, per-model defaults via
``_set_collapse_defaults``, and the two orthogonal bottom axes
``bottom_range`` (which column) / ``bottom_layers`` (how the layer stack
flattens).

These tests exercise ``_project_environment`` directly without spawning
any binary — the projection logic is pure Python.
"""

import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pytest

from uacpy.core.environment import (
    BoundaryProperties, Environment, SeabedColumn,
    Bottom, SedimentLayer,
    SoundSpeedProfile,
)
from uacpy.core.exceptions import ConfigurationError, ExecutableNotFoundError
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.run_settings import RunMode
from uacpy.models._projection import (
    DEFAULT_COLLAPSE, collapse_elastic_boundary,
)


# ---------------------------------------------------------------------
# Per-model defaults
# ---------------------------------------------------------------------

# Each engine's collapse defaults (``spec.collapse``) and the policy a default
# instance resolves from them are held to the saved engine records
# (``tests/data/engine_specs.json``, ``test_capability_flags.py``).

def test_every_model_reduces_a_seabed_to_its_median_column():
    """One rule for the seabed column: the global default is 'median' and
    no model overrides it, so one environment reduces to the same seabed
    whichever engine runs it."""
    from uacpy.models._registry import engine_classes
    assert DEFAULT_COLLAPSE['bottom_range'] == 'median'
    overriding = sorted(cls.__name__ for cls in engine_classes().values()
                        if 'bottom_range' in cls.spec.collapse)
    assert not overriding, (
        f"{overriding} override collapse['bottom_range']; the seabed column "
        f"is one rule for every engine")


@pytest.mark.requires_binary  # constructs a concrete model
def test_user_collapse_overrides_per_model_defaults():
    """User ``collapse={...}`` always wins over model defaults."""
    from uacpy.models import Bounce
    bn = Bounce(verbose=False, collapse={'bottom_range': 'r0'})
    assert bn._collapse['bottom_range'] == 'r0'  # not the global 'median'


def test_unknown_collapse_key_raises():
    """Passing an unknown collapse key raises ``ConfigurationError``."""
    from uacpy.models import Bellhop
    with pytest.raises(ConfigurationError, match='Unknown collapse keys'):
        Bellhop(verbose=False, collapse={'bogus': 'value'})


# ---------------------------------------------------------------------
# bottom_range / bottom_layers — orthogonal axes
# ---------------------------------------------------------------------

def _make_rdlb():
    hs = BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1700.0, density=1.7, attenuation=0.3,
    )
    profiles = []
    for c in (1600.0, 1620.0, 1640.0):
        profiles.append(SeabedColumn(
            layers=[SedimentLayer(thickness=10.0, sound_speed=c, density=1.5,
                                  attenuation=0.2)],
            halfspace=hs,
        ))
    return Bottom.from_columns(profiles, ranges=np.array([0.0, 5000.0, 10000.0]))


def _rdlb_env():
    return Environment(name='rdlb', bathymetry=100.0, ssp=1500.0, bottom=_make_rdlb())


def _bare_model_factory(supports_layered: bool):
    """Build a minimal subclass that doesn't spawn any binary."""
    from uacpy.core.run_settings import RunMode
    from uacpy.models.base import (
        DEFAULT_RUN_TIMEOUT_S, PropagationModel, _host_options,
    )
    from uacpy.models._spec import ModelSpec

    def _never_launched(self, *args):
        raise AssertionError('this stand-in launches nothing')

    class _Bare(PropagationModel):
        # ``spec`` and ``provenance_id`` are required of any subclass that defines
        # the stage hooks; this double spawns no binary, so the source id
        # only has to be a real one — 'acoustics_toolbox' is unrestricted,
        # so nothing here emits a licence warning.
        # ``_supports_layered_bottom`` is set per-instance below, after the
        # spec has been applied.
        spec = ModelSpec(modes=(RunMode.COHERENT_TL,))
        provenance_id = 'acoustics_toolbox'

        def __init__(self, *, use_tmpfs: bool = False,
                     verbose: Union[bool, str] = False,
                     work_dir: Optional[Path] = None,
                     cleanup: Optional[bool] = None,
                     timeout: float = DEFAULT_RUN_TIMEOUT_S,
                     collapse: Optional[Dict[str, str]] = None):
            super().__init__(**_host_options(locals()))
            self._supports_layered_bottom = supports_layered

        _write_input = _launch = _read_output = _to_result = _never_launched

    return _Bare


@pytest.mark.parametrize('range_method,expected_c', [
    ('r0',     1600.0),
    ('rmax',   1640.0),
    ('median', 1620.0),
])
def test_bottom_range_picks_right_profile(range_method, expected_c):
    """``bottom_range`` selects which column contributes; with the layer
    axis flattened to ``top_layer`` we read that column's layer c directly."""
    Bare = _bare_model_factory(supports_layered=False)
    bare = Bare(collapse={
        'bottom_range': range_method,
        'bottom_layers': 'top_layer',
    })
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        proj = bare._project_environment(_rdlb_env())
    assert not proj.bottom.is_range_dependent
    assert proj.bottom.columns[0].halfspace.sound_speed == pytest.approx(expected_c)


def test_layered_kept_when_model_supports_layers():
    """A model that supports layers keeps the layer stack at the chosen
    column — only the range axis collapses."""
    Bare = _bare_model_factory(supports_layered=True)
    bare = Bare(collapse={'bottom_range': 'median'})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        proj = bare._project_environment(_rdlb_env())
    assert not proj.bottom.is_range_dependent and proj.bottom.is_layered
    assert len(proj.bottom.columns[0].layers) == 1


def test_a_collapsed_seafloor_lies_inside_its_profile():
    """Collapsing a sloping seafloor to its deepest point (``'max'``, the
    default) assigns the deeper flat seafloor, and that assignment extends
    a profile ending above it, so the AT writers get a profile that spans
    the water column."""
    env = Environment(bathymetry=[(0.0, 100.0), (5000.0, 200.0)],
                      ssp=1500.0)
    # A profile ending above the deepest point. No public path leaves one
    # (construction and assignment both extend it), so it is stored past
    # them, to reach the projection's own assignment.
    object.__setattr__(env, 'ssp', SoundSpeedProfile.from_pairs(
        [(0.0, 1500.0), (100.0, 1490.0)]))
    bare = _bare_model_factory(supports_layered=False)()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        proj = bare._project_environment(env)
    assert proj.depth == 200.0
    assert proj.ssp.depths[-1] == 200.0
    assert proj.ssp.sound_speed[-1, 0] == 1490.0


@pytest.mark.parametrize('bad_key,bad_val', [
    ('bottom_range',  'bogus'),
    ('bottom_layers', 'bogus'),
])
def test_collapse_invalid_value_raises(bad_key, bad_val):
    """Invalid collapse values raise :class:`ConfigurationError` at
    construction, like unknown keys."""
    Bare = _bare_model_factory(supports_layered=True)
    with pytest.raises(ConfigurationError, match=bad_key):
        Bare(collapse={bad_key: bad_val})


# ---------------------------------------------------------------------
# Bellhop non-quad RD-SSP path honours ``collapse['ssp']``
# ---------------------------------------------------------------------

@pytest.mark.requires_binary  # constructs Bellhop
def test_bellhop_rd_ssp_uses_collapse_policy():
    """When ``ssp.interp != 'quad'`` Bellhop drops the range axis using
    the user's ``collapse['ssp']`` value rather than hardcoded r=0."""
    from uacpy.models import Bellhop
    ssp_2d = SoundSpeedProfile.from_2d(
        depths=np.array([0.0, 100.0]),
        ranges=np.array([0.0, 1000.0, 2000.0]),
        matrix=np.array([[1500.0, 1510.0, 1520.0],
                         [1500.0, 1510.0, 1520.0]])
    )
    env = Environment(
        name='rd-ssp', bathymetry=100.0, ssp=ssp_2d,
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1700.0, density=1.7,
                                  attenuation=0.3),
    )

    src = Source(frequencies=200.0, depths=25.0)
    rcv = Receiver(depths=np.array([50.0]), ranges=np.array([1500.0]))

    # interp_ssp='linear' (≠ 'quad') forces Bellhop's run-path to collapse the
    # 2-D SSP. Drive the actual run (not a hand-rolled collapse) and assert the
    # collapse fired with the user's method — this catches a regression where
    # run() stops honouring collapse['ssp'].
    bh = Bellhop(verbose=False, interp_ssp='linear', collapse={'ssp': 'rmax'})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            bh.run(env, src, rcv, run_mode=RunMode.COHERENT_TL)
        except ExecutableNotFoundError:
            pytest.skip("Bellhop binary not available")
    collapse_w = [w for w in caught
                  if 'collapsed to 1-D' in str(w.message)
                  and "'rmax'" in str(w.message)]
    assert collapse_w, (
        "Bellhop.run did not collapse the 2-D SSP via collapse['ssp']='rmax' "
        f"(warnings seen: {[str(w.message) for w in caught]})"
    )


# ---------------------------------------------------------------------
# Bellhop auto-route detects elastic-Bottom
# ---------------------------------------------------------------------

@pytest.mark.requires_binary  # constructs Bellhop
def test_a_bottom_is_elastic_when_any_range_has_shear():
    """``env.bottom.is_elastic`` is true when ``shear_speed`` is non-zero at
    *any* range, not only at r=0. Bellhop's ``bounce_route`` gates the
    auto-route on ``env.bottom.is_layered`` alone, and ``is_elastic``
    only names the bottom in the notice the route gives,
    so a shear speed that appears first at r>0 decides what the user is told
    the route is for. Asserted on the env API rather than through a run, so
    nothing here proves the route fires."""
    from uacpy.models import Bellhop
    rd = Bottom.from_halfspaces(np.array([0.0, 5000.0, 10000.0]),
        sound_speed=np.array([1600.0, 1650.0, 1700.0]),
        density=np.array([1.5, 1.6, 1.7]),
        attenuation=np.array([0.2, 0.3, 0.4]),
        shear_speed=np.array([0.0, 400.0, 0.0]),  # elastic in the middle
    )
    env = Environment(name='elastic-RD', bathymetry=100.0, ssp=1500.0, bottom=rd)
    assert env.bottom.is_elastic is True

    # Don't actually run BOUNCE — just confirm the predicate fires.
    bh = Bellhop(verbose=False)
    assert bh._supports_range_dependent_bottom is True  # fluid-RD is native


# ---------------------------------------------------------------------
# SeabedColumn.collapse_layers('volume_average') forwards shear_attenuation
# ---------------------------------------------------------------------

def test_layered_volume_average_forwards_shear_attenuation():
    """The volume-averaged collapse must include shear_attenuation in
    the resulting halfspace, not silently drop it — an elastic column that
    collapses to a fluid-lossless half-space changes the physics, not just
    the resolution."""
    hs = BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1800.0, density=1.8, attenuation=0.3,
        shear_speed=600.0, shear_attenuation=1.0,
    )
    layers = [
        SedimentLayer(thickness=10.0, sound_speed=1600.0, density=1.5,
                      attenuation=0.2, shear_speed=300.0, shear_attenuation=0.4),
        SedimentLayer(thickness=10.0, sound_speed=1700.0, density=1.6,
                      attenuation=0.25, shear_speed=400.0, shear_attenuation=0.6),
    ]
    lb = SeabedColumn(layers=layers, halfspace=hs)
    flat = lb.collapse_layers('volume_average')
    assert flat.shear_attenuation > 0.0, (
        "volume_average collapse must forward shear_attenuation, not drop it"
    )


# ---------------------------------------------------------------------
# collapse_elastic_boundary handles Bottom shear ndarrays
# ---------------------------------------------------------------------

def test_collapse_elastic_rd_bottom_zeros_shear_arrays():
    """``collapse_elastic_boundary(rd_bottom, 'fluid')`` zeroes the
    per-range ``shear_speed`` / ``shear_attenuation`` ndarrays while
    preserving shape and dtype."""
    rd = Bottom.from_halfspaces(np.array([0.0, 5000.0, 10000.0]),
        sound_speed=np.array([1600.0, 1650.0, 1700.0]),
        density=np.array([1.5, 1.6, 1.7]),
        attenuation=np.array([0.2, 0.3, 0.4]),
        shear_speed=np.array([0.0, 400.0, 800.0]),
        shear_attenuation=np.array([0.0, 0.5, 1.0]),
    )
    collapsed = collapse_elastic_boundary(rd, 'fluid')
    # Must remain a range-dependent half-space Bottom, shear zeroed per column
    assert isinstance(collapsed, Bottom)
    assert collapsed.is_range_dependent and not collapsed.is_layered
    assert np.all(collapsed.halfspace_shear_speed == 0.0)
    assert np.all(collapsed.halfspace_shear_attenuation == 0.0)
    # Compressional properties preserved
    np.testing.assert_array_equal(
        collapsed.halfspace_sound_speed, np.array([1600.0, 1650.0, 1700.0]))
    np.testing.assert_array_equal(
        collapsed.halfspace_density, np.array([1.5, 1.6, 1.7]))
    # Original input untouched (deepcopy contract)
    assert rd.halfspace_shear_speed[1] == 400.0


@pytest.mark.parametrize('base', ['rigid', 'vacuum'])
def test_collapse_elastic_layers_over_a_parameter_free_half_space(base):
    """An elastic layer over a rigid or vacuum half-space collapses to a
    fluid layer over the same half-space: the half-space carries no shear
    and keeps its fields, where an assignment of zero re-validated it as
    a rigid boundary given a shear_speed and was refused."""
    column = SeabedColumn(
        layers=[SedimentLayer(thickness=20.0, sound_speed=1600.0,
                              density=1.8, attenuation=0.5,
                              shear_speed=300.0, shear_attenuation=0.5)],
        halfspace=BoundaryProperties(acoustic_type=base))
    collapsed = collapse_elastic_boundary(Bottom(columns=[column]), 'fluid')
    assert not collapsed.is_elastic
    assert collapsed.columns[0].layers[0].sound_speed == 1600.0
    assert collapsed.columns[0].halfspace.acoustic_type == base


# ---------------------------------------------------------------------
# Bottom.collapse uses median range
# ---------------------------------------------------------------------

def test_rdlb_collapse_median_range_top_layer():
    """``collapse(range='median', layers='top_layer')`` selects the median
    column and exposes its top-layer sound speed."""
    rdlb = _make_rdlb()
    flat_top = rdlb.collapse(range='median', layers='top_layer')
    cp = flat_top.columns[0].halfspace.sound_speed
    assert cp == pytest.approx(1620.0), (
        f"Expected median-range top layer c ~ 1620, got {cp}"
    )
