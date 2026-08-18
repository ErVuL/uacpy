"""Tests for the collapse policy: per-key validation, per-model defaults via
``_set_collapse_defaults``, and the two orthogonal bottom axes
``bottom_range`` (which column) / ``bottom_layers`` (how the layer stack
flattens).

These tests exercise ``_project_environment`` directly without spawning
any binary — the projection logic is pure Python.
"""

import warnings

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
from uacpy.models.base import DEFAULT_COLLAPSE, RunMode


# ---------------------------------------------------------------------
# Per-model defaults
# ---------------------------------------------------------------------

# (model class name, expected per-key overrides relative to DEFAULT_COLLAPSE)
#
# Two patterns explain the shape of this table. ``bottom_range: 'median'``
# appears for every model except Bellhop and RAM — the two that declare
# ``_supports_range_dependent_bottom`` and so never collapse the range axis at
# all. ``ssp: 'mean'`` is absent for OASR and Bounce because both return a
# reflection coefficient rather than a field: only the bottom stack feeds the
# answer, so the global ``'r0'`` stands.
_PER_MODEL_DEFAULTS = [
    ('Bellhop',     {}),
    ('Kraken',      {'ssp': 'mean', 'bottom_range': 'median'}),
    ('Scooter',     {'ssp': 'mean', 'bottom_range': 'median'}),
    ('SPARC',       {'ssp': 'mean', 'bottom_range': 'median'}),
    ('OAST',        {'ssp': 'mean', 'bottom_range': 'median'}),
    ('OASN',        {'ssp': 'mean', 'bottom_range': 'median'}),
    ('OASR',        {'bottom_range': 'median'}),
    ('OASP',        {'ssp': 'mean', 'bottom_range': 'median'}),
    ('Bounce',      {'bottom_range': 'median'}),
    ('RAM',         {}),
]

_OASES_MODELS = {'OAST', 'OASN', 'OASR', 'OASP'}

# Each model resolves (and existence-checks) its binary in __init__, so the
# construction below needs that binary — requires_binary for all, plus
# requires_oases for the separately-licensed OASES family.
_PER_MODEL_PARAMS = [
    pytest.param(
        name, overrides, id=name,
        marks=([pytest.mark.requires_binary]
               + ([pytest.mark.requires_oases] if name in _OASES_MODELS else [])),
    )
    for name, overrides in _PER_MODEL_DEFAULTS
]


@pytest.mark.parametrize('cls_name,overrides', _PER_MODEL_PARAMS)
def test_per_model_collapse_defaults(cls_name, overrides):
    """Each model installs its physics-aware overrides via
    ``_set_collapse_defaults`` and inherits the rest from
    ``DEFAULT_COLLAPSE``."""
    import uacpy.models as models
    cls = getattr(models, cls_name)
    m = cls(verbose=False)
    for key, expected in DEFAULT_COLLAPSE.items():
        want = overrides.get(key, expected)
        assert m._collapse[key] == want, (
            f"{cls_name}: collapse[{key!r}]={m._collapse[key]!r}, "
            f"expected {want!r}"
        )


@pytest.mark.requires_binary  # constructs a concrete model
def test_user_collapse_overrides_per_model_defaults():
    """User ``collapse={...}`` always wins over model defaults."""
    from uacpy.models import Bounce
    bn = Bounce(verbose=False, collapse={'bottom_range': 'r0'})
    assert bn._collapse['bottom_range'] == 'r0'  # not Bounce's 'median' default


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
    from uacpy.models.base import PropagationModel

    class _Bare(PropagationModel):
        def __init__(self, **kw):
            super().__init__(**kw)
            self._supports_layered_bottom = supports_layered

        def run(self, env, source, receiver, run_mode=None):
            return self._project_environment(env)

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
        proj = bare.run(_rdlb_env(), None, None)
    assert not proj.bottom.is_range_dependent
    assert proj.bottom.columns[0].halfspace.sound_speed == pytest.approx(expected_c)


def test_layered_kept_when_model_supports_layers():
    """A model that supports layers keeps the layer stack at the chosen
    column — only the range axis collapses."""
    Bare = _bare_model_factory(supports_layered=True)
    bare = Bare(collapse={'bottom_range': 'median'})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        proj = bare.run(_rdlb_env(), None, None)
    assert not proj.bottom.is_range_dependent and proj.bottom.is_layered
    assert len(proj.bottom.columns[0].layers) == 1


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

    # interp_ssp='c-linear' (≠ 'quad') forces Bellhop's run-path to collapse the
    # 2-D SSP. Drive the actual run (not a hand-rolled collapse) and assert the
    # collapse fired with the user's method — this catches a regression where
    # run() stops honouring collapse['ssp'].
    bh = Bellhop(verbose=False, interp_ssp='c-linear', collapse={'ssp': 'rmax'})
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
def test_has_elastic_bottom_is_true_when_any_range_has_shear():
    """``env.has_elastic_bottom`` is true when ``shear_speed`` is non-zero at
    *any* range, not only at r=0 — it is one of the two predicates
    ``Bellhop._maybe_route_through_bounce`` ORs together to auto-route through
    BOUNCE (``models/bellhop.py:718-720``). Asserted on the env API rather
    than through a run, so nothing here proves the route fires."""
    from uacpy.models import Bellhop
    rd = Bottom.from_halfspaces(np.array([0.0, 5000.0, 10000.0]),
        sound_speed=np.array([1600.0, 1650.0, 1700.0]),
        density=np.array([1.5, 1.6, 1.7]),
        attenuation=np.array([0.2, 0.3, 0.4]),
        shear_speed=np.array([0.0, 400.0, 0.0]),  # elastic in the middle
    )
    env = Environment(name='elastic-RD', bathymetry=100.0, ssp=1500.0, bottom=rd)
    assert env.has_elastic_bottom is True

    # Don't actually run BOUNCE — just confirm the predicate fires.
    bh = Bellhop(verbose=False)
    assert bh._supports_range_dependent_bottom is True  # fluid-RD is native
    # The auto-route trigger inside Bellhop.run reads env.has_elastic_bottom
    # — assert directly on the env API to avoid binary execution.


# ---------------------------------------------------------------------
# SeabedColumn.collapse('volume_average') forwards shear_attenuation
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
    flat = lb.collapse('volume_average')
    assert flat.shear_attenuation > 0.0, (
        "volume_average collapse must forward shear_attenuation, not drop it"
    )


# ---------------------------------------------------------------------
# _collapse_elastic_boundary handles Bottom shear ndarrays
# ---------------------------------------------------------------------

def test_collapse_elastic_rd_bottom_zeros_shear_arrays():
    """``_collapse_elastic_boundary(rd_bottom, 'fluid')`` zeroes the
    per-range ``shear_speed`` / ``shear_attenuation`` ndarrays while
    preserving shape and dtype."""
    from uacpy.models.base import PropagationModel
    rd = Bottom.from_halfspaces(np.array([0.0, 5000.0, 10000.0]),
        sound_speed=np.array([1600.0, 1650.0, 1700.0]),
        density=np.array([1.5, 1.6, 1.7]),
        attenuation=np.array([0.2, 0.3, 0.4]),
        shear_speed=np.array([0.0, 400.0, 800.0]),
        shear_attenuation=np.array([0.0, 0.5, 1.0]),
    )
    collapsed = PropagationModel._collapse_elastic_boundary(rd, 'fluid')
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
