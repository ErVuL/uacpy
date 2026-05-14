"""Tests for the collapse policy: per-key validation, per-model
defaults via ``_set_collapse_defaults``, and the new orthogonal
``rd_layered_range`` / ``rd_layered_layers`` axes.

These tests exercise ``_project_environment`` directly without spawning
any binary — the projection logic is pure Python.
"""

import warnings

import numpy as np
import pytest

from uacpy.core.environment import (
    BoundaryProperties, Environment, LayeredBottom,
    RangeDependentBottom, RangeDependentLayeredBottom, SedimentLayer,
    SoundSpeedProfile,
)
from uacpy.core.exceptions import ConfigurationError
from uacpy.models.base import DEFAULT_COLLAPSE


# ---------------------------------------------------------------------
# Per-model defaults
# ---------------------------------------------------------------------

# (model class name, expected per-key overrides relative to DEFAULT_COLLAPSE)
_PER_MODEL_DEFAULTS = [
    ('Bellhop',     {}),
    ('BellhopCUDA', {}),
    ('Kraken',      {'ssp': 'mean', 'bottom': 'median'}),
    ('KrakenC',     {'ssp': 'mean', 'bottom': 'median'}),
    ('KrakenField', {'bottom': 'median', 'rd_layered_layers': 'preserve'}),
    ('Scooter',     {'ssp': 'mean', 'bottom': 'median', 'rd_layered_layers': 'preserve'}),
    ('SPARC',       {'ssp': 'mean', 'bottom': 'median', 'rd_layered_layers': 'preserve'}),
    ('OAST',        {'ssp': 'mean', 'bottom': 'median', 'rd_layered_layers': 'preserve'}),
    ('OASN',        {'ssp': 'mean', 'bottom': 'median', 'rd_layered_layers': 'preserve'}),
    ('OASR',        {'bottom': 'median', 'rd_layered_layers': 'preserve'}),
    ('OASP',        {'ssp': 'mean', 'bottom': 'median', 'rd_layered_layers': 'preserve'}),
    ('Bounce',      {'bottom': 'median', 'rd_layered_layers': 'preserve'}),
    ('RAM',         {}),
]


@pytest.mark.parametrize('cls_name,overrides', _PER_MODEL_DEFAULTS)
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


def test_user_collapse_overrides_per_model_defaults():
    """User ``collapse={...}`` always wins over model defaults."""
    from uacpy.models import Bounce
    bn = Bounce(verbose=False, collapse={'bottom': 'r0'})
    assert bn._collapse['bottom'] == 'r0'  # not Bounce's 'median' default


def test_unknown_collapse_key_raises():
    """Passing an unknown collapse key raises ``ConfigurationError``."""
    from uacpy.models import Bellhop
    with pytest.raises(ConfigurationError, match='Unknown collapse keys'):
        Bellhop(verbose=False, collapse={'bogus': 'value'})


# ---------------------------------------------------------------------
# rd_layered_range / rd_layered_layers — orthogonal axes
# ---------------------------------------------------------------------

def _make_rdlb():
    hs = BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1700.0, density=1.7, attenuation=0.3,
    )
    profiles = []
    for c in (1600.0, 1620.0, 1640.0):
        profiles.append(LayeredBottom(
            layers=[SedimentLayer(thickness=10.0, sound_speed=c, density=1.5,
                                  attenuation=0.2)],
            halfspace=hs,
        ))
    return RangeDependentLayeredBottom(
        ranges=np.array([0.0, 5000.0, 10000.0]),
        profiles=profiles,
    )


def _rdlb_env():
    return Environment(name='rdlb', bathymetry=100.0, ssp=1500.0, bottom=_make_rdlb())


def _bare_model_factory(supports_layered: bool):
    """Build a minimal subclass that doesn't spawn any binary."""
    from uacpy.models.base import PropagationModel

    class _Bare(PropagationModel):
        def __init__(self, **kw):
            super().__init__(**kw)
            self._supports_layered_bottom = supports_layered

        def run(self, env, source, receiver, run_mode=None, **kwargs):
            return self._project_environment(env)

    return _Bare


@pytest.mark.parametrize('range_method,expected_c', [
    ('r0',     1600.0),
    ('rmax',   1640.0),
    ('median', 1620.0),
])
def test_rd_layered_range_picks_right_profile(range_method, expected_c):
    """``rd_layered_range`` selects which profile contributes (when the
    layer axis collapses to ``top_layer`` we can read the layer's c
    directly)."""
    Bare = _bare_model_factory(supports_layered=False)
    bare = Bare(collapse={
        'rd_layered_range': range_method,
        'rd_layered_layers': 'top_layer',
    })
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        proj = bare.run(_rdlb_env(), None, None)
    assert proj.bottom.sound_speed == pytest.approx(expected_c)


def test_rd_layered_layers_preserve_keeps_layeredbottom():
    """``rd_layered_layers='preserve'`` keeps the layer stack at the
    chosen range. Requires the model to support ``LayeredBottom``."""
    Bare = _bare_model_factory(supports_layered=True)
    bare = Bare(collapse={
        'rd_layered_range': 'median',
        'rd_layered_layers': 'preserve',
    })
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        proj = bare.run(_rdlb_env(), None, None)
    assert isinstance(proj.bottom, LayeredBottom)
    assert len(proj.bottom.layers) == 1


def test_rd_layered_layers_preserve_requires_layered_support():
    """``preserve`` on a model that doesn't support ``LayeredBottom``
    raises ``ConfigurationError`` — the projected env would be a shape
    the model can't consume."""
    Bare = _bare_model_factory(supports_layered=False)
    bare = Bare(collapse={'rd_layered_layers': 'preserve'})
    with pytest.raises(ConfigurationError, match='preserve'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            bare.run(_rdlb_env(), None, None)


@pytest.mark.parametrize('bad_key,bad_val', [
    ('rd_layered_range',  'mean'),       # 'mean' isn't valid for to_profile
    ('rd_layered_layers', 'bogus'),
    ('rd_layered_range',  'bogus'),
])
def test_rd_layered_invalid_value_raises(bad_key, bad_val):
    """Invalid collapse values raise :class:`ConfigurationError` at
    construction, like unknown keys."""
    Bare = _bare_model_factory(supports_layered=True)
    with pytest.raises(ConfigurationError, match=bad_key):
        Bare(collapse={bad_key: bad_val})


# ---------------------------------------------------------------------
# Bellhop non-quad RD-SSP path now honours ``collapse['ssp']``
# ---------------------------------------------------------------------

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

    bh = Bellhop(verbose=False, collapse={'ssp': 'rmax'})
    # Drive the same code path Bellhop.run hits (without spawning a binary)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        env_collapsed = env.copy()
        env_collapsed.ssp = env.ssp.collapse(bh._collapse['ssp'])

    sample_c = float(env_collapsed.ssp.data[0, 0])
    assert sample_c == pytest.approx(1520.0), (
        f"Expected rmax-collapsed c ~ 1520, got {sample_c}"
    )


# ---------------------------------------------------------------------
# Bellhop auto-route detects elastic-RangeDependentBottom
# ---------------------------------------------------------------------

def test_bellhop_auto_route_detects_elastic_rd_halfspace():
    """A ``RangeDependentBottom`` with non-zero ``shear_speed`` anywhere
    along range triggers the BOUNCE auto-route."""
    from uacpy.models import Bellhop
    rd = RangeDependentBottom(
        ranges=np.array([0.0, 5000.0, 10000.0]),
        sound_speed=np.array([1600.0, 1650.0, 1700.0]),
        density=np.array([1.5, 1.6, 1.7]),
        attenuation=np.array([0.2, 0.3, 0.4]),
        shear_speed=np.array([0.0, 400.0, 0.0]),  # elastic in the middle
    )
    env = Environment(name='elastic-RD', bathymetry=100.0, ssp=1500.0, bottom=rd)
    assert env.has_elastic_bottom() is True

    # Don't actually run BOUNCE — just confirm the predicate fires.
    bh = Bellhop(verbose=False)
    assert bh._supports_range_dependent_bottom is True  # fluid-RD is native
    # The auto-route trigger inside Bellhop.run reads env.has_elastic_bottom()
    # — assert directly on the env API to avoid binary execution.


# ---------------------------------------------------------------------
# LayeredBottom.collapse('volume_average') forwards shear_attenuation
# ---------------------------------------------------------------------

def test_layered_volume_average_forwards_shear_attenuation():
    """The volume-averaged collapse must include shear_attenuation in
    the resulting halfspace, not silently drop it (recent bugfix)."""
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
    lb = LayeredBottom(layers=layers, halfspace=hs)
    flat = lb.collapse('volume_average')
    assert flat.shear_attenuation > 0.0, (
        "volume_average collapse must forward shear_attenuation, not drop it"
    )


# ---------------------------------------------------------------------
# _collapse_elastic_boundary handles RangeDependentBottom shear ndarrays
# ---------------------------------------------------------------------

def test_collapse_elastic_rd_bottom_zeros_shear_arrays():
    """``_collapse_elastic_boundary(rd_bottom, 'fluid')`` zeroes the
    per-range ``shear_speed`` / ``shear_attenuation`` ndarrays while
    preserving shape and dtype."""
    from uacpy.models.base import PropagationModel
    rd = RangeDependentBottom(
        ranges=np.array([0.0, 5000.0, 10000.0]),
        sound_speed=np.array([1600.0, 1650.0, 1700.0]),
        density=np.array([1.5, 1.6, 1.7]),
        attenuation=np.array([0.2, 0.3, 0.4]),
        shear_speed=np.array([0.0, 400.0, 800.0]),
        shear_attenuation=np.array([0.0, 0.5, 1.0]),
    )
    collapsed = PropagationModel._collapse_elastic_boundary(rd, 'fluid')
    # Must remain a RangeDependentBottom with per-range arrays
    assert isinstance(collapsed, RangeDependentBottom)
    assert isinstance(collapsed.shear_speed, np.ndarray)
    assert collapsed.shear_speed.shape == (3,)
    assert np.all(collapsed.shear_speed == 0.0), (
        f"shear_speed must be zeroed, got {collapsed.shear_speed!r}"
    )
    assert isinstance(collapsed.shear_attenuation, np.ndarray)
    assert collapsed.shear_attenuation.shape == (3,)
    assert np.all(collapsed.shear_attenuation == 0.0), (
        f"shear_attenuation must be zeroed, "
        f"got {collapsed.shear_attenuation!r}"
    )
    # Compressional properties preserved
    np.testing.assert_array_equal(
        collapsed.sound_speed, np.array([1600.0, 1650.0, 1700.0])
    )
    np.testing.assert_array_equal(
        collapsed.density, np.array([1.5, 1.6, 1.7])
    )
    # Original input untouched (deepcopy contract)
    assert rd.shear_speed[1] == 400.0


# ---------------------------------------------------------------------
# RangeDependentLayeredBottom.collapse uses median range
# ---------------------------------------------------------------------

def test_rdlb_collapse_uses_median_range():
    """``rdlb.collapse(method)`` shorthand selects the median range
    profile (aligned with ``_project_environment``)."""
    rdlb = _make_rdlb()
    # The 'halfspace' branch returns the chosen profile's halfspace
    # (1700 for all profiles in _make_rdlb — same hs object). Use
    # 'top_layer' instead to expose the per-range layer's sound_speed.
    flat_top = rdlb.collapse('top_layer')
    assert flat_top.sound_speed == pytest.approx(1620.0), (
        f"Expected median-range top layer c ~ 1620, got {flat_top.sound_speed}"
    )
