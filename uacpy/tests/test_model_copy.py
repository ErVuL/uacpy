"""Regression tests for ``PropagationModel.copy`` and the constructor-arg
storage contract.

``model.copy(**overrides)`` is the documented parameter-sweep primitive, so a
copy that silently drops a knob corrupts every sweep that uses it. These tests
pin three invariants:

1. every ``__init__`` parameter along the MRO is stored as ``self.<name>``
   (so ``copy`` and ``__repr__`` can read it back);
2. ``copy`` round-trips a parent-class knob and a ``collapse`` override;
3. ``copy`` accepts an override on a parent-class parameter.
"""

import numpy as np
import pytest

from uacpy.models import (
    Bellhop, BellhopCUDA, RAM, Kraken, KrakenC, KrakenField,
    Bounce, Scooter, SPARC, OAST, OASN, OASR, OASP,
)
from uacpy.models.base import RunMode, _collect_init_params
from uacpy.core.exceptions import ExecutableNotFoundError, ConfigurationError

_MODEL_CLASSES = [
    Bellhop, BellhopCUDA, RAM, Kraken, KrakenC, KrakenField,
    Bounce, Scooter, SPARC, OAST, OASN, OASR, OASP,
]


def _construct(cls):
    try:
        return cls(verbose=False)
    except ExecutableNotFoundError:
        pytest.skip(f"{cls.__name__} binary not installed")


@pytest.mark.requires_binary
@pytest.mark.parametrize("cls", _MODEL_CLASSES, ids=lambda c: c.__name__)
def test_every_init_param_is_stored(cls):
    """Each constructor parameter across the MRO must be an attribute, or
    ``copy``/``repr`` silently drop it."""
    model = _construct(cls)
    missing = [
        name for name, _ in _collect_init_params(cls)
        if not hasattr(model, name)
    ]
    assert not missing, (
        f"{cls.__name__}.__init__ params not stored as self.<name>: {missing}"
    )


@pytest.mark.requires_binary
@pytest.mark.parametrize("cls", _MODEL_CLASSES, ids=lambda c: c.__name__)
def test_copy_round_trips_param_values(cls):
    """``copy`` must reproduce each stored constructor *value*, not just the
    attribute name — a model storing a transformed value under the ctor-arg
    name would double-transform on copy and silently corrupt sweeps."""
    from uacpy.models.base import _values_equal
    model = _construct(cls)
    twin = model.copy()
    for name, _ in _collect_init_params(cls):
        if hasattr(model, name):
            assert _values_equal(getattr(model, name), getattr(twin, name)), (
                f"{cls.__name__}.copy() changed {name!r}"
            )


@pytest.mark.requires_binary
@pytest.mark.parametrize("cls", _MODEL_CLASSES, ids=lambda c: c.__name__)
def test_copy_preserves_collapse(cls):
    """A user ``collapse`` override survives ``copy`` on every model."""
    try:
        model = cls(verbose=False, collapse={"bathymetry": "min"})
    except ExecutableNotFoundError:
        pytest.skip(f"{cls.__name__} binary not installed")
    twin = model.copy()
    assert twin.collapse == {"bathymetry": "min"}
    assert twin._user_collapse == {"bathymetry": "min"}
    assert twin._collapse["bathymetry"] == "min"


@pytest.mark.requires_binary
def test_copy_preserves_parent_class_knobs():
    """KrakenField inherits its spectral knobs from ``_KrakenBase``; ``copy``
    must carry them (and ``field_executable``), not reset them to defaults."""
    try:
        model = KrakenField(verbose=False, c_high=2000.0, n_mesh=30)
    except ExecutableNotFoundError:
        pytest.skip("KrakenField binary not installed")
    twin = model.copy()
    assert twin.c_high == 2000.0
    assert twin.n_mesh == 30
    assert twin.field_executable == model.field_executable
    # overriding a parent-class parameter is allowed and applied
    assert model.copy(c_high=1850.0).c_high == 1850.0


@pytest.mark.requires_binary
def test_bellhop_timeseries_requires_waveform(simple_env, source, receiver_small):
    """``Bellhop.run(TIME_SERIES)`` without a source waveform raises rather
    than silently returning the broadband H(f) transfer function."""
    bellhop = Bellhop(verbose=False)
    with pytest.raises(ConfigurationError):
        bellhop.run(
            simple_env, source, receiver_small,
            run_mode=RunMode.TIME_SERIES,
        )
    # BROADBAND without a waveform is the legitimate H(f) path — must not raise.
    field = bellhop.run(
        simple_env, source, receiver_small,
        run_mode=RunMode.BROADBAND,
    )
    assert np.iscomplexobj(field.data)
