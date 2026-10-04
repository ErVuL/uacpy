"""Regression tests for ``PropagationModel.copy`` and the constructor-arg
storage contract.

``model.copy(**overrides)`` is the documented parameter-sweep primitive, so a
copy that silently drops a knob corrupts every sweep that uses it. What every
registered engine owes (each ``__init__`` parameter stored as
``self.<name>``, every value and a ``collapse`` override carried, an unknown
override refused) is held by ``test_engine_conformance.py``. These tests pin
the engine-specific cases: Kraken's parent-class knobs, Bellhop's backend.
"""

import numpy as np
import pytest

from uacpy.models import Bellhop, Kraken
from uacpy.core.run_settings import RunMode
from uacpy.core.exceptions import ExecutableNotFoundError, ConfigurationError


@pytest.mark.requires_binary
def test_copy_preserves_parent_class_knobs():
    """``copy`` must carry Kraken's spectral knobs (and ``field_executable``),
    not reset them to defaults."""
    try:
        model = Kraken(verbose=False, c_high=2000.0, n_mesh=30)
    except ExecutableNotFoundError:
        pytest.skip("Kraken binary not installed")
    twin = model.copy()
    assert twin.c_high == 2000.0
    assert twin.n_mesh == 30
    assert twin.field_executable == model.field_executable
    # overriding a parent-class parameter is allowed and applied
    assert model.copy(c_high=1850.0).c_high == 1850.0


@pytest.mark.requires_binary
def test_bellhop_copy_preserves_and_overrides_backend():
    """Regression: ``Bellhop.copy()`` must not re-pin the *resolved* binary.

    A no-op copy keeps the resolved ``version`` — flipping it to 'custom'
    would drop the cxx/cuda ``--<dim>`` flag — while a copy with a
    ``backend=`` override re-resolves the binary rather than carrying the
    already-resolved path back in."""
    bh = Bellhop(verbose=False)                 # auto-resolved (cuda > cxx > fortran)
    assert bh._resolved_backend != 'custom'
    twin = bh.copy(beam_type='G')               # no backend override
    assert twin._resolved_backend == bh._resolved_backend           # NOT flipped to 'custom'
    assert twin._exe == bh._exe
    # A backend override on copy re-resolves the executable.
    forced = Bellhop(backend='fortran', verbose=False)
    assert forced._resolved_backend == 'fortran'
    from uacpy.core.exceptions import ExecutableNotFoundError
    try:
        cxx = forced.copy(backend='cxx')
    except ExecutableNotFoundError:
        pytest.skip("bellhopcxx binary not installed")
    assert cxx.backend == 'cxx'
    assert cxx._resolved_backend == 'cxx'
    assert 'cxx' in cxx._exe.name


@pytest.mark.requires_binary
def test_bellhop_timeseries_requires_waveform(simple_env, source, receiver_small):
    """``Bellhop.run(TIME_SERIES)`` without a source waveform raises rather
    than silently returning the broadband H(f) transfer function."""
    bellhop = Bellhop(verbose=False)
    with pytest.raises(ConfigurationError,
                       match='requires source_waveform and sample_rate'):
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


def test_abstract_run_encodes_fixed_keyword_contract():
    """The abstract ``PropagationModel.run`` is the source of truth for the
    fixed, no-``**kwargs`` signature documented in CLAUDE.md / DEV.md. It must
    declare the keyword-only block (``frequencies``/``source_waveform``/
    ``sample_rate``/``output_duration``/``t_start``) and accept no ``**kwargs`` sink, so the
    contract is visible on the base class and not only in each subclass."""
    import inspect

    from uacpy.models.base import PropagationModel

    sig = inspect.signature(PropagationModel.run)
    params = sig.parameters
    # positional-or-keyword core triple + optional run_mode
    for name in ('env', 'source', 'receiver', 'run_mode'):
        assert name in params, f"abstract run() must declare {name!r}"
    # the keyword-only block the whole contract is built around
    kw_only = {n for n, p in params.items()
               if p.kind is inspect.Parameter.KEYWORD_ONLY}
    assert kw_only == {
        'frequencies', 'source_waveform', 'sample_rate', 'output_duration',
        't_start',
    }
    # no **kwargs sink anywhere
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD
                   for p in params.values())
