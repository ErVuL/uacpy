"""Tests for exception handling and public exception types."""

import pickle

import pytest

import uacpy
from uacpy.core.exceptions import (
    UACPYError, InvalidDepthError, ExecutableNotFoundError,
    ModelExecutionError, UnsupportedFeatureError, ConfigurationError,
)
from uacpy.models import Kraken


class TestCustomExceptions:
    """Direct construction of the typed exception classes."""

    def test_invalid_depth_error(self):
        error = InvalidDepthError(depth=150, max_depth=100, context="Source")
        assert isinstance(error, UACPYError)
        assert "150" in str(error)
        assert "100" in str(error)
        assert "Source" in str(error)

    def test_invalid_depth_error_subclass_of_uacpyerror(self):
        try:
            raise InvalidDepthError(depth=200, max_depth=100, context="Receiver")
        except UACPYError as e:
            assert "200" in str(e)

    def test_unsupported_feature_error(self):
        error = UnsupportedFeatureError(
            "Bellhop", "normal mode computation",
            alternatives=['Kraken', 'OASN'],
        )
        assert isinstance(error, UACPYError)
        msg = str(error)
        assert "Bellhop" in msg
        assert "normal mode" in msg.lower()


class TestExceptionPublicExports:
    """Every exception must be reachable from `uacpy` and `uacpy.core`."""

    def test_uacpy_top_level_exports(self):
        for name in ('UACPYError', 'InvalidDepthError', 'UnsupportedFeatureError',
                     'ConfigurationError', 'ExecutableNotFoundError',
                     'ModelExecutionError'):
            assert hasattr(uacpy, name), f"uacpy missing {name}"

    def test_uacpy_core_exports(self):
        import uacpy.core as core
        for name in ('UACPYError', 'InvalidDepthError', 'UnsupportedFeatureError',
                     'ConfigurationError', 'ExecutableNotFoundError',
                     'ModelExecutionError'):
            assert hasattr(core, name), f"uacpy.core missing {name}"

    def test_isinstance_through_uacpy(self):
        err = uacpy.InvalidDepthError(depth=150, max_depth=100, context='Source')
        assert isinstance(err, uacpy.UACPYError)


class TestInputValidation:
    """Constructor-time validation on Source / Receiver / Environment."""

    def test_negative_source_depth(self):
        with pytest.raises(ConfigurationError, match="source depths must be"):
            uacpy.Source(depths=-10, frequencies=100)

    def test_negative_receiver_depth(self):
        with pytest.raises(ConfigurationError, match="receiver depths must be"):
            uacpy.Receiver(depths=[-10], ranges=[1000])

    def test_zero_frequency_rejected(self):
        with pytest.raises(ConfigurationError, match="frequencies"):
            uacpy.Source(depths=50, frequencies=0)

    def test_negative_frequency_rejected(self):
        with pytest.raises(ConfigurationError, match="frequencies"):
            uacpy.Source(depths=50, frequencies=-100)

    def test_zero_environment_depth_rejected(self):
        with pytest.raises(ConfigurationError):
            uacpy.Environment(name='bad', bathymetry=0, ssp=1500)


class TestUnsupportedOperations:
    """Asking a model for something it can't do raises UnsupportedFeatureError."""

    @pytest.mark.requires_binary  # constructs Kraken (resolves its binary)
    def test_kraken_does_not_support_rays(self):
        kraken = Kraken(verbose=False)
        env = uacpy.Environment(name='t', bathymetry=100, ssp=1500)
        source = uacpy.Source(depths=50, frequencies=100)
        receiver = uacpy.Receiver(depths=[10], ranges=[1000])
        with pytest.raises(UnsupportedFeatureError):
            kraken.compute_rays(env, source, receiver)


class TestFieldErrors:
    """Result classes refuse operations that don't apply to their shape."""

    def test_rays_has_no_sel(self):
        from uacpy.core.results import Rays
        r = Rays(rays=[], model='Bellhop')
        with pytest.raises(AttributeError):
            r.at(range=1000, depth=50)

    def test_rays_has_no_to_db(self):
        from uacpy.core.results import Rays
        r = Rays(rays=[], model='Bellhop')
        with pytest.raises(AttributeError):
            r.to_db()


class TestErrorMessages:
    """Error messages should include enough information to act on."""

    def test_invalid_depth_message_helpful(self):
        error = InvalidDepthError(depth=150, max_depth=100, context="Source")
        msg = str(error)
        assert "150" in msg and "100" in msg

    def test_unsupported_feature_lists_alternatives(self):
        error = UnsupportedFeatureError(
            'Kraken', 'ray-path computation', alternatives=['Bellhop'],
        )
        assert 'Bellhop' in str(error)


class TestValidationHelpers:
    """validate_inputs raises typed errors, not bare ValueError."""

    def test_source_deeper_than_env_raises_typed(self, simple_env):
        from uacpy.models.base import PropagationModel

        Model = type('M', (PropagationModel,), {
            'run': lambda self, env, source, receiver, run_mode=None: None,
        })
        m = Model()
        source_deep = uacpy.Source(depths=150, frequencies=100)
        receiver = uacpy.Receiver(depths=[50], ranges=[1000])
        with pytest.raises(InvalidDepthError):
            m.validate_inputs(simple_env, source_deep, receiver)

    def test_receiver_deeper_than_env_warns_not_raises(self, simple_env):
        """Receivers below the model's resolvable depth are accepted with a
        warning (the model returns its below-domain value there), not
        rejected — unlike the source, which is a hard error."""
        import warnings
        from uacpy.models.base import PropagationModel

        Model = type('M', (PropagationModel,), {
            'run': lambda self, env, source, receiver, run_mode=None: None,
        })
        m = Model()
        source = uacpy.Source(depths=50, frequencies=100)
        receiver_deep = uacpy.Receiver(depths=[150], ranges=[1000])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m.validate_inputs(simple_env, source, receiver_deep)
        assert any("below the model's resolvable depth" in str(w.message)
                   for w in caught)


# Typed exceptions must survive pickling so run_parallel returns the real
# per-job error instead of a BrokenProcessPool (these override __init__ with
# multi-positional / keyword-only signatures and need __reduce__).
@pytest.mark.parametrize("exc", [
    InvalidDepthError(99999.0, 4482.0, "Source depth"),
    ExecutableNotFoundError("Bellhop", "bellhop.exe", ["/a", "/b"]),
    ModelExecutionError("Kraken", -6, stdout="o", stderr="e"),
    UnsupportedFeatureError("Kraken", "elastic ice surface", ["Bellhop"]),
    UnsupportedFeatureError("OASR", "freq override", ["OASP"],
                            alternatives_label='run modes'),
])
def test_typed_exceptions_pickle_roundtrip(exc):
    back = pickle.loads(pickle.dumps(exc))
    assert type(back) is type(exc) and str(back) == str(exc)
