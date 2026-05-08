"""Tests for exception handling and public exception types."""

import pytest
import numpy as np

import uacpy
from uacpy.core.exceptions import (
    UACPYError, ExecutableNotFoundError, InvalidDepthError,
    UnsupportedFeatureError, ConfigurationError, ModelExecutionError,
)
from uacpy.models import Bellhop, Kraken


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
        with pytest.raises(ValueError, match="source depths must be"):
            uacpy.Source(depths=-10, frequencies=100)

    def test_negative_receiver_depth(self):
        with pytest.raises(ValueError, match="receiver depths must be"):
            uacpy.Receiver(depths=[-10], ranges=[1000])

    def test_zero_frequency_rejected(self):
        with pytest.raises(ValueError, match="frequencies"):
            uacpy.Source(depths=50, frequencies=0)

    def test_negative_frequency_rejected(self):
        with pytest.raises(ValueError, match="frequencies"):
            uacpy.Source(depths=50, frequencies=-100)

    def test_zero_environment_depth_rejected(self):
        with pytest.raises(ValueError):
            uacpy.Environment(name='bad', depth=0, sound_speed=1500)


class TestUnsupportedOperations:
    """Asking a model for something it can't do raises UnsupportedFeatureError."""

    def test_kraken_does_not_support_rays(self):
        kraken = Kraken(verbose=False)
        env = uacpy.Environment(name='t', depth=100, sound_speed=1500)
        source = uacpy.Source(depths=50, frequencies=100)
        receiver = uacpy.Receiver(depths=[10], ranges=[1000])
        with pytest.raises(UnsupportedFeatureError):
            kraken.compute_rays(env, source, receiver)


class TestFieldErrors:
    """Result classes refuse operations that don't apply to their shape."""

    def test_rays_has_no_get_value(self):
        from uacpy.core.results import Rays
        r = Rays(rays=[], model='Bellhop')
        with pytest.raises(AttributeError):
            r.get_value(range_m=1000, depth=50)

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
            'run': lambda self, *a, **kw: None,
        })
        m = Model()
        source_deep = uacpy.Source(depths=150, frequencies=100)
        receiver = uacpy.Receiver(depths=[50], ranges=[1000])
        with pytest.raises(InvalidDepthError):
            m.validate_inputs(simple_env, source_deep, receiver)

    def test_receiver_deeper_than_env_raises_typed(self, simple_env):
        from uacpy.models.base import PropagationModel

        Model = type('M', (PropagationModel,), {
            'run': lambda self, *a, **kw: None,
        })
        m = Model()
        source = uacpy.Source(depths=50, frequencies=100)
        receiver_deep = uacpy.Receiver(depths=[150], ranges=[1000])
        with pytest.raises(InvalidDepthError):
            m.validate_inputs(simple_env, source, receiver_deep)
