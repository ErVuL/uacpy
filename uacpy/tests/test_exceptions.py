"""
Tests for exception handling and error messages
"""

import pytest
import numpy as np
import uacpy
from uacpy.core.exceptions import (
    UACPYError, ModelError, ExecutableNotFoundError,
    EnvironmentError, InvalidDepthError, UnsupportedFeatureError,
)
from uacpy.models import Bellhop, Kraken


class TestCustomExceptions:
    """Tests for custom exception classes."""

    def test_uacpy_error_is_base(self):
        """Test that UACPYError is base exception."""
        error = UACPYError("Test message")
        assert isinstance(error, Exception)
        assert str(error) == "Test message"

    def test_executable_not_found_error(self):
        """Test ExecutableNotFoundError."""
        error = ExecutableNotFoundError("TestModel", "test.exe")
        assert "TestModel" in str(error)
        assert "test.exe" in str(error)
        assert hasattr(error, 'remediation')

    def test_invalid_depth_error(self):
        """Test InvalidDepthError."""
        error = InvalidDepthError(depth=150, max_depth=100, context="Source")
        assert "150" in str(error)
        assert "100" in str(error)
        assert "Source" in str(error)

    def test_unsupported_feature_error(self):
        """Test UnsupportedFeatureError."""
        error = UnsupportedFeatureError("TestModel", "feature_name", alternatives=['Model1', 'Model2'])
        assert "TestModel" in str(error)
        assert "feature_name" in str(error)
        assert "Model1" in str(error)


class TestInputValidation:
    """Tests for input validation and error handling."""

    def test_source_depth_exceeds_environment(self, simple_env):
        """Test error when source depth exceeds environment depth."""
        source = uacpy.Source(depth=150, frequency=100)
        receiver = uacpy.Receiver(depths=[50], ranges=[1000])

        bellhop = Bellhop(verbose=False)

        with pytest.raises(ValueError, match="Source depth.*exceeds"):
            bellhop.validate_inputs(simple_env, source, receiver)

    def test_receiver_depth_exceeds_environment(self, simple_env):
        """Test error when receiver depth exceeds environment depth."""
        source = uacpy.Source(depth=50, frequency=100)
        receiver = uacpy.Receiver(depths=[150], ranges=[1000])

        bellhop = Bellhop(verbose=False)

        with pytest.raises(ValueError, match="Receiver depth.*exceeds"):
            bellhop.validate_inputs(simple_env, source, receiver)

    def test_negative_source_depth(self, simple_env):
        """Test error for negative source depth."""
        # Source constructor should raise error for negative depth
        with pytest.raises(ValueError, match="Source depths must be positive"):
            source = uacpy.Source(depth=-10, frequency=100)

    def test_negative_receiver_depth(self, simple_env):
        """Test error for negative receiver depth."""
        # Receiver constructor should raise error for negative depth
        with pytest.raises(ValueError, match="Receiver depths must be positive"):
            receiver = uacpy.Receiver(depths=[-10], ranges=[1000])


class TestUnsupportedOperations:
    """Tests for unsupported model operations."""

    @pytest.mark.requires_binary
    def test_kraken_rays_unsupported(self, simple_env, source):
        """Test that Kraken raises error for ray computation."""
        kraken = Kraken(verbose=False)

        with pytest.raises(UnsupportedFeatureError) as exc_info:
            kraken.compute_rays(env=simple_env, source=source)

        assert "ray path computation" in str(exc_info.value)
        assert "Bellhop" in str(exc_info.value)  # Should suggest Bellhop

    @pytest.mark.requires_binary
    def test_bellhop_modes_unsupported(self, simple_env, source):
        """Test that Bellhop raises error for mode computation."""
        bellhop = Bellhop(verbose=False)

        with pytest.raises(UnsupportedFeatureError) as exc_info:
            bellhop.compute_modes(env=simple_env, source=source)

        assert "normal mode computation" in str(exc_info.value)
        assert "Kraken" in str(exc_info.value) or "OASN" in str(exc_info.value)

    @pytest.mark.requires_binary
    def test_kraken_range_dependent_error(self, range_dependent_env, source):
        """Test that Kraken raises error for range-dependent environment."""
        kraken = Kraken(verbose=False)

        # Should raise EnvironmentError for range-dependent environments
        with pytest.raises(EnvironmentError, match="does not support range-dependent"):
            modes = kraken.compute_modes(env=range_dependent_env, source=source)


class TestFieldErrors:
    """Tests for Field object error handling."""

    def test_invalid_field_type(self):
        """Test error for invalid field type."""
        from uacpy.core.field import Field

        with pytest.raises(ValueError, match="field_type must be one of"):
            Field(
                field_type='invalid_type',
                data=np.random.rand(10, 20)
            )

    def test_field_get_value_on_rays(self):
        """Test that get_value raises error for ray fields."""
        from uacpy.core.field import Field

        field = Field(
            field_type='rays',
            data=np.array([]),
            metadata={'rays': []}
        )

        with pytest.raises(ValueError, match="get_value not supported"):
            field.get_value(range_m=1000, depth=50)

    def test_field_to_db_unsupported(self):
        """Test that to_db raises error for unsupported field types."""
        from uacpy.core.field import Field

        field = Field(
            field_type='rays',
            data=np.array([])
        )

        with pytest.raises(ValueError, match="to_db not supported"):
            field.to_db()


class TestErrorMessages:
    """Tests that error messages are helpful."""

    def test_executable_not_found_message_helpful(self):
        """Test that executable not found error has helpful message."""
        error = ExecutableNotFoundError("Bellhop", "bellhop.exe")

        error_str = str(error)
        assert "Bellhop" in error_str
        assert "bellhop.exe" in error_str

        # Check remediation is helpful
        assert error.remediation is not None
        assert "install" in error.remediation.lower()

    def test_unsupported_feature_suggests_alternatives(self):
        """Test that unsupported feature error suggests alternatives."""
        error = UnsupportedFeatureError(
            "Bellhop",
            "normal modes",
            alternatives=['Kraken', 'OASN']
        )

        error_str = str(error)
        assert "Kraken" in error_str or "OASN" in error_str

    def test_invalid_depth_message_helpful(self):
        """Test that invalid depth error has helpful message."""
        error = InvalidDepthError(depth=150, max_depth=100, context="Source")

        error_str = str(error)
        assert "150" in error_str
        assert "100" in error_str
        assert "Source" in error_str

        # Check remediation
        assert error.remediation is not None
        assert "100" in error.remediation


class TestValidationHelpers:
    """Tests for validation helper functions."""

    def test_validate_source_depth(self):
        """Test validate_source_depth helper."""
        from uacpy.core.model_utils import validate_source_depth

        # Valid depth - should not raise
        validate_source_depth(50.0, 100.0)

        # Invalid depth - should raise
        with pytest.raises(InvalidDepthError):
            validate_source_depth(150.0, 100.0)

    def test_validate_receiver_depths(self):
        """Test validate_receiver_depths helper."""
        from uacpy.core.model_utils import validate_receiver_depths

        # Valid depths - should not raise
        validate_receiver_depths(np.array([25, 50, 75]), 100.0)

        # Invalid depths - should raise
        with pytest.raises(InvalidDepthError):
            validate_receiver_depths(np.array([25, 50, 150]), 100.0)
