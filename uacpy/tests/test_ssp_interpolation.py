"""SSP-interpolation method-focused tests."""

import pytest
import numpy as np

from uacpy.core.environment import SoundSpeedProfile
from uacpy import Field
from uacpy.models import Bellhop
from uacpy.core import Environment, Receiver

pytestmark = pytest.mark.requires_binary


class TestSSPInterpolationMethods:
    """End-to-end acceptance of each ``interp_ssp`` scheme.

    Each case writes a deck with a different ``TopOpt(1)`` character and runs
    the binary, so a scheme that produced an unwritable deck (or one AT
    rejects) fails here. Nothing compares the fields, so these do **not**
    show that the chosen scheme changed the interpolation — only that the
    whole path is wired. The character mapping itself is pinned in
    ``test_ssp_presets.py``.
    """

    @pytest.fixture
    def receiver(self):
        return Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.array([1000.0, 3000.0])
        )

    @pytest.mark.requires_binary
    def test_ssp_isovelocity(self, source, receiver):
        """An isovelocity env forces ``TopOpt(1)='C'`` whatever the model's
        ``interp_ssp`` says, so this is the shortcut path."""
        env = Environment(
            name="iso_test",
            bathymetry=100.0,
            ssp=1500.0
        )

        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=env, source=source, receiver=receiver)
        assert isinstance(result, Field)

    @pytest.mark.requires_binary
    def test_ssp_linear(self, source, receiver):
        """Bellhop with linear-connected SSP samples."""
        depths = np.array([0, 50, 100])
        speeds = np.array([1500, 1490, 1480])

        env = Environment(
            name="linear_test",
            bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs(np.column_stack([depths, speeds])),
        )

        bellhop = Bellhop(verbose=False, interp_ssp='linear')
        result = bellhop.compute_tl(env=env, source=source, receiver=receiver)
        assert isinstance(result, Field)

    @pytest.mark.requires_binary
    def test_ssp_cubic(self, source, receiver):
        """Bellhop with cubic-spline-connected SSP samples."""
        depths = np.array([0, 25, 50, 75, 100])
        speeds = np.array([1500, 1495, 1490, 1485, 1480])

        env = Environment(
            name="cubic_test",
            bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs(np.column_stack([depths, speeds])),
        )

        bellhop = Bellhop(verbose=False, interp_ssp='cubic')
        result = bellhop.compute_tl(env=env, source=source, receiver=receiver)
        assert isinstance(result, Field)
