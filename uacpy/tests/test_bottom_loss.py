"""Tests for the plane-wave bottom-loss helper
:func:`uacpy.core.acoustics.bottom_loss_curve` and the matching plot
helper."""

import numpy as np
import pytest

from uacpy.core.acoustics import bottom_loss_curve


class TestBottomLossCurve:
    def test_returns_matched_arrays(self):
        ang, loss = bottom_loss_curve('sand')
        assert ang.shape == loss.shape
        assert ang[0] == 0.0
        assert ang[-1] == 90.0

    def test_grazing_perfect_reflection(self):
        # At grazing=0° all energy reflects → zero loss; at normal
        # incidence some transmits → finite loss for a fluid bottom
        # denser & faster than water.
        _, loss = bottom_loss_curve('sand')
        assert loss[0] == pytest.approx(0.0, abs=1e-6)
        assert loss[-1] > 0.5

    def test_subcritical_loss_smaller_than_supercritical(self):
        # Below the critical angle ray bends fully back into water →
        # loss is small; above it (up to normal incidence) some energy
        # transmits → larger loss. For fluid 'sand' c_2=1650, c_1=1500
        # the critical-angle complement is arccos(c1/c2) ≈ 24.6°
        # measured from the interface.
        ang, loss = bottom_loss_curve('sand')
        below = loss[ang < 20.0].mean()
        above = loss[(ang > 30.0) & (ang < 60.0)].mean()
        assert below < above

    def test_dict_input_works(self):
        m = dict(sound_speed=1700.0, density=1.8, attenuation=0.3)
        ang, loss = bottom_loss_curve(m)
        assert np.all(np.isfinite(loss))

    def test_custom_angle_grid(self):
        custom = np.linspace(5.0, 85.0, 41)
        ang, loss = bottom_loss_curve('limestone', grazing_angles_deg=custom)
        assert np.array_equal(ang, custom)


