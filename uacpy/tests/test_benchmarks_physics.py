"""Tier: physical-formula benchmarks against published equations and their
documented check values.

Unlike the propagation benchmarks, these need **no native binary** — they
validate uacpy's analytic ocean-physics formulas (sound speed, …) against the
canonical published equation and the literature check value. The check value is
an external number from the source paper (so it validates the coefficients
themselves); the table then asserts uacpy reproduces the full equation at many
points, catching any single-coefficient transcription error.

References
----------
* K. V. Mackenzie, "Nine-term equation for sound speed in the oceans,"
  J. Acoust. Soc. Am. 70(3), 807-812 (1981). Published check value:
  c = 1550.744 m/s at T = 25 °C, S = 35 ppt, D = 1000 m.
"""
import pytest

pytestmark = [pytest.mark.benchmark]

from uacpy.core.acoustics import soundspeed


def _mackenzie(T, S, D):
    """Independent transcription of the Mackenzie (1981) nine-term equation
    for sound speed [m/s]; T [°C], S [ppt], D [m]."""
    return (1448.96 + 4.591 * T - 5.304e-2 * T**2 + 2.374e-4 * T**3
            + 1.340 * (S - 35) + 1.630e-2 * D + 1.675e-7 * D**2
            - 1.025e-2 * T * (S - 35) - 7.139e-13 * T * D**3)


def test_mackenzie_published_check_value():
    """uacpy.soundspeed (Mackenzie 1981) reproduces the paper's check value
    c = 1550.744 m/s at T=25 °C, S=35 ppt, D=1000 m — the standard way a
    Mackenzie implementation is validated."""
    c = soundspeed(temperature=25.0, salinity=35.0, depth=1000.0)
    assert c == pytest.approx(1550.744, abs=1e-3), f"got {c} m/s"


def test_soundspeed_matches_mackenzie_equation_table():
    """uacpy.soundspeed reproduces the published nine-term Mackenzie equation
    across a (T, S, D) table — an independent-transcription cross-check that
    catches a wrong coefficient (which would shift only some rows)."""
    table = [
        (0.0, 35.0, 0.0), (10.0, 35.0, 0.0), (20.0, 35.0, 0.0),
        (25.0, 35.0, 1000.0), (4.0, 34.0, 2000.0), (15.0, 30.0, 500.0),
        (30.0, 40.0, 100.0), (2.0, 35.0, 4000.0),
    ]
    for T, S, D in table:
        got = soundspeed(temperature=T, salinity=S, depth=D)
        assert got == pytest.approx(_mackenzie(T, S, D), abs=1e-6), \
            f"T={T},S={S},D={D}: uacpy={got} vs Mackenzie eq {_mackenzie(T, S, D)}"


@pytest.mark.requires_binary
@pytest.mark.parametrize('model_name', ['Kraken', 'Scooter'])
def test_constant_absorption_adds_the_expected_loss(model_name):
    """A ConstantAbsorption baseline must attenuate as (r/lambda) * value.

    AT reads an SSP line as z, alphaR, betaR, rhoR, alphaI, betaI
    (misc/sspMod.f90:334), so the baseline belongs in the fifth column. Emitted
    third it becomes the water column's shear speed, which Kraken resolves to
    an all-NaN field and Scooter segfaults on — this asserts the physical
    effect, which a finiteness check alone would miss.
    """
    import numpy as np
    import uacpy
    from uacpy.core.absorption import ConstantAbsorption

    f, c, value = 100.0, 1500.0, 0.05
    ranges = np.array([1000.0, 2000.0, 3000.0])
    model_cls = getattr(uacpy, model_name)

    def env(a):
        return uacpy.Environment(
            name='abs', bathymetry=200.0, ssp=c,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0,
                density=1.8, attenuation=0.5),
            absorption=ConstantAbsorption(a) if a is not None else None)

    src = uacpy.Source(depths=50.0, frequencies=f)
    rcv = uacpy.Receiver(depths=100.0, ranges=ranges)
    tl0 = np.asarray(model_cls(verbose=False).run(env(None), src, rcv).tl).ravel()
    tl1 = np.asarray(model_cls(verbose=False).run(env(value), src, rcv).tl).ravel()
    extra = tl1 - tl0
    predicted = ranges / (c / f) * value

    assert np.all(np.isfinite(extra)), f"{model_name} produced non-finite TL"
    # Modes travel at grazing angles, so the path exceeds the horizontal
    # range and the measured loss sits at or just above the prediction.
    assert np.all(extra > 0.9 * predicted), f"{extra} vs {predicted}"
    assert np.all(extra < 1.4 * predicted), f"{extra} vs {predicted}"
