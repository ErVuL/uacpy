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
