"""Synthetic ``Field`` builders shared by the result test files.

Not named ``test_*``, so pytest does not collect it.
"""

import numpy as np
from uacpy.core.results import Arrivals
from uacpy.core.results import Field
from uacpy.core.results import PhaseReference
from uacpy.core.results import ReflectionCoefficient


def _field(data=None, **meta):
    depths = np.array([0.0, 10.0, 20.0, 30.0])
    ranges = np.array([100.0, 200.0, 300.0])
    if data is None:
        data = np.arange(12.0).reshape(4, 3) + 1.0
    return Field(data=data, coords={'depth': depths, 'range': ranges},
                 model='Test', **meta)


# ── one decision per quantity fact: unit, coherence, identity ───────────────

def _two_path_grid(ranges=(150.0, 300.0), df=1.0, f_lo=100.0, f_hi=300.0):
    """``H = e^{-2πifr/c}/r + 0.5·e^{-2πif(r+40)/c}/r`` on a ``(2, n_r, n_f)``
    grid, c = 1500 m/s, stamped travelling-wave like a model's."""
    f = np.arange(f_lo, f_hi, df)
    r = np.asarray(ranges, dtype=float)[:, None]
    cell = (np.exp(-2j * np.pi * f * r / 1500.0)
            + 0.5 * np.exp(-2j * np.pi * f * (r + 40.0) / 1500.0)) / r
    return Field(data=np.broadcast_to(cell, (2,) + cell.shape).copy(),
                 coords={'depth': np.array([10.0, 20.0]),
                         'range': r.ravel(), 'frequency': f},
                 frequencies=f, phase_reference=PhaseReference.TRAVELLING_WAVE)


# ── one result of each component type, for the netCDF round trips ─────────

def _bounce():
    return ReflectionCoefficient(
        angles=[10.0, 20.0, 30.0], magnitude=[[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]],
        phase=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], frequencies=[50.0, 60.0],
        model='Bounce')


def _arrivals(**kw):
    cell = {'amplitudes': np.array([1.0, 0.5]), 'phases': np.array([0.1, 0.2]),
            'delays': np.array([0.10, 0.12]),
            'delays_imag': np.array([0.0, -1e-6]),
            'source_angles': np.array([5.0, -7.0]),
            'receiver_angles': np.array([-5.0, 7.0]),
            'n_top_bounces': np.array([0, 1], dtype='int32'),
            'n_bot_bounces': np.array([1, 1], dtype='int32'), 'n_arrivals': 2}
    return Arrivals(by_receiver=[[[cell]]], receiver_depths=[20.0],
                    receiver_ranges=[1000.0], frequencies=1000.0,
                    source_depths=[5.0], model='Bellhop', **kw)


def _broadband(**kw):
    return Field(data=np.arange(4.0).reshape(1, 1, 4) * (1 - 1j),
                 coords={'depth': np.array([20.0]),
                         'range': np.array([1000.0]),
                         'frequency': np.array([90.0, 100.0, 110.0, 120.0])},
                 kind='pressure', unit='Pa', model='Bellhop',
                 frequencies=[90.0, 100.0, 110.0, 120.0], source_depths=[5.0],
                 phase_reference='travelling_wave', **kw)
