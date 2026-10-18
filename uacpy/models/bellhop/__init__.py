"""
Bellhop ray tracing model (Fortran / C++ / CUDA backends)

Supports broadband results via the arrivals-based approach described in the
Bellhop User Guide (Section 9). The workflow:
1. Run Bellhop in arrivals mode ('A') at the center frequency
2. BROADBAND: build the transfer function H(f) from the arrival
   amplitudes, phases and delays
3. TIME_SERIES: delay-and-sum the source pulse over the arrivals
   (``Arrivals.to_time_series``)

This is a key advantage of ray/beam models: broadband results from a
single run, since ray travel times are frequency-independent (geometric).
"""

from uacpy.models.bellhop._model import Bellhop
from uacpy.models.bellhop._settings import BellhopSettings

__all__ = ['Bellhop', 'BellhopSettings']
