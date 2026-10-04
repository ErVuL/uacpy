"""
Bellhop ray tracing model (Fortran / C++ / CUDA backends)

Supports broadband time-series generation via the arrivals-based approach
described in the Bellhop User Guide (Section 9). The workflow:
1. Run Bellhop in arrivals mode ('A') at the center frequency
2. Build frequency-domain transfer function H(f) from arrival
   amplitudes, phases, and delays
3. IFFT to time domain, optionally convolved with a source waveform

This is a key advantage of ray/beam models: broadband results from a
single run, since ray travel times are frequency-independent (geometric).
"""

from uacpy.models.bellhop._model import Bellhop
from uacpy.models.bellhop._settings import BellhopSettings

__all__ = ['Bellhop', 'BellhopSettings']
