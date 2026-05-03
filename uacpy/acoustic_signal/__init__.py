"""Signal processing and generation tools for acoustic signals.

The package is named ``acoustic_signal`` so it does not collide with Python's
own ``signal`` module; ``uacpy`` re-exposes it as ``uacpy.signal``.

Top-level exports below are a curated subset of the most commonly-used
helpers from the four submodules. Everything in the submodules is also
reachable directly (e.g. ``uacpy.signal.generation.tone_burst``).
"""

# --- Curated waveform generators ---------------------------------------------
# Source: acoustic_signal/generation.py
from .generation import (
    bpsk_modulate,
    gaussian_pulse,
    hfm_chirp,
    lfm_chirp,
    ricker_wavelet,
    tone_burst,
)

# --- Curated processing helpers ----------------------------------------------
# Source: acoustic_signal/processing.py
from .processing import (
    add_noise,
    beamform,
    fourier_synthesis,
    make_bandlimited_noise,
    planewave_rep,
)

# --- Curated DSP / utility helpers -------------------------------------------
# Source: acoustic_signal/advanced.py
# Note: `mseq` is defined in both generation.py and advanced.py with different
# signatures; we expose the advanced.py version at the top level (more general,
# accepts either degree or polynomial spec). The generation.py version is still
# reachable via ``uacpy.signal.generation.mseq``.
from .advanced import (
    bb2pb,
    correlate_periodic,
    cw,
    goertzel,
    lfilter0,
    lfilter_gen,
    mseq,
    nco,
    nco_gen,
    pb2bb,
    resample,
    sweep,
    time,
)

# --- Submodules (also reachable as attributes) -------------------------------
from . import advanced, analysis, generation, processing

__all__ = [
    # Generators
    "bpsk_modulate",
    "gaussian_pulse",
    "hfm_chirp",
    "lfm_chirp",
    "ricker_wavelet",
    "tone_burst",
    # Processing
    "add_noise",
    "beamform",
    "fourier_synthesis",
    "make_bandlimited_noise",
    "planewave_rep",
    # Advanced / DSP
    "bb2pb",
    "correlate_periodic",
    "cw",
    "goertzel",
    "lfilter0",
    "lfilter_gen",
    "mseq",
    "nco",
    "nco_gen",
    "pb2bb",
    "resample",
    "sweep",
    "time",
    # Submodules
    "advanced",
    "analysis",
    "generation",
    "processing",
]
