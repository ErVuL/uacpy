"""
Signal processing and generation module for UACPY

Provides tools for generating and processing acoustic signals.
"""

import logging as _logging

_log = _logging.getLogger(__name__)

# Import from generation module (waveforms)
try:
    from .generation import (
        cw_pulse,
        lfm_chirp,
        hfm_chirp,
        mfsk_signal,
        psk_signal,
        noise_signal,
        composite_signal,
    )
except ImportError as _e:
    _log.debug("Could not import generation module: %s", _e)

# Import from processing module (generic signal processing)
try:
    from .processing import (
        planewave_rep,
        beamform,
        add_noise,
        make_bandlimited_noise,
        fourier_synthesis,
    )
except ImportError as _e:
    _log.debug("Could not import processing module: %s", _e)

# Import from advanced signal processing module
try:
    from .advanced import (
        time,
        cw,
        sweep,
        bb2pb,
        pb2bb,
        lfilter0,
        correlate_periodic,
        goertzel,
        nco,
        nco_gen,
        lfilter_gen,
        mseq,
        resample,
    )
except ImportError as _e:
    _log.debug("Could not import advanced module: %s", _e)

# Also keep submodules accessible
try:
    from . import advanced
    from . import analysis
    from . import generation
    from . import processing
except ImportError as _e:
    _log.debug("Could not import submodules: %s", _e)

__all__ = [
    # Waveform generation
    "cw_pulse",
    "lfm_chirp",
    "hfm_chirp",
    "mfsk_signal",
    "psk_signal",
    "noise_signal",
    "composite_signal",
    # Signal processing
    "planewave_rep",
    "beamform",
    "add_noise",
    "make_bandlimited_noise",
    "fourier_synthesis",
    # Advanced signal processing
    "time",
    "cw",
    "sweep",
    "bb2pb",
    "pb2bb",
    "lfilter0",
    "correlate_periodic",
    "goertzel",
    "nco",
    "nco_gen",
    "lfilter_gen",
    "mseq",
    "resample",
    # Modules
    "advanced",
    "analysis",
    "generation",
    "processing",
]
