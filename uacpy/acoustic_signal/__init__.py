"""Signal processing and generation tools for acoustic signals."""

import logging as _logging

_log = _logging.getLogger(__name__)

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

try:
    from . import advanced
    from . import analysis
    from . import generation
    from . import processing
except ImportError as _e:
    _log.debug("Could not import submodules: %s", _e)

__all__ = [
    "cw_pulse",
    "lfm_chirp",
    "hfm_chirp",
    "mfsk_signal",
    "psk_signal",
    "noise_signal",
    "composite_signal",
    "planewave_rep",
    "beamform",
    "add_noise",
    "make_bandlimited_noise",
    "fourier_synthesis",
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
    "advanced",
    "analysis",
    "generation",
    "processing",
]
