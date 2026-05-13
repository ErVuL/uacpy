"""Signal processing and generation tools for acoustic signals.

Named ``acoustic_signal`` so it does not collide with Python's stdlib
``signal`` module.
"""

from .generation import (
    bpsk_modulate,
    gaussian_pulse,
    hfm_chirp,
    lfm_chirp,
    ricker_wavelet,
    ssrp,
    tone_burst,
)

from .analysis import PPSD, PSD, FRF, SEL, FKTransform, Spectrogram

from .processing import (
    add_noise,
    beamform,
    fourier_synthesis,
    make_bandlimited_noise,
    planewave_rep,
)

from . import analysis, generation, processing

__all__ = [
    "bpsk_modulate",
    "gaussian_pulse",
    "hfm_chirp",
    "lfm_chirp",
    "ricker_wavelet",
    "ssrp",
    "tone_burst",
    "PPSD",
    "PSD",
    "FRF",
    "SEL",
    "FKTransform",
    "Spectrogram",
    "add_noise",
    "beamform",
    "fourier_synthesis",
    "make_bandlimited_noise",
    "planewave_rep",
    "analysis",
    "generation",
    "processing",
]
