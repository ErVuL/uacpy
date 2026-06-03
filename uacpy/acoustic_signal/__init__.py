"""Signal processing and generation tools for acoustic signals.

Named ``acoustic_signal`` so it does not collide with Python's stdlib
``signal`` module. Sub-modules, each with one responsibility:

* ``generation`` — waveforms, noise generation, Fourier synthesis
* ``arrays``     — steering vectors + conventional/adaptive beamforming
* ``active``     — matched filter / pulse compression / ambiguity
* ``transforms`` — f-k, tau-p, Radon (gather transforms + inverses)
* ``timefreq``   — Hilbert, spectrogram, wavelet, Wigner-Ville, cepstrum
* ``analysis``   — PSD / PPSD / SEL spectral & level estimators
* ``system_id``  — FRF (transfer-function) estimation
* ``channel``    — time-domain channel simulation
* ``modal``      — modal / dispersion (waveguide warping)
"""

from .generation import (
    add_noise,
    bpsk_modulate,
    fourier_synthesis,
    gaussian_pulse,
    hfm_chirp,
    lfm_chirp,
    make_bandlimited_noise,
    ricker_wavelet,
    ssrp,
    tone_burst,
)
from .analysis import PPSD, PSD, SEL
from .system_id import FRF
from .arrays import (
    bartlett_spectrum,
    beamform,
    music_spectrum,
    mvdr_spectrum,
    sample_covariance,
    steering_vectors,
    taper,
)
from .active import (
    ambiguity_function,
    matched_filter,
    processing_gain,
    pulse_compression,
    shift_to_max_correlation,
)
from .transforms import (
    FK,
    Radon,
    TauP,
    inverse_fk,
    inverse_radon,
    inverse_taup,
    radon_transform,
    taup_transform,
)
from .channel import (
    impulse_response,
    impulse_response_from_transfer_function,
    simulate_reception,
)
from .modal import modal_group_velocity, unwarp_signal, warp_signal
from .timefreq import (
    Spectrogram,
    analytic_signal,
    cepstrum,
    complex_cepstrum,
    cwt,
    envelope,
    instantaneous_frequency,
    inverse_complex_cepstrum,
    inverse_cwt,
    wigner_ville,
)

from .bands import decidecade_bands, decidecade_band_levels, plot_band_levels

from . import (
    active,
    analysis,
    arrays,
    bands,
    channel,
    generation,
    modal,
    system_id,
    timefreq,
    transforms,
)

__all__ = [
    # generation
    "ssrp", "lfm_chirp", "hfm_chirp", "tone_burst", "gaussian_pulse",
    "ricker_wavelet", "bpsk_modulate", "add_noise", "make_bandlimited_noise",
    "fourier_synthesis",
    # spectral / level estimators
    "PSD", "PPSD", "SEL",
    # system identification
    "FRF",
    # arrays
    "steering_vectors", "beamform", "sample_covariance", "bartlett_spectrum",
    "mvdr_spectrum", "music_spectrum", "taper",
    # active
    "matched_filter", "pulse_compression", "processing_gain",
    "ambiguity_function", "shift_to_max_correlation",
    # transforms (gather)
    "FK", "TauP", "Radon", "inverse_fk", "taup_transform", "inverse_taup",
    "radon_transform", "inverse_radon",
    # channel
    "impulse_response", "simulate_reception",
    "impulse_response_from_transfer_function",
    # modal
    "modal_group_velocity", "warp_signal", "unwarp_signal",
    # time-frequency
    "Spectrogram", "analytic_signal", "envelope", "instantaneous_frequency",
    "wigner_ville", "cwt", "inverse_cwt", "cepstrum", "complex_cepstrum",
    "inverse_complex_cepstrum",
    # decidecade bands (ISO 18405 / IEC 61260-1)
    "decidecade_bands", "decidecade_band_levels", "plot_band_levels",
    # sub-modules
    "generation", "arrays", "active", "transforms", "timefreq", "analysis",
    "system_id", "channel", "modal", "bands",
]
