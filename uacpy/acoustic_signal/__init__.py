"""Signal processing and generation tools for acoustic signals.

Named ``acoustic_signal`` so it does not collide with Python's stdlib
``signal`` module. Sub-modules, each with one responsibility:

* ``waveforms`` / ``sequences`` / ``noise_synthesis`` — deterministic pulses,
  coded probe sequences, and stochastic/Fourier synthesis
* ``arrays``     — steering vectors + conventional/adaptive beamforming
* ``active``     — matched filter / pulse compression / ambiguity
* ``transforms`` — f-k, tau-p, Radon (gather transforms + inverses)
* ``timefreq``   — Hilbert, spectrogram, wavelet, Wigner-Ville, cepstrum
* ``analysis``   — PSD / PPSD / SEL spectral & level estimators
* ``system_id``  — FRF (transfer-function) estimation
* ``channel``    — time-domain channel simulation
* ``modal``      — modal / dispersion (waveguide warping)

Class vs. function convention
-----------------------------
Several transforms/estimators expose **both** a CapWords class and a lower_case
function for the same math (e.g. ``FK`` / ``Radon`` / ``TauP`` / ``Spectrogram``
/ ``PSD`` vs. ``radon_transform`` / ``taup_transform`` / ...). They are not
redundant — pick by use:

* **Class** — a *configurable, reusable estimator*. Construct once with its
  parameters (window, nfft, slowness grid, ...), then call it on many inputs;
  it caches/validates configuration and carries result metadata.
* **Function** — a *one-shot* call: transform a single array with the defaults
  (or a few inline kwargs) and get the array back, no object to keep.
"""

from .waveforms import (
    sparc_pulse,
    gaussian_pulse,
    hfm_chirp,
    lfm_chirp,
    nwave,
    ricker_wavelet,
    tone_burst,
)
from .sequences import (
    bpsk_modulate,
    make_mseq_probe,
    mseq,
)
from .noise_synthesis import (
    add_noise,
    fourier_synthesis,
    make_bandlimited_noise,
    make_noise_waveform,
    synthesize_noise_from_psd,
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
    shading_taper,
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
    waveforms,
    sequences,
    noise_synthesis,
    modal,
    system_id,
    timefreq,
    transforms,
)

__all__ = [
    # generation
    "synthesize_noise_from_psd", "lfm_chirp", "hfm_chirp", "tone_burst",
    "gaussian_pulse", "ricker_wavelet", "bpsk_modulate", "add_noise",
    "make_bandlimited_noise", "fourier_synthesis", "sparc_pulse", "nwave",
    "mseq", "make_mseq_probe", "make_noise_waveform",
    # spectral / level estimators
    "PSD", "PPSD", "SEL",
    # system identification
    "FRF",
    # arrays
    "steering_vectors", "beamform", "sample_covariance", "bartlett_spectrum",
    "mvdr_spectrum", "music_spectrum", "shading_taper",
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
    "waveforms", "sequences", "noise_synthesis", "arrays", "active",
    "transforms", "timefreq", "analysis", "system_id", "channel", "modal",
    "bands",
]
