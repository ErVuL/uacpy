"""Signal processing and generation tools for acoustic signals.

Named ``acoustic_signal`` so it does not collide with Python's stdlib
``signal`` module. Sub-modules, each with one responsibility:

* ``waveforms`` / ``sequences`` / ``noise_synthesis`` — deterministic pulses,
  coded probe sequences, and stochastic/Fourier synthesis
* ``arrays``     — steering vectors + conventional/adaptive beamforming
* ``active``     — matched filter / pulse compression / ambiguity
* ``transforms`` — f-k, tau-p, Radon (gather transforms + inverses)
* ``timefreq``   — Hilbert, spectrogram, wavelet, Wigner-Ville, cepstrum
* ``analysis``   — ``psd`` / ``ppsd`` / ``sel`` spectral & level estimators
* ``system_id``  — FRF (transfer-function) estimation
* ``channel``    — time-domain channel simulation
* ``modal``      — modal / dispersion (waveguide warping)

Functional convention
---------------------
Every transform/estimator is a **pure function** returning plain arrays (or a
small data-only namedtuple such as ``PPSDResult``): ``psd`` / ``ppsd`` / ``sel``
/ ``spectrogram`` / ``fk_transform`` / ``radon_transform`` / ``taup_transform``,
with an ``inverse_<name>`` where an inverse is meaningful. Configure via keyword
arguments (``functools.partial`` for the rare configure-once case). This module
imports **no plotting** — all visualisation lives in
:mod:`uacpy.visualization` (``plot_psd``, ``plot_fk``, ``plot_spectrogram`` …).
``FRF`` (``system_id``) remains a class, as it carries fitted state.
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
from .analysis import PPSDResult, PSDResult, SELResult, ppsd, psd, sel
from .system_id import FRF
from .arrays import (
    bartlett_spectrum,
    beamform,
    BeamformResult,
    music_spectrum,
    mvdr_spectrum,
    sample_covariance,
    steering_vectors,
    shading_taper,
)
from .active import (
    AmbiguityResult,
    ambiguity_function,
    matched_filter,
    processing_gain,
    pulse_compression,
)
from .transforms import (
    fk_transform,
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
    spectrogram,
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
from .constant_q import (
    constant_q_transform,
    constant_q_psd,
    constant_q_spectrogram,
    probabilistic_constant_q,
    CQTResult,
    CQPSDResult,
    CQSpectrogramResult,
    CQPPSDResult,
)

from .bands import decidecade_bands, decidecade_band_levels

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
    constant_q,
)

__all__ = [
    # generation
    "synthesize_noise_from_psd", "lfm_chirp", "hfm_chirp", "tone_burst",
    "gaussian_pulse", "ricker_wavelet", "bpsk_modulate", "add_noise",
    "make_bandlimited_noise", "fourier_synthesis", "sparc_pulse", "nwave",
    "mseq", "make_mseq_probe", "make_noise_waveform",
    # spectral / level estimators
    "psd", "ppsd", "PPSDResult", "PSDResult", "SELResult", "sel",
    # system identification
    "FRF",
    # arrays
    "steering_vectors", "beamform", "BeamformResult", "sample_covariance", "bartlett_spectrum",
    "mvdr_spectrum", "music_spectrum", "shading_taper",
    # active
    "matched_filter", "pulse_compression", "processing_gain",
    "ambiguity_function", "AmbiguityResult",
    # transforms (gather)
    "fk_transform", "inverse_fk",
    "taup_transform", "inverse_taup",
    "radon_transform", "inverse_radon",
    # channel
    "impulse_response", "simulate_reception",
    "impulse_response_from_transfer_function",
    # modal
    "modal_group_velocity", "warp_signal", "unwarp_signal",
    # time-frequency
    "spectrogram", "analytic_signal", "envelope", "instantaneous_frequency",
    "wigner_ville", "cwt", "inverse_cwt", "cepstrum", "complex_cepstrum",
    "inverse_complex_cepstrum",
    # constant-Q (Brown 1991)
    "constant_q_transform", "constant_q_psd", "constant_q_spectrogram",
    "probabilistic_constant_q", "CQTResult", "CQPSDResult",
    "CQSpectrogramResult", "CQPPSDResult",
    # decidecade bands (ISO 18405 / IEC 61260-1)
    "decidecade_bands", "decidecade_band_levels",
    # sub-modules
    "waveforms", "sequences", "noise_synthesis", "arrays", "active",
    "transforms", "timefreq", "analysis", "system_id", "channel", "modal",
    "bands", "constant_q",
]
