"""Signal processing and generation tools for acoustic signals.

Named ``acoustic_signal`` so it does not collide with Python's stdlib
``signal`` module. Sub-modules, each with one responsibility:

* ``generate``   — *give me a signal*: parametric waveforms, coded probe
                   sequences, and noise built to a target spectrum
* ``estimate``   — *measure this signal*: one estimator per statistic
                   (``welch``, ``constant_q``, ``sound_exposure`` and a
                   ``probabilistic_`` twin of each), the time-resolved views
                   (``spectrogram``, ``cwt``, ``wigner_ville``, cepstra) and
                   the standard band ladders
* ``arrays``     — *what does this array see*: steering vectors, beamforming,
                   and the gather transforms (f-k, tau-p, Radon), which are
                   the ones that take a receiver spacing ``dx``
* ``detect``     — *is my transmission in there*: matched filter, pulse
                   compression, ambiguity
* ``system``     — *what did the channel do to it*: FRF estimation, channel
                   simulation, modal dispersion and warping

One module per question the package answers. Every public name is re-exported
here, so a caller writes ``uacpy.acoustic_signal.welch`` and never names a
sub-module: these boundaries are for maintainers.

Functional convention
---------------------
Every transform/estimator is a **pure function** returning plain arrays (or a
small data-only namedtuple such as ``SpectralEstimate``):
``welch`` / ``constant_q`` / ``sound_exposure`` / ``spectrogram``
/ ``fk_transform`` / ``radon_transform`` / ``taup_transform``,
with an ``inverse_<name>`` where an inverse is meaningful. Configure via keyword
arguments (``functools.partial`` for the rare configure-once case). This module
imports **no plotting** — all visualisation lives in
:mod:`uacpy.visualization` (``plot_psd``, ``plot_fk``, ``plot_spectrogram`` …).
``FRF`` (``system``) remains a class, as it carries fitted state.

Output dtype
------------
Two estimators preserve a ``float32`` input and the rest promote to
``float64``; nothing here downcasts. Measured on a ``float32`` record:

==============================  ===============================
Welch spectra, ``spectrogram``  ``.power`` stays ``float32``
                                (the axes are always ``float64``)
``envelope``, ``cepstrum``,     promote to ``float64``
the histogram estimators,
``constant_q`` and
``sound_exposure`` estimates,
``wigner_ville``
``analytic_signal``,            ``complex128`` from either input
``fk_transform``
==============================  ===============================

The split follows what each estimator's backend does — the two that preserve
are the scipy ``welch``/``stft`` wrappers — and it is stated rather than
enforced because making the eight agree would change the dtype a Welch
estimate and ``spectrogram`` return today. Stacking a Welch estimate against
a constant-Q one therefore promotes; cast explicitly if a pipeline depends on
the width.

Lazy loading
------------
Sub-modules import on first attribute access (PEP 562), not at package
import. This matters beyond convenience: :mod:`uacpy.io` needs only
``waveforms.sparc_pulse`` (numpy-only), and an eager ``__init__`` here
dragged scipy.signal/scipy.stats into every ``import uacpy``.
"""

import importlib

# Sub-modules, importable on demand as attributes of this package.
_SUBMODULES = frozenset({
    'generate', 'estimate', 'arrays', 'detect', 'system',
})

# Public name -> defining sub-module. Kept in sync with __all__ by the
# consistency check at the bottom of this file (runs at import, costs nothing).
_EXPORTS = {
    # waveforms
    'sparc_pulse': 'generate', 'gaussian_pulse': 'generate',
    'hfm_chirp': 'generate', 'lfm_chirp': 'generate', 'nwave': 'generate',
    'ricker_wavelet': 'generate', 'tone_burst': 'generate',
    # sequences
    'bpsk_modulate': 'generate', 'make_mseq_probe': 'generate',
    'mseq': 'generate',
    # noise_synthesis
    'add_noise': 'generate', 'fourier_synthesis': 'generate',
    'make_bandlimited_noise': 'generate',
    'make_noise_waveform': 'generate',
    'synthesize_noise_from_psd': 'generate',
    # analysis
    'ProbabilisticSpectralEstimate': 'estimate',
    'SpectralEstimate': 'estimate',
    'BAND_TYPES': 'estimate',
    'welch': 'estimate', 'constant_q': 'estimate',
    'probabilistic_welch': 'estimate',
    'probabilistic_constant_q': 'estimate',
    'sound_exposure': 'estimate',
    'probabilistic_sound_exposure': 'estimate',
    # system_id
    'FRF': 'system',
    # arrays
    'bartlett_spectrum': 'arrays', 'beamform': 'arrays',
    'BeamformResult': 'arrays', 'music_spectrum': 'arrays',
    'mvdr_spectrum': 'arrays', 'sample_covariance': 'arrays',
    'steering_vectors': 'arrays', 'shading_taper': 'arrays',
    'beamform_field': 'arrays', 'BeamformedField': 'arrays',
    'plane_wave_array_gain': 'arrays', 'matched_replica_gain': 'arrays',
    'independent_beams': 'arrays',
    # active
    'AmbiguityResult': 'detect', 'ambiguity_function': 'detect',
    'matched_filter': 'detect', 'processing_gain': 'detect',
    'pulse_compression': 'detect',
    # transforms
    'fk_transform': 'arrays', 'inverse_fk': 'arrays',
    'inverse_radon': 'arrays', 'inverse_taup': 'arrays',
    'radon_transform': 'arrays', 'taup_transform': 'arrays',
    'FKResult': 'arrays', 'TauPResult': 'arrays',
    'RadonResult': 'arrays',
    # channel
    'fractional_delay_taps': 'system',
    'impulse_response': 'system',
    'impulse_response_from_transfer_function': 'system',
    'simulate_reception': 'system',
    # modal
    'modal_group_velocity': 'system', 'unwarp_signal': 'system',
    'warp_signal': 'system',
    # timefreq
    'spectrogram': 'estimate', 'analytic_signal': 'estimate',
    'cepstrum': 'estimate', 'ComplexCepstrum': 'estimate',
    'complex_cepstrum': 'estimate', 'cwt': 'estimate', 'envelope': 'estimate',
    'instantaneous_frequency': 'estimate',
    'inverse_complex_cepstrum': 'estimate', 'inverse_cwt': 'estimate',
    'wigner_ville': 'estimate', 'SpectrogramResult': 'estimate',
    'CWTResult': 'estimate', 'WignerVilleResult': 'estimate',
    # the constant-Q transform and spectrogram; the ESTIMATORS live in
    # analysis beside the others, and the module is private so the name
    # ``constant_q`` belongs to the estimator rather than to a module.
    'constant_q_transform': 'estimate',
    'constant_q_spectrogram': 'estimate',
    'CQTResult': 'estimate', 'CQSpectrogramResult': 'estimate',
    # bands
    'decidecade_bands': 'estimate', 'decidecade_band_levels': 'estimate',
}

__all__ = [
    # generation
    "synthesize_noise_from_psd", "lfm_chirp", "hfm_chirp", "tone_burst",
    "gaussian_pulse", "ricker_wavelet", "bpsk_modulate", "add_noise",
    "make_bandlimited_noise", "fourier_synthesis", "sparc_pulse", "nwave",
    "mseq", "make_mseq_probe", "make_noise_waveform",
    # spectral / level estimators
    "welch", "constant_q", "sound_exposure",
    "probabilistic_welch", "probabilistic_constant_q",
    "probabilistic_sound_exposure",
    "SpectralEstimate", "ProbabilisticSpectralEstimate", "BAND_TYPES",
    # system identification
    "FRF",
    # arrays
    "steering_vectors", "beamform", "BeamformResult", "sample_covariance", "bartlett_spectrum",
    "mvdr_spectrum", "music_spectrum", "shading_taper",
    "beamform_field", "BeamformedField", "plane_wave_array_gain",
    "matched_replica_gain", "independent_beams",
    # active
    "matched_filter", "pulse_compression", "processing_gain",
    "ambiguity_function", "AmbiguityResult",
    # transforms (gather)
    "fk_transform", "inverse_fk", "FKResult",
    "taup_transform", "inverse_taup", "TauPResult",
    "radon_transform", "inverse_radon", "RadonResult",
    # channel
    "impulse_response", "simulate_reception",
    "impulse_response_from_transfer_function", "fractional_delay_taps",
    # modal
    "modal_group_velocity", "warp_signal", "unwarp_signal",
    # time-frequency
    "spectrogram", "analytic_signal", "envelope", "instantaneous_frequency",
    "wigner_ville", "cwt", "inverse_cwt", "cepstrum", "ComplexCepstrum",
    "complex_cepstrum",
    "inverse_complex_cepstrum",
    "SpectrogramResult", "CWTResult", "WignerVilleResult",
    # constant-Q (Brown 1991)
    "constant_q_transform", "constant_q_spectrogram",
    "CQTResult", "CQSpectrogramResult",
    # decidecade bands (ISO 18405 / IEC 61260-1)
    "decidecade_bands", "decidecade_band_levels",
    # sub-modules
    "generate", "estimate", "arrays", "detect", "system",
]

if set(__all__) != set(_EXPORTS) | _SUBMODULES:
    raise RuntimeError(
        "acoustic_signal.__all__ is out of sync with the lazy-export tables; "
        f"missing from tables: {sorted(set(__all__) - set(_EXPORTS) - _SUBMODULES)}, "
        f"missing from __all__: {sorted((set(_EXPORTS) | _SUBMODULES) - set(__all__))}"
    )


def __getattr__(name):
    if name in _SUBMODULES:
        module = importlib.import_module(f'{__name__}.{name}')
        globals()[name] = module          # cache: __getattr__ not hit again
        return module
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f'{__name__}.{submodule}'), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(__all__) | set(globals()))
