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
    'active', 'analysis', 'arrays', 'bands', 'channel', 'constant_q',
    'modal', 'noise_synthesis', 'sequences', 'system_id', 'timefreq',
    'transforms', 'waveforms',
})

# Public name -> defining sub-module. Kept in sync with __all__ by the
# consistency check at the bottom of this file (runs at import, costs nothing).
_EXPORTS = {
    # waveforms
    'sparc_pulse': 'waveforms', 'gaussian_pulse': 'waveforms',
    'hfm_chirp': 'waveforms', 'lfm_chirp': 'waveforms', 'nwave': 'waveforms',
    'ricker_wavelet': 'waveforms', 'tone_burst': 'waveforms',
    # sequences
    'bpsk_modulate': 'sequences', 'make_mseq_probe': 'sequences',
    'mseq': 'sequences',
    # noise_synthesis
    'add_noise': 'noise_synthesis', 'fourier_synthesis': 'noise_synthesis',
    'make_bandlimited_noise': 'noise_synthesis',
    'make_noise_waveform': 'noise_synthesis',
    'synthesize_noise_from_psd': 'noise_synthesis',
    # analysis
    'PPSDResult': 'analysis', 'PSDResult': 'analysis', 'SELResult': 'analysis',
    'ppsd': 'analysis', 'psd': 'analysis', 'sel': 'analysis',
    # system_id
    'FRF': 'system_id',
    # arrays
    'bartlett_spectrum': 'arrays', 'beamform': 'arrays',
    'BeamformResult': 'arrays', 'music_spectrum': 'arrays',
    'mvdr_spectrum': 'arrays', 'sample_covariance': 'arrays',
    'steering_vectors': 'arrays', 'shading_taper': 'arrays',
    # active
    'AmbiguityResult': 'active', 'ambiguity_function': 'active',
    'matched_filter': 'active', 'processing_gain': 'active',
    'pulse_compression': 'active',
    # transforms
    'fk_transform': 'transforms', 'inverse_fk': 'transforms',
    'inverse_radon': 'transforms', 'inverse_taup': 'transforms',
    'radon_transform': 'transforms', 'taup_transform': 'transforms',
    'FKResult': 'transforms', 'TauPResult': 'transforms',
    'RadonResult': 'transforms',
    # channel
    'impulse_response': 'channel',
    'impulse_response_from_transfer_function': 'channel',
    'simulate_reception': 'channel',
    # modal
    'modal_group_velocity': 'modal', 'unwarp_signal': 'modal',
    'warp_signal': 'modal',
    # timefreq
    'spectrogram': 'timefreq', 'analytic_signal': 'timefreq',
    'cepstrum': 'timefreq', 'ComplexCepstrum': 'timefreq',
    'complex_cepstrum': 'timefreq', 'cwt': 'timefreq', 'envelope': 'timefreq',
    'instantaneous_frequency': 'timefreq',
    'inverse_complex_cepstrum': 'timefreq', 'inverse_cwt': 'timefreq',
    'wigner_ville': 'timefreq', 'SpectrogramResult': 'timefreq',
    'CWTResult': 'timefreq', 'WignerVilleResult': 'timefreq',
    # constant_q
    'constant_q_transform': 'constant_q', 'constant_q_psd': 'constant_q',
    'constant_q_spectrogram': 'constant_q',
    'probabilistic_constant_q': 'constant_q', 'CQTResult': 'constant_q',
    'CQPSDResult': 'constant_q', 'CQSpectrogramResult': 'constant_q',
    'CQPPSDResult': 'constant_q',
    # bands
    'decidecade_bands': 'bands', 'decidecade_band_levels': 'bands',
}

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
    "fk_transform", "inverse_fk", "FKResult",
    "taup_transform", "inverse_taup", "TauPResult",
    "radon_transform", "inverse_radon", "RadonResult",
    # channel
    "impulse_response", "simulate_reception",
    "impulse_response_from_transfer_function",
    # modal
    "modal_group_velocity", "warp_signal", "unwarp_signal",
    # time-frequency
    "spectrogram", "analytic_signal", "envelope", "instantaneous_frequency",
    "wigner_ville", "cwt", "inverse_cwt", "cepstrum", "ComplexCepstrum",
    "complex_cepstrum",
    "inverse_complex_cepstrum",
    "SpectrogramResult", "CWTResult", "WignerVilleResult",
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
