"""Signal processing and generation tools for acoustic signals.

Named ``acoustic_signal`` so it does not collide with Python's stdlib
``signal`` module. Sub-modules, each with one responsibility:

* ``generate``   — *give me a signal*: parametric waveforms, coded probe
                   sequences, and noise built to a target spectrum
* ``spectral``      — *measure this signal*: one estimator per statistic
                      (``welch``, ``constant_q``, ``sound_exposure`` and a
                      ``probabilistic_`` twin of each)
* ``cqt``           — the constant-Q transform and spectrogram
* ``bands``         — the standard band ladders a level is reported on
* ``timefreq``      — the time-resolved views (``spectrogram``, ``cwt``,
                      ``wigner_ville``, the analytic signal, cepstra)
* ``spectrum_at``   — one tone's phasor, and a spectrum at the frequencies
                      asked for
* ``beamforming``   — *what does this array see*: steering, covariance and
                      beamforming
* ``gathers``       — the gather transforms (f-k, tau-p, Radon), which take a
                      receiver spacing ``dx``
* ``detect``        — *is my transmission in there*: matched filter, pulse
                      compression, ambiguity
* ``frf``           — *what did the channel do to it*: frequency-response
                      estimation
* ``channel``       — channel simulation and the transfer-function operations
* ``delay_profile`` — power-delay-profile statistics and the channel regime
* ``dispersion``    — modal dispersion and warping

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
Frequency-response estimation is functions too (``frf_welch``, ``etfe``,
``periodic_etfe``, ``lsfir`` → ``FRFResult``); ``FRF`` is a class only to
hold one of them configured.

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
``generate.sparc_pulse`` (numpy-only), and an eager ``__init__`` here
dragged scipy.signal/scipy.stats into every ``import uacpy``.
"""

import importlib as _importlib

# Sub-modules, importable on demand as attributes of this package, in the
# order the docstring above lists them. ``__all__`` lists functions, classes
# and constants only, so these stay out of it (and out of ``dir()``).
_SUBMODULE_NAMES = ('generate', 'spectral', 'cqt', 'bands',
                    'timefreq', 'spectrum_at', 'beamforming', 'gathers',
                    'detect', 'frf', 'channel', 'delay_profile',
                    'dispersion')
_SUBMODULES = frozenset(_SUBMODULE_NAMES)

# Public name -> defining sub-module. Kept in sync with __all__ by the
# consistency check at the bottom of this file (runs at import, costs nothing).
_EXPORTS = {
    # generate: parametric waveforms
    'sparc_pulse': 'generate', 'gaussian_pulse': 'generate',
    'hfm_chirp': 'generate', 'lfm_chirp': 'generate', 'nwave': 'generate',
    'ricker_wavelet': 'generate', 'tone_burst': 'generate',
    # generate: coded sequences
    'bpsk_modulate': 'generate', 'make_mseq_probe': 'generate',
    'm_sequence': 'generate',
    # generate: noise
    'make_bandlimited_noise': 'generate',
    'synthesize_noise_from_psd': 'generate',
    # estimate: spectral / level estimators
    'ProbabilisticSpectralEstimate': 'spectral',
    'SpectralEstimate': 'spectral',
    'BAND_TYPES': 'spectral',
    'welch': 'spectral', 'constant_q': 'spectral',
    'spectral_centroid': 'spectral',
    'probabilistic_welch': 'spectral',
    'level_histogram': 'spectral', 'level_percentiles': 'spectral',
    'probabilistic_constant_q': 'spectral',
    'sound_exposure': 'spectral',
    'probabilistic_sound_exposure': 'spectral',
    # system: system identification
    'FRF': 'frf', 'FRFResult': 'frf', 'frf_welch': 'frf', 'etfe': 'frf',
    'periodic_etfe': 'frf', 'lsfir': 'frf',
    # arrays: beamforming
    'bartlett': 'beamforming', 'beamform': 'beamforming',
    'BeamformResult': 'beamforming', 'music_spectrum': 'beamforming',
    'mvdr': 'beamforming', 'sample_covariance': 'beamforming',
    'snapshots': 'beamforming', 'Snapshots': 'beamforming',
    'steering_vectors': 'beamforming', 'shading_taper': 'beamforming',
    'beamform_field': 'beamforming', 'BeamformedField': 'beamforming',
    'plane_wave_array_gain': 'beamforming', 'matched_replica_gain': 'beamforming',
    'realised_array_gain': 'beamforming',
    'independent_beams': 'beamforming',
    # detect: matched filter and ambiguity
    'AmbiguityResult': 'detect', 'ambiguity_function': 'detect',
    'matched_filter': 'detect', 'processing_gain_dB': 'detect',
    'pulse_compression': 'detect',
    # arrays: gather transforms
    'fk_transform': 'gathers', 'inverse_fk': 'gathers',
    'inverse_radon': 'gathers', 'inverse_taup': 'gathers',
    'radon_transform': 'gathers', 'taup_transform': 'gathers',
    'FKResult': 'gathers', 'TauPResult': 'gathers',
    'RadonResult': 'gathers',
    # system (and estimate, for the two phasor functions): channel
    'fractional_delay_taps': 'channel',
    'impulse_response': 'channel',
    'impulse_response_from_transfer_function': 'channel',
    'transfer_function_from_impulse_response': 'channel',
    'arrival_transfer_function': 'channel',
    'arrival_grid_transfer_function': 'channel',
    'received_amplitudes': 'channel',
    'remove_delay': 'channel',
    'delayandsum': 'channel',
    'broadband_propagation_loss': 'channel',
    'gate_transfer_function': 'channel',
    'tone_phasor': 'spectrum_at',
    'waveform_spectrum_at': 'spectrum_at',
    'simulate_arrival_reception': 'channel',
    'simulate_arrival_grid': 'channel',
    'synthesize_time_series': 'channel',
    'uniform_frequency_step': 'channel',
    'ChannelRegime': 'delay_profile',
    'COHERENCE_BANDWIDTH_FACTORS': 'delay_profile',
    'rms_delay_spread': 'delay_profile',
    'energy_support': 'delay_profile',
    'synthesis_band': 'delay_profile',
    'coherence_bandwidth': 'delay_profile',
    'coherence_factor': 'delay_profile',
    'channel_regime': 'delay_profile',
    'channel_response': 'channel',
    'simulate_reception': 'channel',
    # system: modal dispersion and warping
    'modal_group_velocity': 'dispersion', 'unwarp_signal': 'dispersion',
    'warp_signal': 'dispersion',
    # estimate: time-frequency views
    'spectrogram': 'timefreq', 'analytic_signal': 'timefreq',
    'cepstrum': 'timefreq', 'Cepstrum': 'timefreq',
    'ComplexCepstrum': 'timefreq',
    'complex_cepstrum': 'timefreq', 'cwt': 'timefreq', 'envelope': 'timefreq',
    'instantaneous_frequency': 'timefreq',
    'inverse_complex_cepstrum': 'timefreq', 'inverse_cwt': 'timefreq',
    'wigner_ville': 'timefreq', 'SpectrogramResult': 'timefreq',
    'CWTResult': 'timefreq', 'WignerVilleResult': 'timefreq',
    # estimate: the constant-Q transform and spectrogram (the constant-Q
    # estimators sit with the other estimators above)
    'constant_q_transform': 'cqt',
    'constant_q_spectrogram': 'cqt',
    'CQTResult': 'cqt', 'CQSpectrogramResult': 'cqt',
    # estimate: standard band ladders
    'decidecade_bands': 'bands', 'decidecade_band_levels': 'bands',
    'octave_bands': 'bands',
    'standard_bands': 'bands', 'band_levels': 'bands',
    'BandLevels': 'bands',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _SUBMODULES:
        module = _importlib.import_module(f'{__name__}.{name}')
        globals()[name] = module          # cache: __getattr__ not hit again
        return module
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}.")
    value = getattr(_importlib.import_module(f'{__name__}.{submodule}'), name)
    globals()[name] = value
    return value


def __dir__():
    # The public names, loaded or not, plus the module's dunders; the
    # private loader tables and the ``_importlib`` alias stay out.
    return sorted(set(__all__) | {n for n in globals() if n.startswith('__')})
