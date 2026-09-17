"""Underwater acoustic communications: modulation, channels, receivers, coding.

A digital-communications toolbox tuned to the underwater acoustic channel —
severe time-varying multipath, rapid carrier-phase variation, and motion-induced
Doppler scaling. Tier 1 covers the essential coherent link (modulation, fading
channels, adaptive DFE/LMS/RLS equalization with an optional carrier PLL, Doppler
estimation/compensation, frame synchronization, BER/EVM metrics, and an
end-to-end harness); Tier 2 adds the classical layers (pilot-based LS/sparse-OMP
channel estimation, OFDM, convolutional+Viterbi FEC, and DSSS).

References
----------
Istepanian & Stojanovic (eds.). *Underwater Acoustic Digital Signal Processing
    and Communication Systems*.
Proakis & Salehi. *Digital Communications*.
"""

from .modulate import (
    # symbol mapping
    Modulator,
    constellation,
    dpsk_modulate,
    dpsk_demodulate,
    fsk_modulate,
    fsk_demodulate,
    # payload framing
    bytes_to_bits,
    bits_to_bytes,
    pack_frame,
    unpack_frame,
    # OFDM
    ofdm_modulate,
    ofdm_demodulate,
    ofdm_symbol,
    schmidl_cox_preamble,
    schmidl_cox_sync,
    apply_cfo,
    estimate_channel,
    equalize_subcarriers,
    # forward error correction and interleaving
    ConvCode,
    conv_encode,
    viterbi_decode,
    viterbi_hard,
    interleave,
    deinterleave,
    # direct-sequence spread spectrum
    m_sequence,
    spread,
    despread,
    processing_gain_dB,
)
from .link import (
    # channel models
    awgn,
    multipath_channel,
    apply_channel,
    fading_taps,
    apply_fading_channel,
    # passband PHY
    rrc_filter,
    pulse_shape,
    matched_filter as rrc_matched_filter,
    upconvert,
    downconvert,
    symbol_sync,
    # transceivers
    Transmitter,
    CommsReceiver,
    OFDMTransmitter,
    OFDMReceiver,
    # end-to-end harness
    simulate_link,
    ber_sweep,
    LinkResult,
)
from .receive import (
    # equalization
    DFE,
    lms_equalizer,
    rls_equalizer,
    mmse_equalizer,
    slicer,
    # doppler
    doppler_from_speed,
    compensate_doppler,
    estimate_doppler_scale,
    # synchronisation
    matched_filter_metric,
    detect_preamble,
    detect_frames,
    # channel estimation
    ls_estimate,
    omp_estimate,
    # link quality
    bit_error_rate,
    symbol_error_rate,
    evm,
    ber_theory,
)
from .janus import (
    JanusPacket,
    janus_encode,
    janus_decode,
    janus_modulate,
    janus_demodulate,
    janus_detect,
    janus_transmit,
    janus_receive,
)

from . import janus, link, modulate, receive

__all__ = [
    # modulation
    "Modulator", "constellation", "dpsk_modulate", "dpsk_demodulate",
    "fsk_modulate", "fsk_demodulate",
    # metrics
    "bit_error_rate", "symbol_error_rate", "evm", "ber_theory",
    # channel models
    "awgn", "multipath_channel", "apply_channel", "fading_taps",
    "apply_fading_channel",
    # equalization
    "DFE", "lms_equalizer", "rls_equalizer", "mmse_equalizer", "slicer",
    # doppler
    "doppler_from_speed", "compensate_doppler", "estimate_doppler_scale",
    # sync
    "matched_filter_metric", "detect_preamble", "detect_frames",
    # link
    "simulate_link", "ber_sweep", "LinkResult",
    # framing (real payloads)
    "bytes_to_bits", "bits_to_bytes", "pack_frame", "unpack_frame",
    # passband PHY
    "rrc_filter", "pulse_shape", "rrc_matched_filter", "upconvert",
    "downconvert", "symbol_sync",
    # transceiver
    "Transmitter", "CommsReceiver", "OFDMTransmitter", "OFDMReceiver",
    # JANUS (STANAG 4748)
    "JanusPacket", "janus_encode", "janus_decode", "janus_modulate",
    "janus_demodulate", "janus_detect", "janus_transmit", "janus_receive",
    # channel estimation
    "ls_estimate", "omp_estimate",
    # ofdm
    "ofdm_modulate", "ofdm_demodulate", "schmidl_cox_preamble",
    "schmidl_cox_sync", "apply_cfo", "estimate_channel", "ofdm_symbol",
    "equalize_subcarriers",
    # coding
    "ConvCode", "conv_encode", "viterbi_decode", "viterbi_hard", "interleave",
    "deinterleave",
    # DSSS
    "m_sequence", "spread", "despread", "processing_gain_dB",
    # submodules
    "modulate", "link", "receive", "janus",
]
