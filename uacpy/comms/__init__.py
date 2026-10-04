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

from .constellations import (
    SCHEMES,
    Modulator,
    constellation,
    dpsk_modulate,
    dpsk_demodulate,
    fsk_modulate,
    fsk_demodulate,
    slicer,
)
from .ofdm import (
    ofdm_modulate,
    ofdm_demodulate,
    ofdm_symbol,
    subcarrier_response,
    schmidl_cox_preamble,
    schmidl_cox_metric,
    schmidl_cox_sync,
    remove_cfo,
    estimate_channel,
    equalize_subcarriers,
)
from .coding import (
    bytes_to_bits,
    bits_to_bytes,
    pack_frame,
    unpack_frame,
    ConvCode,
    conv_encode,
    viterbi_decode,
    viterbi_hard,
    interleave,
    deinterleave,
    spread,
    despread,
    spreading_gain_dB,
)
from .equalize import (
    DFE,
    lms_equalizer,
    rls_equalizer,
    mmse_equalizer,
    ls_estimate,
    omp_estimate,
    complex_gain,
)
from .sync import (
    symbol_sync,
    doppler_from_speed,
    compensate_doppler,
    estimate_doppler_scale,
    matched_filter_metric,
    detect_preamble,
    detect_frames,
)
from .channel import (
    awgn,
    ebn0_to_snr_dB,
    snr_to_ebn0_dB,
    multipath_channel,
    pulse_shaped_taps,
    apply_channel,
    fading_taps,
    apply_fading_channel,
    ChannelTaps,
)
from .phy import (
    rrc_filter,
    pulse_shape,
    rrc_matched_filter,
    upconvert,
    downconvert,
    rrc_pulse,
    rc_pulse,
)
from .transceiver import (
    Transmitter,
    CommsReceiver,
    OFDMTransmitter,
    OFDMReceiver,
    ReceiverDiagnostics,
)
from .link import (
    simulate_link,
    BerCurve,
    ber_sweep,
    LinkResult,
)
from .metrics import (
    bit_error_rate,
    symbol_error_rate,
    evm,
    ber_theory,
)
# spreading-code workflow reaches it here.
from uacpy.acoustic_signal.generate import m_sequence
from .janus import (
    JanusPacket,
    JanusReception,
    janus_encode,
    janus_decode,
    janus_modulate,
    janus_demodulate,
    janus_detect,
    janus_transmit,
    janus_receive,
)

# Imported so each submodule is reachable as an attribute; not in __all__.
from . import (  # noqa: F401
    channel, coding, constellations, equalize, janus, link, metrics, ofdm,
    phy, sync, transceiver,
)

__all__ = [
    # modulation
    "SCHEMES", "Modulator", "constellation", "dpsk_modulate",
    "dpsk_demodulate", "fsk_modulate", "fsk_demodulate",
    # metrics
    "bit_error_rate", "symbol_error_rate", "evm", "ber_theory",
    # channel models
    "awgn", "ebn0_to_snr_dB", "snr_to_ebn0_dB",
    "multipath_channel", "pulse_shaped_taps",
    "apply_channel", "fading_taps",
    "apply_fading_channel",
    # equalization
    "DFE", "lms_equalizer", "rls_equalizer", "mmse_equalizer", "slicer",
    "complex_gain",
    # doppler
    "doppler_from_speed", "compensate_doppler", "estimate_doppler_scale",
    # sync
    "matched_filter_metric", "detect_preamble", "detect_frames",
    # link
    "simulate_link", "ber_sweep", "BerCurve", "LinkResult", "ChannelTaps",
    # framing (real payloads)
    "bytes_to_bits", "bits_to_bytes", "pack_frame", "unpack_frame",
    # passband PHY
    "rrc_filter", "rrc_pulse", "rc_pulse", "pulse_shape",
    "rrc_matched_filter", "upconvert", "downconvert", "symbol_sync",
    # transceiver
    "Transmitter", "CommsReceiver", "OFDMTransmitter", "OFDMReceiver",
    "ReceiverDiagnostics",
    # JANUS (STANAG 4748)
    "JanusPacket", "JanusReception", "janus_encode", "janus_decode",
    "janus_modulate",
    "janus_demodulate", "janus_detect", "janus_transmit", "janus_receive",
    # channel estimation
    "ls_estimate", "omp_estimate",
    # ofdm
    "ofdm_modulate", "ofdm_demodulate", "subcarrier_response",
    "schmidl_cox_preamble",
    "schmidl_cox_metric", "schmidl_cox_sync", "remove_cfo", "estimate_channel", "ofdm_symbol",
    "equalize_subcarriers",
    # coding
    "ConvCode", "conv_encode", "viterbi_decode", "viterbi_hard", "interleave",
    "deinterleave",
    # DSSS
    "m_sequence", "spread", "despread", "spreading_gain_dB",
]
