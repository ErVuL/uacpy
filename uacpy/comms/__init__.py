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

from .modulation import (
    Modulator,
    constellation,
    dpsk_demodulate,
    dpsk_modulate,
    fsk_demodulate,
    fsk_modulate,
)
from .metrics import (
    ber_theory,
    bit_error_rate,
    evm,
    symbol_error_rate,
)
from .channel_models import (
    apply_channel,
    apply_fading_channel,
    awgn,
    fading_taps,
    multipath_channel,
)
from .equalization import (
    DFE,
    lms_equalizer,
    mmse_equalizer,
    rls_equalizer,
)
from .doppler import (
    compensate_doppler,
    doppler_from_speed,
    estimate_doppler_scale,
)
from .sync import (
    detect_frames,
    detect_preamble,
    matched_filter_metric,
)
from .link import LinkResult, ber_sweep, simulate_link
from .framing import bits_to_bytes, bytes_to_bits, pack_frame, unpack_frame
from .phy import (
    downconvert,
    matched_filter as rrc_matched_filter,
    pulse_shape,
    rrc_filter,
    symbol_sync,
    upconvert,
)
from .transceiver import OFDMReceiver, OFDMTransmitter, Receiver, Transmitter
from .janus import (
    JanusPacket,
    janus_decode,
    janus_demodulate,
    janus_detect,
    janus_encode,
    janus_modulate,
)
from .channel_est import ls_estimate, omp_estimate
from .ofdm import (
    apply_cfo,
    estimate_channel,
    ofdm_demodulate,
    ofdm_modulate,
    schmidl_cox_preamble,
    schmidl_cox_sync,
)
from .coding import (
    ConvCode,
    conv_encode,
    deinterleave,
    interleave,
    viterbi_decode,
)
from .spread import despread, m_sequence, processing_gain_db, spread

from . import (
    channel_est,
    channel_models,
    coding,
    doppler,
    equalization,
    framing,
    janus,
    link,
    metrics,
    modulation,
    ofdm,
    phy,
    sync,
    transceiver,
)

__all__ = [
    # modulation
    "Modulator", "constellation", "dpsk_modulate", "dpsk_demodulate",
    "fsk_modulate", "fsk_demodulate",
    # metrics
    "bit_error_rate", "symbol_error_rate", "evm", "ber_theory",
    # channel models
    "awgn", "multipath_channel", "apply_channel", "fading_taps",
    "apply_fading_channel", # equalization
    "DFE", "lms_equalizer", "rls_equalizer", "mmse_equalizer", # doppler
    "doppler_from_speed", "compensate_doppler", "estimate_doppler_scale",
    # sync
    "matched_filter_metric", "detect_preamble", "detect_frames", # link
    "simulate_link", "ber_sweep", "LinkResult",
    # framing (real payloads)
    "bytes_to_bits", "bits_to_bytes", "pack_frame", "unpack_frame",
    # passband PHY
    "rrc_filter", "pulse_shape", "rrc_matched_filter", "upconvert",
    "downconvert", "symbol_sync",
    # transceiver
    "Transmitter", "Receiver", "OFDMTransmitter", "OFDMReceiver",
    # JANUS (STANAG 4748)
    "JanusPacket", "janus_encode", "janus_decode", "janus_modulate",
    "janus_demodulate", "janus_detect",
    # channel estimation
    "ls_estimate", "omp_estimate",
    # ofdm
    "ofdm_modulate", "ofdm_demodulate", "schmidl_cox_preamble", "schmidl_cox_sync", "apply_cfo", "estimate_channel",
    # coding
    "ConvCode", "conv_encode", "viterbi_decode", "interleave", "deinterleave",
    # spread
    "m_sequence", "spread", "despread", "processing_gain_db",
    # submodules
    "modulation", "metrics", "channel_models", "equalization", "doppler",
    "sync", "link", "channel_est", "ofdm", "coding",
    "framing", "phy", "transceiver", "janus",
]
