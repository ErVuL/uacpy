"""Real-data underwater modem — text → .wav → text.

A real byte payload over the full passband physical layer, recovered through a
realistic underwater channel: frame the message (length header + CRC-32), send
it as FEC + QPSK + preamble + RRC pulse shaping upconverted to passband, push
it through sparse multipath, clock-skew Doppler, delay and noise, then
downconvert, recover timing, sync on the preamble, equalise with a DFE and
carrier PLL, Viterbi-decode, and check the CRC.

Uses: comms.pack_frame/unpack_frame · Transmitter.transmit_passband ·
CommsReceiver.from_passband/receive(return_diagnostics=) · comms.DFE ·
comms.awgn · uacpy.io.write_wav/read_wav · acoustic_signal.spectrogram ·
plot_received_symbols · plot_sync_metric · plot_convergence
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
import uacpy
from uacpy import comms
from uacpy.acoustic_signal import spectrogram

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0xACED)
fs, fc, sps = 96000.0, 24000.0, 8

message = (b"uacpy underwater acoustic modem -- real data over a simulated "
           b"ocean channel. The quick brown fox jumps over 13 lazy dogs!")
frame_bits = comms.pack_frame(message)

# Transmit: FEC + QPSK + preamble + RRC + upconvert. The passband signal is
# what a transducer would emit, so it is also what goes in the .wav.
code = comms.ConvCode(interleave_depth=16)
tx = comms.Transmitter("qpsk", code=code, preamble=256)
waveform = tx.transmit_passband(frame_bits, sample_rate=fs, fc=fc, sps=sps)
wav_path = OUT / 'example_32_modem.wav'
uacpy.io.write_wav(wav_path, waveform, fs, normalize=True)
print(f"  payload  : {len(message)} bytes → {frame_bits.size} framed bits")
print(f"  waveform : {waveform.size} real samples, "
      f"{waveform.size / fs * 1e3:.0f} ms @ {fs / 1e3:.0f} kHz, "
      f"carrier {fc / 1e3:.0f} kHz")

# Channel: the .wav read back (16-bit PCM, scaled to ±1), then 3-path
# multipath, 200 ppm clock skew, propagation delay and noise.
transmitted, _ = uacpy.io.read_wav(wav_path)
impulse_response = np.zeros(40)
impulse_response[[0, 17, 33]] = [1.0, 0.4, 0.2]
received = comms.apply_channel(transmitted, impulse_response)
received = resample_poly(received, 100020, 100000)
received = np.concatenate([np.zeros(11), received])
snr_dB = 22.0
received = comms.awgn(received, snr_dB, rng=rng)

# Receive: downconvert + matched filter + Gardner timing recovery, then frame
# sync on the preamble, adaptive DFE with carrier PLL, Viterbi decode.
dfe = comms.DFE(n_ff=16, n_fb=6, forget=0.997, pll_gain=0.04)
receiver = comms.CommsReceiver("qpsk", code=code, equalizer=dfe, preamble=256)
symbols = receiver.from_passband(received, fs, fc, sps=sps)
# return_diagnostics keeps what the receiver computed on the way to the bits:
# the preamble sync metric and start, the equalised payload symbols and the
# DFE's squared-error curve.
diag = receiver.receive(symbols, return_diagnostics=True)
payload, crc_ok = comms.unpack_frame(diag.bits)
print(f"  channel  : 3-path multipath, 200 ppm Doppler, {snr_dB:.0f} dB SNR")
print(f"  preamble : found at symbol {diag.start}")
print(f"  CRC      : {'OK' if crc_ok else 'FAIL'}, "
      f"payload match {payload == message}")
print(f"  recovered: {payload[:60]!r}{'...' if len(payload) > 60 else ''}")

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
# The record is the .wav's samples, scaled to ±1, read as pascals: the
# levels are dB re 1 Pa²/Hz.
spectrogram(received, fs, nperseg=256, noverlap=200).plot(
    ax=axes[0, 0], ref=1.0, freq_max=fs / 2,
    title='received passband spectrogram')
axes[0, 0].axhline(fc, color='w', ls='--', lw=1)

# The first 200 payload symbols are left out: the DFE is still settling there.
uacpy.plot.plot_received_symbols(diag.symbols[200:], ax=axes[0, 1],
                                 ideal=receiver.modulator.constellation,
                                 title=f"recovered QPSK "
                                       f"(CRC {'OK' if crc_ok else 'FAIL'})")
uacpy.plot.plot_sync_metric(diag.sync_metric, threshold=0.5, ax=axes[1, 0])
uacpy.plot.plot_convergence(diag.mse, ax=axes[1, 1],
                            title='(DFE + carrier PLL)')

fig.savefig(OUT / 'example_32_realdata_modem.png', dpi=120)
plt.close(fig)
