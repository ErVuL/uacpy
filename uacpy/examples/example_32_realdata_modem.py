"""Real-data underwater modem — text → .wav → text.

A real byte payload over the full passband physical layer, recovered through a
realistic underwater channel: frame the message (length header + CRC-32), send
it as FEC + QPSK + preamble + RRC pulse shaping upconverted to passband, push
it through sparse multipath, clock-skew Doppler, delay and noise, then
downconvert, recover timing, sync on the preamble, equalise with a DFE and
carrier PLL, Viterbi-decode, and check the CRC.

Uses: comms.pack_frame/unpack_frame · Transmitter.transmit_passband ·
CommsReceiver.from_passband/receive · comms.DFE · uacpy.io.write_wav ·
plot_scatter · plot_sync_metric · plot_convergence
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
import uacpy
from uacpy import comms

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
waveform = tx.transmit_passband(frame_bits, fs, fc, sps=sps)
uacpy.io.write_wav(OUT / 'example_32_modem.wav', waveform, fs)
print(f"  payload  : {len(message)} bytes → {frame_bits.size} framed bits")
print(f"  waveform : {waveform.size} real samples, "
      f"{waveform.size / fs * 1e3:.0f} ms @ {fs / 1e3:.0f} kHz, "
      f"carrier {fc / 1e3:.0f} kHz")

# Channel: 3-path multipath, 200 ppm clock skew, propagation delay, then noise.
impulse_response = np.zeros(40)
impulse_response[[0, 17, 33]] = [1.0, 0.4, 0.2]
received = np.convolve(waveform, impulse_response)
received = resample_poly(received, 100020, 100000)
received = np.concatenate([np.zeros(11), received])
snr_dB = 22.0
received = received + np.sqrt(
    np.mean(received ** 2) / 10 ** (snr_dB / 10)) * rng.standard_normal(
        received.size)

# Receive: downconvert + matched filter + Gardner timing recovery, then frame
# sync on the preamble, adaptive DFE with carrier PLL, Viterbi decode.
dfe = comms.DFE(n_ff=16, n_fb=6, forget=0.997, pll_bandwidth=0.04)
receiver = comms.CommsReceiver("qpsk", code=code, equalizer=dfe, preamble=256)
symbols = receiver.from_passband(received, fs, fc, sps=sps)
start, sync_metric = comms.detect_preamble(symbols, receiver.preamble,
                                           threshold=0.4)
payload, crc_ok = comms.unpack_frame(receiver.receive(symbols))
print(f"  channel  : 3-path multipath, 200 ppm Doppler, {snr_dB:.0f} dB SNR")
print(f"  preamble : found at symbol {start}")
print(f"  CRC      : {'OK' if crc_ok else 'FAIL'}, "
      f"payload match {payload == message}")
print(f"  recovered: {payload[:60]!r}{'...' if len(payload) > 60 else ''}")

# Re-run the equaliser to keep the symbols and the error curve for the figure;
# receive() returns decoded bits, not these.
aligned = symbols[start:]
delay = dfe.n_ff // 2
reference = np.concatenate([np.zeros(delay, dtype=complex), receiver.preamble])
equalized, mse = comms.DFE(n_ff=16, n_fb=6, forget=0.997,
                           pll_bandwidth=0.04).equalize(
    aligned, receiver.modulator.constellation, train=reference)
payload_symbols = equalized[delay + receiver.preamble.size:]

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
axes[0, 0].specgram(received, NFFT=256, Fs=fs, noverlap=200, cmap='jet')
axes[0, 0].axhline(fc, color='w', ls='--', lw=1)
axes[0, 0].set_title('received passband spectrogram', loc='left')
axes[0, 0].set_xlabel('Time [s]')
axes[0, 0].set_ylabel('Frequency [Hz]')
axes[0, 0].set_ylim(0, fs / 2)

uacpy.plot.plot_scatter(payload_symbols[200:], ax=axes[0, 1],
                        title=f"recovered QPSK "
                              f"(CRC {'OK' if crc_ok else 'FAIL'})")
axes[0, 1].scatter(receiver.modulator.constellation.real,
                   receiver.modulator.constellation.imag,
                   marker='x', s=80, color='k', zorder=5)
uacpy.plot.plot_sync_metric(sync_metric, threshold=0.4, ax=axes[1, 0])
uacpy.plot.plot_convergence(mse, ax=axes[1, 1], title='(DFE + carrier PLL)')

fig.savefig(OUT / 'example_32_realdata_modem.png', dpi=120)
plt.close(fig)
