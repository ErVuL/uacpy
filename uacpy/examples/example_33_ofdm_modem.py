"""OFDM underwater modem — text → .wav → text.

The multicarrier counterpart of example 32: a real payload over OFDM, recovered
through a dispersive, Doppler-shifted channel. FEC + QPSK subcarriers +
Schmidl-Cox preamble + pilot + cyclic prefix, oversampled and upconverted; then
estimate and resample out the common Doppler, Schmidl-Cox the timing and CFO,
FFT, pilot channel estimate, one-tap per-subcarrier equalise with per-symbol
phase tracking, Viterbi decode, check the CRC.

Uses: comms.OFDMTransmitter/OFDMReceiver.transmit_passband ·
OFDMReceiver.receive_passband(return_diagnostics=) ·
comms.estimate_doppler_scale · comms.awgn ·
schmidl_cox_sync · remove_cfo · estimate_channel · uacpy.io.write_wav
· plot_scatter
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
from uacpy.comms import estimate_channel, remove_cfo, schmidl_cox_sync

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0xACED)
fs, fc, oversampling = 96000.0, 24000.0, 4
nsc, cp = 256, 32

message = (b"OFDM underwater modem: a cyclic prefix turns long multipath into "
           b"flat subcarriers. Resample the Doppler, Schmidl-Cox the rest!")
frame_bits = comms.pack_frame(message)

code = comms.ConvCode(interleave_depth=16)
tx = comms.OFDMTransmitter("qpsk", nsc, cp, code=code)
waveform = tx.transmit_passband(frame_bits, fs, fc, oversample=oversampling)
uacpy.io.write_wav(OUT / 'example_33_ofdm.wav', waveform, fs)
band = fs / oversampling
print(f"  payload  : {len(message)} bytes → {frame_bits.size} framed bits")
print(f"  waveform : {waveform.size} real samples, "
      f"{waveform.size / fs * 1e3:.0f} ms @ {fs / 1e3:.0f} kHz, band "
      f"{(fc - band / 2) / 1e3:.0f}-{(fc + band / 2) / 1e3:.0f} kHz")

# Channel: long multipath (the case the cyclic prefix exists for), 150 ppm
# clock-skew Doppler, delay, then noise.
impulse_response = np.zeros(60)
impulse_response[[0, 25, 51]] = [1.0, 0.5, 0.3]
received = np.convolve(waveform, impulse_response)
received = resample_poly(received, 100015, 100000)
received = np.concatenate([np.zeros(37), received])
snr_dB = 24.0
received = comms.awgn(received, snr_dB, rng=rng)

receiver = comms.OFDMReceiver("qpsk", nsc, cp, code=code)
scale, _, _ = comms.estimate_doppler_scale(
    received,
    comms.upconvert(resample_poly(receiver.preamble, oversampling, 1), fs, fc),
    np.linspace(-3e-4, 3e-4, 31))
# return_diagnostics keeps the Schmidl-Cox timing metric the receiver searched
# and the phase-corrected payload symbols it decided on; n_symbols trims those
# to the data symbols frame_bits filled, dropping the block's zero padding.
diag = receiver.receive_passband(
    received, fs, fc, oversample=oversampling, doppler_scale=scale,
    return_diagnostics=True,
    n_symbols=receiver.payload_symbol_count(frame_bits.size))
payload, crc_ok = comms.unpack_frame(diag.bits)
print(f"  channel  : 3-path ({impulse_response.nonzero()[0][-1]}-sample "
      f"spread) + 150 ppm Doppler, {snr_dB:.0f} dB SNR")
print(f"  Doppler  : {scale * 1e3:+.2f}e-3 estimated, resampled out")
print(f"  CRC      : {'OK' if crc_ok else 'FAIL'}, "
      f"payload match {payload == message}")
print(f"  recovered: {payload[:60]!r}{'...' if len(payload) > 60 else ''}")

# The per-subcarrier channel for the figure, from the same public steps the
# receiver takes: sync on the baseband, remove the CFO, estimate H from the
# pilot block (block 0 is the Schmidl-Cox preamble, block 1 the pilot).
baseband = receiver.from_passband(received, fs, fc, oversample=oversampling,
                                  doppler_scale=scale)
start, cfo = schmidl_cox_sync(baseband, nsc)
block = nsc + cp
x = remove_cfo(baseband[start:], cfo)
H = estimate_channel(x[block:2 * block], receiver.pilot_values, nsc, cp)
constellation = receiver.modulator.constellation

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
axes[0, 0].specgram(received, NFFT=256, Fs=fs, noverlap=200, cmap='jet')
axes[0, 0].axhline(fc, color='w', ls='--', lw=1)
axes[0, 0].set_title('received passband spectrogram')
axes[0, 0].set_xlabel('Time [s]')
axes[0, 0].set_ylabel('Frequency [Hz]')
axes[0, 0].set_ylim(0, fs / 2)

axes[0, 1].plot(diag.sync_metric)
axes[0, 1].set_title('Schmidl-Cox timing metric')
axes[0, 1].set_xlabel('Sample index')
axes[0, 1].set_ylabel('Metric')
axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(np.arange(nsc) - nsc // 2,
                np.fft.fftshift(20 * np.log10(np.abs(H) + 1e-9)))
axes[1, 0].set_title('per-subcarrier channel |H|')
axes[1, 0].set_xlabel('Subcarrier')
axes[1, 0].set_ylabel('|H| [dB]')
axes[1, 0].grid(alpha=0.3)

uacpy.plot.plot_scatter(diag.symbols, ax=axes[1, 1],
                        title=f"recovered QPSK "
                              f"(CRC {'OK' if crc_ok else 'FAIL'})")
axes[1, 1].scatter(constellation.real, constellation.imag, marker='x', s=80,
                   color='k', zorder=5)

fig.savefig(OUT / 'example_33_ofdm_modem.png', dpi=120)
plt.close(fig)
