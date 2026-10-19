"""OFDM underwater modem — text → passband waveform → text.

The multicarrier counterpart of example 32: a real payload over OFDM, recovered
through a dispersive, Doppler-shifted channel. FEC + QPSK subcarriers +
Schmidl-Cox preamble + pilot + cyclic prefix, oversampled and upconverted; then
estimate and resample out the common Doppler, Schmidl-Cox the timing and CFO,
FFT, pilot channel estimate, one-tap per-subcarrier equalise with per-symbol
phase tracking, Viterbi decode, check the CRC.

Uses: comms.OFDMTransmitter.transmit_passband ·
OFDMReceiver.receive_passband(return_diagnostics=) ·
comms.estimate_doppler_scale · comms.awgn · OFDMTransmitter.to_passband ·
ReceiverDiagnostics.channel_estimate ·
uacpy.io.write_wav · acoustic_signal.spectrogram · plot_sync_metric ·
plot_subcarriers · plot_received_symbols
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
fs, fc, oversampling = 96000.0, 24000.0, 4
nsc, cp = 256, 32

message = (b"OFDM underwater modem: a cyclic prefix turns long multipath into "
           b"flat subcarriers. Resample the Doppler, Schmidl-Cox the rest!")
frame_bits = comms.pack_frame(message)

code = comms.ConvCode(interleave_depth=16)
tx = comms.OFDMTransmitter("qpsk", nsc, cp, code=code)
waveform = tx.transmit_passband(frame_bits, sample_rate=fs, fc=fc, oversample=oversampling)
uacpy.io.write_wav(OUT / 'example_33_ofdm.wav', waveform, fs, normalize=True)
band = fs / oversampling
print(f"  payload  : {len(message)} bytes → {frame_bits.size} framed bits")
print(f"  waveform : {waveform.size} real samples, "
      f"{waveform.size / fs * 1e3:.0f} ms @ {fs / 1e3:.0f} kHz, band "
      f"{(fc - band / 2) / 1e3:.0f}-{(fc + band / 2) / 1e3:.0f} kHz")

# Channel: long multipath (the case the cyclic prefix exists for), 150 ppm
# clock-skew Doppler, delay, then noise.
impulse_response = np.zeros(60)
impulse_response[[0, 25, 51]] = [1.0, 0.5, 0.3]
received = comms.apply_channel(waveform, impulse_response)
received = resample_poly(received, 100015, 100000)
received = np.concatenate([np.zeros(37), received])
snr_dB = 24.0
received = comms.awgn(received, snr_dB, rng=rng)

receiver = comms.OFDMReceiver("qpsk", nsc, cp, code=code)
scale, _, _ = comms.estimate_doppler_scale(
    received,
    tx.to_passband(tx.preamble, sample_rate=fs, fc=fc, oversample=oversampling),
    np.linspace(-3e-4, 3e-4, 31))
# return_diagnostics keeps the Schmidl-Cox timing metric the receiver searched,
# the per-subcarrier channel it equalised by and the phase-corrected payload
# symbols it decided on; n_symbols trims those to the data symbols frame_bits
# filled, dropping the block's zero padding.
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

constellation = receiver.modulator.constellation

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
# The waveform's samples read as pascals: the levels are dB re 1 Pa²/Hz.
spectrogram(received, fs, nperseg=256, noverlap=200).plot(
    ax=axes[0, 0], ref=1.0, freq_max=fs / 2,
    title='received passband spectrogram')
axes[0, 0].axhline(fc, color='w', ls='--', lw=1)

# schmidl_cox_sync takes the metric's peak as the preamble when it reaches
# the 0.5 default threshold.
uacpy.plot.plot_sync_metric(diag.sync_metric, ax=axes[0, 1], threshold=0.5,
                            title='Schmidl-Cox timing metric')

uacpy.plot.plot_subcarriers(diag.channel_estimate, ax=axes[1, 0],
                            title='per-subcarrier channel |H|')

uacpy.plot.plot_received_symbols(diag.symbols, ax=axes[1, 1], ideal=constellation,
                                 title=f"recovered QPSK "
                                       f"(CRC {'OK' if crc_ok else 'FAIL'})")

fig.savefig(OUT / 'example_33_ofdm_modem.png', dpi=120)
plt.close(fig)
