"""OFDM underwater modem — text → .wav → text.

The multicarrier counterpart of example 32: a real payload over OFDM, recovered
through a dispersive, Doppler-shifted channel. FEC + QPSK subcarriers +
Schmidl-Cox preamble + pilot + cyclic prefix, oversampled and upconverted; then
estimate and resample out the common Doppler, Schmidl-Cox the timing and CFO,
FFT, pilot channel estimate, one-tap per-subcarrier equalise with per-symbol
phase tracking, Viterbi decode, check the CRC.

Uses: comms.OFDMTransmitter/OFDMReceiver.transmit_passband/receive_passband ·
comms.estimate_doppler_scale · ofdm.schmidl_cox_sync/estimate_channel/
ofdm_demodulate · uacpy.io.write_wav · plot_scatter
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
from uacpy.comms.ofdm import (apply_cfo, estimate_channel, ofdm_demodulate,
                              schmidl_cox_sync)

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
received = received + np.sqrt(
    np.mean(received ** 2) / 10 ** (snr_dB / 10)) * rng.standard_normal(
        received.size)

receiver = comms.OFDMReceiver("qpsk", nsc, cp, code=code)
scale, _, _ = comms.estimate_doppler_scale(
    received,
    comms.upconvert(resample_poly(receiver.preamble, oversampling, 1), fs, fc),
    np.linspace(-3e-4, 3e-4, 31))
out_bits = receiver.receive_passband(received, fs, fc, oversample=oversampling,
                                     doppler_scale=scale)
payload, crc_ok = comms.unpack_frame(out_bits)
print(f"  channel  : 3-path ({impulse_response.nonzero()[0][-1]}-sample "
      f"spread) + 150 ppm Doppler, {snr_dB:.0f} dB SNR")
print(f"  Doppler  : {scale * 1e3:+.2f}e-3 estimated, resampled out")
print(f"  CRC      : {'OK' if crc_ok else 'FAIL'}, "
      f"payload match {payload == message}")
print(f"  recovered: {payload[:60]!r}{'...' if len(payload) > 60 else ''}")

# Everything the public API returns is taken from it; only the two quantities
# it does not hand back — the timing metric and the per-block phase-corrected
# symbols — are rebuilt below for the figure.
baseband = receiver.from_passband(received, fs, fc, oversample=oversampling,
                                  doppler_scale=scale)
start, cfo = schmidl_cox_sync(baseband, nsc)
# schmidl_cox_sync returns only (start, cfo), so the metric it searched is
# rebuilt to plot it: P(d) and R(d) are the length-L sliding sums of Schmidl &
# Cox eqs. (5) and (7), and M(d) = |P(d)|^2 / R(d)^2 is eq. (8).
L = nsc // 2
products = np.conj(baseband[:-L]) * baseband[L:]
p = np.array([products[d:d + L].sum() for d in range(baseband.size - 2 * L)])
energy = np.abs(baseband[L:]) ** 2
rr = np.array([energy[d:d + L].sum() for d in range(baseband.size - 2 * L)])
# The metric is a normalised ratio, so in the noise-only lead-in (tiny rr) it
# takes arbitrary values that would dwarf the real plateau on the plot. Divide
# only where the received energy clears a quarter of peak and leave the rest at
# zero. Gating first is also what keeps M(d) itself exact: R(d) is an energy
# sum, so R^2 goes as amplitude^4 and adding any fixed epsilon to it would tie
# the metric to the amplitude the record happens to be held in — the plateau
# here collapses from 1.02 to 0.0003 by a scale of 1e-5.
loud = rr >= 0.25 * rr.max()
metric = np.zeros(rr.size)
metric[loud] = np.abs(p[loud]) ** 2 / rr[loud] ** 2

# Block 0 is the Schmidl-Cox preamble and block 1 the pilot, so the payload
# blocks start at index 2. The pilot channel estimate and the one-tap
# equalization are the library's own; ofdm_demodulate takes the channel as an
# impulse response, so the estimated H(f) goes back through an inverse FFT.
block = nsc + cp
x = apply_cfo(baseband[start:], cfo)
H = estimate_channel(x[block:2 * block], receiver.pilot_freq, nsc, cp)
n_blocks = x.size // block
constellation = receiver.modulator.constellation
blocks = []
if n_blocks > 2:
    equalized = ofdm_demodulate(x[2 * block: n_blocks * block], nsc, cp,
                                channel=np.fft.ifft(H))
    # The decision-directed common-phase correction is per block, and
    # OFDMReceiver.receive returns decoded bits rather than these symbols, so
    # this last step is the one thing still rebuilt by hand.
    for symbols in equalized.reshape(-1, nsc):
        decided = constellation[np.argmin(
            np.abs(symbols[:, None] - constellation[None, :]), axis=1)]
        blocks.append(symbols * np.exp(-1j * np.angle(np.vdot(decided,
                                                              symbols))))
data_symbols = np.concatenate(blocks) if blocks else np.array([])

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
axes[0, 0].specgram(received, NFFT=256, Fs=fs, noverlap=200, cmap='jet')
axes[0, 0].axhline(fc, color='w', ls='--', lw=1)
axes[0, 0].set_title('received passband spectrogram', loc='left')
axes[0, 0].set_xlabel('Time [s]')
axes[0, 0].set_ylabel('Frequency [Hz]')
axes[0, 0].set_ylim(0, fs / 2)

axes[0, 1].plot(metric)
axes[0, 1].set_title('Schmidl-Cox timing metric', loc='left')
axes[0, 1].set_xlabel('Sample index')
axes[0, 1].set_ylabel('Metric')
axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(np.arange(nsc) - nsc // 2,
                np.fft.fftshift(20 * np.log10(np.abs(H) + 1e-9)))
axes[1, 0].set_title('per-subcarrier channel |H|', loc='left')
axes[1, 0].set_xlabel('Subcarrier')
axes[1, 0].set_ylabel('|H| [dB]')
axes[1, 0].grid(alpha=0.3)

uacpy.plot.plot_scatter(data_symbols, ax=axes[1, 1],
                        title=f"recovered QPSK "
                              f"(CRC {'OK' if crc_ok else 'FAIL'})")
axes[1, 1].scatter(constellation.real, constellation.imag, marker='x', s=80,
                   color='k', zorder=5)

fig.savefig(OUT / 'example_33_ofdm_modem.png', dpi=120)
plt.close(fig)
