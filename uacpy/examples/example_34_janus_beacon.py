"""JANUS standard beacon (NATO STANAG 4748).

Build, transmit and decode a standards-compliant JANUS baseline packet — the
open NATO underwater interoperability protocol. A 64-bit packet with CRC-8,
rate-1/2 K=9 convolutional coding and depth-13 interleaving, sent as FH-BFSK in
the initial band, through delay + reverberation + noise, then detected and
decoded.

Uses: comms.JanusPacket · janus.janus_modulate / janus_detect /
janus_demodulate · uacpy.io.write_wav
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy import comms
from uacpy.comms import janus

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0xACED)
fs = 48000.0

# Class user id 16 is "NATO JANUS reference implementation", whose app type 0
# plugin carries an 8-bit station identifier. The 34 app-data bits are filled
# arbitrarily: this example exercises the waveform, not the plugin's fields.
app_data = np.zeros(34, dtype=int)
app_data[:16] = comms.bytes_to_bits(b"SOS")[:16]
packet = comms.JanusPacket(class_id=16, app_type=0, app_data=app_data,
                           mobility=1, tx_rx=1)
bits = packet.to_bits()
print(f"  packet   : 64 bits, v{janus.JANUS_VERSION}, class {packet.class_id}, "
      f"app type {packet.app_type}")
print(f"  band     : Fc={janus.FC_INITIAL / 1e3:.2f} kHz, "
      f"Bw={janus.BW_INITIAL / 1e3:.2f} kHz, Cd=6.25 ms")

# 32 preamble chips + 144 data chips of FH-BFSK.
waveform = janus.janus_modulate(bits, fs)
uacpy.io.write_wav(OUT / 'example_34_janus.wav', waveform, fs,
                   metadata={'title': 'JANUS baseline packet',
                             'comment': 'NATO STANAG 4748, initial band'})
print(f"  waveform : {waveform.size} samples, {waveform.size / fs:.2f} s @ "
      f"{fs / 1e3:.0f} kHz")

# Channel: propagation delay, a 30 ms reverberation echo, then noise.
received = np.concatenate([np.zeros(811), waveform])
delay = int(0.03 * fs)
echo = np.zeros_like(received)
echo[delay:] = 0.4 * received[:received.size - delay]
received = received + echo
snr_dB = 12.0
received = received + np.sqrt(
    np.mean(waveform ** 2) / 10 ** (snr_dB / 10)) * rng.standard_normal(
        received.size)

start, metric = janus.janus_detect(received, fs)
out_bits, crc_ok = janus.janus_demodulate(received, fs)
decoded, _ = janus.JanusPacket.from_bits(out_bits)
print(f"  preamble : detected at sample {start} (GO-CFAR)")
print(f"  CRC      : {'OK' if crc_ok else 'FAIL'}")
print(f"  decoded  : class {decoded.class_id}, app type {decoded.app_type}, "
      f"mobility {decoded.mobility}, "
      f"payload match {np.array_equal(decoded.app_data, app_data)}")

fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
axes[0].specgram(waveform, NFFT=256, Fs=fs, noverlap=224, cmap='jet')
axes[0].axhline(janus.FC_INITIAL, color='w', ls='--', lw=0.8)
axes[0].set_ylim(janus.FC_INITIAL - janus.BW_INITIAL,
                 janus.FC_INITIAL + janus.BW_INITIAL)
axes[0].set_title('FH-BFSK waveform — hopping over 13 tone pairs', loc='left')
axes[0].set_xlabel('Time [s]')
axes[0].set_ylabel('Frequency [Hz]')

axes[1].plot(metric)
axes[1].axvline(int(np.argmax(metric)), color='g', ls=':', lw=1,
                label='detected preamble')
axes[1].set_title('GO-CFAR preamble detection statistic', loc='left')
axes[1].set_xlabel('Alignment column (¼-chip)')
axes[1].set_ylabel('CFAR statistic')
axes[1].grid(alpha=0.3)
axes[1].legend()

fig.savefig(OUT / 'example_34_janus_beacon.png', dpi=120)
plt.close(fig)
