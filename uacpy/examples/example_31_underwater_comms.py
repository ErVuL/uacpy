"""Underwater acoustic communications tour.

uacpy.comms end to end — modulate, push through an underwater channel, recover
bits with an adaptive receiver, and measure each stage: BER against AWGN theory
for QPSK and 16-QAM, a DFE reopening a constellation closed by 3-tap ISI, its
learning curve, preamble synchronisation, wideband Doppler-scale estimation,
OFDM over multipath, convolutional coding with Viterbi decoding, DSSS
processing gain, and a JANUS beacon.

Uses: comms.ber_sweep · multipath_channel · simulate_link · DFE ·
detect_preamble · estimate_doppler_scale / compensate_doppler ·
ofdm_modulate / ofdm_demodulate · conv_encode / viterbi_decode · m_sequence ·
JanusPacket · plot_ber_curve · plot_scatter · plot_convergence ·
plot_sync_metric · plot_doppler_ambiguity · plot_subcarriers
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy import comms

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0xACED)
fs = 12000.0

# BER against Eb/N0, measured, for two constellations.
ebn0 = np.arange(0, 13, 2.0)
ber_qpsk = comms.ber_sweep("qpsk", ebn0, 200000, rng=rng)
ber_qam = comms.ber_sweep("16qam", ebn0, 200000, rng=rng)

# A 3-tap ISI channel closes the constellation; a DFE reopens it.
channel = comms.multipath_channel([1.0, 0.6, 0.3], [0.0, 1 / fs, 2 / fs], fs)
raw = comms.simulate_link("qpsk", 16.0, 40000, channel=channel, rng=rng)
equalized = comms.simulate_link("qpsk", 16.0, 40000, channel=channel,
                                equalizer=comms.DFE(n_ff=12, n_fb=6,
                                                    forget=0.995), rng=rng)
print(f"  ISI QPSK @16 dB : raw BER {raw.ber:.2e} → DFE BER "
      f"{equalized.ber:.2e}, final MSE "
      f"{10 * np.log10(equalized.mse[-2000:].mean()):.1f} dB")

# Preamble synchronisation: find a known symbol sequence in a noisy record.
modulator = comms.Modulator("qpsk")
preamble = modulator.modulate(rng.integers(0, 2, 128))
offset = 250
sync_input = np.concatenate([
    np.zeros(offset, complex), preamble,
    0.2 * (rng.standard_normal(400) + 1j * rng.standard_normal(400))])
found, sync_metric = comms.detect_preamble(sync_input, preamble, threshold=0.5)
print(f"  preamble found at sample {found} (true {offset})")

# Doppler on a moving platform is a time SCALE, not a frequency shift: a
# wideband probe is compressed, and the scale is what the estimator searches.
t = np.arange(4000) / fs
probe = np.exp(1j * 2 * np.pi * (1500 * t + 0.5 * 2e5 * t ** 2))
true_scale = 1.5e-3
received = comms.compensate_doppler(probe, -true_scale)   # closing geometry
scale, scales, peak = comms.estimate_doppler_scale(
    received, probe, np.linspace(0, 3e-3, 61))
print(f"  Doppler scale   : estimated {scale * 1e3:.2f}e-3 "
      f"(true {true_scale * 1e3:.2f}e-3)")

# OFDM: a cyclic prefix turns the multipath into per-subcarrier gains.
ofdm_channel = np.zeros(8, complex)
ofdm_channel[[0, 3, 6]] = [1.0, 0.4, 0.2]
ofdm_bits = rng.integers(0, 2, 4096)
ofdm_rx = comms.awgn(comms.apply_channel(
    comms.ofdm_modulate(modulator.modulate(ofdm_bits), 256, 32),
    ofdm_channel), 25.0, rng=rng)
recovered = comms.ofdm_demodulate(ofdm_rx, 256, 32, channel=ofdm_channel)
ber_ofdm = comms.bit_error_rate(
    ofdm_bits, modulator.demodulate(recovered)[:ofdm_bits.size])
print(f"  OFDM/ZF BER     : {ber_ofdm:.2e}")

# Forward error correction: interleaving spreads a burst so Viterbi can fix it.
info_bits = rng.integers(0, 2, 1000)
coded = comms.interleave(comms.conv_encode(info_bits), 16)
flipped = rng.choice(coded.size, int(0.03 * coded.size), replace=False)
coded[flipped] ^= 1
decoded = comms.viterbi_decode(comms.deinterleave(coded, 16))
print(f"  FEC (R=1/2 K=7) : {flipped.size} channel errors → "
      f"{int(np.sum(decoded[:info_bits.size] != info_bits))} after Viterbi")

spreading_code = comms.m_sequence(5, [5, 2])
print(f"  DSSS            : N={spreading_code.size}, processing gain "
      f"{comms.processing_gain_dB(spreading_code):.1f} dB")

janus_bits = comms.JanusPacket(class_id=16, app_type=0).to_bits()
janus_wav = comms.janus_modulate(janus_bits, 48000.0)
out_bits, crc_ok = comms.janus_demodulate(
    comms.awgn(janus_wav, 12.0, rng=rng).real, 48000.0)
packet, _ = comms.JanusPacket.from_bits(out_bits)
print(f"  JANUS 4748      : 64-bit packet → {janus_wav.size / 48000:.2f} s "
      f"FH-BFSK @12 dB → CRC {'OK' if crc_ok else 'FAIL'} "
      f"(class {packet.class_id})")

fig, axes = plt.subplots(3, 2, figsize=(12, 14), constrained_layout=True)
uacpy.plot.plot_ber_curve(ebn0, ber_qpsk, scheme="qpsk", ax=axes[0, 0],
                          label="QPSK meas", title="AWGN link")
uacpy.plot.plot_ber_curve(ebn0, ber_qam, scheme="16qam", ax=axes[0, 0],
                          label="16QAM meas")
uacpy.plot.plot_scatter(equalized.rx_symbols[2000:], ax=axes[0, 1],
                        title=f"QPSK after DFE (BER {equalized.ber:.1e})",
                        color="C0")
axes[0, 1].scatter(modulator.constellation.real, modulator.constellation.imag,
                   marker="x", s=80, color="k", zorder=5)
uacpy.plot.plot_convergence(equalized.mse, ax=axes[1, 0], title="(DFE, RLS)")
uacpy.plot.plot_sync_metric(sync_metric, threshold=0.5, ax=axes[1, 1])
uacpy.plot.plot_doppler_ambiguity(scales, peak, ax=axes[2, 0])
uacpy.plot.plot_subcarriers(ofdm_channel, 256, ax=axes[2, 1],
                            title=f"(ZF BER {ber_ofdm:.1e})")

fig.savefig(OUT / "example_31_underwater_comms.png", dpi=120)
plt.close(fig)
