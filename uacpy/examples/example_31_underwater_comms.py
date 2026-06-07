"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 31: Underwater Acoustic Communications Tour
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    uacpy.comms end to end — modulate, push through an underwater channel, and
    recover bits with an adaptive receiver:
      • BER vs Eb/N0    — QPSK & 16-QAM measured against AWGN theory
      • Equalization    — DFE reopens a constellation closed by 3-tap ISI
      • Convergence     — the equalizer learning curve (MSE)
      • Synchronization — preamble matched-filter metric
      • Doppler         — scale-factor ambiguity from a wideband probe
      • OFDM            — subcarrier response over a multipath channel
    FEC (convolutional + Viterbi) and DSSS processing gain are printed.

FEATURES DEMONSTRATED:
    ✓ Modulator · simulate_link / ber_sweep · ber_curve · scatter
    ✓ DFE (LMS/RLS + PLL) · plot_convergence · detect_preamble / plot_sync_metric
    ✓ estimate_doppler_scale · plot_doppler_ambiguity · OFDM · conv code · DSSS
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from uacpy import comms  # noqa: E402


def main():
    print("═" * 80)
    print("EXAMPLE 31: Underwater Acoustic Communications Tour")
    print("═" * 80)
    rng = np.random.default_rng(0xACED)
    fs = 12000.0

    # --- (A) BER vs Eb/N0: QPSK & 16-QAM ---
    ebn0 = np.arange(0, 13, 2.0)
    ber_qpsk = comms.ber_sweep("qpsk", ebn0, 200000, rng=rng)
    ber_qam = comms.ber_sweep("16qam", ebn0, 200000, rng=rng)

    # --- (B,C) 3-tap ISI channel equalized by a DFE ---
    chan = comms.multipath_channel([0.0, 1 / fs, 2 / fs], [1.0, 0.6, 0.3], fs)
    raw = comms.simulate_link("qpsk", 16.0, 40000, channel=chan, rng=rng)
    dfe = comms.DFE(n_ff=12, n_fb=6, forget=0.995)
    eq = comms.simulate_link("qpsk", 16.0, 40000, channel=chan,
                             equalizer=dfe, rng=rng)
    print(f"\n  ISI QPSK @16 dB : raw BER {raw.ber:.2e} -> DFE BER {eq.ber:.2e}")
    print(f"  DFE final MSE   : {10*np.log10(eq.mse[-2000:].mean()):.1f} dB")

    # --- (D) preamble synchronization ---
    pre = comms.Modulator("qpsk").modulate(rng.integers(0, 2, 128))
    offset = 250
    noise = 0.2 * (rng.standard_normal(400) + 1j * rng.standard_normal(400))
    rx_sync = np.concatenate([np.zeros(offset, complex), pre, noise])
    k_hat, sync_metric = comms.detect_preamble(rx_sync, pre, threshold=0.5)
    print(f"  preamble found at sample {k_hat} (true {offset})")

    # --- (E) Doppler estimation from a wideband chirp probe ---
    tt = np.arange(4000) / fs
    probe = np.exp(1j * 2 * np.pi * (1500 * tt + 0.5 * 2e5 * tt ** 2))
    a_true = 1.5e-3
    received = comms.compensate_doppler(probe, a_true)
    a_hat, scales, peak = comms.estimate_doppler_scale(
        received, probe, np.linspace(0, 3e-3, 61))
    print(f"  Doppler scale   : est {a_hat*1e3:.2f}e-3 (true {a_true*1e3:.2f}e-3)")

    # --- (F) OFDM over a multipath channel ---
    ofdm_chan = np.zeros(8, complex)
    ofdm_chan[[0, 3, 6]] = [1.0, 0.4, 0.2]
    mod = comms.Modulator("qpsk")
    obits = rng.integers(0, 2, 4096)
    sig = comms.ofdm_modulate(mod.modulate(obits), 256, 32)
    orx = comms.awgn(comms.apply_channel(sig, ofdm_chan), 25.0, rng=rng)
    out = comms.ofdm_demodulate(orx, 256, 32, channel=ofdm_chan)
    ber_ofdm = comms.bit_error_rate(obits, mod.demodulate(out)[:obits.size])
    print(f"  OFDM/ZF BER     : {ber_ofdm:.2e}")

    # --- FEC + DSSS (printed) ---
    info = rng.integers(0, 2, 1000)
    coded = comms.interleave(comms.conv_encode(info), 16)
    flip = rng.choice(coded.size, int(0.03 * coded.size), replace=False)
    coded[flip] ^= 1
    dec = comms.viterbi_decode(comms.deinterleave(coded, 16))
    print(f"  FEC (R=1/2 K=7) : {flip.size} channel errors -> "
          f"{int(np.sum(dec[:info.size] != info))} after Viterbi")
    code = comms.m_sequence(5, [5, 2])
    print(f"  DSSS            : N={code.size}, processing gain "
          f"{comms.processing_gain_db(code):.1f} dB")

    # --- JANUS (NATO STANAG 4748): standards-compliant FH-BFSK beacon ---
    jbits = comms.JanusPacket(class_id=16, app_type=0).to_bits()
    jwav = comms.janus_modulate(jbits, 48000.0)
    jrx = comms.awgn(jwav, 12.0, rng=rng).real
    jbits_out, jok = comms.janus_demodulate(jrx, 48000.0)
    jpkt, _ = comms.JanusPacket.from_bits(jbits_out)
    print(f"  JANUS 4748      : 64-bit packet -> {jwav.size/48000:.2f} s FH-BFSK "
          f"@12 dB -> CRC {'OK' if jok else 'FAIL'} (class {jpkt.class_id})")

    # ----------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(12, 14), constrained_layout=True)

    comms.ber_curve(ebn0, ber_qpsk, scheme="qpsk", ax=axes[0, 0],
                    label="QPSK meas", title="AWGN link")
    comms.ber_curve(ebn0, ber_qam, scheme="16qam", ax=axes[0, 0],
                    label="16QAM meas")

    comms.scatter(eq.rx_symbols[2000:], ax=axes[0, 1],
                  title=f"QPSK after DFE (BER {eq.ber:.1e})", color="C0")
    axes[0, 1].scatter(mod.constellation.real, mod.constellation.imag,
                       marker="x", s=80, color="k", zorder=5)

    comms.plot_convergence(eq.mse, ax=axes[1, 0], title="(DFE, RLS)")

    comms.plot_sync_metric(sync_metric, threshold=0.5, ax=axes[1, 1])

    comms.plot_doppler_ambiguity(scales, peak, ax=axes[2, 0])

    comms.plot_subcarriers(ofdm_chan, 256, ax=axes[2, 1],
                           title=f"(ZF BER {ber_ofdm:.1e})")

    out_path = OUTPUT_DIR / "example_31_underwater_comms.png"
    fig.savefig(out_path, dpi=120)
    print(f"\n  saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
