"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 34: JANUS Standard Beacon (NATO STANAG 4748)
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Build, transmit and decode a standards-compliant JANUS baseline packet — the
    open NATO underwater interoperability protocol:
      • assemble a 64-bit JANUS packet (class 16, the NATO reference
        implementation's own class user id) with CRC-8
      • rate-1/2 K=9 convolutional coding + depth-13 interleaving (144 symbols)
      • FH-BFSK waveform in the initial band (Fc=11520 Hz, Bw=4160 Hz, Cd=6.25 ms),
        with the 32-chip detection preamble
      • channel: propagation delay + reverberation tail + AWGN
      • detect the preamble, non-coherently demodulate, decode, verify CRC

FEATURES DEMONSTRATED:
    ✓ JanusPacket.to_bits / from_bits (+ CRC-8) · janus_modulate / janus_demodulate
    ✓ janus_detect (32-chip preamble) · FH-BFSK spectrogram (the frequency hops)
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import wave
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from uacpy import comms  # noqa: E402
from uacpy.comms import janus  # noqa: E402


def _write_wav(path, signal, fs):
    x = signal / (np.max(np.abs(signal)) + 1e-12)
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(fs))
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def main():
    print("═" * 80)
    print("EXAMPLE 34: JANUS Standard Beacon (NATO STANAG 4748)")
    print("═" * 80)
    rng = np.random.default_rng(0xACED)
    fs = 48000.0

    # --- assemble a beacon packet (class 16, app type 0) ---
    # Class user id 16 is "NATO JANUS reference Implementation" and its app
    # type 0 plugin carries an 8-bit Station Identifier; the 34 app-data bits
    # below are filled arbitrarily because this example exercises the
    # waveform, not the plugin's field layout.
    adb = np.zeros(34, dtype=int)
    adb[:16] = comms.bytes_to_bits(b"SOS")[:16]      # arbitrary 34-bit app data block
    pkt = comms.JanusPacket(class_id=16, app_type=0, app_data=adb,
                            mobility=1, tx_rx=1)
    bits = pkt.to_bits()
    print(f"\n  packet     : 64 bits  (v{janus.JANUS_VERSION}, class {pkt.class_id} "
          f"= NATO ref impl, app type {pkt.app_type})")
    print(f"  band       : Fc={janus.FC_INITIAL/1e3:.2f} kHz, Bw={janus.BW_INITIAL/1e3:.2f} "
          f"kHz, FSw=160 Hz, Cd=6.25 ms")

    # --- FH-BFSK waveform ---
    wav = janus.janus_modulate(bits, fs)
    wav_path = OUTPUT_DIR / "example_34_janus.wav"
    _write_wav(wav_path, wav, fs)
    print(f"  waveform   : {wav.size} samples, {wav.size/fs:.2f} s @ {fs/1e3:.0f} kHz "
          f"(32 preamble + 144 data chips)")
    print(f"  wrote      : {wav_path.name}")

    # --- channel: delay + reverberation echo + AWGN ---
    rx = np.concatenate([np.zeros(811), wav])
    echo = np.zeros_like(rx); d = int(0.03 * fs); echo[d:] = 0.4 * rx[:rx.size - d]
    rx = rx + echo
    snr_db = 12.0
    rx = rx + np.sqrt(np.mean(wav ** 2) / 10 ** (snr_db / 10)) * rng.standard_normal(rx.size)
    print(f"  channel    : 811-sample delay + 30 ms echo (0.4) + {snr_db:.0f} dB SNR")

    # --- detect + decode ---
    start, metric = janus.janus_detect(rx, fs)
    out_bits, crc_ok = janus.janus_demodulate(rx, fs)
    out_pkt, _ = janus.JanusPacket.from_bits(out_bits)
    print(f"\n  preamble   : detected at sample {start} (GO-CFAR)")
    print(f"  CRC        : {'OK' if crc_ok else 'FAIL'}")
    print(f"  decoded    : class {out_pkt.class_id}, app type {out_pkt.app_type}, "
          f"mobility {out_pkt.mobility}")
    print(f"  ADB match  : {np.array_equal(out_pkt.app_data, adb)}")

    # ----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)

    ax = axes[0]
    ax.specgram(wav, NFFT=256, Fs=fs, noverlap=224, cmap='jet')
    ax.axhline(janus.FC_INITIAL, color='w', ls='--', lw=0.8)
    ax.set_ylim(janus.FC_INITIAL - janus.BW_INITIAL, janus.FC_INITIAL + janus.BW_INITIAL)
    ax.set_title('[janus] FH-BFSK waveform — frequency hopping over 13 tone pairs',
                 loc='left')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Frequency [Hz]')

    ax = axes[1]
    ax.plot(metric)
    ax.axvline(int(np.argmax(metric)), color='g', ls=':', lw=1, label='detected preamble')
    ax.set_title('[janus] GO-CFAR preamble detection statistic', loc='left')
    ax.set_xlabel('Alignment column (¼-chip)'); ax.set_ylabel('CFAR statistic')
    ax.grid(alpha=0.3); ax.legend()

    out_path = OUTPUT_DIR / "example_34_janus_beacon.png"
    fig.savefig(out_path, dpi=120)
    print(f"\n  saved      : {out_path.name}")
    plt.close(fig)


if __name__ == "__main__":
    main()
