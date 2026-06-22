"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 32: Real-Data Underwater Modem (text -> .wav -> text)
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Transmit a real byte payload over the full passband physical layer and
    recover it through a realistic underwater channel:
      • frame a text message (length header + CRC-32)
      • Transmitter: FEC + QPSK + preamble + RRC pulse shaping + upconvert
      • write the real passband signal to a .wav file (what a transducer emits)
      • channel: sparse multipath + clock-skew Doppler + delay + AWGN
      • Receiver: downconvert + matched filter + Gardner timing recovery +
        preamble frame-sync + adaptive DFE with carrier PLL + Viterbi decode
      • verify the recovered bytes against the CRC

FEATURES DEMONSTRATED:
    ✓ pack_frame/unpack_frame · Transmitter.transmit_passband · .wav write
    ✓ Receiver.receive_passband (timing recovery + sync + DFE/PLL + FEC)
    ✓ passband spectrogram · recovered constellation · sync metric · convergence
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
from scipy.signal import resample_poly  # noqa: E402

from uacpy import comms  # noqa: E402
from uacpy.visualization import plot_scatter, plot_convergence, plot_sync_metric  # noqa: E402


def _write_wav(path, signal, fs):
    """Write a real signal to a 16-bit mono .wav (normalized)."""
    x = signal / (np.max(np.abs(signal)) + 1e-12)
    pcm = (x * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(fs))
        w.writeframes(pcm.tobytes())


def main():
    print("═" * 80)
    print("EXAMPLE 32: Real-Data Underwater Modem (text -> .wav -> text)")
    print("═" * 80)
    rng = np.random.default_rng(0xACED)
    fs, fc, sps = 96000.0, 24000.0, 8

    message = (b"uacpy underwater acoustic modem -- real data over a simulated "
               b"ocean channel. The quick brown fox jumps over 13 lazy dogs!")
    frame_bits = comms.pack_frame(message)
    print(f"\n  payload    : {len(message)} bytes -> {frame_bits.size} framed bits")

    # --- transmit: FEC + QPSK + preamble + RRC + upconvert ---
    code = comms.ConvCode(interleave_depth=16)
    tx = comms.Transmitter("qpsk", code=code, preamble=256)
    wav = tx.transmit_passband(frame_bits, fs, fc, sps=sps)
    wav_path = OUTPUT_DIR / "example_32_modem.wav"
    _write_wav(wav_path, wav, fs)
    print(f"  waveform   : {wav.size} real samples, {wav.size/fs*1e3:.0f} ms @ "
          f"{fs/1e3:.0f} kHz, carrier {fc/1e3:.0f} kHz")
    print(f"  wrote      : {wav_path.name}")

    # --- underwater channel: multipath + clock-skew Doppler + delay + AWGN ---
    h = np.zeros(40); h[[0, 17, 33]] = [1.0, 0.4, 0.2]
    rx = np.convolve(wav, h)
    rx = resample_poly(rx, 100020, 100000)              # 200 ppm clock skew
    rx = np.concatenate([np.zeros(11), rx])             # propagation delay
    snr_db = 22.0
    rx = rx + np.sqrt(np.mean(rx ** 2) / 10 ** (snr_db / 10)) * rng.standard_normal(rx.size)
    print(f"  channel    : 3-path multipath, 200 ppm Doppler, {snr_db:.0f} dB SNR")

    # --- receive: downconvert + MF + timing recovery + sync + DFE/PLL + FEC ---
    dfe = comms.DFE(n_ff=16, n_fb=6, forget=0.997, pll_bandwidth=0.04)
    rxr = comms.Receiver("qpsk", code=code, equalizer=dfe, preamble=256)
    syms = rxr.from_passband(rx, fs, fc, sps=sps)
    start, sync_metric = comms.detect_preamble(syms, rxr.preamble, threshold=0.4)
    out_bits = rxr.receive(syms)
    payload, crc_ok = comms.unpack_frame(out_bits)
    print(f"\n  preamble   : found at symbol {start}")
    print(f"  CRC        : {'OK' if crc_ok else 'FAIL'}")
    print(f"  recovered  : {payload[:60]!r}{'...' if len(payload) > 60 else ''}")
    print(f"  MATCH      : {payload == message}")

    # equalized constellation (re-run to grab symbols + mse for the plot)
    aligned = syms[start:]
    delay = dfe.n_ff // 2
    ref = np.concatenate([np.zeros(delay, dtype=complex), rxr.preamble])
    eq, mse = comms.DFE(n_ff=16, n_fb=6, forget=0.997, pll_bandwidth=0.04).equalize(
        aligned, rxr.modulator.constellation, train=ref)
    payload_syms = eq[delay + rxr.preamble.size:]

    # ----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.specgram(rx, NFFT=256, Fs=fs, noverlap=200, cmap='jet')
    ax.axhline(fc, color='w', ls='--', lw=1)
    ax.set_title('[modem] received passband spectrogram', loc='left')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Frequency [Hz]'); ax.set_ylim(0, fs / 2)

    plot_scatter(payload_syms[200:], ax=axes[0, 1],
                  title=f"recovered QPSK (CRC {'OK' if crc_ok else 'FAIL'})")
    axes[0, 1].scatter(rxr.modulator.constellation.real,
                       rxr.modulator.constellation.imag, marker='x', s=80,
                       color='k', zorder=5)

    plot_sync_metric(sync_metric, threshold=0.4, ax=axes[1, 0])
    plot_convergence(mse, ax=axes[1, 1], title='(DFE + carrier PLL)')

    out_path = OUTPUT_DIR / "example_32_realdata_modem.png"
    fig.savefig(out_path, dpi=120)
    print(f"\n  saved      : {out_path.name}")
    plt.close(fig)


if __name__ == "__main__":
    main()
