"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 33: OFDM Underwater Modem (text -> .wav -> text)
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    The multicarrier counterpart of example 32 — send a real payload with OFDM
    and recover it through a dispersive, Doppler-shifted underwater channel:
      • frame a text message (length header + CRC-32)
      • OFDMTransmitter: FEC + QPSK subcarriers + Schmidl-Cox preamble + pilot +
        cyclic prefix, oversampled and upconverted to passband (written to .wav)
      • channel: long multipath + clock-skew Doppler + delay + AWGN
      • OFDMReceiver (the practical UW receiver): estimate + resample the common
        Doppler, Schmidl-Cox timing/CFO, FFT, pilot channel estimate, one-tap
        per-subcarrier equalize + per-symbol common-phase tracking, Viterbi
      • verify the recovered bytes against the CRC

FEATURES DEMONSTRATED:
    ✓ OFDMTransmitter.transmit_passband · .wav write
    ✓ OFDMReceiver.receive_passband (resample + Schmidl-Cox + residual CFO)
    ✓ passband spectrogram · Schmidl-Cox timing metric · per-subcarrier channel
      magnitude · recovered constellation
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
from uacpy.comms.ofdm import schmidl_cox_sync, apply_cfo  # noqa: E402


def _write_wav(path, signal, fs):
    x = signal / (np.max(np.abs(signal)) + 1e-12)
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(fs))
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def main():
    print("═" * 80)
    print("EXAMPLE 33: OFDM Underwater Modem (text -> .wav -> text)")
    print("═" * 80)
    rng = np.random.default_rng(0xACED)
    fs, fc, os = 96000.0, 24000.0, 4
    nsc, cp = 256, 32

    message = (b"OFDM underwater modem: a cyclic prefix turns long multipath into "
               b"flat subcarriers. Resample the Doppler, Schmidl-Cox the rest!")
    frame_bits = comms.pack_frame(message)
    print(f"\n  payload    : {len(message)} bytes -> {frame_bits.size} framed bits")

    code = comms.ConvCode(interleave_depth=16)
    tx = comms.OFDMTransmitter("qpsk", nsc, cp, code=code)
    wav = tx.transmit_passband(frame_bits, fs, fc, oversample=os)
    wav_path = OUTPUT_DIR / "example_33_ofdm.wav"
    _write_wav(wav_path, wav, fs)
    band = fs / os
    print(f"  waveform   : {wav.size} real samples, {wav.size/fs*1e3:.0f} ms @ "
          f"{fs/1e3:.0f} kHz, band {(fc-band/2)/1e3:.0f}-{(fc+band/2)/1e3:.0f} kHz")
    print(f"  wrote      : {wav_path.name}")

    # --- channel: long multipath + clock-skew Doppler + delay + AWGN ---
    h = np.zeros(60); h[[0, 25, 51]] = [1.0, 0.5, 0.3]
    rx = np.convolve(wav, h)
    rx = resample_poly(rx, 100015, 100000)              # 150 ppm Doppler
    rx = np.concatenate([np.zeros(37), rx])
    snr_db = 24.0
    rx = rx + np.sqrt(np.mean(rx ** 2) / 10 ** (snr_db / 10)) * rng.standard_normal(rx.size)
    print(f"  channel    : 3-path ({h.nonzero()[0][-1]}-sample spread) + 150 ppm "
          f"Doppler, {snr_db:.0f} dB SNR")

    # --- receive ---
    rxr = comms.OFDMReceiver("qpsk", nsc, cp, code=code)
    scale, _, _ = comms.estimate_doppler_scale(
        rx, comms.upconvert(resample_poly(rxr.preamble, os, 1), fs, fc),
        np.linspace(-3e-4, 3e-4, 31))
    out_bits = rxr.receive_passband(rx, fs, fc, oversample=os, doppler_scale=scale)
    payload, crc_ok = comms.unpack_frame(out_bits)
    print(f"\n  Doppler est: {scale*1e3:+.2f}e-3  (resampled out)")
    print(f"  CRC        : {'OK' if crc_ok else 'FAIL'}")
    print(f"  recovered  : {payload[:60]!r}{'...' if len(payload) > 60 else ''}")
    print(f"  MATCH      : {payload == message}")

    # recompute internals for the figure
    bb = rxr.from_passband(rx, fs, fc, oversample=os, doppler_scale=scale)
    start, cfo = schmidl_cox_sync(bb, nsc, cp)
    L = nsc // 2
    a = np.conj(bb[:-L]) * bb[L:]
    p = np.array([a[d:d + L].sum() for d in range(bb.size - 2 * L)])
    energy = np.abs(bb[L:]) ** 2
    rr = np.array([energy[d:d + L].sum() for d in range(bb.size - 2 * L)])
    metric = np.abs(p) ** 2 / (rr ** 2 + 1e-12)
    metric[rr < 0.25 * rr.max()] = 0.0
    blk = nsc + cp
    x = apply_cfo(bb[start:], cfo)
    pilot_spec = np.fft.fft(x[blk + cp: blk + cp + nsc]) / np.sqrt(nsc)
    H = pilot_spec / (rxr.pilot_freq + 1e-12)
    dat = []
    cst = rxr.modulator.constellation
    for b in range(2, x.size // blk):
        d = np.fft.fft(x[b * blk + cp: b * blk + cp + nsc]) / np.sqrt(nsc) / (H + 1e-12)
        dec = cst[np.argmin(np.abs(d[:, None] - cst[None, :]), axis=1)]
        dat.append(d * np.exp(-1j * np.angle(np.vdot(dec, d))))
    data_syms = np.concatenate(dat) if dat else np.array([])

    # ----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.specgram(rx, NFFT=256, Fs=fs, noverlap=200, cmap='jet')
    ax.axhline(fc, color='w', ls='--', lw=1)
    ax.set_title('[ofdm] received passband spectrogram', loc='left')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Frequency [Hz]'); ax.set_ylim(0, fs / 2)

    ax = axes[0, 1]
    ax.plot(metric)
    ax.set_title('[ofdm] Schmidl-Cox timing metric', loc='left')
    ax.set_xlabel('Sample index'); ax.set_ylabel('Metric'); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(np.arange(nsc) - nsc // 2, np.fft.fftshift(20 * np.log10(np.abs(H) + 1e-9)))
    ax.set_title('[ofdm] per-subcarrier channel |H|', loc='left')
    ax.set_xlabel('Subcarrier'); ax.set_ylabel('|H| [dB]'); ax.grid(alpha=0.3)

    comms.scatter(data_syms, ax=axes[1, 1],
                  title=f"recovered QPSK (CRC {'OK' if crc_ok else 'FAIL'})")
    axes[1, 1].scatter(cst.real, cst.imag, marker='x', s=80, color='k', zorder=5)

    out_path = OUTPUT_DIR / "example_33_ofdm_modem.png"
    fig.savefig(out_path, dpi=120)
    print(f"\n  saved      : {out_path.name}")
    plt.close(fig)


if __name__ == "__main__":
    main()
