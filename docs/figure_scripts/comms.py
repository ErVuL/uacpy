"""Figures for ``docs/guide/comms.md``.

The code here is the code shown on the page: each builder is the worked
example, so a figure cannot drift from the snippet that claims to produce it.
Every random stream is seeded, so the figures are reproducible.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from figure_scripts._common import shallow_water

import uacpy
from uacpy import comms
from uacpy.acoustic_signal import lfm_chirp
from uacpy.models import Bellhop, RunMode
from uacpy.visualization import (
    plot_ber_curve,
    plot_channel,
    plot_convergence,
    plot_doppler_ambiguity,
    plot_eye_diagram,
    plot_scatter,
    plot_sync_metric,
)

GUIDE = True

BAUD = 1000.0           # symbol rate (Hz) used by the single-carrier figures
ISI_GAINS = [1.0, 0.7, 0.45]
ISI_DELAYS = [0.0, 2.0 / BAUD, 5.0 / BAUD]      # s — 0, 2 and 5 symbols late

ROW3 = (13.0, 4.2)
ROW2 = (11.0, 4.2)


def _isi_channel(sample_rate=BAUD):
    """The three-path ISI channel, sampled at ``sample_rate``."""
    return comms.multipath_channel(ISI_GAINS, ISI_DELAYS, sample_rate)


def _smooth(x, n=200):
    """Box-car average — a learning curve is unreadable un-smoothed."""
    return np.convolve(np.asarray(x, dtype=float), np.ones(n) / n, mode='valid')


# ── 1. Equalisation: the constellation, before and after ─────────────────────


def equalization():
    """QPSK through 3-path ISI, raw and through a DFE, plus the learning curve."""
    rng = np.random.default_rng(0)
    channel = _isi_channel()
    raw = comms.simulate_link('qpsk', 16.0, 40000, channel=channel, rng=rng)
    rls = comms.simulate_link('qpsk', 16.0, 40000, channel=channel, rng=rng,
                              equalizer=comms.DFE(n_ff=16, n_fb=8, forget=0.997))
    lms = comms.simulate_link('qpsk', 16.0, 40000, channel=channel, rng=rng,
                              equalizer=comms.DFE(n_ff=16, n_fb=8, step=0.005))

    ideal = comms.Modulator('qpsk').constellation
    fig, axes = plt.subplots(1, 3, figsize=ROW3)
    plot_scatter(raw.rx_symbols[2000:], axes[0], ideal=ideal,
                 title=f'No equalizer — BER {raw.ber:.1e}')
    plot_scatter(rls.rx_symbols[2000:], axes[1], ideal=ideal,
                 title=f'DFE, 16 ff / 8 fb — BER {rls.ber:.1e}')
    plot_convergence(_smooth(lms.mse), axes[2], label=f'LMS, step 0.005 '
                     f'(BER {lms.ber:.1e})', color='C1')
    plot_convergence(_smooth(rls.mse), axes[2], label='RLS, forget 0.997',
                     title='Learning curve (200-symbol average)', color='C0')
    axes[2].set_xlim(0, 6000)
    fig.suptitle('Decision-feedback equalization of a 3-path channel, '
                 'QPSK at Eb/N0 = 16 dB', fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


# ── 2. BER versus Eb/N0, against theory ──────────────────────────────────────


def ber():
    """Measured BER against the closed-form AWGN curves, uncoded and coded."""
    rng = np.random.default_rng(0)
    ebn0 = np.arange(0.0, 13.0, 2.0)
    ber_qpsk = comms.ber_sweep('qpsk', ebn0, 200000, rng=rng)
    ber_qam = comms.ber_sweep('16qam', ebn0, 200000, rng=rng)

    code = comms.ConvCode(interleave_depth=16)
    ec_n0 = np.arange(-3.0, 2.1, 1.0)
    ber_coded = comms.ber_sweep('qpsk', ec_n0, 20000, code=code, rng=rng)

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    plot_ber_curve(ebn0, ber_qpsk, ax, scheme='qpsk', label='QPSK measured')
    plot_ber_curve(ebn0, ber_qam, ax, scheme='16qam', label='16-QAM measured',
                   color='C1')
    # The sweep sets the noise from the energy of the bits it puts on the
    # channel, so a rate-1/2 code must be re-plotted per *information* bit.
    ax.semilogy(ec_n0 - 10 * np.log10(code.rate), np.maximum(ber_coded, 1e-12),
                marker='s', color='C2', label='QPSK + R=1/2 K=7 Viterbi')
    ax.set_ylim(1e-6, 1.0)
    ax.set_title('Bit error rate over an AWGN channel', loc='left')
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ── 3. Pulse shaping and the eye ─────────────────────────────────────────────


def eye():
    """The eye a matched RRC pair opens, and the same eye closed by ISI."""
    rng = np.random.default_rng(0)
    sps = 8
    mod = comms.Modulator('qpsk')
    symbols = mod.modulate(rng.integers(0, 2, 8000))
    tx = comms.pulse_shape(symbols, sps, rolloff=0.25, span=8)

    clean = comms.rrc_matched_filter(comms.awgn(tx, 25.0, rng=rng), sps)
    faded = comms.rrc_matched_filter(
        comms.awgn(comms.apply_channel(tx, _isi_channel(BAUD * sps)), 25.0,
                   rng=rng), sps)

    fig, axes = plt.subplots(1, 2, figsize=ROW2, sharey=True)
    skip = 16 * sps                                  # past the filter transient
    for ax, sig, name in ((axes[0], clean, 'AWGN only'),
                          (axes[1], faded, 'through the 3-path channel')):
        plot_eye_diagram(sig[skip:skip + 400 * sps] / np.abs(clean).max(), sps,
                         ax, n_symbols=2, title=f'Eye — {name}')
        ax.axvline(1.0, color='r', ls='--', lw=1)
    axes[0].set_ylim(-1.6, 1.6)
    fig.suptitle('Root-raised-cosine pulse shaping, QPSK at 8 samples/symbol',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


# ── 4. Finding the frame ─────────────────────────────────────────────────────


def _sc_metric(x, n_subcarriers):
    """Schmidl & Cox timing metric — what ``schmidl_cox_sync`` maximises.

    ``schmidl_cox_sync`` returns only ``(start, cfo)``; the metric itself is
    recomputed here so it can be plotted.
    """
    L = int(n_subcarriers) // 2
    n = x.size - 2 * L
    a = np.concatenate(([0j], np.cumsum(np.conj(x[:-L]) * x[L:])))
    p = a[L:L + n] - a[:n]
    e = np.concatenate(([0.0], np.cumsum(np.abs(x[L:]) ** 2)))
    rr = e[L:L + n] - e[:n]
    metric = np.abs(p) ** 2 / (rr ** 2 + 1e-12)
    metric[rr < 0.25 * rr.max()] = 0.0
    return metric


def sync():
    """Two ways to find a frame: a preamble matched filter, and Schmidl-Cox."""
    rng = np.random.default_rng(0)

    preamble = comms.Modulator('qpsk').modulate(rng.integers(0, 2, 256))
    offset = 400
    payload = comms.Modulator('qpsk').modulate(rng.integers(0, 2, 1200))
    single = comms.awgn(np.concatenate([np.zeros(offset, complex),
                                        preamble, payload]), 6.0, rng=rng)
    start, metric = comms.detect_preamble(single, preamble, threshold=0.4)

    baseband, _, _, _ = _ofdm_frame()
    sc = _sc_metric(baseband, 256)
    sc_start, cfo = comms.schmidl_cox_sync(baseband, 256, 32)

    fig, axes = plt.subplots(1, 2, figsize=ROW2)
    plot_sync_metric(metric, axes[0], threshold=0.4,
                     title=f'Preamble matched filter — peak at {start} '
                           f'(true {offset})')
    axes[0].set_xlabel('Symbol index')
    plot_sync_metric(sc, axes[1],
                     title=f'Schmidl-Cox timing metric — start {sc_start}, '
                           f'CFO {cfo * 1e3:+.2f}e-3 cycles/sample')
    axes[1].axvline(sc_start, color='g', ls=':', lw=1.2, label='detected start')
    axes[1].set_ylabel('Timing metric')
    axes[1].legend(fontsize=9)
    fig.suptitle('Frame synchronization: single-carrier and OFDM',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


# ── 5. OFDM ──────────────────────────────────────────────────────────────────

MESSAGE = b'the cyclic prefix turns long multipath into flat subcarriers'
N_SC, CP = 256, 32
OFDM_SNR_DB = 22.0


def _ofdm_frame():
    """One OFDM frame through a dispersive channel with a delay and a CFO.

    Returns ``(baseband, transmitter, channel, crc_ok)`` — shared by the sync
    and OFDM figures so both show the same frame.
    """
    rng = np.random.default_rng(0)
    code = comms.ConvCode(interleave_depth=16)
    tx = comms.OFDMTransmitter('qpsk', N_SC, CP, code=code)
    frame = tx.transmit(comms.pack_frame(MESSAGE))

    channel = comms.multipath_channel([1.0, 0.55, 0.3], [0.0, 9.0, 21.0],
                                      sample_rate=1.0)     # delays in samples
    rx = comms.awgn(comms.apply_channel(frame, channel), OFDM_SNR_DB, rng=rng)
    rx = np.concatenate([np.zeros(137, dtype=complex), rx])
    rx *= np.exp(2j * np.pi * 2.0e-3 * np.arange(rx.size))   # residual CFO

    bits = comms.OFDMReceiver('qpsk', N_SC, CP, code=code).receive(rx)
    _, crc_ok = comms.unpack_frame(bits)
    return rx, tx, channel, crc_ok


def ofdm():
    """The pilot channel estimate, and the constellation it equalizes."""
    rx, tx, channel, crc_ok = _ofdm_frame()
    blk = N_SC + CP

    start, cfo = comms.schmidl_cox_sync(rx, N_SC, CP)
    x = comms.apply_cfo(rx[start:], cfo)
    h_est = comms.estimate_channel(x[blk:2 * blk], tx.pilot_freq, N_SC, CP)
    h_true = np.fft.fft(channel, N_SC)

    ideal = tx.modulator.constellation
    blocks = []
    for b in range(2, x.size // blk):
        block = np.fft.fft(x[b * blk + CP:b * blk + CP + N_SC]) / np.sqrt(N_SC)
        d = block / h_est                                  # one tap per carrier
        # common-phase-error correction, as OFDMReceiver does per block
        decisions = ideal[np.argmin(np.abs(d[:, None] - ideal[None, :]), axis=1)]
        blocks.append(d * np.exp(-1j * np.angle(np.vdot(decisions, d))))
    symbols = np.concatenate(blocks)
    h_db = np.tile(20 * np.log10(np.abs(h_est)), len(blocks))

    k = np.arange(N_SC) - N_SC // 2
    fig, axes = plt.subplots(1, 2, figsize=ROW2)
    axes[0].plot(k, np.fft.fftshift(20 * np.log10(np.abs(h_true) + 1e-12)),
                 'k-', lw=1.2, label='true |H|')
    axes[0].plot(k, np.fft.fftshift(20 * np.log10(np.abs(h_est) + 1e-12)),
                 'C0.', ms=3, label='pilot LS estimate')
    axes[0].set_xlabel('Subcarrier')
    axes[0].set_ylabel('|H| (dB)')
    axes[0].set_title('Channel across the 256 subcarriers', loc='left')
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    ax = axes[1]
    dots = ax.scatter(symbols.real, symbols.imag, c=h_db, cmap='viridis',
                      s=7, alpha=0.8, vmin=-15, vmax=6)
    ax.scatter(ideal.real, ideal.imag, marker='x', s=80, color='k', zorder=5)
    fig.colorbar(dots, ax=ax, label='subcarrier |H| (dB)')
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.6, 2.6)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_xlabel('In-phase')
    ax.set_ylabel('Quadrature')
    ax.set_title(f'Equalized QPSK, coloured by |H| — '
                 f'CRC {"OK" if crc_ok else "FAIL"}', loc='left')

    fig.suptitle('OFDM: 256 subcarriers, 32-sample cyclic prefix, '
                 '3-path channel at 22 dB SNR', fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


# ── 6. Doppler ───────────────────────────────────────────────────────────────


def doppler():
    """Doppler-scale estimation from a wideband probe, and its accuracy."""
    rng = np.random.default_rng(0)
    fs = 12000.0
    _, probe = lfm_chirp(1000.0, 5000.0, 1.0, fs)
    scales = np.linspace(-1e-3, 4e-3, 101)

    def record(v):
        """What the moving receiver hears: a dilated probe inside a quiet record."""
        heard = comms.compensate_doppler(probe, -comms.doppler_from_speed(v))
        return comms.awgn(np.concatenate([np.zeros(500), heard.real,
                                          np.zeros(500)]), 10.0, rng=rng)

    speed = 2.3                                     # m/s, closing
    a_true = comms.doppler_from_speed(speed)
    a_hat, _, peak = comms.estimate_doppler_scale(record(speed), probe, scales)

    speeds = np.arange(0.0, 4.1, 0.5)
    estimates = np.array([
        comms.estimate_doppler_scale(record(v), probe, scales)[0]
        for v in speeds])

    fig, axes = plt.subplots(1, 2, figsize=ROW2)
    plot_doppler_ambiguity(scales, peak, axes[0],
                           title=f'1 s LFM probe at 10 dB SNR — '
                                 f'estimate {a_hat * 1e3:.2f} e-3')
    axes[0].axvline(a_true * 1e3, color='k', ls=':', lw=1,
                    label=f'true a = {a_true * 1e3:.2f} e-3 ({speed} m/s)')
    axes[0].legend(fontsize=9)
    axes[1].plot(speeds, estimates * 1e3, 'o-', label='estimated')
    axes[1].plot(speeds, [comms.doppler_from_speed(v) * 1e3 for v in speeds],
                 'k--', label='a = v/c')
    axes[1].set_xlabel('Closing speed (m/s)')
    axes[1].set_ylabel('Doppler scale a (×10⁻³)')
    axes[1].set_title('Estimate versus truth', loc='left')
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)
    fig.suptitle('Wideband Doppler: the channel dilates time, it does not '
                 'just shift the carrier', fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


# ── 7. DSSS ──────────────────────────────────────────────────────────────────


def dsss():
    """Spreading buries the signal under the noise, and buys back the SNR."""
    from scipy.signal import welch

    rng = np.random.default_rng(0)
    code = comms.m_sequence(5, [5, 2])
    gain = comms.processing_gain_db(code)
    mod = comms.Modulator('bpsk')

    chip_snr = np.arange(-18.0, 3.1, 3.0)
    ber_spread, ber_plain = [], []
    for snr in chip_snr:
        bits = rng.integers(0, 2, 20000)
        symbols = mod.modulate(bits)
        chips = comms.awgn(comms.spread(symbols, code), snr, rng=rng)
        ber_spread.append(
            comms.bit_error_rate(bits, mod.demodulate(comms.despread(chips, code))))
        ber_plain.append(comms.bit_error_rate(
            bits, mod.demodulate(comms.awgn(symbols, snr, rng=rng))))

    # same energy, same duration: one symbol held over N chips vs spread over N
    symbols = mod.modulate(rng.integers(0, 2, 2000)).real
    held = np.repeat(symbols, code.size)
    chips = comms.spread(symbols, code).real
    noise = np.sqrt(10.0) * rng.standard_normal(held.size)      # chip SNR -10 dB
    f_h, p_h = welch(held + noise, nperseg=512)
    f_s, p_s = welch(chips + noise, nperseg=512)
    _, p_n = welch(noise, nperseg=512)

    fig, axes = plt.subplots(1, 2, figsize=ROW2)
    axes[0].plot(f_h, 10 * np.log10(p_h), label=f'unspread (held {code.size}x)')
    axes[0].plot(f_s, 10 * np.log10(p_s), label='spread over 31 chips')
    axes[0].plot(f_s, 10 * np.log10(p_n), 'k--', lw=1, label='noise alone')
    axes[0].set_xlabel('Normalized frequency (cycles/chip)')
    axes[0].set_ylabel('PSD (dB)')
    axes[0].set_title('Same energy, spread 31-fold', loc='left')
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    fine = np.linspace(chip_snr.min(), chip_snr.max(), 200)
    axes[1].semilogy(chip_snr, np.maximum(ber_plain, 1e-12), 'o-',
                     label='BPSK, no spreading')
    axes[1].semilogy(chip_snr, np.maximum(ber_spread, 1e-12), 's-',
                     label='despread, N = 31')
    axes[1].semilogy(fine, comms.ber_theory('bpsk', fine + gain), 'k--', lw=1,
                     label=f'theory at SNR + {gain:.1f} dB')
    axes[1].set_ylim(1e-6, 1.0)
    axes[1].set_xlabel('Chip SNR (dB)')
    axes[1].set_ylabel('BER')
    axes[1].set_title(f'Processing gain = {gain:.1f} dB', loc='left')
    axes[1].grid(which='both', alpha=0.3)
    axes[1].legend(fontsize=9)
    fig.suptitle('Direct-sequence spread spectrum with a length-31 m-sequence',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


# ── 8. JANUS ─────────────────────────────────────────────────────────────────


def janus():
    """A STANAG 4748 beacon: the frequency hops, and the detector that finds them."""
    from uacpy.comms import janus as janus_mod

    rng = np.random.default_rng(0)
    fs = 48000.0
    app_data = np.zeros(34, dtype=int)
    app_data[:16] = comms.bytes_to_bits(b'SOS')[:16]
    packet = comms.JanusPacket(class_id=16, app_type=0, app_data=app_data,
                               mobility=1)
    waveform = comms.janus_transmit(packet, fs)

    rx = np.concatenate([np.zeros(8000), waveform, np.zeros(4000)])
    rx = comms.awgn(rx, 6.0, rng=rng).real
    start, statistic = comms.janus_detect(rx, fs)
    decoded, crc_ok = comms.janus_receive(rx, fs)

    fig, axes = plt.subplots(1, 2, figsize=ROW2)
    axes[0].specgram(rx, NFFT=256, Fs=fs, noverlap=224, cmap='viridis')
    axes[0].set_ylim(janus_mod.FC_INITIAL - janus_mod.BW_INITIAL,
                     janus_mod.FC_INITIAL + janus_mod.BW_INITIAL)
    axes[0].axhline(janus_mod.FC_INITIAL, color='w', ls='--', lw=0.8)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_title('FH-BFSK over 13 tone pairs', loc='left')
    axes[1].plot(statistic, lw=0.9)
    axes[1].axvline(int(np.argmax(statistic)), color='g', ls=':', lw=1.2,
                    label=f'preamble at sample {start}')
    axes[1].set_xlabel('Alignment column (¼ chip)')
    axes[1].set_ylabel('CFAR statistic')
    axes[1].set_title('GO-CFAR preamble detector', loc='left')
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)
    fig.suptitle(f'JANUS baseline packet at 6 dB SNR — class '
                 f'{decoded.class_id}, CRC {"OK" if crc_ok else "FAIL"}',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


# ── 9. A link through a modelled channel ─────────────────────────────────────


def bellhop_channel():
    """Drive the modem with a Bellhop arrival structure instead of a guess."""
    env, source, _ = shallow_water()
    point = uacpy.Receiver(depths=60.0, ranges=3000.0)
    arrivals = Bellhop(n_beams=4000, alpha=(-10.0, 10.0)).run(
        env, source, point, run_mode=RunMode.ARRIVALS)

    gains = arrivals.amplitudes * np.exp(1j * arrivals.phases)
    delays = arrivals.delays - arrivals.delays.min()
    channel = comms.multipath_channel(gains, delays, BAUD)
    channel /= np.abs(channel).max()

    rng = np.random.default_rng(0)
    dfe = comms.DFE(n_ff=24, n_fb=32, forget=0.999)
    link = comms.simulate_link('qpsk', 20.0, 20000, channel=channel,
                               equalizer=dfe, n_train=2000, rng=rng)

    fig, axes = plt.subplots(1, 3, figsize=ROW3)
    plot_channel(channel, BAUD, (axes[0], axes[1]),
                 title=f'{len(arrivals)} arrivals → {channel.size} taps at 1 kBd')
    plot_scatter(link.rx_symbols[4000:], axes[2],
                 ideal=comms.Modulator('qpsk').constellation,
                 title=f'QPSK after the DFE — BER {link.ber:.1e}')
    fig.suptitle('A 1 kBd QPSK link over the modelled 3 km shallow-water '
                 'channel, Eb/N0 = 20 dB', fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


FIGURES = {
    'comms_equalization': equalization,
    'comms_ber': ber,
    'comms_eye': eye,
    'comms_sync': sync,
    'comms_ofdm': ofdm,
    'comms_doppler': doppler,
    'comms_dsss': dsss,
    'comms_janus': janus,
    'comms_bellhop_channel': bellhop_channel,
}
