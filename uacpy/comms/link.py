"""What a link achieves, end to end: one simulated run over a channel
(:func:`simulate_link`, returning a :class:`LinkResult`) and a bit-error-rate
sweep over Eb/N0 (:func:`ber_sweep`, returning a :class:`BerCurve`)."""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass

import numpy as np

from uacpy.acoustic_signal._results import PlottedResult

from uacpy.comms.channel import (
    ChannelTaps, apply_channel, awgn, ebn0_to_snr_dB,
)
from uacpy.comms.constellations import SCHEMES, Modulator
from uacpy.comms.coding import _encoded
from uacpy.comms.equalize import complex_gain
from uacpy.comms.metrics import bit_error_rate, evm
from uacpy.core._export import CarrierExport
from uacpy.core._repr import FieldsRepr
from uacpy.core._validate import require_generator
from uacpy.core.exceptions import ConfigurationError


@dataclass(eq=False)
class LinkResult(FieldsRepr, CarrierExport):
    """Outcome of one :func:`simulate_link` run.

    On the export protocol: ``to_dict`` / ``from_dict`` through the
    constructor, ``to_xarray`` / ``to_netcdf`` with the fields as JSON.
    """

    ber: float
    evm: float
    scheme: str
    ebn0_dB: float
    tx_symbols: np.ndarray
    rx_symbols: np.ndarray
    mse: np.ndarray | None = None

    def __eq__(self, other):
        """Field-wise equality, with the symbol arrays compared element-wise.

        Hand-written because ``tx_symbols`` / ``rx_symbols`` / ``mse`` are
        ndarrays: the generated ``__eq__`` compares the field tuples, which
        puts an array inside a ``bool()`` and raises "The truth value of an
        array with more than one element is ambiguous".
        """
        if not isinstance(other, LinkResult):
            return NotImplemented
        if (self.ber, self.evm, self.scheme, self.ebn0_dB) != (
                other.ber, other.evm, other.scheme, other.ebn0_dB):
            return False
        if (self.mse is None) != (other.mse is None):
            return False
        if self.mse is not None and not np.array_equal(np.asarray(self.mse),
                                                       np.asarray(other.mse)):
            return False
        return (np.array_equal(np.asarray(self.tx_symbols),
                               np.asarray(other.tx_symbols))
                and np.array_equal(np.asarray(self.rx_symbols),
                                   np.asarray(other.rx_symbols)))


def _require_link_scheme(scheme, who):
    """Refuse a scheme the single-carrier link harness cannot run, naming
    the harness's schemes and the OFDM pair for a multicarrier link."""
    if not isinstance(scheme, str) or scheme.lower() not in SCHEMES:
        raise ConfigurationError(
            f"{who}: unknown scheme {scheme!r}; this harness runs the "
            f"single-carrier constellations {sorted(SCHEMES)}.",
            remediation="For OFDM, run OFDMTransmitter / OFDMReceiver (or "
                        "ofdm_modulate / ofdm_demodulate) over the channel "
                        "instead.")


def simulate_link(scheme, ebn0_dB, n_bits=20000, *, channel=None,
                  equalizer=None, code=None, n_train=400, rng=None):
    """Simulate one link and return a :class:`LinkResult`.

    Symbols are unit-average-energy, so the per-symbol SNR is ``k * Eb/N0``
    (``k`` bits/symbol). With ``channel=None`` and ``equalizer=None`` the BER
    matches the AWGN theory curve. ``channel`` is a static FIR ``h``
    convolved with the SYMBOLS: no pulse shaping is applied on the way out
    and no matched filter on the way in, so the taps must already be the
    channel as seen at the decision instants — symbol-spaced, with the
    raised cosine (transmit pulse times matched filter) folded in, which is
    what :meth:`Arrivals.channel_taps
    <uacpy.core.results.Arrivals.channel_taps>` returns at ``sps=1``
    (``pulse='rc'``, its default there) or, at whole-symbol delays,
    ``pulse='nearest'``. A root-raised-cosine half alone (``pulse='rrc'``)
    is the wrong channel here;
    ``equalizer`` a :class:`~uacpy.comms.equalize.DFE` (trained on the first
    ``n_train`` symbols); ``code`` a
    :class:`~uacpy.comms.coding.ConvCode` applied around the modem (BER then
    measured on the information bits).

    Parameters
    ----------
    scheme : str
        Modulation name accepted by
        :class:`~uacpy.comms.constellations.Modulator` — ``'bpsk'``,
        ``'qpsk'``, ``'16qam'`` and the rest of its table.
    ebn0_dB : float
        Information-bit energy to noise density ratio in dB. A ``code``
        lowers the channel Ec/N0 by its rate, so ``ebn0_dB`` stays the
        information-bit figure that BER curves are plotted against.
    n_bits : int, optional
        Number of information bits to transmit. Defaults to ``20000``.
    channel : array_like or ChannelTaps, optional
        Static FIR channel ``h`` convolved with the transmitted symbols.
        ``None`` is the AWGN-only link. A
        :class:`~uacpy.comms.channel.ChannelTaps` from
        :meth:`~uacpy.core.results.Arrivals.channel_taps`
        is accepted when it was built at ``sps=1``: this harness works on
        the symbol grid, so a tap vector at several samples per symbol
        names a different grid and is refused. Build it with the default
        pulse (``'rc'`` at ``sps=1``) or ``'nearest'``; see above. Its
        ``delays`` place tap 0 ``span/2`` symbols ahead of the arrival
        (the pulse's leading skirt), so the received symbols are read from
        the tap at zero delay on. A bare tap array carries no delay axis:
        its tap 0 is taken as the zero-delay tap, so a caller passing
        ``ChannelTaps.taps`` on its own receives the symbols ``span/2``
        positions late.
    equalizer : uacpy.comms.equalize.DFE, optional
        Equaliser trained on the first ``n_train`` symbols. ``None`` divides
        the received symbols by the channel's least-squares complex gain,
        fitted on those same first ``n_train`` symbols, and slices them: a
        QAM decision needs the constellation's own scale and phase, and a
        channel moves both (16-QAM through a 0.5 tap decodes at BER 0.25
        without it). On a multipath channel this restores the gain of the
        main path only, which is the no-equaliser baseline.
        With ``channel=None`` the symbols are sliced as received.
    code : uacpy.comms.coding.ConvCode, optional
        Code applied around the modem; BER is then measured on the
        information bits rather than the coded ones.
    n_train : int, optional
        Training symbols given to ``equalizer``. Defaults to ``400``.
    rng : numpy.random.Generator, optional
        Random generator for the information bits and the noise. Pass a
        seeded generator for a reproducible result.

    Returns
    -------
    LinkResult
        BER, EVM, the transmitted and received symbols, and the equaliser's
        learning curve when one was used.
    """
    _require_link_scheme(scheme, "simulate_link")
    rng = require_generator(rng, "simulate_link")
    if isinstance(channel, ChannelTaps):
        if int(channel.sps) != 1:
            raise ConfigurationError(
                f"simulate_link: channel= is a ChannelTaps at sps="
                f"{channel.sps}, but this harness convolves SYMBOLS, one "
                f"sample per symbol. Rebuild it with "
                f"Arrivals.channel_taps(symbol_rate, fc=..., sps=1).")
        # The taps start on the pulse's leading skirt, `lead` symbols before
        # the zero-delay tap: that is the offset symbol j reaches the output
        # at, read here because only the ChannelTaps carries delays.
        lead = int(round(-float(channel.delays[0]) * channel.symbol_rate))
        channel = channel.taps
    elif (isinstance(channel, tuple)
          and any(np.ndim(part) > 0 for part in channel)):
        raise ConfigurationError(
            f"simulate_link: channel= is a tuple of {len(channel)} arrays — "
            f"a (times, taps) pair, say — not one FIR channel.",
            remediation="Pass the taps alone, or a ChannelTaps "
                        "(comms.pulse_shaped_taps, or "
                        "Arrivals.channel_taps(symbol_rate, fc=..., sps=1)).")
    else:
        lead = 0
    mod = Modulator(scheme)
    k = mod.bits_per_symbol
    info = rng.integers(0, 2, int(n_bits))
    bits = _encoded(code, info)
    tx = mod.modulate(bits)

    delay = 0 if equalizer is None else equalizer.output_delay
    if channel is not None:
        rx = apply_channel(tx, channel)[lead:]
    else:
        rx = tx.copy()
    rx = rx[: tx.size + delay]
    if rx.size < tx.size + delay:
        rx = np.concatenate([rx, np.zeros(tx.size + delay - rx.size, dtype=complex)])

    # One sample per symbol, complex: the sampled band is the symbol-rate
    # band. A coded frame carries R information bits per transmitted bit, so
    # the code rate enters the conversion.
    rate = float(getattr(code, 'rate', 1.0) or 1.0) if code is not None else 1.0
    snr_dB = ebn0_to_snr_dB(ebn0_dB, bits_per_symbol=k, symbol_rate=1.0,
                            sample_rate=1.0, code_rate=rate)
    rx = awgn(rx, snr_dB, rng=rng)

    mse = None
    if equalizer is not None:
        ref = np.concatenate([np.zeros(delay, dtype=complex), tx])
        eq, mse = equalizer.equalize(rx, mod.constellation, train=ref[: n_train + delay])
        rx_sym = eq[delay: delay + tx.size]
    else:
        rx_sym = rx[: tx.size]
        if channel is not None:
            n_fit = min(int(n_train), tx.size)
            gain = complex_gain(tx[:n_fit], rx_sym[:n_fit])
            if gain and np.isfinite(gain):
                rx_sym = rx_sym / gain

    rx_bits = mod.demodulate(rx_sym)[: bits.size]
    if code is not None:
        rx_bits = code.decode(rx_bits)[: info.size]
        ber = bit_error_rate(reference=info, received=rx_bits)
    else:
        ber = bit_error_rate(reference=bits, received=rx_bits)
    return LinkResult(
        ber=ber,
        evm=evm(received=rx_sym, reference=tx),
        scheme=scheme,
        ebn0_dB=float(ebn0_dB),
        tx_symbols=tx,
        rx_symbols=rx_sym,
        mse=mse,
    )


class BerCurve(PlottedResult, namedtuple("BerCurve", "ebn0_dB ber")):
    """A measured bit-error-rate curve: ``ber`` at each ``ebn0_dB``
    (information-bit Eb/N0, dB), what :func:`ber_sweep` returns.

    The tuple is the measurement, so ``ebn0_dB, ber = ber_sweep(...)``
    unpacks it. What was measured rides on attributes, which survive
    pickling and copying and take no part in equality: ``scheme``,
    ``n_bits`` (information bits per point, so ``1/n_bits`` is the lowest
    rate a point resolves), and the link's ``channel``, ``equalizer``,
    ``code`` and ``n_train``, as :func:`simulate_link` took them.

    :meth:`plot` draws it with
    :func:`~uacpy.plot.plot_ber_curve`, which takes the curve whole and
    marks zero-error points at ``1/n_bits``. The closed-form AWGN curve of
    an uncoded link with no channel, the one link it describes, is
    :func:`~uacpy.comms.ber_theory_curve`, overlaid with
    ``.plot(ax=ax)``.
    """

    _attrs = ("scheme", "n_bits", "channel", "equalizer", "code", "n_train")
    _plotter = "plot_ber_curve"

    def __new__(cls, ebn0_dB, ber, *, scheme, n_bits, channel=None,
                equalizer=None, code=None, n_train=400):
        self = super().__new__(cls, ebn0_dB, ber)
        self.scheme = scheme
        self.n_bits = n_bits
        self.channel = channel
        self.equalizer = equalizer
        self.code = code
        self.n_train = n_train
        return self

    def _field_units(self):
        return {"ebn0_dB": "dB", "ber": ""}


def ber_sweep(scheme, ebn0_dB, n_bits=50000, *, channel=None,
              equalizer=None, code=None, n_train=400, rng=None):
    """Measured BER over a list of Eb/N0 values, as a :class:`BerCurve`.

    One :func:`simulate_link` per Eb/N0, all drawing from the same ``rng``,
    so the points of a sweep are independent realisations rather than the
    same bit stream re-noised.

    Parameters
    ----------
    scheme : str
        Modulation name, as :func:`simulate_link` takes it.
    ebn0_dB : array_like
        Information-bit Eb/N0 values in dB, one BER measured at each. A
        scalar is accepted and measures one point.
    n_bits : int, optional
        Information bits per point. Defaults to ``50000``, more than
        :func:`simulate_link`'s ``20000`` for one point, because a curve
        reaches lower BERs and the floor a sweep can measure is roughly
        ``1/n_bits``.
    channel : array_like or ChannelTaps, optional
        Static FIR channel ``h``, applied at every point, as
        :func:`simulate_link` takes it.
    equalizer : uacpy.comms.equalize.DFE, optional
        Equaliser settings used at every point. ``DFE.equalize`` re-adapts
        its taps from scratch on each call, so the points do not inherit
        each other's convergence.
    code : uacpy.comms.coding.ConvCode, optional
        Code applied around the modem at every point.
    n_train : int, optional
        Training symbols given to ``equalizer`` at every point, as
        :func:`simulate_link` takes it. Defaults to ``400``.
    rng : numpy.random.Generator, optional
        Random generator shared by every point of the sweep. Pass a seeded
        generator for a reproducible result.

    Returns
    -------
    BerCurve
        ``(ebn0_dB, ber)``: the Eb/N0 axis as floats and the measured BER,
        one per entry of ``ebn0_dB``, with the sweep's settings as
        attributes.
    """
    _require_link_scheme(scheme, "ber_sweep")
    rng = require_generator(rng, "ber_sweep")
    ber = np.array([
        simulate_link(scheme, e, n_bits, channel=channel, equalizer=equalizer,
                      code=code, n_train=n_train, rng=rng).ber
        for e in np.atleast_1d(ebn0_dB)
    ])
    return BerCurve(np.atleast_1d(np.asarray(ebn0_dB, dtype=float)),
                    ber, scheme=scheme, n_bits=n_bits, channel=channel,
                    equalizer=equalizer, code=code, n_train=n_train)


#: Information bits per OFDM frame :func:`ofdm_ber_sweep` sends (uacpy
#: guidance). The receiver corrects each block's common phase on its own
#: decisions, so a residual CFO error rotates the later blocks of a long frame
#: past a decision boundary: measured on QPSK at 256 subcarriers, cp 32,
#: Eb/N0 8 dB, BER 0.006-0.008 for frames of 512 and 2048 bits, 0.06 at 8192
#: and 0.20 at 32768.
_OFDM_FRAME_BITS = 2048


def _ofdm_frame_errors(tx, rx, info, ebn0_dB, channel, rng):
    """Bit errors of one OFDM frame carrying ``info``, sent by ``tx``
    through ``channel`` and AWGN at ``ebn0_dB`` and decoded by ``rx``.

    The frame is preceded by one block of silence, as a capture holds
    samples before the frame the receiver synchronises on. The noise is
    :func:`awgn` at the Eb/N0 conversion :func:`ebn0_to_snr_dB` makes, with
    each block of ``n_subcarriers + cp_len`` samples carrying
    ``n_subcarriers`` symbols — the cyclic prefix is energy spent, as
    simulate_link's symbols are. ``awgn`` scales its noise off the mean
    power of the whole record, which the silence, the preamble, the pilot
    and the zero guard block do not share with the payload, so the SNR
    passed to it is referred to the received power of the payload blocks
    (all but the last, whose unused subcarriers are zero-padded)."""
    nsc, cp = tx.n_subcarriers, tx.cp_len
    block = nsc + cp
    frame = np.concatenate([np.zeros(block, dtype=complex),
                            tx.transmit(info)])
    received = frame if channel is None else apply_channel(frame, channel)
    n_data = frame.size // block - 4        # silence, preamble, pilot, guard
    payload = received[3 * block: (3 + max(n_data - 1, 1)) * block]
    rate = float(getattr(tx.code, 'rate', 1.0) or 1.0) if tx.code else 1.0
    snr_dB = float(ebn0_to_snr_dB(
        ebn0_dB, bits_per_symbol=tx.modulator.bits_per_symbol,
        symbol_rate=nsc, sample_rate=block, code_rate=rate))
    p_payload = float(np.mean(np.abs(payload) ** 2))
    p_record = float(np.mean(np.abs(received) ** 2))
    if p_payload > 0.0 and np.isfinite(snr_dB):
        snr_dB += 10.0 * np.log10(p_record / p_payload)
    bits = rx.receive(awgn(received, snr_dB, rng=rng))
    return bit_error_rate(reference=info,
                          received=bits[: info.size]) * info.size


def ofdm_ber_sweep(tx, rx, ebn0_dB, n_bits=50000, *, channel=None,
                   frame_bits=_OFDM_FRAME_BITS, rng=None):
    """Measured BER of an OFDM link over a list of Eb/N0 values, as a
    :class:`BerCurve`: :func:`ber_sweep` for the multicarrier pair.

    At each Eb/N0, ``n_bits`` random information bits go in frames of
    ``frame_bits`` through ``tx`` (:meth:`OFDMTransmitter.transmit`,
    baseband), the static FIR ``channel``, AWGN, and ``rx``
    (:meth:`OFDMReceiver.receive`: sync, CFO, pilot channel estimate,
    per-subcarrier equalisation, decoding), and the decoded bits are
    compared with the sent ones. Eb/N0 is per
    information bit, with the cyclic prefix counted as energy spent and the
    noise referred to the received payload power. All points draw from the
    same ``rng``, so they are independent realisations.

    Parameters
    ----------
    tx : OFDMTransmitter
        The transmitter (constellation, subcarriers, cyclic prefix, code).
    rx : OFDMReceiver
        The matching receiver; its constellation, ``n_subcarriers``,
        ``cp_len`` and code presence must be the transmitter's.
    ebn0_dB : array_like
        Information-bit Eb/N0 values in dB, one BER measured at each.
    n_bits : int, optional
        Information bits per point. Defaults to ``50000``, as
        :func:`ber_sweep`.
    channel : array_like or ChannelTaps, optional
        Static FIR channel at the baseband sample rate, applied to the
        whole frame (:func:`apply_channel`). ``None`` is AWGN alone.
    frame_bits : int, optional
        Information bits per frame. Default 2048 (uacpy guidance): the
        receiver corrects each block's common phase on its own decisions,
        so a residual CFO error loses the later blocks of a long frame
        (measured on QPSK, 256 subcarriers, 8 dB: BER 0.006 at 2048 bits,
        0.06 at 8192, 0.20 at 32768).
    rng : numpy.random.Generator, optional
        Random generator shared by every point. Pass a seeded generator for
        a reproducible result.

    Returns
    -------
    BerCurve
        ``(ebn0_dB, ber)``, with ``scheme='OFDM <constellation>'``,
        ``n_bits``, ``channel`` and ``code`` (the transmitter's) as
        attributes; ``.plot()`` draws it.
    """
    who = "ofdm_ber_sweep"
    mismatch = [name for name, a, b in (
        ("modulation", tx.modulation, rx.modulation),
        ("n_subcarriers", tx.n_subcarriers, rx.n_subcarriers),
        ("cp_len", tx.cp_len, rx.cp_len),
        ("code", tx.code is None, rx.code is None)) if a != b]
    if mismatch:
        raise ConfigurationError(
            f"{who}: the transmitter and the receiver differ in "
            f"{', '.join(mismatch)}, so the receiver cannot decode the "
            f"frames.",
            remediation="Build both from the same constellation, "
                        "n_subcarriers, cp_len and code.")
    n_bits = int(n_bits)
    frame_bits = int(frame_bits)
    if n_bits < 1 or frame_bits < 1:
        raise ConfigurationError(
            f"{who}: n_bits={n_bits} and frame_bits={frame_bits} must both "
            f"be at least 1.")
    rng = require_generator(rng, who)
    sizes = [frame_bits] * (n_bits // frame_bits)
    if n_bits % frame_bits:
        sizes.append(n_bits % frame_bits)

    def point(ebn0):
        try:
            return sum(_ofdm_frame_errors(tx, rx, rng.integers(0, 2, size),
                                          ebn0, channel, rng)
                       for size in sizes) / n_bits
        except ConfigurationError as exc:
            # The receiver refuses a frame its sync placed too late to hold
            # the pilot and a data block; say at which point of the sweep.
            raise ConfigurationError(
                f"{who}: at Eb/N0 = {float(ebn0):g} dB the receiver could "
                f"not decode a frame ({str(exc).splitlines()[0]})",
                remediation="Start the sweep at a higher Eb/N0, where the "
                            "receiver's Schmidl-Cox sync holds (see "
                            "schmidl_cox_sync), or use more subcarriers."
            ) from exc

    ber = np.array([point(e) for e in np.atleast_1d(ebn0_dB)])
    return BerCurve(np.atleast_1d(np.asarray(ebn0_dB, dtype=float)), ber,
                    scheme=f"OFDM {tx.modulation}", n_bits=n_bits,
                    channel=channel, code=tx.code, n_train=None)
