"""End-to-end link simulation: bits -> modulate -> channel -> noise -> receiver.

Ties the comms package together into a single BER measurement, and sweeps it
versus Eb/N0 so results can be checked against :func:`uacpy.comms.ber_theory`.

References
----------
Proakis & Salehi. *Digital Communications* (link budget, BER vs Eb/N0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uacpy.comms.channel_models import apply_channel, awgn
from uacpy.comms.equalization import DFE
from uacpy.comms.metrics import bit_error_rate, evm
from uacpy.comms.modulation import Modulator


@dataclass(eq=False)
class LinkResult:
    """Outcome of one :func:`simulate_link` run."""

    ber: float
    evm: float
    scheme: str
    ebn0_db: float
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
        if (self.ber, self.evm, self.scheme, self.ebn0_db) != (
                other.ber, other.evm, other.scheme, other.ebn0_db):
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



def simulate_link(scheme, ebn0_db, n_bits=20000, *, channel=None,
                  equalizer=None, code=None, n_train=400, rng=None):
    """Simulate one link and return a :class:`LinkResult`.

    Symbols are unit-average-energy, so the per-symbol SNR is ``k * Eb/N0``
    (``k`` bits/symbol). With ``channel=None`` and ``equalizer=None`` the BER
    matches the AWGN theory curve. ``channel`` is a static FIR ``h``;
    ``equalizer`` a :class:`~uacpy.comms.equalization.DFE` (trained on the first
    ``n_train`` symbols); ``code`` a
    :class:`~uacpy.comms.coding.ConvCode` applied around the modem (BER then
    measured on the information bits).

    Parameters
    ----------
    scheme : str
        Modulation name accepted by
        :class:`~uacpy.comms.modulation.Modulator` — ``'bpsk'``, ``'qpsk'``,
        ``'16qam'`` and the rest of its table.
    ebn0_db : float
        Information-bit energy to noise density ratio in dB. A ``code``
        lowers the channel Ec/N0 by its rate, so ``ebn0_db`` stays the
        information-bit figure that BER curves are plotted against.
    n_bits : int, optional
        Number of information bits to transmit. Defaults to ``20000``.
    channel : array_like, optional
        Static FIR channel ``h`` convolved with the transmitted symbols.
        ``None`` is the AWGN-only link.
    equalizer : uacpy.comms.equalization.DFE, optional
        Equaliser trained on the first ``n_train`` symbols. ``None`` slices
        the received symbols straight out.
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
    rng = np.random.default_rng() if rng is None else rng
    mod = Modulator(scheme)
    k = mod.bits_per_symbol
    info = rng.integers(0, 2, int(n_bits))
    bits = code.encode(info) if code is not None else info
    tx = mod.modulate(bits)

    delay = equalizer.n_ff // 2 if isinstance(equalizer, DFE) else 0
    if channel is not None:
        rx = apply_channel(tx, channel)
    else:
        rx = tx.copy()
    rx = rx[: tx.size + delay]
    if rx.size < tx.size + delay:
        rx = np.concatenate([rx, np.zeros(tx.size + delay - rx.size, dtype=complex)])

    ebn0 = 10.0 ** (float(ebn0_db) / 10.0)
    # Es/N0 = (info Eb/N0) x bits-per-symbol x code rate: a coded frame
    # carries R information bits per transmitted bit, so omitting R labels
    # Ec/N0 as Eb/N0 and overstates coding gain by 10log10(1/R).
    rate = float(getattr(code, 'rate', 1.0) or 1.0) if code is not None else 1.0
    rx = awgn(rx, 10.0 * np.log10(k * rate * ebn0), rng=rng)

    mse = None
    if equalizer is not None:
        ref = np.concatenate([np.zeros(delay, dtype=complex), tx])
        eq, mse = equalizer.equalize(rx, mod.constellation, train=ref[: n_train + delay])
        rx_sym = eq[delay: delay + tx.size]
    else:
        rx_sym = rx[: tx.size]

    rx_bits = mod.demodulate(rx_sym)[: bits.size]
    if code is not None:
        rx_bits = code.decode(rx_bits)[: info.size]
        ber = bit_error_rate(info, rx_bits)
    else:
        ber = bit_error_rate(bits, rx_bits)
    return LinkResult(
        ber=ber,
        evm=evm(rx_sym, tx),
        scheme=scheme,
        ebn0_db=float(ebn0_db),
        tx_symbols=tx,
        rx_symbols=rx_sym,
        mse=mse,
    )


def ber_sweep(scheme, ebn0_db_list, n_bits=50000, *, channel=None,
              equalizer=None, code=None, rng=None):
    """Measured BER over a list of Eb/N0 values. Returns a NumPy array.

    One :func:`simulate_link` per Eb/N0, all drawing from the same ``rng``,
    so the points of a sweep are independent realisations rather than the
    same bit stream re-noised.

    Parameters
    ----------
    scheme : str
        Modulation name, as :func:`simulate_link` takes it.
    ebn0_db_list : array_like
        Information-bit Eb/N0 values in dB, one BER measured at each. A
        scalar is accepted and returns a length-1 array.
    n_bits : int, optional
        Information bits per point. Defaults to ``50000``; the BER floor a
        sweep can measure is roughly ``1/n_bits``.
    channel : array_like, optional
        Static FIR channel ``h``, applied at every point.
    equalizer : uacpy.comms.equalization.DFE, optional
        Equaliser settings used at every point. ``DFE.equalize`` re-adapts
        its taps from scratch on each call, so the points do not inherit
        each other's convergence.
    code : uacpy.comms.coding.ConvCode, optional
        Code applied around the modem at every point.
    rng : numpy.random.Generator, optional
        Random generator shared by every point of the sweep. Pass a seeded
        generator for a reproducible result.

    Returns
    -------
    ndarray
        Measured BER, one per entry of ``ebn0_db_list``.
    """
    rng = np.random.default_rng() if rng is None else rng
    return np.array([
        simulate_link(scheme, e, n_bits, channel=channel, equalizer=equalizer,
                      code=code, rng=rng).ber
        for e in np.atleast_1d(ebn0_db_list)
    ])
