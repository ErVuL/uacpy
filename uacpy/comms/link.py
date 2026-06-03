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


@dataclass
class LinkResult:
    """Outcome of one :func:`simulate_link` run."""

    ber: float
    evm: float
    scheme: str
    ebn0_db: float
    tx_symbols: np.ndarray
    rx_symbols: np.ndarray
    mse: np.ndarray | None = None


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
    """
    rng = np.random.default_rng() if rng is None else rng
    mod = Modulator(scheme)
    k = mod.bits_per_symbol
    info = rng.integers(0, 2, int(n_bits))
    bits = code.encode(info) if code is not None else info
    tx = mod.modulate(bits)
    bits = bits[: tx.size * k]                # trim padding bit, if any

    delay = equalizer.n_ff // 2 if isinstance(equalizer, DFE) else 0
    if channel is not None:
        rx = apply_channel(tx, channel)
    else:
        rx = tx.copy()
    rx = rx[: tx.size + delay]
    if rx.size < tx.size + delay:
        rx = np.concatenate([rx, np.zeros(tx.size + delay - rx.size, dtype=complex)])

    ebn0 = 10.0 ** (float(ebn0_db) / 10.0)
    rx = awgn(rx, 10.0 * np.log10(k * ebn0), rng=rng)

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
    """Measured BER over a list of Eb/N0 values. Returns a NumPy array."""
    rng = np.random.default_rng() if rng is None else rng
    return np.array([
        simulate_link(scheme, e, n_bits, channel=channel, equalizer=equalizer,
                      code=code, rng=rng).ber
        for e in np.atleast_1d(ebn0_db_list)
    ])
