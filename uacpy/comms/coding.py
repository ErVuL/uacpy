"""Forward error correction: convolutional coding, Viterbi decoding, interleaving.

A rate-1/2 convolutional code with hard-decision Viterbi decoding plus a block
interleaver to break up the burst errors that fading produces — the classic FEC
layer of an underwater modem.

References
----------
Proakis & Salehi. *Digital Communications* (convolutional codes, Viterbi
    algorithm, interleaving).
"""

from __future__ import annotations

import numpy as np

# Rate-1/2 maximum-free-distance generator pair for K=7 (Proakis & Salehi
# Table 8.3-1, after Odenwalder 1970 / Larsen 1973): d_free = 10, which meets
# that table's upper bound, so no other K=7 rate-1/2 code does better.
DEFAULT_POLYS = (0o171, 0o133)
DEFAULT_K = 7


def _poly_bits(poly, K):
    return [(poly >> (K - 1 - i)) & 1 for i in range(K)]


def conv_encode(bits, polys=DEFAULT_POLYS, K=DEFAULT_K):
    """Rate ``1/len(polys)`` convolutional encoding with zero tail-flush.

    Returns the coded bit stream (``len(polys)`` output bits per input bit,
    including the ``K-1`` flush bits). ``polys`` are octal generator taps.
    """
    b = list(np.asarray(bits, dtype=int).ravel()) + [0] * (K - 1)
    taps = [_poly_bits(p, K) for p in polys]
    reg = [0] * K
    out = []
    for bit in b:
        reg = [bit] + reg[:-1]
        for t in taps:
            out.append(sum(r & g for r, g in zip(reg, t)) & 1)
    return np.array(out, dtype=int)


def viterbi_decode(coded, polys=DEFAULT_POLYS, K=DEFAULT_K):
    """Hard-decision Viterbi decoding (inverse of :func:`conv_encode`).

    Returns the decoded information bits (tail removed).
    """
    n = len(polys)
    c = np.asarray(coded, dtype=int).ravel()
    c = c[: (c.size // n) * n]            # drop a trailing partial symbol (garbage)
    taps = [_poly_bits(p, K) for p in polys]
    n_states = 1 << (K - 1)

    def outputs(state, bit):
        """Encoder output word for a state transition on ``bit``."""
        reg = [bit] + [(state >> (K - 2 - i)) & 1 for i in range(K - 1)]
        return [sum(r & g for r, g in zip(reg, t)) & 1 for t in taps]

    # Butterfly structure: the transition (st, bit) -> ((bit << (K-2)) |
    # (st >> 1)) means state s is reached from exactly two predecessors,
    # 2s and 2s+1 (mod n_states), both on input bit = the top bit of s.
    states = np.arange(n_states)
    prev0 = (states << 1) & (n_states - 1)
    prev1 = prev0 | 1
    bit_in = states >> (K - 2)
    # branch[st, b] = encoder output word (n bits) leaving state st on bit b
    branch = np.array([[outputs(st, b) for b in (0, 1)]
                       for st in range(n_states)], dtype=int)

    nsteps = c.size // n
    rx = c.reshape(nsteps, n)
    # Hamming branch metrics per step for the two transitions into each state:
    # bm0[k, s] = distance of rx[k] from the word on prev0[s] -> s.
    bm_all = np.count_nonzero(branch[None, :, :, :] ^ rx[:, None, None, :],
                              axis=3)
    bm0 = bm_all[:, prev0, bit_in]
    bm1 = bm_all[:, prev1, bit_in]

    pm = np.full(n_states, np.inf)
    pm[0] = 0.0
    back = np.zeros((nsteps, n_states), dtype=np.int32)   # holds prev-state index
    for k in range(nsteps):
        cand0 = pm[prev0] + bm0[k]
        cand1 = pm[prev1] + bm1[k]
        # Strict < keeps the even predecessor on a tie — the same survivor a
        # state-ascending scan that replaces only on improvement selects.
        take1 = cand1 < cand0
        pm = np.where(take1, cand1, cand0)
        back[k] = np.where(take1, prev1, prev0)
    # The K-1 zero flush bits force the encoder to end in state 0, so the
    # traceback starts there; argmin(pm) could pick a different state on
    # noisy input and lose the tail constraint.
    state = 0
    bits = np.zeros(nsteps, dtype=int)
    for k in range(nsteps - 1, -1, -1):
        prev = back[k, state]
        bits[k] = (state >> (K - 2)) & 1
        state = prev
    return bits[: nsteps - (K - 1)]


class ConvCode:
    """Configurable convolutional codec bundling encode/decode + interleaving.

    Holds the generator polynomials, constraint length and (optional) block
    interleaver depth in one place, so :meth:`encode` and :meth:`decode` always
    use matching settings.

    Parameters
    ----------
    polys : tuple of int
        Octal generator polynomials (rate ``1/len(polys)``).
    K : int
        Constraint length.
    interleave_depth : int, optional
        Block interleaver depth; ``None`` disables interleaving.

    Examples
    --------
    >>> code = ConvCode(interleave_depth=16)
    >>> rx = code.decode(code.encode(bits))
    """

    def __init__(self, polys=DEFAULT_POLYS, K: int = DEFAULT_K,
                 interleave_depth=None):
        self.polys = tuple(polys)
        self.K = int(K)
        self.interleave_depth = interleave_depth
        self._info_len = None   # last encoded payload length, for exact decode

    @property
    def rate(self):
        """Code rate ``1 / len(polys)``."""
        return 1.0 / len(self.polys)

    def encode(self, bits):
        """Encode information bits (convolutional + optional interleave)."""
        b = np.asarray(bits, dtype=int).ravel()
        self._info_len = b.size
        coded = conv_encode(b, self.polys, self.K)
        if self.interleave_depth:
            coded = interleave(coded, self.interleave_depth)
        return coded

    def decode(self, coded, info_len=None):
        """Decode (optional deinterleave + Viterbi) back to information bits.

        Parameters
        ----------
        info_len : int, optional
            Number of information bits to return — strips the
            interleaver's block padding (and any whole-block zeros an
            outer framing layer appended) so the payload comes back
            exactly. ``None`` falls back to the length recorded by this
            codec's most recent :meth:`encode` call (loopback use); a
            codec that decodes messages it did not just encode must pass
            ``info_len`` explicitly, otherwise the full Viterbi output
            is returned (or, worse, the *previous* message's length is
            applied). Across separate transmit/receive codecs no length
            is recorded and the caller slices.
        """
        if self.interleave_depth:
            coded = deinterleave(coded, self.interleave_depth)
        bits = viterbi_decode(coded, self.polys, self.K)
        n_keep = info_len if info_len is not None else self._info_len
        if n_keep is not None:
            bits = bits[: int(n_keep)]
        return bits


def interleave(bits, depth):
    """Block-local ``depth x depth`` interleaver (transpose per square block).

    Operating in independent ``depth*depth`` blocks (zero-padded to a whole
    number of blocks) keeps the block boundaries aligned regardless of payload
    length, so an outer framing layer that pads to whole symbols (e.g. OFDM) only
    ever appends complete zero blocks. The interleaver therefore grows the stream
    to a multiple of ``depth*depth``; :class:`ConvCode` records the information
    length so :meth:`ConvCode.decode` can strip the resulting tail exactly.
    """
    b = np.asarray(bits, dtype=int).ravel()
    d = int(depth)
    block = d * d
    nblk = int(np.ceil(b.size / block))
    pad = np.zeros(nblk * block, dtype=int)
    pad[: b.size] = b
    return pad.reshape(nblk, d, d).transpose(0, 2, 1).reshape(-1)


def deinterleave(bits, depth):
    """Inverse of :func:`interleave`; drops a trailing partial (garbage) block."""
    b = np.asarray(bits, dtype=int).ravel()
    d = int(depth)
    block = d * d
    nblk = b.size // block
    b = b[: nblk * block]
    return b.reshape(nblk, d, d).transpose(0, 2, 1).reshape(-1)
