"""What protects and packs the bits: direct-sequence spreading, the
convolutional code and its Viterbi decoders, block interleaving, and the
framing that packs a byte payload with a header and a CRC."""

from __future__ import annotations

import operator
import zlib

import numpy as np

from uacpy.comms.constellations import _require_binary_bits
from uacpy.core.exceptions import ConfigurationError
from uacpy.core._repr import SettingsRepr


def spread(symbols, code):
    """Spread each symbol by the chip ``code`` (Kronecker product). Returns chips.

    Parameters
    ----------
    symbols : array_like
        Symbols to spread.
    code : array_like
        The chip sequence, at least one chip.
    """
    s = np.asarray(symbols, dtype=complex).ravel()
    c = np.asarray(code, dtype=complex).ravel()
    if c.size < 1:
        raise ConfigurationError(
            "spread: empty code (0 chips); pass the spreading sequence the "
            "receiver will despread with, e.g. m_sequence(n_stages, taps).")
    return np.kron(s, c)


def despread(chips, code):
    """Correlate chips against ``code`` per symbol period -> symbol estimates.

    The correlation is normalised by the code energy ``sum(|c|^2)``, so
    ``despread(spread(s, c), c)`` returns ``s`` itself for any code amplitude
    (dividing by the chip count is equivalent only for unit-modulus codes).
    A trailing partial symbol (fewer than ``len(code)`` chips) is dropped.

    Parameters
    ----------
    chips : array_like
        The received chip stream.
    code : array_like
        The chip sequence :func:`spread` used.
    """
    c = np.asarray(code, dtype=complex).ravel()
    x = np.asarray(chips, dtype=complex).ravel()
    if c.size < 1:
        raise ConfigurationError(
            "despread: empty code (0 chips); pass the same spreading "
            "sequence spread() was called with.")
    energy = float(np.sum(np.abs(c) ** 2))
    if energy == 0.0:
        raise ConfigurationError(
            "despread: code energy sum(|c|^2) is zero (every chip is 0), so "
            "the per-symbol correlation carries no signal and the "
            "normalisation is undefined.")
    n = c.size
    nsym = x.size // n
    blocks = x[: nsym * n].reshape(nsym, n)
    return (blocks @ np.conj(c)) / energy


def spreading_gain_dB(code):
    """Spreading gain ``10*log10(N)`` in dB for an ``N``-chip code: the SNR
    that :func:`despread` recovers over the chip SNR.

    The time-bandwidth form ``10*log10(B*T)`` of the same gain, for a
    waveform described by its band and duration rather than by a code, is
    :func:`uacpy.acoustic_signal.processing_gain_dB`.

    Parameters
    ----------
    code : array_like
        The chip sequence; only its length counts.
    """
    return float(10.0 * np.log10(np.asarray(code).size))


# Rate-1/2 maximum-free-distance generator pair for K=7 (Proakis & Salehi
# Table 8.3-1, after Odenwalder 1970 / Larsen 1973): d_free = 10, which meets
# that table's upper bound, so no other K=7 rate-1/2 code does better.
DEFAULT_POLYS = (0o171, 0o133)


DEFAULT_CONSTRAINT_LENGTH = 7


def _poly_bits(poly, constraint_length):
    return [(poly >> (constraint_length - 1 - i)) & 1 for i in range(constraint_length)]


def _encoded(code, bits):
    """``bits`` through ``code``'s encoder, or as an integer array when there
    is no code: the transmitted bit stream of a link with optional FEC."""
    return code.encode(bits) if code is not None else np.asarray(bits, int)


def conv_encode(bits, polys=DEFAULT_POLYS, constraint_length=DEFAULT_CONSTRAINT_LENGTH):
    """Rate ``1/len(polys)`` convolutional encoding with zero tail-flush.

    Returns the coded bit stream (``len(polys)`` output bits per input bit,
    including the ``K-1`` flush bits). ``polys`` are octal generator taps.

    Parameters
    ----------
    bits : array_like of int
        Information bits, 0/1.
    polys : tuple of int, optional
        Generator polynomials, octal taps. Default
        :data:`DEFAULT_POLYS`, the K=7 maximum-free-distance pair.
    constraint_length : int, optional
        Constraint length K. Default 7.
    """
    b = list(_require_binary_bits("conv_encode", bits)) + [0] * (constraint_length - 1)
    taps = [_poly_bits(p, constraint_length) for p in polys]
    reg = [0] * constraint_length
    out = []
    for bit in b:
        reg = [bit] + reg[:-1]
        for t in taps:
            out.append(sum(r & g for r, g in zip(reg, t)) & 1)
    return np.array(out, dtype=int)


def viterbi_decode(coded, polys=DEFAULT_POLYS, constraint_length=DEFAULT_CONSTRAINT_LENGTH):
    """Hard-decision Viterbi decoding (inverse of :func:`conv_encode`).

    Returns the decoded information bits (tail removed).

    Parameters
    ----------
    coded : array_like of int
        Hard-decision coded bits (0/1); a trailing partial output word is
        dropped.
    polys : tuple of int, optional
        The generator polynomials :func:`conv_encode` used.
    constraint_length : int, optional
        The constraint length :func:`conv_encode` used. Default 7.
    """
    # The package's only Viterbi decoder: uacpy.comms.janus.janus_decode calls
    # it with the reversed CMRE generators and constraint_length=9, so JANUS and ConvCode share
    # one add-compare-select and traceback.
    n = len(polys)
    c = np.asarray(coded, dtype=int).ravel()
    c = c[: (c.size // n) * n]            # drop a trailing partial symbol (garbage)
    taps = [_poly_bits(p, constraint_length) for p in polys]
    n_states = 1 << (constraint_length - 1)

    def outputs(state, bit):
        """Encoder output word for a state transition on ``bit``."""
        reg = [bit] + [(state >> (constraint_length - 2 - i)) & 1 for i in range(constraint_length - 1)]
        return [sum(r & g for r, g in zip(reg, t)) & 1 for t in taps]

    # Butterfly structure: the transition (st, bit) -> ((bit << (K-2)) |
    # (st >> 1)) means state s is reached from exactly two predecessors,
    # 2s and 2s+1 (mod n_states), both on input bit = the top bit of s.
    states = np.arange(n_states)
    prev0 = (states << 1) & (n_states - 1)
    prev1 = prev0 | 1
    bit_in = states >> (constraint_length - 2)
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

    bits = viterbi_hard(bm0, bm1, prev0, prev1, n_states,
                        lambda state: (state >> (constraint_length - 2)) & 1)
    return bits[: nsteps - (constraint_length - 1)]


def viterbi_hard(bm0, bm1, prev0, prev1, n_states, bit_of_state):
    """Hard-decision Viterbi over a rate-1/2 butterfly trellis.

    ``bm0[k]``/``bm1[k]`` are the branch metrics into every state at step
    ``k`` from its two predecessors ``prev0``/``prev1``; ``bit_of_state``
    reads the input bit off the arriving state. Starts and ends in state 0
    (the tail flush), so the traceback begins there rather than at the
    survivor with the smallest metric, which noisy input could move.
    Factored out of :func:`viterbi_decode` so a caller that already holds its
    branch metrics can reuse the survivor selection without rebuilding them.

    Parameters
    ----------
    bm0, bm1 : ndarray
        ``(n_steps, n_states)`` branch metrics into each state from its
        predecessors ``prev0`` / ``prev1``.
    prev0, prev1 : ndarray of int
        The two predecessor states of each state.
    n_states : int
        Trellis states, ``2**(K-1)``.
    bit_of_state : callable
        The input bit that leads into a state.
    """
    nsteps = len(bm0)
    pm = np.full(n_states, np.inf)
    pm[0] = 0.0
    back = np.zeros((nsteps, n_states), dtype=np.int32)   # prev-state index
    for k in range(nsteps):
        cand0 = pm[prev0] + bm0[k]
        cand1 = pm[prev1] + bm1[k]
        # Strict < keeps the even predecessor on a tie — the same survivor a
        # state-ascending scan that replaces only on improvement selects.
        take1 = cand1 < cand0
        pm = np.where(take1, cand1, cand0)
        back[k] = np.where(take1, prev1, prev0)
    state = 0
    bits = np.zeros(nsteps, dtype=int)
    for k in range(nsteps - 1, -1, -1):
        bits[k] = bit_of_state(state)
        state = back[k, state]
    return bits


class ConvCode(SettingsRepr):
    """Configurable convolutional codec bundling encode/decode + interleaving.

    Holds the generator polynomials, constraint length and (optional) block
    interleaver depth in one place, so :meth:`encode` and :meth:`decode` always
    use matching settings. The codec holds no per-message state: the
    information length is a property of the message, so :meth:`decode` takes
    it as ``info_len`` (a framed payload carries its own length instead, read
    by :func:`unpack_frame`).

    Parameters
    ----------
    polys : tuple of int
        Octal generator polynomials (rate ``1/len(polys)``).
    constraint_length : int
        Constraint length.
    interleave_depth : int, optional
        Block interleaver depth; ``None`` disables interleaving.

    Examples
    --------
    >>> import numpy as np
    >>> bits = np.random.default_rng(0).integers(0, 2, 64)
    >>> code = ConvCode(interleave_depth=16)
    >>> rx = code.decode(code.encode(bits), info_len=bits.size)
    >>> bool(np.array_equal(rx, bits))
    True
    """

    def __init__(self, polys=DEFAULT_POLYS, constraint_length: int = DEFAULT_CONSTRAINT_LENGTH,
                 interleave_depth=None):
        self.polys = tuple(polys)
        self.constraint_length = int(constraint_length)
        self.interleave_depth = interleave_depth

    @property
    def rate(self):
        """Code rate ``1 / len(polys)``."""
        return 1.0 / len(self.polys)

    def encode(self, bits):
        """Encode information bits (convolutional + optional interleave)."""
        b = _require_binary_bits("ConvCode.encode", bits)
        coded = conv_encode(b, self.polys, self.constraint_length)
        if self.interleave_depth:
            coded = interleave(coded, self.interleave_depth)
        return coded

    def decode(self, coded, info_len=None):
        """Decode (optional deinterleave + Viterbi) back to information bits.

        Parameters
        ----------
        coded : array_like of int
            Hard-decision code bits (0/1) as :meth:`encode` returns them,
            interleaved when ``interleave_depth`` is set. A trailing partial
            symbol is dropped.
        info_len : int, optional
            Number of information bits to return — strips the
            interleaver's block padding (and any whole-block zeros an
            outer framing layer appended) so the payload comes back
            exactly. ``None`` returns the full Viterbi output, the payload
            followed by the decoded padding, which is what a receiver
            holding a framed payload (:func:`unpack_frame`) needs.
        """
        if self.interleave_depth:
            coded = deinterleave(coded, self.interleave_depth)
        bits = viterbi_decode(coded, self.polys, self.constraint_length)
        if info_len is None:
            return bits
        try:
            n_keep = operator.index(info_len)
        except TypeError:
            raise ConfigurationError(
                f"ConvCode.decode: info_len must be an integer; got "
                f"{info_len!r}.") from None
        if not 0 <= n_keep <= bits.size:
            raise ConfigurationError(
                f"ConvCode.decode: info_len={n_keep} must lie in "
                f"0..{bits.size}, the bits the coded stream decodes to.")
        return bits[:n_keep]


def _require_depth(who: str, depth) -> int:
    """Reject an interleaver depth below 1.

    The block size is ``depth*depth``, so depth 0 divides by zero and a
    negative depth reaches the reshape as an unknown dimension — both bare
    numpy/Python errors with nothing naming the argument.
    """
    d = int(depth)
    if d < 1:
        raise ConfigurationError(
            f"{who}: depth must be >= 1 (it is the side of the "
            f"depth x depth interleaver block); got {depth!r}.")
    return d


def interleave(bits, depth):
    """Block-local ``depth x depth`` interleaver (transpose per square block).

    Operating in independent ``depth*depth`` blocks (zero-padded to a whole
    number of blocks) keeps the block boundaries aligned regardless of payload
    length, so an outer framing layer that pads to whole symbols (e.g. OFDM) only
    ever appends complete zero blocks. The interleaver therefore grows the stream
    to a multiple of ``depth*depth``; :meth:`ConvCode.decode` strips the
    resulting tail exactly when given the information length (``info_len``).

    Parameters
    ----------
    bits : array_like of int
        Bits to interleave, 0/1.
    depth : int
        Side of the square block, >= 1.
    """
    b = _require_binary_bits("interleave", bits)
    d = _require_depth("interleave", depth)
    block = d * d
    nblk = int(np.ceil(b.size / block))
    pad = np.zeros(nblk * block, dtype=int)
    pad[: b.size] = b
    return pad.reshape(nblk, d, d).transpose(0, 2, 1).reshape(-1)


def deinterleave(bits, depth):
    """Inverse of :func:`interleave`; drops a trailing partial (garbage) block.

    Parameters
    ----------
    bits : array_like of int
        Interleaved bits.
    depth : int
        The block side :func:`interleave` used.
    """
    b = np.asarray(bits, dtype=int).ravel()
    d = _require_depth("deinterleave", depth)
    block = d * d
    nblk = b.size // block
    b = b[: nblk * block]
    return b.reshape(nblk, d, d).transpose(0, 2, 1).reshape(-1)


_HEADER_BYTES = 4   # uint32 payload length


_CRC_BYTES = 4      # uint32 CRC-32


def bytes_to_bits(data):
    """Unpack bytes to an MSB-first 0/1 bit array.

    Parameters
    ----------
    data : bytes-like
        The bytes to unpack (a ``str`` is refused: encode it first).
    """
    if isinstance(data, str):
        raise ConfigurationError(
            "bytes_to_bits: data is a str; the framing layer carries bytes "
            "— encode it first, e.g. data.encode('utf-8').")
    return np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8))


def bits_to_bytes(bits):
    """Pack an MSB-first 0/1 bit array back to bytes (truncates a partial byte).

    Parameters
    ----------
    bits : array_like of int
        MSB-first bits, 0/1.
    """
    b = np.asarray(bits, dtype=np.uint8).ravel()
    n = (b.size // 8) * 8
    return np.packbits(b[:n]).tobytes()


def pack_frame(payload):
    """Frame a byte payload as ``[len:4][payload][crc32:4]`` -> bit array.

    Parameters
    ----------
    payload : bytes-like
        The payload to frame.
    """
    payload = bytes(payload)
    header = len(payload).to_bytes(_HEADER_BYTES, "big")
    # The CRC covers the header as well as the payload: a corrupted length
    # field otherwise steers the slice that the CRC is then read from, so the
    # check can never see the error that caused it.
    crc = zlib.crc32(header + payload).to_bytes(_CRC_BYTES, "big")
    return bytes_to_bits(header + payload + crc)


def unpack_frame(bits):
    """Recover ``(payload_bytes, crc_ok)`` from a framed bit stream.

    Reads the length header, slices the payload, and checks the trailing
    CRC-32 over ``header + payload``. A stream that is too short, or whose
    length header does not fit the data supplied, is a **failed frame**, not a
    caller error: it returns ``(b"", False)``. Only ``crc_ok=True`` means the
    frame arrived intact — a receiver cannot distinguish "corrupt" from
    "malformed" and should not have to.

    Parameters
    ----------
    bits : array_like of int
        The framed bit stream, as :func:`pack_frame` produced it.
    """
    data = bits_to_bytes(bits)
    if len(data) < _HEADER_BYTES + _CRC_BYTES:
        return b"", False
    length = int.from_bytes(data[:_HEADER_BYTES], "big")
    end = _HEADER_BYTES + length
    if len(data) < end + _CRC_BYTES:
        return b"", False
    payload = data[_HEADER_BYTES:end]
    crc_rx = int.from_bytes(data[end:end + _CRC_BYTES], "big")
    return payload, zlib.crc32(data[:end]) == crc_rx
