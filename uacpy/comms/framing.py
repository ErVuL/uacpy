"""Byte/bit framing for real payloads: length header + CRC.

Turns an arbitrary byte payload (a message, a file) into a bit stream with a
32-bit length header and a trailing CRC-32, so the receiver can recover the
exact bytes and verify integrity. This is the data-plane glue between a file and
the modem.

References
----------
Proakis & Salehi. *Digital Communications* (framing, error detection).
"""

from __future__ import annotations

import zlib

import numpy as np

from uacpy.core.exceptions import ConfigurationError

_HEADER_BYTES = 4   # uint32 payload length
_CRC_BYTES = 4      # uint32 CRC-32


def bytes_to_bits(data):
    """Unpack bytes to an MSB-first 0/1 bit array."""
    return np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8))


def bits_to_bytes(bits):
    """Pack an MSB-first 0/1 bit array back to bytes (truncates a partial byte)."""
    b = np.asarray(bits, dtype=np.uint8).ravel()
    n = (b.size // 8) * 8
    return np.packbits(b[:n]).tobytes()


def pack_frame(payload):
    """Frame a byte payload as ``[len:4][payload][crc32:4]`` -> bit array."""
    payload = bytes(payload)
    header = len(payload).to_bytes(_HEADER_BYTES, "big")
    crc = zlib.crc32(payload).to_bytes(_CRC_BYTES, "big")
    return bytes_to_bits(header + payload + crc)


def unpack_frame(bits):
    """Recover ``(payload_bytes, crc_ok)`` from a framed bit stream.

    Reads the length header, slices the payload, and checks the trailing CRC-32.
    Raises :class:`ConfigurationError` if the stream is too short or truncated.
    """
    data = bits_to_bytes(bits)
    if len(data) < _HEADER_BYTES + _CRC_BYTES:
        raise ConfigurationError("unpack_frame: bit stream shorter than header+CRC")
    length = int.from_bytes(data[:_HEADER_BYTES], "big")
    end = _HEADER_BYTES + length
    if len(data) < end + _CRC_BYTES:
        raise ConfigurationError(
            f"unpack_frame: frame truncated (need {end + _CRC_BYTES} bytes, "
            f"have {len(data)})")
    payload = data[_HEADER_BYTES:end]
    crc_rx = int.from_bytes(data[end:end + _CRC_BYTES], "big")
    return payload, zlib.crc32(payload) == crc_rx
