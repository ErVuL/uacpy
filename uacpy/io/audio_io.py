"""WAV output for signals uacpy computes or measures.

One writer, :func:`write_wav`, covering the encodings underwater work
actually uses:

* ``pcm16`` / ``pcm24`` — what passive-acoustic recorders write and what
  every player and analysis tool reads. Integer encodings are bounded, so
  they carry a signal's *shape*, not its level.
* ``pcm32`` — the same, at the widest integer depth.
* ``float32`` / ``float64`` — unbounded, so a **calibrated** signal keeps its
  absolute values (µPa, or whatever units the caller is in). This is the
  encoding to export a modelled received level in: normalising to full scale
  is exactly what throws the level away.

That difference is why ``normalize`` defaults to *auto* — on for the integer
encodings, which cannot represent a signal outside ±1, and off for the float
encodings, which can. Say ``normalize=True`` or ``normalize=False`` to decide
it yourself.

``metadata`` writes a standard ``LIST``/``INFO`` chunk, the metadata block
players and audio editors display. The broadcast-WAV ``bext`` chunk that
passive-acoustic-monitoring tooling reads is *not* written; nothing in uacpy
needs it yet.

:func:`read_wav` reads back what :func:`write_wav` writes, and
:func:`read_wav_metadata` recovers the ``INFO`` block, so a file this module
produces is not a one-way trip. :func:`read_wav` accepts the encodings the
writer emits, in a plain or a ``WAVE_FORMAT_EXTENSIBLE`` header, so a
recorder's own 24-bit or multichannel file reads too.
"""

import struct
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np

from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, FileFormatError, IOWarning,
)

#: ``encoding`` → (WAVE format tag, bits per sample). Tag 1 is PCM, 3 is
#: IEEE float.
_ENCODINGS = {
    'pcm16': (1, 16),
    'pcm24': (1, 24),
    'pcm32': (1, 32),
    'float32': (3, 32),
    'float64': (3, 64),
}

#: ``WAVE_FORMAT_EXTENSIBLE``: the real format tag is the first two bytes of
#: the SubFormat GUID at ``fmt[24:26]``.
_EXTENSIBLE_TAG = 0xFFFE

#: Above this many channels, a signal with more channels than frames is read
#: as a channel-first ``(n_channels, n)`` array rather than written as a file
#: of thousands of channels a few samples long.
_MAX_CHANNELS_GUESSED = 64

#: ``metadata`` key → its four-character ``INFO`` chunk id.
_INFO_TAGS = {
    'title': 'INAM',
    'artist': 'IART',
    'comment': 'ICMT',
    'date': 'ICRD',
    'software': 'ISFT',
    'copyright': 'ICOP',
    'engineer': 'IENG',
    'source': 'ISRC',
    'subject': 'ISBJ',
    'keywords': 'IKEY',
}


def _chunk(chunk_id: bytes, payload: bytes) -> bytes:
    """A RIFF chunk: id, little-endian size, payload, pad to even length.

    The pad byte is not counted in the declared size — a reader that trusts
    the size and one that walks chunk to chunk have to agree.
    """
    pad = b'\x00' if len(payload) % 2 else b''
    return chunk_id + struct.pack('<I', len(payload)) + payload + pad


def _info_chunk(metadata: dict) -> bytes:
    """The ``LIST``/``INFO`` block for ``metadata``, or ``b''`` if empty."""
    unknown = sorted(set(metadata) - set(_INFO_TAGS))
    if unknown:
        raise ConfigurationError(
            f"write_wav: metadata key(s) {unknown} have no INFO tag.",
            remediation=f"Use one of {sorted(_INFO_TAGS)}, or put the text "
                        f"in 'comment'.",
        )
    entries = b''
    for key, tag in _INFO_TAGS.items():          # a fixed, reproducible order
        if key not in metadata:
            continue
        # INFO text is NUL-terminated; the terminator counts toward the size.
        text = str(metadata[key]).encode('utf-8') + b'\x00'
        entries += _chunk(tag.encode('ascii'), text)
    return _chunk(b'LIST', b'INFO' + entries) if entries else b''


def write_wav(
    filepath: Union[str, Path],
    signal: np.ndarray,
    sample_rate: float,
    *,
    encoding: str = 'pcm16',
    normalize: Optional[bool] = None,
    metadata: Optional[dict] = None,
) -> None:
    """Write a real signal to a ``.wav`` file.

    Parameters
    ----------
    filepath : str or Path
        Destination. Overwritten if it exists.
    signal : ndarray
        Real samples, ``(n,)`` mono or ``(n, n_channels)`` — one column per
        channel, written interleaved. Must be finite.
    sample_rate : float
        Sample rate in Hz, rounded to the nearest integer for the header;
        a rate that is not an integer warns with the relative error.
    encoding : {'pcm16', 'pcm24', 'pcm32', 'float32', 'float64'}
        Sample format. The integer encodings map ±1.0 to full scale; the
        float encodings write the values themselves. Default ``'pcm16'``.
    normalize : bool, optional
        Divide by the peak absolute value before writing. Default ``None``
        selects by encoding: ``True`` for ``pcm*`` (which cannot hold a
        sample outside ±1) and ``False`` for ``float*`` (which can, so a
        calibrated signal keeps its absolute level). An all-zero signal is
        written as silence rather than divided.
    metadata : dict, optional
        Any of ``title``, ``artist``, ``comment``, ``date``, ``software``,
        ``copyright``, ``engineer``, ``source``, ``subject``, ``keywords``,
        written as a ``LIST``/``INFO`` chunk. An unknown key raises.

    Raises
    ------
    ConfigurationError
        Unknown ``encoding`` or metadata key, empty, complex or >2-D
        ``signal``, a 2-D ``signal`` with more channels than frames and more
        than 64 channels (a channel-first array: pass its ``.T``), non-finite
        samples, or a non-positive ``sample_rate``.

    Notes
    -----
    Writing an un-normalised signal to a ``pcm*`` encoding clips anything
    past full scale and warns with the number of samples affected.

    Examples
    --------
    A 10 ms tone at 48 kHz, written to a temporary directory so running the
    example leaves nothing behind:

    >>> import os, tempfile
    >>> t = np.arange(0, 0.01, 1 / 48000)
    >>> tone = np.sin(2 * np.pi * 1000 * t)
    >>> with tempfile.TemporaryDirectory() as d:
    ...     write_wav(os.path.join(d, 'tone.wav'), tone, 48000)
    ...     print(open(os.path.join(d, 'tone.wav'), 'rb').read(4).decode())
    RIFF

    The same signal at a calibrated level, exported without rescaling so the
    absolute values survive:

    >>> with tempfile.TemporaryDirectory() as d:
    ...     path = os.path.join(d, 'level.wav')
    ...     write_wav(path, 12.5 * tone, 48000, encoding='float32',
    ...               metadata={'comment': 'modelled receive level, µPa'})
    ...     print(os.path.getsize(path) > 4 * tone.size)
    True
    """
    if encoding not in _ENCODINGS:
        raise ConfigurationError(
            f"write_wav: unknown encoding {encoding!r}.",
            remediation=f"Use one of {sorted(_ENCODINGS)}; 'float32' keeps a "
                        f"calibrated signal's absolute level.",
        )
    format_tag, bits = _ENCODINGS[encoding]

    if np.iscomplexobj(signal):
        raise ConfigurationError(
            "write_wav: signal is complex; a WAV file holds real samples, and "
            "casting would discard the imaginary part.",
            remediation="Upconvert a baseband record to passband first "
                        "(uacpy.comms.upconvert), or pass np.real(signal) if "
                        "the real part is what you mean.")
    samples = np.asarray(signal, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]
    elif samples.ndim != 2:
        raise ConfigurationError(
            f"write_wav: signal is {samples.ndim}-D.",
            remediation="Pass (n,) for mono or (n, n_channels) for "
                        "multichannel.",
        )
    n_frames, n_channels = samples.shape
    if n_channels > max(n_frames, _MAX_CHANNELS_GUESSED):
        raise ConfigurationError(
            f"write_wav: signal is shaped {samples.shape}, which reads as "
            f"(n_channels, n) — {n_channels} channels of {n_frames} frames "
            f"each.",
            remediation="Pass signal.T: write_wav takes (n, n_channels), the "
                        "layout read_wav returns.")
    if n_frames == 0 or n_channels == 0:
        raise ConfigurationError(
            f"write_wav: signal has shape {samples.shape} — nothing to write.",
            remediation="Pass at least one sample of at least one channel.",
        )
    if not np.isfinite(samples).all():
        n_bad = int((~np.isfinite(samples)).sum())
        raise ConfigurationError(
            f"write_wav: {n_bad} of {samples.size} samples are NaN or inf.",
            remediation="A non-finite sample has no encoding — clean the "
                        "signal (np.nan_to_num, or drop the bad span) first.",
        )
    rate = int(round(float(sample_rate)))
    if rate <= 0:
        raise ConfigurationError(
            f"write_wav: sample rate {sample_rate} is not positive.",
            remediation="Pass the rate the signal was generated at, in Hz.",
        )
    if rate != float(sample_rate):
        # The WAV header holds an integer rate; the file's time axis then
        # runs at rate/sample_rate of the signal's own.
        warnings.warn(
            f"write_wav: the sample rate {float(sample_rate)!r} Hz is written "
            f"as {rate} Hz, the integer the WAV header holds; the file's "
            f"time axis is off by a relative "
            f"{abs(rate - float(sample_rate)) / float(sample_rate):.3g}.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)

    if normalize is None:
        normalize = format_tag == 1
    if normalize:
        peak = float(np.max(np.abs(samples)))
        # No epsilon in the divisor: a fixed one under-scales a quiet signal
        # by its own size. Silence is the only case that cannot be divided,
        # and it is already what it should be.
        if peak > 0.0:
            samples = samples / peak

    flat = np.ascontiguousarray(samples.reshape(-1))   # frame-major interleave
    if format_tag == 1:
        over = np.abs(flat) > 1.0
        if over.any():
            warnings.warn(
                f"write_wav: {int(over.sum())} of {flat.size} samples are "
                f"past full scale for {encoding} and were clipped — pass "
                f"normalize=True to rescale, or encoding='float32' to keep "
                f"the absolute values.",
                FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
            flat = np.clip(flat, -1.0, 1.0)
        # Symmetric full scale (2**(bits-1) - 1), so +1.0 and -1.0 encode to
        # equal and opposite codes and neither overflows.
        quantized = np.round(flat * (2.0 ** (bits - 1) - 1.0))
        if bits == 16:
            data = quantized.astype('<i2').tobytes()
        elif bits == 32:
            data = quantized.astype('<i4').tobytes()
        else:                                    # 24-bit: low 3 bytes of int32
            words = np.ascontiguousarray(quantized.astype('<i4'))
            data = words.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    else:
        data = flat.astype('<f4' if bits == 32 else '<f8').tobytes()

    block_align = n_channels * bits // 8
    fmt = struct.pack(
        '<HHIIHH', format_tag, n_channels, rate,
        rate * block_align, block_align, bits,
    )
    if format_tag != 1:
        # Non-PCM formats carry the extension size (0 here) and a `fact`
        # chunk giving the frame count; readers use both to tell a float
        # file from a PCM one before they reach the samples.
        fmt += struct.pack('<H', 0)
    chunks = _chunk(b'fmt ', fmt)
    if format_tag != 1:
        chunks += _chunk(b'fact', struct.pack('<I', n_frames))
    if metadata:
        chunks += _info_chunk(metadata)

    data_pad = b'\x00' if len(data) % 2 else b''
    riff_size = 4 + len(chunks) + 8 + len(data) + len(data_pad)
    with open(filepath, 'wb') as handle:
        handle.write(b'RIFF' + struct.pack('<I', riff_size) + b'WAVE')
        handle.write(chunks)
        handle.write(b'data' + struct.pack('<I', len(data)))
        handle.write(data)
        handle.write(data_pad)


def _read_riff(filepath: Union[str, Path]) -> bytes:
    """The bytes of a ``.wav`` the caller names.

    A path that does not exist is the argument's fault, so it raises
    :class:`ConfigurationError`; a file that is there but wrong raises
    :class:`FileFormatError` from the walk below.
    """
    path = Path(filepath)
    if not path.is_file():
        raise ConfigurationError(
            f"read_wav: no file at {str(path)!r}.",
            remediation="Check the path and its extension.")
    return path.read_bytes()


def _chunks(raw: bytes):
    """Walk a RIFF file's top-level chunks, yielding ``(id, payload, size)``.

    ``size`` is the length the chunk header declares; ``payload`` holds the
    bytes actually present, so a chunk cut short by the end of the file has
    ``len(payload) < size``. Chunks are word-aligned with a pad byte the
    declared size does not count, so the walk steps by the padded length or
    it drifts one byte and reads the rest of the file as garbage.
    """
    if raw[:4] != b'RIFF' or raw[8:12] != b'WAVE':
        raise FileFormatError(
            "read_wav: not a RIFF/WAVE file.",
            remediation="Check the path; this reads PCM and IEEE-float .wav "
                        "files, not .aiff/.flac/.mp3.")
    offset = 12
    while offset + 8 <= len(raw):
        chunk_id = raw[offset:offset + 4]
        size = struct.unpack('<I', raw[offset + 4:offset + 8])[0]
        yield chunk_id, raw[offset + 8:offset + 8 + size], size
        offset += 8 + size + (size % 2)


def read_wav(filepath: Union[str, Path]):
    """Read a PCM or IEEE-float ``.wav``, such as :func:`write_wav` writes.

    A ``WAVE_FORMAT_EXTENSIBLE`` header (format tag ``0xFFFE``, what most
    24-bit and multichannel recorders write) is read through the format tag
    its SubFormat GUID carries.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    signal : ndarray
        ``(n,)`` for mono, ``(n, n_channels)`` de-interleaved otherwise. An
        integer encoding comes back divided by its full scale, so it lands in
        ±1 whatever its bit depth and a round trip through ``pcm16`` and
        ``pcm24`` gives the same numbers to within their quantisation. A float
        encoding comes back as written — that is the point of it, so nothing
        rescales a calibrated signal on the way in.
    sample_rate : float
        Hz. The argument order mirrors ``write_wav(filepath, signal, sample_rate)``.

    Raises
    ------
    ConfigurationError
        ``filepath`` names no file.
    FileFormatError
        Not a RIFF/WAVE file, no ``fmt ``/``data`` chunk, a ``fmt `` chunk
        shorter than a format block, a ``data`` chunk that is not a whole
        number of frames, or a format this module does not read
        (compressed, or a bit depth outside 16/24/32).

    Warns
    -----
    IOWarning
        The ``data`` chunk holds fewer bytes than its header declares (an
        interrupted copy or recording). The whole frames present are
        returned, and the warning names declared and available bytes.
    """
    raw = _read_riff(filepath)
    fmt = data = None
    for chunk_id, payload, size in _chunks(raw):
        if chunk_id == b'fmt ' and fmt is None:
            fmt = payload
        elif chunk_id == b'data' and data is None:
            data, data_size = payload, size
    if fmt is None or data is None:
        raise FileFormatError(
            f"read_wav: {'fmt ' if fmt is None else 'data'} chunk missing.",
            remediation="The file is truncated or not a wav; every WAVE file "
                        "carries both.")

    if len(fmt) < 16:
        raise FileFormatError(
            f"read_wav: fmt chunk holds {len(fmt)} bytes; a WAVE format "
            f"block is at least 16.",
            remediation="The file is truncated or not a wav.")
    format_tag, n_channels, rate = struct.unpack('<HHI', fmt[:8])
    bits = struct.unpack('<H', fmt[14:16])[0]
    if format_tag == _EXTENSIBLE_TAG and len(fmt) >= 26:
        format_tag = struct.unpack('<H', fmt[24:26])[0]
    if (format_tag, bits) not in {(tag, b) for tag, b in _ENCODINGS.values()}:
        raise FileFormatError(
            f"read_wav: format tag {format_tag} at {bits} bits is not one "
            f"this module handles.",
            remediation=f"It reads {sorted(_ENCODINGS)} — PCM and IEEE float, "
                        f"plain or extensible. A compressed wav needs "
                        f"decoding first (scipy.io.wavfile or soundfile).")

    frame_bytes = n_channels * bits // 8
    if frame_bytes and len(data) < data_size:
        whole = len(data) - len(data) % frame_bytes
        warnings.warn(
            f"read_wav: the data chunk declares {data_size} bytes but the "
            f"file holds {len(data)}; returning the {whole // frame_bytes} "
            f"whole frame(s) present. The file was cut short (an "
            f"interrupted copy or recording).",
            IOWarning, skip_file_prefixes=USER_FRAME_SKIP)
        data = data[:whole]
    if frame_bytes == 0 or len(data) % frame_bytes:
        raise FileFormatError(
            f"read_wav: data chunk holds {len(data)} bytes, not a whole "
            f"number of {frame_bytes}-byte frames ({n_channels} channel(s) "
            f"at {bits} bits).",
            remediation="The file is truncated or its header does not "
                        "describe its samples.")

    if format_tag == 1 and bits == 24:
        # 24-bit has no numpy dtype: widen each 3-byte little-endian sample
        # into the TOP three bytes of an int32 and shift back down, so the
        # sign bit lands where two's complement expects it.
        packed = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
        widened = np.zeros((packed.shape[0], 4), dtype=np.uint8)
        widened[:, 1:] = packed
        samples = (widened.view('<i4').ravel() >> 8).astype(np.float64)
        samples /= 2.0 ** (bits - 1) - 1.0
    elif format_tag == 1:
        samples = np.frombuffer(
            data, dtype='<i2' if bits == 16 else '<i4').astype(np.float64)
        samples /= 2.0 ** (bits - 1) - 1.0
    else:
        samples = np.frombuffer(
            data, dtype='<f4' if bits == 32 else '<f8').astype(np.float64)

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)
    return samples, float(rate)


def read_wav_metadata(filepath: Union[str, Path]) -> dict:
    """The ``LIST``/``INFO`` metadata of a ``.wav``, as :func:`write_wav` keys.

    Returns an empty dict when the file carries no ``INFO`` block, which is the
    common case: most recorders write none.

    Parameters
    ----------
    filepath : str or Path
        The file to read.
    """
    by_tag = {tag: key for key, tag in _INFO_TAGS.items()}
    for chunk_id, payload, _size in _chunks(_read_riff(filepath)):
        if chunk_id != b'LIST' or payload[:4] != b'INFO':
            continue
        found, offset = {}, 4
        while offset + 8 <= len(payload):
            tag = payload[offset:offset + 4].decode('ascii', 'replace')
            size = struct.unpack('<I', payload[offset + 4:offset + 8])[0]
            text = payload[offset + 8:offset + 8 + size]
            if tag in by_tag:
                found[by_tag[tag]] = text.split(b'\x00', 1)[0].decode(
                    'utf-8', 'replace')
            offset += 8 + size + (size % 2)
        return found
    return {}
