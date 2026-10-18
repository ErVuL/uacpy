"""WAV output for signals uacpy computes or measures.

One writer, :func:`write_wav`, covering the encodings underwater work
actually uses:

* ``pcm16`` / ``pcm24`` — what passive-acoustic recorders write and what
  every player and analysis tool reads. Integer encodings are bounded to
  ±1 full scale.
* ``pcm32`` — the same, at the widest integer depth.
* ``float32`` / ``float64`` — unbounded, so a **calibrated** signal keeps its
  absolute values (µPa, or whatever units the caller is in) whatever its
  size. This is the encoding to export a modelled received level in.

The samples are written as given (``normalize=False``): a signal inside ±1
keeps its level in any encoding, and a sample a ``pcm*`` encoding cannot hold
is clipped with a warning. ``normalize=True`` scales the peak to full scale,
which is what a waveform meant for playback wants and what throws a
calibrated level away.

``metadata`` writes a standard ``LIST``/``INFO`` chunk, the metadata block
players and audio editors display. The broadcast-WAV ``bext`` chunk that
passive-acoustic-monitoring tooling reads is *not* written; nothing in uacpy
needs it yet.

:func:`read_wav` reads back what :func:`write_wav` writes, or any span of
frames of it, :func:`read_wav_info` reports the encoding and length from the
headers alone, and :func:`read_wav_metadata` recovers the ``INFO`` block, so
a file this module produces is not a one-way trip. :func:`read_wav` accepts
the encodings the writer emits, in a plain or a ``WAVE_FORMAT_EXTENSIBLE``
header, so a recorder's own 24-bit or multichannel file reads too.
"""

import numbers
import struct
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional, Union

import numpy as np

from uacpy.core._export import ExportRecord
from uacpy.core._validate import _MAX_CHANNELS_GUESSED
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, FileFormatError, IOWarning,
    MissingFileError,
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

#: The largest RIFF size field: a 32-bit unsigned count of the bytes after
#: it, so a ``.wav`` holds at most about 4 GiB of samples.
_MAX_RIFF_SIZE = 0xFFFFFFFF


def _refuse_past_riff_limit(n_bytes: int, encoding: str) -> None:
    """Refuse a ``.wav`` whose RIFF size would be ``n_bytes`` when that does
    not fit the header's 32-bit field."""
    if n_bytes > _MAX_RIFF_SIZE:
        raise ConfigurationError(
            f"write_wav: the file would need a RIFF size of {n_bytes} bytes, "
            f"past the {_MAX_RIFF_SIZE} the WAV header's 32-bit field holds "
            f"(about 4 GiB) at encoding={encoding!r}.",
            remediation="Write fewer samples per file (split the record), "
                        "fewer channels, or a narrower encoding ('pcm16', "
                        "'float32').")

# Above _MAX_CHANNELS_GUESSED channels (core._validate, shared with the
# estimators' time-axis guard), a signal with more channels than frames is
# read as a channel-first ``(n_channels, n)`` array rather than written as a
# file of thousands of channels a few samples long.

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
    normalize: bool = False,
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
        Divide by the peak absolute value before writing, so the peak
        lands on full scale. Default ``False``: the samples are written as
        given, so a calibrated signal keeps its level (full scale is ±1 in
        its units under ``pcm*``). A ``pcm*`` encoding cannot hold a sample
        outside ±1, so such samples are clipped with a warning naming
        ``normalize=True``; a ``float*`` encoding holds any value. With
        ``normalize=True`` a signal whose peak is ``p`` is scaled by
        ``-20·log10(p)`` dB. An all-zero signal is written as silence rather
        than divided.
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
        samples, a non-positive ``sample_rate``, or a file past the about
        4 GiB a WAV header's 32-bit RIFF size can count.

    Notes
    -----
    Writing a signal to a ``pcm*`` encoding without ``normalize=True``
    clips anything past full scale and warns with the number of samples
    affected.

    Examples
    --------
    A 10 ms tone at 48 kHz (peak 1, full scale under ``pcm16``), written to
    a temporary directory so running the example leaves nothing behind:

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

    if normalize:
        peak = float(np.max(np.abs(samples)))
        # No epsilon in the divisor: a fixed one under-scales a quiet signal
        # by its own size. Silence is the only case that cannot be divided,
        # and it is already what it should be.
        if peak > 0.0:
            samples = samples / peak

    flat = np.ascontiguousarray(samples.reshape(-1))   # frame-major interleave
    # The samples alone, before the header bytes: refused before encoding.
    _refuse_past_riff_limit(flat.size * (bits // 8), encoding)
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
    _refuse_past_riff_limit(riff_size, encoding)
    with open(filepath, 'wb') as handle:
        handle.write(b'RIFF' + struct.pack('<I', riff_size) + b'WAVE')
        handle.write(chunks)
        handle.write(b'data' + struct.pack('<I', len(data)))
        handle.write(data)
        handle.write(data_pad)


def _wav_path(filepath: Union[str, Path], who: str) -> Path:
    """The ``.wav`` the caller names, as a :class:`~pathlib.Path`.

    A path that names no file raises
    :class:`~uacpy.core.exceptions.MissingFileError` (a
    :class:`ConfigurationError` and a :class:`FileNotFoundError`) naming
    ``who``; a file that is there but wrong raises :class:`FileFormatError`
    from the walk below.
    """
    path = Path(filepath)
    if not path.is_file():
        raise MissingFileError(
            f"{who}: no file at {str(path)!r}.",
            remediation="Check the path and its extension.")
    return path


def _chunks(handle, file_size: int, who: str):
    """Walk a RIFF file's top-level chunks, yielding ``(id, size, start)``.

    Only the 8-byte chunk headers are read: ``size`` is the length the header
    declares and ``start`` the byte offset of the payload, so a caller reads
    the payload it needs (``min(size, file_size - start)`` bytes are present)
    and skips the rest — the samples of a long recording are never read to
    find the chunk after them. Chunks are word-aligned with a pad byte the
    declared size does not count, so the walk steps by the padded length or
    it drifts one byte and reads the rest of the file as garbage.
    """
    handle.seek(0)
    head = handle.read(12)
    if head[:4] != b'RIFF' or head[8:12] != b'WAVE':
        raise FileFormatError(
            f"{who}: not a RIFF/WAVE file.",
            remediation="Check the path; this reads PCM and IEEE-float .wav "
                        "files, not .aiff/.flac/.mp3.")
    offset = 12
    while offset + 8 <= file_size:
        handle.seek(offset)
        header = handle.read(8)
        size = struct.unpack('<I', header[4:8])[0]
        yield header[:4], size, offset + 8
        offset += 8 + size + (size % 2)


class _WavLayout(NamedTuple):
    """Where a ``.wav``'s samples are and how they are encoded."""

    encoding: str
    format_tag: int
    bits: int
    n_channels: int
    rate: int
    data_start: int
    n_frames: int


def _wav_layout(filepath: Union[str, Path], who: str) -> _WavLayout:
    """The format and the sample span of a ``.wav``, from its headers alone.

    Warns (``IOWarning``) and keeps the whole frames present when the
    ``data`` chunk holds fewer bytes than it declares, and when a header was
    never finalised (data size 0 with samples past the declared RIFF size).
    """
    path = _wav_path(filepath, who)
    file_size = path.stat().st_size
    fmt = data_size = data_start = None
    with open(path, 'rb') as handle:
        for chunk_id, size, start in _chunks(handle, file_size, who):
            if chunk_id == b'fmt ' and fmt is None:
                handle.seek(start)
                fmt = handle.read(min(size, file_size - start))
            elif chunk_id == b'data' and data_size is None:
                data_size, data_start = size, start
            if fmt is not None and data_size is not None:
                break
        handle.seek(4)
        riff_size = struct.unpack('<I', handle.read(4))[0]
    if fmt is None or data_size is None:
        raise FileFormatError(
            f"{who}: {'fmt ' if fmt is None else 'data'} chunk missing.",
            remediation="The file is truncated or not a wav; every WAVE file "
                        "carries both.")

    if len(fmt) < 16:
        raise FileFormatError(
            f"{who}: fmt chunk holds {len(fmt)} bytes; a WAVE format "
            f"block is at least 16.",
            remediation="The file is truncated or not a wav.")
    format_tag, n_channels, rate = struct.unpack('<HHI', fmt[:8])
    bits = struct.unpack('<H', fmt[14:16])[0]
    if format_tag == _EXTENSIBLE_TAG and len(fmt) >= 26:
        format_tag = struct.unpack('<H', fmt[24:26])[0]
    encoding = {value: key for key, value in _ENCODINGS.items()}.get(
        (format_tag, bits))
    if encoding is None:
        raise FileFormatError(
            f"{who}: format tag {format_tag} at {bits} bits is not one "
            f"this module handles.",
            remediation=f"It reads {sorted(_ENCODINGS)} — PCM and IEEE float, "
                        f"plain or extensible. A compressed wav needs "
                        f"decoding first (scipy.io.wavfile or soundfile).")

    frame_bytes = n_channels * bits // 8
    tail = file_size - data_start
    available = min(data_size, tail)
    if (frame_bytes and data_size == 0 and tail > 0
            and (riff_size in (0, 0xFFFFFFFF) or riff_size + 8 <= data_start)):
        # A header never finalised — a recorder that lost power, a writer
        # streaming to a pipe — leaves the data size at 0 and the RIFF size
        # short of (or a placeholder for) the samples written after it. The
        # bytes past the data header are those samples; a genuinely empty
        # data chunk sits inside the RIFF size the header declares.
        whole = tail - tail % frame_bytes
        warnings.warn(
            f"{who}: the data chunk declares 0 bytes but {tail} "
            f"follow it past the RIFF size the header declares; returning "
            f"them as {whole // frame_bytes} whole frame(s). The header was "
            f"never finalised (an interrupted recording, or one streamed to "
            f"a pipe).",
            IOWarning, skip_file_prefixes=USER_FRAME_SKIP)
        available = whole
    if frame_bytes and available < data_size:
        whole = available - available % frame_bytes
        warnings.warn(
            f"{who}: the data chunk declares {data_size} bytes but the "
            f"file holds {available}; returning the {whole // frame_bytes} "
            f"whole frame(s) present. The file was cut short (an "
            f"interrupted copy or recording).",
            IOWarning, skip_file_prefixes=USER_FRAME_SKIP)
        available = whole
    if frame_bytes == 0 or available % frame_bytes:
        raise FileFormatError(
            f"{who}: data chunk holds {available} bytes, not a whole "
            f"number of {frame_bytes}-byte frames ({n_channels} channel(s) "
            f"at {bits} bits).",
            remediation="The file is truncated or its header does not "
                        "describe its samples.")
    return _WavLayout(encoding=encoding, format_tag=format_tag, bits=bits,
                      n_channels=n_channels, rate=rate,
                      data_start=data_start,
                      n_frames=available // frame_bytes)


#: Samples decoded per block. A block holds at most 7 bytes per sample
#: beside the output (the packed bytes, and a 24-bit sample's int32 word;
#: float64 reads straight into the output), so a read peaks at its float64
#: output plus under 0.5 MiB however long the file.
_BLOCK_SAMPLES = 1 << 16


def _frame_count(value, name: str, limit: int, hint: str) -> int:
    """``value`` as a whole number of frames in ``[0, limit]``."""
    number = (float(value) if isinstance(value, numbers.Real)
              and not isinstance(value, bool) else float('nan'))
    if not (np.isfinite(number) and number.is_integer()
            and 0 <= number <= limit):
        raise ConfigurationError(
            f"read_wav: {name}={value!r} is not a whole number of frames "
            f"in [0, {limit}]; {hint}.",
            remediation="Pass an integer frame count, e.g. "
                        "int(round(t * sample_rate)) for a time t in "
                        "seconds; read_wav_info(path).n_frames is the "
                        "file's length.")
    return int(number)


def _read_into(handle, target: memoryview) -> None:
    """Fill ``target`` from ``handle``'s position, or refuse a file that ends
    first."""
    if handle.readinto(target) != len(target):
        raise FileFormatError(
            "read_wav: the file ended before the frames its header "
            "describes (it changed while it was read).",
            remediation="Read it again once nothing is writing to it.")


def _decode(handle, layout: _WavLayout, n_samples: int) -> np.ndarray:
    """``n_samples`` samples from ``handle``'s position as float64, decoded
    block by block into the one output array."""
    out = np.empty(n_samples, dtype=np.float64)
    if (layout.format_tag, layout.bits) == (3, 64) and sys.byteorder == 'little':
        # The file's own little-endian float64: read straight into the output.
        _read_into(handle, memoryview(out.view(np.uint8)))
        return out
    sample_bytes = layout.bits // 8
    block = min(max(1, _BLOCK_SAMPLES // layout.n_channels)
                * layout.n_channels, max(n_samples, 1))
    buffer = bytearray(block * sample_bytes)
    view = memoryview(buffer)
    dtype = {(1, 16): '<i2', (1, 32): '<i4', (3, 32): '<f4',
             (3, 64): '<f8'}.get((layout.format_tag, layout.bits))
    # 24-bit has no numpy dtype: each 3-byte little-endian sample goes into
    # the TOP three bytes of an int32 word, and an arithmetic shift right by
    # 8 brings it down with the sign bit where two's complement expects it.
    # The shift discards the low byte, so it is never cleared.
    words = (np.empty((block, 4), dtype=np.uint8) if dtype is None else None)
    done = 0
    while done < n_samples:
        count = min(block, n_samples - done)
        _read_into(handle, view[:count * sample_bytes])
        if dtype is None:
            words[:count, 1:] = np.frombuffer(
                buffer, dtype=np.uint8, count=3 * count).reshape(count, 3)
            word = words[:count].view('<i4').ravel()
            word >>= 8
            out[done:done + count] = word
        else:
            out[done:done + count] = np.frombuffer(buffer, dtype=dtype,
                                                   count=count)
        done += count
    if layout.format_tag == 1:
        out /= 2.0 ** (layout.bits - 1) - 1.0
    return out


def read_wav(filepath: Union[str, Path], *, start=0, frames=None):
    """Read a PCM or IEEE-float ``.wav``, such as :func:`write_wav` writes,
    or the ``frames`` frames of it from frame ``start``.

    A ``WAVE_FORMAT_EXTENSIBLE`` header (format tag ``0xFFFE``, what most
    24-bit and multichannel recorders write) is read through the format tag
    its SubFormat GUID carries.

    Only the headers and the frames asked for are read: the chunk walk seeks
    past every payload it does not need, and the samples are decoded block by
    block into the output, so the memory a read takes is its float64 output
    (8 bytes per sample) plus a fixed few MiB, whatever the file's length and
    encoding. An excerpt of an hour-long recording is
    ``read_wav(path, start=int(600 * fs), frames=int(10 * fs))``.

    Parameters
    ----------
    filepath : str or Path
        The file to read.
    start : int, optional
        The first frame read (0-based; one frame holds one sample of every
        channel). Default 0. A float with an integral value is accepted.
    frames : int, optional
        How many frames to read from ``start``. ``None`` (default) reads to
        the end of the file.

    Returns
    -------
    signal : ndarray
        ``(n,)`` for mono, ``(n, n_channels)`` de-interleaved otherwise. An
        integer encoding comes back divided by its full scale, so it lands in
        ±1 whatever its bit depth and a round trip through ``pcm16`` and
        ``pcm24`` gives the same numbers to within their quantisation. A float
        encoding comes back as written — that is the point of it, so nothing
        rescales a calibrated signal on the way in. The time axis is the
        first, as in scipy and soundfile: pass ``axis=0`` to the spectral
        functions (``uacpy.welch`` …), whose default is the last axis and
        which refuse a multichannel ``(n, n_channels)`` array without it.
    sample_rate : float
        Hz. The argument order mirrors ``write_wav(filepath, signal, sample_rate)``.

    Raises
    ------
    ConfigurationError
        ``filepath`` names no file, or ``start`` / ``frames`` is not a whole
        number of frames inside the file (``start`` in ``[0, n_frames]``,
        ``start + frames <= n_frames``).
    FileFormatError
        Not a RIFF/WAVE file, no ``fmt ``/``data`` chunk, a ``fmt `` chunk
        shorter than a format block, a ``data`` chunk that is not a whole
        number of frames, or a format this module does not read
        (compressed, or a bit depth outside 16/24/32).

    Warns
    -----
    IOWarning
        The ``data`` chunk holds fewer bytes than its header declares (an
        interrupted copy or recording). The file's frames are then the
        whole frames present, and the warning names declared and available
        bytes. Likewise a header never finalised: a ``data`` chunk declaring
        0 bytes with samples after it past the declared RIFF size is read to
        the end of the file.

    See Also
    --------
    read_wav_info : The encoding, channel count, rate and length, from the
        headers alone.
    """
    layout = _wav_layout(filepath, 'read_wav')
    first = _frame_count(start, 'start', layout.n_frames,
                         f"the file holds {layout.n_frames} frame(s)")
    count = (layout.n_frames - first if frames is None else _frame_count(
        frames, 'frames', layout.n_frames - first,
        f"{layout.n_frames - first} frame(s) follow start={first}"))
    frame_bytes = layout.n_channels * layout.bits // 8
    with open(Path(filepath), 'rb') as handle:
        handle.seek(layout.data_start + first * frame_bytes)
        samples = _decode(handle, layout, count * layout.n_channels)
    if layout.n_channels > 1:
        samples = samples.reshape(-1, layout.n_channels)
    return samples, float(layout.rate)


@dataclass(frozen=True, eq=False)
class WavInfo(ExportRecord):
    """A ``.wav``'s encoding and length, as :func:`read_wav_info` reads them
    from its headers.

    Attributes
    ----------
    encoding : {'pcm16', 'pcm24', 'pcm32', 'float32', 'float64'}
        The sample format, in :func:`write_wav`'s own keys.
    bits : int
        Bits per sample.
    n_channels : int
        Channels per frame.
    sample_rate : float
        Frames per second (Hz).
    n_frames : int
        The whole frames the file holds: what :func:`read_wav` returns, and
        the bound of its ``start`` / ``frames``.
    """

    encoding: str
    bits: int
    n_channels: int
    sample_rate: float
    n_frames: int

    _REPR_UNITS = {'sample_rate': 'Hz'}


def read_wav_info(filepath: Union[str, Path]) -> WavInfo:
    """The encoding, channel count, sample rate and length of a ``.wav``,
    read from its headers without reading its samples.

    The bit depth is what turns :func:`read_wav`'s output back into ADC
    counts. ``read_wav`` divides an integer encoding by its symmetric full
    scale ``2**(bits-1) - 1``; ``uacpy.acoustics.pressure_from_recording``
    takes that output with ``x_scale='read_wav'`` and the bit depth, and
    applies the ADC convention exactly. So a recording calibrated by its ADC
    reference reads as::

        x, fs = read_wav(path)
        p = uacpy.acoustics.pressure_from_recording(
            x, sensitivity, gain,
            volt_params=(read_wav_info(path).bits, v_ref), x_scale='read_wav')

    A float encoding (``info.encoding`` ``'float32'`` / ``'float64'``)
    carries the values themselves and has no counts: pass its samples to
    ``pressure_from_recording`` as volts, ``x * v_ref``, with no
    ``volt_params``.

    Parameters
    ----------
    filepath : str or Path
        The file to read.

    Returns
    -------
    WavInfo
        ``encoding``, ``bits``, ``n_channels``, ``sample_rate`` (Hz) and
        ``n_frames``.

    Raises
    ------
    ConfigurationError
        ``filepath`` names no file.
    FileFormatError
        As :func:`read_wav`.

    Warns
    -----
    IOWarning
        As :func:`read_wav`: a ``data`` chunk cut short, or a header never
        finalised; ``n_frames`` counts the whole frames present.
    """
    layout = _wav_layout(filepath, 'read_wav_info')
    return WavInfo(encoding=layout.encoding, bits=layout.bits,
                   n_channels=layout.n_channels,
                   sample_rate=float(layout.rate), n_frames=layout.n_frames)


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
    path = _wav_path(filepath, 'read_wav_metadata')
    file_size = path.stat().st_size
    with open(path, 'rb') as handle:
        for chunk_id, size, start in _chunks(handle, file_size,
                                             'read_wav_metadata'):
            if chunk_id != b'LIST':
                continue
            handle.seek(start)
            payload = handle.read(min(size, file_size - start))
            if payload[:4] != b'INFO':
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
