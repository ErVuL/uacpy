"""Tests for ``uacpy.io``'s wav writer and readers.

What ``write_wav`` writes is read back three ways: ``scipy.io.wavfile``, the
stdlib ``wave`` module, and this module's own ``read_wav``. The first two are
independent parsers, so a header built by hand here cannot pass by agreeing
only with itself; the third pins the round trip callers actually make.
"""

import struct
import wave
import warnings

import numpy as np
import pytest
from scipy.io import wavfile

from uacpy.core.exceptions import ConfigurationError, FileFormatError
from uacpy.io import (
    WavInfo, read_wav, read_wav_info, read_wav_metadata, write_wav,
)
from uacpy.io import audio_io


FS = 48000


def _tone(peak=0.5, n=960):
    t = np.arange(n) / FS
    return peak * np.sin(2 * np.pi * 1000 * t)


@pytest.mark.parametrize('encoding,dtype', [
    ('pcm16', np.int16),
    ('pcm24', np.int32),      # scipy left-justifies 24-bit into int32
    ('pcm32', np.int32),
    ('float32', np.float32),
    ('float64', np.float64),
])
def test_every_encoding_round_trips(tmp_path, encoding, dtype):
    path = tmp_path / f'{encoding}.wav'
    write_wav(path, _tone(), FS, encoding=encoding)

    rate, data = wavfile.read(path)
    assert rate == FS
    assert data.dtype == dtype
    assert data.size == 960


def test_the_pcm_encodings_are_readable_by_the_stdlib_wave_module(tmp_path):
    """A hand-built RIFF header that only scipy accepts is not a wav file."""
    path = tmp_path / 'tone.wav'
    write_wav(path, _tone(), FS, encoding='pcm24')

    with wave.open(str(path)) as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 3
        assert handle.getframerate() == FS
        assert handle.getnframes() == 960


def test_a_pcm_encoding_normalizes_to_full_scale_by_default(tmp_path):
    path = tmp_path / 'half.wav'
    write_wav(path, _tone(peak=0.5), FS)

    _, data = wavfile.read(path)
    assert np.abs(data).max() == 32767


def test_a_float_encoding_keeps_the_absolute_level_by_default(tmp_path):
    """The point of the float encodings: a calibrated signal exported through
    one still carries its level. Normalising would silently discard it."""
    path = tmp_path / 'calibrated.wav'
    write_wav(path, 12.5 * _tone(peak=0.5), FS, encoding='float32')

    _, data = wavfile.read(path)
    assert np.abs(data).max() == pytest.approx(6.25, rel=1e-6)


def test_normalize_scales_a_signal_far_below_full_scale_exactly(tmp_path):
    """A fixed epsilon in the divisor (the shape this was promoted from used
    ``peak + 1e-12``) under-scales by the signal's own size: at a 1e-9 peak it
    lands three orders of magnitude low instead of at full scale."""
    path = tmp_path / 'tiny.wav'
    write_wav(path, 1e-9 * _tone(peak=1.0), FS)

    _, data = wavfile.read(path)
    assert np.abs(data).max() == 32767


def test_silence_is_written_as_silence(tmp_path):
    """The one signal that cannot be divided by its peak."""
    path = tmp_path / 'silence.wav'
    write_wav(path, np.zeros(100), FS)

    _, data = wavfile.read(path)
    assert data.size == 100
    assert np.abs(data).max() == 0


def test_full_scale_is_not_clipped_but_anything_past_it_is(tmp_path):
    """Both sides of the clip threshold: 1.0 is representable, so it must pass
    silently; the first value beyond it must warn and be clipped."""
    at_full_scale = np.array([1.0, -1.0, 0.5])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        write_wav(tmp_path / 'edge.wav', at_full_scale, FS, normalize=False)

    past_full_scale = np.array([1.0 + 1e-9, -1.0, 0.5])
    with pytest.warns(UserWarning, match='past full scale'):
        write_wav(tmp_path / 'over.wav', past_full_scale, FS, normalize=False)

    _, data = wavfile.read(tmp_path / 'over.wav')
    assert data[0] == 32767


def test_a_float_encoding_does_not_clip(tmp_path):
    """Unbounded is the whole contract — no warning, no ceiling."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        write_wav(tmp_path / 'loud.wav', np.array([250.0, -250.0]), FS,
                  encoding='float32')

    _, data = wavfile.read(tmp_path / 'loud.wav')
    assert np.abs(data).max() == pytest.approx(250.0)


def test_channels_are_written_interleaved(tmp_path):
    """Column per channel in, frame-major interleave out."""
    left = _tone(peak=1.0)
    stereo = np.stack([left, -left], axis=1)
    write_wav(tmp_path / 'stereo.wav', stereo, FS, encoding='float64')

    _, data = wavfile.read(tmp_path / 'stereo.wav')
    assert data.shape == (960, 2)
    assert np.allclose(data[:, 0], left)
    assert np.allclose(data[:, 1], -left)


def test_metadata_is_written_as_a_list_info_chunk(tmp_path):
    path = tmp_path / 'tagged.wav'
    write_wav(path, _tone(), FS,
              metadata={'title': 'probe', 'comment': 'uacpy test',
                        'date': '2026-09-10'})

    raw = path.read_bytes()
    assert b'LIST' in raw and b'INFO' in raw
    assert b'INAM' in raw and b'probe\x00' in raw
    assert b'ICMT' in raw and b'ICRD' in raw
    # The samples are still findable behind the new chunk.
    _, data = wavfile.read(path)
    assert data.size == 960


def test_an_unknown_metadata_key_raises(tmp_path):
    with pytest.raises(ConfigurationError, match='no INFO tag'):
        write_wav(tmp_path / 'bad.wav', _tone(), FS,
                  metadata={'hydrophone': 'SoundTrap 300'})


@pytest.mark.parametrize('kwargs,signal,match', [
    (dict(encoding='pcm8'), None, 'unknown encoding'),
    ({}, np.zeros((2, 2, 2)), '3-D'),
    ({}, np.zeros(0), 'nothing to write'),
    ({}, np.array([1.0, np.nan]), 'NaN or inf'),
    ({}, np.array([1.0, np.inf]), 'NaN or inf'),
    ({}, np.zeros(8, dtype=complex), 'signal is complex'),
    ({}, np.zeros((3, 4800)), r'reads as \(n_channels, n\)'),
    ({}, np.zeros((64, 65)), r'reads as \(n_channels, n\)'),
])
def test_rejected_inputs(tmp_path, kwargs, signal, match):
    data = _tone() if signal is None else signal
    with pytest.raises(ConfigurationError, match=match):
        write_wav(tmp_path / 'rejected.wav', data, FS, **kwargs)


@pytest.mark.parametrize('shape', [(4800, 3), (64, 64), (2, 64), (65, 65)])
def test_a_layout_that_can_be_n_by_channels_is_written(tmp_path, shape):
    """More channels than frames is refused only past 64 channels; 64
    channels of 2 frames, or as many frames as channels, is a file."""
    write_wav(tmp_path / 'ok.wav', np.zeros(shape), FS)
    assert read_wav(tmp_path / 'ok.wav')[0].shape == shape


@pytest.mark.parametrize('rate', [0, -48000])
def test_a_non_positive_sample_rate_raises(tmp_path, rate):
    with pytest.raises(ConfigurationError, match='not positive'):
        write_wav(tmp_path / 'rate.wav', _tone(), rate)


def test_a_non_integer_rate_warns_with_its_relative_error(tmp_path):
    with pytest.warns(UserWarning, match=r'written as 48000 Hz.*8\.33e-06'):
        write_wav(tmp_path / 'frac.wav', _tone(), 48000.4)
    assert read_wav(tmp_path / 'frac.wav')[1] == 48000


def test_an_integer_rate_writes_silently(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        write_wav(tmp_path / 'whole.wav', _tone(), 48000.0)


def test_the_rate_is_named_sample_rate(tmp_path):
    """``sample_rate`` like every other callable in the package; ``fs`` is
    not a name ``write_wav`` knows."""
    write_wav(tmp_path / 'named.wav', _tone(), sample_rate=FS)
    assert read_wav(tmp_path / 'named.wav')[1] == FS
    with pytest.raises(TypeError, match="unexpected keyword argument 'fs'"):
        write_wav(tmp_path / 'old.wav', _tone(), fs=FS)


@pytest.mark.parametrize('n_samples, refused', [(482, False), (500, True),
                                                 (501, True)])
def test_a_file_past_the_riff_size_field_is_refused_by_name(
        tmp_path, monkeypatch, n_samples, refused):
    """The RIFF size field is 32-bit. With its limit lowered to 1000 bytes:
    482 pcm16 samples make a RIFF size of exactly 1000 and are written; 500
    pass the samples-only check (1000 bytes) and are refused on the header
    bytes; 501 are refused before encoding."""
    from uacpy.io import audio_io
    monkeypatch.setattr(audio_io, '_MAX_RIFF_SIZE', 1000)
    path = tmp_path / 'limit.wav'
    if refused:
        with pytest.raises(ConfigurationError, match='^write_wav: .*RIFF size'):
            write_wav(path, _tone(n=n_samples), FS)
        assert not path.exists()
    else:
        write_wav(path, _tone(n=n_samples), FS)
        assert struct.unpack('<I', path.read_bytes()[4:8])[0] == 1000


def test_the_written_file_is_byte_even(tmp_path):
    """RIFF chunks are word-aligned: an odd-length data chunk carries a pad
    byte that the declared size does not count."""
    write_wav(tmp_path / 'odd.wav', np.zeros(3), FS, encoding='pcm24')

    raw = (tmp_path / 'odd.wav').read_bytes()
    assert len(raw) % 2 == 0
    # 3 frames x 3 bytes = 9 declared, 10 written.
    assert raw[raw.index(b'data') + 4:raw.index(b'data') + 8] == (
        (9).to_bytes(4, 'little'))


# ── reading back what the writer wrote ─────────────────────────────────────


@pytest.mark.parametrize('encoding,tolerance', [
    ('pcm16', 2e-5),        # one 16-bit step is 3.05e-5
    ('pcm24', 1e-7),
    ('pcm32', 1e-9),
    ('float32', 1e-7),
    ('float64', 0.0),       # the only encoding that is lossless
])
def test_every_encoding_round_trips_through_read_wav(tmp_path, encoding,
                                                     tolerance):
    """A file this module writes is not a one-way trip, and the error is the
    encoding's own quantisation rather than anything the pair adds."""
    signal = _tone(peak=0.5)
    path = tmp_path / f'{encoding}.wav'
    write_wav(path, signal, FS, encoding=encoding, normalize=False)

    recovered, rate = read_wav(path)
    assert rate == FS
    assert recovered.shape == signal.shape
    assert np.max(np.abs(recovered - signal)) <= tolerance


def test_a_float_round_trip_preserves_an_absolute_level(tmp_path):
    """The reason the float encodings exist: 12.5 goes out and 12.5 comes
    back, where an integer encoding would have to rescale it."""
    calibrated = 12.5 * _tone(peak=1.0)
    write_wav(tmp_path / 'level.wav', calibrated, FS, encoding='float64')

    recovered, _ = read_wav(tmp_path / 'level.wav')
    assert np.max(np.abs(recovered)) == pytest.approx(12.5)


def test_channels_survive_the_round_trip(tmp_path):
    left = _tone(peak=1.0)
    write_wav(tmp_path / 'stereo.wav', np.stack([left, -left], axis=1), FS,
              encoding='float64')

    recovered, _ = read_wav(tmp_path / 'stereo.wav')
    assert recovered.shape == (left.size, 2)
    assert np.allclose(recovered[:, 0], left)
    assert np.allclose(recovered[:, 1], -left)


def test_metadata_survives_the_round_trip(tmp_path):
    written = {'title': 'probe', 'comment': 'round trip', 'date': '2026-09-10'}
    write_wav(tmp_path / 'tagged.wav', _tone(), FS, metadata=written)

    assert read_wav_metadata(tmp_path / 'tagged.wav') == written


def test_a_file_without_an_info_block_reads_as_no_metadata(tmp_path):
    """Most recorders write none, so this is the common case and not an
    error."""
    write_wav(tmp_path / 'bare.wav', _tone(), FS)

    assert read_wav_metadata(tmp_path / 'bare.wav') == {}


def test_reading_something_that_is_not_a_wav_raises(tmp_path):
    (tmp_path / 'not.wav').write_bytes(b'ID3\x04\x00\x00\x00\x00\x00\x00junk')

    with pytest.raises(FileFormatError, match='not a RIFF/WAVE'):
        read_wav(tmp_path / 'not.wav')


def test_a_truncated_wav_names_the_missing_chunk(tmp_path):
    write_wav(tmp_path / 'whole.wav', _tone(), FS)
    raw = (tmp_path / 'whole.wav').read_bytes()
    # Keep the RIFF/WAVE header and the fmt chunk, drop the samples.
    (tmp_path / 'headless.wav').write_bytes(raw[:raw.index(b'data')])

    with pytest.raises(FileFormatError, match='data chunk missing'):
        read_wav(tmp_path / 'headless.wav')


def test_an_odd_length_chunk_does_not_desynchronise_the_walk(tmp_path):
    """RIFF pads chunks to even length without counting the pad in the
    declared size. A reader that steps by the declared size alone drifts one
    byte and reads the rest of the file as garbage — so this writes an INFO
    value of odd length and then reads the samples that follow it."""
    write_wav(tmp_path / 'odd.wav', _tone(), FS,
              metadata={'title': 'abcd'})     # 4 chars + NUL = 5 bytes, odd

    recovered, rate = read_wav(tmp_path / 'odd.wav')
    assert rate == FS and recovered.size == 960
    assert read_wav_metadata(tmp_path / 'odd.wav') == {'title': 'abcd'}


def _riff(fmt_payload, data_payload):
    """A RIFF/WAVE file from raw ``fmt `` and ``data`` payloads."""
    def chunk(cid, payload):
        return cid + struct.pack('<I', len(payload)) + payload + (
            b'\x00' if len(payload) % 2 else b'')
    body = b'WAVE' + chunk(b'fmt ', fmt_payload) + chunk(b'data', data_payload)
    return b'RIFF' + struct.pack('<I', len(body)) + body


_PCM16_MONO_FMT = struct.pack('<HHIIHH', 1, 1, 8000, 16000, 2, 16)


def test_a_sixteen_byte_fmt_chunk_is_the_shortest_that_reads(tmp_path):
    (tmp_path / 'ok.wav').write_bytes(
        _riff(_PCM16_MONO_FMT, struct.pack('<hh', 1000, -1000)))
    samples, rate = read_wav(tmp_path / 'ok.wav')
    assert rate == 8000.0 and samples.shape == (2,)


@pytest.mark.parametrize('n_bytes', [15, 12, 6, 0])
def test_a_truncated_fmt_chunk_raises_the_typed_error(tmp_path, n_bytes):
    (tmp_path / 'short.wav').write_bytes(
        _riff(_PCM16_MONO_FMT[:n_bytes], struct.pack('<hh', 1000, -1000)))
    with pytest.raises(FileFormatError, match='fmt'):
        read_wav(tmp_path / 'short.wav')


@pytest.mark.parametrize('n_channels, n_bytes', [
    (1, 3),    # one and a half 16-bit samples
    (2, 6),    # three samples: whole samples, but not whole stereo frames
])
def test_a_data_chunk_that_is_not_whole_frames_raises_the_typed_error(
        tmp_path, n_channels, n_bytes):
    fmt = struct.pack('<HHIIHH', 1, n_channels, 8000, 16000 * n_channels,
                      2 * n_channels, 16)
    (tmp_path / 'ragged.wav').write_bytes(_riff(fmt, b'\x01' * n_bytes))
    with pytest.raises(FileFormatError, match='data chunk'):
        read_wav(tmp_path / 'ragged.wav')


def test_a_data_chunk_of_whole_frames_reads(tmp_path):
    fmt = struct.pack('<HHIIHH', 1, 2, 8000, 32000, 4, 16)
    (tmp_path / 'frames.wav').write_bytes(_riff(fmt, b'\x01' * 8))
    samples, _rate = read_wav(tmp_path / 'frames.wav')
    assert samples.shape == (2, 2)


def test_a_missing_path_is_the_arguments_fault(tmp_path):
    with pytest.raises(ConfigurationError, match='no file'):
        read_wav(tmp_path / 'absent.wav')
    with pytest.raises(ConfigurationError, match='no file'):
        read_wav_metadata(tmp_path / 'absent.wav')


def _extensible_fmt(n_channels, rate, bits, sub_tag):
    """A 40-byte WAVE_FORMAT_EXTENSIBLE ``fmt `` payload whose SubFormat GUID
    leads with ``sub_tag``."""
    block = n_channels * bits // 8
    guid_tail = b'\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71'
    return (struct.pack('<HHIIHH', 0xFFFE, n_channels, rate, rate * block,
                        block, bits)
            + struct.pack('<HHI', 22, bits, 0)
            + struct.pack('<H', sub_tag) + guid_tail)


def test_an_extensible_header_reads_through_its_subformat(tmp_path):
    """A 24-bit extensible file (SubFormat = PCM) reads as the plain PCM file
    with the same samples."""
    values = np.array([0, 1000, -1000, 8388607, -8388607])
    data = b''.join(int(v).to_bytes(3, 'little', signed=True) for v in values)
    (tmp_path / 'ext.wav').write_bytes(
        _riff(_extensible_fmt(1, 8000, 24, 1), data))
    samples, rate = read_wav(tmp_path / 'ext.wav')
    assert rate == 8000.0
    assert np.allclose(samples, values / (2.0 ** 23 - 1.0))


def test_an_extensible_header_with_a_compressed_subformat_is_refused(tmp_path):
    (tmp_path / 'adpcm.wav').write_bytes(
        _riff(_extensible_fmt(1, 8000, 16, 2), b'\x00' * 8))
    with pytest.raises(FileFormatError, match='format tag 2'):
        read_wav(tmp_path / 'adpcm.wav')


def test_a_data_chunk_cut_short_warns_and_keeps_the_whole_frames(tmp_path):
    whole = _riff(_PCM16_MONO_FMT, struct.pack('<4h', 1, 2, 3, 4))
    # Drop the last sample and a half: 8 declared bytes, 5 present.
    (tmp_path / 'cut.wav').write_bytes(whole[:-3])
    with pytest.warns(UserWarning, match='declares 8 bytes but the file holds 5'):
        samples, _rate = read_wav(tmp_path / 'cut.wav')
    assert np.allclose(samples * (2.0 ** 15 - 1.0), [1, 2])


def _unfinalised(riff_size, samples):
    """A wav whose header was written before its samples and never
    updated: data size 0, the RIFF size ``riff_size``, samples after it."""
    head = (b'WAVE' + b'fmt ' + struct.pack('<I', len(_PCM16_MONO_FMT))
            + _PCM16_MONO_FMT + b'data' + struct.pack('<I', 0))
    return b'RIFF' + struct.pack('<I', riff_size) + head + samples


@pytest.mark.parametrize('riff_size', [36, 0, 0xFFFFFFFF])
def test_a_header_never_finalised_reads_the_samples_after_it_and_warns(
        tmp_path, riff_size):
    """Data size 0 with samples past the declared RIFF size: the samples
    are read to the end of the file (whole frames), with an IOWarning."""
    path = tmp_path / 'unfinalised.wav'
    path.write_bytes(_unfinalised(riff_size,
                                  struct.pack('<4h', 1, 2, 3, 4) + b'\x07'))
    with pytest.warns(UserWarning, match='declares 0 bytes but 9 follow it'):
        samples, rate = read_wav(path)
    assert rate == 8000.0
    assert np.allclose(samples * (2.0 ** 15 - 1.0), [1, 2, 3, 4])


@pytest.mark.parametrize('after', [b'', b'LIST' + struct.pack('<I', 4)
                                   + b'INFO'])
def test_an_empty_data_chunk_inside_the_riff_size_reads_empty_silently(
        tmp_path, after):
    """A genuinely empty recording: data size 0 and nothing after it but
    chunks the RIFF size covers."""
    head = (b'WAVE' + b'fmt ' + struct.pack('<I', len(_PCM16_MONO_FMT))
            + _PCM16_MONO_FMT + b'data' + struct.pack('<I', 0) + after)
    path = tmp_path / 'empty.wav'
    path.write_bytes(b'RIFF' + struct.pack('<I', len(head)) + head)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        samples, _rate = read_wav(path)
    assert samples.size == 0


def test_a_complete_data_chunk_reads_without_a_warning(tmp_path):
    (tmp_path / 'whole.wav').write_bytes(
        _riff(_PCM16_MONO_FMT, struct.pack('<4h', 1, 2, 3, 4)))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        samples, _rate = read_wav(tmp_path / 'whole.wav')
    assert samples.size == 4


# ── reading a span of frames ────────────────────────────────────────────────


def _three_channels(n=1000):
    """Three distinct channels, so a frame or channel misplaced by the
    excerpt arithmetic shows as a wrong value."""
    t = np.arange(n) / FS
    return np.stack([0.5 * np.sin(2 * np.pi * 700 * t),
                     0.4 * np.cos(2 * np.pi * 1300 * t),
                     np.linspace(-0.9, 0.9, n)], axis=1)


@pytest.mark.parametrize('encoding', sorted(audio_io._ENCODINGS))
def test_an_excerpt_is_those_frames_of_the_whole_read(tmp_path, encoding):
    path = tmp_path / f'{encoding}.wav'
    write_wav(path, _three_channels(), FS, encoding=encoding,
              normalize=False)
    whole, rate = read_wav(path)

    part, part_rate = read_wav(path, start=123, frames=456)
    assert part_rate == rate == FS
    assert np.array_equal(part, whole[123:579])
    assert np.array_equal(read_wav(path, start=900)[0], whole[900:])


@pytest.mark.parametrize('encoding', ['pcm24', 'pcm16', 'float32'])
@pytest.mark.parametrize('block_samples', [1, 5, 7, 3000])
def test_an_excerpt_decodes_the_same_across_block_edges(
        tmp_path, monkeypatch, encoding, block_samples):
    """The samples are decoded block by block; a block of one frame, blocks
    that do not divide the excerpt, and one block larger than it all give
    the frames a single-block read gives."""
    path = tmp_path / 'blocks.wav'
    write_wav(path, _three_channels(), FS, encoding=encoding,
              normalize=False)
    reference = read_wav(path, start=10, frames=101)[0]
    monkeypatch.setattr(audio_io, '_BLOCK_SAMPLES', block_samples)
    assert np.array_equal(read_wav(path, start=10, frames=101)[0], reference)


def test_an_empty_excerpt_keeps_the_channel_axis(tmp_path):
    path = tmp_path / 'stereo.wav'
    write_wav(path, _three_channels()[:, :2], FS)
    assert read_wav(path, start=1000, frames=0)[0].shape == (0, 2)
    assert read_wav(path, start=1000)[0].shape == (0, 2)


@pytest.mark.parametrize('kwargs', [
    dict(start=1000),                 # the end: nothing left, legal
    dict(start=400, frames=600),      # exactly to the end
    dict(start=0, frames=1000),
    dict(start=4.0e2, frames=6.0e2),  # integral floats (a rate times a time)
])
def test_a_span_inside_the_file_reads(tmp_path, kwargs):
    path = tmp_path / 'mono.wav'
    write_wav(path, _tone(n=1000), FS)
    signal, _ = read_wav(path, **kwargs)
    assert signal.size == 1000 - int(kwargs['start'])


@pytest.mark.parametrize('kwargs,match', [
    (dict(start=1001), r"start=1001 is not a whole number of frames in "
                       r"\[0, 1000\]"),
    (dict(start=400, frames=601), r"frames=601 .* \[0, 600\]; 600 frame"),
    (dict(start=-1), r"start=-1"),
    (dict(frames=-1), r"frames=-1"),
    (dict(start=0.5), r"start=0\.5"),
    (dict(start='10'), r"start='10'"),
    (dict(frames=float('nan')), r"frames=nan"),
])
def test_a_span_outside_the_file_is_refused_by_name(tmp_path, kwargs, match):
    path = tmp_path / 'mono.wav'
    write_wav(path, _tone(n=1000), FS)
    with pytest.raises(ConfigurationError, match=match) as raised:
        read_wav(path, **kwargs)
    assert str(raised.value).startswith('read_wav: ')
    assert 'read_wav_info(path).n_frames' in raised.value.remediation


def test_an_excerpt_reads_only_its_own_frames(tmp_path):
    """The memory a read takes follows the frames it returns, not the file:
    1000 frames of a 200 000-frame 24-bit file peak near their own 8 kB of
    float64, and a whole read at its 1.6 MB output plus one decode block
    (7 bytes per sample: the packed bytes and the widened int32 word)."""
    import tracemalloc
    n = 200_000
    path = tmp_path / 'long.wav'
    write_wav(path, np.linspace(-0.9, 0.9, n), FS, encoding='pcm24')

    def peak(**kwargs):
        tracemalloc.start()
        try:
            read_wav(path, **kwargs)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    assert peak(start=150_000, frames=1000) < 64 * 1024
    assert 8 * n <= peak() < 8 * n + 7 * audio_io._BLOCK_SAMPLES + 64 * 1024


# ── the encoding, from the headers ─────────────────────────────────────────


@pytest.mark.parametrize('encoding,bits', [
    ('pcm16', 16), ('pcm24', 24), ('pcm32', 32),
    ('float32', 32), ('float64', 64),
])
def test_read_wav_info_reports_what_write_wav_wrote(tmp_path, encoding,
                                                    bits):
    path = tmp_path / f'{encoding}.wav'
    write_wav(path, _three_channels(n=777), 44100, encoding=encoding)
    info = read_wav_info(path)
    assert isinstance(info, WavInfo)
    assert (info.encoding, info.bits, info.n_channels, info.sample_rate,
            info.n_frames) == (encoding, bits, 3, 44100.0, 777)
    assert info.n_frames == read_wav(path)[0].shape[0]
    assert WavInfo.from_dict(info.to_dict()).to_dict() == info.to_dict()


def test_read_wav_info_reads_an_extensible_header_through_its_subformat(
        tmp_path):
    (tmp_path / 'ext.wav').write_bytes(
        _riff(_extensible_fmt(2, 8000, 24, 1), b'\x00' * 12))
    info = read_wav_info(tmp_path / 'ext.wav')
    assert (info.encoding, info.n_channels, info.n_frames) == ('pcm24', 2, 2)


def test_read_wav_info_counts_the_whole_frames_of_a_file_cut_short(tmp_path):
    whole = _riff(_PCM16_MONO_FMT, struct.pack('<4h', 1, 2, 3, 4))
    (tmp_path / 'cut.wav').write_bytes(whole[:-3])
    with pytest.warns(UserWarning, match=r'^read_wav_info: the data chunk '
                                         r'declares 8 bytes'):
        info = read_wav_info(tmp_path / 'cut.wav')
    assert info.n_frames == 2


def test_read_wav_info_refuses_what_read_wav_refuses(tmp_path):
    (tmp_path / 'not.wav').write_bytes(b'ID3\x04\x00\x00\x00\x00\x00\x00junk')
    with pytest.raises(FileFormatError, match='^read_wav_info: not a RIFF'):
        read_wav_info(tmp_path / 'not.wav')
    with pytest.raises(ConfigurationError, match='^read_wav_info: no file'):
        read_wav_info(tmp_path / 'absent.wav')


def test_the_bit_depth_recovers_the_adc_counts_for_volt_params(tmp_path):
    """The route read_wav_info's docstring gives: read_wav's fractions of
    2**(bits-1) - 1 times that full scale are the integer codes written, and
    ``pressure_from_recording(volt_params=(bits, v_ref))`` divides those
    by 2**(bits-1)."""
    from uacpy.acoustics import pressure_from_recording
    codes = np.array([0, 16384, -16384, 32767, -32768])
    fmt = struct.pack('<HHIIHH', 1, 1, 8000, 16000, 2, 16)
    (tmp_path / 'codes.wav').write_bytes(
        _riff(fmt, struct.pack(f'<{codes.size}h', *codes)))
    info = read_wav_info(tmp_path / 'codes.wav')
    x, _ = read_wav(tmp_path / 'codes.wav')
    counts = x * (2 ** (info.bits - 1) - 1)
    assert np.allclose(counts, codes, rtol=0, atol=1e-9)
    p = pressure_from_recording(counts, sensitivity=0, gain=0,
                                volt_params=(info.bits, 2.0))
    assert np.allclose(p, 1e-6 * 2.0 * codes / 2 ** 15)
