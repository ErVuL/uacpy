"""Tests for ``uacpy.io.write_wav``.

The file it writes is read back with ``scipy.io.wavfile`` and the stdlib
``wave`` module — two independent parsers, so a header this writer builds by
hand cannot pass by agreeing only with itself.
"""

import wave
import warnings

import numpy as np
import pytest
from scipy.io import wavfile

from uacpy.core.exceptions import ConfigurationError
from uacpy.io import write_wav


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
])
def test_rejected_inputs(tmp_path, kwargs, signal, match):
    data = _tone() if signal is None else signal
    with pytest.raises(ConfigurationError, match=match):
        write_wav(tmp_path / 'rejected.wav', data, FS, **kwargs)


@pytest.mark.parametrize('rate', [0, -48000])
def test_a_non_positive_sample_rate_raises(tmp_path, rate):
    with pytest.raises(ConfigurationError, match='not positive'):
        write_wav(tmp_path / 'rate.wav', _tone(), rate)


def test_the_written_file_is_byte_even(tmp_path):
    """RIFF chunks are word-aligned: an odd-length data chunk carries a pad
    byte that the declared size does not count."""
    write_wav(tmp_path / 'odd.wav', np.zeros(3), FS, encoding='pcm24')

    raw = (tmp_path / 'odd.wav').read_bytes()
    assert len(raw) % 2 == 0
    # 3 frames x 3 bytes = 9 declared, 10 written.
    assert raw[raw.index(b'data') + 4:raw.index(b'data') + 8] == (
        (9).to_bytes(4, 'little'))
