# JANUS waveforms from the CMRE reference implementation

Fixtures for `test_comms.py::TestJanusInteroperatesWithTheCmreReference`. `janus_cmre.json` holds each file's
generating command (`how`) and the 64 packet bits janus-tx printed for it.

- `cmre_a_clean.wav`, `cmre_b_clean.wav`: janus-tx's own output bytes (CMRE janus-c 3.0.5, `--pset-id 1`, the
  initial band 11 520 Hz / 4160 Hz, `--stream-fs 48000`, S16, default amplitude 0.95).
- `cmre_c_doppler.wav`: janus-tx's own bytes at `--stream-fs 47904` (the header says 47904). Read at 48 000 Hz, it
  is the waveform compressed by the Doppler scale `a = 48000/47904 - 1 = 2.004e-3` (+3.006 m/s at c = 1500 m/s).
- `cmre_a_noisy.wav`: `cmre_a_clean.wav` plus `numpy.random.default_rng(124)` white noise at 0 dB SNR over the
  4160 Hz band (signal power over the 176-chip frame), written as 16-bit PCM without normalisation.

The `uacpy_to_cmre` entries record the other direction. janus-rx decoded uacpy's `janus_modulate` output for
packets A and B (0.95 full scale, rounded to int16) back to the same bits with a valid CRC, and each entry keeps that
int16 waveform's SHA-256, so the test can pin that the waveform janus-rx decoded is still the one uacpy emits.
