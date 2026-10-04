"""The signal and comms defaults hold their values, and each function
reads its name."""
import inspect

import numpy as np

from uacpy.acoustic_signal import _synthesis, generate, spectrum_at
from uacpy.comms import equalize, sync, transceiver
from uacpy.core._validate import UNIFORM_STEP_RTOL
from uacpy.core.results import _base as results_base


def test_the_constants_hold_their_values():
    assert generate.MSEQ_PROBE_LEAD_TIME_S == 0.2
    assert generate.NOISE_SYNTH_MIN_NFFT == 16
    assert generate.NOISE_SYNTH_DEFAULT_NFFT == 2 ** 16
    assert generate.NOISE_SYNTH_MAX_NFFT == 2 ** 18
    assert transceiver.DEFAULT_PREAMBLE_SYMBOLS == 64
    assert (sync.DOPPLER_SCALE_MAX, sync.DOPPLER_SCALE_COUNT) == (5e-3, 601)
    assert equalize.RLS_INIT_DELTA == 1e-2
    assert _synthesis._SCRATCH_BLOCK_ELEMS == 4_000_000
    assert spectrum_at._SCRATCH_BLOCK_ELEMS is _synthesis._SCRATCH_BLOCK_ELEMS
    assert _synthesis._ASSUMED_PATH_SPEED_SPREAD == 0.05
    assert UNIFORM_STEP_RTOL == 1e-6
    assert results_base._AXIS_MATCH_RTOL == 1e-6


def test_the_noise_synthesiser_defaults_to_its_named_nfft():
    nfft = inspect.signature(generate.synthesize_noise_from_psd).parameters[
        'nfft']
    assert nfft.default == generate.NOISE_SYNTH_DEFAULT_NFFT


def test_the_probe_opens_with_its_named_lead_in():
    probe = generate.make_mseq_probe(1000, 2000, sample_rate=10000,
                                     duration=10.0)
    lead = int(generate.MSEQ_PROBE_LEAD_TIME_S * 10000)
    assert not np.any(probe[:lead])
    assert np.any(probe[lead:lead + 100])


def test_the_preamble_has_its_named_length():
    symbols = transceiver._preamble_symbols(None, 'qpsk')
    assert len(symbols) == transceiver.DEFAULT_PREAMBLE_SYMBOLS
