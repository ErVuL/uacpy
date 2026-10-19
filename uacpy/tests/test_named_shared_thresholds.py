"""Recurring thresholds hold their values, and each site reads the name."""
import inspect

import numpy as np

from uacpy.acoustic_signal import bands
from uacpy.comms import equalize
from uacpy.core.acoustics import levels
from uacpy.io import bellhop_writer, env_reader, oases_writer
from uacpy.models.oases import _common as oases_common


def test_the_thresholds_hold_their_values():
    assert bellhop_writer.BELLHOP_SSP_GUARD_RANGE_FACTOR == 1.1
    assert env_reader.BELLHOP_SSP_GUARD_RANGE_FACTOR is \
        bellhop_writer.BELLHOP_SSP_GUARD_RANGE_FACTOR
    assert oases_writer.MIN_SPECTRAL_EXPONENT == 1.5
    assert oases_writer.OASN_LEVEL_DEAD_BAND_DB == 0.01
    assert oases_writer.ISOVELOCITY_RTOL == 1e-6
    assert oases_common.C_LOW_MATCH_RTOL == 1e-6
    assert levels.BAND_EDGE_RTOL == 1e-12
    assert bands.BAND_EDGE_RTOL is levels.BAND_EDGE_RTOL
    assert equalize.LMS_DEFAULT_STEP == 0.01
    assert equalize.RLS_DEFAULT_FORGET == 0.99


def test_the_equalizers_default_to_the_named_values():
    """The DFE's ``step`` is None, derived: an unset LMS step is
    :data:`LMS_DEFAULT_STEP`, so an unset DFE adapts exactly as one given
    that step, on a two-path QPSK record."""
    lms = inspect.signature(equalize.lms_equalizer).parameters
    rls = inspect.signature(equalize.rls_equalizer).parameters
    dfe = inspect.signature(equalize.DFE).parameters
    assert lms['step'].default == equalize.LMS_DEFAULT_STEP
    assert dfe['step'].default is None
    assert rls['forget'].default == equalize.RLS_DEFAULT_FORGET
    qpsk = np.exp(1j * np.pi / 4 * (2 * np.arange(4) + 1))
    symbols = qpsk[np.random.default_rng(0).integers(0, 4, 300)]
    received = np.convolve(symbols, [1.0, 0.45j])[:symbols.size]
    unset = equalize.DFE().equalize(received, qpsk)[0]
    named = equalize.DFE(step=equalize.LMS_DEFAULT_STEP).equalize(
        received, qpsk)[0]
    other = equalize.DFE(step=2 * equalize.LMS_DEFAULT_STEP).equalize(
        received, qpsk)[0]
    np.testing.assert_array_equal(unset, named)
    assert not np.array_equal(unset, other)
