"""Recurring thresholds hold their values, and each site reads the name."""
import inspect

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
    assert oases_common.MIN_SPECTRAL_EXPONENT is \
        oases_writer.MIN_SPECTRAL_EXPONENT
    assert oases_writer.OASN_LEVEL_DEAD_BAND_DB == 0.01
    assert oases_writer.ISOVELOCITY_RTOL == 1e-6
    assert oases_common.C_LOW_MATCH_RTOL == 1e-6
    assert levels.BAND_EDGE_RTOL == 1e-12
    assert bands.BAND_EDGE_RTOL is levels.BAND_EDGE_RTOL
    assert equalize.LMS_DEFAULT_STEP == 0.01
    assert equalize.RLS_DEFAULT_FORGET == 0.99


def test_the_equalizers_default_to_the_named_values():
    lms = inspect.signature(equalize.lms_equalizer).parameters
    rls = inspect.signature(equalize.rls_equalizer).parameters
    dfe = inspect.signature(equalize.DFE).parameters
    assert lms['step'].default == equalize.LMS_DEFAULT_STEP
    assert dfe['step'].default == equalize.LMS_DEFAULT_STEP
    assert rls['forget'].default == equalize.RLS_DEFAULT_FORGET
