"""Every OASES uniformity test reads the one tolerance: an axis
jittered between the default and a looser constant is refused at 1e-6 and
accepted once the constant is loosened."""
import warnings

import numpy as np
import pytest

import uacpy
from uacpy.core import _validate
from uacpy.core.exceptions import ConfigurationError
from uacpy.io import oases_writer
from uacpy.models.oases import _sampling

JITTER = 1e-4      # between UNIFORM_STEP_RTOL (1e-6) and the loosened 1e-3


def _jittered(start, step, n):
    axis = start + step * np.arange(n, dtype=float)
    axis[2] += JITTER * step
    return axis


def _freq_sweep():
    src = uacpy.Source(depths=[50.0], frequencies=_jittered(100.0, 10.0, 6))
    oases_writer._resolve_freq_sweep('w', src, 100.0)


def _range_axis():
    rx = uacpy.Receiver(depths=[50.0], ranges=_jittered(1000.0, 100.0, 6))
    oases_writer._oasp_range_axis('w', rx, None)


def _linear_sweep_notice():
    *_, notice = _sampling._oases_frequency_sweep(
        _jittered(100.0, 10.0, 6), 'OASR')
    if notice is not None:
        raise ConfigurationError(notice)


def _log_sweep_notice():
    freqs = np.geomspace(100.0, 1000.0, 6)
    freqs[2] *= 1.0 + JITTER
    *_, notice = _sampling._oases_frequency_sweep(freqs, 'OASR',
                                                  log_spaced=True)
    if notice is not None:
        raise ConfigurationError(notice)


def _angles(tmp_path):
    env = uacpy.Environment(name='u', bathymetry=100.0, ssp=1500.0)
    oases_writer.write_oasr_input(
        tmp_path / 'r.dat', env, uacpy.Source(depths=[10.0], frequencies=100.0),
        uacpy.Receiver(depths=[10.0], ranges=[100.0]),
        angles=_jittered(10.0, 5.0, 6))


SITES = {'frequency sweep': _freq_sweep, 'range axis': _range_axis,
         'linear sweep notice': _linear_sweep_notice,
         'log sweep notice': _log_sweep_notice, 'OASR angles': _angles}
#: What each site's refusal or notice says about the axis.
REFUSAL = {'frequency sweep': 'frequencies is not uniformly spaced',
           'range axis': 'ranges is not uniformly spaced',
           'linear sweep notice': 'non-equispaced',
           'log sweep notice': 'not log-spaced',
           'OASR angles': 'angles`` is not uniformly spaced'}


def _call(site, tmp_path):
    fn = SITES[site]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(tmp_path) if site == 'OASR angles' else fn()


def test_the_tolerance_is_one_part_in_a_million():
    assert _validate.UNIFORM_STEP_RTOL == 1e-6


@pytest.mark.parametrize('site', sorted(SITES))
def test_each_site_follows_the_one_tolerance(site, tmp_path, monkeypatch):
    with pytest.raises(ConfigurationError, match=REFUSAL[site]):
        _call(site, tmp_path)
    monkeypatch.setattr(_validate, 'UNIFORM_STEP_RTOL', 1e-3)
    _call(site, tmp_path)
