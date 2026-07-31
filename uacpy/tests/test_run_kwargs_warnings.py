"""run() keywords the resolved run mode cannot consume must warn, not be
silently dropped — and OASR.run must accept the full polymorphic contract
signature (frequencies / source_waveform / sample_rate / output_duration).
"""

import inspect
import warnings

import numpy as np
import pytest

import uacpy
import uacpy.models as models
from uacpy.models.base import RunMode


class _ShortCircuit(Exception):
    """Sentinel raised to stop run() right after the warn checkpoint."""


def _boom(*args, **kwargs):
    raise _ShortCircuit


@pytest.fixture
def pekeris():
    env = uacpy.Environment(name='pekeris', bathymetry=100.0, ssp=1500.0)
    source = uacpy.Source(depths=50.0, frequencies=100.0)
    receiver = uacpy.Receiver(depths=[50.0], ranges=[1000.0])
    return env, source, receiver


# (model class name, explicit run_mode, method to short-circuit after the
# warn checkpoint). The short-circuit keeps each case binary-free at run
# time; construction still resolves the executable → requires_binary.
_IGNORED_FREQ_CASES = [
    pytest.param('Kraken', RunMode.COHERENT_TL, '_select_kraken_exe',
                 id='kraken-coherent_tl'),
    pytest.param('Kraken', RunMode.MODES, '_select_kraken_exe',
                 id='kraken-modes'),
    pytest.param('Scooter', RunMode.COHERENT_TL, '_project_environment',
                 id='scooter-coherent_tl'),
    pytest.param('SPARC', RunMode.TIME_SERIES, '_project_environment',
                 id='sparc-time_series'),
    pytest.param('Bellhop', RunMode.COHERENT_TL, '_project_environment',
                 id='bellhop-coherent_tl'),
    pytest.param('Bellhop', RunMode.RAYS, '_project_environment',
                 id='bellhop-rays'),
]


@pytest.mark.requires_binary
@pytest.mark.parametrize('cls_name,run_mode,cutoff', _IGNORED_FREQ_CASES)
def test_unconsumed_frequencies_warns(cls_name, run_mode, cutoff,
                                      pekeris, monkeypatch):
    cls = getattr(models, cls_name)
    model = cls(verbose=False)
    monkeypatch.setattr(cls, cutoff, _boom)
    env, source, receiver = pekeris
    with pytest.warns(UserWarning, match='ignoring frequencies='):
        with pytest.raises(_ShortCircuit):
            model.run(env, source, receiver, run_mode=run_mode,
                      frequencies=np.array([90.0, 100.0, 110.0]))


@pytest.mark.requires_binary
def test_bellhop_coherent_tl_lists_all_ignored_kwargs(pekeris, monkeypatch):
    from uacpy.models import Bellhop
    model = Bellhop(verbose=False)
    monkeypatch.setattr(Bellhop, '_project_environment', _boom)
    env, source, receiver = pekeris
    with pytest.warns(UserWarning,
                      match='ignoring frequencies=, source_waveform='):
        with pytest.raises(_ShortCircuit):
            model.run(env, source, receiver, run_mode=RunMode.COHERENT_TL,
                      frequencies=np.array([90.0, 110.0]),
                      source_waveform=np.sin(np.linspace(0, 1, 64)))


@pytest.mark.requires_binary
def test_sparc_waveform_kwargs_still_warn(pekeris, monkeypatch):
    """SPARC drives its pulse via the constructor pulse_type; the waveform
    kwargs must keep warning through the shared helper."""
    from uacpy.models import SPARC
    model = SPARC(verbose=False)
    monkeypatch.setattr(SPARC, '_project_environment', _boom)
    env, source, receiver = pekeris
    with pytest.warns(UserWarning, match='ignoring source_waveform='):
        with pytest.raises(_ShortCircuit):
            model.run(env, source, receiver,
                      source_waveform=np.zeros(16), sample_rate=1000.0)


@pytest.mark.requires_binary
def test_bellhop_broadband_consumes_frequencies_no_warning(pekeris,
                                                           monkeypatch):
    """BROADBAND consumes frequencies= — the consuming path must stay free
    of the 'ignoring' warning (other UserWarnings may legitimately fire)."""
    from uacpy.models import Bellhop
    model = Bellhop(verbose=False)
    monkeypatch.setattr(Bellhop, '_run_broadband', _boom)
    env, source, receiver = pekeris
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(_ShortCircuit):
            model.run(env, source, receiver, run_mode=RunMode.BROADBAND,
                      frequencies=np.array([90.0, 100.0, 110.0]))
    assert not any('ignoring' in str(w.message) for w in caught)


def test_oasr_run_signature_accepts_full_contract():
    """A polymorphic driver passing the base-contract extras keyword-
    explicitly must not hit TypeError on OASR (signature check only —
    needs no binary)."""
    from uacpy.models.oases import OASR
    params = inspect.signature(OASR.run).parameters
    for name in ('frequencies', 'source_waveform', 'sample_rate',
                 'output_duration'):
        assert name in params, f'OASR.run() lacks {name}'
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[name].default is None


@pytest.mark.requires_oases
def test_oasr_warns_on_waveform_kwargs(pekeris, monkeypatch):
    from uacpy.models.oases import OASR
    model = OASR(verbose=False)
    monkeypatch.setattr(OASR, '_project_environment', _boom)
    env, source, receiver = pekeris
    with pytest.warns(UserWarning, match='ignoring source_waveform='):
        with pytest.raises(_ShortCircuit):
            model.run(env, source, receiver,
                      source_waveform=np.zeros(16), sample_rate=1000.0)
