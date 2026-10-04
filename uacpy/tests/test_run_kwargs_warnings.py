"""run() keywords the resolved run mode cannot consume are refused
(``frequencies=`` on a single-frequency mode) or warned about, never
silently dropped — and OASR.run must accept the full polymorphic contract
signature (frequencies / source_waveform / sample_rate / output_duration).
"""

import inspect

import numpy as np
import pytest

import uacpy
import uacpy.models as models
from uacpy.core.run_settings import RunMode
from uacpy.tests.conftest import StageReached
from uacpy.tests.conftest import recorded_warnings


@pytest.fixture
def pekeris():
    env = uacpy.Environment(name='pekeris', bathymetry=100.0, ssp=1500.0)
    source = uacpy.Source(depths=50.0, frequencies=100.0)
    receiver = uacpy.Receiver(depths=[50.0], ranges=[1000.0])
    return env, source, receiver


# (model class name, single-frequency run_mode). ``run()`` refuses
# ``frequencies=`` before the engine launches, so the run is stopped at the
# deck-writing stage to prove the refusal comes first. The stop keeps each
# case binary-free at run time; construction still resolves the
# executable → requires_binary.
_SINGLE_MODE_FREQ_CASES = [
    pytest.param('Kraken', RunMode.COHERENT_TL, id='kraken-coherent_tl'),
    pytest.param('Kraken', RunMode.MODES, id='kraken-modes'),
    pytest.param('Scooter', RunMode.COHERENT_TL, id='scooter-coherent_tl'),
    pytest.param('RAM', RunMode.COHERENT_TL, id='ram-coherent_tl'),
    pytest.param('Bellhop', RunMode.COHERENT_TL, id='bellhop-coherent_tl'),
    pytest.param('Bellhop', RunMode.RAYS, id='bellhop-rays'),
    pytest.param('Bellhop', RunMode.ARRIVALS, id='bellhop-arrivals'),
]


@pytest.mark.requires_binary
@pytest.mark.parametrize('cls_name,run_mode', _SINGLE_MODE_FREQ_CASES)
def test_frequencies_on_a_single_frequency_mode_raises(cls_name, run_mode,
                                                       pekeris, stop_at):
    cls = getattr(models, cls_name)
    model = cls(verbose=False)
    stop_at(cls, 'write')
    env, source, receiver = pekeris
    with pytest.raises(uacpy.ConfigurationError,
                       match=r'frequency from source\.frequencies'):
        model.run(env, source, receiver, run_mode=run_mode,
                  frequencies=np.array([90.0, 100.0, 110.0]))
    # Without the keyword the same call reaches the engine body.
    with pytest.raises(StageReached):
        model.run(env, source, receiver, run_mode=run_mode)


@pytest.mark.requires_binary
@pytest.mark.parametrize('run_mode', [RunMode.BROADBAND,
                                      RunMode.TIME_SERIES])
def test_frequencies_on_a_broadband_mode_reaches_the_engine(run_mode,
                                                           pekeris,
                                                           stop_at):
    from uacpy.models import Kraken
    model = Kraken(verbose=False)
    stop_at(Kraken, 'write')
    env, source, receiver = pekeris
    # TIME_SERIES needs its pulse to get past stage 2.
    pulse = ({} if run_mode == RunMode.BROADBAND else
             dict(source_waveform=np.hanning(32), sample_rate=1000.0))
    with pytest.raises(StageReached):
        model.run(env, source, receiver, run_mode=run_mode,
                  frequencies=np.array([90.0, 100.0, 110.0]), **pulse)


@pytest.mark.requires_oases
def test_oasp_consumes_frequencies_on_coherent_tl(pekeris, stop_at):
    """OASP's solver always runs a sweep that ``frequencies=`` pins, so its
    single-frequency mode takes the keyword."""
    from uacpy.models import OASP
    model = OASP(verbose=False)
    stop_at(OASP, 'write')
    env, source, receiver = pekeris
    with pytest.raises(StageReached):
        model.run(env, source, receiver, run_mode=RunMode.COHERENT_TL,
                  frequencies=np.array([100.0]))


@pytest.mark.requires_binary
def test_sparc_time_series_takes_frequencies_as_its_pulse_band(pekeris,
                                                               stop_at):
    """``frequencies=`` sets the band SPARC marches (RA-WAVE-4), so an
    unpinned band raises no 'ignoring' warning."""
    from uacpy.models import SPARC
    model = SPARC(verbose=False)
    stop_at(SPARC, 'project')
    env, source, receiver = pekeris
    with recorded_warnings() as caught:
        with pytest.raises(StageReached):
            model.run(env, source, receiver, run_mode=RunMode.TIME_SERIES,
                      frequencies=np.array([90.0, 100.0, 110.0]))
    assert not any('ignoring' in str(w.message) for w in caught)


@pytest.mark.requires_binary
def test_sparc_with_a_pinned_band_warns_on_frequencies(pekeris, stop_at):
    """``SPARC(freq_min=, freq_max=)`` pin the band, so the run ignores
    ``frequencies=`` with a warning."""
    from uacpy.models import SPARC
    model = SPARC(verbose=False, freq_min=50.0, freq_max=150.0)
    stop_at(SPARC, 'project')
    env, source, receiver = pekeris
    with pytest.warns(UserWarning, match='ignoring frequencies='):
        with pytest.raises(StageReached):
            model.run(env, source, receiver, run_mode=RunMode.TIME_SERIES,
                      frequencies=np.array([90.0, 100.0, 110.0]))


@pytest.mark.requires_binary
def test_bellhop_coherent_tl_lists_all_ignored_kwargs(pekeris, stop_at):
    from uacpy.models import Bellhop
    model = Bellhop(verbose=False)
    stop_at(Bellhop, 'project')
    env, source, receiver = pekeris
    with pytest.warns(UserWarning,
                      match='ignoring source_waveform=, sample_rate='):
        with pytest.raises(StageReached):
            model.run(env, source, receiver, run_mode=RunMode.COHERENT_TL,
                      source_waveform=np.sin(np.linspace(0, 1, 64)),
                      sample_rate=1000.0)


@pytest.mark.requires_binary
def test_sparc_consumes_the_waveform_kwargs(pekeris, stop_at):
    """SPARC marches a ``source_waveform`` it is handed (pulse_type 'FN+B'),
    so the waveform kwargs raise no 'ignoring' warning."""
    from uacpy.models import SPARC
    model = SPARC(verbose=False)
    stop_at(SPARC, 'project')
    env, source, receiver = pekeris
    with recorded_warnings() as caught:
        with pytest.raises(StageReached):
            model.run(env, source, receiver,
                      source_waveform=np.zeros(16), sample_rate=1000.0)
    assert not any('ignoring source_waveform=' in str(w.message)
                   for w in caught)


@pytest.mark.requires_binary
def test_bellhop_broadband_consumes_frequencies_no_warning(pekeris,
                                                           stop_at):
    """BROADBAND consumes frequencies= — the consuming path must stay free
    of the 'ignoring' warning (other UserWarnings may legitimately fire)."""
    from uacpy.models import Bellhop
    model = Bellhop(verbose=False)
    stop_at(Bellhop, 'write')
    env, source, receiver = pekeris
    with recorded_warnings() as caught:
        with pytest.raises(StageReached):
            model.run(env, source, receiver, run_mode=RunMode.BROADBAND,
                      frequencies=np.array([90.0, 100.0, 110.0]))
    assert not any('ignoring' in str(w.message) for w in caught)


# (model class name, stage the run stops at after the BROADBAND warn
# checkpoint). Bellhop runs its arrivals and stops as it builds H(f) from
# them; the IFFT synthesizers stop after stage 2, which warns.
_BROADBAND_TIME_AXIS_CASES = [
    pytest.param('Bellhop', 'result', id='bellhop'),
    pytest.param('Scooter', 'project', id='scooter'),
    pytest.param('RAM', 'project', id='ram'),
]


@pytest.mark.requires_binary
@pytest.mark.parametrize('cls_name,stage', _BROADBAND_TIME_AXIS_CASES)
def test_broadband_warns_on_time_axis_kwargs(cls_name, stage, pekeris,
                                             stop_at):
    """BROADBAND returns H(f) and builds no time axis, so ``sample_rate=`` /
    ``output_duration=`` are consumed by nothing and must warn."""
    cls = getattr(models, cls_name)
    model = cls(verbose=False)
    stop_at(cls, stage)
    env, source, receiver = pekeris
    with pytest.warns(UserWarning,
                      match='ignoring sample_rate=, output_duration='):
        with pytest.raises(StageReached):
            model.run(env, source, receiver, run_mode=RunMode.BROADBAND,
                      sample_rate=4000.0, output_duration=1.0)


@pytest.mark.requires_binary
def test_time_series_consumes_time_axis_kwargs_no_warning(pekeris,
                                                          stop_at):
    """Dual of the above: TIME_SERIES does build the time axis from them, so
    the consuming path must stay free of the 'ignoring' warning."""
    from uacpy.models import Scooter
    model = Scooter(verbose=False)
    stop_at(Scooter, 'project')
    env, source, receiver = pekeris
    with recorded_warnings() as caught:
        with pytest.raises(StageReached):
            model.run(env, source, receiver, run_mode=RunMode.TIME_SERIES,
                      source_waveform=np.sin(np.linspace(0, 20, 512)),
                      sample_rate=4000.0, output_duration=0.2)
    assert not any('ignoring' in str(w.message) for w in caught)


# The IFFT synthesizers: stage 2 (``_check_time_series_keywords``) warns
# once on every mode other than TIME_SERIES. Exactly one warning either way.
_COHERENT_TL_TIME_AXIS_CASES = [
    pytest.param('RAM', 'project', id='ram'),
    pytest.param('Scooter', 'project', id='scooter'),
]


@pytest.mark.requires_binary
@pytest.mark.parametrize('cls_name,stage', _COHERENT_TL_TIME_AXIS_CASES)
def test_coherent_tl_warns_once_on_time_axis_kwargs(cls_name, stage, pekeris,
                                                    stop_at):
    """COHERENT_TL consumes neither keyword, so it warns — once, however
    many stages see the keywords."""
    cls = getattr(models, cls_name)
    model = cls(verbose=False)
    stop_at(cls, stage)
    env, source, receiver = pekeris
    with recorded_warnings() as caught:
        with pytest.raises(StageReached):
            model.run(env, source, receiver, run_mode=RunMode.COHERENT_TL,
                      sample_rate=4000.0, output_duration=1.0)
    hits = [w for w in caught
            if 'ignoring sample_rate=, output_duration=' in str(w.message)]
    assert len(hits) == 1, f'{cls_name}: expected 1 warning, got {len(hits)}'


@pytest.mark.requires_oases
def test_oasp_broadband_warns_once_on_time_axis_kwargs(pekeris, stop_at):
    """OASP reaches the same checkpoint from the OASES side."""
    from uacpy.models.oases import OASP
    model = OASP(verbose=False)
    stop_at(OASP, 'project')
    env, source, receiver = pekeris
    with recorded_warnings() as caught:
        with pytest.raises(StageReached):
            model.run(env, source, receiver, run_mode=RunMode.BROADBAND,
                      sample_rate=4000.0, output_duration=1.0)
    hits = [w for w in caught
            if 'ignoring sample_rate=, output_duration=' in str(w.message)]
    assert len(hits) == 1, f'expected 1 warning, got {len(hits)}'


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
def test_oasr_refuses_waveform_kwargs_before_projecting(pekeris,
                                                         stop_at):
    """OASR has no TIME_SERIES mode, so no run of it reads a waveform: the
    keywords are refused by the checking stage, before the environment is
    projected (RA-CONTRACT-3)."""
    from uacpy.models.oases import OASR
    model = OASR(verbose=False)
    stop_at(OASR, 'project')
    env, source, receiver = pekeris
    with pytest.raises(uacpy.UnsupportedFeatureError,
                       match='sample_rate, source_waveform'):
        model.run(env, source, receiver,
                  source_waveform=np.zeros(16), sample_rate=1000.0)


def _hann_pulse(fs=4000.0, n=400, f0=100.0):
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * f0 * t) * np.hanning(n), fs


@pytest.mark.requires_binary
def test_bellhop_time_series_drops_frequencies_with_one_warning(pekeris):
    """Bellhop's TIME_SERIES labels its delay-and-sum trace with the grid the
    pulse implies, so ``frequencies=`` is dropped with a warning naming why,
    and the call resolves exactly as it does without it; without
    ``frequencies=`` that warning is not given."""
    from uacpy.models import Bellhop
    env, source, receiver = pekeris
    pulse, fs = _hann_pulse()
    model = Bellhop(verbose=False)
    kw = dict(run_mode=RunMode.TIME_SERIES, source_waveform=pulse,
              sample_rate=fs)

    def said(**extra):
        with recorded_warnings() as caught:
            settings = model.run_settings(env, source, receiver, **kw,
                                          **extra)
        return settings, [str(w.message) for w in caught
                          if 'frequencies= is ignored' in str(w.message)]

    given, warned = said(frequencies=[50.0, 150.0])
    plain, quiet = said()
    assert warned == [
        "Bellhop.run(run_mode=TIME_SERIES) synthesises p(t) by delay-and-sum "
        "from a single arrivals run at fc; the supplied frequencies= is "
        "ignored. Use run_mode=BROADBAND for an explicit H(f) grid."]
    assert quiet == []
    np.testing.assert_array_equal(given.frequencies, plain.frequencies)
    assert given.engine.pulse_samples == plain.engine.pulse_samples == 400


@pytest.mark.requires_binary
def test_bellhop_broadband_records_where_its_band_came_from(pekeris):
    """The origin of a BROADBAND grid is read off the call itself: the
    ``frequencies=`` it passed, a Source listing a band, or the carrier
    expanded by the band knobs."""
    from uacpy.models import Bellhop
    env, source, receiver = pekeris
    model = Bellhop(verbose=False)

    def origin(src, **kw):
        return model.run_settings(env, src, receiver,
                                  run_mode=RunMode.BROADBAND,
                                  **kw).engine.band_origin

    band = uacpy.Source(depths=50.0, frequencies=[90.0, 110.0])
    assert origin(source, frequencies=[90.0, 100.0, 110.0]) == 'frequencies='
    assert origin(band) == 'Source.frequencies'
    assert origin(source) == (f"{model.n_freqs} bins over fc*(1 +/- "
                              f"{model.bandwidth_factor:g}/2)")
