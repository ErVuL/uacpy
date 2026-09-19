"""A multi-depth ``Source`` runs once per depth on every field model and the
slabs add coherently through ``ResultStack.superpose``.

Two sources are driven by adding the complex field of each one: the engines
are linear in the source amplitude, so ``Σ wᵢ·pᵢ`` over unit-source slabs is
the field of the weighted array. The tests here pin (a) that the per-depth
loop changes nothing per slab, (b) that ``superpose`` is that linear sum,
(c) that the sum behaves like a field of two sources, (d) the time-domain
case, and (e) the guards on weights and on stacks that cannot add.
"""

import numpy as np
import pytest

import uacpy
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Field, ResultStack
from uacpy.models import RunMode
from uacpy.tests.conftest import make_pekeris

# 100 m Pekeris guide at 100 Hz on a 3-depth × 20-range receiver grid: the
# smallest grid on which a Kraken, Scooter, RAM, OAST or Bellhop field is
# still a field.
F0 = 100.0
GUIDE_DEPTH = 100.0


def _env():
    return make_pekeris(name='multi-source', bathymetry=GUIDE_DEPTH)


def _receiver():
    return uacpy.Receiver(depths=[20.0, 50.0, 80.0],
                          ranges=np.linspace(100.0, 2000.0, 20))


def _kraken():
    from uacpy.models.kraken import Kraken
    return Kraken(verbose=False)


def _scooter():
    from uacpy.models.scooter import Scooter
    return Scooter(verbose=False)


def _ram():
    from uacpy.models.ram import RAM
    return RAM(verbose=False)


def _oast():
    from uacpy.models.oases import OAST
    return OAST(verbose=False)


def _bellhop():
    from uacpy.models.bellhop import Bellhop
    return Bellhop(verbose=False)


# (factory, slab tolerance). The looped engines write the same deck for a
# slab and for its stand-alone run, so the two agree to round-off. Bellhop
# writes both depths into one deck and its ``.shd`` is complex64, and the
# parallel beam sum lands in a different order per run, so its slabs agree
# to float32 round-off only.
_FIELD_MODELS = [
    pytest.param(_bellhop, 1e-5, id='Bellhop',
                 marks=pytest.mark.requires_binary),
    pytest.param(_kraken, 1e-9, id='Kraken',
                 marks=pytest.mark.requires_binary),
    pytest.param(_scooter, 1e-9, id='Scooter',
                 marks=pytest.mark.requires_binary),
    pytest.param(_ram, 1e-9, id='RAM', marks=pytest.mark.requires_binary),
    pytest.param(_oast, 1e-9, id='OAST',
                 marks=[pytest.mark.requires_binary,
                        pytest.mark.requires_oases]),
]


def _relative_error(actual, reference):
    """Largest ``|actual - reference| / |reference|`` over the cells where
    the reference is finite and non-zero."""
    ref = np.asarray(reference)
    ok = np.isfinite(ref) & (ref != 0)
    assert ok.any(), "the reference field is empty"
    return float(np.max(np.abs(np.asarray(actual)[ok] - ref[ok])
                        / np.abs(ref[ok])))


def _norm_relative_error(actual, reference):
    """``‖actual - reference‖ / ‖reference‖`` over the finite cells."""
    ref = np.asarray(reference)
    act = np.asarray(actual)
    ok = np.isfinite(ref) & np.isfinite(act)
    return float(np.linalg.norm(act[ok] - ref[ok]) / np.linalg.norm(ref[ok]))


@pytest.fixture(scope='module')
def kraken_stack():
    """One Kraken run over ``Source(depths=[30, 70], weights=[1, -1])``,
    shared by the linearity and guard tests below."""
    pytest.importorskip('uacpy.models.kraken')
    from uacpy.core.exceptions import ExecutableNotFoundError
    try:
        model = _kraken()
    except ExecutableNotFoundError:
        pytest.skip("Kraken binary not available")
    source = uacpy.Source(depths=[30.0, 70.0], frequencies=F0,
                          weights=[1.0, -1.0])
    return model.run(_env(), source, _receiver())


# ── (a) the loop changes nothing per slab ───────────────────────────────


@pytest.mark.parametrize('factory,tolerance', _FIELD_MODELS)
def test_every_field_model_stacks_a_two_depth_source(factory, tolerance):
    """Each slab of the stack is the single-source run at that depth: the
    same environment, receiver and mode go to every depth, so nothing but
    the source depth differs between a slab and its stand-alone run."""
    model = factory()
    env, receiver = _env(), _receiver()
    depths = [30.0, 70.0]
    stack = model.run(env, uacpy.Source(depths=depths, frequencies=F0),
                      receiver)
    assert isinstance(stack, ResultStack)
    assert stack.n_slabs == 2
    assert stack.slab_type is Field
    assert stack.coordinate_name == 'source_depth'
    np.testing.assert_array_equal(stack.coordinate, depths)
    np.testing.assert_array_equal(stack.metadata['source_weights'],
                                  [1.0, 1.0])
    for i, z in enumerate(depths):
        single = model.run(env, uacpy.Source(depths=z, frequencies=F0),
                           receiver)
        assert isinstance(single, Field)
        assert stack[i].data.shape == single.data.shape
        assert stack[i].is_complex == single.is_complex
        assert _relative_error(stack[i].data, single.data) < tolerance
        np.testing.assert_array_equal(stack[i].source_depths, [z])


@pytest.mark.requires_binary
def test_a_single_depth_source_returns_a_field_with_no_weight_stamp():
    """The loop only fires above one depth: the single-source contract is a
    plain ``Field`` with no weight stamp, and an explicit unit weight is
    the same run."""
    model = _kraken()
    field = model.run(_env(), uacpy.Source(depths=50.0, frequencies=F0),
                      _receiver())
    assert isinstance(field, Field)
    assert 'source_weights' not in field.metadata
    assert 'superposed_sources' not in field.metadata
    unit = model.run(_env(), uacpy.Source(depths=50.0, frequencies=F0,
                                          weights=1.0), _receiver())
    np.testing.assert_array_equal(unit.data, field.data)
    assert 'superposed_sources' not in unit.metadata


@pytest.mark.requires_binary
def test_a_single_depth_source_applies_its_weight():
    """One source is the ``n = 1`` case of the same rule: ``weights=2``
    scales the field by 2, ``weights=1j`` rotates it, and the metadata
    records the one depth and weight."""
    model = _kraken()
    unit = model.run(_env(), uacpy.Source(depths=50.0, frequencies=F0),
                     _receiver())
    for w in (2.0, 1j, -0.5 + 0.25j):
        scaled = model.run(_env(), uacpy.Source(depths=50.0, frequencies=F0,
                                                weights=w), _receiver())
        assert isinstance(scaled, Field)
        assert _relative_error(scaled.data, w * unit.data) < 1e-12
        assert scaled.metadata['superposed_sources'] == {
            'depths': [50.0], 'weights': [complex(w)]}


def test_every_concrete_run_is_wrapped_by_the_source_depth_loop():
    """``__init_subclass__`` applies the loop to each concrete ``run`` and
    keeps its declared signature visible."""
    import inspect
    from uacpy.models.bellhop import Bellhop
    from uacpy.models.kraken import Kraken
    from uacpy.models.ram import RAM
    for cls in (Bellhop, Kraken, RAM):
        assert hasattr(cls.run, '__wrapped__'), cls
        names = list(inspect.signature(cls.run).parameters)
        assert names[:5] == ['self', 'env', 'source', 'receiver', 'run_mode']


# ── (b) superpose is the linear sum ─────────────────────────────────────


def test_superpose_is_the_weighted_sum_of_the_slabs(kraken_stack):
    """``superpose([w1, w2])`` is ``w1·p1 + w2·p2`` cell by cell, for real
    and complex weights, and ``[1, 0]`` returns slab 0 unchanged."""
    p1, p2 = kraken_stack[0].data, kraken_stack[1].data
    for w1, w2 in ((1.0, 1.0), (0.3, -2.0), (1.0, 1j), (0.5 + 0.5j, -1j)):
        summed = kraken_stack.superpose([w1, w2])
        assert isinstance(summed, Field)
        assert summed.is_complex
        assert _relative_error(summed.data, w1 * p1 + w2 * p2) < 1e-12
    only_first = kraken_stack.superpose([1.0, 0.0])
    np.testing.assert_array_equal(only_first.data, p1)


def test_superpose_reads_the_source_weights_by_default(kraken_stack):
    """With no argument the stack sums with the weights the ``Source``
    carried, here ``[1, -1]``: the same result as applying them by hand."""
    np.testing.assert_array_equal(kraken_stack.metadata['source_weights'],
                                  [1.0, -1.0])
    by_default = kraken_stack.superpose()
    by_hand = kraken_stack[0].data - kraken_stack[1].data
    assert _relative_error(by_default.data, by_hand) < 1e-12


def test_the_superposed_field_carries_the_stack_identity(kraken_stack):
    """The sum keeps the slabs' grid, phase reference and model, widens
    ``source_depths`` to every depth summed, and records what was added."""
    summed = kraken_stack.superpose([1.0, 1.0])
    first = kraken_stack[0]
    assert list(summed.coords) == list(first.coords)
    for axis in first.coords:
        np.testing.assert_array_equal(summed.coords[axis], first.coords[axis])
    assert summed.phase_reference == first.phase_reference
    assert summed.phase_reference is not None
    assert summed.model == first.model
    np.testing.assert_array_equal(summed.source_depths, [30.0, 70.0])
    assert summed.metadata['superposed_sources'] == {
        'depths': [30.0, 70.0], 'weights': [1.0 + 0j, 1.0 + 0j]}
    assert 'source_weights' not in summed.metadata
    assert 'source_depth' not in summed.pinned
    # The sum is complex pressure, so its dB view is defined.
    assert np.isfinite(summed.dB).any()


# ── (c) the sum behaves like the field of two sources ───────────────────


@pytest.mark.requires_binary
def test_two_sources_straddling_one_depth_add_to_twice_the_single_field():
    """In-phase sources at ``50 ± Δ`` m sum to ``2·p(50)`` up to a term of
    order ``(k_z·Δ)²``; at 100 Hz and Δ = 0.05 m that is below 1e-3. The
    antiphase pair is small but not zero — the odd combination is of order
    ``k_z·Δ`` — which pins that the two slabs really are different fields."""
    model = _kraken()
    env, receiver = _env(), _receiver()
    delta = 0.05
    pair = model.run(env, uacpy.Source(depths=[50.0 - delta, 50.0 + delta],
                                       frequencies=F0), receiver)
    single = model.run(env, uacpy.Source(depths=50.0, frequencies=F0),
                       receiver)
    even = pair.superpose([1.0, 1.0])
    assert _norm_relative_error(even.data, 2.0 * single.data) < 1e-3
    odd = pair.superpose([1.0, -1.0])
    odd_over_even = (np.linalg.norm(np.nan_to_num(odd.data))
                     / np.linalg.norm(np.nan_to_num(even.data)))
    assert 0.0 < odd_over_even < 0.05


@pytest.mark.requires_binary
def test_an_antiphase_pair_from_the_source_matches_the_hand_sum():
    """``Source(weights=[1, -1])`` at 30 m and 70 m: ``run().superpose()``
    is ``p(30) - p(70)`` built from two single-source runs, so the weights
    live in the sum and never reach the engine."""
    model = _kraken()
    env, receiver = _env(), _receiver()
    source = uacpy.Source(depths=[30.0, 70.0], frequencies=F0,
                          weights=[1.0, -1.0])
    summed = model.run(env, source, receiver).superpose()
    by_hand = (model.run(env, uacpy.Source(depths=30.0, frequencies=F0),
                         receiver).data
               - model.run(env, uacpy.Source(depths=70.0, frequencies=F0),
                           receiver).data)
    assert _relative_error(summed.data, by_hand) < 1e-9


# ── (d) time domain ─────────────────────────────────────────────────────


def _time_slab(z, trace):
    """A real time-domain slab: one receiver point, ``trace`` samples."""
    return Field(
        data=np.asarray(trace, dtype=float).reshape(1, 1, -1),
        coords={'depth': [50.0], 'range': [1000.0],
                'time': np.arange(len(trace)) / 1000.0},
        model='Test', frequencies=[90.0, 100.0, 110.0],
        source_depths=[z], phase_reference=None,
        metadata={'source_weights': np.array([1.0, -1.0])},
    )


def test_a_time_series_stack_superposes_to_the_summed_traces():
    """Real traces add sample by sample with real weights; the default
    weights come from the stamp and a complex weight is refused because a
    real trace has no phase to rotate."""
    a = np.sin(np.linspace(0.0, 6.0, 64))
    b = np.cos(np.linspace(0.0, 6.0, 64))
    stack = ResultStack([_time_slab(30.0, a), _time_slab(70.0, b)],
                        [30.0, 70.0])
    summed = stack.superpose([2.0, 0.5])
    assert not summed.is_complex
    assert 'time' in summed.coords
    np.testing.assert_allclose(summed.data[0, 0], 2.0 * a + 0.5 * b,
                               rtol=1e-12)
    np.testing.assert_allclose(stack.superpose().data[0, 0], a - b,
                               rtol=1e-12)
    with pytest.raises(ConfigurationError, match='real time-domain traces'):
        stack.superpose([1.0, 1j])


@pytest.mark.requires_binary
@pytest.mark.slow
def test_a_time_series_run_over_two_depths_stacks_real_traces():
    """Scooter TIME_SERIES over a two-depth ``Source`` returns two real
    time-domain slabs whose superposition is their sum."""
    env = uacpy.Environment(name='ts-pair', bathymetry=GUIDE_DEPTH,
                            ssp=1500.0)
    receiver = uacpy.Receiver(depths=[50.0], ranges=[2000.0])
    fs, n = 2000.0, 256
    t = np.arange(n) / fs
    waveform = np.sin(2 * np.pi * F0 * t) * np.hanning(n)
    stack = _scooter().run(
        env, uacpy.Source(depths=[30.0, 70.0], frequencies=F0), receiver,
        run_mode=RunMode.TIME_SERIES,
        frequencies=np.linspace(60.0, 140.0, 17),
        source_waveform=waveform, sample_rate=fs,
    )
    assert isinstance(stack, ResultStack)
    assert stack.n_slabs == 2
    for _, slab in stack:
        assert 'time' in slab.coords
        assert not slab.is_complex
    summed = stack.superpose()
    np.testing.assert_allclose(summed.data, stack[0].data + stack[1].data,
                               rtol=1e-12)


@pytest.mark.requires_binary
def test_a_bellhop_time_series_pair_shares_one_padded_time_axis():
    """Bellhop sizes its delay-and-sum window from each depth's own
    arrivals, so the two records differ in length; the loop pads the
    shorter one with silence at the end so the pair superposes, and the
    longer slab is the single run untouched."""
    env, receiver, waveform, fs = _bellhop_time_series_setup()
    model = _bellhop()
    stack = model.run(env, uacpy.Source(depths=[30.0, 70.0], frequencies=F0),
                      receiver, run_mode=RunMode.TIME_SERIES,
                      source_waveform=waveform, sample_rate=fs)
    singles = [model.run(env, uacpy.Source(depths=z, frequencies=F0),
                         receiver, run_mode=RunMode.TIME_SERIES,
                         source_waveform=waveform, sample_rate=fs)
               for z in (30.0, 70.0)]
    lengths = [f.coords['time'].size for f in singles]
    assert lengths[0] != lengths[1], "the pair no longer needs padding"
    n_max = max(lengths)
    for slab, single, n in zip(stack.slabs, singles, lengths):
        assert slab.coords['time'].size == n_max
        np.testing.assert_allclose(slab.coords['time'][:n],
                                   single.coords['time'], rtol=1e-12)
        np.testing.assert_array_equal(slab.data[..., :n], single.data)
        assert not np.any(slab.data[..., n:])
        assert slab.metadata['nt'] == n_max
    summed = stack.superpose()
    np.testing.assert_allclose(summed.data, stack[0].data + stack[1].data,
                               rtol=1e-12)


def _bellhop_time_series_setup():
    env = make_pekeris(name='ts-window', bathymetry=GUIDE_DEPTH)
    receiver = uacpy.Receiver(depths=[50.0], ranges=[2000.0])
    fs, n = 2000.0, 256
    t = np.arange(n) / fs
    return env, receiver, np.sin(2 * np.pi * F0 * t) * np.hanning(n), fs


@pytest.mark.requires_binary
def test_a_window_that_ends_before_the_first_echo_names_its_arrival():
    """``output_duration=1.0`` at 2 km ends before the 1.32 s first echo,
    so every trace is silence and the warning names that arrival time and
    the window; a window that holds the echoes raises no window notice."""
    env, receiver, waveform, fs = _bellhop_time_series_setup()
    model = _bellhop()
    source = uacpy.Source(depths=30.0, frequencies=F0)
    with pytest.warns(UserWarning,
                      match=r'\[0, 1\] s window.*earliest echo arrives at '
                            r'1\.3\d* s') as record:
        short = model.run(env, source, receiver,
                          run_mode=RunMode.TIME_SERIES,
                          source_waveform=waveform, sample_rate=fs,
                          output_duration=1.0)
    assert not np.any(short.data)
    assert any('all of the received energy' in str(w.message)
               for w in record)
    import warnings
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        wide = model.run(env, source, receiver,
                         run_mode=RunMode.TIME_SERIES,
                         source_waveform=waveform, sample_rate=fs,
                         output_duration=2.0)
    assert np.any(wide.data)
    assert not any('window does not hold' in str(w.message) for w in record)


@pytest.mark.requires_binary
def test_a_complex_weight_on_a_time_domain_run_is_refused():
    env, receiver, waveform, fs = _bellhop_time_series_setup()
    with pytest.raises(ConfigurationError, match='complex weight cannot'):
        _bellhop().run(env, uacpy.Source(depths=30.0, frequencies=F0,
                                         weights=1j),
                       receiver, run_mode=RunMode.TIME_SERIES,
                       source_waveform=waveform, sample_rate=fs)
    real = _bellhop().run(env, uacpy.Source(depths=30.0, frequencies=F0,
                                            weights=-2.0),
                          receiver, run_mode=RunMode.TIME_SERIES,
                          source_waveform=waveform, sample_rate=fs)
    assert not real.is_complex
    assert real.metadata['superposed_sources'] == {'depths': [30.0],
                                                   'weights': [-2.0]}


def test_superpose_names_the_differing_lengths_of_a_time_axis_mismatch():
    short = _time_slab(30.0, np.ones(8))
    long = _time_slab(70.0, np.ones(10))
    with pytest.raises(ConfigurationError,
                       match=r"shape \(1, 1, 8\) vs \(1, 1, 10\), axis "
                             r"lengths \{'time': \(8, 10\)\}"):
        ResultStack([short, long], [30.0, 70.0]).superpose()


# ── (e) guards ──────────────────────────────────────────────────────────


def test_source_weights_default_to_ones_and_broadcast():
    src = uacpy.Source(depths=[10.0, 20.0, 30.0], frequencies=F0)
    np.testing.assert_array_equal(src.weights, [1.0, 1.0, 1.0])
    assert src.weights.dtype == np.complex128
    assert src.has_unit_weights
    scaled = uacpy.Source(depths=[10.0, 20.0], frequencies=F0, weights=2.0)
    np.testing.assert_array_equal(scaled.weights, [2.0, 2.0])
    assert not scaled.has_unit_weights
    one = uacpy.Source(depths=10.0, frequencies=F0, weights=[1j])
    np.testing.assert_array_equal(one.weights, [1j])


def test_a_length_one_weight_array_does_not_broadcast():
    """Only a scalar broadcasts; ``[2.0]`` on two depths is a length
    mismatch, while the same value as a scalar is accepted."""
    with pytest.raises(ConfigurationError,
                       match=r'2 depth\(s\) but 1 weight\(s\)'):
        uacpy.Source(depths=[10.0, 20.0], frequencies=F0, weights=[2.0])
    with pytest.raises(ConfigurationError,
                       match=r'2 depth\(s\) but 1 weight\(s\)'):
        uacpy.Source(depths=[10.0, 20.0], frequencies=F0,
                     weights=np.array([2.0]))
    ok = uacpy.Source(depths=[10.0, 20.0], frequencies=F0, weights=2.0)
    np.testing.assert_array_equal(ok.weights, [2.0, 2.0])
    both = uacpy.Source(depths=[10.0, 20.0], frequencies=F0,
                        weights=[2.0, 2.0])
    np.testing.assert_array_equal(both.weights, [2.0, 2.0])


def test_source_weights_length_must_match_the_depths():
    with pytest.raises(ConfigurationError,
                       match=r'2 depth\(s\) but 3 weight\(s\)'):
        uacpy.Source(depths=[10.0, 20.0], frequencies=F0,
                     weights=[1.0, 1.0, 1.0])
    with pytest.raises(ConfigurationError, match='scalar or a 1-D vector'):
        uacpy.Source(depths=[10.0, 20.0], frequencies=F0,
                     weights=[[1.0], [1.0]])


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf, np.nan * 1j])
def test_source_weights_must_be_finite(bad):
    with pytest.raises(ConfigurationError, match=r'weights\[1\]'):
        uacpy.Source(depths=[10.0, 20.0], frequencies=F0,
                     weights=[1.0, bad])


def test_source_repr_shows_weights_only_when_not_all_ones():
    plain = uacpy.Source(depths=[10.0, 20.0], frequencies=F0)
    assert 'weights' not in repr(plain)
    assert repr(uacpy.Source(depths=[10.0, 20.0], frequencies=F0,
                             weights=[1.0, 1.0])) == repr(plain)
    weighted = uacpy.Source(depths=[10.0, 20.0], frequencies=F0,
                            weights=[1.0, -1.0])
    assert 'weights=[(1+0j), (-1+0j)]' in repr(weighted)


def test_at_depth_copies_one_depth_with_unit_weight():
    pattern = np.array([[-90.0, -20.0], [0.0, 0.0], [90.0, -20.0]])
    src = uacpy.Source(depths=[10.0, 20.0], frequencies=[90.0, 110.0],
                       source_type='line', beam_pattern=pattern,
                       weights=[2.0, -1j])
    one = src.at_depth(1)
    np.testing.assert_array_equal(one.depths, [20.0])
    np.testing.assert_array_equal(one.frequencies, src.frequencies)
    assert one.source_type == 'line'
    np.testing.assert_array_equal(one.beam_pattern, pattern)
    assert one.has_unit_weights


def _complex_slab(z, value):
    return Field(
        data=np.full((2, 3), value, dtype=complex),
        coords={'depth': [10.0, 20.0], 'range': [100.0, 200.0, 300.0]},
        model='Test', frequencies=F0, source_depths=[z],
        phase_reference='travelling_wave',
    )


def test_superpose_refuses_a_wrong_length_or_non_finite_weight_vector():
    stack = ResultStack([_complex_slab(10.0, 1.0), _complex_slab(20.0, 2.0)],
                        [10.0, 20.0])
    with pytest.raises(ConfigurationError, match=r'2 slabs but 3 weight'):
        stack.superpose([1.0, 1.0, 1.0])
    with pytest.raises(ConfigurationError, match=r'2 slabs but 1 weight'):
        stack.superpose([1.0])
    with pytest.raises(ConfigurationError, match=r'weights\[0\]'):
        stack.superpose([np.nan, 1.0])
    # Both sides of the length guard: exactly one weight per slab adds.
    assert stack.superpose([1.0, 1.0]).data[0, 0] == 3.0 + 0j


def test_superpose_refuses_a_stack_that_lost_its_phase():
    """A dB-only stack cannot add coherently; the complex one it came from
    can, which pins that the refusal is about phase and not about the
    stack."""
    complex_stack = ResultStack(
        [_complex_slab(10.0, 1.0 + 1j), _complex_slab(20.0, 2.0)],
        [10.0, 20.0])
    assert complex_stack.superpose().data[0, 0] == 3.0 + 1j
    db_stack = ResultStack([s.to_dB() for s in complex_stack.slabs],
                           [10.0, 20.0])
    with pytest.raises(ConfigurationError,
                       match='not complex pressure.*coherent sum'):
        db_stack.superpose()


def test_superpose_refuses_slabs_on_different_grids():
    a = _complex_slab(10.0, 1.0)
    b = Field(data=np.ones((2, 3), dtype=complex),
              coords={'depth': [10.0, 20.0],
                      'range': [100.0, 200.0, 400.0]},
              model='Test', frequencies=F0, source_depths=[20.0],
              phase_reference='travelling_wave')
    with pytest.raises(ConfigurationError, match='different grid'):
        ResultStack([a, b], [10.0, 20.0]).superpose()


def test_superpose_refuses_non_field_slabs():
    from uacpy.core.results import Rays
    rays = [Rays(rays=[], model='Test', frequencies=F0, source_depths=[z])
            for z in (10.0, 20.0)]
    with pytest.raises(ConfigurationError, match='not Field'):
        ResultStack(rays, [10.0, 20.0]).superpose()


@pytest.mark.requires_binary
def test_a_multi_depth_source_in_a_non_field_mode_keeps_its_refusal():
    """Mode shapes have no per-source sum, so ``MODES`` on a two-depth
    ``Source`` still raises and names the field modes that do stack."""
    model = _kraken()
    source = uacpy.Source(depths=[30.0, 70.0], frequencies=F0)
    with pytest.raises(ConfigurationError,
                       match='single source depth per MODES run'):
        model.run(_env(), source, _receiver(), run_mode=RunMode.MODES)
    with pytest.raises(ConfigurationError, match='field modes'):
        model.validate_inputs(_env(), source, _receiver(),
                              run_mode=RunMode.MODES)


@pytest.mark.requires_binary
def test_the_zero_range_refusal_fires_before_any_per_depth_run():
    """The loop runs each depth through the wrapper's own validation, so a
    receiver with no positive range is refused on the first depth."""
    model = _kraken()
    source = uacpy.Source(depths=[30.0, 70.0], frequencies=F0)
    with pytest.raises(ConfigurationError, match='largest range is 0 m'):
        model.run(_env(), source,
                  uacpy.Receiver(depths=[50.0], ranges=[0.0]))
