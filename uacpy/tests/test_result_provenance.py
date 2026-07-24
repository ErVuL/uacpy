"""Model-source provenance must survive every derived-result operation.

``model_source`` drives the "Model: … " credit footnote every result plot
draws (:func:`_draw_result_credit`). Any helper that *derives* a new result
from an existing one — filtering rays/arrivals, IFFT-ing a spectrum,
synthesizing a time series, subsetting modes — has to carry it across, or the
derived result silently plots without attribution.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import uacpy
from uacpy.core.results import Arrivals, Field, Modes, PhaseReference, Rays
from uacpy.models.sources import model_source

_SRC = model_source('acoustics_toolbox')
_FREQS = np.linspace(100.0, 300.0, 32)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close('all')


def _broadband():
    rng = np.random.default_rng(0)
    data = (rng.standard_normal((1, 1, _FREQS.size))
            + 1j * rng.standard_normal((1, 1, _FREQS.size)))
    return Field(
        data=data,
        coords={'depth': np.array([25.0]), 'range': np.array([1000.0]),
                'frequency': _FREQS},
        model='Synthetic', source_depths=np.array([5.0]), frequencies=_FREQS,
        phase_reference=PhaseReference.TRAVELLING_WAVE,
        model_source=_SRC, metadata={'c0': 1500.0},
    )


def _rays():
    return Rays(
        rays=[{'r': np.linspace(0, 2000, 50),
               'z': 50 + 10 * np.sin(np.linspace(0, 5, 50)),
               'alpha': 0.0, 'n_top_bounces': 0, 'n_bot_bounces': 0},
              {'r': np.linspace(0, 2000, 50),
               'z': 50 + 20 * np.cos(np.linspace(0, 6, 50)),
               'alpha': 3.0, 'n_top_bounces': 1, 'n_bot_bounces': 1}],
        receiver_depths=np.array([50.0]), receiver_ranges=np.array([2000.0]),
        source_depths=np.array([10.0]), model='Bellhop', model_source=_SRC,
    )


def _arrivals():
    mk = lambda d, a, tb, bb: {                                  # noqa: E731
        'delay': d, 'amplitude': a, 'phase': 0.0,
        'n_top_bounces': tb, 'n_bot_bounces': bb, 'src_angle': 0,
        'rcv_angle': 0, 'kind': 'direct' if not (tb or bb) else 'both',
        'src_idx': 0, 'depth_idx': 0, 'range_idx': 0}
    return Arrivals(
        arrivals=[mk(0.5, 1.0, 0, 0), mk(0.7, 0.4, 1, 1)],
        receiver_depths=np.array([50.0]), receiver_ranges=np.array([2000.0]),
        source_depths=np.array([10.0]), model='Bellhop', model_source=_SRC,
    )


def _modes():
    depths = np.linspace(0, 100, 25)
    phi = np.stack([np.sin((m + 1) * np.pi * depths / 100.0)
                    for m in range(4)], axis=1)
    k = np.array([np.sqrt((2 * np.pi * 100 / 1500) ** 2
                          - ((m + 1) * np.pi / 100.0) ** 2 + 0j)
                  for m in range(4)])
    return Modes(k=k, phi=phi, depths=depths, model='Kraken',
                 frequencies=100.0, model_source=_SRC)


# ── derived results keep the model source ───────────────────────────────────

def test_rays_filter_keeps_model_source():
    r = _rays()
    assert r.filter_by_bounces(kind='direct').model_source is _SRC
    assert r.top_n_by_miss(1).model_source is _SRC
    assert r.truncate_at_receiver().model_source is _SRC


def test_arrivals_filter_keeps_model_source():
    a = _arrivals()
    assert a.filter_by_bounces(kind='direct').model_source is _SRC
    assert a.top_n_by_amplitude(1).model_source is _SRC


def test_ifft_trace_keeps_model_source():
    assert _broadband().to_time_trace().model_source is _SRC


def test_synthesized_time_series_keeps_model_source():
    wf = np.zeros(64); wf[:8] = 1.0
    ts = _broadband().synthesize_time_series(wf, sample_rate=4000.0)
    assert ts.model_source is _SRC


def test_modes_derivations_keep_model_source():
    m = _modes()
    assert m.first_n(2).model_source is _SRC
    assert m.with_attenuation(0.1).model_source is _SRC
    field = m.modal_propagation_loss(
        source_depth=25.0, receiver_depths=np.linspace(5, 95, 6),
        ranges_m=np.linspace(100, 2000, 8))
    assert field.model_source is _SRC


# ── the derived result still renders its credit ─────────────────────────────

def _credits(fig):
    return [t.get_text() for t in fig.texts if 'Model' in t.get_text()]


def test_filtered_rays_plot_shows_credit():
    env = uacpy.Environment(bathymetry=100.0, ssp=1500.0)
    fig, _ = _rays().top_n_by_miss(1).plot(env=env)
    assert _credits(fig)


def test_filtered_arrivals_plot_shows_credit():
    fig, _ = _arrivals().top_n_by_amplitude(1).plot()
    assert _credits(fig)


# ── single-result plotters that were missing the credit ─────────────────────

def _se_field():
    d = np.linspace(5, 95, 8)
    r = np.linspace(100, 5000, 12)
    return Field(
        data=np.tile(np.linspace(20.0, -20.0, 12), (8, 1)),
        coords={'depth': d, 'range': r},
        model='Synthetic', frequencies=100.0, model_source=_SRC,
    )


def test_plot_signal_excess_shows_credit():
    fig, _ = uacpy.plot.plot_signal_excess(_se_field())
    assert _credits(fig)


def test_plot_detection_probability_shows_credit():
    pd = _se_field()
    pd.data = np.clip(pd.data / 40.0 + 0.5, 0.0, 1.0)
    fig, _ = uacpy.plot.plot_detection_probability(pd)
    assert _credits(fig)
