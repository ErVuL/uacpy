"""The transfer function a pulse of a given length sees.

``H(f)`` as a model returns it is the continuous-wave answer: every path
present at once, interfering. A pulse of duration ``T`` does not meet that
channel — two copies of it interfere only where they overlap, so arrivals
further apart than ``T`` arrive as separate echoes and do not add. Two
routes to the pulse's own ``H``:

* :meth:`Arrivals.transfer_function` on a delay-windowed arrival set, which
  is exact but needs paths, so it is Bellhop's alone;
* :meth:`Field.truncate_response`, which cuts the impulse response of any
  broadband ``H(f)`` and therefore works for a wave model too.

The pair of them is the point, so the two are checked against each other on
the same arrivals as well as separately against closed forms.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Arrivals, Field

DF, N_FREQ, F0 = 5.0, 128, 1000.0
FREQS = F0 + DF * np.arange(N_FREQ)
RECORD = 1.0 / DF                       # 200 ms
DT = RECORD / N_FREQ                    # 1.5625 ms, the band's resolution
GAP = 26 * DT                           # 40.625 ms, on a sample instant
AMPS = (1.0, 0.5)


def _two_path_field(gap=GAP, amps=AMPS, freqs=FREQS):
    """``H(f)`` of two paths, the second ``gap`` later and quieter.

    Both delays sit on sample instants of the ``1/df`` record, so the band's
    own resolution kernel is a delta and the test measures the cut rather
    than the leakage around it.
    """
    delays = np.array([1 * DT, 1 * DT + gap])
    H = (np.asarray(amps)[:, None]
         * np.exp(-2j * np.pi * np.outer(delays, freqs))).sum(0)
    return Field(data=H.reshape(1, 1, freqs.size),
                 coords={'depth': np.array([10.0]),
                         'range': np.array([100.0]),
                         'frequency': np.asarray(freqs, dtype=float)})


def _ripple_dB(field):
    mag = np.abs(np.asarray(field.data).ravel())
    return float(20.0 * np.log10(mag.max() / mag.min()))


def _two_path_arrivals(gap=GAP, amps=AMPS):
    """The same two paths as an ``Arrivals``, for the cross-check."""
    return Arrivals(
        arrivals=[{'delay': 1 * DT, 'amplitude': amps[0], 'phase': 0.0,
                   'delay_imag': 0.0},
                  {'delay': 1 * DT + gap, 'amplitude': amps[1], 'phase': 0.0,
                   'delay_imag': 0.0}],
        receiver_depths=[10.0], receiver_ranges=[100.0],
        model='Test', frequencies=F0)


class TestTheFoldWarningIsPinnedOnBothSides:
    """The far-half fold warning, held at both bounds and across windows.

    Everything here was green before it existed. An audit mutation dropped
    the 0.02 threshold by 200x to 0.0001 — which reintroduces the exact
    false positive the measure was changed to remove — and the suite did
    not notice, because the only negative control had no far-half energy at
    all and so passed for any positive threshold. Likewise nothing exercised
    ``window='hann'`` inside a warning assertion, so re-deriving the window
    centre from the taper (which returns a boxcar's LEFT EDGE) would also
    have been green.
    """

    @staticmethod
    def two_paths(df, gap, amplitude=1.0, first=0.010):
        f = np.arange(25.0, 4000.0, df)
        h = (np.exp(-2j * np.pi * f * first)
             + amplitude * np.exp(-2j * np.pi * f * (first + gap)))
        return Field(data=h.reshape(1, 1, -1),
                     coords={'depth': [0.0], 'range': [1.0], 'frequency': f},
                     frequencies=f)

    @staticmethod
    def warned(field, duration=0.005, **kw):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            field.truncate_response(duration, **kw)
        return any('furthest from the window' in str(w.message)
                   for w in caught)

    def test_a_channel_that_cannot_fold_is_not_accused_of_folding(self):
        """The LOW bound, and the side that actually bit.

        Two equal paths 100 ms apart in a 500 ms record: the spread is a
        fifth of the record, so nothing can have wrapped. It carries real
        far-half energy (~0.001 of the total), so a threshold set much
        below 0.02 fires on it — which is what makes this pin the bound
        rather than merely pass.
        """
        assert not self.warned(self.two_paths(2.0, 0.100))
        assert not self.warned(self.two_paths(2.0, 0.100, amplitude=0.7))

    def test_a_folded_path_is_named(self):
        """The HIGH bound: a 100 ms gap in a 50 ms record must wrap."""
        assert self.warned(self.two_paths(20.0, 0.100))
        assert self.warned(self.two_paths(20.0, 0.300))

    def test_the_verdict_does_not_depend_on_the_window_shape(self):
        """The window centre comes from the caller, not from ``argmax``.

        ``argmax`` on a boxcar returns its left edge, so a centre re-derived
        inside the warning rotates the far-half mask by the window's
        half-width for 'boxcar' and not at all for 'hann' — the same
        channel and cut then disagree by window shape.
        """
        # The first two are comfortably either side and would agree even
        # with a rotated mask — they are here as controls. The THIRD sits
        # where the rotation matters: under a centre taken from
        # ``argmax(taper)`` it reads boxcar=False, hann=True, and it was
        # found by sweeping for a split rather than guessed.
        for field, cut in ((self.two_paths(2.0, 0.100), 0.005),
                           (self.two_paths(20.0, 0.100), 0.005),
                           (self.two_paths(5.0, 0.150, amplitude=0.5), 0.020)):
            assert (self.warned(field, window='boxcar', duration=cut)
                    == self.warned(field, window='hann', duration=cut))


class TestAPulseOnlyInterferesWithWhatItOverlaps:
    """``Field.truncate_response`` is the separability criterion applied."""

    def test_a_path_beyond_the_pulse_leaves_the_response(self):
        """The whole point. Two paths 40.6 ms apart ripple by
        ``20log10((a+b)/(a-b))`` under a continuous wave; a 20 ms pulse
        cannot overlap them, so its channel is the first path alone — flat,
        at that path's own amplitude."""
        field = _two_path_field()
        assert _ripple_dB(field) == pytest.approx(
            20 * np.log10(sum(AMPS) / (AMPS[0] - AMPS[1])), abs=1e-6)
        cut = np.abs(np.asarray(
            field.truncate_response(0.020).data).ravel())
        assert cut.max() - cut.min() == pytest.approx(0.0, abs=1e-9), (
            f'the cut channel still ripples by {cut.max() - cut.min():g}')
        assert cut.mean() == pytest.approx(AMPS[0], abs=1e-9)

    def test_a_path_within_the_pulse_interferes_as_under_a_tone(self):
        """A longer pulse overlaps both, and then the cut must give the
        continuous-wave answer back — otherwise it is removing energy on its
        own account rather than applying a criterion."""
        field = _two_path_field()
        cut = field.truncate_response(0.050)
        assert _ripple_dB(cut) == pytest.approx(_ripple_dB(field), abs=1e-6)

    @pytest.mark.parametrize('fraction,keeps', [(0.98, False), (1.02, True)])
    def test_the_window_reaches_one_pulse_length_either_side(self, fraction,
                                                             keeps):
        """Both sides of the threshold. Overlap needs ``|dtau| < T``, so a
        pulse just shorter than the gap must drop the far path and one just
        longer must keep it — a half-width convention fails the first of
        these and a two-pulse one fails the second."""
        cut = _two_path_field().truncate_response(GAP * fraction)
        rippled = _ripple_dB(cut) > 1.0
        assert rippled is keeps, (
            f'pulse {fraction:g} x the {GAP * 1e3:.1f} ms gap gave '
            f'{_ripple_dB(cut):.3f} dB of ripple')

    def test_the_window_reaches_round_the_end_of_the_record(self):
        """The record is one period, so a window centred near its start
        reaches back round to its end — and a path that arrives just before
        the origin sits there. Measuring the offset without that wrap
        silently drops it, and every fixture whose paths all follow the
        origin passes anyway."""
        # Second path 5 ms BEFORE the first, which on a 200 ms record lands
        # at 195 ms: circularly 5 ms away, linearly 190 ms away.
        delays = np.array([1 * DT, 1 * DT - 3 * DT + RECORD])
        H = (np.asarray(AMPS)[:, None]
             * np.exp(-2j * np.pi * np.outer(delays, FREQS))).sum(0)
        field = Field(data=H.reshape(1, 1, FREQS.size),
                      coords={'depth': np.array([10.0]),
                              'range': np.array([100.0]),
                              'frequency': FREQS})
        cut = field.truncate_response(0.020)
        assert _ripple_dB(cut) == pytest.approx(_ripple_dB(field), abs=1e-6), (
            f'a path {3 * DT * 1e3:.2f} ms before the origin was dropped by a '
            f'{20:g} ms pulse: ripple fell from {_ripple_dB(field):.3f} to '
            f'{_ripple_dB(cut):.3f} dB')

    def test_a_response_reaching_the_far_half_is_named(self):
        """Energy where a well-sized record would be empty is named.

        The measure is the FAR HALF of the record from the window, not the
        whole complement of it — energy just outside the window is what a
        cut exists to remove. What lands in the far half is ambiguous: this
        fixture's second path sits at 0.451 of a 70 ms record and has NOT
        folded, and the warning says so, offering both readings rather than
        asserting one. Matching on the far-half phrase rather than on
        'folded' is deliberate: the old name and the old match both claimed
        a fold this fixture does not contain.
        """
        coarse = np.linspace(F0, F0 + 100.0, 8)          # record 70 ms
        field = _two_path_field(gap=0.030, amps=(1.0, 1.0), freqs=coarse)
        with pytest.warns(UserWarning, match='furthest from the window'):
            field.truncate_response(0.002)

    def test_a_decayed_response_is_not_named(self):
        """The complement: the warning must stay silent when the record does
        hold the response, or it says nothing about the case it exists for."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _two_path_field().truncate_response(0.020)

    def test_the_tapered_cut_removes_the_same_path_with_a_lower_skirt(self):
        """``window='hann'`` is the same criterion, softened. It must still
        drop the far path, and its own skirt on H must be below the
        rectangle's."""
        field = _two_path_field()
        box = np.abs(np.asarray(field.truncate_response(0.020).data).ravel())
        han = np.abs(np.asarray(
            field.truncate_response(0.020, window='hann').data).ravel())
        assert box.mean() == pytest.approx(AMPS[0], abs=1e-9)
        assert han.mean() == pytest.approx(AMPS[0], abs=1e-3)
        assert han.max() - han.min() <= box.max() - box.min() + 1e-9

    def test_every_cell_is_cut_on_its_own_arrival(self):
        """``origin='peak'`` is per cell. A field whose two cells arrive at
        different times must keep both, which a single shared origin cannot
        do."""
        early = _two_path_field()
        # A whole number of samples of delay, so this cell's paths land on
        # sample instants too and the test still measures the cut.
        late = Field(data=np.asarray(early.data)
                     * np.exp(-2j * np.pi * FREQS * (38 * DT)),
                     coords=dict(early.coords))
        stacked = Field(
            data=np.concatenate([early.data, late.data], axis=0),
            coords={'depth': np.array([10.0, 20.0]),
                    'range': np.array([100.0]), 'frequency': FREQS})
        cut = np.abs(np.asarray(stacked.truncate_response(0.020).data))
        for cell in (0, 1):
            row = cut[cell, 0]
            assert row.mean() == pytest.approx(AMPS[0], abs=1e-9), (
                f'cell {cell} kept {row.mean():g}, not the {AMPS[0]:g} its '
                f'own first arrival delivers')

    @pytest.mark.parametrize('kwargs,message', [
        ({'duration': 0.0}, 'positive'),
        ({'duration': RECORD}, 'no-op'),
        ({'duration': 0.02, 'window': 'blackman'}, 'boxcar'),
        ({'duration': 0.02, 'origin': 'first'}, 'origin'),
        ({'duration': 0.02, 'origin': 5.0}, 'outside'),
    ])
    def test_it_refuses_what_it_cannot_answer(self, kwargs, message):
        with pytest.raises(ConfigurationError, match=message):
            _two_path_field().truncate_response(**kwargs)

    def test_it_refuses_a_field_with_no_phase(self):
        """A dB or TL field has no impulse response to cut."""
        field = _two_path_field().to_dB()
        with pytest.raises(ConfigurationError, match='complex'):
            field.truncate_response(0.020)

    def test_it_refuses_a_field_with_no_frequency_axis(self):
        field = Field(data=np.ones((1, 3)) + 0j,
                      coords={'depth': np.array([1.0]),
                              'time': np.arange(3.0)})
        with pytest.raises(ConfigurationError, match='frequency axis'):
            field.truncate_response(0.020)


class TestArrivalsAnswerForTheirOwnPaths:
    """``Arrivals.transfer_function`` sums the paths it is left holding."""

    def test_it_is_the_sum_over_paths_of_their_phase_terms(self):
        H = np.asarray(
            _two_path_arrivals().transfer_function(FREQS).data).ravel()
        expected = np.asarray(_two_path_field().data).ravel()
        assert np.allclose(H, expected, atol=1e-12)

    def test_the_absorption_in_im_tau_is_applied_at_each_frequency(self):
        """Bellhop keeps volume absorption in ``Im tau``, so the amplitude a
        path delivers is a function of frequency — ``A exp(2 pi f Im tau)``.
        Freezing it at the result's own frequency hands back a band with no
        absorption slope across it, which a fixture with ``delay_imag = 0``
        cannot tell from the right answer."""
        im_tau = -1e-5
        arr = Arrivals(
            arrivals=[{'delay': 1 * DT, 'amplitude': 1.0, 'phase': 0.0,
                       'delay_imag': im_tau}],
            receiver_depths=[10.0], receiver_ranges=[100.0],
            model='Test', frequencies=F0)
        mag = np.abs(np.asarray(
            arr.transfer_function(FREQS).data).ravel())
        expected = np.exp(2.0 * np.pi * FREQS * im_tau)
        assert np.allclose(mag, expected, rtol=1e-12), (
            f'band tilt is {20 * np.log10(mag[-1] / mag[0]):.4f} dB; '
            f'exp(2 pi f Im tau) gives '
            f'{20 * np.log10(expected[-1] / expected[0]):.4f} dB')
        # And it is a tilt worth having: a frozen amplitude would be flat.
        assert 20 * np.log10(mag[0] / mag[-1]) > 0.3

    def test_it_places_the_answer_on_the_cell_it_came_from(self):
        field = _two_path_arrivals().transfer_function(FREQS)
        assert list(field.coords) == ['depth', 'range', 'frequency']
        assert field.coords['depth'][0] == pytest.approx(10.0)
        assert field.coords['range'][0] == pytest.approx(100.0)

    def test_a_delay_window_is_what_makes_it_a_pulse_s_channel(self):
        """Dropping the far path by hand gives the first path alone — the
        path-domain statement of the same criterion."""
        arr = _two_path_arrivals()
        near = arr.in_delay_window(None, 1 * DT + 0.020)
        assert len(near) == 1
        H = np.abs(np.asarray(near.transfer_function(FREQS).data).ravel())
        assert H.max() - H.min() == pytest.approx(0.0, abs=1e-12)
        assert H.mean() == pytest.approx(AMPS[0], abs=1e-12)

    @pytest.mark.parametrize('frequencies,message', [
        ([], 'empty'),
        ([1000.0, np.nan], 'finite'),
        ([-10.0, 1000.0], 'positive'),
    ])
    def test_it_refuses_a_grid_it_cannot_evaluate_on(self, frequencies,
                                                     message):
        with pytest.raises(ConfigurationError, match=message):
            _two_path_arrivals().transfer_function(frequencies)


class TestTheTwoRoutesToAPulseChannelAgree:
    """The cross-check that licenses the model-independent route.

    ``in_delay_window(...).transfer_function(...)`` knows the paths and drops
    them exactly. ``transfer_function(...).truncate_response(...)`` sees only
    ``H(f)`` and has to find them in the response. On a set where both apply,
    they must land on the same channel — that is what says the second may be
    used on a wave model, which has no paths to drop.
    """

    def test_windowing_the_paths_and_cutting_the_response_agree(self):
        arr = _two_path_arrivals()
        pulse = 0.020
        by_path = np.asarray(
            arr.in_delay_window(None, 1 * DT + pulse)
            .transfer_function(FREQS).data).ravel()
        by_response = np.asarray(
            arr.transfer_function(FREQS).truncate_response(pulse).data
        ).ravel()
        worst = float(np.abs(by_response - by_path).max())
        assert worst < 1e-9, (
            f'the two routes differ by {worst:g}; by-path |H| = '
            f'{np.abs(by_path).mean():g}, by-response '
            f'{np.abs(by_response).mean():g}')

    def test_they_agree_when_the_pulse_keeps_both_paths_too(self):
        """Agreement on the flat case alone would pass for a cut that always
        returns one path. This is the same comparison where the answer is
        the two-path interference."""
        arr = _two_path_arrivals()
        by_path = np.asarray(arr.transfer_function(FREQS).data).ravel()
        by_response = np.asarray(
            arr.transfer_function(FREQS).truncate_response(0.050).data
        ).ravel()
        assert np.abs(by_response - by_path).max() < 1e-9


@pytest.mark.requires_binary
class TestTheSumOverPathsIsTheModelsOwnBroadbandRun:
    """``Arrivals.transfer_function`` against ``RunMode.BROADBAND``.

    The closed forms above pin the expression; this pins that the expression
    is the one the model evaluates. It matters because a caller who trusts
    the agreement stops launching the second run — as
    ``bellhop_bottom_com/symbol_channel.py`` does, on a grid of tens of
    thousands of points that is free as a sum over paths and is not free as a
    model launch.
    """

    RANGE_M, DEPTH_M, FC = 4000.0, 36.0, 50.0

    @classmethod
    def _run(cls):
        from uacpy.core.receiver import Receiver
        from uacpy.core.source import Source
        from uacpy.models import Bellhop, RunMode
        from uacpy.tests.conftest import make_pekeris
        env = make_pekeris(name='pekeris-transfer-function', density=1.7)
        rcv = Receiver(depths=np.array([cls.DEPTH_M]),
                       ranges=np.array([cls.RANGE_M]))
        grid = np.linspace(cls.FC - 25.0, cls.FC + 25.0, 51)
        arrivals = Bellhop(verbose=False).run(
            env, Source(depths=cls.DEPTH_M, frequencies=cls.FC), rcv,
            run_mode=RunMode.ARRIVALS)
        broadband = Bellhop(verbose=False).run(
            env, Source(depths=cls.DEPTH_M, frequencies=grid), rcv,
            run_mode=RunMode.BROADBAND, frequencies=grid)
        return arrivals, broadband, grid

    def test_it_reproduces_the_broadband_run_to_floating_point(self):
        arrivals, broadband, grid = self._run()
        # A one-path cell agrees under any phase convention; this guide has
        # hundreds, so their sum is what is being compared.
        assert len(arrivals) > 100, len(arrivals)
        mine = np.asarray(arrivals.transfer_function(grid).data).ravel()
        theirs = np.asarray(broadband.data).ravel()
        level = float(np.abs(
            20.0 * np.log10(np.abs(mine) / np.abs(theirs))).max())
        phase = float(np.abs(np.angle(mine / theirs)).max())
        assert level < 1e-6, f'levels differ by up to {level:g} dB'
        assert phase < 1e-6, f'phases differ by up to {phase:g} rad'

    def test_it_lands_on_the_cell_the_run_asked_for(self):
        arrivals, broadband, grid = self._run()
        field = arrivals.transfer_function(grid)
        assert field.coords['depth'][0] == pytest.approx(
            broadband.coords['depth'][0])
        assert field.coords['range'][0] == pytest.approx(
            broadband.coords['range'][0])
