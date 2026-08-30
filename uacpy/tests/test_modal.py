"""Tests for ``uacpy.acoustic_signal``'s modal warping.

Warping resamples a dispersive arrival onto a time axis where the modes become
tones, so a modal spectrogram can separate them; unwarping is its inverse.
Three properties matter and each is pinned here:

* **Invertibility.** ``unwarp_signal(warp_signal(x))`` returns ``x``, so the
  transform loses nothing it was not asked to lose.
* **Return order.** ``warp_signal`` is the one function on the
  ``acoustic_signal`` public surface that returns ``(signal, axis)`` rather
  than ``(axis, signal)``, and the order is deliberate: its tuple is exactly
  the argument list :func:`unwarp_signal` takes, so the two compose by
  unpacking. Flipping it would keep every call running and quietly return a
  different signal.
* **The accuracy knob works.** ``oversample`` is honoured as a fraction rather
  than truncated to an integer, which had made every setting below 2 a no-op.
"""

import inspect
import warnings

import numpy as np
import pytest

from uacpy.acoustic_signal import (
    modal_group_velocity,
    unwarp_signal,
    warp_signal,
)
from uacpy.core.exceptions import ConfigurationError

FS = 10000.0


class TestModal:
    def test_group_velocity_of_nondispersive_guide(self):
        f = np.linspace(50, 500, 50)
        c = 1500.0
        kr = 2 * np.pi * f / c  # omega = c*kr -> vg = c
        vg = modal_group_velocity(f, kr)
        assert np.allclose(vg, c, rtol=1e-6)

    def test_group_velocity_of_lossy_modes_uses_re_kr(self):
        """Complex wavenumbers (KRAKENC lossy modes) are handled via the
        Re(k_r) convention: same answer as the real part alone, with no
        ComplexWarning."""
        f = np.linspace(50, 500, 50)
        kr = 2 * np.pi * f / 1500.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            vg = modal_group_velocity(f, kr + 2.5e-4j)
        np.testing.assert_array_equal(vg, modal_group_velocity(f, kr))
        assert np.allclose(vg, 1500.0, rtol=1e-6)

    def test_warp_unwarp_roundtrip(self):
        # Band-limited modal-like transient (warping targets dispersive arrivals).
        n = 1024
        t = np.arange(n) / FS
        x = (np.sin(2 * np.pi * 40 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)) \
            * np.hanning(n)
        w, tw = warp_signal(x, FS, 500.0)
        _, x2 = unwarp_signal(w, tw, FS, 500.0)
        # Both directions resample by linear interpolation onto a nonlinear
        # time axis, so the round trip is lossy by construction; the loss is
        # worst near t_w = 0, where the Jacobian is floored at one sample. The
        # 10-sample trim drops that end, and 0.9 asks that the waveform survive
        # in shape — it is not a round-trip bound. Raise ``oversample`` to
        # tighten it (see TestWarpIsUnitary).
        m = min(x.size, x2.size)
        corr = np.corrcoef(x[10:m - 10], x2[10:m - 10])[0, 1]
        assert corr > 0.9


class TestWarpIsUnitary:
    """``t_w = sqrt(t^2 - t_r^2)`` gives ``t = sqrt(t_w^2 + t_r^2)`` and hence
    ``dt/dt_w = t_w / t``, so the energy-preserving weight is ``sqrt(t_w / t)``.
    Its reciprocal inflates energy by a factor that grows with range — ~30x at
    20 km — which the range-independence test below is what detects.

    Bonnel et al. (2013) is not in the corpus here, so the weight is pinned by
    the conservation law the docstring itself claims rather than by citation.
    """

    FS = 2000.0
    N = 2048

    def _signal(self):
        return np.random.default_rng(0).standard_normal(self.N)

    def test_the_jacobian_matches_a_numerical_derivative(self):
        """Compared on the interior: ``np.gradient`` uses a one-sided stencil at
        the ends, and the curvature of ``t(t_w)`` near ``t_w -> 0`` makes that
        first point disagree by O(1) however fine the grid. The interior agrees
        to ~1e-5, which the reciprocal weight misses by a factor of ~8 here."""
        t_r = 0.6667
        tw = np.linspace(1.0 / self.FS, 1.0, 400)
        t = np.sqrt(tw ** 2 + t_r ** 2)
        numerical = np.gradient(t, tw)[5:-5]
        np.testing.assert_allclose(numerical, (tw / t)[5:-5], rtol=1e-4)
        # The shipped reciprocal is nowhere near it.
        assert not np.allclose(numerical, (t / tw)[5:-5], rtol=0.5)

    def test_energy_ratio_is_range_independent(self):
        """The discriminating property: a wrong Jacobian scales with range."""
        x = self._signal()
        ratios = []
        for range_m in (200.0, 1000.0, 5000.0, 20_000.0):
            w, tw = warp_signal(x, self.FS, range_m, c=1500.0)
            t = range_m / 1500.0 + np.arange(self.N) / self.FS
            ratios.append(float(np.trapezoid(w ** 2, tw)
                                / np.trapezoid(x ** 2, t)))
        assert max(ratios) / min(ratios) < 1.15, f"range-dependent: {ratios}"

    def test_the_inverse_returns_the_input_grid_at_any_oversample(self):
        """The output length follows the warped axis' extent, not ``w.size``."""
        x = self._signal()
        for k in (1, 2, 4, 8):
            w, tw = warp_signal(x, self.FS, 1000.0, c=1500.0, oversample=k)
            assert w.size == self.N * k
            _t, back = unwarp_signal(w, tw, self.FS, 1000.0, c=1500.0)
            assert back.size == self.N

    def test_oversampling_tightens_the_round_trip(self):
        """The map is expansive, so the warped grid is the limiting resolution;
        the docstring's claim that the error halves per doubling is the contract."""
        x = self._signal()
        errs = []
        for k in (1, 2, 4, 8):
            w, tw = warp_signal(x, self.FS, 1000.0, c=1500.0, oversample=k)
            _t, back = unwarp_signal(w, tw, self.FS, 1000.0, c=1500.0)
            errs.append(float(np.linalg.norm(back - x) / np.linalg.norm(x)))
        assert all(b < a for a, b in zip(errs, errs[1:])), errs
        assert errs[-1] < errs[0] / 4.0


class TestWarpSignalOversampleHonoursFractions:
    FS = 5000.0

    def _signal(self):
        return np.random.default_rng(21).normal(size=200)

    @pytest.mark.parametrize("factor,expected", [(1, 200), (1.5, 300),
                                                 (2, 400), (2.9, 580),
                                                 (1.253, 251)])
    def test_the_warped_axis_length_scales_before_rounding(self, factor, expected):
        # int(oversample) truncated every fraction, so 1.5 gave 200 samples —
        # the accuracy knob did nothing below 2. The 1.253 case is the only
        # one whose product (250.5999...) is not already a whole number in
        # float, so it is what separates round() from a bare int().
        w, _tw = warp_signal(self._signal(), self.FS, 1000.0, oversample=factor)
        assert w.size == expected

    @pytest.mark.parametrize("factor", [0, -2, 0.5])
    def test_a_factor_below_one_raises_instead_of_clamping_silently(self, factor):
        with pytest.raises(ConfigurationError, match="oversample must be >= 1"):
            warp_signal(self._signal(), self.FS, 1000.0, oversample=factor)

    def test_a_non_numeric_factor_raises_typed(self):
        with pytest.raises(ConfigurationError, match="must be a number"):
            warp_signal(self._signal(), self.FS, 1000.0, oversample='a')


class TestWarpSignalReturnsSignalFirstToFeedUnwarp:
    """``warp_signal`` is the one function on the ``acoustic_signal`` public
    surface that returns signal-first rather than axis-first. That is
    deliberate: its tuple is the argument list :func:`unwarp_signal` takes.
    Both arrays are the same length, so a swap cannot be caught by shape and
    would surface only as a wrong answer — this pins the coupling."""

    FS = 4000.0
    RANGE = 5000.0

    @staticmethod
    def _chirp(n, fs):
        t = np.arange(n) / fs
        return np.sin(2 * np.pi * (50.0 + 120.0 * t) * t) * np.exp(-3.0 * t)

    def test_the_first_element_is_the_signal_and_the_second_the_axis(self):
        x = self._chirp(1024, self.FS)
        warped, t_warp = warp_signal(x, self.FS, self.RANGE)
        # The axis starts at 0 (the direct arrival maps to t_w = 0) and
        # increases; the signal does neither.
        assert t_warp[0] == pytest.approx(0.0)
        assert np.all(np.diff(t_warp) > 0)
        assert np.any(warped < 0)

    def test_the_tuple_unpacks_straight_into_unwarp_signal(self):
        x = self._chirp(1024, self.FS)
        t, back = unwarp_signal(
            *warp_signal(x, self.FS, self.RANGE, oversample=8),
            self.FS, self.RANGE)
        n = min(back.size, x.size)
        err = (np.linalg.norm(back[:n] - x[:n]) / np.linalg.norm(x[:n]))
        assert err < 0.05, f"round trip relative error {err:.4%}"
        assert t[0] == pytest.approx(self.RANGE / 1500.0)

    def test_swapping_the_pair_destroys_the_round_trip(self):
        """What flipping ``warp_signal``'s return order would have cost."""
        x = self._chirp(1024, self.FS)
        warped, t_warp = warp_signal(x, self.FS, self.RANGE, oversample=8)
        _, good = unwarp_signal(warped, t_warp, self.FS, self.RANGE)
        _, swapped = unwarp_signal(t_warp, warped, self.FS, self.RANGE)
        n = min(good.size, swapped.size, x.size)
        ref = np.linalg.norm(x[:n])
        assert np.linalg.norm(good[:n] - x[:n]) / ref < 0.05
        assert np.linalg.norm(swapped[:n] - x[:n]) / ref > 0.5

    def test_the_documented_order_is_the_one_unwarp_declares(self):
        assert list(inspect.signature(unwarp_signal).parameters)[:2] == [
            'warped', 't_warp']
        lines = inspect.cleandoc(warp_signal.__doc__).splitlines()
        # A real numpydoc section header, not a substring: the underline has to
        # be on the next line and as long as the word.
        heads = [i for i, ln in enumerate(lines[:-1])
                 if ln.strip() == 'Returns'
                 and set(lines[i + 1].strip()) == {'-'}
                 and len(lines[i + 1].strip()) >= len('Returns')]
        assert len(heads) == 1, "warp_signal has no numpydoc Returns section"
        body = '\n'.join(lines[heads[0]:])
        # Named in the order they are returned, and the reason recorded with
        # them: prose elsewhere in the file does not survive an edit here.
        assert body.index('warped') < body.index('t_warp')
        assert 'unwarp_signal' in body


class TestWarpRoundTripErrorMatchesTheDocumentedFigures:
    """``warp_signal``'s ``oversample`` docstring quotes two round-trip error
    figures and a halving rate; both figures are measured here on the named
    signals, so a docstring number and the code cannot part company.

    The two differ by three orders of magnitude because the loss lives at the
    top of the band: white noise is broadband to Nyquist and loses the most on
    the coarser warped grid, while the band-limited modal transient the warp
    exists for loses almost nothing.
    """

    @staticmethod
    def _roundtrip(x, fs, r, oversample, c=1500.0):
        w, tw = warp_signal(x, fs, r, oversample=oversample, c=c)
        _, back = unwarp_signal(w, tw, fs, r, c=c)
        n = min(back.size, x.size)
        return float(np.linalg.norm(back[:n] - x[:n])
                     / np.linalg.norm(x[:n]))

    @staticmethod
    def _transient():
        n = 1024
        t = np.arange(n) / FS
        return (np.sin(2 * np.pi * 40 * t)
                + 0.5 * np.sin(2 * np.pi * 120 * t)) * np.hanning(n)

    @pytest.mark.parametrize('oversample, expected_pct', [(1, 0.0674),
                                                          (8, 0.0079)])
    def test_the_band_limited_transient_figures(self, oversample,
                                                expected_pct):
        got = 100.0 * self._roundtrip(self._transient(), FS, 500.0, oversample)
        assert got == pytest.approx(expected_pct, abs=5e-4)

    @pytest.mark.parametrize('oversample, lo_pct, hi_pct', [(1, 46.0, 61.0),
                                                            (8, 5.8, 8.2)])
    def test_the_white_noise_figures_over_a_grid_of_rates_and_ranges(
            self, oversample, lo_pct, hi_pct):
        x = np.random.default_rng(1).standard_normal(2048)
        errs = [100.0 * self._roundtrip(x, fs, r, oversample)
                for fs in (2000.0, 5000.0, 10000.0)
                for r in (100.0, 500.0, 1000.0, 5000.0, 20000.0)]
        assert min(errs) >= lo_pct and max(errs) <= hi_pct

    def test_the_error_roughly_halves_with_each_doubling(self):
        x = np.random.default_rng(0).standard_normal(2048)
        errs = [self._roundtrip(x, 2000.0, 1000.0, k)
                for k in (1, 2, 4, 8, 16, 32)]
        ratios = [a / b for a, b in zip(errs, errs[1:])]
        assert all(1.8 <= q <= 2.2 for q in ratios), ratios


class TestModalGroupVelocityRefusesAFlatWavenumberAxis:
    """``vg = d(omega)/d(kr)``, so a step of zero in ``k_horizontal`` divides
    by zero and returns ``inf`` — announced by nothing but numpy's own
    "divide by zero encountered in divide", which names no input the caller
    supplied.

    A propagating mode's horizontal wavenumber rises strictly with frequency,
    so a flat step is not a dispersion curve. The check is per-sample rather
    than "is the whole axis constant": a *locally* flat segment is the same
    defect, and it produced a partially-``inf`` result that looks usable.

    Exact zero is the boundary this class owns. A *falling* step is refused by
    :class:`TestModalGroupVelocityRefusesAWavenumberAxisThatFalls` below, with
    its own message, so the two guards partition the axis with no gap.
    """

    FREQ = np.linspace(100.0, 4000.0, 64)

    def test_a_constant_axis_is_refused_and_names_the_samples(self):
        with pytest.raises(ConfigurationError) as exc:
            modal_group_velocity(self.FREQ, np.full(64, 64.0))
        message = str(exc.value)
        assert 'modal_group_velocity' in message
        assert 'k_horizontal is flat in frequency' in message
        assert '64 of 64 frequency sample(s)' in message
        assert 'first at index 0' in message

    def test_a_locally_flat_segment_is_refused_and_located(self):
        kr = np.concatenate([np.linspace(0.5, 6.0, 32), np.full(32, 6.0)])
        with pytest.raises(ConfigurationError) as exc:
            modal_group_velocity(self.FREQ, kr)
        message = str(exc.value)
        assert '32 of 64 frequency sample(s)' in message
        assert 'first at index 32' in message

    def test_a_flat_column_of_a_two_dimensional_set_is_refused(self):
        kr = np.stack([np.linspace(0.5, 12.0, 64), np.full(64, 3.0)], axis=1)
        with pytest.raises(ConfigurationError, match='flat in frequency'):
            modal_group_velocity(self.FREQ, kr)

    @pytest.mark.parametrize('kr', [np.linspace(0.5, 12.0, 64),
                                    np.linspace(1200.0, 1210.0, 64)])
    def test_a_rising_axis_is_accepted_and_emits_no_numpy_warning(self, kr):
        """The other side of the boundary, at two very different slopes.
        ``simplefilter('error')`` is what makes this the negative control for
        the guard: a bare numpy RuntimeWarning here would fail the test rather
        than pass silently."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            vg = modal_group_velocity(self.FREQ, kr)
        assert np.all(np.isfinite(vg))
        assert vg.shape == self.FREQ.shape

    def test_the_smallest_admissible_step_is_accepted(self):
        """The boundary is exact zero, not a tolerance: a tiny but non-zero
        step is the near-cutoff regime and has to keep working."""
        kr = np.linspace(1.0, 1.0 + 1e-9, 64)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            # A 1.6e-11 step at k_r = 1 spans 7e4 float64 spacings, which the
            # finite-difference resolution diagnostic reports; the filter is
            # narrowed to it so a bare numpy RuntimeWarning still fails here.
            warnings.filterwarnings(
                'ignore', message='modal_group_velocity: the wavenumber '
                                  'difference spans')
            vg = modal_group_velocity(self.FREQ, kr)
        assert np.all(np.isfinite(vg))
        with pytest.raises(ConfigurationError):
            modal_group_velocity(self.FREQ, np.linspace(1.0, 1.0, 64))

    def test_a_well_formed_two_dimensional_set_is_accepted(self):
        kr = np.stack([np.linspace(0.5, 12.0, 64),
                       np.linspace(0.4, 10.0, 64)], axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            vg = modal_group_velocity(self.FREQ, kr)
        assert vg.shape == (64, 2)
        assert np.all(np.isfinite(vg))


class TestModalGroupVelocityRefusesAWavenumberAxisThatFalls:
    """``v_g = d(omega)/d(k_r)`` is the speed energy travels at, so it is
    positive and bounded by the medium sound speeds; ``d(k_r)/d(omega) = 1/v_g``
    is therefore positive at every frequency and one mode's ``k_r`` rises
    strictly with frequency. Any falling step hands back a *negative* speed,
    which nothing downstream reads as an error — it is a travel-time
    denominator.

    The two ways a column can fall carry different remedies and are named
    separately:

    * **falls throughout** — a mode set stored against a descending frequency
      axis; reverse it. Every group velocity in the column comes back negative.
    * **doubles back** — a mixed or mode-shifted set; the samples across the
      turn come back negative while the rest of the curve stays ordinary, the
      same partially-poisoned shape the flat check refuses, but finite and
      unwarned.

    ``v_g`` itself is not monotonic: it dips through the Airy minimum (Jensen
    et al., *Computational Ocean Acoustics*, Sect. 2.4.4.4 and Fig. 2.28b,
    which pairs a monotonically falling phase velocity with a group velocity
    minimum). ``k_r`` still climbs there, so an Airy dip must be accepted — the
    guard constrains ``k_r`` and never ``v_g``.
    """

    FREQ = np.linspace(100.0, 4000.0, 64)
    # A ramp of exactly 1/8 per sample: 2 x the step is 0.25, representable,
    # so the perturbations below land on the boundary exactly rather than near
    # it. np.gradient's interior formula is (kr[i+1] - kr[i-1]) / 2.
    RAMP = 0.125 * np.arange(64.0)
    TWICE_STEP = 0.25
    TINY = 2.0 ** -30

    def _dipped(self, drop):
        """The ramp with sample 32 pulled ``drop`` below where it belongs, so
        the central gradient at index 31 is ``(0.25 - drop) / 2``."""
        kr = self.RAMP.copy()
        kr[32] -= drop
        return kr

    def test_a_curve_that_turns_once_is_refused_and_located(self):
        kr = self._dipped(self.TWICE_STEP + self.TINY)
        with pytest.raises(ConfigurationError) as exc:
            modal_group_velocity(self.FREQ, kr)
        message = str(exc.value)
        assert 'modal_group_velocity' in message
        assert 'k_horizontal doubles back in frequency' in message
        assert '1 of 1 mode column(s)' in message
        # Index 31 is the sample whose gradient turns negative, i.e. the
        # one whose returned group velocity would have been negative.
        assert f'first turning at index 31 ({self.FREQ[31]:g} Hz)' in message

    def test_a_turn_just_past_the_boundary_is_refused(self):
        """One side of the boundary: the gradient at index 31 is -2**-31, the
        smallest turn this construction can express."""
        kr = self._dipped(self.TWICE_STEP + self.TINY)
        assert np.gradient(kr)[31] == pytest.approx(-self.TINY / 2, rel=1e-12)
        with pytest.raises(ConfigurationError, match='doubles back'):
            modal_group_velocity(self.FREQ, kr)

    def test_a_dip_just_short_of_the_boundary_is_accepted(self):
        """The other side: the gradient at index 31 is +2**-31 — a barely
        rising curve, which is the near-cutoff regime and has to keep working.
        ``simplefilter('error')`` makes a bare numpy RuntimeWarning a failure.
        """
        kr = self._dipped(self.TWICE_STEP - self.TINY)
        assert np.gradient(kr)[31] == pytest.approx(self.TINY / 2, rel=1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            vg = modal_group_velocity(self.FREQ, kr)
        assert np.all(np.isfinite(vg))
        assert vg[31] > 0.0

    def test_the_boundary_itself_is_the_flat_guard_not_this_one(self):
        """At exactly twice the step the gradient is 0.0, so the two guards
        meet with no gap: neither side of the turn is left unchecked."""
        kr = self._dipped(self.TWICE_STEP)
        assert np.gradient(kr)[31] == 0.0
        with pytest.raises(ConfigurationError, match='flat in frequency'):
            modal_group_velocity(self.FREQ, kr)

    def test_a_turning_column_of_a_two_dimensional_set_is_refused(self):
        kr = np.stack([self.RAMP, self._dipped(self.TWICE_STEP + self.TINY)],
                      axis=1)
        with pytest.raises(ConfigurationError) as exc:
            modal_group_velocity(self.FREQ, kr)
        assert '1 of 2 mode column(s)' in str(exc.value)

    def test_two_modes_stacked_on_the_frequency_axis_are_refused(self):
        """The malformed array the flat guard's message already warns about,
        in the shape that stays finite: mode 1's curve for the low half and
        mode 2's (smaller k_r) for the high half, so k_r steps down at the
        join."""
        kr = np.concatenate([np.linspace(4.0, 8.0, 32),
                             np.linspace(2.0, 6.0, 32)])
        with pytest.raises(ConfigurationError, match='doubles back'):
            modal_group_velocity(self.FREQ, kr)

    def test_a_group_velocity_minimum_is_accepted(self):
        """The Airy phase: ``v_g`` falls to an interior minimum and climbs
        again, while ``k_r`` rises throughout. Built by integrating
        ``d(k_r)/d(omega) = 1/v_g`` over a dipped ``v_g``, so the curve is a
        physically admissible dispersion by construction.
        """
        f = np.linspace(20.0, 400.0, 2048)
        omega = 2 * np.pi * f
        target = 1500.0 - 60.0 * np.exp(-((f - 60.0) / 25.0) ** 2)
        kr = np.concatenate(
            [[0.0], np.cumsum(np.diff(omega) / target[1:])]) + omega[0] / 1500.0
        assert np.all(np.diff(kr) > 0)                    # k_r never turns
        vg = modal_group_velocity(f, kr)
        assert np.all(vg > 0)
        # v_g is NOT monotonic: it dips and recovers.
        assert vg.argmin() not in (0, vg.size - 1)
        assert vg.min() == pytest.approx(1440.0, abs=1.0)
        assert vg[-1] == pytest.approx(1500.0, abs=1.0)

    # ── falls throughout ────────────────────────────────────────────────

    def test_an_axis_that_falls_throughout_is_refused_and_named_as_falling(
            self):
        """Not the doubling-back message: the remedy differs. Every group
        velocity in the column would be negative, and the cause is almost
        always a mode set stored against a descending frequency axis."""
        with pytest.raises(ConfigurationError) as exc:
            modal_group_velocity(self.FREQ, self.RAMP[::-1].copy())
        message = str(exc.value)
        assert 'k_horizontal falls with frequency throughout' in message
        assert '1 of 1 mode column(s)' in message
        assert f'over {self.FREQ[0]:g}-{self.FREQ[-1]:g} Hz' in message
        assert 'reverse k_horizontal' in message
        assert 'doubles back' not in message

    def test_the_smallest_falling_slope_is_refused(self):
        """One side of the boundary: a slope of -2**-30 per sample. The guard
        is on the sign, not on a magnitude, so the shallowest descent is
        refused exactly like the steepest."""
        kr = 1000.0 - self.TINY * np.arange(64.0)
        assert np.all(np.gradient(kr) < 0)
        with pytest.raises(ConfigurationError, match='falls with frequency'):
            modal_group_velocity(self.FREQ, kr)

    def test_the_smallest_rising_slope_is_accepted(self):
        """The other side, at the same magnitude: +2**-30 per sample is a
        barely-dispersive mode and comes back finite and positive."""
        kr = 1000.0 + self.TINY * np.arange(64.0)
        assert np.all(np.gradient(kr) > 0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            # A 2**-30 step at k_r = 1000 spans 8192 float64 spacings, which
            # the finite-difference resolution diagnostic reports; the filter
            # is narrowed to it so a bare numpy RuntimeWarning still fails.
            warnings.filterwarnings(
                'ignore', message='modal_group_velocity: the wavenumber '
                                  'difference spans')
            vg = modal_group_velocity(self.FREQ, kr)
        assert np.all(np.isfinite(vg)) and np.all(vg > 0)

    def test_a_falling_axis_with_one_rising_step_is_named_as_doubling_back(
            self):
        """The boundary between the two messages, one variable apart: the same
        descending ramp with a single sample lifted so one gradient turns
        positive is a *turn*, not a wholly-falling curve."""
        kr = self.RAMP[::-1].copy()
        kr[32] += self.TWICE_STEP + self.TINY
        assert np.gradient(kr)[31] > 0 and np.any(np.gradient(kr) < 0)
        with pytest.raises(ConfigurationError) as exc:
            modal_group_velocity(self.FREQ, kr)
        assert 'doubles back in frequency' in str(exc.value)
        assert 'falls with frequency throughout' not in str(exc.value)

    def test_a_falling_column_of_a_two_dimensional_set_is_refused(self):
        kr = np.stack([self.RAMP + 1.0, self.RAMP[::-1].copy()], axis=1)
        with pytest.raises(ConfigurationError) as exc:
            modal_group_velocity(self.FREQ, kr)
        message = str(exc.value)
        assert 'falls with frequency throughout 1 of 2 mode column(s)' in message

    def test_a_flat_column_beside_a_falling_one_is_named_flat_first(self):
        """The three guards are ordered flat -> turning -> falling, so a set
        carrying more than one defect names the division-by-zero — the only
        one that produces ``inf`` rather than a finite wrong number."""
        kr = np.stack([np.full(64, 3.0), self.RAMP[::-1].copy()], axis=1)
        with pytest.raises(ConfigurationError, match='flat in frequency'):
            modal_group_velocity(self.FREQ, kr)

    def test_a_well_formed_rising_curve_is_accepted(self):
        """The negative control for this guard: an ordinary rising curve, 1-D
        and 2-D, under ``simplefilter('error')``."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            one_d = modal_group_velocity(self.FREQ, self.RAMP + 1.0)
            two_d = modal_group_velocity(
                self.FREQ, np.stack([self.RAMP + 1.0, self.RAMP * 0.9 + 1.0],
                                    axis=1))
        assert np.all(np.isfinite(one_d)) and one_d.shape == (64,)
        assert np.all(np.isfinite(two_d)) and two_d.shape == (64, 2)


class TestModalGroupVelocityReportsWhatTheFrequencyGridResolves:
    """``np.gradient`` over stored wavenumbers has the same optimum as any
    finite difference over quantized data: the truncation error falls as the
    frequency step squared while the storage floor ``spacing(k_r)/|Δk_r|``
    rises, so refining past the crossing makes the answer worse.

    Fixtures are an ideal waveguide (pressure-release surface, rigid seabed),
    where ``kz = (m+1/2)pi/D`` is exact and ``v_g = c**2 k_r/omega`` is closed
    form, so the reference carries no error of its own. Measured interior
    error against it: at 40 points over 30-70 Hz (the step the documentation
    exercises) float32 and float64 agree at 4.97e-04; at 400 points float64
    reaches 5.75e-06 while float32 stalls at 3.06e-05; at 4000 points float64
    reaches 5.83e-08 and float32 degrades to 3.41e-04.
    """

    DEPTH = 100.0
    SPEED = 1500.0

    def _sweep(self, n_points, quantize):
        f = np.linspace(30.0, 70.0, n_points)
        omega = 2.0 * np.pi * f
        kz = (np.arange(3) + 0.5) * np.pi / self.DEPTH
        kr = np.sqrt(omega[:, None] ** 2 / self.SPEED ** 2 - kz[None, :] ** 2)
        if quantize:
            kr = kr.astype(np.float32).astype(float)
        return f, kr

    def test_the_documented_forty_point_sweep_is_silent(self):
        f, kr = self._sweep(40, quantize=True)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            v_g = modal_group_velocity(f, kr)
        assert np.all(v_g > 0) and np.all(v_g < self.SPEED)

    def test_a_ten_times_finer_sweep_of_the_same_band_is_named(self):
        f, kr = self._sweep(400, quantize=True)
        with pytest.warns(UserWarning, match="wavenumber difference spans"):
            modal_group_velocity(f, kr)

    def test_the_finer_sweep_is_silent_when_the_wavenumbers_carry_float64_bits(self):
        """The trigger is the resolution of the data, not the size of the
        step: the same grid on unquantized wavenumbers says nothing."""
        f, kr = self._sweep(400, quantize=False)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            modal_group_velocity(f, kr)

    def test_the_finer_sweep_really_is_less_accurate_than_the_coarser_one(self):
        """The observable behind the warning, on the exact reference: refining
        the quantized sweep tenfold raises the error while the same refinement
        on unquantized wavenumbers lowers it."""
        def interior_error(n_points, quantize):
            f, kr = self._sweep(n_points, quantize)
            omega = 2.0 * np.pi * f
            exact = self.SPEED ** 2 * kr / omega[:, None]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                v_g = modal_group_velocity(f, kr)
            return float(np.max(np.abs(v_g[1:-1] - exact[1:-1])
                                / exact[1:-1]))

        assert interior_error(4000, True) > 5.0 * interior_error(400, True)
        # Truncation falls as the step squared: a tenfold refinement of the
        # unquantized sweep buys the 98x this asserts as at least 50x.
        assert interior_error(4000, False) < 0.02 * interior_error(400, False)


class TestModalGroupVelocityRemedyIsMeasuredNotPrescribed:
    """The storage floor says how much the quantization of ``k_r`` costs; it
    does not say which way to move the frequency grid, because the truncation
    error it trades against turns on ``|d2 k_r/d omega2|``. A step read off the
    floor alone was worse than the grid in hand in **14 of 30** firings across
    the five ideal waveguides below, by up to 16x.

    What a whole sweep can do is walk: decimating by ``K`` multiplies the
    truncation error by ``K**2`` and divides the floor by ``K``, so the widest
    spacing whose answer still agrees with its doubly-decimated self to within
    the floor is found without ever estimating the second derivative, and one
    doubling back from it is still on the storage-limited side. Over the same
    30 firings that step was **never worse** than the grid in hand and 3.3x
    better at the median.

    Fixtures are ideal waveguides (pressure-release surface, rigid seabed),
    where ``kz = (n-1/2)pi/D`` is exact and ``v_g = c**2 k_r/omega`` is closed
    form, so the error being compared carries no reference error of its own.
    """

    #: (water depth m, sound speed m/s, mode number, f_lo Hz, f_hi Hz)
    GUIDES = {
        'D70': (70.0, 1520.0, 3, 40.0, 90.0),
        'D100': (100.0, 1500.0, 1, 30.0, 70.0),
        'D200': (200.0, 1490.0, 8, 40.0, 95.0),
    }

    def _sweep(self, guide, df):
        depth, speed, mode, f_lo, f_hi = self.GUIDES[guide]
        kz = (mode - 0.5) * np.pi / depth
        f = np.linspace(f_lo, f_hi, int(round((f_hi - f_lo) / df)) + 1)
        omega = 2.0 * np.pi * f
        kr = np.sqrt(omega ** 2 / speed ** 2 - kz ** 2)
        exact = speed ** 2 * kr / omega
        return f, kr.astype(np.float32).astype(float), exact

    def _error(self, guide, df):
        f, kr, exact = self._sweep(guide, df)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            v_g = modal_group_velocity(f, kr)
        interior = slice(1, -1)
        return float(np.max(np.abs(v_g[interior] - exact[interior])
                            / exact[interior]))

    def _message(self, guide, df):
        f, kr, _ = self._sweep(guide, df)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            modal_group_velocity(f, kr)
        return next((str(w.message) for w in caught
                     if 'wavenumber difference spans' in str(w.message)), None)

    def test_a_far_too_fine_grid_is_told_how_much_coarser_to_go(self):
        message = self._message('D70', 0.01)
        assert message is not None
        assert 'x coarser' in message and 'Hz)' in message

    def test_a_grid_at_the_optimum_is_told_not_to_widen(self):
        message = self._message('D70', 0.2)
        assert message is not None
        assert 'Widening this grid is not indicated' in message
        assert 'x coarser' not in message

    @pytest.mark.parametrize("guide", ['D70', 'D100', 'D200'])
    @pytest.mark.parametrize("df", [0.05, 0.01])
    def test_the_recommended_grid_is_never_worse_than_the_one_in_hand(
            self, guide, df):
        """The property that makes the remedy safe to follow: at these six
        points the walked step is 0.01x-0.52x the error of the grid in hand,
        while a step read off the floor alone is 3.0x and 3.4x WORSE at two of
        them."""
        import re
        message = self._message(guide, df)
        assert message is not None
        found = re.search(r'spacing is (\d+)x coarser', message)
        if found is None:
            return                       # told not to widen: nothing to follow
        recommended = int(found.group(1)) * df
        assert self._error(guide, recommended) <= self._error(guide, df)

    def test_a_grid_far_past_the_optimum_is_walked_a_long_way_back(self):
        from uacpy.core._finite_difference import coarser_step_multiple
        f, kr, _ = self._sweep('D70', 0.002)
        assert coarser_step_multiple(2.0 * np.pi * f, kr) >= 16

    def test_a_grid_at_the_optimum_is_not_walked_at_all(self):
        from uacpy.core._finite_difference import coarser_step_multiple
        f, kr, _ = self._sweep('D70', 0.2)
        assert coarser_step_multiple(2.0 * np.pi * f, kr) == 1

    def test_a_grid_too_short_to_walk_is_left_alone(self):
        from uacpy.core._finite_difference import coarser_step_multiple
        f, kr, _ = self._sweep('D70', 0.002)
        assert coarser_step_multiple(2.0 * np.pi * f[:4], kr[:4]) == 1


class TestStorageSpacingReadsTheBitsNotTheContainer:
    """The floor is quoted against the grid the values sit on, so that grid has
    to be identified from the values. A float64 array of exact eighths survives
    a float32 round trip while carrying full float64 precision; reporting it as
    float32 storage put a stated grid nine decades from the truth."""

    def test_a_float32_record_is_measured_against_the_float32_spacing(self):
        from uacpy.core._finite_difference import storage_spacing
        kr = np.array([0.4180, 0.4157, 0.4118, 0.4062, 0.3989, 0.3897,
                       0.3785]).astype(np.float32).astype(float)
        assert np.allclose(storage_spacing(kr), np.spacing(kr.astype(np.float32)))

    def test_a_dyadic_float64_ramp_is_measured_against_the_float64_spacing(self):
        from uacpy.core._finite_difference import storage_spacing
        kr = 1.0 + np.arange(64) / 128.0
        assert np.array_equal(kr.astype(np.float32).astype(float), kr)
        assert np.allclose(storage_spacing(kr), np.spacing(kr))

    def test_a_dyadic_float64_ramp_raises_no_storage_warning(self):
        kr = 1.0 + np.arange(64) / 128.0
        f = np.linspace(30.0, 70.0, kr.size)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            modal_group_velocity(f, kr)
