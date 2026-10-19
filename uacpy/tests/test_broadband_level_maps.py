"""A signal with bandwidth or duration reaches a level a tone does not.

Two quantities, two questions, and the tests keep them apart:

* :meth:`Field.broadband_loss` averages ``|H|^2`` over the band and returns a
  loss — Ainslie's broadband propagation loss (sect. 11.3.3, Eq. 11.46), which
  is Abraham's ``L_p`` (sect. 3.2.4.2) once the source spectrum weights it.
  It says how much of the continuous-wave interference the band survives.
* :meth:`Field.sound_exposure_level` integrates ``p(t)^2`` and returns a level
  — Abraham's energy flux density integral (sect. 3.2.1.5), ISO 18405's sound
  exposure. It says how much energy one transmission delivers.

Each is pinned against a closed form, because both are averages and an average
is the kind of quantity that looks plausible while being wrong by a constant.
"""

import re

import numpy as np
import pytest
import warnings

from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Field
from uacpy.core.results.quantities import is_loss
from uacpy.acoustic_signal import tone_burst
from uacpy.tests.conftest import recorded_warnings

SPEED = 1500.0
RATE = 8000.0


def two_path_field(amplitude=0.7, delay=5e-3, frequencies=None):
    """``H = 1 + a exp(-2 pi i f tau)`` on one cell — two paths, one fringe."""
    f = (np.arange(1000.0, 1200.0, 0.5) if frequencies is None
         else np.asarray(frequencies, dtype=float))
    h = 1.0 + amplitude * np.exp(-2j * np.pi * f * delay)
    return Field(data=h.reshape(1, 1, f.size),
                 coords={'depth': [10.0], 'range': [100.0], 'frequency': f},
                 frequencies=f)


def free_field(ranges=(100.0, 200.0), df=25.0, freq_max=4000.0):
    """``H = exp(-i k r) / r`` — spherical spreading, no boundaries.

    The NEGATIVE exponent is uacpy's delay convention, the one
    ``Arrivals.transfer_function`` writes; ``exp(+ikr)`` is an
    incoming wave and puts the pulse before its travel time.
    """
    f = np.arange(df, freq_max, df)
    r = np.asarray(ranges, dtype=float)
    h = np.exp(-2j * np.pi * np.outer(r, f) / SPEED) / r[:, None]
    return Field(data=h[None, ...],
                 coords={'depth': [10.0], 'range': r, 'frequency': f},
                 frequencies=f)


class TestTheBandAverageIsTheLossABandwidthSees:
    def test_a_band_spanning_many_fringes_leaves_the_incoherent_sum(self):
        # Averaging 1 + a^2 + 2a cos(2 pi f tau) over whole fringes kills the
        # cosine, so the band average is the energy sum of the two paths.
        a = 0.7
        loss = float(two_path_field(amplitude=a).broadband_loss().tl[0, 0])
        assert loss == pytest.approx(-10.0 * np.log10(1.0 + a ** 2), abs=1e-3)

    def test_a_single_frequency_band_returns_the_continuous_wave_loss(self):
        # The narrowband limit is the identity that makes the two quantities
        # one family: with no band to average, this IS transmission loss.
        f = np.array([1000.0, 1000.0])
        field = two_path_field(frequencies=f)
        cw = float(field.tl[0, 0, 0])
        got = float(field.broadband_loss().tl[0, 0])
        assert got == pytest.approx(cw, abs=1e-9)

    def test_a_spectrum_on_one_bin_returns_that_bin_alone(self):
        # The weights are |spectrum|^2, so all the weight on one bin must
        # reproduce that bin's continuous-wave loss and none of its neighbours'.
        field = two_path_field()
        weights = np.zeros(field.n_frequencies)
        weights[40] = 1.0
        assert float(field.broadband_loss(source_spectrum=weights).tl[0, 0]) == \
            pytest.approx(float(field.tl[0, 0, 40]), abs=1e-9)

    def test_the_spectrum_scale_cancels(self):
        field = two_path_field()
        shape = np.hanning(field.n_frequencies) + 0.1
        one = float(field.broadband_loss(source_spectrum=shape).tl[0, 0])
        scaled = float(field.broadband_loss(source_spectrum=1e6 * shape).tl[0, 0])
        assert one == pytest.approx(scaled, abs=1e-12)

    def test_a_complex_spectrum_weighs_by_its_magnitude(self):
        field = two_path_field()
        shape = np.hanning(field.n_frequencies) + 0.1
        phased = shape * np.exp(1j * np.linspace(0.0, 7.0, shape.size))
        assert float(field.broadband_loss(source_spectrum=phased).tl[0, 0]) == \
            pytest.approx(float(field.broadband_loss(source_spectrum=shape).tl[0, 0]),
                          abs=1e-12)

    def test_it_averages_whichever_axis_the_frequencies_are_on(self):
        # The frequency axis is last on a model's grid and first on a
        # hand-built one; averaging the wrong axis returns the right shape.
        a, delay = 0.7, 5e-3
        want = -10.0 * np.log10(1.0 + a ** 2)
        f = np.arange(1000.0, 1200.0, 0.5)
        h = 1.0 + a * np.exp(-2j * np.pi * f * delay)
        leading = Field(data=np.moveaxis(
            np.broadcast_to(h, (1, 1, f.size)).copy(), -1, 0),
            coords={'frequency': f, 'depth': [0.0], 'range': [1.0]},
            frequencies=f)
        out = leading.broadband_loss()
        assert list(out.coords) == ['depth', 'range']
        assert float(out.tl[0, 0]) == pytest.approx(want, abs=1e-3)

    def test_a_source_depth_axis_survives_the_average(self):
        a, delay = 0.7, 5e-3
        f = np.arange(1000.0, 1200.0, 0.5)
        h = 1.0 + a * np.exp(-2j * np.pi * f * delay)
        z = np.array([5.0, 10.0])
        stack = Field(
            data=np.broadcast_to(h, (z.size, 1, 1, f.size)).copy(),
            coords={'source_depth': z, 'depth': [0.0], 'range': [1.0],
                    'frequency': f}, frequencies=f, source_depths=z)
        out = stack.broadband_loss()
        assert list(out.coords) == ['source_depth', 'depth', 'range']
        assert float(out.tl[0, 0, 0]) == pytest.approx(
            -10.0 * np.log10(1.0 + a ** 2), abs=1e-3)

    def test_the_phase_of_a_spectrum_on_the_axis_does_not_reach_the_answer(
            self):
        """``spectrum=`` enters only through ``|spectrum|``.

        Scoped deliberately to the ``spectrum=`` route, where it is true by
        the formula. It does NOT generalise to waveforms: ``waveform=``
        evaluates the continuous DTFT on the field's axis, and off a
        waveform's own DFT grid that is not fixed by its ``|rfft|``.
        Whether phase reaches the answer is decided by grid ALIGNMENT, not
        by bandwidth: ``|DTFT| == |rfft|`` on the waveform's own bins,
        spaced ``1/T``, so when ``T*df`` is an integer the field's axis is
        a subset of them and the answer is phase-blind. See
        ``test_phase_reaches_the_answer_only_off_the_waveforms_own_grid``.
        ``test_different_signals_get_different_losses`` pins that side.
        """
        field = two_path_field()
        rng = np.random.default_rng(0)
        magnitude = np.hanning(field.n_frequencies) + 0.05
        phased = magnitude * np.exp(
            1j * rng.uniform(0.0, 2.0 * np.pi, magnitude.size))
        assert float(field.broadband_loss(source_spectrum=phased).tl[0, 0]) == \
            pytest.approx(
                float(field.broadband_loss(source_spectrum=magnitude).tl[0, 0]),
                abs=1e-9)

    def test_it_converges_to_the_continuous_wave_loss_as_the_band_narrows(self):
        """The narrowband limit, which makes this a generalisation of TL.

        Checked as a sequence rather than at one width: a single narrow band
        agreeing proves nothing about the trend, and the trend is the claim.
        """
        field = two_path_field()
        # 1100 Hz sits near a cancellation of this path pair, where the loss
        # is most sensitive to the averaging — an insensitive point would
        # converge whatever the code did.
        centre = 1100.0
        cw = float(field.at(frequency=centre).tl[0, 0])
        errors = []
        for width in (200.0, 50.0, 10.0, 2.0, 0.5):
            band = field.window(frequency=(centre - width / 2.0,
                                           centre + width / 2.0))
            errors.append(abs(float(band.broadband_loss().tl[0, 0]) - cw))
        assert errors == sorted(errors, reverse=True), errors
        assert errors[0] > 10.0                  # the wide band is nowhere near
        assert errors[-1] == pytest.approx(0.0, abs=1e-12)   # one bin is exact

    def test_the_result_is_a_transmission_loss_map(self):
        out = two_path_field().broadband_loss()
        assert list(out.coords) == ['depth', 'range']
        assert (out.kind, out.unit) == ('pressure', 'dB')
        assert is_loss(out.kind)
        assert out.tl.shape == (1, 1)

    def test_the_pinned_frequency_is_the_band_centre(self):
        field = two_path_field()
        f = np.asarray(field.coords['frequency'])
        assert field.broadband_loss().pinned['frequency'] == \
            pytest.approx(float(np.mean(f)))

    def test_both_reducers_record_the_band_they_collapsed(self):
        """A reduced map narrows its identity to one centroid, so the band
        it averaged has nowhere else to live.

        Without it, two maps over different bands about the same centre are
        indistinguishable afterwards, and a plotter captioning one has only
        the centroid — which the map is not. Asserted on BOTH reducers
        because the first attempt recorded it on one and left the other,
        which is the asymmetry an earlier round was about.
        """
        f = np.arange(800.0, 1201.0, 5.0)
        r = np.array([100.0])
        h = (np.exp(-2j * np.pi * np.outer(r, f) / SPEED) / r[:, None])
        field = Field(data=h[None, ...],
                      coords={'depth': [10.0], 'range': r, 'frequency': f},
                      frequencies=f)
        waveform = tone_burst(1000.0, 20, sample_rate=RATE)[1]
        for low, high in ((975.0, 1025.0), (800.0, 1200.0)):
            band = field.window(frequency=(low, high))
            assert band.broadband_loss().band == (low, high)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                exposure = band.sound_exposure_level(source_waveform=waveform,
                                                     sample_rate=RATE)
            assert exposure.band == (low, high)
            # and the identity really is narrowed, so band is the only
            # place the span survives
            assert band.broadband_loss().n_frequencies == 1

    def test_a_cell_nothing_reached_stays_undefined(self):
        # Never substitute: a no-data cell must not average to a level.
        field = two_path_field()
        data = np.concatenate([field.data, np.full_like(field.data, np.nan)],
                              axis=0)
        widened = Field(data=data,
                        coords={'depth': [10.0, 20.0], 'range': [100.0],
                                'frequency': field.coords['frequency']},
                        frequencies=field.coords['frequency'])
        out = widened.broadband_loss().tl
        assert np.isfinite(out[0, 0]) and np.isnan(out[1, 0])

    def test_a_zero_weight_bin_carries_a_no_data_cell_forward(self):
        # The accumulation skips zero-weight bins to save the multiply, so
        # the NaN that ``0 * nan`` would have propagated has to be carried
        # deliberately — otherwise a cell the model never reached at one
        # frequency averages to a confident level.
        field = two_path_field()
        data = field.data.copy()
        data[0, 0, 7] = np.nan
        holed = Field(data=data, coords=dict(field.coords),
                      frequencies=field.coords['frequency'])
        weights = np.ones(field.n_frequencies)
        weights[7] = 0.0
        assert np.isnan(holed.broadband_loss(source_spectrum=weights).tl[0, 0])

    def test_a_no_data_frequency_is_not_averaged_away(self):
        field = two_path_field()
        data = field.data.copy()
        data[0, 0, 7] = np.nan
        holed = Field(data=data, coords=dict(field.coords),
                      frequencies=field.coords['frequency'])
        assert np.isnan(holed.broadband_loss().tl[0, 0])

    def test_a_field_with_no_band_is_refused(self):
        field = two_path_field()
        cw = field.isel(frequency=0)
        with pytest.raises(ConfigurationError, match="frequency axis"):
            cw.broadband_loss()

    def test_a_field_that_lost_its_phase_is_refused(self):
        # A TL grid averaged in place is an INCOHERENT sum (Ainslie Eq. 11.47),
        # which is the approximation this method exists to avoid being.
        with pytest.raises(ConfigurationError, match="complex"):
            two_path_field().to_dB().broadband_loss()

    def test_the_spectrum_of_a_received_trace_is_refused(self):
        # A pressure trace transforms to S·H (Pa·s), whose band mean
        # carries the source spectrum; an impulse response transforms to H.
        fs = 8000.0
        t = np.arange(800) / fs
        x = np.zeros((1, 1, t.size))
        x[0, 0, 40], x[0, 0, 120] = 1.0, 0.5
        coords = {'depth': [10.0], 'range': [100.0], 'time': t}
        trace = Field(data=x, coords=coords, kind='pressure', unit='Pa')
        with pytest.raises(ConfigurationError,
                           match=r"^Field.broadband_loss: this field is the "
                                 r"spectrum S·H of a received pressure "
                                 r"trace \(unit 'Pa·s'\)"):
            trace.to_transfer_function().broadband_loss()
        response = Field(data=x * fs, coords=coords, kind='impulse_response')
        assert np.isfinite(
            response.to_transfer_function().broadband_loss().data).all()

    def test_a_spectrum_off_the_frequency_axis_is_refused(self):
        field = two_path_field()
        with pytest.raises(ConfigurationError, match="frequency axis has"):
            field.broadband_loss(source_spectrum=np.ones(field.n_frequencies + 1))

    def test_a_spectrum_with_a_hole_in_it_is_refused(self):
        # The one guard the mutation sweep found unpinned: without it a NaN
        # weight propagates to every cell and the map reads NaN everywhere,
        # which looks like a model that reached nothing.
        field = two_path_field()
        for bad in (np.nan, np.inf):
            weights = np.ones(field.n_frequencies)
            weights[3] = bad
            with pytest.raises(ConfigurationError, match="finite"):
                field.broadband_loss(source_spectrum=weights)

    def test_the_pinned_frequency_follows_the_spectrum(self):
        # A 500 Hz burst's map is a map of 500 Hz. Labelling it with the
        # axis midpoint is a legal frequency and the wrong one — the same
        # defect Field.window's identity narrowing exists to prevent.
        f = np.arange(25.0, 4000.0, 25.0)
        field = Field(data=np.ones((1, 1, f.size), complex),
                      coords={'depth': [0.0], 'range': [1.0],
                              'frequency': f}, frequencies=f)
        burst = tone_burst(500.0, 5, sample_rate=8000.0)[1]
        weighted = field.broadband_loss(source_waveform=burst, sample_rate=8000.0)
        assert weighted.pinned['frequency'] == pytest.approx(500.0, abs=25.0)
        # and a white source still lands on the band centre
        assert field.broadband_loss().pinned['frequency'] == pytest.approx(
            float(f.mean()), abs=1e-9)

    def test_a_spectrum_carrying_no_energy_is_refused(self):
        field = two_path_field()
        with pytest.raises(ConfigurationError, match="no energy"):
            field.broadband_loss(source_spectrum=np.zeros(field.n_frequencies))

    @pytest.mark.parametrize('bad,match', [
        (lambda n: np.ones((n, 1)), 'must be 1-D'),
        (lambda n: np.ones(n + 1), r'has \d+ samples but the frequency axis'),
        (lambda n: np.r_[np.ones(n - 1), np.nan], 'must be finite'),
        (lambda n: np.zeros(n), 'holds no energy'),
    ])
    def test_each_refusal_names_the_argument_the_caller_passed(self, bad,
                                                               match):
        # One set of checks (broadband_loss's) serves the method
        # and the array function; each names source_spectrum to the Field's
        # caller and weights to the function's.
        from uacpy.acoustic_signal import broadband_loss
        field = two_path_field()
        spectrum = bad(field.n_frequencies)
        with pytest.raises(ConfigurationError,
                           match=rf"broadband_loss: source_spectrum {match}"):
            field.broadband_loss(source_spectrum=spectrum)
        with pytest.raises(ConfigurationError, match=rf"weights {match}"):
            broadband_loss(np.asarray(field.data),
                           np.abs(spectrum) ** 2)


class TestAnySignalGetsItsOwnTransmissionLoss:
    """``broadband_loss(waveform=...)`` — the loss for a signal, not a band.

    The fixture is a two-path channel, so ``|H|`` varies across the band and
    the weight actually changes the answer; a free field would return the
    same spreading loss for every spectrum and pin nothing.
    """

    RATE = 8000.0

    @staticmethod
    def echoing_channel():
        f = np.arange(25.0, 4000.0, 25.0)
        r = np.array([100.0])
        h = (np.exp(-2j * np.pi * np.outer(r, f) / SPEED)
             * (1.0 + 0.6 * np.exp(-2j * np.pi * f * 4e-3)) / r[:, None])
        return Field(data=h[None, ...],
                     coords={'depth': [10.0], 'range': r, 'frequency': f},
                     frequencies=f)

    def test_phase_reaches_the_answer_only_off_the_waveforms_own_grid(self):
        """Grid alignment decides it, not bandwidth.

        ``|DTFT|`` equals ``|rfft|`` only on the waveform's own bins, spaced
        ``1/T``. When ``T*df`` is an integer the field's axis is a subset of
        those bins, so re-phasing a waveform cannot move the answer at all.
        Off that alignment it can.

        This sentence was wrong three times before this test existed — first
        quoting a single draw, then a median from the wrong fixture, then
        attributing it to bandwidth. The bandwidth story dies on 21 cycles,
        which is NARROWER than 20 (34.2 vs 35.6 Hz) and yet spreads where 20
        does not, so the pairing below is the load-bearing part.
        """
        rng = np.random.default_rng(0)
        field = self.echoing_channel()
        df = float(np.diff(np.asarray(field.coords['frequency']))[0])

        def spread(cycles):
            waveform = tone_burst(500.0, cycles, sample_rate=self.RATE)[1]
            spectrum = np.fft.rfft(waveform)
            remixed = np.fft.irfft(
                np.abs(spectrum) * np.exp(
                    1j * rng.uniform(0.0, 2.0 * np.pi, spectrum.size)),
                n=waveform.size)
            straight = field.broadband_loss(source_waveform=waveform,
                                            sample_rate=self.RATE)
            phased = field.broadband_loss(source_waveform=remixed,
                                          sample_rate=self.RATE)
            return abs(float(straight.tl[0, 0]) - float(phased.tl[0, 0]))

        for cycles in (20, 40):                     # T*df = 1.0, 2.0
            assert (tone_burst(500.0, cycles, sample_rate=self.RATE)[1].size
                    / self.RATE * df) == pytest.approx(round(
                        tone_burst(500.0, cycles, sample_rate=self.RATE)[1].size
                        / self.RATE * df))
            assert spread(cycles) == pytest.approx(0.0, abs=1e-9), cycles
        for cycles in (19, 21, 41):                 # 0.95, 1.05, 2.05
            assert spread(cycles) > 0.01, cycles

    def test_different_signals_get_different_losses(self):
        field = self.echoing_channel()
        losses = [
            float(field.broadband_loss(
                source_waveform=w, sample_rate=self.RATE).tl[0, 0])
            for w in (tone_burst(500.0, 5, sample_rate=self.RATE)[1],
                      tone_burst(625.0, 5, sample_rate=self.RATE)[1],
                      tone_burst(500.0, 20, sample_rate=self.RATE)[1])]
        assert len(set(round(x, 3) for x in losses)) == 3, losses
        assert max(losses) - min(losses) > 1.0

    def test_the_energy_sonar_equation_closes_exactly(self):
        """SEL = ESL - TPL, the two routes checked against each other.

        **This pins their CONSISTENCY, not either one's correctness**, and
        the distinction is not academic — it has been fooled twice. Both
        sides read the same ``H`` on the same grid, so any error entering
        upstream of where they split moves both by the same amount and this
        stays green:

        * a record fold biased both by up to -9.27 dB (the aliased record's
          energy is not the true response's);
        * a band placed across two bin offsets biased both by 1.67 dB
          (``freqs[0]/df`` on the ``floor(x + 0.5)`` boundary).

        Neither was visible here. What catches those is a guard at the
        entry — ``sound_exposure_level``'s fold note and ``_synthesis_plan``'s
        placement check — not a wider tolerance on this assertion. Treat a
        green result here as "the two routes agree", never as "the number is
        right".
        """
        field = self.echoing_channel()
        source_level = 190.0
        for waveform in (tone_burst(500.0, 5, sample_rate=self.RATE)[1],
                         tone_burst(625.0, 5, sample_rate=self.RATE)[1],
                         tone_burst(500.0, 20, sample_rate=self.RATE)[1]):
            duration = waveform.size / self.RATE
            scaled = (waveform / np.sqrt(np.mean(waveform ** 2))
                      * 1e-6 * 10 ** (source_level / 20.0))
            loss = float(field.broadband_loss(
                source_waveform=scaled, sample_rate=self.RATE).tl[0, 0])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sel = float(field.sound_exposure_level(source_waveform=scaled,
                                                       sample_rate=self.RATE).dB[0, 0])
            energy_source_level = (source_level
                                   + 10.0 * np.log10(duration))
            assert sel == pytest.approx(energy_source_level - loss, abs=1e-6)

    def test_the_waveform_scale_cancels(self):
        field = self.echoing_channel()
        waveform = tone_burst(500.0, 5, sample_rate=self.RATE)[1]
        quiet = float(field.broadband_loss(
            source_waveform=waveform, sample_rate=self.RATE).tl[0, 0])
        loud = float(field.broadband_loss(
            source_waveform=1e7 * waveform, sample_rate=self.RATE).tl[0, 0])
        assert quiet == pytest.approx(loud, abs=1e-9)

    def test_an_interpolated_rfft_is_not_the_same_weight(self):
        """Why ``waveform=`` exists rather than leaving callers to FFT.

        Resampling an ``rfft`` onto the field's axis by interpolation is a
        convolution with a triangular kernel, not a resampling, and the two
        grids rarely coincide. Measured here at 0.13 dB on a plain burst —
        small, silent, and in the caller's own code where nothing checks it.
        """
        field = self.echoing_channel()
        freqs = np.asarray(field.coords['frequency'])
        waveform = tone_burst(500.0, 5, sample_rate=self.RATE)[1]
        interpolated = np.interp(
            freqs, np.fft.rfftfreq(waveform.size, 1.0 / self.RATE),
            np.abs(np.fft.rfft(waveform)))
        by_hand = float(field.broadband_loss(source_spectrum=interpolated).tl[0, 0])
        exact = float(field.broadband_loss(
            source_waveform=waveform, sample_rate=self.RATE).tl[0, 0])
        assert abs(by_hand - exact) > 0.1

    def test_a_waveform_and_a_spectrum_together_are_refused(self):
        field = self.echoing_channel()
        with pytest.raises(ConfigurationError, match="not both"):
            field.broadband_loss(
                source_spectrum=np.ones(field.n_frequencies),
                source_waveform=tone_burst(500.0, 5, sample_rate=self.RATE)[1],
                sample_rate=self.RATE)

    def test_a_waveform_without_a_sample_rate_is_refused(self):
        field = self.echoing_channel()
        with pytest.raises(ConfigurationError, match="sample_rate"):
            field.broadband_loss(
                source_waveform=tone_burst(500.0, 5, sample_rate=self.RATE)[1])

    def test_the_generators_time_signal_pair_is_refused(self):
        field = self.echoing_channel()
        with pytest.raises(ConfigurationError, match="1-D signal"):
            field.broadband_loss(source_waveform=tone_burst(500.0, 5, sample_rate=self.RATE),
                                 sample_rate=self.RATE)


class TestTheExposureLevelIsTheEnergyOneTransmissionDelivers:
    @staticmethod
    def burst():
        return tone_burst(500.0, 5, sample_rate=RATE)[1]

    @staticmethod
    def quiet(field, waveform, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return field.sound_exposure_level(source_waveform=waveform,
                                              sample_rate=RATE, **kw)

    @pytest.mark.parametrize('level', ['sel', 'weighted sel', 'peak'])
    @pytest.mark.parametrize('domain', ['transfer function', 'traces'])
    def test_a_no_data_cell_stays_nan_on_the_map(self, domain, level):
        # The plain level functions refuse a record holding NaN; on a map a
        # cell no model output reached is no data, NaN, and its neighbours
        # keep their levels.
        field = free_field(ranges=(100.0, 200.0))
        data = np.array(field.data, copy=True)
        data[0, 1, :] = np.nan
        field = field.replace(data=data)
        kw = {}
        if domain == 'traces':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                field = field.synthesize_time_series(
                    source_waveform=self.burst(), sample_rate=RATE)
        else:
            kw = dict(source_waveform=self.burst(), sample_rate=RATE)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if level == 'peak':
                out = field.peak_sound_pressure_level(**kw)
            else:
                out = field.sound_exposure_level(
                    weighting='LF' if level == 'weighted sel' else None,
                    **kw)
        values = np.asarray(out.data, dtype=float)
        assert np.isfinite(values[0, 0]) and np.isnan(values[0, 1])

    @pytest.mark.parametrize('level', ['sel', 'weighted sel', 'peak'])
    @pytest.mark.parametrize('domain', ['transfer function', 'traces'])
    @pytest.mark.parametrize('ref', [1e-6, 20e-6])
    def test_a_cell_nothing_reached_is_the_no_energy_marker(self, domain,
                                                           level, ref):
        # A zero H (or a trace zero throughout) received no energy: the map
        # says so with the marker no_energy_mask finds, -600 dB whatever the
        # reference, as at_source_level does, not a level a quiet cell
        # could read (-180 dB re 1 µPa, -206 dB re 20 µPa).
        from uacpy.acoustics import no_energy_mask
        field = free_field(ranges=(100.0, 200.0))
        data = np.array(field.data, copy=True)
        data[0, 1, :] = 0.0
        field = field.replace(data=data)
        kw = {}
        if domain == 'traces':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                field = field.synthesize_time_series(
                    source_waveform=self.burst(), sample_rate=RATE)
        else:
            kw = dict(source_waveform=self.burst(), sample_rate=RATE)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if level == 'peak':
                out = field.peak_sound_pressure_level(ref=ref, **kw)
            else:
                out = field.sound_exposure_level(
                    weighting='LF' if level == 'weighted sel' else None,
                    ref=ref, **kw)
        values = np.asarray(out.data, dtype=float)
        assert values[0, 1] == -600.0
        np.testing.assert_array_equal(no_energy_mask(values),
                                      [[False, True]])
        assert -200.0 < values[0, 0] < 200.0

    def test_a_flat_band_reproduces_the_source_energy_over_r_squared(self):
        # The whole unit contract in one assertion: a waveform in Pa at 1 m
        # through H = exp(ikr)/r must integrate to int u^2 dt / r^2.
        x = self.burst()
        field = free_field(ranges=(100.0,))
        want = 10.0 * np.log10(np.sum(x ** 2) / RATE / 100.0 ** 2 / 1e-12)
        got = float(self.quiet(field, x).dB[0, 0])
        assert got == pytest.approx(want, abs=1e-6)

    def test_a_band_taper_costs_energy_the_integral_is_defined_to_count(self):
        # Why the default is a flat band. 500 Hz sits an eighth of the way up
        # a 25 Hz-4 kHz band, where a Hann taper stands at 0.135, and the
        # energy follows its square.
        x = self.burst()
        field = free_field(ranges=(100.0,))
        flat = float(self.quiet(field, x, window=None).dB[0, 0])
        tapered = float(self.quiet(field, x, window='hann').dB[0, 0])
        assert tapered - flat == pytest.approx(-17.0, abs=0.1)

    def test_doubling_the_range_costs_spherical_spreading(self):
        sel = self.quiet(free_field(ranges=(100.0, 200.0)), self.burst()).dB
        assert float(sel[0, 0]) - float(sel[0, 1]) == \
            pytest.approx(20.0 * np.log10(2.0), abs=1e-6)

    def test_the_level_follows_the_waveform_amplitude_squared(self):
        field = free_field(ranges=(100.0,))
        x = self.burst()
        quiet = float(self.quiet(field, x).dB[0, 0])
        loud = float(self.quiet(field, 3.0 * x).dB[0, 0])
        assert loud - quiet == pytest.approx(20.0 * np.log10(3.0), abs=1e-6)

    def test_transmitting_the_pulse_twice_adds_three_decibels(self):
        # Energy adds where pressure would interfere: two separated copies in
        # one record carry twice the energy whatever their relative phase.
        field = free_field(ranges=(100.0,))
        x = self.burst()
        once = float(self.quiet(field, x).dB[0, 0])
        twice = float(self.quiet(field, np.concatenate(
            [x, np.zeros(40), x])).dB[0, 0])
        assert twice - once == pytest.approx(10.0 * np.log10(2.0), abs=1e-6)

    def test_the_reference_shifts_the_level_by_its_square(self):
        field = free_field(ranges=(100.0,))
        x = self.burst()
        micro = float(self.quiet(field, x, ref=1e-6).dB[0, 0])
        pascal = float(self.quiet(field, x, ref=1.0).dB[0, 0])
        assert micro - pascal == pytest.approx(120.0, abs=1e-9)

    def test_the_source_level_rides_on_the_waveform_amplitude(self):
        # There is no source-level argument, so the recipe that stands in for
        # one is pinned here: a waveform at SL dB re 1 µPa at 1 m, through
        # 100 m of spherical spreading, must land on SL - 40 + 10log10(T).
        source_level, speed = 190.0, SPEED
        f = np.arange(25.0, 4000.0, 25.0)
        r = np.array([100.0])
        h = (np.exp(-2j * np.pi * np.outer(r, f) / speed) / r[:, None])
        field = Field(data=h[None, ...],
                      coords={'depth': [10.0], 'range': r, 'frequency': f},
                      frequencies=f)
        x = self.burst()
        unit = x / np.sqrt(np.mean(x ** 2))
        duration = x.size / RATE
        scaled = unit * 1e-6 * 10 ** (source_level / 20.0)
        got = float(self.quiet(field, scaled).dB[0, 0])
        want = (source_level - 20.0 * np.log10(100.0)
                + 10.0 * np.log10(duration))
        assert got == pytest.approx(want, abs=1e-6)

    def test_an_unscaled_waveform_gives_the_propagation_term_alone(self):
        # The other route: a unit-rms waveform, with the ENERGY source level
        # ESL = SL + 10log10(T) added afterwards (Ainslie Eq. 3.155).
        source_level = 190.0
        f = np.arange(25.0, 4000.0, 25.0)
        r = np.array([100.0])
        h = (np.exp(-2j * np.pi * np.outer(r, f) / SPEED) / r[:, None])
        field = Field(data=h[None, ...],
                      coords={'depth': [10.0], 'range': r, 'frequency': f},
                      frequencies=f)
        x = self.burst()
        unit = x / np.sqrt(np.mean(x ** 2))
        duration = x.size / RATE
        scaled = unit * 1e-6 * 10 ** (source_level / 20.0)
        propagation = float(self.quiet(field, unit).dB[0, 0])
        energy_source_level = source_level + 10.0 * np.log10(duration)
        path_loss = 10.0 * np.log10(duration / 1e-12) - propagation
        assert energy_source_level - path_loss == pytest.approx(
            float(self.quiet(field, scaled).dB[0, 0]), abs=1e-9)

    def test_the_documented_second_route_adds_esl_to_a_unit_energy_result(self):
        # The recipe the docstring gives: a waveform of unit energy in the
        # reference units (∫u² dt = 1 µPa²·s) returns the propagation term
        # alone, and ESL goes on afterwards. An rms-normalised waveform in
        # its place reads 120 + 10log10(T) dB high (100 dB for 10 ms).
        source_level = 190.0
        f = np.arange(25.0, 4000.0, 25.0)
        r = np.array([100.0])
        h = (np.exp(-2j * np.pi * np.outer(r, f) / SPEED) / r[:, None])
        field = Field(data=h[None, ...],
                      coords={'depth': [10.0], 'range': r, 'frequency': f},
                      frequencies=f)
        x = self.burst()
        duration = x.size / RATE
        direct = float(self.quiet(
            field, x / np.sqrt(np.mean(x ** 2)) * 1e-6
            * 10 ** (source_level / 20.0)).dB[0, 0])
        u = x / np.sqrt(np.sum(x ** 2) / RATE) * 1e-6
        loss = float(self.quiet(field, u).dB[0, 0])
        esl = source_level + 10.0 * np.log10(duration)
        assert esl + loss == pytest.approx(direct, abs=1e-9)
        rms_unit = float(self.quiet(
            field, x / np.sqrt(np.mean(x ** 2))).dB[0, 0])
        assert rms_unit + esl - direct == pytest.approx(
            120.0 + 10.0 * np.log10(duration), abs=1e-9)

    def test_a_wrapped_record_corrupts_the_exposure(self):
        """A fold is NOT harmless to a level map, and nothing warns.

        The opposite of this was asserted in three places and pinned by a
        test that used a single second-path delay — which happened to sit
        where the fold's contribution cancels, reading 0.000002 dB. The
        Parseval argument behind it is wrong: sampling ``H(f)`` every
        ``df`` periodises the response, so a folded arrival adds
        COHERENTLY to what is already there, and the cross term is signed.
        Parseval preserves the energy of the aliased record, which is not
        the energy of the true response.

        Swept over many delays rather than one, so it cannot sit on a null
        again.
        """
        waveform = tone_burst(500.0, 5, sample_rate=RATE)[1]

        def level(df, tau):
            f = np.arange(25.0, 4000.0, df)
            h = (np.exp(-2j * np.pi * f * 0.100)
                 + 0.6 * np.exp(-2j * np.pi * f * tau)) / 100.0
            field = Field(data=h.reshape(1, 1, -1),
                          coords={'depth': [10.0], 'range': [100.0],
                                  'frequency': f}, frequencies=f)
            return float(self.quiet(field, waveform).dB[0, 0])

        # 20 ms record folds every one of these; 1000 ms holds them all.
        errors = np.array([level(50.0, tau) - level(1.0, tau)
                           for tau in np.linspace(0.105, 0.500, 60)])
        assert np.abs(errors).max() > 2.0, errors.max()
        assert np.mean(np.abs(errors) > 0.5) > 0.2, errors
        # The rule that actually holds: dtau*df < 0.5 BOUNDS the error. It
        # does not remove it, and which delays are quiet is not a function
        # of dtau*df alone — frac(f0*dtau) moves it too, so nulls are a
        # property of the axis and not of the product. Asserted as a bound
        # over a sweep, never as a value at a point.
        # On this wide band the error inside the rule is identically zero,
        # so a bound asserted only here cannot tell "the rule BOUNDS the
        # error" from "the rule removes it" — and bounding is what the
        # docstring claims. The residual that makes "bounds" the right word
        # lives on a narrow band, so the assertion goes there, at
        # dtau*df = 0.48: 0.0959 dB, non-trivially under the 0.2 bound.
        inside = [abs(level(df, 0.100 + gap) - level(1.0, 0.100 + gap))
                  for df, gap in ((5.0, 0.030), (10.0, 0.030),
                                  (5.0, 0.060), (2.0, 0.200))]
        assert max(inside) < 0.2, inside          # every dtau*df < 0.5

        def narrow(df, tau):
            f = np.arange(900.0, 1100.0, df)
            h = (np.exp(-2j * np.pi * f * 0.100)
                 + 0.6 * np.exp(-2j * np.pi * f * tau)) / 100.0
            field = Field(data=h.reshape(1, 1, -1),
                          coords={'depth': [10.0], 'range': [100.0],
                                  'frequency': f}, frequencies=f)
            return float(self.quiet(field, waveform).dB[0, 0])

        edge = abs(narrow(16.0, 0.130) - narrow(0.5, 0.130))   # 0.48
        assert 0.01 < edge < 0.2, edge
        outside = [abs(level(50.0, 0.100 + gap) - level(1.0, 0.100 + gap))
                   for gap in (0.020, 0.040, 0.060)]
        assert max(outside) > 2.0, outside        # every dtau*df >= 1

    def test_the_result_is_a_level_and_not_a_loss(self):
        # A loss and a level read opposite ways; sharing a colorbar or an
        # argmax between them inverts one of the two.
        out = self.quiet(free_field(ranges=(100.0,)), self.burst())
        assert (out.kind, out.unit) == ('sound_exposure', 'dB')
        assert not is_loss(out.kind)
        assert list(out.coords) == ['depth', 'range']
        with pytest.raises(AttributeError, match="not 'pressure'"):
            out.tl

    def test_a_field_that_lost_its_phase_is_refused(self):
        """A dB field has no impulse response to integrate.

        Without this the synthesis inverse-transforms the decibels AS
        pressure: measured 124.72 dB against a true 52.68, and 136.20 dB
        through ``at_source_level(190)`` — which is the chain someone after
        an absolute level writes. It returned a ``kind='sound_exposure'``
        Field, so it drew with the right colorbar. Both sibling methods
        guarded this; this one did not.
        """
        field = free_field(ranges=(100.0,))
        waveform = self.burst()
        for lost in (field.to_dB(), field.at_source_level(190.0)):
            with pytest.raises(ConfigurationError, match="complex"):
                lost.sound_exposure_level(source_waveform=waveform,
                                          sample_rate=RATE)

    def test_a_non_positive_reference_is_refused(self):
        field = free_field(ranges=(100.0,))
        for bad in (0.0, -1e-6, np.nan):
            with pytest.raises(ConfigurationError, match="positive pressure"):
                field.sound_exposure_level(source_waveform=self.burst(),
                                           sample_rate=RATE, ref=bad)

    def test_a_cell_sliced_out_of_the_grid_gives_that_cells_exposure(self):
        # The sliced depth sits in `pinned`; the synthesis re-inflates it
        # rather than refusing the slice.
        grid = free_field(ranges=(100.0,))
        whole = grid.sound_exposure_level(source_waveform=self.burst(),
                                          sample_rate=RATE, t_start=0.0)
        one_cell = grid.isel(depth=0).sound_exposure_level(source_waveform=self.burst(),
                                                           sample_rate=RATE,
                                                           t_start=0.0)
        assert list(one_cell.coords) == ['range']
        np.testing.assert_array_equal(one_cell.data, whole.data[0])

    def test_a_band_cutting_the_waveform_states_the_energy_the_sel_keeps(
            self):
        """For an exposure the band edge's cost is the energy outside the
        band, not ringing, and a window would cost more: the warning gives
        the band's share of the waveform's energy, which is exactly what a
        flat H = 1 map loses against the waveform's own SEL, and gives no
        window advice."""
        from uacpy.core.acoustics import sound_exposure_level
        f = np.arange(400.0, 700.0, 5.0)
        H = Field(data=np.ones((1, 1, f.size), dtype=complex),
                  coords={'depth': [10.0], 'range': [100.0], 'frequency': f},
                  frequencies=f)
        burst = np.hanning(64) * np.sin(2 * np.pi * 500.0
                                        * np.arange(64) / RATE)
        with recorded_warnings() as caught:
            sel = H.sound_exposure_level(source_waveform=burst,
                                         sample_rate=RATE)
        text = [str(w.message) for w in caught if 'band' in str(w.message)
                and 'edge' in str(w.message)]
        assert len(text) == 1 and "window='hann'" not in text[0], text
        share = float(re.search(r'holds ([0-9.]+) %', text[0]).group(1))
        lost = float(np.asarray(sel.data).ravel()[0]) - sound_exposure_level(
            burst, RATE)
        assert 100 * 10 ** (lost / 10) == pytest.approx(share, rel=1e-3)

    def test_a_peak_or_a_trace_keeps_the_ringing_advice(self):
        f = np.arange(400.0, 700.0, 5.0)
        H = Field(data=np.ones((1, 1, f.size), dtype=complex),
                  coords={'depth': [10.0], 'range': [100.0], 'frequency': f},
                  frequencies=f)
        burst = np.hanning(64) * np.sin(2 * np.pi * 500.0
                                        * np.arange(64) / RATE)
        for call in (H.peak_sound_pressure_level, H.synthesize_time_series):
            with recorded_warnings() as caught:
                call(source_waveform=burst, sample_rate=RATE)
            assert any('rings at the band edges' in str(w.message)
                       for w in caught)

    @pytest.mark.parametrize('call', ['sel', 'peak', 'series'])
    def test_a_single_frequency_field_is_sent_to_a_broadband_run(self, call):
        # A narrowband TL run has no frequency axis: no slice can give it
        # one, so the refusal names the run that does.
        tl = free_field(ranges=(100.0, 200.0)).isel(frequency=0)
        kw = dict(source_waveform=self.burst(), sample_rate=RATE)
        with pytest.raises(ConfigurationError,
                           match=r"needs the transfer function H\(f\) over a "
                                 r"band") as caught:
            {'sel': tl.sound_exposure_level,
             'peak': tl.peak_sound_pressure_level,
             'series': tl.synthesize_time_series}[call](**kw)
        text = str(caught.value)
        assert 'RunMode.BROADBAND' in text and '.at(' not in text, text

    def test_a_field_whose_axes_are_out_of_order_is_refused(self):
        grid = free_field(ranges=(100.0, 200.0))
        swapped = Field(data=np.moveaxis(grid.data[0], 0, 1),
                        coords={'frequency': grid.coords['frequency'],
                                'range': grid.coords['range']},
                        frequencies=grid.frequencies)
        with pytest.raises(ConfigurationError, match="'frequency' axis"):
            swapped.sound_exposure_level(source_waveform=self.burst(),
                                         sample_rate=RATE)


class TestAWeightedMapIsTheWeightedReceivedPressure:
    """weighting= multiplies H(f) by 10**(W/20) before synthesis: the map is
    the plain-array weighted SEL of each cell's received trace."""

    @staticmethod
    def burst():
        return tone_burst(500.0, 5, sample_rate=RATE)[1]

    @staticmethod
    def quiet(call):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return call()

    @pytest.mark.parametrize('weighting, label', [
        ('VHF', 'VHF'),
        (([100.0, 1000.0, 3000.0], [-30.0, 0.0, -10.0]), 'custom'),
        (lambda f: -10.0 * np.log10(1.0 + (f / 1000.0) ** 2), 'custom'),
    ])
    def test_it_is_the_weighted_exposure_of_the_received_trace(
            self, weighting, label):
        # At 100 m the pulse sits inside the 1/df record, so the trace is
        # not wrapped and the plain-array route sees all of it.
        from uacpy.core.acoustics import sound_exposure_level
        field = free_field(ranges=(100.0,))
        x = self.burst()
        sel = self.quiet(lambda: field.sound_exposure_level(source_waveform=x,
                                                            sample_rate=RATE,
                                                            weighting=weighting))
        trace = self.quiet(lambda: field.synthesize_time_series(source_waveform=x,
                                                                sample_rate=RATE))
        want = sound_exposure_level(np.asarray(trace.data).real,
                                    1.0 / trace.dt, weighting=weighting)
        assert float(sel.data[0, 0]) == pytest.approx(float(want[0, 0]),
                                                      abs=1e-3)
        assert sel.weighting == label

    @pytest.mark.parametrize('weighting, label', [
        ('LF', 'LF'),
        (([100.0, 1000.0, 3000.0], [-30.0, 0.0, -10.0]), 'custom')])
    def test_the_band_route_takes_the_weighting_and_records_it(
            self, weighting, label):
        """broadband_loss(weighting=) weights the band by |S|²·10^(W/10):
        the loss the hand-folded recipe gives, and the map (and the level
        set from it) records the weighting the waveform route records."""
        from uacpy.acoustics import frequency_weighting
        # Two paths 1 ms apart: |H| changes over the band, so the weighting
        # moves the loss (a flat |H| would hide it).
        field = two_path_field(delay=1e-3,
                               frequencies=np.arange(50.0, 3000.0, 10.0))
        f = np.asarray(field.coords['frequency'])
        density = np.minimum(1.0, (500.0 / f) ** 2)
        folded = density * 10 ** (frequency_weighting(f, weighting) / 10)
        by_hand = field.broadband_loss(source_spectrum=np.sqrt(folded))
        loss = self.quiet(lambda: field.broadband_loss(
            source_spectrum=np.sqrt(density), weighting=weighting))
        np.testing.assert_allclose(loss.data, by_hand.data, rtol=0,
                                   atol=1e-9)
        assert loss.weighting == label and by_hand.weighting is None
        assert loss.at_source_level(150.0).weighting == label
        unweighted = field.broadband_loss(source_spectrum=np.sqrt(density))
        assert np.max(np.abs(np.asarray(loss.data)
                             - np.asarray(unweighted.data))) > 0.01

    def test_a_weighting_with_no_weight_on_the_band_is_refused(self):
        with pytest.raises(ConfigurationError,
                           match="^Field.broadband_loss: the weighting "
                                 "carries no weight on this band"):
            free_field().broadband_loss(
                weighting=([1e6, 2e6], [0.0, 0.0]))

    def test_a_flat_weighting_leaves_the_map_unchanged(self):
        field = free_field()
        x = self.burst()
        plain = self.quiet(lambda: field.sound_exposure_level(source_waveform=x,
                                                              sample_rate=RATE))
        flat = self.quiet(lambda: field.sound_exposure_level(source_waveform=x,
                                                             sample_rate=RATE,
                                                             weighting=lambda f: np.zeros_like(f)))
        np.testing.assert_allclose(flat.data, plain.data, atol=1e-9)
        assert plain.weighting is None

    def test_a_weighting_lowers_every_cell_by_the_same_amount(self):
        # The free field's cells differ only by 1/r, which a weighting
        # cannot see.
        field = free_field()
        x = self.burst()
        plain = self.quiet(lambda: field.sound_exposure_level(source_waveform=x,
                                                              sample_rate=RATE)).data
        lf = self.quiet(lambda: field.sound_exposure_level(source_waveform=x,
                                                           sample_rate=RATE,
                                                           weighting='HF')).data
        drop = plain - lf
        assert drop[0, 0] > 10.0
        assert drop[0, 1] == pytest.approx(drop[0, 0], abs=1e-6)


class TestATimeDomainFieldIntegratesTheTracesItHolds:
    """A Field on a time axis already holds the received p(t): its exposure
    and peak need no waveform, and equal the H(f) route's on the same
    traces."""

    @staticmethod
    def pair():
        field = free_field()
        x = tone_burst(500.0, 5, sample_rate=RATE)[1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return field, x, field.synthesize_time_series(source_waveform=x,
                                                          sample_rate=RATE)

    def test_the_traces_give_the_transfer_functions_exposure_and_peak(self):
        field, x, traces = self.pair()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            np.testing.assert_allclose(
                traces.sound_exposure_level().data,
                field.sound_exposure_level(source_waveform=x, sample_rate=RATE).data, atol=1e-9)
            np.testing.assert_allclose(
                traces.peak_sound_pressure_level().data,
                field.peak_sound_pressure_level(source_waveform=x,
                                                sample_rate=RATE).data, atol=1e-9)
        assert traces.sound_exposure_level().axes == ['depth', 'range']

    def test_a_weighting_labels_the_trace_map(self):
        _, _, traces = self.pair()
        assert traces.sound_exposure_level(
            weighting='LF').weighting == 'LF'

    def test_a_waveform_on_traces_and_none_on_h_are_refused(self):
        field, x, traces = self.pair()
        with pytest.raises(ConfigurationError, match='takes no source'):
            traces.sound_exposure_level(source_waveform=x, sample_rate=RATE)
        with pytest.raises(ConfigurationError, match='needs the transmitted'):
            field.sound_exposure_level()

    @pytest.mark.parametrize('setting', [dict(sample_rate=48000.0),
                                         dict(window='hann'),
                                         dict(t_start=0.3)])
    @pytest.mark.parametrize('level', ['sound_exposure_level',
                                       'peak_sound_pressure_level'])
    def test_a_synthesis_setting_on_traces_is_refused(self, level, setting):
        # The traces are integrated whole at their own rate: a rate, a band
        # window or a record start would change nothing, so each is refused
        # (naming the time cut) rather than ignored.
        _, _, traces = self.pair()
        with pytest.raises(ConfigurationError,
                           match=rf"(?s){next(iter(setting))}= synthesise.*"
                                 rf"window\(time="):
            getattr(traces, level)(**setting)
        assert np.all(np.isfinite(getattr(traces, level)().data))


class TestATrackMapSumsTheSingleEventMap:
    @staticmethod
    def single_event_map():
        r = np.geomspace(5.0, 20e3, 300)
        field = free_field(ranges=tuple(r), df=50.0, freq_max=3000.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return field.sound_exposure_level(source_waveform=tone_burst(500.0, 5, sample_rate=RATE)[1],
                                              sample_rate=RATE)

    def test_it_is_the_plain_array_track_sum(self):
        from uacpy.core.acoustics import track_exposure
        sel = self.single_event_map()
        x = np.arange(-10e3, 10e3 + 1, 10.0)
        track = sel.track_exposure(x, closest_approach=[100.0, 1000.0])
        want = track_exposure(sel.data, sel.ranges, x,
                              closest_approach=[100.0, 1000.0], axis=1)
        np.testing.assert_allclose(track.data, want, atol=1e-12)
        assert track.kind == 'sound_exposure'
        np.testing.assert_allclose(track.ranges, [100.0, 1000.0])
        assert track.track_events == x.size

    def test_along_adds_the_plan_view_axis_before_the_distance(self):
        sel = self.single_event_map()
        x = np.arange(-1e3, 1e3 + 1, 50.0)
        plan = sel.track_exposure(x, closest_approach=[100.0, 500.0],
                                  along=[-500.0, 0.0, 500.0])
        assert plan.axes == ['depth', 'along', 'range']
        np.testing.assert_allclose(plan.coords['along'], [-500.0, 0.0, 500.0])
        middle = sel.track_exposure(x, closest_approach=[100.0, 500.0])
        np.testing.assert_allclose(plan.data[:, 1, :], middle.data,
                                   atol=1e-12)

    def test_per_event_gives_the_summands_of_the_total(self):
        """per_event=True is each event's SEL at the receiver, on a
        ``position`` axis after ``range``: their energy sum is the track
        total, and one receiver's slice is its exposure along the
        passage, loudest at closest approach."""
        from uacpy.core.acoustics import sum_levels_dB, track_exposure
        sel = self.single_event_map()
        x = np.arange(-2e3, 2e3 + 1, 50.0)
        total = sel.track_exposure(x, closest_approach=[100.0, 1000.0])
        each = sel.track_exposure(x, closest_approach=[100.0, 1000.0],
                                  per_event=True)
        assert each.axes == ['depth', 'range', 'position']
        np.testing.assert_allclose(each.coords['position'], x)
        assert each.track_events is None and each.kind == 'sound_exposure'
        np.testing.assert_allclose(sum_levels_dB(each.data, axis=-1),
                                   total.data, atol=1e-9)
        np.testing.assert_allclose(each.data, track_exposure(
            sel.data, sel.ranges, x, closest_approach=[100.0, 1000.0],
            axis=1, per_event=True), atol=1e-12)
        passage = each.at(range=100.0).data[0]
        assert int(np.argmax(passage)) == int(np.argmin(np.abs(x)))
        plan = sel.track_exposure(x, closest_approach=[100.0],
                                  along=[-500.0, 0.0], per_event=True)
        assert plan.axes == ['depth', 'along', 'range', 'position']

    def test_the_event_position_axis_exports_in_metres(self):
        pytest.importorskip('xarray')
        each = self.single_event_map().track_exposure(
            np.arange(-1e3, 1e3 + 1, 100.0), closest_approach=[100.0],
            per_event=True)
        assert each.to_xarray().coords['position'].attrs['units'] == 'm'

    def test_the_along_axis_exports_in_metres(self):
        # The plan view's along-track axis is a distance, and its export
        # says so as every other spatial axis does.
        pytest.importorskip('xarray')
        sel = self.single_event_map()
        plan = sel.track_exposure(np.arange(-1e3, 1e3 + 1, 50.0),
                                  closest_approach=[100.0, 500.0],
                                  along=[-500.0, 0.0, 500.0])
        coords = plan.to_xarray().coords
        assert coords['along'].attrs['units'] == 'm'
        assert coords['range'].attrs['units'] == 'm'

    def test_a_map_that_is_not_a_single_event_exposure_is_refused(self):
        sel = self.single_event_map()
        track = sel.track_exposure([0.0], closest_approach=[100.0])
        with pytest.raises(ConfigurationError, match='already a track sum'):
            track.track_exposure([0.0], closest_approach=[100.0])
        with pytest.raises(ConfigurationError, match='single-event SEL map'):
            free_field().track_exposure([0.0], closest_approach=[100.0])

    @staticmethod
    def band_route_level_map(source_level=170.0):
        """The band route's received-level map: the band-averaged loss of a
        white source over the run's band, at a source level."""
        r = np.geomspace(5.0, 20e3, 300)
        return free_field(ranges=tuple(r), df=50.0, freq_max=3000.0
                          ).broadband_loss().at_source_level(source_level)

    def test_a_received_level_map_sums_one_event_per_interval(self):
        """The band route reaches the track sum: each event's SEL is the
        level held over event_duration plus 10·log10(Δt / 1 s), and the
        result is an SEL map like the waveform route's, plan view included."""
        from uacpy.core.acoustics import track_exposure
        level = self.band_route_level_map()
        assert level.kind == 'level'
        x = np.arange(-2e3, 2e3 + 1, 5.0)
        track = level.track_exposure(x, closest_approach=[100.0, 1000.0],
                                     event_duration=2.0)
        want = track_exposure(np.asarray(level.data) + 10 * np.log10(2.0),
                              level.ranges, x,
                              closest_approach=[100.0, 1000.0], axis=1)
        np.testing.assert_allclose(track.data, want, atol=1e-12)
        assert track.kind == 'sound_exposure' and track.unit == 'dB'
        assert track.reference == pytest.approx(1e-12)
        assert track.source_level_dB is None and track.track_events == x.size
        plan = level.track_exposure(x, closest_approach=[100.0, 500.0],
                                    along=[-500.0, 0.0, 500.0],
                                    event_duration=2.0)
        assert plan.axes == ['depth', 'along', 'range']

    def test_event_duration_belongs_to_a_level_map_only(self):
        level = self.band_route_level_map()
        with pytest.raises(ConfigurationError,
                           match='with the time the source holds that level'):
            level.track_exposure([0.0], closest_approach=[100.0])
        with pytest.raises(ConfigurationError,
                           match="already one event's energy"):
            self.single_event_map().track_exposure(
                [0.0], closest_approach=[100.0], event_duration=1.0)
        with pytest.raises(ConfigurationError,
                           match='or a received-level map'):
            free_field().track_exposure([0.0], closest_approach=[100.0],
                                        event_duration=1.0)


class TestThePeakIsTheOtherHalfOfTheDualMetric:
    """``peak_sound_pressure_level`` — Southall's second metric.

    SEL integrates the whole transmission; this reads its loudest single
    excursion. Neither substitutes for the other, and a peak is a property
    of the waveform in TIME, so no reduction of ``|H(f)|`` yields it.
    """

    @staticmethod
    def quiet(field, waveform, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return field.peak_sound_pressure_level(source_waveform=waveform,
                                                   sample_rate=RATE, **kw)

    @staticmethod
    def burst():
        return tone_burst(500.0, 5, sample_rate=RATE)[1]

    def test_it_is_a_level_of_its_own_kind(self):
        """Not ``'level'``: a peak map and a mean-square received-level map
        are both dB re 1 µPa and are not the same reading of the field."""
        out = self.quiet(free_field(ranges=(100.0,)), self.burst())
        assert (out.kind, out.unit) == ('peak_pressure', 'dB')
        assert not is_loss(out.kind)
        assert list(out.coords) == ['depth', 'range']

    def test_doubling_the_range_costs_spherical_spreading(self):
        out = self.quiet(free_field(ranges=(100.0, 200.0)), self.burst())
        # Not exact like SEL's: a peak is a sample, read at the waveform's
        # own rate. At 8 kHz a 500 Hz cycle holds 16 samples, so a sampled
        # crest sits up to 1 - cos(pi/16), 0.17 dB, under the true one.
        assert float(out.dB[0, 0]) - float(out.dB[0, 1]) == pytest.approx(
            20.0 * np.log10(2.0), abs=0.2)

    def test_the_level_follows_the_waveform_amplitude(self):
        field = free_field(ranges=(100.0,))
        x = self.burst()
        quiet = float(self.quiet(field, x).dB[0, 0])
        loud = float(self.quiet(field, 3.0 * x).dB[0, 0])
        assert loud - quiet == pytest.approx(20.0 * np.log10(3.0), abs=1e-6)

    def test_it_is_not_the_exposure_level(self):
        """The two rank the same cell differently by tens of dB, which is
        why criteria name both rather than one."""
        field = free_field(ranges=(100.0,))
        x = self.burst() * 1e-6 * 10 ** (190.0 / 20.0)
        peak = float(self.quiet(field, x).dB[0, 0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sel = float(field.sound_exposure_level(source_waveform=x,
                                                   sample_rate=RATE).dB[0, 0])
        assert abs(peak - sel) > 10.0

    def test_it_is_read_off_the_trace_at_the_callers_rate(self):
        """A maximum is a sample, so the peak is the largest sample of the
        received trace at the waveform's own rate — the trace a recorder at
        that rate holds — and the exposure the same trace's integral."""
        f = np.arange(25.0, 4000.0, 25.0)
        h = (1.0 + 0.6 * np.exp(-2j * np.pi * f * 4e-3)) / 100.0
        field = Field(data=h.reshape(1, 1, -1),
                      coords={'depth': [10.0], 'range': [100.0],
                              'frequency': f}, frequencies=f)
        x = self.burst()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            trace = field.synthesize_time_series(source_waveform=x,
                                                 sample_rate=RATE)
            peak = float(self.quiet(field, x).dB[0, 0])
            sel = float(field.sound_exposure_level(source_waveform=x,
                                                   sample_rate=RATE).dB[0, 0])
        p = np.asarray(trace.data).ravel()
        assert trace.sample_rate == pytest.approx(RATE, rel=1e-12)
        assert peak == pytest.approx(
            20.0 * np.log10(np.max(np.abs(p)) / 1e-6), abs=1e-9)
        assert sel == pytest.approx(
            10.0 * np.log10(np.sum(p ** 2) / RATE / 1e-12), abs=1e-9)

    def test_it_records_the_band_it_collapsed(self):
        f = np.arange(800.0, 1201.0, 5.0)
        r = np.array([100.0])
        h = (np.exp(-2j * np.pi * np.outer(r, f) / SPEED) / r[:, None])
        field = Field(data=h[None, ...],
                      coords={'depth': [10.0], 'range': r, 'frequency': f},
                      frequencies=f)
        out = self.quiet(field, tone_burst(1000.0, 20, sample_rate=RATE)[1])
        assert out.band == (800.0, 1200.0)

    def test_a_field_that_lost_its_phase_is_refused(self):
        field = free_field(ranges=(100.0,))
        for lost in (field.to_dB(), field.at_source_level(190.0)):
            with pytest.raises(ConfigurationError, match="complex"):
                lost.peak_sound_pressure_level(source_waveform=self.burst(),
                                               sample_rate=RATE)

    def test_a_non_positive_reference_is_refused(self):
        field = free_field(ranges=(100.0,))
        for bad in (0.0, -1e-6, np.nan):
            with pytest.raises(ConfigurationError, match="positive pressure"):
                field.peak_sound_pressure_level(source_waveform=self.burst(),
                                                sample_rate=RATE, ref=bad)


class TestTheTwoQuantitiesAnswerDifferentQuestions:
    def test_the_band_average_keeps_energy_the_pulse_gate_discards(self):
        # The distinction the sources draw: averaging over the band removes
        # the INTERFERENCE between paths further apart than 1/B and leaves
        # their energy (Ainslie sect. 3.3.2.1 integrates the whole pulse),
        # where truncate_response removes the path itself.
        a, delay = 0.7, 5e-3
        field = two_path_field(amplitude=a, delay=delay)
        averaged = float(field.broadband_loss().tl[0, 0])
        # A window far shorter than the gap drops the second path entirely,
        # leaving the direct path's 0 dB; the band average keeps its energy.
        gated = field.truncate_response(delay / 10.0, origin=0.0)
        gated_loss = float(gated.broadband_loss().tl[0, 0])
        assert averaged == pytest.approx(-10 * np.log10(1 + a ** 2), abs=1e-3)
        assert gated_loss == pytest.approx(0.0, abs=0.2)
        assert gated_loss > averaged


class TestTheExposureRecordSurvivesEveryWriter:
    """``weighting`` and ``track_events`` are the Field's own attributes,
    not metadata, so they must cross the dict, xarray and netCDF round trips
    a reader acts on."""

    @staticmethod
    def track_map():
        r = np.geomspace(5.0, 5000.0, 100)
        sel = Field(data=(200.0 - 20.0 * np.log10(r))[None, :],
                    coords={'depth': [10.0], 'range': r},
                    kind='sound_exposure', unit='dB', weighting='VHF')
        return sel.track_exposure(np.arange(-500.0, 501.0, 50.0),
                                  closest_approach=[100.0, 200.0])

    def test_a_dict_round_trip_keeps_both(self):
        back = Field.from_dict(self.track_map().to_dict())
        assert (back.weighting, back.track_events) == ('VHF', 21)

    def test_an_xarray_round_trip_keeps_both(self):
        pytest.importorskip('xarray')
        back = Field.from_xarray(self.track_map().to_xarray())
        assert (back.weighting, back.track_events) == ('VHF', 21)

    def test_a_netcdf_round_trip_keeps_both(self, tmp_path):
        pytest.importorskip('netCDF4')
        path = tmp_path / 'track.nc'
        self.track_map().to_netcdf(path)
        back = Field.from_netcdf(path)
        assert (back.weighting, back.track_events) == ('VHF', 21)

    def test_metadata_naming_either_is_refused(self):
        for key in ('weighting', 'track_events'):
            with pytest.raises(ConfigurationError, match='attributes'):
                Field(data=np.zeros((1, 2)),
                      coords={'depth': [1.0], 'range': [1.0, 2.0]},
                      metadata={key: 'x'})


class TestADerivedMapStatesWhatItHolds:
    """The band-collapsed and trace-level maps state their own quantity: a
    band-averaged loss is not coherent, no real-dB map keeps a phase
    convention, an exposure or peak map records the reference pressure it is
    in dB re and carries no source level it was not computed with."""

    @staticmethod
    def h_field(**identity):
        field = free_field()
        return field.replace(phase_reference='travelling_wave', coherent=None,
                             **identity)

    @staticmethod
    def sonar_budget(loss):
        from uacpy.sonar import passive_signal_excess_field
        return passive_signal_excess_field(
            loss, source_level_dB=170.0, noise_level_dB=60.0,
            detection_threshold_dB=0.0)

    def test_a_band_averaged_loss_passes_a_budget_without_the_coherent_warning(
            self):
        from uacpy.core.exceptions import ValidityWarning
        field = self.h_field()
        assert field.coherent is True
        with pytest.warns(ValidityWarning, match='coherent TL'):
            self.sonar_budget(field.at(frequency=1000.0))
        loss = field.broadband_loss()
        assert loss.coherent is False
        with warnings.catch_warnings():
            warnings.simplefilter('error', ValidityWarning)
            self.sonar_budget(loss)

    def test_no_real_dB_map_keeps_a_phase_convention(self):
        field = self.h_field()
        x = tone_burst(500.0, 5, sample_rate=RATE)[1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            traces = field.synthesize_time_series(source_waveform=x,
                                                  sample_rate=RATE)
            maps = [field.to_dB(), field.broadband_loss(),
                    field.sound_exposure_level(source_waveform=x,
                                               sample_rate=RATE),
                    field.peak_sound_pressure_level(source_waveform=x,
                                                    sample_rate=RATE),
                    traces.sound_exposure_level(),
                    traces.peak_sound_pressure_level()]
        assert [m.phase_reference for m in maps] == [None] * len(maps)

    def test_to_dB_keeps_the_coherence_of_the_pressure(self):
        assert self.h_field().to_dB().coherent is True

    @pytest.mark.parametrize('ref, sel_label, peak_label', [
        (1e-6, 'SEL (dB re 1 µPa²·s)', 'Peak SPL (dB re 1 µPa)'),
        (20e-6, 'SEL (dB re (20 µPa)²·s)', 'Peak SPL (dB re 20 µPa)'),
    ])
    def test_an_exposure_map_is_labelled_with_the_reference_it_is_in(
            self, ref, sel_label, peak_label):
        from uacpy.core.results.quantities import label
        field = self.h_field()
        x = tone_burst(500.0, 5, sample_rate=RATE)[1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sel = field.sound_exposure_level(source_waveform=x,
                                             sample_rate=RATE, ref=ref)
            peak = field.peak_sound_pressure_level(source_waveform=x,
                                                   sample_rate=RATE, ref=ref)
            traces = field.synthesize_time_series(source_waveform=x,
                                                  sample_rate=RATE)
            trace_sel = traces.sound_exposure_level(ref=ref)
        assert sel.reference == pytest.approx(ref ** 2, rel=1e-12)
        assert (sel.reference_unit, peak.reference_unit) == ('Pa²·s', 'Pa²')
        assert label(sel.kind, sel.unit, reference=sel.reference) == sel_label
        assert label(peak.kind, peak.unit,
                     reference=peak.reference) == peak_label
        assert trace_sel.reference == pytest.approx(ref ** 2, rel=1e-12)
        # 0 dB is the reference: the linear exposure comes back from it.
        linear = sel.reference * 10.0 ** (sel.data / 10.0)
        np.testing.assert_allclose(
            linear, 1e-12 * 10.0 ** (field.sound_exposure_level(source_waveform=x,
                                                                sample_rate=RATE).data
                                     / 10.0), rtol=1e-9)

    def test_a_reference_names_nothing_on_a_kind_without_one(self):
        from uacpy.core.results.quantities import label
        assert label('ambiguity', 'dB', reference=4e-10) == \
            'Normalised power (dB re max)'

    def test_an_exposure_map_carries_no_source_level_and_a_loss_keeps_it(self):
        field = self.h_field(source_level_dB=190.0)
        x = tone_burst(500.0, 5, sample_rate=RATE)[1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            traces = field.synthesize_time_series(source_waveform=x,
                                                  sample_rate=RATE)
            levels = [field.sound_exposure_level(source_waveform=x,
                                                 sample_rate=RATE),
                      field.peak_sound_pressure_level(source_waveform=x,
                                                      sample_rate=RATE),
                      traces.sound_exposure_level(),
                      traces.peak_sound_pressure_level()]
        assert [m.source_level_dB for m in levels] == [None] * len(levels)
        assert field.broadband_loss().source_level_dB == 190.0

    def test_a_reference_crosses_the_netcdf_round_trip(self, tmp_path):
        pytest.importorskip('netCDF4')
        x = tone_burst(500.0, 5, sample_rate=RATE)[1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sel = self.h_field().sound_exposure_level(source_waveform=x,
                                                      sample_rate=RATE,
                                                      ref=20e-6)
        path = tmp_path / 'sel.nc'
        sel.to_netcdf(path)
        back = Field.from_netcdf(path)
        assert back.reference == pytest.approx(4e-10, rel=1e-12)
        assert back.reference_unit == 'Pa²·s'

    @pytest.mark.parametrize('ref', [0.0, -1e-6, np.nan])
    def test_a_reference_that_is_not_a_positive_pressure_is_refused_on_traces(
            self, ref):
        x = tone_burst(500.0, 5, sample_rate=RATE)[1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            traces = self.h_field().synthesize_time_series(source_waveform=x,
                                                           sample_rate=RATE)
        assert np.isfinite(traces.sound_exposure_level(ref=1e-9).data).all()
        for level in (traces.sound_exposure_level,
                      traces.peak_sound_pressure_level):
            with pytest.raises(ConfigurationError,
                               match='reference must be a positive pressure'):
                level(ref=ref)
