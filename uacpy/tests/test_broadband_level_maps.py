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

import numpy as np
import pytest
import warnings

from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Field
from uacpy.core.results.quantities import is_loss
from uacpy.acoustic_signal import lfm_chirp, tone_burst

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


def free_field(ranges=(100.0, 200.0), df=25.0, f_max=4000.0):
    """``H = exp(-i k r) / r`` — spherical spreading, no boundaries.

    The NEGATIVE exponent is uacpy's delay convention, the one
    ``Arrivals.transfer_function`` writes; ``exp(+ikr)`` is an
    incoming wave and puts the pulse before its travel time.
    """
    f = np.arange(df, f_max, df)
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
        assert float(field.broadband_loss(spectrum=weights).tl[0, 0]) == \
            pytest.approx(float(field.tl[0, 0, 40]), abs=1e-9)

    def test_the_spectrum_scale_cancels(self):
        field = two_path_field()
        shape = np.hanning(field.n_frequencies) + 0.1
        one = float(field.broadband_loss(spectrum=shape).tl[0, 0])
        scaled = float(field.broadband_loss(spectrum=1e6 * shape).tl[0, 0])
        assert one == pytest.approx(scaled, abs=1e-12)

    def test_a_complex_spectrum_weighs_by_its_magnitude(self):
        field = two_path_field()
        shape = np.hanning(field.n_frequencies) + 0.1
        phased = shape * np.exp(1j * np.linspace(0.0, 7.0, shape.size))
        assert float(field.broadband_loss(spectrum=phased).tl[0, 0]) == \
            pytest.approx(float(field.broadband_loss(spectrum=shape).tl[0, 0]),
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
        assert float(field.broadband_loss(spectrum=phased).tl[0, 0]) == \
            pytest.approx(
                float(field.broadband_loss(spectrum=magnitude).tl[0, 0]),
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
        waveform = tone_burst(1000.0, 20, RATE)[1]
        for low, high in ((975.0, 1025.0), (800.0, 1200.0)):
            band = field.window(frequency=(low, high))
            assert band.broadband_loss().metadata['band_hz'] == (low, high)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                exposure = band.sound_exposure_level(waveform, RATE)
            assert exposure.metadata['band_hz'] == (low, high)
            # and the identity really is narrowed, so band_hz is the only
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
        assert np.isnan(holed.broadband_loss(spectrum=weights).tl[0, 0])

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

    def test_a_spectrum_off_the_frequency_axis_is_refused(self):
        field = two_path_field()
        with pytest.raises(ConfigurationError, match="frequency axis has"):
            field.broadband_loss(spectrum=np.ones(field.n_frequencies + 1))

    def test_a_spectrum_with_a_hole_in_it_is_refused(self):
        # The one guard the mutation sweep found unpinned: without it a NaN
        # weight propagates to every cell and the map reads NaN everywhere,
        # which looks like a model that reached nothing.
        field = two_path_field()
        for bad in (np.nan, np.inf):
            weights = np.ones(field.n_frequencies)
            weights[3] = bad
            with pytest.raises(ConfigurationError, match="finite"):
                field.broadband_loss(spectrum=weights)

    def test_the_pinned_frequency_follows_the_spectrum(self):
        # A 500 Hz burst's map is a map of 500 Hz. Labelling it with the
        # axis midpoint is a legal frequency and the wrong one — the same
        # defect Field.window's identity narrowing exists to prevent.
        f = np.arange(25.0, 4000.0, 25.0)
        field = Field(data=np.ones((1, 1, f.size), complex),
                      coords={'depth': [0.0], 'range': [1.0],
                              'frequency': f}, frequencies=f)
        burst = tone_burst(500.0, 5, 8000.0)[1]
        weighted = field.broadband_loss(waveform=burst, sample_rate=8000.0)
        assert weighted.pinned['frequency'] == pytest.approx(500.0, abs=25.0)
        # and a white source still lands on the band centre
        assert field.broadband_loss().pinned['frequency'] == pytest.approx(
            float(f.mean()), abs=1e-9)

    def test_a_spectrum_carrying_no_energy_is_refused(self):
        field = two_path_field()
        with pytest.raises(ConfigurationError, match="no energy"):
            field.broadband_loss(spectrum=np.zeros(field.n_frequencies))


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
            waveform = tone_burst(500.0, cycles, self.RATE)[1]
            spectrum = np.fft.rfft(waveform)
            remixed = np.fft.irfft(
                np.abs(spectrum) * np.exp(
                    1j * rng.uniform(0.0, 2.0 * np.pi, spectrum.size)),
                n=waveform.size)
            straight = field.broadband_loss(waveform=waveform,
                                            sample_rate=self.RATE)
            phased = field.broadband_loss(waveform=remixed,
                                          sample_rate=self.RATE)
            return abs(float(straight.tl[0, 0]) - float(phased.tl[0, 0]))

        for cycles in (20, 40):                     # T*df = 1.0, 2.0
            assert (tone_burst(500.0, cycles, self.RATE)[1].size
                    / self.RATE * df) == pytest.approx(round(
                        tone_burst(500.0, cycles, self.RATE)[1].size
                        / self.RATE * df))
            assert spread(cycles) == pytest.approx(0.0, abs=1e-9), cycles
        for cycles in (19, 21, 41):                 # 0.95, 1.05, 2.05
            assert spread(cycles) > 0.01, cycles

    def test_different_signals_get_different_losses(self):
        field = self.echoing_channel()
        losses = [
            float(field.broadband_loss(
                waveform=w, sample_rate=self.RATE).tl[0, 0])
            for w in (tone_burst(500.0, 5, self.RATE)[1],
                      tone_burst(625.0, 5, self.RATE)[1],
                      tone_burst(500.0, 20, self.RATE)[1])]
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
        for waveform in (tone_burst(500.0, 5, self.RATE)[1],
                         tone_burst(625.0, 5, self.RATE)[1],
                         tone_burst(500.0, 20, self.RATE)[1]):
            duration = waveform.size / self.RATE
            scaled = (waveform / np.sqrt(np.mean(waveform ** 2))
                      * 1e-6 * 10 ** (source_level / 20.0))
            loss = float(field.broadband_loss(
                waveform=scaled, sample_rate=self.RATE).tl[0, 0])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sel = float(field.sound_exposure_level(
                    scaled, self.RATE).dB[0, 0])
            energy_source_level = (source_level
                                   + 10.0 * np.log10(duration))
            assert sel == pytest.approx(energy_source_level - loss, abs=1e-6)

    def test_the_waveform_scale_cancels(self):
        field = self.echoing_channel()
        waveform = tone_burst(500.0, 5, self.RATE)[1]
        quiet = float(field.broadband_loss(
            waveform=waveform, sample_rate=self.RATE).tl[0, 0])
        loud = float(field.broadband_loss(
            waveform=1e7 * waveform, sample_rate=self.RATE).tl[0, 0])
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
        waveform = tone_burst(500.0, 5, self.RATE)[1]
        interpolated = np.interp(
            freqs, np.fft.rfftfreq(waveform.size, 1.0 / self.RATE),
            np.abs(np.fft.rfft(waveform)))
        by_hand = float(field.broadband_loss(spectrum=interpolated).tl[0, 0])
        exact = float(field.broadband_loss(
            waveform=waveform, sample_rate=self.RATE).tl[0, 0])
        assert abs(by_hand - exact) > 0.1

    def test_a_waveform_and_a_spectrum_together_are_refused(self):
        field = self.echoing_channel()
        with pytest.raises(ConfigurationError, match="not both"):
            field.broadband_loss(
                spectrum=np.ones(field.n_frequencies),
                waveform=tone_burst(500.0, 5, self.RATE)[1],
                sample_rate=self.RATE)

    def test_a_waveform_without_a_sample_rate_is_refused(self):
        field = self.echoing_channel()
        with pytest.raises(ConfigurationError, match="sample_rate"):
            field.broadband_loss(
                waveform=tone_burst(500.0, 5, self.RATE)[1])

    def test_the_generators_time_signal_pair_is_refused(self):
        field = self.echoing_channel()
        with pytest.raises(ConfigurationError, match="1-D signal"):
            field.broadband_loss(waveform=tone_burst(500.0, 5, self.RATE),
                                 sample_rate=self.RATE)


class TestTheExposureLevelIsTheEnergyOneTransmissionDelivers:
    @staticmethod
    def burst():
        return tone_burst(500.0, 5, RATE)[1]

    @staticmethod
    def quiet(field, waveform, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return field.sound_exposure_level(waveform, RATE, **kw)

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
        flat = float(self.quiet(field, x, window='none').dB[0, 0])
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
        micro = float(self.quiet(field, x, reference=1e-6).dB[0, 0])
        pascal = float(self.quiet(field, x, reference=1.0).dB[0, 0])
        assert micro - pascal == pytest.approx(120.0, abs=1e-9)

    def test_the_source_level_rides_on_the_waveform_amplitude(self):
        # There is no source-level argument, so the recipe that stands in for
        # one is pinned here: a waveform at SL dB re 1 uPa at 1 m, through
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
        waveform = tone_burst(500.0, 5, RATE)[1]

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
                lost.sound_exposure_level(waveform, RATE)

    def test_a_non_positive_reference_is_refused(self):
        field = free_field(ranges=(100.0,))
        for bad in (0.0, -1e-6, np.nan):
            with pytest.raises(ConfigurationError, match="positive pressure"):
                field.sound_exposure_level(self.burst(), RATE, reference=bad)

    def test_a_field_that_is_not_a_grid_is_refused(self):
        one_cell = free_field(ranges=(100.0,)).isel(depth=0)
        with pytest.raises(ConfigurationError, match="canonical"):
            one_cell.sound_exposure_level(self.burst(), RATE)


class TestAWindowedFieldReportsTheBandItHolds:
    """``Field.window`` narrows the identity, not only the axis.

    Selecting a signal's band out of a wider run is how both quantities above
    are reached, and a field that keeps the run's whole frequency list after
    that answers ``f0`` and ``n_frequencies`` about samples it no longer has.
    ``isel`` already narrows the identity when it pins an axis.
    """

    @staticmethod
    def wideband():
        f = np.arange(1000.0, 2001.0, 100.0)
        return Field(data=np.ones((1, 1, f.size), complex),
                     coords={'depth': [0.0], 'range': [1.0], 'frequency': f},
                     frequencies=f)

    def test_the_band_count_matches_the_axis(self):
        narrowed = self.wideband().window(frequency=(1400.0, 1600.0))
        assert narrowed.n_frequencies == narrowed.coords['frequency'].size == 3

    def test_the_centre_frequency_is_one_the_field_holds(self):
        # 1000 Hz is a legal frequency and the wrong answer, which is why
        # this needs its own assertion rather than a finiteness check.
        narrowed = self.wideband().window(frequency=(1400.0, 1600.0))
        assert narrowed.f0 == pytest.approx(1400.0)

    def test_windowing_another_axis_leaves_the_band_alone(self):
        untouched = self.wideband().window(range=(0.0, 2.0))
        assert untouched.n_frequencies == 11

    def test_source_depths_narrow_with_their_axis(self):
        z = np.array([5.0, 10.0, 15.0])
        f = np.array([100.0, 200.0])
        field = Field(data=np.ones((z.size, 1, 1, f.size), complex),
                      coords={'source_depth': z, 'depth': [0.0],
                              'range': [1.0], 'frequency': f},
                      frequencies=f, source_depths=z)
        narrowed = field.window(source_depth=(9.0, 16.0))
        assert list(np.asarray(narrowed.source_depths)) == [10.0, 15.0]


class TestAGridThatCannotBePlacedOnADftIsRefused:
    """A knife edge in the synthesis plan, found by audit.

    The band is placed at ``floor(f/df + 0.5)`` and de-rotated by ONE common
    offset taken from the first sample. When ``freqs[0]/df`` lands on the .5
    boundary the first sample rounds one way and the rest the other, so part
    of the band sits a whole bin out. It is silent: on 25 Hz-4 kHz with
    ``df`` = 2/3 a two-path SEL read 52.34 dB against a true 54.01, on a
    1500 ms record where nothing folds. Nudging ``df`` by 0.005 Hz is exact,
    so no caller would think to check.
    """

    @staticmethod
    def two_paths(df):
        f = np.arange(25.0, 4000.0, df)
        h = (np.exp(-2j * np.pi * f * 0.100)
             + 0.6 * np.exp(-2j * np.pi * f * 0.130)) / 100.0
        return Field(data=h.reshape(1, 1, -1),
                     coords={'depth': [10.0], 'range': [100.0],
                             'frequency': f}, frequencies=f)

    def level(self, df):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return float(self.two_paths(df).sound_exposure_level(
                tone_burst(500.0, 5, RATE)[1], RATE).dB[0, 0])

    def test_a_grid_on_the_rounding_boundary_is_refused(self):
        for df in (2.0 / 3.0, 25.0 / 40.5):       # freqs[0]/df = 37.5, 40.5
            assert (25.0 / df) % 1 == pytest.approx(0.5), df
            with pytest.raises(ConfigurationError, match="rounding boundary"):
                self.level(df)

    def test_grids_either_side_of_it_agree_exactly(self):
        # The refusal is a knife edge, not a broad rejection: a few parts in
        # 1e3 of df either way is exact, which is why it had to be checked
        # rather than left to a caller to notice.
        for df in (0.665, 0.670, 1.0, 5.0, 25.0):
            assert self.level(df) == pytest.approx(54.0108, abs=1e-3), df


class TestOneSignalAtOneReceiver:
    """``to_time_trace(waveform=...)`` — the received signal at a position.

    ``synthesize_time_series`` convolves every cell; this is the one-receiver
    form, and it took a ``source_spectrum`` the caller had to sample onto the
    field's axis, with the same interpolation trap ``broadband_loss`` had.
    """

    RATE = 8000.0

    @staticmethod
    def spreading_field():
        f = np.arange(25.0, 4000.0, 25.0)
        d = np.array([10.0, 50.0])
        r = np.array([100.0, 500.0])
        rr = r[None, :, None]
        h = (np.exp(-2j * np.pi * f[None, None, :] * rr / SPEED) / rr)
        return Field(data=h * np.ones((d.size, 1, 1)),
                     coords={'depth': d, 'range': r, 'frequency': f},
                     frequencies=f)

    def trace(self, field, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return field.to_time_trace(window='none', **kw)

    def test_the_pulse_arrives_at_the_geometric_travel_time(self):
        field = self.spreading_field()
        waveform = lfm_chirp(300.0, 900.0, 0.020, self.RATE)[1]
        out = self.trace(field, depth=50.0, range=500.0,
                         waveform=waveform, sample_rate=self.RATE)
        pressure = np.asarray(out.data).real
        times = np.asarray(out.coords['time'])
        power = pressure ** 2
        # The CENTROID, not the peak: an LFM's envelope is flat, so which
        # sample peaks is arbitrary within the pulse, and a peak test on a
        # record barely longer than the pulse passes for any placement at
        # all — including a time-advanced field. The centroid of a T-long
        # pulse starting at tau sits at tau + T/2.
        centroid = float(np.sum(times * power) / power.sum())
        expected = 500.0 / SPEED + 0.5 * waveform.size / self.RATE
        assert centroid == pytest.approx(expected, abs=3.0e-3)

    def test_the_amplitude_follows_one_over_range(self):
        field = self.spreading_field()
        waveform = tone_burst(500.0, 5, self.RATE)[1]
        near = self.trace(field, depth=50.0, range=100.0,
                          waveform=waveform, sample_rate=self.RATE)
        far = self.trace(field, depth=50.0, range=500.0,
                         waveform=waveform, sample_rate=self.RATE)
        ratio = (np.abs(np.asarray(near.data).real).max()
                 / np.abs(np.asarray(far.data).real).max())
        assert ratio == pytest.approx(5.0, rel=0.05)

    def test_the_cell_it_used_is_recorded(self):
        field = self.spreading_field()
        out = self.trace(field, depth=50.0, range=500.0,
                         waveform=tone_burst(500.0, 5, self.RATE)[1],
                         sample_rate=self.RATE)
        assert out.pinned['depth'] == 50.0
        assert out.pinned['range'] == 500.0

    def test_a_receiver_off_the_grid_says_so(self):
        """The failure this guards is silent, not loud.

        The match is to the nearest stored coordinate, so asking for 5 km on
        a grid ending at 500 m returns a perfectly ordinary trace — of the
        wrong place.
        """
        field = self.spreading_field()
        with pytest.warns(UserWarning, match="outside the grid"):
            out = field.to_time_trace(
                depth=50.0, range=5000.0, window='none',
                waveform=tone_burst(500.0, 5, self.RATE)[1],
                sample_rate=self.RATE)
        assert out.pinned['range'] == 500.0

    def test_a_receiver_on_the_grid_is_quiet(self):
        field = self.spreading_field()
        # Only this warning is asserted on: the synthesis raises its own
        # unrelated ones (no stamped sound speed on a hand-built Field), and
        # an 'error' filter over all UserWarnings would pass for the wrong
        # reason. 300 m is BETWEEN stored ranges, which is a snap and not an
        # off-grid request.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            field.to_time_trace(
                depth=50.0, range=300.0, window='none',
                waveform=tone_burst(500.0, 5, self.RATE)[1],
                sample_rate=self.RATE)
        assert not [w for w in caught
                    if 'outside the grid' in str(w.message)]

    def test_a_waveform_and_a_spectrum_together_are_refused(self):
        field = self.spreading_field()
        with pytest.raises(ConfigurationError, match="not both"):
            field.to_time_trace(
                depth=50.0, range=500.0,
                waveform=tone_burst(500.0, 5, self.RATE)[1],
                sample_rate=self.RATE,
                source_spectrum=np.ones(field.n_frequencies))

    def test_a_waveform_without_a_sample_rate_is_refused(self):
        field = self.spreading_field()
        with pytest.raises(ConfigurationError, match="sample_rate"):
            field.to_time_trace(
                depth=50.0, range=500.0,
                waveform=tone_burst(500.0, 5, self.RATE)[1])

    def test_the_generators_time_signal_pair_is_refused(self):
        field = self.spreading_field()
        with pytest.raises(ConfigurationError, match="1-D signal"):
            field.to_time_trace(depth=50.0, range=500.0,
                                waveform=tone_burst(500.0, 5, self.RATE),
                                sample_rate=self.RATE)


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
