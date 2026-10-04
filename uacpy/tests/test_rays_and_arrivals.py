"""``Rays`` and ``Arrivals`` (``uacpy.core.results.rays``).

Ray fans filtered and sorted without a solver, the flat arrival list and its
filter chain, the per-cell record rebuilt from it, the channel view, and the
dict and ``np.savez`` round trip.
"""

import inspect
import numpy as np
import pytest
import uacpy
import warnings
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Arrivals


class TestRaysMissDistanceUnits:
    """Ray polylines are in metres; ``Rays._miss_distance_to`` consumes
    them verbatim with no unit rescaling."""

    def test_short_polyline_in_metres_is_not_rescaled(self):
        from uacpy.core.results import Rays
        r_m = np.linspace(0.0, 10.0, 11)
        z_m = np.linspace(0.0, 5.0, 11)
        rays = Rays(
            rays=[{
                'r': r_m, 'z': z_m,
                'launch_angle': 0.0,
                'n_top_bounces': 0, 'n_bot_bounces': 0,
            }],
            is_eigen=False,
            receiver_depths=np.array([5.0]),
            receiver_ranges=np.array([10.0]),
            model='Test', frequencies=100.0,
        )
        # The polyline passes exactly through (r=10, z=5), so miss == 0
        # in metres. (No km->m rescale: that would blow the miss up to
        # ~10 km.)
        miss, _ = rays._miss_distance_to(rays.rays[0], 10.0, 5.0)
        assert miss == pytest.approx(0.0, abs=1e-12)


class TestArrivalsFlatListKeys:
    """``Arrivals.arrivals`` flat list carries writer-aligned bounce keys."""

    def test_arrivals_flat_list_uses_n_bounce_keys(self):
        from uacpy.core.results import Arrivals
        # Build a minimal payload with the canonical IO key naming.
        payload = [[[{
            "delays": np.array([0.1, 0.2]),
            "amplitudes": np.array([1.0, 0.5]),
            "phases": np.array([0.0, 0.1]),
            "n_top_bounces": np.array([0, 1], dtype=int),
            "n_bot_bounces": np.array([1, 2], dtype=int),
            "source_angles": np.array([0.0, 5.0]),
            "receiver_angles": np.array([0.0, -5.0]),
            "n_arrivals": 2,
        }]]]
        arr = Arrivals(
            by_receiver=payload,
            receiver_depths=np.array([50.0]),
            receiver_ranges=np.array([1000.0]),
            model='Test',
            frequencies=100.0,
        )
        table = arr.arrivals
        assert len(table) == 2
        for row in table:
            assert 'n_top_bounces' in row
            assert 'n_bot_bounces' in row
        assert table[0]['kind'] == 'bottom'
        assert table[1]['kind'] == 'both'
        assert table[0]['n_bot_bounces'] == 1
        assert table[1]['n_top_bounces'] == 1


class TestArrivalsFilterChain:
    """Arrivals exposes a Rays-style filter chain over a flat list of
    arrival events; no continuous-axis ``at(...)`` slicer."""

    def _arrivals(self):
        from uacpy.core.results import Arrivals
        # Two cells (1 src × 1 depth × 2 ranges) with a mix of bounce kinds.
        cell0 = {
            "delays": np.array([0.1, 0.2, 0.3]),
            "amplitudes": np.array([1.0, 0.5, 0.2]),
            "phases": np.array([0.0, 0.0, 0.0]),
            "n_top_bounces": np.array([0, 1, 1], dtype=int),
            "n_bot_bounces": np.array([0, 0, 2], dtype=int),
            "source_angles": np.array([0.0, 5.0, 10.0]),
            "receiver_angles": np.array([0.0, -5.0, -10.0]),
        }
        cell1 = {
            "delays": np.array([0.4]),
            "amplitudes": np.array([0.8]),
            "phases": np.array([0.0]),
            "n_top_bounces": np.array([0], dtype=int),
            "n_bot_bounces": np.array([1], dtype=int),
            "source_angles": np.array([2.0]),
            "receiver_angles": np.array([-2.0]),
        }
        return Arrivals(
            by_receiver=[[[cell0, cell1]]],
            receiver_depths=np.array([50.0]),
            receiver_ranges=np.array([1000.0, 2000.0]),
            model='Test', frequencies=100.0,
        )

    def test_flat_list_length(self):
        a = self._arrivals()
        assert len(a) == 4    # 3 from cell0 + 1 from cell1

    def test_phases_returns_the_stored_radians_unchanged(self):
        # The .arr file stores phase in degrees (ArrMod.f90:120 writes
        # RadDeg*Phase) and read_arr_file converts once; the cell and the
        # public .phases accessor both hold radians for exp(1j*phase).
        from uacpy.core.results import Arrivals
        cell = {
            "delays": np.array([0.1]), "amplitudes": np.array([1.0]),
            "phases": np.array([np.pi / 2]),   # radians, as the reader stores
            "n_top_bounces": np.array([0], dtype=int),
            "n_bot_bounces": np.array([0], dtype=int),
            "source_angles": np.array([0.0]), "receiver_angles": np.array([0.0]),
        }
        a = Arrivals(by_receiver=[[[cell]]], receiver_depths=np.array([50.0]),
                     receiver_ranges=np.array([1000.0]),
                     model='Test', frequencies=100.0)
        assert a.phases[0] == pytest.approx(np.pi / 2)

    def test_angle_accessors_return_degrees_in_arrival_order(self):
        # The .arr file stores declination angles in degrees (ArrMod.f90:55-56)
        # and the accessors keep that unit; only phase is radians.
        a = self._arrivals()
        assert a.source_angles.shape == (len(a),)
        assert a.receiver_angles.shape == (len(a),)
        # Same order as every other bulk view, so columns line up elementwise.
        assert list(a.source_angles) == pytest.approx([0.0, 5.0, 10.0, 2.0])
        assert list(a.receiver_angles) == pytest.approx([0.0, -5.0, -10.0, -2.0])

    def test_angle_accessors_survive_filtering_alongside_delays(self):
        # A filter respawns the flat list; the angle columns must be carried
        # through it, or a grazing-angle filter would silently read the
        # unfiltered set.
        a = self._arrivals()
        bottom = a.filter_by_bounces(kind='bottom')
        assert len(bottom.receiver_angles) == len(bottom) == len(bottom.delays)

    def test_angle_accessor_without_the_column_names_the_cause(self):
        from uacpy.core.results import Arrivals
        cell = {
            "delays": np.array([0.1]), "amplitudes": np.array([1.0]),
            "phases": np.array([0.0]),
            "n_top_bounces": np.array([0], dtype=int),
            "n_bot_bounces": np.array([0], dtype=int),
        }
        a = Arrivals(by_receiver=[[[cell]]], receiver_depths=np.array([50.0]),
                     receiver_ranges=np.array([1000.0]),
                     model='Test', frequencies=100.0)
        # _flatten_by_receiver defaults a missing column to zeros, so the
        # accessor works; what must NOT happen is a bare KeyError.
        assert a.source_angles[0] == pytest.approx(0.0)

    def test_filter_by_bounces_kind(self):
        a = self._arrivals()
        direct = a.filter_by_bounces(kind='direct')
        assert len(direct) == 1
        assert direct.arrivals[0]['delay'] == pytest.approx(0.1)
        bottom = a.filter_by_bounces(kind='bottom')
        assert len(bottom) == 1
        assert bottom.arrivals[0]['delay'] == pytest.approx(0.4)

    def test_filter_by_bounces_top_low_high(self):
        a = self._arrivals()
        # 0-1 surface bounces — keeps everything but the 'both' arrival
        # has top=1 too, so all four pass; instead use bot=(1, None).
        few_bot = a.filter_by_bounces(bot=(1, None))
        assert len(few_bot) == 2     # cell0 last + cell1 last
        exact_top = a.filter_by_bounces(top=1)
        assert len(exact_top) == 2   # cell0 idx 1 and 2

    def test_window_keeps_the_delays_inside_the_pair(self):
        a = self._arrivals()
        mid = a.window(delay=(0.15, 0.35))
        assert len(mid) == 2
        assert all(0.15 <= x['delay'] <= 0.35 for x in mid.arrivals)

    @staticmethod
    def _two_arrivals(*, delays, amplitudes):
        from uacpy.core.results import Arrivals
        cell = {
            "delays": np.asarray(delays, dtype=float),
            "amplitudes": np.asarray(amplitudes, dtype=float),
            "phases": np.zeros(len(delays)),
            "n_top_bounces": np.zeros(len(delays), dtype=int),
            "n_bot_bounces": np.zeros(len(delays), dtype=int),
            "source_angles": np.zeros(len(delays)),
            "receiver_angles": np.zeros(len(delays)),
        }
        return Arrivals(by_receiver=[[[cell]]],
                        receiver_depths=np.array([50.0]),
                        receiver_ranges=np.array([1000.0]),
                        model='Test', frequencies=100.0)

    @staticmethod
    def _absorbed_pair(loss_dB, *, phases=(0.0, 0.0)):
        """Two equal-amplitude arrivals; the second loses ``loss_dB`` to
        absorption, carried where Bellhop carries it — the imaginary delay."""
        from uacpy.core.results import Arrivals
        cell = {
            "delays": np.array([0.10, 0.30]),
            "amplitudes": np.array([1.0, 1.0]),
            "phases": np.asarray(phases, dtype=float),
            "n_top_bounces": np.zeros(2, dtype=int),
            "n_bot_bounces": np.zeros(2, dtype=int),
            "source_angles": np.zeros(2), "receiver_angles": np.zeros(2),
            "delays_imag": np.array(
                [0.0, -np.log(10 ** (loss_dB / 20.0)) / (2 * np.pi * 100.0)]),
        }
        return Arrivals(by_receiver=[[[cell]]],
                        receiver_depths=np.array([50.0]),
                        receiver_ranges=np.array([1000.0]),
                        model='Test', frequencies=100.0)

    def test_received_amplitude_applies_the_absorption_in_the_imaginary_delay(
            self):
        """``amplitudes`` is geometric; the loss lives in ``delay_imag``."""
        a = self._absorbed_pair(40.0)
        assert np.allclose(np.abs(a.amplitudes), [1.0, 1.0])
        received = np.abs(a.received_amplitudes)
        # 40 dB down is a factor of exactly 1/100, and the near arrival is
        # untouched — both sides of the exponent, so a sign error fails here.
        assert np.allclose(received, [1.0, 0.01])

    def test_received_amplitude_can_reverse_which_arrival_is_loudest(self):
        """The trap this property exists to close.

        Two arrivals whose geometric amplitudes rank one way and whose
        received levels rank the other. Reading ``amplitudes`` picks the wrong
        path, and the ordering is what a caller acts on."""
        from uacpy.core.results import Arrivals
        cell = {
            "delays": np.array([0.10, 0.30]),
            # The late path is 20 dB LOUDER geometrically ...
            "amplitudes": np.array([1.0, 10.0]),
            "phases": np.zeros(2),
            "n_top_bounces": np.zeros(2, dtype=int),
            "n_bot_bounces": np.zeros(2, dtype=int),
            "source_angles": np.zeros(2), "receiver_angles": np.zeros(2),
            # ... and loses 40 dB to absorption, so it arrives 20 dB quieter.
            "delays_imag": np.array([0.0, -np.log(10 ** (40 / 20.0))
                                     / (2 * np.pi * 100.0)]),
        }
        a = Arrivals(by_receiver=[[[cell]]],
                     receiver_depths=np.array([50.0]),
                     receiver_ranges=np.array([1000.0]),
                     model='Test', frequencies=100.0)
        assert np.argmax(np.abs(a.amplitudes)) == 1          # geometric
        assert np.argmax(np.abs(a.received_amplitudes)) == 0  # what arrives

    def test_received_amplitude_carries_the_arrival_phase(self):
        """It is complex, so it drops straight into a coherent sum."""
        a = self._absorbed_pair(0.0, phases=(0.0, np.pi / 2))
        assert np.isclose(a.received_amplitudes[0], 1.0 + 0.0j)
        assert np.isclose(a.received_amplitudes[1], 1.0j)

    def test_received_amplitude_works_without_a_phase_column(self):
        """A magnitude does not need a phase, and ``_arrival_power`` routes
        through this property, so a hand-built arrival set with no phase must
        still report what reaches the receiver."""
        from uacpy.core.results import Arrivals
        cell = {
            "delays": np.array([0.10]), "amplitudes": np.array([2.0]),
            "n_top_bounces": np.zeros(1, dtype=int),
            "n_bot_bounces": np.zeros(1, dtype=int),
            "source_angles": np.zeros(1), "receiver_angles": np.zeros(1),
        }
        a = Arrivals(by_receiver=[[[cell]]],
                     receiver_depths=np.array([50.0]),
                     receiver_ranges=np.array([1000.0]),
                     model='Test', frequencies=100.0)
        assert np.allclose(np.abs(a.received_amplitudes), [2.0])
        assert np.allclose(a._arrival_power(), [4.0])

    def test_arrival_power_is_the_received_amplitude_squared(self):
        a = self._absorbed_pair(40.0, phases=(0.0, np.deg2rad(37.0)))
        assert np.allclose(a._arrival_power(),
                           np.abs(a.received_amplitudes) ** 2)

    def test_the_spread_weighs_arrivals_by_the_energy_that_reaches_the_receiver(
            self):
        """Bellhop carries volume absorption in the IMAGINARY travel time, not
        in the amplitude column (``exp(w*Im tau)`` — the convention
        ``read_arr_file`` documents for ``delays_imag`` and
        ``arrival_grid_transfer_function`` applies). Weighting on amplitude alone
        scores a late, heavily absorbed path as if the water were lossless —
        and it is exactly the late paths that set a delay spread."""
        from uacpy.core.results import Arrivals
        cell = {
            "delays": np.array([0.10, 0.30]),
            "amplitudes": np.array([1.0, 1.0]),
            "phases": np.zeros(2),
            "n_top_bounces": np.zeros(2, dtype=int),
            "n_bot_bounces": np.zeros(2, dtype=int),
            "source_angles": np.zeros(2), "receiver_angles": np.zeros(2),
            # the late arrival loses ~40 dB to absorption on its longer path
            "delays_imag": np.array([0.0, -np.log(10 ** (40 / 20.0))
                                     / (2 * np.pi * 100.0)]),
        }
        a = Arrivals(by_receiver=[[[cell]]],
                     receiver_depths=np.array([50.0]),
                     receiver_ranges=np.array([1000.0]),
                     model='Test', frequencies=100.0)
        # Equal amplitudes would put the spread at dtau/2 = 0.1 s; the
        # absorbed arrival carries 1e-4 of the energy, so it barely counts.
        assert a.rms_delay_spread() < 0.01

    def test_rms_delay_spread_is_weighted_by_energy(self):
        """Two equal arrivals dtau apart sit +-dtau/2 about their mean, so the
        energy-weighted spread is dtau/2 exactly."""
        a = self._two_arrivals(delays=(0.10, 0.30), amplitudes=(1.0, 1.0))
        assert a.rms_delay_spread() == pytest.approx(0.1)

    def test_a_faint_late_arrival_moves_the_spread_far_less_than_the_range(
            self):
        """Weighting by energy is what keeps a negligible path from deciding
        the number — but it does not make it irrelevant, since the second
        moment squares the lever arm. A path 60 dB down arriving 500x later
        stretches the peak-to-peak range by 500x and this by under 2x."""
        loud = self._two_arrivals(delays=(0.10, 0.11), amplitudes=(1.0, 1.0))
        with_tail = self._two_arrivals(delays=(0.10, 0.11, 5.0),
                                       amplitudes=(1.0, 1.0, 1e-3))
        ptp_growth = (float(np.ptp(with_tail.delays))
                      / float(np.ptp(loud.delays)))
        rms_growth = (with_tail.rms_delay_spread()
                      / loud.rms_delay_spread())
        assert ptp_growth > 400.0
        assert rms_growth < 2.0

    def test_a_single_arrival_has_no_spread(self):
        a = self._two_arrivals(delays=(0.2,), amplitudes=(1.0,))
        assert a.rms_delay_spread() == 0.0

    def test_the_spread_ignores_a_common_time_offset(self):
        """A second central moment about the mean: moving the whole arrival
        set later changes the delays but not the dispersion."""
        early = self._two_arrivals(delays=(0.10, 0.30), amplitudes=(1.0, 2.0))
        late = self._two_arrivals(delays=(5.10, 5.30), amplitudes=(1.0, 2.0))
        assert late.rms_delay_spread() == pytest.approx(
            early.rms_delay_spread())

    def test_synthesis_band_outlasts_the_delay_spread(self):
        """A synthesised record is 1/df long, so a grid chosen without
        reference to the arrivals wraps the late ones onto the early ones.
        The spacing comes from the spread these arrivals actually have."""
        band = self._arrivals().synthesis_band(bandwidth=100.0)
        record = 1.0 / float(band[1] - band[0])
        spread = float(np.ptp(self._arrivals().delays))
        assert record > spread, f"record {record:g} s vs spread {spread:g} s"

    def test_synthesis_band_is_centred_on_the_result_frequency(self):
        band = self._arrivals().synthesis_band(bandwidth=100.0)
        assert float(band[0]) == pytest.approx(50.0)
        assert float(band[-1]) == pytest.approx(150.0)

    def test_a_band_reaching_below_zero_is_refused(self):
        """A band wider than twice its centre runs through 0 Hz into negative
        frequency. ``Source`` rejects those, but a model run accepts them and
        returns an H that is not conjugate-symmetric, which the IFFT then
        turns into a complex trace — so the band is refused where it is
        built, not left for the reader to notice."""
        with pytest.raises(ConfigurationError, match='below 0 Hz'):
            self._arrivals().synthesis_band(bandwidth=8e3)

    def test_a_band_reaching_exactly_zero_is_refused(self):
        with pytest.raises(ConfigurationError, match='below 0 Hz'):
            self._arrivals().synthesis_band(bandwidth=200.0)

    def test_margin_is_the_number_of_frequency_samples_per_fringe(self):
        """Two equal paths dtau apart beat in |H(f)| with period 1/dtau. The
        derived record is margin x dtau, so the grid spacing 1/record puts
        ``margin`` samples on each fringe — to within the one bin the band
        is rounded up to. 1.2 is critically sampled for viewing; 8 draws
        the fringe."""
        dtau, bandwidth = 0.25, 1000.0
        a = self._two_arrivals(delays=(0.10, 0.10 + dtau),
                               amplitudes=(1.0, 1.0))
        for margin in (1.2, 8.0):
            band = a.synthesis_band(bandwidth=bandwidth, centre=2000.0,
                                    margin=margin)
            per_fringe = (1.0 / dtau) / float(band[1] - band[0])
            assert per_fringe == pytest.approx(
                margin, abs=1.0 / (bandwidth * dtau))

    def test_a_shared_record_folds_each_cell_from_its_own_first_arrival(self):
        """Cells sharing one record each start it at their own first
        arrival, so an arrival 0.7 s into the far cell's record folds even
        though, measured from the near cell's first, it would not stand out
        from its neighbour; against the global first both far arrivals
        fold."""
        from uacpy.acoustic_signal.delay_profile import _fold_notice
        delays = np.array([0.10, 0.20, 1.00, 1.70])      # near cell, far cell
        first = np.array([0.10, 0.10, 1.00, 1.00])
        power = np.ones(4)
        per_cell = _fold_notice(delays, power, 0.5, who="t", remediation="r",
                                first=first)
        assert per_cell is not None
        assert "1 arrival(s) past its end" in per_cell and "-6 dB" in per_cell
        global_first = _fold_notice(delays, power, 0.5, who="t", remediation="r")
        assert "2 arrival(s) past its end" in global_first
        assert _fold_notice(delays, power, 2.0, who="t", remediation="r") is None

    def test_synthesis_band_takes_an_explicit_centre(self):
        band = self._arrivals().synthesis_band(bandwidth=2e3, centre=40e3)
        assert float(band[0]) == pytest.approx(39e3)
        assert float(band[-1]) == pytest.approx(41e3)

    def test_a_wider_delay_spread_buys_a_finer_grid(self):
        narrow = self._arrivals().window(delay=(0.1, 0.2))
        assert narrow.synthesis_band(bandwidth=100.0).size < \
            self._arrivals().synthesis_band(bandwidth=100.0).size

    def test_a_single_arrival_yields_a_two_point_grid(self):
        one = self._arrivals().top_n_by_amplitude(1)
        band = one.synthesis_band(bandwidth=100.0)
        assert band.size >= 2 and np.all(np.diff(band) > 0)

    def test_the_loudest_arrival_is_the_one_that_arrives_loudest(self):
        """Ranking reads the amplitude COLUMN, but the column is not what
        reaches the receiver: Bellhop carries volume absorption in the
        imaginary travel time, so a long path can carry a larger geometric
        amplitude and still arrive far quieter. At 40 kHz a 6 km bounce path
        with three times the direct's amplitude lands 55 dB below it."""
        from uacpy.core.results import Arrivals
        f0, alpha_dB_per_km = 40e3, 12.90        # Thorp at 40 kHz
        def dimag(arc_km):
            return -(alpha_dB_per_km * arc_km / 8.6858896) / (2 * np.pi * f0)
        cell = {
            "delays": np.array([1000 / 1500.0, 6000 / 1500.0]),
            "amplitudes": np.array([1.0e-3, 3.0e-3]),
            "phases": np.zeros(2),
            "n_top_bounces": np.array([0, 2]),
            "n_bot_bounces": np.array([0, 3]),
            "source_angles": np.zeros(2), "receiver_angles": np.zeros(2),
            "delays_imag": np.array([dimag(1.0), dimag(6.0)]),
        }
        a = Arrivals(by_receiver=[[[cell]]],
                     receiver_depths=np.array([50.0]),
                     receiver_ranges=np.array([1000.0]),
                     model='Test', frequencies=f0)
        loudest = a.top_n_by_amplitude(1).arrivals[0]
        assert loudest['delay'] == pytest.approx(1000 / 1500.0), (
            "the absorbed bounce path was ranked above the direct one")
        order = [x['delay'] for x in a.sorted_by_amplitude().arrivals]
        assert order[0] < order[1], order

    def test_the_energy_support_is_not_moved_by_a_faint_straggler(self):
        """The span the energy occupies, not the span between the first and
        last ray. A path 100 dB down arriving 40x later stretches the
        peak-to-peak range by 40x and leaves this one where it was."""
        loud = self._two_arrivals(delays=(0.10, 0.12), amplitudes=(1.0, 1.0))
        with_tail = self._two_arrivals(delays=(0.10, 0.12, 5.0),
                                       amplitudes=(1.0, 1.0, 1e-5))
        assert with_tail.energy_support() == pytest.approx(
            loud.energy_support())
        assert (float(np.ptp(with_tail.delays))
                > 40.0 * float(np.ptp(loud.delays)))

    def test_the_energy_support_spans_every_arrival_at_fraction_one(self):
        """fraction=1 asks for all of the energy, which is the peak-to-peak
        span — the measure the default exists to avoid, still available to a
        caller who wants every arrival inside the window."""
        a = self._two_arrivals(delays=(0.10, 0.12, 5.0),
                               amplitudes=(1.0, 1.0, 1e-5))
        assert a.energy_support(1.0) == pytest.approx(float(np.ptp(a.delays)))

    def test_the_energy_support_counts_absorbed_paths_as_the_energy_they_carry(
            self):
        """Absorption rides in the imaginary delay, not the amplitude column,
        so a late path can have a full-size amplitude and carry nothing. The
        window must not be stretched to reach one."""
        from uacpy.core.results import Arrivals
        cell = {
            "delays": np.array([0.10, 0.12, 5.0]),
            "amplitudes": np.array([1.0, 1.0, 1.0]),
            "phases": np.zeros(3),
            "n_top_bounces": np.zeros(3, dtype=int),
            "n_bot_bounces": np.zeros(3, dtype=int),
            "source_angles": np.zeros(3), "receiver_angles": np.zeros(3),
            # the last arrival loses 80 dB to absorption on its longer path
            "delays_imag": np.array(
                [0.0, 0.0, -np.log(10 ** (80 / 20.0)) / (2 * np.pi * 100.0)]),
        }
        a = Arrivals(by_receiver=[[[cell]]],
                     receiver_depths=np.array([50.0]),
                     receiver_ranges=np.array([1000.0]),
                     model='Test', frequencies=100.0)
        assert a.energy_support() == pytest.approx(0.02)

    def test_an_energy_fraction_outside_the_unit_interval_is_refused(self):
        a = self._two_arrivals(delays=(0.1, 0.2), amplitudes=(1.0, 1.0))
        for bad in (0.0, -0.5, 1.5):
            with pytest.raises(ConfigurationError, match='fraction'):
                a.energy_support(bad)

    def test_a_stated_record_sets_the_frequency_spacing(self):
        """The window is the primitive and the spacing follows from it, so a
        caller who knows the window states it and gets 1/df back."""
        band = self._arrivals().synthesis_band(bandwidth=100.0, record=2.0)
        assert 1.0 / float(band[1] - band[0]) >= 2.0

    def test_a_stated_record_and_an_energy_fraction_together_are_refused(self):
        """Two answers to one question — how long the record has to be."""
        with pytest.raises(ConfigurationError, match='record'):
            self._arrivals().synthesis_band(bandwidth=100.0, record=2.0,
                                            energy_fraction=0.99)

    def test_the_derived_grid_is_sized_by_energy_not_by_the_last_arrival(self):
        """The cost of the grid is bins, and sizing on the peak-to-peak span
        spends them holding a ray that carries nothing: here the straggler is
        100 dB down and 400x later, and paying for it would cost 400x the
        bins."""
        a = self._two_arrivals(delays=(0.10, 0.12, 5.0),
                               amplitudes=(1.0, 1.0, 1e-5))
        with pytest.warns(UserWarning, match='fold back'):
            derived = a.synthesis_band(bandwidth=1e3, centre=2e3)
        every = a.synthesis_band(bandwidth=1e3, centre=2e3,
                                 energy_fraction=1.0)
        assert derived.size * 100 < every.size

    def test_a_record_that_leaves_arrivals_out_reports_what_wraps(self):
        """Trading the tail for a shorter record is a choice, not an
        approximation: the arrivals past the end do not vanish, they fold
        onto the early trace, so the level they fold in at is stated."""
        a = self._two_arrivals(delays=(0.10, 0.12, 5.0),
                               amplitudes=(1.0, 1.0, 1e-5))
        with pytest.warns(UserWarning, match=r'-103 dB'):
            a.synthesis_band(bandwidth=1e3, centre=2e3, record=0.05)

    def test_a_record_holding_every_arrival_says_nothing(self):
        a = self._two_arrivals(delays=(0.10, 0.12), amplitudes=(1.0, 1.0))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            a.synthesis_band(bandwidth=1e3, centre=2e3, record=1.0)

    def test_a_record_at_or_below_zero_is_refused(self):
        for bad in (0.0, -1.0):
            with pytest.raises(ConfigurationError,
                               match='synthesis_band record must be positive'):
                self._arrivals().synthesis_band(bandwidth=100.0, record=bad)

    def test_top_n_by_amplitude(self):
        a = self._arrivals()
        top2 = a.top_n_by_amplitude(2)
        assert len(top2) == 2
        amps = [x['amplitude'] for x in top2.arrivals]
        assert amps == sorted(amps, reverse=True)
        assert amps[0] == 1.0   # cell0 first

    def test_filter_chain_returns_arrivals(self):
        from uacpy.core.results import Arrivals
        a = self._arrivals()
        chained = a.filter(lambda x: x['n_bot_bounces'] >= 1).top_n_by_amplitude(1)
        assert isinstance(chained, Arrivals)
        assert len(chained) == 1


class TestSpawnedArrivalCellsMatchTheReadersRecord:
    """``Arrivals._rebuild_by_receiver`` rebuilds the per-cell record that
    ``io/oalib_reader.read_arr_file`` produces, so a filtered or sorted
    ``Arrivals`` feeds the same consumers as a freshly-read one.

    ``acoustic_signal.delayandsum`` reads ``cell['n_arrivals']`` as its first
    statement and indexes the columns from there, so a rebuilt cell missing
    that key raises ``KeyError`` on a public object. Types are asserted
    alongside keys: a 0-d array or a wider integer would make the key sets
    agree while the values diverged, which is the same defect one layer
    down."""

    #: The record ``read_arr_file`` builds for a cell holding two arrivals,
    #: key for key and dtype for dtype (``io/oalib_reader.py``).
    READER_CELL = {
        "amplitudes": np.array([1.0, 0.5], dtype='float64'),
        "phases": np.array([0.0, 30.0], dtype='float64'),
        "delays": np.array([0.1, 0.2], dtype='float64'),
        "delays_imag": np.array([0.0, 0.0], dtype='float64'),
        "source_angles": np.array([0.0, 5.0], dtype='float64'),
        "receiver_angles": np.array([0.0, -5.0], dtype='float64'),
        "n_top_bounces": np.array([0, 1], dtype='int32'),
        "n_bot_bounces": np.array([1, 2], dtype='int32'),
        "n_arrivals": 2,
    }

    def _arrivals(self):
        from uacpy.core.results import Arrivals
        empty = {k: (np.array([], dtype=v.dtype) if isinstance(v, np.ndarray)
                     else 0)
                 for k, v in self.READER_CELL.items()}
        return Arrivals(
            by_receiver=[[[dict(self.READER_CELL), empty]]],
            receiver_depths=np.array([50.0]),
            receiver_ranges=np.array([1000.0, 2000.0]),
            model='Test', frequencies=100.0,
        )

    @staticmethod
    def _signature(cell):
        return {k: (v.dtype.str if isinstance(v, np.ndarray) else type(v))
                for k, v in sorted(cell.items())}

    def test_a_spawned_cell_carries_the_readers_keys_and_types(self):
        spawned = self._arrivals().filter(lambda a: True)
        assert self._signature(spawned.by_receiver[0][0][0]) == \
            self._signature(self.READER_CELL)

    def test_an_emptied_cell_carries_the_readers_keys_and_types(self):
        # The other side of the count boundary: a predicate that keeps
        # nothing still has to produce the reader's record, with zero rows.
        spawned = self._arrivals().filter(lambda a: False)
        cell = spawned.by_receiver[0][0][0]
        assert self._signature(cell) == self._signature(self.READER_CELL)
        assert cell['n_arrivals'] == 0
        assert cell['delays'].size == 0

    def test_n_arrivals_counts_the_rows_that_survived(self):
        spawned = self._arrivals().filter(
            lambda a: a['delay'] < 0.15)
        cell = spawned.by_receiver[0][0][0]
        assert cell['n_arrivals'] == 1
        assert cell['n_arrivals'] == len(cell['delays'])
        assert type(cell['n_arrivals']) is int

    def test_delayandsum_accepts_a_spawned_cell(self):
        from uacpy.acoustic_signal import delayandsum
        source = np.ones(8)
        fresh, _ = delayandsum(dict(self.READER_CELL), source, 10000.0, 1000.0)
        spawned_cell = self._arrivals().filter(
            lambda a: True).by_receiver[0][0][0]
        rebuilt, _ = delayandsum(spawned_cell, source, 10000.0, 1000.0)
        assert rebuilt.shape == fresh.shape
        assert np.allclose(rebuilt, fresh)


class TestRaysFilterAndSortHelpers:
    """Pure-data filtering/sorting on hand-built ray fans (no solver):
    ``filter_by_miss_distance`` / ``sorted_by_miss`` / ``first_n``,
    plus ``Arrivals.sorted_by_amplitude``."""

    @staticmethod
    def _fan(**kwargs):
        from uacpy.core.results import Rays
        # Each polyline has a vertex exactly at r=1000, so the depth offset
        # at that vertex is 3 m, 10 m and 30 m for alpha 1, 2 and 3. The miss
        # distance is measured to the SEGMENTS, so only alpha 1 equals its
        # vertex offset: for the other two the final segment passes nearer to
        # (1000, 50) than its own end point does.
        def ray(alpha, z_end):
            return {'r': np.array([0.0, 500.0, 1000.0]),
                    'z': np.array([10.0, 30.0, z_end]),
                    'launch_angle': alpha, 'n_top_bounces': 0, 'n_bot_bounces': 0}
        return Rays(rays=[ray(1.0, 47.0), ray(2.0, 60.0), ray(3.0, 80.0)],
                    model='Bellhop', frequencies=100.0, **kwargs)

    def _bracketed(self, **kwargs):
        """Two bracketing rays per path, as hat beams actually report them.

        Bellhop writes BOTH rays of the ray tube enclosing the receiver, so a
        dense fan reports one path once per beam. Here each path appears
        twice, with one member nearer the target than the other.
        """
        from uacpy.core.results import Rays

        def ray(alpha, z_end, top, bot):
            return {'r': np.array([0.0, 500.0, 1000.0]),
                    'z': np.array([10.0, 30.0, z_end]),
                    'launch_angle': alpha, 'n_top_bounces': top, 'n_bot_bounces': bot}
        return Rays(rays=[ray(-71.53, 51.0, 1, 0),    # far  member of the pair
                          ray(-71.52, 50.2, 1, 0),    # near member
                          ray(-71.54, 50.5, 1, 1),    # a different path, down
                          ray(+71.54, 50.4, 1, 1)],   # and its mirror, up
                    is_eigen=True, model='Bellhop', frequencies=100.0,
                    receiver_ranges=np.array([1000.0]),
                    receiver_depths=np.array([50.0]), **kwargs)

    def test_distinct_paths_keeps_the_nearer_ray_of_each_bracketing_pair(self):
        kept = self._bracketed().distinct_paths()
        # Three paths from four rays: (1,0,down) was reported twice.
        assert len(kept.rays) == 3
        pair = [r for r in kept.rays
                if (r['n_top_bounces'], r['n_bot_bounces']) == (1, 0)]
        assert len(pair) == 1
        assert pair[0]['launch_angle'] == pytest.approx(-71.52)   # the nearer one

    def test_distinct_paths_keeps_both_directions_of_one_bounce_pattern(self):
        """A path launched up and its mirror launched down are two paths, not
        one: they have different lengths and different phase."""
        kept = self._bracketed().distinct_paths()
        alphas = sorted(r['launch_angle'] for r in kept.rays
                        if (r['n_top_bounces'], r['n_bot_bounces']) == (1, 1))
        assert alphas == pytest.approx([-71.54, 71.54])

    def test_distinct_paths_preserves_the_eigenray_flag(self):
        assert self._bracketed().distinct_paths().is_eigen is True

    def test_distinct_paths_refuses_a_ray_fan(self):
        """A fan's rays are samples of a continuum, not paths: grouping them
        by bounce count would collapse the picture to a handful of rays."""
        fan = self._bracketed(); fan.is_eigen = False
        with pytest.raises(ConfigurationError, match="eigenray"):
            fan.distinct_paths()

    def test_filter_by_miss_distance_keeps_and_annotates(self):
        kept = self._fan().filter_by_miss_distance(
            5.0, target_range_m=1000.0, target_depth_m=50.0)
        assert [r['launch_angle'] for r in kept.rays] == [1.0]
        assert kept.rays[0]['miss_distance_m'] == pytest.approx(3.0,
                                                                abs=1e-12)
        # The parent's ray dicts are not annotated in place.
        assert 'miss_distance_m' not in self._fan().rays[0]

    def test_miss_distance_measures_segments_not_vertices(self):
        """A ray passing exactly through the target misses it by zero, however
        coarsely its polyline is sampled.

        This is the property that makes ``filter_by_miss_distance`` a
        geometric test rather than a report of the ray step: with vertex-only
        distances the answer below would be 500 m, the half-step, and a
        fine-tolerance filter would discard a ray that scores a direct hit.
        """
        from uacpy.core.results import Rays
        through = {'r': np.array([0.0, 500.0, 1500.0]),
                   'z': np.array([50.0, 50.0, 50.0]),
                   'launch_angle': 0.0, 'n_top_bounces': 0, 'n_bot_bounces': 0}
        fan = Rays(rays=[through], model='Bellhop', frequencies=100.0)
        kept = fan.filter_by_miss_distance(1e-9, target_range_m=1000.0,
                                           target_depth_m=50.0)
        assert len(kept.rays) == 1
        assert kept.rays[0]['miss_distance_m'] == pytest.approx(0.0, abs=1e-12)

    def test_miss_distance_clips_to_the_segment_not_its_infinite_line(self):
        """A target beyond the end of a ray measures to the END POINT.

        Without the clip the foot of the perpendicular runs off the end of the
        polyline and the distance collapses toward the infinite line the last
        segment lies on, which would report a ray that stops short as though
        it had carried on to the receiver."""
        from uacpy.core.results import Rays
        stops_short = {'r': np.array([0.0, 100.0]),
                       'z': np.array([50.0, 50.0]),
                       'launch_angle': 0.0, 'n_top_bounces': 0, 'n_bot_bounces': 0}
        fan = Rays(rays=[stops_short], model='Bellhop', frequencies=100.0)
        miss, index = fan._miss_distance_to(fan.rays[0], 1000.0, 50.0)
        assert miss == pytest.approx(900.0)      # to the end point, not 0
        assert index == 1                        # and it is a vertex index

    def test_miss_distance_survives_a_repeated_vertex(self):
        """A zero-length segment must not divide by zero."""
        from uacpy.core.results import Rays
        doubled = {'r': np.array([0.0, 500.0, 500.0, 1000.0]),
                   'z': np.array([50.0, 50.0, 50.0, 50.0]),
                   'launch_angle': 0.0, 'n_top_bounces': 0, 'n_bot_bounces': 0}
        fan = Rays(rays=[doubled], model='Bellhop', frequencies=100.0)
        miss, _ = fan._miss_distance_to(fan.rays[0], 700.0, 50.0)
        assert miss == pytest.approx(0.0, abs=1e-12)

    def test_sorted_by_miss_orders_ascending(self):
        fan = self._fan()
        # Shuffle so the sort has work to do.
        fan.rays.reverse()
        ordered = fan.sorted_by_miss(target_range_m=1000.0,
                                     target_depth_m=50.0)
        assert [r['launch_angle'] for r in ordered.rays] == [1.0, 2.0, 3.0]
        # alpha 1: the perpendicular foot lands past the segment end, so the
        # clip puts the closest approach ON the end point and the answer is
        # the 3 m vertex offset exactly. alpha 2 and 3 slant past the target,
        # so their segments pass closer than their end vertices do.
        np.testing.assert_allclose(
            [r['miss_distance_m'] for r in ordered.rays],
            [3.0, 9.98204845465779, 29.851115706299673], rtol=1e-9)

    def test_sorted_by_miss_defaults_to_single_point_receiver_context(self):
        fan = self._fan(receiver_depths=np.array([50.0]),
                        receiver_ranges=np.array([1000.0]))
        ordered = fan.sorted_by_miss()
        assert [r['launch_angle'] for r in ordered.rays] == [1.0, 2.0, 3.0]

    def test_miss_helpers_without_target_or_context_raise(self):
        with pytest.raises(ConfigurationError, match='target_range_m'):
            self._fan().sorted_by_miss()

    def test_filter_nfirst_keeps_the_first_n_in_order(self):
        first_two = self._fan().first_n(2)
        assert [r['launch_angle'] for r in first_two.rays] == [1.0, 2.0]
        assert isinstance(first_two, type(self._fan()))
        # Composes with the sorters: the 2 closest rays.
        closest_two = self._fan().sorted_by_miss(
            target_range_m=1000.0, target_depth_m=50.0).first_n(2)
        assert [r['launch_angle'] for r in closest_two.rays] == [1.0, 2.0]

    def test_arrivals_sorted_by_amplitude_both_directions(self):
        from uacpy.core.results import Arrivals
        arr = Arrivals(
            arrivals=[{'delay': 0.1, 'amplitude': 0.5},
                      {'delay': 0.2, 'amplitude': 1.0},
                      {'delay': 0.3, 'amplitude': 0.2}],
            receiver_depths=np.array([50.0]),
            receiver_ranges=np.array([1000.0]),
            model='Bellhop', frequencies=100.0)
        down = arr.sorted_by_amplitude()
        assert [a['amplitude'] for a in down.arrivals] == [1.0, 0.5, 0.2]
        up = arr.sorted_by_amplitude(descending=False)
        assert [a['amplitude'] for a in up.arrivals] == [0.2, 0.5, 1.0]
        # The original order is untouched (sorts return copies).
        assert [a['amplitude'] for a in arr.arrivals] == [0.5, 1.0, 0.2]


class TestArrivalDictPhaseUnit:
    """The per-arrival dict carries ``'phase'`` in radians — the unit
    ``read_arr_file`` stores after converting the ``.arr`` degree column —
    the class docstring key list says so, and the ``phases`` accessor and
    ``received_amplitudes`` use it without another conversion."""

    def _arrivals(self):
        cell = {'delays': [0.1], 'amplitudes': [1.0], 'phases': [np.pi],
                'n_top_bounces': [0], 'n_bot_bounces': [0],
                'source_angles': [0.0], 'receiver_angles': [0.0]}
        return Arrivals(by_receiver=[[[cell]]], receiver_depths=[10.0],
                        receiver_ranges=[100.0], model='Test',
                        frequencies=100.0)

    def test_dict_phase_and_accessor_are_both_radians(self):
        arr = self._arrivals()
        assert arr.arrivals[0]['phase'] == pytest.approx(np.pi, rel=1e-12)
        assert arr.phases[0] == pytest.approx(np.pi, rel=1e-12)
        # A pi in the dict flips the sign; 180 read as radians would not.
        assert arr.received_amplitudes[0] == pytest.approx(-1.0 + 0j,
                                                           abs=1e-12)

    def test_class_docstring_names_the_radian_unit_for_phase(self):
        doc = Arrivals.__doc__
        segment = doc.split('``phase``', 1)[1].split('``n_top_bounces``')[0]
        assert 'radians' in segment
        assert 'degrees' not in segment


class TestArrivalsChannelView:
    """``Arrivals.channel_taps`` / ``coherence_bandwidth`` /
    ``channel_regime`` — the arrival list as a modem's channel."""

    @staticmethod
    def _two_path(symbol_rate=1000.0, gap_symbols=3, amp=0.6):
        from uacpy.core.results import Arrivals
        t0 = 0.4
        return Arrivals(
            arrivals=[
                {'delay': t0, 'amplitude': 1.0, 'phase': 0.0,
                 'n_top_bounces': 0, 'n_bot_bounces': 0, 'kind': 'direct'},
                {'delay': t0 + gap_symbols / symbol_rate, 'amplitude': amp,
                 'phase': np.pi, 'n_top_bounces': 1, 'n_bot_bounces': 0,
                 'kind': 'surface'}],
            receiver_depths=[50.0], receiver_ranges=[1000.0],
            model='Test', frequencies=1000.0)

    def test_nearest_sample_taps_reproduce_multipath_channel_tap_for_tap(self):
        from uacpy import comms
        arr = self._two_path()
        fc = 12345.0
        ct = arr.channel_taps(1000.0, fc=fc, sps=1, pulse='nearest')
        rotated = (arr.received_amplitudes
                   * np.exp(-2j * np.pi * fc * arr.delays))
        expect = comms.multipath_channel(
            rotated, arr.delays - arr.delays.min(), sample_rate=1000.0)
        assert np.array_equal(ct.taps, expect)
        assert ct.first_arrival_s == 0.4
        assert np.allclose(ct.delays_s, np.arange(4) / 1000.0)

    def test_the_surface_bounce_tap_is_minus_the_carrier_rotation(self):
        arr = self._two_path()
        fc = 12345.0
        ct = arr.channel_taps(1000.0, fc=fc, sps=1, pulse='nearest')
        tau = arr.delays[1]
        rotation = np.exp(-2j * np.pi * fc * tau)
        assert ct.taps[3] == pytest.approx(-0.6 * rotation)
        # The conjugate rotation is a different tap; the pin is not vacuous.
        assert ct.taps[3] != pytest.approx(-0.6 * np.conj(rotation))

    def test_an_on_grid_arrival_with_a_pulse_reproduces_rrc_filter(self):
        from uacpy.comms import rrc_filter
        arr = self._two_path(gap_symbols=3)
        fc, sps, span = 1000.0, 4, 6      # fc * tau integer: no rotation
        ct = arr.channel_taps(1000.0, fc=fc, sps=sps, pulse='rrc',
                              rolloff=0.35, span=span)
        g = rrc_filter(sps, 0.35, span)
        expect = np.zeros(3 * sps + g.size, complex)
        expect[:g.size] += g
        expect[3 * sps:] += -0.6 * g
        assert ct.taps.shape == expect.shape
        assert np.allclose(ct.taps, expect, atol=1e-12)
        assert ct.delays_s[0] == pytest.approx(-span / 2 / 1000.0)

    def test_unit_energy_normalisation(self):
        arr = self._two_path()
        ct = arr.channel_taps(1000.0, fc=500.0, normalize=True)
        assert np.sum(np.abs(ct.taps) ** 2) == pytest.approx(1.0)

    def test_the_carrier_sets_the_absorption_in_the_taps(self):
        from uacpy.core.results import Arrivals
        arr = Arrivals(
            arrivals=[{'delay': 1.0, 'amplitude': 1.0, 'phase': 0.0,
                       'delay_imag': -1e-4}],
            receiver_depths=[1.0], receiver_ranges=[1.0],
            model='Test', frequencies=100.0)
        low = arr.channel_taps(100.0, fc=100.0, pulse='nearest').taps
        high = arr.channel_taps(100.0, fc=1000.0, pulse='nearest').taps
        assert abs(low[0]) == pytest.approx(np.exp(-2 * np.pi * 100.0 * 1e-4))
        assert abs(high[0]) == pytest.approx(
            np.exp(-2 * np.pi * 1000.0 * 1e-4))

    @staticmethod
    def _one_ms_spread():
        from uacpy.core.results import Arrivals
        # Equal powers at 0 and 2 ms: rms delay spread exactly 1 ms.
        return Arrivals(
            arrivals=[{'delay': 0.0, 'amplitude': 1.0},
                      {'delay': 2e-3, 'amplitude': 1.0}],
            receiver_depths=[1.0], receiver_ranges=[1.0],
            model='Test', frequencies=None)

    def test_coherence_bandwidth_defaults_to_the_inverse_rms_spread(self):
        arr = self._one_ms_spread()
        assert arr.rms_delay_spread() == pytest.approx(1e-3)
        assert arr.coherence_bandwidth() == pytest.approx(1000.0)
        assert arr.coherence_bandwidth(
            convention='inverse_spread') == pytest.approx(1000.0)
        assert arr.channel_regime(500.0).convention == 'inverse_spread'

    def test_the_rappaport_rules_are_named_conventions(self):
        arr = self._one_ms_spread()
        assert arr.coherence_bandwidth(
            convention='rappaport_0.5') == pytest.approx(200.0)
        assert arr.coherence_bandwidth(
            convention='rappaport_0.9') == pytest.approx(20.0)
        with pytest.raises(ConfigurationError, match="convention must be"):
            arr.coherence_bandwidth(convention='rappaport_0.7')
        with pytest.raises(TypeError,
                           match="unexpected keyword argument 'level'"):
            arr.coherence_bandwidth(level=0.5)

    def test_the_conventions_order_the_bandwidth_from_loosest_to_strictest(
            self):
        arr = self._one_ms_spread()
        loose = arr.coherence_bandwidth(convention='inverse_spread')
        half = arr.coherence_bandwidth(convention='rappaport_0.5')
        strict = arr.coherence_bandwidth(convention='rappaport_0.9')
        assert loose > half > strict

    @pytest.mark.parametrize("k, expect", [(1.0, 1000.0), (5.0, 200.0),
                                           (50.0, 20.0)])
    def test_factor_sets_any_positive_divisor(self, k, expect):
        arr = self._one_ms_spread()
        assert arr.coherence_bandwidth(factor=k) == pytest.approx(expect)
        regime = arr.channel_regime(expect + 1.0, factor=k)
        assert regime.frequency_selective
        assert regime.coherence_bandwidth_hz == pytest.approx(expect)
        assert regime.convention == f"factor={k:g}"
        assert f"[factor={k:g}]" in str(regime)
        below = arr.channel_regime(expect - 1.0, factor=k)
        assert not below.frequency_selective

    @pytest.mark.parametrize("k", [0.0, -5.0, np.nan, np.inf])
    def test_a_non_positive_factor_is_refused(self, k):
        arr = self._one_ms_spread()
        with pytest.raises(ConfigurationError, match="factor must be"):
            arr.coherence_bandwidth(factor=k)
        with pytest.raises(ConfigurationError, match="factor must be"):
            arr.channel_regime(100.0, factor=k)

    def test_a_single_arrival_has_infinite_coherence_bandwidth(self):
        from uacpy.core.results import Arrivals
        arr = Arrivals(arrivals=[{'delay': 0.1, 'amplitude': 1.0}],
                       receiver_depths=[1.0], receiver_ranges=[1.0],
                       model='Test', frequencies=None)
        assert arr.coherence_bandwidth() == np.inf
        assert not arr.channel_regime(1e6).frequency_selective

    def test_channel_regime_flips_at_the_coherence_bandwidth(self):
        from uacpy.core.results import Arrivals
        arr = Arrivals(
            arrivals=[{'delay': 0.0, 'amplitude': 1.0},
                      {'delay': 2e-3, 'amplitude': 1.0}],
            receiver_depths=[1.0], receiver_ranges=[1.0],
            model='Test', frequencies=None)   # coherence bandwidth 1000 Hz
        flat = arr.channel_regime(999.0)
        selective = arr.channel_regime(1001.0)
        assert not flat.frequency_selective
        assert selective.frequency_selective
        assert not arr.channel_regime(1000.0).frequency_selective
        # A 25 % excess bandwidth widens the signal past the same threshold.
        assert arr.channel_regime(850.0, rolloff=0.25).frequency_selective
        assert not arr.channel_regime(750.0, rolloff=0.25).frequency_selective
        assert selective.isi_symbols == pytest.approx(1.001)
        assert selective.symbol_duration_s == pytest.approx(1 / 1001.0)
        assert selective.coherence_bandwidth_hz == pytest.approx(1000.0)
        text = str(selective)
        assert text.startswith("frequency-selective") and "1001" in text
        assert "[inverse_spread]" in text
        assert str(flat).startswith("frequency-flat")
        # The stricter convention moves the boundary, not the verdict logic.
        assert arr.channel_regime(
            201.0, convention='rappaport_0.5').frequency_selective
        assert not arr.channel_regime(
            199.0, convention='rappaport_0.5').frequency_selective

    @pytest.mark.parametrize("rate", [0.0, -1.0, np.nan])
    def test_a_non_positive_symbol_rate_is_refused(self, rate):
        arr = self._two_path()
        with pytest.raises(ConfigurationError, match="symbol_rate"):
            arr.channel_taps(rate, fc=1000.0)
        with pytest.raises(ConfigurationError, match="symbol_rate"):
            arr.channel_regime(rate)
        assert arr.channel_taps(1e-3, fc=1000.0).taps.size >= 1

    def test_sps_below_one_and_an_unknown_pulse_are_refused(self):
        arr = self._two_path()
        with pytest.raises(ConfigurationError, match="sps"):
            arr.channel_taps(1000.0, fc=1000.0, sps=0)
        with pytest.raises(ConfigurationError, match="sps"):
            arr.channel_taps(1000.0, fc=1000.0, sps=1.5)
        assert arr.channel_taps(1000.0, fc=1000.0, sps=1.0).sps == 1
        with pytest.raises(ConfigurationError, match="pulse must be"):
            arr.channel_taps(1000.0, fc=1000.0, pulse='raised')
        with pytest.raises(ConfigurationError, match="carrier"):
            arr.channel_taps(1000.0, fc=0.0)

    def test_a_grid_of_cells_needs_a_receiver_choice(self):
        arr = TestArrivalsFilterChain()._arrivals()      # 1 depth x 2 ranges
        with pytest.raises(ConfigurationError,
                           match=r"receiver=\(depth_m, range_m\)"):
            arr.channel_taps(1000.0, fc=100.0)
        one = arr.channel_taps(1000.0, fc=100.0, receiver=(50.0, 2000.0),
                               pulse='nearest')
        assert one.taps.size == 1 and one.first_arrival_s == 0.4
        three = arr.channel_taps(1000.0, fc=100.0,
                                 receiver=(50.0, 1000.0), pulse='nearest')
        assert three.first_arrival_s == 0.1
        with pytest.raises(ConfigurationError, match="range axis"):
            arr.channel_taps(1000.0, fc=100.0, receiver=(50.0, 1500.0))
        with pytest.raises(ConfigurationError,
                           match="no arrivals at receiver"):
            arr.filter_by_bounces(kind='both').channel_taps(
                1000.0, fc=100.0, receiver=(50.0, 2000.0))
        # A single-cell subset of the same grid needs no receiver=.
        sub = arr.filter(lambda a: a['range_idx'] == 1)
        assert sub.channel_taps(1000.0, fc=100.0).first_arrival_s == 0.4

    @staticmethod
    def _decimated_reference(arr, fc, rolloff, span, sps=16):
        """Symbol-spaced taps the long way: ``sps``-spaced root-raised-
        cosine taps, the receiver's matched filter, then one sample per
        symbol at the decision instants, on the time axis of the ``sps=1``
        taps (``(j - span/2)`` symbols from the first arrival)."""
        from uacpy.comms import rrc_filter
        ct = arr.channel_taps(1000.0, fc=fc, sps=sps, pulse='rrc',
                              rolloff=rolloff, span=span)
        full = np.convolve(ct.taps, rrc_filter(sps, rolloff, span))
        # Index m of the convolution sits at (m - span*sps)/sps symbols.
        j = np.arange((full.size - 1) // sps + 1)
        idx = sps * j + span * sps // 2 - span * sps
        idx = idx[(idx >= 0) & (idx < full.size)]
        return full[idx], (idx - span * sps) / sps

    @staticmethod
    def _nmse(taps, times_symbols, ref, ref_times):
        common = np.intersect1d(np.round(times_symbols, 6),
                                np.round(ref_times, 6))
        a = taps[np.isin(np.round(times_symbols, 6), common)]
        b = ref[np.isin(np.round(ref_times, 6), common)]
        return float(np.sum(np.abs(a - b) ** 2) / np.sum(np.abs(b) ** 2))

    @pytest.mark.parametrize("gap_symbols", [3.0, 3.37])
    def test_symbol_spaced_rc_taps_are_the_matched_filtered_channel(
            self, gap_symbols):
        """At ``sps=1`` the raised-cosine taps agree with the ``sps=16``
        root-raised-cosine taps matched-filtered and decimated (NMSE
        measured 8e-5 on-grid, 1.8e-4 at 3.37 symbols); the transmit
        root-raised-cosine alone does not (2.4e-2 and 1.7e-2), which is
        why ``'rc'`` and not ``'rrc'`` is the symbol-spaced default."""
        arr = self._two_path(gap_symbols=gap_symbols)
        fc, rolloff, span = 1000.0, 0.25, 8
        ref, ref_t = self._decimated_reference(arr, fc, rolloff, span)
        rc = arr.channel_taps(1000.0, fc=fc, sps=1, pulse='rc',
                              rolloff=rolloff, span=span)
        rrc = arr.channel_taps(1000.0, fc=fc, sps=1, pulse='rrc',
                               rolloff=rolloff, span=span)
        nmse_rc = self._nmse(rc.taps, rc.delays_s * 1000.0, ref, ref_t)
        nmse_rrc = self._nmse(rrc.taps, rrc.delays_s * 1000.0, ref, ref_t)
        assert nmse_rc < 1e-3
        assert nmse_rrc > 1e-2

    def test_the_default_pulse_follows_the_sample_spacing(self):
        arr = self._two_path(gap_symbols=3.37)
        kw = dict(fc=1000.0, rolloff=0.25, span=8)
        one = arr.channel_taps(1000.0, sps=1, **kw)
        assert np.array_equal(one.taps, arr.channel_taps(
            1000.0, sps=1, pulse='rc', **kw).taps)
        four = arr.channel_taps(1000.0, sps=4, **kw)
        assert np.array_equal(four.taps, arr.channel_taps(
            1000.0, sps=4, pulse='rrc', **kw).taps)
        assert not np.allclose(one.taps, arr.channel_taps(
            1000.0, sps=1, pulse='rrc', **kw).taps)

    def test_rc_taps_of_a_whole_symbol_delay_are_the_nearest_sample_taps(self):
        """On the symbol grid the raised cosine is Nyquist — one at its
        centre, zero at every other symbol — so the ``'rc'`` and
        ``'nearest'`` channels coincide there and differ off it."""
        arr = self._two_path(gap_symbols=3)
        rc = arr.channel_taps(1000.0, fc=1000.0, pulse='rc', span=8)
        near = arr.channel_taps(1000.0, fc=1000.0, pulse='nearest')
        in_symbols = rc.delays_s * 1000.0
        assert np.allclose(in_symbols, np.round(in_symbols), atol=1e-9)
        lead = int(np.argmin(np.abs(rc.delays_s)))
        assert np.allclose(rc.taps[lead:lead + near.taps.size], near.taps,
                           atol=1e-12)
        rest = np.delete(rc.taps, range(lead, lead + near.taps.size))
        assert np.allclose(rest, 0.0, atol=1e-12)
        off = self._two_path(gap_symbols=3.5)
        rc_off = off.channel_taps(1000.0, fc=1000.0, pulse='rc', span=8)
        near_off = off.channel_taps(1000.0, fc=1000.0, pulse='nearest')
        lead = int(np.argmin(np.abs(rc_off.delays_s)))
        assert not np.allclose(rc_off.taps[lead:lead + near_off.taps.size],
                               near_off.taps, atol=1e-3)


@pytest.mark.parametrize('method', ['filter_by_miss_distance', 'sorted_by_miss',
                                    'top_n_by_miss', 'truncate_at_receiver',
                                    'distinct_paths'])
def test_the_rays_target_point_cannot_be_passed_by_position(method):
    """A point is depth-first on a Field and range-first here, so a
    positional ``(5000, 50)`` would score every ray against the wrong point
    with no error. Keyword-only targets make the swap impossible."""
    from uacpy.core.results import Rays
    params = inspect.signature(getattr(Rays, method)).parameters
    for name in ('target_range_m', 'target_depth_m'):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name


class TestArrivalsRoundTripThroughDictAndNpSavez:
    """``Arrivals.to_dict`` writes one column per record key (named as the
    bulk accessors are), the receiver grid, the nested-view shape and the
    identity; ``from_dict`` rebuilds the records and the nested view, in
    memory and through the ``np.savez`` file the docstring documents."""

    @staticmethod
    def _arrivals():
        cell = {'delays': np.array([0.66, 0.67]),
                'delays_imag': np.array([0.0, -1e-5]),
                'amplitudes': np.array([1.0, 0.5]),
                'phases': np.array([0.0, np.pi]),
                'n_top_bounces': np.array([0, 1], dtype='int32'),
                'n_bot_bounces': np.array([0, 1], dtype='int32'),
                'source_angles': np.array([-2.0, 5.0]),
                'receiver_angles': np.array([2.0, -5.0]),
                'n_arrivals': 2}
        return uacpy.Arrivals(
            by_receiver=[[[cell]]], receiver_depths=[50.0],
            receiver_ranges=[1000.0], model='Bellhop',
            frequencies=12000.0,
            phase_reference=uacpy.PhaseReference.TRAVELLING_WAVE,
            metadata={'note': 'x'})

    @staticmethod
    def _via_savez(arrivals):
        import io
        buf = io.BytesIO()
        np.savez(buf, **arrivals.to_dict())
        buf.seek(0)
        with np.load(buf, allow_pickle=True) as npz:
            return uacpy.Arrivals.from_dict(dict(npz))

    def test_the_columns_are_named_as_the_bulk_accessors(self):
        d = self._arrivals().to_dict()
        np.testing.assert_array_equal(d['delays'], [0.66, 0.67])
        np.testing.assert_array_equal(d['phases'], [0.0, np.pi])
        assert list(d['kinds']) == ['direct', 'both']
        assert d['by_receiver_shape'] == (1, 1, 1)
        assert d['phase_reference'] == 'travelling_wave'

    @pytest.mark.parametrize('route', ['memory', 'savez'])
    def test_the_records_and_the_nested_view_come_back(self, route):
        a = self._arrivals()
        back = (uacpy.Arrivals.from_dict(a.to_dict()) if route == 'memory'
                else self._via_savez(a))
        assert back.arrivals == a.arrivals
        for key, value in a.by_receiver[0][0][0].items():
            np.testing.assert_array_equal(back.by_receiver[0][0][0][key], value)
        np.testing.assert_array_equal(back.received_amplitudes,
                                      a.received_amplitudes)
        assert back.phase_reference is uacpy.PhaseReference.TRAVELLING_WAVE
        assert back.metadata['note'] == 'x'

    def test_a_hand_built_set_keeps_only_the_keys_it_had(self):
        a = uacpy.Arrivals(
            arrivals=[{'delay': 1.0, 'amplitude': 1.0, 'phase': 0.0}],
            receiver_depths=[50.0], receiver_ranges=[1000.0])
        back = uacpy.Arrivals.from_dict(a.to_dict())
        assert back.arrivals == a.arrivals
        assert back.by_receiver is None



class TestArrivalsSynthesiseEveryCell:
    """``Arrivals.transfer_function()`` with no receiver and
    ``Arrivals.to_time_series`` take every cell of the per-receiver grid
    (M-20), through the user-level grid functions, NaN where no arrival
    reached; Bellhop's BROADBAND and TIME_SERIES are these methods."""

    FS = 8000.0
    FREQS = np.linspace(900.0, 1100.0, 5)

    @staticmethod
    def _cell(delays, amps):
        delays, amps = np.asarray(delays, float), np.asarray(amps, float)
        return dict(n_arrivals=delays.size, amplitudes=amps, delays=delays,
                    delays_imag=np.zeros_like(delays),
                    phases=np.zeros_like(delays),
                    n_top_bounces=np.zeros(delays.size, int),
                    n_bot_bounces=np.zeros(delays.size, int),
                    source_angles=np.zeros_like(delays),
                    receiver_angles=np.zeros_like(delays))

    def _arrivals(self, by_receiver, depths, ranges):
        return Arrivals(by_receiver=by_receiver, receiver_depths=depths,
                        receiver_ranges=ranges, frequencies=[1000.0])

    def _grid(self):
        empty = self._cell([], [])
        return self._arrivals(
            [[[self._cell([0.5], [1e-3]), self._cell([1.0, 1.2], [5e-4, 1e-4])],
              [empty, self._cell([0.7], [2e-4])]]],
            [20.0, 40.0], [700.0, 1500.0])

    def test_every_cell_is_its_own_transfer_function(self):
        from uacpy.acoustic_signal import arrival_transfer_function
        arr = self._grid()
        H = arr.transfer_function(self.FREQS)
        assert H.data.shape == (2, 2, 5)
        np.testing.assert_array_equal(H.coords['depth'], [20.0, 40.0])
        cell = arr.by_receiver[0][0][1]
        np.testing.assert_array_equal(
            H.data[0, 1], arrival_transfer_function(
                self.FREQS, cell['amplitudes'], cell['delays'],
                delays_imag_s=cell['delays_imag'], phases_rad=cell['phases'],
                trace_frequency=1000.0))
        assert np.isnan(H.data[1, 0]).all()
        assert not np.isnan(H.data[1, 1]).any()

    def test_one_cell_is_picked_by_receiver(self):
        H = self._grid().transfer_function(self.FREQS,
                                           receiver=(40.0, 1500.0))
        assert H.data.shape == (1, 1, 5)

    def test_a_paired_grid_rides_its_depths_on_the_result(self):
        arr = self._arrivals(
            [[[self._cell([0.5], [1e-3]), self._cell([0.9], [1e-3])]]],
            [20.0, 40.0], [700.0, 1500.0])
        H = arr.transfer_function(self.FREQS)
        assert H.data.shape == (2, 5)
        np.testing.assert_array_equal(H.aux_coords['receiver_depth'][1],
                                      [20.0, 40.0])

    def test_several_sources_are_refused_over_every_cell(self):
        one = [[self._cell([0.5], [1e-3])]]
        arr = self._arrivals([one, one], [20.0], [700.0])
        with pytest.raises(ConfigurationError, match='2 source depths'):
            arr.transfer_function(self.FREQS)
        with pytest.raises(ConfigurationError, match='2 source depths'):
            arr.to_time_series(np.ones(16), self.FS)

    def test_the_time_series_is_the_grid_synthesis(self):
        from uacpy.acoustic_signal import simulate_arrival_grid
        arr = self._grid()
        pulse = np.hanning(32)
        t, traces = arr.to_time_series(pulse, self.FS)
        t_ref, ref = simulate_arrival_grid(pulse, arr.by_receiver[0],
                                           self.FS, 1000.0)
        np.testing.assert_array_equal(t, t_ref)
        np.testing.assert_array_equal(traces, ref)
        assert np.isnan(traces[1, 0]).all()
