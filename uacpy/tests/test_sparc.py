"""SPARC time-domain-focused tests."""

import warnings

import pytest
import numpy as np

from uacpy.core.results import Field
from uacpy.models import SPARC
from uacpy.models.base import RunMode
from uacpy.core import Environment, Source, Receiver, BoundaryProperties
from uacpy.core.environment import (
    SeabedColumn, Bottom, SedimentLayer,
)

pytestmark = pytest.mark.requires_binary


class TestSPARCBasic:
    """Basic tests for SPARC model (seismo-acoustic PE)."""

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_sparc_refuses_transmission_loss(self):
        """SPARC computes no CW transmission loss: the pulse-to-CW extraction
        is not quantitative (2.4 dB median with 13 dB excursions on a
        single-mode guide, against 0.07 dB for Scooter), so the native time
        series is the only product offered."""
        from uacpy.core.exceptions import UnsupportedFeatureError
        env = Environment(
            name="sparc_test",
            bathymetry=100.0,
            ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='rigid'),
        )
        with pytest.raises(UnsupportedFeatureError) as ei:
            SPARC(verbose=False).compute_tl(
                env=env,
                source=Source(depths=50.0, frequencies=50.0),
                receiver=Receiver(depths=np.linspace(10, 90, 5),
                                  ranges=np.linspace(100, 3000, 6)))
        # The error must name a model that does compute CW TL.
        assert 'Scooter' in str(ei.value) or 'Kraken' in str(ei.value), (
            f"refusal does not point anywhere useful: {ei.value}")

    def test_sparc_time_series_returns_time_series_field(self):
        """SPARC TIME_SERIES returns a real-valued Field."""
        env = Environment(
            name="sparc_ts",
            bathymetry=100.0,
            ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='rigid'),
        )
        # Smoke test (type/shape/finite only). SPARC time-marching cost grows
        # steeply with frequency and range, so keep both modest here.
        source = Source(depths=50.0, frequencies=30.0)
        receiver = Receiver(
            depths=np.linspace(10, 90, 3),
            ranges=np.linspace(500, 2500, 4),
        )

        sparc = SPARC(verbose=False)
        result = sparc.run(
            env, source, receiver,
            run_mode=RunMode.TIME_SERIES,
        )

        assert isinstance(result, Field)
        assert result.data.shape[0] == len(receiver.depths)
        assert result.data.shape[1] == len(receiver.ranges)
        assert result.data.shape[2] > 0
        # range coord (SPARC's actual grid) length-matches the data columns
        assert result.coords['range'].shape[0] == result.data.shape[1]
        assert np.isrealobj(result.data)
        assert np.all(np.isfinite(result.data))
        assert result.times is not None and result.times.size > 0


# ---------------------------------------------------------------------
# Auto-rigidify walks the inner halfspace of SeabedColumn /
# Bottom and flips its acoustic_type.
# ---------------------------------------------------------------------

class TestSPARCRigidifyLayered:
    """Pure-Python unit tests: do not invoke the SPARC binary."""

    def _hs_halfspace(self):
        return BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1800.0, density=1.8, attenuation=0.3,
        )

    def test_rigidify_layered_bottom_walks_to_halfspace(self):
        """``SeabedColumn`` has no top-level ``acoustic_type``; the
        rigid flag lives on its inner ``.halfspace`` and must be
        flipped there."""
        lb = SeabedColumn(
            layers=[
                SedimentLayer(thickness=10.0, sound_speed=1600.0,
                              density=1.5, attenuation=0.2),
            ],
            halfspace=self._hs_halfspace(),
        )
        env = Environment(
            name='sparc_lb_rigid',
            bathymetry=100.0, ssp=1500.0, bottom=lb,
        )
        sparc = SPARC(verbose=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = sparc._sparc_rigidify_halfspace(env)
        # Warning fired
        rigidify_msgs = [w for w in caught
                         if "auto-converting" in str(w.message)]
        assert len(rigidify_msgs) == 1
        # Inner halfspace acoustic_type flipped
        assert out.bottom.columns[0].halfspace.acoustic_type == 'rigid'
        # The downstream writer dispatches on bottom.halfspace_at — must
        # now report 'rigid'.
        assert out.bottom.halfspace_at(range=0.0).acoustic_type == 'rigid'
        # Original env left intact (copy semantics)
        assert env.bottom.columns[0].halfspace.acoustic_type == 'half-space'

    def test_rigidify_rd_layered_bottom_walks_each_profile(self):
        """``Bottom`` has a halfspace inside each
        per-range profile; every one must be flipped."""
        profA = SeabedColumn(
            layers=[SedimentLayer(thickness=5.0, sound_speed=1550.0,
                                  density=1.3, attenuation=0.1)],
            halfspace=self._hs_halfspace(),
        )
        profB = SeabedColumn(
            layers=[SedimentLayer(thickness=5.0, sound_speed=1700.0,
                                  density=1.7, attenuation=0.2)],
            halfspace=self._hs_halfspace(),
        )
        rdl = Bottom.from_columns([profA, profB], ranges=np.array([0.0, 10000.0]))
        env = Environment(
            name='sparc_rdl_rigid',
            bathymetry=100.0, ssp=1500.0, bottom=rdl,
        )
        sparc = SPARC(verbose=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = sparc._sparc_rigidify_halfspace(env)
        rigidify_msgs = [w for w in caught
                         if "auto-converting" in str(w.message)]
        assert len(rigidify_msgs) == 1
        for prof in out.bottom.columns:
            assert prof.halfspace.acoustic_type == 'rigid'
        # Originals untouched
        for prof in env.bottom.columns:
            assert prof.halfspace.acoustic_type == 'half-space'

    def test_rigidify_vacuum_layered_is_passthrough(self):
        """A SeabedColumn whose halfspace is already ``vacuum`` must
        NOT trigger the warning and must remain ``vacuum``."""
        lb = SeabedColumn(
            layers=[
                SedimentLayer(thickness=10.0, sound_speed=1600.0,
                              density=1.5, attenuation=0.2),
            ],
            halfspace=BoundaryProperties(acoustic_type='vacuum'),
        )
        env = Environment(
            name='sparc_lb_vacuum',
            bathymetry=100.0, ssp=1500.0, bottom=lb,
        )
        sparc = SPARC(verbose=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = sparc._sparc_rigidify_halfspace(env)
        rigidify_msgs = [w for w in caught
                         if "auto-converting" in str(w.message)]
        assert len(rigidify_msgs) == 0
        assert out.bottom.columns[0].halfspace.acoustic_type == 'vacuum'

    @pytest.mark.requires_binary
    def test_layered_bottom_runs_end_to_end(self, tmp_path):
        """SPARC + SeabedColumn completes a binary run. The emitted
        ``.env`` declares ``NMedia = 1 + n_sediment_layers`` so the
        Fortran reader consumes all medium blocks before parsing the
        bottom boundary marker."""
        lb = SeabedColumn(
            layers=[
                SedimentLayer(thickness=10.0, sound_speed=1600.0,
                              density=1.5, attenuation=0.2),
            ],
            halfspace=self._hs_halfspace(),
        )
        env = Environment(
            name='sparc_lb_e2e',
            bathymetry=100.0, ssp=1500.0, bottom=lb,
        )
        sparc = SPARC(verbose=False, work_dir=tmp_path, cleanup=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sparc.run(
                env,
                Source(depths=50.0, frequencies=200.0),
                Receiver(depths=np.array([50.0]),
                         ranges=np.array([1000.0])),
            )
        # SPARC's only run mode is TIME_SERIES, so the trailing axis is time.
        assert res.data.ndim >= 2 and res.data.shape[-1] > 1
        # The emitted .env declares the correct NMedia (=2 for one layer).
        env_path = next(tmp_path.glob('**/*.env'))
        first_lines = env_path.read_text().splitlines()
        # Line 3 is NMedia (after title + frequency).
        assert int(first_lines[2].strip()) == 2, (
            f"NMedia should be 2 (water + 1 sediment layer); got "
            f"{first_lines[2]!r}"
        )


def test_sparc_rejects_geometry_outside_snapshot_mode():
    """output_mode 'R'/'D' never Hankel-transform, so they honour no geometry."""
    from uacpy.core.exceptions import ConfigurationError
    env = Environment(name='sp_rej', bathymetry=200.0, ssp=1500.0)
    rcv = Receiver(depths=100.0, ranges=np.linspace(100, 2000, 20))
    with pytest.raises(ConfigurationError, match="source_type"):
        SPARC(output_mode='R').run(
            env, Source(depths=50, frequencies=200, source_type='line'), rcv)


def test_time_series_warns_when_the_output_grid_aliases():
    """A TIME_SERIES grid whose Nyquist is below the source band must warn.

    SPARC keeps the caller's ``n_t_out`` for TIME_SERIES by contract, but the
    default (512 samples over a ~10 s window => fs ~51 Hz) puts a 100 Hz source
    well above Nyquist. The returned p(t) is plausible-looking and at the wrong
    frequency, so silence is the wrong behaviour.
    """
    import warnings
    env = Environment(name='p', bathymetry=200.0, ssp=1500.0,
                      bottom=BoundaryProperties(acoustic_type='half-space',
                                                sound_speed=1800.0, density=1.8,
                                                attenuation=0.5))
    src = Source(depths=50.0, frequencies=100.0)
    rcv = Receiver(depths=100.0, ranges=np.array([2000.0]))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        SPARC(verbose=False).run(env, src, rcv, run_mode=RunMode.TIME_SERIES)
    alias = [x for x in w if 'alias' in str(x.message)]
    assert alias, "aliased TIME_SERIES grid did not warn"
    assert 'n_t_out>=' in str(alias[0].message), "warning must name the fix"


def test_time_series_does_not_warn_when_adequately_sampled():
    """With enough samples for the band, no aliasing warning."""
    import warnings
    env = Environment(name='p', bathymetry=200.0, ssp=1500.0,
                      bottom=BoundaryProperties(acoustic_type='half-space',
                                                sound_speed=1800.0, density=1.8,
                                                attenuation=0.5))
    src = Source(depths=50.0, frequencies=20.0)
    rcv = Receiver(depths=100.0, ranges=np.array([2000.0]))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        SPARC(verbose=False, n_t_out=4096).run(
            env, src, rcv, run_mode=RunMode.TIME_SERIES)
    assert not [x for x in w if 'alias' in str(x.message)]


class TestPulseTypeValidation:
    """``pulse_type`` is validated positionally against the alphabets read
    from ``Scooter/sparc.f90:126-148`` (shape) and
    ``tslib/sourceMod.f90:68-70,178`` (post-process / sign / filter); short
    strings are right-padded to 4 characters like sparcM.m does."""

    def test_short_string_is_ljust_padded_to_four(self):
        assert SPARC(pulse_type='R', verbose=False).pulse_type == 'R   '

    def test_default_code_survives_verbatim(self):
        assert SPARC(verbose=False).pulse_type == 'PN+B'

    @pytest.mark.parametrize('shape', ['T', 'C'])
    def test_cans_only_shapes_are_rejected(self, shape):
        """``T``/``C`` exist in tslib/cans.f90 but sparc.f90's GetPar rejects
        them with 'Unknown source type' before cans.f90 is reached."""
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match='position 1'):
            SPARC(pulse_type=f'{shape}N+B', verbose=False)

    @pytest.mark.parametrize('code,pos', [
        ('PX+B', 'position 2'),
        ('PN*B', 'position 3'),
        ('PN+Z', 'position 4'),
    ])
    def test_each_position_is_checked_against_its_own_alphabet(self, code, pos):
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match=pos):
            SPARC(pulse_type=code, verbose=False)

    def test_overlong_code_is_rejected(self):
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match='at most 4'):
            SPARC(pulse_type='PN+BN', verbose=False)

    def test_non_string_is_rejected(self):
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match='string'):
            SPARC(pulse_type=42, verbose=False)


class TestOnlyFirstSourceFrequencyIsUsed:
    """``source.frequencies[0]`` is the pulse's nominal centre frequency and
    the only entry SPARC reads (``docs/models/sparc.md`` §7); extra entries
    are dropped and the result records just the first."""

    @staticmethod
    def _rig(frequencies):
        env = Environment(
            name='freq0', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='rigid'))
        return (env, Source(depths=50.0, frequencies=frequencies),
                Receiver(depths=np.array([50.0]), ranges=np.array([1000.0])))

    def test_deck_carries_only_the_first_frequency(self, tmp_path):
        env, source, receiver = self._rig(np.array([30.0, 300.0]))
        deck = tmp_path / 'model.env'
        SPARC(verbose=False)._write_sparc_env(deck, env, source, receiver)
        lines = deck.read_text().splitlines()
        # Line 2 is the deck frequency; the octave pulse band derives from it.
        assert float(lines[1]) == pytest.approx(30.0)
        assert '15.000000 60.000000' in lines, lines

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_result_records_the_first_frequency(self):
        env, source, receiver = self._rig(np.array([30.0, 60.0]))
        result = SPARC(verbose=False).run(env, source, receiver,
                                          run_mode=RunMode.TIME_SERIES)
        assert np.asarray(result.frequencies) == pytest.approx([30.0])


class TestPulseBandDefaultsToOneOctave:
    """``f_min``/``f_max`` default to one octave around the source frequency
    (``max(f/2, 0.1)`` .. ``2f``, ``docs/models/sparc.md`` §5); the deck's
    band line sits directly after the quoted pulse-type line."""

    @staticmethod
    def _deck_lines(tmp_path, **kw):
        env = Environment(
            name='band', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='rigid'))
        deck = tmp_path / 'model.env'
        SPARC(verbose=False, **kw)._write_sparc_env(
            deck, env, Source(depths=50.0, frequencies=100.0),
            Receiver(depths=np.array([50.0]), ranges=np.array([1000.0])))
        return deck.read_text().splitlines()

    def test_default_band_is_half_to_double_the_centre(self, tmp_path):
        lines = self._deck_lines(tmp_path)
        i = lines.index("'PN+B'")
        assert lines[i + 1] == '50.000000 200.000000'

    def test_explicit_band_wins(self, tmp_path):
        lines = self._deck_lines(tmp_path, f_min=40.0, f_max=90.0)
        i = lines.index("'PN+B'")
        assert lines[i + 1] == '40.000000 90.000000'


class TestOutputTimeGridContract:
    """The output grid is ``n_t_out`` samples over ``[0, t_max]``; ``t_start``
    only sets where the integration begins (``docs/models/sparc.md`` §7)."""

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_time_axis_spans_zero_to_t_max_with_shifted_t_start(self):
        env = Environment(
            name='window', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='rigid'))
        result = SPARC(t_max=1.0, t_start=-0.4, verbose=False).run(
            env, Source(depths=50.0, frequencies=50.0),
            Receiver(depths=np.array([50.0]), ranges=np.array([1000.0])),
            run_mode=RunMode.TIME_SERIES)
        times = np.asarray(result.coords['time'], dtype=float)
        assert times.size == 512
        assert times[0] == pytest.approx(0.0, abs=1e-6)
        assert times[-1] == pytest.approx(1.0, rel=1e-4)


@pytest.mark.requires_binary
@pytest.mark.slow
def test_ricker_arrival_peaks_at_travel_time_plus_5_over_2pi_f():
    """``tslib/cans.f90:31-35`` defines the ``'R'`` pulse on ``U = ω·T − 5``,
    so it peaks at ``T = 5/(2πF)`` after the pulse origin and the direct
    arrival at range r peaks at ``r/c + 5/(2πF)``, not at ``r/c``. Deep
    isovelocity water keeps the boundary bounces (path 1077 m, 0.72 s) out of
    the 0.5 s window so the direct arrival is the only one in it."""
    env = Environment(
        name='ricker', bathymetry=1000.0, ssp=1500.0,
        bottom=BoundaryProperties(acoustic_type='rigid'))
    freq = 25.0
    # rmax_safety_margin=10 refines Δk (Nk ≈ 140) so the direct arrival is
    # clean; the r = RMax replica then sits at 4 km / 2.7 s, out of window.
    result = SPARC(pulse_type='RN+N', t_max=0.5, rmax_safety_margin=10.0,
                   verbose=False).run(
        env, Source(depths=500.0, frequencies=freq),
        Receiver(depths=np.array([500.0]), ranges=np.array([400.0])),
        run_mode=RunMode.TIME_SERIES)
    times = np.asarray(result.coords['time'], dtype=float)
    trace = np.abs(np.asarray(result.data, dtype=float)[0, 0])
    peak_t = float(times[np.argmax(trace)])
    travel = 400.0 / 1500.0
    offset = 5.0 / (2.0 * np.pi * freq)
    assert peak_t == pytest.approx(travel + offset, abs=0.010), (
        f"peak at {peak_t:.4f} s, expected {travel + offset:.4f} s "
        f"(= r/c {travel:.4f} + Ricker offset {offset:.4f})")
    # The offset itself: a peak at the bare travel time is a failure.
    assert peak_t - travel > offset / 2.0


class TestSPARCReceiverDepthAxis:
    """Same below-domain policy as Scooter and the Kraken family: the caller's
    depth axis comes back intact, with the depths the finite-difference mesh
    cannot resolve marked no-data rather than clamped onto the deepest
    interface."""

    @staticmethod
    def _env():
        from uacpy.core.bottom import Bottom, SeabedColumn, SedimentLayer
        column = SeabedColumn(
            layers=[SedimentLayer(thickness=20.0, sound_speed=1600.0,
                                  density=1.8, attenuation=0.5)],
            halfspace=BoundaryProperties(acoustic_type='rigid'))
        return Environment(name='media', bathymetry=200.0, ssp=1500.0,
                           bottom=Bottom(columns=[column]))

    def test_requested_depths_are_returned_with_below_mesh_as_nan(self):
        receiver = Receiver(depths=np.array([50.0, 150.0, 300.0]),
                            ranges=np.array([1000.0, 2000.0]))
        result = SPARC(verbose=False).run(
            self._env(), Source(depths=50.0, frequencies=50.0), receiver,
            run_mode=RunMode.TIME_SERIES)
        assert np.asarray(result.coords['depth']) == pytest.approx(
            receiver.depths)
        data = np.asarray(result.data)
        assert data.shape[0] == receiver.depths.size
        assert np.all(np.isfinite(data[:2]))
        assert np.all(np.isnan(data[2]))


class TestSPARCZeroRangeAgreesAcrossOutputModes:
    """A receiver on the source axis must be no-data in all three modes.

    ``Scooter/sparc.f90:622`` weights ``'R'`` by ``SQRT( rkT / Pos%Rr )`` and
    ``:292`` scales ``'D'`` by ``1 / SQRT( pi * Pos%Rr( 1 ) )``. Both divide by
    ``Rr``, so each blows up at ``r = 0`` in its own way — ``'D'`` to ``+Inf``,
    ``'R'`` to ``NaN`` mixed with exact zeros — while ``'S'``, whose Hankel
    transform runs in-tree, yields NaN. Without an explicit mask one model
    reports three different answers for one cell.
    """

    @staticmethod
    def _rig():
        env = Environment(
            name='zero_range', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.5,
                                      attenuation=0.5),
        )
        return (env, Source(depths=25.0, frequencies=50.0),
                Receiver(depths=np.array([30.0, 60.0]),
                         ranges=np.array([0.0, 500.0, 1000.0])))

    @pytest.mark.requires_binary
    @pytest.mark.slow
    @pytest.mark.parametrize('output_mode', ['R', 'D', 'S'])
    def test_zero_range_is_no_data_and_warns(self, output_mode):
        env, source, receiver = self._rig()
        with pytest.warns(UserWarning, match=r'r = 0'):
            result = SPARC(output_mode=output_mode,
                           verbose=False).run(env, source, receiver)
        data = np.asarray(result.data)
        assert np.isnan(data[:, 0, :]).all(), (
            f"output_mode={output_mode!r} left "
            f"{np.count_nonzero(~np.isnan(data[:, 0, :]))} finite value(s) "
            f"at r = 0")

    @pytest.mark.requires_binary
    @pytest.mark.slow
    @pytest.mark.parametrize('output_mode', ['R', 'D', 'S'])
    def test_the_other_ranges_are_untouched(self, output_mode):
        """Masking must not reach past the singular column."""
        env, source, receiver = self._rig()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = SPARC(output_mode=output_mode,
                           verbose=False).run(env, source, receiver)
        data = np.asarray(result.data)
        assert np.isfinite(data[:, 1:, :]).all()
        assert np.any(data[:, 1:, :] != 0.0)

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_no_warning_when_every_receiver_is_off_axis(self):
        env, source, _ = self._rig()
        receiver = Receiver(depths=np.array([30.0]),
                            ranges=np.array([500.0, 1000.0]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = SPARC(output_mode='R', verbose=False).run(
                env, source, receiver)
        assert not [w for w in caught if 'r = 0' in str(w.message)]
        assert np.isfinite(np.asarray(result.data)).all()


class TestSPARCHankelNormalisation:
    """Every output mode carries the full inverse-Hankel weight.

    The far-field kernel is ``H0(kr) ~ sqrt(2/(pi·k·r))·e^{i(kr-pi/4)}``, so
    inverting it weights each wavenumber sample by
    ``dk·k·sqrt(2/(pi·k·r)) = dk·sqrt(2k/(pi·r))``.

    ``sparc.f90``'s ``'D'`` branch carries exactly that — the ``:595`` kernel
    ``sqrt(2)·dk·sqrt(k)`` times the ``:292`` write scale ``1/sqrt(pi·Rr)``.
    Its ``'R'`` branch (``:622-623``) applies ``sqrt(2)·dk·sqrt(k/r)`` with no
    write scale, i.e. the same weight less the ``1/sqrt(pi)``, so the raw
    ``'R'`` trace is ``sqrt(pi)`` (+4.97 dB) hot. uacpy divides that back out.

    The constant is pinned against the raw ``.rts`` rather than only across
    modes: agreement between modes is satisfied by scaling all three onto the
    *wrong* branch, so it cannot detect this on its own.

    Tolerances: ``.rts`` is FORMATTED and written ``'( 12G15.6 )'``
    (``Scooter/sparc.f90:294,299``) from arrays already cast through ``SNGL``,
    so a value round-trips to about 6 significant digits. ``rtol=1e-5`` is an
    order of magnitude above that text precision and orders of magnitude below
    the ``sqrt(pi)`` = 1.77 factor under test.
    """

    @staticmethod
    def _rig():
        env = Environment(
            name='hankel', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.5,
                                      attenuation=0.5),
        )
        return (env, Source(depths=25.0, frequencies=50.0),
                Receiver(depths=np.array([30.0, 60.0]),
                         ranges=np.array([500.0, 1000.0])))

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_all_three_output_modes_agree_in_absolute_level(self):
        env, source, receiver = self._rig()
        peaks = {}
        for mode in ('R', 'D', 'S'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                data = np.asarray(SPARC(output_mode=mode,
                                        verbose=False).run(env, source,
                                                           receiver).data)
            peaks[mode] = float(np.nanmax(np.abs(data)))
        assert peaks['R'] == pytest.approx(peaks['D'], rel=1e-4), peaks
        assert peaks['S'] == pytest.approx(peaks['D'], rel=1e-4), peaks

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_range_native_divides_the_raw_rts_by_sqrt_pi(self, tmp_path):
        """Pins the correction against the file sparc.exe actually wrote."""
        from uacpy.io.oalib_reader import read_rts_file
        env, source, receiver = self._rig()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = SPARC(output_mode='R', verbose=False, cleanup=False,
                           work_dir=tmp_path).run(env, source, receiver)
        rts = sorted(tmp_path.rglob('*.rts'))
        assert rts, f"no .rts under {tmp_path}"
        raw = np.asarray(read_rts_file(rts[0])['p'])       # (nt, n_range)
        returned = np.asarray(result.data)[0]              # depth 0 -> (n_range, nt)
        np.testing.assert_allclose(returned, raw.T / np.sqrt(np.pi),
                                   rtol=1e-5, atol=0.0)

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_vertical_array_passes_the_raw_rts_through_unscaled(self, tmp_path):
        """The 'D' branch already carries the full weight, so nothing is applied."""
        from uacpy.io.oalib_reader import read_rts_file
        env, source, receiver = self._rig()
        single = Receiver(depths=receiver.depths, ranges=np.array([500.0]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = SPARC(output_mode='D', verbose=False, cleanup=False,
                           work_dir=tmp_path).run(env, source, single)
        rts = sorted(tmp_path.rglob('*.rts'))
        assert rts, f"no .rts under {tmp_path}"
        raw = np.asarray(read_rts_file(rts[0])['p'])       # (nt, n_depth)
        returned = np.asarray(result.data)[:, 0, :]        # (n_depth, nt)
        np.testing.assert_allclose(returned, raw.T, rtol=1e-5, atol=0.0)


class TestDefaultOutputWindowTracksTravelTime:
    """``t_max`` defaults to ``2.5 × r_max / c``. ``rmax_safety_margin`` is a
    wavenumber-sampling knob (``Δk ≈ 2π/RMax``); folding it into the output
    window stretched ``[0, t_max]`` by the margin while ``n_t_out`` stayed
    fixed, so the default grid always aliased (measured 4.3 dB peak error at
    50 Hz: peak |p| 8.910e-4 on the stretched window vs 1.468e-3 resolved)."""

    @staticmethod
    def _rig():
        env = Environment(
            name='window', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='rigid'),
        )
        return (env, Source(depths=25.0, frequencies=50.0),
                Receiver(depths=np.array([50.0]), ranges=np.array([1000.0])))

    def _deck_t_max(self, tmp_path, **kw):
        import re
        env, source, receiver = self._rig()
        deck = tmp_path / 'model.env'
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            SPARC(verbose=False, **kw)._write_sparc_env(
                deck, env, source, receiver)
        times = [re.match(r'^0\.0 (\S+) /$', ln)
                 for ln in deck.read_text().splitlines()]
        t_max = float([m for m in times if m][-1].group(1))
        return t_max, [w for w in caught if 'alias' in str(w.message)]

    def test_default_window_is_travel_time_based_and_alias_free(self, tmp_path):
        t_max, alias = self._deck_t_max(tmp_path)
        assert t_max == pytest.approx(2.5 * 1000.0 / 1500.0, rel=1e-4)
        assert not alias, [str(w.message) for w in alias]

    def test_margin_moves_rmax_but_not_the_window(self, tmp_path):
        t_max, _ = self._deck_t_max(tmp_path, rmax_safety_margin=10.0)
        assert t_max == pytest.approx(2.5 * 1000.0 / 1500.0, rel=1e-4)

    def test_explicit_t_max_still_wins(self, tmp_path):
        t_max, _ = self._deck_t_max(tmp_path, t_max=7.0)
        assert t_max == pytest.approx(7.0, rel=1e-9)

    def test_t_start_does_not_shift_the_output_window(self, tmp_path):
        """``t_start`` sets where the march begins, never the ``[0, t_max]``
        output window the deck's time block declares."""
        default, _ = self._deck_t_max(tmp_path)
        shifted, _ = self._deck_t_max(tmp_path, t_start=-0.5)
        assert shifted == pytest.approx(default, rel=1e-9)
