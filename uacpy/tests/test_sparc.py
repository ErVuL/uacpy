"""SPARC time-domain-focused tests."""

import threading
import warnings

import pytest
import numpy as np

from uacpy.core.results import Field
from uacpy.core.exceptions import (
    ConfigurationError, UnsupportedFeatureError,
)
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
    from uacpy.core.exceptions import UnsupportedFeatureError
    env = Environment(name='sp_rej', bathymetry=200.0, ssp=1500.0)
    rcv = Receiver(depths=100.0, ranges=np.linspace(100, 2000, 20))
    with pytest.raises(UnsupportedFeatureError, match="source_type"):
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
    def _rig(ssp=1500.0):
        env = Environment(
            name='window', bathymetry=100.0, ssp=ssp,
            bottom=BoundaryProperties(acoustic_type='rigid'),
        )
        return (env, Source(depths=25.0, frequencies=50.0),
                Receiver(depths=np.array([50.0]), ranges=np.array([1000.0])))

    def _deck_t_max(self, tmp_path, ssp=1500.0, **kw):
        import re
        env, source, receiver = self._rig(ssp)
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

    def test_explicit_t_max_wins(self, tmp_path):
        t_max, _ = self._deck_t_max(tmp_path, t_max=7.0)
        assert t_max == pytest.approx(7.0, rel=1e-9)

    def test_t_start_does_not_shift_the_output_window(self, tmp_path):
        """``t_start`` sets where the march begins, never the ``[0, t_max]``
        output window the deck's time block declares."""
        default, _ = self._deck_t_max(tmp_path)
        shifted, _ = self._deck_t_max(tmp_path, t_start=-0.5)
        assert shifted == pytest.approx(default, rel=1e-9)

    def test_window_follows_the_environments_own_sound_speed(self, tmp_path):
        """The last arrival travels at the slowest speed in the column, so the
        window scales with that speed and not with a fixed 1500 m/s. Pinning it
        to a constant cuts a slow column's window short by exactly the speed
        ratio, and an arrival past ``t_max`` is simply absent from p(t)."""
        from uacpy.core import SoundSpeedProfile
        slow = SoundSpeedProfile.from_pairs([(0.0, 1450.0), (100.0, 1450.0)])
        fast = SoundSpeedProfile.from_pairs([(0.0, 1540.0), (100.0, 1540.0)])
        t_slow, _ = self._deck_t_max(tmp_path, ssp=slow)
        t_fast, _ = self._deck_t_max(tmp_path, ssp=fast)
        assert t_slow == pytest.approx(2.5 * 1000.0 / 1450.0, rel=1e-4)
        assert t_fast == pytest.approx(2.5 * 1000.0 / 1540.0, rel=1e-4)
        assert t_slow > t_fast

    def test_a_profiled_column_uses_its_slowest_sample(self, tmp_path):
        from uacpy.core import SoundSpeedProfile
        profile = SoundSpeedProfile.from_pairs(
            [(0.0, 1520.0), (50.0, 1480.0), (100.0, 1510.0)])
        t_max, _ = self._deck_t_max(tmp_path, ssp=profile)
        assert t_max == pytest.approx(2.5 * 1000.0 / 1480.0, rel=1e-4)

    def test_pinned_sound_speed_wins(self, tmp_path):
        from uacpy.core import SoundSpeedProfile
        slow = SoundSpeedProfile.from_pairs([(0.0, 1450.0), (100.0, 1450.0)])
        t_max, _ = self._deck_t_max(tmp_path, ssp=slow, sound_speed=1500.0)
        assert t_max == pytest.approx(2.5 * 1000.0 / 1500.0, rel=1e-4)


def _rigid_env(depth=100.0, speed=1500.0):
    return Environment(name='rigid', bathymetry=depth, ssp=speed,
                       bottom=BoundaryProperties(acoustic_type='rigid'))


def _point(depth=50.0, freq=30.0, r=1000.0):
    return (Source(depths=depth, frequencies=freq),
            Receiver(depths=np.array([depth]), ranges=np.array([r])))


def _sediment_layer_env(layer_speed, depth=100.0, water_speed=1500.0,
                        thickness=50.0):
    """A water column over one sediment layer over a rigid half-space, run
    through the same projection + rigidify that ``SPARC.run`` applies before
    any deck is written, so what comes back is what the writer actually sees.
    """
    bottom = SeabedColumn(
        layers=[SedimentLayer(thickness=thickness, sound_speed=layer_speed,
                              density=2.0, attenuation=0.2)],
        halfspace=BoundaryProperties(acoustic_type='rigid'),
    )
    env = Environment(name='sparc_sediment_layer', bathymetry=depth,
                      ssp=water_speed, bottom=bottom)
    sparc = SPARC(verbose=False)
    return sparc._sparc_rigidify_halfspace(sparc._project_environment(env))


def _halfspace_env(hs_speed=1900.0, depth=100.0, water_speed=1500.0):
    """A water column over a fluid half-space, put through the same
    projection + rigidify. SPARC's deck carries only vacuum / rigid
    boundaries, so what comes back has no seabed medium left at all."""
    env = Environment(
        name='sparc_halfspace', bathymetry=depth, ssp=water_speed,
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=hs_speed, density=2.0,
                                  attenuation=0.5),
    )
    sparc = SPARC(verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return sparc._sparc_rigidify_halfspace(sparc._project_environment(env))


def _alias_warnings(env, tmp_path, name, **sparc_kwargs):
    source, receiver = _point()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        SPARC(verbose=False, **sparc_kwargs)._write_sparc_env(
            tmp_path / f'{name}.env', env, source, receiver)
    return [w for w in caught if 'range alias' in str(w.message)]


class TestSparcRangeAlias:
    """``sparc.f90:116`` gives ``dk = 2*pi/RMax`` and ``EXTRACT`` (:595, :622)
    sums over that grid directly, so the output is periodic in range with
    period ``RMax``: the receiver at ``r`` carries a copy of the response from
    ``|RMax - r|``, arriving at ``(RMax - r)/c``. At margin ``m`` that is
    ``(m-1)*r/c`` against an auto window of ``2.5*r/c``, so every margin at or
    below 3.5 puts the replica inside the window. Measured at margin 3,
    r = 1000 m: the trace peaks at 0.0014 near t = 1.333 s where a converged
    margin-12 run has 0.0008."""

    def test_default_margin_clears_the_auto_window(self):
        margin = SPARC(verbose=False)._resolve_rmax_safety_margin()
        assert margin > 3.5, (
            f"default rmax_safety_margin={margin} leaves the range replica "
            f"at (margin-1)*r/c inside the 2.5*r/c auto window")
        # The same statement in arrival times, at the measured geometry:
        # r = 1000 m in an isovelocity 1500 m/s guide.
        r, c = 1000.0, 1500.0
        assert (margin - 1.0) * r / c > 2.5 * r / c

    def test_tight_pinned_margin_warns(self, tmp_path):
        env = _rigid_env()
        source, receiver = _point()
        with pytest.warns(UserWarning, match='range alias'):
            SPARC(verbose=False, rmax_safety_margin=3.0)._write_sparc_env(
                tmp_path / 'tight.env', env, source, receiver)

    def test_default_margin_is_silent(self, tmp_path):
        env = _rigid_env()
        source, receiver = _point()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            SPARC(verbose=False)._write_sparc_env(
                tmp_path / 'default.env', env, source, receiver)
        assert not [w for w in caught if 'range alias' in str(w.message)]

    def test_the_warning_names_a_margin_that_clears(self, tmp_path):
        env = _rigid_env()
        source, receiver = _point()
        with pytest.warns(UserWarning, match='rmax_safety_margin') as rec:
            SPARC(verbose=False, rmax_safety_margin=2.0)._write_sparc_env(
                tmp_path / 'tight2.env', env, source, receiver)
        message = ' '.join(str(w.message) for w in rec)
        needed = float(message.split('rmax_safety_margin>')[1].split(')')[0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            SPARC(verbose=False,
                  rmax_safety_margin=needed * 1.01)._write_sparc_env(
                tmp_path / 'wide.env', env, source, receiver)
        assert not [w for w in caught if 'range alias' in str(w.message)]


class TestSparcRangeAliasReadsEveryMedium:
    """The folded replica travels the distance ``RMax - r`` through the whole
    waveguide, so its earliest arrival is set by the fastest medium the deck
    marches, not by the fastest water speed. ``Scooter/sparc.f90:207`` takes
    ``cMax = MAX(cpR, cMax)`` over ``DO medium = 1, SSP%NMedia``, and
    ``write_layer_sections`` writes each sediment layer as one more medium.

    At the default margin 4 and r = 1000 m in a 1500 m/s column the replica
    lands at ``3000 / c_fast`` against a ``2.5 * 1000 / 1500 = 1.667 s``
    window, so the alias enters the window at exactly ``c_fast = 1800`` m/s —
    a 1.2 speed ratio, which an ordinary 1650 m/s sand seabed does not reach
    but a coarse 1900 m/s sediment does.
    """

    def test_the_fastest_sediment_layer_sets_the_fast_bound(self):
        env = _sediment_layer_env(1900.0)
        assert env.bottom.all_sound_speeds() == [1900.0]
        c_slow, c_fast = SPARC(verbose=False)._profile_speed_bounds(env)
        assert (c_slow, c_fast) == (1500.0, 1900.0)

    def test_a_fast_sediment_layer_warns_at_the_default_margin(self, tmp_path):
        found = _alias_warnings(_sediment_layer_env(1900.0), tmp_path, 'fast')
        assert len(found) == 1
        assert '1.579' in str(found[0].message), str(found[0].message)

    def test_a_layer_just_over_the_threshold_warns(self, tmp_path):
        assert _alias_warnings(_sediment_layer_env(1810.0), tmp_path, 'over')

    def test_a_layer_just_under_the_threshold_is_silent(self, tmp_path):
        assert not _alias_warnings(_sediment_layer_env(1790.0),
                                   tmp_path, 'under')

    def test_a_sand_speed_layer_is_silent(self, tmp_path):
        assert not _alias_warnings(_sediment_layer_env(1650.0),
                                   tmp_path, 'sand')

    def test_a_rigidified_halfspace_contributes_no_speed(self, tmp_path):
        """``_sparc_rigidify_halfspace`` leaves the deck with the water column
        as its only medium, so the 1900 m/s the caller wrote is not a speed
        SPARC marches and must not move the fast bound."""
        env = _halfspace_env(1900.0)
        assert env.bottom.all_sound_speeds() == []
        assert SPARC(verbose=False)._profile_speed_bounds(env) == (1500.0,
                                                                   1500.0)
        assert not _alias_warnings(env, tmp_path, 'rigidified')


class TestSparcWindowTruncation:
    """``t_max`` is 2.5 direct travel times, which does not bound a
    waveguide's last arrival: the tail is set by the slowest modal *group*
    velocity, and SPARC's vacuum / rigid boundaries leave it undamped.
    Measured on a 100 m rigid guide at 30 Hz, r = 1000 m: the trace is still
    at 50% of its peak over the final tenth of the default window."""

    @staticmethod
    def _field(tail_level):
        time = np.linspace(0.0, 1.0, 100)
        trace = np.exp(-8.0 * time)
        trace[-10:] = tail_level * trace.max()
        return Field(data=trace.reshape(1, 1, -1),
                     coords={'depth': np.array([50.0]),
                             'range': np.array([1000.0]),
                             'time': time})

    def test_a_still_ringing_trace_is_reported(self):
        with pytest.warns(UserWarning, match='still at'):
            SPARC(verbose=False)._warn_on_truncated_window(self._field(0.5))

    def test_a_decayed_trace_is_not_reported(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            SPARC(verbose=False)._warn_on_truncated_window(self._field(0.01))
        assert not [w for w in caught if 'still at' in str(w.message)]

    def test_an_all_nan_trace_is_not_reported(self):
        field = self._field(0.5)
        field.data = np.full_like(field.data, np.nan)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            SPARC(verbose=False)._warn_on_truncated_window(field)
        assert not [w for w in caught if 'still at' in str(w.message)]

    def test_an_all_nan_trace_raises_no_warning_of_its_own(self):
        """The all-NaN case is decided by masking, so numpy's 'All-NaN slice'
        RuntimeWarning is never raised and never has to be muted."""
        field = self._field(0.5)
        field.data = np.full_like(field.data, np.nan)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            SPARC(verbose=False)._warn_on_truncated_window(field)
        assert [str(w.message) for w in caught] == []

    def test_the_check_opens_no_process_global_warning_filter_window(self):
        """``warnings.filters`` is process-global: a ``catch_warnings()``
        window opened here would swallow every warning other threads raise
        for as long as it is held."""
        opened = []
        real_catch_warnings = warnings.catch_warnings

        class _CountingCatchWarnings(real_catch_warnings):
            def __enter__(self):
                opened.append(1)
                return super().__enter__()

        field = self._field(0.5)
        field.data = np.full_like(field.data, np.nan)
        warnings.catch_warnings = _CountingCatchWarnings
        try:
            SPARC(verbose=False)._warn_on_truncated_window(field)
        finally:
            warnings.catch_warnings = real_catch_warnings
        assert opened == []

    def test_a_warning_raised_on_another_thread_survives_the_check(self):
        """The check runs on one thread while another raises a RuntimeWarning
        it never asked to hide. Handing off through ``warnings.simplefilter``
        puts that warning inside whatever filter window the check installs, so
        no timing decides the outcome."""
        installed_filter = threading.Event()
        probe_raised = threading.Event()
        delivered = []
        real_simplefilter = warnings.simplefilter

        def releasing_simplefilter(*args, **kwargs):
            outcome = real_simplefilter(*args, **kwargs)
            installed_filter.set()
            probe_raised.wait(30.0)
            return outcome

        field = self._field(0.5)
        field.data = np.full_like(field.data, np.nan)

        def run_the_check():
            try:
                SPARC(verbose=False)._warn_on_truncated_window(field)
            finally:
                # Nothing was installed, so raise the probe unconditionally.
                installed_filter.set()

        with warnings.catch_warnings():
            warnings.simplefilter('always')
            warnings.showwarning = (
                lambda message, *a, **k: delivered.append(str(message)))
            warnings.simplefilter = releasing_simplefilter
            try:
                worker = threading.Thread(target=run_the_check)
                worker.start()
                assert installed_filter.wait(30.0)
                warnings.warn('probe from another thread', RuntimeWarning)
                probe_raised.set()
                worker.join(30.0)
            finally:
                warnings.simplefilter = real_simplefilter
        assert not worker.is_alive()
        assert delivered == ['probe from another thread']

    def test_the_peak_helper_masks_nan_and_keeps_infinity(self):
        """``_peak_ignoring_nan`` stands in for ``np.nanmax``: NaN is a gap,
        infinity is a value, an all-NaN input is NaN."""
        from uacpy.models.sparc import _peak_ignoring_nan
        assert _peak_ignoring_nan(np.array([1.0, np.nan, 3.0])) == 3.0
        assert _peak_ignoring_nan(np.array([1.0, np.nan, np.inf])) == np.inf
        assert np.isnan(_peak_ignoring_nan(np.full(4, np.nan)))


class TestSparcDeckContracts:
    """``f_min = 0`` is legal (``sparc.f90:114`` clamps ``kMin`` to 1e-20 for
    exactly that case and ``doc/sparc.htm``'s example deck reads "0.0 15.0");
    ``t_start > 0`` is not, since the march would start from rest after the
    pulse has already turned on (``sparc.f90:409``, ``cans.f90``); and
    ``SubTab`` expands ``0.0 t_max /`` inclusive of both ends, so the output
    sample rate is ``(n_t_out - 1)/t_max``."""

    def test_zero_f_min_reaches_the_deck(self, tmp_path):
        deck = tmp_path / 'fmin0.env'
        source, receiver = _point()
        SPARC(verbose=False, f_min=0.0, f_max=60.0)._write_sparc_env(
            deck, _rigid_env(), source, receiver)
        lines = deck.read_text().splitlines()
        assert lines[lines.index("'PN+B'") + 1] == '0.000000 60.000000'

    def test_negative_f_min_is_refused(self):
        with pytest.raises(ConfigurationError, match='f_min >= 0'):
            SPARC(f_min=-1.0)

    def test_positive_t_start_is_refused(self):
        with pytest.raises(ConfigurationError, match='t_start'):
            SPARC(t_start=0.2)

    @pytest.mark.parametrize('t_start', [0.0, -0.1, -1.0])
    def test_non_positive_t_start_is_kept(self, t_start):
        assert SPARC(t_start=t_start).t_start == t_start

    def test_sample_rate_counts_intervals_not_samples(self):
        # 20 samples over 1 s is 19 intervals -> 19 Hz, Nyquist 9.5 Hz, so a
        # 10 Hz band aliases. Counting samples would read 20 Hz and stay
        # silent on exactly this case.
        with pytest.warns(UserWarning, match='19.0 Hz'):
            SPARC(verbose=False, n_t_out=20)._resolve_n_t_out(10.0, 1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            SPARC(verbose=False, n_t_out=21)._resolve_n_t_out(10.0, 1.0)
        assert not [w for w in caught if 'alias' in str(w.message)]

    def test_multi_frequency_source_names_what_it_reads(self):
        source = Source(depths=50.0, frequencies=np.array([30.0, 300.0]))
        with pytest.warns(UserWarning, match='frequencies'):
            freq = SPARC(verbose=False)._resolve_pulse_frequency(source)
        assert freq == pytest.approx(30.0)

    def test_single_frequency_source_is_silent(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            freq = SPARC(verbose=False)._resolve_pulse_frequency(
                Source(depths=50.0, frequencies=30.0))
        assert freq == pytest.approx(30.0)
        assert not caught

    def test_snapshot_greens_function_cube_is_capped(self):
        receiver = Receiver(depths=np.linspace(10, 90, 30),
                            ranges=np.array([50_000.0]))
        with pytest.raises(UnsupportedFeatureError, match='GiB'):
            SPARC(output_mode='S', verbose=False)._reject_oversized_snapshot(
                receiver, nk=300_000, n_t_out=512)
        # A modest cube passes, and the looped modes are never capped here.
        SPARC(output_mode='S', verbose=False)._reject_oversized_snapshot(
            receiver, nk=300, n_t_out=512)
        SPARC(output_mode='R', verbose=False)._reject_oversized_snapshot(
            receiver, nk=300_000, n_t_out=512)


@pytest.mark.requires_binary
@pytest.mark.slow
def test_sparc_reports_a_truncated_shallow_guide_trace():
    """The end-to-end wiring of the truncation check: a 100 m rigid guide at
    30 Hz still carries half its peak amplitude at the end of the default
    window, and the run has to say so."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        source, receiver = _point()
        SPARC(verbose=False).run(_rigid_env(), source, receiver,
                                 run_mode=RunMode.TIME_SERIES)
    assert [w for w in caught if 'still at' in str(w.message)], (
        "a trace still ringing at t_max was returned without a word")


class TestSparcSizesItsMeshPerMediumAtTheBandTop:
    """``misc/ReadEnvironmentMod.f90:103`` sizes an automatic mesh at
    ``deltaz = c / freq0 / 20`` — 20 points per wavelength at the deck's
    NOMINAL frequency. ``kraken.f90:75`` and ``scooter.f90:106`` then rescale
    it per frequency, which is what lets those wrappers check the mesh at
    ``freq0`` and stop there. ``sparc.f90`` has no such rescaling anywhere, and
    this wrapper writes a pulse band reaching ``2*freq``, so an automatic mesh
    ran the top of that band at 10 points per wavelength: measured 0.375
    relative rms against a converged mesh at 60 Hz / 300 m, improving to 0.106
    when sized at the band top.

    One count per MEDIUM, not one scalar. AT sizes each medium from its own
    thickness and speed, and the deck writer broadcasts a scalar to every
    sediment layer — so the water column's count landed on a 10 m layer and
    over-resolved it by the thickness ratio until the march failed outright.
    """

    @staticmethod
    def _half_space():
        from uacpy.core import BoundaryProperties, Environment
        return Environment(bathymetry=100.0, ssp=1500.0,
                           bottom=BoundaryProperties(acoustic_type='rigid'))

    @staticmethod
    def _layered():
        from uacpy.core import BoundaryProperties, Environment
        from uacpy.core.environment import SeabedColumn, SedimentLayer
        return Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=SeabedColumn(
                layers=[SedimentLayer(thickness=10.0, sound_speed=1600.0,
                                      density=1.5, attenuation=0.2)],
                halfspace=BoundaryProperties(acoustic_type='half-space',
                                             sound_speed=1800.0, density=1.8,
                                             attenuation=0.3)))

    def test_the_count_scales_with_the_band_top(self):
        from uacpy.models import SPARC
        model = SPARC(verbose=False)
        low = model._resolve_n_mesh(self._half_space(), 40.0)
        high = model._resolve_n_mesh(self._half_space(), 120.0)
        assert high[0] > low[0]

    def test_a_thin_layer_gets_its_own_count_not_the_water_columns(self):
        # The regression this guards: broadcasting the water column's count to
        # a 10 m sediment layer over-resolves it by the thickness ratio.
        from uacpy.models import SPARC
        counts = SPARC(verbose=False)._resolve_n_mesh(self._layered(), 120.0)
        assert len(counts) == 2
        assert counts[1] < counts[0]

    def test_a_pinned_count_is_passed_through(self):
        from uacpy.models import SPARC
        assert SPARC(n_mesh=333, verbose=False)._resolve_n_mesh(
            self._layered(), 120.0) == 333
