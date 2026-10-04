"""SPARC time-domain-focused tests."""

import re
import threading
import warnings

import pytest
import numpy as np

from uacpy.core.results import Field
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, UnsupportedFeatureError,
)
from uacpy.models import SPARC
from uacpy.io.oalib_writer import write_sparc_source_time_series
from uacpy.models.sparc import _pulse as sparc_pulse
from uacpy.models.sparc._extract import (
    rts_time_matches, warn_on_truncated_window,
)
from uacpy.models.sparc._plan import (
    checked_n_mesh, profile_speed_bounds, reject_halfspace_bottom,
    reject_oversized_snapshot, resolve_n_mesh, resolve_n_time_samples,
    resolve_rmax_factor,
)
from uacpy.models.sparc._pulse import (
    multi_frequency_notice, resolve_pulse_type, source_series_rows,
)
from uacpy.core.run_settings import RunMode
from uacpy.core import Environment, Source, Receiver, BoundaryProperties
from uacpy.core.environment import (
    SeabedColumn, Bottom, SedimentLayer,
)
from uacpy.tests.conftest import recorded_warnings
from uacpy.tests.conftest import make_pekeris

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
        with pytest.raises(
                UnsupportedFeatureError,
                match='does not support: transmission loss computation') as ei:
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
# A half-space bottom is refused, on every column of a Bottom.
# ---------------------------------------------------------------------

class TestSPARCRefusesAHalfSpaceBottom:
    """Pure-Python unit tests: do not invoke the SPARC binary."""

    def _hs_halfspace(self):
        return BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1800.0, density=1.8, attenuation=0.3,
        )

    def _layered(self, halfspace, speed=1600.0):
        return SeabedColumn(
            layers=[SedimentLayer(thickness=10.0, sound_speed=speed,
                                  density=1.5, attenuation=0.2)],
            halfspace=halfspace,
        )

    def test_a_layered_bottom_over_a_half_space_is_refused(self):
        """``SeabedColumn`` has no top-level ``acoustic_type``; the
        half-space it ends in lives on its inner ``.halfspace``."""
        env = Environment(name='sparc_lb_hs', bathymetry=100.0, ssp=1500.0,
                          bottom=self._layered(self._hs_halfspace()))
        with pytest.raises(ConfigurationError, match="half-space") as exc:
            reject_halfspace_bottom(env)
        assert "acoustic_type='rigid'" in str(exc.value)
        assert 'Scooter' in str(exc.value)

    def test_a_half_space_on_one_column_of_a_bottom_is_refused(self):
        """Every per-range column is checked: a rigid first column does
        not let a half-space further out through."""
        rigid = self._layered(BoundaryProperties(acoustic_type='rigid'))
        halfspace = self._layered(self._hs_halfspace(), speed=1700.0)
        bottom = Bottom.from_columns([rigid, halfspace],
                                     ranges=np.array([0.0, 10000.0]))
        env = Environment(name='sparc_rdl_hs', bathymetry=100.0,
                          ssp=1500.0, bottom=bottom)
        with pytest.raises(ConfigurationError, match="half-space"):
            reject_halfspace_bottom(env)

    @pytest.mark.parametrize('kind', ['vacuum', 'rigid'])
    def test_a_vacuum_or_rigid_floor_passes(self, kind):
        env = Environment(
            name=f'sparc_lb_{kind}', bathymetry=100.0, ssp=1500.0,
            bottom=self._layered(BoundaryProperties(acoustic_type=kind)))
        assert reject_halfspace_bottom(env) is None

    def test_run_refuses_the_default_bottom_before_any_deck(self, tmp_path):
        """The default Environment bottom is a fluid half-space."""
        env = Environment(name='sparc_default_bottom', bathymetry=100.0,
                          ssp=1500.0)
        sparc = SPARC(verbose=False, work_dir=tmp_path, cleanup=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ConfigurationError, match="half-space"):
                sparc.run(env, Source(depths=50.0, frequencies=200.0),
                          Receiver(depths=np.array([50.0]),
                                   ranges=np.array([1000.0])))
        assert not list(tmp_path.glob('**/*.env'))

    @pytest.mark.requires_binary
    def test_layered_bottom_runs_end_to_end(self, tmp_path):
        """SPARC + SeabedColumn completes a binary run. The emitted
        ``.env`` declares ``NMedia = 1 + n_sediment_layers`` so the
        Fortran reader consumes all medium blocks before parsing the
        bottom boundary marker."""
        lb = self._layered(BoundaryProperties(acoustic_type='rigid'))
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
    env = Environment(name='sp_rej', bathymetry=200.0, ssp=1500.0,
                      bottom=BoundaryProperties(acoustic_type='rigid'))
    rcv = Receiver(depths=100.0, ranges=np.linspace(100, 2000, 20))
    with pytest.raises(UnsupportedFeatureError, match="source_type"):
        SPARC(output_mode='R').run(
            env, Source(depths=50, frequencies=200, source_type='line'), rcv)


def test_time_series_warns_when_the_output_grid_aliases():
    """A TIME_SERIES grid whose Nyquist is below the source band must warn.

    SPARC keeps the caller's ``n_time_samples`` for TIME_SERIES by contract, but the
    default (512 samples over a ~10 s window => fs ~51 Hz) puts a 100 Hz source
    well above Nyquist. The returned p(t) is plausible-looking and at the wrong
    frequency, so silence is the wrong behaviour.
    """
    env = Environment(name='p', bathymetry=200.0, ssp=1500.0,
                      bottom=BoundaryProperties(acoustic_type='rigid'))
    src = Source(depths=50.0, frequencies=100.0)
    rcv = Receiver(depths=100.0, ranges=np.array([2000.0]))

    with recorded_warnings() as w:
        SPARC(verbose=False).run(env, src, rcv, run_mode=RunMode.TIME_SERIES)
    alias = [x for x in w if 'alias' in str(x.message)]
    assert alias, "aliased TIME_SERIES grid did not warn"
    assert 'n_time_samples>=' in str(alias[0].message), "warning must name the fix"


def test_time_series_does_not_warn_when_adequately_sampled():
    """With enough samples for the band, no aliasing warning."""
    env = Environment(name='p', bathymetry=200.0, ssp=1500.0,
                      bottom=BoundaryProperties(acoustic_type='rigid'))
    src = Source(depths=50.0, frequencies=20.0)
    rcv = Receiver(depths=100.0, ranges=np.array([2000.0]))
    with recorded_warnings() as w:
        SPARC(verbose=False, n_time_samples=4096).run(
            env, src, rcv, run_mode=RunMode.TIME_SERIES)
    assert not [x for x in w if 'alias' in str(x.message)]


class TestPulseTypeValidation:
    """``pulse_type`` is validated positionally against the alphabets read
    from ``Scooter/sparc.f90:126-148`` (shape) and
    ``tslib/sourceMod.f90:68-70,178`` (post-process / sign / filter); short
    strings are right-padded to 4 characters like sparcM.m does."""

    def test_short_string_is_ljust_padded_to_four(self):
        assert SPARC(pulse_type='R', verbose=False).pulse_type == 'R   '

    def test_unpinned_pulse_is_the_canned_wavelet_without_a_waveform(self):
        model = SPARC(verbose=False)
        assert model.pulse_type is None
        assert resolve_pulse_type(
            None, pulse_type=model.pulse_type)[0] == 'PN+B'
        assert resolve_pulse_type(
            np.ones(4), pulse_type=model.pulse_type)[0] == 'FN+B'

    def test_a_reassigned_pulse_type_is_checked_by_the_run(self):
        """The knobs are checked again by every run (stage 2), so a value
        assigned after construction cannot reach the deck unchecked."""
        model = SPARC(verbose=False)
        model.pulse_type = 'XN+B'
        with pytest.raises(ConfigurationError, match='position 1'):
            model.run_settings(_rigid_env(), *_point())

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
        _write_deck(SPARC(verbose=False), deck, env, source, receiver)
        lines = deck.read_text().splitlines()
        # Line 2 is the deck frequency; the octave pulse band derives from it.
        assert float(lines[1]) == pytest.approx(30.0)
        assert '15.000000 60.000000' in lines, lines

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_result_records_the_pulse_band_and_the_first_frequency(self):
        """RA-WAVE-12: a time-domain Field's ``frequencies`` are the bins
        that produced it; SPARC marches in time and has none, so it carries
        ``[deck_frequency]`` — the one frequency the deck reads — and the
        band it marched is ``band_hz``, the attribute the band
        products use. The stamped settings agree, with no note of a grid
        replaced."""
        env, source, receiver = self._rig(np.array([30.0, 60.0]))
        result = SPARC(verbose=False).run(env, source, receiver,
                                          run_mode=RunMode.TIME_SERIES)
        assert np.asarray(result.frequencies) == pytest.approx([30.0])
        assert result.f0 == pytest.approx(30.0)
        assert result.band_hz == pytest.approx((15.0, 60.0))
        assert 'center_frequency' not in result.metadata
        assert result.run_settings.engine.deck_frequency == 30.0
        assert list(result.run_settings.frequencies) == [30.0]
        assert not [n for n in result.run_settings.notes
                    if n.startswith('frequencies')]

    def test_the_multi_frequency_notice_comes_from_run_settings_not_validate(
            self):
        env, source, receiver = self._rig(np.array([30.0, 300.0]))
        model = SPARC(verbose=False)
        with recorded_warnings() as caught:
            model.validate_inputs(env, source, receiver)
        assert not [w for w in caught if 'frequencies[0]' in str(w.message)]
        with pytest.warns(UserWarning, match=r'frequencies\[0\] = 30'):
            settings = model.run_settings(env, source, receiver)
        assert settings.engine.deck_frequency == 30.0


class TestPulseBandDefaultsToOneOctave:
    """For a canned pulse, ``freq_min``/``freq_max`` default to one octave around
    the source frequency (``max(f/2, 0.1)`` .. ``2f``,
    ``docs/models/sparc.md`` §5); the deck's band line sits directly after
    the quoted pulse-type line."""

    @staticmethod
    def _deck_lines(tmp_path, **kw):
        env = Environment(
            name='band', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='rigid'))
        deck = tmp_path / 'model.env'
        _write_deck(
            SPARC(verbose=False, **kw), deck, env,
            Source(depths=50.0, frequencies=100.0),
            Receiver(depths=np.array([50.0]), ranges=np.array([1000.0])))
        return deck.read_text().splitlines()

    def test_default_band_is_half_to_double_the_centre(self, tmp_path):
        lines = self._deck_lines(tmp_path)
        i = lines.index("'PN+B'")
        assert lines[i + 1] == '50.000000 200.000000'

    def test_explicit_band_wins(self, tmp_path):
        lines = self._deck_lines(tmp_path, freq_min=40.0, freq_max=90.0)
        i = lines.index("'PN+B'")
        assert lines[i + 1] == '40.000000 90.000000'


class TestThePulseBandComesFromTheBaseBandResolver:
    """RA-WAVE-4 / ARCH-6: a waveform SPARC marches is marched over the band
    every synthesising engine resolves from it for the same pulse and record
    (:func:`~uacpy.models._band._time_series_grid`, the pulse padded to
    SPARC's own ``time_max``), not over one octave around
    ``source.frequencies[0]``; ``run(frequencies=)`` sets the band as it sets
    the grid everywhere else; pinned ``freq_min`` / ``freq_max`` win over both.
    Measured on a 100 m rigid guide, r = 250 m, a 100 Hz Ricker: the legacy
    octave gave -9.61 / -6.26 / -6.03 dB against the exact image sum for
    ``Source(frequencies=)`` 50 / 100 / 200 Hz; the waveform band (10-270 Hz)
    gives -0.06 dB for all three. The result's ``frequencies`` is
    ``[deck_frequency]``; the band is ``engine.freq_min`` / ``freq_max``."""

    @staticmethod
    def _call(source_frequency=100.0):
        fs = 4000.0
        t = np.arange(400) / fs
        a = (np.pi * 100.0 * (t - 0.02)) ** 2
        waveform = (1.0 - 2.0 * a) * np.exp(-a)
        return (_rigid_env(), Source(depths=50.0, frequencies=source_frequency),
                Receiver(depths=[50.0], ranges=[250.0]),
                dict(source_waveform=waveform, sample_rate=fs))

    @staticmethod
    def _band(settings):
        return settings.engine.freq_min, settings.engine.freq_max

    def test_a_marched_waveform_takes_the_band_the_synthesising_engines_use(
            self):
        """Same pulse, same record (Scooter's ``output_duration`` = SPARC's
        ``time_max``): same band."""
        from uacpy.models import Scooter
        env, source, receiver, kw = self._call()
        sparc = SPARC(verbose=False).run_settings(env, source, receiver, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scooter = Scooter(verbose=False).run_settings(
                env, source, receiver, RunMode.TIME_SERIES,
                output_duration=sparc.engine.time_max, **kw)
        assert sparc.engine.pulse_type == 'FN+B'
        assert self._band(sparc) == (scooter.frequencies[0],
                                     scooter.frequencies[-1])
        assert list(sparc.frequencies) == [100.0]
        assert "source_waveform" in sparc.engine.freq_max_origin
        assert f"{sparc.engine.time_max:g} s record" in sparc.engine.freq_max_origin

    def test_the_band_of_a_marched_waveform_ignores_the_source_frequency(
            self):
        bands = []
        for f in (50.0, 100.0, 200.0):
            env, source, receiver, kw = self._call(f)
            bands.append(self._band(SPARC(verbose=False, time_max=0.3)
                                    .run_settings(env, source, receiver,
                                                  **kw)))
        assert len(set(bands)) == 1, bands

    def test_frequencies_set_the_band_and_are_not_ignored(self):
        env, source, receiver, _kw = self._call()
        with recorded_warnings() as caught:
            settings = SPARC(verbose=False).run_settings(
                env, source, receiver, frequencies=[80.0, 100.0, 120.0])
        assert self._band(settings) == (80.0, 120.0)
        assert settings.engine.freq_min_origin == 'the span of run(frequencies=...)'
        assert not [w for w in caught if 'ignoring' in str(w.message)]

    def test_pinned_edges_win_and_the_frequencies_are_then_ignored(self):
        env, source, receiver, kw = self._call()
        with pytest.warns(UserWarning, match=r'ignoring frequencies='):
            settings = SPARC(verbose=False, freq_min=20.0, freq_max=150.0
                             ).run_settings(env, source, receiver,
                                            frequencies=[80.0, 120.0], **kw)
        assert self._band(settings) == (20.0, 150.0)
        assert settings.engine.freq_max_origin == 'SPARC(freq_max=…)'

    def test_one_pinned_edge_keeps_the_other_derived(self):
        env, source, receiver, kw = self._call()
        derived = self._band(SPARC(verbose=False).run_settings(
            env, source, receiver, **kw))
        settings = SPARC(verbose=False, freq_max=300.0).run_settings(
            env, source, receiver, **kw)
        assert self._band(settings) == (derived[0], 300.0)
        assert 'source_waveform' in settings.engine.freq_min_origin

    def test_a_band_of_one_frequency_is_refused(self):
        env, source, receiver, _kw = self._call()
        with pytest.raises(ConfigurationError, match='freq_min < freq_max'):
            SPARC(verbose=False).run_settings(env, source, receiver,
                                              frequencies=[100.0])

    def test_the_padding_of_output_duration_reaches_the_band_not_stsfil(
            self):
        """``output_duration`` is SPARC's record end (``time_max``); the band is
        derived from the pulse padded to it, as on the synthesising engines,
        while STSFIL carries the pulse as given."""
        from uacpy.models import Scooter
        env, source, receiver, kw = self._call()
        settings = SPARC(verbose=False).run_settings(
            env, source, receiver, output_duration=0.5, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scooter = Scooter(verbose=False).run_settings(
                env, source, receiver, RunMode.TIME_SERIES,
                output_duration=0.5, **kw)
        assert self._band(settings) == (scooter.frequencies[0],
                                        scooter.frequencies[-1])
        assert settings.engine.time_max == 0.5
        assert settings.engine.sts_samples == 400
        assert settings.time.source_waveform.size == 400

    @staticmethod
    def _burst(n_cycles, fs=2000.0, f0=80.0):
        n = int(round(n_cycles / f0 * fs))
        return np.sin(2.0 * np.pi * f0 * np.arange(n) / fs), fs

    def test_an_integer_cycle_burst_keeps_its_main_lobe(self):
        """A 4.0-cycle rectangular 80 Hz burst: its unpadded DFT is one line,
        which gave the band [80, 100] Hz (-1.10 dB, corr 0.90 against the
        image sum). Measured on the pulse padded to the record the band
        holds the main lobe (60-100 Hz) and its skirts."""
        wave, fs = self._burst(4.0)
        env, source, receiver, _kw = self._call(80.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            settings = SPARC(verbose=False, time_max=0.6).run_settings(
                env, source, receiver, source_waveform=wave,
                sample_rate=fs)
        freq_min, freq_max = self._band(settings)
        assert freq_min < 20.0 and 200.0 < freq_max < 260.0

    def test_a_hard_edged_burst_is_limited_with_a_notice(self):
        """A 4.2-cycle rectangular burst ends on a jump: its -40 dB support
        reaches 990 Hz, which cost 208.6 s against 0.92 s. The band stops
        one -20 dB band-width beyond the -20 dB band (253 Hz: -0.17 dB, corr
        0.992, 3.96 s against the image sum), the origin says so, and one
        notice states the energy left out and the cost of the full band."""
        wave, fs = self._burst(4.2)
        env, source, receiver, _kw = self._call(80.0)
        model = SPARC(verbose=False, time_max=0.6)
        with recorded_warnings() as caught:
            settings = model.run_settings(env, source, receiver,
                                          source_waveform=wave,
                                          sample_rate=fs)
        freq_min, freq_max = self._band(settings)
        assert 240.0 < freq_max < 270.0
        assert settings.engine.freq_max_origin.startswith('the -20 dB band')
        said = [str(w.message) for w in caught if 'falls slowly' in
                str(w.message)]
        assert len(said) == 1
        assert '990 Hz' in said[0] and '% of the pulse' in said[0]
        assert [x.message for x in settings.engine.notices].count(said[0]) == 1

    def test_a_smooth_pulse_is_not_limited(self):
        """The Ricker's -40 dB support lies inside the limit: no notice, both
        edges from the support."""
        env, source, receiver, kw = self._call()
        with recorded_warnings() as caught:
            settings = SPARC(verbose=False).run_settings(
                env, source, receiver, **kw)
        assert not [w for w in caught if 'falls slowly' in str(w.message)]
        assert settings.engine.freq_max_origin.startswith('the -40 dB support')


class TestOutputTimeGridContract:
    """The output grid is ``n_time_samples`` samples over ``[0, time_max]``; ``march_start``
    only sets where the integration begins (``docs/models/sparc.md`` §7)."""

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_time_axis_spans_zero_to_time_max_with_shifted_march_start(self):
        env = Environment(
            name='window', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='rigid'))
        result = SPARC(time_max=1.0, march_start=-0.4, verbose=False).run(
            env, Source(depths=50.0, frequencies=50.0),
            Receiver(depths=np.array([50.0]), ranges=np.array([1000.0])),
            run_mode=RunMode.TIME_SERIES)
        times = np.asarray(result.coords['time'], dtype=float)
        assert times.size == 512
        assert times[0] == pytest.approx(0.0, abs=1e-6)
        assert times[-1] == pytest.approx(1.0, rel=1e-4)

    @pytest.mark.requires_binary
    def test_time_axis_is_uniform_so_the_transfer_function_bridge_runs(self):
        """The .rts labels carry six significant digits (sparc.f90:294), so
        read verbatim their steps spread by ~0.4 %; the coordinate is the
        deck's own uniform grid."""
        env = _rigid_env()
        result = SPARC(verbose=False).run(
            env, Source(depths=50.0, frequencies=50.0),
            Receiver(depths=np.array([50.0]), ranges=np.array([1000.0])),
            run_mode=RunMode.TIME_SERIES)
        times = np.asarray(result.coords['time'], dtype=float)
        steps = np.diff(times)
        np.testing.assert_allclose(steps, steps[0], rtol=1e-12)
        result.to_transfer_function()

    @pytest.mark.parametrize('t, offset_in_half_digits, matches', [
        (1.66667, 0.9, True), (1.66667, 1.5, False),
        (0.00326, 0.9, True), (0.00326, 1.5, False),
        (0.0, 0.0, True),
    ])
    def test_rts_labels_match_the_deck_to_half_their_last_digit(
            self, t, offset_in_half_digits, matches):
        deck = np.array([0.0, t]) if t else np.array([0.0, 0.0])
        half_digit = (0.5 * 10.0 ** (np.floor(np.log10(t)) - 5)) if t else 0.0
        parsed = deck + np.array([0.0, offset_in_half_digits * half_digit])
        assert rts_time_matches(parsed, deck) is matches


class TestSPARCLevelIsIndependentOfWaterDensity:
    """A unit source at 1 m has one field whatever the density of a uniform
    medium between rigid/vacuum boundaries — Scooter and Kraken move by less
    than 1e-6 dB when ``water_density`` goes from 1.0 to 1.027. SPARC's
    march returns rho(z_s)/2 times that field (sparc.f90:434-456 vs
    :529-535), which the wrapper scales out. The raw march moved by +0.2302 dB, which
    is 20*log10(1.027) = 0.2314 dB less a 0.0012 dB residual the division
    leaves; the bound sits well under the 0.23 dB it removes."""

    @pytest.mark.requires_binary
    def test_peak_level_moves_less_than_0_01_dB_with_water_density(self):
        src = Source(depths=50.0, frequencies=50.0)
        rcv = Receiver(depths=np.array([30.0, 70.0]),
                       ranges=np.array([1000.0]))
        peaks = []
        for rho in (1.0, 1.027):
            env = Environment(name='rho', bathymetry=100.0, ssp=1500.0,
                              water_density=rho,
                              bottom=BoundaryProperties(acoustic_type='rigid'))
            result = SPARC(verbose=False).run(env, src, rcv,
                                              run_mode=RunMode.TIME_SERIES)
            peaks.append(np.nanmax(np.abs(np.asarray(result.data))))
        assert abs(20.0 * np.log10(peaks[1] / peaks[0])) < 0.01


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
    # rmax_factor=10 refines Δk (Nk ≈ 140) so the direct arrival is
    # clean; the r = RMax replica then sits at 4 km / 2.7 s, out of window.
    result = SPARC(pulse_type='RN+N', time_max=0.5, rmax_factor=10.0,
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
        from uacpy.core.boundary import SedimentLayer
        from uacpy.core.bottom import Bottom, SeabedColumn
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
            bottom=BoundaryProperties(acoustic_type='rigid'),
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
        with recorded_warnings() as caught:
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
            bottom=BoundaryProperties(acoustic_type='rigid'),
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
    def test_range_native_scales_the_raw_rts_by_2_over_sqrt_pi_and_density(
            self, tmp_path):
        """Pins the correction against the file sparc.exe actually wrote:
        the ``sqrt(pi)`` the 'R' branch is hot by, and the ``2 / rho(z_s)``
        every SPARC output is scaled by
        (``sparc._extract.scale_to_unit_source_level``)."""
        from uacpy.models._conventions import _source_density
        from uacpy.io.oalib_reader import read_rts_file
        env, source, receiver = self._rig()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = SPARC(output_mode='R', verbose=False, cleanup=False,
                           work_dir=tmp_path).run(env, source, receiver)
        rts = sorted(tmp_path.rglob('*.rts'))
        assert rts, f"no .rts under {tmp_path}"
        raw = np.asarray(read_rts_file(rts[0]).pressure)   # (nt, n_range)
        returned = np.asarray(result.data)[0]              # depth 0 -> (n_range, nt)
        rho_s = _source_density(env, float(source.depths[0]))
        np.testing.assert_allclose(returned,
                                   2.0 * raw.T / np.sqrt(np.pi) / rho_s,
                                   rtol=1e-5, atol=0.0)

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_the_looped_modes_stack_onto_their_own_axis(self, tmp_path):
        """``'R'`` and ``'D'`` are one looped helper over transposed axes
        (``sparc._plan._LOOPED_TIME_SERIES_MODES``): each loops the axis
        ``sparc.f90`` cannot write in one run, then stacks onto that axis of
        the shared (depth, range, time) contract. This pins the half the two
        modes do NOT share — which axis is looped, the run count each stamps,
        and the single output time grid every vertical run is written against.
        """
        from uacpy.io.oalib_reader import read_rts_file
        env, source, _rig_receiver = self._rig()
        # Deliberately NOT square: with equal depth and range counts the
        # stacked shape is the same whichever axis was stacked onto, so a
        # transposed stack would pass unnoticed.
        receiver = Receiver(depths=np.array([20.0, 40.0, 60.0]),
                            ranges=np.array([500.0, 1000.0]))
        n_d = len(receiver.depths)
        n_r = len(receiver.ranges)
        assert n_d != n_r and n_d > 1 and n_r > 1

        for mode, key, n_runs, tag in (('R', 'n_depth_runs', n_d, '_d'),
                                       ('D', 'n_range_runs', n_r, '_r')):
            work = tmp_path / mode
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = SPARC(output_mode=mode, verbose=False, cleanup=False,
                               work_dir=work).run(env, source, receiver)
            assert result.metadata[key] == n_runs
            data = np.asarray(result.data)
            assert data.shape[0] == n_d and data.shape[1] == n_r
            decks = sorted(work.rglob('*.env'))
            assert len(decks) == n_runs
            assert all(tag in d.stem for d in decks), [d.stem for d in decks]
            # Every run is written against the FULL receiver extent, so the
            # per-run traces share one time grid and can be stacked at all.
            times = [np.asarray(read_rts_file(f).times, dtype=float)
                     for f in sorted(work.rglob('*.rts'))]
            assert len(times) == n_runs
            for t in times[1:]:
                np.testing.assert_allclose(t, times[0], rtol=0, atol=0)

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_vertical_array_scales_the_raw_rts_by_2_over_density_only(
            self, tmp_path):
        """The 'D' branch already carries the full Hankel weight, so only
        the ``2 / rho(z_s)`` every SPARC output takes is applied."""
        from uacpy.models._conventions import _source_density
        from uacpy.io.oalib_reader import read_rts_file
        env, source, receiver = self._rig()
        single = Receiver(depths=receiver.depths, ranges=np.array([500.0]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = SPARC(output_mode='D', verbose=False, cleanup=False,
                           work_dir=tmp_path).run(env, source, single)
        rts = sorted(tmp_path.rglob('*.rts'))
        assert rts, f"no .rts under {tmp_path}"
        raw = np.asarray(read_rts_file(rts[0]).pressure)   # (nt, n_depth)
        returned = np.asarray(result.data)[:, 0, :]        # (n_depth, nt)
        rho_s = _source_density(env, float(source.depths[0]))
        np.testing.assert_allclose(returned, 2.0 * raw.T / rho_s, rtol=1e-5,
                                   atol=0.0)


class TestDefaultOutputWindowTracksTravelTime:
    """``time_max`` defaults to ``2.5 × r_max / c``. ``rmax_factor`` is a
    wavenumber-sampling knob (``Δk ≈ 2π/RMax``); folding it into the output
    window stretched ``[0, time_max]`` by the margin while ``n_time_samples`` stayed
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

    def _deck_time_max(self, tmp_path, ssp=1500.0, **kw):
        import re
        env, source, receiver = self._rig(ssp)
        deck = tmp_path / 'model.env'
        with recorded_warnings() as caught:
            _write_deck(SPARC(verbose=False, **kw), deck, env, source,
                        receiver)
        times = [re.match(r'^0\.0 (\S+) /$', ln)
                 for ln in deck.read_text().splitlines()]
        time_max = float([m for m in times if m][-1].group(1))
        return time_max, [w for w in caught if 'alias' in str(w.message)]

    def test_default_window_is_travel_time_based_and_alias_free(self, tmp_path):
        time_max, alias = self._deck_time_max(tmp_path)
        assert time_max == pytest.approx(2.5 * 1000.0 / 1500.0, rel=1e-4)
        assert not alias, [str(w.message) for w in alias]

    def test_margin_moves_rmax_but_not_the_window(self, tmp_path):
        time_max, _ = self._deck_time_max(tmp_path, rmax_factor=10.0)
        assert time_max == pytest.approx(2.5 * 1000.0 / 1500.0, rel=1e-4)

    def test_explicit_time_max_wins(self, tmp_path):
        time_max, _ = self._deck_time_max(tmp_path, time_max=7.0)
        assert time_max == pytest.approx(7.0, rel=1e-9)

    def test_march_start_does_not_shift_the_output_window(self, tmp_path):
        """``march_start`` sets where the march begins, never the
        ``[0, time_max]`` output window the deck's time block declares."""
        default, _ = self._deck_time_max(tmp_path)
        shifted, _ = self._deck_time_max(tmp_path, march_start=-0.5)
        assert shifted == pytest.approx(default, rel=1e-9)

    def test_window_follows_the_environments_own_sound_speed(self, tmp_path):
        """The last arrival travels at the slowest speed in the column, so the
        window scales with that speed and not with a fixed 1500 m/s. Pinning it
        to a constant cuts a slow column's window short by exactly the speed
        ratio, and an arrival past ``time_max`` is simply absent from p(t)."""
        from uacpy.core import SoundSpeedProfile
        slow = SoundSpeedProfile.from_pairs([(0.0, 1450.0), (100.0, 1450.0)])
        fast = SoundSpeedProfile.from_pairs([(0.0, 1540.0), (100.0, 1540.0)])
        t_slow, _ = self._deck_time_max(tmp_path, ssp=slow)
        t_fast, _ = self._deck_time_max(tmp_path, ssp=fast)
        assert t_slow == pytest.approx(2.5 * 1000.0 / 1450.0, rel=1e-4)
        assert t_fast == pytest.approx(2.5 * 1000.0 / 1540.0, rel=1e-4)
        assert t_slow > t_fast

    def test_a_profiled_column_uses_its_slowest_sample(self, tmp_path):
        from uacpy.core import SoundSpeedProfile
        profile = SoundSpeedProfile.from_pairs(
            [(0.0, 1520.0), (50.0, 1480.0), (100.0, 1510.0)])
        time_max, _ = self._deck_time_max(tmp_path, ssp=profile)
        assert time_max == pytest.approx(2.5 * 1000.0 / 1480.0, rel=1e-4)

    def test_pinned_sound_speed_wins(self, tmp_path):
        from uacpy.core import SoundSpeedProfile
        slow = SoundSpeedProfile.from_pairs([(0.0, 1450.0), (100.0, 1450.0)])
        time_max, _ = self._deck_time_max(tmp_path, ssp=slow, window_sound_speed=1500.0)
        assert time_max == pytest.approx(2.5 * 1000.0 / 1500.0, rel=1e-4)


def _rigid_env(depth=100.0, speed=1500.0):
    return Environment(name='rigid', bathymetry=depth, ssp=speed,
                       bottom=BoundaryProperties(acoustic_type='rigid'))


def _write_deck(model, path, env, source, receiver, **run_kwargs):
    """Write to ``path`` the first deck ``model.run`` writes for this call —
    from the settings ``run_settings`` resolves (announcing their notices),
    through the stage-4 writer — and return those settings."""
    from uacpy.models.base import StageInputs
    settings = model.run_settings(env, source, receiver, **run_kwargs)
    work_dir = path.parent / f'{path.stem}_work'
    work_dir.mkdir(exist_ok=True)
    inputs = StageInputs(work_dir=work_dir,
                         env=model._project_environment(env), source=source,
                         receiver=receiver, settings=settings)
    model._write_input(inputs).replace(path)
    return settings


def _point(depth=50.0, freq=30.0, r=1000.0):
    return (Source(depths=depth, frequencies=freq),
            Receiver(depths=np.array([depth]), ranges=np.array([r])))


def _sediment_layer_env(layer_speed, depth=100.0, water_speed=1500.0,
                        thickness=50.0):
    """A water column over one sediment layer over a rigid floor, run
    through the same projection ``SPARC.run`` applies before any deck is
    written, so what comes back is what the writer actually sees.
    """
    bottom = SeabedColumn(
        layers=[SedimentLayer(thickness=thickness, sound_speed=layer_speed,
                              density=2.0, attenuation=0.2)],
        halfspace=BoundaryProperties(acoustic_type='rigid'),
    )
    env = Environment(name='sparc_sediment_layer', bathymetry=depth,
                      ssp=water_speed, bottom=bottom)
    return SPARC(verbose=False)._project_environment(env)


def _rigid_floor_env(depth=100.0, water_speed=1500.0):
    """A water column over a rigid floor, put through the same projection.
    SPARC's deck carries only vacuum / rigid boundaries, so it has no seabed
    medium at all."""
    env = Environment(
        name='sparc_rigid_floor', bathymetry=depth, ssp=water_speed,
        bottom=BoundaryProperties(acoustic_type='rigid'),
    )
    return SPARC(verbose=False)._project_environment(env)


def _alias_warnings(env, tmp_path, name, **sparc_kwargs):
    source, receiver = _point()
    with recorded_warnings() as caught:
        _write_deck(SPARC(verbose=False, **sparc_kwargs),
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
        margin = resolve_rmax_factor(
            rmax_factor=SPARC(verbose=False).rmax_factor)
        assert margin > 3.5, (
            f"default rmax_factor={margin} leaves the range replica "
            f"at (margin-1)*r/c inside the 2.5*r/c auto window")
        # The same statement in arrival times, at the measured geometry:
        # r = 1000 m in an isovelocity 1500 m/s guide.
        r, c = 1000.0, 1500.0
        assert (margin - 1.0) * r / c > 2.5 * r / c

    def test_tight_pinned_margin_warns(self, tmp_path):
        env = _rigid_env()
        source, receiver = _point()
        with pytest.warns(UserWarning, match='range alias'):
            _write_deck(SPARC(verbose=False, rmax_factor=3.0),
                        tmp_path / 'tight.env', env, source, receiver)

    def test_default_margin_is_silent(self, tmp_path):
        env = _rigid_env()
        source, receiver = _point()
        with recorded_warnings() as caught:
            _write_deck(SPARC(verbose=False), tmp_path / 'default.env', env,
                        source, receiver)
        assert not [w for w in caught if 'range alias' in str(w.message)]

    def test_the_warning_names_a_margin_that_clears(self, tmp_path):
        env = _rigid_env()
        source, receiver = _point()
        with pytest.warns(UserWarning, match='rmax_factor') as rec:
            _write_deck(SPARC(verbose=False, rmax_factor=2.0),
                        tmp_path / 'tight2.env', env, source, receiver)
        message = ' '.join(str(w.message) for w in rec)
        needed = float(message.split('rmax_factor>')[1].split(')')[0])
        with recorded_warnings() as caught:
            _write_deck(SPARC(verbose=False,
                              rmax_factor=needed * 1.01),
                        tmp_path / 'wide.env', env, source, receiver)
        assert not [w for w in caught if 'range alias' in str(w.message)]


class TestSparcRangeAliasReadsEveryMedium:
    """The folded replica travels the distance ``RMax - r`` through the whole
    waveguide, so its earliest arrival is set by the fastest medium the deck
    marches, not by the fastest water speed. ``Scooter/sparc.f90:202-216`` takes
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
        c_slow, c_fast = profile_speed_bounds(
            env, window_sound_speed=SPARC(verbose=False).window_sound_speed)
        assert (c_slow, c_fast) == (1500.0, 1900.0)

    def test_the_fastest_of_two_layers_sets_the_fast_bound(self):
        """A slower layer over a faster one: ``cMax`` is the maximum over every
        medium, so the 1900 m/s layer below sets it, not the 1600 m/s one on
        top. A single-layer seabed cannot tell a maximum over the seabed
        speeds from a minimum."""
        bottom = SeabedColumn(
            layers=[SedimentLayer(thickness=25.0, sound_speed=1600.0,
                                  density=1.8, attenuation=0.2),
                    SedimentLayer(thickness=25.0, sound_speed=1900.0,
                                  density=2.0, attenuation=0.2)],
            halfspace=BoundaryProperties(acoustic_type='rigid'))
        env = SPARC(verbose=False)._project_environment(Environment(
            name='sparc_two_layers', bathymetry=100.0, ssp=1500.0,
            bottom=bottom))
        assert sorted(env.bottom.all_sound_speeds()) == [1600.0, 1900.0]
        _c_slow, c_fast = profile_speed_bounds(
            env, window_sound_speed=SPARC(verbose=False).window_sound_speed)
        assert c_fast == 1900.0

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

    def test_a_rigid_floor_contributes_no_speed(self, tmp_path):
        """A rigid floor leaves the deck with the water column as its only
        medium, so no seabed speed moves the fast bound."""
        env = _rigid_floor_env()
        assert env.bottom.all_sound_speeds() == []
        assert profile_speed_bounds(
            env, window_sound_speed=SPARC(verbose=False).window_sound_speed) == (1500.0,
                                                                   1500.0)
        assert not _alias_warnings(env, tmp_path, 'rigid_floor')


class TestSparcWindowTruncation:
    """``time_max`` is 2.5 direct travel times, which does not bound a
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
            warn_on_truncated_window(self._field(0.5))

    @pytest.mark.parametrize('pinned_by, says, never', [
        (None, 'The auto window is 2.5 direct travel times', 'set —'),
        ('SPARC(time_max=…)', 'the one SPARC(time_max=…) set', 'auto window'),
    ])
    def test_the_advice_is_about_the_window_that_was_used(self, pinned_by,
                                                          says, never):
        with recorded_warnings() as caught:
            warn_on_truncated_window(self._field(0.5), pinned_by=pinned_by)
        (message,) = [str(w.message) for w in caught
                      if 'still at' in str(w.message)]
        assert says in message and never not in message

    def test_a_decayed_trace_is_not_reported(self):
        with recorded_warnings() as caught:
            warn_on_truncated_window(self._field(0.01))
        assert not [w for w in caught if 'still at' in str(w.message)]

    def test_an_all_nan_trace_is_not_reported(self):
        field = self._field(0.5)
        field.data = np.full_like(field.data, np.nan)
        with recorded_warnings() as caught:
            warn_on_truncated_window(field)
        assert not [w for w in caught if 'still at' in str(w.message)]

    def test_an_all_nan_trace_raises_no_warning_of_its_own(self):
        """The all-NaN case is decided by masking, so numpy's 'All-NaN slice'
        RuntimeWarning is never raised and never has to be muted."""
        field = self._field(0.5)
        field.data = np.full_like(field.data, np.nan)
        with recorded_warnings() as caught:
            warn_on_truncated_window(field)
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
            warn_on_truncated_window(field)
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
                warn_on_truncated_window(field)
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
        from uacpy.models.sparc._extract import _peak_ignoring_nan
        assert _peak_ignoring_nan(np.array([1.0, np.nan, 3.0])) == 3.0
        assert _peak_ignoring_nan(np.array([1.0, np.nan, np.inf])) == np.inf
        assert np.isnan(_peak_ignoring_nan(np.full(4, np.nan)))


class TestSparcDeckContracts:
    """``freq_min = 0`` is legal (``sparc.f90:114`` clamps ``kMin`` to 1e-20 for
    exactly that case and ``doc/sparc.htm``'s example deck reads "0.0 15.0");
    ``t_start > 0`` is not, since the march would start from rest after the
    pulse has already turned on (``sparc.f90:409``, ``cans.f90``); and
    ``SubTab`` expands ``0.0 time_max /`` inclusive of both ends, so the output
    sample rate is ``(n_time_samples - 1)/time_max``."""

    def test_zero_f_min_reaches_the_deck(self, tmp_path):
        deck = tmp_path / 'fmin0.env'
        source, receiver = _point()
        _write_deck(SPARC(verbose=False, freq_min=0.0, freq_max=60.0), deck,
                    _rigid_env(), source, receiver)
        lines = deck.read_text().splitlines()
        assert lines[lines.index("'PN+B'") + 1] == '0.000000 60.000000'

    def test_negative_f_min_is_refused(self):
        with pytest.raises(ConfigurationError, match='freq_min >= 0'):
            SPARC(freq_min=-1.0)

    def test_positive_march_start_is_refused(self):
        with pytest.raises(ConfigurationError, match='march_start'):
            SPARC(march_start=0.2)

    @pytest.mark.parametrize('march_start', [0.0, -0.1, -1.0])
    def test_non_positive_march_start_is_kept(self, march_start):
        assert SPARC(march_start=march_start).march_start == march_start

    def test_sample_rate_counts_intervals_not_samples(self):
        # 20 samples over 1 s is 19 intervals -> 19 Hz, Nyquist 9.5 Hz, so a
        # 10 Hz band aliases. Counting samples would read 20 Hz and stay
        # silent on exactly this case.
        n_time_samples, notice = resolve_n_time_samples(
            10.0, 1.0, n_time_samples=SPARC(verbose=False, n_time_samples=20).n_time_samples)
        assert n_time_samples == 20 and '19.0 Hz' in notice[1]
        assert resolve_n_time_samples(
            10.0, 1.0,
            n_time_samples=SPARC(verbose=False, n_time_samples=21).n_time_samples) == (21, None)

    def test_multi_frequency_source_names_what_it_reads(self):
        source = Source(depths=50.0, frequencies=np.array([30.0, 300.0]))
        note, message, category = multi_frequency_notice(
            source, octave_band=True)
        assert 'source.frequencies[0] = 30' in message
        assert category is FallbackWarning
        assert 'one octave' in message

    def test_the_notice_names_the_band_source_when_it_is_not_the_octave(
            self):
        source = Source(depths=50.0, frequencies=np.array([30.0, 300.0]))
        _note, message, _category = multi_frequency_notice(
            source, octave_band=False)
        assert 'one octave' not in message
        assert 'attenuation' in message

    def test_single_frequency_source_is_silent(self):
        assert multi_frequency_notice(
            Source(depths=50.0, frequencies=30.0), octave_band=True) is None

    @staticmethod
    def _with_free(monkeypatch, nbytes):
        """Pin what the host reports free. The guard sizes the table against
        MemAvailable, so an unmocked test passes or fails on how much RAM the
        machine happens to have rather than on the contract."""
        from uacpy.models import _budget
        monkeypatch.setattr(_budget, 'available_memory_bytes',
                            lambda: nbytes)

    def test_snapshot_greens_function_cube_is_capped(self, monkeypatch):
        self._with_free(monkeypatch, 4 * 1024 ** 3)
        receiver = Receiver(depths=np.linspace(10, 90, 30),
                            ranges=np.array([50_000.0]))
        with pytest.raises(ConfigurationError, match='GiB'):
            reject_oversized_snapshot(
                receiver, nk=300_000, n_time_samples=512, output_mode='S')
        # A modest cube passes, and the looped modes are never capped here.
        reject_oversized_snapshot(
            receiver, nk=300, n_time_samples=512, output_mode='S')

    def test_the_refusal_names_the_table_and_the_budget(self, monkeypatch):
        """A table over the free memory is refused naming its terms; on a
        host whose free memory cannot be read the same table is announced
        with the same terms, not refused; a looped mode is never weighed."""
        receiver = Receiver(depths=np.linspace(10, 90, 30),
                            ranges=np.array([50_000.0]))
        self._with_free(monkeypatch, 4 * 1024 ** 3)
        with pytest.raises(
                ConfigurationError,
                match=r"SPARC: a snapshot \(output_mode='S'\)") as exc:
            reject_oversized_snapshot(
                receiver, nk=300_000, n_time_samples=512, output_mode='S')
        self._with_free(monkeypatch, None)
        notice = reject_oversized_snapshot(
            receiver, nk=300_000, n_time_samples=512, output_mode='S')
        for msg in (str(exc.value), notice.message):
            assert 'n_time_samples=512' in msg, msg
            assert 'Nk=300000' in msg, msg
        assert 'cannot be read' in notice.message
        assert reject_oversized_snapshot(
            receiver, nk=300_000, n_time_samples=512, output_mode='R') is None


@pytest.mark.requires_binary
@pytest.mark.slow
def test_sparc_reports_a_truncated_shallow_guide_trace():
    """The end-to-end wiring of the truncation check: a 100 m rigid guide at
    30 Hz still carries half its peak amplitude at the end of the default
    window, and the run has to say so."""
    with recorded_warnings() as caught:
        source, receiver = _point()
        SPARC(verbose=False).run(_rigid_env(), source, receiver,
                                 run_mode=RunMode.TIME_SERIES)
    assert [w for w in caught if 'still at' in str(w.message)], (
        "a trace still ringing at time_max was returned without a word")


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
        n_mesh = SPARC(verbose=False).n_mesh
        low = resolve_n_mesh(self._half_space(), 40.0, n_mesh=n_mesh)
        high = resolve_n_mesh(self._half_space(), 120.0, n_mesh=n_mesh)
        assert high[0] > low[0]

    def test_a_thin_layer_gets_its_own_count_not_the_water_columns(self):
        # The regression this guards: broadcasting the water column's count to
        # a 10 m sediment layer over-resolves it by the thickness ratio.
        counts = resolve_n_mesh(self._layered(), 120.0,
                                n_mesh=SPARC(verbose=False).n_mesh)
        assert len(counts) == 2
        assert counts[1] < counts[0]

    def test_a_pinned_count_is_passed_through(self):
        n_mesh = SPARC(n_mesh=333, verbose=False).n_mesh
        assert resolve_n_mesh(self._layered(), 120.0, n_mesh=n_mesh) == 333


class TestSourceTimeSeriesFromFile:
    """A ``pulse_type`` opening with ``'F'`` (or ``'B'``, played backwards)
    makes ``sparc.exe`` read its source series from ``STSFIL`` in the work
    directory (``tslib/sourceMod.f90:44-46, 97``). ``SPARC.run`` stages
    that file from ``source_waveform`` / ``sample_rate`` in the layout
    ``ReadSTS`` reads (``:99-107``: a quoted title, ``Nsd SD(1:Nsd)``, then
    ``t s(1:Nsd)`` rows), zero-padded to a power of two when the pulse is
    band-passed (``tslib/bandpassc.f90:24-25`` stops on any other length),
    and refuses a series the binary cannot use."""

    @staticmethod
    def _env():
        return Environment(
            name='sts', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='rigid'))

    @staticmethod
    def _pulse(fs=1000.0, duration=0.2, f0=30.0):
        t = np.arange(0.0, duration, 1.0 / fs)
        return np.exp(-((t - duration / 2) / (duration / 6)) ** 2) * \
            np.sin(2 * np.pi * f0 * t)

    def test_stsfil_carries_the_documented_layout(self, tmp_path):
        path = tmp_path / 'STSFIL'
        waveform = np.array([0.0, 0.5, -0.25])
        # 180 Hz: the 4 padded rows then span the 1/45 s the 15-60 Hz
        # band-pass needs (the resolution guard has its own tests below).
        assert source_series_rows(
            'FN+B', waveform, 180.0, 1, 15.0, 60.0) == (3, 4)
        write_sparc_source_time_series(
            path, Source(depths=50.0, frequencies=30.0), waveform,
            180.0, 4)
        lines = path.read_text().splitlines()
        assert lines[0] == "'uacpy source time series'"
        assert lines[1] == '1 50.000000'
        rows = [line.split() for line in lines[2:]]
        # 3 samples, padded to 4 (a power of two) for the band-pass.
        assert len(rows) == 4
        times = np.array([float(r[0]) for r in rows])
        values = np.array([float(r[1]) for r in rows])
        assert np.allclose(times, np.arange(4) / 180.0)
        assert np.allclose(values, [0.0, 0.5, -0.25, 0.0])
        assert all(len(r) == 2 for r in rows)

    def test_the_deck_names_the_file_letter_and_the_series_is_unpadded_without_the_filter(
            self, tmp_path):
        model = SPARC(pulse_type='BN+N', freq_min=15.0, freq_max=60.0,
                      verbose=False)
        waveform = np.array([0.0, 1.0, 0.0])
        settings = _write_deck(
            model, tmp_path / 'deck.env', self._env(),
            Source(depths=50.0, frequencies=30.0),
            Receiver(depths=[50.0], ranges=[500.0]),
            source_waveform=waveform, sample_rate=1000.0)
        assert "'BN+N'" in (tmp_path / 'deck.env').read_text()
        stsfil = tmp_path / 'deck_work' / 'STSFIL'
        assert (settings.engine.sts_samples, settings.engine.sts_rows) == (3, 3)
        assert len(stsfil.read_text().splitlines()) == 2 + 3

    @pytest.mark.requires_binary
    def test_a_short_gaussian_pulse_marches_to_a_finite_nonzero_trace(
            self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = SPARC(pulse_type='FN+B', verbose=False,
                           work_dir=tmp_path, cleanup=False).run(
                self._env(), Source(depths=50.0, frequencies=30.0),
                Receiver(depths=[50.0], ranges=[500.0]),
                run_mode=RunMode.TIME_SERIES,
                source_waveform=self._pulse(), sample_rate=1000.0)
        assert (tmp_path / 'STSFIL').exists()
        data = np.asarray(result.data, dtype=float)
        assert data.shape[:2] == (1, 1)
        assert np.isfinite(data).all()
        assert np.abs(data).max() > 0.0

    def test_requesting_f_without_a_waveform_names_source_waveform(self):
        with pytest.raises(ConfigurationError,
                           match=r"source_waveform") as ei:
            SPARC(pulse_type='FN+B', verbose=False).run(
                self._env(), Source(depths=50.0, frequencies=30.0),
                Receiver(depths=[50.0], ranges=[500.0]))
        assert 'sourceMod.f90' in str(ei.value)

    def test_a_waveform_without_a_sample_rate_is_refused_too(self):
        with pytest.raises(ConfigurationError, match=r"sample_rate"):
            SPARC(pulse_type='FN+B', verbose=False).run(
                self._env(), Source(depths=50.0, frequencies=30.0),
                Receiver(depths=[50.0], ranges=[500.0]),
                source_waveform=self._pulse())

    def test_a_canned_pulse_warns_that_the_waveform_pair_is_ignored(
            self, monkeypatch):
        model = SPARC(pulse_type='PN+B', verbose=False)

        def no_work_dir():
            raise RuntimeError('stop before the work directory')
        monkeypatch.setattr(model, '_setup_file_manager', no_work_dir)
        with pytest.warns(UserWarning, match=r"ignoring source_waveform="):
            with pytest.raises(RuntimeError,
                               match='stop before the work directory'):
                model.run(self._env(), Source(depths=50.0, frequencies=30.0),
                          Receiver(depths=[50.0], ranges=[500.0]),
                          source_waveform=self._pulse(), sample_rate=1000.0)

    def test_fewer_than_two_samples_is_refused(self):
        with pytest.raises(ConfigurationError, match=r"TF\(2\) - TF\(1\)"):
            SPARC(pulse_type='FN+B', verbose=False)._require_source_time_series(
                np.array([1.0]), 1000.0)

    def test_two_samples_pass_the_length_guard(self):
        assert SPARC(pulse_type='FN+B',
                     verbose=False)._require_source_time_series(
            np.array([1.0, 0.0]), 1000.0) is None

    def test_a_padded_series_shorter_than_the_band_resolution_is_refused(
            self, tmp_path):
        # Band 15-60 Hz needs 1/45 s; 4 samples at 1 kHz span 4 ms.
        with pytest.raises(ConfigurationError,
                           match=r"bandpassc\.f90:14-16") as ei:
            source_series_rows(
                'FN+B', np.array([0.0, 1.0, 0.0]), 1000.0, 1, 15.0, 60.0)
        assert 'np.pad(source_waveform, (0, 20))' in str(ei.value)

    def test_a_padded_series_exactly_at_the_band_resolution_is_written(
            self, tmp_path):
        # 32 samples at 32*45 Hz span exactly 1/45 s.
        assert source_series_rows(
            'FN+B', np.zeros(32), 32.0 * 45.0, 1, 15.0, 60.0) == (32, 32)

    def test_a_series_over_the_binarys_point_cap_is_refused(
            self, tmp_path, monkeypatch):
        # sourceMod.f90:7 caps rows x source depths at 1e7; the cap is
        # lowered so the boundary is reachable without allocating it.
        monkeypatch.setattr(sparc_pulse, '_MAX_STS_POINTS', 8)
        with pytest.raises(ConfigurationError,
                           match=r"sourceMod\.f90:7,105-120") as ei:
            source_series_rows('FN+N', np.zeros(9), 1000.0, 1,
                               15.0, 60.0)
        assert 'at most 8 samples' in str(ei.value)

    def test_a_series_exactly_at_the_point_cap_is_written(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(sparc_pulse, '_MAX_STS_POINTS', 8)
        assert source_series_rows('FN+N', np.zeros(8), 1000.0, 1,
                                  15.0, 60.0) == (8, 8)
        write_sparc_source_time_series(
            tmp_path / 'STSFIL', Source(depths=50.0, frequencies=30.0),
            np.zeros(8), 1000.0, 8)
        assert len((tmp_path / 'STSFIL').read_text().splitlines()) == 2 + 8

    def test_a_series_with_maxnt_rows_is_refused(self, tmp_path,
                                                 monkeypatch):
        # sourceMod.f90:105-120 reads DO it = 1, MaxNt and exits normally only
        # on end-of-file, so a file of exactly MaxNt rows is the fatal; the
        # bound is lowered so the boundary is reachable without allocating it.
        monkeypatch.setattr(sparc_pulse, '_MAX_STS_ROWS', 8)
        with pytest.raises(ConfigurationError, match='at most 7 samples'):
            source_series_rows('FN+N', np.zeros(8), 1000.0, 1,
                               15.0, 60.0)

    def test_a_series_one_row_under_maxnt_is_written(self, tmp_path,
                                                     monkeypatch):
        monkeypatch.setattr(sparc_pulse, '_MAX_STS_ROWS', 8)
        assert source_series_rows(
            'FN+N', np.zeros(7), 1000.0, 1, 15.0, 60.0) == (7, 7)

    def test_the_unfiltered_letter_skips_the_band_resolution_guard(
            self, tmp_path):
        assert source_series_rows(
            'FN+N', np.array([0.0, 1.0, 0.0]), 1000.0, 1, 15.0,
            60.0) == (3, 3)



class TestSparcMarchesTheWaveformItIsHanded:
    """MODELS_B-20 / Theo's decision: the waveform and record length a
    TIME_SERIES run is handed mean on SPARC what they mean on every other
    engine. An unpinned pulse_type marches the waveform (STSFIL, 'FN+B') and
    an unpinned time_max takes output_duration."""

    def test_a_waveform_selects_the_sts_route(self):
        """The settings carry 'FN+B' and the requested record length; the
        model's own knobs are left as they were."""
        model = SPARC(verbose=False)
        settings = model.run_settings(
            _rigid_env(), Source(depths=10.0, frequencies=200.0),
            Receiver(depths=[20.0], ranges=[500.0]), RunMode.TIME_SERIES,
            source_waveform=np.hanning(64), sample_rate=8000.0,
            output_duration=0.8)
        assert settings.engine.pulse_type == 'FN+B'
        assert settings.engine.time_max == 0.8
        assert settings.engine.time_max_origin == 'run(output_duration=…)'
        assert (model.pulse_type, model.time_max) == (None, None)

    @pytest.mark.requires_binary
    def test_each_source_depth_marches_its_waveform_in_its_own_subdirectory(
            self, tmp_path):
        """A two-depth waveform run in a pinned ``work_dir`` gives each depth
        its own ``source_depth_<z>m`` scratch directory, so each slab's
        ``rts_file`` is the file its own launch wrote."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            stack = SPARC(verbose=False, work_dir=tmp_path,
                          cleanup=False).run(
                _rigid_env(), Source(depths=[30.0, 60.0], frequencies=100.0),
                Receiver(depths=[50.0], ranges=[300.0]),
                source_waveform=np.hanning(40) * np.sin(
                    2 * np.pi * 100.0 * np.arange(40) / 400.0),
                sample_rate=400.0, output_duration=0.1)
        paths = [slab.metadata['rts_file'] for slab in stack.slabs]
        assert 'source_depth_30m' in paths[0], paths
        assert 'source_depth_60m' in paths[1], paths
        for depth in ('30', '60'):
            assert (tmp_path / f'source_depth_{depth}m' / 'STSFIL').exists()

    def test_a_pinned_time_max_wins_and_output_duration_is_ignored(self):
        with pytest.warns(UserWarning, match=r'ignoring output_duration='):
            settings = SPARC(time_max=0.5, verbose=False).run_settings(
                _rigid_env(), Source(depths=10.0, frequencies=200.0),
                Receiver(depths=[20.0], ranges=[500.0]),
                source_waveform=np.hanning(64), sample_rate=8000.0,
                output_duration=0.8)
        assert settings.engine.time_max == 0.5

    def test_a_pinned_pulse_is_kept(self):
        assert SPARC(pulse_type='PN+B', verbose=False).pulse_type == 'PN+B'


class TestSparcMeshFitsItsStaticStorage:
    """``Scooter/sparc.f90:31,204-207``: every medium's ``N + 1`` points share
    ``MaxN = 17000``; past that the binary stops with 'Insufficient storage
    for mesh' (measured: 500 m of water at 2 kHz, automatic mesh 26666)."""

    def test_a_pinned_mesh_filling_maxn_exactly_passes(self):
        env = _rigid_env()
        assert checked_n_mesh(
            env, 100.0,
            pinned_n_mesh=SPARC(n_mesh=16999, verbose=False).n_mesh) == (16999,)

    def test_one_point_more_is_refused(self):
        with pytest.raises(ConfigurationError, match='MaxN'):
            checked_n_mesh(
                _rigid_env(), 100.0,
                pinned_n_mesh=SPARC(n_mesh=17000, verbose=False).n_mesh)

    def test_the_automatic_mesh_is_refused_before_the_binary_runs(
            self, tmp_path, monkeypatch):
        model = SPARC(verbose=False)

        def _no_launch(*args, **kwargs):
            raise AssertionError("a binary was launched past the guard")

        monkeypatch.setattr(model, '_run_subprocess', _no_launch)
        monkeypatch.setattr(model, '_run_and_attach_prt', _no_launch)
        with pytest.raises(ConfigurationError, match='freq_max'):
            model.run(_rigid_env(depth=500.0),
                      Source(depths=250.0, frequencies=2000.0),
                      Receiver(depths=np.array([250.0]),
                               ranges=np.array([1000.0])),
                      run_mode=RunMode.TIME_SERIES)


class TestTheThreeEntryPointsRefuseAlike:
    """``validate_inputs``, ``run_settings`` and ``run`` run the same checking
    stages (the carriers in stage 2, the deck's settings in stage 3), so each
    of these calls is refused by all three with one exception and message,
    and ``run`` launches nothing."""

    @staticmethod
    def _half_space():
        return make_pekeris(name='hs')

    # label: (model kwargs, run kwargs, env, receiver depths, the refusal)
    _CASES = {
        'a half-space seabed': (dict(), dict(), 'half_space', None,
                                (ConfigurationError, 'half-space')),
        'an STSFIL pulse without a waveform':
            (dict(pulse_type='FN+B'), dict(), 'rigid', None,
             (ConfigurationError, 'sourceMod.f90:44-46')),
        'an STSFIL series too short for the band-pass':
            (dict(pulse_type='FN+B', freq_min=15.0, freq_max=60.0),
             dict(source_waveform=np.array([0.0, 1.0, 0.0]),
                  sample_rate=1000.0), 'rigid', None,
             (ConfigurationError, 'bandpassc.f90:14-16')),
        'a looped axis over max_launches':
            (dict(max_launches=2), dict(), 'rigid', [20.0, 50.0, 80.0],
             (UnsupportedFeatureError, 'max_launches=2')),
        'n_time_samples below two': (dict(n_time_samples=1), dict(), 'rigid', None,
                              (ConfigurationError, 'n_time_samples=1')),
        'a mesh over MaxN': (dict(n_mesh=17000), dict(), 'rigid', None,
                             (ConfigurationError, 'MaxN')),
        'a band of one frequency':
            (dict(), dict(frequencies=[100.0]), 'rigid', None,
             (ConfigurationError, 'freq_min < freq_max')),
    }

    @pytest.mark.parametrize('label', sorted(_CASES))
    def test_the_refusal_is_the_same_from_every_entry_point(
            self, label, monkeypatch):
        model_kw, call_kw, env_kind, depths, refusal = self._CASES[label]
        env = self._half_space() if env_kind == 'half_space' else _rigid_env()
        source = Source(depths=50.0, frequencies=30.0)
        receiver = Receiver(depths=depths or [50.0], ranges=[1000.0])
        model = SPARC(verbose=False, **model_kw)

        def _no_launch(*a, **k):
            raise AssertionError('a binary was launched for a refused call')
        monkeypatch.setattr(model, '_run_subprocess', _no_launch)
        outcomes = []
        for entry in ('validate_inputs', 'run_settings', 'run'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with pytest.raises(refusal[0],
                                   match=re.escape(refusal[1])) as exc:
                    getattr(model, entry)(env, source, receiver, **call_kw)
            outcomes.append((type(exc.value), str(exc.value)))
        assert len(set(outcomes)) == 1, outcomes
        assert outcomes[0][0] is refusal[0], outcomes[0]
        assert refusal[1] in outcomes[0][1], outcomes[0]


class TestTheDeckIsWrittenFromTheResolvedSettings:
    """What ``run_settings`` shows is what the binary marched: the result
    carries those settings, and ``sparc.exe``'s own echo in the ``.prt``
    (``sparc.f90:108,118``) reads back the band and the wavenumber count
    they record."""

    @pytest.mark.requires_binary
    def test_the_binarys_echo_matches_the_settings(self, tmp_path):
        import re
        env = _rigid_env()
        source, receiver = _point(r=500.0)
        model = SPARC(verbose=False, work_dir=tmp_path, cleanup=False)
        want = model.run_settings(env, source, receiver)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = model.run(env, source, receiver)
        assert result.run_settings == want
        prt = next(tmp_path.rglob('model.prt')).read_text()
        assert int(re.search(r'Nk =\s+(\d+)', prt).group(1)) == want.engine.n_wavenumbers
        band = re.search(r'fMin, fMax =\s+(\S+)\s+(\S+)', prt)
        assert float(band.group(1)) == pytest.approx(want.engine.freq_min,
                                                     rel=1e-6)
        assert float(band.group(2)) == pytest.approx(want.engine.freq_max,
                                                     rel=1e-6)
        deck = next(tmp_path.rglob('model.env')).read_text().splitlines()
        assert deck[deck.index(f"'{want.engine.pulse_type}'") + 1] == (
            f'{want.engine.freq_min:.6f} {want.engine.freq_max:.6f}')
        assert f'0.0 {want.engine.time_max:.6f} /' in deck

    def test_the_settings_round_trip_and_pickle(self):
        import pickle
        from uacpy.core.run_settings import RunSettings
        from uacpy.models.sparc import SparcSettings
        settings = SPARC(verbose=False).run_settings(
            _rigid_env(), *_point())
        assert isinstance(settings.engine, SparcSettings)
        assert isinstance(settings.engine.n_mesh, tuple)
        assert RunSettings.from_dict(settings.to_dict()) == settings
        assert pickle.loads(pickle.dumps(settings)) == settings
        pinned = SPARC(n_mesh=300, verbose=False).run_settings(
            _rigid_env(), *_point())
        assert pinned.engine.n_mesh == (300,)       # one medium, the pinned count
        assert RunSettings.from_dict(pinned.to_dict()) == pinned


class TestSparcSaysItsMarchIsLossless:
    """``sparc.f90:221`` keeps the real part of each complex sound speed
    only, so the march ignores water absorption and seabed attenuation
    (measured: 0.000 dB of water loss where Scooter applies 0.9-3.4 dB, and
    a 0 or 5 dB/wavelength sediment giving the same trace energy). SPARC
    does not claim the capability, the base's lossless-water advice stays
    silent, and a run that sets any attenuation is told it is ignored."""

    NOTICE = 'the time march is lossless'

    @staticmethod
    def _env(absorption=None, layer_attenuation=None):
        from uacpy.core.boundary import SedimentLayer
        from uacpy.core.bottom import Bottom, SeabedColumn
        if layer_attenuation is None:
            bottom = BoundaryProperties(acoustic_type='rigid')
        else:
            bottom = Bottom(columns=[SeabedColumn(
                layers=[SedimentLayer(thickness=20.0, sound_speed=1600.0,
                                      density=1.8,
                                      attenuation=layer_attenuation)],
                halfspace=BoundaryProperties(acoustic_type='rigid'))])
        return Environment(name='s', bathymetry=100.0, ssp=1500.0,
                           bottom=bottom, absorption=absorption)

    def _messages(self, env):
        with recorded_warnings() as caught:
            settings = SPARC(verbose=False).run_settings(
                env, Source(depths=50.0, frequencies=50.0),
                Receiver(depths=np.array([30.0]),
                         ranges=np.array([1000.0])))
        return [str(w.message) for w in caught], settings

    def test_no_attenuation_anywhere_gives_no_notice(self):
        messages, settings = self._messages(self._env(layer_attenuation=0.0))
        assert not [m for m in messages if self.NOTICE in m]
        assert not [m for m in (x.message for x in settings.engine.notices) if self.NOTICE in m]
        # Nor the lossless-water advice to pass Thorp(): it would change
        # nothing here.
        assert not [m for m in messages if 'absorption=Thorp()' in m]

    @pytest.mark.parametrize('kind', ['absorption', 'layer'])
    def test_one_non_zero_attenuation_gives_the_notice(self, kind):
        from uacpy.core.absorption import Thorp
        env = (self._env(absorption=Thorp(), layer_attenuation=0.0)
               if kind == 'absorption'
               else self._env(layer_attenuation=0.5))
        messages, settings = self._messages(env)
        assert [m for m in messages if self.NOTICE in m], messages
        assert [m for m in (x.message for x in settings.engine.notices) if self.NOTICE in m]
        assert any('Scooter' in m for m in (x.message for x in settings.engine.notices))

    def test_the_capability_is_not_claimed(self):
        assert SPARC.spec.traits.consumes_volume_absorption is False
        assert SPARC(verbose=False).supports_feature(
            'volume_attenuation') is False



class TestSparcRefusesAnInvertedWindowBeforeTheDeck:
    """One pinned bound past the other's derived value is refused while the
    settings resolve, naming the pinned bound, as Scooter does
    (``models/_window.resolve_window``); the binary would otherwise stop at
    ReadEnvironmentMod.f90:135 after the deck was written. Over a rigid
    seabed of 1500 m/s water the derived c_low is 0.95 x 1500 = 1425 m/s."""

    @staticmethod
    def _settings(**kw):
        from uacpy.core import BoundaryProperties, Environment
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=BoundaryProperties(acoustic_type='rigid'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return SPARC(verbose=False, **kw).run_settings(
                env, Source(depths=50.0, frequencies=50.0),
                Receiver(depths=[50.0], ranges=[2000.0]))

    def test_a_pinned_c_high_at_the_derived_c_low_is_refused(self):
        with pytest.raises(ConfigurationError,
                           match=r'pinned c_high=1425\.0 m/s is at or below'):
            self._settings(c_high=1425.0)

    def test_a_pinned_c_high_above_the_derived_c_low_resolves(self):
        engine = self._settings(c_high=1425.5).engine
        assert (engine.c_low, engine.c_high) == (1425.0, 1425.5)
