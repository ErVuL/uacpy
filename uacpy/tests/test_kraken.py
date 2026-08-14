"""Normal-mode tests for ``Kraken``, on both the kraken and krakenc backends."""

import warnings

import pytest
import numpy as np
import uacpy

from uacpy.core.results import Field, Modes
from uacpy.models import Kraken
from uacpy.models.base import RunMode
from uacpy.core import Environment, BoundaryProperties, Source, Receiver

pytestmark = pytest.mark.requires_binary


class TestKrakenBackendSelection:
    """The ``Kraken(backend=...)`` override (kraken / krakenc)."""

    def _fluid(self):
        return Environment(
            name='f', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800, density=1.8,
                                      attenuation=0.3))

    def _elastic(self):
        return Environment(
            name='e', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800, density=1.8,
                                      attenuation=0.3, shear_speed=400))

    def test_auto_dispatch(self):
        assert Kraken(verbose=False)._select_kraken_exe(
            self._fluid()).name == 'kraken.exe'
        assert Kraken(verbose=False)._select_kraken_exe(
            self._elastic()).name == 'krakenc.exe'

    def test_force_each_backend(self):
        assert Kraken(verbose=False, backend='krakenc')._select_kraken_exe(
            self._fluid()).name == 'krakenc.exe'
        assert Kraken(verbose=False, backend='kraken')._select_kraken_exe(
            self._fluid()).name == 'kraken.exe'

    def test_force_kraken_on_elastic_raises(self):
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match="elastic media"):
            Kraken(verbose=False, backend='kraken')._select_kraken_exe(
                self._elastic())

    def test_unknown_backend_raises(self):
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match="not a known backend"):
            Kraken(verbose=False, backend='nope')


class TestKrakenBroadband:
    """End-to-end BROADBAND / TIME_SERIES tests for Kraken."""

    @pytest.mark.slow
    def test_kraken_broadband_returns_transfer_function(self):
        """Kraken BROADBAND returns H(f) on the receiver grid."""
        env = Environment(name="kf_bb", bathymetry=100.0, ssp=1500.0)
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.array([1000.0, 3000.0]),
        )
        frequencies = np.linspace(80.0, 120.0, 5)

        kf = Kraken(verbose=False)
        result = kf.run(
            env, source, receiver,
            run_mode=RunMode.BROADBAND,
            frequencies=frequencies,
        )

        assert isinstance(result, Field)
        assert np.iscomplexobj(result.data)
        assert result.data.shape[0] == len(receiver.depths)
        assert result.data.shape[1] == len(receiver.ranges)
        assert result.data.shape[2] > 0

    @pytest.mark.slow
    def test_kraken_time_series_returns_time_series_field(self):
        """Kraken TIME_SERIES with a tonal waveform returns Field."""
        env = Environment(name="kf_ts", bathymetry=100.0, ssp=1500.0)
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(
            depths=np.array([50.0]),
            ranges=np.array([2000.0]),
        )
        fs = 2000.0
        n = 256
        t = np.arange(n) / fs
        waveform = np.sin(2 * np.pi * 100.0 * t) * np.hanning(n)
        # Δf small enough that 1/Δf ≥ waveform duration (256/2000 = 0.128s)
        # → no DFT-wraparound warning from synthesize_time_series.
        frequencies = np.linspace(60.0, 140.0, 17)

        kf = Kraken(verbose=False)
        result = kf.run(
            env, source, receiver,
            run_mode=RunMode.TIME_SERIES,
            frequencies=frequencies,
            source_waveform=waveform,
            sample_rate=fs,
        )

        assert isinstance(result, Field)
        assert result.data.shape[0] == len(receiver.depths)
        assert result.data.shape[1] == len(receiver.ranges)
        assert result.data.shape[2] > 0
        assert np.all(np.isfinite(result.data))

    @pytest.fixture
    def elastic_env(self):
        """Create environment with elastic bottom."""
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0,
            shear_speed=400.0,
            density=1.8,
            attenuation=0.2,
            shear_attenuation=0.5
        )
        return Environment(
            name="krakenc_test",
            bathymetry=100.0,
            ssp=1500.0,
            bottom=bottom
        )

    @pytest.fixture
    def receiver(self):
        return Receiver(depths=[25.0, 50.0, 75.0], ranges=[1000.0, 3000.0])

    @pytest.mark.requires_binary
    def test_krakenc_complex_modes(self, elastic_env, source, receiver):
        """The krakenc backend returns complex eigenvalues on an elastic bottom."""
        krakenc = Kraken(backend='krakenc', verbose=False)

        modes = krakenc.compute_modes(
            env=elastic_env,
            source=source,
        )

        assert isinstance(modes, Modes)
        assert modes.k is not None
        assert len(modes.k) > 0

        # Complex modes should have complex wavenumbers
        k = modes.k
        assert np.any(np.imag(k) != 0), "Should have complex wavenumbers for elastic bottom"


class TestKrakenAttenuationUnit:
    """TopOpt position 3 is hardwired to ``'W'`` (dB/wavelength) — uacpy's
    documented convention. There is no per-model unit override."""

    def test_writer_emits_W_for_attenuation_unit(self, tmp_path):
        kraken = Kraken()
        env = Environment(name='kr', bathymetry=100.0, ssp=1500.0)
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(depths=[25.0, 50.0, 75.0], ranges=[1000.0])
        env_file = tmp_path / 'kraken.env'
        kraken._write_kraken_env(
            env_file, env, source,
            receiver_obj=receiver,
            receiver_depths=receiver.depths,
        )
        text = env_file.read_text()
        topopt_line = text.splitlines()[3]
        # Position 3 (0-indexed 2 inside the quotes) is the unit char.
        assert "'CV W" in topopt_line or topopt_line[3] == 'W'


class TestKrakenModePointsPerMeter:
    """``mode_points_per_meter`` sets the density of the depth grid the modes
    are tabulated on, independently of the solver's own internal mesh."""

    @pytest.mark.parametrize('cls', [Kraken])
    def test_default_is_derived_not_fixed(self, cls):
        # A density fixed in pts/metre satisfies the manuals' ~10
        # points/wavelength at exactly one frequency, so the default is
        # deferred to run() and resolved from f_max / c_min instead. The
        # constructor keeps the sentinel; see TestModeGridTracksFrequency.
        assert cls().mode_points_per_meter is None

    @pytest.mark.parametrize('cls', [Kraken])
    def test_density_kwarg_accepted(self, cls):
        m = cls(mode_points_per_meter=3.0)
        assert m.mode_points_per_meter == 3.0

    def test_compute_modes_uses_mode_points_per_meter(self, monkeypatch):
        """The dense mode-depth grid scales with mode_points_per_meter."""
        captured = {}

        def spy_run(self_, env, source, dense_receiver, *args, **kwargs):
            captured['n_depths'] = len(dense_receiver.depths)
            captured['z_max'] = float(np.max(dense_receiver.depths))
            raise RuntimeError('stop after _compute_modes_impl')

        monkeypatch.setattr(Kraken, 'run', spy_run)

        env = Environment(name='kr_modes', bathymetry=200.0, ssp=1500.0)
        source = Source(depths=100.0, frequencies=50.0)
        kraken = Kraken(mode_points_per_meter=5.0)
        with pytest.raises(RuntimeError, match='stop'):
            kraken.compute_modes(env, source, n_modes=3)
        # 200 m * 5 pts/m = 1000 pts (>=100 floor).
        assert captured['n_depths'] == 1000
        assert captured['z_max'] == pytest.approx(200.0)


class TestKrakenMergedSurface:
    """The merged Kraken serves MODES (modes binary only) and field modes
    (modes → field.exe) from ONE class — exercise both on one instance."""

    def test_compute_modes_then_compute_tl_same_instance(self):
        env = Environment(
            name='k_merge', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800, density=1.8,
                                      attenuation=0.3))
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=np.array([25.0, 50.0, 75.0]),
                       ranges=np.array([1000.0, 3000.0]))
        kr = Kraken(verbose=False)
        modes = kr.compute_modes(env, src)
        assert isinstance(modes, Modes) and len(modes.k) > 0
        tl = kr.compute_tl(env, src, rcv)
        assert isinstance(tl, Field)
        assert tl.shape == (len(rcv.depths), len(rcv.ranges))
        # modes again after the field run — no shared-state regression
        modes2 = kr.compute_modes(env, src)
        assert len(modes2.k) == len(modes.k)


def test_kraken_zero_modes_warns():
    env = Environment(
        bathymetry=100.0, ssp=1500.0,
        bottom=BoundaryProperties(sound_speed=1600.0, density=1.5, attenuation=0.5))
    rcv = Receiver(depths=np.linspace(5, 95, 10), ranges=np.linspace(100, 8000, 15))
    # 1 Hz in a 100 m guide is far below the modal cutoff → 0 trapped modes
    with pytest.warns(UserWarning, match="no propagating field|0 trapped modes"):
        Kraken().compute_tl(env, Source(depths=50.0, frequencies=1.0), rcv)


class TestKrakenSourceGeometry:
    """Kraken reads geometry and directivity from Source (spec 2026-07-25)."""

    @staticmethod
    def _env():
        return Environment(
            name='geom', bathymetry=200.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800, density=1.8,
                                      attenuation=0.3))

    def test_constructor_rejects_source_type(self):
        with pytest.raises(TypeError):
            Kraken(source_type='R')

    def test_constructor_rejects_beam_pattern_file(self):
        with pytest.raises(TypeError):
            Kraken(source_beam_pattern_file=None)

    def test_field_option_position_one_tracks_source_type(self):
        model = Kraken()
        codes = {
            t: model._build_field_option(
                False, Source(depths=50, frequencies=100, source_type=t),
                RunMode.COHERENT_TL)[0]
            for t in ('point', 'line', 'scaled')
        }
        assert codes == {'point': 'R', 'line': 'X', 'scaled': 'S'}

    def test_field_option_position_three_tracks_beam_pattern(self):
        model = Kraken()
        pat = np.array([[-90.0, -20.0], [90.0, 0.0]])
        omni = model._build_field_option(
            False, Source(depths=50, frequencies=100), RunMode.COHERENT_TL)
        directional = model._build_field_option(
            False, Source(depths=50, frequencies=100, beam_pattern=pat),
            RunMode.COHERENT_TL)
        assert omni[2] == ' '
        assert directional[2] == '*'

    def test_beam_pattern_array_writes_sbp(self, tmp_path):
        env = self._env()
        rcv = Receiver(depths=100.0, ranges=np.linspace(100, 2000, 20))
        pat = np.array([[-90.0, -30.0], [0.0, 0.0], [90.0, -30.0]])
        model = Kraken(work_dir=tmp_path, cleanup=False)
        model.run(env, Source(depths=50, frequencies=200,
                              beam_pattern=pat), rcv)
        sbp = list(tmp_path.rglob('*.sbp'))
        assert sbp, "no .sbp written for Source(beam_pattern=...)"
        assert sbp[0].read_text().split()[0] == '3'


def test_coarse_beam_pattern_does_not_hang_field_exe(tmp_path):
    """A coarse ``.sbp`` must complete (patched AT interp1, MODIFICATIONS.md).

    A 3-point pattern puts x(N-1) at 0 deg, so every mode angle lands in
    interp1's final segment. Stock AT clamps its segment index at N-2 while
    the ``DO WHILE`` keeps testing the same condition, so that segment is
    unreachable and the search spins; the patch makes it terminate.
    """
    env = Environment(
        name='coarse', bathymetry=200.0, ssp=1500.0,
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1800, density=1.8,
                                  attenuation=0.3))
    rcv = Receiver(depths=100.0, ranges=np.linspace(100, 2000, 20))
    pat = np.array([[-90.0, -30.0], [0.0, 0.0], [90.0, -30.0]])
    field = Kraken(work_dir=tmp_path, cleanup=False, timeout=90.0).run(
        env, Source(depths=50, frequencies=200, beam_pattern=pat), rcv)
    assert np.isfinite(np.asarray(field.db)).any()


def test_field_exe_timeout_is_not_swallowed(tmp_path, monkeypatch):
    """A timeout must surface as a timeout, not as a downstream parse error.

    _run_field_exe deliberately tolerates a non-zero teardown status, but a
    timeout means the run never finished, and reading the 0-byte .shd it
    leaves behind would surface as a FileFormatError from detect_endian.
    """
    from uacpy.core.exceptions import ModelExecutionError

    model = Kraken(work_dir=tmp_path, cleanup=False)

    def fake_run(cmd, **kwargs):
        raise ModelExecutionError('Kraken', return_code=-1,
                                  stderr="Timed out after 1.0s", timed_out=True)

    monkeypatch.setattr(model, '_run_subprocess', fake_run)
    fm = model._setup_file_manager()
    (fm.work_dir / 'model.shd').write_bytes(b'')

    with pytest.raises(ModelExecutionError) as exc:
        model._run_field_exe(fm, 'model', 'RC C')
    assert exc.value.timed_out
    assert 'timed out' in str(exc.value).lower()


def test_empty_shd_reports_no_usable_output(tmp_path, monkeypatch):
    """A 0-byte .shd is 'no output', not a file to hand to the reader."""
    from uacpy.core.exceptions import ModelExecutionError

    model = Kraken(work_dir=tmp_path, cleanup=False)
    monkeypatch.setattr(model, '_run_subprocess', lambda cmd, **kw: None)
    fm = model._setup_file_manager()
    (fm.work_dir / 'model.shd').write_bytes(b'')

    with pytest.raises(ModelExecutionError, match="no usable .shd"):
        model._run_field_exe(fm, 'model', 'RC C')


def test_two_receiver_depths_are_not_range_offset():
    """NRz==2 must give the same field as those depths inside a larger grid.

    AT's SubTab does not replicate the Rro sentinel below 3 elements, so the
    two-depth deck has to carry its own: an unreplicated ro=-999.9 m reaches
    ``EvaluateMod``'s ``r( ir ) + ro( : )`` and evaluates the shallowest
    receiver's row at r-999.9 — plausible numbers at the wrong ranges.
    """
    env = Environment(name='pek', bathymetry=200.0, ssp=1500.0,
                      bottom=BoundaryProperties(acoustic_type='half-space',
                                                sound_speed=1800.0, density=1.8,
                                                attenuation=0.5))
    src = Source(depths=50.0, frequencies=100.0)
    ranges = np.array([1000., 2000., 3000.])
    m = Kraken(verbose=False)
    tl2 = np.asarray(m.run(env, src, Receiver(depths=[60., 120.], ranges=ranges)).db)
    tl3 = np.asarray(m.run(env, src, Receiver(depths=[60., 120., 180.], ranges=ranges)).db)
    np.testing.assert_allclose(tl2[0], tl3[0], rtol=0, atol=0.05)
    np.testing.assert_allclose(tl2[1], tl3[1], rtol=0, atol=0.05)


def test_mode_count_probe_matches_pekeris_theory():
    """``_count_modes_at_freq`` must return real counts, not a swallowed error.

    Its broad ``except Exception`` maps any failure to "0 modes", which
    ``_propagating_frequency_floor`` reads as "nothing propagates" — so a
    probe that always errors silently disables the whole broadband
    sub-cutoff recovery instead of failing. Counts are therefore checked
    against the Pekeris estimate M ~ (2 D f / c_w) sqrt(1 - (c_w/c_b)^2)
    rather than against uacpy's own output.
    """
    D, c_w, c_b = 200.0, 1500.0, 1800.0
    env = Environment(name='pek', bathymetry=D, ssp=c_w,
                      bottom=BoundaryProperties(acoustic_type='half-space',
                                                sound_speed=c_b, density=1.8,
                                                attenuation=0.5))
    src = Source(depths=50.0, frequencies=100.0)
    rcv = Receiver(depths=100.0, ranges=np.array([1000.0]))
    m = Kraken(verbose=False)
    exe = m._select_kraken_exe(env)

    freqs = np.array([20.0, 50.0, 100.0, 400.0])
    counts = np.array([m._count_modes_at_freq(env, src, rcv, float(f), exe)
                       for f in freqs])
    predicted = (2.0 * D * freqs / c_w) * np.sqrt(1.0 - (c_w / c_b) ** 2)

    assert np.all(counts > 0), f"no modes found at any frequency: {counts}"
    assert np.all(np.abs(counts - predicted) <= 0.15 * predicted + 1.5), (
        f"counts {counts} depart from Pekeris estimate {np.round(predicted, 1)}")
    assert np.all(np.diff(counts) > 0), "mode count must rise with frequency"
    # Everything above propagates, so the sub-cutoff prefix is empty.
    assert m._propagating_frequency_floor(env, src, rcv, freqs, exe) == 0


class TestFortranFatalErrorExitsZero:
    """AT binaries report fatal errors with ``STOP '<string>'``
    (misc/FatalError.f90:30), which gfortran exits 0 for. Without a post-run
    check the failure is invisible, and with a pinned work_dir the previous
    run's output file is still on disk and gets read as this run's answer."""

    @staticmethod
    def _env(depth, c, attenuation=0.5):
        return Environment(
            name='x', bathymetry=float(depth), ssp=float(c),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=float(attenuation)))

    _SRC = staticmethod(lambda: Source(depths=25.0, frequencies=500.0))
    _RCV = staticmethod(lambda: Receiver(depths=[50.0, 60.0],
                                         ranges=[1000.0, 2000.0]))

    @classmethod
    def _env_that_fatals_in_crci(cls, depth, c):
        """An environment whose bottom attenuation trips ``AttenMod : CRCI``.

        ``BoundaryProperties`` now refuses an attenuation past
        ``MAX_ATTENUATION_DB_PER_WAVELENGTH`` (54.575 dB/wavelength, derived from
        the same ``CRCI`` abort), so the value is set after construction. That is
        the point of these two tests: the carrier guard is the first line of
        defence and the ``.prt``/stderr scan is the second, for a fatal that
        arrives some other way — a hand-edited deck, a future carrier gap, or a
        different AT fatal entirely."""
        env = cls._env(depth, c)
        env.bottom.columns[0].halfspace.attenuation = 100.0
        return env

    @pytest.mark.parametrize('banner', [
        "STOP 'Fatal Error: Check the print file for details'",
        'STOP ERROR IN KRAKENC: Rough elastic interface not allowed',
        'STOP FATAL ERROR in BandPass: N must be a power of 2',
        'STOP *** CONTOURS REQUIRE NRFR>1 YOU STUPID FOOL ***',
        'STOP >>>> ERROR: .dat FILE NOT FOUND <<<<',
        'STOP INVALID INPATCH',
    ])
    def test_every_character_stop_form_is_caught(self, tmp_path, banner):
        """The banners share no marker: AT stops both through ``ERROUT`` and
        directly, and of OASES' 46 stop strings only 19 carry ``***`` while 26
        use ``>>> ... <<<`` and one is bare. Matching on a banner therefore
        catches under half of them, so the detection keys on the character-stop
        *form* instead — any ``STOP`` carrying a message string."""
        from uacpy.core.exceptions import ModelExecutionError
        from types import SimpleNamespace
        model = Kraken(verbose=False)
        result = SimpleNamespace(stdout='', stderr=banner, returncode=0)
        with pytest.raises(ModelExecutionError):
            model._raise_on_fortran_fatal(result, tmp_path, 'nonexistent')

    @pytest.mark.parametrize('stderr', [
        '', 'STOP',
        'Note: The following floating-point exceptions are signalling',
    ])
    def test_a_clean_run_is_not_flagged(self, tmp_path, stderr):
        """A bare ``STOP`` is a normal end and prints no code."""
        from types import SimpleNamespace
        model = Kraken(verbose=False)
        model._raise_on_fortran_fatal(
            SimpleNamespace(stdout='', stderr=stderr, returncode=0),
            tmp_path, 'nonexistent')

    def test_fatal_error_is_raised_not_swallowed(self, tmp_path):
        """A 100 dB/wavelength half-space trips 'The complex sound speed has an
        imaginary part > real part' in AttenMod : CRCI. The binary exits 0, so
        only a .prt/stderr scan catches it."""
        from uacpy.core.exceptions import ModelExecutionError
        with pytest.raises(ModelExecutionError) as ei:
            Kraken(work_dir=str(tmp_path / 'w'), timeout=300).run(
                self._env_that_fatals_in_crci(1000.0, 1480.0),
                self._SRC(), self._RCV())
        assert 'FATAL ERROR' in str(ei.value) or 'Fatal Error' in str(ei.value), (
            f"the Fortran diagnostic never reached the user: {ei.value}")

    def test_stale_output_is_not_returned_as_this_runs_answer(self, tmp_path):
        """The dangerous case: run 1 succeeds, run 2 fatals on a *different*
        environment, and the stale .mod/.shd yield run 1's field."""
        from uacpy.core.exceptions import ModelExecutionError
        wd = str(tmp_path / 'shared')
        first = np.asarray(Kraken(work_dir=wd, timeout=300).run(
            self._env(100.0, 1500.0), self._SRC(), self._RCV()).db)
        assert np.all(np.isfinite(first))

        with pytest.raises(ModelExecutionError):
            Kraken(work_dir=wd, timeout=300).run(
                self._env_that_fatals_in_crci(1000.0, 1480.0),
                self._SRC(), self._RCV())


class TestElasticCLowDefault:
    """With cLow=0 on an elastic environment, krakenc.f90:189 folds the shear
    speeds into cMin and :228-230 drops the search floor to ~0.84x the slowest
    shear speed, so the solver returns interfacial (Scholte/Stoneley) modes
    instead of the waterborne field. KRAKEN's docs prescribe the minimum
    compressional speed."""

    @staticmethod
    def _elastic_env(cs_layer=400.0):
        from uacpy.core.environment import SeabedColumn, SedimentLayer
        return Environment(
            name='el', bathymetry=100.0, ssp=1500.0,
            bottom=SeabedColumn(
                layers=[SedimentLayer(thickness=20.0, sound_speed=1700.0,
                                      density=1.8, attenuation=0.2,
                                      shear_speed=cs_layer,
                                      shear_attenuation=0.5)],
                halfspace=BoundaryProperties(
                    acoustic_type='half-space', sound_speed=2000.0,
                    density=2.0, attenuation=0.5, shear_speed=600.0,
                    shear_attenuation=0.5)))

    def test_default_c_low_is_the_min_compressional_speed(self):
        env = self._elastic_env()
        # water 1500, layer cp 1700, halfspace cp 2000 -> 1500
        assert Kraken()._c_low_for(env) == pytest.approx(1500.0)

    def test_fluid_env_still_delegates_to_kraken(self):
        env = Environment(
            name='fl', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.5))
        assert Kraken()._c_low_for(env) == 0.0

    def test_explicit_c_low_always_wins(self):
        assert Kraken(c_low=1234.0)._c_low_for(self._elastic_env()) == 1234.0

    @pytest.mark.parametrize('cs_layer', [400.0, 600.0])
    def test_elastic_layer_tl_is_physical_by_default(self, cs_layer):
        """Default run must not return the interfacial-mode field (700+ dB)."""
        env = self._elastic_env(cs_layer)
        tl = np.asarray(Kraken(timeout=300).run(
            env, Source(depths=36.0, frequencies=100.0),
            Receiver(depths=[20.0, 50.0], ranges=[1000.0, 3000.0])).db)
        finite = tl[np.isfinite(tl)]
        assert finite.size, "no finite TL returned"
        assert finite.max() < 120.0, (
            f"max TL {finite.max():.1f} dB — the mode search converged on "
            f"interfacial modes instead of the waterborne field")


class TestRangeDependentElasticMesh:
    """A range-dependent seabed collapses to one column (``collapse
    ['bottom_range']='median'``), and a median over columns whose shear speeds
    straddle fluid and elastic — ``[0, 0, 400, 600]`` → 200 m/s — yields an
    elastic sediment with a *short* shear wavelength. AT meshes an elastic
    medium on its shear speed (``misc/ReadEnvironmentMod.f90:99-104``), so a
    mesh sized on the compressional speed is rejected with the fatal
    'Mesh is too coarse'."""

    @staticmethod
    def _env(shear):
        from uacpy.core.ssp import SoundSpeedProfile
        from uacpy.core.bottom import Bottom
        bottom = Bottom.from_halfspaces(
            np.array([0.0, 6000.0, 12000.0, 18000.0]),
            sound_speed=np.array([1600.0, 1650.0, 1750.0, 1800.0]),
            density=np.array([1.5, 1.7, 2.0, 2.2]),
            attenuation=np.array([0.8, 0.5, 0.3, 0.2]),
            shear_speed=np.asarray(shear, dtype=float),
            acoustic_type='half-space')
        return Environment(
            name='rd_mixed',
            ssp=SoundSpeedProfile.from_pairs(np.array(
                [[0, 1520.0], [50, 1505.0], [100, 1495.0],
                 [200, 1490.0], [400, 1485.0]])),
            bathymetry=np.array([[0, 100.0], [8000, 120.0], [10000, 150.0],
                                 [15000, 250.0], [20000, 400.0]]),
            bottom=bottom)

    _SRC = staticmethod(lambda: Source(depths=50.0, frequencies=50.0))
    _RCV = staticmethod(lambda: Receiver(depths=np.linspace(5.0, 380.0, 60),
                                         ranges=np.linspace(100.0, 20000.0, 100)))

    def test_mesh_is_sized_on_the_shear_speed(self):
        """``_fixed_mesh_points`` must mesh the elastic medium on ``cs``, not
        ``cp`` — the number AT compares against is ~8x larger."""
        model = Kraken(verbose=False)
        env = self._env([0.0, 0.0, 400.0, 600.0])
        segments, _, _, max_total_depth = model._segment_env_for_field(
            model._project_environment(env))
        n = model._fixed_mesh_points(segments, max_total_depth, 50.0)
        # AT: Nneeded = span / (c / f / 20) per medium, fataling below
        # Nneeded/2. The seabed media span 400.1 - 100 m at the median shear
        # speed of 200 m/s, so AT wants 1500 points.
        assert n >= 1500, (
            f"mesh of {n} points is below AT's 'Mesh is too coarse' floor for "
            f"a 200 m/s shear medium at 50 Hz")
        # Each medium class is bounded by its own span, so the shared value
        # tracks AT's own number instead of over-meshing every medium.
        assert n < 2 * 1500, f"mesh of {n} points over-shoots AT's 1500"

    def test_top_reflection_file_reaches_the_range_dependent_deck(self,
                                                                  tmp_path):
        """``_write_field_env`` branches to ``write_multi_profile_env`` and
        never calls ``_write_kraken_env``, so the knob has to be expressed on
        the projected environment rather than inside the single-profile
        writer. Expressed there only, the range-dependent deck carries a
        vacuum ``TopOpt`` and no staged ``.trc`` — a silently different
        surface on exactly the runs the knob routes to krakenc."""
        from uacpy.io.oalib_writer import write_multi_profile_env
        from uacpy.models._segmentation import segment_environment_by_range

        trc = tmp_path / 'surf.trc'
        trc.write_text("3\n0.0 1.0 0.0\n45.0 0.5 0.0\n90.0 0.0 0.0\n")
        env = Environment(
            name='rd', bathymetry=np.array([[0.0, 200.0], [5000.0, 260.0]]),
            ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.3))
        model = Kraken(top_reflection_file=trc, verbose=False)
        projected = model._project_environment(env)
        assert projected.surface.acoustic_type == 'file'

        envfile = tmp_path / 'kfield.env'
        write_multi_profile_env(
            envfile, segment_environment_by_range(projected, n_segments=None),
            Source(depths=50.0, frequencies=100.0),
            Receiver(depths=np.arange(10.0, 190.0, 20.0),
                     ranges=np.arange(500.0, 5001.0, 500.0)),
            n_mesh=500)

        opts = [ln.split("'")[1] for ln in envfile.read_text().splitlines()
                if len(ln.split("'")) > 1 and len(ln.split("'")[1]) == 6
                and ln.split("'")[1].strip().isalpha()]
        assert opts and all(o[1] == 'F' for o in opts), opts
        assert (tmp_path / 'kfield.trc').exists()

    def test_the_rejection_floor_is_measured_per_medium(self):
        """``misc/ReadEnvironmentMod.f90:104,110`` tests each medium against
        its **own** ``SSP%Depth(m+1) - SSP%Depth(m)``. Bounding the whole
        sub-bottom as one span at the slowest seabed speed overstates
        ``Nneeded`` and rejects decks the reader accepts."""
        from uacpy.core.bottom import Bottom, SeabedColumn, SedimentLayer
        env = Environment(
            name='stack', bathymetry=np.array([[0.0, 20.0], [5000.0, 26.0]]),
            ssp=1500.0,
            bottom=Bottom(columns=[SeabedColumn(
                layers=[SedimentLayer(thickness=50.0, sound_speed=1500.0,
                                      density=1.5, attenuation=0.1)
                        for _ in range(4)],
                halfspace=BoundaryProperties(
                    acoustic_type='half-space', sound_speed=1800.0,
                    density=2.0, attenuation=0.5))]))
        model = Kraken(n_mesh=50, verbose=False)
        segments, _, _, _max_total = model._segment_env_for_field(
            model._project_environment(env))

        from uacpy.io.oalib_writer import at_mesh_floor
        media = model._multi_profile_media(segments)
        # No single medium is thicker than the 50 m sediment layers, so at
        # 1500 m/s and 100 Hz the coarsest wants INT(50/(1500/100/20)) = 66
        # points and the floor is 66 // 2.
        assert max(t for t, _ in media) == pytest.approx(50.0)
        assert at_mesh_floor(media, 100.0) == 33

    def test_fluid_bottom_mesh_is_unchanged(self):
        """The shear term must not inflate the mesh for a fluid seabed."""
        model = Kraken(verbose=False)
        segments, _, _, max_total_depth = model._segment_env_for_field(
            model._project_environment(self._env([0.0, 0.0, 0.0, 0.0])))
        # A fluid seabed meshes on its compressional speed, so AT's own
        # requirement here is ~270 points; 500 is ``_fixed_mesh_points``'
        # own ``max(500, ...)`` floor, not a number the environment drove.
        assert model._fixed_mesh_points(segments, max_total_depth, 50.0) == 500

    @pytest.mark.slow
    def test_mixed_fluid_elastic_bottom_runs(self):
        """The whole point: a fluid→elastic transition across range must give
        a field, not a raw Fortran fatal."""
        result = Kraken(verbose=False, mode_coupling='adiabatic',
                        n_segments=5, timeout=600).run(
            self._env([0.0, 0.0, 400.0, 600.0]), self._SRC(), self._RCV())
        tl = np.asarray(result.db)
        finite = tl[np.isfinite(tl)]
        assert finite.size, "no finite TL returned"
        assert finite.max() < 200.0, (
            f"max TL {finite.max():.1f} dB — not a physical waterborne field")
        # TL must grow with range, not sit at a constant or run backwards.
        at_source_depth = np.asarray(result.at(depth=50.0).db)
        assert at_source_depth[-1] > at_source_depth[4] + 10.0

    @pytest.mark.slow
    def test_uniformly_elastic_bottom_still_runs(self):
        """A uniformly elastic seabed has no fluid column to drag its median
        shear speed down, so it never needs the enlarged mesh — and must not be
        broken by it either."""
        result = Kraken(verbose=False, mode_coupling='adiabatic',
                        n_segments=5, timeout=600).run(
            self._env([300.0, 400.0, 500.0, 600.0]), self._SRC(), self._RCV())
        finite = np.asarray(result.db)[np.isfinite(result.db)]
        assert finite.size and finite.max() < 200.0


class TestBroadbandSingleFrequency:
    """``run_mode=BROADBAND`` with a one-element ``frequencies=`` grid must
    honour the run-mode contract — complex ``H(f)`` on a frequency axis at the
    *requested* frequency — not silently fall back to ``source.frequencies[0]``
    with no frequency coordinate."""

    @staticmethod
    def _env():
        return Environment(
            name='bb1', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.8,
                                      attenuation=0.5))

    _SRC = staticmethod(lambda: Source(depths=50.0, frequencies=50.0))
    _RCV = staticmethod(lambda: Receiver(depths=np.array([30.0, 70.0]),
                                         ranges=np.linspace(1000.0, 5000.0, 6)))

    def test_one_element_grid_keeps_the_frequency_axis(self):
        result = Kraken(verbose=False).run(
            self._env(), self._SRC(), self._RCV(),
            run_mode=RunMode.BROADBAND, frequencies=np.array([137.0]))
        # H(f) is not its own kind: it is pressure with a frequency axis, and
        # the axis is what this test is about.
        assert (result.kind, result.unit) == ('pressure', 'Pa')
        assert list(result.coords) == ['depth', 'range', 'frequency']
        assert result.data.shape == (2, 6, 1)
        assert np.iscomplexobj(result.data)
        # The requested frequency, NOT the source's 50 Hz.
        assert result.frequencies == pytest.approx([137.0])
        assert result.coords['frequency'] == pytest.approx([137.0])

    @pytest.mark.slow
    def test_one_element_grid_matches_the_native_broadband_bin(self):
        """The narrowband lift and kraken's native multi-frequency ``.mod``
        must agree on the shared bin."""
        one = Kraken(verbose=False).run(
            self._env(), self._SRC(), self._RCV(),
            run_mode=RunMode.BROADBAND, frequencies=np.array([137.0]))
        two = Kraken(verbose=False).run(
            self._env(), self._SRC(), self._RCV(),
            run_mode=RunMode.BROADBAND, frequencies=np.array([100.0, 137.0]))
        assert np.nanmax(np.abs(
            np.asarray(one.db)[:, :, 0] - np.asarray(two.db)[:, :, 1])) < 0.5

    def test_one_element_grid_can_synthesize_a_time_series(self):
        """``synthesize_time_series`` requires a canonical (depth, range,
        frequency) Field tagged ``travelling_wave``. The coords are pinned by
        the test above; this pins the phase reference the one-element grid
        carries, so a 2-D fallback cannot reach the synthesis path."""
        result = Kraken(verbose=False).run(
            self._env(), self._SRC(), self._RCV(),
            run_mode=RunMode.BROADBAND, frequencies=np.array([137.0]))
        assert result.phase_reference == 'travelling_wave'


class TestCoupledModeGridReachesTheDeclaredBottom:
    """``EvaluateCMMod.f90:312-316`` stops a coupled run unless every profile's
    mode-tabulation grid ends *exactly* on the bottom the deck declares::

        IF ( z( 1 ) /= depthT .OR. z( NR ) /= depthB ) THEN
           WRITE( *, * ) 'Fatal Error: modes must be tabulated throughout ...'

    ``z`` is the merged source/receiver depth vector the deck asks for
    (``kraken.f90:573,598`` — *not* the mesh) and ``depthB`` is
    ``SSP%Depth( NMedia + 1 )``. So the grid and the deck's bottom must come
    from one computation: if the writer reserves a padding medium the model's
    copy of the arithmetic does not, the grid lands 0.1 m shallow and every
    ``n_segments >= 3`` coupled run dies on a Fortran stop. ``n_segments=2``
    does not catch it — ``RProf(2)`` lands beyond the outermost receiver, so
    ``EvaluateCM`` never crosses a profile boundary.

    AT's own coupled deck holds the same invariant: ``tests/wedge/wedge.env``
    gives all 51 profiles ``NMedia=2``, a common total depth of 2000 m, and
    ``NRz`` spanning ``0.0 2000.0``.
    """

    @staticmethod
    def _env():
        from uacpy.core.ssp import SoundSpeedProfile
        from uacpy.core.bottom import (
            Bottom, SeabedColumn, SedimentLayer)
        hs = BoundaryProperties(acoustic_type='half-space',
                                sound_speed=2500.0, density=2.5,
                                attenuation=0.05)
        return Environment(
            name='coupled_rd',
            ssp=SoundSpeedProfile.from_pairs(
                np.array([[0, 1510.0], [100, 1500.0], [200, 1500.0]])),
            bathymetry=np.array([[0, 100.0], [10000, 150.0], [20000, 200.0]]),
            bottom=Bottom(columns=[SeabedColumn(
                layers=[SedimentLayer(thickness=3.0, sound_speed=1800.0,
                                      density=2.0, attenuation=0.1)],
                halfspace=hs)]))

    _SRC = staticmethod(lambda: Source(depths=30.0, frequencies=100.0))
    #: Ranges must reach the far profiles: ``EvaluateCM`` places the crossings
    #: at ``RProf(i) = 500*(R(i)+R(i-1))`` (``EvaluateCMMod.f90:45-47``), so a
    #: receiver span short of the first of those never calls ``NewProfile`` at
    #: all, and a broken deck still looks healthy.
    _RCV = staticmethod(lambda: Receiver(
        depths=np.linspace(5.0, 195.0, 12),
        ranges=np.linspace(1000.0, 19000.0, 40)))

    @pytest.mark.parametrize('n_segments', [2, 3, 5])
    def test_mode_grid_ends_on_the_bottom_the_deck_declares(self, n_segments):
        """The unit-level invariant, with no executable in the loop: the depth
        the model tabulates modes to is the depth the writer declares."""
        from uacpy.io.oalib_writer import plan_multi_profile_media
        model = Kraken(verbose=False, n_segments=n_segments,
                       mode_coupling='coupled')
        env = model._project_environment(self._env())
        segments, _, _, max_total_depth = model._segment_env_for_field(env)
        declared = plan_multi_profile_media(segments)[1]
        assert max_total_depth == declared, (
            f"mode grid would stop at {max_total_depth} m while the deck "
            f"declares its bottom at {declared} m — EvaluateCM needs equality")

    @pytest.mark.slow
    @pytest.mark.parametrize('n_segments', [3, 5])
    def test_coupled_runs_past_two_profiles(self, n_segments, tmp_path):
        """The end-to-end payoff: coupled modes across three or more profiles
        return a physical field instead of a Fortran fatal.

        The deck it wrote is then read back and checked profile by profile, so
        the invariant is verified on the artefact KRAKEN actually consumed
        rather than on the planner that produced it."""
        import re
        work = tmp_path / 'w'
        result = Kraken(verbose=False, n_segments=n_segments,
                        mode_coupling='coupled', timeout=600,
                        work_dir=str(work), cleanup=False).run(
            self._env(), self._SRC(), self._RCV(),
            run_mode=RunMode.COHERENT_TL)
        assert result.metadata['mode_coupling'] == 'coupled'
        assert result.metadata['n_profiles'] == n_segments
        tl = np.asarray(result.db)
        finite = tl[np.isfinite(tl)]
        assert finite.size, "no finite TL returned"
        assert 20.0 < finite.min() < 200.0, (
            f"TL range [{finite.min():.1f}, {finite.max():.1f}] dB is not a "
            f"physical waterborne field")

        lines = (work / 'kfield.env').read_text().splitlines()
        titles = [i for i, ln in enumerate(lines) if ln.startswith("'coupled_rd")]
        assert len(titles) == n_segments
        for p_i, start in enumerate(titles):
            stop = titles[p_i + 1] if p_i + 1 < len(titles) else len(lines)
            block = lines[start:stop]
            declared = [float(m.group(1)) for m in
                        (re.match(r'^\d+\s+\S+\s+([\d.]+)$', ln) for ln in block)
                        if m][-1]
            depth_rows = [ln for ln in block
                          if ln.rstrip().endswith('/') and len(ln.split()) > 50]
            grid_end = float(depth_rows[-1].split()[-2])
            assert grid_end == declared, (
                f"profile {p_i}: deck declares its bottom at {declared} m but "
                f"tabulates modes only to {grid_end} m — EvaluateCM stops on "
                f"any difference")


class TestFieldExeFatalIsNotMasked:
    """``EvaluateCM``'s depth-grid stop is a bare ``WRITE`` plus an
    argument-less ``STOP`` (``EvaluateCMMod.f90:313-317``), so it carries none
    of the signals uacpy keys off: exit status is 0, stderr is empty, ERROUT's
    ``*** FATAL ERROR ***`` banner never appears, and the text goes to
    ``field.prt`` rather than ``<base_name>.prt`` (``field.f90:44`` hard-codes
    that name). field.exe has already opened the ``.shd`` and written its
    header, so a plausible-looking stub survives the missing/empty check and
    reaches the reader, which reports a header-count mismatch describing the
    stub instead of the failure."""

    def test_a_fatal_in_field_prt_is_raised_with_its_text(self, tmp_path):
        from uacpy.core.exceptions import ModelExecutionError
        model = Kraken(verbose=False)
        (tmp_path / 'field.prt').write_text(
            " Running FIELD\n"
            " Fatal Error: modes must be tabulated throughout the ocean and "
            "sediment to compute the coupling coefs.\n"
            " depths   0.00000000       203.100006\n"
            " z   0.00000000       203.000000\n")
        with pytest.raises(ModelExecutionError) as ei:
            model._raise_on_field_fatal(tmp_path)
        assert 'modes must be tabulated' in str(ei.value), (
            f"field.exe's own diagnosis never reached the user: {ei.value}")

    def test_a_clean_field_prt_passes(self, tmp_path):
        model = Kraken(verbose=False)
        (tmp_path / 'field.prt').write_text(" Running FIELD\n Coherent\n")
        model._raise_on_field_fatal(tmp_path)      # must not raise
        model._raise_on_field_fatal(tmp_path / 'nonexistent')


class TestIncoherentTL:
    """An incoherent modal sum is a *magnitude* sum: AT parks it in the complex
    ``.shd`` slot, where its phase is an artefact of the storage. The result
    must therefore be real dB TL with no phase reference, never a complex
    travelling-wave pressure."""

    @staticmethod
    def _env():
        return Environment(
            name='inc', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.8,
                                      attenuation=0.5))

    _SRC = staticmethod(lambda: Source(depths=50.0, frequencies=150.0))
    _RCV = staticmethod(lambda: Receiver(depths=np.array([30.0, 70.0]),
                                         ranges=np.linspace(1000.0, 5000.0, 9)))

    def test_incoherent_is_real_tl_without_a_phase_reference(self):
        result = Kraken(verbose=False).run(
            self._env(), self._SRC(), self._RCV(),
            run_mode=RunMode.INCOHERENT_TL)
        assert result.kind == 'pressure' and result.unit == 'dB'
        assert not np.iscomplexobj(result.data)
        assert result.phase_reference is None

    def test_coherent_stays_complex_travelling_wave_pressure(self):
        result = Kraken(verbose=False).run(
            self._env(), self._SRC(), self._RCV(),
            run_mode=RunMode.COHERENT_TL)
        assert result.kind == 'pressure'
        assert np.iscomplexobj(result.data)
        assert result.phase_reference == 'travelling_wave'

    def test_incoherent_smooths_the_interference_pattern(self):
        """Physical check that Opt(4:4) actually reached field.exe: summing
        magnitudes removes the modal interference nulls."""
        common = (self._env(), self._SRC(), self._RCV())
        coh = np.asarray(Kraken(verbose=False).run(
            *common, run_mode=RunMode.COHERENT_TL).db)[0]
        inc = np.asarray(Kraken(verbose=False).run(
            *common, run_mode=RunMode.INCOHERENT_TL).db)[0]
        assert np.ptp(inc) < np.ptp(coh), (
            "incoherent TL is no smoother than coherent — Opt(4:4)='I' "
            "never took effect")

    def test_incoherent_run_mode_is_declared(self):
        assert RunMode.INCOHERENT_TL in Kraken.spec.modes

    def test_coupled_incoherent_is_refused_up_front(self):
        """field.f90 ERROUTs on Opt(2:2)='C' + Opt(4:4)='I', which surfaces as
        an opaque missing-.shd error. Refuse with a typed error instead."""
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.core.ssp import SoundSpeedProfile
        env = Environment(
            name='rd', ssp=SoundSpeedProfile.from_pairs(
                np.array([[0, 1500.0], [100, 1490.0]])),
            bathymetry=np.array([[0, 100.0], [5000, 150.0]]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.8,
                                      attenuation=0.5))
        with pytest.raises(ConfigurationError, match='incoherent'):
            Kraken(verbose=False, mode_coupling='coupled').run(
                env, self._SRC(), self._RCV(),
                run_mode=RunMode.INCOHERENT_TL)


class TestModeCountProbeCleanup:
    """``_count_modes_at_freq`` runs O(log N) throwaway probes per broadband
    sub-cutoff recovery. Each allocates a work dir holding an .env/.mod/.prt;
    without a ``finally`` they accumulate for the life of the process."""

    def test_probe_removes_its_work_dir(self):
        env = Environment(
            name='probe', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.8,
                                      attenuation=0.5))
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))

        from uacpy.io.file_manager import FileManager

        model = Kraken(verbose=False)
        created = []
        create = FileManager.create_work_dir

        def _record(self):
            path = create(self)
            created.append(path)
            return path

        # ``work_dir`` is cleared on cleanup, so the path has to be captured
        # where it is handed out.
        FileManager.create_work_dir = _record
        try:
            # One propagating probe and one below cutoff — both must clean up.
            for freq in (100.0, 1.0):
                model._count_modes_at_freq(env, source, receiver, freq,
                                           model._select_kraken_exe(env))
        finally:
            FileManager.create_work_dir = create

        assert len(created) == 2
        leaked = [str(p) for p in created if p.exists()]
        assert not leaked, f"probe work dirs left on disk: {leaked}"


class TestKrakenClassBody:
    """``spec`` and ``source`` must each be assigned once in ``Kraken``'s class
    body. Python keeps only the last assignment, so a duplicate is dead code
    that silently ignores every edit to the earlier copy — and the class body
    is long enough that a second assignment is easy to miss on review."""

    @staticmethod
    def _class_body_assignments():
        import ast
        import inspect
        import uacpy.models.kraken as mod

        tree = ast.parse(inspect.getsource(mod))
        cls = next(n for n in tree.body
                   if isinstance(n, ast.ClassDef) and n.name == 'Kraken')
        names = []
        for stmt in cls.body:
            targets = (stmt.targets if isinstance(stmt, ast.Assign)
                       else [stmt.target] if isinstance(stmt, ast.AnnAssign)
                       else [])
            names.extend(t.id for t in targets if isinstance(t, ast.Name))
        return names

    @pytest.mark.parametrize('name', ['spec', 'source'])
    def test_defined_exactly_once(self, name):
        names = self._class_body_assignments()
        assert names.count(name) == 1, (
            f"Kraken defines {name!r} {names.count(name)} times in its class "
            f"body; only the last one is live")


class TestResolvedPhaseSpeedBoundsMetadata:
    """Every Kraken result that ran the solver records the resolved ``c_low`` /
    ``c_high`` / ``rmax`` the deck was written with, so a user can see the
    bounds their run actually used (they are usually auto-derived, not
    supplied). Mirrors what ``Bounce`` already reports."""

    KEYS = ('c_low', 'c_high', 'rmax')

    @staticmethod
    def _env():
        return Environment(
            name='bounds', bathymetry=200.0,
            ssp=[(0.0, 1500.0), (200.0, 1520.0)],
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.8,
                                      attenuation=0.3))

    @staticmethod
    def _geometry():
        return (Source(depths=50.0, frequencies=100.0),
                Receiver(depths=np.array([25.0, 100.0, 175.0]),
                         ranges=np.linspace(500.0, 5000.0, 6)))

    def _assert_sane(self, result, *, rmax_floor):
        for key in self.KEYS:
            assert key in result.metadata, (
                f"{result.model} result is missing metadata[{key!r}]")
            assert isinstance(result.metadata[key], float)
        assert result.metadata['c_low'] < result.metadata['c_high']
        # c_high brackets the whole medium (SSP max 1520, half-space 1600).
        assert result.metadata['c_high'] >= 1600.0
        # RMax scales the mesh-convergence tolerance (kraken.f90:80), so it
        # must clear the longest range the modes are propagated to.
        assert result.metadata['rmax'] > rmax_floor

    @pytest.mark.parametrize('run_mode', [
        RunMode.MODES, RunMode.COHERENT_TL, RunMode.INCOHERENT_TL,
    ])
    def test_narrowband_paths_record_bounds(self, run_mode):
        source, receiver = self._geometry()
        result = Kraken(verbose=False).run(
            self._env(), source, receiver, run_mode=run_mode)
        assert isinstance(result, Modes if run_mode is RunMode.MODES else Field)
        self._assert_sane(result, rmax_floor=float(receiver.ranges.max()))

    @pytest.mark.slow
    def test_broadband_path_records_bounds(self):
        source, receiver = self._geometry()
        result = Kraken(verbose=False).run(
            self._env(), source, receiver, run_mode=RunMode.BROADBAND,
            frequencies=np.linspace(80.0, 120.0, 5))
        assert result.metadata['native_broadband'] is True
        self._assert_sane(result, rmax_floor=float(receiver.ranges.max()))

    @pytest.mark.slow
    def test_sub_cutoff_zero_fill_keeps_bounds(self):
        """The broadband recovery path rebuilds the Field around the
        propagating sub-band, so it must not drop the bounds on the way."""
        env = Environment(
            name='bounds_cut', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.3))
        source = Source(depths=50.0, frequencies=20.0)
        receiver = Receiver(depths=np.array([25.0, 75.0]),
                            ranges=np.array([1000.0, 3000.0]))
        with pytest.warns(UserWarning, match="below the modal cutoff"):
            result = Kraken(verbose=False).run(
                env, source, receiver, run_mode=RunMode.BROADBAND,
                frequencies=np.array([2.0, 5.0, 10.0, 20.0, 40.0]))
        assert np.allclose(result.data[:, :, :2], 0.0)
        self._assert_sane(result, rmax_floor=float(receiver.ranges.max()))

    def test_range_dependent_field_path_records_bounds(self):
        from uacpy.core.ssp import SoundSpeedProfile

        env = Environment(
            name='bounds_rd', bathymetry=200.0,
            ssp=SoundSpeedProfile(
                depths=[0.0, 200.0],
                data=[[1500.0, 1500.0], [1520.0, 1560.0]],
                ranges=[0.0, 5000.0]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.8,
                                      attenuation=0.3))
        source, receiver = self._geometry()
        result = Kraken(verbose=False, n_segments=3).run(
            env, source, receiver, run_mode=RunMode.COHERENT_TL)
        assert result.metadata['n_profiles'] == 3
        self._assert_sane(result, rmax_floor=float(receiver.ranges.max()))

    def test_pinned_bounds_are_reported_verbatim(self):
        source, receiver = self._geometry()
        result = Kraken(verbose=False, c_low=1400.0, c_high=1700.0,
                        rmax_m=9000.0).run(
            self._env(), source, receiver, run_mode=RunMode.COHERENT_TL)
        assert result.metadata['c_low'] == 1400.0
        assert result.metadata['c_high'] == 1700.0
        assert result.metadata['rmax'] == 9000.0

    def test_list_metadata_describes_the_bounds(self):
        """The user-facing payoff: the keys are self-describing on the result."""
        source, receiver = self._geometry()
        result = Kraken(verbose=False).run(
            self._env(), source, receiver, run_mode=RunMode.COHERENT_TL)
        described = result.list_metadata()
        for key in self.KEYS:
            assert described[key]['documented_type'] == 'float'
            assert described[key]['value_type'] == 'float'
            assert described[key]['description']


class TestReadModesAsc:
    """``read_modes_asc`` parses the ``.moa`` complex records.

    ``read_modes_asc.m:35,52`` reads them with ``fscanf(fid, '%f', [2, n])``
    and MATLAB fills that matrix column-major, so the token stream is
    ``re im re im …`` — interleaved pairs, not a real block followed by an
    imaginary block. Reading them as blocks silently returns different
    wavenumbers and mode shapes. No AT program writes ``.moa``, so the file
    here is hand-built.
    """

    NTOT, M = 5, 3
    Z = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    K = np.array([1.0 + 0.1j, 2.0 + 0.2j, 3.0 + 0.3j])
    # phi[i, m] = (m+1) + 0.1*(i+1) - 1j*(0.01*(m+1) + 0.001*i): every entry
    # distinct, so a mis-paired parse cannot coincidentally match.
    _MODE = np.arange(1, M + 1)[None, :]
    _DEPTH = np.arange(NTOT)[:, None]
    PHI = (_MODE + 0.1 * (_DEPTH + 1)) - 1j * (0.01 * _MODE + 0.001 * _DEPTH)

    @staticmethod
    def _pairs(values):
        """One record of interleaved ``re im`` tokens."""
        out = []
        for v in np.atleast_1d(values):
            out += [f"{float(v.real):.10g}", f"{float(v.imag):.10g}"]
        return ' '.join(out)

    def _write(self, tmp_path, wrap=False):
        k_record = self._pairs(self.K)
        if wrap:
            # A Fortran runtime may break a record across lines; the record,
            # not the line, is the unit.
            toks = k_record.split()
            k_record = ' '.join(toks[:3]) + '\n' + ' '.join(toks[3:])
        lines = [
            "        10",
            "moa round trip",
            f"100.0 1 {self.NTOT} {self.NTOT} {self.M}",
            "  1 100.0 0.0",
            "A 1500.0 0.0 0.0 0.0 1.0 0.0",
            "A 1800.0 0.0 0.0 0.0 1.8 100.0",
            "",
            ' '.join(f"{v:.10g}" for v in self.Z),
            k_record,
        ] + [self._pairs(self.PHI[:, m]) for m in range(self.M)]
        path = tmp_path / 'hand.moa'
        path.write_text('\n'.join(lines) + '\n')
        return path

    def test_round_trip(self, tmp_path):
        from uacpy.io.modes_reader import read_modes_asc

        modes = read_modes_asc(str(self._write(tmp_path)))
        assert modes['freq'] == 100.0
        assert modes['ntot'] == self.NTOT
        assert np.allclose(modes['z'], self.Z)
        assert np.allclose(modes['k'], self.K)
        assert modes['phi'].shape == (self.NTOT, self.M)
        assert np.allclose(modes['phi'], self.PHI)
        assert modes['Top']['BC'] == 'A'
        assert modes['Bot']['cp'] == 1800.0 + 0j

    def test_records_may_wrap_across_lines(self, tmp_path):
        from uacpy.io.modes_reader import read_modes_asc

        modes = read_modes_asc(str(self._write(tmp_path, wrap=True)))
        assert np.allclose(modes['k'], self.K)
        assert np.allclose(modes['phi'], self.PHI)

    def test_mode_subset(self, tmp_path):
        from uacpy.io.modes_reader import read_modes_asc

        modes = read_modes_asc(str(self._write(tmp_path)), modes=[2])
        assert np.allclose(modes['k'], self.K[[1]])
        assert np.allclose(modes['phi'][:, 0], self.PHI[:, 1])
        # 'M' is the number of modes RETURNED, matching read_modes_bin.
        assert modes['M'] == 1 == len(modes['k'])

    def test_dispatcher_reads_the_ascii_file(self, tmp_path):
        from uacpy.io.modes_reader import read_modes

        modes = read_modes(str(self._write(tmp_path)))
        assert np.allclose(modes['k'], self.K)
        # M != 0 gates the halfspace parameters read_modes computes.
        assert 'gamma' in modes['Top']


def test_read_modes_bin_M_counts_the_modes_returned(tmp_path):
    """``M`` means the same thing in both readers: ``len(k)``. Reporting the
    file's total instead would leave a ``modes=`` subset disagreeing with the
    ``k`` / ``phi`` it is handed back with."""
    from uacpy.io.modes_reader import read_modes_bin

    env = Environment(name='mcount', bathymetry=100.0, ssp=1500.0,
                      bottom=BoundaryProperties(acoustic_type='half-space',
                                                sound_speed=1800.0, density=1.8,
                                                attenuation=0.3))
    kraken = Kraken(verbose=False, work_dir=tmp_path, cleanup=False)
    modes = kraken.compute_modes(env, Source(depths=50.0, frequencies=100.0))
    assert modes.n_modes > 1

    mod_file = str(tmp_path / 'modes.mod')
    full = read_modes_bin(mod_file)
    assert full['M'] == len(full['k']) == modes.n_modes

    subset = read_modes_bin(mod_file, modes=[1])
    assert subset['M'] == 1 == len(subset['k']) == subset['phi'].shape[1]


def test_incoherent_tl_on_krakenc_warns():
    """``field.exe``'s Opt(4:4)='I' branch computes ``SQRT(SUM(z**2))``
    (EvaluateMod.f90:66), which is the energy sum only for real mode
    functions. krakenc's complex phi / k leave cross-mode phase in the
    square, so the result is not a strict incoherent sum."""
    env = Environment(name='inc_kc', bathymetry=100.0, ssp=1500.0,
                      bottom=BoundaryProperties(acoustic_type='half-space',
                                                sound_speed=1800.0, density=1.8,
                                                attenuation=0.3))
    source = Source(depths=50.0, frequencies=100.0)
    receiver = Receiver(depths=np.array([50.0]),
                        ranges=np.array([1000.0, 2000.0]))
    with pytest.warns(UserWarning, match="not a strict incoherent sum"):
        Kraken(verbose=False, backend='krakenc').run(
            env, source, receiver, run_mode=RunMode.INCOHERENT_TL)


def test_incoherent_tl_on_kraken_does_not_warn(recwarn):
    """The real-arithmetic path IS an energy sum, so it must stay quiet."""
    env = Environment(name='inc_k', bathymetry=100.0, ssp=1500.0,
                      bottom=BoundaryProperties(acoustic_type='half-space',
                                                sound_speed=1800.0, density=1.8,
                                                attenuation=0.3))
    Kraken(verbose=False, backend='kraken').run(
        env, Source(depths=50.0, frequencies=100.0),
        Receiver(depths=np.array([50.0]), ranges=np.array([1000.0, 2000.0])),
        run_mode=RunMode.INCOHERENT_TL)
    assert not [w for w in recwarn
                if 'incoherent sum' in str(w.message)]


def test_zero_receiver_range_is_no_data(recwarn):
    """``EvaluateMod.f90:71-73`` skips the 1/sqrt(r) cylindrical-spreading
    division at r=0 rather than dividing by zero, leaving a bare modal sum
    that belongs to no range. Report it as no-data, as Scooter's transform
    does on the same grid."""
    env = Environment(name='r0', bathymetry=100.0, ssp=1500.0)
    receiver = Receiver(depths=np.array([25.0, 75.0]),
                        ranges=np.array([0.0, 1000.0, 3000.0]))
    with pytest.warns(UserWarning, match="r = 0"):
        tl = np.asarray(Kraken(verbose=False).compute_tl(
            env, Source(depths=50.0, frequencies=100.0), receiver).db)
    assert np.all(np.isnan(tl[:, 0]))
    assert np.all(np.isfinite(tl[:, 1:]))


def test_mode_depth_grid_spans_the_thickest_bottom_column(monkeypatch):
    """``compute_modes`` sizes its depth grid from the total media depth, which
    must be summed over the THICKEST bottom column. ``bottom.columns[0]`` is
    merely the first in storage order — neither the thickest nor necessarily
    the r=0 one — so keying on it truncates the grid above the sediment of
    every deeper column. Here column 1 is 80 m against column 0's 20 m, so the
    grid has to reach 100 + 80 = 180 m."""
    from uacpy.core.bottom import Bottom, SeabedColumn, SedimentLayer

    def _column(thickness):
        return SeabedColumn(
            layers=[SedimentLayer(thickness=thickness, sound_speed=1600.0,
                                  density=1.8, attenuation=0.5)],
            halfspace=BoundaryProperties(acoustic_type='half-space',
                                         sound_speed=1800.0, density=2.0,
                                         attenuation=0.8))

    env = Environment(
        name='rd_layers', bathymetry=100.0, ssp=1500.0,
        bottom=Bottom(columns=[_column(20.0), _column(80.0)],
                      ranges=[0.0, 5000.0]))

    captured = {}

    def _capture(self, env, source, receiver, run_mode=None, **kwargs):
        captured['depths'] = np.asarray(receiver.depths, dtype=float)

    monkeypatch.setattr(Kraken, 'run', _capture)
    Kraken(verbose=False)._compute_modes_impl(
        env, Source(depths=50.0, frequencies=100.0), None)

    assert captured['depths'][-1] == pytest.approx(180.0)


# ── deck contract: what the vendored reader actually consumes ────────────

def _pekeris(depth=200.0, c=1500.0, ssp=None):
    return Environment(
        name='deck', bathymetry=depth,
        ssp=ssp if ssp is not None else [(0.0, c), (depth, c)],
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1800.0, density=1.8,
                                  attenuation=0.3))


class TestRMaxPrecision:
    """``RMax`` is the only term that makes KRAKEN's Richardson mesh-refinement
    loop run more than one pass: ``Kraken/kraken.f90:80`` exits as soon as
    ``Error * 1000.0 * RMax < 1.0``, so RMax = 0 returns the coarsest mesh with
    no extrapolation. ``misc/ReadEnvironmentMod.f90:138`` reads the field
    list-directed into a REAL(KIND=8), so there is no width to respect."""

    _SRC = staticmethod(lambda: Source(depths=[20.0], frequencies=[500.0]))
    _RCV = staticmethod(lambda: Receiver(depths=[10.0, 25.0, 40.0],
                                         ranges=[10.0, 25.0, 40.0]))

    @staticmethod
    def _env():
        return Environment(name='rm', bathymetry=50.0,
                           ssp=[(0.0, 1500.0), (50.0, 1480.0)])

    @staticmethod
    def _deck_rmax(work_dir):
        """RMax is the record after cLow/cHigh, which follows the BotOpt line
        and its optional half-space row (misc/ReadEnvironmentMod.f90:121-140)."""
        import re
        lines = (work_dir / 'kfield.env').read_text().splitlines()
        i = next(i for i, ln in enumerate(lines)
                 if re.match(r"^'[VRAFP]'\s", ln.strip()))
        i += 1
        while lines[i].strip().endswith('/'):
            i += 1
        assert len(lines[i].split()) == 2, f"not the cLow/cHigh record: {lines[i]!r}"
        return float(lines[i + 1].split()[0])

    def test_a_short_range_run_still_refines_the_mesh(self, tmp_path):
        auto = Kraken(work_dir=tmp_path / 'auto', cleanup=False)
        tl_auto = np.asarray(auto.compute_tl(
            self._env(), self._SRC(), self._RCV()).db)
        assert self._deck_rmax(tmp_path / 'auto') > 0.0, (
            "RMax reached the deck as 0.0 km — kraken.f90:80 then skips every "
            "mesh doubling")

        pinned = Kraken(rmax_m=1000.0, work_dir=tmp_path / 'pin', cleanup=False)
        tl_pinned = np.asarray(pinned.compute_tl(
            self._env(), self._SRC(), self._RCV()).db)
        assert np.nanmax(np.abs(tl_auto - tl_pinned)) < 0.05, (
            "the auto-RMax deck converges to a different field than a pinned "
            "one — the mesh was not refined")


class TestSSPStartsAtTheSurface:
    """``misc/sspMod.f90:355`` takes the top of medium 1 from the first SSP
    row (``IF ( Medium == 1 ) SSP%Depth( 1 ) = SSP%z( 1 )``);
    ``Kraken/kraken.f90:49-51`` hands that depth to ``ReadSzRz`` as ``zMin``
    and ``misc/SourceReceiverPositions.f90:121-139`` clamps every source and
    receiver above it. A profile that starts below the surface would therefore
    model a thinner waveguide than ``env.depth`` declares."""

    _SRC = staticmethod(lambda: Source(depths=[20.0], frequencies=[100.0]))
    _RCV = staticmethod(lambda: Receiver(depths=[5.0, 20.0, 50.0, 150.0],
                                         ranges=[1000.0, 5000.0]))

    def test_deck_first_ssp_sample_is_z0(self, tmp_path):
        with pytest.warns(UserWarning, match='not at the sea surface'):
            Kraken(work_dir=tmp_path, cleanup=False).compute_tl(
                _pekeris(ssp=[(10.0, 1500.0), (200.0, 1500.0)]),
                self._SRC(), self._RCV())
        rows = [ln for ln in (tmp_path / 'kfield.env').read_text().splitlines()
                if ln.strip().endswith('/') and len(ln.split()) == 7]
        assert float(rows[0].split()[0]) == 0.0, (
            f"first SSP row is at {rows[0].split()[0]} m, not the surface")

    def test_the_field_matches_the_same_profile_written_from_z0(self, tmp_path):
        with pytest.warns(UserWarning, match='not at the sea surface'):
            offset = np.asarray(Kraken(
                work_dir=tmp_path / 'off', cleanup=False).compute_tl(
                    _pekeris(ssp=[(10.0, 1500.0), (200.0, 1500.0)]),
                    self._SRC(), self._RCV()).db)
        surface = np.asarray(Kraken(
            work_dir=tmp_path / 'sfc', cleanup=False).compute_tl(
                _pekeris(ssp=[(0.0, 1500.0), (200.0, 1500.0)]),
                self._SRC(), self._RCV()).db)
        assert np.allclose(offset, surface, atol=1e-6), (
            "an SSP starting below the surface models a different waveguide")


class TestReflectionTableBackendDispatch:
    """``Kraken/kraken.f90:47-48`` stops outright on a bottom ``'F'`` or a top
    ``'P'``, and the two mirror cases pass that guard only to be discarded:
    every mode-finding call passes ``ComplexFlag = .FALSE.`` and
    ``Kraken/BCImpedanceMod.f90:113-116,121-125`` then substitute a rigid
    boundary (``f = 0, g = 1``). ``krakenc.exe`` honours all four
    (``Kraken/BCImpedancecMod.f90:88-106``)."""

    _SRC = staticmethod(lambda: Source(depths=[50.0], frequencies=[100.0]))
    _RCV = staticmethod(lambda: Receiver(depths=[20.0, 60.0, 100.0],
                                         ranges=[1000.0, 2000.0, 3000.0]))

    @staticmethod
    def _table(tmp_path, name):
        path = tmp_path / name
        path.write_text("3\n0.0 0.9 180.0\n45.0 0.9 180.0\n90.0 0.9 180.0\n")
        return path

    def _brc_env(self, tmp_path):
        return Environment(
            name='brc', bathymetry=200.0, ssp=[(0.0, 1500.0), (200.0, 1500.0)],
            bottom=BoundaryProperties(
                acoustic_type='file',
                reflection_file=str(self._table(tmp_path, 'bot.brc'))))

    def _trc_env(self, tmp_path):
        from uacpy.core.surface import Surface
        return Environment(
            name='trc', bathymetry=200.0, ssp=[(0.0, 1500.0), (200.0, 1500.0)],
            surface=Surface(properties=[BoundaryProperties(
                acoustic_type='file',
                reflection_file=str(self._table(tmp_path, 'top.trc')))]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.3))

    def test_a_bottom_brc_runs_on_krakenc(self, tmp_path):
        env = self._brc_env(tmp_path)
        model = Kraken(work_dir=tmp_path / 'w', cleanup=False)
        assert model.select_backend(env) == 'krakenc'
        tl = np.asarray(model.compute_tl(env, self._SRC(), self._RCV()).db)
        assert np.all(np.isfinite(tl)) and tl.max() < 200.0

    def test_a_top_trc_gives_the_same_field_on_auto_and_forced_krakenc(
            self, tmp_path):
        env = self._trc_env(tmp_path)
        auto = Kraken(work_dir=tmp_path / 'a', cleanup=False)
        assert auto.select_backend(env) == 'krakenc'
        forced = Kraken(backend='krakenc', work_dir=tmp_path / 'b',
                        cleanup=False)
        assert np.allclose(
            np.asarray(auto.compute_tl(env, self._SRC(), self._RCV()).db),
            np.asarray(forced.compute_tl(env, self._SRC(), self._RCV()).db))

    def test_an_irc_bottom_dispatches_to_krakenc(self, tmp_path):
        table = tmp_path / 'bot.irc'
        table.write_text("'x' 100.0\n1\n 0.0 1.0 1.0 1.0 1.0 0\n")
        env = Environment(
            name='irc', bathymetry=200.0, ssp=[(0.0, 1500.0), (200.0, 1500.0)],
            bottom=BoundaryProperties(acoustic_type='precalc',
                                      reflection_file=str(table)))
        assert Kraken(verbose=False).select_backend(env) == 'krakenc'

    def test_forcing_kraken_on_a_reflection_table_raises(self, tmp_path):
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match='tabulated reflection'):
            Kraken(backend='kraken').select_backend(self._brc_env(tmp_path))

    def test_a_precalc_surface_is_refused(self, tmp_path):
        """``misc/RefCoef.f90:92`` reads an ``.irc`` for ``BotRC == 'P'`` only,
        so ``TopOpt(2)='P'`` leaves the table unpopulated on every binary."""
        from uacpy.core.exceptions import UnsupportedFeatureError
        from uacpy.core.surface import Surface
        env = Environment(
            name='topP', bathymetry=200.0, ssp=[(0.0, 1500.0), (200.0, 1500.0)],
            surface=Surface(properties=[BoundaryProperties(
                acoustic_type='precalc', reflection_file=str(tmp_path / 'x'))]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.3))
        with pytest.raises(UnsupportedFeatureError, match='precalc'):
            Kraken(verbose=False).select_backend(env)


class TestTopReflectionFileKnob:
    """``Kraken(top_reflection_file=...)`` is shorthand for a surface carrying
    the table; both must stage the same ``<root>.trc`` (``misc/RefCoef.f90:64-76``
    opens exactly that name) and return the same field."""

    def test_the_knob_matches_the_carrier_expression(self, tmp_path):
        from uacpy.core.surface import Surface
        table = tmp_path / 'top.trc'
        table.write_text("3\n0.0 0.9 180.0\n45.0 0.9 180.0\n90.0 0.9 180.0\n")
        src = Source(depths=[50.0], frequencies=[100.0])
        rcv = Receiver(depths=[20.0, 60.0], ranges=[1000.0, 2000.0])

        knob = np.asarray(Kraken(
            top_reflection_file=table, work_dir=tmp_path / 'k',
            cleanup=False).compute_tl(_pekeris(), src, rcv).db)
        env = _pekeris()
        env.surface = Surface(properties=[BoundaryProperties(
            acoustic_type='file', reflection_file=str(table))])
        carrier = np.asarray(Kraken(
            work_dir=tmp_path / 'c', cleanup=False).compute_tl(
                env, src, rcv).db)
        assert np.allclose(knob, carrier)
        assert (tmp_path / 'k' / 'kfield.trc').exists()


class TestBroadbandFrequencyLimit:
    """``KrakenField/field.f90:24`` declares ``MaxNfreq = 1000``, allocates
    ``freqVec( MaxNfreq )`` (:164) and runs ``FreqLoop`` to that bound (:168);
    ``KrakenField/ReadModes.f90:187`` fills the same fixed buffer with the
    ``Nfreq`` from the mode-file header. The solver has no such cap, so a
    longer grid is written and only overruns when field.exe reads it back."""

    def test_more_than_maxnfreq_is_refused_before_the_binary_runs(
            self, tmp_path, monkeypatch):
        from uacpy.core.exceptions import ConfigurationError
        model = Kraken(work_dir=tmp_path, cleanup=False)

        def _no_launch(*args, **kwargs):
            raise AssertionError("a binary was launched past the guard")

        monkeypatch.setattr(model, '_run_subprocess', _no_launch)
        monkeypatch.setattr(model, '_run_and_attach_prt', _no_launch)
        with pytest.raises(ConfigurationError, match='MaxNfreq'):
            model.run(_pekeris(depth=50.0), Source(depths=[25.0],
                                                   frequencies=[150.0]),
                      Receiver(depths=[10.0], ranges=[1000.0]),
                      run_mode=RunMode.BROADBAND,
                      frequencies=np.linspace(100.0, 210.0, 1001))

    def test_exactly_maxnfreq_is_allowed_through(self, tmp_path, monkeypatch):
        """``freqVec( MaxNfreq )`` holds 1000 entries and ``FreqLoop`` runs
        ``DO ifreq = 1, MaxNfreq``, so 1000 is the last legal count — the guard
        must be ``>``, not ``>=``. Reaching the launcher is the pass condition;
        the binary itself is never run."""
        model = Kraken(work_dir=tmp_path, cleanup=False)

        def _no_launch(*args, **kwargs):
            raise RuntimeError("reached the launcher")

        monkeypatch.setattr(model, '_run_subprocess', _no_launch)
        monkeypatch.setattr(model, '_run_and_attach_prt', _no_launch)
        with pytest.raises(RuntimeError, match='reached the launcher'):
            model.run(_pekeris(depth=50.0), Source(depths=[25.0],
                                                   frequencies=[150.0]),
                      Receiver(depths=[10.0], ranges=[1000.0]),
                      run_mode=RunMode.BROADBAND,
                      frequencies=np.linspace(100.0, 210.0, 1000))

    def test_a_time_series_grid_hits_the_same_guard(self, tmp_path,
                                                    monkeypatch):
        """TIME_SERIES derives its own frequency grid from the waveform, so the
        cap has to sit on the funnel every broadband path crosses
        (``_compute_field_via_exe``) rather than on the caller's argument."""
        from uacpy.core.exceptions import ConfigurationError
        model = Kraken(work_dir=tmp_path, cleanup=False)

        def _no_launch(*args, **kwargs):
            raise AssertionError("a binary was launched past the guard")

        monkeypatch.setattr(model, '_run_subprocess', _no_launch)
        monkeypatch.setattr(model, '_run_and_attach_prt', _no_launch)
        # Delta_f = 1/duration, band edges from the -40 dB spectral support:
        # a 100->1100 Hz chirp over 2 s derives 2830 bins.
        sample_rate = 4000.0
        duration = 2.0
        t = np.arange(0, duration, 1.0 / sample_rate)
        waveform = np.sin(2 * np.pi * (100.0 * t
                                       + 0.5 * (1000.0 / duration) * t ** 2))
        with pytest.raises(ConfigurationError, match='MaxNfreq'):
            model.run(_pekeris(depth=50.0), Source(depths=[25.0],
                                                   frequencies=[150.0]),
                      Receiver(depths=[10.0], ranges=[1000.0]),
                      run_mode=RunMode.TIME_SERIES,
                      source_waveform=waveform, sample_rate=sample_rate)


class TestFieldExeErroutIsSurfaced:
    """Every ERROUT reached from field.exe writes the uppercase
    ``*** FATAL ERROR ***`` banner (``misc/FatalError.f90:16-24``) into
    ``field.prt`` (``KrakenField/field.f90:44`` hard-codes that name) and stops
    with exit status 0."""

    def test_an_uppercase_errout_banner_is_detected(self, tmp_path):
        from uacpy.core.exceptions import ModelExecutionError
        (tmp_path / 'field.prt').write_text(
            " Running FIELD\n"
            "\n"
            " *** FATAL ERROR ***\n"
            " Generated by program or subroutine: beampattern : ReadPat\n"
            " Source beam-pattern angles are not monotonic\n")
        with pytest.raises(ModelExecutionError) as ei:
            Kraken(verbose=False)._raise_on_field_fatal(tmp_path)
        assert 'not monotonic' in str(ei.value), (
            f"field.exe's own diagnosis never reached the user: {ei.value}")


class TestNoModesIsATypedError:
    """``Kraken/kraken.f90:947-961`` writes a full header and an ``M = 0``
    record before ``CALL ERROUT( 'KRAKEN', 'No modes for given phase speed
    interval' )``, so the ``.mod`` is a normal-sized file and only the mode
    count reports the state."""

    def test_an_empty_phase_speed_window_raises(self, tmp_path):
        from uacpy.core.exceptions import ModelExecutionError
        env = Environment(name='zm', bathymetry=100.0,
                          ssp=[(0.0, 1500.0), (100.0, 1500.0)],
                          bottom=BoundaryProperties(
                              acoustic_type='half-space', sound_speed=1800.0,
                              density=1.8, attenuation=0.3))
        with pytest.raises(ModelExecutionError, match='no mode'):
            Kraken(c_low=1790.0, c_high=1799.0, work_dir=tmp_path,
                   cleanup=False).compute_modes(
                       env, Source(depths=[50.0], frequencies=[20.0]))
        assert (tmp_path / 'modes.mod').stat().st_size > 0, (
            "the no-modes .mod is non-empty, so a size test cannot detect it")


class TestMeshAndSSPGuardsCoverBothDeckPaths:
    """The range-dependent field run writes its own multi-profile deck, so the
    SSP-type and mesh checks have to sit where both paths pass."""

    _SRC = staticmethod(lambda: Source(depths=[50.0], frequencies=[100.0]))
    _RCV = staticmethod(lambda: Receiver(depths=[50.0], ranges=[1000.0, 5000.0]))

    @staticmethod
    def _rd_env():
        return Environment(
            name='rd', bathymetry=[(0.0, 200.0), (10000.0, 150.0)],
            ssp=[(0.0, 1500.0), (200.0, 1500.0)],
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.3))

    @pytest.mark.parametrize('range_dependent', [False, True])
    def test_quad_is_rejected_on_both_paths(self, range_dependent):
        from uacpy.core.exceptions import ConfigurationError
        env = self._rd_env() if range_dependent else _pekeris()
        with pytest.raises(ConfigurationError, match="'quad'"):
            Kraken(interp_ssp='quad').compute_tl(env, self._SRC(), self._RCV())

    @pytest.mark.parametrize('range_dependent', [False, True])
    def test_a_too_coarse_n_mesh_is_rejected_on_both_paths(self,
                                                           range_dependent):
        from uacpy.core.exceptions import ConfigurationError
        env = self._rd_env() if range_dependent else _pekeris()
        with pytest.raises(ConfigurationError, match='Mesh is too coarse'):
            Kraken(n_mesh=5).compute_tl(env, self._SRC(), self._RCV())

    def test_a_pinned_n_mesh_reaches_the_range_dependent_deck(self, tmp_path):
        model = Kraken(n_mesh=4000, work_dir=tmp_path, cleanup=False)
        model.compute_tl(self._rd_env(), self._SRC(), self._RCV())
        mesh_lines = [ln.split() for ln in
                      (tmp_path / 'kfield.env').read_text().splitlines()
                      if len(ln.split()) == 3 and ln.split()[0].isdigit()]
        assert mesh_lines and all(int(ln[0]) == 4000 for ln in mesh_lines), (
            f"n_mesh was discarded on the multi-profile deck: {mesh_lines}")


class TestSeabedColumnPrecision:
    """``misc/sspMod.f90:334`` and ``misc/ReadEnvironmentMod.f90:88,125,285``
    read the attenuation and roughness columns list-directed into REAL(KIND=8);
    the deck can and must carry the user's value."""

    @staticmethod
    def _layered_env(attenuation, roughness):
        from uacpy.core.bottom import Bottom, SeabedColumn, SedimentLayer
        return Environment(
            name='prec', bathymetry=100.0,
            ssp=[(0.0, 1500.0), (100.0, 1500.0)],
            bottom=Bottom(columns=[SeabedColumn(
                layers=[SedimentLayer(thickness=10.0, sound_speed=1600.0,
                                      density=1.7, attenuation=attenuation)],
                halfspace=BoundaryProperties(
                    acoustic_type='half-space', sound_speed=1800.0,
                    density=2.0, attenuation=attenuation,
                    roughness=roughness))]))

    def test_small_attenuation_and_roughness_survive_the_deck(self, tmp_path):
        Kraken(work_dir=tmp_path, cleanup=False).compute_tl(
            self._layered_env(0.014, 0.03),
            Source(depths=[50.0], frequencies=[1000.0]),
            Receiver(depths=[50.0], ranges=[5000.0]))
        text = (tmp_path / 'kfield.env').read_text()
        assert '0.014000' in text, (
            f"a 0.014 dB/wavelength attenuation was rounded away:\n{text}")
        assert '0.030000' in text, (
            f"a 0.03 m interface roughness was rounded away:\n{text}")


class TestBioLayerLimit:
    """``misc/AttenMod.f90:10,18`` size the shared ``bio( MaxBioLayers )`` array
    at 200. ``misc/ReadEnvironmentMod.f90:222-225`` bounds the count before
    filling it, but ``Bellhop/ReadEnvironmentBell.f90:316-317`` loops straight
    to NBioLayers — the same deck block therefore has to be capped by the
    writer, not by whichever reader happens to consume it."""

    def test_more_than_maxbiolayers_is_refused_by_the_writer(self, tmp_path):
        from uacpy.core.absorption import Biological
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.io.oalib_writer import write_bio_layers

        layers = [(float(i), float(i) + 0.5, 400.0, 5.0, 0.1)
                  for i in range(201)]
        with pytest.raises(ConfigurationError, match='MaxBioLayers'):
            with open(tmp_path / 'x.txt', 'w') as f:
                write_bio_layers(f, layers)

        env = Environment(
            name='bio', bathymetry=300.0, ssp=[(0.0, 1500.0), (300.0, 1500.0)],
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.3),
            absorption=Biological(layers=layers))
        with pytest.raises(ConfigurationError, match='MaxBioLayers'):
            Kraken(work_dir=tmp_path, cleanup=False).compute_tl(
                env, Source(depths=[50.0], frequencies=[100.0]),
                Receiver(depths=[50.0], ranges=[1000.0]))


class TestModesErrorMessageReadsTheRealPrtStrings:
    """The `.prt` diagnosis must match what the vendored binaries actually
    write, not what the manual calls the routine.

    `misc/RootFinderSecantMod.f90:80,136` sets
    ``'Failure to converge in RootFinderSecant'``; `Kraken/kraken.f90:359,407`
    and `Kraken/krakenc.f90:388` echo it behind their own
    ``'Warning in KRAKEN[C] - RootFinderSecant'`` banner. The phrase
    ``FAILURE TO CONVERGE IN SECANT`` appears only in
    `Acoustics-Toolbox/doc/kraken.htm:1802`, which names a superseded routine —
    matching on it selects nothing a binary can emit.
    """

    def _message(self, tmp_path, prt_text):
        (tmp_path / 'run.prt').write_text(prt_text)
        return Kraken._modes_error_message(str(tmp_path / 'run'))

    def test_secant_failure_is_recognised(self, tmp_path):
        msg = self._message(
            tmp_path,
            ' Warning in KRAKEN - RootFinderSecant'
            ' Failure to converge in RootFinderSecant\n')
        assert 'RootFinderSecant' in msg
        assert 'c_low' in msg

    def test_krakenc_banner_is_recognised_too(self, tmp_path):
        msg = self._message(
            tmp_path,
            ' Warning in KRAKENC - RootFinderSecant : '
            'Failure to converge in RootFinderSecant\n')
        assert 'RootFinderSecant' in msg

    def test_empty_spectrum_is_reported_as_a_phase_speed_window_problem(
            self, tmp_path):
        msg = self._message(
            tmp_path, ' No modes for given phase speed interval\n')
        assert 'c_low' in msg and 'c_high' in msg


class TestBeamPatternOnMultipleFrequencies:
    """``KrakenField/field.f90:191`` allocates ``kz2``/``thetaT``/``S`` inside
    ``FreqLoop`` guarded only by ``SBPFlag == '*' .AND. iS == 1``, while the
    matching ``DEALLOCATE`` sits after the loop closes (:226). The second
    frequency therefore re-allocates an already-allocated array, which gfortran
    terminates on. The allocation block is outside uacpy's ``rProf`` patch, so it
    is upstream behaviour."""

    PATTERN = np.array([[-90.0, 0.0], [90.0, 0.0]])

    def test_multi_frequency_with_a_beam_pattern_is_refused_before_the_binary_runs(
            self, tmp_path, monkeypatch):
        from uacpy.core.exceptions import UnsupportedFeatureError
        model = Kraken(work_dir=tmp_path, cleanup=False)

        def _no_launch(*args, **kwargs):
            raise AssertionError("a binary was launched past the guard")

        monkeypatch.setattr(model, '_run_subprocess', _no_launch)
        monkeypatch.setattr(model, '_run_and_attach_prt', _no_launch)
        with pytest.raises(UnsupportedFeatureError, match=r'field\.f90:191'):
            model.run(_pekeris(depth=100.0),
                      Source(depths=[25.0], frequencies=[180.0, 200.0, 220.0],
                             beam_pattern=self.PATTERN),
                      Receiver(depths=[50.0], ranges=[1000.0, 2000.0]),
                      run_mode=RunMode.BROADBAND)

    def test_the_gate_is_the_frequency_count_not_the_run_mode(
            self, tmp_path, monkeypatch):
        """The multi-frequency path reaches the funnel with the default
        ``COHERENT_TL``, so a guard keyed on ``run_mode`` would not fire."""
        from uacpy.core.exceptions import UnsupportedFeatureError
        model = Kraken(work_dir=tmp_path, cleanup=False)
        monkeypatch.setattr(
            model, '_run_subprocess',
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("a binary was launched past the guard")))
        with pytest.raises(UnsupportedFeatureError):
            model._compute_field_via_exe(
                _pekeris(depth=100.0),
                Source(depths=[25.0], frequencies=[200.0],
                       beam_pattern=self.PATTERN),
                Receiver(depths=[50.0], ranges=[1000.0]),
                frequencies=np.array([180.0, 200.0, 220.0]),
            )

    @pytest.mark.requires_binary
    def test_a_single_frequency_still_accepts_the_pattern(self, tmp_path):
        model = Kraken(work_dir=tmp_path, cleanup=False)
        tl = model.compute_tl(
            _pekeris(depth=100.0),
            Source(depths=[25.0], frequencies=200.0, beam_pattern=self.PATTERN),
            Receiver(depths=[50.0], ranges=[1000.0, 2000.0]))
        assert np.all(np.isfinite(np.asarray(tl.db)))

    def test_field_completion_marker_separates_teardown_from_a_real_abort(
            self, tmp_path):
        """``field.f90:228`` writes the marker after ``FreqLoop`` and before the
        clean-up block, so its absence means the run died while computing."""
        model = Kraken(work_dir=tmp_path, cleanup=False)
        prt = tmp_path / 'field.prt'
        prt.write_text(' some output\n Field completed successfully\n')
        assert model._field_reached_completion(tmp_path)
        prt.write_text(' some output\n At line 191 of file field.f90\n')
        assert not model._field_reached_completion(tmp_path)


class TestAutoSegmentationIsWritableAtDeckResolution:
    """``models/_segmentation.py`` unions the bathymetry / SSP / RD-bottom change
    points itself, so it must not produce two ranges the ``.flp`` cannot tell
    apart. A bathymetry axis and an SSP axis naming the same physical range
    through different arithmetic differ in the last bits; both survived a
    ``set()``, printed as one token, and ``KrakenField/EvaluateADMod.f90:75``
    divided by the zero gap with no diagnostic — a partly-NaN field, no error,
    no warning."""

    @staticmethod
    def _env(ssp_break_m):
        from uacpy.core import SoundSpeedProfile, Bathymetry
        z = np.array([0.0, 100.0, 200.0])
        ssp = SoundSpeedProfile(
            depths=z,
            data=np.column_stack([[1500.0, 1495.0, 1490.0],
                                  [1500.0, 1497.0, 1492.0],
                                  [1500.0, 1499.0, 1494.0]]),
            ranges=np.array([0.0, ssp_break_m, 10000.0]))
        return Environment(
            bathymetry=Bathymetry(ranges=np.linspace(0.0, 10000.0, 6),
                                  depths=np.full(6, 200.0)),
            ssp=ssp,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.8,
                                      attenuation=0.5))

    def test_segment_axis_never_collapses_at_the_deck_quantum(self):
        from uacpy.models._segmentation import segment_environment_by_range
        from uacpy.io.oalib_writer import DECK_RANGE_QUANTUM_M
        segments = segment_environment_by_range(self._env(4000.0000001))
        ranges = np.array([r for r, _e in segments], dtype=float)
        assert np.all(np.diff(ranges) > DECK_RANGE_QUANTUM_M)

    @pytest.mark.requires_binary
    def test_a_1e_7_metre_shift_in_a_break_range_does_not_change_the_field(self):
        """The two breaks are the same physical range, so the TL must be
        identical — not merely finite."""
        src = Source(depths=50.0, frequencies=200.0)
        rcv = Receiver(depths=np.linspace(20.0, 180.0, 5),
                       ranges=np.linspace(1000.0, 9000.0, 5))
        on_node = np.asarray(Kraken().compute_tl(self._env(4000.0), src, rcv).db)
        off_node = np.asarray(
            Kraken().compute_tl(self._env(4000.0000001), src, rcv).db)
        assert np.all(np.isfinite(on_node)) and np.all(np.isfinite(off_node))
        np.testing.assert_allclose(off_node, on_node, rtol=0, atol=1e-9)


class TestModeGridTracksFrequency:
    """The mode-tabulation grid carries the mode shapes and the coupling
    integrals, so ``kraken.htm`` block (9) and ``field.htm`` §(2) both require
    ~10 points/wavelength on it. A density fixed in pts/**metre** meets that at
    one frequency only: at 1.5 pts/m the grid holds 2250/f points per
    wavelength, i.e. 1.4 at 1600 Hz, and the resulting TL error is silent —
    ``KrakenField/ReadModes.f90:78`` sets its tolerance at a whole wavelength,
    so AT's own "Modes not tabulated near requested pt." never fires."""

    @staticmethod
    def _env():
        return uacpy.Environment(
            bathymetry=100.0, ssp=[(0.0, 1500.0), (100.0, 1480.0)],
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0,
                density=1.8, attenuation=0.5))

    @pytest.mark.parametrize('freq,floor_applies', [(200.0, True), (1600.0, False)])
    def test_default_density_is_derived_from_frequency(self, freq, floor_applies):
        from uacpy.models.kraken import (MODE_POINTS_PER_WAVELENGTH,
                                         MODE_POINTS_PER_METER_FLOOR)
        env = self._env()
        ppm = uacpy.Kraken()._resolve_mode_points_per_meter(env, [freq])
        needed = MODE_POINTS_PER_WAVELENGTH * freq / 1480.0
        if floor_applies:
            # 10*200/1480 = 1.35 < 1.5, so the floor keeps a low-frequency run
            # from getting a coarser grid than the historical fixed density.
            assert ppm == pytest.approx(MODE_POINTS_PER_METER_FLOOR)
        else:
            assert ppm == pytest.approx(needed)
            assert ppm * 1480.0 / freq == pytest.approx(MODE_POINTS_PER_WAVELENGTH)

    def test_explicit_density_is_honoured_but_warns_when_too_coarse(self):
        env = self._env()
        with pytest.warns(UserWarning, match='points per wavelength'):
            ppm = uacpy.Kraken(mode_points_per_meter=1.5)._resolve_mode_points_per_meter(
                env, [1600.0])
        assert ppm == 1.5                      # verbatim, not silently raised

    def test_adequate_explicit_density_is_silent(self):
        env = self._env()
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            uacpy.Kraken(mode_points_per_meter=20.0)._resolve_mode_points_per_meter(
                env, [1600.0])

    @pytest.mark.slow
    def test_default_grid_agrees_with_scooter_at_high_frequency(self):
        # Scooter is the independent arbiter — wavenumber integration has no
        # mode grid at all. At 1.5 pts/m this measured 8.249 dB.
        env = self._env()
        rcv = uacpy.Receiver(depths=[30.0, 50.0, 75.0],
                             ranges=np.linspace(500.0, 5000.0, 10))
        src = uacpy.Source(depths=20.0, frequencies=1600.0)
        sc = np.squeeze(uacpy.Scooter().run(
            env, src, rcv, run_mode=uacpy.RunMode.COHERENT_TL).db)
        kr = np.squeeze(uacpy.Kraken().run(
            env, src, rcv, run_mode=uacpy.RunMode.COHERENT_TL).db)
        assert np.max(np.abs(kr - sc)) < 1.0
