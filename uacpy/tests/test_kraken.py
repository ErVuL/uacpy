"""Kraken / KrakenC normal-mode-focused tests."""

import pytest
import numpy as np

from uacpy.core.results import Field, Modes
from uacpy.models import Kraken
from uacpy.models.base import RunMode
from uacpy.core import Environment, BoundaryProperties, Source, Receiver

pytestmark = pytest.mark.requires_binary


class TestKrakenFieldBackend:
    """The KrakenField ``backend=`` override (kraken / krakenc)."""

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


class TestKrakenFieldBroadband:
    """End-to-end BROADBAND / TIME_SERIES tests for KrakenField."""

    @pytest.mark.slow
    def test_krakenfield_broadband_returns_transfer_function(self):
        """KrakenField BROADBAND returns H(f) on the receiver grid."""
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
    def test_krakenfield_time_series_returns_time_series_field(self):
        """KrakenField TIME_SERIES with a tonal waveform returns Field."""
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

    """Test KrakenC for complex modes with elastic bottom."""

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
        """Test KrakenC complex mode computation."""
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
    """B6: Kraken / KrakenC expose mode_points_per_meter."""

    @pytest.mark.parametrize('cls', [Kraken])
    def test_default_is_1_5(self, cls):
        m = cls()
        assert m.mode_points_per_meter == 1.5

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

    def test_constructor_no_longer_accepts_source_type(self):
        with pytest.raises(TypeError):
            Kraken(source_type='R')

    def test_constructor_no_longer_accepts_beam_pattern_file(self):
        with pytest.raises(TypeError):
            Kraken(source_beam_pattern_file=None)

    def test_field_option_position_one_tracks_source_type(self):
        model = Kraken()
        codes = {
            t: model._build_field_option(
                False, Source(depths=50, frequencies=100, source_type=t))[0]
            for t in ('point', 'line', 'scaled')
        }
        assert codes == {'point': 'R', 'line': 'X', 'scaled': 'S'}

    def test_field_option_position_three_tracks_beam_pattern(self):
        model = Kraken()
        pat = np.array([[-90.0, -20.0], [90.0, 0.0]])
        omni = model._build_field_option(
            False, Source(depths=50, frequencies=100))
        directional = model._build_field_option(
            False, Source(depths=50, frequencies=100, beam_pattern=pat))
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
    """Regression for the patched AT interp1 (MODIFICATIONS.md).

    A 3-point pattern puts x(N-1) at 0 deg, so every mode angle lands in
    interp1's final segment — which looped forever before the patch.
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
    assert np.isfinite(np.asarray(field.tl)).any()


def test_field_exe_timeout_is_not_swallowed(tmp_path, monkeypatch):
    """A timeout must surface as a timeout, not as a downstream parse error.

    _run_field_exe deliberately tolerates a non-zero teardown status, but a
    timeout means the run never finished — reading the 0-byte .shd it leaves
    behind previously surfaced as FileFormatError from detect_endian.
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

    AT's SubTab does not replicate the Rro sentinel below 3 elements, so a
    two-depth run used to hand the shallowest receiver ro=-999.9 m and
    evaluate its row at r-999.9 — plausible numbers at the wrong ranges.
    """
    env = Environment(name='pek', bathymetry=200.0, ssp=1500.0,
                      bottom=BoundaryProperties(acoustic_type='half-space',
                                                sound_speed=1800.0, density=1.8,
                                                attenuation=0.5))
    src = Source(depths=50.0, frequencies=100.0)
    ranges = np.array([1000., 2000., 3000.])
    m = Kraken(verbose=False)
    tl2 = np.asarray(m.run(env, src, Receiver(depths=[60., 120.], ranges=ranges)).tl)
    tl3 = np.asarray(m.run(env, src, Receiver(depths=[60., 120., 180.], ranges=ranges)).tl)
    np.testing.assert_allclose(tl2[0], tl3[0], rtol=0, atol=0.05)
    np.testing.assert_allclose(tl2[1], tl3[1], rtol=0, atol=0.05)


def test_mode_count_probe_matches_pekeris_theory():
    """``_count_modes_at_freq`` must return real counts, not a swallowed error.

    It called ``read_modes_bin(freq=...)`` while the parameter is
    ``frequency=``; the broad ``except Exception`` turned the resulting
    TypeError into "0 modes" at *every* frequency, so
    ``_propagating_frequency_floor`` concluded nothing propagates and the
    broadband sub-cutoff recovery was dead. Counts are checked against the
    Pekeris estimate M ~ (2 D f / c_w) sqrt(1 - (c_w/c_b)^2) rather than
    against uacpy's own output.
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
    def _env(depth, c):
        return Environment(
            name='x', bathymetry=float(depth), ssp=float(c),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.5))

    _SRC = staticmethod(lambda: Source(depths=25.0, frequencies=500.0))
    _RCV = staticmethod(lambda: Receiver(depths=[50.0, 60.0],
                                         ranges=[1000.0, 2000.0]))

    def test_fatal_error_is_raised_not_swallowed(self, tmp_path):
        """n_mesh=1 at 500 Hz trips 'Mesh is too coarse'
        (ReadEnvironmentMod.f90:110). The binary exits 0, so only a .prt/stderr
        scan catches it."""
        from uacpy.core.exceptions import ModelExecutionError
        with pytest.raises(ModelExecutionError) as ei:
            Kraken(work_dir=str(tmp_path / 'w'), n_mesh=1, timeout=300).run(
                self._env(1000.0, 1480.0), self._SRC(), self._RCV())
        assert 'FATAL ERROR' in str(ei.value) or 'Fatal Error' in str(ei.value), (
            f"the Fortran diagnostic never reached the user: {ei.value}")

    def test_stale_output_is_not_returned_as_this_runs_answer(self, tmp_path):
        """The dangerous case: run 1 succeeds, run 2 fatals on a *different*
        environment, and the stale .mod/.shd yield run 1's field."""
        from uacpy.core.exceptions import ModelExecutionError
        wd = str(tmp_path / 'shared')
        first = np.asarray(Kraken(work_dir=wd, timeout=300).run(
            self._env(100.0, 1500.0), self._SRC(), self._RCV()).tl)
        assert np.all(np.isfinite(first))

        with pytest.raises(ModelExecutionError):
            Kraken(work_dir=wd, n_mesh=1, timeout=300).run(
                self._env(1000.0, 1480.0), self._SRC(), self._RCV())


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
            Receiver(depths=[20.0, 50.0], ranges=[1000.0, 3000.0])).tl)
        finite = tl[np.isfinite(tl)]
        assert finite.size, "no finite TL returned"
        assert finite.max() < 120.0, (
            f"max TL {finite.max():.1f} dB — the mode search converged on "
            f"interfacial modes instead of the waterborne field")
