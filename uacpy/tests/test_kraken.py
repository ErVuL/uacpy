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

    def test_fluid_bottom_mesh_is_unchanged(self):
        """The shear term must not inflate the mesh for a fluid seabed."""
        model = Kraken(verbose=False)
        segments, _, _, max_total_depth = model._segment_env_for_field(
            model._project_environment(self._env([0.0, 0.0, 0.0, 0.0])))
        assert model._fixed_mesh_points(segments, max_total_depth, 50.0) == 500

    @pytest.mark.slow
    def test_mixed_fluid_elastic_bottom_runs(self):
        """The whole point: a fluid→elastic transition across range must give
        a field, not a raw Fortran fatal."""
        result = Kraken(verbose=False, mode_coupling='adiabatic',
                        n_segments=5, timeout=600).run(
            self._env([0.0, 0.0, 400.0, 600.0]), self._SRC(), self._RCV())
        tl = np.asarray(result.tl)
        finite = tl[np.isfinite(tl)]
        assert finite.size, "no finite TL returned"
        assert finite.max() < 200.0, (
            f"max TL {finite.max():.1f} dB — not a physical waterborne field")
        # TL must grow with range, not sit at a constant or run backwards.
        at_source_depth = np.asarray(result.at(depth=50.0).tl)
        assert at_source_depth[-1] > at_source_depth[4] + 10.0

    @pytest.mark.slow
    def test_uniformly_elastic_bottom_still_runs(self):
        """The all-elastic case (which never had the short-shear median) must
        not regress from the larger mesh."""
        result = Kraken(verbose=False, mode_coupling='adiabatic',
                        n_segments=5, timeout=600).run(
            self._env([300.0, 400.0, 500.0, 600.0]), self._SRC(), self._RCV())
        finite = np.asarray(result.tl)[np.isfinite(result.tl)]
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
        assert result.kind == 'transfer_function'
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
            np.asarray(one.tl)[:, :, 0] - np.asarray(two.tl)[:, :, 1])) < 0.5

    def test_one_element_grid_can_synthesize_a_time_series(self):
        """The 2-D fallback made ``synthesize_time_series`` raise on its
        canonical-coords check; a real frequency axis feeds it."""
        result = Kraken(verbose=False).run(
            self._env(), self._SRC(), self._RCV(),
            run_mode=RunMode.BROADBAND, frequencies=np.array([137.0]))
        assert result.phase_reference == 'travelling_wave'


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
        assert result.kind == 'tl'
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
            *common, run_mode=RunMode.COHERENT_TL).tl)[0]
        inc = np.asarray(Kraken(verbose=False).run(
            *common, run_mode=RunMode.INCOHERENT_TL).tl)[0]
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
    """``Kraken`` was hand-merged from a ``_KrakenBase`` + subclass pair, which
    left ``spec`` and ``source`` defined twice in the class body. Python takes
    the last definition, so a duplicate is dead code that silently ignores any
    edit to the earlier copy."""

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
        # RMax must clear the outermost receiver for the modal-sum interpolation.
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
