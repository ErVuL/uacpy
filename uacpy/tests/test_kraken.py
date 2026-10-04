"""Normal-mode tests for ``Kraken``, on both the kraken and krakenc backends.

Backend selection, broadband sweeps, complex eigenvalues, the attenuation
unit and mode sampling — and, at the end, what the model does at the edge of
its own validity: a duct whose mode set is entirely non-trapped, and the
dispatch premise the vendored Fortran either supports or contradicts.
"""

import math
import re
import warnings
import tempfile
from pathlib import Path

import pytest
import numpy as np
import uacpy

from uacpy.core.exceptions import (
    ConfigurationError, ModelExecutionError, NumericsWarning,
    UnsupportedFeatureError,
)
from uacpy.core.environment import Bottom, SeabedColumn, SedimentLayer
from uacpy.core.results import Field, Modes
from uacpy.models import Kraken
from uacpy.core.run_settings import RunMode
from uacpy.core import Environment, BoundaryProperties, Source, Receiver
from uacpy.models.kraken import _checks, _grid, _launch, _modes, _window
from uacpy.tests.conftest import make_pekeris
from uacpy.tests.conftest import make_halfspace
from uacpy.tests.conftest import recorded_warnings

pytestmark = pytest.mark.requires_binary

C_WATER = 1500.0
C_BOTTOM = 1700.0


def _duct(depth, c_bottom=C_BOTTOM, **hs):
    """A hard-floored Pekeris duct: exact cutoff for mode m is
    ``(m - 1/2)·c1 / (2 D sqrt(1 - (c1/c2)^2))``."""
    return Environment(
        name=f'pek{depth:g}', bathymetry=depth,
        ssp=[(0.0, C_WATER), (depth, C_WATER)],
        bottom=BoundaryProperties(sound_speed=c_bottom, density=1.8,
                                  attenuation=0.0, **hs))


def _k_for(phase_speeds, freq):
    """Wavenumbers whose phase speeds are ``phase_speeds`` at ``freq`` Hz."""
    return np.asarray([2.0 * math.pi * freq / c for c in phase_speeds],
                      dtype=complex)


class TestRangeDependentBroadbandRunsPerFrequency:
    """A multi-profile deck carries one frequency; a band is a loop of them.

    KRAKEN solves modes at one frequency whatever the environment, so a
    range-dependent band is not a physical limitation — only the deck
    writer's. Refusing it pushed the loop onto every caller.
    """

    @staticmethod
    def _env():
        import uacpy
        rr = np.linspace(0.0, 6000.0, 7)
        dd = np.interp(rr, [0.0, 3000.0, 6000.0], [200.0, 150.0, 175.0])
        return uacpy.Environment(
            name='rd', bathymetry=list(zip(rr, dd)),
            ssp=[(0.0, 1520.0), (200.0, 1498.0)],
            bottom=uacpy.Bottom.from_halfspace(uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1700.0,
                density=1.9, attenuation=0.5)))

    def test_it_returns_a_frequency_axis(self):
        import uacpy
        from uacpy.models import RunMode
        band = np.linspace(190.0, 210.0, 5)
        out = uacpy.Kraken(verbose=False).run(
            self._env(), uacpy.Source(depths=60.0, frequencies=band),
            uacpy.Receiver(depths=[100.0, 120.0], ranges=[3000.0, 5000.0]),
            run_mode=RunMode.BROADBAND)
        assert list(out.coords) == ['depth', 'range', 'frequency']
        assert np.asarray(out.data).shape == (2, 2, band.size)
        np.testing.assert_allclose(out.coords['frequency'], band)

    def test_each_bin_equals_its_own_single_frequency_run(self):
        """The loop must be the single-frequency path, not a near-miss."""
        import uacpy
        from uacpy.models import RunMode
        env = self._env()
        band = np.linspace(195.0, 205.0, 3)
        rcv = uacpy.Receiver(depths=[100.0, 120.0], ranges=[4000.0])
        wide = uacpy.Kraken(verbose=False).run(
            env, uacpy.Source(depths=60.0, frequencies=band), rcv,
            run_mode=RunMode.BROADBAND)
        for i, f in enumerate(band):
            one = uacpy.Kraken(verbose=False).run(
                env, uacpy.Source(depths=60.0, frequencies=float(f)), rcv)
            np.testing.assert_allclose(
                np.asarray(wide.data)[:, :, i], np.asarray(one.data),
                rtol=1e-12, atol=1e-30)

    def test_it_carries_what_a_native_broadband_field_carries(self):
        """Three stamps travel only on the broadband path.

        Built from one narrowband slab's metadata the stacked field loses
        them, and the losses are not cosmetic: without ``frequencies`` it
        denies being broadband, without ``phase_reference`` a complex
        weighted sum is refused as having none, and without ``c_max`` every
        time-series window falls back to a nominal sound speed and warns.
        """
        import uacpy
        from uacpy.models import RunMode
        band = np.linspace(190.0, 210.0, 4)
        src = uacpy.Source(depths=60.0, frequencies=band)
        rcv = uacpy.Receiver(depths=[100.0], ranges=[3000.0])
        flat = uacpy.Environment(
            name='flat', bathymetry=200.0,
            ssp=[(0.0, 1520.0), (200.0, 1498.0)],
            bottom=uacpy.Bottom.from_halfspace(uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1700.0,
                density=1.9, attenuation=0.5)))
        native = uacpy.Kraken(verbose=False).run(flat, src, rcv,
                                                 run_mode=RunMode.BROADBAND)
        looped = uacpy.Kraken(verbose=False).run(self._env(), src, rcv,
                                                 run_mode=RunMode.BROADBAND)
        np.testing.assert_allclose(looped.frequencies, band)
        assert looped.phase_reference == native.phase_reference
        assert (looped.run_settings.waveguide.c_max
                == native.run_settings.waveguide.c_max)
        # and it says which path built it
        assert looped.metadata['native_broadband'] is False

    def test_it_names_the_engine_and_source_a_native_band_names(self):
        """The looped band carries the native band's identity: field.exe as
        ``backend``, the licence credit, the source depths and level."""
        import uacpy
        from uacpy.models import RunMode
        src = uacpy.Source(depths=60.0, frequencies=np.linspace(
            190.0, 210.0, 3), source_level_dB=170.0)
        rcv = uacpy.Receiver(depths=[100.0], ranges=[3000.0])
        model = uacpy.Kraken(verbose=False)
        looped = model.run(self._env(), src, rcv, run_mode=RunMode.BROADBAND)
        assert looped.metadata['native_broadband'] is False
        assert looped.model == 'Kraken'
        assert looped.backend == 'kraken'
        assert looped.model_source is model.provenance
        np.testing.assert_array_equal(looped.source_depths, [60.0])
        assert looped.source_level_dB == 170.0

    def test_a_range_independent_band_is_untouched(self):
        """The native broadband deck must still be the one that runs."""
        import uacpy
        from uacpy.models import RunMode
        flat = uacpy.Environment(
            name='flat', bathymetry=200.0, ssp=[(0.0, 1520.0), (200.0, 1498.0)],
            bottom=uacpy.Bottom.from_halfspace(uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1700.0,
                density=1.9, attenuation=0.5)))
        band = np.linspace(190.0, 210.0, 5)
        out = uacpy.Kraken(verbose=False).run(
            flat, uacpy.Source(depths=60.0, frequencies=band),
            uacpy.Receiver(depths=[100.0], ranges=[3000.0]),
            run_mode=RunMode.BROADBAND)
        assert np.asarray(out.data).shape == (1, 1, band.size)


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

    @staticmethod
    def _launched(model, env):
        """The modes binary a run of ``model`` on ``env`` launches: the
        backend its settings record, resolved to its executable."""
        settings = model.run_settings(
            env, Source(depths=50.0, frequencies=100.0),
            Receiver(depths=[50.0], ranges=[1000.0]))
        return model._modes_exe(settings.engine.backend).name

    def test_backend_auto_dispatch_fluid_kraken_elastic_krakenc(self):
        assert self._launched(Kraken(verbose=False),
                              self._fluid()) == 'kraken.exe'
        assert self._launched(Kraken(verbose=False),
                              self._elastic()) == 'krakenc.exe'

    def test_backend_override_beats_auto_dispatch(self):
        assert self._launched(Kraken(verbose=False, backend='krakenc'),
                              self._fluid()) == 'krakenc.exe'
        assert self._launched(Kraken(verbose=False, backend='kraken'),
                              self._fluid()) == 'kraken.exe'

    def test_leaky_modes_forces_krakenc_even_on_a_fluid_env(self):
        # kraken.md §5 "so the solver attempts leaky modes": leaky
        # eigenvalues are genuinely complex, so
        # leaky_modes=True dispatches to krakenc.exe regardless of the
        # environment's own (fluid) dispatch. Resolution only — no run.
        assert self._launched(Kraken(verbose=False, leaky_modes=True),
                              self._fluid()) == 'krakenc.exe'

    def test_force_kraken_on_elastic_raises(self):
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match="elastic media"):
            self._launched(Kraken(verbose=False, backend='kraken'),
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
        # One bin per requested frequency, and the frequency coordinate IS
        # the request — kraken's native multi-frequency .mod solves the grid
        # verbatim, no resampling.
        assert result.data.shape[2] == len(frequencies)
        np.testing.assert_allclose(
            np.asarray(result.coords['frequency'], dtype=float), frequencies)

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
        # The synthesis window is 1/Δf, anchored at r/c_max = 2000/1600 s
        # (the default sand half-space is the fastest speed in this env), so
        # it must hold the spread from that earliest possible arrival to the
        # 1500 m/s direct arrival plus the 0.128 s waveform: 0.083 + 0.128 s.
        # Δf = 2.5 Hz gives a 0.4 s window; the round trip stays wrap-free.
        frequencies = np.linspace(60.0, 140.0, 33)

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
        data = np.asarray(result.data)
        assert float(np.sum(data ** 2)) > 0.0, "silent trace returned"
        # The 1/df = 0.2 s synthesis window is anchored so the estimated
        # first arrival r/c sits at its centre (field.py _ifft_to_trace), so
        # the time axis must straddle 2000/1500 s and the envelope peak —
        # dominated by the direct arrival convolved with the 0.128 s
        # waveform — lands at or just after it. Catches a wrong sound speed,
        # a zero anchor, or a seconds/milliseconds axis error outright.
        times = np.asarray(result.coords['time'], dtype=float)
        travel = 2000.0 / 1500.0
        assert times[0] <= travel <= times[-1]
        t_peak = float(times[np.argmax(np.abs(data[0, 0]))])
        assert travel - 0.06 <= t_peak <= travel + 0.25


class TestKrakencComplexModes:
    """The krakenc backend's eigenvalues on an elastic bottom."""

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
        # AT's e^{+i omega t} convention makes the outgoing wave e^{-ikr}
        # (DOCUMENTATION.md §15), so a mode that decays with range needs
        # Im(k) <= 0 — the same validity test the modal-agreement suite
        # applies. A positive imaginary part would grow with range.
        assert np.all(np.imag(k) <= 0.0), (
            f"growing modes returned: Im(k) max = {np.imag(k).max()}")


class TestKrakenAttenuationUnit:
    """TopOpt position 3 is hardwired to ``'W'`` (dB/wavelength) — uacpy's
    documented convention. There is no per-model unit override."""

    def test_writer_emits_W_for_attenuation_unit(self, tmp_path):
        kraken = Kraken()
        env = Environment(name='kr', bathymetry=100.0, ssp=1500.0)
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(depths=[25.0, 50.0, 75.0], ranges=[1000.0])
        env_file = tmp_path / 'kraken.env'
        launch = kraken.run_settings(env, source, receiver).engine.launches[0]
        _launch.write_modes_deck(env_file, env, source, receiver, launch,
                                 interp_ssp=kraken.interp_ssp)
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
        # deferred to run() and resolved from freq_max / c_min instead. The
        # constructor keeps the sentinel; see TestModeGridTracksFrequency.
        assert cls().mode_points_per_meter is None

    @pytest.mark.parametrize('cls', [Kraken])
    def test_density_kwarg_accepted(self, cls):
        m = cls(mode_points_per_meter=3.0)
        assert m.mode_points_per_meter == 3.0

    def test_compute_modes_uses_mode_points_per_meter(self):
        """The dense mode-depth grid scales with mode_points_per_meter, as
        the MODES preview with no receiver shows it."""
        env = Environment(name='kr_modes', bathymetry=200.0, ssp=1500.0)
        source = Source(depths=100.0, frequencies=50.0)
        kraken = Kraken(mode_points_per_meter=5.0)
        depths = kraken.run_settings(
            env, source, None,
            run_mode=RunMode.MODES).engine.launches[0].tabulation_depths
        # 200 m * 5 pts/m = 1000 pts (>=100 floor).
        assert len(depths) == 1000
        assert float(np.max(depths)) == pytest.approx(200.0)


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
        with pytest.raises(TypeError,
                           match="unexpected keyword argument 'source_type'"):
            Kraken(source_type='R')

    def test_constructor_rejects_beam_pattern_file(self):
        with pytest.raises(
                TypeError,
                match="unexpected keyword argument 'source_beam_pattern_file'"):
            Kraken(source_beam_pattern_file=None)

    def test_field_option_position_one_tracks_source_type(self):
        model = Kraken()
        codes = {
            t: _grid.build_field_option(
                False, Source(depths=50, frequencies=100, source_type=t),
                RunMode.COHERENT_TL, mode_coupling=model.mode_coupling)[0]
            for t in ('point', 'line', 'scaled')
        }
        assert codes == {'point': 'R', 'line': 'X', 'scaled': 'S'}

    def test_field_option_position_three_tracks_beam_pattern(self):
        model = Kraken()
        pat = np.array([[-90.0, -20.0], [90.0, 0.0]])
        omni = _grid.build_field_option(
            False, Source(depths=50, frequencies=100), RunMode.COHERENT_TL,
            mode_coupling=model.mode_coupling)
        directional = _grid.build_field_option(
            False, Source(depths=50, frequencies=100, beam_pattern=pat),
            RunMode.COHERENT_TL, mode_coupling=model.mode_coupling)
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
        lines = sbp[0].read_text().splitlines()
        assert lines[0].split()[0] == '3'
        # The rows are the pattern verbatim (angle, dB re peak) — refl_io
        # writes both columns at %.6f, so 5e-7 is pure format rounding.
        rows = np.array([[float(v) for v in ln.split()] for ln in lines[1:4]])
        np.testing.assert_allclose(rows, pat, rtol=0, atol=5e-7)


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
    assert np.isfinite(np.asarray(field.dB)).any()


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

    with pytest.raises(ModelExecutionError,
                       match='execution timed out') as exc:
        model._run_field_exe(fm.work_dir, 'model', 'RC C')
    assert exc.value.timed_out
    assert 'timed out' in str(exc.value).lower()


@pytest.mark.parametrize('completed', [True, False])
def test_a_teardown_exit_is_read_only_after_the_field_completed(
        tmp_path, monkeypatch, completed):
    """field.exe's known non-zero teardown exit is read anyway, with a
    warning, when ``field.prt`` holds the completion line
    (``field.f90:240``); without it the failure is raised with the
    ``field.prt`` tail."""
    from uacpy.core.exceptions import ModelExecutionError

    model = Kraken(work_dir=tmp_path, cleanup=False)
    fm = model._setup_file_manager()

    def fake_run(cmd, **kwargs):
        (fm.work_dir / 'field.prt').write_text(
            'Field completed successfully\n' if completed
            else 'stopped in FreqLoop\n')
        (fm.work_dir / 'model.shd').write_bytes(b'\x00' * 8)
        raise ModelExecutionError('Kraken', return_code=1, stderr='free()')

    monkeypatch.setattr(model, '_run_subprocess', fake_run)
    if completed:
        with pytest.warns(UserWarning, match='known Fortran cleanup issue'):
            shd = model._run_field_exe(fm.work_dir, 'model', 'RC C')
        assert shd == fm.work_dir / 'model.shd'
    else:
        with pytest.raises(ModelExecutionError, match='stopped in FreqLoop'):
            model._run_field_exe(fm.work_dir, 'model', 'RC C')


def test_empty_shd_reports_no_usable_output(tmp_path, monkeypatch):
    """A 0-byte .shd is 'no output', not a file to hand to the reader."""
    from uacpy.core.exceptions import ModelExecutionError

    model = Kraken(work_dir=tmp_path, cleanup=False)
    monkeypatch.setattr(model, '_run_subprocess', lambda cmd, **kw: None)
    fm = model._setup_file_manager()
    (fm.work_dir / 'model.shd').write_bytes(b'')

    with pytest.raises(ModelExecutionError, match="no usable .shd"):
        model._run_field_exe(fm.work_dir, 'model', 'RC C')


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
    tl2 = np.asarray(m.run(env, src, Receiver(depths=[60., 120.], ranges=ranges)).dB)
    tl3 = np.asarray(m.run(env, src, Receiver(depths=[60., 120., 180.], ranges=ranges)).dB)
    np.testing.assert_allclose(tl2[0], tl3[0], rtol=0, atol=0.05)
    np.testing.assert_allclose(tl2[1], tl3[1], rtol=0, atol=0.05)


def _band_stage_inputs(model, env, source, receiver, work_dir,
                       frequencies):
    """What a band run of ``model`` hands its launch hooks, in
    ``work_dir``: the resolved settings and the projected environment."""
    from uacpy.models.base import StageInputs
    settings = model.run_settings(env, source, receiver,
                                  run_mode=RunMode.BROADBAND,
                                  frequencies=frequencies)
    return StageInputs(work_dir=Path(work_dir),
                              env=model._project_environment(env),
                              source=source, receiver=receiver,
                              settings=settings)


def test_mode_count_probe_matches_pekeris_theory(tmp_path):
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
    freqs = np.array([20.0, 50.0, 100.0, 400.0])
    inputs = _band_stage_inputs(m, env, src, rcv, tmp_path, freqs)
    counts = np.array([m._count_modes_at_freq(inputs, float(f))
                       for f in freqs])
    predicted = (2.0 * D * freqs / c_w) * np.sqrt(1.0 - (c_w / c_b) ** 2)

    assert np.all(counts > 0), f"no modes found at any frequency: {counts}"
    assert np.all(np.abs(counts - predicted) <= 0.15 * predicted + 1.5), (
        f"counts {counts} depart from Pekeris estimate {np.round(predicted, 1)}")
    assert np.all(np.diff(counts) > 0), "mode count must rise with frequency"
    # Everything above propagates, so the sub-cutoff prefix is empty.
    assert m._propagating_frequency_floor(inputs, freqs) == 0


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

        ``BoundaryProperties`` refuses an attenuation past
        ``MAX_ATTENUATION_DB_PER_WAVELENGTH`` (54.575 dB/wavelength, derived from
        the same ``CRCI`` abort), at construction and on assignment, so the
        value is stored past the carrier's checks with ``object.__setattr__``,
        standing in for a carrier gap. That is
        the point of these two tests: the carrier guard is the first line of
        defence and the ``.prt``/stderr scan is the second, for a fatal that
        arrives some other way — a hand-edited deck, a future carrier gap, or a
        different AT fatal entirely."""
        env = cls._env(depth, c)
        object.__setattr__(env.bottom.columns[0].halfspace, 'attenuation',
                           100.0)
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
        with pytest.raises(ModelExecutionError, match=r'Error output:\nSTOP '):
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
        with pytest.raises(
                ModelExecutionError,
                match='STOP Fatal Error: Check the print file') as ei:
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
            self._env(100.0, 1500.0), self._SRC(), self._RCV()).dB)
        assert np.all(np.isfinite(first))

        with pytest.raises(ModelExecutionError,
                           match='STOP Fatal Error: Check the print file'):
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
        model = Kraken()
        assert _window.c_low_for(
            env, collapse=model._collapse,
            pinned_c_low=model.c_low) == pytest.approx(1500.0)

    def test_fluid_env_delegates_to_kraken(self):
        env = Environment(
            name='fl', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.5))
        model = Kraken()
        assert _window.c_low_for(env, collapse=model._collapse,
                                 pinned_c_low=model.c_low) == 0.0

    def test_explicit_c_low_always_wins(self):
        model = Kraken(c_low=1234.0)
        assert _window.c_low_for(self._elastic_env(), collapse=model._collapse,
                                 pinned_c_low=model.c_low) == 1234.0

    def test_c_low_reads_the_slowest_column_of_a_range_dependent_ssp(self):
        """The floor is stamped into every profile block of the multi-profile
        deck, and ``kraken.f90:230`` / ``krakenc.f90:230`` only ever raise it
        (``cLow = MAX( cLow, cMin )``), so a floor read off the range-0 column
        deletes every mode slower than it in the profiles further out. Measured
        on this environment at 200 Hz: the range-0 reading (1500) returned
        24/14/12/3 modes per profile against 24/26/28/31 for the block minimum,
        a mean 8.1 dB / max 28.4 dB TL difference."""
        from uacpy.core.ssp import SoundSpeedProfile
        from uacpy.core.environment import SeabedColumn
        ssp = SoundSpeedProfile(
            depths=np.array([0.0, 100.0, 200.0]),
            sound_speed=np.array([[1500.0, 1480.0, 1450.0]] * 3),
            ranges=np.array([0.0, 5000.0, 10000.0]))
        env = Environment(
            name='rd-el',
            bathymetry=np.array([[0.0, 200.0], [10000.0, 220.0]]),
            ssp=ssp,
            bottom=SeabedColumn(
                layers=[],
                halfspace=BoundaryProperties(
                    acoustic_type='half-space', sound_speed=1600.0,
                    density=1.8, attenuation=0.2, shear_speed=400.0,
                    shear_attenuation=0.5)))
        assert float(ssp.to_pairs()[:, 1].min()) == 1500.0, "range-0 is faster"
        model = Kraken()
        assert _window.c_low_for(
            env, collapse=model._collapse,
            pinned_c_low=model.c_low) == pytest.approx(1450.0)

    @pytest.mark.parametrize('cs_layer', [400.0, 600.0])
    def test_elastic_layer_tl_is_physical_by_default(self, cs_layer):
        """Default run must not return the interfacial-mode field (700+ dB)."""
        env = self._elastic_env(cs_layer)
        tl = np.asarray(Kraken(timeout=300).run(
            env, Source(depths=36.0, frequencies=100.0),
            Receiver(depths=[20.0, 50.0], ranges=[1000.0, 3000.0])).dB)
        finite = tl[np.isfinite(tl)]
        assert finite.size, "no finite TL returned"
        assert finite.max() < 120.0, (
            f"max TL {finite.max():.1f} dB — the mode search converged on "
            f"interfacial modes instead of the waterborne field")


class TestRangeDependentElasticMesh:
    """A range-dependent seabed whose columns straddle fluid and elastic —
    shear speeds ``[0, 0, 400, 600]`` — is written column by column into the
    profile blocks, so some profiles carry an elastic medium with a *short*
    shear wavelength. AT meshes an elastic medium on its shear speed
    (``misc/ReadEnvironmentMod.f90:99-104``), so a mesh sized on the
    compressional speed is rejected with the fatal 'Mesh is too coarse'."""

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

    def test_auto_mesh_defers_to_kraken_per_profile(self, tmp_path):
        """The automatic path writes ``NG=0`` on every medium line: the
        reader sizes each medium of each profile itself — on the shear
        speed wherever one is set (``misc/ReadEnvironmentMod.f90:99-110``)
        — and the ``.mod`` record length carries no mesh term
        (``kraken.f90:587``), so no shared padded count is pinned across
        profiles."""
        from uacpy.io.oalib_writer import write_multi_profile_env
        model = Kraken(verbose=False)
        env = self._env([0.0, 0.0, 400.0, 600.0])
        segments, _, _, _max_total_depth = _grid.segment_env_for_field(
            model._project_environment(env), log=model._log,
            mode_coupling=model.mode_coupling, n_segments=model.n_segments)
        assert _grid.multi_profile_n_mesh(segments, 50.0,
                                          pinned_n_mesh=model.n_mesh) == 0

        out = tmp_path / 'auto.env'
        write_multi_profile_env(out, segments, self._SRC(), self._RCV(),
                                n_mesh=_grid.multi_profile_n_mesh(
                                    segments, 50.0,
                                    pinned_n_mesh=model.n_mesh),
                                c_low=0.0, c_high=2000.0)
        ngs = [int(ln.split()[0]) for ln in out.read_text().splitlines()
               if len(ln.split()) == 3 and ln.split()[0].isdigit()]
        assert len(ngs) >= 2 * len(segments), f"mesh lines missing: {ngs}"
        assert set(ngs) == {0}, (
            f"expected NG=0 on every mesh line, got {sorted(set(ngs))}")

    def test_an_explicit_n_mesh_round_trips_to_every_profile(self, tmp_path):
        """A user-pinned ``n_mesh`` is still written verbatim on every
        medium line of every profile."""
        from uacpy.io.oalib_writer import write_multi_profile_env
        model = Kraken(n_mesh=2000, verbose=False)
        env = self._env([0.0, 0.0, 400.0, 600.0])
        segments, _, _, _max_total_depth = _grid.segment_env_for_field(
            model._project_environment(env), log=model._log,
            mode_coupling=model.mode_coupling, n_segments=model.n_segments)
        out = tmp_path / 'pinned.env'
        write_multi_profile_env(out, segments, self._SRC(), self._RCV(),
                                n_mesh=_grid.multi_profile_n_mesh(
                                    segments, 50.0,
                                    pinned_n_mesh=model.n_mesh),
                                c_low=0.0, c_high=2000.0)
        ngs = [int(ln.split()[0]) for ln in out.read_text().splitlines()
               if len(ln.split()) == 3 and ln.split()[0].isdigit()]
        assert len(ngs) >= 2 * len(segments), f"mesh lines missing: {ngs}"
        assert set(ngs) == {2000}, (
            f"pinned n_mesh did not round-trip: {sorted(set(ngs))}")

    def test_top_reflection_file_reaches_the_range_dependent_deck(self,
                                                                  tmp_path):
        """``_write_field_env`` branches to ``write_multi_profile_env`` and
        never calls ``_write_kraken_env``, so the knob has to be expressed on
        the projected environment rather than inside the single-profile
        writer. Expressed there only, the range-dependent deck carries a
        vacuum ``TopOpt`` and no staged ``.trc`` — a silently different
        surface on exactly the runs the knob routes to krakenc."""
        from uacpy.io.oalib_writer import write_multi_profile_env
        from uacpy.models.kraken._segments import segment_environment_by_range

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
            n_mesh=500, c_low=0.0, c_high=2000.0)

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
        from uacpy.core.boundary import SedimentLayer
        from uacpy.core.bottom import Bottom, SeabedColumn
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
        segments, _, _, _max_total = _grid.segment_env_for_field(
            model._project_environment(env), log=model._log,
            mode_coupling=model.mode_coupling, n_segments=model.n_segments)

        from uacpy.io.oalib_writer import at_mesh_floor
        media = _grid.multi_profile_media(segments)
        # No single medium is thicker than the 50 m sediment layers, so at
        # 1500 m/s and 100 Hz the coarsest wants INT(50/(1500/100/20)) = 66
        # points and the floor is 66 // 2.
        assert max(t for t, _ in media) == pytest.approx(50.0)
        assert at_mesh_floor(media, 100.0) == 33

    def test_fluid_bottom_pinned_floor_meshes_on_cp(self):
        """For a fluid seabed the pinned-mesh floor is measured on the
        compressional speed — the shear term must not inflate it."""
        from uacpy.io.oalib_writer import at_mesh_floor
        model = Kraken(verbose=False)
        segments, _, _, _max_total_depth = _grid.segment_env_for_field(
            model._project_environment(self._env([0.0, 0.0, 0.0, 0.0])),
            log=model._log, mode_coupling=model.mode_coupling,
            n_segments=model.n_segments)
        media = _grid.multi_profile_media(segments)
        # With every shear speed at 0 the floor is driven by the 400 m water
        # column at 1485 m/s — Nneeded = int(20 * 400 * 50 / 1485) = 269,
        # rejected below 269 // 2 — not by the 200 m/s shear number the
        # elastic variant of this seabed produces (750).
        assert at_mesh_floor(media, 50.0) == 134

    @pytest.mark.slow
    def test_mixed_fluid_elastic_bottom_runs(self):
        """The whole point: a fluid→elastic transition across range must give
        a field, not a raw Fortran fatal."""
        result = Kraken(verbose=False, mode_coupling='adiabatic',
                        n_segments=5, timeout=600).run(
            self._env([0.0, 0.0, 400.0, 600.0]), self._SRC(), self._RCV())
        tl = np.asarray(result.dB)
        # Judge the waterborne field: receivers below the local seafloor sit
        # deep in the fluid half-space of the shallow profiles, where the
        # field is evanescent (measured 213.8 dB at 374 m under a 112 m
        # seafloor).
        env = self._env([0.0, 0.0, 400.0, 600.0])
        depths = np.asarray(result.coords['depth'], dtype=float)
        seafloor = np.asarray(env.bathymetry.eval(
            range=np.asarray(result.coords['range'], dtype=float)),
            dtype=float).ravel()
        water = depths[:, None] <= seafloor[None, :]
        finite = tl[np.isfinite(tl) & water]
        assert finite.size, "no finite TL returned"
        assert finite.max() < 200.0, (
            f"max TL {finite.max():.1f} dB — not a physical waterborne field")
        # TL must grow with range, not sit at a constant or run backwards.
        at_source_depth = np.asarray(result.at(depth=50.0).dB)
        assert at_source_depth[-1] > at_source_depth[4] + 10.0

    @pytest.mark.slow
    def test_uniformly_elastic_bottom_runs(self):
        """A uniformly elastic seabed has no fluid column to drag its median
        shear speed down, so it never needs the enlarged mesh — and must not be
        broken by it either."""
        result = Kraken(verbose=False, mode_coupling='adiabatic',
                        n_segments=5, timeout=600).run(
            self._env([300.0, 400.0, 500.0, 600.0]), self._SRC(), self._RCV())
        finite = np.asarray(result.dB)[np.isfinite(result.dB)]
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
            np.asarray(one.dB)[:, :, 0] - np.asarray(two.dB)[:, :, 1])) < 0.5

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
        from uacpy.core.boundary import SedimentLayer
        from uacpy.core.bottom import Bottom, SeabedColumn
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
        segments, _, _, max_total_depth = _grid.segment_env_for_field(
            env, log=model._log, mode_coupling=model.mode_coupling,
            n_segments=model.n_segments)
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
        tl = np.asarray(result.dB)
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
        with pytest.raises(ModelExecutionError,
                           match='modes must be tabulated throughout') as ei:
            _launch.raise_on_field_fatal(tmp_path, model_name=model.model_name)
        assert 'modes must be tabulated' in str(ei.value), (
            f"field.exe's own diagnosis never reached the user: {ei.value}")

    def test_a_clean_field_prt_passes(self, tmp_path):
        model = Kraken(verbose=False)
        (tmp_path / 'field.prt').write_text(" Running FIELD\n Coherent\n")
        _launch.raise_on_field_fatal(
            tmp_path, model_name=model.model_name)      # must not raise
        _launch.raise_on_field_fatal(tmp_path / 'nonexistent',
                                     model_name=model.model_name)


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
            *common, run_mode=RunMode.COHERENT_TL).dB)[0]
        inc = np.asarray(Kraken(verbose=False).run(
            *common, run_mode=RunMode.INCOHERENT_TL).dB)[0]
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
    sub-cutoff recovery. They write their .env/.mod/.prt into the work
    directory of the run they belong to, under their own root, so they are
    let go with it and claim no directory of their own."""

    def test_the_probe_writes_into_its_runs_work_dir(self, tmp_path):
        env = Environment(
            name='probe', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.8,
                                      attenuation=0.5))
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))

        from uacpy.models._workspace import FileManager

        model = Kraken(verbose=False)
        created = []
        create = FileManager.create_work_dir

        def _record(self):
            path = create(self)
            created.append(path)
            return path

        inputs = _band_stage_inputs(model, env, source, receiver, tmp_path,
                                    np.array([90.0, 100.0]))
        FileManager.create_work_dir = _record
        try:
            # One propagating probe and one below cutoff.
            for freq in (100.0, 1.0):
                model._count_modes_at_freq(inputs, freq)
        finally:
            FileManager.create_work_dir = create

        assert not created, f"the probe claimed work dirs: {created}"
        assert (tmp_path / 'mcut.env').exists()


class TestKrakenClassBody:
    """``spec`` and ``provenance_id`` must each be assigned once in ``Kraken``'s class
    body. Python keeps only the last assignment, so a duplicate is dead code
    that silently ignores every edit to the earlier copy — and the class body
    is long enough that a second assignment is easy to miss on review."""

    @staticmethod
    def _class_body_assignments():
        import ast
        import inspect
        from uacpy.models.kraken import _model as mod

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

    @pytest.mark.parametrize('name', ['spec', 'provenance_id'])
    def test_defined_exactly_once(self, name):
        names = self._class_body_assignments()
        assert names.count(name) == 1, (
            f"Kraken defines {name!r} {names.count(name)} times in its class "
            f"body; only the last one is live")


class TestResolvedPhaseSpeedBoundsAreRunSettings:
    """Every Kraken result that ran the solver states the resolved ``c_low``
    / ``c_high`` / ``rmax_m`` the deck was written with — usually
    auto-derived, not supplied — in its run settings
    (``run_settings.engine.launches``), once: they are not copied into the
    metadata (decision 4)."""

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
            assert key not in result.metadata, (
                f"{result.model} result copies the setting {key!r} into "
                f"its metadata")
        launch = result.run_settings.engine.launches[0]
        assert launch.c_low < max(launch.c_high)
        # c_high brackets the whole medium (SSP max 1520, half-space 1600).
        assert max(launch.c_high) >= 1600.0
        # RMax scales the mesh-convergence tolerance (kraken.f90:80), so it
        # must clear the longest range the modes are propagated to.
        assert launch.rmax_m > rmax_floor

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
    def test_sub_cutoff_bins_are_nan_and_keep_bounds(self):
        """The broadband recovery path rebuilds the Field around the
        propagating sub-band, so it must not drop the bounds on the way. The
        sub-cutoff bins are NaN — the narrowband path's no-data value — not a
        0 that reads as a perfectly quiet channel; the propagating bins are
        finite."""
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
        assert np.all(np.isnan(result.data[:, :, :2]))
        assert np.all(np.isfinite(result.data[:, :, 2:]))
        assert result.sub_cutoff_bins == 2
        self._assert_sane(result, rmax_floor=float(receiver.ranges.max()))

    @pytest.mark.slow
    def test_sub_cutoff_bins_synthesise_as_zero_in_a_time_series(self):
        # The NaN bins are H(f)'s no-data value; synthesis needs finite bins
        # and a sub-cutoff bin adds no modal energy, so the pulse is finite.
        env = Environment(
            name='ts_cut', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.3))
        fs = 200.0
        t = np.arange(64) / fs
        pulse = np.sin(2 * np.pi * 20.0 * t) * np.hanning(t.size)
        with pytest.warns(UserWarning, match="below the modal cutoff"):
            ts = Kraken(verbose=False).run(
                env, Source(depths=50.0, frequencies=20.0),
                Receiver(depths=np.array([25.0]), ranges=np.array([1000.0])),
                run_mode=RunMode.TIME_SERIES, source_waveform=pulse,
                sample_rate=fs, frequencies=np.arange(2.0, 40.5, 0.5))
        assert np.all(np.isfinite(ts.data))
        assert np.max(np.abs(ts.data)) > 0.0

    def test_range_dependent_field_path_records_bounds(self):
        from uacpy.core.ssp import SoundSpeedProfile

        env = Environment(
            name='bounds_rd', bathymetry=200.0,
            ssp=SoundSpeedProfile(
                depths=[0.0, 200.0],
                sound_speed=[[1500.0, 1500.0], [1520.0, 1560.0]],
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
        assert result.run_settings.engine.launches[0].c_low == 1400.0
        assert max(result.run_settings.engine.launches[0].c_high) == 1700.0
        assert result.run_settings.engine.launches[0].rmax_m == 9000.0

    def test_list_metadata_holds_no_copy_of_the_bounds(self):
        """The bounds are described where they live, in the settings
        record, and not a second time as metadata."""
        source, receiver = self._geometry()
        result = Kraken(verbose=False).run(
            self._env(), source, receiver, run_mode=RunMode.COHERENT_TL)
        described = result.list_metadata()
        for key in self.KEYS:
            assert key not in described
        assert 'c_high' in str(result.run_settings)


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
            env, Source(depths=50.0, frequencies=100.0), receiver).dB)
    assert np.all(np.isnan(tl[:, 0]))
    assert np.all(np.isfinite(tl[:, 1:]))


def test_mode_depths_spans_the_column_the_deck_carries():
    """``compute_modes`` writes the r = 0 column's stack into the MODES deck
    (``_bottom_collapse_for(MODES) == 'r0'``) and KRAKEN clamps any receiver
    below that deck onto it (``misc/SourceReceiverPositions.f90:136-139``),
    so the depth grid spans the water column plus THAT column's sediment —
    keyed on ``bottom.at(range=0.0)``, not on storage order and not on the
    thickest column along the track. Here the r = 0 column is 20 m against
    80 m at 5 km, so the grid reaches 100 + 20 = 120 m."""
    from uacpy.core.boundary import SedimentLayer
    from uacpy.core.bottom import Bottom, SeabedColumn

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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')     # the r = 0 reduction is announced
        depths = Kraken(verbose=False).run_settings(
            env, Source(depths=50.0, frequencies=100.0), None,
            run_mode=RunMode.MODES).engine.launches[0].tabulation_depths

    assert depths[-1] == pytest.approx(120.0)


# ── deck contract: what the vendored reader actually consumes ────────────

def _pekeris(depth=200.0, c=1500.0, ssp=None):
    return make_pekeris(
        name='deck', bathymetry=depth,
        ssp=ssp if ssp is not None else [(0.0, c), (depth, c)],
        sound_speed=1800.0, attenuation=0.3)


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

    def test_a_short_range_run_refines_the_mesh(self, tmp_path):
        auto = Kraken(work_dir=tmp_path / 'auto', cleanup=False)
        tl_auto = np.asarray(auto.compute_tl(
            self._env(), self._SRC(), self._RCV()).dB)
        assert self._deck_rmax(tmp_path / 'auto') > 0.0, (
            "RMax reached the deck as 0.0 km — kraken.f90:80 then skips every "
            "mesh doubling")

        pinned = Kraken(rmax_m=1000.0, work_dir=tmp_path / 'pin', cleanup=False)
        tl_pinned = np.asarray(pinned.compute_tl(
            self._env(), self._SRC(), self._RCV()).dB)
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
                    self._SRC(), self._RCV()).dB)
        surface = np.asarray(Kraken(
            work_dir=tmp_path / 'sfc', cleanup=False).compute_tl(
                _pekeris(ssp=[(0.0, 1500.0), (200.0, 1500.0)]),
                self._SRC(), self._RCV()).dB)
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
            surface=Surface(nodes=[BoundaryProperties(
                acoustic_type='file',
                reflection_file=str(self._table(tmp_path, 'top.trc')))]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.3))

    def test_a_bottom_brc_runs_on_krakenc(self, tmp_path):
        env = self._brc_env(tmp_path)
        model = Kraken(work_dir=tmp_path / 'w', cleanup=False)
        assert model.select_backend(env) == 'krakenc'
        tl = np.asarray(model.compute_tl(env, self._SRC(), self._RCV()).dB)
        assert np.all(np.isfinite(tl)) and tl.max() < 200.0

    def test_a_top_trc_gives_the_same_field_on_auto_and_forced_krakenc(
            self, tmp_path):
        env = self._trc_env(tmp_path)
        auto = Kraken(work_dir=tmp_path / 'a', cleanup=False)
        assert auto.select_backend(env) == 'krakenc'
        forced = Kraken(backend='krakenc', work_dir=tmp_path / 'b',
                        cleanup=False)
        assert np.allclose(
            np.asarray(auto.compute_tl(env, self._SRC(), self._RCV()).dB),
            np.asarray(forced.compute_tl(env, self._SRC(), self._RCV()).dB))

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
            surface=Surface(nodes=[BoundaryProperties(
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
            cleanup=False).compute_tl(_pekeris(), src, rcv).dB)
        env = _pekeris()
        env.surface = Surface(nodes=[BoundaryProperties(
            acoustic_type='file', reflection_file=str(table))])
        carrier = np.asarray(Kraken(
            work_dir=tmp_path / 'c', cleanup=False).compute_tl(
                env, src, rcv).dB)
        assert np.allclose(knob, carrier)
        assert (tmp_path / 'k' / 'kfield.trc').exists()


def _rough_surface(sigma, acoustic_type='vacuum', reflection_file=None):
    """A single-node ``Surface`` carrying ``sigma`` on ``acoustic_type``."""
    from uacpy.core.surface import Surface
    kw = {}
    if reflection_file is not None:
        kw['reflection_file'] = str(reflection_file)
    return Surface(nodes=[BoundaryProperties(
        acoustic_type=acoustic_type, roughness=sigma, **kw)])


class TestElasticSeaSurfaceRunsUnderTheCompressionalFloor:
    """An elastic sea surface is accepted, and what makes it work is the
    ``c_low`` floor, not a guard.

    ``krakenc.f90:220-222`` folds ``HSTop%cS`` into ``cMin`` symmetrically
    with the seabed at ``:210-212``, and ``:228-230`` then applies
    ``IF (ElasticFlag) cMin = 0.85 * cMin``. Left to KRAKEN, the search floor
    lands near the ice shear speed and the solver chases the Scholte mode;
    :func:`~uacpy.models.kraken._window.c_low_for` writes the minimum
    compressional speed instead.
    """

    @staticmethod
    def _ice_env(depth=300.0):
        from uacpy.core.surface import Surface
        return Environment(
            name='canopy', bathymetry=depth, ssp=[(0.0, 1500.0),
                                                  (depth, 1500.0)],
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.7,
                                      attenuation=0.5),
            surface=Surface(nodes=[BoundaryProperties(
                acoustic_type='half-space', sound_speed=3500.0,
                shear_speed=1800.0, density=0.9, attenuation=1.0,
                shear_attenuation=2.0)]))

    def test_the_floor_is_the_water_speed_not_the_shear_derived_one(self):
        """Where the clamp lands, computed — not which MAX it is written as.
        ``0.85 * 1800 = 1530`` m/s would sit *above* the 1500 m/s water and
        delete the waterborne modes."""
        model = Kraken(verbose=False)
        floor = _window.c_low_for(self._ice_env(), collapse=model._collapse,
                                  pinned_c_low=model.c_low)
        assert floor == pytest.approx(1500.0)
        assert floor < 0.85 * 1800.0

    def test_an_elastic_surface_is_not_refused(self):
        """No guard rejects it — the model dispatches to krakenc and runs."""
        env = self._ice_env()
        model = Kraken(verbose=False)
        assert model.select_backend(env) == 'krakenc'
        model.validate_inputs(
            env, Source(depths=100.0, frequencies=200.0),
            Receiver(depths=[150.0], ranges=[5000.0]))

    @pytest.mark.parametrize('freq', [200.0, 1000.0, 2000.0])
    def test_an_ice_canopy_returns_finite_physical_tl(self, freq):
        """Including the two frequencies that failed before the floor was
        fixed (1 kHz and 2 kHz: 'No modes for given phase speed interval')."""
        dB = np.asarray(Kraken(verbose=False).compute_tl(
            self._ice_env(), Source(depths=100.0, frequencies=freq),
            Receiver(depths=[150.0], ranges=[5000.0])).dB).ravel()
        assert np.isfinite(dB).all()
        assert (0.0 < dB).all() and (dB < 200.0).all(), (
            f"TL {dB} at {freq:g} Hz is outside the physical band — a "
            f"Scholte-mode solve reads as several hundred dB")


class TestElasticSurfaceFloorIncludesTheFluidSeabed:
    """The derived ``c_low`` under an elastic surface reads the fluid seabed's
    compressional speeds, not just the water column: ``krakenc.f90:230`` only
    ever *raises* the written floor (``cLow = MAX( cLow, cMin )``), so a floor
    at the water minimum deletes every mode ducted in a sediment layer slower
    than the water."""

    @staticmethod
    def _ice_over_slow_mud_env():
        from uacpy.core.surface import Surface
        return Environment(
            name='canopy-mud', bathymetry=100.0, ssp=1500.0,
            bottom=SeabedColumn(
                layers=[SedimentLayer(thickness=20.0, sound_speed=1450.0,
                                      density=1.5, attenuation=0.1)],
                halfspace=BoundaryProperties(
                    acoustic_type='half-space', sound_speed=1700.0,
                    density=1.8, attenuation=0.5)),
            surface=Surface(nodes=[BoundaryProperties(
                acoustic_type='half-space', sound_speed=3500.0,
                shear_speed=1800.0, density=0.9, attenuation=1.0,
                shear_attenuation=2.0)]))

    def test_derived_floor_is_the_slowest_seabed_speed(self):
        # water 1500, mud layer 1450, halfspace 1700 -> 1450
        model = Kraken()
        assert _window.c_low_for(self._ice_over_slow_mud_env(),
                                 collapse=model._collapse,
                                 pinned_c_low=model.c_low) == \
            pytest.approx(1450.0)

    @pytest.mark.requires_binary
    def test_slow_sediment_modes_survive_the_derived_floor(self):
        env = self._ice_over_slow_mud_env()
        source = Source(depths=50.0, frequencies=200.0)
        receiver = Receiver(depths=[30.0, 60.0, 90.0],
                            ranges=[2000.0, 5000.0])
        k_default = Kraken(verbose=False).compute_modes(env, source).k
        k_explicit = Kraken(c_low=1450.0, verbose=False).compute_modes(
            env, source).k
        assert len(k_default) == len(k_explicit), (
            f"the derived floor lost {len(k_explicit) - len(k_default)} of "
            f"{len(k_explicit)} modes")
        dB_default = np.asarray(Kraken(verbose=False).compute_tl(
            env, source, receiver).dB)
        dB_explicit = np.asarray(Kraken(c_low=1450.0, verbose=False)
                                 .compute_tl(env, source, receiver).dB)
        np.testing.assert_allclose(dB_default, dB_explicit, rtol=0, atol=0.01)


class TestSurfaceRoughnessOnATabulatedTop:
    """A tabulated top boundary cannot carry the sea-surface roughness.

    ``Kraken/kraken.f90:850-867`` selects on ``HSTop%BC``; the ``CASE DEFAULT``
    that ``TopOpt(2:2)='F'`` lands in sets ``rho1 = eta1Sq = 0``, so the
    Kuperman-Ingenito determinant ``Del = rho1*eta2 + rho2*eta1``
    (``Kraken/Scattering.f90:21``) is exactly zero, ``Scattering.f90:23`` is
    false and ``KupIng`` returns its initialised ``0.0D0``
    (``Scattering.f90:17``). ``'A'``, ``'V'`` and ``'R'`` all leave ``rho1``
    non-zero and do carry it."""

    @staticmethod
    def _trc(tmp_path):
        table = tmp_path / 'top.trc'
        table.write_text("3\n0.0 0.6 180.0\n45.0 0.6 180.0\n90.0 0.6 180.0\n")
        return table

    def test_the_top_reflection_file_knob_drops_the_roughness_and_says_so(
            self, tmp_path):
        env = _pekeris()
        env.surface = _rough_surface(0.5)
        model = Kraken(top_reflection_file=self._trc(tmp_path), verbose=False)
        with pytest.warns(UserWarning, match='Scattering.f90'):
            projected = model._project_environment(env)
        assert projected.surface.roughness == 0.0

    def test_a_user_built_file_surface_drops_the_roughness_and_says_so(
            self, tmp_path):
        """The second reachable path: no ``top_reflection_file=``, so the
        projection's own rewrite never runs and the drop has to key on the
        resolved surface."""
        env = _pekeris()
        env.surface = _rough_surface(
            0.5, acoustic_type='file', reflection_file=self._trc(tmp_path))
        with pytest.warns(UserWarning, match='Scattering.f90'):
            projected = Kraken(verbose=False)._project_environment(env)
        assert projected.surface.roughness == 0.0

    def test_a_smooth_tabulated_top_projects_without_a_warning(self, tmp_path):
        """The low side of the ``sigma`` threshold: nothing to drop, nothing
        to say."""
        env = _pekeris()
        env.surface = _rough_surface(
            0.0, acoustic_type='file', reflection_file=self._trc(tmp_path))
        with recorded_warnings() as caught:
            projected = Kraken(verbose=False)._project_environment(env)
        assert projected.surface.roughness == 0.0
        assert not [w for w in caught if 'Scattering.f90' in str(w.message)]

    @pytest.mark.parametrize('acoustic_type', ['vacuum', 'rigid', 'half-space'])
    def test_a_top_the_scatter_branch_handles_keeps_the_roughness(
            self, acoustic_type):
        """The other side of the ``HSTop%BC`` boundary — ``'V'``, ``'R'`` and
        ``'A'`` each leave ``rho1`` non-zero, so the drop must not fire."""
        env = _pekeris()
        kw = ({'sound_speed': 340.0, 'density': 0.0012, 'attenuation': 0.0}
              if acoustic_type == 'half-space' else {})
        from uacpy.core.surface import Surface
        env.surface = Surface(nodes=[BoundaryProperties(
            acoustic_type=acoustic_type, roughness=0.5, **kw)])
        with recorded_warnings() as caught:
            projected = Kraken(verbose=False)._project_environment(env)
        assert projected.surface.roughness == 0.5
        assert not [w for w in caught if 'Scattering.f90' in str(w.message)]

    def test_the_roughness_this_guard_protects_moves_a_vacuum_top_field(
            self, tmp_path):
        """The observable is live at this fixture: under a vacuum top the same
        two roughness values the guard separates move the field by percent,
        so a zero difference under a tabulated top is the engine discarding
        the value, not a fixture that cannot see it."""
        src = Source(depths=[25.0], frequencies=[100.0])
        rcv = Receiver(depths=[50.0, 80.0], ranges=[2000.0, 5000.0])
        fields = []
        for i, sigma in enumerate((0.0, 0.5)):
            env = _pekeris(depth=100.0)
            env.surface = _rough_surface(sigma)
            fields.append(np.asarray(Kraken(
                verbose=False, work_dir=tmp_path / f'v{i}',
                cleanup=False).compute_tl(env, src, rcv).data).ravel())
        moved = float(np.max(np.abs(fields[0] - fields[1])))
        scale = float(np.max(np.abs(fields[0])))
        assert moved > 0.005 * scale, (
            f"vacuum-top roughness moved the field by {moved:g} on a scale of "
            f"{scale:g}: the control is too insensitive to license any claim "
            f"about the tabulated case")


class TestFieldTabulationDepthLimit:
    """``KrakenField/ReadModes.f90:8`` declares ``MaxN = 100001`` and reads the
    mode table's depths into the static ``Z( MaxN )`` with no bound test, so
    a table longer than that overruns field.exe's memory (measured: silent
    on a range-independent run, SIGSEGV adiabatic, a record error coupled).
    The table is the tabulation grid merged with the source depths."""

    @staticmethod
    def _grid(n):
        return np.linspace(0.0, 100.0, n)

    def test_exactly_maxn_depths_including_the_source_pass(self):
        from uacpy.models.kraken._grid import _FIELD_MAX_TABULATION_DEPTHS
        grid = self._grid(_FIELD_MAX_TABULATION_DEPTHS - 1)
        source = Source(depths=[float(np.mean(grid[:2]))], frequencies=[100.0])
        _grid.check_field_tabulation_size(grid, source)

    def test_one_depth_past_maxn_is_refused(self):
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.models.kraken._grid import _FIELD_MAX_TABULATION_DEPTHS
        grid = self._grid(_FIELD_MAX_TABULATION_DEPTHS)
        source = Source(depths=[float(np.mean(grid[:2]))], frequencies=[100.0])
        with pytest.raises(ConfigurationError, match='MaxN'):
            _grid.check_field_tabulation_size(grid, source)

    def test_a_fine_grid_is_refused_before_the_binary_runs(
            self, tmp_path, monkeypatch):
        from uacpy.core.exceptions import ConfigurationError
        model = Kraken(mode_points_per_meter=1200, work_dir=tmp_path,
                       cleanup=False)

        def _no_launch(*args, **kwargs):
            raise AssertionError("a binary was launched past the guard")

        monkeypatch.setattr(model, '_run_subprocess', _no_launch)
        monkeypatch.setattr(model, '_run_and_attach_prt', _no_launch)
        with pytest.raises(ConfigurationError, match='MaxN'):
            model.compute_tl(_pekeris(depth=100.0),
                             Source(depths=[50.0], frequencies=[100.0]),
                             Receiver(depths=[30.0], ranges=[1000.0]))


class TestFieldExeFailureCarriesItsOwnLog:
    """field.exe writes ``field.prt`` (``field.f90:44``); ``<base>.prt`` is
    the modes run's log of a successful mode calculation, so a field.exe
    failure quotes the former."""

    @pytest.mark.parametrize('timed_out', [True, False])
    def test_the_raised_error_quotes_field_prt(self, tmp_path, monkeypatch,
                                               timed_out):
        from uacpy.core.exceptions import ModelExecutionError
        from uacpy.models._workspace import FileManager
        model = Kraken()
        fm = FileManager(base_dir=tmp_path)
        fm.create_work_dir()
        (fm.work_dir / 'kfield.prt').write_text('MODES RUN LOG\n')

        def _fails(cmd, cwd, **kwargs):
            (Path(cwd) / 'field.prt').write_text('FIELD RUN LOG\n')
            raise ModelExecutionError('Kraken', return_code=1, stdout=None,
                                      stderr='field.exe failed',
                                      timed_out=timed_out)

        monkeypatch.setattr(model, '_run_subprocess', _fails)
        with pytest.raises(ModelExecutionError,
                           match='field.exe failed') as caught:
            model._run_field_exe(fm.work_dir, 'kfield', 'RA')
        assert 'FIELD RUN LOG' in str(caught.value)
        assert 'MODES RUN LOG' not in str(caught.value)


@pytest.mark.parametrize('interp', ['linear', 'pchip', 'spline'])
def test_the_modes_deck_is_written_with_the_models_ssp_interpolation(
        monkeypatch, interp):
    """``Kraken(interp_ssp=...)`` reaches the modes deck a run writes (the
    single-profile writer, called from the launch stage)."""
    seen = []

    class _Written(Exception):
        pass

    def _spy(*args, **kwargs):
        seen.append(kwargs['interp_ssp'])
        raise _Written

    monkeypatch.setattr(_launch, 'write_kraken_env_file', _spy)
    with pytest.raises(_Written):
        Kraken(verbose=False, interp_ssp=interp).run(
            _pekeris(depth=100.0), Source(depths=50.0, frequencies=100.0),
            Receiver(depths=[20.0, 60.0], ranges=[1000.0]))
    assert seen == [interp]


class TestModeFileSizeWarning:
    """The ``.mod`` grows as frequencies x modes x depths (measured 1.7, 6.2,
    23.6 MB for 32 bins to 0.5, 1, 2 kHz on 100 m). On disk a run estimated
    past 2 GiB is warned about before it fills the work directory; in a work
    directory held in memory the file goes through the memory budget.
    Estimate: ``sum(2*D*f/c_min) * n_depths * 8`` bytes."""

    # 101 depths spanning 100 m, one 1500 Hz bin in 1500 m/s water:
    # 200 modes x 101 depths x 8 bytes = 161600.
    ESTIMATE = 161600

    def _in_memory(self, monkeypatch, free):
        from uacpy.models import _budget
        monkeypatch.setattr(_budget, 'available_memory_bytes', lambda: free)
        return _grid.mode_file_size_notice(
            _pekeris(depth=100.0), np.linspace(0.0, 100.0, 101), [1500.0],
            memory_backed=True)

    def test_in_memory_half_the_free_memory_is_silent(self, monkeypatch):
        assert self._in_memory(monkeypatch, 2 * self.ESTIMATE) is None

    def test_in_memory_over_half_the_free_memory_is_announced(
            self, monkeypatch):
        notice = self._in_memory(monkeypatch, 2 * self.ESTIMATE - 1)
        assert 'held in memory' in notice.message

    def test_in_memory_over_the_free_memory_is_refused(self, monkeypatch):
        from uacpy.core.exceptions import ConfigurationError
        assert self._in_memory(monkeypatch, self.ESTIMATE) is not None
        with pytest.raises(ConfigurationError, match='mode file'):
            self._in_memory(monkeypatch, self.ESTIMATE - 1)

    def test_on_disk_the_free_memory_is_not_weighed(self, monkeypatch):
        from uacpy.models import _budget
        monkeypatch.setattr(_budget, 'available_memory_bytes', lambda: 1)
        assert _grid.mode_file_size_notice(
            _pekeris(depth=100.0), np.linspace(0.0, 100.0, 101), [1500.0],
            memory_backed=False) is None

    def test_the_run_asks_whether_its_work_dir_is_in_memory(
            self, monkeypatch, tmp_path):
        """The launch weighs the file against the free memory exactly when
        its own work directory is memory-backed."""
        from uacpy.models import _budget
        from uacpy.models.kraken import _model as kraken_model
        asked = []

        def fake(policy):
            asked.append(policy.pinned_dir)
            return True
        monkeypatch.setattr(kraken_model, 'work_dir_is_memory_backed', fake)
        monkeypatch.setattr(_budget, 'available_memory_bytes', lambda: 1)
        model = Kraken(work_dir=tmp_path, cleanup=False)
        with pytest.raises(ConfigurationError, match='held in memory'):
            model.run_settings(_pekeris(depth=100.0),
                               Source(depths=50.0, frequencies=100.0),
                               Receiver(depths=[20.0], ranges=[1000.0]))
        assert asked == [tmp_path]

    @pytest.mark.parametrize('threshold, warns', [(161600, False),
                                                  (161599, True)])
    def test_the_warning_sits_on_the_estimate(self, monkeypatch, threshold,
                                              warns):
        env = _pekeris(depth=100.0)
        model = Kraken()
        monkeypatch.setattr(_grid, '_MOD_FILE_WARNING_BYTES', threshold)
        # 101 depths spanning 100 m, one 1500 Hz bin in 1500 m/s water:
        # 200 modes x 101 depths x 8 bytes = 161600.
        notice = _grid.mode_file_size_notice(
            env, np.linspace(0.0, 100.0, 101), [1500.0],
            memory_backed=False)
        assert (notice is not None and 'mode file' in notice[1]) is warns


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
        with pytest.raises(
                ModelExecutionError,
                match='beam-pattern angles are not monotonic') as ei:
            _launch.raise_on_field_fatal(tmp_path, model_name='Kraken')
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
        # At 7.5 Hz this deck makes kraken.exe abort with a raw Fortran
        # RECL error while writing the empty modes.mod, which uacpy wraps
        # as a typed ModelExecutionError but without the friendlier 'no
        # mode' diagnosis that the clean zero-mode path produces.
        with pytest.raises(ModelExecutionError,
                           match='found no mode with a phase speed inside'):
            Kraken(c_low=1790.0, c_high=1799.0, work_dir=tmp_path,
                   cleanup=False).compute_modes(
                       env, Source(depths=[50.0], frequencies=[20.0]))
        assert (tmp_path / 'modes.mod').stat().st_size > 0, (
            "the no-modes .mod is non-empty, so a size test cannot detect it")

    def test_both_backends_name_the_phase_speed_window(self, tmp_path):
        """``krakenc.f90:432-446`` writes records 1 and 5 of the same header
        that ``kraken.f90:947-962`` writes 1 and 7 of, so the krakenc
        no-modes ``.mod`` is 640 bytes against kraken's 896 and the reader
        runs off the end of it. That state arrives as a ``FileFormatError``
        rather than as ``M == 0``, so it reaches a different handler — and
        used to surface as a raw complaint about a short file instead of the
        "widen [c_low, c_high]" guidance the kraken path gives."""
        from uacpy.core.exceptions import ModelExecutionError
        env = Environment(name='zm', bathymetry=100.0,
                          ssp=[(0.0, 1500.0), (100.0, 1500.0)],
                          bottom=BoundaryProperties(
                              acoustic_type='half-space', sound_speed=1800.0,
                              density=1.8, attenuation=0.3))
        src = Source(depths=[50.0], frequencies=[100.0])
        sizes = {}
        for backend in ('kraken', 'krakenc'):
            with pytest.raises(
                    ModelExecutionError,
                    match='found no mode with a phase speed inside') as ei:
                Kraken(verbose=False, backend=backend, c_low=1400.0,
                       c_high=1450.0, work_dir=tmp_path / backend,
                       cleanup=False).compute_modes(env, src)
            assert 'c_low' in str(ei.value) and 'c_high' in str(ei.value), (
                f"{backend} lost the phase-speed diagnosis: {ei.value}")
            sizes[backend] = (tmp_path / backend / 'modes.mod').stat().st_size
        assert sizes['krakenc'] < sizes['kraken'], (
            f"the two dummy .mod files are the same size ({sizes}), so this "
            f"no longer exercises the short-file reader path")


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
        from uacpy.core.exceptions import UnsupportedFeatureError
        env = self._rd_env() if range_dependent else _pekeris()
        with pytest.raises(UnsupportedFeatureError, match="'quad'"):
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
        from uacpy.core.boundary import SedimentLayer
        from uacpy.core.bottom import Bottom, SeabedColumn
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
        return _modes.modes_error_message(str(tmp_path / 'run'))

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

    def test_a_fluid_half_space_is_not_diagnosed_as_elastic(self, tmp_path):
        """``misc/ReadEnvironmentMod.f90:260-266`` echoes 'ACOUSTO-ELASTIC
        half-space' for every ``HS%BC == 'A'`` — the letter a plain FLUID
        Pekeris bottom uses — so the marker cannot separate a fluid seabed
        from an elastic one. Diagnosing on it sent every unrecognised
        fluid-seabed failure to "try backend='krakenc'"."""
        msg = self._message(
            tmp_path,
            '     ACOUSTO-ELASTIC half-space\n'
            ' Some other failure the .prt does not name\n')
        assert 'krakenc' not in msg, msg
        assert 'elastic' not in msg.lower(), msg
        assert 'run.prt' in msg, msg          # the generic pointer instead

    def test_the_marker_is_what_a_fluid_pekeris_deck_really_writes(self):
        """The premise above, taken from the vendored source rather than
        from a hand-written .prt: the CASE('A') branch that prints the
        marker is the same one a fluid half-space takes."""
        src = (Path(uacpy.__file__).parent / 'third_party'
               / 'Acoustics-Toolbox' / 'misc' / 'ReadEnvironmentMod.f90')
        text = src.read_text(errors='replace')
        assert "CASE ( 'A' )" in text
        head = text[:text.index('ACOUSTO-ELASTIC half-space')]
        # The CASE selecting that WRITE is the bare 'A' boundary-condition
        # letter, with no shear test between the two: a fluid half-space and
        # an elastic one both land on it.
        last_case = re.findall(r"CASE\s*\(\s*'([A-Z])'\s*\)", head)[-1]
        assert last_case == 'A', last_case
        between = head[head.rindex("CASE"):]
        assert 'cS' not in between and 'shear' not in between.lower(), between

    def test_a_named_failure_beats_the_generic_pointer(self, tmp_path):
        """Dropping the elastic branch must not swallow the two diagnoses
        that do carry information, including on a deck that also printed the
        half-space marker."""
        msg = self._message(
            tmp_path,
            '     ACOUSTO-ELASTIC half-space\n'
            ' No modes for given phase speed interval\n')
        assert 'c_low' in msg and 'c_high' in msg


class TestBeamPatternOnMultipleFrequencies:
    """``KrakenField/field.f90`` allocated ``kz2``/``thetaT``/``S`` inside
    ``FreqLoop`` guarded only by ``SBPFlag == '*' .AND. iS == 1``, while the
    matching ``DEALLOCATE`` sat after the loop closed, so the second frequency
    re-allocated an already-allocated array and gfortran terminated — and the
    same guard applied the shading to the first source depth only. uacpy
    patches that block (third_party/MODIFICATIONS.md): allocate per frequency,
    apply per source. These pin the capability the patch buys; they fail
    against an unpatched Acoustics-Toolbox."""

    PATTERN = np.array([[-90.0, 0.0], [90.0, 0.0]])
    #: A pattern with a real roll-off, so "was the shading applied?" is a
    #: question the field can answer. The flat ``PATTERN`` above cannot.
    SHAPED = np.array([[-90.0, -40.0], [-20.0, -40.0], [0.0, 0.0],
                       [20.0, -40.0], [90.0, -40.0]])

    @pytest.mark.requires_binary
    def test_a_multi_frequency_run_accepts_the_pattern(self, tmp_path):
        """The re-allocation is per frequency now, so the second bin no
        longer aborts the run."""
        model = Kraken(work_dir=tmp_path, cleanup=False)
        field = model.run(
            _pekeris(depth=100.0),
            Source(depths=[25.0], frequencies=[180.0, 200.0, 220.0],
                   beam_pattern=self.PATTERN),
            Receiver(depths=[50.0], ranges=[1000.0, 2000.0]),
            run_mode=RunMode.BROADBAND)
        assert np.asarray(field.data).shape[-1] == 3
        assert np.all(np.isfinite(np.asarray(field.data)))

    @pytest.mark.requires_binary
    def test_every_source_depth_of_a_stack_is_shaded_not_only_the_first(
            self, tmp_path):
        """The shading is loop-invariant over source depth, so it is computed
        once — but it must be *applied* to every depth. Each slab must equal
        that depth's own single-source run, and a shaped pattern must move it
        away from the unshaded one."""
        env = _pekeris(depth=100.0)
        receiver = Receiver(depths=[30.0, 50.0, 70.0],
                            ranges=[1000.0, 2000.0, 3000.0])
        depths = [25.0, 60.0]
        stack = Kraken(work_dir=tmp_path, cleanup=False).run(
            env, Source(depths=depths, frequencies=200.0,
                        beam_pattern=self.SHAPED), receiver)
        for i, z in enumerate(depths):
            shaded = Kraken().run(
                env, Source(depths=z, frequencies=200.0,
                            beam_pattern=self.SHAPED), receiver)
            plain = Kraken().run(
                env, Source(depths=z, frequencies=200.0), receiver)
            np.testing.assert_allclose(stack[i].data, shaded.data, rtol=1e-9,
                                       err_msg=f"slab {i} (z={z} m) unshaded")
            assert not np.allclose(np.abs(stack[i].data), np.abs(plain.data)), (
                f"the pattern did not reach slab {i} (z={z} m)")

    @pytest.mark.requires_binary
    def test_a_single_frequency_accepts_the_pattern(self, tmp_path):
        model = Kraken(work_dir=tmp_path, cleanup=False)
        tl = model.compute_tl(
            _pekeris(depth=100.0),
            Source(depths=[25.0], frequencies=200.0, beam_pattern=self.PATTERN),
            Receiver(depths=[50.0], ranges=[1000.0, 2000.0]))
        assert np.all(np.isfinite(np.asarray(tl.dB)))

    def test_field_completion_marker_separates_teardown_from_a_real_abort(
            self, tmp_path):
        """``field.f90:240`` writes the marker after ``FreqLoop`` and before the
        clean-up block, so its absence means the run died while computing."""
        model = Kraken(work_dir=tmp_path, cleanup=False)
        prt = tmp_path / 'field.prt'
        prt.write_text(' some output\n Field completed successfully\n')
        assert _launch.field_reached_completion(tmp_path)
        prt.write_text(' some output\n At line 191 of file field.f90\n')
        assert not _launch.field_reached_completion(tmp_path)


class TestKrakenSourceBeamPatternRestrictions:
    """``field.exe`` shades MODE amplitudes, not launch angles
    (``KrakenField/field.f90:189-212``), and the limits that follow from
    that are invisible in the ``.sbp`` file itself. The class docstring
    documents them; these read them back out of the vendored source, so a
    vendored update that lifts one of them shows up as a red test rather
    than as documentation nobody rechecked — which is what happened to the
    first-source-depth restriction, lifted by uacpy's own patch to that
    block (third_party/MODIFICATIONS.md). Two limits remain."""

    PATTERN = np.array([[-90.0, 0.0], [90.0, 0.0]])

    @staticmethod
    def _field_f90():
        return (Path(uacpy.__file__).parent / 'third_party'
                / 'Acoustics-Toolbox' / 'KrakenField'
                / 'field.f90').read_text(errors='replace').splitlines()

    def test_the_shading_reaches_every_source_depth_not_only_the_first(self):
        """The restriction this class used to record. Upstream gated the
        whole block on ``iS == 1``, so only the first source of a
        multi-source run was shaded; uacpy's patch keeps the computation
        under that guard — the shading is loop-invariant over source depth —
        and lifts the *application* out of it. Read back from the source so
        a vendored update that reverts it shows up here."""
        lines = self._field_f90()
        assert 'SourceDepths: DO iS = 1, Pos%Nsz' in lines[183], lines[183]
        # The block is entered for every source depth ...
        assert lines[189].strip() == "IF ( SBPFlag == '*' ) THEN", lines[189]
        assert 'iS == 1' not in lines[189], lines[189]
        # ... the invariant setup still runs once ...
        assert lines[199].strip() == 'IF ( iS == 1 ) THEN', lines[199]
        assert 'interp1' in lines[208], lines[208]
        assert lines[209].strip() == 'END IF', lines[209]
        # ... and the shading is applied outside that inner guard.
        assert 'C( 1 : Msrc ) * REAL( S )' in lines[210], lines[210]

    def test_the_work_arrays_are_reallocated_once_per_frequency(self):
        """The other half of the same patch: upstream allocated under
        ``iS == 1`` with the matching DEALLOCATE after ``FreqLoop``, so a
        second frequency re-allocated an allocated array and aborted."""
        lines = self._field_f90()
        assert 'IF ( ALLOCATED( kz2 ) ) DEALLOCATE' in lines[200], lines[200]
        assert 'ALLOCATE( kz2( MSrc )' in lines[201], lines[201]

    def test_the_reference_speed_is_hard_coded_not_the_source_speed(self):
        lines = self._field_f90()
        assert 'c0' in lines[202] and '1500' in lines[202], lines[202]
        # Porter's own note that this is the wrong speed.
        assert 'should be speed at the source depth' in lines[202], lines[202]

    def test_slow_modes_are_clamped_into_the_zero_degree_bin(self):
        lines = self._field_f90()
        assert 'WHERE ( kz2 < 0 ) kz2 = 0' in lines[205], lines[205]
        # ATAN of a non-negative root over a positive k: [0, 90) only, so the
        # negative half of the pattern table is unreachable.
        assert 'ATAN( SQRT( kz2 )' in lines[207], lines[207]

    def test_bellhop_shades_the_signed_launch_angle_instead(self):
        """Why the same .sbp is not portable: Bellhop interpolates the table
        at SrcDeclAngle, the signed take-off angle in degrees."""
        text = (Path(uacpy.__file__).parent / 'third_party'
                / 'Acoustics-Toolbox' / 'Bellhop'
                / 'bellhop.f90').read_text(errors='replace')
        assert 'SrcDeclAngle = RadDeg * Angles%alpha( ialpha )' in text
        assert ('s    = ( SrcDeclAngle  - SrcBmPat( IBP, 1 ) )' in text)


class TestAutoSegmentationIsWritableAtDeckResolution:
    """``models/kraken/_segments.py`` unions the bathymetry / SSP /
    RD-bottom change points itself, so it must not produce two ranges the
    ``.flp`` cannot tell apart. A bathymetry axis and an SSP axis naming the
    same physical range through different arithmetic differ in the last
    bits; both survived a ``set()``, printed as one token, and
    ``KrakenField/EvaluateADMod.f90:75`` divided by the zero gap with no
    diagnostic — a partly-NaN field, no error, no warning."""

    @staticmethod
    def _env(ssp_break_m):
        from uacpy.core import SoundSpeedProfile, Bathymetry
        z = np.array([0.0, 100.0, 200.0])
        ssp = SoundSpeedProfile(
            depths=z,
            sound_speed=np.column_stack([[1500.0, 1495.0, 1490.0],
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

    def test_a_segment_seafloor_sits_on_its_quantised_profile_end(self):
        """RA-CONTRACT-15: the 0.1 m depth quantum applies to the segment's
        bathymetry as well as to its profile, so mm-scale bathymetry noise
        cannot change the deck, and the profile's last sample IS the
        seafloor (no second sample 4 cm below it, no relabelled one)."""
        from uacpy.core import Bathymetry
        from uacpy.models.kraken._segments import segment_environment_by_range
        env = Environment(
            bathymetry=Bathymetry(ranges=[0.0, 1000.0, 2000.0],
                                  depths=[100.06, 100.04, 100.26]),
            ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.8,
                                      attenuation=0.5))
        for r, seg in segment_environment_by_range(env, n_segments=3):
            depth = float(seg.depth)
            assert depth == round(depth, 1), (r, depth)
            assert float(np.asarray(seg.ssp.depths)[-1]) == depth, (r, depth)

    def test_segment_axis_never_collapses_at_the_deck_quantum(self):
        from uacpy.models.kraken._segments import segment_environment_by_range
        from uacpy.core.deck_limits import DECK_RANGE_RESOLUTION_M
        segments = segment_environment_by_range(self._env(4000.0000001))
        ranges = np.array([r for r, _e in segments], dtype=float)
        assert np.all(np.diff(ranges) > DECK_RANGE_RESOLUTION_M)

    @pytest.mark.requires_binary
    def test_a_1e_7_metre_shift_in_a_break_range_does_not_change_the_field(self):
        """The two breaks are the same physical range, so the TL must be
        identical — not merely finite."""
        src = Source(depths=50.0, frequencies=200.0)
        rcv = Receiver(depths=np.linspace(20.0, 180.0, 5),
                       ranges=np.linspace(1000.0, 9000.0, 5))
        on_node = np.asarray(Kraken().compute_tl(self._env(4000.0), src, rcv).dB)
        off_node = np.asarray(
            Kraken().compute_tl(self._env(4000.0000001), src, rcv).dB)
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
        from uacpy.models.kraken._grid import (
            MODE_POINTS_PER_WAVELENGTH,
            MODE_POINTS_PER_METER_FLOOR,
        )
        env = self._env()
        model = uacpy.Kraken()
        ppm = _grid.mode_points_per_meter(
            env, [freq],
            pinned_mode_points_per_meter=model.mode_points_per_meter)[0]
        needed = MODE_POINTS_PER_WAVELENGTH * freq / 1480.0
        if floor_applies:
            # 10*200/1480 = 1.35 < 1.5, so the floor keeps a low-frequency run
            # from getting a coarser grid than the historical fixed density.
            assert ppm == pytest.approx(MODE_POINTS_PER_METER_FLOOR)
        else:
            assert ppm == pytest.approx(needed)
            assert ppm * 1480.0 / freq == pytest.approx(MODE_POINTS_PER_WAVELENGTH)

    def test_explicit_density_is_honoured_with_a_notice_when_too_coarse(
            self):
        env = self._env()
        model = uacpy.Kraken(mode_points_per_meter=1.5)
        ppm, notice = _grid.mode_points_per_meter(
            env, [1600.0],
            pinned_mode_points_per_meter=model.mode_points_per_meter)
        assert ppm == 1.5                      # verbatim, not silently raised
        assert 'points per wavelength' in notice.message

    def test_adequate_explicit_density_is_silent(self):
        env = self._env()
        model = uacpy.Kraken(mode_points_per_meter=20.0)
        _ppm, notice = _grid.mode_points_per_meter(
            env, [1600.0],
            pinned_mode_points_per_meter=model.mode_points_per_meter)
        assert notice is None

    def test_density_is_sized_on_the_slowest_column_of_a_range_dependent_ssp(self):
        """One tabulation grid is built for the whole multi-profile deck, so
        the density has to clear ~10 pts/wavelength in the slowest water
        anywhere on the track — not just at r = 0. Reading the range-0 column
        of this 1500 → 1000 m/s profile gives 13.3 pts/m against the 20 pts/m
        the block minimum requires, a 0.075 m grid where 0.050 m is needed."""
        from uacpy.core.ssp import SoundSpeedProfile
        from uacpy.models.kraken._grid import MODE_POINTS_PER_WAVELENGTH
        ssp = SoundSpeedProfile(
            depths=np.array([0.0, 100.0, 200.0]),
            sound_speed=np.array([[1500.0, 1200.0, 1000.0]] * 3),
            ranges=np.array([0.0, 5000.0, 10000.0]))
        env = uacpy.Environment(
            bathymetry=np.array([[0.0, 200.0], [10000.0, 220.0]]), ssp=ssp,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1600.0,
                density=1.8, attenuation=0.2))
        assert float(ssp.to_pairs()[:, 1].min()) == 1500.0, "range-0 is faster"
        model = uacpy.Kraken()
        ppm = _grid.mode_points_per_meter(
            env, [2000.0],
            pinned_mode_points_per_meter=model.mode_points_per_meter)[0]
        assert ppm == pytest.approx(
            MODE_POINTS_PER_WAVELENGTH * 2000.0 / 1000.0)

    @pytest.mark.slow
    def test_default_grid_agrees_with_scooter_at_high_frequency(self):
        # Scooter is the independent arbiter — wavenumber integration has no
        # mode grid at all. At 1.5 pts/m this measured 8.249 dB.
        env = self._env()
        rcv = uacpy.Receiver(depths=[30.0, 50.0, 75.0],
                             ranges=np.linspace(500.0, 5000.0, 10))
        src = uacpy.Source(depths=20.0, frequencies=1600.0)
        sc = np.squeeze(uacpy.Scooter().run(
            env, src, rcv, run_mode=uacpy.RunMode.COHERENT_TL).dB)
        kr = np.squeeze(uacpy.Kraken().run(
            env, src, rcv, run_mode=uacpy.RunMode.COHERENT_TL).dB)
        assert np.max(np.abs(kr - sc)) < 1.0


class TestModesPathKeepsTheFullEnvContext:
    """``_checks.modes_single_profile`` reduces a range-dependent env to
    its r=0 profile for the modes solve. The rebuilt env must carry the
    original's altimetry (so ``_project_environment`` still discloses
    dropping it),
    plus the geolocation / transect / date / provenance fields — a reduced
    profile is still the same place and time."""

    @staticmethod
    def _rd_env():
        return Environment(
            name='rd', bathymetry=[(0.0, 100.0), (5000.0, 120.0)],
            ssp=1500.0,
            altimetry=[(0.0, 0.5), (5000.0, -0.5)],
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.5,
                                      attenuation=0.5),
            location=(43.0, 5.0), date='2020-06-01')

    def test_reduced_env_carries_context(self):
        from uacpy.data.sources import SOURCES, DataProvenance
        env = self._rd_env()
        env.bathymetry.data_sources = (DataProvenance(source=SOURCES['gebco']),)
        env.extra_data_sources = (DataProvenance(source=SOURCES['woa23']),)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = Kraken(verbose=False)
            reduced = _checks.modes_single_profile(env,
                                                   collapse=model._collapse,
                                                   model_name=model.model_name)
        assert reduced.altimetry is not env.altimetry
        assert np.array_equal(reduced.altimetry.heights,
                              env.altimetry.heights)
        assert reduced.location == env.location
        assert reduced.date == env.date
        assert reduced.data_sources == env.data_sources
        assert not reduced.is_range_dependent or reduced.altimetry is not None

    def test_the_r0_reduction_is_said_before_the_dropped_keywords(self):
        """Stage 1 of MODES reduces the environment to its r = 0 profile
        before stage 2 drops the TIME_SERIES keywords MODES does not read, so
        the reduction's notice comes first."""
        env = self._rd_env()
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(depths=[50.0], ranges=[1000.0])
        with recorded_warnings() as caught:
            Kraken(verbose=False).run_settings(
                env, source, receiver, run_mode=RunMode.MODES,
                source_waveform=np.ones(8), sample_rate=1000.0)
        said = [str(w.message) for w in caught]
        reduced = [i for i, m in enumerate(said)
                   if 'normal modes are range-independent' in m]
        dropped = [i for i, m in enumerate(said)
                   if 'ignoring source_waveform=' in m]
        assert len(reduced) == 1 and len(dropped) == 1, said
        assert reduced[0] < dropped[0], said

    def test_the_modes_settings_read_the_r0_profile(self):
        """Stage 1 of a MODES call reduces the environment to its r = 0
        profile before anything else reads it, so the waveguide speeds the
        settings record are the r = 0 column's (1500-1520 m/s; a rigid floor
        adds none), not the fastest speed along the track (1560 m/s at
        5 km), which a field mode on the same environment records."""
        from uacpy.core.ssp import SoundSpeedProfile
        env = Environment(
            name='rd_ssp', bathymetry=200.0,
            ssp=SoundSpeedProfile(
                depths=[0.0, 200.0],
                sound_speed=[[1500.0, 1500.0], [1520.0, 1560.0]],
                ranges=[0.0, 5000.0]),
            bottom=BoundaryProperties(acoustic_type='rigid'))
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(depths=[50.0], ranges=[1000.0])
        model = Kraken(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            modes = model.run_settings(env, source, receiver,
                                       run_mode=RunMode.MODES)
            field = model.run_settings(env, source, receiver,
                                       run_mode=RunMode.COHERENT_TL)
        assert (modes.waveguide.c_min, modes.waveguide.c_max) == (1500.0,
                                                                 1520.0)
        assert field.waveguide.c_max == 1560.0

    def test_modes_run_discloses_the_dropped_altimetry(self):
        env = self._rd_env()
        with recorded_warnings() as caught:
            Kraken(verbose=False).run(
                env, Source(depths=25.0, frequencies=100.0),
                Receiver(depths=[50.0], ranges=[1000.0]),
                run_mode=RunMode.MODES)
        assert any('altimetry' in str(w.message) for w in caught), (
            "the modes path silently swallowed the altimetry-collapse "
            "disclosure")

    def test_compute_modes_discloses_each_collapse_once(self):
        # compute_modes hands the env through to run(), whose modes path
        # projects exactly once — the disclosure must not be duplicated by
        # a second projection in the wrapper.
        env = self._rd_env()
        with recorded_warnings() as caught:
            Kraken(verbose=False).compute_modes(
                env, Source(depths=25.0, frequencies=100.0))
        alti = [w for w in caught if 'altimetry' in str(w.message)]
        assert len(alti) == 1, (
            f"altimetry-collapse disclosure emitted {len(alti)} times")


class TestModesPathDisclosesTheCollapseItOverrides:
    """``_checks.modes_single_profile`` samples r = 0 for every
    range-dependent quantity, overriding the configured ``collapse``
    methods. That is the right physics — a single-profile solve at the
    source's own waveguide,
    coherent with the field path whose first segment is that same column,
    where honouring ``'mean'`` for the SSP while the bottom and surface stay
    at r = 0 would build a waveguide that exists at no range at all — but a
    setting the user configured and did not get has to be named."""

    @staticmethod
    def _rd_ssp():
        from uacpy.core.environment import SoundSpeedProfile
        return SoundSpeedProfile.from_2d(
            depths=np.array([0.0, 100.0]),
            ranges=np.array([0.0, 5000.0, 10000.0]),
            matrix=np.array([[1500.0, 1510.0, 1520.0],
                             [1490.0, 1500.0, 1510.0]]))

    def _env(self, *, rd_bathymetry=False):
        bathymetry = ([(0.0, 100.0), (10000.0, 120.0)] if rd_bathymetry
                      else 100.0)
        return Environment(
            name='rd-ssp-modes', bathymetry=bathymetry, ssp=self._rd_ssp(),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.5,
                                      attenuation=0.5))

    @staticmethod
    def _reduce(model, env):
        with recorded_warnings() as caught:
            _checks.modes_single_profile(env, collapse=model._collapse,
                                         model_name=model.model_name)
        return ' '.join(str(w.message) for w in caught)

    def test_the_spec_advertises_no_ssp_collapse(self):
        """``collapse['ssp']`` is read only where the model does NOT support
        range-dependent SSP, so Kraken advertising a method there promised a
        collapse it can never perform."""
        assert 'ssp' not in Kraken.spec.collapse
        assert 'range_dependent_ssp' in Kraken.spec.supports
        model = Kraken(verbose=False)
        assert model._supports_range_dependent_ssp is True
        assert model._collapse['ssp'] == 'r0'

    def test_the_field_path_keeps_every_ssp_range(self):
        """The inherited ``'r0'`` is not applied either: the field path
        segments the range-dependent SSP natively and keeps all three
        columns."""
        model = Kraken(verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            projected = model._project_environment(self._env())
        assert projected.ssp.n_ranges == 3
        assert projected.ssp.is_range_dependent

    def test_a_configured_ssp_collapse_is_named_as_dropped(self):
        model = Kraken(verbose=False, collapse={'ssp': 'mean'})
        message = self._reduce(model, self._env())
        assert "collapse['ssp']='mean'" in message, message

    def test_the_default_ssp_collapse_adds_no_noise(self):
        """The inherited ``'r0'`` IS what the modes path applies, so there is
        nothing to disclose and the message stays the bare r=0 statement."""
        message = self._reduce(Kraken(verbose=False), self._env())
        assert 'r=0 profile' in message
        assert 'drops' not in message, message

    def test_a_configured_bottom_range_collapse_is_named_as_dropped(self):
        env = Environment(
            name='rd-bottom-modes', bathymetry=100.0, ssp=1500.0,
            bottom=_rd_layered_bottom(shear_at_r0=0.0, shear_elsewhere=0.0))
        message = self._reduce(
            Kraken(verbose=False, collapse={'bottom_range': 'median'}), env)
        assert "collapse['bottom_range']='median'" in message, message

    def test_the_bottom_range_default_adds_no_noise(self):
        # Kraken declares no bottom_range default: the field path segments the
        # bottom, and the inherited 'r0' is what the modes path samples.
        env = Environment(
            name='rd-bottom-modes', bathymetry=100.0, ssp=1500.0,
            bottom=_rd_layered_bottom(shear_at_r0=0.0, shear_elsewhere=0.0))
        message = self._reduce(Kraken(verbose=False), env)
        assert 'bottom_range' not in message, message

    def test_a_configured_bathymetry_collapse_is_named_as_dropped(self):
        message = self._reduce(Kraken(verbose=False),
                               self._env(rd_bathymetry=True))
        assert "collapse['bathymetry']='max'" in message, message

    def test_a_range_independent_env_is_not_reduced_or_announced(self):
        env = Environment(
            name='ri', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.5,
                                      attenuation=0.5))
        model = Kraken(verbose=False)
        assert self._reduce(model, env) == ''
        assert _checks.modes_single_profile(env, collapse=model._collapse,
                                            model_name=model.model_name) is env


class TestCoarseMeshIsValidatedAtTheDeckFreq0:
    """A pinned ``n_mesh`` is checked against the AT reader's floor at the
    deck's ``freq0`` — the first frequency — because that is the only place
    the reader ever applies it.

    ``misc/ReadEnvironmentMod.f90:103-112`` sizes ``Nneeded`` from ``freq0``
    alone, during the environment read, and stops with *Mesh is too coarse*
    on ``NG < Nneeded/2`` there and nowhere else. ``kraken.f90:75`` then
    marches each swept frequency on ``N = NG · NV(iSet) · freq/freq0``, so a
    mesh that clears the floor at ``freq0`` stays proportionally as fine at
    every frequency above it. Re-deriving the floor at ``max(frequencies)``
    instead refused decks the binary runs happily.

    ``freq0`` is the deck's first frequency record, which
    ``oalib_writer.write_header`` takes from ``source.frequencies[0]``
    whatever the broadband vector holds — so the two must be pinned
    together or the guard drifts off the frequency it is guarding.
    """

    @staticmethod
    def _env():
        return Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.5,
                                      attenuation=0.5))

    @staticmethod
    def _floor(env, freq):
        from uacpy.io.oalib_writer import at_mesh_floor, at_env_media
        return at_mesh_floor(at_env_media(env), freq)

    _SWEEP = staticmethod(lambda: np.linspace(100.0, 1000.0, 10))

    def _run(self, n_mesh, tmp_path=None):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return Kraken(n_mesh=n_mesh, verbose=False,
                          work_dir=tmp_path,
                          cleanup=tmp_path is None).run(
                self._env(),
                Source(depths=25.0, frequencies=self._SWEEP()),
                Receiver(depths=[50.0], ranges=[1000.0]),
                run_mode=RunMode.BROADBAND)

    def test_the_floor_really_does_climb_across_this_sweep(self):
        # Pin the premise: ~(20·100·f/1500)//2 is 66 at 100 Hz and 666 at
        # 1000 Hz, so n_mesh=200 clears freq0 and not the top of the band.
        # Without that gap the two tests below would agree for trivial
        # reasons.
        env = self._env()
        assert self._floor(env, 100.0) == 66
        assert self._floor(env, 1000.0) == 666

    def test_a_mesh_clearing_freq0_marches_the_whole_sweep(self, tmp_path):
        field = self._run(200, tmp_path)
        assert np.isfinite(np.asarray(field.data)).all()
        deck = sorted(tmp_path.rglob('*.env'))[0].read_text().splitlines()
        assert float(deck[1]) == pytest.approx(100.0), (
            f"deck freq0 is not source.frequencies[0]: {deck[1]!r}")

    def test_a_mesh_below_the_freq0_floor_is_refused(self):
        from uacpy.core.exceptions import ConfigurationError
        too_coarse = self._floor(self._env(), 100.0) - 1
        with pytest.raises(ConfigurationError, match='Mesh is too coarse'):
            self._run(too_coarse)


@pytest.mark.requires_binary
class TestComplexPayloadDtypeIsComplex128:
    """Every uacpy engine returns complex128 pressure; the .shd payload is
    complex64, so the assembly upcasts — including the 1-bin broadband path,
    which used to disagree with the multi-bin one within the same wrapper."""

    @staticmethod
    def _rig():
        env = Environment(
            name='dtype', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.5,
                                      attenuation=0.5))
        return (env, Source(depths=25.0, frequencies=100.0),
                Receiver(depths=np.array([50.0]),
                         ranges=np.array([500.0, 1000.0])))

    def test_coherent_tl_is_complex128(self):
        env, src, rcv = self._rig()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f = Kraken(verbose=False).run(env, src, rcv,
                                          run_mode=RunMode.COHERENT_TL)
        assert np.asarray(f.data).dtype == np.complex128

    @pytest.mark.parametrize('freqs', [[100.0], [95.0, 100.0, 105.0]],
                             ids=['1-bin', '3-bin'])
    def test_broadband_is_complex128_at_any_grid_size(self, freqs):
        env, src, rcv = self._rig()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f = Kraken(verbose=False).run(env, src, rcv,
                                          run_mode=RunMode.BROADBAND,
                                          frequencies=np.array(freqs))
        assert np.asarray(f.data).dtype == np.complex128


@pytest.mark.requires_binary
class TestPrecalcBottomIrcGuard:
    """A 'precalc' bottom stages the user's file verbatim as ``<base>.irc``
    (BOUNCE's Title/freq + NkTab + ``(5G15.7,I5)`` f/g-impedance records,
    ``misc/RefCoef.f90:94-107``). A theta/|R|/phase angle table in that slot
    used to abort the binary with a bare Fortran backtrace at exit 2; the
    header is validated before any launch instead."""

    def test_angle_table_raises_typed_error_before_launch(self, tmp_path):
        table = tmp_path / 'angles.brc'
        table.write_text("3\n0.0 1.0 0.0\n45.0 0.5 0.0\n90.0 0.0 0.0\n")
        env = Environment(
            name='precalc', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='precalc',
                                      reflection_file=str(table)))
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError,
                           match='line 1 is all-numeric') as err:
            Kraken(verbose=False).run(
                env, Source(depths=25.0, frequencies=50.0),
                Receiver(depths=np.array([50.0]), ranges=np.array([1000.0])),
                run_mode=RunMode.COHERENT_TL)
        msg = str(err.value)
        assert '.irc' in msg and '.brc' in msg
        assert "acoustic_type='file'" in msg


class TestFrequencyVectorDefaultsToBroadband:
    """kraken.md §4 "or `BROADBAND` when `frequencies=` has more than one"
    and DOCUMENTATION.md §7 "defaults to a field mode":
    ``run()`` with a multi-element
    ``frequencies=`` kwarg and no ``run_mode`` defaults to BROADBAND — the
    one frequency-vector promotion in the package — while a single-element
    vector leaves the default at COHERENT_TL. Every existing broadband test
    passes ``run_mode`` explicitly, so the default itself was untested.
    Pinned on the resolved settings; no binary runs."""

    _ENV = staticmethod(lambda: Environment(
        name='bb_default', bathymetry=100.0, ssp=1500.0,
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1600.0, density=1.8,
                                  attenuation=0.5)))
    _SRC = staticmethod(lambda: Source(depths=50.0, frequencies=100.0))
    _RCV = staticmethod(lambda: Receiver(depths=np.array([50.0]),
                                         ranges=np.array([1000.0])))

    def test_multi_element_frequencies_dispatch_broadband(self):
        settings = Kraken(verbose=False).run_settings(
            self._ENV(), self._SRC(), self._RCV(),
            frequencies=np.array([95.0, 105.0]))
        assert settings.mode == RunMode.BROADBAND
        assert settings.engine.route == 'band'
        np.testing.assert_array_equal(settings.engine.launches[0].marched_frequencies,
                                      [95.0, 105.0])

    def test_single_element_frequencies_resolve_narrowband_and_are_refused(
            self):
        """A one-element vector keeps the COHERENT_TL default, which takes
        its frequency from the Source and so refuses ``frequencies=``."""
        with pytest.raises(ConfigurationError,
                           match=r'run_mode=COHERENT_TL\) takes its frequency'):
            Kraken(verbose=False).run(
                self._ENV(), self._SRC(), self._RCV(),
                frequencies=np.array([100.0]))


def test_range_dependent_broadband_never_writes_a_broadband_deck(tmp_path,
                                                                 monkeypatch):
    """kraken.md §7 "range-dependent `BROADBAND` run loops":
    ``write_multi_profile_env`` carries ONE frequency, so a range-dependent
    band must be decomposed before any deck is written — never handed to the
    multi-profile writer with a frequency vector, and never silently
    stripped down to one frequency. This traps the writer to prove the
    decomposition happens upstream of it.

    (It replaces a test that pinned the old refusal. The refusal was a deck
    limitation standing in for a physical one: KRAKEN solves modes at one
    frequency whatever the environment, so the band is a loop.)"""
    from uacpy.models.kraken import _launch as kraken_mod
    seen = []
    real_writer = kraken_mod.write_multi_profile_env

    def _spy(*args, **kwargs):
        src = kwargs.get('source')
        seen.append(len(np.atleast_1d(src.frequencies)) if src is not None
                    else 0)
        return real_writer(*args, **kwargs)

    monkeypatch.setattr(kraken_mod, 'write_multi_profile_env', _spy)
    env = Environment(
        name='rd_bb', bathymetry=[(0.0, 100.0), (5000.0, 150.0)],
        ssp=[(0.0, 1500.0), (150.0, 1500.0)],
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1800.0, density=1.8,
                                  attenuation=0.3))
    band = np.array([95.0, 100.0, 105.0])
    out = Kraken(verbose=False, work_dir=tmp_path).run(
        env, Source(depths=50.0, frequencies=band),
        Receiver(depths=[50.0], ranges=[1000.0, 3000.0]),
        run_mode=RunMode.BROADBAND)
    assert list(out.coords) == ['depth', 'range', 'frequency']
    assert np.asarray(out.data).shape[-1] == band.size
    assert seen, 'the multi-profile writer was never reached'
    assert set(seen) == {1}, (
        f"the multi-profile deck was handed {sorted(set(seen))} frequencies; "
        f"it carries exactly one")


class TestRMaxAutoDefaults:
    """kraken.md §5 "`1.05 ×` the outermost receiver range":
    ``rmax_m=None`` resolves to 1.05x the outermost
    receiver range for a narrowband deck and 3x for a broadband sweep — the
    sweep solves every frequency off one Richardson mesh sequence
    (``kraken.f90:80`` exits on ``Error·1000·RMax < 1``), so it gets the
    tighter tolerance as margin. Pinned on the resolved settings
    (``run_settings().engine.launches``), which the deck is written from and
    the result metadata reports. No binary."""

    _SRC = staticmethod(lambda: Source(depths=50.0, frequencies=100.0))
    _RCV = staticmethod(lambda: Receiver(depths=np.array([50.0]),
                                         ranges=np.array([1000.0, 4000.0])))

    def test_compute_rmax_factor_is_pure_arithmetic(self):
        assert _grid.compute_rmax_m(self._RCV()) == pytest.approx(4200.0)
        assert _grid.compute_rmax_m(
            self._RCV(), multiplier=3.0) == pytest.approx(12000.0)

    def _rmax(self, model, **run_kw):
        settings = model.run_settings(_pekeris(), self._SRC(), self._RCV(),
                                      **run_kw)
        return settings.engine.launches[0].rmax_m

    def test_narrowband_deck_gets_1_05x(self):
        assert self._rmax(Kraken(verbose=False)) == pytest.approx(
            1.05 * 4000.0)

    def test_broadband_deck_gets_3x(self):
        assert self._rmax(
            Kraken(verbose=False), run_mode=RunMode.BROADBAND,
            frequencies=np.linspace(80.0, 120.0, 5)) == pytest.approx(
                3.0 * 4000.0)

    def test_one_element_vector_is_not_a_sweep(self):
        # The gate is len(frequencies) > 1, matching the run-mode promotion.
        assert self._rmax(
            Kraken(verbose=False), run_mode=RunMode.BROADBAND,
            frequencies=np.array([100.0])) == pytest.approx(1.05 * 4000.0)

    def test_pinned_rmax_wins_everywhere(self):
        assert self._rmax(
            Kraken(verbose=False, rmax_m=9000.0), run_mode=RunMode.BROADBAND,
            frequencies=np.linspace(80.0, 120.0, 5)) == 9000.0


class TestAutoSegmentationEdges:
    """``models/kraken/_segments.py``: automatic segmentation unions the
    change-point ranges and inserts intermediates so no gap exceeds the 2 km
    ceiling (``_MAX_SEGMENT_LENGTH_M``) — a profile at least every 2 km even
    across a slowly-varying stretch. Pure function; no binary."""

    def test_wedge_with_5km_gaps_splits_at_change_points(self):
        from uacpy.models.kraken._segments import (
            segment_environment_by_range, _MAX_SEGMENT_LENGTH_M)
        env = Environment(
            name='wedge',
            bathymetry=np.array([[0.0, 100.0], [5000.0, 150.0],
                                 [10000.0, 200.0]]),
            ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.8,
                                      attenuation=0.5))
        segments = segment_environment_by_range(env)
        edges = np.array([r for r, _seg in segments], dtype=float)
        # The change points survive verbatim and the axis anchors at r=0.
        assert edges[0] == 0.0
        for change_point in (5000.0, 10000.0):
            assert np.any(np.isclose(edges, change_point)), edges
        # Intermediates cap every gap at the ceiling: each 5 km leg splits
        # into ceil(5000/2000) = 3 sub-segments, so 7 edges in all.
        assert np.all(np.diff(edges) <= _MAX_SEGMENT_LENGTH_M + 1e-9)
        assert edges.size == 7
        # Each segment is a range-independent slice sampled at its own edge,
        # its depth on the 0.1 m quantum its profile ends on.
        for r, seg in segments:
            assert not seg.is_range_dependent
            assert float(seg.bathymetry.eval(range=0.0)) == pytest.approx(
                round(100.0 + r / 100.0, 1))

    def test_a_range_independent_env_is_one_segment(self):
        from uacpy.models.kraken._segments import segment_environment_by_range
        segments = segment_environment_by_range(_pekeris())
        assert len(segments) == 1
        assert segments[0][0] == 0.0


class TestModalCutoffBoundary:
    """kraken.md §7 "Below the modal cutoff there is nothing to sum":
    the docs' 100 m shallow-water channel stops
    supporting a trapped mode between 9 and 10 Hz, which brackets the Pekeris
    estimate ``c_w/(4D·sqrt(1-(c_w/c_b)^2))`` — 8.7 Hz on the 1490 m/s speed at
    the bottom of the column, 9.0 Hz on the 1500 m/s at the top.

    The default ``c_high`` sits 5 % past the bottom speed, so between 7.5 and
    10 Hz the solver does find a root; every one of those modes has a phase
    speed above the 1650 m/s seabed (1699.66 m/s at 8 Hz, 1660.36 at 9), which
    is the continuous spectrum wearing a mode's clothes.
    ``compute_modes`` refuses all three bands, by two different routes: below
    ~7.5 Hz kraken.exe itself finds nothing, above it uacpy rejects a mode set
    that is entirely non-trapped."""

    @staticmethod
    def _doc_channel():
        return Environment(
            name='doc-channel', bathymetry=100.0,
            ssp=[(0.0, 1500.0), (30.0, 1495.0), (100.0, 1490.0)],
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1650.0, density=1.8,
                                      attenuation=0.6))

    def test_7_5_hz_is_below_the_cutoff(self):
        from uacpy.core.exceptions import ModelExecutionError
        # At 7.5 Hz this deck makes kraken.exe abort with a raw Fortran
        # RECL error while writing the empty modes.mod, which uacpy wraps
        # as a typed ModelExecutionError but without the friendlier 'no
        # mode' diagnosis that the clean zero-mode path produces.
        with pytest.raises(ModelExecutionError,
                           match='found no mode with a phase speed inside'):
            Kraken(verbose=False).compute_modes(
                self._doc_channel(), Source(depths=25.0, frequencies=7.5))

    def test_8_hz_finds_a_root_but_not_a_trapped_one(self):
        from uacpy.core.exceptions import ModelExecutionError
        # The 5 % c_high pad puts the search ceiling at 1732.5 m/s, so
        # kraken.exe returns one mode at cp = 1699.66 m/s — above the 1650 m/s
        # seabed, i.e. radiating into it. Before round 24 this counted as
        # "above the cutoff" and the caller got a mode set to propagate.
        with pytest.raises(ModelExecutionError, match='non-trapped'):
            Kraken(verbose=False).compute_modes(
                self._doc_channel(), Source(depths=25.0, frequencies=8.0))

    def test_10_hz_is_above_the_cutoff(self):
        modes = Kraken(verbose=False).compute_modes(
            self._doc_channel(), Source(depths=25.0, frequencies=10.0))
        assert modes.n_modes >= 1


# ─── Guards and cuts must describe the deck that actually runs ─────────────


def _rd_layered_bottom(shear_at_r0, shear_elsewhere):
    """A range-dependent layered bottom whose r = 0 column and median column
    differ in shear, over a fluid half-space."""
    from uacpy.core.environment import Bottom, SeabedColumn, SedimentLayer

    def col(shear):
        return SeabedColumn(
            layers=[SedimentLayer(thickness=10.0, sound_speed=1700.0,
                                  density=1.6, attenuation=0.3,
                                  shear_speed=shear,
                                  shear_attenuation=1.0 if shear else 0.0)],
            halfspace=BoundaryProperties(acoustic_type='half-space',
                                         sound_speed=1900.0, density=2.0,
                                         attenuation=0.1))
    return Bottom.from_columns(
        [col(shear_at_r0), col(shear_elsewhere), col(shear_elsewhere)],
        ranges=np.array([0.0, 5000.0, 10000.0]))


class TestElasticGuardTestsTheColumnTheDeckCarries:
    """``_checks.modes_single_profile`` samples the r = 0 profile of every
    range-dependent quantity, while the field paths segment the bottom and
    write every column into its own profile block. So MODES tests the r = 0
    column alone, and a field run tests every column: an elastic-over-fluid
    column anywhere reaches a profile block and would hang krakenc.exe."""

    _SRC = staticmethod(lambda: Source(depths=[50.0], frequencies=[100.0]))

    @staticmethod
    def _env(bottom):
        return Environment(name='rdguard', bathymetry=100.0, ssp=1500.0,
                           bottom=bottom)

    @pytest.mark.parametrize('field_mode', [
        RunMode.COHERENT_TL, RunMode.INCOHERENT_TL, RunMode.BROADBAND,
        RunMode.TIME_SERIES])
    def test_an_elastic_column_at_r0_is_refused_everywhere(self, field_mode):
        from uacpy.core.exceptions import UnsupportedFeatureError
        env = self._env(_rd_layered_bottom(shear_at_r0=400.0,
                                           shear_elsewhere=0.0))
        model = Kraken(verbose=False)
        with pytest.raises(
                UnsupportedFeatureError,
                match='elastic sediment layer over a fluid halfspace'):
            _checks.reject_acoustic_below_elastic(env, RunMode.MODES,
                                                  model_name=model.model_name)
        with pytest.raises(
                UnsupportedFeatureError,
                match='elastic sediment layer over a fluid halfspace'):
            _checks.reject_acoustic_below_elastic(env, field_mode,
                                                  model_name=model.model_name)

    def test_an_elastic_column_past_r0_is_refused_on_the_field_path_only(self):
        from uacpy.core.exceptions import UnsupportedFeatureError
        env = self._env(_rd_layered_bottom(shear_at_r0=0.0,
                                           shear_elsewhere=400.0))
        model = Kraken(verbose=False)
        # MODES never writes the r > 0 columns.
        _checks.reject_acoustic_below_elastic(env, RunMode.MODES,
                                              model_name=model.model_name)
        with pytest.raises(
                UnsupportedFeatureError,
                match='elastic sediment layer over a fluid halfspace'):
            _checks.reject_acoustic_below_elastic(env, RunMode.COHERENT_TL,
                                                  model_name=model.model_name)

    def test_the_deck_columns_are_r0_for_modes_and_all_for_a_field_run(self):
        env = self._env(_rd_layered_bottom(shear_at_r0=0.0,
                                           shear_elsewhere=0.0))
        assert len(_checks.deck_bottom_columns(env, RunMode.MODES)) == 1
        for mode in (RunMode.COHERENT_TL, RunMode.INCOHERENT_TL,
                     RunMode.BROADBAND, RunMode.TIME_SERIES):
            assert len(_checks.deck_bottom_columns(env, mode)) == 3


def test_incoherent_tl_on_krakenc_is_quiet_for_a_multi_profile_run():
    """``field.f90:214-215`` picks the evaluator by profile count. The
    multi-profile adiabatic one, ``EvaluateADMod.f90:110``, computes
    ``SQRT(SUM(ABS(...)**2))`` — a strict energy sum on either backend — so
    the single-profile ``EvaluateMod.f90:66`` caveat does not apply and the
    warning must not fire. (Multi-profile *coupled* never reaches the
    incoherent branch: field.f90:125-129 refuses that pairing and ``run``
    rejects it up front.)"""
    from uacpy.core.environment import Bathymetry
    env = Environment(
        name='inc_rd', ssp=1500.0,
        bathymetry=Bathymetry(ranges=np.array([0.0, 2000.0, 4000.0]),
                              depths=np.array([100.0, 110.0, 120.0])),
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1800.0, density=1.8,
                                  attenuation=0.3))
    with recorded_warnings() as caught:
        field = Kraken(verbose=False, backend='krakenc',
                       mode_coupling='adiabatic').run(
            env, Source(depths=50.0, frequencies=100.0),
            Receiver(depths=np.array([50.0]),
                     ranges=np.array([1000.0, 2000.0])),
            run_mode=RunMode.INCOHERENT_TL)
    assert field.metadata['n_profiles'] > 1
    assert not [w for w in caught
                if 'strict incoherent sum' in str(w.message)]


class TestOnlyElasticMediaAreMaskedOut:
    """``ReadModes.f90:296-331`` compacts a mode's stress-displacement vector
    to one value per depth, copying ACOUSTIC media verbatim and leaving an
    ELASTIC medium's points unwritten for the ``Comp`` values field.exe lets
    uacpy request. The read index still advances past the elastic block, so
    every acoustic medium is correct — above *and* below it. Masking the
    whole sub-bottom therefore discarded correct values in fluid sediment
    layers."""

    @staticmethod
    def _env():
        from uacpy.core.environment import SeabedColumn, SedimentLayer
        return Environment(
            name='fluid-over-elastic', bathymetry=50.0, ssp=1500.0,
            bottom=SeabedColumn(
                layers=[
                    SedimentLayer(thickness=10.0, sound_speed=1600.0,
                                  density=1.6, attenuation=0.2),
                    SedimentLayer(thickness=15.0, sound_speed=1800.0,
                                  density=1.9, attenuation=0.3,
                                  shear_speed=400.0, shear_attenuation=1.0),
                ],
                halfspace=BoundaryProperties(
                    acoustic_type='half-space', sound_speed=2200.0,
                    density=2.2, attenuation=0.4, shear_speed=600.0)))

    def test_spans_cover_the_elastic_medium_and_everything_under_it(self):
        env = self._env()
        spans = _checks.elastic_depth_intervals(
            env, env.bottom.at(range=0.0))
        # Water 0-50, fluid layer 50-60, elastic layer 60-75, then the
        # half-space — which reads the elastic medium's last sample.
        assert spans == [(60.0, float('inf'))]

    def test_a_fluid_sediment_layer_is_evaluated(self):
        env = self._env()
        depths = np.array([25.0, 55.0, 65.0, 90.0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tl = np.asarray(Kraken(verbose=False).run(
                env, Source(depths=25.0, frequencies=75.0),
                Receiver(depths=depths,
                         ranges=np.array([500.0, 1000.0, 2000.0]))).dB)
        assert np.isfinite(tl[0]).all(), "water column"
        assert np.isfinite(tl[1]).all(), "fluid sediment layer 50-60 m"
        assert not np.isfinite(tl[2]).any(), "elastic layer 60-75 m"
        assert not np.isfinite(tl[3]).any(), "below the elastic medium"
        # The fluid-layer column has to be the physical field, not whatever
        # the packed mode vector happened to hold: an all-fluid stack of the
        # same geometry agrees to within a few dB.
        from uacpy.core.environment import SeabedColumn, SedimentLayer
        fluid = Environment(
            name='all-fluid', bathymetry=50.0, ssp=1500.0,
            bottom=SeabedColumn(
                layers=[
                    SedimentLayer(thickness=10.0, sound_speed=1600.0,
                                  density=1.6, attenuation=0.2),
                    SedimentLayer(thickness=15.0, sound_speed=1800.0,
                                  density=1.9, attenuation=0.3),
                ],
                halfspace=BoundaryProperties(
                    acoustic_type='half-space', sound_speed=2200.0,
                    density=2.2, attenuation=0.4)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ref = np.asarray(Kraken(verbose=False).run(
                fluid, Source(depths=25.0, frequencies=75.0),
                Receiver(depths=depths,
                         ranges=np.array([500.0, 1000.0, 2000.0]))).dB)
        assert np.max(np.abs(tl[1] - ref[1])) < 5.0, (tl[1], ref[1])

    def test_a_depth_at_the_elastic_top_interface_is_kept(self):
        """An interface depth is tabulated once per adjoining medium and
        ``calculateweights.f90:43-49`` brackets it on the upper copy, so the
        top of an elastic medium reads the acoustic sample above it."""
        env = self._env()
        model = Kraken(verbose=False)
        spans = _checks.elastic_depth_intervals(env, env.bottom.at(range=0.0))
        rcv = Receiver(depths=np.array([60.0, 60.5]), ranges=[1000.0])
        _, keep, notice = _checks.partition_elastic_subbottom(
            env, rcv, spans, model_name=model.model_name)
        assert 'elastic sub-bottom' in notice[1]
        # the refusal the message points to is the function of that name
        assert 'see ``reject_acoustic_below_elastic``' in notice[1]
        assert callable(_checks.reject_acoustic_below_elastic)
        np.testing.assert_array_equal(keep, [True, False])


class TestNonTrappedModes:
    """Above the half-space speed a mode radiates into the seabed instead of
    propagating in the duct. Real-arithmetic kraken.exe cannot even represent
    that: ``Kraken/BCImpedanceMod.f90:83-89`` builds
    ``gammaP = SQRT(x - omega2/cP**2)``, whose radicand goes negative there, and
    ``DBLE()`` keeps the real part of a pure-imaginary ``gammaP`` as 0 — leaving
    ``f=0, g=rho``, which is the rigid-bottom CASE 'R' at ``:60-63``. So the
    eigenvalues come back from a *rigid* waveguide plus the radiation-loss
    perturbation of ``kraken.f90:766-773``: a 5 m / 150 Hz duct returns
    cp = 1730.70 m/s against the rigid-bottom prediction 1732.05."""

    _FREQ = 150.0   # exact first cutoff of the 5 m duct is 159.375 Hz

    @pytest.mark.requires_binary
    def test_all_non_trapped_modes_raise_instead_of_answering(self):
        with pytest.raises(ModelExecutionError,
                           match='every one of them is non-trapped') as exc:
            Kraken(verbose=False).compute_modes(
                _duct(5.0), Source(depths=2.5, frequencies=self._FREQ))
        msg = str(exc.value)
        assert 'non-trapped' in msg
        assert 'continuous spectrum' in msg and 'Scooter' in msg

    @pytest.mark.requires_binary
    def test_a_field_run_below_cutoff_warns_before_field_exe_sums(self):
        """``run()`` returns a complete, plausible TL curve there — the modal
        sum is the only place the problem is visible, so the eigenvalues are
        read back before field.exe consumes them. A warning rather than a
        raise: the broadband sub-cutoff recovery legitimately drives single
        below-cutoff bins through this path and zero-fills them."""
        rcv = uacpy.Receiver(depths=np.array([2.5]),
                             ranges=np.array([150.0, 950.0]))
        with pytest.warns(UserWarning, match='every one of them is '
                                             'non-trapped'):
            Kraken(verbose=False).compute_tl(
                _duct(5.0), Source(depths=2.5, frequencies=self._FREQ), rcv)

    def test_some_non_trapped_modes_are_logged_not_raised(self, capsys):
        """The ordinary shallow-water default: 14 modes at 200 Hz in 100 m of
        water, 3 of them above the 1650 m/s seabed (docs/models/kraken.md:
        316-318). Documented behaviour, so it must not raise — but it must not
        be invisible either."""
        model = Kraken(verbose='info')
        env = _duct(100.0, c_bottom=1650.0)
        k = _k_for([1500.0, 1600.0, 1700.0], 200.0)
        _modes.check_non_trapped_modes(k, env, 200.0,
                                       leaky_modes=model.leaky_modes,
                                       log=model._log,
                                       model_name=model.model_name)
        out = capsys.readouterr().out
        assert '1 of 3 modes are non-trapped' in out
        assert '1650.0 m/s' in out

    def test_every_mode_non_trapped_raises_off_the_mode_set_alone(self):
        model = Kraken(verbose=False)
        with pytest.raises(ModelExecutionError, match='non-trapped'):
            _modes.check_non_trapped_modes(
                _k_for([1710.0, 1750.0], 150.0), _duct(5.0), 150.0,
                leaky_modes=model.leaky_modes, log=model._log,
                model_name=model.model_name)

    def test_trapped_modes_say_nothing(self, capsys):
        model = Kraken(verbose='info')
        _modes.check_non_trapped_modes(
            _k_for([1500.0, 1650.0], 150.0), _duct(5.0), 150.0,
            leaky_modes=model.leaky_modes, log=model._log,
            model_name=model.model_name)
        assert 'non-trapped' not in capsys.readouterr().out

    def test_leaky_modes_opt_in_passes_through_silently(self):
        """``leaky_modes=True`` asks for exactly these modes."""
        model = Kraken(leaky_modes=True, verbose=False)
        _modes.check_non_trapped_modes(
            _k_for([1710.0, 1750.0], 150.0), _duct(5.0), 150.0,
            leaky_modes=model.leaky_modes, log=model._log,
            model_name=model.model_name)

    def test_a_boundary_with_no_half_space_has_nothing_to_leak_into(self):
        """vacuum / rigid / reflection-table bottoms carry a placeholder
        sound_speed and resolve to an unbounded c_high, so comparing against
        it would be meaningless."""
        model = Kraken(verbose=False)
        env = Environment(
            name='rigid', bathymetry=5.0, ssp=[(0.0, C_WATER), (5.0, C_WATER)],
            bottom=BoundaryProperties(acoustic_type='rigid'))
        _modes.check_non_trapped_modes(_k_for([9000.0], 150.0), env, 150.0,
                                       leaky_modes=model.leaky_modes,
                                       log=model._log,
                                       model_name=model.model_name)

    def test_an_elastic_half_space_traps_on_its_shear_speed_instead(self):
        """``kraken.f90:209`` clamps cHigh to cS there, so the compressional
        speed is the wrong threshold and this check stands down."""
        model = Kraken(verbose=False)
        env = _duct(5.0, shear_speed=800.0, shear_attenuation=0.5)
        _modes.check_non_trapped_modes(_k_for([1710.0], 150.0), env, 150.0,
                                       leaky_modes=model.leaky_modes,
                                       log=model._log,
                                       model_name=model.model_name)


class TestTheDispatchPremiseMatchesTheVendoredSource:
    """``leaky_modes``' docstring justified forcing krakenc by quoting
    kraken.htm's claim that KRAKEN reduces CHIGH to keep only trapped modes.
    The vendored Fortran has that clamp commented out for the acoustic case,
    so the package stated a premise its own bundled source contradicts."""

    @staticmethod
    def _kraken_f90():
        import uacpy
        return (Path(uacpy.__file__).parent / 'third_party' /
                'Acoustics-Toolbox' / 'Kraken' / 'kraken.f90')

    def test_the_acoustic_c_high_clamp_is_commented_out_upstream(self):
        lines = self._kraken_f90().read_text().splitlines()
        elastic, acoustic = lines[208].strip(), lines[211].strip()
        assert elastic == 'cHigh = MIN( cHigh, DBLE( HSBot%cS ) )'
        assert acoustic == '! cHigh = MIN( cHigh, DBLE( HSBot%cP ) )'

    def test_the_docstring_cites_the_commented_out_line(self):
        doc = Kraken.__doc__
        assert 'kraken.f90:212' in doc
        assert '! cHigh = MIN( cHigh, DBLE( HSBot%cP ) )' in doc
        assert 'does **not** mean "no leaky modes"' in doc


@pytest.mark.requires_binary
class TestKrakenBroadbandStampsThePhysicalCMax:
    """Kraken writes the same stamp on its complex-spectrum results
    (``models/kraken/``, the broadband and ``return_pressure`` branches). Resolving
    the right speed is not the same contract as writing it onto the field,
    and only the write reaches ``to_time_trace``."""

    def test_the_stamp_is_the_seabed_speed_and_anchors_the_window(self):
        env = Environment(name='cmax_bb', bathymetry=100.0, ssp=1500.0,
                          bottom=make_halfspace(3000.0, density=2.0,
                                            attenuation=0.1))
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=np.array([50.0]), ranges=np.array([2000.0]))
        result = Kraken(verbose=False).run(
            env, src, rcv, run_mode=RunMode.BROADBAND,
            frequencies=np.linspace(80.0, 120.0, 5))

        assert result.run_settings.waveguide.c_max == pytest.approx(3000.0)
        assert 'c_max' not in result.metadata

        trace = result.to_time_trace(depth=50.0, range=2000.0)
        t = np.asarray(trace.coords['time'], dtype=float)
        assert t[0] == pytest.approx(2000.0 / 3000.0 - 0.05, abs=0.02)


def _layered(*layers, halfspace_shear=0.0):
    column = SeabedColumn(
        layers=list(layers),
        halfspace=BoundaryProperties(
            acoustic_type='half-space', sound_speed=2000.0, density=2.2,
            attenuation=0.1, shear_speed=halfspace_shear,
            shear_attenuation=1.0 if halfspace_shear else 0.0))
    return Environment(
        name='stack', bathymetry=100.0, ssp=1500.0,
        bottom=Bottom.from_columns([column], ranges=np.array([0.0])))


def _layer(shear=0.0, roughness=0.0, thickness=10.0, speed=1700.0):
    return SedimentLayer(
        thickness=thickness, sound_speed=speed, density=1.6,
        attenuation=0.3, shear_speed=shear,
        shear_attenuation=1.0 if shear else 0.0, roughness=roughness)


class TestKrakenRejectsAcousticBelowElastic:
    """``kraken.f90:170`` sets ``LastAcoustic`` to the deepest acoustic
    medium, so a fluid layer under an elastic one leaves
    ``FirstAcoustic..LastAcoustic`` spanning the elastic medium and
    ``Vector``'s loops walk it as acoustic: krakenc.exe aborts with SIGABRT
    ("double free or corruption"). The guard used to test the half-space
    alone and let this stack through."""

    def test_fluid_layer_below_an_elastic_one_is_refused(self):
        env = _layered(_layer(shear=400.0), _layer(shear=0.0, speed=1800.0),
                       halfspace_shear=600.0)
        with pytest.raises(UnsupportedFeatureError,
                           match='below an elastic one'):
            _checks.reject_acoustic_below_elastic(
                env, RunMode.COHERENT_TL, model_name='Kraken')

    def test_elastic_over_elastic_is_allowed(self):
        env = _layered(_layer(shear=400.0), _layer(shear=500.0),
                       halfspace_shear=600.0)
        _checks.reject_acoustic_below_elastic(
            env, RunMode.COHERENT_TL, model_name='Kraken')

    def test_a_fluid_layer_above_an_elastic_one_is_allowed(self):
        env = _layered(_layer(shear=0.0), _layer(shear=400.0),
                       halfspace_shear=600.0)
        _checks.reject_acoustic_below_elastic(
            env, RunMode.COHERENT_TL, model_name='Kraken')

    def test_the_fluid_halfspace_case_is_refused(self):
        env = _layered(_layer(shear=400.0), halfspace_shear=0.0)
        with pytest.raises(UnsupportedFeatureError, match='fluid halfspace'):
            _checks.reject_acoustic_below_elastic(
                env, RunMode.COHERENT_TL, model_name='Kraken')


class TestKrakenRoughElasticInterface:
    """``kraken.f90:178`` / ``krakenc.f90:182`` stop with 'Rough elastic
    interfaces are not allowed' for any elastic medium whose ``SSP%sigma`` is
    non-zero, and the writer takes that sigma from the layer's own
    ``roughness``. A rough elastic *half-space* is fine — its sigma sits on
    the BotOpt line and feeds KupIng."""

    def test_roughness_on_an_elastic_layer_is_refused(self):
        env = _layered(_layer(shear=300.0, roughness=0.5),
                       halfspace_shear=600.0)
        with pytest.raises(UnsupportedFeatureError, match='Rough elastic'):
            _checks.reject_rough_elastic_layer(
                env, RunMode.COHERENT_TL, model_name='Kraken')

    def test_roughness_on_a_fluid_layer_is_allowed(self):
        env = _layered(_layer(shear=0.0, roughness=0.5))
        _checks.reject_rough_elastic_layer(
            env, RunMode.COHERENT_TL, model_name='Kraken')

    def test_a_smooth_elastic_layer_is_allowed(self):
        env = _layered(_layer(shear=300.0, roughness=0.0),
                       halfspace_shear=600.0)
        _checks.reject_rough_elastic_layer(
            env, RunMode.COHERENT_TL, model_name='Kraken')

    @pytest.mark.parametrize('stem,line', [('kraken', 178), ('krakenc', 182)])
    def test_the_cited_line_is_the_stop_and_not_its_neighbour(self, stem, line):
        """The address in the docstring above and in the raised message has
        to name the ERROUT itself.

        ``kraken.f90:169`` was cited for years and is
        ``IF ( FirstAcoustic == 0 ) FirstAcoustic = Medium`` — a line about
        which medium is first, nine lines from the stop and unrelated to it.
        The packaging citation walk cannot see that: it checks a cited line
        resolves and is not blank, and :169 is both.
        """
        src = (Path(uacpy.__file__).parent / 'third_party'
               / 'Acoustics-Toolbox' / 'Kraken' / f'{stem}.f90')
        lines = src.read_text(errors='replace').splitlines()
        assert 'Rough elastic interfaces are not allowed' in lines[line - 1], (
            line, lines[line - 1])
        assert "ERROUT" in lines[line - 1]
        # The single line the whole tree cites for this stop.
        hits = [i + 1 for i, ln in enumerate(lines)
                if 'Rough elastic interfaces are not allowed' in ln]
        assert hits == [line], hits


class TestKrakenConstructorBounds:
    """``kraken.f90:80`` leaves the mesh-refinement loop once
    ``Error*1000*RMax < 1`` and ``Error`` starts at 1e10, so ``rmax_m <= 0``
    satisfies it on the coarsest mesh (measured 2.37 dB max |dTL| against the
    default). ``c_high`` was only ever compared against an explicit
    ``c_low``, so ``Kraken(c_high=-100)`` constructed and died in the
    Fortran."""

    @pytest.mark.parametrize('rmax', [0.0, -50.0])
    def test_non_positive_rmax_is_refused(self, rmax):
        with pytest.raises(ConfigurationError, match='rmax_m'):
            Kraken(rmax_m=rmax)

    def test_a_positive_rmax_is_kept(self):
        assert Kraken(rmax_m=1000.0).rmax_m == 1000.0

    @pytest.mark.parametrize('c_high', [0.0, -100.0])
    def test_non_positive_c_high_is_refused_without_a_c_low(self, c_high):
        with pytest.raises(ConfigurationError, match='c_high'):
            Kraken(c_high=c_high)

    def test_an_ordered_pair_is_accepted(self):
        model = Kraken(c_low=1400.0, c_high=1800.0)
        assert (model.c_low, model.c_high) == (1400.0, 1800.0)


def test_an_elastic_layers_shear_speed_densifies_the_mode_grid():
    """A layer's shear speed enters the mode-grid minimum, so an elastic
    seabed gets a denser tabulation grid than the same seabed without shear.

    NOT because anything is sampled inside the elastic medium — see
    :func:`test_the_mode_grid_never_reaches_inside_an_elastic_medium` for
    why it cannot be — but because ``c_s`` is the slowest speed in the
    problem and the density it sets applies to the water column, which is
    where the grid lives."""
    model = Kraken(verbose=False)
    elastic = _layered(_layer(shear=300.0), halfspace_shear=600.0)
    fluid = _layered(_layer(shear=0.0))
    ppm_elastic = _grid.mode_points_per_meter(
        elastic, [100.0],
        pinned_mode_points_per_meter=model.mode_points_per_meter)[0]
    ppm_fluid = _grid.mode_points_per_meter(
        fluid, [100.0],
        pinned_mode_points_per_meter=model.mode_points_per_meter)[0]
    assert ppm_elastic > ppm_fluid
    # 10 points per shear wavelength at 100 Hz over c_s = 300 m/s.
    assert ppm_elastic == pytest.approx(10.0 * 100.0 / 300.0, rel=1e-9)


def test_the_mode_grid_never_reaches_inside_an_elastic_medium():
    """The reason the docstring above used to give was wrong, and the
    vendored source is where that shows.

    ``kraken.f90:266`` sizes the tabulation vector over
    ``FirstAcoustic : LastAcoustic`` and ``kraken.f90:560-565`` lays the
    depths out over that same span, so the grid stops at the last ACOUSTIC
    medium and no sample is ever placed in an elastic layer. Any rationale
    phrased as "resolving the mode shape inside the elastic sediment" is
    describing something the solver does not do."""
    lines = (Path(uacpy.__file__).parent / 'third_party'
             / 'Acoustics-Toolbox' / 'Kraken'
             / 'kraken.f90').read_text(errors='replace').splitlines()
    assert 'NTotal  = SUM( N( FirstAcoustic : LastAcoustic ) )' in lines[265]
    assert 'DO Medium = FirstAcoustic, LastAcoustic' in lines[559]
    # The line that writes the depth coordinates, inside that acoustic loop.
    assert 'z( j + 1 : j + N( Medium ) )' in lines[564], lines[564]


@pytest.mark.requires_binary
def test_the_shear_term_changes_the_field_it_is_kept_for():
    """The term stays because it is not value-neutral, so this measures the
    thing that justifies keeping it rather than the code path that reaches
    it. A 20 m elastic layer at 200 Hz: 5.0 pts/m with the shear speed in
    the minimum against the 1.5 pts/m floor without it.

    Measured on a **coupled-mode** run, which is where the tabulation
    density still reaches the field. Since the caller's receiver depths
    joined the tabulation grid (``_write_field_env``), a range-independent
    or adiabatic field reads its mode shapes straight off a tabulated point
    instead of interpolating between two, so the density no longer moves it
    (measured 2.1e-7 dB / 0.0 dB across the same pair). ``EvaluateCM``
    still forms its coupling integrals from the tabulated shapes at the
    profile interfaces, so the density bites there — 2.6e-2 dB."""
    bottom = Bottom([SeabedColumn(
        layers=[SedimentLayer(thickness=20.0, sound_speed=1800.0,
                              shear_speed=400.0, density=1.8,
                              attenuation=0.2)],
        halfspace=BoundaryProperties(acoustic_type='half-space',
                                     sound_speed=2000.0, shear_speed=600.0,
                                     density=2.0, attenuation=0.5))])
    env = Environment(name='elastic-grid', bathymetry=100.0, ssp=1500.0,
                      bottom=bottom)
    # Range-dependent, so the field goes through EvaluateCM's coupling
    # integrals — see the docstring.
    rd_env = Environment(name='elastic-grid-rd',
                         bathymetry=[(0.0, 100.0), (3000.0, 130.0)],
                         ssp=1500.0, bottom=bottom)
    src = Source(depths=50.0, frequencies=200.0)
    rcv = Receiver(depths=np.array([30.0, 60.0]),
                   ranges=np.linspace(500.0, 3000.0, 6))
    from uacpy.models.kraken._grid import MODE_POINTS_PER_METER_FLOOR
    model = Kraken(verbose=False)
    with_shear = _grid.mode_points_per_meter(
        env, 200.0, pinned_mode_points_per_meter=model.mode_points_per_meter)[0]
    assert with_shear == pytest.approx(10.0 * 200.0 / 400.0, rel=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fields = [
            np.asarray(Kraken(verbose=False, mode_coupling='coupled',
                              mode_points_per_meter=ppm)
                       .run(rd_env, src, rcv).dB, dtype=float)
            for ppm in (with_shear, MODE_POINTS_PER_METER_FLOOR)
        ]
    diff = np.abs(fields[0] - fields[1])
    assert np.nanmax(diff) > 0.005, np.nanmax(diff)   # not value-neutral
    assert np.nanmax(diff) < 1.0, np.nanmax(diff)     # nor a large effect


def test_kraken_mode_cutoff_probe_reraises_a_real_failure(monkeypatch,
                                                          tmp_path):
    """``_count_modes_at_freq`` returning 0 makes the caller report "every
    frequency is below the waveguide's modal cutoff" with the remediation
    "raise the frequency band" — so a missing binary, a crash or a disk error
    must not be turned into a zero."""
    model = Kraken(verbose=False)
    env = _rigid_env()
    source, receiver = _point(freq=100.0)

    def _boom(*args, **kwargs):
        raise ModelExecutionError('Kraken', return_code=-6, stdout=None,
                                  stderr='the binary crashed')

    inputs = _band_stage_inputs(model, env, source, receiver, tmp_path,
                                np.array([90.0, 100.0]))
    monkeypatch.setattr(Kraken, '_run_kraken_executable', _boom)
    with pytest.raises(ModelExecutionError, match='crashed'):
        model._count_modes_at_freq(inputs, 100.0)


@pytest.mark.requires_binary
@pytest.mark.slow
class TestKrakenModesBelowAnElasticSeafloor:
    """``kraken.f90:558-568`` tabulates the eigenvector over the ACOUSTIC media
    only, and ``calculateweights.f90`` extrapolates past the last node, so the
    samples ``compute_modes`` returned below an elastic seafloor were a
    straight line: measured increments constant to 3e-8 on a -2.3e-3 step,
    against a 1.8e-3 spread for a fluid-layer control."""

    @staticmethod
    def _modes(shear):
        env = _layered(_layer(shear=shear),
                       halfspace_shear=600.0 if shear else 0.0)
        with recorded_warnings() as caught:
            modes = Kraken(verbose=False).compute_modes(
                env, Source(depths=[50.0], frequencies=[50.0]))
        return modes, [str(w.message) for w in caught]

    def test_elastic_sub_bottom_samples_are_marked_no_data(self):
        modes, messages = self._modes(300.0)
        z = np.asarray(modes.depths, dtype=float)
        phi = np.asarray(modes.phi)
        below = z > 100.0
        assert below.any(), "the mode grid must span the sediment"
        assert not np.isfinite(phi[below, :]).any()
        assert np.isfinite(phi[~below, :]).any()
        assert any('elastic sub-bottom' in m for m in messages)

    def test_a_fluid_sediment_is_left_alone(self):
        modes, messages = self._modes(0.0)
        z = np.asarray(modes.depths, dtype=float)
        phi = np.asarray(modes.phi)
        below = z > 100.0
        assert np.isfinite(phi[below, :]).all()
        assert not any('elastic sub-bottom' in m for m in messages)


def _rigid_env(depth=100.0, speed=1500.0):
    return Environment(name='rigid', bathymetry=depth, ssp=speed,
                       bottom=BoundaryProperties(acoustic_type='rigid'))


def _point(depth=50.0, freq=30.0, r=1000.0):
    return (Source(depths=depth, frequencies=freq),
            Receiver(depths=np.array([depth]), ranges=np.array([r])))


class TestKrakenSegmentsOnWavelengthsNotMetres:
    """``EvaluateADMod.f90:47-51,75`` interpolates k and phi linearly between
    profiles and ``EvaluateCMMod.f90:262-305`` projects the coupling matrix at
    each boundary; neither that Fortran nor ``field.f90:122-134`` tests the
    profile spacing, so an under-segmented track exits 0 with a clean .prt.
    What the interpolation has to follow is the change in the waveguide
    measured in wavelengths, so a fixed metre ceiling cannot bound the error:
    on a 200->100 m wedge over 10 km against a converged 161-profile run, the
    2 km ceiling left 18.18 dB max / 2.84 mean at 100 Hz and 15.72 / 4.25 at
    300 Hz. Subdividing until each segment spans under a quarter wavelength of
    depth change gives 1.74 / 0.16 and 0.64 / 0.06.
    """

    @staticmethod
    def _wedge(d0=200.0, d1=100.0, r_max=10000.0):
        from uacpy.core import BoundaryProperties, Environment
        from uacpy.core.bathymetry import Bathymetry
        return Environment(
            bathymetry=Bathymetry(ranges=[0.0, r_max], depths=[d0, d1]),
            ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.5))

    def test_a_higher_frequency_asks_for_more_profiles(self):
        from uacpy.models.kraken._segments import segment_environment_by_range
        wedge = self._wedge()
        n_100 = len(segment_environment_by_range(wedge, freq=100.0))
        n_300 = len(segment_environment_by_range(wedge, freq=300.0))
        assert n_300 > n_100 > 6, (
            f"{n_100} profiles at 100 Hz and {n_300} at 300 Hz: the count has "
            f"to follow the wavelength, not just the 2 km ceiling")

    def test_each_segment_spans_under_a_quarter_wavelength_of_depth(self):
        import numpy as np
        from uacpy.models.kraken._segments import (
            _SEGMENT_DEPTH_STEP_PER_WAVELENGTH, segment_environment_by_range)
        wedge = self._wedge()
        freq = 300.0
        target = _SEGMENT_DEPTH_STEP_PER_WAVELENGTH * 1500.0 / freq
        edges = [r for r, _ in segment_environment_by_range(wedge, freq=freq)]
        depths = [float(np.asarray(wedge.bathymetry.eval(range=r)).flat[0])
                  for r in edges]
        worst = max(abs(b - a) for a, b in zip(depths, depths[1:]))
        assert worst <= target * 1.01

    def test_a_near_flat_track_is_not_subdivided_on_depth(self):
        # The criterion is slope-aware: 5 m of drop over 20 km needs no extra
        # profiles beyond the metre ceiling, so a gentle track pays nothing.
        from uacpy.models.kraken._segments import segment_environment_by_range
        flat = self._wedge(d0=200.0, d1=195.0, r_max=20000.0)
        assert len(segment_environment_by_range(flat, freq=300.0)) <= 12

    def test_no_frequency_falls_back_to_the_metre_ceiling(self):
        from uacpy.models.kraken._segments import segment_environment_by_range
        wedge = self._wedge()
        assert len(segment_environment_by_range(wedge, freq=None)) == 6


class TestSegmentationSeesTheProfileNotJustTheSeafloor:
    """The depth criterion is blind to a waveguide whose PROFILE moves while
    its depth does not, and adiabatic mode interpolation is just as sensitive
    to that. On a flat 200 m bottom whose column relaxes from a thermocline to
    isovelocity over 20 km the depth rule returned 11 profiles at EVERY
    frequency, and against a converged 201-profile run the 800 Hz field was
    2.32 dB rms / 6.41 dB max out while 100 Hz was 0.08 dB — error scaling with
    frequency at a fixed decomposition, the signature of a criterion that is
    simply not being applied. With the profile-change rule the same case takes
    44 profiles at 800 Hz for 0.14 dB rms.
    """

    @staticmethod
    def _ssp_driven():
        from uacpy.core import BoundaryProperties, Environment
        from uacpy.core.ssp import SoundSpeedProfile
        return Environment(
            bathymetry=200.0,
            ssp=SoundSpeedProfile(
                depths=[0, 50, 100, 150, 200],
                sound_speed=[[1540, 1500], [1520, 1500], [1500, 1500],
                      [1495, 1500], [1493, 1500]],
                ranges=[0.0, 20000.0]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.7,
                                      attenuation=0.5))

    def test_a_range_dependent_profile_asks_for_more_at_higher_frequency(self):
        from uacpy.models.kraken._segments import segment_environment_by_range
        env = self._ssp_driven()
        n_low = len(segment_environment_by_range(env, freq=100.0))
        n_high = len(segment_environment_by_range(env, freq=800.0))
        assert n_high > n_low, (
            f"{n_low} profiles at 100 Hz and {n_high} at 800 Hz: a flat-bottom "
            f"but range-dependent SSP must still scale with frequency")

    def test_each_segment_spans_a_bounded_profile_change(self):
        from uacpy.models.kraken._segments import (
            _max_profile_change, _ssp_change_ceiling,
            segment_environment_by_range)
        env = self._ssp_driven()
        freq = 800.0
        ceiling = _ssp_change_ceiling(env, freq)
        edges = [r for r, _ in segment_environment_by_range(env, freq=freq)]
        worst = max(_max_profile_change(env, a, b)
                    for a, b in zip(edges, edges[1:]))
        assert worst <= ceiling * 1.01

    def test_a_flat_isovelocity_environment_is_not_segmented(self):
        from uacpy.core import BoundaryProperties, Environment
        from uacpy.models.kraken._segments import segment_environment_by_range
        env = Environment(
            bathymetry=200.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.7,
                                      attenuation=0.5))
        assert len(segment_environment_by_range(env, freq=800.0)) == 1


@pytest.mark.requires_binary
class TestTheModeGridSpansTheProfileKrakenSolves:
    """compute_modes solves the r = 0 profile. Sizing its depth grid from the
    deepest point of a range-dependent bathymetry asked KRAKEN for receivers
    below that profile, which it clamped — and the wrapper then warned about
    a grid it had built itself."""

    def test_a_deepening_bathymetry_gives_a_grid_of_the_first_column(self):
        from uacpy.core.source import Source
        from uacpy.models.kraken import Kraken
        env = uacpy.Environment(name='rd', ssp=1500.0,
                                bathymetry=[(0.0, 100.0), (5000.0, 200.0)])
        source = Source(depths=50.0, frequencies=100.0)
        with recorded_warnings() as rec:
            modes = Kraken(verbose=False).compute_modes(env, source)
        assert float(np.max(np.asarray(modes.depths))) == pytest.approx(100.0)
        spurious = [str(w.message) for w in rec
                    if 'resolvable depth' in str(w.message)
                    or 'moved up' in str(w.message)]
        assert spurious == [], spurious


@pytest.mark.requires_binary
class TestNarrowbandLineSourceCarriesTheSameLevelAsBroadband:
    """The ×√k0 line-source level (``_conventions._line_source_unit_at_1m``, the
    package's unit-amplitude-at-1-m convention) has to reach the NARROWBAND
    branch of ``_extract.assemble_field_from_shd``, not just its broadband
    and ``return_pressure`` siblings.

    It once reached only those two, so a ``source_type='line'`` COHERENT_TL
    sat 10·log10(k0) dB from this same wrapper's own single-bin broadband run
    — 3.8 dB at 100 Hz, 10.8 dB at 20 Hz, and the OTHER WAY above
    f = c(z_s)/2π, so it never looked like a fixed convention offset.
    INCOHERENT_TL carried it too: ``field.exe``'s magnitude sum lands in the
    same payload.

    The duct is driven at 20 Hz, where it traps exactly ONE mode. A one-term
    magnitude sum equals the modulus of the one-term coherent sum, so a
    single broadband reference pins BOTH narrowband branches: the level is a
    real positive scalar and multiplies each of them identically.
    """

    #: 100-m Pekeris, 20 Hz: mode 2 cuts on at 1.5·1500/(2·100·√(1-(15/17)²))
    #: = 33 Hz, so the sum has one term. k0 = 2π·20/1500 = 0.084, i.e. a
    #: missing √k0 is a 10.8 dB error — far outside any tolerance here.
    FREQ = 20.0

    @staticmethod
    def _rig(source_type):
        env = Environment(
            name='line-level', bathymetry=100.0, ssp=C_WATER,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=C_BOTTOM, density=1.8,
                                      attenuation=0.5))
        src = Source(
            depths=50.0,
            frequencies=TestNarrowbandLineSourceCarriesTheSameLevelAsBroadband.FREQ,
            source_type=source_type)
        rcv = Receiver(depths=np.array([50.0]),
                       ranges=np.array([500.0, 1000.0]))
        return env, src, rcv

    @staticmethod
    def _tl(run_mode, source_type, **kw):
        env, src, rcv = (
            TestNarrowbandLineSourceCarriesTheSameLevelAsBroadband._rig(
                source_type))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            field = Kraken(verbose=False).run(env, src, rcv,
                                              run_mode=run_mode, **kw)
        return np.asarray(field.dB, dtype=float).ravel()

    def _broadband(self, source_type):
        return self._tl(RunMode.BROADBAND, source_type,
                        frequencies=np.array([self.FREQ]))

    @pytest.mark.parametrize('source_type', ['line', 'point'])
    def test_coherent_tl_equals_the_single_bin_broadband(self, source_type):
        coherent = self._tl(RunMode.COHERENT_TL, source_type)
        broadband = self._broadband(source_type)
        # Same modes, same evaluator, same level: the two branches only
        # differ in how they read the .shd back, so this is cell-for-cell.
        assert np.allclose(coherent, broadband, rtol=0, atol=1e-9), (
            coherent, broadband)

    @pytest.mark.parametrize('source_type', ['line', 'point'])
    def test_incoherent_tl_equals_the_single_bin_broadband(self, source_type):
        incoherent = self._tl(RunMode.INCOHERENT_TL, source_type)
        broadband = self._broadband(source_type)
        # One trapped mode, so SQRT(SUM(z**2)) is |z| — the magnitude of the
        # coherent sum. The residual is field.exe's own single-precision
        # .shd, not a level difference.
        assert np.allclose(incoherent, broadband, rtol=0, atol=1e-4), (
            incoherent, broadband)

    def test_the_line_source_is_actually_scaled_relative_to_the_point_one(self):
        # A control on the two tests above: they would both pass if the level
        # were dropped from EVERY branch. It is not — the line and point
        # levels differ, and by more than the 10.8 dB the bug removed.
        line = self._tl(RunMode.COHERENT_TL, 'line')
        point = self._tl(RunMode.COHERENT_TL, 'point')
        assert np.all(np.abs(line - point) > 1.0), (line, point)

    def test_the_level_factor_equals_sqrt_k0_against_the_raw_shd(self):
        """The FACTOR itself, derived here instead of borrowed.

        Every other test in this class compares two code paths that both call
        ``_conventions._line_source_unit_at_1m``, so a wrong constant *inside* that
        helper — √(4πf/c) for √(2πf/c), or c read at the surface instead of at
        the source — moves both sides together and is invisible to all of
        them. This one divides the returned pressure by the engine's own
        unscaled ``.shd`` and recomputes the expected ratio from f and c(z_s),
        never calling the helper.

        A line-to-point COMPARISON cannot do this job: ``EvaluateMod.f90:36-39``
        weights the line sum by ``C/k`` and the point sum by ``C/√k``, so the
        two differ per mode, not by one global factor.
        """
        from uacpy.io import read_shd_bin

        env, src, rcv = self._rig('line')
        with tempfile.TemporaryDirectory() as work:
            field = Kraken(verbose=False, work_dir=work, cleanup=False).run(
                env, src, rcv, run_mode=RunMode.COHERENT_TL)
            shd = next(Path(work).rglob('*.shd'))
            raw = read_shd_bin(shd)

        # Both sides as amplitudes: the wrapper reports a loss, the .shd holds
        # the complex field field.exe wrote before the wrapper touched it.
        got = np.power(10.0, -np.asarray(field.tl, dtype=float).ravel() / 20.0)
        unscaled = np.abs(np.asarray(raw.pressure, dtype=complex)).ravel()
        keep = np.isfinite(got) & (unscaled > 0)
        assert keep.any(), "the .shd carried no non-zero cell to divide by"

        c_source = float(np.atleast_1d(env.ssp.sound_speed_at(50.0))[0])
        # field.exe also omits the modal sum's 1/rho(z_s), which the wrapper
        # divides out (the source sits in water of density env.water_density).
        expected = (np.sqrt(2.0 * np.pi * self.FREQ / c_source)
                    / env.water_density)
        ratio = got[keep] / unscaled[keep]
        assert np.allclose(ratio, expected, rtol=2e-4, atol=0), (
            f"the wrapper scaled the .shd by {ratio}, but the line-source "
            f"convention is sqrt(k0)/rho_w = sqrt(2*pi*{self.FREQ}/{c_source})"
            f"/{env.water_density} = {expected:.6f}")

    def test_the_reference_speed_is_read_at_the_source_depth(self):
        """c in k0 = 2πf/c is c(z_s), not a global or surface value.

        On an isovelocity duct every candidate speed coincides, so the test
        above cannot separate them. Here the column has a 100 m/s gradient and
        two source depths sit in different water: the offsets they produce
        must differ by 10·log10(c_shallow/c_deep), which is zero for any
        implementation that reads one speed for the whole column.
        """
        from uacpy.core.ssp import SoundSpeedProfile
        ssp = SoundSpeedProfile(depths=np.array([0.0, 100.0]),
                                sound_speed=np.array([1450.0, 1550.0]))
        offsets = []
        for z_s in (20.0, 80.0):
            env = Environment(
                name='line-level-gradient', bathymetry=100.0, ssp=ssp,
                bottom=BoundaryProperties(acoustic_type='half-space',
                                          sound_speed=C_BOTTOM, density=1.8,
                                          attenuation=0.5))
            rcv = Receiver(depths=np.array([50.0]),
                           ranges=np.array([1000.0]))
            levels = {}
            for kind in ('line', 'point'):
                src = Source(depths=z_s, frequencies=self.FREQ,
                             source_type=kind)
                levels[kind] = np.asarray(
                    Kraken(verbose=False).run(
                        env, src, rcv, run_mode=RunMode.COHERENT_TL).tl
                ).ravel()[0]
            offsets.append(levels['point'] - levels['line'])

        c_shallow = float(np.atleast_1d(env.ssp.sound_speed_at(20.0))[0])
        c_deep = float(np.atleast_1d(env.ssp.sound_speed_at(80.0))[0])
        expected = 10.0 * np.log10(c_deep / c_shallow)
        assert abs((offsets[0] - offsets[1]) - expected) < 2e-3, (
            f"offsets {offsets} differ by {offsets[0] - offsets[1]:.4f} dB; "
            f"c(z_s) read at the source depth predicts {expected:.4f} dB "
            f"(c={c_shallow} vs {c_deep}). A column-wide speed gives 0.")


def test_field_exe_non_fatal_warnings_are_surfaced(tmp_path, monkeypatch):
    """field.exe writes its non-fatal ``Warning in ...`` lines to
    ``field.prt`` (``KrakenField/ReadModes.f90:90-111``); the launch reads
    that log back and passes them on as a ``UserWarning``, as every other
    AT launch does through ``_run_and_attach_prt``."""
    model = Kraken(work_dir=tmp_path, cleanup=False)
    fm = model._setup_file_manager()

    def fake_run(cmd, **kwargs):
        (fm.work_dir / 'field.prt').write_text(
            "Warning in ReadModes : Receiver below depth of bottom\n"
            "Field completed successfully\n")
        (fm.work_dir / 'model.shd').write_bytes(b'\x00' * 8)

    monkeypatch.setattr(model, '_run_subprocess', fake_run)
    with pytest.warns(UserWarning, match='Receiver below depth of bottom'):
        model._run_field_exe(fm.work_dir, 'model', 'RC C')


def test_the_mode_grid_is_sized_on_the_column_the_modes_are_solved_on():
    """``compute_modes`` solves the r = 0 profile, so its default depth grid
    spans that column's water plus THAT column's sediment stack — not the
    thickest stack anywhere along the track, which would ask KRAKEN for
    depths below the deck it wrote and draw its 'moved up' warning on a grid
    the wrapper built itself."""
    near = SeabedColumn(
        layers=[SedimentLayer(5, 1650, 1.8, 0.3)],
        halfspace=BoundaryProperties(acoustic_type='half-space',
                                     sound_speed=1800, density=2.0,
                                     attenuation=0.2))
    far = SeabedColumn(
        layers=[SedimentLayer(30, 1650, 1.8, 0.3)],
        halfspace=BoundaryProperties(acoustic_type='half-space',
                                     sound_speed=1800, density=2.0,
                                     attenuation=0.2))
    env = Environment(name='thickening-bed', bathymetry=50.0, ssp=1500.0,
                      bottom=Bottom.from_columns([near, far],
                                                 ranges=np.array([0.0, 5000.0])))
    with recorded_warnings() as rec:
        modes = Kraken(verbose=False).compute_modes(
            env, Source(depths=25.0, frequencies=100.0))
    said = [str(w.message) for w in rec
            if 'resolvable depth' in str(w.message)
            or 'moved up' in str(w.message)]
    assert said == [], said
    assert float(np.max(modes.depths)) <= 55.0 + 1e-9


@pytest.mark.requires_binary
@pytest.mark.parametrize('run_mode', [RunMode.COHERENT_TL,
                                      RunMode.INCOHERENT_TL])
def test_multi_depth_line_source_slab_equals_its_standalone_run(run_mode):
    """The one-launch multi-depth TL deck carries every source depth, and the
    line-source level √(2πf/c(z_s)) belongs to each slab's own depth. On a
    1540→1440 m/s profile a slab levelled with the first depth's c sits
    10·log10(c(10 m)/c(90 m)) = 0.233 dB off at 90 m; the first slab is the
    control that equals its stand-alone run either way."""
    env = Environment(
        name='line-slabs', bathymetry=100.0,
        ssp=uacpy.SoundSpeedProfile.from_pairs([(0.0, 1540.0),
                                               (100.0, 1440.0)]),
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1700.0, density=1.8,
                                  attenuation=0.5))
    rcv = Receiver(depths=np.linspace(10.0, 90.0, 5),
                   ranges=np.linspace(500.0, 3000.0, 5))
    depths = [10.0, 90.0]

    def tl(src):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return Kraken(verbose=False).run(env, src, rcv, run_mode=run_mode)

    stack = tl(Source(depths=depths, frequencies=100.0, source_type='line'))
    for slab, z in zip(stack.slabs, depths):
        alone = tl(Source(depths=z, frequencies=100.0, source_type='line'))
        diff = np.abs(np.asarray(slab.dB, float) - np.asarray(alone.dB, float))
        assert np.nanmax(diff) < 1e-3, (z, np.nanmax(diff))


@pytest.mark.requires_binary
@pytest.mark.parametrize('frequencies', [[100.0], [90.0, 100.0, 110.0]])
def test_broadband_keeps_the_source_level_on_every_grid_size(frequencies):
    """A one-bin grid is solved through the narrowband pipeline on a Source
    pinned to that bin; the pin must keep ``source_level_dB`` so
    ``Field.at_source_level()`` keeps its default, as the multi-bin grid
    does."""
    env = Environment(name='sl', bathymetry=100.0, ssp=C_WATER,
                      bottom=BoundaryProperties(acoustic_type='half-space',
                                                sound_speed=C_BOTTOM,
                                                density=1.8, attenuation=0.5))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        field = Kraken(verbose=False).run(
            env, Source(depths=50.0, frequencies=100.0, source_level_dB=180.0),
            Receiver(depths=[50.0], ranges=[1000.0]),
            run_mode=RunMode.BROADBAND, frequencies=np.array(frequencies))
    assert field.source_level_dB == pytest.approx(180.0)


class TestFieldResultsRecordTheModesBinary:
    """``backend`` on a TL / broadband result is the modes binary the
    dispatch picked (kraken or krakenc), as ``compute_modes`` stamps it; the
    settings' route says field.exe summed the modes."""

    _SRC = Source(depths=25.0, frequencies=100.0)
    _RCV = Receiver(depths=[30.0], ranges=[1000.0])

    @staticmethod
    def _env(shear_speed):
        return Environment(
            name='hs', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800, density=1.8,
                                      attenuation=0.3,
                                      shear_speed=shear_speed))

    def test_fluid_tl_names_kraken(self):
        result = Kraken(verbose=False).run(self._env(0.0), self._SRC, self._RCV)
        assert result.backend == 'kraken'
        assert 'modes_backend' not in result.metadata

    def test_elastic_tl_names_krakenc(self):
        result = Kraken(verbose=False).run(self._env(400.0), self._SRC,
                                           self._RCV)
        assert result.backend == 'krakenc'
        assert 'modes_backend' not in result.metadata

    def test_broadband_carries_the_stamp(self):
        result = Kraken(verbose=False).run(
            self._env(0.0), self._SRC, self._RCV, run_mode=RunMode.BROADBAND,
            frequencies=np.array([90.0, 100.0]))
        assert result.backend == 'kraken'
        assert 'modes_backend' not in result.metadata


class TestSingleRunGroupVelocity:
    """The ``.prt`` group-speed column: krakenc fills it, kraken does not.

    ``krakenc.f90:819`` assigns ``VG = 1/Slow``; the same assignment is
    commented out at ``kraken.f90:815-819``, so kraken prints the column
    unassigned. ``Modes.group_velocity`` must reflect that difference rather
    than hand back a plausible-looking array of zeros.
    """

    def _modes(self, backend, freq=200.0):
        env = _duct(100.0, c_bottom=1600.0)
        return Kraken(backend=backend).compute_modes(
            env, Source(depths=50.0, frequencies=freq))

    def test_kraken_reports_none_rather_than_the_unfilled_zero_column(self):
        m = self._modes('kraken')
        assert m.n_modes > 0
        assert m.group_velocity is None

    def test_krakenc_reports_a_group_speed_per_mode(self):
        m = self._modes('krakenc')
        gv = m.group_velocity
        assert gv is not None
        assert gv.shape == m.k.shape
        assert np.all(np.isfinite(gv))
        assert np.all(gv > 0)

    def test_krakenc_group_speed_sits_below_the_phase_speed(self):
        # The defining property of waveguide dispersion; also the check that
        # the parsed column is the group speed and not the phase-speed column
        # one position to its left.
        m = self._modes('krakenc')
        assert np.all(m.group_velocity < m.phase_speeds)

    def test_single_run_agrees_with_the_two_run_finite_difference_when_trapped(self):
        # The independent route. They are different computations -- one is
        # KRAKENC's perturbation pass, the other d(omega)/dk by differencing
        # two runs -- so agreement on the trapped modes is evidence both are
        # right. Only the trapped ones: see the leaky-mode test below.
        env = _duct(100.0, c_bottom=1600.0)
        m0 = Kraken(backend='krakenc').compute_modes(
            env, Source(depths=50.0, frequencies=200.0))
        m1 = Kraken(backend='krakenc').compute_modes(
            env, Source(depths=50.0, frequencies=200.5))
        fd = m0.group_velocity_between(m1)
        trapped = m0.phase_speeds[:len(fd)] < 1600.0
        assert trapped.sum() >= 5
        assert np.nanmax(
            np.abs(m0.group_velocity[:len(fd)][trapped] - fd[trapped])) < 1.0

    def test_the_two_routes_part_company_on_the_leaky_modes(self):
        # kraken.f90:772 states the caveat outright: the group speed "still
        # uses the perturbation method so group speeds will be wrong for
        # leaky modes". Both routes inherit it, and they do not fail the same
        # way, so their disagreement is confined to exactly those modes. This
        # pins the caveat rather than hiding it behind a loose tolerance.
        env = _duct(100.0, c_bottom=1600.0)
        m0 = Kraken(backend='krakenc').compute_modes(
            env, Source(depths=50.0, frequencies=200.0))
        m1 = Kraken(backend='krakenc').compute_modes(
            env, Source(depths=50.0, frequencies=200.5))
        fd = m0.group_velocity_between(m1)
        cp = m0.phase_speeds[:len(fd)]
        diff = np.abs(m0.group_velocity[:len(fd)] - fd)
        leaky = cp >= 1600.0
        assert leaky.any(), 'default c_high should admit some leaky modes'
        # Trapped: the two agree closely. Leaky: they need not, and here do not.
        assert np.nanmax(diff[~leaky]) < 1.0
        assert np.nanmax(diff[leaky]) > np.nanmax(diff[~leaky])

    def test_strided_print_table_lands_on_the_right_modes(self):
        # kraken.f90:101 prints with stride MAX(1, M/30), so a run with more
        # than 30 modes lists only about 30. The values must be placed by the
        # printed mode INDEX, not by row position, or they would be assigned
        # to modes 1..30 and be wrong for every one past the first.
        m = self._modes('krakenc', freq=1500.0)
        gv = m.group_velocity
        assert m.n_modes > 30
        reported = np.flatnonzero(np.isfinite(gv))
        assert 0 < reported.size < m.n_modes        # genuinely strided
        assert reported[0] == 0                     # mode 1 always printed
        stride = max(1, m.n_modes // 30)
        assert np.all(np.diff(reported) == stride)
        # Whatever landed is still physical.
        cp = m.phase_speeds
        assert np.all(gv[reported] < cp[reported])

    def test_first_n_slices_the_group_speed_alongside_the_modes(self):
        m = self._modes('krakenc')
        five = m.first_n(5)
        assert five.group_velocity.shape == (5,)
        assert np.allclose(five.group_velocity, m.group_velocity[:5])

    def test_shape_mismatch_is_rejected(self):
        m = self._modes('krakenc')
        with pytest.raises(ConfigurationError, match='group_velocity'):
            Modes(k=m.k, phi=m.phi, depths=m.depths,
                  group_velocity=np.ones(len(m.k) + 1), **m.id_kwargs())



# ── Kraken's field carries the modal sum's 1/rho(z_s) (JKPS eq. 5.14) ──
# field.exe omits it, so its field is the unit-source field times rho(z_s).
# Scaling EVERY density by the same factor leaves a unit-source field exactly
# unchanged (reflection depends only on density ratios), so the sharpest pin
# is that invariance: before the fix Kraken moved by -20 log10(1.1) =
# -0.828 dB while every other engine moved by 0.000.
def _rho_scaled_env(scale):
    return Environment(
        name='pekeris', bathymetry=100.0, ssp=1500.0,
        water_density=1.0 * scale,
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1800.0, density=1.8 * scale,
                                  attenuation=0.0))


def _rho_scaled_tl(scale, run_mode='coherent'):
    src = Source(depths=20.0, frequencies=100.0)
    rcv = Receiver(depths=[50.0], ranges=[1000.0, 3000.0, 5000.0])
    k = Kraken()
    if run_mode == 'coherent':
        f = k.compute_tl(_rho_scaled_env(scale), src, rcv)
    else:
        f = k.run(_rho_scaled_env(scale), src, rcv, run_mode=uacpy.RunMode.INCOHERENT_TL)
    return np.asarray(f.dB, dtype=float).ravel()


@pytest.mark.parametrize('run_mode', ['coherent', 'incoherent'])
def test_uniform_density_scaling_leaves_kraken_tl_unchanged(run_mode):
    base = _rho_scaled_tl(1.0, run_mode)
    scaled = _rho_scaled_tl(1.1, run_mode)
    np.testing.assert_allclose(scaled, base, atol=0.02)


def test_modal_sum_divides_by_the_recorded_water_density():
    env = _rho_scaled_env(1.1)
    src = Source(depths=20.0, frequencies=100.0)
    rcv = Receiver(depths=[50.0], ranges=[1000.0, 3000.0])
    k = Kraken()
    modes = k.compute_modes(env, src)
    # The run's own density, exactly: the .mod holds it as float32
    # (1.100000023841858), which is not the value the run divides by.
    assert modes.media.water_density == env.water_density
    field = k.compute_tl(env, src, rcv)
    loss = modes.modal_pressure_field(
        source_depth=20.0, receiver_depths=np.array([50.0]),
        ranges=np.array([1000.0, 3000.0]))
    np.testing.assert_allclose(np.asarray(loss.dB).ravel(),
                               np.asarray(field.dB).ravel(), atol=0.05)
    unit = modes.modal_pressure_field(
        source_depth=20.0, receiver_depths=np.array([50.0]),
        ranges=np.array([1000.0, 3000.0]), source_density=1.0)
    # rho = 1 reproduces field.exe's own (factor-free) output: 0.83 dB louder.
    np.testing.assert_allclose(
        np.asarray(loss.dB).ravel() - np.asarray(unit.dB).ravel(),
        20 * np.log10(1.1), atol=1e-6)


@pytest.mark.parametrize("bottom, closed_form, tolerance_dB", [
    (dict(acoustic_type='half-space', sound_speed=1800, density=2.0,
          attenuation=0.0), 'pekeris', 0.05),
    (dict(acoustic_type='vacuum'), 'ideal_waveguide', 0.2),
    (dict(acoustic_type='rigid'), 'ideal_waveguide', 0.3),
])
def test_kraken_reproduces_the_closed_form_field(bottom, closed_form,
                                                 tolerance_dB):
    """Level AND phase: the analytic field shares the engine's
    travelling-wave carrier and 1 m normalisation."""
    from uacpy import analytic
    from uacpy.core.boundary import BoundaryProperties
    env = uacpy.Environment(bathymetry=100, ssp=1500,
                            bottom=BoundaryProperties(**bottom))
    src = uacpy.Source(depths=[36.0], frequencies=[50.0])
    rx = uacpy.Receiver(depths=np.linspace(5, 95, 10),
                        ranges=np.linspace(1000, 10000, 10))
    ref = getattr(analytic, closed_form)(env, src, rx)
    got = uacpy.Kraken().run(env, src, rx)
    assert np.median(np.abs(got.tl - ref.tl)) < tolerance_dB
    assert np.median(np.abs(np.angle(got.data / ref.data))) < 0.02


@pytest.mark.requires_binary
def test_a_relative_field_executable_is_bound_absolute_at_construction(
        monkeypatch, tmp_path):
    """field.exe launches with ``cwd=`` a scratch dir, so a relative
    ``field_executable`` must be made absolute against the constructor's cwd;
    changing directory afterwards must not change what launches. The verbatim
    argument is kept for ``copy()`` / ``repr``."""
    import os
    exe = Kraken(verbose=False)._resolve_field_executable()
    monkeypatch.chdir(exe.parent.parent)
    rel = Path(exe.parent.name) / exe.name
    model = Kraken(field_executable=rel, verbose=False)
    monkeypatch.chdir(tmp_path)
    assert model.field_executable == rel
    launched = model._resolve_field_executable()
    assert launched.is_absolute()
    assert os.path.samefile(launched, exe)


class TestRangeDependentBottomIsSegmented:
    """Every profile block of the multi-profile ``.env`` is a full
    environment read by its own ``ReadEnvironment`` call
    (``kraken.f90:42-46``), and the ``.mod`` stores each profile's own
    half-space (``ReadModes.f90:69``). So the field path writes each
    segment's own seabed column instead of one collapsed column for the
    whole run. Two columns: 1600 m/s sand out to 5 km, 1900 m/s beyond
    (``Bottom.at`` takes the nearest column)."""

    @staticmethod
    def _column(cp):
        from uacpy.core.environment import SeabedColumn
        return SeabedColumn(layers=[], halfspace=BoundaryProperties(
            acoustic_type='half-space', sound_speed=cp, density=1.8,
            attenuation=0.5))

    def _env(self, columns=(1600.0, 1900.0)):
        from uacpy.core.environment import Bottom
        return Environment(
            name='rd-bottom', bathymetry=100.0, ssp=1500.0,
            bottom=Bottom.from_columns([self._column(c) for c in columns],
                                       ranges=np.array([0.0, 10000.0])))

    def test_projection_keeps_every_column_without_a_collapse_notice(self):
        model = Kraken(verbose=False)
        with recorded_warnings() as caught:
            projected = model._project_environment(self._env())
        assert projected.bottom.is_range_dependent
        assert not [w for w in caught
                    if 'range-dependent bottoms' in str(w.message)]

    def test_each_profile_block_carries_its_own_half_space(self, tmp_path):
        model = Kraken(verbose=False, work_dir=tmp_path, cleanup=False,
                       mode_coupling='adiabatic')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.run(self._env(), Source(depths=50.0, frequencies=100.0),
                      Receiver(depths=[50.0], ranges=[1000.0, 9000.0]))
        deck = (tmp_path / 'kfield.env').read_text()
        assert '1600.' in deck and '1900.' in deck

    @pytest.mark.requires_binary
    def test_the_near_half_equals_its_flat_bottom_run(self):
        # Profiles at 0, 2 and 4 km all carry the 1600 m/s column, so the
        # adiabatic field short of 4 km is the flat 1600 m/s problem; past
        # the switch it is not.
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=[30.0, 70.0],
                       ranges=np.array([1000.0, 2000.0, 3000.0, 9000.0]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rd = Kraken(verbose=False, mode_coupling='adiabatic').run(
                self._env(), src, rcv)
            flat = Kraken(verbose=False).run(
                Environment(name='flat', bathymetry=100.0, ssp=1500.0,
                            bottom=self._column(1600.0).halfspace),
                src, rcv)
        near = np.abs(np.asarray(rd.dB)[:, :3] - np.asarray(flat.dB)[:, :3])
        assert np.nanmax(near) < 0.05, np.nanmax(near)
        far = np.abs(np.asarray(rd.dB)[:, 3] - np.asarray(flat.dB)[:, 3])
        assert np.nanmax(far) > 1.0, far

    def test_coupled_incoherent_is_refused_up_front_on_an_rd_bottom(
            self, tmp_path):
        # A range-dependent bottom alone makes the deck multi-profile, and
        # field.f90:125-129 stops on coupled + incoherent there; the gate
        # names the remedy before any binary runs (the work dir stays empty).
        from uacpy.core.exceptions import ConfigurationError
        model = Kraken(verbose=False, mode_coupling='coupled',
                       work_dir=tmp_path, cleanup=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ConfigurationError,
                               match='incoherent addition'):
                model.run(self._env(), Source(depths=50.0, frequencies=100.0),
                          Receiver(depths=[50.0], ranges=[1000.0]),
                          run_mode=RunMode.INCOHERENT_TL)
        assert not any(tmp_path.iterdir())

    @pytest.mark.requires_binary
    def test_coupled_incoherent_runs_on_a_range_independent_bottom(self):
        # The other side of the gate: one column, one profile, which
        # field.exe accepts.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            field = Kraken(verbose=False, mode_coupling='coupled').run(
                Environment(name='ri', bathymetry=100.0, ssp=1500.0,
                            bottom=self._column(1600.0).halfspace),
                Source(depths=50.0, frequencies=100.0),
                Receiver(depths=[50.0], ranges=[1000.0]),
                run_mode=RunMode.INCOHERENT_TL)
        assert np.all(np.isfinite(np.asarray(field.dB)))


def test_krakenc_keeping_no_modes_names_the_c_high_remedy(tmp_path):
    """KRAKENC's no-mode branch re-OPENs its own mode file
    (``Kraken/krakenc.f90:431-443``), so gfortran stops with an OPEN error
    instead of the 'No modes' diagnosis. Measured: a BOUNCE table of an
    elastic half-space (cp 1600, cs 400) sized for a 5 km receiver, read by
    KRAKENC at 100 Hz with ``c_high=1e9``, keeps no mode; the wrapper says so
    and names ``c_high``."""
    from uacpy.core.exceptions import ModelExecutionError
    from uacpy.models import Bounce
    elastic = Environment(name='nomodes', bathymetry=100.0, ssp=1500.0,
                          bottom=BoundaryProperties(
                              acoustic_type='half-space', sound_speed=1600.0,
                              shear_speed=400.0, density=1.8,
                              attenuation=0.2, shear_attenuation=0.5))
    src = Source(depths=50.0, frequencies=100.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        table = Bounce(verbose=False, work_dir=tmp_path).run(
            elastic, src, Receiver(depths=[25.0, 50.0, 75.0],
                                   ranges=[1000.0, 3000.0, 5000.0]))
    env = Environment(name='nomodes', bathymetry=100.0, ssp=1500.0,
                      bottom=BoundaryProperties(
                          acoustic_type='file',
                          reflection_file=table.metadata['brc_file'],
                          sound_speed=1600.0, density=1.8))
    with pytest.raises(ModelExecutionError, match='kept no modes'):
        Kraken(backend='krakenc', verbose=False, c_low=1400.0,
               c_high=1e9).compute_modes(env=env, source=src)


def test_a_bounce_table_seabed_runs_with_the_default_phase_speed_window(
        tmp_path):
    """A reflection-table seabed carries no sound speed, so the writer's
    default c_high is unbounded (1e9), and on a BOUNCE table of an elastic
    half-space sized for a 5 km receiver KRAKENC then kept no mode. The
    default window over a table is 10x the fastest water speed; measured,
    every window from 5000 to 30000 m/s gave the same TL, 0.41 dB median
    from KRAKENC on the half-space itself."""
    from uacpy.models import Bounce
    bp = BoundaryProperties(acoustic_type='half-space', sound_speed=1600.0,
                            shear_speed=400.0, density=1.8, attenuation=0.2,
                            shear_attenuation=0.5)
    elastic = Environment(name='elastic', bathymetry=100.0, ssp=1500.0,
                          bottom=bp)
    src = Source(depths=50.0, frequencies=100.0)
    rcv = Receiver(depths=np.linspace(10.0, 90.0, 9),
                   ranges=np.linspace(500.0, 10000.0, 40))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        table = Bounce(verbose=False, work_dir=tmp_path / 'b').run(
            elastic, src, Receiver(depths=[50.0], ranges=[5000.0]))
        tabled = Environment(
            name='table', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(
                acoustic_type='file',
                reflection_file=table.metadata['brc_file']))
        model = Kraken(backend='krakenc', verbose=False,
                       work_dir=tmp_path / 'k', cleanup=False)
        via_table = np.asarray(model.run(tabled, src, rcv).dB)
        direct = np.asarray(Kraken(backend='krakenc', verbose=False,
                                   c_high=1e4).run(elastic, src, rcv).dB)
    deck = next((tmp_path / 'k').glob('*.env')).read_text().splitlines()
    assert any(line.split()[:2] == ['0.0', '15000.0'] for line in deck)
    assert np.nanmedian(np.abs(via_table - direct)) < 1.0


# ── the staged protocol: refusals, settings, decks (w2-kraken) ────────────

def _fluid_floor(kind, surface=None):
    """100 m of 1500 m/s water over a rigid or vacuum floor."""
    kw = dict(name=f'{kind}_floor', bathymetry=100.0, ssp=1500.0,
              water_density=1.0,
              bottom=BoundaryProperties(acoustic_type=kind))
    if surface is not None:
        kw['surface'] = surface
    return Environment(**kw)


def _ice():
    from uacpy.core.surface import Surface
    return Surface(nodes=[BoundaryProperties(
        acoustic_type='half-space', sound_speed=3500.0, shear_speed=1800.0,
        density=0.9)])


def _hard_elastic():
    """100 m of water over cp 3000 / cs 1400 / rho 2.2 (RA-WAVE-17)."""
    return Environment(name='hard', bathymetry=100.0, ssp=1500.0,
                       bottom=BoundaryProperties(
                           acoustic_type='half-space', sound_speed=3000.0,
                           shear_speed=1400.0, density=2.2, attenuation=0.1,
                           shear_attenuation=0.2))


def _SRC_30():
    return Source(depths=30.0, frequencies=50.0)


def _RCV_FAR():
    return Receiver(depths=np.array([10.0, 50.0, 80.0]),
                    ranges=np.array([2000.0, 5000.0]))


class TestTheThreeEntryPointsRefuseAlike:
    """``run``, ``run_settings`` and ``validate_inputs`` refuse the same
    Kraken calls, with the same error, before any binary is launched."""

    CASES = [
        ('krakenc over a rigid floor', dict(backend='krakenc'),
         lambda: _fluid_floor('rigid'), {}, ConfigurationError,
         r"backend='krakenc'\) over a rigid seabed"),
        ('krakenc over a vacuum floor', dict(backend='krakenc'),
         lambda: _fluid_floor('vacuum'), {}, ConfigurationError,
         r"backend='krakenc'\) over a vacuum seabed"),
        ('leaky modes over a rigid floor', dict(leaky_modes=True),
         lambda: _fluid_floor('rigid'), {}, ConfigurationError,
         r"leaky_modes=True\) over a rigid seabed"),
        ('leaky modes under ice over a rigid floor', dict(leaky_modes=True),
         lambda: _fluid_floor('rigid', surface=_ice()), {},
         ConfigurationError,
         r"leaky_modes=True\) under an elastic ice canopy over a rigid"),
        ('leaky modes under ice over a vacuum floor', dict(leaky_modes=True),
         lambda: _fluid_floor('vacuum', surface=_ice()), {},
         ConfigurationError,
         r"leaky_modes=True\) under an elastic ice canopy over a vacuum"),
        ('kraken on an elastic seabed', dict(backend='kraken'),
         _hard_elastic, {}, ConfigurationError, 'elastic media'),
        ('coupled incoherent on a range-dependent deck',
         dict(mode_coupling='coupled'),
         lambda: Environment(name='rd', bathymetry=[(0.0, 100.0),
                                                   (5000.0, 120.0)],
                             ssp=1500.0),
         dict(run_mode=RunMode.INCOHERENT_TL), ConfigurationError,
         'incoherent addition'),
        ("interp_ssp='quad'", dict(interp_ssp='quad'),
         lambda: _pekeris(depth=100.0), {}, UnsupportedFeatureError, 'quad'),
        ('window inverted by a pinned c_low', dict(c_low=2000.0),
         lambda: _pekeris(depth=100.0), {}, ConfigurationError,
         'c_low < c_high'),
    ]

    @pytest.mark.parametrize('label, ctor, env, kw, exc, match', CASES,
                             ids=[c[0] for c in CASES])
    def test_every_entry_point_refuses(self, monkeypatch, label, ctor, env,
                                       kw, exc, match):
        model = Kraken(verbose=False, **ctor)

        def _no_launch(*a, **k):
            raise AssertionError('a binary was launched')
        monkeypatch.setattr(model, '_run_subprocess', _no_launch)
        for entry in ('run', 'run_settings', 'validate_inputs'):
            with pytest.raises(exc, match=match):
                getattr(model, entry)(env(), _SRC_30(), _RCV_FAR(), **kw)

    def test_krakenc_over_a_rigid_floor_under_an_elastic_top_is_accepted(self):
        """An environment that needs KRAKENC anyway (an ice canopy) keeps it
        over a rigid floor, with a finite window."""
        settings = Kraken(verbose=False, backend='krakenc').run_settings(
            _fluid_floor('rigid', surface=_ice()), _SRC_30(), _RCV_FAR())
        assert settings.engine.backend == 'krakenc'
        assert settings.engine.launches[0].c_high == (15000.0,)


class TestTheWindowIsDecidedPerBoundaryType:
    """ARCH-8: one resolver (``_window.phase_speed_window``) sets every
    deck's window, per boundary type, and ``run_settings`` records it with
    the rule that set it."""

    @staticmethod
    def _window(model, env, **kw):
        engine = model.run_settings(env, _SRC_30(), _RCV_FAR(), **kw).engine
        return engine.launches[0].c_low, engine.launches[0].c_high, engine

    def test_a_fluid_half_space_gets_five_percent_past_its_speed(self):
        c_low, c_high, engine = self._window(Kraken(verbose=False),
                                             _pekeris(depth=100.0))
        assert c_low == 0.0
        assert c_high == (pytest.approx(1.05 * 1800.0),)
        assert engine.c_high_origin.startswith('1.05 × max')

    @pytest.mark.parametrize('kind', ['rigid', 'vacuum'])
    def test_a_rigid_or_vacuum_floor_is_unbounded_on_kraken(self, kind):
        _c_low, c_high, engine = self._window(Kraken(verbose=False),
                                              _fluid_floor(kind))
        assert engine.backend == 'kraken'
        assert c_high == (1e9,)

    @pytest.mark.parametrize('kind', ['rigid', 'vacuum'])
    def test_a_rigid_or_vacuum_floor_on_krakenc_gets_the_table_window(
            self, kind):
        c_low, c_high, engine = self._window(
            Kraken(verbose=False), _fluid_floor(kind, surface=_ice()))
        assert engine.backend == 'krakenc'
        assert c_high == (pytest.approx(10.0 * 1500.0),)
        assert c_low == pytest.approx(1500.0)

    def test_leaky_modes_and_a_pinned_value_win(self):
        # leaky: 10 x the fastest speed in the profile, the 1800 m/s seabed
        assert self._window(Kraken(verbose=False, leaky_modes=True),
                            _pekeris(depth=100.0))[1] == (pytest.approx(18000.0),)
        assert self._window(Kraken(verbose=False, c_high=1700.0),
                            _pekeris(depth=100.0))[1] == (1700.0,)

    def test_every_profile_of_a_multi_profile_deck_gets_its_own_window(
            self, tmp_path):
        from uacpy.core.bottom import Bottom
        bottom = Bottom.from_halfspaces(
            np.array([0.0, 3000.0]), sound_speed=np.array([1600.0, 1900.0]),
            density=np.array([1.5, 1.8]), attenuation=np.array([0.5, 0.5]),
            acoustic_type='half-space')
        env = Environment(name='rdwin', bathymetry=100.0, ssp=1500.0,
                          bottom=bottom)
        model = Kraken(verbose=False, n_segments=2, work_dir=tmp_path,
                       cleanup=False)
        launch = model.run_settings(env, _SRC_30(),
                                    _RCV_FAR()).engine.launches[0]
        assert launch.c_high == (pytest.approx(1680.0), pytest.approx(1995.0))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            field = model.run(env, _SRC_30(), _RCV_FAR())
        windows = [line.split() for line in
                   (tmp_path / 'kfield.env').read_text().splitlines()
                   if len(line.split()) == 2 and line.split()[0] == '0.0']
        assert [float(w[1]) for w in windows] == [1680.0, 1995.0]
        assert max(field.run_settings.engine.launches[0].c_high) == pytest.approx(1995.0)



def _shoaling(end_depth, start_depth=200.0, length=4000.0):
    """A 25 Hz-scale shoaling track over a 1700 m/s, 0.5 dB/wavelength
    half-space (the ASA wedge's seabed)."""
    return Environment(
        name='shoal', ssp=[(0.0, 1500.0), (start_depth, 1500.0)],
        bathymetry=uacpy.Bathymetry(ranges=[0.0, length],
                                    depths=[start_depth, end_depth]),
        bottom=BoundaryProperties(acoustic_type='half-space', sound_speed=1700.0,
                                  density=1.5, attenuation=0.5))


class TestACoupledFieldThatLeavesTheAdiabaticOneWarns:
    """A coupled run also sums its ``.mod`` adiabatically and warns past
    ``_launch._COUPLED_GAP_WARN_DB`` (10 dB): field.exe's coupled projection
    is not energy-conserving on modes at and above the half-space speed."""

    @staticmethod
    def _gap_warnings(cb, attenuation, frequency):
        env = Environment(
            name='flat', ssp=[(0.0, 1500.0), (101.0, 1500.0)],
            bathymetry=uacpy.Bathymetry(ranges=[0.0, 10000.0],
                                        depths=[100.0, 100.04]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=cb, density=1.8,
                                      attenuation=attenuation))
        with recorded_warnings() as caught:
            Kraken(verbose=False, mode_coupling='coupled',
                   n_segments=401).compute_tl(
                env, Source(depths=50.0, frequencies=frequency),
                Receiver(depths=[25.0, 75.0],
                         ranges=np.arange(100.0, 10001.0, 10.0)))
        return [w for w in caught
                if 'coupled-mode field is' in str(w.message)]

    def test_identical_profiles_that_lose_the_field_warn(self):
        # 401 identical profiles 25 m apart over 1600 m/s, 30 Hz: measured
        # 62 dB quieter than the adiabatic sum of the same modes
        hits = self._gap_warnings(1600.0, 0.02, 30.0)
        assert len(hits) == 1
        assert issubclass(hits[0].category, NumericsWarning)
        assert 'quieter' in str(hits[0].message)

    def test_identical_profiles_that_keep_the_field_do_not_warn(self):
        # 1650 m/s, 0.01 dB/wavelength, 50 Hz: measured 3.3 dB, the largest
        # gap of the runs that are right or merely approximate
        assert self._gap_warnings(1650.0, 0.01, 50.0) == []


class TestLeakyModesOnARangeDependentDeck:
    """``leaky_modes=True`` searches up to ``_window._LEAKY_C_HIGH_FACTOR`` x
    the fastest speed, not 1e9: at 1e9 KRAKENC kept no mode on a multi-profile
    deck, whose 0.1 m padding layer it cannot search through."""

    SRC = Source(depths=100.0, frequencies=25.0)
    RCV = Receiver(depths=[30.0], ranges=np.array([500.0, 1500.0, 3000.0]))

    def _tl(self, env, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.asarray(Kraken(verbose=False, **kw).compute_tl(
                env, self.SRC, self.RCV).dB, dtype=float).ravel()

    @pytest.mark.parametrize('mode_coupling', ['adiabatic', 'coupled'])
    def test_two_profiles_solve_and_do_not_depend_on_the_window(
            self, mode_coupling):
        env = _shoaling(150.0, length=2000.0)
        leaky = self._tl(env, leaky_modes=True, n_segments=2,
                         mode_coupling=mode_coupling)
        wide = self._tl(env, backend='krakenc', c_high=1e5, n_segments=2,
                        mode_coupling=mode_coupling)
        assert np.all(np.isfinite(leaky))
        np.testing.assert_allclose(leaky, wide, atol=0.05)
        # the window leaky_modes used to take
        with pytest.raises(ModelExecutionError, match='KRAKENC kept no modes'):
            self._tl(env, backend='krakenc', c_high=1e9, n_segments=2,
                     mode_coupling=mode_coupling)

    def test_a_profile_with_no_mode_is_refused_with_its_range(self):
        # measured: the 25 Hz leaky search solves every profile down to 50 m
        # and comes back empty at 20 m, where field.exe then returned NaN
        assert np.all(np.isfinite(self._tl(_shoaling(50.0), leaky_modes=True)))
        with pytest.raises(ModelExecutionError,
                           match=r'no mode in profile 13 of 13 \(r = 4000 m, '
                                 r'seafloor 20 m\)'):
            self._tl(_shoaling(20.0), leaky_modes=True)


class TestASegmentShallowerThanTheDepthQuantum:
    """A segment's seafloor is written at 0.1 m resolution; under 0.05 m it
    rounds to 0.0 m and the refusal names the bathymetry, not the SSP."""

    def test_a_sub_quantum_apex_is_refused_and_a_quantum_one_is_kept(self):
        from uacpy.models.kraken._segments import segments_at_ranges
        with pytest.raises(ConfigurationError, match='rounds to 0.0 m'):
            segments_at_ranges(_shoaling(0.04), [0.0, 4000.0])
        segments = segments_at_ranges(_shoaling(0.06), [0.0, 4000.0])
        assert segments[-1][1].depth == pytest.approx(0.1)


class TestTheSettingsRecordTheDeck:
    """``run_settings(...).engine`` holds every value a deck is written
    from, round-trips, and the deck the binary reads carries them."""

    def test_the_settings_round_trip_and_pickle(self):
        import pickle
        from uacpy.models import RunSettings
        settings = Kraken(verbose=False).run_settings(
            _pekeris(depth=100.0), _SRC_30(), _RCV_FAR(),
            run_mode=RunMode.BROADBAND, frequencies=[45.0, 50.0, 55.0])
        assert RunSettings.from_dict(settings.to_dict()) == settings
        assert pickle.loads(pickle.dumps(settings)) == settings
        assert 'launch 0' in repr(settings)

    def test_the_deck_carries_the_settings(self, tmp_path):
        model = Kraken(verbose=False, work_dir=tmp_path, cleanup=False)
        settings = model.run_settings(_pekeris(depth=100.0), _SRC_30(),
                                      _RCV_FAR())
        result = model.run(_pekeris(depth=100.0), _SRC_30(), _RCV_FAR())
        launch = settings.engine.launches[0]
        deck = (tmp_path / 'kfield.env').read_text().splitlines()
        assert [0.0, launch.c_high[0]] in [
            [float(v) for v in line.split()] for line in deck
            if len(line.split()) == 2 and line.split()[0] == '0.0']
        assert f"{launch.rmax_m / 1000.0:.6f}" in deck
        assert result.run_settings.engine == settings.engine
        assert result.run_settings.engine.launches[0].rmax_m == launch.rmax_m

    def test_no_deck_asks_for_krakencs_random_restarts(self, tmp_path):
        """TopOpt(5:5) '.' makes KRAKENC restart its root finder from an
        unseeded RANDOM_NUMBER, so repeated runs differ (measured up to
        9.4 dB on a range-dependent elastic seabed); every deck keeps the
        blank."""
        for env in (_pekeris(depth=100.0), _hard_elastic()):
            kraken = Kraken(verbose=False)
            launch = kraken.run_settings(env, _SRC_30(),
                                         _RCV_FAR()).engine.launches[0]
            deck = tmp_path / 'deck.env'
            _launch.write_modes_deck(deck, env, _SRC_30(), _RCV_FAR(),
                                     launch, interp_ssp=kraken.interp_ssp)
            assert deck.read_text().splitlines()[3][5] == ' '


class TestAModeCountThatDropsAsTheMeshIsRefinedWarns:
    """RA-WAVE-17: KRAKENC loses 1-3 of this seabed's 12 modes at n_mesh
    200, 400, 1000, 2000 and 4000 at 100 Hz (2.9 dB median off Scooter).
    A krakenc run of an elastic problem is solved again on AT's coarsest
    accepted mesh, and one that kept fewer modes than that solve warns."""

    @pytest.mark.parametrize('n_mesh, warns', [(200, True), (250, False),
                                               (0, False)])
    def test_a_mesh_that_loses_modes_warns(self, n_mesh, warns):
        with recorded_warnings() as caught:
            Kraken(verbose=False, n_mesh=n_mesh).run(
                _hard_elastic(), Source(depths=30.0, frequencies=100.0),
                _RCV_FAR())
        hits = [w for w in caught if 'mode count dropped as its mesh'
                in str(w.message)]
        assert bool(hits) is warns

    def test_a_krakenc_run_on_an_elastic_problem_is_checked_on_the_floor_mesh(
            self):
        # AT's floor for 100 m at 50 Hz, meshed at 20 points per 1500 m/s
        # wavelength: max(int(20 * 100 * 50 / 1500), 10) // 2 = 33.
        for n_mesh, check in ((300, 33), (0, 33), (34, 33), (33, None)):
            engine = Kraken(verbose=False, n_mesh=n_mesh).run_settings(
                _hard_elastic(), _SRC_30(), _RCV_FAR()).engine
            assert engine.launches[0].check_n_mesh == check, n_mesh
        fluid = Kraken(verbose=False, backend='krakenc').run_settings(
            _pekeris(depth=100.0), _SRC_30(), _RCV_FAR()).engine
        assert fluid.launches[0].check_n_mesh is None
        table_window = Kraken(verbose=False).run_settings(
            _fluid_floor('rigid', surface=_ice()), _SRC_30(),
            _RCV_FAR()).engine
        assert table_window.launches[0].check_n_mesh is None

    @pytest.mark.parametrize('kept, checked, warns', [(11, 12, True),
                                                      (12, 12, False),
                                                      (12, 11, False)])
    def test_a_count_that_drops_as_the_mesh_is_refined_warns(
            self, monkeypatch, kept, checked, warns):
        from uacpy.models.kraken import _model, _modes
        real = _modes.read_modes

        def _counted(root, **kw):
            modes = real(root, **kw)
            n = checked if root.endswith('mcheck') else kept
            return modes.first_n(n) if n < modes.n_modes else modes
        # The run's own .mod is read in _modes, the check solve's in _model.
        monkeypatch.setattr(_modes, 'read_modes', _counted)
        monkeypatch.setattr(_model, 'read_modes', _counted)
        with recorded_warnings() as caught:
            Kraken(verbose=False, n_mesh=300).run(
                _hard_elastic(), Source(depths=30.0, frequencies=100.0),
                _RCV_FAR())
        hits = [w for w in caught if 'mode count dropped as its mesh'
                in str(w.message)]
        assert bool(hits) is warns


def test_the_check_solves_prt_warnings_stay_out_of_the_run(monkeypatch):
    """The coarser check solve's own ``.prt`` warnings describe the check,
    not the run: only the run's modes solve and field.exe report theirs."""
    def _announce(self, work_dir, base_name):
        warnings.warn(f"prt of {base_name}", UserWarning)
    monkeypatch.setattr(Kraken, '_warn_on_prt_warnings', _announce)
    with recorded_warnings() as caught:
        Kraken(verbose=False, n_mesh=300).run(
            _hard_elastic(), Source(depths=30.0, frequencies=100.0),
            _RCV_FAR())
    said = [str(w.message) for w in caught if 'prt of' in str(w.message)]
    assert 'prt of kfield' in said
    assert 'prt of mcheck' not in said


class TestTheNearFieldIsAnnounced:
    """RA-WAVE-7: an unpinned window that drops a path the receivers need is
    said by ``run`` and ``run_settings``, never by ``validate_inputs``."""

    @staticmethod
    def _geometry(r_min):
        # 100 m Pekeris guide, c_high = 1.05 * 1800 = 1890 m/s: the cut is
        # arccos(1500/1890) = 37.47 deg. Source 50 m, receiver 75 m: the
        # surface-reflected path spans 125 m of depth.
        return (_pekeris(depth=100.0), Source(depths=50.0, frequencies=200.0),
                Receiver(depths=np.array([75.0]),
                         ranges=np.array([r_min, 2000.0])))

    @pytest.mark.parametrize('r_min, warns', [(160.0, True), (165.0, False)])
    def test_the_notice_sits_on_the_cut(self, r_min, warns):
        # atan(125 / 163.0) = 37.48 deg: the boundary lies between the two.
        env, src, rcv = self._geometry(r_min)
        with recorded_warnings() as caught:
            Kraken(verbose=False).run_settings(env, src, rcv)
        hits = [w for w in caught if 'keeps modes carrying paths up to'
                in str(w.message)]
        assert bool(hits) is warns

    def test_validate_inputs_and_a_pinned_window_are_silent(self):
        env, src, rcv = self._geometry(100.0)
        with recorded_warnings() as caught:
            Kraken(verbose=False).validate_inputs(env, src, rcv)
            Kraken(verbose=False, c_high=1890.0).run_settings(env, src, rcv)
        assert not [w for w in caught if 'keeps modes carrying paths'
                    in str(w.message)]


class TestModesTakeEverySourceDepth:
    """JRN-37: ``run(run_mode=MODES)`` takes a multi-depth Source as
    ``compute_modes`` does — the modes do not depend on the source depth —
    and both tabulate the modes at every source depth."""

    def test_run_and_compute_modes_agree_on_a_two_depth_source(self):
        env = _pekeris(depth=100.0)
        src = Source(depths=[30.0, 61.3], frequencies=100.0)
        grid = Receiver(depths=np.linspace(0.0, 100.0, 101), ranges=[0.0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            by_run = Kraken(verbose=False).run(env, src, grid,
                                               run_mode=RunMode.MODES)
            by_wrapper = Kraken(verbose=False, mode_depths=np.linspace(
                0.0, 100.0, 101)).compute_modes(env, src)
        assert by_run.run_settings.depth_loop == 'engine'
        assert np.any(np.isclose(by_run.depths, 61.3))
        assert np.any(np.isclose(by_wrapper.depths, 61.3))
        np.testing.assert_array_equal(by_run.k, by_wrapper.k)


class TestTheModalSumDividesByTheSourcesDensity:
    """ARCH-10 / RA-IO-5: ``modal_pressure_field`` takes ``rho(z_s)`` from
    the mode set's own medium table, as Kraken's run divides by it, and
    ``read_modes`` records the water density the ``.mod`` carries."""

    @staticmethod
    def _sediment_env():
        return Environment(
            name='buried', bathymetry=100.0, ssp=1500.0,
            bottom=SeabedColumn(
                layers=[SedimentLayer(thickness=30.0, sound_speed=1600.0,
                                      density=1.8, attenuation=0.2)],
                halfspace=BoundaryProperties(
                    acoustic_type='half-space', sound_speed=1800.0,
                    density=2.0, attenuation=0.5)))

    def test_a_buried_source_matches_the_run(self):
        env = self._sediment_env()
        src = Source(depths=110.0, frequencies=100.0)
        rcv = Receiver(depths=np.array([20.0, 50.0]),
                       ranges=np.array([1000.0, 2000.0, 5000.0]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            run = Kraken(verbose=False).run(env, src, rcv)
            modes = Kraken(verbose=False).compute_modes(env, src)
        summed = modes.modal_pressure_field(source_depth=110.0,
                                            receiver_depths=rcv.depths,
                                            ranges=rcv.ranges)
        # Dividing by the water density instead is 20 log10(1.8/1.027) =
        # 4.9 dB off.
        assert np.median(np.abs(np.asarray(run.dB)
                                - np.asarray(summed.dB))) < 0.05

    def test_the_density_on_an_interface_is_the_upper_mediums(self):
        from uacpy.core.results import MediaTable
        table = MediaTable(water_density=1.0, tops=[0.0, 100.0, 130.0],
                           densities=[1.0, 1.8, 1.9], bottom_depth=150.0,
                           halfspace_density=2.0)
        assert table.density_at(100.0) == 1.0
        assert table.density_at(100.001) == 1.8
        assert table.density_at(130.0) == 1.8
        assert table.density_at(150.001) == 2.0
        assert MediaTable(water_density=1.027).density_at(120.0) == 1.027

    def test_read_modes_records_the_files_water_density(self, tmp_path):
        from uacpy.io import read_modes
        env = Environment(name='rw1', bathymetry=100.0, ssp=1500.0,
                          water_density=1.0,
                          bottom=BoundaryProperties(
                              acoustic_type='half-space',
                              sound_speed=1700.0, density=1.5,
                              attenuation=0.5))
        src = Source(depths=30.0, frequencies=100.0)
        rcv = Receiver(depths=np.array([50.0]),
                       ranges=np.array([1000.0, 2000.0, 5000.0]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Kraken(verbose=False, work_dir=tmp_path, cleanup=False,
                   mode_depths=np.linspace(0.0, 100.0, 101)
                   ).compute_modes(env, src)
            run = Kraken(verbose=False).run(env, src, rcv)
        modes = read_modes(tmp_path / 'modes.mod')
        assert modes.media.water_density == 1.0
        summed = modes.modal_pressure_field(source_depth=30.0,
                                            receiver_depths=rcv.depths,
                                            ranges=rcv.ranges)
        # The package default 1.027 instead is 20 log10(1.027) = 0.23 dB off.
        assert np.max(np.abs(np.asarray(run.dB) - np.asarray(summed.dB))) \
            < 0.02


@pytest.mark.parametrize('floor, match', [
    ('rigid', 'elastic sediment layer over a rigid floor'),
    ('vacuum', 'elastic sediment layer over a vacuum floor'),
    ('fluid', 'elastic sediment layer over a fluid halfspace'),
])
def test_an_elastic_layer_is_refused_over_the_floor_it_sits_on(floor, match):
    """The refusal names the floor under the elastic layer: a rigid or
    vacuum floor is not a fluid half-space (krakenc runs there, 0.8-1.7 dB
    off Scooter; over a fluid half-space it does not converge)."""
    halfspace = (BoundaryProperties(acoustic_type=floor) if floor != 'fluid'
                 else BoundaryProperties(acoustic_type='half-space',
                                         sound_speed=2000.0, density=2.0,
                                         attenuation=0.3))
    env = Environment(name='layer-over-floor', bathymetry=100.0, ssp=1500.0,
                      bottom=SeabedColumn(
                          layers=[SedimentLayer(thickness=20.0,
                                                sound_speed=1800.0,
                                                shear_speed=400.0,
                                                density=1.8, attenuation=0.2)],
                          halfspace=halfspace))
    with pytest.raises(UnsupportedFeatureError, match=match):
        Kraken(verbose=False).validate_inputs(env, _SRC_30(), _RCV_FAR())


def test_the_run_and_the_modes_divide_by_one_density_rule():
    """ARCH-10: base ``_source_density`` (the run path) is the rule
    ``Modes.modal_pressure_field`` applies to a mode set's table
    (``medium_density_at``): water down to the seafloor, each layer below
    its top, the half-space below the stack, an interface depth in the upper
    medium."""
    from uacpy.models._conventions import _source_density
    from uacpy.core.bottom import medium_density_at
    env = Environment(
        name='stack', bathymetry=100.0, ssp=1500.0, water_density=1.0,
        bottom=SeabedColumn(
            layers=[SedimentLayer(thickness=30.0, sound_speed=1600.0,
                                  density=1.8, attenuation=0.2),
                    SedimentLayer(thickness=20.0, sound_speed=1700.0,
                                  density=1.9, attenuation=0.2)],
            halfspace=BoundaryProperties(acoustic_type='half-space',
                                         sound_speed=1800.0, density=2.0,
                                         attenuation=0.5)))
    table = ([0.0, 100.0, 130.0], [1.0, 1.8, 1.9], 150.0, 2.0)
    for z, rho in ((50.0, 1.0), (100.0, 1.0), (100.5, 1.8), (130.0, 1.8),
                   (130.5, 1.9), (150.0, 1.9), (150.5, 2.0)):
        assert _source_density(env, z) == rho, z
        assert medium_density_at(z, *table) == rho, z


def test_the_steep_path_rule_is_one_for_both_engines():
    """Kraken's near-field notice and Scooter's read the same geometry
    (``models/_window.steep_path_cut``)."""
    from uacpy.models._window import steep_path_cut
    cut, steepest = steep_path_cut(1500.0, 1890.0, [50.0], [75.0],
                                   [160.0, 2000.0])
    assert cut == pytest.approx(np.degrees(np.arccos(1500.0 / 1890.0)))
    assert steepest == pytest.approx(np.degrees(np.arctan(125.0 / 160.0)))
    assert steep_path_cut(1500.0, 1890.0, [50.0], [75.0], [165.0]) is None
    assert steep_path_cut(1500.0, 1500.0, [50.0], [75.0], [1.0]) is None


@pytest.mark.requires_binary
class TestKraken:
    """Tests for Kraken model."""

    def test_kraken_compute_modes(self, simple_env, source):
        """``compute_modes`` with no cap returns every mode kraken.exe found.

        The capped case is the next test: ``n_modes`` is optional and maps to
        the FLP ``MLimit`` field.exe honours.
        """
        kraken = Kraken(verbose=False)
        modes = kraken.compute_modes(env=simple_env, source=source)

        assert isinstance(modes, Modes)
        assert modes.k is not None
        assert modes.phi is not None
        assert len(modes.k) > 0

    def test_kraken_n_modes_clips_output(self, simple_env, source):
        """``n_modes`` caps the number of returned modes from Kraken.

        The 100 m / 100 Hz guide carries well over 3 propagating modes, so
        the cap must deliver exactly 3 while the uncapped run returns more —
        a ``<= 3`` alone is satisfied by a solver that found nothing."""
        kraken = Kraken(verbose=False)
        uncapped = kraken.compute_modes(env=simple_env, source=source)
        capped = kraken.compute_modes(env=simple_env, source=source, n_modes=3)
        assert len(uncapped.k) > 3
        assert len(capped.k) == 3
        assert capped.metadata.get('n_modes_requested') == 3

    def test_kraken_modes_have_wavenumbers(self, simple_env, source):
        """Test that computed modes have valid wavenumbers."""
        kraken = Kraken(verbose=False)
        modes = kraken.compute_modes(env=simple_env, source=source)

        k = modes.k
        assert len(k) > 0
        # Real part of wavenumber should be positive for propagating modes
        # Some modes may have k≈0 (non-propagating), which is valid
        k_real = np.real(k)
        propagating_modes = k_real > 1e-6  # Threshold for propagating vs non-propagating
        assert np.any(propagating_modes), "Should have at least one propagating mode"
        # All propagating modes should have positive wavenumbers
        assert np.all(k_real[propagating_modes] > 0)


@pytest.mark.requires_binary
class TestKrakenInFieldMode:
    """``Kraken`` in its field mode: a single class covers both binaries, so
    asking it for TL (rather than modes) is what makes it run field.exe after
    kraken.exe."""

    def test_kraken_field_mode_compute_tl(self, simple_env, source, receiver_small):
        """``compute_tl`` returns a full depth x range grid, not a mode set."""
        kf = Kraken(verbose=False)
        result = kf.compute_tl(env=simple_env, source=source, receiver=receiver_small)

        assert isinstance(result, Field)
        assert result.shape == (len(receiver_small.depths), len(receiver_small.ranges))


class TestABiologicalLayerBetweenSspNodesAttenuates:
    """A Biological layer 30-70 m in a 100 m isovelocity guide whose SSP has
    nodes at 0 and 100 m only. KRAKEN evaluates the law at SSP nodes
    (``misc/AttenMod.f90:103-104``), so without the edge node pairs the
    writers add, the layer applied 0.000 dB; with them, 8.78 dB at the
    300 Hz resonance over 9-11 km against RAM's 9.00 (RAM samples the layer
    edges itself)."""

    @staticmethod
    def _power(absorption, interp_ssp='linear'):
        env = Environment(
            name='bio', bathymetry=100.0,
            ssp=[(0.0, 1500.0), (100.0, 1500.0)],
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.8,
                                      attenuation=0.5),
            absorption=absorption)
        rcv = Receiver(depths=np.array([20.0, 40.0, 60.0, 80.0]),
                       ranges=np.linspace(9000.0, 11000.0, 21))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            field = Kraken(verbose=False, interp_ssp=interp_ssp).run(
                env, Source(depths=50.0, frequencies=300.0), rcv,
                run_mode=RunMode.COHERENT_TL)
        return np.nanmean(np.abs(np.asarray(field.data)) ** 2)

    def test_the_layer_attenuates_as_ram_does(self):
        mid = uacpy.Biological(layers=[(30.0, 70.0, 300.0, 4.0, 0.125)])
        dtl = 10.0 * np.log10(self._power(None) / self._power(mid))
        assert abs(dtl - 9.0) < 0.5, dtl

    def test_a_pchip_profile_is_refused_before_any_deck(self):
        mid = uacpy.Biological(layers=[(30.0, 70.0, 300.0, 4.0, 0.125)])
        with pytest.raises(ConfigurationError, match="interp_ssp='linear'"):
            self._power(mid, interp_ssp='pchip')


class TestEveryRouteRunsItsOwnSteps:
    """M-23: a Kraken run's stage-4/5 hooks dispatch on its route through one
    table, each route's launch, read and result steps side by side."""

    def test_the_step_table_covers_every_route(self):
        from uacpy.models.kraken import _model, _settings
        assert set(_model._ROUTE_STEPS) == set(_settings._ROUTES)

    @pytest.mark.parametrize('mode, freqs, range_dependent, route', [
        (RunMode.MODES, None, False, 'modes'),
        (RunMode.COHERENT_TL, None, False, 'field'),
        (RunMode.BROADBAND, [90.0, 100.0, 110.0], False, 'band'),
        (RunMode.BROADBAND, [100.0], False, 'band_bin'),
        (RunMode.BROADBAND, [90.0, 100.0, 110.0], True, 'band_by_frequency'),
    ])
    def test_only_the_modes_route_launches_without_a_field_option(
            self, mode, freqs, range_dependent, route):
        """The modes read was keyed on a launch having no field option; the
        table keys it on the route. The two agree on every route."""
        bathymetry = ([(0.0, 100.0), (3000.0, 110.0)] if range_dependent
                      else 100.0)
        env = Environment(
            bathymetry=bathymetry, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.8,
                                      attenuation=0.5))
        kw = {} if freqs is None else {'frequencies': freqs}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            settings = Kraken(verbose=False).run_settings(
                env, Source(depths=50.0, frequencies=100.0),
                Receiver(depths=[20.0, 60.0], ranges=[1000.0, 2000.0]),
                mode, **kw)
        assert settings.engine.route == route
        assert [launch.field_option is None
                for launch in settings.engine.launches] == (
            [route == 'modes'] * len(settings.engine.launches))

    @pytest.mark.requires_binary
    @pytest.mark.parametrize('range_dependent, said', [(True, 1),
                                                       (False, 0)])
    def test_a_band_of_launches_announces_its_cost_once(
            self, capsys, range_dependent, said):
        """A range-dependent band (one launch per bin) projects its cost
        from its first bin, once; a band on one deck says nothing."""
        bathymetry = ([(0.0, 100.0), (3000.0, 110.0)] if range_dependent
                      else 100.0)
        env = Environment(
            bathymetry=bathymetry, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.8,
                                      attenuation=0.5))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Kraken(verbose=True).run(
                env, Source(depths=50.0, frequencies=100.0),
                Receiver(depths=[20.0, 60.0], ranges=[1000.0, 2000.0]),
                RunMode.BROADBAND, frequencies=[90.0, 100.0, 110.0])
        out = capsys.readouterr().out
        assert out.count('separate mode solves and stacks them') == said



def test_the_steep_path_notice_cuts_at_the_fastest_water():
    """The shared notice (``models/_window.steep_path_notice``) cuts at the
    FASTEST water speed, the tightest cut. 1480-1520 m/s water, c_high
    1600 m/s: arccos(1520/1600) = 18.19 deg, arccos(1480/1600) = 22.33 deg;
    a 19.98 deg surface-reflected path (source and receiver at 50 m, 275 m
    range) is announced, and one at 17.9 deg (310 m) is not."""
    from uacpy.core.ssp import SoundSpeedProfile
    from uacpy.models._window import steep_path_notice
    env = Environment(bathymetry=100.0, ssp=SoundSpeedProfile(
        depths=[0.0, 100.0], sound_speed=[1480.0, 1520.0]))
    kw = dict(model_name='Kraken', kept='modes carrying paths',
              summed_in='modal sum', evidence='e', remediation='r')

    def notice(rng):
        return steep_path_notice(
            env, Source(depths=50.0, frequencies=100.0),
            Receiver(depths=[50.0], ranges=[rng]), 1600.0, **kw)
    said = notice(275.0)
    assert said is not None and '18.2' in said.note
    assert said.message.startswith('Kraken: the auto-derived c_high = '
                                   '1600.0 m/s keeps modes carrying paths')
    assert notice(310.0) is None



class TestModesRunThroughTheProtocol:
    """``compute_modes`` is ``run(env, source, None, run_mode=MODES)`` with
    its mode cap as the call's request (M-21, D24): no Receiver is
    fabricated, no model copy is made, and ``run_settings`` with the same
    arguments previews the run."""

    @staticmethod
    def _carriers():
        return (_pekeris(depth=100.0),
                Source(depths=50.0, frequencies=100.0))

    def test_the_preview_is_the_settings_compute_modes_runs(self):
        env, src = self._carriers()
        model = Kraken(verbose=False)
        preview = model.run_settings(env, src, None, run_mode=RunMode.MODES)
        ran = model.compute_modes(env, src).run_settings
        np.testing.assert_array_equal(
            preview.engine.launches[0].tabulation_depths,
            ran.engine.launches[0].tabulation_depths)
        assert preview.engine.rmax_origin == ran.engine.rmax_origin

    def test_the_mode_cap_is_the_calls_not_the_models(self):
        env, src = self._carriers()
        model = Kraken(verbose=False)
        modes = model.compute_modes(env, src, n_modes=3)
        assert modes.run_settings.engine.n_modes == 3
        assert len(modes.k) == 3
        assert model.n_modes is None
        assert model.run_settings(
            env, src, None, run_mode=RunMode.MODES).engine.n_modes is None

    @pytest.mark.parametrize('n_modes, refused', [(0, True), (1, False)])
    def test_a_cap_below_one_is_refused(self, monkeypatch, n_modes,
                                        refused):
        env, src = self._carriers()
        model = Kraken(verbose=False)
        asked = []
        monkeypatch.setattr(
            model, '_run_call',
            lambda env, source, receiver, call: asked.append(
                (receiver, call.engine_request)))
        if refused:
            with pytest.raises(ConfigurationError, match='>= 1'):
                model.compute_modes(env, src, n_modes=n_modes)
            assert not asked
        else:
            model.compute_modes(env, src, n_modes=n_modes)
            ((receiver, request),) = asked
            assert receiver is None and request.n_modes == n_modes

    def test_no_receiver_is_refused_outside_modes(self):
        env, src = self._carriers()
        with pytest.raises(ConfigurationError, match='receiver=NoneType'):
            Kraken(verbose=False).run_settings(env, src, None)

    def test_a_coarse_density_is_a_settings_notice(self):
        env, src = self._carriers()
        model = Kraken(verbose=False, mode_points_per_meter=0.5)
        with pytest.warns(UserWarning, match='points per wavelength') as w:
            settings = model.run_settings(env, src, None,
                                          run_mode=RunMode.MODES)
        assert sum('points per wavelength' in str(x.message)
                   for x in w) == 1
        assert any('mode_points_per_meter 0.5' in (n.note or '')
                   for n in settings.engine.notices)

    @pytest.mark.parametrize('grid', [[-1.0, 10.0], [10.0, 5.0], []])
    def test_a_pinned_grid_is_held_to_a_receivers_rules(self, grid):
        with pytest.raises(ConfigurationError, match='mode_depths'):
            Kraken(mode_depths=np.array(grid))

    def test_a_valid_pinned_grid_is_tabulated_verbatim(self):
        env, src = self._carriers()
        grid = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        engine = Kraken(mode_depths=grid).run_settings(
            env, src, None, run_mode=RunMode.MODES).engine
        np.testing.assert_array_equal(engine.launches[0].tabulation_depths,
                                      grid)


def test_a_uniform_francois_garrison_profile_gives_the_one_row_field():
    """One water row and a uniform profile both put the formula at each
    node's depth into the water rows, so Kraken runs the same deck and
    returns the same field, to the bit."""
    from uacpy.core.absorption import FrancoisGarrison
    row = FrancoisGarrison(12.0, 34.5, 8.0)
    column = FrancoisGarrison([12.0, 12.0], [34.5, 34.5], 8.0,
                              depths=[0.0, 100.0])
    fields = [np.asarray(Kraken(verbose=False).run(
        Environment(name='fg', bathymetry=100.0, ssp=1500.0, bottom='sand',
                    absorption=law),
        Source(depths=30.0, frequencies=4000.0),
        Receiver(depths=[20.0, 60.0],
                 ranges=np.array([500.0, 1500.0, 3000.0])),
        run_mode=RunMode.COHERENT_TL).data) for law in (row, column)]
    assert np.all(np.isfinite(fields[0]))
    np.testing.assert_array_equal(fields[1], fields[0])
