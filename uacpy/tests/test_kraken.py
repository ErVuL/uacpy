"""Normal-mode tests for ``Kraken``, on both the kraken and krakenc backends.

Backend selection, broadband sweeps, complex eigenvalues, the attenuation
unit and mode sampling — and, at the end, what the model does at the edge of
its own validity: a duct whose mode set is entirely non-trapped, and the
dispatch premise the vendored Fortran either supports or contradicts.
"""

import math
import warnings
from pathlib import Path

import pytest
import numpy as np
import uacpy

from uacpy.core.exceptions import (
    ConfigurationError, ModelExecutionError, UnsupportedFeatureError,
)
from uacpy.core.environment import Bottom, SeabedColumn, SedimentLayer
from uacpy.core.results import Field, Modes
from uacpy.models import Kraken
from uacpy.models.base import RunMode
from uacpy.core import Environment, BoundaryProperties, Source, Receiver

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

    def test_backend_auto_dispatch_fluid_kraken_elastic_krakenc(self):
        assert Kraken(verbose=False)._select_kraken_exe(
            self._fluid()).name == 'kraken.exe'
        assert Kraken(verbose=False)._select_kraken_exe(
            self._elastic()).name == 'krakenc.exe'

    def test_backend_override_beats_auto_dispatch(self):
        assert Kraken(verbose=False, backend='krakenc')._select_kraken_exe(
            self._fluid()).name == 'krakenc.exe'
        assert Kraken(verbose=False, backend='kraken')._select_kraken_exe(
            self._fluid()).name == 'kraken.exe'

    def test_leaky_modes_forces_krakenc_even_on_a_fluid_env(self):
        # kraken.md §5 "so the solver attempts leaky modes": leaky
        # eigenvalues are genuinely complex, so
        # leaky_modes=True dispatches to krakenc.exe regardless of the
        # environment's own (fluid) dispatch. Resolution only — no run.
        assert Kraken(verbose=False, leaky_modes=True)._select_kraken_exe(
            self._fluid()).name == 'krakenc.exe'

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

    def test_fluid_env_delegates_to_kraken(self):
        env = Environment(
            name='fl', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.5))
        assert Kraken()._c_low_for(env) == 0.0

    def test_explicit_c_low_always_wins(self):
        assert Kraken(c_low=1234.0)._c_low_for(self._elastic_env()) == 1234.0

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
            data=np.array([[1500.0, 1480.0, 1450.0]] * 3),
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
        assert Kraken()._c_low_for(env) == pytest.approx(1450.0)

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
        segments, _, _, _max_total_depth = model._segment_env_for_field(
            model._project_environment(env))
        assert model._multi_profile_n_mesh(segments, 50.0) == 0

        out = tmp_path / 'auto.env'
        write_multi_profile_env(out, segments, self._SRC(), self._RCV(),
                                n_mesh=model._multi_profile_n_mesh(
                                    segments, 50.0))
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
        segments, _, _, _max_total_depth = model._segment_env_for_field(
            model._project_environment(env))
        out = tmp_path / 'pinned.env'
        write_multi_profile_env(out, segments, self._SRC(), self._RCV(),
                                n_mesh=model._multi_profile_n_mesh(
                                    segments, 50.0))
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

    def test_fluid_bottom_pinned_floor_meshes_on_cp(self):
        """For a fluid seabed the pinned-mesh floor is measured on the
        compressional speed — the shear term must not inflate it."""
        from uacpy.io.oalib_writer import at_mesh_floor
        model = Kraken(verbose=False)
        segments, _, _, _max_total_depth = model._segment_env_for_field(
            model._project_environment(self._env([0.0, 0.0, 0.0, 0.0])))
        media = model._multi_profile_media(segments)
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
        tl = np.asarray(result.db)
        finite = tl[np.isfinite(tl)]
        assert finite.size, "no finite TL returned"
        assert finite.max() < 200.0, (
            f"max TL {finite.max():.1f} dB — not a physical waterborne field")
        # TL must grow with range, not sit at a constant or run backwards.
        at_source_depth = np.asarray(result.at(depth=50.0).db)
        assert at_source_depth[-1] > at_source_depth[4] + 10.0

    @pytest.mark.slow
    def test_uniformly_elastic_bottom_runs(self):
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

    def test_a_short_range_run_refines_the_mesh(self, tmp_path):
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


def _rough_surface(sigma, acoustic_type='vacuum', reflection_file=None):
    """A single-node ``Surface`` carrying ``sigma`` on ``acoustic_type``."""
    from uacpy.core.surface import Surface
    kw = {}
    if reflection_file is not None:
        kw['reflection_file'] = str(reflection_file)
    return Surface(properties=[BoundaryProperties(
        acoustic_type=acoustic_type, roughness=sigma, **kw)])


class TestElasticSeaSurfaceRunsUnderTheCompressionalFloor:
    """An elastic sea surface is accepted, and what makes it work is the
    ``c_low`` floor, not a guard.

    ``krakenc.f90:220-222`` folds ``HSTop%cS`` into ``cMin`` symmetrically
    with the seabed at ``:210-212``, and ``:228-230`` then applies
    ``IF (ElasticFlag) cMin = 0.85 * cMin``. Left to KRAKEN, the search floor
    lands near the ice shear speed and the solver chases the Scholte mode;
    :meth:`Kraken._c_low_for` writes the minimum compressional speed instead.
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
            surface=Surface(properties=[BoundaryProperties(
                acoustic_type='half-space', sound_speed=3500.0,
                shear_speed=1800.0, density=0.9, attenuation=1.0,
                shear_attenuation=2.0)]))

    def test_the_floor_is_the_water_speed_not_the_shear_derived_one(self):
        """Where the clamp lands, computed — not which MAX it is written as.
        ``0.85 * 1800 = 1530`` m/s would sit *above* the 1500 m/s water and
        delete the waterborne modes."""
        floor = Kraken(verbose=False)._c_low_for(self._ice_env())
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
        db = np.asarray(Kraken(verbose=False).compute_tl(
            self._ice_env(), Source(depths=100.0, frequencies=freq),
            Receiver(depths=[150.0], ranges=[5000.0])).db).ravel()
        assert np.isfinite(db).all()
        assert (0.0 < db).all() and (db < 200.0).all(), (
            f"TL {db} at {freq:g} Hz is outside the physical band — a "
            f"Scholte-mode solve reads as several hundred dB")


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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
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
        env.surface = Surface(properties=[BoundaryProperties(
            acoustic_type=acoustic_type, roughness=0.5, **kw)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
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
        # At 7.5 Hz this deck makes kraken.exe abort with a raw Fortran
        # RECL error while writing the empty modes.mod, which uacpy wraps
        # as a typed ModelExecutionError but without the friendlier 'no
        # mode' diagnosis that the clean zero-mode path produces.
        with pytest.raises(ModelExecutionError):
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
            with pytest.raises(ModelExecutionError) as ei:
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
    def test_a_single_frequency_accepts_the_pattern(self, tmp_path):
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

    def test_density_is_sized_on_the_slowest_column_of_a_range_dependent_ssp(self):
        """One tabulation grid is built for the whole multi-profile deck, so
        the density has to clear ~10 pts/wavelength in the slowest water
        anywhere on the track — not just at r = 0. Reading the range-0 column
        of this 1500 → 1000 m/s profile gives 13.3 pts/m against the 20 pts/m
        the block minimum requires, a 0.075 m grid where 0.050 m is needed."""
        from uacpy.core.ssp import SoundSpeedProfile
        from uacpy.models.kraken import MODE_POINTS_PER_WAVELENGTH
        ssp = SoundSpeedProfile(
            depths=np.array([0.0, 100.0, 200.0]),
            data=np.array([[1500.0, 1200.0, 1000.0]] * 3),
            ranges=np.array([0.0, 5000.0, 10000.0]))
        env = uacpy.Environment(
            bathymetry=np.array([[0.0, 200.0], [10000.0, 220.0]]), ssp=ssp,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1600.0,
                density=1.8, attenuation=0.2))
        assert float(ssp.to_pairs()[:, 1].min()) == 1500.0, "range-0 is faster"
        ppm = uacpy.Kraken()._resolve_mode_points_per_meter(env, [2000.0])
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
            env, src, rcv, run_mode=uacpy.RunMode.COHERENT_TL).db)
        kr = np.squeeze(uacpy.Kraken().run(
            env, src, rcv, run_mode=uacpy.RunMode.COHERENT_TL).db)
        assert np.max(np.abs(kr - sc)) < 1.0


class TestModesPathKeepsTheFullEnvContext:
    """``_modes_single_profile`` reduces a range-dependent env to its r=0
    profile for the modes solve. The rebuilt env must carry the original's
    altimetry (so ``_project_environment`` still discloses dropping it),
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
        env = self._rd_env()
        env.data_sources = ('unit-test-source',)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            reduced = Kraken(verbose=False)._modes_single_profile(env)
        assert reduced.altimetry is env.altimetry
        assert reduced.location == env.location
        assert reduced.date == env.date
        assert reduced.data_sources == env.data_sources
        assert not reduced.is_range_dependent or reduced.altimetry is not None

    def test_modes_run_discloses_the_dropped_altimetry(self):
        env = self._rd_env()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            Kraken(verbose=False).compute_modes(
                env, Source(depths=25.0, frequencies=100.0))
        alti = [w for w in caught if 'altimetry' in str(w.message)]
        assert len(alti) == 1, (
            f"altimetry-collapse disclosure emitted {len(alti)} times")


class TestModesPathDisclosesTheCollapseItOverrides:
    """``_modes_single_profile`` samples r = 0 for every range-dependent
    quantity, overriding the configured ``collapse`` methods. That is the
    right physics — a single-profile solve at the source's own waveguide,
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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model._modes_single_profile(env)
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
        assert projected.has_range_dependent_ssp

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

    def test_the_bottom_range_default_is_named_as_dropped(self):
        """``bottom_range='median'`` is a live policy on the field paths and
        a spec default rather than a user choice, so the modes path drops it
        without the user ever asking — which is exactly what needs saying."""
        env = Environment(
            name='rd-bottom-modes', bathymetry=100.0, ssp=1500.0,
            bottom=_rd_layered_bottom(shear_at_r0=0.0, shear_elsewhere=0.0))
        message = self._reduce(Kraken(verbose=False), env)
        assert "collapse['bottom_range']='median'" in message, message

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
        assert model._modes_single_profile(env) is env


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
        with pytest.raises(ConfigurationError) as err:
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
    Pinned by trapping the two dispatch funnels; no binary runs."""

    _ENV = staticmethod(lambda: Environment(
        name='bb_default', bathymetry=100.0, ssp=1500.0,
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1600.0, density=1.8,
                                  attenuation=0.5)))
    _SRC = staticmethod(lambda: Source(depths=50.0, frequencies=100.0))
    _RCV = staticmethod(lambda: Receiver(depths=np.array([50.0]),
                                         ranges=np.array([1000.0])))

    def test_multi_element_frequencies_dispatch_broadband(self, monkeypatch):
        monkeypatch.setattr(
            Kraken, '_compute_broadband_field',
            lambda self, *a, **k: (_ for _ in ()).throw(
                RuntimeError('reached the broadband path')))
        with pytest.raises(RuntimeError, match='reached the broadband path'):
            Kraken(verbose=False).run(
                self._ENV(), self._SRC(), self._RCV(),
                frequencies=np.array([95.0, 105.0]))

    def test_single_element_frequencies_stay_narrowband(self, monkeypatch):
        monkeypatch.setattr(
            Kraken, '_compute_broadband_field',
            lambda self, *a, **k: (_ for _ in ()).throw(
                AssertionError('BROADBAND taken for a 1-element vector')))
        monkeypatch.setattr(
            Kraken, '_compute_field_via_exe',
            lambda self, *a, **k: (_ for _ in ()).throw(
                RuntimeError('reached the narrowband path')))
        with pytest.raises(RuntimeError, match='reached the narrowband path'):
            with warnings.catch_warnings():
                # COHERENT_TL reports the unconsumed frequencies= kwarg.
                warnings.simplefilter('ignore', UserWarning)
                Kraken(verbose=False).run(
                    self._ENV(), self._SRC(), self._RCV(),
                    frequencies=np.array([100.0]))


def test_range_dependent_broadband_raises_before_any_launch(tmp_path,
                                                            monkeypatch):
    """kraken.md §7 "the multi-profile deck has no broadband form"
    / kraken.py ``_write_field_env``:
    ``write_multi_profile_env`` has no broadband form, so a range-dependent
    BROADBAND run raises ``UnsupportedFeatureError`` at deck-writing time
    rather than dropping the frequency vector. The launcher traps prove no
    binary is spent on the refused run."""
    from uacpy.core.exceptions import UnsupportedFeatureError
    model = Kraken(work_dir=tmp_path, cleanup=False)

    def _no_launch(*args, **kwargs):
        raise AssertionError("a binary was launched past the guard")

    monkeypatch.setattr(model, '_run_subprocess', _no_launch)
    monkeypatch.setattr(model, '_run_and_attach_prt', _no_launch)
    env = Environment(
        name='rd_bb', bathymetry=[(0.0, 100.0), (5000.0, 150.0)],
        ssp=[(0.0, 1500.0), (150.0, 1500.0)],
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1800.0, density=1.8,
                                  attenuation=0.3))
    with pytest.raises(UnsupportedFeatureError, match='range-dependent'):
        model.run(env, Source(depths=50.0, frequencies=100.0),
                  Receiver(depths=[50.0], ranges=[1000.0, 3000.0]),
                  run_mode=RunMode.BROADBAND,
                  frequencies=np.array([95.0, 100.0, 105.0]))


class TestRMaxAutoDefaults:
    """kraken.md §5 "`1.05 ×` the outermost receiver range":
    ``rmax_m=None`` resolves to 1.05x the outermost
    receiver range for a narrowband deck and 3x for a broadband sweep — the
    sweep solves every frequency off one Richardson mesh sequence
    (``kraken.f90:80`` exits on ``Error·1000·RMax < 1``), so it gets the
    tighter tolerance as margin. Pinned on the resolved bounds
    ``_write_kraken_env`` returns — the same dict the deck is written from
    and the result metadata reports. Deck writes only; no binary."""

    _SRC = staticmethod(lambda: Source(depths=50.0, frequencies=100.0))
    _RCV = staticmethod(lambda: Receiver(depths=np.array([50.0]),
                                         ranges=np.array([1000.0, 4000.0])))

    def test_compute_rmax_multiplier_is_pure_arithmetic(self):
        assert Kraken._compute_rmax_m(self._RCV()) == pytest.approx(4200.0)
        assert Kraken._compute_rmax_m(
            self._RCV(), multiplier=3.0) == pytest.approx(12000.0)

    def test_narrowband_deck_gets_1_05x(self, tmp_path):
        bounds = Kraken(verbose=False)._write_kraken_env(
            tmp_path / 'nb.env', _pekeris(), self._SRC(),
            receiver_obj=self._RCV(), receiver_depths=self._RCV().depths)
        assert bounds['rmax'] == pytest.approx(1.05 * 4000.0)

    def test_broadband_deck_gets_3x(self, tmp_path):
        bounds = Kraken(verbose=False)._write_kraken_env(
            tmp_path / 'bb.env', _pekeris(), self._SRC(),
            receiver_obj=self._RCV(), receiver_depths=self._RCV().depths,
            frequencies=np.linspace(80.0, 120.0, 5))
        assert bounds['rmax'] == pytest.approx(3.0 * 4000.0)

    def test_one_element_vector_is_not_a_sweep(self, tmp_path):
        # The gate is len(frequencies) > 1, matching the run-mode promotion.
        bounds = Kraken(verbose=False)._write_kraken_env(
            tmp_path / 'one.env', _pekeris(), self._SRC(),
            receiver_obj=self._RCV(), receiver_depths=self._RCV().depths,
            frequencies=np.array([100.0]))
        assert bounds['rmax'] == pytest.approx(1.05 * 4000.0)

    def test_pinned_rmax_wins_everywhere(self, tmp_path):
        bounds = Kraken(verbose=False, rmax_m=9000.0)._write_kraken_env(
            tmp_path / 'pin.env', _pekeris(), self._SRC(),
            receiver_obj=self._RCV(), receiver_depths=self._RCV().depths,
            frequencies=np.linspace(80.0, 120.0, 5))
        assert bounds['rmax'] == 9000.0


class TestAutoSegmentationEdges:
    """``models/_segmentation.py``: automatic segmentation unions the
    change-point ranges and inserts intermediates so no gap exceeds the 2 km
    ceiling (``_MAX_SEGMENT_LENGTH_M``) — a profile at least every 2 km even
    across a slowly-varying stretch. Pure function; no binary."""

    def test_wedge_with_5km_gaps_splits_at_change_points(self):
        from uacpy.models._segmentation import (
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
        # Each segment is a range-independent slice sampled at its own edge.
        for r, seg in segments:
            assert not seg.is_range_dependent
            assert float(seg.bathymetry.eval(range=0.0)) == pytest.approx(
                100.0 + r / 100.0)

    def test_a_range_independent_env_is_one_segment(self):
        from uacpy.models._segmentation import segment_environment_by_range
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
        with pytest.raises(ModelExecutionError):
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
    """``_modes_single_profile`` samples the r = 0 profile of every
    range-dependent quantity, while the field paths reduce the bottom with
    ``collapse['bottom_range']`` ('median'). A guard that assumes one policy
    is wrong on the other run mode in both directions: it let an
    elastic-over-fluid column at r = 0 through to the krakenc.exe hang under
    RunMode.MODES whenever the median column happened to be fluid, and
    refused the mirror case that would have run."""

    _SRC = staticmethod(lambda: Source(depths=[50.0], frequencies=[100.0]))

    @staticmethod
    def _env(bottom):
        return Environment(name='rdguard', bathymetry=100.0, ssp=1500.0,
                           bottom=bottom)

    def test_modes_follows_r0_not_the_median(self):
        from uacpy.core.exceptions import UnsupportedFeatureError
        env = self._env(_rd_layered_bottom(shear_at_r0=400.0,
                                           shear_elsewhere=0.0))
        model = Kraken(verbose=False)
        with pytest.raises(UnsupportedFeatureError):
            model._reject_acoustic_below_elastic(env, RunMode.MODES)
        # The field paths carry the median column, which is fluid here.
        model._reject_acoustic_below_elastic(env, RunMode.COHERENT_TL)

    def test_the_field_paths_follow_the_median_not_r0(self):
        from uacpy.core.exceptions import UnsupportedFeatureError
        env = self._env(_rd_layered_bottom(shear_at_r0=0.0,
                                           shear_elsewhere=400.0))
        model = Kraken(verbose=False)
        model._reject_acoustic_below_elastic(env, RunMode.MODES)
        with pytest.raises(UnsupportedFeatureError):
            model._reject_acoustic_below_elastic(
                env, RunMode.COHERENT_TL)

    def test_the_collapse_helper_names_both_policies(self):
        model = Kraken(verbose=False)
        assert model._bottom_collapse_for(RunMode.MODES) == 'r0'
        for mode in (RunMode.COHERENT_TL, RunMode.INCOHERENT_TL,
                     RunMode.BROADBAND, RunMode.TIME_SERIES):
            assert (model._bottom_collapse_for(mode)
                    == model._collapse['bottom_range'])


def test_incoherent_tl_on_krakenc_is_quiet_for_a_multi_profile_run():
    """``field.f90:202-203`` picks the evaluator by profile count. The
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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
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
        spans = Kraken(verbose=False)._elastic_depth_intervals(
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
                         ranges=np.array([500.0, 1000.0, 2000.0]))).db)
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
                         ranges=np.array([500.0, 1000.0, 2000.0]))).db)
        assert np.max(np.abs(tl[1] - ref[1])) < 5.0, (tl[1], ref[1])

    def test_a_depth_at_the_elastic_top_interface_is_kept(self):
        """An interface depth is tabulated once per adjoining medium and
        ``calculateweights.f90:43-49`` brackets it on the upper copy, so the
        top of an elastic medium reads the acoustic sample above it."""
        env = self._env()
        model = Kraken(verbose=False)
        spans = model._elastic_depth_intervals(env, env.bottom.at(range=0.0))
        rcv = Receiver(depths=np.array([60.0, 60.5]), ranges=[1000.0])
        with pytest.warns(UserWarning, match='elastic sub-bottom'):
            _, keep = model._partition_elastic_subbottom(env, rcv, spans)
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
        with pytest.raises(ModelExecutionError) as exc:
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
        model._check_non_trapped_modes(k, env, 200.0)
        out = capsys.readouterr().out
        assert '1 of 3 modes are non-trapped' in out
        assert '1650.0 m/s' in out

    def test_every_mode_non_trapped_raises_off_the_mode_set_alone(self):
        model = Kraken(verbose=False)
        with pytest.raises(ModelExecutionError, match='non-trapped'):
            model._check_non_trapped_modes(
                _k_for([1710.0, 1750.0], 150.0), _duct(5.0), 150.0)

    def test_trapped_modes_say_nothing(self, capsys):
        model = Kraken(verbose='info')
        model._check_non_trapped_modes(
            _k_for([1500.0, 1650.0], 150.0), _duct(5.0), 150.0)
        assert 'non-trapped' not in capsys.readouterr().out

    def test_leaky_modes_opt_in_passes_through_silently(self):
        """``leaky_modes=True`` asks for exactly these modes."""
        model = Kraken(leaky_modes=True, verbose=False)
        model._check_non_trapped_modes(
            _k_for([1710.0, 1750.0], 150.0), _duct(5.0), 150.0)

    def test_a_boundary_with_no_half_space_has_nothing_to_leak_into(self):
        """vacuum / rigid / reflection-table bottoms carry a placeholder
        sound_speed and resolve to an unbounded c_high, so comparing against
        it would be meaningless."""
        model = Kraken(verbose=False)
        env = Environment(
            name='rigid', bathymetry=5.0, ssp=[(0.0, C_WATER), (5.0, C_WATER)],
            bottom=BoundaryProperties(acoustic_type='rigid'))
        model._check_non_trapped_modes(_k_for([9000.0], 150.0), env, 150.0)

    def test_an_elastic_half_space_traps_on_its_shear_speed_instead(self):
        """``kraken.f90:209`` clamps cHigh to cS there, so the compressional
        speed is the wrong threshold and this check stands down."""
        model = Kraken(verbose=False)
        env = _duct(5.0, shear_speed=800.0, shear_attenuation=0.5)
        model._check_non_trapped_modes(_k_for([1710.0], 150.0), env, 150.0)


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
    (``kraken.py``, the ``broadband or return_pressure`` branch). Resolving
    the right speed is not the same contract as writing it onto the field,
    and only the write reaches ``to_time_trace``."""

    def test_the_stamp_is_the_seabed_speed_and_anchors_the_window(self):
        env = Environment(name='cmax_bb', bathymetry=100.0, ssp=1500.0,
                          bottom=_halfspace(3000.0, density=2.0,
                                            attenuation=0.1))
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=np.array([50.0]), ranges=np.array([2000.0]))
        result = Kraken(verbose=False).run(
            env, src, rcv, run_mode=RunMode.BROADBAND,
            frequencies=np.linspace(80.0, 120.0, 5))

        assert result.metadata['c_max'] == pytest.approx(3000.0)

        trace = result.to_time_trace(depth=50.0, range=2000.0)
        t = np.asarray(trace.coords['time'], dtype=float)
        assert t[0] == pytest.approx(2000.0 / 3000.0 - 0.05, abs=0.02)


def _halfspace(sound_speed, **kwargs):
    return BoundaryProperties(
        acoustic_type='half-space', sound_speed=sound_speed,
        density=kwargs.pop('density', 1.8),
        attenuation=kwargs.pop('attenuation', 0.3), **kwargs)


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
            Kraken(verbose=False)._reject_acoustic_below_elastic(
                env, RunMode.COHERENT_TL)

    def test_elastic_over_elastic_is_allowed(self):
        env = _layered(_layer(shear=400.0), _layer(shear=500.0),
                       halfspace_shear=600.0)
        Kraken(verbose=False)._reject_acoustic_below_elastic(
            env, RunMode.COHERENT_TL)

    def test_a_fluid_layer_above_an_elastic_one_is_allowed(self):
        env = _layered(_layer(shear=0.0), _layer(shear=400.0),
                       halfspace_shear=600.0)
        Kraken(verbose=False)._reject_acoustic_below_elastic(
            env, RunMode.COHERENT_TL)

    def test_the_fluid_halfspace_case_is_refused(self):
        env = _layered(_layer(shear=400.0), halfspace_shear=0.0)
        with pytest.raises(UnsupportedFeatureError, match='fluid halfspace'):
            Kraken(verbose=False)._reject_acoustic_below_elastic(
                env, RunMode.COHERENT_TL)


class TestKrakenRoughElasticInterface:
    """``kraken.f90:169`` / ``krakenc.f90:182`` stop with 'Rough elastic
    interfaces are not allowed' for any elastic medium whose ``SSP%sigma`` is
    non-zero, and the writer takes that sigma from the layer's own
    ``roughness``. A rough elastic *half-space* is fine — its sigma sits on
    the BotOpt line and feeds KupIng."""

    def test_roughness_on_an_elastic_layer_is_refused(self):
        env = _layered(_layer(shear=300.0, roughness=0.5),
                       halfspace_shear=600.0)
        with pytest.raises(UnsupportedFeatureError, match='Rough elastic'):
            Kraken(verbose=False)._reject_rough_elastic_layer(
                env, RunMode.COHERENT_TL)

    def test_roughness_on_a_fluid_layer_is_allowed(self):
        env = _layered(_layer(shear=0.0, roughness=0.5))
        Kraken(verbose=False)._reject_rough_elastic_layer(
            env, RunMode.COHERENT_TL)

    def test_a_smooth_elastic_layer_is_allowed(self):
        env = _layered(_layer(shear=300.0, roughness=0.0),
                       halfspace_shear=600.0)
        Kraken(verbose=False)._reject_rough_elastic_layer(
            env, RunMode.COHERENT_TL)


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


def test_kraken_mode_grid_meshes_at_the_shear_wavelength():
    """AT meshes an elastic medium at ``c_s/(20 f)``
    (``ReadEnvironmentMod.f90:101-103``), and the mode-tabulation grid carries
    the mode shapes and the coupling integrals — sizing it on compressional
    speeds alone under-samples an elastic sediment by ``c_p/c_s``."""
    model = Kraken(verbose=False)
    elastic = _layered(_layer(shear=300.0), halfspace_shear=600.0)
    fluid = _layered(_layer(shear=0.0))
    ppm_elastic = model._resolve_mode_points_per_meter(elastic, [100.0])
    ppm_fluid = model._resolve_mode_points_per_meter(fluid, [100.0])
    assert ppm_elastic > ppm_fluid
    # 10 points per shear wavelength at 100 Hz over c_s = 300 m/s.
    assert ppm_elastic == pytest.approx(10.0 * 100.0 / 300.0, rel=1e-9)


def test_kraken_mode_cutoff_probe_reraises_a_real_failure(monkeypatch):
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

    monkeypatch.setattr(Kraken, '_run_kraken_executable', _boom)
    with pytest.raises(ModelExecutionError, match='crashed'):
        model._count_modes_at_freq(env, source, receiver, 100.0,
                                   model._exe)


@pytest.mark.requires_binary
@pytest.mark.slow
class TestKrakenModesBelowAnElasticSeafloor:
    """``kraken.f90:592`` tabulates the eigenvector over the ACOUSTIC media
    only, and ``calculateweights.f90`` extrapolates past the last node, so the
    samples ``compute_modes`` returned below an elastic seafloor were a
    straight line: measured increments constant to 3e-8 on a -2.3e-3 step,
    against a 1.8e-3 spread for a fluid-layer control."""

    @staticmethod
    def _modes(shear):
        env = _layered(_layer(shear=shear),
                       halfspace_shear=600.0 if shear else 0.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
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
        from uacpy.models._segmentation import segment_environment_by_range
        wedge = self._wedge()
        n_100 = len(segment_environment_by_range(wedge, freq=100.0))
        n_300 = len(segment_environment_by_range(wedge, freq=300.0))
        assert n_300 > n_100 > 6, (
            f"{n_100} profiles at 100 Hz and {n_300} at 300 Hz: the count has "
            f"to follow the wavelength, not just the 2 km ceiling")

    def test_each_segment_spans_under_a_quarter_wavelength_of_depth(self):
        import numpy as np
        from uacpy.models._segmentation import (
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
        from uacpy.models._segmentation import segment_environment_by_range
        flat = self._wedge(d0=200.0, d1=195.0, r_max=20000.0)
        assert len(segment_environment_by_range(flat, freq=300.0)) <= 12

    def test_no_frequency_falls_back_to_the_metre_ceiling(self):
        from uacpy.models._segmentation import segment_environment_by_range
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
                data=[[1540, 1500], [1520, 1500], [1500, 1500],
                      [1495, 1500], [1493, 1500]],
                ranges=[0.0, 20000.0]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.7,
                                      attenuation=0.5))

    def test_a_range_dependent_profile_asks_for_more_at_higher_frequency(self):
        from uacpy.models._segmentation import segment_environment_by_range
        env = self._ssp_driven()
        n_low = len(segment_environment_by_range(env, freq=100.0))
        n_high = len(segment_environment_by_range(env, freq=800.0))
        assert n_high > n_low, (
            f"{n_low} profiles at 100 Hz and {n_high} at 800 Hz: a flat-bottom "
            f"but range-dependent SSP must still scale with frequency")

    def test_each_segment_spans_a_bounded_profile_change(self):
        from uacpy.models._segmentation import (
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
        from uacpy.models._segmentation import segment_environment_by_range
        env = Environment(
            bathymetry=200.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.7,
                                      attenuation=0.5))
        assert len(segment_environment_by_range(env, freq=800.0)) == 1
