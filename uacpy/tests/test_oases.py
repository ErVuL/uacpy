"""OASES writer/reader fidelity tests.

The OASES Fortran is vendored read-only; everything here exercises uacpy's
Python wrapper against what OASES actually reads.
"""

import numpy as np
import pytest


class _FakeProc:
    """A subprocess that reported success without writing anything."""
    returncode = 0
    stdout = ''
    stderr = ''


class TestOasesKeepsTheFullSSP:
    """OASES' real layer limit is ``parameter (NLA = 1001)``
    (third_party/oases/src/compar.f:23), enforced at oaseun31.f:44. Subsampling
    a measured profile to 15 rows costs several dB of TL — and defeats the
    whole point of fetching a CTD/Argo profile through uacpy.data."""

    @staticmethod
    def _duct_env(n=201):
        import uacpy
        z = np.linspace(0.0, 200.0, n)
        c = 1500.0 + 30.0 * np.exp(-((z - 30.0) / 6.0) ** 2) - 0.02 * z
        return uacpy.Environment(
            bathymetry=200.0,
            ssp=uacpy.SoundSpeedProfile.from_pairs(np.column_stack([z, c])),
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0,
                density=1.8, attenuation=0.5))

    def test_every_ssp_sample_reaches_the_deck(self, tmp_path):
        import uacpy
        from uacpy.io.oases_writer import write_oast_input
        env = self._duct_env(201)
        out = tmp_path / 'oast.dat'
        write_oast_input(
            out, env,
            uacpy.Source(depths=30.0, frequencies=500.0),
            uacpy.Receiver(depths=[60.0], ranges=[1000.0, 2000.0]))
        text = out.read_text()
        # The duct peak is the sample a 15-row subsample loses.
        z = np.linspace(0.0, 200.0, 201)
        c = 1500.0 + 30.0 * np.exp(-((z - 30.0) / 6.0) ** 2) - 0.02 * z
        peak = float(c.max())
        assert f"{peak:.2f}"[:6] in text or f"{peak:.1f}" in text, (
            f"the duct peak {peak:.2f} m/s never reached the deck — the SSP "
            f"was subsampled")

    @pytest.mark.requires_binary
    def test_tl_tracks_kraken_on_a_finely_sampled_profile(self):
        """Kraken meshes the full profile, so a large disagreement means
        OASES was handed a different ocean."""
        import uacpy
        import warnings as _w
        from uacpy.models.base import RunMode
        env = self._duct_env(201)
        src = uacpy.Source(depths=30.0, frequencies=500.0)
        rcv = uacpy.Receiver(depths=[30.0, 60.0],
                             ranges=np.linspace(1000.0, 8000.0, 15))
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            k = np.asarray(uacpy.Kraken(timeout=600).run(env, src, rcv).tl)
            o = np.asarray(uacpy.OASES.for_mode(RunMode.COHERENT_TL)
                           .run(env, src, rcv).tl)
        d = np.abs(k - o)
        assert np.nanmedian(d) < 2.0, (
            f"OASES vs Kraken median {np.nanmedian(d):.2f} dB on a 201-point "
            f"duct profile — the SSP is being subsampled before the run")


class TestOaspRangeAxisFidelity:
    """OASP Block VIII carries ``R0`` and ``RSPACE`` in km. Written at %.3f
    that is 1 m resolution, so sub-metre receiver spacing rounds to zero and
    every receiver collapses onto one range. OASP also only supports
    ``r0 + i*dr``, so a non-uniform request cannot be honoured at all."""

    @staticmethod
    def _env():
        import uacpy
        return uacpy.Environment(
            bathymetry=200.0,
            ssp=uacpy.SoundSpeedProfile.from_pairs([(0.0, 1500.0),
                                                    (200.0, 1500.0)]),
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0,
                density=1.8, attenuation=0.5))

    @staticmethod
    def _block_viii(path):
        # NT FR1 FR2 DT R1 DR NR — the last 7-token numeric line.
        for line in reversed(path.read_text().splitlines()):
            parts = line.split()
            if len(parts) == 7:
                return parts
        raise AssertionError("Block VIII not found")

    def _write(self, tmp_path, ranges):
        import uacpy
        from uacpy.io.oases_writer import write_oasp_input
        out = tmp_path / 'oasp.dat'
        write_oasp_input(
            out, self._env(),
            uacpy.Source(depths=50.0, frequencies=500.0),
            uacpy.Receiver(depths=[100.0], ranges=np.asarray(ranges)))
        return out

    def test_sub_metre_spacing_survives(self, tmp_path):
        ranges = np.linspace(100.0, 110.0, 41)          # dr = 0.25 m
        parts = self._block_viii(self._write(tmp_path, ranges))
        dr_km = float(parts[5])
        assert dr_km > 0.0, (
            "RSPACE rounded to 0.000 km — 41 receivers over a 10 m aperture "
            "collapse onto a single range")
        assert dr_km == pytest.approx(0.25e-3, rel=1e-3)

    def test_non_uniform_ranges_are_rejected(self, tmp_path):
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match="uniformly spaced"):
            self._write(tmp_path, np.geomspace(500.0, 10000.0, 8))

    def test_uniform_ranges_round_trip_to_the_last_range(self, tmp_path):
        ranges = np.linspace(500.0, 15000.0, 60)
        parts = self._block_viii(self._write(tmp_path, ranges))
        r1_km, dr_km, nr = float(parts[4]), float(parts[5]), int(parts[6])
        last_m = (r1_km + (nr - 1) * dr_km) * 1000.0
        assert last_m == pytest.approx(15000.0, abs=1.0), (
            f"last range lands at {last_m:.1f} m for a requested 15000 m")


class TestOasesSSPDecimationAtTheRealLimit:
    """Below OASES' own NLA=1001 the profile passes through untouched; above
    it uacpy decimates and says so, rather than handing OASES a deck it
    rejects with '*** TOO MANY LAYERS ***' (oaseun31.f:44)."""

    @staticmethod
    def _ssp(n):
        z = np.linspace(0.0, 200.0, n)
        return np.column_stack([z, 1500.0 + 0.01 * z])

    def test_profile_under_the_limit_is_untouched(self):
        import warnings as _w
        from uacpy.io.oases_writer import _check_ssp_layer_count
        data = self._ssp(1001)
        with _w.catch_warnings():
            _w.simplefilter('error')
            out = _check_ssp_layer_count(data)
        assert out.shape == data.shape
        np.testing.assert_array_equal(out, data)

    def test_profile_over_the_limit_is_decimated_with_a_warning(self):
        from uacpy.io.oases_writer import (_check_ssp_layer_count,
                                           _OASES_MAX_LAYERS)
        data = self._ssp(5000)
        with pytest.warns(UserWarning, match="decimated"):
            out = _check_ssp_layer_count(data)
        assert out.shape[0] <= _OASES_MAX_LAYERS
        # Surface and seafloor must survive so the interfaces still pin.
        np.testing.assert_allclose(out[0], data[0])
        np.testing.assert_allclose(out[-1], data[-1])


@pytest.mark.requires_binary
def test_oast_short_range_run_returns_a_field_not_nan():
    """OAST Block VIII XLEFT/XRIGHT set the FFT output grid, not just a plot
    window. Written at %.1f km, any run shorter than ~50 m rounded XRIGHT to
    0.0 and the whole TL field came back NaN with no exception."""
    import uacpy
    import warnings as _w
    from uacpy.models.base import RunMode
    env = uacpy.Environment(
        bathymetry=200.0,
        ssp=uacpy.SoundSpeedProfile.from_pairs([(0.0, 1500.0), (200.0, 1500.0)]),
        bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                        sound_speed=1800.0, density=1.8,
                                        attenuation=0.5))
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        tl = np.asarray(uacpy.OASES.for_mode(RunMode.COHERENT_TL).run(
            env, uacpy.Source(depths=50.0, frequencies=500.0),
            uacpy.Receiver(depths=np.linspace(10.0, 90.0, 9),
                           ranges=np.linspace(10.0, 40.0, 20))).tl)
    assert np.isfinite(tl).any(), "entire short-range TL field is NaN"
    finite = tl[np.isfinite(tl)]
    assert finite.min() > 0.0 and finite.max() < 200.0


@pytest.mark.requires_binary
def test_oases_stop_banner_exiting_zero_is_raised_with_its_own_message():
    """OASES reports fatal conditions with ``STOP *** … ***``, which gfortran
    exits 0 for, and it writes no .prt — so the only diagnostic is stderr.
    Previously the run fell through to a 'check the .prt log' dead end."""
    import uacpy
    from uacpy.core.exceptions import ModelExecutionError
    from uacpy.models.base import RunMode
    env = uacpy.Environment(
        bathymetry=200.0,
        ssp=uacpy.SoundSpeedProfile.from_pairs([(0.0, 1500.0), (200.0, 1500.0)]),
        bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                        sound_speed=1800.0, density=1.8,
                                        attenuation=0.5))
    # 'o' selects FRCONT, which OASES refuses unless NRFR > 1.
    model = uacpy.OASES.for_mode(RunMode.COHERENT_TL)
    model.options = 'N J T o'
    with pytest.raises(ModelExecutionError) as ei:
        model.run(env, uacpy.Source(depths=50.0, frequencies=500.0),
                  uacpy.Receiver(depths=[100.0], ranges=[1000.0, 2000.0]))
    msg = str(ei.value)
    assert '.prt' not in msg or 'writes no .prt' in msg, (
        f"error still points at a .prt OASES never writes: {msg[:200]}")


def _pekeris_env():
    import uacpy
    return uacpy.Environment(
        bathymetry=100.0,
        ssp=uacpy.SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.0, 1500.0)]),
        bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                        sound_speed=1700.0, density=1.7,
                                        attenuation=0.5))


@pytest.mark.requires_binary
def test_oasp_run_frequencies_honours_the_lower_band_edge():
    """``frequencies=`` sets an (fmin, fmax, N) triple; dropping fmin made
    OASP sweep from DC and cost several times the requested bins."""
    import uacpy
    import warnings as _w
    from uacpy.models import OASP
    from uacpy.models.base import RunMode
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        field = OASP(n_time_samples=512).run(
            _pekeris_env(), uacpy.Source(depths=25.0, frequencies=150.0),
            uacpy.Receiver(depths=np.array([50.0]),
                           ranges=np.array([1000.0, 2000.0])),
            run_mode=RunMode.BROADBAND,
            frequencies=np.linspace(100.0, 200.0, 21))
    f = np.asarray(field.coords['frequency'], dtype=float)
    assert f.min() >= 99.0, f"band starts at {f.min():.2f} Hz, below the request"
    assert f.max() <= 201.0


@pytest.mark.requires_binary
def test_oasp_run_frequencies_warns_when_it_overrides_a_pinned_freq_min():
    import uacpy
    import warnings as _w
    from uacpy.models import OASP
    from uacpy.models.base import RunMode
    with _w.catch_warnings(record=True) as w:
        _w.simplefilter('always')
        OASP(n_time_samples=512, freq_min=10.0).run(
            _pekeris_env(), uacpy.Source(depths=25.0, frequencies=150.0),
            uacpy.Receiver(depths=np.array([50.0]),
                           ranges=np.array([1000.0, 2000.0])),
            run_mode=RunMode.BROADBAND,
            frequencies=np.linspace(100.0, 200.0, 21))
    assert any('freq_min' in str(x.message) for x in w), \
        "silently overrode the constructor's freq_min"


@pytest.mark.requires_binary
def test_oasp_coherent_tl_carries_the_same_phase_reference_as_broadband():
    """COHERENT_TL is one frequency slice of the same .trf array, so it
    must not come back with the phase reference dropped."""
    import uacpy
    import warnings as _w
    from uacpy.models import OASP
    from uacpy.models.base import RunMode
    env = _pekeris_env()
    src = uacpy.Source(depths=25.0, frequencies=150.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]),
                         ranges=np.array([1000.0, 2000.0]))
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        m = OASP(n_time_samples=512)
        nb = m.run(env, src, rcv, run_mode=RunMode.COHERENT_TL)
        bb = m.run(env, src, rcv, run_mode=RunMode.BROADBAND)
    assert nb.phase_reference == bb.phase_reference == 'travelling_wave'
    assert nb.kind == 'pressure' and np.iscomplexobj(nb.data)
    assert nb.data.dtype == np.complex128


@pytest.mark.requires_binary
class TestStaleOutputsAreCleared:
    """Every OASES sub-model names its output files after ``base_name``,
    which is a hard-coded literal — so a pinned ``work_dir`` holding a
    previous run's output would be read back as this run's answer. Each
    sub-model declares its outputs and ``_execute`` clears them first."""

    @staticmethod
    def _env():
        import uacpy
        return (
            uacpy.Environment(
                bathymetry=100.0, ssp=1500.0,
                bottom=uacpy.BoundaryProperties(
                    acoustic_type='half-space', sound_speed=1700.0,
                    density=1.7, attenuation=0.5)),
            uacpy.Source(depths=25.0, frequencies=100.0),
            uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
        )

    def test_oast_stale_plt_is_not_returned(self, tmp_path, monkeypatch):
        from uacpy.models.oases import OAST
        from uacpy.core.exceptions import ModelExecutionError
        (tmp_path / 'oast_run.plt').write_text('stale garbage\n' * 100)
        model = OAST(verbose=False, work_dir=str(tmp_path), cleanup=False)
        monkeypatch.setattr(type(model), '_run_subprocess',
                            lambda self, *a, **k: _FakeProc())
        with pytest.raises(ModelExecutionError, match='did not produce'):
            model.run(*self._env())
        assert not (tmp_path / 'oast_run.plt').exists()

    def test_oasp_stale_trf_is_not_returned(self, tmp_path, monkeypatch):
        from uacpy.models.oases import OASP
        from uacpy.core.exceptions import ModelExecutionError
        (tmp_path / 'oasp_run.trf').write_bytes(b'\x00' * 4096)
        model = OASP(verbose=False, work_dir=str(tmp_path), cleanup=False)
        monkeypatch.setattr(type(model), '_run_subprocess',
                            lambda self, *a, **k: _FakeProc())
        with pytest.raises(ModelExecutionError, match='did not produce'):
            model.run(*self._env())
        assert not (tmp_path / 'oasp_run.trf').exists()

    def test_every_sub_model_declares_its_outputs(self):
        from uacpy.models.oases import OAST, OASN, OASR, OASP
        for cls in (OAST, OASN, OASR, OASP):
            assert cls._OUTPUT_SUFFIXES, f'{cls.__name__} declares no outputs'
