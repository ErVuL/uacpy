"""OASES writer/reader fidelity tests.

The OASES Fortran is vendored read-only; everything here exercises uacpy's
Python wrapper against what OASES actually reads.
"""

import warnings

import numpy as np
import pytest

import uacpy
from uacpy.core.exceptions import (
    ConfigurationError, UnsupportedFeatureError,
)


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
        # The deck carries speeds at %.2f; the [:6] slice drops the last
        # decimal so a 1-decimal deck matches too. peak is 1529.4 m/s, four
        # integer digits, which is what makes the slice land on the point.
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
            k = np.asarray(uacpy.Kraken(timeout=600).run(env, src, rcv).db)
            o = np.asarray(uacpy.OASES.for_mode(RunMode.COHERENT_TL)
                           .run(env, src, rcv).db)
        d = np.abs(k - o)
        # Median, not max: a modal sum and a wavenumber integral put the
        # interference nulls of a ducted profile at slightly different ranges,
        # so a max-norm over 15 ranges is set by null placement rather than by
        # the profile either model was handed. The 2.0 dB bound itself is
        # unsourced — it separates "same ocean" from "subsampled ocean", not
        # two models' accuracy.
        assert np.nanmedian(d) < 2.0, (
            f"OASES vs Kraken median {np.nanmedian(d):.2f} dB on a 201-point "
            f"duct profile — the SSP is being subsampled before the run")


class TestOaspRangeAxisFidelity:
    """OASP Block VIII carries ``R0`` and ``RSPACE`` in km (oasp.tex:133-134),
    so a metre-resolution %.3f would round sub-metre receiver spacing to zero
    and collapse every receiver onto one range. OASP also only evaluates
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
        # 1 m is a gate, not a budget: the writer emits R1/RSPACE at nine
        # decimals of a km, so the realised error here is ~10 um. The %.3f km
        # the docstring warns about would quantise RSPACE to 0.246 km and put
        # this last range 14 m out, which is what the gate separates.
        assert last_m == pytest.approx(15000.0, abs=1.0), (
            f"last range lands at {last_m:.1f} m for a requested 15000 m")


class TestOasesSSPDecimationAtTheRealLimit:
    """Below the water column's share of OASES' NLA=1001 the profile passes
    through untouched; above it uacpy decimates and says so, rather than
    handing OASES a deck it rejects with '*** TOO MANY LAYERS ***'
    (oaseun31.f:44).

    NL counts the whole deck, so the budget is NLA minus the halfspaces and
    sediment layers around the water column — two for the simplest deck.
    """

    #: Upper halfspace + bottom halfspace, the minimum any deck spends.
    N_OTHER = 2

    @staticmethod
    def _ssp(n):
        z = np.linspace(0.0, 200.0, n)
        return np.column_stack([z, 1500.0 + 0.01 * z])

    def test_profile_under_the_limit_is_untouched(self):
        import warnings as _w
        from uacpy.io.oases_writer import (_check_ssp_layer_count,
                                           _OASES_MAX_LAYERS)
        data = self._ssp(_OASES_MAX_LAYERS - self.N_OTHER)
        with _w.catch_warnings():
            _w.simplefilter('error')
            out = _check_ssp_layer_count(data, self.N_OTHER)
        assert out.shape == data.shape
        np.testing.assert_array_equal(out, data)

    def test_profile_over_the_limit_is_decimated_with_a_warning(self):
        from uacpy.io.oases_writer import (_check_ssp_layer_count,
                                           _OASES_MAX_LAYERS)
        data = self._ssp(5000)
        with pytest.warns(UserWarning, match="decimated"):
            out = _check_ssp_layer_count(data, self.N_OTHER)
        assert out.shape[0] <= _OASES_MAX_LAYERS - self.N_OTHER
        # Surface and seafloor must survive so the interfaces still pin.
        np.testing.assert_allclose(out[0], data[0])
        np.testing.assert_allclose(out[-1], data[-1])

    def test_the_seabed_share_is_required_not_assumed(self):
        """Omitting it would bound the SSP alone and overrun NL."""
        from uacpy.io.oases_writer import _check_ssp_layer_count
        with pytest.raises(TypeError):
            _check_ssp_layer_count(self._ssp(10))


@pytest.mark.requires_binary
def test_oast_short_range_run_returns_a_field_not_nan():
    """OAST Block VIII XLEFT/XRIGHT set the FFT output grid, not just a plot
    window. They are in km, so at %.1f any run shorter than ~50 m would round
    XRIGHT to 0.0 and hand back an all-NaN TL field with no exception."""
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
                           ranges=np.linspace(10.0, 40.0, 20))).db)
    assert np.isfinite(tl).any(), "entire short-range TL field is NaN"
    finite = tl[np.isfinite(tl)]
    assert finite.min() > 0.0 and finite.max() < 200.0


@pytest.mark.requires_binary
def test_a_single_frequency_contour_option_is_refused_before_the_run():
    """``unoast31.f:138-140`` ends at ``STOP '*** CONTOURS REQUIRE NRFR>1 …'``
    when option ``'o'`` meets ``NFREQ <= 1``. That is a character stop, so the
    binary exits 0 and writes no ``.prt`` — nothing is left to diagnose from.
    The deck is refusable on inspection, so refuse it."""
    import uacpy
    from uacpy.core.exceptions import ConfigurationError
    from uacpy.models.base import RunMode
    env = uacpy.Environment(
        bathymetry=200.0,
        ssp=uacpy.SoundSpeedProfile.from_pairs([(0.0, 1500.0), (200.0, 1500.0)]),
        bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                        sound_speed=1800.0, density=1.8,
                                        attenuation=0.5))
    model = uacpy.OASES.for_mode(RunMode.COHERENT_TL)
    model.options = 'N J T o'
    with pytest.raises(ConfigurationError, match='CONTOURS REQUIRE'):
        model.run(env, uacpy.Source(depths=50.0, frequencies=500.0),
                  uacpy.Receiver(depths=[100.0], ranges=[1000.0, 2000.0]))


@pytest.mark.parametrize('banner', [
    'STOP *** CMIN/CMAX CONFLICT ***',
    'STOP >>>> ERROR: INPUT FILE NOT FOUND <<<<',
    'STOP INVALID INPATCH',
])
def test_an_oases_stop_banner_does_not_point_at_a_prt(tmp_path, banner):
    """OASES stops with a character code — gfortran exits 0 — and writes no
    ``.prt``, so stderr is the only diagnostic. Only 19 of its 46 banners carry
    ``***``; the rest use ``>>> … <<<`` or no marker at all, so the check is on
    the stop *form*."""
    import uacpy
    from types import SimpleNamespace
    from uacpy.core.exceptions import ModelExecutionError
    from uacpy.models.base import RunMode
    model = uacpy.OASES.for_mode(RunMode.COHERENT_TL)
    result = SimpleNamespace(stdout='', stderr=banner, returncode=0)
    with pytest.raises(ModelExecutionError) as ei:
        model._raise_on_fortran_fatal(result, tmp_path, 'nonexistent')
    msg = str(ei.value)
    assert banner.split('STOP', 1)[1].strip()[:12] in msg, (
        f"the binary's own banner never reached the user: {msg[:200]}")
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
    """``frequencies=`` sets an (fmin, fmax, N) triple; dropping fmin leaves
    OASP sweeping from DC and costing several times the requested bins.

    The band edges land on OASP's own FFT bin grid, not on the request:
    ``LXP1 = max(2, FR1/DLFREQ + 1)`` truncates downwards and
    ``MX = FR2/DLFREQ + 2`` rounds up (unoasp22.f:237-247), so the realised
    band overhangs by up to one bin either side — which is what the 1 Hz of
    slack below absorbs.
    """
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
def test_oasp_multi_frequency_source_centres_the_sweep_on_the_band():
    """A multi-element ``source.frequencies`` names a band; its centre — not
    its first element — sets the deck fc and the derived ``2.5×fc`` sweep
    top, so the computed band covers every requested frequency.
    ``frequencies[0]`` as the centre swept [100, 200, 300] only up to
    2.5×100 = 250 Hz and never computed the top of the band."""
    import uacpy
    import warnings as _w
    from uacpy.models import OASP
    from uacpy.models.base import RunMode
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        field = OASP(n_time_samples=512).run(
            _pekeris_env(),
            uacpy.Source(depths=25.0, frequencies=[100.0, 200.0, 300.0]),
            uacpy.Receiver(depths=np.array([50.0]),
                           ranges=np.array([1000.0])),
            run_mode=RunMode.BROADBAND)
    f = np.asarray(field.coords['frequency'], dtype=float)
    assert f.max() >= 300.0, (
        f"computed band tops out at {f.max():.1f} Hz, below the requested "
        f"300 Hz")
    # The deck fc rides back on the .trf header.
    assert field.metadata['center_frequency'] == pytest.approx(200.0, abs=1.0)


@pytest.mark.requires_binary
def test_oasp_pinned_freq_max_below_the_band_top_raises():
    """A pinned sweep edge that cannot reach the top of the requested band
    names the conflict instead of silently truncating the sweep."""
    import uacpy
    from uacpy.core.exceptions import ConfigurationError
    from uacpy.models import OASP
    from uacpy.models.base import RunMode
    with pytest.raises(ConfigurationError, match="freq_max"):
        OASP(n_time_samples=512, freq_max=250.0).run(
            _pekeris_env(),
            uacpy.Source(depths=25.0, frequencies=[100.0, 200.0, 300.0]),
            uacpy.Receiver(depths=np.array([50.0]),
                           ranges=np.array([1000.0])),
            run_mode=RunMode.BROADBAND)


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
    must not come back with the phase reference dropped. ``travelling_wave``
    declares the carrier ``exp(-i k0 r)`` is still in the data, which is what
    lets a consumer put the causal arrival at ``t = r/c0``
    (``core/results/_base.py:37-39``); an untagged slice is not IFFT-able."""
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


class TestOasnWhiteNoiseDefaultThroughTheModel:
    """OASES adds ``10**(WNLEVDB/10)`` to every covariance diagonal with no
    off switch (oasnun22.f:228, :1157). The OASN model therefore defaults
    ``white_noise_level`` to ``None`` — written as -200 dB, whose 1e-20
    linear power is numerically nil — while an explicit 0.0 reaches the
    deck as a literal 0 dB (unit linear power per sensor)."""

    @staticmethod
    def _deck(tmp_path, monkeypatch, **kw):
        from uacpy.core.exceptions import ModelExecutionError
        from uacpy.models.oases import OASN
        env = uacpy.Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1700.0,
                density=1.8, attenuation=0.5))
        model = OASN(surface_noise_level=70.0, verbose=False,
                     work_dir=str(tmp_path), cleanup=False, **kw)
        monkeypatch.setattr(type(model), '_run_subprocess',
                            lambda self, *a, **k: _FakeProc())
        # The fake subprocess writes no .xsm, so run() raises after the
        # deck is on disk — the deck text is the object under test.
        with pytest.raises(ModelExecutionError, match='did not produce'):
            model.run(env, uacpy.Source(depths=10.0, frequencies=100.0),
                      uacpy.Receiver(depths=[30.0, 50.0], ranges=[0.0]))
        return (tmp_path / 'oasn_run.dat').read_text()

    def test_default_deck_carries_the_nil_level(self, tmp_path, monkeypatch):
        text = self._deck(tmp_path, monkeypatch)
        assert '70.0 -200.0 0.0 0' in text

    def test_explicit_zero_reaches_the_deck_as_0_db(self, tmp_path,
                                                    monkeypatch):
        text = self._deck(tmp_path, monkeypatch, white_noise_level=0.0)
        assert '70.0 0.0 0.0 0' in text


class TestContourOffsetUnderAutomaticSampling:
    """``unoast31.f:429`` sets ``OFFDB=0E0`` inside the ``NWVNOin < 0``
    branch (``unoasp22.f:323`` for OASP), so the binary discards a
    user-supplied contour offset and prints "THE DEFAULT CONTOUR OFFSET IS
    APPLIED". Since ``nw_samples=-1`` is uacpy's own default, a documented
    constructor argument silently had no effect in the default configuration.
    The manual does not say so — ``oast.tex:547-550`` names only IC1/IC2 —
    which is why the warning has to come from uacpy."""

    @pytest.mark.parametrize('cls_name', ['OAST', 'OASP'])
    def test_offset_with_automatic_sampling_warns(self, cls_name):
        cls = getattr(uacpy, cls_name)
        with pytest.warns(UserWarning, match='automatic wavenumber sampling'):
            cls(integration_offset=2.0)

    @pytest.mark.parametrize('cls_name', ['OAST', 'OASP'])
    @pytest.mark.parametrize('kwargs', [
        {'integration_offset': 2.0, 'nw_samples': 4096},   # offset is honoured
        {'nw_samples': -1},                                # no offset to lose
    ])
    def test_no_warning_when_the_offset_reaches_the_kernel_or_is_unset(
            self, cls_name, kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            # The OASES licence notice is emitted once per source per process
            # (base.py::_warn_restricted_source), so whichever test builds the
            # first OASES model in this worker absorbs it. Under xdist that is
            # a coin toss, which would make a strict block flaky for a warning
            # it does not assert on.
            warnings.filterwarnings('ignore', message='.*not redistributable.*')
            warnings.filterwarnings('ignore', message='.*licence.*')
            getattr(uacpy, cls_name)(**kwargs)

    def test_oasn_is_not_affected(self):
        # unoasn22.f:283 tests OFFDBIN, which its automatic branch never
        # touches, so OASN keeps the user's offset and must stay silent.
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            warnings.filterwarnings('ignore', message='.*not redistributable.*')
            warnings.filterwarnings('ignore', message='.*licence.*')
            uacpy.OASN(integration_offset=2.0)


class TestOASRShearReflectionTypesAreRefused:
    """OASR reflects off the seabed, so the incident medium is the water
    column — a fluid, which carries no SV wave. `oasr.tex:143-145` defines
    option 'S' as the P-SV reflection coefficient, which is therefore
    identically zero for every environment uacpy can express: OASES writes a
    column of zeros to the .rco and uacpy returned it as a result. 'P-Slow'
    is the Biot slow wave and needs a poro-elastic medium no carrier has."""

    @pytest.mark.parametrize('reflection_type', ['P-SV', 'P-Slow'])
    def test_zero_valued_reflection_types_raise(self, reflection_type):
        with pytest.raises(UnsupportedFeatureError, match='zeros'):
            uacpy.OASR(reflection_type=reflection_type)

    @pytest.mark.parametrize('reflection_type', [None, 'P-P', 'transmission'])
    def test_usable_reflection_types_are_untouched(self, reflection_type):
        # The discriminating half — measured max|R| 0.999947 (P-P) and
        # 1.273706 (transmission); the guard must not reach them.
        kwargs = ({} if reflection_type is None
                  else {'reflection_type': reflection_type})
        uacpy.OASR(**kwargs)


class TestOASPFrequencyGridSubstitution:
    """OASP is a pulse model: its frequency axis is the FFT ladder implied by
    the time window, not the bins the caller names. Asking for 3 returned 820
    (via ``run(frequencies=)``) or 2047 (via ``Source``) with no warning,
    while Kraken on the identical call returns exactly 3 — so a caller who
    has used one has no reason to expect the other."""

    @staticmethod
    def _env():
        return uacpy.Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0,
                density=1.8, attenuation=0.5))

    def test_run_frequencies_substitution_warns(self):
        with pytest.warns(UserWarning, match='FFT ladder'):
            uacpy.OASP().run(
                self._env(), uacpy.Source(depths=25.0, frequencies=200.0),
                uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
                run_mode=uacpy.RunMode.BROADBAND, frequencies=[150., 200., 250.])

    def test_source_frequencies_substitution_also_warns(self):
        # A multi-element Source.frequencies names bins just as explicitly.
        with pytest.warns(UserWarning, match='FFT ladder'):
            uacpy.OASP().run(
                self._env(),
                uacpy.Source(depths=25.0, frequencies=[150., 200., 250.]),
                uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
                run_mode=uacpy.RunMode.BROADBAND)

    def test_single_frequency_names_nothing_and_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            warnings.filterwarnings('ignore', message='.*not redistributable.*')
            warnings.filterwarnings('ignore', message='.*licence.*')
            uacpy.OASP().run(
                self._env(), uacpy.Source(depths=25.0, frequencies=200.0),
                uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
                run_mode=uacpy.RunMode.BROADBAND)


class TestOASRContourOptionIsLogSwept:
    """`oasr.tex:135` documents option 'C' as a *plot* option ("Loss contours
    plotted in frequency and grazing angle"), but `unoasr21.f:123-125`
    computes F1LOG/DFLOG and `:243` evaluates the coefficients at
    ``EXP(F1LOG + (JJ-1)*DFLOG)`` — it changes the frequencies the physics is
    computed at. `oast.tex:200-203` confirms it from the other side, calling
    'C' the way to get "consistent logarithmic sampling".

    uacpy's equispacing check was therefore exactly **inverted** under 'C': a
    linear request — the one silently regridded, 4 of 5 bins wrong — passed
    without a word, while a correct log-spaced request was warned about and
    told it had been resampled onto a linspace it was never put on."""

    @staticmethod
    def _env():
        return uacpy.Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1600.0,
                density=1.8, attenuation=0.5))

    def _run(self, options, freqs):
        kwargs = {} if options is None else {'options': options}
        return uacpy.OASR(**kwargs).run(
            self._env(), uacpy.Source(depths=25.0, frequencies=freqs),
            uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
            run_mode=uacpy.RunMode.REFLECTION)

    def test_contour_option_with_a_linear_request_warns(self):
        with pytest.warns(UserWarning, match='logarithmic'):
            self._run('N T C', np.linspace(10.0, 10000.0, 5))

    def test_contour_option_with_a_log_request_is_silent(self):
        # The half that was previously warned about wrongly.
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            warnings.filterwarnings('ignore', message='.*licence.*')
            warnings.filterwarnings('ignore', message='.*redistributable.*')
            self._run('N T C', np.geomspace(10.0, 10000.0, 5))

    def test_without_the_option_a_linear_request_is_still_silent(self):
        # The discriminating counterpart: the linear sweep is correct without
        # 'C', so the new check must not reach it.
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            warnings.filterwarnings('ignore', message='.*licence.*')
            warnings.filterwarnings('ignore', message='.*redistributable.*')
            self._run(None, np.linspace(10.0, 10000.0, 5))


class TestOptionLineFitsGETOPT:
    """`GETOPT` reads the option record as ``FORMAT(40A1)`` in all four
    programs (``unoast31.f:978-979`` and siblings), so a letter in column 41
    or beyond is discarded before the option scan — and unlike an unknown
    letter it produces no ``UNKNOWN OPTION`` diagnostic. A dropped option
    letter changes what the run computes, so this is not cosmetic the way the
    title's 80-character `FORMAT(20A4)` truncation is."""

    @staticmethod
    def _env():
        return uacpy.Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1600.0,
                density=1.8, attenuation=0.5))

    def _write(self, options, tmp_path):
        from uacpy.io.oases_writer import write_oast_input
        write_oast_input(str(tmp_path / 't.dat'), self._env(),
                         uacpy.Source(depths=25.0, frequencies=100.0),
                         uacpy.Receiver(depths=[50.0], ranges=[1000.0, 2000.0]),
                         options=options)

    def test_an_over_length_option_line_is_refused(self, tmp_path):
        with pytest.raises(ConfigurationError, match='40'):
            self._write('N J T ' + 'Q ' * 25, tmp_path)

    @pytest.mark.parametrize('options', ['N J T', 'N', 'A' * 40])
    def test_a_fitting_option_line_still_writes(self, options, tmp_path):
        # Exactly 40 is legal — the bound is inclusive, and an off-by-one here
        # would refuse a deck GETOPT reads perfectly.
        self._write(options, tmp_path)
        assert (tmp_path / 't.dat').exists()
