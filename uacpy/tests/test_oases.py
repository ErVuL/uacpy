"""OASES writer/reader fidelity tests.

The OASES Fortran is vendored read-only; everything here exercises uacpy's
Python wrapper against what OASES actually reads — including the geometry
defaults that stop describing anything at the shallow edge of their range.
"""

import inspect
import re
import warnings

import numpy as np
import pytest

import uacpy
from uacpy.core import BoundaryProperties, Environment, Receiver, Source
from uacpy.core.results import Field
from uacpy.io.oases_writer import OasnNoise, OasnReplicaGrid
from uacpy.models import Bellhop, OASN, OASP, OASR
from uacpy.core.run_settings import RunMode
from uacpy.io.oases_writer import oases_frequency_sweep
from uacpy.core.environment import SoundSpeedProfile
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, UnsupportedFeatureError,
)
from uacpy.tests.conftest import make_halfspace
from uacpy.tests.conftest import recorded_warnings
from uacpy.tests.conftest import make_pekeris
from uacpy.models._launch import raise_on_fortran_fatal
from uacpy.models._extract import mask_source_axis


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

    @pytest.mark.requires_oases
    def test_tl_tracks_kraken_on_a_finely_sampled_profile(self):
        """Kraken meshes the full profile, so a large disagreement means
        OASES was handed a different ocean."""
        import uacpy
        import warnings as _w
        env = self._duct_env(201)
        src = uacpy.Source(depths=30.0, frequencies=500.0)
        rcv = uacpy.Receiver(depths=[30.0, 60.0],
                             ranges=np.linspace(1000.0, 8000.0, 15))
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            k = np.asarray(uacpy.Kraken(timeout=600).run(env, src, rcv).dB)
            o = np.asarray(uacpy.OAST().run(env, src, rcv).dB)
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
        with pytest.raises(
                TypeError,
                match="required positional argument: 'n_other_layers'"):
            _check_ssp_layer_count(self._ssp(10))

    def test_warn_false_decimates_without_saying_so(self):
        """The count-only callers ask this one warning for silence, rather
        than muting UserWarning process-wide around the call."""
        from uacpy.io.oases_writer import (_check_ssp_layer_count,
                                           _OASES_MAX_LAYERS)
        data = self._ssp(5000)
        with recorded_warnings() as caught:
            out = _check_ssp_layer_count(data, self.N_OTHER, warn=False)
        assert [str(w.message) for w in caught] == []
        assert out.shape[0] <= _OASES_MAX_LAYERS - self.N_OTHER
        np.testing.assert_allclose(out[0], data[0])
        np.testing.assert_allclose(out[-1], data[-1])


class TestOassInterfaceLookupLeavesGlobalWarningStateAlone:
    """``oass_bottom_interfaces`` only needs the row count the deck will end
    up with, and the writing call reports any decimation, so the lookup is
    silent about it. It must buy that silence from the callee rather than from
    ``warnings.catch_warnings()``, whose filter stack is process-global: such
    a window also swallows warnings other threads raise while it is open.
    """

    @staticmethod
    def _env(n_ssp_rows=None):
        from uacpy.core.boundary import BoundaryProperties
        ssp = 1500.0
        if n_ssp_rows is not None:
            z = np.linspace(0.0, 100.0, n_ssp_rows)
            ssp = uacpy.SoundSpeedProfile.from_pairs(
                [(float(d), 1500.0 + 0.01 * float(d)) for d in z])
        return uacpy.Environment(
            bathymetry=100.0, ssp=ssp,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.8,
                                      attenuation=0.5, roughness=0.5))

    def test_a_decimated_ssp_draws_no_warning_from_the_lookup(self):
        from uacpy.io.oases_writer import oass_bottom_interfaces
        env = self._env(n_ssp_rows=5000)
        with recorded_warnings() as caught:
            first_bottom, _ = oass_bottom_interfaces(env)
        assert [str(w.message) for w in caught] == []
        assert first_bottom > 2

    def test_a_warning_raised_on_another_thread_survives_the_lookup(self):
        """The lookup runs on one thread while another raises a UserWarning
        it never asked to hide. The probe is released from inside the
        row-count call, so no timing decides the outcome."""
        import threading

        from uacpy.io import oases_writer

        inside_the_lookup = threading.Event()
        probe_raised = threading.Event()
        delivered = []
        real_check = oases_writer._check_ssp_layer_count

        def releasing_check(*args, **kwargs):
            inside_the_lookup.set()
            probe_raised.wait(30.0)
            return real_check(*args, **kwargs)

        env = self._env()

        def run_the_lookup():
            try:
                oases_writer.oass_bottom_interfaces(env)
            finally:
                inside_the_lookup.set()

        with warnings.catch_warnings():
            warnings.simplefilter('always')
            warnings.showwarning = (
                lambda message, *a, **k: delivered.append(str(message)))
            oases_writer._check_ssp_layer_count = releasing_check
            try:
                worker = threading.Thread(target=run_the_lookup)
                worker.start()
                assert inside_the_lookup.wait(30.0)
                warnings.warn('probe from another thread', UserWarning)
                probe_raised.set()
                worker.join(30.0)
            finally:
                oases_writer._check_ssp_layer_count = real_check
        assert not worker.is_alive()
        assert delivered == ['probe from another thread']


@pytest.mark.requires_oases
def test_oast_short_range_run_returns_a_field_not_nan():
    """OAST Block VIII XLEFT/XRIGHT set the FFT output grid, not just a plot
    window. They are in km, so at %.1f any run shorter than ~50 m would round
    XRIGHT to 0.0 and hand back an all-NaN TL field with no exception."""
    import uacpy
    import warnings as _w
    env = uacpy.Environment(
        bathymetry=200.0,
        ssp=uacpy.SoundSpeedProfile.from_pairs([(0.0, 1500.0), (200.0, 1500.0)]),
        bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                        sound_speed=1800.0, density=1.8,
                                        attenuation=0.5))
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        tl = np.asarray(uacpy.OAST().run(
            env, uacpy.Source(depths=50.0, frequencies=500.0),
            uacpy.Receiver(depths=np.linspace(10.0, 90.0, 9),
                           ranges=np.linspace(10.0, 40.0, 20))).dB)
    assert np.isfinite(tl).any(), "entire short-range TL field is NaN"
    finite = tl[np.isfinite(tl)]
    assert finite.min() > 0.0 and finite.max() < 200.0


@pytest.mark.requires_oases
def test_a_single_frequency_contour_option_is_refused_before_the_run():
    """``unoast31.f:138-140`` ends at ``STOP '*** CONTOURS REQUIRE NRFR>1 …'``
    when option ``'o'`` meets ``NFREQ <= 1``. That is a character stop, so the
    binary exits 0 and writes no ``.prt`` — nothing is left to diagnose from.
    The deck is refusable on inspection, so refuse it."""
    import uacpy
    from uacpy.core.exceptions import ConfigurationError
    env = uacpy.Environment(
        bathymetry=200.0,
        ssp=uacpy.SoundSpeedProfile.from_pairs([(0.0, 1500.0), (200.0, 1500.0)]),
        bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                        sound_speed=1800.0, density=1.8,
                                        attenuation=0.5))
    model = uacpy.OAST()
    model.options = 'N J T o'
    with pytest.raises(ConfigurationError, match='CONTOURS REQUIRE'):
        model.run(env, uacpy.Source(depths=50.0, frequencies=500.0),
                  uacpy.Receiver(depths=[100.0], ranges=[1000.0, 2000.0]))


@pytest.mark.requires_oases
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
    model = uacpy.OAST()
    result = SimpleNamespace(stdout='', stderr=banner, returncode=0)
    with pytest.raises(ModelExecutionError,
                       match=r'Error output:\nSTOP ') as ei:
        raise_on_fortran_fatal(model.model_name, result, tmp_path,
                               'nonexistent',
                               benign=model._traits.benign_fortran_fatals)
    msg = str(ei.value)
    assert banner.split('STOP', 1)[1].strip()[:12] in msg, (
        f"the binary's own banner never reached the user: {msg[:200]}")
    assert '.prt' not in msg or 'writes no .prt' in msg, (
        f"error still points at a .prt OASES never writes: {msg[:200]}")


def _pekeris_env():
    from uacpy.tests.conftest import make_pekeris
    return make_pekeris(
        ssp=uacpy.SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.0, 1500.0)]),
        density=1.7)


@pytest.mark.requires_oases
def test_oasp_run_frequencies_honours_the_lower_band_edge():
    """``frequencies=`` sets an (freq_min, freq_max, N) triple; dropping freq_min leaves
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
    from uacpy.core.run_settings import RunMode
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


@pytest.mark.requires_oases
def test_oasp_multi_frequency_source_centres_the_sweep_on_the_band():
    """A multi-element ``source.frequencies`` names a band; its centre — not
    its first element — sets the deck fc and the derived ``2.5×fc`` sweep
    top, so the computed band covers every requested frequency.
    ``frequencies[0]`` as the centre swept [100, 200, 300] only up to
    2.5×100 = 250 Hz and never computed the top of the band."""
    import uacpy
    import warnings as _w
    from uacpy.models import OASP
    from uacpy.core.run_settings import RunMode
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
    assert field.metadata['fc'] == pytest.approx(200.0, abs=1.0)


@pytest.mark.requires_oases
def test_oasp_pinned_freq_max_beside_a_source_band_is_refused():
    """A multi-element Source names the band as ``run(frequencies=)`` does,
    so a pinned sweep edge that differs from it contradicts it: refused,
    naming both, rather than silently truncating or overriding the band."""
    import uacpy
    from uacpy.models import OASP
    from uacpy.core.run_settings import RunMode
    with pytest.raises(ConfigurationError, match='contradict each other') as exc:
        OASP(n_time_samples=512, freq_max=250.0).run_settings(
            _pekeris_env(),
            uacpy.Source(depths=25.0, frequencies=[100.0, 200.0, 300.0]),
            uacpy.Receiver(depths=np.array([50.0]),
                           ranges=np.array([1000.0])),
            run_mode=RunMode.BROADBAND)
    assert 'OASP(freq_max=250)' in str(exc.value)


@pytest.mark.requires_oases
@pytest.mark.parametrize('pin', [dict(freq_min=10.0), dict(freq_max=400.0)])
def test_oasp_run_frequencies_beside_a_pinned_edge_is_refused(pin):
    """``frequencies=`` names both band edges; a pinned edge that differs is
    refused on either side."""
    import uacpy
    from uacpy.models import OASP
    from uacpy.core.run_settings import RunMode
    name = next(iter(pin))
    with pytest.raises(ConfigurationError, match=f'OASP\\({name}='):
        OASP(n_time_samples=512, **pin).run_settings(
            _pekeris_env(), uacpy.Source(depths=25.0, frequencies=150.0),
            uacpy.Receiver(depths=np.array([50.0]),
                           ranges=np.array([1000.0, 2000.0])),
            run_mode=RunMode.BROADBAND,
            frequencies=np.linspace(100.0, 200.0, 21))


@pytest.mark.requires_oases
def test_oasp_coherent_tl_carries_the_same_phase_reference_as_broadband():
    """COHERENT_TL is one frequency slice of the same .trf array, so it
    must not come back with the phase reference dropped. ``travelling_wave``
    declares the carrier ``exp(-i k0 r)`` is still in the data, which is what
    lets a consumer put the causal arrival at ``t = r/c0``
    (``PhaseReference.TRAVELLING_WAVE``); an untagged slice is not IFFT-able."""
    import uacpy
    import warnings as _w
    from uacpy.models import OASP
    from uacpy.core.run_settings import RunMode
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

    @pytest.mark.requires_oases
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

    @pytest.mark.requires_oases
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

    def test_which_classes_clear_the_bare_sctout_fort46(self):
        """``fort.46`` is cleared by the three classes whose binary can reach
        the unguarded unit-46 writes, and by no other.

        It is not a family-wide file, and the list says which binary writes
        which name. ``OASS``'s own deck carries no ``'s'`` option, so
        ``SCTOUT`` stays ``.FALSE.`` and the ``fort.46`` left after an OASS
        run belongs to its OAST producer, which clears it. ``OASSP`` stages
        units 45/46 on the mean-field stem's ``.045``/``.046``, different
        files from the bare name, so it neither writes nor reads this one.
        Hoisting the entry onto ``OASES`` would be inert for both and would
        stop the declaration meaning anything.
        """
        import ast
        import importlib
        import inspect
        import pkgutil
        from uacpy.models import oases as package
        from uacpy.models.oases import (
            OASES, OAST, OASN, OASR, OASP, OASS, OASSP)
        clears = {cls.__name__ for cls in
                  (OAST, OASN, OASR, OASP, OASS, OASSP)
                  if 'fort.46' in cls._OUTPUT_FORT_FILES}
        assert clears == {'OAST', 'OASR', 'OASP'}
        assert 'fort.46' not in OASES._OUTPUT_FORT_FILES

        # And the three name the shared constant rather than re-spelling it:
        # a tuple literal here is where the five-line explanation drifts one
        # copy at a time. Read from the source, because CPython folds equal
        # literal tuples in one module to one object, so ``is`` cannot see it.
        spelled_out, declaring = [], set()
        nodes = [node
                 for info in pkgutil.iter_modules(package.__path__)
                 for node in ast.walk(ast.parse(inspect.getsource(
                     importlib.import_module(
                         f'{package.__name__}.{info.name}'))))]
        for node in nodes:
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                targets = getattr(stmt, 'targets', [])
                if isinstance(stmt, ast.AnnAssign):
                    targets = [stmt.target]
                names = [t.id for t in targets if isinstance(t, ast.Name)]
                if '_OUTPUT_FORT_FILES' not in names:
                    continue
                declaring.add(node.name)
                value = stmt.value
                if (isinstance(value, ast.Tuple)
                        and any(getattr(e, 'value', None) == 'fort.46'
                                for e in value.elts)):
                    spelled_out.append(node.name)
        # The scan reached every class that declares the list.
        assert {'OAST', 'OASR', 'OASP'} <= declaring
        assert not spelled_out, (
            f"{spelled_out} spell the shared fort.46 list out again instead "
            f"of naming _SCTOUT_BARE_FORT46")


@pytest.mark.requires_oases
class TestOasnWhiteNoiseDefaultThroughTheModel:
    """OASES adds ``10**(WNLEVDB/10)`` to every covariance diagonal with no
    off switch (oasnun22.f:228, :1157). The OASN model therefore defaults
    ``white_noise_level_dB`` to ``None`` — written as -200 dB, whose 1e-20
    linear power is numerically nil — while an explicit 0.0 reaches the
    deck as a literal 0 dB re 1 µPa²/Hz per sensor."""

    @staticmethod
    def _deck(tmp_path, monkeypatch, **kw):
        from uacpy.core.exceptions import ModelExecutionError
        from uacpy.models.oases import OASN
        env = uacpy.Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1700.0,
                density=1.8, attenuation=0.5))
        model = OASN(surface_noise_level_dB=70.0, verbose=False,
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

    def test_explicit_zero_reaches_the_deck_as_0_dB(self, tmp_path,
                                                    monkeypatch):
        text = self._deck(tmp_path, monkeypatch, white_noise_level_dB=0.0)
        assert '70.0 0.0 0.0 0' in text


class TestTheIntegrationWindowKnob:
    """``integration_window=(IC1, IC2)`` on OAST, OASP and OASN: the record
    holds the pair the deck carries — the given one, the whole ``(1, NW)``
    under a pinned count, ``None`` under automatic sampling — and a given
    pair under automatic sampling, where the manuals say IC1/IC2 have no
    effect, is a notice of the run."""

    _ARGS = {'OASN': (RunMode.COVARIANCE,), 'OAST': (), 'OASP': ()}

    def _engine(self, cls, **kw):
        extra = (dict(surface_noise_level_dB=50.0,
                      discrete_sources=[dict(depth=30.0, x=500.0,
                                             level_dB=60.0)])
                 if cls is OASN else {})
        with recorded_warnings() as caught:
            engine = cls(verbose=False, **extra, **kw).run_settings(
                make_pekeris(), Source(depths=50.0, frequencies=100.0),
                Receiver(depths=[30.0, 60.0], ranges=[500.0, 1000.0]),
                *self._ARGS[cls.__name__]).engine
        said = [str(w.message) for w in caught
                if 'integration_window' in str(w.message)]
        return engine, said

    @pytest.mark.parametrize('cls', [uacpy.OAST, OASP, OASN])
    def test_the_record_holds_the_pair_the_deck_carries(self, cls):
        assert self._engine(cls)[0].integration_window is None
        assert self._engine(cls, n_wavenumbers=1024)[0].integration_window \
            == (1, 1024)
        engine, said = self._engine(cls, n_wavenumbers=1024,
                                    integration_window=(10, 900))
        assert engine.integration_window == (10, 900) and not said

    @pytest.mark.parametrize('cls', [uacpy.OAST, OASP, OASN])
    def test_a_pair_under_automatic_sampling_is_said_unused(self, cls):
        engine, said = self._engine(cls, integration_window=(10, 900))
        assert len(said) == 1 and 'has no effect' in said[0]
        assert said[0] in [n.message for n in engine.notices]

    @pytest.mark.parametrize('cls', [uacpy.OAST, OASP, OASN])
    def test_a_pair_past_the_count_is_refused(self, cls):
        with pytest.raises(ConfigurationError,
                           match=r'integration_window=\(1, 2048\)'):
            cls(n_wavenumbers=1024, integration_window=(1, 2048))


@pytest.mark.requires_oases
class TestOasnReceiverGains:
    """``OASN(receiver_gains_dB=)`` reaches the array records' ``GAIN``
    column (oasnun22.f:36), and OASN multiplies each element's field by
    ``10**(GAIN/20)`` (oasnun22.f:99, :1108): each term carries the product
    of its two elements' amplitude gains, so with the white noise at its nil
    default a -6 dB element's auto-term drops 6 dB, its cross terms 3 dB,
    and the other terms not at all."""

    _CARRIERS = (Source(depths=10.0, frequencies=100.0),
                 Receiver(depths=[20.0, 50.0, 80.0], ranges=[0.0]))

    def _covariance(self, work_dir, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = OASN(verbose=False, surface_noise_level_dB=60.0,
                         work_dir=work_dir, cleanup=False, **kw)
            settings = model.run_settings(make_pekeris(), *self._CARRIERS,
                                          RunMode.COVARIANCE)
            cov = model.compute_covariance(make_pekeris(), *self._CARRIERS)
        return settings.engine, np.asarray(cov.covariance)[0]

    def test_a_gain_scales_its_elements_terms(self, tmp_path):
        _, plain = self._covariance(tmp_path / 'plain')
        engine, gained = self._covariance(tmp_path / 'gained',
                                          receiver_gains_dB=[0.0, -6.0, 0.0])
        np.testing.assert_array_equal(engine.receiver_gains_dB,
                                      [0.0, -6.0, 0.0])
        records = (tmp_path / 'gained' / 'oasn_run.dat').read_text()
        assert '50.00 0 0 1 -6\n' in records
        ratio = np.abs(gained) / np.abs(plain)
        g = 10.0 ** (-6.0 / 20.0)
        np.testing.assert_allclose(ratio[1, 1], g * g, rtol=1e-4)
        np.testing.assert_allclose(ratio[0, 1], g, rtol=1e-4)
        np.testing.assert_allclose(ratio[0, 2], 1.0, rtol=1e-4)


def _offset_notices(model):
    """The ``integration_offset`` notices ``model``'s settings carry for a
    run on a Pekeris guide (OASN: a COVARIANCE run), as warned by
    ``run_settings`` — the stage-3 notice every run says."""
    env = make_pekeris()
    src = Source(depths=50.0, frequencies=100.0)
    rcv = Receiver(depths=[30.0, 60.0], ranges=[500.0, 1000.0])
    args = (env, src, rcv) + ((RunMode.COVARIANCE,)
                              if isinstance(model, OASN) else ())
    with recorded_warnings() as caught:
        settings = model.run_settings(*args)
    warned = [str(w.message) for w in caught
              if 'integration_offset' in str(w.message)]
    recorded = [n.message for n in settings.engine.notices
                if 'integration_offset' in n.message]
    assert warned == recorded
    return warned


def _offset_notices_of_the_chain(model):
    """The ``integration_offset`` warnings an OASSP BROADBAND
    ``run_settings`` gives — its producer's notices, then its own — on a
    Pekeris guide with a rough seafloor."""
    env = Environment(bathymetry=100.0, ssp=1500.0, bottom=BoundaryProperties(
        acoustic_type='half-space', sound_speed=1600.0, density=1.6,
        attenuation=0.2, roughness=0.5))
    with recorded_warnings() as caught:
        model.run_settings(env, Source(depths=50.0, frequencies=100.0),
                           Receiver(depths=[30.0, 60.0],
                                    ranges=[500.0, 1000.0]),
                           RunMode.BROADBAND)
    return [str(w.message) for w in caught
            if 'integration_offset' in str(w.message)]


@pytest.mark.requires_oases
class TestContourOffsetUnderAutomaticSampling:
    """``unoast31.f:429`` sets ``OFFDB=0E0`` inside the ``NWVNOin < 0``
    branch (``unoasp22.f:323`` for OASP), so the binary discards a
    user-supplied contour offset and prints "THE DEFAULT CONTOUR OFFSET IS
    APPLIED". Since ``n_wavenumbers=None`` is uacpy's own default, a documented
    constructor argument silently had no effect in the default configuration.
    The manual does not say so — ``oast.tex:547-550`` names only IC1/IC2 —
    which is why the warning has to come from uacpy."""

    @pytest.mark.parametrize('cls_name', ['OAST', 'OASP'])
    def test_offset_with_automatic_sampling_is_noticed(self, cls_name):
        model = getattr(uacpy, cls_name)(integration_offset=2.0)
        got = _offset_notices(model)
        assert got and 'automatic wavenumber sampling' in got[0], got

    @pytest.mark.parametrize('cls_name', ['OAST', 'OASP'])
    @pytest.mark.parametrize('kwargs', [
        {'integration_offset': 2.0, 'n_wavenumbers': 4096},   # offset is honoured
        {'n_wavenumbers': None},                                # no offset to lose
    ])
    def test_no_notice_when_the_offset_reaches_the_kernel_or_is_unset(
            self, cls_name, kwargs):
        assert not _offset_notices(getattr(uacpy, cls_name)(**kwargs))

    def test_oasn_is_not_affected(self):
        # unoasn22.f:283 tests OFFDBIN, which its automatic branch never
        # touches, so OASN keeps the user's offset and must stay silent.
        assert not _offset_notices(uacpy.OASN(integration_offset=2.0,
                                              surface_noise_level_dB=50.0))


@pytest.mark.requires_oases
class TestLicenceWarningIsOncePerProcess:
    """Constructing any OASES sub-model emits a one-time licence/citation
    UserWarning (oases.md §9; ``base.py`` ``_warn_restricted_provenance`` dedupes
    per provenance id per process). This suite's config filters UserWarnings
    and any earlier test in the worker may already have absorbed the one
    emission, so the once-and-only-once contract is only observable in a
    fresh interpreter."""

    def test_two_constructions_warn_exactly_once(self):
        import subprocess
        import sys
        import textwrap
        code = textwrap.dedent("""
            import warnings
            from uacpy.models import OAST
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                OAST(verbose=False)
                OAST(verbose=False)
            print(sum('Cite:' in str(w.message) for w in caught))
        """)
        proc = subprocess.run([sys.executable, '-c', code],
                              capture_output=True, text=True, timeout=180)
        assert proc.returncode == 0, proc.stderr[-500:]
        assert proc.stdout.strip().splitlines()[-1] == '1', (
            f"expected exactly one licence warning; stdout={proc.stdout!r}")


@pytest.mark.requires_oases
class TestOASPSweepDerivation:
    """The Block III/VIII sweep the deck is written from (oases.md §10), read
    from ``run_settings(...).engine`` — the settings the writer receives:
    the default sweep of a one-carrier BROADBAND run is ``0 .. 2.5·fc`` on
    4096 samples; a requested band (``frequencies=``, a TIME_SERIES pulse,
    the source frequency of a COHERENT_TL run) sets ``FR1``/``FR2`` and the
    smallest power of two ``NX`` whose bins are as fine as the requested
    spacing (``NT = 2^M``, oasp.tex:129), or a pinned ``n_time_samples``
    as given. No binary runs."""

    @staticmethod
    def _settings(model, **run_kwargs):
        env = uacpy.Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0,
                density=1.8, attenuation=0.5))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return model.run_settings(
                env, uacpy.Source(depths=25.0, frequencies=100.0),
                uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
                **run_kwargs)

    def test_freq_max_none_derives_two_and_a_half_fc(self):
        engine = self._settings(OASP(verbose=False),
                                run_mode=RunMode.BROADBAND).engine
        assert engine.freq_max == pytest.approx(2.5 * 100.0)
        assert engine.n_time_samples == 4096

    def test_pinned_freq_max_reaches_the_deck_verbatim(self):
        engine = self._settings(OASP(freq_max=300.0, verbose=False),
                                run_mode=RunMode.BROADBAND).engine
        assert engine.freq_max == pytest.approx(300.0)

    def test_explicit_frequencies_derive_the_power_of_two_of_their_spacing(
            self):
        # df = 5 Hz up to 200 Hz needs ceil(2·200/5) + 2 = 82 samples,
        # rounded up to the next power of two. The Nyquist is lifted by
        # NX/(NX-2) so OASP's top bin, DLFREQ*(NX/2-1), sits on the 200 Hz
        # edge (unoasp22.f:237-246).
        engine = self._settings(
            OASP(verbose=False), run_mode=RunMode.BROADBAND,
            frequencies=np.linspace(50.0, 200.0, 31)).engine
        assert engine.n_time_samples == 128
        assert engine.freq_max == pytest.approx(200.0 * 128 / 126, rel=1e-5)
        top_bin = 2.0 * engine.freq_max / 128 * (128 // 2 - 1)
        assert 200.0 <= top_bin < 200.0 + 200.0 / 126

    @pytest.mark.parametrize('pinned, coarser', [(64, True), (128, False),
                                                 (256, False)])
    def test_a_pinned_count_is_written_as_given(self, pinned, coarser):
        """64 samples give bins 6.45 Hz apart against the 5 Hz asked for:
        the count is kept and a notice names the 128 that reaches it."""
        env = uacpy.Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0,
                density=1.8, attenuation=0.5))
        with recorded_warnings() as caught:
            engine = OASP(n_time_samples=pinned, verbose=False).run_settings(
                env, uacpy.Source(depths=25.0, frequencies=100.0),
                uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
                run_mode=RunMode.BROADBAND,
                frequencies=np.linspace(50.0, 200.0, 31)).engine
        assert engine.n_time_samples == pinned
        assert engine.n_time_samples_origin == 'OASP(n_time_samples=…)'
        said = [str(w.message) for w in caught
                if 'coarser than the requested spacing' in str(w.message)]
        assert bool(said) is coarser, said
        if coarser:
            assert 'n_time_samples=128 or more' in said[0]

    def test_coherent_tl_runs_the_sweep_frequencies_f_would_run(self):
        """OASES-1: a COHERENT_TL run asks for its source frequency, so its
        sweep is the one ``frequencies=[f]`` gives — ``FR1 = f`` and the bin
        at ``f`` (``unoasp22.f:237-248``) — not the 2047-bin default sweep
        whose nearest bin sat at 99.976 Hz (29 deg of phase at 5 km)."""
        model = OASP(verbose=False)
        default = self._settings(model)
        pinned = self._settings(model, frequencies=[100.0])
        assert default.engine == pinned.engine
        assert default.engine.freq_min == 100.0
        assert default.frequencies.size == 1
        assert abs(default.frequencies[0] - 100.0) <= 2e-6 * 100.0
        assert default.engine.marched_frequencies.size <= 2

    def test_an_unpinned_nx_gives_the_record_the_duration_asks_for(self):
        """OASES-6: without ``n_time_samples``, a TIME_SERIES run sizes NX
        from the pulse grid's ``Δf = 1/output_duration`` alone, so the
        record ``1/DLFREQ`` is ``output_duration`` rounded up to a power of
        two of samples — not the 4096-sample floor that made a 1 s request
        a 7.5 s record."""
        fs = 1000.0
        t = np.arange(200) / fs
        pulse = np.hanning(200) * np.sin(2 * np.pi * 100.0 * t)
        settings = self._settings(
            OASP(verbose=False), run_mode=RunMode.TIME_SERIES,
            source_waveform=pulse, sample_rate=fs, output_duration=1.0)
        engine = settings.engine
        record = engine.n_time_samples / (2.0 * engine.freq_max)
        assert 1.0 <= record < 2.0, (engine.n_time_samples, record)
        assert engine.n_time_samples < 4096

    def test_freq_min_defaults_to_none_and_the_sweep_starts_at_zero(self):
        assert OASP.__init__.__kwdefaults__['freq_min'] is None
        engine = self._settings(OASP(verbose=False),
                                run_mode=RunMode.BROADBAND).engine
        assert engine.freq_min == 0.0

    @pytest.mark.parametrize('pin', [dict(freq_min=0.0),
                                     dict(freq_max=300.0)])
    def test_an_edge_pinned_beside_a_named_band_is_refused(self, pin):
        """An explicit freq_min=0, which the old 0.0 default could not tell
        from unset, contradicts run(frequencies=) as freq_max does."""
        name, value = next(iter(pin.items()))
        with pytest.raises(ConfigurationError,
                           match='contradict each other') as exc:
            self._settings(OASP(verbose=False, **pin),
                           run_mode=RunMode.BROADBAND,
                           frequencies=np.linspace(80.0, 120.0, 5))
        assert f'OASP({name}={value:g})' in str(exc.value)

    def test_an_edge_equal_to_the_named_band_is_taken(self):
        engine = self._settings(OASP(verbose=False, freq_min=80.0),
                                run_mode=RunMode.BROADBAND,
                                frequencies=np.linspace(80.0, 120.0, 5)).engine
        assert engine.freq_min == 80.0

    def test_an_edge_beside_a_derived_band_is_said_unused(self):
        fs = 1000.0
        t = np.arange(200) / fs
        pulse = np.hanning(200) * np.sin(2 * np.pi * 100.0 * t)
        env = uacpy.Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0,
                density=1.8, attenuation=0.5))
        with recorded_warnings() as caught:
            OASP(verbose=False, freq_max=300.0).run_settings(
                env, uacpy.Source(depths=25.0, frequencies=100.0),
                uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
                run_mode=RunMode.TIME_SERIES, source_waveform=pulse,
                sample_rate=fs, output_duration=1.0)
        said = [str(w.message) for w in caught
                if 'OASP(freq_max=300) is not used' in str(w.message)]
        assert len(said) == 1, said

    def test_a_pinned_nx_is_written_as_given_on_a_pulse_band(self):
        fs = 1000.0
        t = np.arange(200) / fs
        pulse = np.hanning(200) * np.sin(2 * np.pi * 100.0 * t)
        engine = self._settings(
            OASP(n_time_samples=4096, verbose=False),
            run_mode=RunMode.TIME_SERIES, source_waveform=pulse,
            sample_rate=fs, output_duration=1.0).engine
        assert engine.n_time_samples == 4096


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

    @pytest.mark.requires_oases
    @pytest.mark.parametrize('reflection_type', [None, 'P-P', 'transmission'])
    def test_usable_reflection_types_are_untouched(self, reflection_type):
        # The discriminating half — measured max|R| 0.999947 (P-P) and
        # 1.273706 (transmission); the guard must not reach them.
        kwargs = ({} if reflection_type is None
                  else {'reflection_type': reflection_type})
        uacpy.OASR(**kwargs)

    def test_the_docstrings_list_the_refused_types_as_refused(self):
        # help(OASR) and the §18 table must not offer 'P-SV'/'P-Slow' as
        # values that run: every paragraph naming them names the refusal.
        import inspect
        import re
        # Each engine documents its constructor once (class or __init__).
        docs = [d for d in (uacpy.OASR.__doc__, uacpy.OASR.__init__.__doc__)
                if d and 'reflection_type :' in d]
        assert docs, 'no docstring documents reflection_type'
        for doc in docs:
            section = inspect.cleandoc(doc).split('reflection_type :')[1]
            # The entry ends at the next unindented parameter line.
            section = re.split(r'\n\S', section)[0]
            assert "'P-SV'" in section
            assert 'UnsupportedFeatureError' in section


@pytest.mark.requires_oases  # runs OASP
class TestOASPFrequencyGridSubstitution:
    """OASP is a pulse model: its frequency axis is the FFT ladder implied by
    the time window, not the bins the caller names. Asking for 3 returned 820
    (via ``run(frequencies=)``) or 2047 (via ``Source``) with no warning,
    while Kraken on the identical call returns exactly 3 — so a caller who
    has used one has no reason to expect the other."""

    @staticmethod
    def _env():
        return make_pekeris(sound_speed=1800.0)

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

    @pytest.mark.parametrize('run_mode', ['BROADBAND', 'COHERENT_TL'])
    def test_the_complex_field_is_complex128_like_every_engine(self,
                                                               run_mode):
        # The .trf payload is COMPLEX*8; Kraken, Scooter and Bellhop return
        # complex128, and so does OASP on both complex routes.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            field = uacpy.OASP().run(
                self._env(), uacpy.Source(depths=25.0, frequencies=200.0),
                uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
                run_mode=getattr(uacpy.RunMode, run_mode))
        assert field.data.dtype == np.complex128

    def test_single_frequency_names_nothing_and_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            warnings.filterwarnings('ignore', message='.*not redistributable.*')
            warnings.filterwarnings('ignore', message='.*licence.*')
            uacpy.OASP().run(
                self._env(), uacpy.Source(depths=25.0, frequencies=200.0),
                uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
                run_mode=uacpy.RunMode.BROADBAND)


@pytest.mark.requires_oases
class TestOASRContourOptionIsLogSwept:
    """`oasr.tex:135` documents option 'C' as a *plot* option ("Loss contours
    plotted in frequency and grazing angle"), but `unoasr21.f:123-125`
    computes F1LOG/DFLOG and `:243` evaluates the coefficients at
    ``EXP(F1LOG + (JJ-1)*DFLOG)`` — it changes the frequencies the physics is
    computed at. `oast.tex:200-203` confirms it from the other side, calling
    'C' the way to get "consistent logarithmic sampling".

    Under 'C' the sweep is therefore checked for log spacing: a linear
    request, whose bins the deck cannot carry (4 of 5 would be wrong), is
    refused, and a log-spaced one runs as given."""

    @staticmethod
    def _env():
        return make_pekeris(sound_speed=1600.0)

    def _run(self, options, freqs):
        kwargs = {} if options is None else {'options': options}
        return uacpy.OASR(**kwargs).run(
            self._env(), uacpy.Source(depths=25.0, frequencies=freqs),
            uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
            run_mode=uacpy.RunMode.REFLECTION)

    def test_contour_option_with_a_linear_request_is_refused(self):
        with pytest.raises(ConfigurationError,
                           match='makes the frequency sweep logarithmic'):
            self._run('N T C', np.linspace(10.0, 10000.0, 5))

    def test_contour_option_with_a_log_request_is_silent(self):
        # The correct half: 'C' with a log request is what the option is for,
        # so the check must stay silent here.
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            warnings.filterwarnings('ignore', message='.*licence.*')
            warnings.filterwarnings('ignore', message='.*redistributable.*')
            self._run('N T C', np.geomspace(10.0, 10000.0, 5))

    def test_without_the_option_a_linear_request_is_silent(self):
        # The discriminating counterpart: the linear sweep is correct without
        # 'C', so this check must not reach it.
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
        return make_pekeris(sound_speed=1600.0)

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
    def test_a_fitting_option_line_writes(self, options, tmp_path):
        # Exactly 40 is legal — the bound is inclusive, and an off-by-one here
        # would refuse a deck GETOPT reads perfectly.
        self._write(options, tmp_path)
        assert (tmp_path / 't.dat').exists()


# ─── The contour offset has two ways of never reaching the integration ─────


@pytest.mark.requires_oases
class TestContourOffsetIsWarnedOnBothDiscardPaths:
    """``unoast31.f:126-133`` reads the frequency line WITH the offset token
    only under 'J', 'O' or 'd'; OASP/OASSP only under 'J' or 'd'
    (``unoasp22.f:126-133``, ``unoassp30.f:135-142``). Without one of them the
    binary zeroes the offset before the read, so a value uacpy wrote into the
    deck is not even parsed — the same silent discard as automatic wavenumber
    sampling, which the warning already covered."""

    def test_oast_without_j_warns(self):
        got = _offset_notices(uacpy.OAST(integration_offset=0.5,
                                         n_wavenumbers=1024,
                                         complex_contour=False))
        assert got and 'option line' in got[0], got

    def test_oast_with_j_and_pinned_sampling_is_quiet(self):
        assert not _offset_notices(uacpy.OAST(integration_offset=0.5,
                                              n_wavenumbers=1024))

    def test_oast_warns_under_automatic_sampling(self):
        got = _offset_notices(uacpy.OAST(integration_offset=0.5))
        assert got and 'automatic wavenumber sampling' in got[0], got

    def test_oasp_raw_options_without_j_warns(self):
        got = _offset_notices(uacpy.OASP(integration_offset=0.5,
                                         n_wavenumbers=1024, options='N'))
        assert got and 'option line' in got[0], got

    def test_oasp_default_option_line_is_quiet(self):
        # The writer's default is 'N J', so nothing is discarded.
        assert not _offset_notices(uacpy.OASP(integration_offset=0.5,
                                              n_wavenumbers=1024))

    def test_no_offset_never_warns(self):
        assert not _offset_notices(uacpy.OAST())

    def test_oasn_reads_the_offset_under_lowercase_j(self):
        """OASN's GETOPT sets ICNTIN for 'J' or 'j' (unoasn22.f:673-675),
        and the frequency line is read with the offset under ICNTIN > 0
        (:141-142), so 'j' carries it."""
        assert not _offset_notices(uacpy.OASN(
            options='j', integration_offset=0.5, surface_noise_level_dB=50.0))

    def test_oasn_without_either_j_warns(self):
        got = _offset_notices(uacpy.OASN(
            options='F', integration_offset=0.5, surface_noise_level_dB=50.0))
        assert got and 'option line' in got[0], got

    @pytest.mark.parametrize('cls_name', ['OAST', 'OASP', 'OASN'])
    def test_a_knob_reassigned_after_construction_is_noticed_by_the_run(
            self, cls_name):
        """The notice is a stage-3 one, so the run reads the knobs it runs
        with, not the ones the constructor saw (the class re-checks every
        knob per run since the attributes can be reassigned)."""
        kwargs = {'surface_noise_level_dB': 50.0} if cls_name == 'OASN' else {}
        model = getattr(uacpy, cls_name)(**kwargs)
        assert not _offset_notices(model)
        model.integration_offset = 0.5
        model.options = 'F' if cls_name == 'OASN' else 'K T'
        if cls_name == 'OASP':
            model.options, model.n_wavenumbers = 'N', 1024
        got = _offset_notices(model)
        assert got and 'option line' in got[0], got

    def test_the_oassp_chain_says_it_once(self):
        """OASSP's default producer runs with OASSP's offset; OASSP's own
        settings say whether it reaches the binaries, and the producer's do
        not repeat it."""
        got = _offset_notices_of_the_chain(uacpy.OASSP(
            correlation_length=5.0, integration_offset=0.5))
        assert len(got) == 1 and got[0].startswith('OASSP('), got


# ─── OASR: the parameter letter and the transmission flag are independent ──


def _resolve_reflection_type_of(model):
    """The coefficient an OASR model's deck computes."""
    from uacpy.models.oases.oasr import _resolve_reflection_type
    return _resolve_reflection_type(model.options, model.reflection_type)


@pytest.mark.requires_oases
class TestReflectionTypeProvenance:
    """``unoasr21.f:349-378``: ``N``/``S``/``B`` each assign ``IPARM`` (the
    wave parameter) while ``'t'`` flips the separate ``transmit`` flag, and
    ``oasjun21.f:80-84`` then reads ``trcoef(iparm)`` instead of
    ``rfcoef(iparm)``. Treating the four as mutually exclusive made
    ``'N T t'`` match twice and fall to the ``'P-P'`` fallback, recording a
    reflection label on a transmission payload."""

    @staticmethod
    def _rt(**kw):
        return _resolve_reflection_type_of(uacpy.OASR(**kw))

    def test_the_transmission_letter_reads_the_parameters_transmission(
            self):
        # 't' reads trcoef(IPARM): the P-P one under 'N', the P-SV one under
        # 'S' (oasjun21.f:80-84), so each is labelled by what it is.
        assert self._rt(options='N T t') == 'transmission'
        assert self._rt(options='T t') == 'transmission'
        assert self._rt(options='S T t') == 'P-SV transmission'
        assert self._rt(options='N S T t') == 'P-SV transmission'
        assert self._rt(options='S N T t') == 'transmission'

    def test_the_p_sv_transmission_is_asked_for_by_name(self, tmp_path):
        from uacpy.io.oases_writer import write_oasr_input
        model = uacpy.OASR(reflection_type='P-SV transmission')
        assert _resolve_reflection_type_of(model) == 'P-SV transmission'
        write_oasr_input(tmp_path / 'r.dat', _oast_pekeris_env(),
                         _oast_source(), _oast_receiver(),
                         reflection_type='P-SV transmission')
        assert (tmp_path / 'r.dat').read_text().splitlines()[1].split() == [
            'S', 't', 'T']

    def test_a_reflection_line_keeps_its_parameter(self):
        assert self._rt(options='N T') == 'P-P'
        assert self._rt(options='S T') == 'P-SV'
        assert self._rt(options='B T') == 'P-Slow'

    def test_the_last_parameter_letter_wins(self):
        # GETOPT reassigns IPARM on every parameter letter it meets.
        assert self._rt(options='N S T') == 'P-SV'
        assert self._rt(options='S N T') == 'P-P'

    def test_no_parameter_letter_falls_back_to_the_oases_default(self):
        assert self._rt(options='T') == 'P-P'

    def test_the_typed_argument_wins_when_no_raw_line(self):
        assert self._rt(reflection_type='transmission') == 'transmission'
        assert self._rt() == 'P-P'


@pytest.mark.requires_oases
def test_slowness_sampling_is_refused_after_a_post_construction_mutation():
    """``_reject_slowness_sampling`` runs in ``__init__``, so the ``run``
    path's '.rco'-first output search could never be reached — it was dead.
    Removing it means the invariant has to hold at run time too, which a
    caller mutating ``options`` afterwards would otherwise break."""
    env = uacpy.Environment(
        bathymetry=100.0, ssp=1500.0,
        bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                        sound_speed=1800.0, density=1.8,
                                        attenuation=0.3))
    model = uacpy.OASR(angles_deg=np.linspace(5.0, 85.0, 9))
    model.options = 'N p T'
    with pytest.raises(UnsupportedFeatureError, match="option 'p'"):
        model.run(env, uacpy.Source(depths=50.0, frequencies=100.0),
                  uacpy.Receiver(depths=[50.0], ranges=[1000.0]))


def _registered_carriers(key):
    """The 100 m guide the registered OASES engine ``key`` runs over."""
    from uacpy.models._registry import ENGINES
    bottom = dict(ENGINES[key].example_bottom) or dict(
        acoustic_type='half-space', sound_speed=1700.0, density=1.8,
        attenuation=0.5)
    return (uacpy.Environment(bathymetry=100.0, ssp=1500.0,
                              bottom=uacpy.BoundaryProperties(**bottom)),
            uacpy.Source(depths=50.0, frequencies=100.0),
            uacpy.Receiver(depths=[20.0, 50.0], ranges=[200.0, 400.0]))


def _registered_engine(key, **kwargs):
    from uacpy.models._registry import ENGINES
    entry = ENGINES[key]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return entry.load()(**{**dict(entry.example_kwargs), **kwargs})


class TestTheReplayPinsTheGridOastAndOasrRecord:
    """OAST's ``range_min`` / ``range_max`` and OASR's ``angles_deg`` are
    recorded under the knobs' own names, so ``from_run_settings`` pins the
    grid the run used instead of deriving it again from the next call."""

    @staticmethod
    def _replayed(key, model_class):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            first = _registered_engine(key).run_settings(
                *_registered_carriers(key))
            return first, model_class.from_run_settings(first)

    def test_oast_pins_its_native_range_grid(self):
        first, again = self._replayed('oast', uacpy.OAST)
        assert (again.range_min, again.range_max) == (0.0, 400.0)
        assert (again.range_min, again.range_max) == (
            first.engine.range_min, first.engine.range_max)

    def test_oasr_pins_its_angle_grid(self):
        first, again = self._replayed('oasr', uacpy.OASR)
        assert again.angles_deg is not None and again.angles_deg.size == 181
        np.testing.assert_array_equal(again.angles_deg,
                                      first.engine.angles_deg)


@pytest.mark.requires_oases
class TestAReassignedKnobIsCheckedByEveryRun:
    """A knob the constructor refuses is refused again when it is assigned
    after construction: ``validate_inputs`` and ``run`` both re-run the
    constructor's checks in stage 2 (``_check_knobs``), with the
    constructor's message, before any deck is written."""

    @pytest.mark.parametrize('key,ctor,name,bad,match', [
        ('oast', {}, 'vrec', 5.0, "cannot reach the binary"),
        ('oasr', {}, 'reflection_type', 'P-SV', 'column of zeros'),
        ('oasp', {'freq_max': 200.0}, 'freq_min', 300.0, 'is not below'),
        ('oassp', {}, 'correlation_length', 0.0, 'degenerate'),
        ('oassp', {}, 'roughness_spectrum', 'Pink',
         r"OASSP\(roughness_spectrum='Pink'\)"),
        ('oass', {}, 'roughness_spectrum', 'Pink',
         r"OASS\(roughness_spectrum='Pink'\)"),
    ])
    def test_the_run_refuses_what_the_constructor_refuses(
            self, key, ctor, name, bad, match):
        env, src, rcv = _registered_carriers(key)
        model = _registered_engine(key, **ctor)
        setattr(model, name, bad)
        with pytest.raises((ConfigurationError, UnsupportedFeatureError),
                           match=match):
            model.validate_inputs(env, src, rcv)
        with pytest.raises((ConfigurationError, UnsupportedFeatureError),
                           match=match):
            model.run(env, src, rcv)
        with pytest.raises((ConfigurationError, UnsupportedFeatureError),
                           match=match):
            _registered_engine(key, **{**ctor, name: bad})

    @pytest.mark.parametrize('key', ['oast', 'oasp', 'oasn', 'oass',
                                     'oassp'])
    @pytest.mark.parametrize('kw, match', [
        (dict(c_low=2000.0, c_high=1000.0), 'requires c_low < c_high'),
        (dict(c_low=1500.0, c_high=1500.0), 'requires c_low < c_high'),
        (dict(c_low=0.0), r'c_low \(m/s\) must be positive'),
        (dict(c_low=-5.0), r'c_low \(m/s\) must be positive'),
        (dict(c_high=float('nan')), r'c_high \(m/s\) must be positive'),
    ])
    def test_the_constructor_refuses_a_window_no_integral_can_take(
            self, key, kw, match):
        """CMIN / CMAX bound the wavenumber integral (``k_max = ω/CMIN``,
        ``k_min = ω/CMAX``): each pinned bound is a positive finite speed,
        and a pinned pair is ordered."""
        with pytest.raises(ConfigurationError, match=match):
            _registered_engine(key, **kw)

    @pytest.mark.parametrize('key, kw, match', [
        ('oass', dict(correlation_length=0.0), 'must be a positive length'),
        ('oasp', dict(n_time_samples=-4), 'n_time_samples must be an integer'),
        ('oasp', dict(n_time_samples=2.5), 'n_time_samples must be an integer'),
        ('oasp', dict(n_time_samples=100),
         'n_time_samples=100 is not a power of two'),
        ('oassp', dict(n_time_samples=500),
         'n_time_samples=500 is not a power of two'),
        ('oasp', dict(freq_min=-10.0), 'freq_min=-10.0 must be a finite'),
        ('oasp', dict(freq_max=0.0), 'freq_max'),
        ('oasr', dict(plot_angle_step=2.5),
         'OASR: plot_angle_step must be an integer >= 0'),
        ('oasr', dict(plot_frequency_step=-1),
         'OASR: plot_frequency_step must be an integer >= 0'),
        ('oasn', dict(replica_nx=2.5), 'OASN: replica_nx must be an integer'),
        ('oasn', dict(replica_ny=0), 'OASN: replica_ny must be an integer'),
        ('oasn', dict(replica_nz=3.7), 'OASN: replica_nz must be an integer'),
        ('oass', dict(interface=2.5), 'OASS: interface must be an integer'),
        ('oassp', dict(interface=2.5), 'OASSP: interface must be an integer'),
        ('oassp', dict(realization=1.5),
         'OASSP: realization must be an integer'),
        ('oassp', dict(realization=-1),
         'OASSP: realization must be an integer'),
        ('oassp', dict(n_time_samples=512.5),
         'OASSP: n_time_samples must be an integer'),
        ('oasp', dict(integrand_plot_step=2.7),
         'OASP: integrand_plot_step must be an integer >= 0'),
        ('oasp', dict(integrand_plot_step=-3),
         'OASP: integrand_plot_step must be an integer >= 0'),
        ('oast', dict(n_wavenumbers=70000), 'exceeds OASES'),
        ('oasp', dict(n_wavenumbers=70000), 'exceeds OASES'),
    ])
    def test_a_length_band_or_fft_size_no_run_can_use_is_refused(
            self, key, kw, match):
        """OASS took the correlation lengths OASSP refuses; OASP truncated
        a fractional FFT length and wrote a negative band edge as given."""
        with pytest.raises(ConfigurationError, match=match):
            _registered_engine(key, **kw)

    @pytest.mark.parametrize('key', ['oass', 'oassp'])
    def test_a_negative_correlation_length_is_volume_scattering_refused_alike(
            self, key):
        """OASES reads a negative CL as the volume-scattering record
        (oaseun31.f:76-90); OASS and OASSP refuse it with one type."""
        with pytest.raises(UnsupportedFeatureError,
                           match='volume scattering'):
            _registered_engine(key, correlation_length=-5.0)

    @pytest.mark.parametrize('key, kw', [
        ('oasp', dict(integrand_plot_step=0)),
        ('oast', dict(n_wavenumbers=65536)),
        ('oass', dict(correlation_length=10.0)),
        ('oasp', dict(n_time_samples=np.int64(512))),
        ('oasp', dict(freq_min=0.0)),
        ('oasr', dict(plot_angle_step=0, plot_frequency_step=np.int64(0))),
        ('oasn', dict(replica_nx=1, replica_ny=np.int64(1), replica_nz=1)),
        ('oass', dict(interface=np.int64(1))),
        ('oassp', dict(interface=1, realization=np.int64(0))),
    ])
    def test_the_values_at_the_bounds_are_taken(self, key, kw):
        model = _registered_engine(key, **kw)
        assert all(getattr(model, k) == v for k, v in kw.items())

    @pytest.mark.parametrize('key', ['oast', 'oasp', 'oasn', 'oass'])
    def test_a_pinned_bound_past_the_derived_one_is_refused(self, key):
        """A single pinned ``c_high`` is held to the ``c_low`` the run
        derives (from the water column, or OASS's mean field's CMIN), and
        the refusal names it."""
        env, src, rcv = _registered_carriers(key)
        model = _registered_engine(key, c_high=1000.0)
        with pytest.raises(ConfigurationError,
                           match='pinned c_high=1000.0 m/s is at or below'):
            model.validate_inputs(env, src, rcv)

    def test_an_oass_c_low_above_the_deck_cmax_is_refused(self):
        """OASS pairs a pinned ``c_low`` with the 1e8 m/s CMAX its deck
        writes when ``c_high`` is unset; a ``c_low`` above it would empty
        the wavenumber window, one below it is taken."""
        env, src, rcv = _registered_carriers('oass')
        model = _registered_engine('oass', c_low=1e9)
        with pytest.raises(ConfigurationError,
                           match='pinned c_low=1000000000.0 m/s is at or '
                                 'above'):
            model.validate_inputs(env, src, rcv)
        model = _registered_engine('oass', c_low=9e7)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.validate_inputs(env, src, rcv)

    @pytest.mark.parametrize('key', ['oast', 'oasp', 'oasn'])
    def test_an_ordered_window_is_accepted(self, key):
        env, src, rcv = _registered_carriers(key)
        model = _registered_engine(key, c_low=1499.0, c_high=1500.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.validate_inputs(env, src, rcv)

    @pytest.mark.parametrize('key', ['oassp', 'oass'])
    def test_a_reassigned_spectrum_is_compared_case_folded(self, key):
        """A mixed-case spelling assigned after construction passes the
        check and reaches the deck as the letter it names."""
        env, src, rcv = _registered_carriers(key)
        model = _registered_engine(key)
        model.roughness_spectrum = 'Goff-Jordan'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.validate_inputs(env, src, rcv)
        from uacpy.models.oases.oass import _oass_options
        from uacpy.models.oases.oassp import _oassp_options
        options = (_oassp_options(model.options, model.scattered_only,
                                  model.roughness_spectrum)
                   if key == 'oassp' else
                   _oass_options(model.options, RunMode.REVERBERATION,
                                 model.roughness_spectrum,
                                 model.multiple_scattering))
        assert 'g' in options.split()


@pytest.mark.requires_oases
def test_the_oast_plotted_frequencies_survive_an_xarray_export():
    """OAST records the ``Freq:`` labels of its ``.plp`` curves as
    ``metadata['plotted_frequencies']``, a key ``to_xarray`` keeps; under the
    identity attribute's name ``frequencies`` the export dropped it."""
    pytest.importorskip('xarray')
    env, src, rcv = _registered_carriers('oast')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        field = _registered_engine('oast').run(env, src, rcv)
        exported = field.to_xarray()
    np.testing.assert_array_equal(field.metadata['plotted_frequencies'],
                                  [100.0])
    np.testing.assert_array_equal(exported.attrs['plotted_frequencies'],
                                  [100.0])


# ─── A derived sweep edge needs the same band check as a pinned one ────────


@pytest.mark.requires_oases
class TestOaspBandMustFitUnderTheSweepEdge:
    """The guard sat in an ``elif``, so only a pinned ``freq_max`` was
    checked. The derived edge is ``2.5*fc``, which clears the band whenever
    fc is the band centre — but ``fc`` pins fc, and
    ``OASP(fc=10)`` with a 1000 Hz source wrote FR2 = 25 Hz and
    returned a Field labelled with a ~25 Hz bin."""

    @staticmethod
    def _env():
        return make_pekeris(sound_speed=1800.0, attenuation=0.3)

    @staticmethod
    def _run(**kw):
        frequencies = kw.pop('f')
        return uacpy.OASP(**kw).run(
            TestOaspBandMustFitUnderTheSweepEdge._env(),
            uacpy.Source(depths=50.0, frequencies=frequencies),
            uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
            uacpy.RunMode.BROADBAND)

    def test_a_centre_frequency_below_the_band_is_refused(self):
        with pytest.raises(ConfigurationError, match='derived sweep edge'):
            self._run(f=1000.0, fc=10.0)

    def test_a_pinned_edge_below_the_band_is_refused(self):
        with pytest.raises(ConfigurationError, match='pinned sweep edge'):
            self._run(f=1000.0, fc=100.0, freq_max=200.0)

    def test_the_band_centre_default_always_clears_its_own_band(self):
        # fc is the midpoint, so 2.5*fc > freq_max by construction — the derived
        # edge must not start refusing ordinary runs.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            field = self._run(f=np.linspace(50.0, 150.0, 5))
        assert np.max(np.asarray(field.coords['frequency'])) > 150.0


@pytest.mark.parametrize('factor', [2.5, 3.0])
def test_the_sweep_edge_refusals_state_the_factor_the_edge_is_derived_with(
        monkeypatch, factor):
    """``OASP_FREQ_MAX_PER_CENTER`` decides the derived edge and the text of
    both refusals that name it."""
    from uacpy.models.oases import oasp
    monkeypatch.setattr(oasp, 'OASP_FREQ_MAX_PER_CENTER', factor)
    env = TestOaspBandMustFitUnderTheSweepEdge._env()
    call = (env, uacpy.Source(depths=50.0, frequencies=1000.0),
            uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
            uacpy.RunMode.BROADBAND)
    with pytest.raises(ConfigurationError, match=re.escape(
            f'freq_max={factor:g}×fc={factor * 10.0:.1f} Hz')):
        uacpy.OASP(fc=10.0).run_settings(*call)
    with pytest.raises(ConfigurationError, match=re.escape(
            f'leave it None to derive {factor:g}×fc')):
        uacpy.OASP(fc=100.0, freq_max=200.0).run_settings(
            *call)


# ─── A pinned work_dir must not hand on a previous run's kernels ───────────


@pytest.mark.requires_oases
class TestOptionSKernelFilesAreClearedBeforeLaunch:
    """``_oases_subprocess_env`` gives every run FOR045, and option 's''s
    SCTOUT dump fills it; OASP additionally writes the ``.046`` companion
    (``unoasp22.f:253``). Both are *inputs* to the OASS/OASSP chain — its
    mean-field launch accepts whatever ``<stem>.045`` it finds — so a pair
    left by an earlier run in a pinned ``work_dir`` can be handed on as this
    run's kernels whenever the producer stops before it opens the units.
    Neither suffix was in ``_OUTPUT_SUFFIXES``, so neither was cleared.

    The subprocess is stubbed out: what is under test is that the launch
    path removes the files, and letting a real binary rewrite them would
    hide the very case that matters (the run that never gets that far)."""

    _PRODUCERS = [('OAST', 'oast_run', ('.045',)),
                  ('OASR', 'oasr_run', ('.045',)),
                  ('OASP', 'oasp_run', ('.045', '.046'))]

    @pytest.mark.parametrize('cls_name,stem,suffixes', _PRODUCERS)
    def test_a_previous_runs_kernels_are_removed(self, cls_name, stem,
                                                 suffixes, tmp_path,
                                                 monkeypatch):
        model = getattr(uacpy, cls_name)(work_dir=tmp_path, cleanup=False)
        monkeypatch.setattr(type(model), '_run_subprocess',
                            lambda *a, **k: _FakeProc())
        (tmp_path / f'{stem}.dat').write_text('deck')
        for suffix in suffixes:
            (tmp_path / f'{stem}{suffix}').write_bytes(b'PREVIOUS RUN')
        model._execute(stem, tmp_path)
        for suffix in suffixes:
            assert not (tmp_path / f'{stem}{suffix}').exists(), (
                f"{cls_name} carried {stem}{suffix} forward from an earlier "
                f"run")

    @pytest.mark.parametrize('cls_name,kw', [
        ('OAST', {}), ('OASR', {}), ('OASP', {}), ('OASN', {}),
        ('OASSP', dict(correlation_length=5.0, spectral_exponent=2.5))])
    def test_a_previous_runs_bare_fort_files_are_removed(self, cls_name, kw,
                                                        tmp_path,
                                                        monkeypatch):
        """The files a program writes under a bare unit name (``fort.46``
        for option 's', OASSP's scratch units) are not named after the
        stem, so the launch clears them by name
        (``_OUTPUT_FORT_FILES``)."""
        model = getattr(uacpy, cls_name)(work_dir=tmp_path, cleanup=False,
                                         **kw)
        monkeypatch.setattr(type(model), '_run_subprocess',
                            lambda *a, **k: _FakeProc())
        names = type(model)._OUTPUT_FORT_FILES
        assert names
        for name in names:
            (tmp_path / name).write_bytes(b'PREVIOUS RUN')
        model._execute('x_run', tmp_path)
        assert [n for n in names if (tmp_path / n).exists()] == []

    @pytest.mark.parametrize('cls_name,stem,suffixes', _PRODUCERS)
    def test_the_suffixes_are_declared(self, cls_name, stem, suffixes):
        declared = getattr(uacpy, cls_name)._OUTPUT_SUFFIXES
        assert set(suffixes) <= set(declared), declared


# ─── Block VII's phase-speed window is reachable from the model ────────────


@pytest.mark.requires_oases
class TestPhaseSpeedWindowReachesTheDeck:
    """``CMIN``/``CMAX`` (Block VII, ``unoast31.f:213``) default to the window
    of :func:`~uacpy.core.engine_defaults.default_c_low`: the slowest speed
    of every medium, shear speeds included, × 0.85 over an elastic
    half-space and × 0.90 otherwise (``scooter.f90:193-259``), and 1E8. The
    model's ``c_low=``/``c_high=`` pin either edge, and a pinned edge reaches
    the deck as given; a knob no constructor can set is a knob nobody can
    use."""

    @staticmethod
    def _env():
        return make_pekeris(sound_speed=1800.0, attenuation=0.3,
                            shear_speed=600.0)

    @staticmethod
    def _block_vii(tmp_path, **kw):
        """The ``CMIN CMAX`` record of the deck OAST writes."""
        model = uacpy.OAST(work_dir=tmp_path, cleanup=False, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.run(TestPhaseSpeedWindowReachesTheDeck._env(),
                      uacpy.Source(depths=50.0, frequencies=100.0),
                      uacpy.Receiver(depths=[50.0],
                                     ranges=np.linspace(500.0, 5000.0, 8)))
        deck = (tmp_path / 'oast_run.dat').read_text().splitlines()

        def cmax_of(line):
            tokens = line.split()
            try:
                return float(tokens[1]) if len(tokens) == 2 else None
            except ValueError:
                return None
        # Every window these tests write has CMAX >= 1e7 m/s, which no other
        # two-number record of the deck reaches.
        return [ln.split() for ln in deck
                if (cmax_of(ln) or 0.0) >= 1.0e7][0]

    def test_the_default_window_comes_from_the_slowest_medium(self, tmp_path):
        """The 600 m/s shear speed of the elastic half-space is the slowest
        speed of the problem, and the half-space is elastic: 0.85 x 600 =
        510 m/s (scooter.f90:193-259); the Scholte wave solves
        Brekhovskikh & Lysanov eq. 3.4.29 at 523.2 m/s, and 0.90 x that is
        lower, so it sets CMIN (default_c_low)."""
        cmin, cmax = self._block_vii(tmp_path)
        assert float(cmin) == pytest.approx(0.90 * 523.206, rel=1e-4)
        assert float(cmax) == pytest.approx(1.0e8)

    def test_both_bounds_reach_block_vii(self, tmp_path):
        cmin, cmax = self._block_vii(tmp_path, c_low=400.0, c_high=5.0e7)
        assert float(cmin) == pytest.approx(400.0)
        assert float(cmax) == pytest.approx(5.0e7)

    def test_one_bound_alone_leaves_the_other_derived(self, tmp_path):
        cmin, cmax = self._block_vii(tmp_path, c_low=400.0)
        assert float(cmin) == pytest.approx(400.0)
        assert float(cmax) == pytest.approx(1.0e8)

    @pytest.mark.parametrize('cls_name', ['OAST', 'OASP'])
    def test_the_bounds_round_trip_through_copy(self, cls_name):
        model = getattr(uacpy, cls_name)(c_low=400.0, c_high=5.0e7)
        assert model.copy().c_low == 400.0
        assert model.copy().c_high == 5.0e7

    @pytest.mark.parametrize('cls_name,keys_name',
                             [('OAST', '_OAST_KWARGS'), ('OASP', '_OASP_KWARGS')])
    def test_the_writer_advertises_nothing_unreachable(self, cls_name,
                                                       keys_name):
        import uacpy.io.oases_writer as w
        from uacpy.models._introspect import _collect_init_params
        params = {n for n, _ in _collect_init_params(getattr(uacpy, cls_name))}
        assert {'c_low', 'c_high'} <= params, sorted(getattr(w, keys_name))


# ─── OAST is a single-frequency product, and says which model is not ───────


@pytest.mark.requires_oases
class TestOastRefusesAFrequencySweep:
    """OAST's deck and reader both handle ``NFREQ > 1``, but the result
    cannot: OAST rebuilds its range grid inside the frequency loop, so ``DX``
    scales as ``1/f`` and ``read_oast_tl`` returns ``ranges`` as
    ``(n_freq, n_ranges)`` — while a Field carries one ``'range'``
    coordinate. Sharing an axis would mean interpolating the dB the ``.plt``
    carries (there is no complex pressure on disk), i.e. the null-smearing
    this model warns about, unavoidably rather than by choice."""

    @staticmethod
    def _rig():
        env = uacpy.Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0, density=1.8,
                attenuation=0.3))
        return env, uacpy.Receiver(depths=[50.0],
                                   ranges=np.linspace(500.0, 5000.0, 8))

    def test_a_sweep_is_refused_and_names_oasp(self):
        env, rcv = self._rig()
        with pytest.raises(
                UnsupportedFeatureError,
                match=r'does not support: a \d+-frequency source') as ei:
            uacpy.OAST().run(
                env, uacpy.Source(depths=50.0, frequencies=[100.0, 400.0]),
                rcv)
        # The shared validate_inputs refusal points at RunMode.BROADBAND,
        # which OAST does not implement — following it dead-ends. OAST's own
        # refusal has to name a model that computes the sweep.
        assert 'OASP' in str(ei.value), str(ei.value)

    def test_every_entry_point_raises_the_refusal_naming_oasp(self):
        """The base decides that a sweep is refused on COHERENT_TL and OAST
        phrases it (``_multi_frequency_refusal``), so ``validate_inputs`` and
        ``run_settings`` raise the same UnsupportedFeatureError ``run`` does,
        not the shared one pointing at a BROADBAND mode OAST lacks."""
        env, rcv = self._rig()
        src = uacpy.Source(depths=50.0, frequencies=[100.0, 400.0])
        model = uacpy.OAST()
        for name in ('validate_inputs', 'run_settings'):
            with pytest.raises(UnsupportedFeatureError, match='OASP'):
                getattr(model, name)(env, src, rcv)

    def test_a_single_frequency_returns_a_one_dimensional_range_axis(self):
        env, rcv = self._rig()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            field = uacpy.OAST().run(
                env, uacpy.Source(depths=50.0, frequencies=100.0), rcv)
        assert np.asarray(field.coords['range']).ndim == 1
        assert np.asarray(field.data).ndim == 2

    def test_the_declared_modes_keep_the_refusal_reachable(self):
        # If COHERENT_TL ever leaves the single-frequency modes, or a sweeping
        # mode joins OAST's spec, the 2-D range axis reaches Field() again.
        assert uacpy.OAST.spec.modes == (uacpy.RunMode.COHERENT_TL,)
        assert (uacpy.RunMode.COHERENT_TL
                in uacpy.OAST.spec.traits.single_frequency_modes)


@pytest.mark.requires_oases
class TestOastPlotProductsReachTheResult:
    """``compute_depth_average=True`` adds ``'A'``: OAST writes the
    depth-averaged TL, ``-10·log10`` of ``|p|²`` averaged over the deck's
    receivers (oasfun22.f:209-242), which the result carries as
    ``components['depth_average']``. ``compute_contour=True`` adds ``'C'``,
    whose contour files repeat the TL and which uacpy does not read, so it is
    refused."""

    def test_the_depth_average_is_the_receivers_energy_mean(self):
        env = make_pekeris()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = uacpy.OAST(compute_depth_average=True).run(
                env, Source(depths=50.0, frequencies=100.0),
                Receiver(depths=[25.0, 50.0, 75.0],
                         ranges=np.linspace(200.0, 2000.0, 10)))
        native = result.components['native']
        average = result.components['depth_average']
        np.testing.assert_allclose(average.coords['range'],
                                   native.coords['range'])
        tl = np.asarray(native.data)
        energy_mean = -10.0 * np.log10(np.mean(10.0 ** (-tl / 10.0), axis=0))
        # The .plt carries G13.6 values: 6 significant digits of ~50 dB.
        assert np.max(np.abs(average.data - energy_mean)) < 1e-3
        # The mean of the dB values is a different curve.
        assert np.max(np.abs(average.data - tl.mean(axis=0))) > 1.0

    def test_compute_contour_is_refused(self):
        with pytest.raises(ConfigurationError,
                           match=r"compute_contour=True.*reads no contour"):
            uacpy.OAST(compute_contour=True)


class TestOasnReplicaDepthAxis:
    """``replica_zmin=10.0, replica_zmax=depth-10.0`` is an open-ocean default
    that needs 20 m of water to describe an axis at all. Below that the two
    ends cross; at 5 m the axis ran 10.00 -> -5.00, putting 4 of its 5 points
    above the sea surface. Nothing downstream refuses it — OASN runs the deck
    and returns a Covariance."""

    @staticmethod
    def _axis(tmp_path, depth, **kw):
        from uacpy.io.oases_writer import write_oasn_input
        out = tmp_path / 'oasn_run.dat'
        write_oasn_input(
            out, Environment(bathymetry=depth, ssp=1500.0),
            Source(depths=depth / 2.0, frequencies=100.0),
            uacpy.Receiver(depths=np.linspace(0.2 * depth, 0.6 * depth, 5),
                           ranges=[1000.0]),
            options='R J', **kw)
        for line in out.read_text().splitlines():
            m = re.fullmatch(r'(-?\d+\.\d\d) (-?\d+\.\d\d) (\d+)', line)
            if m:
                return float(m.group(1)), float(m.group(2)), int(m.group(3))
        raise AssertionError('no ZSMIN ZSMAX NSRCZ record in the deck')

    def test_a_five_metre_harbour_searches_only_the_water_column(self, tmp_path):
        with pytest.warns(UserWarning, match='too thin for the default 10 m'):
            zmin, zmax, nz = self._axis(tmp_path, 5.0)
        assert 0.0 < zmin < zmax < 5.0
        assert (zmin, zmax, nz) == (0.5, 4.5, 20)

    def test_a_twenty_metre_column_gets_a_non_degenerate_axis(self, tmp_path):
        with pytest.warns(UserWarning, match='too thin for the default 10 m'):
            zmin, zmax, _ = self._axis(tmp_path, 20.0)
        assert (zmin, zmax) == (2.0, 18.0)

    def test_a_fifteen_metre_column_keeps_its_ordering(self, tmp_path):
        """Between 10 and 20 m the old axis inverted while staying in the
        water, so the damage was ordering rather than domain."""
        with pytest.warns(UserWarning, match='too thin for the default 10 m'):
            zmin, zmax, _ = self._axis(tmp_path, 15.0)
        assert zmin < zmax

    def test_a_deep_enough_column_keeps_the_flat_ten_metre_stand_off(self, tmp_path):
        """Only the depths where a flat 10 m leaves no axis move; 50 m of
        water keeps the 10-40 m grid it already had."""
        assert self._axis(tmp_path, 50.0) == (10.0, 40.0, 20)
        assert self._axis(tmp_path, 100.0) == (10.0, 90.0, 20)

    def test_an_explicit_end_is_written_verbatim(self, tmp_path):
        with pytest.warns(UserWarning, match='too thin for the default 10 m'):
            axis = self._axis(tmp_path, 5.0,
                              replica=OasnReplicaGrid(z=(1.0, None, 3)))
        assert axis == (1.0, 4.5, 3)

    def test_oasn_records_the_scaled_axis_and_its_notice(self):
        """OASN resolves the axis in stage 3: the record carries the
        scaled ends and the notice, which ``run_settings`` says once."""
        with recorded_warnings() as caught:
            settings = uacpy.OASN(verbose=False).run_settings(
                Environment(bathymetry=15.0, ssp=1500.0),
                Source(depths=7.0, frequencies=100.0),
                uacpy.Receiver(depths=[3.0, 7.0, 12.0], ranges=[0.0]),
                RunMode.REPLICA)
        said = [str(w.message) for w in caught if 'too thin' in str(w.message)]
        assert len(said) == 1
        assert said[0] in [n.message for n in settings.engine.notices]
        assert said[0].startswith('OASN: 15 m of water')
        assert (settings.engine.replica_zmin,
                settings.engine.replica_zmax) == (1.5, 13.5)


class TestOasesScatteringChainIsExported:

    def test_scattering_writers_and_rhs_reader_are_in_all(self):
        import uacpy.io
        for name in ('write_oass_input', 'write_oassp_input',
                     'read_oases_rhs_header'):
            assert name in uacpy.io.__all__

    def test_exports_resolve_to_the_oases_module_functions(self):
        import uacpy.io
        from uacpy.io import oases_reader, oases_writer
        assert uacpy.io.write_oass_input is oases_writer.write_oass_input
        assert uacpy.io.write_oassp_input is oases_writer.write_oassp_input
        assert (uacpy.io.read_oases_rhs_header
                is oases_reader.read_oases_rhs_header)


def _pekeris():
    return uacpy.Environment(bathymetry=100.0, ssp=1500.0)


def _flat_receiver():
    return uacpy.Receiver(depths=[50.0], ranges=np.linspace(100.0, 2000.0, 20))


class TestOasesFrequencySweepMustBeUniform:
    """Block III carries ``FREQ1 FREQ2 NFREQ`` and the binaries walk it as
    ``FREQ = FREQ1 + (JJ-1)*DLFREQ`` (``unoast31.f:395``), so a non-uniform
    ``source.frequencies`` runs at a different set of frequencies than the
    one that was asked for, with nothing on disk saying so."""

    def test_a_geometric_ladder_is_refused(self, tmp_path):
        from uacpy.io.oases_writer import write_oast_input
        with pytest.raises(ConfigurationError, match='uniform frequency'):
            write_oast_input(tmp_path / 'oast_run.dat', _pekeris(),
                             uacpy.Source(depths=50.0,
                                          frequencies=[100, 200, 400, 800]),
                             _flat_receiver())

    def test_the_error_names_the_grid_that_would_have_run(self, tmp_path):
        from uacpy.io.oases_writer import write_oast_input
        with pytest.raises(ConfigurationError, match=r'333\.3'):
            write_oast_input(tmp_path / 'oast_run.dat', _pekeris(),
                             uacpy.Source(depths=50.0,
                                          frequencies=[100, 200, 400, 800]),
                             _flat_receiver())

    def test_a_linear_sweep_writes_its_bounds_and_count(self, tmp_path):
        from uacpy.io.oases_writer import write_oast_input
        out = tmp_path / 'oast_run.dat'
        write_oast_input(out, _pekeris(),
                         uacpy.Source(depths=50.0,
                                      frequencies=[100, 200, 300, 400]),
                         _flat_receiver())
        assert out.read_text().splitlines()[2].startswith(
            '100.000000000 400.000000000 4')

    def test_oasn_holds_the_same_rule(self, tmp_path):
        from uacpy.io.oases_writer import write_oasn_input
        with pytest.raises(ConfigurationError, match='uniform frequency'):
            write_oasn_input(tmp_path / 'oasn_run.dat', _pekeris(),
                             uacpy.Source(depths=50.0,
                                          frequencies=[100, 200, 400]),
                             uacpy.Receiver(depths=np.linspace(40, 60, 5),
                                            ranges=[1000.0]),
                             options='R J')


class TestOasesArrayBoundsArePreflighted:
    """Every OASES bound crossed at run time is a character ``STOP``: the
    binary exits 0 and writes no output, so the preflight is the only place
    the caller learns what happened."""

    def test_more_discrete_noise_sources_than_nsmax(self, tmp_path):
        from uacpy.io.oases_writer import write_oasn_input
        sources = [{'depth': 10.0, 'x': 0.0, 'y': 0.0, 'level_dB': 100.0}] * 202
        with pytest.raises(ConfigurationError, match='NSMAX = 201'):
            write_oasn_input(tmp_path / 'oasn_run.dat', _pekeris(),
                             uacpy.Source(depths=50.0, frequencies=100.0),
                             uacpy.Receiver(depths=np.linspace(40, 60, 5),
                                            ranges=[1000.0]),
                             options='N J', noise=OasnNoise(
                                 surface_noise_level_dB=70.0,
                                 discrete_sources=sources))

    def test_exactly_nsmax_discrete_sources_is_accepted(self, tmp_path):
        from uacpy.io.oases_writer import write_oasn_input
        sources = [{'depth': 10.0, 'x': 0.0, 'y': 0.0, 'level_dB': 100.0}] * 201
        out = tmp_path / 'oasn_run.dat'
        write_oasn_input(out, _pekeris(),
                         uacpy.Source(depths=50.0, frequencies=100.0),
                         uacpy.Receiver(depths=np.linspace(40, 60, 5),
                                        ranges=[1000.0]),
                         options='N J', noise=OasnNoise(
                             surface_noise_level_dB=70.0, discrete_sources=sources))
        assert ' 201\n' in out.read_text()

    def test_n_time_samples_above_the_transform_bound(self, tmp_path):
        from uacpy.io.oases_writer import write_oasp_input
        with pytest.raises(ConfigurationError, match='131072'):
            write_oasp_input(tmp_path / 'oasp_run.dat', _pekeris(),
                             uacpy.Source(depths=50.0, frequencies=100.0),
                             _flat_receiver(), n_time_samples=200000)

    def test_a_non_power_of_two_transform_length_warns_with_the_rounding(
            self, tmp_path):
        from uacpy.io.oases_writer import write_oasp_input
        with pytest.warns(UserWarning, match='rounds it up to 4096'):
            write_oasp_input(tmp_path / 'oasp_run.dat', _pekeris(),
                             uacpy.Source(depths=50.0, frequencies=100.0),
                             _flat_receiver(), n_time_samples=3000)

    def test_the_range_axis_starts_at_the_first_receiver_range(
            self, tmp_path):
        """OASES-4: the Receiver is the one place a run's ranges come from.
        Block VIII's R0 is the first receiver range, and a ``range_start=``
        that would shift every range is a keyword the deck does not read."""
        from uacpy.io.oases_writer import write_oasp_input, write_oassp_input
        src = uacpy.Source(depths=50.0, frequencies=100.0)
        rcv = uacpy.Receiver(depths=[50.0], ranges=[700.0, 800.0, 900.0])
        out = tmp_path / 'oasp_run.dat'
        write_oasp_input(out, _pekeris(), src, rcv)
        r0_km, rspace_km, nplots = out.read_text().splitlines()[-1].split()[4:7]
        assert (float(r0_km), float(rspace_km), int(nplots)) == (0.7, 0.1, 3)
        with pytest.raises(TypeError, match="'range_start'"):
            write_oasp_input(out, _pekeris(), src, rcv, range_start=500.0)
        with pytest.raises(TypeError, match="'range_start'"):
            write_oassp_input(out, _pekeris(), src, rcv,
                              interface=4, correlation_length=10.0,
                              n_time_samples=256, freq_min=60.0,
                              freq_max=100.0, time_step=0.005,
                              range_start=500.0)

    def test_a_power_of_two_transform_length_is_silent(self, tmp_path):
        from uacpy.io.oases_writer import write_oasp_input
        with recorded_warnings() as caught:
            write_oasp_input(tmp_path / 'oasp_run.dat', _pekeris(),
                             uacpy.Source(depths=50.0, frequencies=100.0),
                             _flat_receiver(), n_time_samples=4096)
        assert not [w for w in caught
                    if 'n_time_samples' in str(w.message)]


class TestOasrTakesOneFieldParameter:
    """``N``, ``S`` and ``B`` each raise ``NOUT`` (``unoasr21.f:350-372``)
    and OASR STOPs ``'*** ONLY ONE FIELD PARAMETER ALLOWED***'`` above one
    (``:429``). ``'t'`` is not one of them — it flips ``transmit`` and leaves
    ``NOUT`` alone (``:374-376``)."""

    @staticmethod
    def _write(tmp_path, options):
        from uacpy.io.oases_writer import write_oasr_input
        write_oasr_input(tmp_path / 'oasr_run.dat', _pekeris(),
                         uacpy.Source(depths=50.0, frequencies=100.0),
                         _flat_receiver(), options=options)

    @pytest.mark.parametrize('options', ['N S T', 'N B T', 'S B T', 'N S B T'])
    def test_two_field_parameters_are_refused(self, tmp_path, options):
        with pytest.raises(ConfigurationError, match='ONLY ONE FIELD'):
            self._write(tmp_path, options)

    @pytest.mark.parametrize('options', ['N T', 'S T', 'B T', 'N t T'])
    def test_one_field_parameter_is_accepted(self, tmp_path, options):
        self._write(tmp_path, options)
        assert (tmp_path / 'oasr_run.dat').exists()


class TestOasesSourceRecordHoldsOneDepth:
    """INSRC reads a single ``SD`` (``oaseun31.f:1089``, ``:1114``,
    ``:1119``), so a multi-depth Source cannot be written; taking
    ``depths[0]`` answers for a different array than the one supplied."""

    @pytest.mark.parametrize('writer_name',
                             ['write_oast_input', 'write_oasp_input'])
    def test_a_multi_depth_source_is_refused(self, tmp_path, writer_name):
        import uacpy.io.oases_writer as ow
        writer = getattr(ow, writer_name)
        with pytest.raises(ConfigurationError, match='2 depths'):
            writer(tmp_path / 'run.dat', _pekeris(),
                   uacpy.Source(depths=[30.0, 50.0], frequencies=100.0),
                   _flat_receiver())

    def test_a_single_depth_source_writes(self, tmp_path):
        from uacpy.io.oases_writer import write_oast_input
        out = tmp_path / 'oast_run.dat'
        write_oast_input(out, _pekeris(),
                         uacpy.Source(depths=50.0, frequencies=100.0),
                         _flat_receiver())
        assert '50.00 1 0 0 1 0 0' in out.read_text()


class TestOastAndOaspTakeWavenumberBounds:
    """``CMIN``/``CMAX`` truncate the horizontal wavenumber axis at
    ``2*pi*f/cmin`` and ``2*pi*f/cmax`` (``oast.tex:505-515``). The writers
    write a pinned ``c_low``/``c_high`` exactly as given; unpinned, they take
    :func:`~uacpy.core.engine_defaults.default_c_low` (the slowest speed of
    every medium × 0.90 here, the seabed being fluid) and 1E8."""

    @staticmethod
    def _wavenumber_bounds(deck_text):
        """The (CMIN, CMAX) pair as numbers, whatever spelling the writer
        used — OASES reads the record list-directed, so only the values
        are part of the contract."""
        for line in deck_text.splitlines():
            tokens = line.split()
            if len(tokens) != 2:
                continue
            try:
                lo, hi = (float(t) for t in tokens)
            except ValueError:
                continue
            if 0.0 < lo < hi:
                return lo, hi
        raise AssertionError(f'no wavenumber-bound record in:\n{deck_text}')

    def test_oast_honours_an_explicit_pair(self, tmp_path):
        from uacpy.io.oases_writer import write_oast_input
        out = tmp_path / 'oast_run.dat'
        write_oast_input(out, _pekeris(),
                         uacpy.Source(depths=50.0, frequencies=100.0),
                         _flat_receiver(), c_low=200.0, c_high=1.0e6)
        assert self._wavenumber_bounds(out.read_text()) == (200.0, 1.0e6)

    def test_oasp_honours_an_explicit_pair(self, tmp_path):
        from uacpy.io.oases_writer import write_oasp_input
        out = tmp_path / 'oasp_run.dat'
        write_oasp_input(out, _pekeris(),
                         uacpy.Source(depths=50.0, frequencies=100.0),
                         _flat_receiver(), c_low=200.0, c_high=5.0e5)
        assert self._wavenumber_bounds(out.read_text()) == (200.0, 5.0e5)

    def test_a_pinned_bound_survives_the_deck_exactly(self, tmp_path):
        # 1550 used to reach OASES as 1600: the record was written with two
        # significant digits, so a pinned bound moved by up to ~5%.
        from uacpy.io.oases_writer import write_oast_input
        out = tmp_path / 'oast_run.dat'
        write_oast_input(out, _pekeris(),
                         uacpy.Source(depths=50.0, frequencies=100.0),
                         _flat_receiver(), c_low=1450.0, c_high=1550.0)
        assert self._wavenumber_bounds(out.read_text()) == (1450.0, 1550.0)

    @pytest.mark.parametrize('writer_name',
                             ['write_oast_input', 'write_oasp_input'])
    def test_the_default_wavenumber_bounds_are_1350_and_unbounded(self, tmp_path, writer_name):
        import uacpy.io.oases_writer as ow
        out = tmp_path / 'run.dat'
        getattr(ow, writer_name)(
            out, _pekeris(), uacpy.Source(depths=50.0, frequencies=100.0),
            _flat_receiver())
        lo, hi = self._wavenumber_bounds(out.read_text())
        assert lo == 1350.0 and hi >= 1.0e6


class TestOasnReplicaGridResolution:
    """``XSMIN``/``XSMAX`` are km and OASN interpolates the whole replica
    axis from them (``unoasn22.f:210-222``), so the written precision *is*
    the grid's resolution. ``%.3f`` km would quantise every replica onto a
    1 m lattice, coarser than half a wavelength above ~750 Hz."""

    @staticmethod
    def _deck(tmp_path, **kw):
        from uacpy.io.oases_writer import write_oasn_input
        out = tmp_path / 'oasn_run.dat'
        write_oasn_input(out, _pekeris(),
                         uacpy.Source(depths=50.0, frequencies=100.0),
                         uacpy.Receiver(depths=np.linspace(40, 60, 5),
                                        ranges=[1000.0]),
                         options='R J', **kw)
        return out.read_text().splitlines()

    def test_a_sub_metre_replica_step_survives_the_deck(self, tmp_path):
        # Metres in (0.5 mm steps), km on disk.
        lines = self._deck(tmp_path, replica=OasnReplicaGrid(
            x=(500.0005, 2000.0005, 3)))
        assert '0.500000500 2.000000500 3' in lines

    def test_the_depth_axis_stays_in_metres(self, tmp_path):
        """``ZSMIN``/``ZSMAX`` are metres, not km, so %.2f is already a
        centimetre — the km columns are the ones that needed widening."""
        lines = self._deck(tmp_path, replica=OasnReplicaGrid(
            z=(20.0, 80.0, 3)))
        assert '20.00 80.00 3' in lines


def _env(depths=(0.0, 50.0, 100.0), speeds=(1500.0, 1490.0, 1510.0)):
    return uacpy.Environment(
        bathymetry=100.0,
        ssp=SoundSpeedProfile(depths=list(depths), sound_speed=list(speeds)),
        bottom=BoundaryProperties(sound_speed=1700.0, density=1.8,
                                  attenuation=0.5))


def _source():
    return uacpy.Source(depths=50.0, frequencies=100.0)


def _receiver():
    return uacpy.Receiver(depths=[20.0, 60.0], ranges=[1000.0, 2000.0])


class TestWaterLayersNeverCollapseToZeroThickness:
    """An OASES layer record's depth column is ``%.2f`` while
    ``SoundSpeedProfile`` requires adjacent depths to differ by only 1e-6 m,
    so a thermocline sampled at 30.0 and 30.004 m writes two records at
    30.00. INENVI rejects only a *decreasing* depth (oaseun31.f:58-61), so
    the pair passes; PINIT2 then sets ``THICK(I)=V(I+1,1)-V(I,1)`` = 0
    (:1508) and divides by it at ``GRAD=(AKL2-AKU2)/THICK(I)`` (:1537).
    Measured before the guard: ``oast`` on such a deck printed '**** ERROR
    NUMBER : 1', signalled IEEE_DIVIDE_BY_ZERO and left a 0-byte ``.plt``."""

    COLLIDING = ([0.0, 30.0, 30.004, 100.0], [1500.0, 1480.0, 1520.0, 1500.0])

    @pytest.mark.parametrize('writer', ['write_oast_input',
                                        'write_oasp_input',
                                        'write_oasn_input'])
    def test_a_sub_centimetre_thermocline_step_is_refused(self, writer,
                                                          tmp_path):
        """Every OAS* deck emits its water column through the one helper, so
        the guard covers OASSP and OASS (whose extra required arguments keep
        them out of this parametrisation) by construction."""
        import uacpy.io.oases_writer as ow
        with pytest.raises(ConfigurationError, match='30.00 m'):
            getattr(ow, writer)(str(tmp_path / 'a.dat'), _env(*self.COLLIDING),
                                _source(), _receiver())

    def test_the_error_names_both_samples_at_their_own_precision(self,
                                                                 tmp_path):
        from uacpy.io.oases_writer import write_oast_input
        with pytest.raises(ConfigurationError,
                           match='in the OASES layer record') as exc:
            write_oast_input(str(tmp_path / 'c.dat'), _env(*self.COLLIDING),
                             _source(), _receiver())
        # The whole point is telling the caller WHICH pair to separate, so
        # the 4 mm apart depths must not both print as the deck's 30.00.
        assert '30.004' in str(exc.value)
        assert '1480' in str(exc.value) and '1520' in str(exc.value)

    def test_a_pair_oases_folds_to_isovelocity_is_written(self,
                                                         tmp_path):
        """INENVI folds a record whose CC and |CS| agree within 1 cm/s back
        to LAYTYP 1 (oaseun31.f:181-185) before PINIT2 runs, so the layer
        never reaches the division. Measured: that deck runs to '*** OASTL
        FINISHED ***'. The two samples are one sample at the deck's
        resolution, so folding costs nothing and refusing would be noise."""
        from uacpy.io.oases_writer import write_oast_input
        out = tmp_path / 'd.dat'
        write_oast_input(str(out), _env([0.0, 30.0, 30.004, 100.0],
                                        [1500.0, 1480.0, 1480.002, 1500.0]),
                         _source(), _receiver())
        assert out.read_text().count('\n30.00 1480.00') == 2

    def test_the_seafloor_record_may_share_its_depth_with_the_seabed(self,
                                                                    tmp_path):
        """The deepest water record carries CS = 0, which fails the
        ``V(M,3).LT.0`` gate of the gradient branch (oaseun31.f:160) and
        takes the isovelocity one instead. Its depth is shared with the
        first seabed record by construction — that IS the interface — so the
        guard must stop one record short of the seabed."""
        from uacpy.io.oases_writer import write_oast_input
        out = tmp_path / 'e.dat'
        write_oast_input(str(out), _env(), _source(), _receiver())
        lines = out.read_text().splitlines()
        assert '100.00 1510.000000 0.000000 1.000000e-08 0 1.027000 0.0000 0' in lines
        assert any(l.startswith('100.00 1700.000000 ') for l in lines)

    def test_no_half_written_layer_block_survives_the_refusal(self, tmp_path):
        """The depth column is checked before the first record goes out, so
        the deck never carries part of a water column OASES cannot
        integrate."""
        from uacpy.io.oases_writer import write_oast_input
        out = tmp_path / 'f.dat'
        with pytest.raises(ConfigurationError,
                           match='in the OASES layer record'):
            write_oast_input(str(out), _env(*self.COLLIDING),
                             _source(), _receiver())
        assert '30.00 1480.000000' not in out.read_text()


class TestCmaxReachesOasesAtFullPrecision:
    """CMAX sets the lower edge of the wavenumber integration,
    ``WK0 = 2*pi*f/CMAX`` (unoast31.f:461), and under automatic sampling it
    is ``WN1 = 2*pi*FREQ/C2`` in AUTSAM (unoast31.f:1209), which places
    ICW1/ICW2 and WNMIN; CMIN sets the upper edge. Measured: CMAX = 1550 and
    1600 give different ``.plt`` tables from the same deck, so the deck
    writes CMIN and CMAX exactly."""

    def test_oast_writes_c_high_verbatim(self, tmp_path):
        from uacpy.io.oases_writer import write_oast_input
        out = tmp_path / 'a.dat'
        write_oast_input(str(out), _env(), _source(), _receiver(),
                         c_high=1550.0)
        # cmin is 0.9 * min(SSP) per oases_wavenumber_bounds (fluid seabed
        # faster than the water).
        assert '\n1341.0 1550.0\n' in out.read_text()
        assert '1.6e+03' not in out.read_text()

    def test_the_oasp_family_deck_writes_it_verbatim_too(self, tmp_path):
        from uacpy.io.oases_writer import write_oasp_input
        out = tmp_path / 'b.dat'
        write_oasp_input(str(out), _env(), _source(), _receiver(),
                         c_high=1234.5)
        assert '\n1341.0 1234.5\n' in out.read_text()

    def test_the_no_upper_limit_sentinel_is_written_exactly(self,
                                                            tmp_path):
        """1e8 is the value OASES substitutes itself when the deck's
        CMIN/CMAX signs conflict (unoast31.f:226); the list-directed READ at
        :213 takes it as written."""
        from uacpy.io.oases_writer import write_oast_input
        out = tmp_path / 'c.dat'
        write_oast_input(str(out), _env(), _source(), _receiver())
        assert '\n1341.0 100000000.0\n' in out.read_text()

    @pytest.mark.parametrize('frequencies', [[100.0], [80.0, 100.0, 120.0]])
    def test_the_writer_and_the_engine_share_one_centre_frequency(
            self, tmp_path, frequencies):
        """FRC is the midpoint of the band, for a deck written by hand as for
        the engine's default sweep, and FR2 is 2.5 times it."""
        from uacpy.io.oases_writer import write_oasp_input
        source = uacpy.Source(depths=50.0, frequencies=frequencies)
        out = tmp_path / 'fc.dat'
        write_oasp_input(str(out), _env(), source, _receiver())
        centre = 0.5 * (min(frequencies) + max(frequencies))
        lines = out.read_text().splitlines()
        assert any(ln.split()[:1] == [f'{centre:.9f}'] for ln in lines)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            engine = uacpy.OASP(verbose=False).run_settings(
                _env(), source, _receiver(),
                run_mode=uacpy.RunMode.BROADBAND).engine
        assert engine.fc == centre

    def test_the_oasp_deck_carries_oasps_own_no_limit_value(self, tmp_path):
        """OASP's "no upper limit" is 1E8 (unoasp22.f:169, every worked OASP
        deck, oasp.tex:581): the writer and the engine both derive it."""
        from uacpy.io.oases_writer import write_oasp_input
        out = tmp_path / 'e.dat'
        write_oasp_input(str(out), _env(), _source(), _receiver())
        assert '\n1341.0 100000000.0\n' in out.read_text()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            engine = uacpy.OASP(verbose=False).run_settings(
                _env(), _source(), _receiver()).engine
        assert engine.c_high == 1e8

    @pytest.mark.parametrize('writer', ['write_oast_input',
                                        'write_oasp_input'])
    def test_a_fractional_c_low_keeps_every_digit(self, tmp_path, writer):
        from uacpy.io import oases_writer
        out = tmp_path / 'd.dat'
        getattr(oases_writer, writer)(str(out), _env(), _source(),
                                      _receiver(), c_low=1400.37,
                                      c_high=1987.654321)
        assert '\n1400.37 1987.654321\n' in out.read_text()


class TestReceiverDepthsSurviveTheDeckAtFullCount:
    """``Receiver.depths`` need only be 1e-6 m apart while every OASES depth
    column is ``%.2f``. INREC reads the explicit list with one
    ``READ(1,*) (RDC(JJ),jj=1,ir)`` and applies no duplicate test
    (oaseun31.f:1186), so a sub-centimetre array reaches OASES as one
    repeated depth while ``read_oasp_trf`` / ``read_oasn_*`` return the
    distinct depths the caller asked for."""

    def test_an_explicit_depth_list_may_not_repeat_a_token(self, tmp_path):
        from uacpy.io.oases_writer import write_oast_input
        rec = uacpy.Receiver(depths=[50.0, 50.001, 50.002, 50.0035, 50.006],
                             ranges=[1000.0])
        with pytest.raises(ConfigurationError, match='50.00 m'):
            write_oast_input(str(tmp_path / 'a.dat'), _env(), _source(), rec)

    def test_an_equidistant_array_may_not_collapse_its_span(self, tmp_path):
        """The compact form sends only the endpoints, and OASES rebuilds the
        interior from ``RDSTEP=(RDLOW-RD)/(IR-1)`` (oaseun31.f:1167-1168):
        one token for both endpoints makes RDSTEP 0 and puts every receiver
        at one depth."""
        from uacpy.io.oases_writer import write_oast_input
        rec = uacpy.Receiver(depths=[50.0, 50.002, 50.004], ranges=[1000.0])
        with pytest.raises(ConfigurationError, match='RDSTEP'):
            write_oast_input(str(tmp_path / 'b.dat'), _env(), _source(), rec)

    def test_a_fine_interior_under_a_resolvable_span_stays_compact(self,
                                                                   tmp_path):
        """The interior never reaches the file in this form, so an interior
        spacing below 0.01 m is not a collapse — refusing it would reject a
        501-element array over a 1 m aperture, which OASES resolves exactly."""
        from uacpy.io.oases_writer import _receiver_block_lines
        rec = uacpy.Receiver(depths=np.linspace(50.0, 51.0, 501),
                             ranges=[1000.0])
        assert _receiver_block_lines(rec, trailing=' 1') == ['50.00 51.00 501 1']

    def test_the_oasn_element_array_is_guarded_too(self):
        """INPRCV reads its ``Z X Y ITYP GAIN`` records with a single
        list-directed read of 5*NRCV values (oasnun22.f:36) and tests
        nothing either."""
        import io
        from uacpy.io.oases_writer import _emit_receiver_array
        rec = uacpy.Receiver(depths=[50.0, 50.001, 50.002], ranges=[1000.0])
        with pytest.raises(ConfigurationError, match='50.00 m'):
            _emit_receiver_array(io.StringIO(), rec, writer='write_oasn_input')

    def test_a_normally_spaced_array_is_untouched(self):
        import io
        from uacpy.io.oases_writer import _emit_receiver_array
        buf = io.StringIO()
        _emit_receiver_array(buf, uacpy.Receiver(depths=[50.0, 60.0, 70.0],
                                                 ranges=[1000.0]),
                             writer='write_oasn_input')
        assert buf.getvalue().splitlines() == [
            '3', '50.00 0 0 1 0', '60.00 0 0 1 0', '70.00 0 0 1 0']


class TestEveryDiscreteNoiseSourceReachesTheDeck:
    """``oasnun22.f:371,:380`` reads exactly ``NDNS`` records
    (``DO 105 I=1,NDNS / READ(1,*) ZDN(I),XDN(I),YDN(I),DNLEVDB(I)``) and
    Block VI declares ``NDNS``, so writing fewer records than declared leaves
    the reader consuming the following CMIN/CMAX and NW records as the next
    source and dying at ``:380`` with "End of file", exit 2.

    Pinned at N >= 2 deliberately: N = 1 is indistinguishable whether the write
    sits inside the loop or after it, which is why a whole test suite that only
    ever passed one source could not see the difference.
    """

    @staticmethod
    def _deck(tmp_path, n_sources):
        from uacpy.core import (BoundaryProperties, Environment, Receiver,
                                Source)
        from uacpy.io.oases_writer import write_oasn_input
        env = Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.8,
                                      attenuation=0.5))
        # x/y in metres; the deck holds km.
        sources = [{'depth': 80.0, 'x': 1000.0 * (1 + i), 'y': 2000.0,
                    'level_dB': 120.0 + i}
                   for i in range(n_sources)]
        path = tmp_path / f'n{n_sources}.dat'
        write_oasn_input(path, env, Source(depths=50.0, frequencies=100.0),
                         Receiver(depths=[30.0, 60.0], ranges=[1000.0]),
                         noise=OasnNoise(surface_noise_level_dB=70.0,
                                         discrete_sources=sources))
        return [ln for ln in path.read_text().splitlines()
                if ln.startswith('80.00 ')]

    @pytest.mark.parametrize('n_sources', [1, 2, 5])
    def test_the_record_count_matches_the_declared_count(self, n_sources,
                                                         tmp_path):
        assert len(self._deck(tmp_path, n_sources)) == n_sources

    def test_each_source_keeps_its_own_position_and_level(self, tmp_path):
        # Listed farthest first (TestDiscreteSourceOrderDoesNotMatter); the
        # level travels with its position.
        rows = self._deck(tmp_path, 3)
        x = [float(r.split()[1]) for r in rows]
        lv = [float(r.split()[3]) for r in rows]
        assert x == pytest.approx([3.0, 2.0, 1.0])
        assert lv == pytest.approx([122.0, 121.0, 120.0])


class TestDiscreteSourceOrderDoesNotMatter:
    """Under automatic sampling OASN sizes the discrete-source wavenumber
    step from the FIRST record only (``oasnun22.f:426-428``, AUTSMN
    ``:1693``); the wavenumber sum is periodic in range with period
    ``2π/Δk`` (JKPS §4.5.3), so a farther source listed second wrapped back
    into the window. The covariance is a sum over the sources
    (``oasnun22.f:1291``), so the writer lists them farthest first."""

    _NEAR = {'depth': 40.0, 'x': 500.0, 'y': 0.0, 'level_dB': 100.0}
    _FAR = {'depth': 40.0, 'x': 6000.0, 'y': 0.0, 'level_dB': 100.0}

    def test_the_deck_lists_the_farthest_first(self, tmp_path):
        from uacpy.io.oases_writer import write_oasn_input
        sources = [{'depth': 10.0, 'x': 300.0, 'y': 400.0, 'level_dB': 1.0},
                   {'depth': 20.0, 'x': 0.0, 'y': 2000.0, 'level_dB': 2.0},
                   {'depth': 30.0, 'x': 500.0, 'level_dB': 3.0},
                   {'depth': 40.0, 'x': 1000.0, 'y': 0.0, 'level_dB': 4.0}]
        path = tmp_path / 'oasn.dat'
        write_oasn_input(path, make_pekeris(),
                         Source(depths=50.0, frequencies=100.0),
                         Receiver(depths=[30.0, 60.0], ranges=[1000.0]),
                         noise=OasnNoise(discrete_sources=sources))
        rows = [ln.split() for ln in path.read_text().splitlines()
                if ln.split()[:1] in (['10.00'], ['20.00'], ['30.00'],
                                      ['40.00'])
                and len(ln.split()) == 4]
        # Ranges 0.5, 2.0, 0.5, 1.0 km: farthest first, the two at 0.5 km
        # (y included) in the caller's order.
        assert [float(r[3]) for r in rows] == [2.0, 4.0, 1.0, 3.0]

    @pytest.mark.requires_oases
    def test_the_covariance_is_the_sum_of_the_sources_in_either_order(self):
        env = make_pekeris()
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=np.linspace(10.0, 90.0, 9), ranges=[0.0])

        def covariance(sources):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return np.asarray(OASN(discrete_sources=sources).run(
                    env, src, rcv, run_mode=RunMode.COVARIANCE).covariance[0])

        near_first = covariance([self._NEAR, self._FAR])
        far_first = covariance([self._FAR, self._NEAR])
        summed = covariance([self._NEAR]) + covariance([self._FAR])
        assert np.array_equal(near_first, far_first)

        def diagonal_dB(c):
            return 10.0 * np.log10(np.real(np.diag(c)))
        # Measured 0.53 dB; with the nearer source listed first the deck
        # read 12.6-31.7 dB above the farthest-first one.
        assert np.max(np.abs(diagonal_dB(near_first)
                             - diagonal_dB(summed))) < 1.0


class TestTheOasesWaterColumnStartsAtTheSurface:
    """Each OASES layer record carries its own layer's TOP depth
    (``oaseun31.f:54``), and INENVI places the vacuum upper half-space at the
    FIRST water record's depth (``oaseun31.f:56-57``), so every deck's water
    column opens with a record at z = 0 even when the profile's shallowest
    sample lies deeper.

    INENVI reads exactly NL layer records (``oaseun31.f:43, :53-54``) and the
    next READ takes the source record, so the deck's NL must equal the number
    of layer records it writes: one record too many shifts every later record
    and the binary dies on the misread deck (measured: SIGFPE in OAST and
    OASP, "Bad integer" in OASN). Each writer is checked for both counts,
    with the profile starting at the surface, 10 m below it, and with only
    two samples.
    """

    WRITERS = ('write_oast_input', 'write_oasn_input', 'write_oasp_input',
               'write_oass_input', 'write_oassp_input')

    @staticmethod
    def _env(depths, speeds):
        from uacpy.core import BoundaryProperties, Environment
        from uacpy.core.ssp import SoundSpeedProfile
        return Environment(
            bathymetry=100.0,
            ssp=SoundSpeedProfile(depths=depths, sound_speed=speeds),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.8,
                                      attenuation=0.5, roughness=0.5))

    PROFILES = {
        'surface': ([0.0, 55.0, 100.0], [1500.0, 1490.0, 1485.0]),
        'ten_metres_down': ([10.0, 55.0, 100.0], [1500.0, 1490.0, 1485.0]),
        'two_samples': ([10.0, 100.0], [1500.0, 1485.0]),
    }

    def _write(self, writer_name, env, path):
        from uacpy.core import Receiver, Source
        from uacpy.io import oases_writer
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=[30.0, 70.0], ranges=[1000.0, 2000.0])
        writer = getattr(oases_writer, writer_name)
        if writer_name == 'write_oass_input':
            first_bottom, _ = oases_writer.oass_bottom_interfaces(env)
            writer(path, env, src, rcv, 'r', interface=first_bottom,
                   correlation_length=100.0, spectral_exponent=1.9)
        elif writer_name == 'write_oassp_input':
            geom = oases_writer._oasp_layer_geometry(env)
            first_bottom = 1 + geom['n_water_layers'] + 1
            writer(path, env, src, rcv, interface=first_bottom,
                   correlation_length=100.0, spectral_exponent=1.9,
                   n_time_samples=1024, freq_min=50.0, freq_max=150.0,
                   time_step=1e-3)
        else:
            first_bottom = None
            writer(path, env, src, rcv)
        return first_bottom

    @staticmethod
    def _layer_block(path):
        """``(NL, the NL records INENVI reads, the record after them)``,
        located by the vacuum upper half-space that opens the block."""
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        top = next(i for i, ln in enumerate(lines)
                   if ln.split()[:6] == ['0'] * 6)
        n_layers = int(lines[top - 1].split()[0])
        records = [ln.split() for ln in lines[top:top + n_layers]]
        return n_layers, records, lines[top + n_layers].split()

    @pytest.mark.parametrize('writer_name', WRITERS)
    @pytest.mark.parametrize('profile', sorted(PROFILES))
    def test_nl_counts_every_layer_record_written(self, writer_name, profile,
                                                  tmp_path):
        env = self._env(*self.PROFILES[profile])
        path = tmp_path / 'deck.dat'
        first_bottom = self._write(writer_name, env, path)
        n_layers, records, after = self._layer_block(path)
        # Upper half-space, water from z = 0, then the seabed half-space as
        # the NL-th record: 1700 m/s at the 100 m seafloor.
        assert float(records[1][0]) == pytest.approx(0.0)
        assert float(records[-1][0]) == pytest.approx(100.0)
        assert float(records[-1][1]) == pytest.approx(1700.0)
        # The record INENVI's caller reads next is not another layer.
        assert not (len(after) >= 7 and float(after[0]) == pytest.approx(100.0))
        if first_bottom is not None:
            # The roughness override lands on the seabed record: deck layer
            # ``first_bottom`` (1-based) carries the -|RG| scattering form.
            seabed = records[first_bottom - 1]
            assert float(seabed[0]) == pytest.approx(100.0)
            assert float(seabed[6]) == pytest.approx(-0.5)

    @pytest.mark.requires_oases
    @pytest.mark.parametrize('model_name', ['OAST', 'OASP', 'OASN', 'OASS',
                                            'OASSP'])
    def test_each_binary_runs_a_profile_starting_below_the_surface(
            self, model_name):
        import uacpy
        factories = {
            'OAST': lambda: uacpy.OAST(),
            'OASP': lambda: uacpy.OASP(),
            'OASN': lambda: uacpy.OASN(),
            'OASS': lambda: uacpy.OASS(correlation_length=10.0,
                                       rms_roughness=0.5),
            'OASSP': lambda: uacpy.OASSP(correlation_length=5.0),
        }
        env = self._env(*self.PROFILES['ten_metres_down'])
        src = uacpy.Source(depths=50.0, frequencies=100.0)
        rcv = uacpy.Receiver(depths=[30.0, 70.0],
                             ranges=np.linspace(500.0, 2000.0, 4))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = factories[model_name]().run(env, src, rcv)
        # OASN returns a Covariance, the others a field.
        values = (result.covariance if model_name == 'OASN' else result.data)
        assert np.isfinite(np.asarray(values)).any()

    @pytest.mark.requires_oases
    def test_oast_tl_matches_the_surface_anchored_profile(self):
        """The 10 m profile differs from the surface one only in its top
        10 m (a 1500 m/s layer against a 1500->1498.2 m/s gradient), so the
        two TL fields agree to well under a decibel."""
        import uacpy
        src = uacpy.Source(depths=50.0, frequencies=100.0)
        rcv = uacpy.Receiver(depths=[30.0, 70.0],
                             ranges=np.linspace(500.0, 2000.0, 4))
        tl = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for name in ('surface', 'ten_metres_down'):
                tl[name] = np.asarray(uacpy.OAST().run(
                    self._env(*self.PROFILES[name]), src, rcv).tl)
        assert np.median(np.abs(tl['surface'] - tl['ten_metres_down'])) < 1.0


def _oast_pekeris_env():
    """100 m isovelocity guide over a fluid half-space."""
    import uacpy
    return uacpy.Environment(
        bathymetry=100.0, ssp=1500.0,
        bottom=uacpy.BoundaryProperties(
            acoustic_type='half-space', sound_speed=1700.0,
            density=1.8, attenuation=0.5))


def _oast_source():
    import uacpy
    return uacpy.Source(depths=50.0, frequencies=100.0)


def _oast_receiver():
    import uacpy
    return uacpy.Receiver(depths=[10.0], ranges=[1000.0, 5000.0])


class TestOastOasnOaspRefuseUnknownOptionLetters:
    """OASSP, OASR and OASS refuse an option letter their GETOPT never tests;
    OAST, OASN and OASP accepted anything, so a typo cost an option silently.

    What the letter costs differs per binary, and the three messages say so:
    OASP's ladder ends with a bare ``ELSE`` (``unoasp22.f:1051-1052``) and
    OASN's with a bare ``END IF`` (``unoasn22.f:727``), which drop it without
    a word; OAST prints '>>>> UNKNOWN OPTION' to unit 6 (``:1166-1168``),
    which uacpy captures rather than surfaces, so the caller cannot see it
    either.
    """

    @staticmethod
    def _letters(name):
        from uacpy.io import oases_writer
        return getattr(oases_writer, f'_{name}_OPTIONS')

    def test_the_ladders_are_read_in_both_cases(self):
        # The comparisons are written as OPT(I) and opt(i) in the same file;
        # a case-sensitive read drops '4' (dip-slip source, unoast31.f:1117)
        # and every other lowercase-tested letter.
        assert '4' in self._letters('OAST')
        assert 'E' in self._letters('OASP')

    def test_oasn_accepts_digits_as_the_directionality_order(self):
        # unoasn22.f:719-725 reads 1-9 through an ICHAR range test rather than
        # a named letter, so no opt(i).eq.'1' appears in the ladder — reading
        # only the named tests would refuse a legitimate deck.
        assert set('123456789') <= self._letters('OASN')

    @pytest.mark.parametrize('writer,options', [
        ('write_oast_input', 'N J T §'),
        ('write_oasn_input', 'N J §'),
        ('write_oasp_input', 'N J §'),
    ])
    def test_an_unknown_letter_is_refused(self, writer, options, tmp_path):
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.io import oases_writer
        env = _oast_pekeris_env()
        with pytest.raises(ConfigurationError, match='not.*option letters'):
            getattr(oases_writer, writer)(
                tmp_path / 'deck.dat', env,
                _oast_source(), _oast_receiver(), options=options)

    @pytest.mark.parametrize('writer,options,cite', [
        ('write_oast_input', 'N J T E', r'unoast31\.f:299'),
        ('write_oasp_input', 'N J d', r'unoasp22\.f:127'),
    ])
    def test_a_letter_demanding_an_unwritten_block_keeps_its_own_message(
            self, writer, options, cite, tmp_path):
        # Both letters ARE in the GETOPT tables, so the unknown-letter check
        # must not pre-empt the more specific "this writer produces no such
        # input" diagnosis. Ordering, not membership, is what this pins — and
        # the two now carry different types, so a pre-empting unknown-letter
        # ConfigurationError fails this on type before it fails on message.
        from uacpy.core.exceptions import UnsupportedFeatureError
        from uacpy.io import oases_writer
        with pytest.raises(UnsupportedFeatureError, match=cite):
            getattr(oases_writer, writer)(
                tmp_path / 'deck.dat', _oast_pekeris_env(),
                _oast_source(), _oast_receiver(), options=options)


class TestOasnDiscreteBandCount:
    """A pinned discrete-band count is written as given: OASES sets no
    minimum (oasnun22.f:282 runs the three-band split for any
    ``NWS(2) > 1`` and :305-310 one equidistant band below).

    ``NWS(2)`` divides OASN's default complex-contour offset,
    ``OFFDB = 60*V*(SLS(3)-SLS(2))/NWS(2)`` (unoasn22.f:289), the discrete
    band's slowness span over its count, so a count of 1 sets the offset at
    the scale of the whole band; JKPS §4.5.5 warns against an offset chosen
    too large. A count below 10 (uacpy guidance) is therefore noticed.
    """

    def test_the_notice_threshold_is_ten(self):
        from uacpy.io import oases_writer
        assert oases_writer._OASN_FEW_DISCRETE_SAMPLES == 10

    @pytest.mark.parametrize('pinned', [1, 8, 9])
    def test_a_small_count_is_written_as_given_with_a_notice(self, pinned):
        from uacpy.io.oases_writer import (
            _OASN_SURFACE_NOISE_SAMPLES, _noise_nw)
        with recorded_warnings() as rec:
            got = _noise_nw(pinned, _OASN_SURFACE_NOISE_SAMPLES)
        assert got.split() == [str(pinned), str(pinned),
                               str(max(1, pinned // 4))]
        hits = [w for w in rec
                if 'discrete spectral band' in str(w.message)]
        assert len(hits) == 1
        message = str(hits[0].message)
        assert 'written as given' in message
        assert 'unoasn22.f:283-291' in message
        assert 'uacpy guidance' in message
        assert ('one equidistant band' in message) == (pinned == 1)

    @pytest.mark.parametrize('pinned', [10, 40, 400])
    def test_a_count_of_ten_or_more_is_written_silently(self, pinned):
        from uacpy.io.oases_writer import (
            _OASN_SURFACE_NOISE_SAMPLES, _noise_nw)
        with recorded_warnings() as rec:
            got = _noise_nw(pinned, _OASN_SURFACE_NOISE_SAMPLES)
        assert got.split()[:2] == [str(pinned), str(pinned)]
        assert [w for w in rec
                if 'discrete spectral band' in str(w.message)] == []

    @pytest.mark.parametrize('nw', [None, 0, -5])
    def test_an_unpinned_count_writes_the_blocks_own_default(self, nw):
        from uacpy.io.oases_writer import _noise_nw
        assert _noise_nw(nw, (7, 8, 9)) == '7 8 9'


@pytest.mark.requires_oases
class TestOasnDeepNoiseKeepsOasesNormalisation:
    """OASN's deep loop sets the plane-geometry flag only when a surface block
    ran first (``ICDR = 1``, oasnun22.f:509-511), yet divides by
    ``SQRT(FREQ/V)`` as for plane geometry (:705): a deep-only run reads
    ``10·log10(c/f)`` dB above the same sheet computed after a negligible
    surface block. uacpy keeps OASES's level and documents it on
    ``OASN(deep_noise_level_dB=)``."""

    def test_a_deep_only_run_reads_ten_log_c_over_f_high(
            self, tmp_path, monkeypatch):
        from uacpy.io import oases_writer
        from uacpy.models import OASN
        queue = []
        monkeypatch.setattr(oases_writer, '_noise_nw',
                            lambda nw, unpinned: queue.pop(0))

        def diag_db(surface_noise_level_dB, counts, work_dir):
            queue[:] = counts
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cov = OASN(verbose=False, work_dir=work_dir,
                           deep_noise_level_dB=50.0,
                           surface_noise_level_dB=surface_noise_level_dB
                           ).compute_covariance(
                    _oast_pekeris_env(),
                    uacpy.Source(depths=10.0, frequencies=100.0),
                    uacpy.Receiver(depths=[20.0, 50.0, 80.0],
                                   ranges=[1000.0]))
            assert queue == []
            return 10.0 * np.log10(np.abs(np.diag(
                np.asarray(cov.covariance)[0])))

        alone = diag_db(0.0, ['4096 10 16384'], tmp_path / 'alone')
        after = diag_db(0.02, ['400 400 100', '4096 10 16384'],
                        tmp_path / 'after')
        np.testing.assert_allclose(alone - after,
                                   10.0 * np.log10(1500.0 / 100.0), atol=0.15)


class TestOasesDeckTitleFitsTheBinarysBuffer:
    """A long ``env.name`` killed OAST outright.

    Every OASES manual gives the title as ``<= 80 ch.`` (``oast.tex:22`` and
    the same row in oasn/oasp/oasr/oass), and every program reads the record
    as ``20A4``. OAST alone needs three characters more than it reads:
    ``unoast31.f:119`` re-emits the title as
    ``write(ctitle,'(a,a)') atitle(1:ltit),' - '`` into a ``character*80``
    buffer declared at ``:37``, so the internal write overflows once
    ``ltit + 3 > 80``.

    Measured against the binary, and the arithmetic predicts it exactly: a
    77-character title runs, 78 dies with a Fortran "Error termination" and
    exit 2 — surfacing only as a failed run with a backtrace, no diagnosis.
    OASP was checked at 78 and 140 characters and is unaffected, so the
    77-character limit is OAST's alone.
    """

    @staticmethod
    def _env(n):
        import uacpy
        return uacpy.Environment(
            name="X" * n, bathymetry=100.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1700.0,
                density=1.8, attenuation=0.5))

    @staticmethod
    def _title_line(env, tmp_path, writer='write_oast_input'):
        import uacpy
        from uacpy.io import oases_writer
        path = tmp_path / 'deck.dat'
        getattr(oases_writer, writer)(
            path, env, uacpy.Source(depths=50.0, frequencies=100.0),
            uacpy.Receiver(depths=[10.0], ranges=[1000.0, 2000.0]))
        return path.read_text().splitlines()[0]

    def test_the_limits_come_from_the_buffer_arithmetic(self):
        from uacpy.io import oases_writer
        assert oases_writer._OASES_TITLE_CHARS == 80
        # 80 - len(" - ") == 77
        assert oases_writer._OAST_TITLE_CHARS == 77

    def test_a_title_at_the_limit_is_written_whole_and_silently(self, tmp_path):
        with recorded_warnings() as rec:
            line = self._title_line(self._env(77), tmp_path)
        assert len(line) == 77
        assert [w for w in rec if 'deck title' in str(w.message)] == []

    @pytest.mark.parametrize('n', [78, 140])
    def test_a_longer_title_is_truncated_with_a_warning(self, n, tmp_path):
        with recorded_warnings() as rec:
            line = self._title_line(self._env(n), tmp_path)
        assert len(line) == 77
        hits = [w for w in rec if 'deck title' in str(w.message)]
        assert len(hits) == 1
        assert 'unoast31.f:37,119' in str(hits[0].message)

    def test_the_other_writers_keep_the_manuals_eighty(self, tmp_path):
        # OASP has no ' - ' append, so it takes the full 80 the manual allows.
        with recorded_warnings() as rec:
            line = self._title_line(self._env(80), tmp_path,
                                    writer='write_oasp_input')
        assert len(line) == 80
        assert [w for w in rec if 'deck title' in str(w.message)] == []

    def test_a_newline_in_a_name_cannot_split_the_record(self, tmp_path):
        # The title is one record; an embedded newline would shift every
        # block after it.
        line = self._title_line(self._env(5).__class__(
            name="a\nb", bathymetry=100.0, ssp=1500.0,
            bottom=self._env(5).bottom), tmp_path)
        assert line == "a b"


class TestOasesDeckTitleIsTheEnvironmentName:
    """Block I carries ``env.name`` itself. ``Environment`` coerces an
    absent or falsy constructor name to ``'unnamed'`` (``sanitize_title``),
    so every deck a constructor-built environment produces is titled by the
    environment — no writer holds a fallback title of its own."""

    @staticmethod
    def _title_line(tmp_path, **env_kwargs):
        import uacpy
        from uacpy.io.oases_writer import write_oasr_input
        env = uacpy.Environment(bathymetry=100.0, ssp=1500.0, **env_kwargs)
        path = tmp_path / 'deck.dat'
        write_oasr_input(path, env,
                         uacpy.Source(depths=50.0, frequencies=100.0),
                         uacpy.Receiver(depths=[10.0], ranges=[1000.0]))
        return path.read_text().splitlines()[0]

    def test_an_unnamed_environments_deck_is_titled_unnamed(self, tmp_path):
        assert self._title_line(tmp_path) == 'unnamed'

    def test_a_named_environments_deck_carries_the_name(self, tmp_path):
        assert self._title_line(tmp_path, name='wedge trial') == 'wedge trial'


@pytest.mark.requires_binary
class TestOaspBroadbandStampsThePhysicalCMax:
    """OASP writes the stamp on its transfer-function result (``oasp.py``,
    beside ``mask_source_axis``). Its frequency axis is the ``.trf``'s own
    FFT ladder rather than the frequencies asked for, so the stamp is
    asserted on its own here."""

    @pytest.mark.requires_oases
    def test_the_stamp_is_the_seabed_speed(self):
        env = Environment(name='cmax_bb', bathymetry=100.0, ssp=1500.0,
                          bottom=make_halfspace(3000.0, density=2.0,
                                            attenuation=0.1))
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=np.array([50.0]), ranges=np.array([2000.0]))
        result = OASP(verbose=False).run(
            env, src, rcv, run_mode=RunMode.BROADBAND,
            frequencies=np.linspace(80.0, 120.0, 5))

        assert result.run_settings.waveguide.c_max == pytest.approx(3000.0)
        assert 'c_max' not in result.metadata


class TestOaspRecordsItsTimeSampleCount:
    """OASP states its FFT length NX as ``Field.synthesis_floor``: OASP's
    own record of NX samples at DT, ``DLFREQ = 1/(DT*NX)``
    (unoasp22.f:237)."""

    @pytest.mark.requires_oases
    def test_the_field_states_the_runs_fft_length(self):
        env = Environment(name='nt_floor', bathymetry=100.0, ssp=1500.0,
                          bottom=make_halfspace(1700.0, density=1.8,
                                            attenuation=0.5))
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=np.array([50.0]),
                       ranges=np.array([200.0, 400.0]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tf = OASP(n_time_samples=256, verbose=False).run(
                env, src, rcv, run_mode=RunMode.BROADBAND)
        assert tf.synthesis_floor == 256


class TestLogSweptFrequenciesRequireAPositiveLowerBound:
    """OASR option 'C' makes the kernel sweep logarithmic and takes
    ``F1LOG = LOG(FREQ1)`` (``unoasr21.f:123``). A zero bound aborts the
    binary one line earlier (``:116-118``) and a negative one reaches ``LOG``
    unchecked, so the resampler refused neither yet warned that the vector
    "will be evaluated at np.geomspace(0.0, ...)" — a grid it had not built.
    """

    @pytest.mark.parametrize('freq_min', [0.0, -50.0, float('nan')])
    def test_a_non_positive_lower_bound_is_refused(self, freq_min):
        with pytest.raises(ConfigurationError, match='strictly positive'):
            oases_frequency_sweep(
                np.array([freq_min, 100.0, 200.0]), 'OASR', log_spaced=True)

    def test_a_log_spaced_vector_passes(self):
        freqs = np.geomspace(50.0, 400.0, 8)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            freq_min, freq_max, n = oases_frequency_sweep(
                freqs, 'OASR', log_spaced=True)
        assert (freq_min, freq_max, n) == (50.0, 400.0, 8)

    def test_a_linear_vector_is_refused_naming_the_log_grid(self):
        with pytest.raises(ConfigurationError,
                           match='not log-spaced') as exc:
            oases_frequency_sweep(
                np.linspace(50.0, 400.0, 8), 'OASR', log_spaced=True)
        assert 'np.geomspace(50, 400, 8)' in exc.value.remediation

    def test_a_zero_lower_bound_survives_a_linear_sweep(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            freq_min, freq_max, n = oases_frequency_sweep(
                np.linspace(0.0, 400.0, 9), 'OASR', log_spaced=False)
        assert (freq_min, freq_max, n) == (0.0, 400.0, 9)


@pytest.mark.requires_binary
class TestCovarianceIsOfferedByBothOasesProducers:
    """``compute_covariance``'s unsupported-mode message names both
    programs whose spec declares COVARIANCE: OASN and OASS."""

    def test_the_unsupported_message_lists_oass_as_well_as_oasn(self):
        from uacpy.core.exceptions import UnsupportedFeatureError
        with pytest.raises(
                UnsupportedFeatureError,
                match='does not support: covariance-matrix computation'
                ) as excinfo:
            Bellhop(verbose=False).compute_covariance(
                Environment(name='flat', bathymetry=100.0, ssp=1500.0),
                Source(depths=25.0, frequencies=200.0),
                Receiver(depths=np.array([50.0]), ranges=np.array([1000.0])))
        message = str(excinfo.value)
        assert 'OASN' in message and 'OASS' in message

    def test_oass_declares_the_covariance_mode(self):
        from uacpy.models import OASS
        assert RunMode.COVARIANCE in OASS.spec.modes


@pytest.mark.requires_binary
class TestOasrIsTheOnlyLogSweptCaller:
    """The log-swept branch is reachable only through OASR's ``'C'`` option,
    so the refusal cannot fire on a linear-sweep model.
    """

    @pytest.mark.requires_oases
    def test_a_bare_oasr_is_not_log_swept(self):
        from uacpy.models.oases.oasr import _oasr_is_log_swept
        assert _oasr_is_log_swept(OASR(verbose=False).options) is False


class TestOasnContourOffsetNeedsItsLetter:
    """``unoasn22.f:141-145`` reads the ``OFFDBIN`` token only under
    ``ICNTIN > 0``, which only ``'J'``/``'j'`` sets (``:673-675``); without it
    the binary zeroes the offset and consumes a three-token frequency line.
    ``run()`` writes ``self.options or 'J'``, so only a custom string can drop
    it — silently, where every sibling model warns.

    OASN is exempt from the *other* discard: ``:283`` tests ``OFFDBIN``
    itself, so automatic wavenumber sampling (``n_wavenumbers=None``, the
    default) does not zero it here and must not be reported as if it did."""

    @staticmethod
    def _offset_warnings(**kwargs):
        return _offset_notices(OASN(verbose=False,
                                    surface_noise_level_dB=50.0, **kwargs))

    @pytest.mark.requires_oases
    @pytest.mark.parametrize('options', ['N R', 'R', 'N'])
    def test_a_custom_line_without_J_is_reported(self, options):
        messages = self._offset_warnings(options=options,
                                         integration_offset=3.0)
        assert messages, f"options={options!r} dropped the offset in silence"
        assert "'J'" in messages[0]

    @pytest.mark.requires_oases
    @pytest.mark.parametrize('options', [None, 'J', 'N J R'])
    def test_a_line_that_carries_J_is_silent(self, options):
        assert not self._offset_warnings(options=options,
                                         integration_offset=3.0)

    @pytest.mark.requires_oases
    def test_automatic_sampling_alone_is_not_reported(self):
        # n_wavenumbers <= 0 zeroes the offset in OAST / OASP / OASSP, not here.
        assert not self._offset_warnings(options='J', integration_offset=3.0,
                                         n_wavenumbers=None)

    @pytest.mark.requires_oases
    def test_no_offset_is_never_reported(self):
        assert not self._offset_warnings(options='R', integration_offset=0.0)

    def test_integration_offset_is_the_one_contour_offset_knob(self):
        # OFFDBIN has one spelling: integration_offset, as on every other
        # OASES model.
        with pytest.raises(TypeError, match='offdB'):
            OASN(verbose=False, offdB=3.0)

    def test_the_writer_refuses_the_removed_offset_spelling(self, tmp_path):
        from uacpy.io.oases_writer import write_oasn_input
        env = uacpy.Environment(bathymetry=100.0, ssp=1500.0)
        with pytest.raises(TypeError, match='offdB'):
            write_oasn_input(
                tmp_path / 'oasn.dat', env,
                uacpy.Source(depths=50.0, frequencies=100.0),
                uacpy.Receiver(depths=[20.0, 40.0], ranges=[0.0]),
                'N J', offdB=3.0)


@pytest.mark.requires_oases
class TestOasesMasksTheSourceAxis:
    """OASES evaluates its wavenumber integral under the asymptotic Hankel
    carrier, whose ``1/sqrt(r)`` cylindrical spreading is singular at ``r = 0``,
    so the number it returns there belongs to no range: measured on a 100 m
    Pekeris case, OASP ``|p|`` = 1.23 at ``r = 0`` against 5e-3 at 500 m, and
    OASS 56.3 dB. The shared ``mask_source_axis`` NaNs those cells for every
    engine, so the family's grids stay comparable cell by cell.

    A line or scaled source carries no ``1/sqrt(r)`` and is left alone, as it
    is in the sibling models.
    """

    @staticmethod
    def _field(ranges):
        return Field(data=np.ones((2, len(ranges))),
                     coords={'depth': np.array([10.0, 20.0]),
                             'range': np.asarray(ranges, dtype=float)})

    def test_a_point_source_returns_its_axis_column_as_no_data(self):
        field = self._field([0.0, 100.0, 500.0])
        source = Source(depths=10.0, frequencies=100.0)
        with pytest.warns(UserWarning, match=r'OASP: 1 receiver range\(s\) at r = 0'):
            out = mask_source_axis('OASP', field, source)
        assert np.isnan(out.data[:, 0]).all()
        assert np.isfinite(out.data[:, 1:]).all()

    def test_a_grid_clear_of_the_axis_is_untouched_and_silent(self):
        field = self._field([1.0, 100.0])
        source = Source(depths=10.0, frequencies=100.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')       # the warning must not fire
            out = mask_source_axis('OASP', field, source)
        assert np.isfinite(out.data).all()

    def test_a_line_source_keeps_its_axis_column(self):
        field = self._field([0.0, 100.0])
        source = Source(depths=10.0, frequencies=100.0, source_type='line')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            out = mask_source_axis('OASP', field, source)
        assert np.isfinite(out.data).all()


@pytest.mark.requires_oases
class TestOasnPinnedWavenumberCountIntegratesTheWholeWindow:
    """A pinned ``n_wavenumbers`` on OASN is not only a density knob.

    Automatic sampling calls AUTSMN (``oasnun22.f:432``), which returns
    IC1/IC2 bracketing the propagating band ``[2*pi*f/C2, 2*pi*f/C1]`` with
    10% margins (``:1701-1712``). The deck writer emits ``NW 1 NW`` for a
    pinned count and ``:438-440`` takes ICUT1=1, ICUT2=NW verbatim, so the
    pinned deck integrates the whole window, its samples placed
    equidistantly between the wavenumbers of CMAX and CMIN
    (``doc/oast.tex:522-526``). The manual leaves a manual sampling to "the
    usual convergence testing" (``doc/oast.tex:543-546``), and the note says
    that and nothing measured against another engine, at info level, as its
    three siblings say theirs (no warning).
    """

    _GRID = dict(replica_zmin=10.0, replica_zmax=90.0, replica_nz=12,
                 replica_xmin=200.0, replica_xmax=4000.0, replica_nx=18,
                 replica_ny=1)

    def test_a_pinned_count_is_noted_with_the_manuals_convergence_test(
            self, capsys):
        from uacpy.models.oases import OASN
        with recorded_warnings() as rec:
            OASN(**self._GRID, n_wavenumbers=4096, verbose='info')
        hits = [line for line in capsys.readouterr().out.splitlines()
                if 'wavenumber window' in line]
        assert len(hits) == 1 and '[INFO] [OASN]' in hits[0]
        assert 'ICUT2=NW' in hits[0]
        assert 'usual convergence test' in hits[0]
        assert 'Scooter' not in hits[0] and ' dB' not in hits[0]
        assert [w for w in rec if 'wavenumber window' in str(w.message)] == []

    def test_the_default_is_silent(self, capsys):
        from uacpy.models.oases import OASN
        OASN(**self._GRID, verbose='info')
        OASN(**self._GRID, n_wavenumbers=4096)
        assert 'wavenumber window' not in capsys.readouterr().out


@pytest.mark.requires_oases
class TestTheSiblingsNoteAPinnedWavenumberCount:
    """OAST, OASP and OASSP write a pinned ``n_wavenumbers`` as given and say
    at info level what their manual says of it: the samples are placed
    equidistantly over the window (oast.tex:522-526, oasp.tex:642-650,
    oassp.tex:479-488), and automatic sampling is the manual's way to
    "correct answers in the first attempt without the usual convergence
    testing" (oast.tex:543-546, oasp.tex:671-674; OASSP's :511-515 "as for
    the other OASES modules"). No warning: a convergence-tested pin is not
    met with one on every run."""

    #: The knobs each sibling needs besides the count.
    _REQUIRED = {'OAST': {}, 'OASP': {},
                 'OASSP': {'correlation_length': 5.0}}

    @pytest.mark.parametrize('name, cite', [
        ('OAST', 'oast.tex:543-546'), ('OASP', 'oasp.tex:671-674'),
        ('OASSP', 'oassp.tex:511-515')])
    def test_a_pinned_count_is_noted_at_info_level(self, capsys, name, cite):
        with recorded_warnings() as rec:
            model = getattr(uacpy, name)(n_wavenumbers=4096, verbose='info',
                                         **self._REQUIRED[name])
        out = capsys.readouterr().out
        assert f"[INFO] [{name}] {name}(n_wavenumbers=4096): a pinned count" \
            in out
        assert cite in out and 'usual convergence testing' in out
        assert model.n_wavenumbers == 4096
        assert [w for w in rec if 'n_wavenumbers' in str(w.message)] == []

    @pytest.mark.parametrize('name', ['OAST', 'OASP', 'OASSP'])
    def test_automatic_sampling_and_the_default_verbosity_say_nothing(
            self, capsys, name):
        getattr(uacpy, name)(verbose='info', **self._REQUIRED[name])
        getattr(uacpy, name)(n_wavenumbers=4096, **self._REQUIRED[name])
        assert 'a pinned count' not in capsys.readouterr().out


def _oasp(**kwargs):
    """Build an OASP, skipping when the OASES binaries are not installed."""
    try:
        return uacpy.OASP(**kwargs)
    except ExecutableNotFoundError:
        pytest.skip("OASES binaries not installed")


class TestOaspDeclaresItsBandEdgesInDeckOrder:
    """``OASP.__init__`` used to declare ``freq_max, freq_min`` — the reverse
    of the deck it writes, of ``write_oasp_input``, and of its sibling
    ``OASSP``. Block VIII is ``NX FR1 FR2`` (``unoasp22.f:176``) with FR1 the
    LOW index (``:238-239``), so the swapped pair sent ``LX > MX`` into the
    kernel and its frequency loop ran zero times."""

    def test_freq_min_is_declared_before_freq_max(self):
        params = list(inspect.signature(uacpy.OASP.__init__).parameters)
        assert params.index('freq_min') < params.index('freq_max')

    def test_the_declared_order_matches_the_sibling(self):
        def _order(fn):
            names = list(inspect.signature(fn).parameters)
            return names.index('freq_min') < names.index('freq_max')

        assert _order(uacpy.OASP.__init__)
        assert _order(uacpy.OASSP.__init__)

    def test_the_constructor_order_is_the_order_block_viii_is_written_in(
            self, tmp_path):
        """End to end rather than by inspection: the writer's Block VIII line
        is ``NX FR1 FR2 DT R0 RSPACE NPLOTS`` (``unoasp22.f:176``), and FR1 is
        the low index (``:238-239``). The constructor's ``freq_min`` has to
        land in the FR1 slot."""
        from uacpy.io.oases_writer import write_oasp_input

        deck = tmp_path / 'pulse.dat'
        write_oasp_input(
            deck,
            uacpy.Environment(bathymetry=100.0, ssp=1500.0),
            uacpy.Source(depths=50.0, frequencies=100.0),
            uacpy.Receiver(depths=np.array([20.0, 80.0]),
                           ranges=np.array([1000.0, 2000.0])),
            n_time_samples=256, freq_min=10.0, freq_max=200.0,
        )
        block_viii = [
            tok for tok in
            (ln.split() for ln in deck.read_text().splitlines())
            if len(tok) == 7 and tok[0] == '256'
        ]
        assert len(block_viii) == 1, deck.read_text()
        fr1, fr2 = float(block_viii[0][1]), float(block_viii[0][2])
        assert (fr1, fr2) == (10.0, 200.0)

    def test_the_band_edges_cannot_be_bound_positionally(self):
        """A positional ``OASP(exe, n, a, b)`` cannot bind the band edges
        (min-first or max-first) to the wrong physics in silence: every
        constructor knob is keyword-only, so it raises."""
        with pytest.raises(TypeError, match='positional argument but'):
            uacpy.OASP(None, 4096, 100.0, 2000.0)

    @pytest.mark.parametrize('freq_min,freq_max', [(2000.0, 100.0),
                                                   (100.0, 100.0)])
    def test_an_inverted_or_degenerate_band_raises(self, freq_min, freq_max):
        with pytest.raises(ConfigurationError, match='freq_min'):
            _oasp(freq_min=freq_min, freq_max=freq_max)

    def test_the_narrowest_ordered_band_is_accepted(self):
        """The other side of the same threshold: ``freq_min < freq_max`` by
        any margin constructs."""
        model = _oasp(freq_min=100.0, freq_max=100.0 + 1e-9)
        assert model.freq_min == 100.0

    def test_an_underived_freq_max_leaves_the_ordering_untested(self):
        """``freq_max=None`` is resolved at ``run()`` time from the band
        centre, so construction has nothing to compare and must not raise."""
        model = _oasp(freq_min=500.0)
        assert model.freq_max is None
        assert model.freq_min == 500.0

    def test_the_ordered_band_survives_copy(self):
        model = _oasp(freq_min=10.0, freq_max=200.0)
        assert model.copy().freq_min == 10.0
        assert model.copy().freq_max == 200.0
        assert model.copy(freq_max=300.0).freq_max == 300.0


@pytest.mark.requires_oases
class TestOaspRequiresATransferFunctionNotAPlotFile:
    """OASP's output check accepts ``<base>.trf`` and ``<base>.dtrf`` only:
    a ``.plt`` is a plot table, and handing one to ``read_oasp_trf`` fails
    later with "Expected 'PULSETRF'" instead of the missing-output error
    that names the run. A stale ``.plt`` beside no transfer function is the
    case that told them apart."""

    def test_a_stale_plt_beside_no_trf_raises_the_missing_output_error(
            self, tmp_path, monkeypatch):
        import subprocess
        from uacpy.core.exceptions import ModelExecutionError
        env = Environment(
            name='plt-only', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(sound_speed=1600.0, density=1.5,
                                      attenuation=0.5))
        src = Source(depths=25.0, frequencies=150.0)
        rcv = Receiver(depths=[50.0], ranges=[1000.0])
        (tmp_path / 'oasp_run.plt').write_text('stale plot table\n')

        def no_run(self, base_name, work_dir, **kwargs):
            return subprocess.CompletedProcess([base_name], 0, stdout='',
                                               stderr='')
        monkeypatch.setattr(OASP, '_execute', no_run)
        with pytest.raises(ModelExecutionError,
                           match=r"transfer-function") as ei:
            OASP(n_time_samples=512, work_dir=tmp_path, cleanup=False,
                 verbose=False).run(env, src, rcv,
                                    run_mode=RunMode.BROADBAND)
        assert 'PULSETRF' not in str(ei.value)
        assert '.plt' not in str(ei.value)


@pytest.mark.requires_oases
class TestBareStopFatalsOnStdoutRaise:
    """OASES fatals that end in a bare ``STOP`` exit 0 with nothing on stderr
    after the output header is written (measured: an OASP deck with a
    zero-thickness gradient layer printed ``EXECUTION TERMINATED`` and left a
    383-byte header-only ``.trf``). The diagnosis is on stdout only."""

    @staticmethod
    def _completed(stdout):
        import subprocess
        return subprocess.CompletedProcess(['oasp2'], 0, stdout, '')

    def test_execution_terminated_raises_with_the_oases_message(self):
        from uacpy.core.exceptions import ModelExecutionError
        stdout = ('\n **** EXECUTION TERMINATED ***\n\n'
                  ' **** ERROR NUMBER :   1\n')
        with pytest.raises(ModelExecutionError,
                           match='EXECUTION TERMINATED'):
            uacpy.OASP()._raise_on_stdout_fatal(self._completed(stdout))

    def test_progress_traces_pass(self):
        stdout = (' >>> Entering SOLDIS <<<\n >>> Done, CPU=   0.1\n'
                  ' >>>>>WARNING: UNPHYSICAL SPEED RATIO\n')
        uacpy.OASP()._raise_on_stdout_fatal(self._completed(stdout))


class TestOaspRangeAxisFitsIntgr3:
    """INTGR3 stops with ``NP TOO SMALL`` once ``NPLOTS*MBMAXI > NPHALF``
    (16384 ranges for the point source) or ``NOUT*IR*NPLOTS > NP3``
    (458752 depth-range values per output component), after the ``.trf``
    header is written (``oasiun23.f:494-509``). The writer refuses both,
    and passes both boundaries exactly."""

    @staticmethod
    def _write(tmp_path, n_depths, n_ranges):
        from uacpy.core import BoundaryProperties, Environment, Receiver, Source
        from uacpy.io.oases_writer import write_oasp_input
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=BoundaryProperties(
                              acoustic_type='half-space', sound_speed=1700.0,
                              density=1.8, attenuation=0.5))
        rcv = Receiver(depths=np.linspace(5.0, 95.0, n_depths),
                       ranges=np.linspace(100.0, 100.0 + 0.1 * n_ranges, n_ranges))
        write_oasp_input(tmp_path / 'p.dat', env,
                         Source(depths=50.0, frequencies=100.0), rcv)

    def test_the_largest_range_axis_is_written(self, tmp_path):
        self._write(tmp_path, 1, 16384)

    def test_one_range_more_is_refused(self, tmp_path):
        with pytest.raises(ConfigurationError, match='NPHALF'):
            self._write(tmp_path, 1, 16385)

    def test_the_largest_depth_range_product_is_written(self, tmp_path):
        self._write(tmp_path, 28, 16384)

    def test_one_depth_more_is_refused(self, tmp_path):
        with pytest.raises(ConfigurationError, match='NP3'):
            self._write(tmp_path, 29, 16384)


@pytest.mark.requires_oases
@pytest.mark.parametrize('model_name, name', [
    ('OASP', 'x' + 'é' * 79), ('OASN', 'x' + 'é' * 79),
    ('OASR', 'x' + 'é' * 79), ('OAST', 'é' * 79),
])
def test_an_accented_environment_name_runs(model_name, name):
    """Measured before titles were folded to ASCII: OASP/OASN/OASR raised
    ``UnicodeDecodeError`` on the half character their stdout echoed, and
    OAST overflowed its character*80 title buffer (unoast31.f:119)."""
    env = uacpy.Environment(name=name, bathymetry=100.0, ssp=1500.0)
    src = uacpy.Source(depths=30.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=[20.0, 50.0],
                         ranges=np.linspace(500.0, 2000.0, 4))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = getattr(uacpy, model_name)().run(env, src, rcv)
    assert result is not None


@pytest.mark.requires_oases
def test_oasp_source_band_and_frequencies_argument_give_one_axis():
    """A multi-element ``Source.frequencies`` names the BROADBAND band as
    ``run(frequencies=)`` does (measured before: the Source spelling swept
    0.12-249.9 Hz in 2047 bins, the argument spelling 89.97-109.95 Hz in 373
    for the same 90-110 Hz band)."""
    env = uacpy.Environment(bathymetry=100.0, ssp=1500.0)
    rcv = uacpy.Receiver(depths=[50.0], ranges=[1000.0, 2000.0])
    band = np.linspace(90.0, 110.0, 9)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        by_source = uacpy.OASP().run(
            env, uacpy.Source(depths=50.0, frequencies=band), rcv,
            run_mode=RunMode.BROADBAND)
        by_argument = uacpy.OASP().run(
            env, uacpy.Source(depths=50.0, frequencies=100.0), rcv,
            run_mode=RunMode.BROADBAND, frequencies=band)
    f_source = np.asarray(by_source.frequencies, dtype=float)
    f_argument = np.asarray(by_argument.frequencies, dtype=float)
    np.testing.assert_allclose(f_source, f_argument)
    assert f_source.min() > 85.0 and f_source.max() < 115.0


@pytest.mark.requires_oases
class TestOasnSaysHowItWillRunBeforeLaunching:
    """OASN's two notices of how a run will go — the receiver ranges it does
    not read, and a covariance with no noise field — are decided in stage 3,
    so ``run_settings()`` records and says them before anything is written;
    ``validate_inputs()`` says nothing."""

    @staticmethod
    def _call(model, method, **kw):
        env = Environment(bathymetry=100.0, ssp=1500.0)
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=[20.0, 40.0], ranges=[1000.0])
        with recorded_warnings() as caught:
            out = getattr(model, method)(env, src, rcv,
                                         run_mode=RunMode.COVARIANCE, **kw)
        return out, [str(w.message) for w in caught]

    def test_the_preview_records_and_says_both(self):
        settings, said = self._call(OASN(verbose=False), 'run_settings')
        notices = [n.message for n in settings.engine.notices]
        assert any('receiver.ranges is ignored' in n for n in notices)
        assert any('no noise source is configured' in n for n in notices)
        for notice in notices:
            assert said.count(notice) == 1, said

    def test_a_noise_field_leaves_only_the_ranges_notice(self):
        settings, _ = self._call(
            OASN(verbose=False, surface_noise_level_dB=70.0), 'run_settings')
        assert not any('no noise source' in n
                       for n in (x.message for x in settings.engine.notices))
        assert any('receiver.ranges is ignored' in n
                   for n in (x.message for x in settings.engine.notices))

    @pytest.mark.parametrize('ranges, ignored', [
        ([0.0], False), ([1000.0], True), ([0.0, 500.0], True)])
    def test_the_ranges_notice_is_for_any_range_off_the_array(
            self, ranges, ignored):
        """The array sits at x = y = 0: one range at 0 is the array itself
        and is not a notice; one positive range, or two ranges of which the
        first is 0, is."""
        env = Environment(bathymetry=100.0, ssp=1500.0)
        settings = OASN(verbose=False, surface_noise_level_dB=70.0).run_settings(
            env, Source(depths=50.0, frequencies=100.0),
            Receiver(depths=[20.0, 40.0], ranges=ranges),
            run_mode=RunMode.COVARIANCE)
        assert any('receiver.ranges is ignored' in n
                   for n in (x.message for x in settings.engine.notices)) is ignored

    def test_validation_says_neither(self):
        _, said = self._call(OASN(verbose=False), 'validate_inputs')
        assert not [m for m in said if 'receiver.ranges is ignored' in m
                    or 'no noise source' in m], said


class TestOasesAnnouncesTheScholteWaveItsWindowLeavesOut:
    """50 m of water over 20 m of cp 1600 / cs 300 on a 1800 m/s fluid
    half-space at 20 Hz: the water-layer Scholte wave solves Brekhovskikh
    & Lysanov eq. 3.4.29 at 265.8 m/s and the layer-half-space one at
    245.9 m/s; the derived c_low, 0.90 × the slower, takes both in on OAST,
    OASP and OASN alike. OAST's automatic sampling integrates down to
    AUTSAM's CMIN (``unoast31.f:1202-1224``): on a 0-2 km range axis its
    2π/DR term carries it far under the wave whatever c_low is pinned; on
    0-8 km a pinned c_low of 300 m/s reaches 234.4 m/s, under the wave, and
    520 m/s reaches 469.0 m/s, above it. With a pinned n_wavenumbers the
    edge is c_low itself. OAST says so when the geometry reaches within one
    Scholte wavelength (13.29 m) of the seafloor."""

    @staticmethod
    def _env():
        from uacpy.core import SeabedColumn, SedimentLayer
        return Environment(
            bathymetry=50.0, ssp=1500.0,
            bottom=SeabedColumn(
                layers=[SedimentLayer(20.0, sound_speed=1600.0, density=1.8,
                                      attenuation=0.5, shear_speed=300.0,
                                      shear_attenuation=0.5)],
                halfspace=BoundaryProperties(
                    acoustic_type='half-space', sound_speed=1800.0,
                    density=2.0, attenuation=0.5)))

    @staticmethod
    def _settings(model, zs, zr, env=None, ranges=(200.0, 2000.0),
                  frequency=20.0):
        kw = ({'run_mode': RunMode.COVARIANCE} if isinstance(model, OASN)
              else {})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return model.run_settings(
                env or TestOasesAnnouncesTheScholteWaveItsWindowLeavesOut
                ._env(),
                Source(depths=zs, frequencies=frequency),
                Receiver(depths=zr, ranges=list(ranges)), **kw)

    @classmethod
    def _said(cls, model, zs, zr, **kw):
        return [n.message for n in
                cls._settings(model, zs, zr, **kw).engine.notices
                if 'Scholte' in n.message]

    @staticmethod
    def _models(**kw):
        from uacpy.models import OAST
        return [OAST(**kw), OASP(**kw),
                OASN(surface_noise_level_dB=60.0, **kw)]

    @pytest.mark.parametrize('index', [0, 1, 2])
    def test_the_derived_c_low_takes_the_wave_in(self, index):
        model = self._models()[index]
        assert self._said(model, 25.0, [10.0, 49.5]) == []
        if not isinstance(model, OASN):
            assert self._settings(model, 25.0, [49.5]).engine.c_low == \
                pytest.approx(0.90 * 245.89, abs=0.01)

    def test_a_pinned_c_low_past_oasts_sampling_edge_is_told(self):
        from uacpy.models import OAST
        far = (200.0, 8000.0)
        said = self._said(OAST(c_low=520.0), 25.0, [10.0, 49.5], ranges=far)
        assert len(said) == 1, said
        assert said[0].startswith(
            "OAST: the slowest phase speed the run reaches is 469.0 m/s at "
            "20 Hz: OAST's automatic sampling (AUTSAM, "
            "unoast31.f:1202-1224) integrates 1024 wavenumbers up to it on "
            "this deck, c_low = 520.0 m/s over a 0-8000 m range axis, "
            "above the Scholte wave between the water and a "
            "sediment layer, 265.8 m/s"), said[0]
        assert self._said(OAST(c_low=300.0), 25.0, [10.0, 49.5],
                          ranges=far) == []
        pinned = self._said(OAST(c_low=270.0, n_wavenumbers=4096), 25.0,
                            [10.0, 49.5])
        assert len(pinned) == 1 and 'c_low = 270.0 m/s with a pinned ' \
            'n_wavenumbers (unoast31.f:432-433)' in pinned[0], pinned

    @pytest.mark.parametrize('c_low', [300.0, 600.0])
    def test_a_short_range_axis_integrates_past_the_wave(self, c_low):
        """On 0-2 km at 20 Hz AUTSAM's 2π/DR term alone reaches past
        0.94 rad/m, a phase speed of 134 m/s: the wave is in the integral
        for either pinned c_low."""
        from uacpy.models import OAST
        assert self._said(OAST(c_low=c_low), 25.0, [10.0, 49.5]) == []

    def test_a_pinned_c_low_over_limestone_reaches_its_wave(self):
        """30 m of water over the limestone preset at 30 Hz, ranges to
        2 km: OAST(c_low=1380) integrates down to 187.5 m/s on its deck,
        under the 1240.4 m/s Scholte wave, and its TL equals that of
        OAST(c_low=1000) (measured)."""
        from uacpy.models import OAST
        env = Environment(bathymetry=30.0, ssp=1500.0, bottom='limestone')
        assert self._said(OAST(c_low=1380.0), 15.0, [15.0], env=env,
                          ranges=(250.0, 1000.0, 2000.0),
                          frequency=30.0) == []

    @pytest.mark.parametrize(
        'env_name, zs, frequency, c_low, range_min, range_max, '
        'full_hankel, cmin', [
            ('limestone', 15.0, 30.0, 1380.0, 0.0, 2000.0, False, 187.546),
            ('limestone', 15.0, 30.0, 1380.0, 0.0, 20000.0, False, 879.121),
            ('layered', 25.0, 20.0, 600.0, 0.0, 8000.0, False, 468.979),
            ('layered', 25.0, 20.0, 300.0, 0.0, 8000.0, False, 234.432),
            ('layered', 25.0, 20.0, 600.0, 3000.0, 8000.0, False, 234.432),
            ('layered', 25.0, 20.0, 600.0, 0.0, 2000.0, False, 93.7614),
            ('layered', 60.0, 20.0, 600.0, 0.0, 2000.0, False, 100.012),
            ('profile', 45.0, 150.0, 1300.0, 0.0, 12000.0, False, 659.220),
            ('profile', 45.0, 150.0, 1300.0, 0.0, 12000.0, True, 659.200),
            ('profile', 45.0, 150.0, 1300.0, 0.0, 600.0, False, 188.796),
            ('profile', 25.0, 150.0, 1300.0, 0.0, 600.0, False, 190.046),
        ])
    def test_the_sampling_edge_is_the_cmin_oast_prints(
            self, env_name, zs, frequency, c_low, range_min, range_max,
            full_hankel, cmin):
        """The edge equals the CMIN the oast binary printed for the same
        deck (``unoast31.f:471``, six significant digits, measured): the
        range axis, the
        full Hankel transform (option 'f'), and the source's layer speed —
        1600 m/s for a source in the sediment layer, 1510 / 1520 m/s for
        one under the 30 m / 0 m samples of the profile — each move it."""
        from uacpy.io.oases_writer import oases_source_layer_speed
        from uacpy.models._window import oases_integration_edge
        env = {
            'limestone': lambda: Environment(bathymetry=30.0, ssp=1500.0,
                                             bottom='limestone'),
            'layered': self._env,
            'profile': lambda: Environment(
                bathymetry=100.0, ssp=np.array([[0, 1520.0], [30, 1510.0],
                                                [60, 1490.0],
                                                [100, 1495.0]]),
                bottom='limestone'),
        }[env_name]()
        edge, _ = oases_integration_edge(
            c_low, 1e8, automatic=True, frequency=frequency,
            range_min=range_min, range_max=range_max,
            source_speed=oases_source_layer_speed(env, zs),
            full_hankel=full_hankel)
        assert edge == pytest.approx(cmin, abs=1.5e-3)

    def test_far_from_the_bed_oast_says_nothing(self):
        from uacpy.models import OAST
        assert self._said(OAST(c_low=300.0), 25.0, [10.0, 30.0]) == []

    @pytest.mark.parametrize('index', [0, 1, 2])
    def test_an_elastic_half_space_takes_its_scholte_wave_in(self, index):
        """The cs 300 seabed as a half-space: c_low = 0.90 × its 265.8 m/s
        wave, so a receiver 0.5 m above the bed hears no notice and the
        deck's CMIN is 239.2 m/s."""
        env = Environment(
            bathymetry=50.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.8,
                                      attenuation=0.5, shear_speed=300.0,
                                      shear_attenuation=0.5))
        model = self._models()[index]
        settings = self._settings(model, 25.0, [10.0, 49.5], env=env)
        assert [n.message for n in settings.engine.notices
                if 'Scholte' in n.message] == []
        if not isinstance(model, OASN):
            assert settings.engine.c_low == pytest.approx(0.90 * 265.83,
                                                          abs=0.01)
            assert "Scholte wave" in settings.engine.c_low_origin

    # OASP reaches, at its band top FREQM, AUTSAM's CMIN: the window
    # 2π·FREQM/c_low plus its taper (unoasp22.f:1125, :1136), fixed in
    # wavenumber over the band; with a pinned count, c_low itself.

    _OASP_RANGES = tuple(np.linspace(200.0, 2000.0, 10))

    def _oasp_said(self, zr=(10.0, 49.5), **kw):
        return [n.message for n in self._settings(
            OASP(**kw), 25.0, list(zr), ranges=self._OASP_RANGES)
            .engine.notices if 'Scholte' in n.message]

    def test_a_pinned_c_low_past_oasps_edge_is_told(self):
        said = self._oasp_said(c_low=600.0)
        assert len(said) == 1, said
        assert said[0].startswith(
            "OASP: the slowest phase speed the run reaches is 323.2 m/s at "
            "the 20.00 Hz band top: AUTSAM's CMIN for c_low = 600.0 m/s, the "
            "window 2π·FREQM/c_low plus its taper (unoasp22.f:1125, :1136), "
            "above the Scholte wave between the water and a sediment layer, "
            "265.8 m/s"), said[0]
        assert self._oasp_said(c_low=300.0) == []
        assert self._oasp_said(c_low=600.0, zr=(10.0, 30.0)) == []
        pinned = self._oasp_said(c_low=600.0, n_wavenumbers=2048)
        assert len(pinned) == 1 and ("600.0 m/s at the 20.00 Hz band top: "
                                     "c_low = 600.0 m/s with a pinned "
                                     "n_wavenumbers") in pinned[0], pinned

    def test_a_band_says_below_which_frequency_the_wave_is_in(self):
        """20-60 Hz: OASP's window stops at 0.8077 rad/m at every
        frequency (oasp printed CMIN = 2πf/0.8077 on all 12 bins, to
        4e-5 m/s), so the 265.8 m/s wave is inside it below 34.2 Hz."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            settings = OASP(c_low=600.0).run_settings(
                self._env(),
                Source(depths=25.0, frequencies=np.linspace(20.0, 60.0, 9)),
                Receiver(depths=[10.0, 49.5],
                         ranges=list(self._OASP_RANGES)),
                RunMode.BROADBAND)
        said = [n.message for n in settings.engine.notices
                if 'Scholte' in n.message]
        assert len(said) == 1 and (
            "so the field leaves it out at 60.0001 Hz; the window is fixed "
            "in wavenumber at 0.8077 rad/m, so the wave is inside it below "
            "34.2 Hz") in said[0], said

    @pytest.mark.parametrize('env_name, zs, zr, ranges, freqs, kw, cmin', [
        ('layered', 25.0, (10.0, 49.5), (200.0, 2000.0, 10), (20.0,),
         dict(c_low=300.0), 210.034744),
        ('layered', 25.0, (10.0, 49.5), (200.0, 2000.0, 10), (20.0,),
         dict(c_low=600.0), 323.159210),
        ('layered', 25.0, (10.0, 49.5), (200.0, 2000.0, 10), (20.0,),
         dict(c_low=600.0, options='N J F'), 323.449249),
        ('layered', 25.0, (10.0, 49.5), (200.0, 2000.0, 10), (20.0,),
         dict(c_low=600.0, n_wavenumbers=2048), 600.0),
        ('profile', 45.0, (50.0, 99.0), (500.0, 3000.0, 6), (150.0,),
         dict(c_low=1300.0), 1123.12744),
        ('layered', 25.0, (10.0, 49.5), (200.0, 2000.0, 10),
         tuple(np.linspace(20.0, 60.0, 9)), dict(c_low=600.0), 466.724365),
    ])
    def test_oasps_edge_is_the_cmin_oasp_prints_at_its_band_top(
            self, env_name, zs, zr, ranges, freqs, kw, cmin):
        """The edge equals the CMIN the oasp binary printed at its top
        frequency for the same deck (``unoasp22.f:654``, measured): the
        range axis, Filon (option 'F'), a pinned count and the band top
        each move it."""
        from uacpy.io.oases_writer import oases_layer_records
        from uacpy.models.oases._sampling import (_oasp_fft_grid,
                                                  _oasp_wavenumber_top)
        env = (self._env() if env_name == 'layered' else Environment(
            bathymetry=100.0, ssp=np.array([[0, 1520.0], [30, 1510.0],
                                            [60, 1490.0], [100, 1495.0]]),
            bottom='limestone'))
        receiver = Receiver(depths=list(zr), ranges=list(np.linspace(*ranges)))
        source = Source(depths=zs, frequencies=np.asarray(freqs))
        mode = RunMode.BROADBAND if len(freqs) > 1 else None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            engine = OASP(**kw).run_settings(env, source, receiver,
                                             mode).engine
        grid = _oasp_fft_grid(engine.n_time_samples, engine.freq_min,
                              engine.freq_max, engine.time_step)
        top = _oasp_wavenumber_top(
            layer_speeds=oases_layer_records(env), c_low=engine.c_low,
            c_high=engine.c_high, source_depth=zs, receiver_depths=zr,
            receiver_ranges=receiver.ranges, n_time=grid.n_time_samples,
            dt=grid.time_step,
            freq_top=(grid.last_bin - 1) / (grid.time_step
                                            * grid.n_time_samples),
            options=engine.options, n_wavenumbers=engine.n_wavenumbers)
        assert top.edge == pytest.approx(cmin, abs=1.5e-3)

    def test_the_mean_field_oassp_builds_says_no_scholte_wave(self):
        """OASSP's own fitted-c_low rule speaks for the window its mean
        field carries, so its OASP producer adds no Scholte notice."""
        from uacpy.models import OASSP
        env = Environment(
            bathymetry=50.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.8,
                                      attenuation=0.5, shear_speed=300.0,
                                      shear_attenuation=0.5, roughness=0.5))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            OASSP(correlation_length=5.0, c_low=600.0).run_settings(
                env, Source(depths=25.0, frequencies=20.0),
                Receiver(depths=[10.0, 49.5],
                         ranges=list(self._OASP_RANGES)))
        assert not [w for w in caught
                    if 'OASP: the slowest phase speed' in str(w.message)]

    # OASN: the noise blocks reach c_low (slowness bands, oasnun22.f:284);
    # the discrete-source block AUTSMN's CMIN on the farthest source; the
    # replica block c_low under a pinned count.

    _DISCRETE = ({'depth': 49.0, 'x': 3000.0, 'y': 0.0, 'level_dB': 100.0},
                 {'depth': 40.0, 'x': 500.0, 'y': 0.0, 'level_dB': 100.0})

    def _oasn_said(self, mode=RunMode.COVARIANCE, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            notices = OASN(**kw).run_settings(
                self._env(), Source(depths=25.0, frequencies=20.0),
                Receiver(depths=[10.0, 30.0, 49.5], ranges=[0.0]),
                mode).engine.notices
        return [n.message for n in notices if 'Scholte' in n.message]

    def test_a_pinned_c_low_past_oasns_noise_edge_is_told(self):
        said = self._oasn_said(c_low=600.0, surface_noise_level_dB=60.0)
        assert len(said) == 1 and said[0].startswith(
            "OASN: the slowest phase speed the run reaches is c_low = 600.0 "
            "m/s, the slowness edge of the surface-noise integration "
            "(oasnun22.f:284), above the Scholte wave"), said
        assert self._oasn_said(c_low=240.0, surface_noise_level_dB=60.0) == []

    def test_oasns_discrete_sources_reach_autsmns_edge(self):
        """AUTSMN on the farthest source (listed first) at the 20 Hz
        centre: oasn printed CMIN = 375.183 m/s for c_low 600 and 187.546
        for c_low 300 with it 3 km out, and 93.7614 for c_low 600 with it
        89.3 m out, where the RMIN a tenth of its range sets carries the
        count to 4096 (``oasnun22.f:452``, measured)."""
        from uacpy.models._window import oasn_discrete_integration_edge
        for c_low, first, cmin in ((600.0, 3000.0, 375.183),
                                   (300.0, 3000.0, 187.546),
                                   (600.0, 89.3, 93.7614)):
            edge, _ = oasn_discrete_integration_edge(
                c_low, 1e8, automatic=True, frequency=20.0,
                first_range=first)
            assert edge == pytest.approx(cmin, abs=1.5e-3)
        said = self._oasn_said(c_low=600.0, discrete_sources=self._DISCRETE)
        assert len(said) == 1 and "375.2 m/s: OASN's automatic " \
            "discrete-source sampling (AUTSMN" in said[0], said
        assert self._oasn_said(c_low=300.0,
                               discrete_sources=self._DISCRETE) == []

    def test_oasns_pinned_replicas_reach_c_low(self):
        """A pinned count integrates the replicas from c_low
        (``unoasn22.f:243-244``); under automatic sampling AUTSMN reads a
        range minimum the program never sets (``unoasn22.f:233-234``), so
        nothing is said."""
        said = self._oasn_said(RunMode.REPLICA, c_low=600.0,
                               n_wavenumbers=2048)
        assert len(said) == 1 and "the replica integration " \
            "(unoasn22.f:243-244)" in said[0], said
        assert self._oasn_said(RunMode.REPLICA, c_low=600.0) == []


class TestOasesSaysWhatARunDoesNotUse:
    """Ruling 23a: a setting the run does not read gets a notice."""

    @staticmethod
    def _oasn_notices(model, mode):
        env = Environment(bathymetry=100.0, ssp=1500.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return [n.message for n in model.run_settings(
                env, Source(depths=50.0, frequencies=100.0),
                Receiver(depths=[20.0, 40.0], ranges=[0.0]),
                run_mode=mode).engine.notices]

    def test_a_replica_bound_on_a_covariance_run_is_said(self):
        said = self._oasn_notices(
            OASN(surface_noise_level_dB=60.0, replica_xmin=500.0,
                 replica_xmax=900.0), RunMode.COVARIANCE)
        hits = [m for m in said if 'replica grid is read by REPLICA' in m]
        assert len(hits) == 1
        assert 'replica_xmin=500' in hits[0] and 'replica_xmax=900' in hits[0]
        assert not [m for m in self._oasn_notices(
            OASN(surface_noise_level_dB=60.0), RunMode.COVARIANCE)
            if 'replica grid' in m]

    def test_a_noise_field_on_a_replica_run_is_said_and_not_integrated(
            self):
        """A replica run's deck carries 'R' and no 'N': under 'N' OASN
        integrates the noise covariance (CALNSE, unoasn22.f:172-174), which
        a replica run never returns."""
        said = self._oasn_notices(
            OASN(surface_noise_level_dB=60.0, replica_nx=5, replica_nz=5),
            RunMode.REPLICA)
        assert len([m for m in said
                    if 'the noise field given is not used' in m]) == 1
        assert not [m for m in self._oasn_notices(
            OASN(replica_nx=5, replica_nz=5), RunMode.REPLICA)
            if 'noise field' in m]
        env = Environment(bathymetry=100.0, ssp=1500.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            engine = OASN(surface_noise_level_dB=60.0).run_settings(
                env, Source(depths=50.0, frequencies=100.0),
                Receiver(depths=[20.0, 40.0], ranges=[0.0]),
                run_mode=RunMode.REPLICA).engine
        assert engine.options.split() == ['J', 'R']

    def test_the_replica_counts_resolve_on_a_replica_run_only(self):
        """``None`` derives 50 / 1 / 20 on a REPLICA run, recorded as run;
        a COVARIANCE run reads no replica grid, so its record keeps them
        unset and a pinned count is noticed as unused."""
        env = Environment(bathymetry=100.0, ssp=1500.0)
        args = (env, Source(depths=50.0, frequencies=100.0),
                Receiver(depths=[20.0, 40.0], ranges=[0.0]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            replica = OASN().run_settings(
                *args, run_mode=RunMode.REPLICA).engine
            covariance = OASN(surface_noise_level_dB=60.0).run_settings(
                *args, run_mode=RunMode.COVARIANCE).engine
        assert (replica.replica_nx, replica.replica_ny,
                replica.replica_nz) == (50, 1, 20)
        assert (covariance.replica_nx, covariance.replica_ny,
                covariance.replica_nz) == (None, None, None)
        hits = [m for m in self._oasn_notices(
            OASN(surface_noise_level_dB=60.0, replica_nz=7), RunMode.COVARIANCE)
            if 'replica grid is read by REPLICA' in m]
        assert len(hits) == 1 and 'replica_nz=7' in hits[0]

    def test_a_pinned_oasp_fc_beside_a_requested_band_is_said(self):
        env = Environment(bathymetry=100.0, ssp=1500.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            settings = OASP(fc=250.0).run_settings(
                env, Source(depths=50.0, frequencies=100.0),
                Receiver(depths=[50.0], ranges=[1000.0]),
                run_mode=RunMode.BROADBAND,
                frequencies=np.linspace(50.0, 150.0, 11))
        assert settings.engine.fc == 250.0
        assert len([n for n in settings.engine.notices
                    if 'OASP(fc=250) changes nothing computed' in n.message
                    ]) == 1


class TestTheRoughnessNoticeTakesTheMediumAboveTheInterface:
    """The Rayleigh parameter's wavenumber is that of the medium the
    reflected path lengthens in (JKPS §1.7): the water at the seafloor, the
    sediment layer above a sub-bottom interface."""

    @staticmethod
    def _env():
        from uacpy.core.bottom import SeabedColumn
        from uacpy.core.boundary import SedimentLayer
        return Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=SeabedColumn(
                layers=[SedimentLayer(thickness=10.0, sound_speed=1700.0,
                                      density=1.8, attenuation=0.2)],
                halfspace=BoundaryProperties(
                    acoustic_type='half-space', sound_speed=1900.0,
                    density=2.0, attenuation=0.3)))

    @pytest.mark.parametrize('offset, speed', [(0, 1500.0), (1, 1700.0)],
                             ids=['seafloor', 'below-the-layer'])
    def test_the_speed_is_the_layer_above(self, offset, speed):
        from uacpy.io.oases_writer import oass_bottom_interfaces
        from uacpy.models.oases._common import _roughness_notice
        from uacpy.sonar.scattering import _non_perturbative_roughness_notice
        env = self._env()
        first_bottom, _ = oass_bottom_interfaces(env)
        got = _roughness_notice('OASS', env, first_bottom + offset, 1000.0,
                                1.0, oass_index_space=True)
        want = _non_perturbative_roughness_notice('OASS', 1000.0, 1.0,
                                                  sound_speed=speed)
        assert want is not None and got.message == want


@pytest.mark.requires_oases
class TestABoundaryWithoutADeckSpellingIsRefusedBeforeTheRun:
    """OASES spells a sea surface only as vacuum or a fluid/elastic
    half-space and a seabed only as geoacoustic layers over a half-space or
    a vacuum (``oases_writer.refuse_boundaries_without_deck_spelling``); a
    rigid boundary has neither. The refusal is made in the checking stage
    (``OASES._validate_engine``), so ``run_settings()`` refuses exactly what
    ``run()`` refuses. OASR's deck has no sea surface, so it refuses the
    seabed only."""

    RIGID = BoundaryProperties(acoustic_type='rigid')
    SURFACE = r"OASES does not support: a 'rigid' sea surface"
    BOTTOM = r"OASES does not support: a bottom with acoustic_type='rigid'"

    @staticmethod
    def _call(name, env, how, tmp_path):
        kw = {'OASN': dict(surface_noise_level_dB=60.0),
              'OASS': dict(correlation_length=10.0),
              'OASSP': dict(correlation_length=10.0)}.get(name, {})
        model = getattr(uacpy, name)(verbose=False, work_dir=tmp_path, **kw)
        return getattr(model, how)(
            env, Source(depths=50.0, frequencies=100.0),
            Receiver(depths=[30.0, 50.0],
                     ranges=np.linspace(500.0, 2000.0, 4)))

    @pytest.mark.parametrize('how', ['run_settings', 'run'])
    @pytest.mark.parametrize('name',
                             ['OAST', 'OASP', 'OASN', 'OASS', 'OASSP'])
    def test_a_rigid_surface(self, name, how, tmp_path):
        # A rough seabed, which OASS and OASSP need to reach their own checks.
        env = make_pekeris(surface=self.RIGID, roughness=0.5)
        with pytest.raises(UnsupportedFeatureError, match=self.SURFACE):
            self._call(name, env, how, tmp_path)

    @pytest.mark.parametrize('how', ['run_settings', 'run'])
    @pytest.mark.parametrize('name',
                             ['OAST', 'OASP', 'OASN', 'OASR', 'OASS',
                              'OASSP'])
    def test_a_rigid_bottom(self, name, how, tmp_path):
        env = Environment(bathymetry=100.0, ssp=1500.0, bottom=self.RIGID)
        with pytest.raises(UnsupportedFeatureError, match=self.BOTTOM):
            self._call(name, env, how, tmp_path)

    def test_oasr_does_not_read_the_surface(self, tmp_path):
        settings = self._call('OASR', make_pekeris(surface=self.RIGID),
                              'run_settings', tmp_path)
        assert settings.engine is not None
