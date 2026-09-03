"""Tests that apply to every uacpy propagation model, not to one of them.

The shared surface in ``uacpy.models.base``: what ``run()`` accepts and the
order it accepts it in, how the speed bounds and ``c_max`` are derived from
the environment rather than guessed, backend selection as pure
introspection, the broadband frequency guard, and the time-series entry
points.

Per-model behaviour lives in that model's own file (``test_kraken.py``,
``test_ram.py``, ...); what is here is the contract they are all held to,
usually parametrised across every wrapper so a new model cannot quietly
opt out of it.
"""

import types
import warnings

import pytest

import numpy as np
from uacpy.models import (
    Bellhop, RAM, Kraken,
    Bounce, Scooter, SPARC, OAST, OASN, OASP, OASR, OASS, OASSP,
)
from uacpy.models.base import PropagationModel, _smooth_surface
from uacpy.core import Environment, Source
from uacpy.core.bottom import (
    Bottom, BoundaryProperties, SeabedColumn, SedimentLayer,
)
from uacpy.core.surface import Surface
from uacpy.models.base import RunMode
from uacpy.core.receiver import Receiver
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Field, Modes, ReflectionCoefficient


def _halfspace(sound_speed, **kwargs):
    return BoundaryProperties(
        acoustic_type='half-space', sound_speed=sound_speed,
        density=kwargs.pop('density', 1.8),
        attenuation=kwargs.pop('attenuation', 0.3), **kwargs)


ALL_WRAPPERS = [Bellhop, Bounce, Kraken, RAM, Scooter, SPARC,
                OAST, OASN, OASR, OASP, OASSP, OASS]


TIMESERIES_WRAPPERS = [Bellhop, Kraken, RAM, Scooter, OASP, OASSP]


BAD_SAMPLE_RATES = [0.0, -10000.0, float('nan'), float('inf')]

@pytest.mark.requires_binary
class TestBellhop:
    """Tests for Bellhop model. Smoke TL coverage on ``simple_env`` lives
    in ``test_simplified_api.TestComputeAPI`` and ``test_bellhop`` —
    only model-specific scenarios live here."""

    def test_range_dependent_env_returns_full_receiver_grid(self, range_dependent_env, source, receiver_small):
        """Test Bellhop with range-dependent environment."""
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(
            env=range_dependent_env,
            source=source,
            receiver=receiver_small
        )

        assert isinstance(result, Field)
        assert result.shape[0] == len(receiver_small.depths)

    def test_bellhop_cuda_backend_compute_tl(self, simple_env, source, receiver_small):
        """Smoke test for ``Bellhop(backend='cuda')``. Skipped when no
        CUDA/CXX binary is built — the backend gracefully falls back to the
        Fortran binary (with a warning), so detect that and skip.
        """
        from uacpy.models import Bellhop
        bhc = Bellhop(backend='cuda', verbose=False)
        if bhc.version not in ('cuda', 'cxx'):
            pytest.skip("bellhopcuda / bellhopcxx binary not installed")
        result = bhc.compute_tl(env=simple_env, source=source, receiver=receiver_small)
        assert isinstance(result, Field)
        assert result.shape == (len(receiver_small.depths), len(receiver_small.ranges))


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


@pytest.mark.requires_binary
class TestBounce:
    """Tests for Bounce model."""

    def test_bounce_compute_reflection_coefficient(self, simple_env, source, receiver_small, tmp_path):
        """Test Bounce reflection coefficient computation.

        Uses ``work_dir`` (with Bounce's default ``cleanup=False``) so
        the .brc/.irc files survive past the call for the consumer model.
        """
        bounce = Bounce(verbose=False, work_dir=tmp_path)

        # Bounce needs an environment with elastic bottom properties
        from uacpy.core import Environment, BoundaryProperties
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600,
            shear_speed=400,
            density=1.8,
            attenuation=0.2,
            shear_attenuation=0.5
        )
        env_elastic = Environment(
            name="elastic_test",
            bathymetry=simple_env.depth,
            ssp=float(simple_env.ssp.data[0, 0]),
            bottom=bottom
        )

        result = bounce.run(
            env=env_elastic,
            source=source,
            receiver=receiver_small,
        )

        assert isinstance(result, ReflectionCoefficient)
        assert 'brc_file' in result.metadata
        assert result.metadata['brc_file'] is not None

        # Check that .brc file persists in work_dir
        import os
        brc_file = result.metadata['brc_file']
        assert os.path.exists(brc_file), f"BRC file should exist: {brc_file}"

        # Check reflection coefficient data
        assert result.R is not None
        assert result.theta is not None
        assert len(result.R) > 0
        assert len(result.theta) > 0
        # |R| is a passive-boundary amplitude ratio, and the table is a
        # strictly increasing grazing-angle grid on [0, 90].
        R = np.asarray(result.R, dtype=float)
        theta = np.asarray(result.theta, dtype=float)
        assert np.all(R >= 0.0) and np.all(R <= 1.0 + 1e-6)
        assert theta.min() >= 0.0 and theta.max() <= 90.0 + 1e-9
        assert np.all(np.diff(theta) > 0)

    def test_bounce_empty_table_raises(self, simple_env, source, tmp_path):
        """A degenerate RMax (sub-metre receiver range) makes BOUNCE emit a
        reflection table with no angle rows; the wrapper must raise a clear
        ConfigurationError, not silently return an empty ReflectionCoefficient
        (manual-test finding). Caught by the deck-level ``NkTab`` guard before
        the binary runs — the post-run empty-table check is a
        ModelExecutionError, since by then the binary has produced the table.
        """
        from uacpy.core import Environment, BoundaryProperties, Receiver
        bottom = BoundaryProperties(
            acoustic_type='half-space', sound_speed=1600, shear_speed=400,
            density=1.8, attenuation=0.2, shear_attenuation=0.5,
        )
        env_elastic = Environment(
            name="elastic_test", bathymetry=simple_env.depth,
            ssp=float(simple_env.ssp.data[0, 0]), bottom=bottom,
        )
        tiny = Receiver(depths=[50.0], ranges=[1.0])  # RMax = 1 m
        with pytest.raises(ConfigurationError, match="empty reflection-coefficient"):
            Bounce(verbose=False, work_dir=tmp_path).run(
                env=env_elastic, source=source, receiver=tiny)

    def test_bounce_compute_reflection_helper(self, simple_env, source, receiver_small, tmp_path):
        """Verify the convenience method ``Bounce.compute_reflection`` runs."""
        from uacpy.core import Environment, BoundaryProperties
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600, shear_speed=400, density=1.8,
            attenuation=0.2, shear_attenuation=0.5,
        )
        env_elastic = Environment(
            name="elastic_test",
            bathymetry=simple_env.depth,
            ssp=float(simple_env.ssp.data[0, 0]),
            bottom=bottom,
        )
        bounce = Bounce(verbose=False, work_dir=tmp_path)
        result = bounce.compute_reflection(
            env=env_elastic, source=source, receiver=receiver_small,
        )
        assert isinstance(result, ReflectionCoefficient)


@pytest.mark.requires_binary
class TestRAM:
    """Tests for RAM model (mpiramS backend)."""

    def test_ram_returns_finite_tl_grid(self, simple_env, source, receiver_small):
        """Test RAM TL computation."""
        ram = RAM(verbose=False, dr=20.0, dz=2.0)
        result = ram.compute_tl(env=simple_env, source=source, receiver=receiver_small)

        assert isinstance(result, Field)
        assert result.shape[0] > 0  # Has depth dimension
        assert result.shape[1] > 0  # Has range dimension
        assert np.all(np.isfinite(result.data))

    def test_ram_broadband_mode(self, simple_env, source):
        """RAM BROADBAND returns the H(f) transfer function."""
        ram = RAM(Q=2.0, T=2.0, dr=20.0, dz=2.0, verbose=False)
        receiver = Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.array([5000.0])
        )
        result = ram.run(
            simple_env, source, receiver,
            run_mode=RunMode.BROADBAND
        )
        assert isinstance(result, Field)
        assert np.iscomplexobj(result.data)
        # Shape: (n_d, n_r, n_f) — trailing axis is the
        # variable dimension (frequency, here).
        assert result.data.shape[0] > 0  # depth
        assert result.data.shape[1] > 0  # range
        # mpiramS builds its grid as frq = fc + [-nf1..nf1]·df with
        # df = 1/T and nf1 = int((fc/Q - df)/df) + 1 (peramx.f90:353-383):
        # fc=100, Q=2, T=2 → df=0.5, nf1=100, nf=201 spanning 50-150 Hz.
        f = np.asarray(result.coords['frequency'], dtype=float)
        assert result.data.shape[2] == 201
        assert f[0] == pytest.approx(50.0)
        assert f[-1] == pytest.approx(150.0)
        assert np.allclose(np.diff(f), 0.5, atol=1e-6)
        assert f[100] == pytest.approx(100.0)

    def test_ram_time_series_requires_waveform(self, simple_env, source):
        """TIME_SERIES without source_waveform must raise."""
        ram = RAM(Q=2.0, T=2.0, dr=20.0, dz=2.0, verbose=False)
        receiver = Receiver(
            depths=np.array([50.0]),
            ranges=np.array([5000.0])
        )
        with pytest.raises(ConfigurationError, match="source_waveform"):
            ram.run(simple_env, source, receiver,
                    run_mode=RunMode.TIME_SERIES)

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_ram_compute_time_series_helper(self, simple_env, source):
        """Verify the convenience method ``RAM.compute_time_series`` runs.

        The helper takes no ``frequencies=`` so the auto-derive in
        ``_resolve_time_series_frequencies`` fires; the warning is
        expected behaviour, filtered here.
        """
        from uacpy.core.results import Field
        ram = RAM(Q=2.0, T=2.0, dr=20.0, dz=2.0, verbose=False)
        receiver = Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))
        fs = 4000.0
        nt = 64
        t = np.arange(nt) / fs
        sigma = nt / (8.0 * fs)
        f0 = float(np.atleast_1d(source.frequencies)[0])
        wf = (np.sin(2 * np.pi * f0 * (t - t[-1] / 2))
              * np.exp(-((t - t[-1] / 2) ** 2) / (2 * sigma ** 2)))
        result = ram.compute_time_series(
            simple_env, source, receiver,
            source_waveform=wf, sample_rate=fs,
        )
        assert isinstance(result, Field)
        assert result.data.shape[0] == 1
        assert result.data.shape[1] == 1

    def test_compute_time_series_forwards_output_duration(self, simple_env, source):
        """compute_time_series must forward output_duration (+ waveform/rate)
        to run() — it's the knob that sets the synthesised animation window."""
        receiver = Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))
        ram = RAM(verbose=False)
        captured = {}

        def _spy(env, src, rcv, *, run_mode=None, **kw):
            captured.update(run_mode=run_mode, **kw)
            return object()

        ram.run = _spy
        wf = np.ones(8)
        ram.compute_time_series(simple_env, source, receiver,
                                source_waveform=wf, sample_rate=4000.0,
                                output_duration=0.5)
        assert captured['run_mode'] is RunMode.TIME_SERIES
        assert captured['output_duration'] == 0.5
        assert captured['sample_rate'] == 4000.0
        assert captured['source_waveform'] is wf


# OASES instantiation/supported-mode tests live in test_oases_comprehensive.py;
# the cross-model workflow tests below cover Bounce → {Bellhop, Scooter,
# Kraken(backend='krakenc')}.


@pytest.mark.requires_binary
class TestTheSourceAxisIsMaskedByOneMethod:
    """``_mask_source_axis`` NaNs the ``r = 0`` column of a point-source field
    and warns once with one text for every engine; a line or scaled source
    carries no ``1/sqrt(r)`` and keeps its column; a grid clear of the axis is
    returned untouched and silent."""

    @staticmethod
    def _field(ranges):
        return Field(data=np.ones((2, len(ranges))),
                     coords={'depth': np.array([10.0, 20.0]),
                             'range': np.asarray(ranges, dtype=float)})

    @pytest.mark.parametrize("make", [
        lambda: Kraken(verbose=False), lambda: OASP(verbose=False),
        lambda: OASS(verbose=False, correlation_length=10.0),
        lambda: RAM(verbose=False), lambda: Scooter(verbose=False),
    ], ids=['Kraken', 'OASP', 'OASS', 'RAM', 'Scooter'])
    def test_a_point_source_column_is_no_data_with_one_warning(self, make):
        wrapper = make()
        source = Source(depths=10.0, frequencies=100.0)
        with pytest.warns(UserWarning) as record:
            out = wrapper._mask_source_axis(self._field([0.0, 100.0, 500.0]),
                                            source)
        texts = [str(w.message) for w in record if 'r = 0' in str(w.message)]
        assert len(texts) == 1 and texts[0].startswith(
            f"{wrapper.model_name}: 1 receiver range(s) at r = 0, where the "
            "point-source cylindrical-spreading factor 1/sqrt(r) is singular")
        assert np.isnan(out.data[:, 0]).all()
        assert np.isfinite(out.data[:, 1:]).all()

    @pytest.mark.parametrize("source_type", ['line', 'scaled'])
    def test_line_and_scaled_sources_keep_the_column(self, source_type):
        field = self._field([0.0, 100.0])
        source = Source(depths=10.0, frequencies=100.0, source_type=source_type)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            out = Kraken(verbose=False)._mask_source_axis(field, source)
        assert out is field and np.isfinite(out.data).all()

    def test_a_grid_clear_of_the_axis_is_untouched_and_silent(self):
        field = self._field([1.0, 100.0])
        data_before = field.data
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            out = Kraken(verbose=False)._mask_source_axis(
                field, Source(depths=10.0, frequencies=100.0))
        assert out is field and out.data is data_before


class TestTheBroadbandTailIsShared:
    """``_finish_broadband`` returns the BROADBAND transfer function as is and
    synthesises TIME_SERIES with the prepared pulse — the one tail every
    IFFT-based wrapper ends its broadband route with."""

    def test_broadband_returns_the_transfer_function_itself(self):
        tf = object()
        assert Kraken(verbose=False)._finish_broadband(
            tf, RunMode.BROADBAND, None, None) is tf

    def test_time_series_synthesises_with_the_prepared_pulse(self):
        calls = []
        tf = types.SimpleNamespace(
            synthesize_time_series=lambda **kw: calls.append(kw) or 'series')
        out = Kraken(verbose=False)._finish_broadband(
            tf, RunMode.TIME_SERIES, 'pulse', 8000.0)
        assert out == 'series'
        assert calls == [dict(source_waveform='pulse', sample_rate=8000.0)]


class TestBasePlumbing:
    """Shared ``PropagationModel`` behaviour that no single wrapper owns."""

    def test_use_tmpfs_with_pinned_work_dir_warns(self, tmp_path):
        """An ignored user knob is user-facing: it warns, it does not vanish
        into a debug log line the default verbosity never prints."""
        model = Bellhop(verbose=False, work_dir=tmp_path / 'pinned',
                        use_tmpfs=True)
        with pytest.warns(UserWarning, match='use_tmpfs=True'):
            model._setup_file_manager()

    def test_use_tmpfs_without_work_dir_is_silent(self):
        """Dual: the knob is honoured when uacpy owns the directory."""
        import warnings as _warnings
        model = Bellhop(verbose=False, use_tmpfs=True)
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter('always')
            fm = model._setup_file_manager()
        fm.cleanup_work_dir()
        assert not any('use_tmpfs' in str(w.message) for w in caught)

    def test_default_backend_is_the_lowercase_binary_name(self, simple_env,
                                                          source,
                                                          receiver_small):
        """``result.backend`` names the binary that ran, lowercase across the
        package; a model passing no explicit value must not report the
        capitalised class name."""
        from uacpy.core import Environment, BoundaryProperties
        env = Environment(
            name='elastic', bathymetry=simple_env.depth,
            ssp=float(simple_env.ssp.data[0, 0]),
            bottom=BoundaryProperties(
                acoustic_type='half-space', sound_speed=1600, density=1.8,
                attenuation=0.2, shear_speed=400, shear_attenuation=0.5),
        )
        result = Bounce(verbose=False).run(env, source, receiver_small)
        assert result.backend == 'bounce'


class TestModelConsistency:
    """Tests for consistency between different models."""

    # Bellhop ↔ Kraken TL agreement is covered with tighter
    # tolerance in test_cross_model_agreement.py.

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "downstream",
        [
            pytest.param("Bellhop", id="bellhop"),
            pytest.param(
                "Kraken",
                id="krakenc",
                marks=pytest.mark.xfail(
                    reason=(
                        "KRAKENC support for .brc files is experimental in the "
                        "Acoustics Toolbox and currently fails with file format "
                        "errors. Use SCOOTER for production .brc workflows."
                    ),
                    strict=True,
                ),
            ),
            pytest.param("Scooter", id="scooter"),
        ],
    )
    def test_bounce_to_downstream_workflow(
        self, simple_env, source, receiver_small, tmp_path, downstream
    ):
        """BOUNCE → downstream model workflow via .brc reflection coefficients.

        Step 1 computes reflection coefficients on an elastic half-space with
        BOUNCE, persisting the .brc file to ``tmp_path``. Step 2 feeds the
        .brc back into the downstream model (Bellhop / Kraken on its krakenc
        backend / Scooter) and verifies it produces a valid result.
        """
        import os

        from uacpy.core import Environment, BoundaryProperties

        # Step 1 — BOUNCE on elastic bottom
        bottom_elastic = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600,
            shear_speed=400,
            density=1.8,
            attenuation=0.2,
            shear_attenuation=0.5,
        )
        env_elastic = Environment(
            name="elastic_test",
            bathymetry=simple_env.depth,
            ssp=float(simple_env.ssp.data[0, 0]),
            bottom=bottom_elastic,
        )
        bounce = Bounce(verbose=False, work_dir=tmp_path)
        bounce_result = bounce.run(
            env=env_elastic, source=source, receiver=receiver_small,
        )
        assert 'brc_file' in bounce_result.metadata
        brc_file = bounce_result.metadata['brc_file']
        assert os.path.exists(brc_file), "BRC file should exist"

        # Step 2 — feed .brc into the downstream model
        bottom_with_rc = BoundaryProperties(
            acoustic_type='file',
            reflection_file=brc_file,
            sound_speed=1600,
            density=1.8,
        )
        env_with_rc = Environment(
            name="test_with_rc",
            bathymetry=simple_env.depth,
            ssp=float(simple_env.ssp.data[0, 0]),
            bottom=bottom_with_rc,
        )

        c_low_brc = bounce_result.metadata['c_low']
        c_high_brc = bounce_result.metadata['c_high']

        if downstream == "Kraken":
            modes = Kraken(backend='krakenc',
                verbose=False, c_low=c_low_brc, c_high=c_high_brc,
            ).compute_modes(env=env_with_rc, source=source, receiver=receiver_small)
            assert isinstance(modes, Modes)
            assert modes.k is not None and len(modes.k) > 0
            assert modes.phi.shape[1] == len(modes.k)
            assert np.all(np.isfinite(modes.k))
        else:
            model_cls = {"Bellhop": Bellhop, "Scooter": Scooter}[downstream]
            if downstream == "Scooter":
                model = model_cls(
                    verbose=False, c_low=c_low_brc, c_high=c_high_brc,
                )
            else:
                model = model_cls(verbose=False)
            result = model.compute_tl(
                env=env_with_rc, source=source, receiver=receiver_small,
            )
            assert isinstance(result, Field)
            assert result.shape == (
                len(receiver_small.depths), len(receiver_small.ranges)
            )
            assert np.all(np.isfinite(result.data))


class TestUserFrameSkipSpansTheLibrary:
    """``skip_file_prefixes=USER_FRAME_SKIP`` must skip every library frame —
    a warning raised in an io reader a model delegates to still points at the
    user's call — while ``tests`` and ``examples`` stay reportable (their
    files play the caller role the attribution points at)."""

    def test_prefixes_cover_library_subpackages_but_not_tests(self):
        import os
        import uacpy
        from uacpy.models.base import USER_FRAME_SKIP

        pkg = os.path.dirname(os.path.abspath(uacpy.__file__)) + os.sep
        assert USER_FRAME_SKIP
        assert all(p.startswith(pkg) for p in USER_FRAME_SKIP)
        tops = {os.path.relpath(p, pkg).split(os.sep)[0]
                for p in USER_FRAME_SKIP}
        for sub in ('models', 'io', 'core', 'acoustic_signal', 'data'):
            assert sub in tops, f"{sub} missing from USER_FRAME_SKIP"
        assert 'tests' not in tops
        assert 'examples' not in tops


class TestSmoothSurfaceWritesNodesSilently:
    """``_smooth_surface`` zeroes roughness on every node without the
    multi-node broadcast warning the ``Surface`` delegated write emits."""

    def _three_node_surface(self):
        return Surface(
            properties=[
                BoundaryProperties(acoustic_type='vacuum', roughness=r)
                for r in (0.5, 1.0, 1.5)
            ],
            ranges=[0.0, 5000.0, 10000.0],
        )

    def test_every_node_is_zeroed_without_a_warning(self):
        surface = self._three_node_surface()
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            smoothed = _smooth_surface(surface)
        assert [node.roughness for node in smoothed.properties] == [0, 0, 0]

    def test_the_input_surface_keeps_its_roughness(self):
        surface = self._three_node_surface()
        _smooth_surface(surface)
        assert [node.roughness
                for node in surface.properties] == [0.5, 1.0, 1.5]


class TestSpeedBoundsFindThePhysicalExtremes:
    """``PropagationModel._speed_bounds`` spans the water column plus every
    geoacoustic seabed speed — the physical bracket the ``c_max`` stamp is
    taken from."""

    def test_the_halfspace_sets_the_maximum_when_fastest(self):
        env = Environment(name='hs', bathymetry=100.0, ssp=1500.0,
                          bottom=_halfspace(3000.0))
        assert PropagationModel._speed_bounds(env) == (1500.0, 3000.0)

    def test_a_sediment_layer_faster_than_the_halfspace_sets_the_maximum(self):
        bottom = Bottom([SeabedColumn(
            layers=[SedimentLayer(thickness=20.0, sound_speed=1700.0,
                                  density=1.6, attenuation=0.3)],
            halfspace=_halfspace(1600.0))])
        env = Environment(name='layered', bathymetry=100.0, ssp=1500.0,
                          bottom=bottom)
        assert PropagationModel._speed_bounds(env)[1] == 1700.0

    def test_a_rigid_halfspace_contributes_no_speed(self):
        env = Environment(
            name='rigid', bathymetry=100.0,
            ssp=[(0.0, 1500.0), (100.0, 1520.0)],
            bottom=BoundaryProperties(acoustic_type='rigid'))
        assert PropagationModel._speed_bounds(env) == (1500.0, 1520.0)

    def test_it_always_returns_a_two_tuple_of_speeds(self):
        """The premise ``_resolve_c_max`` is written against.

        It used to guard its result with ``if bounds else None``, an arm no
        environment can reach: there is one ``return`` here and it hands back
        a 2-tuple, which is always truthy. The rigid bottom above is the
        emptiest seabed on offer and the water column still fills both
        slots, because an Environment always carries an SSP
        (``ssp=None`` resolves to the isovelocity default).
        """
        for env in (
            Environment(name='bare', bathymetry=100.0),
            Environment(name='rigid', bathymetry=100.0, ssp=1500.0,
                        bottom=BoundaryProperties(acoustic_type='rigid')),
            Environment(name='vacuum', bathymetry=100.0, ssp=1500.0,
                        bottom=BoundaryProperties(acoustic_type='vacuum')),
        ):
            bounds = PropagationModel._speed_bounds(env)
            assert isinstance(bounds, tuple) and len(bounds) == 2, (
                env.name, bounds)
            assert all(np.isfinite(b) and b > 0 for b in bounds), (
                env.name, bounds)
            assert bounds, env.name       # never the falsy arm

    def test_there_is_one_return_and_it_is_unconditional(self):
        """Read from the source, so a second ``return`` added later — one
        that could hand back ``None`` or a bare value — reopens the arm
        ``_resolve_c_max`` no longer checks for."""
        import inspect
        body = inspect.getsource(PropagationModel._speed_bounds)
        assert body.count('return') == 1, body


@pytest.mark.requires_binary
class TestResolveCMaxIsThePhysicalMaximum:
    """``_resolve_c_max`` returns the fastest compressional speed anywhere
    in the environment — the anchor speed ``Field.to_time_trace`` needs,
    never an algorithmic reference."""

    def test_the_seabed_speed_wins_over_the_water_column(self):
        env = Environment(name='cmax', bathymetry=100.0, ssp=1500.0,
                          bottom=_halfspace(3000.0))
        assert Kraken(verbose=False)._resolve_c_max(env) == 3000.0

    def test_the_water_column_wins_under_a_rigid_bottom(self):
        env = Environment(
            name='cmax_rigid', bathymetry=100.0,
            ssp=[(0.0, 1500.0), (100.0, 1520.0)],
            bottom=BoundaryProperties(acoustic_type='rigid'))
        assert Kraken(verbose=False)._resolve_c_max(env) == 1520.0

    def test_it_is_a_speed_on_every_environment_not_sometimes_none(self):
        """``Field.to_time_trace`` anchors its window at ``r / c_max``, so a
        ``None`` here would be a missing anchor, not a benign default. The
        base method cannot produce one — see
        ``TestSpeedBoundsFindThePhysicalExtremes`` for why — and the callers'
        ``is not None`` guards stay only for subclasses."""
        model = Kraken(verbose=False)
        for env in (
            Environment(name='bare', bathymetry=100.0),
            Environment(name='rigid', bathymetry=100.0, ssp=1500.0,
                        bottom=BoundaryProperties(acoustic_type='rigid')),
        ):
            c_max = model._resolve_c_max(env)
            assert isinstance(c_max, float) and c_max > 0, (env.name, c_max)


@pytest.mark.requires_binary
class TestBroadbandNFreqsGuard:
    """``_resolve_broadband_frequencies`` refuses an expansion that cannot
    span the band: ``np.linspace`` with one point returns the lower band
    edge alone and with zero an empty grid."""

    _SRC = Source(depths=50.0, frequencies=100.0)

    def test_n_freqs_one_is_a_configuration_error(self):
        model = Bellhop(verbose=False, n_freqs=1)
        with pytest.raises(ConfigurationError, match=r'n_freqs = 1'):
            model._resolve_broadband_frequencies(
                self._SRC, None,
                n_freqs=model.n_freqs,
                bandwidth_factor=model.bandwidth_factor)

    def test_n_freqs_zero_is_a_configuration_error(self):
        model = Bellhop(verbose=False, n_freqs=0)
        with pytest.raises(ConfigurationError, match=r'n_freqs = 0'):
            model._resolve_broadband_frequencies(
                self._SRC, None,
                n_freqs=model.n_freqs,
                bandwidth_factor=model.bandwidth_factor)

    def test_an_explicit_grid_bypasses_the_guard(self):
        model = Bellhop(verbose=False, n_freqs=0)
        got = model._resolve_broadband_frequencies(
            self._SRC, [90.0, 100.0], n_freqs=model.n_freqs)
        np.testing.assert_allclose(got, [90.0, 100.0])


@pytest.mark.requires_binary
class TestSelectBackendIsPureIntrospection:
    """``Kraken.select_backend`` decides the backend name from the
    environment alone; executable lookup and the ``.irc`` header read live
    on the run path (``_select_kraken_exe``)."""

    @staticmethod
    def _env(bottom):
        return Environment(name='sb', bathymetry=200.0,
                           ssp=[(0.0, 1500.0), (200.0, 1500.0)],
                           bottom=bottom)

    @staticmethod
    def _no_disk(*args, **kwargs):
        raise AssertionError('select_backend touched the executable lookup')

    def test_the_name_decision_reads_no_disk(self, monkeypatch):
        model = Kraken(verbose=False)
        monkeypatch.setattr(model, '_find_executable_in_paths', self._no_disk)
        elastic = self._env(_halfspace(1800.0, shear_speed=400.0,
                                       shear_attenuation=0.5))
        fluid = self._env(_halfspace(1800.0))
        assert model.select_backend(elastic) == 'krakenc'
        assert model.select_backend(fluid) == 'kraken'

    def test_forcing_kraken_on_elastic_media_raises_without_disk(
            self, monkeypatch):
        model = Kraken(verbose=False, backend='kraken')
        monkeypatch.setattr(model, '_find_executable_in_paths', self._no_disk)
        elastic = self._env(_halfspace(1800.0, shear_speed=400.0,
                                       shear_attenuation=0.5))
        with pytest.raises(ConfigurationError, match='elastic media'):
            model.select_backend(elastic)

    def test_a_malformed_irc_bottom_raises_on_the_run_path_only(
            self, tmp_path):
        table = tmp_path / 'bot.irc'
        table.write_text('3\n0.0 1.0 180.0\n45.0 1.0 180.0\n')
        env = self._env(BoundaryProperties(acoustic_type='precalc',
                                           reflection_file=str(table)))
        model = Kraken(verbose=False)
        assert model.select_backend(env) == 'krakenc'
        with pytest.raises(ConfigurationError, match=r'\.irc'):
            model._select_kraken_exe(env)


@pytest.mark.requires_binary
class TestFieldPrtAttach:
    """``_attach_field_prt_path`` records field.exe's hard-coded
    ``field.prt`` under its own metadata key, existence-checked, iff the
    scratch survives — ``_attach_output_paths`` only ever sees the modes
    binary's ``kfield.prt``."""

    @staticmethod
    def _fm(tmp_path):
        return types.SimpleNamespace(get_path=lambda name: tmp_path / name)

    def test_field_prt_is_attached_when_the_scratch_survives(self, tmp_path):
        (tmp_path / 'field.prt').write_text('Field completed successfully\n')
        result = types.SimpleNamespace(metadata={})
        Kraken(verbose=False, cleanup=False)._attach_field_prt_path(
            result, self._fm(tmp_path))
        assert result.metadata['field_prt_file'] == str(tmp_path / 'field.prt')

    def test_a_missing_field_prt_leaves_no_key(self, tmp_path):
        result = types.SimpleNamespace(metadata={})
        Kraken(verbose=False, cleanup=False)._attach_field_prt_path(
            result, self._fm(tmp_path))
        assert 'field_prt_file' not in result.metadata

    def test_cleanup_true_leaves_no_key(self, tmp_path):
        (tmp_path / 'field.prt').write_text('Field completed successfully\n')
        result = types.SimpleNamespace(metadata={})
        Kraken(verbose=False, cleanup=True)._attach_field_prt_path(
            result, self._fm(tmp_path))
        assert 'field_prt_file' not in result.metadata


def _env():
    return Environment(name='triple', bathymetry=100.0, ssp=1500.0)


def _source():
    return Source(depths=25.0, frequencies=200.0)


def _receiver():
    return Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))


def _model(cls):
    # OASSP/OASS refuse construction without the roughness spectrum's
    # correlation length; every other wrapper constructs bare.
    extra = ({'correlation_length': 5.0} if cls in (OASSP, OASS) else {})
    return cls(verbose=False, **extra)


@pytest.mark.requires_binary
class TestRunRejectsSwappedCarriers:
    """Every wrapper's ``run()`` opens with
    ``PropagationModel._require_run_triple``: the (env, source, receiver)
    argument order is checked before run-mode resolution or deck assembly,
    so a swapped pair raises one typed error instead of a raw
    ``AttributeError`` from deep inside the writer."""

    @pytest.mark.parametrize('cls', ALL_WRAPPERS)
    def test_source_passed_in_the_env_slot_raises_naming_the_order(self, cls):
        with pytest.raises(ConfigurationError, match='in that order'):
            _model(cls).run(_source(), _env(), _receiver())

    @pytest.mark.parametrize('cls', ALL_WRAPPERS)
    def test_receiver_and_source_swapped_raises_naming_the_order(self, cls):
        with pytest.raises(ConfigurationError, match='in that order'):
            _model(cls).run(_env(), _receiver(), _source())

    def test_the_error_names_each_wrong_slot_with_the_received_type(self):
        with pytest.raises(ConfigurationError,
                           match='env=Source, source=Environment'):
            _model(Bellhop).run(_source(), _env(), _receiver())

    def test_a_correct_triple_passes_the_validator(self):
        _model(Bellhop)._require_run_triple(_env(), _source(), _receiver())

    def test_carrier_subclasses_pass_the_validator(self):
        class _TaggedSource(Source):
            pass

        tagged = _TaggedSource(depths=25.0, frequencies=200.0)
        _model(Bellhop)._require_run_triple(_env(), tagged, _receiver())


def _waveform(n=256, rate=1000.0, freq=100.0):
    return np.sin(2.0 * np.pi * freq * np.arange(n) / rate)


@pytest.mark.requires_binary
class TestTimeSeriesRequiresAPositiveFiniteSampleRate:
    """``_require_timeseries_signal`` is the one gate every IFFT-based
    TIME_SERIES wrapper passes its ``sample_rate`` through, so the check
    belongs there rather than in six wrappers.

    The test is NaN-closed on purpose: written as ``sample_rate <= 0`` the
    guard admits nan, which compares False against both bounds. Measured
    before the fix on Bellhop, an accepted rate produced a ZeroDivisionError
    at 0 Hz, a raw ValueError at -10 kHz with long delays, and — with short
    delays — a 419-sample trace on a descending time axis, i.e. a
    silently wrong answer at exit 0.
    """

    @pytest.mark.parametrize('cls', TIMESERIES_WRAPPERS,
                             ids=[c.__name__ for c in TIMESERIES_WRAPPERS])
    @pytest.mark.parametrize('rate', BAD_SAMPLE_RATES)
    def test_every_timeseries_wrapper_names_the_bad_rate(self, cls, rate):
        with pytest.raises(ConfigurationError, match='sample_rate'):
            _model(cls)._require_timeseries_signal(
                RunMode.TIME_SERIES, _waveform(), rate)

    def test_a_positive_finite_rate_is_accepted(self):
        _model(Bellhop)._require_timeseries_signal(
            RunMode.TIME_SERIES, _waveform(), 1000.0)

    def test_a_non_numeric_rate_raises_the_typed_error(self):
        with pytest.raises(ConfigurationError, match='sample_rate'):
            _model(Bellhop)._require_timeseries_signal(
                RunMode.TIME_SERIES, _waveform(), 'fast')

    @pytest.mark.parametrize('rate', BAD_SAMPLE_RATES)
    def test_bellhop_run_refuses_the_rate_before_it_traces_rays(self, rate):
        # ``run()`` reaches the guard before _run_broadband, so no deck is
        # written and no binary is spawned.
        with pytest.raises(ConfigurationError, match='sample_rate'):
            _model(Bellhop).run(
                Environment(name='flat', bathymetry=100.0, ssp=1500.0),
                Source(depths=25.0, frequencies=200.0),
                Receiver(depths=np.array([50.0]), ranges=np.array([1000.0])),
                run_mode=RunMode.TIME_SERIES,
                source_waveform=_waveform(), sample_rate=rate)


class TestSynthesizeTimeSeriesRequiresAPositiveFiniteSampleRate:
    """The deep guard in ``core/results/field.py`` backs the wrapper-level one
    for callers that reach ``Field.synthesize_time_series`` directly. It was
    NaN-open: nan slipped past ``sample_rate <= 0`` into the ``int(nfft)``
    sizing and surfaced as ``ValueError: cannot convert float NaN to
    integer``, and inf as an OverflowError.
    """

    @staticmethod
    def _broadband_field():
        freqs = np.linspace(80.0, 120.0, 21)
        return Field(
            data=np.ones((2, 3, freqs.size), dtype=complex),
            coords={'depth': np.array([10.0, 20.0]),
                    'range': np.array([100.0, 200.0, 300.0]),
                    'frequency': freqs},
            metadata={'kind': 'pressure', 'unit': 'Pa'})

    @pytest.mark.parametrize('rate', BAD_SAMPLE_RATES)
    def test_the_typed_error_names_the_rate(self, rate):
        with pytest.raises(ConfigurationError, match='sample_rate'):
            self._broadband_field().synthesize_time_series(_waveform(), rate)

    def test_a_positive_finite_rate_yields_an_ascending_time_axis(self):
        out = self._broadband_field().synthesize_time_series(
            _waveform(), 1000.0)
        t = np.asarray(out.coords['time'])
        assert t.size > 1
        assert np.all(np.diff(t) > 0)


@pytest.mark.requires_binary
class TestComputeModesRequiresAWholeModeCount:
    """``compute_modes`` applies the cap as ``int(n_modes)`` — the copy
    ``Kraken._compute_modes_impl`` runs — which truncates toward zero, so a
    fractional request would run a different cap than the caller asked for.
    ``True`` is refused for the same reason ``Bellhop(n_beams=True)`` is: bool
    is an int subclass and would silently mean 1.
    """

    @staticmethod
    def _compute(n_modes):
        Kraken(verbose=False).compute_modes(
            Environment(name='flat', bathymetry=100.0, ssp=1500.0),
            Source(depths=25.0, frequencies=200.0),
            n_modes=n_modes)

    @pytest.mark.parametrize('n_modes', [50.5, -0.5, float('nan'),
                                         float('inf'), np.float64(3.25)])
    def test_a_fractional_or_non_finite_cap_is_refused(self, n_modes):
        with pytest.raises(ConfigurationError, match='whole number'):
            self._compute(n_modes)

    def test_a_bool_cap_is_refused(self):
        with pytest.raises(ConfigurationError, match='must be an int'):
            self._compute(True)

    def test_a_receiver_in_the_third_slot_is_named(self):
        with pytest.raises(ConfigurationError, match='takes no receiver'):
            self._compute(Receiver(depths=np.array([50.0]),
                                   ranges=np.array([1000.0])))


@pytest.mark.requires_binary
class TestIrregularReceiverGridIsCheckedPairwise:
    """``grid_type='I'`` writes RunType(5:5)='I', where BELLHOP walks the
    depth and range arrays together — receiver *i* is (depths[i], ranges[i]).
    Scoring the below-seafloor check on the Cartesian product therefore
    reports pairs the deck never evaluates: on the 50 m → 200 m slope below,
    both real receivers clear their local seafloor while the cross term
    (100 m at r = 1000 m, floor 50 m) does not.
    """

    @staticmethod
    def _env():
        return Environment(
            name='slope', ssp=1500.0,
            bathymetry=[(0.0, 50.0), (1000.0, 50.0), (5000.0, 200.0)])

    @staticmethod
    def _receiver():
        return Receiver(depths=np.array([20.0, 100.0]),
                        ranges=np.array([1000.0, 5000.0]))

    def test_a_paired_grid_clear_of_its_own_seafloor_is_silent(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            Bellhop(grid_type='I', verbose=False)._check_per_range_receiver_depth(
                self._env(), self._receiver())

    def test_a_rectilinear_grid_reports_the_cross_term(self):
        with pytest.warns(UserWarning, match='below the local seafloor'):
            Bellhop(grid_type='R', verbose=False)._check_per_range_receiver_depth(
                self._env(), self._receiver())

    def test_a_paired_grid_reports_a_receiver_below_its_own_seafloor(self):
        deep = Receiver(depths=np.array([80.0, 100.0]),
                        ranges=np.array([1000.0, 5000.0]))
        with pytest.warns(UserWarning, match=r'range=1000\.0 m, depth=80\.0 m'):
            Bellhop(grid_type='I', verbose=False)._check_per_range_receiver_depth(
                self._env(), deep)

    def test_a_non_bellhop_wrapper_spans_the_product(self):
        with pytest.warns(UserWarning, match='below the local seafloor'):
            Kraken(verbose=False)._check_per_range_receiver_depth(
                self._env(), self._receiver())


# The six wrappers that synthesise p(t) from a broadband transfer function and
# so route their TIME_SERIES arguments through
# ``PropagationModel._require_timeseries_signal``. SPARC also declares
# TIME_SERIES but computes p(t) from its own ``pulse_type`` and never calls
# the helper.


# Rates that are not a positive finite number of Hz. 0 divided by zero in the
# delay-and-sum, -10000 produced a descending time axis, and nan/inf reached
# ``int()`` as a raw ValueError/OverflowError.


# ── what run() can hand back, and what it says it hands back ────────────────

def _result_stack_producers():
    """``{module path: [line numbers]}`` for every ``return ResultStack(…)``
    in shipped code — the producer half of the union the wrappers declare,
    read by AST so the annotations cannot drift away from the code that fills
    them."""
    import ast
    import pathlib

    import uacpy

    package = pathlib.Path(uacpy.__file__).resolve().parent
    found = {}
    for path in sorted(package.rglob('*.py')):
        if set(path.relative_to(package).parts) & {'tests', 'examples',
                                                   'third_party', 'bin',
                                                   '__pycache__'}:
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        lines = [node.lineno for node in ast.walk(tree)
                 if isinstance(node, ast.Return)
                 and isinstance(node.value, ast.Call)
                 and getattr(node.value.func, 'id', '') == 'ResultStack']
        if lines:
            found[str(path.relative_to(package.parent))] = lines
    return found


#: Entry points measured to hand back a ``ResultStack``. Driven, not read: a
#: 200 m guide, a 2-depth ``Source`` and the real binaries, over all 12
#: concrete wrappers × all 10 ``compute_*`` × {1, 2} source depths. Bellhop's
#: TL / RAYS / ARRIVALS / EIGENRAYS returned ``ResultStack`` at 2 depths and a
#: plain ``Result`` at 1.
_STACKING_ENTRY_POINTS = [
    ('PropagationModel', 'run'), ('Bellhop', 'run'),
    ('Bellhop', 'run_with_bounce'),
    ('PropagationModel', 'compute_tl'),
    ('PropagationModel', 'compute_rays'),
    ('PropagationModel', 'compute_arrivals'),
    ('PropagationModel', 'compute_eigenrays'),
]

#: The rest of ``compute_*``. In the same sweep every model that declares
#: these modes **refused** a multi-depth ``Source`` outright with a
#: ``ConfigurationError`` ("<model> takes a single source depth per run", and
#: for Bellhop's broadband pair "runs a single source depth"), so their
#: ``-> Result`` is total. Pinned so widening one needs the measurement
#: repeated rather than assumed.
_SINGLE_RESULT_ENTRY_POINTS = [
    ('PropagationModel', 'compute_modes'),
    ('PropagationModel', 'compute_reflection'),
    ('PropagationModel', 'compute_time_series'),
    ('PropagationModel', 'compute_transfer_function'),
    ('PropagationModel', 'compute_covariance'),
    ('PropagationModel', 'compute_replicas'),
]

_OWNERS = {'PropagationModel': PropagationModel, 'Bellhop': Bellhop}


def test_the_result_stack_producers_are_where_the_annotations_say():
    """The sweep behind the two gates below, so neither can pass against an
    empty set. Stacking happens in **two** places, and the second is the one a
    reader misses: ``Bellhop._run_eigenrays_multi_depth`` builds a stack in
    Python, and the OALIB readers build one whenever a ``.shd`` / ``.arr`` /
    ``.ray`` carries more than one source depth — which is why TL, RAYS and
    ARRIVALS stack too, without any wrapper looking as though they do."""
    producers = _result_stack_producers()
    assert 'uacpy/models/bellhop.py' in producers, producers
    assert 'uacpy/io/oalib_reader.py' in producers, (
        "no OALIB reader builds a ResultStack any more; if the readers stopped "
        "stacking, re-measure which compute_* can return one and narrow the "
        "annotations with it — do not assume\n" + repr(producers))
    assert len(producers['uacpy/io/oalib_reader.py']) >= 3, (
        "the OALIB readers build fewer ResultStacks than the three "
        "(TL, ARRIVALS, RAYS) the wrapper annotations are sized for: "
        + repr(producers))


@pytest.mark.parametrize('owner_name,method_name', _STACKING_ENTRY_POINTS,
                         ids=[f'{o}.{m}' for o, m in _STACKING_ENTRY_POINTS])
def test_an_entry_point_that_can_stack_declares_both_shapes(owner_name,
                                                            method_name):
    """``ResultStack`` is not a ``Result`` subclass — its MRO is
    ``(ResultStack, object)`` — so ``-> Result`` told a caller that
    ``res: Result = model.compute_tl(...)`` was correct while handing them an
    object with a different attribute surface.

    Each of these was driven with a 2-depth ``Source`` and the real binaries
    and returned a ``ResultStack``. The stacking is not visible at the call
    site: for TL, RAYS and ARRIVALS it happens inside the OALIB readers, which
    split a multi-source-depth file into one slab per depth."""
    import typing

    from uacpy.core.results import Result
    from uacpy.core.results.field import ResultStack

    assert not issubclass(ResultStack, Result), (
        "ResultStack is a Result subclass now, so `-> Result` covers it: "
        "narrow the unions and this gate together")
    method = getattr(_OWNERS[owner_name], method_name)
    annotation = typing.get_type_hints(method)['return']
    assert set(typing.get_args(annotation)) == {Result, ResultStack}, (
        f"{owner_name}.{method_name} declares {annotation!r}; driven over a "
        f"multi-depth Source it hands back a ResultStack, which is not a "
        f"Result")
    assert 'ResultStack' in (method.__doc__ or ''), (
        f"{owner_name}.{method_name}'s docstring Returns section does not "
        f"name ResultStack, so a reader who trusts the prose over the "
        f"annotation is told the wrong type")


@pytest.mark.parametrize('owner_name,method_name',
                         _SINGLE_RESULT_ENTRY_POINTS,
                         ids=[f'{o}.{m}' for o, m in
                              _SINGLE_RESULT_ENTRY_POINTS])
def test_an_entry_point_that_refuses_multiple_depths_declares_one_shape(
        owner_name, method_name):
    """The other side, pinned so it cannot be widened on a hunch either.

    Every model declaring these modes raises ``ConfigurationError`` on a
    ``Source`` with more than one depth, so the stack shape is unreachable
    through them and ``-> Result`` is total. Widening one of these means the
    sweep has to be re-run, not re-argued."""
    import typing

    from uacpy.core.results import Result

    method = getattr(_OWNERS[owner_name], method_name)
    annotation = typing.get_type_hints(method)['return']
    assert annotation is Result, (
        f"{owner_name}.{method_name} declares {annotation!r}; every model "
        f"that supports this mode refuses a multi-depth Source, so the "
        f"stack shape is unreachable here")


class TestMissingVolumeAbsorptionIsAnnounced:
    """The Acoustics Toolbox adds volume attenuation only when the option
    string asks for it — ``misc/AttenMod.f90:35-38`` makes ``T``/``F``/``B``
    the letters that add Thorp, Francois-Garrison or biological loss, and the
    SELECT CASE at ``:84`` has no default branch. So an environment with no
    absorption model runs through lossless water, in every model. That is a
    legitimate choice (the analytic benchmarks depend on it), but at high
    frequency it silently discards most of the loss, so it is said out loud
    whenever the omission is worth more than a decibel over the track."""

    @staticmethod
    def _triple(frequency, range_m):
        from uacpy.core.bottom import BoundaryProperties
        env = Environment(
            bathymetry=1000.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1650.0, density=1.9,
                                      attenuation=0.8))
        return (env, Source(depths=100.0, frequencies=frequency),
                Receiver(depths=[200.0], ranges=[range_m]))

    def _warnings_for(self, env, source, receiver):
        from uacpy.models.base import _warn_if_volume_absorption_is_missing
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _warn_if_volume_absorption_is_missing(env, source, receiver)
        return [str(w.message) for w in caught if 'absorption' in
                str(w.message)]

    def test_a_high_frequency_link_says_what_it_is_leaving_out(self):
        env, src, rcv = self._triple(40e3, 1000.0)
        msgs = self._warnings_for(env, src, rcv)
        assert msgs, "lossless water at 40 kHz was not announced"
        assert 'Thorp' in msgs[0] and 'FrancoisGarrison' in msgs[0]

    def test_a_low_frequency_run_stays_quiet(self):
        # 100 Hz over 1 km omits 4.5e-3 dB: below anything that could change
        # a decision, so warning about it would only train users to ignore it.
        env, src, rcv = self._triple(100.0, 1000.0)
        assert not self._warnings_for(env, src, rcv)

    def test_an_environment_that_names_a_model_stays_quiet(self):
        from uacpy.core.absorption import Thorp
        env, src, rcv = self._triple(40e3, 1000.0)
        env.absorption = Thorp()
        assert not self._warnings_for(env, src, rcv)

    def test_the_notice_quantifies_the_loss_it_is_dropping(self):
        from uacpy.core.absorption import Thorp
        env, src, rcv = self._triple(40e3, 1000.0)
        alpha = float(np.atleast_1d(
            Thorp().alpha_db_per_m(40e3, 0.0))[0])
        expected = alpha * 1000.0
        msg = self._warnings_for(env, src, rcv)[0]
        assert f"{expected:.1f}" in msg, msg

    def test_every_model_reaches_the_check(self):
        """It hangs off ``_require_run_triple``, which every wrapper calls, so
        no model can omit absorption quietly. Asserted by CALLING that hook on
        a stand-in model rather than by reading the source, which would stay
        green if the call were disabled."""
        from uacpy.models.base import PropagationModel

        class _AnyModel:
            """Stands in for a wrapper: the hook only reads these two."""
            model_name = 'Stub'
            _consumes_volume_absorption = True

        env, src, rcv = self._triple(40e3, 1000.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            PropagationModel._require_run_triple(_AnyModel(), env, src, rcv)
        assert [w for w in caught if 'absorption' in str(w.message)]

    def test_only_models_that_carry_absorption_advise_setting_it(self):
        """A wrapper that ignores ``env.absorption`` must not tell anyone to
        set it: RAM models no water-column attenuation at all and the OASES
        family substitutes its own empirical law, and both already warn when
        one IS set. Advising it there would contradict their own diagnostic."""
        from uacpy.models import Bellhop, Kraken, Scooter, SPARC, Bounce, RAM
        from uacpy.models.oases import OAST, OASN, OASP, OASR
        for model in (Bellhop, Kraken, Scooter, SPARC):
            assert model._consumes_volume_absorption, model.__name__
        # Bounce reaches the engine's TopOpt(4) too, but it tabulates R(theta)
        # at an interface: its `receiver` sizes the table's resolution rather
        # than describing a path, so the notice would quote that knob as a
        # propagation distance.
        for model in (Bounce, RAM, OAST, OASN, OASP, OASR):
            assert not model._consumes_volume_absorption, model.__name__

    def test_one_user_run_gives_one_notice(self):
        """The notice hangs off ``_require_run_triple``, which the internal
        re-runs call too — a broadband Bellhop run re-runs ARRIVALS at its
        carrier, and the routing path spawns a Bounce. Left alone that hands a
        user two notices quoting two different amounts of dropped loss at two
        different frequencies, one of which they never asked for."""
        from uacpy.models.base import _warn_if_volume_absorption_is_missing
        env, src, rcv = self._triple(40e3, 1000.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _warn_if_volume_absorption_is_missing(env, src, rcv)
            # the inner re-run: same environment, the carrier instead of the
            # band, so the message differs and would not be deduped
            inner = Source(depths=100.0, frequencies=20e3)
            _warn_if_volume_absorption_is_missing(env, inner, rcv)
        said = [str(w.message) for w in caught if 'absorption' in
                str(w.message)]
        assert len(said) == 1, said

    def test_the_default_is_not_to_advise(self):
        """Off by default, so a wrapper added later cannot inherit advice it
        does not honour — it has to say that it carries the model."""
        from uacpy.models.base import PropagationModel
        assert PropagationModel._consumes_volume_absorption is False


class TestTheBoundaryCarriersAlwaysDeclareTheFieldsModelsRead:
    """``models/base.py``, ``kraken.py`` and ``bounce.py`` read
    ``shear_speed`` / ``shear_attenuation`` / ``roughness`` / ``layers``
    straight off their carriers, with no ``getattr`` default and no
    ``or 0.0``. That is only sound while the carriers guarantee the field.

    ``BoundaryProperties.__post_init__`` fills every key of
    ``_ACOUSTIC_DEFAULTS`` unconditionally and then requires each
    non-negative; ``SedimentLayer`` declares the three as non-Optional floats
    defaulting to 0.0; ``SeabedColumn.__post_init__`` normalises ``layers`` to
    a list. Make any of them Optional again and the model layer starts
    comparing ``None > 0`` — which raises rather than reading as zero, so the
    failure would be loud but far from its cause. This pins it at the cause.
    """

    FLOAT_FIELDS = ('shear_speed', 'shear_attenuation', 'roughness')

    @pytest.mark.parametrize('boundary', [
        BoundaryProperties(),
        BoundaryProperties(acoustic_type='half-space', sound_speed=1600.0),
        BoundaryProperties(acoustic_type='rigid'),
        BoundaryProperties(acoustic_type='vacuum'),
        BoundaryProperties(shear_speed=300.0, shear_attenuation=0.2,
                           roughness=0.5),
    ])
    def test_a_boundary_carries_concrete_floats(self, boundary):
        for name in self.FLOAT_FIELDS:
            value = getattr(boundary, name)
            assert isinstance(value, float), (name, type(value), value)
            assert value >= 0.0, (name, value)

    @pytest.mark.parametrize('layer', [
        SedimentLayer(thickness=10.0, sound_speed=1600.0, density=1.6),
        SedimentLayer(thickness=10.0, sound_speed=1600.0, density=1.6,
                      shear_speed=300.0, shear_attenuation=0.2,
                      roughness=0.5),
    ])
    def test_a_sediment_layer_carries_concrete_floats(self, layer):
        for name in self.FLOAT_FIELDS:
            value = getattr(layer, name)
            assert isinstance(value, float), (name, type(value), value)
            assert value >= 0.0, (name, value)

    def test_a_pure_halfspace_column_has_an_empty_layers_list(self):
        column = SeabedColumn(layers=[], halfspace=BoundaryProperties())
        assert column.layers == []
        assert isinstance(column.layers, list)

    def test_the_readers_take_the_largest_roughness_off_the_carriers(self):
        """The call sites, not just the carriers: each folded reader against
        the value the old ``getattr(..., 0.0) or 0.0`` form would have
        produced."""
        from uacpy.models.base import _bottom_roughness, _max_roughness
        layer = SedimentLayer(thickness=10.0, sound_speed=1600.0, density=1.6,
                              shear_speed=300.0, roughness=0.7)
        halfspace = BoundaryProperties(acoustic_type='half-space',
                                       sound_speed=1800.0, roughness=0.2)
        bottom = Bottom([SeabedColumn(layers=[layer], halfspace=halfspace)])
        assert _max_roughness([layer, halfspace]) == pytest.approx(0.7)
        assert _bottom_roughness(bottom) == pytest.approx(0.7)
        # A pure half-space column: the layers list is empty, not absent.
        bare = Bottom([SeabedColumn(layers=[], halfspace=halfspace)])
        assert _bottom_roughness(bare) == pytest.approx(0.2)

    def test_the_elastic_collapse_zeroes_both_shear_fields(self):
        """``_zero_shear`` assigns unconditionally now, so it must still
        reach a layer, a half-space and a surface node."""
        layer = SedimentLayer(thickness=10.0, sound_speed=1600.0, density=1.6,
                              shear_speed=300.0, shear_attenuation=0.2)
        halfspace = BoundaryProperties(acoustic_type='half-space',
                                       sound_speed=1800.0, shear_speed=600.0,
                                       shear_attenuation=0.4)
        bottom = Bottom([SeabedColumn(layers=[layer], halfspace=halfspace)])
        collapsed = PropagationModel._collapse_elastic_boundary(bottom, 'fluid')
        for column in collapsed.columns:
            for carrier in (*column.layers, column.halfspace):
                assert carrier.shear_speed == 0.0
                assert carrier.shear_attenuation == 0.0
        # ... and the original is untouched (it deep-copies).
        assert bottom.columns[0].layers[0].shear_speed == 300.0

        surface = Surface(properties=[
            BoundaryProperties(shear_speed=200.0, shear_attenuation=0.1)])
        smoothed = PropagationModel._collapse_elastic_boundary(
            surface, 'fluid')
        assert smoothed.properties[0].shear_speed == 0.0
        assert smoothed.properties[0].shear_attenuation == 0.0

    def test_has_shear_keeps_its_duck_typed_fall_through(self):
        """``PropagationModel._has_shear``'s ``getattr`` is deliberately NOT
        folded: it is reached only past the Bottom/Surface isinstance branch
        and accepts a bare object that merely carries a shear speed. A sweep
        that folds the idiom everywhere would break this one."""
        class _JustAShearSpeed:
            shear_speed = 400.0

        class _NoShearAtAll:
            pass

        assert PropagationModel._has_shear(_JustAShearSpeed()) is True
        assert PropagationModel._has_shear(_NoShearAtAll()) is False
        assert PropagationModel._has_shear(None) is False
