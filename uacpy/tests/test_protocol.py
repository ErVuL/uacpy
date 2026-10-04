"""Tests that apply to every uacpy propagation model, not to one of them.

The shared surface in ``uacpy.models.base``: what ``run()`` accepts and the
order it accepts it in, how the speed bounds and ``c_max`` are derived from
the environment rather than guessed, backend selection as pure
introspection, the broadband frequency guard, the time-series entry points,
and the staged run protocol (settings, launches, the keyword rule).

Per-model behaviour lives in that model's own file (``test_kraken.py``,
``test_ram.py``, ...); what is here is the contract they are all held to,
usually parametrised across every wrapper so a new model cannot quietly
opt out of it.
"""

import numpy as np
import pytest
import types
import warnings
from dataclasses import dataclass
from uacpy.core import Environment
from uacpy.core import Source
from uacpy.core.bottom import Bottom
from uacpy.core.boundary import BoundaryProperties
from uacpy.core.bottom import SeabedColumn
from uacpy.core.boundary import SedimentLayer
from uacpy.core.exceptions import ConfigurationError, FallbackWarning
from uacpy.core.exceptions import UnsupportedFeatureError
from uacpy.core.receiver import Receiver
from uacpy.core.results import Field
from uacpy.core.results import Modes
from uacpy.core.surface import Surface
from uacpy.models import Bellhop
from uacpy.models import Bounce
from uacpy.models import Kraken
from uacpy.models import OASN
from uacpy.models import OASP
from uacpy.models import OASR
from uacpy.models import OASS
from uacpy.models import OASSP
from uacpy.models import OAST
from uacpy.models import RAM
from uacpy.models import SPARC
from uacpy.models import Scooter
from uacpy.core.run_settings import EngineSettings
from uacpy.core.run_settings import OutputSpec
from uacpy.core.run_settings import RunSettings
from uacpy.models._band import broadband_band, resolve_band
from uacpy.models._checks import check_carrier_types
from uacpy.models._projection import (
    _smooth_surface, collapse_elastic_boundary, has_shear,
)
from uacpy.models._spec import EngineTraits, ModelSpec
from uacpy.models.base import PropagationModel
from uacpy.core.run_settings import RunMode
from uacpy.models.kraken import _launch
from uacpy.tests.conftest import make_halfspace
from uacpy.tests.conftest import recorded_warnings


ALL_WRAPPERS = [Bellhop, Bounce, Kraken, RAM, Scooter, SPARC,
                OAST, OASN, OASR, OASP, OASSP, OASS]


TIMESERIES_WRAPPERS = [Bellhop, Kraken, RAM, Scooter, OASP, OASSP]


BAD_SAMPLE_RATES = [0.0, -10000.0, float('nan'), float('inf')]


_OASES_WRAPPERS = (OAST, OASN, OASR, OASP, OASSP, OASS)


def _wrapper_params(classes):
    """One parameter per wrapper class, named after it; the OASES ones carry
    ``requires_oases`` because constructing them looks up the separately
    installed OASES binaries."""
    return [pytest.param(c, id=c.__name__,
                         marks=(pytest.mark.requires_oases,)
                         if c in _OASES_WRAPPERS else ())
            for c in classes]


# OASES instantiation/supported-mode tests live in test_oases_variants.py;
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
        pytest.param(lambda: Kraken(verbose=False), id='Kraken'),
        pytest.param(lambda: OASP(verbose=False), id='OASP',
                     marks=pytest.mark.requires_oases),
        pytest.param(lambda: OASS(verbose=False, correlation_length=10.0),
                     id='OASS', marks=pytest.mark.requires_oases),
        pytest.param(lambda: RAM(verbose=False), id='RAM'),
        pytest.param(lambda: Scooter(verbose=False), id='Scooter'),
    ])
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


@pytest.mark.requires_binary
class TestTheBroadbandTailIsShared:
    """``_finish_broadband`` returns the BROADBAND transfer function as is and
    synthesises TIME_SERIES with the prepared pulse — the one tail every
    IFFT-based wrapper ends its broadband route with — the transfer function
    carrying the run's settings, so the synthesis reads the waveguide the
    run resolved."""

    @staticmethod
    def _settings(mode, time=None):
        return RunSettings(model='Kraken', mode=mode,
                           frequencies=np.array([100.0]),
                           source_depths=np.array([30.0]), time=time)

    def test_broadband_returns_the_transfer_function_itself(self):
        tf = object()
        assert Kraken(verbose=False)._finish_broadband(
            tf, self._settings(RunMode.BROADBAND)) is tf

    def test_time_series_synthesises_with_the_prepared_pulse(self):
        from uacpy.core.run_settings import TimeSettings
        calls = []

        class _Tf:
            _run_settings = None

            def synthesize_time_series(self, **kw):
                calls.append((self._run_settings, kw))
                return 'series'
        settings = self._settings(
            RunMode.TIME_SERIES,
            TimeSettings(source_waveform=np.ones(4), sample_rate=8000.0))
        out = Kraken(verbose=False)._finish_broadband(_Tf(), settings)
        assert out == 'series'
        (seen, kw), = calls
        assert seen is settings
        assert kw['sample_rate'] == 8000.0 and kw['t_start'] is None
        np.testing.assert_array_equal(kw['source_waveform'], np.ones(4))


@pytest.mark.requires_binary  # constructs models (resolves their binaries)
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
        model = Bellhop(verbose=False, use_tmpfs=True)
        with recorded_warnings() as caught:
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
            ssp=float(simple_env.ssp.sound_speed[0, 0]),
            bottom=BoundaryProperties(
                acoustic_type='half-space', sound_speed=1600, density=1.8,
                attenuation=0.2, shear_speed=400, shear_attenuation=0.5),
        )
        result = Bounce(verbose=False).run(env, source, receiver_small)
        assert result.backend == 'bounce'


@pytest.mark.requires_binary  # runs the models
class TestModelConsistency:
    """Tests for consistency between different models."""

    # Bellhop ↔ Kraken TL agreement is covered with tighter
    # tolerance in test_cross_model_agreement.py.

    @pytest.mark.slow
    @pytest.mark.requires_binary
    def test_krakenc_on_a_bounce_table_matches_the_direct_half_space(
            self, source, tmp_path):
        """KRAKENC reads a BOUNCE ``.brc`` through the Robin pair it builds
        from R at the real-part angle (``Kraken/BCImpedanceMod.f90:96-111``).
        Its trapped modes (c_high = the seabed's compressional speed) match
        KRAKENC on the elastic half-space itself: measured, 5 modes each,
        leading wavenumbers within 1.5e-4 relative."""
        from uacpy.core import BoundaryProperties, Receiver
        elastic = BoundaryProperties(
            acoustic_type='half-space', sound_speed=1600, shear_speed=400,
            density=1.8, attenuation=0.2, shear_attenuation=0.5)
        env = Environment(name='direct', bathymetry=100.0, ssp=1500.0,
                          bottom=elastic)
        table = Bounce(verbose=False, work_dir=tmp_path).run(
            env=env, source=source,
            receiver=Receiver(depths=[50.0], ranges=[5000.0]))
        via_table = Environment(
            name='table', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(
                acoustic_type='file',
                reflection_file=table.metadata['brc_file'],
                sound_speed=1600, density=1.8))
        kw = dict(backend='krakenc', verbose=False, c_low=1400.0,
                  c_high=1600.0)
        direct = Kraken(**kw).compute_modes(env=env, source=source)
        tabled = Kraken(**kw).compute_modes(env=via_table, source=source)
        assert len(tabled.k) == len(direct.k) == 5
        np.testing.assert_allclose(np.real(tabled.k), np.real(direct.k),
                                   rtol=5e-4)

    @pytest.mark.parametrize(
        "downstream",
        [
            pytest.param("Bellhop", id="bellhop"),
            pytest.param("Kraken", id="krakenc"),
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
            ssp=float(simple_env.ssp.sound_speed[0, 0]),
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
            ssp=float(simple_env.ssp.sound_speed[0, 0]),
            bottom=bottom_with_rc,
        )

        c_low_brc = bounce_result.run_settings.engine.c_low
        c_high_brc = bounce_result.run_settings.engine.c_high

        if downstream == "Kraken":
            # BOUNCE's c_high (1e9) sizes a table over the full angular span;
            # as KRAKENC's c_high it asks for every leaky mode, and on this
            # table the root finder keeps none of them (the direct elastic
            # half-space at that c_high ran past the 600 s timeout). The
            # seabed's compressional speed bounds the trapped modes, which
            # the table reproduces.
            modes = Kraken(backend='krakenc',
                verbose=False, c_low=c_low_brc,
                c_high=float(bottom_elastic.sound_speed),
            ).compute_modes(env=env_with_rc, source=source)
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
        from uacpy.core._warn_frames import USER_FRAME_SKIP

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
            nodes=[
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
        assert [node.roughness for node in smoothed.nodes] == [0, 0, 0]

    def test_the_input_surface_keeps_its_roughness(self):
        surface = self._three_node_surface()
        _smooth_surface(surface)
        assert [node.roughness
                for node in surface.nodes] == [0.5, 1.0, 1.5]


class TestSpeedBoundsFindThePhysicalExtremes:
    """``PropagationModel._speed_bounds`` spans the water column plus every
    geoacoustic seabed speed — the physical bracket the ``c_max`` stamp is
    taken from."""

    def test_the_halfspace_sets_the_maximum_when_fastest(self):
        env = Environment(name='hs', bathymetry=100.0, ssp=1500.0,
                          bottom=make_halfspace(3000.0))
        assert PropagationModel._speed_bounds(env) == (1500.0, 3000.0)

    def test_a_sediment_layer_faster_than_the_halfspace_sets_the_maximum(self):
        bottom = Bottom([SeabedColumn(
            layers=[SedimentLayer(thickness=20.0, sound_speed=1700.0,
                                  density=1.6, attenuation=0.3)],
            halfspace=make_halfspace(1600.0))])
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
        """The premise ``run_settings.waveguide`` is written against.

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
        that could hand back ``None`` or a bare value — reopens an arm the
        waveguide record does not check for."""
        import inspect
        body = inspect.getsource(PropagationModel._speed_bounds)
        assert body.count('return') == 1, body


@pytest.mark.requires_binary
class TestTheWaveguideCMaxIsThePhysicalMaximum:
    """``run_settings.waveguide.c_max`` is the fastest compressional speed
    anywhere in the environment — the anchor speed ``Field.to_time_trace``
    needs, never an algorithmic reference — and an engine's result carries
    it there, not as a second copy in its metadata."""

    def _waveguide(self, env):
        return Kraken(verbose=False).run_settings(
            env, Source(depths=50.0, frequencies=100.0),
            Receiver(depths=[50.0], ranges=[1000.0])).waveguide

    def test_the_seabed_speed_wins_over_the_water_column(self):
        env = Environment(name='cmax', bathymetry=100.0, ssp=1500.0,
                          bottom=make_halfspace(3000.0))
        assert self._waveguide(env).c_max == 3000.0

    def test_the_water_column_wins_under_a_rigid_bottom(self):
        env = Environment(
            name='cmax_rigid', bathymetry=100.0,
            ssp=[(0.0, 1500.0), (100.0, 1520.0)],
            bottom=BoundaryProperties(acoustic_type='rigid'))
        assert self._waveguide(env).c_max == 1520.0


@pytest.mark.requires_binary
class TestBroadbandNFreqsGuard:
    """``_band.broadband_band`` refuses an expansion that cannot
    span the band: ``np.linspace`` with one point returns the lower band
    edge alone and with zero an empty grid."""

    _SRC = Source(depths=50.0, frequencies=100.0)

    def test_n_freqs_one_is_a_configuration_error(self):
        model = Bellhop(verbose=False)
        with pytest.raises(ConfigurationError, match=r'n_freqs = 1'):
            broadband_band(self._SRC, 1, model.bandwidth_factor,
                           model_name='Bellhop')

    def test_n_freqs_zero_is_a_configuration_error(self):
        model = Bellhop(verbose=False)
        with pytest.raises(ConfigurationError, match=r'n_freqs = 0'):
            broadband_band(self._SRC, 0, model.bandwidth_factor,
                           model_name='Bellhop')

    def test_an_explicit_grid_bypasses_the_guard(self):
        got = resolve_band(RunMode.BROADBAND, self._SRC, [90.0, 100.0],
                           None, model_name='Bellhop', n_freqs=0)
        np.testing.assert_allclose(got.frequencies, [90.0, 100.0])


@pytest.mark.requires_binary
class TestBroadbandBandwidthFactorGuard:
    """``_band.broadband_band`` names the cause of an empty band:
    a non-positive ``bandwidth_factor`` (the band ``fc·(1 ± bf/2)`` inverts
    or collapses at any fc) is reported as such, and only a centre frequency
    the 1 Hz floor overtakes is blamed on being sub-1 Hz."""

    _SRC = Source(depths=50.0, frequencies=1000.0)

    def _resolve(self, source, bandwidth_factor):
        return broadband_band(source, 8, bandwidth_factor,
                              model_name='Bellhop').frequencies

    @pytest.mark.parametrize('bandwidth_factor', [-0.5, 0.0])
    def test_a_non_positive_factor_is_named_as_the_cause(self, bandwidth_factor):
        with pytest.raises(ConfigurationError,
                           match='bandwidth_factor must be positive') as exc:
            self._resolve(self._SRC, bandwidth_factor)
        text = str(exc.value)
        assert 'bandwidth_factor must be positive' in text, text
        assert 'Sub-1 Hz' not in text, text

    def test_the_smallest_positive_factor_expands_the_band(self):
        got = self._resolve(self._SRC, 1e-3)
        assert got.size == 8 and got[0] < got[-1]

    def test_a_sub_1_hz_centre_frequency_is_blamed_on_the_floor(self):
        with pytest.raises(ConfigurationError, match=r'Sub-1 Hz centre'):
            self._resolve(Source(depths=50.0, frequencies=0.5), 0.5)


@pytest.mark.requires_binary
class TestSelectBackendIsPureIntrospection:
    """``Kraken.select_backend`` decides the backend name from the
    environment alone; the ``.irc`` header read is a refusal of the
    validation stage, and the executable lookup happens at launch."""

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
        elastic = self._env(make_halfspace(1800.0, shear_speed=400.0,
                                       shear_attenuation=0.5))
        fluid = self._env(make_halfspace(1800.0))
        assert model.select_backend(elastic) == 'krakenc'
        assert model.select_backend(fluid) == 'kraken'

    def test_forcing_kraken_on_elastic_media_raises_without_disk(
            self, monkeypatch):
        model = Kraken(verbose=False, backend='kraken')
        monkeypatch.setattr(model, '_find_executable_in_paths', self._no_disk)
        elastic = self._env(make_halfspace(1800.0, shear_speed=400.0,
                                       shear_attenuation=0.5))
        with pytest.raises(ConfigurationError, match='elastic media'):
            model.select_backend(elastic)

    def test_a_malformed_irc_bottom_is_refused_by_validation_not_by_select_backend(
            self, tmp_path):
        table = tmp_path / 'bot.irc'
        table.write_text('3\n0.0 1.0 180.0\n45.0 1.0 180.0\n')
        env = self._env(BoundaryProperties(acoustic_type='precalc',
                                           reflection_file=str(table)))
        model = Kraken(verbose=False)
        assert model.select_backend(env) == 'krakenc'
        with pytest.raises(ConfigurationError, match=r'\.irc'):
            model.validate_inputs(env, Source(depths=50.0, frequencies=100.0),
                                  Receiver(depths=[50.0], ranges=[1000.0]))


@pytest.mark.requires_binary
class TestFieldPrtAttach:
    """``kraken._launch.attach_field_prt_path`` records field.exe's
    hard-coded ``field.prt`` under its own metadata key, existence-checked,
    iff the scratch survives — ``_attach_output_paths`` only ever sees the
    modes binary's ``kfield.prt``."""

    def test_field_prt_is_attached_when_the_scratch_survives(self, tmp_path):
        (tmp_path / 'field.prt').write_text('Field completed successfully\n')
        result = types.SimpleNamespace(metadata={})
        model = Kraken(verbose=False, cleanup=False)
        _launch.attach_field_prt_path(
            result, tmp_path, cleanup=model.cleanup)
        assert result.metadata['field_prt_file'] == str(tmp_path / 'field.prt')

    def test_a_missing_field_prt_leaves_no_key(self, tmp_path):
        result = types.SimpleNamespace(metadata={})
        model = Kraken(verbose=False, cleanup=False)
        _launch.attach_field_prt_path(
            result, tmp_path, cleanup=model.cleanup)
        assert 'field_prt_file' not in result.metadata

    def test_cleanup_true_leaves_no_key(self, tmp_path):
        (tmp_path / 'field.prt').write_text('Field completed successfully\n')
        result = types.SimpleNamespace(metadata={})
        model = Kraken(verbose=False, cleanup=True)
        _launch.attach_field_prt_path(
            result, tmp_path, cleanup=model.cleanup)
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
    ``_checks.check_carrier_types``: the (env, source, receiver)
    argument order is checked before run-mode resolution or deck assembly,
    so a swapped pair raises one typed error instead of a raw
    ``AttributeError`` from deep inside the writer."""

    @pytest.mark.parametrize('cls', _wrapper_params(ALL_WRAPPERS))
    def test_source_passed_in_the_env_slot_raises_naming_the_order(self, cls):
        with pytest.raises(ConfigurationError, match='in that order'):
            _model(cls).run(_source(), _env(), _receiver())

    @pytest.mark.parametrize('cls', _wrapper_params(ALL_WRAPPERS))
    def test_receiver_and_source_swapped_raises_naming_the_order(self, cls):
        with pytest.raises(ConfigurationError, match='in that order'):
            _model(cls).run(_env(), _receiver(), _source())

    def test_the_error_names_each_wrong_slot_with_the_received_type(self):
        with pytest.raises(ConfigurationError,
                           match='env=Source, source=Environment'):
            _model(Bellhop).run(_source(), _env(), _receiver())

    def test_a_correct_triple_passes_the_validator(self):
        check_carrier_types('Bellhop', _env(), _source(), _receiver())

    def test_carrier_subclasses_pass_the_validator(self):
        class _TaggedSource(Source):
            pass

        tagged = _TaggedSource(depths=25.0, frequencies=200.0)
        check_carrier_types('Bellhop', _env(), tagged, _receiver())


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

    @pytest.mark.parametrize('cls', _wrapper_params(TIMESERIES_WRAPPERS))
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
        # ``run()`` reaches the guard in stage 2, so no deck is
        # written and no binary is spawned.
        with pytest.raises(ConfigurationError, match='sample_rate'):
            _model(Bellhop).run(
                Environment(name='flat', bathymetry=100.0, ssp=1500.0),
                Source(depths=25.0, frequencies=200.0),
                Receiver(depths=np.array([50.0]), ranges=np.array([1000.0])),
                run_mode=RunMode.TIME_SERIES,
                source_waveform=_waveform(), sample_rate=rate)


class TestSynthesizeTimeSeriesRequiresAPositiveFiniteSampleRate:
    """The deep guard in ``acoustic_signal/_synthesis.py`` backs the wrapper-level one
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
            kind='pressure', unit='Pa')

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

    def test_a_rectilinear_grid_logs_the_cross_term(self, capsys):
        # 100 m at 1 km is under the 50 m seafloor, but 20 m keeps that range
        # in the water: a rectangular grid's buried cells are an info line.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            Bellhop(grid_type='R', verbose='info')._check_per_range_receiver_depth(
                self._env(), self._receiver())
        assert 'below the local seafloor' in capsys.readouterr().out

    def test_a_paired_grid_reports_a_receiver_below_its_own_seafloor(self):
        deep = Receiver(depths=np.array([80.0, 100.0]),
                        ranges=np.array([1000.0, 5000.0]))
        with pytest.warns(UserWarning, match=r'range=1000\.0 m, depth=80\.0 m'):
            Bellhop(grid_type='I', verbose=False)._check_per_range_receiver_depth(
                self._env(), deep)

    def test_a_non_bellhop_wrapper_spans_the_product(self):
        # Both depths under the 50 m seafloor at 1 km: a buried range.
        deep = Receiver(depths=np.array([80.0, 100.0]),
                        ranges=np.array([1000.0, 5000.0]))
        with pytest.warns(UserWarning, match='lie entirely below the local '
                                             'seafloor'):
            Kraken(verbose=False)._check_per_range_receiver_depth(
                self._env(), deep)


# The six wrappers that synthesise p(t) from a broadband transfer function and
# so route their TIME_SERIES arguments through
# ``PropagationModel._require_timeseries_signal``. SPARC also declares
# TIME_SERIES but computes p(t) from its own ``pulse_type`` and never calls
# the helper.


# Rates that are not a positive finite number of Hz. 0 divided by zero in the
# delay-and-sum, -10000 produced a descending time axis, and nan/inf reached
# ``int()`` as a raw ValueError/OverflowError.


# ── what run() can hand back, and what it says it hands back ────────────────

def _builds_a_result_stack(func):
    """Whether a call to ``func`` builds a ``ResultStack``: the constructor,
    or ``ResultStack.from_slabs``, which stacks two or more slabs."""
    import ast

    if isinstance(func, ast.Name):
        return func.id == 'ResultStack'
    return (isinstance(func, ast.Attribute) and func.attr == 'from_slabs'
            and getattr(func.value, 'id', '') == 'ResultStack')


def _result_stack_producers():
    """``{module path: [line numbers]}`` for every ``return ResultStack(…)``
    and ``return ResultStack.from_slabs(…)`` in shipped code — the producer
    half of the union the wrappers declare,
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
                 and _builds_a_result_stack(node.value.func)]
        if lines:
            found[str(path.relative_to(package.parent))] = lines
    return found


#: Entry points measured to hand back a ``ResultStack``. Driven, not read: a
#: 200 m guide, a 2-depth ``Source`` and the real binaries, over all 12
#: concrete wrappers × all 10 ``compute_*`` × {1, 2} source depths. Bellhop's
#: TL / RAYS / ARRIVALS / EIGENRAYS returned ``ResultStack`` at 2 depths and a
#: plain ``Result`` at 1. The field modes (TL, BROADBAND, TIME_SERIES) stack on
#: every engine since ``PropagationModel.run`` loops the depths
#: (test_multi_source.py drives the five field models).
_STACKING_ENTRY_POINTS = [
    ('PropagationModel', 'run'), ('Bellhop', 'run'),
    ('Bellhop', 'run_with_bounce'),
    ('PropagationModel', 'compute_tl'),
    ('PropagationModel', 'compute_rays'),
    ('PropagationModel', 'compute_arrivals'),
    ('PropagationModel', 'compute_eigenrays'),
    ('PropagationModel', 'compute_time_series'),
    ('PropagationModel', 'compute_transfer_function'),
]


#: The rest of ``compute_*``: the non-field modes. Every model that declares
#: these **refuses** a multi-depth ``Source`` with a ``ConfigurationError``
#: ("<model> takes a single source depth per <MODE> run"), so their
#: ``-> Result`` is total. Pinned so widening one needs the measurement
#: repeated rather than assumed.
_SINGLE_RESULT_ENTRY_POINTS = [
    ('PropagationModel', 'compute_modes'),
    ('PropagationModel', 'compute_reflection'),
    ('PropagationModel', 'compute_covariance'),
    ('PropagationModel', 'compute_replicas'),
]


_OWNERS = {'PropagationModel': PropagationModel, 'Bellhop': Bellhop}


def test_the_result_stack_producers_are_where_the_annotations_say():
    """The sweep behind the two gates below, so neither can pass against an
    empty set. Stacking happens in **three** places, and two of them are the
    ones a reader misses: ``_stacking.run_per_source_depth`` loops a
    field mode over the depths for every engine, and Bellhop's eigenrays
    (its ``spec.traits.python_stacked_modes``), and
    the OALIB readers build one whenever a ``.shd`` / ``.arr`` / ``.ray``
    carries more than one source depth — which is why Bellhop's TL, RAYS and
    ARRIVALS stack without the wrapper looking as though it does (Scooter's
    ``assemble_field_from_grn`` stacks its own per-depth Hankel
    transforms)."""
    producers = _result_stack_producers()
    assert 'uacpy/models/_stacking.py' in producers, producers
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
    from uacpy.core.results.stack import ResultStack

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
        from uacpy.core.boundary import BoundaryProperties
        env = Environment(
            bathymetry=1000.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1650.0, density=1.9,
                                      attenuation=0.8))
        return (env, Source(depths=100.0, frequencies=frequency),
                Receiver(depths=[200.0], ranges=[range_m]))

    def _warnings_for(self, env, source, receiver):
        from uacpy.models._notices import _warn_if_volume_absorption_is_missing
        with recorded_warnings() as caught:
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
            Thorp().alpha_dB_per_m(40e3, 0.0))[0])
        expected = alpha * 1000.0
        msg = self._warnings_for(env, src, rcv)[0]
        assert f"{expected:.1f}" in msg, msg

    def test_every_model_reaches_the_check(self):
        """It hangs off the argument-check stage every entry point runs
        (``run``, ``run_settings``, ``validate_inputs``), so no model can
        omit absorption quietly. Asserted by CALLING that stage on a
        stand-in model rather than by reading the source, which would stay
        green if the call were disabled."""
        env, src, rcv = self._triple(40e3, 1000.0)
        with recorded_warnings() as caught:
            _AbsorbingStub().run_settings(env, src, rcv)
        assert [w for w in caught if 'absorption' in str(w.message)]

    def test_only_models_that_carry_absorption_advise_setting_it(self):
        """A wrapper that ignores ``env.absorption`` must not tell anyone to
        set it. RAM carries the model as a dB/wavelength profile on every
        backend and the OASES propagators as each water layer's AC in
        dB/wavelength, so they advise like the Acoustics Toolbox wrappers.
        OASR's water is a lossless half-space with no path to act on."""
        from uacpy.models import Bellhop, Kraken, Scooter, SPARC, Bounce, RAM
        from uacpy.models.oases import OAST, OASN, OASP, OASR, OASS, OASSP
        for model in (Bellhop, Kraken, Scooter, RAM,
                      OAST, OASN, OASP, OASS, OASSP):
            assert model.spec.traits.consumes_volume_absorption, model.__name__
        # Bounce reaches the engine's TopOpt(4) too, but it tabulates R(theta)
        # at an interface: its `receiver` sizes the table's resolution rather
        # than describing a path, so the notice would quote that knob as a
        # propagation distance. SPARC writes it into the deck, but its march
        # keeps the real part of the sound speed only (sparc.f90:221).
        for model in (Bounce, OASR, SPARC):
            assert not model.spec.traits.consumes_volume_absorption, (
                model.__name__)

    def test_one_user_run_gives_one_notice(self):
        """A run or run_settings started inside a run (a model one of its
        hooks runs) reaches the check too. Left alone that hands a user two
        notices quoting two different amounts of dropped loss at two
        different frequencies, one of which they never asked for. The
        stand-in asks for a 20 kHz run's settings inside a 40 kHz run: one
        notice, the outer one."""
        env, src, rcv = self._triple(40e3, 1000.0)
        with recorded_warnings() as caught:
            _AbsorbingStub().run(env, src, rcv)
        said = [str(w.message) for w in caught if 'absorption' in
                str(w.message)]
        assert len(said) == 1, said
        assert '40000 Hz' in said[0], said

    def test_every_user_run_is_told_even_on_the_same_environment(self):
        """Once per RUN, not once per Environment (RA-CONTRACT-19): a second
        run on the same object, or on a copy of it, is a new run the user
        asked for and is told again; nothing is written on the carrier."""
        env, src, rcv = self._triple(40e3, 1000.0)
        counts = []
        for target in (env, env, env.copy()):
            with recorded_warnings() as caught:
                _AbsorbingStub().run(target, src, rcv)
            counts.append(sum('absorption' in str(w.message)
                              for w in caught))
        assert counts == [1, 1, 1]
        assert not hasattr(env, '_absorption_notice_given')

    def test_the_default_is_not_to_advise(self):
        """Off by default, so a wrapper added later cannot inherit advice it
        does not honour — it has to say that it carries the model."""
        assert EngineTraits().consumes_volume_absorption is False


class TestASinkCollectsTheNotices:
    """A sink collects the notices a resolver gives, as
    :class:`~uacpy.core.run_settings.Notice` records, and nothing is said
    until the run announces them."""

    def test_a_sink_collects_and_no_sink_warns(self):
        from uacpy.models._notices import give_notice
        sink = []
        with recorded_warnings() as caught:
            give_notice(sink, 'collected', FallbackWarning)
        assert [n.message for n in sink] == ['collected'] and caught == []
        assert sink[0].category is FallbackWarning
        with recorded_warnings() as caught:
            give_notice(None, 'said', FallbackWarning)
        assert [(str(w.message), w.category) for w in caught] == [
            ('said', FallbackWarning)]


class TestOneRunnerLaunchesEveryBinary:
    """``_launch.run_launch``: the stale outputs and the ``.prt`` go before
    the binary runs; a failure is raised with the ``.prt`` tail attached and
    no check run, unless ``tolerate_exit`` accepts it, when the checks run
    on ``None``; otherwise the checks run in order on the completed
    process."""

    @staticmethod
    def _launch(tmp_path, **kw):
        from uacpy.models._launch import Launch
        return Launch(argv=('engine', 'deck'), cwd=tmp_path,
                      stale_outputs=('deck.out',), prt_root='deck', **kw)

    @staticmethod
    def _failure():
        from uacpy.core.exceptions import ModelExecutionError
        return ModelExecutionError('Engine', return_code=3, stdout=None,
                                   stderr='boom')

    def test_the_stale_files_go_and_the_checks_run_in_order(self, tmp_path):
        import subprocess
        from uacpy.models._launch import run_launch
        (tmp_path / 'deck.out').write_text('old')
        (tmp_path / 'deck.prt').write_text('old')
        seen = []

        def run(argv, *, cwd, timeout, env):
            seen.append(sorted(p.name for p in tmp_path.iterdir()))
            return subprocess.CompletedProcess(argv, 0, 'out', '')
        result = run_launch(self._launch(tmp_path, checks=(
            lambda r: seen.append(('first', r.stdout)),
            lambda r: seen.append(('second', r.returncode)))), run=run)
        assert seen == [[], ('first', 'out'), ('second', 0)]
        assert result.args == ['engine', 'deck']

    def test_a_failure_carries_the_prt_tail_and_skips_the_checks(
            self, tmp_path):
        from uacpy.core.exceptions import ModelExecutionError
        from uacpy.models._launch import run_launch
        checked = []

        def run(argv, *, cwd, timeout, env):
            (tmp_path / 'deck.prt').write_text('*** FATAL ERROR *** why')
            raise self._failure()
        with pytest.raises(ModelExecutionError, match='why'):
            run_launch(self._launch(tmp_path, checks=(checked.append,)),
                       run=run)
        assert checked == []

    @pytest.mark.parametrize('tolerated', [True, False])
    def test_a_tolerated_failure_runs_the_checks_on_none(self, tmp_path,
                                                         tolerated):
        from uacpy.core.exceptions import ModelExecutionError
        from uacpy.models._launch import run_launch
        checked = []

        def run(argv, *, cwd, timeout, env):
            raise self._failure()
        launch = self._launch(tmp_path, checks=(checked.append,),
                              tolerate_exit=lambda exc: tolerated)
        if tolerated:
            assert run_launch(launch, run=run) is None
            assert checked == [None]
        else:
            with pytest.raises(ModelExecutionError, match='boom'):
                run_launch(launch, run=run)
            assert checked == []


class TestTheBoundaryCarriersAlwaysDeclareTheFieldsModelsRead:
    """``models/_projection.py``, ``models/kraken/`` and ``bounce/_plan.py`` read
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
        from uacpy.models._projection import _bottom_roughness, _max_roughness
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
        collapsed = collapse_elastic_boundary(bottom, 'fluid')
        for column in collapsed.columns:
            for carrier in (*column.layers, column.halfspace):
                assert carrier.shear_speed == 0.0
                assert carrier.shear_attenuation == 0.0
        # ... and the original is untouched (it deep-copies).
        assert bottom.columns[0].layers[0].shear_speed == 300.0

        surface = Surface(nodes=[
            BoundaryProperties(shear_speed=200.0, shear_attenuation=0.1)])
        smoothed = collapse_elastic_boundary(
            surface, 'fluid')
        assert smoothed.nodes[0].shear_speed == 0.0
        assert smoothed.nodes[0].shear_attenuation == 0.0

    def test_has_shear_keeps_its_duck_typed_fall_through(self):
        """``has_shear``'s ``getattr`` is deliberately NOT
        folded: it is reached only past the Bottom/Surface isinstance branch
        and accepts a bare object that merely carries a shear speed. A sweep
        that folds the idiom everywhere would break this one."""
        class _JustAShearSpeed:
            shear_speed = 400.0

        class _NoShearAtAll:
            pass

        assert has_shear(_JustAShearSpeed()) is True
        assert has_shear(_NoShearAtAll()) is False
        assert has_shear(None) is False


def _never_launched(self, *args):
    """A stage hook of a stand-in model whose calls are all refused before
    stage 4."""
    raise AssertionError('refused before any launch')


class _AbsorbingStub(PropagationModel):
    """A model that carries ``env.absorption`` and launches nothing: while
    it builds its result it asks for the settings of a 20 kHz run, as a
    model one of its hooks runs would, and returns a one-cell field."""
    spec = ModelSpec(modes=(RunMode.COHERENT_TL,),
                     traits=EngineTraits(consumes_volume_absorption=True))
    provenance_id = 'acoustics_toolbox'

    def _write_input(self, inputs):
        return inputs.work_dir

    def _launch(self, inputs, deck):
        pass

    def _read_output(self, inputs, deck):
        return None

    def _to_result(self, inputs, deck, raw):
        source, receiver = inputs.source, inputs.receiver
        inner = Source(depths=float(source.depths[0]), frequencies=20e3)
        self.run_settings(inputs.env, inner, receiver)
        return Field(data=np.ones((1, 1), complex),
                     coords={'depth': [float(receiver.depths[0])],
                             'range': [float(receiver.ranges[0])]},
                     **self._result_kwargs(source, frequencies=source.frequencies,
                                           phase_reference='travelling_wave'))


# ── the run protocol: one resolution, one validation, one stamp ──────────

def _stub_triple(n_depths=1, weights=None, frequencies=100.0):
    env = Environment(bathymetry=100.0, ssp=1500.0)
    depths = [30.0, 60.0][:n_depths]
    kw = {} if weights is None else {'weights': weights}
    src = Source(depths=depths, frequencies=frequencies, **kw)
    rcv = Receiver(depths=[20.0, 50.0], ranges=[500.0, 1000.0])
    return env, src, rcv


class _FieldStub(PropagationModel):
    """A field model that launches nothing: COHERENT_TL / BROADBAND /
    TIME_SERIES, ``run_mode=None`` resolving to BROADBAND when a
    multi-element ``frequencies=`` is passed (Kraken's rule), no native
    multi-depth mode. Each call of the engine records the mode, source and
    frequencies it was handed and returns a field whose value is the source
    depth."""
    spec = ModelSpec(modes=(RunMode.COHERENT_TL, RunMode.BROADBAND,
                            RunMode.TIME_SERIES),
                     traits=EngineTraits(consumes_run_t_start=True))
    provenance_id = 'acoustics_toolbox'

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = []

    def _default_run_mode_for(self, frequencies):
        if frequencies is not None and np.size(frequencies) > 1:
            return RunMode.BROADBAND
        return self._default_run_mode()

    def _write_input(self, inputs):
        return inputs.work_dir

    def _launch(self, inputs, deck):
        pass

    def _read_output(self, inputs, deck):
        return None

    def _to_result(self, inputs, deck, raw):
        source, receiver = inputs.source, inputs.receiver
        f = inputs.settings.frequencies
        self.calls.append((inputs.settings.mode, source.depths.tolist(),
                           f.tolist()))
        data = np.full((receiver.depths.size, receiver.ranges.size),
                       float(source.depths[0]), dtype=complex)
        return Field(data=data,
                     coords={'depth': receiver.depths,
                             'range': receiver.ranges},
                     **self._result_kwargs(source, frequencies=f,
                                           phase_reference='travelling_wave'))


class TestRunSettings:
    """``RunSettings``: immutable, printable, round-trips, pickles."""

    def _settings(self):
        env, src, rcv = _stub_triple(2, weights=[1.0, 0.5j])
        return _FieldStub().run_settings(env, src, rcv,
                                         frequencies=[90.0, 100.0, 110.0])

    def test_the_repr_is_one_line_per_setting_and_the_summary_one_line(self):
        s = self._settings()
        text = repr(s)
        assert text.startswith('RunSettings(\n') and text.endswith('\n)')
        lines = text.splitlines()[1:-1]
        assert [ln.split()[0] for ln in lines[:2]] == ['model', 'mode']
        assert 'BROADBAND' in lines[1] and '90, 100, 110 Hz' in text
        assert 'per_depth' in text and 'applied' in text
        assert '\n' not in s.summary()
        assert s.summary().startswith('_FieldStub; broadband; f 90, 100, 110 Hz')

    def test_it_is_frozen_and_its_arrays_are_read_only(self):
        import dataclasses
        s = self._settings()
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.mode = RunMode.COHERENT_TL
        for arr in (s.frequencies, s.source_depths, s.source_weights):
            with pytest.raises(ValueError,
                               match='assignment destination is read-only'):
                arr[0] = 0.0

    def test_to_dict_is_plain_types_and_from_dict_rebuilds_it(self):
        import json
        from uacpy.core.run_settings import RunSettings
        s = self._settings()
        d = s.to_dict()
        json.dumps(d)                       # plain types only
        back = RunSettings.from_dict(d)
        assert back == s
        assert back.mode is RunMode.BROADBAND
        assert back.source_weights.tolist() == [1.0, 0.5j]

    def test_it_pickles_and_stays_read_only(self):
        import pickle
        s = self._settings()
        back = pickle.loads(pickle.dumps(s))
        assert back == s
        with pytest.raises(ValueError,
                           match='assignment destination is read-only'):
            back.frequencies[0] = 0.0


class TestTheStampCarriesTheGridTheResultWasComputedOn:
    """Stage 6 stamps the settings with the frequencies the result carries
    (``_settings_as_run``): unchanged when they are the resolved ones, with
    a note naming both grids when they differ. A note never names two
    identical summaries: when the grids print alike it adds the largest
    difference."""

    @staticmethod
    def _stamp(realised):
        from uacpy.models._extract import _settings_as_run
        env, src, rcv = _stub_triple(1)
        settings = _FieldStub().run_settings(
            env, src, rcv, frequencies=[90.0, 100.0, 110.0])
        return settings, _settings_as_run(
            settings, types.SimpleNamespace(frequencies=np.asarray(realised)))

    def test_the_resolved_grid_stamps_the_settings_unchanged(self):
        settings, stamped = self._stamp([90.0, 100.0, 110.0])
        assert stamped is settings

    def test_a_different_grid_is_stamped_with_a_note_naming_both(self):
        settings, stamped = self._stamp([90.0, 100.0, 120.0])
        np.testing.assert_array_equal(stamped.frequencies,
                                      [90.0, 100.0, 120.0])
        assert stamped.notes[-1] == (
            "frequencies: _FieldStub's run propagated 90, 100, 120 Hz in "
            "place of the resolved 90, 100, 110 Hz")

    def test_a_grid_that_prints_alike_names_its_largest_difference(self):
        settings, stamped = self._stamp([90.0, 100.0, 110.0 + 2e-6])
        assert stamped.notes[-1] == (
            "frequencies: _FieldStub's run propagated 90, 100, 110 Hz in "
            "place of the resolved 90, 100, 110 Hz (largest difference "
            "2e-06 Hz)")


class TestARequestedAxisIsRestoredWithinItsPrecision:
    """``_restore_requested_axis`` gives back the requested axis when the
    one read from an engine's file agrees with it to within ``rtol`` of the
    axis's largest magnitude, the precision the file carries it at, and the
    read-back axis otherwise."""

    @staticmethod
    def _restore(read, requested, rtol):
        from uacpy.models._extract import _restore_requested_axis
        return _restore_requested_axis(np.asarray(read, dtype=float),
                                       np.asarray(requested, dtype=float),
                                       rtol)

    @pytest.mark.parametrize('offset,restored', [(0.9, True), (1.1, False)])
    def test_the_bound_is_rtol_times_the_largest_magnitude(self, offset,
                                                           restored):
        requested = np.array([0.0, 200.0, 400.0])
        read = requested + np.array([0.0, 0.0, offset * 1e-6 * 400.0])
        out = self._restore(read, requested, 1e-6)
        np.testing.assert_array_equal(out, requested if restored else read)

    def test_an_axis_of_another_length_is_returned_as_read(self):
        out = self._restore([0.0, 200.0], [0.0, 200.0, 400.0], 1e-6)
        np.testing.assert_array_equal(out, [0.0, 200.0])

    def test_a_band_read_back_from_twelve_digit_deck_text_is_restored(self):
        """Kraken writes the band ``%.12g`` (``oalib_writer``) and reads it
        back from the ``.shd``."""
        from uacpy.models.kraken._extract import _SHD_FREQUENCY_RTOL
        band = np.linspace(10.0, 1280.0, 127)
        read = np.array([float(f"{f:.12g}") for f in band])
        assert not np.array_equal(read, band)
        np.testing.assert_array_equal(
            self._restore(read, band, _SHD_FREQUENCY_RTOL), band)
        shifted = read + 1e-6
        np.testing.assert_array_equal(
            self._restore(shifted, band, _SHD_FREQUENCY_RTOL), shifted)

    def test_a_range_axis_read_back_in_real4_is_restored(self):
        """OASP and OASSP read ``R0``/``RSPACE`` (km) back from the ``.trf``
        as REAL*4 (``TRFHEAD``, ``oasiun23.f:814``)."""
        from uacpy.models.oases.oasp import _TRF_RANGE_RTOL
        requested = np.array([0.0, 200.0, 400.0])
        read = 1e3 * (np.float32(0.0)
                      + np.arange(3) * float(np.float32(0.2)))
        assert not np.array_equal(read, requested)
        np.testing.assert_array_equal(
            self._restore(read, requested, _TRF_RANGE_RTOL), requested)
        np.testing.assert_array_equal(
            self._restore(read + 0.01, requested, _TRF_RANGE_RTOL),
            read + 0.01)


class TestOneRunProtocol:
    """``run`` resolves every fact once and hands the engine the resolved
    mode; ``run_settings`` and ``validate_inputs`` run the same stages."""

    def test_run_mode_none_with_frequencies_splits_a_multi_depth_source(self):
        """ARCH-1 / RA-CONTRACT-1: the mode that decides the depth loop is
        the mode that runs. ``run_mode=None`` + a frequency vector is
        BROADBAND, which this model does not batch, so a 2-depth Source is
        one run per depth and comes back as a 2-slab stack, each slab its
        own depth's field."""
        from uacpy.core.results import ResultStack
        env, src, rcv = _stub_triple(2)
        model = _FieldStub()
        out = model.run(env, src, rcv, frequencies=[90.0, 100.0, 110.0])
        assert isinstance(out, ResultStack) and out.n_slabs == 2
        assert [c[0] for c in model.calls] == [RunMode.BROADBAND] * 2
        assert [c[1] for c in model.calls] == [[30.0], [60.0]]
        assert [float(sl.data.real.flat[0]) for sl in out.slabs] == [30.0, 60.0]
        assert all(sl.run_mode is RunMode.BROADBAND for sl in out.slabs)

    def test_run_mode_none_with_frequencies_applies_the_weight(self):
        """The other half of RA-CONTRACT-1: a weighted one-depth Source on
        the implicit BROADBAND route is scaled by its weight."""
        env, src, rcv = _stub_triple(1, weights=[2.0])
        out = _FieldStub().run(env, src, rcv, frequencies=[90.0, 110.0])
        assert float(out.data.real.flat[0]) == 60.0

    def test_the_t_start_warning_names_the_mode_that_runs(self):
        """RA-CONTRACT-21: ``run(frequencies=[...], t_start=...)`` on a
        model whose default then is BROADBAND warns naming BROADBAND."""
        env, src, rcv = _stub_triple(1)
        with pytest.warns(UserWarning,
                          match=r'run_mode=BROADBAND\): ignoring t_start='):
            _FieldStub().run(env, src, rcv, frequencies=[90.0, 110.0],
                             t_start=0.0)

    def test_the_engine_receives_the_resolved_mode_never_none(self):
        env, src, rcv = _stub_triple(1)
        model = _FieldStub()
        model.run(env, src, rcv)
        model.run(env, src, rcv, 'coherent_tl')
        assert [c[0] for c in model.calls] == [RunMode.COHERENT_TL] * 2
        assert all(isinstance(c[0], RunMode) for c in model.calls)

    def test_every_result_carries_the_settings_it_ran_with(self):
        env, src, rcv = _stub_triple(2)
        model = _FieldStub()
        want = model.run_settings(env, src, rcv, RunMode.COHERENT_TL)
        out = model.run(env, src, rcv, RunMode.COHERENT_TL)
        for slab in out.slabs:
            assert slab._run_settings == want
        assert want.depth_loop == 'per_depth'
        assert want.frequencies.tolist() == [100.0]

    def test_an_engine_grid_is_recorded_with_a_note(self):
        """An engine whose result is computed on a grid of its own: the
        stamped settings carry what the result was computed on, and say
        so."""
        class _OwnGrid(_FieldStub):
            def _to_result(self, inputs, deck, raw):
                out = super()._to_result(inputs, deck, raw)
                out.frequencies = np.array([99.0])
                return out
        env, src, rcv = _stub_triple(1)
        out = _OwnGrid().run(env, src, rcv)
        s = out._run_settings
        assert s.frequencies.tolist() == [99.0]
        assert s.notes and '99 Hz' in s.notes[0] and '100 Hz' in s.notes[0]

    def test_run_settings_launches_nothing_and_writes_nothing(self, tmp_path):
        env, src, rcv = _stub_triple(2)
        model = _FieldStub(work_dir=tmp_path / 'wd')
        a = model.run_settings(env, src, rcv, frequencies=[90.0, 110.0])
        b = model.run_settings(env, src, rcv, frequencies=[90.0, 110.0])
        assert a == b and model.calls == []
        assert not (tmp_path / 'wd').exists()

    @pytest.mark.parametrize('call', ['run', 'run_settings',
                                      'validate_inputs'])
    @pytest.mark.parametrize('kw,exc,match', [
        (dict(run_mode=RunMode.RAYS), 'UnsupportedFeatureError', 'RAYS'),
        (dict(run_mode=RunMode.COHERENT_TL, frequencies=[90.0, 110.0]),
         'ConfigurationError', 'frequencies='),
        (dict(t_start=float('nan')), 'ConfigurationError', 't_start'),
        (dict(output_duration=-1.0), 'ConfigurationError', 'output_duration'),
    ])
    def test_the_three_entry_points_refuse_the_same_calls(self, call, kw,
                                                          exc, match):
        """RA-CONTRACT-18 / ARCH-9: ``validate_inputs`` IS the checking
        stage of ``run``, keywords included, and ``run_settings`` runs it
        too, so all three raise the same exception."""
        import uacpy
        env, src, rcv = _stub_triple(1)
        model = _FieldStub()
        kw = dict(kw)
        run_mode = kw.pop('run_mode', None)
        with pytest.raises(getattr(uacpy, exc), match=match):
            getattr(model, call)(env, src, rcv, run_mode, **kw)
        assert model.calls == []

    def test_validate_inputs_refuses_what_run_refuses_for_a_missing_env(self):
        """RA-CONTRACT-18: the carrier types are part of the one checking
        stage, so ``validate_inputs(None, ...)`` refuses as ``run`` does
        instead of passing a model that reads no geometry."""
        _, src, rcv = _stub_triple(1)
        with pytest.raises(ConfigurationError, match='takes \\(env'):
            _FieldStub().validate_inputs(None, src, rcv)


# ── the staged path: the same stages through the engine's hooks ────────


@dataclass(frozen=True, eq=False)
class _SeafloorSettings(EngineSettings):
    """What ``_StagedFieldStub`` resolves: the seafloor depth (m) of the
    environment it runs on."""
    seafloor_m: float


_PRESSURE = OutputSpec('Field', kind='pressure', unit='Pa',
                       phase_reference='travelling_wave')


class _StagedFieldStub(PropagationModel):
    """``_FieldStub`` on the stage hooks, launching no binary: the deck
    holds the source depth, the "launch" copies it to an output file, and
    the field's value is that depth. Range-independent, so a sloping seafloor
    is projected to one depth. Every hook records what it was handed; the
    settings refuse a receiver beyond 5 km."""
    spec = ModelSpec(modes=(RunMode.COHERENT_TL, RunMode.BROADBAND))
    provenance_id = 'acoustics_toolbox'
    outputs = {RunMode.COHERENT_TL: _PRESSURE, RunMode.BROADBAND: _PRESSURE}

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = []

    def _resolve_engine_settings(self, env, source, receiver, settings, *,
                                 given_env, request=None):
        self.calls.append(('settings', env, given_env))
        if receiver.range_max > 5000.0:
            raise ConfigurationError('_StagedFieldStub: its grid stops at '
                                     '5 km')
        return _SeafloorSettings(seafloor_m=float(env.depth))

    def _write_input(self, inputs):
        deck = inputs.work_dir / 'stub.env'
        deck.write_text(repr(float(inputs.source.depths[0])))
        self.calls.append(('write', inputs.env, inputs.settings))
        return deck

    def _launch(self, inputs, deck):
        (inputs.work_dir / 'stub.out').write_text(deck.read_text())
        self.calls.append(('launch', deck.name))

    def _read_output(self, inputs, deck):
        return float((inputs.work_dir / 'stub.out').read_text())

    def _to_result(self, inputs, deck, raw):
        rcv = inputs.receiver
        return Field(
            data=np.full((rcv.depths.size, rcv.ranges.size), raw,
                         dtype=complex),
            coords={'depth': rcv.depths, 'range': rcv.ranges},
            **self._result_kwargs(
                inputs.source, frequencies=inputs.settings.frequencies,
                phase_reference=inputs.settings.output.phase_reference))


class _TwoLaunchStub(_StagedFieldStub):
    """``_StagedFieldStub`` made of two launches: launch 1's deck is launch
    0's output plus one, read from ``inputs.earlier``, so the result holds
    the source depth plus one. Records each launch's work directory, the
    outputs it saw and what ``_prepare_launches`` built."""

    def _n_launches(self, settings):
        return 2

    def _prepare_launches(self, env, settings):
        prepared = object()
        self.calls.append(('prepared', prepared))
        return prepared

    def _write_input(self, inputs):
        deck = inputs.work_dir / f'stub{inputs.launch}.env'
        value = (float(inputs.source.depths[0]) if inputs.launch == 0
                 else inputs.earlier[0] + 1.0)
        deck.write_text(repr(value))
        self.calls.append(('write', inputs.launch, inputs.work_dir,
                           inputs.earlier, inputs.prepared))
        return deck

    def _to_result(self, inputs, deck, raw):
        self.calls.append(('result', inputs.launch, deck, raw))
        return super()._to_result(inputs, deck[-1], raw[-1])


class TestTheLaunchesOfOneCall:
    """``_n_launches`` launches share one work directory; launch ``i``
    reads the outputs of the launches before it as ``inputs.earlier``, and
    what ``_prepare_launches`` built as ``inputs.prepared``; ``_to_result``
    gets every deck and output. One launch hands ``_to_result`` its own."""

    def test_launch_1_reads_launch_0_in_the_same_directory(self):
        env, src, rcv = _stub_triple(1)
        model = _TwoLaunchStub()
        field = model.run(env, src, rcv)
        writes = [c for c in model.calls if c[0] == 'write']
        (_, prepared), = [c for c in model.calls if c[0] == 'prepared']
        assert [w[1] for w in writes] == [0, 1]
        assert writes[0][2] == writes[1][2]
        assert writes[0][3] == () and writes[1][3] == (30.0,)
        assert writes[0][4] is prepared and writes[1][4] is prepared
        (_, launch, decks, raws), = [c for c in model.calls
                                     if c[0] == 'result']
        assert launch == 0 and raws == [30.0, 31.0]
        assert [d.name for d in decks] == ['stub0.env', 'stub1.env']
        assert np.all(np.asarray(field.data) == 31.0)

    def test_one_launch_hands_its_own_output(self):
        env, src, rcv = _stub_triple(1)
        model = _StagedFieldStub()
        model.run(env, src, rcv)
        assert [c[0] for c in model.calls] == ['settings', 'write', 'launch']


def _sloping_triple(n_depths=1, weights=None):
    env, src, rcv = _stub_triple(n_depths, weights=weights)
    env = Environment(bathymetry=np.array([[0.0, 100.0], [2000.0, 120.0]]),
                      ssp=1500.0)
    return env, src, rcv


class TestTheStagedPath:
    """A :class:`PropagationModel` runs the protocol through its hooks: the
    environment is projected once, before the checks; the settings are
    resolved once and every hook reads them; the depth loop, the weights
    and the stamp are the base class's; the result is held to the declared
    output contract."""

    def test_the_hooks_run_on_the_projected_environment(self):
        env, src, rcv = _sloping_triple()
        model = _StagedFieldStub()
        with pytest.warns(UserWarning, match='bathymetry'):
            model.run(env, src, rcv)
        (_, seen, given), (_, written, _), _ = model.calls
        assert given is env and not seen.bathymetry.varies_with_range
        assert written is seen and seen.depth == 120.0

    def test_the_settings_carry_the_engine_part_the_waveguide_and_output(
            self):
        env, src, rcv = _sloping_triple()
        model = _StagedFieldStub()
        with pytest.warns(UserWarning, match='bathymetry'):
            settings = model.run_settings(env, src, rcv)
        # The stub's one knob is the base's collapse, recorded as given.
        assert settings.engine == _SeafloorSettings(
            seafloor_m=120.0, knobs={'collapse': None})
        assert settings.output == _PRESSURE
        assert (settings.waveguide.c_min, settings.waveguide.c_max) == (
            1500.0, 1600.0)

    def test_every_launch_writes_the_settings_run_resolved(self):
        env, src, rcv = _stub_triple(1)
        model = _StagedFieldStub()
        out = model.run(env, src, rcv)
        written = [c[2] for c in model.calls if c[0] == 'write']
        assert written == [out.run_settings]
        assert [c[0] for c in model.calls].count('settings') == 1

    def test_a_multi_depth_source_is_one_launch_per_depth(self):
        from uacpy.core.results import ResultStack
        env, src, rcv = _stub_triple(2)
        model = _StagedFieldStub()
        out = model.run(env, src, rcv)
        assert isinstance(out, ResultStack) and out.n_slabs == 2
        assert [float(sl.data.real.flat[0]) for sl in out.slabs] == [30.0,
                                                                      60.0]
        assert [c[0] for c in model.calls].count('launch') == 2

    def test_a_weighted_one_depth_source_is_scaled_once(self):
        env, src, rcv = _stub_triple(1, weights=[2.0])
        out = _StagedFieldStub().run(env, src, rcv)
        assert float(out.data.real.flat[0]) == 60.0

    @pytest.mark.parametrize('call', ['run', 'run_settings',
                                      'validate_inputs'])
    def test_a_settings_refusal_comes_from_every_entry_point(self, call):
        env, src, _ = _stub_triple(1)
        rcv = Receiver(depths=[20.0], ranges=[9000.0])
        model = _StagedFieldStub()
        with pytest.raises(ConfigurationError, match='stops at 5 km'):
            getattr(model, call)(env, src, rcv)
        assert 'launch' not in [c[0] for c in model.calls]

    def test_run_settings_launches_nothing_and_writes_nothing(self,
                                                              tmp_path):
        env, src, rcv = _stub_triple(2)
        model = _StagedFieldStub(work_dir=tmp_path / 'wd')
        assert model.run_settings(env, src, rcv) == model.run_settings(
            env, src, rcv)
        assert [c[0] for c in model.calls] == ['settings', 'settings']
        assert not (tmp_path / 'wd').exists()

    def test_a_result_that_breaks_the_output_contract_is_refused(self):
        class _WrongUnit(_StagedFieldStub):
            outputs = {RunMode.COHERENT_TL: OutputSpec(
                'Field', kind='pressure', unit='dB',
                phase_reference='travelling_wave')}
        from uacpy.core.exceptions import OutputContractError
        with pytest.raises(OutputContractError,
                           match="unit 'Pa', declared 'dB'"):
            _WrongUnit().run(*_stub_triple(1))

    @pytest.mark.parametrize('coherent', [True, False, None])
    def test_stage_6_stamps_the_declared_coherence_on_a_field(self,
                                                              coherent):
        """ARCH-4: an engine's Field carries the coherence its mode
        declares (``outputs[mode].coherent``), assigned once by stage 6 —
        on every slab of a stack — not derived from the payload."""
        class _Declared(_StagedFieldStub):
            outputs = {RunMode.COHERENT_TL: OutputSpec(
                'Field', kind='pressure', unit='Pa',
                phase_reference='travelling_wave', coherent=coherent)}
        result = _Declared().run(*_stub_triple(2))
        for slab in result.slabs:
            assert 'coherent' not in slab.metadata
            assert slab.coherent is coherent

    def test_an_engine_without_spec_is_refused_at_definition(self):
        """The stage hooks make a class concrete, so it must declare the
        ``spec`` and ``provenance_id`` a model a user can hold needs."""
        with pytest.raises(TypeError, match='declares no spec'):
            class _NoSpec(PropagationModel):
                provenance_id = 'acoustics_toolbox'

                def _write_input(self, inputs):
                    return inputs.work_dir

                def _launch(self, inputs, deck):
                    pass

                def _read_output(self, inputs, deck):
                    return None

                def _to_result(self, inputs, deck, raw):
                    return raw


class _StagedTimeSeriesStub(_StagedFieldStub):
    """``_StagedFieldStub`` declaring ``TIME_SERIES`` too, with an engine
    notice: the announce hook records every call it gets."""
    spec = ModelSpec(modes=(RunMode.COHERENT_TL, RunMode.BROADBAND,
                            RunMode.TIME_SERIES))
    outputs = {RunMode.COHERENT_TL: _PRESSURE, RunMode.BROADBAND: _PRESSURE,
               RunMode.TIME_SERIES: _PRESSURE}

    def _announce_engine_settings(self, env, source, receiver, settings):
        self.calls.append(('announce', settings.engine))


def _pulse():
    return dict(source_waveform=np.hanning(40), sample_rate=400.0)


def _said(caught, text):
    return [str(w.message) for w in caught if text in str(w.message)]


class TestTheStagedTimeSeriesStage:
    """On a :class:`PropagationModel` that declares ``TIME_SERIES``, the base owns
    the time-series keywords once per call: the pulse refusal and the
    dropped-keyword warning in stage 2 (before the projection), the
    auto-derived grid and the 1 Hz floor notices in stage 3 (``run`` and
    ``run_settings``, not ``validate_inputs``), and the engine's own notices
    through ``_announce_engine_settings`` on the same terms."""

    def test_a_missing_pulse_is_refused_alike_by_every_entry_point(self):
        env, src, rcv = _stub_triple(1)
        model = _StagedTimeSeriesStub()
        outcomes = []
        for call in ('validate_inputs', 'run_settings', 'run'):
            with pytest.raises(
                    ConfigurationError,
                    match='requires source_waveform and sample_rate') as exc:
                getattr(model, call)(env, src, rcv, RunMode.TIME_SERIES)
            outcomes.append(str(exc.value))
        assert len(set(outcomes)) == 1
        assert 'requires source_waveform and sample_rate' in outcomes[0]
        assert not [c for c in model.calls if c[0] in ('write', 'launch')]

    def test_the_dropped_keywords_warn_once_per_call_not_per_depth(self):
        env, src, rcv = _stub_triple(2)
        with recorded_warnings() as caught:
            _StagedTimeSeriesStub().run(env, src, rcv, RunMode.BROADBAND,
                                        **_pulse())
        said = _said(caught, 'ignoring source_waveform=')
        assert len(said) == 1 and 'BROADBAND returns' in said[0], said

    def test_a_single_frequency_mode_drops_them_with_the_general_reason(
            self):
        env, src, rcv = _stub_triple(1)
        with recorded_warnings() as caught:
            _StagedTimeSeriesStub().run_settings(env, src, rcv, **_pulse())
        said = _said(caught, 'ignoring source_waveform=')
        assert len(said) == 1 and 'BROADBAND/TIME_SERIES only' in said[0]

    def test_the_derived_grid_is_announced_by_run_settings_only(self):
        env, src, rcv = _stub_triple(1)
        model = _StagedTimeSeriesStub()
        for call, expected in (('run_settings', 1), ('validate_inputs', 0)):
            with recorded_warnings() as caught:
                getattr(model, call)(env, src, rcv, RunMode.TIME_SERIES,
                                     **_pulse())
            assert len(_said(caught, 'auto-derived')) == expected, call

    def test_the_derived_grid_is_announced_once_per_call_not_per_depth(
            self):
        env, src, rcv = _stub_triple(2)
        with recorded_warnings() as caught:
            _StagedTimeSeriesStub().run(env, src, rcv, RunMode.TIME_SERIES,
                                        **_pulse())
        assert len(_said(caught, 'auto-derived')) == 1

    def test_the_floored_band_is_announced_by_run_settings_only(self):
        env, src, rcv = _stub_triple(1, frequencies=1.0)
        model = _StagedTimeSeriesStub()
        for call, expected in (('run_settings', 1), ('validate_inputs', 0)):
            with recorded_warnings() as caught:
                getattr(model, call)(env, src, rcv, RunMode.BROADBAND)
            assert len(_said(caught, 'floored at 1 Hz')) == expected, call

    def test_the_engine_announces_through_run_settings_only(self):
        env, src, rcv = _stub_triple(1)
        model = _StagedTimeSeriesStub()
        settings = model.run_settings(env, src, rcv)
        model.validate_inputs(env, src, rcv)
        announced = [c for c in model.calls if c[0] == 'announce']
        assert announced == [('announce', settings.engine)]

    def test_a_model_without_time_series_keeps_its_keyword_rule(self):
        """Bounce-like: no TIME_SERIES, so the waveform keywords are refused
        by the one keyword rule, not dropped here."""
        env, src, rcv = _stub_triple(1)
        with pytest.raises(
                UnsupportedFeatureError,
                match=r'run parameter\(s\): sample_rate, source_waveform'):
            _StagedFieldStub().run_settings(env, src, rcv, **_pulse())


class TestOneKeywordRule:
    """RA-CONTRACT-3: a keyword no mode of the model reads raises
    ``UnsupportedFeatureError`` (``t_start`` included); one the model reads
    on another mode is warned about and dropped."""

    class _ReflectionOnly(PropagationModel):
        spec = ModelSpec(modes=(RunMode.REFLECTION,))
        provenance_id = 'acoustics_toolbox'
        _write_input = _launch = _read_output = _to_result = _never_launched

    def test_the_never_consumed_table_follows_the_modes(self):
        from uacpy.models.base import _RUN_KEYWORDS
        assert self._ReflectionOnly()._run_keywords_never_consumed() == \
            frozenset(_RUN_KEYWORDS)
        assert _FieldStub()._run_keywords_never_consumed() == frozenset()

    class _CoherentOnly(PropagationModel):
        """OAST-like: one single-frequency mode, no broadband path."""
        spec = ModelSpec(modes=(RunMode.COHERENT_TL,))
        provenance_id = 'acoustics_toolbox'
        _write_input = _launch = _read_output = _to_result = _never_launched

    def test_frequencies_on_a_model_with_no_broadband_mode_is_unsupported(
            self):
        """``frequencies=`` is a keyword no mode of this model reads, so it
        is refused as one at every entry point, not pointed at BROADBAND and
        TIME_SERIES, which the model does not have."""
        from uacpy.core.exceptions import UnsupportedFeatureError
        env, src, rcv = _stub_triple(1)
        model = self._CoherentOnly()
        for call in ('run', 'run_settings', 'validate_inputs'):
            with pytest.raises(UnsupportedFeatureError,
                               match=r'run parameter\(s\): frequencies$'):
                getattr(model, call)(env, src, rcv, frequencies=[90.0, 110.0])

    def test_t_start_on_a_model_without_time_series_raises(self):
        from uacpy.core.exceptions import UnsupportedFeatureError
        env, src, rcv = _stub_triple(1)
        with pytest.raises(UnsupportedFeatureError,
                           match=r'run parameter\(s\): t_start$'):
            self._ReflectionOnly().run(env, src, rcv, t_start=0.1)

    def test_every_never_consumed_keyword_is_named_in_one_refusal(self):
        from uacpy.core.exceptions import UnsupportedFeatureError
        env, src, rcv = _stub_triple(1)
        with pytest.raises(UnsupportedFeatureError,
                           match='does not support: run parameter') as info:
            self._ReflectionOnly().validate_inputs(
                env, src, rcv, source_waveform=np.ones(4), sample_rate=10.0,
                output_duration=1.0, t_start=0.1)
        assert str(info.value).splitlines()[0].endswith(
            'output_duration, sample_rate, source_waveform, t_start')

    def test_a_refused_call_emits_no_warning_first(self):
        """ARCH-9: every refusal of a call comes before any warning about
        how it would have run, so a refused weighted call does not also
        warn that its weights would have been dropped."""
        from uacpy.core.exceptions import UnsupportedFeatureError
        env, src, rcv = _stub_triple(1, weights=[2.0])
        with recorded_warnings() as caught:
            with pytest.raises(UnsupportedFeatureError,
                               match=r'run parameter\(s\): frequencies'):
                self._ReflectionOnly().run(env, src, rcv,
                                           frequencies=[100.0])
        assert not caught, [str(w.message) for w in caught]

    def test_t_start_on_another_mode_of_a_consumer_warns_and_drops(self):
        env, src, rcv = _stub_triple(1)
        model = _FieldStub()
        with pytest.warns(UserWarning, match='ignoring t_start='):
            model.run(env, src, rcv, RunMode.COHERENT_TL, t_start=0.2)


class TestModeRefusalHook:
    def test_an_engine_can_refuse_a_mode_in_its_own_words(self):
        """``_refuse_run_mode`` runs before the generic refusal; SPARC uses
        it to point COHERENT_TL at the models that compute CW TL, and every
        entry point raises it."""
        from uacpy.core.exceptions import UnsupportedFeatureError

        class _Named(_FieldStub):
            def _refuse_run_mode(self, run_mode):
                if run_mode == RunMode.RAYS:
                    raise UnsupportedFeatureError('Named', 'RAYS, by name')
        env, src, rcv = _stub_triple(1)
        for call in ('run', 'run_settings', 'validate_inputs'):
            with pytest.raises(UnsupportedFeatureError, match='by name'):
                getattr(_Named(), call)(env, src, rcv, RunMode.RAYS)


@pytest.mark.parametrize('cls,cites', [
    (Bellhop, 'The BELLHOP Manual'),
    (Kraken, 'The KRAKEN Normal Mode Program'),
    (SPARC, 'time-marched fast-field program'),
    (Scooter, 'Computational Ocean Acoustics'),
    (Bounce, 'BOUNCE'),
])
def test_each_acoustics_toolbox_engine_cites_its_own_reference(cls, cites):
    """RA-CONTRACT-10: the five engines share one licence entry, which used
    to hand every one of them the KRAKEN manual. ``citation`` is the
    engine's own reference; the licence facts stay shared."""
    from uacpy.models.provenance import MODEL_PROVENANCE
    entry = MODEL_PROVENANCE['acoustics_toolbox']
    assert cites in entry.citation_for(cls.__name__)
    if cls is not Kraken:
        assert 'KRAKEN' not in entry.citation_for(cls.__name__)
    assert entry.citation_for('SomeOtherEngine') == entry.citation
    if cls is Bellhop:
        model = cls.__new__(cls)
        model.model_name = 'Bellhop'
        model._resolved_backend = 'fortran'
        assert cites in model.citation


def test_the_shared_multi_frequency_refusal_names_the_broadband_modes():
    """``_check_carriers`` refuses a multi-frequency Source on a
    single-frequency mode with ``_multi_frequency_refusal``; the base's
    message is the one every engine without an override raises (OAST names
    OASP instead)."""
    from types import SimpleNamespace
    from uacpy import Source
    src = Source(depths=50.0, frequencies=[90.0, 110.0])
    exc = PropagationModel._multi_frequency_refusal(
        SimpleNamespace(model_name='Kraken'), RunMode.COHERENT_TL, src)
    assert isinstance(exc, ConfigurationError)
    assert str(exc) == (
        f"Kraken.run(run_mode=COHERENT_TL) takes a single source frequency; "
        f"got 2: [90, 110] Hz. For broadband H(f) use "
        f"RunMode.BROADBAND, and for time-domain p(t) use "
        f"RunMode.TIME_SERIES.")


def test_a_refusal_lists_six_frequencies_and_elides_the_middle_of_more():
    from uacpy.models._band import frequencies_text
    assert frequencies_text([100.0, 156.66666, 200.0, 250.0, 300.0, 400.0]) \
        == "[100, 156.7, 200, 250, 300, 400] Hz"
    assert frequencies_text(np.linspace(150.0, 250.0, 7)) == \
        "[150, 166.7, 183.3, …, 216.7, 233.3, 250] Hz"


@pytest.mark.requires_binary
class TestTheBandIsResolvedOnceWithItsNotice:
    """Stage 3 resolves the frequency grid once, as
    :class:`~uacpy.models._band.BandResolution`: the run records its
    frequencies and gives its notice, word for word, on the modes the
    engine announces — never from ``validate_inputs``, and never on a mode
    the engine does not announce (Bellhop's TIME_SERIES labels its trace
    with the grid)."""

    @staticmethod
    def _pulse():
        fs = 4000.0
        t = np.arange(400) / fs
        return np.hanning(t.size) * np.sin(2 * np.pi * 100.0 * t), fs

    def _said(self, call, *args, **kwargs):
        with recorded_warnings() as caught:
            out = call(*args, **kwargs)
        return out, [str(w.message) for w in caught]

    def test_an_announced_band_gives_its_notice_once(self):
        from uacpy.models import Scooter
        from uacpy.models._band import time_series_band
        pulse, fs = self._pulse()
        model = Scooter(verbose=False)
        env, src, rcv = Environment(bathymetry=100.0, ssp=1500.0), \
            Source(depths=50.0, frequencies=100.0), \
            Receiver(depths=[50.0], ranges=[1000.0])
        expected = time_series_band(pulse, fs, model_name='Scooter')
        settings, said = self._said(
            model.run_settings, env, src, rcv,
            run_mode=RunMode.TIME_SERIES, source_waveform=pulse,
            sample_rate=fs)
        assert said.count(expected.notice) == 1, said
        np.testing.assert_array_equal(settings.frequencies,
                                      expected.frequencies)
        _, quiet = self._said(
            model.validate_inputs, env, src, rcv,
            run_mode=RunMode.TIME_SERIES, source_waveform=pulse,
            sample_rate=fs)
        assert expected.notice not in quiet, quiet

    def test_a_mode_the_engine_does_not_announce_says_nothing(self):
        pulse, fs = self._pulse()
        env, src, rcv = Environment(bathymetry=100.0, ssp=1500.0), \
            Source(depths=50.0, frequencies=100.0), \
            Receiver(depths=[50.0], ranges=[1000.0])
        _, said = self._said(
            Bellhop(verbose=False).run_settings, env, src, rcv,
            run_mode=RunMode.TIME_SERIES, source_waveform=pulse,
            sample_rate=fs)
        assert not [m for m in said if 'auto-derived' in m], said
