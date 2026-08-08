"""
Tests for core UACPY classes: Environment, Source, Receiver, Result
"""

import pytest
import numpy as np
import uacpy
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Field


class TestEnvironment:
    """Tests for Environment class."""

    def test_create_simple_environment(self, simple_env):
        """Test creating a simple isovelocity environment."""
        assert simple_env.name == "Test Environment"
        assert simple_env.depth == 100.0
        assert float(simple_env.ssp.data[0, 0]) == 1500.0
        assert simple_env.ssp.shape == 'isovelocity'
        assert not simple_env.is_range_dependent

    def test_create_parabolic_ssp_environment(self, parabolic_ssp_env):
        """Construction of a 100-m parabolic-SSP env."""
        assert parabolic_ssp_env.name == "Parabolic SSP"
        assert parabolic_ssp_env.depth == 100.0
        assert parabolic_ssp_env.ssp.n_depths == 21
        assert parabolic_ssp_env.ssp.shape == 'measured'

    def test_create_munk_environment(self, munk_env):
        """Construction of a deep-water Munk env using from_munk()."""
        assert munk_env.name == "Munk Profile"
        assert munk_env.depth == 5000.0
        assert munk_env.ssp.shape == 'munk'

    def test_range_dependent_environment(self, range_dependent_env):
        """Test range-dependent environment."""
        assert range_dependent_env.is_range_dependent
        assert range_dependent_env.bathymetry.n_ranges == 11
        assert range_dependent_env.bathymetry.depths[0] == 80.0
        assert range_dependent_env.bathymetry.depths[-1] == 120.0

    def test_max_range(self, simple_env, range_dependent_env):
        """max_range is the range extent (0 when range-independent), symmetric
        with depth and matching the bathymetry range axis."""
        assert simple_env.max_range == 0.0
        assert range_dependent_env.max_range == pytest.approx(
            float(range_dependent_env.bathymetry.ranges.max()))

    def test_ssp_pairs_shape(self, simple_env, parabolic_ssp_env):
        """SSP pairs view always has shape (N, 2)."""
        assert simple_env.ssp.to_pairs().shape[1] == 2
        assert parabolic_ssp_env.ssp.to_pairs().shape[1] == 2

    def test_get_representative_depth(self, range_dependent_env):
        """Test getting representative depth for range-dependent environment."""
        median_depth = range_dependent_env.get_representative_depth('median')
        assert 80 <= median_depth <= 120

    def test_invalid_depth(self):
        """Test that negative depth raises error."""
        with pytest.raises(ConfigurationError):
            uacpy.Environment(name="Test", bathymetry=-10, ssp=1500)

    def test_bathymetry_rejects_negative_range(self):
        """Bathymetry ranges are measured from the source; they cannot
        be negative."""
        with pytest.raises(ConfigurationError, match="ranges must be non-negative"):
            uacpy.Environment(
                name="Test",
                bathymetry=[[-100.0, 80.0], [5000.0, 90.0]],
                ssp=1500,
            )


class TestBiologicalLayerValidation:
    """:class:`BiologicalLayer` rejects impossible inputs at construction,
    matching the validation pattern on :class:`SedimentLayer`."""

    def test_valid_biological_layer(self):
        from uacpy.core.absorption import BiologicalLayer
        layer = BiologicalLayer(
            z_top_m=10.0, z_bottom_m=50.0, f0_hz=200.0, Q=20.0, a0=0.5,
        )
        assert layer.f0_hz == 200.0

    @pytest.mark.parametrize("kwargs,match", [
        (dict(z_top_m=50.0, z_bottom_m=10.0, f0_hz=200.0, Q=20.0, a0=0.5),
         "z_bottom_m"),
        (dict(z_top_m=10.0, z_bottom_m=10.0, f0_hz=200.0, Q=20.0, a0=0.5),
         "z_bottom_m"),
        (dict(z_top_m=10.0, z_bottom_m=50.0, f0_hz=-1.0, Q=20.0, a0=0.5),
         "f0_hz"),
        (dict(z_top_m=10.0, z_bottom_m=50.0, f0_hz=200.0, Q=0.0, a0=0.5),
         "Q"),
        (dict(z_top_m=10.0, z_bottom_m=50.0, f0_hz=200.0, Q=20.0, a0=-0.1),
         "a0"),
    ])
    def test_biological_layer_rejects_invalid(self, kwargs, match):
        from uacpy.core.absorption import BiologicalLayer
        with pytest.raises(ConfigurationError, match=match):
            BiologicalLayer(**kwargs)


class TestSource:
    """Tests for Source class."""

    def test_create_source(self, source):
        """Test creating a source."""
        assert source.depths[0] == 50.0
        assert source.frequencies[0] == 100.0

    def test_source_array_conversion(self):
        """Test that single values are converted to arrays."""
        source = uacpy.Source(depths=30.0, frequencies=200.0)
        assert isinstance(source.depths, np.ndarray)
        assert isinstance(source.frequencies, np.ndarray)
        assert len(source.depths) == 1
        assert len(source.frequencies) == 1

    def test_multiple_sources(self):
        """Test multiple source depths."""
        source = uacpy.Source(depths=[10.0, 20.0, 30.0], frequencies=100.0)
        assert len(source.depths) == 3
        assert np.allclose(source.depths, [10, 20, 30])

    def test_multiple_frequencies(self):
        """Test multiple frequencies."""
        source = uacpy.Source(depths=50.0, frequencies=[50.0, 100.0, 200.0])
        assert len(source.frequencies) == 3

    def test_source_depths_must_be_strictly_increasing(self):
        """Multi-element source depths must be sorted (matches Receiver), so
        output rows indexed by source depth stay unambiguous across models."""
        with pytest.raises(ConfigurationError, match="strictly increasing"):
            uacpy.Source(depths=[30.0, 10.0, 20.0], frequencies=100.0)


@pytest.mark.parametrize("ctor,kwargs", [
    # Source / Receiver reject NaN or inf in any
    # ``depths``/``frequencies``/``ranges`` array.
    (uacpy.Source, dict(depths=[10, np.nan], frequencies=100)),
    (uacpy.Source, dict(depths=10, frequencies=[100, np.nan])),
    (uacpy.Receiver, dict(depths=[10, 20], ranges=[100, np.nan])),
    (uacpy.Receiver, dict(depths=[np.nan], ranges=[100])),
    (uacpy.Source, dict(depths=[10, np.inf], frequencies=100)),
    (uacpy.Receiver, dict(depths=[10], ranges=[np.inf])),
])
def test_source_receiver_reject_non_finite(ctor, kwargs):
    """Source and Receiver reject NaN / inf at construction so
    non-finite values cannot leak into env-file writers."""
    with pytest.raises(ConfigurationError, match="finite"):
        ctor(**kwargs)


class TestReceiver:
    """Tests for Receiver class."""

    def test_create_receiver_grid(self, receiver_grid):
        """Test creating receiver grid."""
        assert len(receiver_grid.depths) == 9
        assert len(receiver_grid.ranges) == 11
        assert receiver_grid.receiver_type == 'grid'
        assert receiver_grid.depth_min == 10.0
        assert receiver_grid.depth_max == 90.0
        assert receiver_grid.range_min == 100.0
        assert receiver_grid.range_max == 5000.0

    def test_small_receiver_grid(self, receiver_small):
        """Test small receiver grid."""
        assert len(receiver_small.depths) == 3
        assert len(receiver_small.ranges) == 3

    def test_receiver_line_array(self):
        """Test line array receiver."""
        receiver = uacpy.Receiver(
            depths=[50.0],
            ranges=np.linspace(1000, 10000, 100)
        )
        assert len(receiver.depths) == 1
        assert len(receiver.ranges) == 100

    def test_receiver_type_rejects_unknown(self):
        """Unknown ``receiver_type`` strings raise
        :class:`ConfigurationError`, mirroring the validation on
        ``Source.source_type``."""
        with pytest.raises(ConfigurationError, match="receiver_type"):
            uacpy.Receiver(depths=50, ranges=1000, receiver_type='gird')

    def test_receiver_type_accepts_grid_rejects_line(self):
        rx = uacpy.Receiver(depths=50, ranges=1000, receiver_type='grid')
        assert rx.receiver_type == 'grid'
        with pytest.raises(ConfigurationError, match="receiver_type='line'"):
            uacpy.Receiver(depths=50, ranges=1000, receiver_type='line')


class TestField:
    """Tests for the unified :class:`~uacpy.Field` container."""

    @staticmethod
    def _tl_field(data, ranges, depths, **kw):
        return Field(
            data=data,
            coords={'depth': depths, 'range': ranges},
            model=kw.pop('model', 'Test'),
            frequencies=kw.pop('frequencies', 100.0),
            **kw,
        )

    def test_create_tl_field(self):
        from uacpy.core.results import Field
        data = np.random.rand(10, 20) * 50 + 40  # dB
        ranges = np.linspace(100, 5000, 20)
        depths = np.linspace(10, 90, 10)
        field = self._tl_field(data, ranges, depths)
        assert isinstance(field, Field)
        assert field.shape == (10, 20)
        assert field.n_ranges == 20
        assert field.n_depths == 10
        assert not field.is_complex

    def test_to_dict_roundtrip_preserves_model_source(self):
        from uacpy.models.sources import model_source
        src = model_source('acoustics_toolbox')
        field = self._tl_field(
            np.zeros((2, 2)), np.array([100.0, 200.0]),
            np.array([10.0, 20.0]), model_source=src)
        rt = Field.from_dict(field.to_dict())
        assert rt.model_source is src

    def test_field_at_point(self):
        data = np.arange(100).reshape(10, 10).astype(float)
        ranges = np.linspace(0, 9000, 10)
        depths = np.linspace(0, 90, 10)
        field = self._tl_field(data, ranges, depths)
        value = float(field.at(range=4500, depth=45).tl)
        assert 44 <= value <= 55

    def test_field_at_range(self):
        data = np.arange(100).reshape(10, 10).astype(float)
        ranges = np.linspace(0, 9000, 10)
        depths = np.linspace(0, 90, 10)
        field = self._tl_field(data, ranges, depths)
        values = field.at(range=4500).tl
        assert len(values) == 10
        assert 50 <= values[5] <= 59

    def test_field_at_depth(self):
        data = np.arange(100).reshape(10, 10).astype(float)
        ranges = np.linspace(0, 9000, 10)
        depths = np.linspace(0, 90, 10)
        field = self._tl_field(data, ranges, depths)
        values = field.at(depth=45).tl
        assert len(values) == 10
        assert 40 <= values[5] <= 49

    def test_field_deepcopy(self):
        import copy as _copy
        data = np.random.rand(10, 20)
        ranges = np.linspace(100, 5000, 20)
        depths = np.linspace(10, 90, 10)
        field = self._tl_field(data, ranges, depths)
        field_copy = _copy.deepcopy(field)
        assert type(field_copy) is type(field)
        assert np.array_equal(field_copy.data, field.data)
        assert field_copy is not field
        assert field_copy.data is not field.data

    def test_field_repr(self):
        data = np.random.rand(10, 20)
        ranges = np.linspace(100, 5000, 20)
        depths = np.linspace(10, 90, 10)
        field = self._tl_field(data, ranges, depths)
        repr_str = repr(field)
        assert 'Field' in repr_str
        assert field.shape == (10, 20)

    def test_resample_to_is_keyword_only_and_depth_first(self):
        """The two axis vectors are interchangeable in type and only
        distinguishable by name, so a positional call that swapped them would
        silently resample onto a transposed, mostly-NaN grid instead of
        raising. Keyword-only makes that unrepresentable."""
        ranges = np.linspace(0.0, 1000.0, 5)
        depths = np.linspace(0.0, 100.0, 3)
        # value == depth, so a transposed result is obvious in the numbers.
        data = np.repeat(depths[:, None], ranges.size, axis=1)
        field = self._tl_field(data, ranges, depths)

        with pytest.raises(TypeError):
            field.resample_to(ranges, depths)

        out = field.resample_to(depths=[25.0, 75.0], ranges=[250.0, 750.0])
        assert list(out.coords) == ['depth', 'range']
        assert out.shape == (2, 2)
        np.testing.assert_allclose(out.data, [[25.0, 25.0], [75.0, 75.0]])


class TestFieldValueAccessorsAreWriteGuarded:
    """``Field`` copies on ingest, so no accessor may hand back a writable
    alias of ``data``: ``p = field.p; p *= k`` would otherwise corrupt the
    stored result. Real ``.tl`` is the common path (RAM / OAST /
    Bellhop-incoherent all return real dB)."""

    @staticmethod
    def _real():
        return Field(data=np.array([[60.0, 70.0], [80.0, 90.0]]),
                     coords={'depth': [0.0, 1.0], 'range': [0.0, 1.0]})

    @staticmethod
    def _complex():
        return Field(data=np.array([[1 + 1j, 2 + 0j], [0 + 3j, 4 - 1j]]),
                     coords={'depth': [0.0, 1.0], 'range': [0.0, 1.0]})

    def test_real_tl_is_read_only(self):
        f = self._real()
        tl = f.tl
        assert not tl.flags.writeable
        with pytest.raises(ValueError):
            tl[0, 0] = -999.0
        np.testing.assert_array_equal(f.data, [[60.0, 70.0], [80.0, 90.0]])

    def test_real_tl_still_reads_the_stored_dB_values(self):
        f = self._real()
        np.testing.assert_array_equal(np.asarray(f.tl), f.data)

    def test_scalar_tl_is_read_only_and_castable(self):
        f = self._real().at(depth=0.0, range=1.0)
        assert not f.tl.flags.writeable
        assert float(f.tl) == 70.0

    def test_complex_tl_is_a_fresh_array(self):
        f = self._complex()
        tl = f.tl
        assert not np.shares_memory(tl, f.data)
        tl[0, 0] = -999.0                       # derived array: safe to write
        assert f.data[0, 0] == 1 + 1j

    def test_p_is_read_only(self):
        f = self._complex()
        with pytest.raises(ValueError):
            f.p[0, 0] = 5.0
        assert f.data[0, 0] == 1 + 1j

    def test_magnitude_and_phase_are_fresh_arrays(self):
        f = self._complex()
        mag, ph = f.magnitude, f.phase
        assert not np.shares_memory(mag, f.data)
        assert not np.shares_memory(ph, f.data)
        mag[0, 0] = -1.0
        ph[0, 0] = -1.0
        assert f.data[0, 0] == 1 + 1j
        np.testing.assert_allclose(f.magnitude, np.abs(f.data))
        np.testing.assert_allclose(f.phase, np.angle(f.data))

    def test_to_dict_does_not_alias_the_field(self):
        f = self._real()
        d = f.to_dict()
        d['data'][1, 1] = -1.0
        d['coords']['depth'][0] = 99.0
        assert f.data[1, 1] == 90.0
        assert f.coords['depth'][0] == 0.0


class TestPublicReexports:
    """Public namespace contract."""

    def test_soundspeedprofile_at_top_level(self):
        from uacpy import SoundSpeedProfile
        assert SoundSpeedProfile is uacpy.core.environment.SoundSpeedProfile

    def test_environment_helpers_at_core(self):
        from uacpy.core import SoundSpeedProfile, generate_sea_surface
        assert SoundSpeedProfile is uacpy.core.environment.SoundSpeedProfile
        assert generate_sea_surface is uacpy.core.environment.generate_sea_surface

    def test_acoustic_signal_is_importable_submodule(self):
        import uacpy.acoustic_signal as sig
        assert sig is uacpy.acoustic_signal

    def test_signal_analysis_classes_reachable(self):
        sig = uacpy.acoustic_signal
        # Estimators/transforms are functions now; FRF remains a class.
        for name in ('ppsd', 'psd', 'FRF', 'sel', 'fk_transform', 'spectrogram'):
            assert hasattr(sig, name), f"uacpy.acoustic_signal.{name} not reachable"
        assert 'psd' in sig.__all__
        assert 'FRF' in sig.__all__
        assert 'sel' in sig.__all__
        assert 'fk_transform' in sig.__all__

    def test_metrics_is_importable_submodule(self):
        import uacpy.metrics as m
        assert m is uacpy.metrics
        assert hasattr(m, 'tl_rmse')
        assert hasattr(m, 'tl_max_error')
        assert hasattr(m, 'tl_bias')


class TestModesFieldType:
    """Single canonical ``field_type`` declaration on Modes."""

    def test_modes_field_type_value(self):
        from uacpy.core.results import Modes
        assert Modes.field_type == "modes"


class TestModesComputePhaseSpeeds:
    """``Modes.compute_phase_speeds`` requires a frequency context;
    without one it raises :class:`ValueError`."""

    def _build_modes(self, *, frequencies):
        from uacpy.core.results import Modes
        depths = np.linspace(0, 100, 11)
        # Two trivial modes; numbers don't matter.
        k = np.array([0.4 + 0.0j, 0.3 + 0.0j])
        phi = np.zeros((len(depths), 2))
        return Modes(
            k=k, phi=phi, depths=depths,
            model='Test', frequencies=frequencies,
        )

    def test_phase_speeds_raises_without_frequency(self):
        modes = self._build_modes(frequencies=None)
        with pytest.raises(ConfigurationError, match='requires frequencies'):
            modes.compute_phase_speeds()

    def test_phase_speeds_with_frequency_is_omega_over_k(self):
        modes = self._build_modes(frequencies=100.0)
        v_p = modes.compute_phase_speeds()
        omega = 2.0 * np.pi * 100.0
        expected = omega / np.array([0.4, 0.3])
        np.testing.assert_allclose(v_p, expected, rtol=1e-12)


class TestRaysMissDistanceUnits:
    """Ray polylines are in metres; ``Rays._miss_distance_to`` consumes
    them verbatim with no unit rescaling."""

    def test_short_polyline_in_metres_is_not_rescaled(self):
        from uacpy.core.results import Rays
        r_m = np.linspace(0.0, 10.0, 11)
        z_m = np.linspace(0.0, 5.0, 11)
        rays = Rays(
            rays=[{
                'r': r_m, 'z': z_m,
                'alpha': 0.0,
                'n_top_bounces': 0, 'n_bot_bounces': 0,
            }],
            is_eigen=False,
            receiver_depths=np.array([5.0]),
            receiver_ranges=np.array([10.0]),
            model='Test', frequencies=100.0,
        )
        # The polyline passes exactly through (r=10, z=5), so miss == 0
        # in metres. (No km->m rescale: that would blow the miss up to
        # ~10 km.)
        miss, _ = rays._miss_distance_to(rays.rays[0], 10.0, 5.0)
        assert miss == pytest.approx(0.0, abs=1e-12)


class TestArrivalsFlatListKeys:
    """``Arrivals.arrivals`` flat list carries writer-aligned bounce keys."""

    def test_arrivals_flat_list_uses_n_bounce_keys(self):
        from uacpy.core.results import Arrivals
        # Build a minimal payload with the canonical IO key naming.
        payload = [[[{
            "delays": np.array([0.1, 0.2]),
            "amplitudes": np.array([1.0, 0.5]),
            "phases": np.array([0.0, 0.1]),
            "n_top_bounces": np.array([0, 1], dtype=int),
            "n_bot_bounces": np.array([1, 2], dtype=int),
            "src_angles": np.array([0.0, 5.0]),
            "rcv_angles": np.array([0.0, -5.0]),
            "n_arrivals": 2,
        }]]]
        arr = Arrivals(
            by_receiver=payload,
            receiver_depths=np.array([50.0]),
            receiver_ranges=np.array([1000.0]),
            model='Test',
            frequencies=100.0,
        )
        table = arr.arrivals
        assert len(table) == 2
        for row in table:
            assert 'n_top_bounces' in row
            assert 'n_bot_bounces' in row
        assert table[0]['kind'] == 'bottom'
        assert table[1]['kind'] == 'both'
        assert table[0]['n_bot_bounces'] == 1
        assert table[1]['n_top_bounces'] == 1


class TestArrivalsFilterChain:
    """Arrivals exposes a Rays-style filter chain over a flat list of
    arrival events; no continuous-axis ``at(...)`` slicer."""

    def _arrivals(self):
        from uacpy.core.results import Arrivals
        # Two cells (1 src × 1 depth × 2 ranges) with a mix of bounce kinds.
        cell0 = {
            "delays": np.array([0.1, 0.2, 0.3]),
            "amplitudes": np.array([1.0, 0.5, 0.2]),
            "phases": np.array([0.0, 0.0, 0.0]),
            "n_top_bounces": np.array([0, 1, 1], dtype=int),
            "n_bot_bounces": np.array([0, 0, 2], dtype=int),
            "src_angles": np.array([0.0, 5.0, 10.0]),
            "rcv_angles": np.array([0.0, -5.0, -10.0]),
        }
        cell1 = {
            "delays": np.array([0.4]),
            "amplitudes": np.array([0.8]),
            "phases": np.array([0.0]),
            "n_top_bounces": np.array([0], dtype=int),
            "n_bot_bounces": np.array([1], dtype=int),
            "src_angles": np.array([2.0]),
            "rcv_angles": np.array([-2.0]),
        }
        return Arrivals(
            by_receiver=[[[cell0, cell1]]],
            receiver_depths=np.array([50.0]),
            receiver_ranges=np.array([1000.0, 2000.0]),
            model='Test', frequencies=100.0,
        )

    def test_flat_list_length(self):
        a = self._arrivals()
        assert len(a) == 4    # 3 from cell0 + 1 from cell1

    def test_phases_returns_radians_from_degree_store(self):
        # The .arr file stores phase in degrees (ArrMod.f90 writes RadDeg*Phase);
        # the public .phases accessor must return radians for exp(1j*phase).
        from uacpy.core.results import Arrivals
        cell = {
            "delays": np.array([0.1]), "amplitudes": np.array([1.0]),
            "phases": np.array([90.0]),   # degrees as read from the file
            "n_top_bounces": np.array([0], dtype=int),
            "n_bot_bounces": np.array([0], dtype=int),
            "src_angles": np.array([0.0]), "rcv_angles": np.array([0.0]),
        }
        a = Arrivals(by_receiver=[[[cell]]], receiver_depths=np.array([50.0]),
                     receiver_ranges=np.array([1000.0]),
                     model='Test', frequencies=100.0)
        assert a.phases[0] == pytest.approx(np.pi / 2)

    def test_filter_by_bounces_kind(self):
        a = self._arrivals()
        direct = a.filter_by_bounces(kind='direct')
        assert len(direct) == 1
        assert direct.arrivals[0]['delay'] == pytest.approx(0.1)
        bottom = a.filter_by_bounces(kind='bottom')
        assert len(bottom) == 1
        assert bottom.arrivals[0]['delay'] == pytest.approx(0.4)

    def test_filter_by_bounces_top_low_high(self):
        a = self._arrivals()
        # 0-1 surface bounces — keeps everything but the 'both' arrival
        # has top=1 too, so all four pass; instead use bot=(1, None).
        few_bot = a.filter_by_bounces(bot=(1, None))
        assert len(few_bot) == 2     # cell0 last + cell1 last
        exact_top = a.filter_by_bounces(top=1)
        assert len(exact_top) == 2   # cell0 idx 1 and 2

    def test_in_delay_window(self):
        a = self._arrivals()
        mid = a.in_delay_window(0.15, 0.35)
        assert len(mid) == 2
        assert all(0.15 <= x['delay'] <= 0.35 for x in mid.arrivals)

    def test_top_n_by_amplitude(self):
        a = self._arrivals()
        top2 = a.top_n_by_amplitude(2)
        assert len(top2) == 2
        amps = [x['amplitude'] for x in top2.arrivals]
        assert amps == sorted(amps, reverse=True)
        assert amps[0] == 1.0   # cell0 first

    def test_filter_chain_returns_arrivals(self):
        from uacpy.core.results import Arrivals
        a = self._arrivals()
        chained = a.filter(lambda x: x['n_bot_bounces'] >= 1).top_n_by_amplitude(1)
        assert isinstance(chained, Arrivals)
        assert len(chained) == 1


class TestTimeFieldChainAccessors:
    """A time-domain :class:`Field` carries ``coords={'depth', 'range',
    'time'}``. Slicing follows the same axis-drop rule as 2-D fields."""

    @staticmethod
    def _ts():
        rng = np.random.default_rng(0)
        data = rng.standard_normal((3, 4, 50))
        return Field(
            data=data,
            coords={
                'depth': np.linspace(10, 90, 3),
                'range': np.linspace(100, 1000, 4),
                'time': np.linspace(0, 0.49, 50),
            },
            model='Test', frequencies=100.0,
        )

    def test_at_partial_keeps_remaining_axes(self):
        ts = self._ts()
        sliced = ts.at(depth=50.0)
        assert list(sliced.coords) == ['range', 'time']
        assert sliced.data.shape == (4, 50)
        assert sliced.pinned['depth'] == 50.0

    def test_at_both_spatial_drops_to_trace(self):
        ts = self._ts()
        trace = ts.at(depth=50.0, range=500.0)
        assert list(trace.coords) == ['time']
        assert trace.data.shape == (50,)
        assert set(trace.pinned) == {'depth', 'range'}

    def test_max_records_all_axes_in_pinned(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((3, 4, 50))
        data[2, 1, 30] = 100.0
        ts = Field(
            data=data,
            coords={
                'depth': np.linspace(10, 90, 3),
                'range': np.linspace(100, 1000, 4),
                'time': np.linspace(0, 0.49, 50),
            },
            model='Test', frequencies=100.0,
        )
        m = ts.max()
        assert list(m.coords) == []
        assert float(m.data) == pytest.approx(100.0)
        assert m.pinned['depth'] == ts.coords['depth'][2]
        assert m.pinned['range'] == ts.coords['range'][1]
        assert m.pinned['time'] == ts.coords['time'][30]


class TestReflectionCoefficientChainAccessors:
    """``ReflectionCoefficient.at`` — label slicing of the angle and
    frequency axes; broadband-only kwargs raise on narrowband instances."""

    def _broadband_rc(self):
        from uacpy.core.results import ReflectionCoefficient
        theta = np.linspace(0, 90, 91)            # 91 angles
        freqs = np.array([50.0, 100.0, 200.0])    # 3 frequencies
        R = np.outer(np.cos(np.deg2rad(theta)), np.ones(3)) ** 2
        phi = np.zeros_like(R)
        return ReflectionCoefficient(
            theta=theta, R=R, phi=phi,
            frequencies=freqs, model='Test',
        )

    def test_at_frequency_returns_narrowband(self):
        from uacpy.core.results import ReflectionCoefficient
        rc = self._broadband_rc()
        sliced = rc.at(frequency=100.0)
        assert isinstance(sliced, ReflectionCoefficient)
        assert not sliced.is_broadband
        assert sliced.R.shape == (91,)

    def test_at_angle_keeps_broadband(self):
        from uacpy.core.results import ReflectionCoefficient
        rc = self._broadband_rc()
        sliced = rc.at(angle=45.0)
        assert isinstance(sliced, ReflectionCoefficient)
        assert sliced.theta.shape == (1,)
        assert sliced.R.shape == (1, 3)

    def test_at_both_collapses_to_single_value(self):
        rc = self._broadband_rc()
        sliced = rc.at(angle=45.0, frequency=100.0)
        assert sliced.theta.shape == (1,)
        assert sliced.R.shape == (1,)

    def test_at_frequency_on_narrowband_raises(self):
        from uacpy.core.results import ReflectionCoefficient
        rc = ReflectionCoefficient(
            theta=np.linspace(0, 90, 5),
            R=np.linspace(0, 1, 5),
            phi=np.zeros(5),
            model='Test', frequencies=100.0,
        )
        with pytest.raises(ConfigurationError, match="broadband"):
            rc.at(frequency=100.0)

    @pytest.mark.parametrize('slice_call', [
        lambda rc: rc.at(frequency=100.0),
        lambda rc: rc.at(angle=45.0),
        lambda rc: rc.isel(frequency=1),
        lambda rc: rc.isel(angle=10),
        lambda rc: rc.eval(frequency=120.0),
    ])
    def test_slicing_keeps_provenance(self, slice_call):
        """Every slice carries ``model_source`` (the plot credit) and
        ``phase_reference`` — the same invariant ``Field.id_kwargs``
        enforces."""
        from uacpy.models.sources import model_source
        src = model_source('acoustics_toolbox')
        rc = self._broadband_rc()
        rc.model_source = src
        rc.phase_reference = 'travelling_wave'
        sliced = slice_call(rc)
        assert sliced.model_source is src
        assert sliced.phase_reference == 'travelling_wave'

    def test_at_theta_is_synonym_for_angle(self):
        rc = self._broadband_rc()
        by_angle = rc.at(angle=45.0)
        by_theta = rc.at(theta=45.0)
        assert by_theta.theta.shape == (1,)
        assert np.array_equal(by_theta.theta, by_angle.theta)

    def test_at_unknown_axis_raises(self):
        # Generic Field.at-style form rejects axes it doesn't have.
        rc = self._broadband_rc()
        with pytest.raises(ConfigurationError, match="unknown axis"):
            rc.at(depth=5.0)

    def test_at_angle_and_theta_together_raises(self):
        rc = self._broadband_rc()
        with pytest.raises(ConfigurationError, match="synonyms"):
            rc.at(angle=45.0, theta=45.0)

    def test_isel_positional_angle(self):
        rc = self._broadband_rc()
        s = rc.isel(angle=2)
        assert s.theta.shape == (1,) and s.theta[0] == rc.theta[2]
        assert s.R.shape == (1, 3)

    def test_isel_oob_raises_indexerror(self):
        with pytest.raises(IndexError):
            self._broadband_rc().isel(angle=999)

    def test_eval_interpolates_off_grid_angle(self):
        rc = self._broadband_rc()              # 1° grid → 30.5° is off-grid
        s = rc.eval(angle=30.5)
        assert s.theta[0] == pytest.approx(30.5)
        expect = 0.5 * (np.cos(np.deg2rad(30)) ** 2
                        + np.cos(np.deg2rad(31)) ** 2)   # linear of cos^2
        assert s.R[0, 0] == pytest.approx(expect, abs=1e-6)

    def test_eval_method_cubic(self):
        rc = self._broadband_rc()
        s = rc.eval(angle=30.5, method='cubic')
        assert s.R[0, 0] == pytest.approx(np.cos(np.deg2rad(30.5)) ** 2, abs=1e-3)


class TestSoundSpeedProfileNearestVsInterp:
    """``SoundSpeedProfile.at(...)`` is **nearest** (never fabricates),
    ``.eval(...)`` is **linear**, ``.isel(...)`` is **positional** — the
    grid-library invariant shared with ``Field`` et al."""

    def _ssp(self):
        from uacpy.core.environment import SoundSpeedProfile
        return SoundSpeedProfile(
            depths=np.array([0.0, 100.0, 200.0]),
            data=np.array([[1500.0], [1490.0], [1480.0]])
        )

    def test_at_picks_nearest_depth(self):
        ssp = self._ssp()
        # depth=51 is closer to 100 than to 0 → returns the 100m sample
        sliced = ssp.at(depth=51.0)
        assert sliced.depths[0] == 100.0
        assert sliced.value == 1490.0

    def test_eval_interpolates_linear(self):
        ssp = self._ssp()
        # depth=50 is halfway between (0, 1500) and (100, 1490) → 1495
        sliced = ssp.eval(depth=50.0)
        assert sliced.depths[0] == 50.0
        assert sliced.value == pytest.approx(1495.0)

    def test_isel_positional_depth(self):
        ssp = self._ssp()
        sliced = ssp.isel(depth=1)
        assert sliced.depths[0] == 100.0 and sliced.value == 1490.0
        with pytest.raises(IndexError):
            ssp.isel(depth=9)


class TestSoundSpeedProfileDepthOnlySliceOf2D:
    """A depth-only slice of a range-dependent profile is ambiguous.

    Silently returning the r = 0 column is wrong physics on exactly the
    profiles the 2-D carrier exists for, so ``at`` / ``eval`` / ``isel``
    require the range to be pinned (or the range axis collapsed first).
    """

    def _ssp2d(self):
        from uacpy.core.environment import SoundSpeedProfile
        return SoundSpeedProfile.from_2d(
            depths=[0.0, 100.0], ranges=[0.0, 5000.0],
            matrix=[[1500.0, 1400.0], [1490.0, 1390.0]])

    @pytest.mark.parametrize('call', [
        lambda s: s.at(depth=50.0),
        lambda s: s.eval(depth=50.0),
        lambda s: s.isel(depth=0),
    ])
    def test_depth_only_slice_raises(self, call):
        with pytest.raises(ConfigurationError, match='range-dependent'):
            call(self._ssp2d())

    def test_pinning_the_range_works(self):
        s = self._ssp2d()
        assert s.at(depth=0.0, range=5000.0).value == pytest.approx(1400.0)
        assert s.eval(depth=0.0, range=2500.0).value == pytest.approx(1450.0)
        assert s.isel(depth=0, range=1).value == pytest.approx(1400.0)

    def test_collapsing_the_range_axis_works(self):
        assert self._ssp2d().collapse('mean').at(depth=0.0).value == \
            pytest.approx(1450.0)

    def test_range_only_and_1d_paths_unaffected(self):
        from uacpy.core.environment import SoundSpeedProfile
        assert self._ssp2d().eval(range=5000.0).data.shape == (2, 1)
        flat = SoundSpeedProfile.from_isovelocity(100.0, 1500.0)
        assert flat.at(depth=50.0).value == pytest.approx(1500.0)

    @pytest.mark.parametrize('call', [
        lambda s: s.eval(range=0.0, method='bogus'),
        lambda s: s.eval(depth=50.0, method='bogus'),
    ])
    def test_interp_method_validated_on_every_path(self, call):
        """The membership check runs on entry, so the range-independent
        shortcut cannot swallow a bad method name."""
        from uacpy.core.environment import SoundSpeedProfile
        flat = SoundSpeedProfile.from_isovelocity(100.0, 1500.0)
        with pytest.raises(ConfigurationError, match='interpolation method'):
            call(flat)


class TestSoundSpeedProfileExtendTo:
    """``SoundSpeedProfile.extend_to(z_max)`` is the canonical alignment
    hook used by every env writer. Must extend OR truncate so that
    ``ssp.depths[-1] == z_max`` exactly."""

    def _profile(self, depths, speeds):
        from uacpy.core.environment import SoundSpeedProfile
        return SoundSpeedProfile(
            depths=np.asarray(depths, dtype=float),
            data=np.asarray(speeds, dtype=float).reshape(-1, 1)
        )

    def test_noop_when_depth_max_equals_deepest(self):
        ssp = self._profile([0, 100, 200], [1500, 1490, 1485])
        assert ssp.extend_to(200.0) is ssp

    def test_extend_with_constant_extrapolation(self):
        out = self._profile([0, 100], [1500, 1490]).extend_to(300.0)
        assert out.depths[-1] == 300.0
        assert out.data[-1, 0] == 1490.0

    def test_truncate_with_linear_interpolation(self):
        out = self._profile([0, 100, 200], [1500, 1490, 1480]).extend_to(150.0)
        assert out.depths[-1] == 150.0
        assert out.data[-1, 0] == pytest.approx(1485.0)
        assert (out.depths <= 150.0).all()

    def test_truncate_then_extend_round_trip(self):
        ssp = self._profile([0, 100, 200], [1500, 1490, 1480])
        out = ssp.extend_to(150.0).extend_to(150.0)
        assert out.depths[-1] == 150.0
        assert out.data[-1, 0] == pytest.approx(1485.0)

    def test_noop_under_floating_point_drift(self):
        """``extend_to`` is a no-op when the requested depth matches the
        deepest sample to within a small relative tolerance — a 1-ulp
        drift (from e.g. a round trip through I/O) must not rewrite the
        bottom sample."""
        ssp = self._profile([0, 100, 200], [1500, 1490, 1485])
        # Smallest perturbation that survives a few arithmetic ops:
        perturbed = 200.0 + 1e-12
        out = ssp.extend_to(perturbed)
        assert out is ssp


class TestFieldSlicing:
    """:meth:`Field.at` / :meth:`Field.isel` drop the named axis from
    ``coords`` and record the selected sample in :attr:`pinned`.
    :meth:`Field.max` does the same for every axis."""

    @staticmethod
    def _full_grid(complex_data: bool = True):
        from uacpy.core.results import Field
        if complex_data:
            data = (np.arange(20).reshape(4, 5) + 1j).astype(complex)
        else:
            data = np.arange(20, dtype=float).reshape(4, 5) + 30.0
        return Field(
            data=data,
            coords={
                'depth': np.linspace(10, 90, 4),
                'range': np.linspace(100, 1000, 5),
            },
            model='Test', frequencies=100.0,
        )

    @staticmethod
    def _tf():
        from uacpy.core.results import Field
        data = (np.arange(24).reshape(2, 3, 4) + 1j).astype(complex)
        return Field(
            data=data,
            coords={
                'depth': np.array([10., 20.]),
                'range': np.array([100., 200., 300.]),
                'frequency': np.array([100., 200., 300., 400.]),
            },
            phase_reference='travelling_wave',
            model='Test',
        )

    def test_full_grid_tl_preserves_data_shape(self):
        f = self._full_grid(complex_data=True)
        assert f.tl.shape == f.data.shape == (4, 5)
        assert f.p.shape == f.data.shape

    def test_eval_interpolates_and_differs_from_neighbours(self):
        f = self._full_grid(complex_data=False)
        # on-grid eval matches at; off-grid midpoint differs from both neighbours
        assert np.allclose(f.eval(depth=10.0).data, f.at(depth=10.0).data)
        d0, d1 = float(f.coords['depth'][0]), float(f.coords['depth'][1])
        mid = f.eval(depth=(d0 + d1) / 2)
        assert not np.allclose(mid.data, f.isel(depth=0).data)
        assert not np.allclose(mid.data, f.isel(depth=1).data)

    def test_eval_method_nearest_matches_at(self):
        f = self._full_grid(complex_data=False)
        d0, d1 = float(f.coords['depth'][0]), float(f.coords['depth'][1])
        mid = (d0 + d1) / 2
        assert np.allclose(f.eval(depth=mid, method='nearest').data,
                           f.at(depth=mid).data)

    def test_eval_two_axes_to_scalar(self):
        f = self._full_grid(complex_data=False)
        s = f.eval(depth=55.0, range=550.0)
        assert s.data.shape == () and set(s.pinned) == {'depth', 'range'}

    def test_eval_bad_method_and_unknown_axis_raise(self):
        f = self._full_grid(complex_data=False)
        with pytest.raises(ConfigurationError):
            f.eval(depth=50.0, method='spline')
        with pytest.raises(ConfigurationError, match='unknown axis'):
            f.eval(frequency=200.0)

    def test_p_raises_on_real_data(self):
        f = self._full_grid(complex_data=False)
        with pytest.raises(AttributeError):
            _ = f.p

    def test_at_depth_drops_axis_and_records_pinned(self):
        f = self._full_grid()
        sliced = f.at(depth=50.0)
        assert list(sliced.coords) == ['range']
        assert sliced.data.shape == (5,)
        assert 'depth' in sliced.pinned

    def test_at_range_drops_axis(self):
        f = self._full_grid()
        sliced = f.at(range=500.0)
        assert list(sliced.coords) == ['depth']
        assert sliced.data.shape == (4,)
        assert 'range' in sliced.pinned

    def test_at_point_collapses_to_scalar(self):
        f = self._full_grid()
        point = f.at(range=500.0, depth=50.0)
        assert list(point.coords) == []
        assert point.data.shape == ()
        assert isinstance(float(point.tl), float)

    def test_max_records_every_axis_in_pinned(self):
        f = self._full_grid()
        m = f.max()
        assert list(m.coords) == []
        assert set(m.pinned) == {'depth', 'range'}
        flat = int(np.argmax(np.abs(f.data)))
        d_idx, r_idx = np.unravel_index(flat, f.data.shape)
        assert m.pinned['depth'] == float(f.coords['depth'][d_idx])
        assert m.pinned['range'] == float(f.coords['range'][r_idx])

    def _tl_grid(self, data):
        from uacpy.core.results import Field
        d = np.asarray(data, dtype=float)
        return Field(
            data=d,
            coords={'depth': np.arange(d.shape[0], dtype=float) * 10 + 10,
                    'range': np.arange(d.shape[1], dtype=float) * 100 + 100},
            model='Test', frequencies=100.0,
        )

    def test_max_on_tl_returns_loudest(self):
        # kind='tl': loudest = smallest dB; a NaN no-data cell and an 80 dB
        # cell must lose to 35 dB.
        f = self._tl_grid([[40.0, np.nan], [35.0, 80.0]])
        m = f.max()
        assert float(m.data) == pytest.approx(35.0)
        assert m.pinned['depth'] == 20.0 and m.pinned['range'] == 100.0

    def test_max_on_tl_skips_nan(self):
        f = self._tl_grid([[np.nan, 50.0], [45.0, 60.0]])
        assert float(f.max().data) == pytest.approx(45.0)

    def test_max_all_nan_raises(self):
        from uacpy.core.exceptions import ConfigurationError
        f = self._tl_grid([[np.nan, np.nan]])
        with pytest.raises(ConfigurationError, match='finite'):
            f.max()

    def test_max_complex_unchanged_argmax_abs(self):
        f = self._full_grid(complex_data=True)   # |data| argmax, not dB path
        m = f.max()
        assert abs(complex(m.data)) == pytest.approx(np.max(np.abs(f.data)))

    def test_tf_at_frequency_drops_frequency_axis(self):
        tf = self._tf()
        narrow = tf.at(frequency=300.0)
        assert list(narrow.coords) == ['depth', 'range']
        assert narrow.data.shape == (2, 3)
        assert narrow.pinned['frequency'] == 300.0

    def test_tf_at_spatial_keeps_frequency_axis(self):
        tf = self._tf()
        spec = tf.at(depth=15.0, range=200.0)
        assert list(spec.coords) == ['frequency']
        assert spec.data.shape == (4,)
        # ``depth=15`` is equidistant from samples 10 and 20; argmin picks
        # the first → 10.0.
        assert spec.pinned['depth'] == 10.0
        assert spec.pinned['range'] == 200.0

    def test_tf_at_frequency_narrows_identity(self):
        from uacpy.core.results import Field
        # A broadband field carries both a frequency coord and a frequencies
        # identity (as a wrapper emits). Pinning one frequency narrows the
        # identity so f0 / n_frequencies / repr reflect the pinned value.
        freqs = np.array([100., 200., 300., 400.])
        tf = Field(
            data=(np.arange(24).reshape(2, 3, 4) + 1j).astype(complex),
            coords={'depth': np.array([10., 20.]),
                    'range': np.array([100., 200., 300.]),
                    'frequency': freqs},
            model='Test', frequencies=freqs,
        )
        assert tf.n_frequencies == 4 and tf.f0 == 100.0
        narrow = tf.at(frequency=300.0)
        assert narrow.n_frequencies == 1
        assert narrow.f0 == 300.0
        assert 'f=300 Hz' in repr(narrow)
        # a non-frequency slice keeps the full identity
        assert tf.at(depth=10.0).n_frequencies == 4

    def test_tf_to_tl_returns_real_field(self):
        tf = self._tf()
        tl = tf.to_tl()
        assert not tl.is_complex
        assert tl.data.shape == tf.data.shape


class TestResultStackInvariants:
    """:class:`ResultStack` is a thin composition wrapper. The
    constructor enforces uniform slab type, uniform model / backend /
    frequencies, and matching ``len(slabs) == len(source_depths)`` so
    the stack's read-through properties (``stack.model``,
    ``stack.frequencies``) never silently disagree with a slab."""

    @staticmethod
    def _slab(*, depths=2, ranges=3, frequencies=100.0, model='Test',
              source_depth=50.0, model_source=None, phase_reference=None):
        from uacpy.core.results import Field
        return Field(
            data=np.ones((depths, ranges), dtype=complex),
            coords={
                'depth': np.arange(depths, dtype=float),
                'range': np.arange(ranges, dtype=float) * 100.0,
            },
            model=model,
            frequencies=frequencies,
            source_depths=np.array([float(source_depth)]),
            model_source=model_source,
            phase_reference=phase_reference,
        )

    def test_requires_at_least_one_slab(self):
        from uacpy.core.results import ResultStack
        with pytest.raises(ConfigurationError, match="at least one slab"):
            ResultStack(slabs=[], coordinate=[])

    def test_rejects_length_mismatch(self):
        from uacpy.core.results import ResultStack
        with pytest.raises(ConfigurationError, match="coordinate length"):
            ResultStack(slabs=[self._slab(source_depth=10.0)],
                        coordinate=[10.0, 20.0])

    def test_rejects_mixed_slab_types(self):
        from uacpy.core.results import Rays, ResultStack
        pf = self._slab(source_depth=10.0)
        ry = Rays(rays=[], model='Test', backend='')
        with pytest.raises(ConfigurationError, match="same concrete type"):
            ResultStack(slabs=[pf, ry], coordinate=[10.0, 20.0])

    def test_rejects_disagreeing_frequencies(self):
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0, frequencies=100.0)
        b = self._slab(source_depth=20.0, frequencies=200.0)
        with pytest.raises(ConfigurationError, match="frequencies"):
            ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])

    def test_rejects_disagreeing_model(self):
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0, model='Bellhop')
        b = self._slab(source_depth=20.0, model='Kraken')
        with pytest.raises(ConfigurationError, match="model"):
            ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])

    def test_accepts_uniform_slabs(self):
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0)
        b = self._slab(source_depth=20.0)
        stack = ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])
        assert stack.slab_type is Field
        assert stack.coordinate_name == 'source_depth'
        assert stack.n_slabs == 2
        assert len(stack) == 2
        # Universally-shared metadata reads through from slab[0].
        assert stack.model == 'Test'
        np.testing.assert_array_equal(
            stack.coordinate, np.array([10.0, 20.0]))

    def test_iteration_and_label_select_share_slab_identity(self):
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0)
        b = self._slab(source_depth=20.0)
        stack = ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])
        # __getitem__ returns the same object stored in slabs[i].
        assert stack[0] is a
        assert stack[1] is b
        # at(source_depth=z) routes to the nearest slab by label.
        assert stack.at(source_depth=20.0) is b
        # Iteration yields (source_depth, slab) pairs.
        pairs = list(stack)
        assert pairs == [(10.0, a), (20.0, b)]

    def test_frequency_axis_stack(self):
        """Stacking along ``frequency`` is just a coordinate-name swap.
        Slabs may now legitimately differ on ``frequencies`` (the
        stacking axis) while sharing ``source_depths`` and ``model``."""
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=50.0, frequencies=100.0)
        b = self._slab(source_depth=50.0, frequencies=200.0)
        stack = ResultStack(slabs=[a, b], coordinate=[100.0, 200.0],
                            coordinate_name='frequency')
        assert stack.coordinate_name == 'frequency'
        assert stack.at(frequency=200.0) is b
        # Mis-keyed kwarg → clear TypeError.
        with pytest.raises(ConfigurationError, match="frequency"):
            stack.at(source_depth=200.0)

    def test_frequency_axis_rejects_disagreeing_source_depths(self):
        """When stacking by ``frequency`` the slabs must still agree on
        ``source_depths`` (it's no longer the varying axis)."""
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0, frequencies=100.0)
        b = self._slab(source_depth=99.0, frequencies=200.0)
        with pytest.raises(ConfigurationError, match="source_depths"):
            ResultStack(slabs=[a, b], coordinate=[100.0, 200.0],
                        coordinate_name='frequency')

    def test_external_coordinate_axis(self):
        """An external coordinate (e.g. wind speed) requires both
        ``frequencies`` and ``source_depths`` to agree across slabs;
        ``at(<coordinate_name>=…)`` keys off the custom name."""
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=50.0, frequencies=100.0)
        b = self._slab(source_depth=50.0, frequencies=100.0)
        stack = ResultStack(slabs=[a, b], coordinate=[5.0, 15.0],
                            coordinate_name='wind_speed')
        assert stack.coordinate_name == 'wind_speed'
        assert stack.at(wind_speed=15.0) is b
        # Disagreeing source_depth is now rejected (external coord
        # requires both internal axes to agree).
        c = self._slab(source_depth=99.0, frequencies=100.0)
        with pytest.raises(ConfigurationError, match="source_depths"):
            ResultStack(slabs=[a, c], coordinate=[5.0, 15.0],
                        coordinate_name='wind_speed')

    def test_forwards_the_whole_identity_surface(self):
        """A stack forwards every identity field a slab carries, not just
        model/backend — the plotters read ``model_source`` for the
        model-credit footnote."""
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0, model_source='engine-provenance',
                       phase_reference='travelling_wave')
        b = self._slab(source_depth=20.0, model_source='engine-provenance',
                       phase_reference='travelling_wave')
        stack = ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])
        # Every field of the shared identity surface, so a field added to
        # ``Result.__init__`` cannot silently stop at the stack boundary.
        missing = [k for k in a.id_kwargs() if not hasattr(stack, k)]
        assert not missing, f"ResultStack does not forward {missing}"
        assert stack.model_source == 'engine-provenance'
        assert stack.phase_reference == 'travelling_wave'
        # The stacking axis reads back as the stack coordinate; the other
        # axis reads through from the (verified identical) slabs.
        np.testing.assert_array_equal(stack.source_depths, [10.0, 20.0])
        np.testing.assert_array_equal(stack.frequencies, [100.0])

    def test_frequency_stack_forwards_the_frequency_axis(self):
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=50.0, frequencies=100.0)
        b = self._slab(source_depth=50.0, frequencies=200.0)
        stack = ResultStack(slabs=[a, b], coordinate=[100.0, 200.0],
                            coordinate_name='frequency')
        np.testing.assert_array_equal(stack.frequencies, [100.0, 200.0])
        np.testing.assert_array_equal(stack.source_depths, [50.0])


class TestResultIngestCopiesArrays:
    """Every ``Result`` copies on ingest, and ``Field.to_dict`` is a real
    snapshot — a caller mutating their source array can never reach inside a
    result, and a cached dict never aliases the field it came from."""

    @staticmethod
    def _field():
        from uacpy.core.results import Field
        return Field(
            data=np.ones((2, 3), dtype=complex),
            coords={'depth': np.array([10.0, 20.0]),
                    'range': np.array([100.0, 200.0, 300.0])},
            model='Test', source_depths=np.array([50.0]),
            frequencies=np.array([100.0]))

    def test_result_copies_identity_arrays_on_ingest(self):
        from uacpy.core.results import Field
        sd = np.array([50.0])
        fr = np.array([100.0])
        f = Field(data=np.ones((1, 1)), coords={'depth': [0.0], 'range': [0.0]},
                  source_depths=sd, frequencies=fr)
        sd[0] = 999.0
        fr[0] = 999.0
        assert f.source_depths[0] == 50.0
        assert f.frequencies[0] == 100.0

    def test_to_dict_is_a_snapshot(self):
        f = self._field()
        d = f.to_dict()
        for key, live in (('data', f.data), ('frequencies', f.frequencies),
                          ('source_depths', f.source_depths)):
            assert not np.shares_memory(d[key], live), key
        for name, vec in f.coords.items():
            assert not np.shares_memory(d['coords'][name], vec), name

    def test_covariance_and_replicas_copy_on_ingest(self):
        from uacpy.core.results import Covariance, Replicas
        cov_src = np.zeros((1, 2, 2), dtype=complex)
        cov = Covariance(covariance=cov_src, model='OASN')
        cov_src[0, 0, 0] = 99.0
        assert cov.covariance[0, 0, 0] == 0.0

        rep_src = np.zeros((1, 1, 1, 1, 2), dtype=complex)
        rep = Replicas(replicas=rep_src, replica_z=[0.0], replica_x=[0.0],
                       replica_y=[0.0], model='OASN')
        rep_src[0, 0, 0, 0, 0] = 7.0
        assert rep.replicas[0, 0, 0, 0, 0] == 0.0


class TestCopyAndGeolocation:
    """`.copy()` is universal across carriers + results; Environment carries
    optional geolocation/date provenance that survives copy."""

    def test_copy_symmetry_across_carriers_and_io(self):
        import uacpy
        from uacpy import (Bathymetry, Altimetry, Surface, Bottom,
                           SoundSpeedProfile, BoundaryProperties)
        objs = [
            Bathymetry(ranges=[0, 1000.], depths=[100, 90.]),
            Altimetry(ranges=[0, 1000.], heights=[0, -2.]),
            Surface.coerce(BoundaryProperties(acoustic_type='vacuum')),
            Bottom.from_halfspace(BoundaryProperties()),
            SoundSpeedProfile.from_pairs([(0, 1500), (100, 1490.)]),
            uacpy.Source(depths=50., frequencies=120.),
            uacpy.Receiver(depths=[100.], ranges=[2000.]),
            uacpy.Environment(bathymetry=200., ssp=1500.),
        ]
        for o in objs:
            c = o.copy()
            assert type(c) is type(o) and c is not o

    def test_source_and_receiver_copy_the_whole_attribute_surface(self):
        """``Source`` / ``Receiver`` deep-copy like every other carrier, so an
        attribute added to either is carried across without editing
        ``copy()``, and no array is shared with the original."""
        import uacpy
        objs = [
            uacpy.Source(depths=[10., 50.], frequencies=[100., 200.],
                         source_type='line',
                         beam_pattern=np.array([[-90., -20.], [90., 0.]])),
            uacpy.Receiver(depths=[100., 200.], ranges=[1000., 2000.]),
        ]
        for o in objs:
            c = o.copy()
            assert vars(c).keys() == vars(o).keys(), type(o).__name__
            for name, val in vars(o).items():
                new = getattr(c, name)
                if isinstance(val, np.ndarray):
                    np.testing.assert_array_equal(new, val)
                    assert not np.shares_memory(new, val), name
                else:
                    assert new == val, name

    def test_result_copy_is_independent(self):
        import uacpy
        env = uacpy.Environment(bathymetry=300., ssp=1500.)
        f = uacpy.Bellhop().compute_tl(
            env, uacpy.Source(depths=50., frequencies=150.),
            uacpy.Receiver(depths=[100., 200.], ranges=[2000., 4000.]))
        c = f.copy()
        assert type(c) is type(f) and c is not f

    def test_environment_geolocation_and_date(self):
        import datetime
        import uacpy
        e = uacpy.Environment(bathymetry=200., ssp=1500.,
                              location=(75., 12.5), date='2026-03-15')
        assert e.location == (75., 12.5)
        assert e.date == datetime.date(2026, 3, 15)
        assert e.transect is None
        # transect → location defaults to the midpoint; explicit overrides it
        t = uacpy.Environment(bathymetry=[(0, 200), (10000, 300)], ssp=1500.,
                              transect=((75., 0.), (77., 40.)))
        assert t.location == (76.0, 20.0)
        o = uacpy.Environment(bathymetry=200., ssp=1500., location=(75., 0.),
                              transect=((75., 0.), (77., 40.)))
        assert o.location == (75., 0.)
        # survives a deep copy
        c = t.copy()
        assert c.location == (76., 20.) and c.transect == ((75., 0.), (77., 40.))
        # hand-built env carries none of it
        plain = uacpy.Environment(bathymetry=100., ssp=1500.)
        assert plain.location is None and plain.transect is None and plain.date is None

    @pytest.mark.parametrize("bad", [(91.0, 0.0), ('a', 'b'), 'not-a-date'])
    def test_environment_geolocation_typed_errors(self, bad):
        import uacpy
        with pytest.raises(ConfigurationError):
            if isinstance(bad, str):
                uacpy.Environment(bathymetry=100., ssp=1500., date=bad)
            else:
                uacpy.Environment(bathymetry=100., ssp=1500., location=bad)


class TestSourceGeometryAndBeamPattern:
    """Source owns source geometry and directivity (spec 2026-07-25)."""

    def test_scaled_is_a_valid_source_type(self):
        src = uacpy.Source(depths=50, frequencies=100, source_type='scaled')
        assert src.source_type == 'scaled'

    def test_unknown_source_type_raises(self):
        with pytest.raises(ConfigurationError, match="source_type"):
            uacpy.Source(depths=50, frequencies=100, source_type='Z')

    def test_beam_pattern_defaults_to_none(self):
        assert uacpy.Source(depths=50, frequencies=100).beam_pattern is None

    def test_beam_pattern_accepts_angle_level_array(self):
        pat = np.array([[-90.0, -20.0], [0.0, 0.0], [90.0, -20.0]])
        src = uacpy.Source(depths=50, frequencies=100, beam_pattern=pat)
        assert src.beam_pattern.shape == (3, 2)

    def test_beam_pattern_wrong_shape_raises(self):
        with pytest.raises(ConfigurationError, match=r"N, 2"):
            uacpy.Source(depths=50, frequencies=100,
                         beam_pattern=np.array([1.0, 2.0, 3.0]))

    def test_beam_pattern_non_monotonic_angles_raise(self):
        # beampattern.f90:56-58 rejects this with ERROUT, which gfortran
        # exits 0 on; catching it here is the point of validating in Python.
        pat = np.array([[0.0, 0.0], [-90.0, -20.0], [90.0, -20.0]])
        with pytest.raises(ConfigurationError):
            uacpy.Source(depths=50, frequencies=100, beam_pattern=pat)

    def test_beam_pattern_path_is_stored_as_path(self, tmp_path):
        from pathlib import Path
        sbp = tmp_path / 'pattern.sbp'
        sbp.write_text("2\n-90.0 0.0\n90.0 0.0\n")
        src = uacpy.Source(depths=50, frequencies=100, beam_pattern=sbp)
        assert isinstance(src.beam_pattern, Path)

    def test_copy_carries_geometry_and_pattern(self):
        pat = np.array([[-90.0, -20.0], [90.0, 0.0]])
        src = uacpy.Source(depths=50, frequencies=100,
                           source_type='line', beam_pattern=pat)
        dup = src.copy()
        assert dup.source_type == 'line'
        np.testing.assert_array_equal(dup.beam_pattern, pat)


class TestEnvironmentPredicatesAreProperties:
    """``has_*`` must be properties, not methods.

    As bound methods they are always truthy, so ``if env.has_layered_bottom:``
    silently takes the True branch for every environment. They sit alongside
    ``is_range_dependent``, which was already a property.
    """

    _NAMES = (
        'has_range_dependent_bathymetry', 'has_range_dependent_ssp',
        'has_range_dependent_bottom', 'has_layered_bottom',
        'has_range_dependent_layered_bottom', 'has_elastic_bottom',
        'has_elastic_surface',
    )

    @pytest.mark.parametrize('name', _NAMES)
    def test_is_a_property(self, name):
        import inspect
        from uacpy.core.environment import Environment
        assert isinstance(inspect.getattr_static(Environment, name), property), (
            f"Environment.{name} is a {type(inspect.getattr_static(Environment, name)).__name__}; "
            f"as a method it is always truthy in a boolean test")

    def test_flat_environment_reports_false_not_truthy_method(self):
        env = uacpy.Environment(bathymetry=100.0, ssp=1500.0)
        for name in self._NAMES:
            assert getattr(env, name) is False, f"{name} should be False"


class TestReflectionFileDedupe:
    """``dedupe_reflection_file`` is a .brc/.trc rewriter, not a .irc one.

    BOUNCE writes the .irc with a title/frequency header and six
    fixed-format columns (``bounce.f90:225-228``), read back by the same
    fixed format at ``misc/RefCoef.f90:97-107``. Running the 3-column
    angle dedupe over one stripped the header and four of the columns.
    """

    def _irc(self, tmp_path):
        p = tmp_path / 't.irc'
        p.write_text(
            " ' BOUNCE test '  100.0\n 2\n"
            "     0.1000000     1.0000000     0.0000000"
            "     0.5000000     0.1000000    0\n"
            "     0.2000000     0.9000000     0.1000000"
            "     0.4000000     0.2000000    1\n")
        return p

    def test_irc_is_rejected_and_left_untouched(self, tmp_path):
        from uacpy.io.refl_io import dedupe_reflection_file
        from uacpy.core.exceptions import FileFormatError
        irc = self._irc(tmp_path)
        before = irc.read_text()
        with pytest.raises(FileFormatError, match='row count'):
            dedupe_reflection_file(irc)
        assert irc.read_text() == before

    def test_brc_duplicate_angles_are_collapsed(self, tmp_path):
        from uacpy.io.refl_io import dedupe_reflection_file
        brc = tmp_path / 't.brc'
        brc.write_text("   4\n  0.0 1.0 0.0\n  0.0 1.0 0.0\n"
                       "  30.0 0.8 10.0\n  60.0 0.5 20.0\n")
        dedupe_reflection_file(brc)
        rows = [ln.split() for ln in brc.read_text().splitlines() if ln.strip()]
        assert int(rows[0][0]) == 3
        assert [float(r[0]) for r in rows[1:]] == [0.0, 30.0, 60.0]
