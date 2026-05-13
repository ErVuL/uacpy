"""
Tests for core UACPY classes: Environment, Source, Receiver, Result
"""

import pytest
import numpy as np
import uacpy
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

    def test_create_munk_environment(self, munk_env):
        """Test creating environment with Munk profile."""
        assert munk_env.name == "Munk Profile"
        assert munk_env.depth == 100.0
        assert munk_env.ssp.n_depths == 21
        assert munk_env.ssp.shape == 'measured'

    def test_range_dependent_environment(self, range_dependent_env):
        """Test range-dependent environment."""
        assert range_dependent_env.is_range_dependent
        assert len(range_dependent_env.bathymetry) == 11
        assert range_dependent_env.bathymetry[0, 1] == 80.0
        assert range_dependent_env.bathymetry[-1, 1] == 120.0

    def test_ssp_pairs_shape(self, simple_env, munk_env):
        """SSP pairs view always has shape (N, 2)."""
        assert simple_env.ssp.to_pairs().shape[1] == 2
        assert munk_env.ssp.to_pairs().shape[1] == 2

    def test_get_representative_depth(self, range_dependent_env):
        """Test getting representative depth for range-dependent environment."""
        median_depth = range_dependent_env.get_representative_depth('median')
        assert 80 <= median_depth <= 120

    def test_invalid_depth(self):
        """Test that negative depth raises error."""
        with pytest.raises(ValueError):
            uacpy.Environment(name="Test", bathymetry=-10, ssp=1500)

    def test_bathymetry_rejects_negative_range(self):
        """Bathymetry ranges are measured from the source; they cannot
        be negative."""
        with pytest.raises(ValueError, match="ranges must be non-negative"):
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
        with pytest.raises(ValueError, match=match):
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
    with pytest.raises(ValueError, match="finite"):
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
        """Unknown ``receiver_type`` strings raise :class:`ValueError`,
        mirroring the validation on ``Source.source_type``."""
        with pytest.raises(ValueError, match="receiver_type"):
            uacpy.Receiver(depths=50, ranges=1000, receiver_type='gird')

    def test_receiver_type_accepts_grid_and_line(self):
        rx = uacpy.Receiver(depths=50, ranges=1000, receiver_type='grid')
        assert rx.receiver_type == 'grid'
        rx = uacpy.Receiver(depths=50, ranges=1000, receiver_type='line')
        assert rx.receiver_type == 'line'


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


class TestPublicReexports:
    """Top-level re-exports added for the API audit."""

    def test_environment_helpers_at_top_level(self):
        from uacpy import SoundSpeedProfile, generate_sea_surface
        assert SoundSpeedProfile is uacpy.core.environment.SoundSpeedProfile
        assert generate_sea_surface is uacpy.core.environment.generate_sea_surface

    def test_environment_helpers_at_core(self):
        from uacpy.core import SoundSpeedProfile, generate_sea_surface
        assert SoundSpeedProfile is uacpy.core.environment.SoundSpeedProfile
        assert generate_sea_surface is uacpy.core.environment.generate_sea_surface

    def test_signal_analysis_classes_at_top_level(self):
        sig = uacpy.signal
        for name in ('PPSD', 'PSD', 'FRF', 'SEL', 'FKTransform', 'Spectrogram'):
            assert hasattr(sig, name), f"uacpy.signal.{name} not reachable"
        assert 'PSD' in sig.__all__
        assert 'FRF' in sig.__all__
        assert 'SEL' in sig.__all__
        assert 'FKTransform' in sig.__all__


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
        with pytest.raises(ValueError, match='requires frequencies'):
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
        with pytest.raises(ValueError, match="broadband"):
            rc.at(frequency=100.0)


class TestSoundSpeedProfileNearestVsInterp:
    """``SoundSpeedProfile.at(...)`` is **nearest**;
    ``SoundSpeedProfile.interp(...)`` is **linear** — invariant for the
    whole grid library: ``at`` never fabricates values."""

    def _ssp(self):
        from uacpy.core.environment import SoundSpeedProfile
        return SoundSpeedProfile(
            depths=np.array([0.0, 100.0, 200.0]),
            data=np.array([[1500.0], [1490.0], [1480.0]])
        )

    def test_eval_nearest_picks_nearest_depth(self):
        ssp = self._ssp()
        # depth=51 is closer to 100 than to 0 → returns the 100m sample
        sliced = ssp.eval(depth=51.0, interp='nearest')
        assert sliced.depths[0] == 100.0
        assert sliced.value == 1490.0

    def test_interp_linear(self):
        ssp = self._ssp()
        # depth=50 is halfway between (0, 1500) and (100, 1490) → 1495
        sliced = ssp.eval(depth=50.0, interp='linear')
        assert sliced.depths[0] == 50.0
        assert sliced.value == pytest.approx(1495.0)


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
              source_depth=50.0):
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
        )

    def test_requires_at_least_one_slab(self):
        from uacpy.core.results import ResultStack
        with pytest.raises(ValueError, match="at least one slab"):
            ResultStack(slabs=[], coordinate=[])

    def test_rejects_length_mismatch(self):
        from uacpy.core.results import ResultStack
        with pytest.raises(ValueError, match="coordinate length"):
            ResultStack(slabs=[self._slab(source_depth=10.0)],
                        coordinate=[10.0, 20.0])

    def test_rejects_mixed_slab_types(self):
        from uacpy.core.results import Rays, ResultStack
        pf = self._slab(source_depth=10.0)
        ry = Rays(rays=[], model='Test', backend='')
        with pytest.raises(TypeError, match="same concrete type"):
            ResultStack(slabs=[pf, ry], coordinate=[10.0, 20.0])

    def test_rejects_disagreeing_frequencies(self):
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0, frequencies=100.0)
        b = self._slab(source_depth=20.0, frequencies=200.0)
        with pytest.raises(ValueError, match="frequencies"):
            ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])

    def test_rejects_disagreeing_model(self):
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0, model='Bellhop')
        b = self._slab(source_depth=20.0, model='Kraken')
        with pytest.raises(ValueError, match="model"):
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
        with pytest.raises(TypeError, match="frequency"):
            stack.at(source_depth=200.0)

    def test_frequency_axis_rejects_disagreeing_source_depths(self):
        """When stacking by ``frequency`` the slabs must still agree on
        ``source_depths`` (it's no longer the varying axis)."""
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0, frequencies=100.0)
        b = self._slab(source_depth=99.0, frequencies=200.0)
        with pytest.raises(ValueError, match="source_depths"):
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
        with pytest.raises(ValueError, match="source_depths"):
            ResultStack(slabs=[a, c], coordinate=[5.0, 15.0],
                        coordinate_name='wind_speed')
