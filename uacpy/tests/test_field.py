"""``Field``, the gridded result.

Value accessors and their write guards, label slicing (``at``, ``isel``,
``eval``), the unit, kind and dtype axes, dB views, masking below the
seafloor, windowing and delay removal, time-series synthesis from a transfer
function, and the ``np.savez`` and xarray round trips.

A recurring subject is the label query. ``at()`` and its siblings pick a
node with ``argmin(|axis - label|)``, which ranks nothing when every
distance is NaN and hands back index 0, a real node, so the caller sees a
plausible answer to an unanswerable question. Those guards are pinned on
both sides: the label that must be refused, and the legitimate one next to
it that must still work.
"""

import inspect
import numpy as np
import pytest
import re
import uacpy
import warnings
from uacpy.core.boundary import BoundaryProperties
from uacpy.core.environment import Environment
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Field
from uacpy.core.results import PhaseReference
from uacpy.core.results import ReflectionCoefficient
from uacpy.core.results import SoundSpeeds
from uacpy.core.results.stack import ResultStack
from uacpy.core.source import Source
from uacpy.core.ssp import SoundSpeedProfile
from uacpy.tests._synthetic_fields import _field
from uacpy.tests._synthetic_fields import _two_path_grid
from uacpy.tests.conftest import recorded_warnings


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
        from uacpy.models.provenance import model_provenance
        src = model_provenance('acoustics_toolbox')
        field = self._tl_field(
            np.zeros((2, 2)), np.array([100.0, 200.0]),
            np.array([10.0, 20.0]), model_source=src)
        rt = Field.from_dict(field.to_dict())
        assert rt.model_source is src

    # data[d, r] = 10*d + r in the three tests below, so each value names
    # its own cell and a depth/range transpose (data[r, d]) is caught
    # exactly, which the previous 44-55 band assertions admitted.

    def test_at_point_returns_nearest_cell_value(self):
        data = np.arange(100).reshape(10, 10).astype(float)
        ranges = np.linspace(0, 9000, 10)   # 1000 m spacing
        depths = np.linspace(0, 90, 10)     # 10 m spacing
        field = self._tl_field(data, ranges, depths)
        # Off-centre query: range=4200 → index 4, depth=68 → index 7,
        # so the nearest cell is data[7, 4] = 74 (a transpose reads 47).
        assert float(field.at(range=4200.0, depth=68.0).dB) == 74.0

    def test_at_range_returns_nearest_cell_values(self):
        data = np.arange(100).reshape(10, 10).astype(float)
        ranges = np.linspace(0, 9000, 10)
        depths = np.linspace(0, 90, 10)
        field = self._tl_field(data, ranges, depths)
        values = field.at(range=4200.0).dB
        # Nearest range sample is index 4 (4000 m): the depth column
        # 10*d + 4. A transposed field would return 40..49 instead.
        np.testing.assert_array_equal(values, np.arange(10) * 10.0 + 4.0)

    def test_at_depth_returns_nearest_cell_values(self):
        data = np.arange(100).reshape(10, 10).astype(float)
        ranges = np.linspace(0, 9000, 10)
        depths = np.linspace(0, 90, 10)
        field = self._tl_field(data, ranges, depths)
        values = field.at(depth=68.0).dB
        # Nearest depth sample is index 7 (70 m): the range row 70..79.
        # A transposed field would return 8, 18, ..., 98 instead.
        np.testing.assert_array_equal(values, np.arange(10) + 70.0)

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

    def test_the_repr_of_a_slice_names_where_it_was_taken(self):
        """The peak of a matched-field surface prints ``scalar`` and where
        it was taken, which lives in ``pinned``."""
        data = np.zeros((10, 20))
        data[3, 7] = 1.0
        field = self._tl_field(data, np.linspace(100, 5000, 20),
                               np.linspace(10, 90, 10))
        assert ' at ' not in repr(field)
        peak = field.at(depth=field.depths[3]).at(range=field.ranges[7])
        assert repr(peak).endswith(
            f"scalar, at depth={field.depths[3]:g} m, "
            f"range={field.ranges[7]:g} m)")

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

        with pytest.raises(TypeError, match='takes 1 positional argument but'):
            field.resample_to(ranges, depths)

        out = field.resample_to(depths=[25.0, 75.0], ranges=[250.0, 750.0])
        assert list(out.coords) == ['depth', 'range']
        assert out.shape == (2, 2)
        np.testing.assert_allclose(out.data, [[25.0, 25.0], [75.0, 75.0]])

    @staticmethod
    def _coherent_field(dr, dz=1.0, f0=200.0, c=1500.0):
        """Complex pressure with carrier e^{ikr}, grid spaced ``dz`` x ``dr``."""
        ranges = np.arange(1000.0, 1500.0 + dr, dr)
        depths = np.arange(40.0, 60.0 + dz, dz)
        k = 2.0 * np.pi * f0 / c
        data = np.exp(1j * k * ranges)[None, :] / ranges[None, :]
        return Field(
            data=np.repeat(data, depths.size, axis=0),
            coords={'depth': depths, 'range': ranges},
            model='Test', frequencies=f0,
        )

    @staticmethod
    def _midpoints(field):
        r = field.coords['range']
        z = field.coords['depth']
        return dict(depths=z[:-1] + np.diff(z) / 2, ranges=r[:-1] + np.diff(r) / 2)

    # Quarter wavelength at 200 Hz with c = 1500 m/s. Each axis is bracketed
    # against it independently: a guard that fires on two axes needs the
    # coarse and the resolved case *per axis*, or a regression re-opening the
    # depth half (measured +1.36 dB, silent) passes on the range cases alone.
    QUARTER = 1.875

    @pytest.mark.parametrize('dr,dz,named,silent_axis', [
        (2.0, 1.8, 'range', 'depth'),      # range just over, depth just under
        (1.8, 2.0, 'depth', 'range'),      # depth just over, range just under
    ])
    def test_resample_to_brackets_each_axis_independently(self, dr, dz, named,
                                                          silent_axis):
        """Each axis is checked against the quarter wavelength on its own.
        The pair brackets 1.875 m tightly from both sides *per axis*: the
        named one is coarse at 2.0 m, the other resolved at 1.8 m, so the
        message must name exactly one."""
        field = self._coherent_field(dr, dz)
        with pytest.warns(UserWarning, match=f'{named} samples are') as rec:
            field.resample_to(**self._midpoints(field))
        assert f'{silent_axis} samples are' not in str(rec[0].message)

    @pytest.mark.parametrize('dr,dz,named', [
        (25.0, 1.0, 'range'),        # range alone   — measured +2.6 dB
        (1.0, 5.0, 'depth'),         # depth alone   — measured +1.4 dB, was silent
        (25.0, 5.0, 'depth'),        # both          — measured +4.9 dB
    ])
    def test_resample_to_warns_and_names_the_coarse_axis(self, dr, dz, named):
        """Interpolating a coherent field across opposite-phase lobes biases
        the level upward. Both axes matter and their biases compound, and
        ``resample_to`` always interpolates both — so a guard that reported
        only the range axis stayed silent on a pure depth-axis error and
        misattributed the mixed one."""
        field = self._coherent_field(dr, dz)
        with pytest.warns(UserWarning, match=f'{named} samples are'):
            field.resample_to(**self._midpoints(field))

    @pytest.mark.parametrize('dr,dz', [
        (0.5, 0.5), (1.0, 1.0), (1.8, 1.8),   # both resolved
        (1.8, 0.5), (0.5, 1.8),               # each axis at the bound in turn
    ])
    def test_resample_to_is_silent_when_both_axes_are_resolved(self, dr, dz):
        # The discriminating half: the guard must not fire below the quarter
        # wavelength, or it would be noise on every well-sampled field.
        assert max(dr, dz) < self.QUARTER
        field = self._coherent_field(dr, dz)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            field.resample_to(**self._midpoints(field))

    def test_resample_to_admits_it_cannot_check_a_field_with_no_frequency(self):
        """A wrapped phase step cannot stand in for the frequency: a grid
        spaced a whole wavelength aliases to 0.000 rad and reads as perfectly
        sampled while being maximally undersampled (it misses 47.8 % of coarse
        grids overall). So a Field with no f0 is reported as unverifiable
        rather than silently passing — a silence that reads as a pass is worse
        than an admission."""
        field = self._coherent_field(7.5, dz=1.0)         # dr == one wavelength
        blind = Field(data=field.data, coords=dict(field.coords),
                      model='Test', frequencies=None)
        with pytest.warns(UserWarning, match='carries no frequency'):
            blind.resample_to(**self._midpoints(blind))

    def test_resample_to_never_warns_for_a_real_field(self):
        # A TL field carries no carrier, so it interpolates freely however
        # coarse the grid — keying the guard on dtype rather than on grid
        # spacing alone is what keeps this quiet.
        ranges = np.arange(1000.0, 5000.0, 250.0)
        depths = np.array([50.0, 100.0])
        field = self._tl_field(np.zeros((depths.size, ranges.size)), ranges, depths,
                               frequencies=200.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            field.resample_to(depths=[50.0], ranges=ranges[:-1] + 125.0)


class TestFieldValueAccessorsAreWriteGuarded:
    """``Field`` copies on ingest, so no accessor may hand back a writable
    alias of ``data``: ``p = field.p; p *= k`` would otherwise corrupt the
    stored result. Real ``.dB`` is the common path (RAM / OAST /
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
        tl = f.dB
        assert not tl.flags.writeable
        with pytest.raises(ValueError,
                           match='assignment destination is read-only'):
            tl[0, 0] = -999.0
        np.testing.assert_array_equal(f.data, [[60.0, 70.0], [80.0, 90.0]])

    def test_real_tl_reads_the_stored_dB_values(self):
        f = self._real()
        np.testing.assert_array_equal(np.asarray(f.dB), f.data)

    def test_scalar_tl_is_read_only_and_castable(self):
        f = self._real().at(depth=0.0, range=1.0)
        assert not f.dB.flags.writeable
        assert float(f.dB) == 70.0

    def test_complex_tl_is_a_fresh_array(self):
        f = self._complex()
        tl = f.dB
        assert not np.shares_memory(tl, f.data)
        tl[0, 0] = -999.0                       # derived array: safe to write
        assert f.data[0, 0] == 1 + 1j

    def test_p_is_read_only(self):
        f = self._complex()
        with pytest.raises(ValueError,
                           match='assignment destination is read-only'):
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

    def test_data_is_the_writeable_buffer_the_read_only_views_share(self):
        """The other half of the guard, stated because ``.p``'s read-only flag
        reads like a promise about the field: it is on the *view*, and
        :attr:`data` is the same memory, writeable."""
        f = self._complex()
        assert f.data.flags.writeable
        assert np.shares_memory(f.data, f.p)
        f.data *= 1e6
        assert f.p[0, 0] == (1 + 1j) * 1e6

    def test_the_docs_say_data_is_writeable_and_shares_the_buffer(self):
        # A caller who has internalised ``.p``'s read-only guarantee has no
        # way to learn from the code that ``field.data *= k`` defeats it.
        prose = (Field.__doc__ or '') + '\n' + (Field.p.__doc__ or '')
        assert 'writeable' in prose
        assert 'read-only' in prose
        assert 'same buffer' in prose or 'same memory' in prose

    @pytest.mark.parametrize('dtype', ['float64', 'float32', 'int64'])
    def test_real_dB_aliases_data_in_every_real_dtype(self, dtype):
        """The real branch hands back ``data`` itself, so which engine
        produced the field decides nothing about the contract.

        float32 is the case a user meets: ``to_dB()`` of a ``.shd``-backed
        complex64 Field is float32. float64 is what an in-memory Field and
        the RAM/OAST readers carry. Both alias, and the array taken before a
        write reads the value written after it."""
        f = Field(data=np.ones((2, 2), dtype=dtype),
                  coords={'depth': [0.0, 1.0], 'range': [0.0, 1.0]},
                  unit='dB')
        view = f.dB
        assert np.shares_memory(view, f.data)
        assert not view.flags.writeable
        f.data[0, 0] = 999
        assert view[0, 0] == 999

    @pytest.mark.parametrize('dtype', ['float64', 'float32', 'int64'])
    def test_real_dB_carries_the_fields_own_dtype(self, dtype):
        # The accepted cost of aliasing: no upcast, so the caller reads the
        # stored precision and asks for float64 explicitly if it needs it.
        f = Field(data=np.ones((2, 2), dtype=dtype),
                  coords={'depth': [0.0, 1.0], 'range': [0.0, 1.0]},
                  unit='dB')
        assert f.dB.dtype == np.dtype(dtype)
        assert np.asarray(f.dB, dtype=float).dtype == np.dtype('float64')

    def test_to_dB_of_a_complex64_field_gives_a_float32_field_whose_dB_aliases(self):
        # The in-package route to a non-float64 real field: read_shd_bin
        # returns complex64, so to_dB() of it is float32.
        f = Field(data=np.ones((2, 2), dtype='complex64'),
                  coords={'depth': [0.0, 1.0], 'range': [0.0, 1.0]},
                  unit='Pa')
        real = f.to_dB()
        assert real.data.dtype == np.dtype('float32')
        assert not real.is_complex
        assert np.shares_memory(real.dB, real.data)
        assert real.dB.dtype == np.dtype('float32')

    def test_the_docstring_states_in_prose_that_dB_follows_the_fields_dtype(self):
        """The contract a caller reads before deciding whether to cast.

        Literal spans are stripped first: ``dtype`` also occurs inside the
        ``np.asarray(..., dtype=float)`` example, which shows how to opt out
        of the stored precision rather than stating what ``.dB`` returns.
        Matching it there passes on a docstring that never makes the claim.
        """
        prose = re.sub(r'``[^`]*``', ' ',
                       ' '.join((Field.dB.__doc__ or '').split()))
        assert 'dtype' in prose
        assert 'read-only view' in prose
        assert 'alias' in prose

    @pytest.mark.parametrize('dtype', ['complex128', 'complex64'])
    def test_complex_dB_is_a_fresh_array_in_every_complex_dtype(self, dtype):
        """The other side of the same boundary: the complex branch computes
        ``-20·log10|data|``, so there is nothing of ``data`` to alias and the
        caller owns what it gets."""
        f = Field(data=np.full((2, 2), 1 + 1j, dtype=dtype),
                  coords={'depth': [0.0, 1.0], 'range': [0.0, 1.0]},
                  unit='Pa')
        tl = f.dB
        assert not np.shares_memory(tl, f.data)
        tl[0, 0] = -999.0
        assert f.data[0, 0] == 1 + 1j

    def test_to_dict_does_not_alias_the_field(self):
        f = self._real()
        d = f.to_dict()
        d['data'][1, 1] = -1.0
        d['coords']['depth'][0] = 99.0
        assert f.data[1, 1] == 90.0
        assert f.coords['depth'][0] == 0.0


class TestFieldMaxComplexData:
    """max() ranks complex data by magnitude whatever unit the field is
    tagged with, so no float cast of complex values occurs."""

    def test_complex_dB_tagged_field_uses_magnitude(self):
        f = Field(
            data=np.array([[1 + 1j, 3 + 4j]]),
            coords={'depth': np.array([1.0]),
                    'range': np.array([10.0, 20.0])},
            kind='reverberation', unit='dB',
            frequencies=np.array([100.0]))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m = f.max()
        assert complex(m.data) == 3 + 4j
        assert m.pinned['range'] == 20.0


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
        assert f.dB.shape == f.data.shape == (4, 5)
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

    @pytest.mark.parametrize('label, warns', [
        (500.0, False), (500.001, True), (100.0, False), (99.999, True),
    ])
    def test_eval_warns_past_either_end_of_the_axis(self, label, warns):
        f = Field(data=np.arange(10.0).reshape(2, 5),
                  coords={'depth': np.array([10.0, 20.0]),
                          'range': np.linspace(100.0, 500.0, 5)},
                  kind='pressure', unit='dB')
        with recorded_warnings() as record:
            value = f.eval(range=label, depth=10.0).data
        messages = [str(w.message) for w in record
                    if 'lies outside the' in str(w.message)]
        if warns:
            (message,) = messages
            edge = 500 if label > 500.0 else 100
            assert f"holds the edge value range={edge}" in message
            assert value == f.at(range=float(edge), depth=10.0).data
        else:
            assert messages == []

    def test_eval_bad_method_and_unknown_axis_raise(self):
        f = self._full_grid(complex_data=False)
        with pytest.raises(ConfigurationError,
                           match='interpolation method must be one of'):
            f.eval(depth=50.0, method='spline')
        with pytest.raises(ConfigurationError, match='unknown axis'):
            f.eval(frequency=200.0)

    def test_p_raises_on_real_data(self):
        f = self._full_grid(complex_data=False)
        with pytest.raises(AttributeError,
                           match='complex pressure unavailable'):
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
        assert isinstance(float(point.dB), float)

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
        # unit='dB': loudest = smallest dB; a NaN no-data cell and an 80 dB
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

    def test_max_of_a_complex_field_picks_the_largest_magnitude(self):
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
        assert 'frequency 300 Hz' in repr(narrow)
        # a non-frequency slice keeps the full identity
        assert tf.at(depth=10.0).n_frequencies == 4

    def test_tf_to_tl_returns_real_field(self):
        tf = self._tf()
        tl = tf.to_dB()
        assert not tl.is_complex
        assert tl.data.shape == tf.data.shape

    def test_tf_to_tl_is_minus_20log10_magnitude(self):
        """``to_dB`` is exactly ``-20·log10(|data|)`` (every |data| here is
        far above the PRESSURE_FLOOR clamp, so the clamp is inert)."""
        tf = self._tf()
        tl = tf.to_dB()
        np.testing.assert_allclose(
            tl.data, -20.0 * np.log10(np.abs(tf.data)), rtol=1e-12)
        # One hand-checked value: data flat index 3 is 3+1j, |3+1j|² = 10,
        # so -20·log10(√10) = -10 dB exactly.
        k = np.unravel_index(3, tf.data.shape)
        assert tl.data[k] == pytest.approx(-10.0, abs=1e-12)


class TestFieldEvalSamplingGuard:
    """``Field.eval`` interpolates the same coherent field as
    ``resample_to`` and carried the same +2.3 dB level bias with no warning:
    the guard existed on one public interpolation path and not the other.
    ``eval`` is checked per **requested axis** — interpolating along range
    says nothing about the depth spacing — and ``method='nearest'`` is exempt
    because it fabricates nothing."""

    @staticmethod
    def _field(dr, dz, f0=200.0, c=1500.0):
        ranges = np.arange(1000.0, 1100.0 + dr, dr)
        depths = np.arange(40.0, 60.0 + dz, dz)
        k = 2.0 * np.pi * f0 / c
        row = np.exp(1j * k * ranges)[None, :] / ranges[None, :]
        return Field(data=np.repeat(row, depths.size, axis=0),
                     coords={'depth': depths, 'range': ranges},
                     model='Test', frequencies=f0)

    @pytest.mark.parametrize('axis,dr,dz', [('range', 25.0, 1.0),
                                            ('depth', 1.0, 5.0)])
    def test_eval_warns_on_the_axis_it_interpolates(self, axis, dr, dz):
        field = self._field(dr, dz)
        with pytest.warns(UserWarning, match=f'{axis} samples are'):
            field.eval(**{axis: float(field.coords[axis][0]) + 0.5})

    def test_eval_ignores_a_coarse_axis_it_is_not_interpolating(self):
        # The discriminating half: a coarse depth axis is irrelevant to
        # eval(range=...), and warning about it would be noise.
        field = self._field(dr=1.0, dz=5.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            field.eval(range=1000.5)

    def test_eval_is_silent_on_a_resolved_axis(self):
        field = self._field(dr=1.0, dz=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            field.eval(depth=40.5, range=1000.5)

    def test_nearest_never_warns(self):
        field = self._field(dr=25.0, dz=5.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            field.eval(method='nearest', depth=42.5, range=1012.5)

    def test_resample_to_nearest_never_warns_either(self):
        """The exemption is a property of ``method='nearest'``, not of
        ``eval``: it returns a stored sample on both paths, so there is no
        phase to corrupt and nothing to announce."""
        field = self._field(dr=25.0, dz=5.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            field.resample_to(depths=field.coords['depth'],
                              ranges=field.coords['range'], method='nearest')

    def test_resample_to_warns_when_it_interpolates(self):
        field = self._field(dr=25.0, dz=5.0)
        with pytest.warns(UserWarning, match='samples are'):
            field.resample_to(depths=field.coords['depth'],
                              ranges=field.coords['range'], method='linear')

    @staticmethod
    def _descending(field):
        """The same field with its range axis (and data) reversed — the
        identical physical grid, stored high-to-low."""
        return Field(data=field.data[:, ::-1],
                     coords={'depth': field.coords['depth'],
                             'range': field.coords['range'][::-1]},
                     model='Test', frequencies=field.frequencies)

    def test_a_descending_range_axis_warns_like_ascending(self):
        # ``eval`` walks a descending axis in reverse and returns the same
        # values, so the spacing check must see the same |diff| too.
        down = self._descending(self._field(dr=25.0, dz=1.0))
        with pytest.warns(UserWarning, match='range samples are'):
            down.eval(range=1050.0)

    def test_orientation_leaves_the_eval_values_bit_identical(self):
        up = self._field(dr=25.0, dz=1.0)
        down = self._descending(up)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = up.eval(range=1050.0)
            b = down.eval(range=1050.0)
        np.testing.assert_array_equal(a.data, b.data)


class TestAPhaseViewIsGuardedAtNyquistNotAtTheInterpolationBound:
    """A phase map drawn on a grid coarser than half a wavelength.

    Nothing is interpolated here and no level is biased: the heatmap draws
    flat cells and a line cut joins stored samples. What goes wrong is that
    an aliased phase reads as smooth large-scale structure rather than as
    noise, so the picture does not look wrong. Measured on the 100 m Pekeris
    guide at 200 Hz over 1000-1300 m: at dr = 15 m the panel draws broad
    diagonal bands that are purely an artefact of the grid, while dr = 0.94 m
    over the same window draws the one-wrap-per-wavelength fringes the field
    has. The two agree exactly at the ranges they share, so the field is
    right and only the view is wrong.

    The bound is the HALF wavelength, deliberately not the quarter wavelength
    ``_warn_if_undersampled`` uses, and one test below pins the two apart:
    that guard asks whether uacpy may interpolate between samples, this one
    asks whether the samples resolve the carrier at all.
    """

    HALF = 3.75          # half wavelength at 200 Hz, c = 1500 m/s
    QUARTER = 1.875      # what the INTERPOLATION guard uses, for contrast

    @staticmethod
    def _coherent_field(dr, dz=1.0, f0=200.0, c=1500.0):
        """Complex pressure, carrier e^{ikr}, grid spaced ``dz`` x ``dr``."""
        ranges = np.arange(1000.0, 1300.0 + dr, dr)
        depths = np.arange(40.0, 60.0 + dz, dz)
        k = 2.0 * np.pi * f0 / c
        data = np.exp(1j * k * ranges)[None, :] / ranges[None, :]
        return Field(
            data=np.repeat(data, depths.size, axis=0),
            coords={'depth': depths, 'range': ranges},
            model='Test', frequencies=f0,
        )

    @pytest.mark.parametrize('dr,aliased', [
        (3.70, False),   # just inside Nyquist — coarse but unambiguous
        (3.75, True),    # Nyquist exactly: +pi and -pi are one wrapped value
        (3.80, True),    # past it
    ])
    def test_nyquist_is_the_first_aliased_spacing_not_the_last_good_one(
            self, dr, aliased):
        """Both sides of the bound, and the bound itself.

        At exactly half a wavelength the carrier advances exactly pi between
        samples and the direction of rotation is already unrecoverable, so
        the comparison is ``>=``. A guard written ``>`` passes every test
        that brackets the bound loosely and stays silent on the one grid
        where the ambiguity is exact."""
        field = self._coherent_field(dr)
        if aliased:
            with pytest.warns(UserWarning, match='this view is aliased'):
                field._warn_if_phase_view_aliases('test')
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                field._warn_if_phase_view_aliases('test')

    def test_a_grid_between_the_two_bounds_warns_from_one_guard_only(self):
        """The half wavelength and the quarter wavelength are different
        numbers for different questions, and a later tidy-up that shares one
        constant between the two guards would be caught here.

        At 3.0 m the samples resolve the carrier (under half a wavelength,
        so the phase map is honest) but interpolating between them cuts
        across an opposite-phase lobe (over a quarter), so ``resample_to``
        must still object while the phase view does not."""
        assert self.QUARTER < 3.0 < self.HALF
        field = self._coherent_field(3.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            field._warn_if_phase_view_aliases('test')
        with pytest.warns(UserWarning, match='quarter wavelength'):
            field._warn_if_undersampled('test')

    @pytest.mark.parametrize('dr,dz,named,silent_axis', [
        (4.0, 3.5, 'range', 'depth'),
        (3.5, 4.0, 'depth', 'range'),
    ])
    def test_each_spatial_axis_is_judged_on_its_own(self, dr, dz, named,
                                                    silent_axis):
        """The pair brackets 3.75 m per axis: the named one coarse at 4.0 m,
        the other resolved at 3.5 m. A guard that read only the range axis
        would draw an aliased depth structure in silence."""
        field = self._coherent_field(dr, dz)
        with pytest.warns(UserWarning, match=f'{named} samples are') as rec:
            field._warn_if_phase_view_aliases('test')
        assert f'{silent_axis} samples are' not in str(rec[0].message)

    def test_a_field_with_no_frequency_says_so_rather_than_passing(self):
        """A silence that reads as a pass is worse than an admission: without
        a frequency there is no wavelength to judge the grid by, and the view
        may still be aliased."""
        field = Field(
            data=np.exp(1j * np.linspace(0.0, 40.0, 30))[None, :],
            coords={'depth': np.array([50.0]),
                    'range': np.linspace(1000.0, 1300.0, 30)},
            model='Test',
        )
        with pytest.warns(UserWarning, match='cannot be checked'):
            field._warn_if_phase_view_aliases('test')

    def test_a_frequency_axis_is_left_to_its_own_guard(self):
        """``plot_transfer_function`` draws a phase panel along frequency on
        purpose. That axis carries the same carrier but against range rather
        than wavelength, so it has its own limit c/(4r); this guard firing on
        it would report a metres-per-sample bound for an axis measured in
        hertz."""
        freqs = np.linspace(150.0, 450.0, 61)
        field = Field(
            data=np.exp(-2j * np.pi * freqs * 2.0),
            coords={'frequency': freqs},
            model='Test', frequencies=freqs,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            field._warn_if_phase_view_aliases('test')

    def test_a_real_field_carries_no_carrier_to_alias(self):
        field = self._coherent_field(15.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            field.to_dB()._warn_if_phase_view_aliases('test')

    @pytest.mark.parametrize('value,guarded', [
        ('phase', True), ('real', True), ('imag', True),
        ('dB', False), ('magnitude', False),
    ])
    def test_plot_field_guards_the_carrier_views_and_leaves_the_envelope(
            self, value, guarded):
        """``|p|`` varies on the interference scale, not on the carrier, so
        the same coarse field plots a degraded but honest level panel. Only
        the views that draw the carrier itself are guarded."""
        import matplotlib.pyplot as plt
        field = self._coherent_field(15.0)
        fig, ax = plt.subplots()
        try:
            with recorded_warnings() as rec:
                uacpy.plot.plot_field(field, ax=ax, value=value)
            fired = [w for w in rec
                     if 'this view is aliased' in str(w.message)]
            assert bool(fired) is guarded
        finally:
            plt.close(fig)

    def test_compare_guards_a_one_dimensional_phase_cut_too(self):
        """``compare`` draws its own lines instead of calling ``plot_field``,
        so the guard has to be wired into both doors. A cut along a coarse
        range axis aliases exactly as the map does."""
        import matplotlib.pyplot as plt
        field = self._coherent_field(15.0).at(depth=50.0)
        fig, ax = plt.subplots()
        try:
            with pytest.warns(UserWarning, match='this view is aliased'):
                uacpy.plot.compare([field], ax=ax, value='phase')
        finally:
            plt.close(fig)


class TestFrequencyUndersamplingGuardCoversBothRangeSpellings:
    """The frequency guard's quarter-cycle limit c/(4r) needs the field's
    range, which lives on the ``'range'`` coord for a full-grid eval but in
    ``pinned`` after ``.at(depth=…, range=…)`` collapses the axis — the
    canonical single-cell spectrum interpolates the identical carrier, so
    both spellings must warn. A descending frequency axis stores the same
    bins as an ascending one and returns the same values, so orientation
    must not silence the guard either."""

    @staticmethod
    def _transfer_field(ascending=True):
        depths = np.array([40.0, 50.0])
        ranges = np.array([5.0, 5000.0])
        freqs = np.arange(100.0, 111.0, 1.0)     # 1 Hz bins
        if not ascending:
            freqs = freqs[::-1]
        f = freqs[None, None, :]
        r = ranges[None, :, None]
        data = np.exp(-2j * np.pi * f * r / 1500.0) / r
        return Field(data=np.repeat(data, depths.size, axis=0),
                     coords={'depth': depths, 'range': ranges,
                             'frequency': freqs},
                     model='Test', frequencies=np.sort(freqs))

    def test_full_grid_eval_warns_on_a_coarse_frequency_axis(self):
        # 1 Hz bins against c/(4·5000 m) = 0.075 Hz.
        field = self._transfer_field()
        with pytest.warns(UserWarning, match='frequency samples are'):
            field.eval(frequency=105.5)

    def test_a_range_collapsed_cell_warns_like_the_full_grid(self):
        cell = self._transfer_field().at(depth=45.0, range=5000.0)
        assert 'range' not in cell.coords
        assert cell.pinned['range'] == pytest.approx(5000.0)
        with pytest.warns(UserWarning, match='frequency samples are'):
            cell.eval(frequency=105.5)

    def test_a_collapsed_cell_at_close_range_stays_silent(self):
        # The discriminating half: at the cell's own r = 5 m the limit is
        # c/(4·5 m) = 75 Hz, so 1 Hz bins are finely sampled — the far end
        # of the collapsed axis must not decide for this cell.
        cell = self._transfer_field().at(depth=45.0, range=5.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            cell.eval(frequency=105.5)

    def test_a_descending_frequency_axis_warns_like_ascending(self):
        field = self._transfer_field(ascending=False)
        with pytest.warns(UserWarning, match='frequency samples are'):
            field.eval(frequency=105.5)

    def test_orientation_leaves_the_spectrum_values_bit_identical(self):
        up = self._transfer_field()
        down = self._transfer_field(ascending=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = up.eval(frequency=105.5)
            b = down.eval(frequency=105.5)
        np.testing.assert_array_equal(a.data, b.data)


class TestKindUnitAndDtypeAreIndependentAxes:
    """A Field is described on three axes that must not be collapsed into one:
    ``kind`` (what it is), ``unit`` (what it is measured in) and ``dtype`` (how
    it is stored). Each has a distinct consumer, and each collapse has already
    produced a bug:

    * ``Field.max`` inferred dB-ness from the quantity, so introducing
      ``'reverberation'`` made it return the *quietest* cell of a dB grid.
    * ``compare_models`` keyed on representation, so it refused the ordinary
      RAM-vs-Kraken comparison of one quantity written two ways.
    * ``replica_bank_from_field`` asked for a quantity when what matched-field
      processing actually needs is phase, i.e. the dtype.

    A fourth axis is *not* modelled here and must be read off the producer:
    which direction a dB quantity runs. ``kind='reverberation'`` sounds like a
    level and is a loss, because OASES writes ``-10*log10 E[|p_scat|^2]``.
    Reading the name instead of the engine made ``max()`` return the quietest
    cell; the two losses are enumerated in :meth:`Field.max` for that reason.
    """

    Z = np.array([10.0, 20.0])
    R = np.array([100.0, 200.0])
    DB = np.array([[40.0, 90.0], [70.0, 60.0]])      # 40 dB is the loudest

    def _field(self, data=None, metadata=None, coords=None, **quantity):
        return Field(data=self.DB if data is None else data,
                     coords=coords or {'depth': self.Z, 'range': self.R},
                     model='Test', frequencies=100.0, metadata=metadata,
                     **quantity)

    # ── kind: the quantity ────────────────────────────────────────────
    def test_transmission_loss_is_not_a_separate_kind(self):
        # TL is pressure in dB, so it must share pressure's kind and differ
        # only on the unit axis. This is what lets example_07 compare a RAM
        # TL field against a Kraken complex one.
        tl, px = self._field(), self._field(data=self.DB.astype(complex))
        assert tl.kind == px.kind == 'pressure'
        assert (tl.unit, px.unit) == ('dB', 'Pa')

    def test_a_model_tags_a_different_quantity(self):
        assert self._field(kind='reverberation').kind == \
            'reverberation'

    # ── unit: which way is louder ─────────────────────────────────────
    @pytest.mark.parametrize('kind, expected_max', [
        (None, 40.0),               # pressure in dB: transmission loss
        ('reverberation', 40.0),
    ], ids=['transmission_loss', 'reverberation'])
    def test_a_dB_loss_inverts(self, kind, expected_max):
        """Both dB *losses* run backwards, so the least of either is loudest.

        Reverberation belongs here because that is what the vendored engine
        writes. ``REVINT`` (``oassun26.f:853-858``, the routine option ``'r'``
        reaches) builds an intensity, squares it with ``CVMAGS``, takes
        ``VALG10`` and scales by ``VSMUL(-5E0)``, giving
        ``-10*log10 E[|p_scat|^2]``. The leading minus is the whole point: a
        larger stored number is a *weaker* scattered field, which is why the
        model tags it ``oass_quantity='reverberation_loss_dB'``. Read as a
        level, ``max()`` returned the quietest cell of the grid.

        **A bound stated on the level is the opposite bound on this array.**
        ``oass.tex:163-166`` says option ``p`` (re-scattering included)
        "Yields lower bound for reverb levels" and the default "yields upper
        bound" — so on the stored loss those are the *upper* and *lower*
        bounds respectively, which is why
        ``test_oass.py::test_multiple_scattering_lowers_the_reverberation_level``
        asserts ``multiple.data > single.data``. Any inequality about
        reverberation has to be restated when it crosses this sign, and a
        word swap from "level" to "loss" does not do that."""
        f = self._field(kind=kind)
        assert f.unit == 'dB'
        assert float(f.max().data) == pytest.approx(expected_max)

    def test_a_dB_level_is_not_inverted(self):
        # Signal excess shares the dB unit but is a level, not a loss: more
        # is more. Deciding direction from the unit alone reports the
        # *weakest* cell of a level grid as the strongest.
        f = self._field(kind='signal_excess')
        assert f.unit == 'dB'
        assert float(f.max().data) == pytest.approx(90.0)

    def test_reverberation_max_returns_the_strongest_scattering_not_the_last_range(self):
        """The shape the bug actually took, on a field shaped like an OASS
        run: the loss grows monotonically with range, so reading it as a
        level made ``max()`` a synonym for "the far end of the grid" — and
        the method's own docstring says it slices at the loudest point."""
        ranges = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
        loss = np.array([40.0, 52.3, 62.1, 69.0, 73.98])
        f = Field(data=loss, coords={'range': ranges}, model='Test',
                  frequencies=100.0, kind='reverberation')
        peak = f.max()
        assert float(peak.data) == pytest.approx(loss.min())
        assert peak.pinned['range'] == pytest.approx(1000.0)

    def test_the_reverberation_axis_label_says_loss(self):
        # The label is where a reader of the plot learns the direction, and
        # it said 'level' while the numbers ran the other way.
        from uacpy.core.results.quantities import quantity
        label = quantity('reverberation').units['dB']
        assert label == 'Reverberation loss (dB re unit source)'
        assert 'level' not in label.lower()

    def test_a_time_trace_is_linear_not_dB(self):
        # Real data alone does not mean dB — a time trace is Pa, and treating
        # it as a level would make max() return the trace's *trough*.
        t = np.array([0.0, 1.0, 2.0, 3.0])
        f = self._field(data=np.array([1.0, -5.0, 2.0, 0.5]),
                        coords={'time': t})
        assert f.kind == 'pressure' and f.unit == 'Pa'
        assert float(f.max().data) == pytest.approx(-5.0)   # largest |p|

    def test_a_model_may_pin_the_unit(self):
        # probability_of_detection is dimensionless, so it cannot be derived
        # from the storage the way pressure's Pa/dB split is.
        f = self._field(kind='probability_of_detection', unit='1')
        assert (f.kind, f.unit) == ('probability_of_detection', '1')

    def test_an_unregistered_quantity_is_refused_at_construction(self):
        # The guard is on the funnel: a typo'd tag that survived construction
        # would resurface as a wrong colour scale or a wrong argmax direction
        # with nothing pointing back at the model that set it.
        with pytest.raises(ConfigurationError, match='unknown Field kind'):
            self._field(kind='transmission_loss')
        with pytest.raises(ConfigurationError, match='not measured in'):
            self._field(kind='reverberation', unit='Pa')

    def test_the_untagged_path_never_consults_the_registry(self):
        # Every slice builds a Field; validation must not cost the common
        # case, so an untagged field carries no 'kind'/'unit' metadata at all.
        f = self._field()
        assert 'kind' not in (f.metadata or {})
        assert (f.kind, f.unit) == ('pressure', 'dB')

    # ── dtype: is there phase to work with ────────────────────────────
    def test_matched_field_rejects_a_real_field_of_the_right_kind(self):
        from uacpy.sonar.matched_field import replica_bank_from_field
        tl = self._field()                      # kind='pressure', but real
        assert tl.kind == 'pressure'            # would pass a kind-only guard
        with pytest.raises(ConfigurationError, match='complex'):
            replica_bank_from_field(tl)

    # ── the axes stay independent ─────────────────────────────────────
    def test_compare_models_keys_on_kind_not_representation(self):
        import matplotlib
        matplotlib.use('Agg')
        from uacpy.plot import compare_models
        tl = self._field()
        px = self._field(data=self.DB.astype(complex))
        rv = self._field(kind='reverberation')
        # One quantity, two representations: the ordinary comparison.
        compare_models([tl, px])
        # Same representation, different quantity: refused.
        with pytest.raises(ConfigurationError, match='different physical'):
            compare_models([tl, rv])

    def test_all_three_axes_round_trip(self):
        f = self._field(kind='reverberation')
        d = f.to_dict()
        assert (d['kind'], d['unit']) == ('reverberation', 'dB')
        back = Field.from_dict(d)
        assert (back.kind, back.unit, back.data.dtype) == \
            (f.kind, f.unit, f.data.dtype)


class TestMaskBelowSeafloor:
    """:meth:`Field.mask_below_seafloor` NaN-masks samples strictly below
    the (range-interpolated) seafloor and returns a copy; a sample exactly
    on the seafloor is kept."""

    @staticmethod
    def _field():
        return Field(
            data=np.ones((3, 2)),
            coords={'depth': np.array([50.0, 120.0, 150.0]),
                    'range': np.array([0.0, 1000.0])},
            model='Test', frequencies=100.0)

    def test_masks_only_cells_below_the_local_seafloor(self):
        f = self._field()
        # Sloping seafloor: 100 m at r=0, 140 m at r=1000. Column r=0
        # loses 120 and 150 m; column r=1000 loses only 150 m.
        masked = f.mask_below_seafloor([(0.0, 100.0), (1000.0, 140.0)])
        expected = np.array([[1.0, 1.0],
                             [np.nan, 1.0],
                             [np.nan, np.nan]])
        np.testing.assert_array_equal(masked.data, expected)
        # The parent field is untouched (mask returns a copy).
        assert np.isfinite(f.data).all()

    def test_boundary_cell_on_the_seafloor_is_kept(self):
        # The mask is strict (depth > seafloor), so a receiver exactly on
        # the interface keeps its value.
        f = Field(
            data=np.ones((2, 1)),
            coords={'depth': np.array([100.0, 100.5]),
                    'range': np.array([500.0])},
            model='Test', frequencies=100.0)
        masked = f.mask_below_seafloor([(0.0, 100.0), (1000.0, 100.0)])
        assert masked.data[0, 0] == 1.0          # exactly on the interface
        assert np.isnan(masked.data[1, 0])       # half a metre below

    def test_requires_depth_and_range_as_the_first_axes(self):
        trace = Field(data=np.zeros(4),
                      coords={'time': np.arange(4) * 0.1}, model='Test')
        with pytest.raises(ConfigurationError,
                           match="'depth' and 'range' as the first two axes"):
            trace.mask_below_seafloor([(0.0, 100.0), (1000.0, 100.0)])

    def test_a_trailing_axis_takes_the_mask_of_its_cell(self):
        f = Field(data=np.ones((3, 2, 4)),
                  coords={'depth': np.array([50.0, 120.0, 150.0]),
                          'range': np.array([0.0, 1000.0]),
                          'frequency': np.arange(4) + 100.0},
                  model='Test')
        masked = f.mask_below_seafloor([(0.0, 100.0), (1000.0, 140.0)])
        two_d = self._field().mask_below_seafloor(
            [(0.0, 100.0), (1000.0, 140.0)])
        for i in range(4):
            np.testing.assert_array_equal(masked.data[..., i], two_d.data)


class TestSpectrumAndToneExtraction:
    """`to_transfer_function` (rFFT·dt on the record's bins) and
    `extract_tone` (windowed phasor estimate) on a synthetic pure-tone time
    Field."""

    A0 = 2.5
    PHI0 = 0.7
    F0 = 32.0
    N = 256

    def _tone_field(self):
        # dt = 1/256 s puts every integer frequency exactly on an rFFT bin.
        t = np.arange(self.N) / 256.0
        p = self.A0 * np.cos(2 * np.pi * self.F0 * t + self.PHI0)
        return Field(
            data=p.reshape(1, 1, self.N),
            coords={'depth': np.array([50.0]),
                    'range': np.array([1000.0]),
                    'time': t},
            model='Test')

    def test_the_transfer_function_peaks_at_the_tone_bin(self):
        H = self._tone_field().to_transfer_function()
        X = np.asarray(H.data)
        k = int(np.argmax(np.abs(X[0, 0])))
        assert H.coords['frequency'][k] == self.F0      # exact-bin tone
        # rFFT of A0·cos(2πft+φ0) at the tone bin is (N/2)·A0·e^{iφ0}; the
        # transfer function carries it times dt (t0 = 0: no rotation).
        assert X[0, 0, k] == pytest.approx(
            (self.N / 2) * self.A0 * np.exp(1j * self.PHI0) / 256.0, rel=1e-9)

    def test_extract_tone_recovers_the_complex_amplitude(self):
        tone = self._tone_field().extract_tone(self.F0)
        assert list(tone.coords) == ['depth', 'range']
        assert tone.pinned['frequency'] == self.F0
        np.testing.assert_array_equal(tone.frequencies, [self.F0])
        # Phasor convention p(t) = Re{A·e^{+2πift}} → A = A0·e^{+iφ0}.
        # rel=1e-5: the symmetric (non-periodic) np.hanning taper leaks
        # ~8e-7 of the negative-frequency image into the tone bin
        # (measured); a periodic window would make this exact.
        assert complex(tone.data[0, 0]) == pytest.approx(
            self.A0 * np.exp(1j * self.PHI0), rel=1e-5)

    def test_a_single_time_trace_gives_the_same_phasor(self):
        # The (time,) trace a receiver slice leaves: every non-time axis is
        # kept, so this gives a 0-d phasor equal to the grid's one cell.
        grid = self._tone_field()
        trace = grid.isel(depth=0, range=0)
        assert list(trace.coords) == ['time']
        tone = trace.extract_tone(self.F0)
        assert tone.data.shape == () and list(tone.coords) == []
        assert complex(tone.data) == complex(grid.extract_tone(self.F0).data[0, 0])

    def test_a_field_without_time_is_refused(self):
        with pytest.raises(ConfigurationError, match="'time' axis"):
            self._tone_field().extract_tone(self.F0).extract_tone(self.F0)


class TestFieldDomainAccessorContracts:
    """The documented unit/domain guards on the value accessors
    (docs/guide/results.md §4): ``.dB`` refuses a time-domain trace,
    ``.p`` refuses real data, and ``.dt``/``.sample_rate`` read 0.0 when
    no time axis exists."""

    def test_dB_raises_on_a_time_domain_field(self):
        trace = Field(data=np.zeros((1, 1, 8)),
                      coords={'depth': np.array([10.0]),
                              'range': np.array([100.0]),
                              'time': np.arange(8) * 0.01},
                      model='Test')
        with pytest.raises(AttributeError, match='time-domain'):
            trace.dB

    def test_p_raises_on_a_real_dB_field(self):
        tl = Field(data=np.array([[60.0]]),
                   coords={'depth': np.array([10.0]),
                           'range': np.array([100.0])},
                   model='Test', frequencies=100.0)
        with pytest.raises(AttributeError, match='data is real'):
            tl.p

    def test_dt_and_sample_rate_are_zero_without_a_time_axis(self):
        tl = Field(data=np.array([[60.0]]),
                   coords={'depth': np.array([10.0]),
                           'range': np.array([100.0])},
                   model='Test', frequencies=100.0)
        assert tl.dt == 0.0
        assert tl.sample_rate == 0.0

    def test_dt_and_sample_rate_read_the_time_axis(self):
        trace = Field(data=np.zeros((1, 1, 50)),
                      coords={'depth': np.array([10.0]),
                              'range': np.array([100.0]),
                              'time': np.arange(50) * 0.01},
                      model='Test')
        assert trace.dt == pytest.approx(0.01)
        assert trace.sample_rate == pytest.approx(100.0)


class TestInterpolatedSlicesRequireAMonotonicAxis:
    """``eval`` brackets its query by binary search over the coordinate
    vector, which is only meaningful on a monotonic axis: ascending and
    descending axes interpolate, an interleaved one is refused by name."""

    def _field(self, depths):
        depths = np.asarray(depths, dtype=float)
        ranges = np.array([0.0, 100.0])
        data = np.repeat(depths[:, None], ranges.size, axis=1)
        return Field(data=data, coords={'depth': depths, 'range': ranges},
                     model='Test')

    def test_a_non_monotonic_depth_axis_refuses_an_interpolated_slice(self):
        f = self._field([0.0, 50.0, 25.0, 75.0])
        with pytest.raises(ConfigurationError, match="'depth'"):
            f.eval(depth=30.0)

    def test_an_ascending_axis_interpolates(self):
        f = self._field([0.0, 25.0, 50.0, 75.0])
        got = f.eval(depth=37.5)
        np.testing.assert_allclose(got.data, 37.5)

    def test_a_descending_axis_interpolates(self):
        f = self._field([75.0, 50.0, 25.0, 0.0])
        got = f.eval(depth=37.5)
        np.testing.assert_allclose(got.data, 37.5)


class TestFieldCoordVectorsMustBeFinite:
    """A coord vector is rejected at construction if any element is NaN or
    inf: a non-finite coordinate turns every ``|axis - label|`` distance at
    that sample into NaN, so ``at()``'s argmin can land on it and return a
    sample no label ever named."""

    def _coords(self, depth_axis):
        return {'depth': np.asarray(depth_axis, dtype=float),
                'range': np.array([100.0, 200.0])}

    def test_nan_coordinate_is_rejected_at_construction(self):
        with pytest.raises(ConfigurationError,
                           match=r"Field\.coords\['depth'\] must be finite"):
            Field(data=np.zeros((3, 2)),
                  coords=self._coords([0.0, np.nan, 20.0]))

    def test_inf_coordinate_is_rejected_at_construction(self):
        with pytest.raises(ConfigurationError,
                           match=r"Field\.coords\['depth'\] must be finite"):
            Field(data=np.zeros((3, 2)),
                  coords=self._coords([0.0, np.inf, 20.0]))

    def test_finite_coords_construct_and_at_picks_the_nearest_sample(self):
        f = Field(data=np.arange(6.0).reshape(3, 2),
                  coords=self._coords([0.0, 10.0, 20.0]))
        assert f.at(depth=19.0).pinned['depth'] == pytest.approx(20.0)


class TestSynthesisWarnsWhenNoSpeedStamped:
    """A synthesis window anchored with no producer-stamped speed
    (no :attr:`Field.speeds` record) warns that the
    1500 m/s default is the anchor whenever the anchor delays the window
    start. The geometry here keeps the fast/slow-spread heuristic below
    its own threshold (5 % of the travel time is under the 0.44 s lead
    the 9 Hz band sets), so the warning is pinned to the unstamped-speed
    rule alone."""

    def _broadband(self, speeds):
        freqs = np.linspace(50.0, 59.0, 10)
        return Field(data=np.ones((1, 1, 10), dtype=complex),
                     coords={'depth': np.array([50.0]),
                             'range': np.array([3000.0]),
                             'frequency': freqs},
                     speeds=speeds)

    def _trace_warnings(self, speeds, **kwargs):
        with recorded_warnings() as caught:
            trace = self._broadband(speeds).to_time_trace(
                depth=50.0, range=3000.0, **kwargs)
        return trace, [str(w.message) for w in caught
                       if 'to_time_trace' in str(w.message)]

    def test_unstamped_speed_warns_of_the_default_anchor(self):
        _, messages = self._trace_warnings(None)
        assert len(messages) == 1
        assert "stamped no sound speed" in messages[0]
        assert "1500" in messages[0]

    def test_a_stated_water_max_anchors_silently(self):
        trace, messages = self._trace_warnings(SoundSpeeds(water_max=3000.0))
        assert messages == []
        # 50-59 Hz: the 4/B onset term, 0.444 s, sets the lead.
        assert float(trace.coords['time'][0]) == pytest.approx(1.0 - 4.0 / 9.0)

    def test_a_stated_surface_speed_carries_no_unstamped_speed_warning(self):
        _, messages = self._trace_warnings(SoundSpeeds(surface=1500.0))
        assert all("stamped no sound speed" not in m for m in messages)

    def test_explicit_t_start_silences_the_anchor_warning(self):
        trace, messages = self._trace_warnings(None, t_start=1.0)
        assert messages == []
        assert float(trace.coords['time'][0]) == pytest.approx(1.0)


def _profile_1d():
    return SoundSpeedProfile.from_pairs([[0.0, 1500.0], [100.0, 1490.0]])


class TestSliceLabelsMustBeFiniteScalars:
    """``collapse_axis`` rejects NaN/inf and array-valued labels with a
    typed error, on every carrier that routes ``at``/``eval`` through it.
    A NaN label makes every ``|axis - label|`` distance NaN, so a nearest
    lookup would fall to index 0 — a real sample — and an interpolated
    slice would propagate NaN into the result, whose own validators then
    blame the data."""

    def test_ssp_at_nan_depth_raises_a_typed_label_error(self):
        with pytest.raises(ConfigurationError,
                           match="depth=nan is not a finite label"):
            _profile_1d().at(depth=np.nan)

    def test_ssp_eval_nan_depth_blames_the_label_not_the_data(self):
        with pytest.raises(ConfigurationError,
                           match='depth=nan is not a finite label') as exc:
            _profile_1d().eval(depth=float('nan'))
        assert "not a finite label" in str(exc.value)
        assert "sound speeds must be finite" not in str(exc.value)

    def test_ssp_eval_array_depth_raises_a_typed_scalar_label_error(self):
        with pytest.raises(ConfigurationError,
                           match="is not a scalar label"):
            _profile_1d().eval(depth=[10.0, 20.0])

    def test_ssp_eval_nan_range_on_a_1d_profile_raises(self):
        with pytest.raises(ConfigurationError,
                           match="range=nan is not a finite label"):
            _profile_1d().eval(range=np.nan)

    def test_field_eval_inf_label_raises_a_typed_label_error(self):
        f = Field(data=np.arange(6.0).reshape(3, 2),
                  coords={'depth': np.array([0.0, 10.0, 20.0]),
                          'range': np.array([100.0, 200.0])})
        with pytest.raises(ConfigurationError,
                           match="depth=inf is not a finite label"):
            f.eval(depth=np.inf)

    def test_reflection_at_nan_angle_raises_a_typed_label_error(self):
        rc = ReflectionCoefficient(
            angles=np.linspace(0.0, 90.0, 5),
            magnitude=np.linspace(0.0, 1.0, 5),
            phase=np.zeros(5),
            model='Test', frequencies=100.0,
        )
        with pytest.raises(ConfigurationError,
                           match="angle=nan is not a finite label"):
            rc.at(angle=np.nan)

    def test_finite_scalar_labels_slice_nearest_and_interpolated(self):
        ssp = _profile_1d()
        assert float(ssp.at(depth=99.0).depths[0]) == pytest.approx(100.0)
        assert ssp.eval(depth=50.0).value == pytest.approx(1495.0)


class TestMaskBelowSeafloorValidatesTheRangeAxis:
    """``mask_below_seafloor`` hands its bathymetry to ``np.interp``, which
    takes ``xp`` on trust. A raw ``(N, 2)`` array skipped the check the
    ``Bathymetry`` form gets, so a profile whose range column does not
    increase interpolated against a broken axis and masked the wrong cells
    with no error: the same two-point profile masked 24 cells sorted and 28
    reversed."""

    def test_a_reversed_range_column_is_refused(self):
        rows = np.array([[4000.0, 150.0], [0.0, 100.0]])
        with pytest.raises(ConfigurationError, match='strictly increasing'):
            _field().mask_below_seafloor(rows)

    def test_a_repeated_range_is_refused(self):
        rows = np.array([[0.0, 100.0], [0.0, 150.0]])
        with pytest.raises(ConfigurationError, match='strictly increasing'):
            _field().mask_below_seafloor(rows)

    def test_the_shape_error_comes_from_the_field_method(self):
        """The (N, 2) shape check stays where the caller can see which
        argument it is about, ahead of the bathymetry axis check."""
        with pytest.raises(ConfigurationError, match='mask_below_seafloor'):
            _field().mask_below_seafloor(np.ones((3, 3)))

    def test_a_sorted_profile_masks_what_it_did_before(self):
        f = _field()
        rows = [(0.0, 15.0), (300.0, 25.0)]
        masked = np.asarray(f.mask_below_seafloor(rows).data)
        seafloor = np.interp(f.coords['range'], [0.0, 300.0], [15.0, 25.0])
        want = f.coords['depth'][:, None] > seafloor[None, :]
        np.testing.assert_array_equal(np.isnan(masked), want)

    def test_the_bathymetry_carrier_form_masks_the_same_cells_as_an_array(self):
        from uacpy.core.bathymetry import Bathymetry
        rows = np.array([[0.0, 15.0], [300.0, 25.0]])
        via_array = np.asarray(_field().mask_below_seafloor(rows).data)
        via_carrier = np.asarray(
            _field().mask_below_seafloor(Bathymetry.coerce(rows)).data)
        np.testing.assert_array_equal(np.isnan(via_array),
                                      np.isnan(via_carrier))


class TestFieldAtRejectsALabelItCannotRank:
    """``Field.at`` picks ``argmin(|coord - label|)``. A NaN or inf label
    makes every distance NaN or inf and argmin falls to index 0; a label
    large enough to absorb the whole axis (``|z - 1e300|`` rounds to 1e300
    for every z) ties every sample and does the same. Index 0 is a real
    sample, so both read as a successful slice at the first coordinate."""

    def test_a_nan_label_is_refused(self):
        with pytest.raises(ConfigurationError, match='not a finite label'):
            _field().at(depth=float('nan'))

    @pytest.mark.parametrize('label', [float('inf'), float('-inf')])
    def test_an_infinite_label_is_refused(self, label):
        with pytest.raises(ConfigurationError, match='not a finite label'):
            _field().at(range=label)

    @pytest.mark.parametrize('label', [1e300, -1e300])
    def test_a_label_that_absorbs_the_axis_is_refused(self, label):
        with pytest.raises(ConfigurationError, match='same distance'):
            _field().at(depth=label)

    def test_an_ordinary_out_of_range_label_clamps_to_the_nearest(self):
        """Outside the axis but still rankable is the documented
        nearest-sample behaviour and stays."""
        assert _field().at(depth=-5.0).pinned['depth'] == pytest.approx(0.0)
        assert _field().at(depth=999.0).pinned['depth'] == pytest.approx(30.0)

    @pytest.mark.parametrize('label, warns', [
        (35.0, False), (35.1, True),     # top edge 30 m, half step 5 m
        (-5.0, False), (-5.1, True),     # bottom edge 0 m, half step 5 m
    ])
    def test_a_label_past_half_the_edge_step_warns(self, label, warns):
        with recorded_warnings() as caught:
            _field().at(depth=label)
        off_axis = [w for w in caught if 'outside the' in str(w.message)]
        assert bool(off_axis) is warns

    def test_a_genuine_midpoint_tie_is_answered(self):
        """Two samples exactly equidistant is a real tie, not a lost axis:
        the first wins, as argmin has always done."""
        f = Field(data=np.ones((2, 3)),
                  coords={'depth': np.array([0.0, 10.0]),
                          'range': np.array([100.0, 200.0, 300.0])},
                  model='Test')
        assert f.at(depth=5.0).pinned['depth'] == pytest.approx(0.0)

    def test_a_single_sample_axis_takes_any_finite_label(self):
        """One sample cannot tie with another, so there is nothing to lose."""
        f = Field(data=np.ones((1, 3)),
                  coords={'depth': np.array([7.0]),
                          'range': np.array([100.0, 200.0, 300.0])},
                  model='Test')
        assert f.at(depth=1e300).pinned['depth'] == pytest.approx(7.0)


class TestSlicingAnEmptyAxisIsRefused:
    """The constructor admits an axis of size 0 (an axis sliced to nothing
    is a supported state of ``coords``), but no sample on it can be picked:
    ``at``, ``isel`` and ``eval`` refuse it with one typed error naming the
    axis, in place of numpy's bare ``ValueError`` / ``IndexError``."""

    @staticmethod
    def _empty_depth():
        return Field(data=np.zeros((0, 3)),
                     coords={'depth': np.array([]),
                             'range': np.array([100.0, 200.0, 300.0])},
                     model='Test')

    @pytest.mark.parametrize('slicer', [
        lambda f: f.at(depth=0.0),
        lambda f: f.isel(depth=0),
        lambda f: f.eval(depth=0.0),
    ], ids=['at', 'isel', 'eval'])
    def test_an_empty_axis_raises_a_typed_error_naming_it(self, slicer):
        with pytest.raises(ConfigurationError, match="'depth' has no samples"):
            slicer(self._empty_depth())

    def test_a_populated_axis_beside_an_empty_one_slices(self):
        """The guard is per named axis: the populated one stays usable."""
        out = self._empty_depth().at(range=200.0)
        assert out.pinned['range'] == 200.0
        assert out.coords['depth'].size == 0


class TestToDbRewritesTheUnitTag:
    """``Field.unit`` describes the data, and ``to_dB`` replaces the
    data. Carrying a ``'Pa'`` unit onto ``-20·log10|p|`` left a dB field
    reporting Pa, which sends ``Field.max`` down its linear branch: it then
    ranks by ``|dB|``, where the largest magnitude is the *quietest* sample
    rather than the loudest."""

    AMP = np.array([[10.0, 0.5, 2.0], [3.0, 0.2, 8.0],
                    [1.0, 4.0, 0.1], [6.0, 0.05, 20.0]])

    def _tagged(self):
        return _field(data=self.AMP * (1.0 + 0j), unit='Pa')

    def test_the_tag_follows_the_data(self):
        assert self._tagged().to_dB().unit == 'dB'
        assert self._tagged().to_dB().unit == 'dB'

    def test_the_source_field_keeps_its_own_tag(self):
        f = self._tagged()
        f.to_dB()
        assert f.unit == 'Pa'

    def test_max_finds_the_same_point_tagged_or_not(self):
        tagged = self._tagged().to_dB().max().pinned
        untagged = _field(data=self.AMP * (1.0 + 0j)).to_dB().max().pinned
        assert tagged == untagged

    def test_max_finds_the_loudest_sample_and_not_the_quietest(self):
        loudest = self._tagged().to_dB().max().pinned
        i, j = np.unravel_index(int(np.argmax(self.AMP)), self.AMP.shape)
        assert loudest['depth'] == pytest.approx(
            float(_field().coords['depth'][i]))
        assert loudest['range'] == pytest.approx(
            float(_field().coords['range'][j]))

    def test_an_untagged_field_is_tagged_at_construction_and_retagged_db(self):
        # Construction stores the unit read off the storage; the conversion
        # rewrites it with the data.
        source = _field(data=self.AMP * (1.0 + 0j))
        assert source.unit == 'Pa'
        out = source.to_dB()
        assert out.unit == 'dB' and out.unit == 'dB'

    def test_a_real_field_is_returned_unchanged(self):
        f = _field(unit='dB')
        assert f.to_dB() is f


class TestDbRefusalOffersAnActionableRoute:
    """``Field.dB``'s unit guard is reachable only for real data — the complex
    branch returns first — and ``to_dB()`` returns ``self`` for every real
    field. So the set of fields that can see this message is exactly the set
    on which ``to_dB()`` does nothing, and naming it as the remedy sends the
    reader in a circle."""

    def _dimensionless(self):
        return _field(data=np.full((4, 3), 0.5),
                      kind='probability_of_detection', unit='1')

    def _linear_pressure(self):
        return _field(data=np.full((4, 3), 2.0),
                      kind='pressure', unit='Pa')

    @pytest.mark.parametrize('name', ['_dimensionless', '_linear_pressure'])
    def test_to_dB_is_the_identity_on_every_field_that_reaches_the_guard(
            self, name):
        f = getattr(self, name)()
        assert not f.is_complex
        assert f.unit != 'dB'
        assert f.to_dB() is f

    @pytest.mark.parametrize('name', ['_dimensionless', '_linear_pressure'])
    def test_the_message_names_an_operation_that_changes_the_values(self, name):
        f = getattr(self, name)()
        with pytest.raises(
                AttributeError,
                match='not dB, so its values are not a level') as excinfo:
            f.dB
        message = str(excinfo.value)
        assert 'not dB' in message
        assert 'log10' in message
        # Naming to_dB() is only honest alongside the fact that it is the
        # identity here.
        if 'to_dB()' in message:
            assert 'unchanged' in message

    def test_a_dB_tagged_real_field_is_the_other_side_of_the_guard(self):
        f = _field(data=np.full((4, 3), -60.0), unit='dB')
        assert np.allclose(f.dB, -60.0)

    def test_the_stack_message_names_an_operation_that_changes_the_values(self):
        slab = self._dimensionless()
        stack = ResultStack([slab, slab], np.array([5.0, 10.0]),
                            coordinate_name='source_depth')
        with pytest.raises(
                ConfigurationError,
                match='not dB, so their values are not a level') as excinfo:
            stack.dB
        message = str(excinfo.value)
        assert 'not dB' in message
        assert 'log10' in message
        if 'to_dB()' in message:
            assert 'unchanged' in message


class TestTlIsTheDbViewRestrictedToPressureFields:
    """``Field.tl`` answers with exactly ``Field.dB``'s values on a
    pressure-kind field — the quantity's literature name for the same
    array — and refuses any other kind, whose level view stays ``.dB``."""

    @staticmethod
    def _pressure_field():
        from uacpy.core.results.field import Field
        return Field(data=np.array([[0.01 + 0.001j, 0.002 + 0.0j]]),
                     coords={'depth': [10.0], 'range': [100.0, 200.0]},
                     model='Test')

    @staticmethod
    def _reverberation_field():
        from uacpy.core.results.field import Field
        return Field(data=np.array([[35.0, 30.0]]),
                     coords={'depth': [10.0], 'range': [100.0, 200.0]},
                     model='Test',
                     kind='reverberation', unit='dB')

    def test_tl_of_complex_pressure_equals_dB_and_is_a_positive_loss(self):
        f = self._pressure_field()
        assert np.array_equal(f.tl, f.dB)
        assert (f.tl > 0).all()

    def test_tl_of_a_real_dB_pressure_field_is_the_same_readonly_view(self):
        f = self._pressure_field()
        real = f.to_dB()
        assert real.tl.base is real.data or np.shares_memory(real.tl,
                                                             real.data)
        assert not real.tl.flags.writeable

    def test_tl_of_a_reverberation_field_refuses_and_names_dB(self):
        with pytest.raises(AttributeError, match="'reverberation'.*\\.dB"):
            self._reverberation_field().tl

    def test_stack_tl_matches_stack_dB_on_pressure_slabs(self):
        from uacpy.core.results.stack import ResultStack
        f = self._pressure_field()
        st = ResultStack(slabs=[f, f], coordinate=np.array([10.0, 20.0]),
                         coordinate_name='source_depth')
        assert np.array_equal(st.tl, st.dB)
        assert st.tl.shape == (2, 1, 2)

    def test_stack_tl_of_reverberation_slabs_refuses_and_names_stack_dB(self):
        from uacpy.core.results.stack import ResultStack
        rl = self._reverberation_field()
        st = ResultStack(slabs=[rl, rl], coordinate=np.array([10.0, 20.0]),
                         coordinate_name='source_depth')
        with pytest.raises(ConfigurationError, match='stack\\.dB'):
            st.tl


def test_the_dB_docstring_contrasts_itself_against_tl_not_against_itself():
    """``Field.dB``'s closing paragraph explains why the property is named for
    the unit: on a reverberation field it returns that level, and the
    quantity-named spelling would have mislabelled it. The sentence named
    ``.dB`` — the property it is written on — so it read as saying its own
    name was the misnomer. The property it means is ``.tl``, which exists and
    refuses every non-pressure kind for exactly this reason."""
    doc = ' '.join(Field.dB.__doc__.split())
    assert 'calling it ``.tl`` would have been the same misnomer' in doc
    assert 'calling it ``.dB``' not in doc
    # The claim is only true because the sibling really does refuse: a
    # reverberation field has a level in .dB and no .tl at all.
    rev = Field(data=np.array([60.0, 70.0]), coords={'range': [0.0, 10.0]},
                kind='reverberation', unit='dB')
    assert rev.dB.tolist() == [60.0, 70.0]
    with pytest.raises(AttributeError, match='not a transmission loss'):
        rev.tl


class TestFieldWindowAndShift:
    """``window`` narrows an axis and keeps it; ``shift`` moves its origin.

    ``at`` and ``isel`` collapse an axis to one sample, so before these there
    was no way to cut a field to a time window or move its time origin — the
    examples rebuilt the Field by hand, re-passing five identity fields and
    silently dropping the two (``model_source``, ``metadata``) they forgot.
    Both methods go through ``id_kwargs()``, which is the documented single
    home for that identity surface.
    """

    @staticmethod
    def _field():
        from uacpy.core.results import Field
        return Field(
            data=np.arange(3 * 4 * 10, dtype=float).reshape(3, 4, 10),
            coords={'depth': np.linspace(5.0, 95.0, 3),
                    'range': np.linspace(100.0, 4000.0, 4),
                    'time': np.linspace(-0.1, 0.8, 10)},
            model='probe', backend='synthetic',
            kind='pressure', metadata={'provenance': 'keep me'})

    def test_window_keeps_the_axis_it_narrows(self):
        field = self._field()
        cut = field.window(time=(0.0, 0.5))

        assert 'time' in cut.coords                     # not collapsed
        assert cut.data.shape == (3, 4, 5)
        assert cut.times.min() >= 0.0 and cut.times.max() <= 0.5

    def test_an_open_end_trims_only_the_other(self):
        field = self._field()
        assert field.window(time=(0.0, None)).times.min() >= 0.0
        assert field.window(time=(0.0, None)).times.max() == pytest.approx(
            field.times.max())

    def test_shift_moves_the_coordinate_and_leaves_the_data(self):
        field = self._field()
        moved = field.shift(time=-0.2)

        assert np.allclose(moved.times, field.times - 0.2)
        assert np.array_equal(moved.data, field.data)

    def test_both_carry_the_whole_identity_surface(self):
        """The hand-rebuild these replace copied five fields and dropped
        ``metadata`` — which is where ``kind`` lives, so a tagged quantity
        silently became an untagged one."""
        field = self._field()
        for derived in (field.window(time=(0.0, 0.5)), field.shift(time=1.0)):
            assert derived.model == 'probe'
            assert derived.backend == 'synthetic'
            assert derived.kind == 'pressure'
            assert derived.metadata['provenance'] == 'keep me'

    def test_a_window_that_keeps_nothing_raises(self):
        """An empty axis is not a smaller field; every later slice of it would
        fail somewhere less obvious."""
        with pytest.raises(ConfigurationError, match='keeps no sample'):
            self._field().window(time=(50.0, 60.0))

    @pytest.mark.parametrize('bounds,match', [
        ({'time': (0.5, 0.1)}, 'inverted'),
        ({'time': 0.5}, 'not a .lo, hi. pair'),
        ({'nonexistent': (0.0, 1.0)}, 'unknown axis'),
    ])
    def test_rejected_windows(self, bounds, match):
        with pytest.raises(ConfigurationError, match=match):
            self._field().window(**bounds)

    def test_a_non_finite_shift_raises(self):
        """It would put the whole axis at NaN, losing the coordinate."""
        with pytest.raises(ConfigurationError, match='not finite'):
            self._field().shift(time=np.nan)

    def test_the_two_compose(self):
        """Shift the origin, then cut the window — what comparing solvers on
        one display axis actually takes."""
        field = self._field()
        aligned = field.shift(time=0.1).window(time=(0.0, 0.3))

        assert aligned.times.min() >= 0.0 and aligned.times.max() <= 0.3
        assert aligned.metadata['provenance'] == 'keep me'


class TestFieldRemoveDelay:
    """``remove_delay`` advances a transfer function: ``H(f)·exp(+2πi·f·τ)``.

    The phase of a delay wraps at ``1/τ`` in frequency, so on a grid of spacing
    ``Δf`` it is unambiguous only for ``τ < 1/(2·Δf)``. A long-range ``H(f)``
    is therefore aliased beyond reading, and two models on *different* grids
    alias differently and look like they disagree when they do not.
    """

    FREQUENCIES = np.linspace(50.0, 150.0, 101)
    RANGES = np.array([2000.0, 5000.0])
    SPEED = 1500.0

    @classmethod
    def _transfer_function(cls):
        """A pure delay per range: |H| flat, phase entirely ``-2πf·r/c``."""
        from uacpy.core.results import Field
        delay = cls.RANGES[:, None] / cls.SPEED
        data = 0.5 * np.exp(-2j * np.pi * cls.FREQUENCIES[None, :] * delay)
        return Field(data=data.reshape(1, cls.RANGES.size, -1),
                     coords={'depth': np.array([50.0]), 'range': cls.RANGES,
                             'frequency': cls.FREQUENCIES},
                     model='probe', backend='synthetic',
                     kind='pressure', metadata={'provenance': 'keep me'})

    def test_removing_the_exact_delay_flattens_the_phase(self):
        one_range = self._transfer_function().at(range=5000.0)

        flattened = one_range.remove_delay(seconds=5000.0 / self.SPEED)
        assert np.max(np.abs(np.angle(flattened.data))) < 1e-9

    def test_sound_speed_takes_the_delay_from_the_fields_own_range(self):
        """There is no default tau: r/c is the delay worth removing, and a
        Field carries r but not c."""
        one_range = self._transfer_function().at(range=5000.0)

        assert np.allclose(one_range.remove_delay(sound_speed=self.SPEED).data,
                           one_range.remove_delay(seconds=5000.0 / self.SPEED).data)

    def test_each_range_is_advanced_by_its_own_travel_time(self):
        """The reduced-time convention: on a field that still has a range axis,
        every trace lines up on its own geometric arrival, so one scalar delay
        would be wrong for all but one range."""
        compensated = self._transfer_function().remove_delay(
            sound_speed=self.SPEED)

        assert np.max(np.abs(np.angle(compensated.data))) < 1e-9

    def test_the_magnitude_is_untouched(self):
        """A unit-modulus factor, so |H| and any TL from it are unchanged."""
        field = self._transfer_function()

        assert np.allclose(np.abs(field.remove_delay(sound_speed=self.SPEED).data),
                           np.abs(field.data))

    def test_a_negative_delay_adds_one_back(self):
        field = self._transfer_function()
        tau = 5000.0 / self.SPEED

        there_and_back = field.remove_delay(tau).remove_delay(-tau)
        assert np.allclose(there_and_back.data, field.data)

    def test_it_carries_the_whole_identity_surface(self):
        field = self._transfer_function().remove_delay(sound_speed=self.SPEED)

        assert field.model == 'probe' and field.backend == 'synthetic'
        assert field.metadata['provenance'] == 'keep me'

    def test_it_works_on_a_pinned_frequency(self):
        """``at(frequency=…)`` collapses the axis but keeps the value, so the
        operation is still well defined — a constant phase."""
        one_cell = self._transfer_function().at(range=5000.0, frequency=100.0)

        moved = one_cell.remove_delay(sound_speed=self.SPEED)
        assert abs(float(np.angle(np.ravel(moved.data)[0]))) < 1e-9

    @pytest.mark.parametrize('kwargs,match', [
        ({}, 'exactly one of'),
        ({'seconds': 1.0, 'sound_speed': 1500.0}, 'exactly one of'),
        ({'seconds': np.inf}, 'not finite'),
        ({'sound_speed': 0.0}, 'not a positive, finite speed'),
        ({'sound_speed': -1500.0}, 'not a positive, finite speed'),
    ])
    def test_rejected_arguments(self, kwargs, match):
        with pytest.raises(ConfigurationError, match=match):
            self._transfer_function().remove_delay(**kwargs)

    def test_a_time_domain_field_is_pointed_at_shift(self):
        """The two are the same operation on the two representations, so the
        error names the one that applies."""
        from uacpy.core.results import Field
        trace = Field(data=np.ones((1, 3), dtype=complex),
                      coords={'depth': np.array([5.0]),
                              'time': np.arange(3.0)})

        with pytest.raises(ConfigurationError, match='shift\\(time='):
            trace.remove_delay(seconds=1.0)

    def test_real_data_has_no_phase_to_move(self):
        with pytest.raises(ConfigurationError, match='carries no'):
            self._transfer_function().to_dB().remove_delay(seconds=1.0)

    def test_a_field_without_a_range_cannot_infer_the_delay(self):
        from uacpy.core.results import Field
        no_range = Field(data=np.ones((1, 3), dtype=complex),
                         coords={'depth': np.array([5.0]),
                                 'frequency': np.arange(1.0, 4.0)})

        with pytest.raises(ConfigurationError, match='no range to take'):
            no_range.remove_delay(sound_speed=1500.0)

    def test_two_grids_agree_on_the_residual_once_the_bulk_delay_is_gone(self):
        """The reason the method exists.

        Read the delay back off the unwrapped phase slope. Raw, the two grids
        report wildly different delays for the SAME field — +340 ms and −60 ms
        for a true 3340 ms — because each aliases the bulk delay its own way.
        Compensated, both report the 7 ms residual they can actually resolve.
        """
        from uacpy.core.results import Field
        residual = 0.007
        total = 5000.0 / self.SPEED + residual

        def build(n_points):
            grid = np.linspace(50.0, 150.0, n_points)
            return Field(
                data=np.exp(-2j * np.pi * grid * total).reshape(1, 1, -1),
                coords={'depth': np.array([50.0]),
                        'range': np.array([5000.0]), 'frequency': grid})

        def implied_delay(field):
            grid = np.asarray(field.coords['frequency'], dtype=float)
            phase = np.unwrap(np.angle(np.ravel(field.data)))
            return -np.polyfit(grid, phase, 1)[0] / (2 * np.pi)

        coarse, fine = build(101), build(171)      # 1.0 Hz and 0.585 Hz
        raw = [implied_delay(f) for f in (coarse, fine)]
        fixed = [implied_delay(f.remove_delay(sound_speed=self.SPEED))
                 for f in (coarse, fine)]

        # Both halves matter: without the first, this would pass on a method
        # that did nothing at all.
        assert abs(raw[0] - raw[1]) > 0.1, (
            raw, "the two grids alias alike here, so this pins nothing")
        assert all(abs(tau - residual) < 1e-6 for tau in fixed), fixed


class TestFrequencyAccessorsAgreeOnAnEmptyIdentityList:
    """``f0`` and ``n_frequencies`` read one source, the ``'frequency'`` axis
    when the field has one and the identity list otherwise, so the two
    never disagree on whether a result carries frequencies."""

    @staticmethod
    def _field(frequencies):
        return Field(data=np.zeros((2, 3)),
                     coords={'depth': np.array([0.0, 10.0]),
                             'frequency': np.array([100.0, 200.0, 300.0])},
                     frequencies=frequencies)

    def test_empty_list_counts_the_frequency_coord(self):
        f = self._field(np.array([]))
        assert f.f0 == 100.0
        assert f.n_frequencies == 3

    def test_the_frequency_axis_is_read_before_the_identity_list(self):
        # The axis is what the field holds; the list is what the run swept.
        f = self._field(np.array([50.0]))
        assert f.f0 == 100.0
        assert f.n_frequencies == 3

    def test_without_an_axis_the_identity_list_is_read(self):
        # A time trace keeps the band it was synthesised from as its list.
        f = Field(data=np.zeros((2, 4)),
                  coords={'depth': np.array([0.0, 10.0]),
                          'time': np.arange(4) * 1e-3},
                  frequencies=np.array([200.0, 210.0, 220.0]))
        assert f.f0 == 200.0
        assert f.n_frequencies == 3

    def test_no_list_and_no_coord_counts_zero(self):
        f = Field(data=np.zeros((2, 2)),
                  coords={'depth': np.array([0.0, 10.0]),
                          'range': np.array([1.0, 2.0])})
        assert f.f0 is None
        assert f.n_frequencies == 0


class TestTimeTraceLabelsFollowTheNearestCellRule:
    """``to_time_trace`` selects its cell the way ``at`` does: a finite label
    so far outside the axis that every sample rounds to the same distance is
    refused instead of silently landing on index 0."""

    @staticmethod
    def _broadband():
        n_freq = 16
        data = np.zeros((3, 2, n_freq), dtype=complex)
        data[:, :, :] = 1.0
        return Field(data=data,
                     coords={'depth': np.array([10.0, 20.0, 30.0]),
                             'range': np.array([1000.0, 2000.0]),
                             'frequency': np.linspace(100.0, 250.0, n_freq)})

    @pytest.mark.parametrize('axis', ['depth', 'range'])
    @pytest.mark.parametrize('label', [1e300, -1e300])
    def test_absorbing_label_is_refused_like_at(self, axis, label):
        tf = self._broadband()
        with pytest.raises(ConfigurationError, match='same distance'):
            tf.at(**{axis: label})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ConfigurationError, match='same distance'):
                tf.to_time_trace(**{axis: label})

    @pytest.mark.parametrize('axis, label, expected',
                             [('depth', 1e6, 30.0), ('range', -1e6, 1000.0)])
    def test_far_but_rankable_label_lands_on_the_end_sample(
            self, axis, label, expected):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            trace = self._broadband().to_time_trace(**{axis: label})
        assert trace.pinned[axis] == expected


class TestMaskBelowSeafloorKeepsTheFloatingDtype:
    """The masked copy keeps an inexact payload's dtype (a ``.shd``-backed
    result stays float32) and widens only integers, which cannot hold NaN."""

    @staticmethod
    def _field(dtype):
        return Field(
            data=np.ones((3, 2), dtype=dtype),
            coords={'depth': np.array([0.0, 50.0, 150.0]),
                    'range': np.array([0.0, 1000.0])},
            frequencies=100.0)

    _BATHY = [(0.0, 100.0), (1000.0, 100.0)]

    @pytest.mark.parametrize('dtype', [np.float32, np.float64,
                                       np.complex64, np.complex128])
    def test_inexact_dtype_is_kept(self, dtype):
        masked = self._field(dtype).mask_below_seafloor(self._BATHY)
        assert masked.data.dtype == np.dtype(dtype)
        assert np.isnan(masked.data[2, :]).all()
        assert np.isfinite(masked.data[:2, :]).all()

    def test_integer_payload_is_widened_to_float64(self):
        masked = self._field(np.int32).mask_below_seafloor(self._BATHY)
        assert masked.data.dtype == np.dtype(np.float64)
        assert np.isnan(masked.data[2, :]).all()


class TestAWindowedFieldReportsTheBandItHolds:
    """``Field.window`` narrows the identity, not only the axis.

    Selecting a signal's band out of a wider run is how both quantities above
    are reached, and a field that keeps the run's whole frequency list after
    that answers ``f0`` and ``n_frequencies`` about samples it no longer has.
    ``isel`` already narrows the identity when it pins an axis.
    """

    @staticmethod
    def wideband():
        f = np.arange(1000.0, 2001.0, 100.0)
        return Field(data=np.ones((1, 1, f.size), complex),
                     coords={'depth': [0.0], 'range': [1.0], 'frequency': f},
                     frequencies=f)

    def test_the_band_count_matches_the_axis(self):
        narrowed = self.wideband().window(frequency=(1400.0, 1600.0))
        assert narrowed.n_frequencies == narrowed.coords['frequency'].size == 3

    def test_the_centre_frequency_is_one_the_field_holds(self):
        # 1000 Hz is a legal frequency and the wrong answer, which is why
        # this needs its own assertion rather than a finiteness check.
        narrowed = self.wideband().window(frequency=(1400.0, 1600.0))
        assert narrowed.f0 == pytest.approx(1400.0)

    def test_windowing_another_axis_leaves_the_band_alone(self):
        untouched = self.wideband().window(range=(0.0, 2.0))
        assert untouched.n_frequencies == 11

    def test_source_depths_narrow_with_their_axis(self):
        z = np.array([5.0, 10.0, 15.0])
        f = np.array([100.0, 200.0])
        field = Field(data=np.ones((z.size, 1, 1, f.size), complex),
                      coords={'source_depth': z, 'depth': [0.0],
                              'range': [1.0], 'frequency': f},
                      frequencies=f, source_depths=z)
        narrowed = field.window(source_depth=(9.0, 16.0))
        assert list(np.asarray(narrowed.source_depths)) == [10.0, 15.0]


class TestAFieldRoundTripsThroughNpSavez:
    """``Field.to_dict`` documents an ``np.savez`` route; the file must read
    back into an equal field through ``np.load(allow_pickle=True)`` and
    ``Field.from_dict``, including a ``None`` entry, the pinned dict and a
    fully pinned (0-d) field."""

    @staticmethod
    def _roundtrip(field):
        import io
        buf = io.BytesIO()
        np.savez(buf, **field.to_dict())
        buf.seek(0)
        with np.load(buf, allow_pickle=True) as npz:
            return Field.from_dict(dict(npz))

    def _field(self):
        return Field(data=np.arange(6.0).reshape(2, 3) + 1j,
                     coords={'depth': np.array([10.0, 20.0]),
                             'range': np.array([100.0, 200.0, 300.0])},
                     model='Bellhop', frequencies=np.array([50.0]),
                     phase_reference=uacpy.PhaseReference.TRAVELLING_WAVE,
                     metadata={'note': 'x'})

    def test_the_phase_reference_comes_back_as_the_enum_member(self):
        back = self._roundtrip(self._field())
        assert back.phase_reference is uacpy.PhaseReference.TRAVELLING_WAVE
        in_memory = Field.from_dict(self._field().to_dict())
        assert in_memory.phase_reference is uacpy.PhaseReference.TRAVELLING_WAVE

    def test_a_grid_field_comes_back_equal(self):
        f = self._field()
        back = self._roundtrip(f)
        np.testing.assert_array_equal(back.data, f.data)
        assert list(back.coords) == ['depth', 'range']
        assert (back.model, back.backend, back.kind, back.unit) == \
            (f.model, f.backend, f.kind, f.unit)
        assert back.metadata['note'] == 'x'

    def test_a_fully_pinned_field_keeps_its_0d_data(self):
        f = self._field().isel(depth=1, range=2)
        back = self._roundtrip(f)
        assert back.data.shape == () and back.data == f.data
        assert back.pinned == f.pinned


class TestFieldSpellsEachArgumentOnce:
    """One spelling per concept across ``Field``'s methods: the transmitted
    signal is ``source_waveform``, its spectrum ``source_spectrum``, the
    untapered window ``None`` (``truncate_response`` included), the
    receiver range ``range`` (``BeamformedField.to_time_trace`` included),
    and ``kind`` / ``unit`` are constructor keywords."""

    def test_the_methods_share_the_source_names(self):
        for method in (Field.to_time_trace, Field.broadband_loss,
                       Field.synthesize_time_series,
                       Field.sound_exposure_level,
                       Field.peak_sound_pressure_level):
            params = inspect.signature(method).parameters
            assert 'source_waveform' in params, method.__name__
            assert 'waveform' not in params and 'spectrum' not in params
        for method in (Field.to_time_trace, Field.broadband_loss):
            assert 'source_spectrum' in inspect.signature(method).parameters

    def test_the_rectangular_window_is_none_on_truncate_response(self):
        params = inspect.signature(Field.truncate_response).parameters
        assert params['window'].default is None

    def test_the_beamformed_trace_takes_range(self):
        from uacpy.acoustic_signal.beamforming import BeamformedField
        params = inspect.signature(BeamformedField.to_time_trace).parameters
        assert 'range' in params and 'range_m' not in params

    def test_kind_and_unit_are_constructor_keywords(self):
        f = Field(data=np.array([[60.0, 70.0]]),
                  coords={'depth': np.array([10.0]),
                          'range': np.array([1.0, 2.0])},
                  kind='level', unit='dB')
        assert (f.kind, f.unit) == ('level', 'dB')
        for tag, value in (('kind', 'pressure'), ('unit', 'dB'),
                           ('coherent', True), ('reference', 1.0),
                           ('reference_unit', '1')):
            with pytest.raises(ConfigurationError,
                               match=rf"metadata carries \['{tag}'\]"):
                Field(data=np.array([[1.0]]),
                      coords={'depth': np.array([1.0]),
                              'range': np.array([1.0])},
                      metadata={tag: value})
        with pytest.raises(ConfigurationError, match='unknown Field kind'):
            Field(data=np.array([[1.0]]),
                  coords={'depth': np.array([1.0]), 'range': np.array([1.0])},
                  kind='not_a_quantity')


class TestAFieldRoundTripsThroughXarray:
    """``Field.to_xarray`` gives a labelled DataArray (dims = coords in axis
    order, pinned axes as scalar coords, kind/unit/identity in attrs) and
    ``Field.from_xarray`` reads it back — in memory and through a NetCDF
    file."""

    @staticmethod
    def _field():
        return Field(data=np.arange(6.0).reshape(2, 3) + 60.0,
                     coords={'depth': np.array([10.0, 20.0]),
                             'range': np.array([100.0, 200.0, 300.0])},
                     pinned={'frequency': 250.0}, model='Bellhop',
                     frequencies=np.array([250.0]),
                     source_depths=np.array([5.0]),
                     kind='level', unit='dB', metadata={'note': 'x'})

    def test_the_dataarray_carries_the_axes_and_the_quantity(self):
        pytest.importorskip('xarray')
        da = self._field().to_xarray()
        assert list(da.dims) == ['depth', 'range']
        assert float(da.coords['frequency']) == 250.0
        assert (da.name, da.attrs['kind'], da.attrs['units']) == \
            ('level', 'level', 'dB')
        np.testing.assert_array_equal(da.coords['range'], [100.0, 200.0, 300.0])

    def test_every_axis_carries_its_cf_units(self):
        pytest.importorskip('xarray')
        da = self._field().to_xarray()
        assert 'unit' not in da.attrs
        assert {name: da.coords[name].attrs.get('units')
                for name in ('depth', 'range', 'frequency')} == \
            {'depth': 'm', 'range': 'm', 'frequency': 'Hz'}

    def test_a_dataarray_with_a_unit_attr_reads_that_unit(self):
        # A unit other than the kind's default, so a read that dropped the
        # attr would answer the default instead.
        pytest.importorskip('xarray')
        f = Field(data=np.full(3, 60.0), coords={'time': [0.0, 0.1, 0.2]},
                  kind='pressure', unit='dB')
        da = f.to_xarray()
        da.attrs['unit'] = da.attrs.pop('units')
        assert Field.from_xarray(da).unit == 'dB'

    def test_it_round_trips_in_memory_and_through_netcdf(self, tmp_path):
        xr = pytest.importorskip('xarray')
        f = self._field()
        for back in (Field.from_xarray(f.to_xarray()),):
            np.testing.assert_array_equal(back.data, f.data)
            assert back.pinned == f.pinned
            assert (back.kind, back.unit, back.model) == ('level', 'dB',
                                                          'Bellhop')
            np.testing.assert_array_equal(back.frequencies, [250.0])
            assert back.metadata['note'] == 'x'
        path = tmp_path / 'f.nc'
        f.to_xarray().to_netcdf(path, engine='scipy')
        with xr.open_dataarray(path, engine='scipy') as loaded:
            back = Field.from_xarray(loaded.load())
        np.testing.assert_array_equal(back.data, f.data)
        assert list(back.coords) == ['depth', 'range']
        assert (back.kind, back.unit) == ('level', 'dB')

    def test_a_complex_field_keeps_its_phase_in_memory(self):
        pytest.importorskip('xarray')
        f = Field(data=np.array([[1 + 2j, -1j]]),
                  coords={'depth': np.array([1.0]),
                          'range': np.array([1.0, 2.0])})
        back = Field.from_xarray(f.to_xarray())
        np.testing.assert_array_equal(back.data, f.data)
        assert back.is_complex and back.unit == 'Pa'


def _pressure_trace():
    """A real ``(depth, range, time)`` pressure trace, synthesised."""
    burst = np.sin(2 * np.pi * 200.0 * np.arange(0.0, 0.02, 1 / 2000.0))
    return _two_path_grid().synthesize_time_series(burst, 2000.0,
                                                   t_start=0.0)


class TestTheUnitIsFixedAtConstruction:
    """The unit is resolved once, when a Field is built, and inherited by
    every Field derived from it — a
    selection never re-reads its own storage. Re-read, a pressure trace
    sliced at one instant (no time axis left) became TL in dB (RES-21)."""

    @pytest.mark.parametrize('select', [
        lambda ts: ts.at(time=0.1),
        lambda ts: ts.isel(time=200),
        lambda ts: ts.eval(time=0.1),
        lambda ts: ts.max(),
        lambda ts: ts.window(time=(0.05, 0.1)).at(time=0.08),
    ])
    def test_a_slice_of_a_pressure_trace_stays_in_pascals(self, select):
        ts = _pressure_trace()
        assert ts.unit == 'Pa' and ts.unit == 'Pa'
        snapshot = select(ts)
        assert 'time' not in snapshot.coords
        assert snapshot.unit == 'Pa'

    def test_a_snapshot_is_not_read_as_a_level(self):
        snapshot = _pressure_trace().at(time=0.1)
        with pytest.raises(AttributeError, match="'Pa', not dB"):
            snapshot.dB
        with pytest.raises(ConfigurationError, match='linear pressure'):
            snapshot.at_source_level(180.0)

    def test_the_loudest_sample_of_a_trace_is_the_largest_pressure(self):
        ts = _pressure_trace()
        loudest = ts.max()
        assert abs(float(loudest.data)) == pytest.approx(
            float(np.max(np.abs(ts.data))))

    def test_a_hand_tagged_linear_map_keeps_its_unit_through_max(self):
        # Real magnitudes tagged unit='Pa': max() is the loudest cell. Left
        # untagged, a real map without a time axis is read as TL (RES-13).
        amplitudes = np.array([[0.02, 0.001], [0.005, 0.0001]])
        coords = {'depth': np.array([10.0, 20.0]),
                  'range': np.array([100.0, 200.0])}
        tagged = Field(data=amplitudes, coords=coords, unit='Pa')
        assert tagged.max().pinned == {'depth': 10.0, 'range': 100.0}
        assert tagged.at(depth=10.0).unit == 'Pa'
        untagged = Field(data=amplitudes, coords=coords)
        assert untagged.unit == 'dB'
        assert untagged.max().pinned == {'depth': 20.0, 'range': 200.0}

    def test_the_dict_states_the_unit_at_the_top_level(self):
        trace_dict = _pressure_trace().at(time=0.1).to_dict()
        assert 'unit' not in trace_dict['metadata']
        assert trace_dict['unit'] == 'Pa'
        assert Field.from_dict(trace_dict).unit == 'Pa'


class TestAFileThatKeepsTheQuantityInMetadataLoads:
    """An ``np.savez`` of a Field's dict whose metadata holds the quantity
    tags (the layout files written with them there have) loads, the tags
    restored as the Field's own attributes."""

    @staticmethod
    def _metadata_layout(field):
        d = field.to_dict()
        meta = dict(d['metadata'])
        for tag in ('kind', 'unit', 'coherent', 'reference', 'reference_unit'):
            value = d.pop(tag)
            if tag in ('kind', 'unit'):
                d[tag] = value
            if value is not None:
                meta[tag] = value
        d['metadata'] = meta
        return d

    @pytest.mark.parametrize('make', [
        lambda: _pressure_trace().at(time=0.1),
        lambda: _two_path_grid().at(frequency=200.0).to_dB(),
    ])
    def test_it_loads_through_savez(self, make, tmp_path):
        field = make()
        path = tmp_path / 'field.npz'
        np.savez(path, **self._metadata_layout(field))
        back = Field.from_dict(np.load(path, allow_pickle=True))
        assert (back.kind, back.unit, back.coherent) == \
            (field.kind, field.unit, field.coherent)
        assert not set(back.metadata) & {'kind', 'unit', 'coherent'}
        assert np.array_equal(back.data, field.data)

    def test_an_ambiguity_surface_keeps_its_reference(self):
        from uacpy.core.results import ambiguity_field
        surface = ambiguity_field(np.array([[1.0, 4.0]]),
                                  {'depth': [10.0], 'range': [1.0, 2.0]},
                                  reference_unit='Pa²/Hz')
        back = Field.from_dict(self._metadata_layout(surface))
        assert (back.kind, back.reference, back.reference_unit) == \
            ('ambiguity', 4.0, 'Pa²/Hz')

    def test_the_quantity_round_trips_through_xarray(self):
        pytest.importorskip('xarray')
        from uacpy.core.results import ambiguity_field
        surface = ambiguity_field(np.array([[1.0, 4.0]]),
                                  {'depth': [10.0], 'range': [1.0, 2.0]},
                                  reference_unit='Pa²/Hz')
        back = Field.from_xarray(surface.to_xarray())
        assert (back.kind, back.unit, back.reference, back.reference_unit) \
            == ('ambiguity', 'dB', 4.0, 'Pa²/Hz')
        stated = _two_path_grid().at(frequency=200.0).replace(coherent=False)
        assert Field.from_xarray(stated.to_xarray()).coherent is False


class TestCoherenceIsOneDecisionOnTheField:
    """``Field.coherent`` answers "is this pressure field a coherent sum" in
    one place, from inputs every derived Field inherits (ARCH-4)."""

    def test_a_broadband_slice_is_as_coherent_as_the_narrowband_run(self):
        H = _two_path_grid()
        H.run_mode = 'broadband'
        assert H.at(frequency=200.0).coherent is True

    def test_the_answer_survives_the_conversion_to_db(self):
        p = _two_path_grid().at(frequency=200.0)
        assert p.to_dB().coherent is True

    def test_real_db_is_coherent_only_when_the_run_says_so(self):
        assert _field(data=np.full((4, 3), 60.0)).coherent is False
        assert _field(data=np.full((4, 3), 60.0),
                      run_mode='coherent_tl').coherent is True

    def test_the_answer_is_decided_when_the_field_is_built(self):
        tl = _field(data=np.full((4, 3), 60.0))
        assert tl.coherent is False
        tl.run_mode = 'coherent_tl'
        assert tl.coherent is False

    def test_an_intensity_sum_is_incoherent_whatever_run_made_its_slabs(self):
        p = _two_path_grid().at(frequency=200.0)
        p.run_mode = 'coherent_tl'
        total = ResultStack([p, p], [1.0, 2.0]).superpose(coherent=False)
        assert total.coherent is False

    def test_a_derived_field_keeps_a_stated_coherence(self):
        """The coherence the producer stated, not the one the storage
        would give (complex travelling-wave pressure reads as coherent), is
        what every Field derived from it inherits."""
        stated = _two_path_grid().at(frequency=200.0).replace(coherent=False)
        assert stated.coherent is False
        for view in (stated.at(range=150.0), stated.to_dB(), stated.max()):
            assert view.coherent is False

    def test_a_trace_and_a_non_pressure_kind_have_no_answer(self):
        assert _pressure_trace().coherent is None
        assert _field(kind='level', unit='dB').coherent is None

    @pytest.mark.parametrize('select', [
        lambda ts: ts.at(time=0.1),
        lambda ts: ts.isel(time=200),
        lambda ts: ts.eval(time=0.1),
        lambda ts: ts.max(),
        lambda ts: ts.at(depth=10.0).at(range=150.0).at(time=0.1),
    ], ids=['at', 'isel', 'eval', 'max', 'at_chain'])
    def test_a_snapshot_of_a_trace_has_no_answer_either(self, select):
        snapshot = select(_pressure_trace())
        assert 'time' not in snapshot.coords
        assert snapshot.coherent is None

    def test_a_snapshot_read_back_has_no_answer(self):
        pytest.importorskip('xarray')
        snapshot = _pressure_trace().at(time=0.1)
        assert Field.from_xarray(snapshot.to_xarray()).coherent is None
        assert Field.from_dict(snapshot.to_dict()).coherent is None

    def test_a_trace_of_coherent_pressure_decides_its_own_answer(self):
        H = _two_path_grid()
        assert H.coherent is True
        burst = np.sin(2 * np.pi * 200.0 * np.arange(0.0, 0.02, 1 / 2000.0))
        trace = H.to_time_trace(depth=10.0, range=150.0, t_start=0.0,
                                source_waveform=burst, sample_rate=2000.0)
        assert trace.kind == 'pressure' and trace.coherent is None
        trace = _pressure_trace()
        assert trace.to_transfer_function().coherent is True

    def test_a_stamped_tag_is_the_answer(self):
        tl = _field(data=np.full((4, 3), 60.0), coherent=True)
        assert tl.coherent is True


class TestASlicedCellSynthesisesLikeItsGridCell:
    """``to_time_trace`` and ``synthesize_time_series`` take a cell sliced out
    of a grid; its pinned depth and range stand in for the axes (RES-3)."""

    def test_the_trace_of_a_sliced_cell_is_the_grids_trace(self):
        H = _two_path_grid()
        cell = H.at(depth=10.0, range=300.0)
        assert list(cell.coords) == ['frequency']
        a = cell.to_time_trace(window=None, t_start=0.0)
        b = H.to_time_trace(depth=10.0, range=300.0, window=None,
                            t_start=0.0)
        np.testing.assert_array_equal(a.data, b.data)
        assert a.pinned == b.pinned == {'depth': 10.0, 'range': 300.0}

    def test_a_row_sliced_by_depth_synthesises_along_range(self):
        H = _two_path_grid()
        burst = np.sin(2 * np.pi * 200.0 * np.arange(0.0, 0.02, 1 / 2000.0))
        row = H.at(depth=20.0).synthesize_time_series(burst, 2000.0,
                                                      t_start=0.0)
        whole = H.synthesize_time_series(burst, 2000.0, t_start=0.0)
        assert list(row.coords) == ['range', 'time']
        np.testing.assert_array_equal(row.data, whole.data[1])
        assert row.pinned == {'depth': 20.0}

    def test_a_cell_with_no_range_anywhere_is_refused(self):
        f = np.arange(100.0, 300.0, 1.0)
        spectrum = Field(data=np.ones(f.size, complex),
                         coords={'frequency': f})
        with pytest.raises(ConfigurationError, match='no pinned range'):
            spectrum.to_time_trace()

    def test_a_placeholder_depth_is_not_reported(self):
        f = np.arange(100.0, 300.0, 1.0)
        spectrum = Field(data=np.ones(f.size, complex),
                         coords={'frequency': f}, pinned={'range': 150.0})
        trace = spectrum.to_time_trace(window=None, t_start=0.0)
        assert trace.pinned == {'range': 150.0}


class TestTheSynthesisWindowIsAnchoredOnTheNearestRange:
    """One record for every cell, placed from the range nearest the source
    whichever way the axis runs (RES-12). 150 m and 1500 m at 1500 m/s on a
    1 s record: anchored on 1500 m the window opened at 0.5 s and the 150 m
    arrival wrapped a whole record."""

    def test_a_descending_axis_places_the_record_as_an_ascending_one(self):
        burst = np.sin(2 * np.pi * 200.0 * np.arange(0.0, 0.02, 1 / 2000.0))
        up = _two_path_grid(ranges=(150.0, 1500.0))
        down = Field(data=up.data[:, ::-1, :].copy(),
                     coords={'depth': up.coords['depth'],
                             'range': up.coords['range'][::-1],
                             'frequency': up.coords['frequency']},
                     frequencies=up.frequencies,
                     phase_reference=PhaseReference.TRAVELLING_WAVE,
                     speeds=SoundSpeeds(water_max=1500.0))
        up = up.replace(speeds=SoundSpeeds(water_max=1500.0))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            t_up = up.synthesize_time_series(burst, 2000.0).times[0]
            t_down = down.synthesize_time_series(burst, 2000.0).times[0]
        assert t_up == t_down == 0.0


class TestTheSynthesisReadsTheWaveguideTheRunResolved:
    """A Field from a model run carries ``run_settings.waveguide``; with no
    other speed stated, the synthesis anchors its window on the
    waveguide's c_max and measures the range span's travel time at its c_min
    (ARCH-7). Waveguide here: 1480 m/s water over a 1700 m/s half-space."""

    @staticmethod
    def _settings():
        from uacpy.core.receiver import Receiver
        env = Environment(
            name='fast floor', bathymetry=100.0, ssp=1480.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.8,
                                      attenuation=0.5))
        return uacpy.Kraken().run_settings(
            env, Source(depths=25.0, frequencies=200.0),
            Receiver(depths=np.array([50.0]), ranges=np.array([1000.0])))

    def test_the_window_opens_a_tenth_of_a_record_before_range_over_c_max(self):
        H = _two_path_grid(ranges=(3000.0, 3100.0))
        H._run_settings = self._settings()
        assert H.run_settings.waveguide.c_max == 1700.0
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            trace = H.to_time_trace(depth=10.0, range=3000.0)
        assert float(trace.coords['time'][0]) == pytest.approx(
            3000.0 / 1700.0 - 0.1, abs=1e-9)

    def test_a_field_read_back_from_xarray_keeps_the_waveguide(self):
        pytest.importorskip('xarray')
        H = _two_path_grid(ranges=(3000.0, 3100.0))
        H._run_settings = self._settings()
        back = Field.from_xarray(H.to_xarray())
        assert back.run_settings.to_dict() == H.run_settings.to_dict()
        assert (back.speeds.waveguide_min,
                back.speeds.waveguide_max) == (1480.0, 1700.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            trace = back.to_time_trace(depth=10.0, range=3000.0)
        assert float(trace.coords['time'][0]) == pytest.approx(
            3000.0 / 1700.0 - 0.1, abs=1e-9)

    def test_without_settings_the_window_falls_back_to_the_default(self):
        H = _two_path_grid(ranges=(3000.0, 3100.0))
        with pytest.warns(UserWarning, match='stamped no sound speed'):
            trace = H.to_time_trace(depth=10.0, range=3000.0)
        assert float(trace.coords['time'][0]) == pytest.approx(
            3000.0 / 1500.0 - 0.1, abs=1e-9)

    def test_a_range_span_is_timed_at_the_slowest_speed(self):
        # 1490 m of span: 1.0068 s at 1480 m/s, past the 0.9995 s record;
        # 0.9933 s at the 1500 m/s default, inside it.
        burst = np.sin(2 * np.pi * 200.0 * np.arange(0.0, 0.02, 1 / 2000.0))
        H = _two_path_grid(ranges=(150.0, 1640.0))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            H.synthesize_time_series(burst, 2000.0, t_start=0.0)
        H._run_settings = self._settings()
        with pytest.warns(UserWarning, match='far-range arrivals wrap'):
            H.synthesize_time_series(burst, 2000.0, t_start=0.0)


class TestAZeroHertzBinIsCountedOnce:
    """The synthesis doubles its one-sided sum for the negative
    frequencies; a 0 Hz bin has no negative twin (RES-19)."""

    def test_a_unit_spectrum_from_zero_hertz_integrates_to_one(self):
        f = np.arange(0.0, 100.0, 1.0)
        H = Field(data=np.ones((1, 1, f.size), complex),
                  coords={'depth': np.array([1.0]), 'range': np.array([1.0]),
                          'frequency': f},
                  phase_reference=PhaseReference.TRAVELLING_WAVE)
        h = H.to_time_trace(window=None, t_start=0.0)
        assert float(np.sum(h.data) * h.dt) == pytest.approx(1.0, rel=1e-12)

    def test_a_negative_frequency_is_refused(self):
        f = np.arange(-10.0, 90.0, 1.0)
        H = Field(data=np.ones((1, 1, f.size), complex),
                  coords={'depth': np.array([1.0]), 'range': np.array([1.0]),
                          'frequency': f})
        with pytest.raises(ConfigurationError, match='one-sided'):
            H.to_time_trace(window=None, t_start=0.0)

    def test_a_source_spectrum_off_the_axis_is_refused_by_name(self):
        with pytest.raises(ConfigurationError, match='source_spectrum'):
            _two_path_grid().to_time_trace(source_spectrum=np.ones(5))


class TestTheTransformOfATraceKeepsItsTag:
    """``to_transfer_function`` keeps ``TIME_DOMAIN_NATIVE`` on purpose (its
    spectrum still carries the synthesis's source spectrum or window), and
    the refusal says so rather than naming SPARC alone (RES-1, RES-2)."""

    def test_the_refusal_names_the_tag_and_both_producers(self):
        trace = _two_path_grid().to_time_trace(window=None, t_start=0.0)
        back = trace.to_transfer_function()
        assert back.phase_reference == 'time_domain_native'
        with pytest.raises(ConfigurationError,
                           match='to_transfer_function of a synthesised'):
            back.to_time_trace()

    def test_a_windowed_trace_warns_that_its_transform_is_windowed(self):
        trace = _two_path_grid().to_time_trace(t_start=0.0)
        assert trace.synthesis_window == 'hann'
        with pytest.warns(UserWarning, match="'hann' band window"):
            trace.to_transfer_function()
        plain = _two_path_grid().to_time_trace(window=None, t_start=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            plain.to_transfer_function()


class TestIdentityAxesAreNotShifted:
    def test_shifting_a_frequency_axis_is_refused(self):
        with pytest.raises(ConfigurationError, match='identity axis'):
            _two_path_grid().shift(frequency=10.0)

    def test_shifting_time_moves_the_axis(self):
        ts = _pressure_trace()
        assert ts.shift(time=-0.01).times[0] == pytest.approx(
            ts.times[0] - 0.01)


class TestCountsAndIndicesAreIntegers:
    """A slice ``[:n]`` reads a negative count from the end and ``int()``
    truncates a float, both silently (RES-16)."""

    def test_a_float_position_is_refused(self):
        with pytest.raises(ConfigurationError, match='not an integer index'):
            _field().isel(range=1.7)
        assert _field().isel(range=np.int64(1)).pinned == {'range': 200.0}

    def test_a_stack_position_must_be_an_integer(self):
        p = _field()
        stack = ResultStack([p, p], [1.0, 2.0])
        with pytest.raises(ConfigurationError, match='not an integer index'):
            stack.isel(source_depth=0.5)

    @pytest.mark.parametrize('n', [0, -1])
    def test_a_count_below_one_is_refused(self, n):
        arrivals = uacpy.Arrivals(
            arrivals=[{'delay': float(i), 'amplitude': 1.0 + i, 'phase': 0.0}
                      for i in range(3)],
            receiver_depths=[50.0], receiver_ranges=[1000.0])
        with pytest.raises(ConfigurationError, match='need n >= 1'):
            arrivals.top_n_by_amplitude(n)
        rays = uacpy.core.results.Rays(rays=[
            {'r': np.array([0.0, 1.0]), 'z': np.array([5.0, 6.0]),
             'launch_angle': 1.0, 'n_top_bounces': 0, 'n_bot_bounces': 0}])
        with pytest.raises(ConfigurationError, match='need n >= 1'):
            rays.first_n(n)
        with pytest.raises(ConfigurationError, match='need n >= 1'):
            self._modes().first_n(n)

    @staticmethod
    def _modes():
        from uacpy.core.results import Modes
        return Modes(k=np.array([0.3 + 0j, 0.2 + 0j, 0.1 + 0j]),
                     phi=np.zeros((2, 3)), depths=np.array([0.0, 50.0]),
                     model='Kraken', frequencies=25.0)

    def test_a_fractional_mode_count_is_refused(self):
        with pytest.raises(ConfigurationError,
                           match=r'Modes\.first_n\(2\.5\): the count must be '
                                 r'an integer'):
            self._modes().first_n(2.5)

    @pytest.mark.parametrize('n,kept', [(1, 1), (np.int64(2), 2), (5, 3)])
    def test_an_integer_mode_count_keeps_that_many(self, n, kept):
        assert self._modes().first_n(n).n_modes == kept


class TestMetadataSurvivesXarray:
    """Complex metadata (the Source weights), tuples (``span_m``) and the
    collapsed band (``band_hz``) are
    written in a form a NetCDF attribute holds and restored (RES-4), and a
    stack reads back whole (RES-5)."""

    @staticmethod
    def _stack():
        slabs = [Field(data=np.full((1, 2), 1.0 + 0j),
                       coords={'depth': np.array([10.0]),
                               'range': np.array([100.0, 200.0])},
                       source_depths=np.array([z]),
                       phase_reference=PhaseReference.TRAVELLING_WAVE,
                       source_weights=np.array([1.0, 2.0 + 0j]),
                       metadata={'span_m': (100.0, 300.0)},
                       band_hz=(100.0, 300.0))
                 for z in (5.0, 15.0)]
        return ResultStack(slabs, [5.0, 15.0])

    def test_complex_weights_and_a_band_come_back(self):
        pytest.importorskip('xarray')
        slab = self._stack()[0]
        back = Field.from_xarray(slab.to_xarray())
        np.testing.assert_array_equal(back.source_weights, [1.0, 2.0 + 0j])
        assert back.metadata['span_m'] == (100.0, 300.0)
        assert back.band_hz == (100.0, 300.0)
        assert back.phase_reference is PhaseReference.TRAVELLING_WAVE

    def test_complex_weights_come_back_through_a_netcdf_file(self, tmp_path):
        # A NetCDF attribute holds no complex number: written as they are,
        # the weights make the file unwritable (or were dropped silently).
        xr = pytest.importorskip('xarray')
        tl = _field(data=np.full((4, 3), 60.0),
                    source_weights=np.array([1.0, 2.0 + 0.5j]),
                    metadata={'span_m': (100.0, 300.0)},
                    band_hz=(100.0, 300.0))
        path = tmp_path / 'tl.nc'
        tl.to_xarray().to_netcdf(path, engine='scipy')
        with xr.open_dataarray(path, engine='scipy') as loaded:
            back = Field.from_xarray(loaded.load())
        np.testing.assert_array_equal(back.source_weights, [1.0, 2.0 + 0.5j])
        assert back.metadata['span_m'] == (100.0, 300.0)
        assert back.band_hz == (100.0, 300.0)

    def test_a_stack_superposes_the_same_after_the_round_trip(self):
        pytest.importorskip('xarray')
        stack = self._stack()
        back = ResultStack.from_xarray(stack.to_xarray())
        assert back.coordinate_name == 'source_depth'
        np.testing.assert_array_equal(back.superpose().data,
                                      stack.superpose().data)
        assert abs(back.superpose().data[0, 0]) == pytest.approx(3.0)
        np.testing.assert_array_equal(back[1].source_depths, [15.0])
        assert 'source_depth' not in back[1].pinned

    def test_a_slab_read_alone_narrows_its_identity_to_the_pinned_depth(self):
        pytest.importorskip('xarray')
        da = self._stack().to_xarray().isel(source_depth=1)
        slab = Field.from_xarray(da)
        np.testing.assert_array_equal(slab.source_depths, [15.0])

    def test_an_entry_no_attribute_holds_is_named_in_a_warning(self):
        pytest.importorskip('xarray')
        f = _field(metadata={'nested': {'a': 1}})
        with pytest.warns(UserWarning, match=r"\['nested'\]"):
            f.to_xarray()


class TestFieldIO:
    """Tests for Field I/O operations."""

    @staticmethod
    def _make_field(**meta):
        return Field(
            data=np.random.rand(10, 20),
            coords={
                'depth': np.linspace(10, 90, 10),
                'range': np.linspace(100, 5000, 20),
            },
            model='Bellhop', frequencies=100.0,
            metadata=meta,
        )

    def test_field_metadata_preservation(self):
        field = self._make_field(source_depth=50.0, custom_param='test_value')
        assert field.model == 'Bellhop'
        assert field.f0 == 100.0
        assert list(field.frequencies) == [100.0]
        assert field.metadata['source_depth'] == 50.0
        assert field.metadata['custom_param'] == 'test_value'

    def test_field_deepcopy_preserves_metadata(self):
        import copy as _copy
        field = self._make_field(test_key='test_value')
        field_copy = _copy.deepcopy(field)
        assert field_copy.metadata['test_key'] == 'test_value'
        assert field_copy.metadata is not field.metadata


def _line_field():
    """An irregular receiver line: one sample per range, its depth as an
    auxiliary coordinate along ``range``."""
    return Field(data=np.array([[1.0 + 1j, 2.0, 3.0, 4.0]]),
                 coords={'depth': np.array([0.0]),
                         'range': np.array([100.0, 200.0, 300.0, 400.0])},
                 aux_coords={'receiver_depth': ('range',
                                                [10.0, 20.0, 30.0, 40.0])},
                 frequencies=100.0, model='probe')


class TestReplaceIsTheValidatedCopy:
    """``Field.replace`` builds the new Field through the constructor, so a
    change the constructor would refuse is refused here too."""

    def test_an_unnamed_field_carries_over(self):
        f = _line_field()
        g = f.replace(data=f.data * 2)
        np.testing.assert_array_equal(g.data, f.data * 2)
        assert g.model == 'probe' and g.coords.keys() == f.coords.keys()
        np.testing.assert_array_equal(g.frequencies, [100.0])

    def test_the_copy_owns_its_arrays(self):
        f = _line_field()
        g = f.replace()
        g.data[...] = 0.0
        assert np.all(f.data != 0.0)

    def test_a_shape_the_coords_do_not_fit_is_refused(self):
        with pytest.raises(ConfigurationError, match='does not match coord'):
            _line_field().replace(data=np.zeros((1, 3)))

    def test_an_unknown_name_is_refused(self):
        with pytest.raises(ConfigurationError, match='name no field'):
            _line_field().replace(datum=1.0)

    def test_kind_and_unit_are_written_into_the_tags(self):
        g = _line_field().replace(data=np.ones((1, 4)), kind='level',
                                  unit='dB')
        assert (g.kind, g.unit) == ('level', 'dB')


class TestReindexWidensAnAxis:
    """``Field.reindex`` puts the stored samples at their labels and fills
    the labels it adds; it never drops a stored sample."""

    def test_the_stored_samples_land_at_their_labels(self):
        f = _field()
        target = np.concatenate([f.ranges, [f.ranges[-1] + 1.0]])[::-1]
        g = f.reindex(range=target)
        np.testing.assert_array_equal(g.ranges, target)
        np.testing.assert_array_equal(g.data[:, 1:], f.data[:, ::-1])
        assert np.all(np.isnan(g.data[:, 0]))

    def test_the_fill_value_is_the_callers(self):
        f = _field()
        g = f.reindex(fill_value=-1.0,
                      range=np.append(f.ranges, f.ranges[-1] + 1.0))
        assert np.all(g.data[:, -1] == -1.0)

    def test_a_target_equal_to_the_axis_is_the_field_unchanged(self):
        f = _field()
        np.testing.assert_array_equal(f.reindex(range=f.ranges).data, f.data)

    def test_a_target_missing_a_stored_label_is_refused(self):
        f = _field()
        with pytest.raises(ConfigurationError, match='never drops a stored'):
            f.reindex(range=f.ranges[1:])

    def test_a_repeated_label_is_refused(self):
        f = _field()
        with pytest.raises(ConfigurationError, match='repeats a label'):
            f.reindex(range=np.append(f.ranges, f.ranges[0]))

    def test_an_identity_axis_carries_its_identity(self):
        f = Field(data=np.ones((1, 2)),
                  coords={'depth': np.array([5.0]),
                          'frequency': np.array([100.0, 300.0])},
                  frequencies=[100.0, 300.0])
        g = f.reindex(frequency=[100.0, 200.0, 300.0])
        np.testing.assert_array_equal(g.frequencies, [100.0, 200.0, 300.0])

    def test_an_auxiliary_coordinate_is_filled_with_the_axis(self):
        g = _line_field().reindex(range=[100.0, 150.0, 200.0, 300.0, 400.0])
        dim, depths = g.aux_coords['receiver_depth']
        assert dim == 'range'
        np.testing.assert_array_equal(depths[[0, 2, 3, 4]],
                                      [10.0, 20.0, 30.0, 40.0])
        assert np.isnan(depths[1])


class TestAuxCoordsFollowTheirAxis:
    """An auxiliary coordinate is a label per sample of one axis: it is
    checked against that axis, and it follows the axis through every
    operation."""

    def test_an_entry_on_a_missing_axis_is_refused(self):
        with pytest.raises(ConfigurationError, match='not an axis'):
            Field(data=np.ones((1, 2)),
                  coords={'depth': np.array([0.0]),
                          'range': np.array([1.0, 2.0])},
                  aux_coords={'x': ('time', [1.0, 2.0])})

    def test_an_entry_of_the_wrong_length_is_refused(self):
        with pytest.raises(ConfigurationError, match='one label per sample'):
            Field(data=np.ones((1, 2)),
                  coords={'depth': np.array([0.0]),
                          'range': np.array([1.0, 2.0])},
                  aux_coords={'x': ('range', [1.0, 2.0, 3.0])})

    def test_an_infinite_label_is_refused_and_nan_is_not(self):
        coords = {'depth': np.array([0.0]), 'range': np.array([1.0, 2.0])}
        with pytest.raises(ConfigurationError, match='infinite label'):
            Field(data=np.ones((1, 2)), coords=coords,
                  aux_coords={'x': ('range', [1.0, np.inf])})
        f = Field(data=np.ones((1, 2)), coords=coords,
                  aux_coords={'x': ('range', [1.0, np.nan])})
        assert np.isnan(f.aux_coords['x'][1][1])

    def test_a_window_narrows_it_with_the_axis(self):
        g = _line_field().window(range=(150.0, 350.0))
        np.testing.assert_array_equal(g.aux_coords['receiver_depth'][1],
                                      [20.0, 30.0])

    def test_pinning_its_axis_drops_it(self):
        assert _line_field().at(range=200.0).aux_coords == {}

    def test_pinning_another_axis_keeps_it(self):
        g = _line_field().at(depth=0.0)
        np.testing.assert_array_equal(g.aux_coords['receiver_depth'][1],
                                      [10.0, 20.0, 30.0, 40.0])

    def test_a_shift_keeps_it_on_the_same_samples(self):
        g = _line_field().shift(range=5.0)
        np.testing.assert_array_equal(g.aux_coords['receiver_depth'][1],
                                      [10.0, 20.0, 30.0, 40.0])

    def test_a_relabelled_axis_drops_it(self):
        f = _line_field()
        g = f.replace(coords={'depth': np.array([0.0]),
                              'range': f.ranges + 1.0})
        assert g.aux_coords == {}

    def test_it_round_trips_through_to_dict(self):
        f = _line_field()
        g = Field.from_dict(f.to_dict())
        assert g.aux_coords.keys() == f.aux_coords.keys()
        np.testing.assert_array_equal(g.aux_coords['receiver_depth'][1],
                                      f.aux_coords['receiver_depth'][1])

    def test_a_field_without_one_writes_no_key(self):
        assert 'aux_coords' not in _field().to_dict()

    def test_it_round_trips_through_xarray(self):
        pytest.importorskip('xarray')
        f = _line_field()
        da = f.to_xarray()
        assert da.coords['receiver_depth'].dims == ('range',)
        g = Field.from_xarray(da)
        assert g.aux_coords['receiver_depth'][0] == 'range'
        np.testing.assert_array_equal(g.aux_coords['receiver_depth'][1],
                                      [10.0, 20.0, 30.0, 40.0])


class TestTheSourceIdentityIsAttributes:
    """``source_level_dB`` and ``source_weights`` are the result's own
    identity: a ``metadata`` holding one is refused, every derived result
    and every saved form carries them, and a file that keeps them in its
    metadata loads them as attributes."""

    @staticmethod
    def _levelled():
        return _field(data=np.full((4, 3), 0.01 + 0j),
                      phase_reference=PhaseReference.TRAVELLING_WAVE,
                      source_level_dB=180.0,
                      source_weights=np.array([1.0, -1.0j]))

    @pytest.mark.parametrize('tag,value', [('source_level_dB', 180.0),
                                           ('source_weights', [1.0, -1.0])])
    def test_a_metadata_holding_one_is_refused(self, tag, value):
        from uacpy.core.results import Modes
        with pytest.raises(ConfigurationError,
                           match=rf"metadata carries \['{tag}'\]"):
            _field(metadata={tag: value})
        with pytest.raises(ConfigurationError,
                           match=rf"metadata carries \['{tag}'\]"):
            Modes(k=np.array([0.3 + 0j]), phi=np.zeros((2, 1)),
                  depths=np.array([0.0, 50.0]), metadata={tag: value})

    def test_a_derived_field_carries_both(self):
        field = self._levelled()
        for derived in (field.at(range=200.0), field.to_dB(),
                        field.at_source_level()):
            assert derived.source_level_dB == 180.0
            np.testing.assert_array_equal(derived.source_weights,
                                          [1.0, -1.0j])

    def test_both_round_trip_through_the_dict(self):
        back = Field.from_dict(self._levelled().to_dict())
        assert back.source_level_dB == 180.0
        np.testing.assert_array_equal(back.source_weights, [1.0, -1.0j])
        assert not set(back.metadata) & {'source_level_dB', 'source_weights'}

    def test_a_dict_that_keeps_them_in_metadata_loads_them(self):
        d = self._levelled().to_dict()
        d['metadata'] = {**d['metadata'],
                         'source_level_dB': d.pop('source_level_dB'),
                         'source_weights': d.pop('source_weights')}
        back = Field.from_dict(d)
        assert back.source_level_dB == 180.0
        np.testing.assert_array_equal(back.source_weights, [1.0, -1.0j])
        assert not set(back.metadata) & {'source_level_dB', 'source_weights'}

    def test_both_round_trip_through_xarray(self):
        pytest.importorskip('xarray')
        back = Field.from_xarray(self._levelled().to_xarray())
        assert back.source_level_dB == 180.0
        np.testing.assert_array_equal(back.source_weights, [1.0, -1.0j])

    def test_a_superposed_field_carries_no_weights_but_keeps_the_level(self):
        slabs = [self._levelled().replace(source_depths=np.array([z]),
                                          source_weights=np.array([1.0, -1.0]))
                 for z in (10.0, 20.0)]
        for coherent in (True, False):
            summed = ResultStack(slabs, [10.0, 20.0]).superpose(
                coherent=coherent)
            assert summed.source_weights is None
            assert summed.source_level_dB == 180.0


class TestTheSynthesisInputsAreAttributes:
    """``speeds`` (a :class:`SoundSpeeds` record) and ``synthesis_floor`` are
    the Field's own attributes: a ``metadata`` naming one, or one of their
    metadata spellings, is refused; every derived Field and every saved form
    carries them; and a file that keeps them in its metadata loads them."""

    @staticmethod
    def _stated():
        return _two_path_grid().replace(
            speeds=SoundSpeeds(surface=1490.0, water_max=1520.0,
                               water_min=1480.0),
            synthesis_floor=512)

    @pytest.mark.parametrize('key,named', [
        ('speeds', 'speeds=...'), ('synthesis_floor', 'synthesis_floor=...'),
        ('c0', 'speeds=SoundSpeeds(surface=...)'),
        ('c_max', 'speeds=SoundSpeeds(water_max=...)'),
        ('tdelay_speed', 'speeds=SoundSpeeds(water_min=...)'),
        ('waveguide_c_min', 'speeds=SoundSpeeds(waveguide_min=...)'),
        ('n_time_samples', 'synthesis_floor=...')])
    def test_a_metadata_naming_one_is_refused(self, key, named):
        with pytest.raises(ConfigurationError,
                           match=rf"metadata carries \['{key}'\]") as info:
            _field(metadata={key: 1500.0})
        assert named in info.value.remediation

    def test_a_speeds_that_is_not_a_record_is_refused(self):
        with pytest.raises(ConfigurationError, match='not a SoundSpeeds'):
            _field(speeds={'surface': 1500.0})

    @pytest.mark.parametrize('speed', [0.0, -1500.0, np.nan, np.inf])
    def test_a_speed_that_is_not_finite_and_positive_is_refused(self, speed):
        with pytest.raises(ConfigurationError, match='finite positive'):
            SoundSpeeds(surface=speed)

    def test_a_derived_field_carries_both(self):
        field = self._stated()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            trace = field.to_time_trace(depth=10.0, range=150.0)
        for derived in (field.at(frequency=200.0), field.to_dB(), trace):
            assert derived.speeds == field.speeds
            assert derived.synthesis_floor == 512

    def test_both_round_trip_through_the_dict(self):
        field = self._stated()
        back = Field.from_dict(field.to_dict())
        assert back.speeds == field.speeds
        assert back.synthesis_floor == 512

    def test_a_dict_that_keeps_them_in_metadata_loads_them(self):
        d = self._stated().to_dict()
        del d['speeds'], d['synthesis_floor']
        d['metadata'] = {'c0': 1490.0, 'c_max': 1520.0,
                         'tdelay_speed': 1480.0, 'waveguide_c_min': 1450.0,
                         'waveguide_c_max': 1700.0, 'n_time_samples': 512,
                         'dr': 5.0}
        back = Field.from_dict(d)
        assert back.speeds == SoundSpeeds(
            surface=1490.0, water_max=1520.0, water_min=1480.0,
            waveguide_min=1450.0, waveguide_max=1700.0)
        assert back.synthesis_floor == 512
        assert back.metadata == {'dr': 5.0}

    def test_both_round_trip_through_xarray(self):
        pytest.importorskip('xarray')
        field = self._stated()
        back = Field.from_xarray(field.to_xarray())
        assert back.speeds == field.speeds
        assert back.synthesis_floor == 512
        assert not set(back.metadata) & {'speeds_surface', 'synthesis_floor'}

    def test_an_xarray_that_keeps_them_as_metadata_attrs_loads_them(self):
        pytest.importorskip('xarray')
        da = _two_path_grid().to_xarray()
        da.attrs.update({'c0': 1490.0, 'waveguide_c_min': 1450.0,
                         'waveguide_c_max': 1700.0, 'n_time_samples': 512})
        back = Field.from_xarray(da)
        assert back.speeds == SoundSpeeds(surface=1490.0, waveguide_min=1450.0,
                                          waveguide_max=1700.0)
        assert back.synthesis_floor == 512


@pytest.mark.parametrize('reported, stored', [
    (1018.000000000011, 1018),   # a few ULP above a whole count stays put
    (1024.9999999999, 1025),     # a few ULP below is not truncated away
    (1025.0 - 1e-9, 1025),
    (1025.0 + 1e-9, 1025),
    (1018.4, 1019),              # a fractional count is covered, not cut
    (512, 512),
])
def test_synthesis_floor_is_the_least_whole_count_covering_the_report(
        reported, stored):
    """``synthesis_floor`` keeps every sample of a real-valued count: at
    ``2**k + 1`` a truncation would halve the power-of-two FFT."""
    f = _two_path_grid().replace(synthesis_floor=reported)
    assert f.synthesis_floor == stored
    assert isinstance(f.synthesis_floor, int)
