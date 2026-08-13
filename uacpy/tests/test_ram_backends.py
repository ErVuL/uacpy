"""Tests for the RAM multi-backend dispatcher and the Collins-style I/O."""

import warnings
from pathlib import Path

import numpy as np
import pytest

from uacpy.core.environment import (
    BoundaryProperties,
    Environment,
    SeabedColumn,
    SedimentLayer,
)
from uacpy import Field
from uacpy.core.receiver import Receiver
from uacpy.core.source import Source
from uacpy.io.ramsurf_writer import write_ramin
from uacpy.models import RAM, RunMode
from uacpy.core.exceptions import (
    ConfigurationError,
    ModelExecutionError,
    UnsupportedFeatureError,
)
from uacpy.core.constants import TL_MAX_DB


class _FakeProc:
    """Stands in for a subprocess that exits 0 and writes no output file."""
    returncode = 0
    stdout = ''
    stderr = ''


# ─── Fixtures ──────────────────────────────────────────────────────────────


def _fluid_bottom():
    return BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1600.0, density=1.5, attenuation=0.5,
    )


def _fluid_layered_bottom():
    """A layered bottom with no shear — the regime RAMGEO is built for."""
    return SeabedColumn(
        layers=[
            SedimentLayer(
                thickness=15, sound_speed=1650, density=1.6, attenuation=0.4,
            ),
        ],
        halfspace=BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1900, density=1.9, attenuation=0.2,
        ),
    )


def _elastic_bottom():
    return SeabedColumn(
        layers=[
            SedimentLayer(
                thickness=20, sound_speed=1700, density=1.5,
                attenuation=0.5, shear_speed=400, shear_attenuation=1.0,
            ),
        ],
        halfspace=BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1900, density=2.0, attenuation=0.1,
            shear_speed=600, shear_attenuation=0.5,
        ),
    )


def _rough_altimetry():
    # Surface depression — ramsurf only models h <= 0.
    return [(0.0, 0.0), (1000.0, -2.0), (5000.0, 0.0)]


def _env(*, bottom, altimetry=None):
    return Environment(
        name='test', bathymetry=100.0, ssp=1500.0,
        bottom=bottom, altimetry=altimetry,
    )


# ─── Collins bottom-profile depth reference ────────────────────────────────


@pytest.mark.requires_binary  # constructs RAM (resolves its binary)
class TestCollinsDepthReference:
    """ramgeo/ramsurf read bottom-profile depths as depth-below-seafloor:
    their ``matrc`` restarts the bottom-array index at ``ii=1`` on the grid
    point below the local seafloor (``ii=1; do i=iz+1,nz+2`` —
    ``ramgeo1.5.f:262-269``, ``ramsurf1.5.f:249-256``). rams0.5's ``matrc``
    indexes the same arrays absolutely (``lamb(i)`` over ``i=iz+1,ib+2``,
    ``rams0.5.f:491-498``), so its depths run from z=0. The segment builder
    must emit each convention."""

    def _segments(self, kind):
        env = _env(bottom=_fluid_layered_bottom())     # 100 m water, 15 m layer
        return RAM(verbose=False)._collins_range_segments(
            env, kind, zmax=400.0, freq=250.0)

    @pytest.mark.parametrize('kind', ['ramgeo', 'ramsurf'])
    def test_relative_for_ramgeo_ramsurf(self, kind):
        seg = self._segments(kind)[0]
        depths = [d for d, _ in seg['bottom_c']]
        assert depths[0] == pytest.approx(0.0)          # top of sediment
        assert depths[1] == pytest.approx(15.0)         # layer bottom
        assert max(depths) == pytest.approx(300.0)      # zmax - seafloor
        # layer speed then half-space speed
        assert seg['bottom_c'][0][1] == pytest.approx(1650.0)
        assert seg['bottom_c'][-1][1] == pytest.approx(1900.0)

    def test_absolute_for_rams(self):
        seg = self._segments('rams')[0]
        depths = [d for d, _ in seg['bottom_c']]
        assert depths[0] == pytest.approx(100.0)        # seafloor (absolute)
        assert depths[1] == pytest.approx(115.0)        # layer bottom
        assert max(depths) == pytest.approx(400.0)      # zmax (absolute)


@pytest.mark.requires_binary
class TestBackendScopedKnobWarnings:
    """Row 5 of ``ram.in`` holds ``ns rs`` on the fluid Collins codes and
    ``irot theta`` on rams0.5 — mutually exclusive, so overriding the pair
    the selected backend does not read is discarded silently unless warned.
    """

    @pytest.mark.parametrize('backend,kw,expect', [
        ('rams', dict(ns_stability=5, rs_stability=1234.0),
         'ns_stability, rs_stability'),
        ('rams', dict(rams_theta=60.0), None),
        ('ramgeo', dict(rams_theta=60.0, rams_irot=0),
         'rams_theta, rams_irot'),
        ('ramgeo', dict(ns_stability=5, rs_stability=1234.0), None),
        ('mpiramS', dict(rams_theta=60.0), 'rams_theta'),
        ('ramsurf', dict(rams_irot=0), 'rams_irot'),
    ])
    def test_unreadable_knobs_warn(self, backend, kw, expect):
        import warnings as _w
        m = RAM(verbose=False, **kw)
        with _w.catch_warnings(record=True) as w:
            _w.simplefilter('always')
            m._warn_on_mpirams_only_overrides(backend)
        hits = [str(x.message) for x in w if 'ignores these' in str(x.message)]
        if expect is None:
            assert not hits, f"{backend} reads {kw} but warned: {hits}"
        else:
            assert any(expect in h for h in hits), \
                f"{backend} discards {kw} without warning; got {hits}"


# ─── Backend selection (no binary needed) ─────────────────────────────────


@pytest.mark.requires_binary  # constructs RAM (resolves its binary) to probe select_backend
class TestBackendSelection:
    """Pure-Python dispatch logic — no native binaries required."""

    def test_fluid_flat_simple_selects_mpirams(self):
        # A simple half-space fluid bottom auto-routes to mpiramS (native
        # half-space, more accurate than ramgeo's synthetic-layer wrapping).
        env = _env(bottom=_fluid_bottom())
        assert RAM(verbose=False, dr=20.0, dz=2.0).select_backend(env) == 'mpiramS'

    def test_fluid_flat_layered_narrowband_selects_ramgeo(self):
        env = _env(bottom=_fluid_layered_bottom())
        ram = RAM(verbose=False, dr=20.0, dz=2.0)
        assert ram.select_backend(env, RunMode.COHERENT_TL) == 'ramgeo'
        # run_mode=None defaults to the COHERENT_TL choice
        assert ram.select_backend(env) == 'ramgeo'

    def test_fluid_flat_layered_broadband_stays_mpirams(self):
        env = _env(bottom=_fluid_layered_bottom())
        ram = RAM(verbose=False, dr=20.0, dz=2.0)
        assert ram.select_backend(env, RunMode.BROADBAND) == 'mpiramS'
        assert ram.select_backend(env, RunMode.TIME_SERIES) == 'mpiramS'

    def test_ramgeo_accepts_forced_simple_bottom(self):
        # ramgeo runs on a plain half-space when forced: the writer wraps it
        # as a synthetic single layer.
        env = _env(bottom=_fluid_bottom())
        assert RAM(verbose=False, dr=20.0, dz=2.0,
                   backend='ramgeo').select_backend(env) == 'ramgeo'

    def test_backend_override_forces_choice(self):
        env = _env(bottom=_fluid_layered_bottom())
        assert RAM(verbose=False, dr=20.0, dz=2.0,
                   backend='mpiramS').select_backend(env) == 'mpiramS'
        assert RAM(verbose=False, dr=20.0, dz=2.0,
                   backend='ramgeo').select_backend(env) == 'ramgeo'

    def test_backend_override_unknown_raises(self):
        with pytest.raises(ConfigurationError, match="not a known backend"):
            RAM(verbose=False, dr=20.0, dz=2.0, backend='nope')

    def test_backend_override_fluid_on_elastic_raises(self):
        env = _env(bottom=_elastic_bottom())
        for bk in ('mpiramS', 'ramgeo', 'ramsurf'):
            with pytest.raises(ConfigurationError, match="fluid PE"):
                RAM(verbose=False, dr=20.0, dz=2.0,
                    backend=bk).select_backend(env)

    def test_backend_override_flat_on_rough_raises(self):
        env = _env(bottom=_fluid_bottom(), altimetry=_rough_altimetry())
        for bk in ('mpiramS', 'ramgeo', 'rams'):
            with pytest.raises(ConfigurationError, match="flat pressure-release"):
                RAM(verbose=False, dr=20.0, dz=2.0,
                    backend=bk).select_backend(env)

    def test_backend_override_ramsurf_needs_altimetry(self):
        env = _env(bottom=_fluid_bottom())  # flat
        with pytest.raises(ConfigurationError, match="variable surface"):
            RAM(verbose=False, dr=20.0, dz=2.0,
                backend='ramsurf').select_backend(env)

    def test_elastic_flat_selects_rams(self):
        env = _env(bottom=_elastic_bottom())
        assert RAM(verbose=False, dr=20.0, dz=2.0).select_backend(env) == 'rams'

    def test_fluid_rough_selects_ramsurf(self):
        env = _env(bottom=_fluid_bottom(), altimetry=_rough_altimetry())
        assert RAM(verbose=False, dr=20.0, dz=2.0).select_backend(env) == 'ramsurf'

    def test_elastic_rough_raises_not_implemented(self):
        env = _env(bottom=_elastic_bottom(), altimetry=_rough_altimetry())
        with pytest.raises(UnsupportedFeatureError, match="elastic bottom \\+ sea-surface altimetry"):
            RAM(verbose=False, dr=20.0, dz=2.0).select_backend(env)


# ─── SeabedColumn → piecewise ────────────────────────────────────────────


class TestPiecewiseBreakpoints:

    def test_two_layers_emit_step_function(self):
        lb = SeabedColumn(
            layers=[
                SedimentLayer(thickness=10, sound_speed=1550,
                              density=1.3, attenuation=0.5),
                SedimentLayer(thickness=20, sound_speed=1650,
                              density=1.7, attenuation=0.3),
            ],
            halfspace=BoundaryProperties(
                acoustic_type='half-space',
                sound_speed=1800, density=2.0, attenuation=0.1,
            ),
        )
        bp = lb.to_piecewise_breakpoints(
            seafloor_depth=100, zmax=200,
            properties=('sound_speed', 'density', 'attenuation'),
        )
        depths = [d for d, _ in bp['sound_speed']]
        values = [v for _, v in bp['sound_speed']]
        # Three layers (two sediment + halfspace), each emits two breakpoints
        # at top/bottom with the same value
        assert depths == [100, 110, 110, 130, 130, 200]
        assert values == [1550, 1550, 1650, 1650, 1800, 1800]

    def test_elastic_properties_round_trip(self):
        lb = SeabedColumn(
            layers=[SedimentLayer(thickness=5, sound_speed=1700,
                                  density=1.6, attenuation=0.4,
                                  shear_speed=350, shear_attenuation=1.5)],
            halfspace=BoundaryProperties(
                acoustic_type='half-space',
                sound_speed=2000, density=2.2, attenuation=0.2,
                shear_speed=700, shear_attenuation=0.3,
            ),
        )
        bp = lb.to_piecewise_breakpoints(
            seafloor_depth=50, zmax=200,
            properties=('shear_speed', 'shear_attenuation'),
        )
        # Shear speed step from 350 → 700 across the sediment / half-space boundary
        assert bp['shear_speed'][1] == (55.0, 350.0)
        assert bp['shear_speed'][2] == (55.0, 700.0)
        assert bp['shear_attenuation'][1] == (55.0, 1.5)
        assert bp['shear_attenuation'][2] == (55.0, 0.3)

    def test_missing_property_defaults_to_zero(self):
        # SedimentLayer without explicit shear → 0.0 emitted
        lb = SeabedColumn(
            layers=[SedimentLayer(thickness=5, sound_speed=1600,
                                  density=1.5, attenuation=0.5)],
            halfspace=BoundaryProperties(
                acoustic_type='half-space',
                sound_speed=1800, density=2.0, attenuation=0.1),
        )
        bp = lb.to_piecewise_breakpoints(
            seafloor_depth=50, zmax=200,
            properties=('shear_speed',),
        )
        assert all(v == 0.0 for _, v in bp['shear_speed'])


# ─── Writer round-trip ────────────────────────────────────────────────────


class TestRamInWriter:

    def test_ramsurf_kind_includes_surface_block(self, tmp_path):
        out = tmp_path / 'ram.in'
        write_ramin(
            str(out), kind='ramsurf',
            fc=100.0, zs=50.0, zr_line=50.0,
            rmax=5000.0, dr=10.0, ndr=2,
            zmax=400.0, dz=1.0, ndz=2, zmplt=200.0,
            c0=1500.0, np_pade=4,
            surface=[(0.0, 0.0), (5000.0, 0.0)],
            bathymetry=[(0.0, 100.0), (5000.0, 100.0)],
            range_segments=[dict(
                range=0.0,
                water_ssp=[(0.0, 1500.0), (100.0, 1500.0)],
                bottom_c=[(0.0, 1600.0), (400.0, 1600.0)],
                bottom_rho=[(0.0, 1.5), (400.0, 1.5)],
                bottom_attn=[(0.0, 0.5), (400.0, 0.5)],
            )],
        )
        text = out.read_text()
        # surface block + bathy + 4 profile blocks = 6 terminators
        assert text.count('-1 -1') == 6

    def test_rams_kind_uses_irot_theta_and_six_profiles(self, tmp_path):
        out = tmp_path / 'rams.in'
        write_ramin(
            str(out), kind='rams',
            fc=100.0, zs=50.0, zr_line=50.0,
            rmax=5000.0, dr=10.0, ndr=2,
            zmax=400.0, dz=1.0, ndz=2, zmplt=200.0,
            c0=1500.0, np_pade=4, irot=1, theta=45.0,
            bathymetry=[(0.0, 100.0), (5000.0, 100.0)],
            range_segments=[dict(
                range=0.0,
                water_ssp=[(0.0, 1500.0), (100.0, 1500.0)],
                bottom_c=[(0.0, 1600.0), (400.0, 1600.0)],
                bottom_cs=[(0.0, 400.0), (400.0, 400.0)],
                bottom_rho=[(0.0, 1.5), (400.0, 1.5)],
                bottom_attn=[(0.0, 0.5), (400.0, 0.5)],
                bottom_attns=[(0.0, 1.0), (400.0, 1.0)],
            )],
        )
        text = out.read_text()
        # Row 5 for rams carries (irot, theta) — note 45.0 instead of 0.0
        assert '1500.000000 4 1 45.000000' in text
        # bath + 6 profile blocks (cw, cp, cs, rho, attnp, attns) = 7 terminators
        assert text.count('-1 -1') == 7

    def test_ramgeo_kind_is_fluid_flat(self, tmp_path):
        """ramgeo = ramsurf format minus the surface block, no shear: row-5
        carries (ns, rs) and there are only 4 profile blocks per range."""
        out = tmp_path / 'ramgeo.in'
        write_ramin(
            str(out), kind='ramgeo',
            fc=100.0, zs=50.0, zr_line=50.0,
            rmax=5000.0, dr=10.0, ndr=2,
            zmax=400.0, dz=1.0, ndz=2, zmplt=200.0,
            c0=1500.0, np_pade=4, ns_stab=1, rs_stab=0.0,
            bathymetry=[(0.0, 100.0), (5000.0, 100.0)],
            range_segments=[dict(
                range=0.0,
                water_ssp=[(0.0, 1500.0), (100.0, 1500.0)],
                bottom_c=[(0.0, 1600.0), (400.0, 1600.0)],
                bottom_rho=[(0.0, 1.5), (400.0, 1.5)],
                bottom_attn=[(0.0, 0.5), (400.0, 0.5)],
            )],
        )
        text = out.read_text()
        # Row 5 carries (ns, rs) — fluid, no irot/theta
        assert '1500.000000 4 1 0.000000' in text
        # bathy + 4 profile blocks (cw, cp, rho, attnp) = 5 terminators; no
        # surface block (cf. ramsurf's 6).
        assert text.count('-1 -1') == 5

    def test_rams_requires_shear_profiles(self, tmp_path):
        out = tmp_path / 'rams.in'
        with pytest.raises(ConfigurationError, match="bottom_cs and bottom_attns"):
            write_ramin(
                str(out), kind='rams',
                fc=100.0, zs=50.0, zr_line=50.0,
                rmax=5000.0, dr=10.0, ndr=2,
                zmax=400.0, dz=1.0, ndz=2, zmplt=200.0,
                c0=1500.0, np_pade=4,
                bathymetry=[(0.0, 100.0), (5000.0, 100.0)],
                range_segments=[dict(
                    range=0.0,
                    water_ssp=[(0.0, 1500.0)],
                    bottom_c=[(0.0, 1600.0)],
                    bottom_rho=[(0.0, 1.5)],
                    bottom_attn=[(0.0, 0.5)],
                )],
            )


# ─── Integration: each Collins binary end-to-end ─────────────────────────


@pytest.mark.requires_binary
class TestCollinsBinaries:
    """Exercise the actual rams0.5 / ramsurf1.5 binaries via the wrapper.
    Skipped automatically when the binaries are not built."""

    def _src_rcv(self):
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.linspace(500.0, 5000.0, 20),
        )
        return src, rcv

    def test_ramsurf_rough_surface_runs_clean(self):
        env = _env(bottom=_fluid_bottom(), altimetry=_rough_altimetry())
        src, rcv = self._src_rcv()
        ram = RAM(verbose=False, dr=20.0, dz=2.0)
        result = ram.run(env, src, rcv, run_mode=RunMode.COHERENT_TL)
        assert isinstance(result, Field)
        assert result.backend == 'ramsurf'
        assert result.data.shape == (3, 20)
        assert np.all(np.isfinite(result.data))
        # Sensible TL range (no gain, bounded loss)
        assert 0 < result.tl.min() < 60
        assert result.tl.max() < 200

    def test_rams_elastic_runs(self):
        env = _env(bottom=_elastic_bottom())
        src, rcv = self._src_rcv()
        ram = RAM(verbose=False, dr=20.0, dz=2.0)
        result = ram.run(env, src, rcv, run_mode=RunMode.COHERENT_TL)
        assert isinstance(result, Field)
        assert result.backend == 'rams'
        assert result.data.shape == (3, 20)
        assert np.all(np.isfinite(result.data))

    def test_ramgeo_layered_runs(self):
        """A fluid layered Pekeris auto-routes to RAMGEO and gives a sane,
        finite, bounded TL field."""
        env = _env(bottom=_fluid_layered_bottom())
        src, rcv = self._src_rcv()
        result = RAM(verbose=False, dr=20.0, dz=1.0).run(
            env, src, rcv, run_mode=RunMode.COHERENT_TL)
        assert isinstance(result, Field)
        assert result.backend == 'ramgeo'
        assert result.data.shape == (3, 20)
        assert np.all(np.isfinite(result.data))
        assert 0 < result.tl.min() < 80
        assert result.tl.max() < 200

    def test_ramgeo_agrees_with_mpirams(self):
        """RAMGEO and mpiramS are independent PE codes; on the same fluid
        layered Pekeris their TL envelopes should agree closely."""
        env = _env(bottom=_fluid_layered_bottom())
        src, rcv = self._src_rcv()
        geo = RAM(verbose=False, dr=20.0, dz=1.0, backend='ramgeo').run(
            env, src, rcv)
        mpi = RAM(verbose=False, dr=20.0, dz=1.0, backend='mpiramS').run(
            env, src, rcv)
        d = np.abs(geo.tl - mpi.tl)
        # The bound is loose because two independent PE codes discretise a
        # 15 m layer differently, and still discriminating because the thing
        # under test — ramgeo reading bottom depths from the local seafloor,
        # mpiramS from z=0 (see TestCollinsDepthReference) — misplaces the
        # whole sub-bottom by the water depth when it is wrong.
        assert np.nanmedian(d) < 3.0

    def test_ramgeo_broadband_runs(self):
        """A forced ramgeo serves BROADBAND via its complex-envelope patch —
        capability parity with rams0.5 / ramsurf1.5."""
        env = _env(bottom=_fluid_layered_bottom())
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=np.array([25.0, 50.0, 75.0]),
                       ranges=np.linspace(500.0, 5000.0, 20))
        # Smoke test (type/shape/finite only): a coarse grid keeps it cheap.
        f = RAM(verbose=False, np_pade=6, dr=8.0, dz=1.0, zmax=200.0,
                Q=2.0, T=2.0, backend='ramgeo').run(
            env, src, rcv, run_mode=RunMode.BROADBAND)
        assert isinstance(f, Field)
        assert f.backend == 'ramgeo'
        assert f.phase_reference == 'travelling_wave'
        assert f.data.ndim == 3 and f.frequencies.size > 1
        assert np.all(np.isfinite(f.data))

    @pytest.mark.slow
    def test_mpirams_broadband_rd_ssp_short_range(self):
        """BROADBAND with a range-dependent SSP at rmax < 5 km must not take
        mpiramS's ihorz=1 path (nrp=nint(rmax/10000)=0 → zero-length allocate
        → SIGABRT); the 2-D ssp.dat already carries the per-range profiles."""
        from uacpy.core.ssp import SoundSpeedProfile
        ssp = SoundSpeedProfile.from_2d(
            depths=[0.0, 50.0, 100.0], ranges=[0.0, 1500.0, 3000.0],
            matrix=np.array([[1500.0, 1505.0, 1510.0],
                             [1495.0, 1500.0, 1505.0],
                             [1490.0, 1495.0, 1500.0]]))
        env = Environment(name='rd', bathymetry=100.0, ssp=ssp,
                          bottom=_fluid_bottom())
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=np.array([30.0, 60.0]),
                       ranges=np.linspace(500.0, 3000.0, 10))
        f = RAM(verbose=False, np_pade=6, dr=8.0, dz=1.0, zmax=200.0,
                Q=2.0, T=2.0, backend='mpiramS').run(
            env, src, rcv, run_mode=RunMode.BROADBAND)
        assert isinstance(f, Field)
        assert np.all(np.isfinite(f.data))

    @pytest.mark.slow
    def test_mpirams_broadband_below_domain_depth_is_nan(self):
        """A receiver depth below the PE domain must come back NaN in
        BROADBAND, matching the COHERENT_TL below-domain convention — not a
        plausible-looking edge-extrapolated H(f)."""
        env = _env(bottom=_fluid_bottom())
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=np.array([50.0, 300.0]),   # 300 m > zmax=200
                       ranges=np.linspace(500.0, 3000.0, 10))
        with pytest.warns(UserWarning, match="below the model's resolvable"):
            f = RAM(verbose=False, np_pade=6, dr=8.0, dz=1.0, zmax=200.0,
                    Q=2.0, T=2.0, backend='mpiramS').run(
                env, src, rcv, run_mode=RunMode.BROADBAND)
        assert np.all(np.isfinite(f.data[0]))            # in-domain row
        assert np.all(np.isnan(f.data[1].real))          # below-domain row

    @pytest.mark.slow
    def test_collins_backend_broadband_returns_transfer_function(self):
        """ramsurf BROADBAND emits the patched complex envelope and the
        wrapper assembles an engineering travelling-wave H(f), tagged
        identically to mpiramS so synthesize_time_series treats both
        backends the same."""
        env = _env(bottom=_fluid_bottom(), altimetry=_rough_altimetry())
        src, rcv = self._src_rcv()
        # Smoke test (type/shape/finite only): a coarse grid keeps it cheap.
        ram = RAM(verbose=False, np_pade=6, dr=8.0, dz=1.0, zmax=200.0,
                  Q=2.0, T=2.0)
        f = ram.run(env, src, rcv, run_mode=RunMode.BROADBAND)
        assert isinstance(f, Field)
        assert f.backend == 'ramsurf'
        assert f.phase_reference == 'travelling_wave'
        # Shape: (n_d, n_r, n_f) — trailing axis is variable.
        assert f.data.ndim == 3
        assert f.data.shape[0] == len(rcv.depths)
        assert f.data.shape[1] == len(rcv.ranges)
        assert f.data.shape[2] == f.frequencies.size
        assert f.frequencies.size > 1
        assert np.all(np.isfinite(f.data))

    def test_collins_backend_time_series_requires_waveform(self):
        env = _env(bottom=_fluid_bottom(), altimetry=_rough_altimetry())
        src, rcv = self._src_rcv()
        ram = RAM(verbose=False, dr=20.0, dz=2.0)
        with pytest.raises(ConfigurationError, match="source_waveform"):
            ram.run(env, src, rcv, run_mode=RunMode.TIME_SERIES)


# ─── Per-backend pinned TL on Pekeris waveguide ─────────────────────────────


@pytest.mark.requires_binary
@pytest.mark.slow
class TestRamPekerisReference:
    """Per-backend pinned-TL agreement against Kraken on a Pekeris waveguide.

    Cross-model RMSE tests average over a window and can hide localized
    errors; this asserts agreement at specific (range, depth) sample
    points for each of the three RAM dispatcher backends. Kraken is the
    reference (mode sum on Pekeris is essentially the analytical
    solution).

    ``tol_db=4.5`` bounds a *method* difference, not numerical noise: a Padé
    PE and an exact mode sum part company by genuine wide-angle error, which
    grows with the water/seabed speed contrast. The same value is shared with
    the strict xfail in :class:`TestMpiramsAutoGridIsAccurateEnough`, so it is
    what separates a pinned grid from the auto grid — loosening it turns that
    open defect into a silent pass.
    """

    _SRC_DEPTH = 36.0
    _SRC_FREQ = 50.0
    _RCV_DEPTHS = np.array([20.0, 36.0, 80.0])
    _RCV_RANGES = np.linspace(500.0, 8000.0, 200)
    # Probe windows are 600 m wide so the median over the window smooths
    # out modal interference fringes — the test catches envelope-level
    # disagreements (wrapper bugs in scaling, attenuation, or carrier
    # bake-in) rather than fringe-phase offsets between PE and modes.
    _PROBE_WINDOWS = ((1500.0, 2100.0), (3500.0, 4100.0), (6500.0, 7100.0))
    _PROBE_DEPTHS = (20.0, 36.0)

    def _src_rcv(self):
        src = Source(depths=self._SRC_DEPTH, frequencies=self._SRC_FREQ)
        rcv = Receiver(depths=self._RCV_DEPTHS, ranges=self._RCV_RANGES)
        return src, rcv

    def _kraken_reference(self, env, src, rcv):
        from uacpy.models import Kraken
        return Kraken(verbose=False).run(
            env, src, rcv, run_mode=RunMode.COHERENT_TL,
        )

    def _window_median_tl(self, field, depth, r_lo, r_hi):
        ranges = np.asarray(field.ranges)
        depths = np.asarray(field.depths)
        zi = int(np.argmin(np.abs(depths - depth)))
        mask = (ranges >= r_lo) & (ranges <= r_hi)
        tl_strip = field.tl[zi, mask]
        return float(np.median(tl_strip[np.isfinite(tl_strip)]))

    def _assert_window_agreement(self, ram_field, ref_field, tol_db, label):
        for (r_lo, r_hi) in self._PROBE_WINDOWS:
            for z in self._PROBE_DEPTHS:
                tl_ram = self._window_median_tl(ram_field, z, r_lo, r_hi)
                tl_ref = self._window_median_tl(ref_field, z, r_lo, r_hi)
                assert np.isfinite(tl_ram), (
                    f"{label}: NaN median in z={z}, r=[{r_lo},{r_hi}]"
                )
                assert abs(tl_ram - tl_ref) < tol_db, (
                    f"{label}: window-median TL mismatch at z={z} m, "
                    f"r=[{r_lo},{r_hi}] m — RAM={tl_ram:.2f} dB, "
                    f"ref={tl_ref:.2f} dB, tol={tol_db} dB"
                )

    def test_mpirams_pekeris_fluid(self):
        """``dr`` is pinned so this measures the wrapper — scaling,
        attenuation, carrier bake-in — against Kraken, not the auto grid.
        The auto grid has its own defect, pinned in
        :class:`TestMpiramsAutoGridIsAccurateEnough`; leaving it in here would
        conflate a grid-selection error with a wrapper error."""
        env = _env(bottom=BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1700.0, density=1.7, attenuation=0.5,
        ))
        src, rcv = self._src_rcv()
        ram_field = RAM(verbose=False, dr=20.0).run(
            env, src, rcv, run_mode=RunMode.COHERENT_TL,
        )
        assert ram_field.backend == 'mpiramS'
        ref_field = self._kraken_reference(env, src, rcv)
        self._assert_window_agreement(
            ram_field, ref_field, tol_db=4.5, label='mpiramS',
        )

    def test_rams_pekeris_elastic(self):
        env = _env(bottom=_elastic_bottom())
        src, rcv = self._src_rcv()
        ram_field = RAM(verbose=False).run(
            env, src, rcv, run_mode=RunMode.COHERENT_TL,
        )
        assert ram_field.backend == 'rams'
        ref_field = self._kraken_reference(env, src, rcv)
        self._assert_window_agreement(
            ram_field, ref_field, tol_db=4.5, label='rams0.5',
        )

    def test_ramsurf_pekeris_flat_altimetry(self):
        # Flat altimetry triggers ramsurf dispatch on a fluid Pekeris;
        # answer should still match Kraken within the same tolerance as
        # mpiramS since the surface is undeformed.
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1700.0, density=1.7, attenuation=0.5,
        )
        env_ram = _env(
            bottom=bottom,
            altimetry=[(0.0, 0.0), (10000.0, 0.0)],
        )
        env_ref = _env(bottom=bottom)
        src, rcv = self._src_rcv()
        ram_field = RAM(verbose=False).run(
            env_ram, src, rcv, run_mode=RunMode.COHERENT_TL,
        )
        assert ram_field.backend == 'ramsurf'
        ref_field = self._kraken_reference(env_ref, src, rcv)
        self._assert_window_agreement(
            ram_field, ref_field, tol_db=4.5, label='ramsurf1.5',
        )


def test_rams_elastic_field_is_physical_on_a_fast_shear_seabed():
    """A fast-shear elastic seabed yields a physical field on the auto grid.

    The Collins elastic march stays convergent because the shear wavelength
    bounds Δz (:func:`rams_dz_shear_cap`); on a grid coarser than λ_s/8 it
    diverges instead, and the wrapper then clamps the wreckage to 200 dB. So
    the discriminating assertions are that nothing is clamped and nothing is
    non-finite — not merely that TL is non-negative.
    """
    import uacpy
    el = uacpy.BoundaryProperties(sound_speed=1800.0, density=2.0,
                                  attenuation=0.1, shear_speed=800.0,
                                  shear_attenuation=0.2)
    env = uacpy.Environment(bathymetry=200.0, ssp=1500.0,
                            bottom=uacpy.Bottom([uacpy.SeabedColumn([], el)]))
    src = Source(depths=50.0, frequencies=100.0)
    rcv = Receiver(depths=np.linspace(10, 190, 8), ranges=np.linspace(500, 6000, 20))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        tl = np.asarray(RAM(backend='rams').compute_tl(env, src, rcv).tl)
    assert not [w for w in caught
                if 'unphysically negative' in str(w.message)], \
        "the march diverged and the field was clamped"
    assert np.isfinite(tl).all(), "no sample may be NaN/inf"
    assert (tl >= 0).all(), "TL must be >= 0 (no field gain)"
    assert not (tl == 200.0).any(), "no sample may sit on the clamp"


class TestCollinsArrayLimits:
    """Overrunning a Collins binary's fixed arrays must raise before launch.

    All three are locally enlarged over upstream and all three ``stop`` with a
    diagnostic on an overrun, but they exit before writing ``tl.grid``, so the
    failure otherwise reaches the caller as a ``FileFormatError`` about a
    truncated file. ``mz`` is consumed at different rates: rams0.5 interleaves
    the elastic field vector (2*nz+4), the fluid codes use nz+2.
    """

    @staticmethod
    def _env(n_points):
        from uacpy.core.environment import Bathymetry
        r = np.linspace(0.0, 20000.0, n_points)
        d = 200.0 + 10.0 * np.sin(r / 3000.0)
        return Environment(
            name='rd', bathymetry=Bathymetry(ranges=r, depths=d), ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.5, shear_speed=300.0))

    @staticmethod
    def _fluid_env():
        return Environment(
            name='flat', bathymetry=200.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.5))

    def test_rams_rejects_more_bathymetry_points_than_mr(self):
        from uacpy.core.exceptions import ConfigurationError
        src = Source(depths=50.0, frequencies=50.0)
        rcv = Receiver(depths=100.0, ranges=np.array([5000.0]))
        with pytest.raises(ConfigurationError, match="mr=505"):
            RAM(verbose=False, backend='rams').run(self._env(600), src, rcv)

    @pytest.mark.parametrize('kind,mz', [('ramgeo', 20002), ('rams', 40004)])
    def test_pinned_dz_past_mz_is_rejected(self, kind, mz):
        """A pinned ``dz`` bypasses the MAX_DEPTH_POINTS cap, so nz can exceed
        mz. The binary stops with 'Need to increase parameter mz', writes no
        output, and the reader then reports a truncated tl.grid — the mz bound
        has to be caught before launch instead."""
        from uacpy.core.exceptions import ConfigurationError
        src = Source(depths=50.0, frequencies=50.0)
        rcv = Receiver(depths=100.0, ranges=np.array([5000.0]))
        env = self._env(5) if kind == 'rams' else self._fluid_env()
        with pytest.raises(ConfigurationError, match=f"mz={mz}"):
            RAM(verbose=False, backend=kind, dz=0.005).run(env, src, rcv)

    def test_auto_dz_past_mz_is_coarsened_not_rejected(self):
        """An auto-picked grid is uacpy's own choice, so a bound it cannot meet
        is coarsened with a warning rather than raised (same policy as
        MAX_DEPTH_POINTS). Reachable with a pinned zmax and an auto dz."""
        m = RAM(verbose=False, backend='ramgeo', zmax=100000.0)
        with pytest.warns(UserWarning, match="fit the binary's depth arrays"):
            dr, dz, zmax = m._resolve_collins_grid(
                self._fluid_env(), 50.0, 'ramgeo', 5000.0, None, None, None)
        needed, mz, _ = m._collins_mz_budget('ramgeo', zmax)
        assert needed(dz) <= mz

    def test_broadband_auto_dz_is_coarsened_like_narrowband(self):
        """The broadband sweep hands its own Lytaev ``dz`` down as an
        override; that is still uacpy's choice, so it is coarsened to fit
        ``mz`` rather than raising a ConfigurationError telling the user to
        coarsen a value they never pinned."""
        env = Environment(
            name='deep-slope',
            bathymetry=[(0.0, 50.0), (10000.0, 5000.0)], ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.7,
                                      attenuation=0.5))
        src = Source(depths=25.0, frequencies=500.0)
        rcv = Receiver(depths=np.array([100.0]),
                       ranges=np.array([5000.0, 10000.0]))
        m = RAM(verbose=False, backend='ramgeo', Q=50.0, T=0.02)
        with pytest.warns(UserWarning, match="fit the binary's depth arrays"):
            field = m.run(env, src, rcv, run_mode=RunMode.BROADBAND)
        needed, mz, _ = m._collins_mz_budget('ramgeo', field.metadata['zmax'])
        assert needed(field.metadata['dz']) <= mz

    def test_broadband_pinned_dz_past_mz_is_still_rejected(self):
        """Coarsening applies only to an auto grid; a pinned ``dz`` the
        binary cannot hold is still an error the caller must resolve."""
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.models.base import RunMode
        src = Source(depths=50.0, frequencies=50.0)
        rcv = Receiver(depths=100.0, ranges=np.array([5000.0]))
        with pytest.raises(ConfigurationError, match="mz=20002"):
            RAM(verbose=False, backend='ramgeo', dz=0.005,
                Q=50.0, T=0.2).run(self._fluid_env(), src, rcv,
                                   run_mode=RunMode.BROADBAND)

    def test_broadband_path_also_checks_the_limits(self):
        """The narrowband entry point is not the only way in — the broadband
        sweep calls the per-frequency runner directly."""
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.models.base import RunMode
        src = Source(depths=50.0, frequencies=np.array([40.0, 50.0, 60.0]))
        rcv = Receiver(depths=100.0, ranges=np.array([5000.0]))
        with pytest.raises(ConfigurationError, match="mr=505"):
            RAM(verbose=False, backend='rams').run(
                self._env(600), src, rcv, run_mode=RunMode.BROADBAND)

    def test_mz_consumption_rate_matches_the_fortran_indexing(self):
        """rams0.5 interleaves the elastic field vector, so it indexes 2*nz+4
        where the fluid codes index nz+2. Getting this factor wrong understates
        rams' depth capacity by 2x.

        Anchored on the *upstream* field-array initialisation loop
        (``rams0.5.f:154``, ``ramsurf1.5.f:140``, ``ramgeo1.5.f:155``) as well
        as the ``.gt.mz`` guard, because rams0.5's guard is itself a uacpy
        addition (third_party/MODIFICATIONS.md) and cannot vouch for the rate
        on its own."""
        import re
        from pathlib import Path
        from uacpy.models.ram import _COLLINS_ARRAY_LIMITS
        root = Path(__file__).resolve().parents[1] / 'third_party'
        srcs = {'rams': root / 'ramsurf' / 'rams0.5.f',
                'ramsurf': root / 'ramsurf' / 'ramsurf1.5.f',
                'ramgeo': root / 'ramgeo' / 'ramgeo1.5.f'}
        for kind, path in srcs.items():
            if not path.exists():
                pytest.skip(f"{path.name} not vendored here")
            lim = _COLLINS_ARRAY_LIMITS[kind]
            expected = (f"{lim['nz_factor']}*nz+{lim['nz_pad']}"
                        if lim['nz_factor'] != 1 else f"nz+{lim['nz_pad']}")
            text = path.read_text(errors='ignore')
            guard = re.search(r'if\(([^)]*nz[^)]*)\.gt\.mz\)', text)
            assert guard, f"no mz bounds check in {path.name}"
            assert guard.group(1).replace(' ', '') == expected, (
                f"{path.name} guards {guard.group(1)!r}, table says {expected!r}")
            init = re.search(r'do +\d+ +i=1,(\d?\*?nz\+\d)\n +u\(i\)', text)
            assert init, f"no field-array init loop in {path.name}"
            assert init.group(1).replace(' ', '') == expected, (
                f"{path.name} initialises u to {init.group(1)!r}, "
                f"table says {expected!r}")

    def test_limit_table_matches_the_vendored_sources(self):
        """The table must track the actual ``parameter (mr=…)`` declarations."""
        import re
        from pathlib import Path
        from uacpy.models.ram import _COLLINS_ARRAY_LIMITS
        root = Path(__file__).resolve().parents[1] / 'third_party'
        srcs = {'rams': root / 'ramsurf' / 'rams0.5.f',
                'ramsurf': root / 'ramsurf' / 'ramsurf1.5.f',
                'ramgeo': root / 'ramgeo' / 'ramgeo1.5.f'}
        for kind, path in srcs.items():
            if not path.exists():
                pytest.skip(f"{path.name} not vendored here")
            m = re.search(r'parameter\s*\(mr=(\d+),mz=(\d+)',
                          path.read_text(errors='ignore'))
            assert m, f"could not find array dims in {path.name}"
            assert _COLLINS_ARRAY_LIMITS[kind]['mr'] == int(m.group(1))
            assert _COLLINS_ARRAY_LIMITS[kind]['mz'] == int(m.group(2))


def test_forcing_rams_on_a_fluid_bottom_is_rejected():
    """rams0.5 is the elastic PE; on a fluid bottom it returns a null field —
    TL saturated at the ``TL_MAX_DB`` sentinel at every range, which looks like
    an answer rather than a failure.

    ``_validate_forced_backend`` therefore rejects each Collins backend when
    its defining feature is absent: ramsurf without altimetry, rams without
    shear. Auto-dispatch never routes fluid to rams, so only ``backend='rams'``
    can reach the null field.
    """
    from uacpy.core.exceptions import ConfigurationError
    fluid = Environment(
        name='f', bathymetry=200.0, ssp=1500.0,
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1800.0, density=1.8,
                                  attenuation=0.5))
    src = Source(depths=50.0, frequencies=100.0)
    rcv = Receiver(depths=100.0, ranges=np.array([1000.0]))

    assert RAM(verbose=False).select_backend(fluid) != 'rams'
    with pytest.raises(ConfigurationError, match="elastic PE"):
        RAM(verbose=False, backend='rams').run(fluid, src, rcv)

    # With shear present, rams is both auto-selected and functional.
    elastic = Environment(
        name='e', bathymetry=200.0, ssp=1500.0,
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1800.0, density=1.8,
                                  attenuation=0.5, shear_speed=300.0))
    assert RAM(verbose=False).select_backend(elastic) == 'rams'
    tl = np.asarray(RAM(verbose=False, backend='rams').run(
        elastic, src, rcv).tl).ravel()
    assert np.all(np.isfinite(tl)) and np.all(tl < 150.0), tl


class TestCollinsArrayBoundaries:
    """The Fortran reads N pairs *plus* the ``-1 -1`` terminator into index
    N+1 before testing ``i.gt.mr`` (ramgeo1.5.f:146, ramsurf1.5.f:131), and the
    writer may append one more point to reach the last receiver — so the guard
    must bound what is actually written, not ``env.bathymetry.n_ranges``."""

    @staticmethod
    def _bathy_env(n):
        from uacpy.core.environment import Bathymetry
        r = np.linspace(0.0, 20000.0, n)
        d = 200.0 + 10.0 * np.sin(r / 3000.0)
        return Environment(
            name='b', bathymetry=Bathymetry(ranges=r, depths=d), ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.5))

    _SRC = staticmethod(lambda: Source(depths=50.0, frequencies=50.0))
    _RCV = staticmethod(lambda: Receiver(depths=100.0, ranges=np.array([5000.0])))

    def test_bathymetry_at_the_boundary_raises_not_truncated_read(self):
        """504 written points + terminator == mr fits; 505 does not, and
        must be refused here rather than surface downstream as a truncated
        ``tl.grid``."""
        from uacpy.core.exceptions import ConfigurationError
        m = RAM(verbose=False, backend='ramgeo', timeout=600)
        tl = np.asarray(m.run(self._bathy_env(504), self._SRC(), self._RCV()).tl)
        assert np.all(np.isfinite(tl))
        with pytest.raises(ConfigurationError, match="mr=505"):
            RAM(verbose=False, backend='ramgeo', timeout=600).run(
                self._bathy_env(505), self._SRC(), self._RCV())

    def test_ramsurf_altimetry_count_is_guarded(self):
        """ramsurf1.5.f:82-88 reads the surface into rsrf(mr)/zsrf(mr) with no
        bounds check of its own, so an over-long altimetry silently returns a
        ~430 dB null field (or segfaults) instead of raising."""
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.core.environment import Altimetry
        r = np.linspace(0.0, 20000.0, 600)
        env = Environment(
            name='a', bathymetry=200.0, ssp=1500.0,
            altimetry=Altimetry(ranges=r, heights=np.full(r.size, -2.0)),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.5))
        with pytest.raises(ConfigurationError, match="mr=505"):
            RAM(verbose=False, backend='ramsurf', timeout=600).run(
                env, self._SRC(), self._RCV())


@pytest.mark.requires_binary
class TestUpslopeSubBottomIsNotStale:
    """mpiramS rebuilds its sub-bottom arrays when the seafloor index moves.

    ``profl`` fills them at absolute depth indices and ``matrc`` reads them at
    the current ``iz``, so deferring the rebuild leaves a band of seabed
    carrying water sound speed below a *rising* seafloor. Downslope is immune
    (the stale band lands above ``iz``), so the two slopes agreeing is the
    signature of a correct sub-bottom.
    """

    @staticmethod
    def _wedge(depths):
        from uacpy.core.environment import Bathymetry
        r = np.linspace(0.0, 6000.0, 61)
        return Environment(
            name='wedge', ssp=1500.0,
            bathymetry=Bathymetry(ranges=r, depths=depths),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.7,
                                      attenuation=0.5))

    def _disagreement(self, depths):
        env = self._wedge(depths)
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=[30.0, 70.0],
                       ranges=np.linspace(500.0, 5000.0, 19))
        kw = dict(dr=10.0, dz=0.5, zmax=500.0, timeout=600, verbose=False)
        a = np.asarray(RAM(backend='mpiramS', **kw).run(env, src, rcv).tl)
        b = np.asarray(RAM(backend='ramgeo', **kw).run(env, src, rcv).tl)
        return np.abs(a - b)

    def test_upslope_agrees_with_ramgeo(self):
        up = self._disagreement(np.linspace(200.0, 100.0, 61))
        assert np.nanmedian(up) < 1.5, (
            f"mpiramS vs ramgeo median {np.nanmedian(up):.2f} dB on an upslope "
            f"wedge — the sub-bottom is stale below the rising seafloor")

    def test_upslope_and_downslope_are_symmetric(self):
        up = np.nanmedian(self._disagreement(np.linspace(200.0, 100.0, 61)))
        down = np.nanmedian(self._disagreement(np.linspace(100.0, 200.0, 61)))
        assert abs(up - down) < 1.0, (
            f"upslope {up:.2f} dB vs downslope {down:.2f} dB — a one-sided "
            f"error is the sub-bottom staleness signature")


class TestBackendIndependentResultShape:
    """``Field.kind`` must not depend on which backend auto-dispatch picked."""

    _FLUID = BoundaryProperties(acoustic_type='half-space', sound_speed=1700.0,
                                density=1.7, attenuation=0.5)

    def _run(self, **env_kw):
        env = _env(bottom=self._FLUID, **env_kw)
        src = Source(depths=25.0, frequencies=50.0)
        rcv = Receiver(depths=np.arange(10.0, 90.0, 10.0),
                       ranges=np.arange(500.0, 5000.0, 500.0))
        return RAM(verbose=False).run(env, src, rcv,
                                      run_mode=RunMode.COHERENT_TL)

    def test_mpirams_and_collins_agree_on_kind_and_phase_reference(self):
        mpirams = self._run()
        collins = self._run(altimetry=[(0.0, 0.0), (10000.0, 0.0)])
        assert mpirams.backend == 'mpiramS' and collins.backend == 'ramsurf'
        for f in (mpirams, collins):
            assert f.kind == 'pressure'
            assert np.iscomplexobj(f.data)
            assert f.phase_reference == 'travelling_wave'

    def test_the_two_backends_agree_on_level(self):
        """A flat altimetry only changes the binary, not the physics."""
        a, b = np.asarray(self._run().tl), np.asarray(
            self._run(altimetry=[(0.0, 0.0), (10000.0, 0.0)]).tl)
        ok = np.isfinite(a) & np.isfinite(b)
        assert np.nanmedian(np.abs(a[ok] - b[ok])) < 1.0

    def test_collins_tl_still_reproduces_the_binary_grid(self):
        """Deriving TL from the complex envelope must not shift the level.

        Interpolating the complex field instead of its modulus biases the
        median by ~4 dB, because opposite-phase lobes cancel across an
        interference null. Resampling the modulus leaves a residual well
        under 0.1 dB — linear interpolation of |ψ| and of log|ψ| are not the
        same operation — which is negligible against the 1–4.5 dB tolerances
        the cross-model benchmarks use."""
        env = _env(bottom=self._FLUID,
                   altimetry=[(0.0, 0.0), (10000.0, 0.0)])
        src = Source(depths=25.0, frequencies=50.0)
        rcv = Receiver(depths=np.arange(1.0, 99.0, 1.0),
                       ranges=np.arange(100.0, 10000.0, 50.0))
        m = RAM(verbose=False)
        field = m.run(env, src, rcv, run_mode=RunMode.COHERENT_TL)
        raw = m._run_collins_one_freq(env, src, rcv, kind='ramsurf', freq=50.0,
                                      theta=m._theta_for_freq(50.0))
        from uacpy.models.ram import _interp_to_receiver_grid
        native = _interp_to_receiver_grid(
            raw['depths'], raw['ranges'], np.asarray(raw['tl'], float),
            rcv.depths.astype(float), rcv.ranges.astype(float))
        got = np.asarray(field.tl)
        # Compare where the binary's own TL is unclamped; the near-source
        # samples the divergence clamp rewrites are not a level comparison.
        ok = (np.isfinite(native) & np.isfinite(got)
              & (native > 0.0) & (native < TL_MAX_DB))
        assert abs(float(np.median(got[ok] - native[ok]))) < 0.15


@pytest.mark.requires_binary
class TestCollinsBroadbandLevel:
    """The BROADBAND sweep resamples the same envelope as COHERENT_TL.

    The Collins output grid is ``dr·ndr``-spaced, sized for Padé accuracy
    and unrelated to the envelope's range Nyquist. With the default
    ``c0`` (Lytaev Eq. 15, ~1687 m/s against a 1500 m/s water column) ψ
    turns >90° between adjacent output samples, so interpolating the
    complex field averages across opposite-phase lobes and raises the
    median TL by ~1.5-2.3 dB on this reference.
    """

    _ENV = dict(
        bathymetry=[(0.0, 100.0), (10000.0, 100.0)],
        ssp=[(0.0, 1500.0), (100.0, 1500.0)],
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1700.0, density=1.7,
                                  attenuation=0.5),
    )

    def _setup(self):
        env = Environment(name='pekeris', **self._ENV)
        src = Source(depths=36.0, frequencies=250.0)
        rcv = Receiver(depths=np.array([36.0]),
                       ranges=np.linspace(200.0, 8000.0, 50))
        return env, src, rcv

    def test_broadband_fc_slice_matches_the_binary_grid(self):
        env, src, rcv = self._setup()
        m = RAM(backend='ramgeo', Q=8.0, T=0.05, verbose=False)
        bb = m.run(env, src, rcv, run_mode=RunMode.BROADBAND)
        got = np.asarray(bb.at(frequency=250.0).to_tl().data, dtype=float)

        # Re-run the binary on the sweep's own numerics and dB-interpolate
        # its tl.grid — the reference the COHERENT_TL test uses.
        rmax = float(np.max(rcv.ranges))
        fc, Q, T = m._resolve_broadband_grid(src)
        bw, df = fc / Q, 1.0 / T
        nf1 = max(1, int((bw - df) / df) + 1)
        freqs = np.array([(i - nf1) * df + fc for i in range(2 * nf1 + 1)])
        freqs = freqs[freqs > 0.0]
        dr_b, _ = m._compute_grid_lytaev(env, float(freqs[0]),
                                         max_range=rmax, kind='ramgeo')
        _, dz_b = m._compute_grid_lytaev(env, float(freqs[-1]),
                                         max_range=rmax, kind='ramgeo')
        raw = m._run_collins_one_freq(
            env, src, rcv, kind='ramgeo', freq=250.0,
            theta=m._theta_for_freq(250.0), dr_override=dr_b,
            dz_override=dz_b,
            zmax_override=m._compute_zmax(env, float(freqs[0])))

        from uacpy.models.ram import _interp_to_receiver_grid
        native = _interp_to_receiver_grid(
            raw['depths'], raw['ranges'], np.asarray(raw['tl'], float),
            rcv.depths.astype(float), rcv.ranges.astype(float))
        ok = (np.isfinite(native) & np.isfinite(got)
              & (native > 0.0) & (native < TL_MAX_DB))
        bias = float(np.median(got[ok] - native[ok]))
        assert abs(bias) < 0.15, (
            f"BROADBAND fc slice is {bias:+.3f} dB off the binary's own "
            f"tl.grid — the envelope is being resampled as a complex field")

    def test_broadband_fc_slice_matches_the_narrowband_run(self):
        env, src, rcv = self._setup()
        bb = RAM(backend='ramgeo', Q=8.0, T=0.05, verbose=False).run(
            env, src, rcv, run_mode=RunMode.BROADBAND)
        nb = RAM(backend='ramgeo', verbose=False).run(
            env, src, rcv, run_mode=RunMode.COHERENT_TL)
        got = np.asarray(bb.at(frequency=250.0).to_tl().data, dtype=float)
        ref = np.asarray(nb.tl, dtype=float)
        ok = np.isfinite(got) & np.isfinite(ref)
        bias = float(np.median(got[ok] - ref[ok]))
        assert abs(bias) < 0.25, (
            f"BROADBAND fc slice sits {bias:+.3f} dB off the COHERENT_TL "
            f"run of the same environment")

    def test_metadata_c_min_is_the_environment_minimum(self):
        """``c_min`` documents the slowest speed the solver brackets, not
        the Padé reference ``c0`` — which on this env is 1687 m/s against a
        1500 m/s water column."""
        env, src, rcv = self._setup()
        m = RAM(backend='ramgeo', Q=8.0, T=0.05, verbose=False)
        field = m.run(env, src, rcv, run_mode=RunMode.BROADBAND)
        assert field.metadata['c_min'] == pytest.approx(1500.0)
        assert field.metadata['c0'] != pytest.approx(1500.0)


def test_run_frequencies_preserves_every_source_field():
    """``run(frequencies=…)`` rebuilds the Source; it must not silently drop
    ``source_type`` / ``beam_pattern`` and bypass their validation."""
    env = _env(bottom=BoundaryProperties(
        acoustic_type='half-space', sound_speed=1700.0, density=1.7,
        attenuation=0.5))
    src = Source(depths=25.0, frequencies=50.0, source_type='line')
    rcv = Receiver(depths=np.array([30.0]), ranges=np.array([1000.0]))
    m = RAM(verbose=False)
    # Inspect the rebuilt Source directly via the public path instead of
    # patching: a line source is rejected downstream, so the guard firing at
    # all proves source_type survived the rebuild.
    with pytest.raises((ConfigurationError, UnsupportedFeatureError)):
        m.run(env, src, rcv, run_mode=RunMode.COHERENT_TL,
              frequencies=np.array([40.0, 50.0, 60.0]))


@pytest.mark.requires_binary
class TestStaleOutputsAreCleared:
    """The Collins and mpiramS binaries write fixed filenames with no
    run-specific stem, so a pinned ``work_dir`` holding a previous run's
    ``tl.grid`` / ``pcomplex.bin`` / ``psif.dat`` would be read back as this
    run's answer. Each launch clears them first."""

    def _setup(self, tmp_path, **kw):
        env = _env(bottom=_fluid_bottom(), **kw)
        src = Source(depths=25.0, frequencies=50.0)
        rcv = Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))
        return env, src, rcv

    def test_collins_stale_grid_is_not_returned(self, tmp_path, monkeypatch):
        env, src, rcv = self._setup(tmp_path)
        for name in ('tl.grid', 'pcomplex.bin'):
            (tmp_path / name).write_bytes(b'\x00' * 4096)
        m = RAM(backend='ramgeo', verbose=False, work_dir=str(tmp_path),
                cleanup=False)
        monkeypatch.setattr(type(m), '_run_subprocess',
                            lambda self, *a, **k: _FakeProc())
        with pytest.raises(ModelExecutionError, match='tl.grid'):
            m.run(env, src, rcv, run_mode=RunMode.COHERENT_TL)
        assert not (tmp_path / 'tl.grid').exists()
        assert not (tmp_path / 'pcomplex.bin').exists()

    def test_mpirams_stale_psif_is_not_returned(self, tmp_path, monkeypatch):
        env, src, rcv = self._setup(tmp_path)
        (tmp_path / 'psif.dat').write_bytes(b'\x00' * 4096)
        m = RAM(backend='mpiramS', verbose=False, work_dir=str(tmp_path),
                cleanup=False)
        monkeypatch.setattr(type(m), '_run_subprocess',
                            lambda self, *a, **k: _FakeProc())
        with pytest.raises(ModelExecutionError, match='psif.dat'):
            m.run(env, src, rcv, run_mode=RunMode.COHERENT_TL)
        assert not (tmp_path / 'psif.dat').exists()


@pytest.mark.requires_binary
def test_warnings_are_attributed_to_the_callers_frame():
    """Ten frames separate ``_compute_grid_lytaev`` from user code, so a
    hand-counted ``stacklevel`` cannot span it. ``skip_file_prefixes`` skips
    all of ``uacpy/models/`` instead."""
    import warnings as _warnings
    env = _env(bottom=_fluid_bottom())
    src = Source(depths=25.0, frequencies=50.0)
    rcv = Receiver(depths=np.array([50.0, 500.0]), ranges=np.array([1000.0]))
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter('always')
        RAM(backend='ramgeo', accuracy=1e-6, verbose=False).run(
            env, src, rcv, run_mode=RunMode.COHERENT_TL)
    ram_warnings = [w for w in caught if str(w.message).startswith('RAM')]
    assert ram_warnings, 'expected at least one RAM warning'
    for w in ram_warnings:
        assert w.filename == __file__, (
            f"{str(w.message)[:60]!r} was attributed to {w.filename}, "
            f"not the caller's frame")


# ─── Regression: Collins output grid must cover the receiver grid ──────────


@pytest.mark.requires_binary
class TestCollinsOutputGridCoversReceivers:
    """The Collins binaries write only at ``r = k·dr·ndr`` and stop at the
    first ``r >= rmax``, so a march handed ``rmax = max_range`` verbatim ends
    up to ``(ndr-1)·dr`` short and the outermost receiver column reads NaN.
    ``_collins_output_stride`` extends the marched rmax instead."""

    def _run(self, backend, bottom, *, rmax, n_ranges, **kw):
        env = _env(bottom=bottom)
        src = Source(depths=50.0, frequencies=250.0)
        rcv = Receiver(depths=np.array([10.0, 50.0, 90.0]),
                       ranges=np.linspace(100.0, rmax, n_ranges))
        return RAM(backend=backend, verbose=False, **kw).run(
            env, src, rcv, run_mode=RunMode.COHERENT_TL)

    @staticmethod
    def _all_nan_columns(field):
        d = field.data
        return [j for j in range(d.shape[1])
                if np.all(~np.isfinite(d[:, j]))]

    def test_ramgeo_coarse_dr_keeps_the_last_range(self):
        # ndr = floor((10000/3)/1000) = 3, so a march to rmax = 10000 m
        # verbatim writes its last record at 9999 m and leaves the receiver
        # at 10000 m outside the output grid.
        field = self._run('ramgeo', _fluid_bottom(), rmax=10000.0,
                          n_ranges=100, dr=3.0)
        assert self._all_nan_columns(field) == []

    def test_rams_default_grid_keeps_the_last_range(self):
        field = self._run('rams', _elastic_bottom(), rmax=5000.0, n_ranges=50)
        assert self._all_nan_columns(field) == []

    def test_stride_math_reaches_max_range(self):
        # Every written range is k·dr·ndr; the march stops at the first
        # r >= rmax_march, which must be a multiple of dr·ndr past max_range.
        for dr, max_range in ((3.0, 10000.0), (1.2, 5000.0), (0.7, 12345.0)):
            ndr, rmax_march = RAM._collins_output_stride(
                dr, max_range, np.linspace(100.0, max_range, 40))
            n_steps = int(np.ceil(rmax_march / dr - 1e-9))
            assert n_steps % ndr == 0, (dr, max_range, ndr)
            assert n_steps * dr >= max_range - 1e-9

    def test_stride_caps_output_record_count(self):
        # A near-field receiver would otherwise force ndr=1 over a long run.
        ndr, _ = RAM._collins_output_stride(
            0.5, 200_000.0, np.array([1.0, 200_000.0]))
        assert 200_000.0 / (0.5 * ndr) <= 20_000 + 1


# ─── Regression: the shallowest output depth ──────────────────────────────


@pytest.mark.requires_binary
class TestSurfaceDepthIsResolvable:
    """``rams0.5``'s ``outpt`` loops ``do 1 i=1+ndz,nzplt,ndz``, so its
    shallowest stored sample sits at ``ndz·dz`` and a receiver at z=0 lies
    outside the interpolation grid. The surface is pressure-release in every
    Collins backend, so the z=0 node is prepended at the deep-shadow floor."""

    def test_rams_receiver_at_the_surface_is_not_nan(self):
        env = _env(bottom=_elastic_bottom())
        src = Source(depths=50.0, frequencies=250.0)
        rcv = Receiver(depths=np.array([0.0, 1.0, 50.0]),
                       ranges=np.linspace(100.0, 5000.0, 50))
        field = RAM(backend='rams', verbose=False).run(
            env, src, rcv, run_mode=RunMode.COHERENT_TL)
        d = field.data
        assert [i for i in range(d.shape[0])
                if np.all(~np.isfinite(d[i, :]))] == []
        # z=0 carries no energy: the pressure-release boundary value.
        assert np.allclose(field.tl[0, :], TL_MAX_DB)
        assert np.all(field.tl[1, :] < TL_MAX_DB)

    def test_prepended_node_is_skipped_when_the_grid_starts_at_zero(self):
        depths = np.array([0.0, 1.0, 2.0])
        tl = np.zeros((3, 4))
        pc = np.zeros((3, 4), dtype=complex)
        out_d, out_tl, out_pc = RAM._prepend_surface_node(depths, tl, pc)
        assert out_d is not None and out_d.size == 3
        assert out_tl.shape == (3, 4) and out_pc.shape == (3, 4)


# ─── Regression: phase across the decimated output grid ───────────────────


@pytest.mark.requires_binary
class TestEnvelopePhaseSurvivesResampling:
    """``rams0.5`` bakes the carrier into u (``solve`` multiplies by
    ``g0`` each step), and every Collins backend leaves ``k_r - k0`` in the
    envelope. Linear interpolation of the unit phasor is only faithful while
    the rotation between source samples is under π, so
    ``_interp_envelope_to_receiver_grid`` divides the carrier out first."""

    def _raw(self, backend, bottom, freq=250.0):
        env = _env(bottom=bottom)
        src = Source(depths=50.0, frequencies=freq)
        rcv = Receiver(depths=np.array([10.0, 50.0, 90.0]),
                       ranges=np.linspace(200.0, 3000.0, 40))
        model = RAM(backend=backend, verbose=False)
        theta = model._theta_for_freq(freq)
        raw = model._run_collins_one_freq(
            env, src, rcv, kind=backend, freq=freq, theta=theta)
        rate = model._collins_carrier_rate(env, backend, freq, theta)
        return raw, rate

    @staticmethod
    def _phase_error(raw, rate, decimation):
        from uacpy.models.ram import _interp_envelope_to_receiver_grid
        d, r, psi = raw['depths'], raw['ranges'], raw['pcomplex']
        src_r = r[decimation - 1::decimation]
        src_psi = psi[:, decimation - 1::decimation]
        keep = (r >= src_r[0]) & (r <= src_r[-1])
        truth = psi[:, keep]
        out = _interp_envelope_to_receiver_grid(
            d, src_r, src_psi, d, r[keep], carrier_rate=rate)
        # Deep nulls carry no meaningful phase; weight them out.
        good = (np.isfinite(out) & np.isfinite(truth)
                & (np.abs(truth) > 0.05 * np.nanmedian(np.abs(truth))))
        return np.degrees(np.abs(np.angle(out[good] / truth[good])))

    def test_rams_decimated_by_eight_keeps_phase(self):
        raw, rate = self._raw('rams', _elastic_bottom())
        assert rate > 0.5, 'rams bakes the whole carrier into its envelope'
        no_fix = self._phase_error(raw, 0.0, 8)
        fixed = self._phase_error(raw, rate, 8)
        assert np.median(no_fix) > 45.0
        assert np.median(fixed) < 10.0

    def test_ramgeo_carrier_rate_is_the_c0_offset(self):
        raw, rate = self._raw('ramgeo', _fluid_bottom())
        no_fix = self._phase_error(raw, 0.0, 8)
        fixed = self._phase_error(raw, rate, 8)
        assert 0.0 < rate < 0.5
        assert np.median(fixed) < np.median(no_fix) / 2.0

    def test_carrier_round_trip_is_exact_on_the_source_grid(self):
        from uacpy.models.ram import _interp_envelope_to_receiver_grid
        rng = np.random.default_rng(0)
        d = np.array([0.0, 1.0, 2.0])
        r = np.linspace(10.0, 100.0, 19)
        psi = rng.normal(size=(3, 19)) + 1j * rng.normal(size=(3, 19))
        out = _interp_envelope_to_receiver_grid(
            d, r, psi, d, r, carrier_rate=1.0472)
        assert np.allclose(out, psi, atol=1e-10)


# ─── Regression: mpiramS output depth grid is not uniform ─────────────────


@pytest.mark.requires_binary
class TestBroadbandDepthGridIsNonUniform:
    """``flat_earth=True`` (the default) makes peramx un-transform its output
    depth axis with ``zg/(1 + eps/2 + eps²/3)`` (peramx.f90:427-432) — a
    quadratic map, so ``_run_broadband`` cannot bracket receiver depths with
    the uniform ``(z - zg[0]) / (zg[1] - zg[0])``."""

    def _deep_env(self):
        return Environment(
            name='deep', bathymetry=3000.0,
            ssp=[(0.0, 1520.0), (1000.0, 1490.0), (3000.0, 1520.0)],
            bottom=_fluid_bottom(),
        )

    def test_broadband_depths_land_on_the_real_grid(self):
        env = self._deep_env()
        src = Source(depths=100.0, frequencies=50.0)
        rcv = Receiver(depths=np.array([200.0, 1500.0, 2900.0]),
                       ranges=np.linspace(1000.0, 8000.0, 8))
        model = RAM(backend='mpiramS', verbose=False, dr=50.0, dz=2.0,
                    Q=50.0, T=1.0)
        field = model.run(env, src, rcv, run_mode=RunMode.BROADBAND)
        assert field.data.shape[0] == 3
        assert np.all(np.isfinite(field.data))

    def test_uniform_assumption_would_misplace_deep_receivers(self):
        # Prove the grid really is non-uniform, so the interpolation change
        # is not a no-op: rebuild peramx's un-transform.
        #
        # 6371 km is the mean Earth radius; mpiramS itself uses the WGS-84
        # equatorial 6378137 m (``mpiramS/src/param.f90:10``, ``invRe=1/Re``).
        # The 0.1% difference shifts eps far below the millimetre-scale
        # departures asserted here, so it does not affect the demonstration —
        # but this is a stand-in for the un-transform, not a copy of it.
        zg = np.arange(0, 5000.0, 2.0)
        eps = zg / 6371000.0
        zg1 = zg / (1.0 + eps / 2.0 + eps ** 2 / 3.0)
        spacing = np.diff(zg1)
        assert spacing.max() - spacing.min() > 1e-4
        uniform = zg1[0] + np.arange(zg1.size) * spacing[0]
        assert np.max(np.abs(zg1 - uniform)) > 1.0     # metres


# ─── Regression: the artificial absorbing layer on the Collins path ───────


@pytest.mark.requires_binary
class TestCollinsAbsorbingLayer:
    """Collins' readme (``third_party/ramsurf/readme.orig:127-134``) requires
    the attenuation to be ramped up over the deepest part of the PE domain,
    not carried flat from the half-space value down to zmax."""

    def _attn(self, bottom, **kw):
        env = _env(bottom=bottom)
        model = RAM(backend='ramgeo', verbose=False, **kw)
        zmax = kw.get('zmax') or model._compute_zmax(env, 250.0)
        segs = model._collins_range_segments(
            env, 'ramgeo', zmax=zmax, freq=250.0)
        return segs[0]['bottom_attn'], zmax

    def test_attenuation_ramps_to_the_absorbing_value(self):
        attn, zmax = self._attn(_fluid_bottom())
        assert attn[-1][0] == pytest.approx(zmax - 100.0)   # seafloor-relative
        assert attn[-1][1] == pytest.approx(10.0)           # default attn
        assert attn[0][1] == pytest.approx(0.5)             # seabed's own value

    def test_ramp_starts_below_the_modelled_sediment(self):
        attn, _ = self._attn(_fluid_layered_bottom(), zmax=400.0,
                             absorbing_layer_width=5.0)
        depths = [d for d, _ in attn]
        values = [v for _, v in attn]
        # The 15 m layer keeps its own attenuation.
        assert values[0] == pytest.approx(0.4)
        assert max(d for d, v in attn if v <= 0.4) >= 15.0
        # 5 wavelengths of ramp at the domain floor.
        assert depths[-1] == pytest.approx(300.0)
        assert values[-1] == pytest.approx(10.0)

    def test_absorbing_knobs_are_not_reported_as_ignored(self):
        """The Collins path honours both absorbing-layer knobs, so listing
        them as mpiramS-only would warn the caller off a setting that works."""
        import warnings as _warnings
        names = [n for n, _ in RAM._MPIRAMS_ONLY_SETTINGS]
        assert 'absorbing_layer_width' not in names
        assert 'absorbing_layer_attn' not in names
        model = RAM(backend='ramgeo', verbose=False,
                    absorbing_layer_width=8.0, absorbing_layer_attn=15.0)
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter('always')
            model._warn_on_mpirams_only_overrides('ramgeo')
        assert not any('absorbing_layer' in str(w.message) for w in caught)


# ─── Regression: sediment speed referenced to the LOCAL seafloor ──────────


def _bottom_props(model, env, work_dir, freq=100.0):
    """``_prepare_bottom_properties`` with the absorber geometry the deck
    writer derives, so tests see the same profile mpiramS is given."""
    dz = model._effective_dz()
    zmax = model._mpirams_zmax(env, freq, dz)
    span = model._absorber_span(env, freq, zmax)
    return model._prepare_bottom_properties(env, work_dir, span, zmax)


class TestSedimentSpeedFollowsTheLocalSeafloor:
    """mpiramS rebuilds the sediment speed as ``csg = cwg + cs``
    (``mpiramS/src/ram.f90:332-333``) against the water column at the *local*
    seafloor, so the offset uacpy writes must be referenced to the same
    column."""

    def _wedge(self, bottom=None):
        return Environment(
            name='wedge',
            bathymetry=[(0.0, 200.0), (10000.0, 100.0)],
            ssp=[(0.0, 1520.0), (100.0, 1500.0), (200.0, 1480.0)],
            bottom=bottom or _fluid_bottom(),       # cb = 1600 m/s
        )

    @staticmethod
    def _written_csg(model, env, tmp_path, freq=100.0):
        """``(range, z_ctrl, csg)`` per written sediment profile.

        ``csg = cwg + cs`` (``ram.f90:332-333``) is what mpiramS rebuilds, so
        it is the quantity the deck has to get right — not the raw offset.
        """
        dz = model._effective_dz()
        zmax = model._mpirams_zmax(env, freq, dz)
        sedlayer, nzs, _cs, _rho, _attn, isedrd, sed_name = _bottom_props(
            model, env, tmp_path, freq)
        assert isedrd == 1 and sed_name
        # Profile blocks are 4 lines: header, cs, rho, attn; the header range
        # is in km.
        rows = (tmp_path / sed_name).read_text().splitlines()
        out = []
        for i in range(len(rows) // 4):
            rng = 1000.0 * float(rows[4 * i].split()[1])
            cs_row = np.fromstring(rows[4 * i + 1], sep=' ')
            seafloor = float(np.asarray(env.bathymetry.eval(range=rng)).flat[0])
            z = model._control_point_depths(seafloor, sedlayer, nzs, zmax)
            out.append((rng, z, model._ssp_column(env, rng, z) + cs_row))
        return out

    @pytest.mark.requires_binary
    def test_halfspace_offset_reproduces_cb_at_every_range(self, tmp_path):
        env = self._wedge()
        model = RAM(backend='mpiramS', verbose=False)
        ranges, _cwg = model._varying_seafloor_speeds(env)
        # The seafloor slides continuously between the two declared breaks and
        # mpiramS picks the nearest profile, so the axis samples the slope.
        assert ranges is not None and len(ranges) > 2
        for _rng, _z, csg in self._written_csg(model, env, tmp_path):
            assert csg[1:] == pytest.approx(1600.0, abs=1e-6)

    @pytest.mark.requires_binary
    def test_flat_environment_still_writes_one_profile(self, tmp_path):
        env = _env(bottom=_fluid_bottom())
        model = RAM(backend='mpiramS', verbose=False)
        assert model._varying_seafloor_speeds(env) == (None, None)
        _, _, cs, _, _, isedrd, sed_name = _bottom_props(
            model, env, tmp_path)
        assert isedrd == 0 and sed_name == ''
        assert cs[2] == pytest.approx(1600.0 - 1500.0)

    @pytest.mark.requires_binary
    def test_layered_bottom_uses_the_local_reference_too(self, tmp_path):
        # 15 m @1650, hs @1900
        env = self._wedge(bottom=_fluid_layered_bottom())
        model = RAM(backend='mpiramS', verbose=False)
        for rng, z, csg in self._written_csg(model, env, tmp_path):
            seafloor = float(np.asarray(env.bathymetry.eval(range=rng)).flat[0])
            in_layer = (z > seafloor) & (z < seafloor + 15.0)
            assert csg[in_layer] == pytest.approx(1650.0, abs=1e-6)
            # The last interior control point already carries the half-space.
            assert csg[-2] == pytest.approx(1900.0, abs=1e-6)


# ─── Regression: the tl.line receiver depth is bounds-checked ─────────────


@pytest.mark.requires_binary
def test_tl_line_receiver_depth_stays_inside_the_pe_arrays(tmp_path):
    """``ramsurf1.5.f:427`` reads ``u(ir)`` / ``f3(ir+1)`` with no bounds
    check, so an unclamped ``zr_line`` past the PE domain reads out of
    bounds. The writer must receive a depth inside ``nz = zmax/dz - 0.5``."""
    env = _env(bottom=_fluid_bottom())
    src = Source(depths=50.0, frequencies=100.0)
    rcv = Receiver(depths=np.array([50.0, 5000.0]),
                   ranges=np.linspace(500.0, 3000.0, 10))
    model = RAM(backend='ramgeo', verbose=False, dr=20.0, dz=2.0,
                work_dir=str(tmp_path), cleanup=False)
    with pytest.warns(UserWarning, match='exceed the PE domain'):
        model.run(env, src, rcv, run_mode=RunMode.COHERENT_TL)
    zmax, dz = (float(x) for x in
                (tmp_path / 'ramgeo.in').read_text().splitlines()[3].split()[:2])
    zr_line = float((tmp_path / 'ramgeo.in').read_text().splitlines()[1].split()[2])
    nz = int(zmax / dz - 0.5)
    assert 1 + zr_line / dz + 1 <= nz + 2


@pytest.mark.requires_binary
class TestRamStampsCMax:
    """Every RAM result carries ``c_max``, the fastest compressional speed the
    PE domain meshes through.

    ``Field.to_time_trace`` anchors the output window on ``r / c_max`` — the
    earliest arrival. Without the key it falls back to an estimate and warns,
    so a result that omits it degrades the time series silently.
    """

    #: Water 1500, sediment 1650, basement 1900 — the basement is fastest and
    #: the RAM domain meshes through it, so it sets the earliest arrival.
    EXPECTED = 1900.0

    def _rig(self):
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=np.array([25.0, 75.0]),
                       ranges=np.linspace(500.0, 5000.0, 10))
        return _env(bottom=_fluid_layered_bottom()), src, rcv

    @pytest.mark.parametrize('backend', ['mpiramS', 'ramgeo'])
    def test_narrowband_carries_c_max(self, backend):
        env, src, rcv = self._rig()
        result = RAM(backend=backend, verbose=False, dr=20.0, dz=2.0).run(
            env, src, rcv, run_mode=RunMode.COHERENT_TL)
        assert result.metadata['c_max'] == pytest.approx(self.EXPECTED)

    @pytest.mark.parametrize('backend', ['mpiramS', 'ramgeo'])
    def test_broadband_carries_c_max(self, backend):
        env, src, rcv = self._rig()
        src = Source(depths=50.0, frequencies=np.linspace(80.0, 120.0, 5))
        result = RAM(backend=backend, verbose=False, dr=20.0, dz=2.0).run(
            env, src, rcv, run_mode=RunMode.BROADBAND)
        assert result.metadata['c_max'] == pytest.approx(self.EXPECTED)

    def test_c_max_includes_the_seabed_not_just_the_water(self):
        """A water-only c_max would anchor the window behind the
        bottom-refracted first arrival."""
        env, _, _ = self._rig()
        ram = RAM(verbose=False)
        assert ram._resolve_c_max(env) == pytest.approx(self.EXPECTED)
        assert float(np.max(env.ssp.data)) < self.EXPECTED

    def test_time_series_does_not_warn_about_an_estimated_window(self):
        """The whole point of the key: a long-range trace anchors exactly."""
        env, _, _ = self._rig()
        fs, n = 2000.0, 256
        t = np.arange(n) / fs
        waveform = np.sin(2 * np.pi * 100.0 * t) * np.hanning(n)
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=np.array([75.0]), ranges=np.array([40000.0]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = RAM(verbose=False, dr=50.0, dz=2.0).run(
                env, src, rcv, run_mode=RunMode.TIME_SERIES,
                frequencies=np.linspace(60.0, 140.0, 17),
                source_waveform=waveform, sample_rate=fs,
            )
        assert result.metadata['c_max'] == pytest.approx(self.EXPECTED)
        assert not [w for w in caught if 'wrap to the end' in str(w.message)]


@pytest.mark.requires_binary
class TestRamAppliesTheStabilityFloor:
    """The Lytaev search optimises accuracy alone; RAM then bounds the
    resulting Δz to what its backend needs, and says so.

    A cost floor ``λ_p/16`` (``LAMBDA_PER_DZ_FLOOR``) stops the optimizer
    demanding an absurdly fine depth grid. For ``rams`` the shear wavelength
    is the binding physical scale, so :func:`rams_dz_shear_cap` caps Δz at
    ``λ_s/14`` and also bounds how far the cost floor may coarsen it. The
    bounds live here rather than in ``optimize_grid`` so one function owns
    the grid actually marched.
    """

    def _elastic_env(self, shear_speed):
        return _env(bottom=SeabedColumn(
            layers=[SedimentLayer(thickness=20, sound_speed=1700,
                                  density=1.5, attenuation=0.5,
                                  shear_speed=shear_speed,
                                  shear_attenuation=1.0)],
            halfspace=BoundaryProperties(
                acoustic_type='half-space', sound_speed=1900, density=2.0,
                attenuation=0.1, shear_speed=shear_speed,
                shear_attenuation=0.5),
        ))

    def test_the_shear_wavelength_caps_dz(self):
        """A fast shear seabed at low frequency makes λ_s/8 the binding bound,
        so the returned grid must resolve it.

        Collins (1991) JASA 89(3) 1050-1057 resolves λ_s at 14-85 points per
        wavelength in all four of its worked examples; a grid coarser than
        λ_s/8 does not merely lose accuracy, the rams0.5 march diverges.
        """
        from uacpy.models._pade_optimizer import rams_dz_shear_cap
        env = self._elastic_env(shear_speed=1200.0)
        freq = 30.0
        cap = rams_dz_shear_cap(1200.0, freq)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dr, dz = RAM(backend='rams', verbose=False)._compute_grid_lytaev(
                env, freq=freq, max_range=5000.0, kind='rams')
        assert dz <= cap * (1.0 + 1e-9), f"dz={dz} coarser than the cap {cap}"
        assert dr > 0

    def test_missing_the_budget_warns_only_when_the_caller_pinned_one(self):
        """The floor sits above the default Lytaev Δz at ordinary
        frequencies, so warning unconditionally would fire on nearly every
        run. It is a status line by default and a UserWarning only when the
        caller asked for an accuracy that is then not delivered."""
        env = self._elastic_env(shear_speed=1200.0)
        kw = dict(env=env, freq=30.0, max_range=5000.0, kind='rams')

        def grid_warnings(model):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model._compute_grid_lytaev(**kw)
            return [w for w in caught
                    if 'shear-mode stability' in str(w.message)]

        assert not grid_warnings(RAM(backend='rams', verbose=False))
        assert grid_warnings(
            RAM(backend='rams', verbose=False, accuracy=1e-3))

    def test_a_fluid_env_gets_the_acoustic_floor_not_the_shear_one(self):
        """``rams_dz_shear_cap`` returns 0 without shear, so the acoustic
        λ_p/16 bound is what applies."""
        from uacpy.models.ram import LAMBDA_PER_DZ_FLOOR
        env = _env(bottom=_fluid_bottom())
        freq = 100.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _dr, dz, *_ = RAM(verbose=False)._compute_grid_lytaev(
                env, freq=freq, max_range=5000.0, kind='mpiramS')
        assert dz >= 1500.0 / (LAMBDA_PER_DZ_FLOOR * freq) * (1.0 - 1e-9)

    def test_an_explicit_dz_is_not_overridden_by_the_floor(self):
        """The floor is a default-path guard; a pinned grid is the user's."""
        env = self._elastic_env(shear_speed=1200.0)
        model = RAM(backend='rams', verbose=False, dr=20.0, dz=0.25)
        assert model.dz == 0.25


# ─── Regression: the Collins output grid reaches the deepest receiver ─────


class TestCollinsPlotDepthReachesTheDeepestReceiver:
    """The Collins codes truncate ``nzplt = zmplt/dz - 0.5`` into an integer
    (``ramgeo1.5.f:131``, ``ramsurf1.5.f:112``, ``rams0.5.f:133``) and store
    output up to grid index ``nzplt``, whose depth is ``(nzplt-1)·dz``
    (``ri = 1 + zr/dz``, ``ramgeo1.5.f:126-127``). ``zmplt`` must therefore
    carry more than one ``dz`` of headroom over the deepest requested depth,
    plus whatever ``depth_decimation`` strides past."""

    @staticmethod
    def _fortran_deepest(zmplt, dz, ndz, kind):
        """Literal transcription of the vendored output loops: ``nzplt`` from
        the truncating integer assignment, then ``do i = start, nzplt, ndz``
        with ``start = ndz`` (fluid) or ``1+ndz`` (rams0.5), index ``i`` at
        depth ``(i-1)·dz``."""
        nzplt = int(zmplt / dz - 0.5)
        start = 1 + ndz if kind == 'rams' else ndz
        visited = range(start, nzplt + 1, ndz)
        return (visited[-1] - 1) * dz if len(visited) else -1.0

    @pytest.mark.parametrize('kind', ['ramgeo', 'ramsurf', 'rams'])
    @pytest.mark.parametrize('ndz', [1, 2, 5])
    @pytest.mark.parametrize('dz', [0.25, 0.5, 1.0, 2.2222222])
    @pytest.mark.parametrize('target', [100.0, 97.5, 63.7])
    def test_the_binary_formula_stores_a_sample_at_or_below_the_target(
            self, dz, target, ndz, kind):
        zmplt = RAM._collins_zmplt(target, dz, 400.0, ndz, kind)
        assert zmplt <= 400.0
        deepest = RAM._collins_deepest_output(zmplt, dz, ndz, kind)
        assert deepest == pytest.approx(
            self._fortran_deepest(zmplt, dz, ndz, kind))
        assert deepest >= target

    @pytest.mark.requires_binary
    @pytest.mark.parametrize('kind', ['ramgeo', 'ramsurf', 'rams'])
    def test_the_generated_deck_reaches_the_seafloor(self, kind, tmp_path):
        """End-to-end: a receiver on the seafloor must come back finite on
        every Collins backend, with no warning."""
        env = _env(
            bottom=(_elastic_bottom() if kind == 'rams'
                    else _fluid_layered_bottom()),
            altimetry=([(0.0, 0.0), (5000.0, 0.0)] if kind == 'ramsurf'
                       else None),
        )
        src = Source(depths=20.0, frequencies=100.0)
        rcv = Receiver(depths=np.array([50.0, 100.0]), ranges=np.array([3000.0]))
        model = RAM(backend=kind, verbose=False, dr=10.0, dz=0.25, zmax=400.0,
                    work_dir=str(tmp_path), cleanup=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            field = model.run(env, src, rcv, run_mode=RunMode.COHERENT_TL)
        assert np.all(np.isfinite(np.asarray(field.data)))
        assert not [w for w in caught if 'exceed the PE domain' in str(w.message)]

        deck = (tmp_path / {'rams': 'rams.in',
                            'ramgeo': 'ramgeo.in'}.get(kind, 'ram.in'))
        _zmax, dz, ndz, zmplt = (
            float(x) for x in deck.read_text().splitlines()[3].split()[:4])
        assert RAM._collins_deepest_output(zmplt, dz, int(ndz), kind) >= 100.0

    @pytest.mark.requires_binary
    def test_decimated_output_still_reaches_the_seafloor(self, tmp_path):
        """``depth_decimation`` strides the same loop, so the deepest visited
        index — not ``nzplt`` — is what has to reach the receiver."""
        env = _env(bottom=_fluid_layered_bottom())
        src = Source(depths=20.0, frequencies=100.0)
        rcv = Receiver(depths=np.array([100.0]), ranges=np.array([3000.0]))
        model = RAM(backend='ramgeo', verbose=False, dr=10.0, dz=0.25,
                    zmax=400.0, depth_decimation=5, work_dir=str(tmp_path),
                    cleanup=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            field = model.run(env, src, rcv, run_mode=RunMode.COHERENT_TL)
        assert np.all(np.isfinite(np.asarray(field.data)))


# ─── Regression: the mpiramS sub-bottom keeps its layer/half-space step ───


@pytest.mark.requires_binary
class TestMpiramsLayeredSubBottomIsNotSmeared:
    """``profl`` puts the sediment control points at ``zwork = [0, d,
    d + k·sedlayer/(nzs-3), max(zg(n), …)]`` and interpolates them linearly
    (``mpiramS/src/ram.f90:321-337``, ``gorp`` at ``:360-390``). Only one control
    point lies between ``d + sedlayer`` and the domain floor, so whatever
    contrast sits between the last two points is spread over the entire
    sub-bottom. The half-space must therefore already be reached at
    ``d + sedlayer`` — control point ``nzs-1``, not ``nzs``."""

    def _layered_env(self):
        return Environment(
            name='layered', bathymetry=100.0, ssp=1500.0,
            bottom=SeabedColumn(
                layers=[SedimentLayer(thickness=5.0, sound_speed=1520.0,
                                      density=1.3, attenuation=0.2)],
                halfspace=BoundaryProperties(
                    acoustic_type='half-space', sound_speed=2800.0,
                    density=2.5, attenuation=0.1)))

    def test_the_deck_puts_the_halfspace_at_the_last_interior_point(
            self, tmp_path):
        model = RAM(backend='mpiramS', verbose=False)
        sedlayer, nzs, cs, rho, attn, isedrd, _sed = \
            _bottom_props(model, self._layered_env(), tmp_path)
        assert isedrd == 0
        cwg = 1500.0
        # Fortran index nzs-1 is at d + sedlayer, index nzs at the domain
        # floor; both must already carry the half-space.
        assert cs[nzs - 2] + cwg == pytest.approx(2800.0)
        assert cs[nzs - 1] + cwg == pytest.approx(2800.0)
        assert rho[nzs - 2] == pytest.approx(2.5)
        assert rho[nzs - 1] == pytest.approx(2.5)
        # Fortran index nzs-2 is one dz_sed above it and still in the layer.
        assert cs[nzs - 3] + cwg == pytest.approx(1520.0)
        assert sedlayer == pytest.approx(5.0)

    @staticmethod
    def _cross_backend_median(env, tmp_path, *, zmax, tag, dz=0.25):
        src = Source(depths=20.0, frequencies=100.0)
        rcv = Receiver(depths=np.arange(2.0, 99.0, 4.0),
                       ranges=np.arange(1000.0, 10001.0, 500.0))

        def tl(backend):
            model = RAM(backend=backend, verbose=False, dr=10.0, dz=dz,
                        zmax=zmax,
                        work_dir=str(tmp_path / f'{tag}_{backend}'))
            data = np.abs(np.asarray(
                model.run(env, src, rcv, run_mode=RunMode.COHERENT_TL).data
            ).squeeze())
            with np.errstate(divide='ignore'):
                return -20.0 * np.log10(data)

        diff = np.abs(tl('mpiramS') - tl('ramgeo'))
        return float(np.median(diff[np.isfinite(diff)]))

    @pytest.mark.slow
    def test_mpirams_and_ramgeo_converge_as_the_grid_refines(self, tmp_path):
        """Cross-backend agreement must IMPROVE as ``dz`` refines.

        The RAM manual prescribes settling these questions by convergence test
        — "accuracy may be controlled by performing simple convergence tests to
        determine ... the location of the lower boundary, and the thickness of
        the absorbing layer" — so the physics to assert is the trend, not a
        level. What remains at a fixed ``dz`` is honest discretisation of a
        5 m, high-contrast layer: the two codes place its step differently
        (``gorp`` interpolates the control points, while ``zread``'s
        ``if(i.eq.iold)i=i+1`` — ``ramgeo1.5.f:223`` — bumps the discontinuity
        to the next node when two breakpoints land on one grid index), and
        that difference shrinks with the cell size rather than persisting.
        """
        env = self._layered_env()
        coarse = self._cross_backend_median(env, tmp_path, zmax=700.0,
                                            tag='coarse', dz=0.5)
        fine = self._cross_backend_median(env, tmp_path, zmax=700.0,
                                          tag='fine', dz=0.125)
        assert fine < coarse, (
            f"refining dz from 0.5 m to 0.125 m moved the mpiramS/ramgeo "
            f"median from {coarse:.3f} dB to {fine:.3f} dB; a discretisation "
            f"difference must shrink with the cell size")


# ─── Regression: the mpiramS absorbing layer is the width that was asked ──


class TestMpiramsAbsorbingLayerHasTheRequestedWidth:
    """``profl`` interpolates the sediment arrays linearly between control
    point ``nzs-1`` at ``seafloor + sedlayer`` and control point ``nzs`` at
    ``zmax`` (``mpiramS/src/ram.f90:321-329,332-337``), and only the last point
    carries ``absorbing_layer_attn``. The absorbing layer is therefore the span
    ``[seafloor + sedlayer, zmax]``, so ``sedlayer`` is what sets its width.

    Collins sizes that layer at "the lower **few wavelengths** of the grid"
    (RAM manual), which is what ``absorbing_layer_width`` requests. Running the
    ramp from just below the seabed instead replaces the seabed's own
    attenuation with an artificial gradient across the whole sub-bottom.
    """

    @staticmethod
    def _halfspace_env(cb=1600.0, rho=1.8, att=0.5):
        return Environment(
            name='hs', bathymetry=100.0, ssp=1500.0,
            bottom=SeabedColumn(layers=[], halfspace=BoundaryProperties(
                acoustic_type='half-space', sound_speed=cb, density=rho,
                attenuation=att)))

    @staticmethod
    def _geometry(model, env, zmax, freq=100.0):
        """``(absorber_width, requested_width)`` for the deck as written."""
        sedlayer, *_ = model._prepare_bottom_properties(
            env, Path('.'), model._absorber_span(env, freq, zmax), zmax)
        requested = (model.absorbing_layer_width * model._resolve_c0(env)
                     / freq)
        return zmax - float(env.depth) - sedlayer, requested

    @pytest.mark.requires_binary
    @pytest.mark.parametrize('zmax', [700.0, 1200.0, 2000.0])
    def test_a_pinned_zmax_does_not_turn_the_sub_bottom_into_sponge(self, zmax):
        env = self._halfspace_env()
        model = RAM(backend='mpiramS', verbose=False, dz=0.25, zmax=zmax)
        width, requested = self._geometry(model, env, zmax)
        assert width == pytest.approx(requested, rel=1e-6)

    @pytest.mark.requires_binary
    def test_absorbing_layer_width_actually_moves_the_absorber(self):
        env = self._halfspace_env()
        widths = []
        for w in (10.0, 20.0, 40.0):
            model = RAM(backend='mpiramS', verbose=False, dz=0.25, zmax=2000.0,
                        absorbing_layer_width=w)
            width, requested = self._geometry(model, env, 2000.0)
            assert width == pytest.approx(requested, rel=1e-6)
            widths.append(width)
        assert widths[0] < widths[1] < widths[2]

    @pytest.mark.requires_binary
    def test_the_absorber_never_eats_into_the_modelled_sediment(self):
        """A stack thicker than the room below it keeps its own attenuation;
        the layer is squeezed instead, exactly as
        ``_ramp_absorbing_attenuation`` does for the Collins backends."""
        env = Environment(
            name='thick', bathymetry=100.0, ssp=1500.0,
            bottom=SeabedColumn(
                layers=[SedimentLayer(thickness=180.0, sound_speed=1700.0,
                                      density=1.8, attenuation=0.3)],
                halfspace=BoundaryProperties(
                    acoustic_type='half-space', sound_speed=2000.0,
                    density=2.0, attenuation=0.5)))
        model = RAM(backend='mpiramS', verbose=False, dz=0.25, zmax=400.0)
        sedlayer, *_ = model._prepare_bottom_properties(
            env, Path('.'), model._absorber_span(env, 100.0, 400.0), 400.0)
        assert sedlayer >= 180.0

    @pytest.mark.slow
    def test_the_two_backends_agree_once_the_absorber_matches(self, tmp_path):
        """The deck geometry asserted above, seen as physics. A half-space is
        a medium both PEs represent exactly, so once their absorbing layers
        span the same depths there is nothing left to separate them — any
        residual here is the absorber, not the seabed."""
        env = self._halfspace_env()
        src = Source(depths=20.0, frequencies=100.0)
        rcv = Receiver(depths=np.arange(2.0, 99.0, 4.0),
                       ranges=np.arange(1000.0, 10001.0, 500.0))

        def tl(backend):
            model = RAM(backend=backend, verbose=False, dr=10.0, dz=0.25,
                        zmax=700.0,
                        work_dir=str(tmp_path / f'agree_{backend}'))
            data = np.abs(np.asarray(
                model.run(env, src, rcv, run_mode=RunMode.COHERENT_TL).data
            ).squeeze())
            with np.errstate(divide='ignore'):
                return -20.0 * np.log10(data)

        diff = np.abs(tl('mpiramS') - tl('ramgeo'))
        median = float(np.median(diff[np.isfinite(diff)]))
        assert median < 0.5, f"mpiramS/ramgeo median {median:.3f} dB"


# ─── Regression: one profile-transition rule for every backend ───────────


class TestRangeSegmentMarkersAreMidpoints:
    """``if(r.ge.rp)`` (``ramgeo1.5.f:359``, ``ramsurf1.5.f:364``,
    ``rams0.5.f:332``) switches to a section at its marker range, while
    mpiramS marches with the nearest profile (``minloc(abs(rp-rint))``,
    ``mpiramS/src/ram.f90:218,228``) — the rule ``Bottom.at`` declares. The
    marker is the midpoint so the two agree."""

    def _env(self, breaks):
        columns = [
            SeabedColumn(layers=[], halfspace=BoundaryProperties(
                acoustic_type='half-space', sound_speed=c,
                density=1.5, attenuation=0.5))
            for c in (1700.0, 1550.0, 1450.0)[:len(breaks)]
        ]
        from uacpy.core.bottom import Bottom
        return Environment(
            name='rd', bathymetry=100.0, ssp=1500.0,
            bottom=Bottom(columns=columns, ranges=list(breaks)))

    @pytest.mark.requires_binary  # constructs RAM (resolves its binary)
    def test_markers_sit_halfway_between_consecutive_breakpoints(self):
        segs = RAM(verbose=False)._collins_range_segments(
            self._env([0.0, 2500.0, 6000.0]), 'ramgeo', zmax=300.0, freq=150.0)
        assert [s['range'] for s in segs] == [0.0, 1250.0, 4250.0]

    @pytest.mark.requires_binary
    def test_the_written_deck_carries_the_midpoint(self, tmp_path):
        src = Source(depths=25.0, frequencies=150.0)
        rcv = Receiver(depths=np.array([50.0]),
                       ranges=np.array([1000.0, 5000.0]))
        model = RAM(backend='ramgeo', verbose=False, dr=20.0, dz=0.5,
                    zmax=300.0, work_dir=str(tmp_path), cleanup=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.run(self._env([0.0, 2500.0]), src, rcv,
                      run_mode=RunMode.COHERENT_TL)
        rows = (tmp_path / 'ramgeo.in').read_text().splitlines()
        markers = [r.strip() for r in rows if len(r.split()) == 1
                   and r.strip() not in ('-1 -1',)]
        assert '1250.000000' in markers


# ─── Regression: the broadband band must stay strictly positive ──────────


class TestBroadbandBandStaysPositive:
    """``peramx.f90:345-362`` builds ``frq(1) = fc - nf1/T`` with no
    positivity guard in the serial driver uacpy builds; its MPI sibling stops
    on exactly this test and names ``Q``
    (``mpiramS/src/peramx_mpi.f90:417-423``)."""

    def test_the_vector_matches_the_fortran_construction(self):
        frq = RAM._broadband_frequencies(100.0, 4.0, 0.2)
        assert frq == pytest.approx([75.0, 80.0, 85.0, 90.0, 95.0, 100.0,
                                     105.0, 110.0, 115.0, 120.0, 125.0])

    def test_a_band_edge_at_or_below_zero_is_rejected(self):
        with pytest.raises(ConfigurationError, match='lower band edge'):
            RAM._broadband_frequencies(100.0, 0.5, 0.2)
        with pytest.raises(ConfigurationError, match='lower band edge'):
            RAM._broadband_frequencies(50.0, 50.0, 0.02)   # frq(1) == 0

    @pytest.mark.requires_binary
    @pytest.mark.parametrize('backend', ['mpiramS', 'ramgeo'])
    def test_both_backends_reject_the_same_configuration(self, backend):
        env = _env(bottom=_fluid_bottom())
        src = Source(depths=25.0, frequencies=100.0)
        rcv = Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))
        with pytest.raises(ConfigurationError, match='lower band edge'):
            RAM(backend=backend, verbose=False, Q=0.5, T=0.2).run(
                env, src, rcv, run_mode=RunMode.BROADBAND)


# ─── Regression: the mpiramS depth grid really is deltaz-spaced ──────────


@pytest.mark.requires_binary
@pytest.mark.parametrize('flat_earth', [True, False])
def test_mpirams_zmax_is_an_exact_multiple_of_deltaz(tmp_path, flat_earth):
    """``peramx.f90:374,387`` builds the depth grid as
    ``linspace(0, zmax, floor(zmax/deltaz - 0.5) + 2)``, whose spacing is
    ``zmax/(icount-1)``, while the depth operator (``ram.f90:51``, consumed at
    ``matrc.f90:60-62``) and the seafloor index (``ram.f90:101``) use
    ``deltaz``. They coincide only for a ``zmax`` that is a multiple of
    ``deltaz``.

    The depth that matters is the one ``peramx.f90:371`` reads
    (``zmax = maxval(zw)``), which under the default ``flat_earth`` is the
    written column *after* ``peramx.f90:264-266`` rescales it — so the check
    has to apply that map, not re-use the written value.
    """
    env = _env(bottom=_fluid_bottom())
    src = Source(depths=20.0, frequencies=100.0)
    rcv = Receiver(depths=np.array([50.0]), ranges=np.array([2000.0]))
    model = RAM(backend='mpiramS', verbose=False, work_dir=str(tmp_path),
                cleanup=False, flat_earth=flat_earth)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.run(env, src, rcv, run_mode=RunMode.COHERENT_TL)

    deltaz = float((tmp_path / 'in.pe').read_text().splitlines()[4])
    written = np.loadtxt(tmp_path / 'ssp.dat', skiprows=1)[:, 0].max()
    # eps = z/Re; z' = z(1 + eps/2 + eps^2/3), Re from param.f90:10.
    if flat_earth:
        eps = written / 6378137.0
        zmax = written * (1.0 + eps / 2.0 + eps * eps / 3.0)
        assert zmax != pytest.approx(written, rel=1e-12), (
            "the transform is a no-op here, so this parametrisation proves "
            "nothing")
    else:
        zmax = written
    icount = int(np.floor(zmax / deltaz - 0.5)) + 2
    assert zmax / (icount - 1) == pytest.approx(deltaz, rel=1e-12)


# ─── Regression: theta_max carries the seabed-slope term ─────────────────


@pytest.mark.requires_binary  # constructs RAM (resolves its binary)
class TestThetaMaxIncludesTheBottomSlope:
    """Lytaev (2023) §5.1 estimates the spectrum bracket as
    ``θ_max = max(θ_max^src, θ_max^bottom)``, the second being the steepest
    slope between bottom and water. ``theta_max`` on the constructor is the
    source term only; the seabed term comes from ``env.bathymetry``."""

    def _sloped(self, run_m, drop_m):
        return Environment(
            name='slope', ssp=1500.0, bottom=_fluid_bottom(),
            bathymetry=[(0.0, 100.0), (float(run_m), 100.0 + float(drop_m))])

    def test_a_flat_seabed_leaves_the_constructor_value_alone(self):
        model = RAM(verbose=False)
        assert model._resolve_theta_max(_env(bottom=_fluid_bottom())) == 30.0
        assert model._resolve_theta_max(self._sloped(10000.0, 100.0)) == 30.0

    def test_a_steep_slope_widens_the_bracket(self):
        model = RAM(verbose=False)
        env = self._sloped(200.0, 400.0)         # atan(400/200) = 63.43 deg
        assert model._resolve_theta_max(env) == pytest.approx(63.4349, abs=1e-3)

    def test_the_wider_bracket_reaches_c0_and_the_grid(self):
        model = RAM(verbose=False)
        flat = _env(bottom=_fluid_bottom())
        steep = self._sloped(200.0, 400.0)
        assert model._resolve_c0(steep) > model._resolve_c0(flat)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dr_flat, _ = model._compute_grid_lytaev(
                flat, 100.0, max_range=5000.0, kind='mpiramS')
            dr_steep, _ = model._compute_grid_lytaev(
                steep, 100.0, max_range=5000.0, kind='mpiramS')
        assert dr_steep < dr_flat


# ─── Regression: the sub-bottom does not inherit the water gradient ──────


@pytest.mark.requires_binary
class TestSubBottomIsIndependentOfSspTabulationDepth:
    """``profl`` rebuilds the sediment speed as ``csg = cwg + cs``
    (``mpiramS/src/ram.f90:332-333``) at every depth, so a water gradient
    tabulated below the seabed would be added into the sediment unless the
    offset cancels it there. ``_sediment_offsets`` takes ``cs`` per control
    point, which cancels *any* column — so the written SSP stays true and the
    sub-bottom still depends on the seabed alone."""

    def _env(self, ssp):
        return Environment(name='grad', bathymetry=100.0, ssp=ssp,
                           bottom=_fluid_bottom())          # cb = 1600 m/s

    def test_the_written_column_keeps_the_true_ssp(self, tmp_path):
        """Holding it flat below the seabed would corrupt real water on a
        slope: mpiramS picks this column by nearest neighbour
        (``ram.f90:295-296``) while interpolating the seafloor continuously
        (``ram.f90:315-317``)."""
        env = self._env([(0.0, 1450.0), (400.0, 1550.0)])
        RAM(backend='mpiramS', verbose=False)._prepare_ssp(
            env, tmp_path, 100.0, 0.25)
        ssp = np.loadtxt(tmp_path / 'ssp.dat', skiprows=1)
        below = ssp[ssp[:, 0] > 100.0]
        assert np.ptp(below[:, 1]) > 50.0, "column was flattened below the seabed"
        assert below[:, 1] == pytest.approx(
            np.interp(below[:, 0], [0.0, 400.0], [1450.0, 1550.0]), abs=0.2)

    def test_the_offset_cancels_the_water_column_in_the_sub_bottom(self,
                                                                  tmp_path):
        """``cwg + cs`` must be the requested ``cb`` at every control point,
        which is what makes the sub-bottom independent of the column."""
        env = self._env([(0.0, 1450.0), (400.0, 1550.0)])
        model = RAM(backend='mpiramS', verbose=False, dz=0.25, zmax=400.0)
        zmax = model._mpirams_zmax(env, 100.0, 0.25)
        sedlayer, nzs, cs, _rho, _attn, isedrd, _sed = \
            model._prepare_bottom_properties(
                env, tmp_path, model._absorber_span(env, 100.0, zmax), zmax)
        assert isedrd == 0
        z_ctrl = model._control_point_depths(100.0, sedlayer, nzs, zmax)
        csg = model._ssp_column(env, 0.0, z_ctrl) + cs
        assert csg[1:] == pytest.approx(1600.0, abs=1e-6)

    def test_tl_is_the_same_however_deep_the_ssp_is_tabulated(self, tmp_path):
        src = Source(depths=20.0, frequencies=100.0)
        rcv = Receiver(depths=np.arange(2.0, 99.0, 4.0),
                       ranges=np.arange(1000.0, 10001.0, 1000.0))

        def tl(ssp, tag):
            model = RAM(backend='mpiramS', verbose=False, dr=10.0, dz=0.25,
                        zmax=400.0, work_dir=str(tmp_path / tag))
            data = np.abs(np.asarray(
                model.run(self._env(ssp), src, rcv,
                          run_mode=RunMode.COHERENT_TL).data).squeeze())
            with np.errstate(divide='ignore'):
                return -20.0 * np.log10(data)

        deep = tl([(0.0, 1450.0), (400.0, 1550.0)], 'deep')
        cut = tl([(0.0, 1450.0), (100.0, 1475.0)], 'cut')
        # Not bit-identical: uacpy cancels the column with the linear
        # interpolation it writes, while ``gorp2`` (``ram.f90:296``) splines it
        # back onto the grid, so the two tabulations leave a spline-vs-linear
        # residual of order 1e-4 dB rather than exact agreement.
        assert np.nanmax(np.abs(deep - cut)) < 1e-2


# ─── Regression: the marched step is the requested one ───────────────────


@pytest.mark.requires_binary
class TestMpiramsMarchStepIsRestored:
    """``ram.f90`` shrinks ``dr`` to ``rend-rnow`` to land exactly on an output
    range; upstream does not restore it, so every later output range marches at
    that leftover step. uacpy writes *every* receiver range into ``ranges.dat``,
    which makes the marched step — and hence the answer and the cost — a
    function of how many receivers were asked for. The uacpy patch restores the
    full step (``third_party/mpiramS/src/ram.f90``; see ``MODIFICATIONS.md``),
    and this pins that the far-range answer is receiver-count independent."""

    @staticmethod
    def _env():
        return Environment(
            name='pekeris', bathymetry=100.0, ssp=1500.0,
            bottom=SeabedColumn(layers=[], halfspace=BoundaryProperties(
                acoustic_type='half-space', sound_speed=1600.0, density=1.8,
                attenuation=0.5)))

    def _tl_at_far_range(self, n_ranges, tmp_path):
        ranges = (np.array([10000.0]) if n_ranges == 1
                  else np.linspace(10000.0 / n_ranges, 10000.0, n_ranges))
        model = RAM(backend='mpiramS', verbose=False,
                    work_dir=str(tmp_path / f'n{n_ranges}'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            data = np.abs(np.asarray(model.run(
                self._env(), Source(depths=20.0, frequencies=250.0),
                Receiver(depths=np.array([50.0]), ranges=ranges),
                run_mode=RunMode.COHERENT_TL).data).squeeze())
        return float(-20.0 * np.log10(np.atleast_1d(data)[-1]))

    @pytest.mark.slow
    def test_tl_does_not_depend_on_how_many_ranges_were_requested(self,
                                                                  tmp_path):
        """Asking for more receivers must not change the march, because the
        step it marches with is what ``Result`` reports as ``dr``."""
        alone = self._tl_at_far_range(1, tmp_path)
        crowded = self._tl_at_far_range(50, tmp_path)
        assert crowded == pytest.approx(alone, abs=1e-4), (
            f"TL at 10 km moved from {alone:.6f} dB to {crowded:.6f} dB when "
            f"49 nearer receivers were added")


@pytest.mark.requires_binary
class TestMpiramsAutoGridIsAccurateEnough:
    """OPEN DEFECT: mpiramS's auto ``dr`` is far too coarse on a shallow
    waveguide, and Lytaev's error model does not see it.

    For the Pekeris case below (100 m water, cb=1700 m/s, 50 Hz, 8 km)
    ``optimize_grid`` returns ``dr = 108.1 m`` — 3.6 wavelengths — whose
    window-median TL sits 5.75 dB from Kraken at z=20 m, r=6.5-7.1 km. Pinning
    ``dr`` converges: 50 m → 5.31 dB, 20 m → 0.57 dB, and 10 m / 5 m reproduce
    20 m exactly. The error model disagrees with all of that — it predicts
    9.37e-4 at ``dr=108`` against 9.05e-4 at ``dr=10``, i.e. essentially no
    ``dr`` sensitivity — so the grid is chosen on a term that does not govern
    the accuracy here.

    Unlike ``rams`` (which gets both ``rams_dr_safety_factor`` and a
    ``dr <= c_min/(5f)`` wavelength cap) mpiramS applies no cap to the
    optimiser's ``dr``. A blanket wavelength cap is not obviously right
    either: upstream's own ``ram.in`` marches 50 Hz at ``dr = 500 m``, so
    large steps are legitimate for the problems RAM was built for.

    The patched march step (see :class:`TestMpiramsMarchStepIsRestored`) is
    what makes this visible: with the remainder step left unrestored the run
    silently refines its own grid past the optimiser's ``dr``, so the coarse
    choice never reaches the physics.
    """

    @pytest.mark.slow
    @pytest.mark.xfail(strict=True, reason=(
        "mpiramS auto dr = 108 m (3.6 lambda) on a 100 m Pekeris waveguide at "
        "50 Hz puts the window-median TL 5.75 dB from Kraken; pinning dr=20 m "
        "gives 0.57 dB and refining further changes nothing. Lytaev's error "
        "model predicts near-identical error for dr=108 and dr=10, so the cap "
        "has to come from elsewhere. Fixing it makes this XPASS."))
    def test_the_auto_grid_matches_kraken_as_well_as_a_pinned_one(self):
        env = _env(bottom=BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1700.0, density=1.7, attenuation=0.5,
        ))
        ref = TestRamPekerisReference()
        src, rcv = ref._src_rcv()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            auto = RAM(verbose=False).run(
                env, src, rcv, run_mode=RunMode.COHERENT_TL)
        ref._assert_window_agreement(
            auto, ref._kraken_reference(env, src, rcv), tol_db=4.5,
            label='mpiramS auto grid')


@pytest.mark.requires_oases
@pytest.mark.benchmark
class TestRamsElasticGridAgreesWithWavenumberIntegration:
    """Collins (1991) JASA 89(3) 1050-1057 example D, arbitrated against OASES.

    The case is range-independent, so OAST (wavenumber integration) is
    essentially exact for an elastic half-space — an independent model rather
    than another PE backend, which is the only way to say which grid is right.
    Collins states his own grid for this example: ``Δr = 5 m and Δz = 0.5 m``,
    i.e. λ_s/64. A rams0.5 march on a grid coarser than λ_s/8 diverges.
    """

    @staticmethod
    def _example_d():
        import uacpy
        from uacpy.core.environment import SoundSpeedProfile
        env = uacpy.Environment(
            bathymetry=200.0,
            ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (200.0, 1500.0)]),
            bottom=BoundaryProperties(
                acoustic_type='half-space', sound_speed=1700.0,
                shear_speed=800.0, density=1.5, attenuation=0.5,
                shear_attenuation=0.5))
        return (env, Source(depths=100.0, frequencies=25.0),
                Receiver(depths=[30.0],
                         ranges=np.array([1000.0, 2000.0, 4000.0, 8000.0])))

    def test_the_auto_grid_tracks_the_exact_reference(self):
        from uacpy.models import OAST
        from uacpy.models import RAM as _RAM
        env, src, rcv = self._example_d()
        assert _RAM().select_backend(env) == 'rams'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ref = np.asarray(OAST().compute_tl(env, src, rcv).tl).ravel()
            tl = np.asarray(_RAM().compute_tl(env, src, rcv).tl).ravel()
        # Padé-PE against exact wavenumber integration on a high-contrast
        # elastic half-space; a diverged march reads 100+ dB out here.
        assert np.max(np.abs(tl - ref)) < 8.0
        assert not (tl == 200.0).any(), "no sample may sit on the clamp"

    def test_the_grid_resolves_the_shear_wavelength(self):
        from uacpy.models._pade_optimizer import rams_dz_shear_cap
        from uacpy.models import RAM as _RAM
        env, src, _rcv = self._example_d()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _dr, dz = _RAM(backend='rams', verbose=False)._compute_grid_lytaev(
                env, freq=25.0, max_range=8000.0, kind='rams')
        assert dz <= rams_dz_shear_cap(800.0, 25.0) * (1.0 + 1e-9)
