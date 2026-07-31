"""Tests for the RAM multi-backend dispatcher and the Collins-style I/O."""


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
from uacpy.core.exceptions import ConfigurationError, UnsupportedFeatureError


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
    """ramgeo/ramsurf read bottom-profile depths as depth-below-seafloor
    (``matrc`` restarts the profile index at the grid point under the local
    seafloor; the vendored ``ramgeo.in`` confirms it), while rams0.5 indexes
    absolutely from z=0. The segment builder must emit each convention."""

    def _segments(self, kind):
        env = _env(bottom=_fluid_layered_bottom())     # 100 m water, 15 m layer
        return RAM(verbose=False)._collins_range_segments(env, kind, zmax=400.0)

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
        # 'simple bottom should also be accepted' — ramgeo runs on a plain
        # half-space when forced (wrapped as a synthetic single layer).
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
        # Post depth-reference fix the two PEs agree to ~1 dB median; the
        # tight bound guards the seafloor-relative profile convention.
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
    """Per-backend pinned-TL agreement against KrakenField on a Pekeris
    waveguide.

    Cross-model RMSE tests average over a window and can hide localized
    errors; this asserts agreement at specific (range, depth) sample
    points for each of the three RAM dispatcher backends. KrakenField is
    the reference (mode sum on Pekeris is essentially the analytical
    solution).
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
        env = _env(bottom=BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1700.0, density=1.7, attenuation=0.5,
        ))
        src, rcv = self._src_rcv()
        ram_field = RAM(verbose=False).run(
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


def test_rams_elastic_no_negative_tl():
    """rams0.5 (Collins elastic PE) diverges for fast shear; the wrapper must
    surface that (warn + clamp) rather than return physically-impossible
    negative TL (field gain)."""
    import uacpy
    el = uacpy.BoundaryProperties(sound_speed=1800.0, density=2.0,
                                  attenuation=0.1, shear_speed=800.0,
                                  shear_attenuation=0.2)
    env = uacpy.Environment(bathymetry=200.0, ssp=1500.0,
                            bottom=uacpy.Bottom([uacpy.SeabedColumn([], el)]))
    src = Source(depths=50.0, frequencies=100.0)
    rcv = Receiver(depths=np.linspace(10, 190, 8), ranges=np.linspace(500, 6000, 20))
    with pytest.warns(UserWarning, match="unphysically negative|OAST"):
        tl = np.asarray(RAM(backend='rams').compute_tl(env, src, rcv).tl)
    finite = np.isfinite(tl)
    assert (tl[finite] >= 0).all(), "TL must be >= 0 (no field gain)"


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
        rams' depth capacity by 2x."""
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
    """rams0.5 is the elastic PE; on a fluid bottom it returns a null field.

    ``_validate_forced_backend`` already rejects ramsurf when its defining
    feature (altimetry) is absent. rams had no symmetric rule, so forcing it
    onto a fluid environment silently produced TL saturated at 200 dB at every
    range instead of an error. Auto-dispatch never routes fluid to rams.
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
        """504 written points + terminator == mr fits; 505 does not. The 505
        case previously reached the caller as a FileFormatError about a
        truncated tl.grid — exactly what this guard exists to prevent."""
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
