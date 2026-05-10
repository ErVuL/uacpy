"""Tests for the RAM multi-backend dispatcher and the Collins-style I/O."""


import numpy as np
import pytest

from uacpy.core.environment import (
    BoundaryProperties,
    Environment,
    LayeredBottom,
    SedimentLayer,
)
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


def _elastic_bottom():
    return LayeredBottom(
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


# ─── Backend selection (no binary needed) ─────────────────────────────────


class TestBackendSelection:
    """Pure-Python dispatch logic — no native binaries required."""

    def test_fluid_flat_selects_mpirams(self):
        env = _env(bottom=_fluid_bottom())
        assert RAM(verbose=False, dr=20.0, dz=2.0).select_backend(env) == 'mpiramS'

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


# ─── LayeredBottom → piecewise ────────────────────────────────────────────


class TestPiecewiseBreakpoints:

    def test_two_layers_emit_step_function(self):
        lb = LayeredBottom(
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
        lb = LayeredBottom(
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
        lb = LayeredBottom(
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

    def test_rams_requires_shear_profiles(self, tmp_path):
        out = tmp_path / 'rams.in'
        with pytest.raises(ValueError, match="bottom_cs and bottom_attns"):
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
        assert result.field_type == 'tl'
        assert result.metadata['backend'] == 'ramsurf'
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
        assert result.field_type == 'tl'
        assert result.metadata['backend'] == 'rams'
        assert result.data.shape == (3, 20)
        assert np.all(np.isfinite(result.data))

    def test_collins_backend_broadband_returns_transfer_function(self):
        """ramsurf BROADBAND emits the patched complex envelope and the
        wrapper assembles an engineering travelling-wave H(f), tagged
        identically to mpiramS so synthesize_time_series treats both
        backends the same."""
        env = _env(bottom=_fluid_bottom(), altimetry=_rough_altimetry())
        src, rcv = self._src_rcv()
        ram = RAM(verbose=False, np_pade=6, dr=2.0, dz=0.25, zmax=400.0,
                  Q=2.0, T=2.0)
        f = ram.run(env, src, rcv, run_mode=RunMode.BROADBAND)
        assert f.field_type == 'transfer_function'
        assert f.metadata['backend'] == 'ramsurf'
        assert f.metadata['phase_reference'] == 'travelling_wave'
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
        from uacpy.models import KrakenField
        return KrakenField(verbose=False).run(
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
        with pytest.warns(UserWarning, match="raised dz from"):
            ram_field = RAM(verbose=False).run(
                env, src, rcv, run_mode=RunMode.COHERENT_TL,
            )
        assert ram_field.metadata['backend'] == 'mpiramS'
        ref_field = self._kraken_reference(env, src, rcv)
        self._assert_window_agreement(
            ram_field, ref_field, tol_db=4.5, label='mpiramS',
        )

    def test_rams_pekeris_elastic(self):
        env = _env(bottom=_elastic_bottom())
        src, rcv = self._src_rcv()
        with pytest.warns(UserWarning, match="raised dz from"):
            ram_field = RAM(verbose=False).run(
                env, src, rcv, run_mode=RunMode.COHERENT_TL,
            )
        assert ram_field.metadata['backend'] == 'rams'
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
        with pytest.warns(UserWarning, match="raised dz from"):
            ram_field = RAM(verbose=False).run(
                env_ram, src, rcv, run_mode=RunMode.COHERENT_TL,
            )
        assert ram_field.metadata['backend'] == 'ramsurf'
        ref_field = self._kraken_reference(env_ref, src, rcv)
        self._assert_window_agreement(
            ram_field, ref_field, tol_db=4.5, label='ramsurf1.5',
        )
