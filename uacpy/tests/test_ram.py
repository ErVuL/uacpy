"""RAM parabolic-equation-focused tests."""

import pytest
import numpy as np

from uacpy.models import RAM
from uacpy import Field
from uacpy.core import (
    Environment, Source, Receiver, SoundSpeedProfile, BoundaryProperties,
)

pytestmark = pytest.mark.requires_binary


class TestRAMAdvancedParameters:
    """Test RAM Pade orders and stability parameters."""

    @pytest.fixture
    def ram_env(self):
        return Environment(
            name="ram_test",
            bathymetry=100.0,
            ssp=1500.0
        )

    @pytest.fixture
    def ram_source(self):
        return Source(depths=50.0, frequencies=50.0)

    @pytest.fixture
    def ram_receiver(self):
        return Receiver(
            depths=np.linspace(10, 90, 9),
            ranges=np.linspace(100, 5000, 11)
        )

    @pytest.mark.parametrize('np_pade', [2, 6, 8])
    def test_ram_pade_order(self, ram_env, ram_source, ram_receiver, np_pade):
        """RAM converges across the supported Padé-coefficient counts."""
        ram = RAM(verbose=False, dr=20.0, dz=2.0, np_pade=np_pade)
        result = ram.compute_tl(
            env=ram_env, source=ram_source, receiver=ram_receiver,
        )
        assert isinstance(result, Field)
        assert np.all(np.isfinite(result.data))

    def test_ram_stability_parameter(self, ram_env, ram_source, ram_receiver):
        """Test RAM stability parameter."""
        ram = RAM(verbose=False, dr=20.0, dz=2.0, ns_stability=1)
        result = ram.compute_tl(
            env=ram_env, source=ram_source, receiver=ram_receiver,
        )
        assert isinstance(result, Field)
        assert np.all(np.isfinite(result.data))

    def test_ram_custom_dr_dz(self, ram_env, ram_source, ram_receiver):
        """Test RAM with custom range and depth steps."""
        ram = RAM(verbose=False, dr=10.0, dz=0.5)
        result = ram.compute_tl(
            env=ram_env, source=ram_source, receiver=ram_receiver,
        )
        assert isinstance(result, Field)
        assert np.all(np.isfinite(result.data))

    def test_ram_tl_honors_constructor_Q_T(
        self, ram_env, ram_source, ram_receiver, monkeypatch,
    ):
        """RAM(Q=…, T=…) values reach the in.pe file written for COHERENT_TL.

        Regression for the hardcoded ``Q_tl=1e6, T_tl=1.0`` in ``_run_tl``.
        """
        from uacpy.io import mpirams_writer as mpw
        from uacpy.models import ram as ram_mod
        captured = {}

        def fake_write_inpe(*args, **kwargs):
            captured['Q'] = kwargs['Q']
            captured['T'] = kwargs['T']
            raise RuntimeError("stop after writing in.pe")

        monkeypatch.setattr(mpw, 'write_inpe', fake_write_inpe)
        monkeypatch.setattr(ram_mod, 'write_inpe', fake_write_inpe)

        ram = RAM(Q=4.0, T=20.0, dr=20.0, dz=2.0, verbose=False)
        with pytest.raises(RuntimeError, match="stop after writing in.pe"):
            ram.compute_tl(env=ram_env, source=ram_source, receiver=ram_receiver)
        assert captured['Q'] == 4.0
        assert captured['T'] == 20.0


class TestRAMRangeDependentSSPShortRange:
    """A range-dependent SSP over a SHORT receiver range must not crash mpiramS.

    mpiramS's horizontal-interpolation branch sized its SSP resample grid as
    ``nrp = nint(rmax/10000)`` (``third_party/mpiramS/src/peramx.f90:245``),
    which rounds to 0 for any max receiver range below 5 km — a zero-length
    allocation, an all-NaN field and a SIGABRT (exit -6); below 15 km it
    silently mis-resampled to a single profile. uacpy now drives mpiramS with
    ``ihorz=0`` so it steps directly between the per-range profiles it writes
    itself.
    """

    _BOTTOM = BoundaryProperties(
        acoustic_type='half-space', sound_speed=1800.0,
        density=1.8, attenuation=0.5,
    )

    def _rd_env(self):
        """100 m channel, SSP varying from a near to a far column over 3 km."""
        z = np.array([0.0, 100.0])
        data = np.column_stack([[1500.0, 1490.0], [1520.0, 1480.0]])
        ssp = SoundSpeedProfile(
            depths=z, data=data, ranges=np.array([0.0, 3000.0]),
        )
        return Environment(bathymetry=100.0, ssp=ssp, bottom=self._BOTTOM)

    @pytest.mark.parametrize('rmax', [1500.0, 2000.0, 2500.0, 4000.0])
    def test_short_range_is_finite(self, rmax):
        """rmax < 5 km used to give nrp=nint(rmax/10000)=0 -> SIGABRT."""
        field = RAM(timeout=120).compute_tl(
            env=self._rd_env(),
            source=Source(depths=25.0, frequencies=50.0),
            receiver=Receiver(depths=[50.0], ranges=[rmax]),
        )
        assert isinstance(field, Field)
        data = np.asarray(field.data)
        assert data.size and np.isfinite(data).all()
        # A physical TL at ~rmax in a 100 m channel is well inside (0, 120) dB.
        tl = -20.0 * np.log10(np.abs(data).clip(1e-12))
        assert np.all((tl > 0.0) & (tl < 120.0))

    def test_short_range_keeps_range_dependence(self):
        """The fix must not silently collapse range dependence: a varying SSP
        must give a different short-range field than a range-independent one."""
        src = Source(depths=25.0, frequencies=50.0)
        rcv = Receiver(depths=[50.0], ranges=[1000.0, 2000.0, 3000.0])
        z = np.array([0.0, 100.0])
        ri = Environment(
            bathymetry=100.0,
            ssp=SoundSpeedProfile(depths=z, data=np.array([[1500.0], [1490.0]])),
            bottom=self._BOTTOM,
        )

        def tl(env):
            d = np.asarray(RAM(timeout=120).compute_tl(
                env=env, source=src, receiver=rcv).data).ravel()
            return -20.0 * np.log10(np.abs(d).clip(1e-12))

        tl_ri, tl_rd = tl(ri), tl(self._rd_env())
        assert np.isfinite(tl_ri).all() and np.isfinite(tl_rd).all()
        # Differing SSP columns must move the field by more than numerical noise.
        assert not np.allclose(tl_ri, tl_rd, atol=0.5)


def test_default_run_does_not_warn_about_its_own_accuracy_target():
    """A plain ``RAM()`` must not warn that uacpy's own default is unmet.

    The mpiramS stability floor (lambda_p/16) sits above the Lytaev dz for the
    default epsilon at any ordinary frequency, so warning on the default target
    fired on essentially every run — alarm fatigue that trains callers to
    filter uacpy warnings entirely. An accuracy the caller *pinned* and did not
    get is still a warning.
    """
    import warnings
    env = Environment(name='p', bathymetry=200.0, ssp=1500.0,
                      bottom=BoundaryProperties(acoustic_type='half-space',
                                                sound_speed=1800.0, density=1.8,
                                                attenuation=0.5))
    src = Source(depths=50.0, frequencies=100.0)
    rcv = Receiver(depths=100.0, ranges=np.array([1000.0]))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        RAM(verbose=False).run(env, src, rcv)
    budget = [x for x in w if 'accuracy budget' in str(x.message)]
    assert not budget, f"default run warned about its own default: {budget}"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        RAM(verbose=False, accuracy=1e-3).run(env, src, rcv)
    budget = [x for x in w if 'accuracy budget' in str(x.message)]
    assert budget, "an explicitly pinned accuracy that is not met must warn"
