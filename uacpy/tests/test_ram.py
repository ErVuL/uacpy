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
        """Every supported Padé-coefficient count marches to a finite field.

        ``np_pade`` is the number of terms in the rational approximation of
        the propagator, and each term costs one tridiagonal solve per range
        step. A count the coefficient solver cannot deliver shows up as a
        stopped binary or an all-NaN grid, not as a small accuracy change, so
        finiteness is the discriminating assertion."""
        ram = RAM(verbose=False, dr=20.0, dz=2.0, np_pade=np_pade)
        result = ram.compute_tl(
            env=ram_env, source=ram_source, receiver=ram_receiver,
        )
        assert isinstance(result, Field)
        assert np.all(np.isfinite(result.data))

    def test_ram_stability_parameter(self, ram_env, ram_source, ram_receiver):
        """``ns_stability`` is the number of evanescent-spectrum points at
        which the rational approximation is forced to vanish, trading
        ``2n - ns`` accuracy constraints for stability (RAM manual §2). It
        changes the Padé coefficients, so the march must still complete and
        stay finite."""
        ram = RAM(verbose=False, dr=20.0, dz=2.0, ns_stability=1)
        result = ram.compute_tl(
            env=ram_env, source=ram_source, receiver=ram_receiver,
        )
        assert isinstance(result, Field)
        assert np.all(np.isfinite(result.data))

    def test_ram_custom_dr_dz(self, ram_env, ram_source, ram_receiver):
        """A pinned (dr, dz) bypasses the Lytaev optimiser entirely, so this
        exercises the path where uacpy marches the caller's own grid."""
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

        ``_run_tl`` defaults the pair to ``(1e6, 1.0)`` to collapse mpiramS's
        broadband window onto a single bin; those defaults must apply only
        when the caller left Q and T unset.
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

    mpiramS's horizontal-interpolation branch (``ihorz=1``) sizes its SSP
    resample grid as ``nrp = nint(maxval(rmax)/10000)``
    (``third_party/mpiramS/src/peramx.f90:245``). That rounds to 0 for any max
    receiver range below 5 km — a zero-length allocation, an all-NaN field and
    a SIGABRT (exit -6) — and to 1 below 15 km, which resamples the whole run
    onto a single profile. uacpy therefore drives mpiramS with ``ihorz=0``, so
    it steps directly between the per-range profiles uacpy writes itself.
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
        """Every rmax here is below the 5 km at which ``nint(rmax/10000)``
        first rounds up to 1, so each one hits the zero-length branch if the
        ``ihorz=1`` path is ever taken."""
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
        """``ihorz=0`` must not collapse range dependence: a varying SSP has to
        give a different short-range field than a range-independent one."""
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
    default epsilon at any ordinary frequency, so a warning on the default
    target would fire on essentially every run — alarm fatigue that trains
    callers to filter uacpy warnings entirely. An accuracy the caller *pinned*
    and did not get is still a warning.
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


def test_copy_preserves_the_unpinned_accuracy_default():
    """``copy()`` rebuilds from the stored constructor arguments, so a
    materialised default would come back as a caller-pinned value and flip
    the dz-floor message from status to warning on the round-trip."""
    m = RAM(verbose=False)
    c, cc = m.copy(), m.copy().copy()
    assert (m.accuracy, c.accuracy, cc.accuracy) == (None, None, None)
    assert not any(x._accuracy_explicit for x in (m, c, cc))
    assert all(x._accuracy == 1e-3 for x in (m, c, cc))

    p = RAM(verbose=False, accuracy=1e-6)
    assert p.copy()._accuracy_explicit and p.copy()._accuracy == 1e-6


class TestSedimentBlockIsResolvedByZread:
    """``zread`` (``ramsurf1.5.f:194-225``, identical in ``ramgeo1.5.f:215-246``
    and ``rams0.5.f:218-249``) pins each block point to the node
    ``i = 1.5 + z/dz`` and remembers only the immediately preceding index, so its
    collision push-down at :208 protects one duplicate depth and no more. A layer
    thinner than ``dz/2`` therefore had its two faces land in one cell, the
    deeper value overwrote the shallower, and the fill loop at :218-219 ramped
    linearly across the rest of the sub-bottom. Measured on the *default*
    dispatch path (``_prefer_ramgeo`` routes any layered fluid bottom there):
    a 0.6 m mud layer over an 1800 m/s basement came out as a 692 m gradient
    1500 → 1800 m/s, 22.6 dB from Scooter on the same environment."""

    @staticmethod
    def _env(thickness):
        from uacpy.core.bottom import Bottom, SeabedColumn, SedimentLayer
        halfspace = BoundaryProperties(sound_speed=1800.0, density=2.0,
                                       attenuation=0.5)
        layers = ([SedimentLayer(thickness=thickness, sound_speed=1500.0,
                                 density=1.2, attenuation=0.2)]
                  if thickness is not None else [])
        return Environment(
            name='block', bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs(
                np.array([[0.0, 1500.0], [100.0, 1500.0]])),
            bottom=Bottom(columns=[SeabedColumn(layers=layers,
                                                halfspace=halfspace)]))

    @staticmethod
    def _src_rcv():
        return (Source(depths=36.0, frequencies=50.0),
                Receiver(depths=[20.0, 50.0],
                         ranges=np.linspace(500.0, 8000.0, 151)))

    @staticmethod
    def _zread_nodes(block, dz):
        """The vendored node assignment, verbatim: ``i = 1.5 + z/dz`` with the
        one-step collision push-down. Returns the overwritten nodes."""
        assigned, iold, clobbered = {}, None, []
        for z, value in block:
            i = int(1.5 + z / dz)
            if iold is not None and i == iold:
                i += 1
            if i in assigned and assigned[i] != value:
                clobbered.append(i)
            assigned[i] = value
            iold = i
        return clobbered

    @staticmethod
    def _deck_blocks(path):
        """Every ``(depth, value)`` block in a ``ramgeo.in``, split on the
        ``-1 -1`` terminators. Parsed from the file the binary is handed, so this
        test does not share a code path with the model's own block builder."""
        blocks, current = [], []
        for line in path.read_text().splitlines():
            fields = line.split()
            if len(fields) != 2:
                continue
            if fields[0] == '-1':
                if current:
                    blocks.append(current)
                current = []
                continue
            try:
                current.append((float(fields[0]), float(fields[1])))
            except ValueError:
                continue
        return blocks

    @pytest.mark.parametrize('thickness', [0.6, 0.9, 1.0, 3.0])
    def test_no_block_point_is_overwritten(self, thickness, tmp_path):
        """The mechanism test: run the Fortran's own arithmetic over the deck
        uacpy actually wrote, and require every point to survive.

        The block is parsed out of ``ramgeo.in`` rather than rebuilt, because
        rebuilding it is exactly what went wrong once — the deck carries depths
        *relative to the seafloor*, wraps a bare half-space as a synthetic layer,
        and has extra attenuation points from the absorbing ramp."""
        env = self._env(thickness)
        src, rcv = self._src_rcv()
        result = RAM(verbose=False, work_dir=str(tmp_path),
                     cleanup=False).run(env, src, rcv)
        dz = float(result.metadata['dz'])
        deck = tmp_path / 'ramgeo.in'
        assert deck.exists(), f"no deck written: {sorted(p.name for p in tmp_path.iterdir())}"
        blocks = self._deck_blocks(deck)
        assert blocks, "parsed no blocks out of the deck"
        for n, block in enumerate(blocks):
            assert self._zread_nodes(block, dz) == [], (
                f"block {n} of the deck: a {thickness} m layer on dz={dz:.4f} m "
                f"loses points to zread's node collision, so the layer is "
                f"replaced by a linear ramp. Block: {block}")

    @pytest.mark.parametrize('thickness', [0.6, 0.9])
    def test_agrees_with_scooter(self, thickness):
        """Arbitrated against wavenumber integration, never another PE backend.
        These are the two thicknesses that collided on the auto grid: 22.60 dB at
        0.6 m and 22.21 dB at 0.9 m before the cap, 0.67 and 1.64 dB after.

        Thicker layers are deliberately not asserted here. 1.0-3.0 m does not
        collide, so this fix leaves its grid alone, and it sits 5.6-6.3 dB from
        Scooter — a layer spanned by about one cell of a grid the Lytaev
        optimiser chose for the *water* wavelength. That is ordinary
        under-resolution, a separate question from the lost block point."""
        from uacpy.models import Scooter
        env = self._env(thickness)
        src, rcv = self._src_rcv()
        ram = np.asarray(RAM(verbose=False).run(env, src, rcv).tl)
        scooter = np.asarray(Scooter(verbose=False, c_low=1400.0,
                                     c_high=1e9).run(env, src, rcv).tl)
        ranges = np.asarray(rcv.ranges)
        worst = 0.0
        for iz in (0, 1):
            for lo, hi in ((1500.0, 2100.0), (3500.0, 4100.0), (6500.0, 7100.0)):
                sel = (ranges >= lo) & (ranges <= hi)
                worst = max(worst, abs(float(np.nanmedian(
                    ram[iz, sel] - scooter[iz, sel]))))
        assert worst < 3.0, (
            f"{thickness} m layer: RAM is {worst:.2f} dB from Scooter")

    @pytest.mark.parametrize('thickness,dz', [(3.0, 2.0), (2.0, 1.9), (5.0, 4.0)])
    def test_a_clean_grid_is_left_alone(self, thickness, dz):
        """The bound ``gap >= 2*dz`` is sufficient but nowhere near necessary, so
        it must never be used to *judge* a grid — only to pick a replacement. A
        3 m step on ``dz = 2 m`` assigns nodes 1, 3, 4 and the skipped node is
        filled between two equal values, i.e. nothing is lost. Judging by the
        bound rejected this, which broke ``test_ram_with_rdl``."""
        env = self._env(thickness)
        src, rcv = self._src_rcv()
        result = RAM(verbose=False, dz=dz).run(env, src, rcv)
        assert float(result.metadata['dz']) == pytest.approx(dz)

    @pytest.mark.parametrize('dz', [None, 1.886792])
    def test_mpirams_is_exempt(self, dz):
        """Only the three Collins backends pin block points to grid nodes — the
        ``1.5+zi/dz`` arithmetic appears in ``ramsurf1.5.f``, ``ramgeo1.5.f`` and
        ``rams0.5.f`` and in no mpiramS source, which interpolates the profile
        onto the grid with ``interpolators.f90``'s ``interp1``. Tightening or
        refusing an mpiramS run would be a false positive."""
        env = self._env(0.6)
        src, rcv = self._src_rcv()
        result = RAM(verbose=False, backend='mpiramS', dz=dz).run(env, src, rcv)
        if dz is not None:
            assert float(result.metadata['dz']) == pytest.approx(dz)

    def test_a_bottom_without_layers_is_untouched(self):
        """The cap must not tighten a grid with no block step to resolve."""
        env_plain, env_layered = self._env(None), self._env(0.6)
        src, rcv = self._src_rcv()
        dz_plain = float(RAM(verbose=False).run(env_plain, src, rcv)
                         .metadata['dz'])
        dz_layered = float(RAM(verbose=False).run(env_layered, src, rcv)
                           .metadata['dz'])
        assert dz_plain > dz_layered
        # The writer wraps a pure half-space as one synthetic layer, so the block
        # is not empty — what matters is that its points do not collide.
        assert not RAM(verbose=False)._block_loses_a_point(
            env_plain, dz_plain, 800.0, 'ramgeo', 50.0)

    def test_a_pinned_dz_that_mangles_the_block_raises(self):
        """A pinned ``dz`` is the caller's choice everywhere else in this model,
        but here it silently changes the environment, so it has to raise."""
        from uacpy.core.exceptions import ConfigurationError
        env = self._env(0.6)
        src, rcv = self._src_rcv()
        with pytest.raises(ConfigurationError, match='overwrites the shallower'):
            RAM(verbose=False, dz=1.886792).run(env, src, rcv)

    def test_a_dz_the_caller_pinned_small_enough_is_accepted(self):
        env = self._env(0.6)
        src, rcv = self._src_rcv()
        result = RAM(verbose=False, dz=0.25).run(env, src, rcv)
        assert float(result.metadata['dz']) == pytest.approx(0.25)

    def test_an_unrepresentable_block_raises_rather_than_coarsening(self):
        """When the required refinement busts the binary's ``mz``, coarsening
        would silently substitute the ramp — so this cannot be met by
        coarsening and must be reported."""
        from uacpy.core.exceptions import ConfigurationError
        env = self._env(0.01)
        src, rcv = self._src_rcv()
        with pytest.raises(ConfigurationError, match='cannot be met by'):
            RAM(verbose=False, backend='ramgeo').run(env, src, rcv)
