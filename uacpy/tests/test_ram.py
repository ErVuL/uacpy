"""RAM parabolic-equation-focused tests.

Pade parameters and stability, the range/depth grids the deck carries, and the
tolerances the Fortran march itself imposes on where a receiver can land.
"""

import types
import warnings
from pathlib import Path

import pytest
import numpy as np

import uacpy
from uacpy.models import RAM, RunMode
from uacpy.models.ram import MAX_BATHY_SECTIONS, MPIRAMS_RANGE_TOL_M
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError,
)
from uacpy import Field
from uacpy.core import (Altimetry, Bathymetry, 
    Environment, Source, Receiver, SoundSpeedProfile, BoundaryProperties,
    Bottom, SeabedColumn, SedimentLayer,
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

    @staticmethod
    def _inpe_pade_line(work_dir):
        """``in.pe`` is positional: line index 6 carries ``np_pade nss``
        (peramx.f90:74-97, order pinned by ``write_inpe``)."""
        decks = sorted(work_dir.rglob('in.pe'))
        assert decks, f"no in.pe under {work_dir}"
        return decks[0].read_text().splitlines()[6].split()

    @pytest.mark.parametrize('np_pade', [2, 6, 8])
    def test_ram_pade_order(self, ram_env, ram_source, ram_receiver, np_pade,
                            tmp_path):
        """Every supported Padé-coefficient count marches to a finite field.

        ``np_pade`` is the number of terms in the rational approximation of
        the propagator, and each term costs one tridiagonal solve per range
        step. A count the coefficient solver cannot deliver shows up as a
        stopped binary or an all-NaN grid, not as a small accuracy change, so
        finiteness is the discriminating assertion — plus the deck itself,
        which must carry the requested count."""
        ram = RAM(verbose=False, dr=20.0, dz=2.0, np_pade=np_pade,
                  work_dir=tmp_path, cleanup=False)
        result = ram.compute_tl(
            env=ram_env, source=ram_source, receiver=ram_receiver,
        )
        assert isinstance(result, Field)
        assert np.all(np.isfinite(result.data))
        assert int(self._inpe_pade_line(tmp_path)[0]) == np_pade

    def test_ram_stability_parameter(self, ram_env, ram_source, ram_receiver,
                                     tmp_path):
        """``ns_stability`` is the number of evanescent-spectrum points at
        which the rational approximation is forced to vanish, trading
        ``2n - ns`` accuracy constraints for stability (RAM manual §2). It
        changes the Padé coefficients, so the march must still complete and
        stay finite — and the deck's ``np nss`` line must carry the count."""
        ram = RAM(verbose=False, dr=20.0, dz=2.0, ns_stability=1,
                  work_dir=tmp_path, cleanup=False)
        result = ram.compute_tl(
            env=ram_env, source=ram_source, receiver=ram_receiver,
        )
        assert isinstance(result, Field)
        assert np.all(np.isfinite(result.data))
        assert int(self._inpe_pade_line(tmp_path)[1]) == 1

    def test_ram_custom_dr_dz(self, ram_env, ram_source, ram_receiver):
        """A pinned (dr, dz) bypasses the Lytaev optimiser entirely, so this
        exercises the path where uacpy marches the caller's own grid — and
        the metadata must report that grid, not an optimiser suggestion."""
        ram = RAM(verbose=False, dr=10.0, dz=0.5)
        result = ram.compute_tl(
            env=ram_env, source=ram_source, receiver=ram_receiver,
        )
        assert isinstance(result, Field)
        assert np.all(np.isfinite(result.data))
        assert float(result.metadata['dr']) == pytest.approx(10.0)
        assert float(result.metadata['dz']) == pytest.approx(0.5)

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
        ram = np.asarray(RAM(verbose=False).run(env, src, rcv).db)
        scooter = np.asarray(Scooter(verbose=False, c_low=1400.0,
                                     c_high=1e9).run(env, src, rcv).db)
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


class TestAbsorbingRampLeavesTheSedimentColumnAlone:
    """``_ramp_absorbing_attenuation`` must not overwrite the deepest layer.

    Collins' readme (``ramsurf/readme.orig:127-133``) makes the ramp *"a
    buffer region"* below the modelled seabed, separate from the seabed's own
    attenuation. The block carries duplicated abscissae at each interface, so
    the point that pins a layer's value at its own base sits exactly at the
    ramp start whenever the absorber clamps to the sediment base — a strict
    ``<`` dropped it, and ``np.interp`` on the duplicate then returned the
    half-space value, marching a 0.02 dB/lambda layer at up to 5.0."""

    @staticmethod
    def _ramp(pairs, z_sediment_base, z_bottom, width=1000.0, floor=10.0):
        m = RAM.__new__(RAM)
        m.absorbing_layer_attn = floor
        return RAM._ramp_absorbing_attenuation(
            m, pairs, z_sediment_base, z_bottom, width)

    def test_clamped_ramp_keeps_the_layer_attenuation(self):
        # z_abs == z_sediment_base == 30: the clamped case the defect needed.
        out = self._ramp([(0.0, 0.02), (30.0, 0.02), (30.0, 5.0), (173.3, 5.0)],
                         30.0, 173.3)
        assert (0.0, 0.02) in out and (30.0, 0.02) in out      # layer intact
        assert (30.0, 5.0) in out                              # half-space starts
        assert out[-1] == (173.3, 10.0)                        # ramps to the floor

    def test_two_layers_both_survive(self):
        out = self._ramp([(0.0, 0.02), (10.0, 0.02), (10.0, 0.3), (40.0, 0.3),
                          (40.0, 5.0), (200.0, 5.0)], 40.0, 200.0)
        for p in [(0.0, 0.02), (10.0, 0.02), (10.0, 0.3), (40.0, 0.3), (40.0, 5.0)]:
            assert p in out

    def test_an_unclamped_ramp_starts_at_the_seabed(self):
        # The discriminating counterpart: pinning the layer must not drag the
        # ramp's start up with it. Here z_abs = 400 - 100 = 300.
        out = self._ramp([(0.0, 0.02), (30.0, 0.02), (30.0, 5.0), (400.0, 5.0)],
                         30.0, 400.0, width=100.0)
        assert (300.0, 5.0) in out       # half-space held flat to the ramp start
        assert out[-1] == (400.0, 10.0)


class TestElasticLayerFollowsSlopingBathymetry:
    """``rams0.5.f:490-516`` reads ``lamb(i)``/``mub(i)``/``rhob(i)`` at the
    absolute depth index, unlike ``ramgeo1.5.f:262-268`` whose ``matrc``
    re-anchors at the local seafloor (``ii=1 ... ii=ii+1``). So on rams a
    layered elastic column stays where the first profile section put it: with
    breakpoints taken only from the bottom and the SSP, a 20 m layer on a
    100 -> 300 m slope thinned to nothing within ~1 km, silently."""

    @staticmethod
    def _env(layered):
        hs = BoundaryProperties(
            acoustic_type='half-space', sound_speed=2200.0, density=2.2,
            attenuation=0.1, shear_speed=900.0, shear_attenuation=0.2)
        col = (SeabedColumn(
            layers=[SedimentLayer(thickness=20.0, sound_speed=1600.0,
                                  density=1.4, attenuation=0.3,
                                  shear_speed=300.0, shear_attenuation=1.0)],
            halfspace=hs)
            if layered else
            SeabedColumn.from_halfspace(hs, water_depth=100.0))
        return Environment(bathymetry=[(0.0, 100.0), (10000.0, 300.0)],
                           ssp=1500.0, bottom=Bottom([col]))

    def test_layered_elastic_column_is_reanchored_along_the_slope(self):
        segs = RAM()._collins_range_segments(
            self._env(layered=True), 'rams', zmax=700.0, freq=100.0)
        tops = [s['bottom_cs'][0][0] for s in segs]
        assert len(segs) > 10                       # was 1
        assert tops[0] == pytest.approx(100.0, abs=1.0)
        assert tops[-1] > 280.0                     # tracks to the deep end
        assert all(b >= a for a, b in zip(tops, tops[1:]))

    def test_half_space_column_gains_no_sections(self):
        # from_halfspace(synthesize=True) gives every profile point the same
        # value, so anchoring cannot matter and extra sections are pure cost.
        segs = RAM()._collins_range_segments(
            self._env(layered=False), 'rams', zmax=700.0, freq=100.0)
        assert len(segs) == 1

    def test_seafloor_relative_backends_gain_no_sections(self):
        # ramgeo/ramsurf re-anchor in matrc already.
        for kind in ('ramgeo', 'ramsurf'):
            segs = RAM()._collins_range_segments(
                self._env(layered=True), kind, zmax=700.0, freq=100.0)
            assert len(segs) == 1, kind




class TestProfileSectionsAreAllReachable:
    """``profl`` reads exactly ONE section marker per call
    (``ramgeo1.5.f:195``, defaulted at ``:194``), ``updat`` re-enters it only
    ``if(r.ge.rp)``
    (``:359``), and the march calls ``updat`` once per range step
    (``:78-84``). So at most one section is consumed per ``dr``, and because
    ``profl`` reads *sequentially* from the deck, sections closer together
    than ``dr`` are never reached at all — the march falls permanently
    behind and every later section is lost.

    The run still exits 0 and writes a full grid, computed from a truncated
    environment: measured at up to 8.75 dB with 101 sections written and 19
    reachable. ``ram.pdf`` p.8 states the rule — "The size of the smallest
    region is an upper bound on Delta-r."
    """

    @staticmethod
    def _env(n_cols):
        r_ax = np.linspace(0.0, 10000.0, n_cols)
        c = np.where(r_ax < 5000.0, 1500.0, 1400.0)
        return Environment(
            bathymetry=220.0,
            ssp=SoundSpeedProfile(depths=[0.0, 220.0],
                                  data=np.vstack([c, c]), ranges=r_ax),
            bottom=Bottom.from_halfspace(BoundaryProperties(
                sound_speed=1700.0, density=1.8, attenuation=0.5)),
        )

    def _segments(self, n_cols):
        m = RAM(backend='ramgeo', dz=2.0, zmax=700.0, verbose=False)
        return m, m._collins_range_segments(self._env(n_cols), 'ramgeo',
                                            700.0, 50.0)

    @pytest.mark.parametrize('n_cols', [21, 51, 101, 201])
    def test_every_section_is_reachable_within_the_march(self, n_cols):
        m, segs = self._segments(n_cols)
        markers = sorted({float(s['range']) for s in segs})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dr = m._constrain_dr_to_sections(500.0, segs, pinned=False)
        assert int(10000.0 / dr) >= len(markers) - 1, (
            f'{len(markers)} sections but only {int(10000.0/dr)} range steps '
            f'— the surplus would be silently dropped')

    def test_a_coarse_environment_leaves_dr_alone(self):
        # The discriminating counterpart: when the sections are already
        # further apart than dr there is nothing to constrain, and dr must
        # come back untouched rather than be clamped on every run.
        m, segs = self._segments(3)
        assert m._constrain_dr_to_sections(10.0, segs, pinned=False) == 10.0

    def test_a_pinned_dr_warns_rather_than_being_changed_in_silence(self):
        # Single-config rule: uacpy may not quietly rewrite a knob the user
        # set. It still reduces dr — wrong numbers are worse than a warning.
        m, segs = self._segments(101)
        with pytest.warns(UserWarning, match='silently dropped'):
            dr = m._constrain_dr_to_sections(500.0, segs, pinned=True)
        assert dr < 500.0


class TestSeafloorOutsideTheGrid:
    """``ram.pdf`` p.7 puts the grid bottom "well below the ocean bottom
    interface". Nothing enforces it — ``ramgeo1.5.f:133-135`` clamps
    ``iz=min(nz,iz)`` and ``rams0.5.f:135`` does not clamp at all — so a
    pinned ``zmax`` above the seafloor runs happily and returns numbers up to
    27.0 dB from an equivalent run with the seabed inside the grid."""

    ENV = Environment(
        bathymetry=220.0,
        ssp=SoundSpeedProfile(depths=[0.0, 220.0], data=[1500.0, 1500.0]),
        bottom=Bottom.from_halfspace(BoundaryProperties(
            sound_speed=1700.0, density=1.8, attenuation=0.5)))

    def _model(self, zmax):
        return RAM(backend='ramgeo', zmax=zmax, dz=2.0, dr=50.0, verbose=False)

    def test_pinned_zmax_above_the_seafloor_warns(self):
        with pytest.warns(UserWarning, match='outside the PE grid'):
            self._model(100.0)._warn_if_seafloor_outside_grid(100.0, self.ENV)

    def test_a_grid_that_clears_the_seafloor_is_silent(self):
        # The discriminating counterpart — this must not warn on every run.
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self._model(700.0)._warn_if_seafloor_outside_grid(700.0, self.ENV)

    def test_an_auto_zmax_is_never_flagged(self):
        # _compute_zmax clears the seafloor by construction, so only a pinned
        # value can reach the guard.
        m = RAM(backend='ramgeo', dz=2.0, dr=50.0, verbose=False)
        assert m.zmax is None
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            m.run(self.ENV, Source(depths=100.0, frequencies=50.0),
                  Receiver(depths=[50.0], ranges=np.array([1000.0, 2000.0])))


class TestSourceAgainstTheDepressedSurface:
    """``matrc`` zeroes every row down to ``izsrf``
    (``ramsurf1.5.f:281-290``) while ``selfs`` plants the source without
    consulting it (``:396-400``), so a source inside the depression yields an
    identically-zero field. ``_clamp_collins_envelope`` cannot catch it:
    ``|u| = 0`` is a large POSITIVE TL, not a negative or NaN one."""

    @staticmethod
    def _surface(depth, partial=False):
        return ([(0.0, 0.0), (5000.0, depth), (10000.0, depth)] if partial
                else [(0.0, depth), (10000.0, depth)])

    def _model(self):
        return RAM(backend='ramsurf', zmax=700.0, dz=2.0, dr=50.0,
                   verbose=False)

    def test_source_inside_the_depression_at_r0_is_refused(self):
        with pytest.raises(ConfigurationError, match='identically zero'):
            self._model()._check_source_below_depressed_surface(
                self._surface(30.0), 10.0, 2.0)

    def test_a_keel_deeper_than_the_source_further_along_only_warns(self):
        # The dangerous variant: the field dies partway and reads as a
        # shadow zone. The near field is still meaningful, so warn.
        with pytest.warns(UserWarning, match='shadow zone'):
            self._model()._check_source_below_depressed_surface(
                self._surface(30.0, partial=True), 10.0, 2.0)

    def test_the_message_states_the_observable_not_inf(self):
        # outpt adds eps=1e-20 before the log (ramsurf1.5.f:101), so a dead
        # field reports ~414-437 dB. Telling the user to look for inf sends
        # them after something that never appears.
        with pytest.raises(ConfigurationError) as exc:
            self._model()._check_source_below_depressed_surface(
                self._surface(30.0), 10.0, 2.0)
        msg = str(exc.value)
        assert '414-437 dB' in msg, 'the message must name the observable'
        assert 'eps=1e-20' in msg, 'and why it is not inf'

    def test_a_source_below_every_depression_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self._model()._check_source_below_depressed_surface(
                self._surface(30.0), 100.0, 2.0)


class TestSeafloorGuardCoversEveryBackend:
    """The seabed-outside-the-grid pathology is not Collins-specific.
    mpiramS clamps the seafloor index identically —
    ``mpiramS/src/ram.f90:101`` ``iz=min(nz,iz)`` against
    ``ramgeo1.5.f:135`` — so a guard wired only into the Collins grid
    resolver leaves a reachable, silent ~29 dB error on the other backend.
    A correct helper does not bound its consumers; the check belongs on the
    funnel both paths share."""

    ENV = Environment(
        bathymetry=220.0,
        ssp=SoundSpeedProfile(depths=[0.0, 220.0], data=[1500.0, 1500.0]),
        bottom=Bottom.from_halfspace(BoundaryProperties(
            sound_speed=1700.0, density=1.8, attenuation=0.5)))

    @pytest.mark.parametrize('backend', ['ramgeo', 'ramsurf', 'rams',
                                         'mpiramS'])
    def test_a_pinned_zmax_below_the_seafloor_warns_on_every_backend(
            self, backend):
        m = RAM(backend=backend, zmax=200.0, dz=2.0, dr=50.0, verbose=False)
        with pytest.warns(UserWarning, match='outside the PE grid'):
            m._compute_zmax(self.ENV, 50.0)

    @pytest.mark.parametrize('backend', ['ramgeo', 'mpiramS'])
    def test_a_grid_that_clears_the_seafloor_stays_silent(self, backend):
        m = RAM(backend=backend, zmax=1500.0, dz=2.0, dr=50.0, verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            m._compute_zmax(self.ENV, 50.0)

    @pytest.mark.parametrize('dz,zsrf,zs,alive', [
        (0.7, 30.0, 29.7, True),    # zeroed only to 29.4 — the field survives
        (0.7, 30.0, 29.0, False),   # inside the zeroed rows
        (2.0, 30.0, 29.0, False),
    ])
    def test_the_band_the_truncation_leaves_alive_is_not_refused(
            self, dz, zsrf, zs, alive):
        """``izsrf = 1.0 + zsrf/dz`` (``ramsurf1.5.f:115``) truncates, and
        ``matrc`` zeroes rows ``1..izsrf`` (``:282``) where row ``i`` sits at
        ``(i-1)*dz``. The zeroed region therefore ends up to one ``dz``
        ABOVE ``zsrf``, so comparing against ``zsrf`` itself refuses sources
        that are measurably alive (84-91 dB at dz=0.7, zsrf=30, zs=29.7)."""
        m = RAM(backend='ramsurf', zmax=700.0, dz=dz, dr=50.0, verbose=False)
        surface = [(0.0, zsrf), (10000.0, zsrf)]
        if alive:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m._check_source_below_depressed_surface(surface, zs, dz)
        else:
            with pytest.raises(ConfigurationError):
                m._check_source_below_depressed_surface(surface, zs, dz)


class TestRamsSeafloorIndexMustFitTheGrid:
    """``rams0.5`` alone does not clamp the seafloor index: ``rams0.5.f:135``
    is a bare ``iz=z/dz`` where ``ramgeo1.5.f:135`` and ``ramsurf1.5.f:120``
    both apply ``min(nz,iz)``. Past ``nz`` the fluid-solid stencil at
    ``rams0.5.f:590`` reads ``lamw(iz+2)`` — one slot beyond ``profl``'s
    initialised ``1..nz+2`` (``:200-205``) — and divides by an unwritten
    element, so the **whole field** returns NaN from a run that exits 0.

    Measured on three independent grids, the boundary is exactly
    ``iz = nz+1``. The fluid backends clamp and degrade to "seafloor at the
    grid bottom", returning usable numbers, so only rams is fatal.

    The condition is on the INDEX, not the depth: ``nz = zmax/dz - 0.5`` and
    ``iz = z/dz``, so a ``zmax`` up to half a cell below the seabed clears a
    naive ``zmax > depth`` test and still dies — measured, ``zmax=200.3`` over
    a 200 m seabed with ``dz=1``.
    """

    ENV = Environment(
        bathymetry=200.0,
        ssp=SoundSpeedProfile(depths=[0.0, 200.0], data=[1500.0, 1500.0]),
        bottom=Bottom.from_halfspace(BoundaryProperties(
            sound_speed=1800.0, density=2.0, attenuation=0.1,
            shear_speed=800.0, shear_attenuation=0.2)))

    def _model(self, backend, zmax):
        return RAM(backend=backend, zmax=zmax, dz=1.0, dr=25.0, verbose=False)

    def test_rams_refuses_a_zmax_that_pushes_iz_past_nz(self):
        # 0.3 m below the seabed: zmax > depth, so a depth-only test passes.
        with pytest.raises(ConfigurationError, match='iz > nz|does not.*clamp'):
            self._model('rams', 200.3)._resolve_collins_grid(
                self.ENV, 100.0, 'rams', 6000.0, None, None, None)

    def test_half_a_cell_more_is_accepted(self):
        # The discriminating counterpart: iz == nz is in-domain and the run
        # returns finite numbers. The guard must not creep upward.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._model('rams', 200.6)._resolve_collins_grid(
                self.ENV, 100.0, 'rams', 6000.0, None, None, None)

    @pytest.mark.parametrize('backend', ['ramgeo', 'ramsurf'])
    def test_the_clamped_backends_only_warn(self, backend):
        # They apply min(nz,iz) and return usable numbers, so refusing them
        # would reject a run that works.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._model(backend, 200.3)._resolve_collins_grid(
                self.ENV, 100.0, backend, 6000.0, None, None, None)


class TestRamsSeafloorIndexFloor:
    """The counterpart to the upper bound: ``rams0.5`` clamps ``iz`` at
    neither end. Below 2 it indexes before the start of its own arrays —
    ``matrc:418`` reads ``lamw(0)`` and ``matrc:717`` forms ``i0=2*iz-3=-1``
    to read ``r5(0)``. A bounds-checked build aborts; the shipped ``-O2``
    build exits 0 and returns a **fully finite, plausible** field, which is
    what makes it worth refusing rather than warning about.

    ``updat`` recomputes the index from the interpolated bathymetry at every
    range step (``rams0.5.f:305``), so the SHALLOWEST point on the track
    decides it. Every other depth guard in the class is written against
    ``env.depth`` — the deepest — which is why this one exists separately and
    why the case is reachable with no pinned knobs at all.

    The threshold moves with the auto ``dz``, so these cases are stated
    against the resolved grid rather than a remembered number.
    ``_align_dz_with_seafloor`` takes the coarsest ``dz`` that divides the
    shelf depth and still clears the shear-wavelength cap — 5.714 m here
    (c_s = 800 m/s, 10 Hz). So a shelf *deeper* than that cap gets split into
    at least two cells and is in-domain (6 m → dz 3.0, iz 2; 8 m → dz 4.0,
    iz 2), while a shelf *at or under* the cap cannot be split at all: the
    coarsest divisor is the depth itself, the whole water column becomes one
    cell, and ``iz`` falls to 1 (5 m → dz 5.0; 3 m → dz 3.0). The refusal
    boundary is therefore the shear cap, and both sides of it are pinned
    below.
    """

    @staticmethod
    def _env(shelf):
        return Environment(
            bathymetry=Bathymetry(ranges=[0.0, 8000.0, 20000.0],
                                  depths=[200.0, shelf, shelf]),
            ssp=SoundSpeedProfile(depths=[0.0, 200.0], data=[1500.0, 1500.0]),
            bottom=Bottom.from_halfspace(BoundaryProperties(
                sound_speed=1800.0, density=2.0, attenuation=0.1,
                shear_speed=800.0, shear_attenuation=0.2)))

    # 20 m, not 3 m: at 10 Hz the auto dz is 5.7 m, so a 3 m source lands in
    # row 1 and trips the source-row guard instead of the one under test.
    SRC = Source(depths=20.0, frequencies=10.0)
    RCV = Receiver(depths=[3.0, 6.0], ranges=np.linspace(1000.0, 15000.0, 15))

    @pytest.mark.parametrize('shelf', [5.0, 3.0])
    def test_a_shoaling_track_that_drives_iz_below_two_is_refused(self, shelf):
        # Both sit at or under the 5.714 m shear cap, so the column cannot be
        # split and iz = 1 — the case that returns 30/30 finite values with no
        # warning while reading out of bounds.
        with pytest.raises(ConfigurationError, match='shoals'):
            RAM(backend='rams', verbose=False).run(
                self._env(shelf), self.SRC, self.RCV)

    @pytest.mark.parametrize('shelf', [40.0, 14.0, 6.0])
    def test_a_track_that_stays_above_two_cells_runs(self, shelf):
        # The discriminating counterpart — iz >= 2 is in-domain and must not
        # be refused. 6 m is the tightest admitted case: just past the shear
        # cap, so the alignment can split it into two cells (dz 3.0, iz 2).
        # Pinning it here means a future change to the auto dz cannot move
        # the boundary without one of these two tests failing.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f = RAM(backend='rams', verbose=False).run(
                self._env(shelf), self.SRC, self.RCV)
        assert np.isfinite(f.db).all()

    def test_the_clamped_backends_are_unaffected(self):
        # ramgeo applies max(2,iz) (ramgeo1.5.f:134), so the same track is
        # fine there and refusing it would be wrong.
        env = Environment(
            bathymetry=Bathymetry(ranges=[0.0, 8000.0, 20000.0],
                                  depths=[200.0, 8.0, 8.0]),
            ssp=SoundSpeedProfile(depths=[0.0, 200.0], data=[1500.0, 1500.0]),
            bottom=Bottom.from_halfspace(BoundaryProperties(
                sound_speed=1800.0, density=2.0, attenuation=0.1)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            RAM(backend='ramgeo', verbose=False).run(env, self.SRC, self.RCV)


class TestSubBottomMarginIsEnoughForTheAbsorber:
    """``ram.pdf`` p.7 asks for the grid bottom "well below the ocean bottom
    interface", with the attenuation raised "over the lower few wavelengths".
    A guard keyed on ``zmax > env.depth`` draws its line where the error has
    already saturated: measured on ramgeo over a 220 m seabed against an
    ample grid, ``zmax=220`` is 27.0 dB (caught) but ``zmax=222`` — two
    metres of sub-bottom — is 16.0 dB and used to pass in silence.

    The threshold is three wavelengths, calibrated against that curve rather
    than assumed: it must stay silent where the error is negligible."""

    ENV = Environment(
        bathymetry=220.0,
        ssp=SoundSpeedProfile(depths=[0.0, 220.0], data=[1500.0, 1500.0]),
        bottom=Bottom.from_halfspace(BoundaryProperties(
            sound_speed=1700.0, density=1.8, attenuation=0.5)))

    def _warns(self, zmax):
        m = RAM(backend='ramgeo', zmax=zmax, dz=2.0, dr=50.0, verbose=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            m._warn_if_seafloor_outside_grid(zmax, self.ENV, dz=2.0,
                                             kind='ramgeo', freq=50.0)
        return any('grid' in str(w.message) or 'absorbing' in str(w.message)
                   for w in caught)

    @pytest.mark.parametrize('zmax,measured_db', [(300.0, 3.1), (222.0, 16.0),
                                                  (220.0, 27.0)])
    def test_a_thin_sub_bottom_margin_warns(self, zmax, measured_db):
        assert self._warns(zmax), f'{measured_db} dB error passed in silence'

    @pytest.mark.parametrize('zmax,measured_db', [(700.0, 0.32), (1500.0, 0.0)])
    def test_an_ample_margin_stays_silent(self, zmax, measured_db):
        # The discriminating half. Comparing against uacpy's own 20-lambda
        # auto pad would fire here, at 0.32 dB — a warning users would learn
        # to ignore.
        assert not self._warns(zmax), \
            f'{measured_db} dB error warned — the threshold is too eager'


class TestSourceRowIsActuallySolved:
    """Every RAM binary plants the source with ``si=1.0+zs/dz`` / ``is=ifix(si)``
    and splits it across ``u(is)``, ``u(is+1)`` (``ramgeo1.5.f:389-393``,
    ``ramsurf1.5.f:396-400``, ``rams0.5.f:357-361``,
    ``mpiramS/src/ram.f90:110-114``), and every solver sweeps from row 2
    (``ramgeo1.5.f:312``, ``rams0.5.f:836``, ``solvetri.f90:47``).

    So ``zs < dz`` freezes part of the source in ``u(1)`` for the whole march,
    where it acts as a permanent Dirichlet source on the pressure-release
    surface: the field gets *louder* as the source approaches the surface,
    which is backwards. Measured against **Kraken** as an independent arbiter
    — never backend-vs-backend — at ~46 dB mean and 72 dB peak on uacpy's own
    default grid, silent on every backend.

    This is the flat-surface case of the row-1 kill that
    ``_check_source_below_depressed_surface`` catches for a ramsurf keel; it
    needs no altimetry and is not ramsurf-specific.
    """

    @staticmethod
    def _env(elastic=False, altimetry=False):
        bp = (BoundaryProperties(sound_speed=1800.0, density=2.0,
                                 attenuation=0.1, shear_speed=800.0,
                                 shear_attenuation=0.2) if elastic else
              BoundaryProperties(sound_speed=1700.0, density=1.8,
                                 attenuation=0.5))
        kw = {}
        if altimetry:
            kw['altimetry'] = Altimetry(ranges=[0.0, 5000.0],
                                        heights=[0.0, 0.0])
        return Environment(
            bathymetry=100.0,
            ssp=SoundSpeedProfile(depths=[0.0, 100.0], data=[1500.0, 1500.0]),
            bottom=Bottom.from_halfspace(bp), **kw)

    RCV = Receiver(depths=[10.0, 50.0, 90.0],
                   ranges=np.linspace(500.0, 5000.0, 10))

    @pytest.mark.parametrize('backend', ['ramgeo', 'ramsurf', 'rams',
                                         'mpiramS'])
    def test_a_source_inside_the_first_cell_is_refused(self, backend):
        env = self._env(elastic=(backend == 'rams'),
                        altimetry=(backend == 'ramsurf'))
        with pytest.raises(ConfigurationError, match='shallower than one'):
            RAM(backend=backend, verbose=False).run(
                env, Source(depths=0.5, frequencies=100.0), self.RCV)

    @pytest.mark.parametrize('backend', ['ramgeo', 'mpiramS'])
    def test_a_source_below_the_first_cell_runs(self, backend):
        # The discriminating counterpart.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f = RAM(backend=backend, verbose=False).run(
                self._env(), Source(depths=25.0, frequencies=100.0), self.RCV)
        assert np.isfinite(f.db).any()

    def test_the_guard_is_keyed_to_dz_not_to_a_fixed_depth(self):
        # A 0.5 m source is fine once dz is small enough to resolve it — the
        # bound is the mechanism (is >= 2), not an absolute shallow-source
        # rule.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f = RAM(backend='ramgeo', dz=0.1, dr=50.0, verbose=False).run(
                self._env(), Source(depths=0.5, frequencies=100.0), self.RCV)
        assert np.isfinite(f.db).any()


class TestEveryPerRangeStreamBoundsDr:
    """``updat`` advances three indices one step per range step: the profile
    marker (``if(r.ge.rp)``), the bathymetry index
    (``ramgeo1.5.f:348`` ``if(r.ge.rb(ib+1))ib=ib+1``) and, on ramsurf, the
    altimetry index (``ramsurf1.5.f:346``). Bounding only the first lets the
    other two fall behind, after which the seafloor is linearly extrapolated
    from a pair of points far astern — range dependence silently lost.

    Reachable on the default path: auto ``dr`` is 216 m at 25 Hz, so any
    EMODnet-DTM-resolution bathymetry (~115 m) trips it.
    """

    @staticmethod
    def _bathy(n):
        r = np.linspace(0.0, 5000.0, n)
        d = np.interp(r, [0.0, 2000.0, 3500.0, 5000.0],
                      [100.0, 100.0, 60.0, 100.0])
        return Bathymetry(ranges=r, depths=d)

    def _model(self):
        return RAM(backend='ramgeo', dz=0.5, verbose=False)

    def test_bathymetry_finer_than_dr_bounds_dr(self):
        env = Environment(
            bathymetry=self._bathy(101),          # 50 m spacing
            ssp=SoundSpeedProfile(depths=[0.0, 100.0], data=[1500.0, 1500.0]),
            bottom=Bottom.from_halfspace(BoundaryProperties(
                sound_speed=1700.0, density=1.8, attenuation=0.5)))
        m = self._model()
        segs = m._collins_range_segments(env, 'ramgeo', 200.0, 100.0)
        bathy = [float(x) for x, _ in env.bathymetry.to_pairs().tolist()]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = m._constrain_dr_to_sections(250.0, segs, pinned=False,
                                              bathymetry_ranges=bathy)
        assert out == pytest.approx(50.0)

    def test_coarse_bathymetry_leaves_dr_alone(self):
        env = Environment(
            bathymetry=self._bathy(11),           # 500 m spacing
            ssp=SoundSpeedProfile(depths=[0.0, 100.0], data=[1500.0, 1500.0]),
            bottom=Bottom.from_halfspace(BoundaryProperties(
                sound_speed=1700.0, density=1.8, attenuation=0.5)))
        m = self._model()
        segs = m._collins_range_segments(env, 'ramgeo', 200.0, 100.0)
        bathy = [float(x) for x, _ in env.bathymetry.to_pairs().tolist()]
        assert m._constrain_dr_to_sections(
            250.0, segs, pinned=False, bathymetry_ranges=bathy) == 250.0

    def test_a_constructor_pinned_dr_warns_before_being_changed(self):
        # Single-config rule: dr set on __init__ is as user-set as one passed
        # to run(), and must not be rewritten in silence.
        env = Environment(
            bathymetry=self._bathy(101),
            ssp=SoundSpeedProfile(depths=[0.0, 100.0], data=[1500.0, 1500.0]),
            bottom=Bottom.from_halfspace(BoundaryProperties(
                sound_speed=1700.0, density=1.8, attenuation=0.5)))
        m = RAM(backend='ramgeo', dr=250.0, dz=0.5, verbose=False)
        segs = m._collins_range_segments(env, 'ramgeo', 200.0, 100.0)
        bathy = [float(x) for x, _ in env.bathymetry.to_pairs().tolist()]
        with pytest.warns(UserWarning, match='silently dropped'):
            m._constrain_dr_to_sections(250.0, segs, pinned=True,
                                        bathymetry_ranges=bathy)


class TestZeroRangeReceiverIsNaN:
    """A receiver at r = 0 sits on the source axis, where the point-source
    cylindrical-spreading factor 1/sqrt(r) is singular: the column is
    returned as NaN with a warning — the convention every model shares —
    never a finite value computed at a substituted range but labelled r = 0.
    The broadband path handles its own zero-range output bin separately by
    clipping the coordinate to match the scaled data."""

    @staticmethod
    def _setup():
        env = Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.5,
                                      attenuation=0.5))
        src = Source(depths=25.0, frequencies=100.0)
        rcv = Receiver(depths=np.array([50.0]),
                       ranges=np.array([0.0, 500.0, 1000.0]))
        return env, src, rcv

    def test_mpirams_tl_r0_column_is_nan_and_warns_once(self):
        env, src, rcv = self._setup()
        with pytest.warns(UserWarning, match=r"starts at r=0 m") as record:
            field = RAM(verbose=False).run(env, src, rcv,
                                           run_mode=RunMode.COHERENT_TL)
        r0_warnings = [w for w in record
                       if 'r=0' in str(w.message) or 'r <= 0' in str(w.message)]
        assert len(r0_warnings) == 1, (
            f"expected the shared run() warning alone, got "
            f"{[str(w.message) for w in r0_warnings]}")
        assert np.isnan(field.data[:, 0]).all()
        assert np.isfinite(field.data[:, 1:]).all()
        np.testing.assert_allclose(field.coords['range'],
                                   [0.0, 500.0, 1000.0])


class TestBroadbandSynthesisWindowAnchor:
    """The auto time-window of ``to_time_trace`` anchors on physical speeds
    (``c_max``), never on the Padé expansion point. Stamped as ``c0``, the
    expansion point (1677 m/s here, against an all-1500 m/s medium) entered
    the fastest-speed max and opened the window a half-second early: the
    3.333 s water arrival fell past the window end and wrapped silently to
    ~2.94 s. It now rides on ``pe_reference_speed``, which the anchor
    ignores."""

    def test_time_trace_window_contains_the_water_arrival(self):
        env = Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1500.0, density=1.5,
                                      attenuation=0.5))
        src = Source(depths=25.0, frequencies=100.0)
        rcv = Receiver(depths=np.array([50.0]), ranges=np.array([5000.0]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tf = RAM(verbose=False, Q=4.0, T=0.4, theta_max=45.0).run(
                env, src, rcv, run_mode=RunMode.BROADBAND)
            trace = tf.to_time_trace(depth=50.0, range=5000.0)
        assert tf.metadata['pe_reference_speed'] > 1600.0
        assert 'c0' not in tf.metadata
        t = np.asarray(trace.coords['time'], dtype=float)
        arrival = 5000.0 / 1500.0
        assert t[0] <= arrival <= t[-1], (
            f"synthesis window [{t[0]:.3f}, {t[-1]:.3f}] s misses the "
            f"{arrival:.3f} s water arrival")
        peak = float(t[np.argmax(np.abs(trace.data))])
        assert abs(peak - arrival) < 0.05, (
            f"energy peak at {peak:.3f} s is not the {arrival:.3f} s arrival "
            f"(a wrapped record puts it ~one window earlier)")


def _elastic_halfspace():
    return BoundaryProperties(
        acoustic_type='half-space', sound_speed=1700.0, shear_speed=400.0,
        density=1.8, attenuation=0.5, shear_attenuation=0.5)


class TestConstructorGuards:
    """Pure-Python constructor contracts; no binary is launched."""

    @pytest.mark.parametrize('bad', [1, 11, 0, -3])
    def test_np_pade_outside_2_to_10_raises(self, bad):
        with pytest.raises(ConfigurationError, match=r'np_pade'):
            RAM(np_pade=bad, verbose=False)

    def test_np_pade_must_be_an_integer(self):
        with pytest.raises(ConfigurationError, match=r'np_pade'):
            RAM(np_pade=6.0, verbose=False)

    def test_rams_theta_accepts_a_callable(self):
        m = RAM(rams_theta=lambda f: 20.0 + f / 100.0, verbose=False)
        assert m._theta_for_freq(200.0) == pytest.approx(22.0)
        assert m._theta_for_freq(500.0) == pytest.approx(25.0)

    def test_rams_theta_float_out_of_range_raises(self):
        with pytest.raises(ConfigurationError, match=r'rams_theta'):
            RAM(rams_theta=95.0, verbose=False)

    def test_low_absorbing_layer_attn_warns(self):
        """Below 1 dB/λ the artificial absorber lets domain-bottom
        reflections back into the field (ram.py's Collins-readme note)."""
        with pytest.warns(UserWarning, match='absorbing_layer_attn'):
            RAM(absorbing_layer_attn=0.5, verbose=False)

    def test_ordinary_absorbing_layer_attn_is_silent(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            RAM(absorbing_layer_attn=5.0, verbose=False)
        assert not [w for w in caught
                    if 'absorbing_layer_attn' in str(w.message)]


class TestClampCollinsEnvelope:
    """``_clamp_collins_envelope`` sanitises one Collins run: NaN/inf TL and
    unphysically negative TL (|p/p0| > 1, impossible for a passive medium)
    are clamped to the TL_MAX_DB floor with a counting UserWarning, phase
    preserved; a small negative near 0 and the z = 0 surface node are valid
    values, clamped without being counted."""

    FLOOR = 10.0 ** (-200.0 / 20.0)

    @staticmethod
    def _raw():
        pcomplex = np.array([
            0.3 * np.exp(0.7j),      # valid sample
            2.0 * np.exp(0.3j),      # TL = -6 dB: divergence
            np.nan + 1j * np.nan,    # NaN
            5.0 + 0.0j,              # TL = +inf marker below
            1.06 * np.exp(-0.2j),    # TL = -0.5 dB: noise around 0
            0.0 + 0.0j,              # surface node
        ])
        tl = np.array([10.46, -6.0, np.nan, np.inf, -0.5, 200.0])
        return {'tl': tl, 'pcomplex': pcomplex, 'frequency': 100.0}

    @staticmethod
    def _envs():
        fluid = Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.5,
                                      attenuation=0.5))
        elastic = Environment(bathymetry=100.0, ssp=1500.0,
                              bottom=_elastic_halfspace())
        return fluid, elastic

    def test_divergent_samples_are_floored_with_a_counting_warning(self):
        fluid, _ = self._envs()
        m = RAM(verbose=False)
        with pytest.warns(UserWarning, match=r'3/6 TL samples') as rec:
            out = m._clamp_collins_envelope(self._raw(), fluid, 'ramgeo')
        assert '200' in str(rec[0].message)
        # Divergent magnitudes pinned to the 200 dB floor, phase kept.
        assert out[1] == pytest.approx(self.FLOOR * np.exp(0.3j))
        assert out[2] == pytest.approx(self.FLOOR + 0.0j)
        assert out[3] == pytest.approx(self.FLOOR + 0.0j)

    def test_valid_samples_pass_through_unchanged(self):
        fluid, _ = self._envs()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = RAM(verbose=False)._clamp_collins_envelope(
                self._raw(), fluid, 'ramgeo')
        assert out[0] == pytest.approx(0.3 * np.exp(0.7j))

    def test_noise_and_surface_node_are_not_counted_as_divergence(self):
        fluid, _ = self._envs()
        raw = self._raw()
        # Keep only the three uncounted kinds: valid, small negative, zero.
        for key in ('tl', 'pcomplex'):
            raw[key] = raw[key][[0, 4, 5]]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            out = RAM(verbose=False)._clamp_collins_envelope(
                raw, fluid, 'ramgeo')
        assert not [w for w in caught if 'TL samples' in str(w.message)]
        # |p/p0| slightly above 1 is capped to 1, phase kept.
        assert out[1] == pytest.approx(1.0 * np.exp(-0.2j))
        # The z = 0 pressure-release node is raised to the floor, not 600 dB.
        assert out[2] == pytest.approx(self.FLOOR + 0.0j)

    def test_rams_elastic_divergence_names_the_stable_alternatives(self):
        _, elastic = self._envs()
        with pytest.warns(UserWarning, match='OAST / Scooter'):
            RAM(verbose=False)._clamp_collins_envelope(
                self._raw(), elastic, 'rams')


class TestGridResolverConstraints:
    """The Lytaev optimiser minimises its error model alone; ram.md §5 lists
    the constraints applied to its output afterwards. Feeding
    ``_compute_grid_lytaev`` a known optimiser result via a stub isolates
    exactly those constraints."""

    @staticmethod
    def _stub(model, dr, dz):
        def fake(**kw):
            return {'dr': dr, 'dz': dz}, kw['eps0'], kw['theta0']
        model._optimize_grid_relaxing = fake
        return model

    @staticmethod
    def _envs():
        fluid = Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.5,
                                      attenuation=0.5))
        elastic = Environment(bathymetry=100.0, ssp=1500.0,
                              bottom=_elastic_halfspace())
        return fluid, elastic

    def test_rams_dr_safety_factor_divides_the_optimum(self):
        # dr_opt = 10 → safety 10/5 = 2 m, tighter than the λ cap
        # c_min/(5f) = 1500/500 = 3 m.
        _, elastic = self._envs()
        m = self._stub(RAM(backend='rams', verbose=False), dr=10.0, dz=1.0)
        dr, _ = m._compute_grid_lytaev(elastic, 100.0, max_range=5000.0,
                                       kind='rams')
        assert dr == pytest.approx(2.0)

    def test_rams_wavelength_cap_binds_when_tighter(self):
        # With the safety factor disabled the λ cap c_min/(5f) = 3 m rules.
        _, elastic = self._envs()
        m = self._stub(RAM(backend='rams', rams_dr_safety_factor=1.0,
                           verbose=False), dr=10.0, dz=1.0)
        dr, _ = m._compute_grid_lytaev(elastic, 100.0, max_range=5000.0,
                                       kind='rams')
        assert dr == pytest.approx(1500.0 / (5.0 * 100.0))

    def test_rams_dz_is_capped_at_the_shear_wavelength(self):
        from uacpy.models._pade_optimizer import rams_dz_shear_cap
        _, elastic = self._envs()
        m = self._stub(RAM(backend='rams', verbose=False), dr=10.0, dz=1.0)
        _, dz = m._compute_grid_lytaev(elastic, 100.0, max_range=5000.0,
                                       kind='rams')
        assert dz == pytest.approx(rams_dz_shear_cap(400.0, 100.0))

    def test_depth_grid_is_capped_at_10000_points(self):
        # dz = 5 mm over 100 m asks for 20000 depth points; the runtime cap
        # coarsens it to ~1 cm with a warning. f = 10 kHz keeps the λ_p/16
        # stability floor (≈ 9.4 mm) below the capped value.
        fluid, _ = self._envs()
        m = self._stub(RAM(backend='ramgeo', verbose=False),
                       dr=10.0, dz=0.005)
        with pytest.warns(UserWarning, match='10000'):
            _, dz = m._compute_grid_lytaev(fluid, 10000.0, max_range=2000.0,
                                           kind='ramgeo')
        assert dz == pytest.approx(0.01, rel=1e-6)


class TestRamsurfSurfaceCrestClamp:
    """ramsurf1.5 takes a surface depression profile (zsrf >= 0 below z=0),
    so altimetry wave crests (height > 0) are clamped to sea level with a
    UserWarning (ram.md §7); depressions pass through negated."""

    @staticmethod
    def _env(altimetry):
        return Environment(name='surf', bathymetry=100.0, ssp=1500.0,
                           bottom=BoundaryProperties(
                               acoustic_type='half-space', sound_speed=1600.0,
                               density=1.5, attenuation=0.5),
                           altimetry=altimetry)

    def test_positive_crests_are_clamped_to_zero_with_a_warning(self):
        m = RAM(backend='ramsurf', verbose=False)
        env = self._env([(0.0, 0.0), (500.0, 1.5), (1000.0, -2.0),
                         (5000.0, 0.0)])
        with pytest.warns(UserWarning, match='clamped'):
            surface = m._build_ramsurf_surface(env, 5000.0)
        depths = dict(surface)
        assert depths[500.0] == pytest.approx(0.0)     # crest clamped
        assert depths[1000.0] == pytest.approx(2.0)    # trough negated
        assert all(z >= 0.0 for z in depths.values())

    def test_all_depression_profile_is_silent(self):
        m = RAM(backend='ramsurf', verbose=False)
        env = self._env([(0.0, 0.0), (1000.0, -2.0), (5000.0, 0.0)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            surface = m._build_ramsurf_surface(env, 5000.0)
        assert not [w for w in caught if 'clamped' in str(w.message)]
        assert dict(surface)[1000.0] == pytest.approx(2.0)

    def test_missing_altimetry_is_a_configuration_error(self):
        m = RAM(backend='ramsurf', verbose=False)
        env = Environment(name='flat', bathymetry=100.0, ssp=1500.0)
        with pytest.raises(ConfigurationError, match='altimetry'):
            m._build_ramsurf_surface(env, 5000.0)


class TestRangeDependentBottomIsHonoured:
    """The Collins decks carry one profile block per range segment, so a
    bottom that changes along the track must move the field relative to a
    uniform-bottom control marched on the identical pinned grid."""

    @staticmethod
    def _column(speed, rho, attn, hs_speed):
        return SeabedColumn(
            layers=[SedimentLayer(thickness=5.0, sound_speed=speed,
                                  density=rho, attenuation=attn)],
            halfspace=BoundaryProperties(
                acoustic_type='half-space', sound_speed=hs_speed,
                density=rho + 0.2, attenuation=attn))

    def _run(self, bottom):
        env = Environment(name='rd', bathymetry=100.0, ssp=1500.0,
                          bottom=bottom)
        return RAM(backend='ramgeo', dr=25.0, dz=1.0, verbose=False).run(
            env, Source(depths=25.0, frequencies=75.0),
            Receiver(depths=np.array([30.0, 60.0, 90.0]),
                     ranges=np.linspace(500.0, 4500.0, 9)),
            run_mode=RunMode.COHERENT_TL)

    def test_rd_bottom_field_differs_from_the_uniform_control(self):
        near = self._column(1550.0, 1.5, 0.1, 1600.0)
        far = self._column(1800.0, 2.1, 1.0, 2400.0)
        rd = np.asarray(self._run(Bottom.from_columns(
            [near, far], ranges=np.array([0.0, 2000.0]))).db)
        uniform = np.asarray(self._run(Bottom.from_columns(
            [near, near], ranges=np.array([0.0, 2000.0]))).db)
        ok = np.isfinite(rd) & np.isfinite(uniform)
        assert ok.any()
        # Beyond the 2 km transition the two seabeds must separate the TL.
        far_half = ok & (np.arange(rd.shape[1])[None, :] >= 4)
        assert np.max(np.abs(rd[far_half] - uniform[far_half])) > 1.0, (
            "a range-dependent bottom produced the same field as the "
            "uniform control — the deck's per-range segments were dropped")


def test_near_side_receivers_lie_within_the_march_exit_tolerance() -> None:
    """The premise the near-side ``np.clip(..., rout[0], None)`` rests on.

    ``rout[0]`` is the achieved range of ``receiver.ranges[0]``: the deck
    writer emits ``receiver.ranges`` verbatim and in order, and the carrier
    forbids a non-increasing range axis. So the nearest receiver is the first
    one written, and the march's ``if (abs(rnow-rend)<0.1_wp) exit`` bounds how
    far ``rout[0]`` can sit above it — the same 10 cm the far side snaps.
    Nothing can be silently dragged further than that, which is why the near
    side carries no tolerance of its own.
    """
    from uacpy.io.mpirams_writer import write_ranges_file
    from uacpy.models.ram import MPIRAMS_RANGE_TOL_M

    ranges = [500.0, 1500.0, 3000.0]
    receiver = Receiver(depths=[10.0, 20.0], ranges=ranges)
    stored = np.asarray(receiver.ranges, dtype=float)

    # The carrier forbids a range axis that is not strictly increasing, so
    # ranges[0] is the minimum.
    assert np.all(np.diff(stored) > 0)
    assert float(stored[0]) == float(stored.min())
    with pytest.raises(uacpy.ConfigurationError):
        Receiver(depths=[10.0], ranges=[3000.0, 500.0])

    # The deck carries those ranges through unchanged and in order.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ranges.dat"
        write_ranges_file(path, receiver.ranges)
        written = [float(line) for line in
                   path.read_text().split() if line.strip()]
    assert written == list(stored)

    # And the tolerance the far side uses is the march's own exit test, which
    # is what bounds the near side too.
    source = (Path(uacpy.__file__).parent / "third_party" / "mpiramS" /
              "src" / "ram.f90")
    if not source.is_file():
        pytest.skip("vendored mpiramS source not present")
    assert (f"abs(rnow-rend)<{MPIRAMS_RANGE_TOL_M}_wp"
            in source.read_text(encoding="utf-8", errors="replace"))


class TestMpiramsOutputRangeSpacing:
    """``rout(irr)`` is the position the march stopped at, not the range that
    was asked for, so two output ranges closer together than the march's exit
    tolerance share one entry. ``validate_inputs`` refuses that grid; these
    pin the threshold against the march itself, on both sides."""

    ENV = Environment(name='spacing', bathymetry=100.0, ssp=1500.0)
    SOURCE = Source(depths=50.0, frequencies=50.0)

    @staticmethod
    def _receiver(ranges):
        return Receiver(depths=[20.0, 40.0], ranges=ranges)

    @staticmethod
    def _march(ranges):
        """Return the ``rout`` the binary writes for ``ranges``."""
        import tempfile
        from uacpy.io.mpirams_reader import read_psif
        model = RAM(backend='mpiramS', verbose=False, dr=20.0, dz=2.0)
        receiver = Receiver(depths=[20.0], ranges=ranges)
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            model._write_mpirams_deck(
                TestMpiramsOutputRangeSpacing.ENV,
                TestMpiramsOutputRangeSpacing.SOURCE,
                receiver, work_dir, 50.0, 1e6, 1.0, 20.0, 2.0)
            model._run_binary(work_dir)
            return np.asarray(read_psif(work_dir)['rout'], dtype=float)

    @pytest.mark.parametrize('run_mode', [RunMode.COHERENT_TL,
                                          RunMode.BROADBAND])
    def test_a_pair_below_the_exit_tolerance_is_refused(self, run_mode):
        """The message carries the two ranges, their separation and the limit
        — the three numbers needed to respace the grid."""
        receiver = self._receiver([500.0, 1000.0, 1000.05, 1500.0])
        model = RAM(backend='mpiramS', verbose=False)
        with pytest.raises(ConfigurationError) as exc:
            model.validate_inputs(self.ENV, self.SOURCE, receiver,
                                  run_mode=run_mode)
        message = str(exc.value)
        assert '1000.05' in message
        assert '0.05 m apart' in message
        assert f'{MPIRAMS_RANGE_TOL_M} m' in message
        assert 'ranges' in exc.value.remediation

    def test_a_pair_at_the_exit_tolerance_is_accepted(self):
        ranges = [500.0, 1000.0, 1000.1, 1500.0]
        assert ranges[2] - ranges[1] >= MPIRAMS_RANGE_TOL_M
        RAM(backend='mpiramS', verbose=False).validate_inputs(
            self.ENV, self.SOURCE, self._receiver(ranges),
            run_mode=RunMode.COHERENT_TL)

    @pytest.mark.parametrize('base, collapses', [(1000.0, False),
                                                 (3000.0, True)])
    def test_the_guard_refuses_the_grids_the_march_collapses(self, base,
                                                             collapses):
        """Two grids both spelled ``base`` and ``base + 0.1`` sit on opposite
        sides of the tolerance once the addition is rounded to a double, and
        the march applies its exit test to those same doubles. The guard has
        to follow the arithmetic rather than the nominal spacing."""
        ranges = [base, base + 0.1]
        assert ((ranges[1] - ranges[0]) < MPIRAMS_RANGE_TOL_M) is collapses
        rout = self._march(ranges)
        assert bool(rout[1] - rout[0] <= 0.0) is collapses
        model = RAM(backend='mpiramS', verbose=False)
        if collapses:
            with pytest.raises(ConfigurationError):
                model.validate_inputs(self.ENV, self.SOURCE,
                                      self._receiver(ranges),
                                      run_mode=RunMode.COHERENT_TL)
        else:
            model.validate_inputs(self.ENV, self.SOURCE,
                                  self._receiver(ranges),
                                  run_mode=RunMode.COHERENT_TL)

    def test_a_collins_backend_accepts_the_grid_mpirams_refuses(self):
        """ramgeo marches its own uniform output grid and interpolates it onto
        receiver.ranges, so the tolerance is mpiramS's alone."""
        receiver = self._receiver([500.0, 1000.0, 1000.05, 1500.0])
        RAM(backend='ramgeo', verbose=False).validate_inputs(
            self.ENV, self.SOURCE, receiver, run_mode=RunMode.COHERENT_TL)


def _elastic_column(thickness, speed):
    return SeabedColumn(
        layers=[SedimentLayer(thickness=thickness, sound_speed=speed,
                              density=1.5, attenuation=0.5,
                              shear_speed=400.0, shear_attenuation=1.0)],
        halfspace=BoundaryProperties(
            acoustic_type='half-space', sound_speed=1900.0, density=2.0,
            attenuation=0.1, shear_speed=600.0, shear_attenuation=0.5),
    )


def _fluid_column(thickness, speed):
    return SeabedColumn(
        layers=[SedimentLayer(thickness=thickness, sound_speed=speed,
                              density=1.6, attenuation=0.4)],
        halfspace=BoundaryProperties(
            acoustic_type='half-space', sound_speed=1900.0, density=1.9,
            attenuation=0.2),
    )


def _env_range_dependent_elastic():
    """Range-dependent in all three of bathymetry, SSP and bottom.

    The bathymetry is deliberately non-monotonic so ``_bathy_anchor_ranges``
    emits many sections, and the layer thickness varies with range so the
    sediment base — where the absorbing ramp starts — moves between them.
    """
    ssp = SoundSpeedProfile(
        depths=np.array([0.0, 50.0, 200.0]),
        data=np.array([[1500.0, 1510.0],
                       [1490.0, 1495.0],
                       [1480.0, 1485.0]]),
        ranges=np.array([0.0, 6000.0]),
    )
    bottom = Bottom(
        columns=[_elastic_column(20.0, 1700.0), _elastic_column(8.0, 1750.0),
                 _elastic_column(30.0, 1650.0)],
        ranges=[0.0, 3000.0, 6000.0],
    )
    return Environment(
        name='rd-elastic',
        bathymetry=[(0.0, 120.0), (2000.0, 180.0), (4000.0, 90.0),
                    (6000.0, 200.0)],
        ssp=ssp, bottom=bottom,
    )


def _env_flat_fluid():
    return Environment(name='flat-fluid', bathymetry=100.0, ssp=1500.0,
                       bottom=_fluid_column(15.0, 1650.0))


def _env_slope_range_independent_bottom():
    """Sloping seafloor over a bottom that carries a single column."""
    return Environment(
        name='slope-ri', bathymetry=[(0.0, 60.0), (5000.0, 400.0)],
        ssp=1500.0, bottom=_fluid_column(4.0, 1600.0),
    )


def _env_slope_range_dependent_bottom():
    return Environment(
        name='slope-rd',
        bathymetry=[(0.0, 60.0), (2500.0, 300.0), (5000.0, 400.0)],
        ssp=1500.0,
        bottom=Bottom(columns=[_fluid_column(4.0, 1600.0),
                               _fluid_column(9.0, 1700.0)],
                      ranges=[0.0, 5000.0]),
    )


def _exact(segments):
    """A segment list rendered so ``==`` compares the doubles themselves.

    ``repr`` round-trips a float exactly and separates ``0.0`` from ``-0.0``,
    which plain equality does not.
    """
    return [{key: (repr(value) if not isinstance(value, list)
                   else [(repr(a), repr(b)) for a, b in value])
             for key, value in segment.items()}
            for segment in segments]


# 105 Hz is where this environment's absorbing ramp starts *inside* the
# sediment for some sections and at its base for others — see
# ``test_the_attenuation_block_changes_length_within_one_deck``.
_DECK_FREQS = (1.0, 10.0, 50.0, 105.0, 200.0, 800.0, 5000.0, 20000.0)


class TestHoistedCollinsDeck:
    """``_collins_deck_base`` + ``_ramp_range_segments`` must reproduce
    ``_collins_range_segments`` exactly at every frequency, because the
    broadband loop cuts the deck once for the whole band and re-ramps it."""

    @pytest.mark.parametrize('kind', ['ramgeo', 'ramsurf', 'rams'])
    @pytest.mark.parametrize('flat', [True, False])
    def test_one_base_survives_being_reramped_and_its_decks_mangled(
            self, kind, flat):
        """A sweep re-ramps one payload hundreds of times and hands each
        finished deck to a writer. Nothing that happens to a deck may reach
        the payload, so the frequencies are visited out of order and each
        deck is mutated before the next one is cut."""
        model = RAM(verbose=False)
        env = _env_flat_fluid() if flat else _env_range_dependent_elastic()
        zmax = 400.0 if flat else 500.0
        base = model._collins_deck_base(env, kind, zmax)
        for freq in sorted(_DECK_FREQS, reverse=True) + list(_DECK_FREQS):
            reramped = model._ramp_range_segments(base, freq, kind=kind,
                                                  zmax=zmax)
            rebuilt = model._collins_range_segments(env, kind, zmax, freq)
            assert _exact(reramped) == _exact(rebuilt), f"freq={freq}"
            for segment in reramped:
                for value in segment.values():
                    if isinstance(value, list):
                        value.clear()

    def test_the_deck_matches_the_values_recorded_before_the_hoist(self):
        """The independent pin: decks recorded from the single-stage builder
        the split replaced. Every other check here compares the two stages
        against a rebuild that now goes *through* them, so this is what
        stands between the deck and a shared drift."""
        model = RAM(verbose=False)

        flat = model._collins_range_segments(_env_flat_fluid(), 'rams',
                                             400.0, 800.0)
        assert _exact(flat) == [{
            'range': '0.0',
            'water_ssp': [('0.0', '1500.0'), ('100.0', '1500.0')],
            'bottom_c': [('100.0', '1650.0'), ('115.0', '1650.0'),
                         ('115.0', '1900.0'), ('400.0', '1900.0')],
            'bottom_rho': [('100.0', '1.6'), ('115.0', '1.6'),
                           ('115.0', '1.9'), ('400.0', '1.9')],
            'bottom_attn': [('100.0', '0.4'), ('115.0', '0.4'),
                            ('115.0', '0.2'),
                            ('355.8503444387448', '0.2'), ('400.0', '10.0')],
            'bottom_cs': [('100.0', '0.0'), ('115.0', '0.0'),
                          ('115.0', '0.0'), ('400.0', '0.0')],
            'bottom_attns': [('100.0', '0.0'), ('115.0', '0.0'),
                             ('115.0', '0.0'), ('400.0', '0.0')],
        }]

        sloped = model._collins_range_segments(
            _env_range_dependent_elastic(), 'ramgeo', 500.0, 50.0)
        assert _exact(sloped) == [
            {'range': '0.0',
             'water_ssp': [('0.0', '1500.0'), ('50.0', '1490.0'),
                           ('200.0', '1480.0')],
             'bottom_c': [('0.0', '1700.0'), ('20.0', '1700.0'),
                          ('20.0', '1900.0'), ('380.0', '1900.0')],
             'bottom_rho': [('0.0', '1.5'), ('20.0', '1.5'),
                            ('20.0', '2.0'), ('380.0', '2.0')],
             'bottom_attn': [('0.0', '0.5'), ('20.0', '0.5'),
                             ('20.0', '0.1'), ('380.0', '10.0')]},
            {'range': '1500.0',
             'water_ssp': [('0.0', '1505.0'), ('50.0', '1492.5'),
                           ('200.0', '1482.5')],
             'bottom_c': [('0.0', '1750.0'), ('8.0', '1750.0'),
                          ('8.0', '1900.0'), ('365.0', '1900.0')],
             'bottom_rho': [('0.0', '1.5'), ('8.0', '1.5'),
                            ('8.0', '2.0'), ('365.0', '2.0')],
             'bottom_attn': [('0.0', '0.5'), ('8.0', '0.5'),
                             ('8.0', '0.1'), ('365.0', '10.0')]},
            {'range': '4500.0',
             'water_ssp': [('0.0', '1510.0'), ('50.0', '1495.0'),
                           ('200.0', '1485.0')],
             'bottom_c': [('0.0', '1650.0'), ('30.0', '1650.0'),
                          ('30.0', '1900.0'), ('300.0', '1900.0')],
             'bottom_rho': [('0.0', '1.5'), ('30.0', '1.5'),
                            ('30.0', '2.0'), ('300.0', '2.0')],
             'bottom_attn': [('0.0', '0.5'), ('30.0', '0.5'),
                             ('30.0', '0.1'), ('300.0', '10.0')]},
        ]

        # The 62-section elastic deck, spot-checked at both ends: full text
        # here would be unreadable, and the ends are where the seafloor is
        # shallowest and deepest.
        wide = _exact(model._collins_range_segments(
            _env_range_dependent_elastic(), 'rams', 500.0, 200.0))
        assert len(wide) == 62
        assert wide[0]['range'] == '0.0'
        assert wide[-1]['range'] == '5973.6070381231675'
        assert wide[0]['bottom_attn'] == [
            ('120.0', '0.5'), ('140.0', '0.5'), ('140.0', '0.1'),
            ('324.8631220870285', '0.1'), ('500.0', '10.0')]
        assert wide[-1]['bottom_attn'] == [
            ('200.0', '0.5'), ('230.0', '0.5'), ('230.0', '0.1'),
            ('324.8631220870285', '0.1'), ('500.0', '10.0')]

    def test_the_attenuation_block_changes_length_with_frequency(self):
        """The block is not a fixed-shape array that could be scaled in
        place: the ramp keeps every control point at or above its start
        depth, and that count moves with ``absorbing_width ∝ 1/f``."""
        model = RAM(verbose=False)
        env = _env_range_dependent_elastic()
        lengths = {
            freq: {len(seg['bottom_attn'])
                   for seg in model._collins_range_segments(
                       env, 'rams', 500.0, freq)}
            for freq in (50.0, 800.0)
        }
        assert lengths[50.0] == {4}
        assert lengths[800.0] == {5}

    def test_the_attenuation_block_changes_length_within_one_deck(self):
        """Worse than a per-frequency length: at one frequency the sections
        of a single deck disagree, because the seafloor — and with it the
        sediment base the ramp is clamped to — moves with range."""
        model = RAM(verbose=False)
        segments = model._collins_range_segments(
            _env_range_dependent_elastic(), 'rams', 500.0, 105.0)
        lengths = [len(seg['bottom_attn']) for seg in segments]
        assert set(lengths) == {4, 5}
        assert min(lengths.count(4), lengths.count(5)) > 10

    @pytest.mark.parametrize('kind,zmax', [('ramgeo', 501.0), ('rams', 500.0)])
    def test_a_deck_cut_against_another_grid_is_refused(self, kind, zmax):
        """Condition the hoist rests on: ``zmax`` is fixed for the band.
        Nothing in the code forces that, so reuse is checked rather than
        assumed — a stale deck would march a different domain in silence."""
        model = RAM(verbose=False)
        base = model._collins_deck_base(
            _env_range_dependent_elastic(), 'ramsurf', 500.0)
        with pytest.raises(AssertionError, match='zmax'):
            model._ramp_range_segments(base, 100.0, kind=kind, zmax=zmax)

    def test_the_broadband_loop_owns_one_deck_and_keeps_it_off_the_model(self):
        """The deck is a local of the sweep, not a cache on ``self``: an
        attribute would outlive this ``env`` and be handed to the next
        ``run()``. Every bin must also get the *same* payload, at the
        band's own ``zmax``."""
        model = RAM(verbose=False, backend='ramgeo', dr=50.0, dz=0.5,
                    zmax=400.0, Q=4.0, T=0.2)
        env = _env_slope_range_dependent_bottom()
        source = Source(depths=[40.0], frequencies=[150.0])
        receiver = Receiver(depths=np.array([30.0, 90.0]),
                            ranges=np.array([500.0, 1500.0]))

        class _Stop(Exception):
            pass

        seen, builds = [], []
        before = set(vars(model))
        real_base = model._collins_deck_base

        def spy_base(*args, **kwargs):
            built = real_base(*args, **kwargs)
            builds.append(built)
            return built

        def stub_one_freq(*args, **kwargs):
            seen.append(kwargs['deck_base'])
            if len(seen) == 3:
                raise _Stop
            return dict(tl=np.zeros((2, 2)), pcomplex=np.zeros((2, 2), complex),
                        depths=np.array([30.0, 90.0]),
                        ranges=np.array([500.0, 1500.0]),
                        dr=50.0, dz=0.5, zmax=400.0)

        model._collins_deck_base = spy_base
        model._run_collins_one_freq = stub_one_freq
        with pytest.raises(_Stop):
            model.run(env, source, receiver, run_mode='broadband')

        assert len(builds) == 1
        assert all(deck is builds[0] for deck in seen)
        assert builds[0]['zmax'] == 400.0
        assert set(vars(model)) - before <= {'_collins_deck_base',
                                             '_run_collins_one_freq'}
        assert not any(value is builds[0] for value in vars(model).values())


class TestBathyAnchorRanges:

    @pytest.mark.parametrize('env_factory', [
        _env_slope_range_independent_bottom,
        _env_slope_range_dependent_bottom,
        _env_range_dependent_elastic,
    ])
    def test_the_vectorised_probe_matches_the_scalar_one_exactly(
            self, env_factory):
        """The fix replaced 1024 scalar ``eval`` calls with one vectorised
        call. ``_query_profile`` runs the same arithmetic either way, so the
        anchors must land on identical doubles — asserted against a scalar
        re-derivation of the whole method, not a tolerance."""
        env = env_factory()
        bottom = env.bottom
        r_axis = np.atleast_1d(np.asarray(env.bathymetry.ranges, dtype=float))
        r_end = float(np.max(r_axis))
        thicknesses = [
            float(layer.thickness)
            for r in r_axis
            for layer in bottom.at(range=float(r)).layers
            if float(layer.thickness) > 0.0
        ]
        tol = 0.5 * min(thicknesses)
        probe = np.linspace(0.0, r_end, 1024)
        floor = np.array(
            [float(np.asarray(env.bathymetry.eval(range=float(r))).flat[0])
             for r in probe])
        expected, anchor = [], floor[0]
        for r, z in zip(probe[1:], floor[1:]):
            if abs(z - anchor) >= tol:
                expected.append(float(r))
                anchor = z
        if len(expected) > MAX_BATHY_SECTIONS:
            idx = np.linspace(0, len(expected) - 1, MAX_BATHY_SECTIONS)
            expected = [expected[int(round(i))] for i in idx]

        got = RAM._bathy_anchor_ranges(env, bottom)
        assert [repr(v) for v in got] == [repr(v) for v in expected]

    def test_the_sloping_case_keeps_its_recorded_anchors(self):
        """A second, independent pin: values recorded from the scalar-probe
        implementation, so a change to *both* paths still fails."""
        env = _env_slope_range_independent_bottom()
        got = RAM._bathy_anchor_ranges(env, env.bottom)
        assert len(got) == 64
        assert repr(got[0]) == '34.21309872922776'
        assert repr(got[-1]) == '4995.112414467253'

    def test_a_single_column_bottom_answers_every_node_the_same(self):
        """What the copy de-duplication rests on: a bottom with one column
        has nothing to choose between, so every node resolves to it and the
        layer thicknesses cannot differ by range."""
        env = _env_slope_range_independent_bottom()
        bottom = env.bottom
        assert not bottom.is_range_dependent
        per_node = {tuple(float(layer.thickness)
                          for layer in bottom.at(range=float(r)).layers)
                    for r in np.linspace(0.0, 5000.0, 37)}
        assert len(per_node) == 1


class TestRamShearCapKeepsSeafloorAlignment:
    """``rams0.5`` places the seafloor at ``iz = int(zb/dz)`` (``:135``), a
    truncation with its cliff at integer ``zb/dz``, so ``dz`` has to divide
    the water depth. The shear-wavelength cap is a *bound* on dz, not a grid:
    assigning ``lambda_s/14`` raw gives ``h/dz = 233.33`` at c_s = 300 m/s,
    f = 50 Hz, h = 100 m. Measured against a seafloor-aligned h/2000
    reference: 9.08 dB max / 1.37 dB rms unsnapped, 0.81 / 0.13 snapped."""

    @staticmethod
    def _elastic_env(depth=100.0, shear=300.0):
        return Environment(
            name='elastic', bathymetry=depth, ssp=1500.0,
            bottom=BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0, density=1.8,
                attenuation=0.5, shear_speed=shear, shear_attenuation=0.2))

    @pytest.mark.parametrize('freq', [50.0, 100.0, 250.0])
    def test_auto_grid_puts_the_seafloor_on_a_node(self, freq):
        from uacpy.models._pade_optimizer import rams_dz_shear_cap
        env = self._elastic_env()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, dz = RAM(backend='rams', verbose=False)._compute_grid_lytaev(
                env, freq, max_range=5000.0, kind='rams')
        layers = 100.0 / dz
        assert layers == pytest.approx(round(layers), abs=1e-6), (
            f"h/dz = {layers} is not an integer, so the seafloor falls "
            f"between depth nodes")
        assert dz <= rams_dz_shear_cap(300.0, freq) * (1 + 1e-12), (
            "the aligned dz must still resolve the shear wavelength")

    def test_alignment_helper_respects_the_bound_direction(self):
        model = RAM(backend='rams', verbose=False)
        env = self._elastic_env()
        tighter = model._align_dz_with_seafloor(env, 0.428571)
        coarser = model._align_dz_with_seafloor(env, 0.428571, coarsen=True)
        assert tighter <= 0.428571 < coarser
        for dz in (tighter, coarser):
            assert 100.0 / dz == pytest.approx(round(100.0 / dz), abs=1e-6)

    def test_an_already_aligned_dz_is_left_where_it_is(self):
        model = RAM(backend='rams', verbose=False)
        env = self._elastic_env()
        aligned = RAM._snap_dz_to_seafloor(100.0, 250)
        assert model._align_dz_with_seafloor(env, aligned) == pytest.approx(
            aligned, rel=1e-9)


class TestRamBackendBinaryResolution:
    """``RAM(backend='rams')`` never executes ``s_mpiram``, so resolving it at
    construction refuses a configuration that runs; and a pinned
    ``executable`` has to reach whichever binary the dispatch lands on."""

    @pytest.mark.parametrize('backend,expected', [
        ('rams', 'rams0.5'), ('ramsurf', 'ramsurf1.5'), ('ramgeo', 'ramgeo'),
        ('mpiramS', 's_mpiram'), (None, 's_mpiram'),
    ])
    def test_the_resolved_binary_follows_the_backend(self, backend, expected):
        assert RAM(backend=backend, verbose=False)._exe.name == expected

    def test_an_unknown_backend_is_refused_before_any_lookup(self):
        with pytest.raises(ConfigurationError, match='not a known backend'):
            RAM(backend='definitely-not-a-backend', verbose=False)

    def test_a_pinned_executable_reaches_the_collins_path(self):
        exe = RAM(backend='rams', verbose=False)._exe
        model = RAM(backend='rams', executable=exe, verbose=False)
        assert model._collins_binary('rams') == exe
        assert model._collins_binary('ramgeo') == exe

    def test_a_pinned_missing_executable_is_named(self):
        from uacpy.core.exceptions import ExecutableNotFoundError
        with pytest.raises(ExecutableNotFoundError):
            RAM(backend='rams', executable='/no/such/binary', verbose=False)


def test_ram_source_row_guard_runs_before_the_deck_is_written(tmp_path):
    """``_check_source_row_is_solved`` refuses a source shallower than one
    depth cell; on the mpiramS path it used to fire after ssp.dat, the
    bathymetry file, the sediment file and ranges.dat were already on disk."""
    env = Environment(
        name='shallow_source', bathymetry=100.0, ssp=1500.0,
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1600.0, density=1.6,
                                  attenuation=0.5))
    source = Source(depths=0.05, frequencies=50.0)
    receiver = Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))
    model = RAM(backend='mpiramS', verbose=False)
    with pytest.raises(ConfigurationError, match='shallower than one depth'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model._write_mpirams_deck(env, source, receiver, tmp_path,
                                      50.0, 0.0, 0.0, dr=10.0, dz=1.0)
    assert list(tmp_path.iterdir()) == [], (
        "the refusal left a half-written deck behind")


class TestRamHalfSpaceSedimentLayerIsFloored:
    """A bare half-space contributes no layer thickness of its own, so without
    a floor ``sedlayer`` collapses to one depth cell and the attenuation ramp
    starts at the seabed itself — replacing the seabed's own attenuation over
    the whole sub-bottom, which ``_absorber_span`` exists to prevent. The floor
    is the same synthetic thickness the Collins path wraps a half-space in
    (``SeabedColumn.from_halfspace``: 10 % of the water depth, at least 5 m),
    so the ramp begins below genuinely modelled seabed.
    """

    @staticmethod
    def _halfspace_env(depth):
        return Environment(
            bathymetry=depth, ssp=1500.0,
            bottom=BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0, density=1.8,
                attenuation=0.5, shear_speed=800.0, shear_attenuation=0.2))

    @pytest.mark.parametrize('depth, expected', [(100.0, 10.0), (30.0, 5.0)])
    def test_the_floor_follows_the_synthetic_layer_thickness(self, depth,
                                                             expected,
                                                             tmp_path):
        # 100 m -> 10 % of the water depth; 30 m -> the 5 m minimum, since
        # 10 % of 30 m is under it.
        sedlayer = RAM(backend='rams', verbose=False)._prepare_bottom_properties(
            self._halfspace_env(depth), tmp_path,
            1.0, depth * 2.0)[0]
        assert sedlayer == pytest.approx(expected)

    def test_a_deeper_absorber_span_sets_the_thickness(self, tmp_path):
        # The floor is a minimum, not an override: every builder stretches
        # sedlayer to the absorbing layer's start.
        sedlayer = RAM(backend='rams', verbose=False)._prepare_bottom_properties(
            self._halfspace_env(100.0), tmp_path,
            250.0, 200.0)[0]
        assert sedlayer == pytest.approx(250.0)


class TestRamSectionSpacingWarnsOnlyForACallersOwnDr:
    """``dr`` is always bounded by the closest profile-section spacing, because
    ``profl`` reads sequentially and one section per ``dr`` (``ramgeo1.5.f:195``,
    ``:359``, ``:78-84``) — sections closer together than ``dr`` are never
    reached, and the run still exits 0 on the truncated environment.

    The ``pinned`` flag decides only whether the caller is TOLD. It has to
    track whose ``dr`` it is: the Collins broadband loop passes one ``dr``
    override for the whole band, so keying the warning off the override's mere
    presence announced an adjustment to a value the caller never chose.
    """

    SEGMENTS = [{'range': r} for r in (0.0, 100.0, 200.0, 300.0)]

    def test_an_auto_derived_dr_is_bounded_silently(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            dr = RAM(backend='ramgeo', verbose=False)._constrain_dr_to_sections(
                250.0, self.SEGMENTS, pinned=False)
        assert dr == pytest.approx(100.0)

    def test_a_callers_own_dr_is_bounded_and_announced(self):
        with pytest.warns(UserWarning, match='profile-section spacing'):
            dr = RAM(backend='ramgeo', verbose=False)._constrain_dr_to_sections(
                250.0, self.SEGMENTS, pinned=True)
        assert dr == pytest.approx(100.0)      # bounded either way

    def test_a_dr_already_inside_the_spacing_is_left_alone(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            dr = RAM(backend='ramgeo', verbose=False)._constrain_dr_to_sections(
                50.0, self.SEGMENTS, pinned=True)
        assert dr == pytest.approx(50.0)


class TestCollinsBroadbandSizesItsRangeStepAtTheBandTop:
    """A broadband march runs one grid for the whole band, so the step has to
    resolve the SMALLEST wavelength in it. Sizing ``dr`` at ``f_min`` leaves it
    ``f_max/f_min`` times too coarse at the top: measured on a 100 m Pekeris
    guide (1700/1.7/0.5) against Scooter at 500 Hz, ramgeo with ``dr`` from
    ``f_min = 100 Hz`` (54.05 m) is 8.85 dB rms / 16.86 dB max out, against
    1.64 / 5.24 with ``dr`` from ``f_max`` (12.81 m).

    ``rams`` keeps ``f_min`` deliberately, and that is not the same question:
    its rotated-Pade elastic march is only marginally stable, so the extra
    range steps a finer ``dr`` implies inject a spurious acausal precursor into
    the broadband synthesis. The stability argument is specific to the elastic
    march; applying it to the fluid backends only under-resolved them.
    """

    @staticmethod
    def _env():
        from uacpy.core import BoundaryProperties, Environment
        return Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.7,
                                      attenuation=0.5))

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_the_broadband_top_bin_matches_a_resolved_narrowband_run(self):
        """The behaviour the sizing exists for: the top bin of a BROADBAND
        march must agree with a narrowband run at that same frequency. With dr
        sized at f_min the broadband bin was marching a step 4x too coarse for
        it, and the two disagreed by 7 dB rms.
        """
        import numpy as np
        from uacpy.core import Receiver, Source
        from uacpy.models import RAM
        from uacpy.models.base import RunMode
        env = self._env()
        ranges = np.linspace(500.0, 5000.0, 19)
        rcv = Receiver(depths=[50.0], ranges=ranges)
        band = np.linspace(100.0, 500.0, 9)
        wide = RAM(backend='ramgeo', verbose=False).run(
            env, Source(depths=25.0, frequencies=band), rcv,
            run_mode=RunMode.BROADBAND)
        db = np.asarray(wide.db, dtype=float)
        top = db[..., -1].ravel() if db.ndim > 1 else db.ravel()
        narrow = np.asarray(RAM(backend='ramgeo', verbose=False).run(
            env, Source(depths=25.0, frequencies=500.0), rcv).db,
            dtype=float).ravel()
        rms = float(np.sqrt(np.nanmean((top - narrow) ** 2)))
        assert rms < 2.0, f"broadband top bin is {rms:.2f} dB rms from the " \
                          f"narrowband run at the same frequency"


class TestRamWarnsWhenReceiverRangesStartAtZero:
    """RAM writes no r=0 column on any backend: the self-starter sets ``r=dr``
    before its only ``outpt`` call (``ramgeo1.5.f:124,172``) and the march loop
    advances ``r=r+dr`` at ``:82`` BEFORE ``outpt`` at ``:83``.

    The Collins near-range guard filters ``near[near > 0.0]`` — deliberately,
    so its "pin a smaller dr" advice stays honest, since that advice is exactly
    what cannot fix zero — so r=0 fell through every check and returned bare
    NaN. Bellhop warns on identical geometry, so ``np.linspace(0, R, N)`` cost
    a silent column on one engine and a warning on another.
    """

    @staticmethod
    def _env():
        from uacpy.core import BoundaryProperties, Environment
        return Environment(bathymetry=100.0, ssp=1500.0,
                           bottom=BoundaryProperties(
                               acoustic_type='half-space', sound_speed=1600.0,
                               density=1.5, attenuation=0.5))

    @staticmethod
    def _zero_range_warnings(record):
        return [w for w in record if 'r=0 m' in str(w.message)]

    def test_a_zero_first_range_warns_and_the_column_is_nan(self):
        import warnings
        import numpy as np
        from uacpy.core import Receiver, Source
        from uacpy.models import RAM
        rcv = Receiver(depths=[50.0], ranges=np.array([0.0, 500.0, 1000.0]))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            field = RAM(verbose=False).run(
                self._env(), Source(depths=25.0, frequencies=100.0), rcv)
        assert len(self._zero_range_warnings(rec)) == 1
        tl = np.asarray(field.to_db().db, dtype=float).ravel()
        assert np.isnan(tl[0]) and np.isfinite(tl[1:]).all()

    def test_a_positive_first_range_is_silent(self):
        import warnings
        import numpy as np
        from uacpy.core import Receiver, Source
        from uacpy.models import RAM
        rcv = Receiver(depths=[50.0], ranges=np.array([500.0, 1000.0]))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            RAM(verbose=False).run(
                self._env(), Source(depths=25.0, frequencies=100.0), rcv)
        assert self._zero_range_warnings(rec) == []

    def test_a_broadband_sweep_warns_once_not_once_per_frequency(self):
        # The check sits ahead of the backend dispatch precisely so the
        # frequency loop does not repeat it.
        import warnings
        import numpy as np
        from uacpy.core import Receiver, Source
        from uacpy.models import RAM
        from uacpy.models.base import RunMode
        rcv = Receiver(depths=[50.0], ranges=np.array([0.0, 500.0, 1000.0]))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            RAM(verbose=False).run(
                self._env(),
                Source(depths=25.0, frequencies=[80.0, 100.0, 120.0]),
                rcv, run_mode=RunMode.BROADBAND)
        assert len(self._zero_range_warnings(rec)) == 1

    def test_the_collins_family_warns_on_the_same_geometry(self):
        # mpiramS and the Collins backends reach the guard by different
        # dispatch arms; one wording covers both.
        import warnings
        import numpy as np
        from uacpy.core import Receiver, Source
        from uacpy.models import RAM
        rcv = Receiver(depths=[50.0], ranges=np.array([0.0, 500.0, 1000.0]))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            RAM(backend='ramgeo', verbose=False).run(
                self._env(), Source(depths=25.0, frequencies=100.0), rcv)
        hits = self._zero_range_warnings(rec)
        assert len(hits) == 1
        assert 'ramgeo' in str(hits[0].message)


class TestRamLeavesRealSeabedBeforeTheAbsorber:
    """``_absorber_span`` puts the artificial attenuation ramp over the deepest
    ``absorbing_layer_width`` wavelengths, so the NON-absorbing sub-bottom the
    automatic grid leaves is ``(zmax - depth) - absorbing_width``. With the old
    ``depth + dz + absorbing_width`` that was exactly ``dz`` — 1.0 m at 25 Hz,
    0.31 m at 300 Hz — i.e. the ramp began one cell below the seafloor, which
    is the very thing ``_absorber_span``'s own docstring warns against.

    Measured against Kraken on a 200 m guide (source 30 m, receiver 150 m,
    1-20 km), comparing RANGE-SMOOTHED levels so interference fringes cannot be
    mistaken for a level error: at 25 Hz over a lossless seabed the old grid
    was biased +2.24 dB, against +0.07 dB on a converged deep grid. Two bottom
    wavelengths of real seabed reach +0.12 dB; one is not enough (+0.54 dB) and
    four is indistinguishable from converged. A lossy seabed absorbs the tails
    itself and hides most of it (+0.14 dB at 0.5 dB/lambda), and by 50 Hz the
    effect is under 0.04 dB at every attenuation.
    """

    @staticmethod
    def _half_space(att=0.0):
        import uacpy
        return uacpy.Environment(
            bathymetry=200.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0,
                density=1.8, attenuation=att))

    @staticmethod
    def _layered():
        from uacpy.core import BoundaryProperties, Environment
        from uacpy.core.environment import SeabedColumn, SedimentLayer
        return Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=SeabedColumn(
                layers=[SedimentLayer(thickness=5.0, sound_speed=1520.0,
                                      density=1.3, attenuation=0.2)],
                halfspace=BoundaryProperties(
                    acoustic_type='half-space', sound_speed=2800.0,
                    density=2.5, attenuation=0.1)))

    def _real_seabed(self, model, env, freq):
        c0 = model._resolve_c0(env)
        absorbing = model.absorbing_layer_width * c0 / freq
        return (model._adequate_zmax(env, freq) - env.depth) - absorbing

    def test_the_span_is_two_bottom_wavelengths_not_one_cell(self):
        from uacpy.models import RAM
        model = RAM(verbose=False)
        for freq in (25.0, 100.0, 300.0):
            got = self._real_seabed(model, self._half_space(), freq)
            assert got == pytest.approx(2.0 * 1800.0 / freq, rel=1e-6)

    def test_the_span_scales_with_the_seabed_speed_not_the_water(self):
        # A faster seabed has a longer wavelength and so needs a thicker pad;
        # reading a 'sound_speed' attribute off Bottom (which has none) would
        # silently size it by the water column instead.
        from uacpy.models import RAM
        model = RAM(verbose=False)
        assert model._seabed_sound_speed(self._half_space(), 1500.0) == \
            pytest.approx(1800.0)

    def test_a_layered_bottom_keeps_the_tight_span(self):
        # The same quantity sets the span mpiramS distributes a FIXED number
        # of sediment control points over, so widening it on a modelled stack
        # smears the layer contrasts the deck exists to represent.
        from uacpy.models import RAM
        model = RAM(verbose=False)
        env = self._layered()
        got = self._real_seabed(model, env, 100.0)
        assert got < 2.0 * 2800.0 / 100.0
        assert got == pytest.approx(model._compute_dz(env, 100.0,
                                                      model._resolve_c0(env)))


def _mpirams_model():
    try:
        return uacpy.RAM(backend='mpiramS', verbose=False)
    except ExecutableNotFoundError:
        pytest.skip("mpiramS binary not installed")


def _assemble(model, rout, ranges, tmp_path):
    """Drive ``_assemble_tl_field`` over a synthesised mpiramS result whose
    ``rout`` is whatever the caller wants — including shorter than the ranges
    that were requested, which is what a march that stopped early would
    report."""
    env = uacpy.Environment(bathymetry=200.0, ssp=1500.0)
    source = uacpy.Source(depths=50.0, frequencies=100.0)
    receiver = uacpy.Receiver(depths=np.array([25.0, 75.0]),
                              ranges=np.asarray(ranges, dtype=float))
    zg = np.linspace(0.0, 400.0, 81)
    rout = np.asarray(rout, dtype=float)
    result = {
        'psif': np.full((zg.size, 1, rout.size), 1e-3, dtype=np.complex128),
        'zg': zg,
        'rout': rout,
        'frq': np.array([100.0]),
    }
    fm = types.SimpleNamespace(work_dir=Path(tmp_path))
    return model._assemble_tl_field(
        result, env, source, receiver, fm, 100.0, 10.0, 5.0, 0.0)


@pytest.mark.requires_binary
class TestMpiramsRangesBeyondTheMarchAreNotRelabelled:
    """``rout`` is the receiver grid handed back (uacpy writes
    ``receiver.ranges`` into ``ranges.dat``), so on a completed march the two
    agree. When they do not, the old ``max(1e-6, 0.5·receiver spacing)``
    tolerance was wide enough to clip an unmarched receiver onto ``rout[-1]``
    and return that position's field under the requested range's label."""

    def test_a_short_rout_leaves_the_unmarched_columns_nan(self, tmp_path):
        model = _mpirams_model()
        with pytest.warns(UserWarning, match='exceed the PE marched range'):
            field = _assemble(model, rout=[1000.0, 2000.0],
                              ranges=[1000.0, 2000.0, 3000.0],
                              tmp_path=tmp_path)
        assert np.all(np.isfinite(field.data[:, :2]))
        assert np.all(np.isnan(field.data[:, 2]))

    def test_the_unmarched_column_is_not_a_copy_of_the_last_marched_one(
            self, tmp_path):
        """The substitution this pin exists for: silently equal columns are
        exactly what the old clip produced."""
        model = _mpirams_model()
        with pytest.warns(UserWarning):
            field = _assemble(model, rout=[1000.0, 2000.0],
                              ranges=[1000.0, 2000.0, 3000.0],
                              tmp_path=tmp_path)
        assert not np.array_equal(field.data[:, 1], field.data[:, 2])

    def test_a_range_inside_the_march_exit_band_is_interpolated(
            self, tmp_path):
        """``ram.f90:169`` stops stepping once it is within 10 cm of an output
        range and records that position, so a completed march can report
        ``rout[-1]`` up to ``MPIRAMS_RANGE_TOL_M`` short. That is the same
        range under a rounding, and must not be dropped."""
        model = _mpirams_model()
        last = 3000.0 - MPIRAMS_RANGE_TOL_M
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            field = _assemble(model, rout=[1000.0, 2000.0, last],
                              ranges=[1000.0, 2000.0, 3000.0],
                              tmp_path=tmp_path)
        assert np.all(np.isfinite(field.data))

    def test_a_range_just_past_the_exit_band_is_dropped(self, tmp_path):
        """The other side of the same threshold, 1 µm further out."""
        model = _mpirams_model()
        last = 3000.0 - MPIRAMS_RANGE_TOL_M - 1e-6
        with pytest.warns(UserWarning, match='exceed the PE marched range'):
            field = _assemble(model, rout=[1000.0, 2000.0, last],
                              ranges=[1000.0, 2000.0, 3000.0],
                              tmp_path=tmp_path)
        assert np.all(np.isnan(field.data[:, 2]))

    def test_the_tolerance_does_not_scale_with_the_receiver_spacing(
            self, tmp_path):
        """The replaced tolerance was half the receiver spacing. With 20 km
        between receivers it reached 10 km, so a march that stopped 3 km short
        was reported as reached."""
        model = _mpirams_model()
        with pytest.warns(UserWarning, match='exceed the PE marched range'):
            field = _assemble(model, rout=[20000.0, 40000.0],
                              ranges=[20000.0, 40000.0, 60000.0],
                              tmp_path=tmp_path)
        assert np.all(np.isnan(field.data[:, 2]))

    def test_a_complete_march_is_silent_and_finite_on_every_column(self, tmp_path):
        """The identity case, which is what every real mpiramS run does:
        ``rout`` is ``receiver.ranges``, no warning, every column finite."""
        model = _mpirams_model()
        ranges = [1000.0, 1731.7, 4000.0, 9999.5]
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            field = _assemble(model, rout=ranges, ranges=ranges,
                              tmp_path=tmp_path)
        assert np.all(np.isfinite(field.data))

    def test_the_tolerance_constant_matches_the_fortran_exit_test(self):
        """``MPIRAMS_RANGE_TOL_M`` is not a tuned number: it is the march's own
        ``if (abs(rnow-rend)<0.1_wp) exit``."""
        src = (Path(uacpy.__file__).parent / 'third_party' / 'mpiramS' /
               'src' / 'ram.f90')
        if not src.is_file():
            pytest.skip("vendored mpiramS source not present")
        text = src.read_text(encoding='utf-8', errors='replace')
        assert f'abs(rnow-rend)<{MPIRAMS_RANGE_TOL_M}_wp' in text


class TestRsStabilityIsInertOnAMultiRangeOutputGrid:
    """``rs_stability`` names an absolute range; on mpiramS it acts as one
    only when the receiver carries a single range.

    ``mpiramS/src/ram.f90:68`` sets ``rsc = |rg(nr)| - rs`` once, and
    ``:251`` tests ``abs(rend - rnow) < rsc`` where ``:166`` has reassigned
    ``rend`` to the *current* output range. The Collins codes test the
    absolute range instead (``ramgeo/ramgeo1.5.f:368``: ``if(r.ge.rs)``).
    """

    RANGES = list(np.linspace(500.0, 20000.0, 40))

    @staticmethod
    def _env():
        return Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(sound_speed=1700.0, density=1.7,
                                      attenuation=0.5))

    @staticmethod
    def _run(ranges, rs):
        kw = dict(backend='mpiramS', verbose=False)
        if rs is not None:
            kw['rs_stability'] = rs
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            field = RAM(**kw).run(
                TestRsStabilityIsInertOnAMultiRangeOutputGrid._env(),
                Source(depths=25.0, frequencies=100.0),
                Receiver(depths=[20.0, 50.0, 80.0], ranges=ranges))
        return np.asarray(field.data)

    def test_a_pinned_rs_stability_warns_on_a_multi_range_grid(self):
        with pytest.warns(UserWarning, match='rs_stability'):
            RAM(backend='mpiramS', rs_stability=10000.0, verbose=False).run(
                self._env(), Source(depths=25.0, frequencies=100.0),
                Receiver(depths=[50.0], ranges=[500.0, 20000.0]))

    def test_two_output_ranges_are_already_enough_to_warn(self):
        """The low side of the grid-size threshold."""
        model = RAM(backend='mpiramS', rs_stability=10000.0, verbose=False)
        with pytest.warns(UserWarning, match='rs_stability'):
            model._warn_rs_stability_inert_on_a_multi_range_grid(
                Receiver(depths=[50.0], ranges=[500.0, 20000.0]))

    def test_a_single_output_range_does_not_warn(self):
        """The other side: with one output range ``rend`` never moves and the
        test reduces to the absolute one the parameter name promises."""
        model = RAM(backend='mpiramS', rs_stability=10000.0, verbose=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model._warn_rs_stability_inert_on_a_multi_range_grid(
                Receiver(depths=[50.0], ranges=[20000.0]))
        assert not [w for w in caught if 'rs_stability' in str(w.message)]

    def test_an_unpinned_rs_stability_does_not_warn(self):
        model = RAM(backend='mpiramS', verbose=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model._warn_rs_stability_inert_on_a_multi_range_grid(
                Receiver(depths=[50.0], ranges=self.RANGES))
        assert not [w for w in caught if 'rs_stability' in str(w.message)]

    @pytest.mark.slow
    def test_the_measurement_the_warning_reports(self):
        """Two ``rs_stability`` values 9 km apart give bit-identical output on
        a 40-range grid and different output on a single-range one — the
        second half is what shows the observable is live rather than the
        fixture being blind to ``rs_stability`` altogether."""
        multi = [self._run(self.RANGES, rs) for rs in (10000.0, 19000.0)]
        assert np.array_equal(multi[0], multi[1])
        single = [self._run([20000.0], rs) for rs in (10000.0, 19000.0)]
        assert not np.array_equal(single[0], single[1])


class TestRamOutputDomainStopsAtTheSeafloor:
    """RAM's output domain is the water column, and one number expresses it.

    ``_mask_below_seafloor`` returns NaN below the seafloor although the PE
    marches through the sediment and computes a field there — measured on this
    fixture (mask neutralised at runtime): |p| at 120 m is 5.4e-4 on mpiramS
    and 6.8e-4 on ramgeo, against Kraken 6.3e-4 and Scooter 6.2e-4. The same
    depth gates source placement, so a source in the sediment is refused on
    RAM alone. These pin the three halves of that: the depth, the explanation
    the refusal carries, and the warning that must keep accompanying the
    receiver NaNs.
    """

    @staticmethod
    def _layered_env():
        return Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=Bottom(columns=[SeabedColumn(
                layers=[SedimentLayer(thickness=50.0, sound_speed=1600.0,
                                      density=1.6, attenuation=0.2)],
                halfspace=BoundaryProperties(sound_speed=1800.0, density=2.0,
                                             attenuation=0.5))]))

    @staticmethod
    def _flat_env():
        return Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(sound_speed=1700.0, density=1.7,
                                      attenuation=0.5))

    def test_the_cut_is_the_seafloor_not_the_media_column(self):
        model = RAM(verbose=False)
        env = self._layered_env()
        assert model._max_receiver_depth(env) == 100.0
        assert model._total_media_depth(env) == 150.0

    def test_ram_declares_the_cut_itself(self):
        """Not inherited: RAM declares ``layered_bottom``, so a rule derived
        from that flag would return the media column and move the cut without
        anything failing."""
        assert '_max_receiver_depth' in vars(RAM)

    def test_a_buried_source_is_refused_with_the_reason(self):
        from uacpy.core.exceptions import InvalidDepthError
        with pytest.raises(InvalidDepthError) as exc:
            RAM(verbose=False).validate_inputs(
                self._layered_env(),
                Source(depths=120.0, frequencies=100.0),
                Receiver(depths=[50.0], ranges=[2000.0]))
        text = str(exc.value)
        assert 'output convention' in text
        assert 'not something the solver cannot compute' in text
        assert 'Kraken, Scooter, SPARC' in text

    def test_a_source_below_a_bottom_with_no_sediment_gets_no_extra_note(self):
        """The other side of the note's own condition: with nothing meshed
        below the seafloor there is no sediment column to name."""
        from uacpy.core.exceptions import InvalidDepthError
        with pytest.raises(InvalidDepthError) as exc:
            RAM(verbose=False).validate_inputs(
                self._flat_env(),
                Source(depths=120.0, frequencies=100.0),
                Receiver(depths=[50.0], ranges=[2000.0]))
        assert 'output convention' not in str(exc.value)

    def test_a_below_seafloor_receiver_warns_and_comes_back_nan(self):
        """The NaN below the seafloor stays a *warned* NaN. Widening the cut
        to the media column would accept the buried source and silence this
        warning while ``_mask_below_seafloor`` kept returning NaN — trading a
        warned NaN for a silent one."""
        model = RAM(verbose=False)
        with pytest.warns(UserWarning, match='receiver depth'):
            field = model.run(
                self._layered_env(),
                Source(depths=25.0, frequencies=100.0),
                Receiver(depths=[50.0, 120.0], ranges=[2000.0]))
        data = np.asarray(field.data)
        assert np.isfinite(data[0]).all(), (
            "the water-column receiver must be finite, or this fixture cannot "
            "tell a masked sample from a failed run")
        assert np.isnan(data[1]).all()


def _column(thickness, speed):
    return SeabedColumn(
        layers=[SedimentLayer(thickness=thickness, sound_speed=speed,
                              density=1.5, attenuation=0.5,
                              shear_speed=400.0, shear_attenuation=1.0),
                SedimentLayer(thickness=2.0 * thickness, sound_speed=speed + 50,
                              density=1.7, attenuation=0.6)],
        halfspace=BoundaryProperties(
            acoustic_type='half-space', sound_speed=1900.0, density=2.0,
            attenuation=0.1, shear_speed=600.0, shear_attenuation=0.5),
    )


def _wide_range_dependent_env(n_columns=24, n_bathy=97, r_end=60000.0):
    """A bottom with many columns under a wavy seafloor with many nodes.

    The two axes deliberately do not line up, so most bathymetry nodes fall
    between bottom columns and the nearest-column rule actually has to choose.
    """
    ranges = np.linspace(0.0, r_end, n_columns)
    bottom = Bottom(
        columns=[_column(10.0 + 5.0 * (i % 7), 1650.0 + 10.0 * (i % 11))
                 for i in range(n_columns)],
        ranges=ranges)
    r_bathy = np.linspace(0.0, r_end, n_bathy)
    z_bathy = 120.0 + 60.0 * np.sin(r_bathy / r_end * 6.0 * np.pi)
    return Environment(name='wide-rd',
                       bathymetry=list(zip(r_bathy.tolist(), z_bathy.tolist())),
                       ssp=1500.0, bottom=bottom)


def _range_independent_env():
    return Environment(name='ri', bathymetry=[(0.0, 60.0), (5000.0, 400.0)],
                       ssp=1500.0, bottom=_column(4.0, 1600.0))


def _anchor_ranges_via_copies(env, bottom):
    """``_bathy_anchor_ranges`` as it read the bottom before the fix.

    Kept verbatim rather than parameterised: the point of the comparison is
    that the copying read and the live read produce the *same doubles*, and a
    shared helper could only prove they share a helper.
    """
    r_axis = np.atleast_1d(np.asarray(env.bathymetry.ranges, dtype=float))
    r_end = float(np.max(r_axis))
    if not r_end > 0.0:
        return []
    columns = ([bottom.at(range=float(r)) for r in r_axis]
               if bottom.is_range_dependent
               else [bottom.at(range=float(r_axis[0]))])
    thicknesses = [
        float(layer.thickness)
        for col in columns
        for layer in col.layers
        if float(layer.thickness) > 0.0
    ]
    if not thicknesses:
        return []
    tol = 0.5 * min(thicknesses)
    probe = np.linspace(0.0, r_end, 1024)
    floor = np.asarray(env.bathymetry.eval(range=probe), dtype=float)
    out, anchor = [], floor[0]
    for r, z in zip(probe[1:], floor[1:]):
        if abs(z - anchor) >= tol:
            out.append(float(r))
            anchor = z
    if len(out) > MAX_BATHY_SECTIONS:
        idx = np.linspace(0, len(out) - 1, MAX_BATHY_SECTIONS)
        out = [out[int(round(i))] for i in idx]
    return out


class TestBathyAnchorRangesReadsWithoutCopying:

    @pytest.mark.parametrize('env_factory',
                             [_wide_range_dependent_env, _range_independent_env],
                             ids=['range-dependent', 'range-independent'])
    def test_it_produces_the_doubles_the_copying_read_produced(self,
                                                               env_factory):
        env = env_factory()
        got = RAM._bathy_anchor_ranges(env, env.bottom)
        assert got == _anchor_ranges_via_copies(env, env.bottom)

    @pytest.mark.parametrize('env_factory',
                             [_wide_range_dependent_env, _range_independent_env],
                             ids=['range-dependent', 'range-independent'])
    def test_it_never_reaches_the_copying_accessor(self, env_factory,
                                                   monkeypatch):
        # ``Bottom.at`` is the deep copy. Making it explode is the sharpest
        # statement that the hot loop does not call it — and the assertion
        # that fails the moment such a call appears.
        env = env_factory()

        def explode(self, **kwargs):
            raise AssertionError("_bathy_anchor_ranges called Bottom.at")

        monkeypatch.setattr(Bottom, 'at', explode)
        assert RAM._bathy_anchor_ranges(env, env.bottom) == \
            pytest.approx(_ANCHORS_WIDE if env.name == 'wide-rd'
                          else _ANCHORS_RI)

    def test_a_bottom_with_no_sediment_yields_no_anchors(self):
        env = Environment(
            name='halfspace', bathymetry=[(0.0, 60.0), (5000.0, 400.0)],
            ssp=1500.0,
            bottom=SeabedColumn(layers=[], halfspace=BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0, density=1.9,
                attenuation=0.2)))
        assert RAM._bathy_anchor_ranges(env, env.bottom) == []


# The anchor ranges the two fixtures produce, captured once through the
# copying reference implementation so the monkeypatched test above has
# something to compare against that ``Bottom.at`` did not compute.
_ANCHORS_WIDE = _anchor_ranges_via_copies(_wide_range_dependent_env(),
                                          _wide_range_dependent_env().bottom)


_ANCHORS_RI = _anchor_ranges_via_copies(_range_independent_env(),
                                        _range_independent_env().bottom)


def test_the_captured_anchor_lists_are_not_empty():
    # An anti-vacuity floor on the comparison above: two empty lists compare
    # equal, and the monkeypatched test would then assert nothing.
    assert len(_ANCHORS_WIDE) > 10
    assert len(_ANCHORS_RI) > 10
