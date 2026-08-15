import warnings
"""RAM parabolic-equation-focused tests."""

import pytest
import numpy as np

from uacpy.models import RAM, RunMode
from uacpy.core.exceptions import ConfigurationError
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

    def test_unclamped_ramp_still_starts_where_it_should(self):
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
    why the case is reachable with no pinned knobs at all: uacpy's auto ``dz``
    scales with wavelength, so at 10 Hz ``dz`` is 5.7 m and any shelf below
    11.4 m trips it.
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

    @pytest.mark.parametrize('shelf', [8.0, 3.0])
    def test_a_shoaling_track_that_drives_iz_below_two_is_refused(self, shelf):
        # 8 m is the dangerous one: it used to return 30/30 finite values
        # with no warning while reading out of bounds.
        with pytest.raises(ConfigurationError, match='shoals'):
            RAM(backend='rams', verbose=False).run(
                self._env(shelf), self.SRC, self.RCV)

    @pytest.mark.parametrize('shelf', [40.0, 14.0])
    def test_a_track_that_stays_above_two_cells_still_runs(self, shelf):
        # The discriminating counterpart — iz >= 2 is in-domain and must not
        # be refused.
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
    def test_a_source_below_the_first_cell_still_runs(self, backend):
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
